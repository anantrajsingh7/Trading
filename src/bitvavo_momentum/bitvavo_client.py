"""Transparent Bitvavo REST/WebSocket client.

A small hand-written client is used rather than a third-party SDK so that every
network call, retry, rate-limit decision and raw payload is auditable - which is
the whole point of a research system whose conclusions depend on the data.

Key behaviours
--------------
* **Rate limiting** - Bitvavo publishes the remaining budget in the
  ``bitvavo-ratelimit-remaining`` / ``bitvavo-ratelimit-resetat`` response
  headers. Those headers are the authority; the local weight table is only used
  to *predict* the cost of the next call so we can pause before being banned.
* **Retries** - exponential backoff with jitter on network errors, HTTP 5xx and
  HTTP 429. Client errors (4xx other than 429) are not retried; they are bugs or
  policy problems, and retrying them just burns the budget.
* **Raw capture** - every successful response can be handed to a sink before any
  parsing, so the unmodified payload is preserved (Phase 1 requirement).
* **Credentials** - read from the environment via :class:`Credentials`. Never
  logged, never written to disk, never accepted as a function argument literal.

The signature scheme follows Bitvavo's documented HMAC-SHA256 construction:
``HMAC(secret, timestamp_ms + method + '/v2' + path_with_query + body)``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import requests

from .config import Credentials
from .logging_utils import get_logger

log = get_logger(__name__)

REST_BASE = "https://api.bitvavo.com/v2"
WS_URL = "wss://ws.bitvavo.com/v2/"

#: Best-effort weight estimates used only for *predictive* throttling. The
#: authoritative remaining budget always comes from the response headers.
DEFAULT_WEIGHTS: dict[str, int] = {
    "time": 1,
    "markets": 1,
    "assets": 1,
    "candles": 1,
    "book": 1,
    "trades": 5,
    "ticker/price": 1,
    "ticker/book": 1,
    "ticker/24h": 25,
    "account": 1,
    "balance": 5,
}

MAX_CANDLES_PER_REQUEST = 1440


class BitvavoError(RuntimeError):
    """Base class for Bitvavo client failures."""


class BitvavoAPIError(BitvavoError):
    """Non-retryable API-level error (4xx, or an error object in the body)."""

    def __init__(self, status: int, code: int | None, message: str, path: str):
        super().__init__(f"HTTP {status} on {path}: [{code}] {message}")
        self.status = status
        self.code = code
        self.api_message = message
        self.path = path


class BitvavoRateLimitError(BitvavoError):
    """The rate limit was hit or the key/IP was banned."""

    def __init__(self, message: str, reset_at_ms: int | None = None):
        super().__init__(message)
        self.reset_at_ms = reset_at_ms


class BitvavoNetworkError(BitvavoError):
    """Transport-level failure that survived all retries."""


@dataclass
class RateLimitState:
    """Mirror of the server-side budget, updated from response headers."""

    limit: int = 1000
    remaining: int = 1000
    reset_at_ms: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update_from_headers(self, headers: Mapping[str, str]) -> None:
        with self.lock:
            remaining = headers.get("bitvavo-ratelimit-remaining")
            reset_at = headers.get("bitvavo-ratelimit-resetat")
            limit = headers.get("bitvavo-ratelimit-limit")
            if remaining is not None:
                try:
                    self.remaining = int(remaining)
                except ValueError:
                    pass
            if reset_at is not None:
                try:
                    self.reset_at_ms = int(reset_at)
                except ValueError:
                    pass
            if limit is not None:
                try:
                    self.limit = int(limit)
                except ValueError:
                    pass

    def spend(self, weight: int) -> None:
        with self.lock:
            self.remaining = max(0, self.remaining - weight)

    def seconds_until_reset(self) -> float:
        if not self.reset_at_ms:
            return 0.0
        return max(0.0, (self.reset_at_ms / 1000.0) - time.time())


class BitvavoClient:
    """Minimal, explicit Bitvavo v2 REST client."""

    def __init__(
        self,
        credentials: Credentials | None = None,
        base_url: str = REST_BASE,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 1.5,
        backoff_cap: float = 60.0,
        min_remaining_budget: int = 60,
        raw_sink: Callable[[str, dict[str, Any], Any], None] | None = None,
        session: requests.Session | None = None,
        user_agent: str = "bitvavo-momentum-agent/0.1 (research; contact via repo)",
    ) -> None:
        self.credentials = credentials or Credentials.from_env()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.min_remaining_budget = min_remaining_budget
        self.raw_sink = raw_sink
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self.rate_limit = RateLimitState()
        self.request_count = 0
        self.retry_count = 0

    # -- low level -------------------------------------------------------------
    def _sign(self, timestamp_ms: int, method: str, path_with_query: str, body: str) -> str:
        message = f"{timestamp_ms}{method.upper()}/v2{path_with_query}{body}"
        assert self.credentials.api_secret is not None
        return hmac.new(
            self.credentials.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, method: str, path_with_query: str, body: str) -> dict[str, str]:
        ts = int(time.time() * 1000)
        return {
            "bitvavo-access-key": self.credentials.api_key or "",
            "bitvavo-access-signature": self._sign(ts, method, path_with_query, body),
            "bitvavo-access-timestamp": str(ts),
            "bitvavo-access-window": str(self.credentials.access_window_ms),
        }

    def _throttle(self, weight: int) -> None:
        """Pause proactively when the published budget is nearly exhausted."""
        if self.rate_limit.remaining - weight <= self.min_remaining_budget:
            wait = self.rate_limit.seconds_until_reset()
            if wait > 0:
                log.warning(
                    "Rate-limit budget low (remaining=%s); sleeping %.1fs until reset",
                    self.rate_limit.remaining,
                    wait,
                )
                time.sleep(wait + 0.5)
                self.rate_limit.remaining = self.rate_limit.limit

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.backoff_cap, self.backoff_base**attempt)
        delay *= 0.5 + random.random()  # jitter, avoids synchronised retries
        log.info("Retrying in %.2fs (attempt %d/%d)", delay, attempt, self.max_retries)
        time.sleep(delay)

    def request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        body: Any = None,
        authenticate: bool = False,
        weight: int | None = None,
    ) -> Any:
        """Perform one REST call with retries, throttling and raw capture."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        query = f"?{urlencode(params)}" if params else ""
        path_with_query = f"{path}{query}"
        url = f"{self.base_url}{path_with_query}"
        body_str = json.dumps(body, separators=(",", ":")) if body is not None else ""
        endpoint_key = path.strip("/").split("/")[-1]
        weight = weight if weight is not None else DEFAULT_WEIGHTS.get(endpoint_key, 1)

        if authenticate and not self.credentials.is_authenticated:
            raise BitvavoError(
                f"{path} requires credentials; set BITVAVO_API_KEY/BITVAVO_API_SECRET "
                "(read-only key) in the environment."
            )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle(weight)
            headers: dict[str, str] = {}
            if authenticate:
                headers.update(self._auth_headers(method, path_with_query, body_str))
            if body_str:
                headers["Content-Type"] = "application/json"

            try:
                self.request_count += 1
                response = self.session.request(
                    method.upper(),
                    url,
                    headers=headers,
                    data=body_str or None,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                self.retry_count += 1
                log.warning("Network error on %s: %s", path, exc)
                if attempt == self.max_retries:
                    break
                self._sleep_backoff(attempt)
                continue

            self.rate_limit.update_from_headers(response.headers)
            self.rate_limit.spend(weight)

            if response.status_code == 429:
                self.retry_count += 1
                reset_at = self.rate_limit.reset_at_ms or None
                wait = self.rate_limit.seconds_until_reset() or min(self.backoff_cap, 2**attempt)
                log.warning("HTTP 429 on %s; sleeping %.1fs", path, wait)
                if attempt == self.max_retries:
                    raise BitvavoRateLimitError(f"Rate limited on {path}", reset_at)
                time.sleep(wait + 0.5)
                continue

            if 500 <= response.status_code < 600:
                self.retry_count += 1
                last_error = BitvavoAPIError(response.status_code, None, response.text[:200], path)
                log.warning("HTTP %s on %s", response.status_code, path)
                if attempt == self.max_retries:
                    break
                self._sleep_backoff(attempt)
                continue

            try:
                payload = response.json()
            except ValueError as exc:
                raise BitvavoError(f"Non-JSON response from {path}: {response.text[:200]!r}") from exc

            if isinstance(payload, dict) and "errorCode" in payload:
                code = payload.get("errorCode")
                message = str(payload.get("error", ""))
                # 105 = rate limited, 110/111 = banned - treat as rate-limit class.
                if code in (105, 110, 111):
                    raise BitvavoRateLimitError(f"{code}: {message}")
                raise BitvavoAPIError(response.status_code, code, message, path)

            if response.status_code >= 400:
                raise BitvavoAPIError(response.status_code, None, response.text[:200], path)

            if self.raw_sink is not None:
                try:
                    self.raw_sink(path, dict(params), payload)
                except Exception:  # capture must never break the download
                    log.exception("raw_sink failed for %s (continuing)", path)

            return payload

        raise BitvavoNetworkError(f"{path} failed after {self.max_retries} attempts: {last_error}")

    # -- public endpoints ------------------------------------------------------
    def get_time(self) -> int:
        """Server time in ms. Cheap connectivity/clock-skew probe."""
        return int(self.request("GET", "/time")["time"])

    def get_markets(self, market: str | None = None) -> list[dict[str, Any]]:
        """Market rules: status, precision, min order sizes, supported order types."""
        payload = self.request("GET", "/markets", params={"market": market})
        return payload if isinstance(payload, list) else [payload]

    def get_assets(self, symbol: str | None = None) -> list[dict[str, Any]]:
        payload = self.request("GET", "/assets", params={"symbol": symbol})
        return payload if isinstance(payload, list) else [payload]

    def get_candles(
        self,
        market: str,
        interval: str = "1m",
        limit: int = MAX_CANDLES_PER_REQUEST,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[list[Any]]:
        """Raw OHLCV rows ``[timestamp_ms, open, high, low, close, volume]``.

        Bitvavo returns newest-first; ordering is **not** assumed anywhere
        downstream - :mod:`data_downloader` sorts explicitly.
        """
        return self.request(
            "GET",
            f"/{market}/candles",
            params={
                "interval": interval,
                "limit": min(int(limit), MAX_CANDLES_PER_REQUEST),
                "start": start_ms,
                "end": end_ms,
            },
        )

    def get_ticker_price(self, market: str | None = None) -> Any:
        return self.request("GET", "/ticker/price", params={"market": market})

    def get_ticker_book(self, market: str | None = None) -> Any:
        """Best bid/ask and sizes - the spread input for the execution model."""
        return self.request("GET", "/ticker/book", params={"market": market})

    def get_ticker_24h(self, market: str | None = None) -> Any:
        return self.request(
            "GET", "/ticker/24h", params={"market": market}, weight=1 if market else 25
        )

    def get_book(self, market: str, depth: int = 25) -> dict[str, Any]:
        return self.request("GET", f"/{market}/book", params={"depth": depth})

    def get_trades(
        self,
        market: str,
        limit: int = 500,
        start_ms: int | None = None,
        end_ms: int | None = None,
        trade_id_from: str | None = None,
        trade_id_to: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.request(
            "GET",
            f"/{market}/trades",
            params={
                "limit": limit,
                "start": start_ms,
                "end": end_ms,
                "tradeIdFrom": trade_id_from,
                "tradeIdTo": trade_id_to,
            },
        )

    # -- authenticated (read-only) --------------------------------------------
    def get_account(self) -> dict[str, Any]:
        """Account info including the *account-specific* maker/taker fee tier."""
        return self.request("GET", "/account", authenticate=True)

    def get_fees(self) -> dict[str, float] | None:
        """Return ``{'maker': x, 'taker': y}`` as fractions, or ``None`` if public-only."""
        if not self.credentials.is_authenticated:
            return None
        try:
            account = self.get_account()
        except BitvavoError as exc:
            log.warning("Could not fetch account fees (%s); using configured fallbacks", exc)
            return None
        fees = account.get("fees") or {}
        try:
            return {"maker": float(fees["maker"]), "taker": float(fees["taker"])}
        except (KeyError, TypeError, ValueError):
            log.warning("Unexpected /account fee payload shape; using configured fallbacks")
            return None

    # NOTE: there is deliberately no order-placement method on this client.
    # Order submission lives behind the triple safety gate in `paper_trader.py`.

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> BitvavoClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class BitvavoWebSocket:
    """Thin WebSocket wrapper for live candles, trades and order-book updates.

    Used only by the forward paper-trading path (Phase 11). Research and
    backtesting never touch it, so it stays deliberately small: connect,
    subscribe, dispatch, reconnect with backoff.
    """

    def __init__(
        self,
        url: str = WS_URL,
        on_message: Callable[[dict[str, Any]], None] | None = None,
        reconnect_base: float = 2.0,
        reconnect_cap: float = 60.0,
    ) -> None:
        self.url = url
        self.on_message = on_message
        self.reconnect_base = reconnect_base
        self.reconnect_cap = reconnect_cap
        self._subscriptions: list[dict[str, Any]] = []
        self._ws: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def subscribe_candles(self, markets: Iterable[str], interval: str = "1m") -> None:
        self._subscriptions.append(
            {"name": "candles", "interval": [interval], "markets": list(markets)}
        )

    def subscribe_trades(self, markets: Iterable[str]) -> None:
        self._subscriptions.append({"name": "trades", "markets": list(markets)})

    def subscribe_book(self, markets: Iterable[str]) -> None:
        self._subscriptions.append({"name": "book", "markets": list(markets)})

    def subscribe_ticker24h(self, markets: Iterable[str]) -> None:
        self._subscriptions.append({"name": "ticker24h", "markets": list(markets)})

    def _send_subscriptions(self) -> None:
        if not self._subscriptions or self._ws is None:
            return
        self._ws.send(json.dumps({"action": "subscribe", "channels": self._subscriptions}))
        log.info("Subscribed to %d WebSocket channel groups", len(self._subscriptions))

    def _run(self) -> None:
        import websocket  # imported lazily: research paths must not need it

        attempt = 0
        while not self._stop.is_set():
            try:
                self._ws = websocket.create_connection(self.url, timeout=30)
                attempt = 0
                self._send_subscriptions()
                while not self._stop.is_set():
                    raw = self._ws.recv()
                    if not raw:
                        continue
                    try:
                        message = json.loads(raw)
                    except ValueError:
                        log.warning("Dropping non-JSON WebSocket frame")
                        continue
                    if self.on_message is not None:
                        try:
                            self.on_message(message)
                        except Exception:
                            log.exception("on_message handler raised (stream continues)")
            except Exception as exc:
                if self._stop.is_set():
                    break
                attempt += 1
                delay = min(self.reconnect_cap, self.reconnect_base**attempt) * (0.5 + random.random())
                log.warning("WebSocket error (%s); reconnecting in %.1fs", exc, delay)
                time.sleep(delay)
            finally:
                try:
                    if self._ws is not None:
                        self._ws.close()
                except Exception:
                    pass
                self._ws = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="bitvavo-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=5)
