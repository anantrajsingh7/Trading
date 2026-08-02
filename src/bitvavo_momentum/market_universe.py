"""Market universe construction and survivorship-bias control.

Two distinct notions of "universe" live here and must not be confused:

``live universe``
    What is tradeable *right now* (``/v2/markets`` with ``status == 'trading'``).
    Used by the scanner.

``research universe``
    Every market for which local history exists, **including markets that were
    later delisted, suspended or halted**. Restricting research to today's
    tradeable list would delete exactly the coins that blew up, which is textbook
    survivorship bias and would flatter any momentum result.

Eligibility inside the research universe is decided *per timestamp* using only
trailing information (rolling volume, listing age), so a coin that was illiquid
in 2022 and liquid in 2024 is correctly excluded then and included now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .logging_utils import get_logger
from .storage import ParquetStore
from .timeutils import to_utc

log = get_logger(__name__)


@dataclass
class MarketRules:
    """Trading rules for one market, taken from ``/v2/markets``."""

    market: str
    base: str
    quote: str
    status: str
    price_precision: int = 5
    min_order_in_base: float = 0.0
    min_order_in_quote: float = 5.0
    max_order_in_base: float | None = None
    max_order_in_quote: float | None = None
    quantity_decimals: int = 8
    order_types: tuple[str, ...] = ()

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> MarketRules:
        def _f(key: str, default: float | None) -> float | None:
            value = payload.get(key)
            if value in (None, ""):
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _i(key: str, default: int) -> int:
            value = payload.get(key)
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        return cls(
            market=str(payload.get("market", "")),
            base=str(payload.get("base", "")),
            quote=str(payload.get("quote", "")),
            status=str(payload.get("status", "unknown")),
            price_precision=_i("pricePrecision", 5),
            min_order_in_base=_f("minOrderInBaseAsset", 0.0) or 0.0,
            min_order_in_quote=_f("minOrderInQuoteAsset", 5.0) or 5.0,
            max_order_in_base=_f("maxOrderInBaseAsset", None),
            max_order_in_quote=_f("maxOrderInQuoteAsset", None),
            # Bitvavo exposes base-asset decimals via /v2/assets; default to 8.
            quantity_decimals=_i("quantityDecimals", 8),
            order_types=tuple(payload.get("orderTypes", []) or ()),
        )

    @property
    def is_trading(self) -> bool:
        return self.status.lower() == "trading"

    def round_price(self, price: float) -> float:
        """Bitvavo prices carry *significant digits*, not fixed decimals."""
        if price <= 0:
            return price
        from decimal import ROUND_HALF_EVEN, Decimal

        digits = max(1, int(self.price_precision))
        d = Decimal(str(price))
        exponent = d.adjusted()  # position of the most significant digit
        quant = Decimal(1).scaleb(exponent - digits + 1)
        return float(d.quantize(quant, rounding=ROUND_HALF_EVEN))

    def round_amount(self, amount: float) -> float:
        factor = 10 ** int(self.quantity_decimals)
        return int(amount * factor) / factor  # always round DOWN, never over-order

    def meets_minimum(self, amount: float, price: float) -> bool:
        quote_value = amount * price
        if self.min_order_in_quote and quote_value < self.min_order_in_quote:
            return False
        if self.min_order_in_base and amount < self.min_order_in_base:
            return False
        return True


def fetch_market_rules(client, quote: str = "EUR") -> dict[str, MarketRules]:
    """Fetch and index market rules for all markets with the given quote."""
    rules: dict[str, MarketRules] = {}
    for payload in client.get_markets():
        rule = MarketRules.from_api(payload)
        if quote and rule.quote.upper() != quote.upper():
            continue
        rules[rule.market] = rule
    log.info("Fetched rules for %d %s markets", len(rules), quote)
    return rules


def live_universe(
    rules: dict[str, MarketRules],
    exclude_bases: list[str] | None = None,
) -> list[str]:
    """Markets currently tradeable, minus stablecoin/fiat-proxy pairs."""
    excluded = {b.upper() for b in (exclude_bases or [])}
    return sorted(
        m for m, r in rules.items() if r.is_trading and r.base.upper() not in excluded
    )


def research_universe(
    store: ParquetStore,
    interval: str = "1m",
    exclude_bases: list[str] | None = None,
) -> list[str]:
    """Every market with usable local history, delisted ones included."""
    excluded = {b.upper() for b in (exclude_bases or [])}
    markets = store.usable_datasets(interval)
    kept = [m for m in markets if m.split("-")[0].upper() not in excluded]
    dropped = len(markets) - len(kept)
    if dropped:
        log.info("Excluded %d stablecoin/fiat-proxy markets from the research universe", dropped)
    return kept


def rolling_liquidity(
    candles: pd.DataFrame,
    window_days: int = 30,
    interval_minutes: int = 1,
) -> pd.Series:
    """Trailing median daily EUR quote volume, in quote currency.

    Uses ``closed='left'``-style shifting so the value at time *t* only reflects
    bars that closed strictly before *t*.
    """
    if candles.empty:
        return pd.Series(dtype="float64")
    frame = candles.set_index("timestamp").sort_index()
    quote_volume = frame["volume"] * frame["close"]
    daily = quote_volume.resample("1D").sum()
    trailing = daily.rolling(window_days, min_periods=max(3, window_days // 4)).median().shift(1)
    return trailing.reindex(frame.index, method="ffill")


def eligibility_mask(
    candles: pd.DataFrame,
    min_median_daily_quote_volume_eur: float,
    min_history_days: int,
    window_days: int = 30,
) -> pd.Series:
    """Boolean series: was this market eligible at each timestamp?

    Both conditions are strictly backward-looking:

    * trailing median daily quote volume above the floor;
    * at least ``min_history_days`` of history already observed.
    """
    if candles.empty:
        return pd.Series(dtype="bool")
    frame = candles.set_index("timestamp").sort_index()
    liquidity = rolling_liquidity(candles, window_days=window_days)
    age_days = (frame.index - frame.index[0]).total_seconds() / 86400.0
    mask = (liquidity.fillna(0.0) >= float(min_median_daily_quote_volume_eur)) & (
        pd.Series(age_days, index=frame.index) >= float(min_history_days)
    )
    mask.name = "eligible"
    return mask


def summarise_universe(store: ParquetStore, interval: str = "1m") -> pd.DataFrame:
    """One row per market: coverage window, rows and validation status."""
    manifest = store.read_manifest()
    if manifest.empty:
        return pd.DataFrame()
    subset = manifest[manifest["interval"] == interval].copy()
    for col in ("first_timestamp", "last_timestamp"):
        if col in subset.columns:
            subset[col] = pd.to_datetime(subset[col], utc=True, errors="coerce")
    subset["coverage_days"] = (
        subset["last_timestamp"] - subset["first_timestamp"]
    ).dt.total_seconds() / 86400.0
    columns = [
        "market",
        "interval",
        "first_timestamp",
        "last_timestamp",
        "coverage_days",
        "n_rows",
        "missing_fraction",
        "zero_volume_fraction",
        "validation_status",
    ]
    return subset[[c for c in columns if c in subset.columns]].sort_values("market")


def delisted_markets(
    store: ParquetStore,
    live: list[str],
    interval: str = "1m",
    stale_after_days: float = 7.0,
    as_of: pd.Timestamp | None = None,
) -> list[str]:
    """Markets present in local history but no longer trading (kept for research)."""
    summary = summarise_universe(store, interval)
    if summary.empty:
        return []
    reference = to_utc(as_of) if as_of is not None else summary["last_timestamp"].max()
    stale = summary[
        (reference - summary["last_timestamp"]).dt.total_seconds() / 86400.0 > stale_after_days
    ]
    live_set = set(live)
    return sorted(set(stale["market"]) | (set(summary["market"]) - live_set))
