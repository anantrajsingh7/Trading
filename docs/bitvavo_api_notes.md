# Bitvavo API notes

What this system uses, how it behaves, and what must be verified against the
live API before trusting any of it.

> **Verification status.** The client in `src/bitvavo_momentum/bitvavo_client.py`
> is unit-tested against a scripted transport (`tests/test_client_and_storage.py`)
> but has **not** been exercised against `api.bitvavo.com` from this environment,
> because outbound access to that host is blocked here. Treat the response-shape
> notes below as the implemented assumptions, and confirm them with
> `scripts/download_history.py --smoke-test` on a connected machine. The client
> is written to fail loudly rather than silently mis-parse.

---

## Endpoints used

| Endpoint | Purpose | Phase |
|---|---|---|
| `GET /v2/time` | connectivity probe and clock-skew check | 1 |
| `GET /v2/markets` | status, `pricePrecision`, `minOrderInQuoteAsset`, `minOrderInBaseAsset`, order types | 1, 5, 11 |
| `GET /v2/assets` | base-asset decimals | 1 |
| `GET /v2/{market}/candles` | OHLCV history — the research backbone | 1 |
| `GET /v2/ticker/price` | last price | 11 |
| `GET /v2/ticker/book` | best bid/ask → live spread | 5, 11 |
| `GET /v2/ticker/24h` | 24h quote volume for universe ranking | 1, 11 |
| `GET /v2/{market}/book` | depth snapshot for the forward record | 11 |
| `GET /v2/{market}/trades` | tick data, for resolving intrabar ambiguity on specific events | 4 |
| `GET /v2/account` | **account-specific** maker/taker fees (authenticated) | 5 |
| `wss://ws.bitvavo.com/v2/` | live candles, trades, book, ticker24h | 11 |

There is deliberately **no** order-placement, order-cancellation or withdrawal
method on the client. A test (`test_client_exposes_no_order_placement_method`)
fails if one is ever added.

---

## Candles

* Response shape: `[[timestamp_ms, open, high, low, close, volume], ...]`, with
  numeric fields as **strings**. The parser coerces explicitly.
* Bitvavo returns candles **newest-first**. Nothing downstream relies on that:
  `candles_to_frame` sorts ascending and de-duplicates on timestamp, so an
  upstream ordering change cannot corrupt the archive.
* Maximum 1440 candles per request.
* The downloader requests explicit `[start, end)` windows of exactly
  `limit × interval` duration rather than following a cursor, which makes
  pagination independent of ordering and trivially resumable.
* `volume` is in the **base** asset. Quote volume is computed locally as
  `volume × close`.

### History depth

The practical depth of 1-minute history is **not** assumed. Walking 1-minute
windows back to 2021 for a coin listed in 2024 would waste thousands of empty
calls, so the downloader first pulls **daily** candles (≈1500 of them cover four
years in one or two requests) to learn the market's true first trading day, then
walks the 1-minute grid forward from there. The discovered first timestamp is
recorded per market in the manifest.

Expect meaningfully less 1-minute history than daily history. Plan the research
around what the manifest actually reports, not around what you hoped for.

---

## Rate limiting

Bitvavo publishes the remaining budget in response headers:

* `bitvavo-ratelimit-remaining`
* `bitvavo-ratelimit-resetat` (epoch ms)
* `bitvavo-ratelimit-limit`

**Those headers are the authority.** `RateLimitState` mirrors them on every
response. The local weight table (`DEFAULT_WEIGHTS`) is only used to *predict*
the cost of the next call so the client can pause before the budget runs out —
if a weight estimate is wrong, the headers correct it on the very next response.

Behaviour:

* When the predicted post-call budget would fall below `min_remaining_budget`
  (default 60), the client sleeps until the published reset time.
* HTTP 429 → sleep until reset, then retry.
* Error codes 105 / 110 / 111 (rate limited / banned) raise
  `BitvavoRateLimitError` rather than being retried blindly.
* 5xx and network errors → exponential backoff with jitter, up to 5 attempts.
* **4xx other than 429 are never retried** — they are bugs or policy problems,
  and retrying them just burns budget.

Being banned for exceeding the limit is a real outcome, which is why the
downloader also sleeps a configurable interval between calls (`--sleep`).

---

## Authentication

HMAC-SHA256 over `timestamp_ms + METHOD + "/v2" + path_with_query + body`, hex
digest, sent as:

```
bitvavo-access-key
bitvavo-access-signature
bitvavo-access-timestamp
bitvavo-access-window
```

The signature construction is asserted in
`test_signature_is_deterministic_and_matches_the_documented_scheme`.

**Clock skew matters.** The access window (default 10s) means a local clock more
than a few seconds off will cause authenticated requests to be rejected. The
smoke test prints the skew against `/v2/time`.

### Key permissions

Only ever create a key with **View information**. Never **Trade**. **Never**
**Withdraw funds** — no part of this system needs it, and a key that cannot
withdraw cannot be used to drain the account if it leaks. Restrict the key to
your IP address in the Bitvavo UI if you can.

Nothing in Phases 1–9 needs a key at all.

---

## WebSocket

`wss://ws.bitvavo.com/v2/` with a subscribe frame:

```json
{"action": "subscribe", "channels": [{"name": "candles", "interval": ["1m"], "markets": ["BTC-EUR"]}]}
```

`BitvavoWebSocket` reconnects with exponential backoff and re-subscribes. It is
used only by the forward paper-trading path; research and backtesting never touch
it, which keeps `websocket-client` off the critical path.

---

## Market rules

From `/v2/markets`:

* `pricePrecision` is **significant digits**, not decimal places.
  `MarketRules.round_price` implements it with `Decimal` rather than `round()`;
  getting this wrong produces orders the exchange rejects.
* `minOrderInQuoteAsset` (typically 5 EUR) and `minOrderInBaseAsset` are both
  enforced. The backtest rejects fills below the minimum rather than pretending
  a 2 EUR position was possible.
* Amounts are always rounded **down** (`MarketRules.round_amount`), never up —
  rounding up can exceed available balance.
* `status` distinguishes `trading` from halted/delisted markets. The *live*
  universe uses it; the *research* universe deliberately ignores it, to avoid
  survivorship bias.

---

## Things to verify on first live contact

1. `--smoke-test` output: row count, monotonic timestamps, zero duplicates,
   timezone-aware UTC, clock skew under 30s.
2. The true earliest 1-minute candle for a few markets — this determines how much
   research is actually possible.
3. Whether `quantityDecimals` (or an equivalent) appears in `/v2/markets`; the
   code falls back to 8 decimals and reads `/v2/assets` where needed.
4. The exact `volumeQuote` field name on `/v2/ticker/24h` used for universe
   ranking.
5. Account fee values if you authenticate, so the backtest uses your real tier.
