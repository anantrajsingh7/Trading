# Assumptions and limitations

This document is the honest counterpart to the README. It lists what the system
assumes, what it cannot know, and where a result could be wrong even when every
line of code is correct.

---

## 0. The blocking limitation in this environment

`api.bitvavo.com` is unreachable from the machine this repository was developed
on. The egress proxy answers HTTP 403 to `CONNECT api.bitvavo.com:443`, and the
same applies to every other exchange host tested. Consequences:

* **No historical candles were downloaded. No backtest was run on market data.
  No performance number in this repository describes a real market.**
* The Phase 1 client, downloader, validator and manifest are implemented and
  unit-tested against a scripted transport, but they have not been exercised
  against the live API. The first thing to run on a connected machine is
  `scripts/download_history.py --smoke-test`, which checks connectivity, clock
  skew, timestamp ordering, duplicates and validation status on one market.
* Everything under `data/results/synthetic/` came from
  `bitvavo_momentum.synthetic` — a random walk with an *injected* pattern. It
  proves the pipeline runs. It proves nothing else. The synthetic generator's
  `continuation_probability` is a knob; setting it to 0.5 and still seeing a
  positive backtest would indicate a bug in the backtester, which is exactly why
  the generator exists.

Substituting another exchange's data for Bitvavo's would have been the easy
route and the wrong one: Bitvavo's fee schedule, tick sizes, minimum order sizes,
liquidity and listing set are what the hypothesis is about.

---

## 1. Data

**Interval.** All research runs on 1-minute candles. Coarser intervals are
aggregated locally from the same 1-minute bars, so a 15-minute bar is by
construction the aggregate of the 1-minute bars used everywhere else. Downloading
each interval separately would allow the timeframes to disagree.

**Missing candles.** On Bitvavo a missing 1-minute candle means *no trades
occurred*, not *price was unchanged*. Forward-filling those bars would invent
liquidity that did not exist and would make thin coins look tradeable. Missing
bars are kept as explicit gaps, counted in the dataset manifest, and any event
whose look-back window is more than 20% gaps is discarded.

**Depth of history.** Bitvavo's public candle endpoint does not serve unlimited
1-minute history, and the practical depth is not documented as a fixed number.
The downloader therefore *probes* the true first available timestamp per market
(using daily candles, which is cheap) and records it, rather than assuming a
start date. Expect materially less 1-minute history than daily history. If the
usable 1-minute window turns out to be short, the honest response is fewer
independent events and wider confidence intervals — not a longer backtest built
from a different interval.

**Survivorship.** Delisted, suspended and halted markets are retained in the
research universe wherever local history exists. This only works for markets that
were downloaded *before* they were delisted; a market removed from
`/v2/markets` before you first ran the downloader cannot be recovered from the
API. **The archive is therefore biased toward survivors in proportion to how late
you started collecting.** Running the downloader regularly from today onward is
the only fix, and it is a slow one.

**Microstructure.** Historical candles contain no quotes. The bid-ask spread and
order-book imbalance **cannot be reconstructed** from them. What the code
provides are clearly named proxies — `spread_proxy_bps` (a high/low range
estimator) and `illiquidity_amihud` — and nothing in the reporting layer presents
them as measurements. Live order-book data is captured only going forward, by the
scanner.

**Fees.** Account-specific maker/taker fees are used when an API key is
configured; otherwise the configured fallback (25/15 bps) applies. Bitvavo's fee
tiers depend on 30-day volume, so a real account's fees drift over time. The
backtest uses a *constant* fee. For a small account this is conservative in the
right direction (tier 0 is the most expensive tier).

---

## 2. Event definition

* An event is stamped at the bar whose close first crosses the threshold. The
  crossing must be genuine: the previous bar must have been below it.
* De-duplication defaults to `first_touch` with a 240-minute cooldown. Without
  it a single 30% rally fires hundreds of overlapping 1-minute events and every
  downstream statistic silently assumes a sample size it does not have.
* `peak_of_cluster` mode exists for descriptive work and is flagged non-causal —
  identifying the cluster peak requires seeing the rest of the cluster.
* Eligibility (30-day median quote volume, listing age) is evaluated at each
  timestamp from trailing data only.

---

## 3. Execution

Modelled: fees on both sides, half-spread on both sides, slippage rising with
realised volatility and illiquidity, latency (entry never on the signal bar),
partial fills, rejected orders, capacity limits as a share of bar volume,
minimum order sizes, price/amount precision, gap-through-stop fills at the bar
open plus extra slippage, and limit orders that are touched but not filled.

Not modelled, and worth knowing:

* **Real order-book depth.** Slippage is a parametric function, not a walk down
  an actual book. For orders that are small relative to the market this is
  adequate; for large orders it is optimistic even in the stress scenario.
* **Exchange downtime and maintenance windows.** A halt during an open position
  is not simulated.
* **Queue position.** Limit-fill probability is a scalar, not a queue model.
* **Adverse selection on limit fills.** In reality, the limit orders that fill
  are disproportionately the ones you would rather not have filled. The scenario
  fill probabilities are a crude stand-in.

**Intrabar ambiguity.** When one bar's range contains both the stop and the
target, minute candles cannot order the two touches. The stop is taken and the
trade is flagged `ambiguous_bar`. The reported share of ambiguous exits is the
measure of how much a result depends on this assumption; tick-level data would be
needed to resolve it properly, and Bitvavo's public trade endpoint could supply
it for specific events if a result ever hinges on this.

---

## 4. Statistics

* **Trades are not independent.** Momentum events cluster across coins during
  market-wide moves. Any statistic assuming iid observations overstates
  significance. `robustness.independent_trade_count` reports the clustered count,
  and confidence intervals use a moving-block bootstrap.
* **Multiple testing.** The grid in `run_backtest.py` evaluates dozens to
  hundreds of configurations. The count is recorded and used to deflate the
  Sharpe ratio. A "best of 200" result with a nominal p-value of 0.02 is not a
  finding.
* **Suppressed statistics.** Sharpe, Sortino and annualised return return `NaN`
  below 20 trades rather than a number that invites over-reading.
* **Annualisation.** An event strategy is flat most of the time. Ratios are
  computed on a daily equity curve *including* flat days, so a strategy is not
  credited with the Sharpe of a continuously invested one.
* **The t-statistics in the event study are naive** and labelled as such; the
  bootstrap is the authority.

---

## 5. Scope

* Spot only. No shorting, no leverage, no margin, no derivatives. Strategy G
  tests the exhaustion hypothesis *long-only*, by waiting for a deeper
  correction — it never places an unsupported short.
* EUR-quoted markets only.
* No machine learning. Rule-based baselines first, by design; a model would add
  degrees of freedom before the underlying effect has been established.
* Live order execution is disabled and no order-placement code exists.

---

## 6. What a positive result would and would not mean

If the pipeline eventually reports a positive net expectancy that survives the
untouched test set, that would mean: *on the specific markets, period and cost
assumptions tested, this rule would have made money.* It would not mean the
effect persists, that it survives at larger size, that the fee schedule stays put,
or that the coins that produced it will keep listing. Paper trading is the next
step after that, not live capital — and paper trading has to run long enough to
produce its own independent sample.

The system is deliberately built so that "insufficient evidence" costs nothing to
report.
