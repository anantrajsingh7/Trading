# bitvavo-momentum-agent

Research, backtesting and paper-trading system for short-horizon momentum events
on Bitvavo EUR markets.

**The question this system exists to answer:** when a coin rises roughly 10–12%
within two to three hours, does it continue often enough to produce a *positive
net expectancy after realistic fees, spread, slippage and failed fills*?

The system is built to be able to answer **no**. "Insufficient evidence" is a
supported, first-class outcome, and the reporting code will state it plainly
rather than dressing up a marginal result.

> No promise of profit is made anywhere in this repository. Nothing here is
> financial advice. Live order execution is disabled and no order-placement code
> exists in this codebase.

---

## Result: the hypothesis is rejected

**Run date:** 2026-08-03 · **Data:** 40 Bitvavo EUR markets, 1-minute candles,
2025-01-01 → 2026-08-02 (12.7 M bars) · **Events:** 495 at the primary
specification (+10% / 120 min), 21,751 across the full 36-spec grid.

### Phase 2 — descriptive event study

Mean forward return after a qualifying event, measured from the open of the next
bar, **gross of all costs**:

| Horizon | Mean | Median | Hit rate |
|---|---|---|---|
| 15 min | −0.13% | −0.15% | 45.7% |
| 1 h | −0.11% | −0.26% | 46.3% |
| 2 h | −0.20% | −0.32% | 46.5% |
| 8 h | −0.19% | −0.31% | 48.1% |
| 24 h | −0.55% | −1.25% | 45.3% |
| 48 h | −0.59% | −2.27% | 42.8% |

Negative at every horizon, hit rate below 50% at every horizon, and the median
below the mean throughout — most events bleed while a few rip, which is the
distribution that makes this pattern feel real to a human observer.

On the primary specification (n = 495) a moving-block bootstrap puts only the
30-minute horizon outside zero, and on the **negative** side (95% CI −0.77% to
−0.06%). The honest reading is *no positive continuation*, with weak evidence of
slight negative drift shortly after the event.

### Phases 3–8 — strategies and costs

All **51 configurations** (17 entry strategies × 3 exit policies) returned
negative net expectancy on training data. Every profit factor was below 1.0.

| | |
|---|---|
| Best configuration (consolidation breakout) | **−0.45%** per trade |
| Selected candidate on validation | **−1.79%** per trade |
| Same candidate under stress execution | **−2.45%** per trade |

Mean reversion after the spike (Strategy G) also failed, at −0.67% per trade, so
the exhaustion hypothesis is closed at this timescale too.

**Verdict: rejected on five independent criteria** — insufficient independent
trades, negative net expectancy after costs, 62% of gross profit from one coin,
78% from one month, and negative expectancy under stress. The untouched test set
was never opened, because nothing survived validation to justify opening it.

Full record: `data/results/research_report.md`, `report.html`,
`rejected_strategies.md`, `strategy_comparison.csv`.

### Phase 9 — non-impulse entry families

The impulse hypothesis being closed does not close *trading*, so three entry
families that do not depend on a spike were tested next: trend continuation
(Strategy 1), volatility-compression breakout (Strategy 2) and Donchian channel
breakout (Strategy 8) — 16 variants over 20 markets, 124,752 signals, 60,480 in
the training split.

**0 of 16 had positive net expectancy.** Best was `S8_donchian_48` at −1.30% per
trade (profit factor 0.374); worst was `S8_donchian_24_notrend` at −2.32% with a
−68.9% drawdown.

Two things stop that from being the end of the story, and both are recorded here
rather than buried:

* Win rates were **5.1%–8.5%**. All 16 variants ran through one exit policy
  (+10% target, 1.5×ATR stop, 48h clock). A win rate that low is evidence the
  *target* was almost never reached — which is a fact about the exit, not
  necessarily about the entry.
* The **matched-random baseline over the same signals was also negative**
  (−0.03% at 4h, −0.47% at 48h). The 2025 training window fell, and every one of
  these strategies is long-only. Losing money in that period is not by itself
  evidence of a bad signal.

`scripts/run_exit_research.py` separates the two explanations before anything is
declared dead: stage 1 measures what each signal was worth with *no exit rule at
all* across 6/12/24/36/48h, plus how far price actually ran in favour. A
buy-and-hold-to-horizon return is the ceiling on what any non-clairvoyant exit
can average, so a family below the cost floor at every horizon cannot be rescued
by exit tuning — and one above it has an exit problem worth solving.

Full record: `strategy_families_train.csv`, `exit_reason_breakdown.csv`,
`exit_stage1_verdict.csv`.

### Phase 10 — the cost structure, which explains all of the above

Stage 1 of the exit research measured what each entry family's signal was worth
with **no exit rule at all**: 18–34 bps gross at the best holding period, against
a 77 bps round trip. Thirteen of sixteen families were *positive* gross while the
matched-random baseline was negative, so the signals were not noise — they were
worth less than half the toll.

Rotation (Strategy 7) was then run to attack the toll rather than the signal:
0 of 24 combinations positive. The turnover table showed why, and it contradicted
the premise the test was built on — the default grid rebalances every 24 hours,
which produced 128–341 position turns per slot per year and **99%–262% annual
cost drag**. The low-turnover thesis was never actually tested by that grid.

`scripts/cost_structure.py` writes down the arithmetic that was implicit all
along. Annual cost is fixed by holding period alone, before any question of skill:

| Holding period | Round trips/yr | Annual cost drag @ 77 bps |
|---|---|---|
| 6 hours | 1460 | 1124% |
| 24 hours | 365 | 281% |
| **48 hours (spec maximum)** | **182** | **140%** |
| 1 week | 52 | 40% |
| 1 month | 12 | 9.4% |
| 3 months | 4 | 3.1% |

**The spec's 48-hour maximum holding period and profitability after 77 bps costs
are mutually exclusive.** No entry signal, exit rule or filter changes that; it
is arithmetic. `run_rotation.py --allow-long-holds` adds weekly/fortnightly/
monthly variants that breach the 48-hour cap deliberately, so the trade-off can
be measured rather than argued about.

### Phase 11 — the regime gate

Every rejected family is long-only, and the training window fell. Random entries
in the same markets also lost money, which makes *"the entries are bad"* and
*"being long was bad"* observationally similar — and only one of those is fixable
by a filter. `scripts/run_regime_gate.py` separates them, in two steps:

1. **Does the label carry information?** The same signals, grouped by the regime
   they fired in. If uptrend signals returned what downtrend signals returned,
   no gate built on these labels can help and step 2 is measuring noise.
2. **Does a gate do work?** Five presets — fixed in `regime_gate.PRESETS`,
   written before any result was seen — scored by **what they reject** as well as
   what they keep.

A regime filter is the easiest thing here to fool yourself with: ~580 days, a
handful of plausible definitions, and total freedom in choosing which labels
count as "allowed". Three structural defences:

- allowed-label sets are **declared up front**, never chosen from performance;
- a gate is judged on `separation` (kept minus rejected) — a gate whose rejects
  were fine is discarding trades at random, and a separation below a quarter of
  the round-trip cost is reported as *"a smaller sample, not a filter"*;
- `share_kept` and `n_kept` sit beside every return figure, so a flattering mean
  on 4% of the signals cannot pass unnoticed.

Labels are shifted one day and volatility quantiles are expanding, so a label
never sees the day it labels.

### Why it fails

The realistic round-trip cost is **77 bps**. Gross forward returns are around
−20 bps at the 2-hour horizon. No amount of entry timing, exit tuning or
filtering closes a gap that starts on the wrong side of zero. High turnover is
the structural problem: ~200 round trips a year burns roughly 15% of capital in
costs alone.

### Reproducing this

```bash
python scripts/download_history.py --smoke-test
python scripts/download_history.py --markets top:40 --interval 1m --history-start 2025-01-01T00:00:00Z
python scripts/run_research.py
python scripts/run_backtest.py --quick
```

Everything under `data/results/synthetic/` was generated from artificial data by
`bitvavo_momentum.synthetic` and carries a banner saying so. **Those numbers
describe a random process with an injected pattern. They say nothing about
Bitvavo markets and must never be quoted as a finding.**

---

## Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dashboard,dev]"
cp .env.example .env        # optional; research needs no credentials at all
pytest -q
```

Python 3.12+. Core dependencies: pandas, NumPy, SciPy, statsmodels, PyArrow,
DuckDB, requests, PyYAML. Dashboard extra: Streamlit + Plotly.

---

## Credentials and safety

* **Research and backtesting need no API key.** Everything runs on public
  endpoints.
* If you do configure a key, it must be **read-only**: enable *View information*
  only. Never enable *Trade*, and **never** enable *Withdraw funds*.
* Credentials are read from environment variables only. They are never written to
  source, config, logs or Git — `logging_utils.RedactSecretsFilter` strips them
  from every log record, and a test asserts it.
* **Live trading requires three independent switches** (`live_trading.enabled` in
  config, `ALLOW_LIVE_TRADING=true` in the environment, and an explicit CLI
  acknowledgement) — *and even then there is no order-placement code*. Enabling
  execution requires writing and reviewing an adapter deliberately. That is the
  intended friction.

---

## Project layout

```
bitvavo-momentum-agent/
├── config/
│   ├── research.yaml          # data, events, splits, regimes, robustness
│   ├── risk.yaml              # execution scenarios, fees, sizing, risk limits
│   └── paper_trading.yaml     # scanner, paper trading, alerts
├── data/
│   ├── raw/                   # unmodified API payloads (gzipped JSON)
│   ├── processed/             # tidy Parquet + dataset manifest
│   ├── results/               # research artefacts
│   └── synthetic/             # synthetic-run artefacts, never mixed with real
├── src/bitvavo_momentum/      # the package
├── dashboard/app.py           # Streamlit dashboard
├── scripts/                   # entry points
└── tests/                     # look-ahead and safety tests first
```

### Deviations from the requested structure, and why

1. **`src/bitvavo_momentum/` instead of flat modules in `src/`.** A real package
   is importable (`pip install -e .`), testable without `sys.path` hacks, and
   keeps names like `features` and `storage` out of the global namespace. Every
   requested module name exists inside it.
2. **Modules added.** `timeutils.py` (one place where UTC↔Amsterdam conversion
   happens), `storage.py` (raw capture, Parquet, dataset manifest), `metrics.py`
   and `robustness.py` (Phases 7 and 8 were too large to live inside
   `reporting.py`), `regimes.py` (Phase 9), `pipeline.py` (shared orchestration
   so scripts and dashboard cannot compute different numbers from the same
   inputs), `logging_utils.py`, `config.py`, `synthetic.py`.
3. **`data/synthetic/` and `data/results/synthetic/`.** Synthetic output is
   physically separated from real output so the two can never be confused.

---

## How look-ahead bias is prevented

This is the part that matters most; everything else is bookkeeping.

| Mechanism | Where |
|---|---|
| A candle stamped `T` is only *known* at `T + interval`; features stamped `T` mean "available at `T+interval`" | `features.py` docstring |
| No `shift(-n)`, no `center=True`, no full-sample statistics anywhere in feature code | `features.py` |
| Volume baselines are shifted by one bar so the judged bar is excluded from its own baseline | `features.build_features` |
| `EntryPlan` **raises** if execution is not strictly after the decision bar | `strategies.EntryPlan.__post_init__` |
| Entry price is the *next bar's open*, never the signal bar's close | `strategies`, `backtester` |
| Forward-looking columns are prefixed `fwd_` and `assert_no_forward_columns` enforces their absence | `event_detector.py` |
| Regime labels are shifted one day: a label never uses the close of the day it labels | `regimes.classify_regimes` |
| Volatility regimes use *expanding* quantiles, not full-sample quantiles | `regimes._expanding_quantile_rank` |
| Test asserts features are bit-identical when future data is truncated away | `tests/test_no_lookahead.py` |
| Test asserts the event set is unchanged when future bars are removed | `tests/test_no_lookahead.py` |

Other biases:

* **Survivorship** — delisted and suspended markets stay in the research
  universe wherever local history exists (`market_universe.research_universe`).
* **Selection** — eligibility (liquidity, listing age) is evaluated *per
  timestamp* from trailing data, so a coin illiquid in 2022 is excluded then and
  included later.
* **Multiple testing** — the number of configurations evaluated is recorded and
  fed into the deflated Sharpe ratio.
* **Clustered events** — momentum events arrive together across coins. The
  honest sample size is the cluster count (`robustness.independent_trade_count`),
  and confidence intervals use a moving-block bootstrap.

---

## Execution realism

Three fixed scenarios in `config/risk.yaml`, defined **before** any strategy was
evaluated. Headline numbers use `realistic`; stress results are always shown
alongside.

| | optimistic | realistic | stress |
|---|---|---|---|
| half-spread | 2.5 bps | 7.5 bps | 20 bps |
| base slippage | 2 bps | 6 bps | 15 bps |
| market fill probability | 100% | 98% | 92% |
| limit fill probability (when touched) | 90% | 60% | 35% |
| partial-fill probability | 0% | 15% | 35% |
| max share of bar volume | 10% | 5% | 2% |
| minimum round-trip cost | 59 bps | **77 bps** | 120 bps |

That last row is the bar a strategy must clear before it earns anything. Also
modelled: gap-through-stop fills at the bar open plus extra slippage, capacity
truncation, minimum order sizes, price/amount precision, and rejected orders.

**Intrabar ambiguity:** when one bar's range contains both the stop and the
target, minute data cannot say which came first. The **stop** is taken and the
trade is flagged; the share of ambiguous exits is reported, and a strategy whose
result depends on many of them is rejected.

---

## Research protocol

```
events → train (earliest 50%) → validation (next 25%) → test (most recent 25%)
                                                          ↑ locked by default
```

* Parameters are optimised on **train** only.
* Strategy selection uses **validation** only.
* The **test set is locked** behind `splits.unlock_test_set: false`. Opening it
  is a config change visible in Git history, and it is meant to happen once.
* Rolling walk-forward (train 12m / test 3m / step 3m) re-selects parameters
  inside every window.
* An embargo (default 48h) separates splits so no trade straddles a boundary.

### Rejection criteria (Phase 7)

A strategy is rejected — regardless of how good the equity curve looks — if it
has too few independent trades, non-positive net expectancy after costs, profit
concentrated in one coin or one month, unacceptable drawdown, negative stress
expectancy, or too many intrabar-ambiguous exits. These are coded in
`metrics.rejection_reasons`, not applied by judgement after the fact.

---

## Strategies compared

| | Strategy |
|---|---|
| A | Immediate entry on the bar after the event |
| B | Pullback continuation (1–5%, 0.25/0.5/1 ATR) with confirmation |
| C | Consolidation breakout (15–120 min range) with volume confirmation |
| D | VWAP / EMA9 / EMA20 retest with reclaim |
| E | Volume-confirmed continuation (z-score vs the coin's own history) |
| F | Cross-sectional: rank concurrent events, take the top 1–3 |
| G | Exhaustion / mean reversion — the opposite hypothesis, long-only (no shorting is assumed or simulated) |
| H | Hybrid, with a hard cap on the number of filters |
| — | Benchmarks: random entry (same market, same window) and do-nothing |

Exits (`ExitPolicy`) are independent of entries: fixed TP/SL, ATR stop, trailing
stop, chandelier, break-even arming, partial take-profit, time stop, VWAP/EMA
loss and maximum-adverse-excursion exit.

---

## Entry points

| Command | Phase |
|---|---|
| `scripts/download_history.py --smoke-test` | 1 — connectivity, ordering, timestamps |
| `scripts/download_history.py --markets top:40` | 1 — resumable download + manifest |
| `scripts/run_research.py` | 2 — event dataset and descriptive event study |
| `scripts/run_backtest.py` | 3–8 — strategy grid, splits, robustness, verdict |
| `scripts/run_walk_forward.py` | 6/8 — rolling out-of-sample validation |
| `scripts/run_scanner.py` | 11 — live scanner, paper mode only |
| `scripts/run_dashboard.py` | 12 — Streamlit dashboard |

The research scripts accept `--source synthetic` to exercise the pipeline
without market data.

---

## Dashboard

`python scripts/run_dashboard.py` → Overview, Live scanner, Backtest results,
Event explorer, Trade journal, Data & status.

All internal state is UTC; every displayed timestamp is converted to
Europe/Amsterdam and labelled. A synthetic-mode toggle makes artificial results
unmistakable. The dashboard only *reads* artefacts — it never recomputes a
strategy, so what you see is what the report says.

---

## Outputs

Written to `data/results/` (or `data/results/synthetic/`):

`research_report.md`, `backtest_summary.csv`, `strategy_comparison.csv`,
`trade_log.parquet`, `event_dataset.parquet`, `walk_forward_results.csv`,
`parameter_stability.csv`, `rejected_strategies.md`,
`assumptions_and_limitations.md`, `report.html`, `run_context.json`.

Every report carries a provenance header: data source, market count, date range,
event count, **number of configurations evaluated**, and execution scenario.

---

## Tests

```bash
pytest -q
```

The suite leads with the tests that matter: look-ahead prevention, execution-cost
realism, portfolio limits, credential redaction, and the live-trading lock.

---

## Documentation

* [`docs/assumptions_and_limitations.md`](docs/assumptions_and_limitations.md) — what is assumed and what cannot be known
* [`docs/bitvavo_api_notes.md`](docs/bitvavo_api_notes.md) — endpoints used, rate limits, history depth
* [`docs/research_protocol.md`](docs/research_protocol.md) — the protocol, in detail
