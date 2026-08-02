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

## Status of the research

**No empirical result has been produced yet, because no market data has been
downloaded.** The environment this repository was built in has no network route
to `api.bitvavo.com` (the egress proxy returns HTTP 403 on CONNECT for every
exchange host). Rather than substitute another exchange's data or invent
figures, the pipeline was built complete and verified end-to-end on clearly
labelled synthetic series.

To produce real results, run the pipeline from a machine that can reach Bitvavo:

```bash
python scripts/download_history.py --smoke-test          # verify connectivity
python scripts/download_history.py --markets top:40      # download
python scripts/run_research.py                           # Phase 2 event study
python scripts/run_backtest.py                           # Phases 3–8
python scripts/run_walk_forward.py                       # rolling validation
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
| minimum round-trip cost | 59 bps | **89 bps** | 170 bps |

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
