# Research protocol

The order of operations, and why each step comes where it does. Following this
in order is what makes the eventual answer trustworthy; running it out of order
is how a system convinces itself of something false.

---

## The question

> When a Bitvavo EUR market rises ~10–12% within two to three hours, does it
> continue often enough to yield a **positive net expectancy after realistic
> fees, spread, slippage and failed fills**?

The null hypothesis is that the forward-return distribution after such an event
is indistinguishable from a randomly-timed entry in the same market over the same
holding period. That null is the thing evidence has to defeat.

Central metric:

```
net expectancy per trade = win_rate × avg_net_win − loss_rate × avg_net_loss
```

Win rate is explicitly *not* a selection criterion.

---

## Step 1 — Data (Phase 1)

```bash
python scripts/download_history.py --smoke-test
python scripts/download_history.py --markets top:40 --interval 1m
```

The smoke test comes first and is not optional: it verifies connectivity, clock
skew, timestamp ordering, duplicates, and that validation passes on one market
over a short range. Only then is a large download worth starting.

Every market gets a manifest row: first/last timestamp, row count, missing
intervals, duplicates, zero-volume periods, download time, API source, validation
status. Nothing downstream consumes a dataset whose row says `FAIL`.

Raw API payloads are written to `data/raw/` **before** parsing, so a parsing bug
found in six months can be fixed without re-downloading.

---

## Step 2 — Describe before you trade (Phase 2)

```bash
python scripts/run_research.py
```

This step produces **no strategy**. It answers, descriptively:

* Across look-backs {30, 60, 90, 120, 180, 240} minutes and thresholds {5, 7.5,
  10, 12, 15, 20}%, what does the forward-return distribution look like at
  horizons out to 48 hours?
* What are the mean, median, hit rate, MFE and MAE?
* What does a moving-block bootstrap say about the confidence interval on the
  mean?

Forward returns are measured from the **open of the bar after the event bar** —
the first price a trader could realistically have transacted at. These figures
are **gross**. The minimum round-trip cost in the realistic scenario is ~89 bps;
any mean forward return below that is not a trading opportunity, whatever its
t-statistic.

If the event study shows nothing at any horizon, the honest thing is to stop
here and report that.

---

## Step 3 — Strategies and costs (Phases 3–5)

```bash
python scripts/run_backtest.py
```

Eight entry strategies (A–H) × exit policies × three execution scenarios, all
compared on **training data only**. Benchmarks — random entry matched by market
and window, and do-nothing — are in the same table, run through the same code.

Two disciplines matter here:

1. **Every event produces a row**, whether it became a trade or not. The funnel
   from "event detected" to "trade taken" is always visible, so a strategy that
   looks good because it silently skipped 90% of events is caught immediately.
2. **Portfolio constraints apply chronologically.** Positions occupy slots and
   risk budget from entry to exit, so no strategy can pretend to have taken
   twenty simultaneous trades on a €10,000 account.

---

## Step 4 — Chronological validation (Phase 6)

```
train (earliest 50%) │ embargo │ validation (25%) │ embargo │ test (25%, LOCKED)
```

* Parameters are optimised on **train**.
* Strategy selection uses **validation**.
* The **test set stays locked** (`splits.unlock_test_set: false`) until a single
  final evaluation. Unlocking it is a config change that appears in Git history.

Then rolling walk-forward:

```bash
python scripts/run_walk_forward.py
```

Train 12 months → test the next 3 → roll forward 3 → repeat. The exit policy is
re-selected inside every training window, so the concatenated test record
consists only of decisions made without knowledge of the period they were applied
to. A strategy that is genuinely robust produces a *majority of positive windows*,
not one spectacular window and several bad ones.

---

## Step 5 — Robustness (Phase 8)

* Moving-block bootstrap confidence intervals (blocks, because trades cluster).
* Monte Carlo reshuffling of the trade *sequence* — how much of the observed
  drawdown is ordering luck?
* Permutation test against random entries in the same markets.
* Deflated Sharpe ratio, discounted for the number of configurations evaluated.
* Parameter-stability neighbourhood analysis: a configuration that works only at
  one exact parameter setting, with poor neighbours, is a curve-fit. A plateau is
  a finding.
* Feature ablation: remove each filter in turn. A filter that can be removed
  without hurting out-of-sample performance has not earned its complexity.
* Cost sensitivity: scale costs 0.5× to 3× and find where the edge dies.

---

## Step 6 — Accept or reject (Phase 7)

Rejection is automatic, coded in `metrics.rejection_reasons`:

| Criterion | Threshold |
|---|---|
| independent trades | < 100 → reject |
| net expectancy after costs | ≤ 0 → reject |
| profit share from the top coin | > 40% → reject |
| profit share from the top month | > 40% → reject |
| maximum drawdown | worse than −25% → reject |
| stress-scenario expectancy | ≤ 0 → reject |
| intrabar-ambiguous exits | > 30% → reject |

"Insufficient evidence" is a valid, expected, and often correct outcome. The
report says so plainly.

---

## Step 7 — Paper trading (Phase 11)

Only after a candidate survives the above. Approval is a deliberate act: writing
`data/results/approved_strategy.json`. The scanner **refuses to start** without
it — it will not invent parameters.

Paper trading uses the *same execution model* as the backtest, so paper results
are directly comparable with the research expectation. Every signal is recorded,
including rejected and expired ones. The tested rules are never changed after
seeing a trade outcome; changing them means going back to Step 3 with a new
configuration and a fresh out-of-sample window.

---

## Step 8 — Live trading

Not covered by this repository. It requires three independent switches *and* an
execution adapter that does not exist here and would have to be written and
reviewed deliberately. That is the point.

---

## Discipline notes

* **Do not look at the test set while designing.** The lock is a speed bump, not
  a cage; the discipline is yours.
* **Record every rejection.** `rejected_strategies.md` exists so the number of
  hypotheses tested is known. A winner drawn from an unrecorded search of 200
  candidates is not a winner.
* **Prefer the simpler strategy** when two perform similarly. Every extra filter
  is a degree of freedom that has to be paid for out of the same finite data.
* **Report the clustered trade count**, not the raw one, when claiming a sample
  size.
