"""
dual_momentum_rotation — RESEARCH strategy (not in production scan).

WHY THIS STRATEGY, chosen from our own findings:

Our realistic-cost backtests showed friction of ~€1,828 across ~100 trades
(~€18 per round trip once spread, slippage, commission and EUR/USD
conversion are counted). On a €30,000 account that is the dominant drag —
it is what turned "10% CAGR" paper results into 1-3% realistic ones.

The structural answer to a cost problem is not a better entry signal, it
is FEWER TRADES. This strategy makes ~10-15 decisions a year instead of
hundreds, trades highly liquid ETFs with tight spreads instead of single
stocks, and therefore keeps essentially all of whatever edge it has.

It is also the most published edge in the literature (Antonacci's dual
momentum; Faber's tactical allocation): relative momentum picks what to
hold, absolute momentum decides whether to hold anything at all.

Mechanics
  * Monthly decision (last trading day), executed next session.
  * Relative momentum: blended 3/6/12-month total return, ranked across
    the sector/asset universe.
  * Absolute momentum: a sleeve is only held if its blended momentum
    also beats the cash proxy. Otherwise that sleeve goes to cash/bonds.
  * Hold top N equal-weighted. No stops — the monthly re-rank IS the
    risk control, which is what keeps turnover (and cost) low.

This is a portfolio-allocation strategy, not a trade picker, so it does
not implement BaseStrategy.generate_signal(). It has its own selector and
its own monthly-rebalance backtest.

Expected profile: modest returns, materially lower drawdown than
buy-and-hold, and very low cost drag. It is NOT a high-return strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Sector + broad-market sleeves (highly liquid, tight spreads)
RISK_UNIVERSE = [
    "XLK",  # technology
    "XLF",  # financials
    "XLV",  # health care
    "XLE",  # energy
    "XLI",  # industrials
    "XLY",  # consumer discretionary
    "XLP",  # consumer staples
    "XLU",  # utilities
    "XLB",  # materials
    "XLRE", # real estate
    "XLC",  # communication services
    "SPY",  # broad market
    "QQQ",  # nasdaq
]

# Defensive sleeves used when absolute momentum fails
SAFE_UNIVERSE = ["IEF", "SHY"]     # intermediate / short treasuries
CASH_PROXY = "BIL"                 # T-bill proxy: the absolute-momentum hurdle

TOP_N = 3
LOOKBACKS = (63, 126, 252)         # ~3, 6, 12 months
WEIGHTS = (1 / 3, 1 / 3, 1 / 3)


def blended_momentum(closes: pd.DataFrame,
                     lookbacks=LOOKBACKS,
                     weights=WEIGHTS) -> pd.DataFrame:
    """
    Causal blended total return. Row t uses only prices up to t.
    closes: DataFrame date x ticker.
    """
    out = None
    for lb, w in zip(lookbacks, weights):
        r = closes / closes.shift(lb) - 1
        out = r * w if out is None else out + r * w
    return out


def month_end_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last available trading day of each month."""
    s = pd.Series(index, index=index)
    return list(s.groupby([index.year, index.month]).last().values)


def select(momentum_row: pd.Series,
           cash_momentum: float,
           top_n: int = TOP_N,
           safe_assets: list[str] | None = None) -> dict[str, float]:
    """
    Return {ticker: weight} for one rebalance date.

    Relative momentum ranks the risk sleeves; absolute momentum then
    vetoes any sleeve that cannot beat cash. Vetoed sleeves are
    reallocated to the best defensive asset (or held as cash).
    """
    safe_assets = safe_assets or SAFE_UNIVERSE
    risk = momentum_row.reindex(
        [t for t in RISK_UNIVERSE if t in momentum_row.index]).dropna()
    if risk.empty:
        return {}

    ranked = risk.sort_values(ascending=False)
    chosen = list(ranked.index[:top_n])
    slot = 1.0 / top_n
    weights: dict[str, float] = {}

    # Best available defensive sleeve for vetoed slots
    safe_scores = momentum_row.reindex(
        [t for t in safe_assets if t in momentum_row.index]).dropna()
    best_safe = safe_scores.idxmax() if not safe_scores.empty else None

    for t in chosen:
        if np.isfinite(cash_momentum) and ranked[t] <= cash_momentum:
            # absolute momentum fails -> defensive, or stay in cash
            if best_safe is not None and safe_scores[best_safe] > cash_momentum:
                weights[best_safe] = weights.get(best_safe, 0.0) + slot
            # else: leave the slot in cash (weight simply not allocated)
        else:
            weights[t] = weights.get(t, 0.0) + slot
    return weights


def full_universe() -> list[str]:
    return RISK_UNIVERSE + SAFE_UNIVERSE + [CASH_PROXY]
