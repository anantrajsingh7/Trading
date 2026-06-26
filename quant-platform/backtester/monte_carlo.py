"""
Monte Carlo simulation for robustness testing.

Takes a list of BacktestTrade results and simulates 1000 alternative
orderings to answer: "How sensitive is this strategy to the order
trades happened in?"

Key insight: If performance varies wildly across simulations, the strategy
is sensitive to clustering of wins/losses — a sign of fragility.
Robust strategies show consistent metrics across all simulations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.models import BacktestTrade, PerformanceMetrics
from backtester.metrics import compute_metrics
from core.logging import get_logger

log = get_logger("backtester.monte_carlo")


def run_monte_carlo(
    strategy: str,
    trades: list[BacktestTrade],
    initial_capital: float = 100_000.0,
    n_simulations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo simulation via trade shuffling.

    Returns dict with:
      cagr_5th, cagr_50th, cagr_95th
      max_dd_5th, max_dd_50th, max_dd_95th (worst = 5th pct for DD)
      sharpe_5th, sharpe_50th, sharpe_95th
      ruin_probability: fraction of simulations where DD > 50%
      all_metrics: list of PerformanceMetrics (one per sim)
    """
    if len(trades) < 10:
        log.warning("Monte Carlo requires ≥10 trades, got %d", len(trades))
        return {}

    rng = np.random.default_rng(seed)
    pnl_pcts = np.array([t.pnl_pct for t in trades])

    cagrs     = []
    max_dds   = []
    sharpes   = []
    all_finals = []

    for _ in range(n_simulations):
        shuffled = rng.choice(pnl_pcts, size=len(pnl_pcts), replace=True)
        equity   = _simulate_equity(initial_capital, shuffled)
        ec = pd.Series(equity)

        final_capital = equity[-1]
        total_return  = (final_capital - initial_capital) / initial_capital
        n_years       = len(equity) / 252.0
        cagr = (final_capital / initial_capital) ** (1 / max(n_years, 0.001)) - 1

        rolling_max = ec.cummax()
        dd = (ec - rolling_max) / rolling_max
        max_dd = float(dd.min())

        daily_ret = ec.pct_change().dropna()
        std = float(daily_ret.std(ddof=1))
        sharpe = (float(daily_ret.mean()) / std * np.sqrt(252)) if std > 0 else 0.0

        cagrs.append(cagr)
        max_dds.append(max_dd)
        sharpes.append(sharpe)
        all_finals.append(final_capital)

    alpha = 1 - confidence
    low_p  = alpha / 2 * 100
    high_p = (1 - alpha / 2) * 100

    result = {
        "n_simulations": n_simulations,
        "confidence":    confidence,
        # CAGR percentiles
        "cagr_5th":    float(np.percentile(cagrs, 5)),
        "cagr_50th":   float(np.percentile(cagrs, 50)),
        "cagr_95th":   float(np.percentile(cagrs, 95)),
        # Drawdown (lower is worse)
        "max_dd_5th":  float(np.percentile(max_dds, 5)),   # Worst 5%
        "max_dd_50th": float(np.percentile(max_dds, 50)),
        "max_dd_95th": float(np.percentile(max_dds, 95)),  # Best 5%
        # Sharpe
        "sharpe_5th":  float(np.percentile(sharpes, 5)),
        "sharpe_50th": float(np.percentile(sharpes, 50)),
        "sharpe_95th": float(np.percentile(sharpes, 95)),
        # Ruin probability (equity drops >50%)
        "ruin_probability": float(np.mean([d < -0.50 for d in max_dds])),
    }

    log.info(
        "%s MC(%d): CAGR 5th/50th/95th = %.1f%%/%.1f%%/%.1f%% | DD worst = %.1f%%",
        strategy, n_simulations,
        result["cagr_5th"] * 100, result["cagr_50th"] * 100, result["cagr_95th"] * 100,
        result["max_dd_5th"] * 100,
    )
    return result


def _simulate_equity(initial: float, pnl_pcts: np.ndarray) -> np.ndarray:
    """Compound a sequence of trade returns on initial capital."""
    equity = np.empty(len(pnl_pcts) + 1)
    equity[0] = initial
    # Each trade uses 1% of current capital as risk
    for i, ret in enumerate(pnl_pcts):
        equity[i + 1] = equity[i] * (1 + ret * 0.01)  # 1% risk sizing approximation
    return equity
