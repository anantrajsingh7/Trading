"""
Performance metrics computation.

Takes a list of BacktestTrade and an equity curve (pd.Series indexed by date)
and returns a fully populated PerformanceMetrics object.

Never optimises for return alone. Always computes risk-adjusted metrics.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from core.models import BacktestTrade, PerformanceMetrics


def compute_metrics(
    strategy: str,
    period: str,
    trades: list[BacktestTrade],
    equity_curve: pd.Series,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.05,
) -> PerformanceMetrics:
    """
    Compute full institutional performance metrics.

    equity_curve — daily portfolio value indexed by date
    trades       — list of completed BacktestTrade records
    """
    m = PerformanceMetrics(strategy=strategy, period=period)

    if not trades or len(equity_curve) < 2:
        return m

    # ── Trade statistics ───────────────────────────────────────────
    m.n_trades  = len(trades)
    winners     = [t for t in trades if t.pnl > 0]
    losers      = [t for t in trades if t.pnl <= 0]
    m.n_winners = len(winners)
    m.n_losers  = len(losers)

    m.win_rate  = m.n_winners / m.n_trades
    m.loss_rate = m.n_losers  / m.n_trades

    win_pcts  = [t.pnl_pct for t in winners]
    loss_pcts = [abs(t.pnl_pct) for t in losers]

    m.avg_win_pct  = float(np.mean(win_pcts))  if win_pcts  else 0.0
    m.avg_loss_pct = float(np.mean(loss_pcts)) if loss_pcts else 0.0

    m.avg_win_loss_ratio = (
        m.avg_win_pct / m.avg_loss_pct if m.avg_loss_pct > 0 else 0.0
    )

    total_wins   = sum(t.pnl for t in winners)
    total_losses = abs(sum(t.pnl for t in losers))
    m.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    # Kelly %
    if m.avg_loss_pct > 0:
        m.kelly_pct = m.win_rate - (m.loss_rate / m.avg_win_loss_ratio)
        m.kelly_pct = max(0.0, m.kelly_pct)

    # Expectancy (R multiples approximation)
    m.expectancy = m.win_rate * m.avg_win_pct - m.loss_rate * m.avg_loss_pct

    # Average hold
    m.avg_hold_days = float(np.mean([t.holding_days for t in trades]))

    # ── Return metrics ─────────────────────────────────────────────
    final_val     = float(equity_curve.iloc[-1])
    m.total_return = (final_val - initial_capital) / initial_capital

    n_years = len(equity_curve) / 252.0
    if n_years > 0:
        m.cagr = (final_val / initial_capital) ** (1 / n_years) - 1
    m.annual_return = m.cagr

    m.trades_per_year = m.n_trades / max(n_years, 0.01)

    # ── Daily returns ─────────────────────────────────────────────
    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) < 2:
        return m

    ann_factor = math.sqrt(252)
    mean_ret   = float(daily_returns.mean())
    std_ret    = float(daily_returns.std(ddof=1))
    down_std   = float(daily_returns[daily_returns < 0].std(ddof=1)) or std_ret

    excess_ret = mean_ret - risk_free_rate / 252
    m.sharpe_ratio  = (excess_ret / std_ret * ann_factor) if std_ret > 0 else 0.0
    m.sortino_ratio = (excess_ret / down_std * ann_factor) if down_std > 0 else 0.0

    # ── Drawdown ──────────────────────────────────────────────────
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max

    m.max_drawdown = float(drawdown.min())
    m.avg_drawdown = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0

    # Longest drawdown (days underwater)
    in_dd = drawdown < 0
    if in_dd.any():
        # Find contiguous underwater periods
        transitions = in_dd.astype(int).diff().fillna(0)
        starts = transitions[transitions == 1].index
        ends   = transitions[transitions == -1].index
        if len(starts) > 0:
            max_days = 0
            for start in starts:
                subsequent_ends = ends[ends > start]
                end = subsequent_ends[0] if len(subsequent_ends) > 0 else equity_curve.index[-1]
                days = (end - start).days
                max_days = max(max_days, days)
            m.longest_drawdown_days = max_days

    m.calmar_ratio    = m.cagr / abs(m.max_drawdown) if m.max_drawdown != 0 else 0.0
    m.recovery_factor = m.total_return / abs(m.max_drawdown) if m.max_drawdown != 0 else 0.0

    # ── Consistency score ─────────────────────────────────────────
    # Percentage of rolling 12-month periods that were profitable
    if len(daily_returns) >= 252:
        rolling_1yr = daily_returns.rolling(252).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        ).dropna()
        profitable_pct = (rolling_1yr > 0).mean()
        m.consistency_score = float(profitable_pct * 100)

    # ── Robustness rank (composite score) ────────────────────────
    # Weighted combination that avoids overfitting on raw return
    scores = []
    if m.sharpe_ratio > 0:    scores.append(min(m.sharpe_ratio / 3.0, 1.0) * 25)
    if m.profit_factor > 0:   scores.append(min((m.profit_factor - 1) / 2.0, 1.0) * 20)
    if m.win_rate > 0:        scores.append(min(m.win_rate / 0.6, 1.0) * 15)
    if m.max_drawdown < 0:    scores.append(max(1 + m.max_drawdown / 0.3, 0.0) * 20)
    scores.append(m.consistency_score / 100 * 20)
    m.robustness_rank = float(sum(scores))

    return m


def print_metrics(m: PerformanceMetrics) -> None:
    """Pretty-print metrics to stdout."""
    print(f"\n{'─'*55}")
    print(f"Strategy: {m.strategy}  |  Period: {m.period}")
    print(f"{'─'*55}")
    print(f"Trades:         {m.n_trades:>6}   Win rate:   {m.win_rate*100:>6.1f}%")
    print(f"CAGR:           {m.cagr*100:>6.1f}%  Profit factor: {m.profit_factor:>5.2f}")
    print(f"Sharpe:         {m.sharpe_ratio:>6.2f}   Sortino:    {m.sortino_ratio:>6.2f}")
    print(f"Max Drawdown:   {m.max_drawdown*100:>6.1f}%  Calmar:     {m.calmar_ratio:>6.2f}")
    print(f"Expectancy:     {m.expectancy*100:>6.2f}%  Kelly%:     {m.kelly_pct*100:>6.1f}%")
    print(f"Consistency:    {m.consistency_score:>6.1f}%  Robustness: {m.robustness_rank:>6.1f}")
    print(f"{'─'*55}\n")
