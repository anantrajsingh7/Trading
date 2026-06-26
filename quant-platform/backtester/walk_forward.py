"""
Walk-Forward Optimization.

Splits the backtest period into rolling in-sample / out-of-sample windows
to test whether a strategy holds up on unseen data.

Method:
  - Window size: configurable (default: 2 years in-sample, 6 months OOS)
  - Rolls forward by OOS period each iteration
  - In-sample: used for parameter selection (not done here — strategy params
    are fixed by design to avoid overfitting)
  - Out-of-sample: pure holdout, untouched during development

Why this matters:
  A strategy that backtests well on 5 years of data but fails walk-forward
  is overfit to that specific period. Walk-forward is the minimum robustness
  check before treating results as credible.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from backtester.engine import Backtester
from core.models import BacktestResult
from core.logging import get_logger
from strategies.base import BaseStrategy

log = get_logger("backtester.walk_forward")


def walk_forward(
    strategy: BaseStrategy,
    data: dict[str, pd.DataFrame],
    cfg: dict,
    in_sample_years: float = 2.0,
    oos_months: int = 6,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict:
    """
    Run walk-forward optimization.

    Returns dict with:
      windows: list of per-window BacktestResult (OOS only)
      combined_trades: all OOS trades concatenated
      summary: aggregated OOS metrics
    """
    # Date range
    all_dates = sorted(set(
        d for df in data.values() for d in df.index.date
    ))
    if not all_dates:
        return {}

    period_start = start_date or all_dates[0]
    period_end   = end_date   or all_dates[-1]

    in_sample_days = int(in_sample_years * 365)
    oos_days       = int(oos_months * 30)

    windows = []
    cursor  = period_start

    while True:
        is_start = cursor
        is_end   = cursor + timedelta(days=in_sample_days)
        oos_start = is_end + timedelta(days=1)
        oos_end   = oos_start + timedelta(days=oos_days)

        if oos_start >= period_end:
            break

        oos_end = min(oos_end, period_end)

        log.info(
            "WF window: IS %s→%s | OOS %s→%s",
            is_start, is_end, oos_start, oos_end,
        )

        backtester = Backtester(strategy, cfg)
        oos_result = backtester.run(data, start_date=oos_start, end_date=oos_end)
        windows.append({
            "is_start":  is_start,
            "is_end":    is_end,
            "oos_start": oos_start,
            "oos_end":   oos_end,
            "result":    oos_result,
        })

        cursor = oos_start

    if not windows:
        return {"windows": [], "n_windows": 0}

    all_oos_trades = []
    for w in windows:
        all_oos_trades.extend(w["result"].trades)

    n_profitable = sum(
        1 for w in windows
        if w["result"].metrics and w["result"].metrics.cagr > 0
    )

    summary = {
        "n_windows":           len(windows),
        "n_profitable_windows": n_profitable,
        "pct_profitable":      n_profitable / len(windows),
        "total_oos_trades":    len(all_oos_trades),
        "windows":             windows,
        "all_oos_trades":      all_oos_trades,
    }

    log.info(
        "Walk-forward: %d windows, %d profitable (%.0f%%), %d OOS trades",
        len(windows), n_profitable, n_profitable / len(windows) * 100,
        len(all_oos_trades),
    )

    return summary
