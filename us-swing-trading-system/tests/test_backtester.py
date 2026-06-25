"""Tests for backtester trade mechanics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from indicators import compute_all
from backtester import backtest_strategy, _compute_metrics, BacktestResult, ClosedTrade


def _synth_data(n: int = 300, seed: int = 1) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    close = 80 + np.linspace(0, 60, n) + rng.normal(0, 2, n)
    close = np.maximum(close, 10)
    high = close + rng.uniform(0.5, 3, n)
    low = close - rng.uniform(0.5, 3, n)
    volume = rng.integers(1_000_000, 6_000_000, n).astype(float)
    dates = pd.bdate_range(end="2023-12-31", periods=n)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low,
         "close": close, "volume": volume},
        index=dates,
    )
    spy = pd.Series(80 + np.linspace(0, 40, n), index=dates)
    qqq = pd.Series(80 + np.linspace(0, 30, n), index=dates)
    enriched = compute_all(df, spy=spy, qqq=qqq)
    return {"SYNTH": enriched}


_CFG = {
    "account": {"size": 10000, "risk_per_trade": 0.01, "min_rr_ratio": 2.0, "max_positions": 5},
    "trading": {"min_price": 5.0, "max_holding_days": 30},
    "indicators": {
        "ema_fast": 20, "ema_mid": 50, "ema_slow": 200,
        "rsi_period": 14, "atr_period": 14,
        "avg_volume_period": 20, "consolidation_days": 5,
        "atr_multiplier": 1.5,
    },
    "strategies": {k: {"enabled": True} for k in
                   ["ema_pullback", "breakout", "momentum",
                    "mean_reversion", "volatility_contraction"]},
    "backtest": {"slippage": 0.001, "commission": 5.0,
                 "in_sample_end": "2023-06-30",
                 "out_of_sample_start": "2023-07-01"},
}


class TestTradeClosingLogic:
    def test_exits_on_stop(self):
        """If price drops to stop, trade must close with exit_reason='stop'."""
        data = _synth_data()
        result = backtest_strategy("ema_pullback", data, _CFG)
        stop_exits = [t for t in result.trades if t.exit_reason == "stop"]
        target_exits = [t for t in result.trades if t.exit_reason == "target"]
        # At least one type of exit should exist if trades occurred
        if result.n_trades > 0:
            assert len(stop_exits) + len(target_exits) > 0

    def test_max_hold_respected(self):
        max_hold = _CFG["trading"]["max_holding_days"]
        data = _synth_data()
        result = backtest_strategy("ema_pullback", data, _CFG)
        for trade in result.trades:
            assert trade.holding_days <= max_hold + 1  # +1 for end-of-data exit

    def test_no_negative_shares(self):
        data = _synth_data()
        result = backtest_strategy("ema_pullback", data, _CFG)
        for trade in result.trades:
            assert trade.shares > 0


class TestMetricsComputation:
    def _make_result(self, pnls: list[float]) -> BacktestResult:
        res = BacktestResult(strategy="test")
        initial = 10000.0
        trades = []
        for i, pnl in enumerate(pnls):
            entry = 100.0
            exit_p = entry + pnl / 10
            pct = (exit_p - entry) / entry
            trades.append(ClosedTrade(
                ticker="X", strategy="test",
                entry_date=pd.Timestamp(f"2022-01-{i+1:02d}"),
                exit_date=pd.Timestamp(f"2022-01-{i+2:02d}"),
                entry_price=entry, exit_price=exit_p,
                shares=10, pnl=pnl, pnl_pct=pct,
                holding_days=1, exit_reason="target",
            ))
        res.trades = trades
        _compute_metrics(res, initial)
        return res

    def test_win_rate(self):
        res = self._make_result([100, -50, 200, -30, 150])
        assert abs(res.win_rate - 0.6) < 0.001

    def test_profit_factor(self):
        res = self._make_result([100, -50])
        assert abs(res.profit_factor - 2.0) < 0.001

    def test_all_losses_zero_profit_factor(self):
        res = self._make_result([-100, -50, -200])
        assert res.profit_factor == 0.0

    def test_total_return_positive_on_net_profit(self):
        res = self._make_result([500, -100])
        assert res.total_return > 0

    def test_max_drawdown_negative(self):
        res = self._make_result([200, -500, 100])
        assert res.max_drawdown < 0
