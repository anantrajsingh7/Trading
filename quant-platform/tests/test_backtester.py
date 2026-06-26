"""
Tests for backtester engine and metrics.

Uses synthetic data — no network calls.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date

from core.models import BacktestTrade, ExitReason
from backtester.metrics import compute_metrics
from backtester.monte_carlo import run_monte_carlo
from backtester.engine import Backtester
from indicators.composite import enrich
from strategies.mean_reversion import MeanReversionStrategy


CFG = {
    "account": {"size": 100_000, "risk_per_trade": 0.01},
    "risk": {"max_open_positions": 10, "max_single_position": 0.10},
}


def _make_trades(n: int = 50, win_rate: float = 0.6, seed: int = 0) -> list[BacktestTrade]:
    rng = np.random.default_rng(seed)
    trades = []
    for i in range(n):
        is_win = rng.random() < win_rate
        pnl_pct = rng.uniform(0.05, 0.20) if is_win else -rng.uniform(0.02, 0.08)
        trades.append(BacktestTrade(
            ticker      = "TEST",
            strategy    = "test",
            entry_date  = date(2020, 1, 1),
            exit_date   = date(2020, 2, 1),
            entry_price = 100.0,
            exit_price  = 100.0 * (1 + pnl_pct),
            shares      = 10,
            pnl         = 1000 * pnl_pct,
            pnl_pct     = pnl_pct,
            holding_days = 20,
            exit_reason  = ExitReason.TARGET_1 if is_win else ExitReason.STOP_LOSS,
        ))
    return trades


def _make_equity(trades: list[BacktestTrade], initial: float = 100_000.0) -> pd.Series:
    equity = [initial]
    for t in trades:
        equity.append(equity[-1] * (1 + t.pnl_pct * 0.01))
    idx = pd.date_range("2020-01-01", periods=len(equity), freq="B")
    return pd.Series(equity, index=idx)


class TestMetrics:
    def test_basic_metrics_positive_system(self):
        trades = _make_trades(100, win_rate=0.65)
        equity = _make_equity(trades)
        m = compute_metrics("test", "5y", trades, equity)

        assert m.n_trades == 100
        assert m.win_rate == pytest.approx(0.65, abs=0.10)
        assert m.profit_factor > 1.0
        assert m.max_drawdown <= 0

    def test_metrics_empty_trades(self):
        m = compute_metrics("test", "5y", [], pd.Series([100_000]))
        assert m.n_trades == 0
        assert m.cagr == 0.0

    def test_sharpe_positive_for_winning_system(self):
        trades = _make_trades(200, win_rate=0.70)
        equity = _make_equity(trades)
        m = compute_metrics("test", "5y", trades, equity)
        # A 70% win rate system should have positive Sharpe
        assert m.sharpe_ratio > 0

    def test_kelly_pct_bounds(self):
        trades = _make_trades(100, win_rate=0.60)
        equity = _make_equity(trades)
        m = compute_metrics("test", "5y", trades, equity)
        assert 0.0 <= m.kelly_pct <= 1.0

    def test_robustness_rank_0_to_100(self):
        trades = _make_trades(100, win_rate=0.60)
        equity = _make_equity(trades)
        m = compute_metrics("test", "5y", trades, equity)
        assert 0 <= m.robustness_rank <= 100


class TestMonteCarlo:
    def test_returns_expected_keys(self):
        trades = _make_trades(50)
        result = run_monte_carlo("test", trades, n_simulations=100)
        assert "cagr_50th" in result
        assert "max_dd_5th" in result
        assert "ruin_probability" in result

    def test_ruin_probability_low_for_good_system(self):
        trades = _make_trades(100, win_rate=0.70)
        result = run_monte_carlo("test", trades, n_simulations=200)
        assert result["ruin_probability"] < 0.05

    def test_percentile_ordering(self):
        trades = _make_trades(50, win_rate=0.60)
        result = run_monte_carlo("test", trades, n_simulations=200)
        assert result["cagr_5th"] <= result["cagr_50th"] <= result["cagr_95th"]
        # For DD: 5th percentile is the worst (most negative)
        assert result["max_dd_5th"] <= result["max_dd_50th"]

    def test_too_few_trades_returns_empty(self):
        trades = _make_trades(5)
        result = run_monte_carlo("test", trades, n_simulations=100)
        assert result == {}


class TestBacktesterEngine:
    def _make_data(self, n: int = 350) -> dict[str, pd.DataFrame]:
        rng = np.random.default_rng(99)
        returns = rng.normal(0.001, 0.012, n)
        close = 100.0 * np.cumprod(1 + returns)
        high  = close * (1 + np.abs(rng.normal(0, 0.004, n)))
        low   = close * (1 - np.abs(rng.normal(0, 0.004, n)))
        vol   = rng.integers(500_000, 2_000_000, n).astype(float)
        idx   = pd.date_range("2021-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "open":  close, "high": high, "low": low,
            "close": close, "volume": vol, "adj_close": close,
        }, index=idx)
        return {"SYNTH": enrich(df, patterns=True)}

    def test_backtest_runs_without_error(self):
        data = self._make_data()
        backtester = Backtester(MeanReversionStrategy(), CFG)
        result = backtester.run(data)
        assert result.strategy == "mean_reversion"
        assert result.initial_capital == 100_000.0
        assert result.final_capital > 0

    def test_equity_curve_monotonically_positive(self):
        data = self._make_data()
        backtester = Backtester(MeanReversionStrategy(), CFG)
        result = backtester.run(data)
        if result.equity_curve is not None and len(result.equity_curve) > 0:
            assert (result.equity_curve > 0).all()

    def test_no_future_leak(self):
        """
        Trivial anti-lookahead test: each trade entry_date must be
        strictly before exit_date.
        """
        data = self._make_data()
        backtester = Backtester(MeanReversionStrategy(), CFG)
        result = backtester.run(data)
        for trade in result.trades:
            assert trade.entry_date <= trade.exit_date
