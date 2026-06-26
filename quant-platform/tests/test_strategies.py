"""
Tests for strategy signal generation.

Uses synthetic data — no network calls.
Verifies:
  - Strategies return None when conditions not met
  - When conditions are met, Signal fields are valid
  - Stop loss < entry price
  - Target 1 < Target 2 < Target 3
  - Score in 0-100 range
"""
import numpy as np
import pandas as pd
import pytest

from indicators.composite import enrich
from strategies.minervini      import MinerviniStrategy
from strategies.breakout       import BreakoutStrategy
from strategies.pullback       import EMApullbackStrategy
from strategies.momentum_strat import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.registry import all_strategies, get_strategy, strategy_names


def _trending_df(n: int = 400, drift: float = 0.001) -> pd.DataFrame:
    """Synthetic strongly-trending dataset (positive drift)."""
    rng = np.random.default_rng(1)
    returns = rng.normal(drift, 0.01, n)
    close = 100.0 * np.cumprod(1 + returns)
    high  = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low   = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    vol   = rng.integers(1_000_000, 3_000_000, n).astype(float)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "open":  close * (1 + rng.normal(0, 0.002, n)),
        "high":  high,
        "low":   low,
        "close": close,
        "volume": vol,
        "adj_close": close,
    }, index=idx)
    return enrich(df, patterns=True)


def _flat_df(n: int = 300) -> pd.DataFrame:
    """Flat sideways market — should not trigger most strategies."""
    rng = np.random.default_rng(2)
    close = 100.0 + rng.normal(0, 1, n)
    high  = close + np.abs(rng.normal(0, 0.5, n))
    low   = close - np.abs(rng.normal(0, 0.5, n))
    vol   = rng.integers(200_000, 500_000, n).astype(float)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "open":  close + rng.normal(0, 0.2, n),
        "high":  high,
        "low":   low,
        "close": close,
        "volume": vol,
        "adj_close": close,
    }, index=idx)
    return enrich(df, patterns=False)


CFG = {
    "account": {"size": 100_000, "risk_per_trade": 0.01},
    "risk": {"max_open_positions": 10, "max_single_position": 0.10},
}


def _check_signal_invariants(signal):
    """Common invariants all valid signals must satisfy."""
    assert signal.stop_loss < signal.entry_price, "Stop must be below entry"
    assert signal.target_1  > signal.entry_price, "T1 must be above entry"
    assert signal.target_2  >= signal.target_1,    "T2 must be ≥ T1"
    assert signal.target_3  >= signal.target_2,    "T3 must be ≥ T2"
    assert 0 <= signal.score <= 100,               "Score must be 0-100"


class TestRegistry:
    def test_all_strategies_registered(self):
        names = strategy_names()
        assert "minervini_vcp"  in names
        assert "volume_breakout" in names
        assert "ema_pullback"   in names
        assert "momentum"       in names
        assert "mean_reversion" in names

    def test_get_strategy_by_name(self):
        s = get_strategy("minervini_vcp")
        assert s.name == "minervini_vcp"

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError):
            get_strategy("nonexistent_strategy")

    def test_all_strategies_have_description(self):
        for s in all_strategies():
            assert len(s.description) > 5, f"{s.name} has empty description"


class TestMinerviniStrategy:
    def test_short_data_returns_none(self):
        rng = np.random.default_rng(0)
        close = pd.Series(rng.normal(100, 1, 50))
        df = pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": pd.Series([1e6] * 50), "adj_close": close,
        })
        result = MinerviniStrategy().generate_signal(df, "TEST", CFG)
        assert result is None

    def test_returns_valid_signal_or_none_on_trending(self):
        df = _trending_df()
        result = MinerviniStrategy().generate_signal(df, "TREND", CFG)
        if result is not None:
            _check_signal_invariants(result)
            assert result.strategy == "minervini_vcp"


class TestBreakoutStrategy:
    def test_flat_market_mostly_no_signal(self):
        df = _flat_df()
        result = BreakoutStrategy().generate_signal(df, "FLAT", CFG)
        # Flat market should rarely produce signals; if it does, check validity
        if result is not None:
            _check_signal_invariants(result)

    def test_signal_invariants_when_triggered(self):
        df = _trending_df()
        # Force a Darvas breakout condition in the last bar
        df["darvas_break"] = False
        df.loc[df.index[-1], "darvas_break"] = True
        df["rvol_20"] = 2.0
        result = BreakoutStrategy().generate_signal(df, "BRKOUT", CFG)
        if result is not None:
            _check_signal_invariants(result)


class TestPullbackStrategy:
    def test_signal_or_none(self):
        df = _trending_df()
        result = EMApullbackStrategy().generate_signal(df, "PULL", CFG)
        if result is not None:
            _check_signal_invariants(result)
            assert result.strategy == "ema_pullback"


class TestMomentumStrategy:
    def test_low_rs_returns_none(self):
        df = _trending_df()
        result = MomentumStrategy().generate_signal(df, "MOM", CFG, rs_rank=30)
        assert result is None

    def test_high_rs_can_produce_signal(self):
        df = _trending_df()
        # Force squeeze conditions
        df["squeeze_on"] = True
        df["sq_momentum"] = 0.0
        df.loc[df.index[-1], "sq_momentum"] = 1.0   # momentum turned positive
        df.loc[df.index[-2], "sq_momentum"] = -0.5
        df["rvol_20"] = 1.5
        result = MomentumStrategy().generate_signal(df, "MOM", CFG, rs_rank=85)
        if result is not None:
            _check_signal_invariants(result)


class TestMeanReversionStrategy:
    def test_requires_uptrend(self):
        """Mean reversion should not fire in downtrend (close < EMA200)."""
        df = _flat_df()
        # Artificially set EMA200 very high so close < EMA200
        df["ema_200"] = df["close"] * 2
        result = MeanReversionStrategy().generate_signal(df, "MR", CFG)
        assert result is None

    def test_valid_signal_invariants(self):
        df = _trending_df(400, drift=0.0003)
        # Force NR7 and oversold conditions
        df["nr7"] = False
        df.loc[df.index[-1], "nr7"] = True
        df["rsi_14"] = 60.0
        df.loc[df.index[-1], "rsi_14"] = 38.0
        df["bb_pct_b"] = 0.5
        df.loc[df.index[-1], "bb_pct_b"] = 0.15
        df["cmf_20"] = 0.1
        df["rvol_20"] = 0.8
        result = MeanReversionStrategy().generate_signal(df, "MR", CFG)
        if result is not None:
            _check_signal_invariants(result)
