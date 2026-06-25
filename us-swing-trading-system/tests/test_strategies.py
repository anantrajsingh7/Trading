"""Tests for strategy signal generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from indicators import compute_all
from strategies import (
    ema_pullback_signals,
    breakout_signals,
    momentum_signals,
    mean_reversion_signals,
    volatility_contraction_signals,
    get_signals,
    STRATEGIES,
)


def _uptrend_df(n: int = 300) -> pd.DataFrame:
    """Synthetic uptrend market data with indicators."""
    rng = np.random.default_rng(7)
    close = 50 + np.linspace(0, 100, n) + rng.normal(0, 2, n)
    close = np.maximum(close, 10)
    high = close + rng.uniform(0.5, 3, n)
    low = close - rng.uniform(0.5, 3, n)
    volume = rng.integers(1_000_000, 8_000_000, n).astype(float)
    dates = pd.bdate_range(end="2023-12-31", periods=n)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low,
         "close": close, "volume": volume},
        index=dates,
    )
    spy = pd.Series(50 + np.linspace(0, 80, n) + rng.normal(0, 1, n), index=dates)
    qqq = pd.Series(50 + np.linspace(0, 70, n) + rng.normal(0, 1, n), index=dates)
    return compute_all(df, spy=spy, qqq=qqq)


def _downtrend_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    close = 150 - np.linspace(0, 100, n) + rng.normal(0, 2, n)
    close = np.maximum(close, 5)
    high = close + rng.uniform(0.5, 3, n)
    low = close - rng.uniform(0.5, 3, n)
    volume = rng.integers(500_000, 2_000_000, n).astype(float)
    dates = pd.bdate_range(end="2023-12-31", periods=n)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low,
         "close": close, "volume": volume},
        index=dates,
    )
    return compute_all(df)


_DEFAULT_CFG = {
    "indicators": {
        "ema_fast": 20, "ema_mid": 50, "ema_slow": 200,
        "rsi_period": 14, "atr_period": 14,
        "avg_volume_period": 20, "consolidation_days": 5,
        "atr_multiplier": 1.5,
    },
    "account": {"min_rr_ratio": 2.0},
}


class TestSignalReturnTypes:
    """All signal functions must return boolean Series of same length."""

    def test_all_strategies_return_series(self):
        df = _uptrend_df()
        for key in STRATEGIES:
            sig = get_signals(df, key, _DEFAULT_CFG)
            assert isinstance(sig, pd.Series), f"{key} should return Series"
            assert len(sig) == len(df), f"{key} length mismatch"
            assert sig.dtype == bool or sig.dtype == object

    def test_no_future_leak(self):
        """Signal at index i must not use data from i+1 onwards."""
        df = _uptrend_df()
        # If we remove the last row, signals for all earlier rows must be identical
        for key in STRATEGIES:
            sig_full = get_signals(df, key, _DEFAULT_CFG)
            sig_trimmed = get_signals(df.iloc[:-1], key, _DEFAULT_CFG)
            # All but the last row should match
            pd.testing.assert_series_equal(
                sig_full.iloc[:-1].reset_index(drop=True),
                sig_trimmed.reset_index(drop=True),
                check_names=False,
            )


class TestEMAPullback:
    def test_requires_uptrend(self):
        df_down = _downtrend_df()
        sig = ema_pullback_signals(df_down, _DEFAULT_CFG)
        # In a clear downtrend, should produce very few or no signals
        assert sig.sum() < len(df_down) * 0.05

    def test_uptrend_can_produce_signals(self):
        df = _uptrend_df(n=500)
        sig = ema_pullback_signals(df, _DEFAULT_CFG)
        # Not asserting specific count, just that the function runs cleanly
        assert isinstance(sig, pd.Series)


class TestBreakout:
    def test_volume_requirement(self):
        df = _uptrend_df()
        # Zero out volume – no breakout signals should fire
        df_low_vol = df.copy()
        df_low_vol["volume"] = 100
        df_low_vol["avg_volume"] = 1_000_000
        sig = breakout_signals(df_low_vol, _DEFAULT_CFG)
        assert sig.sum() == 0

    def test_below_ema200_no_signal(self):
        df = _downtrend_df()
        sig = breakout_signals(df, _DEFAULT_CFG)
        # Close below EMA200 should suppress all breakout signals
        below_200 = df["close"] < df["ema200"]
        assert (sig & below_200).sum() == 0


class TestMomentum:
    def test_rsi_filter(self):
        df = _uptrend_df(n=500)
        # Force RSI way out of range
        df_mod = df.copy()
        df_mod["rsi"] = 90.0
        sig = momentum_signals(df_mod, _DEFAULT_CFG)
        assert sig.sum() == 0


class TestGetSignals:
    def test_unknown_strategy_raises(self):
        df = _uptrend_df()
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_signals(df, "nonexistent_strategy", _DEFAULT_CFG)
