"""Tests for indicator calculations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from indicators import ema, rsi, atr, compute_all


def _make_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 10)
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    dates = pd.bdate_range(end="2023-12-31", periods=n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low,
         "close": close, "volume": volume},
        index=dates,
    )


class TestEMA:
    def test_values_between_min_max(self):
        df = _make_df()
        result = ema(df["close"], 20)
        assert result.between(df["close"].min() * 0.5,
                              df["close"].max() * 1.5).all()

    def test_no_nan_after_warmup(self):
        df = _make_df()
        result = ema(df["close"], 20)
        # ewm with adjust=False starts computing immediately
        assert result.iloc[50:].isna().sum() == 0

    def test_fast_ema_more_responsive(self):
        df = _make_df()
        fast = ema(df["close"], 10)
        slow = ema(df["close"], 50)
        # fast should be closer to current price on average
        diff_fast = (df["close"] - fast).abs().mean()
        diff_slow = (df["close"] - slow).abs().mean()
        assert diff_fast < diff_slow


class TestRSI:
    def test_bounded_0_to_100(self):
        df = _make_df()
        result = rsi(df["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rising_market_high_rsi(self):
        # Monotonically rising series should have RSI near 100
        close = pd.Series(range(1, 101), dtype=float)
        result = rsi(close, 14)
        valid = result.dropna()
        assert len(valid) > 0 and float(valid.iloc[-1]) > 90

    def test_falling_market_low_rsi(self):
        close = pd.Series(range(50, 0, -1), dtype=float)
        result = rsi(close, 14)
        assert result.iloc[-1] < 10


class TestATR:
    def test_positive(self):
        df = _make_df()
        result = atr(df, 14)
        assert (result.dropna() > 0).all()

    def test_wider_range_higher_atr(self):
        df1 = _make_df()
        df2 = df1.copy()
        df2["high"] = df2["close"] + 10
        df2["low"] = df2["close"] - 10
        a1 = atr(df1, 14).mean()
        a2 = atr(df2, 14).mean()
        assert a2 > a1


class TestComputeAll:
    def test_columns_present(self):
        df = _make_df()
        out = compute_all(df)
        expected = {"ema20", "ema50", "ema200", "rsi", "atr",
                    "avg_volume", "high52w", "low52w", "rs_spy", "rs_qqq"}
        assert expected.issubset(out.columns)

    def test_high52w_gte_close(self):
        df = _make_df()
        out = compute_all(df)
        assert (out["high52w"] >= out["close"]).all()

    def test_low52w_lte_close(self):
        df = _make_df()
        out = compute_all(df)
        assert (out["low52w"] <= out["close"]).all()
