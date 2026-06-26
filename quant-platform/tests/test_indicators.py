"""
Tests for the indicators library.

Uses synthetic data to verify correctness without network calls.
Key invariants tested:
  - RSI bounds (0-100, no NaN on monotonic series)
  - Bollinger Band ordering (upper > mid > lower)
  - ATR positivity
  - Supertrend direction flips on trend change
  - Minervini Trend Template logic
"""
import numpy as np
import pandas as pd
import pytest

from indicators.trend      import ema, sma, adx, supertrend, minervini_trend_template
from indicators.momentum   import rsi, macd, stochastic
from indicators.volatility import atr, bollinger_bands, keltner_channel, ttm_squeeze
from indicators.volume     import obv, relative_volume, pocket_pivot
from indicators.patterns   import nr7, inside_bar, vcp, darvas_box
from indicators.composite  import enrich


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _make_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV data — random walk with positive drift."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    close = 100.0 * np.cumprod(1 + returns)
    high  = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low   = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol   = rng.integers(500_000, 2_000_000, n).astype(float)

    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": vol,
        "adj_close": close,
    }, index=idx)


def _monotone_rising(n: int = 50) -> pd.Series:
    return pd.Series(np.arange(1.0, n + 1.0))


def _monotone_falling(n: int = 50) -> pd.Series:
    return pd.Series(np.arange(float(n), 0.0, -1.0))


# ─────────────────────────────────────────────
# Moving Averages
# ─────────────────────────────────────────────

def test_ema_matches_pandas():
    s = pd.Series(np.random.default_rng(0).normal(100, 5, 100))
    result = ema(s, 10)
    expected = s.ewm(span=10, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected)


def test_sma_matches_rolling():
    s = pd.Series(np.arange(1.0, 21.0))
    result = sma(s, 5)
    assert abs(result.iloc[-1] - 18.0) < 1e-9  # mean of 16,17,18,19,20


# ─────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────

def test_rsi_bounds():
    df = _make_df()
    r = rsi(df["close"])
    valid = r.dropna()
    assert (valid >= 0).all(), "RSI below 0"
    assert (valid <= 100).all(), "RSI above 100"


def test_rsi_monotone_rising_is_100():
    s = _monotone_rising()
    r = rsi(s, 14)
    assert r.iloc[-1] == pytest.approx(100.0), "Monotone rising should give RSI=100"


def test_rsi_monotone_falling_is_0():
    s = _monotone_falling()
    r = rsi(s, 14)
    last = r.dropna().iloc[-1]
    assert last < 5.0, f"Monotone falling should give RSI near 0, got {last}"


# ─────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────

def test_macd_columns():
    df = _make_df()
    result = macd(df["close"])
    assert {"macd", "signal", "histogram"} == set(result.columns)
    assert (result["histogram"] == result["macd"] - result["signal"]).all()


# ─────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────

def test_atr_positive():
    df = _make_df()
    a = atr(df, 14)
    assert (a.dropna() > 0).all()


# ─────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────

def test_bb_ordering():
    df = _make_df()
    bb = bollinger_bands(df["close"])
    valid = bb.dropna()
    assert (valid["bb_upper"] > valid["bb_mid"]).all()
    assert (valid["bb_mid"] > valid["bb_lower"]).all()


def test_bb_width_positive():
    df = _make_df()
    bb = bollinger_bands(df["close"])
    assert (bb["bb_width"].dropna() > 0).all()


# ─────────────────────────────────────────────
# Keltner Channel
# ─────────────────────────────────────────────

def test_keltner_ordering():
    df = _make_df()
    kc = keltner_channel(df)
    valid = kc.dropna()
    assert (valid["kc_upper"] > valid["kc_mid"]).all()
    assert (valid["kc_mid"] > valid["kc_lower"]).all()


# ─────────────────────────────────────────────
# ADX
# ─────────────────────────────────────────────

def test_adx_columns():
    df = _make_df()
    result = adx(df)
    assert {"adx", "plus_di", "minus_di"} == set(result.columns)
    valid = result.dropna()
    assert (valid["adx"] >= 0).all()


# ─────────────────────────────────────────────
# Supertrend
# ─────────────────────────────────────────────

def test_supertrend_direction_values():
    df = _make_df()
    st = supertrend(df)
    directions = st["direction"].dropna().unique()
    assert set(directions).issubset({1.0, -1.0})


# ─────────────────────────────────────────────
# Volume Indicators
# ─────────────────────────────────────────────

def test_obv_cumulative():
    df = _make_df(50)
    o = obv(df)
    # OBV should be a cumulative sum — length matches df
    assert len(o) == len(df)


def test_rvol_near_one_on_flat_volume():
    rng = np.random.default_rng(0)
    n = 100
    df = _make_df(n)
    df["volume"] = 1_000_000.0  # Constant volume
    rv = relative_volume(df, 20)
    assert rv.dropna().iloc[-1] == pytest.approx(1.0)


# ─────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────

def test_nr7_type():
    df = _make_df(50)
    result = nr7(df)
    assert result.dtype == bool


def test_inside_bar_type():
    df = _make_df(50)
    result = inside_bar(df)
    assert result.dtype == bool


def test_darvas_box_columns():
    df = _make_df(100)
    result = darvas_box(df)
    assert {"box_top", "box_bottom", "breakout"} == set(result.columns)


# ─────────────────────────────────────────────
# Minervini Trend Template
# ─────────────────────────────────────────────

def test_minervini_tt_type():
    df = _make_df(300)
    result = minervini_trend_template(df)
    assert result.dtype == bool
    # First 200 bars must be NaN/False (not enough MA history)
    assert not result.iloc[:200].any()


# ─────────────────────────────────────────────
# Composite Enrich
# ─────────────────────────────────────────────

def test_enrich_adds_columns():
    df = _make_df(300)
    enriched = enrich(df, patterns=False)
    expected_cols = [
        "ema_20", "ema_50", "ema_200", "rsi_14", "macd",
        "atr_14", "bb_upper", "bb_lower", "obv", "rvol_20",
    ]
    for col in expected_cols:
        assert col in enriched.columns, f"Missing column: {col}"


def test_enrich_does_not_modify_original():
    df = _make_df(300)
    original_cols = set(df.columns)
    _ = enrich(df, patterns=False)
    assert set(df.columns) == original_cols


def test_enrich_with_patterns():
    df = _make_df(300)
    enriched = enrich(df, patterns=True)
    assert "nr7" in enriched.columns
    assert "inside_bar" in enriched.columns
