"""
Trend indicators.

All functions accept a pd.Series (or DataFrame column) and return a pd.Series
unless otherwise noted. NaN-safe: leading NaN is expected during warm-up.

Implemented:
  ema, sma, wma, hma (Hull MA)
  adx (Average Directional Index)
  supertrend
  ichimoku (returns dict of Series)
  minervini_trend_template (boolean Series — all 8 conditions)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Moving Averages
# ──────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average — fast and smooth."""
    half = max(1, period // 2)
    sqrt_p = max(1, int(np.sqrt(period)))
    return wma(2 * wma(series, half) - wma(series, period), sqrt_p)


# ──────────────────────────────────────────────
# ADX / DMI
# ──────────────────────────────────────────────

def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index.
    Returns DataFrame with columns: adx, plus_di, minus_di
    """
    high = df["high"]
    low  = df["low"]
    close = df["close"]

    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    plus_dm  = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    # When both moves are positive, keep only the larger
    both_up   = (high - prev_high) > (prev_low - low)
    both_down = (prev_low - low)   > (high - prev_high)
    plus_dm  = plus_dm.where(both_up, 0.0)
    minus_dm = minus_dm.where(both_down, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_s   = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_s  = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_s = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di  = 100 * plus_s  / atr_s
    minus_di = 100 * minus_s / atr_s

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_line = dx.ewm(alpha=1 / period, adjust=False).mean()

    return pd.DataFrame({
        "adx": adx_line,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }, index=df.index)


# ──────────────────────────────────────────────
# Supertrend
# ──────────────────────────────────────────────

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Supertrend indicator.
    Returns DataFrame with columns: supertrend, direction (1=bullish, -1=bearish)
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    hl2 = (high + low) / 2

    # ATR via Wilder smoothing
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()

    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    # Iterative band adjustment
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    for i in range(1, len(df)):
        prev_upper = final_upper.iat[i - 1]
        prev_lower = final_lower.iat[i - 1]
        prev_close = close.iat[i - 1]

        final_upper.iat[i] = (
            min(upper_band.iat[i], prev_upper)
            if prev_close <= prev_upper else upper_band.iat[i]
        )
        final_lower.iat[i] = (
            max(lower_band.iat[i], prev_lower)
            if prev_close >= prev_lower else lower_band.iat[i]
        )

    # Direction: 1 = bullish, -1 = bearish
    direction = pd.Series(index=df.index, dtype=float)
    supertrend_line = pd.Series(index=df.index, dtype=float)

    for i in range(1, len(df)):
        prev_dir = direction.iat[i - 1] if i > 1 else 1.0
        c = close.iat[i]

        if prev_dir == 1:
            direction.iat[i] = 1 if c > final_lower.iat[i] else -1
        else:
            direction.iat[i] = -1 if c < final_upper.iat[i] else 1

        supertrend_line.iat[i] = (
            final_lower.iat[i] if direction.iat[i] == 1 else final_upper.iat[i]
        )

    return pd.DataFrame({
        "supertrend": supertrend_line,
        "direction": direction,
    }, index=df.index)


# ──────────────────────────────────────────────
# Ichimoku Cloud
# ──────────────────────────────────────────────

def ichimoku(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Ichimoku Kinko Hyo.
    Returns dict:
      tenkan, kijun, senkou_a, senkou_b, chikou
    senkou lines are shifted +26 bars (future cloud).
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    def mid(h: pd.Series, l: pd.Series, n: int) -> pd.Series:
        return (h.rolling(n).max() + l.rolling(n).min()) / 2

    tenkan  = mid(high, low, 9)
    kijun   = mid(high, low, 26)
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = mid(high, low, 52).shift(26)
    chikou  = close.shift(-26)

    return {
        "tenkan":   tenkan,
        "kijun":    kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou":   chikou,
    }


# ──────────────────────────────────────────────
# Minervini Trend Template
# ──────────────────────────────────────────────

def minervini_trend_template(df: pd.DataFrame) -> pd.Series:
    """
    Mark Minervini's 8-condition Trend Template.
    Returns a boolean Series (True = stock qualifies).

    Conditions:
    1. Price > 150-day SMA
    2. Price > 200-day SMA
    3. 150-day SMA > 200-day SMA
    4. 200-day SMA trending up for at least 1 month (20 days)
    5. 50-day SMA > 150-day SMA and > 200-day SMA
    6. Price > 50-day SMA
    7. Price at least 25% above 52-week low
    8. Price within 25% of 52-week high
    """
    close = df["close"]

    ma50  = sma(close, 50)
    ma150 = sma(close, 150)
    ma200 = sma(close, 200)

    high_52w = close.rolling(252).max()
    low_52w  = close.rolling(252).min()

    # Condition 4: 200-MA higher than it was 20 days ago
    ma200_trending_up = ma200 > ma200.shift(20)

    c1 = close > ma150
    c2 = close > ma200
    c3 = ma150 > ma200
    c4 = ma200_trending_up
    c5 = (ma50 > ma150) & (ma50 > ma200)
    c6 = close > ma50
    c7 = close >= 1.25 * low_52w   # ≥25% above 52w low
    c8 = close >= 0.75 * high_52w  # within 25% of 52w high

    return c1 & c2 & c3 & c4 & c5 & c6 & c7 & c8


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience: add all trend columns to a DataFrame in-place.
    Returns the same DataFrame with new columns added.
    """
    close = df["close"]

    for p in [5, 10, 20, 50, 100, 150, 200]:
        df[f"ema_{p}"] = ema(close, p)
        df[f"sma_{p}"] = sma(close, p)

    adx_df = adx(df)
    df["adx"]      = adx_df["adx"]
    df["plus_di"]  = adx_df["plus_di"]
    df["minus_di"] = adx_df["minus_di"]

    st = supertrend(df)
    df["supertrend"]   = st["supertrend"]
    df["st_direction"] = st["direction"]

    df["minervini_tt"] = minervini_trend_template(df)

    return df
