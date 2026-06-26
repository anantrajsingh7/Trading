"""
Volume indicators.

OBV, Relative Volume, VWAP, AVWAP, Accumulation/Distribution,
Chaikin Oscillator, Volume-Weighted RSI (not MFI — separate from momentum.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff())
    direction.iloc[0] = 0
    return (direction * df["volume"]).cumsum()


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    RVOL: today's volume relative to average of past `period` days.
    >1.5 = elevated, >2.0 = very high. Pocket Pivot needs RVOL > 1.
    """
    avg_vol = df["volume"].rolling(period).mean()
    return df["volume"] / avg_vol.replace(0, np.nan)


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Rolling VWAP (session-agnostic, 1-day lookback approximation for daily bars).
    For intraday use, reset at session open externally.
    For daily bars this is cumulative from first row — use avwap for anchored.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tpv = (tp * df["volume"]).cumsum()
    cum_vol  = df["volume"].cumsum()
    return cum_tpv / cum_vol.replace(0, np.nan)


def avwap(df: pd.DataFrame, anchor_idx: int = 0) -> pd.Series:
    """
    Anchored VWAP from a specific bar index.
    anchor_idx = 0 means anchor at first row.
    For 52-week low anchor, pass the index position of the low.

    Returns Series aligned to df.index; NaN before anchor.
    """
    if anchor_idx < 0 or anchor_idx >= len(df):
        return pd.Series(np.nan, index=df.index)

    sub = df.iloc[anchor_idx:]
    tp  = (sub["high"] + sub["low"] + sub["close"]) / 3
    cum_tpv = (tp * sub["volume"]).cumsum()
    cum_vol  = sub["volume"].cumsum()
    result = cum_tpv / cum_vol.replace(0, np.nan)
    return result.reindex(df.index)


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line."""
    denom = (df["high"] - df["low"]).replace(0, np.nan)
    clv   = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / denom
    return (clv * df["volume"]).cumsum()


def chaikin_oscillator(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.Series:
    """Chaikin Oscillator = EMA(fast) - EMA(slow) of A/D line."""
    ad = accumulation_distribution(df)
    return (
        ad.ewm(span=fast, adjust=False).mean()
        - ad.ewm(span=slow, adjust=False).mean()
    )


def volume_sma(df: pd.DataFrame, period: int = 50) -> pd.Series:
    return df["volume"].rolling(period).mean()


def pocket_pivot(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Pocket Pivot Volume (Gil Morales / Chris Kacher).
    True when today's up-day volume exceeds the highest down-day volume
    in the past `period` bars.
    Price must close above its 10-day EMA (caller can enforce separately).
    Returns boolean Series.
    """
    close = df["close"]
    vol   = df["volume"]

    is_up   = close > close.shift(1)
    is_down = ~is_up

    # Highest down-day volume over lookback
    down_vol = vol.where(is_down, 0.0)
    max_down_vol = down_vol.rolling(period).max()

    return is_up & (vol > max_down_vol)


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all volume columns to DataFrame. Returns same DataFrame."""
    df["obv"]       = obv(df)
    df["rvol_20"]   = relative_volume(df, 20)
    df["rvol_10"]   = relative_volume(df, 10)
    df["vwap"]      = vwap(df)
    df["ad_line"]   = accumulation_distribution(df)
    df["chaikin"]   = chaikin_oscillator(df)
    df["vol_sma_50"] = volume_sma(df, 50)
    df["vol_sma_20"] = volume_sma(df, 20)
    df["pocket_pivot"] = pocket_pivot(df)
    return df
