"""
Volatility indicators.

ATR, Bollinger Bands, Keltner Channel, Historical Volatility,
TTM Squeeze (BB inside KC detection).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.
    Returns DataFrame: bb_mid, bb_upper, bb_lower, bb_width, bb_pct_b
    """
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)

    return pd.DataFrame({
        "bb_mid":   mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct_b": pct_b,
    }, index=series.index)


def keltner_channel(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Keltner Channel.
    Returns DataFrame: kc_mid, kc_upper, kc_lower
    """
    close = df["close"]
    mid   = close.ewm(span=ema_period, adjust=False).mean()
    atr_v = atr(df, atr_period)
    upper = mid + multiplier * atr_v
    lower = mid - multiplier * atr_v

    return pd.DataFrame({
        "kc_mid":   mid,
        "kc_upper": upper,
        "kc_lower": lower,
    }, index=df.index)


def historical_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Annualised historical volatility (close-to-close log returns)."""
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(period).std(ddof=1) * np.sqrt(252) * 100


def ttm_squeeze(df: pd.DataFrame) -> pd.DataFrame:
    """
    TTM Squeeze by John Carter.
    squeeze_on = True when BB is inside KC (low volatility coiling).
    momentum column is the oscillator (positive = bullish).
    Returns DataFrame: squeeze_on, momentum
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    bb = bollinger_bands(close, 20, 2.0)
    kc = keltner_channel(df, 20, 10, 1.5)

    squeeze_on = (bb["bb_upper"] < kc["kc_upper"]) & (bb["bb_lower"] > kc["kc_lower"])

    # Momentum oscillator (simplified): delta of mid-point of close range vs EMA
    highest_high = high.rolling(20).max()
    lowest_low   = low.rolling(20).min()
    mid_hl = (highest_high + lowest_low) / 2
    mid_ema = close.ewm(span=20, adjust=False).mean()
    delta   = close - (mid_hl + mid_ema) / 2
    momentum = delta.ewm(span=5, adjust=False).mean()

    return pd.DataFrame({
        "squeeze_on": squeeze_on,
        "momentum":   momentum,
    }, index=df.index)


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all volatility columns to DataFrame. Returns same DataFrame."""
    close = df["close"]

    df["atr_14"] = atr(df, 14)
    df["atr_7"]  = atr(df, 7)

    # ATR % (normalised)
    df["atr_pct"] = df["atr_14"] / close * 100

    bb = bollinger_bands(close)
    df["bb_upper"] = bb["bb_upper"]
    df["bb_mid"]   = bb["bb_mid"]
    df["bb_lower"] = bb["bb_lower"]
    df["bb_width"] = bb["bb_width"]
    df["bb_pct_b"] = bb["bb_pct_b"]

    kc = keltner_channel(df)
    df["kc_upper"] = kc["kc_upper"]
    df["kc_mid"]   = kc["kc_mid"]
    df["kc_lower"] = kc["kc_lower"]

    df["hist_vol_20"] = historical_volatility(close, 20)

    sq = ttm_squeeze(df)
    df["squeeze_on"]  = sq["squeeze_on"]
    df["sq_momentum"] = sq["momentum"]

    return df
