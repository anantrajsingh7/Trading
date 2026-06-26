"""
Momentum indicators.

RSI, MACD, Stochastic, ROC, Williams %R, CCI, MFI, CMF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Returns 0-100. RSI=100 when there are no down days."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    result = 100 - (100 / (1 + rs))
    return result.where(avg_loss != 0, 100.0)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD indicator.
    Returns DataFrame: macd, signal, histogram
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }, index=series.index)


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K and %D).
    Returns DataFrame: stoch_k, stoch_d
    """
    low_min  = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom    = (high_max - low_min).replace(0, np.nan)

    k_raw = 100 * (df["close"] - low_min) / denom
    stoch_k = k_raw.rolling(smooth_k).mean()
    stoch_d = stoch_k.rolling(d_period).mean()

    return pd.DataFrame({
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
    }, index=df.index)


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (%)."""
    return (series / series.shift(period) - 1) * 100


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R. Range: -100 to 0. Overbought: >-20, Oversold: <-80."""
    high_max = df["high"].rolling(period).max()
    low_min  = df["low"].rolling(period).min()
    denom    = (high_max - low_min).replace(0, np.nan)
    return -100 * (high_max - df["close"]) / denom


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (volume-weighted RSI)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]

    positive_mf = mf.where(tp > tp.shift(1), 0.0)
    negative_mf = mf.where(tp < tp.shift(1), 0.0)

    pos_sum = positive_mf.rolling(period).sum()
    neg_sum = negative_mf.rolling(period).sum()

    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow. Range: -1 to +1."""
    denom = (df["high"] - df["low"]).replace(0, np.nan)
    clv   = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / denom
    mf_vol = clv * df["volume"]
    vol_sum = df["volume"].rolling(period).sum().replace(0, np.nan)
    return mf_vol.rolling(period).sum() / vol_sum


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all momentum columns to DataFrame. Returns same DataFrame."""
    close = df["close"]

    df["rsi_14"] = rsi(close, 14)
    df["rsi_7"]  = rsi(close, 7)

    macd_df = macd(close)
    df["macd"]           = macd_df["macd"]
    df["macd_signal"]    = macd_df["signal"]
    df["macd_histogram"] = macd_df["histogram"]

    stoch_df = stochastic(df)
    df["stoch_k"] = stoch_df["stoch_k"]
    df["stoch_d"] = stoch_df["stoch_d"]

    df["roc_10"] = roc(close, 10)
    df["roc_20"] = roc(close, 20)
    df["williams_r"] = williams_r(df)
    df["cci_20"] = cci(df)
    df["mfi_14"] = mfi(df)
    df["cmf_20"] = cmf(df)

    return df
