"""
Strategy signal generation.
Each strategy function receives a DataFrame with pre-computed indicators
and returns a boolean Series: True on days where an entry signal fires.
All signals are forward-safe (no look-ahead): signal on row i uses data
from rows <= i only.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: swing-low lookback for stop-loss placement
# ---------------------------------------------------------------------------

def _recent_swing_low(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Rolling minimum low over recent bars (proxy for swing low)."""
    return df["low"].rolling(lookback, min_periods=2).min()


# ---------------------------------------------------------------------------
# Strategy 1: EMA Trend Pullback
# ---------------------------------------------------------------------------

def ema_pullback_signals(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Rules:
    - Close above EMA200
    - EMA20 > EMA50 > EMA200
    - Price pulls back within 3% of EMA20 or EMA50
    - RSI between 40 and 60
    - Entry when today's close > previous day's high
    """
    ind = cfg.get("indicators", {})
    close = df["close"]
    prev_high = df["high"].shift(1)

    trend_ok = (
        (close > df["ema200"]) &
        (df["ema20"] > df["ema50"]) &
        (df["ema50"] > df["ema200"])
    )
    # pullback: price within 3% above EMA20 or EMA50
    near_ema20 = (close >= df["ema20"] * 0.97) & (close <= df["ema20"] * 1.03)
    near_ema50 = (close >= df["ema50"] * 0.97) & (close <= df["ema50"] * 1.05)
    pullback = near_ema20 | near_ema50

    rsi_ok = (df["rsi"] >= 40) & (df["rsi"] <= 60)
    entry_trigger = close > prev_high

    signal = trend_ok & pullback & rsi_ok & entry_trigger
    return signal.fillna(False)


# ---------------------------------------------------------------------------
# Strategy 2: Breakout
# ---------------------------------------------------------------------------

def breakout_signals(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Rules:
    - Close breaks above 20-day or 50-day resistance (prior day's rolling high)
    - Volume > 1.5x 20-day avg volume
    - Close above EMA50 and EMA200
    - Not more than 12% above EMA50
    """
    close = df["close"]
    # use prior-day resistance to avoid look-ahead
    prev_res20 = df["res20"].shift(1)
    prev_res50 = df["res50"].shift(1)

    breaks_res = (close > prev_res20) | (close > prev_res50)
    volume_ok = df["volume"] > df["avg_volume"] * 1.5
    above_ema = (close > df["ema50"]) & (close > df["ema200"])
    not_extended = close <= df["ema50"] * 1.12

    signal = breaks_res & volume_ok & above_ema & not_extended
    return signal.fillna(False)


# ---------------------------------------------------------------------------
# Strategy 3: Momentum Continuation
# ---------------------------------------------------------------------------

def momentum_signals(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Rules:
    - Close above EMA20, EMA50, EMA200
    - Within 10% of 52-week high
    - RSI between 50 and 70
    - RS vs SPY and QQQ positive
    - Volume above 20-day avg
    """
    close = df["close"]

    trend_ok = (close > df["ema20"]) & (close > df["ema50"]) & (close > df["ema200"])
    near_52wh = close >= df["high52w"] * 0.90
    rsi_ok = (df["rsi"] >= 50) & (df["rsi"] <= 70)
    rs_ok = (df["rs_spy"] > 0) & (df["rs_qqq"] > 0)
    vol_ok = df["volume"] > df["avg_volume"]

    signal = trend_ok & near_52wh & rsi_ok & rs_ok & vol_ok
    return signal.fillna(False)


# ---------------------------------------------------------------------------
# Strategy 4: Mean Reversion in Uptrend
# ---------------------------------------------------------------------------

def mean_reversion_signals(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Rules:
    - Close above EMA200
    - Price pulls back near EMA50 (within 5%)
    - RSI recovers: yesterday < 45, today >= 50
    """
    close = df["close"]
    prev_rsi = df["rsi"].shift(1)

    above_200 = close > df["ema200"]
    near_ema50 = (close >= df["ema50"] * 0.95) & (close <= df["ema50"] * 1.05)
    rsi_recovery = (prev_rsi < 45) & (df["rsi"] >= 50)

    signal = above_200 & near_ema50 & rsi_recovery
    return signal.fillna(False)


# ---------------------------------------------------------------------------
# Strategy 5: Volatility Contraction Breakout (VCP)
# ---------------------------------------------------------------------------

def volatility_contraction_signals(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Rules:
    - Above EMA50 and EMA200
    - ATR contracted: current ATR < 80% of 20-day ATR average
    - Price consolidates in tight range for at least 5 days
    - Entry on breakout above consolidation high
    - Volume > 1.3x avg
    """
    ind = cfg.get("indicators", {})
    consol_days = int(ind.get("consolidation_days", 5))

    close = df["close"]
    above_ema = (close > df["ema50"]) & (close > df["ema200"])
    atr_contracted = df["atr"] < df["atr20_avg"] * 0.80

    # tight range: rolling high - rolling low over consolidation window
    consol_high = df["high"].rolling(consol_days, min_periods=consol_days).max()
    consol_low = df["low"].rolling(consol_days, min_periods=consol_days).min()
    tight_range = (consol_high - consol_low) / consol_low < 0.06  # within 6%

    # breakout: today's close > yesterday's consolidation high
    prev_consol_high = consol_high.shift(1)
    breakout = close > prev_consol_high

    vol_ok = df["volume"] > df["avg_volume"] * 1.3

    signal = above_ema & atr_contracted & tight_range & breakout & vol_ok
    return signal.fillna(False)


# ---------------------------------------------------------------------------
# Signal dispatch
# ---------------------------------------------------------------------------

STRATEGIES = {
    "ema_pullback": ema_pullback_signals,
    "breakout": breakout_signals,
    "momentum": momentum_signals,
    "mean_reversion": mean_reversion_signals,
    "volatility_contraction": volatility_contraction_signals,
}

STRATEGY_NAMES = {
    "ema_pullback": "EMA Trend Pullback",
    "breakout": "Breakout",
    "momentum": "Momentum Continuation",
    "mean_reversion": "Mean Reversion in Uptrend",
    "volatility_contraction": "Volatility Contraction Breakout",
}


def get_signals(df: pd.DataFrame, strategy_key: str, cfg: dict) -> pd.Series:
    func = STRATEGIES.get(strategy_key)
    if func is None:
        raise ValueError(f"Unknown strategy: {strategy_key}")
    return func(df, cfg)
