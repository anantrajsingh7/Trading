"""
Chart pattern detectors.

Each function returns a boolean Series (True = pattern detected on that bar)
or a structured result dict where the pattern requires multiple output fields.

Implemented:
  vcp          — Volatility Contraction Pattern (Minervini)
  darvas_box   — Darvas Box breakout
  nr7          — Narrowest Range in 7 bars
  inside_bar   — Inside Bar (mother bar / inside bar pair)
  flat_base    — Flat Base (IBD-style)
  cup_handle   — Cup & Handle (simplified geometric detection)
  ascending_triangle — Flat resistance + rising support
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# VCP — Volatility Contraction Pattern
# ─────────────────────────────────────────────

def vcp(
    df: pd.DataFrame,
    lookback: int = 60,
    min_contractions: int = 3,
    depth_decay: float = 0.6,   # Each contraction must be ≤60% of previous
    vol_contraction: float = 0.5,
) -> pd.Series:
    """
    Volatility Contraction Pattern (Mark Minervini).
    Looks for 3+ progressively smaller pullbacks with contracting volume.

    Returns boolean Series — True on the bar where VCP conditions are met
    and price is near the pivot high.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    result = pd.Series(False, index=df.index)

    for i in range(lookback, len(df)):
        window_c   = close.iloc[i - lookback: i + 1]
        window_h   = high.iloc[i - lookback: i + 1]
        window_l   = low.iloc[i - lookback: i + 1]
        window_v   = volume.iloc[i - lookback: i + 1]

        # Find local peaks (pivot highs) in window
        highs  = window_h.values
        lows   = window_l.values
        closes = window_c.values
        vols   = window_v.values

        # Simple contraction: measure range of each third of the window
        n  = len(highs)
        t1 = slice(0, n // 3)
        t2 = slice(n // 3, 2 * n // 3)
        t3 = slice(2 * n // 3, n)

        r1 = (highs[t1].max() - lows[t1].min()) / closes[t1].mean()
        r2 = (highs[t2].max() - lows[t2].min()) / closes[t2].mean()
        r3 = (highs[t3].max() - lows[t3].min()) / closes[t3].mean()

        v1 = vols[t1].mean()
        v3 = vols[t3].mean()

        # Contracting ranges and drying-up volume
        ranges_contracting = (r2 <= r1 * depth_decay) and (r3 <= r2 * depth_decay)
        volume_drying      = v3 <= v1 * vol_contraction

        # Price near the 60-day high (pivot)
        pivot = highs.max()
        near_pivot = closes[-1] >= pivot * 0.97

        if ranges_contracting and volume_drying and near_pivot:
            result.iloc[i] = True

    return result


# ─────────────────────────────────────────────
# Darvas Box
# ─────────────────────────────────────────────

def darvas_box(df: pd.DataFrame, box_period: int = 3) -> pd.DataFrame:
    """
    Darvas Box — identifies box top and bottom.
    A new box starts when price breaks above the previous box top.

    Returns DataFrame: box_top, box_bottom, breakout (bool)
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    n     = len(df)

    box_top    = pd.Series(np.nan, index=df.index)
    box_bottom = pd.Series(np.nan, index=df.index)
    breakout   = pd.Series(False, index=df.index)

    current_top    = high.iloc[0]
    current_bottom = low.iloc[0]
    confirm_top    = 0
    confirm_bot    = 0

    for i in range(1, n):
        h = high.iloc[i]
        l = low.iloc[i]
        c = close.iloc[i]

        if h > current_top:
            current_top = h
            confirm_top = 0
        else:
            confirm_top += 1

        if l < current_bottom:
            current_bottom = l
            confirm_bot = 0
        else:
            confirm_bot += 1

        if confirm_top >= box_period and confirm_bot >= box_period:
            box_top.iloc[i]    = current_top
            box_bottom.iloc[i] = current_bottom

            if c > current_top and i > 0 and not pd.isna(box_top.iloc[i - 1]):
                breakout.iloc[i] = True

    # Forward-fill box boundaries between breakouts
    box_top    = box_top.ffill()
    box_bottom = box_bottom.ffill()

    return pd.DataFrame({
        "box_top":    box_top,
        "box_bottom": box_bottom,
        "breakout":   breakout,
    }, index=df.index)


# ─────────────────────────────────────────────
# NR7 — Narrowest Range in 7 Bars
# ─────────────────────────────────────────────

def nr7(df: pd.DataFrame) -> pd.Series:
    """
    NR7 — today's High-Low range is the narrowest of the last 7 bars.
    Often precedes a sharp directional move.
    Returns boolean Series.
    """
    daily_range = df["high"] - df["low"]
    min_range_7 = daily_range.rolling(7).min()
    return daily_range == min_range_7


def nr4(df: pd.DataFrame) -> pd.Series:
    daily_range = df["high"] - df["low"]
    min_range_4 = daily_range.rolling(4).min()
    return daily_range == min_range_4


# ─────────────────────────────────────────────
# Inside Bar
# ─────────────────────────────────────────────

def inside_bar(df: pd.DataFrame) -> pd.Series:
    """
    Inside Bar — today's high < yesterday's high AND today's low > yesterday's low.
    Returns boolean Series (True on the inside bar).
    """
    return (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))


# ─────────────────────────────────────────────
# Flat Base
# ─────────────────────────────────────────────

def flat_base(df: pd.DataFrame, period: int = 25, max_depth_pct: float = 0.15) -> pd.Series:
    """
    IBD Flat Base — price consolidates within 15% for ~5 weeks (25 days).
    Returns boolean Series — True when today completes a valid flat base.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    result = pd.Series(False, index=df.index)

    for i in range(period, len(df)):
        window_h = high.iloc[i - period: i + 1]
        window_l = low.iloc[i - period: i + 1]
        window_c = close.iloc[i - period: i + 1]

        base_high = window_h.max()
        base_low  = window_l.min()
        depth     = (base_high - base_low) / base_high

        if depth <= max_depth_pct:
            # Price should be near top of base (within 5%)
            current = close.iloc[i]
            if current >= base_high * 0.95:
                result.iloc[i] = True

    return result


# ─────────────────────────────────────────────
# Cup & Handle (simplified)
# ─────────────────────────────────────────────

def cup_handle(
    df: pd.DataFrame,
    cup_period: int = 60,
    handle_period: int = 15,
    max_depth_pct: float = 0.35,
) -> pd.Series:
    """
    Cup & Handle (William O'Neil).
    Cup: U-shaped pullback 12-35% over 7-65 weeks.
    Handle: tight consolidation <12% over 1-2 weeks in upper half of cup.
    Returns boolean Series — True when breakout from handle is detected.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    result = pd.Series(False, index=df.index)

    min_period = cup_period + handle_period
    for i in range(min_period, len(df)):
        # Cup phase
        cup_h = high.iloc[i - min_period: i - handle_period + 1]
        cup_l = low.iloc[i - min_period: i - handle_period + 1]

        cup_high  = cup_h.max()
        cup_low   = cup_l.min()
        cup_depth = (cup_high - cup_low) / cup_high

        if not (0.12 <= cup_depth <= max_depth_pct):
            continue

        # Right side of cup should recover to ~cup_high
        rs_close = close.iloc[i - handle_period]
        if rs_close < cup_high * 0.90:
            continue

        # Handle phase
        hdl_h = high.iloc[i - handle_period: i + 1]
        hdl_l = low.iloc[i - handle_period: i + 1]
        hdl_depth = (hdl_h.max() - hdl_l.min()) / hdl_h.max()

        if hdl_depth > 0.12:
            continue

        # Handle must form in upper half of cup
        if hdl_l.min() < cup_high - (cup_high - cup_low) * 0.5:
            continue

        # Breakout: today breaks above handle high
        if close.iloc[i] > hdl_h.max():
            result.iloc[i] = True

    return result


# ─────────────────────────────────────────────
# Ascending Triangle
# ─────────────────────────────────────────────

def ascending_triangle(
    df: pd.DataFrame,
    period: int = 30,
    resistance_tolerance: float = 0.02,
    min_touches: int = 2,
) -> pd.Series:
    """
    Ascending Triangle — flat resistance + rising support.
    Returns boolean Series — True at breakout above resistance.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    result = pd.Series(False, index=df.index)

    for i in range(period, len(df)):
        window_h = high.iloc[i - period: i + 1]
        window_l = low.iloc[i - period: i + 1]

        resistance = window_h.max()

        # Count touches near resistance
        near_res = (window_h >= resistance * (1 - resistance_tolerance)).sum()
        if near_res < min_touches:
            continue

        # Support should be rising — compare first-half vs second-half lows
        half = period // 2
        first_low  = window_l.iloc[:half].min()
        second_low = window_l.iloc[half:].min()
        if second_low <= first_low:
            continue

        # Breakout
        if close.iloc[i] > resistance:
            result.iloc[i] = True

    return result


# ─────────────────────────────────────────────
# Composite helper
# ─────────────────────────────────────────────

def add_pattern_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add pattern detection columns. Returns same DataFrame."""
    df["nr7"]         = nr7(df)
    df["nr4"]         = nr4(df)
    df["inside_bar"]  = inside_bar(df)
    df["vcp"]         = vcp(df)
    df["flat_base"]   = flat_base(df)
    df["cup_handle"]  = cup_handle(df)
    df["asc_triangle"] = ascending_triangle(df)

    darvas = darvas_box(df)
    df["darvas_top"]    = darvas["box_top"]
    df["darvas_bottom"] = darvas["box_bottom"]
    df["darvas_break"]  = darvas["breakout"]

    return df
