"""
Composite indicator builder.

Single call: enrich(df) returns a DataFrame with all 60+ indicator columns
added in the correct dependency order. This is what every strategy module
imports instead of calling individual indicator files.
"""
from __future__ import annotations

import pandas as pd

from indicators.trend      import add_trend_indicators
from indicators.momentum   import add_momentum_indicators
from indicators.volatility import add_volatility_indicators
from indicators.volume     import add_volume_indicators
from indicators.patterns   import add_pattern_indicators
from core.logging import get_logger

log = get_logger("indicators.composite")

_MIN_ROWS_REQUIRED = 252  # Need 1yr+ for reliable 200-day indicators


def enrich(df: pd.DataFrame, patterns: bool = True) -> pd.DataFrame:
    """
    Add all technical indicators to a copy of df.
    Returns enriched DataFrame. Original is not modified.

    patterns=False skips computationally expensive pattern detection
    (useful for backtesting large universes on non-pattern strategies).
    """
    if df is None or len(df) < 60:
        return df

    df = df.copy()

    try:
        df = add_trend_indicators(df)
    except Exception as e:
        log.debug("Trend indicators failed: %s", e)

    try:
        df = add_momentum_indicators(df)
    except Exception as e:
        log.debug("Momentum indicators failed: %s", e)

    try:
        df = add_volatility_indicators(df)
    except Exception as e:
        log.debug("Volatility indicators failed: %s", e)

    try:
        df = add_volume_indicators(df)
    except Exception as e:
        log.debug("Volume indicators failed: %s", e)

    if patterns:
        try:
            df = add_pattern_indicators(df)
        except Exception as e:
            log.debug("Pattern indicators failed: %s", e)

    return df


def latest(df: pd.DataFrame) -> pd.Series:
    """Return the last row of an enriched DataFrame as a Series."""
    return df.iloc[-1]


def enough_history(df: pd.DataFrame) -> bool:
    """True if df has enough bars for 200-day indicators to be non-NaN."""
    return df is not None and len(df) >= _MIN_ROWS_REQUIRED
