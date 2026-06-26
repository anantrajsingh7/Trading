"""
Market Regime Classification.

Uses SPY (US market proxy) to determine whether it's safe to deploy capital.
The regime gates the scanner — in Bear Strong, no new longs.

Five regimes:
  BULL_STRONG    — Full deployment, all strategies active
  BULL_MODERATE  — Selective, prefer high-quality setups
  NEUTRAL        — Reduced size (50%), only highest-score signals
  BEAR_MODERATE  — Defensive, cash-heavy, very few longs
  BEAR_STRONG    — Full cash, no new longs
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from core.models import MarketRegime, RegimeSnapshot
from indicators.trend    import ema, sma
from indicators.momentum import rsi
from indicators.volatility import atr
from core.logging import get_logger

log = get_logger("scanner.market_regime")

# VIX thresholds
_VIX_ELEVATED = 20.0
_VIX_HIGH     = 30.0
_VIX_EXTREME  = 40.0


def classify_regime(
    spy_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
) -> RegimeSnapshot:
    """
    Classify current market regime from SPY data.

    spy_df — enriched SPY DataFrame (indicators already computed)
    vix_df — VIX DataFrame (optional; uses ATR proxy if absent)
    """
    if spy_df is None or len(spy_df) < 200:
        return RegimeSnapshot(
            date=date.today(),
            regime=MarketRegime.NEUTRAL,
            spy_trend="UNKNOWN",
            spy_price=0.0,
            spy_vs_ema50=0.0,
            spy_vs_ema200=0.0,
            vix=20.0,
            breadth_pct_above_200=0.5,
            advance_decline=0.0,
            notes="Insufficient data",
        )

    last = spy_df.iloc[-1]
    close  = float(last["close"])
    ema50  = float(last.get("ema_50",  close))
    ema200 = float(last.get("ema_200", close))
    rsi_v  = float(last.get("rsi_14", 50))
    adx_v  = float(last.get("adx", 20))

    vs_ema50  = (close / ema50  - 1) * 100
    vs_ema200 = (close / ema200 - 1) * 100

    # 200-EMA slope (20-bar)
    ema200_20ago = float(spy_df["ema_200"].iloc[-21]) if len(spy_df) > 21 else ema200
    ema200_rising = ema200 > ema200_20ago

    # VIX (use ATR% as proxy if no VIX data)
    if vix_df is not None and not vix_df.empty:
        vix_level = float(vix_df["close"].iloc[-1])
    else:
        atr_pct = float(last.get("atr_pct", 1.0))
        vix_level = atr_pct * 10  # Rough approximation

    # ── Regime classification logic ────────────────────────────────
    # Primary: price vs EMAs
    above_ema50  = close > ema50
    above_ema200 = close > ema200

    # Secondary: momentum
    bull_momentum = rsi_v > 50 and adx_v > 20

    # Volatility stress
    stress = vix_level >= _VIX_HIGH

    if above_ema50 and above_ema200 and ema200_rising and not stress:
        regime = MarketRegime.BULL_STRONG if bull_momentum else MarketRegime.BULL_MODERATE
        trend  = "UPTREND"

    elif above_ema200 and not stress:
        regime = MarketRegime.BULL_MODERATE
        trend  = "UPTREND"

    elif not above_ema200 and above_ema50:
        regime = MarketRegime.NEUTRAL
        trend  = "CHOPPY"

    elif not above_ema50 and not above_ema200:
        if vix_level >= _VIX_EXTREME or vs_ema200 < -15:
            regime = MarketRegime.BEAR_STRONG
            trend  = "DOWNTREND"
        else:
            regime = MarketRegime.BEAR_MODERATE
            trend  = "DOWNTREND"

    else:
        regime = MarketRegime.NEUTRAL
        trend  = "CHOPPY"

    notes = (
        f"SPY {close:.2f} | EMA50 {vs_ema50:+.1f}% | "
        f"EMA200 {vs_ema200:+.1f}% | VIX~{vix_level:.1f} | RSI {rsi_v:.0f}"
    )

    log.info("Market regime: %s (%s)", regime.value, notes)

    return RegimeSnapshot(
        date=date.today(),
        regime=regime,
        spy_trend=trend,
        spy_price=close,
        spy_vs_ema50=vs_ema50,
        spy_vs_ema200=vs_ema200,
        vix=vix_level,
        breadth_pct_above_200=0.0,   # Populated by scanner with full universe data
        advance_decline=0.0,
        notes=notes,
    )


def regime_position_scale(regime: MarketRegime) -> float:
    """
    Return a scaling factor (0.0-1.0) for position sizing based on regime.
    Applied to normal risk per trade.
    """
    return {
        MarketRegime.BULL_STRONG:   1.00,
        MarketRegime.BULL_MODERATE: 0.75,
        MarketRegime.NEUTRAL:       0.50,
        MarketRegime.BEAR_MODERATE: 0.25,
        MarketRegime.BEAR_STRONG:   0.00,
    }[regime]


def regime_max_positions(regime: MarketRegime, configured_max: int = 10) -> int:
    """Max open positions allowed given current regime."""
    scales = {
        MarketRegime.BULL_STRONG:   1.0,
        MarketRegime.BULL_MODERATE: 0.8,
        MarketRegime.NEUTRAL:       0.5,
        MarketRegime.BEAR_MODERATE: 0.3,
        MarketRegime.BEAR_STRONG:   0.0,
    }
    return max(0, int(configured_max * scales[regime]))
