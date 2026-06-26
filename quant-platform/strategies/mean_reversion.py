"""
Mean Reversion Strategy — NR7 / Inside Bar / Bollinger Squeeze.

For stocks that are temporarily oversold within a larger uptrend.
NOT a counter-trend fade strategy — we only buy dips in bull phases.

Entry conditions:
  1. NR7 or Inside Bar (coiling, low volatility)
  2. Price above 200 EMA (macro uptrend intact)
  3. RSI ≤ 40 (short-term oversold)
  4. Bollinger %B < 0.2 (near lower band)
  5. CMF > -0.1 (not under active distribution)

Exit: Mean reversion target = BB mid / 20-EMA
Stop: Below the NR7/Inside Bar low
"""
from __future__ import annotations

from typing import Optional
import uuid

import pandas as pd

from core.models import Signal, SignalStrength, Direction
from strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def description(self) -> str:
        return "NR7/Inside Bar oversold bounce within uptrend"

    def generate_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        cfg: dict,
    ) -> Optional[Signal]:
        if len(df) < 200:
            return None

        last  = self._last(df)
        close = float(last["close"])

        # ── Macro uptrend required ─────────────────────────────────
        ema200 = float(last.get("ema_200", 0))
        if close < ema200:
            return None

        # ── Coiling pattern ───────────────────────────────────────
        nr7_signal    = bool(last.get("nr7", False))
        inside_signal = bool(last.get("inside_bar", False))
        if not (nr7_signal or inside_signal):
            return None

        # ── Oversold indicators ───────────────────────────────────
        rsi    = float(last.get("rsi_14", 50))
        bb_pct = float(last.get("bb_pct_b", 0.5))
        cmf    = float(last.get("cmf_20", 0))

        if rsi > 45:
            return None

        if bb_pct > 0.3:
            return None

        if cmf < -0.15:  # Active distribution — skip
            return None

        # ── Volume drying up (non-distribution selling) ───────────
        rvol = float(last.get("rvol_20", 1.0))
        if rvol > 1.3:  # High volume pullback could be distribution
            return None

        # ── Score ─────────────────────────────────────────────────
        score = 30
        score += 15 if nr7_signal else 10
        score += 15 if rsi < 35 else 10 if rsi < 40 else 5
        score += 15 if bb_pct < 0.1 else 10 if bb_pct < 0.2 else 5
        score += 10 if cmf > 0 else 5 if cmf > -0.05 else 0
        score += 10 if last.get("adx", 0) > 20 else 0
        score += 5  if last.get("st_direction", 0) == 1 else 0

        strength = (
            SignalStrength.STRONG    if score >= 80
            else SignalStrength.MODERATE if score >= 60
            else SignalStrength.WEAK
        )

        entry    = close
        # Stop below the recent low (NR7 / inside bar low)
        bar_low  = float(last["low"])
        stop     = min(
            self._pct_stop(entry, 0.05),
            bar_low * 0.99,
        )

        # Mean reversion targets (shorter than trend-following)
        bb_mid  = float(last.get("bb_mid", close * 1.03))
        ema20   = float(last.get("ema_20", close * 1.02))
        t1      = max(bb_mid, ema20)           # Revert to mean
        t2      = float(last.get("ema_50", close * 1.05))
        t3      = float(last.get("ema_20", close * 1.08)) * 1.05

        return Signal(
            id=str(uuid.uuid4()),
            ticker=ticker,
            strategy=self.name,
            direction=Direction.LONG,
            strength=strength,
            entry_price=entry,
            stop_loss=stop,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            score=score,
            notes=f"{'NR7' if nr7_signal else 'Inside Bar'} | RSI={rsi:.1f} | BB%={bb_pct:.2f} | CMF={cmf:.2f}",
        )
