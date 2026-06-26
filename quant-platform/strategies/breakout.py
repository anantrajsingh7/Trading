"""
Volume Breakout + Darvas Box Strategy.

Buys breakouts above multi-week resistance with volume confirmation.
Three sub-triggers:
  1. Darvas Box breakout (price breaks above box top)
  2. Flat Base breakout (IBD-style)
  3. Ascending Triangle breakout

All require volume ≥ 1.5× average to confirm institutional buying.
"""
from __future__ import annotations

from typing import Optional
import uuid

import pandas as pd

from core.models import Signal, SignalStrength, Direction
from strategies.base import BaseStrategy


class BreakoutStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "volume_breakout"

    @property
    def description(self) -> str:
        return "Volume breakout from Darvas Box / Flat Base / Ascending Triangle"

    def generate_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        cfg: dict,
    ) -> Optional[Signal]:
        if len(df) < 100:
            return None

        last  = self._last(df)
        close = float(last["close"])

        # Must be in a general uptrend (above 50 & 150 EMA)
        ema50  = last.get("ema_50", 0)
        ema150 = last.get("ema_150", 0)
        if close < ema50 or close < ema150:
            return None

        rvol = last.get("rvol_20", 1.0)
        if rvol < 1.5:
            return None

        # ── Check which pattern triggered ─────────────────────────
        darvas_break = bool(last.get("darvas_break", False))
        flat_break   = bool(last.get("flat_base", False))
        asc_break    = bool(last.get("asc_triangle", False))

        if not (darvas_break or flat_break or asc_break):
            return None

        # ── Score ─────────────────────────────────────────────────
        score = 40  # Base for confirmed pattern breakout
        score += 20 if darvas_break else 10 if flat_break else 5
        score += 15 if rvol > 2.0 else 10 if rvol > 1.5 else 0
        score += 10 if last.get("adx", 0) > 25 else 0
        score += 10 if last.get("rsi_14", 50) > 55 else 0
        score += 5  if last.get("macd", 0) > 0 else 0

        strength = (
            SignalStrength.STRONG    if score >= 80
            else SignalStrength.MODERATE if score >= 60
            else SignalStrength.WEAK
        )

        entry = close
        # Stop just below the base/box bottom
        box_bottom = last.get("darvas_bottom", close * 0.92)
        stop = max(
            self._pct_stop(entry, 0.07),
            float(box_bottom) * 0.99,
        )
        t1, t2, t3 = self._targets(entry, stop)

        trigger = (
            "Darvas breakout" if darvas_break
            else "Flat base breakout" if flat_break
            else "Ascending triangle breakout"
        )

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
            notes=f"{trigger} | RVOL={rvol:.2f} | RSI={last.get('rsi_14', 0):.1f}",
        )
