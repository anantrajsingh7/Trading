"""
EMA Pullback Strategy.

Buy when price pulls back to a rising EMA in a Stage 2 uptrend.
Three flavors:
  1. 20-EMA pullback (aggressive, shorter hold)
  2. 50-EMA pullback (standard swing)
  3. AVWAP pullback (institutional anchor level)

Filters:
  - Overall trend: close > EMA 150 & 200
  - EMA must be rising (slope > 0)
  - RSI oversold on pullback (RSI ≤ 45)
  - Volume drying up on pullback (confirmation of non-distribution)
"""
from __future__ import annotations

from typing import Optional
import uuid

import pandas as pd

from core.models import Signal, SignalStrength, Direction
from strategies.base import BaseStrategy


class EMApullbackStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "ema_pullback"

    @property
    def description(self) -> str:
        return "EMA 20/50 pullback in Stage 2 uptrend with RSI reset"

    def generate_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        cfg: dict,
    ) -> Optional[Signal]:
        if len(df) < 200:
            return None

        last  = self._last(df)
        prev  = self._prev(df, 1)
        close = float(last["close"])

        ema150 = float(last.get("ema_150", 0))
        ema200 = float(last.get("ema_200", 0))
        ema50  = float(last.get("ema_50",  0))
        ema20  = float(last.get("ema_20",  0))

        # Must be in Stage 2 (above 150 and 200 EMA)
        if close < ema150 or close < ema200:
            return None

        # 200-EMA must be rising
        ema200_prev = float(df["ema_200"].iloc[-20])
        if ema200 <= ema200_prev:
            return None

        rsi   = float(last.get("rsi_14", 50))
        rvol  = float(last.get("rvol_20", 1.0))

        # ── Identify pullback type ─────────────────────────────────
        ema_target = None
        ema_label  = ""

        # Touch 20-EMA (price crossed below and now reclaiming, or touching)
        low_near_ema20 = (
            float(df["low"].iloc[-3:].min()) <= ema20 * 1.01
            and close > ema20
        )
        low_near_ema50 = (
            float(df["low"].iloc[-5:].min()) <= ema50 * 1.02
            and close > ema50
        )

        if low_near_ema20 and ema20 > ema50 > ema150:
            ema_target = ema20
            ema_label  = "20-EMA"
        elif low_near_ema50:
            ema_target = ema50
            ema_label  = "50-EMA"
        else:
            return None

        # RSI should be reset (pulled back from overbought)
        if rsi > 55:
            return None

        # Volume should be contracting on pullback (weak selling)
        if rvol > 1.5:
            return None

        # ── Confirm bounce: today closes above the EMA ─────────────
        if close <= ema_target:
            return None

        # Check that price is recovering (higher close than prev bar)
        if close <= float(prev["close"]):
            return None

        # ── Score ──────────────────────────────────────────────────
        score = 40
        score += 20 if ema_label == "20-EMA" else 15
        score += 15 if last.get("minervini_tt", False) else 0
        score += 10 if rsi < 45 else 5
        score += 10 if last.get("adx", 0) > 25 else 0
        score += 5  if last.get("st_direction", 0) == 1 else 0

        strength = (
            SignalStrength.STRONG    if score >= 80
            else SignalStrength.MODERATE if score >= 60
            else SignalStrength.WEAK
        )

        entry = close
        stop  = max(
            self._pct_stop(entry, 0.06),
            ema_target * 0.97,
        )
        t1, t2, t3 = self._targets(entry, stop)

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
            notes=f"{ema_label} pullback | RSI={rsi:.1f} | RVOL={rvol:.2f}",
        )
