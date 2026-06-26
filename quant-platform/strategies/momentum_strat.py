"""
Dual Momentum / CANSLIM-inspired Momentum Strategy.

Combines:
  - 6-month price momentum (ROC_120)
  - RS rank vs universe (relative strength)
  - Earnings momentum proxy (price gap confirmation)
  - TTM Squeeze breakout

Buy when:
  1. 6-month ROC in top 30% of universe (caller injects rs_rank)
  2. TTM Squeeze fires (squeeze_on was True, momentum turns positive)
  3. RSI 50-80 (momentum zone, not exhausted)
  4. Volume > 1.3× average
"""
from __future__ import annotations

from typing import Optional
import uuid

import pandas as pd

from core.models import Signal, SignalStrength, Direction
from strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def description(self) -> str:
        return "Dual momentum + TTM squeeze breakout"

    def generate_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        cfg: dict,
        rs_rank: float = 50.0,  # Percentile vs universe (0-100)
    ) -> Optional[Signal]:
        if len(df) < 130:
            return None

        last  = self._last(df)
        close = float(last["close"])

        # ── Relative Strength ─────────────────────────────────────
        if rs_rank < 70:  # Must be in top 30%
            return None

        # ── Trend filter ──────────────────────────────────────────
        ema50  = last.get("ema_50", 0)
        ema200 = last.get("ema_200", 0)
        if close < ema50 or close < ema200:
            return None

        # ── TTM Squeeze fire ──────────────────────────────────────
        # squeeze_on was True in past 5 bars, now momentum turns positive
        if "squeeze_on" not in df.columns or "sq_momentum" not in df.columns:
            return None

        recent_squeeze = df["squeeze_on"].iloc[-10:-1].any()
        momentum_now   = float(last.get("sq_momentum", 0))
        momentum_prev  = float(df["sq_momentum"].iloc[-2]) if len(df) > 1 else 0

        squeeze_fire = recent_squeeze and momentum_now > 0 and momentum_prev <= 0

        # Also allow: momentum accelerating (second derivative positive)
        momentum_accel = (
            momentum_now > momentum_prev > float(df["sq_momentum"].iloc[-3])
            and momentum_now > 0
        )

        if not (squeeze_fire or momentum_accel):
            return None

        # ── RSI in momentum zone ──────────────────────────────────
        rsi = float(last.get("rsi_14", 50))
        if not (52 <= rsi <= 80):
            return None

        # ── Volume ───────────────────────────────────────────────
        rvol = float(last.get("rvol_20", 1.0))
        if rvol < 1.3:
            return None

        # ── Score ─────────────────────────────────────────────────
        score = 30
        score += min(int((rs_rank - 70) / 30 * 25), 25)  # Up to 25 for RS
        score += 20 if squeeze_fire else 10
        score += 10 if rvol > 1.5 else 5
        score += 10 if last.get("adx", 0) > 25 else 0
        score += 5  if last.get("macd", 0) > last.get("macd_signal", 0) else 0

        strength = (
            SignalStrength.STRONG    if score >= 80
            else SignalStrength.MODERATE if score >= 60
            else SignalStrength.WEAK
        )

        entry = close
        stop  = self._atr_stop(last, multiplier=2.0)
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
            notes=f"RS_rank={rs_rank:.0f} | {'Squeeze fire' if squeeze_fire else 'Momentum accel'} | RSI={rsi:.1f}",
        )
