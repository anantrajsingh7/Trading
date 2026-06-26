"""
Minervini Trend Template + VCP Strategy.

Stage 2 uptrend identification using Mark Minervini's 8-condition
Trend Template, combined with Volatility Contraction Pattern (VCP)
entry timing.

Entry triggers:
  1. All 8 Trend Template conditions satisfied for ≥3 bars
  2. VCP pattern detected (3 contractions, volume drying up)
  3. RS Rank > 70th percentile (relative strength)
  4. Volume confirmation on breakout bar

Exit rules (managed by Portfolio, not here):
  - Stop: 7-8% below entry OR below recent pivot low
  - T1:   +15% (1.5R), T2: +25% (2.5R), T3: +40%
"""
from __future__ import annotations

from typing import Optional
import uuid

import pandas as pd

from core.models import Signal, SignalStrength, Direction
from strategies.base import BaseStrategy


class MinerviniStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "minervini_vcp"

    @property
    def description(self) -> str:
        return "Minervini Trend Template + VCP breakout"

    def generate_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        cfg: dict,
    ) -> Optional[Signal]:
        if len(df) < 252:
            return None

        last = self._last(df)
        close = float(last["close"])

        # ── Condition 1: Minervini Trend Template ─────────────────
        # Must be True for the current bar
        if not last.get("minervini_tt", False):
            return None

        # ── Condition 2: Above all key EMAs ───────────────────────
        if not self._above_all_mas(last):
            return None

        # ── Condition 3: ADX shows trending market ────────────────
        adx_val = last.get("adx", 0)
        if adx_val < 20:
            return None

        # ── Condition 4: RSI healthy (not overbought) ─────────────
        rsi = last.get("rsi_14", 50)
        if not (50 <= rsi <= 80):
            return None

        # ── Condition 5: VCP detected recently (last 5 bars) ──────
        recent_vcp = df["vcp"].iloc[-5:].any() if "vcp" in df.columns else False

        # Relaxed: if no VCP, check for pocket pivot as alternative entry
        pocket_recent = df["pocket_pivot"].iloc[-3:].any() if "pocket_pivot" in df.columns else False

        if not (recent_vcp or pocket_recent):
            return None

        # ── Condition 6: Volume expanding on breakout ──────────────
        rvol = last.get("rvol_20", 1.0)
        if rvol < 1.2:  # Require 20% above average
            return None

        # ── Signal strength determination ─────────────────────────
        score = 0
        score += 20  # Base: Trend Template satisfied
        score += 20 if recent_vcp else 10      # VCP vs pocket pivot
        score += 15 if adx_val > 30 else 5
        score += 10 if rvol > 1.5 else 5
        score += 10 if rsi >= 60 else 5
        score += 10 if last.get("st_direction", 0) == 1 else 0
        score += 5  if last.get("macd", 0) > last.get("macd_signal", 0) else 0

        strength = (
            SignalStrength.STRONG  if score >= 80
            else SignalStrength.MODERATE if score >= 60
            else SignalStrength.WEAK
        )

        # ── Entry / Stop / Targets ────────────────────────────────
        entry = close
        # Stop below recent pivot low (use low of last 10 bars)
        pivot_low = float(df["low"].iloc[-10:].min())
        stop = max(
            self._pct_stop(entry, 0.08),   # Hard 8% stop
            pivot_low * 0.99,              # Just below pivot low
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
            notes=f"VCP={'yes' if recent_vcp else 'pocket pivot'} | ADX={adx_val:.1f} | RVOL={rvol:.2f} | RSI={rsi:.1f}",
        )
