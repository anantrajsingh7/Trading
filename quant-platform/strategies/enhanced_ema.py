"""
Enhanced EMA Pullback — RESEARCH strategy (not in the production scan list).

Rules (see research/enhanced_ema_research.py for the portfolio-level parts:
market-regime gate, breadth, RS>=80, earnings blackout, sector/heat limits —
those need cross-sectional data a per-ticker strategy cannot see):

Per-ticker structure (this file):
  Trend:    close > SMA50 > SMA150 > SMA200, SMA200 rising, price > $10,
            avg dollar volume >= $20M
  Pullback: a 20-60 day high, then a 3-10% pullback toward EMA20/EMA50 on
            below-normal volume, RSI reset into ~40-60, no decisive close
            below EMA50, no large bearish gap/distribution candle
  Entry confirmation (two modes, tested separately):
    "reversal" — bullish reversal candle after the pullback; entry is a
                 buy-stop above that candle's high, filled on a LATER day
    "reclaim"  — close back above EMA20 after a valid pullback; entry at
                 the next session (never at the signalling close)
  Initial stop: pullback swing low - buffer*ATR (buffer 0/0.25/0.5 tested);
                reject if stop distance > 7%. Never widened.

All computations in compute_setup_table() are causal (rolling/shift only),
so the same table drives both live scanning and bar-by-bar backtesting
without lookahead.
"""
from __future__ import annotations

from typing import Optional
import uuid

import numpy as np
import pandas as pd

from core.models import Signal, SignalStrength, Direction
from strategies.base import BaseStrategy

MIN_PRICE = 10.0
MIN_DOLLAR_VOL = 20_000_000
PULLBACK_MIN = 0.03
PULLBACK_MAX = 0.10
MAX_STOP_PCT = 0.07


def compute_setup_table(
    df: pd.DataFrame,
    stop_atr_buffer: float = 0.25,
    entry_mode: str = "reversal",
) -> pd.DataFrame:
    """
    Return a causal per-bar table with columns:
      setup_ok       — all structural+pullback conditions true on this bar
      trigger        — buy-stop price valid for LATER sessions (NaN if none)
      init_stop      — initial stop if entered off this bar's setup
      reject_reason  — first failed structural gate ('' if setup_ok)
    Only rolling / shifted data is used — row t never sees rows > t.
    """
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))

    sma50  = c.rolling(50).mean()
    sma150 = c.rolling(150).mean()
    sma200 = c.rolling(200).mean()
    sma200_rising = sma200 > sma200.shift(20)

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    # ATR(14) — Wilder
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    dollar_vol = (c * v).rolling(20).mean()
    vol_avg20 = v.rolling(20).mean()

    # Recent swing high (20-60d) and pullback depth from it
    swing_high = h.shift(1).rolling(60).max()
    high_20 = h.shift(1).rolling(20).max()
    made_recent_high = high_20 >= swing_high * 0.98   # the high is recent, not 3 months old
    pullback_depth = 1 - c / swing_high

    # Pullback toward EMA20/EMA50 (low within 1.5% of either)
    near_ema = (l <= ema20 * 1.015) | (l <= ema50 * 1.015)

    # Pullback volume below normal (avg of last 3 bars vs 20d avg)
    pullback_vol_low = v.rolling(3).mean() < vol_avg20

    # No decisive close below EMA50 (>2% below on a close, last 5 bars)
    below_ema50 = (c < ema50 * 0.98).rolling(5).max().astype(bool)

    # Distribution candle: big red bar on heavy volume, or big bearish gap
    big_red = ((o - c) / o > 0.04) & (v > 2 * vol_avg20)
    big_gap = (o / c.shift() - 1) < -0.05
    distribution = (big_red | big_gap).rolling(5).max().astype(bool)

    # Structural trend gate
    trend_ok = (
        (c > sma50) & (sma50 > sma150) & (sma150 > sma200) & sma200_rising
        & (c > MIN_PRICE) & (dollar_vol >= MIN_DOLLAR_VOL)
    )

    pullback_ok = (
        made_recent_high
        & pullback_depth.between(PULLBACK_MIN, PULLBACK_MAX)
        & near_ema
        & pullback_vol_low
        & rsi.between(35, 62)          # "approximately 40-60"
        & ~below_ema50
        & ~distribution
    )

    # ── Entry confirmation ─────────────────────────────────────────
    if entry_mode == "reversal":
        # Bullish reversal candle: closes up, in upper half of range,
        # and closes above prior close
        rng = (h - l).replace(0, np.nan)
        reversal = (c > o) & ((c - l) / rng > 0.5) & (c > c.shift())
        confirmed = pullback_ok & reversal
        trigger = h.where(confirmed) * 1.001       # buy-stop just above candle high
    else:  # "reclaim"
        reclaim = (c > ema20) & (c.shift() <= ema20.shift())
        confirmed = pullback_ok & reclaim
        trigger = c.where(confirmed) * 1.001       # next-session entry near close

    swing_low = l.rolling(10).min()
    init_stop = swing_low - stop_atr_buffer * atr

    out = pd.DataFrame(index=df.index)
    out["setup_ok"] = (trend_ok & confirmed).fillna(False)
    out["trigger"] = trigger
    out["init_stop"] = init_stop
    out["atr"] = atr
    out["rsi"] = rsi
    stop_pct = 1 - init_stop / trigger
    out["setup_ok"] &= (stop_pct <= MAX_STOP_PCT).fillna(False)
    return out


class EnhancedEMAPullbackStrategy(BaseStrategy):
    """Registry wrapper — reads the last row of the causal setup table."""

    def __init__(self, stop_atr_buffer: float = 0.25, entry_mode: str = "reversal"):
        self.stop_atr_buffer = stop_atr_buffer
        self.entry_mode = entry_mode
        super().__init__()

    @property
    def name(self) -> str:
        return "enhanced_ema_pullback"

    @property
    def description(self) -> str:
        return "Trend-template EMA pullback with confirmed trigger entry (research)"

    def generate_signal(self, df: pd.DataFrame, ticker: str, cfg: dict) -> Optional[Signal]:
        if len(df) < 220:
            return None
        table = compute_setup_table(df, self.stop_atr_buffer, self.entry_mode)
        last = table.iloc[-1]
        if not bool(last["setup_ok"]):
            return None

        trigger = float(last["trigger"])
        stop = float(last["init_stop"])
        risk = trigger - stop
        if risk <= 0:
            return None
        t1, t2, t3 = trigger + 1.5 * risk, trigger + 2.5 * risk, trigger + 4.0 * risk

        return Signal(
            id=str(uuid.uuid4()),
            ticker=ticker,
            strategy=self.name,
            direction=Direction.LONG,
            strength=SignalStrength.MODERATE,
            entry_price=trigger,
            stop_loss=stop,
            target_1=t1, target_2=t2, target_3=t3,
            score=70,
            atr=float(last["atr"]),
            notes=f"{self.entry_mode} entry | trigger={trigger:.2f} | RSI={last['rsi']:.0f}",
        )
