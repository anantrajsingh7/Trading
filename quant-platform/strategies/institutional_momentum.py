"""
institutional_momentum_trend — RESEARCH strategy (not in production scan).

Systematic momentum / trend-following in the institutional style:
select the strongest stocks, trade only established Stage-2 trends, enter
after a controlled pullback, keep losses small, let winners run on an ATR
trail, and scale risk by market regime and volatility.

This is deliberately NOT a high-win-rate strategy. The design accepts a
low win rate in exchange for a long right tail.

Everything in compute_setup_table() is causal — only rolling windows and
shifted series — so the same function drives live scanning and bar-by-bar
backtesting without lookahead.

Portfolio-level rules (regime gate, breadth, VIX scaling, sector caps,
heat, earnings) live in research/portfolio_sim.py because they need
cross-sectional data a per-ticker strategy cannot see.
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
DEFAULT_RS_MIN = 80.0

# Entry-quality bands: how far above EMA20 the entry sits
EQ_IDEAL = 0.02        # <= 2% above EMA20
EQ_ACCEPTABLE = 0.05   # <= 5%
EQ_EXTENDED = 0.08     # <= 8%; beyond that -> DO_NOT_CHASE


def entry_quality(entry: float, ema20: float) -> str:
    """IDEAL / ACCEPTABLE / EXTENDED / DO_NOT_CHASE."""
    if ema20 <= 0:
        return "DO_NOT_CHASE"
    ext = (entry - ema20) / ema20
    if ext <= EQ_IDEAL:
        return "IDEAL"
    if ext <= EQ_ACCEPTABLE:
        return "ACCEPTABLE"
    if ext <= EQ_EXTENDED:
        return "EXTENDED"
    return "DO_NOT_CHASE"


def compute_setup_table(
    df: pd.DataFrame,
    stop_atr_buffer: float = 0.25,
    entry_mode: str = "reclaim",
) -> pd.DataFrame:
    """
    Causal per-bar table. Columns:
      setup_ok, trigger, init_stop, atr, rsi, ema20, ema50,
      ret20, ret60, ret120, pullback_pct, rel_vol, from_high,
      trend_quality, pullback_quality, vol_contraction, confirm_quality
    Row t uses only data up to and including t.
    """
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))

    sma50, sma150, sma200 = (c.rolling(n).mean() for n in (50, 150, 200))
    sma200_rising = sma200 > sma200.shift(20)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    dollar_vol = (c * v).rolling(20).mean()
    vol_avg20 = v.rolling(20).mean()

    # ── Momentum (medium-term, must already be positive) ───────────
    ret20 = c / c.shift(20) - 1
    ret60 = c / c.shift(60) - 1
    ret120 = c / c.shift(120) - 1
    momentum_ok = (ret60 > 0) & (ret120 > 0)

    # ── Trend template ─────────────────────────────────────────────
    trend_ok = (
        (c > sma50) & (sma50 > sma150) & (sma150 > sma200) & sma200_rising
        & (c > MIN_PRICE) & (dollar_vol >= MIN_DOLLAR_VOL)
    )

    # ── Controlled pullback from a recent 20-60d high ──────────────
    swing_high = h.shift(1).rolling(60).max()
    high20 = h.shift(1).rolling(20).max()
    made_recent_high = high20 >= swing_high * 0.98
    pullback_pct = 1 - c / swing_high
    near_ema = (l <= ema20 * 1.015) | (l <= ema50 * 1.015)
    rel_vol = v.rolling(3).mean() / vol_avg20
    vol_contracting = rel_vol < 1.0
    below_ema50 = (c < ema50 * 0.98).rolling(5).max().astype(bool)
    big_red = ((o - c) / o > 0.04) & (v > 2 * vol_avg20)
    big_gap = (o / c.shift() - 1) < -0.05
    distribution = (big_red | big_gap).rolling(5).max().astype(bool)

    pullback_ok = (
        made_recent_high
        & pullback_pct.between(PULLBACK_MIN, PULLBACK_MAX)
        & near_ema
        & vol_contracting
        & rsi.between(35, 62)
        & ~below_ema50
        & ~distribution
    )

    # ── Entry confirmation ─────────────────────────────────────────
    if entry_mode == "reversal":
        rng = (h - l).replace(0, np.nan)
        confirm = (c > o) & ((c - l) / rng > 0.5) & (c > c.shift())
        trigger = h.where(pullback_ok & confirm) * 1.001
    else:  # "reclaim" (primary)
        confirm = (c > ema20) & (c.shift() <= ema20.shift())
        trigger = c.where(pullback_ok & confirm) * 1.001

    swing_low = l.rolling(10).min()
    init_stop = swing_low - stop_atr_buffer * atr
    stop_pct = 1 - init_stop / trigger

    ok = (trend_ok & momentum_ok & pullback_ok & confirm).fillna(False)
    ok &= (stop_pct <= MAX_STOP_PCT).fillna(False)
    ok &= (init_stop < trigger).fillna(False)

    # ── Sub-scores for the Momentum Quality Score (0-1 each) ───────
    trend_quality = (
        ((c / sma50 - 1).clip(0, 0.15) / 0.15) * 0.5
        + ((sma50 / sma200 - 1).clip(0, 0.30) / 0.30) * 0.5
    ).clip(0, 1)
    # best pullback ≈ 5-6%: score peaks mid-band
    pullback_quality = (1 - ((pullback_pct - 0.055).abs() / 0.045)).clip(0, 1)
    vol_contraction = ((1 - rel_vol) / 0.4).clip(0, 1)
    # RSI closest to 50 = cleanest reset
    confirm_quality = (1 - ((rsi - 50).abs() / 20)).clip(0, 1)

    out = pd.DataFrame(index=df.index)
    out["setup_ok"] = ok
    out["trigger"] = trigger
    out["init_stop"] = init_stop
    out["atr"] = atr
    out["rsi"] = rsi
    out["ema20"] = ema20
    out["ema50"] = ema50
    out["ret20"] = ret20
    out["ret60"] = ret60
    out["ret120"] = ret120
    out["pullback_pct"] = pullback_pct
    out["rel_vol"] = rel_vol
    out["from_high"] = 1 - c / h.rolling(252).max()
    out["trend_quality"] = trend_quality
    out["pullback_quality"] = pullback_quality
    out["vol_contraction"] = vol_contraction
    out["confirm_quality"] = confirm_quality
    return out


def momentum_quality_score(
    row: pd.Series, rs_pct: float, regime_mult: float, dollar_vol: float
) -> float:
    """
    Transparent 0-100 score:
      RS 30% | Trend 20% | Pullback 15% | Volume contraction 10%
      Entry confirmation 10% | Market regime 10% | Liquidity 5%
    """
    liquidity = min(1.0, dollar_vol / 100_000_000) if dollar_vol > 0 else 0.0
    parts = (
        0.30 * (rs_pct / 100.0),
        0.20 * float(row.get("trend_quality", 0) or 0),
        0.15 * float(row.get("pullback_quality", 0) or 0),
        0.10 * float(row.get("vol_contraction", 0) or 0),
        0.10 * float(row.get("confirm_quality", 0) or 0),
        0.10 * regime_mult,
        0.05 * liquidity,
    )
    return round(100 * sum(parts), 1)


class InstitutionalMomentumStrategy(BaseStrategy):
    """Registry wrapper — reads the last row of the causal setup table."""

    def __init__(self, stop_atr_buffer: float = 0.25, entry_mode: str = "reclaim"):
        self.stop_atr_buffer = stop_atr_buffer
        self.entry_mode = entry_mode
        super().__init__()

    @property
    def name(self) -> str:
        return "institutional_momentum_trend"

    @property
    def description(self) -> str:
        return "Institutional momentum/trend: RS leader, Stage-2, controlled pullback, ATR trail"

    def generate_signal(self, df: pd.DataFrame, ticker: str, cfg: dict) -> Optional[Signal]:
        if len(df) < 260:
            return None
        tab = compute_setup_table(df, self.stop_atr_buffer, self.entry_mode)
        last = tab.iloc[-1]
        if not bool(last["setup_ok"]):
            return None
        trigger, stop = float(last["trigger"]), float(last["init_stop"])
        risk = trigger - stop
        if risk <= 0:
            return None
        return Signal(
            id=str(uuid.uuid4()), ticker=ticker, strategy=self.name,
            direction=Direction.LONG, strength=SignalStrength.MODERATE,
            entry_price=trigger, stop_loss=stop,
            target_1=trigger + 1.5 * risk,
            target_2=trigger + 3.0 * risk,
            target_3=trigger + 5.0 * risk,
            score=70, atr=float(last["atr"]),
            notes=(f"{self.entry_mode} | pullback {last['pullback_pct']*100:.1f}% "
                   f"| relvol {last['rel_vol']:.2f} | RSI {last['rsi']:.0f}"),
        )
