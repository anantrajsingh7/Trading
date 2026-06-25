"""
Risk management: position sizing, stop loss, targets.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TradeSetup:
    ticker: str
    strategy: str
    entry: float
    stop_loss: float
    target1: float
    target2: float
    risk_per_share: float
    rr_ratio: float
    shares: int
    dollar_risk: float
    position_value: float


def calc_stop_loss(df: pd.DataFrame, entry: float,
                   atr_mult: float = 1.5,
                   lookback: int = 10) -> float:
    """Stop below recent swing low or 1.5 ATR, whichever is tighter."""
    atr_val = float(df["atr"].iloc[-1])
    swing_low = float(df["low"].rolling(lookback).min().iloc[-1])
    atr_stop = entry - atr_mult * atr_val
    # use the higher (less risky) of the two
    return max(swing_low - 0.01, atr_stop)


def calc_targets(entry: float, stop: float,
                 min_rr: float = 2.0) -> tuple[float, float]:
    risk = entry - stop
    target1 = entry + min_rr * risk
    target2 = entry + (min_rr + 1) * risk
    return target1, target2


def position_size(account_size: float, risk_pct: float,
                  entry: float, stop: float) -> tuple[int, float, float]:
    """
    Returns (shares, dollar_risk, position_value).
    """
    risk_amount = account_size * risk_pct
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0, 0.0, 0.0
    shares = int(risk_amount / risk_per_share)
    dollar_risk = shares * risk_per_share
    position_value = shares * entry
    return shares, dollar_risk, position_value


def build_trade_setup(ticker: str, strategy: str,
                      df: pd.DataFrame, cfg: dict) -> TradeSetup | None:
    acc = cfg.get("account", {})
    account_size = float(acc.get("size", 10000))
    risk_pct = float(acc.get("risk_per_trade", 0.01))
    min_rr = float(acc.get("min_rr_ratio", 2.0))
    min_price = float(cfg.get("trading", {}).get("min_price", 5.0))
    atr_mult = float(cfg.get("indicators", {}).get("atr_multiplier", 1.5))

    last = df.iloc[-1]
    entry = float(last["close"])

    if entry < min_price:
        return None

    stop = calc_stop_loss(df, entry, atr_mult)
    if stop >= entry:
        return None

    t1, t2 = calc_targets(entry, stop, min_rr)
    risk_per_share = entry - stop
    rr = (t1 - entry) / risk_per_share if risk_per_share > 0 else 0.0

    if rr < min_rr:
        return None

    shares, dollar_risk, pos_val = position_size(account_size, risk_pct, entry, stop)
    if shares == 0:
        return None

    return TradeSetup(
        ticker=ticker,
        strategy=strategy,
        entry=round(entry, 2),
        stop_loss=round(stop, 2),
        target1=round(t1, 2),
        target2=round(t2, 2),
        risk_per_share=round(risk_per_share, 2),
        rr_ratio=round(rr, 2),
        shares=shares,
        dollar_risk=round(dollar_risk, 2),
        position_value=round(pos_val, 2),
    )
