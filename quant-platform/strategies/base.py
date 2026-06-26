"""
Base strategy interface.

Every strategy must subclass BaseStrategy and implement:
  - generate_signal(df, ticker, cfg) -> Optional[Signal]
  - name (property)
  - description (property)

The Scanner calls generate_signal() on an enriched DataFrame (all indicators
already computed). The strategy just reads columns and decides yes/no.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from core.models import Signal, SignalStrength
from core.logging import get_logger


class BaseStrategy(ABC):
    """
    Abstract base for all swing trading strategies.

    Subclasses should be stateless — all inputs come in via generate_signal().
    """

    def __init__(self):
        self.log = get_logger(f"strategy.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'minervini_tt'."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description shown in reports."""

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        cfg: dict,
    ) -> Optional[Signal]:
        """
        Analyse enriched DataFrame and return a Signal or None.

        df     — OHLCV + all indicators (output of indicators.composite.enrich)
        ticker — ticker symbol
        cfg    — full config dict from config.yaml

        Return None if no trade opportunity.
        Return a Signal if entry conditions are met.
        """

    # ──────────────────────────────────────────
    # Helpers available to all strategies
    # ──────────────────────────────────────────

    def _last(self, df: pd.DataFrame) -> pd.Series:
        """Return the most recent bar."""
        return df.iloc[-1]

    def _prev(self, df: pd.DataFrame, n: int = 1) -> pd.Series:
        return df.iloc[-(1 + n)]

    def _atr_stop(self, row: pd.Series, multiplier: float = 2.0) -> float:
        """ATR-based stop loss below current price."""
        atr = row.get("atr_14", row["close"] * 0.02)
        return float(row["close"] - multiplier * atr)

    def _pct_stop(self, price: float, pct: float = 0.07) -> float:
        """Fixed percentage stop."""
        return price * (1 - pct)

    def _targets(
        self,
        entry: float,
        stop: float,
        r_multiples: tuple[float, float, float] = (1.5, 2.5, 4.0),
    ) -> tuple[float, float, float]:
        """Compute T1, T2, T3 based on risk multiples."""
        risk = entry - stop
        return (
            entry + r_multiples[0] * risk,
            entry + r_multiples[1] * risk,
            entry + r_multiples[2] * risk,
        )

    def _volume_expanding(self, df: pd.DataFrame, lookback: int = 10) -> bool:
        """True if today's volume is above the recent average."""
        avg = df["volume"].iloc[-lookback - 1: -1].mean()
        return float(df["volume"].iloc[-1]) > avg

    def _above_all_mas(self, row: pd.Series) -> bool:
        """True if close is above EMA20, EMA50, EMA150, EMA200."""
        c = row["close"]
        return all(
            c > row.get(f"ema_{p}", 0)
            for p in [20, 50, 150, 200]
        )
