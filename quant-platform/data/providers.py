"""
Data providers — modular plug-in architecture.

Each provider implements the DataProvider protocol:
  fetch(ticker, period, interval) -> pd.DataFrame

Currently implemented:
  YFinanceProvider — free, no API key, covers US + India

Future providers (add without touching other code):
  PolygonProvider, AlphaVantageProvider, TwelveDataProvider, etc.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from core.logging import get_logger

log = get_logger("data.providers")


# ============================================================
# Provider Protocol (ABC)
# ============================================================

class DataProvider(ABC):
    """Every data provider must implement this interface."""

    @abstractmethod
    def fetch(self, ticker: str, period: str = "5y",
              interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Returns OHLCV DataFrame with columns:
          open, high, low, close, adj_close, volume
        Index: DatetimeIndex (UTC-normalised)
        Returns None if data unavailable.
        """

    @abstractmethod
    def fetch_info(self, ticker: str) -> dict:
        """Returns company metadata dict."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ============================================================
# yfinance Provider
# ============================================================

class YFinanceProvider(DataProvider):
    """
    Yahoo Finance via yfinance library.
    Free, no API key required.
    Covers US, India (.NS), Europe (.L, .PA, etc.)

    Limitations:
    - Rate limited to ~2000 requests/hour
    - 15-minute delayed quotes
    - Occasionally has data quality issues
    """

    def __init__(self, sleep: float = 0.25):
        self._sleep = sleep

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch(self, ticker: str, period: str = "5y",
              interval: str = "1d") -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            time.sleep(self._sleep)
            raw = yf.download(
                ticker, period=period, interval=interval,
                auto_adjust=True, progress=False, threads=False,
            )
            if raw is None or raw.empty:
                log.debug("%s: empty response from yfinance", ticker)
                return None

            # Flatten MultiIndex columns (happens with single ticker sometimes)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            # adj_close same as close when auto_adjust=True
            df["adj_close"] = df["close"]
            df.index.name = "date"
            df.index = pd.to_datetime(df.index).normalize()

            cols = ["open", "high", "low", "close", "adj_close", "volume"]
            df = df[[c for c in cols if c in df.columns]].copy()
            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]
            return df

        except Exception as e:
            log.warning("%s: yfinance error — %s", ticker, e)
            return None

    def fetch_info(self, ticker: str) -> dict:
        try:
            import yfinance as yf
            time.sleep(self._sleep)
            info = yf.Ticker(ticker).info
            return {
                "name": info.get("shortName") or info.get("longName") or ticker,
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "country": info.get("country", "US"),
            }
        except Exception:
            return {"name": ticker, "sector": "Unknown",
                    "industry": "Unknown", "market_cap": 0, "country": "US"}


# ============================================================
# Provider Registry
# ============================================================

_PROVIDERS: dict[str, DataProvider] = {}


def register_provider(provider: DataProvider) -> None:
    _PROVIDERS[provider.name] = provider


def get_provider(name: str = "yfinance") -> DataProvider:
    if name not in _PROVIDERS:
        if name == "yfinance":
            register_provider(YFinanceProvider())
        else:
            raise ValueError(f"Provider not registered: {name}. "
                             f"Available: {list(_PROVIDERS.keys())}")
    return _PROVIDERS[name]


# Pre-register defaults
register_provider(YFinanceProvider())
