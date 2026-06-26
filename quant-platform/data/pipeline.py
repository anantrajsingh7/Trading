"""
Data pipeline orchestrator.

Single entry point: get_data(ticker, period) returns a clean, validated
OHLCV DataFrame, hitting cache first and provider on miss/stale.

Also handles batch fetching with progress reporting and error isolation
(one bad ticker never aborts the whole batch).
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from tqdm import tqdm

from core.config import get_config
from core.logging import get_logger
from data import cache
from data.providers import get_provider

log = get_logger("data.pipeline")

_MIN_ROWS = 60  # Require at least 60 bars for indicator warm-up


def get_data(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    provider_name: str = "yfinance",
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV for a single ticker. Cache-first, provider on stale/miss.

    Returns a validated DataFrame or None if data unavailable.
    """
    cache_key = f"{period}_{interval}"

    if not force_refresh and not cache.is_stale(ticker, cache_key):
        df = cache.read(ticker, cache_key)
        if df is not None and len(df) >= _MIN_ROWS:
            return df

    provider = get_provider(provider_name)
    df = provider.fetch(ticker, period=period, interval=interval)

    if df is None or df.empty:
        log.debug("%s: no data from provider", ticker)
        return None

    if len(df) < _MIN_ROWS:
        log.debug("%s: insufficient data (%d bars)", ticker, len(df))
        return None

    df = _validate(df, ticker)
    if df is None:
        return None

    cache.write(ticker, cache_key, df)
    return df


def get_data_batch(
    tickers: list[str],
    period: str = "5y",
    interval: str = "1d",
    provider_name: str = "yfinance",
    force_refresh: bool = False,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch multiple tickers. Returns dict of {ticker: DataFrame}.
    Failed tickers are silently excluded.
    """
    results: dict[str, pd.DataFrame] = {}
    iterator = tqdm(tickers, desc="Fetching data", unit="ticker") if show_progress else tickers

    for ticker in iterator:
        try:
            df = get_data(
                ticker, period=period, interval=interval,
                provider_name=provider_name, force_refresh=force_refresh,
            )
            if df is not None:
                results[ticker] = df
        except Exception as e:
            log.warning("%s: pipeline error — %s", ticker, e)

    log.info("Fetched %d / %d tickers", len(results), len(tickers))
    return results


def _validate(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Sanity-check OHLCV integrity. Returns None on fatal issues."""
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.debug("%s: missing columns %s", ticker, missing)
        return None

    # Drop rows where OHLC relationship is violated
    bad_hl = df["high"] < df["low"]
    bad_oc = (df["open"] <= 0) | (df["close"] <= 0)
    mask = ~(bad_hl | bad_oc)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        log.debug("%s: dropped %d invalid OHLC rows", ticker, n_dropped)
    df = df[mask].copy()

    if len(df) < _MIN_ROWS:
        log.debug("%s: too few valid rows after cleaning (%d)", ticker, len(df))
        return None

    # Forward-fill small gaps (weekends already handled by provider, but be safe)
    df = df.ffill(limit=3)

    return df
