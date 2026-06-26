"""
Two-level data cache.

Level 1: Parquet files (fast, columnar, ~100x faster than CSV)
Level 2: SQLite (queryable, relational)

Why two levels?
  Parquet gives fast bulk reads for backtesting (load 500 tickers in seconds).
  SQLite gives queryable history (when was data last updated? which tickers?).
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from core.config import get_cache_dir
from core.logging import get_logger

log = get_logger("data.cache")

_PARQUET_DIR = None


def _parquet_dir() -> Path:
    global _PARQUET_DIR
    if _PARQUET_DIR is None:
        _PARQUET_DIR = get_cache_dir() / "parquet"
        _PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    return _PARQUET_DIR


def _parquet_path(ticker: str, period: str) -> Path:
    safe = ticker.replace("/", "_").replace("\\", "_").replace(":", "_")
    return _parquet_dir() / f"{safe}_{period}.parquet"


def _meta_path(ticker: str, period: str) -> Path:
    safe = ticker.replace("/", "_").replace("\\", "_").replace(":", "_")
    return _parquet_dir() / f"{safe}_{period}.meta"


def is_stale(ticker: str, period: str, max_age_hours: int = 20) -> bool:
    """Returns True if the cache is missing or older than max_age_hours."""
    meta = _meta_path(ticker, period)
    p = _parquet_path(ticker, period)
    if not p.exists() or not meta.exists():
        return True
    try:
        ts = datetime.fromisoformat(meta.read_text().strip())
        age = datetime.utcnow() - ts
        return age.total_seconds() > max_age_hours * 3600
    except Exception:
        return True


def read(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Load from parquet cache. Returns None if not cached."""
    p = _parquet_path(ticker, period)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        log.debug("Cache read error %s: %s", ticker, e)
        p.unlink(missing_ok=True)
        return None


def write(ticker: str, period: str, df: pd.DataFrame) -> None:
    """Write DataFrame to parquet cache and update timestamp."""
    if df is None or df.empty:
        return
    p = _parquet_path(ticker, period)
    try:
        df.to_parquet(p, compression="snappy")
        _meta_path(ticker, period).write_text(datetime.utcnow().isoformat())
    except Exception as e:
        log.debug("Cache write error %s: %s", ticker, e)


def invalidate(ticker: str, period: str | None = None) -> None:
    """Remove cached data for a ticker."""
    safe = ticker.replace("/", "_").replace("\\", "_").replace(":", "_")
    for p in _parquet_dir().glob(f"{safe}_*.parquet"):
        if period is None or p.stem.endswith(f"_{period}"):
            p.unlink(missing_ok=True)


def cache_stats() -> dict:
    d = _parquet_dir()
    files = list(d.glob("*.parquet"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "cached_files": len(files),
        "total_mb": round(total_bytes / 1024 / 1024, 1),
        "cache_dir": str(d),
    }
