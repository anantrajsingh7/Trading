"""
Data loading layer.
Supports two modes:
  - csv: read OHLCV files from data/raw/<TICKER>.csv
  - api: download from yfinance (default) or other providers
"""
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from utils import get_root, load_config

log = logging.getLogger("swing.data")


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info("Loaded %d S&P 500 tickers", len(tickers))
        return tickers
    except Exception as e:
        log.warning("Could not fetch S&P 500 tickers: %s", e)
        return []


def get_nasdaq100_tickers() -> list[str]:
    """Fetch Nasdaq-100 tickers from Wikipedia."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            attrs={"id": "constituents"},
        )
        tickers = tables[0]["Ticker"].str.replace(".", "-", regex=False).tolist()
        log.info("Loaded %d Nasdaq-100 tickers", len(tickers))
        return tickers
    except Exception as e:
        log.warning("Could not fetch Nasdaq-100 tickers: %s", e)
        return []


def get_custom_tickers(csv_path: str) -> list[str]:
    p = get_root() / csv_path
    if not p.exists():
        return []
    df = pd.read_csv(p, header=None)
    tickers = df.iloc[:, 0].astype(str).str.strip().tolist()
    log.info("Loaded %d custom tickers from %s", len(tickers), csv_path)
    return tickers


def get_universe(cfg: dict) -> list[str]:
    u_cfg = cfg.get("universe", {})
    tickers: list[str] = []
    if u_cfg.get("sp500", True):
        tickers += get_sp500_tickers()
    if u_cfg.get("nasdaq100", True):
        tickers += get_nasdaq100_tickers()
    custom_path = u_cfg.get("custom_csv", "data/tickers.csv")
    tickers += get_custom_tickers(custom_path)
    # deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tickers:
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


# ---------------------------------------------------------------------------
# CSV provider
# ---------------------------------------------------------------------------

def _load_csv(ticker: str, raw_dir: Path) -> Optional[pd.DataFrame]:
    candidates = [
        raw_dir / f"{ticker}.csv",
        raw_dir / f"{ticker.upper()}.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"])
            df = df.rename(columns=str.lower)
            required = {"date", "open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                log.warning("%s: missing columns. Expected %s, got %s",
                            ticker, required, set(df.columns))
                return None
            df = df.sort_values("date").set_index("date")
            # prefer adjusted_close
            if "adjusted_close" in df.columns:
                df["close"] = df["adjusted_close"]
            return df
    return None


# ---------------------------------------------------------------------------
# yfinance provider
# ---------------------------------------------------------------------------

def _load_yfinance(ticker: str, period: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        raw = yf.download(ticker, period=period, auto_adjust=True,
                          progress=False, threads=False)
        if raw is None or raw.empty:
            log.warning("%s: no data from yfinance", ticker)
            return None
        # yfinance may return MultiIndex columns when downloading single ticker
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df.index.name = "date"
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        log.warning("%s: yfinance error – %s", ticker, e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ticker_data(ticker: str, cfg: dict) -> Optional[pd.DataFrame]:
    mode = cfg.get("data", {}).get("mode", "api")
    root = get_root()
    if mode == "csv":
        raw_dir = root / cfg["data"].get("csv_dir", "data/raw")
        return _load_csv(ticker, raw_dir)
    # api mode
    provider = cfg.get("data", {}).get("api_provider", "yfinance")
    period = cfg.get("data", {}).get("default_period", "2y")
    if provider == "yfinance":
        return _load_yfinance(ticker, period)
    log.error("Unknown provider: %s", provider)
    return None


def load_all(tickers: list[str], cfg: dict,
             cache: bool = True) -> dict[str, pd.DataFrame]:
    """Return {ticker: DataFrame} for all tickers that successfully load."""
    processed_dir = get_root() / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}

    from tqdm import tqdm
    for ticker in tqdm(tickers, desc="Loading data"):
        cache_file = processed_dir / f"{ticker}.parquet"
        if cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                result[ticker] = df
                continue
            except Exception:
                pass
        df = load_ticker_data(ticker, cfg)
        if df is not None and len(df) >= 60:
            if cache:
                df.to_parquet(cache_file)
            result[ticker] = df
    log.info("Loaded %d tickers", len(result))
    return result
