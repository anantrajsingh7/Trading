"""
Market universe management.

Fetches constituent lists for major indices.
Deduplicates across indices.
Applies market-cap and liquidity filters.
"""
from __future__ import annotations

import time
from functools import lru_cache

import pandas as pd

from core.logging import get_logger

log = get_logger("data.universe")

# Static fallback tickers (used when web scraping fails)
_SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "JPM","TSLA","UNH","V","XOM","MA","JNJ","PG","HD","COST","MRK","ABBV",
    "CVX","BAC","NFLX","KO","PEP","WMT","CRM","AMD","ORCL","MCD","ACN","LIN",
    "TMO","CSCO","ADBE","ABT","GE","DHR","TXN","QCOM","INTU","AMGN","MS","VZ",
    "NKE","CAT","SPGI","GS","LOW","RTX","HON","BLK","ELV","BKNG","AXP","IBM",
    "UPS","ISRG","SYK","VRTX","REGN","MDT","GILD","MMM","PLD","CI","SBUX",
    "ADI","MU","LRCX","KLAC","AMAT","MRVL","PANW","CRWD","SNPS","CDNS",
    "NOW","WDAY","ZS","DDOG","FTNT","OKTA","MDB","NET","PLTR","RBLX",
]

_NASDAQ100_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","AMD","QCOM","INTU","ADBE","CSCO","TXN","AMGN","HON","SBUX",
    "ISRG","VRTX","REGN","GILD","MDLZ","PANW","CRWD","KLAC","LRCX","AMAT",
    "SNPS","CDNS","MRVL","MU","ADI","MCHP","ASML","ORLY","AZN","PCAR",
    "MNST","ROST","KDP","KHC","PAYX","FAST","ODFL","DXCM","IDXX","VRSK",
]

_RUSSELL1000_FALLBACK = _SP500_FALLBACK  # Approximation


@lru_cache(maxsize=None)
def get_sp500_tickers() -> list[str]:
    try:
        df = _read_html_ua(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info("Fetched %d S&P 500 tickers", len(tickers))
        return tickers
    except Exception as e:
        log.warning("S&P 500 fetch failed (%s) — using fallback list", e)
        return _SP500_FALLBACK


def _read_html_ua(url: str, **kwargs):
    """pd.read_html with a browser User-Agent (Wikipedia 403s the default)."""
    from io import StringIO
    import requests
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        timeout=20,
    )
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text), **kwargs)


@lru_cache(maxsize=None)
def get_nasdaq100_tickers() -> list[str]:
    try:
        tables = _read_html_ua(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            attrs={"id": "constituents"},
        )
        df = tables[0]
        col = "Ticker" if "Ticker" in df.columns else df.columns[1]
        tickers = df[col].str.replace(".", "-", regex=False).tolist()
        log.info("Fetched %d Nasdaq-100 tickers", len(tickers))
        return tickers
    except Exception as e:
        log.warning("Nasdaq-100 fetch failed (%s) — using fallback list", e)
        return _NASDAQ100_FALLBACK


def get_nifty50_tickers() -> list[str]:
    return [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
        "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","TITAN.NS",
        "BAJFINANCE.NS","NESTLEIND.NS","WIPRO.NS","ULTRACEMCO.NS","SUNPHARMA.NS",
        "TECHM.NS","TATAMOTORS.NS","POWERGRID.NS","NTPC.NS","HCLTECH.NS",
        "BAJAJFINSV.NS","ONGC.NS","JSWSTEEL.NS","TATASTEEL.NS","COALINDIA.NS",
        "DRREDDY.NS","CIPLA.NS","ADANIPORTS.NS","HINDALCO.NS","DIVISLAB.NS",
        "EICHERMOT.NS","BPCL.NS","GRASIM.NS","APOLLOHOSP.NS","HEROMOTOCO.NS",
        "INDUSINDBK.NS","SHRIRAMFIN.NS","TATACONSUM.NS","SBILIFE.NS","HDFCLIFE.NS",
        "BAJAJ-AUTO.NS","BRITANNIA.NS","M&M.NS","UPL.NS","LTIM.NS",
    ]


def get_universe(cfg: dict, market: str = "US") -> list[str]:
    """
    Build the full ticker universe for a market.
    Deduplicates and applies basic filters.
    """
    tickers: list[str] = []

    if market == "US":
        m_cfg = cfg.get("markets", {}).get("us", {})
        indices = m_cfg.get("indices", ["SP500", "NASDAQ100"])
        if "SP500" in indices:
            tickers += get_sp500_tickers()
        if "NASDAQ100" in indices:
            tickers += get_nasdaq100_tickers()

    elif market == "INDIA":
        tickers += get_nifty50_tickers()

    # Custom tickers from config (if any)
    custom = cfg.get("universe", {}).get("custom", [])
    tickers += custom

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for t in tickers:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)

    log.info("Universe (%s): %d unique tickers", market, len(unique))
    return unique
