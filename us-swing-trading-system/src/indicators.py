"""
Technical indicator calculations.
All functions take a DataFrame with columns: open, high, low, close, volume
and return a new DataFrame with indicator columns appended.
No look-ahead bias: every value uses only data up to that row.
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    # when avg_loss == 0 the stock has only risen → RSI = 100
    rs = avg_gain / avg_loss.where(avg_loss != 0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss != 0, 100.0)
    return result


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def rolling_high(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).max()


def rolling_low(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).min()


def relative_strength(ticker_close: pd.Series,
                      benchmark_close: pd.Series,
                      period: int = 20) -> pd.Series:
    """RS = (ticker return over period) - (benchmark return over period)."""
    ticker_ret = ticker_close.pct_change(period)
    bench_ret = benchmark_close.pct_change(period)
    return ticker_ret - bench_ret


def compute_all(df: pd.DataFrame,
                spy: pd.Series | None = None,
                qqq: pd.Series | None = None,
                cfg: dict | None = None) -> pd.DataFrame:
    """
    Compute all indicators and return an enriched DataFrame.
    spy / qqq should be aligned Series of benchmark close prices.
    """
    c = cfg or {}
    ind = c.get("indicators", {})

    fast = int(ind.get("ema_fast", 20))
    mid = int(ind.get("ema_mid", 50))
    slow = int(ind.get("ema_slow", 200))
    rsi_p = int(ind.get("rsi_period", 14))
    atr_p = int(ind.get("atr_period", 14))
    avg_vol_p = int(ind.get("avg_volume_period", 20))

    out = df.copy()
    out["ema20"] = ema(out["close"], fast)
    out["ema50"] = ema(out["close"], mid)
    out["ema200"] = ema(out["close"], slow)
    out["rsi"] = rsi(out["close"], rsi_p)
    out["atr"] = atr(out, atr_p)
    out["atr20_avg"] = out["atr"].rolling(20, min_periods=5).mean()
    out["avg_volume"] = out["volume"].rolling(avg_vol_p, min_periods=5).mean()

    # 52-week high/low (252 trading days)
    out["high52w"] = rolling_high(out["high"], 252)
    out["low52w"] = rolling_low(out["low"], 252)

    # 20-day and 50-day resistance (rolling highs)
    out["res20"] = rolling_high(out["close"], 20)
    out["res50"] = rolling_high(out["close"], 50)

    if spy is not None:
        aligned_spy = spy.reindex(out.index, method="ffill")
        out["rs_spy"] = relative_strength(out["close"], aligned_spy)
    else:
        out["rs_spy"] = np.nan

    if qqq is not None:
        aligned_qqq = qqq.reindex(out.index, method="ffill")
        out["rs_qqq"] = relative_strength(out["close"], aligned_qqq)
    else:
        out["rs_qqq"] = np.nan

    return out
