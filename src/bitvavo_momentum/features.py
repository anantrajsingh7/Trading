"""Causal feature engineering.

**Timing convention (the single most important thing in this file).**
A candle stamped ``T`` covers ``[T, T + interval)`` and is therefore only *known*
at ``T + interval``. Every feature in this module is computed from bars up to and
including bar ``T``, and the row is stamped ``T``. Consumers must treat a feature
row stamped ``T`` as "information available at ``T + interval``", which is exactly
what :mod:`event_detector` and :mod:`backtester` do: the earliest permissible
execution for a signal derived from bar ``T`` is bar ``T + 1``.

No feature here uses ``.shift(-n)``, ``center=True`` or any forward-looking
window. The test suite asserts this by recomputing features on truncated data and
requiring bit-identical values (``tests/test_no_lookahead.py``).

Features that cannot be reconstructed from historical candles - notably the
bid-ask spread and order-book imbalance - are provided as clearly named
*proxies* (``spread_proxy_bps``) and flagged, so that no report can silently
present a proxy as a measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .timeutils import interval_to_minutes

log = get_logger(__name__)

RETURN_WINDOWS = (15, 30, 60, 120, 180, 240)
EMA_SPANS = (9, 20, 50)


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def _bars(minutes: int, interval_minutes: int) -> int:
    return max(1, int(round(minutes / interval_minutes)))


def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    ranges = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(frame: pd.DataFrame, period_bars: int) -> pd.Series:
    tr = true_range(frame["high"], frame["low"], frame["close"].shift(1))
    return tr.ewm(alpha=1.0 / period_bars, adjust=False, min_periods=period_bars).mean()


def rsi(close: pd.Series, period_bars: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period_bars, adjust=False, min_periods=period_bars).mean()
    avg_loss = loss.ewm(alpha=1.0 / period_bars, adjust=False, min_periods=period_bars).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.where(avg_loss > 0, 100.0).where(avg_gain > 0, out.fillna(50.0))


def rolling_vwap(frame: pd.DataFrame, window_bars: int) -> pd.Series:
    """Rolling VWAP over a trailing window (typical price weighted by volume)."""
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    pv = (typical * frame["volume"]).rolling(window_bars, min_periods=1).sum()
    vol = frame["volume"].rolling(window_bars, min_periods=1).sum()
    return pv / vol.replace(0.0, np.nan)


def session_vwap(frame: pd.DataFrame) -> pd.Series:
    """VWAP anchored to the start of each UTC day - resets at 00:00 UTC."""
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    day = frame.index.floor("1D")
    pv = (typical * frame["volume"]).groupby(day).cumsum()
    vol = frame["volume"].groupby(day).cumsum()
    return pv / vol.replace(0.0, np.nan)


def consecutive_positive(close: pd.Series) -> pd.Series:
    """Number of consecutive strictly-positive bar returns ending at each bar."""
    up = close.diff().gt(0.0)
    group = (~up).cumsum()
    return up.groupby(group).cumsum().astype("float64")


# --------------------------------------------------------------------------- #
# per-market feature frame
# --------------------------------------------------------------------------- #
def build_features(
    candles: pd.DataFrame,
    interval: str = "1m",
    volume_baseline_bars: int = 1440,
    consolidation_lookback_bars: int = 240,
) -> pd.DataFrame:
    """Compute the full causal feature set for one market.

    Parameters
    ----------
    candles
        Tidy OHLCV with a ``timestamp`` column (UTC). Gaps may be present; they
        are left as NaN rather than filled, because a missing Bitvavo candle
        means "no trades", not "unchanged price".
    """
    if candles.empty:
        return pd.DataFrame()

    im = interval_to_minutes(interval)
    frame = candles.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()

    out = pd.DataFrame(index=frame.index)
    out["open"] = frame["open"]
    out["high"] = frame["high"]
    out["low"] = frame["low"]
    out["close"] = frame["close"]
    out["volume"] = frame["volume"]
    out["quote_volume"] = frame["volume"] * frame["close"]
    if "was_missing" in frame.columns:
        out["was_missing"] = frame["was_missing"].astype(bool)
    else:
        out["was_missing"] = False

    close = frame["close"]

    # -- returns over the tested look-back windows ---------------------------
    for minutes in RETURN_WINDOWS:
        n = _bars(minutes, im)
        out[f"ret_{minutes}m"] = close / close.shift(n) - 1.0

    # -- volume behaviour ----------------------------------------------------
    baseline_min_periods = max(5, min(60, volume_baseline_bars // 2))
    baseline = frame["volume"].rolling(volume_baseline_bars, min_periods=baseline_min_periods).median()
    # Shift by one bar: the baseline must exclude the bar being judged.
    out["volume_baseline"] = baseline.shift(1)
    for minutes in (15, 30, 60, 120):
        n = _bars(minutes, im)
        recent = frame["volume"].rolling(n, min_periods=1).mean()
        out[f"rel_volume_{minutes}m"] = recent / out["volume_baseline"].replace(0.0, np.nan)
    qv_baseline = (
        out["quote_volume"].rolling(volume_baseline_bars, min_periods=baseline_min_periods).median().shift(1)
    )
    out["quote_volume_baseline"] = qv_baseline
    out["quote_volume_60m"] = out["quote_volume"].rolling(_bars(60, im), min_periods=1).sum()
    out["quote_volume_1440m"] = out["quote_volume"].rolling(_bars(1440, im), min_periods=30).sum()

    # Volume z-score against the coin's OWN history (Strategy E input).
    z_min_periods = max(10, min(120, volume_baseline_bars // 2))
    vol_mean = frame["volume"].rolling(volume_baseline_bars, min_periods=z_min_periods).mean().shift(1)
    vol_std = frame["volume"].rolling(volume_baseline_bars, min_periods=z_min_periods).std().shift(1)
    out["volume_zscore"] = (frame["volume"] - vol_mean) / vol_std.replace(0.0, np.nan)

    # -- volatility ----------------------------------------------------------
    bar_return = close.pct_change()
    for minutes in (30, 60, 240, 1440):
        n = _bars(minutes, im)
        out[f"realised_vol_{minutes}m"] = bar_return.rolling(n, min_periods=max(5, n // 4)).std()
    out["atr_60m"] = atr(frame, _bars(60, im))
    out["atr_240m"] = atr(frame, _bars(240, im))
    out["atr_60m_pct"] = out["atr_60m"] / close

    # -- oscillators / trend -------------------------------------------------
    out["rsi_14"] = rsi(close, 14)
    out["rsi_60m"] = rsi(close, _bars(60, im))
    for span in EMA_SPANS:
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        out[f"ema_{span}"] = ema
        out[f"dist_ema_{span}"] = close / ema - 1.0

    out["vwap_session"] = session_vwap(frame)
    out["dist_vwap_session"] = close / out["vwap_session"] - 1.0
    out["vwap_240m"] = rolling_vwap(frame, _bars(240, im))
    out["dist_vwap_240m"] = close / out["vwap_240m"] - 1.0

    # -- candle structure ----------------------------------------------------
    body = (frame["close"] - frame["open"]).abs()
    full = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    out["body_ratio"] = body / full
    out["upper_wick_ratio"] = (frame["high"] - frame[["open", "close"]].max(axis=1)) / full
    out["lower_wick_ratio"] = (frame[["open", "close"]].min(axis=1) - frame["low"]) / full
    out["consecutive_up_bars"] = consecutive_positive(close)

    # -- structure: highs, lows, pullback, breakout, consolidation ----------
    for minutes in (60, 120, 240, 1440):
        n = _bars(minutes, im)
        out[f"high_{minutes}m"] = frame["high"].rolling(n, min_periods=1).max()
        out[f"low_{minutes}m"] = frame["low"].rolling(n, min_periods=1).min()
    out["pullback_from_high_240m"] = close / out["high_240m"] - 1.0
    out["distance_to_low_240m"] = close / out["low_240m"] - 1.0

    # Breakout: close above the prior window's high, where "prior" excludes the
    # current bar (shift(1)) so the comparison is not trivially self-satisfied.
    prior_high_60 = frame["high"].rolling(_bars(60, im), min_periods=1).max().shift(1)
    prior_high_240 = frame["high"].rolling(_bars(240, im), min_periods=1).max().shift(1)
    out["breakout_60m"] = (close > prior_high_60).astype("float64")
    out["breakout_240m"] = (close > prior_high_240).astype("float64")
    out["prior_high_60m"] = prior_high_60
    out["prior_high_240m"] = prior_high_240

    # Consolidation: how long price has stayed inside a narrow range.
    n_cons = consolidation_lookback_bars
    roll_high = frame["high"].rolling(n_cons, min_periods=5).max()
    roll_low = frame["low"].rolling(n_cons, min_periods=5).min()
    out["range_width_pct"] = (roll_high - roll_low) / close
    out["consolidation_bars"] = _consolidation_length(frame, im)

    # Support/resistance proxies: distance to the trailing extremes.
    out["dist_resistance_1440m"] = close / out["high_1440m"] - 1.0
    out["dist_support_1440m"] = close / out["low_1440m"] - 1.0

    # -- microstructure proxies (NOT measurements) ---------------------------
    # Historical candles contain no quotes. The Corwin-Schultz style high/low
    # estimator gives an order-of-magnitude spread proxy; it is named accordingly
    # and never presented as an observed spread.
    hl_range = (frame["high"] - frame["low"]) / close
    out["spread_proxy_bps"] = (hl_range.rolling(_bars(60, im), min_periods=5).median() * 10_000 / 4.0)
    out["illiquidity_amihud"] = (
        bar_return.abs() / out["quote_volume"].replace(0.0, np.nan)
    ).rolling(_bars(1440, im), min_periods=30).median()

    # -- calendar ------------------------------------------------------------
    out["hour_utc"] = out.index.hour.astype("int16")
    out["day_of_week"] = out.index.dayofweek.astype("int16")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")

    out.index.name = "timestamp"
    return out


def _consolidation_length(frame: pd.DataFrame, interval_minutes: int, band: float = 0.02) -> pd.Series:
    """Bars since price last moved outside a +/-``band`` channel around itself.

    Implemented as a causal expanding count: a bar is "in consolidation" when the
    trailing 30-minute range is narrower than ``band``; the counter accumulates
    while that holds and resets otherwise.
    """
    n = _bars(30, interval_minutes)
    roll_high = frame["high"].rolling(n, min_periods=2).max()
    roll_low = frame["low"].rolling(n, min_periods=2).min()
    narrow = ((roll_high - roll_low) / frame["close"]) <= band
    narrow = narrow.fillna(False)
    group = (~narrow).cumsum()
    return narrow.groupby(group).cumsum().astype("float64")


# --------------------------------------------------------------------------- #
# cross-sectional / market-wide context
# --------------------------------------------------------------------------- #
def build_market_context(
    candles_by_market: dict[str, pd.DataFrame],
    interval: str = "1m",
    reference_markets: tuple[str, ...] = ("BTC-EUR", "ETH-EUR"),
    breadth_windows: tuple[int, ...] = (60, 120, 240, 1440),
) -> pd.DataFrame:
    """Market-wide context: BTC/ETH returns and cross-sectional breadth.

    Breadth at time ``t`` is the fraction of markets **with data at ``t``** whose
    trailing return over the window is positive. Markets that had not listed yet
    are excluded from both numerator and denominator, so breadth is not distorted
    by the growing universe.
    """
    if not candles_by_market:
        return pd.DataFrame()

    im = interval_to_minutes(interval)
    closes: dict[str, pd.Series] = {}
    for market, frame in candles_by_market.items():
        if frame is None or frame.empty:
            continue
        series = frame.copy()
        series["timestamp"] = pd.to_datetime(series["timestamp"], utc=True)
        s = series.drop_duplicates("timestamp", keep="last").set_index("timestamp")["close"].sort_index()
        closes[market] = s
    if not closes:
        return pd.DataFrame()

    panel = pd.DataFrame(closes).sort_index()
    context = pd.DataFrame(index=panel.index)

    for ref in reference_markets:
        if ref not in panel.columns:
            log.warning("Reference market %s missing; its context columns will be NaN", ref)
            continue
        ref_close = panel[ref].ffill()
        prefix = ref.split("-")[0].lower()
        for minutes in RETURN_WINDOWS:
            n = _bars(minutes, im)
            context[f"{prefix}_ret_{minutes}m"] = ref_close / ref_close.shift(n) - 1.0
        for minutes in (1440, 10080):
            n = _bars(minutes, im)
            context[f"{prefix}_ret_{minutes}m"] = ref_close / ref_close.shift(n) - 1.0
        context[f"{prefix}_realised_vol_1440m"] = (
            ref_close.pct_change().rolling(_bars(1440, im), min_periods=60).std()
        )

    for minutes in breadth_windows:
        n = _bars(minutes, im)
        rets = panel / panel.shift(n) - 1.0
        valid = rets.notna()
        positive = (rets > 0) & valid
        denom = valid.sum(axis=1)
        context[f"breadth_positive_{minutes}m"] = positive.sum(axis=1) / denom.replace(0, np.nan)
        context[f"n_markets_{minutes}m"] = denom
        context[f"median_market_ret_{minutes}m"] = rets.median(axis=1)

    context.index.name = "timestamp"
    return context


def attach_context(features: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Left-join market context onto a per-market feature frame.

    ``reindex(method='ffill')`` is used so a market with a missing bar inherits
    the most recent *past* context value - never a future one.
    """
    if features.empty or context.empty:
        return features
    aligned = context.reindex(features.index, method="ffill")
    overlapping = [c for c in aligned.columns if c in features.columns]
    if overlapping:
        aligned = aligned.drop(columns=overlapping)
    return features.join(aligned, how="left")
