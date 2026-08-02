"""Phase 9: transparent, causal market-regime classification.

Every label at time ``t`` is computed from data available strictly before ``t``.
The definitions are deliberately crude - EMA cross, rolling volatility quantile,
cross-sectional breadth - because a regime filter that needs a clustering model
to explain it cannot be sanity-checked when it fails in live trading.

Volatility quantiles are taken against the **expanding** history rather than the
full sample. Using full-sample quantiles would leak knowledge of the future
volatility distribution into a label used for entry decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger(__name__)

BTC_BULL = "btc_bull"
BTC_BEAR = "btc_bear"
BTC_SIDEWAYS = "btc_sideways"
VOL_HIGH = "high_vol"
VOL_LOW = "low_vol"
VOL_MID = "mid_vol"
BREADTH_BROAD = "broad_altcoin_momentum"
BREADTH_NARROW = "narrow_leadership"
BREADTH_MID = "mixed_breadth"
RISK_ON = "risk_on"
RISK_OFF = "risk_off"


@dataclass
class RegimeConfig:
    fast_ema_days: int = 10
    slow_ema_days: int = 50
    sideways_band: float = 0.02
    vol_lookback_days: int = 30
    vol_high_quantile: float = 0.70
    vol_low_quantile: float = 0.30
    breadth_lookback_minutes: int = 1440
    breadth_broad_threshold: float = 0.60
    breadth_narrow_threshold: float = 0.40

    @classmethod
    def from_config(cls, research_config: dict[str, Any]) -> RegimeConfig:
        cfg = research_config.get("regimes", {})
        trend = cfg.get("btc_trend", {})
        vol = cfg.get("volatility", {})
        breadth = cfg.get("breadth", {})
        return cls(
            fast_ema_days=int(trend.get("fast_ema_days", 10)),
            slow_ema_days=int(trend.get("slow_ema_days", 50)),
            sideways_band=float(trend.get("sideways_band", 0.02)),
            vol_lookback_days=int(vol.get("lookback_days", 30)),
            vol_high_quantile=float(vol.get("high_quantile", 0.70)),
            vol_low_quantile=float(vol.get("low_quantile", 0.30)),
            breadth_lookback_minutes=int(breadth.get("lookback_minutes", 1440)),
            breadth_broad_threshold=float(breadth.get("broad_threshold", 0.60)),
            breadth_narrow_threshold=float(breadth.get("narrow_threshold", 0.40)),
        )


def _expanding_quantile_rank(series: pd.Series, min_periods: int = 30) -> pd.Series:
    """Rank of each value within the history *up to and including* that point."""
    values = series.to_numpy(dtype="float64")
    out = np.full(len(values), np.nan)
    seen: list[float] = []
    for i, value in enumerate(values):
        if np.isfinite(value):
            seen.append(value)
        if len(seen) >= min_periods and np.isfinite(value):
            arr = np.asarray(seen)
            out[i] = float((arr <= value).mean())
    return pd.Series(out, index=series.index)


def classify_regimes(
    btc_daily: pd.DataFrame,
    config: RegimeConfig | None = None,
    breadth: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily regime labels from BTC daily candles (and optional breadth series).

    ``btc_daily`` must be tidy OHLCV with a ``timestamp`` column at 1-day
    resolution. The returned frame is indexed by day; :func:`attach_regimes`
    forward-fills it onto intraday timestamps so that a label applied at 10:00
    was computed from the previous day's close.
    """
    config = config or RegimeConfig()
    if btc_daily.empty:
        return pd.DataFrame()

    data = btc_daily.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
    close = data["close"]

    fast = close.ewm(span=config.fast_ema_days, adjust=False, min_periods=config.fast_ema_days).mean()
    slow = close.ewm(span=config.slow_ema_days, adjust=False, min_periods=config.slow_ema_days).mean()
    spread = (fast / slow - 1.0)

    trend = pd.Series(BTC_SIDEWAYS, index=close.index, dtype="object")
    trend[spread > config.sideways_band] = BTC_BULL
    trend[spread < -config.sideways_band] = BTC_BEAR
    trend[spread.isna()] = np.nan

    realised_vol = close.pct_change().rolling(config.vol_lookback_days, min_periods=10).std()
    vol_rank = _expanding_quantile_rank(realised_vol, min_periods=60)
    vol_label = pd.Series(VOL_MID, index=close.index, dtype="object")
    vol_label[vol_rank >= config.vol_high_quantile] = VOL_HIGH
    vol_label[vol_rank <= config.vol_low_quantile] = VOL_LOW
    vol_label[vol_rank.isna()] = np.nan

    out = pd.DataFrame(
        {
            "btc_trend": trend,
            "btc_ema_spread": spread,
            "btc_realised_vol_30d": realised_vol,
            "btc_vol_rank": vol_rank,
            "volatility_regime": vol_label,
            "btc_ret_30d": close / close.shift(30) - 1.0,
        }
    )

    if breadth is not None and not breadth.empty:
        daily_breadth = breadth.resample("1D").mean().reindex(out.index).ffill()
        breadth_label = pd.Series(BREADTH_MID, index=out.index, dtype="object")
        breadth_label[daily_breadth >= config.breadth_broad_threshold] = BREADTH_BROAD
        breadth_label[daily_breadth <= config.breadth_narrow_threshold] = BREADTH_NARROW
        breadth_label[daily_breadth.isna()] = np.nan
        out["breadth"] = daily_breadth
        out["breadth_regime"] = breadth_label
    else:
        out["breadth"] = np.nan
        out["breadth_regime"] = np.nan

    # Risk appetite: trend and breadth agreeing.
    risk = pd.Series(np.nan, index=out.index, dtype="object")
    bull = out["btc_trend"] == BTC_BULL
    bear = out["btc_trend"] == BTC_BEAR
    broad = out.get("breadth_regime") == BREADTH_BROAD
    narrow = out.get("breadth_regime") == BREADTH_NARROW
    risk[bull & (broad | out["breadth_regime"].isna())] = RISK_ON
    risk[bear | narrow] = RISK_OFF
    out["risk_regime"] = risk

    # Shift by one day: a label must not use the close of the day it labels.
    shifted = out.shift(1)
    shifted.index.name = "timestamp"
    return shifted


def attach_regimes(frame: pd.DataFrame, regimes: pd.DataFrame, time_column: str = "event_time") -> pd.DataFrame:
    """Left-join daily regime labels onto an event or trade table."""
    if frame.empty or regimes.empty:
        return frame
    out = frame.copy()
    times = pd.to_datetime(out[time_column], utc=True)
    day = times.dt.floor("1D")
    aligned = regimes.reindex(regimes.index.union(pd.DatetimeIndex(day.unique()))).sort_index().ffill()
    joined = aligned.reindex(pd.DatetimeIndex(day))
    for column in regimes.columns:
        out[column] = joined[column].to_numpy()
    return out


def regime_performance(trades: pd.DataFrame, regime_column: str = "btc_trend") -> pd.DataFrame:
    """Per-regime metric table (thin wrapper over :func:`metrics.breakdown`)."""
    from .metrics import breakdown

    if trades.empty or regime_column not in trades.columns:
        return pd.DataFrame()
    return breakdown(trades, by=regime_column)


def regime_gate(regime_label: str | float | None, allowed: list[str] | None) -> bool:
    """Should trading be enabled in this regime?

    Unknown/NaN labels return ``False``: a regime filter that silently passes
    when it has no information is not a filter.
    """
    if allowed is None:
        return True
    if regime_label is None or (isinstance(regime_label, float) and not np.isfinite(regime_label)):
        return False
    return str(regime_label) in allowed
