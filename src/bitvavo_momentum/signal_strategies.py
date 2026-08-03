"""Signal-scanning strategy families (spec Strategies 1, 2, 8).

These differ in shape from the event-conditioned strategies in
:mod:`strategies`. Those wait for a momentum *event* and then decide when to
enter; these scan every completed bar for a setup. To avoid duplicating the
backtester, risk manager, cost model and metrics, each one emits a **signal
table with the same schema as the event table**, which the existing
:class:`~.backtester.Backtester` consumes unchanged.

Timing convention, inherited and enforced
-----------------------------------------
Indicators are computed from completed bars only. A signal stamped at bar ``T``
became known at the close of ``T``; the backtester will not fill it before
``T + 1``. Every rolling window that could otherwise peek is shifted explicitly,
and ``tests/test_signal_strategies.py`` re-runs each generator on truncated data
and requires the earlier signals to be unchanged.

Why these three families
------------------------
The impulse-conditioned families (spec Strategies 3-6) were closed by the event
study: forward returns after a price impulse degrade monotonically with the size
and speed of that impulse, and the best of 288 threshold x lookback x horizon
cells reached +0.42% gross against a 77 bps cost floor. Trend, compression
breakout and Donchian breakout condition on entirely different information -
structure and volatility state rather than recent return - and trade less often,
which matters when every round trip costs 77 bps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .features import atr as _atr
from .logging_utils import get_logger
from .timeutils import interval_to_minutes, session_bucket

log = get_logger(__name__)

SIGNAL_SCHEMA_COLUMNS = [
    "market", "event_time", "event_spec", "event_lookback_return",
    "close", "atr_60m", "realised_vol_60m", "volume", "session_bucket",
]


# --------------------------------------------------------------------------- #
# setup-timeframe indicators
# --------------------------------------------------------------------------- #
def compute_setup_features(
    frame: pd.DataFrame,
    interval: str = "15m",
    ema_spans: tuple[int, ...] = (20, 50, 200),
    atr_period: int = 14,
    volume_baseline_bars: int = 96,
) -> pd.DataFrame:
    """Indicators for the signal-scanning families, all causal.

    Percentile-ranked features (Bollinger width, ATR, range) use an **expanding**
    window rather than the full sample: knowing that today's volatility is in the
    bottom decile *of the whole history* requires knowing the future, and a
    compression strategy built on that ranks as brilliant in backtest and fails
    live.
    """
    if frame.empty:
        return pd.DataFrame()

    interval_minutes = interval_to_minutes(interval)
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()

    out = pd.DataFrame(index=data.index)
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = data[column]
    if "n_source_bars" in data.columns:
        out["n_source_bars"] = data["n_source_bars"]
    out["quote_volume"] = data["volume"] * data["close"]

    close = data["close"]

    # -- trend ---------------------------------------------------------------
    for span in ema_spans:
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        out[f"ema_{span}"] = ema
        out[f"dist_ema_{span}"] = close / ema - 1.0
    if 20 in ema_spans and 50 in ema_spans:
        out["ema20_above_ema50"] = (out["ema_20"] > out["ema_50"]).astype("float64")
        out["ema_spread_20_50"] = out["ema_20"] / out["ema_50"] - 1.0
    if 50 in ema_spans and 200 in ema_spans:
        out["ema50_above_ema200"] = (out["ema_50"] > out["ema_200"]).astype("float64")
    # Slope over the last 5 bars, normalised by price.
    out["ema20_slope"] = (out["ema_20"] / out["ema_20"].shift(5) - 1.0)

    # -- volatility ----------------------------------------------------------
    atr_series = _atr(data, atr_period)
    out["atr"] = atr_series
    out["atr_pct"] = atr_series / close
    # Alias so the existing backtester's ATR-based stops and sizing work unchanged.
    out["atr_60m"] = atr_series

    bar_return = close.pct_change()
    out["realised_vol"] = bar_return.rolling(atr_period, min_periods=max(3, atr_period // 3)).std()
    out["realised_vol_60m"] = out["realised_vol"]

    # -- compression measures ------------------------------------------------
    middle = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std()
    out["bb_width"] = (2.0 * 2.0 * sd) / middle.replace(0.0, np.nan)
    out["range_pct"] = (
        data["high"].rolling(20, min_periods=5).max() - data["low"].rolling(20, min_periods=5).min()
    ) / close

    for name in ("bb_width", "atr_pct", "range_pct"):
        out[f"{name}_pctile"] = _expanding_percentile(out[name], min_periods=200)

    # -- volume --------------------------------------------------------------
    baseline = data["volume"].rolling(volume_baseline_bars, min_periods=20).median().shift(1)
    out["volume_baseline"] = baseline
    out["rel_volume"] = data["volume"] / baseline.replace(0.0, np.nan)

    # -- candle structure ----------------------------------------------------
    span = (data["high"] - data["low"]).replace(0.0, np.nan)
    out["close_position"] = (close - data["low"]) / span
    out["body_ratio"] = (close - data["open"]).abs() / span
    out["upper_wick_ratio"] = (data["high"] - data[["open", "close"]].max(axis=1)) / span

    # -- structure: Donchian channels (prior bars only) ----------------------
    for window in (12, 24, 48, 96):
        out[f"donchian_high_{window}"] = data["high"].rolling(window, min_periods=window).max().shift(1)
        out[f"donchian_low_{window}"] = data["low"].rolling(window, min_periods=window).min().shift(1)

    # -- liquidity proxies ---------------------------------------------------
    bars_per_day = max(1, int(1440 / interval_minutes))
    out["quote_volume_24h"] = out["quote_volume"].rolling(bars_per_day, min_periods=4).sum()
    out["spread_proxy_bps"] = (
        ((data["high"] - data["low"]) / close).rolling(bars_per_day, min_periods=4).median() * 10_000 / 4.0
    )

    out["hour_utc"] = out.index.hour.astype("int16")
    out["day_of_week"] = out.index.dayofweek.astype("int16")
    out.index.name = "timestamp"
    return out


def _expanding_percentile(series: pd.Series, min_periods: int = 200) -> pd.Series:
    """Percentile rank of each value within the history up to and including it.

    Expanding, never full-sample: a compression filter that knows the eventual
    distribution of volatility is using the future. Including the current bar is
    causal - its value is known at its own close - and is the conventional
    definition.
    """
    # pandas' expanding rank is Cython and O(n log n); the obvious Python loop
    # that rebuilds the history array on every bar is O(n^2) and took ~45s per
    # column on 19 months of 15-minute data, which made a 20-market run
    # unusable. Measured speed-up at n=20,000: ~400x.
    return series.expanding(min_periods=min_periods).rank(method="max", pct=True)


# --------------------------------------------------------------------------- #
# base class
# --------------------------------------------------------------------------- #
@dataclass
class SignalStrategy:
    """Scans completed bars and emits a backtester-compatible signal table."""

    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    cooldown_bars: int = 8

    def conditions(self, features: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def generate(self, features: pd.DataFrame, market: str) -> pd.DataFrame:
        """Emit de-duplicated signals for one market."""
        if features.empty:
            return pd.DataFrame()

        mask = self.conditions(features).fillna(False).astype(bool)
        # Require a genuine transition; a condition true for 200 consecutive bars
        # is one setup, not 200 independent observations.
        mask = mask & ~mask.shift(1, fill_value=False)
        if not mask.any():
            return pd.DataFrame()

        kept: list[pd.Timestamp] = []
        last_position = -10**9
        positions = np.flatnonzero(mask.to_numpy())
        for position in positions:
            if position - last_position >= self.cooldown_bars:
                kept.append(features.index[position])
                last_position = position
        if not kept:
            return pd.DataFrame()

        signals = features.loc[kept].copy()
        signals.insert(0, "market", market)
        signals.insert(1, "event_time", signals.index)
        signals["event_spec"] = self.name
        signals["event_family"] = self.family
        signals["event_lookback_return"] = signals.get(
            "dist_ema_20", pd.Series(np.nan, index=signals.index)
        )
        signals["session_bucket"] = [session_bucket(ts) for ts in kept]
        for key, value in self.params.items():
            signals[f"param_{key}"] = value
        return signals.reset_index(drop=True)

    def describe(self) -> dict[str, Any]:
        return {"strategy": self.name, "family": self.family, **self.params}


# --------------------------------------------------------------------------- #
# Strategy 1 - time-series trend following
# --------------------------------------------------------------------------- #
@dataclass
class TrendFollowing(SignalStrategy):
    """EMA-stack trend with a breakout or pullback trigger.

    Variants are deliberately few and each is economically distinct rather than
    a nudge of the same idea: the alignment condition, whether slope is
    required, and whether entry is on strength (breakout) or weakness
    (pullback).
    """

    require_ema50_above_200: bool = False
    require_positive_slope: bool = True
    trigger: str = "breakout"           # "breakout" | "pullback"
    breakout_window: int = 24
    pullback_tolerance: float = 0.005
    min_rel_volume: float = 1.0

    def conditions(self, features: pd.DataFrame) -> pd.Series:
        close = features["close"]
        aligned = features.get("ema20_above_ema50", pd.Series(False, index=features.index)).astype(bool)
        above = (close > features["ema_20"]) & (close > features["ema_50"])
        ok = aligned & above

        if self.require_ema50_above_200 and "ema50_above_ema200" in features:
            ok &= features["ema50_above_ema200"].astype(bool)
        if self.require_positive_slope:
            ok &= features["ema20_slope"] > 0

        if self.trigger == "breakout":
            channel = features.get(f"donchian_high_{self.breakout_window}")
            if channel is None:
                return pd.Series(False, index=features.index)
            ok &= close > channel
            if self.min_rel_volume:
                ok &= features["rel_volume"].fillna(0.0) >= self.min_rel_volume
        else:  # pullback toward EMA20 that holds
            near = (close / features["ema_20"] - 1.0).abs() <= self.pullback_tolerance
            # Confirmation: this bar closed up and in the upper half of its range.
            recovering = (close > features["open"]) & (features["close_position"] > 0.5)
            ok &= near & recovering
        return ok


# --------------------------------------------------------------------------- #
# Strategy 2 - volatility-compression breakout
# --------------------------------------------------------------------------- #
@dataclass
class CompressionBreakout(SignalStrategy):
    """Quiet range, then an expansion through resistance on volume.

    Compression is measured by an expanding percentile of Bollinger width, ATR
    or rolling range - whichever the variant selects - so "quiet" means quiet
    relative to this market's own past, not relative to a cross-sectional
    constant.
    """

    compression_measure: str = "bb_width_pctile"
    compression_max_pctile: float = 0.25
    compression_bars: int = 4
    breakout_window: int = 20
    min_rel_volume: float = 1.5
    min_close_position: float = 0.6
    require_trend: bool = False
    max_extension_atr: float = 3.0

    def conditions(self, features: pd.DataFrame) -> pd.Series:
        measure = features.get(self.compression_measure)
        if measure is None:
            return pd.Series(False, index=features.index)

        # Compression must have held for several bars BEFORE this one.
        was_compressed = (
            (measure <= self.compression_max_pctile)
            .rolling(self.compression_bars, min_periods=self.compression_bars)
            .min()
            .shift(1)
            .astype("float64")
            > 0
        )

        close = features["close"]
        resistance = features["high"].rolling(self.breakout_window, min_periods=self.breakout_window).max().shift(1)
        ok = was_compressed & (close > resistance)
        ok &= features["rel_volume"].fillna(0.0) >= self.min_rel_volume
        ok &= features["close_position"].fillna(0.0) >= self.min_close_position

        if self.require_trend and "ema20_above_ema50" in features:
            ok &= features["ema20_above_ema50"].astype(bool)

        # Extension filter: reject a breakout that has already run too far from
        # the EMA in ATR terms - that is the exhaustion pattern the event study
        # measured, and it is monotonically punished.
        atr = features["atr"].replace(0.0, np.nan)
        extension = (close - features["ema_20"]) / atr
        ok &= extension.fillna(0.0) <= self.max_extension_atr
        return ok


# --------------------------------------------------------------------------- #
# Strategy 8 - Donchian / turtle-style breakout
# --------------------------------------------------------------------------- #
@dataclass
class DonchianBreakout(SignalStrategy):
    """Break of the prior N-bar high, optionally trend- and volume-filtered.

    The classic turtle rule adapted to a 48-hour maximum hold. Channel windows
    are quoted in setup-timeframe bars: at 15 minutes, 12/24/48/96 bars are
    3/6/12/24 hours.
    """

    channel_window: int = 24
    require_trend: bool = True
    min_rel_volume: float = 1.0
    max_extension_atr: float = 4.0

    def conditions(self, features: pd.DataFrame) -> pd.Series:
        channel = features.get(f"donchian_high_{self.channel_window}")
        if channel is None:
            return pd.Series(False, index=features.index)

        close = features["close"]
        ok = close > channel
        if self.require_trend and "ema20_above_ema50" in features:
            ok &= features["ema20_above_ema50"].astype(bool)
        if self.min_rel_volume:
            ok &= features["rel_volume"].fillna(0.0) >= self.min_rel_volume

        atr = features["atr"].replace(0.0, np.nan)
        extension = (close - features["ema_20"]) / atr
        ok &= extension.fillna(0.0) <= self.max_extension_atr
        return ok


# --------------------------------------------------------------------------- #
# Strategy 11 - exhaustion avoidance (a veto, not an entry)
# --------------------------------------------------------------------------- #
def exhaustion_veto(
    features: pd.DataFrame,
    max_single_bar_share: float = 0.6,
    max_upper_wick_ratio: float = 0.5,
    max_extension_atr: float = 4.0,
    max_recent_return: float = 0.12,
    recent_bars: int = 8,
) -> pd.Series:
    """``True`` where an entry should be **rejected** as exhausted.

    Every threshold here is motivated by the measured gradient: forward returns
    after an impulse degrade monotonically with its size and with its speed, so
    the veto targets moves that are large, fast, or concentrated in one bar.

    Returns a boolean series aligned to ``features``; strategies AND their entry
    condition with ``~exhaustion_veto(...)``.
    """
    if features.empty:
        return pd.Series(dtype=bool)

    close = features["close"]
    recent_return = close / close.shift(recent_bars) - 1.0
    bar_move = (close - features["open"]).abs()
    window_move = (close - close.shift(recent_bars)).abs().replace(0.0, np.nan)
    single_bar_share = bar_move / window_move

    atr = features["atr"].replace(0.0, np.nan)
    extension = (close - features["ema_20"]) / atr

    veto = (
        (recent_return > max_recent_return)
        | (single_bar_share > max_single_bar_share)
        | (features["upper_wick_ratio"] > max_upper_wick_ratio)
        | (extension > max_extension_atr)
    )
    veto.name = "exhaustion_veto"
    return veto.fillna(False)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def trend_variants() -> list[SignalStrategy]:
    return [
        TrendFollowing(name="S1_trend_breakout_24", family="trend", trigger="breakout",
                       breakout_window=24, params={"trigger": "breakout", "window": 24}),
        TrendFollowing(name="S1_trend_breakout_48", family="trend", trigger="breakout",
                       breakout_window=48, params={"trigger": "breakout", "window": 48}),
        TrendFollowing(name="S1_trend_pullback", family="trend", trigger="pullback",
                       params={"trigger": "pullback"}),
        TrendFollowing(name="S1_trend_strict", family="trend", trigger="breakout",
                       breakout_window=24, require_ema50_above_200=True,
                       params={"trigger": "breakout", "window": 24, "ema200": True}),
    ]


def compression_variants() -> list[SignalStrategy]:
    return [
        CompressionBreakout(name=f"S2_compression_{w}", family="compression",
                            breakout_window=w, params={"window": w})
        for w in (12, 16, 20, 24, 32)
    ] + [
        CompressionBreakout(name="S2_compression_atr", family="compression",
                            compression_measure="atr_pct_pctile", breakout_window=20,
                            params={"measure": "atr", "window": 20}),
        CompressionBreakout(name="S2_compression_trend", family="compression",
                            breakout_window=20, require_trend=True,
                            params={"window": 20, "trend": True}),
    ]


def donchian_variants() -> list[SignalStrategy]:
    return [
        DonchianBreakout(name=f"S8_donchian_{w}", family="donchian", channel_window=w,
                         params={"window": w})
        for w in (12, 24, 48, 96)
    ] + [
        DonchianBreakout(name="S8_donchian_24_notrend", family="donchian", channel_window=24,
                         require_trend=False, params={"window": 24, "trend": False}),
    ]


def default_signal_strategies() -> list[SignalStrategy]:
    """The untested families: trend, compression breakout, Donchian.

    16 variants total - small enough that the multiple-testing correction stays
    meaningful, and each variant is economically distinct rather than a nudge.
    """
    return trend_variants() + compression_variants() + donchian_variants()


def generate_signals(
    strategies: list[SignalStrategy],
    features_by_market: dict[str, pd.DataFrame],
    apply_exhaustion_veto: bool = False,
    eligibility_by_market: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Run every strategy across every market and concatenate the signals."""
    frames: list[pd.DataFrame] = []
    for strategy in strategies:
        for market, features in features_by_market.items():
            if features is None or features.empty:
                continue
            working = features
            if apply_exhaustion_veto:
                veto = exhaustion_veto(features)
                working = features[~veto]
                if working.empty:
                    continue
            if eligibility_by_market is not None:
                mask = eligibility_by_market.get(market)
                if mask is not None and not mask.empty:
                    aligned = mask.reindex(working.index, method="ffill").fillna(False).astype(bool)
                    working = working[aligned]
                    if working.empty:
                        continue
            signals = strategy.generate(working, market)
            if not signals.empty:
                frames.append(signals)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True).sort_values(["event_time", "market"])
    log.info(
        "Generated %d signals across %d strategies and %d markets",
        len(combined), len(strategies), len(features_by_market),
    )
    return combined.reset_index(drop=True)


__all__ = [
    "CompressionBreakout",
    "DonchianBreakout",
    "SignalStrategy",
    "TrendFollowing",
    "compression_variants",
    "compute_setup_features",
    "default_signal_strategies",
    "donchian_variants",
    "exhaustion_veto",
    "generate_signals",
    "trend_variants",
]
