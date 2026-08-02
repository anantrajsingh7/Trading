"""Phase 2: objective momentum-event detection.

An event is defined as::

    close(t) / close(t - N minutes) - 1 >= X

evaluated on closed bars only. The event is *stamped* at bar ``t``, meaning the
information became available at the close of bar ``t``; the earliest executable
moment is therefore bar ``t + 1`` (enforced in :mod:`backtester`).

De-duplication
--------------
A 30% rally would otherwise fire hundreds of overlapping 1-minute events and make
every subsequent statistic look like it rests on thousands of independent
observations when it rests on a handful of rallies. Two modes are supported:

``first_touch``
    Keep the first bar that crosses the threshold, then suppress everything for
    ``cooldown_minutes``. This is the causal choice and the default - it is what
    a live scanner would actually do.

``peak_of_cluster``
    Within a cluster, keep the bar with the largest look-back return. Useful for
    descriptive work; **not** usable for signal generation because identifying the
    peak requires seeing the rest of the cluster. Functions that produce
    tradeable signals reject this mode.

Forward returns
---------------
:func:`add_forward_returns` deliberately looks into the future. It exists only
for the descriptive event study (Phase 2, step 8) and is never used to build a
signal. The column names are prefixed ``fwd_`` so leakage into a feature matrix
is trivially detectable, and :func:`assert_no_forward_columns` enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .timeutils import interval_to_minutes, session_bucket

log = get_logger(__name__)

FORWARD_PREFIX = "fwd_"


@dataclass
class EventSpec:
    """One (look-back, threshold) definition under test."""

    lookback_minutes: int
    threshold: float
    cooldown_minutes: int = 240
    dedup_mode: str = "first_touch"
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"L{self.lookback_minutes}m_T{self.threshold:.3%}".replace(".000%", "%")
        if self.dedup_mode not in ("first_touch", "peak_of_cluster"):
            raise ValueError(f"Unknown dedup_mode {self.dedup_mode!r}")

    @property
    def is_causal(self) -> bool:
        return self.dedup_mode == "first_touch"


def _dedup_first_touch(candidate_index: pd.DatetimeIndex, cooldown: pd.Timedelta) -> list[pd.Timestamp]:
    kept: list[pd.Timestamp] = []
    last: pd.Timestamp | None = None
    for ts in candidate_index:
        if last is None or (ts - last) >= cooldown:
            kept.append(ts)
            last = ts
    return kept


def _dedup_peak(
    candidate_index: pd.DatetimeIndex,
    values: pd.Series,
    cooldown: pd.Timedelta,
) -> list[pd.Timestamp]:
    """Keep the strongest bar per contiguous cluster (non-causal, study only)."""
    kept: list[pd.Timestamp] = []
    cluster: list[pd.Timestamp] = []
    for ts in candidate_index:
        if cluster and (ts - cluster[-1]) > cooldown:
            kept.append(max(cluster, key=lambda t: values.get(t, -np.inf)))
            cluster = []
        cluster.append(ts)
    if cluster:
        kept.append(max(cluster, key=lambda t: values.get(t, -np.inf)))
    return kept


def detect_events(
    features: pd.DataFrame,
    spec: EventSpec,
    market: str,
    interval: str = "1m",
    eligibility: pd.Series | None = None,
    max_missing_fraction_lookback: float = 0.20,
    require_nonzero_volume_candles: int = 3,
) -> pd.DataFrame:
    """Detect de-duplicated momentum events for one market.

    Data-quality gates applied at event time (all backward-looking):

    * the look-back window must not be mostly missing bars - a "+12% move" built
      from three prints in two hours is a data artefact, not a rally;
    * the window must contain at least ``require_nonzero_volume_candles`` bars
      with real volume;
    * the market must be eligible (liquidity/age) at that timestamp.
    """
    if features.empty:
        return pd.DataFrame()

    im = interval_to_minutes(interval)
    n = max(1, int(round(spec.lookback_minutes / im)))

    close = features["close"]
    lookback_return = close / close.shift(n) - 1.0

    crossed = lookback_return >= spec.threshold
    # Require a genuine crossing: the previous bar was below the threshold.
    crossed = crossed & ~crossed.shift(1, fill_value=False)

    missing = features.get("was_missing")
    if missing is None:
        missing_fraction = pd.Series(0.0, index=features.index)
    else:
        missing_fraction = missing.astype(float).rolling(n, min_periods=1).mean()
    quality_ok = missing_fraction <= max_missing_fraction_lookback

    nonzero_volume_bars = (features["volume"] > 0).rolling(n, min_periods=1).sum()
    volume_ok = nonzero_volume_bars >= require_nonzero_volume_candles

    candidates = crossed & quality_ok & volume_ok & lookback_return.notna()
    if eligibility is not None and not eligibility.empty:
        aligned = eligibility.reindex(features.index, method="ffill").fillna(False).astype(bool)
        candidates = candidates & aligned

    candidate_index = features.index[candidates.fillna(False).to_numpy()]
    if len(candidate_index) == 0:
        return pd.DataFrame()

    cooldown = pd.Timedelta(minutes=spec.cooldown_minutes)
    if spec.dedup_mode == "first_touch":
        kept = _dedup_first_touch(pd.DatetimeIndex(candidate_index), cooldown)
    else:
        kept = _dedup_peak(pd.DatetimeIndex(candidate_index), lookback_return, cooldown)

    if not kept:
        return pd.DataFrame()

    events = features.loc[kept].copy()
    events.insert(0, "market", market)
    events.insert(1, "event_time", events.index)
    events["event_spec"] = spec.name
    events["event_lookback_minutes"] = spec.lookback_minutes
    events["event_threshold"] = spec.threshold
    events["event_lookback_return"] = lookback_return.loc[kept].to_numpy()
    events["event_dedup_mode"] = spec.dedup_mode
    events["event_missing_fraction"] = missing_fraction.loc[kept].to_numpy()
    events["session_bucket"] = [session_bucket(ts) for ts in kept]
    events = events.reset_index(drop=True)

    log.debug(
        "%s %s: %d events from %d raw crossings", market, spec.name, len(events), len(candidate_index)
    )
    return events


def add_forward_returns(
    events: pd.DataFrame,
    candles_by_market: dict[str, pd.DataFrame],
    horizons_minutes: tuple[int, ...] | list[int] = (15, 30, 60, 120, 240, 480, 1440),
    interval: str = "1m",
) -> pd.DataFrame:
    """Attach forward returns, MFE and MAE. **Descriptive study only.**

    The reference price is the *next bar's open* after the event bar, i.e. the
    first price a trader could realistically have transacted at. Using the event
    bar's own close would overstate every result.
    """
    if events.empty:
        return events

    im = interval_to_minutes(interval)
    out = events.copy()
    for column in [f"{FORWARD_PREFIX}ret_{h}m" for h in horizons_minutes]:
        out[column] = np.nan
    for h in horizons_minutes:
        out[f"{FORWARD_PREFIX}mfe_{h}m"] = np.nan
        out[f"{FORWARD_PREFIX}mae_{h}m"] = np.nan
    out[f"{FORWARD_PREFIX}reference_price"] = np.nan

    prepared: dict[str, pd.DataFrame] = {}
    for market, frame in candles_by_market.items():
        if frame is None or frame.empty:
            continue
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        prepared[market] = data.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()

    for market, group in out.groupby("market", sort=False):
        frame = prepared.get(market)
        if frame is None:
            continue
        index = frame.index
        for row_id, event_time in group["event_time"].items():
            pos = index.searchsorted(pd.Timestamp(event_time), side="right")
            if pos >= len(index):
                continue
            entry_ref = float(frame["open"].iloc[pos])
            if not np.isfinite(entry_ref) or entry_ref <= 0:
                continue
            out.at[row_id, f"{FORWARD_PREFIX}reference_price"] = entry_ref
            for h in horizons_minutes:
                n = max(1, int(round(h / im)))
                end = min(len(index), pos + n)
                window = frame.iloc[pos:end]
                if window.empty:
                    continue
                exit_price = float(window["close"].iloc[-1])
                out.at[row_id, f"{FORWARD_PREFIX}ret_{h}m"] = exit_price / entry_ref - 1.0
                out.at[row_id, f"{FORWARD_PREFIX}mfe_{h}m"] = float(window["high"].max()) / entry_ref - 1.0
                out.at[row_id, f"{FORWARD_PREFIX}mae_{h}m"] = float(window["low"].min()) / entry_ref - 1.0
    return out


def assert_no_forward_columns(frame: pd.DataFrame, context: str = "") -> None:
    """Raise if any ``fwd_`` column reached a place it must not be."""
    leaked = [c for c in frame.columns if c.startswith(FORWARD_PREFIX)]
    if leaked:
        raise AssertionError(
            f"Forward-looking columns leaked into {context or 'feature matrix'}: {leaked}"
        )


def build_event_dataset(
    features_by_market: dict[str, pd.DataFrame],
    specs: list[EventSpec],
    eligibility_by_market: dict[str, pd.Series] | None = None,
    interval: str = "1m",
    max_missing_fraction_lookback: float = 0.20,
    require_nonzero_volume_candles: int = 3,
) -> pd.DataFrame:
    """Run every spec across every market and concatenate the events."""
    eligibility_by_market = eligibility_by_market or {}
    frames: list[pd.DataFrame] = []
    for market, features in features_by_market.items():
        if features is None or features.empty:
            continue
        for spec in specs:
            events = detect_events(
                features,
                spec,
                market=market,
                interval=interval,
                eligibility=eligibility_by_market.get(market),
                max_missing_fraction_lookback=max_missing_fraction_lookback,
                require_nonzero_volume_candles=require_nonzero_volume_candles,
            )
            if not events.empty:
                frames.append(events)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True).sort_values(["event_time", "market"])
    return combined.reset_index(drop=True)


def specs_from_config(config: dict[str, Any]) -> list[EventSpec]:
    """Build the full grid of event specs from ``config/research.yaml``."""
    events_cfg = config.get("events", {})
    lookbacks = events_cfg.get("lookback_minutes", [120])
    thresholds = events_cfg.get("thresholds", [0.10])
    cooldown = int(events_cfg.get("cooldown_minutes", 240))
    dedup = str(events_cfg.get("dedup_mode", "first_touch"))
    return [
        EventSpec(lookback_minutes=int(lb), threshold=float(th), cooldown_minutes=cooldown, dedup_mode=dedup)
        for lb in lookbacks
        for th in thresholds
    ]


def primary_spec_from_config(config: dict[str, Any]) -> EventSpec:
    events_cfg = config.get("events", {})
    primary = events_cfg.get("primary", {})
    return EventSpec(
        lookback_minutes=int(primary.get("lookback_minutes", 120)),
        threshold=float(primary.get("threshold", 0.10)),
        cooldown_minutes=int(events_cfg.get("cooldown_minutes", 240)),
        dedup_mode=str(events_cfg.get("dedup_mode", "first_touch")),
    )


# --------------------------------------------------------------------------- #
# descriptive event study (Phase 2, step 8)
# --------------------------------------------------------------------------- #
def event_study(
    events: pd.DataFrame,
    horizons_minutes: tuple[int, ...] | list[int] = (15, 30, 60, 120, 240, 480, 1440),
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """Summarise forward outcomes per horizon - before any strategy exists.

    Reports mean, median, hit rate, dispersion and a paired t-statistic against
    zero. The t-statistic assumes independent observations; overlapping events on
    correlated coins violate that, so it is reported as a rough guide and the
    bootstrap in :mod:`robustness` is the authority.
    """
    if events.empty:
        return pd.DataFrame()
    keys = group_by or []
    rows: list[dict[str, Any]] = []
    groups = events.groupby(keys, dropna=False) if keys else [((), events)]
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        for h in horizons_minutes:
            col = f"{FORWARD_PREFIX}ret_{h}m"
            if col not in group.columns:
                continue
            series = group[col].dropna()
            if series.empty:
                continue
            n = len(series)
            mean = float(series.mean())
            std = float(series.std(ddof=1)) if n > 1 else np.nan
            tstat = mean / (std / np.sqrt(n)) if std and n > 1 else np.nan
            row = dict(zip(keys, key_tuple, strict=False)) if keys else {}
            row.update(
                {
                    "horizon_minutes": h,
                    "n_events": n,
                    "mean_return": mean,
                    "median_return": float(series.median()),
                    "hit_rate": float((series > 0).mean()),
                    "std_return": std,
                    "p05": float(series.quantile(0.05)),
                    "p95": float(series.quantile(0.95)),
                    "t_stat_naive": tstat,
                    "mean_mfe": float(group.get(f"{FORWARD_PREFIX}mfe_{h}m", pd.Series(dtype=float)).mean()),
                    "mean_mae": float(group.get(f"{FORWARD_PREFIX}mae_{h}m", pd.Series(dtype=float)).mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)
