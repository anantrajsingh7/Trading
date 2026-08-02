from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitvavo_momentum.event_detector import EventSpec, detect_events, event_study
from bitvavo_momentum.features import build_features, build_market_context
from bitvavo_momentum.strategies import (
    ConsolidationBreakout,
    DynamicLevelRetest,
    ExhaustionMeanReversion,
    ExitPolicy,
    PullbackEntry,
    VolumeConfirmedEntry,
    rank_cross_sectional,
)


def _ramp_features(pre: int = 200, ramp: int = 120, post: int = 300, size: float = 0.12) -> pd.DataFrame:
    """Flat, then a controlled +size% ramp, then flat - a known event."""
    flat_before = np.full(pre, 100.0)
    ramp_path = np.linspace(100.0, 100.0 * (1 + size), ramp)
    flat_after = np.full(post, 100.0 * (1 + size))
    close = np.concatenate([flat_before, ramp_path, flat_after])
    index = pd.date_range("2024-01-01T00:00:00Z", periods=len(close), freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "open": close,
            "high": close * 1.0002,
            "low": close * 0.9998,
            "close": close,
            "volume": np.full(len(close), 100.0),
        }
    )
    return build_features(frame, "1m", volume_baseline_bars=100)


def test_event_fires_when_the_threshold_is_crossed():
    features = _ramp_features(size=0.12)
    spec = EventSpec(lookback_minutes=120, threshold=0.10, cooldown_minutes=240)
    events = detect_events(features, spec, "T-EUR", require_nonzero_volume_candles=1)
    assert len(events) == 1
    assert events["event_lookback_return"].iloc[0] >= 0.10


def test_event_does_not_fire_below_the_threshold():
    features = _ramp_features(size=0.05)
    spec = EventSpec(lookback_minutes=120, threshold=0.10)
    assert detect_events(features, spec, "T-EUR", require_nonzero_volume_candles=1).empty


def test_cooldown_collapses_one_rally_into_one_event():
    """Without de-duplication a single rally would fire dozens of times."""
    features = _ramp_features(pre=200, ramp=300, post=300, size=0.40)
    spec_no_cooldown = EventSpec(lookback_minutes=120, threshold=0.10, cooldown_minutes=1)
    spec_cooldown = EventSpec(lookback_minutes=120, threshold=0.10, cooldown_minutes=600)
    many = detect_events(features, spec_no_cooldown, "T-EUR", require_nonzero_volume_candles=1)
    few = detect_events(features, spec_cooldown, "T-EUR", require_nonzero_volume_candles=1)
    assert len(few) == 1
    assert len(many) >= len(few)


def test_events_require_data_quality_in_the_lookback():
    features = _ramp_features(size=0.12)
    features = features.copy()
    features["was_missing"] = True  # entire look-back is gaps
    spec = EventSpec(lookback_minutes=120, threshold=0.10)
    assert detect_events(features, spec, "T-EUR", max_missing_fraction_lookback=0.2).empty


def test_events_respect_the_eligibility_mask():
    features = _ramp_features(size=0.12)
    ineligible = pd.Series(False, index=features.index)
    spec = EventSpec(lookback_minutes=120, threshold=0.10)
    assert detect_events(features, spec, "T-EUR", eligibility=ineligible,
                         require_nonzero_volume_candles=1).empty


def test_peak_of_cluster_mode_is_flagged_non_causal():
    causal = EventSpec(lookback_minutes=120, threshold=0.10, dedup_mode="first_touch")
    peak = EventSpec(lookback_minutes=120, threshold=0.10, dedup_mode="peak_of_cluster")
    assert causal.is_causal
    assert not peak.is_causal
    with pytest.raises(ValueError):
        EventSpec(lookback_minutes=120, threshold=0.10, dedup_mode="magic")


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #
def _forward(path: list[float]) -> pd.DataFrame:
    close = np.asarray(path, dtype="float64")
    index = pd.date_range("2024-01-01T00:00:00Z", periods=len(close), freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
            "volume": np.full(len(close), 500.0), "vwap_session": close * 0.99,
            "ema_9": close * 0.99, "ema_20": close * 0.98,
        },
        index=index,
    )


def _event_row(**extra) -> pd.Series:
    row = {"close": 100.0, "atr_60m": 1.0, "volume_baseline": 100.0,
           "volume_zscore": 3.0, "rel_volume_60m": 3.0}
    row.update(extra)
    return pd.Series(row)


def test_pullback_waits_for_the_dip_then_confirmation():
    # rises to 105, falls to 102 (>2% off the high), then turns up
    path = [100, 102, 105, 104, 103, 102, 101.5, 103, 104, 105, 106]
    plan = PullbackEntry(pullback_pct=0.02, max_wait_minutes=60).find_entry(_event_row(), _forward(path))
    assert plan is not None
    assert plan.execution_offset > 3, "must not enter before the pullback occurred"


def test_pullback_does_not_fire_without_a_dip():
    path = [100, 101, 102, 103, 104, 105]
    assert PullbackEntry(pullback_pct=0.05).find_entry(_event_row(), _forward(path)) is None


def test_breakout_requires_a_tight_range_first():
    consolidation = [100.0] * 40
    breakout = [101.0, 102.0]
    plan = ConsolidationBreakout(
        min_consolidation_minutes=15, max_consolidation_minutes=60,
        max_range_pct=0.01, min_relative_volume=1.0,
    ).find_entry(_event_row(), _forward(consolidation + breakout))
    assert plan is not None
    assert plan.execution_offset >= 40


def test_breakout_rejects_a_wide_range():
    noisy = list(np.linspace(90, 110, 40)) + [115.0, 116.0]
    plan = ConsolidationBreakout(
        min_consolidation_minutes=15, max_consolidation_minutes=60, max_range_pct=0.01
    ).find_entry(_event_row(), _forward(noisy))
    assert plan is None


def test_volume_strategy_rejects_low_volume_events():
    strategy = VolumeConfirmedEntry(min_volume_zscore=2.0, min_relative_volume=2.0)
    assert strategy.find_entry(_event_row(volume_zscore=0.5), _forward([100.0] * 5)) is None
    assert strategy.find_entry(_event_row(volume_zscore=3.5), _forward([100.0] * 5)) is not None


def test_retest_requires_a_touch_and_a_reclaim():
    path = [105, 104, 103, 102, 101, 102, 103]
    plan = DynamicLevelRetest(level="vwap_session", tolerance_pct=0.02).find_entry(
        _event_row(), _forward(path)
    )
    assert plan is None or plan.execution_offset > 0


def test_mean_reversion_waits_for_a_deep_retrace():
    path = [100, 102, 105, 103, 100, 98, 96, 95, 95.5, 96.5, 97.5]
    plan = ExhaustionMeanReversion(min_retrace_pct=0.05, stabilisation_bars=2).find_entry(
        _event_row(), _forward(path)
    )
    assert plan is None or plan.execution_offset >= 6


def test_cross_sectional_ranking_selects_only_the_top_n():
    rows = []
    for i, score in enumerate([0.10, 0.08, 0.06, 0.04]):
        rows.append({
            "market": f"M{i}-EUR",
            "event_time": pd.Timestamp("2024-01-01T00:00:00Z"),
            "ret_60m": score,
            "realised_vol_60m": 0.01,
            "spread_proxy_bps": 10.0,
            "quote_volume_1440m": 1e6,
        })
    ranked = rank_cross_sectional(pd.DataFrame(rows), top_n=2)
    assert int(ranked["cs_selected"].sum()) == 2
    assert set(ranked[ranked["cs_selected"]]["market"]) == {"M0-EUR", "M1-EUR"}


def test_exit_policy_names_itself_readably():
    policy = ExitPolicy(take_profit_pct=0.03, stop_loss_pct=0.02, time_stop_minutes=240)
    assert "tp0.03" in policy.name and "sl0.02" in policy.name and "t240m" in policy.name


# --------------------------------------------------------------------------- #
# context / study
# --------------------------------------------------------------------------- #
def test_market_context_breadth_ignores_markets_without_data(universe):
    context = build_market_context(universe, "1m")
    assert not context.empty
    breadth = context["breadth_positive_1440m"].dropna()
    assert ((breadth >= 0) & (breadth <= 1)).all()


def test_event_study_reports_sample_sizes():
    events = pd.DataFrame({
        "market": ["A-EUR"] * 10,
        "event_time": pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC"),
        "fwd_ret_60m": np.linspace(-0.02, 0.05, 10),
        "fwd_mfe_60m": np.full(10, 0.04),
        "fwd_mae_60m": np.full(10, -0.02),
    })
    study = event_study(events, (60,))
    assert study["n_events"].iloc[0] == 10
    assert 0 <= study["hit_rate"].iloc[0] <= 1
