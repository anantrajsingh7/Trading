"""The most important tests in the project.

If any of these fail, every number the system produces is worthless: a strategy
that can see the future always looks profitable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitvavo_momentum.data_validator import reindex_to_grid
from bitvavo_momentum.event_detector import (
    EventSpec,
    add_forward_returns,
    assert_no_forward_columns,
    detect_events,
)
from bitvavo_momentum.features import build_features
from bitvavo_momentum.strategies import EntryPlan, ImmediateEntry, PullbackEntry


def test_features_do_not_change_when_future_data_is_removed(candles):
    """Truncating the series must not alter any earlier feature value."""
    grid = reindex_to_grid(candles, "1m")
    full = build_features(grid, "1m")

    cut = len(grid) // 2
    truncated = build_features(grid.iloc[:cut].copy(), "1m")

    common = truncated.index
    # Skip the session VWAP on the final partial day of the truncated series:
    # it is anchored to the UTC day and legitimately differs where the day is cut.
    last_day = common[-1].floor("1D")
    comparable = common[common < last_day]
    assert len(comparable) > 1000, "not enough overlap to make the test meaningful"

    numeric = [c for c in full.columns if pd.api.types.is_float_dtype(full[c])]
    mismatched = []
    for column in numeric:
        a = full.loc[comparable, column].to_numpy(dtype="float64")
        b = truncated.loc[comparable, column].to_numpy(dtype="float64")
        both_nan = np.isnan(a) & np.isnan(b)
        if not np.allclose(a[~both_nan], b[~both_nan], rtol=1e-9, atol=1e-12, equal_nan=True):
            mismatched.append(column)
    assert not mismatched, f"these features changed when future data was removed: {mismatched}"


def test_event_detection_is_stable_under_truncation(features):
    """An event detected at time t must still be detected without later data."""
    spec = EventSpec(lookback_minutes=60, threshold=0.05, cooldown_minutes=120)
    full = detect_events(features, spec, "TEST-EUR")
    if full.empty:
        pytest.skip("no events in the fixture at this threshold")

    cut_time = full["event_time"].iloc[len(full) // 2]
    truncated_features = features[features.index <= cut_time]
    truncated = detect_events(truncated_features, spec, "TEST-EUR")

    earlier = set(full[full["event_time"] <= cut_time]["event_time"])
    assert earlier == set(truncated["event_time"]), (
        "event set changed when future bars were removed - detection is not causal"
    )


def test_forward_columns_are_prefixed_and_detectable(features, universe, candles):
    spec = EventSpec(lookback_minutes=60, threshold=0.05)
    events = detect_events(features, spec, "TEST-EUR")
    if events.empty:
        pytest.skip("no events in the fixture")

    assert_no_forward_columns(events, "raw event table")
    with_forward = add_forward_returns(events, {"TEST-EUR": candles}, (30, 60))
    with pytest.raises(AssertionError):
        assert_no_forward_columns(with_forward, "event table with forward returns")


def test_entry_plan_rejects_same_bar_execution():
    """Filling on the bar that generated the signal is look-ahead by construction."""
    with pytest.raises(ValueError):
        EntryPlan(decision_offset=5, execution_offset=5, order_type="market")
    with pytest.raises(ValueError):
        EntryPlan(decision_offset=5, execution_offset=4, order_type="market")
    EntryPlan(decision_offset=5, execution_offset=6, order_type="market")  # fine


def test_immediate_entry_uses_the_next_bar_open(features):
    forward = features.iloc[100:200]
    event = features.iloc[99]
    plan = ImmediateEntry().find_entry(event, forward)
    assert plan is not None
    assert plan.execution_offset == 0
    assert plan.reference_price == pytest.approx(float(forward["open"].iloc[0]))


def test_pullback_entry_never_uses_a_future_bar(features):
    """The strategy's decision bar must precede the execution bar it returns."""
    strategy = PullbackEntry(pullback_pct=0.01, max_wait_minutes=120)
    for start in range(200, 2000, 137):
        event = features.iloc[start - 1]
        forward = features.iloc[start : start + 200]
        plan = strategy.find_entry(event, forward)
        if plan is None:
            continue
        assert plan.execution_offset > plan.decision_offset
        # Re-running with only the bars up to the decision must give the same answer.
        limited = forward.iloc[: plan.decision_offset + 1]
        replayed = strategy.find_entry(event, limited)
        if replayed is not None:
            assert replayed.decision_offset == plan.decision_offset


def test_volume_baseline_excludes_the_current_bar(features):
    """A baseline that includes the bar being judged leaks that bar's information."""
    assert "volume_baseline" in features.columns
    # The first non-NaN baseline must appear strictly after the first non-NaN
    # rolling median would, because of the deliberate shift(1).
    baseline = features["volume_baseline"]
    raw_median = features["volume"].rolling(1440, min_periods=60).median()
    first_baseline = baseline.first_valid_index()
    first_raw = raw_median.first_valid_index()
    assert first_baseline > first_raw
