"""Multi-timeframe causality and baseline-control tests.

The first group is the highest-value set in the new code: a higher-timeframe
look-ahead leak is invisible in results and inflates every downstream number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitvavo_momentum.baseline import (
    buy_and_hold_benchmark,
    conditional_lift,
    lift_confidence_interval,
    matched_random_baseline,
    unconditional_forward_returns,
)
from bitvavo_momentum.timeframes import (
    TimeframeSet,
    align_higher_timeframe,
    assert_causal_join,
    bars_per,
    resample_ohlcv,
)


def _minutes(n: int = 600, start: str = "2025-01-01T00:00:00Z", drift: float = 0.0) -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = 100.0 * np.exp(np.arange(n) * drift)
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 10.0),
        }
    )


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def test_resample_aggregates_correctly():
    frame = _minutes(60)
    out = resample_ohlcv(frame, "15m")
    assert len(out) == 4
    assert out["volume"].iloc[0] == 150.0
    assert out["n_source_bars"].iloc[0] == 15
    assert out["open"].iloc[0] == pytest.approx(frame["open"].iloc[0])
    assert out["close"].iloc[0] == pytest.approx(frame["close"].iloc[14])


def test_resample_drops_empty_buckets_rather_than_inventing_bars():
    frame = _minutes(60).drop(index=range(15, 30)).reset_index(drop=True)
    out = resample_ohlcv(frame, "15m")
    assert len(out) == 3, "the fully-missing bucket must be dropped, not filled"


def test_partial_bars_are_flagged_by_source_count():
    frame = _minutes(60).drop(index=range(16, 29)).reset_index(drop=True)
    out = resample_ohlcv(frame, "15m")
    partial = out[out["n_source_bars"] < 15]
    assert len(partial) == 1, "a 15m bar built from 2 prints must be distinguishable"


# --------------------------------------------------------------------------- #
# causal alignment - the critical group
# --------------------------------------------------------------------------- #
def test_higher_timeframe_join_never_uses_an_unclosed_bar():
    base = resample_ohlcv(_minutes(600), "5m")
    higher = resample_ohlcv(_minutes(600), "1h")
    higher["hourly_close"] = higher["close"]

    merged = align_higher_timeframe(
        base, higher, "1h", columns=["hourly_close", "timestamp"],
        suffix="_ctx", base_interval="5m",
    )
    assert_causal_join(merged, "timestamp_ctx", "1h", "5m")


def test_join_attaches_the_previous_hour_not_the_current_one():
    """A 5m bar at 09:05 must see the hour covering 08:00-09:00."""
    base = resample_ohlcv(_minutes(600), "5m")
    higher = resample_ohlcv(_minutes(600), "1h")
    higher["marker"] = range(len(higher))

    merged = align_higher_timeframe(
        base, higher, "1h", columns=["marker", "timestamp"], suffix="_h", base_interval="5m"
    )
    row = merged[merged["timestamp"] == pd.Timestamp("2025-01-01T09:05:00Z")]
    if row.empty:
        pytest.skip("fixture too short")
    attached = pd.Timestamp(row["timestamp_h"].iloc[0])
    assert attached == pd.Timestamp("2025-01-01T08:00:00Z"), (
        f"expected the 08:00 bar (closes 09:00), got {attached}"
    )


def test_bar_closing_exactly_at_decision_time_is_available():
    """A 5m bar stamped 08:55 decides at 09:00; the 08:00 hourly bar just closed."""
    base = resample_ohlcv(_minutes(600), "5m")
    higher = resample_ohlcv(_minutes(600), "1h")
    higher["marker"] = range(len(higher))
    merged = align_higher_timeframe(
        base, higher, "1h", columns=["marker", "timestamp"], suffix="_h", base_interval="5m"
    )
    row = merged[merged["timestamp"] == pd.Timestamp("2025-01-01T08:55:00Z")]
    if row.empty:
        pytest.skip("fixture too short")
    assert pd.Timestamp(row["timestamp_h"].iloc[0]) == pd.Timestamp("2025-01-01T08:00:00Z")


def test_assert_causal_join_catches_a_deliberate_leak():
    base = resample_ohlcv(_minutes(600), "5m")
    higher = resample_ohlcv(_minutes(600), "1h")
    leaked = pd.merge_asof(
        base.sort_values("timestamp"),
        higher[["timestamp", "close"]].sort_values("timestamp").rename(
            columns={"timestamp": "timestamp_h", "close": "close_h"}
        ),
        left_on="timestamp", right_on="timestamp_h", direction="backward",
    )
    with pytest.raises(AssertionError, match="had not closed"):
        assert_causal_join(leaked, "timestamp_h", "1h", "5m")


def test_timeframe_set_stacks_without_leak():
    tfs = TimeframeSet(_minutes(1200), market="T-EUR")
    stacked = tfs.stacked()
    assert not stacked.empty
    assert any(c.endswith("_15m") for c in stacked.columns)
    assert any(c.endswith("_1h") for c in stacked.columns)
    assert_causal_join(stacked, "timestamp_15m", "15m", "5m")
    assert_causal_join(stacked, "timestamp_1h", "1h", "5m")


def test_bars_per_holding_periods():
    assert bars_per("5m", 48 * 60) == 576
    assert bars_per("15m", 24 * 60) == 96
    assert bars_per("1h", 6 * 60) == 6


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def test_unconditional_baseline_recovers_a_known_drift():
    """A series with constant upward drift must show positive baseline returns."""
    drift = 0.00002  # per minute
    universe = {"A-EUR": _minutes(5000, drift=drift)}
    out = unconditional_forward_returns(universe, horizons_minutes=(60,), sample_every=10)
    assert len(out) == 1
    expected = np.exp(drift * 60) - 1.0
    assert out["mean_return"].iloc[0] == pytest.approx(expected, rel=0.05)
    assert out["hit_rate"].iloc[0] > 0.99


def test_matched_random_baseline_uses_the_event_markets_and_window():
    universe = {"A-EUR": _minutes(3000), "B-EUR": _minutes(3000)}
    events = pd.DataFrame(
        {
            "market": ["A-EUR"] * 20,
            "event_time": pd.date_range("2025-01-01T02:00:00Z", periods=20, freq="30min", tz="UTC"),
        }
    )
    out = matched_random_baseline(events, universe, horizons_minutes=(60,), draws_per_event=5)
    assert not out.empty
    # 20 events on A-EUR x 5 draws each; B-EUR contributes nothing because no
    # event occurred there - market composition is held fixed by construction.
    assert out["n_samples"].iloc[0] == pytest.approx(100, abs=5)


def test_conditional_lift_separates_edge_from_drift():
    """A strategy matching the baseline exactly must show zero lift."""
    event_study = pd.DataFrame(
        {"horizon_minutes": [60], "n_events": [500], "mean_return": [0.004],
         "median_return": [0.003], "hit_rate": [0.53]}
    )
    baseline = pd.DataFrame(
        {"horizon_minutes": [60], "n_samples": [10000], "mean_return": [0.004],
         "median_return": [0.003], "hit_rate": [0.53]}
    )
    out = conditional_lift(event_study, baseline, round_trip_cost_bps=77.0)
    assert out["lift"].iloc[0] == pytest.approx(0.0)
    assert not bool(out["beats_cost"].iloc[0])


def test_conditional_lift_flags_a_real_but_untradeable_edge():
    event_study = pd.DataFrame(
        {"horizon_minutes": [60], "n_events": [500], "mean_return": [0.006],
         "median_return": [0.004], "hit_rate": [0.55]}
    )
    baseline = pd.DataFrame(
        {"horizon_minutes": [60], "n_samples": [10000], "mean_return": [0.002],
         "median_return": [0.001], "hit_rate": [0.50]}
    )
    out = conditional_lift(event_study, baseline, round_trip_cost_bps=77.0)
    assert out["lift_bps"].iloc[0] == pytest.approx(40.0)
    assert not bool(out["beats_cost"].iloc[0]), "40 bps of lift cannot pay a 77 bps round trip"


def test_conditional_lift_accepts_a_tradeable_edge():
    event_study = pd.DataFrame(
        {"horizon_minutes": [60], "n_events": [500], "mean_return": [0.014],
         "median_return": [0.010], "hit_rate": [0.60]}
    )
    baseline = pd.DataFrame(
        {"horizon_minutes": [60], "n_samples": [10000], "mean_return": [0.002],
         "median_return": [0.001], "hit_rate": [0.50]}
    )
    out = conditional_lift(event_study, baseline, round_trip_cost_bps=77.0)
    assert bool(out["beats_cost"].iloc[0])


def test_lift_confidence_interval_brackets_a_known_difference():
    rng = np.random.default_rng(1)
    events = rng.normal(0.005, 0.02, 800)
    baseline = rng.normal(0.001, 0.02, 4000)
    out = lift_confidence_interval(events, baseline, iterations=500)
    assert out["ci_low"] < 0.004 < out["ci_high"]
    assert out["p_value_lift_gt_zero"] < 0.05


def test_lift_confidence_interval_finds_nothing_when_there_is_nothing():
    rng = np.random.default_rng(2)
    events = rng.normal(0.001, 0.02, 800)
    baseline = rng.normal(0.001, 0.02, 4000)
    out = lift_confidence_interval(events, baseline, iterations=500)
    assert out["ci_low"] < 0 < out["ci_high"], "no true difference must produce a CI spanning zero"


def test_buy_and_hold_benchmark_computes_drawdown():
    universe = {"BTC-EUR": _minutes(20000, drift=0.00001)}
    out = buy_and_hold_benchmark(universe, markets=("BTC-EUR",))
    assert len(out) == 1
    assert out["total_return"].iloc[0] > 0
    assert out["max_drawdown"].iloc[0] <= 0
