"""Regime-gate tests.

The gate is the most fittable thing in this project, so the tests concentrate on
the three ways it could lie: by seeing the future, by passing when it knows
nothing, and by looking good because it rejected trades at random rather than
because it rejected bad ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitvavo_momentum.regime_gate import (
    PRESETS,
    GateConfig,
    apply_gate,
    compute_breadth,
    evaluate_gate,
    gate_comparison,
    gate_mask,
    regime_forward_returns,
)
from bitvavo_momentum.regimes import BTC_BEAR, BTC_BULL, BTC_SIDEWAYS, classify_regimes


def _daily(closes: list[float], start: str = "2025-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(closes), freq="1D", tz="UTC")
    close = np.asarray(closes, dtype="float64")
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "timestamp": index,
        "open": opens,
        "high": np.maximum(opens, close) * 1.001,
        "low": np.minimum(opens, close) * 0.999,
        "close": close,
        "volume": np.full(len(close), 1000.0),
    })


def _intraday(closes: list[float], start: str = "2025-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(closes), freq="1h", tz="UTC")
    close = np.asarray(closes, dtype="float64")
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, close),
        "low": np.minimum(opens, close),
        "close": close,
        "volume": np.full(len(close), 1000.0),
    }, index=index)


@pytest.fixture(scope="module")
def trending_regimes():
    """200 days up, then 150 down - both long enough for the EMAs to commit."""
    rng = np.random.default_rng(3)
    up = 100.0 * np.exp(np.cumsum(rng.normal(0.004, 0.02, 200)))
    down = up[-1] * np.exp(np.cumsum(rng.normal(-0.004, 0.02, 150)))
    return classify_regimes(_daily(list(up) + list(down)))


# --------------------------------------------------------------------------- #
# causality
# --------------------------------------------------------------------------- #
def test_labels_do_not_change_when_later_days_are_removed(trending_regimes):
    """A label computed today must not depend on next month's prices.

    This is the failure that makes a regime filter look magnificent in a
    backtest and useless live.
    """
    rng = np.random.default_rng(3)
    up = 100.0 * np.exp(np.cumsum(rng.normal(0.004, 0.02, 200)))
    down = up[-1] * np.exp(np.cumsum(rng.normal(-0.004, 0.02, 150)))
    full_frame = _daily(list(up) + list(down))

    full = classify_regimes(full_frame)
    truncated = classify_regimes(full_frame.iloc[:220])

    common = truncated.index
    for column in ("btc_trend", "volatility_regime"):
        a = full.loc[common, column]
        b = truncated[column]
        both_missing = a.isna() & b.isna()
        assert (a[~both_missing] == b[~both_missing]).all(), (
            f"{column} changed when future days were removed"
        )


def test_a_label_reflects_the_previous_day_not_its_own(trending_regimes):
    """classify_regimes shifts by a day; the first labelled row must lag."""
    assert trending_regimes["btc_trend"].iloc[0] is np.nan or pd.isna(
        trending_regimes["btc_trend"].iloc[0]
    ), "the first day cannot carry a label computed from itself"


def test_breadth_excludes_markets_that_had_not_listed():
    early = _intraday(list(100.0 + np.arange(400) * 0.1), start="2025-01-01")
    late = _intraday(list(100.0 + np.arange(400) * 0.1), start="2025-02-01")
    breadth = compute_breadth({"OLD-EUR": early, "NEW-EUR": late}, lookback_minutes=1440,
                              freq_minutes=60)
    assert not breadth.empty
    assert breadth.max() <= 1.0
    assert breadth.min() >= 0.0


def test_breadth_is_one_when_everything_rises_and_zero_when_everything_falls():
    up = {f"U{i}-EUR": _intraday(list(100.0 + np.arange(300) * 0.5)) for i in range(3)}
    down = {f"D{i}-EUR": _intraday(list(200.0 - np.arange(300) * 0.5)) for i in range(3)}
    assert compute_breadth(up, 1440, 60).dropna().iloc[-1] == pytest.approx(1.0)
    assert compute_breadth(down, 1440, 60).dropna().iloc[-1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# the gate refuses to pass when it knows nothing
# --------------------------------------------------------------------------- #
def test_missing_labels_are_rejected_not_waved_through():
    frame = pd.DataFrame({"btc_trend": [BTC_BULL, np.nan, BTC_BEAR]})
    mask = gate_mask(frame, GateConfig(name="bull", allowed_btc_trend=[BTC_BULL]))
    assert list(mask) == [True, False, False], (
        "a filter that passes on a missing label is not a filter"
    )


def test_a_missing_volatility_label_is_not_treated_as_calm():
    frame = pd.DataFrame({"btc_trend": [BTC_BULL] * 3,
                          "volatility_regime": ["low_vol", np.nan, "high_vol"]})
    mask = gate_mask(frame, GateConfig(name="calm", blocked_volatility=["high_vol"]))
    assert list(mask) == [True, False, False]


def test_ungated_preset_keeps_everything():
    frame = pd.DataFrame({"btc_trend": [BTC_BULL, BTC_BEAR, np.nan]})
    ungated = next(p for p in PRESETS if p.name == "ungated")
    assert gate_mask(frame, ungated).all()


def test_apply_gate_drops_signals_outside_the_allowed_regime(trending_regimes):
    days = trending_regimes.dropna(subset=["btc_trend"]).index
    signals = pd.DataFrame({
        "market": "AAA-EUR",
        "event_spec": "S1",
        "event_time": days + pd.Timedelta(hours=10),
    })
    kept = apply_gate(signals, trending_regimes, GateConfig(name="bull", allowed_btc_trend=[BTC_BULL]))
    assert 0 < len(kept) < len(signals)
    assert (kept["btc_trend"] == BTC_BULL).all()


# --------------------------------------------------------------------------- #
# the gate is scored by what it rejects
# --------------------------------------------------------------------------- #
def _outcomes_for(regimes: pd.DataFrame, by_label: dict[str, float], seed: int = 1) -> pd.DataFrame:
    """Signals whose forward return depends on the regime they fired in."""
    rng = np.random.default_rng(seed)
    days = regimes.dropna(subset=["btc_trend"]).index
    labels = regimes.loc[days, "btc_trend"]
    means = labels.map(by_label).fillna(0.0).to_numpy(dtype="float64")
    return pd.DataFrame({
        "market": "AAA-EUR",
        "event_spec": "S1",
        "event_time": days + pd.Timedelta(hours=10),
        "fwd_return_2160m": means + rng.normal(0, 0.001, len(days)),
    })


def test_separation_is_positive_when_the_gate_blocks_the_bad_regime(trending_regimes):
    outcomes = _outcomes_for(trending_regimes,
                             {BTC_BULL: 0.03, BTC_SIDEWAYS: 0.0, BTC_BEAR: -0.03})
    table = gate_comparison(outcomes, trending_regimes, "fwd_return_2160m",
                            round_trip_cost_bps=77.0)
    bull = table[table["gate"] == "bull_only"].iloc[0]
    assert bull["separation"] > 0
    assert bull["kept_gross_mean"] > bull["rejected_gross_mean"]
    assert bool(bull["beats_cost"])


def test_separation_is_near_zero_when_the_regime_does_not_matter(trending_regimes):
    """The case that must not look like success.

    Returns are identical in every regime, so the gate is discarding trades at
    random. Its kept mean is unchanged and separation is ~0 - which is exactly
    how a useless filter should read.
    """
    outcomes = _outcomes_for(trending_regimes,
                             {BTC_BULL: 0.01, BTC_SIDEWAYS: 0.01, BTC_BEAR: 0.01}, seed=7)
    table = gate_comparison(outcomes, trending_regimes, "fwd_return_2160m",
                            round_trip_cost_bps=77.0)
    bull = table[table["gate"] == "bull_only"].iloc[0]
    assert abs(bull["separation"]) < 0.002
    assert bull["kept_gross_mean"] == pytest.approx(bull["rejected_gross_mean"], abs=0.002)


def test_coverage_is_reported_so_a_tiny_sample_cannot_hide(trending_regimes):
    outcomes = _outcomes_for(trending_regimes, {BTC_BULL: 0.03, BTC_BEAR: -0.03})
    table = gate_comparison(outcomes, trending_regimes, "fwd_return_2160m")
    assert {"n_kept", "share_kept", "n_rejected"} <= set(table.columns)
    assert (table["share_kept"] <= 1.0).all()
    ungated = table[table["gate"] == "ungated"].iloc[0]
    assert ungated["share_kept"] == pytest.approx(1.0)


def test_regime_breakdown_separates_the_labels(trending_regimes):
    outcomes = _outcomes_for(trending_regimes, {BTC_BULL: 0.03, BTC_BEAR: -0.03})
    table = regime_forward_returns(outcomes, trending_regimes, "fwd_return_2160m",
                                   round_trip_cost_bps=77.0)
    assert not table.empty
    bull = table[table["regime"] == BTC_BULL]
    bear = table[table["regime"] == BTC_BEAR]
    if not bull.empty and not bear.empty:
        assert bull["gross_mean"].iloc[0] > bear["gross_mean"].iloc[0]
    assert table["share_of_signals"].sum() == pytest.approx(1.0, abs=0.01)


# --------------------------------------------------------------------------- #
# the verdict refuses to overclaim
# --------------------------------------------------------------------------- #
def test_verdict_calls_out_a_gate_that_does_not_separate(trending_regimes):
    outcomes = _outcomes_for(trending_regimes,
                             {BTC_BULL: 0.02, BTC_SIDEWAYS: 0.02, BTC_BEAR: 0.02}, seed=11)
    report = evaluate_gate(outcomes, trending_regimes, "fwd_return_2160m",
                           round_trip_cost_bps=77.0)
    verdict = report.verdict()
    assert "does not separate" in verdict or "smaller sample" in verdict


def test_verdict_reports_failure_when_nothing_clears_the_cost_floor(trending_regimes):
    outcomes = _outcomes_for(trending_regimes,
                             {BTC_BULL: 0.001, BTC_SIDEWAYS: -0.01, BTC_BEAR: -0.02})
    report = evaluate_gate(outcomes, trending_regimes, "fwd_return_2160m",
                           round_trip_cost_bps=77.0)
    assert "No gate lifted returns above" in report.verdict()


def test_verdict_never_promises_more_than_a_candidate(trending_regimes):
    """Even a gate that works perfectly on train is only ever a candidate."""
    outcomes = _outcomes_for(trending_regimes, {BTC_BULL: 0.05, BTC_BEAR: -0.05})
    verdict = evaluate_gate(outcomes, trending_regimes, "fwd_return_2160m").verdict()
    assert "candidate for validation" in verdict
    assert "not a result" in verdict


def test_a_separation_smaller_than_a_quarter_of_the_toll_is_not_a_filter(trending_regimes):
    """The boundary case, pinned.

    Kept 2.10% against rejected 2.00% is a 10 bps edge on a 77 bps toll. The sign
    is right and it would read as success on a naive comparison; it is inside
    noise and cannot change the economics.
    """
    outcomes = _outcomes_for(trending_regimes,
                             {BTC_BULL: 0.021, BTC_SIDEWAYS: 0.020, BTC_BEAR: 0.020}, seed=5)
    verdict = evaluate_gate(outcomes, trending_regimes, "fwd_return_2160m",
                            round_trip_cost_bps=77.0).verdict()
    assert "not a filter" in verdict


def test_empty_inputs_return_empty_frames_not_exceptions():
    assert compute_breadth({}).empty
    assert gate_mask(pd.DataFrame(), GateConfig()).empty
    assert gate_comparison(pd.DataFrame(), pd.DataFrame(), "fwd_return_60m").empty
    assert regime_forward_returns(pd.DataFrame(), pd.DataFrame(), "fwd_return_60m").empty
    assert "No gate could be evaluated" in evaluate_gate(
        pd.DataFrame(), pd.DataFrame(), "fwd_return_60m"
    ).verdict()
