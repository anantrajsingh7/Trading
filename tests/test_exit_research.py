"""Exit-attribution tests.

Two properties carry the module. First, the outcome measurements must describe
the future *only* as an outcome - the entry price has to be the bar after the
signal, never the signal bar itself, or every number is inflated by one bar of
hindsight. Second, the excursion counts must be true reachability counts,
because the whole point of the stage-1 verdict is that a take-profit nobody
reached tells you nothing about the entry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitvavo_momentum.exit_research import (
    excursion_table,
    exit_reason_breakdown,
    forward_return_table,
    signal_outcomes,
    stage_one_verdict,
)


def _frame(closes: list[float], start: str = "2025-01-01") -> pd.DataFrame:
    """OHLC where open == previous close and high/low bracket the bar exactly."""
    index = pd.date_range(start, periods=len(closes), freq="15min", tz="UTC")
    close = np.asarray(closes, dtype="float64")
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, close),
            "low": np.minimum(opens, close),
            "close": close,
            "volume": np.full(len(close), 1000.0),
        },
        index=index,
    )


def _signals(frame: pd.DataFrame, positions: list[int], spec: str = "S1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": "AAA-EUR",
            "event_spec": spec,
            "event_time": [frame.index[p] for p in positions],
        }
    )


# --------------------------------------------------------------------------- #
# causality
# --------------------------------------------------------------------------- #
def test_entry_is_the_bar_after_the_signal_not_the_signal_bar():
    """A signal at bar i must be filled at open[i+1], which equals close[i].

    If the entry were taken at open[i] the study would be buying before the bar
    that produced the signal had finished - one bar of free information, and the
    single easiest way to fake an edge.
    """
    closes = [100.0, 110.0, 121.0] + [121.0] * 20
    frame = _frame(closes)
    outcomes = signal_outcomes(_signals(frame, [1]), {"AAA-EUR": frame},
                               horizons_minutes=(60,), interval="15m",
                               max_holding_minutes=60)
    assert len(outcomes) == 1
    # open[2] == close[1] == 110. Buying at open[1] (== 100) would be look-ahead.
    assert outcomes["entry_price"].iloc[0] == pytest.approx(110.0)


def test_horizon_return_uses_the_close_that_many_bars_later():
    closes = [100.0] * 3 + [100.0, 101.0, 102.0, 103.0, 104.0] + [104.0] * 10
    frame = _frame(closes)
    outcomes = signal_outcomes(_signals(frame, [3]), {"AAA-EUR": frame},
                               horizons_minutes=(60,), interval="15m",
                               max_holding_minutes=60)
    entry = outcomes["entry_price"].iloc[0]          # open[4] == close[3] == 100
    expected = frame["close"].iloc[4 + 4] / entry - 1.0
    assert outcomes["fwd_return_60m"].iloc[0] == pytest.approx(expected)


def test_signals_without_a_full_forward_window_are_nan_not_truncated():
    """Clamping the exit to the last bar reports a short hold as a full one.

    In a falling period that systematically flatters late signals, because the
    truncated window omits the part of the decline that had not happened yet.
    """
    frame = _frame([100.0] * 30)
    late = _signals(frame, [28])
    outcomes = signal_outcomes(late, {"AAA-EUR": frame}, horizons_minutes=(2880,),
                               interval="15m", max_holding_minutes=2880)
    assert outcomes.empty or outcomes["fwd_return_2880m"].isna().all()


def test_outcomes_do_not_change_when_later_signals_are_added():
    """Each signal's outcome is a property of its own forward window."""
    frame = _frame(list(100.0 + np.arange(60) * 0.5))
    few = signal_outcomes(_signals(frame, [5]), {"AAA-EUR": frame},
                          horizons_minutes=(60, 240), interval="15m", max_holding_minutes=240)
    many = signal_outcomes(_signals(frame, [5, 20, 30]), {"AAA-EUR": frame},
                           horizons_minutes=(60, 240), interval="15m", max_holding_minutes=240)
    assert many.iloc[0]["fwd_return_240m"] == pytest.approx(few.iloc[0]["fwd_return_240m"])


# --------------------------------------------------------------------------- #
# excursions and reachability
# --------------------------------------------------------------------------- #
def test_reach_up_counts_a_level_the_price_touched():
    # Rises 12% within the window, then falls back to flat.
    closes = [100.0, 100.0] + [100.0 + i for i in range(1, 13)] + [100.0] * 40
    frame = _frame(closes)
    outcomes = signal_outcomes(_signals(frame, [1]), {"AAA-EUR": frame},
                               horizons_minutes=(360,), interval="15m",
                               max_holding_minutes=360)
    table = excursion_table(outcomes, levels=(0.10, 0.15), max_holding_minutes=360)
    assert table["reach_up_10pct"].iloc[0] == pytest.approx(1.0)
    assert table["reach_up_15pct"].iloc[0] == pytest.approx(0.0)


def test_reach_up_is_an_upper_bound_on_a_take_profit_hit_rate():
    """A path that peaks below the target can never produce a target exit."""
    closes = [100.0, 100.0] + [100.0 + i * 0.5 for i in range(1, 13)] + [100.0] * 40
    frame = _frame(closes)
    outcomes = signal_outcomes(_signals(frame, [1]), {"AAA-EUR": frame},
                               horizons_minutes=(360,), interval="15m",
                               max_holding_minutes=360)
    table = excursion_table(outcomes, levels=(0.10,), max_holding_minutes=360)
    assert table["reach_up_10pct"].iloc[0] == 0.0, (
        "a +10% target is unreachable on a path that peaked at +6%"
    )


def test_mae_is_negative_for_a_path_that_falls_first():
    closes = [100.0, 100.0, 95.0, 90.0, 105.0] + [105.0] * 20
    frame = _frame(closes)
    outcomes = signal_outcomes(_signals(frame, [1]), {"AAA-EUR": frame},
                               horizons_minutes=(60,), interval="15m",
                               max_holding_minutes=60)
    table = excursion_table(outcomes, levels=(0.05,), max_holding_minutes=60)
    assert table["median_mae"].iloc[0] < 0
    assert table["reach_down_5pct"].iloc[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #
def test_beats_cost_requires_clearing_the_round_trip_floor():
    """A gross mean of +0.4% against a 77 bps floor is a loss, not an edge."""
    outcomes = pd.DataFrame({
        "event_spec": ["S1"] * 200,
        "fwd_return_360m": np.full(200, 0.004),
    })
    table = forward_return_table(outcomes, (360,), round_trip_cost_bps=77.0)
    assert not table["beats_cost"].iloc[0]
    assert table["net_mean"].iloc[0] == pytest.approx(0.004 - 0.0077)

    generous = forward_return_table(outcomes, (360,), round_trip_cost_bps=20.0)
    assert generous["beats_cost"].iloc[0]


def test_verdict_picks_the_best_horizon_per_family():
    outcomes = pd.DataFrame({
        "event_spec": ["S1"] * 100 + ["S2"] * 100,
        "fwd_return_360m": np.concatenate([np.full(100, 0.02), np.full(100, -0.01)]),
        "fwd_return_2880m": np.concatenate([np.full(100, 0.005), np.full(100, -0.03)]),
    })
    forward = forward_return_table(outcomes, (360, 2880), round_trip_cost_bps=77.0)
    verdict = stage_one_verdict(forward, pd.DataFrame())
    s1 = verdict[verdict["event_spec"] == "S1"].iloc[0]
    s2 = verdict[verdict["event_spec"] == "S2"].iloc[0]
    assert s1["best_horizon_hours"] == pytest.approx(6.0)
    assert bool(s1["survives"])
    assert not bool(s2["survives"]), "a family negative at every horizon cannot survive"


def test_a_family_negative_everywhere_cannot_be_rescued_by_exits():
    """The claim stage 1 rests on, stated as a test.

    Mean forward return is the average of every path the exit rule could carve
    up. If it is below cost at every horizon, no fixed stop/target/clock triple
    has a positive *mean* to work with, so gridding exits can only find noise.
    """
    rng = np.random.default_rng(7)
    losing = rng.normal(-0.01, 0.03, 500)
    outcomes = pd.DataFrame({
        "event_spec": ["S1"] * 500,
        "fwd_return_360m": losing,
        "fwd_return_2880m": losing * 2,
    })
    verdict = stage_one_verdict(
        forward_return_table(outcomes, (360, 2880), round_trip_cost_bps=77.0), pd.DataFrame()
    )
    assert not bool(verdict["survives"].iloc[0])


# --------------------------------------------------------------------------- #
# exit reasons
# --------------------------------------------------------------------------- #
def test_exit_reason_breakdown_shares_sum_to_one_per_family():
    trades = pd.DataFrame({
        "event_spec": ["S1"] * 6 + ["S2"] * 4,
        "exit_reason": ["max_holding"] * 5 + ["take_profit"] + ["stop_loss"] * 4,
        "net_return": [-0.01] * 5 + [0.09] + [-0.02] * 4,
        "net_pnl_eur": [-5.0] * 5 + [45.0] + [-10.0] * 4,
    })
    table = exit_reason_breakdown(trades)
    assert table.groupby("event_spec")["share"].sum().round(6).eq(1.0).all()
    holding = table[(table["event_spec"] == "S1") & (table["exit_reason"] == "max_holding")]
    assert holding["share"].iloc[0] == pytest.approx(5 / 6)
    assert holding["mean_net_pct"].iloc[0] == pytest.approx(-0.01)


def test_exit_reason_breakdown_tolerates_open_trades():
    trades = pd.DataFrame({
        "event_spec": ["S1", "S1"],
        "exit_reason": ["stop_loss", None],
        "net_return": [-0.02, np.nan],
    })
    table = exit_reason_breakdown(trades)
    assert len(table) == 1
    assert table["n_trades"].iloc[0] == 1


def test_empty_inputs_return_empty_frames_not_exceptions():
    assert signal_outcomes(pd.DataFrame(), {}).empty
    assert forward_return_table(pd.DataFrame()).empty
    assert excursion_table(pd.DataFrame()).empty
    assert exit_reason_breakdown(pd.DataFrame()).empty
    assert stage_one_verdict(pd.DataFrame(), pd.DataFrame()).empty
