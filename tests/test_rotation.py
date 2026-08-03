"""Relative-strength rotation tests.

The critical group is cross-sectional look-ahead: a ranking that can see any
part of the future picks winners perfectly and is worthless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitvavo_momentum.rotation import (
    RotationConfig,
    RotationStrategy,
    build_panel,
    rank_universe,
    target_holdings,
)
from bitvavo_momentum.signal_strategies import compute_setup_features


def _market(n: int = 2000, drift: float = 0.0, seed: int = 1, start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.004, n)
    close = 100.0 * np.exp(np.cumsum(steps))
    index = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    previous = np.concatenate([[100.0], close[:-1]])
    noise = np.abs(rng.normal(0, 0.001, n)) * close
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": previous,
            "high": np.maximum(previous, close) + noise,
            "low": np.minimum(previous, close) - noise,
            "close": close,
            "volume": np.abs(rng.normal(5000, 800, n)),
        }
    )


@pytest.fixture(scope="module")
def universe():
    """Five markets with deliberately different drifts, strongest first."""
    drifts = {"AAA-EUR": 0.0012, "BBB-EUR": 0.0008, "CCC-EUR": 0.0004,
              "DDD-EUR": 0.0, "EEE-EUR": -0.0008}
    return {
        market: compute_setup_features(_market(drift=drift, seed=i), "15m")
        for i, (market, drift) in enumerate(drifts.items())
    }


# --------------------------------------------------------------------------- #
# panel and causality
# --------------------------------------------------------------------------- #
def test_panel_aligns_markets_on_a_common_grid(universe):
    panel = build_panel(universe, "close", 60)
    assert not panel.empty
    assert set(panel.columns) == set(universe)
    assert panel.index.is_monotonic_increasing


def test_panel_does_not_backfill_unlisted_markets():
    """A market that lists later must be NaN before it existed, not back-filled."""
    early = compute_setup_features(_market(n=500, seed=1, start="2025-01-01"), "15m")
    late = compute_setup_features(_market(n=500, seed=2, start="2025-03-01"), "15m")
    panel = build_panel({"OLD-EUR": early, "NEW-EUR": late}, "close", 60)
    first_rows = panel.loc[panel.index < pd.Timestamp("2025-02-01", tz="UTC"), "NEW-EUR"]
    assert first_rows.isna().all(), "a coin that had not listed must not have prices"


def test_ranking_is_unchanged_when_future_data_is_removed(universe):
    config = RotationConfig()
    panel = build_panel(universe, "close", 60)
    full = rank_universe(panel, config, 60)

    cut = panel.index[len(panel) // 2]
    truncated = rank_universe(panel[panel.index <= cut], config, 60)

    common = truncated.index
    a = full.loc[common].to_numpy(dtype="float64")
    b = truncated.to_numpy(dtype="float64")
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], equal_nan=True), (
        "cross-sectional ranking changed when future bars were removed"
    )


def test_skip_recent_excludes_the_latest_bars_from_the_signal(universe):
    """The momentum window must end before the ranking instant.

    Both tables are asserted non-empty first: comparing two all-NaN frames
    returns "equal" and would let this test pass or fail for reasons unrelated
    to the behaviour it claims to check.
    """
    panel = build_panel(universe, "close", 60)
    no_skip = rank_universe(panel, RotationConfig(skip_recent_minutes=0, min_universe_size=3), 60)
    with_skip = rank_universe(panel, RotationConfig(skip_recent_minutes=180, min_universe_size=3), 60)

    assert no_skip.notna().any().any(), "fixture produced no rankings at all"
    assert with_skip.notna().any().any(), "fixture produced no rankings at all"
    assert not no_skip.equals(with_skip), "skipping recent bars must change the ranking"


# --------------------------------------------------------------------------- #
# ranking behaviour
# --------------------------------------------------------------------------- #
def test_strongest_market_ranks_first_most_of_the_time(universe):
    panel = build_panel(universe, "close", 60)
    ranks = rank_universe(panel, RotationConfig(risk_adjust=False, min_universe_size=3), 60)
    valid = ranks.dropna(how="all")
    assert not valid.empty
    top_counts = valid.idxmin(axis=1).value_counts()
    assert top_counts.index[0] == "AAA-EUR", (
        f"the highest-drift market should rank first most often, got {top_counts.index[0]}"
    )


def test_ranking_requires_a_minimum_universe(universe):
    small = {k: universe[k] for k in list(universe)[:2]}
    panel = build_panel(small, "close", 60)
    ranks = rank_universe(panel, RotationConfig(min_universe_size=5), 60)
    assert ranks.dropna(how="all").empty, "ranking two coins is not a cross-section"


def test_positive_momentum_filter_excludes_fallers(universe):
    panel = build_panel(universe, "close", 60)
    ranks = rank_universe(panel, RotationConfig(require_positive_momentum=True,
                                                min_universe_size=1), 60)
    # EEE has negative drift; it should be ranked far less often than AAA.
    assert ranks["EEE-EUR"].notna().sum() < ranks["AAA-EUR"].notna().sum()


# --------------------------------------------------------------------------- #
# holdings and turnover
# --------------------------------------------------------------------------- #
def test_holdings_respect_top_n(universe):
    panel = build_panel(universe, "close", 60)
    ranks = rank_universe(panel, RotationConfig(top_n=2, min_universe_size=3), 60)
    holdings = target_holdings(ranks, RotationConfig(top_n=2, min_universe_size=3), 60)
    assert holdings.sum(axis=1).max() <= 2


def test_buffer_reduces_turnover(universe):
    panel = build_panel(universe, "close", 60)
    tight = RotationConfig(top_n=2, buffer_rank_multiple=1.0, min_universe_size=3)
    loose = RotationConfig(top_n=2, buffer_rank_multiple=3.0, min_universe_size=3)
    ranks_t = rank_universe(panel, tight, 60)
    ranks_l = rank_universe(panel, loose, 60)
    changes_t = (target_holdings(ranks_t, tight, 60).diff().abs().sum().sum())
    changes_l = (target_holdings(ranks_l, loose, 60).diff().abs().sum().sum())
    assert changes_l <= changes_t, "a wider retention buffer must not increase turnover"


def test_less_frequent_rebalancing_reduces_turnover(universe):
    panel = build_panel(universe, "close", 60)
    daily = RotationConfig(rebalance_minutes=24 * 60, min_universe_size=3)
    weekly = RotationConfig(rebalance_minutes=168 * 60, min_universe_size=3)
    ranks = rank_universe(panel, daily, 60)
    changes_daily = target_holdings(ranks, daily, 60).diff().abs().sum().sum()
    changes_weekly = target_holdings(ranks, weekly, 60).diff().abs().sum().sum()
    assert changes_weekly < changes_daily


def test_turnover_cost_drag_accounts_for_position_weight(universe):
    """Cost drag is a share of TOTAL capital, so it must divide by position count.

    A round trip in one of three equally-weighted slots costs the round-trip
    rate on a third of the portfolio. Charging the full rate per round trip
    overstates the drag threefold and would wrongly disqualify every
    multi-position variant.
    """
    strategy = RotationStrategy(RotationConfig(top_n=3, min_universe_size=3))
    stats = strategy.turnover(universe, round_trip_cost_bps=77.0)
    assert stats["round_trips_per_year"] > 0
    assert stats["avg_positions_held"] > 1.0

    expected = stats["round_trips_per_year"] * 0.0077 / stats["avg_positions_held"]
    assert stats["annual_cost_drag"] == pytest.approx(expected)
    assert stats["annual_cost_drag"] < stats["round_trips_per_year"] * 0.0077


def test_concentrating_into_one_position_raises_per_slot_cost(universe):
    """Holding one name means each rebalance turns the whole portfolio."""
    single = RotationStrategy(RotationConfig(top_n=1, min_universe_size=3)).turnover(universe)
    spread = RotationStrategy(RotationConfig(top_n=3, min_universe_size=3)).turnover(universe)
    if not single or not spread:
        pytest.skip("fixture produced no holdings")
    assert single["avg_positions_held"] < spread["avg_positions_held"]


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #
def test_signals_have_the_backtester_schema(universe):
    strategy = RotationStrategy(RotationConfig(top_n=3, min_universe_size=3))
    signals = strategy.generate(universe)
    if signals.empty:
        pytest.skip("no rotation signals on the fixture")
    for column in ("market", "event_time", "event_spec", "close", "atr_60m",
                   "realised_vol_60m", "volume", "session_bucket"):
        assert column in signals.columns


def test_signals_mark_entries_not_every_held_bar(universe):
    strategy = RotationStrategy(RotationConfig(top_n=3, rebalance_minutes=24 * 60,
                                               min_universe_size=3))
    signals = strategy.generate(universe)
    if signals.empty:
        pytest.skip("no rotation signals")
    # Far fewer signals than held hours: entries only.
    panel = build_panel(universe, "close", 60)
    assert len(signals) < len(panel) * 0.2


def test_signal_timestamps_exist_in_the_feature_index(universe):
    strategy = RotationStrategy(RotationConfig(top_n=3, min_universe_size=3))
    signals = strategy.generate(universe)
    if signals.empty:
        pytest.skip("no rotation signals")
    for market, group in signals.groupby("market"):
        index = universe[market].index
        assert group["event_time"].isin(index).all(), (
            "every signal must land on a real feature bar, never an interpolated one"
        )


def test_rotation_signals_backtest_end_to_end(universe):
    from bitvavo_momentum.backtester import Backtester
    from bitvavo_momentum.execution_model import ExecutionModel, ExecutionScenario
    from bitvavo_momentum.risk_manager import RiskLimits, SizingConfig
    from bitvavo_momentum.strategies import ExitPolicy, ImmediateEntry

    strategy = RotationStrategy(RotationConfig(top_n=3, min_universe_size=3))
    signals = strategy.generate(universe)
    if signals.empty:
        pytest.skip("no rotation signals")

    engine = Backtester(
        ExecutionModel(ExecutionScenario(name="t"), seed=1),
        ExitPolicy(take_profit_pct=0.10, stop_loss_pct=None, atr_stop_multiple=1.5,
                   time_stop_minutes=48 * 60),
        SizingConfig(method="fixed_eur", fixed_eur_amount=500.0,
                     max_position_pct_of_equity=1.0, max_participation_of_recent_volume=1.0),
        RiskLimits(max_concurrent_positions=3),
        starting_equity=10_000.0, interval="15m", max_holding_minutes=48 * 60,
        min_order_quote_eur=1.0,
    )
    result = engine.run(signals, universe, ImmediateEntry())
    assert len(result.trades) == len(signals)
