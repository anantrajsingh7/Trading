from __future__ import annotations

import pytest

from bitvavo_momentum.execution_model import BPS, ExecutionModel, ExecutionScenario


@pytest.fixture
def scenario() -> ExecutionScenario:
    return ExecutionScenario(
        name="test", half_spread_bps=10.0, slippage_base_bps=5.0,
        slippage_vol_coefficient=0.0, slippage_illiquidity_coefficient=0.0,
        fill_probability_market=1.0, fill_probability_limit=1.0,
        partial_fill_probability=0.0, taker_fee_bps=25.0, maker_fee_bps=15.0,
        max_participation_of_bar_volume=1.0,
    )


def test_buy_pays_spread_and_slippage(scenario):
    model = ExecutionModel(scenario, seed=1)
    fill = model.execute_market_order(100.0, 1.0, "buy", bar_volume=1e9)
    assert fill.filled
    assert fill.price == pytest.approx(100.0 * (1 + 15 * BPS))
    assert fill.fee_eur == pytest.approx(fill.notional_eur * 25 * BPS)


def test_sell_receives_less_than_the_reference_price(scenario):
    model = ExecutionModel(scenario, seed=1)
    fill = model.execute_market_order(100.0, 1.0, "sell", bar_volume=1e9)
    assert fill.price == pytest.approx(100.0 * (1 - 15 * BPS))


def test_round_trip_at_a_flat_price_loses_money(scenario):
    """The fundamental sanity check: costs must be real."""
    model = ExecutionModel(scenario, seed=1)
    buy = model.execute_market_order(100.0, 1.0, "buy", bar_volume=1e9)
    sell = model.execute_market_order(100.0, buy.quantity, "sell", bar_volume=1e9)
    pnl = (sell.notional_eur - sell.fee_eur) - (buy.notional_eur + buy.fee_eur)
    assert pnl < 0
    # 2 x (25 bps fee + 15 bps spread+slip) = 80 bps of a EUR 100 notional.
    assert pnl == pytest.approx(-0.80, abs=0.02)


def test_capacity_truncates_the_fill(scenario):
    scenario.max_participation_of_bar_volume = 0.05
    model = ExecutionModel(scenario, seed=1)
    fill = model.execute_market_order(100.0, 10.0, "buy", bar_volume=100.0)
    assert fill.quantity == pytest.approx(5.0)
    assert "capacity" in fill.reason


def test_order_below_minimum_is_rejected(scenario):
    model = ExecutionModel(scenario, seed=1)
    fill = model.execute_market_order(100.0, 0.001, "buy", bar_volume=1e9, min_order_quote_eur=5.0)
    assert not fill.filled
    assert "minimum" in fill.reason


def test_limit_order_not_touched_is_not_filled(scenario):
    model = ExecutionModel(scenario, seed=1)
    fill = model.execute_limit_order(99.0, 1.0, "buy", bar_low=99.5, bar_high=101.0)
    assert not fill.filled
    assert "not touched" in fill.reason


def test_limit_fill_earns_maker_fee_and_pays_no_spread(scenario):
    model = ExecutionModel(scenario, seed=1)
    fill = model.execute_limit_order(99.0, 1.0, "buy", bar_low=98.0, bar_high=101.0, bar_volume=1e9)
    assert fill.filled
    assert fill.price == pytest.approx(99.0)
    assert fill.spread_cost_eur == 0.0
    assert fill.fee_eur == pytest.approx(99.0 * 15 * BPS)


def test_limit_can_be_touched_without_filling():
    scenario = ExecutionScenario(name="t", fill_probability_limit=0.0)
    model = ExecutionModel(scenario, seed=1)
    fill = model.execute_limit_order(99.0, 1.0, "buy", bar_low=98.0, bar_high=101.0)
    assert not fill.filled
    assert "queue" in fill.reason


def test_stop_that_gaps_fills_worse_than_the_stop(scenario):
    model = ExecutionModel(scenario, seed=1)
    # The bar opens at 90 with a stop at 95: the stop price was never available.
    fill = model.execute_stop(95.0, 1.0, bar_open=90.0, bar_low=88.0, bar_high=91.0, bar_volume=1e9)
    assert fill.filled
    assert fill.price < 95.0
    assert "gapped" in fill.reason


def test_stop_without_gap_fills_near_the_stop(scenario):
    model = ExecutionModel(scenario, seed=1)
    fill = model.execute_stop(95.0, 1.0, bar_open=99.0, bar_low=94.0, bar_high=99.5, bar_volume=1e9)
    assert fill.filled
    assert 94.0 <= fill.price <= 95.0


def test_stress_scenario_costs_more_than_realistic():
    risk_config = {
        "fees": {"fallback_taker_bps": 25.0, "fallback_maker_bps": 15.0},
        "execution_scenarios": {
            "realistic": {"half_spread_bps": 7.5, "slippage_base_bps": 6.0},
            "stress": {"half_spread_bps": 20.0, "slippage_base_bps": 15.0},
        },
    }
    realistic = ExecutionScenario.from_config("realistic", risk_config)
    stress = ExecutionScenario.from_config("stress", risk_config)
    a = ExecutionModel(realistic, seed=1).describe()["round_trip_cost_bps_minimum"]
    b = ExecutionModel(stress, seed=1).describe()["round_trip_cost_bps_minimum"]
    assert b > a


def test_account_fees_override_fallbacks():
    risk_config = {
        "fees": {"fallback_taker_bps": 25.0, "fallback_maker_bps": 15.0,
                 "use_account_fees_when_available": True},
        "execution_scenarios": {"realistic": {}},
    }
    scenario = ExecutionScenario.from_config(
        "realistic", risk_config, account_fees={"taker": 0.0015, "maker": 0.0005}
    )
    assert scenario.taker_fee_bps == pytest.approx(15.0)
    assert scenario.maker_fee_bps == pytest.approx(5.0)


def test_reset_makes_runs_reproducible(scenario):
    scenario.partial_fill_probability = 0.5
    model = ExecutionModel(scenario, seed=7)
    first = [model.execute_market_order(100.0, 1.0, "buy", bar_volume=1e9).quantity for _ in range(20)]
    model.reset()
    second = [model.execute_market_order(100.0, 1.0, "buy", bar_volume=1e9).quantity for _ in range(20)]
    assert first == second
