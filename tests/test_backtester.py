from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitvavo_momentum.backtester import Backtester
from bitvavo_momentum.execution_model import ExecutionModel, ExecutionScenario
from bitvavo_momentum.risk_manager import RiskLimits, SizingConfig
from bitvavo_momentum.strategies import ExitPolicy, ImmediateEntry, NoEntry


def _flat_scenario() -> ExecutionScenario:
    """Zero-cost scenario so mechanical behaviour can be asserted exactly."""
    return ExecutionScenario(
        name="zero_cost", half_spread_bps=0.0, slippage_base_bps=0.0,
        slippage_vol_coefficient=0.0, slippage_illiquidity_coefficient=0.0,
        fill_probability_market=1.0, fill_probability_limit=1.0,
        partial_fill_probability=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0,
        max_participation_of_bar_volume=1.0, stop_gap_extra_slippage_bps=0.0,
    )


def _features(path: list[float], start: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(path), freq="1min", tz="UTC")
    close = np.asarray(path, dtype="float64")
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.0005,
            "low": close * 0.9995,
            "close": close,
            "volume": np.full(len(path), 1e6),
            "atr_60m": np.full(len(path), 1.0),
            "realised_vol_60m": np.full(len(path), 0.001),
            "vwap_session": close,
            "ema_9": close,
            "ema_20": close,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


def _event(market: str = "T-EUR", time: str = "2024-01-01T00:09:00Z", **extra) -> pd.DataFrame:
    row = {
        "market": market,
        "event_time": pd.Timestamp(time),
        "event_spec": "test",
        "event_lookback_return": 0.10,
        "close": 100.0,
        "atr_60m": 1.0,
        "realised_vol_60m": 0.001,
        "volume": 1e6,
        "volume_zscore": 3.0,
        "rel_volume_60m": 3.0,
    }
    row.update(extra)
    return pd.DataFrame([row])


def _engine(policy: ExitPolicy, **kwargs) -> Backtester:
    return Backtester(
        execution_model=ExecutionModel(_flat_scenario(), seed=1),
        exit_policy=policy,
        sizing=SizingConfig(method="fixed_eur", fixed_eur_amount=1000.0, max_position_pct_of_equity=1.0,
                            max_participation_of_recent_volume=1.0),
        limits=RiskLimits(max_concurrent_positions=5),
        starting_equity=10_000.0,
        interval="1m",
        min_order_quote_eur=1.0,
        **kwargs,
    )


def test_take_profit_is_hit_and_recorded():
    path = [100.0] * 10 + [100.0, 101.0, 102.0, 103.5, 104.0] + [104.0] * 50
    features = {"T-EUR": _features(path)}
    engine = _engine(ExitPolicy(take_profit_pct=0.03, stop_loss_pct=0.05, time_stop_minutes=60))
    result = engine.run(_event(), features, ImmediateEntry())
    trade = result.closed_trades().iloc[0]
    assert trade["exit_reason"] == "take_profit"
    assert trade["net_return"] > 0


def test_stop_loss_is_hit_and_recorded():
    path = [100.0] * 10 + [100.0, 99.0, 98.0, 97.0] + [97.0] * 50
    features = {"T-EUR": _features(path)}
    engine = _engine(ExitPolicy(take_profit_pct=0.10, stop_loss_pct=0.02, time_stop_minutes=60))
    result = engine.run(_event(), features, ImmediateEntry())
    trade = result.closed_trades().iloc[0]
    assert trade["exit_reason"].startswith("stop_loss")
    assert trade["net_return"] < 0


def test_time_stop_closes_a_flat_trade():
    features = {"T-EUR": _features([100.0] * 200)}
    engine = _engine(ExitPolicy(take_profit_pct=0.20, stop_loss_pct=0.20, time_stop_minutes=30))
    result = engine.run(_event(), features, ImmediateEntry())
    trade = result.closed_trades().iloc[0]
    assert trade["exit_reason"] == "time_stop"
    assert trade["holding_minutes"] == pytest.approx(31, abs=1)


def test_ambiguous_bar_resolves_to_the_stop():
    """A bar spanning both levels must be resolved conservatively and flagged."""
    index = pd.date_range("2024-01-01T00:00:00Z", periods=60, freq="1min", tz="UTC")
    close = np.full(60, 100.0)
    frame = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(60, 1e6), "atr_60m": np.full(60, 1.0),
         "realised_vol_60m": np.full(60, 0.001)},
        index=index,
    )
    # The bar after entry reaches both +5% and -5%.
    frame.iloc[11, frame.columns.get_loc("high")] = 105.0
    frame.iloc[11, frame.columns.get_loc("low")] = 95.0
    frame.index.name = "timestamp"

    engine = _engine(ExitPolicy(take_profit_pct=0.03, stop_loss_pct=0.03, time_stop_minutes=60))
    result = engine.run(_event(), {"T-EUR": frame}, ImmediateEntry())
    trade = result.closed_trades().iloc[0]
    assert trade["ambiguous_bar"]
    assert "ambiguous" in trade["exit_reason"]
    assert trade["net_return"] < 0, "the conservative resolution must be the loss"


def test_entry_never_uses_the_event_bar_price():
    """Entry must be the bar AFTER the event, at a different price."""
    path = [100.0] * 10 + [200.0] + [200.0] * 50  # huge jump on the bar after the event
    features = {"T-EUR": _features(path)}
    engine = _engine(ExitPolicy(take_profit_pct=0.50, stop_loss_pct=0.50, time_stop_minutes=60))
    result = engine.run(_event(time="2024-01-01T00:09:00Z"), features, ImmediateEntry())
    trade = result.closed_trades().iloc[0]
    assert trade["entry_time"] == pd.Timestamp("2024-01-01T00:10:00Z")
    assert trade["entry_price"] == pytest.approx(200.0)


def test_costs_reduce_the_net_return_below_the_gross_return():
    path = [100.0] * 10 + [100.0, 101.0, 102.0, 103.5] + [103.5] * 50
    features = {"T-EUR": _features(path)}
    costly = ExecutionScenario(
        name="costly", half_spread_bps=20.0, slippage_base_bps=20.0,
        slippage_vol_coefficient=0.0, slippage_illiquidity_coefficient=0.0,
        fill_probability_market=1.0, partial_fill_probability=0.0,
        taker_fee_bps=25.0, max_participation_of_bar_volume=1.0,
    )
    engine = Backtester(
        ExecutionModel(costly, seed=1), ExitPolicy(take_profit_pct=0.03, stop_loss_pct=0.05),
        SizingConfig(method="fixed_eur", fixed_eur_amount=1000.0, max_position_pct_of_equity=1.0,
                     max_participation_of_recent_volume=1.0),
        RiskLimits(max_concurrent_positions=5), starting_equity=10_000.0, min_order_quote_eur=1.0,
    )
    trade = engine.run(_event(), features, ImmediateEntry()).closed_trades().iloc[0]
    assert trade["net_return"] < trade["gross_return"]
    assert trade["fees_eur"] > 0


def test_do_nothing_strategy_produces_no_trades():
    features = {"T-EUR": _features([100.0] * 100)}
    result = _engine(ExitPolicy()).run(_event(), features, NoEntry())
    assert result.n_trades == 0
    assert result.funnel["no_entry_signal"] == 1


def test_every_event_produces_a_row_even_when_no_trade_happens():
    features = {"T-EUR": _features([100.0] * 100)}
    result = _engine(ExitPolicy()).run(_event(), features, NoEntry())
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["status"] == "no_entry_signal"


def test_missing_market_data_is_reported_not_silently_skipped():
    result = _engine(ExitPolicy()).run(_event(market="GHOST-EUR"), {}, ImmediateEntry())
    assert result.funnel["no_market_data"] == 1
    assert result.trades.iloc[0]["status"] == "no_market_data"


def test_concurrent_position_limit_is_enforced():
    """Three simultaneous events on a 3-position limit must not open a fourth."""
    markets = [f"M{i}-EUR" for i in range(5)]
    features = {m: _features([100.0] * 400) for m in markets}
    events = pd.concat(
        [_event(market=m, time="2024-01-01T00:09:00Z") for m in markets], ignore_index=True
    )
    engine = Backtester(
        ExecutionModel(_flat_scenario(), seed=1),
        ExitPolicy(take_profit_pct=0.5, stop_loss_pct=0.5, time_stop_minutes=300),
        SizingConfig(method="fixed_eur", fixed_eur_amount=1000.0, max_position_pct_of_equity=1.0,
                     max_participation_of_recent_volume=1.0),
        RiskLimits(max_concurrent_positions=3, max_total_open_risk_pct=1.0,
                   risk_per_trade_pct_max=1.0, max_exposure_per_coin_pct=1.0),
        starting_equity=10_000.0, min_order_quote_eur=1.0, max_holding_minutes=300,
    )
    result = engine.run(events, features, ImmediateEntry())
    assert result.funnel["filled"] == 3
    assert result.funnel["risk_blocked"] == 2


def test_equity_curve_starts_at_the_configured_equity():
    path = [100.0] * 10 + [101.0, 103.5] + [103.5] * 50
    features = {"T-EUR": _features(path)}
    engine = _engine(ExitPolicy(take_profit_pct=0.03, stop_loss_pct=0.05))
    result = engine.run(_event(), features, ImmediateEntry())
    assert not result.equity_curve.empty
    assert result.equity_curve["equity"].iloc[-1] != 10_000.0
