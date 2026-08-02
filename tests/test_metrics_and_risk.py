from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitvavo_momentum.metrics import (
    compute_trade_metrics,
    concentration_report,
    rejection_reasons,
)
from bitvavo_momentum.risk_manager import (
    OpenPosition,
    RiskLimits,
    RiskManager,
    SizingConfig,
    position_quantity,
)


def _trades(returns: list[float], markets: list[str] | None = None, start: str = "2024-01-01") -> pd.DataFrame:
    n = len(returns)
    markets = markets or ["A-EUR"] * n
    entry = pd.date_range(start, periods=n, freq="1D", tz="UTC")
    notional = 1000.0
    return pd.DataFrame(
        {
            "market": markets,
            "strategy": "S",
            "exit_policy": "P",
            "scenario": "realistic",
            "entry_time": entry,
            "exit_time": entry + pd.Timedelta(hours=2),
            "entry_price": 100.0,
            "exit_price": [100.0 * (1 + r) for r in returns],
            "quantity": notional / 100.0,
            "notional_eur": notional,
            "net_return": returns,
            "gross_return": [r + 0.002 for r in returns],
            "net_pnl_eur": [r * notional for r in returns],
            "gross_pnl_eur": [(r + 0.002) * notional for r in returns],
            "fees_eur": 2.0,
            "spread_cost_eur": 1.0,
            "slippage_cost_eur": 1.0,
            "holding_minutes": 120,
            "mfe_pct": 0.02,
            "mae_pct": -0.01,
            "ambiguous_bar": False,
            "fill_fraction": 1.0,
            "status": "closed",
        }
    )


def test_expectancy_matches_the_definition():
    returns = [0.03, -0.01, 0.03, -0.01, -0.01]
    stats = compute_trade_metrics(_trades(returns))
    win_rate = 2 / 5
    expected = win_rate * 0.03 - (1 - win_rate) * 0.01
    assert stats["net_expectancy"] == pytest.approx(expected)
    assert stats["win_rate"] == pytest.approx(win_rate)
    assert stats["payoff_ratio"] == pytest.approx(3.0)


def test_high_win_rate_with_a_large_loser_is_negative_expectancy():
    """The exact failure mode that win-rate-only selection would miss."""
    returns = [0.005] * 9 + [-0.10]
    stats = compute_trade_metrics(_trades(returns))
    assert stats["win_rate"] == pytest.approx(0.9)
    assert stats["net_expectancy"] < 0


def test_empty_trades_returns_zeros_not_errors():
    stats = compute_trade_metrics(pd.DataFrame(columns=["status", "net_return"]))
    assert stats["n_trades"] == 0
    assert np.isnan(stats["win_rate"])


def test_ratios_are_suppressed_on_tiny_samples():
    stats = compute_trade_metrics(_trades([0.01] * 5))
    assert np.isnan(stats["sharpe"]), "Sharpe on 5 trades would be noise dressed as a statistic"


def test_concentration_detects_one_coin_carrying_everything():
    returns = [0.10] + [0.0005] * 20
    markets = ["HERO-EUR"] + ["OTHER-EUR"] * 20
    report = concentration_report(_trades(returns, markets))
    assert report["top_coin"] == "HERO-EUR"
    assert report["top_coin_profit_share"] > 0.8


def test_rejection_rules_fire_on_small_samples_and_negative_stress():
    metrics = compute_trade_metrics(_trades([0.01, -0.005] * 10))
    concentration = concentration_report(_trades([0.01, -0.005] * 10))
    reasons = rejection_reasons(metrics, concentration, stress_metrics={"net_expectancy": -0.002},
                                min_trades=100)
    assert any("trades" in r for r in reasons)
    assert any("stress" in r for r in reasons)


# --------------------------------------------------------------------------- #
# risk
# --------------------------------------------------------------------------- #
def test_atr_risk_sizing_risks_the_configured_fraction():
    config = SizingConfig(method="atr_risk", risk_per_trade_pct=0.005, max_position_pct_of_equity=1.0,
                          max_participation_of_recent_volume=1.0)
    quantity, _ = position_quantity(config, equity=10_000.0, entry_price=100.0, stop_price=95.0)
    assert quantity * (100.0 - 95.0) == pytest.approx(50.0)


def test_sizing_refuses_without_a_valid_stop():
    config = SizingConfig(method="atr_risk")
    quantity, why = position_quantity(config, equity=10_000.0, entry_price=100.0, stop_price=None)
    assert quantity == 0
    assert "stop" in why


def test_position_cap_binds():
    config = SizingConfig(method="fixed_pct", fixed_pct_of_equity=0.9, max_position_pct_of_equity=0.15,
                          max_participation_of_recent_volume=1.0)
    quantity, why = position_quantity(config, equity=10_000.0, entry_price=100.0)
    assert quantity * 100.0 == pytest.approx(1500.0)
    assert "capped" in why


def test_concurrent_position_limit():
    risk = RiskManager(RiskLimits(max_concurrent_positions=2), 10_000.0)
    for i in range(2):
        risk.register_fill(OpenPosition(f"M{i}-EUR", 1.0, 100.0, 95.0, pd.Timestamp("2024-01-01", tz="UTC")))
    decision = risk.can_open("M3-EUR", pd.Timestamp("2024-01-01", tz="UTC"), 1.0, 100.0, 95.0)
    assert not decision.allowed


def test_averaging_down_is_blocked():
    risk = RiskManager(RiskLimits(allow_averaging_down=False), 10_000.0)
    risk.register_fill(OpenPosition("A-EUR", 1.0, 100.0, 95.0, pd.Timestamp("2024-01-01", tz="UTC")))
    decision = risk.can_open("A-EUR", pd.Timestamp("2024-01-01T01:00:00Z"), 1.0, 90.0, 85.0)
    assert not decision.allowed
    assert "averaging down" in decision.reasons[0]


def test_chasing_far_above_the_entry_zone_is_blocked():
    risk = RiskManager(RiskLimits(max_chase_above_entry_zone_pct=0.005), 10_000.0)
    decision = risk.can_open(
        "A-EUR", pd.Timestamp("2024-01-01", tz="UTC"), 1.0,
        entry_price=110.0, stop_price=100.0, entry_zone_high=100.0,
    )
    assert not decision.allowed


def test_open_risk_budget_scales_the_size_down():
    risk = RiskManager(RiskLimits(max_total_open_risk_pct=0.02, risk_per_trade_pct_max=0.02), 10_000.0)
    risk.register_fill(OpenPosition("A-EUR", 30.0, 100.0, 95.0, pd.Timestamp("2024-01-01", tz="UTC")))
    decision = risk.can_open("B-EUR", pd.Timestamp("2024-01-01T01:00:00Z"), 30.0, 100.0, 95.0)
    if decision.allowed:
        assert decision.scaled_quantity < 30.0
    else:
        assert "risk budget" in " ".join(decision.reasons)


def test_no_leverage_caps_at_available_cash():
    risk = RiskManager(RiskLimits(allow_leverage=False, max_exposure_per_coin_pct=1.0,
                                  max_total_open_risk_pct=1.0, risk_per_trade_pct_max=1.0), 1_000.0)
    decision = risk.can_open("A-EUR", pd.Timestamp("2024-01-01", tz="UTC"), 100.0, 100.0, 95.0)
    assert decision.allowed
    assert decision.scaled_quantity * 100.0 <= 1_000.0 + 1e-9


def test_daily_loss_limit_pauses_trading():
    risk = RiskManager(RiskLimits(max_daily_loss_pct=0.03), 10_000.0,
                       {"pause_minutes_after_trip": 60})
    # 3% of EUR 10,000 is EUR 300: the first loss stays under it, the pair does not.
    risk.register_exit("A-EUR", -200.0, pd.Timestamp("2024-01-01T10:00:00Z"))
    assert risk.paused_until is None
    risk.register_exit("B-EUR", -150.0, pd.Timestamp("2024-01-01T11:00:00Z"))
    assert risk.paused_until is not None
    decision = risk.can_open("C-EUR", pd.Timestamp("2024-01-01T11:30:00Z"), 1.0, 100.0, 95.0)
    assert not decision.allowed
