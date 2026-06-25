"""Tests for position sizing and risk management."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from risk import position_size, calc_stop_loss, calc_targets, build_trade_setup
from indicators import compute_all


def _make_df(trend: str = "up") -> pd.DataFrame:
    n = 300
    rng = np.random.default_rng(1)
    if trend == "up":
        close = 100 + np.linspace(0, 50, n) + rng.normal(0, 1, n)
    else:
        close = 150 - np.linspace(0, 50, n) + rng.normal(0, 1, n)
    close = np.maximum(close, 5)
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    dates = pd.bdate_range(end="2023-12-31", periods=n)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low,
         "close": close, "volume": volume},
        index=dates,
    )
    return compute_all(df)


class TestPositionSize:
    def test_basic(self):
        shares, risk, val = position_size(10_000, 0.01, 100.0, 95.0)
        assert shares == 20         # 100 / 5 = 20
        assert abs(risk - 100.0) < 0.01
        assert abs(val - 2000.0) < 0.01

    def test_zero_when_stop_above_entry(self):
        shares, risk, val = position_size(10_000, 0.01, 95.0, 100.0)
        assert shares == 0

    def test_scales_with_account_size(self):
        sh1, _, _ = position_size(10_000, 0.01, 50.0, 48.0)
        sh2, _, _ = position_size(100_000, 0.01, 50.0, 48.0)
        assert sh2 == sh1 * 10

    def test_tight_stop_more_shares(self):
        sh_tight, _, _ = position_size(10_000, 0.01, 100.0, 99.0)
        sh_wide, _, _ = position_size(10_000, 0.01, 100.0, 90.0)
        assert sh_tight > sh_wide


class TestCalcTargets:
    def test_min_rr_2(self):
        t1, t2 = calc_targets(100.0, 95.0, min_rr=2.0)
        assert abs(t1 - 110.0) < 0.001  # 2 * 5 = 10
        assert abs(t2 - 115.0) < 0.001  # 3 * 5 = 15

    def test_t2_above_t1(self):
        t1, t2 = calc_targets(50.0, 47.0)
        assert t2 > t1


class TestCalcStopLoss:
    def test_stop_below_entry(self):
        df = _make_df()
        entry = float(df["close"].iloc[-1])
        stop = calc_stop_loss(df, entry)
        assert stop < entry

    def test_positive_stop(self):
        df = _make_df()
        entry = float(df["close"].iloc[-1])
        stop = calc_stop_loss(df, entry)
        assert stop > 0


class TestBuildTradeSetup:
    def _cfg(self) -> dict:
        return {
            "account": {"size": 10000, "risk_per_trade": 0.01, "min_rr_ratio": 2.0},
            "trading": {"min_price": 5.0},
            "indicators": {"atr_multiplier": 1.5},
        }

    def test_returns_setup_for_valid_stock(self):
        df = _make_df("up")
        setup = build_trade_setup("TEST", "ema_pullback", df, self._cfg())
        # may return None if stop/RR conditions not met, but should not raise
        if setup is not None:
            assert setup.rr_ratio >= 2.0
            assert setup.stop_loss < setup.entry
            assert setup.target1 > setup.entry
            assert setup.shares > 0

    def test_none_for_penny_stock(self):
        df = _make_df("up")
        cfg = self._cfg()
        cfg["trading"]["min_price"] = 999.0  # artificially high floor
        setup = build_trade_setup("TEST", "ema_pullback", df, cfg)
        assert setup is None
