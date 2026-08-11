"""Tests for dual_momentum_rotation selection logic. Synthetic data only."""
import numpy as np
import pandas as pd

from strategies.dual_momentum import (
    blended_momentum, select, month_end_dates, full_universe,
    RISK_UNIVERSE, CASH_PROXY,
)


def _closes(n=400, tickers=None):
    tickers = tickers or ["XLK", "XLF", "XLV", "IEF", "SHY", "BIL"]
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.0006 * (len(tickers) - i)      # XLK strongest, BIL flattest
        data[t] = 100 * np.cumprod(1 + np.full(n, drift))
    return pd.DataFrame(data, index=idx)


class TestMomentum:
    def test_blended_is_causal(self):
        c = _closes()
        full = blended_momentum(c)
        cut = blended_momentum(c.iloc[:-10])
        pd.testing.assert_frame_equal(full.iloc[:-10], cut, check_freq=False)

    def test_stronger_asset_ranks_higher(self):
        m = blended_momentum(_closes()).iloc[-1]
        assert m["XLK"] > m["XLF"] > m["XLV"]

    def test_month_end_dates_are_month_ends(self):
        idx = pd.date_range("2024-01-01", periods=200, freq="B")
        ends = month_end_dates(idx)
        assert len(ends) >= 8
        for d in ends:
            d = pd.Timestamp(d)
            nxt = d + pd.Timedelta(days=1)
            assert nxt.month != d.month or nxt not in idx


class TestSelection:
    def test_picks_top_n_equal_weight(self):
        m = blended_momentum(_closes()).iloc[-1]
        w = select(m, cash_momentum=-1.0, top_n=3)
        assert len(w) == 3
        assert all(abs(x - 1 / 3) < 1e-9 for x in w.values())
        assert "XLK" in w

    def test_absolute_momentum_veto_moves_to_defensive(self):
        """When cash outperforms everything, risk sleeves must be vetoed."""
        m = blended_momentum(_closes()).iloc[-1]
        w = select(m, cash_momentum=999.0, top_n=3)
        assert not any(t in w for t in ["XLK", "XLF", "XLV"]), \
            "risk sleeves must be vetoed when they cannot beat cash"

    def test_weights_never_exceed_one(self):
        m = blended_momentum(_closes()).iloc[-1]
        for cash in (-1.0, 0.0, 999.0):
            w = select(m, cash_momentum=cash, top_n=3)
            assert sum(w.values()) <= 1.0 + 1e-9

    def test_empty_input_is_safe(self):
        assert select(pd.Series(dtype=float), 0.0) == {}

    def test_universe_includes_cash_proxy(self):
        u = full_universe()
        assert CASH_PROXY in u
        assert all(t in u for t in RISK_UNIVERSE)


class TestProductionUnchanged:
    def test_rotation_not_in_production_scanner(self):
        from swing_dashboard import GOOD_STRATEGIES
        assert GOOD_STRATEGIES == ["ema_pullback", "minervini_vcp", "volume_breakout"]
        assert "dual_momentum_rotation" not in GOOD_STRATEGIES
