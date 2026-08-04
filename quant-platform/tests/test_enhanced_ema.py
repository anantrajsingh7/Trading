"""
Unit tests for the enhanced_ema_pullback strategy and research simulator:
signal detection, trigger entry on a later day, stop construction/rejection,
gap-through-stop handling, and trailing-stop sequencing.
All synthetic data — no network.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from strategies.enhanced_ema import compute_setup_table, EnhancedEMAPullbackStrategy
from research.portfolio_sim import simulate, SimConfig


def make_df(closes, volumes=None, spread=0.01):
    """Build a synthetic OHLCV frame from a close series."""
    n = len(closes)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    c = pd.Series(closes, index=idx, dtype=float)
    o = c.shift(1).fillna(c.iloc[0])
    h = pd.concat([o, c], axis=1).max(axis=1) * (1 + spread)
    l = pd.concat([o, c], axis=1).min(axis=1) * (1 - spread)
    v = pd.Series(volumes if volumes is not None else [2e7] * n, index=idx, dtype=float)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})


def uptrend_with_pullback(n_trend=260, pullback=0.06, recover_days=2):
    """Long uptrend to a high, 6% pullback on low volume, then reversal candle."""
    trend = list(np.linspace(20, 100, n_trend))
    peak = trend[-1]
    pull = list(np.linspace(peak, peak * (1 - pullback), 6))
    rec = list(np.linspace(pull[-1], pull[-1] * 1.02, recover_days))
    closes = trend + pull + rec
    vols = [3e7] * n_trend + [1.2e7] * 6 + [1.5e7] * recover_days
    return make_df(closes, vols)


class TestSignal:
    def test_setup_fires_on_valid_pullback(self):
        df = uptrend_with_pullback()
        tab = compute_setup_table(df, stop_atr_buffer=0.25, entry_mode="reversal")
        assert tab["setup_ok"].iloc[-3:].any(), "expected a setup after pullback+reversal"

    def test_no_setup_without_pullback(self):
        df = make_df(list(np.linspace(20, 100, 270)))  # straight up, no pullback
        tab = compute_setup_table(df)
        assert not tab["setup_ok"].iloc[-1]

    def test_no_setup_in_downtrend(self):
        df = make_df(list(np.linspace(100, 40, 270)))
        tab = compute_setup_table(df)
        assert not tab["setup_ok"].any()

    def test_deep_pullback_rejected(self):
        df = uptrend_with_pullback(pullback=0.18)   # 18% > 10% max
        tab = compute_setup_table(df)
        assert not tab["setup_ok"].iloc[-1]

    def test_wide_stop_rejected(self):
        # huge ATR buffer forces stop > 7% away → setup must be rejected
        df = uptrend_with_pullback()
        tab = compute_setup_table(df, stop_atr_buffer=25.0)
        assert not tab["setup_ok"].iloc[-3:].any()

    def test_strategy_class_returns_signal(self):
        df = uptrend_with_pullback()
        strat = EnhancedEMAPullbackStrategy()
        sig = None
        for cut in range(len(df) - 3, len(df) + 1):
            sig = sig or strat.generate_signal(df.iloc[:cut], "TEST", {})
        assert sig is not None
        assert sig.stop_loss < sig.entry_price
        assert (sig.entry_price - sig.stop_loss) / sig.entry_price <= 0.07 + 1e-9


def run_sim_on(df, trigger, stop, entry_type="stop", trail=2.5, signal_day_offset=0):
    """Wire one ticker + one signal into the simulator."""
    idx = df.index
    tab = pd.DataFrame(index=idx, data={
        "signal": False, "trigger": np.nan, "init_stop": np.nan, "atr": 1.0,
    })
    sig_day = idx[signal_day_offset]
    tab.loc[sig_day, ["signal", "trigger", "init_stop"]] = [True, trigger, stop]
    days = [d.date() for d in idx]
    regime = pd.Series("BULL_STRONG", index=days)
    rs = pd.DataFrame(99.0, index=idx, columns=["T"])
    eurusd = pd.Series(1.08, index=idx)
    earnings = {"T": [date(2030, 1, 1)]}
    cfg = SimConfig(trail_atr_mult=trail, entry_type=entry_type, costs="optimistic")
    return simulate({"T": tab}, {"T": df}, regime, rs, {"T": "Tech"}, eurusd, earnings, cfg)


class TestExecution:
    def test_entry_is_next_day_not_signal_day(self):
        closes = [100] * 10
        df = make_df(closes)
        res = run_sim_on(df, trigger=100.5, stop=95.0)
        if not res.trades.empty:
            assert res.trades.iloc[0].entry_date > df.index[0].date()

    def test_open_gap_above_trigger_rejected(self):
        # day0 close 100 → signal; next day GAPS OPEN at 105 (>2% above trigger)
        closes = [100, 100, 105, 105, 105, 105]
        df = make_df(closes, spread=0.001)
        gap_day = df.index[1]
        df.loc[gap_day, ["open", "high", "low", "close"]] = [105.0, 105.5, 104.5, 105.0]
        res = run_sim_on(df, trigger=100.5, stop=95.0)
        assert res.trades.empty
        assert "OPEN_GAP_ABOVE_TRIGGER" in set(res.rejections.reason)

    def test_gap_through_stop_exits_at_open(self):
        # enter ~101, then GAP OPEN at 80 — exit must be near 80, not at the stop
        closes = [100, 100, 101, 102, 80, 80, 80]
        df = make_df(closes, spread=0.001)
        gap_day = df.index[4]
        df.loc[gap_day, ["open", "high", "low", "close"]] = [80.0, 81.0, 79.0, 80.0]
        res = run_sim_on(df, trigger=100.5, stop=95.0)
        assert not res.trades.empty
        t = res.trades.iloc[0]
        assert t.exit_reason == "GAP_STOP"
        assert t.exit_px < 90, "gap exit must be at the gapped open, not the stop"

    def test_trailing_stop_never_falls_and_activates_next_day(self):
        # rise to 130 then collapse: trail = hi_close - 2.5*ATR(=1) ≈ 127.5
        closes = [100, 100, 101, 110, 120, 130, 90, 90]
        df = make_df(closes, spread=0.001)
        res = run_sim_on(df, trigger=100.5, stop=95.0)
        assert not res.trades.empty
        t = res.trades.iloc[0]
        # collapse day gaps open at 130→90: prior-day active stop ≈127.5 → GAP_STOP at open
        assert t.exit_reason in ("GAP_STOP", "TRAILING_STOP")
        assert t.pnl_eur > 0, "trailing stop should have locked in profit"

    def test_stop_uses_yesterdays_level_not_todays(self):
        # Day 3 closes at 120 (new trail ≈117.5) but its own intraday LOW is
        # 110. Yesterday's active stop was 95 — so the 110 low must NOT stop
        # us out. A buggy engine applying today's close-derived stop to
        # today's earlier low would exit here.
        closes = [100, 100, 101, 120, 119, 119]
        df = make_df(closes, spread=0.001)
        rally_day = df.index[3]
        df.loc[rally_day, ["open", "high", "low", "close"]] = [101.0, 121.0, 110.0, 120.0]
        res = run_sim_on(df, trigger=100.5, stop=95.0)
        assert not res.trades.empty
        assert res.trades.iloc[0].exit_reason == "END_OF_BACKTEST", \
            "position must survive: today's stop applies tomorrow, not today"


class TestPortfolioGates:
    def test_unknown_earnings_rejected(self):
        df = make_df([100] * 10)
        idx = df.index
        tab = pd.DataFrame(index=idx, data={
            "signal": False, "trigger": np.nan, "init_stop": np.nan, "atr": 1.0})
        tab.loc[idx[0], ["signal", "trigger", "init_stop"]] = [True, 100.5, 95.0]
        days = [d.date() for d in idx]
        res = simulate(
            {"T": tab}, {"T": df}, pd.Series("BULL_STRONG", index=days),
            pd.DataFrame(99.0, index=idx, columns=["T"]), {"T": "Tech"},
            pd.Series(1.08, index=idx), {"T": None},     # earnings UNKNOWN
            SimConfig(entry_type="stop", costs="optimistic"),
        )
        assert res.trades.empty
        assert "EARNINGS_UNKNOWN" in set(res.rejections.reason)

    def test_bear_regime_blocks_entries(self):
        df = make_df([100] * 10)
        idx = df.index
        tab = pd.DataFrame(index=idx, data={
            "signal": False, "trigger": np.nan, "init_stop": np.nan, "atr": 1.0})
        tab.loc[idx[0], ["signal", "trigger", "init_stop"]] = [True, 100.5, 95.0]
        days = [d.date() for d in idx]
        res = simulate(
            {"T": tab}, {"T": df}, pd.Series("BEAR_STRONG", index=days),
            pd.DataFrame(99.0, index=idx, columns=["T"]), {"T": "Tech"},
            pd.Series(1.08, index=idx), {"T": [date(2030, 1, 1)]},
            SimConfig(entry_type="stop", costs="optimistic"),
        )
        assert res.trades.empty
        assert "REGIME_BEAR" in set(res.rejections.reason)
