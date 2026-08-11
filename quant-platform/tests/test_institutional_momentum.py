"""
Tests for institutional_momentum_trend and the second dashboard section,
plus regression tests that the EXISTING production dashboard is unchanged.
Synthetic data only — no network.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from strategies.institutional_momentum import (
    compute_setup_table, momentum_quality_score, entry_quality,
    InstitutionalMomentumStrategy,
)
from research.portfolio_sim import SimConfig, vol_multiplier


def make_df(closes, volumes=None, spread=0.01):
    n = len(closes)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    c = pd.Series(closes, index=idx, dtype=float)
    o = c.shift(1).fillna(c.iloc[0])
    h = pd.concat([o, c], axis=1).max(axis=1) * (1 + spread)
    l = pd.concat([o, c], axis=1).min(axis=1) * (1 - spread)
    v = pd.Series(volumes if volumes is not None else [3e7] * n, index=idx, dtype=float)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})


def leader_with_pullback(n_trend=300, pullback=0.06):
    """Stage-2 uptrend -> controlled pullback on light volume -> EMA20 reclaim."""
    trend = list(np.linspace(20, 100, n_trend))
    peak = trend[-1]
    pull = list(np.linspace(peak, peak * (1 - pullback), 6))
    rec = list(np.linspace(pull[-1], pull[-1] * 1.03, 3))
    vols = [4e7] * n_trend + [1.2e7] * 6 + [2e7] * 3
    return make_df(trend + pull + rec, vols)


class TestSetup:
    def test_fires_on_leader_pullback(self):
        tab = compute_setup_table(leader_with_pullback())
        assert tab["setup_ok"].tail(4).any()

    def test_no_signal_in_downtrend(self):
        tab = compute_setup_table(make_df(list(np.linspace(120, 40, 300))))
        assert not tab["setup_ok"].any()

    def test_requires_positive_medium_term_momentum(self):
        """Flat/negative 60d and 120d momentum must block the setup."""
        flat = [50.0] * 300
        tab = compute_setup_table(make_df(flat))
        assert not tab["setup_ok"].any()

    def test_deep_pullback_rejected(self):
        tab = compute_setup_table(leader_with_pullback(pullback=0.20))
        assert not tab["setup_ok"].tail(4).any()

    def test_stop_never_exceeds_seven_percent(self):
        tab = compute_setup_table(leader_with_pullback())
        ok = tab[tab["setup_ok"]]
        if len(ok):
            stop_pct = 1 - ok["init_stop"] / ok["trigger"]
            assert (stop_pct <= 0.07 + 1e-9).all()

    def test_wide_buffer_rejects(self):
        tab = compute_setup_table(leader_with_pullback(), stop_atr_buffer=30.0)
        assert not tab["setup_ok"].any()

    def test_is_causal_no_lookahead(self):
        """Truncating future bars must not change earlier rows."""
        df = leader_with_pullback()
        full = compute_setup_table(df)
        cut = compute_setup_table(df.iloc[:-5])
        pd.testing.assert_series_equal(
            full["setup_ok"].iloc[:-5], cut["setup_ok"], check_names=False)

    def test_strategy_wrapper_signal_is_sane(self):
        df = leader_with_pullback()
        s = InstitutionalMomentumStrategy()
        sig = None
        for cut in range(len(df) - 4, len(df) + 1):
            sig = sig or s.generate_signal(df.iloc[:cut], "T", {})
        if sig:
            assert sig.stop_loss < sig.entry_price
            assert sig.strategy == "institutional_momentum_trend"


class TestScoring:
    def test_entry_quality_bands(self):
        assert entry_quality(101, 100) == "IDEAL"
        assert entry_quality(104, 100) == "ACCEPTABLE"
        assert entry_quality(107, 100) == "EXTENDED"
        assert entry_quality(115, 100) == "DO_NOT_CHASE"

    def test_score_rises_with_rs_and_regime(self):
        row = pd.Series({"trend_quality": 0.8, "pullback_quality": 0.9,
                         "vol_contraction": 0.7, "confirm_quality": 0.8})
        low = momentum_quality_score(row, 60, 0.5, 5e7)
        high = momentum_quality_score(row, 95, 1.0, 5e7)
        assert 0 <= low <= 100 and 0 <= high <= 100
        assert high > low

    def test_score_bounded(self):
        row = pd.Series({"trend_quality": 1, "pullback_quality": 1,
                         "vol_contraction": 1, "confirm_quality": 1})
        assert momentum_quality_score(row, 100, 1.0, 1e9) <= 100.0


class TestVolatilityScaling:
    def test_bands(self):
        cfg = SimConfig()
        assert vol_multiplier(15, cfg) == 1.0      # normal
        assert vol_multiplier(28, cfg) == 0.5      # elevated
        assert vol_multiplier(40, cfg) == 0.0      # extreme -> no new trades
        assert vol_multiplier(None, cfg) == 1.0    # unknown -> inert
        assert vol_multiplier(float("nan"), cfg) == 1.0


class TestProductionUnchanged:
    """The existing dashboard must keep working exactly as before."""

    def test_production_strategy_list_untouched(self):
        from swing_dashboard import GOOD_STRATEGIES
        assert GOOD_STRATEGIES == ["ema_pullback", "minervini_vcp", "volume_breakout"], \
            "production scanner strategy list must not change"

    def test_new_strategies_not_in_production_list(self):
        from swing_dashboard import GOOD_STRATEGIES
        assert "institutional_momentum_trend" not in GOOD_STRATEGIES
        assert "enhanced_ema_pullback" not in GOOD_STRATEGIES

    def test_new_strategy_is_registered_for_research(self):
        from strategies.registry import strategy_names, get_strategy
        assert "institutional_momentum_trend" in strategy_names()
        assert get_strategy("institutional_momentum_trend").name == "institutional_momentum_trend"

    def test_production_exit_config_untouched(self):
        from swing_dashboard import EXIT_MODE, MAX_HOLD_DAYS, MIN_HIST_PF, MIN_HIST_WINRATE
        assert EXIT_MODE == "trailing" and MAX_HOLD_DAYS is None
        assert MIN_HIST_PF == 1.2 and MIN_HIST_WINRATE == 0.50


class TestSection:
    def test_no_setups_renders_explicit_message(self):
        from reporting.institutional_section import build_section
        html, md = build_section({}, {}, "BULL_STRONG", 1.08, {}, lambda t: 30, vix=15.0)
        assert "NO QUALIFYING INSTITUTIONAL MOMENTUM SETUPS TODAY" in html
        assert any("NO QUALIFYING" in line for line in md)

    def test_bear_regime_produces_no_rows(self):
        from reporting.institutional_section import build_section
        html, _ = build_section({"T": leader_with_pullback()}, {"T": 95},
                                "BEAR_STRONG", 1.08, {"T": "Tech"}, lambda t: 30, vix=15.0)
        assert "NO QUALIFYING" in html

    def test_extreme_vix_blocks_everything(self):
        from reporting.institutional_section import build_section
        html, _ = build_section({"T": leader_with_pullback()}, {"T": 95},
                                "BULL_STRONG", 1.08, {"T": "Tech"}, lambda t: 30, vix=45.0)
        assert "NO QUALIFYING" in html

    def test_never_labels_paper_eligible_without_strategy_evidence(self):
        """A stock's own history must never earn PAPER ELIGIBLE — only
        strategy-level out-of-sample evidence can. Check the table BODY
        (the legend legitimately names every status)."""
        import reporting.institutional_section as sec
        assert sec.STRATEGY_PAPER_ELIGIBLE is False, \
            "must stay False until the backtest clears the acceptance standard"
        html, md = sec.build_section({"T": leader_with_pullback()}, {"T": 95},
                                     "BULL_STRONG", 1.08, {"T": "Tech"},
                                     lambda t: 30, vix=15.0)
        body = html.split("<tbody>")[1].split("</tbody>")[0] if "<tbody>" in html else ""
        assert "PAPER ELIGIBLE" not in body, "no row may claim PAPER ELIGIBLE yet"
        assert not any("PAPER ELIGIBLE" in line and line.startswith("|") for line in md)

    def test_word_validated_not_used_for_institutional_rows(self):
        """Spec: never reuse 'VALIDATED' for this strategy's rows."""
        import reporting.institutional_section as sec
        html, _ = sec.build_section({"T": leader_with_pullback()}, {"T": 95},
                                    "BULL_STRONG", 1.08, {"T": "Tech"},
                                    lambda t: 30, vix=15.0)
        body = html.split("<tbody>")[1].split("</tbody>")[0] if "<tbody>" in html else ""
        assert "VALIDATED" not in body.upper()
