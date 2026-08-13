"""Tests for the forward signal journal (accumulating OOS record)."""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import reporting.signal_journal as sj


@pytest.fixture(autouse=True)
def tmp_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(sj, "JOURNAL", tmp_path / "journal.csv")


def bar_df(o, h, l, c):
    idx = pd.date_range("2026-01-01", periods=1, freq="B")
    return pd.DataFrame({"open": [o], "high": [h], "low": [l],
                         "close": [c], "volume": [1e7]}, index=idx)


SIG = [dict(strategy="s", ticker="T", entry=100.0, init_stop=95.0, trail_atr=6.0)]


class TestJournal:
    def test_adds_and_persists(self):
        sj.run(SIG, {"T": bar_df(100, 101, 99, 100)}, date(2026, 1, 1))
        j = sj.load()
        assert len(j) == 1 and j.iloc[0].status == "OPEN"

    def test_no_duplicate_while_open(self):
        sj.run(SIG, {"T": bar_df(100, 101, 99, 100)}, date(2026, 1, 1))
        sj.run(SIG, {"T": bar_df(101, 102, 100, 101)}, date(2026, 1, 2))
        assert len(sj.load()) == 1, "must not re-add an already-open signal"

    def test_stop_hit_closes_with_negative_r(self):
        sj.run(SIG, {"T": bar_df(100, 101, 99, 100)}, date(2026, 1, 1))
        sj.run([], {"T": bar_df(99, 99.5, 94.0, 94.5)}, date(2026, 1, 2))
        j = sj.load()
        assert j.iloc[0].status == "CLOSED"
        assert float(j.iloc[0].r_multiple) == pytest.approx(-1.0, abs=0.01)

    def test_gap_below_stop_exits_at_open_not_stop(self):
        sj.run(SIG, {"T": bar_df(100, 101, 99, 100)}, date(2026, 1, 1))
        sj.run([], {"T": bar_df(80, 81, 79, 80)}, date(2026, 1, 2))
        j = sj.load()
        assert float(j.iloc[0].exit_price) == pytest.approx(80.0)
        assert float(j.iloc[0].r_multiple) < -1.0, "gap loss must exceed 1R"

    def test_trailing_stop_ratchets_up_never_down(self):
        sj.run(SIG, {"T": bar_df(100, 101, 99, 100)}, date(2026, 1, 1))
        sj.run([], {"T": bar_df(110, 121, 109, 120)}, date(2026, 1, 2))
        raised = float(sj.load().iloc[0].active_stop)
        assert raised == pytest.approx(114.0)          # 120 - 6
        sj.run([], {"T": bar_df(118, 119, 115, 116)}, date(2026, 1, 3))
        assert float(sj.load().iloc[0].active_stop) == pytest.approx(raised), \
            "stop must never move down"

    def test_winner_records_positive_r(self):
        sj.run(SIG, {"T": bar_df(100, 101, 99, 100)}, date(2026, 1, 1))
        sj.run([], {"T": bar_df(110, 131, 109, 130)}, date(2026, 1, 2))   # trail -> 124
        sj.run([], {"T": bar_df(126, 127, 120, 121)}, date(2026, 1, 3))   # hits 124
        j = sj.load()
        assert j.iloc[0].status == "CLOSED"
        assert float(j.iloc[0].r_multiple) == pytest.approx((124.0 - 100) / 5, abs=0.05)

    def test_scorecard_math(self):
        j = pd.DataFrame([
            {**{c: np.nan for c in sj.COLUMNS}, "status": "CLOSED", "r_multiple": 2.0},
            {**{c: np.nan for c in sj.COLUMNS}, "status": "CLOSED", "r_multiple": -1.0},
            {**{c: np.nan for c in sj.COLUMNS}, "status": "OPEN"},
        ])
        sc = sj.scorecard(j)
        assert sc["closed"] == 2 and sc["open"] == 1
        assert sc["win_rate"] == pytest.approx(0.5)
        assert sc["expectancy_r"] == pytest.approx(0.5)
        assert sc["profit_factor"] == pytest.approx(2.0)

    def test_empty_journal_is_safe(self):
        sc = sj.scorecard(sj.load())
        assert sc["closed"] == 0
        html, md = sj.render(sj.load(), sc)
        assert "Forward Signal Journal" in html and md
