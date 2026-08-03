from __future__ import annotations

import numpy as np
import pandas as pd

from bitvavo_momentum.data_validator import FAIL, PASS, WARN, reindex_to_grid, validate_candles


def _frame(n: int = 1000, start: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = np.linspace(100.0, 110.0, n)
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 10.0),
        }
    )


def test_clean_dataset_passes():
    result = validate_candles(_frame(), "X-EUR", "1m", min_rows=100)
    assert result.status == PASS
    assert result.n_missing == 0
    assert result.n_duplicates == 0


def test_duplicates_are_counted_not_silently_dropped():
    frame = pd.concat([_frame(600), _frame(600).iloc[:10]], ignore_index=True)
    result = validate_candles(frame, "X-EUR", "1m", min_rows=100)
    assert result.n_duplicates == 10


def test_missing_bars_are_counted():
    frame = _frame(1000).drop(index=range(100, 150)).reset_index(drop=True)
    result = validate_candles(frame, "X-EUR", "1m", min_rows=100)
    assert result.n_missing == 50
    assert 0 < result.missing_fraction < 0.1


def test_ohlc_violation_fails():
    frame = _frame(600)
    frame.loc[10, "high"] = frame.loc[10, "low"] - 1.0
    result = validate_candles(frame, "X-EUR", "1m", min_rows=100)
    assert result.status == FAIL
    assert result.n_ohlc_violations >= 1


def test_non_positive_price_fails():
    frame = _frame(600)
    frame.loc[5, "close"] = 0.0
    result = validate_candles(frame, "X-EUR", "1m", min_rows=100)
    assert result.status == FAIL


def test_thin_market_is_warned_but_kept():
    """Bitvavo omits candles for minutes with no trades.

    An 80%-missing 1-minute series is a thin market, not corrupt data, and real
    EUR mid-caps sit in that range. Failing it here would delete exactly the
    volatile markets the momentum hypothesis is about; thin markets are instead
    excluded by the liquidity mask and the per-event look-back gate.
    """
    frame = _frame(2000).iloc[::5].reset_index(drop=True)  # 80% of bars absent
    result = validate_candles(frame, "X-EUR", "1m", min_rows=100)
    assert result.status == WARN
    assert result.is_usable
    assert "thin market" in "; ".join(result.notes)


def test_essentially_empty_dataset_fails():
    frame = _frame(20000).iloc[::99].reset_index(drop=True)  # ~99% of bars absent
    result = validate_candles(frame, "X-EUR", "1m", min_rows=100)
    assert result.status == FAIL
    assert not result.is_usable


def test_reindex_marks_gaps_without_inventing_prices():
    frame = _frame(200).drop(index=range(50, 60)).reset_index(drop=True)
    gridded = reindex_to_grid(frame, "1m", fill_prices=False)
    assert len(gridded) == 200
    assert gridded["was_missing"].sum() == 10
    # Missing bars must stay NaN: a missing Bitvavo candle means no trades.
    assert gridded.loc[gridded["was_missing"], "close"].isna().all()


def test_reindex_with_fill_still_flags_what_was_filled():
    frame = _frame(200).drop(index=range(50, 60)).reset_index(drop=True)
    gridded = reindex_to_grid(frame, "1m", fill_prices=True)
    assert gridded["close"].notna().all()
    assert gridded["was_missing"].sum() == 10
    assert (gridded.loc[gridded["was_missing"], "volume"] == 0).all()


def test_empty_dataset_fails_rather_than_passing_vacuously():
    result = validate_candles(pd.DataFrame(), "X-EUR", "1m")
    assert result.status == FAIL
    assert not result.is_usable
