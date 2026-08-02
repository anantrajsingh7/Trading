"""Data validation.

Every dataset is checked before it is allowed into the research pipeline. The
checks exist because each corresponding failure mode silently corrupts results:

* **ordering / duplicates** - a single out-of-order or duplicated bar breaks
  every rolling window computed afterwards;
* **OHLC sanity** - ``high < low`` or ``close`` outside ``[low, high]`` means the
  row cannot be trusted for stop/target simulation;
* **missing bars** - on Bitvavo a missing 1m candle means *no trades occurred*,
  not "price was unchanged". Forward-filling those bars would invent liquidity
  that did not exist. They are kept as explicit NaN gaps and counted;
* **zero-volume bars** - present in the feed but untradeable in practice;
* **stale/frozen price runs** - long constant-price stretches usually indicate a
  dead market rather than a genuine trading opportunity.

The validator never repairs data silently. It reports, and the caller decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .timeutils import expected_index, interval_to_minutes

log = get_logger(__name__)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class ValidationResult:
    market: str
    interval: str
    status: str
    n_rows: int
    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None
    n_expected: int
    n_missing: int
    missing_fraction: float
    n_duplicates: int
    n_zero_volume: int
    zero_volume_fraction: float
    n_ohlc_violations: int
    n_non_positive_prices: int
    n_negative_volume: int
    longest_gap_minutes: float
    longest_flat_run_bars: int
    max_abs_return: float
    notes: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.status != FAIL

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["notes"] = "; ".join(self.notes)
        return data


def validate_candles(
    frame: pd.DataFrame,
    market: str,
    interval: str = "1m",
    max_missing_fraction: float = 0.35,
    max_zero_volume_fraction: float = 0.60,
    max_plausible_bar_return: float = 5.0,
    min_rows: int = 500,
) -> ValidationResult:
    """Validate one OHLCV dataset. Never mutates the input."""
    notes: list[str] = []

    if frame.empty:
        return ValidationResult(
            market=market,
            interval=interval,
            status=FAIL,
            n_rows=0,
            first_timestamp=None,
            last_timestamp=None,
            n_expected=0,
            n_missing=0,
            missing_fraction=1.0,
            n_duplicates=0,
            n_zero_volume=0,
            zero_volume_fraction=0.0,
            n_ohlc_violations=0,
            n_non_positive_prices=0,
            n_negative_volume=0,
            longest_gap_minutes=0.0,
            longest_flat_run_bars=0,
            max_abs_return=0.0,
            notes=["empty dataset"],
        )

    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

    n_duplicates = int(data["timestamp"].duplicated().sum())
    if n_duplicates:
        notes.append(f"{n_duplicates} duplicate timestamps")
    data = data.drop_duplicates("timestamp", keep="last")

    if not data["timestamp"].is_monotonic_increasing:
        notes.append("timestamps were not sorted on read")
        data = data.sort_values("timestamp")
    data = data.reset_index(drop=True)

    first_ts = data["timestamp"].iloc[0]
    last_ts = data["timestamp"].iloc[-1]

    full_index = expected_index(first_ts, last_ts, interval)
    n_expected = len(full_index)
    present = set(data["timestamp"])
    n_missing = int(n_expected - len(present & set(full_index)))
    missing_fraction = n_missing / n_expected if n_expected else 0.0

    deltas = data["timestamp"].diff().dt.total_seconds().div(60.0)
    longest_gap = float(deltas.max()) if len(deltas) > 1 else 0.0
    bar_minutes = interval_to_minutes(interval)
    off_grid = int((data["timestamp"].astype("int64") % (bar_minutes * 60 * 1_000_000_000) != 0).sum())
    if off_grid:
        notes.append(f"{off_grid} timestamps not aligned to the {interval} grid")

    prices = data[["open", "high", "low", "close"]].astype("float64")
    n_non_positive = int((prices <= 0).any(axis=1).sum())
    ohlc_violation = (
        (prices["high"] < prices["low"])
        | (prices["close"] > prices["high"])
        | (prices["close"] < prices["low"])
        | (prices["open"] > prices["high"])
        | (prices["open"] < prices["low"])
    )
    n_ohlc_violations = int(ohlc_violation.sum())

    volume = data["volume"].astype("float64")
    n_negative_volume = int((volume < 0).sum())
    n_zero_volume = int((volume == 0).sum())
    zero_volume_fraction = n_zero_volume / len(data)

    close = prices["close"].replace(0.0, np.nan)
    returns = close.pct_change()
    max_abs_return = float(returns.abs().max()) if len(returns) > 1 else 0.0

    flat = close.diff().eq(0.0)
    longest_flat_run = int(
        flat.groupby((~flat).cumsum()).cumcount().add(1).where(flat, 0).max() if len(flat) else 0
    )

    status = PASS
    if n_non_positive or n_negative_volume or n_ohlc_violations:
        status = FAIL
        notes.append(
            f"{n_ohlc_violations} OHLC violations, {n_non_positive} non-positive prices, "
            f"{n_negative_volume} negative volumes"
        )
    if len(data) < min_rows:
        status = FAIL if status == FAIL else WARN
        notes.append(f"only {len(data)} rows (< {min_rows})")
    if missing_fraction > max_missing_fraction:
        status = FAIL
        notes.append(f"missing fraction {missing_fraction:.2%} above limit {max_missing_fraction:.0%}")
    elif missing_fraction > 0:
        notes.append(f"{n_missing} missing bars ({missing_fraction:.2%}) - treated as no-trade gaps")
    if zero_volume_fraction > max_zero_volume_fraction:
        status = FAIL if status != FAIL else FAIL
        notes.append(f"zero-volume fraction {zero_volume_fraction:.2%} - market effectively dead")
    if max_abs_return > max_plausible_bar_return:
        status = WARN if status == PASS else status
        notes.append(
            f"largest single-bar return {max_abs_return:.1%} - inspect for bad print or thin book"
        )
    if longest_flat_run > 1440:
        status = WARN if status == PASS else status
        notes.append(f"price frozen for {longest_flat_run} consecutive bars")

    result = ValidationResult(
        market=market,
        interval=interval,
        status=status,
        n_rows=int(len(data)),
        first_timestamp=first_ts,
        last_timestamp=last_ts,
        n_expected=n_expected,
        n_missing=n_missing,
        missing_fraction=float(missing_fraction),
        n_duplicates=n_duplicates,
        n_zero_volume=n_zero_volume,
        zero_volume_fraction=float(zero_volume_fraction),
        n_ohlc_violations=n_ohlc_violations,
        n_non_positive_prices=n_non_positive,
        n_negative_volume=n_negative_volume,
        longest_gap_minutes=longest_gap if np.isfinite(longest_gap) else 0.0,
        longest_flat_run_bars=longest_flat_run,
        max_abs_return=max_abs_return,
        notes=notes,
    )
    log.debug("Validation %s %s -> %s (%s)", market, interval, status, "; ".join(notes) or "clean")
    return result


def reindex_to_grid(
    frame: pd.DataFrame,
    interval: str = "1m",
    fill_prices: bool = False,
) -> pd.DataFrame:
    """Place candles on the complete time grid, keeping gaps explicit.

    ``fill_prices=False`` (default) leaves missing bars as NaN so downstream code
    can decide, per feature, whether a no-trade period is meaningful.

    ``fill_prices=True`` forward-fills OHLC to the last close and sets volume to
    zero. Use it only where a continuous price path is genuinely required (e.g.
    marking an open position to market); never for entry-signal generation, and
    a ``was_missing`` flag is always attached so the distinction survives.
    """
    if frame.empty:
        return frame
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
    grid = expected_index(data.index[0], data.index[-1], interval)
    out = data.reindex(grid)
    out["was_missing"] = out["close"].isna()
    if fill_prices:
        out["close"] = out["close"].ffill()
        for col in ("open", "high", "low"):
            out[col] = out[col].fillna(out["close"])
        out["volume"] = out["volume"].fillna(0.0)
    out.index.name = "timestamp"
    return out.reset_index()


def validation_frame(results: list[ValidationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])
