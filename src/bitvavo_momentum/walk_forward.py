"""Phase 6: chronological splitting and walk-forward validation.

Two protocols, both strictly forward-only:

**Fixed split** - earliest 50% train, next 25% validation, most recent 25% test,
separated by an embargo so no trade straddles a boundary. The test set is locked
behind :func:`unlock_test_set`, which refuses to open unless the configuration
flag is explicitly set. This is a speed bump against the most common way research
goes wrong: peeking, adjusting, re-peeking.

**Rolling walk-forward** - train 12 months, test the next 3, roll forward 3, and
repeat. Parameters are re-chosen inside every training window, so the reported
out-of-sample record is the concatenation of decisions that were made with no
knowledge of the period they were applied to.

Both split on **event time**, not on row count, so a period with more events does
not silently dominate a split.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .logging_utils import get_logger

log = get_logger(__name__)


class TestSetLocked(RuntimeError):
    """Raised when code tries to read the untouched test set without permission."""

    __test__ = False  # not a pytest class despite the name


@dataclass
class Split:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def contains(self, times: pd.Series) -> pd.Series:
        ts = pd.to_datetime(times, utc=True)
        return (ts >= self.start) & (ts < self.end)

    def __str__(self) -> str:
        return f"{self.name}[{self.start:%Y-%m-%d} .. {self.end:%Y-%m-%d})"


@dataclass
class SplitSet:
    train: Split
    validation: Split
    test: Split
    embargo_minutes: int = 0

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"split": s.name, "start_utc": s.start, "end_utc": s.end,
                 "days": (s.end - s.start).total_seconds() / 86400.0}
                for s in (self.train, self.validation, self.test)
            ]
        )


def chronological_splits(
    events: pd.DataFrame,
    train_fraction: float = 0.50,
    validation_fraction: float = 0.25,
    embargo_minutes: int = 2880,
    time_column: str = "event_time",
) -> SplitSet:
    """Split the *time axis* into train / validation / test with an embargo."""
    if events.empty:
        raise ValueError("Cannot split an empty event set")
    times = pd.to_datetime(events[time_column], utc=True)
    start, end = times.min(), times.max()
    span = end - start
    embargo = pd.Timedelta(minutes=embargo_minutes)

    train_end = start + span * train_fraction
    validation_start = train_end + embargo
    validation_end = start + span * (train_fraction + validation_fraction)
    test_start = validation_end + embargo

    splits = SplitSet(
        train=Split("train", start, train_end),
        validation=Split("validation", validation_start, validation_end),
        test=Split("test", test_start, end + pd.Timedelta(seconds=1)),
        embargo_minutes=embargo_minutes,
    )
    log.info("Splits: %s | %s | %s", splits.train, splits.validation, splits.test)
    return splits


def apply_split(events: pd.DataFrame, split: Split, time_column: str = "event_time") -> pd.DataFrame:
    if events.empty:
        return events
    return events[split.contains(events[time_column])].copy()


def unlock_test_set(config: dict[str, Any]) -> bool:
    """Gate on the untouched test set.

    Returns ``True`` only when ``splits.unlock_test_set`` is explicitly ``true``
    in ``config/research.yaml``. The intent is that this is flipped once, for the
    single final evaluation, and that flipping it is a visible act recorded in
    version control.
    """
    allowed = bool(config.get("splits", {}).get("unlock_test_set", False))
    if not allowed:
        log.warning(
            "Test set is LOCKED (splits.unlock_test_set=false). "
            "Design and select strategies on train/validation only."
        )
    return allowed


def require_test_set(config: dict[str, Any]) -> None:
    if not unlock_test_set(config):
        raise TestSetLocked(
            "The final test set is locked. Set splits.unlock_test_set: true in "
            "config/research.yaml when you are ready to evaluate ONCE."
        )


# --------------------------------------------------------------------------- #
# rolling walk-forward
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardWindow:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def walk_forward_windows(
    events: pd.DataFrame,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    embargo_minutes: int = 2880,
    time_column: str = "event_time",
) -> list[WalkForwardWindow]:
    """Generate rolling (train, test) windows across the available history."""
    if events.empty:
        return []
    times = pd.to_datetime(events[time_column], utc=True)
    start, end = times.min(), times.max()
    embargo = pd.Timedelta(minutes=embargo_minutes)

    windows: list[WalkForwardWindow] = []
    train_start = start
    index = 0
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end + embargo
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_start >= end:
            break
        windows.append(
            WalkForwardWindow(index, train_start, train_end, test_start, min(test_end, end))
        )
        index += 1
        train_start = train_start + pd.DateOffset(months=step_months)
    log.info("Generated %d walk-forward windows", len(windows))
    return windows


def run_walk_forward(
    events: pd.DataFrame,
    windows: list[WalkForwardWindow],
    select_on_train: Callable[[pd.DataFrame], Any],
    evaluate_on_test: Callable[[pd.DataFrame, Any], dict[str, Any]],
    min_train_events: int = 100,
    time_column: str = "event_time",
) -> pd.DataFrame:
    """Execute the walk-forward loop.

    ``select_on_train`` chooses a configuration using only the training slice;
    ``evaluate_on_test`` applies that exact configuration to the test slice. The
    returned frame is the concatenated out-of-sample record.
    """
    rows: list[dict[str, Any]] = []
    times = pd.to_datetime(events[time_column], utc=True)
    for window in windows:
        train_mask = (times >= window.train_start) & (times < window.train_end)
        test_mask = (times >= window.test_start) & (times < window.test_end)
        train_events = events[train_mask]
        test_events = events[test_mask]

        row = window.to_dict()
        row["n_train_events"] = int(len(train_events))
        row["n_test_events"] = int(len(test_events))

        if len(train_events) < min_train_events:
            row["status"] = f"skipped: only {len(train_events)} training events"
            rows.append(row)
            continue
        if test_events.empty:
            row["status"] = "skipped: no test events"
            rows.append(row)
            continue

        try:
            selection = select_on_train(train_events)
            metrics = evaluate_on_test(test_events, selection)
            row["status"] = "ok"
            row["selection"] = str(selection)
            row.update(metrics)
        except Exception as exc:  # one bad window must not kill the study
            log.exception("Walk-forward window %d failed", window.index)
            row["status"] = f"error: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_walk_forward(results: pd.DataFrame) -> dict[str, Any]:
    """Summarise the out-of-sample record across windows."""
    if results.empty:
        return {"n_windows": 0}
    ok = results[results.get("status") == "ok"] if "status" in results.columns else results
    if ok.empty:
        return {"n_windows": int(len(results)), "n_ok_windows": 0}
    out: dict[str, Any] = {
        "n_windows": int(len(results)),
        "n_ok_windows": int(len(ok)),
        "total_test_trades": int(ok.get("n_trades", pd.Series(dtype=float)).sum()),
    }
    for column in ("net_expectancy", "win_rate", "profit_factor", "total_net_pnl_eur", "max_drawdown"):
        if column in ok.columns:
            series = ok[column].astype(float)
            out[f"mean_{column}"] = float(series.mean())
            out[f"median_{column}"] = float(series.median())
            out[f"worst_{column}"] = float(series.min())
    if "net_expectancy" in ok.columns:
        positive = (ok["net_expectancy"].astype(float) > 0).sum()
        out["share_windows_positive"] = float(positive / len(ok))
    return out
