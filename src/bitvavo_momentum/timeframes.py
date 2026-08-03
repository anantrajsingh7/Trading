"""Multi-timeframe alignment.

The research design uses three timeframes:

* **5-minute** - execution and stop/target simulation;
* **15-minute** - setup identification;
* **1-hour** - trend and regime filters.

The trap this module exists to prevent
--------------------------------------
Naively joining a 1-hour indicator onto a 5-minute bar leaks the future. The
09:00 hourly bar is not *known* until 10:00; attaching it to the 09:05 bar hands
the strategy 55 minutes of hindsight, and it will look brilliant in backtest and
fail live.

:func:`align_higher_timeframe` therefore stamps every higher-timeframe row with
its **close** time and joins backwards only, so a 5-minute bar at 09:05 sees the
hourly bar that closed at 09:00 (covering 08:00-09:00) and nothing newer.

All resampling is done locally from the 1-minute archive, so a 15-minute bar is
by construction the aggregate of the 1-minute bars used everywhere else.
"""

from __future__ import annotations

import pandas as pd

from .logging_utils import get_logger
from .timeutils import interval_to_minutes

log = get_logger(__name__)

EXECUTION_TF = "5m"
SETUP_TF = "15m"
CONTEXT_TF = "1h"


def resample_ohlcv(
    candles: pd.DataFrame,
    target_interval: str,
    min_source_bars: int = 1,
) -> pd.DataFrame:
    """Aggregate finer candles into ``target_interval``.

    Buckets with no underlying trades are dropped rather than forward-filled: a
    Bitvavo gap means no trades occurred, and inventing a flat bar there would
    manufacture tradeable liquidity that never existed.

    ``n_source_bars`` is retained so downstream code can tell a full 15-minute
    bar (15 one-minute bars) from one built from two prints.
    """
    if candles.empty:
        return candles

    minutes = interval_to_minutes(target_interval)
    data = candles.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()

    grouped = data.resample(f"{minutes}min", label="left", closed="left")
    out = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    out["n_source_bars"] = grouped["close"].count()
    out = out[out["n_source_bars"] >= min_source_bars]
    out["interval_minutes"] = minutes
    out.index.name = "timestamp"
    return out.reset_index()


def add_close_time(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Attach ``close_time`` = bar open + interval.

    This is the instant the bar becomes usable. Every cross-timeframe join keys
    on it rather than on the open timestamp.
    """
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["close_time"] = out["timestamp"] + pd.Timedelta(minutes=interval_to_minutes(interval))
    return out


def align_higher_timeframe(
    base: pd.DataFrame,
    higher: pd.DataFrame,
    higher_interval: str,
    columns: list[str] | None = None,
    suffix: str | None = None,
    base_interval: str | None = None,
) -> pd.DataFrame:
    """Join higher-timeframe columns onto a base frame without look-ahead.

    Parameters
    ----------
    base
        Finer-timeframe frame with a ``timestamp`` column (bar open).
    higher
        Coarser-timeframe frame with a ``timestamp`` column (bar open).
    higher_interval
        Interval of ``higher``; used to compute when its bars close.
    base_interval
        Interval of ``base``. When given, the join is evaluated at the base
        bar's own **close** time, matching the convention that a decision made
        on bar ``T`` uses information available at ``T + interval``.

    Returns
    -------
    A copy of ``base`` with the requested higher-timeframe columns attached,
    each carrying only values from bars that had already closed.
    """
    if base.empty or higher.empty:
        return base

    left = base.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    decision_time = left["timestamp"]
    if base_interval is not None:
        decision_time = decision_time + pd.Timedelta(minutes=interval_to_minutes(base_interval))
    left["_decision_time"] = decision_time

    right = add_close_time(higher, higher_interval)
    keep = columns or [c for c in right.columns if c not in ("timestamp", "close_time")]
    right = right[["close_time", *keep]].copy()

    suffix = suffix if suffix is not None else f"_{higher_interval}"
    right = right.rename(columns={c: f"{c}{suffix}" for c in keep})

    merged = pd.merge_asof(
        left.sort_values("_decision_time"),
        right.sort_values("close_time"),
        left_on="_decision_time",
        right_on="close_time",
        direction="backward",
        allow_exact_matches=True,   # a bar closing exactly at the decision time IS known
    )
    merged = merged.drop(columns=["_decision_time", "close_time"], errors="ignore")
    return merged.sort_values("timestamp").reset_index(drop=True)


class TimeframeSet:
    """The three research timeframes for one market, built from 1-minute data."""

    def __init__(
        self,
        candles_1m: pd.DataFrame,
        execution_tf: str = EXECUTION_TF,
        setup_tf: str = SETUP_TF,
        context_tf: str = CONTEXT_TF,
        market: str = "",
    ) -> None:
        self.market = market
        self.execution_tf = execution_tf
        self.setup_tf = setup_tf
        self.context_tf = context_tf
        self.base = candles_1m
        self.execution = resample_ohlcv(candles_1m, execution_tf)
        self.setup = resample_ohlcv(candles_1m, setup_tf)
        self.context = resample_ohlcv(candles_1m, context_tf)

    def coverage(self) -> dict[str, int]:
        return {
            "base_1m": len(self.base),
            self.execution_tf: len(self.execution),
            self.setup_tf: len(self.setup),
            self.context_tf: len(self.context),
        }

    def completeness(self) -> dict[str, float]:
        """Fraction of each higher bar that was actually traded.

        A 15-minute bar built from 3 one-minute prints is not the same object as
        one built from 15, and strategies that key on candle shape need to know.
        """
        out: dict[str, float] = {}
        for name, frame, interval in (
            (self.execution_tf, self.execution, self.execution_tf),
            (self.setup_tf, self.setup, self.setup_tf),
            (self.context_tf, self.context, self.context_tf),
        ):
            if frame.empty:
                out[name] = float("nan")
                continue
            expected = interval_to_minutes(interval)
            out[name] = float((frame["n_source_bars"] / expected).mean())
        return out

    def stacked(
        self,
        setup_columns: list[str] | None = None,
        context_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Execution-timeframe frame with setup and context columns joined causally."""
        frame = self.execution
        if frame.empty:
            return frame
        if not self.setup.empty:
            frame = align_higher_timeframe(
                frame, self.setup, self.setup_tf, setup_columns,
                base_interval=self.execution_tf,
            )
        if not self.context.empty:
            frame = align_higher_timeframe(
                frame, self.context, self.context_tf, context_columns,
                base_interval=self.execution_tf,
            )
        return frame


def bars_per(interval: str, minutes: int) -> int:
    """How many ``interval`` bars fit in ``minutes``. Used for holding-period caps."""
    return max(1, int(round(minutes / interval_to_minutes(interval))))


def assert_causal_join(
    merged: pd.DataFrame,
    higher_timestamp_column: str,
    higher_interval: str,
    base_interval: str,
    timestamp_column: str = "timestamp",
) -> None:
    """Raise if any joined higher-timeframe bar had not closed by decision time.

    Called by the tests; cheap enough to call in research code when a new join is
    introduced.
    """
    if merged.empty or higher_timestamp_column not in merged.columns:
        return
    decision = pd.to_datetime(merged[timestamp_column], utc=True) + pd.Timedelta(
        minutes=interval_to_minutes(base_interval)
    )
    higher_close = pd.to_datetime(merged[higher_timestamp_column], utc=True) + pd.Timedelta(
        minutes=interval_to_minutes(higher_interval)
    )
    violations = (higher_close > decision) & higher_close.notna()
    if bool(violations.any()):
        first = merged.loc[violations, [timestamp_column, higher_timestamp_column]].head(3)
        raise AssertionError(
            f"{int(violations.sum())} rows use a higher-timeframe bar that had not closed. "
            f"First offenders:\n{first}"
        )


def resample_universe(
    candles_by_market: dict[str, pd.DataFrame],
    intervals: tuple[str, ...] = (EXECUTION_TF, SETUP_TF, CONTEXT_TF),
) -> dict[str, dict[str, pd.DataFrame]]:
    """Resample every market to every requested interval."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for market, frame in candles_by_market.items():
        if frame is None or frame.empty:
            continue
        out[market] = {interval: resample_ohlcv(frame, interval) for interval in intervals}
    total = sum(len(v[intervals[0]]) for v in out.values()) if out else 0
    log.info("Resampled %d markets to %s (%d execution bars)", len(out), ", ".join(intervals), total)
    return out


def daily_from_minutes(candles_1m: pd.DataFrame) -> pd.DataFrame:
    """UTC daily bars - the input to regime classification."""
    return resample_ohlcv(candles_1m, "1d")


def summarise_timeframes(candles_by_market: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-market bar counts and completeness at each research timeframe."""
    rows = []
    for market, frame in candles_by_market.items():
        if frame is None or frame.empty:
            continue
        tfs = TimeframeSet(frame, market=market)
        row: dict[str, object] = {"market": market}
        row.update(tfs.coverage())
        row.update({f"completeness_{k}": v for k, v in tfs.completeness().items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values("market").reset_index(drop=True) if not frame.empty else frame


__all__ = [
    "CONTEXT_TF",
    "EXECUTION_TF",
    "SETUP_TF",
    "TimeframeSet",
    "add_close_time",
    "align_higher_timeframe",
    "assert_causal_join",
    "bars_per",
    "daily_from_minutes",
    "resample_ohlcv",
    "resample_universe",
    "summarise_timeframes",
]
