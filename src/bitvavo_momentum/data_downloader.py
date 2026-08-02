"""Resumable historical OHLCV downloader.

Design notes
------------
*Fixed windows, not cursors.* Bitvavo returns at most 1440 candles per call and
orders them newest-first. Rather than depending on that ordering, the downloader
requests explicit ``[start, end)`` windows of exactly ``limit * interval``
duration and sorts locally. Ordering changes upstream therefore cannot corrupt
the archive.

*Cheap history probe.* Walking 1-minute windows from 2021 for a coin listed in
2024 would waste thousands of empty calls. The downloader first pulls daily
candles (a handful of calls covering years) to learn the market's true first
trading day, then walks the 1-minute grid from there.

*Resumable.* Existing Parquet is read first; the download restarts from the last
stored bar minus a small overlap, so an interrupted run costs one window, not a
full re-download. Overlapping bars are de-duplicated on ``timestamp``.

*Raw first.* Every payload goes to :class:`~.storage.RawStore` before parsing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from .bitvavo_client import BitvavoAPIError, BitvavoClient, BitvavoError
from .data_validator import validate_candles
from .logging_utils import get_logger
from .storage import CANDLE_COLUMNS, DatasetRecord, ParquetStore, RawStore
from .timeutils import interval_to_minutes, now_utc, to_epoch_ms, to_utc

log = get_logger(__name__)

# Re-request this many bars of overlap when resuming, so a partially written
# final bar is replaced rather than trusted.
RESUME_OVERLAP_BARS = 5


@dataclass
class DownloadOutcome:
    market: str
    interval: str
    rows_before: int
    rows_after: int
    pages: int
    api_calls: int
    started_utc: pd.Timestamp
    finished_utc: pd.Timestamp
    status: str
    message: str = ""


def candles_to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    """Convert raw ``[ts_ms, o, h, l, c, v]`` rows into a typed, sorted frame."""
    if not rows:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    frame = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


class HistoryDownloader:
    """Downloads and archives OHLCV history for a set of markets."""

    def __init__(
        self,
        client: BitvavoClient,
        raw_store: RawStore,
        parquet_store: ParquetStore,
        max_candles_per_request: int = 1440,
        sleep_between_calls: float = 0.05,
    ) -> None:
        self.client = client
        self.raw_store = raw_store
        self.store = parquet_store
        self.limit = int(max_candles_per_request)
        self.sleep_between_calls = sleep_between_calls
        # Route raw capture through the store unless the caller wired their own.
        if self.client.raw_sink is None:
            self.client.raw_sink = raw_store.as_sink()

    # -- helpers ---------------------------------------------------------------
    def probe_first_candle(self, market: str, floor_utc: pd.Timestamp) -> pd.Timestamp | None:
        """Earliest daily candle at or after ``floor_utc`` - the true listing date.

        Daily candles are used because ~1500 of them span four years, so the probe
        costs one or two API calls instead of thousands.
        """
        cursor = to_utc(floor_utc)
        end = now_utc()
        while cursor < end:
            window_end = min(end, cursor + timedelta(days=self.limit))
            try:
                rows = self.client.get_candles(
                    market,
                    interval="1d",
                    limit=self.limit,
                    start_ms=to_epoch_ms(cursor),
                    end_ms=to_epoch_ms(window_end),
                )
            except BitvavoAPIError as exc:
                log.warning("Probe failed for %s: %s", market, exc)
                return None
            frame = candles_to_frame(rows)
            if not frame.empty:
                return frame["timestamp"].iloc[0]
            cursor = window_end
        return None

    # -- main entry point ------------------------------------------------------
    def download_market(
        self,
        market: str,
        interval: str = "1m",
        history_start: str | pd.Timestamp = "2021-01-01T00:00:00Z",
        end: pd.Timestamp | None = None,
        force_full: bool = False,
        max_windows: int | None = None,
    ) -> DownloadOutcome:
        """Download (or extend) one market's history and persist it."""
        started = now_utc()
        bar_minutes = interval_to_minutes(interval)
        window_span = timedelta(minutes=bar_minutes * self.limit)
        end_ts = to_utc(end) if end is not None else now_utc()

        existing = pd.DataFrame(columns=CANDLE_COLUMNS) if force_full else self.store.read_candles(market, interval)
        rows_before = len(existing)

        if not existing.empty:
            cursor = existing["timestamp"].max() - timedelta(minutes=bar_minutes * RESUME_OVERLAP_BARS)
            log.info("%s %s: resuming from %s (%d existing rows)", market, interval, cursor, rows_before)
        else:
            first = self.probe_first_candle(market, to_utc(history_start))
            if first is None:
                finished = now_utc()
                log.warning("%s: no candles available at or after %s", market, history_start)
                return DownloadOutcome(
                    market, interval, rows_before, rows_before, 0,
                    self.client.request_count, started, finished, "NO_DATA",
                    "probe returned no candles",
                )
            cursor = max(to_utc(history_start), first)
            log.info("%s %s: first candle probed at %s", market, interval, first)

        collected: list[pd.DataFrame] = []
        pages = 0
        calls_at_start = self.client.request_count
        consecutive_empty = 0

        while cursor < end_ts:
            if max_windows is not None and pages >= max_windows:
                log.info("%s: stopping after max_windows=%d", market, max_windows)
                break
            window_end = min(end_ts, cursor + window_span)
            try:
                rows = self.client.get_candles(
                    market,
                    interval=interval,
                    limit=self.limit,
                    start_ms=to_epoch_ms(cursor),
                    end_ms=to_epoch_ms(window_end),
                )
            except BitvavoError as exc:
                finished = now_utc()
                log.error("%s %s: aborting window at %s: %s", market, interval, cursor, exc)
                if collected:
                    self._persist(market, interval, existing, collected)
                return DownloadOutcome(
                    market, interval, rows_before,
                    rows_before + sum(len(f) for f in collected), pages,
                    self.client.request_count - calls_at_start, started, finished,
                    "PARTIAL", str(exc),
                )

            frame = candles_to_frame(rows)
            pages += 1
            if frame.empty:
                consecutive_empty += 1
            else:
                consecutive_empty = 0
                collected.append(frame)
            # A long dead stretch is normal for thin markets; keep walking, but log.
            if consecutive_empty and consecutive_empty % 50 == 0:
                log.info("%s: %d consecutive empty windows at %s", market, consecutive_empty, cursor)

            cursor = window_end
            if self.sleep_between_calls:
                time.sleep(self.sleep_between_calls)

        merged = self._persist(market, interval, existing, collected)
        finished = now_utc()
        return DownloadOutcome(
            market=market,
            interval=interval,
            rows_before=rows_before,
            rows_after=len(merged),
            pages=pages,
            api_calls=self.client.request_count - calls_at_start,
            started_utc=started,
            finished_utc=finished,
            status="OK" if len(merged) else "NO_DATA",
        )

    def _persist(
        self,
        market: str,
        interval: str,
        existing: pd.DataFrame,
        collected: list[pd.DataFrame],
    ) -> pd.DataFrame:
        parts = [f for f in ([existing] if not existing.empty else []) + collected if not f.empty]
        if not parts:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        merged = (
            pd.concat(parts, ignore_index=True)
            .sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )
        self.store.write_candles(market, interval, merged)
        return merged

    # -- manifest --------------------------------------------------------------
    def record_dataset(
        self,
        market: str,
        interval: str,
        outcome: DownloadOutcome,
        max_missing_fraction: float = 0.35,
    ) -> DatasetRecord:
        """Validate what was stored and write the manifest row."""
        frame = self.store.read_candles(market, interval)
        result = validate_candles(
            frame, market=market, interval=interval, max_missing_fraction=max_missing_fraction
        )
        record = DatasetRecord(
            market=market,
            interval=interval,
            first_timestamp=result.first_timestamp,
            last_timestamp=result.last_timestamp,
            n_rows=result.n_rows,
            n_missing_intervals=result.n_missing,
            missing_fraction=result.missing_fraction,
            n_duplicate_rows=result.n_duplicates,
            n_zero_volume_rows=result.n_zero_volume,
            zero_volume_fraction=result.zero_volume_fraction,
            download_started_utc=outcome.started_utc,
            download_finished_utc=outcome.finished_utc,
            api_source=f"{self.client.base_url}/{{market}}/candles?interval={interval}",
            validation_status=result.status,
            validation_notes="; ".join(result.notes),
            n_raw_pages=outcome.pages,
            file_path=str(self.store.candle_path(market, interval)),
            extra={
                "download_status": outcome.status,
                "download_message": outcome.message,
                "api_calls": outcome.api_calls,
                "rows_added": outcome.rows_after - outcome.rows_before,
                "longest_gap_minutes": result.longest_gap_minutes,
                "max_abs_bar_return": result.max_abs_return,
                "longest_flat_run_bars": result.longest_flat_run_bars,
            },
        )
        self.store.upsert_manifest(record)
        return record

    def download_many(
        self,
        markets: list[str],
        interval: str = "1m",
        history_start: str | pd.Timestamp = "2021-01-01T00:00:00Z",
        force_full: bool = False,
        max_windows: int | None = None,
    ) -> pd.DataFrame:
        """Download several markets, recording a manifest row for each."""
        rows = []
        for i, market in enumerate(markets, start=1):
            log.info("[%d/%d] %s %s", i, len(markets), market, interval)
            try:
                outcome = self.download_market(
                    market,
                    interval=interval,
                    history_start=history_start,
                    force_full=force_full,
                    max_windows=max_windows,
                )
                record = self.record_dataset(market, interval, outcome)
                rows.append(record.to_row())
            except Exception as exc:  # one bad market must not stop the run
                log.exception("%s failed: %s", market, exc)
        return pd.DataFrame(rows)


def resample_candles(frame: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """Aggregate 1-minute candles into a coarser interval.

    Resampling locally (rather than downloading each interval separately)
    guarantees that a 15-minute bar is exactly the aggregate of the 1-minute bars
    the research uses - no silent disagreement between timeframes. Buckets with
    no underlying trades stay NaN rather than being invented.
    """
    if frame.empty:
        return frame
    minutes = interval_to_minutes(target_interval)
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.set_index("timestamp").sort_index()
    agg = data.resample(f"{minutes}min", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    agg = agg[agg["close"].notna() | agg["open"].notna()]
    return agg.reset_index()
