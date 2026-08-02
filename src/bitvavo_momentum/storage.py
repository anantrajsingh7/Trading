"""Storage layer: raw capture, Parquet datasets and a dataset manifest.

Layout::

    data/raw/<market>/<interval>/<endpoint>_<page_start_ms>.json.gz   # unmodified
    data/raw/reference/markets_<utc_stamp>.json                       # market rules
    data/processed/candles/<market>__<interval>.parquet               # tidy OHLCV
    data/processed/manifest.parquet                                   # dataset ledger
    data/results/...                                                  # research output

The manifest is the Phase 1 deliverable: for every (market, interval) it records
coverage, gaps, duplicates, zero-volume periods, download time, source and
validation status. Nothing downstream is allowed to consume a dataset whose
manifest row says ``validation_status == 'FAIL'``.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .logging_utils import get_logger
from .timeutils import now_utc, to_utc

log = get_logger(__name__)

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
MANIFEST_NAME = "manifest.parquet"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


@dataclass
class DatasetRecord:
    """One row of the dataset manifest (Phase 1 bookkeeping requirement)."""

    market: str
    interval: str
    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None
    n_rows: int
    n_missing_intervals: int
    missing_fraction: float
    n_duplicate_rows: int
    n_zero_volume_rows: int
    zero_volume_fraction: float
    download_started_utc: pd.Timestamp
    download_finished_utc: pd.Timestamp
    api_source: str
    validation_status: str = "UNVALIDATED"
    validation_notes: str = ""
    n_raw_pages: int = 0
    file_path: str = ""
    schema_version: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["extra"] = json.dumps(row["extra"], default=str)
        return row


class RawStore:
    """Persists unmodified API payloads before any parsing happens."""

    def __init__(self, root: Path | str, compress: bool = True):
        self.root = Path(root)
        self.compress = compress
        self.root.mkdir(parents=True, exist_ok=True)
        self.pages_written = 0

    def path_for(self, path: str, params: dict[str, Any]) -> Path:
        market = str(params.get("market") or "")
        if not market and path.startswith("/") and path.count("/") >= 2:
            market = path.strip("/").split("/")[0]
        interval = str(params.get("interval") or "na")
        endpoint = path.strip("/").split("/")[-1] or "root"
        start = params.get("start") or params.get("end") or "latest"
        folder = self.root / (_safe_name(market) if market else "reference") / _safe_name(interval)
        folder.mkdir(parents=True, exist_ok=True)
        suffix = ".json.gz" if self.compress else ".json"
        return folder / f"{_safe_name(endpoint)}_{_safe_name(str(start))}{suffix}"

    def write(self, path: str, params: dict[str, Any], payload: Any) -> Path:
        target = self.path_for(path, params)
        blob = json.dumps(
            {
                "captured_utc": now_utc().isoformat(),
                "endpoint": path,
                "params": params,
                "payload": payload,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        if self.compress:
            with gzip.open(target, "wb") as handle:
                handle.write(blob)
        else:
            target.write_bytes(blob)
        self.pages_written += 1
        return target

    def as_sink(self):
        """Return a callable suitable for ``BitvavoClient(raw_sink=...)``."""

        def _sink(path: str, params: dict[str, Any], payload: Any) -> None:
            self.write(path, params, payload)

        return _sink

    def read_all(self, market: str, interval: str) -> list[Any]:
        """Re-read every captured page for a (market, interval) pair."""
        folder = self.root / _safe_name(market) / _safe_name(interval)
        if not folder.exists():
            return []
        out: list[Any] = []
        for file in sorted(folder.iterdir()):
            try:
                if file.suffix == ".gz":
                    with gzip.open(file, "rb") as handle:
                        out.append(json.loads(handle.read().decode("utf-8")))
                elif file.suffix == ".json":
                    out.append(json.loads(file.read_text(encoding="utf-8")))
            except Exception:
                log.exception("Could not read raw page %s", file)
        return out


class ParquetStore:
    """Tidy, validated datasets plus the manifest ledger."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.candles_dir = self.root / "candles"
        self.candles_dir.mkdir(parents=True, exist_ok=True)

    # -- candles ---------------------------------------------------------------
    def candle_path(self, market: str, interval: str) -> Path:
        return self.candles_dir / f"{_safe_name(market)}__{_safe_name(interval)}.parquet"

    def write_candles(self, market: str, interval: str, frame: pd.DataFrame) -> Path:
        if frame.empty:
            log.warning("Refusing to write empty candle frame for %s %s", market, interval)
            return self.candle_path(market, interval)
        out = frame.copy()
        if "timestamp" not in out.columns:
            out = out.reset_index().rename(columns={"index": "timestamp"})
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        target = self.candle_path(market, interval)
        out.to_parquet(target, index=False, compression="zstd")
        return target

    def read_candles(
        self,
        market: str,
        interval: str,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> pd.DataFrame:
        path = self.candle_path(market, interval)
        if not path.exists():
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        if start is not None:
            frame = frame[frame["timestamp"] >= to_utc(start)]
        if end is not None:
            frame = frame[frame["timestamp"] <= to_utc(end)]
        return frame.sort_values("timestamp").reset_index(drop=True)

    def available_markets(self, interval: str | None = None) -> list[str]:
        markets: set[str] = set()
        for file in self.candles_dir.glob("*.parquet"):
            try:
                market, file_interval = file.stem.split("__", 1)
            except ValueError:
                continue
            if interval is None or file_interval == interval:
                markets.add(market)
        return sorted(markets)

    # -- manifest --------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def read_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.manifest_path)

    def upsert_manifest(self, records: list[DatasetRecord] | DatasetRecord) -> pd.DataFrame:
        if isinstance(records, DatasetRecord):
            records = [records]
        new = pd.DataFrame([r.to_row() for r in records])
        current = self.read_manifest()
        if not current.empty:
            key = ["market", "interval"]
            merged_keys = set(map(tuple, new[key].to_numpy().tolist()))
            mask = ~current[key].apply(tuple, axis=1).isin(merged_keys)
            combined = pd.concat([current[mask], new], ignore_index=True)
        else:
            combined = new
        combined = combined.sort_values(["market", "interval"]).reset_index(drop=True)
        combined.to_parquet(self.manifest_path, index=False, compression="zstd")
        return combined

    def usable_datasets(self, interval: str) -> list[str]:
        """Markets whose manifest row for ``interval`` did not fail validation."""
        manifest = self.read_manifest()
        if manifest.empty:
            return self.available_markets(interval)
        subset = manifest[
            (manifest["interval"] == interval) & (manifest["validation_status"] != "FAIL")
        ]
        return sorted(subset["market"].unique().tolist())


class ResultStore:
    """Research outputs (CSV/Parquet/HTML/Markdown) under ``data/results``."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_frame(self, name: str, frame: pd.DataFrame) -> Path:
        target = self.path(name)
        if target.suffix == ".parquet":
            frame.to_parquet(target, index=False, compression="zstd")
        elif target.suffix == ".csv":
            frame.to_csv(target, index=False)
        else:
            raise ValueError(f"Unsupported result format: {target.suffix}")
        log.info("Wrote %s (%d rows)", target, len(frame))
        return target

    def write_text(self, name: str, text: str) -> Path:
        target = self.path(name)
        target.write_text(text, encoding="utf-8")
        log.info("Wrote %s (%d chars)", target, len(text))
        return target

    def write_json(self, name: str, payload: Any) -> Path:
        target = self.path(name)
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Wrote %s", target)
        return target

    def read_json(self, name: str) -> Any | None:
        target = self.path(name)
        if not target.exists():
            return None
        return json.loads(target.read_text(encoding="utf-8"))


def duckdb_connection(path: Path | str | None = None):
    """Open a DuckDB connection for ad-hoc SQL over the Parquet datasets.

    DuckDB is used for exploratory queries only; every number in the reports is
    produced by the pandas code paths that the tests cover.
    """
    import duckdb

    return duckdb.connect(str(path) if path else ":memory:")
