#!/usr/bin/env python3
"""Download Bitvavo OHLCV history (Phase 1).

Examples
--------
    # connectivity + one market, short range - always do this first
    python scripts/download_history.py --smoke-test

    # every EUR market, 1-minute candles, resumable
    python scripts/download_history.py --markets all --interval 1m

    # top 40 markets by 24h quote volume
    python scripts/download_history.py --markets top:40 --interval 1m

Credentials are optional and read from the environment only. Without them the
downloader uses public endpoints, which is sufficient for all research.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from bitvavo_momentum.bitvavo_client import BitvavoClient, BitvavoError  # noqa: E402
from bitvavo_momentum.config import Config, Credentials, load_dotenv_if_present  # noqa: E402
from bitvavo_momentum.data_downloader import HistoryDownloader  # noqa: E402
from bitvavo_momentum.logging_utils import setup_logging  # noqa: E402
from bitvavo_momentum.market_universe import fetch_market_rules, live_universe  # noqa: E402
from bitvavo_momentum.storage import ParquetStore, RawStore  # noqa: E402
from bitvavo_momentum.timeutils import now_utc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--markets", default="top:20",
                        help="'all', 'top:N', or a comma-separated list such as BTC-EUR,ETH-EUR")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--history-start", default=None, help="ISO timestamp; defaults to config")
    parser.add_argument("--force-full", action="store_true", help="ignore existing data and re-download")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="cap API windows per market (useful for a quick trial)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="one market, ~2 days: verifies connectivity, ordering and timestamps")
    parser.add_argument("--sleep", type=float, default=0.05, help="seconds between API calls")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve_markets(client: BitvavoClient, spec: str, config: Config) -> list[str]:
    exclude = config.get("research", "universe", "exclude_bases", default=[])
    rules = fetch_market_rules(client, config.get("research", "data", "quote_currency", default="EUR"))
    tradeable = live_universe(rules, exclude)

    if spec == "all":
        selected = tradeable
    elif spec.startswith("top:"):
        n = int(spec.split(":", 1)[1])
        tickers = client.get_ticker_24h()
        volumes = {}
        for row in tickers if isinstance(tickers, list) else [tickers]:
            market = row.get("market")
            if market in tradeable:
                try:
                    volumes[market] = float(row.get("volumeQuote") or 0.0)
                except (TypeError, ValueError):
                    volumes[market] = 0.0
        selected = [m for m, _ in sorted(volumes.items(), key=lambda kv: kv[1], reverse=True)[:n]]
    else:
        selected = [m.strip().upper() for m in spec.split(",") if m.strip()]

    references = config.get("research", "data", "reference_markets", default=["BTC-EUR", "ETH-EUR"])
    for reference in references:
        if reference not in selected and reference in rules:
            selected.append(reference)
    return selected


def smoke_test(client: BitvavoClient, downloader: HistoryDownloader, interval: str) -> int:
    """One market, a short range: proves connectivity, ordering and timestamps."""
    log = setup_logging()
    market = "BTC-EUR"
    log.info("Smoke test: server time")
    server_ms = client.get_time()
    server_time = pd.Timestamp(server_ms, unit="ms", tz="UTC")
    skew = abs((server_time - now_utc()).total_seconds())
    log.info("Server time %s (local clock skew %.1fs)", server_time, skew)
    if skew > 30:
        log.warning("Clock skew above 30s can break authenticated requests")

    start = now_utc() - pd.Timedelta(days=2)
    outcome = downloader.download_market(market, interval=interval, history_start=start)
    frame = downloader.store.read_candles(market, interval)

    print(f"\nmarket           : {market}")
    print(f"interval         : {interval}")
    print(f"rows             : {len(frame)}")
    if frame.empty:
        print("NO DATA RETURNED - check network access to api.bitvavo.com")
        return 1
    print(f"first timestamp  : {frame['timestamp'].min()}")
    print(f"last timestamp   : {frame['timestamp'].max()}")
    print(f"monotonic        : {frame['timestamp'].is_monotonic_increasing}")
    print(f"duplicates       : {int(frame['timestamp'].duplicated().sum())}")
    print(f"tz aware (UTC)   : {frame['timestamp'].dt.tz is not None}")
    print(f"api calls        : {outcome.api_calls}")
    record = downloader.record_dataset(market, interval, outcome)
    print(f"validation       : {record.validation_status} ({record.validation_notes or 'clean'})")
    print(f"missing bars     : {record.n_missing_intervals} ({record.missing_fraction:.2%})")
    print(f"zero-volume bars : {record.n_zero_volume_rows} ({record.zero_volume_fraction:.2%})")
    return 0


def main() -> int:
    args = parse_args()
    load_dotenv_if_present()
    log = setup_logging(args.log_level, "logs/download.log")

    config = Config.load()
    config.ensure_dirs()
    credentials = Credentials.from_env()
    log.info("Credentials: %s", credentials)

    raw_store = RawStore(config.path("raw_dir"))
    parquet_store = ParquetStore(config.path("processed_dir"))

    with BitvavoClient(credentials, raw_sink=raw_store.as_sink()) as client:
        downloader = HistoryDownloader(
            client, raw_store, parquet_store,
            max_candles_per_request=int(config.get("research", "data", "max_candles_per_request", default=1440)),
            sleep_between_calls=args.sleep,
        )

        try:
            if args.smoke_test:
                return smoke_test(client, downloader, args.interval)

            markets = resolve_markets(client, args.markets, config)
            log.info("Downloading %d markets: %s", len(markets), ", ".join(markets[:10]) + ("..." if len(markets) > 10 else ""))

            history_start = args.history_start or config.get(
                "research", "data", "history_start_utc", default="2021-01-01T00:00:00Z"
            )
            manifest = downloader.download_many(
                markets, interval=args.interval, history_start=history_start,
                force_full=args.force_full, max_windows=args.max_windows,
            )
        except BitvavoError as exc:
            log.error("Download aborted: %s", exc)
            return 2

    if manifest.empty:
        log.error("No datasets were downloaded")
        return 2

    columns = ["market", "n_rows", "first_timestamp", "last_timestamp",
               "missing_fraction", "zero_volume_fraction", "validation_status"]
    print("\n" + manifest[[c for c in columns if c in manifest.columns]].to_string(index=False))
    failed = manifest[manifest["validation_status"] == "FAIL"]
    print(f"\n{len(manifest)} datasets, {len(failed)} failed validation")
    if not failed.empty:
        print("Failed markets:", ", ".join(failed["market"].tolist()))
    print(f"Manifest: {parquet_store.manifest_path}")
    print(f"Raw pages captured: {raw_store.pages_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
