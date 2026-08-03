#!/usr/bin/env python3
"""Re-validate stored datasets and rewrite the manifest. No network access.

Use after changing a validation threshold, or when the manifest's verdict no
longer reflects how the data should be judged. Reads the Parquet already on disk
and rewrites ``validation_status`` in place; the candles themselves are never
modified.

    python scripts/revalidate.py                 # apply config threshold
    python scripts/revalidate.py --max-missing 0.95
    python scripts/revalidate.py --dry-run       # show what would change
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from bitvavo_momentum.config import Config  # noqa: E402
from bitvavo_momentum.data_validator import validate_candles  # noqa: E402
from bitvavo_momentum.logging_utils import setup_logging  # noqa: E402
from bitvavo_momentum.storage import ParquetStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--max-missing", type=float, default=None,
                        help="override research.data.max_missing_fraction")
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    config = Config.load()
    store = ParquetStore(config.path("processed_dir"))

    max_missing = args.max_missing
    if max_missing is None:
        max_missing = float(config.get("research", "data", "max_missing_fraction", default=0.95))

    manifest = store.read_manifest()
    if manifest.empty:
        print("No manifest found. Run scripts/download_history.py first.")
        return 1

    rows = manifest[manifest["interval"] == args.interval]
    if rows.empty:
        print(f"No {args.interval} datasets in the manifest.")
        return 1

    print(f"Re-validating {len(rows)} datasets with max_missing_fraction={max_missing:.0%}\n")
    changes: list[dict] = []

    for _, row in rows.iterrows():
        market = row["market"]
        frame = store.read_candles(market, args.interval)
        result = validate_candles(
            frame, market=market, interval=args.interval,
            max_missing_fraction=max_missing, min_rows=args.min_rows,
        )
        before = str(row.get("validation_status", "UNVALIDATED"))
        changes.append(
            {
                "market": market,
                "n_rows": result.n_rows,
                "missing_fraction": round(result.missing_fraction, 4),
                "before": before,
                "after": result.status,
                "changed": before != result.status,
                "notes": "; ".join(result.notes),
            }
        )
        if not args.dry_run:
            index = manifest.index[
                (manifest["market"] == market) & (manifest["interval"] == args.interval)
            ]
            manifest.loc[index, "validation_status"] = result.status
            manifest.loc[index, "validation_notes"] = "; ".join(result.notes)
            manifest.loc[index, "n_missing_intervals"] = result.n_missing
            manifest.loc[index, "missing_fraction"] = result.missing_fraction

    report = pd.DataFrame(changes).sort_values("missing_fraction")
    print(report[["market", "n_rows", "missing_fraction", "before", "after", "changed"]].to_string(index=False))

    counts = report["after"].value_counts().to_dict()
    print(f"\nVerdicts: {counts}")
    print(f"Changed: {int(report['changed'].sum())} of {len(report)}")

    if args.dry_run:
        print("\nDry run - manifest not written.")
        return 0

    manifest.to_parquet(store.manifest_path, index=False, compression="zstd")
    print(f"\nManifest rewritten: {store.manifest_path}")
    usable = store.usable_datasets(args.interval)
    print(f"Datasets now usable for research: {len(usable)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
