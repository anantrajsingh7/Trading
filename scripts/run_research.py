#!/usr/bin/env python3
"""Phase 2 research: build the event dataset and the descriptive event study.

This runs *before* any strategy exists. Its job is to answer, descriptively,
whether the forward-return distribution after a +X% / N-minute event differs from
what a random entry in the same market would have produced - and to do so across
the full grid of look-backs and thresholds so the primary hypothesis is seen in
context rather than in isolation.

    python scripts/run_research.py                     # real data
    python scripts/run_research.py --source synthetic  # pipeline self-test only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bitvavo_momentum.config import Config, load_dotenv_if_present  # noqa: E402
from bitvavo_momentum.event_detector import (  # noqa: E402
    event_study,
    primary_spec_from_config,
    specs_from_config,
)
from bitvavo_momentum.logging_utils import setup_logging  # noqa: E402
from bitvavo_momentum.pipeline import build_dataset, build_events, result_store_for  # noqa: E402
from bitvavo_momentum.reporting import (  # noqa: E402
    RunContext,
    build_assumptions_document,
    build_html_report,
    build_research_report,
    event_study_figure,
)
from bitvavo_momentum.robustness import block_bootstrap_metric  # noqa: E402
from bitvavo_momentum.synthetic import SyntheticConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--synthetic-markets", type=int, default=8)
    parser.add_argument("--synthetic-days", type=int, default=120)
    parser.add_argument("--primary-only", action="store_true",
                        help="only the primary spec (+10% / 120m) instead of the full grid")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv_if_present()
    log = setup_logging(args.log_level, "logs/research.log")
    config = Config.load()
    config.ensure_dirs()

    synthetic_config = SyntheticConfig(n_minutes=args.synthetic_days * 24 * 60) if args.source == "synthetic" else None
    dataset = build_dataset(
        config, source=args.source, interval=args.interval, max_markets=args.max_markets,
        synthetic_config=synthetic_config, n_synthetic_markets=args.synthetic_markets,
    )

    store = result_store_for(config, dataset)
    context = RunContext(
        data_source=dataset.source,
        n_markets=len(dataset.candles),
        first_timestamp=dataset.first_timestamp,
        last_timestamp=dataset.last_timestamp,
        execution_scenario=config.get("risk", "headline_scenario", default="realistic"),
        is_synthetic=dataset.is_synthetic,
        has_real_data=bool(dataset.candles) and not dataset.is_synthetic,
    )

    if not dataset.candles:
        log.error("No usable market data. Run scripts/download_history.py first.")
        report = build_research_report(context)
        store.write_text("research_report.md", report)
        store.write_text(
            "assumptions_and_limitations.md",
            build_assumptions_document(
                context,
                extra_limitations=["No candle data was present in data/processed at run time."],
            ),
        )
        print("No data available - wrote a report that reports exactly that.")
        return 1

    specs = [primary_spec_from_config(config.research)] if args.primary_only else specs_from_config(config.research)
    log.info("Testing %d event specifications", len(specs))

    events = build_events(dataset, specs, config, interval=args.interval)
    context.n_events = int(len(events))
    context.n_configurations_evaluated = len(specs)

    if events.empty:
        log.error("No events detected across %d specs", len(specs))
        store.write_text("research_report.md", build_research_report(context))
        return 1

    horizons = tuple(config.get("research", "events", "forward_horizons", default=[15, 30, 60, 120, 240, 480, 1440]))

    overall = event_study(events, horizons)
    by_spec = event_study(events, horizons, group_by=["event_lookback_minutes", "event_threshold"])
    by_market = event_study(events, horizons, group_by=["market"])
    by_session = event_study(events, horizons, group_by=["session_bucket"])
    by_regime = (
        event_study(events, horizons, group_by=["btc_trend"]) if "btc_trend" in events.columns else pd.DataFrame()
    )

    # Bootstrap confidence intervals on the primary spec.
    primary = primary_spec_from_config(config.research)
    primary_events = events[
        (events["event_lookback_minutes"] == primary.lookback_minutes)
        & (np.isclose(events["event_threshold"], primary.threshold))
    ]
    bootstrap_rows = []
    for horizon in horizons:
        column = f"fwd_ret_{horizon}m"
        if column not in primary_events.columns:
            continue
        stats = block_bootstrap_metric(
            primary_events[column],
            block_size=10,
            iterations=int(config.get("research", "robustness", "bootstrap_iterations", default=2000)),
            seed=int(config.get("research", "robustness", "random_seed", default=20260802)),
        )
        stats["horizon_minutes"] = horizon
        bootstrap_rows.append(stats)
    bootstrap = pd.DataFrame(bootstrap_rows)

    store.write_frame("event_dataset.parquet", events)
    store.write_frame("event_study_overall.csv", overall)
    store.write_frame("event_study_by_spec.csv", by_spec)
    store.write_frame("event_study_by_market.csv", by_market)
    store.write_frame("event_study_by_session.csv", by_session)
    if not by_regime.empty:
        store.write_frame("event_study_by_regime.csv", by_regime)
    if not bootstrap.empty:
        store.write_frame("event_study_bootstrap.csv", bootstrap)
    if not dataset.validation.empty:
        store.write_frame("data_validation.csv", dataset.validation)

    report = build_research_report(
        context,
        event_study=overall,
        robustness_summary={"regime_performance": by_regime} if not by_regime.empty else None,
        verdict=(
            "This run covers Phase 2 only: it describes the forward-return distribution after the "
            "event and makes no claim about tradability. Costs are not yet applied. Run "
            "`scripts/run_backtest.py` for the net-of-cost evaluation, which is the only one that "
            "can support or reject the hypothesis."
        ),
    )
    store.write_text("research_report.md", report)
    store.write_text(
        "assumptions_and_limitations.md",
        build_assumptions_document(context),
    )
    html = build_html_report(
        context,
        figures={"Mean forward return by horizon": event_study_figure(overall)},
        tables={"Event study by specification": by_spec, "Bootstrap (primary spec)": bootstrap},
        title="Bitvavo momentum - Phase 2 event study",
    )
    store.write_text("event_study.html", html)

    print(f"\nEvents detected: {len(events)} across {events['market'].nunique()} markets")
    print(f"Specifications tested: {len(specs)}")
    if not overall.empty:
        print("\nDescriptive forward returns (GROSS, before any costs):")
        print(overall[["horizon_minutes", "n_events", "mean_return", "median_return", "hit_rate"]].to_string(index=False))
    if not bootstrap.empty:
        print("\nBlock-bootstrap 95% CI on the mean (primary spec):")
        print(bootstrap[["horizon_minutes", "n", "observed", "ci_low", "ci_high", "p_value_gt_zero"]].to_string(index=False))
    print(f"\nArtefacts written to {store.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
