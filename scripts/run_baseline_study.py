#!/usr/bin/env python3
"""Milestone 3: the control study.

Answers a question the original event study never asked: **compared to what?**

A +0.42% mean 60-minute forward return means one thing if a random entry in the
same markets returned 0.00%, and something entirely different if it returned
+0.40%. The first is a small edge; the second is market drift with extra steps.

Three references are computed:

1. **unconditional** - every bar in the eligible universe, thinned;
2. **matched random** - random entries in the same markets and the same calendar
   window as the real events, so only the *timing* differs;
3. **buy and hold** - BTC and ETH over the same period.

Then the conditional lift (event mean minus matched-random mean) is compared
against the round-trip cost bar.

    python scripts/run_baseline_study.py
    python scripts/run_baseline_study.py --max-markets 20   # faster first pass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from bitvavo_momentum.baseline import (  # noqa: E402
    buy_and_hold_benchmark,
    conditional_lift,
    lift_confidence_interval,
    matched_random_baseline,
    unconditional_forward_returns,
)
from bitvavo_momentum.config import Config, load_dotenv_if_present  # noqa: E402
from bitvavo_momentum.event_detector import event_study, specs_from_config  # noqa: E402
from bitvavo_momentum.execution_model import ExecutionModel, load_scenarios  # noqa: E402
from bitvavo_momentum.logging_utils import setup_logging  # noqa: E402
from bitvavo_momentum.pipeline import build_dataset, build_events, result_store_for  # noqa: E402
from bitvavo_momentum.timeframes import summarise_timeframes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=120,
                        help="thin the unconditional sample (bars between observations)")
    parser.add_argument("--draws-per-event", type=int, default=5)
    parser.add_argument("--primary-only", action="store_true",
                        help="primary spec only instead of the full 36-spec grid")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv_if_present()
    log = setup_logging(args.log_level, "logs/baseline.log")
    config = Config.load()
    config.ensure_dirs()

    dataset = build_dataset(config, source=args.source, interval=args.interval, max_markets=args.max_markets)
    store = result_store_for(config, dataset)

    if not dataset.candles:
        log.error("No usable market data. Run scripts/download_history.py first.")
        print("BLOCKED: no real candle data in data/processed. Nothing computed.")
        return 1

    headline = config.get("risk", "headline_scenario", default="realistic")
    scenarios = load_scenarios(config.risk)
    cost_bps = ExecutionModel(scenarios[headline]).describe()["round_trip_cost_bps_minimum"]
    horizons = tuple(config.get("research", "events", "forward_horizons",
                                default=[15, 30, 60, 120, 240, 480, 1440, 2880]))
    seed = int(config.get("research", "robustness", "random_seed", default=20260802))

    log.info("Round-trip cost bar: %.1f bps (%s scenario)", cost_bps, headline)

    # -- timeframe coverage (Milestone 3 data-quality output) ----------------
    coverage = summarise_timeframes(dataset.candles)
    if not coverage.empty:
        store.write_frame("timeframe_coverage.csv", coverage)

    # -- events --------------------------------------------------------------
    specs = specs_from_config(config.research)
    if args.primary_only:
        specs = specs[:1]
    events = build_events(dataset, specs, config, interval=args.interval)
    if events.empty:
        log.error("No events detected")
        return 1

    study = event_study(events, horizons)

    # -- baselines -----------------------------------------------------------
    log.info("Computing unconditional baseline (sample_every=%d)...", args.sample_every)
    unconditional = unconditional_forward_returns(
        dataset.candles, horizons, args.interval,
        sample_every=args.sample_every, eligibility_by_market=dataset.eligibility,
    )

    log.info("Computing matched-random baseline (%d draws per event)...", args.draws_per_event)
    matched = matched_random_baseline(
        events, dataset.candles, horizons, args.interval,
        draws_per_event=args.draws_per_event, seed=seed,
    )

    hold = buy_and_hold_benchmark(
        dataset.candles,
        markets=tuple(config.get("research", "data", "reference_markets", default=["BTC-EUR", "ETH-EUR"])),
        starting_equity=float(config.get("risk", "portfolio", "starting_equity_eur", default=10000.0)),
    )

    lift = conditional_lift(study, matched, round_trip_cost_bps=cost_bps)

    # -- per-horizon confidence intervals on the lift ------------------------
    ci_rows = []
    for horizon in horizons:
        column = f"fwd_ret_{horizon}m"
        if column not in events.columns or matched.empty:
            continue
        base_row = matched[matched["horizon_minutes"] == horizon]
        if base_row.empty:
            continue
        # Reconstruct a baseline sample of the right size and moments for the CI.
        import numpy as np

        rng = np.random.default_rng(seed + horizon)
        synthetic_base = rng.normal(
            float(base_row["mean_return"].iloc[0]),
            float(base_row["std_return"].iloc[0]),
            int(min(20000, base_row["n_samples"].iloc[0])),
        )
        stats = lift_confidence_interval(events[column], synthetic_base, iterations=1000, seed=seed)
        stats["horizon_minutes"] = horizon
        ci_rows.append(stats)
    lift_ci = pd.DataFrame(ci_rows)

    # -- write ---------------------------------------------------------------
    store.write_frame("baseline_unconditional.csv", unconditional)
    store.write_frame("baseline_matched_random.csv", matched)
    store.write_frame("baseline_buy_and_hold.csv", hold)
    store.write_frame("conditional_lift.csv", lift)
    if not lift_ci.empty:
        store.write_frame("conditional_lift_ci.csv", lift_ci)

    # -- report --------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"BASELINE CONTROL STUDY   |   cost bar = {cost_bps:.1f} bps round trip ({headline})")
    print(f"{'=' * 78}")
    print(f"Markets: {len(dataset.candles)}   Events: {len(events)}   Specs: {len(specs)}")
    print(f"Period: {dataset.first_timestamp} .. {dataset.last_timestamp}\n")

    if not unconditional.empty:
        print("UNCONDITIONAL (entry at an arbitrary moment, gross):")
        print(unconditional[["horizon_minutes", "n_samples", "mean_return", "median_return", "hit_rate"]]
              .to_string(index=False))

    if not matched.empty:
        print("\nMATCHED RANDOM (same markets, same window, random timing, gross):")
        print(matched[["horizon_minutes", "n_samples", "mean_return", "median_return", "hit_rate"]]
              .to_string(index=False))

    if not lift.empty:
        print("\nCONDITIONAL LIFT (event mean - matched-random mean):")
        columns = ["horizon_minutes", "n_events", "event_mean", "baseline_mean",
                   "lift_bps", "lift_net_bps", "beats_cost"]
        print(lift[[c for c in columns if c in lift.columns]].to_string(index=False))
        winners = lift[lift["beats_cost"]]
        print(f"\nHorizons where the conditional lift alone clears {cost_bps:.0f} bps: "
              f"{len(winners)} of {len(lift)}")
        if not winners.empty:
            print(winners[["horizon_minutes", "lift_bps", "lift_net_bps"]].to_string(index=False))

    if not hold.empty:
        print("\nBUY AND HOLD over the same period:")
        print(hold[["market", "total_return", "annualised_return", "max_drawdown"]].to_string(index=False))

    print(f"\nArtefacts written to {store.root}")
    print("\nInterpretation: a positive event mean with zero lift means the signal added "
          "nothing beyond being invested. Only `beats_cost = True` is tradeable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
