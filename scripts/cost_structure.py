#!/usr/bin/env python3
"""The constraint every strategy in this project has been losing to.

Five families and 91 configurations have now been rejected, and the rejections
all have the same shape: a signal worth tens of basis points against a round trip
that costs 77. This script does no backtesting and reads no market data. It
writes down the arithmetic that was implicit the whole time.

A round trip costs a fixed percentage of the position. The number of round trips
per year is set by the holding period. So the annual cost of being invested is
fixed by the holding period alone, before any question of skill:

    annual cost = (hours per year / holding hours) x round-trip cost

At 77 bps and the spec's 48-hour maximum hold, that is roughly 140% of capital
per year. No edge measured in this project - or plausibly available in liquid
EUR spot markets - covers that. The 48-hour cap and the cost floor are jointly
unsatisfiable, and no amount of signal research changes it.

    python scripts/cost_structure.py
    python scripts/cost_structure.py --cost-bps 36    # maker-only, higher fee tier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from bitvavo_momentum.config import Config  # noqa: E402
from bitvavo_momentum.execution_model import ExecutionModel, load_scenarios  # noqa: E402
from bitvavo_momentum.exit_research import cost_drag_table  # noqa: E402

HOLDING_HOURS = (6, 12, 24, 36, 48, 24 * 7, 24 * 14, 24 * 30, 24 * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cost-bps", type=float, default=None,
                        help="override the round-trip cost (default: the configured realistic scenario)")
    parser.add_argument("--invested-fraction", type=float, default=1.0,
                        help="fraction of the year the book holds a position (default 1.0)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config.load()
    config.ensure_dirs()

    if args.cost_bps is not None:
        cost_bps = args.cost_bps
        label = f"{cost_bps:.0f} bps (override)"
    else:
        scenarios = load_scenarios(config.risk)
        headline = config.get("risk", "headline_scenario", default="realistic")
        cost_bps = ExecutionModel(scenarios[headline]).describe()["round_trip_cost_bps_minimum"]
        label = f"{cost_bps:.0f} bps ({headline})"

    table = cost_drag_table(HOLDING_HOURS, cost_bps, args.invested_fraction)

    print(f"\n{'=' * 88}")
    print(f"COST STRUCTURE - round trip {label}, invested {args.invested_fraction:.0%} of the time")
    print(f"{'=' * 88}")
    print("Arithmetic, not a backtest. The annual cost of trading is fixed by how long")
    print("you hold, before any question of whether the signal is any good.\n")

    display = table.copy()
    display["annual_cost_drag"] = (display["annual_cost_drag"] * 100).round(1).astype(str) + "%"
    display["required_gross_per_trade"] = (display["required_gross_per_trade"] * 100).round(2).astype(str) + "%"
    display["round_trips_per_year"] = display["round_trips_per_year"].round(0).astype(int)
    print(display[["holding_hours", "holding_days", "round_trips_per_year",
                   "annual_cost_drag", "required_gross_per_trade"]].to_string(index=False))

    cap = table[table["holding_hours"] == 48]
    if not cap.empty:
        drag = float(cap["annual_cost_drag"].iloc[0])
        print(f"\nAt the spec's 48-hour maximum holding period the annual drag is {drag * 100:.0f}%")
        print("of capital. Every strategy tested in this project has been asked to beat that.")

    # Compare against what the entry families were actually measured to be worth.
    verdict_path = config.path("results_dir") / "exit_stage1_verdict.csv"
    if verdict_path.exists():
        verdict = pd.read_csv(verdict_path)
        if "breakeven_cost_bps" in verdict.columns:
            best = verdict.sort_values("breakeven_cost_bps", ascending=False).head(5)
            print(f"\n{'=' * 88}")
            print("MEASURED SIGNAL VALUE vs THE COST FLOOR")
            print(f"{'=' * 88}")
            print("breakeven_cost_bps is what each family's signal was actually worth, gross,")
            print(f"at its best holding period. The floor it has to clear is {cost_bps:.0f} bps.\n")
            columns = [c for c in ("event_spec", "n_signals", "best_horizon_hours",
                                   "best_gross_mean", "breakeven_cost_bps") if c in best.columns]
            print(best[columns].to_string(index=False))
            gap = cost_bps - float(best["breakeven_cost_bps"].iloc[0])
            print(f"\nShortfall for the best family: {gap:.0f} bps per round trip.")
            print("Closing it requires either a signal worth more or execution that costs less.")
            print("Nothing measured so far suggests the first; the second is a fee-tier and")
            print("order-type question, not a research question.")
    else:
        print(f"\n(No {verdict_path.name} found - run scripts/run_exit_research.py "
              "--stage1-only to measure what the signals are worth.)")

    print("\nThis file contains no claim that any configuration is profitable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
