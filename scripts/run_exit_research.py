#!/usr/bin/env python3
"""Milestone 5: is it the entries or the exits?

Every one of the 16 entry variants lost money in the family comparison, but all
of them were run through one exit policy (+10% target, 1.5xATR stop, 48h clock)
with a 5%-8% win rate. That win rate is evidence about the *target*, not
necessarily about the entries, so this script tests the two explanations apart.

Stage 1 - exit-independent, cheap, decisive
    Forward returns from each signal at 6/12/24/36/48 hours with no exit rule at
    all, plus the MFE/MAE profile. Buy-and-hold-to-horizon is the ceiling on what
    any non-clairvoyant exit can extract on average, so a family that is below
    the cost floor at every horizon cannot be rescued by exit tuning and is
    closed here.

Stage 2 - the exit grid, only for families that clear stage 1
    Stops (fixed %, ATR, chandelier trail), break-even moves, targets and the
    five holding periods, run through the real backtester on the TRAIN split.

The untouched test set is never read. Stage 2's best cell is an in-sample
maximum over many configurations and is reported as such - it is a candidate for
validation, never a result.

    python scripts/run_exit_research.py --max-markets 20
    python scripts/run_exit_research.py --max-markets 20 --stage1-only
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from bitvavo_momentum import metrics as M  # noqa: E402
from bitvavo_momentum.backtester import Backtester  # noqa: E402
from bitvavo_momentum.config import Config, load_dotenv_if_present  # noqa: E402
from bitvavo_momentum.execution_model import ExecutionModel, load_scenarios  # noqa: E402
from bitvavo_momentum.exit_research import (  # noqa: E402
    DEFAULT_HORIZONS,
    excursion_table,
    exit_reason_breakdown,
    forward_return_table,
    signal_outcomes,
    stage_one_verdict,
)
from bitvavo_momentum.logging_utils import setup_logging  # noqa: E402
from bitvavo_momentum.pipeline import build_dataset, result_store_for  # noqa: E402
from bitvavo_momentum.risk_manager import RiskLimits, SizingConfig  # noqa: E402
from bitvavo_momentum.robustness import independent_trade_count  # noqa: E402
from bitvavo_momentum.signal_strategies import (  # noqa: E402
    compute_setup_features,
    default_signal_strategies,
    generate_signals,
)
from bitvavo_momentum.strategies import ExitPolicy, ImmediateEntry  # noqa: E402
from bitvavo_momentum.timeframes import SETUP_TF, resample_ohlcv  # noqa: E402
from bitvavo_momentum.walk_forward import apply_split, chronological_splits  # noqa: E402

# Stage 2 grid. Deliberately small and economically distinct: each axis asks a
# different question rather than filling a hyperparameter space.
STOPS: tuple[dict, ...] = (
    {"atr_stop_multiple": 1.0},
    {"atr_stop_multiple": 1.5},
    {"atr_stop_multiple": 2.5},
    {"stop_loss_pct": 0.03},
    {"chandelier_atr_multiple": 3.0},
)
TARGETS: tuple[dict, ...] = (
    {"take_profit_pct": None},          # let the stop and the clock do the work
    {"take_profit_pct": 0.03},
    {"take_profit_pct": 0.05},
    {"take_profit_pct": 0.10},
)
BREAKEVENS: tuple[dict, ...] = (
    {"breakeven_after_pct": None},
    {"breakeven_after_pct": 0.015},
    {"breakeven_after_pct": 0.03},
)
HOLDING_PERIODS: tuple[int, ...] = (6 * 60, 12 * 60, 24 * 60, 36 * 60, 48 * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--setup-tf", default=SETUP_TF)
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--stage1-only", action="store_true",
                        help="skip the exit grid even if a family clears stage 1")
    parser.add_argument("--max-families", type=int, default=3,
                        help="how many stage-1 survivors earn an exit grid")
    parser.add_argument("--force-stage2", action="store_true",
                        help="run the exit grid on the least-bad families even if none survive "
                             "stage 1 (diagnostic only - the result cannot be a recommendation)")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_policies(holding_minutes: int) -> list[ExitPolicy]:
    policies = []
    for stop, target, breakeven in product(STOPS, TARGETS, BREAKEVENS):
        params = {**stop, **target, **breakeven, "time_stop_minutes": holding_minutes}
        params.setdefault("stop_loss_pct", None)
        policies.append(ExitPolicy(**params))
    return policies


def main() -> int:  # noqa: PLR0915 - a linear protocol reads better in one place
    args = parse_args()
    load_dotenv_if_present()
    log = setup_logging(args.log_level, "logs/exit_research.log")
    config = Config.load()
    config.ensure_dirs()

    dataset = build_dataset(config, source=args.source, interval="1m", max_markets=args.max_markets)
    store = result_store_for(config, dataset)
    if not dataset.candles:
        print("BLOCKED: no real candle data in data/processed. Run scripts/download_history.py first.")
        return 1

    starting_equity = float(config.get("risk", "portfolio", "starting_equity_eur", default=10000.0))
    headline = config.get("risk", "headline_scenario", default="realistic")
    scenarios = load_scenarios(config.risk)
    cost_bps = ExecutionModel(scenarios[headline]).describe()["round_trip_cost_bps_minimum"]
    seed = int(config.get("research", "robustness", "random_seed", default=20260802))
    sizing = SizingConfig.from_config(config.risk)
    limits = RiskLimits.from_config(config.risk)

    log.info("Resampling %d markets to %s...", len(dataset.candles), args.setup_tf)
    features_by_market: dict[str, pd.DataFrame] = {}
    for market, frame in dataset.candles.items():
        resampled = resample_ohlcv(frame, args.setup_tf)
        if len(resampled) < 300:
            continue
        computed = compute_setup_features(resampled, args.setup_tf)
        if not computed.empty:
            features_by_market[market] = computed
    if not features_by_market:
        print("BLOCKED: no market had enough setup-timeframe history.")
        return 1
    log.info("Features ready for %d markets", len(features_by_market))

    strategies = default_signal_strategies()
    signals = generate_signals(strategies, features_by_market, apply_exhaustion_veto=False,
                               eligibility_by_market=None)
    if signals.empty:
        print("No signals generated. Nothing to evaluate.")
        return 1

    splits = chronological_splits(
        signals,
        train_fraction=float(config.get("research", "splits", "train_fraction", default=0.5)),
        validation_fraction=float(config.get("research", "splits", "validation_fraction", default=0.25)),
        embargo_minutes=int(config.get("research", "splits", "embargo_minutes", default=2880)),
    )
    train = apply_split(signals, splits.train)
    log.info("Train signals: %d of %d (validation and test withheld)", len(train), len(signals))
    if train.empty:
        print("BLOCKED: the training split contains no signals.")
        return 1

    # ------------------------------------------------------------------ #
    # STAGE 1 - exit-independent
    # ------------------------------------------------------------------ #
    log.info("Stage 1: forward returns and excursions for %d train signals...", len(train))
    outcomes = signal_outcomes(train, features_by_market, DEFAULT_HORIZONS,
                               interval=args.setup_tf, max_holding_minutes=48 * 60)
    if outcomes.empty:
        print("BLOCKED: no signal had a full forward window inside the data.")
        return 1

    forward = forward_return_table(outcomes, DEFAULT_HORIZONS, round_trip_cost_bps=cost_bps)
    excursions = excursion_table(outcomes, max_holding_minutes=48 * 60)
    verdict = stage_one_verdict(forward, excursions, take_profit_pct=0.10)

    store.write_frame("exit_stage1_forward_returns.csv", forward)
    store.write_frame("exit_stage1_excursions.csv", excursions)
    store.write_frame("exit_stage1_verdict.csv", verdict)

    print(f"\n{'=' * 100}")
    print(f"STAGE 1 - NO EXIT RULE - TRAIN SPLIT - cost floor {cost_bps:.0f} bps round trip")
    print(f"{'=' * 100}")
    print("Mean return from each signal held blind to the horizon. This is the ceiling")
    print("on what any exit rule can average, so a family below the cost floor everywhere")
    print("cannot be fixed by changing stops or targets.\n")
    print(verdict.to_string(index=False))

    print("\nBest horizon per family (top 15 rows of the full table):")
    columns = ["event_spec", "horizon_hours", "n_signals", "gross_mean", "net_mean",
               "hit_rate", "hit_rate_after_cost", "beats_cost"]
    print(forward[[c for c in columns if c in forward.columns]].head(15).to_string(index=False))

    print("\nExcursion profile - how far price ran within 48h (reach_up_10pct is the")
    print("ceiling on the +10% take-profit's hit rate):")
    columns = ["event_spec", "n_signals", "median_mfe", "median_mae", "mfe_mae_ratio",
               "reach_up_10pct", "reach_up_5pct", "reach_down_5pct"]
    print(excursions[[c for c in columns if c in excursions.columns]].to_string(index=False))

    survivors = verdict[verdict["survives"]] if "survives" in verdict.columns else pd.DataFrame()
    print(f"\nFamilies with a positive mean net return at some holding period: "
          f"{len(survivors)} of {len(verdict)}")

    if survivors.empty and not args.force_stage2:
        print("\nSTAGE 1 CLOSES THIS QUESTION. No entry family paid for its own execution at any")
        print("holding period from 6h to 48h, with no exit rule imposed at all. The exit policy")
        print("was not the reason the family comparison lost money; the entries were.")
        print("Re-run with --force-stage2 to grid exits anyway as a diagnostic, but a profitable")
        print("cell found that way would be an artefact of the search, not an edge.")
        print(f"\nArtefacts written to {store.root}")
        return 0

    if args.stage1_only:
        print(f"\nStage 2 skipped (--stage1-only). Artefacts written to {store.root}")
        return 0

    # ------------------------------------------------------------------ #
    # STAGE 2 - exit grid on the survivors
    # ------------------------------------------------------------------ #
    if survivors.empty:
        chosen = verdict.head(args.max_families)
        print(f"\n--force-stage2: gridding exits on the {len(chosen)} least-bad families. "
              "DIAGNOSTIC ONLY.")
    else:
        chosen = survivors.head(args.max_families)

    names = chosen["event_spec"].tolist()
    grid_rows: list[dict] = []
    total = len(names) * len(build_policies(48 * 60)) * len(HOLDING_PERIODS)
    print(f"\nStage 2: {len(names)} famil(ies) x {total // max(1, len(names))} exit "
          f"configurations = {total} backtests.")
    done, started = 0, time.time()

    for name in names:
        subset = train[train["event_spec"] == name]
        if subset.empty:
            continue
        for holding_minutes in HOLDING_PERIODS:
            for policy in build_policies(holding_minutes):
                engine = Backtester(
                    ExecutionModel(scenarios[headline], seed=seed), policy, sizing, limits,
                    starting_equity=starting_equity, interval=args.setup_tf,
                    max_holding_minutes=holding_minutes,
                    circuit_breakers=config.get("risk", "circuit_breakers", default={}),
                )
                result = engine.run(subset, features_by_market, ImmediateEntry(), seed=seed)
                done += 1
                if done % 25 == 0:
                    elapsed = time.time() - started
                    rate = elapsed / done
                    log.info("  %d/%d configurations (%.0fs elapsed, ~%.0fs remaining)",
                             done, total, elapsed, rate * (total - done))
                if result is None or result.trades.empty:
                    continue
                row = M.compute_trade_metrics(result.trades, starting_equity)
                row.update({
                    "strategy": name,
                    "exit_policy": policy.name,
                    "holding_hours": holding_minutes // 60,
                    "stop": (f"atr{policy.atr_stop_multiple}" if policy.atr_stop_multiple
                             else f"chand{policy.chandelier_atr_multiple}" if policy.chandelier_atr_multiple
                             else f"pct{policy.stop_loss_pct}"),
                    "target": policy.take_profit_pct,
                    "breakeven": policy.breakeven_after_pct,
                    "scenario": headline, "split": "train",
                })
                closed = result.closed_trades()
                if not closed.empty:
                    row.update(independent_trade_count(closed))
                grid_rows.append(row)

    if not grid_rows:
        print("No exit configuration produced any trades.")
        return 1

    grid = pd.DataFrame(grid_rows).sort_values("net_expectancy", ascending=False).reset_index(drop=True)
    store.write_frame("exit_grid_train.csv", grid)

    print(f"\n{'=' * 100}")
    print(f"STAGE 2 - EXIT GRID - TRAIN SPLIT - {len(grid)} configurations searched")
    print(f"{'=' * 100}")
    columns = ["strategy", "stop", "target", "breakeven", "holding_hours", "n_trades",
               "n_clusters", "win_rate", "net_expectancy", "profit_factor", "max_drawdown"]
    print(grid[[c for c in columns if c in grid.columns]].head(20).to_string(index=False))

    positive = grid[grid["net_expectancy"] > 0]
    print(f"\nConfigurations with positive net expectancy: {len(positive)} of {len(grid)}")
    print(f"Searching {len(grid)} cells makes the top row a maximum, not an estimate. It is a")
    print("candidate for the validation split and nothing more.")

    # What actually closed the trades in the top configuration? If it is still
    # dominated by time-stops, the grid has not found an exit rule that engages.
    best = grid.iloc[0]
    best_policy = next(
        (p for p in build_policies(int(best["holding_hours"]) * 60) if p.name == best["exit_policy"]),
        None,
    )
    if best_policy is not None:
        engine = Backtester(
            ExecutionModel(scenarios[headline], seed=seed), best_policy, sizing, limits,
            starting_equity=starting_equity, interval=args.setup_tf,
            max_holding_minutes=int(best["holding_hours"]) * 60,
            circuit_breakers=config.get("risk", "circuit_breakers", default={}),
        )
        replay = engine.run(train[train["event_spec"] == best["strategy"]],
                            features_by_market, ImmediateEntry(), seed=seed)
        reasons = exit_reason_breakdown(replay.closed_trades())
        if not reasons.empty:
            store.write_frame("exit_grid_best_exit_reasons.csv", reasons)
            print(f"\nHow the top configuration's trades ended ({best['strategy']}, "
                  f"{best['exit_policy']}):")
            print(reasons.to_string(index=False))

    # Holding-period marginal: does the clock matter independently of the stop?
    holding = grid.groupby("holding_hours").agg(
        configurations=("net_expectancy", "size"),
        mean_net_expectancy=("net_expectancy", "mean"),
        best_net_expectancy=("net_expectancy", "max"),
        mean_win_rate=("win_rate", "mean"),
    ).reset_index()
    store.write_frame("holding_period_comparison.csv", holding)
    print("\nHOLDING-PERIOD COMPARISON (averaged across the exit grid):")
    print(holding.to_string(index=False))

    print(f"\nArtefacts written to {store.root}")
    print("The untouched test set was NOT read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
