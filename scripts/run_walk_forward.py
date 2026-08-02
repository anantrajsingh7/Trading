#!/usr/bin/env python3
"""Phase 6/8: rolling walk-forward validation.

For each window the exit policy is re-selected on the training slice and then
applied unchanged to the following test slice. The concatenated test slices form
an out-of-sample record in which every decision was made without knowledge of the
period it was applied to.

    python scripts/run_walk_forward.py --source real
    python scripts/run_walk_forward.py --source synthetic --synthetic-days 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from bitvavo_momentum import metrics as M  # noqa: E402
from bitvavo_momentum.backtester import Backtester  # noqa: E402
from bitvavo_momentum.config import Config, load_dotenv_if_present  # noqa: E402
from bitvavo_momentum.event_detector import primary_spec_from_config  # noqa: E402
from bitvavo_momentum.execution_model import ExecutionModel, load_scenarios  # noqa: E402
from bitvavo_momentum.logging_utils import setup_logging  # noqa: E402
from bitvavo_momentum.pipeline import build_dataset, build_events, result_store_for  # noqa: E402
from bitvavo_momentum.risk_manager import RiskLimits, SizingConfig  # noqa: E402
from bitvavo_momentum.strategies import default_exit_policies, default_strategies  # noqa: E402
from bitvavo_momentum.synthetic import SyntheticConfig  # noqa: E402
from bitvavo_momentum.walk_forward import (  # noqa: E402
    aggregate_walk_forward,
    run_walk_forward,
    walk_forward_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--strategy", default="A_immediate", help="entry strategy held fixed across windows")
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--synthetic-markets", type=int, default=8)
    parser.add_argument("--synthetic-days", type=int, default=400)
    parser.add_argument("--train-months", type=int, default=None)
    parser.add_argument("--test-months", type=int, default=None)
    parser.add_argument("--step-months", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv_if_present()
    log = setup_logging(args.log_level, "logs/walk_forward.log")
    config = Config.load()
    config.ensure_dirs()

    starting_equity = float(config.get("risk", "portfolio", "starting_equity_eur", default=10000.0))
    headline = config.get("risk", "headline_scenario", default="realistic")
    seed = int(config.get("research", "robustness", "random_seed", default=20260802))

    synthetic_config = SyntheticConfig(n_minutes=args.synthetic_days * 24 * 60) if args.source == "synthetic" else None
    dataset = build_dataset(
        config, source=args.source, interval=args.interval, max_markets=args.max_markets,
        synthetic_config=synthetic_config, n_synthetic_markets=args.synthetic_markets,
    )
    store = result_store_for(config, dataset)

    if not dataset.candles:
        log.error("No usable market data. Run scripts/download_history.py first.")
        return 1

    events = build_events(dataset, [primary_spec_from_config(config.research)], config, interval=args.interval)
    if events.empty:
        log.error("No events detected")
        return 1

    strategies = {s.name: s for s in default_strategies()}
    strategy = strategies.get(args.strategy)
    if strategy is None:
        log.error("Unknown strategy %r; available: %s", args.strategy, ", ".join(sorted(strategies)))
        return 2

    policies = default_exit_policies()
    scenarios = load_scenarios(config.risk)
    sizing = SizingConfig.from_config(config.risk)
    limits = RiskLimits.from_config(config.risk)

    def backtest(events_slice: pd.DataFrame, policy):
        engine = Backtester(
            ExecutionModel(scenarios[headline], seed=seed), policy, sizing, limits,
            starting_equity=starting_equity, interval=args.interval,
            circuit_breakers=config.get("risk", "circuit_breakers", default={}),
        )
        return engine.run(events_slice, dataset.features, strategy, seed=seed)

    def select_on_train(train_events: pd.DataFrame):
        """Choose the exit policy with the best in-sample net expectancy."""
        best, best_score = None, float("-inf")
        for policy in policies:
            trades = backtest(train_events, policy).trades
            stats = M.compute_trade_metrics(trades, starting_equity)
            if stats["n_trades"] < 10:
                continue
            score = stats["net_expectancy"]
            if score is not None and score > best_score:
                best, best_score = policy, score
        return best or policies[0]

    def evaluate_on_test(test_events: pd.DataFrame, policy):
        stats = M.compute_trade_metrics(backtest(test_events, policy).trades, starting_equity)
        stats["exit_policy"] = policy.name
        return stats

    windows = walk_forward_windows(
        events,
        train_months=args.train_months or int(config.get("research", "walk_forward", "train_months", default=12)),
        test_months=args.test_months or int(config.get("research", "walk_forward", "test_months", default=3)),
        step_months=args.step_months or int(config.get("research", "walk_forward", "step_months", default=3)),
        embargo_minutes=int(config.get("research", "splits", "embargo_minutes", default=2880)),
    )
    if not windows:
        span_days = (events["event_time"].max() - events["event_time"].min()).total_seconds() / 86400
        log.error(
            "No walk-forward windows fit in %.0f days of events. Use shorter --train-months/--test-months "
            "or download more history.", span_days,
        )
        return 1

    results = run_walk_forward(
        events, windows, select_on_train, evaluate_on_test,
        min_train_events=int(config.get("research", "walk_forward", "min_train_events", default=100)),
    )
    summary = aggregate_walk_forward(results)

    stability = pd.DataFrame()
    if "exit_policy" in results.columns and "net_expectancy" in results.columns:
        stability = (
            results[results.get("status") == "ok"]
            .groupby("exit_policy")
            .agg(
                n_windows=("window", "count"),
                mean_net_expectancy=("net_expectancy", "mean"),
                worst_net_expectancy=("net_expectancy", "min"),
                total_trades=("n_trades", "sum"),
            )
            .reset_index()
        )

    store.write_frame("walk_forward_results.csv", results)
    if not stability.empty:
        store.write_frame("parameter_stability.csv", stability)
    store.write_json("walk_forward_summary.json", summary)

    print(f"\nWalk-forward windows: {len(windows)}")
    display = [c for c in ["window", "train_start", "test_start", "status", "exit_policy",
                           "n_trades", "win_rate", "net_expectancy", "max_drawdown"] if c in results.columns]
    print(results[display].to_string(index=False))
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nArtefacts written to {store.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
