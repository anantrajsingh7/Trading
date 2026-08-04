#!/usr/bin/env python3
"""Milestone 4: test each strategy family independently on real data.

Runs the signal-scanning families (spec Strategies 1, 2, 8) through the existing
validation protocol: chronological train/validation split, the realistic and
stress execution scenarios, portfolio limits applied chronologically, and the
automatic rejection rules.

Deliberately NOT here:

* the impulse-conditioned families (spec Strategies 3-6) - closed by the event
  study, which found forward returns degrading monotonically with impulse size
  and speed and a best-of-288 cell of +0.42% against a 77 bps cost floor;
* parameter optimisation - families are compared at their default settings
  first, and only survivors earn a parameter sweep. Optimising 16 variants x 12
  stops x 11 break-evens before knowing whether any family works at all is how
  a search of that size manufactures a winner from noise.

The untouched test set is never read here.

    python scripts/run_strategy_families.py
    python scripts/run_strategy_families.py --max-markets 20 --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from bitvavo_momentum import metrics as M  # noqa: E402
from bitvavo_momentum.backtester import Backtester  # noqa: E402
from bitvavo_momentum.baseline import matched_random_baseline  # noqa: E402
from bitvavo_momentum.config import Config, load_dotenv_if_present  # noqa: E402
from bitvavo_momentum.execution_model import ExecutionModel, load_scenarios  # noqa: E402
from bitvavo_momentum.exit_research import exit_reason_breakdown  # noqa: E402
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

# Holding periods the spec requires comparing, in minutes.
HOLDING_PERIODS = (6 * 60, 12 * 60, 24 * 60, 36 * 60, 48 * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--setup-tf", default=SETUP_TF)
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--holding-minutes", type=int, default=48 * 60)
    parser.add_argument("--compare-holding-periods", action="store_true",
                        help="also sweep 6/12/24/36/48h for the surviving families")
    parser.add_argument("--apply-veto", action="store_true",
                        help="apply the exhaustion veto to every family")
    parser.add_argument("--quick", action="store_true", help="one exit policy instead of three")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def exit_policies(quick: bool, holding_minutes: int) -> list[ExitPolicy]:
    """A small, economically distinct exit set - not a grid search."""
    policies = [
        ExitPolicy(take_profit_pct=0.10, stop_loss_pct=None, atr_stop_multiple=1.5,
                   time_stop_minutes=holding_minutes, name="tp10_atr1.5_t48h"),
    ]
    if quick:
        return policies
    return policies + [
        ExitPolicy(take_profit_pct=0.15, stop_loss_pct=None, atr_stop_multiple=2.0,
                   time_stop_minutes=holding_minutes, name="tp15_atr2_t48h"),
        ExitPolicy(take_profit_pct=None, stop_loss_pct=None, atr_stop_multiple=1.5,
                   chandelier_atr_multiple=3.0, time_stop_minutes=holding_minutes,
                   name="chandelier3_t48h"),
    ]


def main() -> int:  # noqa: PLR0915 - a linear protocol reads better in one place
    args = parse_args()
    load_dotenv_if_present()
    log = setup_logging(args.log_level, "logs/strategy_families.log")
    config = Config.load()
    config.ensure_dirs()

    dataset = build_dataset(config, source=args.source, interval="1m", max_markets=args.max_markets)
    store = result_store_for(config, dataset)
    if not dataset.candles:
        log.error("No usable market data. Run scripts/download_history.py first.")
        print("BLOCKED: no real candle data in data/processed. Nothing computed.")
        return 1

    starting_equity = float(config.get("risk", "portfolio", "starting_equity_eur", default=10000.0))
    headline = config.get("risk", "headline_scenario", default="realistic")
    scenarios = load_scenarios(config.risk)
    cost_bps = ExecutionModel(scenarios[headline]).describe()["round_trip_cost_bps_minimum"]
    seed = int(config.get("research", "robustness", "random_seed", default=20260802))
    sizing = SizingConfig.from_config(config.risk)
    limits = RiskLimits.from_config(config.risk)

    # -- build setup-timeframe features -------------------------------------
    log.info("Resampling %d markets to %s and computing setup features...",
             len(dataset.candles), args.setup_tf)
    features_by_market: dict[str, pd.DataFrame] = {}
    for market, frame in dataset.candles.items():
        resampled = resample_ohlcv(frame, args.setup_tf)
        if len(resampled) < 300:
            continue
        computed = compute_setup_features(resampled, args.setup_tf)
        if not computed.empty:
            features_by_market[market] = computed
    log.info("Features ready for %d markets", len(features_by_market))
    if not features_by_market:
        print("BLOCKED: no market had enough setup-timeframe history.")
        return 1

    # -- generate signals ----------------------------------------------------
    strategies = default_signal_strategies()
    signals = generate_signals(
        strategies, features_by_market,
        apply_exhaustion_veto=args.apply_veto,
        eligibility_by_market=None,   # eligibility already applied upstream
    )
    if signals.empty:
        print("No signals generated by any family. Nothing to evaluate.")
        return 1

    counts = signals.groupby("event_spec").size().rename("n_signals").reset_index()
    store.write_frame("signal_counts.csv", counts)
    print("\nSignals generated per variant:")
    print(counts.to_string(index=False))

    # -- chronological split -------------------------------------------------
    splits = chronological_splits(
        signals,
        train_fraction=float(config.get("research", "splits", "train_fraction", default=0.5)),
        validation_fraction=float(config.get("research", "splits", "validation_fraction", default=0.25)),
        embargo_minutes=int(config.get("research", "splits", "embargo_minutes", default=2880)),
    )

    def evaluate(slice_signals: pd.DataFrame, strategy_name: str, policy: ExitPolicy,
                 scenario_name: str, holding_minutes: int):
        subset = slice_signals[slice_signals["event_spec"] == strategy_name]
        if subset.empty:
            return None
        engine = Backtester(
            ExecutionModel(scenarios[scenario_name], seed=seed), policy, sizing, limits,
            starting_equity=starting_equity, interval=args.setup_tf,
            max_holding_minutes=holding_minutes,
            circuit_breakers=config.get("risk", "circuit_breakers", default={}),
        )
        return engine.run(subset, features_by_market, ImmediateEntry(), seed=seed)

    train = apply_split(signals, splits.train)
    validation = apply_split(signals, splits.validation)
    log.info("Signals per split: train=%d validation=%d (test withheld)", len(train), len(validation))

    # -- family comparison on TRAIN data only -------------------------------
    policies = exit_policies(args.quick, args.holding_minutes)
    rows: list[dict] = []
    trade_log: list[pd.DataFrame] = []
    total = len(strategies) * len(policies)
    done = 0
    started = time.time()
    for strategy in strategies:
        n_signals = int((train["event_spec"] == strategy.name).sum())
        log.info("[%d/%d] %s: %d train signals x %d exit policies",
                 done + 1, total, strategy.name, n_signals, len(policies))
        for policy in policies:
            result = evaluate(train, strategy.name, policy, headline, args.holding_minutes)
            done += 1
            if done % 5 == 0:
                elapsed = time.time() - started
                rate = elapsed / max(1, done)
                log.info("  %d/%d combinations (%.0fs elapsed, ~%.0fs remaining)",
                         done, total, elapsed, rate * (total - done))
            if result is None:
                continue
            row = M.compute_trade_metrics(result.trades, starting_equity)
            row.update({
                "strategy": strategy.name, "family": strategy.family,
                "exit_policy": policy.name, "scenario": headline, "split": "train",
                "holding_minutes": args.holding_minutes,
            })
            row.update({f"funnel_{k}": v for k, v in result.funnel.items()})
            closed = result.closed_trades()
            if not closed.empty:
                row.update(independent_trade_count(closed))
                trade_log.append(closed)
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        print("No family produced any trades on the training split.")
        return 1
    table = table.sort_values("net_expectancy", ascending=False).reset_index(drop=True)
    store.write_frame("strategy_families_train.csv", table)

    print(f"\n{'=' * 92}")
    print(f"FAMILY COMPARISON - TRAIN SPLIT ONLY - {headline} costs ({cost_bps:.0f} bps round trip)")
    print(f"{'=' * 92}")
    columns = ["strategy", "exit_policy", "n_trades", "n_clusters", "win_rate",
               "net_expectancy", "profit_factor", "max_drawdown"]
    print(table[[c for c in columns if c in table.columns]].head(20).to_string(index=False))

    positive = table[table["net_expectancy"] > 0]
    print(f"\nCombinations with positive net expectancy on train: {len(positive)} of {len(table)}")

    # Which exit actually closed the trades? A family whose trades are almost all
    # time-stops was tested against the clock, not against its stop and target.
    if trade_log:
        reasons = exit_reason_breakdown(pd.concat(trade_log, ignore_index=True))
        if not reasons.empty:
            store.write_frame("exit_reason_breakdown.csv", reasons)
            print("\nHow trades actually ended (all variants pooled):")
            pooled = reasons.groupby("exit_reason").agg(
                n_trades=("n_trades", "sum"),
                mean_net_pct=("mean_net_pct", "mean"),
            ).reset_index()
            pooled["share"] = pooled["n_trades"] / pooled["n_trades"].sum()
            print(pooled.sort_values("n_trades", ascending=False).to_string(index=False))

    # -- promote survivors to validation ------------------------------------
    survivors = positive[positive["n_trades"] >= 30]
    validation_rows: list[dict] = []
    if survivors.empty:
        print("\nNo family cleared even the training filter (positive expectancy, >=30 trades).")
    else:
        print(f"\nPromoting {len(survivors)} combination(s) to validation...")
        for _, candidate in survivors.iterrows():
            policy = next(p for p in policies if p.name == candidate["exit_policy"])
            for scenario_name in (headline, "stress"):
                result = evaluate(validation, candidate["strategy"], policy,
                                  scenario_name, args.holding_minutes)
                if result is None:
                    continue
                row = M.compute_trade_metrics(result.trades, starting_equity)
                row.update({
                    "strategy": candidate["strategy"], "family": candidate["family"],
                    "exit_policy": policy.name, "scenario": scenario_name, "split": "validation",
                })
                concentration = M.concentration_report(result.trades)
                row.update({f"conc_{k}": v for k, v in concentration.items()
                            if isinstance(v, int | float)})
                if scenario_name == headline:
                    row["rejection_reasons"] = "; ".join(
                        M.rejection_reasons(row, concentration, min_trades=100)
                    ) or "none"
                validation_rows.append(row)

    validation_table = pd.DataFrame(validation_rows)
    if not validation_table.empty:
        store.write_frame("strategy_families_validation.csv", validation_table)
        print(f"\n{'=' * 92}")
        print("VALIDATION SPLIT")
        print(f"{'=' * 92}")
        columns = ["strategy", "exit_policy", "scenario", "n_trades", "win_rate",
                   "net_expectancy", "profit_factor", "max_drawdown"]
        print(validation_table[[c for c in columns if c in validation_table.columns]].to_string(index=False))
        rejected = validation_table[validation_table.get("rejection_reasons", "").fillna("") != "none"]
        if not rejected.empty and "rejection_reasons" in rejected.columns:
            print("\nRejection reasons:")
            for _, row in rejected.iterrows():
                if isinstance(row.get("rejection_reasons"), str) and row["rejection_reasons"] != "none":
                    print(f"  {row['strategy']} + {row['exit_policy']}: {row['rejection_reasons']}")

    # -- holding-period sweep on survivors only -----------------------------
    if args.compare_holding_periods and not survivors.empty:
        holding_rows = []
        for _, candidate in survivors.head(3).iterrows():
            policy_base = next(p for p in policies if p.name == candidate["exit_policy"])
            for minutes in HOLDING_PERIODS:
                policy = ExitPolicy(
                    take_profit_pct=policy_base.take_profit_pct,
                    stop_loss_pct=policy_base.stop_loss_pct,
                    atr_stop_multiple=policy_base.atr_stop_multiple,
                    chandelier_atr_multiple=policy_base.chandelier_atr_multiple,
                    time_stop_minutes=minutes,
                    name=f"{policy_base.name}_h{minutes // 60}h",
                )
                result = evaluate(train, candidate["strategy"], policy, headline, minutes)
                if result is None:
                    continue
                row = M.compute_trade_metrics(result.trades, starting_equity)
                row.update({"strategy": candidate["strategy"], "holding_hours": minutes // 60})
                holding_rows.append(row)
        if holding_rows:
            holding = pd.DataFrame(holding_rows)
            store.write_frame("holding_period_comparison.csv", holding)
            print("\nHOLDING-PERIOD COMPARISON (train, survivors only):")
            print(holding[["strategy", "holding_hours", "n_trades", "net_expectancy",
                           "max_drawdown"]].to_string(index=False))

    # -- baseline reference --------------------------------------------------
    baseline = matched_random_baseline(signals, dataset.candles, (240, 1440, 2880), "1m", seed=seed)
    if not baseline.empty:
        store.write_frame("strategy_families_baseline.csv", baseline)
        print("\nMatched-random baseline over the same signals (gross, for reference):")
        print(baseline[["horizon_minutes", "n_samples", "mean_return", "hit_rate"]].to_string(index=False))

    print(f"\nArtefacts written to {store.root}")
    print("\nThe untouched test set was NOT read. Families that survive validation earn a "
          "parameter sweep; families that do not are rejected here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
