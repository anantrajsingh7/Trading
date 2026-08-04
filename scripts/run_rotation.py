#!/usr/bin/env python3
"""Strategy 7: cross-sectional relative-strength rotation.

Why this is the next experiment and not another entry family
------------------------------------------------------------
Everything rejected so far - the impulse hypothesis, trend continuation,
compression breakout, Donchian - shares one economic shape: a short holding
period, a signal worth tens of basis points, and a 77 bps round trip charged
against every one of them. Stage 1 of the exit research put a number on it. The
best family's signal was worth about 34 bps gross at its best horizon. The
problem was never that the signals were informationless; it was that each trade
had to clear a toll larger than the information it carried.

Rotation attacks the toll rather than the signal - in principle. The default
grid does not, and the turnover table proves it: rebalancing every 24 hours over
20 markets produced 128-341 position turns per slot per year and an annual cost
drag of 99%-262% of capital. That is not low turnover, and the 0-of-24 result it
produced is a result about a high-turnover book, not about rotation as an idea.

The drag is set by the holding period and nothing else. At 77 bps, holding for
48 hours costs about 140% of capital a year; a week costs 40%; a month costs 9%.
``--allow-long-holds`` adds the weekly, fortnightly and monthly variants that
actually test the thesis. See ``scripts/cost_structure.py`` for the arithmetic.

``--max-holding-hours`` enforces a hard cap independent of the rebalance
schedule, so a one-week limit can be tested honestly: a position the ranking
would have retained is force-closed and reopened, and the backtester charges the
full round trip for it. That is the real price of a holding limit, and it is
charged rather than assumed away.

The turnover diagnostic prints before any return figure, so the bar a variant
has to clear is visible before its result is.

    python scripts/run_rotation.py --max-markets 20
    python scripts/run_rotation.py --max-markets 20 --allow-long-holds --max-holding-hours 168
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
from bitvavo_momentum.exit_research import (  # noqa: E402
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
from bitvavo_momentum.rotation import (  # noqa: E402
    default_rotation_strategies,
    low_turnover_rotation_strategies,
)
from bitvavo_momentum.signal_strategies import compute_setup_features  # noqa: E402
from bitvavo_momentum.strategies import ExitPolicy, ImmediateEntry  # noqa: E402
from bitvavo_momentum.timeframes import SETUP_TF, resample_ohlcv  # noqa: E402
from bitvavo_momentum.walk_forward import apply_split, chronological_splits  # noqa: E402

# Stage-1 horizons, in minutes: 6h out to 14 days. The default 48-hour grid
# cannot distinguish a ranking whose value compounds over a week from one that
# peaks after a day, and that distinction decides whether a weekly strategy is
# worth building.
LONG_HORIZONS: tuple[int, ...] = (
    6 * 60, 12 * 60, 24 * 60, 48 * 60, 72 * 60, 120 * 60, 168 * 60, 240 * 60, 336 * 60,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--setup-tf", default=SETUP_TF)
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--rebalance-hours", type=int, nargs="*", default=None,
                        help="override the rebalance grid, e.g. --rebalance-hours 24 72 168")
    parser.add_argument("--max-holding-hours", type=float, default=None,
                        help="force-close any position after this many hours, independent of "
                             "the rebalance schedule (e.g. 168 for a strict one-week cap)")
    parser.add_argument("--allow-long-holds", action="store_true",
                        help="add weekly/fortnightly/monthly rebalance variants. These hold "
                             "longer than the spec's 48-hour maximum; opt in deliberately.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:  # noqa: PLR0915 - a linear protocol reads better in one place
    args = parse_args()
    load_dotenv_if_present()
    log = setup_logging(args.log_level, "logs/rotation.log")
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
    if len(features_by_market) < 8:
        print(f"BLOCKED: rotation needs a cross-section. Only {len(features_by_market)} markets "
              "have enough history; ranking fewer than 8 is not a cross-section.")
        return 1
    log.info("Features ready for %d markets", len(features_by_market))

    strategies = default_rotation_strategies()
    if args.allow_long_holds:
        strategies = strategies + low_turnover_rotation_strategies()
        print("\nNOTE: --allow-long-holds adds variants that rebalance every 7, 14 and 30")
        print("days. Without --max-holding-hours their positions can be held that long, which")
        print("exceeds a one-week holding limit. Pass --max-holding-hours 168 to enforce one")
        print("week; the fortnightly and monthly variants then serve only as a reference for")
        print("what the holding limit costs.")
    if args.rebalance_hours:
        strategies = [s for s in strategies
                      if s.config.rebalance_minutes // 60 in set(args.rebalance_hours)]
        if not strategies:
            print("No rotation variant matches the requested rebalance hours.")
            return 1

    # -- turnover first: the whole thesis is that this is small ---------------
    turnover_rows = []
    for strategy in strategies:
        stats = strategy.turnover(features_by_market, round_trip_cost_bps=cost_bps)
        if stats:
            turnover_rows.append(stats)
    if turnover_rows:
        turnover = pd.DataFrame(turnover_rows).sort_values("annual_cost_drag")
        store.write_frame("rotation_turnover.csv", turnover)
        print(f"\n{'=' * 100}")
        print(f"TURNOVER AND COST DRAG - {cost_bps:.0f} bps round trip")
        print(f"{'=' * 100}")
        print("annual_cost_drag is the share of TOTAL capital spent on execution per year.")
        print("A strategy must beat this before it has earned anything.\n")
        print(turnover[["strategy", "avg_positions_held", "round_trips_per_year",
                        "turns_per_slot_per_year", "annual_cost_drag"]].to_string(index=False))

    # -- signals --------------------------------------------------------------
    all_signals = []
    for strategy in strategies:
        signals = strategy.generate(features_by_market)
        if signals.empty:
            log.warning("%s produced no signals", strategy.name)
            continue
        log.info("%s: %d signals", strategy.name, len(signals))
        all_signals.append(signals)
    if not all_signals:
        print("BLOCKED: no rotation variant produced a single signal. Check that the universe "
              "clears min_universe_size and the liquidity filters.")
        return 1

    signals = pd.concat(all_signals, ignore_index=True).sort_values("event_time")
    store.write_frame("rotation_signal_counts.csv",
                      signals.groupby("event_spec").size().rename("n_signals").reset_index())

    splits = chronological_splits(
        signals,
        train_fraction=float(config.get("research", "splits", "train_fraction", default=0.5)),
        validation_fraction=float(config.get("research", "splits", "validation_fraction", default=0.25)),
        embargo_minutes=int(config.get("research", "splits", "embargo_minutes", default=2880)),
    )
    train = apply_split(signals, splits.train)
    validation = apply_split(signals, splits.validation)
    log.info("Signals per split: train=%d validation=%d (test withheld)", len(train), len(validation))
    if train.empty:
        print("BLOCKED: the training split contains no rotation signals.")
        return 1

    # -- stage 1: is the ranking informative before any exit rule? ------------
    # Horizons run to 14 days deliberately. The required gross return per trade
    # is the round-trip cost and does not shrink with holding period; what a
    # longer hold buys is more time to accumulate it. So the decisive question
    # for a weekly strategy is whether the ranking's value keeps compounding out
    # to a week, or peaks after a day and decays - and a 48-hour cap cannot see
    # the difference.
    hold_minutes = int(max(LONG_HORIZONS))
    outcomes = signal_outcomes(train, features_by_market, LONG_HORIZONS,
                               interval=args.setup_tf, max_holding_minutes=hold_minutes)
    if not outcomes.empty:
        forward = forward_return_table(outcomes, LONG_HORIZONS, round_trip_cost_bps=cost_bps)
        excursions = excursion_table(outcomes, max_holding_minutes=hold_minutes)
        verdict = stage_one_verdict(forward, excursions)
        store.write_frame("rotation_stage1_verdict.csv", verdict)
        store.write_frame("rotation_stage1_forward_returns.csv", forward)
        print(f"\n{'=' * 100}")
        print("STAGE 1 - RANKING VALUE WITH NO EXIT RULE (train split)")
        print(f"{'=' * 100}")
        print(f"Horizons from 6 hours to {hold_minutes // 1440} days. Each trade must gross")
        print(f"{cost_bps:.0f} bps to break even regardless of how long it is held, so the")
        print("question is whether the ranking's value keeps accruing with time.\n")
        print(verdict.to_string(index=False))

        print("\nHow the ranking's value evolves with holding period (best 3 variants):")
        top = verdict.head(3)["event_spec"].tolist()
        curve = forward[forward["event_spec"].isin(top)].sort_values(
            ["event_spec", "horizon_minutes"])
        columns = ["event_spec", "horizon_hours", "n_signals", "gross_mean",
                   "net_mean", "hit_rate", "beats_cost"]
        print(curve[[c for c in columns if c in curve.columns]].to_string(index=False))
        if bool(forward["beats_cost"].any()):
            print("\nAt least one variant/horizon clears the cost floor gross. That is a "
                  "candidate, not a result: it still has to survive the backtest below, "
                  "then validation.")

    # -- backtest -------------------------------------------------------------
    # The exit is the rebalance, so the only exit rule is the clock plus a
    # protective stop. A take-profit would fight the strategy: rotation's whole
    # premise is holding the leader until it stops leading.
    policies = [
        ExitPolicy(take_profit_pct=None, stop_loss_pct=None, atr_stop_multiple=2.5,
                   time_stop_minutes=None, name="atr2.5_hold_to_rebalance"),
        ExitPolicy(take_profit_pct=None, stop_loss_pct=None, atr_stop_multiple=2.5,
                   breakeven_after_pct=0.03, time_stop_minutes=None,
                   name="atr2.5_be3_hold_to_rebalance"),
        ExitPolicy(take_profit_pct=None, stop_loss_pct=None, chandelier_atr_multiple=3.0,
                   time_stop_minutes=None, name="chandelier3_hold_to_rebalance"),
    ]

    rows: list[dict] = []
    trade_log: list[pd.DataFrame] = []
    total, done, started = len(strategies) * len(policies), 0, time.time()
    for strategy in strategies:
        subset = train[train["event_spec"] == strategy.name]
        if subset.empty:
            continue
        # Hold to the next rebalance, unless a shorter hard cap is requested.
        # A cap below the rebalance period forces a close-and-reopen on a name
        # the ranking would have retained, and the backtester charges the full
        # round trip for it - which is the real cost of a strict holding limit,
        # not a modelling artefact.
        holding_minutes = strategy.config.rebalance_minutes
        if args.max_holding_hours:
            holding_minutes = min(holding_minutes, int(args.max_holding_hours * 60))
        for policy in policies:
            engine = Backtester(
                ExecutionModel(scenarios[headline], seed=seed), policy, sizing, limits,
                starting_equity=starting_equity, interval=args.setup_tf,
                max_holding_minutes=holding_minutes,
                circuit_breakers=config.get("risk", "circuit_breakers", default={}),
            )
            result = engine.run(subset, features_by_market, ImmediateEntry(), seed=seed)
            done += 1
            if done % 3 == 0:
                elapsed = time.time() - started
                log.info("  %d/%d combinations (%.0fs elapsed)", done, total, elapsed)
            if result is None or result.trades.empty:
                continue
            row = M.compute_trade_metrics(result.trades, starting_equity)
            row.update({
                "strategy": strategy.name, "family": strategy.family,
                "exit_policy": policy.name, "rebalance_hours": holding_minutes // 60,
                "scenario": headline, "split": "train",
            })
            closed = result.closed_trades()
            if not closed.empty:
                row.update(independent_trade_count(closed))
                trade_log.append(closed)
            rows.append(row)

    if not rows:
        print("No rotation variant produced any trades on the training split.")
        return 1

    table = pd.DataFrame(rows).sort_values("net_expectancy", ascending=False).reset_index(drop=True)
    store.write_frame("rotation_train.csv", table)
    print(f"\n{'=' * 100}")
    print(f"ROTATION - TRAIN SPLIT - {headline} costs ({cost_bps:.0f} bps round trip)")
    print(f"{'=' * 100}")
    columns = ["strategy", "exit_policy", "n_trades", "n_clusters", "win_rate",
               "net_expectancy", "profit_factor", "max_drawdown", "total_return"]
    print(table[[c for c in columns if c in table.columns]].head(24).to_string(index=False))

    positive = table[table["net_expectancy"] > 0]
    print(f"\nCombinations with positive net expectancy on train: {len(positive)} of {len(table)}")

    if trade_log:
        reasons = exit_reason_breakdown(pd.concat(trade_log, ignore_index=True))
        if not reasons.empty:
            store.write_frame("rotation_exit_reasons.csv", reasons)
            pooled = reasons.groupby("exit_reason").agg(
                n_trades=("n_trades", "sum"), mean_net_pct=("mean_net_pct", "mean"),
            ).reset_index()
            pooled["share"] = pooled["n_trades"] / pooled["n_trades"].sum()
            print("\nHow trades ended (all variants pooled):")
            print(pooled.sort_values("n_trades", ascending=False).to_string(index=False))

    # -- validation for survivors only ---------------------------------------
    survivors = positive[positive["n_trades"] >= 30]
    if survivors.empty:
        print("\nNo rotation variant cleared the training filter (positive expectancy, "
              ">=30 trades). Nothing is promoted; the test set stays closed.")
    else:
        validation_rows = []
        print(f"\nPromoting {len(survivors)} combination(s) to validation...")
        for _, candidate in survivors.iterrows():
            policy = next(p for p in policies if p.name == candidate["exit_policy"])
            strategy = next(s for s in strategies if s.name == candidate["strategy"])
            subset = validation[validation["event_spec"] == strategy.name]
            if subset.empty:
                continue
            for scenario_name in (headline, "stress"):
                engine = Backtester(
                    ExecutionModel(scenarios[scenario_name], seed=seed), policy, sizing, limits,
                    starting_equity=starting_equity, interval=args.setup_tf,
                    max_holding_minutes=(
                        min(strategy.config.rebalance_minutes, int(args.max_holding_hours * 60))
                        if args.max_holding_hours else strategy.config.rebalance_minutes
                    ),
                    circuit_breakers=config.get("risk", "circuit_breakers", default={}),
                )
                result = engine.run(subset, features_by_market, ImmediateEntry(), seed=seed)
                if result is None or result.trades.empty:
                    continue
                row = M.compute_trade_metrics(result.trades, starting_equity)
                row.update({"strategy": strategy.name, "exit_policy": policy.name,
                            "scenario": scenario_name, "split": "validation"})
                concentration = M.concentration_report(result.trades)
                row.update({f"conc_{k}": v for k, v in concentration.items()
                            if isinstance(v, int | float)})
                if scenario_name == headline:
                    row["rejection_reasons"] = "; ".join(
                        M.rejection_reasons(row, concentration, min_trades=100)
                    ) or "none"
                validation_rows.append(row)
        if validation_rows:
            validation_table = pd.DataFrame(validation_rows)
            store.write_frame("rotation_validation.csv", validation_table)
            print(f"\n{'=' * 100}")
            print("VALIDATION SPLIT")
            print(f"{'=' * 100}")
            columns = ["strategy", "exit_policy", "scenario", "n_trades", "win_rate",
                       "net_expectancy", "profit_factor", "max_drawdown", "rejection_reasons"]
            print(validation_table[[c for c in columns if c in validation_table.columns]]
                  .to_string(index=False))

    baseline = matched_random_baseline(train, dataset.candles, (1440, 2880, 10080), "1m", seed=seed)
    if not baseline.empty:
        store.write_frame("rotation_baseline.csv", baseline)
        print("\nMatched-random baseline over the same signals (gross, for reference):")
        print(baseline[["horizon_minutes", "n_samples", "mean_return", "hit_rate"]].to_string(index=False))

    print(f"\nArtefacts written to {store.root}")
    print("The untouched test set was NOT read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
