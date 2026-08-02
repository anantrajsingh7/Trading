#!/usr/bin/env python3
"""Phases 3-8: strategy comparison, chronological splits and robustness.

Protocol enforced by this script:

1. build events (Phase 2);
2. split the time axis into train / validation / test with an embargo;
3. compare every strategy on **train** data across all three execution scenarios;
4. select on **validation** data;
5. evaluate the selection on the **test** set only if it has been explicitly
   unlocked in ``config/research.yaml`` - otherwise report that it stays locked;
6. run robustness: bootstrap, Monte Carlo, permutation vs random entries,
   deflated Sharpe, concentration, cost sensitivity;
7. apply the Phase 7 rejection rules and write the verdict.

    python scripts/run_backtest.py --source real
    python scripts/run_backtest.py --source synthetic   # pipeline self-test only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bitvavo_momentum import metrics as M  # noqa: E402
from bitvavo_momentum import robustness as R  # noqa: E402
from bitvavo_momentum.backtester import Backtester  # noqa: E402
from bitvavo_momentum.config import Config, load_dotenv_if_present  # noqa: E402
from bitvavo_momentum.event_detector import primary_spec_from_config  # noqa: E402
from bitvavo_momentum.execution_model import ExecutionModel, load_scenarios  # noqa: E402
from bitvavo_momentum.logging_utils import setup_logging  # noqa: E402
from bitvavo_momentum.pipeline import build_dataset, build_events, result_store_for  # noqa: E402
from bitvavo_momentum.reporting import (  # noqa: E402
    RunContext,
    build_assumptions_document,
    build_html_report,
    build_rejected_strategies,
    build_research_report,
    drawdown_figure,
    equity_curve_figure,
    monte_carlo_figure,
    trade_distribution_figure,
    write_all,
)
from bitvavo_momentum.risk_manager import RiskLimits, SizingConfig  # noqa: E402
from bitvavo_momentum.strategies import (  # noqa: E402
    ExitPolicy,
    default_exit_policies,
    default_strategies,
    rank_cross_sectional,
)
from bitvavo_momentum.synthetic import SyntheticConfig  # noqa: E402
from bitvavo_momentum.walk_forward import apply_split, chronological_splits, unlock_test_set  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--synthetic-markets", type=int, default=8)
    parser.add_argument("--synthetic-days", type=int, default=120)
    parser.add_argument("--max-holding-minutes", type=int, default=1440)
    parser.add_argument("--quick", action="store_true", help="fewer exit policies, for a fast check")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:  # noqa: PLR0915 - a linear protocol reads better in one place
    args = parse_args()
    load_dotenv_if_present()
    log = setup_logging(args.log_level, "logs/backtest.log")
    config = Config.load()
    config.ensure_dirs()

    starting_equity = float(config.get("risk", "portfolio", "starting_equity_eur", default=10000.0))
    headline = config.get("risk", "headline_scenario", default="realistic")

    synthetic_config = SyntheticConfig(n_minutes=args.synthetic_days * 24 * 60) if args.source == "synthetic" else None
    dataset = build_dataset(
        config, source=args.source, interval=args.interval, max_markets=args.max_markets,
        synthetic_config=synthetic_config, n_synthetic_markets=args.synthetic_markets,
    )
    store = result_store_for(config, dataset)
    context = RunContext(
        data_source=dataset.source, n_markets=len(dataset.candles),
        first_timestamp=dataset.first_timestamp, last_timestamp=dataset.last_timestamp,
        execution_scenario=headline, is_synthetic=dataset.is_synthetic,
        has_real_data=bool(dataset.candles) and not dataset.is_synthetic,
    )

    if not dataset.candles:
        log.error("No usable market data. Run scripts/download_history.py first.")
        store.write_text("research_report.md", build_research_report(context))
        store.write_text("assumptions_and_limitations.md", build_assumptions_document(context))
        print("No data available - no backtest was run and no numbers are reported.")
        return 1

    spec = primary_spec_from_config(config.research)
    events = build_events(dataset, [spec], config, interval=args.interval)
    context.n_events = int(len(events))
    if events.empty:
        log.error("No events detected for the primary specification %s", spec.name)
        store.write_text("research_report.md", build_research_report(context))
        return 1

    events = rank_cross_sectional(events, top_n=2)

    # -- splits --------------------------------------------------------------
    splits = chronological_splits(
        events,
        train_fraction=float(config.get("research", "splits", "train_fraction", default=0.5)),
        validation_fraction=float(config.get("research", "splits", "validation_fraction", default=0.25)),
        embargo_minutes=int(config.get("research", "splits", "embargo_minutes", default=2880)),
    )
    train_events = apply_split(events, splits.train)
    validation_events = apply_split(events, splits.validation)
    test_events = apply_split(events, splits.test)
    log.info("Events per split: train=%d validation=%d test=%d",
             len(train_events), len(validation_events), len(test_events))

    scenarios = load_scenarios(config.risk)
    sizing = SizingConfig.from_config(config.risk)
    limits = RiskLimits.from_config(config.risk)
    strategies = default_strategies()
    policies = default_exit_policies()[:3] if args.quick else default_exit_policies()

    def run_one(events_slice: pd.DataFrame, strategy, policy: ExitPolicy, scenario_name: str):
        model = ExecutionModel(scenarios[scenario_name], seed=int(
            config.get("research", "robustness", "random_seed", default=20260802)))
        engine = Backtester(
            model, policy, sizing, limits, starting_equity=starting_equity,
            interval=args.interval, max_holding_minutes=args.max_holding_minutes,
            circuit_breakers=config.get("risk", "circuit_breakers", default={}),
        )
        return engine.run(events_slice, dataset.features, strategy,
                          label=f"{strategy.name}|{policy.name}|{scenario_name}")

    # -- Phase 3/4 comparison on TRAIN data only -----------------------------
    log.info("Phase 3/4: comparing %d strategies x %d exit policies on train data",
             len(strategies), len(policies))
    train_rows = []
    train_results = {}
    for strategy in strategies:
        for policy in policies:
            result = run_one(train_events, strategy, policy, headline)
            row = M.compute_trade_metrics(result.trades, starting_equity)
            row.update({"strategy": strategy.name, "exit_policy": policy.name,
                        "scenario": headline, "split": "train"})
            row.update({f"funnel_{k}": v for k, v in result.funnel.items()})
            train_rows.append(row)
            train_results[f"{strategy.name}|{policy.name}"] = result
    train_table = pd.DataFrame(train_rows)
    context.n_configurations_evaluated = len(train_table)

    min_trades = int(config.get("research", "robustness", "min_independent_trades", default=100))
    viable = train_table[train_table["n_trades"] >= min(min_trades, 30)]
    if viable.empty:
        viable = train_table[train_table["n_trades"] >= 10]

    rejections: list[dict] = []
    candidate = None
    if not viable.empty:
        ranked = viable.sort_values("net_expectancy", ascending=False)
        candidate = ranked.iloc[0].to_dict()
        for _, row in ranked.iloc[1:11].iterrows():
            rejections.append({
                "name": f"{row['strategy']} + {row['exit_policy']}",
                "config": {"strategy": row["strategy"], "exit_policy": row["exit_policy"]},
                "n_trades": int(row["n_trades"]),
                "net_expectancy": row["net_expectancy"],
                "reasons": ["lower train net expectancy than the selected candidate"],
            })

    # -- validation ----------------------------------------------------------
    validation_metrics = None
    stress_metrics = None
    headline_metrics = None
    concentration = None
    candidate_result = None
    if candidate is not None:
        strategy = next(s for s in strategies if s.name == candidate["strategy"])
        policy = next(p for p in policies if p.name == candidate["exit_policy"])
        log.info("Candidate selected on train: %s + %s", strategy.name, policy.name)

        candidate_result = run_one(validation_events, strategy, policy, headline)
        validation_metrics = M.compute_trade_metrics(candidate_result.trades, starting_equity)
        headline_metrics = validation_metrics
        stress_result = run_one(validation_events, strategy, policy, "stress")
        stress_metrics = M.compute_trade_metrics(stress_result.trades, starting_equity)
        concentration = M.concentration_report(candidate_result.trades)

    # -- test set (locked by default) ---------------------------------------
    test_metrics = None
    if candidate is not None and unlock_test_set(config.research):
        log.warning("TEST SET UNLOCKED - this evaluation should happen exactly once")
        strategy = next(s for s in strategies if s.name == candidate["strategy"])
        policy = next(p for p in policies if p.name == candidate["exit_policy"])
        test_result = run_one(test_events, strategy, policy, headline)
        test_metrics = M.compute_trade_metrics(test_result.trades, starting_equity)

    split_comparison = R.degradation_report(
        {k: candidate.get(k) for k in candidate} if candidate else {},
        validation_metrics or {},
        test_metrics,
    )

    # -- robustness ----------------------------------------------------------
    robustness_summary: dict = {}
    monte_carlo = None
    rejection_reasons: list[str] = []
    if candidate_result is not None:
        closed = candidate_result.closed_trades()
        if not closed.empty:
            returns = closed["net_return"].dropna()
            robustness_summary["bootstrap"] = R.block_bootstrap_metric(
                returns, iterations=int(config.get("research", "robustness", "bootstrap_iterations", default=2000))
            )
            monte_carlo = R.monte_carlo_paths(
                returns, starting_equity,
                iterations=int(config.get("research", "robustness", "monte_carlo_iterations", default=2000)),
            )
            robustness_summary["monte_carlo"] = {
                k: v for k, v in monte_carlo.items() if not isinstance(v, np.ndarray)
            }
            robustness_summary["independent_trades"] = R.independent_trade_count(closed)

            random_strategy = next((s for s in strategies if s.name == "Z_random"), None)
            if random_strategy is not None:
                policy = next(p for p in policies if p.name == candidate["exit_policy"])
                random_result = run_one(validation_events, random_strategy, policy, headline)
                random_closed = random_result.closed_trades()
                if not random_closed.empty:
                    robustness_summary["permutation_vs_random"] = R.permutation_test_vs_random(
                        returns, random_closed["net_return"].dropna(),
                        iterations=int(config.get("research", "robustness", "permutation_iterations", default=1000)),
                    )

            robustness_summary["deflated_sharpe"] = R.deflated_sharpe_ratio(
                sharpe=validation_metrics.get("sharpe", np.nan),
                n_trials=context.n_configurations_evaluated,
                n_observations=int(validation_metrics.get("n_trades", 0)),
            )
            regime_column = "btc_trend" if "btc_trend" in events.columns else None
            if regime_column:
                merged = closed.merge(
                    events[["market", "event_time", regime_column]],
                    on=["market", "event_time"], how="left",
                )
                robustness_summary["regime_performance"] = M.breakdown(merged, by=regime_column)

        rejection_reasons = M.rejection_reasons(
            validation_metrics or {}, concentration or {}, stress_metrics,
            min_trades=min_trades,
            max_coin_share=float(config.get("research", "robustness", "max_single_coin_profit_share", default=0.4)),
            max_month_share=float(config.get("research", "robustness", "max_single_month_profit_share", default=0.4)),
        )

    # -- verdict -------------------------------------------------------------
    if candidate is None:
        verdict = (
            "**Insufficient evidence.** No strategy/exit combination produced enough trades on the "
            "training data to support any conclusion. No strategy is approved for paper trading."
        )
    elif rejection_reasons:
        verdict = (
            f"**Rejected: not suitable for paper trading.** The best candidate "
            f"(`{candidate['strategy']}` + `{candidate['exit_policy']}`) failed these criteria on "
            "validation data:\n\n" + "\n".join(f"- {r}" for r in rejection_reasons) +
            "\n\nThe test set remains untouched. 'Insufficient evidence' is the reported outcome."
        )
    else:
        verdict = (
            f"**Candidate passes the validation-stage criteria.** `{candidate['strategy']}` + "
            f"`{candidate['exit_policy']}` under the {headline} execution scenario. This is *not* an "
            "approval to trade: the final test set has not been opened, and paper trading should "
            "precede any consideration of real capital. Approve it explicitly by writing "
            "`data/results/approved_strategy.json` before the scanner will run."
        )

    # -- outputs -------------------------------------------------------------
    all_trades = pd.concat(
        [r.trades for r in train_results.values() if not r.trades.empty], ignore_index=True
    ) if train_results else pd.DataFrame()

    strategy_comparison = (
        train_table.sort_values("net_expectancy", ascending=False).reset_index(drop=True)
        if not train_table.empty else pd.DataFrame()
    )

    report = build_research_report(
        context,
        strategy_comparison=strategy_comparison,
        headline_metrics=headline_metrics,
        stress_metrics=stress_metrics,
        split_comparison=split_comparison,
        robustness_summary=robustness_summary,
        concentration=concentration,
        rejection_reasons=rejection_reasons,
        verdict=verdict,
    )

    figures = {}
    if candidate_result is not None:
        figures["Equity curve (validation)"] = equity_curve_figure(candidate_result.equity_curve)
        figures["Drawdown (validation)"] = drawdown_figure(candidate_result.equity_curve)
        figures["Trade distribution"] = trade_distribution_figure(candidate_result.closed_trades())
    if monte_carlo and "final_equity_samples" in monte_carlo:
        figures["Monte Carlo"] = monte_carlo_figure(monte_carlo["final_equity_samples"])

    html = build_html_report(
        context, figures=figures,
        tables={"Strategy comparison (train)": strategy_comparison,
                "Train / validation / test": split_comparison},
        title="Bitvavo momentum - backtest report",
    )

    written = write_all(
        store, context, report,
        backtest_summary=train_table,
        strategy_comparison=strategy_comparison,
        trade_log=all_trades,
        event_dataset=events,
        rejected_markdown=build_rejected_strategies(rejections, context),
        assumptions_markdown=build_assumptions_document(
            context, execution_description=ExecutionModel(scenarios[headline]).describe()
        ),
        html_report=html,
    )
    store.write_json("robustness_summary.json", robustness_summary)

    # -- console summary -----------------------------------------------------
    print(f"\nEvents: {len(events)} (train {len(train_events)} / validation {len(validation_events)} / test {len(test_events)})")
    print(f"Configurations evaluated on train: {context.n_configurations_evaluated}")
    if not strategy_comparison.empty:
        cols = ["strategy", "exit_policy", "n_trades", "win_rate", "net_expectancy", "profit_factor", "max_drawdown"]
        print("\nTop 10 on TRAIN data (headline scenario):")
        print(strategy_comparison[cols].head(10).to_string(index=False))
    if validation_metrics:
        print(f"\nCandidate on VALIDATION: n={validation_metrics['n_trades']} "
              f"net_expectancy={validation_metrics['net_expectancy']:.5f} "
              f"max_dd={validation_metrics['max_drawdown']:.2%}")
    if stress_metrics:
        print(f"Under STRESS execution:     n={stress_metrics['n_trades']} "
              f"net_expectancy={stress_metrics['net_expectancy']:.5f}")
    print("\nVERDICT:")
    print(verdict)
    print(f"\nArtefacts: {', '.join(sorted(written))} -> {store.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
