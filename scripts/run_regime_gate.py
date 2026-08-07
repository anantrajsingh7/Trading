#!/usr/bin/env python3
"""Does not being long in a downtrend fix anything?

Five families and 91 configurations have been rejected, and all of them are
long-only over a training window in which the market fell - a window in which
random entries in the same markets also lost money. That makes "the entries are
bad" and "being long was bad" observationally similar, and only one of them is
fixable by a filter.

This script separates them. It classifies each day into a market regime from
BTC's trend, the universe's breadth and realised volatility - all causally, with
every label shifted a day so it is computed from data that closed before the day
it labels - and then asks two questions in order:

1. **Does the label carry information at all?** Forward returns of the same
   signals, grouped by regime. If uptrend signals returned what downtrend
   signals returned, no gate built on these labels can help, and step 2 is
   noise. This is checked before any gate is applied.

2. **Does a gate built on it do work?** Five pre-declared presets are scored by
   what they reject as well as what they keep. A gate earns nothing by admitting
   fewer trades; it earns its place only if the trades it blocks were worse.

The presets are fixed in ``regime_gate.PRESETS`` and were written before any
result was seen. Nothing here selects allowed regimes by their returns.

    python scripts/run_regime_gate.py --max-markets 20
    python scripts/run_regime_gate.py --max-markets 20 --backtest
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
from bitvavo_momentum.execution_model import ExecutionModel, load_scenarios  # noqa: E402
from bitvavo_momentum.exit_research import signal_outcomes  # noqa: E402
from bitvavo_momentum.logging_utils import setup_logging  # noqa: E402
from bitvavo_momentum.pipeline import build_dataset, result_store_for  # noqa: E402
from bitvavo_momentum.regime_gate import (  # noqa: E402
    PRESETS,
    apply_gate,
    compute_breadth,
    evaluate_gate,
)
from bitvavo_momentum.regimes import RegimeConfig, classify_regimes  # noqa: E402
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

HORIZONS: tuple[int, ...] = (6 * 60, 12 * 60, 24 * 60, 36 * 60, 48 * 60, 72 * 60, 168 * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--setup-tf", default=SETUP_TF)
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--benchmark-market", default="BTC-EUR",
                        help="market whose daily trend defines the regime (default BTC-EUR)")
    parser.add_argument("--horizon-hours", type=float, default=36.0,
                        help="horizon used to score the gate (default 36h, the best measured "
                             "for the entry families)")
    parser.add_argument("--backtest", action="store_true",
                        help="also run the full backtester gated vs ungated on the leading family")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:  # noqa: PLR0915 - a linear protocol reads better in one place
    args = parse_args()
    load_dotenv_if_present()
    log = setup_logging(args.log_level, "logs/regime_gate.log")
    config = Config.load()
    config.ensure_dirs()

    dataset = build_dataset(config, source=args.source, interval="1m", max_markets=args.max_markets)
    store = result_store_for(config, dataset)
    if not dataset.candles:
        print("BLOCKED: no real candle data in data/processed. Run scripts/download_history.py first.")
        return 1

    headline = config.get("risk", "headline_scenario", default="realistic")
    scenarios = load_scenarios(config.risk)
    cost_bps = ExecutionModel(scenarios[headline]).describe()["round_trip_cost_bps_minimum"]
    seed = int(config.get("research", "robustness", "random_seed", default=20260802))

    # -- benchmark for the regime -------------------------------------------
    benchmark = dataset.candles.get(args.benchmark_market)
    if benchmark is None:
        available = ", ".join(sorted(dataset.candles)[:10])
        print(f"BLOCKED: {args.benchmark_market} is not in the dataset, so the market regime "
              f"cannot be classified. Available markets include: {available}")
        print("Pass --benchmark-market with one of them, or download BTC-EUR history.")
        return 1

    btc_daily = resample_ohlcv(benchmark, "1d").reset_index()
    if "timestamp" not in btc_daily.columns:
        btc_daily = btc_daily.rename(columns={btc_daily.columns[0]: "timestamp"})
    log.info("Benchmark %s: %d daily bars", args.benchmark_market, len(btc_daily))

    # -- setup features ------------------------------------------------------
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

    breadth = compute_breadth(features_by_market, lookback_minutes=1440, freq_minutes=60)
    regimes = classify_regimes(btc_daily, RegimeConfig.from_config(config.research), breadth)
    if regimes.empty:
        print("BLOCKED: regime classification produced nothing. Check the benchmark history.")
        return 1
    store.write_frame("regime_labels.csv", regimes.reset_index())

    labelled = regimes.dropna(subset=["btc_trend"])
    print(f"\n{'=' * 96}")
    print("REGIME CLASSIFICATION")
    print(f"{'=' * 96}")
    print(f"{len(labelled)} labelled days from {args.benchmark_market}. Every label is shifted one")
    print("day, so the label on a given day was computed from data that closed before it.\n")
    for column in ("btc_trend", "volatility_regime", "risk_regime"):
        if column in labelled.columns:
            counts = labelled[column].value_counts(dropna=False)
            share = (counts / counts.sum() * 100).round(1)
            print(f"{column}: " + ", ".join(f"{k} {v}%" for k, v in share.items()))

    # -- signals and split ---------------------------------------------------
    strategies = default_signal_strategies()
    signals = generate_signals(strategies, features_by_market, apply_exhaustion_veto=False,
                               eligibility_by_market=None)
    if signals.empty:
        print("No signals generated. Nothing to gate.")
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

    outcomes = signal_outcomes(train, features_by_market, HORIZONS,
                               interval=args.setup_tf, max_holding_minutes=int(max(HORIZONS)))
    if outcomes.empty:
        print("BLOCKED: no signal had a full forward window inside the data.")
        return 1

    horizon_minutes = int(args.horizon_hours * 60)
    horizon_column = f"fwd_return_{horizon_minutes}m"
    if horizon_column not in outcomes.columns:
        horizon_column = f"fwd_return_{HORIZONS[3]}m"
        print(f"\n(--horizon-hours not in the computed grid; scoring on {horizon_column} instead.)")

    report = evaluate_gate(outcomes, regimes, horizon_column, PRESETS, cost_bps)
    store.write_frame("regime_gate_by_regime.csv", report.by_regime)
    store.write_frame("regime_gate_by_preset.csv", report.by_preset)

    # -- question 1 ----------------------------------------------------------
    print(f"\n{'=' * 96}")
    print(f"STEP 1 - DOES THE REGIME LABEL CARRY INFORMATION? ({horizon_column}, train split)")
    print(f"{'=' * 96}")
    print("Same signals, grouped by the regime they fired in. If these rows look alike,")
    print("no gate built on these labels can help and step 2 is measuring noise.\n")
    print(report.by_regime.to_string(index=False))

    # -- question 2 ----------------------------------------------------------
    print(f"\n{'=' * 96}")
    print("STEP 2 - DOES A GATE DO WORK?")
    print(f"{'=' * 96}")
    print("separation = what the gate kept minus what it rejected. A gate that admits")
    print("fewer trades without separating them has done nothing but shrink the sample.\n")
    columns = ["gate", "n_kept", "share_kept", "kept_gross_mean", "rejected_gross_mean",
               "separation", "kept_net_mean", "kept_hit_rate", "beats_cost"]
    print(report.by_preset[[c for c in columns if c in report.by_preset.columns]].to_string(index=False))

    print(f"\nVERDICT: {report.verdict()}")

    if not args.backtest:
        print(f"\nArtefacts written to {store.root}")
        print("Re-run with --backtest to put the leading gate through the full backtester.")
        print("The untouched test set was NOT read.")
        return 0

    # -- optional: full backtest, gated vs ungated ---------------------------
    gated_presets = [p for p in PRESETS if p.name != "ungated"]
    best_name = None
    if not report.by_preset.empty:
        candidates = report.by_preset[report.by_preset["gate"] != "ungated"]
        if not candidates.empty:
            best_name = candidates.loc[candidates["kept_net_mean"].idxmax(), "gate"]
    best_gate = next((p for p in gated_presets if p.name == best_name), gated_presets[0])

    # The leading entry family from the family comparison, held for the horizon
    # the gate was scored at.
    family = train["event_spec"].value_counts().index[0]
    policy = ExitPolicy(take_profit_pct=None, stop_loss_pct=None, atr_stop_multiple=2.0,
                        time_stop_minutes=horizon_minutes, name=f"atr2_t{args.horizon_hours:.0f}h")
    sizing = SizingConfig.from_config(config.risk)
    limits = RiskLimits.from_config(config.risk)
    starting_equity = float(config.get("risk", "portfolio", "starting_equity_eur", default=10000.0))

    rows = []
    for label, subset in (
        ("ungated", train[train["event_spec"] == family]),
        (best_gate.name, apply_gate(train[train["event_spec"] == family], regimes, best_gate)),
    ):
        if subset.empty:
            print(f"\n{label}: the gate admitted no signals for {family}.")
            continue
        engine = Backtester(
            ExecutionModel(scenarios[headline], seed=seed), policy, sizing, limits,
            starting_equity=starting_equity, interval=args.setup_tf,
            max_holding_minutes=horizon_minutes,
            circuit_breakers=config.get("risk", "circuit_breakers", default={}),
        )
        result = engine.run(subset, features_by_market, ImmediateEntry(), seed=seed)
        if result is None or result.trades.empty:
            continue
        row = M.compute_trade_metrics(result.trades, starting_equity)
        row.update({"strategy": family, "gate": label, "exit_policy": policy.name,
                    "n_signals_in": len(subset), "split": "train"})
        closed = result.closed_trades()
        if not closed.empty:
            row.update(independent_trade_count(closed))
        rows.append(row)

    if rows:
        table = pd.DataFrame(rows)
        store.write_frame("regime_gate_backtest.csv", table)
        print(f"\n{'=' * 96}")
        print(f"BACKTEST - {family} - gated vs ungated (train split)")
        print(f"{'=' * 96}")
        columns = ["gate", "n_signals_in", "n_trades", "n_clusters", "win_rate",
                   "net_expectancy", "profit_factor", "max_drawdown"]
        print(table[[c for c in columns if c in table.columns]].to_string(index=False))
        print("\nA gate that improves expectancy while cutting the trade count has bought")
        print("that improvement with sample size. Judge it on n_clusters, not on the mean.")

    print(f"\nArtefacts written to {store.root}")
    print("The untouched test set was NOT read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
