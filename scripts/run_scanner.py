#!/usr/bin/env python3
"""Phase 11: live momentum scanner in PAPER mode.

Requires an approved strategy card at ``data/results/approved_strategy.json``.
The card is written by a human decision after the research pipeline has produced
out-of-sample evidence - the scanner will not invent parameters, because a live
scanner running untested rules is exactly how research discipline is lost.

    python scripts/run_scanner.py --once          # single pass, then exit
    python scripts/run_scanner.py                 # continuous polling
    python scripts/run_scanner.py --dry-run-card  # write a template card and stop

Live order execution is unavailable: this repository contains no order-placement
code. See `paper_trader.assert_live_trading_allowed`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from bitvavo_momentum.alerts import AlertDispatcher, should_alert  # noqa: E402
from bitvavo_momentum.bitvavo_client import BitvavoClient, BitvavoError  # noqa: E402
from bitvavo_momentum.config import Config, Credentials, load_dotenv_if_present  # noqa: E402
from bitvavo_momentum.data_downloader import candles_to_frame  # noqa: E402
from bitvavo_momentum.execution_model import ExecutionModel, ExecutionScenario  # noqa: E402
from bitvavo_momentum.logging_utils import setup_logging  # noqa: E402
from bitvavo_momentum.market_universe import fetch_market_rules, live_universe  # noqa: E402
from bitvavo_momentum.paper_trader import PaperTrader  # noqa: E402
from bitvavo_momentum.risk_manager import RiskLimits, RiskManager, SizingConfig  # noqa: E402
from bitvavo_momentum.scanner import (  # noqa: E402
    MomentumScanner,
    StrategyCard,
    StrategyCardMissing,
    signals_to_frame,
)
from bitvavo_momentum.storage import ResultStore  # noqa: E402
from bitvavo_momentum.timeutils import format_display, now_utc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="one scan pass, then exit")
    parser.add_argument("--max-markets", type=int, default=60)
    parser.add_argument("--lookback-minutes", type=int, default=None)
    parser.add_argument("--dry-run-card", action="store_true",
                        help="write a template strategy card (all zeros) and exit")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def write_template_card(path: Path) -> None:
    card = StrategyCard(
        strategy="TEMPLATE - replace after out-of-sample validation",
        event_lookback_minutes=120,
        event_threshold=0.10,
        exit_policy={"take_profit_pct": 0.03, "stop_loss_pct": 0.02, "time_stop_minutes": 240},
        filters={},
        allowed_regimes=None,
        historical_sample=0,
        historical_win_rate=float("nan"),
        historical_net_expectancy=float("nan"),
        historical_max_drawdown=float("nan"),
        validated_on="NOT VALIDATED",
        approved_utc=now_utc().isoformat(),
        notes=(
            "TEMPLATE ONLY. historical_sample=0 means every signal will be flagged "
            "insufficient_evidence. Replace this file with a card produced from a real "
            "out-of-sample evaluation before treating any output as meaningful."
        ),
    )
    card.save(path)
    print(f"Wrote template card to {path}")
    print("It is deliberately marked NOT VALIDATED - signals from it carry an evidence warning.")


def main() -> int:
    args = parse_args()
    load_dotenv_if_present()
    log = setup_logging(args.log_level, "logs/scanner.log")
    config = Config.load()
    config.ensure_dirs()

    scanner_cfg = config.get("paper", "scanner", default={})
    card_path = Path(config.root) / str(scanner_cfg.get("approved_strategy_card", "data/results/approved_strategy.json"))

    if args.dry_run_card:
        write_template_card(card_path)
        return 0

    try:
        card = StrategyCard.load(card_path)
    except StrategyCardMissing as exc:
        log.error("%s", exc)
        print("\nRun `python scripts/run_scanner.py --dry-run-card` to see the expected format.")
        return 2

    if card.historical_sample <= 0:
        log.warning(
            "Strategy card reports zero historical sample. Every signal will be flagged "
            "insufficient_evidence, which is correct - do not act on them."
        )

    mode = config.get("paper", "mode", default="paper")
    if config.live_trading_allowed(cli_ack=False):
        log.warning("Live-trading switches are set, but this scanner only ever paper-trades.")
    log.info("Scanner starting in %s mode using card: %s", mode, card.strategy)

    scenario = ExecutionScenario.from_config(
        config.get("paper", "paper", "execution_scenario", default="realistic"), config.risk
    )
    sizing = SizingConfig.from_config(config.risk)
    risk = RiskManager(
        RiskLimits.from_config(config.risk),
        float(config.get("risk", "portfolio", "starting_equity_eur", default=10000.0)),
        config.get("risk", "circuit_breakers", default={}),
    )
    execution = ExecutionModel(scenario)
    trader = PaperTrader(
        execution, risk,
        state_dir=Path(config.root) / str(config.get("paper", "paper", "state_dir", default="data/results/paper")),
        record_every_signal=bool(config.get("paper", "paper", "record_every_signal", default=True)),
    )
    dispatcher = AlertDispatcher(config.paper)
    results = ResultStore(config.path("results_dir") / "scanner")

    credentials = Credentials.from_env()
    lookback = args.lookback_minutes or (card.event_lookback_minutes + 240)
    poll_seconds = float(scanner_cfg.get("poll_seconds", 30))

    with BitvavoClient(credentials) as client:
        try:
            rules = fetch_market_rules(client, scanner_cfg.get("quote_currency", "EUR"))
        except BitvavoError as exc:
            log.error("Cannot reach Bitvavo: %s", exc)
            return 2

        markets = live_universe(rules, scanner_cfg.get("exclude_bases", []))[: args.max_markets]
        log.info("Monitoring %d markets", len(markets))
        scanner = MomentumScanner(card, scanner_cfg, config.get("paper", "signal_quality", default={}),
                                  scenario, sizing, risk, rules)

        while True:
            pass_start = now_utc()
            signals = []
            prices: dict[str, float] = {}
            errors = 0

            for market in markets:
                try:
                    rows = client.get_candles(market, "1m", limit=min(1440, lookback + 60))
                    candles = candles_to_frame(rows)
                    if candles.empty:
                        continue
                    prices[market] = float(candles["close"].iloc[-1])
                    book = None
                    signal = scanner.evaluate_market(
                        market, candles, book=book, ticker_24h=None,
                        btc_regime="unknown", market_regime="unknown", as_of=pass_start,
                    )
                    if signal is None:
                        continue
                    signals.append(signal)
                    print("\n" + signal.to_text())

                    gate = should_alert(signal, config.paper, risk.snapshot(), data_is_current=True)
                    if gate.allowed:
                        dispatcher.send_signal(signal)
                        trader.open_from_signal(signal)
                    else:
                        trader.record_signal(signal, "gated", "; ".join(gate.reasons))
                        dispatcher.send_informational(market, "; ".join(gate.reasons))
                    risk.report_api_success()
                except BitvavoError as exc:
                    errors += 1
                    risk.report_api_error(pass_start)
                    log.warning("%s: %s", market, exc)

            closed = trader.update(prices, pass_start)
            trader.save_state()

            if signals:
                results.write_frame(
                    f"signals_{pass_start.strftime('%Y%m%d')}.csv", signals_to_frame(signals)
                )
            snapshot = trader.portfolio_snapshot(prices)
            log.info(
                "Pass at %s: %d signals, %d closed, equity %.2f EUR, %d API errors",
                format_display(pass_start), len(signals), len(closed), snapshot["equity"], errors,
            )

            if args.once:
                print("\nPortfolio snapshot:")
                for key, value in snapshot.items():
                    print(f"  {key}: {value}")
                return 0
            time.sleep(poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nScanner stopped.")
        raise SystemExit(0) from None
