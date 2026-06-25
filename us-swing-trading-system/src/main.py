"""
Entry point for the US Swing Trading System.

Commands:
  python src/main.py backtest   – backtest all strategies
  python src/main.py scan       – run daily scanner
  python src/main.py report     – generate full report from saved results
  python src/main.py all        – run everything
"""
import argparse
import logging
import sys
from pathlib import Path

# allow "import utils" style from within src/
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config, setup_logging, ensure_dirs, get_root
from data_loader import load_all, get_universe, load_ticker_data
from indicators import compute_all
from backtester import run_all_backtests, BacktestResult
from scanner import scan_universe, market_regime
from reporting import (
    save_backtest_results,
    save_strategy_ranking,
    save_equity_curve,
    save_strategy_comparison,
    save_daily_watchlist,
    save_top_candidates,
)


def _load_benchmarks(cfg: dict) -> tuple:
    """Load SPY and QQQ with indicators, return (spy_df, qqq_df)."""
    log = logging.getLogger("swing")
    spy_raw = load_ticker_data("SPY", cfg)
    qqq_raw = load_ticker_data("QQQ", cfg)
    spy_df = compute_all(spy_raw, cfg=cfg) if spy_raw is not None else None
    qqq_df = compute_all(qqq_raw, cfg=cfg) if qqq_raw is not None else None
    if spy_df is None:
        log.warning("Could not load SPY benchmark")
    if qqq_df is None:
        log.warning("Could not load QQQ benchmark")
    return spy_df, qqq_df


def _prepare_data(cfg: dict, limit: int | None = None) -> dict:
    """Download/load universe, compute indicators, return enriched dict."""
    log = logging.getLogger("swing")
    tickers = get_universe(cfg)
    if limit:
        tickers = tickers[:limit]
    log.info("Universe: %d tickers", len(tickers))

    spy_df, qqq_df = _load_benchmarks(cfg)
    spy_close = spy_df["close"] if spy_df is not None else None
    qqq_close = qqq_df["close"] if qqq_df is not None else None

    raw_data = load_all(tickers, cfg, cache=True)
    enriched: dict = {}
    for ticker, df in raw_data.items():
        try:
            enriched[ticker] = compute_all(df, spy=spy_close, qqq=qqq_close, cfg=cfg)
        except Exception as e:
            log.debug("Indicator error %s: %s", ticker, e)

    log.info("Enriched %d tickers with indicators", len(enriched))
    return enriched, spy_df, qqq_df


def cmd_backtest(cfg: dict, args) -> dict[str, BacktestResult]:
    log = logging.getLogger("swing")
    log.info("=== BACKTEST MODE ===")
    limit = getattr(args, "limit", None)
    data, spy_df, qqq_df = _prepare_data(cfg, limit=limit)
    results = run_all_backtests(data, cfg)
    save_backtest_results(results)
    _, best = save_strategy_ranking(results)
    save_equity_curve(results)
    save_strategy_comparison(results)
    log.info("Backtest complete. Best strategy: %s", best)
    return results


def cmd_scan(cfg: dict, args,
             results: dict[str, BacktestResult] | None = None) -> None:
    log = logging.getLogger("swing")
    log.info("=== DAILY SCANNER ===")
    limit = getattr(args, "limit", None)
    data, spy_df, qqq_df = _prepare_data(cfg, limit=limit)

    # pick best strategy from saved ranking if available
    best_strategy = None
    ranking_file = get_root() / "reports" / "strategy_ranking.csv"
    if results:
        import pandas as pd
        from backtester import BacktestResult as BR
        rows = [r.to_dict() for r in results.values()]
        import pandas as pd
        df_rank = pd.DataFrame(rows)
    elif ranking_file.exists():
        import pandas as pd
        df_rank = pd.read_csv(ranking_file)
        if len(df_rank) > 0 and "strategy" in df_rank.columns:
            pass  # use all strategies, scanner will rank by score

    regime = market_regime(spy_df, qqq_df)
    log.info("Market regime: SPY=%s | QQQ=%s | Mode=%s",
             regime["spy_trend"], regime["qqq_trend"], regime["risk_mode"])

    if regime["risk_mode"] == "no-trade":
        log.warning("Market in downtrend – no-trade mode. Scanner still runs for research.")

    candidates = scan_universe(data, cfg)
    save_daily_watchlist(candidates)

    if results is None and ranking_file.exists():
        results = {}  # empty – report will skip best-strategy section

    save_top_candidates(candidates, results or {}, regime, cfg)

    if candidates:
        log.info("Top candidate: %s (score %.1f/10)",
                 candidates[0].ticker, candidates[0].setup_score)
    else:
        log.info("No candidates found today.")


def cmd_report(cfg: dict, args) -> None:
    """Re-generate reports from saved data (no new downloads)."""
    log = logging.getLogger("swing")
    log.info("=== REPORT MODE ===")
    # just re-scan with cached data
    cmd_scan(cfg, args)


def cmd_all(cfg: dict, args) -> None:
    log = logging.getLogger("swing")
    log.info("=== RUNNING EVERYTHING ===")
    results = cmd_backtest(cfg, args)
    cmd_scan(cfg, args, results=results)
    log.info("All done. Reports saved to reports/")


def main():
    parser = argparse.ArgumentParser(
        description="US Swing Trading Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  backtest   Backtest all 5 strategies and save results
  scan       Run daily scanner and produce watchlist
  report     Re-generate reports from cached data
  all        Run backtest + scan + full report
        """,
    )
    parser.add_argument("command", choices=["backtest", "scan", "report", "all"])
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: config.yaml in project root)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tickers (useful for testing)")
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file"),
    )

    dispatch = {
        "backtest": cmd_backtest,
        "scan": cmd_scan,
        "report": cmd_report,
        "all": cmd_all,
    }
    dispatch[args.command](cfg, args)


if __name__ == "__main__":
    main()
