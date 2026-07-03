"""
Backtest runner — the honest test of whether a strategy actually makes money.

For each strategy, runs a 5-year backtest across a basket of liquid tickers,
then Monte Carlo (robustness to trade ordering) and walk-forward (out-of-sample
survival). Prints the metrics that matter — not just return, but Sharpe,
drawdown, expectancy, and OOS consistency.

Usage:
    python backtest_runner.py                 # all strategies, default basket
    python backtest_runner.py minervini_vcp   # one strategy
"""
from __future__ import annotations

import sys

from core.config import get_config
from core.logging import setup_logging
from data.pipeline import get_data_batch
from indicators.composite import enrich
from strategies.registry import all_strategies, get_strategy
from backtester.engine import Backtester
from backtester.metrics import print_metrics
from backtester.monte_carlo import run_monte_carlo
from backtester.walk_forward import walk_forward

setup_logging("WARNING")

# Liquid, survivorship-light basket (large caps that existed 5y ago)
BASKET = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AMD", "AVGO",
    "JPM", "V", "MA", "COST", "HD", "NFLX", "CRM", "ADBE", "QCOM",
    "TXN", "AMAT", "LRCX", "MU", "NOW", "PANW", "ORCL", "CSCO",
]

PERIOD = "5y"


def run_one(strategy, data, cfg, exit_mode="targets", max_hold_days=None):
    hold = f", timestop={max_hold_days}d" if max_hold_days else ""
    label = f"{strategy.name.upper()}  [exit={exit_mode}{hold}]"
    print(f"\n{'#'*60}\n#  {label}  —  {strategy.description}\n{'#'*60}")

    bt = Backtester(strategy, cfg, exit_mode=exit_mode, max_hold_days=max_hold_days)
    result = bt.run(data)

    if not result.trades:
        print("  No trades generated over the period. Cannot evaluate.")
        return

    print_metrics(result.metrics)

    # Monte Carlo — is the result robust to the order trades happened in?
    mc = run_monte_carlo(strategy.name, result.trades, n_simulations=1000)
    if mc:
        print(f"  Monte Carlo (1000 sims):")
        print(f"    CAGR   5th/50th/95th : {mc['cagr_5th']*100:6.1f}% / "
              f"{mc['cagr_50th']*100:6.1f}% / {mc['cagr_95th']*100:6.1f}%")
        print(f"    MaxDD  worst (5th)   : {mc['max_dd_5th']*100:6.1f}%")
        print(f"    Ruin probability     : {mc['ruin_probability']*100:6.1f}%")

    # Walk-forward — does it survive on data it never 'saw'?
    wf = walk_forward(strategy, data, cfg, in_sample_years=2, oos_months=6)
    if wf.get("n_windows"):
        print(f"  Walk-forward: {wf['n_profitable_windows']}/{wf['n_windows']} "
              f"OOS windows profitable ({wf['pct_profitable']*100:.0f}%), "
              f"{wf['total_oos_trades']} OOS trades")

    # ── Verdict ────────────────────────────────────────────────────
    m = result.metrics
    verdict = []
    verdict.append("PASS" if m.sharpe_ratio >= 1.0 else "WEAK")
    verdict.append("PASS" if m.profit_factor >= 1.3 else "WEAK")
    verdict.append("PASS" if m.max_drawdown > -0.30 else "RISKY")
    verdict.append("PASS" if (mc and mc['ruin_probability'] < 0.05) else "RISKY")
    print(f"  VERDICT  Sharpe:{verdict[0]}  ProfitFactor:{verdict[1]}  "
          f"Drawdown:{verdict[2]}  Ruin:{verdict[3]}")


def main():
    cfg = get_config()
    print(f"Fetching {len(BASKET)} tickers, {PERIOD} history...")
    raw = get_data_batch(BASKET, period=PERIOD, show_progress=True)
    print(f"Enriching {len(raw)} tickers with indicators...")
    data = {t: enrich(df, patterns=True) for t, df in raw.items()}

    if len(sys.argv) > 1:
        s = get_strategy(sys.argv[1])
        # Head-to-head: all exit styles, plus swing-length time stop
        run_one(s, data, cfg, exit_mode="targets")
        run_one(s, data, cfg, exit_mode="trailing")
        run_one(s, data, cfg, exit_mode="hybrid")
        run_one(s, data, cfg, exit_mode="hybrid", max_hold_days=21)
    else:
        for s in all_strategies():
            run_one(s, data, cfg, exit_mode="targets")

    print("\n" + "="*60)
    print("Read the VERDICT lines. A strategy worth trading should show")
    print("Sharpe >= 1.0, ProfitFactor >= 1.3, drawdown better than -30%,")
    print("ruin < 5%, AND a majority of walk-forward windows profitable.")
    print("If it only wins in-sample but fails walk-forward, it's overfit.")
    print("="*60)


if __name__ == "__main__":
    main()
