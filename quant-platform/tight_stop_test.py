"""
Tight-stop experiment — tests the exact idea:
  "stop just below the buying price, take +5-10% profit in a few days"

Runs the ema_pullback entry signal (our proven best) through a grid of
fixed brackets: stop 1/2/3/5% below entry x target +5/7/10%, each with a
10-trading-day time limit, and compares against the current champion
(ATR trailing stop). Same stocks, same 5 years, same entries — only the
exit changes, so the comparison is clean.

Usage:
    python tight_stop_test.py
"""
from __future__ import annotations

from core.config import get_config
from core.logging import setup_logging
from data.pipeline import get_data_batch
from indicators.composite import enrich
from strategies.registry import get_strategy
from backtester.engine import Backtester

setup_logging("WARNING")

BASKET = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AMD", "AVGO",
    "JPM", "V", "MA", "COST", "HD", "NFLX", "CRM", "ADBE", "QCOM",
    "TXN", "AMAT", "LRCX", "MU", "NOW", "PANW", "ORCL", "CSCO",
]

STOPS   = [0.01, 0.02, 0.03, 0.05]      # stop 1% / 2% / 3% / 5% below entry
TARGETS = [0.05, 0.07, 0.10]            # take profit +5% / +7% / +10%
MAX_HOLD = 14                           # calendar days ≈ 10 trading days


def main():
    cfg = get_config()
    strat = get_strategy("ema_pullback")

    print(f"Fetching {len(BASKET)} tickers, 5y history...")
    raw = get_data_batch(BASKET, period="5y", show_progress=True)
    print("Enriching with indicators...")
    data = {t: enrich(df, patterns=True) for t, df in raw.items()}

    print(f"\n{'STOP':>6} {'TARGET':>8} {'TRADES':>7} {'WIN%':>6} "
          f"{'PF':>6} {'CAGR':>7} {'SHARPE':>7} {'MAXDD':>7}   VERDICT")
    print("-" * 78)

    results = []
    for sp in STOPS:
        for tp in TARGETS:
            bt = Backtester(strat, cfg, exit_mode="fixed",
                            stop_pct=sp, target_pct=tp, max_hold_days=MAX_HOLD)
            res = bt.run(data)
            m = res.metrics
            if not res.trades:
                print(f"{sp*100:5.0f}% {tp*100:7.0f}% {'—':>7}")
                continue
            ok = m.profit_factor >= 1.3 and m.sharpe_ratio > 0
            results.append((sp, tp, m))
            print(f"{sp*100:5.0f}% {tp*100:7.0f}% {m.n_trades:7d} "
                  f"{m.win_rate*100:5.1f}% {m.profit_factor:6.2f} "
                  f"{m.cagr*100:6.1f}% {m.sharpe_ratio:7.2f} "
                  f"{m.max_drawdown*100:6.1f}%   {'VIABLE' if ok else 'weak'}")

    # Champion baseline for comparison
    bt = Backtester(strat, cfg, exit_mode="trailing")
    res = bt.run(data)
    m = res.metrics
    print("-" * 78)
    print(f"{'ATR':>6} {'trail':>8} {m.n_trades:7d} "
          f"{m.win_rate*100:5.1f}% {m.profit_factor:6.2f} "
          f"{m.cagr*100:6.1f}% {m.sharpe_ratio:7.2f} "
          f"{m.max_drawdown*100:6.1f}%   ← current champion")

    print("""
How to read this:
  WIN%  — how often the bracket hit the target before the stop
  PF    — profit factor: total wins / total losses (>1.3 = healthy)
  A tight stop (1-2%) needs a high win rate to survive, because each
  win pays for several stop-outs only if wins actually happen often.
  Whichever row beats the champion on PF *and* Sharpe is a real
  candidate — if none does, the tight-stop idea loses to trailing.
""")


if __name__ == "__main__":
    main()
