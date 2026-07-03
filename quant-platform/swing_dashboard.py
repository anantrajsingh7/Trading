"""
Swing Dashboard — combined strategies + per-stock backtest validation.

For a 1-2 week swing trader. Each morning it:
  1. Scans the universe with the strategies that survived backtesting
  2. Combines signals — a stock flagged by MORE strategies ranks higher (confluence)
  3. Backtests the triggering strategy ON THAT SPECIFIC STOCK's 5-year history
  4. ONLY shows stocks where that setup has actually worked on that stock
     (historical win rate >= 50% and profit factor >= 1.2)
  5. Shows live R:R alongside the historical win rate / expectancy

This is the honest version: it doesn't just say "here's a setup" — it says
"here's a setup, and here's how this exact setup has performed on this exact
stock in the past."

Usage:
    python swing_dashboard.py            # US market
    python swing_dashboard.py india      # India market
"""
from __future__ import annotations

import sys
import webbrowser
from datetime import date, datetime
from pathlib import Path

from core.config import get_config
from core.logging import setup_logging, get_logger
from data.pipeline import get_data, get_data_batch
from data.universe import get_universe
from indicators.composite import enrich
from strategies.registry import get_strategy
from scanner.market_regime import classify_regime
from backtester.engine import Backtester

setup_logging("WARNING")
log = get_logger("swing_dashboard")
_ROOT = Path(__file__).resolve().parent

MARKET = "INDIA" if (len(sys.argv) > 1 and sys.argv[1].lower() == "india") else "US"
CUR = "₹" if MARKET == "INDIA" else "€"
ACCOUNT = 500_000 if MARKET == "INDIA" else 25_000

# Only the strategies that passed walk-forward / made money in backtest.
# mean_reversion (loses money) and momentum (fires nothing) are excluded.
GOOD_STRATEGIES = ["ema_pullback", "minervini_vcp", "volume_breakout"]

# A stock only makes the board if its per-stock backtest clears these:
MIN_HIST_TRADES   = 5
MIN_HIST_WINRATE  = 0.50
MIN_HIST_PF       = 1.2

# Trailing exit let winners run — proven better for swing holds
EXIT_MODE = "trailing"

# Swing target: a stock the trader wants ~10% on
SWING_TARGET_PCT = 0.10


def _regime_color(r):
    return {"BULL_STRONG": "#3fb950", "BULL_MODERATE": "#58a6ff", "NEUTRAL": "#e3b341",
            "BEAR_MODERATE": "#ffa657", "BEAR_STRONG": "#f85149"}.get(r, "#8b949e")


def build():
    cfg = get_config()
    today = date.today()

    # ── Market regime ──────────────────────────────────────────────
    regime = "NEUTRAL"
    proxy = "^NSEI" if MARKET == "INDIA" else "SPY"
    pdf = get_data(proxy, period="1y")
    if pdf is not None:
        regime = classify_regime(enrich(pdf, patterns=False)).regime.value
    scale_pct = {"BULL_STRONG": 100, "BULL_MODERATE": 75, "NEUTRAL": 50,
                 "BEAR_MODERATE": 25, "BEAR_STRONG": 0}.get(regime, 50)
    rcolor = _regime_color(regime)

    # ── Fetch + enrich universe ────────────────────────────────────
    universe = get_universe(cfg, MARKET)
    print(f"Scanning {len(universe)} tickers with per-stock backtest validation...")
    raw = get_data_batch(universe, period="5y", show_progress=True)
    enriched = {t: enrich(df, patterns=True) for t, df in raw.items()}

    strategies = [get_strategy(n) for n in GOOD_STRATEGIES]

    # ── Find today's signals + validate each on the stock's history ─
    # candidates[ticker] = {strategies:[...], signal: best_signal, scores:{}}
    candidates = {}
    for ticker, df in enriched.items():
        if len(df) < 252:
            continue
        for strat in strategies:
            try:
                sig = strat.generate_signal(df, ticker, cfg)
            except Exception:
                continue
            if sig is None or sig.stop_loss >= sig.entry_price:
                continue

            # Backtest THIS strategy on THIS stock's history
            bt = Backtester(strat, cfg, exit_mode=EXIT_MODE)
            res = bt.run({ticker: df})
            m = res.metrics
            if (m.n_trades < MIN_HIST_TRADES or m.win_rate < MIN_HIST_WINRATE
                    or m.profit_factor < MIN_HIST_PF):
                continue  # this setup hasn't worked on this stock — skip

            c = candidates.setdefault(ticker, {
                "strategies": [], "signal": sig, "best_score": 0,
                "hist_winrate": m.win_rate, "hist_pf": m.profit_factor,
                "hist_expectancy": m.expectancy, "hist_trades": m.n_trades,
            })
            c["strategies"].append(strat.name)
            if sig.score > c["best_score"]:
                c["best_score"] = sig.score
                c["signal"] = sig
                c["hist_winrate"] = m.win_rate
                c["hist_pf"] = m.profit_factor
                c["hist_expectancy"] = m.expectancy
                c["hist_trades"] = m.n_trades

    # ── Rank: confluence first, then historical edge ───────────────
    ranked = []
    for ticker, c in candidates.items():
        sig = c["signal"]
        risk_ps = sig.entry_price - sig.stop_loss
        # confluence bonus: +score for each extra strategy agreeing
        confluence = len(set(c["strategies"]))
        combined = c["best_score"] + (confluence - 1) * 10 + c["hist_pf"] * 5
        ranked.append({
            "ticker": ticker, "strategies": sorted(set(c["strategies"])),
            "confluence": confluence, "score": c["best_score"],
            "entry": sig.entry_price, "stop": sig.stop_loss,
            "t1": sig.target_1, "t2": sig.target_2,
            "swing_target": sig.entry_price * (1 + SWING_TARGET_PCT),
            "risk_ps": risk_ps,
            "rr_swing": (sig.entry_price * SWING_TARGET_PCT) / risk_ps if risk_ps > 0 else 0,
            "hist_winrate": c["hist_winrate"], "hist_pf": c["hist_pf"],
            "hist_expectancy": c["hist_expectancy"], "hist_trades": c["hist_trades"],
            "combined": combined,
        })
    ranked.sort(key=lambda x: x["combined"], reverse=True)

    # ── Position sizing (1% risk, capped at 10% of account) ────────
    risk_eur = ACCOUNT * 0.01 * (scale_pct / 100)
    for r in ranked:
        qty = max(1, int(risk_eur / r["risk_ps"])) if r["risk_ps"] > 0 else 1
        if qty * r["entry"] > ACCOUNT * 0.10:
            qty = max(1, int(ACCOUNT * 0.10 / r["entry"]))
        r["qty"] = qty

    # ── Build HTML ─────────────────────────────────────────────────
    rows = ""
    for r in ranked[:20]:
        tk = r["ticker"].replace(".NS", "")
        conf_badge = ("<span style='color:#3fb950;font-weight:700;'>●●●</span>" if r["confluence"] >= 3
                      else "<span style='color:#58a6ff;font-weight:700;'>●●</span>" if r["confluence"] == 2
                      else "<span style='color:#8b949e;'>●</span>")
        wr_color = "#3fb950" if r["hist_winrate"] >= 0.6 else "#58a6ff" if r["hist_winrate"] >= 0.5 else "#e3b341"
        strat_str = ", ".join(s.replace("_", " ") for s in r["strategies"])
        rows += f"""
        <tr>
          <td><strong style="color:#f0f6fc;">{tk}</strong></td>
          <td style="text-align:center;">{conf_badge}</td>
          <td style="color:#58a6ff;font-size:12px;">{strat_str}</td>
          <td>{CUR}{r['entry']:,.2f}</td>
          <td style="color:#f85149;">{CUR}{r['stop']:,.2f}</td>
          <td style="color:#3fb950;">{CUR}{r['swing_target']:,.2f}</td>
          <td style="color:#3fb950;font-weight:700;">{r['rr_swing']:.1f}:1</td>
          <td style="color:{wr_color};font-weight:700;">{r['hist_winrate']*100:.0f}%</td>
          <td style="color:#c9d1d9;">{r['hist_pf']:.2f}</td>
          <td style="color:#8b949e;">{r['hist_trades']}</td>
          <td>{r['qty']}</td>
        </tr>"""

    if not rows:
        rows = ('<tr><td colspan="11" style="text-align:center;color:#8b949e;padding:24px;">'
                'No validated setups today. No stock is both flagging a setup AND has a '
                'winning backtest history for that setup. Stay in cash.</td></tr>')

    n = len(ranked)
    banner = (f"{n} validated setup(s) — each has flagged today AND won historically on that stock"
              if n else "No validated setups today — patience, stay in cash.")
    bcolor = "#3fb950" if n else "#e3b341"

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Swing Dashboard | {today}</title>
<style>
  :root{{--bg:#0d1117;--card:#161b22;--border:#21262d;--green:#3fb950;--red:#f85149;
    --yellow:#e3b341;--blue:#58a6ff;--text:#c9d1d9;--muted:#8b949e;--white:#f0f6fc;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:-apple-system,Segoe UI,monospace;font-size:14px;}}
  .header{{background:linear-gradient(135deg,#1a1f2e,#0d1117);border-bottom:1px solid var(--border);padding:20px 32px;}}
  .header h1{{font-size:22px;color:var(--white);}}
  .badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:700;border:1px solid;}}
  .container{{max-width:1500px;margin:0 auto;padding:24px 32px;}}
  .section-title{{font-size:13px;font-weight:700;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid var(--border);}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:24px;}}
  table{{width:100%;border-collapse:collapse;font-size:13px;}}
  th{{text-align:left;padding:8px 10px;color:var(--muted);font-size:10px;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid var(--border);}}
  td{{padding:10px;border-bottom:1px solid rgba(33,38,45,0.5);}}
  .alert{{background:rgba(63,185,80,0.08);border:1px solid {bcolor};border-radius:6px;padding:12px 16px;margin-bottom:20px;color:{bcolor};font-weight:700;}}
  .legend{{color:var(--muted);font-size:11px;margin-top:12px;line-height:1.6;}}
</style></head><body>
<div class="header">
  <h1>📈 Swing Trader Dashboard — Validated Setups</h1>
  <div style="color:var(--muted);margin-top:6px;font-size:13px;">
    {today.strftime('%A, %B %d, %Y')} at {datetime.now().strftime('%H:%M')} &nbsp;|&nbsp;
    <span class="badge" style="color:{rcolor};border-color:{rcolor};">{regime.replace('_',' ')}</span>
    &nbsp;Scale {scale_pct}% &nbsp;|&nbsp; Account {CUR}{ACCOUNT:,} &nbsp;|&nbsp;
    Risk/trade {CUR}{risk_eur:,.0f} &nbsp;|&nbsp; <span style="color:var(--green);">LIVE + BACKTESTED</span>
  </div>
</div>
<div class="container">
  <div class="alert">⚡ {banner}</div>
  <div class="section-title">Today's Validated Swing Setups</div>
  <div class="card">
    <table>
      <thead><tr>
        <th>Stock</th><th>Conf.</th><th>Strategy</th><th>Entry</th><th>Stop</th>
        <th>Target +10%</th><th>R:R</th><th>Hist Win%</th><th>Hist PF</th><th>#Trades</th><th>Qty</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="legend">
      <strong>Conf.</strong> = confluence: ●●● three strategies agree, ●● two, ● one. More dots = higher conviction.<br>
      <strong>Hist Win%</strong> / <strong>Hist PF</strong> = how this exact strategy performed on THIS stock over 5 years (win rate & profit factor).<br>
      <strong>Only stocks with historical win rate ≥50% and profit factor ≥1.2 are shown.</strong>
      Target is +10% swing goal; R:R compares that to your stop risk. Qty sized to 1% account risk, capped at 10% position.
    </div>
  </div>
  <div style="text-align:center;color:var(--muted);font-size:11px;margin-top:24px;padding-top:16px;border-top:1px solid var(--border);">
    Quant Platform — research only, not financial advice — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </div>
</div></body></html>"""

    reports = _ROOT / "reports"
    reports.mkdir(exist_ok=True)
    prefix = "india_" if MARKET == "INDIA" else ""
    out = reports / f"{prefix}swing_dashboard_{today.isoformat()}.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nDashboard: {out}")
    print(f"  {len(ranked)} validated setups (flagged today AND winning backtest on that stock)")
    webbrowser.open(out.resolve().as_uri())
    return out


if __name__ == "__main__":
    build()
