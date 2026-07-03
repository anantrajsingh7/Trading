"""
Daily LIVE dashboard — runs the scanner on real market data, builds an
HTML dashboard from the ACTUAL setups found today, saves it, and opens it
in your browser.

This is different from reporting/generate_dashboard.py (which uses static
placeholder setups). This one uses live scanner output.

Usage:
    python daily_dashboard.py           # US market
    python daily_dashboard.py india     # India market

Schedule it with Windows Task Scheduler to run every weekday morning.
"""
from __future__ import annotations

import sys
import webbrowser
from datetime import date, datetime
from pathlib import Path

from core.logging import setup_logging, get_logger
from scanner.engine import Scanner

setup_logging("WARNING")
log = get_logger("daily_dashboard")

_ROOT = Path(__file__).resolve().parent

MARKET = "INDIA" if (len(sys.argv) > 1 and sys.argv[1].lower() == "india") else "US"
CUR = "₹" if MARKET == "INDIA" else "€"
ACCOUNT = 500_000 if MARKET == "INDIA" else 25_000


def _regime_color(regime: str) -> str:
    return {
        "BULL_STRONG": "#3fb950", "BULL_MODERATE": "#58a6ff",
        "NEUTRAL": "#e3b341", "BEAR_MODERATE": "#ffa657", "BEAR_STRONG": "#f85149",
    }.get(regime, "#8b949e")


def build():
    today = date.today()

    # ── Run the live scan ──────────────────────────────────────────
    scanner = Scanner(market=MARKET)
    scanner._min_score = 0.0          # show full ranked board
    opps = scanner.run(save_to_db=False)

    tradeable = [o for o in opps if o.score >= 85]

    # Regime is derived inside the scan; re-read from the top opportunity's
    # context if present, else default. We re-classify cheaply from SPY here.
    regime = "NEUTRAL"
    try:
        from data.pipeline import get_data
        from indicators.composite import enrich
        from scanner.market_regime import classify_regime
        proxy = "^NSEI" if MARKET == "INDIA" else "SPY"
        pdf = get_data(proxy, period="1y")
        if pdf is not None:
            regime = classify_regime(enrich(pdf, patterns=False)).regime.value
    except Exception as e:
        log.warning("regime lookup failed: %s", e)

    scale_pct = {
        "BULL_STRONG": 100, "BULL_MODERATE": 75, "NEUTRAL": 50,
        "BEAR_MODERATE": 25, "BEAR_STRONG": 0,
    }.get(regime, 50)
    rcolor = _regime_color(regime)

    # ── Build setup rows from LIVE data ────────────────────────────
    rows = ""
    for o in opps[:20]:
        tk = o.ticker.replace(".NS", "")
        sc = "#3fb950" if o.score >= 85 else "#58a6ff" if o.score >= 70 else "#8b949e"
        rows += f"""
        <tr>
          <td><strong style="color:#f0f6fc;">{tk}</strong></td>
          <td style="color:#58a6ff;">{o.strategy}</td>
          <td><span style="color:{sc};font-weight:700;">{o.score:.0f}/100</span></td>
          <td>{CUR}{o.ideal_entry:,.2f}</td>
          <td style="color:#f85149;">{CUR}{o.stop_loss:,.2f}</td>
          <td style="color:#3fb950;">{CUR}{o.target_1:,.2f}</td>
          <td style="color:#3fb950;">{CUR}{o.target_2:,.2f}</td>
          <td style="color:#3fb950;">{o.rr_ratio_t2:.1f}:1</td>
          <td>{o.position_size_shares}</td>
          <td style="color:#8b949e;">{o.rs_rank:.0f}</td>
        </tr>"""

    if not rows:
        rows = ('<tr><td colspan="10" style="text-align:center;color:#8b949e;'
                'padding:24px;">No signals generated today — no setups match '
                'any strategy. Stay in cash.</td></tr>')

    banner = (
        f"{len(tradeable)} tradeable setup(s) (score ≥ 85)"
        if tradeable else
        "No tradeable setups today (nothing scored ≥ 85) — patience, stay in cash."
    )
    banner_color = "#3fb950" if tradeable else "#e3b341"

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{MARKET} Live Dashboard | {today}</title>
<style>
  :root{{--bg:#0d1117;--card:#161b22;--border:#21262d;--green:#3fb950;--red:#f85149;
    --yellow:#e3b341;--blue:#58a6ff;--text:#c9d1d9;--muted:#8b949e;--white:#f0f6fc;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:-apple-system,Segoe UI,monospace;font-size:14px;}}
  .header{{background:linear-gradient(135deg,#1a1f2e,#0d1117);border-bottom:1px solid var(--border);padding:20px 32px;}}
  .header h1{{font-size:22px;color:var(--white);}}
  .badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:700;border:1px solid;}}
  .container{{max-width:1400px;margin:0 auto;padding:24px 32px;}}
  .section-title{{font-size:13px;font-weight:700;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid var(--border);}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:24px;}}
  table{{width:100%;border-collapse:collapse;font-size:13px;}}
  th{{text-align:left;padding:8px 12px;color:var(--muted);font-size:11px;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid var(--border);}}
  td{{padding:10px 12px;border-bottom:1px solid rgba(33,38,45,0.5);}}
  .alert{{background:rgba(227,179,65,0.1);border:1px solid {banner_color};border-radius:6px;padding:12px 16px;margin-bottom:20px;color:{banner_color};font-weight:700;}}
</style></head><body>
<div class="header">
  <h1>📊 {MARKET} Live Trading Dashboard</h1>
  <div style="color:var(--muted);margin-top:6px;font-size:13px;">
    {today.strftime('%A, %B %d, %Y')} at {datetime.now().strftime('%H:%M')} &nbsp;|&nbsp;
    <span class="badge" style="color:{rcolor};border-color:{rcolor};">{regime.replace('_',' ')}</span>
    &nbsp;Position scale: {scale_pct}% &nbsp;|&nbsp; Account: {CUR}{ACCOUNT:,} &nbsp;|&nbsp;
    <span style="color:var(--green);">LIVE DATA</span>
  </div>
</div>
<div class="container">
  <div class="alert">⚡ {banner}</div>
  <div class="section-title">Today's Ranked Setups (Live Scan)</div>
  <div class="card">
    <table>
      <thead><tr>
        <th>Ticker</th><th>Strategy</th><th>Score</th><th>Entry</th><th>Stop</th>
        <th>T1</th><th>T2</th><th>R:R T2</th><th>Qty</th><th>RS</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div style="color:var(--muted);font-size:11px;margin-top:12px;">
      * Generated from live market data. Only setups scoring ≥85 are tradeable.
        Qty sized to 1% account risk × {scale_pct}% regime scale.
    </div>
  </div>
  <div style="text-align:center;color:var(--muted);font-size:11px;margin-top:24px;padding-top:16px;border-top:1px solid var(--border);">
    Quant Platform v1.0 — research only, not financial advice — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </div>
</div></body></html>"""

    reports = _ROOT / "reports"
    reports.mkdir(exist_ok=True)
    prefix = "india_" if MARKET == "INDIA" else ""
    out = reports / f"{prefix}live_dashboard_{today.isoformat()}.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard: {out}")
    print(f"  {len(opps)} signals, {len(tradeable)} tradeable (score>=85), regime={regime}")

    # Open in default browser
    webbrowser.open(out.resolve().as_uri())
    return out


if __name__ == "__main__":
    build()
