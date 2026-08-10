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

import os
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
# Prices are quoted in the LOCAL market currency (USD for US stocks);
# the account and risk budget are in the account currency (EUR).
CUR = "₹" if MARKET == "INDIA" else "$"          # price/quote currency
ACCT_CUR = "₹" if MARKET == "INDIA" else "€"     # account currency
ACCOUNT = 500_000 if MARKET == "INDIA" else 30_000
MAX_RISK = 5_000 if MARKET == "INDIA" else 300   # hard cap per trade (account ccy)

# Only the strategies that passed walk-forward / made money in backtest.
# mean_reversion (loses money) and momentum (fires nothing) are excluded.
GOOD_STRATEGIES = ["ema_pullback", "minervini_vcp", "volume_breakout"]

# A stock only makes the board if its per-stock backtest clears these:
MIN_HIST_TRADES   = 5
MIN_HIST_WINRATE  = 0.50
MIN_HIST_PF       = 1.2

# Exit config — decided by A/B/C/D backtest on 2026-07-03:
#   targets 6.4% CAGR / trailing 10.1% / hybrid 7.5% / hybrid+21d 6.8%
# Pure trailing won decisively. Partial profit-taking and time stops
# both clipped the tail winners that carry the whole edge.
EXIT_MODE = "trailing"
MAX_HOLD_DAYS = None

# Minervini RS gate: only trade stocks outperforming >=70% of universe
RS_MIN = 70

# Never hold a 1-2 week swing into earnings
EARNINGS_BLACKOUT_DAYS = 14

# Portfolio heat: never more than this many concurrent open positions
MAX_OPEN = 5

# Swing target: a stock the trader wants ~10% on
SWING_TARGET_PCT = 0.10

# Board always shows at least this many names (validated first, then
# best near-misses clearly labelled WATCH with the reason they fell short)
BOARD_MIN = 5


def _regime_color(r):
    return {"BULL_STRONG": "#3fb950", "BULL_MODERATE": "#58a6ff", "NEUTRAL": "#e3b341",
            "BEAR_MODERATE": "#ffa657", "BEAR_STRONG": "#f85149"}.get(r, "#8b949e")


def _extension(df) -> tuple[float, float]:
    """(% below 52-week high, % above EMA20) — the 'am I chasing?' check.

    The 52-week high is the highest INTRADAY high, matching the convention
    every quote site uses. Using closing prices understates the distance
    from the high and made the board disagree with the user's broker.
    """
    c = df["close"]
    highs = df["high"] if "high" in df else c
    hi52 = float(highs.tail(252).max())
    last = float(c.iloc[-1])
    ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
    from_high = (hi52 - last) / hi52 * 100 if hi52 > 0 else 0.0
    above_ema = (last - ema20) / ema20 * 100 if ema20 > 0 else 0.0
    return from_high, above_ema


def _eurusd() -> float:
    """USD per 1 EUR. US stocks quote in USD; the account is in EUR, so
    the risk budget must be converted before sizing. Falls back to 1.08."""
    try:
        fx = get_data("EURUSD=X", period="1mo")
        if fx is not None and len(fx):
            rate = float(fx["close"].iloc[-1])
            if 0.5 < rate < 2.0:
                return rate
    except Exception as e:
        log.warning("EURUSD fetch failed (%s) — using 1.08", e)
    return 1.08


def _rs_ranks(enriched: dict) -> dict:
    """RS score = 0.4*ROC3M + 0.2*ROC6M + 0.2*ROC9M + 0.2*ROC12M,
    expressed as percentile rank vs the scanned universe."""
    scores = {}
    for t, df in enriched.items():
        c = df["close"]
        if len(c) < 252:
            continue
        try:
            scores[t] = (0.4 * (c.iloc[-1] / c.iloc[-63] - 1)
                         + 0.2 * (c.iloc[-1] / c.iloc[-126] - 1)
                         + 0.2 * (c.iloc[-1] / c.iloc[-189] - 1)
                         + 0.2 * (c.iloc[-1] / c.iloc[-252] - 1))
        except Exception:
            pass
    vals = list(scores.values())
    return {t: sum(1 for v in vals if v <= s) / len(vals) * 100
            for t, s in scores.items()} if vals else {}


def _days_to_earnings(ticker: str):
    """Days until next earnings, or None if unknown. Best-effort."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            nxt = min(d for d in dates if d >= date.today())
            return (nxt - date.today()).days
    except Exception:
        pass
    return None


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

    FX = _eurusd()

    # ── Fetch + enrich universe ────────────────────────────────────
    universe = get_universe(cfg, MARKET)
    print(f"Scanning {len(universe)} tickers with per-stock backtest validation...")
    raw = get_data_batch(universe, period="5y", show_progress=True)

    from tqdm import tqdm
    enriched = {}
    for t, df in tqdm(raw.items(), desc="Computing indicators", unit="stock"):
        enriched[t] = enrich(df, patterns=True)

    strategies = [get_strategy(n) for n in GOOD_STRATEGIES]
    rs_ranks = _rs_ranks(enriched)

    # ── Find today's signals + validate each on the stock's history ─
    # candidates[ticker] = {strategies:[...], signal: best_signal, scores:{}}
    candidates = {}
    near_misses = {}
    for ticker, df in tqdm(enriched.items(), desc="Scanning + validating", unit="stock"):
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
            bt = Backtester(strat, cfg, exit_mode=EXIT_MODE,
                            max_hold_days=MAX_HOLD_DAYS)
            res = bt.run({ticker: df})
            m = res.metrics

            # Determine validation status + reason
            reasons = []
            if m.n_trades < MIN_HIST_TRADES:
                reasons.append(f"only {m.n_trades} hist. trades")
            if m.win_rate < MIN_HIST_WINRATE:
                reasons.append(f"win rate {m.win_rate*100:.0f}%")
            if m.profit_factor < MIN_HIST_PF:
                reasons.append(f"PF {m.profit_factor:.2f}")
            rs = rs_ranks.get(ticker, 0)
            if rs < RS_MIN:
                reasons.append(f"RS rank {rs:.0f}")
            validated = not reasons

            bucket = candidates if validated else near_misses
            c = bucket.setdefault(ticker, {
                "strategies": [], "signal": sig, "best_score": 0,
                "hist_winrate": m.win_rate, "hist_pf": m.profit_factor,
                "hist_expectancy": m.expectancy, "hist_trades": m.n_trades,
                "reason": "; ".join(reasons),
            })
            c["strategies"].append(strat.name)
            if sig.score > c["best_score"]:
                c["best_score"] = sig.score
                c["signal"] = sig
                c["hist_winrate"] = m.win_rate
                c["hist_pf"] = m.profit_factor
                c["hist_expectancy"] = m.expectancy
                c["hist_trades"] = m.n_trades
                c["reason"] = "; ".join(reasons)

    # ── Rank: confluence first, then historical edge ───────────────
    def _to_row(ticker, c, status):
        sig = c["signal"]
        risk_ps = sig.entry_price - sig.stop_loss
        confluence = len(set(c["strategies"]))
        combined = c["best_score"] + (confluence - 1) * 10 + c["hist_pf"] * 5
        return {
            "ticker": ticker, "strategies": sorted(set(c["strategies"])),
            "confluence": confluence, "score": c["best_score"],
            "entry": sig.entry_price, "stop": sig.stop_loss,
            "t1": sig.target_1, "t2": sig.target_2,
            "swing_target": sig.entry_price * (1 + SWING_TARGET_PCT),
            "risk_ps": risk_ps,
            "rr_swing": (sig.entry_price * SWING_TARGET_PCT) / risk_ps if risk_ps > 0 else 0,
            "hist_winrate": c["hist_winrate"], "hist_pf": c["hist_pf"],
            "hist_expectancy": c["hist_expectancy"], "hist_trades": c["hist_trades"],
            "combined": combined, "status": status, "reason": c.get("reason", ""),
            "rs": rs_ranks.get(ticker, 0),
            "from_high": _extension(enriched[ticker])[0],
            "above_ema": _extension(enriched[ticker])[1],
        }

    ranked = [_to_row(t, c, "VALIDATED") for t, c in candidates.items()]
    ranked.sort(key=lambda x: x["combined"], reverse=True)

    # Fill the board to at least BOARD_MIN with the best near-misses,
    # clearly labelled WATCH with the reason they fell short.
    watch = [_to_row(t, c, "WATCH") for t, c in near_misses.items()
             if t not in candidates]
    watch.sort(key=lambda x: x["combined"], reverse=True)
    n_fill = max(0, BOARD_MIN - len(ranked))
    ranked += watch[:max(n_fill, 2)]   # always show at least 2 watch names for context

    # ── Earnings blackout: never hold a 1-2 week swing into earnings ─
    print("Checking earnings dates for board candidates...")
    for r in ranked[:20]:
        d = _days_to_earnings(r["ticker"])
        if d is not None and d <= EARNINGS_BLACKOUT_DAYS:
            if r["status"] == "VALIDATED":
                r["status"] = "WATCH"
            r["reason"] = (r["reason"] + "; " if r["reason"] else "") + f"earnings in {d}d"

    # ── EXPERT SYSTEM (paper): enhanced_ema_pullback, the research
    #    winner (reclaim entry, stop=swing_low−0.25×ATR, 3.5×ATR trail).
    #    Shown separately from production; forward/paper validation. ──
    from strategies.enhanced_ema import compute_setup_table
    expert_rows = []
    if scale_pct > 0:
        for t, df in enriched.items():
            if len(df) < 220 or rs_ranks.get(t, 0) < 80:
                continue
            try:
                etab = compute_setup_table(df, stop_atr_buffer=0.25, entry_mode="reclaim")
            except Exception:
                continue
            erow = etab.iloc[-1]
            if not bool(erow["setup_ok"]):
                continue
            d = _days_to_earnings(t)
            if d is not None and d <= EARNINGS_BLACKOUT_DAYS:
                continue
            trigger = float(erow["trigger"])
            estop = float(erow["init_stop"])
            eatr = float(erow["atr"])
            risk_ps = trigger - estop
            if risk_ps <= 0:
                continue
            eqty = max(1, int(min(ACCOUNT * 0.01, MAX_RISK) * FX / risk_ps))
            if eqty * trigger > ACCOUNT * 0.10 * FX:
                eqty = max(1, int(ACCOUNT * 0.10 * FX / trigger))
            expert_rows.append(dict(
                ticker=t.replace(".NS", ""), entry=trigger, stop=estop,
                stop_pct=risk_ps / trigger * 100, trail_dist=3.5 * eatr,
                rs=rs_ranks.get(t, 0), qty=eqty))
    expert_rows.sort(key=lambda r: -r["rs"])

    # ── Position sizing (1% risk, hard-capped at MAX_RISK,
    #    position capped at 10% of account) ────────────────────────
    # Prices are USD; the budget is EUR. Convert the budget to the quote
    # currency before dividing by a USD risk-per-share, otherwise every
    # position comes out ~8% too small.
    risk_eur = min(ACCOUNT * 0.01 * (scale_pct / 100), MAX_RISK)
    risk_quote = risk_eur * FX                    # EUR budget -> USD
    max_pos_quote = ACCOUNT * 0.10 * FX           # EUR cap    -> USD
    for r in ranked:
        qty = max(1, int(risk_quote / r["risk_ps"])) if r["risk_ps"] > 0 else 1
        if qty * r["entry"] > max_pos_quote:
            qty = max(1, int(max_pos_quote / r["entry"]))
        r["qty"] = qty
        r["risk_acct"] = qty * r["risk_ps"] / FX  # actual risk in EUR

    # ── Build HTML ─────────────────────────────────────────────────
    rows = ""
    for r in ranked[:20]:
        tk = r["ticker"].replace(".NS", "")
        conf_badge = ("<span style='color:#3fb950;font-weight:700;'>●●●</span>" if r["confluence"] >= 3
                      else "<span style='color:#58a6ff;font-weight:700;'>●●</span>" if r["confluence"] == 2
                      else "<span style='color:#8b949e;'>●</span>")
        wr_color = "#3fb950" if r["hist_winrate"] >= 0.6 else "#58a6ff" if r["hist_winrate"] >= 0.5 else "#e3b341"
        strat_str = ", ".join(s.replace("_", " ") for s in r["strategies"])
        if r["status"] == "VALIDATED":
            status_html = "<span style='color:#3fb950;font-weight:700;'>✔ VALIDATED</span>"
        else:
            status_html = (f"<span style='color:#e3b341;font-weight:700;'>⚠ WATCH</span>"
                           f"<br><span style='color:#8b949e;font-size:10px;'>{r['reason']}</span>")
        dim = "" if r["status"] == "VALIDATED" else "opacity:0.75;"
        rows += f"""
        <tr style="{dim}">
          <td><strong style="color:#f0f6fc;">{tk}</strong></td>
          <td>{status_html}</td>
          <td style="text-align:center;">{conf_badge}</td>
          <td style="color:#58a6ff;font-size:12px;">{strat_str}</td>
          <td>{CUR}{r['entry']:,.2f}</td>
          <td style="color:#f85149;">{CUR}{r['stop']:,.2f}</td>
          <td style="color:#3fb950;">{CUR}{r['swing_target']:,.2f}</td>
          <td style="color:#3fb950;font-weight:700;">{r['rr_swing']:.1f}:1</td>
          <td style="color:{wr_color};font-weight:700;">{r['hist_winrate']*100:.0f}%</td>
          <td style="color:#c9d1d9;">{r['hist_pf']:.2f}</td>
          <td style="color:#8b949e;">{r['hist_trades']}</td>
          <td style="color:{'#3fb950' if r['rs'] >= 70 else '#8b949e'};">{r['rs']:.0f}</td>
          <td style="color:{'#e3b341' if r['from_high'] < 1 else '#c9d1d9'};">{r['from_high']:.1f}%</td>
          <td style="color:{'#f85149' if r['above_ema'] > 10 else '#3fb950' if r['above_ema'] < 5 else '#e3b341'};">{r['above_ema']:+.1f}%</td>
          <td>{r['qty']}</td>
        </tr>"""

    if not rows:
        rows = ('<tr><td colspan="14" style="text-align:center;color:#8b949e;padding:24px;">'
                'No setups flagged at all today. Stay in cash.</td></tr>')

    # Expert section rows
    ex_rows = ""
    for r in expert_rows[:10]:
        ex_rows += f"""
        <tr>
          <td><strong style="color:#f0f6fc;">{r['ticker']}</strong></td>
          <td>{CUR}{r['entry']:,.2f}</td>
          <td style="color:#f85149;">{CUR}{r['stop']:,.2f} (−{r['stop_pct']:.1f}%)</td>
          <td style="color:#e3b341;">close − {CUR}{r['trail_dist']:,.2f}</td>
          <td style="color:#8b949e;">{r['rs']:.0f}</td>
          <td>{r['qty']}</td>
        </tr>"""
    if not ex_rows:
        ex_rows = ('<tr><td colspan="6" style="text-align:center;color:#8b949e;'
                   'padding:18px;">No expert-system setups today. The reclaim '
                   'entry is selective (~1-2 signals/week) — no signal is a signal.</td></tr>')

    expert_html = f"""
  <div class="section-title">Expert System — enhanced_ema_pullback (PAPER validation)</div>
  <div class="card">
    <table>
      <thead><tr>
        <th>Stock</th><th>Buy-stop Entry</th><th>Initial Stop</th>
        <th>Exit: 3.5×ATR trail</th><th>RS</th><th>Qty</th>
      </tr></thead>
      <tbody>{ex_rows}</tbody>
    </table>
    <div class="legend">
      Research winner (PF 1.99 vs 1.28 backtested). <strong>Entry:</strong> buy next session
      when price trades through the buy-stop; skip if it opens &gt;2% above it.
      <strong>Exit:</strong> trail a stop 3.5×ATR below the highest close since entry —
      it rises, never falls; no profit target. <strong>Status: PAPER ONLY</strong> until 20
      forward trades confirm the backtest.
    </div>
  </div>"""

    n_val = sum(1 for r in ranked if r["status"] == "VALIDATED")
    n_watch = sum(1 for r in ranked if r["status"] == "WATCH")
    banner = (f"{n_val} VALIDATED setup(s) + {n_watch} watchlist name(s). "
              f"Only trade VALIDATED rows; WATCH rows show why they fell short."
              if n_val else
              f"No fully validated setups today — {n_watch} watchlist name(s) shown for context. Stay in cash or wait.")
    bcolor = "#3fb950" if n_val else "#e3b341"

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
    &nbsp;Scale {scale_pct}% &nbsp;|&nbsp; Account {ACCT_CUR}{ACCOUNT:,} &nbsp;|&nbsp;
    Risk/trade {ACCT_CUR}{risk_eur:,.0f} &nbsp;|&nbsp; Max open {MAX_OPEN} (heat {ACCT_CUR}{risk_eur*MAX_OPEN:,.0f})
    &nbsp;|&nbsp; <span style="color:var(--green);">LIVE + BACKTESTED</span>
  </div>
</div>
<div class="container">
  <div class="alert">⚡ {banner}</div>
  <div class="section-title">Today's Validated Swing Setups</div>
  <div class="card">
    <table>
      <thead><tr>
        <th>Stock</th><th>Status</th><th>Conf.</th><th>Strategy</th><th>Entry</th><th>Stop</th>
        <th>Target +10%</th><th>R:R</th><th>Hist Win%</th><th>Hist PF</th><th>#Trades</th><th>RS</th>
        <th>Below 52w High</th><th>vs EMA20</th><th>Qty</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="legend">
      <strong>Conf.</strong> = confluence: ●●● three strategies agree, ●● two, ● one. More dots = higher conviction.<br>
      <strong>Hist Win%</strong> / <strong>Hist PF</strong> = how this exact strategy performed on THIS stock over 5 years (win rate & profit factor).<br>
      <strong>VALIDATED requires:</strong> historical win rate ≥50%, profit factor ≥1.2, RS rank ≥70 (stock outperforms 70% of universe),
      and no earnings within {EARNINGS_BLACKOUT_DAYS} days. Exit: ATR trailing stop — let winners run (A/B tested: beats fixed targets and partial banking).<br>
      Target is +10% swing goal; R:R compares that to your stop risk. Qty sized to 1% account risk, capped at 10% position.
      <strong>Never hold more than {MAX_OPEN} positions at once</strong> — total open risk stays ≤ {ACCT_CUR}{risk_eur*MAX_OPEN:,.0f}.
    </div>
  </div>
  {expert_html}
  <div style="text-align:center;color:var(--muted);font-size:11px;margin-top:24px;padding-top:16px;border-top:1px solid var(--border);">
    Quant Platform — research only, not financial advice — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </div>
</div></body></html>"""

    reports = _ROOT / "reports"
    reports.mkdir(exist_ok=True)
    prefix = "india_" if MARKET == "INDIA" else ""
    out = reports / f"{prefix}swing_dashboard_{today.isoformat()}.html"
    out.write_text(html, encoding="utf-8")

    # Markdown summary — phone-readable straight on GitHub
    md = [f"# Swing Scan — {today.strftime('%A, %B %d, %Y')}",
          "",
          f"**Regime:** {regime.replace('_', ' ')} (scale {scale_pct}%)  ",
          f"**Account:** {CUR}{ACCOUNT:,} | Risk/trade {ACCT_CUR}{risk_eur:,.0f} | Max open {MAX_OPEN}",
          "",
          f"> {banner}",
          "",
          "| Stock | Status | Strategy | Entry | Stop | Target +10% | R:R | HistWin | PF | RS | BelowHigh | vsEMA20 | Qty |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ranked[:15]:
        tk = r["ticker"].replace(".NS", "")
        status = "✅" if r["status"] == "VALIDATED" else f"⚠ {r['reason']}"
        md.append(f"| **{tk}** | {status} | {', '.join(r['strategies'])} "
                  f"| {CUR}{r['entry']:,.2f} | {CUR}{r['stop']:,.2f} "
                  f"| {CUR}{r['swing_target']:,.2f} | {r['rr_swing']:.1f}:1 "
                  f"| {r['hist_winrate']*100:.0f}% | {r['hist_pf']:.2f} "
                  f"| {r['rs']:.0f} | {r['from_high']:.1f}% | {r['above_ema']:+.1f}% | {r['qty']} |")
    md += ["", "## Expert System — enhanced_ema_pullback (PAPER)", ""]
    if expert_rows:
        md += ["| Stock | Buy-stop Entry | Initial Stop | Exit (3.5×ATR trail) | RS | Qty |",
               "|---|---|---|---|---|---|"]
        for r in expert_rows[:10]:
            md.append(f"| **{r['ticker']}** | {CUR}{r['entry']:,.2f} "
                      f"| {CUR}{r['stop']:,.2f} (−{r['stop_pct']:.1f}%) "
                      f"| close − {CUR}{r['trail_dist']:,.2f} | {r['rs']:.0f} | {r['qty']} |")
    else:
        md.append("_No expert setups today — the reclaim entry is selective; no signal is a signal._")
    md += ["", f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — research only, not financial advice._"]
    (reports / f"{prefix}latest.md").write_text("\n".join(md), encoding="utf-8")
    (reports / f"{prefix}latest.html").write_text(html, encoding="utf-8")

    print(f"\nDashboard: {out}")
    print(f"  {n_val} VALIDATED + {n_watch} WATCH names on the board")
    if not os.getenv("CI"):
        webbrowser.open(out.resolve().as_uri())
    return out


if __name__ == "__main__":
    build()
