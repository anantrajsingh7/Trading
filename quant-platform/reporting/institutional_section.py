"""
Second dashboard section: INSTITUTIONAL MOMENTUM / TREND SETUPS.

Lives in its own module so the existing "Today's Validated Swing Setups"
dashboard is untouched. swing_dashboard.py calls build_section() and
appends the returned HTML/markdown underneath the existing table.

Research / paper only. Nothing here feeds the production scanner.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from strategies.institutional_momentum import (
    compute_setup_table, momentum_quality_score, entry_quality,
)

# Portfolio rules for THIS strategy only (independent of the swing book)
ACCOUNT_EUR = 30_000.0
RISK_PCT = 0.005          # 0.50% = €150
MAX_OPEN = 4
MAX_HEAT_PCT = 0.02       # €600
MAX_POSITION_PCT = 0.10
MAX_SECTOR_PCT = 0.25
TRAIL_ATR = 3.0
RS_MIN = 80.0
EARNINGS_BLACKOUT = 14    # calendar days ≈ 10 trading days

REGIME_MULT = {"BULL_STRONG": 1.0, "BULL_MODERATE": 0.75, "NEUTRAL": 0.5,
               "BEAR_MODERATE": 0.0, "BEAR_STRONG": 0.0}

STATUS = {
    "PAPER_ELIGIBLE": ("🟢", "PAPER ELIGIBLE", "#3fb950"),
    "SETUP_VALID":    ("🔵", "SETUP VALID", "#58a6ff"),
    "WATCH":          ("🟡", "WATCH", "#e3b341"),
    "INSUFFICIENT":   ("⚪", "INSUFFICIENT SAMPLE", "#8b949e"),
    "REJECTED":       ("🔴", "REJECTED", "#f85149"),
    "BLOCKED":        ("⛔", "PORTFOLIO BLOCKED", "#ffa657"),
}

# Strategy-level acceptance from the last backtest run. Until the
# backtest clears every gate this stays False, and no row may be
# labelled PAPER ELIGIBLE — a stock's own history is never enough.
STRATEGY_PAPER_ELIGIBLE = False


def _vol_state(vix: float | None) -> tuple[str, float]:
    if vix is None or not np.isfinite(vix):
        return "UNKNOWN", 1.0
    if vix >= 35:
        return "EXTREME", 0.0
    if vix >= 25:
        return "ELEVATED", 0.5
    return "NORMAL", 1.0


def build_section(
    enriched: dict[str, pd.DataFrame],
    rs_ranks: dict[str, float],
    regime: str,
    fx: float,
    sectors: dict[str, str],
    days_to_earnings,
    vix: float | None = None,
    oos_stats: dict | None = None,
) -> tuple[str, list[str]]:
    """Return (html, markdown_lines) for the institutional section."""
    reg_mult = REGIME_MULT.get(regime, 0.5)
    vol_state, vol_mult = _vol_state(vix)
    risk_eur = ACCOUNT_EUR * RISK_PCT * reg_mult * vol_mult
    oos = oos_stats or {}

    rows = []
    if reg_mult > 0 and vol_mult > 0:
        for t, df in enriched.items():
            if len(df) < 260:
                continue
            rs = rs_ranks.get(t, 0.0)
            try:
                tab = compute_setup_table(df, 0.25, "reclaim")
            except Exception:
                continue
            r = tab.iloc[-1]
            if not bool(r["setup_ok"]):
                continue

            trigger, stop, atr = float(r["trigger"]), float(r["init_stop"]), float(r["atr"])
            risk_ps = trigger - stop
            if risk_ps <= 0:
                continue

            dv = float((df["close"] * df["volume"]).tail(20).mean())
            score = momentum_quality_score(r, rs, reg_mult, dv)
            eq = entry_quality(trigger, float(r["ema20"]))

            dte = days_to_earnings(t)
            earn_txt = "unknown" if dte is None else (
                f"{dte}d" if dte < 9000 else "none scheduled")

            # sizing: EUR risk -> USD, then caps
            qty = max(1, int(risk_eur * fx / risk_ps))
            pos_usd = qty * trigger
            if pos_usd > ACCOUNT_EUR * MAX_POSITION_PCT * fx:
                qty = max(1, int(ACCOUNT_EUR * MAX_POSITION_PCT * fx / trigger))
                pos_usd = qty * trigger
            risk_actual = qty * risk_ps / fx
            pos_eur = pos_usd / fx

            # ── status ────────────────────────────────────────────
            reasons = []
            if rs < RS_MIN:
                reasons.append(f"RS {rs:.0f} < {RS_MIN:.0f}")
            if dte is None:
                reasons.append("EARNINGS_UNKNOWN")
            elif dte <= EARNINGS_BLACKOUT:
                reasons.append(f"earnings {dte}d")
            if eq == "DO_NOT_CHASE":
                reasons.append("extended")

            st = oos.get(t, {})
            n_oos = st.get("trades", 0)

            if reasons:
                status = "REJECTED" if ("EARNINGS_UNKNOWN" in reasons[0]
                                        or rs < RS_MIN) else "WATCH"
            elif not STRATEGY_PAPER_ELIGIBLE:
                # strategy-level evidence not yet established
                status = "SETUP_VALID" if n_oos >= 25 else "INSUFFICIENT"
            else:
                status = "PAPER_ELIGIBLE"

            rows.append(dict(
                ticker=t, status=status, reasons="; ".join(reasons),
                score=score, eq=eq, entry=trigger,
                zone_lo=trigger * 0.995, zone_hi=trigger * 1.02,
                stop=stop, trail=TRAIL_ATR * atr,
                risk_pct=risk_ps / trigger * 100,
                rs=rs, ret20=float(r["ret20"]) * 100,
                ret60=float(r["ret60"]) * 100, ret120=float(r["ret120"]) * 100,
                vs_ema20=(trigger / float(r["ema20"]) - 1) * 100,
                vs_ema50=(trigger / float(r["ema50"]) - 1) * 100,
                from_high=float(r["from_high"]) * 100,
                pullback=float(r["pullback_pct"]) * 100,
                rel_vol=float(r["rel_vol"]), rsi=float(r["rsi"]),
                sector=sectors.get(t, "Unknown"), earnings=earn_txt,
                oos_trades=n_oos, oos_win=st.get("win_rate", np.nan),
                oos_pf=st.get("pf", np.nan), oos_exp=st.get("expectancy", np.nan),
                qty=qty, risk_eur=risk_actual, pos_eur=pos_eur,
            ))

    rows.sort(key=lambda x: -x["score"])

    # portfolio heat as rows are taken in rank order (assumes flat book)
    heat = 0.0
    sector_used: dict[str, float] = {}
    for r in rows:
        blocked = None
        if len([x for x in rows[:rows.index(r)] if x.get("_taken")]) >= MAX_OPEN:
            blocked = "max positions"
        elif heat + r["risk_eur"] > ACCOUNT_EUR * MAX_HEAT_PCT:
            blocked = "heat cap"
        elif (sector_used.get(r["sector"], 0) + r["pos_eur"]) > ACCOUNT_EUR * MAX_SECTOR_PCT:
            blocked = "sector cap"
        if blocked and r["status"] in ("PAPER_ELIGIBLE", "SETUP_VALID", "INSUFFICIENT"):
            r["status"] = "BLOCKED"
            r["reasons"] = blocked
        else:
            r["_taken"] = True
            heat += r["risk_eur"]
            sector_used[r["sector"]] = sector_used.get(r["sector"], 0) + r["pos_eur"]
        r["heat_after"] = heat / ACCOUNT_EUR * 100

    return _render(rows, regime, reg_mult, vol_state, vol_mult, risk_eur, vix)


def _fmt(x, p=2, suffix=""):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}{suffix}"


def _render(rows, regime, reg_mult, vol_state, vol_mult, risk_eur, vix):
    hdr = ["STOCK", "STATUS", "SCORE", "ENTRY QUALITY", "ENTRY", "ENTRY ZONE",
           "INIT STOP", "TRAIL", "INIT RISK", "RS", "20D", "60D", "120D",
           "vsEMA20", "vsEMA50", "BELOW 52W", "PULLBACK", "RELVOL", "RSI",
           "REGIME", "SECTOR", "EARNINGS", "OOS N", "OOS WIN%", "OOS PF",
           "OOS EXP", "QTY", "€RISK", "POS €", "HEAT AFTER"]

    if not rows:
        body = (f'<tr><td colspan="{len(hdr)}" style="text-align:center;color:#8b949e;'
                f'padding:26px;font-size:14px;">NO QUALIFYING INSTITUTIONAL MOMENTUM '
                f'SETUPS TODAY.<br><span style="font-size:11px;">Regime {regime} · '
                f'volatility {vol_state}. Rules were not relaxed to populate this table.'
                f'</span></td></tr>')
        md = ["", "## INSTITUTIONAL MOMENTUM / TREND SETUPS", "",
              "**NO QUALIFYING INSTITUTIONAL MOMENTUM SETUPS TODAY.**", "",
              f"_Regime {regime} · volatility {vol_state}. Rules were not relaxed._"]
    else:
        body = ""
        for r in rows:
            emoji, label, colour = STATUS[r["status"]]
            body += "<tr>" + "".join([
                f'<td><strong style="color:#f0f6fc;">{r["ticker"]}</strong></td>',
                f'<td style="color:{colour};white-space:nowrap;">{emoji} {label}'
                + (f'<br><span style="color:#8b949e;font-size:10px;">{r["reasons"]}</span>'
                   if r["reasons"] else "") + "</td>",
                f'<td style="font-weight:700;color:#58a6ff;">{r["score"]:.1f}</td>',
                f'<td>{r["eq"]}</td>',
                f'<td>${r["entry"]:,.2f}</td>',
                f'<td style="font-size:11px;">${r["zone_lo"]:,.2f}–${r["zone_hi"]:,.2f}</td>',
                f'<td style="color:#f85149;">${r["stop"]:,.2f}</td>',
                f'<td style="color:#e3b341;">close−${r["trail"]:,.2f}</td>',
                f'<td>{r["risk_pct"]:.1f}%</td>',
                f'<td>{r["rs"]:.0f}</td>',
                f'<td>{r["ret20"]:+.1f}%</td>',
                f'<td>{r["ret60"]:+.1f}%</td>',
                f'<td>{r["ret120"]:+.1f}%</td>',
                f'<td>{r["vs_ema20"]:+.1f}%</td>',
                f'<td>{r["vs_ema50"]:+.1f}%</td>',
                f'<td>{r["from_high"]:.1f}%</td>',
                f'<td>{r["pullback"]:.1f}%</td>',
                f'<td>{r["rel_vol"]:.2f}</td>',
                f'<td>{r["rsi"]:.0f}</td>',
                f'<td style="font-size:11px;">{regime}</td>',
                f'<td style="font-size:11px;">{r["sector"]}</td>',
                f'<td style="font-size:11px;">{r["earnings"]}</td>',
                f'<td>{r["oos_trades"]}</td>',
                f'<td>{_fmt(r["oos_win"], 0, "%")}</td>',
                f'<td>{_fmt(r["oos_pf"])}</td>',
                f'<td>{_fmt(r["oos_exp"])}</td>',
                f'<td>{r["qty"]}</td>',
                f'<td>€{r["risk_eur"]:,.0f}</td>',
                f'<td>€{r["pos_eur"]:,.0f}</td>',
                f'<td>{r["heat_after"]:.2f}%</td>',
            ]) + "</tr>"

        md = ["", "## INSTITUTIONAL MOMENTUM / TREND SETUPS", "",
              "| Stock | Status | Score | Entry Quality | Entry | Stop | Trail | RS | "
              "60D | Pullback | RelVol | Sector | Earnings | Qty | €Risk |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            emoji, label, _ = STATUS[r["status"]]
            md.append(
                f"| **{r['ticker']}** | {emoji} {label}"
                + (f" ({r['reasons']})" if r["reasons"] else "")
                + f" | {r['score']:.1f} | {r['eq']} | ${r['entry']:,.2f} "
                f"| ${r['stop']:,.2f} | close−${r['trail']:,.2f} | {r['rs']:.0f} "
                f"| {r['ret60']:+.1f}% | {r['pullback']:.1f}% | {r['rel_vol']:.2f} "
                f"| {r['sector']} | {r['earnings']} | {r['qty']} | €{r['risk_eur']:,.0f} |")

    # per-candidate explanations (top 3)
    expl_html, expl_md = "", []
    for r in rows[:3]:
        why = (f"RS {r['rs']:.0f} · Stage-2 uptrend · {r['pullback']:.1f}% controlled pullback · "
               f"volume contracted {max(0,(1-r['rel_vol'])*100):.0f}% · EMA20 reclaimed · "
               f"regime {regime} · earnings {r['earnings']}")
        inval = (f"entry gaps &gt;2% above ${r['entry']:,.2f} · close below ${r['stop']:,.2f} · "
                 f"regime turns bear · earnings date changes")
        expl_html += (f'<div style="margin:10px 0;padding:10px 12px;background:#0d1117;'
                      f'border-left:3px solid #58a6ff;border-radius:4px;">'
                      f'<strong style="color:#f0f6fc;">{r["ticker"]}</strong> '
                      f'<span style="color:#8b949e;font-size:11px;">(score {r["score"]:.1f})</span>'
                      f'<div style="color:#3fb950;font-size:11px;margin-top:6px;">'
                      f'<strong>WHY IT QUALIFIES:</strong> {why}</div>'
                      f'<div style="color:#f85149;font-size:11px;margin-top:4px;">'
                      f'<strong>WHAT INVALIDATES IT:</strong> {inval}</div></div>')
        expl_md += [f"**{r['ticker']}** (score {r['score']:.1f})",
                    f"- WHY IT QUALIFIES: {why}",
                    f"- WHAT INVALIDATES IT: {inval.replace('&gt;', '>')}", ""]

    vix_txt = f"VIX {vix:.1f}" if vix is not None and np.isfinite(vix) else "VIX unavailable"
    html = f"""
  <div class="section-title">Institutional Momentum / Trend Setups</div>
  <div class="card">
    <div style="color:#8b949e;font-size:11px;margin-bottom:10px;">
      Independent research strategy — separate book, separate rules.
      Regime <strong>{regime}</strong> (×{reg_mult:.2f}) · Volatility <strong>{vol_state}</strong>
      (×{vol_mult:.2f}) · {vix_txt} · Risk/trade <strong>€{risk_eur:,.0f}</strong> (0.50% base)
      · Max {MAX_OPEN} positions · Heat cap €{ACCOUNT_EUR*MAX_HEAT_PCT:,.0f}
      · Exit: {TRAIL_ATR}×ATR trailing stop, no profit target
    </div>
    <div style="overflow-x:auto;">
      <table><thead><tr>{''.join(f'<th>{h}</th>' for h in hdr)}</tr></thead>
      <tbody>{body}</tbody></table>
    </div>
    {expl_html}
    <div class="legend">
      🟢 PAPER ELIGIBLE · 🔵 SETUP VALID · 🟡 WATCH · ⚪ INSUFFICIENT SAMPLE ·
      🔴 REJECTED · ⛔ PORTFOLIO BLOCKED.
      <strong>No row is labelled PAPER ELIGIBLE until the strategy itself clears the
      acceptance standard out-of-sample</strong> (≥200 OOS trades, PF ≥1.25, ≥65%
      positive walk-forward windows, drawdown ≤12%). A few good trades on one stock
      is not evidence. Research only — not financial advice.
    </div>
  </div>"""

    md += [""] + expl_md
    return html, md
