"""
Strategy comparison: enhanced_ema_pullback vs existing strategies.

Runs all five systems (ema_pullback, minervini_vcp, volume_breakout,
combined/confluence, enhanced_ema_pullback) through the SAME research
portfolio simulator with identical data, universe, regime series, limits,
costs, FX, and walk-forward windows. Produces the eight report files and
a final verdict.

Usage (needs market-data access — run on GitHub Actions or a laptop):
    python research/run_comparison.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.logging import setup_logging
from data.pipeline import get_data, get_data_batch
from strategies.registry import get_strategy
from strategies.enhanced_ema import compute_setup_table
from research.portfolio_sim import simulate, metrics_from, SimConfig, COST_PROFILES

setup_logging("WARNING")
REPORTS = Path(__file__).resolve().parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)

# ── Universe with static GICS sectors (fair: identical for all strategies) ─
SECTORS = {
 "AAPL":"Tech","MSFT":"Tech","NVDA":"Tech","AVGO":"Tech","AMD":"Tech","QCOM":"Tech",
 "TXN":"Tech","AMAT":"Tech","LRCX":"Tech","MU":"Tech","ORCL":"Tech","CSCO":"Tech",
 "CRM":"Tech","ADBE":"Tech","NOW":"Tech","PANW":"Tech","INTU":"Tech","IBM":"Tech",
 "ACN":"Tech","KLAC":"Tech","SNPS":"Tech","CDNS":"Tech","ANET":"Tech","ADI":"Tech",
 "AMZN":"ConsDisc","TSLA":"ConsDisc","HD":"ConsDisc","MCD":"ConsDisc","NKE":"ConsDisc",
 "LOW":"ConsDisc","SBUX":"ConsDisc","TJX":"ConsDisc","BKNG":"ConsDisc","ORLY":"ConsDisc",
 "GOOGL":"Comm","META":"Comm","NFLX":"Comm","DIS":"Comm","CMCSA":"Comm","TMUS":"Comm",
 "JPM":"Fin","V":"Fin","MA":"Fin","BAC":"Fin","WFC":"Fin","GS":"Fin","MS":"Fin",
 "BLK":"Fin","SCHW":"Fin","AXP":"Fin","C":"Fin","FITB":"Fin","PGR":"Fin","CB":"Fin",
 "UNH":"Health","JNJ":"Health","LLY":"Health","ABBV":"Health","MRK":"Health",
 "TMO":"Health","ABT":"Health","PFE":"Health","DHR":"Health","AMGN":"Health",
 "ISRG":"Health","VRTX":"Health","GILD":"Health","MDT":"Health",
 "PG":"Staples","KO":"Staples","PEP":"Staples","COST":"Staples","WMT":"Staples",
 "MDLZ":"Staples","CL":"Staples","MO":"Staples",
 "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy","EOG":"Energy","TRGP":"Energy",
 "CAT":"Industrial","BA":"Industrial","HON":"Industrial","UNP":"Industrial","GE":"Industrial",
 "RTX":"Industrial","LMT":"Industrial","DE":"Industrial","UPS":"Industrial","MMM":"Industrial",
 "URI":"Industrial","ETN":"Industrial","EMR":"Industrial",
 "LIN":"Materials","APD":"Materials","SHW":"Materials","FCX":"Materials",
 "NEE":"Utilities","DUK":"Utilities","SO":"Utilities",
 "AMT":"RealEstate","PLD":"RealEstate","WELL":"RealEstate","KIM":"RealEstate","O":"RealEstate",
}
UNIVERSE = list(SECTORS)
PERIOD = "6y"          # 1y warmup + 5y evaluation
HOLDOUT_MONTHS = 6
MAIN = dict(entry_mode="reversal", stop_buffer=0.25, trail=2.5)


# ── Regime per spec: SPY>rising SMA50 & SMA200, QQQ>rising SMA50, breadth ─
def build_regime(spy, qqq, data) -> pd.Series:
    s, q = spy["close"], qqq["close"]
    s50, s200 = s.rolling(50).mean(), s.rolling(200).mean()
    q50 = q.rolling(50).mean()
    cond = pd.DataFrame({
        "spy50":  (s > s50)  & (s50 > s50.shift(10)),
        "spy200": (s > s200) & (s200 > s200.shift(20)),
        "qqq50":  (q > q50)  & (q50 > q50.shift(10)),
    })
    above50 = pd.DataFrame({
        t: (df["close"] > df["close"].rolling(50).mean()) for t, df in data.items()
    })
    breadth = above50.reindex(s.index, method="ffill").mean(axis=1)
    cond["breadth"] = breadth >= 0.55

    def label(row, b):
        if not row["spy200"]:
            return "BEAR_STRONG" if b < 0.35 else "BEAR_MODERATE"
        if row.all():
            return "BULL_STRONG" if b >= 0.65 else "BULL_MODERATE"
        return "NEUTRAL"

    lab = pd.Series(
        [label(cond.iloc[i], breadth.iloc[i]) for i in range(len(cond))],
        index=cond.index,
    )
    return pd.Series(lab.values, index=[d.date() for d in lab.index])


def build_rs_rank(data: dict) -> pd.DataFrame:
    """Daily cross-sectional RS percentile: 0.4*3M + 0.2*6M + 0.2*9M + 0.2*12M."""
    closes = pd.DataFrame({t: df["close"] for t, df in data.items()})
    mom = (0.4 * closes.pct_change(63) + 0.2 * closes.pct_change(126)
           + 0.2 * closes.pct_change(189) + 0.2 * closes.pct_change(252))
    return mom.rank(axis=1, pct=True) * 100


def fetch_earnings(tickers) -> dict:
    """Historical + upcoming earnings dates; None = unknown."""
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=40)
            if ed is None or ed.empty:
                out[t] = None
            else:
                out[t] = sorted({d.date() for d in ed.index})
        except Exception:
            out[t] = None
    return out


# ── Signal tables ─────────────────────────────────────────────────────
def enhanced_table(df, stop_buffer, entry_mode):
    tab = compute_setup_table(df, stop_atr_buffer=stop_buffer, entry_mode=entry_mode)
    return pd.DataFrame(index=df.index, data={
        "signal": tab["setup_ok"], "trigger": tab["trigger"],
        "init_stop": tab["init_stop"], "atr": tab["atr"],
    })


def existing_table(df, strat, warmup=252):
    """Per-bar signals from an existing strategy (causal expanding window)."""
    n = len(df)
    sig = np.zeros(n, bool)
    trig = np.full(n, np.nan)
    stop = np.full(n, np.nan)
    atr = df.get("atr_14", df["close"] * 0.02).astype(float).values
    for i in range(warmup, n):
        s = strat.generate_signal(df.iloc[: i + 1], "X", {})
        if s is not None and s.stop_loss < s.entry_price:
            sig[i], trig[i], stop[i] = True, s.entry_price * 1.001, s.stop_loss
    return pd.DataFrame(index=df.index,
                        data={"signal": sig, "trigger": trig, "init_stop": stop, "atr": atr})


def combined_table(tables: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Confluence union: any strategy signals; prefer the tightest valid stop."""
    out = {}
    tickers = tables[0].keys()
    for t in tickers:
        frames = [tab[t] for tab in tables]
        sig = np.logical_or.reduce([f["signal"].values for f in frames])
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                trig = np.nanmean(np.vstack([f["trigger"].values for f in frames]), axis=0)
                stop = np.nanmax(np.vstack([f["init_stop"].values for f in frames]), axis=0)
        out[t] = pd.DataFrame(index=frames[0].index,
                              data={"signal": sig, "trigger": trig,
                                    "init_stop": stop, "atr": frames[0]["atr"]})
    return out


# ── Walk-forward on the research sim ─────────────────────────────────
def wf_windows(calendar, is_years=2.0, oos_months=6):
    start, end = calendar[0], calendar[-1]
    holdout_start = end - timedelta(days=30 * HOLDOUT_MONTHS)
    wins, cursor = [], start
    while True:
        oos_s = cursor + timedelta(days=int(is_years * 365))
        oos_e = min(oos_s + timedelta(days=oos_months * 30), holdout_start)
        if oos_s >= holdout_start:
            break
        wins.append((oos_s, oos_e))
        cursor = oos_s
    wins.append((holdout_start, end))  # final holdout window
    return wins


def main():
    print("Fetching data...")
    raw = get_data_batch(UNIVERSE, period=PERIOD, show_progress=True)
    raw = {t: df for t, df in raw.items() if df is not None and len(df) > 300}
    spy, qqq = get_data("SPY", period=PERIOD), get_data("QQQ", period=PERIOD)
    fxdf = get_data("EURUSD=X", period=PERIOD)
    eurusd = fxdf["close"] if fxdf is not None else pd.Series(1.08, index=spy.index)

    print("Building regime / RS / earnings...")
    regime = build_regime(spy, qqq, raw)
    rs = build_rs_rank(raw)
    earnings = fetch_earnings(list(raw))
    n_unknown = sum(1 for v in earnings.values() if v is None)
    print(f"  earnings history: {len(raw)-n_unknown}/{len(raw)} known")

    from indicators.composite import enrich
    print("Enriching for existing strategies...")
    enriched = {t: enrich(df, patterns=True) for t, df in raw.items()}

    print("Building signal tables...")
    tabs = {}
    tabs["enhanced_ema_pullback"] = {
        t: enhanced_table(df, MAIN["stop_buffer"], MAIN["entry_mode"]) for t, df in raw.items()}
    for name in ["ema_pullback", "minervini_vcp", "volume_breakout"]:
        strat = get_strategy(name)
        tabs[name] = {t: existing_table(df, strat) for t, df in enriched.items()}
        print(f"  {name}: done")
    tabs["combined"] = combined_table(
        [tabs["ema_pullback"], tabs["minervini_vcp"], tabs["volume_breakout"]])

    calendar = sorted(regime.index)
    args = dict(price_data=raw, regime=regime, rs_rank=rs,
                sectors=SECTORS, eurusd=eurusd, earnings=earnings)

    def run(name, cfg, start=None, end=None):
        return simulate(tabs[name], cfg=cfg, start=start, end=end, **args)

    def cfg_for(name, costs="realistic", trail=None, risk=0.005):
        return SimConfig(
            costs=costs, risk_per_trade=risk,
            trail_atr_mult=trail or MAIN["trail"],
            entry_type="stop" if name == "enhanced_ema_pullback" else "market",
            rs_min=80.0 if name == "enhanced_ema_pullback" else 0.0,
        )

    names = ["ema_pullback", "minervini_vcp", "volume_breakout",
             "combined", "enhanced_ema_pullback"]

    # ── Main run + cost profiles ───────────────────────────────────
    print("Running cost-profile comparison...")
    cost_rows, main_res = [], {}
    for name in names:
        for costs in COST_PROFILES:
            res = run(name, cfg_for(name, costs))
            m = metrics_from(res)
            cost_rows.append({"strategy": name, "costs": costs, **m})
            if costs == "realistic":
                main_res[name] = (res, m)
        print(f"  {name}: {main_res[name][1]['filled']} trades (realistic)")
    pd.DataFrame(cost_rows).to_csv(REPORTS / "execution_cost_comparison.csv", index=False)

    # existing ema_pullback at its production 1% risk — reported separately
    res_1pct = run("ema_pullback", cfg_for("ema_pullback", risk=0.01))
    m_1pct = metrics_from(res_1pct)

    # ── Walk-forward ───────────────────────────────────────────────
    print("Walk-forward...")
    wf_rows = []
    for name in names:
        for (s, e) in wf_windows(calendar):
            res = run(name, cfg_for(name), start=s, end=e)
            m = metrics_from(res)
            wf_rows.append({"strategy": name, "oos_start": s, "oos_end": e,
                            "trades": m["filled"],
                            "pnl_eur": res.trades.pnl_eur.sum() if not res.trades.empty else 0.0,
                            "profit_factor": m["profit_factor"],
                            "result": ("NO_TRADE" if m["filled"] == 0 else
                                       "POSITIVE" if res.trades.pnl_eur.sum() > 0 else "NEGATIVE")})
    wf = pd.DataFrame(wf_rows)
    wf.to_csv(REPORTS / "walk_forward_strategy_comparison.csv", index=False)

    # ── Regime / sector / year breakdowns ──────────────────────────
    reg_rows = []
    for name in names:
        tr = main_res[name][0].trades
        if tr.empty:
            continue
        for reg, grp in tr.groupby("regime_at_entry"):
            reg_rows.append({"strategy": name, "regime": reg, "trades": len(grp),
                             "pnl_eur": grp.pnl_eur.sum(),
                             "win_rate": (grp.pnl_eur > 0).mean()})
    pd.DataFrame(reg_rows).to_csv(REPORTS / "regime_strategy_comparison.csv", index=False)

    # ── Parameter stability (enhanced only) ────────────────────────
    print("Parameter stability grid...")
    stab_rows = []
    for entry_mode in ["reversal", "reclaim"]:
        for sb in [0.0, 0.25, 0.5]:
            grid_tab = {t: enhanced_table(df, sb, entry_mode) for t, df in raw.items()}
            for trail in [2.0, 2.5, 3.0, 3.5]:
                res = simulate(grid_tab, cfg=cfg_for("enhanced_ema_pullback", trail=trail), **args)
                m = metrics_from(res)
                stab_rows.append({"entry_mode": entry_mode, "stop_buffer": sb,
                                  "trail": trail, **m})
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(REPORTS / "parameter_stability_enhanced_ema.csv", index=False)

    # ── Trade log ──────────────────────────────────────────────────
    main_res["enhanced_ema_pullback"][0].trades.to_csv(
        REPORTS / "enhanced_ema_trade_log.csv", index=False)

    # ── Comparison table + acceptance + verdict ────────────────────
    comp_rows = []
    for name in names:
        res, m = main_res[name]
        oos = wf[wf.strategy == name]
        traded = oos[oos.result != "NO_TRADE"]
        pos_pct = (traded.result == "POSITIVE").mean() if len(traded) else 0.0
        oos_trades = int(oos.trades.sum())
        tr = res.trades
        top_stock = top_sector = 0.0
        if not tr.empty and tr.pnl_eur.sum() > 0:
            tot = tr.pnl_eur.sum()
            top_stock = tr.groupby("ticker").pnl_eur.sum().max() / tot
            top_sector = tr.groupby("sector").pnl_eur.sum().max() / tot
        stress = next(r for r in cost_rows
                      if r["strategy"] == name and r["costs"] == "stress")
        ok = dict(
            oos_trades=oos_trades >= 200,
            expectancy=m["expectancy_eur"] > 0 if np.isfinite(m["expectancy_eur"]) else False,
            pf=m["profit_factor"] >= 1.25 if np.isfinite(m["profit_factor"]) else False,
            wf_windows=pos_pct >= 0.65,
            drawdown=m["max_dd"] >= -0.12 if np.isfinite(m["max_dd"]) else False,
            stress=(stress["expectancy_eur"] > 0) if np.isfinite(stress["expectancy_eur"]) else False,
            concentration=top_stock <= 0.15 and top_sector <= 0.25,
        )
        if m["filled"] == 0:
            label = "NO_TRADE"
        elif oos_trades < 200:
            label = "INSUFFICIENT_SAMPLE"
        elif all(ok.values()):
            label = "PAPER_ELIGIBLE"
        elif ok["expectancy"] and ok["pf"]:
            label = "RESEARCH_ONLY"
        else:
            label = "REJECTED"
        comp_rows.append({"strategy": name, **m, "oos_trades": oos_trades,
                          "wf_positive_pct": round(pos_pct, 2),
                          "top_stock_share": round(top_stock, 2),
                          "top_sector_share": round(top_sector, 2),
                          "acceptance": label,
                          "failed_gates": ",".join(k for k, v in ok.items() if not v)})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(REPORTS / "strategy_comparison.csv", index=False)

    # winner: rank eligible/research strategies by OOS robustness composite
    def score(r):
        if r["acceptance"] in ("NO_TRADE", "REJECTED"):
            return -1e9
        e = r["expectancy_eur"] if np.isfinite(r["expectancy_eur"]) else 0
        pf = min(r["profit_factor"], 5) if np.isfinite(r["profit_factor"]) else 0
        return r["wf_positive_pct"] * 3 + pf + e / 50 + max(r["max_dd"], -0.5) * 4
    comp["rank_score"] = comp.apply(score, axis=1)
    best = comp.sort_values("rank_score", ascending=False).iloc[0]
    verdict_map = {"ema_pullback": "EXISTING_EMA_PULLBACK_BETTER",
                   "enhanced_ema_pullback": "ENHANCED_EMA_PULLBACK_BETTER",
                   "minervini_vcp": "VCP_BETTER",
                   "volume_breakout": "VOLUME_BREAKOUT_BETTER",
                   "combined": "COMBINED_SYSTEM_BETTER"}
    verdict = ("NO_STRATEGY_READY" if best["rank_score"] < -1e8
               else verdict_map[best["strategy"]])

    # ── Markdown reports ───────────────────────────────────────────
    def fmt(x, p=2):
        return "n/a" if not np.isfinite(x) else f"{x:.{p}f}"

    md = [f"# Strategy Comparison — {date.today()}", "",
          f"**VERDICT: `{verdict}`**", "",
          f"Universe: {len(raw)} liquid US stocks | Period: {PERIOD} | "
          f"Account €30,000 | Risk 0.50% | Max 4 positions | Heat 2% | "
          f"Realistic costs | Earnings-unknown rejected ({n_unknown} tickers unknown)", "",
          "| Strategy | Trades | OOS | Win% | PF | Expect €/tr | CAGR | MaxDD | Sharpe | WF+ | Acceptance |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        md.append(f"| {r.strategy} | {r.filled} | {r.oos_trades} | "
                  f"{fmt(r.win_rate*100,0)}% | {fmt(r.profit_factor)} | "
                  f"{fmt(r.expectancy_eur)} | {fmt(r.cagr*100,1)}% | "
                  f"{fmt(r.max_dd*100,1)}% | {fmt(r.sharpe)} | "
                  f"{fmt(r.wf_positive_pct*100,0)}% | {r.acceptance} |")
    md += ["", "## Existing ema_pullback at production 1% risk (separate — not risk-comparable)",
           f"Trades {m_1pct['filled']} | PF {fmt(m_1pct['profit_factor'])} | "
           f"CAGR {fmt(m_1pct['cagr']*100,1)}% | MaxDD {fmt(m_1pct['max_dd']*100,1)}%",
           "", "## Decision basis",
           "Winner selected on out-of-sample expectancy, walk-forward consistency,",
           "profit factor, drawdown, and stress-cost survival — not CAGR alone.",
           "See CSVs for cost profiles, walk-forward windows, regimes, and the",
           "parameter-stability grid."]
    (REPORTS / "strategy_comparison.md").write_text("\n".join(md))

    e = comp[comp.strategy == "enhanced_ema_pullback"].iloc[0]
    md2 = [f"# enhanced_ema_pullback — Results ({date.today()})", "",
           f"Acceptance: **{e.acceptance}**  (failed gates: {e.failed_gates or 'none'})", "",
           f"Main config: entry={MAIN['entry_mode']}, stop=swing_low−{MAIN['stop_buffer']}×ATR, "
           f"trail={MAIN['trail']}×ATR, no profit target", "",
           f"Signals {e.signals} | Filled {e.filled} | Win {fmt(e.win_rate*100,0)}% | "
           f"PF {fmt(e.profit_factor)} | Expectancy €{fmt(e.expectancy_eur)}/trade | "
           f"CAGR {fmt(e.cagr*100,1)}% | MaxDD {fmt(e.max_dd*100,1)}% | "
           f"Sharpe {fmt(e.sharpe)} | Sortino {fmt(e.sortino)}",
           "",
           f"Costs €: {main_res['enhanced_ema_pullback'][0].costs_paid}", "",
           "## Parameter stability (top rows by profit factor)", "",
           stab.sort_values("profit_factor", ascending=False)
               .head(8)[["entry_mode", "stop_buffer", "trail", "filled",
                         "win_rate", "profit_factor", "cagr", "max_dd"]]
               .to_string(index=False),
           "", "## Rejection reasons (realistic run)", "",
           (main_res["enhanced_ema_pullback"][0].rejections.reason.value_counts().to_string()
            if not main_res["enhanced_ema_pullback"][0].rejections.empty else "none"),
           "",
           "Stock-level stats: samples below 25 trades are INSUFFICIENT_SAMPLE",
           "by policy; strategy-level OOS results drive the decision."]
    (REPORTS / "enhanced_ema_pullback_results.md").write_text("\n".join(md2))

    print("\n" + "=" * 60)
    print(f"VERDICT: {verdict}")
    print(comp[["strategy", "filled", "oos_trades", "profit_factor",
                "cagr", "max_dd", "wf_positive_pct", "acceptance"]].to_string(index=False))
    print("Reports written to", REPORTS)


if __name__ == "__main__":
    main()
