"""
institutional_momentum_trend — separate backtest + comparison.

Runs the new strategy through the SAME research simulator, universe,
period, costs, FX, and portfolio limits as the existing strategies, then
compares all five and applies the acceptance standard.

Nothing here touches the production scanner or the existing dashboard.

Usage:
    python research/run_institutional_backtest.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_config
from core.logging import setup_logging
from data.pipeline import get_data, get_data_batch
from indicators.composite import enrich
from strategies.registry import get_strategy
from strategies.institutional_momentum import compute_setup_table as inst_table
from strategies.enhanced_ema import compute_setup_table as enh_table
from research.portfolio_sim import simulate, metrics_from, SimConfig, COST_PROFILES
from research.run_comparison import (
    SECTORS, UNIVERSE, build_regime, build_rs_rank, fetch_earnings,
    existing_table, combined_table,
)

setup_logging("WARNING")
REPORTS = Path(__file__).resolve().parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)

PERIOD = "6y"
HOLDOUT_MONTHS = 6
MAIN = dict(entry_mode="reclaim", stop_buffer=0.25, trail=3.0, rs_min=80.0)


def signal_frame(df, tab):
    return pd.DataFrame(index=df.index, data={
        "signal": tab["setup_ok"], "trigger": tab["trigger"],
        "init_stop": tab["init_stop"], "atr": tab["atr"],
    })


def wf_windows(calendar, is_years=2.0, oos_months=6):
    start, end = calendar[0], calendar[-1]
    holdout = end - timedelta(days=30 * HOLDOUT_MONTHS)
    wins, cursor = [], start
    while True:
        oos_s = cursor + timedelta(days=int(is_years * 365))
        oos_e = min(oos_s + timedelta(days=oos_months * 30), holdout)
        if oos_s >= holdout:
            break
        wins.append((oos_s, oos_e))
        cursor = oos_s
    wins.append((holdout, end))
    return wins


def main():
    cfg = get_config()
    print("Fetching data...")
    raw = get_data_batch(UNIVERSE, period=PERIOD, show_progress=True)
    raw = {t: df for t, df in raw.items() if df is not None and len(df) > 300}
    spy, qqq = get_data("SPY", period=PERIOD), get_data("QQQ", period=PERIOD)
    fxdf = get_data("EURUSD=X", period=PERIOD)
    eurusd = fxdf["close"] if fxdf is not None else pd.Series(1.08, index=spy.index)
    vixdf = get_data("^VIX", period=PERIOD)
    vix = vixdf["close"] if vixdf is not None else None
    print(f"  VIX: {'loaded' if vix is not None else 'unavailable - vol scaling inert'}")

    regime = build_regime(spy, qqq, raw)
    rs = build_rs_rank(raw)
    earnings = fetch_earnings(list(raw))
    print("Enriching...")
    enriched = {t: enrich(df, patterns=True) for t, df in raw.items()}

    print("Building signal tables...")
    tabs = {}
    tabs["institutional_momentum_trend"] = {
        t: signal_frame(df, inst_table(df, MAIN["stop_buffer"], MAIN["entry_mode"]))
        for t, df in raw.items()}
    tabs["enhanced_ema_pullback"] = {
        t: signal_frame(df, enh_table(df, 0.25, "reclaim")) for t, df in raw.items()}
    for name in ["ema_pullback", "minervini_vcp", "volume_breakout"]:
        strat = get_strategy(name)
        tabs[name] = {t: existing_table(df, strat) for t, df in enriched.items()}
        print(f"  {name}: done")
    tabs["combined"] = combined_table(
        [tabs["ema_pullback"], tabs["minervini_vcp"], tabs["volume_breakout"]])

    args = dict(price_data=raw, regime=regime, rs_rank=rs, sectors=SECTORS,
                eurusd=eurusd, earnings=earnings, vix=vix)
    names = ["ema_pullback", "minervini_vcp", "volume_breakout",
             "combined", "enhanced_ema_pullback", "institutional_momentum_trend"]

    def cfg_for(name, costs="realistic", trail=None, rs_min=None, risk=0.005):
        is_new = name == "institutional_momentum_trend"
        return SimConfig(
            costs=costs, risk_per_trade=risk,
            trail_atr_mult=trail or (MAIN["trail"] if is_new else 2.5),
            max_positions=4, max_heat=0.02,
            max_position_value=0.10, max_sector_exposure=0.25,
            entry_type="stop" if name in ("enhanced_ema_pullback",
                                          "institutional_momentum_trend") else "market",
            rs_min=(rs_min if rs_min is not None
                    else (MAIN["rs_min"] if is_new else
                          80.0 if name == "enhanced_ema_pullback" else 0.0)),
        )

    def run(name, c, start=None, end=None):
        return simulate(tabs[name], cfg=c, start=start, end=end, **args)

    # ── Cost profiles ──────────────────────────────────────────────
    print("Cost profiles...")
    cost_rows, main_res = [], {}
    for name in names:
        for costs in COST_PROFILES:
            res = run(name, cfg_for(name, costs))
            m = metrics_from(res)
            cost_rows.append({"strategy": name, "costs": costs, **m})
            if costs == "realistic":
                main_res[name] = (res, m)
        print(f"  {name}: {main_res[name][1]['filled']} trades")
    pd.DataFrame(cost_rows).to_csv(REPORTS / "institutional_cost_comparison.csv", index=False)

    # ── Walk-forward ───────────────────────────────────────────────
    print("Walk-forward...")
    calendar = sorted(regime.index)
    wf_rows = []
    for name in names:
        for (s, e) in wf_windows(calendar):
            res = run(name, cfg_for(name), start=s, end=e)
            m = metrics_from(res)
            pnl = res.trades.pnl_eur.sum() if not res.trades.empty else 0.0
            wf_rows.append({"strategy": name, "oos_start": s, "oos_end": e,
                            "trades": m["filled"], "pnl_eur": pnl,
                            "profit_factor": m["profit_factor"],
                            "result": ("NO_TRADE" if m["filled"] == 0 else
                                       "POSITIVE" if pnl > 0 else "NEGATIVE")})
    wf = pd.DataFrame(wf_rows)
    wf.to_csv(REPORTS / "institutional_walk_forward.csv", index=False)

    # ── Parameter stability (new strategy only) ────────────────────
    print("Parameter stability...")
    stab = []
    for em in ["reclaim", "reversal"]:
        for sb in [0.0, 0.25, 0.5]:
            grid = {t: signal_frame(df, inst_table(df, sb, em)) for t, df in raw.items()}
            tabs["_grid"] = grid
            for trail in [2.5, 3.0, 3.5]:
                for rs_min in [70, 75, 80, 85, 90]:
                    res = simulate(grid, cfg=cfg_for("institutional_momentum_trend",
                                                     trail=trail, rs_min=rs_min), **args)
                    m = metrics_from(res)
                    stab.append({"entry_mode": em, "stop_buffer": sb, "trail": trail,
                                 "rs_min": rs_min, **m})
    pd.DataFrame(stab).to_csv(REPORTS / "institutional_parameter_stability.csv", index=False)

    # ── Trade log + breakdowns ─────────────────────────────────────
    inst_res, inst_m = main_res["institutional_momentum_trend"]
    inst_res.trades.to_csv(REPORTS / "institutional_trade_log.csv", index=False)

    reg_rows, sec_rows, yr_rows = [], [], []
    for name in names:
        tr = main_res[name][0].trades
        if tr.empty:
            continue
        for reg, g in tr.groupby("regime_at_entry"):
            reg_rows.append({"strategy": name, "regime": reg, "trades": len(g),
                             "pnl_eur": g.pnl_eur.sum(), "win_rate": (g.pnl_eur > 0).mean()})
        for sec, g in tr.groupby("sector"):
            sec_rows.append({"strategy": name, "sector": sec, "trades": len(g),
                             "pnl_eur": g.pnl_eur.sum()})
        tr = tr.copy()
        tr["year"] = pd.to_datetime(tr.exit_date).dt.year
        for yr, g in tr.groupby("year"):
            yr_rows.append({"strategy": name, "year": int(yr), "trades": len(g),
                            "pnl_eur": g.pnl_eur.sum()})
    pd.DataFrame(reg_rows).to_csv(REPORTS / "institutional_regime_performance.csv", index=False)
    pd.DataFrame(sec_rows).to_csv(REPORTS / "institutional_sector_performance.csv", index=False)
    pd.DataFrame(yr_rows).to_csv(REPORTS / "institutional_yearly_performance.csv", index=False)

    # ── Acceptance + comparison ────────────────────────────────────
    comp = []
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
        stress = next(r for r in cost_rows if r["strategy"] == name and r["costs"] == "stress")
        gates = dict(
            oos_trades=oos_trades >= 200,
            expectancy=bool(m["expectancy_eur"] > 0) if np.isfinite(m["expectancy_eur"]) else False,
            pf=bool(m["profit_factor"] >= 1.25) if np.isfinite(m["profit_factor"]) else False,
            wf=pos_pct >= 0.65,
            dd=bool(m["max_dd"] >= -0.12) if np.isfinite(m["max_dd"]) else False,
            stress=bool(stress["expectancy_eur"] > 0) if np.isfinite(stress["expectancy_eur"]) else False,
            concentration=top_stock <= 0.15 and top_sector <= 0.25,
        )
        if m["filled"] == 0:
            label = "NO_TRADE"
        elif all(gates.values()):
            label = "PAPER_ELIGIBLE"
        elif oos_trades < 200:
            label = "INSUFFICIENT_SAMPLE"
        elif gates["expectancy"] and gates["pf"]:
            label = "RESEARCH_ONLY"
        else:
            label = "REJECTED"
        comp.append({"strategy": name, **m, "oos_trades": oos_trades,
                     "wf_positive_pct": round(pos_pct, 2),
                     "top_stock_share": round(top_stock, 2),
                     "top_sector_share": round(top_sector, 2),
                     "stress_expectancy": stress["expectancy_eur"],
                     "acceptance": label,
                     "failed_gates": ",".join(k for k, v in gates.items() if not v)})
    comp = pd.DataFrame(comp)
    comp.to_csv(REPORTS / "institutional_strategy_comparison.csv", index=False)

    # ── Verdict: existing best vs institutional ────────────────────
    def f(x, p=2):
        return "n/a" if not np.isfinite(x) else f"{x:.{p}f}"

    existing = comp[comp.strategy.isin(
        ["ema_pullback", "minervini_vcp", "volume_breakout", "combined"])]
    best_ex = existing.sort_values("expectancy_eur", ascending=False).iloc[0]
    inst = comp[comp.strategy == "institutional_momentum_trend"].iloc[0]

    wins = {
        "expectancy": inst.expectancy_eur > best_ex.expectancy_eur,
        "profit_factor": inst.profit_factor > best_ex.profit_factor,
        "cagr": inst.cagr > best_ex.cagr,
        "drawdown": inst.max_dd > best_ex.max_dd,          # less negative = better
        "walk_forward": inst.wf_positive_pct > best_ex.wf_positive_pct,
        "stress": inst.stress_expectancy > best_ex.stress_expectancy,
    }
    n_win = sum(wins.values())
    if inst.acceptance == "NO_TRADE":
        verdict = "NO_STRATEGY_READY"
    elif inst.acceptance == "PAPER_ELIGIBLE" and n_win >= 4:
        verdict = "INSTITUTIONAL_MOMENTUM_BETTER"
    elif n_win >= 4:
        verdict = "INSTITUTIONAL_MOMENTUM_PROMISING_BUT_UNPROVEN"
    elif n_win >= 2:
        verdict = "BOTH_COMPLEMENT_EACH_OTHER"
    elif best_ex.acceptance in ("PAPER_ELIGIBLE", "RESEARCH_ONLY"):
        verdict = "EXISTING_STRATEGY_BETTER"
    else:
        verdict = "NO_STRATEGY_READY"

    md = [f"# institutional_momentum_trend — Backtest & Comparison ({date.today()})", "",
          f"**VERDICT: `{verdict}`**", "",
          f"Universe {len(raw)} | Period {PERIOD} | Account €30,000 | Risk 0.50% | "
          f"Max 4 positions | Heat 2% | Sector 25% | Realistic costs | "
          f"Config: {MAIN['entry_mode']} entry, {MAIN['stop_buffer']}×ATR stop buffer, "
          f"{MAIN['trail']}×ATR trail, RS≥{MAIN['rs_min']:.0f}", "",
          "| Strategy | Trades | OOS | Win% | PF | Expect € | CAGR | MaxDD | Sharpe | Sortino | WF+ | Acceptance |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        md.append(f"| {r.strategy} | {r.filled} | {r.oos_trades} | {f(r.win_rate*100,0)}% | "
                  f"{f(r.profit_factor)} | {f(r.expectancy_eur)} | {f(r.cagr*100,1)}% | "
                  f"{f(r.max_dd*100,1)}% | {f(r.sharpe)} | {f(r.sortino)} | "
                  f"{f(r.wf_positive_pct*100,0)}% | {r.acceptance} |")

    md += ["", "## Head-to-head: best existing vs institutional", "",
           f"Best existing by expectancy: **{best_ex.strategy}**", "",
           "| Metric | " + str(best_ex.strategy) + " | institutional_momentum_trend | Winner |",
           "|---|---|---|---|",
           f"| Expectancy €/trade | {f(best_ex.expectancy_eur)} | {f(inst.expectancy_eur)} | "
           f"{'INST' if wins['expectancy'] else 'EXISTING'} |",
           f"| Profit factor | {f(best_ex.profit_factor)} | {f(inst.profit_factor)} | "
           f"{'INST' if wins['profit_factor'] else 'EXISTING'} |",
           f"| CAGR | {f(best_ex.cagr*100,1)}% | {f(inst.cagr*100,1)}% | "
           f"{'INST' if wins['cagr'] else 'EXISTING'} |",
           f"| Max drawdown | {f(best_ex.max_dd*100,1)}% | {f(inst.max_dd*100,1)}% | "
           f"{'INST' if wins['drawdown'] else 'EXISTING'} |",
           f"| Walk-forward + | {f(best_ex.wf_positive_pct*100,0)}% | {f(inst.wf_positive_pct*100,0)}% | "
           f"{'INST' if wins['walk_forward'] else 'EXISTING'} |",
           f"| Stress expectancy € | {f(best_ex.stress_expectancy)} | {f(inst.stress_expectancy)} | "
           f"{'INST' if wins['stress'] else 'EXISTING'} |",
           "", f"Institutional wins {n_win}/6 comparisons.", "",
           "## Acceptance detail — institutional_momentum_trend", "",
           f"Status: **{inst.acceptance}**  |  failed gates: {inst.failed_gates or 'none'}", "",
           f"- OOS trades: {inst.oos_trades} (need ≥200)",
           f"- Profit factor: {f(inst.profit_factor)} (need ≥1.25)",
           f"- Expectancy: €{f(inst.expectancy_eur)}/trade (need >0)",
           f"- Walk-forward positive: {f(inst.wf_positive_pct*100,0)}% (need ≥65%)",
           f"- Max drawdown: {f(inst.max_dd*100,1)}% (need ≥ −12%)",
           f"- Stress expectancy: €{f(inst.stress_expectancy)} (need >0)",
           f"- Largest stock share of profit: {f(inst.top_stock_share*100,0)}% (need ≤15%)",
           f"- Largest sector share of profit: {f(inst.top_sector_share*100,0)}% (need ≤25%)",
           "", "## Parameter stability (top by profit factor)", "",
           pd.DataFrame(stab).sort_values("profit_factor", ascending=False)
             .head(10)[["entry_mode", "stop_buffer", "trail", "rs_min",
                        "filled", "win_rate", "profit_factor", "cagr", "max_dd"]]
             .to_string(index=False),
           "", "_Research only. Production scanner unchanged._"]
    (REPORTS / "institutional_momentum_results.md").write_text("\n".join(md))

    print("\n" + "=" * 60)
    print(f"VERDICT: {verdict}   (institutional wins {n_win}/6)")
    print(comp[["strategy", "filled", "oos_trades", "profit_factor",
                "cagr", "max_dd", "wf_positive_pct", "acceptance"]].to_string(index=False))


if __name__ == "__main__":
    main()
