"""
dual_momentum_rotation — monthly-rebalance backtest.

Separate engine because this is an allocation strategy, not a trade
picker: no stops, monthly decisions, weights instead of positions.
Costs, EUR/USD conversion and next-session execution are modelled the
same way as the trade-based simulator so results stay comparable.

Usage:
    python research/run_dual_momentum.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.logging import setup_logging
from data.pipeline import get_data, get_data_batch
from strategies.dual_momentum import (
    full_universe, blended_momentum, month_end_dates, select,
    CASH_PROXY, TOP_N,
)
from research.portfolio_sim import COST_PROFILES

setup_logging("WARNING")
REPORTS = Path(__file__).resolve().parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)

PERIOD = "10y"
ACCOUNT_EUR = 30_000.0
HOLDOUT_YEARS = 2


def run(closes: pd.DataFrame, eurusd: pd.Series, costs="realistic",
        top_n=TOP_N, lookbacks=(63, 126, 252),
        start=None, end=None) -> dict:
    """Monthly rebalance. Returns equity curve, trade log, weights history."""
    P = COST_PROFILES[costs]
    mom = blended_momentum(closes, lookbacks)
    idx = closes.index
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    if len(idx) < 300:
        return {}

    rebal = [d for d in month_end_dates(idx) if d in set(idx)]
    daily_ret = closes.pct_change().fillna(0.0)

    equity = ACCOUNT_EUR
    curve, weights_hist, trades = {}, [], []
    current: dict[str, float] = {}

    for i, d in enumerate(idx):
        # apply today's return to yesterday's weights
        if current:
            r = sum(w * float(daily_ret.loc[d, t]) for t, w in current.items()
                    if t in daily_ret.columns and np.isfinite(daily_ret.loc[d, t]))
            equity *= (1 + r)
        curve[d.date()] = equity

        # decide on month end, execute NEXT session (no same-close fill)
        if d in rebal and i + 1 < len(idx):
            row = mom.loc[d]
            cash_m = float(row.get(CASH_PROXY, 0.0)) if CASH_PROXY in row.index else 0.0
            target = select(row, cash_m, top_n)

            turnover = sum(abs(target.get(t, 0) - current.get(t, 0))
                           for t in set(target) | set(current))
            if turnover > 0:
                # one-way cost on the traded fraction
                cost_rate = P["spread"] / 2 + P["slip"] + P["fx"] / 2
                equity *= (1 - turnover * cost_rate)
                for t in sorted(set(target) | set(current)):
                    if abs(target.get(t, 0) - current.get(t, 0)) > 1e-9:
                        trades.append(dict(
                            date=idx[i + 1].date(), ticker=t,
                            from_w=round(current.get(t, 0), 4),
                            to_w=round(target.get(t, 0), 4)))
            current = target
            weights_hist.append(dict(date=d.date(), **{k: round(v, 3)
                                                       for k, v in target.items()}))

    eq = pd.Series(curve).sort_index()
    return dict(equity=eq, trades=pd.DataFrame(trades),
                weights=pd.DataFrame(weights_hist))


def metrics(eq: pd.Series, trades: pd.DataFrame) -> dict:
    if eq is None or len(eq) < 2:
        return {}
    yrs = len(eq) / 252
    ret = eq.pct_change().dropna()
    dd = eq / eq.cummax() - 1
    monthly = eq.resample("ME").last().pct_change().dropna() if isinstance(
        eq.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    if not isinstance(eq.index, pd.DatetimeIndex):
        tmp = eq.copy()
        tmp.index = pd.to_datetime(tmp.index)
        monthly = tmp.resample("ME").last().pct_change().dropna()
    down = ret[ret < 0].std()
    return dict(
        final_eur=round(float(eq.iloc[-1]), 0),
        cagr=(float(eq.iloc[-1]) / ACCOUNT_EUR) ** (1 / max(yrs, 1e-9)) - 1,
        max_dd=float(dd.min()),
        sharpe=float((ret.mean() * 252 - 0.02) / (ret.std() * np.sqrt(252)))
        if ret.std() > 0 else np.nan,
        sortino=float((ret.mean() * 252 - 0.02) / (down * np.sqrt(252)))
        if down and down > 0 else np.nan,
        positive_months=float((monthly > 0).mean()) if len(monthly) else np.nan,
        n_rebalance_trades=int(len(trades)),
        trades_per_year=round(len(trades) / max(yrs, 1e-9), 1),
    )


def main():
    tickers = full_universe()
    print(f"Fetching {len(tickers)} ETFs, {PERIOD}...")
    raw = get_data_batch(tickers, period=PERIOD, show_progress=True)
    raw = {t: df for t, df in raw.items() if df is not None and len(df) > 300}
    closes = pd.DataFrame({t: df["close"] for t, df in raw.items()}).dropna(how="all")
    print(f"  usable: {list(closes.columns)}")

    fxdf = get_data("EURUSD=X", period=PERIOD)
    eurusd = fxdf["close"] if fxdf is not None else pd.Series(1.08, index=closes.index)
    spy = closes["SPY"] if "SPY" in closes else None

    # ── Main run + cost profiles ───────────────────────────────────
    rows, main_run = [], None
    for costs in COST_PROFILES:
        r = run(closes, eurusd, costs=costs)
        if not r:
            continue
        m = metrics(r["equity"], r["trades"])
        rows.append({"costs": costs, **m})
        if costs == "realistic":
            main_run = r
    pd.DataFrame(rows).to_csv(REPORTS / "rotation_cost_comparison.csv", index=False)

    # ── Parameter stability ────────────────────────────────────────
    stab = []
    for top_n in (2, 3, 4, 5):
        for lb in [(63, 126, 252), (126, 252, 252), (21, 63, 126)]:
            r = run(closes, eurusd, top_n=top_n, lookbacks=lb)
            if r:
                stab.append({"top_n": top_n, "lookbacks": str(lb),
                             **metrics(r["equity"], r["trades"])})
    pd.DataFrame(stab).to_csv(REPORTS / "rotation_parameter_stability.csv", index=False)

    # ── Holdout (last N years unseen) ──────────────────────────────
    split = closes.index[-1] - pd.DateOffset(years=HOLDOUT_YEARS)
    r_is = run(closes, eurusd, end=split)
    r_oos = run(closes, eurusd, start=split)
    m_is = metrics(r_is["equity"], r_is["trades"]) if r_is else {}
    m_oos = metrics(r_oos["equity"], r_oos["trades"]) if r_oos else {}

    main_m = metrics(main_run["equity"], main_run["trades"])
    main_run["trades"].to_csv(REPORTS / "rotation_trade_log.csv", index=False)
    main_run["weights"].to_csv(REPORTS / "rotation_weights_history.csv", index=False)
    main_run["equity"].to_csv(REPORTS / "rotation_equity_curve.csv")

    # ── Buy & hold benchmark ───────────────────────────────────────
    bh = {}
    if spy is not None:
        s = spy.loc[main_run["equity"].index[0].__str__():] if False else spy
        s = s[s.index >= pd.Timestamp(main_run["equity"].index[0])]
        eq_bh = ACCOUNT_EUR * (s / s.iloc[0])
        bh = metrics(eq_bh, pd.DataFrame())

    def f(x, p=2, s=""):
        return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}{s}"

    md = [f"# dual_momentum_rotation — Backtest ({date.today()})", "",
          f"Universe: {len(closes.columns)} liquid ETFs | Period {PERIOD} | "
          f"Account €{ACCOUNT_EUR:,.0f} | Monthly rebalance | Top {TOP_N} equal weight | "
          f"Absolute-momentum filter vs {CASH_PROXY}", "",
          "## Headline (realistic costs)", "",
          "| Metric | Rotation | SPY buy & hold |",
          "|---|---|---|",
          f"| CAGR | {f(main_m.get('cagr',np.nan)*100,2,'%')} | {f(bh.get('cagr',np.nan)*100,2,'%')} |",
          f"| Max drawdown | {f(main_m.get('max_dd',np.nan)*100,1,'%')} | {f(bh.get('max_dd',np.nan)*100,1,'%')} |",
          f"| Sharpe | {f(main_m.get('sharpe',np.nan))} | {f(bh.get('sharpe',np.nan))} |",
          f"| Sortino | {f(main_m.get('sortino',np.nan))} | {f(bh.get('sortino',np.nan))} |",
          f"| Positive months | {f(main_m.get('positive_months',np.nan)*100,0,'%')} | {f(bh.get('positive_months',np.nan)*100,0,'%')} |",
          f"| Final equity | €{f(main_m.get('final_eur',np.nan),0)} | €{f(bh.get('final_eur',np.nan),0)} |",
          "",
          f"**Turnover: {main_m.get('trades_per_year','n/a')} rebalance trades per year** — "
          f"the entire point. At ~€18/round-trip friction, a strategy trading "
          f"~12 times a year loses ~€216/yr to costs; the swing book at ~100 "
          f"trades/yr loses ~€1,800.", "",
          "## Cost sensitivity", "",
          pd.DataFrame(rows)[["costs", "cagr", "max_dd", "sharpe", "trades_per_year"]]
            .to_string(index=False), "",
          "## Holdout (last %d years unseen)" % HOLDOUT_YEARS, "",
          f"In-sample CAGR {f(m_is.get('cagr',np.nan)*100,2,'%')} | "
          f"MaxDD {f(m_is.get('max_dd',np.nan)*100,1,'%')}",
          f"Holdout  CAGR {f(m_oos.get('cagr',np.nan)*100,2,'%')} | "
          f"MaxDD {f(m_oos.get('max_dd',np.nan)*100,1,'%')}", "",
          "## Parameter stability", "",
          pd.DataFrame(stab)[["top_n", "lookbacks", "cagr", "max_dd", "sharpe",
                              "trades_per_year"]].to_string(index=False),
          "", "_Research only. Production scanner unchanged._"]
    (REPORTS / "rotation_results.md").write_text("\n".join(md))

    print("\n" + "=" * 60)
    print(f"Rotation  CAGR {f(main_m.get('cagr',np.nan)*100,2,'%')} | "
          f"MaxDD {f(main_m.get('max_dd',np.nan)*100,1,'%')} | "
          f"Sharpe {f(main_m.get('sharpe',np.nan))} | "
          f"{main_m.get('trades_per_year')} trades/yr")
    print(f"SPY B&H   CAGR {f(bh.get('cagr',np.nan)*100,2,'%')} | "
          f"MaxDD {f(bh.get('max_dd',np.nan)*100,1,'%')} | "
          f"Sharpe {f(bh.get('sharpe',np.nan))}")
    print(f"Holdout   CAGR {f(m_oos.get('cagr',np.nan)*100,2,'%')}")


if __name__ == "__main__":
    main()
