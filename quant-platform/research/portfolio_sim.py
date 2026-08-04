"""
Research portfolio simulator — realistic, EUR-denominated, cost-aware.

Additive research code: the production backtester (backtester/engine.py) is
untouched. This simulator exists so ALL strategies (existing and new) can be
compared under identical, stricter execution assumptions:

  * signals fire on close, orders work the NEXT session (no same-close fills)
  * trigger (buy-stop) orders: reject if open gaps >2% above trigger
  * spread + slippage + commission + EUR/USD conversion costs (3 profiles)
  * gap-through-stop: open below stop exits at the OPEN with slippage
  * correct stop sequencing: today's exit uses YESTERDAY's active stop;
    the trailing stop is recomputed after the close and activates tomorrow;
    it may rise but never fall
  * portfolio constraints: max positions, total heat cap, position value cap,
    sector exposure cap, cash; rejected orders are recorded with reasons
  * market-regime gate with risk allocation scaling
  * earnings blackout; unknown earnings ⇒ EARNINGS_UNKNOWN rejection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd


# ── Cost profiles (fractions of notional, per side unless noted) ──────
COST_PROFILES = {
    "optimistic": dict(spread=0.0002, slip=0.0002, commission=0.0, fx=0.0005, gap_extra=0.0),
    "realistic":  dict(spread=0.0005, slip=0.0010, commission=1.0, fx=0.0015, gap_extra=0.0),
    "stress":     dict(spread=0.0010, slip=0.0030, commission=2.0, fx=0.0030, gap_extra=0.0020),
}

REGIME_ALLOC = {
    "BULL_STRONG": 1.00, "BULL_MODERATE": 0.75,
    "NEUTRAL": 0.375, "BEAR_MODERATE": 0.0, "BEAR_STRONG": 0.0,
}


@dataclass
class SimConfig:
    account_eur: float = 30_000.0
    risk_per_trade: float = 0.005
    max_positions: int = 4
    max_heat: float = 0.02
    max_position_value: float = 0.10
    max_sector_exposure: float = 0.25
    trail_atr_mult: float = 2.5
    costs: str = "realistic"
    trigger_valid_days: int = 3
    earnings_blackout_days: int = 14      # calendar ≈ 10 trading days
    require_regime_gate: bool = True
    rs_min: float = 80.0
    entry_type: str = "market"            # "market" (existing strats) | "stop" (enhanced)


@dataclass
class SimResult:
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity: pd.Series = field(default_factory=pd.Series)
    rejections: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_signals: int = 0
    costs_paid: dict = field(default_factory=dict)


def simulate(
    signal_table: dict[str, pd.DataFrame],
    price_data: dict[str, pd.DataFrame],
    regime: pd.Series,
    rs_rank: pd.DataFrame,
    sectors: dict[str, str],
    eurusd: pd.Series,
    earnings: dict[str, list],
    cfg: SimConfig,
    start: date | None = None,
    end: date | None = None,
) -> SimResult:
    """
    signal_table[tkr]: DataFrame indexed like price df with columns
        signal (bool), trigger (float), init_stop (float), atr (float)
    regime: daily regime label series (union calendar)
    rs_rank: DataFrame date x ticker of RS percentile (0-100)
    eurusd: daily EURUSD close (USD per EUR)
    earnings[tkr]: sorted list of earnings dates, or None = unknown
    """
    P = COST_PROFILES[cfg.costs]
    calendar = sorted(regime.index)
    if start:
        calendar = [d for d in calendar if d >= start]
    if end:
        calendar = [d for d in calendar if d <= end]
    if not calendar:
        return SimResult()

    def fx(day):
        try:
            v = eurusd.asof(pd.Timestamp(day))
            return float(v) if np.isfinite(v) else 1.08
        except Exception:
            return 1.08

    cash_eur = cfg.account_eur
    positions: dict[str, dict] = {}
    orders: list[dict] = []
    trades, rejects, equity = [], [], {}
    n_signals = 0
    costs_paid = {"commission": 0.0, "spread_slip": 0.0, "fx": 0.0, "gap_loss": 0.0}

    bars = {t: df for t, df in price_data.items()}
    bar_ix = {t: {d: i for i, d in enumerate(df.index.date)} for t, df in bars.items()}

    def bar(t, day):
        i = bar_ix[t].get(day)
        return None if i is None else bars[t].iloc[i]

    def equity_eur(day):
        val = cash_eur
        r = fx(day)
        for t, p in positions.items():
            b = bar(t, day)
            px = float(b["close"]) if b is not None else p["last_close"]
            val += p["shares"] * px / r
        return val

    def days_to_earnings(t, day):
        ev = earnings.get(t)
        if ev is None:
            return None                      # unknown
        future = [e for e in ev if e >= day]
        return (future[0] - day).days if future else 9999

    def close_position(t, p, day, px_usd, reason, gap_loss_usd=0.0):
        nonlocal cash_eur
        r = fx(day)
        notional = p["shares"] * px_usd
        comm = P["commission"]
        slip_cost = notional * (P["spread"] / 2 + P["slip"] * 0)  # slippage applied in px already
        fx_cost = notional * P["fx"] / 2
        proceeds_eur = (notional - comm - slip_cost - fx_cost) / r
        cash_eur += proceeds_eur
        costs_paid["commission"] += comm / r
        costs_paid["spread_slip"] += slip_cost / r
        costs_paid["fx"] += fx_cost / r
        costs_paid["gap_loss"] += gap_loss_usd * p["shares"] / r
        pnl_eur = proceeds_eur - p["cost_eur"]
        trades.append(dict(
            ticker=t, sector=sectors.get(t, "Unknown"),
            entry_date=p["entry_date"], exit_date=day,
            entry_px=p["entry_px"], exit_px=px_usd, shares=p["shares"],
            pnl_eur=pnl_eur, pnl_pct=pnl_eur / p["cost_eur"],
            hold_days=(day - p["entry_date"]).days,
            exit_reason=reason, regime_at_entry=p["regime"],
        ))

    for day in calendar:
        reg = regime.get(day, "NEUTRAL")
        alloc = REGIME_ALLOC.get(reg, 0.0)

        # ── 1) EXITS first — against YESTERDAY's active stop ───────
        for t in list(positions):
            p, b = positions[t], bar(t, day)
            if b is None:
                continue
            o, h, l, c = (float(b[k]) for k in ("open", "high", "low", "close"))
            stop = p["active_stop"]
            if o <= stop:  # gap through stop → out at open with slippage
                px = o * (1 - P["slip"] - P["gap_extra"])
                close_position(t, p, day, px, "GAP_STOP", gap_loss_usd=max(0.0, stop - px))
                del positions[t]
                continue
            if l <= stop:
                px = stop * (1 - P["slip"])
                close_position(t, p, day, px, "TRAILING_STOP")
                del positions[t]
                continue
            # after the close: ratchet trailing stop, active TOMORROW
            p["hi_close"] = max(p["hi_close"], c)
            atr_t = signal_table[t]["atr"].asof(pd.Timestamp(day))
            atr_t = float(atr_t) if np.isfinite(atr_t) else c * 0.02
            new_stop = p["hi_close"] - cfg.trail_atr_mult * atr_t
            p["pending_stop"] = max(p["active_stop"], new_stop)
            p["last_close"] = c

        # ── 2) FILL working orders at today's open/trigger ─────────
        still_working = []
        for od in orders:
            t = od["ticker"]
            b = bar(t, day)
            if b is None:
                if od["expires"] >= day:
                    still_working.append(od)
                continue
            o, h = float(b["open"]), float(b["high"])
            trig = od["trigger"]
            fill = None
            if cfg.entry_type == "stop":
                if o > trig * 1.02:
                    rejects.append(dict(date=day, ticker=t, reason="OPEN_GAP_ABOVE_TRIGGER"))
                elif o >= trig:
                    fill = o
                elif h >= trig:
                    fill = trig
                elif od["expires"] >= day:
                    still_working.append(od)
                    continue
            else:  # market-on-open
                fill = o

            if fill is None:
                continue
            fill *= (1 + P["spread"] / 2 + P["slip"])

            # portfolio gates at fill time
            eq = equity_eur(day)
            r = fx(day)
            stop = od["init_stop"]
            if stop >= fill or (fill - stop) / fill > 0.07:
                rejects.append(dict(date=day, ticker=t, reason="STOP_TOO_WIDE"))
                continue
            reg_now = regime.get(day, "NEUTRAL")
            if REGIME_ALLOC.get(reg_now, 0) == 0 and cfg.require_regime_gate:
                rejects.append(dict(date=day, ticker=t, reason="REGIME_BEAR"))
                continue
            if len(positions) >= cfg.max_positions or t in positions:
                rejects.append(dict(date=day, ticker=t, reason="MAX_POSITIONS"))
                continue
            open_heat = sum(
                (pp["entry_px"] - pp["active_stop"]) * pp["shares"] / r / eq
                for pp in positions.values() if pp["active_stop"] < pp["entry_px"]
            )
            risk_eur = eq * cfg.risk_per_trade * REGIME_ALLOC.get(reg_now, 0)
            if open_heat + risk_eur / eq > cfg.max_heat:
                rejects.append(dict(date=day, ticker=t, reason="HEAT_CAP"))
                continue
            sector = sectors.get(t, "Unknown")
            sector_val = sum(
                pp["shares"] * pp["last_close"] / r
                for tt, pp in positions.items() if sectors.get(tt) == sector
            )
            shares = int(risk_eur * r / (fill - stop))
            pos_val_eur = shares * fill / r
            if pos_val_eur > eq * cfg.max_position_value:
                shares = int(eq * cfg.max_position_value * r / fill)
                pos_val_eur = shares * fill / r
            if shares < 1:
                rejects.append(dict(date=day, ticker=t, reason="TOO_SMALL"))
                continue
            if (sector_val + pos_val_eur) / eq > cfg.max_sector_exposure:
                rejects.append(dict(date=day, ticker=t, reason="SECTOR_CAP"))
                continue
            comm = P["commission"]
            fx_cost = shares * fill * P["fx"] / 2
            cost_eur = (shares * fill + comm + fx_cost) / r
            if cost_eur > cash_eur:
                rejects.append(dict(date=day, ticker=t, reason="INSUFFICIENT_CASH"))
                continue
            cash_eur -= cost_eur
            costs_paid["commission"] += comm / r
            costs_paid["fx"] += fx_cost / r
            costs_paid["spread_slip"] += shares * fill * (P["spread"] / 2 + P["slip"]) / r
            positions[t] = dict(
                shares=shares, entry_px=fill, entry_date=day, cost_eur=cost_eur,
                active_stop=stop, pending_stop=stop,
                hi_close=fill, last_close=fill, regime=reg_now,
            )
        orders = still_working

        # ── 3) NEW signals from today's close → orders for tomorrow ─
        for t, tab in signal_table.items():
            i = bar_ix[t].get(day)
            if i is None:
                continue
            row = tab.iloc[i]
            if not bool(row["signal"]):
                continue
            n_signals += 1
            if cfg.require_regime_gate and alloc == 0:
                rejects.append(dict(date=day, ticker=t, reason="REGIME_BEAR"))
                continue
            if cfg.rs_min > 0:
                try:
                    rs = float(rs_rank.loc[pd.Timestamp(day), t])
                except Exception:
                    rs = np.nan
                if not np.isfinite(rs) or rs < cfg.rs_min:
                    rejects.append(dict(date=day, ticker=t, reason="RS_LOW"))
                    continue
            dte = days_to_earnings(t, day)
            if dte is None:
                rejects.append(dict(date=day, ticker=t, reason="EARNINGS_UNKNOWN"))
                continue
            if dte <= cfg.earnings_blackout_days:
                rejects.append(dict(date=day, ticker=t, reason="EARNINGS_SOON"))
                continue
            orders.append(dict(
                ticker=t, trigger=float(row["trigger"]),
                init_stop=float(row["init_stop"]),
                expires=day + timedelta(days=cfg.trigger_valid_days * 2),
            ))

        # ── 4) activate pending stops for tomorrow; mark equity ────
        for p in positions.values():
            p["active_stop"] = max(p["active_stop"], p.get("pending_stop", p["active_stop"]))
        equity[day] = equity_eur(day)

    # close remaining at last close
    last_day = calendar[-1]
    for t, p in list(positions.items()):
        close_position(t, p, last_day, p["last_close"], "END_OF_BACKTEST")

    return SimResult(
        trades=pd.DataFrame(trades),
        equity=pd.Series(equity).sort_index(),
        rejections=pd.DataFrame(rejects),
        n_signals=n_signals,
        costs_paid=costs_paid,
    )


def metrics_from(res: SimResult, initial_eur: float = 30_000.0) -> dict:
    """Strategy-level metrics per the comparison spec."""
    tr, eq = res.trades, res.equity
    out = dict(signals=res.n_signals, filled=0, win_rate=np.nan, avg_win=np.nan,
               avg_loss=np.nan, wl_ratio=np.nan, expectancy_eur=np.nan,
               profit_factor=np.nan, cagr=np.nan, max_dd=np.nan, sharpe=np.nan,
               sortino=np.nan, exposure=np.nan, turnover=np.nan,
               longest_loss_streak=0, verdict="NO_TRADE",
               **{f"cost_{k}": round(v, 2) for k, v in res.costs_paid.items()})
    if tr.empty or len(eq) < 2:
        return out
    out["filled"] = len(tr)
    wins, losses = tr[tr.pnl_eur > 0], tr[tr.pnl_eur <= 0]
    out["win_rate"] = len(wins) / len(tr)
    out["avg_win"] = wins.pnl_eur.mean() if len(wins) else 0.0
    out["avg_loss"] = losses.pnl_eur.mean() if len(losses) else 0.0
    out["wl_ratio"] = abs(out["avg_win"] / out["avg_loss"]) if out["avg_loss"] else np.inf
    out["expectancy_eur"] = tr.pnl_eur.mean()
    gross_w, gross_l = wins.pnl_eur.sum(), abs(losses.pnl_eur.sum())
    out["profit_factor"] = gross_w / gross_l if gross_l > 0 else np.inf
    yrs = max(len(eq) / 252, 1e-9)
    out["cagr"] = (eq.iloc[-1] / initial_eur) ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    out["max_dd"] = dd.min()
    ret = eq.pct_change().dropna()
    if ret.std() > 0:
        out["sharpe"] = (ret.mean() * 252 - 0.02) / (ret.std() * np.sqrt(252))
        downside = ret[ret < 0].std()
        out["sortino"] = (ret.mean() * 252 - 0.02) / (downside * np.sqrt(252)) if downside > 0 else np.inf
    # exposure: fraction of days with a position on
    days_in = pd.Series(0, index=eq.index)
    for _, t in tr.iterrows():
        days_in.loc[(days_in.index >= t.entry_date) & (days_in.index <= t.exit_date)] = 1
    out["exposure"] = days_in.mean()
    out["turnover"] = (tr.shares * tr.entry_px).sum() / initial_eur / yrs
    streak = best = 0
    for pnl in tr.sort_values("exit_date").pnl_eur:
        streak = streak + 1 if pnl <= 0 else 0
        best = max(best, streak)
    out["longest_loss_streak"] = best
    out["verdict"] = ""
    return out
