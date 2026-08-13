"""
Forward signal journal — the missing piece.

Every strategy we have sits at INSUFFICIENT_SAMPLE. No backtest fixes
that; only forward time does. This module turns the daily scan into an
accumulating out-of-sample record:

  * every new signal is appended to reports/signal_journal.csv
  * every open signal is marked-to-market each day
  * the trailing-stop rule is applied exactly as the strategy specifies
    (stop ratchets up on the close, never down; a gap below the active
    stop exits at the open)
  * closed signals get a realised R-multiple

After a few months this is genuine forward evidence — the thing that
decides whether the backtests were real, with no effort from the trader.

Journal-only. It never places orders and never feeds the scanner.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

JOURNAL = Path(__file__).resolve().parent.parent / "reports" / "signal_journal.csv"

COLUMNS = [
    "signal_date", "strategy", "ticker", "entry", "init_stop", "trail_atr",
    "status", "active_stop", "high_close", "exit_date", "exit_price",
    "r_multiple", "last_price", "last_update",
]


# These columns hold text. An all-empty text column round-trips through CSV as
# float NaN, and writing a date string into it then raises a dtype error —
# so they are forced back to object dtype on load.
TEXT_COLUMNS = ["signal_date", "strategy", "ticker", "status",
                "exit_date", "last_update"]


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for c in TEXT_COLUMNS:
        df[c] = df[c].astype(object).where(df[c].notna(), "")
    return df


def load() -> pd.DataFrame:
    if JOURNAL.exists():
        df = pd.read_csv(JOURNAL)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        return _coerce(df[COLUMNS])
    return _coerce(pd.DataFrame(columns=COLUMNS))


def add_signals(journal: pd.DataFrame, rows: list[dict], today: date) -> pd.DataFrame:
    """Append signals not already open/recorded for this strategy+ticker."""
    new = []
    for r in rows:
        dup = journal[(journal.ticker == r["ticker"])
                      & (journal.strategy == r["strategy"])
                      & (journal.status == "OPEN")]
        if len(dup):
            continue
        new.append({
            "signal_date": today.isoformat(), "strategy": r["strategy"],
            "ticker": r["ticker"], "entry": r["entry"],
            "init_stop": r["init_stop"], "trail_atr": r.get("trail_atr", np.nan),
            "status": "OPEN", "active_stop": r["init_stop"],
            "high_close": r["entry"], "exit_date": "", "exit_price": np.nan,
            "r_multiple": np.nan, "last_price": r["entry"],
            "last_update": today.isoformat(),
        })
    if not new:
        return journal
    return _coerce(pd.concat([journal, pd.DataFrame(new)], ignore_index=True))


def update_open(journal: pd.DataFrame, prices: dict[str, pd.DataFrame],
                today: date) -> pd.DataFrame:
    """
    Mark open signals to market and apply the trailing rule.

    Sequencing matches the backtester: today's open/low are tested against
    the stop that was ACTIVE coming into today; only after the close is a
    new trail computed, and it becomes active tomorrow.
    """
    for i, row in journal[journal.status == "OPEN"].iterrows():
        df = prices.get(row.ticker)
        if df is None or df.empty:
            continue
        bar = df.iloc[-1]
        o, l, c = float(bar["open"]), float(bar["low"]), float(bar["close"])
        entry, stop = float(row.entry), float(row.active_stop)
        risk = entry - float(row.init_stop)
        if risk <= 0:
            continue

        exit_px = None
        if o <= stop:                       # gap through the stop
            exit_px = o
        elif l <= stop:
            exit_px = stop

        if exit_px is not None:
            journal.at[i, "status"] = "CLOSED"
            journal.at[i, "exit_date"] = today.isoformat()
            journal.at[i, "exit_price"] = round(exit_px, 4)
            journal.at[i, "r_multiple"] = round((exit_px - entry) / risk, 3)
        else:
            hi = max(float(row.high_close), c)
            journal.at[i, "high_close"] = round(hi, 4)
            trail = float(row.trail_atr) if np.isfinite(row.trail_atr) else 0.0
            if trail > 0:
                new_stop = hi - trail
                if new_stop > stop:         # ratchets up, never down
                    journal.at[i, "active_stop"] = round(new_stop, 4)
        journal.at[i, "last_price"] = round(c, 4)
        journal.at[i, "last_update"] = today.isoformat()
    return journal


def scorecard(journal: pd.DataFrame) -> dict:
    """Running forward record. R-multiples, so it is size-independent."""
    closed = journal[journal.status == "CLOSED"].copy()
    open_ = journal[journal.status == "OPEN"]
    out = {"total": len(journal), "open": len(open_), "closed": len(closed),
           "win_rate": np.nan, "avg_r": np.nan, "expectancy_r": np.nan,
           "best_r": np.nan, "worst_r": np.nan, "profit_factor": np.nan}
    if closed.empty:
        return out
    r = pd.to_numeric(closed.r_multiple, errors="coerce").dropna()
    if r.empty:
        return out
    wins, losses = r[r > 0], r[r <= 0]
    out["win_rate"] = len(wins) / len(r)
    out["avg_r"] = float(r.mean())
    out["expectancy_r"] = float(r.mean())
    out["best_r"] = float(r.max())
    out["worst_r"] = float(r.min())
    out["profit_factor"] = (float(wins.sum() / abs(losses.sum()))
                            if len(losses) and losses.sum() != 0 else np.inf)
    return out


def render(journal: pd.DataFrame, sc: dict) -> tuple[str, list[str]]:
    """HTML + markdown for the dashboard."""
    def f(x, p=2, s=""):
        return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}{s}"

    need = max(0, 20 - sc["closed"])
    progress = (f"{sc['closed']}/20 closed — {need} more before the forward record "
                f"means anything." if need else
                f"{sc['closed']} closed trades — the forward record is now readable.")

    open_rows = ""
    for _, r in journal[journal.status == "OPEN"].tail(12).iterrows():
        try:
            unreal = (float(r.last_price) - float(r.entry)) / (float(r.entry) - float(r.init_stop))
        except Exception:
            unreal = np.nan
        colour = "#3fb950" if np.isfinite(unreal) and unreal > 0 else "#f85149"
        open_rows += (f"<tr><td><strong>{r.ticker}</strong></td>"
                      f"<td style='font-size:11px;'>{r.strategy}</td>"
                      f"<td>{r.signal_date}</td><td>${float(r.entry):,.2f}</td>"
                      f"<td style='color:#f85149;'>${float(r.active_stop):,.2f}</td>"
                      f"<td>${float(r.last_price):,.2f}</td>"
                      f"<td style='color:{colour};'>{f(unreal)}R</td></tr>")
    if not open_rows:
        open_rows = ("<tr><td colspan='7' style='text-align:center;color:#8b949e;"
                     "padding:16px;'>No open tracked signals.</td></tr>")

    html = f"""
  <div class="section-title">Forward Signal Journal (automatic out-of-sample record)</div>
  <div class="card">
    <div style="color:#8b949e;font-size:12px;margin-bottom:10px;">
      {progress} &nbsp;|&nbsp; Tracked: <strong>{sc['total']}</strong> ·
      Open <strong>{sc['open']}</strong> · Closed <strong>{sc['closed']}</strong>
    </div>
    <table><thead><tr><th>Metric</th><th>Forward result</th></tr></thead><tbody>
      <tr><td>Win rate</td><td>{f(sc['win_rate']*100 if np.isfinite(sc['win_rate']) else np.nan,0,'%')}</td></tr>
      <tr><td>Expectancy</td><td>{f(sc['expectancy_r'])}R per trade</td></tr>
      <tr><td>Profit factor</td><td>{f(sc['profit_factor'])}</td></tr>
      <tr><td>Best / worst</td><td>{f(sc['best_r'])}R / {f(sc['worst_r'])}R</td></tr>
    </tbody></table>
    <div style="margin-top:14px;color:#8b949e;font-size:11px;">OPEN SIGNALS</div>
    <table><thead><tr><th>Stock</th><th>Strategy</th><th>Signalled</th><th>Entry</th>
      <th>Active stop</th><th>Last</th><th>Unrealised</th></tr></thead>
      <tbody>{open_rows}</tbody></table>
    <div class="legend">
      Results are in <strong>R-multiples</strong> (multiples of the initial risk), so they
      are independent of position size. The trailing stop is applied exactly as the
      strategy specifies — it rises on the close, never falls, and a gap below it exits
      at the open. This record is what decides whether the backtests were real.
    </div>
  </div>"""

    md = ["", "## Forward Signal Journal", "",
          f"{progress}  Tracked {sc['total']} · open {sc['open']} · closed {sc['closed']}", "",
          f"- Win rate: {f(sc['win_rate']*100 if np.isfinite(sc['win_rate']) else np.nan,0,'%')}",
          f"- Expectancy: {f(sc['expectancy_r'])}R per trade",
          f"- Profit factor: {f(sc['profit_factor'])}"]
    return html, md


def run(signal_rows: list[dict], prices: dict[str, pd.DataFrame],
        today: date) -> tuple[str, list[str]]:
    """Load, update, append, save, render. Safe to call every scan."""
    j = load()
    j = update_open(j, prices, today)
    j = add_signals(j, signal_rows, today)
    JOURNAL.parent.mkdir(exist_ok=True)
    j.to_csv(JOURNAL, index=False)
    return render(j, scorecard(j))
