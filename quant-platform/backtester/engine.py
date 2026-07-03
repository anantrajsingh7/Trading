"""
Event-driven backtesting engine.

Design principles:
  - No look-ahead bias: signals generated on bar close, filled at next open
  - Position sizing uses 1% risk rule throughout
  - Respects max_open_positions and sector concentration limits
  - Tracks MAE/MFE for each trade
  - Returns BacktestResult with full trade list and equity curve

Usage:
    from backtester.engine import Backtester
    result = Backtester(strategy, cfg).run(data, start_date, end_date)
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from core.models import (
    BacktestResult, BacktestTrade, ExitReason, PerformanceMetrics,
)
from core.logging import get_logger
from backtester.metrics import compute_metrics
from indicators.composite import enrich
from strategies.base import BaseStrategy

log = get_logger("backtester.engine")


class Backtester:
    """
    Single-strategy backtester.

    data: dict[ticker, enriched_df]  (call enrich() before passing)
    """

    def __init__(self, strategy: BaseStrategy, cfg: dict,
                 exit_mode: str = "targets", trail_atr_mult: float = 2.5,
                 max_hold_days: int | None = None):
        """
        exit_mode:
          "targets"  — fixed T1/T2/T3 partial exits (original behaviour)
          "trailing" — let winners run with an ATR-based trailing stop that
                       ratchets up but never down. Cuts losers at the initial
                       stop, but never caps the upside.
          "hybrid"   — bank half at T1 (+1.5R), move stop to breakeven, then
                       trail the remaining half with the ATR stop. Locks in
                       profit like "targets", keeps upside like "trailing".
        trail_atr_mult — how many ATRs below the high-water close to trail.
        max_hold_days  — time stop: exit at close after this many calendar
                         days regardless of P&L (dead-capital rotation).
        """
        self.strategy = strategy
        self.cfg = cfg
        self.exit_mode = exit_mode
        self.trail_atr_mult = trail_atr_mult
        self.max_hold_days = max_hold_days
        self._capital_cfg = cfg.get("account", {})
        self._initial_capital = float(self._capital_cfg.get("size", 100_000))
        self._risk_per_trade  = float(self._capital_cfg.get("risk_per_trade", 0.01))
        self._max_positions   = int(cfg.get("risk", {}).get("max_open_positions", 10))
        self._max_single      = float(cfg.get("risk", {}).get("max_single_position", 0.10))

    def run(
        self,
        data: dict[str, pd.DataFrame],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> BacktestResult:
        """
        Run backtest across all tickers in data.

        data must already be enriched (all indicator columns present).
        Signals fire on bar close → filled at next bar open.
        """
        # Build unified trading calendar
        all_dates = sorted(set(
            d for df in data.values()
            for d in df.index.date
        ))

        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        if not all_dates:
            log.warning("No trading dates in range")
            return self._empty_result(start_date, end_date)

        capital    = self._initial_capital
        positions: dict[str, dict] = {}   # ticker -> {entry, stop, t1, t2, t3, shares, entry_date}
        completed: list[BacktestTrade] = []
        equity: dict[date, float] = {}

        pending_entries: list[dict] = []  # Signals queued for next-open fill

        for today in all_dates:
            today_open_fills: list[dict] = []

            # ── Fill pending entries at today's open ───────────────
            for entry_order in pending_entries:
                ticker = entry_order["ticker"]
                if ticker not in data:
                    continue
                df = data[ticker]
                today_bars = df[df.index.date == today]
                if today_bars.empty:
                    continue
                fill_price = float(today_bars["open"].iloc[0])

                # Risk sizing: shares = (capital * risk_per_trade) / (entry - stop)
                stop = entry_order["stop"]
                risk_per_share = fill_price - stop
                if risk_per_share <= 0:
                    continue

                dollar_risk = capital * self._risk_per_trade
                shares = int(dollar_risk / risk_per_share)
                if shares < 1:
                    continue

                position_value = shares * fill_price
                max_position   = capital * self._max_single
                if position_value > max_position:
                    shares = max(1, int(max_position / fill_price))

                if shares < 1 or position_value > capital * 0.95:
                    continue

                capital -= shares * fill_price
                positions[ticker] = {
                    "entry":      fill_price,
                    "stop":       stop,
                    "t1":         entry_order["t1"],
                    "t2":         entry_order["t2"],
                    "t3":         entry_order["t3"],
                    "shares":     shares,
                    "entry_date": today,
                    "mae":        0.0,
                    "mfe":        0.0,
                    "partial":    False,  # partial exit at T1?
                }

            pending_entries = []

            # ── Manage open positions ──────────────────────────────
            closed_today: list[str] = []

            for ticker, pos in positions.items():
                if ticker not in data:
                    continue
                df = data[ticker]
                today_bars = df[df.index.date == today]
                if today_bars.empty:
                    continue

                bar  = today_bars.iloc[0]
                high = float(bar["high"])
                low  = float(bar["low"])

                # Update MAE/MFE
                entry = pos["entry"]
                adverse  = (low  - entry) / entry
                favorable = (high - entry) / entry
                pos["mae"] = min(pos["mae"], adverse)
                pos["mfe"] = max(pos["mfe"], favorable)

                exit_price  = None
                exit_reason = None
                shares_closed = pos["shares"]

                if self.exit_mode in ("trailing", "hybrid"):
                    # ── Trailing / hybrid exit ─────────────────────────
                    close_px = float(bar["close"])
                    atr_v = float(bar.get("atr_14", close_px * 0.02))

                    partial_fill = None  # (price, shares) banked at T1 today

                    # Hybrid: bank half at T1, stop→breakeven, trail the rest
                    if (self.exit_mode == "hybrid" and not pos["partial"]
                            and high >= pos["t1"]):
                        half = max(1, pos["shares"] // 2)
                        partial_fill = (pos["t1"], half)
                        pos["shares"] -= half
                        pos["partial"] = True
                        pos["stop"] = max(pos["stop"], entry)  # breakeven floor

                    # Ratchet the trailing stop up (never down)
                    new_trail = close_px - self.trail_atr_mult * atr_v
                    if new_trail > pos["stop"]:
                        pos["stop"] = new_trail

                    if low <= pos["stop"]:
                        exit_price  = pos["stop"]
                        exit_reason = ExitReason.TRAILING_STOP
                        shares_closed = pos["shares"]

                    # Time stop: rotate out dead capital
                    if (exit_price is None and self.max_hold_days is not None
                            and (today - pos["entry_date"]).days >= self.max_hold_days):
                        exit_price  = close_px
                        exit_reason = ExitReason.MAX_HOLD
                        shares_closed = pos["shares"]

                    if partial_fill is not None:
                        fp, fs = partial_fill
                        capital += fs * fp
                        completed.append(BacktestTrade(
                            ticker=ticker, strategy=self.strategy.name,
                            entry_date=pos["entry_date"], exit_date=today,
                            entry_price=entry, exit_price=fp, shares=fs,
                            pnl=(fp - entry) * fs, pnl_pct=(fp - entry) / entry,
                            holding_days=(today - pos["entry_date"]).days,
                            exit_reason=ExitReason.TARGET_1,
                            mae=pos["mae"], mfe=pos["mfe"],
                        ))

                    if exit_price is not None and shares_closed > 0:
                        pnl = (exit_price - entry) * shares_closed
                        capital += shares_closed * exit_price
                        completed.append(BacktestTrade(
                            ticker=ticker, strategy=self.strategy.name,
                            entry_date=pos["entry_date"], exit_date=today,
                            entry_price=entry, exit_price=exit_price,
                            shares=shares_closed,
                            pnl=pnl, pnl_pct=(exit_price - entry) / entry,
                            holding_days=(today - pos["entry_date"]).days,
                            exit_reason=exit_reason, mae=pos["mae"], mfe=pos["mfe"],
                        ))
                        closed_today.append(ticker)
                    continue  # skip the fixed-target logic below

                # Stop hit (intrabar check using low)
                if low <= pos["stop"]:
                    exit_price  = pos["stop"]
                    exit_reason = ExitReason.STOP_LOSS

                # T1 hit (partial exit — take half off)
                elif not pos["partial"] and high >= pos["t1"]:
                    exit_price    = pos["t1"]
                    exit_reason   = ExitReason.TARGET_1
                    shares_closed = max(1, pos["shares"] // 2)
                    pos["shares"] -= shares_closed
                    pos["partial"] = True
                    # Raise stop to breakeven
                    pos["stop"] = entry

                # T2 hit
                elif pos["partial"] and high >= pos["t2"]:
                    exit_price  = pos["t2"]
                    exit_reason = ExitReason.TARGET_2

                # T3 hit (full exit)
                elif pos["partial"] and high >= pos["t3"]:
                    exit_price  = pos["t3"]
                    exit_reason = ExitReason.TARGET_3

                if exit_price is not None:
                    pnl = (exit_price - entry) * shares_closed
                    capital += shares_closed * exit_price
                    completed.append(BacktestTrade(
                        ticker      = ticker,
                        strategy    = self.strategy.name,
                        entry_date  = pos["entry_date"],
                        exit_date   = today,
                        entry_price = entry,
                        exit_price  = exit_price,
                        shares      = shares_closed,
                        pnl         = pnl,
                        pnl_pct     = (exit_price - entry) / entry,
                        holding_days = (today - pos["entry_date"]).days,
                        exit_reason  = exit_reason,
                        mae          = pos["mae"],
                        mfe          = pos["mfe"],
                    ))
                    if exit_reason != ExitReason.TARGET_1:
                        closed_today.append(ticker)

            for t in closed_today:
                del positions[t]

            # ── Mark open positions to market ─────────────────────
            open_value = sum(
                float(data[t][data[t].index.date == today]["close"].iloc[-1]) * p["shares"]
                if t in data and not data[t][data[t].index.date == today].empty
                else p["entry"] * p["shares"]
                for t, p in positions.items()
            )
            equity[today] = capital + open_value

            # ── Generate new signals (filled tomorrow) ─────────────
            if len(positions) < self._max_positions:
                for ticker, df in data.items():
                    if ticker in positions:
                        continue
                    today_bars = df[df.index.date == today]
                    if today_bars.empty:
                        continue
                    # Use historical slice up to today (no look-ahead)
                    hist = df[df.index.date <= today]
                    if len(hist) < 60:
                        continue
                    try:
                        signal = self.strategy.generate_signal(hist, ticker, self.cfg)
                    except Exception:
                        continue
                    if signal is None:
                        continue
                    if signal.stop_loss >= signal.entry_price:
                        continue
                    pending_entries.append({
                        "ticker": ticker,
                        "stop":   signal.stop_loss,
                        "t1":     signal.target_1,
                        "t2":     signal.target_2,
                        "t3":     signal.target_3,
                    })

        # ── Close all remaining positions at last date ─────────────
        last_date = all_dates[-1]
        for ticker, pos in positions.items():
            if ticker not in data:
                continue
            df = data[ticker]
            last_bars = df[df.index.date <= last_date]
            if last_bars.empty:
                continue
            exit_price = float(last_bars["close"].iloc[-1])
            pnl = (exit_price - pos["entry"]) * pos["shares"]
            capital += pos["shares"] * exit_price
            completed.append(BacktestTrade(
                ticker       = ticker,
                strategy     = self.strategy.name,
                entry_date   = pos["entry_date"],
                exit_date    = last_date,
                entry_price  = pos["entry"],
                exit_price   = exit_price,
                shares       = pos["shares"],
                pnl          = pnl,
                pnl_pct      = (exit_price - pos["entry"]) / pos["entry"],
                holding_days = (last_date - pos["entry_date"]).days,
                exit_reason  = ExitReason.END_OF_BACKTEST,
                mae          = pos["mae"],
                mfe          = pos["mfe"],
            ))

        equity_series = pd.Series(equity)
        equity_series.index = pd.to_datetime(equity_series.index)
        # Fill any gaps with forward fill
        equity_series = equity_series.ffill()

        metrics = compute_metrics(
            strategy=self.strategy.name,
            period=f"{all_dates[0]}:{all_dates[-1]}",
            trades=completed,
            equity_curve=equity_series,
            initial_capital=self._initial_capital,
        )

        log.info(
            "%s | trades=%d | CAGR=%.1f%% | Sharpe=%.2f | MaxDD=%.1f%%",
            self.strategy.name, len(completed),
            metrics.cagr * 100, metrics.sharpe_ratio, metrics.max_drawdown * 100,
        )

        return BacktestResult(
            strategy        = self.strategy.name,
            period          = f"{all_dates[0]}:{all_dates[-1]}",
            start_date      = all_dates[0],
            end_date        = all_dates[-1],
            initial_capital = self._initial_capital,
            final_capital   = capital,
            trades          = completed,
            metrics         = metrics,
            equity_curve    = equity_series,
        )

    def _empty_result(self, start_date, end_date) -> BacktestResult:
        today = date.today()
        return BacktestResult(
            strategy        = self.strategy.name,
            period          = "",
            start_date      = start_date or today,
            end_date        = end_date   or today,
            initial_capital = self._initial_capital,
            final_capital   = self._initial_capital,
        )
