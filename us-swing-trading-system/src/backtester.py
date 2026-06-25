"""
Event-driven backtester.
- No look-ahead bias: signals on row i use data up to row i.
- Fixed fractional risk: 1% per trade.
- Tracks open trades, closes on stop, target or max holding period.
- Separate in-sample / out-of-sample windows.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategies import get_signals, STRATEGY_NAMES
from risk import calc_stop_loss, calc_targets, position_size

log = logging.getLogger("swing.backtest")


@dataclass
class ClosedTrade:
    ticker: str
    strategy: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    holding_days: int
    exit_reason: str  # "target", "stop", "max_hold", "end"


@dataclass
class BacktestResult:
    strategy: str
    trades: list[ClosedTrade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    # computed metrics
    total_return: float = 0.0
    cagr: float = 0.0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    avg_holding: float = 0.0
    n_trades: int = 0
    best_ticker: str = ""
    worst_ticker: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_return_%": round(self.total_return * 100, 2),
            "cagr_%": round(self.cagr * 100, 2),
            "win_rate_%": round(self.win_rate * 100, 2),
            "loss_rate_%": round(self.loss_rate * 100, 2),
            "avg_winner_%": round(self.avg_winner * 100, 2),
            "avg_loser_%": round(self.avg_loser * 100, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown_%": round(self.max_drawdown * 100, 2),
            "sharpe": round(self.sharpe, 2),
            "avg_holding_days": round(self.avg_holding, 1),
            "n_trades": self.n_trades,
            "best_ticker": self.best_ticker,
            "worst_ticker": self.worst_ticker,
        }


def _compute_metrics(result: BacktestResult, initial_equity: float) -> None:
    trades = result.trades
    if not trades:
        return

    result.n_trades = len(trades)
    pnls = [t.pnl for t in trades]
    pcts = [t.pnl_pct for t in trades]
    winners = [p for p in pcts if p > 0]
    losers = [p for p in pcts if p <= 0]
    result.win_rate = len(winners) / len(trades)
    result.loss_rate = len(losers) / len(trades)
    result.avg_winner = float(np.mean(winners)) if winners else 0.0
    result.avg_loser = float(np.mean(losers)) if losers else 0.0
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    result.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
    result.avg_holding = float(np.mean([t.holding_days for t in trades]))

    # equity curve
    equity = initial_equity
    eq_series: dict[pd.Timestamp, float] = {}
    for t in sorted(trades, key=lambda x: x.exit_date):
        equity += t.pnl
        eq_series[t.exit_date] = equity
    curve = pd.Series(eq_series)
    result.equity_curve = curve

    result.total_return = (equity - initial_equity) / initial_equity

    # CAGR
    if len(curve) >= 2:
        years = (curve.index[-1] - curve.index[0]).days / 365.25
        if years > 0:
            result.cagr = (equity / initial_equity) ** (1 / years) - 1

    # max drawdown
    running_max = curve.cummax()
    dd = (curve - running_max) / running_max
    result.max_drawdown = float(dd.min()) if len(dd) > 0 else 0.0

    # Sharpe (daily returns approximation)
    daily_ret = pd.Series([t.pnl_pct for t in sorted(trades, key=lambda x: x.exit_date)])
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        result.sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))

    # best / worst ticker
    ticker_pnl: dict[str, float] = {}
    for t in trades:
        ticker_pnl[t.ticker] = ticker_pnl.get(t.ticker, 0) + t.pnl
    if ticker_pnl:
        result.best_ticker = max(ticker_pnl, key=ticker_pnl.__getitem__)
        result.worst_ticker = min(ticker_pnl, key=ticker_pnl.__getitem__)


def backtest_strategy(strategy_key: str,
                      data: dict[str, pd.DataFrame],
                      cfg: dict,
                      start_date: str | None = None,
                      end_date: str | None = None) -> BacktestResult:
    acc = cfg.get("account", {})
    initial_equity = float(acc.get("size", 10000))
    risk_pct = float(acc.get("risk_per_trade", 0.01))
    max_hold = int(cfg.get("trading", {}).get("max_holding_days", 30))
    atr_mult = float(cfg.get("indicators", {}).get("atr_multiplier", 1.5))
    min_rr = float(acc.get("min_rr_ratio", 2.0))
    slippage = float(cfg.get("backtest", {}).get("slippage", 0.001))
    commission = float(cfg.get("backtest", {}).get("commission", 5.0))

    result = BacktestResult(strategy=STRATEGY_NAMES.get(strategy_key, strategy_key))
    all_trades: list[ClosedTrade] = []

    for ticker, df in data.items():
        if len(df) < 60:
            continue
        df_window = df.copy()
        if start_date:
            df_window = df_window[df_window.index >= start_date]
        if end_date:
            df_window = df_window[df_window.index <= end_date]
        if len(df_window) < 20:
            continue

        try:
            signals = get_signals(df, strategy_key, cfg)
            # align signals to window
            signals = signals.reindex(df_window.index).fillna(False)
        except Exception as e:
            log.debug("Signal error %s %s: %s", ticker, strategy_key, e)
            continue

        in_trade = False
        entry_price = 0.0
        stop = 0.0
        target1 = 0.0
        entry_date = None
        shares = 0

        for i, (date, row) in enumerate(df_window.iterrows()):
            if in_trade:
                price = float(row["close"])
                hold_days = i - entry_idx  # type: ignore[name-defined]
                exit_reason = None
                exit_price = price

                if price <= stop:
                    exit_reason = "stop"
                    exit_price = stop
                elif price >= target1:
                    exit_reason = "target"
                    exit_price = target1
                elif hold_days >= max_hold:
                    exit_reason = "max_hold"

                if exit_reason or i == len(df_window) - 1:
                    if exit_reason is None:
                        exit_reason = "end"
                    exit_price *= (1 - slippage)
                    pnl = shares * (exit_price - entry_price) - commission
                    pnl_pct = (exit_price - entry_price) / entry_price
                    all_trades.append(ClosedTrade(
                        ticker=ticker,
                        strategy=strategy_key,
                        entry_date=entry_date,  # type: ignore[arg-type]
                        exit_date=date,
                        entry_price=round(entry_price, 4),
                        exit_price=round(exit_price, 4),
                        shares=shares,
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 4),
                        holding_days=hold_days,
                        exit_reason=exit_reason,
                    ))
                    in_trade = False
            else:
                if bool(signals.get(date, False)):
                    # entry next bar open approximated as close * (1 + slippage)
                    ep = float(row["close"]) * (1 + slippage)
                    # build a sub-df up to current row for stop calc
                    sub = df_window.iloc[: i + 1]
                    stop_price = calc_stop_loss(sub, ep, atr_mult)
                    if stop_price >= ep:
                        continue
                    t1, _ = calc_targets(ep, stop_price, min_rr)
                    sh, _, _ = position_size(initial_equity, risk_pct, ep, stop_price)
                    if sh == 0:
                        continue
                    in_trade = True
                    entry_price = ep
                    stop = stop_price
                    target1 = t1
                    entry_date = date
                    entry_idx = i  # type: ignore[assignment]
                    shares = sh

    result.trades = all_trades
    _compute_metrics(result, initial_equity)
    return result


def run_all_backtests(data: dict[str, pd.DataFrame],
                      cfg: dict) -> dict[str, BacktestResult]:
    from strategies import STRATEGIES
    bt_cfg = cfg.get("backtest", {})
    is_end = bt_cfg.get("in_sample_end")
    oos_start = bt_cfg.get("out_of_sample_start")

    results: dict[str, BacktestResult] = {}
    for key in STRATEGIES:
        strat_cfg = cfg.get("strategies", {}).get(key, {})
        if not strat_cfg.get("enabled", True):
            continue
        log.info("Backtesting: %s (in-sample)", key)
        res = backtest_strategy(key, data, cfg, end_date=is_end)
        log.info("  -> %d trades, PF=%.2f, WR=%.1f%%",
                 res.n_trades, res.profit_factor, res.win_rate * 100)
        results[key] = res
    return results
