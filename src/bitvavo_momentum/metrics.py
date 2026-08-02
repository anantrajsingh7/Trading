"""Phase 7: evaluation metrics.

The central metric is **net expectancy per trade**::

    E = win_rate * average_net_win - loss_rate * average_net_loss

Win rate alone is explicitly not a selection criterion: a strategy that wins 85%
of the time and gives it all back on the other 15% is a losing strategy, and a
momentum system with a 35% win rate and a 3:1 payoff is a good one.

Annualisation caveat: an event strategy trades sporadically. Sharpe and
annualised return are computed on the *daily equity curve including flat days*,
so a strategy with 40 trades in three years is not credited with the Sharpe of a
continuously invested one. Where the sample is too small for a statistic to mean
anything, ``np.nan`` is returned rather than a number that invites over-reading.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .timeutils import session_bucket

log = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 365.0  # crypto trades every day
MIN_TRADES_FOR_RATIOS = 20


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator not in (0, 0.0) else np.nan


def compute_trade_metrics(trades: pd.DataFrame, starting_equity: float = 10_000.0) -> dict[str, Any]:
    """Full metric set for one closed-trade table."""
    closed = trades[trades["status"] == "closed"] if "status" in trades.columns else trades
    closed = closed.dropna(subset=["net_return"]) if not closed.empty else closed

    if closed.empty:
        return {
            "n_trades": 0, "n_wins": 0, "n_losses": 0, "win_rate": np.nan,
            "avg_win": np.nan, "avg_loss": np.nan, "payoff_ratio": np.nan,
            "gross_expectancy": np.nan, "net_expectancy": np.nan,
            "net_expectancy_bps": np.nan, "profit_factor": np.nan,
            "total_net_pnl_eur": 0.0, "cumulative_net_return": 0.0,
            "annualised_return": np.nan, "sharpe": np.nan, "sortino": np.nan,
            "max_drawdown": np.nan, "calmar": np.nan, "recovery_factor": np.nan,
            "avg_holding_minutes": np.nan, "median_holding_minutes": np.nan,
            "exposure": np.nan, "turnover_eur": 0.0, "total_fees_eur": 0.0,
            "total_spread_cost_eur": 0.0, "total_slippage_cost_eur": 0.0,
            "max_consecutive_losses": 0, "best_trade": np.nan, "worst_trade": np.nan,
            "median_trade": np.nan, "p05_trade": np.nan, "p95_trade": np.nan,
            "var_95": np.nan, "expected_shortfall_95": np.nan,
            "ambiguous_exit_share": np.nan, "avg_fill_fraction": np.nan,
            "n_distinct_markets": 0, "n_distinct_months": 0,
        }

    net = closed["net_return"].astype(float)
    gross = closed["gross_return"].astype(float)
    wins = net[net > 0]
    losses = net[net <= 0]

    win_rate = len(wins) / len(net)
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0
    net_expectancy = win_rate * avg_win - (1.0 - win_rate) * avg_loss

    gross_wins = closed.loc[closed["net_pnl_eur"] > 0, "net_pnl_eur"].sum()
    gross_losses = -closed.loc[closed["net_pnl_eur"] <= 0, "net_pnl_eur"].sum()

    total_pnl = float(closed["net_pnl_eur"].sum())
    equity_curve = _daily_equity(closed, starting_equity)
    max_dd, ann_return, sharpe, sortino = _curve_stats(equity_curve, len(net))

    streak = 0
    max_streak = 0
    for value in net:
        streak = streak + 1 if value <= 0 else 0
        max_streak = max(max_streak, streak)

    months = pd.to_datetime(closed["entry_time"], utc=True).dt.tz_localize(None).dt.to_period("M")
    exposure = _exposure(closed, equity_curve)

    return {
        "n_trades": int(len(net)),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": _safe_div(avg_win, avg_loss),
        "gross_expectancy": float(gross.mean()),
        "net_expectancy": float(net_expectancy),
        "net_expectancy_bps": float(net_expectancy * 10_000),
        "profit_factor": _safe_div(gross_wins, gross_losses),
        "total_net_pnl_eur": total_pnl,
        "cumulative_net_return": _safe_div(total_pnl, starting_equity),
        "annualised_return": ann_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": _safe_div(ann_return, abs(max_dd)) if max_dd and np.isfinite(max_dd) else np.nan,
        "recovery_factor": _safe_div(total_pnl / starting_equity, abs(max_dd)) if max_dd else np.nan,
        "avg_holding_minutes": float(closed["holding_minutes"].mean()),
        "median_holding_minutes": float(closed["holding_minutes"].median()),
        "exposure": exposure,
        "turnover_eur": float(closed["notional_eur"].sum() * 2),
        "total_fees_eur": float(closed["fees_eur"].sum()),
        "total_spread_cost_eur": float(closed["spread_cost_eur"].sum()),
        "total_slippage_cost_eur": float(closed["slippage_cost_eur"].sum()),
        "max_consecutive_losses": int(max_streak),
        "best_trade": float(net.max()),
        "worst_trade": float(net.min()),
        "median_trade": float(net.median()),
        "p05_trade": float(net.quantile(0.05)),
        "p95_trade": float(net.quantile(0.95)),
        "var_95": float(net.quantile(0.05)),
        "expected_shortfall_95": float(net[net <= net.quantile(0.05)].mean()) if len(net) >= 20 else np.nan,
        "ambiguous_exit_share": float(closed["ambiguous_bar"].astype(float).mean()),
        "avg_fill_fraction": float(closed["fill_fraction"].astype(float).mean()),
        "n_distinct_markets": int(closed["market"].nunique()),
        "n_distinct_months": int(months.nunique()),
    }


def _daily_equity(closed: pd.DataFrame, starting_equity: float) -> pd.DataFrame:
    ordered = closed.sort_values("exit_time")
    times = pd.to_datetime(ordered["exit_time"], utc=True)
    pnl = pd.Series(ordered["net_pnl_eur"].to_numpy(), index=times)
    daily = pnl.resample("1D").sum()
    if daily.empty:
        return pd.DataFrame(columns=["equity", "drawdown"])
    equity = starting_equity + daily.cumsum()
    frame = pd.DataFrame({"equity": equity})
    frame["peak"] = frame["equity"].cummax()
    frame["drawdown"] = frame["equity"] / frame["peak"] - 1.0
    return frame


def _curve_stats(equity: pd.DataFrame, n_trades: int) -> tuple[float, float, float, float]:
    if equity.empty or len(equity) < 2:
        return np.nan, np.nan, np.nan, np.nan
    max_dd = float(equity["drawdown"].min())
    returns = equity["equity"].pct_change().dropna()
    if len(returns) < 2 or n_trades < MIN_TRADES_FOR_RATIOS:
        return max_dd, np.nan, np.nan, np.nan
    days = max(1.0, (equity.index[-1] - equity.index[0]).days)
    total_growth = equity["equity"].iloc[-1] / equity["equity"].iloc[0]
    ann_return = float(total_growth ** (TRADING_DAYS_PER_YEAR / days) - 1.0) if total_growth > 0 else np.nan
    std = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else np.nan
    downside = returns[returns < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
    sortino = float(returns.mean() / dstd * np.sqrt(TRADING_DAYS_PER_YEAR)) if dstd and dstd > 0 else np.nan
    return max_dd, ann_return, sharpe, sortino


def _exposure(closed: pd.DataFrame, equity: pd.DataFrame) -> float:
    """Share of calendar time with at least one position open."""
    if closed.empty or equity.empty:
        return np.nan
    total_minutes = float(closed["holding_minutes"].sum())
    span = pd.to_datetime(closed["exit_time"], utc=True).max() - pd.to_datetime(
        closed["entry_time"], utc=True
    ).min()
    span_minutes = span.total_seconds() / 60.0
    return _safe_div(total_minutes, span_minutes)


# --------------------------------------------------------------------------- #
# breakdowns
# --------------------------------------------------------------------------- #
def breakdown(
    trades: pd.DataFrame,
    by: str,
    starting_equity: float = 10_000.0,
    min_trades: int = 5,
) -> pd.DataFrame:
    """Metric table split by an attribute (coin, regime, year, hour, ...)."""
    closed = trades[trades["status"] == "closed"].copy() if "status" in trades.columns else trades.copy()
    if closed.empty:
        return pd.DataFrame()

    if by == "year":
        closed["_key"] = pd.to_datetime(closed["entry_time"], utc=True).dt.year
    elif by == "month":
        closed["_key"] = pd.to_datetime(closed["entry_time"], utc=True).dt.strftime("%Y-%m")
    elif by == "hour_utc":
        closed["_key"] = pd.to_datetime(closed["entry_time"], utc=True).dt.hour
    elif by == "day_of_week":
        closed["_key"] = pd.to_datetime(closed["entry_time"], utc=True).dt.day_name()
    elif by == "session":
        closed["_key"] = [session_bucket(t) for t in pd.to_datetime(closed["entry_time"], utc=True)]
    elif by in closed.columns:
        closed["_key"] = closed[by]
    else:
        raise ValueError(f"Unknown breakdown key {by!r}")

    rows = []
    for key, group in closed.groupby("_key", dropna=False):
        metrics = compute_trade_metrics(group, starting_equity)
        metrics[by] = key
        metrics["below_min_trades"] = metrics["n_trades"] < min_trades
        rows.append(metrics)
    frame = pd.DataFrame(rows)
    cols = [by] + [c for c in frame.columns if c != by]
    return frame[cols].sort_values(by).reset_index(drop=True)


def concentration_report(trades: pd.DataFrame) -> dict[str, Any]:
    """How much of the profit rests on one coin or one month?

    A strategy whose entire edge is one coin in one month is a story about that
    coin, not a strategy.
    """
    closed = trades[trades["status"] == "closed"] if "status" in trades.columns else trades
    if closed.empty:
        return {"total_net_pnl_eur": 0.0}
    total = float(closed["net_pnl_eur"].sum())
    by_coin = closed.groupby("market")["net_pnl_eur"].sum().sort_values(ascending=False)
    months = pd.to_datetime(closed["entry_time"], utc=True).dt.tz_localize(None).dt.to_period("M").astype(str)
    by_month = closed.groupby(months)["net_pnl_eur"].sum().sort_values(ascending=False)

    def _share(series: pd.Series) -> float:
        positive = series[series > 0].sum()
        return _safe_div(float(series.iloc[0]), float(positive)) if len(series) and positive > 0 else np.nan

    return {
        "total_net_pnl_eur": total,
        "top_coin": by_coin.index[0] if len(by_coin) else None,
        "top_coin_pnl_eur": float(by_coin.iloc[0]) if len(by_coin) else np.nan,
        "top_coin_profit_share": _share(by_coin),
        "top_month": by_month.index[0] if len(by_month) else None,
        "top_month_pnl_eur": float(by_month.iloc[0]) if len(by_month) else np.nan,
        "top_month_profit_share": _share(by_month),
        "n_markets": int(closed["market"].nunique()),
        "n_months": int(months.nunique()),
        "pnl_without_top_coin": float(total - (by_coin.iloc[0] if len(by_coin) else 0.0)),
        "pnl_without_top_month": float(total - (by_month.iloc[0] if len(by_month) else 0.0)),
    }


def summarise_results(results: dict[str, Any], starting_equity: float = 10_000.0) -> pd.DataFrame:
    """One metric row per backtest in a ``{label: BacktestResult}`` mapping."""
    rows = []
    for label, result in results.items():
        metrics = compute_trade_metrics(result.trades, starting_equity)
        metrics["label"] = label
        parts = label.split("|")
        metrics["strategy"] = parts[0] if parts else label
        metrics["exit_policy"] = parts[1] if len(parts) > 1 else ""
        metrics["scenario"] = parts[2] if len(parts) > 2 else ""
        metrics.update({f"funnel_{k}": v for k, v in result.funnel.items()})
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    lead = ["label", "strategy", "exit_policy", "scenario", "n_trades", "win_rate",
            "net_expectancy", "net_expectancy_bps", "profit_factor", "total_net_pnl_eur",
            "max_drawdown", "sharpe"]
    ordered = [c for c in lead if c in frame.columns] + [c for c in frame.columns if c not in lead]
    return frame[ordered].sort_values("net_expectancy", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# rejection rules (Phase 7)
# --------------------------------------------------------------------------- #
def rejection_reasons(
    metrics: dict[str, Any],
    concentration: dict[str, Any],
    stress_metrics: dict[str, Any] | None = None,
    min_trades: int = 100,
    max_coin_share: float = 0.40,
    max_month_share: float = 0.40,
    max_drawdown_limit: float = -0.25,
) -> list[str]:
    """Apply the Phase 7 rejection criteria. An empty list means "not rejected"."""
    reasons: list[str] = []
    if metrics.get("n_trades", 0) < min_trades:
        reasons.append(
            f"only {metrics.get('n_trades', 0)} trades (< {min_trades} independent observations)"
        )
    expectancy = metrics.get("net_expectancy")
    if expectancy is None or not np.isfinite(expectancy) or expectancy <= 0:
        reasons.append(f"net expectancy after costs is not positive ({expectancy})")
    coin_share = concentration.get("top_coin_profit_share")
    if coin_share is not None and np.isfinite(coin_share) and coin_share > max_coin_share:
        reasons.append(
            f"{coin_share:.0%} of gross profit comes from {concentration.get('top_coin')}"
        )
    month_share = concentration.get("top_month_profit_share")
    if month_share is not None and np.isfinite(month_share) and month_share > max_month_share:
        reasons.append(
            f"{month_share:.0%} of gross profit comes from {concentration.get('top_month')}"
        )
    dd = metrics.get("max_drawdown")
    if dd is not None and np.isfinite(dd) and dd < max_drawdown_limit:
        reasons.append(f"maximum drawdown {dd:.1%} beyond the {max_drawdown_limit:.0%} limit")
    if stress_metrics is not None:
        stress_expectancy = stress_metrics.get("net_expectancy")
        if stress_expectancy is None or not np.isfinite(stress_expectancy) or stress_expectancy <= 0:
            reasons.append("net expectancy is negative under the stress execution scenario")
    ambiguous = metrics.get("ambiguous_exit_share")
    if ambiguous is not None and np.isfinite(ambiguous) and ambiguous > 0.30:
        reasons.append(
            f"{ambiguous:.0%} of exits are intrabar-ambiguous - the result rests on an assumption"
        )
    return reasons
