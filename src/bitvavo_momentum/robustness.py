"""Phase 8: overfitting controls and statistical robustness.

Tools here answer one question in several ways: *could this result plausibly have
arisen by chance, or from choices made after seeing the data?*

* :func:`bootstrap_metric` - confidence intervals by resampling trades.
* :func:`block_bootstrap_metric` - same, preserving short-term dependence between
  consecutive trades (overlapping events on correlated coins are not iid).
* :func:`monte_carlo_paths` - reshuffles the trade *sequence* to show how much of
  the observed drawdown is ordering luck.
* :func:`permutation_test_vs_random` - compares against entries at random times
  in the same markets.
* :func:`deflated_sharpe_ratio` - discounts the Sharpe ratio for the number of
  configurations tried, per Bailey & Lopez de Prado.
* :func:`parameter_stability` - neighbouring parameters should also work. One
  spike in a heatmap is a coincidence, a plateau is a finding.

A note on independence: momentum events cluster (one market-wide pump creates
many simultaneous events). Statistics that assume iid trades therefore overstate
significance. The block bootstrap and the clustered event count in
:func:`independent_trade_count` are the honest versions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger(__name__)


def _as_array(values: Sequence[float] | pd.Series) -> np.ndarray:
    arr = np.asarray(pd.Series(values).dropna(), dtype="float64")
    return arr


def bootstrap_metric(
    values: Sequence[float] | pd.Series,
    statistic: Callable[[np.ndarray], float] = np.mean,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260802,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for a per-trade statistic."""
    arr = _as_array(values)
    if len(arr) < 5:
        return {"observed": float(statistic(arr)) if len(arr) else np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "n": len(arr), "p_value_gt_zero": np.nan}
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype="float64")
    for i in range(iterations):
        sample = rng.choice(arr, size=len(arr), replace=True)
        draws[i] = statistic(sample)
    alpha = (1.0 - confidence) / 2.0
    return {
        "observed": float(statistic(arr)),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1 - alpha)),
        "n": int(len(arr)),
        "p_value_gt_zero": float((draws <= 0).mean()),
    }


def block_bootstrap_metric(
    values: Sequence[float] | pd.Series,
    block_size: int = 10,
    statistic: Callable[[np.ndarray], float] = np.mean,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260802,
) -> dict[str, float]:
    """Moving-block bootstrap: keeps neighbouring trades together."""
    arr = _as_array(values)
    n = len(arr)
    if n < max(10, block_size * 2):
        return bootstrap_metric(arr, statistic, iterations, confidence, seed)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    draws = np.empty(iterations, dtype="float64")
    starts_max = n - block_size
    for i in range(iterations):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([arr[s : s + block_size] for s in starts])[:n]
        draws[i] = statistic(sample)
    alpha = (1.0 - confidence) / 2.0
    return {
        "observed": float(statistic(arr)),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1 - alpha)),
        "n": n,
        "block_size": block_size,
        "p_value_gt_zero": float((draws <= 0).mean()),
    }


def monte_carlo_paths(
    trade_returns: Sequence[float] | pd.Series,
    starting_equity: float = 10_000.0,
    iterations: int = 2000,
    seed: int = 20260802,
    position_fraction: float = 1.0,
) -> dict[str, Any]:
    """Reshuffle the trade order and report the distribution of outcomes.

    The set of trades is held fixed; only their *sequence* changes. A strategy
    whose drawdown depends heavily on ordering is fragile even if its expectancy
    is positive.
    """
    arr = _as_array(trade_returns)
    if len(arr) < 5:
        return {"n": int(len(arr))}
    rng = np.random.default_rng(seed)
    finals = np.empty(iterations)
    drawdowns = np.empty(iterations)
    for i in range(iterations):
        path = rng.permutation(arr)
        equity = starting_equity * np.cumprod(1.0 + path * position_fraction)
        peak = np.maximum.accumulate(equity)
        finals[i] = equity[-1]
        drawdowns[i] = float((equity / peak - 1.0).min())
    observed_equity = starting_equity * np.cumprod(1.0 + arr * position_fraction)
    observed_peak = np.maximum.accumulate(observed_equity)
    return {
        "n": int(len(arr)),
        "iterations": iterations,
        "observed_final_equity": float(observed_equity[-1]),
        "observed_max_drawdown": float((observed_equity / observed_peak - 1.0).min()),
        "median_final_equity": float(np.median(finals)),
        "p05_final_equity": float(np.quantile(finals, 0.05)),
        "p95_final_equity": float(np.quantile(finals, 0.95)),
        "probability_of_loss": float((finals < starting_equity).mean()),
        "median_max_drawdown": float(np.median(drawdowns)),
        "p05_max_drawdown": float(np.quantile(drawdowns, 0.05)),
        "worst_max_drawdown": float(drawdowns.min()),
        "final_equity_samples": finals,
        "max_drawdown_samples": drawdowns,
    }


def permutation_test_vs_random(
    strategy_returns: Sequence[float] | pd.Series,
    random_returns: Sequence[float] | pd.Series,
    iterations: int = 1000,
    seed: int = 20260802,
) -> dict[str, float]:
    """Two-sample permutation test on the mean difference.

    Null hypothesis: strategy-timed entries and randomly-timed entries in the
    same markets come from the same distribution.
    """
    a = _as_array(strategy_returns)
    b = _as_array(random_returns)
    if len(a) < 5 or len(b) < 5:
        return {"observed_difference": np.nan, "p_value": np.nan, "n_strategy": len(a), "n_random": len(b)}
    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(iterations):
        shuffled = rng.permutation(pooled)
        diff = shuffled[: len(a)].mean() - shuffled[len(a) :].mean()
        if diff >= observed:
            count += 1
    return {
        "observed_difference": observed,
        "p_value": (count + 1) / (iterations + 1),
        "n_strategy": int(len(a)),
        "n_random": int(len(b)),
        "mean_strategy": float(a.mean()),
        "mean_random": float(b.mean()),
    }


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict[str, float]:
    """Probability that the observed Sharpe survives multiple-testing adjustment.

    Bailey & Lopez de Prado's deflated Sharpe: the expected maximum Sharpe from
    ``n_trials`` independent random strategies is subtracted before testing.
    Returns NaN rather than a false number when the inputs are unusable.
    """
    from scipy.stats import norm

    if not np.isfinite(sharpe) or n_observations < 10 or n_trials < 1:
        return {"deflated_sharpe": np.nan, "expected_max_sharpe": np.nan, "p_value": np.nan}
    euler = 0.5772156649
    if n_trials > 1:
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        expected_max = (1 - euler) * z1 + euler * z2
    else:
        expected_max = 0.0
    denominator = np.sqrt(
        max(
            1e-12,
            1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2,
        )
    )
    statistic = (sharpe - expected_max) * np.sqrt(n_observations - 1) / denominator
    return {
        "deflated_sharpe": float(statistic),
        "expected_max_sharpe": float(expected_max),
        "p_value": float(1.0 - norm.cdf(statistic)),
        "n_trials": int(n_trials),
    }


def independent_trade_count(trades: pd.DataFrame, cluster_minutes: int = 240) -> dict[str, int]:
    """Count trade *clusters* as well as trades.

    Ten coins pumping together in one hour is closer to one observation than to
    ten. The cluster count is the number to quote when claiming a sample size.
    """
    if trades.empty:
        return {"n_trades": 0, "n_clusters": 0, "n_markets": 0}
    times = pd.to_datetime(trades["entry_time"], utc=True).sort_values()
    gaps = times.diff().dt.total_seconds().div(60.0)
    clusters = int((gaps.isna() | (gaps > cluster_minutes)).sum())
    return {
        "n_trades": int(len(trades)),
        "n_clusters": clusters,
        "n_markets": int(trades["market"].nunique()),
        "cluster_minutes": cluster_minutes,
    }


def parameter_stability(
    results: pd.DataFrame,
    param_columns: list[str],
    metric: str = "net_expectancy",
    min_trades: int = 30,
) -> pd.DataFrame:
    """Neighbourhood stability of the metric across a parameter grid.

    For each parameter combination, the mean metric of its immediate neighbours
    (one step in any single dimension) is reported. A combination that is good
    only because its neighbours are bad is a curve-fit.
    """
    if results.empty or not param_columns:
        return pd.DataFrame()
    frame = results.copy()
    if "n_trades" in frame.columns:
        frame = frame[frame["n_trades"] >= min_trades]
    if frame.empty:
        return pd.DataFrame()

    grids = {col: sorted(frame[col].dropna().unique().tolist()) for col in param_columns}
    lookup = frame.set_index(param_columns)[metric].to_dict()

    rows = []
    for _, row in frame.iterrows():
        key = tuple(row[c] for c in param_columns)
        neighbours: list[float] = []
        for dim, col in enumerate(param_columns):
            values = grids[col]
            try:
                idx = values.index(row[col])
            except ValueError:
                continue
            for offset in (-1, 1):
                j = idx + offset
                if 0 <= j < len(values):
                    neighbour_key = list(key)
                    neighbour_key[dim] = values[j]
                    value = lookup.get(tuple(neighbour_key))
                    if value is not None and np.isfinite(value):
                        neighbours.append(float(value))
        record = {col: row[col] for col in param_columns}
        record[metric] = row[metric]
        record["n_neighbours"] = len(neighbours)
        record["neighbour_mean"] = float(np.mean(neighbours)) if neighbours else np.nan
        record["neighbour_min"] = float(np.min(neighbours)) if neighbours else np.nan
        record["neighbour_std"] = float(np.std(neighbours)) if len(neighbours) > 1 else np.nan
        record["stability_ratio"] = (
            record["neighbour_mean"] / row[metric]
            if row[metric] not in (0, np.nan) and np.isfinite(row[metric]) and row[metric] != 0
            else np.nan
        )
        record["is_isolated_peak"] = bool(
            neighbours and np.isfinite(row[metric]) and row[metric] > 0 and record["neighbour_mean"] <= 0
        )
        if "n_trades" in row:
            record["n_trades"] = row["n_trades"]
        rows.append(record)
    return pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)


def sensitivity_to_costs(
    run_backtest: Callable[[float], dict[str, Any]],
    cost_multipliers: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
) -> pd.DataFrame:
    """Re-run the backtest with scaled costs and report where the edge dies."""
    rows = []
    for multiplier in cost_multipliers:
        metrics = run_backtest(multiplier)
        metrics["cost_multiplier"] = multiplier
        rows.append(metrics)
    frame = pd.DataFrame(rows)
    if "net_expectancy" in frame.columns:
        positive = frame[frame["net_expectancy"] > 0]["cost_multiplier"]
        frame.attrs["breakeven_cost_multiplier"] = float(positive.max()) if len(positive) else np.nan
    return frame


def degradation_report(
    train: dict[str, Any], validation: dict[str, Any], test: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Side-by-side train / validation / test comparison of the headline metrics."""
    keys = [
        "n_trades", "win_rate", "net_expectancy", "net_expectancy_bps",
        "profit_factor", "max_drawdown", "sharpe", "total_net_pnl_eur",
    ]
    rows = []
    for name, metrics in (("train", train), ("validation", validation), ("test", test)):
        if metrics is None:
            continue
        row = {"split": name}
        row.update({k: metrics.get(k) for k in keys})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) >= 2 and "net_expectancy" in frame.columns:
        base = frame.loc[frame["split"] == "train", "net_expectancy"]
        if len(base) and base.iloc[0] not in (0, None) and np.isfinite(base.iloc[0]) and base.iloc[0] != 0:
            frame["expectancy_vs_train"] = frame["net_expectancy"] / base.iloc[0]
    return frame
