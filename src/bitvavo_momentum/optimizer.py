"""Parameter search, selection rules and feature ablation.

The optimizer is intentionally boring: an exhaustive grid over a small parameter
space, scored on **training data only**, with a selection rule that prefers
robust plateaus over isolated peaks.

Selection rule (in order):

1. discard combinations with too few trades;
2. discard combinations whose neighbours perform badly (isolated peaks);
3. among the rest, rank by net expectancy penalised by neighbourhood dispersion;
4. break ties toward the *simpler* configuration - fewer filters, rounder
   parameters.

Every call records how many configurations were evaluated. That number feeds the
deflated Sharpe ratio: trying 400 combinations and reporting the best one is a
different claim from testing a single hypothesis, and the report says so.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .metrics import compute_trade_metrics
from .robustness import parameter_stability

log = get_logger(__name__)


@dataclass
class SearchResult:
    table: pd.DataFrame
    best: dict[str, Any] | None
    n_configurations: int
    selection_notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return self.table


def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    """Cartesian product of named parameter axes."""
    keys = list(axes)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(axes[k] for k in keys))]


def evaluate_grid(
    configurations: Iterable[dict[str, Any]],
    run: Callable[[dict[str, Any]], pd.DataFrame],
    starting_equity: float = 10_000.0,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Run every configuration and collect its metrics.

    ``run`` receives a configuration dict and must return a **trade table**.
    Keeping metric computation here (rather than inside ``run``) guarantees every
    configuration is scored by exactly the same code.
    """
    rows: list[dict[str, Any]] = []
    configurations = list(configurations)
    for i, config in enumerate(configurations, start=1):
        trades = run(config)
        metrics = compute_trade_metrics(trades, starting_equity)
        metrics.update(config)
        rows.append(metrics)
        if progress_every and i % progress_every == 0:
            log.info("Evaluated %d/%d configurations", i, len(configurations))
    return pd.DataFrame(rows)


def select_robust(
    results: pd.DataFrame,
    param_columns: list[str],
    metric: str = "net_expectancy",
    min_trades: int = 50,
    dispersion_penalty: float = 0.5,
    simplicity_columns: list[str] | None = None,
) -> SearchResult:
    """Pick a configuration that its neighbours also support."""
    notes: list[str] = []
    n_configs = int(len(results))
    if results.empty:
        return SearchResult(results, None, 0, ["no configurations evaluated"])

    viable = results[results["n_trades"] >= min_trades].copy()
    notes.append(f"{len(viable)}/{n_configs} configurations had at least {min_trades} trades")
    if viable.empty:
        return SearchResult(results, None, n_configs, notes + ["nothing met the minimum trade count"])

    stability = parameter_stability(viable, param_columns, metric=metric, min_trades=min_trades)
    if stability.empty:
        return SearchResult(results, None, n_configs, notes + ["stability table could not be built"])

    merged = viable.merge(
        stability[param_columns + ["neighbour_mean", "neighbour_std", "neighbour_min", "is_isolated_peak"]],
        on=param_columns,
        how="left",
    )

    non_isolated = merged[~merged["is_isolated_peak"].fillna(False)]
    if non_isolated.empty:
        notes.append("every positive configuration was an isolated peak - treating all as unreliable")
        non_isolated = merged
    else:
        notes.append(f"{len(merged) - len(non_isolated)} isolated peaks discarded")

    scored = non_isolated.copy()
    scored["robust_score"] = scored[metric].astype(float) - dispersion_penalty * scored[
        "neighbour_std"
    ].fillna(0.0).astype(float)
    scored = scored.sort_values(["robust_score", "n_trades"], ascending=[False, False])

    if simplicity_columns:
        top_score = scored["robust_score"].iloc[0]
        close = scored[scored["robust_score"] >= top_score * 0.95] if top_score > 0 else scored.head(1)
        if len(close) > 1:
            close = close.assign(
                _complexity=close[simplicity_columns].notna().sum(axis=1)
            ).sort_values(["_complexity", "robust_score"], ascending=[True, False])
            notes.append("tie broken toward the simpler configuration")
            scored = close

    best = scored.iloc[0].to_dict()
    notes.append(
        f"selected {({k: best.get(k) for k in param_columns})} with {metric}={best.get(metric):.5f} "
        f"on {int(best.get('n_trades', 0))} trades"
    )
    return SearchResult(scored.reset_index(drop=True), best, n_configs, notes)


def feature_ablation(
    base_filters: dict[str, tuple[str, float]],
    run_with_filters: Callable[[dict[str, tuple[str, float]]], pd.DataFrame],
    starting_equity: float = 10_000.0,
    metric: str = "net_expectancy",
) -> pd.DataFrame:
    """Remove one filter at a time and measure the damage.

    A filter that can be removed without hurting out-of-sample performance is not
    earning its complexity and should go. This is the mechanism behind the
    Strategy H complexity cap.
    """
    rows: list[dict[str, Any]] = []

    full = compute_trade_metrics(run_with_filters(base_filters), starting_equity)
    rows.append({"removed_filter": "(none - full model)", **full})

    for name in base_filters:
        reduced = {k: v for k, v in base_filters.items() if k != name}
        metrics = compute_trade_metrics(run_with_filters(reduced), starting_equity)
        metrics["removed_filter"] = name
        metrics["delta_vs_full"] = metrics.get(metric, np.nan) - full.get(metric, np.nan)
        rows.append(metrics)

    if len(base_filters) > 0:
        empty = compute_trade_metrics(run_with_filters({}), starting_equity)
        empty["removed_filter"] = "(all filters removed)"
        empty["delta_vs_full"] = empty.get(metric, np.nan) - full.get(metric, np.nan)
        rows.append(empty)

    frame = pd.DataFrame(rows)
    lead = ["removed_filter", "n_trades", metric, "delta_vs_full", "win_rate", "profit_factor"]
    ordered = [c for c in lead if c in frame.columns] + [c for c in frame.columns if c not in lead]
    return frame[ordered]


def heatmap_frame(
    results: pd.DataFrame, x: str, y: str, metric: str = "net_expectancy"
) -> pd.DataFrame:
    """Pivot a result table into a 2-D grid for the parameter heatmap."""
    if results.empty or x not in results.columns or y not in results.columns:
        return pd.DataFrame()
    return results.pivot_table(index=y, columns=x, values=metric, aggfunc="mean")


def count_trials(*tables: pd.DataFrame) -> int:
    """Total configurations evaluated - the multiple-testing denominator."""
    return int(sum(len(t) for t in tables if t is not None and not t.empty))
