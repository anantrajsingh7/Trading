"""Milestone 5: separate entry quality from exit choice.

Why this module exists
----------------------
The family comparison ran every entry family through a *single* exit policy
(``tp10_atr1.5_t48h``) and every one of the 16 variants lost money. The win
rates were 5.1%-8.5%, which is not the signature of a coin flip that lost - it
is the signature of a take-profit that was almost never reached. That leaves two
completely different explanations on the table:

1. the entries carry no information, and no exit rule can rescue them; or
2. the entries carry some information, and a +10% target inside 48 hours was
   simply the wrong way to harvest it.

Reporting "family rejected" without separating those two is how a study throws
away a working signal because of an arbitrary exit parameter - and equally how it
talks itself into re-optimising exits forever on entries that were never alive.

The separation is done in two stages, cheap test first.

Stage 1 - exit-independent (``signal_outcomes``, ``forward_return_table``,
``excursion_table``). What did price do after the signal, with *no* exit rule at
all? A buy-and-hold-to-horizon return is the upper bound on what any
non-clairvoyant exit rule can extract on average, because an exit rule can only
redistribute that path, not add to it. If the mean forward return is below the
round-trip cost at *every* horizon, explanation 2 is dead and no exit grid is
worth running. The excursion profile answers the companion question directly:
what fraction of signals ever traded +10% in favour, and how far did they go
against first?

Stage 2 - the exit grid, run only on families that survive stage 1.

Forward-looking columns are prefixed ``fwd_`` so ``assert_no_forward_columns``
catches them if one ever leaks into a feature set. They are outcome
measurements, never inputs to a decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .timeutils import interval_to_minutes

log = get_logger(__name__)

# Excursion levels to report, as fractions. The +10% level is the one the family
# comparison used as its take-profit.
DEFAULT_LEVELS: tuple[float, ...] = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15)

# The holding periods the spec requires comparing.
DEFAULT_HORIZONS: tuple[int, ...] = (6 * 60, 12 * 60, 24 * 60, 36 * 60, 48 * 60)


def _indexed(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a timestamp-indexed, de-duplicated, sorted copy."""
    if isinstance(frame.index, pd.DatetimeIndex):
        data = frame[~frame.index.duplicated(keep="last")].sort_index()
        return data
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()


def signal_outcomes(
    signals: pd.DataFrame,
    features_by_market: dict[str, pd.DataFrame],
    horizons_minutes: tuple[int, ...] = DEFAULT_HORIZONS,
    interval: str = "15m",
    max_holding_minutes: int | None = None,
    time_column: str = "event_time",
) -> pd.DataFrame:
    """One row per signal: forward returns at each horizon, plus MFE and MAE.

    Entry is the open of the bar *after* the signal bar, matching the backtester
    and the event study. A signal closer to the end of its market's history than
    the horizon gets ``NaN`` rather than a truncated window: clamping the exit to
    the last available bar would silently report a short holding period as if it
    had run the full course, which flatters late signals in a falling market.
    """
    if signals.empty:
        return pd.DataFrame()

    interval_minutes = interval_to_minutes(interval)
    hold_minutes = max_holding_minutes or max(horizons_minutes)
    hold_bars = max(1, int(round(hold_minutes / interval_minutes)))
    horizon_bars = {h: max(1, int(round(h / interval_minutes))) for h in horizons_minutes}

    chunks: list[pd.DataFrame] = []
    for market, group in signals.groupby("market"):
        frame = features_by_market.get(market)
        if frame is None or frame.empty:
            continue
        data = _indexed(frame)
        n = len(data)
        if n < hold_bars + 2:
            continue

        times = pd.to_datetime(group[time_column], utc=True)
        positions = data.index.get_indexer(times)
        entry_pos = positions + 1
        known = (positions >= 0) & (entry_pos < n)
        if not known.any():
            continue

        opens = data["open"].to_numpy(dtype="float64")
        closes = data["close"].to_numpy(dtype="float64")
        highs = pd.Series(data["high"].to_numpy(dtype="float64"))
        lows = pd.Series(data["low"].to_numpy(dtype="float64"))

        # Max high / min low over [i, i + hold_bars - 1], NaN where the window
        # runs past the end of the data.
        window_high = highs.rolling(hold_bars).max().shift(-(hold_bars - 1)).to_numpy()
        window_low = lows.rolling(hold_bars).min().shift(-(hold_bars - 1)).to_numpy()

        safe_pos = np.where(known, entry_pos, 0)
        entry_price = np.where(known, opens[safe_pos], np.nan)
        entry_price = np.where(np.isfinite(entry_price) & (entry_price > 0), entry_price, np.nan)

        out = pd.DataFrame(
            {
                "market": market,
                "event_spec": group["event_spec"].to_numpy(),
                "event_time": times.to_numpy(),
                "entry_price": entry_price,
            }
        )
        for horizon, bars in horizon_bars.items():
            exit_pos = entry_pos + bars
            usable = known & (exit_pos < n)
            safe_exit = np.where(usable, np.where(exit_pos < n, exit_pos, 0), 0)
            values = np.where(usable, closes[safe_exit] / entry_price - 1.0, np.nan)
            out[f"fwd_return_{horizon}m"] = values

        out[f"fwd_mfe_{hold_minutes}m"] = window_high[safe_pos] / entry_price - 1.0
        out[f"fwd_mae_{hold_minutes}m"] = window_low[safe_pos] / entry_price - 1.0
        out.loc[~known, out.columns.difference(["market", "event_spec", "event_time"])] = np.nan
        chunks.append(out)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).sort_values("event_time").reset_index(drop=True)


def forward_return_table(
    outcomes: pd.DataFrame,
    horizons_minutes: tuple[int, ...] = DEFAULT_HORIZONS,
    round_trip_cost_bps: float = 77.0,
    group_column: str = "event_spec",
) -> pd.DataFrame:
    """Per family and horizon: what a signal was worth with no exit rule at all.

    ``net_mean`` subtracts the round-trip cost. ``beats_cost`` is the only column
    that matters at this stage: if it is False everywhere for a family, that
    family's entries did not pay for their own execution over any holding period
    the spec allows, and the exit policy was not the reason it lost.
    """
    if outcomes.empty:
        return pd.DataFrame()

    cost = round_trip_cost_bps * 1e-4
    rows: list[dict[str, Any]] = []
    for name, group in outcomes.groupby(group_column):
        for horizon in horizons_minutes:
            column = f"fwd_return_{horizon}m"
            if column not in group.columns:
                continue
            values = group[column].to_numpy(dtype="float64")
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            mean = float(values.mean())
            rows.append(
                {
                    group_column: name,
                    "horizon_minutes": horizon,
                    "horizon_hours": horizon / 60.0,
                    "n_signals": int(values.size),
                    "gross_mean": mean,
                    "gross_median": float(np.median(values)),
                    "hit_rate": float((values > 0).mean()),
                    "hit_rate_after_cost": float((values > cost).mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                    "p05": float(np.quantile(values, 0.05)),
                    "p95": float(np.quantile(values, 0.95)),
                    "net_mean": mean - cost,
                    "beats_cost": bool(mean > cost),
                    # The round-trip cost at which this family would break even.
                    # Compare it against the fee schedule you can actually get:
                    # it says how much cheaper execution would have to become
                    # before the signal is worth trading, which is a far more
                    # actionable number than "negative".
                    "breakeven_cost_bps": mean * 1e4,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["net_mean"], ascending=False).reset_index(drop=True)


def excursion_table(
    outcomes: pd.DataFrame,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    max_holding_minutes: int = 48 * 60,
    group_column: str = "event_spec",
) -> pd.DataFrame:
    """How far price ran in favour, and against, within the holding window.

    ``reach_up_X`` is the fraction of signals whose high ever touched +X%. It is
    an upper bound on the hit rate of a take-profit at that level - the real hit
    rate is lower, because a stop may fire first. A take-profit whose ``reach_up``
    is a few percent cannot produce winners no matter how good the entry is, and
    a family tested only at that level has not really been tested.
    """
    if outcomes.empty:
        return pd.DataFrame()

    mfe_column = f"fwd_mfe_{max_holding_minutes}m"
    mae_column = f"fwd_mae_{max_holding_minutes}m"
    if mfe_column not in outcomes.columns or mae_column not in outcomes.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for name, group in outcomes.groupby(group_column):
        mfe = group[mfe_column].to_numpy(dtype="float64")
        mae = group[mae_column].to_numpy(dtype="float64")
        usable = np.isfinite(mfe) & np.isfinite(mae)
        mfe, mae = mfe[usable], mae[usable]
        if mfe.size == 0:
            continue
        row: dict[str, Any] = {
            group_column: name,
            "n_signals": int(mfe.size),
            "median_mfe": float(np.median(mfe)),
            "median_mae": float(np.median(mae)),
            "mean_mfe": float(mfe.mean()),
            "mean_mae": float(mae.mean()),
        }
        # A ratio below 1 means the average signal went further against than for.
        row["mfe_mae_ratio"] = float(mfe.mean() / abs(mae.mean())) if mae.mean() != 0 else float("nan")
        for level in levels:
            tag = f"{level * 100:.0f}"
            row[f"reach_up_{tag}pct"] = float((mfe >= level).mean())
            row[f"reach_down_{tag}pct"] = float((mae <= -level).mean())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("median_mfe", ascending=False).reset_index(drop=True)


def exit_reason_breakdown(
    trades: pd.DataFrame,
    group_columns: tuple[str, ...] = ("event_spec",),
) -> pd.DataFrame:
    """Which exit actually closed each trade, and what it earned.

    A family whose trades are 90% time-stops has not been tested against its
    stop and target at all - it has been tested against the clock.
    """
    if trades is None or trades.empty or "exit_reason" not in trades.columns:
        return pd.DataFrame()

    closed = trades[trades["exit_reason"].notna()].copy()
    if closed.empty:
        return pd.DataFrame()

    groups = [c for c in group_columns if c in closed.columns]
    keys = [*groups, "exit_reason"]

    if "net_return" in closed.columns:
        closed["net_pnl_pct"] = closed["net_return"]
    elif {"net_pnl_eur", "notional_eur"} <= set(closed.columns):
        closed["net_pnl_pct"] = closed["net_pnl_eur"] / closed["notional_eur"].replace(0, np.nan)

    aggregations: dict[str, tuple[str, str]] = {"n_trades": ("exit_reason", "size")}
    if "net_pnl_pct" in closed.columns:
        aggregations["mean_net_pct"] = ("net_pnl_pct", "mean")
    if "net_pnl_eur" in closed.columns:
        aggregations["total_net_eur"] = ("net_pnl_eur", "sum")
    if "holding_minutes" in closed.columns:
        aggregations["median_holding_minutes"] = ("holding_minutes", "median")

    aggregated = closed.groupby(keys).agg(**aggregations).reset_index()

    totals = aggregated.groupby(groups)["n_trades"].transform("sum") if groups else aggregated["n_trades"].sum()
    aggregated["share"] = aggregated["n_trades"] / totals
    sort_keys = [*groups, "n_trades"]
    return aggregated.sort_values(sort_keys, ascending=[True] * len(groups) + [False]).reset_index(drop=True)


def stage_one_verdict(
    forward: pd.DataFrame,
    excursions: pd.DataFrame,
    take_profit_pct: float = 0.10,
    group_column: str = "event_spec",
) -> pd.DataFrame:
    """Per family: does an exit grid have anything to work with?

    ``survives`` is deliberately generous - it asks only whether *some* holding
    period had a positive mean net return, not whether that return was
    significant. A family that fails this cannot be saved by exit tuning; a
    family that passes it has earned stage 2, and nothing more.
    """
    if forward.empty:
        return pd.DataFrame()

    tag = f"{take_profit_pct * 100:.0f}"
    reach_column = f"reach_up_{tag}pct"
    rows: list[dict[str, Any]] = []
    for name, group in forward.groupby(group_column):
        best = group.loc[group["net_mean"].idxmax()]
        row = {
            group_column: name,
            "n_signals": int(best["n_signals"]),
            "best_horizon_hours": float(best["horizon_hours"]),
            "best_gross_mean": float(best["gross_mean"]),
            "best_net_mean": float(best["net_mean"]),
            "breakeven_cost_bps": float(best["breakeven_cost_bps"]),
            "horizons_tested": int(len(group)),
            "survives": bool(best["net_mean"] > 0),
        }
        if not excursions.empty and reach_column in excursions.columns:
            match = excursions[excursions[group_column] == name]
            if not match.empty:
                row[f"reach_tp_{tag}pct"] = float(match.iloc[0][reach_column])
                row["median_mfe"] = float(match.iloc[0]["median_mfe"])
                row["median_mae"] = float(match.iloc[0]["median_mae"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("best_net_mean", ascending=False).reset_index(drop=True)
