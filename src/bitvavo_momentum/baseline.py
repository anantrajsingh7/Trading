"""Unconditional baselines - the control every event study needs.

Why this module exists
----------------------
Our first study reported a +0.42% mean 60-minute forward return for the best
event specification and treated it as a (failed) edge. That reading was
incomplete: **it was never compared against doing nothing**. If a random entry in
the same markets over the same period also returned +0.4% in 60 minutes, the
event condition contributed exactly zero, and the whole exercise measured market
drift rather than a signal.

Three controls are provided, in increasing strictness:

``unconditional_forward_returns``
    What every bar in the universe returned over the horizon. The widest possible
    reference: "what if I had bought at a uniformly random moment?"

``matched_random_baseline``
    Random entries drawn from the *same markets* and the *same calendar window*
    as the real events, so market composition and period are held fixed and only
    the timing differs. This is the control that isolates the timing claim.

``conditional_lift``
    Event mean minus baseline mean, with a bootstrap confidence interval on the
    difference. This is the number that should have been reported all along.

A strategy has to beat the baseline, not zero. Beating zero is not an edge in a
market that drifts upward; failing to beat zero is not damning in one that drifts
down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .timeutils import interval_to_minutes, to_utc

log = get_logger(__name__)


@dataclass
class BaselineResult:
    horizon_minutes: int
    n_samples: int
    mean_return: float
    median_return: float
    hit_rate: float
    std_return: float
    p05: float
    p95: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _forward_returns_from_frame(
    frame: pd.DataFrame,
    horizon_bars: int,
    entry_column: str = "open",
) -> pd.Series:
    """Return from the NEXT bar's open to the close ``horizon_bars`` later.

    Mirrors the event study exactly: entry at the first realistically executable
    price after the decision bar, exit at the close of the horizon.
    """
    entry = frame[entry_column].shift(-1)
    exit_price = frame["close"].shift(-horizon_bars)
    return (exit_price / entry - 1.0).dropna()


def unconditional_forward_returns(
    candles_by_market: dict[str, pd.DataFrame],
    horizons_minutes: tuple[int, ...] = (15, 30, 60, 120, 240, 480, 1440, 2880),
    interval: str = "1m",
    sample_every: int = 60,
    eligibility_by_market: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Forward-return distribution for entries at arbitrary moments.

    ``sample_every`` thins the sample (default: one observation per hour of
    1-minute data). Overlapping windows are not independent, so the thinning is
    about compute, not statistics - the block bootstrap handles dependence.
    """
    interval_minutes = interval_to_minutes(interval)
    rows: list[dict[str, Any]] = []
    pooled: dict[int, list[np.ndarray]] = {h: [] for h in horizons_minutes}

    for market, frame in candles_by_market.items():
        if frame is None or frame.empty:
            continue
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        data = data.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()

        if eligibility_by_market is not None:
            mask = eligibility_by_market.get(market)
            if mask is not None and not mask.empty:
                aligned = mask.reindex(data.index, method="ffill").fillna(False).astype(bool)
                data = data[aligned]
        if len(data) < 2:
            continue

        for horizon in horizons_minutes:
            bars = max(1, int(round(horizon / interval_minutes)))
            series = _forward_returns_from_frame(data, bars)
            if series.empty:
                continue
            thinned = series.iloc[::sample_every]
            values = thinned.to_numpy(dtype="float64")
            values = values[np.isfinite(values)]
            if values.size:
                pooled[horizon].append(values)

    for horizon, chunks in pooled.items():
        if not chunks:
            continue
        values = np.concatenate(chunks)
        rows.append(
            BaselineResult(
                horizon_minutes=horizon,
                n_samples=int(values.size),
                mean_return=float(values.mean()),
                median_return=float(np.median(values)),
                hit_rate=float((values > 0).mean()),
                std_return=float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                p05=float(np.quantile(values, 0.05)),
                p95=float(np.quantile(values, 0.95)),
                source="unconditional",
            ).to_dict()
        )
    return pd.DataFrame(rows)


def matched_random_baseline(
    events: pd.DataFrame,
    candles_by_market: dict[str, pd.DataFrame],
    horizons_minutes: tuple[int, ...] = (15, 30, 60, 120, 240, 480, 1440, 2880),
    interval: str = "1m",
    draws_per_event: int = 5,
    seed: int = 20260802,
    time_column: str = "event_time",
) -> pd.DataFrame:
    """Random entries matched to the events' markets and calendar window.

    For each real event, ``draws_per_event`` random timestamps are drawn from the
    same market, uniformly within the overall event period. Market composition,
    sample period and horizon are therefore identical to the event set; the only
    difference is *when* the entry happened.

    This isolates exactly the claim under test: does the event timing carry
    information beyond being invested in that coin at all?
    """
    if events.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    interval_minutes = interval_to_minutes(interval)
    times = pd.to_datetime(events[time_column], utc=True)
    window_start, window_end = times.min(), times.max()

    prepared: dict[str, pd.DataFrame] = {}
    for market, frame in candles_by_market.items():
        if frame is None or frame.empty:
            continue
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        data = data.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
        prepared[market] = data[(data.index >= window_start) & (data.index <= window_end)]

    counts = events["market"].value_counts().to_dict()
    pooled: dict[int, list[float]] = {h: [] for h in horizons_minutes}

    for market, n_events in counts.items():
        data = prepared.get(market)
        if data is None or len(data) < 10:
            continue
        n_draws = int(n_events * draws_per_event)
        max_bars = max(horizons_minutes) // interval_minutes + 2
        upper = len(data) - max_bars
        if upper <= 1:
            continue
        positions = rng.integers(0, upper, size=n_draws)

        opens = data["open"].to_numpy(dtype="float64")
        closes = data["close"].to_numpy(dtype="float64")
        for horizon in horizons_minutes:
            bars = max(1, int(round(horizon / interval_minutes)))
            entry = opens[positions + 1]
            exit_price = closes[np.minimum(positions + bars, len(closes) - 1)]
            with np.errstate(divide="ignore", invalid="ignore"):
                returns = exit_price / entry - 1.0
            returns = returns[np.isfinite(returns)]
            pooled[horizon].extend(returns.tolist())

    rows = []
    for horizon, values in pooled.items():
        if not values:
            continue
        arr = np.asarray(values, dtype="float64")
        rows.append(
            BaselineResult(
                horizon_minutes=horizon,
                n_samples=int(arr.size),
                mean_return=float(arr.mean()),
                median_return=float(np.median(arr)),
                hit_rate=float((arr > 0).mean()),
                std_return=float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
                p05=float(np.quantile(arr, 0.05)),
                p95=float(np.quantile(arr, 0.95)),
                source="matched_random",
            ).to_dict()
        )
    return pd.DataFrame(rows)


def conditional_lift(
    event_study: pd.DataFrame,
    baseline: pd.DataFrame,
    round_trip_cost_bps: float = 77.0,
    iterations: int = 2000,
    seed: int = 20260802,
) -> pd.DataFrame:
    """Event mean minus baseline mean, per horizon, with the cost bar applied.

    Three columns decide whether anything is tradeable:

    ``lift``
        event mean - baseline mean. Positive means the *timing* added something.
    ``lift_net_bps``
        the same, minus round-trip costs. This is the tradeable quantity.
    ``beats_cost``
        whether the conditional lift alone clears costs.

    A strategy can have a positive mean return and zero lift (it is just buying
    the market), or positive lift and negative net (real information, too small
    to trade). Both are common and both are worth knowing.
    """
    if event_study.empty or baseline.empty:
        return pd.DataFrame()

    events = event_study.groupby("horizon_minutes").agg(
        event_mean=("mean_return", "mean"),
        event_median=("median_return", "mean"),
        event_hit_rate=("hit_rate", "mean"),
        n_events=("n_events", "sum"),
    ).reset_index()

    base = baseline[["horizon_minutes", "mean_return", "median_return", "hit_rate", "n_samples"]].rename(
        columns={
            "mean_return": "baseline_mean",
            "median_return": "baseline_median",
            "hit_rate": "baseline_hit_rate",
            "n_samples": "n_baseline",
        }
    )

    merged = events.merge(base, on="horizon_minutes", how="inner")
    merged["lift"] = merged["event_mean"] - merged["baseline_mean"]
    merged["lift_bps"] = merged["lift"] * 10_000
    merged["lift_net_bps"] = merged["lift_bps"] - round_trip_cost_bps
    merged["beats_cost"] = merged["lift_net_bps"] > 0
    merged["hit_rate_lift"] = merged["event_hit_rate"] - merged["baseline_hit_rate"]
    merged["gross_net_bps"] = merged["event_mean"] * 10_000 - round_trip_cost_bps
    return merged.sort_values("horizon_minutes").reset_index(drop=True)


def lift_confidence_interval(
    event_returns,
    baseline_returns,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260802,
) -> dict[str, float]:
    """Bootstrap confidence interval on the difference of two means.

    Both samples are resampled independently, which is the right null for
    "different timing, same market and period".
    """
    a = np.asarray(pd.Series(event_returns).dropna(), dtype="float64")
    b = np.asarray(pd.Series(baseline_returns).dropna(), dtype="float64")
    if a.size < 5 or b.size < 5:
        return {"lift": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value_lift_gt_zero": float("nan"), "n_event": int(a.size), "n_baseline": int(b.size)}

    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype="float64")
    for i in range(iterations):
        draws[i] = rng.choice(a, a.size, replace=True).mean() - rng.choice(b, b.size, replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "lift": float(a.mean() - b.mean()),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1 - alpha)),
        "p_value_lift_gt_zero": float((draws <= 0).mean()),
        "n_event": int(a.size),
        "n_baseline": int(b.size),
    }


def buy_and_hold_benchmark(
    candles_by_market: dict[str, pd.DataFrame],
    markets: tuple[str, ...] = ("BTC-EUR", "ETH-EUR"),
    starting_equity: float = 10_000.0,
) -> pd.DataFrame:
    """Buy-and-hold reference over the available window.

    Any active strategy must beat this after costs and after the operator's time,
    and frequently does not.
    """
    rows = []
    for market in markets:
        frame = candles_by_market.get(market)
        if frame is None or frame.empty:
            continue
        data = frame.dropna(subset=["close"]).sort_values("timestamp")
        if len(data) < 2:
            continue
        first, last = float(data["close"].iloc[0]), float(data["close"].iloc[-1])
        start_ts, end_ts = to_utc(data["timestamp"].iloc[0]), to_utc(data["timestamp"].iloc[-1])
        days = max(1.0, (end_ts - start_ts).total_seconds() / 86400.0)
        total_return = last / first - 1.0
        daily = data.set_index("timestamp")["close"].resample("1D").last().dropna()
        returns = daily.pct_change().dropna()
        equity = starting_equity * (daily / daily.iloc[0])
        drawdown = float((equity / equity.cummax() - 1.0).min()) if len(equity) else float("nan")
        rows.append(
            {
                "market": market,
                "start": start_ts,
                "end": end_ts,
                "days": days,
                "total_return": total_return,
                "annualised_return": (1.0 + total_return) ** (365.0 / days) - 1.0 if total_return > -1 else float("nan"),
                "max_drawdown": drawdown,
                "volatility_annualised": float(returns.std(ddof=1) * np.sqrt(365)) if len(returns) > 1 else float("nan"),
                "final_equity_eur": starting_equity * (1.0 + total_return),
                "n_round_trips": 1,
                "source": "buy_and_hold",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "BaselineResult",
    "buy_and_hold_benchmark",
    "conditional_lift",
    "lift_confidence_interval",
    "matched_random_baseline",
    "unconditional_forward_returns",
]
