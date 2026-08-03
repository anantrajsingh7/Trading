"""Spec Strategy 7: relative-strength rotation.

Structurally different from every other family here. The others ask *"is this
market setting up?"* one market at a time. This one asks *"which markets are
strongest right now?"* and takes only the leaders - a cross-sectional question
that needs the whole universe aligned on a common clock.

Why it is worth testing after the impulse families failed
---------------------------------------------------------
The event study showed that conditioning on a market's *own* recent return is
uninformative-to-harmful: forward returns degrade monotonically with the size
and speed of the impulse. Cross-sectional ranking asks a different question -
not "did this move a lot?" but "did this move more than its peers, per unit of
risk?" - and it rebalances on a schedule rather than on every trigger, which
cuts turnover. At 77 bps a round trip, turnover is the binding constraint:
rebalancing weekly costs roughly a quarter of what rebalancing on every signal
does.

Look-ahead control
------------------
Ranking at time ``t`` uses only bars that closed at or before ``t``. The panel
is built by aligning every market onto a common index and forward-filling
**past** values only; a market with no bar yet at ``t`` is excluded from the
ranking rather than back-filled, so the universe grows as coins list rather
than existing retroactively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .timeutils import session_bucket

log = get_logger(__name__)


@dataclass
class RotationConfig:
    """Ranking and rebalancing rules."""

    lookback_minutes: int = 24 * 60
    volatility_lookback_minutes: int = 72 * 60
    rebalance_minutes: int = 24 * 60
    top_n: int | None = 3
    top_fraction: float | None = None      # alternative to top_n: e.g. 0.10 = top 10%
    risk_adjust: bool = True               # divide momentum by trailing volatility
    skip_recent_minutes: int = 60          # exclude the most recent bars from the signal
    min_universe_size: int = 8
    buffer_rank_multiple: float = 2.0      # hold until rank falls beyond top_n * this
    min_quote_volume_24h: float = 25_000.0
    max_spread_proxy_bps: float = 60.0
    require_positive_momentum: bool = True

    def label(self) -> str:
        selector = f"top{self.top_n}" if self.top_n else f"top{self.top_fraction:.0%}"
        adjust = "riskadj" if self.risk_adjust else "raw"
        return (
            f"S7_rotation_{self.lookback_minutes // 60}h_{adjust}_{selector}"
            f"_rb{self.rebalance_minutes // 60}h"
        )


def build_panel(
    features_by_market: dict[str, pd.DataFrame],
    column: str = "close",
    freq_minutes: int = 60,
) -> pd.DataFrame:
    """Align every market onto a common time grid.

    Forward-fill carries the last *observed* value forward, which is legitimate
    (it is the last known price). Markets with no observation yet remain NaN and
    are excluded from ranking at that timestamp - they had not listed.
    """
    series: dict[str, pd.Series] = {}
    for market, frame in features_by_market.items():
        if frame is None or frame.empty or column not in frame.columns:
            continue
        s = frame[column]
        if not isinstance(s.index, pd.DatetimeIndex):
            continue
        series[market] = s[~s.index.duplicated(keep="last")].sort_index()
    if not series:
        return pd.DataFrame()

    panel = pd.DataFrame(series).sort_index()
    grid = panel.resample(f"{freq_minutes}min").last()
    return grid.ffill()


def rank_universe(
    close_panel: pd.DataFrame,
    config: RotationConfig,
    freq_minutes: int = 60,
    volume_panel: pd.DataFrame | None = None,
    spread_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cross-sectional score per market per timestamp. Higher is stronger.

    ``skip_recent_minutes`` deliberately excludes the most recent bars from the
    momentum window. Short-horizon reversal contaminates a raw ranking - the
    event study measured exactly that effect - so the signal is the return up to
    a short lag, not up to the instant of ranking.
    """
    if close_panel.empty:
        return pd.DataFrame()

    bars = max(1, int(round(config.lookback_minutes / freq_minutes)))
    skip = max(0, int(round(config.skip_recent_minutes / freq_minutes)))
    vol_bars = max(2, int(round(config.volatility_lookback_minutes / freq_minutes)))

    lagged = close_panel.shift(skip)
    momentum = lagged / lagged.shift(bars) - 1.0

    score = momentum
    if config.risk_adjust:
        volatility = close_panel.pct_change().rolling(vol_bars, min_periods=max(3, vol_bars // 3)).std()
        score = momentum / volatility.replace(0.0, np.nan)

    eligible = score.notna()
    if volume_panel is not None and config.min_quote_volume_24h:
        eligible &= volume_panel.reindex_like(score).fillna(0.0) >= config.min_quote_volume_24h
    if spread_panel is not None and config.max_spread_proxy_bps:
        eligible &= spread_panel.reindex_like(score).fillna(np.inf) <= config.max_spread_proxy_bps
    if config.require_positive_momentum:
        eligible &= momentum > 0

    score = score.where(eligible)
    ranks = score.rank(axis=1, ascending=False, method="first")
    ranks = ranks.where(score.notna())

    universe_size = score.notna().sum(axis=1)
    ranks = ranks.where(universe_size >= config.min_universe_size, other=np.nan)
    return ranks


def target_holdings(
    ranks: pd.DataFrame,
    config: RotationConfig,
    freq_minutes: int = 60,
) -> pd.DataFrame:
    """Boolean holdings matrix after rebalancing and hysteresis.

    Two mechanisms keep turnover down, both of which matter more than the
    ranking rule itself at 77 bps a round trip:

    * rebalancing happens only every ``rebalance_minutes``; between rebalances
      the book is held;
    * a held position is retained until its rank falls beyond
      ``top_n * buffer_rank_multiple``, so a coin oscillating around rank 3 is
      not bought and sold repeatedly.
    """
    if ranks.empty:
        return pd.DataFrame()

    step = max(1, int(round(config.rebalance_minutes / freq_minutes)))
    rebalance_points = ranks.index[::step]

    holdings = pd.DataFrame(False, index=ranks.index, columns=ranks.columns)
    current: set[str] = set()

    for timestamp in ranks.index:
        if timestamp in rebalance_points:
            row = ranks.loc[timestamp]
            available = row.dropna()
            if available.empty:
                current = set()
            else:
                if config.top_n is not None:
                    n = config.top_n
                else:
                    n = max(1, int(round(len(available) * (config.top_fraction or 0.1))))
                incoming = set(available[available <= n].index)
                keep_threshold = n * config.buffer_rank_multiple
                retained = {m for m in current if m in available.index and available[m] <= keep_threshold}
                # Retained names have priority; new entries fill the remaining slots.
                current = set(list(retained)[:n])
                for market in sorted(incoming, key=lambda m: available[m]):
                    if len(current) >= n:
                        break
                    current.add(market)
        if current:
            holdings.loc[timestamp, sorted(current)] = True
    return holdings


def holdings_to_signals(
    holdings: pd.DataFrame,
    features_by_market: dict[str, pd.DataFrame],
    config: RotationConfig,
) -> pd.DataFrame:
    """Convert a holdings matrix into backtester-compatible entry signals.

    One signal per (market, entry) transition - the bar a market *enters* the
    book. Exits are handled by the exit policy and the rebalance horizon, which
    keeps this family comparable with every other family in the same engine
    rather than needing a bespoke portfolio simulator.
    """
    if holdings.empty:
        return pd.DataFrame()

    entries = holdings & ~holdings.shift(1, fill_value=False)
    rows: list[pd.DataFrame] = []
    label = config.label()

    for market in holdings.columns:
        timestamps = entries.index[entries[market].to_numpy()]
        if len(timestamps) == 0:
            continue
        features = features_by_market.get(market)
        if features is None or features.empty:
            continue
        # Snap each rotation timestamp to the last feature bar at or before it.
        positions = features.index.searchsorted(timestamps, side="right") - 1
        positions = positions[positions >= 0]
        if len(positions) == 0:
            continue
        snapped = features.iloc[np.unique(positions)].copy()
        snapped.insert(0, "market", market)
        snapped.insert(1, "event_time", snapped.index)
        snapped["event_spec"] = label
        snapped["event_family"] = "rotation"
        snapped["event_lookback_return"] = snapped.get(
            "dist_ema_20", pd.Series(np.nan, index=snapped.index)
        )
        snapped["session_bucket"] = [session_bucket(ts) for ts in snapped.index]
        rows.append(snapped.reset_index(drop=True))

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True).sort_values(["event_time", "market"])
    log.info("Rotation %s produced %d entry signals", label, len(combined))
    return combined.reset_index(drop=True)


@dataclass
class RotationStrategy:
    """End-to-end rotation: panel, ranking, holdings, signals."""

    config: RotationConfig = field(default_factory=RotationConfig)
    freq_minutes: int = 60

    @property
    def name(self) -> str:
        return self.config.label()

    @property
    def family(self) -> str:
        return "rotation"

    def generate(self, features_by_market: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close_panel = build_panel(features_by_market, "close", self.freq_minutes)
        if close_panel.empty:
            return pd.DataFrame()
        volume_panel = build_panel(features_by_market, "quote_volume_24h", self.freq_minutes)
        spread_panel = build_panel(features_by_market, "spread_proxy_bps", self.freq_minutes)

        ranks = rank_universe(
            close_panel, self.config, self.freq_minutes,
            volume_panel if not volume_panel.empty else None,
            spread_panel if not spread_panel.empty else None,
        )
        holdings = target_holdings(ranks, self.config, self.freq_minutes)
        return holdings_to_signals(holdings, features_by_market, self.config)

    def turnover(
        self,
        features_by_market: dict[str, pd.DataFrame],
        round_trip_cost_bps: float = 77.0,
    ) -> dict[str, Any]:
        """Diagnostic: how often does the book change, and what does that cost?

        The cost figure is expressed as a fraction of **total portfolio capital**,
        which requires dividing by the number of concurrent positions: a round
        trip in one of three equally-weighted slots costs the round-trip rate on
        a third of capital, not on all of it. Reporting the undivided figure
        overstates the drag by the position count and would make every
        multi-position variant look far worse than it is.
        """
        close_panel = build_panel(features_by_market, "close", self.freq_minutes)
        if close_panel.empty:
            return {}
        ranks = rank_universe(close_panel, self.config, self.freq_minutes)
        holdings = target_holdings(ranks, self.config, self.freq_minutes)
        if holdings.empty:
            return {}

        changes = int((holdings != holdings.shift(1, fill_value=False)).sum().sum())
        days = max(1.0, (holdings.index[-1] - holdings.index[0]).total_seconds() / 86400.0)
        round_trips_per_year = changes / 2.0 / days * 365.0
        avg_positions = float(holdings.sum(axis=1).mean())

        cost_rate = round_trip_cost_bps * 1e-4
        position_weight = 1.0 / avg_positions if avg_positions > 0 else 0.0
        return {
            "strategy": self.name,
            "n_position_changes": changes,
            "days": days,
            "round_trips_per_year": round_trips_per_year,
            "avg_positions_held": avg_positions,
            "turns_per_slot_per_year": round_trips_per_year * position_weight,
            "annual_cost_drag": round_trips_per_year * cost_rate * position_weight,
            "round_trip_cost_bps": round_trip_cost_bps,
        }

    def describe(self) -> dict[str, Any]:
        return {"strategy": self.name, "family": self.family, **self.config.__dict__}


def default_rotation_strategies() -> list[RotationStrategy]:
    """A small grid over the choices that actually matter economically.

    Lookback (how far back momentum is measured), rebalance frequency (which
    sets turnover, hence cost) and breadth (how concentrated). Deliberately not
    a sweep over every parameter - eight variants keeps the multiple-testing
    correction meaningful.
    """
    variants: list[RotationStrategy] = []
    for lookback_hours in (12, 24, 72, 168):
        variants.append(
            RotationStrategy(RotationConfig(
                lookback_minutes=lookback_hours * 60,
                rebalance_minutes=24 * 60,
                top_n=3,
            ))
        )
    variants.append(RotationStrategy(RotationConfig(
        lookback_minutes=24 * 60, rebalance_minutes=24 * 60, top_n=1)))
    variants.append(RotationStrategy(RotationConfig(
        lookback_minutes=24 * 60, rebalance_minutes=24 * 60, top_n=5)))
    variants.append(RotationStrategy(RotationConfig(
        lookback_minutes=24 * 60, rebalance_minutes=48 * 60, top_n=3)))
    variants.append(RotationStrategy(RotationConfig(
        lookback_minutes=24 * 60, rebalance_minutes=24 * 60, top_n=3, risk_adjust=False)))
    return variants


__all__ = [
    "RotationConfig",
    "RotationStrategy",
    "build_panel",
    "default_rotation_strategies",
    "holdings_to_signals",
    "rank_universe",
    "target_holdings",
]
