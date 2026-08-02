"""Phase 3 + 4: entry strategies and exit policies.

Each strategy is an independent, testable object with one job: given a momentum
event and the bars that follow, decide **whether and when** an entry order would
have been placed. Strategies never see the outcome of the trade, never see bars
beyond the bar they are currently evaluating, and never place the entry on the
signal bar itself.

The contract is deliberately narrow::

    plan = strategy.find_entry(event_row, forward)   # forward = bars AFTER event

``forward`` is iterated in order; at iteration ``i`` a strategy may read
``forward.iloc[:i+1]`` (bars that have closed) and may place an order that
executes at ``i+1`` or later. :class:`EntryPlan` records which bar the order was
placed for, and the backtester enforces that it is never earlier.

Exits live in :class:`ExitPolicy` so that any entry strategy can be paired with
any exit configuration - Phase 4 requires testing them separately and in
combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# plans and policies
# --------------------------------------------------------------------------- #
@dataclass
class EntryPlan:
    """An order the strategy would have placed, and when."""

    decision_offset: int          # index in `forward` whose CLOSE triggered the decision
    execution_offset: int         # earliest bar that may fill (>= decision_offset + 1)
    order_type: str               # "market" | "limit"
    reference_price: float | None = None   # for market orders: next bar open
    limit_price: float | None = None       # for limit orders
    valid_for_bars: int = 1       # how long a limit order rests
    reason: str = ""
    entry_zone_low: float | None = None
    entry_zone_high: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.execution_offset <= self.decision_offset:
            raise ValueError(
                "execution_offset must be strictly after decision_offset "
                f"({self.execution_offset} <= {self.decision_offset}) - this would be look-ahead"
            )


@dataclass
class ExitPolicy:
    """Phase 4 exit configuration. All levels are relative to the fill price."""

    take_profit_pct: float | None = 0.03
    stop_loss_pct: float | None = 0.02
    atr_stop_multiple: float | None = None      # overrides stop_loss_pct when set
    atr_target_multiple: float | None = None
    trailing_stop_pct: float | None = None
    chandelier_atr_multiple: float | None = None
    breakeven_after_pct: float | None = None
    partial_take_profit_pct: float | None = None
    partial_take_fraction: float = 0.5
    time_stop_minutes: int | None = 240
    exit_below_vwap: bool = False
    exit_below_ema_span: int | None = None
    exit_below_consolidation_low: bool = False
    max_adverse_excursion_pct: float | None = None
    regime_exit: bool = False
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            parts = []
            if self.take_profit_pct is not None:
                parts.append(f"tp{self.take_profit_pct:.3g}")
            if self.atr_stop_multiple is not None:
                parts.append(f"atrsl{self.atr_stop_multiple:.3g}")
            elif self.stop_loss_pct is not None:
                parts.append(f"sl{self.stop_loss_pct:.3g}")
            if self.trailing_stop_pct is not None:
                parts.append(f"trail{self.trailing_stop_pct:.3g}")
            if self.chandelier_atr_multiple is not None:
                parts.append(f"chand{self.chandelier_atr_multiple:.3g}")
            if self.time_stop_minutes is not None:
                parts.append(f"t{self.time_stop_minutes}m")
            self.name = "_".join(parts) or "no_exit_rules"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# base class
# --------------------------------------------------------------------------- #
class Strategy:
    """Base entry strategy."""

    key = "base"

    def __init__(self, name: str | None = None, max_wait_minutes: int = 120, **params: Any):
        self.name = name or self.key
        self.max_wait_minutes = int(max_wait_minutes)
        self.params = params

    # -- helpers available to subclasses ---------------------------------------
    @staticmethod
    def _next_open(forward: pd.DataFrame, offset: int) -> float | None:
        if offset + 1 >= len(forward):
            return None
        value = float(forward["open"].iloc[offset + 1])
        return value if np.isfinite(value) and value > 0 else None

    def _max_bars(self, forward: pd.DataFrame, interval_minutes: int) -> int:
        return min(len(forward), max(1, int(self.max_wait_minutes / interval_minutes)))

    def find_entry(
        self,
        event: pd.Series,
        forward: pd.DataFrame,
        interval_minutes: int = 1,
    ) -> EntryPlan | None:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"strategy": self.name, "key": self.key, **self.params, "max_wait_minutes": self.max_wait_minutes}


# --------------------------------------------------------------------------- #
# Strategy A - immediate momentum entry
# --------------------------------------------------------------------------- #
class ImmediateEntry(Strategy):
    """Buy at the first realistically executable price after the event.

    The event is known at the close of the event bar. ``forward.iloc[0]`` is the
    first bar after it; its **open** is the earliest price a market order could
    plausibly touch. The signal bar's own close is never used.
    """

    key = "A_immediate"

    def find_entry(self, event, forward, interval_minutes=1):
        if forward.empty:
            return None
        price = float(forward["open"].iloc[0])
        if not np.isfinite(price) or price <= 0:
            return None
        return EntryPlan(
            decision_offset=-1,           # decision made at the event bar close
            execution_offset=0,           # fills on the first bar after the event
            order_type="market",
            reference_price=price,
            reason="immediate market entry on the bar after the event",
            entry_zone_low=price,
            entry_zone_high=price * 1.005,
        )


# --------------------------------------------------------------------------- #
# Strategy B - pullback continuation
# --------------------------------------------------------------------------- #
class PullbackEntry(Strategy):
    """Wait for a retracement, then require evidence that the fall has stopped.

    ``pullback_pct`` or ``pullback_atr`` defines the depth (measured from the
    post-event high). Entry is triggered only after a bar closes above the
    previous bar's high once the pullback level has been touched - "stabilises or
    resumes upward movement" made mechanical.
    """

    key = "B_pullback"

    def __init__(
        self,
        pullback_pct: float | None = 0.02,
        pullback_atr: float | None = None,
        require_confirmation: bool = True,
        max_wait_minutes: int = 180,
        name: str | None = None,
    ):
        super().__init__(
            name=name or f"B_pullback_{pullback_pct if pullback_pct is not None else f'{pullback_atr}atr'}",
            max_wait_minutes=max_wait_minutes,
            pullback_pct=pullback_pct,
            pullback_atr=pullback_atr,
            require_confirmation=require_confirmation,
        )
        self.pullback_pct = pullback_pct
        self.pullback_atr = pullback_atr
        self.require_confirmation = require_confirmation

    def find_entry(self, event, forward, interval_minutes=1):
        if forward.empty:
            return None
        n = self._max_bars(forward, interval_minutes)
        atr_value = float(event.get("atr_60m", np.nan))
        event_close = float(event["close"])

        if self.pullback_atr is not None and np.isfinite(atr_value) and atr_value > 0:
            depth = self.pullback_atr * atr_value
        elif self.pullback_pct is not None:
            depth = self.pullback_pct * event_close
        else:
            return None

        running_high = event_close
        touched = False
        for i in range(n):
            bar = forward.iloc[i]
            running_high = max(running_high, float(bar["high"]))
            target_level = running_high - depth
            if not touched and float(bar["low"]) <= target_level:
                touched = True
                continue
            if touched:
                if not self.require_confirmation:
                    price = self._next_open(forward, i)
                    if price is None:
                        return None
                    return EntryPlan(i, i + 1, "market", reference_price=price,
                                     reason="pullback depth reached")
                prev = forward.iloc[i - 1] if i > 0 else bar
                confirmed = float(bar["close"]) > float(prev["high"])
                if confirmed:
                    price = self._next_open(forward, i)
                    if price is None:
                        return None
                    return EntryPlan(
                        i, i + 1, "market", reference_price=price,
                        reason="pullback reached and upward continuation confirmed",
                        entry_zone_low=target_level,
                        entry_zone_high=price * 1.005,
                        diagnostics={"pullback_depth": depth, "running_high": running_high},
                    )
        return None


# --------------------------------------------------------------------------- #
# Strategy C - consolidation breakout
# --------------------------------------------------------------------------- #
class ConsolidationBreakout(Strategy):
    """Wait for a tight range to form, then buy the break of its high on volume."""

    key = "C_breakout"

    def __init__(
        self,
        min_consolidation_minutes: int = 15,
        max_consolidation_minutes: int = 120,
        max_range_pct: float = 0.03,
        min_relative_volume: float = 1.5,
        max_wait_minutes: int = 240,
        name: str | None = None,
    ):
        super().__init__(
            name=name or f"C_breakout_{min_consolidation_minutes}-{max_consolidation_minutes}m",
            max_wait_minutes=max_wait_minutes,
            min_consolidation_minutes=min_consolidation_minutes,
            max_consolidation_minutes=max_consolidation_minutes,
            max_range_pct=max_range_pct,
            min_relative_volume=min_relative_volume,
        )
        self.min_bars = max(1, int(min_consolidation_minutes))
        self.max_bars = max(self.min_bars, int(max_consolidation_minutes))
        self.max_range_pct = max_range_pct
        self.min_relative_volume = min_relative_volume

    def find_entry(self, event, forward, interval_minutes=1):
        if forward.empty:
            return None
        n = self._max_bars(forward, interval_minutes)
        min_bars = max(1, int(self.min_bars / interval_minutes))
        max_bars = max(min_bars, int(self.max_bars / interval_minutes))
        baseline_volume = float(event.get("volume_baseline", np.nan))

        for i in range(min_bars, n):
            window = forward.iloc[max(0, i - max_bars) : i]      # closed bars only
            if len(window) < min_bars:
                continue
            hi = float(window["high"].max())
            lo = float(window["low"].min())
            close_i = float(forward["close"].iloc[i])
            if not np.isfinite(hi) or not np.isfinite(lo) or hi <= 0:
                continue
            if (hi - lo) / hi > self.max_range_pct:
                continue                                          # not a consolidation
            if close_i <= hi:
                continue                                          # no breakout yet
            rel_vol = np.nan
            bar_volume = float(forward["volume"].iloc[i])
            if np.isfinite(baseline_volume) and baseline_volume > 0:
                rel_vol = bar_volume / baseline_volume
            if np.isfinite(rel_vol) and rel_vol < self.min_relative_volume:
                continue                                          # unconfirmed break
            price = self._next_open(forward, i)
            if price is None:
                return None
            return EntryPlan(
                i, i + 1, "market", reference_price=price,
                reason=f"break of {len(window)}-bar consolidation high on {rel_vol:.2f}x volume",
                entry_zone_low=hi,
                entry_zone_high=price * 1.005,
                diagnostics={"consolidation_high": hi, "consolidation_low": lo, "rel_volume": rel_vol},
            )
        return None


# --------------------------------------------------------------------------- #
# Strategy D - VWAP / EMA retest
# --------------------------------------------------------------------------- #
class DynamicLevelRetest(Strategy):
    """Buy the retest of VWAP or an EMA, confirmed by a close back above it."""

    key = "D_retest"

    def __init__(
        self,
        level: str = "vwap_session",
        tolerance_pct: float = 0.002,
        max_wait_minutes: int = 180,
        name: str | None = None,
    ):
        super().__init__(
            name=name or f"D_retest_{level}",
            max_wait_minutes=max_wait_minutes,
            level=level,
            tolerance_pct=tolerance_pct,
        )
        self.level = level
        self.tolerance_pct = tolerance_pct

    def find_entry(self, event, forward, interval_minutes=1):
        if forward.empty or self.level not in forward.columns:
            return None
        n = self._max_bars(forward, interval_minutes)
        touched = False
        for i in range(n):
            bar = forward.iloc[i]
            level_value = float(bar[self.level])
            if not np.isfinite(level_value) or level_value <= 0:
                continue
            band = level_value * (1.0 + self.tolerance_pct)
            if not touched and float(bar["low"]) <= band:
                touched = True
                continue
            if touched and float(bar["close"]) > level_value:
                price = self._next_open(forward, i)
                if price is None:
                    return None
                return EntryPlan(
                    i, i + 1, "market", reference_price=price,
                    reason=f"retest of {self.level} held and price closed back above it",
                    entry_zone_low=level_value,
                    entry_zone_high=price * 1.005,
                    diagnostics={"level_value": level_value},
                )
        return None


# --------------------------------------------------------------------------- #
# Strategy E - volume-confirmed continuation
# --------------------------------------------------------------------------- #
class VolumeConfirmedEntry(Strategy):
    """Immediate entry, but only when the move carried statistically unusual volume.

    The threshold is expressed as a z-score against the coin's *own* trailing
    volume distribution, so a normally-quiet coin is not permanently disqualified
    and a normally-busy one is not permanently favoured.
    """

    key = "E_volume"

    def __init__(
        self,
        min_volume_zscore: float = 2.0,
        min_relative_volume: float = 2.0,
        max_wait_minutes: int = 5,
        name: str | None = None,
    ):
        super().__init__(
            name=name or f"E_volume_z{min_volume_zscore:g}",
            max_wait_minutes=max_wait_minutes,
            min_volume_zscore=min_volume_zscore,
            min_relative_volume=min_relative_volume,
        )
        self.min_volume_zscore = min_volume_zscore
        self.min_relative_volume = min_relative_volume

    def find_entry(self, event, forward, interval_minutes=1):
        if forward.empty:
            return None
        z = float(event.get("volume_zscore", np.nan))
        rel = float(event.get("rel_volume_60m", np.nan))
        if not np.isfinite(z) or z < self.min_volume_zscore:
            return None
        if np.isfinite(rel) and rel < self.min_relative_volume:
            return None
        price = float(forward["open"].iloc[0])
        if not np.isfinite(price) or price <= 0:
            return None
        return EntryPlan(
            -1, 0, "market", reference_price=price,
            reason=f"volume z-score {z:.2f} >= {self.min_volume_zscore:g}",
            entry_zone_low=price, entry_zone_high=price * 1.005,
            diagnostics={"volume_zscore": z, "rel_volume_60m": rel},
        )


# --------------------------------------------------------------------------- #
# Strategy F - cross-sectional momentum
# --------------------------------------------------------------------------- #
class CrossSectionalMomentum(Strategy):
    """Rank concurrent events and take only the strongest.

    Ranking happens in :func:`rank_cross_sectional` *before* the backtest loop,
    using only contemporaneous information. This class then behaves like an
    immediate entry for the events that survived the ranking.
    """

    key = "F_cross_sectional"

    def __init__(self, top_n: int = 2, name: str | None = None, **params):
        super().__init__(name=name or f"F_cross_sectional_top{top_n}", max_wait_minutes=5, top_n=top_n, **params)
        self.top_n = top_n

    def find_entry(self, event, forward, interval_minutes=1):
        if forward.empty:
            return None
        if not bool(event.get("cs_selected", True)):
            return None
        price = float(forward["open"].iloc[0])
        if not np.isfinite(price) or price <= 0:
            return None
        return EntryPlan(
            -1, 0, "market", reference_price=price,
            reason=f"top-{self.top_n} cross-sectional risk-adjusted momentum",
            entry_zone_low=price, entry_zone_high=price * 1.005,
        )


def rank_cross_sectional(
    events: pd.DataFrame,
    top_n: int = 2,
    window_minutes: int = 60,
    score_column: str = "ret_60m",
    vol_column: str = "realised_vol_60m",
    max_spread_proxy_bps: float | None = 60.0,
    min_quote_volume_1440m: float | None = 100_000.0,
) -> pd.DataFrame:
    """Flag the top-N risk-adjusted events inside each time bucket.

    Score = trailing return / trailing volatility, both known at the event bar.
    Events are bucketed by ``window_minutes`` so that "concurrent" is well defined
    without peeking forward.
    """
    if events.empty:
        return events
    out = events.copy()
    out["cs_score"] = out[score_column] / out[vol_column].replace(0.0, np.nan)
    eligible = pd.Series(True, index=out.index)
    if max_spread_proxy_bps is not None and "spread_proxy_bps" in out.columns:
        eligible &= out["spread_proxy_bps"].fillna(np.inf) <= max_spread_proxy_bps
    if min_quote_volume_1440m is not None and "quote_volume_1440m" in out.columns:
        eligible &= out["quote_volume_1440m"].fillna(0.0) >= min_quote_volume_1440m
    out["cs_eligible"] = eligible

    bucket = pd.to_datetime(out["event_time"], utc=True).dt.floor(f"{window_minutes}min")
    out["cs_bucket"] = bucket
    ranked = (
        out[out["cs_eligible"]]
        .groupby("cs_bucket")["cs_score"]
        .rank(ascending=False, method="first")
    )
    out["cs_rank"] = ranked
    out["cs_selected"] = out["cs_rank"].le(top_n).fillna(False)
    return out


# --------------------------------------------------------------------------- #
# Strategy G - exhaustion / mean reversion
# --------------------------------------------------------------------------- #
class ExhaustionMeanReversion(Strategy):
    """Test the opposite hypothesis: the spike is exhaustion, not continuation.

    No short selling is assumed or simulated - Bitvavo spot has none. The long
    implementation waits for a *deep* correction from the post-event high and for
    stabilisation, then buys. If the exhaustion hypothesis is right this beats
    buying the breakout; if it is wrong it will show up as a worse expectancy,
    which is equally informative.
    """

    key = "G_meanreversion"

    def __init__(
        self,
        min_retrace_pct: float = 0.05,
        stabilisation_bars: int = 3,
        max_wait_minutes: int = 480,
        name: str | None = None,
    ):
        super().__init__(
            name=name or f"G_meanrev_{min_retrace_pct:.0%}".replace("%", "pct"),
            max_wait_minutes=max_wait_minutes,
            min_retrace_pct=min_retrace_pct,
            stabilisation_bars=stabilisation_bars,
        )
        self.min_retrace_pct = min_retrace_pct
        self.stabilisation_bars = stabilisation_bars

    def find_entry(self, event, forward, interval_minutes=1):
        if forward.empty:
            return None
        n = self._max_bars(forward, interval_minutes)
        running_high = float(event["close"])
        retraced_at: int | None = None
        for i in range(n):
            bar = forward.iloc[i]
            running_high = max(running_high, float(bar["high"]))
            drawdown = float(bar["low"]) / running_high - 1.0
            if retraced_at is None and drawdown <= -self.min_retrace_pct:
                retraced_at = i
                continue
            if retraced_at is not None and i - retraced_at >= self.stabilisation_bars:
                window = forward.iloc[retraced_at : i + 1]
                stabilised = float(window["low"].iloc[-1]) >= float(window["low"].min())
                rising = float(bar["close"]) > float(forward["close"].iloc[i - 1])
                if stabilised and rising:
                    price = self._next_open(forward, i)
                    if price is None:
                        return None
                    return EntryPlan(
                        i, i + 1, "market", reference_price=price,
                        reason=f"{self.min_retrace_pct:.0%} retrace from post-event high, then stabilisation",
                        entry_zone_low=float(window["low"].min()),
                        entry_zone_high=price * 1.005,
                        diagnostics={"post_event_high": running_high},
                    )
        return None


class NoEntry(Strategy):
    """Benchmark: do nothing after the event. Every metric is exactly zero."""

    key = "Z_do_nothing"

    def find_entry(self, event, forward, interval_minutes=1):
        return None


# --------------------------------------------------------------------------- #
# Strategy H - hybrid confirmation
# --------------------------------------------------------------------------- #
class HybridConfirmation(Strategy):
    """Combine a small number of independently validated conditions.

    Complexity is capped on purpose: ``max_conditions`` refuses to construct a
    filter stack deeper than the evidence can support. Each condition must have
    earned its place by improving *out-of-sample* results in the ablation study
    (:func:`bitvavo_momentum.optimizer.feature_ablation`).
    """

    key = "H_hybrid"

    def __init__(
        self,
        base_strategy: Strategy | None = None,
        conditions: dict[str, tuple[str, float]] | None = None,
        max_conditions: int = 4,
        name: str | None = None,
    ):
        conditions = conditions or {}
        if len(conditions) > max_conditions:
            raise ValueError(
                f"{len(conditions)} conditions exceeds max_conditions={max_conditions}; "
                "each extra filter needs out-of-sample justification"
            )
        super().__init__(name=name or "H_hybrid", max_wait_minutes=180, conditions=conditions)
        self.base = base_strategy or PullbackEntry(pullback_pct=0.02)
        self.conditions = conditions

    def _passes(self, event: pd.Series) -> tuple[bool, list[str], list[str]]:
        passed: list[str] = []
        failed: list[str] = []
        for column, (op, threshold) in self.conditions.items():
            value = float(event.get(column, np.nan))
            if not np.isfinite(value):
                failed.append(f"{column} unavailable")
                continue
            ok = value >= threshold if op == ">=" else value <= threshold
            (passed if ok else failed).append(f"{column} {op} {threshold:g} (was {value:.4g})")
        return not failed, passed, failed

    def find_entry(self, event, forward, interval_minutes=1):
        ok, passed, failed = self._passes(event)
        if not ok:
            return None
        plan = self.base.find_entry(event, forward, interval_minutes)
        if plan is not None:
            plan.reason = f"{plan.reason}; filters passed: {', '.join(passed) or 'none'}"
            plan.diagnostics["filters_passed"] = passed
            plan.diagnostics["filters_failed"] = failed
        return plan


# --------------------------------------------------------------------------- #
# benchmark: random entries
# --------------------------------------------------------------------------- #
class RandomEntry(Strategy):
    """Benchmark: enter at a random bar within the same window, same market.

    Matching market and holding period isolates the question "is the *timing*
    informative?" from "did the coin go up anyway?".
    """

    key = "Z_random"

    def __init__(self, max_wait_minutes: int = 120, seed: int = 20260802, name: str | None = None):
        super().__init__(name=name or "Z_random", max_wait_minutes=max_wait_minutes, seed=seed)
        self.rng = np.random.default_rng(seed)

    def find_entry(self, event, forward, interval_minutes=1):
        if len(forward) < 2:
            return None
        n = self._max_bars(forward, interval_minutes)
        i = int(self.rng.integers(0, max(1, n - 1)))
        price = self._next_open(forward, i)
        if price is None:
            return None
        return EntryPlan(i, i + 1, "market", reference_price=price, reason="random entry benchmark")


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def default_strategies() -> list[Strategy]:
    """The Phase 3 comparison set, at their default parameters."""
    return [
        ImmediateEntry(),
        PullbackEntry(pullback_pct=0.01, name="B_pullback_1pct"),
        PullbackEntry(pullback_pct=0.02, name="B_pullback_2pct"),
        PullbackEntry(pullback_pct=0.03, name="B_pullback_3pct"),
        PullbackEntry(pullback_pct=0.05, name="B_pullback_5pct"),
        PullbackEntry(pullback_pct=None, pullback_atr=0.5, name="B_pullback_0.5atr"),
        PullbackEntry(pullback_pct=None, pullback_atr=1.0, name="B_pullback_1atr"),
        ConsolidationBreakout(),
        DynamicLevelRetest(level="vwap_session", name="D_retest_vwap"),
        DynamicLevelRetest(level="ema_9", name="D_retest_ema9"),
        DynamicLevelRetest(level="ema_20", name="D_retest_ema20"),
        VolumeConfirmedEntry(),
        CrossSectionalMomentum(top_n=2),
        ExhaustionMeanReversion(min_retrace_pct=0.05),
        ExhaustionMeanReversion(min_retrace_pct=0.08, name="G_meanrev_8pct"),
        RandomEntry(),
        NoEntry(),
    ]


def default_exit_policies() -> list[ExitPolicy]:
    """A deliberately small Phase 4 grid; the optimizer expands it on train data."""
    return [
        ExitPolicy(take_profit_pct=0.02, stop_loss_pct=0.01, time_stop_minutes=240),
        ExitPolicy(take_profit_pct=0.03, stop_loss_pct=0.02, time_stop_minutes=240),
        ExitPolicy(take_profit_pct=0.05, stop_loss_pct=0.03, time_stop_minutes=480),
        ExitPolicy(take_profit_pct=0.05, stop_loss_pct=None, atr_stop_multiple=1.5, time_stop_minutes=480),
        ExitPolicy(take_profit_pct=None, stop_loss_pct=0.02, trailing_stop_pct=0.03, time_stop_minutes=1440),
        ExitPolicy(take_profit_pct=None, stop_loss_pct=None, chandelier_atr_multiple=3.0, time_stop_minutes=1440),
        ExitPolicy(take_profit_pct=0.04, stop_loss_pct=0.02, breakeven_after_pct=0.02, time_stop_minutes=480),
        ExitPolicy(
            take_profit_pct=0.06, stop_loss_pct=0.025, partial_take_profit_pct=0.02,
            partial_take_fraction=0.5, time_stop_minutes=480,
        ),
    ]


STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    cls.key: cls
    for cls in (
        ImmediateEntry,
        PullbackEntry,
        ConsolidationBreakout,
        DynamicLevelRetest,
        VolumeConfirmedEntry,
        CrossSectionalMomentum,
        ExhaustionMeanReversion,
        HybridConfirmation,
        RandomEntry,
        NoEntry,
    )
}
