"""Applying the regime classification as a trading gate, and testing whether it helps.

Why this exists
---------------
Every strategy tested in this project is long-only, and the training window fell.
The matched-random baseline over the same signals was also negative, which means
the honest reading of five rejected families is not "the signals are wrong" but
"being long was wrong". If that reading is right, the fix is not a better entry -
it is not being in the market when the market is falling. That is what a regime
gate does, and it is the last untested lever with real leverage.

:mod:`regimes` already classifies. This module applies the classification, and -
more importantly - measures whether applying it was worth anything.

The trap this module is built to avoid
--------------------------------------
A regime filter is the easiest thing in quantitative finance to fool yourself
with. There are only ~580 days in the sample, a handful of plausible regime
definitions, and enormous freedom in choosing which labels count as "allowed".
Pick the labels whose realised returns were positive and you have not built a
filter, you have drawn a line around the winners after the race.

Three defences, all structural rather than advisory:

1. **The presets are declared up front.** :data:`PRESETS` is a fixed, small set
   of economically motivated rules ("only when BTC is in an uptrend"), written
   before any result was seen. The allowed-label sets are never chosen from
   performance.
2. **The gate is scored by what it rejects, not by what it keeps.** A filter
   earns its place only if the signals it throws away were *worse* than the ones
   it keeps. ``rejected_gross_mean`` is reported next to ``kept_gross_mean``
   everywhere; a gate whose rejects were fine is noise dressed as insight.
3. **Coverage is always reported.** A gate that admits 4% of signals will show a
   flattering mean on a sample too small to mean anything, so ``share_kept`` and
   ``n_kept`` sit beside every return figure.

Causality is inherited from :func:`regimes.classify_regimes`, which shifts every
daily label by one day, and from :func:`compute_breadth`, which builds its panel
from past observations only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .regimes import (
    BTC_BULL,
    BTC_SIDEWAYS,
    RISK_ON,
    VOL_HIGH,
    attach_regimes,
)
from .rotation import build_panel

log = get_logger(__name__)


@dataclass
class GateConfig:
    """Which regime labels permit a long position.

    ``None`` for a dimension means "do not filter on it". A label the classifier
    could not compute (NaN, typically early history before the EMAs warm up) is
    always rejected: a filter that passes when it knows nothing is not a filter.
    """

    name: str = "ungated"
    allowed_btc_trend: list[str] | None = None
    allowed_risk_regime: list[str] | None = None
    blocked_volatility: list[str] | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "gate": self.name,
            "allowed_btc_trend": ",".join(self.allowed_btc_trend or []) or "any",
            "allowed_risk_regime": ",".join(self.allowed_risk_regime or []) or "any",
            "blocked_volatility": ",".join(self.blocked_volatility or []) or "none",
        }


# Declared before any result was inspected. Each is a rule a discretionary
# trader would recognise, not a subset chosen for its returns.
PRESETS: tuple[GateConfig, ...] = (
    GateConfig(name="ungated"),
    GateConfig(name="bull_only", allowed_btc_trend=[BTC_BULL]),
    GateConfig(name="not_bear", allowed_btc_trend=[BTC_BULL, BTC_SIDEWAYS]),
    GateConfig(name="risk_on", allowed_risk_regime=[RISK_ON]),
    GateConfig(name="bull_calm", allowed_btc_trend=[BTC_BULL], blocked_volatility=[VOL_HIGH]),
)


def compute_breadth(
    features_by_market: dict[str, pd.DataFrame],
    lookback_minutes: int = 1440,
    freq_minutes: int = 60,
) -> pd.Series:
    """Fraction of the universe with a positive trailing return.

    Breadth is the one regime input that is not derivable from BTC alone: a
    market where BTC rises while everything else bleeds is a different place to
    trade an altcoin basket than one where the whole board is green.

    The panel forward-fills past observations only, so a coin that had not
    listed yet is absent from the denominator rather than counted as flat.
    """
    panel = build_panel(features_by_market, "close", freq_minutes)
    if panel.empty:
        return pd.Series(dtype="float64")

    bars = max(1, int(round(lookback_minutes / freq_minutes)))
    returns = panel / panel.shift(bars) - 1.0
    observed = returns.notna()
    positive = (returns > 0).where(observed)
    denominator = observed.sum(axis=1).replace(0, np.nan)
    breadth = positive.sum(axis=1) / denominator
    breadth.name = "breadth"
    return breadth


def gate_mask(frame: pd.DataFrame, config: GateConfig) -> pd.Series:
    """Boolean mask: may a long be opened on each row?

    ``frame`` must already carry the regime columns (see
    :func:`regimes.attach_regimes`). Rows whose labels are missing are rejected.
    """
    if frame.empty:
        return pd.Series(dtype="bool")

    allow = pd.Series(True, index=frame.index)

    if config.allowed_btc_trend is not None:
        labels = frame.get("btc_trend")
        allow &= False if labels is None else labels.isin(config.allowed_btc_trend)
    if config.allowed_risk_regime is not None:
        labels = frame.get("risk_regime")
        allow &= False if labels is None else labels.isin(config.allowed_risk_regime)
    if config.blocked_volatility is not None:
        labels = frame.get("volatility_regime")
        if labels is None:
            allow &= False
        else:
            # A missing volatility label is not evidence of calm.
            allow &= labels.notna() & ~labels.isin(config.blocked_volatility)

    return allow.astype(bool)


def apply_gate(
    signals: pd.DataFrame,
    regimes: pd.DataFrame,
    config: GateConfig,
    time_column: str = "event_time",
) -> pd.DataFrame:
    """Return only the signals the gate permits, with regime columns attached."""
    if signals.empty or regimes.empty:
        return signals
    joined = attach_regimes(signals, regimes, time_column=time_column)
    return joined[gate_mask(joined, config)].reset_index(drop=True)


def regime_forward_returns(
    outcomes: pd.DataFrame,
    regimes: pd.DataFrame,
    horizon_column: str,
    round_trip_cost_bps: float = 77.0,
    regime_column: str = "btc_trend",
    time_column: str = "event_time",
) -> pd.DataFrame:
    """What each regime label was actually worth, before any gate is chosen.

    This is the diagnostic that decides whether a gate can work at all. If
    signals in an uptrend returned the same as signals in a downtrend, no
    partition of those labels can help, and every preset below is noise.
    """
    if outcomes.empty or regimes.empty or horizon_column not in outcomes.columns:
        return pd.DataFrame()

    joined = attach_regimes(outcomes, regimes, time_column=time_column)
    if regime_column not in joined.columns:
        return pd.DataFrame()

    cost = round_trip_cost_bps * 1e-4
    total = len(joined)
    rows: list[dict[str, Any]] = []
    for label, group in joined.groupby(joined[regime_column].fillna("unknown")):
        values = group[horizon_column].to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        mean = float(values.mean())
        rows.append(
            {
                "regime_column": regime_column,
                "regime": label,
                "n_signals": int(values.size),
                "share_of_signals": len(group) / total,
                "gross_mean": mean,
                "gross_median": float(np.median(values)),
                "hit_rate": float((values > 0).mean()),
                "net_mean": mean - cost,
                "beats_cost": bool(mean > cost),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("gross_mean", ascending=False).reset_index(drop=True)


def gate_comparison(
    outcomes: pd.DataFrame,
    regimes: pd.DataFrame,
    horizon_column: str,
    presets: tuple[GateConfig, ...] = PRESETS,
    round_trip_cost_bps: float = 77.0,
    time_column: str = "event_time",
) -> pd.DataFrame:
    """Score each preset by what it keeps *and* what it throws away.

    ``rejected_gross_mean`` is the column that matters. A gate is only doing
    work if the signals it blocks were worse than the ones it admits; if the
    rejects were fine, the gate is discarding trades at random and its apparent
    improvement is sampling noise.

    ``separation`` is kept minus rejected - the size of the gate's actual
    discrimination, in return terms.
    """
    if outcomes.empty or regimes.empty or horizon_column not in outcomes.columns:
        return pd.DataFrame()

    joined = attach_regimes(outcomes, regimes, time_column=time_column)
    cost = round_trip_cost_bps * 1e-4
    total = len(joined)

    rows: list[dict[str, Any]] = []
    for config in presets:
        mask = gate_mask(joined, config)
        kept = joined.loc[mask, horizon_column].to_numpy(dtype="float64")
        rejected = joined.loc[~mask, horizon_column].to_numpy(dtype="float64")
        kept = kept[np.isfinite(kept)]
        rejected = rejected[np.isfinite(rejected)]
        if kept.size == 0:
            rows.append({**config.describe(), "n_kept": 0, "share_kept": 0.0,
                         "kept_gross_mean": float("nan"), "rejected_gross_mean": float("nan"),
                         "separation": float("nan"), "kept_net_mean": float("nan"),
                         "kept_hit_rate": float("nan"), "beats_cost": False})
            continue
        kept_mean = float(kept.mean())
        rejected_mean = float(rejected.mean()) if rejected.size else float("nan")
        rows.append(
            {
                **config.describe(),
                "n_kept": int(kept.size),
                "n_rejected": int(rejected.size),
                "share_kept": kept.size / max(1, total),
                "kept_gross_mean": kept_mean,
                "rejected_gross_mean": rejected_mean,
                "separation": kept_mean - rejected_mean if rejected.size else float("nan"),
                "kept_net_mean": kept_mean - cost,
                "kept_hit_rate": float((kept > 0).mean()),
                "beats_cost": bool(kept_mean > cost),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class GateReport:
    """Everything needed to judge a gate, kept together so nothing is quoted alone."""

    by_regime: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_preset: pd.DataFrame = field(default_factory=pd.DataFrame)
    horizon_column: str = ""
    round_trip_cost_bps: float = 77.0

    def verdict(self) -> str:
        """One sentence, stated conservatively."""
        if self.by_preset.empty:
            return "No gate could be evaluated - no signals carried regime labels."
        gated = self.by_preset[self.by_preset["gate"] != "ungated"]
        if gated.empty:
            return "Only the ungated case was evaluated."
        best = gated.loc[gated["kept_net_mean"].idxmax()]
        if not bool(best["beats_cost"]):
            return (
                f"No gate lifted returns above the {self.round_trip_cost_bps:.0f} bps cost "
                f"floor. Best was '{best['gate']}' at {best['kept_gross_mean'] * 1e4:.0f} bps "
                f"gross on {best['n_kept']} signals."
            )
        # A gate whose discrimination is small next to the toll cannot change the
        # economics no matter how the sign comes out, and separation of a few bps
        # on a few hundred trades is well inside sampling noise. A quarter of the
        # round trip is a judgement call, stated here rather than buried: it is
        # the point below which the gate is not worth the trades it costs.
        floor = 0.25 * self.round_trip_cost_bps * 1e-4
        if not np.isfinite(best["separation"]) or best["separation"] <= floor:
            return (
                f"'{best['gate']}' clears the cost floor but does not separate: it kept "
                f"{best['kept_gross_mean'] * 1e4:.0f} bps and rejected "
                f"{best['rejected_gross_mean'] * 1e4:.0f} bps, a difference too small to be "
                "doing work. That is a smaller sample, not a filter."
            )
        return (
            f"'{best['gate']}' keeps {best['share_kept']:.0%} of signals at "
            f"{best['kept_gross_mean'] * 1e4:.0f} bps gross versus "
            f"{best['rejected_gross_mean'] * 1e4:.0f} bps for what it rejected, on "
            f"{best['n_kept']} signals. A candidate for validation, not a result."
        )


def evaluate_gate(
    outcomes: pd.DataFrame,
    regimes: pd.DataFrame,
    horizon_column: str,
    presets: tuple[GateConfig, ...] = PRESETS,
    round_trip_cost_bps: float = 77.0,
    regime_column: str = "btc_trend",
) -> GateReport:
    """Run both diagnostics and package them so neither can be quoted alone."""
    return GateReport(
        by_regime=regime_forward_returns(outcomes, regimes, horizon_column,
                                         round_trip_cost_bps, regime_column),
        by_preset=gate_comparison(outcomes, regimes, horizon_column, presets,
                                  round_trip_cost_bps),
        horizon_column=horizon_column,
        round_trip_cost_bps=round_trip_cost_bps,
    )


__all__ = [
    "PRESETS",
    "GateConfig",
    "GateReport",
    "apply_gate",
    "compute_breadth",
    "evaluate_gate",
    "gate_comparison",
    "gate_mask",
    "regime_forward_returns",
]
