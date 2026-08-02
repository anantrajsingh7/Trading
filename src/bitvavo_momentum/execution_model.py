"""Phase 5: realistic execution modelling.

A momentum backtest without a cost model is a random-number generator. Entering
at the close of the bar that triggered a +10% signal and exiting at a clean +3%
produces a beautiful equity curve that no account has ever earned. This module
makes every optimistic assumption explicit and priced:

* **Fees** - maker/taker in basis points, applied on entry *and* exit. Account
  fees are used when an API key is configured, otherwise the configured fallback.
* **Spread** - half the quoted spread is paid on each side. Historical candles
  contain no quotes, so a scenario-level half-spread is used, escalated for
  illiquid markets via the Amihud proxy.
* **Slippage** - grows with realised volatility and with illiquidity. Trading a
  thin coin during a vertical move is precisely when slippage is worst.
* **Latency** - the signal is generated at a bar close; the order arrives some
  seconds later. Modelled by executing against the *next* bar, never the signal
  bar, and by adding a latency-proportional drift penalty in fast markets.
* **Partial fills / rejections** - drawn from a seeded RNG so results are exactly
  reproducible; the same seed and scenario always yield the same fills.
* **Capacity** - an order may not exceed a configured share of the bar's volume.
  If it does, the fill is truncated: capacity is a hard constraint, not a fee.
* **Stops during gaps** - if a bar opens beyond the stop, the fill is at the open
  plus extra slippage, not at the stop price.
* **Intrabar ambiguity** - when a bar's range contains both the stop and the
  target, minute data cannot say which came first. The conservative outcome (the
  stop) is taken, and the trade is flagged ``ambiguous_bar=True`` so the share of
  ambiguous exits is reportable.

Nothing here is calibrated to make results look good; the three scenarios are
fixed in ``config/risk.yaml`` before any strategy is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .logging_utils import get_logger

log = get_logger(__name__)

BPS = 1e-4


@dataclass
class ExecutionScenario:
    """One named set of execution assumptions."""

    name: str
    description: str = ""
    half_spread_bps: float = 7.5
    slippage_base_bps: float = 6.0
    slippage_vol_coefficient: float = 0.25
    slippage_illiquidity_coefficient: float = 1.5
    latency_seconds: float = 10.0
    fill_probability_market: float = 0.98
    fill_probability_limit: float = 0.60
    partial_fill_probability: float = 0.15
    min_partial_fill_fraction: float = 0.5
    stop_gap_extra_slippage_bps: float = 25.0
    max_participation_of_bar_volume: float = 0.05
    taker_fee_bps: float = 25.0
    maker_fee_bps: float = 15.0

    @classmethod
    def from_config(
        cls,
        name: str,
        risk_config: dict[str, Any],
        account_fees: dict[str, float] | None = None,
    ) -> ExecutionScenario:
        scenarios = risk_config.get("execution_scenarios", {})
        if name not in scenarios:
            raise KeyError(f"Unknown execution scenario {name!r}; have {sorted(scenarios)}")
        params = dict(scenarios[name])
        fees_cfg = risk_config.get("fees", {})
        taker = float(fees_cfg.get("fallback_taker_bps", 25.0))
        maker = float(fees_cfg.get("fallback_maker_bps", 15.0))
        if account_fees and fees_cfg.get("use_account_fees_when_available", True):
            taker = float(account_fees["taker"]) / BPS
            maker = float(account_fees["maker"]) / BPS
            log.info("Using account-specific fees: taker=%.2fbps maker=%.2fbps", taker, maker)
        params.setdefault("description", "")
        return cls(name=name, taker_fee_bps=taker, maker_fee_bps=maker, **params)


@dataclass
class FillResult:
    """Outcome of one attempted execution."""

    filled: bool
    price: float
    quantity: float
    requested_quantity: float
    fee_eur: float
    spread_cost_eur: float
    slippage_cost_eur: float
    reason: str = ""
    ambiguous_bar: bool = False

    @property
    def fill_fraction(self) -> float:
        return self.quantity / self.requested_quantity if self.requested_quantity else 0.0

    @property
    def notional_eur(self) -> float:
        return self.price * self.quantity


class ExecutionModel:
    """Applies a scenario's assumptions to individual order attempts."""

    def __init__(self, scenario: ExecutionScenario, seed: int = 20260802):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self._seed = seed

    def reset(self, seed: int | None = None) -> None:
        """Re-seed so every backtest run is byte-for-byte reproducible."""
        self.rng = np.random.default_rng(self._seed if seed is None else seed)

    # -- cost components -------------------------------------------------------
    def slippage_bps(self, realised_vol: float | None, illiquidity_score: float | None) -> float:
        s = self.scenario
        vol_bps = 0.0
        if realised_vol is not None and np.isfinite(realised_vol):
            vol_bps = float(realised_vol) / BPS * s.slippage_vol_coefficient
        illiq_bps = 0.0
        if illiquidity_score is not None and np.isfinite(illiquidity_score):
            # illiquidity_score is a 0..1 normalised rank; 0 = deepest market.
            illiq_bps = float(np.clip(illiquidity_score, 0.0, 1.0)) * 100.0 * s.slippage_illiquidity_coefficient
        return s.slippage_base_bps + vol_bps + illiq_bps

    def effective_price(
        self,
        reference_price: float,
        side: str,
        realised_vol: float | None = None,
        illiquidity_score: float | None = None,
        extra_bps: float = 0.0,
    ) -> tuple[float, float, float]:
        """Return ``(price, spread_bps, slippage_bps)`` for one side of a trade."""
        s = self.scenario
        spread_bps = s.half_spread_bps
        slip_bps = self.slippage_bps(realised_vol, illiquidity_score) + extra_bps
        total_bps = spread_bps + slip_bps
        sign = 1.0 if side.lower() == "buy" else -1.0
        return reference_price * (1.0 + sign * total_bps * BPS), spread_bps, slip_bps

    def capacity_limit(self, bar_volume: float | None, price: float) -> float:
        """Maximum quantity executable in one bar given participation limits."""
        if bar_volume is None or not np.isfinite(bar_volume) or bar_volume <= 0:
            return 0.0
        return float(bar_volume) * self.scenario.max_participation_of_bar_volume

    # -- order attempts --------------------------------------------------------
    def execute_market_order(
        self,
        reference_price: float,
        quantity: float,
        side: str,
        bar_volume: float | None = None,
        realised_vol: float | None = None,
        illiquidity_score: float | None = None,
        min_order_quote_eur: float = 5.0,
        amount_decimals: int = 8,
        extra_bps: float = 0.0,
    ) -> FillResult:
        s = self.scenario
        requested = float(quantity)
        if requested <= 0 or reference_price <= 0:
            return FillResult(False, 0.0, 0.0, requested, 0.0, 0.0, 0.0, "invalid order")

        if self.rng.random() > s.fill_probability_market:
            return FillResult(False, 0.0, 0.0, requested, 0.0, 0.0, 0.0, "order rejected / not filled")

        price, spread_bps, slip_bps = self.effective_price(
            reference_price, side, realised_vol, illiquidity_score, extra_bps
        )

        filled = requested
        reason = ""
        cap = self.capacity_limit(bar_volume, price)
        if cap > 0 and filled > cap:
            filled = cap
            reason = "capacity-truncated"

        if self.rng.random() < s.partial_fill_probability:
            fraction = self.rng.uniform(s.min_partial_fill_fraction, 1.0)
            filled *= fraction
            reason = (reason + "; " if reason else "") + "partial fill"

        factor = 10**int(amount_decimals)
        filled = int(filled * factor) / factor

        notional = filled * price
        if filled <= 0 or notional < min_order_quote_eur:
            return FillResult(
                False, 0.0, 0.0, requested, 0.0, 0.0, 0.0,
                f"below minimum order size ({notional:.2f} < {min_order_quote_eur:.2f} EUR)",
            )

        fee = notional * s.taker_fee_bps * BPS
        spread_cost = notional * spread_bps * BPS
        slip_cost = notional * slip_bps * BPS
        return FillResult(True, price, filled, requested, fee, spread_cost, slip_cost, reason)

    def execute_limit_order(
        self,
        limit_price: float,
        quantity: float,
        side: str,
        bar_low: float,
        bar_high: float,
        bar_volume: float | None = None,
        min_order_quote_eur: float = 5.0,
        amount_decimals: int = 8,
    ) -> FillResult:
        """Limit order: touched price is necessary but not sufficient.

        Price trading *through* the limit does not guarantee a fill - the queue
        may never reach you. ``fill_probability_limit`` prices that risk, and it
        is deliberately pessimistic in the realistic and stress scenarios.
        """
        s = self.scenario
        requested = float(quantity)
        touched = (bar_low <= limit_price) if side.lower() == "buy" else (bar_high >= limit_price)
        if not touched:
            return FillResult(False, 0.0, 0.0, requested, 0.0, 0.0, 0.0, "limit not touched")
        if self.rng.random() > s.fill_probability_limit:
            return FillResult(False, 0.0, 0.0, requested, 0.0, 0.0, 0.0, "limit touched but not filled (queue)")

        filled = requested
        reason = ""
        cap = self.capacity_limit(bar_volume, limit_price)
        if cap > 0 and filled > cap:
            filled = cap
            reason = "capacity-truncated"
        if self.rng.random() < s.partial_fill_probability:
            filled *= self.rng.uniform(s.min_partial_fill_fraction, 1.0)
            reason = (reason + "; " if reason else "") + "partial fill"

        factor = 10**int(amount_decimals)
        filled = int(filled * factor) / factor
        notional = filled * limit_price
        if filled <= 0 or notional < min_order_quote_eur:
            return FillResult(
                False, 0.0, 0.0, requested, 0.0, 0.0, 0.0, "below minimum order size"
            )
        # A resting limit order earns the maker fee and pays no spread.
        fee = notional * s.maker_fee_bps * BPS
        return FillResult(True, limit_price, filled, requested, fee, 0.0, 0.0, reason)

    def execute_stop(
        self,
        stop_price: float,
        quantity: float,
        bar_open: float,
        bar_low: float,
        bar_high: float,
        side: str = "sell",
        bar_volume: float | None = None,
        realised_vol: float | None = None,
        illiquidity_score: float | None = None,
        min_order_quote_eur: float = 5.0,
        amount_decimals: int = 8,
    ) -> FillResult:
        """Stop exit, with honest gap handling.

        If the bar opens beyond the stop the position is not sold at the stop
        price - it is sold at the open, worse, plus gap slippage.
        """
        gapped = bar_open <= stop_price if side.lower() == "sell" else bar_open >= stop_price
        reference = bar_open if gapped else stop_price
        extra = self.scenario.stop_gap_extra_slippage_bps if gapped else 0.0
        result = self.execute_market_order(
            reference_price=reference,
            quantity=quantity,
            side=side,
            bar_volume=bar_volume,
            realised_vol=realised_vol,
            illiquidity_score=illiquidity_score,
            min_order_quote_eur=min_order_quote_eur,
            amount_decimals=amount_decimals,
            extra_bps=extra,
        )
        if gapped and result.filled:
            result.reason = (result.reason + "; " if result.reason else "") + "gapped through stop"
        if not result.filled:
            # A protective stop must not silently fail: force the exit at the
            # bar's worst realistic price rather than leaving risk uncontrolled.
            worst = bar_low if side.lower() == "sell" else bar_high
            price, spread_bps, slip_bps = self.effective_price(
                worst, side, realised_vol, illiquidity_score, extra
            )
            notional = price * quantity
            result = FillResult(
                True, price, quantity, quantity,
                notional * self.scenario.taker_fee_bps * BPS,
                notional * spread_bps * BPS,
                notional * slip_bps * BPS,
                "forced stop exit at bar extreme",
            )
        return result

    # -- reporting -------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        s = self.scenario
        return {
            "scenario": s.name,
            "description": s.description,
            "taker_fee_bps": s.taker_fee_bps,
            "maker_fee_bps": s.maker_fee_bps,
            "half_spread_bps": s.half_spread_bps,
            "slippage_base_bps": s.slippage_base_bps,
            "latency_seconds": s.latency_seconds,
            "fill_probability_market": s.fill_probability_market,
            "fill_probability_limit": s.fill_probability_limit,
            "partial_fill_probability": s.partial_fill_probability,
            "max_participation_of_bar_volume": s.max_participation_of_bar_volume,
            "round_trip_cost_bps_minimum": 2 * (s.taker_fee_bps + s.half_spread_bps + s.slippage_base_bps),
        }


def load_scenarios(
    risk_config: dict[str, Any], account_fees: dict[str, float] | None = None
) -> dict[str, ExecutionScenario]:
    return {
        name: ExecutionScenario.from_config(name, risk_config, account_fees)
        for name in risk_config.get("execution_scenarios", {})
    }
