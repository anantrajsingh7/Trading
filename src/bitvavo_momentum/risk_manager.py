"""Phase 10: position sizing and risk management.

The sizing methods are research objects (they are compared in the backtest); the
:class:`RiskManager` limits are safety objects (they are enforced identically in
backtest, paper trading and - if it were ever enabled - live trading, so that
paper results remain comparable with research results).

Deliberately absent: leverage, borrowing, averaging down, martingale, and any
form of Kelly sizing. Kelly requires a trustworthy edge estimate; on a strategy
whose edge is what we are trying to establish, it is a way to lose money faster.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger(__name__)


def _week_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Monday 00:00 of the week containing ``ts``, preserving the timezone."""
    ts = pd.Timestamp(ts)
    return ts.normalize() - pd.Timedelta(days=int(ts.dayofweek))


# --------------------------------------------------------------------------- #
# sizing
# --------------------------------------------------------------------------- #
@dataclass
class SizingConfig:
    method: str = "atr_risk"
    fixed_eur_amount: float = 250.0
    fixed_pct_of_equity: float = 0.02
    vol_target_annual: float = 0.35
    risk_per_trade_pct: float = 0.0035
    atr_stop_multiple: float = 1.5
    max_position_pct_of_equity: float = 0.15
    max_participation_of_recent_volume: float = 0.05

    @classmethod
    def from_config(cls, risk_config: dict[str, Any]) -> SizingConfig:
        sizing = risk_config.get("sizing", {})
        atr_cfg = sizing.get("atr_risk", {})
        return cls(
            method=str(sizing.get("method", "atr_risk")),
            fixed_eur_amount=float(sizing.get("fixed_eur_amount", 250.0)),
            fixed_pct_of_equity=float(sizing.get("fixed_pct_of_equity", 0.02)),
            vol_target_annual=float(sizing.get("vol_target_annual", 0.35)),
            risk_per_trade_pct=float(atr_cfg.get("risk_per_trade_pct", 0.0035)),
            atr_stop_multiple=float(atr_cfg.get("atr_stop_multiple", 1.5)),
            max_position_pct_of_equity=float(sizing.get("max_position_pct_of_equity", 0.15)),
            max_participation_of_recent_volume=float(
                sizing.get("max_participation_of_recent_volume", 0.05)
            ),
        )


def position_quantity(
    config: SizingConfig,
    equity: float,
    entry_price: float,
    stop_price: float | None = None,
    realised_vol_per_bar: float | None = None,
    bars_per_year: float = 525_600.0,
    recent_volume_per_bar: float | None = None,
) -> tuple[float, str]:
    """Return ``(quantity, explanation)`` for one position.

    Every method is capped by ``max_position_pct_of_equity`` and by a
    participation limit against recent traded volume, because a position that
    cannot be exited is not a position, it is a hostage.
    """
    if entry_price <= 0 or equity <= 0:
        return 0.0, "invalid equity or price"

    method = config.method
    if method == "fixed_eur":
        notional = min(config.fixed_eur_amount, equity)
        why = f"fixed EUR {config.fixed_eur_amount:.2f}"
    elif method == "fixed_pct":
        notional = equity * config.fixed_pct_of_equity
        why = f"{config.fixed_pct_of_equity:.2%} of equity"
    elif method == "vol_target":
        if realised_vol_per_bar is None or not np.isfinite(realised_vol_per_bar) or realised_vol_per_bar <= 0:
            return 0.0, "volatility unavailable for vol_target sizing"
        annual_vol = realised_vol_per_bar * np.sqrt(bars_per_year)
        notional = equity * min(1.0, config.vol_target_annual / annual_vol)
        why = f"vol target {config.vol_target_annual:.0%} vs realised {annual_vol:.0%}"
    elif method == "atr_risk":
        if stop_price is None or not np.isfinite(stop_price) or stop_price <= 0 or stop_price >= entry_price:
            return 0.0, "no valid stop for risk-based sizing"
        risk_eur = equity * config.risk_per_trade_pct
        risk_per_unit = entry_price - stop_price
        quantity = risk_eur / risk_per_unit
        notional = quantity * entry_price
        why = (
            f"risk {config.risk_per_trade_pct:.2%} of equity "
            f"({risk_eur:.2f} EUR) over a {risk_per_unit / entry_price:.2%} stop"
        )
    else:
        return 0.0, f"unknown sizing method {method!r}"

    cap = equity * config.max_position_pct_of_equity
    if notional > cap:
        notional = cap
        why += f"; capped at {config.max_position_pct_of_equity:.0%} of equity"

    quantity = notional / entry_price
    if recent_volume_per_bar is not None and np.isfinite(recent_volume_per_bar) and recent_volume_per_bar > 0:
        volume_cap = recent_volume_per_bar * config.max_participation_of_recent_volume
        if quantity > volume_cap:
            quantity = volume_cap
            why += f"; capped at {config.max_participation_of_recent_volume:.0%} of recent bar volume"
    return float(max(0.0, quantity)), why


# --------------------------------------------------------------------------- #
# risk limits
# --------------------------------------------------------------------------- #
@dataclass
class RiskLimits:
    risk_per_trade_pct_min: float = 0.0025
    risk_per_trade_pct_max: float = 0.0050
    max_total_open_risk_pct: float = 0.02
    max_concurrent_positions: int = 3
    max_exposure_per_coin_pct: float = 0.15
    max_daily_loss_pct: float = 0.03
    max_weekly_loss_pct: float = 0.06
    max_drawdown_pause_pct: float = 0.10
    allow_averaging_down: bool = False
    allow_martingale: bool = False
    allow_leverage: bool = False
    allow_borrowing: bool = False
    max_chase_above_entry_zone_pct: float = 0.005

    @classmethod
    def from_config(cls, risk_config: dict[str, Any]) -> RiskLimits:
        limits = risk_config.get("limits", {})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in limits.items() if k in known})


@dataclass
class OpenPosition:
    market: str
    quantity: float
    entry_price: float
    stop_price: float
    entry_time: pd.Timestamp
    strategy: str = ""

    @property
    def notional(self) -> float:
        return self.quantity * self.entry_price

    @property
    def risk_eur(self) -> float:
        return max(0.0, (self.entry_price - self.stop_price) * self.quantity)


@dataclass
class RiskDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    scaled_quantity: float | None = None

    def __bool__(self) -> bool:
        return self.allowed


class RiskManager:
    """Enforces portfolio-level limits and circuit breakers."""

    def __init__(
        self,
        limits: RiskLimits,
        starting_equity: float = 10_000.0,
        circuit_breakers: dict[str, Any] | None = None,
    ):
        self.limits = limits
        self.starting_equity = float(starting_equity)
        self.equity = float(starting_equity)
        self.peak_equity = float(starting_equity)
        self.circuit_breakers = circuit_breakers or {}
        self.open_positions: dict[str, OpenPosition] = {}
        self.realised_pnl_by_day: dict[pd.Timestamp, float] = {}
        self.realised_pnl_by_week: dict[pd.Timestamp, float] = {}
        self.paused_until: pd.Timestamp | None = None
        self.pause_reason: str = ""
        self.consecutive_api_errors = 0

    # -- state -----------------------------------------------------------------
    @property
    def open_risk_eur(self) -> float:
        return sum(p.risk_eur for p in self.open_positions.values())

    @property
    def current_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return self.equity / self.peak_equity - 1.0

    def register_fill(self, position: OpenPosition) -> None:
        self.open_positions[f"{position.market}:{position.entry_time.isoformat()}"] = position

    def register_exit(self, key_or_market: str, realised_pnl: float, exit_time: pd.Timestamp) -> None:
        for key in list(self.open_positions):
            if key == key_or_market or key.startswith(f"{key_or_market}:"):
                self.open_positions.pop(key, None)
                break
        self.equity += realised_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        day = pd.Timestamp(exit_time).floor("1D")
        week = _week_start(pd.Timestamp(exit_time))
        self.realised_pnl_by_day[day] = self.realised_pnl_by_day.get(day, 0.0) + realised_pnl
        self.realised_pnl_by_week[week] = self.realised_pnl_by_week.get(week, 0.0) + realised_pnl
        self._check_circuit_breakers(exit_time)

    # -- gates -----------------------------------------------------------------
    def _check_circuit_breakers(self, now: pd.Timestamp) -> None:
        pause_minutes = float(self.circuit_breakers.get("pause_minutes_after_trip", 60))
        day = pd.Timestamp(now).floor("1D")
        week = _week_start(pd.Timestamp(now))
        daily = self.realised_pnl_by_day.get(day, 0.0)
        weekly = self.realised_pnl_by_week.get(week, 0.0)

        if daily <= -self.limits.max_daily_loss_pct * self.starting_equity:
            self._pause(now, pause_minutes, f"daily loss limit hit ({daily:.2f} EUR)")
        elif weekly <= -self.limits.max_weekly_loss_pct * self.starting_equity:
            self._pause(now, 24 * 60, f"weekly loss limit hit ({weekly:.2f} EUR)")
        elif self.current_drawdown <= -self.limits.max_drawdown_pause_pct:
            self._pause(now, 24 * 60, f"drawdown {self.current_drawdown:.2%} beyond pause threshold")

    def _pause(self, now: pd.Timestamp, minutes: float, reason: str) -> None:
        self.paused_until = pd.Timestamp(now) + pd.Timedelta(minutes=minutes)
        self.pause_reason = reason
        log.warning("Trading paused until %s: %s", self.paused_until, reason)

    def report_api_error(self, now: pd.Timestamp) -> None:
        self.consecutive_api_errors += 1
        threshold = int(self.circuit_breakers.get("pause_on_consecutive_api_errors", 5))
        if self.consecutive_api_errors >= threshold:
            self._pause(now, float(self.circuit_breakers.get("pause_minutes_after_trip", 60)),
                        f"{self.consecutive_api_errors} consecutive API errors")

    def report_api_success(self) -> None:
        self.consecutive_api_errors = 0

    def report_slippage(self, slippage_bps: float, now: pd.Timestamp) -> None:
        threshold = float(self.circuit_breakers.get("pause_on_slippage_bps_above", 100.0))
        if slippage_bps > threshold:
            self._pause(now, float(self.circuit_breakers.get("pause_minutes_after_trip", 60)),
                        f"abnormal slippage {slippage_bps:.1f} bps")

    def can_open(
        self,
        market: str,
        now: pd.Timestamp,
        quantity: float,
        entry_price: float,
        stop_price: float,
        entry_zone_high: float | None = None,
    ) -> RiskDecision:
        """Full pre-trade check. Returns the (possibly reduced) permitted size."""
        reasons: list[str] = []
        now = pd.Timestamp(now)

        if self.paused_until is not None and now < self.paused_until:
            return RiskDecision(False, [f"trading paused: {self.pause_reason}"])
        if self.paused_until is not None and now >= self.paused_until:
            self.paused_until = None
            self.pause_reason = ""

        if len(self.open_positions) >= self.limits.max_concurrent_positions:
            return RiskDecision(False, [f"already at {self.limits.max_concurrent_positions} open positions"])

        if not self.limits.allow_averaging_down and any(
            p.market == market for p in self.open_positions.values()
        ):
            return RiskDecision(False, [f"position already open in {market} (averaging down disabled)"])

        if entry_zone_high is not None and entry_price > entry_zone_high * (
            1.0 + self.limits.max_chase_above_entry_zone_pct
        ):
            return RiskDecision(
                False,
                [
                    f"price {entry_price:.6g} is more than "
                    f"{self.limits.max_chase_above_entry_zone_pct:.2%} above the tested entry zone"
                ],
            )

        if stop_price >= entry_price:
            return RiskDecision(False, ["stop is not below the entry price"])

        trade_risk = (entry_price - stop_price) * quantity
        max_trade_risk = self.equity * self.limits.risk_per_trade_pct_max
        if trade_risk > max_trade_risk and trade_risk > 0:
            scale = max_trade_risk / trade_risk
            quantity *= scale
            trade_risk = max_trade_risk
            reasons.append(f"size reduced to respect {self.limits.risk_per_trade_pct_max:.2%} per-trade risk")

        remaining_risk = self.equity * self.limits.max_total_open_risk_pct - self.open_risk_eur
        if remaining_risk <= 0:
            return RiskDecision(False, ["portfolio open-risk budget exhausted"])
        if trade_risk > remaining_risk:
            scale = remaining_risk / trade_risk
            quantity *= scale
            reasons.append("size reduced to respect the portfolio open-risk budget")

        notional = quantity * entry_price
        coin_cap = self.equity * self.limits.max_exposure_per_coin_pct
        if notional > coin_cap:
            quantity = coin_cap / entry_price
            reasons.append(f"size capped at {self.limits.max_exposure_per_coin_pct:.0%} exposure per coin")

        if not self.limits.allow_leverage:
            invested = sum(p.notional for p in self.open_positions.values())
            free_cash = max(0.0, self.equity - invested)
            if quantity * entry_price > free_cash:
                quantity = free_cash / entry_price
                reasons.append("size capped at available cash (no leverage, no borrowing)")

        if quantity <= 0:
            return RiskDecision(False, reasons + ["resulting size is zero"])
        return RiskDecision(True, reasons, scaled_quantity=quantity)

    def snapshot(self) -> dict[str, Any]:
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": self.current_drawdown,
            "open_positions": len(self.open_positions),
            "open_risk_eur": self.open_risk_eur,
            "open_risk_pct": self.open_risk_eur / self.equity if self.equity else 0.0,
            "paused_until": self.paused_until,
            "pause_reason": self.pause_reason,
        }


def make_sizer(
    sizing: SizingConfig,
) -> Callable[..., tuple[float, str]]:
    """Bind a :class:`SizingConfig` into the callable the backtester expects."""

    def _sizer(
        equity: float,
        entry_price: float,
        stop_price: float | None = None,
        realised_vol_per_bar: float | None = None,
        recent_volume_per_bar: float | None = None,
    ) -> tuple[float, str]:
        return position_quantity(
            sizing,
            equity=equity,
            entry_price=entry_price,
            stop_price=stop_price,
            realised_vol_per_bar=realised_vol_per_bar,
            recent_volume_per_bar=recent_volume_per_bar,
        )

    return _sizer
