"""Event-driven backtester.

Execution rules that the whole system depends on
-----------------------------------------------
1. A signal derived from bar ``T`` can never fill before bar ``T + 1``. The
   :class:`~.strategies.EntryPlan` constructor rejects anything else.
2. Prices used for fills come from :mod:`execution_model`, never raw from the
   candle - fees, spread and slippage are applied on both sides.
3. When a single bar's range contains both the stop and the take-profit, minute
   candles cannot resolve the order of events. The **stop** is taken and the
   trade is flagged ``ambiguous_bar``. The share of ambiguous exits is reported;
   if it is large, the result depends on an assumption rather than on data.
4. Portfolio constraints are applied chronologically: positions occupy slots and
   risk budget from entry until exit, so a strategy cannot pretend to have taken
   twenty simultaneous trades on a EUR 10,000 account.
5. Every event produces a row - filled, rejected, unfilled or skipped - so the
   funnel from "event detected" to "trade taken" is always visible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .execution_model import ExecutionModel
from .logging_utils import get_logger
from .risk_manager import OpenPosition, RiskLimits, RiskManager, SizingConfig, make_sizer
from .strategies import ExitPolicy, Strategy
from .timeutils import interval_to_minutes

log = get_logger(__name__)

TRADE_COLUMNS = [
    "market", "strategy", "exit_policy", "scenario", "event_time", "event_spec",
    "event_lookback_return", "entry_time", "entry_price", "quantity", "notional_eur",
    "stop_price", "target_price", "exit_time", "exit_price", "exit_reason",
    "bars_held", "holding_minutes", "gross_return", "net_return", "gross_pnl_eur",
    "net_pnl_eur", "fees_eur", "spread_cost_eur", "slippage_cost_eur", "mfe_pct",
    "mae_pct", "ambiguous_bar", "fill_fraction", "entry_reason", "status",
]


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    funnel: dict[str, int]
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        if self.trades.empty:
            return 0
        return int((self.trades["status"] == "closed").sum())

    def closed_trades(self) -> pd.DataFrame:
        if self.trades.empty:
            return self.trades
        return self.trades[self.trades["status"] == "closed"].copy()


class Backtester:
    """Simulates one (strategy, exit policy, execution scenario) combination."""

    def __init__(
        self,
        execution_model: ExecutionModel,
        exit_policy: ExitPolicy,
        sizing: SizingConfig | None = None,
        limits: RiskLimits | None = None,
        starting_equity: float = 10_000.0,
        interval: str = "1m",
        max_holding_minutes: int = 1440,
        min_order_quote_eur: float = 5.0,
        amount_decimals: int = 8,
        circuit_breakers: dict[str, Any] | None = None,
        enforce_portfolio_limits: bool = True,
    ) -> None:
        self.execution = execution_model
        self.exit_policy = exit_policy
        self.sizing = sizing or SizingConfig()
        self.limits = limits or RiskLimits()
        self.starting_equity = float(starting_equity)
        self.interval = interval
        self.interval_minutes = interval_to_minutes(interval)
        self.max_holding_minutes = int(max_holding_minutes)
        self.min_order_quote_eur = float(min_order_quote_eur)
        self.amount_decimals = int(amount_decimals)
        self.circuit_breakers = circuit_breakers or {}
        self.enforce_portfolio_limits = enforce_portfolio_limits
        self.sizer: Callable[..., tuple[float, str]] = make_sizer(self.sizing)

    # -- level computation -----------------------------------------------------
    def _initial_stop(self, entry_price: float, event: pd.Series) -> float | None:
        policy = self.exit_policy
        if policy.atr_stop_multiple is not None:
            atr_value = float(event.get("atr_60m", np.nan))
            if np.isfinite(atr_value) and atr_value > 0:
                return entry_price - policy.atr_stop_multiple * atr_value
            return None
        if policy.stop_loss_pct is not None:
            return entry_price * (1.0 - policy.stop_loss_pct)
        if policy.chandelier_atr_multiple is not None or policy.trailing_stop_pct is not None:
            return entry_price * (1.0 - 0.10)  # wide backstop; trailing does the work
        return None

    def _initial_target(self, entry_price: float, event: pd.Series) -> float | None:
        policy = self.exit_policy
        if policy.atr_target_multiple is not None:
            atr_value = float(event.get("atr_60m", np.nan))
            if np.isfinite(atr_value) and atr_value > 0:
                return entry_price + policy.atr_target_multiple * atr_value
        if policy.take_profit_pct is not None:
            return entry_price * (1.0 + policy.take_profit_pct)
        return None

    # -- exit simulation -------------------------------------------------------
    def _simulate_exit(
        self,
        forward: pd.DataFrame,
        entry_offset: int,
        entry_price: float,
        quantity: float,
        stop_price: float | None,
        target_price: float | None,
        event: pd.Series,
    ) -> dict[str, Any]:
        policy = self.exit_policy
        max_bars = min(len(forward), entry_offset + int(self.max_holding_minutes / self.interval_minutes))
        time_stop_bars = (
            int(policy.time_stop_minutes / self.interval_minutes)
            if policy.time_stop_minutes is not None
            else None
        )

        atr_value = float(event.get("atr_60m", np.nan))
        realised_vol = float(event.get("realised_vol_60m", np.nan))
        illiquidity = float(event.get("illiquidity_rank", np.nan))

        running_high = entry_price
        running_low = entry_price
        current_stop = stop_price
        breakeven_armed = False
        remaining = quantity
        partial_taken = False
        realised_proceeds = 0.0
        fees = 0.0
        spread_cost = 0.0
        slippage_cost = 0.0
        ambiguous = False
        exit_reason = "max_holding"
        exit_offset = max_bars - 1
        exit_price = float(forward["close"].iloc[exit_offset]) if max_bars > entry_offset else entry_price

        for i in range(entry_offset, max_bars):
            bar = forward.iloc[i]
            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])
            bar_volume = float(bar.get("volume", np.nan))
            if not np.isfinite(bar_close):
                continue  # no-trade bar: nothing could have been executed

            running_high = max(running_high, bar_high)
            running_low = min(running_low, bar_low)

            # --- dynamic stop updates (using bars that have already closed) ---
            if policy.breakeven_after_pct is not None and not breakeven_armed:
                if running_high >= entry_price * (1.0 + policy.breakeven_after_pct):
                    breakeven_armed = True
                    current_stop = max(current_stop or 0.0, entry_price)
            if policy.trailing_stop_pct is not None:
                trail = running_high * (1.0 - policy.trailing_stop_pct)
                current_stop = max(current_stop or 0.0, trail)
            if policy.chandelier_atr_multiple is not None and np.isfinite(atr_value) and atr_value > 0:
                chandelier = running_high - policy.chandelier_atr_multiple * atr_value
                current_stop = max(current_stop or 0.0, chandelier)

            hit_stop = current_stop is not None and bar_low <= current_stop
            hit_target = target_price is not None and bar_high >= target_price

            # --- partial take-profit ------------------------------------------
            if (
                policy.partial_take_profit_pct is not None
                and not partial_taken
                and bar_high >= entry_price * (1.0 + policy.partial_take_profit_pct)
                and remaining > 0
            ):
                partial_qty = remaining * policy.partial_take_fraction
                level = entry_price * (1.0 + policy.partial_take_profit_pct)
                fill = self.execution.execute_market_order(
                    reference_price=level, quantity=partial_qty, side="sell",
                    bar_volume=bar_volume, realised_vol=realised_vol,
                    illiquidity_score=illiquidity, min_order_quote_eur=self.min_order_quote_eur,
                    amount_decimals=self.amount_decimals,
                )
                if fill.filled:
                    realised_proceeds += fill.notional_eur - fill.fee_eur
                    fees += fill.fee_eur
                    spread_cost += fill.spread_cost_eur
                    slippage_cost += fill.slippage_cost_eur
                    remaining -= fill.quantity
                    partial_taken = True

            # --- terminal exits -----------------------------------------------
            if hit_stop and hit_target:
                # Minute data cannot order the two touches. Take the stop.
                ambiguous = True
                fill = self.execution.execute_stop(
                    stop_price=current_stop, quantity=remaining, bar_open=bar_open,
                    bar_low=bar_low, bar_high=bar_high, bar_volume=bar_volume,
                    realised_vol=realised_vol, illiquidity_score=illiquidity,
                    min_order_quote_eur=self.min_order_quote_eur, amount_decimals=self.amount_decimals,
                )
                exit_reason = "stop_loss(ambiguous_bar)"
            elif hit_stop:
                fill = self.execution.execute_stop(
                    stop_price=current_stop, quantity=remaining, bar_open=bar_open,
                    bar_low=bar_low, bar_high=bar_high, bar_volume=bar_volume,
                    realised_vol=realised_vol, illiquidity_score=illiquidity,
                    min_order_quote_eur=self.min_order_quote_eur, amount_decimals=self.amount_decimals,
                )
                exit_reason = "trailing_stop" if policy.trailing_stop_pct or policy.chandelier_atr_multiple else "stop_loss"
                if breakeven_armed and current_stop is not None and abs(current_stop - entry_price) < 1e-12:
                    exit_reason = "breakeven_stop"
            elif hit_target:
                fill = self.execution.execute_market_order(
                    reference_price=target_price, quantity=remaining, side="sell",
                    bar_volume=bar_volume, realised_vol=realised_vol, illiquidity_score=illiquidity,
                    min_order_quote_eur=self.min_order_quote_eur, amount_decimals=self.amount_decimals,
                )
                exit_reason = "take_profit"
            else:
                # --- close-based exits (evaluated on the bar close) -----------
                triggered = None
                if policy.exit_below_vwap and "vwap_session" in forward.columns:
                    level = float(bar.get("vwap_session", np.nan))
                    if np.isfinite(level) and bar_close < level:
                        triggered = "vwap_loss"
                if triggered is None and policy.exit_below_ema_span is not None:
                    col = f"ema_{policy.exit_below_ema_span}"
                    if col in forward.columns:
                        level = float(bar.get(col, np.nan))
                        if np.isfinite(level) and bar_close < level:
                            triggered = f"ema{policy.exit_below_ema_span}_loss"
                if triggered is None and policy.max_adverse_excursion_pct is not None:
                    if running_low / entry_price - 1.0 <= -policy.max_adverse_excursion_pct:
                        triggered = "max_adverse_excursion"
                if triggered is None and time_stop_bars is not None and (i - entry_offset) >= time_stop_bars:
                    triggered = "time_stop"
                if triggered is None:
                    continue
                fill = self.execution.execute_market_order(
                    reference_price=bar_close, quantity=remaining, side="sell",
                    bar_volume=bar_volume, realised_vol=realised_vol, illiquidity_score=illiquidity,
                    min_order_quote_eur=self.min_order_quote_eur, amount_decimals=self.amount_decimals,
                )
                exit_reason = triggered

            if fill.filled:
                realised_proceeds += fill.notional_eur - fill.fee_eur
                fees += fill.fee_eur
                spread_cost += fill.spread_cost_eur
                slippage_cost += fill.slippage_cost_eur
                remaining -= fill.quantity
                exit_price = fill.price
            else:
                # Could not exit on this bar (thin book); mark to close and retry.
                exit_price = bar_close
            exit_offset = i
            if remaining <= 1e-12:
                break

        # Any residual quantity is marked out at the final available close.
        if remaining > 1e-12:
            final_bar = forward.iloc[min(exit_offset, len(forward) - 1)]
            final_close = float(final_bar["close"])
            fill = self.execution.execute_market_order(
                reference_price=final_close, quantity=remaining, side="sell",
                bar_volume=float(final_bar.get("volume", np.nan)), realised_vol=realised_vol,
                illiquidity_score=illiquidity, min_order_quote_eur=0.0,
                amount_decimals=self.amount_decimals,
            )
            if fill.filled:
                realised_proceeds += fill.notional_eur - fill.fee_eur
                fees += fill.fee_eur
                spread_cost += fill.spread_cost_eur
                slippage_cost += fill.slippage_cost_eur
                exit_price = fill.price
            else:
                realised_proceeds += final_close * remaining
                exit_price = final_close
            remaining = 0.0

        return {
            "exit_offset": exit_offset,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "proceeds_eur": realised_proceeds,
            "fees_eur": fees,
            "spread_cost_eur": spread_cost,
            "slippage_cost_eur": slippage_cost,
            "mfe_pct": running_high / entry_price - 1.0,
            "mae_pct": running_low / entry_price - 1.0,
            "ambiguous_bar": ambiguous,
        }

    # -- main loop -------------------------------------------------------------
    def run(
        self,
        events: pd.DataFrame,
        features_by_market: dict[str, pd.DataFrame],
        strategy: Strategy,
        label: str = "",
        seed: int | None = None,
    ) -> BacktestResult:
        self.execution.reset(seed)
        risk = RiskManager(self.limits, self.starting_equity, self.circuit_breakers)
        funnel = {
            "events": 0, "no_market_data": 0, "no_forward_data": 0, "no_entry_signal": 0,
            "risk_blocked": 0, "unfilled": 0, "filled": 0,
        }
        rows: list[dict[str, Any]] = []
        pending_exits: list[tuple[pd.Timestamp, str, float]] = []

        if events.empty:
            return BacktestResult(pd.DataFrame(columns=TRADE_COLUMNS), pd.DataFrame(), funnel, label)

        ordered = events.sort_values("event_time").reset_index(drop=True)
        max_forward_minutes = strategy.max_wait_minutes + self.max_holding_minutes
        forward_bars = int(max_forward_minutes / self.interval_minutes) + 2

        for _, event in ordered.iterrows():
            funnel["events"] += 1
            event_time = pd.Timestamp(event["event_time"])

            # Release capital/slots for trades that closed before this event.
            pending_exits.sort(key=lambda item: item[0])
            while pending_exits and pending_exits[0][0] <= event_time:
                exit_time, key, pnl = pending_exits.pop(0)
                risk.register_exit(key, pnl, exit_time)

            market = str(event["market"])
            features = features_by_market.get(market)
            if features is None or features.empty:
                funnel["no_market_data"] += 1
                rows.append(self._skip_row(event, strategy, "no_market_data"))
                continue

            pos = features.index.searchsorted(event_time, side="right")
            forward = features.iloc[pos : pos + forward_bars]
            if len(forward) < 2:
                funnel["no_forward_data"] += 1
                rows.append(self._skip_row(event, strategy, "no_forward_data"))
                continue

            plan = strategy.find_entry(event, forward, self.interval_minutes)
            if plan is None:
                funnel["no_entry_signal"] += 1
                rows.append(self._skip_row(event, strategy, "no_entry_signal"))
                continue

            exec_offset = plan.execution_offset
            if exec_offset >= len(forward):
                funnel["no_forward_data"] += 1
                rows.append(self._skip_row(event, strategy, "entry_beyond_data"))
                continue

            entry_bar = forward.iloc[exec_offset]
            reference_price = plan.reference_price or float(entry_bar["open"])
            if not np.isfinite(reference_price) or reference_price <= 0:
                funnel["unfilled"] += 1
                rows.append(self._skip_row(event, strategy, "invalid_entry_price"))
                continue

            entry_time = forward.index[exec_offset]
            stop_price = self._initial_stop(reference_price, event)
            target_price = self._initial_target(reference_price, event)

            quantity, _why = self.sizer(
                equity=risk.equity,
                entry_price=reference_price,
                stop_price=stop_price,
                realised_vol_per_bar=float(event.get("realised_vol_60m", np.nan)),
                recent_volume_per_bar=float(event.get("volume", np.nan)),
            )
            if quantity <= 0:
                funnel["risk_blocked"] += 1
                rows.append(self._skip_row(event, strategy, "zero_size"))
                continue

            if self.enforce_portfolio_limits:
                decision = risk.can_open(
                    market=market, now=entry_time, quantity=quantity,
                    entry_price=reference_price, stop_price=stop_price or 0.0,
                    entry_zone_high=plan.entry_zone_high,
                )
                if not decision.allowed:
                    funnel["risk_blocked"] += 1
                    rows.append(self._skip_row(event, strategy, "risk_blocked: " + "; ".join(decision.reasons)))
                    continue
                quantity = decision.scaled_quantity or quantity

            if plan.order_type == "limit" and plan.limit_price:
                fill = self.execution.execute_limit_order(
                    limit_price=plan.limit_price, quantity=quantity, side="buy",
                    bar_low=float(entry_bar["low"]), bar_high=float(entry_bar["high"]),
                    bar_volume=float(entry_bar.get("volume", np.nan)),
                    min_order_quote_eur=self.min_order_quote_eur, amount_decimals=self.amount_decimals,
                )
            else:
                fill = self.execution.execute_market_order(
                    reference_price=reference_price, quantity=quantity, side="buy",
                    bar_volume=float(entry_bar.get("volume", np.nan)),
                    realised_vol=float(event.get("realised_vol_60m", np.nan)),
                    illiquidity_score=float(event.get("illiquidity_rank", np.nan)),
                    min_order_quote_eur=self.min_order_quote_eur, amount_decimals=self.amount_decimals,
                )
            if not fill.filled:
                funnel["unfilled"] += 1
                rows.append(self._skip_row(event, strategy, f"unfilled: {fill.reason}"))
                continue

            funnel["filled"] += 1
            entry_price = fill.price
            filled_qty = fill.quantity
            stop_price = self._initial_stop(entry_price, event)
            target_price = self._initial_target(entry_price, event)

            outcome = self._simulate_exit(
                forward=forward, entry_offset=exec_offset, entry_price=entry_price,
                quantity=filled_qty, stop_price=stop_price, target_price=target_price, event=event,
            )

            entry_cost = entry_price * filled_qty
            total_fees = fill.fee_eur + outcome["fees_eur"]
            net_pnl = outcome["proceeds_eur"] - entry_cost - fill.fee_eur
            gross_pnl = (outcome["exit_price"] - entry_price) * filled_qty
            exit_time = forward.index[outcome["exit_offset"]]
            key = f"{market}:{pd.Timestamp(entry_time).isoformat()}"

            risk.register_fill(
                OpenPosition(
                    market=market, quantity=filled_qty, entry_price=entry_price,
                    stop_price=stop_price or entry_price * 0.9,
                    entry_time=pd.Timestamp(entry_time), strategy=strategy.name,
                )
            )
            pending_exits.append((pd.Timestamp(exit_time), key, net_pnl))

            rows.append(
                {
                    "market": market,
                    "strategy": strategy.name,
                    "exit_policy": self.exit_policy.name,
                    "scenario": self.execution.scenario.name,
                    "event_time": event_time,
                    "event_spec": event.get("event_spec", ""),
                    "event_lookback_return": event.get("event_lookback_return", np.nan),
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "quantity": filled_qty,
                    "notional_eur": entry_cost,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "exit_time": exit_time,
                    "exit_price": outcome["exit_price"],
                    "exit_reason": outcome["exit_reason"],
                    "bars_held": outcome["exit_offset"] - exec_offset + 1,
                    "holding_minutes": (outcome["exit_offset"] - exec_offset + 1) * self.interval_minutes,
                    "gross_return": gross_pnl / entry_cost if entry_cost else np.nan,
                    "net_return": net_pnl / entry_cost if entry_cost else np.nan,
                    "gross_pnl_eur": gross_pnl,
                    "net_pnl_eur": net_pnl,
                    "fees_eur": total_fees,
                    "spread_cost_eur": fill.spread_cost_eur + outcome["spread_cost_eur"],
                    "slippage_cost_eur": fill.slippage_cost_eur + outcome["slippage_cost_eur"],
                    "mfe_pct": outcome["mfe_pct"],
                    "mae_pct": outcome["mae_pct"],
                    "ambiguous_bar": outcome["ambiguous_bar"],
                    "fill_fraction": fill.fill_fraction,
                    "entry_reason": plan.reason,
                    "status": "closed",
                }
            )

        for exit_time, key, pnl in sorted(pending_exits, key=lambda item: item[0]):
            risk.register_exit(key, pnl, exit_time)

        trades = pd.DataFrame(rows)
        if not trades.empty:
            for col in TRADE_COLUMNS:
                if col not in trades.columns:
                    trades[col] = np.nan
            trades = trades[TRADE_COLUMNS]
        else:
            trades = pd.DataFrame(columns=TRADE_COLUMNS)

        equity = self.build_equity_curve(trades)
        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            funnel=funnel,
            label=label or f"{strategy.name}|{self.exit_policy.name}|{self.execution.scenario.name}",
            meta={
                "strategy": strategy.describe(),
                "exit_policy": self.exit_policy.to_dict(),
                "execution": self.execution.describe(),
                "sizing": self.sizing.__dict__,
                "limits": self.limits.__dict__,
                "starting_equity": self.starting_equity,
                "final_equity": risk.equity,
            },
        )

    def _skip_row(self, event: pd.Series, strategy: Strategy, status: str) -> dict[str, Any]:
        return {
            "market": event.get("market"),
            "strategy": strategy.name,
            "exit_policy": self.exit_policy.name,
            "scenario": self.execution.scenario.name,
            "event_time": pd.Timestamp(event["event_time"]),
            "event_spec": event.get("event_spec", ""),
            "event_lookback_return": event.get("event_lookback_return", np.nan),
            "status": status,
        }

    def build_equity_curve(self, trades: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
        """Closed-trade equity curve, resampled onto a regular grid."""
        closed = trades[trades["status"] == "closed"] if not trades.empty else trades
        if closed.empty:
            return pd.DataFrame(columns=["timestamp", "equity", "drawdown"])
        ordered = closed.sort_values("exit_time")
        equity = self.starting_equity + ordered["net_pnl_eur"].cumsum()
        curve = pd.DataFrame(
            {"timestamp": pd.to_datetime(ordered["exit_time"], utc=True).to_numpy(), "equity": equity.to_numpy()}
        )
        curve = curve.groupby("timestamp", as_index=False).last()
        grid = curve.set_index("timestamp").resample(freq).last().ffill()
        grid["equity"] = grid["equity"].fillna(self.starting_equity)
        grid["peak"] = grid["equity"].cummax()
        grid["drawdown"] = grid["equity"] / grid["peak"] - 1.0
        return grid.reset_index()


def run_grid(
    events: pd.DataFrame,
    features_by_market: dict[str, pd.DataFrame],
    strategies: list[Strategy],
    exit_policies: list[ExitPolicy],
    execution_models: dict[str, ExecutionModel],
    sizing: SizingConfig | None = None,
    limits: RiskLimits | None = None,
    starting_equity: float = 10_000.0,
    interval: str = "1m",
    max_holding_minutes: int = 1440,
    seed: int = 20260802,
    progress: bool = True,
) -> dict[str, BacktestResult]:
    """Run every (strategy x exit policy x scenario) combination."""
    results: dict[str, BacktestResult] = {}
    total = len(strategies) * len(exit_policies) * len(execution_models)
    done = 0
    for scenario_name, model in execution_models.items():
        for policy in exit_policies:
            engine = Backtester(
                execution_model=model, exit_policy=policy, sizing=sizing, limits=limits,
                starting_equity=starting_equity, interval=interval,
                max_holding_minutes=max_holding_minutes,
            )
            for strategy in strategies:
                label = f"{strategy.name}|{policy.name}|{scenario_name}"
                results[label] = engine.run(events, features_by_market, strategy, label=label, seed=seed)
                done += 1
                if progress and done % 10 == 0:
                    log.info("Backtest grid %d/%d", done, total)
    return results
