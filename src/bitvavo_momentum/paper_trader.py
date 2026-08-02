"""Phase 11: paper trading.

Paper positions are marked with the **same execution model** the backtest uses,
so a paper track record can be compared with the research expectation directly.
If paper fills were assumed to be perfect, a divergence between paper and
research would tell us nothing.

Live-order safety
-----------------
:func:`assert_live_trading_allowed` is the only place in this codebase that could
ever authorise a real order, and it requires three independent switches:

1. ``live_trading.enabled: true`` in ``config/paper_trading.yaml``
2. ``ALLOW_LIVE_TRADING=true`` in the environment
3. an explicit ``cli_ack=True`` passed by a human-invoked script

Even with all three, no order-placement code exists in this repository - the
function raises with instructions rather than sending anything. Adding execution
is a deliberate, reviewable change, not a config toggle.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .execution_model import ExecutionModel
from .logging_utils import get_logger
from .risk_manager import OpenPosition, RiskManager
from .scanner import Signal
from .timeutils import now_utc, to_utc

log = get_logger(__name__)


class LiveTradingDisabled(RuntimeError):
    """Raised whenever something attempts to place a real order."""


def assert_live_trading_allowed(config: Config, cli_ack: bool = False) -> None:
    if not config.live_trading_allowed(cli_ack=cli_ack):
        raise LiveTradingDisabled(
            "Live order execution is disabled. It requires ALL of: "
            "live_trading.enabled=true in config/paper_trading.yaml, "
            "ALLOW_LIVE_TRADING=true in the environment, and an explicit CLI acknowledgement."
        )
    raise LiveTradingDisabled(
        "All three live-trading switches are set, but this repository contains no "
        "order-placement code by design. Implement and review an execution adapter "
        "deliberately before trading real money."
    )


@dataclass
class PaperPosition:
    market: str
    strategy: str
    signal_id: str
    entry_time_utc: pd.Timestamp
    entry_price: float
    quantity: float
    stop_price: float
    target_1: float
    target_2: float
    max_holding_minutes: int
    entry_fee_eur: float = 0.0
    entry_spread_cost_eur: float = 0.0
    entry_slippage_cost_eur: float = 0.0
    high_water_price: float = 0.0
    trailing_stop_pct: float | None = None
    notes: str = ""

    def unrealised(self, price: float) -> float:
        return (price - self.entry_price) * self.quantity

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperTrade:
    market: str
    strategy: str
    signal_id: str
    entry_time_utc: pd.Timestamp
    exit_time_utc: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    exit_reason: str
    gross_pnl_eur: float
    net_pnl_eur: float
    fees_eur: float
    spread_cost_eur: float
    slippage_cost_eur: float
    holding_minutes: float
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


class PaperTrader:
    """Tracks simulated positions driven by live prices."""

    def __init__(
        self,
        execution: ExecutionModel,
        risk: RiskManager,
        state_dir: str | Path = "data/results/paper",
        record_every_signal: bool = True,
    ) -> None:
        self.execution = execution
        self.risk = risk
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.record_every_signal = record_every_signal
        self.positions: dict[str, PaperPosition] = {}
        self.trades: list[PaperTrade] = []
        self.signal_log: list[dict[str, Any]] = []
        self._load_state()

    # -- persistence -----------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.state_dir / "paper_state.json"

    @property
    def journal_path(self) -> Path:
        return self.state_dir / "trade_journal.parquet"

    @property
    def signal_log_path(self) -> Path:
        return self.state_dir / "signal_log.parquet"

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Could not read paper state; starting fresh")
            return
        for row in payload.get("positions", []):
            row["entry_time_utc"] = to_utc(row["entry_time_utc"])
            position = PaperPosition(**row)
            self.positions[position.signal_id] = position
            self.risk.register_fill(
                OpenPosition(
                    market=position.market, quantity=position.quantity,
                    entry_price=position.entry_price, stop_price=position.stop_price,
                    entry_time=position.entry_time_utc, strategy=position.strategy,
                )
            )
        self.risk.equity = float(payload.get("equity", self.risk.equity))
        self.risk.peak_equity = float(payload.get("peak_equity", self.risk.peak_equity))
        log.info("Restored %d open paper positions", len(self.positions))

    def save_state(self) -> None:
        payload = {
            "saved_utc": now_utc().isoformat(),
            "equity": self.risk.equity,
            "peak_equity": self.risk.peak_equity,
            "positions": [
                {**p.to_row(), "entry_time_utc": p.entry_time_utc.isoformat()}
                for p in self.positions.values()
            ],
        }
        self.state_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        if self.trades:
            pd.DataFrame([t.to_row() for t in self.trades]).to_parquet(
                self.journal_path, index=False, compression="zstd"
            )
        if self.signal_log:
            pd.DataFrame(self.signal_log).to_parquet(
                self.signal_log_path, index=False, compression="zstd"
            )

    # -- signal handling -------------------------------------------------------
    def record_signal(self, signal: Signal, outcome: str, detail: str = "") -> None:
        """Record every signal - taken, rejected, expired - for later audit."""
        if not self.record_every_signal:
            return
        row = signal.to_row()
        row["outcome"] = outcome
        row["outcome_detail"] = detail
        row["recorded_utc"] = now_utc()
        self.signal_log.append(row)

    def open_from_signal(self, signal: Signal, reference_price: float | None = None) -> PaperPosition | None:
        """Attempt a simulated entry using the shared execution model."""
        if signal.signal_id in self.positions:
            self.record_signal(signal, "duplicate", "a position for this signal is already open")
            return None
        if signal.expires_utc is not None and now_utc() > to_utc(signal.expires_utc):
            self.record_signal(signal, "expired", "signal TTL elapsed before entry")
            return None
        if signal.action != "valid":
            self.record_signal(signal, "rejected", f"action={signal.action}")
            return None

        price = float(reference_price or signal.current_price)
        if price > signal.entry_zone_high:
            self.record_signal(signal, "rejected", "price moved above the tested entry zone")
            return None

        fill = self.execution.execute_market_order(
            reference_price=price, quantity=signal.position_size_base, side="buy",
            bar_volume=None, realised_vol=None, illiquidity_score=None,
        )
        if not fill.filled:
            self.record_signal(signal, "unfilled", fill.reason)
            return None

        position = PaperPosition(
            market=signal.market, strategy=signal.strategy, signal_id=signal.signal_id,
            entry_time_utc=now_utc(), entry_price=fill.price, quantity=fill.quantity,
            stop_price=signal.stop_loss, target_1=signal.target_1, target_2=signal.target_2,
            max_holding_minutes=signal.max_holding_minutes, entry_fee_eur=fill.fee_eur,
            entry_spread_cost_eur=fill.spread_cost_eur, entry_slippage_cost_eur=fill.slippage_cost_eur,
            high_water_price=fill.price, notes=fill.reason,
        )
        self.positions[signal.signal_id] = position
        self.risk.register_fill(
            OpenPosition(
                market=position.market, quantity=position.quantity, entry_price=position.entry_price,
                stop_price=position.stop_price, entry_time=position.entry_time_utc,
                strategy=position.strategy,
            )
        )
        self.record_signal(signal, "filled", f"{fill.quantity:.8f} @ {fill.price:.6g}")
        log.info("Paper entry %s %.8f @ %.6g", position.market, position.quantity, position.entry_price)
        return position

    # -- position management ---------------------------------------------------
    def update(self, prices: dict[str, float], as_of: datetime | pd.Timestamp | None = None) -> list[PaperTrade]:
        """Mark open positions and close any whose exit condition has triggered."""
        as_of = to_utc(as_of) if as_of is not None else now_utc()
        closed: list[PaperTrade] = []

        for signal_id, position in list(self.positions.items()):
            price = prices.get(position.market)
            if price is None or not np.isfinite(price) or price <= 0:
                continue
            position.high_water_price = max(position.high_water_price, price)

            stop = position.stop_price
            if position.trailing_stop_pct:
                stop = max(stop, position.high_water_price * (1.0 - position.trailing_stop_pct))

            held_minutes = (as_of - position.entry_time_utc).total_seconds() / 60.0
            reason = None
            if price <= stop:
                reason = "stop_loss"
            elif price >= position.target_2:
                reason = "target_2"
            elif price >= position.target_1:
                reason = "target_1"
            elif held_minutes >= position.max_holding_minutes:
                reason = "time_stop"
            if reason is None:
                continue

            fill = self.execution.execute_market_order(
                reference_price=price, quantity=position.quantity, side="sell",
            )
            exit_price = fill.price if fill.filled else price
            fees = position.entry_fee_eur + (fill.fee_eur if fill.filled else 0.0)
            spread = position.entry_spread_cost_eur + (fill.spread_cost_eur if fill.filled else 0.0)
            slippage = position.entry_slippage_cost_eur + (fill.slippage_cost_eur if fill.filled else 0.0)
            gross = (exit_price - position.entry_price) * position.quantity
            net = gross - fees

            trade = PaperTrade(
                market=position.market, strategy=position.strategy, signal_id=signal_id,
                entry_time_utc=position.entry_time_utc, exit_time_utc=as_of,
                entry_price=position.entry_price, exit_price=exit_price, quantity=position.quantity,
                exit_reason=reason, gross_pnl_eur=gross, net_pnl_eur=net, fees_eur=fees,
                spread_cost_eur=spread, slippage_cost_eur=slippage, holding_minutes=held_minutes,
                notes=fill.reason if fill.filled else "exit could not be filled; marked at last price",
            )
            self.trades.append(trade)
            closed.append(trade)
            self.positions.pop(signal_id, None)
            self.risk.register_exit(position.market, net, as_of)
            log.info("Paper exit %s %s @ %.6g (net %.2f EUR)", position.market, reason, exit_price, net)

        if closed:
            self.save_state()
        return closed

    # -- reporting -------------------------------------------------------------
    def portfolio_snapshot(self, prices: dict[str, float] | None = None) -> dict[str, Any]:
        prices = prices or {}
        unrealised = sum(
            p.unrealised(prices[p.market]) for p in self.positions.values() if p.market in prices
        )
        realised = sum(t.net_pnl_eur for t in self.trades)
        snapshot = self.risk.snapshot()
        snapshot.update(
            {
                "realised_pnl_eur": realised,
                "unrealised_pnl_eur": unrealised,
                "portfolio_value_eur": self.risk.equity + unrealised,
                "n_open_positions": len(self.positions),
                "n_closed_trades": len(self.trades),
                "n_signals_recorded": len(self.signal_log),
            }
        )
        return snapshot

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_row() for t in self.trades])

    def positions_frame(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame()
        return pd.DataFrame([p.to_row() for p in self.positions.values()])

    def signals_frame(self) -> pd.DataFrame:
        if not self.signal_log:
            return pd.DataFrame()
        return pd.DataFrame(self.signal_log)
