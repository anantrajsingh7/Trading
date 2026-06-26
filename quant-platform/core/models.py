"""
Canonical data models for the entire platform.

Design principle: Every component in the system communicates through
these dataclasses. This is the single source of truth for what a
Trade, Signal, BacktestResult, or Opportunity looks like.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional


# ============================================================
# Enumerations
# ============================================================

class Market(str, Enum):
    US = "US"
    INDIA = "INDIA"
    CRYPTO = "CRYPTO"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class ExitReason(str, Enum):
    TARGET_1 = "TARGET_1"
    TARGET_2 = "TARGET_2"
    TARGET_3 = "TARGET_3"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    MAX_HOLD = "MAX_HOLD"
    MANUAL = "MANUAL"
    END_OF_BACKTEST = "END_OF_BACKTEST"


class MarketRegime(str, Enum):
    BULL_STRONG = "BULL_STRONG"       # All systems go
    BULL_MODERATE = "BULL_MODERATE"   # Selective aggression
    NEUTRAL = "NEUTRAL"               # Reduced size, high quality only
    BEAR_MODERATE = "BEAR_MODERATE"   # Defensive / cash
    BEAR_STRONG = "BEAR_STRONG"       # Full cash, no longs


class SignalStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


# ============================================================
# Market Data
# ============================================================

@dataclass
class OHLCV:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: Optional[float] = None

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def true_range(self) -> float:
        return self.high - self.low


@dataclass
class TickerInfo:
    ticker: str
    name: str
    sector: str
    industry: str
    market: Market
    market_cap: float = 0.0
    country: str = "US"


# ============================================================
# Signals & Opportunities
# ============================================================

@dataclass
class Signal:
    """Raw signal from a strategy — before risk sizing."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str = ""
    strategy: str = ""
    date: date = field(default_factory=date.today)
    direction: Direction = Direction.LONG
    strength: SignalStrength = SignalStrength.MODERATE
    score: float = 0.0             # 0-100
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0
    atr: float = 0.0
    notes: str = ""


@dataclass
class Opportunity:
    """
    Fully enriched trade opportunity — what the scanner outputs.
    Everything a trader needs to make a decision.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    date: date = field(default_factory=date.today)

    # Identity
    ticker: str = ""
    company: str = ""
    sector: str = ""
    industry: str = ""
    market: Market = Market.US

    # Strategy
    strategy: str = ""
    score: float = 0.0             # 0–100 composite score
    confidence: str = ""           # "HIGH" / "MEDIUM"

    # Price levels
    current_price: float = 0.0
    ideal_entry: float = 0.0
    buy_zone_low: float = 0.0
    buy_zone_high: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0

    # Risk
    risk_per_share: float = 0.0
    rr_ratio_t1: float = 0.0
    rr_ratio_t2: float = 0.0
    position_size_shares: int = 0
    position_size_usd: float = 0.0
    dollar_risk: float = 0.0
    expected_hold_days: int = 0

    # Context
    atr: float = 0.0
    rs_rank: float = 0.0           # 0–100 percentile vs universe
    rs_vs_spy: float = 0.0
    volume_ratio: float = 0.0      # Today vol / 50-day avg vol
    adx: float = 0.0
    rsi: float = 0.0

    # Qualitative
    why_it_qualifies: str = ""
    catalysts: str = ""
    risk_warnings: str = ""
    invalidation: str = ""
    fundamental_snapshot: str = ""

    def to_dict(self) -> dict:
        return {k: (v.value if isinstance(v, Enum) else v)
                for k, v in self.__dict__.items()}


# ============================================================
# Trades
# ============================================================

@dataclass
class Trade:
    """A live or historical trade record."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str = ""
    company: str = ""
    strategy: str = ""
    market: Market = Market.US
    direction: Direction = Direction.LONG

    entry_date: Optional[date] = None
    entry_price: float = 0.0
    shares: int = 0

    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0

    exit_date: Optional[date] = None
    exit_price: float = 0.0
    exit_reason: Optional[ExitReason] = None

    status: TradeStatus = TradeStatus.OPEN
    notes: str = ""
    score_at_entry: float = 0.0

    @property
    def pnl(self) -> float:
        if self.exit_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def holding_days(self) -> int:
        if not self.entry_date:
            return 0
        end = self.exit_date or date.today()
        return (end - self.entry_date).days

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


# ============================================================
# Backtest
# ============================================================

@dataclass
class BacktestTrade:
    ticker: str
    strategy: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    holding_days: int
    exit_reason: ExitReason
    mae: float = 0.0               # Maximum Adverse Excursion
    mfe: float = 0.0               # Maximum Favorable Excursion


@dataclass
class PerformanceMetrics:
    """
    Comprehensive strategy performance metrics.
    The full set used by institutional quant funds.
    """
    strategy: str
    period: str
    n_trades: int = 0
    n_winners: int = 0
    n_losers: int = 0

    # Returns
    total_return: float = 0.0
    cagr: float = 0.0
    annual_return: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    longest_drawdown_days: int = 0
    recovery_factor: float = 0.0

    # Trade statistics
    win_rate: float = 0.0
    loss_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_win_loss_ratio: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0        # per trade, in R multiples
    kelly_pct: float = 0.0

    # Frequency
    avg_hold_days: float = 0.0
    trades_per_year: float = 0.0

    # Composite
    consistency_score: float = 0.0  # 0-100, penalises lumpy returns
    robustness_rank: float = 0.0    # Final ranking score

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class BacktestResult:
    strategy: str
    period: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    trades: list[BacktestTrade] = field(default_factory=list)
    metrics: Optional[PerformanceMetrics] = None
    equity_curve: Optional[object] = None      # pd.Series

    def to_dict(self) -> dict:
        d = {
            "strategy": self.strategy,
            "period": self.period,
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
        }
        if self.metrics:
            d.update(self.metrics.to_dict())
        return d


# ============================================================
# Market Regime
# ============================================================

@dataclass
class RegimeSnapshot:
    date: date
    regime: MarketRegime
    spy_trend: str
    spy_price: float
    spy_vs_ema50: float
    spy_vs_ema200: float
    vix: float
    breadth_pct_above_200: float
    advance_decline: float
    leading_sectors: list[str] = field(default_factory=list)
    lagging_sectors: list[str] = field(default_factory=list)
    notes: str = ""
