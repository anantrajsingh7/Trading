"""Phase 11: live momentum scanner and signal generation.

The scanner is the research system pointed at the present. It reuses the exact
same feature, event and level code as the backtest - if it did not, paper results
would not be comparable with research results and the whole exercise would be
decorative.

Guard rails
-----------
* The scanner refuses to run unless an **approved strategy card** exists
  (``data/results/approved_strategy.json``), written by the research pipeline
  after out-of-sample evaluation. Parameters cannot be invented at scan time.
* Every signal carries its historical evidence: sample size, win rate, net
  expectancy, worst historical drawdown - and an explicit warning when the sample
  is too small to support a conclusion.
* Signals are estimates. The wording in :meth:`Signal.to_text` never states an
  outcome as certain.
* Signals expire. An entry zone computed at 14:00 is not valid at 18:00, and an
  expired signal is recorded as expired rather than quietly refreshed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .execution_model import ExecutionScenario
from .features import build_features
from .logging_utils import get_logger
from .market_universe import MarketRules
from .risk_manager import RiskManager, SizingConfig, position_quantity
from .timeutils import format_display, now_utc, to_utc

log = get_logger(__name__)

ACTION_WATCH = "watch"
ACTION_VALID = "valid"
ACTION_EXTENDED = "extended"
ACTION_INVALID = "invalid"
ACTION_EXPIRED = "expired"

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_INSUFFICIENT = "insufficient_evidence"


class StrategyCardMissing(RuntimeError):
    """Raised when the scanner is started without an approved strategy."""


@dataclass
class StrategyCard:
    """The frozen, out-of-sample-validated configuration the scanner may use."""

    strategy: str
    event_lookback_minutes: int
    event_threshold: float
    exit_policy: dict[str, Any]
    filters: dict[str, Any] = field(default_factory=dict)
    allowed_regimes: list[str] | None = None
    historical_sample: int = 0
    historical_win_rate: float = float("nan")
    historical_net_expectancy: float = float("nan")
    historical_max_drawdown: float = float("nan")
    historical_avg_holding_minutes: float = float("nan")
    validated_on: str = ""
    approved_utc: str = ""
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> StrategyCard:
        p = Path(path)
        if not p.exists():
            raise StrategyCardMissing(
                f"No approved strategy card at {p}. Run the research pipeline and "
                "approve a strategy before scanning; the scanner will not invent parameters."
            )
        payload = json.loads(p.read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return p


@dataclass
class Signal:
    """One scanner output. Every field required by Phase 11 is present."""

    market: str
    timestamp_utc: pd.Timestamp
    current_price: float
    lookback_minutes: int
    lookback_return: float
    volume_ratio: float
    quote_volume_24h_eur: float
    spread_bps: float
    btc_regime: str
    market_regime: str
    strategy: str
    entry_zone_low: float
    entry_zone_high: float
    invalidation_level: float
    stop_loss: float
    target_1: float
    target_2: float
    trailing_rule: str
    max_holding_minutes: int
    reward_to_risk: float
    estimated_fee_bps: float
    estimated_slippage_bps: float
    position_size_base: float
    position_size_eur: float
    confidence: str
    reasons_passed: list[str]
    reasons_could_fail: list[str]
    resembles_historical_sample: bool
    historical_sample_size: int
    historical_win_rate: float
    historical_net_expectancy: float
    historical_max_drawdown: float
    action: str = ACTION_WATCH
    sample_warning: str = ""
    signal_id: str = ""
    expires_utc: pd.Timestamp | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["reasons_passed"] = " | ".join(self.reasons_passed)
        row["reasons_could_fail"] = " | ".join(self.reasons_could_fail)
        row["timestamp_amsterdam"] = format_display(self.timestamp_utc)
        return row

    def to_text(self) -> str:
        """Human-readable summary. Deliberately hedged - these are estimates."""
        lines = [
            f"{self.market} | {format_display(self.timestamp_utc)}",
            f"  +{self.lookback_return:.1%} over {self.lookback_minutes}m at {self.current_price:.6g} EUR",
            f"  setup: {self.strategy} ({self.action}, confidence: {self.confidence})",
            f"  entry zone {self.entry_zone_low:.6g}-{self.entry_zone_high:.6g} | "
            f"stop {self.stop_loss:.6g} | T1 {self.target_1:.6g} | T2 {self.target_2:.6g}",
            f"  estimated reward:risk {self.reward_to_risk:.2f} | "
            f"suggested size {self.position_size_eur:.2f} EUR",
            f"  costs assumed: {self.estimated_fee_bps:.1f} bps fees + "
            f"{self.estimated_slippage_bps:.1f} bps slippage/spread per side",
            f"  regime: BTC {self.btc_regime}, market {self.market_regime}",
            f"  historical analogues: n={self.historical_sample_size}, "
            f"win rate {self.historical_win_rate:.0%}, "
            f"net expectancy {self.historical_net_expectancy:+.2%} per trade "
            f"(estimates from past data, not a forecast)",
            f"  passed: {'; '.join(self.reasons_passed) or 'n/a'}",
            f"  could fail because: {'; '.join(self.reasons_could_fail) or 'n/a'}",
        ]
        if self.sample_warning:
            lines.append(f"  WARNING: {self.sample_warning}")
        return "\n".join(lines)


class MomentumScanner:
    """Detects live events and turns the validated ones into signals."""

    def __init__(
        self,
        card: StrategyCard,
        scanner_config: dict[str, Any],
        signal_quality: dict[str, Any],
        scenario: ExecutionScenario,
        sizing: SizingConfig,
        risk_manager: RiskManager,
        market_rules: dict[str, MarketRules] | None = None,
    ) -> None:
        self.card = card
        self.config = scanner_config
        self.quality = signal_quality
        self.scenario = scenario
        self.sizing = sizing
        self.risk = risk_manager
        self.market_rules = market_rules or {}
        self._last_signal_time: dict[str, pd.Timestamp] = {}
        self._signals_today: dict[tuple[str, str], int] = {}

    # -- gating ---------------------------------------------------------------
    def _data_is_fresh(self, candles: pd.DataFrame, as_of: pd.Timestamp) -> tuple[bool, str]:
        if candles.empty:
            return False, "no candle data"
        last = to_utc(candles["timestamp"].max())
        age = (to_utc(as_of) - last).total_seconds()
        limit = float(self.config.get("max_data_age_seconds", 90))
        if age > limit:
            return False, f"data is {age:.0f}s old (limit {limit:.0f}s)"
        return True, ""

    def _duplicate_suppressed(self, market: str, as_of: pd.Timestamp) -> bool:
        last = self._last_signal_time.get(market)
        if last is None:
            return False
        window = pd.Timedelta(minutes=float(self.config.get("duplicate_suppression_minutes", 240)))
        return (to_utc(as_of) - last) < window

    def _daily_cap_reached(self, market: str, as_of: pd.Timestamp) -> bool:
        key = (market, to_utc(as_of).strftime("%Y-%m-%d"))
        cap = int(self.config.get("max_signals_per_market_per_day", 3))
        return self._signals_today.get(key, 0) >= cap

    # -- signal construction ---------------------------------------------------
    def evaluate_market(
        self,
        market: str,
        candles: pd.DataFrame,
        book: dict[str, Any] | None = None,
        ticker_24h: dict[str, Any] | None = None,
        btc_regime: str = "unknown",
        market_regime: str = "unknown",
        as_of: pd.Timestamp | None = None,
    ) -> Signal | None:
        """Evaluate one market. Returns a :class:`Signal` or ``None``."""
        as_of = to_utc(as_of) if as_of is not None else now_utc()

        fresh, why = self._data_is_fresh(candles, as_of)
        if not fresh:
            log.debug("%s skipped: %s", market, why)
            return None

        features = build_features(candles, "1m")
        if features.empty or len(features) < self.card.event_lookback_minutes + 10:
            return None
        latest = features.iloc[-1]

        lookback_col = f"ret_{self.card.event_lookback_minutes}m"
        lookback_return = float(latest.get(lookback_col, np.nan))
        if not np.isfinite(lookback_return) or lookback_return < self.card.event_threshold:
            return None

        if self._duplicate_suppressed(market, as_of) or self._daily_cap_reached(market, as_of):
            log.debug("%s suppressed: duplicate or daily cap", market)
            return None

        price = float(latest["close"])
        atr_value = float(latest.get("atr_60m", np.nan))
        spread_bps = self._spread_bps(book, latest)
        quote_volume_24h = self._quote_volume_24h(ticker_24h, latest)

        passed: list[str] = [
            f"+{lookback_return:.1%} over {self.card.event_lookback_minutes}m "
            f"(threshold {self.card.event_threshold:.1%})"
        ]
        could_fail: list[str] = [
            "the sample of similar historical events is small relative to the variety of market conditions",
            "spread and depth can widen sharply during a fast move, making the modelled fill optimistic",
            "momentum events cluster, so several open positions can fail together",
        ]

        max_spread = float(self.config.get("max_spread_bps", 60))
        if spread_bps > max_spread:
            log.debug("%s rejected: spread %.1f bps > %.1f", market, spread_bps, max_spread)
            return None
        passed.append(f"spread {spread_bps:.1f} bps within the {max_spread:.0f} bps limit")

        min_volume = float(self.config.get("min_median_daily_quote_volume_eur", 25000))
        if quote_volume_24h < min_volume:
            return None
        passed.append(f"24h quote volume {quote_volume_24h:,.0f} EUR above the {min_volume:,.0f} EUR floor")

        for column, condition in (self.card.filters or {}).items():
            value = float(latest.get(column, np.nan))
            op, threshold = condition if isinstance(condition, list | tuple) else (">=", condition)
            if not np.isfinite(value):
                could_fail.append(f"{column} unavailable - filter could not be evaluated")
                return None
            ok = value >= float(threshold) if op == ">=" else value <= float(threshold)
            if not ok:
                log.debug("%s rejected by filter %s %s %s (was %s)", market, column, op, threshold, value)
                return None
            passed.append(f"{column} {op} {threshold} (is {value:.4g})")

        if self.card.allowed_regimes is not None and btc_regime not in self.card.allowed_regimes:
            log.debug("%s rejected: regime %s not in %s", market, btc_regime, self.card.allowed_regimes)
            return None
        if self.card.allowed_regimes is not None:
            passed.append(f"regime {btc_regime} is in the validated set")

        # -- levels -----------------------------------------------------------
        policy = self.card.exit_policy or {}
        stop_pct = policy.get("stop_loss_pct")
        atr_multiple = policy.get("atr_stop_multiple")
        if atr_multiple and np.isfinite(atr_value) and atr_value > 0:
            stop = price - float(atr_multiple) * atr_value
            trailing_rule = f"chandelier/ATR stop at {atr_multiple} x ATR(60m)"
        elif stop_pct:
            stop = price * (1.0 - float(stop_pct))
            trailing_rule = policy.get("trailing_stop_pct") and (
                f"trail {float(policy['trailing_stop_pct']):.1%} below the running high"
            ) or "fixed stop, no trail"
        else:
            return None

        take_profit = policy.get("take_profit_pct")
        target_1 = price * (1.0 + float(take_profit)) if take_profit else price + 2 * (price - stop)
        target_2 = price + 2.0 * (target_1 - price)

        entry_low = price
        entry_high = price * (1.0 + float(self.config.get("extended_above_entry_zone_pct", 0.005)))
        risk_per_unit = price - stop
        if risk_per_unit <= 0:
            return None
        reward_to_risk = (target_1 - price) / risk_per_unit

        fee_bps = self.scenario.taker_fee_bps
        slippage_bps = self.scenario.half_spread_bps + self.scenario.slippage_base_bps
        # Costs are paid on both sides; a 1.5 R:R gross can be well under 1 net.
        round_trip_cost = 2 * (fee_bps + slippage_bps) * 1e-4 * price
        net_reward = (target_1 - price) - round_trip_cost
        reward_to_risk_net = net_reward / risk_per_unit if risk_per_unit > 0 else np.nan

        min_rr = float(self.quality.get("min_reward_to_risk", 1.5))
        if not np.isfinite(reward_to_risk_net) or reward_to_risk_net < min_rr:
            log.debug("%s rejected: net R:R %.2f < %.2f", market, reward_to_risk_net, min_rr)
            return None
        passed.append(f"net reward:risk {reward_to_risk_net:.2f} after modelled costs")

        quantity, _why = position_quantity(
            self.sizing, equity=self.risk.equity, entry_price=price, stop_price=stop,
            realised_vol_per_bar=float(latest.get("realised_vol_60m", np.nan)),
            recent_volume_per_bar=float(latest.get("volume", np.nan)),
        )
        decision = self.risk.can_open(
            market=market, now=as_of, quantity=quantity, entry_price=price,
            stop_price=stop, entry_zone_high=entry_high,
        )
        action = ACTION_VALID
        if not decision.allowed:
            action = ACTION_INVALID
            could_fail.extend(decision.reasons)
        else:
            quantity = decision.scaled_quantity or quantity
            passed.extend(decision.reasons)

        rules = self.market_rules.get(market)
        if rules is not None:
            quantity = rules.round_amount(quantity)
            if not rules.meets_minimum(quantity, price):
                action = ACTION_INVALID
                could_fail.append("size below the market's minimum order value")

        confidence, warning = self._confidence()

        signal = Signal(
            market=market,
            timestamp_utc=as_of,
            current_price=price,
            lookback_minutes=self.card.event_lookback_minutes,
            lookback_return=lookback_return,
            volume_ratio=float(latest.get("rel_volume_60m", np.nan)),
            quote_volume_24h_eur=quote_volume_24h,
            spread_bps=spread_bps,
            btc_regime=btc_regime,
            market_regime=market_regime,
            strategy=self.card.strategy,
            entry_zone_low=entry_low,
            entry_zone_high=entry_high,
            invalidation_level=stop,
            stop_loss=stop,
            target_1=target_1,
            target_2=target_2,
            trailing_rule=str(trailing_rule),
            max_holding_minutes=int(policy.get("time_stop_minutes") or 240),
            reward_to_risk=float(reward_to_risk_net),
            estimated_fee_bps=fee_bps,
            estimated_slippage_bps=slippage_bps,
            position_size_base=float(quantity),
            position_size_eur=float(quantity * price),
            confidence=confidence,
            reasons_passed=passed,
            reasons_could_fail=could_fail,
            resembles_historical_sample=self._resembles_sample(latest, lookback_return),
            historical_sample_size=self.card.historical_sample,
            historical_win_rate=self.card.historical_win_rate,
            historical_net_expectancy=self.card.historical_net_expectancy,
            historical_max_drawdown=self.card.historical_max_drawdown,
            action=action,
            sample_warning=warning,
            signal_id=f"{market}-{as_of.strftime('%Y%m%dT%H%M%SZ')}",
            expires_utc=as_of + pd.Timedelta(minutes=float(self.config.get("signal_ttl_minutes", 60))),
        )
        self._last_signal_time[market] = as_of
        key = (market, as_of.strftime("%Y-%m-%d"))
        self._signals_today[key] = self._signals_today.get(key, 0) + 1
        # `reward_to_risk` reported above is the NET figure; keep gross for context.
        signal.reasons_passed.append(f"gross reward:risk before costs was {reward_to_risk:.2f}")
        return signal

    # -- helpers ---------------------------------------------------------------
    def _spread_bps(self, book: dict[str, Any] | None, latest: pd.Series) -> float:
        if book:
            try:
                bid = float(book["bids"][0][0])
                ask = float(book["asks"][0][0])
                if bid > 0 and ask > 0:
                    return (ask - bid) / ((ask + bid) / 2.0) * 10_000
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        proxy = float(latest.get("spread_proxy_bps", np.nan))
        return proxy if np.isfinite(proxy) else float("inf")

    def _quote_volume_24h(self, ticker: dict[str, Any] | None, latest: pd.Series) -> float:
        if ticker:
            for key in ("volumeQuote", "volume_quote"):
                try:
                    return float(ticker[key])
                except (KeyError, TypeError, ValueError):
                    continue
        value = float(latest.get("quote_volume_1440m", np.nan))
        return value if np.isfinite(value) else 0.0

    def _confidence(self) -> tuple[str, str]:
        bands = self.quality.get("confidence_bands", {})
        n = self.card.historical_sample
        expectancy_bps = self.card.historical_net_expectancy * 10_000 if np.isfinite(
            self.card.historical_net_expectancy
        ) else -np.inf
        for name in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW):
            band = bands.get(name, {})
            if n >= int(band.get("min_sample", 10**9)) and expectancy_bps >= float(
                band.get("min_net_expectancy_bps", 10**9)
            ):
                return name, ""
        minimum = int(self.quality.get("min_historical_sample", 50))
        warning = (
            f"only {n} comparable historical events were available "
            f"(minimum for a supported estimate is {minimum}); "
            "treat every statistic attached to this signal as provisional"
        )
        return CONFIDENCE_INSUFFICIENT, warning

    def _resembles_sample(self, latest: pd.Series, lookback_return: float) -> bool:
        """Crude similarity check against the conditions the strategy was tested in."""
        if self.card.historical_sample <= 0:
            return False
        checks = [
            lookback_return <= 3 * max(self.card.event_threshold, 1e-9),
            np.isfinite(float(latest.get("rel_volume_60m", np.nan))),
            np.isfinite(float(latest.get("atr_60m", np.nan))),
        ]
        return all(checks)


def signals_to_frame(signals: list[Signal]) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame()
    return pd.DataFrame([s.to_row() for s in signals])
