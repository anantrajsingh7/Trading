"""
Daily Scanner Engine.

Runs every strategy against every ticker in the universe.
Enriches raw signals with:
  - Relative strength rank vs universe
  - Company metadata
  - Risk-sized position parameters
  - Market regime gate

Returns a list of Opportunity objects sorted by score (descending).
Only opportunities scoring ≥ min_score (default 85) are returned.

Anti-overfit rule: scanner logic never looks at future data.
All enrichment uses only data available at scan time.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.config import get_config
from core.database import get_session, save_signal
from core.logging import get_logger
from core.models import Market, MarketRegime, Opportunity, Signal
from data.pipeline import get_data, get_data_batch
from data.universe import get_universe
from indicators.composite import enrich
from scanner.market_regime import classify_regime, regime_position_scale
from strategies.registry import all_strategies

log = get_logger("scanner.engine")


class Scanner:
    """
    Runs the daily scan for a given market.

    Usage:
        scanner = Scanner(market="US")
        opps = scanner.run()
    """

    def __init__(
        self,
        market: str = "US",
        cfg: Optional[dict] = None,
        provider: str = "yfinance",
    ):
        self.market    = market
        self.cfg       = cfg or get_config()
        self.provider  = provider
        self.strategies = all_strategies()
        self._min_score = float(
            self.cfg.get("scanner", {}).get("min_score", 85)
        )

    def run(
        self,
        tickers: Optional[list[str]] = None,
        save_to_db: bool = True,
    ) -> list[Opportunity]:
        """
        Run full daily scan.

        tickers — override universe (useful for targeted scans)
        save_to_db — persist results to SQLite signals table
        """
        log.info("Scanner starting | market=%s | min_score=%.0f", self.market, self._min_score)

        # ── Build universe ────────────────────────────────────────
        universe = tickers or get_universe(self.cfg, self.market)
        log.info("Universe: %d tickers", len(universe))

        # ── Fetch market proxy (SPY for regime) ───────────────────
        spy_df = get_data("SPY", period="2y", provider_name=self.provider)
        if spy_df is not None:
            spy_df = enrich(spy_df, patterns=False)

        regime_snap = classify_regime(spy_df)
        regime = regime_snap.regime
        log.info("Regime: %s", regime.value)

        if regime == MarketRegime.BEAR_STRONG:
            log.warning("BEAR_STRONG regime — no new longs, scan aborted")
            return []

        pos_scale = regime_position_scale(regime)

        # ── Fetch all data ────────────────────────────────────────
        raw_data = get_data_batch(
            universe, period="5y", provider_name=self.provider, show_progress=True
        )
        log.info("Data fetched: %d / %d tickers", len(raw_data), len(universe))

        # ── Enrich with indicators ────────────────────────────────
        enriched: dict[str, pd.DataFrame] = {}
        for ticker, df in tqdm(raw_data.items(), desc="Computing indicators"):
            try:
                enriched[ticker] = enrich(df, patterns=True)
            except Exception as e:
                log.debug("%s: enrich error — %s", ticker, e)

        # ── Compute RS rank for entire universe ───────────────────
        rs_ranks = self._compute_rs_ranks(enriched)

        # ── Run strategies ────────────────────────────────────────
        raw_signals: list[tuple[str, Signal]] = []
        for ticker, df in enriched.items():
            if len(df) < 252:
                continue
            for strategy in self.strategies:
                try:
                    signal = strategy.generate_signal(df, ticker, self.cfg)
                    if signal is not None:
                        raw_signals.append((ticker, signal))
                except Exception as e:
                    log.debug("%s/%s error: %s", ticker, strategy.name, e)

        log.info("Raw signals: %d", len(raw_signals))

        # ── Enrich signals → Opportunities ────────────────────────
        opportunities: list[Opportunity] = []
        for ticker, signal in raw_signals:
            rs_rank = rs_ranks.get(ticker, 50.0)
            opp = self._enrich_signal(
                signal      = signal,
                df          = enriched[ticker],
                rs_rank     = rs_rank,
                pos_scale   = pos_scale,
                regime      = regime,
            )
            if opp.score >= self._min_score:
                opportunities.append(opp)

        # ── Sort by score ─────────────────────────────────────────
        opportunities.sort(key=lambda o: o.score, reverse=True)
        log.info("Qualified opportunities: %d (score ≥ %.0f)", len(opportunities), self._min_score)

        # ── Persist to DB ─────────────────────────────────────────
        if save_to_db and opportunities:
            session = get_session()
            try:
                for opp in opportunities:
                    save_signal(opp, session)
            finally:
                session.close()

        return opportunities

    # ────────────────────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────────────────────

    def _compute_rs_ranks(self, enriched: dict[str, pd.DataFrame]) -> dict[str, float]:
        """
        Compute relative strength rank for each ticker.
        RS score = (3M ROC * 0.4) + (6M ROC * 0.2) + (9M ROC * 0.2) + (12M ROC * 0.2)
        Ranked as percentile vs universe.
        """
        scores: dict[str, float] = {}
        for ticker, df in enriched.items():
            try:
                c = df["close"]
                if len(c) < 252:
                    continue
                roc3  = (c.iloc[-1] / c.iloc[-63]  - 1) * 100 if len(c) >= 63  else 0
                roc6  = (c.iloc[-1] / c.iloc[-126] - 1) * 100 if len(c) >= 126 else 0
                roc9  = (c.iloc[-1] / c.iloc[-189] - 1) * 100 if len(c) >= 189 else 0
                roc12 = (c.iloc[-1] / c.iloc[-252] - 1) * 100 if len(c) >= 252 else 0
                scores[ticker] = 0.4 * roc3 + 0.2 * roc6 + 0.2 * roc9 + 0.2 * roc12
            except Exception:
                pass

        if not scores:
            return {}

        values = list(scores.values())
        ranks: dict[str, float] = {}
        for ticker, score in scores.items():
            rank = sum(1 for v in values if v <= score) / len(values) * 100
            ranks[ticker] = rank

        return ranks

    def _enrich_signal(
        self,
        signal: Signal,
        df: pd.DataFrame,
        rs_rank: float,
        pos_scale: float,
        regime: MarketRegime,
    ) -> Opportunity:
        """Convert a raw Signal into a fully enriched Opportunity."""
        last  = df.iloc[-1]
        close = float(last["close"])

        # Risk calculations
        risk_per_share = signal.entry_price - signal.stop_loss
        if risk_per_share <= 0:
            risk_per_share = close * 0.07

        account_size = float(self.cfg.get("account", {}).get("size", 100_000))
        risk_pct     = float(self.cfg.get("account", {}).get("risk_per_trade", 0.01))
        max_single   = float(self.cfg.get("account", {}).get("max_single_position", 0.10))
        dollar_risk  = account_size * risk_pct * pos_scale
        shares       = max(1, int(dollar_risk / risk_per_share))

        # Cap position so it never exceeds the max-single-position limit
        # (protects small accounts from tight-stop setups sizing huge).
        max_pos_value = account_size * max_single
        if shares * signal.entry_price > max_pos_value:
            shares = max(1, int(max_pos_value / signal.entry_price))

        pos_usd      = shares * signal.entry_price

        rr_t1 = (signal.target_1 - signal.entry_price) / risk_per_share if risk_per_share else 0
        rr_t2 = (signal.target_2 - signal.entry_price) / risk_per_share if risk_per_share else 0

        # Composite score — blend strategy score with RS rank
        final_score = signal.score * 0.70 + rs_rank * 0.30

        # Risk warnings
        warnings = []
        if rs_rank < 50:
            warnings.append("Low RS rank")
        if regime in (MarketRegime.NEUTRAL, MarketRegime.BEAR_MODERATE):
            warnings.append(f"Regime={regime.value}")
        atr_v = float(last.get("atr_14", close * 0.02))
        if atr_v / close > 0.04:
            warnings.append("High daily ATR (>4%)")

        return Opportunity(
            date          = date.today(),
            ticker        = signal.ticker,
            strategy      = signal.strategy,
            score         = round(final_score, 1),
            confidence    = "HIGH" if final_score >= 90 else "MEDIUM",
            current_price = close,
            ideal_entry   = signal.entry_price,
            buy_zone_low  = signal.entry_price * 0.995,
            buy_zone_high = signal.entry_price * 1.005,
            stop_loss     = signal.stop_loss,
            target_1      = signal.target_1,
            target_2      = signal.target_2,
            target_3      = signal.target_3,
            risk_per_share       = risk_per_share,
            rr_ratio_t1          = round(rr_t1, 2),
            rr_ratio_t2          = round(rr_t2, 2),
            position_size_shares = shares,
            position_size_usd    = round(pos_usd, 2),
            dollar_risk          = round(dollar_risk, 2),
            atr       = atr_v,
            rs_rank   = round(rs_rank, 1),
            volume_ratio = float(last.get("rvol_20", 1.0)),
            adx       = float(last.get("adx", 0)),
            rsi       = float(last.get("rsi_14", 50)),
            why_it_qualifies = signal.notes,
            risk_warnings    = "; ".join(warnings) if warnings else "None",
            invalidation     = f"Close below ${signal.stop_loss:.2f}",
        )
