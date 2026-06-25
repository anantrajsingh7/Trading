"""
Daily scanner: finds stocks that match strategy rules using the most
recent available data. Scores each candidate and builds the watchlist.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategies import STRATEGIES, STRATEGY_NAMES, get_signals
from risk import build_trade_setup, TradeSetup

log = logging.getLogger("swing.scanner")

# Company names fallback (expanded later via yfinance .info)
_COMPANY_CACHE: dict[str, str] = {}


@dataclass
class Candidate:
    ticker: str
    company: str
    strategy_key: str
    strategy_name: str
    current_price: float
    entry: float
    stop_loss: float
    target1: float
    target2: float
    risk_per_share: float
    rr_ratio: float
    shares: int
    position_value: float
    setup_score: float
    rs_spy: float
    rs_qqq: float
    volume_ratio: float
    trend_status: str
    why_qualifies: str
    why_may_fail: str
    invalidation: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "company": self.company,
            "strategy": self.strategy_name,
            "current_price": self.current_price,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "target1": self.target1,
            "target2": self.target2,
            "risk_per_share": self.risk_per_share,
            "rr_ratio": self.rr_ratio,
            "shares": self.shares,
            "position_value": self.position_value,
            "setup_score": self.setup_score,
            "rs_spy": round(self.rs_spy * 100, 2) if not np.isnan(self.rs_spy) else "N/A",
            "rs_qqq": round(self.rs_qqq * 100, 2) if not np.isnan(self.rs_qqq) else "N/A",
            "volume_ratio": round(self.volume_ratio, 2),
            "trend_status": self.trend_status,
            "why_qualifies": self.why_qualifies,
            "why_may_fail": self.why_may_fail,
            "invalidation": self.invalidation,
        }


def _get_company_name(ticker: str) -> str:
    if ticker in _COMPANY_CACHE:
        return _COMPANY_CACHE[ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ticker
        _COMPANY_CACHE[ticker] = name
        return name
    except Exception:
        return ticker


def _score_candidate(df: pd.DataFrame, strategy_key: str,
                     setup: TradeSetup) -> tuple[float, str, str, str, str]:
    """
    Returns (score, trend_status, why_qualifies, why_may_fail, invalidation).
    Score out of 10:
      Trend alignment         2 pts
      Volume confirmation     2 pts
      Relative strength       2 pts
      Clean entry/stop        2 pts
      RR >= 2:1               2 pts
    """
    last = df.iloc[-1]
    close = float(last["close"])
    score = 0.0
    reasons: list[str] = []
    risks: list[str] = []

    # 1. Trend alignment (2 pts)
    above_200 = close > float(last["ema200"])
    ema20_gt_50 = float(last["ema20"]) > float(last["ema50"])
    ema50_gt_200 = float(last["ema50"]) > float(last["ema200"])
    trend_pts = sum([above_200, ema20_gt_50, ema50_gt_200]) / 3 * 2
    score += trend_pts
    if trend_pts >= 1.5:
        reasons.append("Strong uptrend: EMA20 > EMA50 > EMA200")
        trend_status = "Strong uptrend"
    elif above_200:
        reasons.append("Above EMA200, partial trend alignment")
        trend_status = "Moderate uptrend"
    else:
        risks.append("Below EMA200 – trend not confirmed")
        trend_status = "Downtrend / sideways"

    # 2. Volume confirmation (2 pts)
    avg_vol = float(last["avg_volume"])
    cur_vol = float(last["volume"])
    vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 0.0
    if vol_ratio >= 1.5:
        score += 2.0
        reasons.append(f"Volume {vol_ratio:.1f}x above average – strong confirmation")
    elif vol_ratio >= 1.0:
        score += 1.0
        reasons.append(f"Volume {vol_ratio:.1f}x average – adequate")
    else:
        risks.append(f"Volume below average ({vol_ratio:.1f}x) – weak confirmation")

    # 3. Relative strength (2 pts)
    rs_spy = float(last.get("rs_spy", np.nan))
    rs_qqq = float(last.get("rs_qqq", np.nan))
    rs_pts = 0.0
    if not np.isnan(rs_spy) and rs_spy > 0:
        rs_pts += 1.0
        reasons.append("Outperforming SPY")
    if not np.isnan(rs_qqq) and rs_qqq > 0:
        rs_pts += 1.0
        reasons.append("Outperforming QQQ")
    score += rs_pts
    if rs_pts == 0:
        risks.append("Underperforming both SPY and QQQ")

    # 4. Clean entry/stop structure (2 pts)
    rps = setup.risk_per_share
    entry = setup.entry
    stop_pct = rps / entry if entry > 0 else 1.0
    if stop_pct <= 0.07:  # stop within 7%
        score += 2.0
        reasons.append(f"Clean stop structure ({stop_pct*100:.1f}% risk)")
    elif stop_pct <= 0.12:
        score += 1.0
        reasons.append(f"Acceptable stop ({stop_pct*100:.1f}% risk)")
    else:
        risks.append(f"Wide stop ({stop_pct*100:.1f}% risk) – position size limited")

    # 5. Risk/reward (2 pts)
    if setup.rr_ratio >= 3.0:
        score += 2.0
        reasons.append(f"Excellent R:R of {setup.rr_ratio:.1f}")
    elif setup.rr_ratio >= 2.0:
        score += 1.5
        reasons.append(f"Good R:R of {setup.rr_ratio:.1f}")
    else:
        score += 0.5
        risks.append(f"Marginal R:R of {setup.rr_ratio:.1f}")

    why_qualifies = "; ".join(reasons) if reasons else "Meets basic strategy criteria"
    why_may_fail = "; ".join(risks) if risks else "Gap-down through stop; broad market selloff"

    invalidation = (
        f"Close below ${setup.stop_loss:.2f} (stop loss) or "
        f"broad market (SPY) breaks key support"
    )

    return score, trend_status, why_qualifies, why_may_fail, invalidation


def scan_universe(data: dict[str, pd.DataFrame],
                  cfg: dict,
                  best_strategy: str | None = None) -> list[Candidate]:
    """
    Scan all tickers for today's signals.
    If best_strategy is given, only scan that strategy.
    Returns sorted list of Candidates (highest score first).
    """
    min_price = float(cfg.get("trading", {}).get("min_price", 5.0))
    min_avg_vol = float(cfg.get("trading", {}).get("min_avg_volume", 500000))
    min_rr = float(cfg.get("account", {}).get("min_rr_ratio", 2.0))

    strategies_to_scan = [best_strategy] if best_strategy else list(STRATEGIES.keys())
    candidates: list[Candidate] = []

    for strategy_key in strategies_to_scan:
        strat_cfg = cfg.get("strategies", {}).get(strategy_key, {})
        if not strat_cfg.get("enabled", True):
            continue

        for ticker, df in data.items():
            if len(df) < 60:
                continue

            last = df.iloc[-1]
            close = float(last["close"])
            avg_vol = float(last.get("avg_volume", 0))

            # basic filters
            if close < min_price:
                continue
            if avg_vol < min_avg_vol:
                continue

            try:
                signals = get_signals(df, strategy_key, cfg)
            except Exception as e:
                log.debug("Signal error %s: %s", ticker, e)
                continue

            # check if LAST bar has a signal
            if not bool(signals.iloc[-1]):
                continue

            setup = build_trade_setup(ticker, strategy_key, df, cfg)
            if setup is None or setup.rr_ratio < min_rr:
                continue

            score, trend, qualifies, fails, invalid = _score_candidate(
                df, strategy_key, setup)

            rs_spy = float(last.get("rs_spy", np.nan))
            rs_qqq = float(last.get("rs_qqq", np.nan))
            vol_ratio = (float(last["volume"]) / avg_vol) if avg_vol > 0 else 0.0

            company = _get_company_name(ticker)
            candidates.append(Candidate(
                ticker=ticker,
                company=company,
                strategy_key=strategy_key,
                strategy_name=STRATEGY_NAMES.get(strategy_key, strategy_key),
                current_price=round(close, 2),
                entry=setup.entry,
                stop_loss=setup.stop_loss,
                target1=setup.target1,
                target2=setup.target2,
                risk_per_share=setup.risk_per_share,
                rr_ratio=setup.rr_ratio,
                shares=setup.shares,
                position_value=setup.position_value,
                setup_score=round(score, 1),
                rs_spy=rs_spy,
                rs_qqq=rs_qqq,
                volume_ratio=vol_ratio,
                trend_status=trend,
                why_qualifies=qualifies,
                why_may_fail=fails,
                invalidation=invalid,
            ))

    candidates.sort(key=lambda c: c.setup_score, reverse=True)
    log.info("Scanner found %d candidates", len(candidates))
    return candidates


def market_regime(spy_df: pd.DataFrame | None,
                  qqq_df: pd.DataFrame | None) -> dict:
    """Classify current market regime based on SPY/QQQ trend."""
    def _classify(df):
        if df is None or len(df) < 5:
            return "Unknown", "N/A"
        last = df.iloc[-1]
        close = float(last["close"])
        ema20 = float(last.get("ema20", np.nan))
        ema50 = float(last.get("ema50", np.nan))
        ema200 = float(last.get("ema200", np.nan))
        if np.isnan(ema200):
            return "Unknown", "N/A"
        if close > ema20 > ema50 > ema200:
            return "Strong uptrend", "aggressive"
        elif close > ema50 > ema200:
            return "Uptrend", "neutral"
        elif close > ema200:
            return "Mixed", "defensive"
        else:
            return "Downtrend", "no-trade"

    spy_trend, spy_mode = _classify(spy_df)
    qqq_trend, qqq_mode = _classify(qqq_df)

    # risk mode = most conservative of the two
    mode_order = ["aggressive", "neutral", "defensive", "no-trade"]
    risk_mode = spy_mode if mode_order.index(spy_mode) >= mode_order.index(qqq_mode) else qqq_mode

    return {
        "spy_trend": spy_trend,
        "qqq_trend": qqq_trend,
        "risk_mode": risk_mode,
    }
