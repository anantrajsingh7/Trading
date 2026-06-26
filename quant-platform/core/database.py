"""
Database layer — SQLAlchemy + SQLite.

Stores:
  - price_cache: OHLCV data (avoid re-downloading)
  - ticker_meta: Company info
  - backtest_runs: Historical backtest results
  - trades: Portfolio trade log
  - signals: Scanner output history
  - research_journal: Weekly research notes

Design: SQLite for local development. Change DATABASE_URL env var
to postgresql://... to move to production PostgreSQL instantly.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, Integer,
    String, Text, create_engine, event, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.config import get_db_path


class Base(DeclarativeBase):
    pass


# ============================================================
# ORM Models
# ============================================================

class PriceCache(Base):
    """Cached OHLCV data — avoids hitting APIs on every run."""
    __tablename__ = "price_cache"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    ticker      = Column(String(20), nullable=False, index=True)
    date        = Column(Date, nullable=False, index=True)
    open        = Column(Float)
    high        = Column(Float)
    low         = Column(Float)
    close       = Column(Float)
    adj_close   = Column(Float)
    volume      = Column(Float)
    provider    = Column(String(30), default="yfinance")
    created_at  = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        # Unique constraint prevents duplicate rows
        {"sqlite_autoincrement": True},
    )


class TickerMeta(Base):
    __tablename__ = "ticker_meta"
    ticker        = Column(String(20), primary_key=True)
    name          = Column(String(200))
    sector        = Column(String(100))
    industry      = Column(String(100))
    market        = Column(String(10))
    market_cap    = Column(Float)
    country       = Column(String(10))
    updated_at    = Column(DateTime, default=datetime.utcnow)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id            = Column(String(36), primary_key=True)
    strategy      = Column(String(100), index=True)
    period        = Column(String(10))
    run_date      = Column(DateTime, default=datetime.utcnow)
    start_date    = Column(Date)
    end_date      = Column(Date)
    n_tickers     = Column(Integer)
    n_trades      = Column(Integer)
    cagr          = Column(Float)
    sharpe        = Column(Float)
    sortino       = Column(Float)
    calmar        = Column(Float)
    max_drawdown  = Column(Float)
    win_rate      = Column(Float)
    profit_factor = Column(Float)
    expectancy    = Column(Float)
    robustness    = Column(Float)
    metrics_json  = Column(Text)       # Full metrics as JSON


class TradeRecord(Base):
    __tablename__ = "trades"
    id              = Column(String(36), primary_key=True)
    ticker          = Column(String(20), index=True)
    company         = Column(String(200))
    strategy        = Column(String(100))
    market          = Column(String(10))
    direction       = Column(String(10))
    entry_date      = Column(Date)
    entry_price     = Column(Float)
    shares          = Column(Integer)
    stop_loss       = Column(Float)
    target_1        = Column(Float)
    target_2        = Column(Float)
    target_3        = Column(Float)
    exit_date       = Column(Date)
    exit_price      = Column(Float)
    exit_reason     = Column(String(30))
    status          = Column(String(20), default="OPEN", index=True)
    pnl             = Column(Float)
    pnl_pct         = Column(Float)
    holding_days    = Column(Integer)
    score_at_entry  = Column(Float)
    notes           = Column(Text)
    created_at      = Column(DateTime, default=datetime.utcnow)


class SignalRecord(Base):
    __tablename__ = "signals"
    id             = Column(String(36), primary_key=True)
    scan_date      = Column(Date, index=True)
    ticker         = Column(String(20))
    company        = Column(String(200))
    strategy       = Column(String(100))
    score          = Column(Float)
    entry_price    = Column(Float)
    stop_loss      = Column(Float)
    target_1       = Column(Float)
    rr_ratio       = Column(Float)
    rs_rank        = Column(Float)
    details_json   = Column(Text)


class ResearchJournal(Base):
    __tablename__ = "research_journal"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    week_ending = Column(Date, index=True)
    title       = Column(String(200))
    content     = Column(Text)
    strategies_reviewed = Column(Text)
    changes_made        = Column(Text)
    created_at  = Column(DateTime, default=datetime.utcnow)


# ============================================================
# Engine & Session Factory
# ============================================================

_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        db_path = get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
        _engine = create_engine(url, echo=False)
        # Enable WAL mode for better concurrent read performance
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(conn, _):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
        Base.metadata.create_all(_engine)
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory()


def save_signal(opp, session: Session | None = None) -> None:
    """Persist a scanner Opportunity to the signals table."""
    from core.models import Opportunity
    close_session = session is None
    if session is None:
        session = get_session()
    try:
        rec = SignalRecord(
            id=opp.id,
            scan_date=opp.date,
            ticker=opp.ticker,
            company=opp.company,
            strategy=opp.strategy,
            score=opp.score,
            entry_price=opp.ideal_entry,
            stop_loss=opp.stop_loss,
            target_1=opp.target_1,
            rr_ratio=opp.rr_ratio_t1,
            rs_rank=opp.rs_rank,
            details_json=json.dumps(opp.to_dict(), default=str),
        )
        session.merge(rec)
        session.commit()
    finally:
        if close_session:
            session.close()
