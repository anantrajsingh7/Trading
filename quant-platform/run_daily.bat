@echo off
REM Daily swing dashboard launcher for Windows Task Scheduler (10:00 AM).
REM Edit the PROJECT path below if you cloned somewhere else.

set PROJECT=C:\Users\priya\Trading\quant-platform

cd /d "%PROJECT%"
call venv\Scripts\activate.bat

REM Pull latest strategy/code updates first (safe: fast-forward only)
git pull --ff-only origin claude/swing-trading-analysis-system-e439iw

REM Validated swing dashboard (combined strategies + per-stock backtest)
python swing_dashboard.py

REM India edition (uncomment if wanted)
REM python swing_dashboard.py india
