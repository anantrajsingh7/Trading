@echo off
REM Daily dashboard launcher for Windows Task Scheduler.
REM Edit the PROJECT path below if you cloned somewhere else.

set PROJECT=C:\Users\priya\Trading\quant-platform

cd /d "%PROJECT%"
call venv\Scripts\activate.bat

REM US market dashboard
python daily_dashboard.py

REM India market dashboard (uncomment the next line if you want it too)
REM python daily_dashboard.py india
