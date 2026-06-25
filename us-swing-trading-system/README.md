# US Swing Trading Analysis System

A local Python system for research-grade US stock swing-trading analysis.
It backtests five strategies, ranks them by robustness, scans today's market,
and produces a ranked watchlist with entry, stop-loss, target, and position-size
recommendations.

> **RISK DISCLAIMER**
> This tool is for **educational and research purposes only**.
> It does **not** place real trades, does **not** manage real money, and does
> **not** constitute financial advice. Past backtest results are no guarantee of
> future performance. Trading involves significant risk of loss. Always consult a
> qualified financial professional before making investment decisions.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Adding Tickers](#adding-tickers)
6. [CSV Mode](#csv-mode)
7. [API Mode](#api-mode)
8. [Running Backtests](#running-backtests)
9. [Running the Daily Scanner](#running-the-daily-scanner)
10. [Generating Reports](#generating-reports)
11. [Running Everything at Once](#running-everything-at-once)
12. [Interpreting the Output](#interpreting-the-output)
13. [Strategy Descriptions](#strategy-descriptions)
14. [Running Tests](#running-tests)
15. [Adding a New Data Provider](#adding-a-new-data-provider)

---

## Requirements

- Python 3.11+
- Internet connection (for API mode)

## Installation

```bash
# 1. Clone or download the project
cd us-swing-trading-system

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy .env.example to .env (no keys required for yfinance)
cp .env.example .env
```

---

## Project Structure

```
us-swing-trading-system/
├── config.yaml          ← All tunable parameters
├── .env.example         ← Environment variable template
├── requirements.txt
├── README.md
├── data/
│   ├── raw/             ← Drop your OHLCV CSV files here (CSV mode)
│   ├── processed/       ← Auto-populated parquet cache
│   └── tickers.csv      ← Custom ticker list (one per line)
├── src/
│   ├── main.py          ← CLI entry point
│   ├── data_loader.py   ← Data acquisition layer
│   ├── indicators.py    ← Technical indicators
│   ├── strategies.py    ← Signal generation for 5 strategies
│   ├── backtester.py    ← Event-driven backtester
│   ├── scanner.py       ← Daily market scanner
│   ├── risk.py          ← Position sizing and stop-loss logic
│   ├── reporting.py     ← Report and chart generation
│   └── utils.py         ← Config, logging, helpers
├── reports/             ← All outputs go here (auto-created)
└── tests/               ← Unit tests
```

---

## Configuration

All parameters are in `config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `account.size` | 10000 | Account size in USD |
| `account.risk_per_trade` | 0.01 | Risk per trade (1%) |
| `account.max_positions` | 5 | Maximum concurrent positions |
| `account.min_rr_ratio` | 2.0 | Minimum risk/reward ratio |
| `trading.min_price` | 5.0 | Minimum stock price (avoid penny stocks) |
| `trading.min_avg_volume` | 500000 | Minimum 20-day average volume |
| `trading.max_holding_days` | 30 | Max holding period (trading days) |
| `backtest.slippage` | 0.001 | Slippage per trade (0.1%) |
| `backtest.commission` | 5.0 | Round-trip commission in USD |
| `data.mode` | api | `"api"` or `"csv"` |
| `data.api_provider` | yfinance | Currently: `yfinance` |
| `data.default_period` | 2y | History to download |

---

## Adding Tickers

**Custom list:** Edit `data/tickers.csv`. One ticker per line, no header:

```
AAPL
MSFT
NVDA
```

**Universe:** In `config.yaml`, set `universe.sp500: true` and/or
`universe.nasdaq100: true` to include those universes automatically.

---

## CSV Mode

Place OHLCV CSV files in `data/raw/` named `<TICKER>.csv`.

Required columns (case-insensitive):

| Column | Description |
|--------|-------------|
| `date` | Trade date (parseable date format) |
| `open` | Open price |
| `high` | High price |
| `low` | Low price |
| `close` | Close price |
| `volume` | Volume |
| `adjusted_close` | *(Optional)* Adjusted close – preferred if present |

In `config.yaml`, set:

```yaml
data:
  mode: csv
  csv_dir: data/raw
```

---

## API Mode

Uses `yfinance` by default – **no API key required**.

```yaml
data:
  mode: api
  api_provider: yfinance
  default_period: 2y
```

Downloaded data is cached as Parquet files in `data/processed/` to avoid
repeated network calls. Delete the cache files to force a re-download.

---

## Running Backtests

```bash
python src/main.py backtest
```

To test quickly on a small universe:

```bash
python src/main.py backtest --limit 20
```

Outputs:
- `reports/backtest_results.csv` – per-strategy metrics
- `reports/strategy_ranking.csv` – ranked strategies by composite score
- `reports/equity_curve.png` – equity curves for all strategies
- `reports/strategy_comparison.md` – detailed comparison

---

## Running the Daily Scanner

```bash
python src/main.py scan
```

Scans the full universe using today's data (or most recent available) and
outputs:
- `reports/daily_watchlist.csv` – all candidates
- `reports/top_candidates.md` – formatted report with buy candidates

---

## Generating Reports

Re-generates reports from cached data without re-running backtests:

```bash
python src/main.py report
```

---

## Running Everything at Once

```bash
python src/main.py all
```

Runs backtest → scan → full report in sequence.

---

## Interpreting the Output

### `reports/top_candidates.md`

```
## Market Regime
SPY and QQQ trend classification:
  Strong uptrend → "aggressive" risk mode
  Uptrend       → "neutral"
  Mixed          → "defensive"
  Downtrend      → "no-trade"

## Today's Top Buy Candidates   (setup score ≥ 7/10)
## Watchlist Only               (score 5–6.9)
## Avoid                        (score < 5)
## Final Decision
  "High-quality candidates available"  → top candidates exist
  "Watchlist only, no entry yet"       → nothing ripe yet
  "No high-quality swing trade today." → market weak or no setups
```

### Setup Score (out of 10)

| Component | Points |
|-----------|--------|
| Trend alignment (EMA stack) | 2 |
| Volume confirmation | 2 |
| Relative strength vs SPY + QQQ | 2 |
| Clean entry/stop structure | 2 |
| Risk/reward ≥ 2:1 | 2 |

### Position Sizing

Position size is calculated so that if the stop loss is hit, you lose exactly
1% of your account. Example:

- Account: $10,000
- Risk per trade: 1% = $100
- Entry: $50, Stop: $48 → Risk per share: $2
- Shares: 100 / 2 = **50 shares** at $50 = $2,500 position

---

## Strategy Descriptions

| Strategy | Core Idea |
|----------|-----------|
| **EMA Trend Pullback** | Buy dips to EMA20/50 in a strong uptrend |
| **Breakout** | Buy breaks above resistance on high volume |
| **Momentum Continuation** | Buy strong stocks near 52-week highs |
| **Mean Reversion in Uptrend** | Buy RSI recoveries near EMA50 |
| **Volatility Contraction Breakout** | Buy tight consolidations (VCP pattern) |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover:
- Indicator calculations (EMA, RSI, ATR)
- Position sizing math
- Strategy signal generation and no look-ahead bias
- Backtester trade closing logic and metrics

---

## Adding a New Data Provider

1. Open `src/data_loader.py`
2. Add a new function `_load_<provider>(ticker, ...)` that returns a DataFrame
   with columns `open, high, low, close, volume` and a DatetimeIndex named `date`
3. Add a branch in `load_ticker_data()` under `# api mode`
4. Set `data.api_provider: <provider>` in `config.yaml`
5. Add any required API keys to `.env.example` and `.env`

---

## Notes

- The system never places real trades.
- All signals are verified to be free of look-ahead bias.
- Backtest results use adjusted close prices where available.
- Slippage and commission are applied to every trade.
- In-sample vs out-of-sample split is configurable in `config.yaml`.
