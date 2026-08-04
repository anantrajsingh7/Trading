# Strategy Comparison — 2026-08-04

**VERDICT: `EXISTING_EMA_PULLBACK_BETTER`**

Universe: 107 liquid US stocks | Period: 6y | Account €30,000 | Risk 0.50% | Max 4 positions | Heat 2% | Realistic costs | Earnings-unknown rejected (0 tickers unknown)

| Strategy | Trades | OOS | Win% | PF | Expect €/tr | CAGR | MaxDD | Sharpe | WF+ | Acceptance |
|---|---|---|---|---|---|---|---|---|---|---|
| ema_pullback | 273 | 69 | 32% | 1.16 | 8.68 | 1.3% | -8.5% | -0.12 | 50% | INSUFFICIENT_SAMPLE |
| minervini_vcp | 194 | 62 | 35% | 1.20 | 7.70 | 0.8% | -8.0% | -0.30 | 0% | INSUFFICIENT_SAMPLE |
| volume_breakout | 191 | 48 | 38% | 1.02 | 0.57 | 0.1% | -5.7% | -0.78 | 50% | INSUFFICIENT_SAMPLE |
| combined | 278 | 62 | 32% | 1.00 | 0.01 | 0.0% | -15.5% | -0.42 | 50% | INSUFFICIENT_SAMPLE |
| enhanced_ema_pullback | 169 | 40 | 27% | 1.02 | 0.84 | 0.1% | -8.5% | -0.48 | 50% | INSUFFICIENT_SAMPLE |

## Existing ema_pullback at production 1% risk (separate — not risk-comparable)
Trades 269 | PF 1.09 | CAGR 0.9% | MaxDD -14.6%

## Decision basis
Winner selected on out-of-sample expectancy, walk-forward consistency,
profit factor, drawdown, and stress-cost survival — not CAGR alone.
See CSVs for cost profiles, walk-forward windows, regimes, and the
parameter-stability grid.