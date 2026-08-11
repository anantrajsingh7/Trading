# Strategy Comparison — 2026-08-11

**VERDICT: `ENHANCED_EMA_PULLBACK_BETTER`**

Universe: 107 liquid US stocks | Period: 6y | Account €30,000 | Risk 0.50% | Max 4 positions | Heat 2% | Realistic costs | Earnings-unknown rejected (1 tickers unknown)

| Strategy | Trades | OOS | Win% | PF | Expect €/tr | CAGR | MaxDD | Sharpe | WF+ | Acceptance |
|---|---|---|---|---|---|---|---|---|---|---|
| ema_pullback | 193 | 59 | 27% | 1.11 | 6.49 | 0.7% | -8.2% | -0.26 | 100% | INSUFFICIENT_SAMPLE |
| minervini_vcp | 138 | 44 | 38% | 1.31 | 14.47 | 1.1% | -6.7% | -0.21 | 50% | INSUFFICIENT_SAMPLE |
| volume_breakout | 125 | 33 | 38% | 1.14 | 5.46 | 0.4% | -5.4% | -0.63 | 100% | INSUFFICIENT_SAMPLE |
| combined | 190 | 50 | 30% | 1.07 | 3.64 | 0.4% | -8.7% | -0.35 | 50% | INSUFFICIENT_SAMPLE |
| enhanced_ema_pullback | 100 | 24 | 37% | 2.09 | 52.16 | 2.7% | -6.0% | 0.19 | 50% | INSUFFICIENT_SAMPLE |

## Existing ema_pullback at production 1% risk (separate — not risk-comparable)
Trades 180 | PF 1.03 | CAGR 0.3% | MaxDD -13.5%

## Practical constraints (1% risk, 5 positions, 5% heat — realistic costs)
| Strategy | Trades | Win% | PF | Expect €/tr | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|---|---|---|
| ema_pullback | 231 | 29% | 1.18 | 14.13 | 1.8% | -12.9% | 0.00 |
| enhanced_ema_pullback | 106 | 35% | 1.51 | 38.85 | 2.2% | -9.0% | 0.06 |

## Decision basis
Winner selected on out-of-sample expectancy, walk-forward consistency,
profit factor, drawdown, and stress-cost survival — not CAGR alone.
See CSVs for cost profiles, walk-forward windows, regimes, and the
parameter-stability grid.