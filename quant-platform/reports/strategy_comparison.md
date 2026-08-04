# Strategy Comparison — 2026-08-04

**VERDICT: `ENHANCED_EMA_PULLBACK_BETTER`**

Universe: 107 liquid US stocks | Period: 6y | Account €30,000 | Risk 0.50% | Max 4 positions | Heat 2% | Realistic costs | Earnings-unknown rejected (0 tickers unknown)

| Strategy | Trades | OOS | Win% | PF | Expect €/tr | CAGR | MaxDD | Sharpe | WF+ | Acceptance |
|---|---|---|---|---|---|---|---|---|---|---|
| ema_pullback | 186 | 50 | 28% | 1.28 | 16.39 | 1.6% | -8.1% | -0.06 | 50% | INSUFFICIENT_SAMPLE |
| minervini_vcp | 136 | 44 | 39% | 1.39 | 17.77 | 1.3% | -6.7% | -0.15 | 50% | INSUFFICIENT_SAMPLE |
| volume_breakout | 126 | 35 | 36% | 1.09 | 3.49 | 0.2% | -5.3% | -0.68 | 100% | INSUFFICIENT_SAMPLE |
| combined | 182 | 47 | 32% | 1.30 | 17.51 | 1.7% | -8.6% | -0.04 | 50% | INSUFFICIENT_SAMPLE |
| enhanced_ema_pullback | 102 | 25 | 35% | 1.99 | 48.26 | 2.6% | -6.1% | 0.15 | 50% | INSUFFICIENT_SAMPLE |

## Existing ema_pullback at production 1% risk (separate — not risk-comparable)
Trades 183 | PF 1.26 | CAGR 2.2% | MaxDD -10.9%

## Practical constraints (1% risk, 5 positions, 5% heat — realistic costs)
| Strategy | Trades | Win% | PF | Expect €/tr | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|---|---|---|
| ema_pullback | 231 | 29% | 1.18 | 14.42 | 1.8% | -13.6% | 0.01 |
| enhanced_ema_pullback | 109 | 33% | 1.44 | 33.84 | 2.0% | -9.0% | 0.02 |

## Decision basis
Winner selected on out-of-sample expectancy, walk-forward consistency,
profit factor, drawdown, and stress-cost survival — not CAGR alone.
See CSVs for cost profiles, walk-forward windows, regimes, and the
parameter-stability grid.