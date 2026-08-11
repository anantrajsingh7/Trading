# institutional_momentum_trend — Backtest & Comparison (2026-08-11)

**VERDICT: `INSTITUTIONAL_MOMENTUM_PROMISING_BUT_UNPROVEN`**

Universe 107 | Period 6y | Account €30,000 | Risk 0.50% | Max 4 positions | Heat 2% | Sector 25% | Realistic costs | Config: reclaim entry, 0.25×ATR stop buffer, 3.0×ATR trail, RS≥80

| Strategy | Trades | OOS | Win% | PF | Expect € | CAGR | MaxDD | Sharpe | Sortino | WF+ | Acceptance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ema_pullback | 277 | 79 | 32% | 1.05 | 2.55 | 0.4% | -8.7% | -0.31 | -0.36 | 0% | INSUFFICIENT_SAMPLE |
| minervini_vcp | 196 | 59 | 35% | 1.18 | 6.84 | 0.7% | -7.9% | -0.33 | -0.36 | 50% | INSUFFICIENT_SAMPLE |
| volume_breakout | 186 | 48 | 39% | 1.12 | 3.61 | 0.4% | -5.7% | -0.63 | -0.73 | 50% | INSUFFICIENT_SAMPLE |
| combined | 269 | 71 | 32% | 0.93 | -3.18 | -0.5% | -13.3% | -0.56 | -0.66 | 50% | INSUFFICIENT_SAMPLE |
| enhanced_ema_pullback | 122 | 30 | 34% | 1.36 | 16.00 | 1.1% | -6.8% | -0.27 | -0.28 | 50% | INSUFFICIENT_SAMPLE |
| institutional_momentum_trend | 110 | 27 | 35% | 1.69 | 34.76 | 2.0% | -6.4% | 0.02 | 0.02 | 50% | INSUFFICIENT_SAMPLE |

## Head-to-head: best existing vs institutional

Best existing by expectancy: **minervini_vcp**

| Metric | minervini_vcp | institutional_momentum_trend | Winner |
|---|---|---|---|
| Expectancy €/trade | 6.84 | 34.76 | INST |
| Profit factor | 1.18 | 1.69 | INST |
| CAGR | 0.7% | 2.0% | INST |
| Max drawdown | -7.9% | -6.4% | INST |
| Walk-forward + | 50% | 50% | EXISTING |
| Stress expectancy € | -8.82 | 23.68 | INST |

Institutional wins 5/6 comparisons.

## Acceptance detail — institutional_momentum_trend

Status: **INSUFFICIENT_SAMPLE**  |  failed gates: oos_trades,wf,concentration

- OOS trades: 27 (need ≥200)
- Profit factor: 1.69 (need ≥1.25)
- Expectancy: €34.76/trade (need >0)
- Walk-forward positive: 50% (need ≥65%)
- Max drawdown: -6.4% (need ≥ −12%)
- Stress expectancy: €23.68 (need >0)
- Largest stock share of profit: 34% (need ≤15%)
- Largest sector share of profit: 50% (need ≤25%)

## Parameter stability (top by profit factor)

entry_mode  stop_buffer  trail  rs_min  filled  win_rate  profit_factor     cagr    max_dd
   reclaim         0.50    3.0      90      51  0.411765       2.462190 0.020053 -0.037352
   reclaim         0.25    3.0      90      57  0.403509       2.386362 0.022237 -0.047487
   reclaim         0.25    3.5      90      56  0.392857       2.217656 0.020349 -0.045378
   reclaim         0.50    3.5      90      50  0.400000       2.210312 0.017986 -0.036544
   reclaim         0.50    3.5      75      98  0.357143       2.200989 0.030632 -0.057788
   reclaim         0.50    3.5      80      90  0.377778       2.188278 0.027150 -0.055879
   reclaim         0.00    3.0      90      65  0.369231       2.173750 0.022561 -0.047579
   reclaim         0.25    3.5      75     107  0.364486       2.170969 0.031708 -0.059150
   reclaim         0.25    3.5      80      99  0.373737       2.088424 0.027036 -0.059424
   reclaim         0.00    3.5      90      63  0.365079       2.051695 0.020616 -0.042204

_Research only. Production scanner unchanged._