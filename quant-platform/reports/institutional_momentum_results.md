# institutional_momentum_trend — Backtest & Comparison (2026-08-11)

**VERDICT: `INSTITUTIONAL_MOMENTUM_PROMISING_BUT_UNPROVEN`**

Universe 107 | Period 6y | Account €30,000 | Risk 0.50% | Max 4 positions | Heat 2% | Sector 25% | Realistic costs | Config: reclaim entry, 0.25×ATR stop buffer, 3.0×ATR trail, RS≥80

| Strategy | Trades | OOS | Win% | PF | Expect € | CAGR | MaxDD | Sharpe | Sortino | WF+ | Acceptance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ema_pullback | 277 | 79 | 32% | 1.05 | 2.55 | 0.4% | -8.7% | -0.31 | -0.36 | 0% | INSUFFICIENT_SAMPLE |
| minervini_vcp | 197 | 59 | 35% | 1.17 | 6.60 | 0.7% | -8.1% | -0.33 | -0.36 | 50% | INSUFFICIENT_SAMPLE |
| volume_breakout | 186 | 48 | 39% | 1.12 | 3.61 | 0.4% | -5.7% | -0.63 | -0.73 | 50% | INSUFFICIENT_SAMPLE |
| combined | 273 | 71 | 32% | 0.91 | -4.56 | -0.7% | -14.4% | -0.62 | -0.73 | 50% | INSUFFICIENT_SAMPLE |
| enhanced_ema_pullback | 122 | 30 | 34% | 1.36 | 16.00 | 1.1% | -6.8% | -0.27 | -0.28 | 50% | INSUFFICIENT_SAMPLE |
| institutional_momentum_trend | 126 | 30 | 35% | 1.61 | 31.14 | 2.1% | -5.7% | 0.03 | 0.04 | 50% | INSUFFICIENT_SAMPLE |

## Head-to-head: best existing vs institutional

Best existing by expectancy: **minervini_vcp**

| Metric | minervini_vcp | institutional_momentum_trend | Winner |
|---|---|---|---|
| Expectancy €/trade | 6.60 | 31.14 | INST |
| Profit factor | 1.17 | 1.61 | INST |
| CAGR | 0.7% | 2.1% | INST |
| Max drawdown | -8.1% | -5.7% | INST |
| Walk-forward + | 50% | 50% | EXISTING |
| Stress expectancy € | -9.02 | 12.77 | INST |

Institutional wins 5/6 comparisons.

## Acceptance detail — institutional_momentum_trend

Status: **INSUFFICIENT_SAMPLE**  |  failed gates: oos_trades,wf,concentration

- OOS trades: 30 (need ≥200)
- Profit factor: 1.61 (need ≥1.25)
- Expectancy: €31.14/trade (need >0)
- Walk-forward positive: 50% (need ≥65%)
- Max drawdown: -5.7% (need ≥ −12%)
- Stress expectancy: €12.77 (need >0)
- Largest stock share of profit: 37% (need ≤15%)
- Largest sector share of profit: 41% (need ≤25%)

## Parameter stability (top by profit factor)

entry_mode  stop_buffer  trail  rs_min  filled  win_rate  profit_factor     cagr    max_dd
   reclaim         0.25    3.0      90      68  0.352941       2.162638 0.022322 -0.043101
   reclaim         0.25    3.5      90      64  0.359375       2.092654 0.021055 -0.043353
   reclaim         0.50    3.0      90      62  0.354839       2.068407 0.018835 -0.044907
   reclaim         0.25    3.5      85      94  0.308511       1.985796 0.025620 -0.048214
   reclaim         0.00    3.5      85     105  0.304762       1.957777 0.029533 -0.054523
   reclaim         0.25    3.5      75     116  0.362069       1.950210 0.029098 -0.055210
   reclaim         0.50    3.5      90      59  0.355932       1.944341 0.017326 -0.045524
   reclaim         0.25    3.5      80     111  0.351351       1.917727 0.027303 -0.056671
  reversal         0.50    3.0      90      59  0.322034       1.893941 0.015828 -0.040782
   reclaim         0.00    3.0      90      81  0.320988       1.839636 0.020972 -0.049944

_Research only. Production scanner unchanged._