# enhanced_ema_pullback — Results (2026-08-04)

Acceptance: **INSUFFICIENT_SAMPLE**  (failed gates: oos_trades,wf_windows,concentration)

Main config: entry=reclaim, stop=swing_low−0.25×ATR, trail=3.5×ATR, no profit target

Signals 801 | Filled 102 | Win 35% | PF 1.99 | Expectancy €48.26/trade | CAGR 2.6% | MaxDD -6.1% | Sharpe 0.15 | Sortino 0.17

Costs €: {'commission': 183.2654402153724, 'spread_slip': 306.1824514928021, 'fx': 308.91599749023175, 'gap_loss': 240.8761528276685}

## Parameter stability (top rows by profit factor)

entry_mode  stop_buffer  trail  filled  win_rate  profit_factor     cagr    max_dd
   reclaim         0.50    3.5      93  0.354839       2.069221 0.025669 -0.056951
   reclaim         0.25    3.5     102  0.352941       1.986304 0.025780 -0.060526
   reclaim         0.00    3.5     109  0.321101       1.800147 0.025326 -0.058718
   reclaim         0.50    3.0     103  0.359223       1.668109 0.018950 -0.060106
  reversal         0.50    3.5     118  0.305085       1.647838 0.020698 -0.056128
   reclaim         0.25    3.0     113  0.336283       1.629891 0.019214 -0.065446
  reversal         0.50    3.0     129  0.310078       1.571625 0.019846 -0.062114
   reclaim         0.00    3.0     122  0.319672       1.475440 0.017175 -0.065217

## Rejection reasons (realistic run)

reason
RS_LOW                    410
REGIME_BEAR               133
MAX_POSITIONS              81
EARNINGS_SOON              47
STOP_TOO_WIDE               8
SECTOR_CAP                  6
OPEN_GAP_ABOVE_TRIGGER      5

Stock-level stats: samples below 25 trades are INSUFFICIENT_SAMPLE
by policy; strategy-level OOS results drive the decision.