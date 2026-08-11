# enhanced_ema_pullback — Results (2026-08-11)

Acceptance: **INSUFFICIENT_SAMPLE**  (failed gates: oos_trades,wf_windows,concentration)

Main config: entry=reclaim, stop=swing_low−0.25×ATR, trail=3.5×ATR, no profit target

Signals 801 | Filled 100 | Win 37% | PF 2.09 | Expectancy €52.16/trade | CAGR 2.7% | MaxDD -6.0% | Sharpe 0.19 | Sortino 0.21

Costs €: {'commission': 179.8706003044363, 'spread_slip': 299.78470767030547, 'fx': 302.65963589977093, 'gap_loss': 242.37442936469773}

## Parameter stability (top rows by profit factor)

entry_mode  stop_buffer  trail  filled  win_rate  profit_factor     cagr    max_dd
   reclaim         0.50    3.5      91  0.373626       2.170969 0.026967 -0.056861
   reclaim         0.25    3.5     100  0.370000       2.088231 0.027207 -0.060414
   reclaim         0.00    3.5     108  0.333333       1.869954 0.026559 -0.058687
   reclaim         0.50    3.0     101  0.376238       1.736259 0.020145 -0.060200
  reversal         0.50    3.5     116  0.318966       1.712186 0.022197 -0.056080
   reclaim         0.25    3.0     111  0.351351       1.698235 0.020491 -0.065114
  reversal         0.50    3.0     127  0.322835       1.621249 0.021175 -0.061995
   reclaim         0.00    3.0     121  0.330579       1.553217 0.019325 -0.064937

## Rejection reasons (realistic run)

reason
RS_LOW                    411
REGIME_BEAR               133
MAX_POSITIONS              82
EARNINGS_SOON              46
STOP_TOO_WIDE               8
SECTOR_CAP                  6
OPEN_GAP_ABOVE_TRIGGER      5
EARNINGS_UNKNOWN            1

Stock-level stats: samples below 25 trades are INSUFFICIENT_SAMPLE
by policy; strategy-level OOS results drive the decision.