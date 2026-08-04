# enhanced_ema_pullback — Results (2026-08-04)

Acceptance: **INSUFFICIENT_SAMPLE**  (failed gates: oos_trades,pf,wf_windows,stress,concentration)

Main config: entry=reversal, stop=swing_low−0.25×ATR, trail=2.5×ATR, no profit target

Signals 1816 | Filled 169 | Win 27% | PF 1.02 | Expectancy €0.84/trade | CAGR 0.1% | MaxDD -8.5% | Sharpe -0.48 | Sortino -0.52

Costs €: {'commission': 303.2825713634845, 'spread_slip': 508.2018331284131, 'fx': 508.72099215764274, 'gap_loss': 507.92453159649386}

## Parameter stability (top rows by profit factor)

entry_mode  stop_buffer  trail  filled  win_rate  profit_factor     cagr    max_dd
   reclaim         0.50    3.5      93  0.344086       2.080715 0.025794 -0.056951
   reclaim         0.25    3.5     102  0.343137       1.984317 0.025750 -0.060526
   reclaim         0.00    3.5     109  0.311927       1.797295 0.025274 -0.058718
   reclaim         0.50    3.0     103  0.349515       1.674862 0.019072 -0.060106
  reversal         0.50    3.5     117  0.299145       1.655041 0.020954 -0.056128
   reclaim         0.25    3.0     113  0.327434       1.627040 0.019169 -0.065446
  reversal         0.50    3.0     128  0.304688       1.577677 0.020101 -0.062114
   reclaim         0.00    3.0     122  0.311475       1.473454 0.017141 -0.065217

## Rejection reasons (realistic run)

reason
RS_LOW                    950
REGIME_BEAR               293
MAX_POSITIONS             211
EARNINGS_SOON              84
SECTOR_CAP                 20
STOP_TOO_WIDE              16
OPEN_GAP_ABOVE_TRIGGER      8

Stock-level stats: samples below 25 trades are INSUFFICIENT_SAMPLE
by policy; strategy-level OOS results drive the decision.