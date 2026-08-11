# dual_momentum_rotation — Backtest (2026-08-11)

Universe: 16 liquid ETFs | Period 10y | Account €30,000 | Monthly rebalance | Top 3 equal weight | Absolute-momentum filter vs BIL

## Headline (realistic costs)

| Metric | Rotation | SPY buy & hold |
|---|---|---|
| CAGR | 10.54% | 15.35% |
| Max drawdown | -31.4% | -33.7% |
| Sharpe | 0.53 | 0.77 |
| Sortino | 0.62 | 0.93 |
| Positive months | 57% | 71% |
| Final equity | €81442 | €124516 |

**Turnover: 19.2 rebalance trades per year** — the entire point. At ~€18/round-trip friction, a strategy trading ~12 times a year loses ~€216/yr to costs; the swing book at ~100 trades/yr loses ~€1,800.

## Cost sensitivity

     costs     cagr    max_dd   sharpe  trades_per_year
optimistic 0.116001 -0.313821 0.587971             19.2
 realistic 0.105377 -0.313821 0.534970             19.2
    stress 0.083665 -0.313821 0.424716             19.2

## Holdout (last 2 years unseen)

In-sample CAGR 9.41% | MaxDD -31.4%
Holdout  CAGR 12.26% | MaxDD -17.4%

## Parameter stability

 top_n       lookbacks     cagr    max_dd   sharpe  trades_per_year
     2  (63, 126, 252) 0.098860 -0.295357 0.475284             16.1
     2 (126, 252, 252) 0.137343 -0.295357 0.643100             11.8
     2   (21, 63, 126) 0.082164 -0.297292 0.399178             22.7
     3  (63, 126, 252) 0.105377 -0.313821 0.534970             19.2
     3 (126, 252, 252) 0.129780 -0.295617 0.656006             16.4
     3   (21, 63, 126) 0.094341 -0.258136 0.506344             28.1
     4  (63, 126, 252) 0.096057 -0.311381 0.505526             21.2
     4 (126, 252, 252) 0.115107 -0.311381 0.605245             20.2
     4   (21, 63, 126) 0.096196 -0.217198 0.550790             32.4
     5  (63, 126, 252) 0.091091 -0.312853 0.492037             23.4
     5 (126, 252, 252) 0.102015 -0.312853 0.550215             20.9
     5   (21, 63, 126) 0.103257 -0.189503 0.630719             36.4

_Research only. Production scanner unchanged._