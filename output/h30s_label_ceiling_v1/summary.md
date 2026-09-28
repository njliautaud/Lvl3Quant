# h30s label-ceiling pre-flight (HC #428 R2)
OOT window: 20260224 -> 20260429, 56 days
Commission cost: 0.376 ticks. Market cost: 1.376 ticks.

## Cross-day means of |label| percentiles (ticks)

| horizon | p50 | p75 | p90 | p95 | p99 | %|lab|>0.376 | %|lab|>1.376 |
|---|---|---|---|---|---|---|---|
| 1s | 0.972 | 1.774 | 2.953 | 3.972 | 8.075 | 0.792 | 0.361 |
| 5s | 1.868 | 3.453 | 5.774 | 7.981 | 15.453 | 0.891 | 0.599 |
| 10s | 2.481 | 4.708 | 7.962 | 10.868 | 19.868 | 0.919 | 0.690 |
| 30s | 4.170 | 7.849 | 13.170 | 17.519 | 30.613 | 0.950 | 0.802 |

## Verdict
- h=30s p90 |label| = 13.170 ticks across-day mean
- VERDICT: Above market cost. h=30s viable with market orders if signal selects top-decile moves.
