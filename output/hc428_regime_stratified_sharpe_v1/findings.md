# HC #428 R1 — Regime-stratified Sharpe verdict

Source: 34-day baseline per-day stratification (48 cells).
Gates: regime_gap ≤ 0.50, mean_net_ticks > 0, both regime means ≥ 0.

**Cells passing all gates: 1**

## Passing cells
| horizon   | side   | bucket   |   n_days |   mean_net_ticks |   sharpe |   sharpe_green |   sharpe_red |   sharpe_flat |   mean_green |   mean_red |   n_green |   n_red |   n_flat |   regime_gap |
|:----------|:-------|:---------|---------:|-----------------:|---------:|---------------:|-------------:|--------------:|-------------:|-----------:|----------:|--------:|---------:|-------------:|
| 5s        | short  | top_1pct |       29 |         0.983254 |  3.07184 |        2.27722 |      4.49494 |     -0.863153 |     0.328947 |    2.23687 |        13 |      11 |        5 |     0.493382 |