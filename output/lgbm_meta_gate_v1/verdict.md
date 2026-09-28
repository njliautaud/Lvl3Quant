# LightGBM Meta-Gate v1 -- Verdict (HC #486 R4 prototype)

Run finished 2026-05-22 09:11:11.
Folds trained: **17** (sliding 15-day train, 1-day OOT, slide by 1)
Target: `target_fifo_tp4sl3_net` (1 if FIFO-net > 0)
Gate threshold: P(profitable) >= **0.3**

## Summary table

| Mode              | Days | Trades  | Net ticks | $        | Mean per-day Sh | Median Sh | Mean PF | WR    |
|-------------------|------|---------|-----------|----------|-----------------|-----------|---------|-------|
| Gated (LGBM)      |   14 |     124 |     -63.8 |     -798 |          -0.777 |     0.000 |   1.279 | 15.3% |
| All signals       |   17 | 739,779 | -123170.3 | -1539629 |         -17.313 |   -17.668 |   0.736 | 13.4% |
| Direction baseline|   17 | 397,185 |  -67947.6 |  -849344 |         -12.921 |   -13.695 |   0.733 | 13.3% |

## Top-10 meta-gate features (gain-summed across folds)

- `pred_p_up_5s` -- gain 823596
- `pred_log_ret_60s_q90` -- gain 543023
- `pred_log_ret_60s_q10` -- gain 508852
- `pred_p_up_10s` -- gain 409212
- `pred_log_ret_5s` -- gain 401815
- `pred_log_ret_10s` -- gain 387726
- `pred_fifo_tp4sl3_hit_tp` -- gain 376780
- `pred_log_ret_60s` -- gain 329230
- `pred_log_ret_30s` -- gain 328362
- `pred_pred_mfe_60s_ticks` -- gain 323726

## Verdict

**REJECT** -- Gated mode does not beat the take-all baseline.

Next steps: regime-stratified breakdown (HC #428 R1) and threshold sweep.
