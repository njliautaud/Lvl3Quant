# Queue-Position Model v0 — HC #471 R5 / HC #469 R5(a)
Generated: 2026-05-22 21:41 ET. Wall: 6.9s. Device: cuda.

**Compliance**: HC #471 R5 (queue-position model prototype), HC #469 R5(a) (queue-aware fill gating).

## Task
Predict `queue_ahead` (realized FIFO queue position at fill time) from a pre-signal
window of K=200 book events × 30 features (5-level book + flow metrics).

## Headline

| Metric | Value |
|---|---|
| n_test | 2762 |
| MAE (log queue) | 0.6417 |
| Baseline MAE (predict-mean) | 0.5530 |
| **Lift vs mean baseline** | **-16.0%** |
| MAE (raw queue count) | 9.28 |
| P90 abs error (raw) | 18.36 |
| R² (log space) | -0.2090 |
| Mean bias (raw) | -4.62 |

## Per-day breakdown (test)

| Date | n | MAE_log | MAE_raw | y_mean | pred_mean |
|---|---|---|---|---|---|
| 20260309 | 1842 | 0.636 | 8.23 | 14.9 | 9.3 |
| 20260310 | 340 | 0.672 | 11.15 | 19.4 | 15.4 |
| 20260311 | 81 | 0.586 | 6.40 | 10.9 | 15.6 |
| 20260312 | 389 | 0.663 | 12.83 | 18.5 | 16.7 |
| 20260313 | 110 | 0.610 | 10.57 | 19.4 | 12.5 |

## Next steps
- Wire as a confluence head: trade only when predicted_queue_ahead × P(joiner_drain) < threshold.
- Extend window to K=500 or K=1000 events; compare lift.
- Add joiner/leaver volume features (HC #469 R5(b)) and re-train.
- Move to per-event sequence model (Transformer / Mamba) for sharper temporal pickup.