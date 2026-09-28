# h5s low-LR multifold — auto-graded aggregate

Headline cell: 1s_top10% (best fold-0 surviving). Auto-graded as folds land. Updated by `h5s_multifold_auto_grader.sh` (15-min cron).

Note: **Canonical** = queue-aware FIFOReplayEngine, passive limit, ES 0.376 commission, TP 2 / SL 1. **Gated** = q≥0.50 fill-prob gate (from fill_prob_head_v1.lgb). Label-FIFO column is the screening proxy from `h5s_lowlr_fifo_sweep.py` (signed_pred × label − commission). Canonical is the production-relevant metric.

| Fold | OOT Date | Label-FIFO net/trade | Label N | Canonical net/trade | Canon N | Gated (q>=0.5) net/trade | Gated N |
|------|----------|---------------------|---------|---------------------|---------|--------------------------|--------|
| 00 | 20260412 | +3.890 | 94 | -0.296 | 25 | +0.047 | 13 |
| 01 | 20260413 | +0.225 | 2265 | -0.599 | 1127 | -0.614 | 585 |
| 02 | 20260414 | +0.261 | 2102 | -0.666 | 886 | -0.698 | 459 |
| 03 | 20260415 | +0.347 | 2459 | -0.679 | 1029 | -0.658 | 729 |
| 03 | 20260415 | +0.347 | 2459 | -0.679 | 1029 | -0.658 | 729 |
| 04 | 20260416 | +0.368 | 2519 | -0.637 | 1182 | -0.635 | 958 |
| 05 | 20260417 | +0.333 | 2790 | -0.600 | 1470 | -0.574 | 877 |
| 06 | 20260419 | +0.468 | 77 | -0.481 | 19 | -0.435 | 17 |
| 07 | 20260420 | +0.361 | 2570 | -0.571 | 1159 | -0.517 | 600 |
| 08 | 20260426 | -0.108 | 41 | -0.076 | 5 | -0.501 | 4 |
| 09 | 20260427 | +0.292 | 2368 | -0.681 | 903 | -0.700 | 719 |
