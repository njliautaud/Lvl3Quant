# v3.3 σ-WEIGHTED HEAD RANKING — HC #363 deliverable 2

Source: `fold_00_sigma.json` (32 learned σ params from fold-0 epoch=4 batch=22000)
Joined with: `per_head_master.csv` (per-head best (side, band) cell by Sharpe, n_fills ≥ 30)

## Alignment

- Spearman ρ(confidence rank, Sharpe rank) over 24 heads with valid Sharpe = **-0.063**
- Weak alignment — σ-confidence and FIFO Sharpe are largely independent. σ may reflect TRAINING residual variance, not deployment edge.

## Top 15 by σ-confidence (lowest σ first — model's 'most trustworthy' heads)

| Rank | Head | σ | Best (side, band, n) | FIFO Sharpe | Net t/fill | Passive net | Day-conc |
|---|---|---|---|---|---|---|---|
| 1 | log_ret_5min | 0.0497 | (LONG, Top5%, n=2291) | -0.164 | -0.467 | -0.843 | 44% |
| 2 | log_ret_60s | 0.0497 | (SHORT, Top10%, n=48) | 0.006 | 0.022 | -0.354 | 39% |
| 3 | log_ret_60s_q10 | 0.0497 | (SHORT, Top1%, n=589) | 0.222 | 0.668 | 0.292 | 95% |
| 4 | log_ret_60s_q50 | 0.0497 | (SHORT, Top1%, n=48) | 0.366 | 1.188 | 0.812 | 92% |
| 5 | log_ret_60s_q90 | 0.0497 | (LONG, Top0.1%, n=30) | 0.174 | 0.557 | 0.181 | 31% |
| 6 | p_reversal_60s | 0.0497 | (SHORT, Top0.1%, n=35) | 0.507 | 1.347 | 0.971 | 44% |
| 7 | p_up_60s | 0.0497 | (SHORT, Top10%, n=6879) | 0.134 | 0.474 | 0.098 | 33% |
| 8 | pred_mae_60s_ticks | 0.0497 | no cells ≥30 fills | n/a | n/a | n/a | n/a |
| 9 | pred_mfe_60s_ticks | 0.0497 | no cells ≥30 fills | n/a | n/a | n/a | n/a |
| 10 | p_reversal_30s | 0.1660 | (SHORT, Top20%, n=6893) | 0.149 | 0.530 | 0.154 | 39% |
| 11 | p_reversal_15s | 0.1807 | (SHORT, Top20%, n=7163) | 0.150 | 0.532 | 0.156 | 38% |
| 12 | pred_realized_vol_30s_ticks | 0.1874 | no cells ≥30 fills | n/a | n/a | n/a | n/a |
| 13 | fifo_tp8sl5_hit_tp | 0.2038 | no cells ≥30 fills | n/a | n/a | n/a | n/a |
| 14 | fifo_tp4sl3_hit_tp | 0.2968 | no cells ≥30 fills | n/a | n/a | n/a | n/a |
| 15 | pred_mae_30s_ticks | 0.4576 | no cells ≥30 fills | n/a | n/a | n/a | n/a |

## Top 15 by FIFO Sharpe (best operational cell)

| Rank | Head | σ | Best (side, band, n) | FIFO Sharpe | Joint score (Sharpe/σ) |
|---|---|---|---|---|---|
| 1 | p_reversal_60s | 0.0497 | (SHORT, Top0.1%, n=35) | 0.507 | 10.204 |
| 2 | fifo_tp8sl5_net | 0.8983 | (SHORT, Top0.1%, n=114) | 0.455 | 0.507 |
| 3 | log_ret_10s_q50 | 1.3690 | (SHORT, Top0.1%, n=30) | 0.390 | 0.285 |
| 4 | log_ret_60s_q50 | 0.0497 | (SHORT, Top1%, n=48) | 0.366 | 7.376 |
| 5 | log_ret_30s_q50 | 1.7247 | (SHORT, Top0.5%, n=72) | 0.355 | 0.206 |
| 6 | fifo_tp4sl3_net | 0.7219 | (SHORT, Top0.1%, n=68) | 0.261 | 0.361 |
| 7 | log_ret_60s_q10 | 0.0497 | (SHORT, Top1%, n=589) | 0.222 | 4.464 |
| 8 | log_ret_30s | 7.4853 | (SHORT, Top1%, n=163) | 0.208 | 0.028 |
| 9 | p_up_30s | 0.8318 | (SHORT, Top10%, n=5116) | 0.190 | 0.229 |
| 10 | log_ret_60s_q90 | 0.0497 | (LONG, Top0.1%, n=30) | 0.174 | 3.503 |
| 11 | p_up_10s | 0.8278 | (SHORT, Top20%, n=10470) | 0.171 | 0.206 |
| 12 | p_up_5s | 0.8234 | (SHORT, Top10%, n=5098) | 0.169 | 0.205 |
| 13 | log_ret_10s | 4.6799 | (SHORT, Top20%, n=4291) | 0.160 | 0.034 |
| 14 | log_ret_5s | 3.4853 | (SHORT, Top10%, n=2343) | 0.157 | 0.045 |
| 15 | p_reversal_15s | 0.1807 | (SHORT, Top20%, n=7163) | 0.150 | 0.829 |

## Top 15 by JOINT score (Sharpe / σ — combined confidence × performance)

| Rank | Head | σ | Best (side, band, n) | FIFO Sharpe | Joint score |
|---|---|---|---|---|---|
| 1 | p_reversal_60s | 0.0497 | (SHORT, Top0.1%, n=35) | 0.507 | 10.204 |
| 2 | log_ret_60s_q50 | 0.0497 | (SHORT, Top1%, n=48) | 0.366 | 7.376 |
| 3 | log_ret_60s_q10 | 0.0497 | (SHORT, Top1%, n=589) | 0.222 | 4.464 |
| 4 | log_ret_60s_q90 | 0.0497 | (LONG, Top0.1%, n=30) | 0.174 | 3.503 |
| 5 | p_up_60s | 0.0497 | (SHORT, Top10%, n=6879) | 0.134 | 2.705 |
| 6 | p_reversal_30s | 0.1660 | (SHORT, Top20%, n=6893) | 0.149 | 0.895 |
| 7 | p_reversal_15s | 0.1807 | (SHORT, Top20%, n=7163) | 0.150 | 0.829 |
| 8 | fifo_tp8sl5_net | 0.8983 | (SHORT, Top0.1%, n=114) | 0.455 | 0.507 |
| 9 | fifo_tp4sl3_net | 0.7219 | (SHORT, Top0.1%, n=68) | 0.261 | 0.361 |
| 10 | log_ret_10s_q50 | 1.3690 | (SHORT, Top0.1%, n=30) | 0.390 | 0.285 |
| 11 | p_up_30s | 0.8318 | (SHORT, Top10%, n=5116) | 0.190 | 0.229 |
| 12 | p_up_10s | 0.8278 | (SHORT, Top20%, n=10470) | 0.171 | 0.206 |
| 13 | log_ret_30s_q50 | 1.7247 | (SHORT, Top0.5%, n=72) | 0.355 | 0.206 |
| 14 | p_up_5s | 0.8234 | (SHORT, Top10%, n=5098) | 0.169 | 0.205 |
| 15 | log_ret_10s_q10 | 0.9122 | (SHORT, Top0.1%, n=118) | 0.146 | 0.160 |

