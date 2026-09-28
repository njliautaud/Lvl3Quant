# Full Pair + Triplet Confluence Sweep Report
Compliance: HC #466 (full confluence matrix), HC #467 (stream target = realized log_ret_5s), HC #344 (day-conc cap).
Wall time: 401.6s. Pair rows: 642. Triplet rows: 534.

## PAIRS — positive net edge & day-conc pass & n>=100

110 pairs survive these gates.

| rank | head_a | head_b | top % | n | mean ticks | net | hit | Sharpe | day_conc |
|------|--------|--------|-------|---|------------|-----|-----|--------|----------|
| 1 | pred_k | pred_log_ret_60s_q90 | 5.0 | 321 | 2.299 | +1.923 | 48.9% | +0.180 | 0.36 |
| 2 | pred_log_ret_1s | pred_p_up_5s | 5.0 | 147 | 1.929 | +1.553 | 62.6% | +0.287 | 0.24 |
| 3 | pred_log_ret_60s | pred_p_up_5s | 1.0 | 212 | 1.509 | +1.133 | 49.1% | +0.137 | 0.08 |
| 4 | pred_log_ret_10s_q50 | pred_log_ret_60s | 1.0 | 274 | 1.465 | +1.089 | 53.6% | +0.139 | 0.09 |
| 5 | pred_log_ret_5s | pred_log_ret_60s | 1.0 | 196 | 1.462 | +1.086 | 51.5% | +0.126 | 0.09 |
| 6 | pred_log_ret_5s | pred_log_ret_60s | 5.0 | 212 | 1.429 | +1.053 | 51.4% | +0.124 | 0.08 |
| 7 | pred_log_ret_1s | pred_p_up_10s | 1.0 | 252 | 1.325 | +0.949 | 57.9% | +0.173 | 0.35 |
| 8 | pred_log_ret_5s | pred_p_up_5s | 1.0 | 287 | 1.321 | +0.945 | 51.9% | +0.125 | 0.07 |
| 9 | pred_log_ret_30s_q50 | pred_log_ret_5s | 1.0 | 270 | 1.306 | +0.930 | 52.6% | +0.119 | 0.08 |
| 10 | pred_log_ret_10s_q50 | pred_log_ret_60s | 5.0 | 289 | 1.294 | +0.918 | 52.6% | +0.118 | 0.09 |
| 11 | pred_log_ret_60s | pred_p_up_30s | 1.0 | 238 | 1.265 | +0.889 | 48.7% | +0.111 | 0.08 |
| 12 | pred_log_ret_60s | pred_p_up_10s | 1.0 | 319 | 1.262 | +0.886 | 50.8% | +0.120 | 0.08 |
| 13 | pred_log_ret_5s | pred_p_up_5s | 5.0 | 491 | 1.246 | +0.870 | 54.2% | +0.127 | 0.11 |
| 14 | pred_log_ret_30s_q50 | pred_log_ret_5s | 5.0 | 363 | 1.198 | +0.822 | 51.5% | +0.114 | 0.10 |
| 15 | pred_log_ret_10s_q50 | pred_p_up_5s | 1.0 | 364 | 1.180 | +0.804 | 51.1% | +0.113 | 0.08 |
| 16 | pred_log_ret_10s_q50 | pred_log_ret_60s | 10.0 | 336 | 1.171 | +0.795 | 50.9% | +0.108 | 0.08 |
| 17 | pred_log_ret_5s | pred_log_ret_60s | 10.0 | 242 | 1.145 | +0.769 | 49.6% | +0.095 | 0.07 |
| 18 | pred_log_ret_10s_q50 | pred_p_up_5s | 5.0 | 597 | 1.138 | +0.762 | 54.1% | +0.115 | 0.11 |
| 19 | pred_log_ret_60s | pred_p_up_10s | 5.0 | 334 | 1.121 | +0.745 | 50.6% | +0.101 | 0.07 |
| 20 | pred_log_ret_60s | pred_p_up_10s | 10.0 | 339 | 1.114 | +0.738 | 50.7% | +0.101 | 0.07 |

## PAIRS — top-10 overall by net ticks (incl. failing gates)

| rank | head_a | head_b | top % | n | mean ticks | net | hit | day_conc | gate |
|------|--------|--------|-------|---|------------|-----|-----|----------|------|
| 1 | pred_log_ret_10s_q50 | pred_log_ret_60s_q10 | 5.0 | 67 | 2.799 | +2.423 | 44.8% | 0.22 | fail |
| 2 | pred_k | pred_log_ret_60s_q90 | 5.0 | 321 | 2.299 | +1.923 | 48.9% | 0.36 | PASS |
| 3 | pred_log_ret_1s | pred_p_up_5s | 5.0 | 147 | 1.929 | +1.553 | 62.6% | 0.24 | PASS |
| 4 | pred_log_ret_30s_q90 | pred_log_ret_5s | 10.0 | 53 | 1.679 | +1.303 | 50.9% | 0.13 | fail |
| 5 | pred_log_ret_1s | pred_log_ret_30s_q90 | 10.0 | 52 | 1.663 | +1.287 | 51.9% | 0.13 | fail |
| 6 | pred_fifo_tp8sl5_net | pred_log_ret_5s | 1.0 | 82 | 1.537 | +1.161 | 56.1% | 0.12 | fail |
| 7 | pred_log_ret_60s | pred_p_up_5s | 1.0 | 212 | 1.509 | +1.133 | 49.1% | 0.08 | PASS |
| 8 | pred_p_up_5s | pred_p_up_60s | 5.0 | 91 | 1.473 | +1.097 | 57.1% | 0.12 | fail |
| 9 | pred_log_ret_10s_q50 | pred_log_ret_60s | 1.0 | 274 | 1.465 | +1.089 | 53.6% | 0.09 | PASS |
| 10 | pred_log_ret_5s | pred_log_ret_60s | 1.0 | 196 | 1.462 | +1.086 | 51.5% | 0.09 | PASS |

## TRIPLETS — positive net edge & day-conc pass & n>=50

286 triplets survive these gates.

| rank | a | b | c | top % | n | mean | net | hit | Sharpe | day_conc |
|------|---|---|---|-------|---|------|-----|-----|--------|----------|
| 1 | pred_log_ret_5s | pred_log_ret_10s_q50 | pred_log_ret_60s_q10 | 5.0 | 53 | 3.453 | +3.077 | 45.3% | +0.224 | 0.25 |
| 2 | pred_log_ret_10s_q50 | pred_log_ret_30s_q50 | pred_log_ret_60s_q10 | 5.0 | 63 | 2.992 | +2.616 | 46.0% | +0.206 | 0.24 |
| 3 | pred_p_up_5s | pred_log_ret_10s_q50 | pred_log_ret_60s_q10 | 5.0 | 62 | 2.895 | +2.519 | 45.2% | +0.197 | 0.24 |
| 4 | pred_log_ret_10s_q50 | pred_p_up_30s | pred_log_ret_60s_q10 | 5.0 | 64 | 2.883 | +2.507 | 45.3% | +0.198 | 0.23 |
| 5 | pred_log_ret_10s_q50 | pred_p_up_10s | pred_log_ret_60s_q10 | 5.0 | 67 | 2.799 | +2.423 | 44.8% | +0.196 | 0.22 |
| 6 | pred_log_ret_60s | pred_log_ret_10s_q50 | pred_log_ret_60s_q10 | 5.0 | 54 | 2.722 | +2.346 | 38.9% | +0.172 | 0.24 |
| 7 | pred_log_ret_60s | pred_log_ret_5s | pred_log_ret_60s_q10 | 5.0 | 63 | 2.500 | +2.124 | 41.3% | +0.167 | 0.24 |
| 8 | pred_log_ret_5s | pred_p_up_5s | pred_log_ret_60s_q10 | 5.0 | 88 | 2.108 | +1.732 | 46.6% | +0.158 | 0.23 |
| 9 | pred_log_ret_60s | pred_log_ret_10s_q50 | pred_log_ret_60s_q10 | 10.0 | 97 | 2.000 | +1.624 | 42.3% | +0.154 | 0.19 |
| 10 | pred_p_up_5s | pred_log_ret_10s_q50 | pred_log_ret_1s | 5.0 | 143 | 1.976 | +1.600 | 62.9% | +0.293 | 0.25 |

## TRIPLETS — top-10 overall by net ticks (incl. failing gates)

| rank | a | b | c | top % | n | mean | net | hit | day_conc | gate |
|------|---|---|---|-------|---|------|-----|-----|----------|------|
| 1 | pred_log_ret_5s | pred_log_ret_10s_q50 | pred_log_ret_60s_q10 | 5.0 | 53 | 3.453 | +3.077 | 45.3% | 0.25 | PASS |
| 2 | pred_log_ret_10s_q50 | pred_log_ret_30s_q50 | pred_log_ret_60s_q10 | 5.0 | 63 | 2.992 | +2.616 | 46.0% | 0.24 | PASS |
| 3 | pred_p_up_5s | pred_log_ret_10s_q50 | pred_log_ret_60s_q10 | 5.0 | 62 | 2.895 | +2.519 | 45.2% | 0.24 | PASS |
| 4 | pred_log_ret_10s_q50 | pred_p_up_30s | pred_log_ret_60s_q10 | 5.0 | 64 | 2.883 | +2.507 | 45.3% | 0.23 | PASS |
| 5 | pred_log_ret_10s_q50 | pred_p_up_10s | pred_log_ret_60s_q10 | 5.0 | 67 | 2.799 | +2.423 | 44.8% | 0.22 | PASS |
| 6 | pred_log_ret_60s | pred_log_ret_10s_q50 | pred_log_ret_60s_q10 | 5.0 | 54 | 2.722 | +2.346 | 38.9% | 0.24 | PASS |
| 7 | pred_log_ret_60s | pred_log_ret_5s | pred_log_ret_60s_q10 | 5.0 | 63 | 2.500 | +2.124 | 41.3% | 0.24 | PASS |
| 8 | pred_log_ret_5s | pred_p_up_5s | pred_log_ret_60s_q10 | 5.0 | 88 | 2.108 | +1.732 | 46.6% | 0.23 | PASS |
| 9 | pred_log_ret_5s | pred_log_ret_30s_q50 | pred_fifo_tp8sl5_net | 1.0 | 31 | 2.097 | +1.721 | 58.1% | 0.13 | fail |
| 10 | pred_log_ret_60s | pred_log_ret_10s_q50 | pred_log_ret_60s_q10 | 10.0 | 97 | 2.000 | +1.624 | 42.3% | 0.19 | PASS |
