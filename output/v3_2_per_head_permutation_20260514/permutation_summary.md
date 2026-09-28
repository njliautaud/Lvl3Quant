# v3.2 Per-Head Permutation Test — Results

_2026-05-14T05:35:15Z_

Tested top 120 candidates from `per_head_master.json` with 1000 sign-shuffles each.

## SURVIVORS (p < 0.05 AND observed_mean > 0.376 commission)

**13 survivors.** These ARE statistically distinguishable from random:

| Head | Side | Band | n_fills | obs_t | passive_net | null_mean | null_p95 | p |
|---|---|---|---|---|---|---|---|---|
| log_ret_60s | SHORT | Top0.5% | 76 | 1.119 | 0.743 | 0.189 | 0.423 | 0.000 |
| log_ret_60s | SHORT | Top1% | 137 | 0.803 | 0.427 | 0.118 | 0.297 | 0.000 |
| log_ret_60s_q10 | SHORT | Top10% | 171 | 0.587 | 0.211 | 0.453 | 0.514 | 0.000 |
| p_reversal_15s | SHORT | Top20% | 8143 | 0.604 | 0.228 | 0.402 | 0.445 | 0.000 |
| log_ret_60s | SHORT | Top5% | 531 | 0.511 | 0.135 | 0.287 | 0.396 | 0.000 |
| p_reversal_15s | SHORT | Top10% | 4431 | 0.667 | 0.291 | 0.429 | 0.490 | 0.000 |
| p_reversal_15s | SHORT | Top5% | 2173 | 0.647 | 0.271 | 0.299 | 0.391 | 0.000 |
| p_reversal_15s | SHORT | Top1% | 562 | 0.601 | 0.225 | 0.291 | 0.499 | 0.006 |
| log_ret_1s | SHORT | Top20% | 5964 | 0.539 | 0.163 | 0.461 | 0.513 | 0.007 |
| log_ret_60s | SHORT | Top10% | 998 | 0.539 | 0.163 | 0.433 | 0.512 | 0.015 |
| log_ret_1s | SHORT | Top0.1% | 35 | 1.390 | 1.014 | 0.747 | 1.296 | 0.028 |
| log_ret_1s | SHORT | Top10% | 3166 | 0.561 | 0.185 | 0.485 | 0.552 | 0.029 |
| log_ret_60s | SHORT | Top20% | 1989 | 0.547 | 0.171 | 0.485 | 0.542 | 0.030 |

## ALL TESTED (sorted by p-value)

| Head | Side | Band | n | obs | null_mean | null_p95 | p | verdict |
|---|---|---|---|---|---|---|---|---|
| log_ret_60s | SHORT | Top0.5% | 76 | 1.119 | 0.189 | 0.423 | 0.000 | SIGNIF |
| log_ret_60s | SHORT | Top1% | 137 | 0.803 | 0.118 | 0.297 | 0.000 | SIGNIF |
| log_ret_60s_q10 | SHORT | Top10% | 171 | 0.587 | 0.453 | 0.514 | 0.000 | SIGNIF |
| p_reversal_15s | SHORT | Top20% | 8143 | 0.604 | 0.402 | 0.445 | 0.000 | SIGNIF |
| log_ret_60s | SHORT | Top5% | 531 | 0.511 | 0.287 | 0.396 | 0.000 | SIGNIF |
| log_ret_1s | SHORT | Top20% | 5964 | 0.539 | 0.461 | 0.513 | 0.007 | SIGNIF |
| log_ret_60s | SHORT | Top10% | 998 | 0.539 | 0.433 | 0.512 | 0.015 | SIGNIF |
| log_ret_1s | SHORT | Top0.1% | 35 | 1.390 | 0.747 | 1.296 | 0.028 | SIGNIF |
| log_ret_1s | SHORT | Top10% | 3166 | 0.561 | 0.485 | 0.552 | 0.029 | SIGNIF |
| log_ret_60s | SHORT | Top20% | 1989 | 0.547 | 0.485 | 0.542 | 0.030 | SIGNIF |
| log_ret_1s | SHORT | Top0.5% | 172 | 0.626 | 0.366 | 0.629 | 0.053 | no |
| p_reversal_15s | SHORT | Top0.5% | 300 | 0.789 | 0.568 | 0.840 | 0.096 | no |
| log_ret_60s_q10 | SHORT | Top5% | 80 | 0.563 | 0.496 | 0.584 | 0.096 | no |
| log_ret_10s | SHORT | Top20% | 6247 | 0.560 | 0.525 | 0.576 | 0.115 | no |
| log_ret_1s | SHORT | Top1% | 356 | 0.540 | 0.423 | 0.603 | 0.149 | no |
| log_ret_60s | LONG | Top0.1% | 151 | 0.611 | 0.336 | 0.791 | 0.182 | no |
| log_ret_5s | SHORT | Top10% | 3110 | 0.527 | 0.497 | 0.563 | 0.243 | no |
| log_ret_1s | SHORT | Top5% | 1610 | 0.534 | 0.497 | 0.586 | 0.255 | no |
| p_reversal_60s | SHORT | Top0.1% | 65 | 0.758 | 0.588 | 1.018 | 0.287 | no |
| log_ret_30s | SHORT | Top5% | 1617 | 0.588 | 0.563 | 0.660 | 0.352 | no |
| p_up_60s | SHORT | Top1% | 801 | 0.540 | 0.504 | 0.678 | 0.369 | no |
| p_reversal_60s | SHORT | Top0.5% | 328 | 0.512 | 0.469 | 0.682 | 0.374 | no |
| log_ret_30s_q50 | SHORT | Top0.5% | 116 | 0.885 | 0.824 | 1.139 | 0.392 | no |
| log_ret_60s_q50 | SHORT | Top20% | 9429 | 0.506 | 0.496 | 0.549 | 0.400 | no |
| log_ret_10s_q50 | SHORT | Top20% | 6089 | 0.531 | 0.524 | 0.576 | 0.419 | no |
| log_ret_30s_q10 | SHORT | Top0.5% | 412 | 0.776 | 0.743 | 1.021 | 0.425 | no |
| fifo_tp4sl3_net | SHORT | Top20% | 15285 | 0.519 | 0.515 | 0.559 | 0.431 | no |
| p_reversal_60s | SHORT | Top20% | 8409 | 0.501 | 0.498 | 0.541 | 0.451 | no |
| p_up_30s | SHORT | Top1% | 565 | 0.559 | 0.545 | 0.759 | 0.466 | no |
| log_ret_30s_q50 | SHORT | Top20% | 5840 | 0.580 | 0.578 | 0.627 | 0.467 | no |
| p_up_10s | SHORT | Top10% | 5337 | 0.555 | 0.553 | 0.622 | 0.474 | no |
| p_up_5s | SHORT | Top5% | 2861 | 0.510 | 0.507 | 0.595 | 0.478 | no |
| log_ret_10s_q10 | SHORT | Top0.1% | 100 | 1.556 | 1.546 | 2.063 | 0.479 | no |
| log_ret_10s_q10 | SHORT | Top1% | 625 | 0.654 | 0.651 | 0.881 | 0.479 | no |
| log_ret_10s_q10 | SHORT | Top0.5% | 398 | 0.860 | 0.865 | 1.152 | 0.499 | no |
| log_ret_5s | SHORT | Top20% | 6214 | 0.536 | 0.538 | 0.587 | 0.512 | no |
| p_up_30s | SHORT | Top5% | 2574 | 0.527 | 0.530 | 0.623 | 0.516 | no |
| log_ret_30s_q10 | SHORT | Top0.1% | 109 | 1.578 | 1.616 | 2.116 | 0.522 | no |
| p_reversal_15s | SHORT | Top0.1% | 49 | 0.816 | 1.092 | 1.710 | 0.752 | no |
| p_reversal_30s | SHORT | Top0.1% | 66 | nan | nan | nan | nan | no |
| p_reversal_30s | SHORT | Top0.5% | 290 | nan | nan | nan | nan | no |
| log_ret_30s_q50 | SHORT | Top5% | 1493 | 0.590 | 0.591 | 0.691 | 0.505 | no |
| p_up_5s | SHORT | Top20% | 10746 | 0.569 | 0.570 | 0.620 | 0.505 | no |
| p_up_30s | SHORT | Top20% | 10317 | 0.591 | 0.593 | 0.641 | 0.513 | no |
| p_up_30s | SHORT | Top10% | 5035 | 0.586 | 0.590 | 0.658 | 0.521 | no |
| p_up_5s | SHORT | Top10% | 5394 | 0.529 | 0.531 | 0.598 | 0.521 | no |
| log_ret_30s | SHORT | Top0.5% | 122 | 0.692 | 0.704 | 1.034 | 0.525 | no |
| log_ret_10s_q50 | SHORT | Top10% | 3023 | 0.513 | 0.516 | 0.586 | 0.525 | no |
| p_up_10s | SHORT | Top20% | 10551 | 0.570 | 0.571 | 0.619 | 0.531 | no |
| log_ret_60s_q50 | SHORT | Top10% | 5197 | 0.543 | 0.547 | 0.618 | 0.544 | no |
| log_ret_60s_q50 | SHORT | Top5% | 1948 | 0.574 | 0.582 | 0.690 | 0.552 | no |
| log_ret_30s_q50 | SHORT | Top1% | 254 | 0.665 | 0.733 | 0.963 | 0.688 | no |
| log_ret_30s | SHORT | Top20% | 6297 | 0.565 | 0.582 | 0.631 | 0.712 | no |
| log_ret_30s | SHORT | Top10% | 3229 | 0.570 | 0.595 | 0.664 | 0.719 | no |
| log_ret_30s_q50 | SHORT | Top10% | 2993 | 0.563 | 0.596 | 0.665 | 0.771 | no |
| log_ret_30s | SHORT | Top1% | 279 | 0.507 | 0.620 | 0.844 | 0.797 | no |
| log_ret_10s_q50 | SHORT | Top5% | 1552 | 0.509 | 0.563 | 0.661 | 0.813 | no |
| log_ret_5s | SHORT | Top0.1% | 35 | 0.833 | 1.155 | 1.718 | 0.828 | no |
| log_ret_5s | SHORT | Top5% | 1628 | 0.534 | 0.588 | 0.686 | 0.834 | no |
| p_reversal_30s | SHORT | Top1% | 550 | nan | nan | nan | nan | no |
| p_reversal_15s | SHORT | Top10% | 4431 | 0.667 | 0.429 | 0.490 | 0.000 | SIGNIF |
| p_reversal_15s | SHORT | Top5% | 2173 | 0.647 | 0.299 | 0.391 | 0.000 | SIGNIF |
| p_reversal_15s | LONG | Top10% | 3473 | -0.257 | -0.432 | -0.371 | 0.000 | no |
| log_ret_1s | LONG | Top0.5% | 237 | 0.044 | -0.373 | -0.114 | 0.002 | no |
| p_reversal_15s | LONG | Top5% | 1650 | -0.162 | -0.299 | -0.208 | 0.004 | no |
| log_ret_1s | LONG | Top1% | 433 | -0.123 | -0.428 | -0.243 | 0.005 | no |
| p_reversal_15s | SHORT | Top1% | 562 | 0.601 | 0.291 | 0.499 | 0.006 | SIGNIF |
| p_reversal_60s | SHORT | Top5% | 2740 | 0.463 | 0.429 | 0.513 | 0.237 | no |
| p_up_60s | SHORT | Top0.1% | 86 | 0.440 | 0.303 | 0.828 | 0.321 | no |
| log_ret_60s | LONG | Top0.5% | 540 | -0.135 | -0.201 | 0.032 | 0.326 | no |
| log_ret_60s | LONG | Top1% | 891 | -0.099 | -0.136 | 0.051 | 0.361 | no |
| p_reversal_30s | LONG | Top10% | 3573 | -0.221 | -0.223 | -0.155 | 0.456 | no |
| p_up_60s | SHORT | Top5% | 3268 | 0.431 | 0.426 | 0.523 | 0.459 | no |
| log_ret_10s_q10 | SHORT | Top20% | 9692 | 0.475 | 0.475 | 0.527 | 0.470 | no |
| p_reversal_15s | LONG | Top0.5% | 147 | -0.571 | -0.582 | -0.317 | 0.473 | no |
| p_up_5s | SHORT | Top0.1% | 74 | 0.498 | 0.476 | 0.974 | 0.476 | no |
| p_up_30s | SHORT | Top0.1% | 39 | 0.414 | 0.391 | 1.176 | 0.479 | no |
| log_ret_10s_q90 | LONG | Top0.5% | 509 | -0.238 | -0.250 | 0.000 | 0.479 | no |
| p_reversal_30s | LONG | Top1% | 301 | -0.324 | -0.330 | -0.128 | 0.481 | no |
| log_ret_10s_q10 | SHORT | Top5% | 2547 | 0.426 | 0.425 | 0.528 | 0.482 | no |
| p_reversal_30s | LONG | Top5% | 1748 | -0.058 | -0.060 | 0.041 | 0.491 | no |
| p_up_30s | SHORT | Top0.5% | 267 | 0.466 | 0.467 | 0.756 | 0.496 | no |
| log_ret_30s_q10 | SHORT | Top10% | 4684 | 0.443 | 0.443 | 0.522 | 0.496 | no |
| p_up_10s | SHORT | Top0.1% | 73 | 0.397 | 0.389 | 0.930 | 0.497 | no |
| log_ret_30s_q10 | SHORT | Top20% | 8901 | 0.458 | 0.460 | 0.515 | 0.498 | no |
| p_up_5s | SHORT | Top0.5% | 353 | 0.477 | 0.477 | 0.727 | 0.505 | no |
| p_reversal_60s | SHORT | Top1% | 670 | 0.425 | 0.424 | 0.585 | 0.506 | no |
| log_ret_60s_q50 | SHORT | Top0.5% | 240 | -0.068 | -0.057 | 0.347 | 0.506 | no |
| log_ret_30s_q10 | SHORT | Top5% | 2547 | 0.437 | 0.437 | 0.537 | 0.507 | no |
| p_up_10s | SHORT | Top5% | 2790 | 0.502 | 0.503 | 0.592 | 0.509 | no |
| fifo_tp8sl5_net | SHORT | Top20% | 16889 | 0.500 | 0.501 | 0.545 | 0.509 | no |
| log_ret_10s_q10 | SHORT | Top10% | 4807 | 0.426 | 0.426 | 0.501 | 0.510 | no |
| p_up_10s | SHORT | Top0.5% | 314 | 0.386 | 0.387 | 0.646 | 0.511 | no |
| fifo_tp8sl5_net | SHORT | Top10% | 9697 | 0.491 | 0.493 | 0.552 | 0.512 | no |
| fifo_tp8sl5_net | SHORT | Top1% | 970 | -0.175 | -0.171 | 0.010 | 0.516 | no |
| log_ret_60s_q50 | SHORT | Top1% | 461 | 0.325 | 0.330 | 0.598 | 0.520 | no |
| fifo_tp8sl5_net | SHORT | Top5% | 5150 | 0.381 | 0.385 | 0.460 | 0.528 | no |
| log_ret_30s_q10 | SHORT | Top1% | 684 | 0.490 | 0.500 | 0.724 | 0.535 | no |
| p_up_10s | SHORT | Top1% | 597 | 0.487 | 0.503 | 0.707 | 0.544 | no |
| p_up_60s | SHORT | Top20% | 11567 | 0.452 | 0.455 | 0.500 | 0.546 | no |
| p_up_60s | SHORT | Top10% | 6126 | 0.426 | 0.432 | 0.500 | 0.548 | no |
| p_up_5s | SHORT | Top1% | 633 | 0.477 | 0.494 | 0.682 | 0.576 | no |
| log_ret_10s | SHORT | Top10% | 3222 | 0.504 | 0.520 | 0.587 | 0.639 | no |
| log_ret_10s_q50 | SHORT | Top0.5% | 129 | 0.477 | 0.557 | 0.880 | 0.639 | no |
| log_ret_10s | SHORT | Top0.5% | 165 | 0.391 | 0.471 | 0.779 | 0.644 | no |
| p_up_60s | SHORT | Top0.5% | 432 | 0.333 | 0.419 | 0.672 | 0.707 | no |
| fifo_tp4sl3_net | SHORT | Top10% | 8935 | 0.480 | 0.500 | 0.554 | 0.719 | no |
| fifo_tp4sl3_net | SHORT | Top5% | 4741 | 0.360 | 0.389 | 0.466 | 0.742 | no |
| p_reversal_60s | SHORT | Top10% | 4896 | 0.428 | 0.472 | 0.529 | 0.893 | no |
| p_reversal_30s | SHORT | Top20% | 7750 | nan | nan | nan | nan | no |
| fifo_tp4sl3_net | SHORT | Top1% | 1061 | -0.109 | -0.052 | 0.107 | 0.734 | no |
| log_ret_10s | SHORT | Top5% | 1617 | 0.479 | 0.522 | 0.617 | 0.775 | no |
| log_ret_5s | SHORT | Top0.5% | 166 | 0.340 | 0.470 | 0.756 | 0.776 | no |
| log_ret_10s_q50 | SHORT | Top1% | 285 | 0.450 | 0.574 | 0.804 | 0.809 | no |
| log_ret_10s | SHORT | Top1% | 305 | 0.343 | 0.460 | 0.674 | 0.812 | no |
| log_ret_60s_q10 | SHORT | Top20% | 357 | 0.440 | 0.490 | 0.536 | 0.957 | no |
| p_reversal_30s | SHORT | Top10% | 3897 | nan | nan | nan | nan | no |
| log_ret_5s | SHORT | Top1% | 330 | 0.314 | 0.544 | 0.739 | 0.966 | no |
| p_reversal_30s | SHORT | Top5% | 1871 | nan | nan | nan | nan | no |
| p_reversal_15s | LONG | Top1% | 310 | -0.516 | -0.293 | -0.085 | 0.971 | no |

---
Total compute: 711.5s for 120x1000 permutations
