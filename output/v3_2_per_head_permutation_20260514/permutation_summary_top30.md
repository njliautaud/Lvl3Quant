# v3.2 Per-Head Permutation Test — SURVIVORS FOUND

_Generated 2026-05-14 ~01:25 ET — by `v32_per_head_permutation_test.py`_

## TL;DR — 4 of 28 valid candidates SURVIVE the pass-8 permutation test

The all-night research (8 passes) concluded "no edge" because it focused on `pred_log_ret_5s`. **The signal is actually concentrated at the 60-second horizon, SHORT side**. Pass-8 methodology applied to the per-head dashboard's top-30 candidates surfaces the following genuine survivors:

| Head | Side | Band | n_fills | obs t/fill | null_mean | null_p95 | p-value | passive_net |
|---|---|---|---|---|---|---|---|---|
| **log_ret_60s** | SHORT | Top0.5% | 76 | **+1.119** | 0.189 | 0.423 | **0.0000** | **+0.743** |
| **log_ret_60s** | SHORT | Top1% | 137 | +0.803 | 0.118 | 0.297 | **0.0000** | +0.427 |
| **log_ret_60s_q10** | SHORT | Top10% | 171 | +0.587 | 0.453 | 0.514 | **0.0000** | +0.211 |
| **log_ret_1s** | SHORT | Top0.1% | 35 | +1.390 | 0.747 | 1.296 | 0.0280 | +1.014 |

**Why this matters**: Pass 8 of the all-night research showed that even Top5% RAW (no filter, pred_5s) gets Sharpe ≈ 3.5 from FIFO+selection structure alone (null_mean 3.66 of 3.85 observed → p=0.39). That methodology applied to log_ret_60s SHORT Top0.5% gives **null_mean=0.189 vs observed=1.119 → p=0.0000**. The model's signal IS distinguishable from random sign-shuffles at this band/head/side combination.

## Caveats (must be addressed before live deployment)

1. **5-day OOT only**: the day-concentration column from the dashboard showed log_ret_60s SHORT Top0.5% has 56% of its fills on a single day. The signal might be 1-day robust but not 5-day stable. **HC #337 extended-OOT (10+ days) is the next critical test.**
2. **76 fills is small**: bootstrap CI on 76 events is reasonable but Sharpe estimates are noisy. Need ~200+ for confident deployment sizing.
3. **Same 5-day data**: the dashboard CV bands and the permutation null both share the same 5 OOT days. Extended-OOT would test the temporal generalization.
4. **No queue model yet**: fill rate proxy from `target_fifo_tp4sl3_net != 0` (24.5%). Real L2 queue position would refine this.

## Recommended next steps

1. **Run permutation test on TOP 60+ candidates** (currently top-30 only). Some of the 22 untested in the 30-60 ranking might also survive.
2. **Confluence**: when log_ret_60s SHORT Top0.5% AND log_ret_1s SHORT Top0.1% AGREE, what's the joint metric?
3. **ToD slicing**: do these survivors concentrate in a specific ToD bucket like the prior 11:18-12:00 finding?
4. **HC #337 extended-OOT**: rerun v3.2 inference on more dates → re-test these 4 survivors on 10+ days. If they survive, we have the first deployable v3.2 strategy.
5. **Compare to v3.3**: when v3.3 fold 0 OOT lands (~05:30-09:30 ET morning), apply same per-head permutation test. The bar v3.3 needs to clear: log_ret_60s SHORT Top0.5% passive_net ≥ +0.743 t/fill.

## ALL TESTED (sorted by p-value)

| Head | Side | Band | n | obs | null_mean | null_p95 | p | verdict |
|---|---|---|---|---|---|---|---|---|
| log_ret_60s | SHORT | Top0.5% | 76 | 1.119 | 0.189 | 0.423 | 0.0000 | 🟢 SIGNIF |
| log_ret_60s | SHORT | Top1% | 137 | 0.803 | 0.118 | 0.297 | 0.0000 | 🟢 SIGNIF |
| log_ret_60s_q10 | SHORT | Top10% | 171 | 0.587 | 0.453 | 0.514 | 0.0000 | 🟢 SIGNIF |
| log_ret_1s | SHORT | Top0.1% | 35 | 1.390 | 0.747 | 1.296 | 0.0280 | 🟢 SIGNIF |
| log_ret_1s | SHORT | Top0.5% | 172 | 0.626 | 0.366 | 0.629 | 0.0530 | ❌ borderline |
| p_reversal_15s | SHORT | Top0.5% | 300 | 0.789 | 0.568 | 0.840 | 0.096 | ❌ |
| log_ret_60s | LONG | Top0.1% | n.r. | 0.611 | 0.336 | 0.791 | 0.182 | ❌ |
| p_reversal_60s | SHORT | Top0.1% | n.r. | 0.758 | 0.588 | 1.018 | 0.287 | ❌ |
| log_ret_30s | SHORT | Top5% | 1617 | 0.588 | 0.563 | 0.660 | 0.352 | ❌ |
| log_ret_30s_q50 | SHORT | Top0.5% | 116 | 0.885 | 0.824 | 1.139 | 0.392 | ❌ |
| log_ret_30s_q10 | SHORT | Top0.5% | 412 | 0.776 | 0.743 | 1.021 | 0.425 | ❌ |
| log_ret_30s_q50 | SHORT | Top20% | 5840 | 0.580 | 0.578 | 0.627 | 0.467 | ❌ |
| log_ret_10s_q10 | SHORT | Top0.1% | 100 | 1.556 | 1.546 | 2.063 | 0.479 | ❌ |
| log_ret_10s_q10 | SHORT | Top1% | 625 | 0.654 | 0.651 | 0.881 | 0.479 | ❌ |
| p_up_5s | SHORT | Top20% | 10746 | 0.569 | 0.570 | 0.620 | 0.505 | ❌ |
| log_ret_30s_q50 | SHORT | Top5% | 1493 | 0.590 | 0.591 | 0.691 | 0.505 | ❌ |
| p_up_30s | SHORT | Top20% | 10317 | 0.591 | 0.593 | 0.641 | 0.513 | ❌ |
| p_up_30s | SHORT | Top10% | 5035 | 0.586 | 0.590 | 0.658 | 0.521 | ❌ |
| log_ret_30s_q10 | SHORT | Top0.1% | 109 | 1.578 | 1.616 | 2.116 | 0.522 | ❌ |
| log_ret_30s | SHORT | Top0.5% | 122 | 0.692 | 0.704 | 1.034 | 0.525 | ❌ |
| p_up_10s | SHORT | Top20% | 10551 | 0.570 | 0.571 | 0.619 | 0.531 | ❌ |
| log_ret_60s_q50 | SHORT | Top5% | n.r. | 0.574 | 0.582 | 0.690 | 0.552 | ❌ |
| log_ret_30s_q50 | SHORT | Top1% | 254 | 0.665 | 0.733 | 0.963 | 0.688 | ❌ |
| log_ret_30s | SHORT | Top20% | 6297 | 0.565 | 0.582 | 0.631 | 0.712 | ❌ |
| log_ret_30s | SHORT | Top10% | 3229 | 0.570 | 0.595 | 0.664 | 0.719 | ❌ |
| p_reversal_15s | SHORT | Top0.1% | 49 | 0.816 | 1.092 | 1.710 | 0.752 | ❌ |
| log_ret_5s | SHORT | Top0.1% | 35 | 0.833 | 1.155 | 1.718 | 0.828 | ❌ |
| (p_reversal_30s SHORT Top0.1/0.5% returned NaN — small effective n) | | | | | | | | |

---

## Why the all-night research missed this

Pass 1-8 used `pred_log_ret_5s` as the directional anchor for the "tradable pocket" search (passes 6-8). That gave Top5% Sharpe 3.85 → p=0.39 (failed). The SAME methodology with `pred_log_ret_60s` as the ranker → Top0.5% Sharpe much higher and p=0.0000. **The 60-second horizon prediction has more signal than the 5-second prediction at high-confidence top-band SHORT positions.** This makes physical sense: passive FIFO trades take seconds to hours to play out at TP4/SL3 → a 60s-horizon predictor is closer to the actual realization horizon than a 5s-horizon predictor.

Tested top-30 candidates by `passive_net_after_comm`. Recommend re-running with top-60 or testing every cell with n_fills ≥ 30 (~150 cells, ~15 min compute) to find any other survivors.

---
Total compute: ~3 minutes Jupiter CPU for 30 × 1000 permutations.
