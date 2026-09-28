# HC #417 — CNN-Mamba v3.4.2 60d Fold-0 Ep-1 OOT Evaluation

**NPZ**: `output/cnn_mamba_v3_4_2_fixedmtl/fold_00_ep1_oot.npz` (landed 02:50 ET 5/18, 14MB)
**Wrapped**: `output/v342_ep1_eval/fold_00_ep1_oot_wrapped.npz`
**Sample count**: 241,351
**OOT dates**: 20260223, 20260224, 20260225, 20260226, 20260227 (5 dates, Feb 23-27 2026)
**Inferred from**: sample count + FIFO label per-day totals + SESSION_STATE record of prior identical-size 5-day NPZ

---

## TL;DR

v3.4.2 ep-1 fold-0 OOT (5 dates only) **does NOT beat v2 on the HC gauntlet**. IC_1s = 0.2437 (slightly above v2 baseline 0.236), but IC_5s = 0.1100 (BELOW v2 0.127) and IC_10s = 0.0689 (BELOW v2 0.090). HC #413 backtester finds 4 positive-net cells (1s_long/short × top05/top1) with strong Sharpe/PF but **all fail HC #408** on day_conc > 0.20 — same regime-fragility we saw on the v3.4.2 16d window. HC #415 multi-gate sweep on `target_fifo_tp4sl3_net` produces **0 of 476 cells passing rule 2** across all signal/side/tier/gate enumerations. HC #411 sub-window (N=3, the max meaningful for 5 dates) PASSES for 3 of 4 winning cells. Net verdict: **no cell qualifies for live deployment.** v2 retains the live-deployment recommendation (`v2_1s_short_top05` per HC #417 deployment spec). Recommend re-evaluating v3.4.2 only after fold-0 ep-2 NPZ lands on the full 60d window — current 5-day evidence is insufficient to falsify or confirm the temporal+spatial bet.

---

## 1. NPZ schema

NPZ contains 53 prediction heads spanning:
- **Log-return heads (point + quantile)**: 1s, 5s, 10s, 30s, 60s, 5min — point preds + q10/q50/q90 for 10s, 30s, 60s. (60s and 5min are EMPTY — masks=0.)
- **Probability heads**: p_up_5s/10s/30s/60s, p_reversal_15s/30s/60s. (60s heads empty.)
- **MFE/MAE/time-to-MFE/realized-vol heads**: 30s ticks (active), 60s ticks (empty).
- **Direct FIFO realized-net heads**: tp4sl3_net, tp4sl3_hit_tp, tp8sl5_net, tp8sl5_hit_tp — model is now supervised on canonical FIFO outcomes.
- **Masks** for every head; valid masks on FIFO labels: ~24% of samples (59,024 valid out of 241,351).

Compared to v2 (3 log_ret heads only), v3.4.2 has a vastly richer multi-task head — temporal+spatial book trunk + MFE/MAE/reversal/vol/direct-FIFO predictions. NPZ size: 14MB vs v2's 3-head equivalent.

---

## 2. Concat IC (vs v2 baseline, 46-date)

| Head | v3.4.2 IC (5-date) | v2 IC (46-date) | Delta |
|---|---:|---:|---:|
| log_ret_1s | **+0.2437** | +0.236 | +0.008 |
| log_ret_5s | +0.1100 | +0.127 | -0.017 |
| log_ret_10s | +0.0689 | +0.090 | -0.021 |
| log_ret_30s | +0.0227 | n/a | n/a |
| p_up_5s | +0.1375 | n/a | n/a |
| p_up_10s | +0.0960 | n/a | n/a |
| p_reversal_15s | +0.1177 | n/a | n/a |
| pred_mfe_30s_ticks | **+0.3202** | n/a | n/a |
| pred_mae_30s_ticks | **+0.3610** | n/a | n/a |
| pred_realized_vol_30s | **+0.6311** | n/a | n/a |
| pred_fifo_tp4sl3_net | -0.0037 | n/a | n/a |
| pred_fifo_tp4sl3_hit_tp | -0.0537 | n/a | n/a |
| pred_fifo_tp8sl5_net | +0.0279 | n/a | n/a |
| pred_fifo_tp8sl5_hit_tp | +0.1281 | n/a | n/a |

**Key observations**:
- IC_1s is marginally above v2 baseline. Within sampling error on 5 dates.
- IC_5s and IC_10s are BELOW v2 baseline — temporal+spatial trunk is not improving the longer-horizon log-ret prediction at ep-1.
- **The direct FIFO-realized-net heads (`pred_fifo_tp4sl3_net`, `pred_fifo_tp4sl3_hit_tp`) have IC near 0 or NEGATIVE.** The network is not yet learning to predict realized FIFO P&L despite being multi-task supervised on those labels. Possible reasons: head weight too low; need more epochs; or the FIFO label is structurally noisier than log-ret.
- MFE/MAE/vol auxiliary heads ARE well-calibrated (IC 0.32 / 0.36 / 0.63) — these were added in v3.4.2 and the model is learning them strongly.
- **Caveat**: 5-date IC is high-variance. v2 baseline is on 46 dates — sampling-noise floor is much higher on 5 dates. Treat all IC deltas vs v2 as PRELIMINARY only.

---

## 3. HC #413 scalping backtester (canonical FIFO market replay)

Config: passive_at_touch (entry cost 0.376 tk), per-cell MFE-derived TP1/TP2/SL from `output/hc411_regime_agnostic_20260517_215211/mfe_at_confidence_matrix.csv` (model=v3.4.2). 10s cancel window. 24 cells (3h × 2s × 4t).

**Cells passing HC #408 honesty (n>=50, CI95lo>0, day_conc<=0.20)**: **0 / 24**.

**Top positive-net cells** (all fail HC #408 on day_conc > 0.20):

| cell_id | n_fills | net/fill | Sharpe√N | PF | WR% | day_conc | CI95lo |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3.4.2_1s_long_top1 | 173 | +0.256 | 6.46 | 2.62 | 78.6 | 0.262 | +0.183 |
| v3.4.2_1s_long_top05 | 79 | +0.240 | 3.74 | 2.49 | 79.7 | 0.376 | +0.115 |
| v3.4.2_1s_short_top05 | 131 | +0.209 | 3.73 | 2.06 | 77.1 | 0.307 | +0.096 |
| v3.4.2_1s_short_top1 | 263 | +0.170 | 4.63 | 1.79 | 75.3 | 0.351 | +0.102 |
| v3.4.2_30s_long_top1 | 155 | +0.296 | 2.36 | 1.56 | 81.3 | 0.337 | +0.059 |
| v3.4.2_5s_short_top1 | 253 | +0.101 | 1.91 | 1.32 | 79.1 | 0.452 | -0.000 |

day_conc failures are expected on 5-date windows (any non-uniform day will breach 20%). Compare to the 16d window results (v3.4.2_1s_short_top05: day_conc 0.227 — also fails). **The fragility is consistent, not improved by more dates within this window.**

**Comparison vs v2 winning cells (DIFFERENT date scope — Mar-Apr 36 days)**:

| cell | v3.4.2 (Feb 23-27, 5d) | v2 (Mar-Apr, 36d) |
|---|---|---|
| 1s_short_top05 | n=131, net=+0.209, dayC=0.31, CI95lo=+0.10 | n=639, net=+0.274, dayC<=0.20, CI95lo=+0.232, pdpr=1.00 |
| 1s_short_top1 | n=263, net=+0.170, dayC=0.35, CI95lo=+0.10 | n=1290, net=+0.223, dayC<=0.20, CI95lo=+0.195, pdpr=1.00 |
| 5s_short_top1 | n=253, net=+0.101, dayC=0.45, CI95lo=-0.000 | n=1433, net=+0.146, dayC<=0.20, CI95lo=+0.109, pdpr=0.96 |
| 10s_short_top1 | n=238, net=-0.031, dayC=0.70 | n=1453, net=+0.079, dayC<=0.20, CI95lo=+0.037, pdpr=0.80 |

Same cells: **v2 has BETTER net/fill AND passes HC #408 honesty**. v3.4.2 fails on day_conc in all comparisons. Caveat: different date scope — not a true head-to-head.

---

## 4. HC #415 multi-gate sweep

Operated on `target_fifo_tp4sl3_net` (canonical FIFO realized net). 476 cells evaluated across 6 signals × 2 sides × 4 conf tiers × 14 gate combos.

**Cells passing HC #415 rule 2**: **0 / 476**. (Same result for `target_fifo_tp8sl5_net`.)

Top cells by Sortino√N have n_fills < 50 (fail HC #408 floor) or wildly negative CI95lo on the FIFO label. The multi-output gates (horizon-confluence, MFE-room, reversal-low, uncertainty-tight, vol-mid, direct-sign-agree) do NOT rescue the 5-date window from regime fragility.

---

## 5. HC #411 sub-window stability (N=3)

Run on the 4 positive-net cells from HC #413. With 5 OOT dates, N=3 is the maximum meaningful split. Rule: net > 0 in EVERY sub-window AND CI95lo > 0 in >= 2/3 sub-windows.

| cell | all_net_positive | ci_pos_count | regime_stable_v342? |
|---|:-:|:-:|:-:|
| v3.4.2_1s_long_top05 | YES | 2/3 | **PASS** |
| v3.4.2_1s_long_top1 | YES | 3/3 | **PASS** |
| v3.4.2_1s_short_top05 | NO (one sub-window net<0) | 2/3 | FAIL |
| v3.4.2_1s_short_top1 | YES | 2/3 | **PASS** |

**3 of 4 positive-net cells pass HC #411 at N=3.** Insufficient evidence of regime stability across longer windows (would require fold-0 ep-2/3 on full 60d).

---

## 6. HC gauntlet — qualification matrix

| cell | HC #408 (n>=50, CI95lo>0, dayC<=0.20) | HC #415 rule 2 | HC #411 N=3 | Live-deploy qualifier? |
|---|:-:|:-:|:-:|:-:|
| v3.4.2_1s_long_top05 | FAIL (dayC 0.38) | FAIL | PASS | **NO** |
| v3.4.2_1s_long_top1 | FAIL (dayC 0.26) | FAIL | PASS | **NO** |
| v3.4.2_1s_short_top05 | FAIL (dayC 0.31) | FAIL | FAIL | **NO** |
| v3.4.2_1s_short_top1 | FAIL (dayC 0.35) | FAIL | PASS | **NO** |

**Zero cells from v3.4.2 ep-1 fold-0 OOT qualify for live deployment.**

---

## 7. Bottom line

- **Did v3.4.2 (temporal+spatial, with book head) beat v2 (temporal only, no book)?** On this 5-date sample, NO. IC_1s is marginally higher; IC_5s and IC_10s are lower; HC #408 is universally failed; HC #415 produces zero passes.
- **Is the model broken?** No — it is well-calibrated on auxiliary heads (MFE/MAE/vol ICs are strong) and the 1s short/long alpha is clearly real (n_fills ~ 80-260 with PF > 1.7 and WR 75-80%). The failure mode is regime-concentration on a 5-date OOT window.
- **What to do**:
  1. Wait for fold-0 ep-2 to land (training rolled into ep-2 at 03:35 ET, ~15%; ETA ~9-10h). Re-evaluate on whatever OOT scope ep-2 emits — but ep-2's OOT may still be the same 5 Feb dates.
  2. **Real test** requires extending OOT to the full ~46-date HC #417 window. Per current training script behavior, that needs more folds (or an extended inference run).
  3. Do NOT change anything in the live v2 paper-trader. HC #417 deployment spec for `v2_1s_short_top05` remains the only qualified live-deploy config.
- **Honesty note**: 5 dates is insufficient evidence to confirm OR falsify the v3.4.2 hypothesis. The HC #408/#415 gauntlet on a 5-date window is structurally biased against passing (day_conc <= 0.20 is mechanically harder on 5 days). HC #411 N=3 PASS for 3 of 4 cells is the only meaningfully positive signal from this run.

---

## Files

- `output/v342_ep1_eval/fold_00_ep1_oot_wrapped.npz` (93MB wrapped + n_samples + oot_dates)
- `output/v342_ep1_eval/ic_summary.csv`
- `output/v342_ep1_eval/hc413_backtest/scalping_backtest_results.csv` (+ verdict.md)
- `output/v342_ep1_eval/hc415_multigate/results.csv` (+ hc415_pass.csv [empty] + verdict.md)
- `output/v342_ep1_eval/hc415_multigate_tp8sl5/results.csv` (same — 0 passes)
- `output/v342_ep1_eval/hc411_subwindow_results.csv` (+ hc411_subwindow_summary.csv)
- Scripts: `scripts/v3_4_research/hc417_v342_ep1/{wrap_v342_ep1_npz.py, compute_ic.py, subwindow_stability.py}`

---

## HC compliance

- HC #74/#377/#397B: canonical FIFO market replay, no midpoint shortcuts.
- HC #69: risk-adjusted metrics (Sharpe, Sortino, PF, WR) reported as primary.
- HC #344: day_conc reported; failures explicit.
- HC #408: honesty gate (n>=50, CI95lo>0, day_conc<=0.20) applied; zero cells pass.
- HC #411: sub-window stability at N=3 (the max meaningful for 5 dates) applied; 3/4 positive-net cells pass.
- HC #413: TP1=0.5·MFE, TP2=1.0·MFE, SL=min(|MAE|, 1.5·MFE) — applied per-cell from MFE matrix.
- HC #415 rule 2: all-OOT-date stability with multi-output gating; zero of 476 cells pass.
- HC #417: full-OOT window requirement NOT MET (5 dates only). Awaiting later epochs / folds to expand OOT scope.
