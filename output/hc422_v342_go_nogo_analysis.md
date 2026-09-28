# HC #422 — CNN-Mamba v3.4.2 GO / NO-GO Analysis

**Date**: 2026-05-18 (final draft per HC #422 Rule 4)
**Author**: HoQ
**Subject**: Neptune PID 311170, v3.4.2 fold-0, currently Ep 3 batch ~228,500 / 265,649 (~86%), 2 epochs remain
**Active log**: `/home/nick/Lvl3Quant/logs/v3_4_2/v342_resume_book_gate_fix_20260517_213545.log`
**MLflow run**: `e5f0f79b313d4ac4aa461df8b7af2385` (exp `CNNMamba_v3_4_2_fixed_mtl`)
**HC binding**: HC #422 Rule 2 (TP/SL-conditioned heads FORBIDDEN), Rule 4 (default = halt + pivot if inconclusive)

---

## TOP LINE — **SURGICAL**: kill v3.4.2 now, pivot Neptune to v2 retrain WITHOUT the 4 forbidden heads, route v3.3 NPZs to execution research on Jupiter.

(NO-GO on the current process. "Surgical" = the v3.4.2 *architecture* is retainable for a future relaunch; the *current run* is unrecoverable per Rule 2 plus book_gate collapse. We retrain with the same backbone + 24 permitted heads. We do NOT in-place rip heads out of the live PID — there is no checkpoint surgery procedure that keeps the multi-head loss valid without re-initializing the head ModuleDict and re-warming, which is equivalent cost to a clean restart.)

---

## 1. What v3.4.2 actually contains (heads audit)

Source: `alpha_discovery/deep_models/train_cnn_mamba_v3_2_1.py` lines 240–292 (re-used by v3_4 trainer), `train_cnn_mamba_v3_4.py` lines 74–76, 559.

**Total head set: 28 heads**, all `Linear(trunk_dim → 1)` on a shared trunk, joint multi-head uncertainty-weighted loss.

| Group | Heads | Count | HC #422 Rule 2 |
|---|---|---:|:---:|
| Directional regression | `log_ret_1s/5s/10s/30s/60s/5min` | 6 | PERMITTED |
| Directional probability | `p_up_5s/10s/30s/60s` | 4 | PERMITTED |
| Quantile (pinball) | `log_ret_{10,30,60}s_q{10,50,90}` | 9 | PERMITTED (calibration) |
| Path magnitude | `pred_mfe_{30,60}s_ticks`, `pred_mae_{30,60}s_ticks` | 4 | PERMITTED (MFE/MAE) |
| Time | `pred_time_to_mfe_secs` | 1 | PERMITTED |
| Reversal | `p_reversal_{15,30,60}s` | 3 | PERMITTED |
| Volatility | `pred_realized_vol_30s_ticks` | 1 | PERMITTED |
| **Legacy aux (FIFO TP/SL-conditioned)** | `fifo_tp4sl3_net`, `fifo_tp8sl5_net`, `fifo_tp4sl3_hit_tp`, `fifo_tp8sl5_hit_tp` | **4** | **❌ FORBIDDEN** |

**Verdict**: 24 of 28 heads are permitted. The 4 `fifo_tp*sl*_*` heads are exactly what HC #422 Rule 2 outlaws ("heads on 'fills with TP=X SL=Y' — those are leakage to specific exec setups"). They carry low λ (0.1 each in `LOSS_LAMBDA`), but they ARE in the loss graph and ARE shaping the shared trunk through gradient feedback.

## 2. In-flight performance (MLflow + log scrape)

| Metric | ep-1 (logged sep.) | ep-3 end (MLflow step 2) | v2 baseline | Pass vs v2? |
|---|---:|---:|---:|:---:|
| IC_1s | 0.247 | **0.274** | 0.222 | ✅ WIN |
| IC_5s | 0.111 | **0.128** | 0.141 | ❌ losing |
| IC_10s | 0.058 | **0.090** | 0.106 | ❌ losing |
| `book_gate_tanh` | 0.462 (init=0.5 post HC #409 fix) | **0.0158** | — | **COLLAPSED** |

The book-gate collapse is the killer. HC #416 conditioned deployment on `book_gate_tanh > 0.3`. The model has gradient-descended the spatial book trunk's contribution to ~1.6% of its post-fix init — i.e. the network has *learned to ignore the spatial pathway*. The dual-trunk thesis (temporal Mamba + spatial book CNN) is empirically falsified inside its own training.

**Optimistic ep-5 IC projection** (linear extrapolation, ep-1→ep-3 slopes):
- IC_1s → 0.301 (+0.08 vs v2) ✅
- IC_5s → 0.145 (barely crosses v2's 0.141) marginal
- IC_10s → 0.122 ✅ marginal

**Realistic ep-5 projection** (50% slope decay, standard learning-curve shape):
- IC_5s → 0.137 (still under v2) ❌
- IC_10s → 0.114 (barely over v2) marginal

Even on the optimistic curve, the model crosses v2 at 5s/10s only in the final epoch, and the book_gate problem remains regardless of IC.

## 3. Cost of each path

| Path | Neptune-GPU-hours | What we get | What we lose |
|---|---:|---|---|
| **GO (finish ep-3, ep-4, ep-5)** | ~9.0h (1.0h ep-3 tail + 4.0h ep-4 + 4.0h ep-5) | A final IC verdict on a model that is (i) Rule-2 non-compliant and (ii) book-gate-collapsed → cannot deploy regardless of IC. | 9h of pre-RTH-tomorrow Neptune time. Blocks v2 weekly-retrain (HC #344 overdue 3 weeks per today's HC #421-A diagnosis). |
| **NO-GO (kill now, do nothing)** | 0h | Sunk-cost release. ep-1 + ep-3 IC numbers already logged in MLflow. | Wasted the ~17h already burned, but those are sunk regardless. |
| **SURGICAL (kill now, relaunch v2 retrain with 24 permitted heads + book-gate-fix init, training cutoff 2026-05-15)** | ~9.0h pivoted | (a) Fresh v2 with current-week data — directly addresses HC #421 Issue A regime-drift (live pred_1s mean +0.27 vs backtest +0.06, std collapsed 0.43→0.26). (b) Rule-2-compliant head set deployable immediately. (c) Frees Jupiter to start v3.3 execution research (Rule 5) using existing v3.3 NPZs — zero new GPU cost. | Nothing — the v3.4.2 checkpoints stay on disk for forensic comparison. |

## 4. Can we keep v3.4.2 and surgically rip out the TP/SL heads?

No, not in-place on PID 311170. The loss function `JointMultiHeadLossV33_UncertaintyWeighted(head_names=ALL_HEAD_NAMES)` is parameterized at construction with the full 28-head list and owns per-head learnable uncertainty parameters. Removing 4 heads mid-run requires (a) editing `ALL_HEAD_NAMES`, (b) reconstructing the loss module, (c) re-initializing the `nn.ModuleDict` heads to a 24-entry version (the shared trunk is fine, but the head `ModuleDict` keys are baked in), (d) reloading optimizer state filtered for the 4 dead heads. By the time we've done that we've effectively restarted; the spec-clean path is: kill + relaunch with `LEGACY_AUX_HEADS = []` set at module load.

**However**, the v3.4.2 backbone + 24 heads is a *legitimate future model*. We can relaunch it cleanly later. That's the "Surgical" framing: surgery on the head-set definition, not on the running checkpoint.

## 5. What v3.3 execution research needs from a model

Per HC #422 Rule 8 ("execution setups must use ALL outputs"), the RL/MLP/LGBM gating agents need:

**Model heads** (all permitted, all already produced by v2 + v3.3 inference): `pred_log_ret_{1,5,10,30}s`, `p_up_{5,10,30}s`, the q10/q50/q90 quantile bands at 10s/30s, `pred_mfe_30s_ticks`, `pred_mae_30s_ticks`, `pred_realized_vol_30s_ticks`, `pred_time_to_mfe_secs`, the 3 `p_reversal_*` heads, and a confidence/rank feature.

**Plus book features**: imbalance, depth, microprice, queue position (NOT model heads — they're computed at gate-time from the live book stream).

**Critically: v3.3 NPZs already exist on disk** (HC #415 sweep produced them). Execution research can start TODAY on Jupiter CPU with zero additional GPU cost. The blocker for Jupiter idleness (HC #422 Rule 9 violation) is the dispatcher, not data availability.

## 6. Recommendation (200-300 words)

**SURGICAL — kill PID 311170 now; pivot Neptune to v2 weekly-retrain on 24 permitted heads; route v3.3 execution work to Jupiter.**

The decision is overdetermined. Three independent reasons each individually justify halt:

1. **Rule-2 non-compliance is binary and unfixable in-flight.** Even if v3.4.2 hits IC ceiling, four heads in its loss graph are explicitly forbidden by HC #422 §2. A "GO" outcome produces an undeployable model. The marginal value of finishing training is zero.

2. **Book-gate collapsed from 0.462 → 0.0158 (97% gone).** The architecture's central premise — that a spatial book-CNN trunk meaningfully augments temporal Mamba — is empirically falsified by the gradient choosing to ignore it. HC #416's deployment gate (`tanh > 0.3`) cannot pass without architectural surgery beyond what 2 more epochs can fix.

3. **IC trajectory does not catch v2 at 5s under realistic learning-curve decay.** Optimistic linear extrapolation is the only path to v2 parity, and that's the path with no decay — which is not how loss curves behave at ep-4/ep-5.

The opportunity cost is high: HC #344 mandates weekly v2 retrains; we're overdue by 3 weeks; HC #421 Issue A diagnosed live regime-drift (pred_1s mean +0.27 vs backtest +0.06) as a *consequence* of that staleness. Neptune's next 9h is far more valuable retraining v2 on data through 2026-05-15 (respecting HC #422 Rule 3 Apr-29 canonical boundary) than finishing a model that cannot deploy.

Default per HC #422 Rule 4 was "halt at ep-2 boundary, pivot to execution." We're at ep-3 ~86%. Halt now.

## 7. Immediate next actions (autonomous per HC #393)

1. `ssh neptune "kill 311170"` + MLflow tag `halted_hc422_rule4=true halted_reason=rule2_noncompliance_plus_book_gate_collapse`.
2. Relaunch v2 retrain on Neptune: backbone unchanged, `LEGACY_AUX_HEADS=[]`, training cutoff 2026-05-15, `V32_BOOK_GATE_INIT=0.5` per HC #409, weekly retrain mandate per HC #344.
3. Seed Jupiter task queue with v3.3 execution research (Rule 5) using existing v3.3 NPZs — feed all 24 permitted heads as features to RL/MLP/LGBM gating per Rule 8. No new GPU cost.

---
**End HC #422 v3.4.2 GO/NO-GO Analysis — verdict SURGICAL.**
