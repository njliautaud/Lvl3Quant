# HC #486 Step 3 — Stream-Coherent Base Model + Meta-Layer Training Spec

**Status:** Plan only. Step 0 (fold-0 v3.4.2 baseline anchor) must land first.

---

## 0. Executive Premise

The retrospective work already proved that *filtering* an off-the-shelf prediction stream cannot manufacture edge — auto-correlation is endogenous to the model and external book-pressure streams only "work" with K-event lookahead. The only remaining lever is to *train* the model so its outputs are coherent enough to be filterable. This spec adds (a) a sub-second head, (b) a supervised stream-coherence auxiliary loss whose label is realized future market motion (NOT model self-agreement), then (c) a meta-gate trained on the resulting prediction stream + book + quote. Pressure-exit is offline-tested only.

---

## A. Architectural changes to CNN-Mamba v3.4.2

### A.1 The new head — decision: `log_ret_250ms`, NOT `log_ret_200ms`

User's verbatim ask says "200ms" but the trainer is event-cadence with `STRIDE = 250` (events) — confirmed at `train_cnn_mamba_v3_2.py:163`. Wall-clock per event is ~6–10 ms in active periods. Label horizon is independent of stride; we can build any sub-second label.

The label pipeline in `scripts/generate_v3_2_alpha_labels.py` uses `HORIZONS_SECS = [1, 5, 10, 30, 60]` (line 66) — `_two_pointer_indices(ts_ns, horizon_ns)` takes ns directly, not int seconds. **Decision: ship `target_mfe_250ms_ticks` / `target_mae_250ms_ticks` and `pred_log_ret_250ms`**. Rationale:

- 250 ms is short enough to satisfy the user's "200ms" intent.
- 250 ms reliably contains ≥1 trade event (median MBO trade gap RTH <60 ms) — labels won't be mostly NaN.
- 200 ms hits the boundary-NaN floor we observed in HC #485 audits.

**Files to modify (when implementation begins — NOT NOW):**

1. `scripts/generate_v3_2_alpha_labels.py` line 66: add `0.25` to `HORIZONS_SECS` (refactor horizon-name fmt: `target_mfe_250ms_ticks` for h=0.25).
2. `alpha_discovery/deep_models/train_cnn_mamba_v3_2.py`:
   - line 182: add `"log_ret_250ms"` to `DIR_REG_HEADS`.
   - lines 224–243: add `LOSS_LAMBDA["log_ret_250ms"] = 1.0`.
   - Model class auto-extends via `nn.ModuleDict` at line 412 — no model surgery.
   - `SmartV32Dataset.__getitem__` (line 1048): add `labels_250ms` to per-day NPZ loader; semantics match `labels_1s` in `scripts/precompute_labels_smart_v3.py`.

### A.2 Output of the new head

Same `nn.Linear(trunk_dim, 1)` as siblings, MSE loss (`JointMultiHeadLossV32.forward` lines 554–559, `name.startswith("log_ret")` branch). `LOG_RET_CAP_TICKS = 100.0` (line 250) stays — non-binding for 250 ms.

### A.3 Why we are NOT touching trunk dim, backbone widths, or quantile/path head set

One-shot minimum-risk addition. ≈193 new params in a multi-million-param trunk cannot meaningfully perturb 1s/5s/10s/30s heads under multi-task averaging. Sub-trunks, attention reweighting → v3.5, out of scope.

---

## B. Stream-stability auxiliary loss

### B.1 The pitfall to avoid

"Stream stability" has three meanings; only one is supervised:
- (W1) Model self-agreement across consecutive predictions — gameable by constant output.
- (W2) Agreement with an external signal — rejected by Step 2 work.
- **(CORRECT)** Agreement with REALIZED future log-returns. Treat the next K realized event-log-returns as a supervised target series; model predicts properties (sign-consistency, drift, variance).

### B.2 The label

For event index t, the realized stream window = next K event-log-returns at native event cadence:

- `stream_sign_consistency_K`: frac of {Δp_{t+1},…,Δp_{t+K}} whose sign matches `sign(Σ Δp)`. Range [0.5, 1.0]; mapped `2*(x-0.5)` to [0,1].
- `stream_drift_K_ticks`: `sum(Δp_{t+1..t+K})`, capped ±50.
- `stream_var_K_ticks`: `std(Δp_{t+1..t+K})`, capped 50.

**Decision: K = 40.** At stride=250 events, ~10 ms wall-clock active, K=40 events ≈ 400 ms = the model's own freshness horizon. Sits inside existing 1s/5s windows so leakage-prevention identical.

### B.3 Where to put it in the loss

Three new heads in `ALL_HEAD_NAMES` (line 208 of `train_cnn_mamba_v3_2.py`):

- `p_stream_sign_consistency_K40` — BCE routing via `p_*` matcher at line 539.
- `pred_stream_drift_K40_ticks` — Huber via `_ticks` matcher at line 548.
- `pred_stream_var_K40_ticks` — Huber, same routing.

`LOSS_LAMBDA`:
- `p_stream_sign_consistency_K40: 0.5`
- `pred_stream_drift_K40_ticks: 0.5`
- `pred_stream_var_K40_ticks: 0.3`

Fixed lambdas as prior; if launching via uncertainty-weighted v3.3 trainer, learnable log-σ auto-calibrates.

### B.4 Label generation

In `scripts/generate_v3_2_alpha_labels.py`, add `_compute_stream_metrics(p_path, ts_ns, K=40)` returning 3 arrays. Wire into `process_one_date` alongside horizon loop at line 195. Boundary-NaN where `t+K` exceeds session close.

Dataset wiring: alpha keys loaded via `alpha.get(...)` in dataset `__getitem__` line 1080; add 3 keys after reversal-heads block at line 1159.

### B.5 No leakage gate

K=40 events ≈ 400 ms ⊂ existing 60 s MFE/MAE label window. Zero new leakage surface. Streaming-causal normalization (HC #470 R2) applies to features only — already enforced at `_compute_feature_stats` line 912.

---

## C. Meta-layer ("metal layer")

### C.1 Two phases — strict ordering

**Phase i:** train base model (Sections A + B). Phase ii does NOT begin until Section E gates pass.

**Phase ii:** small model consuming (a) window of recent base preds, (b) current book/quote, (c) cheap event-cadence features. Output: `meta_gate_score ∈ [0,1]`.

### C.2 Choice: LightGBM, not MLP

- ~1.5–3M samples/fold at stride=250.
- Trains in 5–15 min on Jupiter CPU; MLP needs GPU dispatch.
- Interpretable feature importance for regime-failure debug.
- Easy ensemble at inference.

If GBT underfits → 3-layer MLP fallback. Prior on financial gating: GBT wins.

### C.3 Starting feature set — 12 features

1. `pred_log_ret_250ms_t`
2. `pred_log_ret_1s_t`
3. `pred_log_ret_10s_t`
4. `mean(pred_log_ret_250ms_{t-K+1..t})` (K=40)
5. `sign-consistency(pred_log_ret_250ms_{t-K+1..t})` — model-stream as FEATURE only
6. `pred_stream_sign_consistency_K40_t`
7. `pred_stream_drift_K40_ticks_t`
8. `current_spread_ticks`
9. `current_book_imbalance` (event-feature idx 12, line 594 derive)
10. `signal_persistence` (event-feature idx 17)
11. `last_K_realized_drift_ticks` (past prices, no leakage)
12. `time_since_rth_open_s`

Prune to 6–8 after first GBT feature-importance pass.

### C.4 Training target

**Decision: realized 5-second forward log-return in ticks, capped ±20, regressed.** NOT PnL — PnL conflates exit policy with entry quality.

Why 5 s: meta-gate filters at *shorter* horizon than base model's best (10 s). Measures next-step decision quality, not asymptotic prediction.

Classification version ("≥X ticks within 5s") is a derived metric for threshold search only.

### C.5 Training procedure

- Inference base model over full fold-0 train window → ~3M training rows.
- Time-respect split: train first 40 of 60 train days, val last 20.
- LightGBM: `num_leaves=31, lr=0.05, n_estimators=500, early_stopping=30`.
- Score on held-out OOT (5 days same fold).
- Calibrate gate threshold via OOT grid search (policy param — OOT-tuned policy is canonical).

---

## D. Pressure-based exit logic (offline-only)

### D.1 Definitions

- **Stream sign-flip exit:** at event t after entry, compute `sign(mean(pred_log_ret_250ms_{t-K+1..t}))` K=40. Exit on opposite sign for **3 consecutive** events (debounce).
- **Coherence-drop exit:** exit when `pred_stream_sign_consistency_K40_t < 0.55` for 3 consecutive events.
- **Hard cap:** HC #428 R2 caps still apply — 1.5h max hold, TP at p90 MFE, SL at p90 |MAE|. Pressure exits fire first in healthy trades; caps = insurance.

### D.2 Backtest comparison

Same fold-0 inference NPZ, two exit policies side by side on identical meta-gated entries:
- **Baseline:** bracket-exit-v1.
- **Pressure:** sign-flip OR coherence-drop OR cap.

Metrics: net PnL, hold-time dist, WR, prof-days, Sharpe, regime-stratified Sharpe (HC #428 R1), day-conc (HC #344).

### D.3 Live wiring BLOCKED

Live pressure-exit needs T2/T3 real-time producers — separate pipeline, out of scope. Eventual home: `jupiter_v33_shadow_processor.py`. Document and stop.

---

## E. Acceptance gates (binding)

ALL must pass on fold 0 OOT:

1. **250 ms head edge:** Concat IC `pred_log_ret_250ms` > 0 by ≥1 SD of bootstrap (1000 samples). ≤0 → dead head, nothing to gate.
2. **Stream-stability calibration:** Spearman IC between `pred_stream_sign_consistency_K40` and realized > 0.08. Below → aux loss learned nothing.
3. **No poisoning:** Concat IC at 1s/5s/10s/30s within 0.01 of v3.4.2 baseline. Drop >0.01 → rollback.
4. **Regime-agnostic Sharpe (HC #428 R1):** `|Sharpe_g − Sharpe_r| / max ≤ 0.50` on pressure-exit net returns.
5. **MFE-within-horizon (HC #428 R2):** TP ≤ p90 MFE.
6. **Day concentration (HC #344):** ≤ 0.70 single-day PnL.
7. **Profitable-day count:** ≥ 30/47 OOT days net positive.
8. **Long-short balance (HC #475 R1):** both sides pass OR documented short-only per HC #475 R2.

Gates 1+2+3 alone = green-light Phase ii. Rest needed at Step 6 (offline backtest).

---

## F. Risk + rollback

### F.1 Failure modes

- **(F1) New head poisons existing.** Gate 3 fail → rollback to v3.4.2 fold-0 ckpt. Diagnose: halve `LOSS_LAMBDA["log_ret_250ms"]` and aux lambdas; retry once.
- **(F2) 250 ms head IC ≤ 0.** Two cases:
  - Label too noisy (likely): drop 250 ms head, ship aux-loss-only variant.
  - No capacity at sub-second (less likely): trunk widening → v3.5.
- **(F3) Stream-stability heads collapse to constant.** BCE plateau at label mean. Fix: raise λ OR switch from uncertainty-weighted v3.3 to fixed-lambda v3.2 trainer (1-line change).
- **(F4) Meta-gate doesn't beat unconditional.** GBT val R² ≤ 0 → revert to single-snapshot top-X% gating, redesign feature set. Phase i still ships.

### F.2 Compute budget (Neptune RTX 3090)

- v3.4.2 fold-0 baseline (in flight): 6–8 hours wall.
- +4 heads (1 dir + 3 aux): ≤5% wall increase. **6.5–8.5 hours/fold**.
- 10-fold serial: ~3 days. Parallel fleet: ~1 day.
- Label regen Jupiter CPU: ~100 hours serial, ~13 hours 8-core parallel.
- Meta-layer GBT: 15 min Jupiter.

---

## G. Ordering / dependencies

DO NOT REORDER:

- **Step 0 (NOW, in-flight):** Wait for Neptune v3.4.2 fold-0 retrain. Baseline anchor for gate 3. NO Step 1 work until this lands.
- **Step 1:** Generate 250ms + K40 stream-stability label columns. Offline Jupiter CPU via updated `generate_v3_2_alpha_labels.py`. NaN-audit per HC #485 before bulk regen.
- **Step 2:** Wire new head + aux loss into `train_cnn_mamba_v3_2.py` (or v3_4_pyramid if active). Single trainer edit, no arch changes. Smoke-test 5 days locally before dispatch.
- **Step 3:** Dispatch fold 0 of new model. Check gates 1, 2, 3.
- **Step 4:** Pass → full 10-fold across fleet. Fail → F.1 rollback, single retry, re-evaluate spec.
- **Step 5:** Meta-layer training. ONLY after base model in `models_alpha_signed/` with metadata cards.
- **Step 6:** Offline pressure-exit backtest vs bracket-v1. Gates 4–8.
- **Step 7:** Live wiring — DEFERRED, blocked on live T2/T3 producer pipeline.

---

## Executive Summary

We are building a sub-second prediction head plus a supervised stream-coherence auxiliary loss into the existing v3.4.2 CNN-Mamba alpha model, then training a small LightGBM meta-gate on the resulting prediction stream + book context. The new head is `pred_log_ret_250ms` (not 200 ms — sub-stride causes label-NaN explosion; 250 ms is the practical floor). The auxiliary loss has three new heads regressing realized future stream sign-consistency, drift, and variance over K=40 events (~400 ms wall clock). The auxiliary labels are REALIZED future returns, not model self-agreement — that's the entire reason this differs from the rejected Step 1/2 retrospective filter work. The meta-gate is a LightGBM regressor on 12 features (last-K base preds, current book/quote state, regime hints) targeting realized 5-second forward log-return. Pressure-based exits are tested offline only; live wiring is blocked on the live T2/T3 producer pipeline and explicitly out of scope. **First concrete action after Neptune fold-0 lands:** extend `scripts/generate_v3_2_alpha_labels.py` to emit the four new label columns and run the per-date NaN audit on three reference days before bulk regen.

---

### Critical Files for Implementation
- `/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_cnn_mamba_v3_2.py`
- `/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_cnn_mamba_v3_4.py`
- `/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_cnn_mamba_v3_4_pyramid.py`
- `/home/jupiter/Lvl3Quant/scripts/generate_v3_2_alpha_labels.py`
- `/home/jupiter/Lvl3Quant/DIRECTIVES.md`
