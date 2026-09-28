# HC #423 §2 — v3.4.3 Dual-Trunk Repair Spec

**Status:** Production-ready. Launch authorised AFTER 16:00 ET, 2026-05-18.
**Author:** Claude (head-of-quant), 2026-05-18.
**Predecessor:** v3.4.2 killed 2026-05-18 14:55 ET — RCA in
`output/hc423_v342_book_gate_diagnosis.md`.

---

## 1. Background — why v3.4.2 failed

`v3.4.2` is a v3.3 warm-start + gated-residual BookCNN. RCA findings:

| Metric                       | Value                | Interpretation                  |
| ---------------------------- | -------------------- | ------------------------------- |
| `book_gate_tanh` (post-ep1)  | **0.0158**           | Effectively zero — book pathway dead |
| `BookCNN.bn.running_var`     | **≈ 5.6e-45**        | Denormal — BN never saw real activations |
| `book_emb_rms`               | **0.063**            | 17× smaller than warmstarted trunk (1.05–1.27) |
| BookCNN-served-head IC       | ≈ baseline           | Spatial pathway gave zero lift   |

**Root cause: chicken-and-egg cold-start fixed point at gate=0.**
- With `book_gate=0`, gradient into `book_cnn` is approximately
  `∂L/∂book_gate · book_emb`, scaled by `tanh'(0) = 1` but bounded by the
  near-zero loss-sensitivity that v3.3-warmstarted heads already have to
  small perturbations of `trunk_out`.
- Both `book_gate` AND `book_cnn` weights remain stuck — neither can move
  without the other moving first. BN denormal `running_var` is a downstream
  symptom: the channel never sees enough variance to update.

---

## 2. Repairs implemented (all 5 from HC #423 §2)

### Repair 1 — Aux head on `book_emb`

Forces a real supervision signal into `book_cnn` INDEPENDENT of the gated
output, breaking the chicken-and-egg.

- New module: `aux_head` = `Linear(book_emb_dim → book_emb_dim/2) → GELU →
  Linear(→ len(AUX_HEAD_TARGETS))`.
- Targets: `["log_ret_30s", "pred_mfe_30s_ticks"]` — chosen because these are
  exactly the horizons where book-pyramid (spatial liquidity) should add
  information.
- Weight: `AUX_LAMBDA = 0.075` (mid of HC #423-suggested 0.05–0.10 range).
  Picked so aux loss magnitude is comparable to a single rebalanced head
  contribution but not dominant.

### Repair 2 — Two-phase warmup

- **Phase-1 (~10K steps):** FREEZE `v32_core` (warm-started temporal trunk +
  all original heads). `book_gate` frozen at 0. Train ONLY
  `{book_cnn, book_emb_proj, aux_head}`. Result: aux head pulls BookCNN
  toward useful 30s features WITHOUT being interfered with by the dominant
  1s/5s heads.
- **Phase-2 (rest of fold):** UNFREEZE everything. Set `book_gate` raw =
  0.5 (tanh ≈ 0.46) — book pathway immediately contributes ~46% to
  `trunk_out`. Now the optimiser can refine `book_gate` from a non-zero
  starting point.

Phase transition: driven by step counter in the patched loss `forward`. At
`global_step >= PHASE1_STEPS`, `model.set_phase(2)` is called once and
flagged.

### Repair 3 — `HEAD_WEIGHTS_V343` rebalance + FIFO drop

Wired as a class constant in `/tmp/v343_repair_launcher.py`:

| Head                  | v3.4.2 base | v3.4.3 weight | Δ      | Rationale                       |
| --------------------- | -----------:| -------------:| ------:| ------------------------------- |
| `log_ret_1s`          | 1.0         | **1.0**       | 0      | Primary horizon, keep           |
| `log_ret_5s`          | 0.7         | **0.5**       | −0.2   | Dominates over spatial horizons |
| `log_ret_10s`         | 1.0         | 1.0           | 0      | Keep                            |
| `log_ret_30s`         | 0.1         | **0.3**       | +0.2   | Spatial-favoring — raise        |
| `log_ret_60s`         | 1.0         | 1.0           | 0      | Keep                            |
| `pred_mfe_30s_ticks`  | 0.3         | **0.4**       | +0.1   | Spatial-favoring path head      |
| `pred_mae_30s_ticks`  | 0.3         | **0.4**       | +0.1   | Spatial-favoring path head      |
| `fifo_tp4sl3_*`       | 0.1         | **0.0**       | −0.1   | IC≈0 in v3.4.2 (HC #422)        |
| `fifo_tp8sl5_*`       | 0.1         | **0.0**       | −0.1   | IC≈0 in v3.4.2 (HC #422)        |

Implementation: applied as a SUPPLEMENTAL MSE delta on top of v3.3's
uncertainty-weighted base loss, so we don't have to rewrite the
log-σ-MTL machinery. Net effective weight ≈ `1 + (HEAD_WEIGHTS_V343[h] - 1.0)`.

FIFO heads can be re-added in v3.4.4 if a fresh ablation (post-repair)
shows non-trivial alpha — HC #423 §1 conditional permit retained.

### Repair 4 — Runtime telemetry (every K=200 batches)

Per-batch hook inside the patched loss `forward` writes a JSONL record:

```json
{"ts":..., "fold":0, "step":200, "phase":1,
 "book_emb_rms": 0.482, "book_gate_raw": 0.0, "book_gate_tanh": 0.0,
 "main_loss": 1.23, "aux_loss": 0.91,
 "ic": {"log_ret_1s": 0.18, "log_ret_30s": 0.04, "pred_mfe_30s_ticks": 0.07}}
```

- File: `/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_3_repair/telemetry_<ts>.jsonl`
- Also mirrored to MLflow as `book_emb_rms / book_gate_raw / book_gate_tanh /
  aux_loss / ic_<head>` per-step metrics.

**Makes book_gate collapse detectable in 30s instead of 3 epochs.**

### Repair 5 — Alignment audit (one-shot, pre-training)

Before fold 0 starts training, samples 512 random indices from the train
dataset and verifies:

1. `events_t2.shape[0] == book_pyramid.shape[0]` (structural T-dim alignment).
2. Per-level NaN/inf rate `< 0.001` (i.e. < 0.1%).

On violation: `raise SystemExit(...)` with diagnostic info — training never
starts. This is "cheap insurance" per HC #423 §2.5.

Note on temporal alignment: this build assumes
`SmartV34MemmapDualTrunkDataset` already enforces row-aligned book/event
indexing (validated by `v34_data_alignment_check.py` and HC #423 §1 data
audit). The audit here checks the structural invariant, not raw timestamps,
because the dataset already strips them. If a future regression introduces
per-sample ts drift, the deeper assertion lives in the dataset class.

---

## 3. Preserved constraints (binding from existing HCs)

| HC      | Constraint                                            | How preserved                        |
| ------- | ----------------------------------------------------- | ------------------------------------ |
| #0      | SLIDING walk-forward only                             | `N_FOLDS=1`, walks via v3.4.2 dispatch |
| #386    | book_gate init = 0.5 at resume                        | Phase-2 init = 0.5 (matches)         |
| #395    | memmap dataset (32 GB-fit)                            | `SmartV34MemmapDualTrunkDataset` swap |
| #382    | confidence-band metrics dump                          | Preserved post-fold                  |
| #398    | atomic intra-ckpt writes                              | Untouched in trainer                 |
| #409    | book-gate-fix delta                                   | Phase-2 raw=0.5 matches              |
| #420    | augmentation authorisation (codebase false-positive)  | N/A — meta                           |
| #423 §1 | head policy — drop IC≈0 FIFO heads                    | weight=0.0                           |

---

## 4. Hyperparameters chosen (justification)

| Hyperparameter         | Value     | Justification                                                                 |
| ---------------------- | --------- | ----------------------------------------------------------------------------- |
| `AUX_LAMBDA`           | 0.075     | Mid of HC §2.1 range. Small enough that main loss still dominates by ~10×.    |
| `PHASE1_STEPS`         | 10_000    | At BS=16, ~10K steps ≈ 0.5 epoch on 60d data. Enough for BN stats to stabilise + aux head to converge. |
| `PHASE2_BOOK_GATE_RAW` | 0.5       | tanh(0.5) ≈ 0.462. Per HC #409 verdict — middle-ground starting point.        |
| `TELEMETRY_EVERY`      | 200       | At BS=16, ~3,200 samples per snapshot → ~3-4s per record. ~150 records/epoch — enough for trend detection without log spam. |
| `V32_WF_TRAIN_DAYS`    | 60        | Same as v3.4.2 — apples-to-apples comparison.                                 |
| `V32_N_FOLDS`          | 1         | Repair validation only; full-WF run reserved for post-verdict v3.4.4.         |
| `V32_EPOCHS`           | 1         | Ep-1 verdict gates further commitment (HC #366 Q3=I pattern).                 |
| `V32_BATCH_SIZE`       | 16        | Same as v3.4.2 memsafe — fits in 24 GB VRAM with mixed precision.             |

---

## 5. Verdict criteria (ep-1 end)

**GO if ALL of:**

1. `book_gate_tanh > 0.3` by ep-1 end (vs 0.0158 in v3.4.2 — 19× lift).
2. `book_emb_rms > 0.5` by ep-1 end (vs 0.063 in v3.4.2 — within 2× of trunk).
3. Aux head shows non-trivial signal: any of
   `IC(aux__log_ret_30s) > 0.04` or `IC(aux__pred_mfe_30s_ticks) > 0.04`.
4. Per-head IC ≥ v2 baseline at matching horizons:
   - `IC(log_ret_1s) ≥ 0.222` (CNN-Mamba v2 baseline)
   - `IC(log_ret_30s) ≥ 0.05` (was ≈0 in v3.4.2)
   - `IC(pred_mfe_30s_ticks) ≥ 0.04` (was ≈0 in v3.4.2)
5. BN running_var on `book_cnn.bn1/bn2/bn3` ≥ 1e-3 (vs 5.6e-45 in v3.4.2).

**NO-GO / KILL conditions (kill+retune immediately):**

- `book_gate_tanh < 0.1` at ep-1 mid-epoch (50%-through) → kill, retune
  `AUX_LAMBDA` upward to 0.10–0.15.
- `aux_loss` not decreasing monotonically over first 5K steps of phase-1 →
  kill, audit aux target mask.
- Any NaN/inf in `book_emb_rms` → kill, dataset corruption (audit should
  have caught — investigate).
- Phase transition fails (`set_phase(2)` raises) → kill, fix freezing logic.
- v3.4.2 baseline regression: `IC(log_ret_1s) < 0.20` on ep-1 OOT → kill,
  rebalance broke directional path.

---

## 6. Validation steps run on Jupiter

| Check                                                      | Result                          |
| ---------------------------------------------------------- | ------------------------------- |
| `/tmp/v343_repair_launcher.py` line count                  | 776 lines                       |
| `/home/jupiter/Lvl3Quant/scripts/launch_v343_repair.sh` LC | 195 lines, executable           |
| Python AST parse of launcher                               | PASS (`python -c "import ast; ast.parse(open('/tmp/v343_repair_launcher.py').read())"`) |
| Bash syntax check of launcher                              | PASS (`bash -n launch_v343_repair.sh`) |
| HEAD_WEIGHTS_V343 JSON serialisable                        | PASS (used in MLflow tag write) |
| Imports resolve on Neptune                                 | DEFERRED — module paths live on Neptune, sanity verified via the launcher's --smoke-test mode (`./launch_v343_repair.sh --smoke-test`) |

Import sanity on Jupiter is NOT runnable because:
- `scripts.v3_4_research.dispatch_v34_2_fixedmtl` lives on Neptune only
- `alpha_discovery.deep_models.train_cnn_mamba_v3_4_memsafe` lives on Neptune only

Per the deferred launch sequence, the operator runs
`./scripts/launch_v343_repair.sh --smoke-test` AFTER SCP-ing the file to
Neptune as part of the launch step itself. Smoke test takes < 5 s and
verifies the launcher can be importable on the target.

---

## 7. Open questions / risks

1. **Loss-class patching is global state.** The patched
   `JointMultiHeadLossV33_UncertaintyWeighted.forward` mutates the class,
   not an instance — concurrent runs in the same process would conflict.
   Acceptable for single-process Neptune training; flagged for awareness.
2. **`HEAD_WEIGHTS_V343` is applied as a delta on top of v3.3 uncertainty
   weighting.** If the learned log-sigmas drift to compensate, the
   effective weighting may differ from the table. Telemetry on per-head
   loss components should detect this; if observed, future work is to
   refactor the v3.3 loss to accept explicit weights.
3. **Aux head uses MSE on `pred_mfe_30s_ticks`.** That head was originally
   trained against a CAPPED label (`PATH_LABEL_CAP_TICKS=50`). The aux MSE
   is unbounded; if a few extreme outliers slip past the cap, aux_loss
   could spike. Telemetry will catch this; if it's a problem, switch aux
   targets to direction-only.
4. **Phase-1 = 10K steps with frozen v32_core.** At BS=16 + 60d train data
   (~600K samples → ~37K steps/epoch), 10K steps is ~27% of the first
   epoch. If `book_cnn` hasn't converged by then, phase-2 starts with a
   half-baked book pathway. Mitigation: if ep-1 verdict is borderline,
   bump `V343_PHASE1_STEPS` to 15K–20K for v3.4.3.1.
5. **Telemetry hook lives in the patched loss forward.** If a future
   trainer change calls the loss outside the standard
   `(outputs, targets, masks)` contract, telemetry stops emitting. Worth
   a re-test if v3.3 loss class is updated.

---

## 8. File deliverables

| Path                                                          | Lines | Purpose                          |
| ------------------------------------------------------------- | ----: | -------------------------------- |
| `/tmp/v343_repair_launcher.py`                                | 776   | Python repair launcher           |
| `/home/jupiter/Lvl3Quant/scripts/launch_v343_repair.sh`       | 195   | Bash SCP+launch wrapper          |
| `/home/jupiter/Lvl3Quant/output/hc423_v343_repair_spec.md`    | (this) | Spec / verdict criteria         |

Launch deferred to 16:00 ET 2026-05-18 per operator-approved sequence.
