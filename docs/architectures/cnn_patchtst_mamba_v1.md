# cnn_patchtst_mamba_v1 — Canonical 4-Branch Fusion Architecture

**Locked in DIRECTIVES.md 19:21 ET, 19:32 ET, 21:08 ET (5-branch interim), and 22:05 ET (BOOK CNN DROPPED — 4 branches final) — Apr 27, 2026.**
User architectural intent: *"i assumed mamba would be the best backbone to combine signals from book cnn event temporal cnn and patchtst.. since it will take those inputs ASWELL as the typical smartv3 or v4 dataset as input aswell to output our ultimate outputs for signals to trade with"* + *"the same way the cnn mamba is wired together we could make it the cnn patchtst mamba"* + *"ok then drop the book cnn"* (22:05 ET).

> **HISTORICAL NOTE (22:05 ET):** Book CNN was originally branch #1 of 5 (21:08 ET diagram). It was dropped after the standalone Book CNN h128 on Feb-Mar 2026 dates hit IC_10s ≈ +0.027 — well below the cnn_mamba_v2 baseline 0.132 and below the 0.05 floor. The freed param budget reallocates to wider Event CNN OR deeper Mamba (N=6 layers) OR longer PatchTST window. Final fusion is **4 branches**.

## Theory of Operation

Mamba is the **temporal integrator**. Each event step gets a fused feature vector composed of FOUR per-step views of the same event, and Mamba's selective state-space dynamics learn which view to weight when. This preserves Mamba's strengths (O(L) over long event streams, time-delta-aware decay, native gating) while giving it richer per-step inputs than the existing cnn_mamba_v2 has.

| Branch | Role | Strength | Why It Helps Mamba |
|---|---|---|---|
| **Event Temporal CNN** | Local microstructure patterns (3-5 step kernels) | Spread dynamics, trade clustering | Pre-extracted local patterns reduce burden on Mamba state |
| **PatchTST** | Long-window directional context (patch=25, ~1000-event window) | Bidirectional attention within patches | Provides "what's coming in this patch" signal that causal Mamba can't recover alone |
| **smart_v4 raw** | Engineered features (vol, ofi, microprice, spread, ...) per event | Domain-knowledge alpha | Mamba sees the raw indicators directly, not just neural-encoded views |
| **VOL stream** | vol_lgbm_v3 per-event predictions (vol_10s/30s/60s) | Regime-aware sizing/gating signal | Mamba sees forward-vol hint without re-deriving from raw events |

## Architecture Diagram

```
                            events (B, L, F_raw, smart_v4)
                                       │
              ┌────────────────────┬───┴────────────┬─────────────────┐
              ▼                    ▼                ▼                 ▼
          Event Temporal       PatchTST         smart_v4 raw      VOL stream
          CNN front-end        front-end        passthrough       (vol_lgbm_v3
          (3 layers, k=5,      (patch=25,       / thin MLP         per-event preds:
           GELU + residual)    TransformerX2,   per-step)          vol_10s/30s/60s)
                               ALiBi attn)
              │                    │                │                 │
          per-step             per-patch         per-step          per-step
          (B, L, c_evt)        (B, N_p, d_pt)    (B, L, c_v4)      (B, L, 3)
                                   │
                              upsample patches
                              → (B, L, c_pt)
                              [repeat-each-patch
                               patch_size times]
                                   │
              └────────────────────┴────────────────┴─────────────────┘
                                       │
              CONCAT per-step → (B, L, c_evt + c_pt + c_v4 + 3)
                                       │
                          Linear → LayerNorm
                                       │
                          (B, L, d_model)            d_model = 128 or 192
                                       │
                          MambaBlock × N             N = 3 or 4 (or 6 with freed-up Book CNN budget)
                          (selective state,
                           time-delta aware,
                           pre-norm residual)
                                       │
                          final LayerNorm
                                       │
                          last-step pooling → (B, d_model)
                                       │
       ┌─────────────────┬──────────┬──┴──────────┬─────────────────┐
       ▼                 ▼          ▼             ▼                 ▼
   Δ1s, Δ5s,         MFE head      MAE head    realized-vol     [optional]
   Δ10s head         (max fav.     (max adv.   head
   (3 targets)       excursion     excursion   (target sanity-
                     quantiles)    quantiles)  check vs vol_lgbm)
                                       │
                          → trade signals
                            (direction × magnitude × adaptive TP/SL bands)
```

## Output Heads (Per User Directive 19:32 ET)

Final last-step head emits **6 targets** (extensible to 7 if vol head added):

1. **Δ1s** — predicted return at 1 second horizon
2. **Δ5s** — predicted return at 5 second horizon
3. **Δ10s** — predicted return at 10 second horizon (PRIMARY metric for IC reporting)
4. **MFE** — predicted maximum favorable excursion (next 60s window)
5. **MAE** — predicted maximum adverse excursion (next 60s window)
6. **(optional) realized vol** — predicted vol band for the next 10s

**MFE/MAE enable adaptive TP/SL execution** — instead of a fixed 8-tick TP / 15-tick SL, each trade's exit thresholds are conditioned on the model's expected MFE/MAE for THAT specific entry. User explicit: *"that could be helpful for adaptive tp and sl"*.

## Per-Branch Implementation Notes

### Event Temporal CNN front-end
- Input: raw event features (B, L, F_raw)
- Architecture: identical to existing `train_cnn_mamba.py` CNN front-end (3 Conv1d layers, kernel=5, GELU, residual)
- Embedding dim: c_evt ≈ 64

### PatchTST front-end
- Input: events + book_dynamics features
- Patch: size=25, stride=25 (non-overlapping for v1 — overlapping is v2 stretch)
- Transformer: 2 blocks, 4 heads, ALiBi attention
- Output: per-patch (B, N_patches=L/25, d_pt=64)
- **Upsample to per-step**: each patch token replicated `patch_size` times → (B, L, c_pt=64)
- Stretch: try learned upsampling (deconv) instead of repeat
- Source: existing `train_event_patchtst.py` PatchTST class — extract per-patch outputs before mean-pool

### smart_v4 raw passthrough
- Input: precomputed engineered features per event (29 dim — confirmed via `precompute_features_smart_v4.py`)
- Architecture: identity OR thin Linear+LayerNorm+GELU (project to common scale)
- Source: existing `precompute_features_smart_v4.py` outputs (29 features: 6 raw + 9 v1-derived + 7 v2 + 3 v3 + 4 v4 incl. time_of_day_sin/cos, vol_regime, trade_arrival_rate)
- Embedding dim: c_v4 ≈ 29-32

### VOL stream (per directive 21:08 ET)
- Input: vol_lgbm_v3 per-event predictions, 3 channels (vol_10s, vol_30s, vol_60s)
- Architecture: per-fold normalization (train-only mean/std) → optional thin LayerNorm → concat at per-step concat point. NO trainable params for v1 (PHASE A — frozen-input). PHASE B (later) trains a small VolNet jointly if PHASE A shows lift.
- Source: `train_vol_lgbm_v3.py` outputs (`vol_v3_{date}_predictions.npz`)
- **LEAKAGE NOTE:** vol_lgbm_v3 is per-day sliding-window causal (60d strictly prior). Currently only 10 OOT dates have predictions (Feb 23 - Mar 5, 2026). For deep model TRAINING events, vol predictions must be regenerated across the full training-window date range, each from a 60d-prior model — leak-free by construction. ETA ~30 vol models × ~8 min on Jupiter CPU ≈ 4 h (weekly cadence).
- Embedding dim: c_vol = 3 (frozen-input)

### Mamba backbone
- Identical to existing `train_cnn_mamba.py` — d_state=32-64, n_layers=3-4, dt_rank=16, d_conv=4
- Time-delta conditioning from `time_delta_log` (raw event feature index 0)
- Pre-norm residual blocks
- USE_CUDA_MAMBA flag for fast selective scan when on GPU

## Total Parameter Budget (revised after Book CNN drop, 22:05 ET)
- Branch encoders: ~100K-250K combined (Event CNN ~50K, PatchTST ~200K, smart_v4 passthrough ~5K, VOL stream ~0 — frozen)
- Concat → Linear → d_model: ~50K
- Mamba × 3-6: ~200K-600K (Book CNN's freed ~50K reallocates here — try N=6 layers)
- Heads (6 targets × small MLPs): ~50K
- **Total target: 400K - 1.5M parameters** (Book CNN drop saved ~50K param + ~30 GFLOPs/forward)

## Training Protocol
- **MANDATORY**: same WF folds as cnn_mamba_v2 (60-day expanding window, walk-forward)
- **MANDATORY**: same OOT dates so apples-to-apples vs cnn_mamba_v2 baseline (IC_10s = 0.132)
- **MANDATORY**: MLflow logging from epoch 0
- **MANDATORY**: launch_with_watchdog.sh wrapper
- **MANDATORY**: save .pt weights AND .npz preds (all 6 target heads) per fold
- Warm-start branch encoders from existing weights where possible:
  - Event CNN ← `cnn_mamba_v2` CNN weights
  - PatchTST ← standalone PatchTST weights
  - Mamba ← `cnn_mamba_v2` Mamba weights
  - VOL stream ← frozen (no weights, just consumed inputs)
- Compute: Neptune RTX 3090 (24 GB VRAM, 31 GiB RAM — peak RSS ≤ 22 GiB per Apr 27 hardware-fact correction)

## Success Criteria
- **PRIMARY**: 10s concat IC > 0.132 (beats cnn_mamba_v2 baseline)
- **SECONDARY**: per-tier DA at top 0.5/1/5/10% — must NOT invert at any tier × horizon cell
- **TERTIARY**: MFE/MAE head Spearman correlation > 0.10 with realized MFE/MAE (validates adaptive TP/SL is feasible)
- **EXECUTION**: Sortino on tp9_z2.3 + Q5 confluence > current baseline (Sortino 12.25 from cnn_mamba_v2) — user wants the model SHIP something better than what we have, otherwise stay on cnn_mamba_v2

## Open Empirical Questions (Gating Build)
1. **Mamba vs dilated CNN backbone (per directive 22:05 ET point 5):** rerun `train_event_cnn_1d.py` (raw 6-channel + dilated CNN) on smart_v3_mar fold layout. If EventCNN1D ≥ cnn_mamba_v2 on Mar-Apr concat IC, **swap the backbone** of this fusion to dilated CNN before building. If less, Mamba stays. **STATUS: in-flight on Neptune** (launched 23:10 ET 2026-04-27, 11 folds × 60d sliding train × 1 OOT day each, matching cnn_mamba_v2_smart_v3_mar exactly).
2. **Vol regen scope:** confirm cnn_patchtst_mamba_v1 training-window date range and run weekly-cadence regen on Jupiter CPU before launching the deep model.

## Status
- 19:32 ET Apr 27: Architecture spec saved (this doc).
- 21:08 ET Apr 27: Extended to 5 branches (vol added).
- 22:05 ET Apr 27: Book CNN dropped — back to 4 branches (this rev).
- 23:10 ET Apr 27: EventCNN1D backbone bakeoff launched on Neptune (gates the build).
- Build of `train_cnn_patchtst_mamba.py` is GATED on (a) EventCNN1D backbone-bakeoff result, (b) vol regen across full training window complete.
