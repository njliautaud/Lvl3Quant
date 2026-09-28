# Fusion Backbone Bake-Off — 3 Variants

**Locked in DIRECTIVES.md 19:32 ET (Apr 27, 2026); revised 22:05 ET to drop Book CNN per user directive (final 4 branches: Event CNN + PatchTST + smart_v4 raw + VOL stream).**
User explicit: *"save the experiments for comparison the 3 versions u layed out."* + *"ok then drop the book cnn"* (22:05 ET).

## Goal

Empirically determine whether **Mamba-as-backbone** is actually adding value vs simpler alternatives. User's intuition is that Mamba should be the temporal integrator (per `docs/architectures/cnn_patchtst_mamba_v1.md`); user also said *"but i could be wrong"* — this bake-off settles it with data, not theory.

## Pre-Bakeoff Empirical Question (NEW, 22:05 ET)
Before running the 3-variant bake-off, a separate question gates the build: **is Mamba even the right backbone vs a dirt-simple dilated CNN on the era that matters (Mar-Apr 2026)?**
- Test: `train_event_cnn_1d.py` raw 6-channel + dilated CNN, same fold layout as cnn_mamba_v2_smart_v3_mar (11 folds × 60d sliding train × 1 OOT day each).
- **STATUS:** launched on Neptune 23:10 ET 2026-04-27 → `/home/nick/Lvl3Quant/output/event_cnn_1d_smart_v3_mar/`.
- **If EventCNN1D ≥ cnn_mamba_v2 IC_10s on Mar-Apr era:** swap the backbone in this bake-off — v1_cnn replaces v1_mamba (4-branch concat → dilated CNN backbone → last-step head). v2 (no backbone) and v3 (xattn) unchanged.
- **If EventCNN1D < cnn_mamba_v2:** proceed with the original v1_mamba spec below.

## The 3 Variants

All three share **identical branch encoders** (Event CNN + PatchTST + smart_v4 raw + VOL stream — Book CNN dropped 22:05 ET, weights tied across variants) and are trained on **identical WF folds + identical OOT dates** so the comparison is apples-to-apples. Only the integration layer differs.

### v1_mamba — Per-Step Mamba Backbone (USER'S WIRING)

```
4 branches (Event CNN + PatchTST + smart_v4 + VOL, per-step) → concat → Linear → (B, L, d_model) → MambaBlock × N → last-step head
```

- **Hypothesis**: Mamba's selective state space provides per-step gating across the 4 branches; O(L) lets us see every event; time-delta-aware decay handles bursty MBO timing
- **Cost**: highest — full Mamba over (B, L, d_model)
- **Build risk**: low — wiring proven in `train_cnn_mamba.py`
- **Source canonical**: `docs/architectures/cnn_patchtst_mamba_v1.md`
- **Script (planned)**: `alpha_discovery/deep_models/train_cnn_patchtst_mamba.py`

### v2_late_mlp — No Backbone, Late Embedding Fusion

```
4 branches (Event CNN + PatchTST + smart_v4 + VOL) → each pooled to (B, d_branch) → concat (B, 4*d_branch) → MLP → 6 heads
```

- **Hypothesis**: Each branch already does its own temporal modeling internally; a backbone on top is just window-dressing. Concat the 4 final embeddings and let a small MLP do the routing.
- **Cost**: lowest — no Mamba, just MLP at the end
- **Build risk**: low — this is essentially what `train_triple_fusion.py` does today (which we identified as wired wrong only because Mamba ran on length-1 token; if Mamba isn't doing anything useful there, this should match its IC)
- **Plays the role of**: sanity check / null hypothesis. If v1 ≈ v2, Mamba isn't adding value and we ship v2 (simpler, faster).
- **Script (planned)**: `alpha_discovery/deep_models/train_cnn_patchtst_late_mlp.py`

### v3_xattn — Cross-Attention Transformer Between Branches (FALLBACK)

```
4 branches (per-step) → cross-attention between (Event ↔ PatchTST ↔ smart_v4 ↔ VOL)
                     → fused per-step (B, L, d_model) → small transformer/MLP head
```

- **Hypothesis**: If Mamba's single-vector state is the bottleneck for tracking 4 disparate signal flavors, explicit cross-attention routes information per-step without a state bottleneck. More expressive than concat-then-Mamba.
- **Cost**: highest — O(L²) attention or constrained linear-attn variant
- **Build risk**: HIGH — new wiring, easy to overfit on small data, hardest to train
- **Plays the role of**: fallback IF v1 fails AND v2 fails (i.e., neither backbone-vs-no-backbone configuration beats cnn_mamba_v2)
- **Only built after v1 + v2 results are in.**
- **Script (planned)**: `alpha_discovery/deep_models/train_cnn_patchtst_xattn.py`

## Decision Matrix

| Outcome | Decision |
|---|---|
| v1 IC_10s **>** v2 IC_10s by ≥ 0.01 | Ship v1 (Mamba backbone validated; user's intuition empirically right) |
| v1 ≈ v2 (within 0.005) | Ship v2 (simpler, faster, same alpha; user's intuition was right in spirit but Mamba's marginal contribution is noise) |
| v2 **>** v1 by ≥ 0.01 | Investigate why v1 underperforms — likely Mamba state is overfitting on small data or the per-step concat is too high-dim. Ship v2. |
| Both v1 + v2 fail to beat cnn_mamba_v2 (IC_10s < 0.132) | Build v3_xattn. Architecture isn't the problem — branch composition is. Ablate branches one at a time. |
| v3_xattn also fails | Drop the 4-branch idea. Go back to cnn_mamba_v2 single-stream and look elsewhere for alpha (different feature engineering, different label horizon, different loss). |

## Shared Training Protocol (Apples-to-Apples)

- **WF folds**: identical to cnn_mamba_v2 (60-day expanding window, walk-forward). Lock the fold splits FIRST and reuse.
- **OOT dates**: same as current cnn_mamba_v2 OOT (Mar 2-Mar 13 once Mar 13+ data lands; until then, use the existing 9-day overlap)
- **Branch encoder weights**: identical initialization across all 3 variants (warm-start from same checkpoints)
- **Optimizer**: AdamW, lr=2e-4, weight decay=1e-2, cosine schedule
- **Batch size**: 128, accumulate to effective 256 if VRAM tight
- **Epochs per fold**: 4-6 (cnn_mamba_v2 standard)
- **Loss**: multi-horizon Huber on (Δ1s, Δ5s, Δ10s) + quantile loss on (MFE, MAE)
- **MLflow experiment names**: `FusionBakeoff_v1_mamba`, `FusionBakeoff_v2_late_mlp`, `FusionBakeoff_v3_xattn`
- **Storage**: separate output dirs (`/home/jupiter/Lvl3Quant/output/fusion_bakeoff_v{1,2,3}/`) — NEVER share dirs (caused fold 76+ leakage previously)

## Comparison Reporting

For each variant report:

1. Per-fold IC at 1s, 5s, 10s
2. Concat IC across all OOT folds
3. Per-tier DA at top 0.5/1/5/10% × each horizon
4. MFE/MAE head Spearman vs realized
5. Execution Sortino with tp9_z2.3 baseline + Q5 confluence (using `sortino_bootstrap_candidates.py` after no-leakage audit)
6. Total params + GPU-hours per fold (cost factor)
7. **Single comparison table** — all 3 variants side-by-side on the same metrics

## Status

- 19:32 ET Apr 27: Bake-off spec saved (this doc). v1 build pending. v2 build queued behind v1. v3 conditional on v1+v2 results.
- 22:05 ET Apr 27: Book CNN dropped from shared branch list — final 4 branches = Event CNN + PatchTST + smart_v4 + VOL stream. Pre-bakeoff backbone question added (EventCNN1D vs cnn_mamba_v2).
- 23:10 ET Apr 27: EventCNN1D backbone-bakeoff launched on Neptune. Bake-off variants gated on this result.
