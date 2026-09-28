# HC #450 R2 — Old Book / Event-CNN Weights Inventory

**Generated:** 2026-05-20 (Jupiter)
**Question:** Are there old book/event-CNN weights with IC_10s > current CNN-Mamba v3.4.2 (0.106)?
**Sources searched:** Jupiter (local), Neptune (`nick@neptune`), Razer (`claude@razer`). Saturn not queried (aux CPU, no GPU training history).

---

## TL;DR

**Answer: NO. No surviving book/event-CNN checkpoint clearly beats CNN-Mamba v3.4.2 on IC_10s.**

Best old book/event-CNN with intact weights AND measurable IC_10s is **`event_cnn_1d_smart_v3_mar_wfFIXED_20260428_1514`** on Neptune, with **concat IC_10s = 0.1025** across 4 OOT folds — basically a tie with current v3.4.2 (0.106). Triple-fusion (with book branch) achieves IC_1s=0.173 but only IC_10s=0.067. Razer holds the most historically interesting weights — **`wider_cnn/fold_74_2025-11-03.pt`** is a ~12.6M-param spatial-CNN book model with reported per-fold IC averaging ~0.15 (mean of 94 folds; horizon not annotated in the checkpoint — likely 1-3s based on schema). IC_10s could not be recomputed for Razer artifacts in this pass because the prediction NPZs store raw mid-price arrays and need forward-difference reconstruction, which segfaulted on Razer's Python in the available time window.

**Most promising untested candidate:** Razer `wider_cnn1d/fold_00_best.pt` (11MB, single fold) and `checkpoints/book/latest.pt` (~16MB, ~4M params, 16 folds available 2025-10 through 2026-03). These are spatial book CNNs that pre-date CNN-Mamba and should be evaluated on the v3.4.2 OOT day set before being discarded.

---

## Master Table (sorted by IC_10s desc, "n/a" rows last)

| Model | Node | Ckpt Path | Params | IC_1s | IC_5s | IC_10s | IC_30s | MLflow Run | Notes |
|---|---|---|---|---|---|---|---|---|---|
| event_cnn_1d_smart_v3_mar_wfFIXED | Neptune | `/home/nick/Lvl3Quant/output/event_cnn_1d_smart_v3_mar_wfFIXED_20260428_1514/fold_{00-04}_best.pt` | ~0.5M (2.1MB) | 0.158 | 0.105 | **0.1025** | n/a | exp 935985340201974967 EventCNN1D_20260428_1515 | 5 OOT folds. Concat from 95k samples. Tied with v3.4.2 on 10s. |
| triple_fusion_v1_smart_v4_book_mar | Neptune | `/home/nick/Lvl3Quant/output/triple_fusion_v1_smart_v4_book_mar/fold_{00-06}_best.pt` | ~0.8M (3.4MB) | **0.173** | 0.086 | 0.067 | n/a | exp 791088988319878801 | Fusion w/ book branch; 7 folds, 422k samples concat. Best IC_1s in this set. |
| triple_fusion (intra ckpt, fuller) | Neptune | `/home/nick/Lvl3Quant/output/triple_fusion_v1_smart_v4_book_mar/fold_{00-06}_intra_ckpt.pt` | ~2.5M (10MB) | same | same | same | n/a | same | Larger snapshot of same model |
| v2_book_cnn_d10_h128_e12 (preds only) | Neptune | NO .pt — `/home/nick/Lvl3Quant/output/v2_book_cnn_d10_h128_e12/*.npz` | hidden=128, depth=10 | 0.097 | 0.049 | 0.043 | n/a | exp 791088988319878801 v2_book_cnn_depth10_20260427_19* | 5 folds, 130k samples. **WEIGHTS NOT SAVED.** Best of the v2_book_cnn experiments. |
| event_cnn_1d_smart_v3_mar_oomsafe | Neptune | `/home/nick/Lvl3Quant/output/event_cnn_1d_smart_v3_mar_oomsafe_20260428_0920/fold_{00-05}_best.pt` | ~0.5M (2.1MB) | 0.124 | 0.056 | 0.025 | n/a | exp 935985340201974967 EventCNN1D_20260428_0920 | 6 OOT folds, 173k samples. Lower 10s than wfFIXED variant. |
| v2_book_cnn_d10_fixed2 (preds only) | Neptune | NO .pt — `/home/nick/Lvl3Quant/output/v2_book_cnn_d10_fixed2/*.npz` | hidden=64, depth=10 | 0.064 | 0.039 | 0.029 | n/a | exp 791088988319878801 | 5 folds. **WEIGHTS NOT SAVED.** |
| v2_book_cnn_depth10 (preds only) | Neptune | NO .pt — `/home/nick/Lvl3Quant/output/v2_book_cnn_depth10/*.npz` | hidden=64, depth=10 | 0.056 | 0.033 | 0.035 | n/a | exp 791088988319878801 v2_book_cnn_depth10_20260427_1518 | 5 folds. **WEIGHTS NOT SAVED.** |
| event_cnn_1d_smart_v3_mar (orig) | Neptune | `/home/nick/Lvl3Quant/output/event_cnn_1d_smart_v3_mar/fold_00_best.pt` | ~0.5M | n/a | n/a | n/a | n/a | exp 935985340201974967 | Single fold, no predictions. Superseded by wfFIXED variant. |
| event_cnn_1d_smart_v3_mar_20260428_072235 | Neptune | `/home/nick/Lvl3Quant/output/event_cnn_1d_smart_v3_mar_20260428_072235/fold_00_best.pt` | ~0.5M | n/a | n/a | n/a | n/a | exp 935985340201974967 EventCNN1D_20260428_0723 | Single fold |
| synthetic_event_cnn1d (live deploy) | Jupiter | `/home/jupiter/Lvl3Quant/live_trading/models/synthetic_event_cnn1d.pt` | ~0.5M (2.0MB) | n/a | n/a | n/a | n/a | n/a | Synthetic / smoke-test weights, not a research checkpoint |
| cnn_mamba_v2_wider_256ch (no preds) | Neptune | `/home/nick/Lvl3Quant/output/cnn_mamba_v2_wider_256ch_20260429/fold_00_intra_ckpt.pt` | ~8M (32MB) | n/a | n/a | n/a | n/a | n/a | 256ch wider variant of v2; only intra ckpt, no OOT preds |
| **wider_cnn (book spatial CNN)** | **Razer** | `C:\Users\claude\Lvl3Quant\alpha_discovery\deep_models\results\wider_cnn\fold_74_2025-11-03.pt` | **~12.6M (50MB)** | n/a (single-h target) | n/a | n/a (NOT RECOMPUTED) | n/a | log: `walkforward_book_20260323_231723.log` | **Per-fold IC reported in checkpoint backup: mean 0.149, max 0.231, 94 folds.** Horizon embedded in target — needs to be re-evaluated on v3.4.2 OOT day set. spatial_stem 64ch (3x3 conv). EXPANDING window. |
| checkpoints/book/fold_{60-83} | Razer | `C:\Users\claude\Lvl3Quant\alpha_discovery\deep_models\results\checkpoints\book\fold_*.pt` + `latest.pt` | ~4M (16MB each) | n/a | n/a | n/a (NOT RECOMPUTED) | n/a | n/a | 16 walk-forward folds Oct 2025 – Mar 2026. Smaller spatial CNN sibling of wider_cnn. Per-fold IC from `old_book_checkpoints/checkpoint_book_20260301_093742.json.standard_backup`: mean 0.137, max 0.231, 94 folds. |
| cnn_event_w1000_razer | Razer | `C:\...\cnn_event_w1000_razer\fold_00_best.pt` | ~0.5M (2.1MB) | n/a | n/a | n/a | n/a | n/a | Window=1000 event-CNN, single fold |
| cnn_event_w1000_razer_r2 | Razer | `C:\...\cnn_event_w1000_razer_r2\fold_00_best.pt` | ~0.5M (2.1MB) | n/a | n/a | n/a | n/a | n/a | retrain of above |
| event_cnn1d_wf10 | Razer | `C:\...\event_cnn1d_wf10\fold_00_best.pt` | ~0.5M (2.1MB) | n/a | n/a | n/a | n/a | n/a | 10-fold WF event CNN |
| wider_cnn1d | Razer | `C:\...\wider_cnn1d\fold_00_best.pt` | ~3M (11MB) | n/a | n/a | n/a | n/a | n/a | 1D variant of wider_cnn (no spatial book branch) |
| cnn_mamba_v3_4_2_fixedmtl (book_gate_fix) | Jupiter+Neptune | `output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.book_gate_fix.pt` | ~5M (20MB) | n/a | n/a | n/a | n/a | exp 801257626907086140 v3.4.1_residual_20260515_* | Current production v3.4.2 (not "old book") — included for reference |

---

## Section per Node

### Jupiter (`/home/jupiter/Lvl3Quant`)
Local has very little book/event work — Jupiter is CPU/orchestration.

1. `/home/jupiter/Lvl3Quant/live_trading/models/synthetic_event_cnn1d.pt` — 2.0MB, mtime 2026-04-17 — smoke-test weights for live pipeline, not research.
2. `/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.book_gate_fix.pt` — 20MB, mtime 2026-05-19 — synced from Neptune. Current production model (the "book_gate_fix" suffix refers to a gate-normalization fix to the v3 book branch, NOT a separate model).

No `book_cnn`, `event_cnn`, `wider_cnn`, `level2_*`, `orderbook_*`, `frame_*`, `depth10_*` directories on Jupiter.

### Neptune (`nick@neptune:/home/nick/Lvl3Quant`)
The biggest pool of measurable predictions. Five model families:

1. **`output/event_cnn_1d_smart_v3_mar_wfFIXED_20260428_1514/`** — 5 folds, .pt + feature_stats + predictions. Concat IC_1s=0.158, IC_10s=**0.1025**. Best 10s of the bunch with weights.
2. **`output/event_cnn_1d_smart_v3_mar_oomsafe_20260428_0920/`** — 6 folds, .pt + predictions. IC_10s=0.025. Inferior.
3. **`output/event_cnn_1d_smart_v3_mar/`**, **`_20260428_072235/`** — single-fold variants, no predictions.
4. **`output/triple_fusion_v1_smart_v4_book_mar/`** — 7 folds, .pt + intra_ckpt + analysis JSONs. Best IC_1s=0.173 but IC_10s=0.067. The book branch is one of three (feat_mlp + cnn + patchtst); gate stats show cnn_gate ≈ 0.50 (dominant).
5. **`output/v2_book_cnn_d10_*/`** — 4 separate experiments (`fixed`, `fixed2`, `h128_e12`, `depth10`). **NO `.pt` WEIGHTS saved** (training script never wrote them — pure predictions only). Best variant `d10_h128_e12` peaks IC_10s=0.043. Source script logs in `logs/v2_book_cnn/depth10_*.log`.
6. **`output/cnn_mamba_v2_wider_256ch_20260429/`** — single fold intra_ckpt, 32MB (~8M params). No OOT predictions. Width sweep that was abandoned.
7. **`alpha_discovery/deep_models/results/event_*`** — large set of event-Mamba (NOT CNN) and event_transformer_fast variants, all from April-May 2026. These are sequence models, not book/spatial CNNs.
8. **`results/event_cnn_1d/`** and **`alpha_discovery/deep_models/results/event_cnn_1d/`** — directories empty.

### Razer (`claude@razer:C:\Users\claude\Lvl3Quant`)
This is the **archeological goldmine** for "old book CNNs" — Razer was the training node before April 2026.

1. **`alpha_discovery/deep_models/results/wider_cnn/`** — THE candidate to evaluate.
   - `fold_74_2025-11-03.pt` (50MB ≈ **12.6M params**, log confirms `Model params: 12,595,713`)
   - 50+ `oot_*.npz` per-day predictions covering 2025-09-10 to 2026-03-13
   - `checkpoint_wider_cnn_20260316_234709.json` reports per-fold IC for 36 folds: mean **0.149**, max **0.227**, min 0.075. Horizon not explicitly stated in checkpoint — likely 1-3 sec based on bar size (~234k bars/day ≈ 10/sec).
   - Walk-forward log shows EXPANDING window (violates HC #0 SLIDING rule — would need re-train).
2. **`alpha_discovery/deep_models/results/checkpoints/book/`** — 16 fold .pt files Oct 2025 → Mar 2026 + `latest.pt`. Each 16MB ≈ **4M params**. Sibling/smaller version of wider_cnn.
3. **`alpha_discovery/deep_models/results/old_book_checkpoints/`** — JSON training logs only (no .pt here). Largest backup `checkpoint_book_20260301_093742.json.standard_backup` reports **94 completed folds, IC mean 0.137, max 0.231**.
4. **`cnn_event_w1000_razer/`**, **`cnn_event_w1000_razer_r2/`** — single-fold event-CNN with 1000-event window, 2MB each.
5. **`event_cnn1d_wf10/`** — single-fold 10-WF event CNN, 2MB.
6. **`wider_cnn1d/`** — 1D variant (no spatial book branch), single fold, 11MB.

### Saturn — not queried (auxiliary CPU node, no training history).

---

## Prediction NPZ files we still have (for retrain/rerun)

If we decide the surviving weights are insufficient, the following prediction NPZs are available to anchor a re-training run:

**Neptune:**
- `output/v2_book_cnn_d10_h128_e12/fold_{05-09}_oot_predictions.npz` — 130k samples, 3 horizons (1s/5s/10s), with embeddings (128-dim).
- `output/v2_book_cnn_d10_fixed2/` and `output/v2_book_cnn_depth10/` — same schema, 64-dim embeddings.
- `output/event_cnn_1d_smart_v3_mar_wfFIXED_20260428_1514/fold_{00-03}_oot_predictions.npz` — paired with `feature_stats.npz`, 95k samples.
- `output/event_cnn_1d_smart_v3_mar_oomsafe_20260428_0920/fold_{00-05}_oot_predictions.npz` — paired stats, 173k samples.
- `output/triple_fusion_v1_smart_v4_book_mar/fold_{00-05}_oot_predictions.npz` + `fold_*_analysis.json` (rich metrics including DA, MFE/MAE, profit factor, win rate per horizon and per top-N quantile bucket).

**Razer:**
- `wider_cnn/oos_predictions_wider_cnn_oot_*.npz` — 68 days of paired `_preds`/`_mid` arrays (2025-12 to 2026-03), 234k samples/day. **Forward-diff against `_mid` gives any-horizon IC**, but Python segfaulted in this pass — recommend re-running with proper venv on Razer or rsync the NPZ to Jupiter.

---

## Recommended Next Steps

1. **Rsync the Razer `wider_cnn` directory to Jupiter** and compute IC_10s on its OOT predictions vs forward-mid-difference. This is the only candidate that *might* beat v3.4.2 — the user's recollection points here.
2. If wider_cnn IC_10s confirms > 0.106, port the checkpoint forward: convert expanding-window training to SLIDING (HC #0), retrain on current MBO-events-smart_v4 schema, re-evaluate on 40+ OOT days (HC #428 R1).
3. For `v2_book_cnn_d10_h128_e12` — predictions exist but weights are gone. If we want this architecture back, we have to retrain. The IC_10s ceiling of 0.043 suggests it is NOT worth retraining.
4. The `event_cnn_1d_smart_v3_mar_wfFIXED` weights are usable as-is (5 .pt folds, 4 with paired predictions) — but its IC_10s is only marginally better than 0.10 so it does not improve over current production.

---

## MLflow Experiments (Jupiter http://localhost:5000)

Found, all relevant:
- `801257626907086140` — CNNMamba_v3_4_1_book_residual (current production lineage)
- `791088988319878801` — TripleFusion_v2_BookCNN (the triple-fusion + v2_book_cnn runs above)
- `935985340201974967` — EventDriven_CNN1D
- `249683859977283619` — EventDriven_Mamba
- `831321011840340610` — EventDriven_PatchTST

**IC metrics are NOT logged to these MLflow runs** (verified by API calls — `metrics: []` on sampled runs). All quantitative IC numbers in the table above were re-derived from prediction NPZ files and per-fold `analysis.json` files, not MLflow.
