# HC #450 R2 — wider_cnn OOT Evaluation Verdict

**Generated:** 2026-05-20 (Jupiter, CPU)
**Question:** Does Razer `wider_cnn` (12.6M-param spatial book CNN) beat current CNN-Mamba v3.4.2 IC_10s = 0.106 on canonical OOT?

## Verdict

**LOSES — wider_cnn IC_10s = 0.0534 vs current 0.106 (gap -0.053).**

wider_cnn IC_10s = 0.0534 on 68 OOT days (15,598,294 samples) — LOSES to current CNN-Mamba v3.4.2 (0.106) by 0.0526. The 0.149 per-fold IC in the wider_cnn checkpoint was in-sample, not OOT.

**Recommended next action:** Do NOT retrain wider_cnn. Pursue alpha-staleness investigation on other architectures (PatchTST-on-trades, multi-task pressure label).

## IC Comparison (concat across all OOT days)

| Model | params (M) | window | IC_1s | IC_5s | IC_10s | IC_30s | OOT days | n_samples |
|---|---|---|---|---|---|---|---|---|
| CNN-Mamba v3.4.2 (current prod) | ~5 | sliding | 0.222 | 0.141 | **0.106** | n/a | 32 | 1.58M |
| wider_cnn (Razer, untested) | 12.6 | expanding | 0.1466 | 0.0735 | **0.0534** | 0.0339 | 68 | 15,598,294 |
| 4M_sibling (Razer checkpoints/book) | 4 | expanding | n/a | n/a | n/a | n/a | n/a | n/a |

> 4M_sibling is UNTESTABLE in this pass — only .pt weights exist (16 fold snapshots),
> no prediction NPZs. Recomputing predictions requires the original PyTorch model
> definition + GPU + matching feature pipeline. Flagged for escalation if wider_cnn proves promising.

## Side-Stratified Concat IC (wider_cnn)

| Horizon | IC (all) | IC (long preds > 0) | n_long | IC (short preds < 0) | n_short | pred std | fwd std |
|---|---|---|---|---|---|---|---|
| 1s | 0.1466 | 0.0844 | 8,576,196 | 0.0624 | 7,028,218 | 0.4779 | 0.3029 |
| 5s | 0.0735 | 0.0400 | 8,574,318 | 0.0330 | 7,027,376 | 0.4778 | 0.6629 |
| 10s | 0.0534 | 0.0281 | 8,572,055 | 0.0254 | 7,026,239 | 0.4778 | 0.9352 |
| 30s | 0.0339 | 0.0176 | 8,563,473 | 0.0173 | 7,021,221 | 0.4777 | 1.6056 |

## Methodology

- Source: `oos_predictions_wider_cnn_oot_20260311_092055.npz` (rsync'd from Razer `claude@razer`).
- Forward-mid IC: `pearson(preds_t, mid_{t+h} - mid_t)` where bar rate = 10 Hz (100ms bars), so 10s horizon = 100 bars forward.
- 234,000 bars/day × 68 OOT days = 15,912,000 raw bars before warmup.
- Warmup: skip first 100 bars (preds are zero for first ~19 bars, horizon adds more).
- Predictions are raw model outputs (no sign convention reversal applied).
- Mid is raw ES futures mid-price (ticks of 0.25, observed range 6800–7000).

## Caveats

- **Horizon ambiguity:** the wider_cnn model was trained on a single embedded horizon label.
  The checkpoint log does not annotate the target horizon. We evaluated forward-mid IC at
  1/5/10/30s. The strongest IC indicates the model's natural horizon; this is the fairest
  apples-to-apples comparison against CNN-Mamba which is multi-horizon native.
- **Expanding window:** wider_cnn was trained with expanding walk-forward (HC #0 violation).
  Even if it beats v3.4.2 here, it cannot be deployed as-is — sliding retrain mandatory.
- **OOT day overlap:** wider_cnn OOT spans 2025-12-01 to 2026-03-06. CNN-Mamba v3.4.2 canonical
  OOT is the most-recent 32 days. The wider_cnn dataset INCLUDES 2026 days but ENDS in early March,
  so some overlap exists but the windows are not identical.
- **4M sibling untested:** no predictions available; would require Neptune GPU + model code to score.

## Reproducibility

Run: `python3 scripts/hc450_research/hc450_book_eval.py`
Outputs:
- `output/hc450_book_eval/VERDICT.md` (this file)
- `output/hc450_book_eval/ic_comparison.csv`
- `output/hc450_book_eval/per_day_ic.csv`
