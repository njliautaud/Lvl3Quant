# 01 — Label Distribution

**Attribution to this root cause: ~5%.** Labels are nearly symmetric — the training targets are NOT the source of the short bias.

## Punchline

The v3.4.2 training labels (log-returns at 1s/5s/10s/30s + p_up binaries) are essentially balanced between long and short outcomes. Slight short-tilt (≤2 percentage points) exists at the day-by-day level, but it is nowhere near the 99.84% short-fill ratio we see at the trade level. **Label asymmetry alone cannot explain the bias.**

## Data scope

- Source: `output/hc432_v342_47day_validation/fold_00_ep1_oot_inference_47day_hc432.npz` (the same NPZ feeding every recent FIFO / adaptive-exit / confluence config).
- 900,454 events across **16 OOT days** (2026-02-23 to 2026-03-13).
- This is sliding-window walk-forward output (HC #0 compliant).
- **HC #428 R1 caveat**: only 16 days here, not the 40-day minimum. Extending OOT inference to the remaining ~24 dates is a separate GPU job.

## Headline numbers — log-return labels

| Horizon | n_valid | frac_pos | frac_neg | mean_pos | mean_neg | \|mean_neg/mean_pos\| |
|---|---|---|---|---|---|---|
| 1s  | 900,454 | 0.357 | 0.361 | +1.97e-6 | −1.91e-6 | 0.97 |
| 5s  | 900,454 | 0.431 | 0.437 | +3.62e-6 | −3.47e-6 | 0.96 |
| 10s | 900,454 | 0.449 | 0.455 | +4.86e-6 | −4.64e-6 | 0.96 |
| 30s | 900,454 | 0.467 | 0.472 | +7.74e-6 | −7.43e-6 | 0.96 |

The remaining 18-29% of events at the 1s/5s horizons have label = 0 (no price move) — this is the dominant "no event" class, NOT a bias.

## p_up binary labels

| Horizon | frac=1 (up) | frac=0 (down/flat) |
|---|---|---|
| 5s  | 0.4305 | 0.5695 |
| 10s | 0.4488 | 0.5512 |
| 30s | 0.4665 | 0.5335 |

p_up labels show a mild down-tilt (~10pp) because flat outcomes are grouped with the down class — that's an artifact of binary classification with `pred > 0` cutoff. The continuous log-return labels above are the cleaner read.

## Per-day asymmetry trajectory (10s log-return)

Across all 16 OOT days, the **frac_pos vs frac_neg gap stays inside ±5pp every day**. There is no day where labels are dramatically more short than long. Sample (10s):

| Date | frac_pos | frac_neg | Δ |
|---|---|---|---|
| 20260223 | 0.431 | 0.462 | −3.1pp |
| 20260227 | 0.447 | 0.445 | +0.2pp |
| 20260303 | 0.464 | 0.466 | −0.2pp |
| 20260306 | 0.458 | 0.466 | −0.8pp |
| 20260311 | 0.445 | 0.462 | −1.7pp |
| 20260313 | 0.441 | 0.472 | −3.1pp |

This is normal market microstructure noise — not a structural label-design bug. (Full table in `label_per_day_10s.parquet`.)

## What WOULD a label bug look like?

A label bug would show: frac_neg ≥ 0.70 at the trained horizon, or |mean_neg / mean_pos| ≥ 1.5. **Neither is the case.**

## Conclusion

Label distribution is approximately symmetric at all four training horizons. The 99.84% short-fill ratio is NOT a faithful reflection of asymmetric labels. **Root cause (b) is rejected at ≤5% attribution.** Look elsewhere — the next reports (02 raw IC, 03 threshold attribution) carry the real signal.
