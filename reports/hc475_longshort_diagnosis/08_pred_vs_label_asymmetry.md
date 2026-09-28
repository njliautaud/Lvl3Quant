# HC #475 R1 Q4 — Pred vs Label Asymmetry

Generated: 2026-05-21T17:21:24
Source: `/home/jupiter/Lvl3Quant/output/hc432_v342_47day_validation/fold_00_ep1_oot_inference_47day_hc432.npz`

Reading: if `gap` = `pred_pos_share - label_pos_share` is within ±0.10, the model is FAITHFULLY
reflecting the label distribution → the labels are the lever to pull (HC #475 R4 R-bullet 2).
If `|gap| > 0.10`, the model is amplifying or dampening label asymmetry → representational defect,
requires loss-function / architecture-level fix.

| Head | pred_pos% | label_pos% | gap | verdict |
|------|-----------|------------|-----|---------|
| `log_ret_1s` | 0.687 | 0.356 | +0.331 | MODEL-DEFECT |
| `log_ret_5s` | 0.687 | 0.431 | +0.256 | MODEL-DEFECT |
| `log_ret_10s` | 0.685 | 0.450 | +0.235 | MODEL-DEFECT |
| `log_ret_30s` | 0.274 | 0.469 | -0.194 | MODEL-DEFECT |
| `log_ret_60s` | 0.004 | 0.000 | +0.004 | FAITHFUL |
| `log_ret_5min` | 0.701 | 0.000 | +0.701 | MODEL-DEFECT |
| `p_up_5s` | 0.024 | 0.431 | -0.406 | MODEL-DEFECT |
| `p_up_10s` | 0.474 | 0.449 | +0.025 | FAITHFUL |
| `p_up_30s` | 0.760 | 0.466 | +0.293 | MODEL-DEFECT |
| `p_up_60s` | 0.991 | 0.000 | +0.991 | MODEL-DEFECT |
| `log_ret_10s_q10` | 0.000 | 0.450 | -0.450 | MODEL-DEFECT |
| `log_ret_10s_q50` | 0.755 | 0.450 | +0.306 | MODEL-DEFECT |
| `log_ret_10s_q90` | 1.000 | 0.450 | +0.550 | MODEL-DEFECT |
| `log_ret_30s_q10` | 0.000 | 0.469 | -0.469 | MODEL-DEFECT |
| `log_ret_30s_q50` | 0.637 | 0.469 | +0.168 | MODEL-DEFECT |
| `log_ret_30s_q90` | 1.000 | 0.469 | +0.531 | MODEL-DEFECT |
| `log_ret_60s_q10` | 0.000 | 0.000 | +0.000 | FAITHFUL |
| `log_ret_60s_q50` | 0.687 | 0.000 | +0.687 | MODEL-DEFECT |
| `log_ret_60s_q90` | 0.687 | 0.000 | +0.687 | MODEL-DEFECT |
| `fifo_tp4sl3_net` | 0.000 | 0.076 | -0.076 | FAITHFUL |
| `fifo_tp8sl5_net` | 0.000 | 0.065 | -0.065 | FAITHFUL |

## Strategic implication per HC #475

- All `FAITHFUL` rows = labeling redesign is the right lever (HC #475 R4 'better labeling' clause).
- Any `MODEL-DEFECT` rows = loss-function or architecture-level fix needed for that head.
- The action plan splits between the two: keep usable heads, retrain or drop defective heads.