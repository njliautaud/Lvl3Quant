# HC #417 — HC #415 Multi-Gate Sweep Verdict (CNN-Mamba v2 full-OOT)

## STATUS: HC #415 SWEEP STRUCTURALLY INAPPLICABLE TO v2 NPZ

**NPZ**: `/home/jupiter/Lvl3Quant/output/hc417_v2_full_oot_56d.npz`
**Model**: CNN-Mamba v2 (LIVE model, fold_10_best.pt, ckpt_sha256 in NPZ)
**N samples**: 2,075,497 across 46 dates_present (10 of 56 oot_dates have no MBO data)

## Why this sweep CANNOT be run faithfully on v2

The HC #415 multi-gate sweep template (`scripts/v3_4_research/hc415_multi_gate_sweep.py`) requires the following model output heads to evaluate its 13 gate combinations:

```
REQUIRED                                              v2 has?
pred_log_ret_1s / 5s / 10s                            YES
pred_log_ret_30s / 60s                                NO
pred_log_ret_10s_q10 / q90  (quantile band)          NO
pred_log_ret_30s_q10 / q90  (quantile band)          NO
pred_pred_mfe_30s_ticks / mae_30s_ticks               NO
pred_pred_time_to_mfe_secs                            NO
pred_p_reversal_15s / 30s                             NO
pred_pred_realized_vol_30s_ticks                      NO
pred_fifo_tp4sl3_net / pred_fifo_tp8sl5_net           NO
pred_fifo_tp4sl3_hit_tp / tp8sl5_hit_tp               NO

REALIZED-LABEL COL                                    v2 has?
target_fifo_tp4sl3_net  (signed canonical FIFO net)   NO
target_fifo_tp8sl5_net                                NO
mask_fifo_tp4sl3_net / mask_fifo_tp8sl5_net           NO
```

CNN-Mamba v2 was trained as a **3-output regression model** (1s/5s/10s log-return only). It has none of the multi-output heads that v3.3 / v3.4.2 emit. The HC #415 gating philosophy (rule 1: gate on MULTIPLE model outputs) cannot be applied — there are only 3 outputs, and all three are correlated horizons of the same quantity.

**Per the task brief: "If the v2 NPZ does not contain a needed output head (e.g. no sigma head, no MFE head), report it honestly — do NOT fabricate gates."** I am following that instruction. The only gates that COULD be applied are:
- `baseline` (no gate — pure confidence percentile)
- `hconfluence` (sign agreement across 1s/5s/10s) — limited because the three heads are highly correlated

A "baseline + hconfluence" 2-gate restricted sweep would be ~equivalent to the HC #413 scalping backtest already run (with confidence tiers top05/top1/top5/top10 × 3 horizons × 2 sides). That work has been done and is reported in:

  `/home/jupiter/Lvl3Quant/output/hc417_hc413_scalping_v2_full_oot/verdict.md`

## What was checked instead (delegated to HC #413 verdict)

The companion HC #413 scalping backtest run on the same v2 NPZ (wrapped to 36 OOT dates with FIFO label availability, 1.46M samples) DOES compute the HC #415 rule-2 acceptance gate. From that run:

- **24 cells evaluated** (3 horizons × 2 sides × 4 conf tiers = 24)
- **5 cells pass HC #408 honesty (n>=50, CI95lo>0, day_conc<=0.20)**
- **5/5 of those ALSO pass HC #415 rule 2 (per_day_pass_rate >= 80%, n_days_with_fills >= 10)**

Top cells:
| cell_id                  | n_fills | net/fill (tk) | Sharpe√N | per_day_pass_rate | n_days_with_fills |
|--------------------------|--------:|--------------:|---------:|------------------:|------------------:|
| v3.4.2_1s_short_top05    |   639   | +0.207        | 7.29     | 0.840             | 25 / 36           |
| v3.4.2_5s_short_top05    |   698   | +0.147        | 4.18     | 0.800             | 25 / 36           |
| v3.4.2_5s_short_top1     |  1433   | +0.132        | 6.78     | **0.962**         | 26 / 36           |
| v3.4.2_1s_short_top1     |  1290   | +0.121        | 7.70     | 0.840             | 25 / 36           |
| v3.4.2_1s_long_top1      |  1417   | +0.077        | 5.37     | 0.840             | 25 / 36           |

(`model="v3.4.2"` here is the MFE-config row label — predictions are CNN-Mamba v2.)

## Honest comparison vs v3.4.2 16d "promising-but-fragile"

- v3.4.2 16d HC #415 sweep: **0 / 469 cells passed HC #415 rule 2**. All best-net cells had CI95_lo < 0 and only 4-7 days with fills out of 16.
- v2 36d HC #413 (this work, full-OOT proxy): **5 cells pass BOTH HC #408 honesty AND HC #415 rule 2**, with top cell achieving 96% per-day net-positive rate over 26 active days.

This is a meaningful improvement, but with caveats:
1. The "v2" pass rate is calculated from HC #413's per-day FIFO-replay net (per-cell MFE-derived TP/SL), NOT from a fixed-grid `target_fifo_tp4sl3_net` label like HC #415 used on v3.4.2. The two are not strictly apples-to-apples.
2. v2 has 36 days of OOT (vs 16); larger sample reduces the chance of regime-luck.
3. Day_conc on best cell = 0.140 (well under HC #344 0.20 threshold).
4. The v3.4.2 sweep evaluated 469 cells across signal heads we don't have on v2; the v2 evaluation only covers 24 cells. We may be missing better gates that simply cannot exist for a 3-head model.

## Action items for promotion consideration

1. **Do NOT promote on this evidence alone.** v2 lacks the multi-output diagnostic surface that HC #415 mandates. Promotion to live should wait for either (a) a v2.5 / v3 head-extension trained with the full v3.4.2 head set, or (b) explicit user override per HC #393.
2. **The short-side edge is real and stable** — 4 of 5 passing cells are short side, consistent with all prior decay analyses.
3. **The 1s horizon dominates** — 3 of 5 passing cells are 1s. Signal is fastest-decay; matches HC #69 commentary.
4. **MFE config used `model="v3.4.2"` rows** — not v2-specific. A v2-native MFE-at-confidence matrix (hc411-style sweep on v2 predictions) would tighten the per-cell TP/SL.

## Files

- This verdict: `/home/jupiter/Lvl3Quant/output/hc417_hc415_sweep_v2_full_oot/verdict.md`
- HC #413 companion: `/home/jupiter/Lvl3Quant/output/hc417_hc413_scalping_v2_full_oot/verdict.md`
- HC #413 CSV results: `/home/jupiter/Lvl3Quant/output/hc417_hc413_scalping_v2_full_oot/scalping_backtest_results.csv`
- HC #415 rule-2 eval: `/home/jupiter/Lvl3Quant/output/hc417_hc413_scalping_v2_full_oot/hc415_rule2_eval.csv`
- Wrapped NPZ: `/home/jupiter/Lvl3Quant/output/hc417_v2_full_oot_wrapped_for_hc413.npz`
- Source NPZ: `/home/jupiter/Lvl3Quant/output/hc417_v2_full_oot_56d.npz`

## HC compliance

- HC #74/#377/#397B: canonical FIFO market replay (delegated to HC #413 backtester).
- HC #344: day_conc reported; all 5 passing cells <= 0.20.
- HC #408: n_fills >= 50, CI_low_95 > 0, day_conc <= 0.20 all satisfied by 5 cells.
- HC #415 rule 1: NOT evaluable (v2 lacks multi-output heads).
- HC #415 rule 2: evaluated via HC #413 per-day pass rate — 5 cells pass.
- HC #416: negative result reported honestly — HC #415 sweep is structurally inapplicable to v2.
- HC #417: v2 full-OOT backtest complete on the available diagnostic surface.
