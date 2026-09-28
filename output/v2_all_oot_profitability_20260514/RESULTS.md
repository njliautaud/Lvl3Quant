# HC #355 — v2 All-OOT-Dates Profitability Sweep

**Run date:** 2026-05-14 09:43:32
**Model:** cnn_mamba_v2_smart_v3 (champion v2; IC_1s≈0.22, IC_5s≈0.14, IC_10s≈0.11)

## Honest scope statement

The user's question was: "does this setup survive ALL OOT dates from Feb to Apr 29? Do we have any config with v2 that is profitable?"

**Available v2 OOT prediction coverage:** 47 unique dates from 20260223 → 20260427 (28 of which have FIFO bid/ask labels with queue-aware fill rates).

**Feb 1-22 has NO v2 OOT predictions** — the earliest v2 OOT day is `20260223`. Feb coverage = Feb 23 → Feb 27 (5 days from smart_v3_mar fold_00..fold_04). Mar 1-5 from fold_05..09. Mar 6 → Apr 29 from per-day NPZs.

This is the FULL set of dates the v2 model has produced OUT-OF-SAMPLE predictions for. Backfilling Feb 1-22 would require either retraining/inference on earlier windows or different fold schedules — neither of which exists today.

## Methodology

**Stage 1 — Fast face-value sweep (HC #320 FIFO-FLOOR):**
- For every (side, band, horizon, order_type, cancel_evals, hold_s) tuple in the directive grid, compute per-trade NET tick returns using v2 predictions + per-horizon realized tick moves.
- Cost stack: passive = 0.376 ticks commission; market/IOC = 1.376 ticks (commission + 1-tick spread crossing); back-off = passive cost − 1 tick realized (entry 1 tick worse).
- Fill probability for passive orders: derived from observed FIFO label fill rate (11.6% short / 6.2% long at touch with 2s cancel) scaled by `1 − (1−p_base)^(cancel_evals/8)`. Back-off gets a queue-position advantage modeled as ~2× base prob capped at 70%.
- Hold-seconds shorter than the prediction horizon: edge scaled by `sqrt(hold/horizon)` (conservative; per HC #355 decay analysis edge dies by 30s).

**Stage 2 — Real FIFO market replay (HC #357):**
- Top-N stage-1 stable configs are re-run through `FIFOReplayEngine` on 5 sample dates each.
- Includes: queue position on arrival, fill probability by queue depletion, adverse selection post-fill, commission, cancel/replace logic, max-hold exit.
- This is the queue+adv-sel layer demanded by HC #357 / HC #349.

## 1. Date Coverage

Total dates analyzed: **47**, FIFO-labeled: **28**.

<details><summary>Full date list</summary>

- `20260223` (n_windows=24,665, fifo=NO)
- `20260224` (n_windows=23,692, fifo=NO)
- `20260225` (n_windows=14,082, fifo=NO)
- `20260226` (n_windows=27,618, fifo=NO)
- `20260227` (n_windows=30,121, fifo=NO)
- `20260302` (n_windows=29,725, fifo=NO)
- `20260303` (n_windows=39,602, fifo=NO)
- `20260304` (n_windows=22,654, fifo=NO)
- `20260305` (n_windows=37,189, fifo=NO)
- `20260306` (n_windows=81,293, fifo=YES)
- `20260309` (n_windows=69,279, fifo=YES)
- `20260310` (n_windows=65,666, fifo=YES)
- `20260311` (n_windows=58,561, fifo=YES)
- `20260312` (n_windows=62,414, fifo=YES)
- `20260313` (n_windows=61,000, fifo=YES)
- `20260316` (n_windows=50,941, fifo=YES)
- `20260317` (n_windows=29,968, fifo=YES)
- `20260318` (n_windows=31,804, fifo=YES)
- `20260319` (n_windows=43,872, fifo=YES)
- `20260320` (n_windows=65,967, fifo=NO)
- `20260322` (n_windows=2,427, fifo=NO)
- `20260323` (n_windows=99,030, fifo=NO)
- `20260324` (n_windows=85,582, fifo=NO)
- `20260325` (n_windows=67,238, fifo=NO)
- `20260326` (n_windows=63,419, fifo=NO)
- `20260327` (n_windows=67,713, fifo=NO)
- `20260329` (n_windows=2,113, fifo=NO)
- `20260330` (n_windows=68,867, fifo=NO)
- `20260331` (n_windows=87,996, fifo=NO)
- `20260401` (n_windows=67,072, fifo=YES)
- `20260402` (n_windows=75,132, fifo=YES)
- `20260403` (n_windows=4,000, fifo=YES)
- `20260405` (n_windows=1,707, fifo=YES)
- `20260406` (n_windows=47,032, fifo=YES)
- `20260407` (n_windows=77,534, fifo=YES)
- `20260408` (n_windows=64,446, fifo=YES)
- `20260409` (n_windows=51,566, fifo=YES)
- `20260410` (n_windows=44,393, fifo=YES)
- `20260412` (n_windows=1,868, fifo=YES)
- `20260413` (n_windows=45,285, fifo=YES)
- `20260414` (n_windows=42,023, fifo=YES)
- `20260415` (n_windows=49,166, fifo=YES)
- `20260416` (n_windows=50,371, fifo=YES)
- `20260417` (n_windows=55,783, fifo=YES)
- `20260419` (n_windows=1,525, fifo=YES)
- `20260420` (n_windows=51,390, fifo=YES)
- `20260427` (n_windows=47,351, fifo=YES)

</details>

## 2. Stability Verdict (Stage 1 — FIFO-floor)

**Stability bar (per HC #355):**
- Sharpe > 1.0 on > 70% of dates
- Worst-date Sharpe > -1.0
- Net positive on > 50% of dates

**Configs passing all 3 criteria: 0 / 1,575**

### HONEST NULL RESULT

Under the FIFO-floor cost stack, **ZERO v2 configs pass the stability bar across Feb 23 → Apr 29 2026**.

This is an UPPER BOUND on what v2 can deliver: stage 1 does NOT include queue-position penalty or adverse-selection cost (those would only make results worse). Therefore the honest answer to the user's question is:

> **No v2 config has been demonstrated to be Sharpe>1 stable across all OOT dates from Feb 23 to Apr 29 2026.** The CLAUDE.md canonical claim of "v2 top-10% short, +1.56t avg 60.5% WR" is a face-value AVERAGE across OOT data — it does NOT mean every date is profitable, and it does NOT survive realistic execution costs uniformly.

## 3. Top-5 Configs Overall (by aggregate Sharpe)

| Rank | side | band | horiz | order_type | cancel | hold | n_trades | net_ticks | avg Sharpe | %dates Sh>1 | worst Sh | %dates net+ | passes |
|-----:|:----:|-----:|:-----:|:-----------|------:|-----:|--------:|----------:|-----------:|------------:|--------:|-------------:|:-------:|
| 1 | long | P99.5 | 10s | passive_at_touch | 20 | 10.0 | 4,600 | -23.6 | +0.11 | 20% | -3.16 | 64% | no |
| 2 | long | P99.5 | 10s | passive_at_touch | 35 | 10.0 | 6,524 | +183.0 | +0.10 | 26% | -2.90 | 59% | no |
| 3 | long | P99.5 | 10s | passive_at_touch | 50 | 10.0 | 7,715 | -227.8 | +0.08 | 26% | -2.89 | 60% | no |
| 4 | long | P99.0 | 10s | passive_at_touch | 35 | 10.0 | 12,919 | -347.0 | +0.06 | 36% | -3.20 | 53% | no |
| 5 | long | P99.0 | 10s | passive_at_touch | 50 | 10.0 | 15,396 | -1030.9 | -0.03 | 30% | -3.63 | 51% | no |

## 4. Per-Date Detail for Top Config (#1)

**Config:** ` long P99.5  10s passive_at_touch     cancel=20 hold=10.0s`

| date | n | net_ticks | Sharpe | WR | MDD |
|:-----|--:|---------:|------:|----:|----:|
| 20260223 | 31 | +27.3 | +0.76 | 58.1% | -29.3 |
| 20260224 | 32 | -13.0 | -0.48 | 50.0% | -21.6 |
| 20260225 | 25 | +25.6 | +1.46 | 72.0% | -13.8 |
| 20260226 | 35 | +43.8 | +1.18 | 57.1% | -28.5 |
| 20260227 | 32 | +27.0 | +0.94 | 53.1% | -19.5 |
| 20260302 | 35 | +121.8 | +3.29 | 68.6% | -10.5 |
| 20260303 | 60 | +105.4 | +2.02 | 60.0% | -22.1 |
| 20260304 | 26 | +7.2 | +0.26 | 46.2% | -16.8 |
| 20260305 | 54 | +73.7 | +1.31 | 44.4% | -45.9 |
| 20260306 | 49 | +6.6 | +0.15 | 42.9% | -31.4 |
| 20260309 | 135 | -7.8 | -0.09 | 50.4% | -101.9 |
| 20260310 | 121 | -137.5 | -1.89 | 44.6% | -145.0 |
| 20260311 | 89 | -8.5 | -0.17 | 48.3% | -54.7 |
| 20260312 | 77 | -20.0 | -0.32 | 48.1% | -59.6 |
| 20260313 | 95 | -35.7 | -0.54 | 44.2% | -101.9 |
| 20260316 | 138 | +73.1 | +1.49 | 55.1% | -18.1 |
| 20260317 | 127 | -121.3 | -3.16 | 37.8% | -123.1 |
| 20260318 | 133 | +7.5 | +0.13 | 54.1% | -58.0 |
| 20260319 | 140 | -19.1 | -0.26 | 47.9% | -83.2 |
| 20260320 | 100 | +44.4 | +0.92 | 50.0% | -29.2 |
| 20260322 | 3 | -23.1 | -1.53 | 33.3% | -3.4 |
| 20260323 | 105 | +107.0 | +0.83 | 51.4% | -81.5 |
| 20260324 | 122 | +114.1 | +1.46 | 54.1% | -63.0 |
| 20260325 | 91 | +17.8 | +0.32 | 50.5% | -52.3 |
| 20260326 | 80 | -54.1 | -1.61 | 43.8% | -70.4 |
| 20260327 | 86 | +48.7 | +0.72 | 55.8% | -40.9 |
| 20260329 | 1 | -9.9 | +0.00 | 0.0% | +0.0 |
| 20260330 | 86 | +2.2 | +0.05 | 44.2% | -49.2 |
| 20260331 | 136 | +117.9 | +1.36 | 55.9% | -74.3 |
| 20260401 | 189 | +76.4 | +0.91 | 54.5% | -55.9 |
| 20260402 | 235 | +58.6 | +0.41 | 48.1% | -85.0 |
| 20260403 | 5 | -1.9 | -0.39 | 60.0% | -1.4 |
| 20260405 | 5 | +0.1 | +0.02 | 40.0% | -4.9 |
| 20260406 | 128 | +34.4 | +0.47 | 53.9% | -58.5 |
| 20260407 | 216 | -279.7 | -0.92 | 55.1% | -489.8 |
| 20260408 | 175 | -179.8 | -2.65 | 44.0% | -211.6 |
| 20260409 | 188 | -138.2 | -2.09 | 43.6% | -189.9 |
| 20260410 | 144 | -110.1 | -2.83 | 42.4% | -114.9 |
| 20260412 | 8 | +9.5 | +0.66 | 50.0% | -10.0 |
| 20260413 | 160 | -114.7 | -2.03 | 46.2% | -127.4 |
| 20260414 | 160 | -34.7 | -0.73 | 41.2% | -53.7 |
| 20260415 | 158 | +40.6 | +0.98 | 53.8% | -25.8 |
| 20260416 | 121 | +5.0 | +0.16 | 52.1% | -55.8 |
| 20260417 | 194 | +1.1 | +0.02 | 58.8% | -56.4 |
| 20260419 | 7 | +19.4 | +2.13 | 85.7% | -1.9 |
| 20260420 | 133 | +7.0 | +0.14 | 56.4% | -52.8 |
| 20260427 | 134 | +29.1 | +0.74 | 53.7% | -42.8 |

## 5. Stage 2 — Real FIFO Market Replay (HC #357)

Stage 2 skipped — no stage-1 configs passed the stability bar, so there is nothing worth validating with the full queue+adv-sel stack.

## 6. One-line strongest-v2-config recommendation

> **No v2 config passes the stability bar.** The currently-deployed paper config (HC #354: short-side, P95, passive limit) corresponds to NO row that passes 3 stability criteria simultaneously. The v2 short top-10% face-value edge claim does NOT generalize uniformly across Feb 23 → Apr 29 2026 OOT dates. Live paper trading remains the production queue+adv-sel test.

## 7. Files

- `stage1_per_date.csv.gz` — full per-(date,config) metrics matrix
- `stage1_aggregated.csv.gz` — per-config aggregated stability stats
- `stage2_real_replay.json` — real FIFO replay results for top candidates
- `gate_sweep_heatmap.png` — side × band × order_type avg-Sharpe heatmap
- `per_date_pnl_curves.png` — per-date cumulative NET PnL for top-5 configs
- `run.log` — full execution log

---

## 8. CRITICAL FINDING — Signal Integrity Discrepancy Between Prediction Sources

During this analysis, a sharp discontinuity in v2 model behavior was uncovered between the two prediction sources used to cover the Feb 23 → Apr 29 window:

**v2 IC_1s (face-value information coefficient) by source:**

| Date | Source | IC_1s | n |
|:-----|:-------|------:|--:|
| 20260223 | `cnn_mamba_v2_smart_v3_mar/fold_00` | **+0.226** | 24,665 |
| 20260306 | `cnn_mamba_v2_all_oot/2026...` | **+0.003** | 81,293 |
| 20260427 | `cnn_mamba_v2_all_oot/2026...` | **+0.123** | 47,351 |

**Aggregate face-value edge of "v2 short top-10%" by source:**

| Source | Date range | n_dates | n_trades | avg short move @ 1s | WR |
|:-------|:----------|--------:|---------:|--------------------:|---:|
| `cnn_mamba_v2_smart_v3_mar/fold_*` | Feb 23 – Mar 5 | 10 | 27,420 | **+0.620 ticks** | **53.2%** |
| `cnn_mamba_v2_all_oot/*` | Mar 6 – Apr 27 | 37 | ~166,000 | **−0.009 ticks** | **41.6%** |

The fold-source numbers MATCH CLAUDE.md and today's gate-coverage audit (HC #352 reported +0.62t / 53.2% WR for P90 short). The per-day-source numbers do NOT match — the v2 edge has either decayed catastrophically after Mar 5, OR (more likely) the per-day inference pipeline used to generate `cnn_mamba_v2_all_oot/` introduced a defect.

### Most plausible causes

1. **Feature-stats mismatch** — the per-day NPZs may have been computed with the WRONG fold's `feature_stats.npz`, breaking input normalization.
2. **Checkpoint drift** — different `.pt` weights used per inference batch.
3. **Sign convention flip** — label sign / target convention may have been inverted in a re-export.
4. **Real signal decay** — March 6 onward, market regime shifted enough to kill v2's edge.

### Why this matters for the user's question

The user asked whether v2 has any config profitable across ALL OOT dates Feb → Apr 29. The honest answer is two-fold:

- **YES — on the original canonical OOT (Feb 23 – Mar 5)**: short P90 at 1s/5s/10s shows the documented +0.62t edge, 53% WR. With passive limits this nets ~+0.244t/trade after commission. But concentrating analysis on just 10 days does NOT prove uniformity across a longer window.
- **NO — across the Mar 6 → Apr 27 per-day window as currently exposed**: the predictions in `cnn_mamba_v2_all_oot/` show near-zero IC and slightly-negative face-value edge. Whether this reflects real signal decay or a prediction-pipeline bug, it means **we cannot today claim v2 has any uniformly profitable config across the full Feb–Apr window**.

### Recommended next action

**Audit the `cnn_mamba_v2_all_oot/` inference pipeline before trusting any of its predictions.** Specifically:

1. Verify which `fold_*_best.pt` produced which per-day NPZ (`oot_files` field tells us the input but not which checkpoint).
2. Re-run inference for a sample date (e.g., 20260306) using the canonical `fold_09_best.pt` + `fold_09_feature_stats.npz` from `cnn_mamba_v2_smart_v3_mar/` and check IC vs the existing per-day NPZ.
3. If IC recovers (back to ~0.20) → existing per-day NPZ is broken; re-generate all of them.
4. If IC stays low → real signal decay; verifies v3.x is needed and answers the user's question with "v2 alone cannot trade profitably uniformly after Mar 5".

This audit is OUTSIDE the scope of HC #355 (which is the profitability sweep) but is now a HIGHER PRIORITY than further v2 backtesting because every downstream conclusion depends on which interpretation is correct.

---
Per HC #349: stage-1 numbers are FIFO-floor; stage-2 numbers include queue + adv-sel. Per HC #355: stability verdict computed exactly as specified.
Per HC #307D: this is a new analysis script under scripts/v3_3_research/; trainer code untouched.