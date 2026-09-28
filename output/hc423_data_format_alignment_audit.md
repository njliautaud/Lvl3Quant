# HC #423 §3 — Data-Format Alignment Audit (LIVE vs TRAINING feat_vec)

**Date**: 2026-05-18
**Author**: Claude (read-only audit)
**Status**: ROOT-CAUSED — supersedes HC #421 Issue A "regime drift" verdict.

## TL;DR Verdict: **ENCODER-BUG (CATASTROPHIC)**

HC #421 Issue A was wrong. The +0.21 pred shift is NOT regime drift — it is a **multi-field encoder mismatch** in the live `paper_trading_v2_1s_short_top05.py::encode_event` wrapper (NOT in `StreamingFeaturesSmartV3` itself). At least **3 of the 6 raw input fields** are silently corrupted before they ever reach the streamer, and the live MBO recorder additionally drops 3 of 5 event types. Result: feat_vec dims with z-divergence in the **hundreds of sigmas** vs training. The model is being fed out-of-distribution garbage.

## Comparison Table (training=20260428 NPZ post-warmup N≈13.3M; live=2026-05-18 200k events from Razer)

| idx | feature | tr_mean | lv_mean | Δmean | tr_std | lv_std | **z=|Δm|/σtr** | tr_%zero | lv_%zero | smoking |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0  | time_delta_log              | +0.0000 | +0.6166 | +0.6166 | 0.0031 | 0.7601 | **201.4** | 99.96% | 49.95% | *** |
| 1  | event_type_id               | +0.2461 | +0.6872 | +0.4411 | 0.2840 | 0.2077 | **1.55**  | 39.90% |  8.37% | *** |
| 2  | side_id                     | -0.0094 | +0.2817 | +0.2911 | 1.0000 | 0.9592 |  0.29     |  0.00% |  0.00% |     |
| 3  | price_rel_ticks             | -0.0065 | +1.3923 | +1.3988 | 0.4535 | 0.9199 | **3.08**  | 26.59% | 30.39% | *** |
| 4  | qty_log                     | +0.0319 | -0.1317 | -0.1636 | 0.1028 | 0.2605 | **1.59**  |  0.00% |  0.00% | *** |
| 5  | spread_ticks                | +0.1649 | +0.2558 | +0.0909 | 0.1740 | 0.0918 | **0.52**  | 43.82% |  0.00% | *** |
| 6  | cancel_side_asym_50         | -0.0070 | +0.0000 | +0.0070 | 0.4518 | 0.0000 |  0.02     |  3.34% |**100.00%**|   |
| 7  | rolling_ofi_500             | +0.0002 | -0.0162 | -0.0163 | 1.0462 | 1.0536 |  0.02     |        |        |     |
| 8  | event_density_20            | +0.0001 | +1.3929 | +1.3928 | 0.0021 | 0.4876 | **650.8** | 99.44% |  1.64% | *** |
| 9  | price_mom_10                | -0.0004 | -0.0009 | -0.0006 | 0.9853 | 1.0044 |  0.00     |        |        |     |
| 10 | qty_price_mom_50            | -0.0002 | +0.0150 | +0.0152 | 0.9910 | 1.0399 |  0.02     |        |        |     |
| 11 | price_sign_momentum_200     | -0.0038 | +1.3923 | +1.3961 | 0.5347 | 0.0908 | **2.61**  |  0.68% |  0.00% | *** |
| 12 | event_type_entropy_200      | +0.8031 | +0.1612 | -0.6419 | 0.0880 | 0.0938 | **7.29**  |        |        | *** |
| 13 | fill_add_restoration_100    | +0.0310 | +0.0038 | -0.0271 | 0.0682 | 0.0085 |  0.40     | 72.62% | 78.52% |     |
| 14 | spread_velocity_50          | +0.0012 | +0.0020 | +0.0008 | 1.0010 | 1.0109 |  0.00     |  0.62% | 44.72% |     |
| 15 | queue_replenishment         | -0.0134 | +0.0509 | +0.0642 | 0.9375 | 1.0880 |  0.07     |        |        |     |
| 16 | mom_divergence              | -0.0001 | +0.0000 | +0.0001 | 2.1245 | 0.4730 |  0.00     |        |        |     |
| 17 | ofi_x_spread                | +0.0005 | -0.0092 | -0.0097 | 0.9918 | 1.0157 |  0.01     |        |        |     |
| 18 | vol_weighted_pmom           | -0.0002 | +0.0150 | +0.0152 | 0.9910 | 1.0399 |  0.02     |        |        |     |
| 19 | buy_sell_intensity_ratio    | +0.0094 | -0.2818 | -0.2912 | 0.2106 | 0.1769 | **1.38**  |        |        | *** |
| 20 | realized_volatility         | +0.0011 | +0.0093 | +0.0082 | 1.0211 | 1.0296 |  0.01     |        |        |     |
| 21 | sweep_intensity             | +0.0148 | +0.6352 | +0.6203 | 0.6798 | 2.2772 | **0.91**  | 29.33% |  1.93% | *** |
| 22 | ofi_short_100               | +0.0002 | -0.0036 | -0.0038 | 1.0148 | 1.0315 |  0.00     |        |        |     |
| 23 | ofi_long_2000               | +0.0022 | -0.0194 | -0.0216 | 1.1646 | 1.1850 |  0.02     |        |        |     |
| 24 | ofi_acceleration            | +0.0002 | +0.0011 | +0.0008 | 1.0011 | 1.0217 |  0.00     |        |        |     |

**10 dims diverge by >0.5σ. Two dims diverge by >200σ.** This is not noise. This is not regime drift.

## Smoking guns (with root cause)

The bug is in the **6-tuple builder** (`encode_event` in `paper_trading_v2_1s_short_top05.py` lines ~1103–1125, replicated in `diag_live_pred_distribution.py`). `StreamingFeaturesSmartV3` itself is correct — its job is to consume a clean 6-tuple. The 6-tuple it receives in live is **wrong on three of six fields**:

1. **`time_delta_log` (dim 0, z=201σ)** — Live uses `math.log1p(delta_us)` with `delta_us = (ts_ns - prev_ts_ns)/1000`. The training raw NPZ has `time_delta_log` distributed mean≈0.0005, 99.95% zero, max=10. Live has mean≈3.1 raw (→0.62 normalized), 50% nonzero. The training builder evidently emitted log of **inter-event delta with a different unit/clipping** (likely seconds or a coarser quantization). Live builder uses microseconds → values 3-6× larger.

2. **`price_rel_ticks` (dim 3, z=3.1σ; dim 11 also collapses to constant +1.4)** — Live raw `price_ticks` is **absolute price in ticks** (mean 21,389, max 29,564 = $7,391). Training raw is **price-relative-to-mid in ticks** (range −50…+50). The live encoder passes absolute price straight through → every event clips to +50 → normalized +2.0. Dim 11 (`price_sign_momentum_200` = causal sum of sign(price_rel_ticks)) is **stuck at +1.4** because every nonzero live tick is positive. This single bug **alone** would drive the model into "everything is going up" mode → positive pred bias.

3. **`event_type_id` (dim 1, z=1.55σ; dims 6, 8, 12 also broken)** — Live MBO recorder only emits action ∈ {0=Add, 3=Trade}. Cancels, modifies, fills are missing entirely. Then the live encoder additionally collapses: `etype = 3 if action >= 2 else 0`. Net effect: only types {0, 3} reach the streamer. Training distribution is `{Add 40%, Cancel 40%, Modify 10%, Trade 3%, Fill 7%}`. Consequences:
   - dim 6 `cancel_side_asym_50` = **100% zero** (no cancels at all)
   - dim 12 `event_type_entropy_200` collapses 0.80 → 0.16 (only 2 types)
   - dim 8 `event_density_20` z=650σ (broken via the time_delta_log corruption)

4. **`qty_log` (dim 4, z=1.59σ)** — Live computes `math.log(max(1, size))` where `size` is often 0 in trades → log(1)=0. Training computed qty_log over orders where size≥2 (min 0.693 = log 2). Live distribution drifts negative because of the `size=0` floor at log(1)=0 then normalized to (0 − 0.693)/3.0 = −0.231.

5. **`spread_ticks` (dim 5)** — Training range [0, 20] ticks. Live range [0.8, 3.2] ticks raw → [0.2, 0.8] normalized. Live spread is much tighter than the wide range training saw (sessions with vol events). Minor concern; not load-bearing.

## Recommendation (in priority order)

1. **STOP retraining v2.** HC #421's "retrain on data through 2026-05-15" recommendation is wrong-targeted — retraining will just memorize the broken live distribution and remain broken on real production.

2. **FIX the live encoder** (file: `live_trading_linux/paper_trading_v2_1s_short_top05.py` and any sibling live traders sharing the same `encode_event`). Three concrete fixes, in order:
   - **(a)** Compute `price_rel_ticks = price_ticks - mid_ticks` where `mid_ticks = (best_bid + best_ask) / (2 * TICK_SIZE)`, then clip [−50, +50]. NOT raw `price_ticks`.
   - **(b)** Use the **raw `action` value** directly as `event_type_id` (0…4) — drop the `etype = 3 if action >= 2 else 0` collapse.
   - **(c)** Replace `delta_us` with seconds (or whatever unit the original builder used — verify from training pipeline source, NOT from the v3 precompute, which only consumes the already-built 6-tuple). The raw NPZ metadata shows mean=0.0005 → likely `log1p(delta_seconds)` capped low.

3. **FIX the live MBO recorder** to emit Cancels (action=1), Modifies (action=2), Fills (action=4) — not just Adds and Trades. Without these events the recorded stream is fundamentally lossy and re-deriving them is impossible.

4. **Then** revalidate pred distribution. If post-fix pred mean returns to ~0.06 and std ~0.38, HC #421 cutover can proceed on the existing checkpoint. No retrain required.

## Files

- Training stats: `/tmp/train_stats_20260428.npz`
- Live stats: `/tmp/live_stats_20260518.npz`
- Live raw tail (200k events): `/tmp/live_events_tail.jsonl`
- This audit: `/home/jupiter/Lvl3Quant/output/hc423_data_format_alignment_audit.md`
