# v7 FIFO Slice Analysis — HC #494 R1 Hunt — 2026-05-28

Source: output/fifo_v7_grade/fills.parquet (18,675 fills, 17 OOT days)
Baseline FIFO net: -0.621 t/trade (REJECTED in HC #491 R1 report)
Cost model: passive limit = 0.376t commission only (HC #494 R4 — FIFO net leads)

## DATA AVAILABILITY GAP

The fills parquet contains: date, direction, hold_s, fill_type, net_ticks, net_dollars,
queue_ahead, queue_wait_ns, slippage_ticks, pred_strength. **There is NO entry_time column.**

- **Slice A (30-min time-of-day buckets)**: CANNOT COMPUTE without entry timestamps.
- **Slice B (microprice agreement at entry−100ms)**: CANNOT COMPUTE — same blocker.
- **Slice C (regime × confidence)**: COMPUTED in full below.
- Substituted Slices A/B with the best ex-ante features actually available in fills:
  pred_strength deciles and queue_ahead buckets (both observable at order placement).

## EX-ANTE-ONLY RESULTS (no post-hoc fill_type filters)

Total cells evaluated: 128. **Zero cells with FIFO net > 0.**

### Best 25 cells by FIFO net ticks/trade

| slice          | cell                |    n |   ndays |   net_t |    wr |   pos_days |   avg_trades_per_day |
|:---------------|:--------------------|-----:|--------:|--------:|------:|-----------:|---------------------:|
| conf×queue×dir | d9_q1-2_long        |   85 |      13 | -0.3348 | 0.4   |          1 |                  6.5 |
| conf×queue×dir | d10_q3-5_short      |   74 |      13 | -0.376  | 0.378 |          4 |                  5.7 |
| conf×queue×dir | d10_q1-2_long       |   78 |      13 | -0.4337 | 0.333 |          2 |                  6   |
| q×regime×dir   | q6-10_RED_long      |   66 |       1 | -0.4518 | 0.364 |          0 |                 66   |
| conf×queue×dir | d9_q6-10_long       |  153 |      14 | -0.4544 | 0.373 |          2 |                 10.9 |
| conf×queue×dir | d9_q6-10_short      |  133 |      13 | -0.4662 | 0.346 |          1 |                 10.2 |
| conf×queue×dir | d8_q6-10_long       |  168 |      13 | -0.4802 | 0.327 |          2 |                 12.9 |
| q×regime×dir   | q3-5_FLAT_long      |  184 |       3 | -0.4928 | 0.359 |          0 |                 61.3 |
| q×regime×dir   | q1-2_GREEN_long     |  597 |       9 | -0.51   | 0.323 |          0 |                 66.3 |
| regime×conf    | top1%_GREEN_short   |   78 |       9 | -0.5106 | 0.346 |          1 |                  8.7 |
| regime×conf    | top2%_GREEN_short   |  151 |       9 | -0.5217 | 0.331 |          0 |                 16.8 |
| pred_strength  | d1_long             |  709 |       8 | -0.522  | 0.312 |          1 |                 88.6 |
| q×regime×dir   | q1-2_RED_short      |   34 |       1 | -0.5231 | 0.353 |          0 |                 34   |
| conf×queue×dir | d8_q1-2_short       |   88 |      16 | -0.5294 | 0.364 |          3 |                  5.5 |
| queue_ahead    | q1-2_long           |  793 |      17 | -0.5305 | 0.313 |          0 |                 46.6 |
| q×regime×dir   | q6-10_FLAT_long     |  329 |       3 | -0.5325 | 0.343 |          0 |                109.7 |
| conf×queue×dir | d10_q3-5_long       |   86 |      15 | -0.533  | 0.337 |          3 |                  5.7 |
| q×regime×dir   | q3-5_RED_short      |   44 |       1 | -0.5351 | 0.386 |          0 |                 44   |
| q×regime×dir   | q1-2_GREEN_short    |  630 |       9 | -0.5403 | 0.317 |          0 |                 70   |
| q×regime×dir   | q3-5_FLAT_short     |  182 |       3 | -0.5408 | 0.313 |          0 |                 60.7 |
| conf×queue×dir | d9_q3-5_short       |   91 |      13 | -0.5463 | 0.341 |          2 |                  7   |
| pred_strength  | d1_both             | 1868 |      14 | -0.547  | 0.305 |          1 |                133.4 |
| queue_ahead    | q1-2_both           | 1646 |      17 | -0.547  | 0.313 |          0 |                 96.8 |
| regime×conf    | top2%_GREEN_both    |  271 |       9 | -0.5494 | 0.325 |          0 |                 30.1 |
| regime×conf    | top0.5%_GREEN_short |   43 |       9 | -0.5504 | 0.349 |          0 |                  4.8 |

### Worst 10 (for completeness)

| slice          | cell              |    n |   ndays |   net_t |    wr |   pos_days |   avg_trades_per_day |
|:---------------|:------------------|-----:|--------:|--------:|------:|-----------:|---------------------:|
| q×regime×dir   | q1-2_RED_long     |   40 |       1 | -0.776  | 0.2   |          0 |                 40   |
| regime×conf    | top1%_GREEN_long  |   60 |       9 | -0.776  | 0.233 |          1 |                  6.7 |
| q×regime×dir   | q11+_RED_long     |  359 |       1 | -0.7479 | 0.265 |          0 |                359   |
| conf×queue×dir | d8_q3-5_short     |   96 |      13 | -0.7354 | 0.281 |          0 |                  7.4 |
| q×regime×dir   | q11+_RED_short    |  391 |       1 | -0.7277 | 0.271 |          0 |                391   |
| regime×conf    | top10%_FLAT_short |  178 |       3 | -0.7159 | 0.264 |          0 |                 59.3 |
| regime×conf    | top2%_FLAT_long   |   46 |       3 | -0.713  | 0.261 |          0 |                 15.3 |
| regime×conf    | all_RED_long      |  515 |       1 | -0.7042 | 0.276 |          0 |                515   |
| conf×queue×dir | d10_q11+_long     |  671 |      13 | -0.6927 | 0.277 |          0 |                 51.6 |
| regime×conf    | all_RED_both      | 1038 |       1 | -0.692  | 0.287 |          0 |               1038   |

## SLICE-BY-SLICE HEADLINES

### Slice A — TIME-OF-DAY: NOT COMPUTABLE
fills.parquet lacks entry_time. Would require regenerating fills with timestamps
(directive says no regeneration). Recommend escalating: rerun harness with timestamp logging,
OR join via prediction-row index → original prediction timestamps if NPZ has them (it does not).

### Slice B — MICROPRICE AGREEMENT: NOT COMPUTABLE
Requires entry_time to join MBO event stream for microprice at t−100ms. Same blocker as A.

### Slice C — REGIME × CONFIDENCE (computable)
Best regime/conf cell: **top1% GREEN short — FIFO net −0.511 t/trade, n=78, 9 days, 1 positive day.**
All regime/conf combinations remain negative; no qualifying HC #494 R1 candidate.

### Note on post-hoc 'tp_only' slice
Filtering by fill_type=='tp' yields +1.624 t/trade (1.000 WR) — this is the TP payoff itself,
not an edge. Selecting only winners ex-post is look-ahead bias and is NOT tradeable.
Same applies to max_hold filter (+0.029 t/trade).

## BOTTOM LINE

**No ex-ante filter of v7 FIFO fills clears HC #494 R1.** The −0.62 t/trade adverse
selection from FIFO queue dynamics is present in every confidence bucket, every queue depth,
every regime, every direction. Confirms the HC #491 R1 root cause: 1s prediction horizon
incompatible with limit-order queue wait dynamics. Recommend escalating per HC #491 next axis:
retrain on 5s horizon target where MFE extends to 30s.