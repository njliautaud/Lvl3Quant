# Razer Direction Horizon-Stratified v1 — Verdict

Run timestamp: 2026-05-31 18:42 ET
Node: Razer (RTX 3070, XGB GPU)
OOT window: [20260317, 20260429]  (16 dates, after burn-in 9 OOS folds per bucket)
Features (12): v1 first-passage heads (9) + CNN-Mamba-v2 join (3)
v2 join coverage: 86.78%
MLflow run id: `3c3767cce83b42ae869081607751d16d` exp `razer_direction_horizon_strat_v1`

## Bucket axis
First-passage time (FPT) = min(tp1_dt_ns, sl1_dt_ns) / 1e9.
Brief asked for {<5s, 5-15s, >=15s} but ~99% of walks hit a 1-tick barrier in <5s
(median FPT 0.018s, 95th percentile 1.76s). Empirically log-scale-equivalent bins used:
- h_short:  FPT <  0.1s
- h_mid:    0.1s <= FPT < 1.0s
- h_long:   FPT >= 1.0s

## Results table

| bucket  | n_total | n_oos (after \|dmid\|>=1t) | OOS AUC | OOS Brier | pos_rate | fold AUC mean +/- std (n=9) |
|---------|--------:|---------------------------:|--------:|----------:|---------:|-----------------------------|
| h_short | 143,244 |                     69,551 |  0.5098 |    0.2504 |   0.4968 | 0.5141 +/- 0.0221           |
| h_mid   |  66,760 |                     30,678 |  0.4968 |    0.2502 |   0.5040 | 0.5041 +/- 0.0103           |
| h_long  |  20,881 |                      8,344 |  0.4934 |    0.2503 |   0.4944 | 0.5003 +/- 0.0267           |

All three buckets land at random-coin AUC. Best bucket (h_short) overall 0.5098 is
~8 points below the 0.58 accept gate. Fold-AUC means inside one std of 0.50.

## Top-q confidence regrade (gross ticks; net = gross - 1.376t taker cost)
Taker break-even WR ~57.6%.

| bucket  |   q   |    n | WR     | mean gross (t) | mean net taker (t) |
|---------|------:|-----:|-------:|----------------|---------------------|
| h_short |  1.0% |  696 | 0.4957 |  +0.1861       |  -1.1899            |
| h_short |  5.0% | 3478 | 0.5239 |  +0.2779       |  -1.0981            |
| h_short | 10.0% | 6961 | 0.5182 |  +0.2250       |  -1.1510            |
| h_mid   |  1.0% |  307 | 0.5212 |  +0.4756       |  -0.9004            |
| h_mid   |  5.0% | 1534 | 0.4922 |  +0.1323       |  -1.2437            |
| h_mid   | 10.0% | 3069 | 0.4917 |  +0.0560       |  -1.3200            |
| h_long  |  1.0% |   84 | 0.5119 |  +0.3393       |  -1.0367            |
| h_long  |  5.0% |  418 | 0.4641 |  +0.0622       |  -1.3138            |
| h_long  | 10.0% |  836 | 0.4797 |  +0.0173       |  -1.3934            |

No bucket clears WR >= 57.6% at any top-q tier. Every regrade row is net-negative
under taker cost. Even top-1% confidence picks WR sits at 49-52% — pure coin flip.

## Verdict

**NO-GO — direction-at-short-horizon hypothesis dead.**

Accepted buckets (AUC >= 0.58 AND top-1% net taker > 0): `[]`

Direction edge is NOT recovered by isolating fast first-passage events. The earlier
direction_v1 Q1-by-hold_npts AUC 0.5526 result does NOT translate to FPT-by-horizon
training — it was likely a stratification artifact (hold_npts couples with volatility
regime more than horizon-of-edge).

## Axis-rotation recommendation (escalate to user)

All three horizon-axis cuts of direction-classification land at AUC ~0.50. Combined
with today's earlier NO-GOs (taker K-framework v1/v2/v3, continuous MFE regressor,
direction_v1 full features), the SIGN-of-net-move target on existing first-passage
walks does not appear learnable from the current feature set, regardless of horizon
bucket.

Suggested next axes (pick one, NOT all):
1. **Reframe target**: instead of {long-side-continuation vs reversal}, predict
   {side-of-MFE-favorable-leg-magnitude > side-of-MAE-adverse-leg-magnitude} —
   asymmetric MFE/MAE ratio at fixed horizon, fed into queue-position-aware sizing.
2. **Reframe features**: drop the v1 first-passage heads (which only carry
   horizon-collapsed predictions of MFE/MAE/toxicity) and rebuild features from
   raw MBO at native h in {1s, 3s, 5s} per the original brief recommendation —
   requires the Jupiter v3.5 multi-head label build to finish first.
3. **Reframe scope**: stop trying to add a direction model on top of CNN-Mamba-v2.
   Empirically CNN-Mamba-v2's own 1s prediction carries the edge; downstream
   classifiers built on its concatenated point estimates appear to lose information.
   Use the v2 raw output + queue-position execution layer directly.

Recommendation: **option 2** (native-horizon rebuild) but DEFER until v3.5 build
completes (Jupiter is busy with it). In the interim, route Razer GPU back to
queue-position execution research (HC #506 R5 FIFO).
