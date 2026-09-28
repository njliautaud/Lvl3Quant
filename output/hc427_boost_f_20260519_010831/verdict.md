# Boosting (f) verdict — confidence-conditional ensemble v3.3 / v3.4.2

HC #427 R5 boosting technique #5 (f). Generated: 20260519_010831

## Hypothesis

Boost (a) (uniform 0.5/0.5) tied v3.4.2 SOLO on v3.4.2-basis (11 vs 12) but BEAT v3.3 SOLO (+43% on v3.3-basis). Boost (c) (single global weight sweep) was NULL — no scalar weight beat baseline. Implies the optimal weight varies across the prediction distribution. **Confidence-conditional weighting** lets each model take over where its edge is strongest.

## Confidence proxy

`|v3.4.2 pred_log_ret_5s|` — dominant horizon among the 12 LOO-robust v3.4.2 configs. Quartile bins computed globally across all 241,351 OOT samples.

## Scheme results (v3.4.2 sweep top-20 basis, LOO across 5 OOT days)

| scheme | w33(Q1,Q2,Q3,Q4) | n_robust / 20 | top-trial worst-day Sh | top-trial mean Sh |
|---|---|---:|---:|---:|
| baseline_v342 | (0.0, 0.0, 0.0, 0.0) | 12 | 7.25 | 17.2 |
| baseline_uniform | (0.5, 0.5, 0.5, 0.5) | 11 | 12.28 | 20.1 |
| monotone_v342 | (0.7, 0.5, 0.3, 0.0) | 9 | 10.3 | 22.35 |
| monotone_v33 | (0.0, 0.3, 0.5, 0.7) | 11 | 0.0 | 19.88 |
| extremes_v342 | (0.0, 0.5, 0.5, 0.0) | 12 | 9.57 | 18.0 |
| extremes_v33 | (1.0, 0.5, 0.5, 1.0) | 8 | 6.69 | 15.22 |
| q4_pure_v342 | (0.4, 0.4, 0.4, 0.0) | 11 | 4.6 | 14.93 |
| q4_pure_v33 | (0.4, 0.4, 0.4, 1.0) | 10 | 0.0 | 10.07 |

## Verdict

⚪ **NULL** — best scheme `baseline_v342` matched baseline 12/20 robust. Confidence-conditional weighting offers no boost on v3.4.2-basis configs. (May still help v3.3-basis — separate run.)

## Interpretation

If the winning scheme is `monotone_v342` (low-conf → v3.3, high-conf → v3.4.2): consistent with v3.3 being better calibrated in the tail of weak signals while v3.4.2's sharper predictions dominate high-conviction trades. If `monotone_v33` wins: v3.4.2 over-confident in the tails, v3.3 corrects. If extremes-pure wins: blending hurts at conviction extremes (signal is bimodal). NULL/NEGATIVE → the two models do not separate edge by confidence regime.

## HC #427 R5 counter

Techniques tested so far (per session #67 + this run):
- (a) mean ensemble v3.3+v3.4.2 — ✅ POSITIVE on v3.3 basis (+43% n_robust)
- (b) meta-LGBM gate — ❌ NEGATIVE
- (c) weighted-ensemble sweep — ⚪ NULL
- (f) confidence-conditional ensemble — see verdict above
