# Boosting (g) verdict — horizon-stacking (v3.4.2 self-ensemble)

HC #427 R5 boosting technique (g). Generated: 20260519_071556

## Hypothesis

v3.4.2's 1s/5s/10s/30s prediction heads are jointly trained on a shared encoder — they share information but produce distinct horizon-specific outputs. Stacking hypothesis: a longer-horizon decision is sharper when shorter-horizon heads AGREE on direction (cross-horizon confluence). We test by linearly blending shorter heads into each target head and re-running the existing LOO validator. Pure self-ensemble — v3.3 not used here.

## Scheme leaderboard (v3.4.2 sweep top-20 basis, LOO across 5 OOT days)

| scheme | n_robust / 20 | top-trial worst-day Sh | top-trial mean Sh | top fills | top PF |
|---|---:|---:|---:|---:|---:|
| stack_30s_with_10s_5s | 13 | 7.25 | 17.2 | 57 | 0.0 |
| baseline_v342 | 12 | 7.25 | 17.2 | 57 | 0.0 |
| pyramid_10s_full | 12 | 7.25 | 17.2 | 57 | 0.0 |
| pyramid_30s_full | 12 | 7.25 | 17.2 | 57 | 0.0 |
| stack_5s_with_1s | 11 | 7.31 | 15.65 | 50 | 0.0 |
| stack_10s_with_5s_1s | 11 | 7.25 | 17.2 | 57 | 0.0 |
| all_short_to_long_mild | 10 | 5.83 | 14.43 | 55 | 0.0 |
| consensus_4h_uniform | 10 | 8.44 | 26.79 | 45 | 0.0 |

## Verdict

✅ **POSITIVE** — scheme `stack_30s_with_10s_5s` produced **13/20 robust** vs baseline v3.4.2 SOLO 12/20 (Δ = +1).

Top-3 robust configs under winning scheme:
- trial=1110 | 5s/long/passive_at_touch_plus_2 | mean_Sh=17.2 worst_Sh=7.25 fills=57 pf=0.0 prof_days=4
- trial=1554 | 30s/short/passive_at_touch_plus_2 | mean_Sh=25.76 worst_Sh=9.2 fills=90 pf=0.0 prof_days=5
- trial=479 | 5s/long/passive_at_touch_plus_2 | mean_Sh=18.08 worst_Sh=6.69 fills=53 pf=0.0 prof_days=4

## Net-new robust trials (boost-g only)

Trials newly LOO-robust under any boost-g scheme but NOT in v3.4.2 SOLO (12) and NOT already counted under boost-f/h ([563, 1296, 2142, 2326]):
`[]` (count = 0)

## HC #427 R5 boost-counter (updated)

- (a) mean ensemble v3.3+v3.4.2 — ✅ POSITIVE on v3.3 basis (+43% n_robust)
- (b) meta-LGBM gate — ❌ NEGATIVE
- (c) weighted-ensemble sweep — ⚪ NULL
- (f) confidence-conditional ensemble — ⚪ NULL (+1 net-new trial 2142)
- (h) volatility-regime gating — ✅ POSITIVE (+4 net-new trials)
- (g) horizon-stacking — see verdict above (+0 net-new)
