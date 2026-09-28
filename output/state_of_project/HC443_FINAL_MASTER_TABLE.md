# HC #443 FINAL MASTER TABLE — generated 2026-05-20 00:50:16.228390

## Decision: **❌ FALLBACK — ship live data-collection harness**

Total experiments evaluated: 19

## All runs

| config | side | h | band | TP | SL | hold | cnl | order | n | mean_tk | PF | WR | Sharpe | R1 | R2 | comb |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| hc442_v2_canon_c10 | short | 1 | top0.5 | 3.0 | 0.5 | 1.5 | 10.0 | passive_at_touch | 3455.0 | -0.3126 | 0.514 | 25.12 | -16.87 | True | False | False |
| hc442_v2_canon_c1 | short | 1 | top0.5 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | 2431.0 | -0.3264 | 0.511 | 22.83 | -14.40 | True | True | False |
| hc442_v2_canonical_cancel1 | short | 1 | top0.5 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | 1.0 | -0.8760 | 0.000 | 0.00 | 0.00 | False | True | False |
| hc443_band_top1_short | short | 1 | top1 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | 4189.0 | -0.2914 | 0.547 | 25.11 | -16.84 | True | True | False |
| hc443_band_top5_short | short | 1 | top5 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | 19774.0 | -0.2670 | 0.574 | 26.95 | -33.42 | True | True | False |
| hc443_chase_sl05_tp3_h15 | short | 1 | top0.5 | 3.0 | 0.5 | 1.5 | 1.0 | chase | 2431.0 | -0.3264 | 0.511 | 22.83 | -14.40 | True | True | False |
| hc443_h5_hold5_top0p5_short | short | 5 | top0.5 | 3.0 | 0.5 | 5.0 | 5.0 | passive_at_touch | 2625.0 | -0.2907 | 0.588 | 19.16 | -11.82 | True | True | False |
| hc443_long_top05_baseline | long | 1 | top0.5 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | 2933.0 | -0.3680 | 0.420 | 26.49 | -20.31 | True | True | False |
| hc443_market_entry_tp3 | short | 1 | top0.5 | 3.0 | 0.5 | 1.5 | 1.0 | market | 4299.0 | -0.6764 | 0.161 | 7.28 | -31.45 | False | True | False |
| hc443_multih_1s5s_top10 | short | None | top10 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | nan | +nan | nan | nan | nan | None | None | None |
| hc443_multih_1s5s_top10_v3 | short | None | top10 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | 29995.0 | -0.2612 | 0.581 | nan | -40.21 | None | None | None |
| hc443_multih_confluence_top10 | short | None | top10 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | nan | +nan | nan | nan | nan | None | None | None |
| hc443_multih_confluence_top10_v3b | short | None | top10 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | 27568.0 | -0.2596 | 0.583 | nan | -38.32 | None | None | None |
| hc443_multih_confluence_top30 | short | None | top30 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | nan | +nan | nan | nan | nan | None | None | None |
| hc443_multih_confluence_top30_v2 | short | None | top30 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | nan | +nan | nan | nan | nan | None | None | None |
| hc443_multih_confluence_top30_v3 | short | None | top30 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | 99154.0 | -0.2455 | 0.604 | nan | -67.66 | None | None | None |
| hc443_tight_sl1_tp2_h05 | short | 1 | top0.5 | 2.0 | 1.0 | 0.5 | 0.5 | passive_at_touch | 2076.0 | -0.6366 | 0.303 | 29.58 | -26.21 | True | True | False |
| hc443_upside_sl3_tp8_h10 | short | 1 | top0.5 | 8.0 | 3.0 | 10.0 | 1.0 | passive_at_touch | 2431.0 | -0.7248 | 0.649 | 34.39 | -9.42 | True | False | False |
| hc443_wider_sl2_tp3 | short | 1 | top0.5 | 3.0 | 2.0 | 1.5 | 1.0 | passive_at_touch | 2431.0 | -0.6454 | 0.459 | 38.75 | -17.10 | True | True | False |

## No config passed gates

**Action**: ship live data-collection harness as Friday deliverable.

Best run by mean_tk (still negative):

| config | side | h | band | TP | SL | hold_s | cancel_s | order | n | mean_tk | PF | WR | Sharpe | Sortino | R1 | R2 | combined | g_n | g_tk | r_n | r_tk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hc443_multih_confluence_top30_v3 | short | None | top30 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | 99154.0 | -0.24554131956350722 | 0.6037553074521999 | nan | -67.66441031084288 | nan | None | None | None | nan | nan | nan | nan |
| hc443_multih_confluence_top10_v3b | short | None | top10 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | 27568.0 | -0.259633197910621 | 0.582612687380777 | nan | -38.31936938654192 | nan | None | None | None | nan | nan | nan | nan |
| hc443_multih_1s5s_top10_v3 | short | None | top10 | 3.0 | 0.5 | 1.5 | 1.0 | passive_at_touch | 29995.0 | -0.2611641940323387 | 0.5808301784378087 | nan | -40.21422563252024 | nan | None | None | None | nan | nan | nan | nan |