# Residual Learning v1 - REPORT

Generated: 2026-05-22 13:04:29 EDT

- HC: #488 creativity-mandate axis #4 (residual stacking on v3.4.2)
- Split: 16 CAL days / 16 TEST days (chronological, causal)
- Skipped dates: ['20260308', '20260315']
- Total rows: 1,364,517   CAL rows: 905,713   TEST rows: 458,804
- Feature count: 63
- LGBM: 100 trees, 200 leaves, lr=0.05

## VERDICT: REJECT

Stratified winners (corrected preds): **0**   vs baseline (raw-pred stratification): **6**
Pooled deploy-gate winners (corrected): **0**   vs baseline pooled: **0**

## Residual predictability (per horizon x side, OOS R^2)

| horizon   | side   |   n_cal |   n_test |   r2_test |   mae_test |   mae_improvement_vs_zero |
|:----------|:-------|--------:|---------:|----------:|-----------:|--------------------------:|
| 1s        | long   |  613732 |   242105 |   -0.0028 |     1.4272 |                    0.0055 |
| 1s        | short  |  291981 |   216699 |   -0.0116 |     1.4471 |                   -0.0002 |
| 5s        | long   |  612885 |   242010 |   -0.0125 |     2.7793 |                   -0.0142 |
| 5s        | short  |  291630 |   216794 |   -0.0201 |     2.8424 |                   -0.0287 |
| 10s       | long   |  609914 |   239270 |   -0.031  |     3.8372 |                   -0.0524 |
| 10s       | short  |  293797 |   219534 |   -0.0387 |     3.8935 |                   -0.0657 |
| 30s       | long   |  230951 |   158637 |   -0.0763 |     7.1535 |                    0.0163 |
| 30s       | short  |  670696 |   300167 |   -0.0553 |     6.0656 |                   -0.0188 |

- Best R^2: **-0.0028**   Mean R^2: **-0.0310**
- **Verdict: residuals essentially NOISE.** No model edge to rescue.

## Top 5 features explaining residual (aggregate gain across models)

- min_of_day_et  (sum-gain=82,493,965)
- ofi_aggressive_30s  (sum-gain=36,168,649)
- trade_rate_5s  (sum-gain=23,818,230)
- event_rate_5s  (sum-gain=22,992,315)
- ofi_book_30s  (sum-gain=21,573,574)

## Pooled deploy-gate results (corrected preds, top of summary)

| horizon   | side   | conf    |   n_trades |   n_days |   net_ticks |   sharpe |     wr |     pf |   prof_days |   day_conc |   regime_imbalance |   n_gates_failed | passes_all   |
|:----------|:-------|:--------|-----------:|---------:|------------:|---------:|-------:|-------:|------------:|-----------:|-------------------:|-----------------:|:-------------|
| 1s        | long   | top1pc  |       1893 |       13 |      0.9288 |   8.2991 | 0.5547 | 1.697  |      0.6923 |     0.1775 |                  1 |                1 | False        |
| 5s        | short  | top1pc  |       2338 |       13 |      1.3768 |   6.0555 | 0.5697 | 1.3793 |      0.6154 |     0.2232 |                  1 |                1 | False        |
| 10s       | short  | top1pc  |       2430 |       13 |      1.76   |   5.9057 | 0.5654 | 1.3801 |      0.4615 |     0.2481 |                  1 |                2 | False        |
| 5s        | long   | top1pc  |       2261 |       12 |      1.1704 |   5.3191 | 0.5113 | 1.3774 |      0.4167 |     0.3656 |                  1 |                2 | False        |
| 10s       | short  | top5pc  |      12115 |       13 |      0.4601 |   4.2593 | 0.5228 | 1.1182 |      0.4615 |     0.2414 |                  1 |                2 | False        |
| 1s        | long   | top5pc  |       9431 |       13 |      0.154  |   3.6812 | 0.497  | 1.1172 |      0.5385 |     0.2045 |                  1 |                2 | False        |
| 5s        | long   | top5pc  |      11283 |       12 |      0.2698 |   3.1411 | 0.4884 | 1.0931 |      0.3333 |     0.3597 |                  1 |                3 | False        |
| 1s        | long   | top10pc |      18856 |       13 |      0.0585 |   2.2516 | 0.494  | 1.0508 |      0.5385 |     0.3502 |                  1 |                4 | False        |
| 30s       | short  | top10pc |      17024 |       13 |      0.1804 |   2.0079 | 0.4842 | 1.0433 |      0.3077 |     0.1998 |                  1 |                3 | False        |
| 5s        | long   | top10pc |      22560 |       12 |      0.0824 |   1.5777 | 0.4872 | 1.0328 |      0.3333 |     0.2415 |                  1 |                4 | False        |

## Stratified winners (corrected preds, HC #428 pass)

(no stratified cells clear HC #428)

## Closest miss (stratified, corrected preds)

- 10s  top5pc short   midday vd=9: n=1037 net=+7.850t Sh=21.21 fails=[imbalance]
- 10s top10pc short   midday vd=9: n=1433 net=+6.476t Sh=20.80 fails=[imbalance]
- 10s  top1pc short   midday vd=9: n=317 net=+11.802t Sh=19.37 fails=[imbalance]
- 30s top10pc short   midday vd=9: n=576 net=+9.290t Sh=14.13 fails=[imbalance]
-  5s  top1pc short     open vd=9: n=629 net=+3.326t Sh=11.18 fails=[imbalance]

## Interpretation

- Residual correction does NOT expand the winning-cell count beyond baseline 6. v3.4.2 residuals are either pure noise (R^2 ~ 0) or only weakly predictable in a way that doesn't translate to deploy-grade cells under HC #428 gates.
- The R^2 floor confirms v3.4.2 OOT residuals are noise. There is no extractable alpha left in those predictions via post-hoc correction at this feature set.
- This corroborates the conformal-wrapper and TOD x velocity findings: the broken side of v3.4.2 is not rescuable by features we have access to.

