# HC #428 R1 Sweep Per-Day Audit
**Timestamp**: 20260519_104632
**Sweep**: v342_execution_optuna_20260519
**Predictions**: fold_00_predictions.npz (5 OOT dates: ['20260223', '20260224', '20260225', '20260226', '20260227'])

## Caveat
Per-sample date column was unavailable in v3.4.2 NPZ; this audit uses
equal-split fallback (each date ≈ n_total/5 samples). RESULTS ARE
APPROXIMATE; treat as monoculture detector, not as exact metrics replay.

## Regime Classification of Sweep Window
[{'date_s': '20260223', 'close_minus_open_ticks': -222.0, 'trend_label': 'down'}, {'date_s': '20260224', 'close_minus_open_ticks': 173.0, 'trend_label': 'up'}, {'date_s': '20260225', 'close_minus_open_ticks': 133.0, 'trend_label': 'up'}, {'date_s': '20260226', 'close_minus_open_ticks': -193.0, 'trend_label': 'down'}, {'date_s': '20260227', 'close_minus_open_ticks': -39.0, 'trend_label': 'down'}]

## Top-20 Configs Summary
 trial horizon  side  sweep_sharpe  total_fills  day_conc  n_days_with_fills  sharpe_min_day  sharpe_max_day  hc344_day_conc_pass
   929      1s short     35.003253       111811  0.217608                  5       33.330261       44.970434                 True
  3728      1s short     34.857197       111652  0.217596                  5       62.098511       73.083895                 True
  1652      1s short     34.590728       111686  0.217601                  5       56.122810       67.443115                 True
  3380      1s short     32.036685        84869  0.256619                  5       19.097111       34.849686                 True
  1681      1s short     31.598084       112198  0.217464                  5       62.166465       73.054955                 True
  2680      1s short     31.394241       112262  0.217473                  5       45.248422       57.042836                 True
   485      1s short     31.166033       112415  0.217507                  5       55.863817       68.229795                 True
   825      1s short     30.226354        84869  0.256619                  5       28.226513       41.882271                 True
  2835      1s short     29.620049       112365  0.217603                  5        4.607036       20.260303                 True
  4798      1s  long     29.265305       106825  0.209604                  5       40.474184       49.030911                 True
   231      1s short     28.429988        71166  0.293975                  5       11.916718       27.934708                 True
  3775      1s short     27.669913       110647  0.217737                  5       19.680589       33.059621                 True
   669      1s  long     27.489512       106491  0.209661                  5        7.550047       15.004267                 True
   200      1s  long     27.251110        88547  0.213977                  5       38.693945       46.005541                 True
  2593      1s short     26.626614       109137  0.217681                  5       58.364614       68.990814                 True
  3959      1s short     26.613062       112999  0.217321                  5       44.239312       55.893781                 True
   478      1s short     26.606574       110613  0.217705                  5       31.385762       44.527959                 True
  1382      1s  long     26.428444        94260  0.212667                  5       34.322225       41.748636                 True
  1265      1s short     26.340069        99795  0.232256                  5       43.608821       53.881558                 True
  3301      1s short     26.055717       114520  0.216844                  5       44.831037       57.539472                 True

## Gate Results
- HC #344 day-conc ≤ 0.70 passing: 20/20
- Configs with fills on all 5 days: 20/20
- Configs with fills on ≥3 days: 20/20

## Verdict
HC #428 R1 PROPER VALIDATION requires v3.4.2 predictions on the FULL 40+ day
OOT range. The current 5-day NPZ is insufficient. This audit only exposes
intra-sweep-window day-attribution; the deeper monoculture test (sweep window
= all-green or all-red days) requires regime labels of 20260223-0227 (now
running in parallel, PID look in logs/regime_label_sweep_window.log).

**Next step**: re-export v3.4.2 OOT NPZ over the full 40-day window
(per HC #430-A follow-up #3); only then can the sweep configs pass HC #428 R1.