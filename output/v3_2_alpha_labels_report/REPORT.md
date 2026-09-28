# v3.2 Alpha Labels (MFE/MAE within horizon) — STATUS=PARTIAL

- started_at: 2026-05-22T08:54:40.043509+00:00
- finished_at: 2026-05-22T10:20:16.963913+00:00
- git_sha: eecd1803523f34a2008e64019247c71cc0aa1acd
- n_files_in: 140
- n_files_out (ok): 139
- reason: 70 open-market dates with NaN>0.50

## Smoke-test (HC #485 R2)

Dates: ['20251121', '20251225', '20251102']

| date | status | N | elapsed_s |
|------|--------|---|-----------|
| 20251121 | ok | 30384706 | 98.9 |
| 20251225 | ok | 4307 | 0.0 |
| 20251102 | ok | 116727 | 0.3 |

### Smoke-test NaN fractions

| date | column | nan_frac |
|------|--------|----------|
| 20251121 | target_mfe_1s_ticks | 0.000466 |
| 20251121 | target_mae_1s_ticks | 0.000466 |
| 20251121 | target_mfe_5s_ticks | 0.000429 |
| 20251121 | target_mae_5s_ticks | 0.000429 |
| 20251121 | target_mfe_10s_ticks | 0.000447 |
| 20251121 | target_mae_10s_ticks | 0.000447 |
| 20251121 | target_mfe_30s_ticks | 0.000470 |
| 20251121 | target_mae_30s_ticks | 0.000470 |
| 20251121 | target_mfe_60s_ticks | 0.000512 |
| 20251121 | target_mae_60s_ticks | 0.000512 |
| 20251225 | target_mfe_1s_ticks | 1.000000 |
| 20251225 | target_mae_1s_ticks | 1.000000 |
| 20251225 | target_mfe_5s_ticks | 1.000000 |
| 20251225 | target_mae_5s_ticks | 1.000000 |
| 20251225 | target_mfe_10s_ticks | 1.000000 |
| 20251225 | target_mae_10s_ticks | 1.000000 |
| 20251225 | target_mfe_30s_ticks | 1.000000 |
| 20251225 | target_mae_30s_ticks | 1.000000 |
| 20251225 | target_mfe_60s_ticks | 1.000000 |
| 20251225 | target_mae_60s_ticks | 1.000000 |
| 20251102 | target_mfe_1s_ticks | 0.050717 |
| 20251102 | target_mae_1s_ticks | 0.050717 |
| 20251102 | target_mfe_5s_ticks | 0.047196 |
| 20251102 | target_mae_5s_ticks | 0.047196 |
| 20251102 | target_mfe_10s_ticks | 0.047384 |
| 20251102 | target_mae_10s_ticks | 0.047384 |
| 20251102 | target_mfe_30s_ticks | 0.049346 |
| 20251102 | target_mae_30s_ticks | 0.049346 |
| 20251102 | target_mfe_60s_ticks | 0.054966 |
| 20251102 | target_mae_60s_ticks | 0.054966 |

## Corpus summary (HC #485 R3)

| column | avg_nan_frac | worst_nan_frac |
|--------|--------------|----------------|
| target_mfe_1s_ticks | 0.065764 | 1.000000 |
| target_mae_1s_ticks | 0.065764 | 1.000000 |
| target_mfe_5s_ticks | 0.066305 | 1.000000 |
| target_mae_5s_ticks | 0.066305 | 1.000000 |
| target_mfe_10s_ticks | 0.067162 | 1.000000 |
| target_mae_10s_ticks | 0.067162 | 1.000000 |
| target_mfe_30s_ticks | 0.069487 | 1.000000 |
| target_mae_30s_ticks | 0.069487 | 1.000000 |
| target_mfe_60s_ticks | 0.071893 | 1.000000 |
| target_mae_60s_ticks | 0.071893 | 1.000000 |

## Sample distribution medians (date=20251121)

| col | median | p10 | p90 |
|-----|--------|-----|-----|
| target_mfe_1s_ticks | 1.5000 | 0.0000 | 4.5000 |
| target_mae_1s_ticks | -1.5000 | -4.0000 | 0.0000 |
| target_mfe_5s_ticks | 3.0000 | 0.5000 | 9.0000 |
| target_mae_5s_ticks | -3.0000 | -8.5000 | -0.5000 |
| target_mfe_10s_ticks | 4.5000 | 1.0000 | 13.0000 |
| target_mae_10s_ticks | -4.0000 | -12.5000 | -1.0000 |
| target_mfe_30s_ticks | 7.5000 | 1.5000 | 21.5000 |
| target_mae_30s_ticks | -7.0000 | -20.5000 | -1.5000 |
| target_mfe_60s_ticks | 10.5000 | 2.0000 | 30.0000 |
| target_mae_60s_ticks | -10.0000 | -28.0000 | -2.0000 |
