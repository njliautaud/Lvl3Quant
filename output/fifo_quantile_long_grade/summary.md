# HC #493 R3 — Quantile-pinball LONG FIFO Regrade

- side: long, top-pct per day: 0.01
- horizon h=1s, TP=2.0t SL=1.0t hold<=1.5s cancel<=1.0s
- Engine: `alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine` (HC #74)
- Predictions: `output/hc488_dlinear_quantile_v1/fold_{21,22}_preds.npz`
- Dates: 20260427, 20260428 (the two days the +0.65 claim was made on)

## Per-day net (FIFO)

|     date |     n |   sum_ticks |   mean_ticks |       wr |
|---------:|------:|------------:|-------------:|---------:|
| 20260427 | 10278 |    -5760.53 |    -0.560472 | 0.367873 |
| 20260428 | 13683 |    -7431.31 |    -0.543105 | 0.346635 |

## Cohort summary

| cohort        |     n |   n_days |   mean_tk_net |   sharpe_ann |   sortino_ann |         pf |         wr |   day_pos_pct |
|:--------------|------:|---------:|--------------:|-------------:|--------------:|-----------:|-----------:|--------------:|
| overall_long  | 23961 |        2 |    -0.550554  |     -88.6281 |      -88.6281 |   0.338732 |   0.355745 |             0 |
| regime_green  | 10278 |        1 |    -0.560472  |     nan      |      nan      |   0.311013 |   0.367873 |             0 |
| regime_red    |   nan |      nan |   nan         |     nan      |      nan      | nan        | nan        |           nan |
| regime_flat   | 13683 |        1 |    -0.543105  |     nan      |      nan      |   0.358731 |   0.346635 |             0 |
| exit_max_hold |  6952 |        2 |     0.0410742 |      26.4365 |      inf      |   1.26004  |   0.750144 |             1 |
| exit_tp       |  3309 |        2 |     1.624     |      43.8529 |      inf      | inf        |   1        |             1 |
| exit_sl       | 13700 |        2 |    -1.376     |     -66.8037 |      -66.8037 |   0        |   0        |             0 |