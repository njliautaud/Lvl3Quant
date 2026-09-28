# HC #493 R3 — v7 prod FIFO replay grade

- side: both, top-pct per day: 0.05
- TP=2.0t SL=1.0t hold≤1.5s cancel≤1.0s
- Engine: `alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine` (HC #74)
- v7 prod concat preds: /home/jupiter/Lvl3Quant/output/meta_v7_prod/concat_oot_predictions.npz

| cohort       |     n |   n_days |   mean_tk_net |   sharpe_ann |   sortino_ann |       pf |       wr |   day_pos_pct |
|:-------------|------:|---------:|--------------:|-------------:|--------------:|---------:|---------:|--------------:|
| overall      | 18675 |       17 |     -0.621462 |     -25.3312 |      -25.3312 | 0.33372  | 0.29494  |             0 |
| short_only   |  9497 |       17 |     -0.628132 |     -24.9331 |      -24.9331 | 0.330832 | 0.29146  |             0 |
| long_only    |  9178 |       17 |     -0.61456  |     -25.56   |      -25.56   | 0.336747 | 0.29854  |             0 |
| regime_green | 14540 |       12 |     -0.621392 |     -29.3736 |      -29.3736 | 0.335977 | 0.294498 |             0 |
| regime_red   |  2870 |        4 |     -0.61276  |     -13.5569 |      -13.5569 | 0.336091 | 0.298606 |             0 |
| regime_flat  |  1265 |        1 |     -0.642008 |     nan      |      nan      | 0.301923 | 0.2917   |             0 |