# Razer Meta-Model Stream-Continuation Report

Compliance: HC #466 (per-head + confidence cuts + confluence), HC #467
(stream-continuation hold time is OUTPUT), HC #428 R1 (regime gate), HC #344
(day-concentration <= 0.70), HC #468 (Razer GPU on alpha research).

## Setup
- 32 v4 prediction heads -> 3 meta-models trained on Razer RTX 3070 GPU.
- Target: target_log_ret_5s (5s realized log return in ticks).
- Walk-forward: sliding, 15 train days -> 1 test day. 17 OOT days predicted.
- Stream-continuation sweep: 5 conf_cuts x 3 exit_M x 2 exit_floor = 30 configs per model.
- Baseline (CNN-Mamba v4 standalone heads): top Sharpe = -0.349.
- Baseline (screening confluence pair pred_log_ret_1s + pred_p_up_5s): +1.93 ticks at 62.6% hit rate.

## Headline — Top 5 meta-model configs by Sharpe (regime-pass only)

| model | conf cut | exit_M | floor | n_trades | mean hold (s) | WR | mean net ticks | Sharpe | regime_skew | day_conc |
|-------|----------|--------|-------|----------|---------------|----|-----------------|--------|-------------|----------|
| mlp | top 10.0% | 2 | 0.25 | 33839 | 20.5 | 43.9% | -0.030 | -0.084 | 0.30 | 0.19 |
| mlp | top 10.0% | 2 | 0.50 | 35144 | 18.9 | 45.5% | -0.047 | -0.127 | 0.42 | 0.21 |
| xgb | top 20.0% | 2 | 0.25 | 48869 | 24.1 | 40.2% | -0.119 | -0.305 | 0.39 | 0.25 |
| xgb | top 20.0% | 2 | 0.50 | 53473 | 20.3 | 43.4% | -0.145 | -0.367 | 0.36 | 0.24 |
| xgb | top 20.0% | 3 | 0.25 | 35900 | 39.9 | 25.2% | -0.183 | -0.536 | 0.37 | 0.21 |

## Top 5 by Sharpe regardless of gates (for transparency)

| model | conf cut | exit_M | floor | n_trades | WR | mean net ticks | Sharpe | regime_pass | day_conc_pass |
|-------|----------|--------|-------|----------|----|-----------------|--------|--------------|---------------|
| mlp | top 10.0% | 3 | 0.25 | 26224 | 32.3% | 0.007 | 0.020 | no | yes |
| mlp | top 5.0% | 3 | 0.50 | 16242 | 34.4% | 0.004 | 0.010 | no | yes |
| mlp | top 10.0% | 3 | 0.50 | 26823 | 33.8% | -0.002 | -0.004 | no | yes |
| mlp | top 5.0% | 3 | 0.25 | 15845 | 32.0% | -0.004 | -0.011 | no | yes |
| mlp | top 10.0% | 2 | 0.25 | 33839 | 43.9% | -0.030 | -0.084 | yes | yes |

## Best config per model

- **XGB**: conf top 20.0%, exit_M=2, floor=0.25 -> 48869 trades, mean_net=-0.119t, WR=40.2%, Sharpe=-0.305, regime_pass=yes
- **LGBM**: conf top 50.0%, exit_M=2, floor=0.50 -> 81646 trades, mean_net=-0.144t, WR=42.1%, Sharpe=-0.379, regime_pass=no
- **MLP**: conf top 10.0%, exit_M=3, floor=0.25 -> 26224 trades, mean_net=0.007t, WR=32.3%, Sharpe=0.020, regime_pass=no

## Verdict vs baselines

- Standalone-head baseline Sharpe: -0.349
- Best meta-model Sharpe: 0.020 (mlp, conf top 10.0%)
- **Meta-model beats standalone-head baseline: YES**

- Screening-confluence pair: +1.93 ticks mean signed realized (different metric: not per-trade Sharpe).
- Best meta-model mean net ticks per trade: 0.007
- **Meta-model beats confluence-pair mean net ticks: NO**

## Feature importances — XGBoost (top 10 of 32 heads)

| rank | head | importance |
|------|------|------------|
| 1 | pred_log_ret_5s | 0.05724 |
| 2 | pred_log_ret_1s | 0.05480 |
| 3 | pred_log_ret_60s_q50 | 0.03428 |
| 4 | pred_pred_mae_60s_ticks | 0.03398 |
| 5 | pred_p_up_5s | 0.03297 |
| 6 | pred_p_reversal_60s | 0.03250 |
| 7 | pred_log_ret_10s | 0.03250 |
| 8 | pred_log_ret_10s_q50 | 0.03175 |
| 9 | pred_log_ret_60s_q10 | 0.03157 |
| 10 | pred_fifo_tp4sl3_hit_tp | 0.03107 |

## Feature importances — LightGBM (top 10 of 32 heads)

| rank | head | importance |
|------|------|------------|
| 1 | pred_p_up_60s | 1254.35294 |
| 2 | pred_log_ret_60s_q10 | 1166.11765 |
| 3 | pred_p_reversal_60s | 1129.47059 |
| 4 | pred_p_up_30s | 1061.76471 |
| 5 | pred_fifo_tp4sl3_hit_tp | 1053.94118 |
| 6 | pred_log_ret_10s | 1037.94118 |
| 7 | pred_pred_mae_60s_ticks | 971.35294 |
| 8 | pred_log_ret_30s_q50 | 967.17647 |
| 9 | pred_log_ret_60s | 945.76471 |
| 10 | pred_log_ret_10s_q50 | 928.76471 |

---
Wall time: 52.5s. Total config rows: 90.