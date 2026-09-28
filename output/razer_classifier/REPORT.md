# Razer Classifier Stream-Continuation Report

Compliance: HC #466 (full-output utilization), HC #467 (stream-continuation),
HC #428 R1 (regime gate), HC #344 (day-conc cap), HC #468 (Razer GPU on alpha).

## Setup
- Target: y_profitable_trigger = (any of 4 confluence-pair triggers) AND realized 5s move > 0.5 ticks in predicted direction.
- Class balance: 25,375 positives out of 1,578,006 events (1.61%).
- 32 v4 prediction heads -> 2 classifiers (XGBoost-GPU, PyTorch-MLP) trained on Razer RTX 3070.
- Walk-forward: 15-day-train / 1-day-test sliding, 17 OOT folds.
- Direction: sign(pred_log_ret_1s). Confidence: classifier probability.
- Stream-continuation sweep: 5 conf_cuts x 3 exit_M x 2 exit_floor = 30 configs per model.
- Baselines: standalone-head Sharpe -0.349; meta-regression MLP Sharpe -0.084.

## Headline — Top 5 classifier configs by Sharpe (regime-pass only)

| model | conf cut | exit_M | floor | n_trades | mean hold (s) | WR | mean net ticks | Sharpe | regime_skew | day_conc |
|-------|----------|--------|-------|----------|---------------|----|-----------------|--------|-------------|----------|
| xgb | top 20.0% | 3 | 0.50 | 42737 | 27.7 | 39.2% | -0.204 | -0.525 | 0.45 | 0.15 |
| xgb | top 20.0% | 3 | 0.25 | 41380 | 29.6 | 37.3% | -0.206 | -0.535 | 0.21 | 0.15 |
| xgb | top 10.0% | 3 | 0.50 | 28940 | 24.3 | 42.5% | -0.208 | -0.541 | 0.49 | 0.16 |
| xgb | top 10.0% | 3 | 0.25 | 27970 | 26.4 | 40.6% | -0.212 | -0.555 | 0.29 | 0.17 |
| xgb | top 5.0% | 5 | 0.50 | 14240 | 41.3 | 28.5% | -0.218 | -0.574 | 0.48 | 0.21 |

## Top 5 overall by Sharpe (gates shown)

| model | conf cut | exit_M | floor | n_trades | WR | mean net ticks | Sharpe | regime_pass | day_conc_pass |
|-------|----------|--------|-------|----------|----|-----------------|--------|--------------|---------------|
| xgb | top 1.0% | 3 | 0.25 | 4964 | 47.5% | -0.016 | -0.039 | no | yes |
| xgb | top 1.0% | 3 | 0.50 | 4982 | 47.8% | -0.024 | -0.059 | no | yes |
| xgb | top 1.0% | 2 | 0.25 | 5540 | 50.1% | -0.100 | -0.297 | no | yes |
| xgb | top 1.0% | 2 | 0.50 | 5573 | 50.2% | -0.126 | -0.367 | no | yes |
| xgb | top 1.0% | 5 | 0.25 | 4355 | 33.3% | -0.184 | -0.411 | no | yes |

## Best config per classifier

- **XGB**: conf top 1.0%, exit_M=3, floor=0.25 -> 4964 trades, mean_net=-0.016t, WR=47.5%, Sharpe=-0.039, regime_pass=no
- **MLP**: conf top 1.0%, exit_M=3, floor=0.50 -> 3606 trades, mean_net=-0.191t, WR=36.9%, Sharpe=-0.476, regime_pass=no

## Verdict vs baselines

- Standalone-head baseline Sharpe: -0.349
- Meta-REGRESSION MLP best Sharpe: -0.084
- Best classifier Sharpe: -0.039 (xgb, conf top 1.0%)
- **Classifier beats standalone-head baseline: YES**
- **Classifier beats meta-regression baseline: YES**
- **Classifier reaches positive Sharpe: NO**

## Feature importances — XGBoost Classifier (top 10 of 32 heads)

| rank | head | importance |
|------|------|------------|
| 1 | pred_log_ret_1s | 0.33973 |
| 2 | pred_log_ret_5s | 0.25269 |
| 3 | pred_pred_mfe_30s_ticks | 0.05531 |
| 4 | pred_pred_mae_30s_ticks | 0.03313 |
| 5 | pred_p_up_5s | 0.03222 |
| 6 | pred_p_up_10s | 0.02399 |
| 7 | pred_log_ret_5min | 0.02242 |
| 8 | pred_pred_realized_vol_30s_ticks | 0.01827 |
| 9 | pred_log_ret_60s | 0.01496 |
| 10 | pred_p_reversal_60s | 0.01341 |

---
Wall time: 35.0s. Total config rows: 60.