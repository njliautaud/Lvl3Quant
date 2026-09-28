# Adverse Selection Feature Study V1

**Config**: tp4sl3_short | **Dates**: 12 days | **Windows**: 502,065

## Outcome Distribution
- Unfilled: 348,199 (69.4%)
- Profitable fills: 65,430 (13.0%)
- Adverse fills: 88,436 (17.6%)
- Fill rate: 30.6% | Adverse rate (of fills): 57.5%

## Top Features Predicting ADVERSE Fills
| Feature | Cohen's d | AUC | Interpretation |
|---------|-----------|-----|----------------|
| win_std_queue_replenishment | +0.0137 | 0.506 | Higher values -> more profitable |
| win_std_rolling_ofi_500 | +0.0135 | 0.505 | Higher values -> more profitable |
| win_trend_buy_sell_intensity | -0.0130 | 0.497 | Higher values -> more adverse |
| win_std_ofi_acceleration | +0.0127 | 0.505 | Higher values -> more profitable |
| win_std_ofi_short_100 | +0.0120 | 0.504 | Higher values -> more profitable |
| win_trend_ofi_short_100 | +0.0107 | 0.503 | Higher values -> more profitable |
| win_trend_side_id | +0.0106 | 0.503 | Higher values -> more profitable |
| win_trend_ofi_acceleration | +0.0104 | 0.503 | Higher values -> more profitable |
| win_mean_spread_velocity_50 | -0.0104 | 0.498 | Higher values -> more adverse |
| win_std_realized_volatility | -0.0101 | 0.497 | Higher values -> more adverse |

## Logistic Regression AUC: 0.5065
Baseline adverse rate: 57.5%

## Key Findings
- Strongest separator: **win_std_queue_replenishment** (|d|=0.014, AUC=0.506)
- Logistic regression with 25 decision-point features achieves AUC=0.507
