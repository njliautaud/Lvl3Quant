# Prediction Analysis Framework

Comprehensive tools for evaluating model performance beyond IC. Analyzes exploitability, risk/reward, time decay, and regime-conditional performance.

## Overview

This framework provides modular analysis tools to answer critical questions:
- **How exploitable is this model?** (not just IC, but Sharpe, win rate, expectancy)
- **What threshold should I use?** (magnitude filtering, percentile analysis)
- **How long to hold?** (time decay, signal half-life)
- **When does it work?** (regime analysis: volatility, time of day, spread)
- **What's the risk?** (MFE/MAE, optimal stops/targets)

## Modules

### 1. `prediction_loader.py`
Load and standardize prediction files from walk-forward folds.

**Key classes:**
- `PredictionData`: Container for predictions, labels, timestamps, metadata
- `load_npz_predictions()`: Load single .npz file
- `load_model_predictions()`: Load all folds for a model
- `filter_predictions()`: Filter by time range or magnitude

**Example:**
```python
from prediction_loader import load_npz_predictions

pred_data = load_npz_predictions("cnn_s76_concat_oot_predictions.npz", timeframe="10s")
print(pred_data)  # PredictionData(model=cnn_s76, fold=concat, N=316970, IC=0.132)
```

### 2. `mfe_mae_analysis.py`
Maximum Favorable/Adverse Excursion analysis.

**Key metrics:**
- MFE: Maximum favorable excursion (best price before target)
- MAE: Maximum adverse excursion (worst price before target)
- MFE/MAE ratio: Risk/reward
- Optimal stop/target levels

**Example:**
```python
from mfe_mae_analysis import calculate_mfe_mae_simple, analyze_mfe_mae, optimal_stop_target

# Calculate MFE/MAE from tick data
excursion = calculate_mfe_mae_simple(pred_data, tick_data, tick_timestamps, horizon_ns=10e9)

# Analyze
stats = analyze_mfe_mae(excursion)
print(f"Mean MFE: {stats['mean_mfe']:.2f} ticks")
print(f"Mean MAE: {stats['mean_mae']:.2f} ticks")
print(f"Win rate: {stats['win_rate']:.3f}")

# Find optimal stop/target
stop, target = optimal_stop_target(excursion, risk_reward_ratio=2.0)
print(f"Optimal stop: {stop:.2f} ticks, target: {target:.2f} ticks")
```

**Note:** Requires tick-by-tick price data to track path-dependent moves. If you only have labels at fixed horizons, MFE/MAE will be less precise.

### 3. `time_decay_analysis.py`
IC decay over time after prediction.

**Key metrics:**
- IC at multiple horizons (1s, 2s, 5s, 10s, 30s, 60s)
- Half-life of signal
- Optimal holding period
- Prediction stability (reversal rate)

**Example:**
```python
from time_decay_analysis import calculate_time_decay_from_labels, plot_time_decay

# Using pre-computed labels
labels_by_horizon = {
    "1s": labels_1s,
    "5s": labels_5s,
    "10s": labels_10s,
    "30s": labels_30s
}

decay_metrics = calculate_time_decay_from_labels(predictions, labels_by_horizon)
print(f"Peak IC: {np.max(decay_metrics.ic_by_horizon):.4f}")
print(f"Half-life: {decay_metrics.half_life_sec:.1f}s")
print(f"Optimal horizon: {decay_metrics.optimal_horizon_sec:.1f}s")

# Plot
plot_time_decay(decay_metrics, save_path="time_decay.png")
```

### 4. `prediction_distribution.py`
Prediction magnitude distribution and calibration.

**Key analyses:**
- Magnitude distribution (width of predictions)
- Calibration: do larger predictions = larger actual moves?
- Percentile IC (top decile vs bottom decile)
- Threshold sweep: find optimal magnitude threshold

**Example:**
```python
from prediction_distribution import (
    analyze_prediction_magnitude,
    calibration_analysis,
    threshold_sweep_analysis,
    plot_threshold_sweep
)

# Magnitude stats
mag_stats = analyze_prediction_magnitude(pred_data)
print(f"Mean |pred|: {mag_stats['mean_abs']:.3f} ticks")
print(f"P90: {mag_stats['percentiles']['p90']:.3f} ticks")

# Calibration
calib = calibration_analysis(pred_data, num_bins=10)
print(f"Calibration slope: {calib['calibration_slope']:.3f}")  # Should be ~1.0

# Threshold sweep
sweep = threshold_sweep_analysis(pred_data)
print(f"Optimal threshold: {sweep['optimal_threshold']:.3f} ticks")
print(f"Sharpe at threshold: {sweep['optimal_sharpe']:.3f}")

plot_threshold_sweep(sweep, save_path="threshold_sweep.png")
```

### 5. `regime_analysis.py`
IC by regime: volatility, time of day, spread, trend.

**Key analyses:**
- Volatility regime: low/med/high vol
- Time of day: market hours
- Spread regime: tight vs wide spreads
- Trend regime: trending vs mean-reverting
- Multi-regime conditional analysis

**Example:**
```python
from regime_analysis import volatility_regime_analysis, time_of_day_analysis, plot_regime_comparison

# Volatility regime
vol_regime = volatility_regime_analysis(pred_data, tick_data, tick_timestamps, lookback_ns=300e9)
print(f"IC by regime: {vol_regime['ic_by_regime']}")

# Time of day
tod_regime = time_of_day_analysis(pred_data, hour_bins=6)
print(f"IC by hour: {tod_regime['ic_by_hour']}")

plot_regime_comparison(vol_regime, "volatility", save_path="vol_regime.png")
```

### 6. `exploitability_report.py`
**Main entry point**: Comprehensive exploitability report combining all analyses.

**Example:**
```python
from exploitability_report import generate_exploitability_report, compare_models

# Single model report
report = generate_exploitability_report(
    pred_file=Path("cnn_s76_concat_oot_predictions.npz"),
    timeframe="10s",
    output_dir=Path("reports/"),
    include_plots=True
)

# Prints summary and saves JSON + plots
# Output:
#   - exploitability_reports/cnn_s76_concat_exploitability.json
#   - exploitability_reports/cnn_s76_concat_pred_hist.png
#   - exploitability_reports/cnn_s76_concat_calibration.png
#   - exploitability_reports/cnn_s76_concat_threshold.png

# Compare multiple models
reports = compare_models(
    pred_files=[
        Path("cnn_s76_concat_oot_predictions.npz"),
        Path("transformer_v1_concat_oot_predictions.npz")
    ],
    timeframe="10s",
    output_dir=Path("reports/")
)
```

## Quick Start

### 1. Analyze existing predictions

```bash
cd /home/jupiter/Lvl3Quant/analysis
python exploitability_report.py ../data/cnn_s76_concat_oot_predictions.npz 10s
```

This generates:
- Console summary with key metrics and recommendations
- JSON report with all analysis results
- Plots: histogram, calibration, threshold sweep

### 2. Python API

```python
from pathlib import Path
from exploitability_report import generate_exploitability_report

# Load and analyze
report = generate_exploitability_report(
    pred_file=Path("/home/jupiter/Lvl3Quant/data/cnn_s76_concat_oot_predictions.npz"),
    timeframe="10s"
)

# Access results
print(f"IC: {report.ic_metrics['ic']:.4f}")
print(f"Optimal threshold: {report.threshold_sweep['optimal_threshold']:.3f}")
print(f"Recommendations: {report.recommendations}")

# Save JSON
report.save_json(Path("report.json"))
```

### 3. Batch analysis

```python
from pathlib import Path
from glob import glob
from exploitability_report import compare_models

# Find all prediction files
pred_files = [Path(f) for f in glob("/home/jupiter/Lvl3Quant/data/*_concat_oot_predictions.npz")]

# Generate reports for all
reports = compare_models(pred_files, timeframe="10s", output_dir=Path("reports/"))

# Compare
for model_name, report in reports.items():
    print(f"{model_name}: IC={report.ic_metrics['ic']:.4f}, "
          f"Sharpe@optimal={report.threshold_sweep['optimal_sharpe']:.3f}")
```

## Output Format

### JSON Report Structure
```json
{
  "metadata": {
    "model": "cnn_s76",
    "fold": "concat",
    "ic": 0.132,
    "timeframe": "10s"
  },
  "ic_metrics": {
    "ic": 0.132,
    "rank_ic": 0.145,
    "directional_accuracy": 0.623,
    "num_predictions": 316970
  },
  "magnitude_stats": {
    "mean_abs": 0.458,
    "median_abs": 0.312,
    "percentiles": {"p90": 1.234}
  },
  "threshold_sweep": {
    "optimal_threshold": 0.75,
    "optimal_sharpe": 0.842
  },
  "recommendations": {
    "overall": "STRONG SIGNAL - High exploitability",
    "threshold": "Use magnitude threshold >= 0.75 ticks",
    "selectivity": "Trade only top 10% by magnitude (Sharpe=1.2)"
  }
}
```

## Key Metrics Explained

### IC (Information Coefficient)
- Pearson correlation between predictions and labels
- IC > 0.15: Strong signal
- IC > 0.10: Moderate signal
- IC > 0.05: Weak signal
- IC < 0.05: No exploitable signal

### Sharpe Ratio
- Mean return / StdDev of returns
- Sharpe > 1.0: Excellent
- Sharpe > 0.5: Good
- Sharpe > 0.3: Marginal
- Sharpe < 0.3: Not tradeable

### MFE/MAE Ratio
- Risk/reward ratio
- MFE/MAE > 2.0: Favorable risk/reward
- MFE/MAE > 1.5: Acceptable
- MFE/MAE < 1.0: Unfavorable

### Calibration Slope
- Slope of actual vs predicted magnitude
- Slope ~1.0: Well calibrated
- Slope < 0.5: Predictions under-sized
- Slope > 1.5: Predictions over-sized

## Advanced Usage

### Custom Regime Analysis

```python
from regime_analysis import multi_regime_conditional_analysis

# Define custom regimes
regimes = {
    'volatility': vol_regime_assignments,  # (N,) array of regime IDs
    'time_of_day': hour_assignments,
    'spread': spread_regime_assignments
}

# Analyze IC conditioned on all regime combinations
conditional_ic = multi_regime_conditional_analysis(pred_data, regimes)

# Find best regime combination
best_combo = max(conditional_ic.items(), key=lambda x: x[1]['ic'])
print(f"Best regime: {best_combo[0]}, IC={best_combo[1]['ic']:.4f}")
```

### Custom Threshold Optimization

```python
from prediction_distribution import threshold_sweep_analysis

# Custom thresholds
custom_thresholds = np.linspace(0, 2.0, 50)

sweep = threshold_sweep_analysis(pred_data, thresholds=custom_thresholds)

# Find threshold with best Sharpe > X and min trade count
min_trades = 1000
valid_mask = sweep['trade_count_by_threshold'] >= min_trades
valid_sharpes = sweep['sharpe_by_threshold'][valid_mask]
valid_thresholds = sweep['thresholds'][valid_mask]

best_idx = np.nanargmax(valid_sharpes)
print(f"Best threshold: {valid_thresholds[best_idx]:.3f} (Sharpe={valid_sharpes[best_idx]:.3f})")
```

## Dependencies

Required:
- numpy
- scipy
- matplotlib
- pathlib
- logging
- json

Optional for advanced features:
- pandas (for time series analysis)
- seaborn (for enhanced plots)

## Integration with Training Pipeline

The analysis framework is designed to integrate with the training pipeline:

1. Training script saves `.npz` files with predictions, labels, timestamps
2. After training completes, run exploitability report
3. Report informs execution strategy and deployment decisions

**Example integration:**

```python
# In training script (train_event_cnn_1d.py)
np.savez(
    output_path / f"fold_{fold:02d}_oot_predictions.npz",
    preds_1s=preds_1s,
    preds_5s=preds_5s,
    preds_10s=preds_10s,
    labels_1s=labels_1s,
    labels_5s=labels_5s,
    labels_10s=labels_10s,
    timestamps=timestamps,
    concat_ic_10s=concat_ic
)

# After training
from exploitability_report import generate_exploitability_report

report = generate_exploitability_report(
    pred_file=output_path / f"fold_{fold:02d}_oot_predictions.npz",
    timeframe="10s"
)

# Log to MLflow
import mlflow
mlflow.log_metric("optimal_sharpe", report.threshold_sweep['optimal_sharpe'])
mlflow.log_metric("optimal_threshold", report.threshold_sweep['optimal_threshold'])
mlflow.log_artifact(output_path / "exploitability_report.json")
```

## Troubleshooting

### Missing Timestamps
If `pred_data.timestamps is None`, time-dependent analyses (time decay, regime analysis) will fail. Ensure training scripts save timestamps:

```python
np.savez(..., timestamps=timestamps_ns)  # nanosecond timestamps
```

### MFE/MAE Requires Tick Data
MFE/MAE analysis requires high-resolution tick data to track path-dependent moves. If unavailable, use labels at multiple horizons as a proxy.

### Calibration Slope Far from 1.0
- Slope < 0.5: Model predictions too conservative (scale up)
- Slope > 1.5: Model predictions too aggressive (scale down)

Scale predictions before deployment:
```python
calibration_factor = 1.0 / calibration_slope
scaled_preds = predictions * calibration_factor
```

## Future Enhancements

Planned features:
- [ ] Transaction cost analysis (slippage, fees)
- [ ] Portfolio construction (position sizing, risk parity)
- [ ] Drawdown analysis (max drawdown, recovery time)
- [ ] Regime transition analysis (performance when regimes change)
- [ ] Multi-asset correlation analysis
- [ ] Live monitoring dashboard

## Contact

Issues or questions: See project maintainer in `CLAUDE.md`
