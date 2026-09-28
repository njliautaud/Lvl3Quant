# Prediction Analysis Framework — Usage Guide

## Quick Command Reference

### 1. Analyze Single Model (Concat)
```bash
cd /home/jupiter/Lvl3Quant/analysis
python exploitability_report.py ../data/cnn_s76_concat_oot_predictions.npz 10s
```

**Output:**
- Console summary with recommendations
- JSON report: `exploitability_reports/cnn_s76_concat_exploitability.json`
- Plots: histogram, calibration, threshold sweep

---

### 2. Batch Analysis (All Models)
```bash
cd /home/jupiter/Lvl3Quant/analysis
python batch_analyze.py --data-dir ../data --output-dir batch_reports
```

**Output:**
- Individual reports for each model
- Summary table: `batch_reports/summary_table.json`
- Comparison table printed to console

**Options:**
```bash
# Analyze specific model only
python batch_analyze.py --model cnn

# Skip plots (faster)
python batch_analyze.py --no-plots

# Include fold-level files (not just concat)
python batch_analyze.py --fold-level
```

---

### 3. Fold Progression Analysis
```bash
cd /home/jupiter/Lvl3Quant/analysis
python batch_analyze.py --fold-progression cnn_s76
```

**Output:**
- IC statistics across folds
- Stability metrics (CV)
- Trend analysis (improving/degrading/stable)

---

### 4. Python API (Programmatic)

#### Load Predictions
```python
from prediction_loader import load_npz_predictions

pred_data = load_npz_predictions(
    "cnn_s76_concat_oot_predictions.npz",
    timeframe="10s"
)

print(pred_data)
# PredictionData(model=cnn_s76, fold=concat, N=316970, IC=0.132)
```

#### Generate Report
```python
from exploitability_report import generate_exploitability_report
from pathlib import Path

report = generate_exploitability_report(
    pred_file=Path("../data/cnn_s76_concat_oot_predictions.npz"),
    timeframe="10s",
    output_dir=Path("reports/"),
    include_plots=True
)

# Access results
print(f"IC: {report.ic_metrics['ic']:.4f}")
print(f"Optimal threshold: {report.threshold_sweep['optimal_threshold']:.3f}")
print(f"Recommendations: {report.recommendations}")
```

#### Compare Models
```python
from exploitability_report import compare_models
from pathlib import Path
from glob import glob

pred_files = [Path(f) for f in glob("../data/*_concat_oot_predictions.npz")]

reports = compare_models(
    pred_files=pred_files,
    timeframe="10s",
    output_dir=Path("comparison_reports/")
)

for model_name, report in reports.items():
    print(f"{model_name}: IC={report.ic_metrics['ic']:.4f}")
```

#### Magnitude Analysis
```python
from prediction_distribution import (
    analyze_prediction_magnitude,
    threshold_sweep_analysis,
    plot_threshold_sweep
)

# Magnitude stats
mag_stats = analyze_prediction_magnitude(pred_data)
print(f"Mean: {mag_stats['mean_abs']:.3f}")
print(f"P90: {mag_stats['percentiles']['p90']:.3f}")

# Find optimal threshold
sweep = threshold_sweep_analysis(pred_data)
print(f"Optimal: {sweep['optimal_threshold']:.3f} ticks")
print(f"Sharpe: {sweep['optimal_sharpe']:.3f}")

# Plot
plot_threshold_sweep(sweep, save_path="threshold.png")
```

#### Time Decay Analysis
```python
from time_decay_analysis import calculate_time_decay_from_labels
import numpy as np

# Load multi-horizon labels
data = np.load("cnn_s76_concat_oot_predictions.npz")
predictions = data['preds_10s']

labels_by_horizon = {
    '1s': data['labels_1s'],
    '5s': data['labels_5s'],
    '10s': data['labels_10s']
}

# Calculate decay
decay_metrics = calculate_time_decay_from_labels(predictions, labels_by_horizon)

print(f"Peak IC: {np.max(decay_metrics.ic_by_horizon):.4f}")
print(f"Optimal horizon: {decay_metrics.optimal_horizon_sec:.1f}s")
print(f"Half-life: {decay_metrics.half_life_sec:.1f}s")
```

#### Regime Analysis
```python
from regime_analysis import (
    time_of_day_analysis,
    plot_regime_comparison
)

# Time of day
tod_regime = time_of_day_analysis(pred_data, hour_bins=6)
print(f"IC by hour: {tod_regime['ic_by_hour']}")

# Plot
plot_regime_comparison(tod_regime, "time_of_day", save_path="tod.png")
```

---

## Example Workflows

### Workflow 1: After Training Completes
```bash
# 1. Generate exploitability report
cd /home/jupiter/Lvl3Quant/analysis
python exploitability_report.py ../data/new_model_concat_oot_predictions.npz 10s

# 2. Review console output for quick assessment
# 3. Check plots in exploitability_reports/
# 4. Read JSON for detailed metrics

# 5. If IC looks promising, run fold progression analysis
python batch_analyze.py --fold-progression new_model
```

### Workflow 2: Model Comparison (Monthly)
```bash
# Generate reports for all models
cd /home/jupiter/Lvl3Quant/analysis
python batch_analyze.py --data-dir ../data --output-dir monthly_reports_2026_04

# Review summary table
cat monthly_reports_2026_04/summary_table.json

# Identify best model
# Deploy to production based on:
# - Highest IC
# - Highest Sharpe at optimal threshold
# - Lowest fold-to-fold IC variance (stable)
```

### Workflow 3: Custom Analysis (Python Script)
```python
#!/usr/bin/env python3
"""Custom analysis: top decile IC for all models."""

from pathlib import Path
from glob import glob
from prediction_loader import load_npz_predictions
from prediction_distribution import percentile_ic_analysis

pred_files = glob("/home/jupiter/Lvl3Quant/data/*_concat_oot_predictions.npz")

results = []

for pred_file in pred_files:
    pred_data = load_npz_predictions(pred_file, timeframe="10s")
    pct_ic = percentile_ic_analysis(pred_data, percentiles=[10])

    top_10_ic = pct_ic['top_10']['ic']
    top_10_sharpe = pct_ic['top_10']['sharpe']

    results.append({
        'model': pred_data.metadata['model'],
        'top_10_ic': top_10_ic,
        'top_10_sharpe': top_10_sharpe
    })

# Sort by top decile IC
results.sort(key=lambda x: x['top_10_ic'], reverse=True)

print("Top Decile Performance:")
for r in results:
    print(f"{r['model']:<30} IC={r['top_10_ic']:.4f}, Sharpe={r['top_10_sharpe']:.3f}")
```

---

## Key Metrics and Thresholds

### IC (Information Coefficient)
- **IC > 0.15**: STRONG — High exploitability
- **IC > 0.10**: MODERATE — Exploitable with careful execution
- **IC > 0.05**: WEAK — Marginal exploitability
- **IC < 0.05**: NO SIGNAL — Not exploitable

### Sharpe Ratio
- **Sharpe > 1.0**: Excellent
- **Sharpe > 0.5**: Good
- **Sharpe > 0.3**: Marginal
- **Sharpe < 0.3**: Not tradeable

### Directional Accuracy
- **Dir Acc > 0.55**: Strong directional signal
- **Dir Acc > 0.52**: Moderate signal
- **Dir Acc < 0.51**: No directional edge

### Calibration Slope
- **Slope ~1.0**: Well calibrated
- **Slope < 0.5**: Under-sized (scale up predictions)
- **Slope > 1.5**: Over-sized (scale down predictions)

### Fold Stability (CV)
- **CV < 0.2**: Very stable across folds
- **CV < 0.4**: Stable
- **CV > 0.5**: Unstable (high variance)

---

## Integration with Training Pipeline

### Step 1: Training Script Saves Predictions
```python
# In train_event_cnn_1d.py (after each fold)
np.savez(
    output_path / f"fold_{fold:02d}_oot_predictions.npz",
    preds_1s=preds_1s,
    preds_5s=preds_5s,
    preds_10s=preds_10s,
    labels_1s=labels_1s,
    labels_5s=labels_5s,
    labels_10s=labels_10s,
    timestamps=timestamps_ns,
    concat_ic_10s=concat_ic
)
```

### Step 2: Generate Exploitability Report
```python
# After training completes
from exploitability_report import generate_exploitability_report

report = generate_exploitability_report(
    pred_file=output_path / "model_concat_oot_predictions.npz",
    timeframe="10s"
)
```

### Step 3: Log to MLflow
```python
import mlflow

mlflow.log_metric("ic", report.ic_metrics['ic'])
mlflow.log_metric("optimal_sharpe", report.threshold_sweep['optimal_sharpe'])
mlflow.log_metric("optimal_threshold", report.threshold_sweep['optimal_threshold'])
mlflow.log_metric("top_10_ic", report.percentile_ic['top_10']['ic'])
mlflow.log_artifact(str(output_path / "exploitability_report.json"))
```

### Step 4: Deployment Decision
```python
# Criteria for deployment:
# 1. IC > 0.10
# 2. Sharpe at optimal threshold > 0.5
# 3. Fold CV < 0.4 (stable)
# 4. Top 10% IC > 0.15

ic = report.ic_metrics['ic']
sharpe = report.threshold_sweep['optimal_sharpe']
# fold_cv from fold_progression_analysis

if ic > 0.10 and sharpe > 0.5:
    print("DEPLOY: Model meets exploitability criteria")
    # Trigger deployment workflow
else:
    print("DO NOT DEPLOY: Insufficient exploitability")
```

---

## Troubleshooting

### Issue: "Key 'preds_10s' not found"
**Cause:** Prediction file uses different format.

**Fix:** Update loader supports both formats. If error persists, check file structure:
```bash
python -c "import numpy as np; d = np.load('file.npz'); print(list(d.keys()))"
```

### Issue: "pred_data.timestamps is None"
**Cause:** Training script didn't save timestamps.

**Impact:** Time-dependent analyses (time decay, regime) will fail.

**Fix:** Update training script to save timestamps:
```python
np.savez(..., timestamps=timestamps_ns)
```

### Issue: Calibration slope far from 1.0
**Cause:** Model predictions not calibrated to actual magnitudes.

**Fix:** Scale predictions before deployment:
```python
calibration_factor = 1.0 / calibration_slope
scaled_preds = predictions * calibration_factor
```

### Issue: High IC but low Sharpe
**Cause:** IC measures correlation, but Sharpe measures risk-adjusted returns. High correlation with low magnitude = low Sharpe.

**Fix:** Use threshold filtering to trade only high-magnitude predictions:
```python
optimal_threshold = report.threshold_sweep['optimal_threshold']
# Trade only |prediction| >= optimal_threshold
```

---

## Advanced Topics

### Custom Regime Analysis
```python
from regime_analysis import multi_regime_conditional_analysis

# Define custom regimes (must be (N,) arrays of integer regime IDs)
regimes = {
    'volatility': vol_regime_assignments,
    'time_of_day': hour_assignments,
    'spread': spread_regime_assignments
}

# Analyze IC conditioned on all combinations
conditional_ic = multi_regime_conditional_analysis(pred_data, regimes)

# Find best combination
best = max(conditional_ic.items(), key=lambda x: x[1]['ic'])
print(f"Best regime: {best[0]}, IC={best[1]['ic']:.4f}")
```

### Transaction Cost Analysis
```python
# Not yet implemented, but planned:
# - Slippage modeling
# - Maker/taker fee impact
# - Fill rate by prediction magnitude
# - Net Sharpe after costs

# Workaround: manually adjust Sharpe estimate
commission_ticks = 0.376  # $4.70 RT / $12.50 per tick (AMP)
crossing_cost_ticks = 1.0  # ~1 tick for market order (measure from data!)

total_cost = commission_ticks + crossing_cost_ticks  # 1.376 ticks for market orders

# Adjust returns
adjusted_returns = pred_data.predictions * pred_data.labels - total_cost
adjusted_sharpe = np.mean(adjusted_returns) / np.std(adjusted_returns)
```

---

## File Locations

- **Source code**: `/home/jupiter/Lvl3Quant/analysis/`
- **Example reports**: `/home/jupiter/Lvl3Quant/analysis/example_reports/`
- **Batch reports**: `/home/jupiter/Lvl3Quant/analysis/batch_reports/`
- **Prediction files**: `/home/jupiter/Lvl3Quant/data/`

---

## Support

For issues or questions, see project maintainer in `/home/jupiter/teleclaude-main/CLAUDE.md`
