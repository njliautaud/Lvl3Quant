"""
Prediction Analysis Framework

Comprehensive tools for evaluating model performance beyond IC.

Key modules:
- prediction_loader: Load and standardize prediction files
- mfe_mae_analysis: Maximum favorable/adverse excursion
- time_decay_analysis: IC decay over time
- prediction_distribution: Magnitude, calibration, thresholds
- regime_analysis: Conditional performance by regime
- exploitability_report: Complete analysis framework

Quick start:
    from exploitability_report import generate_exploitability_report
    from pathlib import Path

    report = generate_exploitability_report(
        pred_file=Path("model_concat_oot_predictions.npz"),
        timeframe="10s"
    )

See README.md for detailed documentation.
"""

__version__ = "1.0.0"

from .prediction_loader import (
    PredictionData,
    load_npz_predictions,
    load_model_predictions,
    filter_predictions,
    concat_predictions
)

from .exploitability_report import (
    ExploitabilityReport,
    generate_exploitability_report,
    compare_models
)

__all__ = [
    'PredictionData',
    'load_npz_predictions',
    'load_model_predictions',
    'filter_predictions',
    'concat_predictions',
    'ExploitabilityReport',
    'generate_exploitability_report',
    'compare_models',
]
