"""
Prediction Loader — Load and standardize prediction files from walk-forward folds

Supports multiple file formats:
- Fold-level predictions: model_fold_XX_oot_predictions.npz
- Concat predictions: model_concat_oot_predictions.npz
- Legacy formats with timestamps embedded in filename

Standardized output:
- predictions: (N,) array of predicted mid-price changes
- labels: (N,) array of actual mid-price changes
- timestamps: (N,) array of nanosecond timestamps (if available)
- metadata: dict with model info, IC, fold info, etc.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import logging

logger = logging.getLogger(__name__)


class PredictionData:
    """Container for prediction data with metadata."""

    def __init__(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ):
        self.predictions = predictions
        self.labels = labels
        self.timestamps = timestamps
        self.metadata = metadata or {}

        # Validate shapes
        assert predictions.shape == labels.shape, "Predictions and labels must have same shape"
        if timestamps is not None:
            assert timestamps.shape[0] == predictions.shape[0], "Timestamps must match prediction count"

    def __len__(self):
        return len(self.predictions)

    def __repr__(self):
        model = self.metadata.get('model', 'unknown')
        fold = self.metadata.get('fold', 'concat')
        ic = self.metadata.get('ic', None)
        ic_str = f"IC={ic:.4f}" if ic is not None else "IC=N/A"
        return f"PredictionData(model={model}, fold={fold}, N={len(self)}, {ic_str})"


def load_npz_predictions(
    file_path: Union[str, Path],
    timeframe: str = "10s"
) -> PredictionData:
    """
    Load predictions from .npz file.

    Args:
        file_path: Path to .npz file
        timeframe: Which timeframe to load ("1s", "5s", "10s", "30s")

    Returns:
        PredictionData object with standardized format

    Expected .npz structure:
        - preds_<timeframe>: (N,) predictions
        - labels_<timeframe>: (N,) labels
        - concat_ic_<timeframe>: () scalar IC (optional)
        - timestamps: (N,) nanosecond timestamps (optional)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {file_path}")

    data = np.load(file_path)

    # Extract predictions and labels
    # Handle two formats:
    # 1. Concat format: preds_1s, preds_5s, preds_10s (separate arrays)
    # 2. Fold format: predictions (N, 3), labels (N, 3), horizons [1, 5, 10]

    pred_key = f"preds_{timeframe}"
    label_key = f"labels_{timeframe}"

    if pred_key in data:
        # Concat format
        predictions = data[pred_key]
        labels = data[label_key]
    elif 'predictions' in data and data['predictions'].ndim == 2:
        # Fold format: (N, num_horizons)
        horizons = data.get('horizons', [1, 5, 10])

        # Map timeframe to index
        timeframe_to_idx = {
            '1s': 0,
            '5s': 1,
            '10s': 2,
            '30s': 3 if len(horizons) > 3 else 2
        }

        if timeframe not in timeframe_to_idx:
            raise ValueError(f"Timeframe {timeframe} not supported for fold format. Available: {list(timeframe_to_idx.keys())}")

        idx = timeframe_to_idx[timeframe]
        if idx >= data['predictions'].shape[1]:
            raise ValueError(f"Timeframe {timeframe} (idx={idx}) not available in predictions shape {data['predictions'].shape}")

        predictions = data['predictions'][:, idx]
        labels = data['labels'][:, idx]
    else:
        raise ValueError(f"Unexpected file format in {file_path}. Available keys: {list(data.keys())}")

    # Extract timestamps if available
    timestamps = data.get('timestamps', None)

    # Extract metadata
    metadata = {}

    # Parse filename for model and fold info
    filename = file_path.stem  # e.g., "cnn_s76_fold_03_oot_predictions"

    # Extract fold number
    fold_match = re.search(r'fold[_\s]+(\d+)', filename)
    if fold_match:
        metadata['fold'] = int(fold_match.group(1))
    elif 'concat' in filename.lower():
        metadata['fold'] = 'concat'
    else:
        metadata['fold'] = 'unknown'

    # Extract model name (everything before fold or concat)
    model_match = re.match(r'^([a-zA-Z0-9_]+)', filename)
    if model_match:
        metadata['model'] = model_match.group(1)
    else:
        metadata['model'] = 'unknown'

    # Extract IC if available
    ic_key = f"concat_ic_{timeframe}"
    if ic_key in data:
        metadata['ic'] = float(data[ic_key])

    # Add file path
    metadata['file_path'] = str(file_path)
    metadata['timeframe'] = timeframe

    logger.info(f"Loaded {len(predictions):,} predictions from {file_path.name} (timeframe={timeframe})")

    return PredictionData(predictions, labels, timestamps, metadata)


def load_model_predictions(
    model_dir: Union[str, Path],
    model_pattern: str = "*_predictions.npz",
    timeframe: str = "10s",
    concat_only: bool = False
) -> List[PredictionData]:
    """
    Load all prediction files for a model.

    Args:
        model_dir: Directory containing prediction .npz files
        model_pattern: Glob pattern to match prediction files
        timeframe: Which timeframe to load
        concat_only: If True, only load concat predictions

    Returns:
        List of PredictionData objects (one per fold, or single concat)
    """
    model_dir = Path(model_dir)

    # Find all matching files
    pred_files = sorted(model_dir.glob(model_pattern))

    if concat_only:
        pred_files = [f for f in pred_files if 'concat' in f.stem.lower()]

    if not pred_files:
        logger.warning(f"No prediction files found in {model_dir} matching {model_pattern}")
        return []

    # Load all files
    results = []
    for pred_file in pred_files:
        try:
            pred_data = load_npz_predictions(pred_file, timeframe)
            results.append(pred_data)
        except Exception as e:
            logger.error(f"Failed to load {pred_file.name}: {e}")
            continue

    logger.info(f"Loaded {len(results)} prediction files from {model_dir}")
    return results


def concat_predictions(pred_list: List[PredictionData]) -> PredictionData:
    """
    Concatenate multiple PredictionData objects into one.

    Args:
        pred_list: List of PredictionData objects

    Returns:
        Single concatenated PredictionData object
    """
    if not pred_list:
        raise ValueError("Empty prediction list")

    if len(pred_list) == 1:
        return pred_list[0]

    # Concatenate arrays
    predictions = np.concatenate([p.predictions for p in pred_list])
    labels = np.concatenate([p.labels for p in pred_list])

    # Concatenate timestamps if all have them
    if all(p.timestamps is not None for p in pred_list):
        timestamps = np.concatenate([p.timestamps for p in pred_list])
    else:
        timestamps = None

    # Merge metadata
    metadata = {
        'model': pred_list[0].metadata.get('model', 'unknown'),
        'fold': 'concat',
        'timeframe': pred_list[0].metadata.get('timeframe', '10s'),
        'num_folds': len(pred_list),
        'source_folds': [p.metadata.get('fold') for p in pred_list]
    }

    # Recalculate IC
    if len(predictions) > 0:
        from scipy.stats import pearsonr
        ic, _ = pearsonr(predictions, labels)
        metadata['ic'] = ic

    return PredictionData(predictions, labels, timestamps, metadata)


def filter_predictions(
    pred_data: PredictionData,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    min_magnitude: Optional[float] = None,
    max_magnitude: Optional[float] = None
) -> PredictionData:
    """
    Filter predictions by time range or magnitude.

    Args:
        pred_data: PredictionData object
        start_time: Start timestamp (nanoseconds)
        end_time: End timestamp (nanoseconds)
        min_magnitude: Minimum absolute prediction magnitude
        max_magnitude: Maximum absolute prediction magnitude

    Returns:
        Filtered PredictionData object
    """
    mask = np.ones(len(pred_data), dtype=bool)

    # Time filter
    if pred_data.timestamps is not None:
        if start_time is not None:
            mask &= pred_data.timestamps >= start_time
        if end_time is not None:
            mask &= pred_data.timestamps < end_time

    # Magnitude filter
    if min_magnitude is not None:
        mask &= np.abs(pred_data.predictions) >= min_magnitude
    if max_magnitude is not None:
        mask &= np.abs(pred_data.predictions) <= max_magnitude

    # Apply filter
    filtered_preds = pred_data.predictions[mask]
    filtered_labels = pred_data.labels[mask]
    filtered_timestamps = pred_data.timestamps[mask] if pred_data.timestamps is not None else None

    # Update metadata
    metadata = pred_data.metadata.copy()
    metadata['filtered'] = True
    metadata['original_count'] = len(pred_data)
    metadata['filtered_count'] = len(filtered_preds)

    logger.info(f"Filtered {len(pred_data):,} -> {len(filtered_preds):,} predictions")

    return PredictionData(filtered_preds, filtered_labels, filtered_timestamps, metadata)
