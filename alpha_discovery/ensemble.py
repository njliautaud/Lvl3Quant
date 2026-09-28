"""
Ensemble Framework — Combine predictions from multiple model types.

Combines LightGBM (tabular) + CNN/LSTM/Transformer (temporal) predictions
using walk-forward-safe methods (no lookahead).

Methods:
1. Equal weight average
2. IC-weighted average (weight by recent IC)
3. Rank average (convert to ranks first, then average)
4. Stacking (train a meta-learner on OOS predictions)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr, rankdata


def equal_weight_ensemble(predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """Simple average of all model predictions.

    Args:
        predictions: {model_name: prediction_array} — all same length

    Returns:
        Ensemble prediction array
    """
    arrays = list(predictions.values())
    stacked = np.column_stack(arrays)
    return np.nanmean(stacked, axis=1)


def rank_ensemble(predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """Rank-based ensemble — convert each model's predictions to ranks, then average.

    More robust to outliers and different prediction scales.
    """
    ranked = []
    for name, preds in predictions.items():
        valid = np.isfinite(preds)
        ranks = np.full_like(preds, np.nan)
        ranks[valid] = rankdata(preds[valid]) / valid.sum()  # Normalize to [0, 1]
        ranked.append(ranks)

    stacked = np.column_stack(ranked)
    return np.nanmean(stacked, axis=1)


def ic_weighted_ensemble(predictions: Dict[str, np.ndarray],
                          actuals: np.ndarray,
                          lookback_folds: int = 5,
                          fold_ics: Optional[Dict[str, List[float]]] = None) -> np.ndarray:
    """Weight models by their recent IC performance.

    Uses the last `lookback_folds` ICs to compute weights.
    Models with negative IC get zero weight.
    """
    if fold_ics is None:
        # If no fold ICs provided, fall back to equal weight
        return equal_weight_ensemble(predictions)

    weights = {}
    for name in predictions:
        if name in fold_ics and len(fold_ics[name]) >= lookback_folds:
            recent_ics = fold_ics[name][-lookback_folds:]
            w = max(0, np.mean(recent_ics))  # Zero out negative IC models
        else:
            w = 0.0
        weights[name] = w

    total_w = sum(weights.values())
    if total_w == 0:
        return equal_weight_ensemble(predictions)

    # Normalize weights
    for name in weights:
        weights[name] /= total_w

    result = np.zeros(len(next(iter(predictions.values()))), dtype=np.float32)
    for name, preds in predictions.items():
        valid = np.isfinite(preds)
        result[valid] += weights[name] * preds[valid]

    return result


def stacking_ensemble(predictions: Dict[str, np.ndarray],
                       actuals: np.ndarray,
                       train_mask: np.ndarray,
                       test_mask: np.ndarray) -> np.ndarray:
    """Stacking meta-learner — train Ridge regression on OOS predictions.

    Args:
        predictions: {model_name: full_prediction_array}
        actuals: True target values
        train_mask: Boolean mask for meta-learner training
        test_mask: Boolean mask for meta-learner test

    Returns:
        Ensemble predictions for test_mask indices
    """
    from sklearn.linear_model import Ridge

    model_names = sorted(predictions.keys())
    X_meta = np.column_stack([predictions[name] for name in model_names])

    # Valid rows: all models have predictions AND actual is finite
    valid_train = train_mask & np.all(np.isfinite(X_meta), axis=1) & np.isfinite(actuals)
    valid_test = test_mask & np.all(np.isfinite(X_meta), axis=1)

    if valid_train.sum() < 100:
        return equal_weight_ensemble({n: predictions[n][test_mask] for n in model_names})

    meta = Ridge(alpha=1.0)
    meta.fit(X_meta[valid_train], actuals[valid_train])

    result = np.full(test_mask.sum(), np.nan, dtype=np.float32)
    test_indices = np.where(test_mask)[0]
    valid_in_test = valid_test[test_mask]
    result[valid_in_test] = meta.predict(X_meta[test_mask][valid_in_test]).astype(np.float32)

    return result


class WalkForwardEnsemble:
    """Walk-forward ensemble evaluation matching the LightGBM protocol.

    Collects OOS predictions from multiple models across folds,
    then combines them using various ensemble methods.
    """

    def __init__(self):
        self.fold_predictions = {}  # {model_name: {fold_idx: predictions}}
        self.fold_actuals = {}      # {fold_idx: actuals}
        self.fold_ics = {}          # {model_name: [ic_per_fold]}

    def add_fold_predictions(self, model_name: str, fold_idx: int,
                              predictions: np.ndarray, actuals: np.ndarray):
        """Register OOS predictions from a model for a specific fold."""
        if model_name not in self.fold_predictions:
            self.fold_predictions[model_name] = {}
            self.fold_ics[model_name] = []

        self.fold_predictions[model_name][fold_idx] = predictions
        self.fold_actuals[fold_idx] = actuals

        # Compute fold IC
        valid = np.isfinite(predictions) & np.isfinite(actuals)
        if valid.sum() > 10:
            ic = float(spearmanr(predictions[valid], actuals[valid])[0])
            self.fold_ics[model_name].append(ic)

    def evaluate(self, method: str = 'rank') -> Dict:
        """Run ensemble evaluation across all folds.

        Args:
            method: 'equal', 'rank', 'ic_weighted'

        Returns:
            Dict with ensemble IC, ICIR, t-stat, comparison to individual models
        """
        # Collect all folds where ALL models have predictions
        all_folds = set.intersection(*[
            set(self.fold_predictions[m].keys())
            for m in self.fold_predictions
        ])

        if not all_folds:
            return {'error': 'no common folds'}

        ensemble_ics = []
        individual_ics = {m: [] for m in self.fold_predictions}

        for fold in sorted(all_folds):
            actuals = self.fold_actuals[fold]

            # Get predictions from each model
            preds = {
                m: self.fold_predictions[m][fold]
                for m in self.fold_predictions
            }

            # Compute individual ICs
            for m, p in preds.items():
                valid = np.isfinite(p) & np.isfinite(actuals)
                if valid.sum() > 10:
                    ic = float(spearmanr(p[valid], actuals[valid])[0])
                    individual_ics[m].append(ic)

            # Compute ensemble
            if method == 'equal':
                ens_pred = equal_weight_ensemble(preds)
            elif method == 'rank':
                ens_pred = rank_ensemble(preds)
            elif method == 'ic_weighted':
                ens_pred = ic_weighted_ensemble(preds, actuals, fold_ics=self.fold_ics)
            else:
                raise ValueError(f"Unknown method: {method}")

            valid = np.isfinite(ens_pred) & np.isfinite(actuals)
            if valid.sum() > 10:
                ic = float(spearmanr(ens_pred[valid], actuals[valid])[0])
                ensemble_ics.append(ic)

        def _stats(ics):
            if not ics:
                return {'ic': 0, 'icir': 0, 'tstat': 0, 'n': 0}
            arr = np.array(ics)
            m, s = arr.mean(), arr.std()
            return {
                'ic': float(m),
                'icir': float(m / s) if s > 0 else 0,
                'tstat': float(m / s * np.sqrt(len(arr))) if s > 0 else 0,
                'n': len(arr),
            }

        result = {
            'method': method,
            'ensemble': _stats(ensemble_ics),
            'individual': {m: _stats(ics) for m, ics in individual_ics.items()},
            'n_folds': len(all_folds),
            'n_models': len(self.fold_predictions),
        }

        # Determine if ensemble beats best individual
        best_individual = max(
            result['individual'].values(),
            key=lambda x: x['ic']
        )
        result['ensemble_beats_best'] = result['ensemble']['ic'] > best_individual['ic']
        result['improvement_over_best'] = result['ensemble']['ic'] - best_individual['ic']

        return result

    def format_report(self) -> str:
        """Generate comparison report across all ensemble methods."""
        lines = [
            "=" * 60,
            "ENSEMBLE COMPARISON REPORT",
            "=" * 60,
            "",
        ]

        for method in ['equal', 'rank', 'ic_weighted']:
            result = self.evaluate(method)
            if 'error' in result:
                lines.append(f"{method}: ERROR - {result['error']}")
                continue

            ens = result['ensemble']
            lines.append(f"Method: {method.upper()}")
            lines.append(f"  Ensemble: IC={ens['ic']:.4f} ICIR={ens['icir']:.2f} t={ens['tstat']:.2f}")

            for m, stats in sorted(result['individual'].items()):
                marker = " <-- best" if stats['ic'] == max(s['ic'] for s in result['individual'].values()) else ""
                lines.append(f"  {m:20s}: IC={stats['ic']:.4f} ICIR={stats['icir']:.2f}{marker}")

            beat = "YES" if result['ensemble_beats_best'] else "NO"
            lines.append(f"  Ensemble beats best: {beat} (delta={result['improvement_over_best']:+.4f})")
            lines.append("")

        return "\n".join(lines)
