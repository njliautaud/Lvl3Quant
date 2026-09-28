"""
Walk-Forward Trainer for Temporal Models.

Same evaluation protocol as the LightGBM scanner for fair comparison:
- Expanding window: train on days [0..test_day-1], test on test_day
- 1-day purge gap between train and test
- Per-fold IC, ICIR, t-stat metrics
- Feature standardization (required for neural nets, unlike LightGBM)
- Day-boundary-safe windowing (sequences never cross overnight gaps)

Key differences from LightGBM:
- Input is windowed sequences (batch, seq_len, n_features), not flat rows
- Features are z-score standardized per training set (neural nets need this)
- Training uses Adam + early stopping on validation loss
- Gradient clipping for stability
"""

import gc
import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, ttest_1samp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from alpha_discovery.temporal_models import create_model, count_parameters

logger = logging.getLogger("temporal_trainer")


class TemporalWalkForwardEvaluator:
    """
    Walk-forward evaluation for temporal models on MBO features.

    Mirrors the LightGBM MBOAlphaScanner.walk_forward_evaluate() protocol
    but with windowed sequences as input.
    """

    def __init__(
        self,
        features: np.ndarray,
        mid_prices: np.ndarray,
        day_boundaries: List[int],
        seq_len: int = 50,
        stride: int = 10,
        batch_size: int = 256,
        device: str = 'auto',
    ):
        """
        Args:
            features: (N, n_features) flat feature matrix from mbo_features
            mid_prices: (N,) mid prices
            day_boundaries: cumulative day start indices [0, n_day1, n_day1+n_day2, ...]
            seq_len: number of timesteps per input window
            stride: step between windows (controls overlap)
            batch_size: training batch size
            device: 'cuda', 'cpu', or 'auto'
        """
        self.features = features
        self.mid_prices = mid_prices
        self.day_boundaries = day_boundaries
        self.seq_len = seq_len
        self.stride = stride
        self.batch_size = batch_size
        self.n_features = features.shape[1]

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.n_days = len(day_boundaries) - 1
        logger.info(f"TemporalWalkForward: {self.n_days} days, {len(features):,} bars, "
                     f"seq_len={seq_len}, stride={stride}, device={self.device}")

    def _create_windows_for_days(
        self,
        day_indices: List[int],
        target: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create windowed sequences from specified days.

        CRITICAL: Windows never cross day boundaries (no overnight leakage).

        Args:
            day_indices: list of day indices to include
            target: (N,) target values
            mean, std: feature normalization stats (if None, computed from data)

        Returns:
            X_windows: (n_windows, seq_len, n_features) — normalized
            y_windows: (n_windows,) — target at window end
            mean: (n_features,) — feature means used
            std: (n_features,) — feature stds used
        """
        all_X = []
        all_y = []

        for day_idx in day_indices:
            day_start = self.day_boundaries[day_idx]
            day_end = self.day_boundaries[day_idx + 1]
            day_features = self.features[day_start:day_end]
            day_targets = target[day_start:day_end]

            n_day = len(day_features)
            if n_day < self.seq_len + 1:
                continue

            # Create windows within this day
            for start in range(0, n_day - self.seq_len, self.stride):
                end = start + self.seq_len
                target_idx = end - 1  # Target at last bar of window

                # Skip if target is NaN
                if not np.isfinite(day_targets[target_idx]):
                    continue

                # Check if any feature row is entirely NaN (day boundary warmup)
                window = day_features[start:end]
                if np.all(np.isnan(window[-1])):
                    continue

                all_X.append(window)
                all_y.append(day_targets[target_idx])

        if not all_X:
            return None, None, mean, std

        X = np.stack(all_X).astype(np.float32)  # (n_windows, seq_len, n_features)
        y = np.array(all_y, dtype=np.float32)    # (n_windows,)

        # Compute or apply normalization
        if mean is None or std is None:
            # Compute from training data (flatten to 2D for stats)
            X_flat = X.reshape(-1, self.n_features)
            mean = np.nanmean(X_flat, axis=0)
            std = np.nanstd(X_flat, axis=0)
            std[std < 1e-8] = 1.0  # Prevent division by zero

        # Apply normalization
        X = (X - mean[np.newaxis, np.newaxis, :]) / std[np.newaxis, np.newaxis, :]

        # Replace remaining NaN with 0 (neural nets can't handle NaN)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y, mean, std

    def evaluate_model(
        self,
        model_type: str,
        target: np.ndarray,
        target_name: str,
        horizon_name: str,
        model_size: str = 'small',
        min_train_days: int = 5,
        n_epochs: int = 15,
        lr: float = 1e-3,
        patience: int = 5,
        val_fraction: float = 0.15,
    ) -> dict:
        """
        Walk-forward evaluation of a temporal model.

        Same protocol as LightGBM:
        - Expanding window of training days
        - 1-day purge gap
        - Test on single next day
        - Spearman IC as primary metric

        Args:
            model_type: 'cnn', 'lstm', 'transformer'
            target: (N,) target array
            target_name: name for logging
            horizon_name: horizon name for logging
            model_size: 'small', 'medium', 'large'
            min_train_days: minimum training days before first test
            n_epochs: max training epochs per fold
            lr: learning rate
            patience: early stopping patience
            val_fraction: fraction of training data for validation

        Returns:
            dict with IC, ICIR, t-stat, fold metrics, etc.
        """
        label = f"{model_type}_{model_size}_{horizon_name}_{target_name}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-forward: {label}")
        logger.info(f"{'='*60}")

        if self.n_days < min_train_days + 2:
            return {'error': f'Need {min_train_days+2} days, have {self.n_days}',
                    'model_type': model_type, 'horizon': horizon_name,
                    'target': target_name}

        fold_ics = []
        fold_metrics = []
        all_preds = []
        all_actuals = []
        total_train_time = 0

        for test_day in range(min_train_days, self.n_days):
            fold_start = time.time()

            # Training days: 0 to test_day-2 (1-day purge gap)
            train_days = list(range(0, test_day - 1))
            test_days = [test_day]

            if len(train_days) < min_train_days:
                continue

            # Create windows for training data
            X_train, y_train, mean, std = self._create_windows_for_days(
                train_days, target)

            if X_train is None or len(X_train) < 100:
                logger.warning(f"  Fold {test_day}: insufficient training windows, skipping")
                continue

            # Create windows for test data (using training normalization)
            X_test, y_test, _, _ = self._create_windows_for_days(
                test_days, target, mean=mean, std=std)

            if X_test is None or len(X_test) < 10:
                logger.warning(f"  Fold {test_day}: insufficient test windows, skipping")
                continue

            # Split training into train/val (temporal: last val_fraction)
            n_train = len(X_train)
            n_val = max(int(n_train * val_fraction), 10)
            n_train_actual = n_train - n_val

            X_tr = X_train[:n_train_actual]
            y_tr = y_train[:n_train_actual]
            X_val = X_train[n_train_actual:]
            y_val = y_train[n_train_actual:]

            # Create model (fresh each fold)
            model = create_model(
                model_type=model_type,
                n_features=self.n_features,
                n_outputs=1,
                model_size=model_size,
            ).to(self.device)

            if test_day == min_train_days:
                n_params = count_parameters(model)
                logger.info(f"  Model: {model_type} ({model_size}), {n_params:,} params")

            # Train
            preds = self._train_and_predict(
                model, X_tr, y_tr, X_val, y_val, X_test,
                n_epochs=n_epochs, lr=lr, patience=patience,
            )

            fold_time = time.time() - fold_start
            total_train_time += fold_time

            # Compute fold IC
            if len(preds) > 10:
                try:
                    ic_fold = float(spearmanr(preds, y_test)[0])
                    if np.isfinite(ic_fold):
                        fold_ics.append(ic_fold)
                        hr_fold = float((np.sign(preds) == np.sign(y_test)).mean())
                        fold_metrics.append({
                            'day': test_day,
                            'ic': ic_fold,
                            'hit_rate': hr_fold,
                            'n_train': len(X_tr),
                            'n_test': len(X_test),
                            'train_time': fold_time,
                        })
                        logger.info(
                            f"  Fold {test_day}: IC={ic_fold:+.4f} HR={hr_fold:.1%} "
                            f"({len(X_tr)} train, {len(X_test)} test) [{fold_time:.1f}s]"
                        )
                except Exception as e:
                    logger.warning(f"  Fold {test_day}: IC computation failed: {e}")

            all_preds.append(preds)
            all_actuals.append(y_test)

            # Cleanup
            del model, X_train, y_train, X_test, y_test, X_tr, y_tr, X_val, y_val
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        if not all_preds:
            return {'error': 'No valid predictions',
                    'model_type': model_type, 'horizon': horizon_name,
                    'target': target_name}

        # Aggregate results
        predictions = np.concatenate(all_preds)
        actuals = np.concatenate(all_actuals)

        valid = np.isfinite(predictions) & np.isfinite(actuals)
        p, a = predictions[valid], actuals[valid]

        if len(p) < 50:
            return {'error': f'Too few predictions: {len(p)}',
                    'model_type': model_type, 'horizon': horizon_name,
                    'target': target_name}

        # Compute metrics (same as LightGBM scanner)
        ic = float(spearmanr(p, a)[0])
        hr = float((np.sign(p) == np.sign(a)).mean())

        winners = np.abs(a[np.sign(p) == np.sign(a)]).sum()
        losers = np.abs(a[np.sign(p) != np.sign(a)]).sum()
        pf = float(winners / losers) if losers > 0 else 0.0

        if len(fold_ics) > 2:
            ic_mean = float(np.mean(fold_ics))
            ic_std = float(np.std(fold_ics))
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            tstat = ic_mean / ic_std * np.sqrt(len(fold_ics)) if ic_std > 0 else 0.0
            try:
                _, pvalue = ttest_1samp(fold_ics, 0)
                pvalue = float(pvalue)
            except Exception:
                pvalue = 1.0
        else:
            ic_mean, ic_std = ic, 0.0
            icir, tstat, pvalue = 0.0, 0.0, 1.0

        passed = abs(ic) > 0.01 and abs(tstat) > 2.0
        n_params = count_parameters(
            create_model(model_type, self.n_features, 1, model_size)
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"RESULT: {label}")
        logger.info(f"  IC={ic:.4f} ICIR={icir:.2f} t={tstat:.2f} HR={hr:.1%} PF={pf:.2f}")
        logger.info(f"  {len(fold_ics)} folds, {len(p):,} predictions, {n_params:,} params")
        logger.info(f"  Total train time: {total_train_time:.0f}s")
        logger.info(f"  {'PASS' if passed else 'FAIL'}")
        logger.info(f"{'='*60}")

        return {
            'model_type': model_type,
            'model_size': model_size,
            'horizon': horizon_name,
            'target': target_name,
            'n_params': n_params,
            'seq_len': self.seq_len,
            'stride': self.stride,
            # Core metrics
            'ic': ic,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'tstat': tstat,
            'pvalue': pvalue,
            'hit_rate': hr,
            'profit_factor': pf,
            # Counts
            'n_predictions': len(p),
            'n_folds': len(fold_ics),
            # Per-fold
            'fold_metrics': fold_metrics,
            'fold_ics': [float(x) for x in fold_ics],
            # Timing
            'total_train_time': total_train_time,
            'avg_fold_time': total_train_time / max(len(fold_ics), 1),
            # Verdict
            'passed': passed,
        }

    def _train_and_predict(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        n_epochs: int = 15,
        lr: float = 1e-3,
        patience: int = 5,
    ) -> np.ndarray:
        """
        Train model on one fold and return test predictions.

        Uses:
        - AdamW optimizer with weight decay
        - Cosine annealing LR schedule
        - Gradient clipping
        - Early stopping on validation MSE
        """
        # Convert to tensors
        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
        X_v = torch.tensor(X_val, dtype=torch.float32)
        y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
        X_te = torch.tensor(X_test, dtype=torch.float32)

        # DataLoaders
        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            pin_memory=(self.device.type == 'cuda'),
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        # Training loop
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = model(batch_X)
                loss = nn.functional.mse_loss(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = []
                for i in range(0, len(X_v), self.batch_size):
                    batch = X_v[i:i + self.batch_size].to(self.device)
                    val_pred.append(model(batch).cpu())
                val_pred = torch.cat(val_pred)
                val_loss = nn.functional.mse_loss(val_pred, y_v).item()

            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Predict on test set
        model.eval()
        with torch.no_grad():
            test_preds = []
            for i in range(0, len(X_te), self.batch_size):
                batch = X_te[i:i + self.batch_size].to(self.device)
                pred = model(batch).cpu().numpy().flatten()
                test_preds.append(pred)

        return np.concatenate(test_preds)


def format_comparison_report(
    lgbm_results: dict,
    temporal_results: List[dict],
) -> str:
    """
    Format a comparison report between LightGBM baseline and temporal models.
    """
    lines = [
        "",
        "MODEL PROGRESSION — COMPARISON REPORT",
        "=" * 90,
        "",
        f"{'Model':<25s} {'IC':>7s} {'ICIR':>6s} {'t':>6s} {'HR':>6s} "
        f"{'PF':>6s} {'Folds':>5s} {'Params':>8s} {'Time':>7s} {'Pass':>5s}",
        "-" * 90,
    ]

    # LightGBM baseline
    if 'error' not in lgbm_results:
        lines.append(
            f"{'LightGBM (baseline)':<25s} "
            f"{lgbm_results['ic']:>7.4f} {lgbm_results['icir']:>6.2f} "
            f"{lgbm_results['tstat']:>6.2f} {lgbm_results['hit_rate']:>6.1%} "
            f"{lgbm_results['profit_factor']:>6.2f} {lgbm_results['n_folds']:>5d} "
            f"{'tree':>8s} {lgbm_results.get('elapsed_sec', 0):>6.0f}s "
            f"{'YES' if lgbm_results['passed'] else 'no':>5s}"
        )
    else:
        lines.append(f"{'LightGBM (baseline)':<25s}  ERROR: {lgbm_results['error']}")

    lines.append("-" * 90)

    # Temporal models
    for r in sorted(temporal_results, key=lambda x: abs(x.get('ic', 0)), reverse=True):
        if 'error' in r:
            name = f"{r.get('model_type', '?')} ({r.get('model_size', '?')})"
            lines.append(f"{name:<25s}  ERROR: {r['error']}")
            continue

        name = f"{r['model_type']} ({r['model_size']})"
        passed = "YES" if r['passed'] else "no"
        lines.append(
            f"{name:<25s} "
            f"{r['ic']:>7.4f} {r['icir']:>6.2f} "
            f"{r['tstat']:>6.2f} {r['hit_rate']:>6.1%} "
            f"{r['profit_factor']:>6.2f} {r['n_folds']:>5d} "
            f"{r['n_params']:>8,} {r['total_train_time']:>6.0f}s "
            f"{passed:>5s}"
        )

    lines.append("=" * 90)

    # Analysis
    lines.append("\nANALYSIS:")

    # Compare ICs
    lgbm_ic = lgbm_results.get('ic', 0) if 'error' not in lgbm_results else 0
    better_models = [r for r in temporal_results
                     if 'error' not in r and abs(r['ic']) > abs(lgbm_ic)]
    worse_models = [r for r in temporal_results
                    if 'error' not in r and abs(r['ic']) <= abs(lgbm_ic)]

    if better_models:
        lines.append(f"\n  Models BEATING LightGBM baseline (IC={lgbm_ic:.4f}):")
        for r in sorted(better_models, key=lambda x: abs(x['ic']), reverse=True):
            improvement = (abs(r['ic']) - abs(lgbm_ic)) / max(abs(lgbm_ic), 0.001) * 100
            lines.append(
                f"    {r['model_type']} ({r['model_size']}): "
                f"IC={r['ic']:.4f} (+{improvement:.1f}%)"
            )
    else:
        lines.append(f"\n  No temporal model beat LightGBM baseline (IC={lgbm_ic:.4f})")

    if worse_models:
        lines.append(f"\n  Models BELOW LightGBM baseline:")
        for r in worse_models:
            lines.append(
                f"    {r['model_type']} ({r['model_size']}): IC={r['ic']:.4f}"
            )

    # Per-fold comparison
    lines.append("\nPER-FOLD IC EVOLUTION:")
    lines.append("-" * 80)

    if 'error' not in lgbm_results and lgbm_results.get('fold_ics'):
        ics_str = " ".join(f"{x:+.3f}" for x in lgbm_results['fold_ics'][:20])
        lines.append(f"  {'LightGBM':<20s}: [{ics_str}]")

    for r in temporal_results:
        if 'error' in r or not r.get('fold_ics'):
            continue
        name = f"{r['model_type']} ({r['model_size']})"
        ics_str = " ".join(f"{x:+.3f}" for x in r['fold_ics'][:20])
        lines.append(f"  {name:<20s}: [{ics_str}]")

    # Verdict
    lines.append("\nVERDICT:")
    best_temporal = max(
        (r for r in temporal_results if 'error' not in r),
        key=lambda x: abs(x['ic']),
        default=None,
    )

    if best_temporal and abs(best_temporal['ic']) > abs(lgbm_ic) * 1.1:
        lines.append(
            f"  TEMPORAL MODELS WIN: {best_temporal['model_type']} "
            f"(IC={best_temporal['ic']:.4f}) beats LightGBM "
            f"(IC={lgbm_ic:.4f}) by "
            f"{(abs(best_temporal['ic'])/max(abs(lgbm_ic),0.001)-1)*100:.1f}%"
        )
        lines.append(f"  -> Proceed to ensemble/optimization phase")
    elif best_temporal and abs(best_temporal['ic']) > abs(lgbm_ic):
        lines.append(
            f"  MARGINAL: {best_temporal['model_type']} "
            f"(IC={best_temporal['ic']:.4f}) slightly better than LightGBM "
            f"(IC={lgbm_ic:.4f}). Not convincing — need more data."
        )
    else:
        lines.append(
            f"  LIGHTGBM HOLDS: No temporal model significantly beats "
            f"the baseline. Sequential patterns may not exist at this "
            f"horizon, or more data is needed."
        )

    return "\n".join(lines)
