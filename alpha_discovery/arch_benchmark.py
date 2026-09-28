#!/usr/bin/env python3
"""
Architecture Benchmark — Head-to-head comparison of model architectures
for MBO microstructure alpha prediction.

Uses identical walk-forward methodology across all models:
  - Expanding window training with 1-day purge gap
  - 80/20 internal split for early stopping
  - MFE_net target at specified horizon
  - Metrics: IC, ICIR, HR, PF, training time

Architectures tested:
  1. LightGBM   (baseline, GPU)
  2. XGBoost     (GPU tree method)
  3. CatBoost    (symmetric trees)
  4. MLP         (PyTorch, GPU)
  5. TCN         (Temporal Conv Net, PyTorch, GPU)
  6. TabNet      (Attention-based, PyTorch, GPU)

Usage:
  # Quick 20-day test of all architectures
  python arch_benchmark.py --n-days 20 --arch all

  # Full 70-day test of specific architecture
  python arch_benchmark.py --n-days 70 --arch xgb

  # Server mode (CPU only)
  python arch_benchmark.py --n-days 70 --arch xgb,catboost --device cpu

  # List available architectures
  python arch_benchmark.py --list
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, ttest_1samp

# ============================================================
# Logging
# ============================================================

def clean_features(X):
    """Replace NaN/inf in features with 0. GBMs handle NaN natively but PyTorch crashes."""
    X = X.copy()
    mask = ~np.isfinite(X)
    if mask.any():
        X[mask] = 0.0
    return X
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('arch_benchmark')

# ============================================================
# Model Wrapper Base Class
# ============================================================
class ModelWrapper:
    """Base class for all model architectures."""
    name: str = "base"
    supports_gpu: bool = False

    def __init__(self, n_features: int, device: str = 'gpu'):
        self.n_features = n_features
        self.device = device

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cleanup(self):
        """Free model memory."""
        pass


# ============================================================
# 1. LightGBM
# ============================================================
class LightGBMWrapper(ModelWrapper):
    name = "lightgbm"
    supports_gpu = True

    def __init__(self, n_features, device='gpu'):
        super().__init__(n_features, device)
        import lightgbm as lgb
        self.lgb = lgb
        params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 100,
            'verbose': -1,
            'n_jobs': -1,
            'objective': 'regression',
            'metric': 'rmse',
        }
        if device == 'gpu':
            params['device'] = 'gpu'
            params['max_bin'] = 63
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[self.lgb.early_stopping(30, verbose=False)],
        )

    def predict(self, X):
        return self.model.predict(X)

    def cleanup(self):
        del self.model
        gc.collect()


# ============================================================
# 2. XGBoost
# ============================================================
class XGBoostWrapper(ModelWrapper):
    name = "xgboost"
    supports_gpu = True

    def __init__(self, n_features, device='gpu'):
        super().__init__(n_features, device)
        import xgboost as xgb
        self.xgb = xgb
        params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 100,
            'verbosity': 0,
            'n_jobs': -1,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'early_stopping_rounds': 30,
        }
        if device == 'gpu':
            params['tree_method'] = 'gpu_hist'
            params['device'] = 'cuda'
        else:
            params['tree_method'] = 'hist'
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    def predict(self, X):
        return self.model.predict(X)

    def cleanup(self):
        del self.model
        gc.collect()


# ============================================================
# 3. CatBoost
# ============================================================
class CatBoostWrapper(ModelWrapper):
    name = "catboost"
    supports_gpu = True

    def __init__(self, n_features, device='gpu'):
        super().__init__(n_features, device)
        from catboost import CatBoostRegressor
        params = {
            'iterations': 300,
            'depth': 6,
            'learning_rate': 0.05,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'rsm': 0.7,  # colsample_bytree equivalent
            'l2_leaf_reg': 1.0,
            'min_data_in_leaf': 100,
            'verbose': 0,
            'loss_function': 'RMSE',
            'early_stopping_rounds': 30,
            'thread_count': -1,
        }
        if device == 'gpu':
            params['task_type'] = 'GPU'
            del params['rsm']  # rsm not supported on GPU
        self.model = CatBoostRegressor(**params)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False,
        )

    def predict(self, X):
        return self.model.predict(X)

    def cleanup(self):
        del self.model
        gc.collect()


# ============================================================
# 4. MLP (PyTorch)
# ============================================================
class MLPWrapper(ModelWrapper):
    name = "mlp"
    supports_gpu = True

    def __init__(self, n_features, device='gpu'):
        super().__init__(n_features, device)
        import torch
        import torch.nn as nn
        self.torch = torch
        self.nn = nn
        self.torch_device = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')

        # Build MLP: input -> 512 -> 256 -> 128 -> 1
        self.model = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        ).to(self.torch_device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.scaler = None  # Will be set during fit for feature normalization

    def _normalize(self, X, fit=False):
        X = clean_features(X)
        if fit:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0) + 1e-8
        result = (X - self.mean) / self.std
        result = np.clip(result, -10, 10)  # Prevent extreme values
        return result.astype(np.float32)

    def fit(self, X_train, y_train, X_val, y_val, max_epochs=50, batch_size=8192):
        torch = self.torch

        # Normalize features
        X_train_n = self._normalize(X_train, fit=True)
        X_val_n = self._normalize(X_val)

        # Convert to tensors
        X_tr = torch.tensor(X_train_n, dtype=torch.float32, device=self.torch_device)
        y_tr = torch.tensor(y_train.astype(np.float32), dtype=torch.float32, device=self.torch_device).unsqueeze(1)
        X_va = torch.tensor(X_val_n, dtype=torch.float32, device=self.torch_device)
        y_va = torch.tensor(y_val.astype(np.float32), dtype=torch.float32, device=self.torch_device).unsqueeze(1)

        best_val_loss = float('inf')
        patience = 7
        patience_counter = 0
        best_state = None

        dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = torch.nn.functional.mse_loss(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_va)
                val_loss = torch.nn.functional.mse_loss(val_pred, y_va).item()
            self.model.train()

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()

        # Free training tensors
        del X_tr, y_tr, X_va, y_va, dataset, loader
        torch.cuda.empty_cache() if self.torch_device.type == 'cuda' else None

    def predict(self, X):
        torch = self.torch
        X_n = self._normalize(X)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_n, dtype=torch.float32, device=self.torch_device)
            # Predict in chunks to avoid OOM
            chunk_size = 100000
            preds = []
            for i in range(0, len(X_t), chunk_size):
                chunk = X_t[i:i+chunk_size]
                p = self.model(chunk).cpu().numpy().flatten()
                preds.append(p)
            del X_t
        return np.concatenate(preds)

    def cleanup(self):
        del self.model, self.optimizer, self.scheduler
        self.torch.cuda.empty_cache() if self.torch_device.type == 'cuda' else None
        gc.collect()


# ============================================================
# 5. TCN (Temporal Convolutional Network)
# ============================================================
class TCNWrapper(ModelWrapper):
    name = "tcn"
    supports_gpu = True

    def __init__(self, n_features, device='gpu', window_size=20, top_k_features=80):
        super().__init__(n_features, device)
        import torch
        import torch.nn as nn
        self.torch = torch
        self.nn = nn
        self.window_size = window_size
        self.top_k = min(top_k_features, n_features)
        self.torch_device = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')
        self.feature_indices = None  # Will be set based on LGB importance

        # TCN architecture: 1D convolutions over time dimension
        # Input: (batch, top_k, window_size) → permute to (batch, top_k, window_size)
        n_in = self.top_k
        self.model = nn.Sequential(
            # Temporal conv block 1
            nn.Conv1d(n_in, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Temporal conv block 2
            nn.Conv1d(128, 64, kernel_size=3, padding=1, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Temporal conv block 3
            nn.Conv1d(64, 32, kernel_size=3, padding=1, dilation=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Global average pool + FC
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        ).to(self.torch_device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

    def _select_features(self, X_train, y_train):
        """Select top-K features using a quick LightGBM for importance ranking."""
        import lightgbm as lgb
        # Quick 50-tree LGB to rank features
        quick_model = lgb.LGBMRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            verbose=-1, n_jobs=-1,
        )
        # Use a small sample for speed
        n_sample = min(200000, len(X_train))
        idx = np.random.choice(len(X_train), n_sample, replace=False)
        quick_model.fit(X_train[idx], y_train[idx])
        importances = quick_model.feature_importances_
        self.feature_indices = np.argsort(importances)[-self.top_k:]
        del quick_model
        gc.collect()

    def _create_windows(self, X, day_boundaries=None):
        """Create rolling window sequences. Shape: (N - window + 1, top_k, window)."""
        X_sel = X[:, self.feature_indices]
        N = len(X_sel)
        W = self.window_size

        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(X_sel, window_shape=W, axis=0)
        # Shape: (N-W+1, top_k, W)
        return windows

    def _normalize(self, X, fit=False):
        X = clean_features(X)
        if fit:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0) + 1e-8
        result = (X - self.mean) / self.std
        result = np.clip(result, -10, 10)
        return result.astype(np.float32)

    def fit(self, X_train, y_train, X_val, y_val, max_epochs=30, batch_size=4096):
        torch = self.torch

        # Select top features
        if self.feature_indices is None:
            self._select_features(X_train, y_train)

        # Normalize
        X_train_n = self._normalize(X_train, fit=True)
        X_val_n = self._normalize(X_val)

        # Create windows
        W = self.window_size
        X_tr_win = self._create_windows(X_train_n)  # (N-W+1, top_k, W)
        y_tr = y_train[W-1:]  # Align target with last bar of window

        X_va_win = self._create_windows(X_val_n)
        y_va = y_val[W-1:]

        # Trim to match
        min_tr = min(len(X_tr_win), len(y_tr))
        X_tr_win = X_tr_win[:min_tr]
        y_tr = y_tr[:min_tr]

        min_va = min(len(X_va_win), len(y_va))
        X_va_win = X_va_win[:min_va]
        y_va = y_va[:min_va]

        # Convert to tensors
        X_tr_t = torch.tensor(X_tr_win, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        X_va_t = torch.tensor(X_va_win, dtype=torch.float32, device=self.torch_device)
        y_va_t = torch.tensor(y_va, dtype=torch.float32, device=self.torch_device).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                              pin_memory=True, num_workers=0)

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.torch_device)
                y_batch = y_batch.to(self.torch_device)
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = torch.nn.functional.mse_loss(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = []
                for i in range(0, len(X_va_t), batch_size):
                    vp = self.model(X_va_t[i:i+batch_size])
                    val_preds.append(vp)
                val_pred = torch.cat(val_preds)
                val_loss = torch.nn.functional.mse_loss(val_pred, y_va_t).item()
            self.model.train()

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()

        del X_tr_t, y_tr_t, X_va_t, y_va_t, dataset, loader
        torch.cuda.empty_cache() if self.torch_device.type == 'cuda' else None

    def predict(self, X):
        torch = self.torch
        X_n = self._normalize(X)
        X_win = self._create_windows(X_n)  # (N-W+1, top_k, W)

        self.model.eval()
        chunk_size = 50000
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_win), chunk_size):
                chunk = torch.tensor(X_win[i:i+chunk_size], dtype=torch.float32,
                                   device=self.torch_device)
                p = self.model(chunk).cpu().numpy().flatten()
                preds.append(p)
                del chunk

        result = np.concatenate(preds)
        # Pad front with NaN (first window_size-1 bars have no prediction)
        padded = np.full(len(X), np.nan, dtype=np.float32)
        padded[self.window_size - 1:self.window_size - 1 + len(result)] = result
        return padded

    def cleanup(self):
        del self.model, self.optimizer, self.scheduler
        self.torch.cuda.empty_cache() if self.torch_device.type == 'cuda' else None
        gc.collect()


# ============================================================
# 6. TabNet
# ============================================================
class TabNetWrapper(ModelWrapper):
    name = "tabnet"
    supports_gpu = True

    def __init__(self, n_features, device='gpu'):
        super().__init__(n_features, device)
        self.torch_device = 'cuda' if device == 'gpu' else 'cpu'
        self.model = None
        self.mean = None
        self.std = None

    def _normalize(self, X, fit=False):
        X = clean_features(X)
        if fit:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0) + 1e-8
        result = (X - self.mean) / self.std
        result = np.clip(result, -10, 10)
        return result.astype(np.float32)

    def fit(self, X_train, y_train, X_val, y_val):
        from pytorch_tabnet.tab_model import TabNetRegressor

        X_train_n = self._normalize(X_train, fit=True)
        X_val_n = self._normalize(X_val)

        self.model = TabNetRegressor(
            n_d=32,
            n_a=32,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-4,
            optimizer_fn=__import__('torch').optim.AdamW,
            optimizer_params={'lr': 1e-3, 'weight_decay': 1e-4},
            scheduler_fn=__import__('torch').optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={'mode': 'min', 'factor': 0.5, 'patience': 3},
            mask_type='entmax',
            device_name=self.torch_device,
            verbose=0,
        )

        self.model.fit(
            X_train=X_train_n.astype(np.float32),
            y_train=y_train.reshape(-1, 1).astype(np.float32),
            eval_set=[(X_val_n.astype(np.float32), y_val.reshape(-1, 1).astype(np.float32))],
            eval_metric=['rmse'],
            max_epochs=50,
            patience=7,
            batch_size=8192,
            virtual_batch_size=2048,
        )

    def predict(self, X):
        X_n = self._normalize(X)
        return self.model.predict(X_n.astype(np.float32)).flatten()

    def cleanup(self):
        del self.model
        import torch
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================
# Architecture Registry
# ============================================================
ARCHITECTURES = {
    'lgbm':     LightGBMWrapper,
    'xgb':      XGBoostWrapper,
    'catboost': CatBoostWrapper,
    'mlp':      MLPWrapper,
    'tcn':      TCNWrapper,
    'tabnet':   TabNetWrapper,
}


# ============================================================
# Walk-Forward Evaluator (shared across all architectures)
# ============================================================
def walk_forward_benchmark(
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: List[int],
    model_class,
    device: str = 'gpu',
    min_train_days: int = 5,
    n_features: int = 0,
) -> dict:
    """
    Run walk-forward evaluation for a given model architecture.

    Identical methodology to MBOAlphaScanner.walk_forward_evaluate():
    - Expanding window training
    - 1-day purge gap
    - 80/20 internal split for early stopping
    """
    n_days = len(day_boundaries) - 1
    if n_days < min_train_days + 1:
        return {'error': f'Need {min_train_days+1} days, have {n_days}'}

    n_feat = n_features or features.shape[1]

    all_preds = []
    all_actuals = []
    fold_ics = []
    total_train_time = 0.0
    n_folds = 0

    for test_day in range(min_train_days, n_days):
        train_end_day = test_day - 1  # 1-day purge gap
        train_start = day_boundaries[0]
        train_end = day_boundaries[train_end_day + 1]
        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]

        X_train = features[train_start:train_end]
        y_train = target[train_start:train_end]
        X_test = features[test_start:test_end]
        y_test = target[test_start:test_end]

        # Remove NaN targets
        train_valid = np.isfinite(y_train)
        test_valid = np.isfinite(y_test)

        if train_valid.sum() < 500 or test_valid.sum() < 100:
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_te = X_test[test_valid]
        y_te = y_test[test_valid]

        # 80/20 internal split for early stopping
        split = int(len(X_tr) * 0.8)

        try:
            t0 = time.time()
            model = model_class(n_feat, device)
            model.fit(X_tr[:split], y_tr[:split], X_tr[split:], y_tr[split:])
            train_time = time.time() - t0
            total_train_time += train_time

            preds = model.predict(X_te)

            # Handle TCN NaN padding
            valid_preds = np.isfinite(preds)
            if not valid_preds.all():
                preds = preds[valid_preds]
                y_te = y_te[valid_preds]

            all_preds.append(preds)
            all_actuals.append(y_te)

            # Per-fold IC
            if len(preds) > 50:
                fold_ic = float(spearmanr(preds, y_te)[0])
                if np.isfinite(fold_ic):
                    fold_ics.append(fold_ic)

            n_folds += 1
            model.cleanup()
            del model

            if n_folds <= 3 or n_folds % 5 == 0:
                logger.info(
                    f"  Fold {n_folds} (day {test_day}): "
                    f"IC={fold_ics[-1]:.4f}, "
                    f"train={train_time:.0f}s"
                )

        except Exception as e:
            logger.warning(f"  Fold day {test_day} failed: {e}")
            continue

    if not all_preds:
        return {'error': 'No valid predictions'}

    # Compute metrics
    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)

    valid = np.isfinite(predictions) & np.isfinite(actuals)
    p, a = predictions[valid], actuals[valid]

    if len(p) < 50:
        return {'error': f'Too few predictions: {len(p)}'}

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
    else:
        ic_mean = ic
        icir, tstat = 0.0, 0.0

    return {
        'ic': ic,
        'icir': icir,
        'tstat': tstat,
        'hit_rate': hr,
        'profit_factor': pf,
        'n_predictions': len(p),
        'n_folds': n_folds,
        'total_train_time': total_train_time,
        'avg_fold_time': total_train_time / n_folds if n_folds > 0 else 0,
        'fold_ics': fold_ics,
    }


# ============================================================
# Data Loading (reuses existing infrastructure)
# ============================================================
def load_data(feature_cache_dir: str, snapshot_cache_dir: str,
              n_days: int = 20, horizon: str = 'ret_10s') -> dict:
    """Load features and compute MFE targets."""

    # Add parent to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from mbo_alpha_scan import MBOAlphaScanner
    from run_mfe_scan import compute_mfe_targets

    logger.info(f"Loading {n_days} days of features from cache...")
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    load_info = scanner.load_precomputed_features(
        feature_cache_dir=feature_cache_dir,
        snapshot_cache_dir=snapshot_cache_dir,
        n_days=n_days,
        extra_cols=0,
    )

    # Compute MFE target
    hz_sec = int(horizon.replace('ret_', '').replace('s', ''))
    logger.info(f"Computing MFE targets for {horizon} ({hz_sec}s)...")
    mfe_targets = compute_mfe_targets(
        scanner.mid_prices,
        scanner.day_boundaries,
        sample_interval_ms=100,
        horizons_sec={f'{hz_sec}s': hz_sec},
        tick_size=0.25,
    )

    target_key = f'mfe_net_{hz_sec}s'
    target = mfe_targets[target_key]

    logger.info(f"Data loaded: {scanner.features.shape[0]:,} bars, "
                f"{scanner.features.shape[1]} features, {n_days} days")

    return {
        'features': scanner.features,
        'target': target,
        'day_boundaries': scanner.day_boundaries,
        'feature_names': scanner.feature_names,
        'n_days': n_days,
        'horizon': horizon,
    }


# ============================================================
# Main Benchmark Runner
# ============================================================
def run_benchmark(args):
    """Run the full architecture comparison."""

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = results_dir / f'arch_benchmark_{timestamp}.log'

    # Add file handler
    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(message)s', '%H:%M:%S'))
    logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info("ARCHITECTURE BENCHMARK")
    logger.info(f"Architectures: {args.arch}")
    logger.info(f"Days: {args.n_days}")
    logger.info(f"Horizon: {args.horizon}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    # Load data once
    data = load_data(
        feature_cache_dir=args.feature_cache,
        snapshot_cache_dir=args.snapshot_cache,
        n_days=args.n_days,
        horizon=args.horizon,
    )

    features = data['features']
    target = data['target']
    day_boundaries = data['day_boundaries']

    # Determine which architectures to run
    if args.arch == 'all':
        arch_list = list(ARCHITECTURES.keys())
    else:
        arch_list = [a.strip() for a in args.arch.split(',')]

    # Filter by device compatibility
    if args.device == 'cpu':
        # All architectures support CPU
        pass

    results = {}

    for arch_name in arch_list:
        if arch_name not in ARCHITECTURES:
            logger.warning(f"Unknown architecture: {arch_name}, skipping")
            continue

        model_class = ARCHITECTURES[arch_name]

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"TESTING: {arch_name.upper()}")
        logger.info("=" * 60)

        t0 = time.time()
        try:
            result = walk_forward_benchmark(
                features=features,
                target=target,
                day_boundaries=day_boundaries,
                model_class=model_class,
                device=args.device,
                min_train_days=args.min_train_days,
                n_features=features.shape[1],
            )
            result['wall_time'] = time.time() - t0
            result['architecture'] = arch_name
            results[arch_name] = result

            if 'error' not in result:
                logger.info(f"\n  {arch_name.upper()} RESULTS:")
                logger.info(f"  IC={result['ic']:.4f}  ICIR={result['icir']:.2f}  "
                          f"t={result['tstat']:.2f}")
                logger.info(f"  HR={result['hit_rate']:.1%}  PF={result['profit_factor']:.2f}")
                logger.info(f"  Time: {result['wall_time']:.0f}s total, "
                          f"{result['avg_fold_time']:.0f}s/fold")
            else:
                logger.error(f"  {arch_name} FAILED: {result['error']}")

        except Exception as e:
            logger.error(f"  {arch_name} CRASHED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[arch_name] = {'error': str(e), 'architecture': arch_name}

        # Force cleanup between models
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ============================================================
    # Print comparison table
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("ARCHITECTURE COMPARISON TABLE")
    logger.info("=" * 80)
    logger.info(f"{'Architecture':<12} {'IC':>8} {'ICIR':>8} {'t-stat':>8} "
                f"{'HR':>8} {'PF':>8} {'Time(s)':>10} {'Status':>10}")
    logger.info("-" * 80)

    # Sort by IC descending
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('ic', -999),
        reverse=True,
    )

    for arch_name, result in sorted_results:
        if 'error' in result:
            logger.info(f"{arch_name:<12} {'--':>8} {'--':>8} {'--':>8} "
                       f"{'--':>8} {'--':>8} {'--':>10} {'FAILED':>10}")
        else:
            logger.info(
                f"{arch_name:<12} {result['ic']:>8.4f} {result['icir']:>8.2f} "
                f"{result['tstat']:>8.2f} {result['hit_rate']:>7.1%} "
                f"{result['profit_factor']:>8.2f} {result['wall_time']:>10.0f} "
                f"{'OK':>10}"
            )

    logger.info("=" * 80)

    # Save results
    json_file = results_dir / f'arch_benchmark_{timestamp}.json'
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: (vv if not isinstance(vv, np.floating) else float(vv))
                           for kk, vv in v.items()
                           if kk != 'fold_ics'}
        if 'fold_ics' in v:
            save_results[k]['fold_ics'] = [float(x) for x in v['fold_ics']]

    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'n_days': args.n_days,
                'horizon': args.horizon,
                'device': args.device,
                'min_train_days': args.min_train_days,
            },
            'results': save_results,
        }, f, indent=2)

    logger.info(f"\nResults saved to: {json_file}")
    logger.info(f"Log saved to: {log_file}")

    return results


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Architecture Benchmark for MBO Alpha')
    parser.add_argument('--arch', type=str, default='all',
                       help='Architecture(s) to test: lgbm,xgb,catboost,mlp,tcn,tabnet or "all"')
    parser.add_argument('--n-days', type=int, default=20,
                       help='Number of days to use (20=quick test, 70=full)')
    parser.add_argument('--horizon', type=str, default='ret_10s',
                       help='Target horizon (ret_3s, ret_5s, ret_10s)')
    parser.add_argument('--device', type=str, default='gpu',
                       help='Device: gpu or cpu')
    parser.add_argument('--feature-cache', type=str,
                       default=None,
                       help='Path to feature cache directory')
    parser.add_argument('--snapshot-cache', type=str,
                       default=None,
                       help='Path to snapshot cache directory')
    parser.add_argument('--min-train-days', type=int, default=5,
                       help='Minimum training days before first prediction')
    parser.add_argument('--list', action='store_true',
                       help='List available architectures and exit')

    args = parser.parse_args()

    if args.list:
        print("Available architectures:")
        for name, cls in ARCHITECTURES.items():
            print(f"  {name:<12} - {cls.__name__} (GPU: {cls.supports_gpu})")
        return

    # Default paths
    if args.feature_cache is None:
        base = Path(__file__).parent.parent
        args.feature_cache = str(base / 'data' / 'processed' / 'mbo_features_cache')
    if args.snapshot_cache is None:
        base = Path(__file__).parent.parent
        args.snapshot_cache = str(base / 'data' / 'processed' / 'medium_snapshots_cache')

    run_benchmark(args)


if __name__ == '__main__':
    main()
