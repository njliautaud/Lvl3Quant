#!/usr/bin/env python3
"""
Regime Prediction Model v2 — 5-Day Binary Green/Red Predictor
==============================================================
Predicts whether the next 5 trading days will be "green" (SPY return > 0)
or "red" (SPY return < 0).

Purpose: If accuracy > 55%, we can deploy growth allocation (leveraged momentum
on TQQQ) only during predicted-green periods, reducing R1 regime gap.

Walk-forward: 252-day sliding train, 21-day test (~150 folds over 13 years).
Models: LGBM (CPU) + MLP (GPU).
Target: sign(SPY 5-day forward return) — binary classification.

HC compliance:
  - Sliding window ONLY (HC #0)
  - All HC #705 adversarial checks built in
  - MLflow logging mandatory
  - R1 regime test included
"""

import json
import os
import sys
import time
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')

# ==============================================================
# CONFIGURATION
# ==============================================================

# Auto-detect home directory (works on Jupiter or Neptune)
import socket
_hostname = socket.gethostname()
if 'neptune' in _hostname.lower() or os.path.exists('/home/nick'):
    _BASE = Path('/home/nick/Lvl3Quant')
else:
    _BASE = Path('/home/jupiter/Lvl3Quant')

OUTPUT_DIR = _BASE / 'output' / 'growth_research' / 'regime_predictor'
CACHE_DIR = _BASE / 'output' / 'growth_research' / 'cache'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DAYS = 252        # 1 year sliding window
TEST_DAYS = 21          # 1 month OOT per fold
FWD_DAYS = 5            # Predict 5-day forward return sign
START_YEAR = 2012       # First OOT fold starts here
N_PERMUTATIONS = 200    # Permutation test shuffles
MLFLOW_EXPERIMENT = "regime_predictor_v1"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Try MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
    print("MLflow available")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARNING: MLflow not available, results will only be saved locally")


# ==============================================================
# 1. DATA DOWNLOAD & FEATURE ENGINEERING
# ==============================================================

def download_data(start='2010-01-01', end=None):
    """Download all required data via yfinance with caching."""
    if end is None:
        end = dt.date.today().isoformat()

    cache_file = CACHE_DIR / f'regime_v2_data_{end}.parquet'
    if cache_file.exists():
        print(f"Loading cached data from {cache_file.name}")
        return pd.read_parquet(cache_file)

    tickers = {
        'SPY': 'SPY',
        'VIX': '^VIX',
        'VIX3M': '^VIX3M',   # VIX term structure
        'TLT': 'TLT',
        'IEF': 'IEF',
        'HYG': 'HYG',
        'GLD': 'GLD',
        'EEM': 'EEM',
        # Sector ETFs for breadth approximation
        'XLK': 'XLK', 'XLF': 'XLF', 'XLV': 'XLV', 'XLE': 'XLE',
        'XLI': 'XLI', 'XLP': 'XLP', 'XLU': 'XLU', 'XLY': 'XLY',
        'XLC': 'XLC', 'XLRE': 'XLRE', 'XLB': 'XLB',
    }

    data = {}
    for name, ticker in tickers.items():
        print(f"  Downloading {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 50:
                data[name] = df
        except Exception as e:
            print(f"    WARNING: Failed to download {name}: {e}")

    spy = data['SPY'].copy()

    # Build raw feature dataframe
    result = pd.DataFrame(index=spy.index)
    result['spy_close'] = spy['Close']
    result['spy_volume'] = spy['Volume']

    # VIX
    if 'VIX' in data:
        result['vix'] = data['VIX']['Close'].reindex(spy.index)
    if 'VIX3M' in data:
        result['vix3m'] = data['VIX3M']['Close'].reindex(spy.index)

    # Cross-asset prices
    for name in ['TLT', 'IEF', 'HYG', 'GLD', 'EEM']:
        if name in data:
            result[f'{name.lower()}_close'] = data[name]['Close'].reindex(spy.index)

    # Sector ETF closes for breadth
    sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLP', 'XLU', 'XLY', 'XLC', 'XLRE', 'XLB']
    sector_closes = {}
    for name in sector_etfs:
        if name in data:
            sector_closes[name] = data[name]['Close'].reindex(spy.index)
    if sector_closes:
        sector_df = pd.DataFrame(sector_closes)
        # % of sectors above their 200-day MA
        above_200 = (sector_df > sector_df.rolling(200).mean()).mean(axis=1)
        result['breadth_pct_above_200ma'] = above_200
        # % of sectors with positive 1-month return
        pos_1m = (sector_df.pct_change(21) > 0).mean(axis=1)
        result['breadth_pct_pos_1m'] = pos_1m

    result = result.ffill().dropna()
    result.to_parquet(cache_file)
    print(f"Data cached: {len(result)} rows, {result.index[0].date()} to {result.index[-1].date()}")
    return result


def engineer_features(df):
    """Create all ML features from raw data. All lagging — no leakage."""
    feat = pd.DataFrame(index=df.index)
    spy = df['spy_close']
    spy_ret = spy.pct_change()

    # --- SPY returns ---
    for w in [5, 10, 20, 50, 200]:
        feat[f'spy_ret_{w}d'] = spy.pct_change(w)

    # --- SPY SMA ratios ---
    for w in [5, 10, 20, 50, 200]:
        ma = spy.rolling(w).mean()
        feat[f'spy_sma{w}_ratio'] = spy / ma - 1

    # --- RSI(14) ---
    delta = spy.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat['spy_rsi14'] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = spy.ewm(span=12, adjust=False).mean()
    ema26 = spy.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feat['spy_macd'] = macd_line / spy  # normalize
    feat['spy_macd_signal'] = signal_line / spy
    feat['spy_macd_hist'] = (macd_line - signal_line) / spy

    # --- Realized volatility ---
    for w in [10, 20, 60]:
        feat[f'spy_rvol_{w}d'] = spy_ret.rolling(w).std() * np.sqrt(252)

    # --- Drawdown from 52-week high ---
    feat['spy_dd_52w'] = spy / spy.rolling(252).max() - 1

    # --- Consecutive days below 200MA ---
    below_200 = (spy < spy.rolling(200).mean()).astype(int)
    feat['spy_days_below_200ma'] = below_200.groupby((below_200 != below_200.shift()).cumsum()).cumsum()

    # --- VIX features ---
    if 'vix' in df.columns:
        vix = df['vix']
        feat['vix_level'] = vix
        for w in [5, 10, 20]:
            feat[f'vix_{w}d_change'] = vix.pct_change(w)
        feat['vix_ma20_ratio'] = vix / vix.rolling(20).mean() - 1

        # VIX term structure (VIX vs VIX3M)
        if 'vix3m' in df.columns:
            vix3m = df['vix3m']
            feat['vix_term_structure'] = vix / vix3m.replace(0, np.nan) - 1  # <0 = contango (normal), >0 = backwardation (stress)
        else:
            # Proxy: VIX vs 20d realized vol
            feat['vix_vs_rvol'] = vix / 100 - feat['spy_rvol_20d']

    # --- Credit spread: HYG-IEF spread changes ---
    if 'hyg_close' in df.columns and 'ief_close' in df.columns:
        hyg_ief = df['hyg_close'] / df['ief_close']
        feat['credit_spread_ratio'] = hyg_ief
        for w in [5, 10, 21]:
            feat[f'credit_spread_{w}d'] = hyg_ief.pct_change(w)

    # --- Cross-asset momentum ---
    for name in ['tlt', 'gld', 'eem']:
        col = f'{name}_close'
        if col in df.columns:
            p = df[col]
            for w in [5, 10, 21, 63]:
                feat[f'{name}_ret_{w}d'] = p.pct_change(w)

    # --- Breadth ---
    if 'breadth_pct_above_200ma' in df.columns:
        feat['breadth_above_200ma'] = df['breadth_pct_above_200ma']
        feat['breadth_above_200ma_5d_chg'] = df['breadth_pct_above_200ma'].diff(5)
    if 'breadth_pct_pos_1m' in df.columns:
        feat['breadth_pos_1m'] = df['breadth_pct_pos_1m']

    # --- Calendar features ---
    feat['day_of_week'] = df.index.dayofweek
    feat['month'] = df.index.month
    feat['quarter_end'] = ((df.index.month % 3 == 0) & (df.index.day >= 25)).astype(int)
    feat['month_sin'] = np.sin(2 * np.pi * feat['month'] / 12)
    feat['month_cos'] = np.cos(2 * np.pi * feat['month'] / 12)

    # --- Volume features ---
    vol = df['spy_volume']
    feat['spy_vol_ratio_20d'] = vol / vol.rolling(20).mean()

    # --- SPY-TLT rolling correlation ---
    if 'tlt_close' in df.columns:
        tlt_r = df['tlt_close'].pct_change()
        feat['spy_tlt_corr_60d'] = spy_ret.rolling(60).corr(tlt_r)

    return feat


def create_target(spy_close, fwd_days=5):
    """Binary target: 1 if SPY 5-day forward return > 0 (green), 0 if red."""
    fwd_ret = spy_close.pct_change(fwd_days).shift(-fwd_days)
    target = (fwd_ret > 0).astype(int)
    target[fwd_ret.isna()] = np.nan
    return target, fwd_ret


# ==============================================================
# 2. MODELS
# ==============================================================

class RegimeMLP(nn.Module):
    """2-layer MLP for binary regime classification."""
    def __init__(self, input_dim, hidden1=128, hidden2=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_lgbm(X_train, y_train, X_val, y_val, feature_names):
    """Train LGBM binary classifier with early stopping."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
    }
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    return model


def train_mlp(X_train, y_train, X_val, y_val, input_dim, epochs=150, lr=1e-3, batch_size=128):
    """Train binary MLP classifier on GPU."""
    model = RegimeMLP(input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)
    y_v = torch.FloatTensor(y_val).to(DEVICE)

    best_val_loss = float('inf')
    best_state = None
    patience = 25
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            batch_idx = indices[i:i+batch_size]
            xb, yb = X_t[batch_idx], y_t[batch_idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_v)
            val_loss = criterion(val_out, y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ==============================================================
# 3. WALK-FORWARD VALIDATION
# ==============================================================

def walk_forward(features, target, fwd_ret, spy_close,
                 train_days=TRAIN_DAYS, test_days=TEST_DAYS):
    """
    Sliding window walk-forward: 252-day train, 21-day test.
    Returns per-fold predictions and metrics.
    """
    feature_names = features.columns.tolist()
    dates = features.index
    n = len(dates)

    # Storage
    all_results = []
    fold_metrics = []
    feature_importance_accum = np.zeros(len(feature_names))
    n_folds = 0

    scaler = StandardScaler()

    # Start first test at index = train_days
    # Make sure we have enough data before start_year
    start_idx = None
    for i, d in enumerate(dates):
        if d.year >= START_YEAR and i >= train_days:
            start_idx = i
            break

    if start_idx is None:
        print("ERROR: Not enough data to start walk-forward")
        return None

    test_start = start_idx
    total_folds_est = (n - test_start) // test_days
    print(f"  Estimated {total_folds_est} folds starting at {dates[test_start].date()}")

    while test_start + test_days <= n:
        train_end = test_start
        train_start = train_end - train_days
        test_end = min(test_start + test_days, n)

        X_train_full = features.iloc[train_start:train_end].values
        y_train_full = target.iloc[train_start:train_end].values
        X_test = features.iloc[test_start:test_end].values
        y_test = target.iloc[test_start:test_end].values
        test_dates = dates[test_start:test_end]
        test_fwd_ret = fwd_ret.iloc[test_start:test_end].values
        test_spy = spy_close.iloc[test_start:test_end].values

        # Skip if NaN in target
        train_valid = ~np.isnan(y_train_full)
        test_valid = ~np.isnan(y_test)
        if train_valid.sum() < 100 or test_valid.sum() < 5:
            test_start += test_days
            continue

        X_train_full = np.nan_to_num(X_train_full[train_valid], nan=0.0)
        y_train_full = y_train_full[train_valid]
        X_test_clean = np.nan_to_num(X_test[test_valid], nan=0.0)
        y_test_clean = y_test[test_valid]

        # Split train into train/val (80/20)
        val_split = int(len(X_train_full) * 0.8)
        X_tr, X_val = X_train_full[:val_split], X_train_full[val_split:]
        y_tr, y_val = y_train_full[:val_split], y_train_full[val_split:]

        # --- LGBM ---
        lgbm_model = train_lgbm(X_tr, y_tr, X_val, y_val, feature_names)
        lgbm_prob = lgbm_model.predict(X_test_clean)  # P(green)
        feature_importance_accum += lgbm_model.feature_importance(importance_type='gain')

        # --- MLP ---
        scaler.fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test_clean)

        mlp_model = train_mlp(X_tr_s, y_tr, X_val_s, y_val, input_dim=X_tr.shape[1])
        with torch.no_grad():
            mlp_logits = mlp_model(torch.FloatTensor(X_test_s).to(DEVICE))
            mlp_prob = torch.sigmoid(mlp_logits).cpu().numpy()

        # --- Ensemble ---
        ens_prob = 0.6 * lgbm_prob + 0.4 * mlp_prob

        # Store per-sample results
        valid_dates = test_dates[test_valid]
        valid_fwd_ret = test_fwd_ret[test_valid]
        for i in range(len(y_test_clean)):
            all_results.append({
                'date': valid_dates[i],
                'actual': int(y_test_clean[i]),
                'prob_lgbm': float(lgbm_prob[i]),
                'prob_mlp': float(mlp_prob[i]),
                'prob_ensemble': float(ens_prob[i]),
                'fwd_ret_5d': float(valid_fwd_ret[i]) if not np.isnan(valid_fwd_ret[i]) else 0.0,
                'fold': n_folds,
            })

        # Per-fold metrics
        try:
            auc_lgbm = roc_auc_score(y_test_clean, lgbm_prob)
            auc_mlp = roc_auc_score(y_test_clean, mlp_prob)
            auc_ens = roc_auc_score(y_test_clean, ens_prob)
        except:
            auc_lgbm = auc_mlp = auc_ens = 0.5

        acc_lgbm = accuracy_score(y_test_clean, (lgbm_prob > 0.5).astype(int))
        acc_mlp = accuracy_score(y_test_clean, (mlp_prob > 0.5).astype(int))
        acc_ens = accuracy_score(y_test_clean, (ens_prob > 0.5).astype(int))

        fold_metrics.append({
            'fold': n_folds,
            'test_start': str(valid_dates[0].date()),
            'test_end': str(valid_dates[-1].date()),
            'n_test': len(y_test_clean),
            'base_rate': float(y_test_clean.mean()),
            'auc_lgbm': round(auc_lgbm, 4),
            'auc_mlp': round(auc_mlp, 4),
            'auc_ensemble': round(auc_ens, 4),
            'acc_lgbm': round(acc_lgbm, 4),
            'acc_mlp': round(acc_mlp, 4),
            'acc_ensemble': round(acc_ens, 4),
        })

        n_folds += 1
        if n_folds % 25 == 0:
            print(f"    Fold {n_folds}: {valid_dates[0].date()} | "
                  f"AUC lgbm={auc_lgbm:.3f} mlp={auc_mlp:.3f} ens={auc_ens:.3f} | "
                  f"Acc ens={acc_ens:.3f}")

        test_start += test_days

    print(f"  Completed {n_folds} walk-forward folds")

    # Feature importance
    fi = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance_accum / max(n_folds, 1)
    }).sort_values('importance', ascending=False)

    return {
        'results': pd.DataFrame(all_results),
        'fold_metrics': pd.DataFrame(fold_metrics),
        'feature_importance': fi,
        'n_folds': n_folds,
    }


# ==============================================================
# 4. HC #705 ADVERSARIAL CHECKS
# ==============================================================

def run_adversarial_checks(results_df, spy_close, n_permutations=N_PERMUTATIONS):
    """
    HC #705: All adversarial checks built into the script.
    Returns dict of check results + warnings.
    """
    checks = {}
    warnings_list = []

    actuals = results_df['actual'].values
    probs = results_df['prob_ensemble'].values
    preds = (probs > 0.5).astype(int)
    dates = pd.DatetimeIndex(results_df['date'])

    real_acc = accuracy_score(actuals, preds)
    try:
        real_auc = roc_auc_score(actuals, probs)
    except:
        real_auc = 0.5

    # -------------------------------------------------------
    # (c) PERMUTATION TEST — 200 shuffles
    # -------------------------------------------------------
    print("\n  [HC705-c] Permutation test...")
    perm_accs = []
    perm_aucs = []
    for i in range(n_permutations):
        shuffled = np.random.permutation(actuals)
        perm_accs.append(accuracy_score(shuffled, preds))
        try:
            perm_aucs.append(roc_auc_score(shuffled, probs))
        except:
            perm_aucs.append(0.5)

    perm_acc_p = (np.array(perm_accs) >= real_acc).mean()
    perm_auc_p = (np.array(perm_aucs) >= real_auc).mean()

    checks['permutation'] = {
        'real_accuracy': round(real_acc, 4),
        'real_auc': round(real_auc, 4),
        'perm_acc_mean': round(np.mean(perm_accs), 4),
        'perm_acc_std': round(np.std(perm_accs), 4),
        'perm_auc_mean': round(np.mean(perm_aucs), 4),
        'perm_p_value_acc': round(perm_acc_p, 4),
        'perm_p_value_auc': round(perm_auc_p, 4),
        'pass': bool(perm_acc_p < 0.05),
    }
    if perm_acc_p >= 0.05:
        warnings_list.append(f"WARNING: Permutation test FAIL (p={perm_acc_p:.3f}). Model may not have real edge.")
    print(f"    Accuracy: {real_acc:.4f} vs perm mean {np.mean(perm_accs):.4f} (p={perm_acc_p:.3f})")
    print(f"    AUC: {real_auc:.4f} vs perm mean {np.mean(perm_aucs):.4f} (p={perm_auc_p:.3f})")

    # -------------------------------------------------------
    # (d) SUB-PERIOD CONSISTENCY — split into 2 halves
    # -------------------------------------------------------
    print("\n  [HC705-d] Sub-period consistency...")
    mid = len(actuals) // 2
    half1_acc = accuracy_score(actuals[:mid], preds[:mid])
    half2_acc = accuracy_score(actuals[mid:], preds[mid:])
    try:
        half1_auc = roc_auc_score(actuals[:mid], probs[:mid])
        half2_auc = roc_auc_score(actuals[mid:], probs[mid:])
    except:
        half1_auc = half2_auc = 0.5

    mid_date = dates[mid]
    checks['sub_period'] = {
        'period1': f'{dates[0].date()} to {mid_date.date()}',
        'period2': f'{mid_date.date()} to {dates[-1].date()}',
        'acc_period1': round(half1_acc, 4),
        'acc_period2': round(half2_acc, 4),
        'auc_period1': round(half1_auc, 4),
        'auc_period2': round(half2_auc, 4),
        'both_above_50pct': bool(half1_acc > 0.5 and half2_acc > 0.5),
    }
    if not (half1_acc > 0.5 and half2_acc > 0.5):
        warnings_list.append(f"WARNING: Sub-period consistency FAIL. P1 acc={half1_acc:.3f}, P2 acc={half2_acc:.3f}")
    print(f"    Period 1: Acc={half1_acc:.4f}, AUC={half1_auc:.4f}")
    print(f"    Period 2: Acc={half2_acc:.4f}, AUC={half2_auc:.4f}")

    # -------------------------------------------------------
    # (e) OUTLIER REMOVAL — remove top 5 most confident correct predictions
    # -------------------------------------------------------
    print("\n  [HC705-e] Outlier removal check...")
    # Simulate strategy returns
    fwd_rets = results_df['fwd_ret_5d'].values
    strat_rets = np.where(preds == 1, fwd_rets, 0)  # in market when predicting green
    strat_rets_series = pd.Series(strat_rets)

    # Compute original Sharpe
    orig_sharpe = strat_rets_series.mean() / strat_rets_series.std() * np.sqrt(252 / FWD_DAYS) if strat_rets_series.std() > 0 else 0

    # Remove top 5 by absolute strategy return
    top5_idx = np.argsort(np.abs(strat_rets))[-5:]
    mask = np.ones(len(strat_rets), dtype=bool)
    mask[top5_idx] = False
    reduced = strat_rets_series[mask]
    reduced_sharpe = reduced.mean() / reduced.std() * np.sqrt(252 / FWD_DAYS) if reduced.std() > 0 else 0

    sharpe_drop = 1 - reduced_sharpe / orig_sharpe if orig_sharpe != 0 else 0

    checks['outlier_removal'] = {
        'original_sharpe': round(orig_sharpe, 4),
        'reduced_sharpe': round(reduced_sharpe, 4),
        'sharpe_drop_pct': round(sharpe_drop * 100, 1),
        'pass': bool(abs(sharpe_drop) < 0.5),
    }
    if abs(sharpe_drop) >= 0.5:
        warnings_list.append(f"WARNING: Outlier-driven! Sharpe drops {sharpe_drop*100:.0f}% when top 5 trades removed.")
    print(f"    Sharpe: {orig_sharpe:.4f} → {reduced_sharpe:.4f} (drop {sharpe_drop*100:.1f}%)")

    # -------------------------------------------------------
    # (f) R1 REGIME TEST — green/red/flat SPY days
    # -------------------------------------------------------
    print("\n  [HC705-f] R1 Regime test...")
    # Classify actual regime by SPY close-to-close on prediction date
    spy_daily_ret = spy_close.pct_change()
    pred_spy_ret = spy_daily_ret.reindex(dates).values

    green_mask = pred_spy_ret > 0.001
    red_mask = pred_spy_ret < -0.001
    flat_mask = ~green_mask & ~red_mask

    def regime_acc(mask):
        if mask.sum() < 10:
            return np.nan
        return accuracy_score(actuals[mask], preds[mask])

    green_acc = regime_acc(green_mask)
    red_acc = regime_acc(red_mask)
    flat_acc = regime_acc(flat_mask)

    # R1 gap: |green - red| / max(|green|, |red|)
    if not np.isnan(green_acc) and not np.isnan(red_acc) and max(green_acc, red_acc) > 0:
        r1_gap = abs(green_acc - red_acc) / max(green_acc, red_acc)
    else:
        r1_gap = np.nan

    checks['regime_test'] = {
        'acc_green_days': round(float(green_acc), 4) if not np.isnan(green_acc) else None,
        'acc_red_days': round(float(red_acc), 4) if not np.isnan(red_acc) else None,
        'acc_flat_days': round(float(flat_acc), 4) if not np.isnan(flat_acc) else None,
        'n_green': int(green_mask.sum()),
        'n_red': int(red_mask.sum()),
        'n_flat': int(flat_mask.sum()),
        'r1_gap': round(float(r1_gap), 4) if not np.isnan(r1_gap) else None,
        'pass': bool(not np.isnan(r1_gap) and r1_gap < 0.50),
    }
    if not np.isnan(r1_gap) and r1_gap >= 0.50:
        warnings_list.append(f"WARNING: R1 regime gap = {r1_gap:.3f} (>0.50 threshold). Model may be regime-tailored.")
    print(f"    Green-day acc: {green_acc:.4f} ({int(green_mask.sum())} days)")
    print(f"    Red-day acc:   {red_acc:.4f} ({int(red_mask.sum())} days)")
    print(f"    R1 gap: {r1_gap:.4f} {'PASS' if r1_gap < 0.50 else 'FAIL'}")

    # -------------------------------------------------------
    # (a/b) PRICING SANITY — N/A for this model (no options pricing)
    # -------------------------------------------------------
    checks['pricing_sanity'] = {'status': 'N/A — no options pricing in this model'}

    # -------------------------------------------------------
    # CALIBRATION
    # -------------------------------------------------------
    print("\n  Calibration check...")
    try:
        prob_true, prob_pred = calibration_curve(actuals, probs, n_bins=10, strategy='uniform')
        brier = brier_score_loss(actuals, probs)
        checks['calibration'] = {
            'brier_score': round(brier, 4),
            'calibration_bins': {f'{pp:.2f}': round(float(pt), 4) for pp, pt in zip(prob_pred, prob_true)},
        }
        print(f"    Brier score: {brier:.4f}")
    except:
        checks['calibration'] = {'status': 'could not compute'}

    return checks, warnings_list


# ==============================================================
# 5. STRATEGY SIMULATION — VALUE OF PREDICTIONS
# ==============================================================

def simulate_growth_strategy(results_df, spy_close):
    """
    Simulate deploying growth (TQQQ-proxy = 3x SPY) only during predicted-green.
    Compare to always-in and 200MA baseline.
    """
    dates = pd.DatetimeIndex(results_df['date'])
    probs = results_df['prob_ensemble'].values
    fwd_rets = results_df['fwd_ret_5d'].values
    spy_daily_ret = spy_close.pct_change()

    # For simplicity, use SPY daily returns (not TQQQ) for regime filter evaluation
    # The value is in the FILTERING, not the leverage
    daily_rets = spy_daily_ret.reindex(dates).fillna(0).values

    strategies = {}

    # 1) Always-in SPY
    cum = (1 + pd.Series(daily_rets)).cumprod()
    strategies['always_in'] = _compute_metrics(daily_rets, 'Always-In SPY')

    # 2) ML-filtered: in market only when P(green) > 0.5
    ml_rets = np.where(probs > 0.5, daily_rets, 0)
    strategies['ml_filtered_50'] = _compute_metrics(ml_rets, 'ML P(green)>0.50')

    # 3) ML-filtered: in market only when P(green) > 0.55 (conservative)
    ml_rets_55 = np.where(probs > 0.55, daily_rets, 0)
    strategies['ml_filtered_55'] = _compute_metrics(ml_rets_55, 'ML P(green)>0.55')

    # 4) ML-filtered: in market only when P(green) > 0.60 (very conservative)
    ml_rets_60 = np.where(probs > 0.60, daily_rets, 0)
    strategies['ml_filtered_60'] = _compute_metrics(ml_rets_60, 'ML P(green)>0.60')

    # 5) 200MA baseline
    ma200 = spy_close.rolling(200).mean()
    above_200 = (spy_close > ma200).reindex(dates).shift(1).fillna(False).values.astype(float)
    ma_rets = above_200 * daily_rets
    strategies['ma200_baseline'] = _compute_metrics(ma_rets, '200MA Baseline')

    # 6) R1 gap for each strategy
    # Green/red classification based on actual 5-day forward returns
    actual_regime = (fwd_rets > 0).astype(int)
    for key in strategies:
        if key == 'always_in':
            strat_daily = daily_rets
        elif key == 'ml_filtered_50':
            strat_daily = ml_rets
        elif key == 'ml_filtered_55':
            strat_daily = ml_rets_55
        elif key == 'ml_filtered_60':
            strat_daily = ml_rets_60
        elif key == 'ma200_baseline':
            strat_daily = ma_rets

        green_days = actual_regime == 1
        red_days = actual_regime == 0

        green_sharpe = _sharpe(strat_daily[green_days]) if green_days.sum() > 20 else 0
        red_sharpe = _sharpe(strat_daily[red_days]) if red_days.sum() > 20 else 0

        max_s = max(abs(green_sharpe), abs(red_sharpe))
        r1_gap = abs(green_sharpe - red_sharpe) / max_s if max_s > 0 else 0

        strategies[key]['green_sharpe'] = round(green_sharpe, 4)
        strategies[key]['red_sharpe'] = round(red_sharpe, 4)
        strategies[key]['r1_gap'] = round(r1_gap, 4)

    return strategies


def _sharpe(rets):
    if len(rets) < 2 or np.std(rets) == 0:
        return 0
    return np.mean(rets) / np.std(rets) * np.sqrt(252)


def _compute_metrics(daily_rets, name):
    rets = pd.Series(daily_rets)
    cum = (1 + rets).cumprod()
    total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    years = len(rets) / 252
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else 0

    sharpe = _sharpe(daily_rets)
    downside = rets[rets < 0].std()
    sortino = rets.mean() / downside * np.sqrt(252) if downside > 0 else 0

    rolling_max = cum.cummax()
    max_dd = (cum / rolling_max - 1).min()

    time_in_mkt = (rets != 0).mean()
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')
    wr = (rets[rets != 0] > 0).mean() if (rets != 0).any() else 0

    print(f"    {name}: CAGR={cagr:.3f} Sharpe={sharpe:.3f} Sortino={sortino:.3f} "
          f"MaxDD={max_dd:.3f} PF={pf:.3f} WR={wr:.3f} TimeInMkt={time_in_mkt:.1%}")

    return {
        'name': name,
        'total_return': round(float(total_ret), 4),
        'cagr': round(float(cagr), 4),
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'max_drawdown': round(float(max_dd), 4),
        'profit_factor': round(float(pf), 4),
        'win_rate': round(float(wr), 4),
        'time_in_market': round(float(time_in_mkt), 4),
        'years': round(years, 2),
    }


# ==============================================================
# 6. MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("REGIME PREDICTOR v2 — 5-Day Binary Green/Red")
    print(f"Train={TRAIN_DAYS}d, Test={TEST_DAYS}d, Fwd={FWD_DAYS}d, Start={START_YEAR}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # --- MLflow setup ---
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            # Use Jupiter's MLflow (Tailscale IP) when running on Neptune
            _mlflow_uri = "http://localhost:5000" if _hostname != 'neptune' and os.path.exists('/home/jupiter') else "http://jupiter:5000"
            mlflow.set_tracking_uri(_mlflow_uri)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(run_name=f"regime_pred_v2_{dt.datetime.now().strftime('%Y%m%d_%H%M')}")
            mlflow.log_params({
                'train_days': TRAIN_DAYS,
                'test_days': TEST_DAYS,
                'fwd_days': FWD_DAYS,
                'start_year': START_YEAR,
                'device': str(DEVICE),
                'n_permutations': N_PERMUTATIONS,
            })
            print(f"MLflow run started: {mlflow_run.info.run_id}")
        except Exception as e:
            print(f"WARNING: MLflow connection failed: {e}")
            mlflow_run = None

    # Step 1: Download data
    print("\n[1/6] Downloading data...")
    raw_data = download_data(start='2010-01-01')
    print(f"  Raw data: {len(raw_data)} rows, {raw_data.index[0].date()} to {raw_data.index[-1].date()}")

    # Step 2: Feature engineering
    print("\n[2/6] Engineering features...")
    features = engineer_features(raw_data)
    target, fwd_ret = create_target(raw_data['spy_close'], fwd_days=FWD_DAYS)

    # Align and drop NaN
    valid_mask = features.notna().all(axis=1) & target.notna()
    features = features[valid_mask]
    target = target[valid_mask]
    fwd_ret = fwd_ret[valid_mask]

    n_features = features.shape[1]
    base_rate = target.mean()
    print(f"  Features: {n_features}")
    print(f"  Valid samples: {len(features)}")
    print(f"  Base rate (green): {base_rate:.3f}")

    if MLFLOW_AVAILABLE and mlflow_run:
        try:
            mlflow.log_params({'n_features': n_features, 'n_samples': len(features), 'base_rate': round(base_rate, 4)})
        except:
            pass

    # Step 3: Walk-forward validation
    print("\n[3/6] Running walk-forward validation...")
    wf = walk_forward(features, target, fwd_ret, raw_data['spy_close'])

    if wf is None:
        print("FATAL: Walk-forward failed")
        return

    results_df = wf['results']
    fold_df = wf['fold_metrics']
    fi = wf['feature_importance']

    # Overall metrics
    print("\n[4/6] Overall metrics...")
    actuals = results_df['actual'].values

    overall_metrics = {}
    for model in ['lgbm', 'mlp', 'ensemble']:
        probs = results_df[f'prob_{model}'].values
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(actuals, preds)
        try:
            auc = roc_auc_score(actuals, probs)
        except:
            auc = 0.5
        brier = brier_score_loss(actuals, probs) if len(np.unique(actuals)) > 1 else 1.0

        overall_metrics[model] = {
            'accuracy': round(acc, 4),
            'auc': round(auc, 4),
            'brier_score': round(brier, 4),
            'base_rate': round(float(actuals.mean()), 4),
            'accuracy_lift': round(acc - actuals.mean(), 4),  # vs always predicting majority class
        }
        print(f"  {model.upper():10s}: Acc={acc:.4f} AUC={auc:.4f} Brier={brier:.4f} "
              f"Lift={acc - actuals.mean():.4f}")

    # Accuracy by actual regime (did the model catch bear markets?)
    print("\n  Accuracy by actual 5d regime:")
    green_mask = actuals == 1
    red_mask = actuals == 0
    for model in ['lgbm', 'mlp', 'ensemble']:
        preds = (results_df[f'prob_{model}'].values > 0.5).astype(int)
        green_acc = accuracy_score(actuals[green_mask], preds[green_mask]) if green_mask.sum() > 0 else 0
        red_acc = accuracy_score(actuals[red_mask], preds[red_mask]) if red_mask.sum() > 0 else 0
        print(f"    {model.upper():10s}: Green-week acc={green_acc:.4f} ({green_mask.sum()} samples), "
              f"Red-week acc={red_acc:.4f} ({red_mask.sum()} samples)")
        overall_metrics[model]['acc_green_weeks'] = round(green_acc, 4)
        overall_metrics[model]['acc_red_weeks'] = round(red_acc, 4)

    # Feature importance
    print(f"\n  Top 15 features (LGBM gain):")
    for _, row in fi.head(15).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.1f}")

    # Step 5: Adversarial checks (HC #705)
    print("\n[5/6] Adversarial checks (HC #705)...")
    adv_checks, warnings_list = run_adversarial_checks(results_df, raw_data['spy_close'])

    if warnings_list:
        print("\n  " + "=" * 50)
        for w in warnings_list:
            print(f"  {w}")
        print("  " + "=" * 50)

    # Step 6: Strategy simulation
    print("\n[6/6] Strategy simulation — value of predictions...")
    strategies = simulate_growth_strategy(results_df, raw_data['spy_close'])

    # Log to MLflow
    if MLFLOW_AVAILABLE and mlflow_run:
        try:
            for model, m in overall_metrics.items():
                for k, v in m.items():
                    mlflow.log_metric(f'{model}_{k}', v)
            for check_name, check in adv_checks.items():
                if isinstance(check, dict):
                    for k, v in check.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            mlflow.log_metric(f'adv_{check_name}_{k}', v)
            for strat_name, strat in strategies.items():
                for k, v in strat.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f'strat_{strat_name}_{k}', v)
            mlflow.log_metric('runtime_seconds', time.time() - t0)
        except Exception as e:
            print(f"  MLflow logging error: {e}")

    # Save results
    elapsed = time.time() - t0
    final_results = {
        'timestamp': dt.datetime.now().isoformat(),
        'runtime_seconds': round(elapsed, 1),
        'config': {
            'train_days': TRAIN_DAYS,
            'test_days': TEST_DAYS,
            'fwd_days': FWD_DAYS,
            'start_year': START_YEAR,
            'n_features': n_features,
            'n_folds': wf['n_folds'],
            'base_rate_green': round(float(base_rate), 4),
            'device': str(DEVICE),
            'validation': 'sliding_window',
            'n_permutations': N_PERMUTATIONS,
        },
        'overall_metrics': overall_metrics,
        'adversarial_checks': adv_checks,
        'adversarial_warnings': warnings_list,
        'strategies': strategies,
        'feature_importance_top20': [
            {'feature': row['feature'], 'importance': round(float(row['importance']), 2)}
            for _, row in fi.head(20).iterrows()
        ],
    }

    # Determine if useful
    ens_acc = overall_metrics['ensemble']['accuracy']
    ens_auc = overall_metrics['ensemble']['auc']
    perm_pass = adv_checks.get('permutation', {}).get('pass', False)

    useful = ens_acc > 0.55 and perm_pass
    ml_filtered = strategies.get('ml_filtered_50', {})
    always_in = strategies.get('always_in', {})
    ml_sharpe = ml_filtered.get('sharpe', 0)
    base_sharpe = always_in.get('sharpe', 0)
    ml_r1 = ml_filtered.get('r1_gap', 999)
    base_r1 = always_in.get('r1_gap', 999)

    conclusion = (
        f"Ensemble accuracy: {ens_acc:.1%} (base rate {base_rate:.1%}), AUC: {ens_auc:.3f}. "
        f"Permutation test: {'PASS' if perm_pass else 'FAIL'}. "
        f"ML-filtered Sharpe: {ml_sharpe:.3f} vs Always-In: {base_sharpe:.3f}. "
        f"ML-filtered R1 gap: {ml_r1:.3f} vs Always-In R1: {base_r1:.3f}. "
    )
    if useful:
        conclusion += "VERDICT: Model shows real edge above 55% threshold. Worth deploying as growth overlay."
    else:
        conclusion += "VERDICT: Model does NOT meet 55% accuracy + permutation test threshold. Not useful for growth timing."

    final_results['conclusion'] = conclusion
    final_results['useful'] = useful

    # Save files
    results_file = OUTPUT_DIR / 'regime_predictor_v2_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    results_df.to_csv(OUTPUT_DIR / 'predictions_v2.csv', index=False)
    fold_df.to_csv(OUTPUT_DIR / 'fold_results_v2.csv', index=False)
    fi.to_csv(OUTPUT_DIR / 'feature_importance_v2.csv', index=False)

    # End MLflow run
    if MLFLOW_AVAILABLE and mlflow_run:
        try:
            mlflow.log_artifact(str(results_file))
            mlflow.log_artifact(str(OUTPUT_DIR / 'predictions_v2.csv'))
            mlflow.log_artifact(str(OUTPUT_DIR / 'fold_results_v2.csv'))
            mlflow.log_artifact(str(OUTPUT_DIR / 'feature_importance_v2.csv'))
            mlflow.end_run()
        except Exception as e:
            print(f"  MLflow artifact logging error: {e}")
            try:
                mlflow.end_run()
            except:
                pass

    print(f"\n{'=' * 70}")
    print("CONCLUSION:")
    print(conclusion)
    print(f"Runtime: {elapsed:.0f}s")
    print(f"{'=' * 70}")

    return final_results


if __name__ == '__main__':
    results = main()
