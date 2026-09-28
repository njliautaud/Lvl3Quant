#!/usr/bin/env python3
"""
Entry Timing Model v1 — Stock Return Classifier for Options Income Strategies
==============================================================================
Predicts whether each stock in the options universe will be flat/up or down
over the next 5 trading days. Uses a 1D-CNN classifier with walk-forward validation.

Architecture: 1D-CNN (temporal convolutions over 20-day feature windows)
Features: returns, volume_ratio, RSI, bollinger_%B, MACD_hist, ATR_ratio,
          VIX, VIX_term_structure, sector_relative_strength
Universe: 30 tickers from options engines
Walk-forward: 3yr train / 6mo test, rolling forward

Output: /home/nick/Lvl3Quant/output/entry_timing_model_v1/
"""

import os
import sys
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# ─── Config ───────────────────────────────────────────────────────────────────
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'TSLA',
    'JPM', 'BAC', 'GS', 'JNJ', 'UNH', 'PG', 'KO', 'MCD',
    'HD', 'WMT', 'NFLX', 'DIS', 'CRM', 'INTC', 'PYPL', 'COST',
    'V', 'MA', 'AVGO', 'SBUX', 'CVS', 'COIN'
]

LOOKBACK = 20          # 20 trading days of features
FORWARD_DAYS = 5       # predict 5-day forward return
TRAIN_YEARS = 3        # walk-forward train window
TEST_MONTHS = 6        # walk-forward test window
BATCH_SIZE = 256
EPOCHS = 50
LR = 5e-4
LABEL_SMOOTHING = 0.05  # soften noisy labels
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = Path('/home/nick/Lvl3Quant/output/entry_timing_model_v1')
DATA_START = '2014-01-01'  # extra buffer for feature computation

NUM_FEATURES = 15  # expanded feature set for v1.1


# ─── Data Download ────────────────────────────────────────────────────────────
def download_data():
    """Download OHLCV data for all tickers + VIX + VIX3M + SPY."""
    print(f"[DATA] Downloading {len(TICKERS)} tickers + VIX/VIX3M/SPY from {DATA_START}...")

    all_tickers = TICKERS + ['SPY', '^VIX', '^VIX3M']

    data = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=DATA_START, auto_adjust=True, progress=False)
            if len(df) > 100:
                data[ticker] = df
                print(f"  {ticker}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
            else:
                print(f"  {ticker}: SKIPPED (only {len(df)} rows)")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    return data


# ─── Feature Engineering ──────────────────────────────────────────────────────
def compute_features(price_df, vix_series, vix3m_series, spy_returns):
    """Compute all features for a single ticker."""
    df = price_df.copy()
    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
    high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
    low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
    volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

    # 1. Daily returns
    returns = close.pct_change()

    # 2. Volume ratio (today / 20d avg)
    vol_ma20 = volume.rolling(20).mean()
    volume_ratio = volume / vol_ma20

    # 3. RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi / 100.0  # normalize to [0,1]

    # 4. Bollinger %B (20-day, 2 std)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    boll_pct_b = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    # 5. MACD histogram
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = macd_line - signal_line
    # Normalize by price
    macd_hist_norm = macd_hist / close

    # 6. ATR ratio (14-day ATR / close)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_ratio = atr / close

    # 7. VIX level (normalized)
    vix_aligned = vix_series.reindex(close.index, method='ffill') / 100.0

    # 8. VIX term structure (VIX / VIX3M)
    vix3m_aligned = vix3m_series.reindex(close.index, method='ffill')
    vix_raw = vix_series.reindex(close.index, method='ffill')
    vix_term = (vix_raw / vix3m_aligned.replace(0, np.nan)).fillna(1.0)

    # 9. Sector relative strength (stock 20d return - SPY 20d return)
    stock_ret20 = close.pct_change(20)
    spy_ret20 = spy_returns.reindex(close.index, method='ffill')
    sector_rel = stock_ret20 - spy_ret20

    # 10. 5-day momentum (short-term trend)
    mom5 = close.pct_change(5)

    # 11. 20-day realized volatility (annualized)
    realized_vol = returns.rolling(20).std() * np.sqrt(252)

    # 12. Distance from 52-week high (drawdown indicator)
    high_252 = close.rolling(252, min_periods=50).max()
    dist_from_high = (close - high_252) / high_252

    # 13. Volume trend (5d avg / 20d avg — accumulation/distribution)
    vol_ma5 = volume.rolling(5).mean()
    vol_trend = vol_ma5 / vol_ma20

    # 14. Price-to-SMA50 ratio (trend strength)
    sma50 = close.rolling(50).mean()
    price_to_sma50 = (close - sma50) / sma50

    # 15. Overnight gap indicator (open vs prev close — proxy for sentiment)
    open_price = df['Open'].squeeze() if isinstance(df['Open'], pd.DataFrame) else df['Open']
    overnight_gap = (open_price - close.shift(1)) / close.shift(1)

    # Combine
    features = pd.DataFrame({
        'returns': returns,
        'volume_ratio': volume_ratio,
        'rsi': rsi,
        'boll_pct_b': boll_pct_b,
        'macd_hist': macd_hist_norm,
        'atr_ratio': atr_ratio,
        'vix': vix_aligned,
        'vix_term': vix_term,
        'sector_rel': sector_rel,
        'mom5': mom5,
        'realized_vol': realized_vol,
        'dist_from_high': dist_from_high,
        'vol_trend': vol_trend,
        'price_to_sma50': price_to_sma50,
        'overnight_gap': overnight_gap,
    }, index=close.index)

    # Forward return label: 1 if 5-day forward return >= 0, else 0
    fwd_return = close.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
    features['label'] = (fwd_return >= 0).astype(float)
    features.loc[fwd_return.isna(), 'label'] = np.nan

    return features


def build_dataset(data):
    """Build the full dataset with rolling windows for all tickers."""
    print("\n[FEATURES] Computing features for all tickers...")

    # Get VIX and SPY data
    vix_close = data['^VIX']['Close'].squeeze()
    vix3m_close = data['^VIX3M']['Close'].squeeze() if '^VIX3M' in data else vix_close
    spy_close = data['SPY']['Close'].squeeze()
    spy_ret20 = spy_close.pct_change(20)

    all_windows = []
    all_labels = []
    all_dates = []
    all_ticker_ids = []

    for tidx, ticker in enumerate(TICKERS):
        if ticker not in data:
            print(f"  {ticker}: SKIPPED (no data)")
            continue

        features = compute_features(data[ticker], vix_close, vix3m_close, spy_ret20)
        feature_cols = [c for c in features.columns if c != 'label']

        # Create rolling windows
        valid_mask = features[feature_cols + ['label']].notna().all(axis=1)
        feat_arr = features[feature_cols].values
        label_arr = features['label'].values
        dates = features.index

        count = 0
        for i in range(LOOKBACK, len(features) - 1):
            if not valid_mask.iloc[i]:
                continue
            # Check all lookback days have valid features
            window_start = i - LOOKBACK
            if not valid_mask.iloc[window_start:i].all():
                continue

            window = feat_arr[window_start:i]  # shape: (LOOKBACK, NUM_FEATURES)
            label = label_arr[i]

            if np.isnan(label) or np.any(np.isnan(window)):
                continue

            all_windows.append(window)
            all_labels.append(label)
            all_dates.append(dates[i])
            all_ticker_ids.append(tidx)
            count += 1

        print(f"  {ticker}: {count} samples")

    X = np.array(all_windows, dtype=np.float32)  # (N, 20, 9)
    y = np.array(all_labels, dtype=np.float32)    # (N,)
    dates = np.array(all_dates)
    tickers = np.array(all_ticker_ids)

    print(f"\n[DATASET] Total: {len(X)} samples, class balance: {y.mean():.3f} positive")
    return X, y, dates, tickers


# ─── Model ────────────────────────────────────────────────────────────────────
class CNN1DClassifier(nn.Module):
    """
    Enhanced 1D-CNN with residual connections and temporal attention.
    Input: (batch, lookback=20, features) -> permuted to (batch, features, lookback)
    """
    def __init__(self, n_features=15, lookback=20):
        super().__init__()
        # Initial projection
        self.conv_in = nn.Conv1d(n_features, 64, kernel_size=1)
        self.bn_in = nn.BatchNorm1d(64)

        # Residual block 1
        self.conv1a = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm1d(64)
        self.conv1b = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm1d(64)

        # Residual block 2
        self.conv2a = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm1d(128)
        self.conv2b = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm1d(128)
        self.proj2 = nn.Conv1d(64, 128, kernel_size=1)  # skip connection projection

        # Temporal attention
        self.attn_query = nn.Linear(128, 32)
        self.attn_key = nn.Linear(128, 32)
        self.attn_value = nn.Linear(128, 128)

        # Classifier head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, lookback, features) -> (batch, features, lookback)
        x = x.permute(0, 2, 1)

        # Initial projection
        x = self.relu(self.bn_in(self.conv_in(x)))

        # Residual block 1
        residual = x
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.bn1b(self.conv1b(x))
        x = self.relu(x + residual)

        # Residual block 2
        residual = self.proj2(x)
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = self.relu(x + residual)

        # Temporal attention: (batch, 128, time) -> (batch, time, 128)
        x_t = x.permute(0, 2, 1)
        q = self.attn_query(x_t)
        k = self.attn_key(x_t)
        v = self.attn_value(x_t)
        attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (32 ** 0.5), dim=-1)
        x_attn = torch.bmm(attn_weights, v)  # (batch, time, 128)

        # Pool over time
        x_attn = x_attn.permute(0, 2, 1)  # (batch, 128, time)
        x = self.pool(x_attn).squeeze(-1)  # (batch, 128)

        # Classifier
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # (batch, 1)
        return x.squeeze(-1)


# ─── Training ─────────────────────────────────────────────────────────────────
def train_one_fold(X_train, y_train, X_val, y_val, fold_id):
    """Train model on one walk-forward fold."""
    # Standardize features
    N_train, L, F = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, F)
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_flat.reshape(N_train, L, F)

    N_val = X_val.shape[0]
    X_val_flat = X_val.reshape(-1, F)
    X_val_flat = scaler.transform(X_val_flat)
    X_val_scaled = X_val_flat.reshape(N_val, L, F)

    # Handle class imbalance with pos_weight
    pos_ratio = y_train.mean()
    neg_ratio = 1 - pos_ratio
    pos_weight = torch.tensor([neg_ratio / max(pos_ratio, 1e-6)]).to(DEVICE)

    # Label smoothing: soften targets to reduce overfitting to noisy labels
    y_train_smooth = y_train * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train_scaled),
        torch.tensor(y_train_smooth)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled),
        torch.tensor(y_val)
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    # Model
    model = CNN1DClassifier(n_features=F, lookback=L).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = 0
    best_state = None
    patience = 7
    no_improve = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        scheduler.step()

        # Validate
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE)
                pred = model(xb)
                val_preds.append(torch.sigmoid(pred).cpu().numpy())
                val_labels.append(yb.numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)

        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except ValueError:
            val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Load best model and get final predictions
    model.load_state_dict(best_state)
    model.eval()
    val_preds_final = []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(DEVICE)
            pred = model(xb)
            val_preds_final.append(torch.sigmoid(pred).cpu().numpy())

    val_preds_final = np.concatenate(val_preds_final)

    return model, scaler, val_preds_final, best_auc


def walk_forward_train(X, y, dates, tickers):
    """Walk-forward cross-validation: 3yr train, 6mo test, roll forward."""
    print(f"\n{'='*70}")
    print(f"[WALK-FORWARD] Starting walk-forward validation on {DEVICE}")
    print(f"  Train window: {TRAIN_YEARS} years, Test window: {TEST_MONTHS} months")
    print(f"{'='*70}\n")

    unique_dates = np.sort(np.unique(dates))
    min_date = unique_dates[0]
    max_date = unique_dates[-1]

    # Define fold boundaries
    train_days = TRAIN_YEARS * 252
    test_days = TEST_MONTHS * 21  # ~21 trading days per month

    fold_results = []
    all_oot_preds = []
    all_oot_labels = []
    all_oot_dates = []
    all_oot_tickers = []

    fold_id = 0
    start_idx = 0

    while start_idx + train_days + test_days <= len(unique_dates):
        train_end_idx = start_idx + train_days
        test_end_idx = min(train_end_idx + test_days, len(unique_dates))

        train_start_date = unique_dates[start_idx]
        train_end_date = unique_dates[train_end_idx - 1]
        test_start_date = unique_dates[train_end_idx]
        test_end_date = unique_dates[test_end_idx - 1]

        # Split data
        train_mask = (dates >= train_start_date) & (dates <= train_end_date)
        test_mask = (dates >= test_start_date) & (dates <= test_end_date)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_dates = dates[test_mask]
        test_tickers = tickers[test_mask]

        if len(X_train) < 500 or len(X_test) < 50:
            start_idx += test_days
            continue

        print(f"[FOLD {fold_id}] Train: {train_start_date.date()} to {train_end_date.date()} "
              f"({len(X_train)} samples) | Test: {test_start_date.date()} to {test_end_date.date()} "
              f"({len(X_test)} samples)")

        model, scaler, test_preds, best_auc = train_one_fold(X_train, y_train, X_test, y_test, fold_id)

        # Metrics
        pred_binary = (test_preds >= 0.5).astype(int)
        acc = accuracy_score(y_test, pred_binary)
        try:
            auc = roc_auc_score(y_test, test_preds)
        except ValueError:
            auc = 0.5
        prec_pos = precision_score(y_test, pred_binary, pos_label=1, zero_division=0)
        rec_pos = recall_score(y_test, pred_binary, pos_label=1, zero_division=0)
        prec_neg = precision_score(y_test, pred_binary, pos_label=0, zero_division=0)
        rec_neg = recall_score(y_test, pred_binary, pos_label=0, zero_division=0)

        fold_result = {
            'fold': fold_id,
            'train_start': str(train_start_date.date()),
            'train_end': str(train_end_date.date()),
            'test_start': str(test_start_date.date()),
            'test_end': str(test_end_date.date()),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'accuracy': round(acc, 4),
            'auc': round(auc, 4),
            'precision_up': round(prec_pos, 4),
            'recall_up': round(rec_pos, 4),
            'precision_down': round(prec_neg, 4),
            'recall_down': round(rec_neg, 4),
            'class_balance': round(y_test.mean(), 4),
        }
        fold_results.append(fold_result)

        all_oot_preds.extend(test_preds.tolist())
        all_oot_labels.extend(y_test.tolist())
        all_oot_dates.extend([str(d.date()) for d in test_dates])
        all_oot_tickers.extend(test_tickers.tolist())

        print(f"  -> Acc: {acc:.4f} | AUC: {auc:.4f} | "
              f"Prec(up/dn): {prec_pos:.3f}/{prec_neg:.3f} | "
              f"Rec(up/dn): {rec_pos:.3f}/{rec_neg:.3f}")

        fold_id += 1
        start_idx += test_days  # Roll forward by test window size

    return fold_results, np.array(all_oot_preds), np.array(all_oot_labels), all_oot_dates, all_oot_tickers


# ─── Permutation Test ─────────────────────────────────────────────────────────
def permutation_test(X, y, dates, real_auc, n_permutations=5):
    """Shuffle labels and retrain to establish null distribution."""
    print(f"\n[PERMUTATION TEST] Running {n_permutations} permutations...")

    perm_aucs = []
    unique_dates = np.sort(np.unique(dates))
    train_days = TRAIN_YEARS * 252
    test_days = TEST_MONTHS * 21

    for perm in range(n_permutations):
        # Shuffle labels
        y_shuffled = y.copy()
        np.random.shuffle(y_shuffled)

        # Run one fold (use the last fold for speed)
        start_idx = len(unique_dates) - train_days - test_days
        if start_idx < 0:
            start_idx = 0

        train_end_idx = start_idx + train_days
        test_end_idx = min(train_end_idx + test_days, len(unique_dates))

        train_start_date = unique_dates[start_idx]
        train_end_date = unique_dates[train_end_idx - 1]
        test_start_date = unique_dates[train_end_idx]
        test_end_date = unique_dates[test_end_idx - 1]

        train_mask = (dates >= train_start_date) & (dates <= train_end_date)
        test_mask = (dates >= test_start_date) & (dates <= test_end_date)

        X_tr, y_tr = X[train_mask], y_shuffled[train_mask]
        X_te, y_te = X[test_mask], y_shuffled[test_mask]

        if len(X_tr) < 500 or len(X_te) < 50:
            continue

        _, _, preds, _ = train_one_fold(X_tr, y_tr, X_te, y_te, f"perm_{perm}")

        try:
            pauc = roc_auc_score(y_te, preds)
        except ValueError:
            pauc = 0.5

        perm_aucs.append(pauc)
        print(f"  Permutation {perm+1}/{n_permutations}: AUC = {pauc:.4f}")

    perm_mean = np.mean(perm_aucs)
    perm_std = np.std(perm_aucs) if len(perm_aucs) > 1 else 0.01
    z_score = (real_auc - perm_mean) / max(perm_std, 1e-6)

    print(f"\n  Real AUC: {real_auc:.4f}")
    print(f"  Permuted AUC: {perm_mean:.4f} ± {perm_std:.4f}")
    print(f"  Z-score: {z_score:.2f} (must be > 2.0)")
    print(f"  PASS: {'YES' if z_score > 2.0 else 'NO'}")

    return {
        'real_auc': round(real_auc, 4),
        'perm_mean': round(perm_mean, 4),
        'perm_std': round(perm_std, 4),
        'z_score': round(z_score, 2),
        'passed': z_score > 2.0,
        'perm_aucs': [round(a, 4) for a in perm_aucs],
    }


# ─── Feature Importance ──────────────────────────────────────────────────────
def compute_feature_importance(X, y, dates):
    """Permutation-based feature importance on last fold."""
    print("\n[FEATURE IMPORTANCE] Computing via permutation...")

    unique_dates = np.sort(np.unique(dates))
    train_days = TRAIN_YEARS * 252
    test_days = TEST_MONTHS * 21

    start_idx = len(unique_dates) - train_days - test_days
    if start_idx < 0:
        start_idx = 0

    train_end_idx = start_idx + train_days
    test_end_idx = min(train_end_idx + test_days, len(unique_dates))

    train_mask = (dates >= unique_dates[start_idx]) & (dates <= unique_dates[train_end_idx - 1])
    test_mask = (dates >= unique_dates[train_end_idx]) & (dates <= unique_dates[test_end_idx - 1])

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    # Train baseline model
    model, scaler, base_preds, _ = train_one_fold(X_tr, y_tr, X_te, y_te, 'feat_imp')
    try:
        base_auc = roc_auc_score(y_te, base_preds)
    except ValueError:
        base_auc = 0.5

    feature_names = ['returns', 'volume_ratio', 'rsi', 'boll_pct_b', 'macd_hist',
                     'atr_ratio', 'vix', 'vix_term', 'sector_rel',
                     'mom5', 'realized_vol', 'dist_from_high', 'vol_trend',
                     'price_to_sma50', 'overnight_gap']

    importances = {}
    for fidx, fname in enumerate(feature_names):
        # Shuffle one feature across the time dimension
        X_te_perm = X_te.copy()
        perm_idx = np.random.permutation(len(X_te_perm))
        X_te_perm[:, :, fidx] = X_te_perm[perm_idx, :, fidx]

        # Re-predict
        N_te, L, F = X_te_perm.shape
        X_te_flat = scaler.transform(X_te_perm.reshape(-1, F)).reshape(N_te, L, F)

        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X_te_flat, dtype=torch.float32).to(DEVICE)
            preds = torch.sigmoid(model(xt)).cpu().numpy()

        try:
            perm_auc = roc_auc_score(y_te, preds)
        except ValueError:
            perm_auc = 0.5

        drop = base_auc - perm_auc
        importances[fname] = round(drop, 4)
        print(f"  {fname}: AUC drop = {drop:+.4f}")

    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
    return importances


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    start_time = dt.datetime.now()
    print(f"{'='*70}")
    print(f"  Entry Timing Model v1 — Stock Return Classifier")
    print(f"  Started: {start_time}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download data
    data = download_data()

    if len(data) < 10:
        print("ERROR: Too few tickers downloaded. Aborting.")
        sys.exit(1)

    # 2. Build dataset
    X, y, dates, tickers = build_dataset(data)

    # 3. Walk-forward training
    fold_results, all_preds, all_labels, all_dates, all_tickers = walk_forward_train(X, y, dates, tickers)

    if len(fold_results) == 0:
        print("ERROR: No valid folds. Aborting.")
        sys.exit(1)

    # 4. Aggregate OOT metrics
    agg_binary = (all_preds >= 0.5).astype(int)
    agg_acc = accuracy_score(all_labels, agg_binary)
    try:
        agg_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        agg_auc = 0.5

    print(f"\n{'='*70}")
    print(f"  AGGREGATE OOT RESULTS")
    print(f"  Accuracy: {agg_acc:.4f}")
    print(f"  AUC: {agg_auc:.4f}")
    print(f"  Samples: {len(all_labels)}")
    print(f"  Class balance: {np.mean(all_labels):.3f} positive")
    print(f"{'='*70}")
    print(classification_report(all_labels, agg_binary, target_names=['Down', 'Up/Flat']))

    # 5. Permutation test
    perm_results = permutation_test(X, y, dates, agg_auc, n_permutations=5)

    # 6. Feature importance
    feat_imp = compute_feature_importance(X, y, dates)

    # 7. Save results
    elapsed = (dt.datetime.now() - start_time).total_seconds()

    results = {
        'model': 'CNN1D_Classifier_v1',
        'device': str(DEVICE),
        'tickers': TICKERS,
        'n_tickers_used': len([t for t in TICKERS if t in data]),
        'lookback_days': LOOKBACK,
        'forward_days': FORWARD_DAYS,
        'train_years': TRAIN_YEARS,
        'test_months': TEST_MONTHS,
        'total_samples': len(X),
        'aggregate_accuracy': round(agg_acc, 4),
        'aggregate_auc': round(agg_auc, 4),
        'per_fold': fold_results,
        'permutation_test': perm_results,
        'feature_importance': feat_imp,
        'elapsed_seconds': round(elapsed, 1),
        'timestamp': str(dt.datetime.now()),
    }

    results_path = OUTPUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n[SAVED] Results: {results_path}")

    # Save predictions
    preds_df = pd.DataFrame({
        'date': all_dates,
        'ticker_id': all_tickers,
        'prediction': all_preds,
        'label': all_labels,
    })
    preds_path = OUTPUT_DIR / 'oot_predictions.csv'
    preds_df.to_csv(preds_path, index=False)
    print(f"[SAVED] OOT predictions: {preds_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"  Model: 1D-CNN Classifier (3-layer conv + FC)")
    print(f"  Walk-forward folds: {len(fold_results)}")
    print(f"  Aggregate OOT Accuracy: {agg_acc:.4f}")
    print(f"  Aggregate OOT AUC: {agg_auc:.4f}")
    print(f"  Permutation test: {'PASSED' if perm_results['passed'] else 'FAILED'} "
          f"(z={perm_results['z_score']:.1f})")
    print(f"  Top features: {list(feat_imp.keys())[:3]}")
    print(f"  Runtime: {elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    main()
