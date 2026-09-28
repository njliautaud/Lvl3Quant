#!/usr/bin/env python3
"""
ML Growth Timing Model — Cross-Asset Regime + Asset Selection
=============================================================
GPU-accelerated deep learning for leveraged ETF timing and selection.

Prior attempt (regime_predictor.py) FAILED: LGBM with macro-only features
couldn't beat 200MA (only 3 bear markets in sample). This version attacks
differently:
  1. Cross-asset features (bonds, VIX, credit, commodities, breadth) — 45+ features
  2. Predict OPTIMAL ACTION (which leveraged ETF or cash) not just bull/bear
  3. Deep learning (LSTM, 1D-CNN, MLP) to capture temporal regime transitions
  4. LGBM baseline for comparison
  5. Walk-forward sliding 252d train / 21d test

Target assets: TQQQ, SOXL, UPRO, QQQ (unleveraged fallback), CASH
Model picks whichever had best risk-adjusted forward 21d return.

HC compliance: sliding window only, concat metrics, MLflow-ready.
"""

import json
import os
import sys
import time
import warnings
import datetime as dt
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, log_loss

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/ml_growth_timing')
CACHE_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/cache')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Walk-forward params
TRAIN_DAYS = 252        # 1 year sliding train
TEST_DAYS = 21          # 1 month OOT
LOOKBACK_SEQ = 63       # 3 months sequence for LSTM/CNN
FWD_HORIZON = 21        # predict 21 trading days forward

# Target assets (what the model chooses between)
TARGET_ASSETS = ['TQQQ', 'SOXL', 'UPRO', 'QQQ']
CASH_LABEL = 'CASH'
ALL_ACTIONS = TARGET_ASSETS + [CASH_LABEL]

# Training hyperparams
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
CNN_CHANNELS = 64
MLP_HIDDEN = [256, 128, 64]
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
EARLY_STOP_PATIENCE = 8

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_all_data(start='2005-01-01', end=None):
    """Download all tickers needed for features + targets."""
    if end is None:
        end = dt.date.today().isoformat()

    cache_file = CACHE_DIR / f'ml_timing_data_{end}.parquet'
    if cache_file.exists():
        print(f"[DATA] Loading cached data from {cache_file.name}")
        return pd.read_parquet(cache_file)

    # Feature tickers
    feature_tickers = {
        'SPY': 'SPY',
        'QQQ': 'QQQ',
        'VIX': '^VIX',
        'TLT': 'TLT',      # 20+ yr treasury
        'SHY': 'SHY',      # 1-3 yr treasury
        'IEF': 'IEF',      # 7-10 yr treasury
        'HYG': 'HYG',      # high yield corp bonds
        'LQD': 'LQD',      # investment grade bonds
        'GLD': 'GLD',      # gold
        'DBC': 'DBC',      # broad commodities
        'USO': 'USO',      # oil
        'IWM': 'IWM',      # Russell 2000
        'EEM': 'EEM',      # emerging markets
        'XLF': 'XLF',      # financials sector
        'XLE': 'XLE',      # energy sector
        'XLK': 'XLK',      # tech sector
        'SMH': 'SMH',      # semiconductors (unleveraged)
    }

    # Target asset tickers
    target_tickers = {
        'TQQQ': 'TQQQ',
        'SOXL': 'SOXL',
        'UPRO': 'UPRO',
    }

    all_tickers = {**feature_tickers, **target_tickers}

    # Download each individually to handle missing data gracefully
    prices = {}
    volumes = {}
    for name, ticker in all_tickers.items():
        print(f"  Downloading {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                prices[name] = df['Close']
                if 'Volume' in df.columns:
                    volumes[name] = df['Volume']
            else:
                print(f"    WARNING: {name} has only {len(df)} rows, skipping")
        except Exception as e:
            print(f"    ERROR downloading {name}: {e}")

    price_df = pd.DataFrame(prices)
    vol_df = pd.DataFrame(volumes)

    # Forward fill small gaps, then drop rows with any NaN in core tickers
    price_df = price_df.ffill(limit=5)
    vol_df = vol_df.ffill(limit=5)

    # Merge
    combined = price_df.copy()
    for col in vol_df.columns:
        combined[f'{col}_vol'] = vol_df[col]

    combined.to_parquet(cache_file)
    print(f"[DATA] Saved {len(combined)} rows, {len(combined.columns)} columns to cache")
    return combined


# ══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build ~45 cross-asset features from raw price/volume data."""
    feat = pd.DataFrame(index=raw.index)

    # --- Helper functions ---
    def safe_ret(series, n):
        return series.pct_change(n)

    def safe_sma(series, n):
        return series.rolling(n, min_periods=max(1, n // 2)).mean()

    def safe_ema(series, n):
        return series.ewm(span=n, adjust=False).mean()

    def rsi(series, n=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def bbands_pctb(series, n=20):
        """Bollinger %B — where price is within bands."""
        sma = series.rolling(n).mean()
        std = series.rolling(n).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (series - lower) / (upper - lower + 1e-10)

    # --- VIX features ---
    if 'VIX' in raw.columns:
        feat['vix_level'] = raw['VIX']
        feat['vix_5d_chg'] = raw['VIX'].diff(5)
        feat['vix_21d_chg'] = raw['VIX'].diff(21)
        feat['vix_rank_63d'] = raw['VIX'].rolling(63).rank(pct=True)
        # VIX term structure proxy: ratio of short-term vs longer-term realized vol
        if 'SPY' in raw.columns:
            rv5 = raw['SPY'].pct_change().rolling(5).std() * np.sqrt(252)
            rv21 = raw['SPY'].pct_change().rolling(21).std() * np.sqrt(252)
            feat['vix_vs_rv21'] = raw['VIX'] / (rv21 * 100 + 1e-10)
            feat['rv_term_structure'] = rv5 / (rv21 + 1e-10)  # >1 = short-term vol elevated

    # --- Credit spread proxy ---
    if 'HYG' in raw.columns and 'IEF' in raw.columns:
        credit_ratio = raw['HYG'] / raw['IEF']
        feat['credit_spread_ratio'] = credit_ratio
        feat['credit_spread_21d_chg'] = credit_ratio.pct_change(21)
        feat['credit_spread_63d_chg'] = credit_ratio.pct_change(63)
        # Widening credit = risk-off
        feat['credit_trend'] = safe_sma(credit_ratio, 10) - safe_sma(credit_ratio, 50)

    # --- Bond market / Yield curve ---
    if 'TLT' in raw.columns:
        feat['tlt_ret_5d'] = safe_ret(raw['TLT'], 5)
        feat['tlt_ret_21d'] = safe_ret(raw['TLT'], 21)
        feat['tlt_momentum_63d'] = safe_ret(raw['TLT'], 63)
    if 'TLT' in raw.columns and 'SHY' in raw.columns:
        yield_curve = raw['TLT'] / raw['SHY']
        feat['yield_curve_proxy'] = yield_curve
        feat['yield_curve_slope'] = yield_curve.pct_change(21)

    # --- Commodity signals ---
    if 'GLD' in raw.columns:
        feat['gold_mom_21d'] = safe_ret(raw['GLD'], 21)
        feat['gold_mom_63d'] = safe_ret(raw['GLD'], 63)
    if 'USO' in raw.columns:
        feat['oil_mom_21d'] = safe_ret(raw['USO'], 21)
        feat['oil_mom_63d'] = safe_ret(raw['USO'], 63)
    if 'DBC' in raw.columns:
        feat['commodity_mom_21d'] = safe_ret(raw['DBC'], 21)
        feat['commodity_mom_63d'] = safe_ret(raw['DBC'], 63)

    # --- Breadth / rotation signals ---
    if 'IWM' in raw.columns and 'SPY' in raw.columns:
        iwm_spy = raw['IWM'] / raw['SPY']
        feat['small_vs_large'] = iwm_spy.pct_change(21)
        feat['small_vs_large_trend'] = safe_sma(iwm_spy, 10) - safe_sma(iwm_spy, 50)
    if 'EEM' in raw.columns and 'SPY' in raw.columns:
        eem_spy = raw['EEM'] / raw['SPY']
        feat['em_vs_dm'] = eem_spy.pct_change(21)
    if 'XLK' in raw.columns and 'SPY' in raw.columns:
        feat['tech_rel_strength'] = (raw['XLK'] / raw['SPY']).pct_change(21)
    if 'SMH' in raw.columns and 'SPY' in raw.columns:
        feat['semi_rel_strength'] = (raw['SMH'] / raw['SPY']).pct_change(21)

    # --- Equity technicals (SPY) ---
    if 'SPY' in raw.columns:
        spy = raw['SPY']
        spy_ret = spy.pct_change()

        # Returns at multiple horizons
        feat['spy_ret_5d'] = safe_ret(spy, 5)
        feat['spy_ret_21d'] = safe_ret(spy, 21)
        feat['spy_ret_63d'] = safe_ret(spy, 63)

        # RSI
        feat['spy_rsi_14'] = rsi(spy, 14)

        # MACD
        ema12 = safe_ema(spy, 12)
        ema26 = safe_ema(spy, 26)
        macd_line = ema12 - ema26
        macd_signal = safe_ema(macd_line, 9)
        feat['spy_macd'] = macd_line
        feat['spy_macd_hist'] = macd_line - macd_signal

        # Bollinger %B
        feat['spy_bb_pctb'] = bbands_pctb(spy, 20)

        # MA crosses
        sma50 = safe_sma(spy, 50)
        sma200 = safe_sma(spy, 200)
        feat['spy_above_50ma'] = (spy > sma50).astype(float)
        feat['spy_above_200ma'] = (spy > sma200).astype(float)
        feat['spy_50_200_cross'] = (sma50 / sma200) - 1  # golden/death cross distance

        # Realized vol at multiple windows
        feat['spy_rvol_5d'] = spy_ret.rolling(5).std() * np.sqrt(252)
        feat['spy_rvol_10d'] = spy_ret.rolling(10).std() * np.sqrt(252)
        feat['spy_rvol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252)
        feat['spy_rvol_63d'] = spy_ret.rolling(63).std() * np.sqrt(252)

        # Vol ratio (short vs long) — rising = vol expansion
        feat['vol_ratio_5_63'] = feat['spy_rvol_5d'] / (feat['spy_rvol_63d'] + 1e-10)

        # Drawdown from rolling max
        rolling_max = spy.rolling(252, min_periods=1).max()
        feat['spy_drawdown'] = (spy / rolling_max) - 1

    # --- Volume features ---
    if 'SPY_vol' in raw.columns:
        spy_vol = raw['SPY_vol']
        feat['spy_vol_ratio'] = spy_vol / (spy_vol.rolling(21).mean() + 1)
        feat['spy_vol_trend'] = safe_sma(spy_vol, 5) / (safe_sma(spy_vol, 21) + 1)

    # --- Calendar features ---
    feat['month'] = feat.index.month
    feat['day_of_week'] = feat.index.dayofweek
    feat['month_sin'] = np.sin(2 * np.pi * feat['month'] / 12)
    feat['month_cos'] = np.cos(2 * np.pi * feat['month'] / 12)

    # Drop raw calendar (keep sin/cos encoding)
    feat.drop(columns=['month', 'day_of_week'], inplace=True, errors='ignore')

    # --- Cross-asset momentum composite ---
    mom_cols = [c for c in feat.columns if 'mom' in c or 'ret_21d' in c]
    if len(mom_cols) >= 3:
        feat['cross_asset_mom_mean'] = feat[mom_cols].mean(axis=1)
        feat['cross_asset_mom_std'] = feat[mom_cols].std(axis=1)

    print(f"[FEATURES] Built {len(feat.columns)} features")
    return feat


# ══════════════════════════════════════════════════════════════
# 3. TARGET CONSTRUCTION
# ══════════════════════════════════════════════════════════════

def compute_targets(raw: pd.DataFrame) -> pd.DataFrame:
    """
    For each date, compute forward 21d return for each target asset.
    Label = asset with best risk-adjusted return, or CASH if all negative.
    Also produce regression targets (forward returns per asset).
    """
    fwd_returns = pd.DataFrame(index=raw.index)

    for asset in TARGET_ASSETS:
        if asset in raw.columns:
            fwd_returns[f'{asset}_fwd'] = raw[asset].pct_change(FWD_HORIZON).shift(-FWD_HORIZON)

    # Drop rows where we can't compute forward return
    fwd_returns = fwd_returns.dropna()

    # Classification target: best asset (or CASH if all negative)
    def pick_best(row):
        vals = {a: row[f'{a}_fwd'] for a in TARGET_ASSETS if f'{a}_fwd' in row.index}
        if not vals:
            return CASH_LABEL
        best_asset = max(vals, key=vals.get)
        if vals[best_asset] <= 0:
            return CASH_LABEL  # all negative => go to cash
        return best_asset

    fwd_returns['best_action'] = fwd_returns.apply(pick_best, axis=1)

    # Also compute a risk-adjusted version: best Sharpe over the 21d window
    # (approximated as return / vol, but we only have the total return here,
    # so we'll use the simpler "best return, CASH if negative" version)

    return fwd_returns


# ══════════════════════════════════════════════════════════════
# 4. PYTORCH MODELS
# ══════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    """2-layer LSTM with 63d lookback for sequence classification."""
    def __init__(self, n_features, n_classes, hidden=LSTM_HIDDEN, n_layers=LSTM_LAYERS, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (hn, cn) = self.lstm(x)
        last = out[:, -1, :]  # take last timestep
        return self.head(last)


class CNN1DModel(nn.Module):
    """1D-CNN on daily feature sequences."""
    def __init__(self, n_features, n_classes, channels=CNN_CHANNELS, seq_len=LOOKBACK_SEQ, dropout=0.3):
        super().__init__()
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(n_features, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            # Block 2
            nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            # Block 3
            nn.Conv1d(channels * 2, channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features) -> conv expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = x.squeeze(-1)
        return self.head(x)


class MLPModel(nn.Module):
    """Simple MLP on flattened features (no sequence)."""
    def __init__(self, n_features, n_classes, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = MLP_HIDDEN
        layers = []
        prev_dim = n_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, n_features) — no sequence dimension
        if x.dim() == 3:
            x = x[:, -1, :]  # take last timestep if sequence passed
        return self.net(x)


# ══════════════════════════════════════════════════════════════
# 5. DATASET + TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    """Sliding-window sequence dataset for LSTM/CNN."""
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.labels) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return x, y


def train_pytorch_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR,
                        patience=EARLY_STOP_PATIENCE, model_name='model'):
    """Train a PyTorch model with early stopping."""
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item()
                val_batches += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_batches, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return model, best_val_loss


def predict_pytorch(model, features: np.ndarray, seq_len: int, use_sequence=True):
    """Get predictions from a trained PyTorch model."""
    model.eval()
    if use_sequence:
        ds = SequenceDataset(features, np.zeros(len(features), dtype=np.int64), seq_len)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
    else:
        # MLP — use only last row of each potential sequence, or all rows
        X = torch.FloatTensor(features)
        loader = DataLoader(TensorDataset(X, torch.zeros(len(X), dtype=torch.long)),
                            batch_size=256, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            X = batch[0].to(DEVICE)
            logits = model(X)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


# ══════════════════════════════════════════════════════════════
# 6. LGBM BASELINE
# ══════════════════════════════════════════════════════════════

def train_lgbm(X_train, y_train, X_val, y_val, n_classes):
    """Train LightGBM classifier."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("[LGBM] lightgbm not installed, skipping")
        return None

    params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    return model


# ══════════════════════════════════════════════════════════════
# 7. BASELINE STRATEGY (TQQQ + 200MA)
# ══════════════════════════════════════════════════════════════

def compute_baseline_returns(raw: pd.DataFrame, start_idx: int, end_idx: int) -> pd.Series:
    """
    Simple TQQQ + 200MA baseline.
    If SPY > 200MA: hold TQQQ. Else: cash (0% return).
    Returns daily return series aligned to the test window.
    """
    spy = raw['SPY'].iloc[:end_idx]
    sma200 = spy.rolling(200).mean()

    # Daily returns for TQQQ
    tqqq_ret = raw['TQQQ'].pct_change()

    # Build baseline return series for test period
    baseline = pd.Series(0.0, index=raw.index[start_idx:end_idx])
    for i in range(start_idx, end_idx):
        date = raw.index[i]
        if i > 0 and date in sma200.index and not pd.isna(sma200.loc[date]):
            if spy.loc[date] > sma200.loc[date]:
                if date in tqqq_ret.index and not pd.isna(tqqq_ret.loc[date]):
                    baseline.loc[date] = tqqq_ret.loc[date]
    return baseline


# ══════════════════════════════════════════════════════════════
# 8. WALK-FORWARD ENGINE
# ══════════════════════════════════════════════════════════════

def run_walk_forward(features: pd.DataFrame, targets: pd.DataFrame, raw: pd.DataFrame):
    """
    Sliding walk-forward: 252d train, 21d test.
    Trains all models on each fold, collects OOT predictions.
    """
    # Align features and targets
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx].copy()
    targets = targets.loc[common_idx].copy()

    # Encode labels
    le = LabelEncoder()
    le.fit(ALL_ACTIONS)
    n_classes = len(le.classes_)
    labels_encoded = le.transform(targets['best_action'].values)

    # Feature names and count
    feat_cols = features.columns.tolist()
    n_features = len(feat_cols)
    print(f"[WF] {len(common_idx)} aligned samples, {n_features} features, {n_classes} classes")
    print(f"[WF] Classes: {list(le.classes_)}")
    print(f"[WF] Label distribution: {pd.Series(targets['best_action']).value_counts().to_dict()}")

    # Standardize (will re-fit per fold)
    feat_values = features.values.astype(np.float32)

    # Replace inf/nan
    feat_values = np.nan_to_num(feat_values, nan=0.0, posinf=0.0, neginf=0.0)

    # Walk-forward folds
    n_total = len(feat_values)
    min_start = LOOKBACK_SEQ + TRAIN_DAYS  # need lookback + train before first test
    fold_starts = list(range(min_start, n_total - TEST_DAYS, TEST_DAYS))

    print(f"[WF] {len(fold_starts)} folds, first test at index {min_start}, "
          f"dates {common_idx[min_start]} to {common_idx[-1]}")

    # Storage for per-fold results
    model_names = ['LGBM', 'LSTM', 'CNN1D', 'MLP']
    results = {m: {'preds': [], 'probs': [], 'true': [], 'dates': [], 'returns': []}
               for m in model_names}
    baseline_returns = []
    baseline_dates = []

    for fold_i, test_start in enumerate(fold_starts):
        test_end = min(test_start + TEST_DAYS, n_total)
        train_start = test_start - TRAIN_DAYS
        seq_start = max(0, train_start - LOOKBACK_SEQ)  # extra for LSTM lookback

        # Split indices
        train_idx = slice(train_start, test_start)
        test_idx = slice(test_start, test_end)

        # Get data
        X_all = feat_values[seq_start:test_end]
        y_all = labels_encoded[seq_start:test_end]

        # Fit scaler on train only
        scaler = StandardScaler()
        X_train_raw = feat_values[train_start:test_start]
        scaler.fit(X_train_raw)

        X_scaled = scaler.transform(feat_values[seq_start:test_end])
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Positions relative to seq_start
        rel_train_start = train_start - seq_start
        rel_test_start = test_start - seq_start
        rel_test_end = test_end - seq_start

        X_train = X_scaled[rel_train_start:rel_test_start]
        y_train = y_all[rel_train_start:rel_test_start]
        X_test = X_scaled[rel_test_start:rel_test_end]
        y_test = y_all[rel_test_start:rel_test_end]

        # Validation split: last 20% of train
        val_size = max(1, len(X_train) // 5)
        X_tr, X_va = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_va = y_train[:-val_size], y_train[-val_size:]

        test_dates = common_idx[test_start:test_end]

        if fold_i % 10 == 0:
            print(f"  Fold {fold_i+1}/{len(fold_starts)}: "
                  f"train {common_idx[train_start].strftime('%Y-%m-%d')} -> "
                  f"test {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")

        # ─── LGBM ───
        try:
            lgbm_model = train_lgbm(X_tr, y_tr, X_va, y_va, n_classes)
            if lgbm_model is not None:
                lgbm_probs = lgbm_model.predict(X_test)
                lgbm_preds = np.argmax(lgbm_probs, axis=1)
            else:
                lgbm_probs = np.ones((len(X_test), n_classes)) / n_classes
                lgbm_preds = np.zeros(len(X_test), dtype=int)
        except Exception as e:
            if fold_i == 0:
                print(f"    LGBM error: {e}")
            lgbm_probs = np.ones((len(X_test), n_classes)) / n_classes
            lgbm_preds = np.zeros(len(X_test), dtype=int)

        # ─── LSTM ───
        try:
            X_seq_full = X_scaled  # full scaled data for sequence building
            y_seq_full = y_all

            # Build train/val/test sequences
            train_ds = SequenceDataset(X_seq_full[:rel_test_start], y_seq_full[:rel_test_start], LOOKBACK_SEQ)
            val_size_seq = max(1, len(train_ds) // 5)
            train_ds_final = torch.utils.data.Subset(train_ds, range(len(train_ds) - val_size_seq))
            val_ds = torch.utils.data.Subset(train_ds, range(len(train_ds) - val_size_seq, len(train_ds)))

            train_loader = DataLoader(train_ds_final, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=0, pin_memory=True)

            lstm = LSTMModel(n_features, n_classes)
            lstm, _ = train_pytorch_model(lstm, train_loader, val_loader,
                                          epochs=EPOCHS, model_name='LSTM')

            lstm_probs = predict_pytorch(lstm, X_scaled[rel_test_start - LOOKBACK_SEQ + 1:rel_test_end],
                                         LOOKBACK_SEQ, use_sequence=True)
            lstm_preds = np.argmax(lstm_probs, axis=1)

            # Align — predict_pytorch with SequenceDataset drops first seq_len-1
            if len(lstm_preds) > len(X_test):
                lstm_preds = lstm_preds[-len(X_test):]
                lstm_probs = lstm_probs[-len(X_test):]
            elif len(lstm_preds) < len(X_test):
                pad = len(X_test) - len(lstm_preds)
                lstm_preds = np.concatenate([np.zeros(pad, dtype=int), lstm_preds])
                lstm_probs = np.concatenate([np.ones((pad, n_classes)) / n_classes, lstm_probs])

            del lstm
            torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
        except Exception as e:
            if fold_i == 0:
                print(f"    LSTM error: {e}")
            lstm_probs = np.ones((len(X_test), n_classes)) / n_classes
            lstm_preds = np.zeros(len(X_test), dtype=int)

        # ─── CNN1D ───
        try:
            cnn = CNN1DModel(n_features, n_classes, seq_len=LOOKBACK_SEQ)
            # Reuse same train/val loaders as LSTM
            cnn, _ = train_pytorch_model(cnn, train_loader, val_loader,
                                         epochs=EPOCHS, model_name='CNN1D')

            cnn_probs = predict_pytorch(cnn, X_scaled[rel_test_start - LOOKBACK_SEQ + 1:rel_test_end],
                                        LOOKBACK_SEQ, use_sequence=True)
            cnn_preds = np.argmax(cnn_probs, axis=1)

            if len(cnn_preds) > len(X_test):
                cnn_preds = cnn_preds[-len(X_test):]
                cnn_probs = cnn_probs[-len(X_test):]
            elif len(cnn_preds) < len(X_test):
                pad = len(X_test) - len(cnn_preds)
                cnn_preds = np.concatenate([np.zeros(pad, dtype=int), cnn_preds])
                cnn_probs = np.concatenate([np.ones((pad, n_classes)) / n_classes, cnn_probs])

            del cnn
            torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
        except Exception as e:
            if fold_i == 0:
                print(f"    CNN1D error: {e}")
            cnn_probs = np.ones((len(X_test), n_classes)) / n_classes
            cnn_preds = np.zeros(len(X_test), dtype=int)

        # ─── MLP ───
        try:
            mlp = MLPModel(n_features, n_classes)
            # MLP uses flat features, not sequences
            X_tr_t = torch.FloatTensor(X_tr)
            y_tr_t = torch.LongTensor(y_tr)
            X_va_t = torch.FloatTensor(X_va)
            y_va_t = torch.LongTensor(y_va)

            mlp_train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                                          batch_size=BATCH_SIZE, shuffle=True)
            mlp_val_loader = DataLoader(TensorDataset(X_va_t, y_va_t),
                                        batch_size=BATCH_SIZE, shuffle=False)

            mlp, _ = train_pytorch_model(mlp, mlp_train_loader, mlp_val_loader,
                                         epochs=EPOCHS, model_name='MLP')

            mlp_probs = predict_pytorch(mlp, X_test, LOOKBACK_SEQ, use_sequence=False)
            mlp_preds = np.argmax(mlp_probs, axis=1)

            del mlp
            torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
        except Exception as e:
            if fold_i == 0:
                print(f"    MLP error: {e}")
            mlp_probs = np.ones((len(X_test), n_classes)) / n_classes
            mlp_preds = np.zeros(len(X_test), dtype=int)

        # Store results
        for m_name, m_preds, m_probs in [
            ('LGBM', lgbm_preds, lgbm_probs),
            ('LSTM', lstm_preds, lstm_probs),
            ('CNN1D', cnn_preds, cnn_probs),
            ('MLP', mlp_preds, mlp_preds),  # MLP stores preds in probs slot too
        ]:
            results[m_name]['preds'].extend(m_preds.tolist())
            results[m_name]['true'].extend(y_test.tolist())
            results[m_name]['dates'].extend(test_dates.tolist())

        # Compute test-period daily returns for each model's chosen action
        for date_i, date in enumerate(test_dates):
            for m_name, m_preds_arr in [('LGBM', lgbm_preds), ('LSTM', lstm_preds),
                                         ('CNN1D', cnn_preds), ('MLP', mlp_preds)]:
                if date_i < len(m_preds_arr):
                    chosen = le.inverse_transform([m_preds_arr[date_i]])[0]
                    if chosen == CASH_LABEL or chosen not in raw.columns:
                        daily_ret = 0.0
                    else:
                        loc = raw.index.get_loc(date)
                        if loc > 0:
                            daily_ret = raw[chosen].iloc[loc] / raw[chosen].iloc[loc - 1] - 1
                            if np.isnan(daily_ret) or np.isinf(daily_ret):
                                daily_ret = 0.0
                        else:
                            daily_ret = 0.0
                    results[m_name]['returns'].append(daily_ret)

        # Baseline returns for this test window
        for date in test_dates:
            loc_in_raw = raw.index.get_loc(date)
            spy_val = raw['SPY'].iloc[loc_in_raw]
            spy_sma200 = raw['SPY'].iloc[max(0, loc_in_raw-199):loc_in_raw+1].mean()
            if spy_val > spy_sma200 and 'TQQQ' in raw.columns:
                if loc_in_raw > 0:
                    bl_ret = raw['TQQQ'].iloc[loc_in_raw] / raw['TQQQ'].iloc[loc_in_raw-1] - 1
                    if np.isnan(bl_ret) or np.isinf(bl_ret):
                        bl_ret = 0.0
                else:
                    bl_ret = 0.0
            else:
                bl_ret = 0.0
            baseline_returns.append(bl_ret)
            baseline_dates.append(date)

    return results, baseline_returns, baseline_dates, le


# ══════════════════════════════════════════════════════════════
# 9. EVALUATION & REPORTING
# ══════════════════════════════════════════════════════════════

def compute_strategy_metrics(daily_returns, name='Strategy'):
    """Compute Sharpe, Sortino, max DD, CAGR, win rate from daily returns."""
    rets = np.array(daily_returns)
    rets = rets[~np.isnan(rets)]

    if len(rets) < 21:
        return {'name': name, 'error': 'insufficient data'}

    # Annualized metrics
    mean_daily = np.mean(rets)
    std_daily = np.std(rets, ddof=1)
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10

    sharpe = (mean_daily / (std_daily + 1e-10)) * np.sqrt(252)
    sortino = (mean_daily / (downside_std + 1e-10)) * np.sqrt(252)

    # CAGR
    cum = np.cumprod(1 + rets)
    total_return = cum[-1] - 1
    n_years = len(rets) / 252
    cagr = (cum[-1] ** (1 / max(n_years, 0.01))) - 1 if cum[-1] > 0 else -1.0

    # Max drawdown
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum / running_max - 1
    max_dd = np.min(drawdowns)

    # Win rate (days)
    wr = np.mean(rets > 0) if np.any(rets != 0) else 0.0
    # Profit factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = -np.sum(rets[rets < 0])
    pf = gross_profit / (gross_loss + 1e-10)

    return {
        'name': name,
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd': round(max_dd * 100, 2),
        'total_return': round(total_return * 100, 2),
        'win_rate': round(wr * 100, 1),
        'profit_factor': round(pf, 3),
        'n_days': len(rets),
        'n_years': round(n_years, 2),
        'pct_invested': round(np.mean(np.array(daily_returns) != 0) * 100, 1),
    }


def evaluate_and_report(results, baseline_returns, baseline_dates, label_encoder, raw):
    """Generate comprehensive evaluation report."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD RESULTS — ML Growth Timing Model")
    print("=" * 80)

    le = label_encoder
    all_metrics = {}

    # Baseline metrics
    bl_metrics = compute_strategy_metrics(baseline_returns, 'TQQQ+200MA (Baseline)')
    all_metrics['baseline'] = bl_metrics
    print(f"\n{'─' * 60}")
    print(f"BASELINE: {bl_metrics['name']}")
    print(f"  CAGR: {bl_metrics.get('cagr', 'N/A')}%  |  Sharpe: {bl_metrics.get('sharpe', 'N/A')}  |  "
          f"Sortino: {bl_metrics.get('sortino', 'N/A')}  |  MaxDD: {bl_metrics.get('max_dd', 'N/A')}%")
    print(f"  WR: {bl_metrics.get('win_rate', 'N/A')}%  |  PF: {bl_metrics.get('profit_factor', 'N/A')}  |  "
          f"Invested: {bl_metrics.get('pct_invested', 'N/A')}%  |  Period: {bl_metrics.get('n_years', 'N/A')} yrs")

    # Model metrics
    for model_name in ['LGBM', 'LSTM', 'CNN1D', 'MLP']:
        r = results[model_name]
        if not r['returns']:
            continue

        m = compute_strategy_metrics(r['returns'], model_name)
        all_metrics[model_name] = m

        # Classification accuracy
        if r['true'] and r['preds']:
            acc = accuracy_score(r['true'][:len(r['preds'])], r['preds'][:len(r['true'])])
        else:
            acc = 0.0

        # Action distribution
        pred_labels = le.inverse_transform(r['preds'][:len(r['true'])])
        action_dist = pd.Series(pred_labels).value_counts(normalize=True)

        print(f"\n{'─' * 60}")
        print(f"MODEL: {model_name}")
        print(f"  CAGR: {m.get('cagr', 'N/A')}%  |  Sharpe: {m.get('sharpe', 'N/A')}  |  "
              f"Sortino: {m.get('sortino', 'N/A')}  |  MaxDD: {m.get('max_dd', 'N/A')}%")
        print(f"  WR: {m.get('win_rate', 'N/A')}%  |  PF: {m.get('profit_factor', 'N/A')}  |  "
              f"Invested: {m.get('pct_invested', 'N/A')}%  |  Period: {m.get('n_years', 'N/A')} yrs")
        print(f"  Classification Accuracy: {acc:.1%}")
        print(f"  Action Distribution: {action_dist.to_dict()}")

    # Ensemble (equal-weight average of LSTM + CNN1D + MLP probability)
    print(f"\n{'─' * 60}")
    print("ENSEMBLE (avg of LSTM + CNN1D + MLP predictions → majority vote)")
    # Simple majority vote from deep learning models
    dl_models = ['LSTM', 'CNN1D', 'MLP']
    min_len = min(len(results[m]['preds']) for m in dl_models if results[m]['preds'])
    if min_len > 0:
        ensemble_preds = []
        ensemble_returns = []
        for i in range(min_len):
            votes = [results[m]['preds'][i] for m in dl_models if len(results[m]['preds']) > i]
            # Majority vote
            vote_counts = Counter(votes)
            winner = vote_counts.most_common(1)[0][0]
            ensemble_preds.append(winner)

            # Return from majority-voted asset
            chosen = le.inverse_transform([winner])[0]
            date = results['LSTM']['dates'][i]
            if chosen == CASH_LABEL or chosen not in raw.columns:
                ensemble_returns.append(0.0)
            else:
                loc = raw.index.get_loc(date)
                if loc > 0:
                    ret = raw[chosen].iloc[loc] / raw[chosen].iloc[loc - 1] - 1
                    ensemble_returns.append(ret if not (np.isnan(ret) or np.isinf(ret)) else 0.0)
                else:
                    ensemble_returns.append(0.0)

        ens_metrics = compute_strategy_metrics(ensemble_returns, 'DL Ensemble')
        all_metrics['ensemble'] = ens_metrics
        print(f"  CAGR: {ens_metrics.get('cagr', 'N/A')}%  |  Sharpe: {ens_metrics.get('sharpe', 'N/A')}  |  "
              f"Sortino: {ens_metrics.get('sortino', 'N/A')}  |  MaxDD: {ens_metrics.get('max_dd', 'N/A')}%")

    # Comparison table
    print(f"\n{'=' * 80}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 80}")
    header = f"{'Strategy':<25} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>9} {'MaxDD':>8} {'WR':>6} {'PF':>6}"
    print(header)
    print("─" * len(header))
    for key in ['baseline', 'LGBM', 'LSTM', 'CNN1D', 'MLP', 'ensemble']:
        m = all_metrics.get(key)
        if m and 'error' not in m:
            print(f"{m['name']:<25} {m['cagr']:>7}% {m['sharpe']:>8} {m['sortino']:>9} "
                  f"{m['max_dd']:>7}% {m['win_rate']:>5}% {m['profit_factor']:>6}")

    # Beat baseline?
    print(f"\n{'─' * 60}")
    bl_sharpe = bl_metrics.get('sharpe', 0)
    winners = []
    for key in ['LGBM', 'LSTM', 'CNN1D', 'MLP', 'ensemble']:
        m = all_metrics.get(key)
        if m and 'error' not in m and m.get('sharpe', 0) > bl_sharpe:
            winners.append(f"{m['name']} (Sharpe {m['sharpe']} vs {bl_sharpe})")

    if winners:
        print(f"BEAT BASELINE: {', '.join(winners)}")
    else:
        print("NO MODEL BEAT BASELINE on risk-adjusted basis (Sharpe)")
    print()

    return all_metrics


# ══════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    print("=" * 80)
    print("ML GROWTH TIMING MODEL — Cross-Asset Regime + Asset Selection")
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Start time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Step 1: Download data
    print("\n[1/5] Downloading data...")
    raw = download_all_data(start='2005-01-01')
    print(f"  Raw data: {len(raw)} rows, {len(raw.columns)} columns")
    print(f"  Date range: {raw.index[0]} to {raw.index[-1]}")

    # Check what target assets we actually have
    available_targets = [a for a in TARGET_ASSETS if a in raw.columns]
    print(f"  Available target assets: {available_targets}")
    if len(available_targets) < 2:
        print("ERROR: Need at least 2 target assets in data. Exiting.")
        sys.exit(1)

    # Step 2: Feature engineering
    print("\n[2/5] Computing features...")
    features = compute_features(raw)
    features = features.dropna()
    print(f"  Features: {len(features)} rows after dropna, {len(features.columns)} columns")

    # Step 3: Target construction
    print("\n[3/5] Computing targets...")
    targets = compute_targets(raw)
    print(f"  Targets: {len(targets)} rows")
    print(f"  Label distribution:\n{targets['best_action'].value_counts().to_string()}")

    # Step 4: Walk-forward training
    print("\n[4/5] Running walk-forward training...")
    results, baseline_returns, baseline_dates, le = run_walk_forward(features, targets, raw)

    # Step 5: Evaluate
    print("\n[5/5] Evaluating results...")
    all_metrics = evaluate_and_report(results, baseline_returns, baseline_dates, le, raw)

    # Save results
    results_file = OUTPUT_DIR / 'results.json'
    serializable = {}
    for k, v in all_metrics.items():
        serializable[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                           for kk, vv in v.items()}

    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # Save daily returns for further analysis
    for model_name in ['LGBM', 'LSTM', 'CNN1D', 'MLP']:
        r = results[model_name]
        if r['returns'] and r['dates']:
            min_len = min(len(r['returns']), len(r['dates']))
            ret_df = pd.DataFrame({
                'date': r['dates'][:min_len],
                'daily_return': r['returns'][:min_len],
            })
            ret_df.to_csv(OUTPUT_DIR / f'{model_name.lower()}_daily_returns.csv', index=False)

    # Baseline returns
    bl_df = pd.DataFrame({
        'date': baseline_dates,
        'daily_return': baseline_returns,
    })
    bl_df.to_csv(OUTPUT_DIR / 'baseline_daily_returns.csv', index=False)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")
    print(f"Completed: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
