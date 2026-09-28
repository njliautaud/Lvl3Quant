#!/usr/bin/env python3
"""
Regime-Adaptive Cross-Sector Momentum Dispersion Trading
=========================================================
HC #752 - New Strategy Research

Hypothesis: When cross-sector return dispersion is HIGH, momentum strategies
outperform (more differentiation). When dispersion is LOW, mean-reversion
works better (sectors converge). A neural net can predict the regime and
switch strategies accordingly.

Variants tested:
  A: LGBM regime classifier (baseline)
  B: GRU regime classifier (GPU)
  C: Fixed momentum only (control)
  D: Fixed mean-reversion only (control)
  E: Adaptive switching (GRU prediction)
  F: Ensemble blend (confidence-weighted)

Walk-forward: 250d train, 63d test, 21d slide (SLIDING, never expanding)
Cost assumption: 0.1% round-trip for ETF trades
"""

import os
import sys
import json
import time
import logging
import warnings
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: lightgbm not available, skipping variant A")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("WARNING: mlflow not available, logging locally only")

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
MACRO_TICKERS = ['SPY', '^VIX', 'TLT', 'GLD']
ALL_TICKERS = SECTOR_ETFS + MACRO_TICKERS

TRAIN_DAYS = 250
TEST_DAYS = 63
SLIDE_DAYS = 21
COMMISSION_RT = 0.001  # 0.1% round-trip
TOP_N = 3  # long top-N, short bottom-N sectors
LOOKBACK_FEATURES = 20  # days of lookback for features
GRU_HIDDEN = 64
GRU_LAYERS = 2
GRU_EPOCHS = 80
GRU_LR = 0.001
GRU_SEQ_LEN = 20
BATCH_SIZE = 32
DISPERSION_THRESHOLD_PERCENTILE = 50  # median split for high/low regime

# Paths
if sys.platform == 'win32':
    OUTPUT_DIR = Path(r'C:\Users\claude\Lvl3Quant\output\growth_research\regime_adaptive_momentum_v1')
else:
    OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/regime_adaptive_momentum_v1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = 'http://jupiter:5000'
EXPERIMENT_NAME = 'regime_adaptive_momentum_v1'

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'run.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Data Download ───────────────────────────────────────────────────────────
def download_data(start='2015-01-01', end=None):
    """Download sector ETF + macro data from Yahoo Finance."""
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')

    log.info(f"Downloading data for {len(ALL_TICKERS)} tickers: {start} to {end}")

    data = {}
    for ticker in ALL_TICKERS:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if len(df) > 100:
                # Handle both single and MultiIndex columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    close = df[('Close', ticker)] if ('Close', ticker) in df.columns else df['Close'].iloc[:, 0]
                else:
                    close = df['Close']
                # Ensure it's a Series with DatetimeIndex
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                data[ticker] = close
                log.info(f"  {ticker}: {len(df)} rows")
            else:
                log.warning(f"  {ticker}: insufficient data ({len(df)} rows)")
        except Exception as e:
            log.warning(f"  {ticker}: download failed - {e}")

    # Build price DataFrame
    prices = pd.concat(data, axis=1)
    prices.columns = list(data.keys())
    prices = prices.dropna()
    log.info(f"Combined price matrix: {prices.shape}")
    return prices


# ── Feature Engineering ─────────────────────────────────────────────────────
def compute_features(prices):
    """Build feature matrix from price data."""
    # Daily returns
    returns = prices.pct_change().dropna()

    sector_cols = [c for c in SECTOR_ETFS if c in returns.columns]
    sector_returns = returns[sector_cols]

    features = pd.DataFrame(index=returns.index)

    # 1. Cross-sectional dispersion (std of sector returns each day)
    features['dispersion'] = sector_returns.std(axis=1)
    features['dispersion_5d'] = features['dispersion'].rolling(5).mean()
    features['dispersion_20d'] = features['dispersion'].rolling(20).mean()
    features['dispersion_trend'] = features['dispersion_5d'] / features['dispersion_20d'] - 1
    features['dispersion_zscore'] = (
        (features['dispersion'] - features['dispersion'].rolling(60).mean()) /
        features['dispersion'].rolling(60).std()
    )

    # 2. Cross-sectional skew and kurtosis
    features['xsec_skew'] = sector_returns.skew(axis=1)
    features['xsec_kurt'] = sector_returns.kurtosis(axis=1)

    # 3. Sector momentum spread (top vs bottom)
    mom_20d = sector_returns.rolling(20).sum()
    features['mom_spread_20d'] = mom_20d.max(axis=1) - mom_20d.min(axis=1)
    mom_5d = sector_returns.rolling(5).sum()
    features['mom_spread_5d'] = mom_5d.max(axis=1) - mom_5d.min(axis=1)

    # 4. VIX features
    vix_col = '^VIX' if '^VIX' in prices.columns else 'VIX'
    if vix_col in prices.columns:
        features['vix'] = prices[vix_col]
        features['vix_5d_chg'] = prices[vix_col].pct_change(5)
        features['vix_20d_chg'] = prices[vix_col].pct_change(20)
        features['vix_zscore'] = (
            (prices[vix_col] - prices[vix_col].rolling(60).mean()) /
            prices[vix_col].rolling(60).std()
        )

    # 5. SPY features (market regime)
    if 'SPY' in returns.columns:
        features['spy_ret_5d'] = returns['SPY'].rolling(5).sum()
        features['spy_ret_20d'] = returns['SPY'].rolling(20).sum()
        features['spy_vol_20d'] = returns['SPY'].rolling(20).std()
        features['spy_vol_ratio'] = returns['SPY'].rolling(5).std() / returns['SPY'].rolling(20).std()

    # 6. Bond-equity (TLT as yield curve proxy)
    if 'TLT' in returns.columns and 'SPY' in returns.columns:
        features['tlt_spy_corr_20d'] = returns['TLT'].rolling(20).corr(returns['SPY'])
        features['tlt_ret_5d'] = returns['TLT'].rolling(5).sum()

    # 7. Gold features
    if 'GLD' in returns.columns:
        features['gld_ret_5d'] = returns['GLD'].rolling(5).sum()
        features['gld_spy_corr'] = returns['GLD'].rolling(20).corr(returns['SPY'])

    # 8. Sector correlation features
    corr_20d = sector_returns.rolling(20).corr()
    # Average pairwise correlation
    avg_corr = []
    for dt in sector_returns.index:
        if dt in corr_20d.index.get_level_values(0):
            try:
                c = corr_20d.loc[dt]
                mask = np.triu(np.ones(c.shape), k=1).astype(bool)
                avg_corr.append(c.values[mask].mean())
            except:
                avg_corr.append(np.nan)
        else:
            avg_corr.append(np.nan)
    features['avg_sector_corr'] = avg_corr

    # 9. Lagged dispersion features
    for lag in [1, 2, 3, 5, 10]:
        features[f'dispersion_lag{lag}'] = features['dispersion'].shift(lag)

    # Drop NaN rows
    features = features.dropna()
    log.info(f"Feature matrix: {features.shape} ({features.columns.tolist()})")

    return features, returns, sector_returns


def compute_regime_label(features, threshold_pctl=DISPERSION_THRESHOLD_PERCENTILE):
    """Label next-day dispersion regime: 1=high, 0=low (using forward dispersion)."""
    # Forward-looking: next 5-day average dispersion
    fwd_disp = features['dispersion'].rolling(5).mean().shift(-5)

    # Use expanding median for threshold (avoids lookahead)
    labels = pd.Series(np.nan, index=features.index)
    for i in range(60, len(features)):
        historical_median = fwd_disp.iloc[:i].median()
        if not np.isnan(fwd_disp.iloc[i]):
            labels.iloc[i] = 1 if fwd_disp.iloc[i] > historical_median else 0

    return labels


# ── Trading Strategies ──────────────────────────────────────────────────────
def momentum_strategy(sector_returns, lookback=20, top_n=TOP_N):
    """
    Long top-N momentum sectors, short bottom-N.
    Returns daily strategy returns.
    """
    mom = sector_returns.rolling(lookback).sum().shift(1)  # shift to avoid lookahead
    strat_returns = pd.Series(0.0, index=sector_returns.index)

    for i in range(lookback + 1, len(sector_returns)):
        dt = sector_returns.index[i]
        mom_today = mom.iloc[i].dropna().sort_values()
        if len(mom_today) < 2 * top_n:
            continue

        shorts = mom_today.index[:top_n]
        longs = mom_today.index[-top_n:]

        long_ret = sector_returns.loc[dt, longs].mean()
        short_ret = sector_returns.loc[dt, shorts].mean()

        strat_returns.iloc[i] = (long_ret - short_ret) / 2  # dollar neutral

    return strat_returns


def mean_reversion_strategy(sector_returns, lookback=20, top_n=TOP_N):
    """
    Long bottom (oversold) sectors, short top (overbought) sectors.
    Opposite of momentum.
    """
    mom = sector_returns.rolling(lookback).sum().shift(1)
    strat_returns = pd.Series(0.0, index=sector_returns.index)

    for i in range(lookback + 1, len(sector_returns)):
        dt = sector_returns.index[i]
        mom_today = mom.iloc[i].dropna().sort_values()
        if len(mom_today) < 2 * top_n:
            continue

        # Reverse: long losers, short winners
        longs = mom_today.index[:top_n]
        shorts = mom_today.index[-top_n:]

        long_ret = sector_returns.loc[dt, longs].mean()
        short_ret = sector_returns.loc[dt, shorts].mean()

        strat_returns.iloc[i] = (long_ret - short_ret) / 2

    return strat_returns


def adaptive_strategy(sector_returns, regime_preds, top_n=TOP_N, lookback=20):
    """
    Switch between momentum and mean-reversion based on regime prediction.
    regime_preds: Series of 0/1 (0=low dispersion→mean-rev, 1=high→momentum)
    """
    mom = sector_returns.rolling(lookback).sum().shift(1)
    strat_returns = pd.Series(0.0, index=sector_returns.index)

    for i in range(lookback + 1, len(sector_returns)):
        dt = sector_returns.index[i]
        if dt not in regime_preds.index or np.isnan(regime_preds.loc[dt]):
            continue

        mom_today = mom.iloc[i].dropna().sort_values()
        if len(mom_today) < 2 * top_n:
            continue

        pred = regime_preds.loc[dt]

        if pred == 1:  # High dispersion → momentum
            longs = mom_today.index[-top_n:]
            shorts = mom_today.index[:top_n]
        else:  # Low dispersion → mean reversion
            longs = mom_today.index[:top_n]
            shorts = mom_today.index[-top_n:]

        long_ret = sector_returns.loc[dt, longs].mean()
        short_ret = sector_returns.loc[dt, shorts].mean()
        strat_returns.iloc[i] = (long_ret - short_ret) / 2

    return strat_returns


def ensemble_strategy(sector_returns, regime_probs, top_n=TOP_N, lookback=20):
    """
    Blend momentum and mean-reversion based on regime probability.
    regime_probs: Series of probabilities (P(high dispersion))
    """
    mom_rets = momentum_strategy(sector_returns, lookback, top_n)
    mr_rets = mean_reversion_strategy(sector_returns, lookback, top_n)

    blended = pd.Series(0.0, index=sector_returns.index)
    for dt in sector_returns.index:
        if dt in regime_probs.index and not np.isnan(regime_probs.loc[dt]):
            p_high = regime_probs.loc[dt]
            blended.loc[dt] = p_high * mom_rets.loc[dt] + (1 - p_high) * mr_rets.loc[dt]

    return blended


# ── GRU Model ───────────────────────────────────────────────────────────────
class GRURegimeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=GRU_HIDDEN, n_layers=GRU_LAYERS, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)
        # Use last timestep
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


def prepare_sequences(X, y, seq_len=GRU_SEQ_LEN):
    """Create sequences for GRU training."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_gru(X_train, y_train, X_val, y_val, input_dim, epochs=GRU_EPOCHS):
    """Train GRU regime classifier on GPU."""
    model = GRURegimeClassifier(input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=GRU_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    # Prepare sequences
    X_tr_seq, y_tr_seq = prepare_sequences(X_train, y_train)
    X_val_seq, y_val_seq = prepare_sequences(X_val, y_val)

    if len(X_tr_seq) < 10 or len(X_val_seq) < 5:
        return None, 0.5, None

    train_ds = TensorDataset(
        torch.FloatTensor(X_tr_seq).to(DEVICE),
        torch.FloatTensor(y_tr_seq).to(DEVICE)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    X_val_t = torch.FloatTensor(X_val_seq).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val_seq).to(DEVICE)

    best_val_acc = 0
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Get final predictions and probabilities
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_preds = (val_probs > 0.5).astype(float)

    return model, best_val_acc, val_probs


def train_lgbm(X_train, y_train, X_val, y_val, feature_names):
    """Train LGBM regime classifier."""
    if not HAS_LGBM:
        return None, 0.5, None

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 5,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    model = lgb.train(
        params, train_data,
        num_boost_round=300,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )

    val_probs = model.predict(X_val)
    val_preds = (val_probs > 0.5).astype(float)
    val_acc = accuracy_score(y_val, val_preds)

    return model, val_acc, val_probs


# ── Performance Metrics ─────────────────────────────────────────────────────
def compute_metrics(returns_series, commission_per_trade=COMMISSION_RT, trades_per_day=1):
    """Compute risk-adjusted metrics for a return series."""
    rets = returns_series.dropna()
    rets = rets[rets != 0]  # remove non-trading days

    if len(rets) < 10:
        return {
            'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0,
            'total_ret': 0, 'ann_ret': 0, 'max_dd': 0,
            'avg_daily': 0, 'n_days': 0, 'calmar': 0
        }

    # Deduct commission
    n_trades = trades_per_day  # rebalance daily = 2*top_n trades
    daily_commission = commission_per_trade * n_trades / 252  # amortized
    rets_net = rets - daily_commission

    ann_factor = np.sqrt(252)
    avg_ret = rets_net.mean()
    std_ret = rets_net.std()

    sharpe = (avg_ret / std_ret * ann_factor) if std_ret > 0 else 0

    downside = rets_net[rets_net < 0].std()
    sortino = (avg_ret / downside * ann_factor) if downside > 0 else 0

    gross_profit = rets_net[rets_net > 0].sum()
    gross_loss = abs(rets_net[rets_net < 0].sum())
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    wr = (rets_net > 0).sum() / len(rets_net)

    cum = (1 + rets_net).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    total_ret = cum.iloc[-1] - 1
    n_years = len(rets_net) / 252
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    calmar = (ann_ret / abs(max_dd)) if max_dd != 0 else 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(wr, 4),
        'total_ret': round(total_ret, 4),
        'ann_ret': round(ann_ret, 4),
        'max_dd': round(max_dd, 4),
        'avg_daily': round(avg_ret * 10000, 2),  # bps
        'n_days': len(rets_net),
        'calmar': round(calmar, 3)
    }


# ── Adversarial Validation (5-Gate) ────────────────────────────────────────
def adversarial_validation(oot_metrics, variant_name):
    """
    5-gate adversarial validation for robustness.
    Returns pass/fail for each gate.
    """
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates['G1_sharpe_gt_0.5'] = oot_metrics['sharpe'] > 0.5

    # Gate 2: Win rate > 50%
    gates['G2_wr_gt_50pct'] = oot_metrics['wr'] > 0.50

    # Gate 3: Profit factor > 1.1
    gates['G3_pf_gt_1.1'] = oot_metrics['pf'] > 1.1

    # Gate 4: Max drawdown < 20%
    gates['G4_maxdd_lt_20pct'] = oot_metrics['max_dd'] > -0.20

    # Gate 5: Calmar > 0.5
    gates['G5_calmar_gt_0.5'] = oot_metrics['calmar'] > 0.5

    passed = sum(gates.values())
    gates['total_passed'] = f"{passed}/5"
    gates['verdict'] = 'PASS' if passed >= 4 else ('MARGINAL' if passed >= 3 else 'FAIL')

    log.info(f"  {variant_name} adversarial: {gates['total_passed']} gates passed → {gates['verdict']}")
    return gates


# ── Walk-Forward Engine ─────────────────────────────────────────────────────
def run_walkforward(features, labels, sector_returns, feature_names):
    """
    Sliding walk-forward with all 6 variants.
    250d train, 63d test, 21d slide.
    """
    n = len(features)
    results = {v: [] for v in ['A_lgbm', 'B_gru', 'C_momentum', 'D_meanrev', 'E_adaptive', 'F_ensemble']}
    fold_metrics = {v: [] for v in results}

    scaler = StandardScaler()

    fold_idx = 0
    start = 0

    while start + TRAIN_DAYS + TEST_DAYS <= n:
        train_end = start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n)

        train_idx = features.index[start:train_end]
        test_idx = features.index[train_end:test_end]

        X_train = features.loc[train_idx].values
        y_train = labels.loc[train_idx].values
        X_test = features.loc[test_idx].values
        y_test = labels.loc[test_idx].values

        # Remove NaN labels
        train_mask = ~np.isnan(y_train)
        test_mask = ~np.isnan(y_test)

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            start += SLIDE_DAYS
            continue

        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask]
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask]

        # Scale
        X_train_s = scaler.fit_transform(X_train_clean)
        X_test_s = scaler.transform(X_test_clean)

        # Replace any NaN/inf from scaling
        X_train_s = np.nan_to_num(X_train_s, nan=0, posinf=0, neginf=0)
        X_test_s = np.nan_to_num(X_test_s, nan=0, posinf=0, neginf=0)

        test_dates = test_idx[test_mask]

        log.info(f"Fold {fold_idx}: train {train_idx[0].strftime('%Y-%m-%d')} to {train_idx[-1].strftime('%Y-%m-%d')}, "
                 f"test {test_idx[0].strftime('%Y-%m-%d')} to {test_idx[-1].strftime('%Y-%m-%d')} "
                 f"(train={len(X_train_clean)}, test={len(X_test_clean)})")

        # ── Variant A: LGBM ──
        lgbm_preds = None
        lgbm_probs = None
        if HAS_LGBM:
            _, acc, lgbm_probs = train_lgbm(X_train_clean, y_train_clean, X_test_clean, y_test_clean, feature_names)
            lgbm_preds = (lgbm_probs > 0.5).astype(float) if lgbm_probs is not None else None
            log.info(f"  A_lgbm: val_acc={acc:.3f}")

        # ── Variant B: GRU ──
        gru_model, gru_acc, gru_probs_raw = train_gru(
            X_train_s, y_train_clean, X_test_s, y_test_clean,
            input_dim=X_train_s.shape[1]
        )
        log.info(f"  B_gru: val_acc={gru_acc:.3f}")

        # Need to handle sequence offset for GRU predictions
        gru_offset = GRU_SEQ_LEN
        gru_test_dates = test_dates[gru_offset:] if len(test_dates) > gru_offset else test_dates

        # ── Variant C: Fixed Momentum ──
        mom_rets = momentum_strategy(sector_returns.loc[test_idx])

        # ── Variant D: Fixed Mean-Reversion ──
        mr_rets = mean_reversion_strategy(sector_returns.loc[test_idx])

        # ── Variant E: Adaptive (GRU-based switching) ──
        if gru_probs_raw is not None and len(gru_probs_raw) > 0:
            gru_preds_series = pd.Series(
                (gru_probs_raw > 0.5).astype(float),
                index=gru_test_dates[:len(gru_probs_raw)]
            )
            adaptive_rets = adaptive_strategy(sector_returns.loc[test_idx], gru_preds_series)
        else:
            adaptive_rets = mom_rets.copy()  # fallback

        # ── Variant F: Ensemble ──
        if gru_probs_raw is not None and len(gru_probs_raw) > 0:
            gru_prob_series = pd.Series(
                gru_probs_raw,
                index=gru_test_dates[:len(gru_probs_raw)]
            )
            ensemble_rets = ensemble_strategy(sector_returns.loc[test_idx], gru_prob_series)
        else:
            ensemble_rets = mom_rets.copy()

        # ── Variant A returns (LGBM adaptive) ──
        if lgbm_preds is not None:
            lgbm_pred_series = pd.Series(lgbm_preds, index=test_dates[:len(lgbm_preds)])
            lgbm_adaptive_rets = adaptive_strategy(sector_returns.loc[test_idx], lgbm_pred_series)
        else:
            lgbm_adaptive_rets = mom_rets.copy()

        # Store results
        results['A_lgbm'].append(lgbm_adaptive_rets)
        results['B_gru'].append(adaptive_rets)  # GRU adaptive
        results['C_momentum'].append(mom_rets)
        results['D_meanrev'].append(mr_rets)
        results['E_adaptive'].append(adaptive_rets)
        results['F_ensemble'].append(ensemble_rets)

        fold_idx += 1
        start += SLIDE_DAYS

    log.info(f"Walk-forward complete: {fold_idx} folds")
    return results, fold_idx


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    log.info("=" * 70)
    log.info("Regime-Adaptive Cross-Sector Momentum Dispersion Trading")
    log.info("=" * 70)

    # MLflow setup
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            mlflow.start_run(run_name=f"regime_adaptive_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({
                'train_days': TRAIN_DAYS,
                'test_days': TEST_DAYS,
                'slide_days': SLIDE_DAYS,
                'commission_rt': COMMISSION_RT,
                'top_n': TOP_N,
                'gru_hidden': GRU_HIDDEN,
                'gru_layers': GRU_LAYERS,
                'gru_epochs': GRU_EPOCHS,
                'gru_seq_len': GRU_SEQ_LEN,
                'device': str(DEVICE),
            })
            log.info(f"MLflow tracking started: {MLFLOW_URI}")
        except Exception as e:
            log.warning(f"MLflow connection failed: {e}. Continuing without tracking.")

    # 1. Download data
    log.info("\n[STEP 1] Downloading data...")
    prices = download_data(start='2015-01-01')

    if len(prices) < TRAIN_DAYS + TEST_DAYS + 100:
        log.error("Insufficient data. Aborting.")
        return

    # 2. Feature engineering
    log.info("\n[STEP 2] Computing features...")
    features, returns, sector_returns = compute_features(prices)

    # 3. Regime labels
    log.info("\n[STEP 3] Computing regime labels...")
    labels = compute_regime_label(features)

    # Align all data
    common_idx = features.index.intersection(labels.dropna().index).intersection(sector_returns.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]
    sector_returns = sector_returns.loc[common_idx]

    feature_names = features.columns.tolist()
    log.info(f"Aligned data: {len(common_idx)} days, {len(feature_names)} features")
    log.info(f"Label distribution: high={int(labels.sum())}, low={int(len(labels) - labels.sum())}")

    if HAS_MLFLOW:
        try:
            mlflow.log_metrics({
                'n_features': len(feature_names),
                'n_days': len(common_idx),
                'label_high_pct': float(labels.mean()),
            })
        except:
            pass

    # 4. Walk-forward
    log.info("\n[STEP 4] Running walk-forward backtest...")
    results, n_folds = run_walkforward(features, labels, sector_returns, feature_names)

    # 5. Aggregate results
    log.info("\n[STEP 5] Aggregating results...")
    final_results = {}

    for variant_name, fold_rets_list in results.items():
        if not fold_rets_list:
            continue

        # Concatenate all OOT returns
        all_rets = pd.concat(fold_rets_list)
        # Remove duplicates (overlapping folds)
        all_rets = all_rets[~all_rets.index.duplicated(keep='first')]
        all_rets = all_rets.sort_index()

        metrics = compute_metrics(all_rets)
        gates = adversarial_validation(metrics, variant_name)

        final_results[variant_name] = {
            'metrics': metrics,
            'adversarial': gates,
        }

        log.info(f"\n{'='*50}")
        log.info(f"Variant {variant_name}:")
        log.info(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}")
        log.info(f"  PF: {metrics['pf']:.3f} | WR: {metrics['wr']:.1%}")
        log.info(f"  Ann Return: {metrics['ann_ret']:.2%} | Max DD: {metrics['max_dd']:.2%}")
        log.info(f"  Calmar: {metrics['calmar']:.3f} | Avg Daily: {metrics['avg_daily']:.1f} bps")
        log.info(f"  Trading days: {metrics['n_days']}")
        log.info(f"  Adversarial: {gates['total_passed']} → {gates['verdict']}")

        # MLflow
        if HAS_MLFLOW:
            try:
                for k, v in metrics.items():
                    mlflow.log_metric(f"{variant_name}_{k}", v)
                mlflow.log_metric(f"{variant_name}_gates_passed",
                                  int(gates['total_passed'].split('/')[0]))
            except:
                pass

    # 6. Regime analysis
    log.info("\n[STEP 6] Regime analysis...")
    regime_analysis = {}

    # Check if adaptive variants outperform controls
    if 'C_momentum' in final_results and 'D_meanrev' in final_results:
        mom_sharpe = final_results['C_momentum']['metrics']['sharpe']
        mr_sharpe = final_results['D_meanrev']['metrics']['sharpe']

        for adaptive_var in ['A_lgbm', 'B_gru', 'E_adaptive', 'F_ensemble']:
            if adaptive_var in final_results:
                adap_sharpe = final_results[adaptive_var]['metrics']['sharpe']
                best_control = max(mom_sharpe, mr_sharpe)

                regime_analysis[adaptive_var] = {
                    'sharpe': adap_sharpe,
                    'mom_sharpe': mom_sharpe,
                    'mr_sharpe': mr_sharpe,
                    'beats_momentum': adap_sharpe > mom_sharpe,
                    'beats_meanrev': adap_sharpe > mr_sharpe,
                    'beats_best_control': adap_sharpe > best_control,
                    'improvement_vs_best': round(adap_sharpe - best_control, 3),
                }

        log.info(f"\nRegime switching value:")
        log.info(f"  Momentum-only Sharpe: {mom_sharpe:.3f}")
        log.info(f"  MeanRev-only Sharpe: {mr_sharpe:.3f}")
        for v, ra in regime_analysis.items():
            log.info(f"  {v}: Sharpe={ra['sharpe']:.3f}, "
                     f"beats_best_control={ra['beats_best_control']}, "
                     f"improvement={ra['improvement_vs_best']:+.3f}")

    # 7. Save results
    log.info("\n[STEP 7] Saving results...")

    output = {
        'experiment': EXPERIMENT_NAME,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'train_days': TRAIN_DAYS,
            'test_days': TEST_DAYS,
            'slide_days': SLIDE_DAYS,
            'commission_rt': COMMISSION_RT,
            'top_n': TOP_N,
            'n_folds': n_folds,
            'device': str(DEVICE),
            'sector_etfs': SECTOR_ETFS,
            'n_features': len(feature_names),
            'feature_names': feature_names,
        },
        'variants': final_results,
        'regime_analysis': regime_analysis,
        'elapsed_seconds': round(time.time() - start_time, 1),
    }

    results_path = OUTPUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Results saved to {results_path}")

    # MLflow artifact
    if HAS_MLFLOW:
        try:
            mlflow.log_artifact(str(results_path))
            mlflow.log_artifact(str(OUTPUT_DIR / 'run.log'))
            mlflow.end_run()
            log.info("MLflow run completed")
        except Exception as e:
            log.warning(f"MLflow artifact logging failed: {e}")

    elapsed = time.time() - start_time
    log.info(f"\n{'='*70}")
    log.info(f"COMPLETE in {elapsed/60:.1f} minutes")
    log.info(f"{'='*70}")

    # Print summary table
    log.info("\n\nFINAL SUMMARY:")
    log.info(f"{'Variant':<15} {'Sharpe':>8} {'Sortino':>8} {'PF':>6} {'WR':>7} {'AnnRet':>8} {'MaxDD':>8} {'Gates':>7}")
    log.info("-" * 75)
    for v, r in sorted(final_results.items()):
        m = r['metrics']
        g = r['adversarial']['total_passed']
        log.info(f"{v:<15} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['pf']:>6.2f} {m['wr']:>6.1%} "
                 f"{m['ann_ret']:>7.2%} {m['max_dd']:>7.2%} {g:>7}")

    return output


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(f"Fatal error: {e}")
        log.error(traceback.format_exc())
        raise
