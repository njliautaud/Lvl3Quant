#!/usr/bin/env python3
"""
BPS MLP Spread Predictor — Dynamic Selection via Neural Network
================================================================
Extends the GA optimizer (static ticker weights) with a learned model that
predicts P(profitable) for each individual spread given current market features.

Architecture: 2-layer MLP (64->32->1), BatchNorm, Dropout
Walk-forward: 120d train, 30d OOS, sliding
Target: binary — did spread hit profit-take (1) or lose money (0)?
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import time
import json
import os
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Try MLflow, make optional
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PRICES_PATH = '/home/nick/Lvl3Quant/wheel_strategy_v1/data/cache/prices.parquet'
UNIVERSE_PATH = '/home/nick/Lvl3Quant/wheel_strategy_v1/data/cache/universe.parquet'
OUTPUT_DIR = '/home/nick/Lvl3Quant/wheel_strategy_v1/backtest/mlp_output'

# MLflow — try multiple URIs
MLFLOW_URIS = [
    'http://localhost:5000',
    'http://jupiter:5000',
    'http://jupiter:5000',
]

# BPS parameters (match GA optimizer)
DTE = 7
SPREAD_WIDTH = 5.0
TARGET_DELTA = -0.25
PROFIT_TAKE_PCT = 0.50
PREMIUM_FLOOR = 0.10
IV_MULTIPLIER = 1.35
RF_ANNUAL = 0.045

# Walk-forward
TRAIN_DAYS = 120
OOS_DAYS = 30
STEP_DAYS = 30

# MLP
HIDDEN1 = 64
HIDDEN2 = 32
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 8

PRED_THRESHOLD = 0.55

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------
def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_put_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def find_strike_for_delta(S, target_delta, T, r, sigma, n_iter=30):
    K_lo, K_hi = S * 0.70, S * 1.00
    for _ in range(n_iter):
        K_mid = (K_lo + K_hi) / 2
        d = bs_put_delta(S, K_mid, T, r, sigma)
        if d < target_delta:
            K_hi = K_mid
        else:
            K_lo = K_mid
    return (K_lo + K_hi) / 2


def simulate_bps_trade(S_entry, S_exit, sigma):
    """Returns (pnl, credit, is_profitable) for one spread."""
    T = DTE / 252.0
    r = RF_ANNUAL
    iv = sigma * IV_MULTIPLIER

    K_short = find_strike_for_delta(S_entry, TARGET_DELTA, T, r, iv)
    K_long = K_short - SPREAD_WIDTH

    short_put_price = bs_put_price(S_entry, K_short, T, r, iv)
    long_put_price = bs_put_price(S_entry, K_long, T, r, iv)
    credit = short_put_price - long_put_price

    if credit < PREMIUM_FLOOR:
        return 0.0, 0.0, None

    max_loss = SPREAD_WIDTH - credit

    short_put_intrinsic = max(K_short - S_exit, 0)
    long_put_intrinsic = max(K_long - S_exit, 0)
    spread_intrinsic = short_put_intrinsic - long_put_intrinsic

    pnl = credit - spread_intrinsic
    if pnl > credit * PROFIT_TAKE_PCT:
        pnl = credit * PROFIT_TAKE_PCT
    pnl = max(pnl, -max_loss)

    is_profitable = 1 if pnl > 0 else 0
    return pnl, credit, is_profitable


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Feature + Label Generation
# ---------------------------------------------------------------------------
def build_dataset(prices_df, universe_df):
    logger.info("Building spread dataset from price history...")

    sector_map = dict(zip(universe_df['ticker'], universe_df['sector']))
    sectors = sorted(universe_df['sector'].unique())
    sector_to_idx = {s: i for i, s in enumerate(sectors)}

    prices = prices_df.sort_values(['ticker', 'date']).copy()
    tickers = sorted(prices['ticker'].unique())

    all_records = []

    for ticker in tickers:
        tdf = prices[prices['ticker'] == ticker].copy()
        tdf = tdf.set_index('date').sort_index()

        if len(tdf) < 130:
            continue

        tdf['mom_5d'] = tdf['close'].pct_change(5)
        tdf['mom_20d'] = tdf['close'].pct_change(20)
        tdf['mom_60d'] = tdf['close'].pct_change(60)
        tdf['rsi_14'] = compute_rsi(tdf['close'], 14)
        tdf['vol_ratio'] = tdf['rv_20'] / (tdf['rv_60'] + 1e-8)
        tdf['dow'] = tdf.index.dayofweek

        trade_idx = list(range(0, len(tdf) - DTE, 5))

        for idx in trade_idx:
            if idx < 60:
                continue

            entry_date = tdf.index[idx]
            exit_idx = min(idx + DTE, len(tdf) - 1)

            S_entry = tdf.iloc[idx]['close']
            S_exit = tdf.iloc[exit_idx]['close']
            rv20 = tdf.iloc[idx].get('rv_20', np.nan)
            rv60 = tdf.iloc[idx].get('rv_60', np.nan)

            if pd.isna(rv20) or pd.isna(rv60) or rv20 <= 0:
                continue

            pnl, credit, is_profitable = simulate_bps_trade(S_entry, S_exit, rv20)
            if is_profitable is None:
                continue

            credit_norm = credit / SPREAD_WIDTH

            all_records.append({
                'date': entry_date,
                'ticker': ticker,
                'rv_20': rv20,
                'rv_60': rv60,
                'vol_ratio': tdf.iloc[idx].get('vol_ratio', np.nan),
                'mom_5d': tdf.iloc[idx].get('mom_5d', np.nan),
                'mom_20d': tdf.iloc[idx].get('mom_20d', np.nan),
                'mom_60d': tdf.iloc[idx].get('mom_60d', np.nan),
                'rsi_14': tdf.iloc[idx].get('rsi_14', np.nan),
                'credit_norm': credit_norm,
                'dow': tdf.iloc[idx].get('dow', 0),
                'sector_idx': sector_to_idx.get(sector_map.get(ticker, 'Unknown'), 0),
                'pnl': pnl,
                'target': is_profitable,
            })

    df = pd.DataFrame(all_records)
    df = df.dropna()

    # Market-level features
    mkt_mom = df.groupby('date')['mom_20d'].mean().rename('mkt_mom_20d')
    mkt_vol = df.groupby('date')['mom_5d'].std().rename('mkt_vol')
    df = df.merge(mkt_mom, on='date', how='left')
    df = df.merge(mkt_vol, on='date', how='left')

    # Sector one-hot
    for i, s in enumerate(sectors):
        df[f'sector_{s}'] = (df['sector_idx'] == i).astype(float)

    logger.info(f"Dataset built: {len(df)} samples, "
                f"{df['target'].mean():.1%} positive rate, "
                f"{df['ticker'].nunique()} tickers, "
                f"date range {df['date'].min().date()} to {df['date'].max().date()}")

    return df, sectors


# ---------------------------------------------------------------------------
# MLP Model
# ---------------------------------------------------------------------------
class SpreadMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, HIDDEN1),
            nn.BatchNorm1d(HIDDEN1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.BatchNorm1d(HIDDEN2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN2, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_fold(X_train, y_train, X_val, y_val, n_features):
    model = SpreadMLP(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    pos_rate = y_train.mean()
    if pos_rate > 0 and pos_rate < 1:
        pos_weight = torch.tensor([(1 - pos_rate) / pos_rate]).to(DEVICE)
    else:
        pos_weight = torch.tensor([1.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(DEVICE),
        torch.FloatTensor(y_train).to(DEVICE)
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).squeeze(-1)
            val_loss = criterion(val_logits, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t).squeeze(-1)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

    # Replace NaN predictions with 0.5 (uninformative)
    val_probs = np.nan_to_num(val_probs, nan=0.5)
    val_probs = np.clip(val_probs, 0.0, 1.0)

    acc = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
    try:
        auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.5
    except ValueError:
        auc = 0.5

    return model, val_probs, acc, auc, best_val_loss


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------
def walk_forward_evaluate(df, feature_cols):
    logger.info("Starting walk-forward evaluation...")

    dates = sorted(df['date'].unique())
    n_dates = len(dates)

    all_oos_results = []
    fold_metrics = []
    fold_num = 0

    start = 0
    while start + TRAIN_DAYS + OOS_DAYS <= n_dates:
        train_end = start + TRAIN_DAYS
        oos_end = train_end + OOS_DAYS

        train_dates = set(dates[start:train_end])
        oos_dates = set(dates[train_end:oos_end])

        train_mask = df['date'].isin(train_dates)
        oos_mask = df['date'].isin(oos_dates)

        train_df = df[train_mask]
        oos_df = df[oos_mask]

        if len(train_df) < 100 or len(oos_df) < 20:
            start += STEP_DAYS
            continue

        scaler = StandardScaler()
        X_train_raw = train_df[feature_cols].values.copy()
        X_oos_raw = oos_df[feature_cols].values.copy()
        # Replace inf/nan before scaling
        X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=10.0, neginf=-10.0)
        X_oos_raw = np.nan_to_num(X_oos_raw, nan=0.0, posinf=10.0, neginf=-10.0)
        X_train = scaler.fit_transform(X_train_raw)
        X_oos = scaler.transform(X_oos_raw)
        # Clip extreme scaled values
        X_train = np.clip(X_train, -5, 5)
        X_oos = np.clip(X_oos, -5, 5)
        y_train = train_df['target'].values.astype(np.float32)
        y_oos = oos_df['target'].values.astype(np.float32)

        model, _, val_acc, val_auc, val_loss = train_one_fold(
            X_train, y_train, X_oos, y_oos, len(feature_cols)
        )

        model.eval()
        with torch.no_grad():
            X_oos_t = torch.FloatTensor(X_oos).to(DEVICE)
            oos_logits = model(X_oos_t).squeeze(-1)
            oos_probs = torch.sigmoid(oos_logits).cpu().numpy()

        # Handle NaN predictions
        oos_probs = np.nan_to_num(oos_probs, nan=0.5)
        oos_probs = np.clip(oos_probs, 0.0, 1.0)
        oos_preds = (oos_probs >= 0.5).astype(int)
        oos_acc = accuracy_score(y_oos, oos_preds)
        oos_prec = precision_score(y_oos, oos_preds, zero_division=0)
        oos_recall = recall_score(y_oos, oos_preds, zero_division=0)
        try:
            oos_auc = roc_auc_score(y_oos, oos_probs) if len(np.unique(y_oos)) > 1 else 0.5
        except ValueError:
            oos_auc = 0.5

        oos_result = oos_df[['date', 'ticker', 'pnl', 'target']].copy()
        oos_result['pred_prob'] = oos_probs
        oos_result['pred'] = oos_preds
        oos_result['fold'] = fold_num
        all_oos_results.append(oos_result)

        fold_metrics.append({
            'fold': fold_num,
            'train_start': str(dates[start].date()),
            'oos_start': str(dates[train_end].date()),
            'oos_end': str(dates[min(oos_end, n_dates) - 1].date()),
            'n_train': len(train_df),
            'n_oos': len(oos_df),
            'oos_acc': oos_acc,
            'oos_prec': oos_prec,
            'oos_recall': oos_recall,
            'oos_auc': oos_auc,
            'base_rate': float(y_oos.mean()),
        })

        logger.info(f"Fold {fold_num}: OOS acc={oos_acc:.3f} prec={oos_prec:.3f} "
                     f"recall={oos_recall:.3f} AUC={oos_auc:.3f} "
                     f"base_rate={y_oos.mean():.3f} n={len(oos_df)}")

        start += STEP_DAYS
        fold_num += 1

    oos_all = pd.concat(all_oos_results, ignore_index=True)
    return oos_all, fold_metrics


# ---------------------------------------------------------------------------
# Performance Analysis
# ---------------------------------------------------------------------------
def analyze_performance(oos_df):
    results = {}

    ga_tickers = ['CL', 'F', 'GM', 'HOOD', 'JNJ', 'LLY', 'MCD', 'NFLX', 'NVDA',
                  'OXY', 'PANW', 'PFE', 'PLTR', 'SMCI', 'T', 'TGT', 'TMUS',
                  'TSLA', 'VZ', 'WMT']

    strategies = {
        'all_trades': oos_df,
        'ga_selected': oos_df[oos_df['ticker'].isin(ga_tickers)],
        'mlp_selected': oos_df[oos_df['pred_prob'] >= PRED_THRESHOLD],
        'mlp_high_conf': oos_df[oos_df['pred_prob'] >= 0.65],
        'mlp_top_quintile': oos_df[oos_df['pred_prob'] >= oos_df['pred_prob'].quantile(0.80)],
    }

    for name, sdf in strategies.items():
        if len(sdf) < 10:
            logger.warning(f"Strategy '{name}' has <10 trades, skipping")
            continue

        weekly_pnl = sdf.groupby('date')['pnl'].sum()
        n_weeks = len(weekly_pnl)
        mean_pnl = weekly_pnl.mean()
        std_pnl = weekly_pnl.std()

        rf_weekly = RF_ANNUAL / 52
        sharpe = (mean_pnl - rf_weekly) / (std_pnl + 1e-10) * np.sqrt(52)

        downside = weekly_pnl[weekly_pnl < rf_weekly] - rf_weekly
        downside_std = np.sqrt((downside**2).mean()) if len(downside) > 0 else 1e-10
        sortino = (mean_pnl - rf_weekly) / (downside_std + 1e-10) * np.sqrt(52)

        gross_profit = sdf[sdf['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(sdf[sdf['pnl'] < 0]['pnl'].sum())
        pf = gross_profit / (gross_loss + 1e-10)

        wr = (sdf['pnl'] > 0).mean()
        trades_per_week = len(sdf) / n_weeks if n_weeks > 0 else 0

        results[name] = {
            'n_trades': int(len(sdf)),
            'n_weeks': int(n_weeks),
            'trades_per_week': round(trades_per_week, 1),
            'win_rate': round(float(wr), 4),
            'sharpe': round(float(sharpe), 3),
            'sortino': round(float(sortino), 3),
            'profit_factor': round(float(pf), 3),
            'mean_weekly_pnl': round(float(mean_pnl), 4),
            'total_pnl': round(float(sdf['pnl'].sum()), 2),
        }

        logger.info(f"  {name}: Sharpe={sharpe:.3f} Sortino={sortino:.3f} "
                     f"PF={pf:.2f} WR={wr:.1%} trades={len(sdf)} "
                     f"trades/wk={trades_per_week:.1f}")

    return results


# ---------------------------------------------------------------------------
# MLflow helper
# ---------------------------------------------------------------------------
def setup_mlflow():
    """Set up MLflow with local file tracking."""
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed, logging to files only")
        return False
    try:
        local_uri = "file:///home/nick/Lvl3Quant/wheel_strategy_v1/backtest/mlruns"
        mlflow.set_tracking_uri(local_uri)
        mlflow.set_experiment("bps_mlp_predictor")
        logger.info("MLflow using local file tracking")
        return True
    except Exception as e:
        logger.warning(f"MLflow setup failed: {e}")
        return False

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(f"Device: {DEVICE}")
    logger.info(f"Loading data...")

    prices = pd.read_parquet(PRICES_PATH)
    universe = pd.read_parquet(UNIVERSE_PATH)

    df, sectors = build_dataset(prices, universe)

    meta_cols = ['date', 'ticker', 'pnl', 'target', 'sector_idx', 'date_idx']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")

    # MLflow setup (optional)
    use_mlflow = setup_mlflow()

    run_ctx = mlflow.start_run(run_name="bps_mlp_wf_v1") if use_mlflow else None

    try:
        if use_mlflow:
            run_ctx.__enter__()
            mlflow.log_params({
                'hidden1': HIDDEN1, 'hidden2': HIDDEN2, 'dropout': DROPOUT,
                'lr': LR, 'weight_decay': WEIGHT_DECAY, 'batch_size': BATCH_SIZE,
                'epochs': EPOCHS, 'patience': PATIENCE,
                'train_days': TRAIN_DAYS, 'oos_days': OOS_DAYS, 'step_days': STEP_DAYS,
                'pred_threshold': PRED_THRESHOLD, 'n_features': len(feature_cols),
                'n_samples': len(df), 'n_tickers': df['ticker'].nunique(),
                'base_rate': round(df['target'].mean(), 4),
            })

        # Walk-forward
        oos_df, fold_metrics = walk_forward_evaluate(df, feature_cols)

        # Aggregate
        mean_acc = np.mean([f['oos_acc'] for f in fold_metrics])
        mean_auc = np.mean([f['oos_auc'] for f in fold_metrics])
        mean_prec = np.mean([f['oos_prec'] for f in fold_metrics])
        mean_recall = np.mean([f['oos_recall'] for f in fold_metrics])

        logger.info(f"\n{'='*60}")
        logger.info(f"WALK-FORWARD RESULTS ({len(fold_metrics)} folds)")
        logger.info(f"Mean OOS Accuracy:  {mean_acc:.4f}")
        logger.info(f"Mean OOS AUC:       {mean_auc:.4f}")
        logger.info(f"Mean OOS Precision: {mean_prec:.4f}")
        logger.info(f"Mean OOS Recall:    {mean_recall:.4f}")
        logger.info(f"Base rate:          {oos_df['target'].mean():.4f}")

        # Performance
        logger.info(f"\n{'='*60}")
        logger.info("STRATEGY COMPARISON (OOS only)")
        perf = analyze_performance(oos_df)

        if use_mlflow:
            mlflow.log_metrics({
                'mean_oos_accuracy': mean_acc, 'mean_oos_auc': mean_auc,
                'mean_oos_precision': mean_prec, 'mean_oos_recall': mean_recall,
                'n_folds': len(fold_metrics), 'n_oos_samples': len(oos_df),
            })
            for sn, sm in perf.items():
                for mn, mv in sm.items():
                    mlflow.log_metric(f"{sn}_{mn}", mv)

        # Save outputs
        oos_df.to_parquet(os.path.join(OUTPUT_DIR, 'oos_predictions.parquet'), index=False)
        with open(os.path.join(OUTPUT_DIR, 'fold_metrics.json'), 'w') as f:
            json.dump(fold_metrics, f, indent=2)
        with open(os.path.join(OUTPUT_DIR, 'strategy_comparison.json'), 'w') as f:
            json.dump(perf, f, indent=2)

        # Save full results summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(DEVICE),
            'n_samples': len(df),
            'n_tickers': int(df['ticker'].nunique()),
            'n_folds': len(fold_metrics),
            'base_rate': round(float(df['target'].mean()), 4),
            'mean_oos_accuracy': round(float(mean_acc), 4),
            'mean_oos_auc': round(float(mean_auc), 4),
            'mean_oos_precision': round(float(mean_prec), 4),
            'mean_oos_recall': round(float(mean_recall), 4),
            'strategies': perf,
            'runtime_seconds': round(time.time() - t0, 1),
        }
        with open(os.path.join(OUTPUT_DIR, 'run_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        if use_mlflow:
            mlflow.log_artifact(os.path.join(OUTPUT_DIR, 'fold_metrics.json'))
            mlflow.log_artifact(os.path.join(OUTPUT_DIR, 'strategy_comparison.json'))
            mlflow.log_metric('runtime_seconds', time.time() - t0)

        elapsed = time.time() - t0
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETED in {elapsed:.1f}s")

        # Final table
        logger.info(f"\n{'='*60}")
        logger.info("FINAL COMPARISON TABLE")
        logger.info(f"{'Strategy':<20} {'Sharpe':>8} {'Sortino':>8} {'PF':>6} {'WR':>6} {'Trades':>8} {'$/wk':>8}")
        logger.info('-' * 70)
        for name, m in perf.items():
            logger.info(f"{name:<20} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
                        f"{m['profit_factor']:>6.2f} {m['win_rate']:>6.1%} "
                        f"{m['n_trades']:>8} {m['mean_weekly_pnl']:>8.4f}")

    finally:
        if use_mlflow and run_ctx:
            try:
                run_ctx.__exit__(None, None, None)
            except Exception:
                pass


if __name__ == '__main__':
    main()
