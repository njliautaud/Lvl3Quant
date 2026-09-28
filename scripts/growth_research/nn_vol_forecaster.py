#!/usr/bin/env python3
"""
Neural Network Volatility Forecaster
=====================================
Predicts next-5-day realized vol for SPY/UPRO using LSTM/GRU on GPU.
Purpose: improve the vol-adjusted leverage timing system.

Currently we use trailing 21d realized vol with a 20%/30% threshold.
A forward-looking vol prediction could give 1-3 day lead time,
avoiding whipsaws and catching regime transitions earlier.

Features:
- Multi-horizon realized vol (5d, 10d, 21d, 63d)
- VIX term structure (VIX/VIX3M ratio)
- Volume profile changes
- Cross-asset vol (GLD, TLT, HYG)
- Intraday range (high-low)
- Returns distribution features (skew, kurtosis)

Walk-forward: 504d train, 21d OOT, sliding.
Target: next 5-day realized vol (annualized)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from datetime import datetime
import os, sys, json, warnings
warnings.filterwarnings('ignore')

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type != 'cuda':
    print("WARNING: No GPU available, will be slow")

OUTPUT = os.path.expanduser('~/Lvl3Quant/output/nn_vol_forecaster')
os.makedirs(OUTPUT, exist_ok=True)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# DATA
# ============================================================
print("\n" + "=" * 70)
print("NEURAL NETWORK VOLATILITY FORECASTER")
print("Predicts 5-day forward realized vol for leverage timing")
print("=" * 70)

tickers = ['SPY', 'QQQ', 'GLD', 'TLT', 'HYG', 'IEF', 'EEM', '^VIX']
print(f"\nDownloading {len(tickers)} assets...")
df = yf.download(tickers, start='2006-01-01', progress=False)
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# Extract OHLCV for SPY
spy_close = df['Close']['SPY'] if isinstance(df.columns, pd.MultiIndex) else df['Close']
spy_high = df['High']['SPY'] if isinstance(df.columns, pd.MultiIndex) else df['High']
spy_low = df['Low']['SPY'] if isinstance(df.columns, pd.MultiIndex) else df['Low']
spy_volume = df['Volume']['SPY'] if isinstance(df.columns, pd.MultiIndex) else df['Volume']

# All closes for cross-asset features
closes = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df[['SPY']]
closes = closes.ffill().dropna(how='all')

# VIX
vix = closes['^VIX'] if '^VIX' in closes.columns else None

# Align everything
idx = closes.dropna(subset=['SPY']).index
spy_close = spy_close.reindex(idx)
spy_high = spy_high.reindex(idx)
spy_low = spy_low.reindex(idx)
spy_volume = spy_volume.reindex(idx)

print(f"  {len(idx)} days, {idx[0].date()} to {idx[-1].date()}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\nBuilding features...")

features = pd.DataFrame(index=idx)

# SPY returns
ret = spy_close.pct_change()

# Realized vol at multiple horizons
for h in [5, 10, 21, 42, 63]:
    features[f'rvol_{h}d'] = ret.rolling(h).std() * np.sqrt(252)

# Garman-Klass vol (uses OHLC)
log_hl = np.log(spy_high / spy_low)
log_co = np.log(spy_close / spy_close.shift(1))
gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
for h in [5, 10, 21]:
    features[f'gk_vol_{h}d'] = np.sqrt(gk_var.rolling(h).mean() * 252)

# Parkinson vol
park_var = log_hl**2 / (4 * np.log(2))
for h in [5, 10, 21]:
    features[f'park_vol_{h}d'] = np.sqrt(park_var.rolling(h).mean() * 252)

# VIX features
if vix is not None:
    features['vix'] = vix
    features['vix_5d_change'] = vix.pct_change(5)
    features['vix_10d_change'] = vix.pct_change(10)
    # VIX vs realized (vol risk premium)
    features['vrp_5d'] = vix / 100 - features['rvol_5d']
    features['vrp_21d'] = vix / 100 - features['rvol_21d']

# Volume features
vol_ma20 = spy_volume.rolling(20).mean()
features['vol_ratio'] = spy_volume / vol_ma20
features['vol_ratio_5d'] = spy_volume.rolling(5).mean() / vol_ma20

# Return distribution features (rolling)
for h in [21, 63]:
    features[f'ret_skew_{h}d'] = ret.rolling(h).skew()
    features[f'ret_kurt_{h}d'] = ret.rolling(h).kurt()

# Cross-asset vols
for ticker in ['GLD', 'TLT', 'HYG', 'EEM']:
    if ticker in closes.columns:
        t_ret = closes[ticker].pct_change()
        features[f'{ticker}_rvol_21d'] = t_ret.rolling(21).std() * np.sqrt(252)
        # Correlation with SPY
        features[f'{ticker}_corr_21d'] = ret.rolling(21).corr(t_ret)

# Range features
features['range_pct'] = (spy_high - spy_low) / spy_close
features['range_5d_avg'] = features['range_pct'].rolling(5).mean()
features['range_ratio'] = features['range_pct'] / features['range_pct'].rolling(21).mean()

# Momentum features (vol tends to cluster)
features['ret_1d'] = ret
features['ret_5d'] = spy_close.pct_change(5)
features['abs_ret_5d'] = ret.abs().rolling(5).mean()
features['max_dd_21d'] = spy_close.rolling(21).apply(
    lambda x: (x / x.cummax() - 1).min(), raw=False
)

# Vol of vol
features['vol_of_vol_21d'] = features['rvol_5d'].rolling(21).std()

# ============================================================
# TARGET: next 5-day realized vol
# ============================================================
target = ret.shift(-5).rolling(5).std() * np.sqrt(252)
# Shift so target[t] = vol from t+1 to t+5
target = target.shift(-4)  # Align correctly
target.name = 'target_vol_5d'

# Drop NaN
valid_mask = features.notna().all(axis=1) & target.notna()
features = features[valid_mask]
target = target[valid_mask]
print(f"  {len(features)} valid samples, {features.shape[1]} features")
print(f"  Target (5d fwd vol): mean={target.mean():.4f}, std={target.std():.4f}")

# ============================================================
# DATASET & MODEL
# ============================================================
LOOKBACK = 20  # Use 20 days of feature history as input sequence

class VolDataset(Dataset):
    def __init__(self, features_np, targets_np, lookback=LOOKBACK):
        self.features = features_np
        self.targets = targets_np
        self.lookback = lookback

    def __len__(self):
        return len(self.features) - self.lookback

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.lookback]  # (lookback, n_features)
        y = self.targets[idx+self.lookback-1]  # scalar
        return torch.FloatTensor(x), torch.FloatTensor([y])


class VolLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]  # Take last hidden state
        return self.head(last)


class VolGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers,
                         batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last = gru_out[:, -1, :]
        return self.head(last)


# ============================================================
# WALK-FORWARD TRAINING
# ============================================================
TRAIN_DAYS = 504  # ~2 years
OOT_DAYS = 21     # 1 month OOT
MIN_EPOCHS = 20
MAX_EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 7

print(f"\n{'=' * 70}")
print(f"WALK-FORWARD TRAINING")
print(f"Train: {TRAIN_DAYS}d, OOT: {OOT_DAYS}d, Lookback: {LOOKBACK}d")
print(f"Device: {device}")
print(f"{'=' * 70}")

# Normalize features per-fold
feat_names = features.columns.tolist()
feat_np = features.values.astype(np.float32)
tgt_np = target.values.astype(np.float32)
dates = features.index

# Walk-forward loop
n_folds = (len(feat_np) - TRAIN_DAYS - LOOKBACK) // OOT_DAYS
print(f"  Total folds: {n_folds}")

all_preds = []
all_actuals = []
all_dates = []
fold_metrics = []

for fold in range(n_folds):
    train_start = fold * OOT_DAYS
    train_end = train_start + TRAIN_DAYS
    oot_end = min(train_end + OOT_DAYS, len(feat_np) - LOOKBACK)

    if oot_end <= train_end:
        break

    # Normalize using train stats only
    train_feat = feat_np[train_start:train_end]
    mu = train_feat.mean(axis=0)
    sigma = train_feat.std(axis=0) + 1e-8

    train_feat_norm = (feat_np[train_start:train_end] - mu) / sigma
    oot_feat_norm = (feat_np[train_end:oot_end] - mu) / sigma

    train_tgt = tgt_np[train_start:train_end]
    oot_tgt = tgt_np[train_end:oot_end]

    # Target normalization
    tgt_mu = train_tgt.mean()
    tgt_sigma = train_tgt.std() + 1e-8
    train_tgt_norm = (train_tgt - tgt_mu) / tgt_sigma

    # Datasets
    train_ds = VolDataset(train_feat_norm, train_tgt_norm, LOOKBACK)

    if len(train_ds) < BATCH_SIZE:
        continue

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True)

    # Model (alternate LSTM/GRU)
    input_dim = feat_np.shape[1]
    if fold % 2 == 0:
        model = VolLSTM(input_dim).to(device)
    else:
        model = VolGRU(input_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)
    criterion = nn.HuberLoss(delta=1.0)

    # Training
    best_loss = float('inf')
    patience_count = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_dl)

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if patience_count >= PATIENCE and epoch >= MIN_EPOCHS:
            break

    # Load best and predict OOT
    model.load_state_dict(best_state)
    model.eval()

    # OOT predictions
    oot_ds = VolDataset(oot_feat_norm, (oot_tgt - tgt_mu) / tgt_sigma, LOOKBACK)
    if len(oot_ds) == 0:
        continue

    preds_fold = []
    actuals_fold = []

    with torch.no_grad():
        for i in range(len(oot_ds)):
            x, y = oot_ds[i]
            x = x.unsqueeze(0).to(device)
            pred_norm = model(x).cpu().item()
            # Denormalize
            pred_vol = pred_norm * tgt_sigma + tgt_mu
            actual_vol = y.item() * tgt_sigma + tgt_mu
            preds_fold.append(pred_vol)
            actuals_fold.append(actual_vol)

    preds_fold = np.array(preds_fold)
    actuals_fold = np.array(actuals_fold)

    # Fold metrics
    corr = np.corrcoef(preds_fold, actuals_fold)[0, 1] if len(preds_fold) > 2 else 0
    mae = np.mean(np.abs(preds_fold - actuals_fold))

    oot_dates = dates[train_end + LOOKBACK - 1: train_end + LOOKBACK - 1 + len(preds_fold)]

    all_preds.extend(preds_fold)
    all_actuals.extend(actuals_fold)
    all_dates.extend(oot_dates)

    fold_metrics.append({
        'fold': fold,
        'corr': corr,
        'mae': mae,
        'n_samples': len(preds_fold),
        'epochs': epoch + 1,
        'model': 'LSTM' if fold % 2 == 0 else 'GRU'
    })

    if fold % 20 == 0:
        print(f"  Fold {fold:3d}/{n_folds}: corr={corr:.3f}, MAE={mae:.4f}, "
              f"epochs={epoch+1}, model={'LSTM' if fold%2==0 else 'GRU'}")

# ============================================================
# RESULTS
# ============================================================
print(f"\n{'=' * 70}")
print("RESULTS")
print(f"{'=' * 70}")

all_preds = np.array(all_preds)
all_actuals = np.array(all_actuals)

# Overall metrics
concat_corr = np.corrcoef(all_preds, all_actuals)[0, 1]
concat_mae = np.mean(np.abs(all_preds - all_actuals))
concat_rmse = np.sqrt(np.mean((all_preds - all_actuals) ** 2))

# Rank correlation (more robust)
from scipy.stats import spearmanr
rank_corr, rank_p = spearmanr(all_preds, all_actuals)

print(f"\nConcat Pearson correlation: {concat_corr:.4f}")
print(f"Concat Spearman correlation: {rank_corr:.4f} (p={rank_p:.2e})")
print(f"MAE: {concat_mae:.4f}")
print(f"RMSE: {concat_rmse:.4f}")
print(f"Total OOT samples: {len(all_preds)}")
print(f"Total folds: {len(fold_metrics)}")

# Per-fold stats
corrs = [f['corr'] for f in fold_metrics]
print(f"\nPer-fold correlation: mean={np.mean(corrs):.4f}, "
      f"median={np.median(corrs):.4f}, std={np.std(corrs):.4f}")
print(f"  Folds with corr > 0.5: {sum(1 for c in corrs if c > 0.5)}/{len(corrs)}")
print(f"  Folds with corr > 0.3: {sum(1 for c in corrs if c > 0.3)}/{len(corrs)}")
print(f"  Folds with corr < 0: {sum(1 for c in corrs if c < 0)}/{len(corrs)}")

# ============================================================
# TRADING SIMULATION: Does better vol prediction improve leverage timing?
# ============================================================
print(f"\n{'=' * 70}")
print("LEVERAGE TIMING BACKTEST")
print("Compare: 21d trailing vol vs NN predicted vol for UPRO/SPY switching")
print(f"{'=' * 70}")

# Get UPRO and SPY returns for the OOT period
try:
    upro = yf.download('UPRO', start='2006-01-01', progress=False)['Close']
    spy_bt = yf.download('SPY', start='2006-01-01', progress=False)['Close']
    gld_bt = yf.download('GLD', start='2006-01-01', progress=False)['Close']
    tlt_bt = yf.download('TLT', start='2006-01-01', progress=False)['Close']
except:
    upro = spy_bt = gld_bt = tlt_bt = None

if upro is not None and len(all_dates) > 0:
    # Build prediction series
    pred_series = pd.Series(all_preds, index=all_dates)
    actual_series = pd.Series(all_actuals, index=all_dates)

    # Get asset returns
    upro_ret = upro.pct_change()
    spy_ret = spy_bt.pct_change()
    safe_ret = 0.5 * gld_bt.pct_change() + 0.5 * tlt_bt.pct_change()

    # Align to prediction dates
    common_dates = pred_series.index.intersection(upro_ret.index)
    common_dates = common_dates.intersection(spy_ret.index)
    common_dates = common_dates.intersection(safe_ret.index)

    pred_aligned = pred_series.reindex(common_dates)
    upro_aligned = upro_ret.reindex(common_dates)
    spy_aligned = spy_ret.reindex(common_dates)
    safe_aligned = safe_ret.reindex(common_dates)

    # Strategy 1: Trailing 21d vol (baseline)
    trailing_vol = spy_bt.pct_change().rolling(21).std() * np.sqrt(252)
    trailing_aligned = trailing_vol.reindex(common_dates).ffill()

    # Vol thresholds
    LOW_THRESH = 0.20
    HIGH_THRESH = 0.30

    def run_vol_strategy(vol_signal, upro_r, spy_r, safe_r, low=LOW_THRESH, high=HIGH_THRESH):
        """UPRO when vol < low, SPY when low-high, safe when > high."""
        port_ret = pd.Series(0.0, index=vol_signal.index)
        regime = pd.Series('', index=vol_signal.index)
        for i in range(len(vol_signal)):
            v = vol_signal.iloc[i]
            if hasattr(v, '__len__'):
                v = float(v.iloc[0]) if len(v) > 0 else np.nan
            if pd.isna(v):
                port_ret.iloc[i] = spy_r.iloc[i]
                regime.iloc[i] = 'spy'
            elif v < low:
                port_ret.iloc[i] = upro_r.iloc[i]
                regime.iloc[i] = 'upro'
            elif v < high:
                port_ret.iloc[i] = spy_r.iloc[i]
                regime.iloc[i] = 'spy'
            else:
                port_ret.iloc[i] = safe_r.iloc[i]
                regime.iloc[i] = 'safe'
        return port_ret, regime

    # Run both strategies
    baseline_ret, baseline_regime = run_vol_strategy(
        trailing_aligned, upro_aligned, spy_aligned, safe_aligned)
    nn_ret, nn_regime = run_vol_strategy(
        pred_aligned, upro_aligned, spy_aligned, safe_aligned)

    # Also run buy-and-hold UPRO
    bh_ret = upro_aligned

    def calc_metrics(returns, name):
        cum = (1 + returns).cumprod()
        total_ret = cum.iloc[-1] - 1
        years = len(returns) / 252
        cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        vol = returns.std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else 0

        # Max drawdown
        rolling_max = cum.cummax()
        dd = cum / rolling_max - 1
        max_dd = dd.min()

        # Sortino
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = cagr / downside if downside > 0 else 0

        return {
            'name': name,
            'cagr': cagr,
            'vol': vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_dd': max_dd,
            'total_ret': total_ret
        }

    m_baseline = calc_metrics(baseline_ret, 'Trailing 21d Vol')
    m_nn = calc_metrics(nn_ret, 'NN Vol Forecast')
    m_bh = calc_metrics(bh_ret, 'Buy & Hold UPRO')

    print(f"\nBacktest period: {common_dates[0].date()} to {common_dates[-1].date()} "
          f"({len(common_dates)} days)")
    print(f"\n{'Strategy':<20} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Vol':>8}")
    print("-" * 60)
    for m in [m_baseline, m_nn, m_bh]:
        print(f"{m['name']:<20} {m['cagr']:>7.1%} {m['sharpe']:>8.3f} "
              f"{m['sortino']:>8.3f} {m['max_dd']:>7.1%} {m['vol']:>7.1%}")

    # Regime allocation comparison
    print(f"\nRegime allocation:")
    print(f"  Baseline — UPRO: {(baseline_regime=='upro').mean():.1%}, "
          f"SPY: {(baseline_regime=='spy').mean():.1%}, "
          f"Safe: {(baseline_regime=='safe').mean():.1%}")
    print(f"  NN Pred  — UPRO: {(nn_regime=='upro').mean():.1%}, "
          f"SPY: {(nn_regime=='spy').mean():.1%}, "
          f"Safe: {(nn_regime=='safe').mean():.1%}")

    # Key question: does NN predict vol SPIKES better?
    # Look at top-decile actual vol days
    high_vol_thresh = actual_series.quantile(0.9)
    high_vol_dates = actual_series[actual_series > high_vol_thresh].index
    high_vol_dates = high_vol_dates.intersection(common_dates)

    if len(high_vol_dates) > 10:
        # Did NN predict these correctly?
        nn_pred_high = pred_aligned.reindex(high_vol_dates)
        trailing_pred_high = trailing_aligned.reindex(high_vol_dates)

        nn_caught = (nn_pred_high > LOW_THRESH).mean()
        trailing_caught = (trailing_pred_high > LOW_THRESH).mean()

        print(f"\nVol spike detection (top 10% actual vol days, n={len(high_vol_dates)}):")
        print(f"  NN correctly flagged high vol: {nn_caught:.1%}")
        print(f"  Trailing correctly flagged: {trailing_caught:.1%}")
        print(f"  NN lead advantage: {nn_caught - trailing_caught:+.1%}")

    # Save results
    results = {
        'concat_pearson': float(concat_corr),
        'concat_spearman': float(rank_corr),
        'mae': float(concat_mae),
        'rmse': float(concat_rmse),
        'n_samples': int(len(all_preds)),
        'n_folds': len(fold_metrics),
        'per_fold_corr_mean': float(np.mean(corrs)),
        'per_fold_corr_median': float(np.median(corrs)),
        'baseline_sharpe': float(m_baseline['sharpe']),
        'nn_sharpe': float(m_nn['sharpe']),
        'sharpe_improvement': float(m_nn['sharpe'] - m_baseline['sharpe']),
        'baseline_maxdd': float(m_baseline['max_dd']),
        'nn_maxdd': float(m_nn['max_dd']),
    }
else:
    results = {
        'concat_pearson': float(concat_corr),
        'concat_spearman': float(rank_corr),
        'mae': float(concat_mae),
        'n_samples': int(len(all_preds)),
        'n_folds': len(fold_metrics),
    }

# Save
with open(f'{OUTPUT}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

np.savez(f'{OUTPUT}/predictions.npz',
         preds=all_preds, actuals=all_actuals,
         dates=np.array([str(d) for d in all_dates]))

print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")
if concat_corr > 0.5:
    print("STRONG signal — NN vol prediction beats trailing")
    print("   Worth deploying as vol timing enhancement")
elif concat_corr > 0.3:
    print("MODERATE signal — NN vol prediction has some edge")
    print("   May help at extremes, check spike detection rate")
else:
    print("WEAK signal — NN vol prediction doesn't add enough")
    print("   Trailing vol is sufficient for leverage timing")

print("\nDONE")
