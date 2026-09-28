#!/usr/bin/env python3
"""
Deep Regime-Switching Portfolio with Transformer Attention v1
=============================================================
A Transformer-based model that learns cross-asset REGIME patterns from
60-day sequences and optimally allocates between 4 asset groups:
  - Growth: QQQ, MTUM, QUAL
  - Income: SCHD, JEPI, VNQ
  - Safety: TLT, GLD, SHY
  - Aggressive: TQQQ, UPRO (leveraged, risk-on only)

Novel: Uses multi-head attention (4 heads, 2 layers) on a SEQUENCE of the last
60 trading days of cross-asset features. The attention mechanism discovers
complex regime transitions that simple momentum/vol signals miss.

HC #724 compliant: ALL features use T-1 data only.
HC #718 compliant: train_end + GAP before test_start.
Walk-forward: 504d train, 21d test, 21d gap. Sliding from 2015-2026.
Transaction costs: 10 bps ETFs, 20 bps leveraged.
Regime-agnostic test mandatory.
Permutation test: 50 shuffles.
"""

import os, sys, time, math, warnings, datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from collections import defaultdict
from copy import deepcopy

warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
SEQ_LEN = 60
TRAIN_WINDOW = 504
TEST_WINDOW = 21
LABEL_HORIZON = 21
GAP_DAYS = LABEL_HORIZON
COST_BPS_ETF = 10
COST_BPS_LEV = 20
N_PERMUTATIONS = 50
EPOCHS = 80
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"[{datetime.datetime.now()}] Regime Transformer v1 starting on {DEVICE}")
print(f"Config: seq={SEQ_LEN}d, train={TRAIN_WINDOW}d, test={TEST_WINDOW}d, "
      f"gap={GAP_DAYS}d, label_horizon={LABEL_HORIZON}d")
print(f"Transformer: d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, "
      f"dropout={DROPOUT}")
print(f"Costs: ETF={COST_BPS_ETF}bps, Leveraged={COST_BPS_LEV}bps")
print("=" * 80)

# ─── Asset Groups ─────────────────────────────────────────────────────────────
ASSET_GROUPS = {
    'growth':     ['QQQ', 'MTUM', 'QUAL'],
    'income':     ['SCHD', 'JEPI', 'VNQ'],
    'safety':     ['TLT', 'GLD', 'SHY'],
    'aggressive': ['TQQQ', 'UPRO'],
}
GROUP_NAMES = list(ASSET_GROUPS.keys())
N_GROUPS = len(GROUP_NAMES)

FEATURE_TICKERS = {
    'SPY': 'SPY', 'TLT': 'TLT', 'GLD': 'GLD', 'HYG': 'HYG',
    'UUP': 'UUP', 'USO': 'USO', 'VIX': '^VIX', 'SHY': 'SHY',
}
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLP', 'XLU', 'XLY', 'XLC', 'XLRE', 'XLB']

ALL_TICKERS = set()
for group_assets in ASSET_GROUPS.values():
    ALL_TICKERS.update(group_assets)
for name, ticker in FEATURE_TICKERS.items():
    ALL_TICKERS.add(ticker)
ALL_TICKERS.update(SECTOR_ETFS)

# ─── Data Download ────────────────────────────────────────────────────────────
import yfinance as yf

print("\n[1/8] Downloading market data...")
start_date = '2012-01-01'
end_date = datetime.date.today().strftime('%Y-%m-%d')

raw_data = {}
for ticker in sorted(ALL_TICKERS):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 100:
            raw_data[ticker] = df
            print(f"  {ticker}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
        else:
            print(f"  WARNING: {ticker} only {len(df)} rows, skipping")
    except Exception as e:
        print(f"  WARNING: Failed to download {ticker}: {e}")

close_prices = pd.DataFrame({t: raw_data[t]['Close'] for t in raw_data if t != '^VIX'})
close_prices = close_prices.dropna(how='all').ffill().bfill()

if '^VIX' in raw_data:
    vix_level = raw_data['^VIX']['Close'].reindex(close_prices.index).ffill().bfill()
else:
    vix_level = pd.Series(20.0, index=close_prices.index)

print(f"  Aligned matrix: {close_prices.shape[0]} days x {close_prices.shape[1]} assets")
returns = close_prices.pct_change().fillna(0)

# ─── Group Daily Returns (for evaluation) ────────────────────────────────────
group_daily_returns = {}
for gname, assets in ASSET_GROUPS.items():
    available = [a for a in assets if a in returns.columns]
    if len(available) > 0:
        group_daily_returns[gname] = returns[available].mean(axis=1)
    else:
        print(f"  WARNING: No assets for group {gname}")
        group_daily_returns[gname] = pd.Series(0.0, index=returns.index)

group_daily_ret_df = pd.DataFrame(group_daily_returns)

# ─── Feature Engineering ──────────────────────────────────────────────────────
print("\n[2/8] Engineering features (all T-1, anti-lookahead)...")

feat_tickers = [t for t in ['SPY', 'TLT', 'GLD', 'HYG', 'UUP', 'USO'] if t in returns.columns]
features_dict = {}

for t in feat_tickers:
    r = returns[t]
    c = close_prices[t]
    features_dict[f'{t}_ret_5d'] = r.rolling(5).sum().shift(1)
    features_dict[f'{t}_ret_20d'] = r.rolling(20).sum().shift(1)
    features_dict[f'{t}_ret_60d'] = r.rolling(60).sum().shift(1)
    features_dict[f'{t}_vol_5d'] = r.rolling(5).std().shift(1)
    features_dict[f'{t}_vol_20d'] = r.rolling(20).std().shift(1)
    features_dict[f'{t}_vol_60d'] = r.rolling(60).std().shift(1)
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    features_dict[f'{t}_above_sma50'] = (c.shift(1) > sma50.shift(1)).astype(float)
    features_dict[f'{t}_above_sma200'] = (c.shift(1) > sma200.shift(1)).astype(float)
    features_dict[f'{t}_sma50_dist'] = (c.shift(1) - sma50.shift(1)) / sma50.shift(1)

features_dict['VIX_level'] = vix_level.shift(1)
features_dict['VIX_5d_chg'] = vix_level.diff(5).shift(1)
features_dict['VIX_20d_chg'] = vix_level.diff(20).shift(1)
features_dict['VIX_zscore'] = ((vix_level - vix_level.rolling(60).mean()) /
                                vix_level.rolling(60).std()).shift(1)

if 'TLT' in close_prices.columns and 'SHY' in close_prices.columns:
    yc_ratio = close_prices['TLT'] / close_prices['SHY']
    features_dict['yield_curve_slope'] = yc_ratio.pct_change(20).shift(1)
    features_dict['yield_curve_level'] = ((yc_ratio - yc_ratio.rolling(60).mean()) /
                                           yc_ratio.rolling(60).std()).shift(1)

if 'HYG' in close_prices.columns and 'TLT' in close_prices.columns:
    hyg_tlt = returns['HYG'] - returns['TLT']
    features_dict['credit_spread_5d'] = hyg_tlt.rolling(5).sum().shift(1)
    features_dict['credit_spread_20d'] = hyg_tlt.rolling(20).sum().shift(1)

if len(feat_tickers) >= 3:
    vol_mat = pd.DataFrame({t: returns[t].rolling(20).std() for t in feat_tickers})
    features_dict['vol_dispersion'] = vol_mat.std(axis=1).shift(1)
    features_dict['vol_mean'] = vol_mat.mean(axis=1).shift(1)

available_sectors = [s for s in SECTOR_ETFS if s in close_prices.columns]
if len(available_sectors) >= 5:
    breadth_signals = []
    for s in available_sectors:
        sma50 = close_prices[s].rolling(50).mean()
        breadth_signals.append((close_prices[s].shift(1) > sma50.shift(1)).astype(float))
    breadth_df = pd.concat(breadth_signals, axis=1)
    features_dict['sector_breadth'] = breadth_df.mean(axis=1)
    features_dict['sector_breadth_chg'] = features_dict['sector_breadth'].diff(5)

if 'SPY' in returns.columns and 'TLT' in returns.columns:
    features_dict['spy_tlt_corr_20d'] = returns['SPY'].rolling(20).corr(returns['TLT']).shift(1)
    features_dict['spy_tlt_corr_60d'] = returns['SPY'].rolling(60).corr(returns['TLT']).shift(1)

features_df = pd.DataFrame(features_dict, index=close_prices.index)
features_df = features_df.replace([np.inf, -np.inf], np.nan)
features_df = features_df.iloc[200:]
feature_names = list(features_df.columns)
N_FEATURES = len(feature_names)
print(f"  {N_FEATURES} features engineered")
print(f"  Feature date range: {features_df.index[0].date()} to {features_df.index[-1].date()}")

# ─── Label Construction ───────────────────────────────────────────────────────
print("\n[3/8] Constructing labels (forward 21d group Sharpe -> optimal allocation)...")

group_returns_series = {gn: group_daily_returns[gn] for gn in GROUP_NAMES}

# Forward 21-day Sharpe for each group
group_fwd_sharpe = {}
for gname, gret in group_returns_series.items():
    fwd_mean = gret.rolling(LABEL_HORIZON).mean().shift(-LABEL_HORIZON)
    fwd_std = gret.rolling(LABEL_HORIZON).std().shift(-LABEL_HORIZON)
    fwd_sharpe = (fwd_mean / fwd_std.clip(lower=1e-6)) * np.sqrt(252)
    group_fwd_sharpe[gname] = fwd_sharpe

sharpe_df = pd.DataFrame(group_fwd_sharpe).reindex(features_df.index)

print(f"  Group full-sample stats (annualized):")
for gname, gret in group_returns_series.items():
    ann_ret = gret.mean() * 252
    ann_vol = gret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    print(f"    {gname}: ret={ann_ret:.1%}, vol={ann_vol:.1%}, Sharpe={sharpe:.2f}")

# Soft labels: softmax of forward Sharpe
sharpe_vals = sharpe_df.values.copy()
sharpe_vals = np.nan_to_num(sharpe_vals, nan=0.0)
temp = 1.0
exp_vals = np.exp(sharpe_vals / temp - np.nanmax(sharpe_vals / temp, axis=1, keepdims=True))
soft_labels = exp_vals / exp_vals.sum(axis=1, keepdims=True)
soft_labels_df = pd.DataFrame(soft_labels, index=sharpe_df.index, columns=GROUP_NAMES)

best_group_idx = sharpe_vals.argmax(axis=1)
print(f"\n  Best group distribution:")
for i, gname in enumerate(GROUP_NAMES):
    pct = (best_group_idx == i).sum() / len(best_group_idx) * 100
    print(f"    {gname}: {pct:.1f}%")

# ─── Transformer Model ───────────────────────────────────────────────────────
print("\n[4/8] Building Transformer model...")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class RegimeTransformer(nn.Module):
    def __init__(self, n_features, n_groups, d_model=64, n_heads=4, n_layers=2,
                 dropout=0.2, seq_len=60):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.regime_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_groups),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        batch_size = x.size(0)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.transformer(h)
        query = self.regime_query.expand(batch_size, -1, -1)
        pooled, attn_weights = self.pool_attn(query, h, h)
        pooled = pooled.squeeze(1)
        logits = self.output_head(pooled)
        return logits, attn_weights.squeeze(1)


# ─── Dataset Preparation ─────────────────────────────────────────────────────
print("\n[5/8] Preparing walk-forward sequences...")

common_idx = features_df.index.intersection(soft_labels_df.index)
features_aligned = features_df.loc[common_idx]
labels_aligned = soft_labels_df.loc[common_idx]

# Also align the daily group returns for evaluation
group_daily_aligned = group_daily_ret_df.reindex(common_idx).fillna(0)

valid_mask = ~(features_aligned.isna().any(axis=1) | labels_aligned.isna().any(axis=1))
features_aligned = features_aligned[valid_mask]
labels_aligned = labels_aligned[valid_mask]
group_daily_aligned = group_daily_aligned[valid_mask]

print(f"  Valid samples: {len(features_aligned)}")
print(f"  Date range: {features_aligned.index[0].date()} to {features_aligned.index[-1].date()}")

feature_values = features_aligned.values.astype(np.float32)
label_values = labels_aligned.values.astype(np.float32)
group_daily_values = group_daily_aligned.values.astype(np.float32)
dates = features_aligned.index

def create_sequences(feat, labels, seq_len):
    X, Y, indices = [], [], []
    for i in range(seq_len, len(feat)):
        X.append(feat[i-seq_len:i])
        Y.append(labels[i])
        indices.append(i)
    return np.array(X), np.array(Y), np.array(indices)

X_all, Y_all, idx_all = create_sequences(feature_values, label_values, SEQ_LEN)
print(f"  Total sequences: {len(X_all)}")

# ─── Walk-Forward Training & Evaluation ───────────────────────────────────────
print("\n[6/8] Walk-forward training (sliding window)...")

wf_results = []
# Store model predictions (allocation weights) keyed by date for daily evaluation
date_to_allocation = {}
fold_attn_weights_list = []

fold = 0
train_start = 0

while True:
    train_end = train_start + TRAIN_WINDOW
    test_start = train_end + GAP_DAYS
    test_end = test_start + TEST_WINDOW

    if test_end > len(X_all):
        break

    fold += 1
    test_indices = idx_all[test_start:test_end]
    test_dates_range = dates[test_indices]

    X_train = X_all[train_start:train_end]
    Y_train = Y_all[train_start:train_end]
    X_test = X_all[test_start:test_end]

    # Normalize features (fit on train only)
    scaler = RobustScaler()
    n_seq, seq_l, n_feat = X_train.shape
    scaler.fit(X_train.reshape(-1, n_feat))
    X_train_norm = np.clip(scaler.transform(X_train.reshape(-1, n_feat)).reshape(n_seq, seq_l, n_feat), -5, 5)

    n_test = X_test.shape[0]
    X_test_norm = np.clip(scaler.transform(X_test.reshape(-1, n_feat)).reshape(n_test, seq_l, n_feat), -5, 5)

    X_tr = torch.FloatTensor(X_train_norm).to(DEVICE)
    Y_tr = torch.FloatTensor(Y_train).to(DEVICE)
    X_te = torch.FloatTensor(X_test_norm).to(DEVICE)

    model = RegimeTransformer(
        n_features=N_FEATURES, n_groups=N_GROUPS,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=DROPOUT, seq_len=SEQ_LEN
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_dataset = TensorDataset(X_tr, Y_tr)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits, _ = model(xb)
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, yb, reduction='batchmean')
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            loss = loss - 0.01 * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits, test_attn = model(X_te)
        test_probs = F.softmax(test_logits, dim=1).cpu().numpy()
        test_attn_np = test_attn.cpu().numpy()

    # Store allocation for each test date
    for j, dt in enumerate(test_dates_range):
        date_to_allocation[dt] = test_probs[j]

    fold_attn_weights_list.append(test_attn_np)

    # Compute fold-level metrics using DAILY group returns (not forward cumulative)
    test_daily_group = group_daily_values[test_indices]  # (TEST_WINDOW, N_GROUPS)
    fold_daily_ret = (test_probs * test_daily_group).sum(axis=1)

    # Transaction costs at rebalance boundaries
    agg_idx = GROUP_NAMES.index('aggressive')
    costs = np.zeros(len(fold_daily_ret))
    if len(fold_daily_ret) > 1:
        wc = np.abs(np.diff(test_probs, axis=0))
        for i in range(1, len(fold_daily_ret)):
            agg_t = wc[i-1, agg_idx]
            other_t = wc[i-1].sum() - agg_t
            costs[i] = (other_t * COST_BPS_ETF + agg_t * COST_BPS_LEV) / 10000

    fold_daily_net = fold_daily_ret - costs
    ew_daily = test_daily_group.mean(axis=1)
    b6040_daily = 0.6 * test_daily_group[:, GROUP_NAMES.index('growth')] + \
                  0.4 * test_daily_group[:, GROUP_NAMES.index('safety')]

    def quick_sharpe(r):
        if len(r) < 2 or np.std(r) == 0:
            return 0.0
        return np.mean(r) / np.std(r) * np.sqrt(252)

    result = {
        'fold': fold,
        'test_start': test_dates_range[0].date(),
        'test_end': test_dates_range[-1].date(),
        'model_sharpe': quick_sharpe(fold_daily_net),
        'ew_sharpe': quick_sharpe(ew_daily),
        'b6040_sharpe': quick_sharpe(b6040_daily),
        'model_return': fold_daily_net.sum(),
        'ew_return': ew_daily.sum(),
        'avg_alloc': {gn: test_probs[:, i].mean() for i, gn in enumerate(GROUP_NAMES)},
        'best_loss': best_loss,
        'epochs_used': epoch + 1,
    }
    wf_results.append(result)

    if fold % 5 == 0 or fold <= 3:
        alloc_str = " | ".join([f"{gn}:{test_probs[:, i].mean():.0%}"
                                for i, gn in enumerate(GROUP_NAMES)])
        print(f"  Fold {fold:3d}: {result['test_start']} to {result['test_end']} | "
              f"Sharpe: M={result['model_sharpe']:+.2f} EW={result['ew_sharpe']:+.2f} "
              f"6040={result['b6040_sharpe']:+.2f} | {alloc_str}")

    train_start += TEST_WINDOW

print(f"\n  Completed {fold} walk-forward folds")

# ─── Aggregate Results (DAILY returns, not 21d cumulative) ────────────────────
print("\n[7/8] Aggregate performance analysis...")

# Build daily portfolio return series from date_to_allocation
oot_dates = sorted(date_to_allocation.keys())
oot_allocations = np.array([date_to_allocation[d] for d in oot_dates])
oot_daily_group = group_daily_ret_df.loc[oot_dates].values  # (N_days, N_groups)

# Model daily returns
model_daily = (oot_allocations * oot_daily_group).sum(axis=1)

# Transaction costs
agg_idx = GROUP_NAMES.index('aggressive')
costs_agg = np.zeros(len(model_daily))
wc_full = np.abs(np.diff(oot_allocations, axis=0))
for i in range(1, len(model_daily)):
    agg_t = wc_full[i-1, agg_idx]
    other_t = wc_full[i-1].sum() - agg_t
    costs_agg[i] = (other_t * COST_BPS_ETF + agg_t * COST_BPS_LEV) / 10000

model_daily_net = model_daily - costs_agg

# Baselines
ew_daily_full = oot_daily_group.mean(axis=1)
growth_i = GROUP_NAMES.index('growth')
safety_i = GROUP_NAMES.index('safety')
b6040_daily_full = 0.6 * oot_daily_group[:, growth_i] + 0.4 * oot_daily_group[:, safety_i]

# SPY daily
spy_daily = returns['SPY'].reindex(pd.DatetimeIndex(oot_dates)).values

def compute_metrics(ret, name):
    ret = np.array(ret)
    n = len(ret)
    if n < 2 or np.nanstd(ret) == 0:
        return {'name': name, 'n_days': n, 'sharpe': 0}
    ann_ret = np.nanmean(ret) * 252
    ann_vol = np.nanstd(ret) * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    downside = ret[ret < 0]
    dv = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = ann_ret / dv
    cum = np.nancumsum(ret)
    rmax = np.maximum.accumulate(cum)
    dd = rmax - cum
    max_dd = dd.max()
    calmar = ann_ret / max_dd if max_dd > 0 else float('inf')
    gp = ret[ret > 0].sum()
    gl = abs(ret[ret < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    wr = (ret > 0).sum() / n
    total_ret = np.exp(np.nancumsum(np.log1p(ret))[-1]) - 1 if n > 0 else 0
    return {
        'name': name, 'n_days': n, 'ann_return': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd, 'calmar': calmar,
        'profit_factor': pf, 'win_rate': wr, 'total_return': total_ret,
    }

metrics = {
    'model_net': compute_metrics(model_daily_net, 'Transformer (net)'),
    'model_gross': compute_metrics(model_daily, 'Transformer (gross)'),
    'ew': compute_metrics(ew_daily_full, 'Equal Weight'),
    'b6040': compute_metrics(b6040_daily_full, '60/40'),
    'spy': compute_metrics(spy_daily, 'SPY B&H'),
}

print("\n" + "=" * 80)
print("AGGREGATE OUT-OF-SAMPLE RESULTS (DAILY RETURNS)")
print("=" * 80)

for m in [metrics['model_net'], metrics['model_gross'], metrics['ew'], metrics['b6040'], metrics['spy']]:
    if 'sharpe' in m and m['sharpe'] != 0:
        print(f"\n  {m['name']}:")
        print(f"    Sharpe:     {m['sharpe']:.3f}")
        print(f"    Sortino:    {m['sortino']:.3f}")
        print(f"    Ann Return: {m['ann_return']:.1%}")
        print(f"    Ann Vol:    {m['ann_vol']:.1%}")
        print(f"    Max DD:     {m['max_dd']:.1%}")
        print(f"    Calmar:     {m['calmar']:.3f}")
        print(f"    PF:         {m['profit_factor']:.2f}")
        print(f"    WR:         {m['win_rate']:.1%}")
        print(f"    Total Ret:  {m['total_return']:.1%}")
        print(f"    N days:     {m['n_days']}")

print(f"\n  Average OOT allocation:")
for i, gn in enumerate(GROUP_NAMES):
    print(f"    {gn}: {oot_allocations[:, i].mean():.1%}")

# ─── Regime-Agnostic Analysis ────────────────────────────────────────────────
print("\n" + "-" * 80)
print("REGIME-AGNOSTIC ANALYSIS")
print("-" * 80)

green_mask = spy_daily > 0.002
red_mask = spy_daily < -0.002
flat_mask = ~green_mask & ~red_mask

regime_sharpes = {}
for regime_name, mask in [('GREEN', green_mask), ('RED', red_mask), ('FLAT', flat_mask)]:
    mask = np.array(mask, dtype=bool)
    if mask.sum() > 10:
        rm = model_daily_net[mask]
        re = ew_daily_full[mask]
        ms = rm.mean() / rm.std() * np.sqrt(252) if rm.std() > 0 else 0
        es = re.mean() / re.std() * np.sqrt(252) if re.std() > 0 else 0
        regime_sharpes[regime_name] = ms
        label = {'GREEN': 'SPY>+0.2%', 'RED': 'SPY<-0.2%', 'FLAT': '|SPY|<0.2%'}[regime_name]
        print(f"  {regime_name} ({label}, {mask.sum()} days): Model Sharpe={ms:+.2f}, EW Sharpe={es:+.2f}")

gs = regime_sharpes.get('GREEN', 0)
rs = regime_sharpes.get('RED', 0)
denom = max(abs(gs), abs(rs), 0.01)
regime_div = abs(gs - rs) / denom
print(f"\n  Regime divergence: {regime_div:.2f} (threshold: <0.50)")
print(f"  {'PASS' if regime_div < 0.50 else 'WARNING: Regime-tailored'}")

# ─── Permutation Test ────────────────────────────────────────────────────────
print("\n" + "-" * 80)
print(f"PERMUTATION TEST ({N_PERMUTATIONS} shuffles)")
print("-" * 80)

actual_sharpe = metrics['model_net']['sharpe']
perm_sharpes = []

for perm in range(N_PERMUTATIONS):
    shuffled_alloc = oot_allocations.copy()
    np.random.shuffle(shuffled_alloc)
    perm_ret = (shuffled_alloc * oot_daily_group).sum(axis=1)
    # Costs
    pwc = np.abs(np.diff(shuffled_alloc, axis=0))
    pc = np.zeros(len(perm_ret))
    for i in range(1, len(perm_ret)):
        at = pwc[i-1, agg_idx]
        ot = pwc[i-1].sum() - at
        pc[i] = (ot * COST_BPS_ETF + at * COST_BPS_LEV) / 10000
    perm_net = perm_ret - pc
    if perm_net.std() > 0:
        perm_sharpes.append(perm_net.mean() / perm_net.std() * np.sqrt(252))
    else:
        perm_sharpes.append(0)

perm_sharpes = np.array(perm_sharpes)
perm_p = (perm_sharpes >= actual_sharpe).sum() / N_PERMUTATIONS

print(f"  Actual OOT Sharpe:   {actual_sharpe:.3f}")
print(f"  Permutation mean:    {perm_sharpes.mean():.3f}")
print(f"  Permutation std:     {perm_sharpes.std():.3f}")
print(f"  Permutation p-value: {perm_p:.3f}")
if perm_p < 0.05:
    print(f"  SIGNIFICANT at 5% — model allocation adds value")
elif perm_p < 0.10:
    print(f"  MARGINAL significance (p<0.10)")
else:
    print(f"  NOT significant — allocation doesn't beat random")

# ─── Temporal Stability & Attention Analysis ──────────────────────────────────
print("\n" + "-" * 80)
print("TEMPORAL STABILITY & ATTENTION ANALYSIS")
print("-" * 80)

fold_sharpes = [r['model_sharpe'] for r in wf_results]
n_folds = len(fold_sharpes)
h1 = fold_sharpes[:n_folds//2]
h2 = fold_sharpes[n_folds//2:]
print(f"  First half avg fold Sharpe:  {np.mean(h1):.3f}")
print(f"  Second half avg fold Sharpe: {np.mean(h2):.3f}")
stable = abs(np.mean(h1) - np.mean(h2)) < 0.5 * max(abs(np.mean(h1)), abs(np.mean(h2)), 0.01)
print(f"  Temporal stability: {'STABLE' if stable else 'UNSTABLE'}")

# Attention distribution
all_attn = np.concatenate(fold_attn_weights_list, axis=0)
avg_attn = all_attn.mean(axis=0)
print(f"\n  Attention weight distribution across {SEQ_LEN}-day sequence:")
print(f"    Days  1-10 (oldest):  {avg_attn[:10].sum():.1%}")
print(f"    Days 11-30 (mid):     {avg_attn[10:30].sum():.1%}")
print(f"    Days 31-50 (recent):  {avg_attn[30:50].sum():.1%}")
print(f"    Days 51-60 (latest):  {avg_attn[50:].sum():.1%}")
print(f"    Peak attention at day: {avg_attn.argmax() + 1}")

# ─── Yearly Breakdown ─────────────────────────────────────────────────────────
print("\n" + "-" * 80)
print("YEARLY PERFORMANCE BREAKDOWN")
print("-" * 80)

oot_dates_pd = pd.DatetimeIndex(oot_dates)
yearly_data = pd.DataFrame({
    'model': model_daily_net,
    'ew': ew_daily_full,
    'spy': spy_daily,
}, index=oot_dates_pd)

print(f"\n  {'Year':<6} {'Model Sharpe':>12} {'EW Sharpe':>10} {'SPY Sharpe':>10} {'Model Ret':>10} {'SPY Ret':>10}")
print(f"  {'-'*60}")

for year in sorted(yearly_data.index.year.unique()):
    yr = yearly_data[yearly_data.index.year == year]
    if len(yr) < 20:
        continue
    ms = yr['model'].mean() / yr['model'].std() * np.sqrt(252) if yr['model'].std() > 0 else 0
    es = yr['ew'].mean() / yr['ew'].std() * np.sqrt(252) if yr['ew'].std() > 0 else 0
    ss = yr['spy'].mean() / yr['spy'].std() * np.sqrt(252) if yr['spy'].std() > 0 else 0
    mr = yr['model'].sum()
    sr = yr['spy'].sum()
    print(f"  {year:<6} {ms:>12.2f} {es:>10.2f} {ss:>10.2f} {mr:>9.1%} {sr:>9.1%}")

# ─── Final Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)
print(f"  Model: Transformer ({N_LAYERS}L, {N_HEADS}H, d={D_MODEL})")
print(f"  Features: {N_FEATURES} cross-asset features, {SEQ_LEN}d sequences")
print(f"  Walk-forward: {fold} folds, {TRAIN_WINDOW}d train / {TEST_WINDOW}d test / {GAP_DAYS}d gap")
print(f"  OOT test period: {oot_dates[0].date()} to {oot_dates[-1].date()}")
print(f"  OOT Sharpe (net): {metrics['model_net']['sharpe']:.3f}")
print(f"  vs Equal Weight:  {metrics['ew']['sharpe']:.3f}")
print(f"  vs 60/40:         {metrics['b6040']['sharpe']:.3f}")
print(f"  vs SPY B&H:       {metrics['spy']['sharpe']:.3f}")
print(f"  Permutation p:    {perm_p:.3f}")
print(f"  Regime divergence: {regime_div:.2f}")
print(f"  Costs: {COST_BPS_ETF}bps ETF, {COST_BPS_LEV}bps leveraged")

alpha_ew = metrics['model_net']['sharpe'] - metrics['ew']['sharpe']
alpha_spy = metrics['model_net']['sharpe'] - metrics['spy']['sharpe']
print(f"\n  Alpha over EW:  {alpha_ew:+.3f} Sharpe points")
print(f"  Alpha over SPY: {alpha_spy:+.3f} Sharpe points")

beats_ew = alpha_ew > 0
beats_spy = alpha_spy > 0
significant = perm_p < 0.10
regime_ok = regime_div < 0.50

print(f"\n  VERDICT:")
if beats_ew and significant and regime_ok:
    print(f"    PROMISING — beats equal-weight, significant, regime-agnostic")
elif beats_spy and regime_ok:
    print(f"    INTERESTING — beats SPY but {'not significant' if not significant else 'regime-dependent'}")
else:
    issues = []
    if not beats_ew: issues.append("doesn't beat equal-weight")
    if not significant: issues.append(f"not significant (p={perm_p:.2f})")
    if not regime_ok: issues.append(f"regime-dependent (div={regime_div:.2f})")
    print(f"    NEEDS WORK — {', '.join(issues)}")

print(f"\n[{datetime.datetime.now()}] Experiment complete.")
