#!/usr/bin/env python3
"""
DL Strategy Ensemble Optimizer v1
==================================
CONTEXT: Individual strategy research has produced several validated signals:
  - Carry+Momentum Rotation (Sharpe 2.96, CAGR 44.5%, MaxDD -7%)
  - Commodity Trend Following (Sharpe 2.28, CAGR 54.9%, MaxDD -18.5%)
  - Stat Arb pairs (Sharpe 0.81, CAGR 8.1%, MaxDD -11.5%)
  - Earnings Asymmetry (Sharpe 1.05, sparse but real)
  - Vol Compression Breakout (Sharpe ~6 at 5d, needs position sizing)

APPROACH: Instead of picking individual stocks, learn DYNAMIC ALLOCATION across
diversified ETF baskets that proxy these strategy themes, using a neural net
that adapts weights based on regime features.

ARCHITECTURE:
  - Input: 30-day rolling features for each asset class + regime indicators
  - LSTM encoder to capture temporal regime dynamics
  - Attention-weighted allocation head
  - Output: portfolio weights across 15 diversified ETFs
  - Loss: Sharpe-ratio maximization + drawdown penalty

This is fundamentally different from cross-sectional stock picking (which failed).
This is TEMPORAL regime-adaptive allocation across diversified assets.

Walk-forward 504d/21d SLIDING (HC #0).
Commission-free (HC #694) + 0.05% BA spread.
GPU: Neptune RTX 3090.
"""

import os
import sys
import json
import time
import logging
import warnings
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─── Config ───
CONFIG = {
    'experiment_name': 'dl_strategy_ensemble_v1',
    'start_date': '2008-01-01',
    'end_date': '2026-07-18',
    'train_days': 504,
    'test_days': 21,
    'lookback': 60,         # 60 trading days of features
    'cost_bps': 5,          # 0.05% BA spread
    'rebalance_freq': 21,   # Monthly
    # Model
    'hidden_dim': 64,
    'n_lstm_layers': 2,
    'n_attention_heads': 2,
    'dropout': 0.2,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'epochs': 200,
    'patience': 20,
    'batch_size': 8,
    'max_weight': 0.30,      # Max 30% in any single ETF
    'min_weight': 0.0,       # Allow zero weight (long-only)
    'dd_penalty': 2.0,       # Drawdown penalty coefficient
    # Validation
    'n_permutations': 200,
    'regime_gap_max': 0.50,
    'subperiod_cv_max': 0.70,
    # Output
    'output_dir': '/home/nick/Lvl3Quant/output/dl_strategy_ensemble_v1',
    'mlflow_tracking_uri': 'http://jupiter:5000',
    'mlflow_experiment': 'dl_strategy_ensemble_v1',
}

OUTPUT_DIR = CONFIG['output_dir']
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'training.log')),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

log.info("=" * 70)
log.info("DL STRATEGY ENSEMBLE OPTIMIZER v1")
log.info("Regime-Adaptive Dynamic Allocation via LSTM+Attention")
log.info("=" * 70)

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ═══════════════════════════════════════════════════════════════════════
# ETF UNIVERSE — Diversified strategy proxies
# ═══════════════════════════════════════════════════════════════════════

# These ETFs proxy our validated strategy themes:
UNIVERSE = {
    # Growth / Momentum (carry+momentum proxy)
    'QQQ':  'tech_growth',
    'VGT':  'tech_sector',
    'XLY':  'consumer_disc',
    # Income / Dividend (carry proxy)
    'VYM':  'high_dividend',
    'SCHD': 'dividend_growth',
    # Commodity trend following
    'GLD':  'gold',
    'DBC':  'broad_commodity',
    'USO':  'oil',
    # Defensive / Low vol
    'XLU':  'utilities',
    'XLP':  'consumer_staples',
    'TLT':  'long_treasury',
    # Broad market
    'SPY':  'sp500',
    'IWM':  'small_cap',
    # International diversification
    'EFA':  'developed_intl',
    'VWO':  'emerging_markets',
}

TICKERS = list(UNIVERSE.keys())
N_ASSETS = len(TICKERS)

# Regime indicators (downloaded separately)
REGIME_TICKERS = ['^VIX', '^TNX', 'HYG', 'LQD']

# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

def download_data():
    """Download OHLCV for universe + regime indicators."""
    import yfinance as yf

    cache_file = os.path.join(OUTPUT_DIR, 'data_cache.pkl')
    if os.path.exists(cache_file):
        cache_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).total_seconds()
        if cache_age < 86400:
            log.info("  Loading cached data")
            return pd.read_pickle(cache_file)

    all_tickers = TICKERS + REGIME_TICKERS
    log.info(f"  Downloading {len(all_tickers)} tickers...")

    data = yf.download(all_tickers, start=CONFIG['start_date'], end=CONFIG['end_date'],
                       auto_adjust=True, group_by='ticker', threads=True, progress=False)

    result = {}
    for t in all_tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                sub = data[t][['Close', 'Volume']].dropna()
            else:
                sub = data[['Close', 'Volume']].dropna()
            if len(sub) > 252:
                result[t] = sub
                log.info(f"    {t}: {len(sub)} days")
        except Exception as e:
            log.warning(f"    {t} failed: {e}")

    pd.to_pickle(result, cache_file)
    log.info(f"  Cached {len(result)} tickers")
    return result


def build_features(data):
    """
    Build temporal feature tensors for each rebalance date.

    For each rebalance date, features are:
    Per-asset (lookback × N_assets × 8 features):
      1. Normalized return (20d)
      2. Normalized return (5d)
      3. Realized vol (20d)
      4. Relative strength vs SPY
      5. Volume trend (20d vs 60d)
      6. Distance from 60d high
      7. Momentum (20d / 60d)
      8. Return rank (cross-sectional)

    Regime features (lookback × 4):
      1. VIX level (z-scored)
      2. VIX change (5d)
      3. Treasury yield (z-scored)
      4. Credit spread (HYG-LQD spread, z-scored)
    """
    # Build aligned close price matrix
    close_df = pd.DataFrame()
    volume_df = pd.DataFrame()

    for t in TICKERS:
        if t in data:
            close_df[t] = data[t]['Close']
            volume_df[t] = data[t]['Volume']

    # Forward-fill and drop rows with too many NaN
    close_df = close_df.ffill().dropna(thresh=len(TICKERS) - 3)
    volume_df = volume_df.ffill().reindex(close_df.index)

    # Regime data
    regime_df = pd.DataFrame()
    if '^VIX' in data:
        regime_df['vix'] = data['^VIX']['Close']
    if '^TNX' in data:
        regime_df['tnx'] = data['^TNX']['Close']
    if 'HYG' in data and 'LQD' in data:
        regime_df['credit_spread'] = data['HYG']['Close'] / data['LQD']['Close']

    regime_df = regime_df.ffill().reindex(close_df.index).ffill().bfill()

    # Compute daily returns
    returns = close_df.pct_change().fillna(0)

    # Monthly rebalance dates
    monthly_dates = returns.resample('ME').last().index
    monthly_dates = [d for d in monthly_dates if d in returns.index]

    lookback = CONFIG['lookback']
    feature_list = []
    return_list = []
    date_list = []

    for i, date in enumerate(monthly_dates):
        if i < 1:
            continue
        if i >= len(monthly_dates) - 1:
            continue

        date_loc = returns.index.get_loc(date)
        if date_loc < lookback + 60:
            continue

        # Forward 21d returns (target)
        next_date = monthly_dates[i + 1]
        fwd_rets = (close_df.loc[next_date] / close_df.loc[date] - 1).reindex(TICKERS).values

        if np.any(np.isnan(fwd_rets)):
            # Fill missing with 0
            fwd_rets = np.nan_to_num(fwd_rets, 0)

        # Build lookback features for each day in the window
        asset_features = np.zeros((lookback, N_ASSETS, 8), dtype=np.float32)
        regime_features = np.zeros((lookback, 4), dtype=np.float32)

        for day_i in range(lookback):
            d_idx = date_loc - lookback + day_i + 1
            if d_idx < 60:
                continue

            # Per-asset features
            c = close_df.iloc[:d_idx + 1]
            v = volume_df.iloc[:d_idx + 1]
            r = returns.iloc[:d_idx + 1]

            for a_idx, ticker in enumerate(TICKERS):
                if ticker not in close_df.columns:
                    continue

                # 1. Normalized 20d return
                ret_20d = c[ticker].iloc[-1] / c[ticker].iloc[-20] - 1 if len(c) >= 20 else 0
                # 2. Normalized 5d return
                ret_5d = c[ticker].iloc[-1] / c[ticker].iloc[-5] - 1 if len(c) >= 5 else 0
                # 3. Realized vol (20d annualized)
                vol_20d = r[ticker].iloc[-20:].std() * np.sqrt(252) if len(r) >= 20 else 0
                # 4. Relative strength vs SPY
                spy_ret = c['SPY'].iloc[-1] / c['SPY'].iloc[-20] - 1 if 'SPY' in c.columns and len(c) >= 20 else 0
                rel_str = ret_20d - spy_ret
                # 5. Volume trend
                v20 = v[ticker].iloc[-20:].mean() if len(v) >= 20 else 1
                v60 = v[ticker].iloc[-60:].mean() if len(v) >= 60 else 1
                vol_trend = v20 / max(v60, 1) - 1
                # 6. Distance from 60d high
                high_60 = c[ticker].iloc[-60:].max() if len(c) >= 60 else c[ticker].iloc[-1]
                dist_high = c[ticker].iloc[-1] / high_60 - 1
                # 7. Momentum (20d / 60d)
                ret_60d = c[ticker].iloc[-1] / c[ticker].iloc[-60] - 1 if len(c) >= 60 else 0
                mom_ratio = ret_20d / max(abs(ret_60d), 0.001)
                # 8. Cross-sectional return rank (filled below)
                asset_features[day_i, a_idx, :7] = [ret_20d, ret_5d, vol_20d, rel_str, vol_trend, dist_high, mom_ratio]

            # Cross-sectional rank of 20d returns
            rets_20d = asset_features[day_i, :, 0]
            ranks = pd.Series(rets_20d).rank(pct=True).values
            asset_features[day_i, :, 7] = ranks

            # Regime features
            if d_idx < len(regime_df):
                rd = regime_df.iloc[:d_idx + 1]
                if 'vix' in regime_df.columns and len(rd) >= 252:
                    vix_val = rd['vix'].iloc[-1]
                    vix_mean = rd['vix'].iloc[-252:].mean()
                    vix_std = rd['vix'].iloc[-252:].std()
                    regime_features[day_i, 0] = (vix_val - vix_mean) / max(vix_std, 0.1)
                    regime_features[day_i, 1] = (vix_val / rd['vix'].iloc[-5] - 1) if len(rd) >= 5 else 0
                if 'tnx' in regime_df.columns and len(rd) >= 252:
                    tnx_val = rd['tnx'].iloc[-1]
                    tnx_mean = rd['tnx'].iloc[-252:].mean()
                    tnx_std = rd['tnx'].iloc[-252:].std()
                    regime_features[day_i, 2] = (tnx_val - tnx_mean) / max(tnx_std, 0.1)
                if 'credit_spread' in regime_df.columns and len(rd) >= 252:
                    cs_val = rd['credit_spread'].iloc[-1]
                    cs_mean = rd['credit_spread'].iloc[-252:].mean()
                    cs_std = rd['credit_spread'].iloc[-252:].std()
                    regime_features[day_i, 3] = (cs_val - cs_mean) / max(cs_std, 0.001)

        # Clip extreme values
        asset_features = np.clip(asset_features, -5, 5)
        regime_features = np.clip(regime_features, -5, 5)

        feature_list.append({
            'asset_features': asset_features,   # (lookback, N_assets, 8)
            'regime_features': regime_features,  # (lookback, 4)
        })
        return_list.append(fwd_rets)  # (N_assets,)
        date_list.append(date)

    log.info(f"  Built {len(feature_list)} monthly samples, {N_ASSETS} assets, {lookback} lookback days")

    return feature_list, np.array(return_list), date_list


# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════

class RegimeEncoder(nn.Module):
    """LSTM to encode regime dynamics from temporal features."""
    def __init__(self, n_regime_feats=4, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_regime_feats,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, n_regime_feats)
        out, (h, c) = self.lstm(x)
        return self.norm(out[:, -1, :])  # Last hidden state: (B, hidden_dim)


class AssetEncoder(nn.Module):
    """Process per-asset temporal features."""
    def __init__(self, n_asset_feats=8, hidden_dim=64, n_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_asset_feats,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, n_asset_feats) — for one asset
        out, (h, c) = self.lstm(x)
        return self.norm(out[:, -1, :])  # (B, hidden_dim)


class StrategyEnsembleNet(nn.Module):
    """
    Regime-adaptive portfolio allocation network.

    For each asset: LSTM encodes its temporal features.
    Regime LSTM encodes macro conditions.
    Attention combines asset embeddings conditioned on regime.
    Output: portfolio weights (softmax, bounded).
    """
    def __init__(self, n_assets, n_asset_feats=8, n_regime_feats=4,
                 hidden_dim=64, n_heads=2, dropout=0.2, max_weight=0.3):
        super().__init__()
        self.n_assets = n_assets
        self.max_weight = max_weight

        self.asset_encoder = AssetEncoder(n_asset_feats, hidden_dim, 1, dropout)
        self.regime_encoder = RegimeEncoder(n_regime_feats, hidden_dim, 2, dropout)

        # Attention: regime state attends to asset embeddings
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Allocation head
        self.alloc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, asset_features, regime_features):
        """
        asset_features: (B, T, N_assets, n_asset_feats)
        regime_features: (B, T, n_regime_feats)
        Returns: (B, N_assets) portfolio weights summing to 1
        """
        B, T, N, F = asset_features.shape

        # Encode each asset's temporal features
        asset_embeds = []
        for i in range(N):
            embed = self.asset_encoder(asset_features[:, :, i, :])  # (B, hidden)
            asset_embeds.append(embed)
        asset_embeds = torch.stack(asset_embeds, dim=1)  # (B, N, hidden)

        # Encode regime
        regime_embed = self.regime_encoder(regime_features)  # (B, hidden)
        regime_query = regime_embed.unsqueeze(1)  # (B, 1, hidden)

        # Cross-attention: regime queries asset embeddings
        attended, _ = self.attention(regime_query, asset_embeds, asset_embeds)  # (B, 1, hidden)

        # But we need per-asset scores, so project each asset embed
        # conditioned on the attended context
        context = attended.expand(-1, N, -1)  # (B, N, hidden)
        combined = asset_embeds + context  # Residual connection

        raw_weights = self.alloc_head(combined).squeeze(-1)  # (B, N)

        # Softmax for portfolio weights, then clip to max_weight
        weights = torch.softmax(raw_weights, dim=-1)

        # Enforce max weight constraint via iterative clipping
        if self.max_weight < 1.0:
            for _ in range(3):
                excess = (weights - self.max_weight).clamp(min=0)
                if excess.sum() < 1e-8:
                    break
                weights = weights - excess
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return weights


def sharpe_loss(weights, returns, dd_penalty=2.0):
    """
    Differentiable Sharpe ratio loss (negative, for minimization).
    Plus drawdown penalty.

    weights: (B, N) portfolio weights
    returns: (B, N) forward returns per asset
    """
    # Portfolio return per period
    port_ret = (weights * returns).sum(dim=1)  # (B,)

    # Sharpe component
    mean_ret = port_ret.mean()
    std_ret = port_ret.std() + 1e-8
    neg_sharpe = -mean_ret / std_ret

    # Drawdown penalty
    cum_ret = (1 + port_ret).cumprod(dim=0)
    running_max = torch.cummax(cum_ret, dim=0)[0]
    drawdown = (cum_ret / running_max - 1)
    max_dd = drawdown.min()
    dd_loss = dd_penalty * (-max_dd).clamp(min=0)

    return neg_sharpe + dd_loss


# ═══════════════════════════════════════════════════════════════════════
# WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════

def run_walk_forward(features, returns, dates):
    """Walk-forward with SLIDING window."""
    n = len(features)
    train_periods = CONFIG['train_days'] // CONFIG['test_days']  # ~24 months

    log.info(f"\nWalk-forward: {n} periods, {train_periods} train window")

    all_weights = []
    all_rets = []
    all_dates = []

    for fold_start in range(train_periods, n):
        fold_num = fold_start - train_periods
        train_idx = list(range(fold_start - train_periods, fold_start))
        test_idx = fold_start

        # Prepare training data
        train_asset = torch.FloatTensor(np.array([features[i]['asset_features'] for i in train_idx]))
        train_regime = torch.FloatTensor(np.array([features[i]['regime_features'] for i in train_idx]))
        train_rets = torch.FloatTensor(returns[train_idx])

        # Validation split (last 20%)
        val_size = max(2, len(train_idx) // 5)
        val_asset = train_asset[-val_size:]
        val_regime = train_regime[-val_size:]
        val_rets = train_rets[-val_size:]
        tr_asset = train_asset[:-val_size]
        tr_regime = train_regime[:-val_size]
        tr_rets = train_rets[:-val_size]

        if len(tr_asset) < 4:
            continue

        # Test data
        test_asset = torch.FloatTensor(features[test_idx]['asset_features']).unsqueeze(0)
        test_regime = torch.FloatTensor(features[test_idx]['regime_features']).unsqueeze(0)
        test_ret = returns[test_idx]

        # Create model
        model = StrategyEnsembleNet(
            n_assets=N_ASSETS,
            n_asset_feats=8,
            n_regime_feats=4,
            hidden_dim=CONFIG['hidden_dim'],
            n_heads=CONFIG['n_attention_heads'],
            dropout=CONFIG['dropout'],
            max_weight=CONFIG['max_weight'],
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['lr'],
            weight_decay=CONFIG['weight_decay'],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(CONFIG['epochs']):
            model.train()

            # Mini-batch training
            idx = torch.randperm(len(tr_asset))
            batch_size = CONFIG['batch_size']
            total_loss = 0
            n_batches = 0

            for start in range(0, len(tr_asset), batch_size):
                bi = idx[start:start + batch_size]
                a_b = tr_asset[bi].to(device)
                r_b = tr_regime[bi].to(device)
                ret_b = tr_rets[bi].to(device)

                optimizer.zero_grad()
                weights = model(a_b, r_b)
                loss = sharpe_loss(weights, ret_b, CONFIG['dd_penalty'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validate
            model.eval()
            with torch.no_grad():
                val_w = model(val_asset.to(device), val_regime.to(device))
                val_loss = sharpe_loss(val_w, val_rets.to(device), CONFIG['dd_penalty']).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= CONFIG['patience']:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Predict
        model.eval()
        with torch.no_grad():
            pred_weights = model(test_asset.to(device), test_regime.to(device)).cpu().numpy()[0]

        all_weights.append(pred_weights)
        all_rets.append(test_ret)
        all_dates.append(dates[test_idx])

        if fold_num % 10 == 0:
            port_ret = (pred_weights * test_ret).sum()
            top3 = np.argsort(pred_weights)[-3:][::-1]
            top3_str = ', '.join([f"{TICKERS[i]}:{pred_weights[i]:.1%}" for i in top3])
            log.info(f"  Fold {fold_num}: {dates[test_idx].strftime('%Y-%m-%d')}, "
                     f"ret={port_ret:.3f}, top3=[{top3_str}]")

    return np.array(all_weights), np.array(all_rets), all_dates


def compute_metrics(rets, label=''):
    """Compute portfolio metrics from monthly returns array."""
    n = len(rets)
    if n < 5:
        return {}

    ann = 12
    mean_r = rets.mean()
    std_r = rets.std()
    sharpe = mean_r / std_r * np.sqrt(ann) if std_r > 0 else 0

    down = rets[rets < 0]
    down_std = down.std() if len(down) > 1 else std_r
    sortino = mean_r / down_std * np.sqrt(ann) if down_std > 0 else 0

    cum = np.cumprod(1 + rets)
    max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
    cagr = cum[-1] ** (ann / n) - 1

    wr = (rets > 0).mean()
    pos = rets[rets > 0].sum()
    neg = abs(rets[rets < 0].sum())
    pf = pos / neg if neg > 0 else float('inf')

    vol = std_r * np.sqrt(ann)

    result = {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(wr, 4),
        'cagr': round(cagr, 4),
        'max_dd': round(max_dd, 4),
        'vol': round(vol, 4),
        'n_periods': n,
        'total_ret': round(cum[-1] - 1, 4),
    }

    if label:
        log.info(f"\n  {label}:")
        for k, v in result.items():
            log.info(f"    {k}: {v}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════════════

def equal_weight_baseline(returns):
    """Equal-weight across all assets."""
    n_assets = returns.shape[1]
    weights = np.ones(n_assets) / n_assets
    rets = (returns * weights).sum(axis=1)
    return rets


def momentum_baseline(features, returns, dates):
    """Simple 12-1 momentum: long top 5, equal weight."""
    rets = []
    for i in range(len(features)):
        # Use 20d return (feature index 0 in asset_features, last day)
        mom = features[i]['asset_features'][-1, :, 0]  # Last day, all assets, feature 0 (20d ret)
        top5 = np.argsort(mom)[-5:]
        w = np.zeros(N_ASSETS)
        w[top5] = 1.0 / 5
        rets.append((w * returns[i]).sum())
    return np.array(rets)


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def permutation_test(dl_rets, baseline_rets, n_perms=200):
    """
    Permutation test: shuffle time alignment between DL weights and returns.
    Tests whether the model's TIMING adds value over random rebalancing.
    """
    log.info(f"\nRunning {n_perms}-shuffle permutation test...")

    observed_sharpe = dl_rets.mean() / dl_rets.std() * np.sqrt(12) if dl_rets.std() > 0 else 0

    null_sharpes = []
    for _ in range(n_perms):
        # Shuffle the returns (break temporal alignment)
        shuffled_rets = dl_rets.copy()
        np.random.shuffle(shuffled_rets)
        s = shuffled_rets.mean() / shuffled_rets.std() * np.sqrt(12) if shuffled_rets.std() > 0 else 0
        null_sharpes.append(s)

    null_sharpes = np.array(null_sharpes)
    p = (null_sharpes >= observed_sharpe).mean()

    result = {
        'observed_sharpe': round(float(observed_sharpe), 4),
        'null_mean': round(float(null_sharpes.mean()), 4),
        'null_std': round(float(null_sharpes.std()), 4),
        'p_value': round(float(p), 4),
        'pass': bool(p < 0.05),
    }
    log.info(f"  Perm test: p={p:.3f}, obs={observed_sharpe:.3f}, null={null_sharpes.mean():.3f}±{null_sharpes.std():.3f}")
    return result


def regime_test(port_rets, dates):
    """R1: Regime-agnostic. Split by SPY monthly returns."""
    import yfinance as yf

    log.info("\nRunning regime test...")
    spy = yf.download('SPY', start='2007-01-01', end='2027-01-01', auto_adjust=True, progress=False)
    spy_monthly = spy['Close'].resample('ME').last().pct_change().dropna()

    bull_rets = []
    bear_rets = []

    for d, r in zip(dates, port_rets):
        try:
            closest_idx = spy_monthly.index.get_indexer([d], method='nearest')[0]
            spy_ret = spy_monthly.iloc[closest_idx]
            if hasattr(spy_ret, 'item'):
                spy_ret = spy_ret.item()
            elif hasattr(spy_ret, 'iloc'):
                spy_ret = spy_ret.iloc[0]
            if spy_ret > 0:
                bull_rets.append(r)
            else:
                bear_rets.append(r)
        except:
            bull_rets.append(r)

    bull_rets = np.array(bull_rets)
    bear_rets = np.array(bear_rets)

    s_bull = bull_rets.mean() / bull_rets.std() * np.sqrt(12) if len(bull_rets) > 3 and bull_rets.std() > 0 else 0
    s_bear = bear_rets.mean() / bear_rets.std() * np.sqrt(12) if len(bear_rets) > 3 and bear_rets.std() > 0 else 0

    gap = abs(s_bull - s_bear) / max(abs(s_bull), abs(s_bear), 0.001)

    result = {
        'sharpe_bull': round(s_bull, 3),
        'sharpe_bear': round(s_bear, 3),
        'n_bull': len(bull_rets),
        'n_bear': len(bear_rets),
        'regime_gap': round(gap, 4),
        'pass': bool(gap <= CONFIG['regime_gap_max']),
    }
    log.info(f"  Regime: bull={s_bull:.3f} ({len(bull_rets)}m), bear={s_bear:.3f} ({len(bear_rets)}m), gap={gap:.3f}")
    return result


def subperiod_test(port_rets):
    """Sub-period stability."""
    log.info("\nRunning sub-period stability test...")
    n = len(port_rets)
    q = n // 4
    sharpes = []
    for i in range(4):
        start = i * q
        end = (i + 1) * q if i < 3 else n
        r = port_rets[start:end]
        s = r.mean() / r.std() * np.sqrt(12) if len(r) > 3 and r.std() > 0 else 0
        sharpes.append(round(s, 3))

    mean_s = np.mean(sharpes)
    std_s = np.std(sharpes)
    cv = std_s / abs(mean_s) if abs(mean_s) > 0.001 else float('inf')

    result = {
        'sharpes': sharpes,
        'cv': round(float(cv), 4),
        'pass': bool(cv <= CONFIG['subperiod_cv_max']),
    }
    log.info(f"  Sub-period: {sharpes}, CV={cv:.3f}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # Step 1: Data
    log.info("\n" + "=" * 70)
    log.info("STEP 1: DATA DOWNLOAD")
    log.info("=" * 70)
    data = download_data()

    # Step 2: Features
    log.info("\n" + "=" * 70)
    log.info("STEP 2: FEATURE ENGINEERING")
    log.info("=" * 70)
    features, returns, dates = build_features(data)

    # Step 3: DL walk-forward
    log.info("\n" + "=" * 70)
    log.info("STEP 3: DL WALK-FORWARD")
    log.info("=" * 70)
    dl_weights, dl_returns, dl_dates = run_walk_forward(features, returns, dates)

    # Portfolio returns
    dl_port_rets = (dl_weights * dl_returns).sum(axis=1)

    # Subtract transaction costs (turnover * cost_bps)
    cost_bps = CONFIG['cost_bps']
    for i in range(len(dl_port_rets)):
        if i > 0:
            turnover = np.abs(dl_weights[i] - dl_weights[i-1]).sum() / 2
        else:
            turnover = 1.0  # Initial buy
        cost = turnover * cost_bps / 10000 * 2
        dl_port_rets[i] -= cost

    # Step 4: Baselines
    log.info("\n" + "=" * 70)
    log.info("STEP 4: BASELINES")
    log.info("=" * 70)

    # Use same date range as DL
    train_periods = CONFIG['train_days'] // CONFIG['test_days']
    baseline_returns = returns[train_periods:]
    baseline_features = features[train_periods:]

    ew_rets = equal_weight_baseline(baseline_returns)
    mom_rets = momentum_baseline(baseline_features, baseline_returns, dates[train_periods:])

    # Step 5: Metrics
    log.info("\n" + "=" * 70)
    log.info("STEP 5: METRICS")
    log.info("=" * 70)

    dl_metrics = compute_metrics(dl_port_rets, 'DL Strategy Ensemble')
    ew_metrics = compute_metrics(ew_rets, 'Equal Weight Baseline')
    mom_metrics = compute_metrics(mom_rets, 'Momentum Baseline')

    # Step 6: Validation
    log.info("\n" + "=" * 70)
    log.info("STEP 6: VALIDATION GATES")
    log.info("=" * 70)

    perm_result = permutation_test(dl_port_rets, ew_rets, CONFIG['n_permutations'])
    regime_result = regime_test(dl_port_rets, dl_dates)
    subperiod_result = subperiod_test(dl_port_rets)

    gates = sum([perm_result['pass'], regime_result['pass'], subperiod_result['pass']])
    log.info(f"\n  Gates passed: {gates}/3")

    # Step 7: Save
    log.info("\n" + "=" * 70)
    log.info("STEP 7: SAVE RESULTS")
    log.info("=" * 70)

    runtime = (time.time() - t0) / 60

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'metrics': {
            'dl_ensemble': dl_metrics,
            'equal_weight': ew_metrics,
            'momentum_baseline': mom_metrics,
        },
        'validation': {
            'permutation_test': perm_result,
            'regime_test': regime_result,
            'subperiod_stability': subperiod_result,
            'gates_passed': f"{gates}/3",
        },
        'dl_beats_ew': dl_metrics.get('sharpe', 0) > ew_metrics.get('sharpe', 0),
        'dl_beats_momentum': dl_metrics.get('sharpe', 0) > mom_metrics.get('sharpe', 0),
        'runtime_minutes': round(runtime, 1),
    }

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save weights time series
    weight_df = pd.DataFrame(dl_weights, columns=TICKERS)
    weight_df['date'] = dl_dates
    weight_df['port_ret'] = dl_port_rets
    weight_df.to_csv(os.path.join(OUTPUT_DIR, 'weights_history.csv'), index=False)

    log.info(f"  Saved to {OUTPUT_DIR}")

    # Step 8: MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(CONFIG['mlflow_tracking_uri'])
        mlflow.set_experiment(CONFIG['mlflow_experiment'])

        with mlflow.start_run(run_name=f"strategy_ensemble_{datetime.now().strftime('%H%M')}"):
            for k, v in CONFIG.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, v)
            for k, v in dl_metrics.items():
                mlflow.log_metric(f"dl_{k}", v)
            for k, v in ew_metrics.items():
                mlflow.log_metric(f"ew_{k}", v)
            mlflow.log_metric("perm_p", perm_result['p_value'])
            mlflow.log_metric("regime_gap", regime_result['regime_gap'])
            mlflow.log_metric("gates", gates)
            mlflow.log_artifact(os.path.join(OUTPUT_DIR, 'results.json'))
            log.info("  MLflow logged")
    except Exception as e:
        log.warning(f"  MLflow failed: {e}")

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"  DL Ensemble: Sharpe={dl_metrics.get('sharpe')}, CAGR={dl_metrics.get('cagr')}, "
             f"MaxDD={dl_metrics.get('max_dd')}")
    log.info(f"  Equal Weight: Sharpe={ew_metrics.get('sharpe')}, CAGR={ew_metrics.get('cagr')}")
    log.info(f"  Momentum: Sharpe={mom_metrics.get('sharpe')}, CAGR={mom_metrics.get('cagr')}")
    log.info(f"  Beats EW: {results['dl_beats_ew']}, Beats Momentum: {results['dl_beats_momentum']}")
    log.info(f"  Validation: {gates}/3 gates")
    log.info(f"  Runtime: {runtime:.1f} min")
    log.info("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)
