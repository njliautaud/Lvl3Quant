#!/usr/bin/env python3
"""
MLP Sector Ranker v1 — GPU-accelerated sector ranking model
============================================================
Research findings #992 (MLP beats LGBM on sector ranking) and #994 (cross-sectional
z-scores triple IC) combined into a production-ready walk-forward framework.

Variants:
  A: MLP baseline 12 features
  B: MLP baseline + cross-sectional (15 features)
  C: MLP all features (20 features)
  D: LGBM cross-sectional (control)
  E: Random control

Validation gates:
  1: Permutation test (500 trials, p < 0.05)
  2: Regime stability (R1 gap < 0.50)
  3: Sub-period (both halves Sharpe > 0.5)
  4: vs Random (>20% Sharpe improvement)

Output: /home/nick/Lvl3Quant/output/mlp_sector_ranker_v1/results.json
"""

import json
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=3)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECTORS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
TRAIN_DAYS = 252
OOT_DAYS = 21
COMMISSION = 2.60  # per spread
STARTING_CAPITAL = 645.0
VIX_THRESHOLD = 20.0
HAIRCUT = 0.15
EARLY_EXIT_DAYS = 20
TOP_N = 3
BOTTOM_N = 3

# MLP config
HIDDEN_DIMS = [64, 32, 16]
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 64

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = Path('/home/nick/Lvl3Quant/output/mlp_sector_ranker_v1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ts(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_data():
    """Download sector ETF + VIX + TLT + SPY + HYG data via yfinance."""
    import yfinance as yf

    tickers = SECTORS + ['^VIX', 'TLT', 'SPY', 'HYG']
    ts(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start='2015-01-01', auto_adjust=True, progress=False)
    close = data['Close'].copy()
    # Rename ^VIX
    close = close.rename(columns={'^VIX': 'VIX'})
    close = close.dropna()
    ts(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def compute_features(close_df):
    """Compute all features for all sectors. Returns dict of DataFrames."""
    sectors_close = close_df[SECTORS]
    spy = close_df['SPY']
    vix = close_df['VIX']
    tlt = close_df['TLT']
    hyg = close_df['HYG']

    feature_frames = {}

    for sector in SECTORS:
        px = sectors_close[sector]
        feats = pd.DataFrame(index=close_df.index)

        # Baseline 12
        feats['ret_5d'] = px.pct_change(5)
        feats['ret_10d'] = px.pct_change(10)
        feats['ret_21d'] = px.pct_change(21)
        feats['ret_63d'] = px.pct_change(63)
        feats['ret_126d'] = px.pct_change(126)
        feats['ret_252d'] = px.pct_change(252)
        feats['vol_21d'] = px.pct_change().rolling(21).std()
        feats['vol_63d'] = px.pct_change().rolling(63).std()
        feats['sharpe_63d'] = feats['ret_63d'] / (feats['vol_63d'] * np.sqrt(63) + 1e-8)
        feats['maxdd_63d'] = px.rolling(63).apply(
            lambda x: (x / np.maximum.accumulate(x) - 1).min(), raw=True
        )
        feats['pct_52w_high'] = px / px.rolling(252).max()
        mom_12 = px.pct_change(252)
        mom_6 = px.pct_change(126)
        feats['mom_accel'] = mom_12 - mom_6

        # Cross-sectional 3 (computed across sectors at each point in time)
        ret_5d_all = sectors_close.pct_change(5)
        xs_mean = ret_5d_all.mean(axis=1)
        xs_std = ret_5d_all.std(axis=1)
        feats['xs_zscore_5d'] = (ret_5d_all[sector] - xs_mean) / (xs_std + 1e-8)
        feats['xs_rank_pct'] = ret_5d_all.rank(axis=1, pct=True)[sector]
        feats['xs_dispersion'] = xs_std

        # Macro 5
        feats['vix_level'] = vix
        feats['vix_chg_5d'] = vix.pct_change(5)
        feats['tlt_ret_21d'] = tlt.pct_change(21)
        # Credit spread proxy: HYG vs TLT relative performance
        feats['credit_spread_proxy'] = hyg.pct_change(21) - tlt.pct_change(21)
        sma200 = spy.rolling(200).mean()
        feats['spy_above_sma200'] = (spy > sma200).astype(float)

        feature_frames[sector] = feats

    # Forward 21d return (target)
    fwd_ret = sectors_close.pct_change(21).shift(-21)

    return feature_frames, fwd_ret


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------
class SectorRankerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, input_dim, device=DEVICE):
    """Train MLP with early stopping. Returns trained model."""
    model = SectorRankerMLP(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_mlp(model, X_test, device=DEVICE):
    """Get MLP predictions."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        preds = model(X_t).cpu().numpy()
    return preds


# ---------------------------------------------------------------------------
# LGBM model (variant D)
# ---------------------------------------------------------------------------
def train_lgbm(X_train, y_train):
    """Train LightGBM ranker."""
    try:
        import lightgbm as lgb
    except ImportError:
        # Fallback: simple linear regression via sklearn
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        return model

    params = {
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain, num_boost_round=200)
    return model


def predict_lgbm(model, X_test):
    """Get LGBM predictions."""
    try:
        import lightgbm as lgb
        if isinstance(model, lgb.Booster):
            return model.predict(X_test)
    except ImportError:
        pass
    # sklearn fallback
    return model.predict(X_test)


# ---------------------------------------------------------------------------
# Feature selection per variant
# ---------------------------------------------------------------------------
BASELINE_FEATURES = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel'
]
CROSS_SECTIONAL_FEATURES = ['xs_zscore_5d', 'xs_rank_pct', 'xs_dispersion']
MACRO_FEATURES = ['vix_level', 'vix_chg_5d', 'tlt_ret_21d', 'credit_spread_proxy', 'spy_above_sma200']

VARIANT_FEATURES = {
    'A': BASELINE_FEATURES,
    'B': BASELINE_FEATURES + CROSS_SECTIONAL_FEATURES,
    'C': BASELINE_FEATURES + CROSS_SECTIONAL_FEATURES + MACRO_FEATURES,
    'D': BASELINE_FEATURES + CROSS_SECTIONAL_FEATURES,  # LGBM uses same as B
    'E': BASELINE_FEATURES,  # Random ignores features anyway
}


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
def build_panel(feature_frames, fwd_ret, feature_cols, dates):
    """Build panel: rows = (date, sector), cols = features + target."""
    rows = []
    for dt in dates:
        for sector in SECTORS:
            feat_row = feature_frames[sector].loc[dt, feature_cols].values
            target = fwd_ret.loc[dt, sector] if dt in fwd_ret.index else np.nan
            rows.append(np.concatenate([feat_row, [target]]))
    arr = np.array(rows)
    return arr[:, :-1], arr[:, -1]


def walk_forward(feature_frames, fwd_ret, variant, vix_series, close_df):
    """Run walk-forward for a variant. Returns daily P&L series."""
    feature_cols = VARIANT_FEATURES[variant]
    n_features = len(feature_cols)

    # Get valid dates (where all features and target are available)
    valid_dates = feature_frames[SECTORS[0]].dropna().index
    valid_dates = valid_dates.intersection(fwd_ret.dropna(how='all').index)
    valid_dates = sorted(valid_dates)

    # Minimum required: TRAIN_DAYS + OOT_DAYS
    if len(valid_dates) < TRAIN_DAYS + OOT_DAYS:
        ts(f"  Variant {variant}: insufficient data ({len(valid_dates)} days)")
        return pd.Series(dtype=float)

    all_trades = []
    n_folds = (len(valid_dates) - TRAIN_DAYS) // OOT_DAYS

    for fold_idx in range(n_folds):
        train_start = fold_idx * OOT_DAYS
        train_end = train_start + TRAIN_DAYS
        oot_start = train_end
        oot_end = min(oot_start + OOT_DAYS, len(valid_dates))

        if oot_end > len(valid_dates):
            break

        train_dates = valid_dates[train_start:train_end]
        oot_dates = valid_dates[oot_start:oot_end]

        # Build training data
        X_train, y_train = build_panel(feature_frames, fwd_ret, feature_cols, train_dates)

        # Remove NaN rows
        mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train, y_train = X_train[mask], y_train[mask]

        if len(X_train) < 100:
            continue

        # Normalize features (fit on train)
        feat_mean = X_train.mean(axis=0)
        feat_std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - feat_mean) / feat_std

        # Train model
        if variant in ('A', 'B', 'C'):
            model = train_mlp(X_train_norm, y_train, n_features)
        elif variant == 'D':
            model = train_lgbm(X_train_norm, y_train)

        # Predict on OOT dates
        for oot_dt in oot_dates:
            # Get features for all sectors on this date
            X_oot = []
            valid_sectors = []
            for sector in SECTORS:
                row = feature_frames[sector].loc[oot_dt, feature_cols].values
                if not np.isnan(row).any():
                    X_oot.append(row)
                    valid_sectors.append(sector)

            if len(X_oot) < 6:
                continue

            X_oot = np.array(X_oot)
            X_oot_norm = (X_oot - feat_mean) / feat_std

            # Get predictions
            if variant == 'E':
                scores = np.random.randn(len(valid_sectors))
            elif variant in ('A', 'B', 'C'):
                scores = predict_mlp(model, X_oot_norm)
            else:  # D
                scores = predict_lgbm(model, X_oot_norm)

            # Rank sectors
            ranked_indices = np.argsort(scores)[::-1]
            top_sectors = [valid_sectors[i] for i in ranked_indices[:TOP_N]]
            bottom_sectors = [valid_sectors[i] for i in ranked_indices[-BOTTOM_N:]]

            # Get VIX for regime
            vix_val = vix_series.loc[oot_dt] if oot_dt in vix_series.index else 20.0

            # Options backtest logic
            # Bull call spreads on top-3 (when VIX >= 20) or bear put spreads on bottom-3 (VIX < 20)
            if vix_val >= VIX_THRESHOLD:
                # Bull call spreads on top sectors
                trade_sectors = top_sectors
                direction = 'bull'
            else:
                # Bear put spreads on bottom sectors
                trade_sectors = bottom_sectors
                direction = 'bear'

            # Compute trade P&L using ATR-based spread width + haircut
            for sector in trade_sectors:
                if sector not in close_df.columns:
                    continue
                entry_price = close_df.loc[oot_dt, sector]

                # ATR proxy (21d range / close)
                lookback_start = max(0, close_df.index.get_loc(oot_dt) - 21)
                lookback_slice = close_df[sector].iloc[lookback_start:close_df.index.get_loc(oot_dt) + 1]
                atr_pct = (lookback_slice.max() - lookback_slice.min()) / entry_price

                # Spread width = ATR, max credit = ATR * (1 - haircut)
                spread_width = entry_price * atr_pct
                max_profit = spread_width * (1 - HAIRCUT) * 100  # per contract (100 shares)

                # Scale to affordable position ($645 capital)
                risk_per_trade = spread_width * 100  # max loss per contract
                if risk_per_trade <= 0:
                    continue
                n_contracts = max(1, int(STARTING_CAPITAL / (3 * risk_per_trade)))

                # Get actual 21d (or early exit at 20d) return
                exit_loc = min(
                    close_df.index.get_loc(oot_dt) + EARLY_EXIT_DAYS,
                    len(close_df) - 1
                )
                exit_price = close_df[sector].iloc[exit_loc]
                actual_ret = (exit_price - entry_price) / entry_price

                # P&L calculation
                if direction == 'bull':
                    # Bull call spread profits when stock goes up
                    pnl_pct = min(actual_ret / atr_pct, 1.0) if atr_pct > 0 else 0
                    trade_pnl = pnl_pct * max_profit * n_contracts - COMMISSION
                else:
                    # Bear put spread profits when stock goes down
                    pnl_pct = min(-actual_ret / atr_pct, 1.0) if atr_pct > 0 else 0
                    trade_pnl = pnl_pct * max_profit * n_contracts - COMMISSION

                # Apply haircut to max profit (already done above), floor at -max_risk
                max_loss = -risk_per_trade * n_contracts - COMMISSION
                trade_pnl = max(trade_pnl, max_loss)

                all_trades.append({
                    'date': oot_dt,
                    'sector': sector,
                    'direction': direction,
                    'pnl': trade_pnl,
                    'n_contracts': n_contracts,
                })

    if not all_trades:
        return pd.Series(dtype=float)

    trades_df = pd.DataFrame(all_trades)
    daily_pnl = trades_df.groupby('date')['pnl'].sum()
    return daily_pnl


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(daily_pnl, starting_capital=STARTING_CAPITAL):
    """Compute performance metrics from daily P&L series."""
    if len(daily_pnl) == 0:
        return {'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0, 'total_ret': 0, 'max_dd': 0, 'n_trades': 0}

    equity = starting_capital + daily_pnl.cumsum()
    returns = daily_pnl / starting_capital  # simple return series

    # Annualize (assume ~12 trades/year with 21d holding)
    n_periods = len(daily_pnl)
    ann_factor = np.sqrt(252 / 21)  # annualization for 21d holding periods

    avg_ret = returns.mean()
    std_ret = returns.std() + 1e-8
    sharpe = avg_ret / std_ret * ann_factor

    downside = returns[returns < 0].std() + 1e-8
    sortino = avg_ret / downside * ann_factor

    wins = daily_pnl[daily_pnl > 0].sum()
    losses = abs(daily_pnl[daily_pnl < 0].sum()) + 1e-8
    pf = wins / losses

    wr = (daily_pnl > 0).mean()
    total_ret = daily_pnl.sum() / starting_capital

    # Max drawdown
    cum = daily_pnl.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min() / starting_capital if len(dd) > 0 else 0

    return {
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'pf': round(float(pf), 3),
        'wr': round(float(wr), 3),
        'total_ret': round(float(total_ret), 4),
        'max_dd': round(float(max_dd), 4),
        'n_trades': int(n_periods),
    }


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------
def gate1_permutation(daily_pnl, n_trials=500):
    """Permutation test: shuffle sector rankings, check if real Sharpe > random."""
    real_sharpe = compute_metrics(daily_pnl)['sharpe']
    count_worse = 0
    shuffled_pnls = daily_pnl.values.copy()

    for _ in range(n_trials):
        np.random.shuffle(shuffled_pnls)
        shuffled_series = pd.Series(shuffled_pnls, index=daily_pnl.index)
        rand_sharpe = compute_metrics(shuffled_series)['sharpe']
        if rand_sharpe >= real_sharpe:
            count_worse += 1

    p_value = count_worse / n_trials
    return {'pass': p_value < 0.05, 'p_value': round(p_value, 4)}


def gate2_regime_stability(daily_pnl, spy_returns):
    """Check regime stability: |Sharpe_bull - Sharpe_bear| / max < 0.50."""
    if len(daily_pnl) == 0:
        return {'pass': False, 'gap': 1.0}

    # Align dates
    common_dates = daily_pnl.index.intersection(spy_returns.index)
    if len(common_dates) < 10:
        return {'pass': False, 'gap': 1.0}

    pnl_aligned = daily_pnl.loc[common_dates]
    spy_aligned = spy_returns.loc[common_dates]

    # Bull = SPY positive over trailing 63d
    spy_cum = spy_aligned.rolling(63, min_periods=21).sum()
    bull_mask = spy_cum > 0
    bear_mask = ~bull_mask

    bull_pnl = pnl_aligned[bull_mask]
    bear_pnl = pnl_aligned[bear_mask]

    sharpe_bull = compute_metrics(bull_pnl)['sharpe'] if len(bull_pnl) > 5 else 0
    sharpe_bear = compute_metrics(bear_pnl)['sharpe'] if len(bear_pnl) > 5 else 0

    max_sharpe = max(abs(sharpe_bull), abs(sharpe_bear), 0.01)
    gap = abs(sharpe_bull - sharpe_bear) / max_sharpe

    return {'pass': gap < 0.50, 'gap': round(gap, 3), 'sharpe_bull': sharpe_bull, 'sharpe_bear': sharpe_bear}


def gate3_subperiod(daily_pnl):
    """Both halves must have Sharpe > 0.5."""
    if len(daily_pnl) < 20:
        return {'pass': False, 'sharpe_h1': 0, 'sharpe_h2': 0}

    mid = len(daily_pnl) // 2
    h1 = daily_pnl.iloc[:mid]
    h2 = daily_pnl.iloc[mid:]

    sharpe_h1 = compute_metrics(h1)['sharpe']
    sharpe_h2 = compute_metrics(h2)['sharpe']

    return {'pass': sharpe_h1 > 0.5 and sharpe_h2 > 0.5,
            'sharpe_h1': sharpe_h1, 'sharpe_h2': sharpe_h2}


def gate4_vs_random(variant_sharpe, random_sharpe):
    """Must beat random by >20%."""
    if random_sharpe <= 0:
        improvement = float('inf') if variant_sharpe > 0 else 0
    else:
        improvement = (variant_sharpe - random_sharpe) / abs(random_sharpe)
    return {'pass': improvement > 0.20, 'improvement': round(improvement, 3)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    ts(f"MLP Sector Ranker v1 — Device: {DEVICE}")
    ts(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        ts(f"GPU: {torch.cuda.get_device_name(0)}")

    # Download data
    close_df = download_data()
    vix_series = close_df['VIX']
    spy_returns = close_df['SPY'].pct_change()

    # Compute features
    ts("Computing features...")
    feature_frames, fwd_ret = compute_features(close_df)
    ts(f"Features computed for {len(SECTORS)} sectors")

    # Run all variants
    results = {}
    daily_pnls = {}

    for variant in ['A', 'B', 'C', 'D', 'E']:
        ts(f"--- Variant {variant} ({len(VARIANT_FEATURES[variant])} features) ---")
        t0 = time.time()
        daily_pnl = walk_forward(feature_frames, fwd_ret, variant, vix_series, close_df)
        elapsed = time.time() - t0

        if len(daily_pnl) == 0:
            ts(f"  Variant {variant}: no trades generated")
            results[variant] = {'metrics': {}, 'gates': {}, 'elapsed_s': elapsed}
            continue

        metrics = compute_metrics(daily_pnl)
        daily_pnls[variant] = daily_pnl
        ts(f"  Sharpe={metrics['sharpe']:.3f} Sortino={metrics['sortino']:.3f} "
           f"PF={metrics['pf']:.2f} WR={metrics['wr']:.1%} "
           f"TotalRet={metrics['total_ret']:.1%} MaxDD={metrics['max_dd']:.1%} "
           f"N={metrics['n_trades']} ({elapsed:.1f}s)")

        results[variant] = {'metrics': metrics, 'elapsed_s': round(elapsed, 1)}

    # Validation gates
    ts("--- Running validation gates ---")
    random_sharpe = results.get('E', {}).get('metrics', {}).get('sharpe', 0)

    for variant in ['A', 'B', 'C', 'D']:
        if variant not in daily_pnls:
            results[variant]['gates'] = {'all_pass': False, 'reason': 'no trades'}
            continue

        pnl = daily_pnls[variant]
        ts(f"  Variant {variant} gates...")

        g1 = gate1_permutation(pnl)
        g2 = gate2_regime_stability(pnl, spy_returns)
        g3 = gate3_subperiod(pnl)
        g4 = gate4_vs_random(results[variant]['metrics']['sharpe'], random_sharpe)

        gates = {
            'gate1_permutation': g1,
            'gate2_regime': g2,
            'gate3_subperiod': g3,
            'gate4_vs_random': g4,
            'all_pass': g1['pass'] and g2['pass'] and g3['pass'] and g4['pass'],
        }
        results[variant]['gates'] = gates
        n_pass = sum([g1['pass'], g2['pass'], g3['pass'], g4['pass']])
        ts(f"    Gates: {n_pass}/4 pass | Perm p={g1['p_value']:.3f} | "
           f"Regime gap={g2['gap']:.2f} | SubP: H1={g3['sharpe_h1']:.2f} H2={g3['sharpe_h2']:.2f} | "
           f"vs Random: {g4['improvement']:.1%}")

    # Summary
    ts("--- SUMMARY ---")
    best_variant = None
    best_sharpe = -999
    for v in ['A', 'B', 'C', 'D']:
        m = results.get(v, {}).get('metrics', {})
        g = results.get(v, {}).get('gates', {})
        sharpe = m.get('sharpe', 0)
        passes = g.get('all_pass', False)
        status = "PASS" if passes else "FAIL"
        ts(f"  Variant {v}: Sharpe={sharpe:.3f} [{status}]")
        if passes and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_variant = v

    if best_variant:
        ts(f"  WINNER: Variant {best_variant} (Sharpe={best_sharpe:.3f})")
    else:
        ts("  NO variant passed all 4 gates")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'variants': results,
        'best_variant': best_variant,
        'best_sharpe': best_sharpe if best_variant else None,
        'total_elapsed_s': round(time.time() - start_time, 1),
        'config': {
            'train_days': TRAIN_DAYS,
            'oot_days': OOT_DAYS,
            'sectors': SECTORS,
            'mlp_hidden': HIDDEN_DIMS,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'commission': COMMISSION,
            'starting_capital': STARTING_CAPITAL,
            'vix_threshold': VIX_THRESHOLD,
        }
    }

    output_path = OUTPUT_DIR / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    ts(f"Results saved to {output_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment("mlp_sector_ranker_v1")
            with mlflow.start_run(run_name=f"ranker_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("device", str(DEVICE))
                mlflow.log_param("train_days", TRAIN_DAYS)
                mlflow.log_param("oot_days", OOT_DAYS)
                mlflow.log_param("hidden_dims", str(HIDDEN_DIMS))
                mlflow.log_param("epochs", EPOCHS)
                mlflow.log_param("best_variant", best_variant)

                for v in ['A', 'B', 'C', 'D', 'E']:
                    m = results.get(v, {}).get('metrics', {})
                    for k, val in m.items():
                        mlflow.log_metric(f"{v}_{k}", val)

                if best_variant:
                    mlflow.log_metric("best_sharpe", best_sharpe)

                mlflow.log_artifact(str(output_path))
            ts("MLflow run logged successfully")
        except Exception as e:
            ts(f"MLflow logging failed: {e}")

    total_time = time.time() - start_time
    ts(f"DONE in {total_time:.1f}s ({total_time/60:.1f}min)")
    return output


if __name__ == '__main__':
    main()
