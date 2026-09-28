#!/usr/bin/env python3
"""
ML Volatility-Timed Premium Selling v1
=======================================
Predicts optimal days to sell options premium (CSP on SPY).

HC #724 compliant: All features use T-1 data only. Walk-forward sliding window.
HC #718 compliant: train_end = test_start - LABEL_HORIZON gap. Permutation tests shuffle signals.
Transaction costs: 5 bps per trade.

Experiment runs TWO passes:
  Pass 1: T-0 features (lookahead baseline - should be better)
  Pass 2: T-1 features (proper anti-lookahead)
If T-0 >> T-1, flags lookahead bias.

Architecture: 3-layer MLP with dropout + batch norm.
Walk-forward: 252d train, 21d test, 5d gap (label horizon).
"""

import os
import sys
import time
import warnings
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
TRAIN_WINDOW = 252       # Trading days for training
TEST_WINDOW = 21         # Trading days for testing
LABEL_HORIZON = 5        # Forward 5-day return for CSP simulation
GAP_DAYS = LABEL_HORIZON # Gap between train end and test start (HC #718)
TRANSACTION_COST_BPS = 5 # 5 bps per trade
N_PERMUTATIONS = 100     # Permutation test iterations
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
DROPOUT = 0.3
HIDDEN_DIM = 128
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"[{datetime.datetime.now()}] ML Vol-Timed Income v1 starting on {DEVICE}")
print(f"Config: train={TRAIN_WINDOW}d, test={TEST_WINDOW}d, gap={GAP_DAYS}d, "
      f"label_horizon={LABEL_HORIZON}d, cost={TRANSACTION_COST_BPS}bps")
print("=" * 80)

# ─── Data Download ────────────────────────────────────────────────────────────
import yfinance as yf

print("\n[1/7] Downloading market data...")
tickers = {
    'SPY': 'SPY',
    'VIX': '^VIX',
    'VIX3M': '^VIX3M',
    'HYG': 'HYG',
    'TLT': 'TLT',
}

# For market breadth: use equal-weight ETF as proxy
breadth_tickers = ['RSP']  # Equal-weight S&P 500

start_date = '2010-01-01'
end_date = datetime.date.today().strftime('%Y-%m-%d')

data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[name] = df
        print(f"  {name} ({ticker}): {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    except Exception as e:
        print(f"  WARNING: Failed to download {name}: {e}")

for t in breadth_tickers:
    try:
        df = yf.download(t, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[t] = df
        print(f"  {t}: {len(df)} rows")
    except Exception as e:
        print(f"  WARNING: Failed to download {t}: {e}")

# ─── Feature Engineering ──────────────────────────────────────────────────────
print("\n[2/7] Engineering features...")

spy = data['SPY'][['Close', 'Volume']].copy()
spy.columns = ['spy_close', 'spy_volume']

# Merge all data on SPY's index
features_df = spy.copy()

# VIX level
if 'VIX' in data:
    vix = data['VIX'][['Close']].rename(columns={'Close': 'vix'})
    features_df = features_df.join(vix, how='left')
    features_df['vix'] = features_df['vix'].ffill()

# VIX term structure (VIX/VIX3M ratio)
if 'VIX3M' in data:
    vix3m = data['VIX3M'][['Close']].rename(columns={'Close': 'vix3m'})
    features_df = features_df.join(vix3m, how='left')
    features_df['vix3m'] = features_df['vix3m'].ffill()
    features_df['vix_term_structure'] = features_df['vix'] / features_df['vix3m'].clip(lower=1.0)
else:
    features_df['vix_term_structure'] = 1.0

# IV Rank: percentile of current VIX vs trailing 252d
features_df['iv_rank'] = features_df['vix'].rolling(252).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8) if len(x) == 252 else np.nan,
    raw=False
)

# Realized vol (20d) vs implied vol spread
features_df['spy_ret'] = features_df['spy_close'].pct_change()
features_df['realized_vol_20d'] = features_df['spy_ret'].rolling(20).std() * np.sqrt(252) * 100
features_df['rv_iv_spread'] = features_df['realized_vol_20d'] - features_df['vix']

# RSI(14)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

features_df['rsi_14'] = compute_rsi(features_df['spy_close'], 14)

# Days since last >2% drawdown
features_df['big_drop'] = (features_df['spy_ret'] < -0.02).astype(int)
features_df['days_since_big_drop'] = 0
counter = 0
days_since = []
for drop in features_df['big_drop'].values:
    if drop == 1:
        counter = 0
    else:
        counter += 1
    days_since.append(counter)
features_df['days_since_big_drop'] = days_since

# 20d volume trend (current 5d avg / 20d avg)
features_df['vol_trend_20d'] = (
    features_df['spy_volume'].rolling(5).mean() /
    features_df['spy_volume'].rolling(20).mean().clip(lower=1)
)

# Credit spread proxy: HYG - TLT total return diff (yield spread proxy)
if 'HYG' in data and 'TLT' in data:
    hyg = data['HYG'][['Close']].rename(columns={'Close': 'hyg_close'})
    tlt = data['TLT'][['Close']].rename(columns={'Close': 'tlt_close'})
    features_df = features_df.join(hyg, how='left').join(tlt, how='left')
    features_df['hyg_close'] = features_df['hyg_close'].ffill()
    features_df['tlt_close'] = features_df['tlt_close'].ffill()
    features_df['hyg_ret_20d'] = features_df['hyg_close'].pct_change(20)
    features_df['tlt_ret_20d'] = features_df['tlt_close'].pct_change(20)
    features_df['credit_spread_proxy'] = features_df['hyg_ret_20d'] - features_df['tlt_ret_20d']
else:
    features_df['credit_spread_proxy'] = 0.0

# Market breadth proxy: RSP/SPY ratio (equal-weight vs cap-weight)
if 'RSP' in data:
    rsp = data['RSP'][['Close']].rename(columns={'Close': 'rsp_close'})
    features_df = features_df.join(rsp, how='left')
    features_df['rsp_close'] = features_df['rsp_close'].ffill()
    features_df['breadth_ratio'] = features_df['rsp_close'] / features_df['spy_close'].clip(lower=1)
    features_df['breadth_ratio_chg_20d'] = features_df['breadth_ratio'].pct_change(20)
else:
    features_df['breadth_ratio_chg_20d'] = 0.0

# Put/call ratio proxy: VIX change momentum (no direct P/C data in yfinance)
features_df['vix_momentum_5d'] = features_df['vix'].pct_change(5)
features_df['vix_momentum_10d'] = features_df['vix'].pct_change(10)

# ─── Label Construction (CSP Simulation) ──────────────────────────────────────
print("\n[3/7] Constructing CSP return labels...")

# CSP return proxy: Selling ATM put on SPY with ~5 DTE
# Premium earned ≈ f(IV, DTE). Assignment loss = max(0, strike - close_at_expiry).
# Simplified: premium ≈ SPY_price * IV * sqrt(DTE/252) * adjustment_factor
# For ATM put with 5 DTE: premium ≈ SPY * (VIX/100) * sqrt(5/252) * 0.4 (BS approx for ATM)
# P&L = premium_collected - max(0, entry_price - exit_price)

features_df['spy_fwd_5d_ret'] = features_df['spy_close'].pct_change(LABEL_HORIZON).shift(-LABEL_HORIZON)

# ATM put premium estimate (Black-Scholes approximation for ATM)
# For ATM: premium ≈ S * sigma * sqrt(T) * 0.4 (where 0.4 ≈ N'(0)/sqrt(2*pi) * 2)
T_years = LABEL_HORIZON / 252.0
features_df['estimated_premium_pct'] = (features_df['vix'] / 100.0) * np.sqrt(T_years) * 0.4

# CSP P&L: collect premium, lose if SPY drops below strike (ATM = current price)
# P&L = premium - max(0, -fwd_return) * 100% (put assignment loss)
features_df['csp_pnl_pct'] = features_df['estimated_premium_pct'] - np.maximum(0, -features_df['spy_fwd_5d_ret'])

# Binary label: 1 = good day to sell premium (positive CSP P&L after costs)
cost_pct = TRANSACTION_COST_BPS / 10000.0
features_df['csp_pnl_net'] = features_df['csp_pnl_pct'] - cost_pct

# Also keep continuous label for regression
features_df['label_continuous'] = features_df['csp_pnl_net']
features_df['label_binary'] = (features_df['csp_pnl_net'] > 0).astype(float)

# ─── Feature Selection ────────────────────────────────────────────────────────
FEATURE_COLS = [
    'vix', 'vix_term_structure', 'iv_rank', 'realized_vol_20d', 'rv_iv_spread',
    'rsi_14', 'days_since_big_drop', 'vol_trend_20d', 'credit_spread_proxy',
    'breadth_ratio_chg_20d', 'vix_momentum_5d', 'vix_momentum_10d',
]

# Drop rows with NaNs in features or label
subset_cols = FEATURE_COLS + ['label_continuous', 'label_binary', 'csp_pnl_net',
                               'spy_close', 'spy_fwd_5d_ret', 'estimated_premium_pct']
features_df = features_df.dropna(subset=subset_cols)

print(f"  Dataset: {len(features_df)} rows from {features_df.index[0].date()} to {features_df.index[-1].date()}")
print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Baseline CSP win rate: {(features_df['label_binary'] > 0).mean():.1%}")
print(f"  Baseline avg CSP P&L: {features_df['csp_pnl_net'].mean()*100:.3f}% per trade")


# ─── Model Definition ─────────────────────────────────────────────────────────
class PremiumTimingMLP(nn.Module):
    """3-layer MLP with dropout and batch norm for premium selling timing."""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─── Walk-Forward Engine ──────────────────────────────────────────────────────
def run_walk_forward(features_df, feature_cols, lag=1, label='T-1'):
    """
    Run walk-forward with sliding window.
    lag=0: T-0 features (lookahead baseline)
    lag=1: T-1 features (proper anti-lookahead, HC #724)
    """
    print(f"\n{'='*80}")
    print(f"[Pass] {label} features (lag={lag})")
    print(f"{'='*80}")

    df = features_df.copy()

    # Apply lag to features: shift features by `lag` days
    # lag=1 means features from T-1 are used to predict T's label
    if lag > 0:
        for col in feature_cols:
            df[col] = df[col].shift(lag)
        df = df.dropna(subset=feature_cols)

    dates = df.index.values
    n = len(df)

    # Walk-forward parameters
    min_start = TRAIN_WINDOW + GAP_DAYS  # Need enough data for first train + gap
    all_predictions = []
    all_actuals = []
    all_dates = []
    all_spy_close = []
    all_spy_fwd_ret = []
    all_premium_pct = []

    fold_count = 0
    fold_metrics = []

    i = min_start
    while i + TEST_WINDOW <= n:
        # Define windows
        train_end = i - GAP_DAYS  # HC #718: gap between train and test
        train_start = max(0, train_end - TRAIN_WINDOW)

        if train_end - train_start < 100:
            i += TEST_WINDOW
            continue

        test_start = i
        test_end = min(i + TEST_WINDOW, n)

        # Extract data
        X_train = df[feature_cols].iloc[train_start:train_end].values.astype(np.float32)
        y_train = df['label_continuous'].iloc[train_start:train_end].values.astype(np.float32)

        X_test = df[feature_cols].iloc[test_start:test_end].values.astype(np.float32)
        y_test = df['label_continuous'].iloc[test_start:test_end].values.astype(np.float32)

        test_dates = df.index[test_start:test_end]
        test_spy_close = df['spy_close'].iloc[test_start:test_end].values
        test_fwd_ret = df['spy_fwd_5d_ret'].iloc[test_start:test_end].values
        test_premium = df['estimated_premium_pct'].iloc[test_start:test_end].values

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Convert to tensors
        X_tr = torch.FloatTensor(X_train_s).to(DEVICE)
        y_tr = torch.FloatTensor(y_train).to(DEVICE)
        X_te = torch.FloatTensor(X_test_s).to(DEVICE)

        train_ds = TensorDataset(X_tr, y_tr)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        # Train model
        model = PremiumTimingMLP(len(feature_cols), HIDDEN_DIM, DROPOUT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        model.train()
        for epoch in range(EPOCHS):
            for xb, yb in train_dl:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Predict
        model.eval()
        with torch.no_grad():
            preds = model(X_te).cpu().numpy()

        all_predictions.extend(preds)
        all_actuals.extend(y_test)
        all_dates.extend(test_dates)
        all_spy_close.extend(test_spy_close)
        all_spy_fwd_ret.extend(test_fwd_ret)
        all_premium_pct.extend(test_premium)

        fold_count += 1
        i += TEST_WINDOW

    if fold_count == 0:
        print("  ERROR: No folds completed!")
        return None

    print(f"  Completed {fold_count} walk-forward folds")

    # ─── Strategy Evaluation ──────────────────────────────────────────────
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)
    dates_arr = np.array(all_dates)
    spy_close_arr = np.array(all_spy_close)
    fwd_ret_arr = np.array(all_spy_fwd_ret)
    premium_arr = np.array(all_premium_pct)

    results_df = pd.DataFrame({
        'date': dates_arr,
        'prediction': predictions,
        'actual_pnl': actuals,
        'spy_close': spy_close_arr,
        'spy_fwd_ret': fwd_ret_arr,
        'premium_pct': premium_arr,
    }).set_index('date')

    # Strategy: trade only when model predicts positive P&L (prediction > threshold)
    # Use median prediction as adaptive threshold
    threshold = np.median(predictions)

    results_df['signal'] = (results_df['prediction'] > threshold).astype(int)
    results_df['strategy_pnl'] = results_df['signal'] * results_df['actual_pnl']

    # Apply transaction costs only on trade days
    results_df['strategy_pnl_net'] = results_df['strategy_pnl'] - (results_df['signal'] * cost_pct)

    # Baseline: sell premium every day
    results_df['baseline_pnl'] = results_df['actual_pnl']

    # ─── Compute Metrics ──────────────────────────────────────────────────
    def compute_metrics(returns, name):
        """Compute risk-adjusted metrics for a return series."""
        if len(returns) == 0 or returns.std() == 0:
            return {}
        ann_factor = np.sqrt(252 / LABEL_HORIZON)  # Annualize based on trade frequency
        mean_ret = returns.mean()
        std_ret = returns.std()

        sharpe = mean_ret / std_ret * ann_factor if std_ret > 0 else 0
        downside = returns[returns < 0].std()
        sortino = mean_ret / downside * ann_factor if downside > 0 else 0

        # Cumulative returns
        cum_ret = (1 + returns).cumprod()
        total_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
        n_years = len(returns) * LABEL_HORIZON / 252.0
        cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

        # Max drawdown
        peak = cum_ret.expanding().max()
        dd = (cum_ret - peak) / peak
        max_dd = dd.min()

        # Win rate and profit factor
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        wr = len(wins) / len(returns) if len(returns) > 0 else 0
        pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float('inf')

        n_trades = (returns != 0).sum()

        return {
            'name': name,
            'sharpe': sharpe,
            'sortino': sortino,
            'cagr': cagr * 100,
            'max_dd': max_dd * 100,
            'wr': wr * 100,
            'pf': pf,
            'n_trades': n_trades,
            'avg_pnl_bps': mean_ret * 10000,
            'total_ret_pct': total_ret * 100,
        }

    strat_rets = results_df['strategy_pnl_net']
    strat_rets_trade_only = strat_rets[results_df['signal'] == 1]
    base_rets = results_df['baseline_pnl']

    m_strat = compute_metrics(strat_rets_trade_only, f'{label} ML Strategy')
    m_base = compute_metrics(base_rets, 'Baseline (sell daily)')

    print(f"\n  --- {label} RESULTS ---")
    for m in [m_strat, m_base]:
        if m:
            print(f"  {m['name']:30s} | Sharpe: {m['sharpe']:6.2f} | Sortino: {m['sortino']:6.2f} | "
                  f"CAGR: {m['cagr']:6.1f}% | MaxDD: {m['max_dd']:6.1f}% | "
                  f"WR: {m['wr']:5.1f}% | PF: {m['pf']:5.2f} | Trades: {m['n_trades']}")

    # ─── Regime-Agnostic Test ─────────────────────────────────────────────
    print(f"\n  --- REGIME-AGNOSTIC TEST ({label}) ---")
    results_df['spy_daily_ret'] = results_df['spy_close'].pct_change()

    # Classify days by rolling 20d SPY return
    results_df['spy_20d_ret'] = results_df['spy_close'].pct_change(20)
    results_df['regime'] = 'flat'
    results_df.loc[results_df['spy_20d_ret'] > 0.02, 'regime'] = 'green'
    results_df.loc[results_df['spy_20d_ret'] < -0.02, 'regime'] = 'red'

    regime_sharpes = {}
    for regime in ['green', 'red', 'flat']:
        mask = (results_df['regime'] == regime) & (results_df['signal'] == 1)
        regime_rets = results_df.loc[mask, 'strategy_pnl_net']
        if len(regime_rets) > 10:
            ann_factor = np.sqrt(252 / LABEL_HORIZON)
            s = regime_rets.mean() / regime_rets.std() * ann_factor if regime_rets.std() > 0 else 0
            regime_sharpes[regime] = s
            print(f"    {regime:5s} regime: Sharpe={s:.2f}, trades={len(regime_rets)}, "
                  f"WR={100*(regime_rets>0).mean():.1f}%")
        else:
            print(f"    {regime:5s} regime: insufficient data ({len(regime_rets)} trades)")

    if 'green' in regime_sharpes and 'red' in regime_sharpes:
        sg, sr = regime_sharpes['green'], regime_sharpes['red']
        ratio = abs(sg - sr) / max(abs(sg), abs(sr), 1e-8)
        status = "PASS" if ratio < 0.50 else "FAIL"
        print(f"    Regime asymmetry: {ratio:.2f} ({status}, threshold=0.50)")
    else:
        print(f"    Regime asymmetry: insufficient data to compute")

    return {
        'metrics_strategy': m_strat,
        'metrics_baseline': m_base,
        'regime_sharpes': regime_sharpes,
        'predictions': predictions,
        'actuals': actuals,
        'results_df': results_df,
    }


# ─── Run Both Passes ─────────────────────────────────────────────────────────
print("\n[4/7] Running walk-forward pass 1: T-0 features (lookahead baseline)...")
t0_start = time.time()
results_t0 = run_walk_forward(features_df, FEATURE_COLS, lag=0, label='T-0 (lookahead)')
t0_elapsed = time.time() - t0_start
print(f"  T-0 pass completed in {t0_elapsed:.0f}s")

print("\n[5/7] Running walk-forward pass 2: T-1 features (proper, HC #724)...")
t1_start = time.time()
results_t1 = run_walk_forward(features_df, FEATURE_COLS, lag=1, label='T-1 (proper)')
t1_elapsed = time.time() - t1_start
print(f"  T-1 pass completed in {t1_elapsed:.0f}s")

# ─── Lag Sensitivity Test ─────────────────────────────────────────────────────
print("\n[6/7] LAG SENSITIVITY TEST (T-0 vs T-1)")
print("=" * 80)
if results_t0 and results_t1:
    s0 = results_t0['metrics_strategy']['sharpe']
    s1 = results_t1['metrics_strategy']['sharpe']
    print(f"  T-0 Sharpe: {s0:.3f}")
    print(f"  T-1 Sharpe: {s1:.3f}")
    if abs(s0) > 0:
        degradation = (s0 - s1) / abs(s0) * 100
        print(f"  Degradation: {degradation:.1f}%")
        if degradation > 50:
            print(f"  *** WARNING: T-0 >> T-1 ({degradation:.0f}% degradation). "
                  f"Likely LOOKAHEAD BIAS in features! ***")
        elif degradation > 25:
            print(f"  CAUTION: Moderate degradation ({degradation:.0f}%). "
                  f"Some features may have lookahead component.")
        else:
            print(f"  OK: Modest degradation ({degradation:.0f}%). "
                  f"T-1 features appear to carry genuine signal.")
    else:
        print(f"  T-0 Sharpe is zero/negative - no lookahead bias concern.")

# ─── Permutation Test (on T-1 results) ───────────────────────────────────────
print(f"\n[7/7] PERMUTATION TEST ({N_PERMUTATIONS} iterations, T-1 signal)...")
print("=" * 80)
if results_t1 is not None:
    actual_sharpe = results_t1['metrics_strategy']['sharpe']
    rdf = results_t1['results_df'].copy()

    perm_sharpes = []
    signal_mask = rdf['signal'].values.copy()
    actual_pnls = rdf['strategy_pnl_net'].values.copy()
    trade_pnls = rdf['actual_pnl'].values.copy()

    for p in range(N_PERMUTATIONS):
        # Shuffle signal-to-date mapping (HC #718: shuffle signals, NOT returns)
        shuffled_signal = np.random.permutation(signal_mask)
        perm_pnl = shuffled_signal * trade_pnls - shuffled_signal * cost_pct
        trade_only = perm_pnl[shuffled_signal == 1]
        if len(trade_only) > 10 and trade_only.std() > 0:
            ann_factor = np.sqrt(252 / LABEL_HORIZON)
            perm_s = trade_only.mean() / trade_only.std() * ann_factor
            perm_sharpes.append(perm_s)

    if perm_sharpes:
        perm_sharpes = np.array(perm_sharpes)
        p_value = (perm_sharpes >= actual_sharpe).mean()
        print(f"  Actual strategy Sharpe: {actual_sharpe:.3f}")
        print(f"  Permutation Sharpe: mean={perm_sharpes.mean():.3f}, "
              f"std={perm_sharpes.std():.3f}, p95={np.percentile(perm_sharpes,95):.3f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  SIGNIFICANT at 5% level — signal has predictive power beyond random.")
        elif p_value < 0.10:
            print(f"  Marginally significant at 10% level.")
        else:
            print(f"  NOT significant — model may not have real edge over random timing.")

# ─── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FINAL SUMMARY: ML Vol-Timed Income v1")
print("=" * 80)

if results_t1:
    m = results_t1['metrics_strategy']
    b = results_t1['metrics_baseline']
    print(f"\n  T-1 ML Strategy (proper, no lookahead):")
    print(f"    Sharpe:  {m['sharpe']:.2f}")
    print(f"    Sortino: {m['sortino']:.2f}")
    print(f"    CAGR:    {m['cagr']:.1f}%")
    print(f"    MaxDD:   {m['max_dd']:.1f}%")
    print(f"    WR:      {m['wr']:.1f}%")
    print(f"    PF:      {m['pf']:.2f}")
    print(f"    Trades:  {m['n_trades']}")
    print(f"\n  Baseline (sell premium every day):")
    print(f"    Sharpe:  {b['sharpe']:.2f}")
    print(f"    Sortino: {b['sortino']:.2f}")
    print(f"    CAGR:    {b['cagr']:.1f}%")
    print(f"    MaxDD:   {b['max_dd']:.1f}%")
    print(f"    WR:      {b['wr']:.1f}%")
    print(f"    PF:      {b['pf']:.2f}")

    if m['sharpe'] > b['sharpe']:
        improvement = m['sharpe'] - b['sharpe']
        print(f"\n  ML strategy OUTPERFORMS baseline by {improvement:.2f} Sharpe points.")
    else:
        shortfall = b['sharpe'] - m['sharpe']
        print(f"\n  ML strategy UNDERPERFORMS baseline by {shortfall:.2f} Sharpe points.")
        print(f"  Consider: different threshold, ensemble with LGBM, or more features.")

print(f"\n  Total runtime: {time.time() - t0_start:.0f}s")
print(f"\n[{datetime.datetime.now()}] Experiment complete.")
