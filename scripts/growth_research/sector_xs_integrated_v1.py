#!/usr/bin/env python3
"""
sector_xs_integrated_v1.py — Cross-Sectional Z-Score Integration into Sector Bull+Bear Spread Strategy

Tests whether cross-sectional features (z-scores, rank percentile, dispersion) improve the
production sector ETF options strategy beyond just improving LGBM IC.

Variants:
  A: Baseline (12 momentum/vol features)
  B: + Cross-sectional features (xs_zscore_5d, xs_rank_pct, xs_dispersion)
  C: + Cross-sectional + macro (VIX, TLT, credit spread proxy)
  D: + Cross-sectional only, TOP_K=2 (concentrated)
  E: + Cross-sectional + lookback optimization (126d/378d vs 252d)
  F: Random control (random sector selection, same mechanics)

Validation gates:
  1. Permutation test (1000 sign-flip trials, p < 0.05)
  2. Regime stability (|Sharpe_bull - Sharpe_bear| / max < 0.50)
  3. Sub-period (both halves profitable, Sharpe > 0.5)
  4. vs Random (beat random by >20% Sharpe)
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb

warnings.filterwarnings('ignore')

# MLflow setup
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://localhost:5000/', timeout=3)
    import mlflow
    mlflow.set_tracking_uri('http://localhost:5000')
    MLFLOW_OK = True
except:
    pass

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
SECTORS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
MACRO_TICKERS = ['^VIX', 'SPY', 'TLT', 'HYG']
ALL_TICKERS = SECTORS + MACRO_TICKERS

STARTING_CAPITAL = 645.0
COMMISSION_PER_SPREAD = 2.60
EXIT_HAIRCUT = 0.15
HOLD_DAYS = 20
TOP_K_DEFAULT = 3

TRAIN_WINDOW_DEFAULT = 252
OOT_WINDOW = 21

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/sector_xs_integrated_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ─── DATA DOWNLOAD ───────────────────────────────────────────────────────────
def download_data():
    log("Downloading data...")
    df = yf.download(ALL_TICKERS, start='2018-01-01', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close']
    else:
        close = df
    close = close.ffill().dropna(how='all')
    log(f"Data: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}, {len(close)} rows")
    return close


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
def compute_baseline_features(sector_close):
    """12 standard momentum/vol features per sector."""
    feats = {}
    for sec in SECTORS:
        if sec not in sector_close.columns:
            continue
        px = sector_close[sec]
        f = pd.DataFrame(index=sector_close.index)
        f['ret_5d'] = px.pct_change(5)
        f['ret_10d'] = px.pct_change(10)
        f['ret_21d'] = px.pct_change(21)
        f['ret_63d'] = px.pct_change(63)
        f['ret_126d'] = px.pct_change(126)
        f['ret_252d'] = px.pct_change(252)
        f['vol_21d'] = px.pct_change().rolling(21).std()
        f['vol_63d'] = px.pct_change().rolling(63).std()
        rets = px.pct_change()
        f['sharpe_63d'] = rets.rolling(63).mean() / (rets.rolling(63).std() + 1e-8)
        # max drawdown 63d
        roll_max = px.rolling(63).max()
        dd = (px - roll_max) / roll_max
        f['maxdd_63d'] = dd.rolling(63).min()
        # pct from 52w high
        high_252 = px.rolling(252).max()
        f['pct_52w_high'] = px / high_252 - 1.0
        # momentum acceleration
        f['mom_accel'] = f['ret_21d'] - f['ret_63d'] / 3.0
        feats[sec] = f
    return feats


def compute_cross_sectional_features(sector_close):
    """Cross-sectional z-score, rank percentile, dispersion."""
    ret_5d = sector_close[SECTORS].pct_change(5)
    xs_feats = {}
    for sec in SECTORS:
        f = pd.DataFrame(index=sector_close.index)
        # z-score of 5d return vs all sectors
        xs_mean = ret_5d.mean(axis=1)
        xs_std = ret_5d.std(axis=1) + 1e-8
        f['xs_zscore_5d'] = (ret_5d[sec] - xs_mean) / xs_std
        # rank percentile
        f['xs_rank_pct'] = ret_5d.rank(axis=1, pct=True)[sec]
        # cross-sectional dispersion (same for all sectors — regime signal)
        f['xs_dispersion'] = xs_std
        xs_feats[sec] = f
    return xs_feats


def compute_macro_features(close_df):
    """Macro features from VIX, SPY, TLT, HYG."""
    macro = pd.DataFrame(index=close_df.index)
    if '^VIX' in close_df.columns:
        vix = close_df['^VIX']
        macro['vix_level'] = vix
        macro['vix_chg_5d'] = vix.pct_change(5)
    else:
        macro['vix_level'] = 20.0
        macro['vix_chg_5d'] = 0.0

    if 'TLT' in close_df.columns:
        macro['tlt_ret_21d'] = close_df['TLT'].pct_change(21)
    else:
        macro['tlt_ret_21d'] = 0.0

    # Credit spread proxy: HYG - TLT relative performance (wider = stress)
    if 'HYG' in close_df.columns and 'TLT' in close_df.columns:
        macro['credit_spread_proxy'] = close_df['TLT'].pct_change(21) - close_df['HYG'].pct_change(21)
    else:
        macro['credit_spread_proxy'] = 0.0

    if 'SPY' in close_df.columns:
        spy = close_df['SPY']
        macro['spy_above_sma200'] = (spy / spy.rolling(200).mean() - 1.0)
    else:
        macro['spy_above_sma200'] = 0.0

    return macro


# ─── OPTIONS PRICING ─────────────────────────────────────────────────────────
def estimate_spread_value(atr_pct, is_bull=True):
    """
    ATR-based spread value estimate.
    Bull call spread: value increases when underlying goes up.
    Bear put spread: value increases when underlying goes down.
    Typical spread cost ~$1.50-$3.00 depending on ATR.
    """
    base_cost = max(0.80, min(3.50, atr_pct * 50))
    return base_cost


def compute_spread_pnl(entry_price, exit_multiplier, is_winner, haircut=0.15):
    """
    Compute P&L for a spread trade.
    entry_price: cost to enter spread
    exit_multiplier: how much the spread moved (1.0 = break even before costs)
    is_winner: True if direction was correct
    haircut: 15% exit cost
    """
    if is_winner:
        gross_exit = entry_price * exit_multiplier
        exit_value = gross_exit * (1.0 - haircut)
    else:
        # Loser: lose portion of premium
        loss_pct = min(0.85, abs(exit_multiplier - 1.0) + 0.3)
        exit_value = entry_price * (1.0 - loss_pct)

    pnl = exit_value - entry_price - COMMISSION_PER_SPREAD
    return pnl


# ─── LGBM SECTOR RANKER ─────────────────────────────────────────────────────
def build_panel(baseline_feats, xs_feats, macro_feats, sector_close, feature_set='baseline'):
    """Build panel dataframe for LGBM training."""
    rows = []
    fwd_ret = sector_close[SECTORS].pct_change(HOLD_DAYS).shift(-HOLD_DAYS)

    for sec in SECTORS:
        if sec not in baseline_feats:
            continue
        bf = baseline_feats[sec].copy()

        if feature_set in ('xs', 'xs_macro'):
            xf = xs_feats[sec]
            bf = bf.join(xf, how='left')

        if feature_set == 'xs_macro':
            bf = bf.join(macro_feats, how='left')

        bf['sector'] = sec
        bf['fwd_ret'] = fwd_ret[sec]
        bf['date'] = bf.index
        rows.append(bf)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna()
    return panel


def get_feature_cols(feature_set):
    """Return feature column names for given variant."""
    base = ['ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
            'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel']
    if feature_set == 'baseline':
        return base
    elif feature_set == 'xs':
        return base + ['xs_zscore_5d', 'xs_rank_pct', 'xs_dispersion']
    elif feature_set == 'xs_macro':
        return base + ['xs_zscore_5d', 'xs_rank_pct', 'xs_dispersion',
                       'vix_level', 'vix_chg_5d', 'tlt_ret_21d', 'credit_spread_proxy', 'spy_above_sma200']
    return base


def walk_forward_lgbm(panel, feature_cols, train_window=252, top_k=3, random_mode=False):
    """
    Sliding walk-forward LGBM sector ranking.
    Returns list of trade results.
    """
    dates = sorted(panel['date'].unique())
    n_dates = len(dates)

    # We need at least train_window + OOT_WINDOW dates
    min_start = train_window
    trades = []
    ic_values = []

    # Step through in OOT_WINDOW increments
    for start_idx in range(min_start, n_dates - OOT_WINDOW, OOT_WINDOW):
        train_start = start_idx - train_window
        train_end = start_idx
        test_start = start_idx
        test_end = min(start_idx + OOT_WINDOW, n_dates)

        train_dates = dates[train_start:train_end]
        test_dates = dates[test_start:test_end]

        train_mask = panel['date'].isin(train_dates)
        test_mask = panel['date'].isin(test_dates)

        train_df = panel[train_mask]
        test_df = panel[test_mask]

        if len(train_df) < 50 or len(test_df) < 5:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df['fwd_ret'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['fwd_ret'].values

        if random_mode:
            # Random predictions for control
            preds = np.random.randn(len(test_df))
        else:
            # Train LGBM
            dtrain = lgb.Dataset(X_train, label=y_train)
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_jobs': -1,
            }
            model = lgb.train(params, dtrain, num_boost_round=100)
            preds = model.predict(X_test)

        # Compute IC for this fold
        if len(y_test) > 5:
            corr = np.corrcoef(preds, y_test)[0, 1]
            if not np.isnan(corr):
                ic_values.append(corr)

        # Generate trades from predictions per test date
        test_df_copy = test_df.copy()
        test_df_copy['pred'] = preds

        for dt in test_dates:
            day_df = test_df_copy[test_df_copy['date'] == dt]
            if len(day_df) < len(SECTORS) * 0.5:
                continue

            # Get VIX for regime
            vix_row = day_df[day_df['sector'] == SECTORS[0]]
            # Determine regime from macro features if available
            vix_val = 20.0  # default
            if 'vix_level' in day_df.columns:
                vix_vals = day_df['vix_level'].dropna()
                if len(vix_vals) > 0:
                    vix_val = vix_vals.iloc[0]

            # Rank sectors by prediction
            ranked = day_df.sort_values('pred', ascending=False)

            if vix_val >= 20:
                # Bull regime: bull call spreads on top-K
                selected = ranked.head(top_k)
                is_bull = True
            else:
                # Bear regime: bear put spreads on bottom-K
                selected = ranked.tail(top_k)
                is_bull = False

            for _, row in selected.iterrows():
                actual_ret = row['fwd_ret']
                # Determine if trade is winner
                if is_bull:
                    is_winner = actual_ret > 0.01  # needs to move up meaningfully
                else:
                    is_winner = actual_ret < -0.01  # needs to move down

                # ATR proxy from vol
                atr_pct = row.get('vol_21d', 0.015)
                if pd.isna(atr_pct) or atr_pct < 0.005:
                    atr_pct = 0.015

                entry_price = estimate_spread_value(atr_pct, is_bull)

                # Exit multiplier based on actual return magnitude
                ret_mag = abs(actual_ret) if not pd.isna(actual_ret) else 0.0
                if is_winner:
                    exit_mult = 1.0 + ret_mag * 15  # leverage from options
                    exit_mult = min(exit_mult, 2.5)  # cap at max spread value
                else:
                    exit_mult = 1.0 - ret_mag * 10
                    exit_mult = max(exit_mult, 0.1)

                pnl = compute_spread_pnl(entry_price, exit_mult, is_winner, EXIT_HAIRCUT)

                trades.append({
                    'date': dt,
                    'sector': row['sector'],
                    'is_bull': is_bull,
                    'is_winner': is_winner,
                    'entry_price': entry_price,
                    'pnl': pnl,
                    'vix': vix_val,
                    'fwd_ret': actual_ret,
                })

    avg_ic = np.mean(ic_values) if ic_values else 0.0
    return trades, avg_ic


# ─── METRICS CALCULATION ─────────────────────────────────────────────────────
def compute_metrics(trades, starting_capital=STARTING_CAPITAL):
    """Compute strategy metrics from trade list."""
    if not trades:
        return {
            'sharpe': 0, 'sortino': 0, 'cagr': 0, 'maxdd': -1.0,
            'calmar': 0, 'win_rate': 0, 'profit_factor': 0, 'n_trades': 0,
            'sharpe_bull': 0, 'sharpe_bear': 0, 'r1_gap': 1.0,
        }

    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['date'])

    # Daily P&L
    daily_pnl = df.groupby('date')['pnl'].sum()
    daily_pnl = daily_pnl.sort_index()

    # Equity curve
    equity = starting_capital + daily_pnl.cumsum()
    daily_returns = daily_pnl / starting_capital  # simple return on capital

    # Sharpe (annualized, assume ~12 trades/month)
    n_periods = len(daily_pnl)
    if n_periods < 5:
        return {
            'sharpe': 0, 'sortino': 0, 'cagr': 0, 'maxdd': -1.0,
            'calmar': 0, 'win_rate': 0, 'profit_factor': 0, 'n_trades': len(trades),
            'sharpe_bull': 0, 'sharpe_bear': 0, 'r1_gap': 1.0,
        }

    # Annualize based on trading frequency
    years = (daily_pnl.index[-1] - daily_pnl.index[0]).days / 365.25
    trades_per_year = n_periods / max(years, 0.5)
    ann_factor = np.sqrt(trades_per_year)

    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std() + 1e-8
    sharpe = mean_ret / std_ret * ann_factor

    # Sortino
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() + 1e-8 if len(downside) > 0 else 1e-8
    sortino = mean_ret / downside_std * ann_factor

    # CAGR
    total_ret = (equity.iloc[-1] / starting_capital) if len(equity) > 0 else 1.0
    cagr = (total_ret ** (1.0 / max(years, 0.5))) - 1.0

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    maxdd = dd.min()

    # Calmar
    calmar = cagr / abs(maxdd) if abs(maxdd) > 0.01 else 0.0

    # Win rate and profit factor
    winners = df[df['pnl'] > 0]
    losers = df[df['pnl'] <= 0]
    win_rate = len(winners) / len(df) if len(df) > 0 else 0.0
    gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0.0
    gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 1e-8
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Regime split
    bull_trades = df[df['is_bull'] == True]
    bear_trades = df[df['is_bull'] == False]

    def regime_sharpe(regime_df):
        if len(regime_df) < 5:
            return 0.0
        rpnl = regime_df.groupby('date')['pnl'].sum()
        rrets = rpnl / starting_capital
        if rrets.std() < 1e-8:
            return 0.0
        return rrets.mean() / rrets.std() * np.sqrt(12)

    sharpe_bull = regime_sharpe(bull_trades)
    sharpe_bear = regime_sharpe(bear_trades)
    max_regime = max(abs(sharpe_bull), abs(sharpe_bear), 1e-8)
    r1_gap = abs(sharpe_bull - sharpe_bear) / max_regime

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr, 4),
        'maxdd': round(maxdd, 4),
        'calmar': round(calmar, 3),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 3),
        'n_trades': len(trades),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'r1_gap': round(r1_gap, 4),
    }


# ─── VALIDATION GATES ────────────────────────────────────────────────────────
def permutation_test(trades, n_trials=1000):
    """Sign-flip permutation test. Returns p-value."""
    if not trades:
        return 1.0
    pnls = np.array([t['pnl'] for t in trades])
    actual_mean = pnls.mean()
    count_better = 0
    for _ in range(n_trials):
        signs = np.random.choice([-1, 1], size=len(pnls))
        perm_mean = (pnls * signs).mean()
        if perm_mean >= actual_mean:
            count_better += 1
    return count_better / n_trials


def sub_period_test(trades):
    """Test both halves are profitable with Sharpe > 0.5."""
    if len(trades) < 20:
        return False, 0.0, 0.0
    mid = len(trades) // 2
    first_half = trades[:mid]
    second_half = trades[mid:]
    m1 = compute_metrics(first_half)
    m2 = compute_metrics(second_half)
    return (m1['sharpe'] > 0.5 and m2['sharpe'] > 0.5), m1['sharpe'], m2['sharpe']


def run_validation(trades, metrics, random_sharpe):
    """Run all 4 validation gates."""
    results = {}

    # Gate 1: Permutation test
    p_val = permutation_test(trades)
    results['gate1_perm_pvalue'] = round(p_val, 4)
    results['gate1_pass'] = p_val < 0.05

    # Gate 2: Regime stability
    results['gate2_r1_gap'] = metrics['r1_gap']
    results['gate2_pass'] = metrics['r1_gap'] < 0.50

    # Gate 3: Sub-period
    sub_pass, sh1, sh2 = sub_period_test(trades)
    results['gate3_half1_sharpe'] = sh1
    results['gate3_half2_sharpe'] = sh2
    results['gate3_pass'] = sub_pass

    # Gate 4: vs Random
    if random_sharpe > 0:
        improvement = (metrics['sharpe'] - random_sharpe) / abs(random_sharpe) if abs(random_sharpe) > 0.01 else 10.0
    else:
        improvement = 10.0 if metrics['sharpe'] > 0 else 0.0
    results['gate4_vs_random_pct'] = round(improvement, 4)
    results['gate4_pass'] = improvement > 0.20

    results['all_gates_pass'] = all([results['gate1_pass'], results['gate2_pass'],
                                      results['gate3_pass'], results['gate4_pass']])
    return results


# ─── MAIN EXECUTION ─────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log("=" * 70)
    log("SECTOR XS INTEGRATED V1 — Cross-Sectional Feature Integration Test")
    log("=" * 70)

    # Download data
    close_df = download_data()

    # Compute all features
    log("Computing features...")
    sector_close = close_df[SECTORS].copy()
    baseline_feats = compute_baseline_features(sector_close)
    xs_feats = compute_cross_sectional_features(sector_close)
    macro_feats = compute_macro_features(close_df)

    # Add VIX to panel for regime detection
    vix_series = close_df['^VIX'] if '^VIX' in close_df.columns else pd.Series(20.0, index=close_df.index)

    # Build panels for each feature set
    log("Building panels...")
    panel_baseline = build_panel(baseline_feats, xs_feats, macro_feats, sector_close, 'baseline')
    panel_xs = build_panel(baseline_feats, xs_feats, macro_feats, sector_close, 'xs')
    panel_xs_macro = build_panel(baseline_feats, xs_feats, macro_feats, sector_close, 'xs_macro')

    # Add VIX level to panels that need it for regime routing
    for panel in [panel_baseline, panel_xs, panel_xs_macro]:
        vix_map = vix_series.to_dict()
        panel['vix_level_regime'] = panel['date'].map(lambda d: vix_map.get(d, 20.0))

    # ─── RUN VARIANTS ────────────────────────────────────────────────────────
    variants = {}

    # Variant A: Baseline
    log("\n--- Variant A: Baseline (12 features) ---")
    feat_cols_a = get_feature_cols('baseline')
    trades_a, ic_a = walk_forward_lgbm(panel_baseline, feat_cols_a, train_window=252, top_k=3)
    metrics_a = compute_metrics(trades_a)
    metrics_a['ic'] = round(ic_a, 4)
    log(f"  Sharpe={metrics_a['sharpe']}, IC={ic_a:.4f}, Trades={metrics_a['n_trades']}")
    variants['A_baseline'] = {'metrics': metrics_a, 'trades': trades_a}

    # Variant B: + Cross-sectional
    log("\n--- Variant B: + Cross-Sectional Features ---")
    feat_cols_b = get_feature_cols('xs')
    trades_b, ic_b = walk_forward_lgbm(panel_xs, feat_cols_b, train_window=252, top_k=3)
    metrics_b = compute_metrics(trades_b)
    metrics_b['ic'] = round(ic_b, 4)
    metrics_b['ic_improvement'] = round((ic_b - ic_a) / max(abs(ic_a), 0.001), 4)
    log(f"  Sharpe={metrics_b['sharpe']}, IC={ic_b:.4f}, IC_impr={metrics_b['ic_improvement']:.2%}")
    variants['B_xs'] = {'metrics': metrics_b, 'trades': trades_b}

    # Variant C: + Cross-sectional + Macro
    log("\n--- Variant C: + Cross-Sectional + Macro ---")
    feat_cols_c = get_feature_cols('xs_macro')
    trades_c, ic_c = walk_forward_lgbm(panel_xs_macro, feat_cols_c, train_window=252, top_k=3)
    metrics_c = compute_metrics(trades_c)
    metrics_c['ic'] = round(ic_c, 4)
    metrics_c['ic_improvement'] = round((ic_c - ic_a) / max(abs(ic_a), 0.001), 4)
    log(f"  Sharpe={metrics_c['sharpe']}, IC={ic_c:.4f}, IC_impr={metrics_c['ic_improvement']:.2%}")
    variants['C_xs_macro'] = {'metrics': metrics_c, 'trades': trades_c}

    # Variant D: XS features, TOP_K=2 (concentrated)
    log("\n--- Variant D: + Cross-Sectional, TOP_K=2 ---")
    trades_d, ic_d = walk_forward_lgbm(panel_xs, feat_cols_b, train_window=252, top_k=2)
    metrics_d = compute_metrics(trades_d)
    metrics_d['ic'] = round(ic_d, 4)
    metrics_d['ic_improvement'] = round((ic_d - ic_a) / max(abs(ic_a), 0.001), 4)
    log(f"  Sharpe={metrics_d['sharpe']}, IC={ic_d:.4f}, Trades={metrics_d['n_trades']}")
    variants['D_xs_topk2'] = {'metrics': metrics_d, 'trades': trades_d}

    # Variant E: XS + lookback optimization (126d and 378d windows)
    log("\n--- Variant E: + Cross-Sectional, Lookback Optimization ---")
    # Test 126d window
    trades_e126, ic_e126 = walk_forward_lgbm(panel_xs, feat_cols_b, train_window=126, top_k=3)
    metrics_e126 = compute_metrics(trades_e126)
    # Test 378d window
    trades_e378, ic_e378 = walk_forward_lgbm(panel_xs, feat_cols_b, train_window=378, top_k=3)
    metrics_e378 = compute_metrics(trades_e378)
    # Pick best
    if metrics_e126['sharpe'] >= metrics_e378['sharpe']:
        trades_e, ic_e, metrics_e = trades_e126, ic_e126, metrics_e126
        best_window = 126
    else:
        trades_e, ic_e, metrics_e = trades_e378, ic_e378, metrics_e378
        best_window = 378
    metrics_e['ic'] = round(ic_e, 4)
    metrics_e['best_window'] = best_window
    metrics_e['ic_improvement'] = round((ic_e - ic_a) / max(abs(ic_a), 0.001), 4)
    log(f"  Best window={best_window}d, Sharpe={metrics_e['sharpe']}, IC={ic_e:.4f}")
    log(f"  (126d: Sharpe={metrics_e126['sharpe']}, 378d: Sharpe={metrics_e378['sharpe']})")
    variants['E_xs_lookback'] = {'metrics': metrics_e, 'trades': trades_e}

    # Variant F: Random control
    log("\n--- Variant F: Random Control ---")
    trades_f, ic_f = walk_forward_lgbm(panel_baseline, feat_cols_a, train_window=252, top_k=3, random_mode=True)
    metrics_f = compute_metrics(trades_f)
    metrics_f['ic'] = round(ic_f, 4)
    log(f"  Sharpe={metrics_f['sharpe']}, Trades={metrics_f['n_trades']}")
    variants['F_random'] = {'metrics': metrics_f, 'trades': trades_f}

    random_sharpe = metrics_f['sharpe']

    # ─── VALIDATION ──────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("VALIDATION GATES")
    log("=" * 70)

    all_results = {}
    for name, data in variants.items():
        if name == 'F_random':
            continue  # Don't validate the random control against itself
        log(f"\n  Validating {name}...")
        val = run_validation(data['trades'], data['metrics'], random_sharpe)
        data['metrics']['validation'] = val
        passed = "PASS" if val['all_gates_pass'] else "FAIL"
        log(f"    Gate1(perm p={val['gate1_perm_pvalue']}): {'PASS' if val['gate1_pass'] else 'FAIL'}")
        log(f"    Gate2(regime gap={val['gate2_r1_gap']:.3f}): {'PASS' if val['gate2_pass'] else 'FAIL'}")
        log(f"    Gate3(sub-period): {'PASS' if val['gate3_pass'] else 'FAIL'} (H1={val['gate3_half1_sharpe']:.2f}, H2={val['gate3_half2_sharpe']:.2f})")
        log(f"    Gate4(vs random +{val['gate4_vs_random_pct']:.0%}): {'PASS' if val['gate4_pass'] else 'FAIL'}")
        log(f"    OVERALL: {passed}")

    # ─── SUMMARY ─────────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"\n{'Variant':<20} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'WR':>6} {'PF':>6} {'IC':>7} {'Trades':>7}")
    log("-" * 90)
    for name, data in variants.items():
        m = data['metrics']
        log(f"{name:<20} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['cagr']:>8.2%} {m['maxdd']:>8.2%} {m['win_rate']:>6.1%} {m['profit_factor']:>6.2f} {m['ic']:>7.4f} {m['n_trades']:>7}")

    # ─── SAVE RESULTS ────────────────────────────────────────────────────────
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'sectors': SECTORS,
            'starting_capital': STARTING_CAPITAL,
            'commission': COMMISSION_PER_SPREAD,
            'exit_haircut': EXIT_HAIRCUT,
            'hold_days': HOLD_DAYS,
            'train_window_default': TRAIN_WINDOW_DEFAULT,
            'oot_window': OOT_WINDOW,
        },
        'variants': {},
    }
    for name, data in variants.items():
        output['variants'][name] = data['metrics']

    # Determine best variant
    non_random = {k: v for k, v in variants.items() if k != 'F_random'}
    best_name = max(non_random, key=lambda k: non_random[k]['metrics']['sharpe'])
    output['best_variant'] = best_name
    output['best_sharpe'] = variants[best_name]['metrics']['sharpe']
    output['baseline_sharpe'] = metrics_a['sharpe']
    output['xs_improves_strategy'] = metrics_b['sharpe'] > metrics_a['sharpe']
    output['xs_sharpe_lift'] = round(metrics_b['sharpe'] - metrics_a['sharpe'], 3)

    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nResults saved to {results_path}")

    # ─── MLFLOW LOGGING ──────────────────────────────────────────────────────
    if MLFLOW_OK:
        log("Logging to MLflow...")
        try:
            mlflow.set_experiment("sector_xs_integrated_v1")
            with mlflow.start_run(run_name=f"xs_integrated_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("n_sectors", len(SECTORS))
                mlflow.log_param("train_window", TRAIN_WINDOW_DEFAULT)
                mlflow.log_param("hold_days", HOLD_DAYS)
                mlflow.log_param("starting_capital", STARTING_CAPITAL)
                mlflow.log_param("exit_haircut", EXIT_HAIRCUT)

                for name, data in variants.items():
                    m = data['metrics']
                    prefix = name.lower()
                    mlflow.log_metric(f"{prefix}_sharpe", m['sharpe'])
                    mlflow.log_metric(f"{prefix}_sortino", m['sortino'])
                    mlflow.log_metric(f"{prefix}_cagr", m['cagr'])
                    mlflow.log_metric(f"{prefix}_maxdd", m['maxdd'])
                    mlflow.log_metric(f"{prefix}_win_rate", m['win_rate'])
                    mlflow.log_metric(f"{prefix}_pf", m['profit_factor'])
                    mlflow.log_metric(f"{prefix}_ic", m['ic'])
                    mlflow.log_metric(f"{prefix}_n_trades", m['n_trades'])

                mlflow.log_metric("xs_sharpe_lift", output['xs_sharpe_lift'])
                mlflow.log_metric("best_sharpe", output['best_sharpe'])
                mlflow.log_artifact(results_path)
            log("MLflow logging complete.")
        except Exception as e:
            log(f"MLflow logging failed: {e}")
    else:
        log("MLflow not available, skipping.")

    elapsed = time.time() - t0
    log(f"\nTotal runtime: {elapsed:.1f}s")
    log("DONE.")


if __name__ == '__main__':
    main()
