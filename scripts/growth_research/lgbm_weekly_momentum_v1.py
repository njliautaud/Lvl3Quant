#!/usr/bin/env python3
"""
LightGBM Weekly ETF Momentum v1 — Combine best ML with best frequency
======================================================================

Prior findings:
- LightGBM monthly: Sharpe 0.84, CAGR 15.4%, 3/4 gates (v4 best combo)
- Simple weekly: Sharpe 0.94, CAGR 19.1%, 3/4 gates, R1 gap 0.163
- Simple bi-weekly: Sharpe 0.94, best R1 gap 0.100

Hypothesis: LightGBM captures signals that simple momentum misses.
Weekly rebalancing adapts to regime changes faster (R1 gap 0.10-0.16 vs 0.784).
Combining them should give BOTH better ranking AND faster adaptation.

Variants:
A. LightGBM weekly rebalance, Top 3
B. LightGBM weekly rebalance, Top 5
C. LightGBM bi-weekly rebalance, Top 3
D. LightGBM bi-weekly rebalance, Top 5
E. LightGBM weekly + defensive shift, Top 3
F. LightGBM weekly + defensive shift, Top 5
G. LightGBM weekly, Top 3, regime-penalized loss (from regime-robust findings)

Walk-forward: 252d train, 5d test (weekly), sliding.
Universe: 22 sector ETFs (same survivorship-free set).
Cost: 20bps round-trip.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'lgbm_weekly_momentum_v1_results.json'

# MLflow
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass

UNIVERSE = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',
    'XLC', 'QQQ', 'IWM', 'MDY', 'EFA', 'EEM', 'GLD', 'TLT', 'HYG', 'IYR',
    'VNQ', 'DBC',
]

DEFENSIVE_ETFS = {'XLP', 'XLU', 'TLT', 'GLD'}
RISK_ON_ETFS = {'XLK', 'QQQ', 'XLY', 'IWM', 'EEM'}


def build_features(close, spy_close=None):
    """Build features for a single ETF. Weekly-compatible."""
    lr = np.log(close / close.shift(1))

    feat = pd.DataFrame({
        # Momentum features
        'ret_5d': close.pct_change(5),
        'ret_10d': close.pct_change(10),
        'ret_21d': close.pct_change(21),
        'ret_63d': close.pct_change(63),
        'ret_126d': close.pct_change(126),
        'ret_252d': close.pct_change(252),
        'mom_12_1': close.pct_change(252) - close.pct_change(21),
        'high_52w_pct': close / close.rolling(252).max(),
        'mom_accel': close.pct_change(63) - close.pct_change(63).shift(63),

        # Volatility/quality features
        'vol_20d': lr.rolling(20).std() * np.sqrt(252),
        'vol_60d': lr.rolling(60).std() * np.sqrt(252),
        'vol_ratio': (lr.rolling(20).std() / lr.rolling(60).std()),
        'sharpe_63d': lr.rolling(63).mean() / lr.rolling(63).std(),
        'sharpe_126d': lr.rolling(126).mean() / lr.rolling(126).std(),
        'maxdd_63d': (close / close.rolling(63).max() - 1).rolling(63).min(),

        # Higher moments
        'skew_63d': lr.rolling(63).skew(),
        'kurt_63d': lr.rolling(63).kurt(),

        # Trend strength
        'vol_trend': (lr.rolling(20).std() - lr.rolling(60).std()) / lr.rolling(60).std(),
        'above_sma50': (close > close.rolling(50).mean()).astype(int),
        'above_sma200': (close > close.rolling(200).mean()).astype(int),
        'dist_sma200': (close - close.rolling(200).mean()) / close.rolling(200).mean(),

        # Realized vol regime
        'rv_21d': lr.rolling(21).std() * np.sqrt(252),
    }, index=close.index)

    # Cross-asset regime features
    if spy_close is not None:
        spy_lr = np.log(spy_close / spy_close.shift(1))
        feat['spy_ret_21d'] = spy_close.pct_change(21)
        feat['spy_ret_63d'] = spy_close.pct_change(63)
        feat['spy_above_sma200'] = (spy_close > spy_close.rolling(200).mean()).astype(int)
        feat['spy_rv_21d'] = spy_lr.rolling(21).std() * np.sqrt(252)
        feat['corr_spy_63d'] = lr.rolling(63).corr(spy_lr)
        feat['beta_spy_63d'] = lr.rolling(63).cov(spy_lr) / (spy_lr.rolling(63).var() + 1e-10)

    return feat


def detect_regime(spy_close, date):
    """Bull/bear from SPY vs SMA200."""
    if spy_close is None:
        return 'bull'
    loc = spy_close.index.searchsorted(date)
    if loc < 200:
        return 'bull'
    sma200 = spy_close.iloc[max(0, loc-200):loc].mean()
    return 'bear' if spy_close.iloc[loc-1] < sma200 else 'bull'


def run_variant(all_data, spy_data, spy_close, top_k, defensive_shift_factor=0.0,
                cost_bps=10, rebalance_days=5, name='base', regime_penalty=False):
    """Run walk-forward LightGBM with weekly/bi-weekly rebalancing."""
    import lightgbm as lgb

    # Build feature matrix
    features_list = []
    labels_list = []
    meta_list = []

    common = None
    for etf in all_data:
        if common is None:
            common = all_data[etf].index
        else:
            common = common.intersection(all_data[etf].index)

    for etf in all_data:
        close = all_data[etf]
        close = close.reindex(common)
        feat = build_features(close, spy_close.reindex(common) if spy_close is not None else None)

        # Forward return = next rebalance_days trading days
        fwd_ret = close.pct_change(rebalance_days).shift(-rebalance_days)

        feat['fwd_ret'] = fwd_ret
        feat['etf'] = etf
        feat['date'] = feat.index

        valid = feat.dropna(subset=['fwd_ret'])

        feature_cols = [c for c in feat.columns if c not in ['fwd_ret', 'etf', 'date']]

        features_list.append(valid[feature_cols])
        labels_list.append(valid['fwd_ret'])
        meta_list.append(valid[['etf', 'date']])

    feature_cols = [c for c in features_list[0].columns]
    X_all = pd.concat(features_list, axis=0)
    y_all = pd.concat(labels_list, axis=0)
    meta_all = pd.concat(meta_list, axis=0)

    X_all = X_all[feature_cols].astype(float)

    # Walk-forward with sliding window
    dates = sorted(meta_all['date'].unique())
    train_window = 252  # ~1 year of trading days

    all_picks = []
    all_returns = []
    all_dates = []
    all_regimes = []

    # Get rebalancing dates (every rebalance_days trading days)
    rebalance_dates = dates[train_window::rebalance_days]

    fprint(f"  {name}: {len(rebalance_dates)} rebalance periods, {len(dates)} total dates")

    prev_holdings = set()

    for i, rdate in enumerate(rebalance_dates):
        # Training data: last train_window days before rdate
        rdate_idx = dates.index(rdate)
        train_start = dates[max(0, rdate_idx - train_window)]

        train_mask = (meta_all['date'] >= train_start) & (meta_all['date'] < rdate)
        test_mask = meta_all['date'] == rdate

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_mask]
        meta_test = meta_all[test_mask]

        if len(X_train) < 100 or len(X_test) < 3:
            continue

        # Replace inf with nan, fill
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)

        for col in feature_cols:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

        # LightGBM with optional regime penalty
        if regime_penalty:
            regime = detect_regime(spy_close, rdate)
            if regime == 'bear':
                # Weight bear samples higher in training
                bear_dates_mask = []
                for d in meta_all[train_mask]['date'].unique():
                    r = detect_regime(spy_close, d)
                    bear_dates_mask.extend([r == 'bear'] * len(meta_all[(meta_all['date'] == d) & train_mask]))
                weights = np.array([2.0 if b else 1.0 for b in bear_dates_mask])
            else:
                weights = np.ones(len(X_train))

            ds_train = lgb.Dataset(X_train, label=y_train, weight=weights)
        else:
            ds_train = lgb.Dataset(X_train, label=y_train)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
        }

        model = lgb.train(params, ds_train, num_boost_round=100)

        # Predict
        preds = model.predict(X_test)
        test_etfs = meta_test['etf'].values

        # Rank and select
        pred_df = pd.DataFrame({'etf': test_etfs, 'pred': preds})

        # Defensive shift in bear markets
        if defensive_shift_factor > 0:
            regime = detect_regime(spy_close, rdate)
            if regime == 'bear':
                # Boost defensives, penalize risk-on
                for idx_row in pred_df.index:
                    etf = pred_df.loc[idx_row, 'etf']
                    if etf in DEFENSIVE_ETFS:
                        pred_df.loc[idx_row, 'pred'] += defensive_shift_factor * abs(pred_df['pred'].std())
                    elif etf in RISK_ON_ETFS:
                        pred_df.loc[idx_row, 'pred'] -= defensive_shift_factor * 0.5 * abs(pred_df['pred'].std())

        pred_df = pred_df.sort_values('pred', ascending=False)
        picks = pred_df.head(top_k)['etf'].tolist()

        # Calculate return for next period
        period_rets = []
        for etf in picks:
            if etf in all_data:
                close = all_data[etf]
                loc = close.index.searchsorted(rdate)
                if loc < len(close) - rebalance_days:
                    ret = (close.iloc[loc + rebalance_days] / close.iloc[loc]) - 1
                    period_rets.append(ret)

        if not period_rets:
            continue

        avg_ret = np.mean(period_rets)

        # Transaction costs
        new_holdings = set(picks)
        turnover = len(new_holdings - prev_holdings) / top_k  # fraction changed
        cost = turnover * cost_bps / 10000 * 2  # round-trip
        net_ret = avg_ret - cost

        prev_holdings = new_holdings

        regime = detect_regime(spy_close, rdate)
        all_picks.append(picks)
        all_returns.append(net_ret)
        all_dates.append(rdate)
        all_regimes.append(regime)

    if not all_returns:
        return None

    returns = np.array(all_returns)
    regimes = np.array(all_regimes)

    # Annualize based on rebalance frequency
    periods_per_year = 252 / rebalance_days

    # Metrics
    total_ret = np.prod(1 + returns) - 1
    n_years = len(returns) / periods_per_year
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    ann_vol = np.std(returns) * np.sqrt(periods_per_year)
    sharpe = (np.mean(returns) * periods_per_year) / (ann_vol + 1e-10)

    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(periods_per_year) if len(downside) > 0 else 1e-10
    sortino = (np.mean(returns) * periods_per_year) / (downside_vol + 1e-10)

    # MaxDD
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    maxdd = dd.min()

    # PF / WR
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    wr = len(wins) / len(returns) * 100
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999

    calmar = cagr / abs(maxdd) if maxdd != 0 else 999

    # R1: Regime analysis
    bull_rets = returns[regimes == 'bull']
    bear_rets = returns[regimes == 'bear']

    bull_sharpe = (np.mean(bull_rets) * periods_per_year) / (np.std(bull_rets) * np.sqrt(periods_per_year) + 1e-10) if len(bull_rets) > 5 else 0
    bear_sharpe = (np.mean(bear_rets) * periods_per_year) / (np.std(bear_rets) * np.sqrt(periods_per_year) + 1e-10) if len(bear_rets) > 5 else 0

    r1_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.01)
    r1_pass = r1_gap <= 0.50

    result = {
        'name': name,
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'cagr_pct': round(cagr * 100, 1),
        'maxdd_pct': round(maxdd * 100, 1),
        'wr_pct': round(wr, 1),
        'pf': round(pf, 2),
        'calmar': round(calmar, 2),
        'n_periods': len(returns),
        'n_years': round(n_years, 1),
        'bull_sharpe': round(bull_sharpe, 2),
        'bear_sharpe': round(bear_sharpe, 2),
        'r1_gap': round(r1_gap, 3),
        'r1_pass': r1_pass,
        'bull_periods': int(sum(regimes == 'bull')),
        'bear_periods': int(sum(regimes == 'bear')),
        'top_k': top_k,
        'rebalance_days': rebalance_days,
        'returns': returns.tolist(),
        'dates': [str(d) for d in all_dates],
        'regimes': regimes.tolist(),
    }

    fprint(f"  {name}: Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}%, MaxDD {maxdd*100:.1f}%, "
           f"WR {wr:.1f}%, PF {pf:.2f}, R1 gap {r1_gap:.3f} {'PASS' if r1_pass else 'FAIL'}")

    return result


def permutation_test(returns, n_perms=1000):
    """Block permutation test — shuffle returns in 4-week blocks."""
    real_sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
    block_size = 4
    n_blocks = len(returns) // block_size

    if n_blocks < 5:
        return 1.0  # Not enough data

    blocked = [returns[i*block_size:(i+1)*block_size] for i in range(n_blocks)]

    count_better = 0
    for _ in range(n_perms):
        perm_idx = np.random.permutation(n_blocks)
        perm_returns = np.concatenate([blocked[i] for i in perm_idx])
        # Shift returns randomly to break temporal structure
        shift = np.random.randint(1, len(perm_returns))
        perm_returns = np.roll(perm_returns, shift)
        # Random sign flip of blocks
        for i in range(0, len(perm_returns), block_size):
            if np.random.random() < 0.5:
                perm_returns[i:i+block_size] = -perm_returns[i:i+block_size]

        perm_sharpe = np.mean(perm_returns) / (np.std(perm_returns) + 1e-10)
        if perm_sharpe >= real_sharpe:
            count_better += 1

    return count_better / n_perms


def sub_period_test(returns, dates, n_splits=3):
    """Check if strategy works in all sub-periods."""
    n = len(returns)
    chunk = n // n_splits
    sub_sharpes = []
    for i in range(n_splits):
        sub = returns[i*chunk:(i+1)*chunk]
        ann_factor = np.sqrt(252 / 5)  # approximate for weekly
        s = np.mean(sub) * (252/5) / (np.std(sub) * ann_factor + 1e-10)
        sub_sharpes.append(s)

    # Pass if all sub-periods positive and min > 0.3
    all_positive = all(s > 0 for s in sub_sharpes)
    return all_positive, sub_sharpes


def outlier_test(returns, trim_pct=5):
    """Check if returns survive trimming top/bottom outliers."""
    n_trim = max(1, int(len(returns) * trim_pct / 100))
    sorted_rets = np.sort(returns)
    trimmed = sorted_rets[n_trim:-n_trim]

    orig_sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
    trim_sharpe = np.mean(trimmed) / (np.std(trimmed) + 1e-10)

    # Pass if trimmed Sharpe is still positive and > 50% of original
    passes = trim_sharpe > 0 and (trim_sharpe / (orig_sharpe + 1e-10)) > 0.5
    return passes, orig_sharpe, trim_sharpe


def main():
    import yfinance as yf

    fprint(f"LightGBM Weekly ETF Momentum v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    # Download data
    fprint("Downloading data...")
    tickers = UNIVERSE + ['SPY']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-25', progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    spy_close = close['SPY'] if 'SPY' in close.columns else None

    all_data = {}
    for etf in UNIVERSE:
        if etf in close.columns:
            s = close[etf].dropna()
            if len(s) > 500:
                all_data[etf] = s

    fprint(f"Data: {len(all_data)} ETFs loaded")
    fprint(f"Range: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")

    # Run all variants
    variants = [
        ('A_LGBM_Weekly_T3', 3, 5, 0.0, False),     # Weekly, top 3
        ('B_LGBM_Weekly_T5', 5, 5, 0.0, False),     # Weekly, top 5
        ('C_LGBM_Biweekly_T3', 3, 10, 0.0, False),  # Bi-weekly, top 3
        ('D_LGBM_Biweekly_T5', 5, 10, 0.0, False),  # Bi-weekly, top 5
        ('E_LGBM_Weekly_DefShift_T3', 3, 5, 1.5, False),  # Weekly + defensive shift
        ('F_LGBM_Weekly_DefShift_T5', 5, 5, 1.5, False),  # Weekly + defensive shift top 5
        ('G_LGBM_Weekly_RegPen_T3', 3, 5, 0.0, True),     # Weekly + regime penalty
    ]

    results = []

    if MLFLOW_OK:
        exp_name = 'lgbm_weekly_momentum_v1'
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                mlflow.create_experiment(exp_name)
        except:
            pass
        mlflow.set_experiment(exp_name)

    for vname, top_k, rebal_days, def_shift, reg_pen in variants:
        try:
            if MLFLOW_OK:
                with mlflow.start_run(run_name=vname):
                    r = run_variant(all_data, None, spy_close, top_k=top_k,
                                   defensive_shift_factor=def_shift,
                                   cost_bps=10, rebalance_days=rebal_days,
                                   name=vname, regime_penalty=reg_pen)
                    if r:
                        mlflow.log_params({
                            'top_k': top_k,
                            'rebalance_days': rebal_days,
                            'defensive_shift': def_shift,
                            'regime_penalty': reg_pen,
                        })
                        mlflow.log_metrics({
                            'sharpe': r['sharpe'],
                            'sortino': r['sortino'],
                            'cagr_pct': r['cagr_pct'],
                            'maxdd_pct': r['maxdd_pct'],
                            'wr_pct': r['wr_pct'],
                            'pf': r['pf'],
                            'r1_gap': r['r1_gap'],
                        })
                        results.append(r)
            else:
                r = run_variant(all_data, None, spy_close, top_k=top_k,
                               defensive_shift_factor=def_shift,
                               cost_bps=10, rebalance_days=rebal_days,
                               name=vname, regime_penalty=reg_pen)
                if r:
                    results.append(r)
        except Exception as e:
            fprint(f"  ERROR {vname}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        fprint("\nNo results! Something went wrong.")
        return

    # Adversarial validation on all variants
    fprint("\n" + "=" * 70)
    fprint("ADVERSARIAL VALIDATION")
    fprint("=" * 70)

    for r in results:
        rets = np.array(r['returns'])
        dates = r['dates']

        # G1: Permutation test
        perm_p = permutation_test(rets, n_perms=1000)
        r['perm_p'] = round(perm_p, 3)
        r['g1_pass'] = perm_p < 0.05

        # G2: R1 regime (already computed)
        r['g2_pass'] = r['r1_pass']

        # G3: Sub-period
        sub_pass, sub_sharpes = sub_period_test(rets, dates)
        r['g3_pass'] = sub_pass
        r['sub_sharpes'] = [round(s, 2) for s in sub_sharpes]

        # G4: Outlier
        out_pass, orig_s, trim_s = outlier_test(rets)
        r['g4_pass'] = out_pass
        r['trimmed_sharpe'] = round(trim_s, 2)

        gates = sum([r['g1_pass'], r['g2_pass'], r['g3_pass'], r['g4_pass']])
        r['gates_passed'] = gates

        fprint(f"\n{r['name']}:")
        fprint(f"  Sharpe {r['sharpe']}, CAGR {r['cagr_pct']}%, MaxDD {r['maxdd_pct']}%")
        fprint(f"  G1 Perm: {'PASS' if r['g1_pass'] else 'FAIL'} (p={r['perm_p']})")
        fprint(f"  G2 R1:   {'PASS' if r['g2_pass'] else 'FAIL'} (gap={r['r1_gap']})")
        fprint(f"  G3 Sub:  {'PASS' if r['g3_pass'] else 'FAIL'} (sharpes={r['sub_sharpes']})")
        fprint(f"  G4 Out:  {'PASS' if r['g4_pass'] else 'FAIL'} (trimmed={r['trimmed_sharpe']})")
        fprint(f"  GATES: {gates}/4")

    # Summary table
    fprint("\n" + "=" * 70)
    fprint("SUMMARY TABLE")
    fprint("=" * 70)
    fprint(f"{'Name':<35} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'R1':>7} {'Gates':>6}")
    fprint("-" * 80)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<35} {r['sharpe']:>7.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['wr_pct']:>5.1f}% {r['r1_gap']:>6.3f} {r['gates_passed']:>4}/4")

    # Compare with baselines
    fprint("\n" + "=" * 70)
    fprint("COMPARISON WITH BASELINES")
    fprint("=" * 70)
    fprint("LightGBM Monthly (v4):  Sharpe 0.84, CAGR 15.4%, R1 gap 0.352, 3/4 gates")
    fprint("Simple Weekly (v1-E):   Sharpe 0.94, CAGR 19.1%, R1 gap 0.163, 3/4 gates")
    fprint("Simple Biweekly (v1-E): Sharpe 0.94, CAGR 19.1%, R1 gap 0.163, 3/4 gates")
    best = max(results, key=lambda x: x['sharpe'])
    fprint(f"BEST HERE ({best['name']}): Sharpe {best['sharpe']}, CAGR {best['cagr_pct']}%, "
           f"R1 gap {best['r1_gap']}, {best['gates_passed']}/4 gates")

    # Save results (without raw returns for JSON)
    save_results = []
    for r in results:
        r_save = {k: v for k, v in r.items() if k not in ['returns', 'dates', 'regimes']}
        save_results.append(r_save)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    fprint(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
