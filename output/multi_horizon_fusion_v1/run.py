#!/usr/bin/env python3
"""
Multi-Horizon Signal Fusion Analysis v1
========================================
Research question: Does combining 1-month and 3-month asymmetric signals
improve stock selection vs using 1-month alone?

Walk-forward: 504d rolling train, 21d test step, predict 1m forward return
Anti-lookahead: ALL signals computed on T-1 data.
"""

import os, sys, time, warnings, json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/multi_horizon_fusion_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[{datetime.now():%H:%M:%S}] Multi-Horizon Fusion Analysis v1 starting...", flush=True)

# ============================================================
# 1. DATA DOWNLOAD
# ============================================================
import yfinance as yf

TICKERS_50 = [
    'AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','BRK-B','UNH','JNJ',
    'JPM','V','PG','XOM','HD','CVX','MA','ABBV','MRK','LLY',
    'PEP','KO','COST','AVGO','TMO','WMT','MCD','CSCO','ACN','ABT',
    'DHR','TXN','NEE','PM','UNP','LIN','RTX','LOW','HON','AMGN',
    'IBM','CAT','BA','GE','INTC','QCOM','SBUX','MDT','BLK','GILD'
]

print(f"[{datetime.now():%H:%M:%S}] Downloading data...", flush=True)
all_syms = TICKERS_50 + ['SPY', '^VIX']
raw_data = {}
for sym in all_syms:
    try:
        df = yf.download(sym, start='2014-01-01', end='2026-07-18', progress=False, auto_adjust=True)
        if len(df) > 252:
            raw_data[sym] = df
    except:
        pass

spy = raw_data.pop('SPY')
vix = raw_data.pop('^VIX')
valid_tickers = sorted(raw_data.keys())
print(f"[{datetime.now():%H:%M:%S}] Got {len(valid_tickers)} stocks, SPY has {len(spy)} days", flush=True)

# ============================================================
# 2. FEATURE ENGINEERING (T-1 lagged)
# ============================================================
print(f"[{datetime.now():%H:%M:%S}] Computing features...", flush=True)

def compute_features(prices_df, spy_df, vix_df, ticker):
    df = prices_df[['Close', 'Volume']].copy()
    df.columns = ['close', 'volume']
    df = df.dropna()

    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_21d'] = df['close'].pct_change(21)
    df['ret_63d'] = df['close'].pct_change(63)
    df['fwd_ret_21d'] = df['close'].shift(-21) / df['close'] - 1

    # 1M features
    df['vol_21d'] = df['ret_1d'].rolling(21).std() * np.sqrt(252)
    df['vol_63d'] = df['ret_1d'].rolling(63).std() * np.sqrt(252)
    df['mom_1m'] = df['ret_21d']
    df['vol_surge'] = df['volume'] / df['volume'].rolling(21).mean()
    df['dist_52w_high'] = df['close'] / df['close'].rolling(252).max() - 1

    # Vol pctrank - use vectorized approach instead of rolling apply
    vol_series = df['vol_63d']
    vol_pctrank = vol_series.copy() * np.nan
    for i in range(252, len(vol_series)):
        window = vol_series.iloc[i-252:i+1]
        vol_pctrank.iloc[i] = (window < window.iloc[-1]).sum() / (len(window) - 1)
    df['vol_pctrank'] = vol_pctrank

    # 3M features
    df['mom_3m'] = df['ret_63d']
    df['vol_126d'] = df['ret_1d'].rolling(126).std() * np.sqrt(252)
    sma200 = df['close'].rolling(200).mean()
    df['dist_200sma'] = df['close'] / sma200 - 1
    df['mom_6m'] = df['close'].pct_change(126)
    df['mean_reversion_3m'] = -df['ret_63d']
    df['vol_trend_3m'] = df['volume'].rolling(63).mean() / df['volume'].rolling(126).mean() - 1

    # Market features
    vix_close = vix_df['Close'].reindex(df.index, method='ffill')
    df['vix'] = vix_close
    df['vix_21d_chg'] = vix_close.pct_change(21)

    spy_close = spy_df['Close'].reindex(df.index, method='ffill')
    df['spy_ret_21d'] = spy_close.pct_change(21)
    df['spy_ret_63d'] = spy_close.pct_change(63)
    spy_sma200 = spy_close.rolling(200).mean()
    df['spy_above_200sma'] = (spy_close > spy_sma200).astype(float)
    df['beta_63d'] = df['ret_1d'].rolling(63).corr(spy_close.pct_change())

    # T-1 LAG all features
    feature_cols = [c for c in df.columns if c not in ['close', 'volume', 'fwd_ret_21d']]
    for c in feature_cols:
        df[c] = df[c].shift(1)

    df['ticker'] = ticker
    return df

all_features = []
for i, t in enumerate(valid_tickers):
    feat = compute_features(raw_data[t], spy, vix, t)
    all_features.append(feat)
    if (i + 1) % 10 == 0:
        print(f"  Features computed for {i+1}/{len(valid_tickers)} stocks", flush=True)

panel = pd.concat(all_features, axis=0)
panel = panel.dropna(subset=['fwd_ret_21d'])
panel.index.name = 'date'
panel = panel.reset_index()

FEATURES_1M = ['vol_21d', 'vol_63d', 'mom_1m', 'vol_surge', 'dist_52w_high',
               'vol_pctrank', 'vix', 'spy_ret_21d', 'ret_1d', 'ret_5d']
FEATURES_3M = ['mom_3m', 'vol_126d', 'dist_200sma', 'mom_6m', 'mean_reversion_3m',
               'vol_trend_3m', 'vix_21d_chg', 'spy_ret_63d', 'spy_above_200sma', 'beta_63d']
FEATURES_ALL = FEATURES_1M + FEATURES_3M
TARGET = 'fwd_ret_21d'

panel = panel.dropna(subset=FEATURES_ALL + [TARGET])
print(f"[{datetime.now():%H:%M:%S}] Panel: {len(panel)} rows, {panel['date'].min().date()} to {panel['date'].max().date()}", flush=True)

# ============================================================
# 3. WALK-FORWARD ENGINE (optimized)
# ============================================================
import lightgbm as lgb
from scipy.stats import spearmanr

TRAIN_DAYS = 504
TEST_STEP = 21

# Pre-index dates for fast lookup
dates = sorted(panel['date'].unique())
date_to_idx = {d: i for i, d in enumerate(dates)}
panel['date_idx'] = panel['date'].map(date_to_idx)
n_dates = len(dates)

print(f"[{datetime.now():%H:%M:%S}] Total dates: {n_dates}, folds: ~{(n_dates - TRAIN_DAYS) // TEST_STEP}", flush=True)

LGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 24,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 30,
    'verbose': -1,
    'n_jobs': 4,
    'random_state': 42,
}

def walk_forward(panel_df, feature_cols, model_name):
    """Optimized walk-forward with date_idx for fast slicing."""
    t0 = time.time()
    results = []
    fold = 0

    start_idx = TRAIN_DAYS
    while start_idx + TEST_STEP <= n_dates:
        train_start = max(0, start_idx - TRAIN_DAYS)
        train_mask = (panel_df['date_idx'] >= train_start) & (panel_df['date_idx'] < start_idx)
        test_mask = (panel_df['date_idx'] >= start_idx) & (panel_df['date_idx'] < start_idx + TEST_STEP)

        train = panel_df.loc[train_mask, feature_cols + [TARGET]].dropna()
        test = panel_df.loc[test_mask].dropna(subset=feature_cols + [TARGET])

        if len(train) < 500 or len(test) < 10:
            start_idx += TEST_STEP
            continue

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            train[feature_cols].values, train[TARGET].values,
            eval_set=[(test[feature_cols].values, test[TARGET].values)],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )

        preds = model.predict(test[feature_cols].values)

        keep_cols = ['date', 'ticker', TARGET]
        for c in ['vol_pctrank', 'mom_1m', 'vol_surge']:
            if c in test.columns:
                keep_cols.append(c)

        fold_result = test[keep_cols].copy()
        fold_result['pred'] = preds
        fold_result['fold'] = fold
        results.append(fold_result)

        fold += 1
        start_idx += TEST_STEP

        if fold % 20 == 0:
            elapsed = time.time() - t0
            print(f"    {model_name}: fold {fold}, {elapsed:.0f}s elapsed", flush=True)

    all_results = pd.concat(results, ignore_index=True)
    elapsed = time.time() - t0
    print(f"  {model_name}: {fold} folds, {len(all_results)} preds, {elapsed:.0f}s", flush=True)
    return all_results

# ============================================================
# 4. RUN ALL STRATEGIES
# ============================================================

print(f"\n[{datetime.now():%H:%M:%S}] === Strategy A: 1m-only ===", flush=True)
results_A = walk_forward(panel, FEATURES_1M, 'A_1m_only')

print(f"[{datetime.now():%H:%M:%S}] === Strategy B: 3m-only ===", flush=True)
results_B = walk_forward(panel, FEATURES_3M, 'B_3m_only')

print(f"[{datetime.now():%H:%M:%S}] === Strategy C: Combined 1m+3m ===", flush=True)
results_C = walk_forward(panel, FEATURES_ALL, 'C_combined')

# Strategy D: Agreement
print(f"[{datetime.now():%H:%M:%S}] === Strategy D: Agreement filter ===", flush=True)
results_D = results_A[['date', 'ticker', TARGET, 'pred', 'fold']].merge(
    results_B[['date', 'ticker', 'pred']].rename(columns={'pred': 'pred_3m'}),
    on=['date', 'ticker'], how='inner'
)
results_D['agree'] = (np.sign(results_D['pred']) == np.sign(results_D['pred_3m']))
results_D_filtered = results_D[results_D['agree']].copy()
print(f"  D: {len(results_D_filtered)}/{len(results_D)} survive agreement "
      f"({100*len(results_D_filtered)/max(1,len(results_D)):.1f}%)", flush=True)

# Strategy E: Cascade
print(f"[{datetime.now():%H:%M:%S}] === Strategy E: Cascade ===", flush=True)
results_E = results_D.copy()
results_E['rank_3m'] = results_E.groupby('date')['pred_3m'].rank(pct=True)
results_E_filtered = results_E[results_E['rank_3m'] <= 0.5].copy()
print(f"  E: {len(results_E_filtered)}/{len(results_E)} survive cascade "
      f"({100*len(results_E_filtered)/max(1,len(results_E)):.1f}%)", flush=True)

# Add filter columns to D and E
for col in ['vol_pctrank', 'mom_1m', 'vol_surge']:
    if col in results_A.columns:
        a_lookup = results_A[['date', 'ticker', col]].drop_duplicates()
        if col not in results_D_filtered.columns:
            results_D_filtered = results_D_filtered.merge(a_lookup, on=['date', 'ticker'], how='left')
        if col not in results_E_filtered.columns:
            results_E_filtered = results_E_filtered.merge(a_lookup, on=['date', 'ticker'], how='left')

# ============================================================
# 5. EVALUATION
# ============================================================
print(f"\n[{datetime.now():%H:%M:%S}] Evaluating strategies...", flush=True)

spy_200sma = spy['Close'].rolling(200).mean()

def evaluate_strategy(results_df, name, pred_col='pred', apply_asymmetric=True):
    df = results_df.copy()
    ic_val, ic_pval = spearmanr(df[pred_col], df[TARGET])

    fold_ics = []
    for f in df['fold'].unique():
        fd = df[df['fold'] == f]
        if len(fd) > 5:
            fic, _ = spearmanr(fd[pred_col], fd[TARGET])
            if not np.isnan(fic):
                fold_ics.append(fic)
    ic_mean = np.mean(fold_ics) if fold_ics else 0
    ic_std = np.std(fold_ics) if fold_ics else 0
    icir = ic_mean / ic_std if ic_std > 0 else 0

    portfolio_rets = []
    regime_rets = {'bull': [], 'bear': []}

    for date in sorted(df['date'].unique()):
        day_df = df[df['date'] == date]
        if len(day_df) < 10:
            continue

        if apply_asymmetric and all(c in day_df.columns for c in ['vol_pctrank', 'mom_1m', 'vol_surge']):
            asym = day_df[
                (day_df['vol_pctrank'] >= 0.80) &
                (day_df['mom_1m'] < 0) &
                (day_df['vol_surge'] > 1.5)
            ]
            if len(asym) >= 2:
                day_df = asym

        n_q = max(1, len(day_df) // 5)
        shorts = day_df.nsmallest(n_q, pred_col)
        longs = day_df.nlargest(n_q, pred_col)

        ls_ret = longs[TARGET].mean() - shorts[TARGET].mean()
        short_ret = -shorts[TARGET].mean()

        portfolio_rets.append({
            'date': date, 'ls_ret': ls_ret, 'short_ret': short_ret,
            'long_ret': longs[TARGET].mean(), 'n_stocks': len(day_df),
        })

        # Regime
        spy_dates = spy.index[spy.index <= date]
        if len(spy_dates) > 0:
            last = spy_dates[-1]
            if last in spy_200sma.index:
                sc = float(spy.loc[last, 'Close'])
                sm = float(spy_200sma.loc[last])
                if not np.isnan(sm):
                    regime = 'bull' if sc > sm else 'bear'
                    regime_rets[regime].append(ls_ret)

    port_df = pd.DataFrame(portfolio_rets)
    if len(port_df) == 0:
        return {'name': name, 'error': 'no trades'}

    ls_mean = port_df['ls_ret'].mean()
    ls_std = port_df['ls_ret'].std()
    ppy = 252 / 21

    sharpe = (ls_mean / ls_std * np.sqrt(ppy)) if ls_std > 0 else 0
    sortino_denom = port_df['ls_ret'][port_df['ls_ret'] < 0].std()
    sortino = (ls_mean / sortino_denom * np.sqrt(ppy)) if sortino_denom > 0 else 0
    cagr = (1 + ls_mean) ** ppy - 1
    cum_ret = (1 + port_df['ls_ret']).cumprod()
    max_dd = (cum_ret / cum_ret.cummax() - 1).min()
    hit_rate = (port_df['ls_ret'] > 0).mean()
    pf_wins = port_df['ls_ret'][port_df['ls_ret'] > 0].sum()
    pf_losses = abs(port_df['ls_ret'][port_df['ls_ret'] < 0].sum())
    profit_factor = pf_wins / pf_losses if pf_losses > 0 else np.inf

    short_mean = port_df['short_ret'].mean()
    short_std = port_df['short_ret'].std()
    short_sharpe = (short_mean / short_std * np.sqrt(ppy)) if short_std > 0 else 0
    short_hit = (port_df['short_ret'] > 0).mean()

    regime_sharpes = {}
    for regime in ['bull', 'bear']:
        r = regime_rets[regime]
        if len(r) > 2:
            rm, rs = np.mean(r), np.std(r)
            regime_sharpes[regime] = rm / rs * np.sqrt(ppy) if rs > 0 else 0
        else:
            regime_sharpes[regime] = np.nan

    if not np.isnan(regime_sharpes.get('bull', np.nan)) and not np.isnan(regime_sharpes.get('bear', np.nan)):
        max_abs = max(abs(regime_sharpes['bull']), abs(regime_sharpes['bear']))
        regime_div = abs(regime_sharpes['bull'] - regime_sharpes['bear']) / max_abs if max_abs > 0 else 0
    else:
        regime_div = np.nan

    return {
        'name': name,
        'concat_IC': ic_val, 'ic_pval': ic_pval,
        'mean_fold_IC': ic_mean, 'ICIR': icir,
        'n_folds': len(fold_ics), 'n_predictions': len(df),
        'sharpe_LS': sharpe, 'sortino_LS': sortino,
        'cagr_LS': cagr, 'maxDD_LS': max_dd,
        'hit_rate_LS': hit_rate, 'profit_factor_LS': profit_factor,
        'short_sharpe': short_sharpe, 'short_hit_rate': short_hit,
        'sharpe_bull': regime_sharpes.get('bull', np.nan),
        'sharpe_bear': regime_sharpes.get('bear', np.nan),
        'regime_divergence': regime_div,
        'n_rebal_periods': len(port_df),
    }

metrics = {}
for name, res_df, asym in [
    ('A_1m_only', results_A, True),
    ('A_1m_nofilter', results_A, False),
    ('B_3m_only', results_B, False),
    ('C_combined', results_C, True),
    ('C_combined_nofilter', results_C, False),
    ('D_agreement', results_D_filtered, True),
    ('D_agreement_nofilter', results_D_filtered, False),
    ('E_cascade', results_E_filtered, True),
    ('E_cascade_nofilter', results_E_filtered, False),
]:
    print(f"  {name}...", flush=True)
    m = evaluate_strategy(res_df, name, apply_asymmetric=asym)
    metrics[name] = m

# ============================================================
# 6. PERMUTATION TEST
# ============================================================
print(f"\n[{datetime.now():%H:%M:%S}] Permutation test (200 shuffles)...", flush=True)

best_name = max(
    [k for k, v in metrics.items() if 'error' not in v],
    key=lambda k: metrics[k]['sharpe_LS']
)
print(f"  Best: {best_name} (Sharpe={metrics[best_name]['sharpe_LS']:.3f})", flush=True)

strat_map = {
    'A_1m_only': results_A, 'A_1m_nofilter': results_A,
    'B_3m_only': results_B,
    'C_combined': results_C, 'C_combined_nofilter': results_C,
    'D_agreement': results_D_filtered, 'D_agreement_nofilter': results_D_filtered,
    'E_cascade': results_E_filtered, 'E_cascade_nofilter': results_E_filtered,
}
best_results = strat_map[best_name]

real_ic = metrics[best_name]['concat_IC']
perm_ics = []
np.random.seed(42)
for i in range(200):
    shuffled_pred = np.random.permutation(best_results['pred'].values)
    pic, _ = spearmanr(shuffled_pred, best_results[TARGET].values)
    perm_ics.append(pic)
perm_ics = np.array(perm_ics)
perm_pval = (np.abs(perm_ics) >= np.abs(real_ic)).mean()
print(f"  Real IC={real_ic:.4f}, p={perm_pval:.4f}", flush=True)

# ============================================================
# 7. LAG SENSITIVITY
# ============================================================
print(f"\n[{datetime.now():%H:%M:%S}] Lag sensitivity (T-2 test)...", flush=True)

panel_t2 = panel.copy()
for c in FEATURES_ALL:
    panel_t2[c] = panel_t2.groupby('ticker')[c].shift(1)
panel_t2 = panel_t2.dropna(subset=FEATURES_ALL + [TARGET])
panel_t2['date_idx'] = panel_t2['date'].map(date_to_idx)

if best_name.startswith('A') or best_name.startswith('a'):
    lag_feats = FEATURES_1M
elif best_name.startswith('B') or best_name.startswith('b'):
    lag_feats = FEATURES_3M
else:
    lag_feats = FEATURES_ALL

results_lag = walk_forward(panel_t2, lag_feats, f'{best_name}_T2')
lag_ic, _ = spearmanr(results_lag['pred'], results_lag[TARGET])
print(f"  T-1 IC={real_ic:.4f}, T-2 IC={lag_ic:.4f}", flush=True)

# ============================================================
# 8. REPORT
# ============================================================
print(f"\n[{datetime.now():%H:%M:%S}] Generating report...", flush=True)

lines = []
lines.append("=" * 80)
lines.append("MULTI-HORIZON SIGNAL FUSION ANALYSIS v1")
lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
lines.append("=" * 80)
lines.append("")
lines.append(f"Universe: {len(valid_tickers)} large-cap US stocks | Period: {panel['date'].min().date()} to {panel['date'].max().date()}")
lines.append(f"Walk-forward: {TRAIN_DAYS}d train, {TEST_STEP}d test step | Compute: CPU LightGBM")
lines.append("")
lines.append("-" * 80)
lines.append("STRATEGY COMPARISON (Long-Short Portfolio)")
lines.append("-" * 80)
lines.append("")

hdr = f"{'Strategy':<25} {'IC':>7} {'ICIR':>6} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'HitRate':>7} {'PF':>6}"
lines.append(hdr)
lines.append("-" * len(hdr))

for name in ['A_1m_only', 'A_1m_nofilter', 'B_3m_only',
             'C_combined', 'C_combined_nofilter',
             'D_agreement', 'D_agreement_nofilter',
             'E_cascade', 'E_cascade_nofilter']:
    m = metrics[name]
    if 'error' in m:
        lines.append(f"{name:<25} ERROR: {m['error']}")
        continue
    lines.append(
        f"{name:<25} {m['concat_IC']:>7.4f} {m['ICIR']:>6.2f} {m['sharpe_LS']:>7.3f} "
        f"{m['sortino_LS']:>8.3f} {m['cagr_LS']:>7.1%} {m['maxDD_LS']:>7.1%} "
        f"{m['hit_rate_LS']:>7.1%} {m['profit_factor_LS']:>6.2f}"
    )

lines.append("")
lines.append("-" * 80)
lines.append("SHORT-ONLY METRICS")
lines.append("-" * 80)
for name in ['A_1m_only', 'B_3m_only', 'C_combined', 'D_agreement', 'E_cascade']:
    m = metrics[name]
    if 'error' not in m:
        lines.append(f"  {name:<23} Short Sharpe={m['short_sharpe']:.3f}  Hit={m['short_hit_rate']:.1%}")

lines.append("")
lines.append("-" * 80)
lines.append("REGIME-STRATIFIED SHARPE (Bull=SPY>200SMA, Bear=SPY<200SMA)")
lines.append("-" * 80)
for name in ['A_1m_only', 'B_3m_only', 'C_combined', 'D_agreement', 'E_cascade']:
    m = metrics[name]
    if 'error' not in m:
        div = m['regime_divergence']
        passes = "PASS" if (not np.isnan(div) and div <= 0.50) else "FAIL"
        lines.append(f"  {name:<23} Bull={m['sharpe_bull']:.3f}  Bear={m['sharpe_bear']:.3f}  Div={div:.3f}  {passes}")

lines.append("")
lines.append("-" * 80)
lines.append("PERMUTATION TEST (200 shuffles)")
lines.append("-" * 80)
lines.append(f"  Best strategy: {best_name}")
lines.append(f"  Real IC: {real_ic:.4f}  |  Perm p-value: {perm_pval:.4f}  |  Significant: {'YES' if perm_pval < 0.05 else 'NO'}")

lines.append("")
lines.append("-" * 80)
lines.append("LAG SENSITIVITY")
lines.append("-" * 80)
lines.append(f"  T-1 IC: {real_ic:.4f}  |  T-2 IC: {lag_ic:.4f}  |  Decay: {real_ic - lag_ic:.4f}")
lines.append(f"  Lag-robust: {'YES' if lag_ic > real_ic * 0.7 else 'NO (>30% decay)'}")

lines.append("")
lines.append("=" * 80)
lines.append("CONCLUSIONS")
lines.append("=" * 80)

baseline_m = metrics['A_1m_only']
best_m = metrics[best_name]

if best_name.startswith('A'):
    lines.append("VERDICT: NO IMPROVEMENT from multi-horizon fusion")
    lines.append("  The 1m-only baseline is the best approach.")
else:
    delta = best_m['sharpe_LS'] - baseline_m['sharpe_LS']
    if delta > 0.1:
        lines.append(f"VERDICT: IMPROVEMENT — {best_name} beats baseline by {delta:.3f} Sharpe")
    else:
        lines.append(f"VERDICT: MARGINAL — {best_name} beats baseline by only {delta:.3f} Sharpe")

lines.append("")
for name in ['C_combined', 'D_agreement', 'E_cascade']:
    m = metrics.get(name, {})
    if 'error' not in m:
        delta = m['sharpe_LS'] - baseline_m['sharpe_LS']
        sign = '+' if delta > 0 else ''
        lines.append(f"  {name}: Sharpe {m['sharpe_LS']:.3f} ({sign}{delta:.3f} vs baseline)")

lines.append("")
lines.append("RECOMMENDATIONS:")
any_better = False
for name in ['C_combined', 'D_agreement', 'E_cascade']:
    m = metrics.get(name, {})
    if 'error' not in m and m['sharpe_LS'] > baseline_m['sharpe_LS'] + 0.05:
        any_better = True
        lines.append(f"  * {name} shows improvement — investigate further")
if not any_better:
    lines.append("  * No fusion approach meaningfully beats the 1m baseline")
    lines.append("  * Stick with existing 1m-only ranker")
    lines.append("  * 3m horizon adds noise, not signal, for 1m forward prediction")

lines.append("")
lines.append(f"Permutation: {'PASS' if perm_pval < 0.05 else 'FAIL'} | Lag: {'PASS' if lag_ic > real_ic * 0.7 else 'FAIL'}")
lines.append("=" * 80)

report = "\n".join(lines)
with open(OUTPUT_DIR / "summary_report.txt", 'w') as f:
    f.write(report)

# Save metrics JSON
def convert_np(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

with open(OUTPUT_DIR / "metrics.json", 'w') as f:
    json.dump({k: {kk: convert_np(vv) for kk, vv in v.items()} for k, v in metrics.items()}, f, indent=2)

for name, df in [('A', results_A), ('B', results_B), ('C', results_C)]:
    df.to_parquet(OUTPUT_DIR / f"predictions_{name}.parquet", index=False)

print(f"\n{report}", flush=True)
print(f"\n[{datetime.now():%H:%M:%S}] DONE.", flush=True)
