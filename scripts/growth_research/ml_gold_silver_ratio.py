#!/usr/bin/env python3
"""
ML Gold/Silver Ratio Trading
===============================
HC #714: ML for exploratory research. HC #710: Cross-asset signals.

Concept: The gold/silver ratio (GSR) is a classic mean-reversion indicator.
When GSR is high (>80), silver is cheap relative to gold → long silver.
When GSR is low (<60), gold is cheap relative to silver → long gold.

ML Enhancement: GBM predicts the optimal timing for ratio convergence trades
using features: ratio momentum, rate environment, USD, VIX, commodity complex.

Walk-forward LightGBM (252d sliding, 21d advance).
Full adversarial validation.
HC #713: Fixed $100K, no DCA.
"""

import numpy as np
import pandas as pd
import warnings
import json
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
import yfinance as yf
import lightgbm as lgb

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_gold_silver_ratio")
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML GOLD/SILVER RATIO TRADING")
print("=" * 70)

# ─── Config ───
GOLD_TICKER = 'GLD'
SILVER_TICKER = 'SLV'
MACRO_TICKERS = ['^VIX', 'TLT', 'UUP', 'DBC', 'HYG', 'SPY', 'CPER', '^TNX']

TRAIN_DAYS = 252
ADVANCE_DAYS = 21
INITIAL_CAPITAL = 100_000
N_PERMS = 200

# ─── Download Data ───
print("\nDownloading data...")
all_tickers = list(set([GOLD_TICKER, SILVER_TICKER] + MACRO_TICKERS))
data = {}

for t in all_tickers:
    try:
        df = yf.download(t, start='2006-01-01', end='2026-07-20',
                         progress=False, auto_adjust=True)
        if len(df) > 100:
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', t)].copy()
            else:
                close = df['Close'].copy()
            clean = t.replace('^', '')
            close.name = clean
            data[clean] = close
            print(f"  {t}: {len(df)} days")
    except Exception as e:
        print(f"  {t}: FAILED ({e})")

prices = pd.DataFrame(data).dropna(how='all').ffill().dropna()
print(f"\nAligned: {len(prices)} days, {prices.shape[1]} assets")

if GOLD_TICKER not in prices.columns or SILVER_TICKER not in prices.columns:
    print("ERROR: Need both GLD and SLV")
    exit(1)

# ─── Compute Gold/Silver Ratio ───
returns = prices.pct_change()
prices['GSR'] = prices[GOLD_TICKER] / prices[SILVER_TICKER]
print(f"GSR range: {prices['GSR'].min():.1f} to {prices['GSR'].max():.1f}")
print(f"GSR mean: {prices['GSR'].mean():.1f}")

# ─── Build Features ───
print("\nBuilding features...")
features = pd.DataFrame(index=prices.index)

# GSR features (core)
features['gsr'] = prices['GSR']
features['gsr_zscore_63d'] = (prices['GSR'] - prices['GSR'].rolling(63).mean()) / prices['GSR'].rolling(63).std()
features['gsr_zscore_126d'] = (prices['GSR'] - prices['GSR'].rolling(126).mean()) / prices['GSR'].rolling(126).std()
features['gsr_zscore_252d'] = (prices['GSR'] - prices['GSR'].rolling(252).mean()) / prices['GSR'].rolling(252).std()
features['gsr_mom_5d'] = prices['GSR'].pct_change(5)
features['gsr_mom_21d'] = prices['GSR'].pct_change(21)
features['gsr_mom_63d'] = prices['GSR'].pct_change(63)
features['gsr_vol_21d'] = prices['GSR'].pct_change().rolling(21).std()
features['gsr_range_21d'] = (prices['GSR'].rolling(21).max() - prices['GSR'].rolling(21).min()) / prices['GSR'].rolling(21).mean()

# Gold & Silver individual features
for metal in [GOLD_TICKER, SILVER_TICKER]:
    features[f'{metal}_mom_5d'] = prices[metal].pct_change(5)
    features[f'{metal}_mom_21d'] = prices[metal].pct_change(21)
    features[f'{metal}_mom_63d'] = prices[metal].pct_change(63)
    features[f'{metal}_vol_21d'] = returns[metal].rolling(21).std()
    features[f'{metal}_drawdown'] = prices[metal] / prices[metal].cummax() - 1

# Silver/Gold relative volatility
features['slv_gld_vol_ratio'] = returns[SILVER_TICKER].rolling(21).std() / returns[GOLD_TICKER].rolling(21).std().clip(lower=1e-6)

# Macro features
macro_names = [t.replace('^', '') for t in MACRO_TICKERS if t.replace('^', '') in prices.columns]
for m in macro_names:
    features[f'{m}_ret_21d'] = prices[m].pct_change(21)
    features[f'{m}_vol_21d'] = returns[m].rolling(21).std() if m in returns.columns else 0

if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_zscore'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()

if 'UUP' in prices.columns:
    features['usd_strength'] = prices['UUP'].pct_change(63)

features = features.dropna()
print(f"Features: {features.shape[1]} columns, {len(features)} days")

# ─── Walk-Forward Backtest ───
print("\nRunning walk-forward backtest...")

# Label: which metal outperforms in next 21 days?
# 0 = gold outperforms, 1 = silver outperforms
gold_fwd = prices[GOLD_TICKER].pct_change(ADVANCE_DAYS).shift(-ADVANCE_DAYS)
silver_fwd = prices[SILVER_TICKER].pct_change(ADVANCE_DAYS).shift(-ADVANCE_DAYS)

rebal_indices = list(range(TRAIN_DAYS, len(features) - ADVANCE_DAYS, ADVANCE_DAYS))
print(f"Rebalancing periods: {len(rebal_indices)}")

all_dates = []
all_ml_returns = []
all_baseline_returns = []  # Simple ratio threshold
all_gold_returns = []
all_silver_returns = []
all_spy_returns = []
all_positions = []

# HC #718 R3: Transaction costs — 5 bps per leg on turnover
COST_BPS = 5
prev_ml_pos = None

for idx_num, start_idx in enumerate(rebal_indices):
    train_start = max(0, start_idx - TRAIN_DAYS)
    # HC #718: label gap = ADVANCE_DAYS to prevent look-ahead
    train_end = start_idx - ADVANCE_DAYS
    fwd_start = start_idx
    fwd_end = min(start_idx + ADVANCE_DAYS, len(features))

    if fwd_end <= fwd_start:
        continue

    train_dates = features.index[train_start:train_end]
    fwd_dates = features.index[fwd_start:fwd_end]

    # Training data
    train_X = features.loc[train_dates].values
    # Label: silver outperforms gold?
    train_gold_fwd = gold_fwd.reindex(train_dates).dropna()
    train_silver_fwd = silver_fwd.reindex(train_dates).dropna()
    common = train_gold_fwd.index.intersection(train_silver_fwd.index).intersection(features.index)

    if len(common) < 50:
        continue

    train_X_clean = features.loc[common].values
    train_y = (silver_fwd.loc[common] > gold_fwd.loc[common]).astype(int).values

    if len(set(train_y)) < 2:
        continue

    # Replace NaN/Inf
    train_X_clean = np.nan_to_num(train_X_clean, nan=0, posinf=0, neginf=0)

    # Train model
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, n_jobs=1
    )
    model.fit(train_X_clean, train_y)

    # Predict
    rebal_date = features.index[start_idx]
    pred_X = np.nan_to_num(features.loc[rebal_date].values.reshape(1, -1), nan=0, posinf=0, neginf=0)
    prob_silver = model.predict_proba(pred_X)[0, 1]

    # ML position: >0.6 → silver, <0.4 → gold, else 50/50
    if prob_silver > 0.6:
        ml_gold_w, ml_silver_w = 0.0, 1.0
        pos = 'silver'
    elif prob_silver < 0.4:
        ml_gold_w, ml_silver_w = 1.0, 0.0
        pos = 'gold'
    else:
        ml_gold_w, ml_silver_w = 0.5, 0.5
        pos = 'neutral'

    # Simple baseline: GSR z-score threshold
    gsr_z = features.loc[rebal_date, 'gsr_zscore_126d'] if 'gsr_zscore_126d' in features.columns else 0
    if gsr_z > 1.0:  # GSR high → silver cheap → long silver
        base_gold_w, base_silver_w = 0.0, 1.0
    elif gsr_z < -1.0:  # GSR low → gold cheap → long gold
        base_gold_w, base_silver_w = 1.0, 0.0
    else:
        base_gold_w, base_silver_w = 0.5, 0.5

    # HC #718 R3: transaction costs on position change
    rebal_cost = 0.0
    if prev_ml_pos is not None and pos != prev_ml_pos:
        rebal_cost = 2 * COST_BPS / 10000  # sell old + buy new
    elif prev_ml_pos is None and pos != 'neutral':
        rebal_cost = COST_BPS / 10000  # initial entry
    prev_ml_pos = pos

    # Apply for forward period
    for day_idx, date in enumerate(fwd_dates):
        if date not in returns.index:
            continue

        gold_ret = returns.loc[date, GOLD_TICKER]
        silver_ret = returns.loc[date, SILVER_TICKER]
        spy_ret = returns.loc[date, 'SPY'] if 'SPY' in returns.columns else 0

        ml_ret = ml_gold_w * gold_ret + ml_silver_w * silver_ret
        # HC #718 R3: apply rebalance cost on first day of period only
        if day_idx == 0:
            ml_ret -= rebal_cost
        base_ret = base_gold_w * gold_ret + base_silver_w * silver_ret

        all_dates.append(date)
        all_ml_returns.append(ml_ret)
        all_baseline_returns.append(base_ret)
        all_gold_returns.append(gold_ret)
        all_silver_returns.append(silver_ret)
        all_spy_returns.append(spy_ret)
        all_positions.append(pos)

    if (idx_num + 1) % 20 == 0:
        print(f"  Period {idx_num+1}/{len(rebal_indices)}")

print(f"\nBacktest complete: {len(all_dates)} days, {len(rebal_indices)} rebalances")

# ─── Compute Metrics ───
def compute_metrics(rets, name):
    r = np.array(rets, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 20:
        return {'name': name, 'error': 'insufficient'}
    ann_ret = np.mean(r) * 252
    ann_vol = np.std(r) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    neg = r[r < 0]
    neg_vol = np.std(neg) * np.sqrt(252) if len(neg) > 0 else 1e-6
    sortino = ann_ret / neg_vol
    cum = np.cumprod(1 + r)
    max_dd = np.min(cum / np.maximum.accumulate(cum) - 1)
    cagr = cum[-1] ** (252 / len(r)) - 1
    wr = (r > 0).mean()
    pf = r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).sum() != 0 else 999
    calmar = (cagr) / abs(max_dd) if max_dd != 0 else 0
    return {
        'name': name, 'Sharpe': round(sharpe, 3), 'Sortino': round(sortino, 3),
        'CAGR': round(cagr * 100, 1), 'MaxDD': round(max_dd * 100, 1),
        'Calmar': round(calmar, 2), 'WR': round(wr * 100, 1), 'PF': round(pf, 2),
        'n_days': len(r),
    }

ml_m = compute_metrics(all_ml_returns, 'ML Gold/Silver')
base_m = compute_metrics(all_baseline_returns, 'Simple Ratio Threshold')
gold_m = compute_metrics(all_gold_returns, 'Gold Buy-Hold')
silver_m = compute_metrics(all_silver_returns, 'Silver Buy-Hold')
spy_m = compute_metrics(all_spy_returns, 'SPY Buy-Hold')

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
for m in [ml_m, base_m, gold_m, silver_m, spy_m]:
    print(f"\n{m['name']}:")
    for k, v in m.items():
        if k != 'name':
            print(f"  {k}: {v}")

# Position distribution
pos_counts = pd.Series(all_positions).value_counts()
print(f"\nPosition distribution:")
for p, c in pos_counts.items():
    print(f"  {p}: {c} days ({c/len(all_positions)*100:.1f}%)")

# ─── Yearly Returns ───
yearly = {'ML': {}, 'Baseline': {}, 'Gold': {}, 'SPY': {}}
for d, ml_r, base_r, g_r, spy_r in zip(all_dates, all_ml_returns, all_baseline_returns, all_gold_returns, all_spy_returns):
    yr = d.year
    for key, r in [('ML', ml_r), ('Baseline', base_r), ('Gold', g_r), ('SPY', spy_r)]:
        if yr not in yearly[key]:
            yearly[key][yr] = []
        yearly[key][yr].append(r)
for key in yearly:
    for yr in yearly[key]:
        yearly[key][yr] = round((np.prod(1 + np.array(yearly[key][yr])) - 1) * 100, 1)

neg_years = sum(1 for y in yearly.get('ML', {}).values() if isinstance(y, (int, float)) and y < 0)
print(f"\nYearly Returns (%): ML | Baseline | Gold | SPY")
for yr in sorted(set(y for s in yearly.values() for y in s)):
    ml_y = yearly.get('ML', {}).get(yr, '-')
    b_y = yearly.get('Baseline', {}).get(yr, '-')
    g_y = yearly.get('Gold', {}).get(yr, '-')
    s_y = yearly.get('SPY', {}).get(yr, '-')
    print(f"  {yr}: {ml_y} | {b_y} | {g_y} | {s_y}")

# ─── Adversarial Validation ───
print(f"\n{'='*70}")
print("ADVERSARIAL VALIDATION")
print(f"{'='*70}")

ml_rets = np.array(all_ml_returns, dtype=float)
ml_rets = ml_rets[~np.isnan(ml_rets)]
real_sharpe = ml_m['Sharpe']

# 1. Permutation
# HC #718: shuffle signals, not returns
# Shuffle position assignments (gold/silver/neutral) across dates, recompute returns
print(f"\n1. PERMUTATION ({N_PERMS} shuffles, signal shuffle)...")
perm_sharpes = []
unique_positions_perm = list(set(all_positions))
for _ in range(N_PERMS):
    perm_rets = []
    for d_idx_p in range(len(all_dates)):
        date_p = all_dates[d_idx_p]
        random_pos = np.random.choice(unique_positions_perm)
        if date_p not in returns.index:
            perm_rets.append(0)
            continue
        gold_ret_p = returns.loc[date_p, GOLD_TICKER]
        silver_ret_p = returns.loc[date_p, SILVER_TICKER]
        if random_pos == 'gold':
            perm_rets.append(gold_ret_p)
        elif random_pos == 'silver':
            perm_rets.append(silver_ret_p)
        else:
            perm_rets.append(0.5 * gold_ret_p + 0.5 * silver_ret_p)
    pr = np.array(perm_rets, dtype=float)
    pr = pr[~np.isnan(pr)]
    pa = np.mean(pr) * 252
    pv = np.std(pr) * np.sqrt(252)
    perm_sharpes.append(pa / pv if pv > 0 else 0)
perm_mean = np.mean(perm_sharpes)
perm_p = np.mean([s >= real_sharpe for s in perm_sharpes])
perm_pass = perm_p < 0.05
print(f"  Real: {real_sharpe:.3f}, Perm: {perm_mean:.3f}, p={perm_p:.3f} {'PASS' if perm_pass else 'FAIL'}")

# 2. Sub-period
print(f"\n2. SUB-PERIOD...")
n = len(ml_rets)
sub_sharpes = []
for q in range(4):
    s, e = q * (n // 4), (q + 1) * (n // 4) if q < 3 else n
    qr = ml_rets[s:e]
    qa = np.mean(qr) * 252
    qv = np.std(qr) * np.sqrt(252)
    sub_sharpes.append(round(qa / qv if qv > 0 else 0, 3))
sub_cv = np.std(sub_sharpes) / np.mean(sub_sharpes) if np.mean(sub_sharpes) > 0 else 999
all_pos = all(s > 0 for s in sub_sharpes)
sub_pass = all_pos and sub_cv < 1.0
print(f"  Quarters: {sub_sharpes}, CV={sub_cv:.3f} {'PASS' if sub_pass else 'FAIL'}")

# 3. Outlier
print(f"\n3. OUTLIER...")
p5, p95 = np.percentile(ml_rets, [5, 95])
trimmed = ml_rets[(ml_rets >= p5) & (ml_rets <= p95)]
ta = np.mean(trimmed) * 252
tv = np.std(trimmed) * np.sqrt(252)
trim_sharpe = ta / tv if tv > 0 else 0
out_deg = (trim_sharpe - real_sharpe) / abs(real_sharpe) * 100 if real_sharpe != 0 else 0
out_pass = out_deg > -30
print(f"  Full: {real_sharpe:.3f}, Trimmed: {trim_sharpe:.3f}, Deg: {out_deg:.1f}% {'PASS' if out_pass else 'FAIL'}")

# 4. R1 Regime
print(f"\n4. R1 REGIME...")
spy_r = np.array(all_spy_returns, dtype=float)
spy_21d = pd.Series(spy_r).rolling(21).sum().values
green = spy_21d > 0
red = spy_21d < 0
if green.sum() > 20 and red.sum() > 20:
    g_sr = np.mean(ml_rets[green]) * 252 / (np.std(ml_rets[green]) * np.sqrt(252)) if np.std(ml_rets[green]) > 0 else 0
    r_sr = np.mean(ml_rets[red]) * 252 / (np.std(ml_rets[red]) * np.sqrt(252)) if np.std(ml_rets[red]) > 0 else 0
    gap = abs(g_sr - r_sr) / max(abs(g_sr), abs(r_sr), 0.01)
    r1_pass = gap < 0.50
else:
    g_sr, r_sr, gap = 0, 0, 999
    r1_pass = False
print(f"  Green: {g_sr:.3f}, Red: {r_sr:.3f}, Gap: {gap:.3f} {'PASS' if r1_pass else 'FAIL'}")

gates = sum([perm_pass, sub_pass, out_pass, r1_pass])
print(f"\nGATES: {gates}/4")

# SPY correlation
min_len = min(len(ml_rets), len(spy_r))
spy_corr = np.corrcoef(ml_rets[:min_len], spy_r[:min_len])[0, 1]
print(f"SPY Correlation: {spy_corr:.3f}")

# ─── Save ───
output = {
    'strategy': 'ML Gold/Silver Ratio Trading',
    'type': 'precious metals relative value',
    'ml_metrics': ml_m,
    'baseline_metrics': base_m,
    'gold_metrics': gold_m,
    'silver_metrics': silver_m,
    'spy_metrics': spy_m,
    'yearly_returns': yearly,
    'negative_years': neg_years,
    'position_distribution': pos_counts.to_dict(),
    'adversarial': {
        'permutation': {'sharpe': real_sharpe, 'perm_mean': round(perm_mean, 3), 'p_value': round(perm_p, 3), 'verdict': 'PASS' if perm_pass else 'FAIL'},
        'sub_period': {'sharpes': sub_sharpes, 'cv': round(sub_cv, 3), 'verdict': 'PASS' if sub_pass else 'FAIL'},
        'outlier': {'full': real_sharpe, 'trimmed': round(trim_sharpe, 3), 'degradation': round(out_deg, 1), 'verdict': 'PASS' if out_pass else 'FAIL'},
        'r1_regime': {'green': round(g_sr, 3), 'red': round(r_sr, 3), 'gap': round(gap, 3), 'verdict': 'PASS' if r1_pass else 'FAIL'},
        'gates_passed': f'{gates}/4',
    },
    'spy_correlation': round(spy_corr, 3),
    'timestamp': datetime.now().isoformat(),
}

with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nSaved to {OUTPUT}")
print("DONE")
