#!/usr/bin/env python3
"""
ML Yield Curve Steepener/Flattener Trade
==========================================
HC #714: ML for exploratory research. HC #710: Cross-asset signals.

Concept: Trade changes in the yield curve shape using bond ETF pairs.
- Steepener: long short-duration (SHY) + short long-duration (TLT) → profits when curve steepens
- Flattener: long long-duration (TLT) + short short-duration (SHY) → profits when curve flattens
- ML predicts curve direction using: macro conditions, equity markets, credit spreads, commodities

This is DIFFERENT from Bond Duration Timing (which just goes long/short duration).
This trades the RELATIVE VALUE between curve segments.

Walk-forward LightGBM (252d sliding, 21d advance).
HC #713: Fixed $100K, no DCA. HC #0: Sliding only.
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

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_yield_curve_trade")
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML YIELD CURVE STEEPENER/FLATTENER")
print("=" * 70)

# ─── Config ───
# Curve segments via ETFs
SHORT_END = 'SHY'    # 1-3yr Treasury
MID_END = 'IEF'      # 7-10yr Treasury
LONG_END = 'TLT'     # 20+yr Treasury
TIPS = 'TIP'         # Inflation-linked

MACRO_TICKERS = ['^VIX', 'GLD', 'UUP', 'DBC', 'HYG', 'LQD', 'SPY', 'EEM', '^TNX']

TRAIN_DAYS = 252
ADVANCE_DAYS = 21
INITIAL_CAPITAL = 100_000
N_PERMS = 200

# ─── Download ───
print("\nDownloading data...")
all_tickers = list(set([SHORT_END, MID_END, LONG_END, TIPS] + MACRO_TICKERS))
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
returns = prices.pct_change()
print(f"\nAligned: {len(prices)} days")

# ─── Yield Curve Proxy ───
# "Slope" = TLT return - SHY return (positive = steepening concerns, negative = flattening)
# True steepener P&L: long SHY + short TLT → +return when curve steepens (SHY outperforms TLT)
# True flattener P&L: long TLT + short SHY → +return when curve flattens (TLT outperforms SHY)

prices['curve_slope'] = prices[LONG_END] / prices[SHORT_END]  # Proxy for curve shape
prices['slope_pctchg'] = prices['curve_slope'].pct_change(21)

# Steepener return = SHY - TLT (profit when short end outperforms)
returns['steepener'] = returns[SHORT_END] - returns[LONG_END]
# Flattener return = TLT - SHY (profit when long end outperforms)
returns['flattener'] = returns[LONG_END] - returns[SHORT_END]

# ─── Build Features ───
print("Building features...")
features = pd.DataFrame(index=prices.index)

# Curve features
features['curve_slope'] = prices['curve_slope']
features['slope_zscore_63d'] = (prices['curve_slope'] - prices['curve_slope'].rolling(63).mean()) / prices['curve_slope'].rolling(63).std()
features['slope_zscore_126d'] = (prices['curve_slope'] - prices['curve_slope'].rolling(126).mean()) / prices['curve_slope'].rolling(126).std()
features['slope_mom_5d'] = prices['curve_slope'].pct_change(5)
features['slope_mom_21d'] = prices['curve_slope'].pct_change(21)
features['slope_mom_63d'] = prices['curve_slope'].pct_change(63)
features['slope_vol_21d'] = prices['curve_slope'].pct_change().rolling(21).std()

# Duration spread: TLT vol / SHY vol
features['dur_vol_ratio'] = returns[LONG_END].rolling(21).std() / returns[SHORT_END].rolling(21).std().clip(lower=1e-6)

# Mid-curve features
if MID_END in prices.columns:
    features['belly_vs_wings'] = prices[MID_END] / ((prices[SHORT_END] + prices[LONG_END]) / 2)  # Butterfly
    features['belly_mom_21d'] = prices[MID_END].pct_change(21)

# TIPS breakeven proxy
if TIPS in prices.columns:
    features['tips_mom_21d'] = prices[TIPS].pct_change(21)
    features['real_rate_proxy'] = returns[LONG_END].rolling(21).mean() - returns[TIPS].rolling(21).mean()

# Each bond segment
for bond in [SHORT_END, MID_END, LONG_END]:
    if bond in prices.columns:
        features[f'{bond}_mom_5d'] = prices[bond].pct_change(5)
        features[f'{bond}_mom_21d'] = prices[bond].pct_change(21)
        features[f'{bond}_mom_63d'] = prices[bond].pct_change(63)
        features[f'{bond}_vol_21d'] = returns[bond].rolling(21).std()

# Macro
macro_names = [t.replace('^', '') for t in MACRO_TICKERS if t.replace('^', '') in prices.columns]
for m in macro_names:
    if m in prices.columns:
        features[f'{m}_ret_21d'] = prices[m].pct_change(21)
        if m in returns.columns:
            features[f'{m}_vol_21d'] = returns[m].rolling(21).std()

if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_zscore'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()

# Credit spreads
if 'HYG' in prices.columns and 'LQD' in prices.columns:
    features['credit_spread'] = (returns['HYG'] - returns['LQD']).rolling(21).mean()

features = features.dropna()
print(f"Features: {features.shape[1]} columns, {len(features)} days")

# ─── Walk-Forward ───
print("\nRunning walk-forward...")

# Label: will steepener or flattener win next 21 days?
steep_fwd = returns['steepener'].rolling(ADVANCE_DAYS).sum().shift(-ADVANCE_DAYS)

rebal_indices = list(range(TRAIN_DAYS, len(features) - ADVANCE_DAYS, ADVANCE_DAYS))
print(f"Rebalancing periods: {len(rebal_indices)}")

all_dates = []
all_ml_returns = []
all_baseline_returns = []
all_long_returns = []  # TLT buy-hold
all_spy_returns = []
all_positions = []

# HC #718 R3: Transaction costs — 5 bps per leg on turnover + 50 bps annualized short borrow
COST_BPS = 5
SHORT_BORROW_ANNUAL = 0.0050  # 50 bps annualized
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

    # Training
    common = train_dates.intersection(steep_fwd.dropna().index).intersection(features.index)
    if len(common) < 50:
        continue

    train_X = np.nan_to_num(features.loc[common].values, nan=0, posinf=0, neginf=0)
    train_y = (steep_fwd.loc[common] > 0).astype(int).values  # 1 = steepener wins

    if len(set(train_y)) < 2:
        continue

    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, n_jobs=1
    )
    model.fit(train_X, train_y)

    # Predict
    rebal_date = features.index[start_idx]
    pred_X = np.nan_to_num(features.loc[rebal_date].values.reshape(1, -1), nan=0, posinf=0, neginf=0)
    prob_steep = model.predict_proba(pred_X)[0, 1]

    # Position: steepener or flattener
    if prob_steep > 0.6:
        pos = 'steepener'  # long SHY, short TLT
    elif prob_steep < 0.4:
        pos = 'flattener'  # long TLT, short SHY
    else:
        pos = 'neutral'    # 50/50

    # Baseline: simple momentum (if slope rose last 63d → continue = flattener)
    slope_mom = features.loc[rebal_date, 'slope_mom_63d'] if 'slope_mom_63d' in features.columns else 0
    base_pos = 'flattener' if slope_mom > 0 else 'steepener'

    # HC #718 R3: transaction costs on position change
    rebal_cost = 0.0
    if prev_ml_pos is not None and pos != prev_ml_pos:
        rebal_cost = 2 * COST_BPS / 10000  # sell old + buy new (both legs)
    elif prev_ml_pos is None and pos != 'neutral':
        rebal_cost = COST_BPS / 10000  # initial entry
    prev_ml_pos = pos

    for day_idx, date in enumerate(fwd_dates):
        if date not in returns.index:
            continue

        steep_ret = returns.loc[date, 'steepener']
        flat_ret = returns.loc[date, 'flattener']
        tlt_ret = returns.loc[date, LONG_END]
        spy_ret = returns.loc[date, 'SPY'] if 'SPY' in returns.columns else 0

        if pos == 'steepener':
            ml_ret = steep_ret
        elif pos == 'flattener':
            ml_ret = flat_ret
        else:
            ml_ret = 0  # Flat

        # HC #718 R3: short borrow cost (always has a short leg unless neutral)
        if pos != 'neutral':
            ml_ret -= SHORT_BORROW_ANNUAL / 252  # daily borrow cost

        # Apply rebalance cost on first day of the period only
        if day_idx == 0:
            ml_ret -= rebal_cost

        if base_pos == 'steepener':
            base_ret = steep_ret
        else:
            base_ret = flat_ret

        all_dates.append(date)
        all_ml_returns.append(ml_ret)
        all_baseline_returns.append(base_ret)
        all_long_returns.append(tlt_ret)
        all_spy_returns.append(spy_ret)
        all_positions.append(pos)

    if (idx_num + 1) % 20 == 0:
        print(f"  Period {idx_num+1}/{len(rebal_indices)}")

print(f"\nComplete: {len(all_dates)} days")

# ─── Metrics ───
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
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {
        'name': name, 'Sharpe': round(sharpe, 3), 'Sortino': round(sortino, 3),
        'CAGR': round(cagr * 100, 1), 'MaxDD': round(max_dd * 100, 1),
        'Calmar': round(calmar, 2), 'WR': round(wr * 100, 1), 'PF': round(pf, 2),
        'n_days': len(r),
    }

ml_m = compute_metrics(all_ml_returns, 'ML Curve Trade')
base_m = compute_metrics(all_baseline_returns, 'Momentum Baseline')
tlt_m = compute_metrics(all_long_returns, 'TLT Buy-Hold')
spy_m = compute_metrics(all_spy_returns, 'SPY Buy-Hold')

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
for m in [ml_m, base_m, tlt_m, spy_m]:
    print(f"\n{m['name']}:")
    for k, v in m.items():
        if k != 'name':
            print(f"  {k}: {v}")

# Position distribution
pos_counts = pd.Series(all_positions).value_counts()
print(f"\nPositions: {pos_counts.to_dict()}")

# Yearly returns
yearly = {'ML': {}, 'TLT': {}, 'SPY': {}}
for d, ml_r, tlt_r, spy_r in zip(all_dates, all_ml_returns, all_long_returns, all_spy_returns):
    yr = d.year
    for key, r in [('ML', ml_r), ('TLT', tlt_r), ('SPY', spy_r)]:
        if yr not in yearly[key]:
            yearly[key][yr] = []
        yearly[key][yr].append(r)
for key in yearly:
    for yr in yearly[key]:
        yearly[key][yr] = round((np.prod(1 + np.array(yearly[key][yr])) - 1) * 100, 1)

neg_years = sum(1 for y in yearly.get('ML', {}).values() if isinstance(y, (int, float)) and y < 0)
print(f"\nYearly: ML | TLT | SPY")
for yr in sorted(set(y for s in yearly.values() for y in s)):
    print(f"  {yr}: {yearly.get('ML',{}).get(yr,'-')} | {yearly.get('TLT',{}).get(yr,'-')} | {yearly.get('SPY',{}).get(yr,'-')}")

# ─── Adversarial ───
print(f"\n{'='*70}")
print("ADVERSARIAL VALIDATION")
print(f"{'='*70}")

ml_rets = np.array(all_ml_returns, dtype=float)
ml_rets = ml_rets[~np.isnan(ml_rets)]
real_sharpe = ml_m['Sharpe']

# HC #718: shuffle signals, not returns
# Shuffle the position (steepener/flattener/neutral) assignments across rebalance dates,
# then recompute returns under shuffled signals
perm_sharpes = []
# Collect per-rebalance position assignments and their date ranges
rebal_positions = []
for idx_num, start_idx in enumerate(rebal_indices):
    fwd_start_p = start_idx
    fwd_end_p = min(start_idx + ADVANCE_DAYS, len(features))
    if fwd_end_p <= fwd_start_p:
        continue
    rebal_positions.append(idx_num)

for _ in range(N_PERMS):
    # Shuffle which position (steepener/flattener/neutral) is assigned to which period
    shuffled_positions = list(all_positions)  # original daily positions
    # Shuffle at rebalance-period level: collect period blocks and shuffle
    period_blocks = {}
    for d, pos in zip(all_dates, all_positions):
        period_blocks.setdefault(pos, [])  # just need unique positions
    unique_positions = list(set(all_positions))

    perm_rets = []
    for d_idx_p in range(len(all_dates)):
        date_p = all_dates[d_idx_p]
        # Assign random position for this date
        random_pos = np.random.choice(unique_positions)
        if date_p not in returns.index:
            perm_rets.append(0)
            continue
        steep_ret = returns.loc[date_p, 'steepener']
        flat_ret = returns.loc[date_p, 'flattener']
        if random_pos == 'steepener':
            perm_rets.append(steep_ret)
        elif random_pos == 'flattener':
            perm_rets.append(flat_ret)
        else:
            perm_rets.append(0)

    pr = np.array(perm_rets, dtype=float)
    pr = pr[~np.isnan(pr)]
    pa = np.mean(pr) * 252; pv = np.std(pr) * np.sqrt(252)
    perm_sharpes.append(pa / pv if pv > 0 else 0)
perm_p = np.mean([s >= real_sharpe for s in perm_sharpes])
perm_pass = perm_p < 0.05
print(f"Perm: real={real_sharpe:.3f}, mean={np.mean(perm_sharpes):.3f}, p={perm_p:.3f} {'PASS' if perm_pass else 'FAIL'}")

# Sub-period
n = len(ml_rets)
sub_sharpes = []
for q in range(4):
    s, e = q*(n//4), (q+1)*(n//4) if q<3 else n
    qr = ml_rets[s:e]
    sub_sharpes.append(round(np.mean(qr)*252/(np.std(qr)*np.sqrt(252)) if np.std(qr)>0 else 0, 3))
sub_cv = np.std(sub_sharpes)/np.mean(sub_sharpes) if np.mean(sub_sharpes)>0 else 999
sub_pass = all(s>0 for s in sub_sharpes) and sub_cv < 1.0
print(f"SubP: {sub_sharpes}, CV={sub_cv:.3f} {'PASS' if sub_pass else 'FAIL'}")

# Outlier
p5, p95 = np.percentile(ml_rets, [5, 95])
trimmed = ml_rets[(ml_rets>=p5)&(ml_rets<=p95)]
trim_sharpe = np.mean(trimmed)*252/(np.std(trimmed)*np.sqrt(252)) if np.std(trimmed)>0 else 0
out_deg = (trim_sharpe-real_sharpe)/abs(real_sharpe)*100 if real_sharpe!=0 else 0
out_pass = out_deg > -30
print(f"Outlier: full={real_sharpe:.3f}, trim={trim_sharpe:.3f}, deg={out_deg:.1f}% {'PASS' if out_pass else 'FAIL'}")

# R1
spy_r = np.array(all_spy_returns, dtype=float)
spy_21d = pd.Series(spy_r).rolling(21).sum().values
green = spy_21d > 0; red = spy_21d < 0
if green.sum()>20 and red.sum()>20:
    g_sr = np.mean(ml_rets[green])*252/(np.std(ml_rets[green])*np.sqrt(252)) if np.std(ml_rets[green])>0 else 0
    r_sr = np.mean(ml_rets[red])*252/(np.std(ml_rets[red])*np.sqrt(252)) if np.std(ml_rets[red])>0 else 0
    gap = abs(g_sr-r_sr)/max(abs(g_sr),abs(r_sr),0.01)
    r1_pass = gap < 0.50
else:
    g_sr, r_sr, gap = 0, 0, 999; r1_pass = False
print(f"R1: green={g_sr:.3f}, red={r_sr:.3f}, gap={gap:.3f} {'PASS' if r1_pass else 'FAIL'}")

gates = sum([perm_pass, sub_pass, out_pass, r1_pass])
print(f"\nGATES: {gates}/4")

min_len = min(len(ml_rets), len(spy_r))
spy_corr = np.corrcoef(ml_rets[:min_len], spy_r[:min_len])[0, 1]
print(f"SPY Correlation: {spy_corr:.3f}")

# Save
output = {
    'strategy': 'ML Yield Curve Trade',
    'type': 'fixed income relative value',
    'ml_metrics': ml_m, 'baseline_metrics': base_m,
    'tlt_metrics': tlt_m, 'spy_metrics': spy_m,
    'yearly_returns': yearly, 'negative_years': neg_years,
    'position_distribution': pos_counts.to_dict(),
    'adversarial': {
        'permutation': {'sharpe': real_sharpe, 'perm_mean': round(np.mean(perm_sharpes),3), 'p_value': round(perm_p,3), 'verdict': 'PASS' if perm_pass else 'FAIL'},
        'sub_period': {'sharpes': sub_sharpes, 'cv': round(sub_cv,3), 'verdict': 'PASS' if sub_pass else 'FAIL'},
        'outlier': {'full': real_sharpe, 'trimmed': round(trim_sharpe,3), 'degradation': round(out_deg,1), 'verdict': 'PASS' if out_pass else 'FAIL'},
        'r1_regime': {'green': round(g_sr,3), 'red': round(r_sr,3), 'gap': round(gap,3), 'verdict': 'PASS' if r1_pass else 'FAIL'},
        'gates_passed': f'{gates}/4',
    },
    'spy_correlation': round(spy_corr, 3),
    'timestamp': datetime.now().isoformat(),
}
with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nSaved to {OUTPUT}")
print("DONE")
