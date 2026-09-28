#!/usr/bin/env python3
"""
ML Sector Spread Trading (Long/Short)
=======================================
HC #714: ML for exploratory research.

Concept: Long the top-3 sectors + short the bottom-3 sectors, ML-timed.
The ML model predicts whether sector momentum will persist or reverse.
- When persistence predicted: long winners, short losers (momentum)
- When reversal predicted: long losers, short winners (mean-reversion)

This is market-neutral (long/short) → near-zero beta → income strategy.

Universe: 11 SPDR sector ETFs.
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

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_sector_spread")
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML SECTOR SPREAD (LONG/SHORT)")
print("=" * 70)

# ─── Config ───
SECTOR_TICKERS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
BENCHMARK = 'SPY'
MACRO_TICKERS = ['^VIX', 'GLD', 'TLT', 'HYG', 'UUP', 'DBC']

TRAIN_DAYS = 252
ADVANCE_DAYS = 21
N_LONG = 3
N_SHORT = 3
N_PERMS = 200

# ─── Download ───
print("\nDownloading data...")
all_tickers = list(set(SECTOR_TICKERS + [BENCHMARK] + MACRO_TICKERS))
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

valid_sectors = [t for t in SECTOR_TICKERS if t in prices.columns]
print(f"Sectors: {len(valid_sectors)}: {valid_sectors}")

# ─── Features ───
print("Building features...")
features = pd.DataFrame(index=prices.index)

# Sector momentum & dispersion
for lookback in [5, 21, 63]:
    sector_mom = pd.DataFrame({s: prices[s].pct_change(lookback) for s in valid_sectors})
    features[f'sector_dispersion_{lookback}d'] = sector_mom.std(axis=1)
    features[f'sector_breadth_{lookback}d'] = (sector_mom > 0).sum(axis=1) / len(valid_sectors)
    # Cross-sectional momentum spread (top - bottom)
    features[f'sector_spread_{lookback}d'] = sector_mom.apply(lambda row: row.nlargest(N_LONG).mean() - row.nsmallest(N_SHORT).mean(), axis=1)

# Individual sector features
for s in valid_sectors:
    features[f'{s}_mom_21d'] = prices[s].pct_change(21)
    features[f'{s}_mom_63d'] = prices[s].pct_change(63)
    features[f'{s}_vol_21d'] = returns[s].rolling(21).std()
    # Relative strength vs SPY
    if BENCHMARK in prices.columns:
        features[f'{s}_rs_21d'] = prices[s].pct_change(21) - prices[BENCHMARK].pct_change(21)

# Macro
macro_names = [t.replace('^', '') for t in MACRO_TICKERS if t.replace('^', '') in prices.columns]
for m in macro_names:
    features[f'{m}_ret_21d'] = prices[m].pct_change(21)
    if m in returns.columns:
        features[f'{m}_vol_21d'] = returns[m].rolling(21).std()

if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_zscore'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()

features = features.dropna()
print(f"Features: {features.shape[1]} columns, {len(features)} days")

# ─── Walk-Forward ───
print("\nWalk-forward...")

# Label: does momentum persist? (top-3 sectors continue to outperform bottom-3?)
# 1 = momentum persists, 0 = reversal
rebal_indices = list(range(TRAIN_DAYS, len(features) - ADVANCE_DAYS, ADVANCE_DAYS))
print(f"Periods: {len(rebal_indices)}")

all_dates = []
all_ml_returns = []
all_mom_returns = []  # Pure momentum baseline
all_spy_returns = []

for idx_num, start_idx in enumerate(rebal_indices):
    train_start = max(0, start_idx - TRAIN_DAYS)
    train_end = start_idx
    fwd_start = start_idx
    fwd_end = min(start_idx + ADVANCE_DAYS, len(features))

    if fwd_end <= fwd_start:
        continue

    train_dates = features.index[train_start:train_end]
    fwd_dates = features.index[fwd_start:fwd_end]

    # Build training labels
    train_labels = []
    train_features = []
    for t_idx in range(len(train_dates) - ADVANCE_DAYS - 1):
        t_date = train_dates[t_idx]
        t_fwd_date = train_dates[min(t_idx + ADVANCE_DAYS, len(train_dates) - 1)]

        if t_date not in features.index or t_fwd_date not in prices.index:
            continue

        # Rank sectors by past 21d momentum at time t
        past_mom = {s: prices.loc[t_date, s] / prices.loc[prices.index[max(0, prices.index.get_loc(t_date) - 21)], s] - 1
                     for s in valid_sectors if t_date in prices.index}
        if len(past_mom) < N_LONG + N_SHORT:
            continue

        sorted_sectors = sorted(past_mom, key=past_mom.get, reverse=True)
        top = sorted_sectors[:N_LONG]
        bottom = sorted_sectors[-N_SHORT:]

        # Forward returns
        if t_fwd_date in prices.index:
            top_fwd = np.mean([prices.loc[t_fwd_date, s] / prices.loc[t_date, s] - 1 for s in top])
            bottom_fwd = np.mean([prices.loc[t_fwd_date, s] / prices.loc[t_date, s] - 1 for s in bottom])
            momentum_profit = top_fwd - bottom_fwd  # L/S portfolio

            label = 1 if momentum_profit > 0 else 0
            train_labels.append(label)
            train_features.append(features.loc[t_date].fillna(0).values)

    if len(train_features) < 30 or len(set(train_labels)) < 2:
        continue

    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)

    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, n_jobs=1
    )
    model.fit(X_train, y_train)

    # Current ranking
    rebal_date = features.index[start_idx]
    past_21d_start = prices.index[max(0, prices.index.get_loc(rebal_date) - 21)]
    curr_mom = {s: prices.loc[rebal_date, s] / prices.loc[past_21d_start, s] - 1
                for s in valid_sectors if rebal_date in prices.index}
    sorted_now = sorted(curr_mom, key=curr_mom.get, reverse=True)
    top_now = sorted_now[:N_LONG]
    bottom_now = sorted_now[-N_SHORT:]

    # Predict: momentum or reversal?
    pred_X = np.nan_to_num(features.loc[rebal_date].values.reshape(1, -1), nan=0, posinf=0, neginf=0)
    prob_momentum = model.predict_proba(pred_X)[0, 1]

    # ML decision
    if prob_momentum > 0.55:
        # Momentum: long winners, short losers
        long_sectors = top_now
        short_sectors = bottom_now
    elif prob_momentum < 0.45:
        # Reversal: long losers, short winners
        long_sectors = bottom_now
        short_sectors = top_now
    else:
        # Neutral: no trade
        long_sectors = []
        short_sectors = []

    for date in fwd_dates:
        if date not in returns.index:
            continue

        # ML L/S return
        if long_sectors and short_sectors:
            long_ret = np.mean([returns.loc[date, s] for s in long_sectors if s in returns.columns])
            short_ret = np.mean([returns.loc[date, s] for s in short_sectors if s in returns.columns])
            ml_ret = long_ret - short_ret
        else:
            ml_ret = 0

        # Pure momentum baseline (always long winners, short losers)
        mom_long_ret = np.mean([returns.loc[date, s] for s in top_now if s in returns.columns])
        mom_short_ret = np.mean([returns.loc[date, s] for s in bottom_now if s in returns.columns])
        mom_ret = mom_long_ret - mom_short_ret

        spy_ret = returns.loc[date, BENCHMARK] if BENCHMARK in returns.columns else 0

        all_dates.append(date)
        all_ml_returns.append(ml_ret)
        all_mom_returns.append(mom_ret)
        all_spy_returns.append(spy_ret)

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

ml_m = compute_metrics(all_ml_returns, 'ML Sector Spread')
mom_m = compute_metrics(all_mom_returns, 'Pure Momentum L/S')
spy_m = compute_metrics(all_spy_returns, 'SPY Buy-Hold')

print(f"\n{'='*70}")
for m in [ml_m, mom_m, spy_m]:
    print(f"\n{m['name']}: Sharpe={m.get('Sharpe','?')}, CAGR={m.get('CAGR','?')}%, MaxDD={m.get('MaxDD','?')}%")

# Adversarial
ml_rets = np.array(all_ml_returns, dtype=float)
ml_rets = ml_rets[~np.isnan(ml_rets)]
real_sharpe = ml_m['Sharpe']

perm_sharpes = []
for _ in range(N_PERMS):
    s = ml_rets.copy(); np.random.shuffle(s)
    pa = np.mean(s)*252; pv = np.std(s)*np.sqrt(252)
    perm_sharpes.append(pa/pv if pv>0 else 0)
perm_p = np.mean([s>=real_sharpe for s in perm_sharpes])
perm_pass = perm_p < 0.05

n = len(ml_rets)
sub_sharpes = []
for q in range(4):
    s,e = q*(n//4),(q+1)*(n//4) if q<3 else n
    qr = ml_rets[s:e]
    sub_sharpes.append(round(np.mean(qr)*252/(np.std(qr)*np.sqrt(252)) if np.std(qr)>0 else 0, 3))
sub_cv = np.std(sub_sharpes)/np.mean(sub_sharpes) if np.mean(sub_sharpes)>0 else 999
sub_pass = all(s>0 for s in sub_sharpes) and sub_cv < 1.0

p5,p95 = np.percentile(ml_rets,[5,95])
trimmed = ml_rets[(ml_rets>=p5)&(ml_rets<=p95)]
trim_sharpe = np.mean(trimmed)*252/(np.std(trimmed)*np.sqrt(252)) if np.std(trimmed)>0 else 0
out_pass = (trim_sharpe-real_sharpe)/abs(real_sharpe)*100 > -30 if real_sharpe!=0 else True

spy_r = np.array(all_spy_returns, dtype=float)
spy_21d = pd.Series(spy_r).rolling(21).sum().values
green = spy_21d > 0; red = spy_21d < 0
if green.sum()>20 and red.sum()>20:
    g_sr = np.mean(ml_rets[green])*252/(np.std(ml_rets[green])*np.sqrt(252)) if np.std(ml_rets[green])>0 else 0
    r_sr = np.mean(ml_rets[red])*252/(np.std(ml_rets[red])*np.sqrt(252)) if np.std(ml_rets[red])>0 else 0
    gap = abs(g_sr-r_sr)/max(abs(g_sr),abs(r_sr),0.01)
    r1_pass = gap < 0.50
else:
    g_sr,r_sr,gap = 0,0,999; r1_pass = False

gates = sum([perm_pass, sub_pass, out_pass, r1_pass])
min_len = min(len(ml_rets), len(spy_r))
spy_corr = np.corrcoef(ml_rets[:min_len], spy_r[:min_len])[0, 1]

print(f"\nADVERSARIAL: {gates}/4")
print(f"  Perm: p={perm_p:.3f} {'PASS' if perm_pass else 'FAIL'}")
print(f"  SubP: {sub_sharpes} CV={sub_cv:.3f} {'PASS' if sub_pass else 'FAIL'}")
print(f"  Outlier: {real_sharpe:.3f}->{trim_sharpe:.3f} {'PASS' if out_pass else 'FAIL'}")
print(f"  R1: g={g_sr:.3f} r={r_sr:.3f} gap={gap:.3f} {'PASS' if r1_pass else 'FAIL'}")
print(f"SPY Corr: {spy_corr:.3f}")

# Yearly
yearly = {'ML': {}}
for d, r in zip(all_dates, all_ml_returns):
    yr = d.year
    if yr not in yearly['ML']: yearly['ML'][yr] = []
    yearly['ML'][yr].append(r)
for yr in yearly['ML']:
    yearly['ML'][yr] = round((np.prod(1+np.array(yearly['ML'][yr]))-1)*100, 1)
neg_years = sum(1 for y in yearly['ML'].values() if y < 0)

# Save
output = {
    'strategy': 'ML Sector Spread (Long/Short)', 'type': 'market-neutral sector relative value',
    'ml_metrics': ml_m, 'momentum_metrics': mom_m, 'spy_metrics': spy_m,
    'yearly_returns': yearly, 'negative_years': neg_years,
    'adversarial': {
        'permutation': {'sharpe': real_sharpe, 'perm_mean': round(np.mean(perm_sharpes),3), 'p_value': round(perm_p,3), 'verdict': 'PASS' if perm_pass else 'FAIL'},
        'sub_period': {'sharpes': sub_sharpes, 'cv': round(sub_cv,3), 'verdict': 'PASS' if sub_pass else 'FAIL'},
        'outlier': {'full': real_sharpe, 'trimmed': round(trim_sharpe,3), 'verdict': 'PASS' if out_pass else 'FAIL'},
        'r1_regime': {'green': round(g_sr,3), 'red': round(r_sr,3), 'gap': round(gap,3), 'verdict': 'PASS' if r1_pass else 'FAIL'},
        'gates_passed': f'{gates}/4',
    },
    'spy_correlation': round(spy_corr, 3),
    'timestamp': datetime.now().isoformat(),
}
with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nSaved. DONE.")
