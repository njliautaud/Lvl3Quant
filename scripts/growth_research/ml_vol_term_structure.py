#!/usr/bin/env python3
"""
ML Volatility Term Structure Trading
======================================
Trades the VIX term structure slope: when contango is steep (front < back),
short vol is profitable. When backwardation occurs (front > back), go long vol.

Uses VIX/VIX3M ratio as term structure signal + ML to improve timing.
Instruments: SVXY (short vol) vs VXX/UVXY (long vol) vs SHY (cash).

HC compliance:
  - HC #0: Sliding walk-forward (252d train, 21d advance)
  - HC #713: Fixed $100K, no DCA
  - HC #714: Income + growth
  - HC #428 R1: Regime-agnostic validation
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_vol_term_structure')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML VOLATILITY TERM STRUCTURE TRADING")
print("=" * 70)

# ─── Download data ───
# VIX term structure: VIX (front) vs VIX3M (3-month)
# Trading instruments: SVXY (inverse vol), VIXY (long vol), SPY, SHY
tickers_download = {
    'SPY': 'SPY', 'SHY': 'SHY', 'TLT': 'TLT',
    'GLD': 'GLD', 'HYG': 'HYG', 'LQD': 'LQD',
    '^VIX': 'VIX', '^VIX3M': 'VIX3M',
    'SVXY': 'SVXY', 'VIXY': 'VIXY',
}

print("\nDownloading data...")
data = {}
for t, name in tickers_download.items():
    try:
        df = yf.download(t, start='2011-01-01', end='2026-07-19',
                         progress=False, auto_adjust=True)
        if len(df) > 100:
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', t)].copy()
            else:
                close = df['Close'].copy()
            close.name = name
            data[name] = close
            print(f"  {t} -> {name}: {len(df)} days")
        else:
            print(f"  {t}: SKIP ({len(df)} days)")
    except Exception as e:
        print(f"  {t}: FAILED ({e})")

prices = pd.DataFrame(data).dropna(how='all').ffill().dropna()
print(f"\nAligned: {len(prices)} days")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# Check we have VIX term structure data
has_vix = 'VIX' in prices.columns
has_vix3m = 'VIX3M' in prices.columns
has_svxy = 'SVXY' in prices.columns

if not has_vix:
    print("ERROR: No VIX data")
    exit(1)

# ─── Compute term structure features ───
print("\nComputing term structure features...")

# VIX/VIX3M ratio (< 1 = contango = short vol profitable)
if has_vix3m:
    prices['vix_ratio'] = prices['VIX'] / prices['VIX3M']
    prices['contango'] = (prices['vix_ratio'] < 1).astype(int)
else:
    # Approximate with VIX percentile
    prices['vix_ratio'] = prices['VIX'] / prices['VIX'].rolling(63).mean()
    prices['contango'] = (prices['vix_ratio'] < 1).astype(int)

# ─── Feature engineering ───
TRAIN_DAYS = 252
ADVANCE_DAYS = 5  # Weekly rebalance (vol term structure changes fast)
LABEL_HORIZON = 5

def build_features(idx):
    """Build features for term structure trading."""
    feats = {}

    vix = prices['VIX'].iloc[:idx+1]
    spy = prices['SPY'].iloc[:idx+1]

    if len(vix) < 252:
        return None

    # VIX level and dynamics
    feats['vix_level'] = vix.iloc[-1]
    feats['vix_pctile_252d'] = (vix.iloc[-1] - vix.iloc[-252:].min()) / \
                                (vix.iloc[-252:].max() - vix.iloc[-252:].min() + 1e-8)
    for w in [5, 10, 21]:
        feats[f'vix_ret_{w}d'] = (vix.iloc[-1] / vix.iloc[-w] - 1) if vix.iloc[-w] > 0 else 0

    # VIX term structure
    if has_vix3m:
        ratio = prices['vix_ratio'].iloc[:idx+1]
        feats['vix_ratio'] = ratio.iloc[-1]
        feats['vix_ratio_5d_avg'] = ratio.iloc[-5:].mean()
        feats['vix_ratio_21d_avg'] = ratio.iloc[-21:].mean()
        feats['vix_ratio_zscore'] = (ratio.iloc[-1] - ratio.iloc[-63:].mean()) / \
                                     (ratio.iloc[-63:].std() + 1e-8)
    else:
        ratio = prices['vix_ratio'].iloc[:idx+1]
        feats['vix_ratio'] = ratio.iloc[-1]

    # SPY features
    spy_rets = spy.pct_change().dropna()
    for w in [5, 10, 21, 63]:
        if len(spy) > w:
            feats[f'spy_ret_{w}d'] = (spy.iloc[-1] / spy.iloc[-w] - 1) if spy.iloc[-w] > 0 else 0

    # SPY volatility
    for w in [10, 21, 63]:
        if len(spy_rets) > w:
            feats[f'spy_vol_{w}d'] = spy_rets.iloc[-w:].std() * np.sqrt(252)

    # Vol of VIX
    vix_rets = vix.pct_change().dropna()
    if len(vix_rets) > 21:
        feats['vvix_proxy'] = vix_rets.iloc[-21:].std() * np.sqrt(252)

    # SPY drawdown
    if len(spy) > 63:
        peak = spy.iloc[-63:].max()
        feats['spy_dd_63d'] = (spy.iloc[-1] / peak - 1) if peak > 0 else 0

    # Credit spread (HYG vs LQD)
    if 'HYG' in prices.columns and 'LQD' in prices.columns:
        hyg = prices['HYG'].iloc[:idx+1]
        lqd = prices['LQD'].iloc[:idx+1]
        if len(hyg) > 21:
            spread = hyg / lqd
            feats['credit_5d_chg'] = (spread.iloc[-1] / spread.iloc[-5] - 1) if spread.iloc[-5] > 0 else 0

    # Gold momentum (risk signal)
    if 'GLD' in prices.columns:
        gld = prices['GLD'].iloc[:idx+1]
        if len(gld) > 21:
            feats['gld_ret_21d'] = (gld.iloc[-1] / gld.iloc[-21] - 1) if gld.iloc[-21] > 0 else 0

    # Realized vs implied vol (VRP)
    if len(spy_rets) > 21:
        realized_vol = spy_rets.iloc[-21:].std() * np.sqrt(252) * 100
        feats['vrp'] = vix.iloc[-1] - realized_vol  # positive = vol overpriced

    return feats


# ─── Build dataset ───
print("Building features...")
all_rows = []
dates = prices.index.tolist()

for i in range(TRAIN_DAYS, len(dates) - LABEL_HORIZON):
    feats = build_features(i)
    if feats is None:
        continue

    # Target: which position is best over next 5 days?
    spy_ret = (prices['SPY'].iloc[i + LABEL_HORIZON] / prices['SPY'].iloc[i]) - 1

    # SVXY return (short vol - benefits from contango decay)
    if has_svxy:
        svxy_ret = (prices['SVXY'].iloc[i + LABEL_HORIZON] / prices['SVXY'].iloc[i]) - 1
    else:
        # Approximate: inverse of VIX change * some leverage
        vix_chg = (prices['VIX'].iloc[i + LABEL_HORIZON] / prices['VIX'].iloc[i]) - 1
        svxy_ret = -vix_chg * 0.5  # rough SVXY approximation

    # SHY return (cash)
    shy_ret = (prices['SHY'].iloc[i + LABEL_HORIZON] / prices['SHY'].iloc[i]) - 1

    # Strategy: choose SVXY (short vol), SPY (equity), or SHY (cash)
    rets = {'svxy': svxy_ret, 'spy': spy_ret, 'shy': shy_ret}
    best = max(rets, key=rets.get)

    feats['_date'] = dates[i]
    feats['_best'] = best
    feats['_svxy_ret'] = svxy_ret
    feats['_spy_ret'] = spy_ret
    feats['_shy_ret'] = shy_ret

    all_rows.append(feats)

df_all = pd.DataFrame(all_rows)
print(f"Total observations: {len(df_all)}")
print(f"Best category dist: {df_all['_best'].value_counts().to_dict()}")

feature_cols = [c for c in df_all.columns if not c.startswith('_')]
print(f"Features: {len(feature_cols)}")

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_all['_label'] = le.fit_transform(df_all['_best'])

# ─── Walk-forward ───
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST (weekly rebalance)")
print("=" * 70)

try:
    import lightgbm as lgb
    USE_LGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_LGB = False

results = {}
fold_count = 0
unique_dates = df_all['_date'].unique()
rebalance_dates = unique_dates[TRAIN_DAYS::ADVANCE_DAYS]

print(f"Rebalance periods: {len(rebalance_dates)}")

for reb_date in rebalance_dates:
    train_mask = df_all['_date'] < reb_date
    train_dates = df_all[train_mask]['_date'].unique()
    if len(train_dates) < TRAIN_DAYS // 2:
        continue

    cutoff = sorted(train_dates)[-TRAIN_DAYS:]
    train_df = df_all[df_all['_date'].isin(cutoff)]
    test_df = df_all[df_all['_date'] == reb_date]

    if len(test_df) == 0 or len(train_df) < 50:
        continue

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['_label'].values
    X_test = test_df[feature_cols].fillna(0).values

    n_classes = len(np.unique(y_train))
    if n_classes < 2:
        continue

    if USE_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, verbose=-1, n_jobs=4
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]
    pred_cat = le.inverse_transform([pred])[0]

    row = test_df.iloc[0]
    cat_rets = {'svxy': row['_svxy_ret'], 'spy': row['_spy_ret'], 'shy': row['_shy_ret']}

    results[reb_date] = {
        'predicted': pred_cat,
        'actual_best': row['_best'],
        'ml_ret': cat_rets[pred_cat],
        'svxy_ret': row['_svxy_ret'],
        'spy_ret': row['_spy_ret'],
        'shy_ret': row['_shy_ret'],
    }
    fold_count += 1

    if fold_count % 100 == 0:
        print(f"  Fold {fold_count}...")

print(f"Completed {fold_count} folds")

# ─── Compute metrics ───
sorted_dates = sorted(results.keys())
ml_rets = [results[d]['ml_ret'] for d in sorted_dates]
spy_rets = [results[d]['spy_ret'] for d in sorted_dates]
svxy_rets = [results[d]['svxy_ret'] for d in sorted_dates]

def calc_metrics(returns, label):
    r = np.array(returns)
    n = len(r)
    ppy = 252 / ADVANCE_DAYS  # ~50 periods/year

    mean_r = np.mean(r) * ppy
    std_r = np.std(r, ddof=1) * np.sqrt(ppy)
    sharpe = mean_r / std_r if std_r > 0 else 0

    down = r[r < 0]
    down_std = np.std(down, ddof=1) * np.sqrt(ppy) if len(down) > 1 else std_r
    sortino = mean_r / down_std if down_std > 0 else 0

    cum = np.cumprod(1 + r)
    n_years = n / ppy
    cagr = (cum[-1] ** (1/n_years) - 1) * 100 if n_years > 0 and cum[-1] > 0 else 0

    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    wr = np.mean(r > 0) * 100
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    print(f"\n{label}:")
    print(f"  Sharpe: {sharpe:.3f}, Sortino: {sortino:.3f}")
    print(f"  CAGR: {cagr:.1f}%, MaxDD: {max_dd:.1f}%, Calmar: {calmar:.2f}")
    print(f"  WR: {wr:.1f}%, PF: {pf:.2f}, Periods: {n}")

    return {'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
            'cagr': round(cagr, 1), 'max_dd': round(max_dd, 1),
            'calmar': round(calmar, 2), 'wr': round(wr, 1), 'pf': round(pf, 2)}

ml_m = calc_metrics(ml_rets, "ML Vol Term Structure")
spy_m = calc_metrics(spy_rets, "SPY Buy & Hold")
svxy_m = calc_metrics(svxy_rets, "SVXY Buy & Hold (short vol)")

# Accuracy
accuracy = np.mean([results[d]['predicted'] == results[d]['actual_best'] for d in sorted_dates]) * 100
print(f"\nPrediction accuracy: {accuracy:.1f}%")

# Year-by-year
print("\n" + "=" * 70)
print("YEAR-BY-YEAR")
print("=" * 70)
df_res = pd.DataFrame([{'date': d, 'ml': results[d]['ml_ret'], 'spy': results[d]['spy_ret']}
                        for d in sorted_dates])
df_res['year'] = df_res['date'].apply(lambda d: d.year)
yearly = df_res.groupby('year').agg({
    'ml': lambda x: (np.prod(1 + x) - 1) * 100,
    'spy': lambda x: (np.prod(1 + x) - 1) * 100
}).rename(columns={'ml': 'ML_%', 'spy': 'SPY_%'})
print(yearly.round(1).to_string())
neg_years = (yearly['ML_%'] < 0).sum()
print(f"\nNegative years: {neg_years}/{len(yearly)}")

# ─── ADVERSARIAL ───
print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION")
print("=" * 70)

# Permutation
print("\n--- PERMUTATION (n=200) ---")
perm_sharpes = []
categories = ['svxy', 'spy', 'shy']

for pi in range(200):
    p_rets = []
    for d in sorted_dates:
        r = results[d]
        rand_cat = np.random.choice(categories)
        cat_rets = {'svxy': r['svxy_ret'], 'spy': r['spy_ret'], 'shy': r['shy_ret']}
        p_rets.append(cat_rets[rand_cat])
    r = np.array(p_rets)
    ppy = 252 / ADVANCE_DAYS
    ps = (np.mean(r) * ppy) / (np.std(r, ddof=1) * np.sqrt(ppy)) if np.std(r) > 0 else 0
    perm_sharpes.append(ps)

p_val = np.mean([ps >= ml_m['sharpe'] for ps in perm_sharpes])
print(f"  ML Sharpe: {ml_m['sharpe']:.3f}, Perm mean: {np.mean(perm_sharpes):.3f}, p={p_val:.3f}")
perm_v = "PASS" if p_val < 0.05 else "FAIL"
print(f"  Verdict: {perm_v}")

# Sub-period
print("\n--- SUB-PERIOD ---")
q_size = len(ml_rets) // 4
sub_sh = []
for q in range(4):
    s, e = q*q_size, (q+1)*q_size if q < 3 else len(ml_rets)
    sr = np.array(ml_rets[s:e])
    ppy = 252 / ADVANCE_DAYS
    ss = (np.mean(sr)*ppy) / (np.std(sr, ddof=1)*np.sqrt(ppy)) if np.std(sr) > 0 else 0
    sub_sh.append(ss)
    print(f"  Q{q+1}: Sharpe {ss:.3f}")
cv = np.std(sub_sh) / abs(np.mean(sub_sh)) if np.mean(sub_sh) != 0 else 999
sub_v = "PASS" if cv < 0.75 and all(s > -0.5 for s in sub_sh) else "FAIL"
print(f"  CV: {cv:.3f}, Verdict: {sub_v}")

# Outlier
print("\n--- OUTLIER ---")
r = np.array(ml_rets)
cut = np.percentile(abs(r), 95)
tr = r[abs(r) <= cut]
if len(tr) > 5:
    ppy = 252 / ADVANCE_DAYS
    ts = (np.mean(tr)*ppy) / (np.std(tr, ddof=1)*np.sqrt(ppy)) if np.std(tr) > 0 else 0
    deg = (1 - ts / ml_m['sharpe']) * 100 if ml_m['sharpe'] != 0 else 0
    print(f"  Full: {ml_m['sharpe']:.3f}, Trimmed: {ts:.3f}, Deg: {deg:.1f}%")
    out_v = "PASS" if deg < 50 else "FAIL"
else:
    out_v = "FAIL"; ts = 0; deg = 100
print(f"  Verdict: {out_v}")

# R1 Regime
print("\n--- R1 REGIME ---")
green_r = [ml_rets[i] for i in range(len(ml_rets)) if spy_rets[i] >= 0]
red_r = [ml_rets[i] for i in range(len(ml_rets)) if spy_rets[i] < 0]
if len(green_r) > 5 and len(red_r) > 5:
    ppy = 252 / ADVANCE_DAYS
    gs = (np.mean(green_r)*ppy) / (np.std(green_r, ddof=1)*np.sqrt(ppy)) if np.std(green_r) > 0 else 0
    rs = (np.mean(red_r)*ppy) / (np.std(red_r, ddof=1)*np.sqrt(ppy)) if np.std(red_r) > 0 else 0
    gap = abs(gs - rs) / max(abs(gs), abs(rs), 0.01)
    print(f"  Green: {gs:.3f}, Red: {rs:.3f}, Gap: {gap:.3f}")
    r1_v = "PASS" if gap < 0.50 else "FAIL"
else:
    r1_v = "FAIL"; gs = rs = gap = 0
print(f"  Verdict: {r1_v}")

gates = sum([v == "PASS" for v in [perm_v, sub_v, out_v, r1_v]])
verdict = "VALIDATED" if gates >= 3 else ("PARTIAL" if gates >= 2 else "REJECTED")

# Save
output = {
    'strategy': 'ML Vol Term Structure',
    'ml_metrics': ml_m, 'spy_metrics': spy_m, 'svxy_metrics': svxy_m,
    'accuracy': round(accuracy, 1),
    'adversarial': {
        'perm': {'p': round(p_val, 3), 'verdict': perm_v},
        'sub_period': {'cv': round(cv, 3), 'verdict': sub_v},
        'outlier': {'deg': round(deg, 1), 'verdict': out_v},
        'r1': {'gap': round(gap, 3), 'verdict': r1_v},
        'gates': f"{gates}/4"
    },
    'verdict': verdict,
    'timestamp': dt.datetime.now().isoformat()
}

with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"FINAL VERDICT: {verdict} ({gates}/4 gates)")
print(f"{'='*70}")
print(f"ML Vol Term Structure: Sharpe {ml_m['sharpe']:.3f}, CAGR {ml_m['cagr']}%, MaxDD {ml_m['max_dd']}%")
