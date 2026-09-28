#!/usr/bin/env python3
"""
ML Dynamic Collar — Protective Overlay for Growth Portfolio
=============================================================
Long SPY + ML-timed protective puts + covered calls.
ML decides when to buy puts (high crash risk) vs sell calls (low vol)
vs do nothing (normal regime). Monetizes vol surface dynamics.

Purpose: Reduce portfolio drawdowns while maintaining upside.
This is a HEDGE/OVERLAY strategy, not standalone alpha.

HC compliance:
  - HC #0: Sliding walk-forward
  - HC #713: Fixed $100K, no DCA
  - HC #709: Growth with drawdown protection
  - HC #714: Income + growth
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_dynamic_collar')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML DYNAMIC COLLAR — PROTECTIVE OVERLAY")
print("=" * 70)

# ─── Download data ───
tickers = {
    'SPY': 'SPY', '^VIX': 'VIX', '^VIX3M': 'VIX3M',
    'TLT': 'TLT', 'GLD': 'GLD', 'HYG': 'HYG', 'LQD': 'LQD',
    'SHY': 'SHY', 'USO': 'USO', 'EEM': 'EEM'
}

print("\nDownloading data...")
data = {}
for t, name in tickers.items():
    try:
        df = yf.download(t, start='2006-01-01', end='2026-07-19',
                         progress=False, auto_adjust=True)
        if len(df) > 100:
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', t)].copy()
            else:
                close = df['Close'].copy()
            close.name = name
            data[name] = close
            print(f"  {t}: {len(df)} days")
    except Exception as e:
        print(f"  {t}: FAILED ({e})")

prices = pd.DataFrame(data).dropna(how='all').ffill().dropna()
print(f"\nAligned: {len(prices)} days ({prices.index[0].date()} to {prices.index[-1].date()})")

# ─── Simulate collar economics ───
# Protective put cost: approximated from VIX
# Covered call premium: approximated from VIX
# Dynamic: ML decides regime (protection/income/neutral)

TRAIN_DAYS = 252
ADVANCE_DAYS = 21  # Monthly rebalance
LABEL_HORIZON = 21
N_PERMS = 200

def estimate_put_cost(vix_level, days_to_expiry=21, moneyness=0.95):
    """Approximate cost of 5% OTM put as fraction of notional.
    Uses simplified BS-like approximation. Per HC #713 R2: clearly labeled as approximate.
    """
    time_factor = np.sqrt(days_to_expiry / 252)
    # Approximate: put cost ~ vix * time_factor * adjustment for OTM
    otm_adj = np.exp(-((1 - moneyness) / (vix_level/100 * time_factor + 1e-6))**2 / 2)
    cost = vix_level / 100 * time_factor * otm_adj * 0.5  # 50% of ATM as rough OTM
    return max(cost, 0.001)  # minimum 0.1%

def estimate_call_premium(vix_level, days_to_expiry=21, moneyness=1.05):
    """Approximate premium received from 5% OTM covered call."""
    time_factor = np.sqrt(days_to_expiry / 252)
    otm_adj = np.exp(-((moneyness - 1) / (vix_level/100 * time_factor + 1e-6))**2 / 2)
    premium = vix_level / 100 * time_factor * otm_adj * 0.4
    return max(premium, 0.0005)


def build_features(idx):
    """Build regime features."""
    feats = {}
    spy = prices['SPY'].iloc[:idx+1]
    vix = prices['VIX'].iloc[:idx+1] if 'VIX' in prices.columns else None

    if len(spy) < 252:
        return None

    spy_rets = spy.pct_change().dropna()

    # SPY momentum
    for w in [5, 10, 21, 63, 126, 252]:
        if len(spy) > w:
            feats[f'spy_ret_{w}d'] = (spy.iloc[-1] / spy.iloc[-w] - 1) if spy.iloc[-w] > 0 else 0

    # SPY vs MAs
    for w in [20, 50, 100, 200]:
        if len(spy) > w:
            ma = spy.iloc[-w:].mean()
            feats[f'spy_vs_ma{w}'] = (spy.iloc[-1] / ma - 1) if ma > 0 else 0

    # Realized vol
    for w in [10, 21, 63]:
        if len(spy_rets) > w:
            feats[f'realized_vol_{w}d'] = spy_rets.iloc[-w:].std() * np.sqrt(252) * 100

    # Vol regime
    if len(spy_rets) > 63:
        feats['vol_ratio_10_63'] = (spy_rets.iloc[-10:].std()) / (spy_rets.iloc[-63:].std() + 1e-8)

    # Drawdown
    if len(spy) > 63:
        peak = spy.iloc[-63:].max()
        feats['spy_dd_63d'] = (spy.iloc[-1] / peak - 1) if peak > 0 else 0
    if len(spy) > 252:
        peak252 = spy.iloc[-252:].max()
        feats['spy_dd_252d'] = (spy.iloc[-1] / peak252 - 1) if peak252 > 0 else 0

    # VIX features
    if vix is not None and len(vix) > 63:
        feats['vix_level'] = vix.iloc[-1]
        feats['vix_pctile_63'] = (vix.iloc[-1] - vix.iloc[-63:].min()) / \
                                  (vix.iloc[-63:].max() - vix.iloc[-63:].min() + 1e-8)
        feats['vix_pctile_252'] = (vix.iloc[-1] - vix.iloc[-252:].min()) / \
                                   (vix.iloc[-252:].max() - vix.iloc[-252:].min() + 1e-8) if len(vix) > 252 else 0.5
        for w in [5, 21]:
            feats[f'vix_ret_{w}d'] = (vix.iloc[-1] / vix.iloc[-w] - 1) if vix.iloc[-w] > 0 else 0

    # VIX term structure
    if 'VIX3M' in prices.columns:
        vix3m = prices['VIX3M'].iloc[:idx+1]
        if len(vix3m) > 5:
            feats['vix_ratio'] = vix.iloc[-1] / (vix3m.iloc[-1] + 1e-8)

    # Cross-asset
    for ticker in ['TLT', 'GLD', 'HYG', 'USO', 'EEM']:
        if ticker in prices.columns:
            p = prices[ticker].iloc[:idx+1]
            if len(p) > 21:
                feats[f'{ticker.lower()}_ret_21d'] = (p.iloc[-1] / p.iloc[-21] - 1) if p.iloc[-21] > 0 else 0

    # Credit spread
    if 'HYG' in prices.columns and 'LQD' in prices.columns:
        hyg = prices['HYG'].iloc[:idx+1]
        lqd = prices['LQD'].iloc[:idx+1]
        if len(hyg) > 21:
            spread = hyg / lqd
            feats['credit_chg_21d'] = (spread.iloc[-1] / spread.iloc[-21] - 1) if spread.iloc[-21] > 0 else 0

    # Put cost and call premium at current vol
    if vix is not None:
        feats['put_cost'] = estimate_put_cost(vix.iloc[-1])
        feats['call_premium'] = estimate_call_premium(vix.iloc[-1])

    return feats


# ─── Build dataset ───
print("\nBuilding features...")
all_rows = []
dates = prices.index.tolist()

for i in range(TRAIN_DAYS, len(dates) - LABEL_HORIZON):
    feats = build_features(i)
    if feats is None:
        continue

    spy_ret = (prices['SPY'].iloc[i + LABEL_HORIZON] / prices['SPY'].iloc[i]) - 1
    vix_now = prices['VIX'].iloc[i] if 'VIX' in prices.columns else 15

    # Simulate 3 regimes:
    # 1. PROTECTION: Buy put (cost -X%, but saves you if crash)
    put_cost = estimate_put_cost(vix_now)
    protected_ret = max(spy_ret, -0.05) - put_cost  # Put limits loss to 5%

    # 2. INCOME: Sell covered call (earn premium, cap upside at 5%)
    call_prem = estimate_call_premium(vix_now)
    income_ret = min(spy_ret, 0.05) + call_prem  # Call caps gain at 5%

    # 3. NEUTRAL: Just hold SPY
    neutral_ret = spy_ret

    # Which was best?
    rets = {'protect': protected_ret, 'income': income_ret, 'neutral': neutral_ret}
    best = max(rets, key=rets.get)

    feats['_date'] = dates[i]
    feats['_best'] = best
    feats['_protect_ret'] = protected_ret
    feats['_income_ret'] = income_ret
    feats['_neutral_ret'] = neutral_ret
    feats['_spy_ret'] = spy_ret

    all_rows.append(feats)

df_all = pd.DataFrame(all_rows)
print(f"Total observations: {len(df_all)}")
print(f"Regime distribution: {df_all['_best'].value_counts().to_dict()}")

feature_cols = [c for c in df_all.columns if not c.startswith('_')]
print(f"Features: {len(feature_cols)}")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_all['_label'] = le.fit_transform(df_all['_best'])

# ─── Walk-forward ───
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST")
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

    if len(np.unique(y_train)) < 2:
        continue

    if USE_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, verbose=-1, n_jobs=4
        )
    else:
        model = GradientBoostingClassifier(n_estimators=100, max_depth=4)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]
    pred_cat = le.inverse_transform([pred])[0]

    row = test_df.iloc[0]
    cat_rets = {'protect': row['_protect_ret'], 'income': row['_income_ret'], 'neutral': row['_neutral_ret']}

    results[reb_date] = {
        'predicted': pred_cat,
        'actual_best': row['_best'],
        'ml_ret': cat_rets[pred_cat],
        'spy_ret': row['_spy_ret'],
        'protect_ret': row['_protect_ret'],
        'income_ret': row['_income_ret'],
        'neutral_ret': row['_neutral_ret'],
    }
    fold_count += 1

    if fold_count % 50 == 0:
        print(f"  Fold {fold_count}...")

print(f"Completed {fold_count} folds")

# ─── Metrics ───
sorted_dates = sorted(results.keys())
ml_rets = [results[d]['ml_ret'] for d in sorted_dates]
spy_rets = [results[d]['spy_ret'] for d in sorted_dates]

# Allocation stats
allocs = pd.Series([results[d]['predicted'] for d in sorted_dates])
print(f"\nAllocation: {allocs.value_counts().to_dict()}")

def calc_metrics(returns, label):
    r = np.array(returns)
    n = len(r)
    ppy = 252 / ADVANCE_DAYS

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

ml_m = calc_metrics(ml_rets, "ML Dynamic Collar")
spy_m = calc_metrics(spy_rets, "SPY Buy & Hold (no collar)")

# Year-by-year
df_res = pd.DataFrame({'date': sorted_dates, 'ml': ml_rets, 'spy': spy_rets})
df_res['year'] = df_res['date'].apply(lambda d: d.year)
yearly = df_res.groupby('year').agg({
    'ml': lambda x: (np.prod(1+x)-1)*100,
    'spy': lambda x: (np.prod(1+x)-1)*100
}).rename(columns={'ml': 'ML_%', 'spy': 'SPY_%'})

print("\n" + "=" * 70)
print("YEAR-BY-YEAR")
print(yearly.round(1).to_string())
neg_years = (yearly['ML_%'] < 0).sum()
print(f"\nNegative years: {neg_years}/{len(yearly)}")

# ─── ADVERSARIAL ───
print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION")

# Permutation
print("\n--- PERMUTATION ---")
perm_sharpes = []
cats = ['protect', 'income', 'neutral']
for pi in range(N_PERMS):
    p_rets = []
    for d in sorted_dates:
        r = results[d]
        rand_cat = np.random.choice(cats)
        p_rets.append({'protect': r['protect_ret'], 'income': r['income_ret'],
                        'neutral': r['neutral_ret']}[rand_cat])
    r = np.array(p_rets)
    ppy = 252 / ADVANCE_DAYS
    ps = (np.mean(r)*ppy) / (np.std(r, ddof=1)*np.sqrt(ppy)) if np.std(r) > 0 else 0
    perm_sharpes.append(ps)

p_val = np.mean([ps >= ml_m['sharpe'] for ps in perm_sharpes])
print(f"  ML: {ml_m['sharpe']:.3f}, Perm mean: {np.mean(perm_sharpes):.3f}, p={p_val:.3f}")
perm_v = "PASS" if p_val < 0.05 else "FAIL"

# Sub-period
print("\n--- SUB-PERIOD ---")
q_sz = len(ml_rets) // 4
sub_sh = []
for q in range(4):
    s, e = q*q_sz, (q+1)*q_sz if q < 3 else len(ml_rets)
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
ppy = 252 / ADVANCE_DAYS
ts = (np.mean(tr)*ppy) / (np.std(tr, ddof=1)*np.sqrt(ppy)) if len(tr) > 5 and np.std(tr) > 0 else 0
deg = (1 - ts / ml_m['sharpe']) * 100 if ml_m['sharpe'] != 0 else 0
out_v = "PASS" if deg < 50 else "FAIL"
print(f"  Full: {ml_m['sharpe']:.3f}, Trim: {ts:.3f}, Deg: {deg:.1f}%, Verdict: {out_v}")

# R1
print("\n--- R1 REGIME ---")
green_r = [ml_rets[i] for i in range(len(ml_rets)) if spy_rets[i] >= 0]
red_r = [ml_rets[i] for i in range(len(ml_rets)) if spy_rets[i] < 0]
if len(green_r) > 5 and len(red_r) > 5:
    gs = (np.mean(green_r)*ppy) / (np.std(green_r, ddof=1)*np.sqrt(ppy)) if np.std(green_r) > 0 else 0
    rs = (np.mean(red_r)*ppy) / (np.std(red_r, ddof=1)*np.sqrt(ppy)) if np.std(red_r) > 0 else 0
    gap = abs(gs-rs) / max(abs(gs), abs(rs), 0.01)
    print(f"  Green: {gs:.3f}, Red: {rs:.3f}, Gap: {gap:.3f}")
    r1_v = "PASS" if gap < 0.50 else "FAIL"
else:
    r1_v = "FAIL"; gs = rs = gap = 0
print(f"  Verdict: {r1_v}")

gates = sum([v == "PASS" for v in [perm_v, sub_v, out_v, r1_v]])
verdict = "VALIDATED" if gates >= 3 else ("PARTIAL" if gates >= 2 else "REJECTED")

# Key finding: does the collar REDUCE MaxDD vs SPY?
dd_reduction = ml_m['max_dd'] - spy_m['max_dd']
print(f"\n  MaxDD reduction vs SPY: {dd_reduction:+.1f}pp")
print(f"  {'✅ Collar reduces drawdowns' if dd_reduction > 0 else '❌ Collar does NOT reduce drawdowns'}")

# Save
output = {
    'strategy': 'ML Dynamic Collar',
    'purpose': 'drawdown protection overlay',
    'ml_metrics': ml_m, 'spy_metrics': spy_m,
    'dd_reduction_pp': round(dd_reduction, 1),
    'allocation': allocs.value_counts().to_dict(),
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
print(f"ML Dynamic Collar: Sharpe {ml_m['sharpe']}, MaxDD {ml_m['max_dd']}%")
print(f"SPY no collar: Sharpe {spy_m['sharpe']}, MaxDD {spy_m['max_dd']}%")
