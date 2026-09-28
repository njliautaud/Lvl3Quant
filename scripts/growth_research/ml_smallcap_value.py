#!/usr/bin/env python3
"""
ML Small Cap Value Rotation — Fama-French Value Premium
========================================================
ML rotation between small-cap value (IWN, VBR, SLYV) and large-cap growth
(VUG, IWF, MGK). Exploits the Fama-French value premium with ML timing —
when does value outperform growth and vice versa?

Walk-forward GBM with sliding window (252d train, 21d advance).
Full adversarial validation: permutation, sub-period, outlier, R1 regime.
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ─── Config ───
VALUE_TICKERS = ['IWN', 'VBR', 'SLYV']    # small-cap value
GROWTH_TICKERS = ['VUG', 'IWF', 'MGK']      # large-cap growth
BLEND_TICKERS = ['VTI', 'RSP']               # total market / equal-weight
ALL_TICKERS = VALUE_TICKERS + GROWTH_TICKERS + BLEND_TICKERS
BENCHMARK = 'SPY'

# Regime signals
REGIME_TICKERS = ['TLT', 'HYG', 'GLD', 'DBC', 'XLF', 'XLE']

TRAIN_DAYS = 252
ADVANCE_DAYS = 21
LABEL_HORIZON = 21
INITIAL_CAPITAL = 100_000
N_PERMS = 200
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_smallcap_value')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML SMALL CAP VALUE ROTATION — FAMA-FRENCH PREMIUM")
print("=" * 70)

# ─── Download data ───
print(f"\nDownloading data...")
data = {}
for t in ALL_TICKERS + [BENCHMARK] + REGIME_TICKERS + ['^VIX']:
    try:
        df = yf.download(t, start='2006-01-01', end='2026-07-19',
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
        else:
            print(f"  {t}: SKIP ({len(df)} days)")
    except Exception as e:
        print(f"  {t}: FAILED ({e})")

prices = pd.DataFrame(data).dropna(how='all').ffill().dropna()
print(f"\nAligned: {len(prices)} days, {prices.shape[1]} assets")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

valid_value = [t for t in VALUE_TICKERS if t in prices.columns]
valid_growth = [t for t in GROWTH_TICKERS if t in prices.columns]
valid_blend = [t for t in BLEND_TICKERS if t in prices.columns]
valid_all = valid_value + valid_growth + valid_blend
print(f"Value: {valid_value}, Growth: {valid_growth}, Blend: {valid_blend}")

if len(valid_all) < 4:
    print("ERROR: Need at least 4 valid ETFs")
    exit(1)


# ─── Feature engineering ───
def build_features(prices_df, date_idx):
    """Build value/growth rotation features."""
    feats = {}

    for ticker in valid_all + ['SPY']:
        if ticker not in prices_df.columns:
            continue
        p = prices_df[ticker].iloc[:date_idx+1]
        if len(p) < 252:
            continue
        prefix = ticker.lower()

        # Returns at multiple horizons
        for w in [5, 10, 21, 63, 126, 252]:
            if len(p) > w:
                feats[f'{prefix}_ret_{w}d'] = (p.iloc[-1] / p.iloc[-w] - 1) if p.iloc[-w] > 0 else 0

        # Trend
        for w in [50, 200]:
            if len(p) > w:
                ma = p.iloc[-w:].mean()
                feats[f'{prefix}_vs_ma{w}'] = (p.iloc[-1] / ma - 1) if ma > 0 else 0

        # Vol
        rets = p.pct_change().dropna()
        if len(rets) > 63:
            feats[f'{prefix}_vol_21d'] = rets.iloc[-21:].std() * np.sqrt(252)
            feats[f'{prefix}_vol_63d'] = rets.iloc[-63:].std() * np.sqrt(252)

        # Drawdown
        if len(p) > 252:
            peak = p.iloc[-252:].max()
            feats[f'{prefix}_dd_252'] = (p.iloc[-1] / peak - 1) if peak > 0 else 0

    # VALUE vs GROWTH spread — the core signal
    if valid_value and valid_growth:
        for w in [21, 63, 126, 252]:
            val_ret = np.mean([(prices_df[t].iloc[date_idx] / prices_df[t].iloc[max(0,date_idx-w)] - 1)
                                for t in valid_value if t in prices_df.columns])
            gro_ret = np.mean([(prices_df[t].iloc[date_idx] / prices_df[t].iloc[max(0,date_idx-w)] - 1)
                                for t in valid_growth if t in prices_df.columns])
            feats[f'value_vs_growth_{w}d'] = val_ret - gro_ret

    # Rate sensitivity proxy (TLT)
    if 'TLT' in prices_df.columns:
        tlt = prices_df['TLT'].iloc[:date_idx+1]
        if len(tlt) > 63:
            feats['tlt_ret_21d'] = (tlt.iloc[-1] / tlt.iloc[-21] - 1)
            feats['tlt_ret_63d'] = (tlt.iloc[-1] / tlt.iloc[-63] - 1)
            feats['tlt_ret_126d'] = (tlt.iloc[-1] / tlt.iloc[-126] - 1) if len(tlt) > 126 else 0

    # Financials (value proxy) and Energy
    for sec in ['XLF', 'XLE']:
        if sec in prices_df.columns:
            p = prices_df[sec].iloc[:date_idx+1]
            if len(p) > 63:
                feats[f'{sec.lower()}_ret_21d'] = (p.iloc[-1] / p.iloc[-21] - 1)
                feats[f'{sec.lower()}_ret_63d'] = (p.iloc[-1] / p.iloc[-63] - 1)
                # Relative to SPY
                spy = prices_df['SPY'].iloc[:date_idx+1]
                if len(spy) > 21:
                    feats[f'{sec.lower()}_vs_spy_21d'] = (p.iloc[-1]/p.iloc[-21]) / (spy.iloc[-1]/spy.iloc[-21]) - 1

    # VIX
    if 'VIX' in prices_df.columns:
        vix = prices_df['VIX'].iloc[:date_idx+1]
        if len(vix) > 252:
            feats['vix_level'] = vix.iloc[-1]
            feats['vix_pctile_252'] = (vix.iloc[-1] - vix.iloc[-252:].min()) / \
                                       (vix.iloc[-252:].max() - vix.iloc[-252:].min() + 1e-8)
            feats['vix_chg_21d'] = (vix.iloc[-1] / vix.iloc[-21] - 1) if vix.iloc[-21] > 0 else 0

    # Credit spreads (value correlates with credit)
    if 'HYG' in prices_df.columns:
        hyg = prices_df['HYG'].iloc[:date_idx+1]
        if len(hyg) > 63:
            feats['hyg_ret_21d'] = (hyg.iloc[-1] / hyg.iloc[-21] - 1)
            feats['hyg_ret_63d'] = (hyg.iloc[-1] / hyg.iloc[-63] - 1)

    # Commodities (value is energy/materials heavy)
    if 'DBC' in prices_df.columns:
        dbc = prices_df['DBC'].iloc[:date_idx+1]
        if len(dbc) > 63:
            feats['dbc_ret_21d'] = (dbc.iloc[-1] / dbc.iloc[-21] - 1)
            feats['dbc_ret_63d'] = (dbc.iloc[-1] / dbc.iloc[-63] - 1)

    # Small vs large spread (breadth)
    if valid_value and 'SPY' in prices_df.columns:
        val_vol = np.mean([prices_df[t].iloc[max(0,date_idx-62):date_idx+1].pct_change().std()
                           for t in valid_value])
        spy_vol = prices_df['SPY'].iloc[max(0,date_idx-62):date_idx+1].pct_change().std()
        feats['small_vs_large_vol'] = val_vol / spy_vol if spy_vol > 0 else 1

    return feats


# ─── Build dataset ───
print("\nBuilding features...")
all_rows = []
dates = prices.index.tolist()

for i in range(TRAIN_DAYS, len(dates) - LABEL_HORIZON):
    feats = build_features(prices, i)
    if not feats:
        continue

    value_ret = np.mean([(prices[t].iloc[i + LABEL_HORIZON] / prices[t].iloc[i] - 1)
                          for t in valid_value]) if valid_value else 0
    growth_ret = np.mean([(prices[t].iloc[i + LABEL_HORIZON] / prices[t].iloc[i] - 1)
                           for t in valid_growth]) if valid_growth else 0
    blend_ret = np.mean([(prices[t].iloc[i + LABEL_HORIZON] / prices[t].iloc[i] - 1)
                          for t in valid_blend]) if valid_blend else 0

    category_rets = {'value': value_ret, 'growth': growth_ret, 'blend': blend_ret}
    best_cat = max(category_rets, key=category_rets.get)

    feats['_date'] = dates[i]
    feats['_best_category'] = best_cat
    feats['_value_ret'] = value_ret
    feats['_growth_ret'] = growth_ret
    feats['_blend_ret'] = blend_ret

    all_rows.append(feats)

df_all = pd.DataFrame(all_rows)
print(f"Total observations: {len(df_all)}")
print(f"Category distribution: {df_all['_best_category'].value_counts().to_dict()}")

feature_cols = [c for c in df_all.columns if not c.startswith('_')]
print(f"Features: {len(feature_cols)}")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_all['_label_encoded'] = le.fit_transform(df_all['_best_category'])

# ─── Walk-forward ───
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST")
print("=" * 70)

try:
    import lightgbm as lgb
    USE_LGB = True
    print("Using LightGBM")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_LGB = False
    print("Using sklearn GBM")

results_by_date = {}
fold_count = 0
unique_dates = df_all['_date'].unique()
rebalance_dates = unique_dates[TRAIN_DAYS::ADVANCE_DAYS]

print(f"Running walk-forward: {len(rebalance_dates)} rebalance periods")

for reb_date in rebalance_dates:
    train_mask = df_all['_date'] < reb_date
    train_dates = df_all[train_mask]['_date'].unique()
    if len(train_dates) < TRAIN_DAYS // 2:
        continue

    cutoff_dates = sorted(train_dates)[-TRAIN_DAYS:]
    train_df = df_all[df_all['_date'].isin(cutoff_dates)]
    test_df = df_all[df_all['_date'] == reb_date]

    if len(test_df) == 0 or len(train_df) < 50:
        continue

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['_label_encoded'].values
    X_test = test_df[feature_cols].fillna(0).values

    n_classes = len(np.unique(y_train))
    if n_classes < 2:
        continue

    if USE_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, verbose=-1, n_jobs=4,
            num_class=n_classes if n_classes > 2 else None
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20
        )

    model.fit(X_train, y_train)
    pred_class = model.predict(X_test)[0]
    pred_category = le.inverse_transform([pred_class])[0]

    row = test_df.iloc[0]
    results_by_date[reb_date] = {
        'predicted': pred_category,
        'actual_best': row['_best_category'],
        'value_ret': row['_value_ret'],
        'growth_ret': row['_growth_ret'],
        'blend_ret': row['_blend_ret']
    }

    fold_count += 1
    if fold_count % 50 == 0:
        print(f"  Fold {fold_count}/{len(rebalance_dates)}...")

print(f"Completed {fold_count} folds")

# ─── Portfolio returns ───
print("\n" + "=" * 70)
print("PORTFOLIO CONSTRUCTION")
print("=" * 70)

ml_returns = []
ew_returns = []
spy_returns_list = []
portfolio_dates = []
accuracy_count = 0

for reb_date in sorted(results_by_date.keys()):
    r = results_by_date[reb_date]

    cat_rets = {'value': r['value_ret'], 'growth': r['growth_ret'], 'blend': r['blend_ret']}
    ml_ret = cat_rets[r['predicted']]
    ml_returns.append(ml_ret)

    ew_ret = np.mean(list(cat_rets.values()))
    ew_returns.append(ew_ret)

    d_idx = prices.index.get_loc(reb_date)
    if d_idx + LABEL_HORIZON < len(prices):
        spy_ret = (prices['SPY'].iloc[d_idx + LABEL_HORIZON] / prices['SPY'].iloc[d_idx]) - 1
    else:
        spy_ret = 0
    spy_returns_list.append(spy_ret)

    portfolio_dates.append(reb_date)

    if r['predicted'] == r['actual_best']:
        accuracy_count += 1

accuracy = accuracy_count / len(results_by_date) * 100
print(f"Category prediction accuracy: {accuracy:.1f}%")

# ─── Metrics ───
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

    return {
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr, 1), 'max_dd': round(max_dd, 1),
        'calmar': round(calmar, 2), 'wr': round(wr, 1),
        'pf': round(pf, 2), 'n_periods': n
    }

ml_metrics = calc_metrics(ml_returns, "ML Small Cap Value")
ew_metrics = calc_metrics(ew_returns, "Equal-Weight All")
spy_metrics = calc_metrics(spy_returns_list, "SPY Buy & Hold")

# ─── Year-by-year ───
print("\n" + "=" * 70)
print("YEAR-BY-YEAR RETURNS")
print("=" * 70)

df_port = pd.DataFrame({
    'date': portfolio_dates,
    'ml_return': ml_returns,
    'ew_return': ew_returns,
    'spy_return': spy_returns_list
})
df_port['year'] = df_port['date'].apply(lambda d: d.year)

yearly = df_port.groupby('year').agg({
    'ml_return': lambda x: (np.prod(1 + x) - 1) * 100,
    'ew_return': lambda x: (np.prod(1 + x) - 1) * 100,
    'spy_return': lambda x: (np.prod(1 + x) - 1) * 100
}).rename(columns={'ml_return': 'ML_%', 'ew_return': 'EW_%', 'spy_return': 'SPY_%'})

print(yearly.round(1).to_string())
negative_years = (yearly['ML_%'] < 0).sum()
print(f"\nNegative years: {negative_years}/{len(yearly)}")

# ─── ADVERSARIAL VALIDATION ───
print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION")
print("=" * 70)

ml_sharpe = ml_metrics['sharpe']

# 1. Permutation test
print("\n--- PERMUTATION TEST ---")
perm_sharpes = []
categories = ['value', 'growth', 'blend']

for perm_i in range(N_PERMS):
    perm_rets = []
    for r in results_by_date.values():
        random_cat = np.random.choice(categories)
        cat_rets = {'value': r['value_ret'], 'growth': r['growth_ret'], 'blend': r['blend_ret']}
        perm_rets.append(cat_rets[random_cat])

    r = np.array(perm_rets)
    ppy = 252 / ADVANCE_DAYS
    mean_r = np.mean(r) * ppy
    std_r = np.std(r, ddof=1) * np.sqrt(ppy)
    perm_sharpes.append(mean_r / std_r if std_r > 0 else 0)

    if (perm_i + 1) % 50 == 0:
        print(f"  Perm {perm_i + 1}/{N_PERMS}...")

p_value = np.mean([ps >= ml_sharpe for ps in perm_sharpes])
print(f"  Observed: {ml_sharpe:.3f}, Perm mean: {np.mean(perm_sharpes):.3f}, p={p_value:.3f}")
perm_verdict = "PASS" if p_value < 0.05 else "FAIL"
print(f"  Verdict: {perm_verdict}")

# 2. Sub-period
print("\n--- SUB-PERIOD ---")
n_total = len(ml_returns)
q_size = n_total // 4
sub_sharpes = []
for q in range(4):
    s = q * q_size
    e = (q+1) * q_size if q < 3 else n_total
    sub_r = np.array(ml_returns[s:e])
    ppy = 252 / ADVANCE_DAYS
    sub_sharpe = (np.mean(sub_r) * ppy) / (np.std(sub_r, ddof=1) * np.sqrt(ppy)) if np.std(sub_r) > 0 else 0
    sub_sharpes.append(sub_sharpe)
    print(f"  Q{q+1}: Sharpe {sub_sharpe:.3f}")

cv = np.std(sub_sharpes) / abs(np.mean(sub_sharpes)) if np.mean(sub_sharpes) != 0 else 999
sub_verdict = "PASS" if cv < 0.75 and all(s > -0.5 for s in sub_sharpes) else "FAIL"
print(f"  CV: {cv:.3f}, Verdict: {sub_verdict}")

# 3. Outlier
print("\n--- OUTLIER ---")
r = np.array(ml_returns)
cutoff = np.percentile(abs(r), 95)
trimmed = r[abs(r) <= cutoff]
if len(trimmed) > 5:
    ppy = 252 / ADVANCE_DAYS
    trim_sharpe = (np.mean(trimmed) * ppy) / (np.std(trimmed, ddof=1) * np.sqrt(ppy)) if np.std(trimmed) > 0 else 0
    deg = (1 - trim_sharpe / ml_sharpe) * 100 if ml_sharpe != 0 else 0
    print(f"  Full: {ml_sharpe:.3f}, Trimmed: {trim_sharpe:.3f}, Degradation: {deg:.1f}%")
    outlier_verdict = "PASS" if deg < 50 else "FAIL"
else:
    outlier_verdict = "FAIL"
    trim_sharpe = 0
    deg = 100
print(f"  Verdict: {outlier_verdict}")

# 4. R1 Regime
print("\n--- R1 REGIME ---")
green_rets = [ml_returns[i] for i in range(len(ml_returns)) if spy_returns_list[i] >= 0]
red_rets = [ml_returns[i] for i in range(len(ml_returns)) if spy_returns_list[i] < 0]
print(f"  Green: {len(green_rets)}, Red: {len(red_rets)}")

if len(green_rets) > 5 and len(red_rets) > 5:
    ppy = 252 / ADVANCE_DAYS
    g_sharpe = (np.mean(green_rets) * ppy) / (np.std(green_rets, ddof=1) * np.sqrt(ppy)) if np.std(green_rets) > 0 else 0
    r_sharpe = (np.mean(red_rets) * ppy) / (np.std(red_rets, ddof=1) * np.sqrt(ppy)) if np.std(red_rets) > 0 else 0
    gap = abs(g_sharpe - r_sharpe) / max(abs(g_sharpe), abs(r_sharpe), 0.01)
    print(f"  Green: {g_sharpe:.3f}, Red: {r_sharpe:.3f}, Gap: {gap:.3f}")
    r1_verdict = "PASS" if gap < 0.50 else "FAIL"
else:
    r1_verdict = "FAIL"
    g_sharpe = r_sharpe = gap = 0
print(f"  Verdict: {r1_verdict}")

# ─── Save ───
gates = sum([v == "PASS" for v in [perm_verdict, sub_verdict, outlier_verdict, r1_verdict]])

results = {
    'strategy': 'ML Small Cap Value Rotation',
    'type': 'value/growth factor rotation',
    'accuracy': round(accuracy, 1),
    'ml_metrics': ml_metrics,
    'ew_metrics': ew_metrics,
    'spy_metrics': spy_metrics,
    'yearly_returns': yearly.to_dict(),
    'negative_years': int(negative_years),
    'adversarial': {
        'permutation': {'sharpe': ml_sharpe, 'perm_mean': round(np.mean(perm_sharpes), 3), 'p_value': round(p_value, 3), 'verdict': perm_verdict},
        'sub_period': {'sharpes': [round(s, 3) for s in sub_sharpes], 'cv': round(cv, 3), 'verdict': sub_verdict},
        'outlier': {'full': ml_sharpe, 'trimmed': round(trim_sharpe, 3), 'degradation': round(deg, 1), 'verdict': outlier_verdict},
        'r1_regime': {'green': round(g_sharpe, 3), 'red': round(r_sharpe, 3), 'gap': round(gap, 3), 'verdict': r1_verdict},
        'gates_passed': f"{gates}/4"
    },
    'spy_correlation': round(np.corrcoef(ml_returns, spy_returns_list)[0, 1], 3),
    'timestamp': dt.datetime.now().isoformat()
}

with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

df_port.to_parquet(OUTPUT_DIR / 'portfolio_returns.parquet', index=False)

verdict = "VALIDATED" if gates >= 3 else ("PARTIAL" if gates >= 2 else "REJECTED")
print(f"\n{'='*70}")
print(f"FINAL VERDICT: {verdict} ({gates}/4 gates)")
print(f"{'='*70}")
print(f"ML Small Cap Value: Sharpe {ml_sharpe:.3f}, CAGR {ml_metrics['cagr']}%, MaxDD {ml_metrics['max_dd']}%")
print(f"Accuracy: {accuracy:.1f}%")
print(f"Gates: Perm={perm_verdict}, SubP={sub_verdict}, Outlier={outlier_verdict}, R1={r1_verdict}")
