#!/usr/bin/env python3
"""
ML Volatility Risk Premium Harvesting
======================================
ML predicts when it's safe to harvest vol premium (short vol via SVXY/VIXY inverse,
long SPY) vs when to be flat/defensive. Uses VIX term structure slope,
VIX/VIX3M ratio, realized-vs-implied vol spread, and macro regime signals.

Walk-forward LightGBM (252d train, 21d advance, sliding window).
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
# Short vol assets (benefit from contango/vol decline)
SHORT_VOL_TICKERS = ['SVXY']   # Short VIX ETF
# Defensive
DEFENSIVE_TICKERS = ['SHY', 'TLT', 'GLD']
# Equity
EQUITY_TICKERS = ['SPY', 'QQQ']
# Vol signals
VOL_TICKERS = ['^VIX', '^VIX3M']
# All tradable
ALL_TICKERS = SHORT_VOL_TICKERS + DEFENSIVE_TICKERS + EQUITY_TICKERS
BENCHMARK = 'SPY'

REGIME_TICKERS = ['HYG', 'TLT', 'GLD', 'DBC', 'SPY', 'IEF']

TRAIN_DAYS = 252
ADVANCE_DAYS = 21
LABEL_HORIZON = 21
INITIAL_CAPITAL = 100_000
N_PERMS = 200
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_vol_risk_premium')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML VOLATILITY RISK PREMIUM HARVESTING")
print("=" * 70)

# ─── Download data ───
print(f"\nDownloading data...")
data = {}
download_tickers = ALL_TICKERS + [BENCHMARK] + REGIME_TICKERS + ['^VIX', '^VIX3M']
download_tickers = list(set(download_tickers))

for t in download_tickers:
    try:
        df = yf.download(t, start='2012-01-01', end='2026-07-19',
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

valid_short_vol = [t for t in ['SVXY'] if t in prices.columns]
valid_defensive = [t for t in DEFENSIVE_TICKERS if t in prices.columns]
valid_equity = [t for t in EQUITY_TICKERS if t in prices.columns]
print(f"Short vol: {valid_short_vol}, Defensive: {valid_defensive}, Equity: {valid_equity}")


# ─── Feature engineering ───
def build_features(prices_df, date_idx):
    """Build vol risk premium / regime features."""
    feats = {}

    # VIX features
    if 'VIX' in prices_df.columns:
        vix = prices_df['VIX'].iloc[:date_idx+1]
        if len(vix) > 252:
            feats['vix_level'] = vix.iloc[-1]
            feats['vix_pctile_252'] = (vix.iloc[-1] - vix.iloc[-252:].min()) / \
                                       (vix.iloc[-252:].max() - vix.iloc[-252:].min() + 1e-8)
            feats['vix_pctile_63'] = (vix.iloc[-1] - vix.iloc[-63:].min()) / \
                                      (vix.iloc[-63:].max() - vix.iloc[-63:].min() + 1e-8)
            for w in [5, 10, 21, 63]:
                if len(vix) > w:
                    feats[f'vix_chg_{w}d'] = (vix.iloc[-1] / vix.iloc[-w] - 1) if vix.iloc[-w] > 0 else 0
            # VIX mean reversion signal
            feats['vix_vs_mean_63'] = vix.iloc[-1] / vix.iloc[-63:].mean() - 1
            feats['vix_vs_mean_252'] = vix.iloc[-1] / vix.iloc[-252:].mean() - 1

    # VIX term structure (contango = safe to sell vol)
    if 'VIX' in prices_df.columns and 'VIX3M' in prices_df.columns:
        vix_s = prices_df['VIX'].iloc[:date_idx+1]
        vix3m = prices_df['VIX3M'].iloc[:date_idx+1]
        if len(vix_s) > 21 and len(vix3m) > 21:
            ratio = vix_s.iloc[-1] / (vix3m.iloc[-1] + 1e-8)
            feats['vix_term_structure'] = ratio  # <1 = contango (safe), >1 = backwardation (danger)
            feats['vix_term_ma5'] = np.mean([vix_s.iloc[-i] / (vix3m.iloc[-i] + 1e-8) for i in range(1, 6)])
            feats['vix_term_ma21'] = np.mean([vix_s.iloc[-i] / (vix3m.iloc[-i] + 1e-8) for i in range(1, min(22, len(vix_s)))])

    # Realized vs implied vol spread (VRP)
    if 'SPY' in prices_df.columns and 'VIX' in prices_df.columns:
        spy = prices_df['SPY'].iloc[:date_idx+1]
        vix_s = prices_df['VIX'].iloc[:date_idx+1]
        if len(spy) > 63:
            spy_rets = spy.pct_change().dropna()
            rv_21 = spy_rets.iloc[-21:].std() * np.sqrt(252) * 100
            rv_63 = spy_rets.iloc[-63:].std() * np.sqrt(252) * 100
            feats['vrp_21d'] = vix_s.iloc[-1] - rv_21  # VRP = IV - RV
            feats['vrp_63d'] = vix_s.iloc[-1] - rv_63
            feats['rv_21d'] = rv_21
            feats['rv_63d'] = rv_63

    # Asset momentum & vol for all tradable
    for ticker in valid_short_vol + valid_defensive + valid_equity:
        if ticker not in prices_df.columns:
            continue
        p = prices_df[ticker].iloc[:date_idx+1]
        if len(p) < 252:
            continue
        prefix = ticker.lower()

        for w in [5, 10, 21, 63, 126, 252]:
            if len(p) > w:
                feats[f'{prefix}_ret_{w}d'] = (p.iloc[-1] / p.iloc[-w] - 1) if p.iloc[-w] > 0 else 0

        for w in [50, 200]:
            if len(p) > w:
                ma = p.iloc[-w:].mean()
                feats[f'{prefix}_vs_ma{w}'] = (p.iloc[-1] / ma - 1) if ma > 0 else 0

        rets = p.pct_change().dropna()
        if len(rets) > 63:
            feats[f'{prefix}_vol_21d'] = rets.iloc[-21:].std() * np.sqrt(252)
            feats[f'{prefix}_vol_63d'] = rets.iloc[-63:].std() * np.sqrt(252)

    # Credit spread (risk appetite)
    if 'HYG' in prices_df.columns and 'TLT' in prices_df.columns:
        hyg = prices_df['HYG'].iloc[:date_idx+1]
        tlt = prices_df['TLT'].iloc[:date_idx+1]
        if len(hyg) > 63:
            spread = hyg / tlt
            feats['credit_spread_21d'] = (spread.iloc[-1] / spread.iloc[-21] - 1) if spread.iloc[-21] > 0 else 0
            feats['credit_spread_63d'] = (spread.iloc[-1] / spread.iloc[-63] - 1) if spread.iloc[-63] > 0 else 0

    # SPY drawdown
    if 'SPY' in prices_df.columns:
        spy = prices_df['SPY'].iloc[:date_idx+1]
        if len(spy) > 252:
            peak = spy.iloc[-252:].max()
            feats['spy_dd_from_peak'] = (spy.iloc[-1] / peak - 1) if peak > 0 else 0

    return feats


# ─── Build dataset ───
print("\nBuilding features...")
all_rows = []
dates = prices.index.tolist()

# Strategy: predict which regime to be in
# "short_vol" = harvest VRP (SVXY or SPY-heavy)
# "defensive" = hide in bonds/gold
# "neutral"   = cash-like / balanced

for i in range(TRAIN_DAYS, len(dates) - LABEL_HORIZON):
    feats = build_features(prices, i)
    if not feats:
        continue

    # Forward returns for each category
    short_vol_ret = np.mean([(prices[t].iloc[i + LABEL_HORIZON] / prices[t].iloc[i] - 1)
                              for t in valid_short_vol]) if valid_short_vol else 0
    # Blend SVXY with SPY for short vol bucket
    if 'SPY' in prices.columns:
        spy_fwd = (prices['SPY'].iloc[i + LABEL_HORIZON] / prices['SPY'].iloc[i] - 1)
        short_vol_ret = 0.5 * short_vol_ret + 0.5 * spy_fwd if valid_short_vol else spy_fwd

    defensive_ret = np.mean([(prices[t].iloc[i + LABEL_HORIZON] / prices[t].iloc[i] - 1)
                              for t in valid_defensive]) if valid_defensive else 0

    # Determine best category
    neutral_ret = 0.001 * (LABEL_HORIZON / 252)  # ~0.1% annualized cash proxy
    category_rets = {'short_vol': short_vol_ret, 'defensive': defensive_ret, 'neutral': neutral_ret}
    best_cat = max(category_rets, key=category_rets.get)

    feats['_date'] = dates[i]
    feats['_best_category'] = best_cat
    feats['_short_vol_ret'] = short_vol_ret
    feats['_defensive_ret'] = defensive_ret
    feats['_neutral_ret'] = neutral_ret

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
        'short_vol_ret': row['_short_vol_ret'],
        'defensive_ret': row['_defensive_ret'],
        'neutral_ret': row['_neutral_ret']
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

    cat_rets = {'short_vol': r['short_vol_ret'], 'defensive': r['defensive_ret'], 'neutral': r['neutral_ret']}
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

ml_metrics = calc_metrics(ml_returns, "ML Vol Risk Premium")
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
categories = ['short_vol', 'defensive', 'neutral']

for perm_i in range(N_PERMS):
    perm_rets = []
    for r in results_by_date.values():
        random_cat = np.random.choice(categories)
        cat_rets = {'short_vol': r['short_vol_ret'], 'defensive': r['defensive_ret'], 'neutral': r['neutral_ret']}
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
    'strategy': 'ML Volatility Risk Premium Harvesting',
    'type': 'vol risk premium rotation',
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
print(f"ML Vol Risk Premium: Sharpe {ml_sharpe:.3f}, CAGR {ml_metrics['cagr']}%, MaxDD {ml_metrics['max_dd']}%")
print(f"Accuracy: {accuracy:.1f}%")
print(f"Gates: Perm={perm_verdict}, SubP={sub_verdict}, Outlier={outlier_verdict}, R1={r1_verdict}")
