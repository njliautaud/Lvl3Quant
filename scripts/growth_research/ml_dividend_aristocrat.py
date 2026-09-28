#!/usr/bin/env python3
"""
ML Dividend Aristocrat Rotation
=================================
HC #714: Income + growth dual focus. ML for exploratory research.

Concept: Rotate among dividend-focused ETFs using ML to predict which
dividend style (high-yield, dividend growth, covered call income, REIT)
will outperform in the next month. Income-focused strategy.

Universe:
  - SCHD: Schwab US Dividend (quality dividend growth)
  - VYM: Vanguard High Dividend Yield
  - DVY: iShares Select Dividend (high yield)
  - NOBL: ProShares S&P Dividend Aristocrats
  - VNQ: Vanguard REIT (real estate income)
  - JEPI: JPMorgan Equity Premium Income (covered call)
  - HDV: iShares Core High Dividend
  - SDY: SPDR S&P Dividend ETF
  Safety:
  - SHY: 1-3yr Treasury (safety)

Walk-forward LightGBM (252d train, 21d advance, sliding window).
HC #713: Fixed $100K capital, no DCA.
HC #0: Sliding windows only.
Full adversarial validation.
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

BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_dividend_aristocrat"
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML DIVIDEND ARISTOCRAT ROTATION")
print("=" * 70)

# ─── Config ───
INCOME_TICKERS = ['SCHD', 'VYM', 'DVY', 'NOBL', 'HDV', 'SDY', 'VNQ']
SAFETY_TICKER = 'SHY'
BENCHMARK = 'SPY'

MACRO_TICKERS = ['^VIX', 'GLD', 'TLT', 'HYG', 'LQD', 'UUP', 'DBC', 'SPY']

TRAIN_DAYS = 252
ADVANCE_DAYS = 21
LABEL_HORIZON = 21
INITIAL_CAPITAL = 100_000
N_TOP = 3  # Hold top 3 ETFs each period
N_PERMS = 200

print(f"Universe: {INCOME_TICKERS}")
print(f"Safety: {SAFETY_TICKER}")
print(f"Top-N selection: {N_TOP}")

# ─── Download Data ───
print(f"\nDownloading data...")
all_tickers = list(set(INCOME_TICKERS + [SAFETY_TICKER, BENCHMARK] + MACRO_TICKERS))
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
        else:
            print(f"  {t}: SKIP ({len(df)} days)")
    except Exception as e:
        print(f"  {t}: FAILED ({e})")

prices = pd.DataFrame(data).dropna(how='all').ffill().dropna()
print(f"\nAligned: {len(prices)} days, {prices.shape[1]} assets")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

valid_income = [t for t in INCOME_TICKERS if t in prices.columns]
print(f"Available income ETFs: {valid_income}")

if len(valid_income) < 3:
    print("ERROR: Need at least 3 income ETFs")
    exit(1)

# ─── Build Features ───
print(f"\nBuilding features...")
returns = prices.pct_change()

feature_frames = []
for t in valid_income + [SAFETY_TICKER]:
    if t not in prices.columns:
        continue
    feat = pd.DataFrame(index=prices.index)
    # Momentum
    for w in [5, 10, 21, 63, 126, 252]:
        feat[f'{t}_mom_{w}d'] = prices[t].pct_change(w)
    # Volatility
    feat[f'{t}_vol_21d'] = returns[t].rolling(21).std()
    feat[f'{t}_vol_63d'] = returns[t].rolling(63).std()
    # Relative strength vs SPY
    if BENCHMARK in prices.columns:
        feat[f'{t}_rs_spy_21d'] = prices[t].pct_change(21) - prices[BENCHMARK].pct_change(21)
        feat[f'{t}_rs_spy_63d'] = prices[t].pct_change(63) - prices[BENCHMARK].pct_change(63)
    # Drawdown
    feat[f'{t}_drawdown'] = prices[t] / prices[t].cummax() - 1
    feature_frames.append(feat)

# Macro features
macro_feat = pd.DataFrame(index=prices.index)
macro_names = [t.replace('^', '') for t in MACRO_TICKERS if t.replace('^', '') in prices.columns]
for m in macro_names:
    macro_feat[f'{m}_ret_21d'] = prices[m].pct_change(21)
    macro_feat[f'{m}_vol_21d'] = returns[m].rolling(21).std() if m in returns.columns else 0

# Cross-asset spreads
if 'HYG' in prices.columns and 'LQD' in prices.columns:
    macro_feat['credit_spread'] = (returns['HYG'] - returns['LQD']).rolling(21).mean()
if 'VIX' in prices.columns:
    macro_feat['vix_level'] = prices['VIX']
    macro_feat['vix_zscore'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()

feature_frames.append(macro_feat)
features = pd.concat(feature_frames, axis=1).dropna()
print(f"Features: {features.shape[1]} columns, {len(features)} days")

# ─── Walk-Forward Backtest ───
print(f"\nRunning walk-forward backtest...")

# Create rebalancing dates
rebal_indices = list(range(TRAIN_DAYS, len(features) - ADVANCE_DAYS, ADVANCE_DAYS))
print(f"Rebalancing periods: {len(rebal_indices)}")

all_dates = []
all_ml_returns = []
all_ew_returns = []
all_spy_returns = []
all_selections = []

for idx_num, start_idx in enumerate(rebal_indices):
    train_start = max(0, start_idx - TRAIN_DAYS)
    train_end = start_idx
    fwd_start = start_idx
    fwd_end = min(start_idx + ADVANCE_DAYS, len(features))

    if fwd_end <= fwd_start:
        continue

    train_dates = features.index[train_start:train_end]
    fwd_dates = features.index[fwd_start:fwd_end]

    # For each income ETF, predict: will it be in top-N next period?
    # Label: rank by forward return, top-N = 1, rest = 0
    fwd_rets_train = {}
    for t in valid_income:
        if t in prices.columns:
            fwd_rets_train[t] = prices[t].pct_change(LABEL_HORIZON).shift(-LABEL_HORIZON)

    if not fwd_rets_train:
        continue

    fwd_df = pd.DataFrame(fwd_rets_train)
    common_dates = train_dates.intersection(fwd_df.dropna().index).intersection(features.index)

    if len(common_dates) < 50:
        continue

    # Build ML dataset: for each date, predict which ETFs will outperform
    X_train_list = []
    y_train_list = []

    for dt_idx in common_dates:
        row_rets = fwd_df.loc[dt_idx].dropna()
        if len(row_rets) < N_TOP:
            continue

        # Top-N performers
        top_n = row_rets.nlargest(N_TOP).index.tolist()

        # For each ETF, create a sample
        for t in valid_income:
            if t not in row_rets.index:
                continue
            feat_cols = [c for c in features.columns if c.startswith(f'{t}_') or not any(c.startswith(f'{vt}_') for vt in valid_income)]
            if dt_idx in features.index:
                x = features.loc[dt_idx, feat_cols].fillna(0).values
                y = 1 if t in top_n else 0
                X_train_list.append(x)
                y_train_list.append(y)

    if len(X_train_list) < 20:
        # Equal weight fallback
        for date in fwd_dates:
            all_dates.append(date)
            fwd_day_rets = returns.loc[date, valid_income].mean() if date in returns.index else 0
            all_ml_returns.append(fwd_day_rets)
            all_ew_returns.append(fwd_day_rets)
            spy_ret = returns.loc[date, BENCHMARK] if date in returns.index and BENCHMARK in returns.columns else 0
            all_spy_returns.append(spy_ret)
        continue

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    if len(set(y_train)) < 2:
        continue

    # Train model
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, n_jobs=1
    )
    model.fit(X_train, y_train)

    # Predict: score each ETF at rebalancing date
    scores = {}
    rebal_date = features.index[start_idx]
    for t in valid_income:
        feat_cols = [c for c in features.columns if c.startswith(f'{t}_') or not any(c.startswith(f'{vt}_') for vt in valid_income)]
        if rebal_date in features.index:
            x = features.loc[rebal_date, feat_cols].fillna(0).values.reshape(1, -1)
            prob = model.predict_proba(x)[0, 1]
            scores[t] = prob

    if not scores:
        continue

    # Select top-N
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [t for t, s in sorted_scores[:N_TOP]]
    all_selections.append({'date': str(rebal_date.date()), 'selected': selected,
                           'scores': {t: round(s, 3) for t, s in sorted_scores}})

    # Apply for forward period
    for date in fwd_dates:
        if date not in returns.index:
            continue

        # ML selection: equal weight top-N
        ml_ret = returns.loc[date, selected].mean() if all(s in returns.columns for s in selected) else 0
        all_ml_returns.append(ml_ret)

        # Equal weight all income ETFs
        ew_ret = returns.loc[date, valid_income].mean()
        all_ew_returns.append(ew_ret)

        # SPY
        spy_ret = returns.loc[date, BENCHMARK] if BENCHMARK in returns.columns else 0
        all_spy_returns.append(spy_ret)

        all_dates.append(date)

    if (idx_num + 1) % 20 == 0:
        print(f"  Period {idx_num+1}/{len(rebal_indices)}")

print(f"\nBacktest complete: {len(all_dates)} trading days, {len(all_selections)} rebalances")

# ─── Compute Metrics ───
def compute_metrics(returns_list, name):
    r = np.array(returns_list, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 20:
        return {'name': name, 'error': 'insufficient data'}

    ann_ret = np.mean(r) * 252
    ann_vol = np.std(r) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg = r[r < 0]
    neg_vol = np.std(neg) * np.sqrt(252) if len(neg) > 0 else 1e-6
    sortino = ann_ret / neg_vol

    cum = np.cumprod(1 + r)
    max_dd = np.min(cum / np.maximum.accumulate(cum) - 1)
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    cagr = cum[-1] ** (252 / len(r)) - 1
    wr = (r > 0).mean()

    pos_sum = r[r > 0].sum() if (r > 0).sum() > 0 else 0
    neg_sum = abs(r[r < 0].sum()) if (r < 0).sum() > 0 else 1e-6
    pf = pos_sum / neg_sum

    return {
        'name': name,
        'Sharpe': round(sharpe, 3),
        'Sortino': round(sortino, 3),
        'CAGR': round(cagr * 100, 1),
        'MaxDD': round(max_dd * 100, 1),
        'Calmar': round(calmar, 2),
        'WR': round(wr * 100, 1),
        'PF': round(pf, 2),
        'n_days': len(r),
    }

ml_m = compute_metrics(all_ml_returns, 'ML Dividend Rotation')
ew_m = compute_metrics(all_ew_returns, 'Equal Weight Income')
spy_m = compute_metrics(all_spy_returns, 'SPY Buy-Hold')

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
for m in [ml_m, ew_m, spy_m]:
    print(f"\n{m['name']}:")
    for k, v in m.items():
        if k != 'name':
            print(f"  {k}: {v}")

# ─── Yearly Returns ───
yearly = {'ML': {}, 'EW': {}, 'SPY': {}}
for d, ml_r, ew_r, spy_r in zip(all_dates, all_ml_returns, all_ew_returns, all_spy_returns):
    yr = d.year
    for key, r in [('ML', ml_r), ('EW', ew_r), ('SPY', spy_r)]:
        if yr not in yearly[key]:
            yearly[key][yr] = []
        yearly[key][yr].append(r)

for key in yearly:
    for yr in yearly[key]:
        yearly[key][yr] = round((np.prod(1 + np.array(yearly[key][yr])) - 1) * 100, 1)

print(f"\nYearly Returns (%):")
years = sorted(set(y for s in yearly.values() for y in s))
neg_years = 0
for yr in years:
    ml_yr = yearly.get('ML', {}).get(yr, '-')
    if isinstance(ml_yr, (int, float)) and ml_yr < 0:
        neg_years += 1
    ew_yr = yearly.get('EW', {}).get(yr, '-')
    spy_yr = yearly.get('SPY', {}).get(yr, '-')
    print(f"  {yr}: ML={ml_yr}%, EW={ew_yr}%, SPY={spy_yr}%")

# ─── Adversarial Validation ───
print(f"\n{'='*70}")
print("ADVERSARIAL VALIDATION")
print(f"{'='*70}")

ml_rets = np.array(all_ml_returns, dtype=float)
ml_rets = ml_rets[~np.isnan(ml_rets)]
real_sharpe = ml_m['Sharpe']

# 1. Permutation
print(f"\n1. PERMUTATION TEST ({N_PERMS} shuffles)...")
perm_sharpes = []
for _ in range(N_PERMS):
    shuffled = ml_rets.copy()
    np.random.shuffle(shuffled)
    p_ann = np.mean(shuffled) * 252
    p_vol = np.std(shuffled) * np.sqrt(252)
    perm_sharpes.append(p_ann / p_vol if p_vol > 0 else 0)

perm_mean = np.mean(perm_sharpes)
perm_p = np.mean([s >= real_sharpe for s in perm_sharpes])
perm_pass = perm_p < 0.05
print(f"  Real: {real_sharpe:.3f}, Perm mean: {perm_mean:.3f}, p={perm_p:.3f}")
print(f"  VERDICT: {'PASS' if perm_pass else 'FAIL'}")

# 2. Sub-period
print(f"\n2. SUB-PERIOD...")
n = len(ml_rets)
sub_sharpes = []
for q in range(4):
    s, e = q * (n//4), (q+1) * (n//4) if q < 3 else n
    qr = ml_rets[s:e]
    qa = np.mean(qr) * 252
    qv = np.std(qr) * np.sqrt(252)
    sub_sharpes.append(round(qa / qv if qv > 0 else 0, 3))
all_pos = all(s > 0 for s in sub_sharpes)
sub_cv = np.std(sub_sharpes) / np.mean(sub_sharpes) if np.mean(sub_sharpes) > 0 else 999
sub_pass = all_pos and sub_cv < 1.0
print(f"  Quarters: {sub_sharpes}, CV={sub_cv:.3f}")
print(f"  VERDICT: {'PASS' if sub_pass else 'FAIL'}")

# 3. Outlier
print(f"\n3. OUTLIER REMOVAL...")
p5, p95 = np.percentile(ml_rets, [5, 95])
trimmed = ml_rets[(ml_rets >= p5) & (ml_rets <= p95)]
ta = np.mean(trimmed) * 252
tv = np.std(trimmed) * np.sqrt(252)
trim_sharpe = ta / tv if tv > 0 else 0
out_deg = (trim_sharpe - real_sharpe) / abs(real_sharpe) * 100 if real_sharpe != 0 else 0
out_pass = out_deg > -30
print(f"  Full: {real_sharpe:.3f}, Trimmed: {trim_sharpe:.3f}, Deg: {out_deg:.1f}%")
print(f"  VERDICT: {'PASS' if out_pass else 'FAIL'}")

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
    print(f"  Green: {g_sr:.3f}, Red: {r_sr:.3f}, Gap: {gap:.3f}")
else:
    r1_pass = False
    g_sr, r_sr, gap = 0, 0, 999
    print(f"  Insufficient regime data")
print(f"  VERDICT: {'PASS' if r1_pass else 'FAIL'}")

gates = sum([perm_pass, sub_pass, out_pass, r1_pass])
print(f"\nGATES: {gates}/4")

# ─── SPY Correlation ───
min_len = min(len(ml_rets), len(spy_r))
spy_corr = np.corrcoef(ml_rets[:min_len], spy_r[:min_len])[0, 1]
print(f"SPY Correlation: {spy_corr:.3f}")

# ─── Dividend Yield Estimate ───
# Approximate from historical data
avg_div_yield = 0.03  # ~3% for dividend ETFs
print(f"Estimated additional dividend yield: ~{avg_div_yield*100:.0f}% on top of price returns")

# ─── Save ───
output = {
    'strategy': 'ML Dividend Aristocrat Rotation',
    'type': 'income-focused dividend ETF rotation',
    'income_etfs': valid_income,
    'n_top': N_TOP,
    'ml_metrics': ml_m,
    'ew_metrics': ew_m,
    'spy_metrics': spy_m,
    'yearly_returns': yearly,
    'negative_years': neg_years,
    'adversarial': {
        'permutation': {'sharpe': real_sharpe, 'perm_mean': round(perm_mean, 3), 'p_value': round(perm_p, 3), 'verdict': 'PASS' if perm_pass else 'FAIL'},
        'sub_period': {'sharpes': sub_sharpes, 'cv': round(sub_cv, 3), 'verdict': 'PASS' if sub_pass else 'FAIL'},
        'outlier': {'full': real_sharpe, 'trimmed': round(trim_sharpe, 3), 'degradation': round(out_deg, 1), 'verdict': 'PASS' if out_pass else 'FAIL'},
        'r1_regime': {'green': round(g_sr, 3), 'red': round(r_sr, 3), 'gap': round(gap, 3), 'verdict': 'PASS' if r1_pass else 'FAIL'},
        'gates_passed': f'{gates}/4',
    },
    'spy_correlation': round(spy_corr, 3),
    'recent_selections': all_selections[-5:] if all_selections else [],
    'estimated_dividend_yield': f'{avg_div_yield*100:.0f}%',
    'timestamp': datetime.now().isoformat(),
}

with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

pd.DataFrame({
    'date': all_dates,
    'ml_return': all_ml_returns,
    'ew_return': all_ew_returns,
    'spy_return': all_spy_returns,
}).to_parquet(OUTPUT / 'daily_returns.parquet', index=False)

print(f"\nResults saved to {OUTPUT}")
print("DONE")
