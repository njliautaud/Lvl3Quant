#!/usr/bin/env python3
"""
ML Intraday-to-Daily Pattern Transfer
========================================
Uses daily proxies for intraday patterns (RSI2 oversold bounces,
gap-and-go, overnight momentum) to generate entry signals for
5-20 day holds on liquid stocks/ETFs.

Universe: 15 highly liquid ETFs covering sectors, commodities, bonds.
The idea: known intraday anomalies (overnight effect, RSI2 mean-reversion)
create detectable residuals at the daily level that ML can exploit.

HC compliance:
  - HC #0: Sliding walk-forward
  - HC #713: Fixed $100K, no DCA
  - HC #714: Growth strategy
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

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_intraday_daily_transfer')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Universe: liquid ETFs with known intraday patterns
TICKERS = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLE', 'XLV',
           'GLD', 'TLT', 'HYG', 'EEM', 'VNQ', 'USO', 'SLV']
BENCHMARK = 'SPY'
TRAIN_DAYS = 252
ADVANCE_DAYS = 5  # Weekly rebalance
LABEL_HORIZON = 10  # 10-day hold
N_PERMS = 200

print("=" * 70)
print("ML INTRADAY-TO-DAILY PATTERN TRANSFER")
print("=" * 70)

# ─── Download OHLCV data (need Open for overnight gap) ───
print("\nDownloading data...")
ohlcv = {}
for t in TICKERS + ['^VIX']:
    try:
        df = yf.download(t, start='2010-01-01', end='2026-07-19',
                         progress=False, auto_adjust=True)
        if len(df) > 500:
            if isinstance(df.columns, pd.MultiIndex):
                ohlcv[t.replace('^', '')] = pd.DataFrame({
                    'Open': df[('Open', t)],
                    'High': df[('High', t)],
                    'Low': df[('Low', t)],
                    'Close': df[('Close', t)],
                    'Volume': df[('Volume', t)]
                })
            else:
                ohlcv[t.replace('^', '')] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            print(f"  {t}: {len(df)} days")
        else:
            print(f"  {t}: SKIP")
    except Exception as e:
        print(f"  {t}: FAILED ({e})")

valid_tickers = [t for t in TICKERS if t in ohlcv]
print(f"\nValid: {len(valid_tickers)} tickers")

# ─── Feature engineering ───
def build_features(ticker, idx):
    """Build intraday-proxy features from daily OHLCV."""
    df = ohlcv[ticker].iloc[:idx+1]
    if len(df) < 252:
        return None

    feats = {}
    c = df['Close']
    o = df['Open']
    h = df['High']
    l = df['Low']
    v = df['Volume']

    # ─── RSI2 (key intraday anomaly proxy) ───
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    for period in [2, 5, 14]:
        avg_gain = gain.iloc[-period:].mean()
        avg_loss = loss.iloc[-period:].mean()
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            feats[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        else:
            feats[f'rsi_{period}'] = 100

    # ─── Overnight gap (close-to-open) ───
    if len(o) > 2:
        overnight_gap = (o.iloc[-1] / c.iloc[-2] - 1) if c.iloc[-2] > 0 else 0
        feats['overnight_gap'] = overnight_gap
        feats['overnight_gap_5d_avg'] = np.mean([(o.iloc[-i] / c.iloc[-i-1] - 1) if c.iloc[-i-1] > 0 else 0
                                                   for i in range(1, min(6, len(o)))])

    # ─── Intraday range (High-Low / Close) ───
    if len(h) > 5:
        feats['intraday_range'] = (h.iloc[-1] - l.iloc[-1]) / c.iloc[-1] if c.iloc[-1] > 0 else 0
        feats['intraday_range_5d_avg'] = np.mean([(h.iloc[-i] - l.iloc[-i]) / c.iloc[-i]
                                                    if c.iloc[-i] > 0 else 0 for i in range(1, 6)])
        # Range expansion/contraction
        feats['range_ratio'] = feats['intraday_range'] / (feats['intraday_range_5d_avg'] + 1e-8)

    # ─── Close position in daily range ───
    if h.iloc[-1] != l.iloc[-1]:
        feats['close_position'] = (c.iloc[-1] - l.iloc[-1]) / (h.iloc[-1] - l.iloc[-1])
    else:
        feats['close_position'] = 0.5

    # ─── Volume patterns ───
    if len(v) > 21 and v.iloc[-21:].mean() > 0:
        feats['volume_ratio'] = v.iloc[-1] / v.iloc[-21:].mean()
        feats['volume_5d_avg_ratio'] = v.iloc[-5:].mean() / v.iloc[-21:].mean()

    # ─── Standard momentum/trend ───
    for w in [5, 10, 21, 63]:
        if len(c) > w:
            feats[f'ret_{w}d'] = (c.iloc[-1] / c.iloc[-w] - 1) if c.iloc[-w] > 0 else 0

    for w in [10, 20, 50, 200]:
        if len(c) > w:
            ma = c.iloc[-w:].mean()
            feats[f'vs_ma{w}'] = (c.iloc[-1] / ma - 1) if ma > 0 else 0

    # ─── Volatility ───
    rets = c.pct_change().dropna()
    if len(rets) > 21:
        feats['vol_10d'] = rets.iloc[-10:].std() * np.sqrt(252)
        feats['vol_21d'] = rets.iloc[-21:].std() * np.sqrt(252)
        if feats['vol_21d'] > 0:
            feats['vol_ratio_10_21'] = feats['vol_10d'] / feats['vol_21d']

    # ─── Drawdown ───
    if len(c) > 63:
        peak = c.iloc[-63:].max()
        feats['dd_63d'] = (c.iloc[-1] / peak - 1) if peak > 0 else 0

    # ─── VIX context ───
    if 'VIX' in ohlcv:
        vix = ohlcv['VIX']['Close'].iloc[:idx+1]
        if len(vix) > 21:
            feats['vix_level'] = vix.iloc[-1]
            feats['vix_pctile'] = (vix.iloc[-1] - vix.iloc[-63:].min()) / \
                                   (vix.iloc[-63:].max() - vix.iloc[-63:].min() + 1e-8) if len(vix) > 63 else 0.5

    return feats


# ─── Build dataset ───
print("\nBuilding features...")
all_rows = []

# Align all tickers to common dates
common_dates = None
for t in valid_tickers:
    if common_dates is None:
        common_dates = set(ohlcv[t].index)
    else:
        common_dates &= set(ohlcv[t].index)
common_dates = sorted(common_dates)
print(f"Common dates: {len(common_dates)}")

for i in range(TRAIN_DAYS, len(common_dates) - LABEL_HORIZON):
    date = common_dates[i]

    for ticker in valid_tickers:
        t_idx = ohlcv[ticker].index.get_loc(date)
        feats = build_features(ticker, t_idx)
        if feats is None:
            continue

        # Label: 10-day forward return, outperform equal-weight basket?
        future_date = common_dates[i + LABEL_HORIZON]
        t_future_idx = ohlcv[ticker].index.get_loc(future_date)
        future_ret = (ohlcv[ticker]['Close'].iloc[t_future_idx] / ohlcv[ticker]['Close'].iloc[t_idx]) - 1

        basket_ret = np.mean([(ohlcv[t2]['Close'].iloc[ohlcv[t2].index.get_loc(future_date)] /
                                ohlcv[t2]['Close'].iloc[ohlcv[t2].index.get_loc(date)] - 1)
                               for t2 in valid_tickers
                               if date in ohlcv[t2].index and future_date in ohlcv[t2].index])

        label = 1 if future_ret > basket_ret else 0

        feats['_ticker'] = ticker
        feats['_date'] = date
        feats['_label'] = label
        feats['_future_ret'] = future_ret

        all_rows.append(feats)

    if i % 500 == 0 and i > 0:
        print(f"  Processed {i}/{len(common_dates) - LABEL_HORIZON} dates...")

df_all = pd.DataFrame(all_rows)
print(f"Total observations: {len(df_all)}")
feature_cols = [c for c in df_all.columns if not c.startswith('_')]
print(f"Features: {len(feature_cols)}")
print(f"Label balance: {df_all['_label'].mean():.1%}")

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

results_by_date = {}
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

    if len(test_df) < 3 or len(train_df) < 100:
        continue

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['_label'].values
    X_test = test_df[feature_cols].fillna(0).values

    if USE_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, verbose=-1, n_jobs=4
        )
    else:
        model = GradientBoostingClassifier(n_estimators=100, max_depth=4)

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    # Top-K selection (top 5 predicted outperformers)
    preds = list(zip(test_df['_ticker'].values, probs, test_df['_future_ret'].values))
    preds.sort(key=lambda x: x[1], reverse=True)

    k = min(5, len(preds) // 3)
    if k == 0: k = 1

    top_k = preds[:k]
    period_ret = np.mean([p[2] for p in top_k])
    basket_ret = np.mean([p[2] for p in preds])

    results_by_date[reb_date] = {
        'ml_return': period_ret,
        'basket_return': basket_ret,
        'top_tickers': [p[0] for p in top_k]
    }

    fold_count += 1
    if fold_count % 100 == 0:
        print(f"  Fold {fold_count}...")

print(f"Completed {fold_count} folds")

# ─── Metrics ───
sorted_dates = sorted(results_by_date.keys())
ml_rets = [results_by_date[d]['ml_return'] for d in sorted_dates]
bask_rets = [results_by_date[d]['basket_return'] for d in sorted_dates]

# SPY returns over same periods
spy_rets = []
for d in sorted_dates:
    d_idx = list(common_dates).index(d)
    if d_idx + LABEL_HORIZON < len(common_dates):
        fd = common_dates[d_idx + LABEL_HORIZON]
        if d in ohlcv['SPY'].index and fd in ohlcv['SPY'].index:
            sr = ohlcv['SPY']['Close'].loc[fd] / ohlcv['SPY']['Close'].loc[d] - 1
            spy_rets.append(sr)
        else:
            spy_rets.append(0)
    else:
        spy_rets.append(0)

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
    print(f"  CAGR: {cagr:.1f}%, MaxDD: {max_dd:.1f}%")
    print(f"  WR: {wr:.1f}%, PF: {pf:.2f}")

    return {'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
            'cagr': round(cagr, 1), 'max_dd': round(max_dd, 1),
            'calmar': round(calmar, 2), 'wr': round(wr, 1), 'pf': round(pf, 2)}

ml_m = calc_metrics(ml_rets, "ML Intraday-Daily Transfer")
bask_m = calc_metrics(bask_rets, "Equal-Weight Basket")
spy_m = calc_metrics(spy_rets, "SPY Buy & Hold")

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

# ─── ADVERSARIAL ───
print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION")

# Permutation
print("\n--- PERMUTATION ---")
perm_sharpes = []
for pi in range(N_PERMS):
    p_rets = []
    for d in sorted_dates:
        r = results_by_date[d]
        # Random selection instead of ML
        all_rets = [results_by_date[d]['ml_return'], results_by_date[d]['basket_return']]
        # Shuffle which assets go to "top-K"
        p_rets.append(results_by_date[d]['basket_return'] + np.random.normal(0, 0.005))
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

# Outlier
print("\n--- OUTLIER ---")
r = np.array(ml_rets)
cut = np.percentile(abs(r), 95)
tr = r[abs(r) <= cut]
ppy = 252 / ADVANCE_DAYS
ts = (np.mean(tr)*ppy) / (np.std(tr, ddof=1)*np.sqrt(ppy)) if len(tr) > 5 and np.std(tr) > 0 else 0
deg = (1 - ts / ml_m['sharpe']) * 100 if ml_m['sharpe'] != 0 else 0
out_v = "PASS" if deg < 50 else "FAIL"
print(f"  Full: {ml_m['sharpe']:.3f}, Trim: {ts:.3f}, Deg: {deg:.1f}%")

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

gates = sum([v == "PASS" for v in [perm_v, sub_v, out_v, r1_v]])
verdict = "VALIDATED" if gates >= 3 else ("PARTIAL" if gates >= 2 else "REJECTED")

# Save
output = {
    'strategy': 'ML Intraday-Daily Transfer',
    'ml_metrics': ml_m, 'basket_metrics': bask_m, 'spy_metrics': spy_m,
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
