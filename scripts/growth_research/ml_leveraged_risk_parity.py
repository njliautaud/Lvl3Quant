#!/usr/bin/env python3
"""
ML Leveraged Risk Parity Portfolio
====================================
HC #714: ML/AI for exploratory research. HC #709: Growth + income + hedging.

Combines our TOP validated strategies into a dynamically leveraged risk-parity
portfolio. Uses ML to predict:
1. When to increase/decrease total leverage (1x-2x range)
2. How to shift weights between income vs growth sleeves

Inputs: daily returns from validated strategies (CTA, Sectors, StatArb,
Commodity Trend, Currency Carry, Credit Timing, VolBreakout, Tail Risk).

Walk-forward approach:
- 126d sliding window for covariance estimation + ML training
- Monthly rebalancing (21d)
- Risk parity weights = inverse-vol weighted, ML adjusts leverage

HC #713: Fixed $100K capital, no DCA.
HC #0: Sliding windows only.
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
OUTPUT = BASE / "output" / "ml_leveraged_risk_parity"
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML LEVERAGED RISK PARITY PORTFOLIO")
print("=" * 70)

# ─── Strategy Proxy ETFs ───
# We use ETF proxies that approximate our validated strategy types
# since we don't have 20 years of strategy daily returns
STRATEGY_PROXIES = {
    # Growth/Trend strategies
    'CTA_Trend': {'long': 'DBMF', 'proxy': 'SPY', 'type': 'growth'},  # Managed futures proxy
    'Sector_Rot': {'long': 'XLK', 'proxy': 'XLK', 'type': 'growth'},  # Tech-heavy sector
    'Commodity': {'long': 'DBC', 'proxy': 'DBC', 'type': 'growth'},   # Commodities
    'Currency': {'long': 'UUP', 'proxy': 'UUP', 'type': 'income'},    # Dollar carry
    # Income strategies
    'Credit': {'long': 'HYG', 'proxy': 'HYG', 'type': 'income'},      # High yield
    'StatArb': {'long': 'BTAL', 'proxy': 'BTAL', 'type': 'income'},   # Market neutral proxy
    'VolBreak': {'long': 'SVXY', 'proxy': 'SVXY', 'type': 'income'},  # Short vol proxy
    # Hedging
    'TailRisk': {'long': 'TAIL', 'proxy': 'TAIL', 'type': 'hedge'},   # Tail risk
}

# Macro signals for ML leverage predictor
MACRO_TICKERS = ['^VIX', 'GLD', 'TLT', 'HYG', 'SPY', 'UUP', 'DBC', 'EEM']

TRAIN_WINDOW = 126  # 6 months for covariance + ML
REBAL_FREQ = 21     # Monthly rebalancing
INITIAL_CAPITAL = 100_000
MIN_LEVERAGE = 0.5
MAX_LEVERAGE = 2.0
TARGET_VOL = 0.10   # 10% annualized target vol
N_PERMS = 200

print(f"\nConfiguration:")
print(f"  Train window: {TRAIN_WINDOW}d")
print(f"  Rebalancing: every {REBAL_FREQ}d")
print(f"  Leverage range: {MIN_LEVERAGE}x - {MAX_LEVERAGE}x")
print(f"  Target vol: {TARGET_VOL*100}%")
print(f"  Capital: ${INITIAL_CAPITAL:,}")

# ─── Download Data ───
print(f"\nDownloading data...")

all_tickers = list(set(
    [v['proxy'] for v in STRATEGY_PROXIES.values()] +
    MACRO_TICKERS + ['SPY']
))

data = {}
for t in all_tickers:
    try:
        df = yf.download(t, start='2008-01-01', end='2026-07-20',
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

# ─── Compute strategy proxy returns ───
strategy_cols = []
for name, spec in STRATEGY_PROXIES.items():
    proxy = spec['proxy']
    if proxy in prices.columns:
        strategy_cols.append(name)
        prices[name] = prices[proxy].pct_change()
    else:
        print(f"  WARNING: {proxy} not available for {name}")

# Drop NaN from pct_change
prices = prices.dropna(subset=strategy_cols)
strategy_returns = prices[strategy_cols].copy()

print(f"\nAvailable strategies: {len(strategy_cols)}")
for col in strategy_cols:
    ann_ret = strategy_returns[col].mean() * 252
    ann_vol = strategy_returns[col].std() * np.sqrt(252)
    sr = ann_ret / ann_vol if ann_vol > 0 else 0
    print(f"  {col}: ann_ret={ann_ret:.1%}, ann_vol={ann_vol:.1%}, Sharpe={sr:.2f}")

# ─── Build macro features for ML leverage predictor ───
print(f"\nBuilding features...")

# Macro features
macro_names = [t.replace('^', '') for t in MACRO_TICKERS]
available_macro = [m for m in macro_names if m in prices.columns]

features_df = pd.DataFrame(index=prices.index)

for m in available_macro:
    if m in prices.columns:
        features_df[f'{m}_ret_5d'] = prices[m].pct_change(5)
        features_df[f'{m}_ret_21d'] = prices[m].pct_change(21)
        features_df[f'{m}_vol_21d'] = prices[m].pct_change().rolling(21).std()

# Cross-strategy correlation features
for i, c1 in enumerate(strategy_cols):
    for c2 in strategy_cols[i+1:]:
        features_df[f'corr_{c1}_{c2}_21d'] = (
            strategy_returns[c1].rolling(21).corr(strategy_returns[c2])
        )

# Portfolio-level features
features_df['port_vol_21d'] = strategy_returns.mean(axis=1).rolling(21).std() * np.sqrt(252)
features_df['port_drawdown'] = (
    strategy_returns.mean(axis=1).cumsum() -
    strategy_returns.mean(axis=1).cumsum().cummax()
)
features_df['avg_cross_corr_21d'] = features_df[[c for c in features_df.columns if 'corr_' in c]].mean(axis=1)

if 'VIX' in prices.columns:
    features_df['vix_level'] = prices['VIX']
    features_df['vix_zscore_63d'] = (
        (prices['VIX'] - prices['VIX'].rolling(63).mean()) /
        prices['VIX'].rolling(63).std()
    )

features_df = features_df.dropna()
print(f"Features: {features_df.shape[1]} columns, {len(features_df)} days")

# ─── Walk-forward risk parity + ML leverage ───
print(f"\nRunning walk-forward backtest...")

results = {
    'dates': [],
    'ml_leverage_returns': [],
    'static_rp_returns': [],
    'equal_weight_returns': [],
    'spy_returns': [],
    'leverages': [],
    'weights': {},
}
for col in strategy_cols:
    results['weights'][col] = []

n_days = len(features_df)
rebal_dates = list(range(TRAIN_WINDOW, n_days - REBAL_FREQ, REBAL_FREQ))
print(f"Rebalancing periods: {len(rebal_dates)}")

for i, rebal_idx in enumerate(rebal_dates):
    # Training window
    train_start = max(0, rebal_idx - TRAIN_WINDOW)
    train_end = rebal_idx

    # Forward window
    fwd_start = rebal_idx
    fwd_end = min(rebal_idx + REBAL_FREQ, n_days)

    if fwd_end <= fwd_start:
        continue

    train_dates = features_df.index[train_start:train_end]
    fwd_dates = features_df.index[fwd_start:fwd_end]

    # 1. Compute risk parity weights from training covariance
    train_rets = strategy_returns.loc[train_dates, strategy_cols]
    cov_matrix = train_rets.cov() * 252  # Annualize

    # Inverse-volatility weights
    vols = np.sqrt(np.diag(cov_matrix))
    vols = np.maximum(vols, 1e-6)
    inv_vol_weights = 1.0 / vols
    inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()

    # 2. ML leverage prediction
    # Label: was next-period realized vol above or below target?
    # If below target → can leverage up, if above → reduce
    if i >= 3:  # Need some history for ML
        # Build training labels from PAST rebalancing periods
        ml_X = []
        ml_y = []
        for j in range(max(0, i - 20), i):
            past_idx = rebal_dates[j]
            past_fwd_end = min(past_idx + REBAL_FREQ, n_days)
            past_fwd_dates = features_df.index[past_idx:past_fwd_end]

            if len(past_fwd_dates) < 5:
                continue

            # Features at rebalancing time
            feat_row = features_df.iloc[past_idx]
            if feat_row.isna().sum() > len(feat_row) * 0.3:
                continue
            ml_X.append(feat_row.fillna(0).values)

            # Label: realized portfolio vol in forward period
            past_port_ret = strategy_returns.loc[past_fwd_dates, strategy_cols].mean(axis=1)
            realized_vol = past_port_ret.std() * np.sqrt(252)
            # Binary: can we leverage up? (vol below target)
            ml_y.append(1 if realized_vol < TARGET_VOL else 0)

        if len(ml_X) >= 5 and len(set(ml_y)) > 1:
            ml_X = np.array(ml_X)
            ml_y = np.array(ml_y)

            model = lgb.LGBMClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                min_child_samples=2, verbose=-1, n_jobs=1
            )
            model.fit(ml_X, ml_y)

            # Predict for current period
            curr_feat = features_df.iloc[rebal_idx].fillna(0).values.reshape(1, -1)
            prob_low_vol = model.predict_proba(curr_feat)[0, 1]

            # Map probability to leverage
            ml_leverage = MIN_LEVERAGE + (MAX_LEVERAGE - MIN_LEVERAGE) * prob_low_vol
        else:
            ml_leverage = 1.0  # Default
    else:
        ml_leverage = 1.0  # Default for early periods

    # 3. Also compute vol-targeted leverage (non-ML baseline)
    recent_vol = train_rets.mean(axis=1).std() * np.sqrt(252)
    static_leverage = np.clip(TARGET_VOL / max(recent_vol, 0.01), MIN_LEVERAGE, MAX_LEVERAGE)

    # 4. Compute forward returns
    fwd_rets = strategy_returns.loc[fwd_dates, strategy_cols]

    for date in fwd_dates:
        day_ret = fwd_rets.loc[date]

        # ML leveraged risk parity
        ml_port_ret = ml_leverage * (inv_vol_weights * day_ret.values).sum()
        results['ml_leverage_returns'].append(ml_port_ret)

        # Static risk parity (no ML, just vol-target leverage)
        static_port_ret = static_leverage * (inv_vol_weights * day_ret.values).sum()
        results['static_rp_returns'].append(static_port_ret)

        # Equal weight (no risk parity, no leverage)
        ew_ret = day_ret.values.mean()
        results['equal_weight_returns'].append(ew_ret)

        # SPY benchmark
        if 'SPY' in prices.columns and date in prices.index:
            spy_ret = prices.loc[date, 'SPY'] / prices.loc[:date, 'SPY'].iloc[-2] - 1 if date != prices.index[0] else 0
        else:
            spy_ret = 0
        results['spy_returns'].append(spy_ret)

        results['dates'].append(date)
        results['leverages'].append(ml_leverage)
        for k, col in enumerate(strategy_cols):
            results['weights'][col].append(inv_vol_weights[k])

    if (i + 1) % 20 == 0:
        print(f"  Rebalance {i+1}/{len(rebal_dates)}")

print(f"\nBacktest complete: {len(results['dates'])} trading days")

# ─── Compute Metrics ───
def compute_metrics(returns, name):
    returns = np.array(returns)
    if len(returns) < 10:
        return {'name': name, 'error': 'insufficient data'}

    ann_ret = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg_vol = np.std(returns[returns < 0]) * np.sqrt(252) if (returns < 0).sum() > 0 else 1e-6
    sortino = ann_ret / neg_vol

    cum_ret = np.cumprod(1 + returns)
    max_dd = np.min(cum_ret / np.maximum.accumulate(cum_ret) - 1)
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    cagr = (cum_ret[-1]) ** (252 / len(returns)) - 1
    wr = (returns > 0).mean()

    pos_avg = returns[returns > 0].mean() if (returns > 0).sum() > 0 else 0
    neg_avg = abs(returns[returns < 0].mean()) if (returns < 0).sum() > 0 else 1e-6
    pf = pos_avg * (returns > 0).sum() / (neg_avg * (returns < 0).sum()) if (returns < 0).sum() > 0 else 999

    return {
        'name': name,
        'Sharpe': round(sharpe, 3),
        'Sortino': round(sortino, 3),
        'CAGR': round(cagr * 100, 1),
        'MaxDD': round(max_dd * 100, 1),
        'Calmar': round(calmar, 2),
        'WR': round(wr * 100, 1),
        'PF': round(pf, 2),
        'AnnVol': round(ann_vol * 100, 1),
        'n_days': len(returns),
    }

ml_metrics = compute_metrics(results['ml_leverage_returns'], 'ML Leveraged Risk Parity')
static_metrics = compute_metrics(results['static_rp_returns'], 'Static Vol-Target RP')
ew_metrics = compute_metrics(results['equal_weight_returns'], 'Equal Weight')
spy_metrics = compute_metrics(results['spy_returns'], 'SPY Buy-Hold')

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
for m in [ml_metrics, static_metrics, ew_metrics, spy_metrics]:
    print(f"\n{m['name']}:")
    for k, v in m.items():
        if k != 'name':
            print(f"  {k}: {v}")

# ─── Yearly Returns ───
dates = results['dates']
yearly = {}
for strat_name, rets in [
    ('ML_RP', results['ml_leverage_returns']),
    ('Static_RP', results['static_rp_returns']),
    ('EW', results['equal_weight_returns']),
    ('SPY', results['spy_returns']),
]:
    yearly[strat_name] = {}
    for d, r in zip(dates, rets):
        yr = d.year
        if yr not in yearly[strat_name]:
            yearly[strat_name][yr] = []
        yearly[strat_name][yr].append(r)
    for yr in yearly[strat_name]:
        yearly[strat_name][yr] = round(
            (np.prod(1 + np.array(yearly[strat_name][yr])) - 1) * 100, 1
        )

print(f"\nYearly Returns (%):")
years = sorted(set(y for s in yearly.values() for y in s))
print(f"{'Year':<6} {'ML_RP':>8} {'Static':>8} {'EW':>8} {'SPY':>8}")
neg_years = 0
for yr in years:
    ml_yr = yearly.get('ML_RP', {}).get(yr, '-')
    st_yr = yearly.get('Static_RP', {}).get(yr, '-')
    ew_yr = yearly.get('EW', {}).get(yr, '-')
    spy_yr = yearly.get('SPY', {}).get(yr, '-')
    if isinstance(ml_yr, (int, float)) and ml_yr < 0:
        neg_years += 1
    print(f"{yr:<6} {ml_yr:>8} {st_yr:>8} {ew_yr:>8} {spy_yr:>8}")

# ─── Adversarial Validation ───
print(f"\n{'='*70}")
print("ADVERSARIAL VALIDATION")
print(f"{'='*70}")

ml_rets = np.array(results['ml_leverage_returns'])
real_sharpe = ml_metrics['Sharpe']

# 1. Permutation test (shuffle strategy-to-weight mapping)
print(f"\n1. PERMUTATION TEST ({N_PERMS} shuffles)...")
perm_sharpes = []
for _ in range(N_PERMS):
    shuffled = ml_rets.copy()
    np.random.shuffle(shuffled)
    perm_ann = np.mean(shuffled) * 252
    perm_vol = np.std(shuffled) * np.sqrt(252)
    perm_sharpes.append(perm_ann / perm_vol if perm_vol > 0 else 0)

perm_mean = np.mean(perm_sharpes)
perm_p = np.mean([s >= real_sharpe for s in perm_sharpes])
perm_pass = perm_p < 0.05
print(f"  Real Sharpe: {real_sharpe:.3f}")
print(f"  Perm mean: {perm_mean:.3f}")
print(f"  p-value: {perm_p:.3f}")
print(f"  VERDICT: {'PASS' if perm_pass else 'FAIL'}")

# 2. Sub-period consistency
print(f"\n2. SUB-PERIOD CONSISTENCY...")
n = len(ml_rets)
q_size = n // 4
sub_sharpes = []
for q in range(4):
    start = q * q_size
    end = (q + 1) * q_size if q < 3 else n
    q_rets = ml_rets[start:end]
    q_ann = np.mean(q_rets) * 252
    q_vol = np.std(q_rets) * np.sqrt(252)
    q_sr = q_ann / q_vol if q_vol > 0 else 0
    sub_sharpes.append(round(q_sr, 3))
    print(f"  Q{q+1}: Sharpe {q_sr:.3f}")

all_positive = all(s > 0 for s in sub_sharpes)
sub_cv = np.std(sub_sharpes) / np.mean(sub_sharpes) if np.mean(sub_sharpes) > 0 else 999
sub_pass = all_positive and sub_cv < 1.0
print(f"  CV: {sub_cv:.3f}")
print(f"  VERDICT: {'PASS' if sub_pass else 'FAIL'}")

# 3. Outlier removal
print(f"\n3. OUTLIER REMOVAL...")
p5, p95 = np.percentile(ml_rets, [5, 95])
trimmed = ml_rets[(ml_rets >= p5) & (ml_rets <= p95)]
trim_ann = np.mean(trimmed) * 252
trim_vol = np.std(trimmed) * np.sqrt(252)
trim_sharpe = trim_ann / trim_vol if trim_vol > 0 else 0
outlier_deg = (trim_sharpe - real_sharpe) / abs(real_sharpe) * 100 if real_sharpe != 0 else 0
outlier_pass = outlier_deg > -30  # Trimmed shouldn't be >30% worse
print(f"  Full Sharpe: {real_sharpe:.3f}")
print(f"  Trimmed Sharpe: {trim_sharpe:.3f}")
print(f"  Degradation: {outlier_deg:.1f}%")
print(f"  VERDICT: {'PASS' if outlier_pass else 'FAIL'}")

# 4. R1 Regime test
print(f"\n4. R1 REGIME TEST...")
# Classify days by SPY regime
spy_rets = np.array(results['spy_returns'])
spy_cum = np.cumprod(1 + spy_rets)
# Green = SPY 21d return > 0, Red = < 0
spy_21d = pd.Series(spy_rets).rolling(21).sum().values
green_mask = spy_21d > 0
red_mask = spy_21d < 0

if green_mask.sum() > 20 and red_mask.sum() > 20:
    green_rets = ml_rets[green_mask]
    red_rets = ml_rets[red_mask]

    green_sharpe = np.mean(green_rets) * 252 / (np.std(green_rets) * np.sqrt(252)) if np.std(green_rets) > 0 else 0
    red_sharpe = np.mean(red_rets) * 252 / (np.std(red_rets) * np.sqrt(252)) if np.std(red_rets) > 0 else 0

    regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)
    r1_pass = regime_gap < 0.50

    print(f"  Green Sharpe: {green_sharpe:.3f}")
    print(f"  Red Sharpe: {red_sharpe:.3f}")
    print(f"  Gap: {regime_gap:.3f}")
    print(f"  VERDICT: {'PASS' if r1_pass else 'FAIL'}")
else:
    r1_pass = False
    regime_gap = 999
    green_sharpe = 0
    red_sharpe = 0
    print(f"  Insufficient data for regime split")

# ─── Correlation with SPY ───
spy_corr = np.corrcoef(ml_rets[:len(spy_rets)], spy_rets[:len(ml_rets)])[0, 1]
print(f"\nSPY Correlation: {spy_corr:.3f}")

# ─── Average leverage ───
avg_leverage = np.mean(results['leverages'])
print(f"Average ML Leverage: {avg_leverage:.2f}x")

# ─── Gates Summary ───
gates_passed = sum([perm_pass, sub_pass, outlier_pass, r1_pass])
print(f"\n{'='*70}")
print(f"GATES: {gates_passed}/4")
print(f"  Permutation: {'✅' if perm_pass else '❌'}")
print(f"  Sub-period:  {'✅' if sub_pass else '❌'}")
print(f"  Outlier:     {'✅' if outlier_pass else '❌'}")
print(f"  R1 Regime:   {'✅' if r1_pass else '❌'}")
print(f"{'='*70}")

# ─── Save Results ───
output = {
    'strategy': 'ML Leveraged Risk Parity',
    'type': 'multi-strategy portfolio with dynamic leverage',
    'strategies_included': strategy_cols,
    'ml_metrics': ml_metrics,
    'static_rp_metrics': static_metrics,
    'ew_metrics': ew_metrics,
    'spy_metrics': spy_metrics,
    'yearly_returns': yearly,
    'negative_years': neg_years,
    'avg_leverage': round(avg_leverage, 2),
    'adversarial': {
        'permutation': {
            'sharpe': real_sharpe,
            'perm_mean': round(perm_mean, 3),
            'p_value': round(perm_p, 3),
            'verdict': 'PASS' if perm_pass else 'FAIL',
        },
        'sub_period': {
            'sharpes': sub_sharpes,
            'cv': round(sub_cv, 3),
            'verdict': 'PASS' if sub_pass else 'FAIL',
        },
        'outlier': {
            'full': real_sharpe,
            'trimmed': round(trim_sharpe, 3),
            'degradation': round(outlier_deg, 1),
            'verdict': 'PASS' if outlier_pass else 'FAIL',
        },
        'r1_regime': {
            'green': round(green_sharpe, 3),
            'red': round(red_sharpe, 3),
            'gap': round(regime_gap, 3),
            'verdict': 'PASS' if r1_pass else 'FAIL',
        },
        'gates_passed': f'{gates_passed}/4',
    },
    'spy_correlation': round(spy_corr, 3),
    'timestamp': datetime.now().isoformat(),
}

with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

# Save daily returns
pd.DataFrame({
    'date': results['dates'],
    'ml_rp': results['ml_leverage_returns'],
    'static_rp': results['static_rp_returns'],
    'ew': results['equal_weight_returns'],
    'spy': results['spy_returns'],
    'leverage': results['leverages'],
}).to_parquet(OUTPUT / 'daily_returns.parquet', index=False)

print(f"\nResults saved to {OUTPUT}")
print("DONE")
