"""
Lane 3: Volatility Regime Switching
=====================================
Strategy:
- Predict vol regimes (low/medium/high) using VIX, realized vol, term structure
- Low vol: leveraged equity (TQQQ/UPRO)
- Medium vol: unleveraged equity (QQQ/SPY)
- High vol: bonds/cash (TLT/SHY)
- Walk-forward: does regime prediction add value vs static allocation?

Data quality: VIX available from CBOE via yfinance. TQQQ inception 2010.
We use QQQ as a proxy and apply 3x leverage mathematically for pre-TQQQ period.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading data for Vol Regime Switching strategy...")

tickers_needed = {
    'QQQ': 'equity',
    'SPY': 'equity',
    'TLT': 'bonds',
    'SHY': 'cash_proxy',
    '^VIX': 'vix',
    'TQQQ': 'leveraged',
    'UPRO': 'leveraged',
}

data = {}
for ticker, role in tickers_needed.items():
    try:
        df = yf.download(ticker, start='2011-01-01', end='2026-07-11', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df
        print(f"  {ticker} ({role}): {len(df)} days")
    except Exception as e:
        print(f"  {ticker} FAILED: {e}")

# Build aligned dataset
closes = pd.DataFrame({
    'QQQ': data['QQQ']['Close'],
    'SPY': data['SPY']['Close'],
    'TLT': data['TLT']['Close'],
    'SHY': data['SHY']['Close'],
    'VIX': data['^VIX']['Close'],
})
if 'TQQQ' in data:
    closes['TQQQ'] = data['TQQQ']['Close']
if 'UPRO' in data:
    closes['UPRO'] = data['UPRO']['Close']

closes = closes.dropna(subset=['QQQ', 'SPY', 'TLT', 'SHY', 'VIX'])
closes = closes.ffill()

returns = closes.pct_change()

# Synthetic 3x leveraged if needed (for early period)
if 'TQQQ' not in closes.columns:
    returns['TQQQ'] = returns['QQQ'] * 3 - 0.01/252  # Rough expense + borrowing cost
if 'UPRO' not in closes.columns:
    returns['UPRO'] = returns['SPY'] * 3 - 0.01/252

print(f"  Aligned dataset: {len(closes)} days")

# --- Regime Features ---
def compute_regime_features(closes_df):
    """Compute features for regime prediction."""
    vix = closes_df['VIX']
    spy_ret = closes_df['SPY'].pct_change()

    features = pd.DataFrame(index=closes_df.index)

    # VIX level
    features['vix'] = vix
    features['vix_ma20'] = vix.rolling(20).mean()
    features['vix_zscore'] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()

    # VIX term structure proxy: VIX vs 20d realized vol
    rv_20 = spy_ret.rolling(20).std() * np.sqrt(252) * 100
    features['vix_rv_ratio'] = vix / rv_20.clip(lower=1)

    # Realized vol at different horizons
    features['rv_5'] = spy_ret.rolling(5).std() * np.sqrt(252)
    features['rv_20'] = spy_ret.rolling(20).std() * np.sqrt(252)
    features['rv_60'] = spy_ret.rolling(60).std() * np.sqrt(252)

    # Vol of vol
    features['vov'] = vix.pct_change().rolling(20).std()

    # SPY trend
    features['spy_sma50'] = (closes_df['SPY'] / closes_df['SPY'].rolling(50).mean()) - 1
    features['spy_sma200'] = (closes_df['SPY'] / closes_df['SPY'].rolling(200).mean()) - 1

    # Credit stress proxy: TLT/SPY ratio change (flight to safety)
    features['tlt_spy_ratio'] = (closes_df['TLT'] / closes_df['SPY']).pct_change(5)

    return features.dropna()


def classify_regime_simple(vix_val, rv_20_val, vix_ma20_val):
    """Simple rule-based regime classification."""
    if vix_val < 15 and rv_20_val < 0.12:
        return 'low'
    elif vix_val > 25 or rv_20_val > 0.25:
        return 'high'
    else:
        return 'medium'


def classify_regime_trained(features_row, thresholds):
    """Trained regime classification using optimized thresholds."""
    vix = features_row.get('vix', 20)
    rv20 = features_row.get('rv_20', 0.15)
    vix_z = features_row.get('vix_zscore', 0)
    spy_trend = features_row.get('spy_sma200', 0)

    if vix < thresholds['vix_low'] and rv20 < thresholds['rv_low'] and spy_trend > -0.05:
        return 'low'
    elif vix > thresholds['vix_high'] or rv20 > thresholds['rv_high'] or vix_z > thresholds['vix_z_high']:
        return 'high'
    else:
        return 'medium'


# --- Walk-Forward ---
TRAIN_DAYS = 252
TEST_DAYS = 21

features = compute_regime_features(closes)

# Threshold candidates for training
THRESHOLD_SETS = [
    {'vix_low': 14, 'rv_low': 0.10, 'vix_high': 25, 'rv_high': 0.22, 'vix_z_high': 1.5},
    {'vix_low': 15, 'rv_low': 0.12, 'vix_high': 22, 'rv_high': 0.20, 'vix_z_high': 1.2},
    {'vix_low': 16, 'rv_low': 0.13, 'vix_high': 20, 'rv_high': 0.18, 'vix_z_high': 1.0},
    {'vix_low': 13, 'rv_low': 0.11, 'vix_high': 28, 'rv_high': 0.25, 'vix_z_high': 2.0},
    {'vix_low': 15, 'rv_low': 0.12, 'vix_high': 24, 'rv_high': 0.22, 'vix_z_high': 1.5},
]

# Allocation per regime
ALLOCATIONS = {
    'low': {'TQQQ': 0.5, 'UPRO': 0.5},       # Full leverage
    'medium': {'QQQ': 0.5, 'SPY': 0.5},        # Unleveraged
    'high': {'TLT': 0.5, 'SHY': 0.5},          # Safety
}

common_idx = features.index.intersection(returns.index)
features = features.loc[common_idx]
returns_aligned = returns.loc[common_idx]

print("\nRunning walk-forward vol regime switching...")

oot_returns_strategy = []
oot_returns_benchmark = []  # 60/40 SPY/TLT
oot_returns_spy = []
oot_dates = []
window_results = []
regime_counts = {'low': 0, 'medium': 0, 'high': 0}

start = max(TRAIN_DAYS, 252)
while start + TEST_DAYS <= len(features):
    train_slice = slice(start - TRAIN_DAYS, start)

    # Train: find best threshold set
    best_thresh = THRESHOLD_SETS[0]
    best_sharpe = -999

    for ts in THRESHOLD_SETS:
        train_rets = []
        for t in range(train_slice.start, train_slice.stop):
            feat_row = features.iloc[t]
            regime = classify_regime_trained(feat_row, ts)
            alloc = ALLOCATIONS[regime]
            day_ret = sum(returns_aligned[asset].iloc[t] * w for asset, w in alloc.items()
                         if asset in returns_aligned.columns)
            train_rets.append(day_ret)

        tr = np.array(train_rets)
        if len(tr) > 10 and tr.std() > 0:
            s = tr.mean() / tr.std() * np.sqrt(252)
            if s > best_sharpe:
                best_sharpe = s
                best_thresh = ts

    # Test OOT
    test_end = min(start + TEST_DAYS, len(features))
    for t in range(start, test_end):
        feat_row = features.iloc[t]
        regime = classify_regime_trained(feat_row, best_thresh)
        regime_counts[regime] += 1

        alloc = ALLOCATIONS[regime]
        day_ret = sum(returns_aligned[asset].iloc[t] * w for asset, w in alloc.items()
                     if asset in returns_aligned.columns)
        oot_returns_strategy.append(day_ret)

        # Benchmark: 60/40
        bench_ret = returns_aligned['SPY'].iloc[t] * 0.6 + returns_aligned['TLT'].iloc[t] * 0.4
        oot_returns_benchmark.append(bench_ret)

        spy_ret_day = returns_aligned['SPY'].iloc[t]
        oot_returns_spy.append(spy_ret_day)

        oot_dates.append(features.index[t])

    window_results.append({
        'test_start': str(features.index[start].date()),
        'best_thresh_vix_low': best_thresh['vix_low'],
        'best_thresh_vix_high': best_thresh['vix_high'],
        'train_sharpe': round(best_sharpe, 3),
    })

    start += TEST_DAYS

print(f"  Completed {len(window_results)} windows")
print(f"  Regime distribution: {regime_counts}")

# --- Analysis ---
strat = pd.Series(oot_returns_strategy, index=pd.DatetimeIndex(oot_dates))
bench = pd.Series(oot_returns_benchmark, index=pd.DatetimeIndex(oot_dates))
spy_s = pd.Series(oot_returns_spy, index=pd.DatetimeIndex(oot_dates))

strat = strat.groupby(strat.index).mean().clip(-0.5, 0.5)
bench = bench.groupby(bench.index).mean().clip(-0.5, 0.5)
spy_s = spy_s.groupby(spy_s.index).mean().clip(-0.5, 0.5)

n_days = len(strat)

# Regime gap analysis
spy_full = data['SPY'].reindex(strat.index)
if len(spy_full) > 0 and 'Open' in spy_full.columns:
    green = spy_full['Close'] > spy_full['Open']
    red = ~green
    g_rets = strat[green.reindex(strat.index, fill_value=False)]
    r_rets = strat[red.reindex(strat.index, fill_value=False)]
    sg = float(g_rets.mean() / g_rets.std() * np.sqrt(252)) if len(g_rets) > 10 and g_rets.std() > 0 else 0
    sr = float(r_rets.mean() / r_rets.std() * np.sqrt(252)) if len(r_rets) > 10 and r_rets.std() > 0 else 0
    regime_gap = abs(sg - sr) / max(abs(sg), abs(sr), 0.001)
else:
    sg = sr = regime_gap = 0

def calc_metrics(series, label):
    if len(series) == 0 or series.std() == 0:
        return {}
    ar = float(series.mean() * 252)
    av = float(series.std() * np.sqrt(252))
    sharpe = ar / av
    ds = series[series < 0]
    dv = float(ds.std() * np.sqrt(252)) if len(ds) > 0 else av
    sortino = ar / dv if dv > 0 else 0
    cum = (1 + series).cumprod()
    tr = float(cum.iloc[-1])
    yrs = len(series) / 252
    cagr = (tr ** (1/yrs) - 1) if yrs > 0 and tr > 0 else 0
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    mdd = max(mdd, -1.0)
    return {
        f'{label}_ann_ret_pct': round(ar * 100, 2),
        f'{label}_ann_vol_pct': round(av * 100, 2),
        f'{label}_sharpe': round(sharpe, 3),
        f'{label}_sortino': round(sortino, 3),
        f'{label}_cagr_pct': round(cagr * 100, 2),
        f'{label}_max_dd_pct': round(mdd * 100, 2),
    }

strat_metrics = calc_metrics(strat, 'strategy')
bench_metrics = calc_metrics(bench, 'benchmark_60_40')
spy_metrics = calc_metrics(spy_s, 'spy_buyhold')

summary = {
    'strategy': 'Lane 3: Vol Regime Switching',
    'oot_days': n_days,
    'oot_years': round(n_days/252, 1),
    'regime_distribution': regime_counts,
    **strat_metrics,
    **bench_metrics,
    **spy_metrics,
    'regime_sharpe_green': round(sg, 3),
    'regime_sharpe_red': round(sr, 3),
    'regime_gap': round(regime_gap, 3),
    'regime_test_pass': regime_gap < 0.50,
    'walk_forward_windows': len(window_results),
    'value_added_vs_60_40_sharpe': round(
        strat_metrics.get('strategy_sharpe', 0) - bench_metrics.get('benchmark_60_40_sharpe', 0), 3
    ),
    'survivorship_bias': 'LOW - using broad ETFs, not individual stocks',
    'data_notes': 'TQQQ/UPRO used where available, synthetic 3x leverage for earlier period',
}

print("\n" + "="*60)
print("LANE 3: VOL REGIME SWITCHING - RESULTS")
print("="*60)
for k, v in summary.items():
    print(f"  {k}: {v}")

with open(os.path.join(OUTPUT_DIR, 'lane3_vol_regime_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/lane3_vol_regime_results.json")
