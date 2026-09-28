"""
Approach 2: Concentrated Conviction Bets
- Rank ETFs by multi-factor score (momentum, relative strength, vol breakout)
- Hold top 3 predicted performers
- Walk-forward: 252d train, 21d test, sliding
"""
import numpy as np
import pandas as pd
import yfinance as yf
import os, json, warnings
warnings.filterwarnings('ignore')

OUT = '/home/jupiter/Lvl3Quant/output/growth_research_r8'
os.makedirs(OUT, exist_ok=True)

# Universe: sector + thematic ETFs (avoid single stocks for survivorship)
universe = [
    'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE',  # sectors
    'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ', 'GDX', 'XBI', 'SMH', 'ARKK', 'KWEB',  # thematic
    'SPY', 'TLT', 'GLD', 'DBA'  # broad + diversifiers
]

print("Downloading data...")
data = yf.download(universe + ['^VIX'], start='2012-01-01', end='2026-07-01', auto_adjust=True, progress=False)
close = data['Close'].ffill()
volume = data['Volume'].ffill()

# Drop tickers with insufficient data
min_days = 252 * 3
valid = close.columns[close.count() >= min_days]
close = close[valid]
print(f"Valid tickers: {len(valid)} - {list(valid)}")

# Build multi-factor ranking features per ETF
def compute_factors(prices, vol_data=None):
    """Compute factor scores for each ETF at each date"""
    factors = {}

    for col in prices.columns:
        if col == '^VIX':
            continue
        s = prices[col].dropna()
        f = pd.DataFrame(index=s.index)

        # Momentum factors
        f['mom_1m'] = s.pct_change(21)
        f['mom_3m'] = s.pct_change(63)
        f['mom_6m'] = s.pct_change(126)
        f['mom_12m'] = s.pct_change(252)

        # Momentum excluding last month (classic Jegadeesh-Titman)
        f['mom_12_1'] = s.pct_change(252) - s.pct_change(21)

        # Relative strength (vs rolling mean)
        f['rs_50'] = s / s.rolling(50).mean() - 1
        f['rs_200'] = s / s.rolling(200).mean() - 1

        # Volatility breakout (current vol vs historical)
        ret = s.pct_change()
        f['vol_ratio'] = ret.rolling(21).std() / ret.rolling(63).std()
        f['vol_21'] = ret.rolling(21).std() * np.sqrt(252)

        # Mean reversion signal
        f['zscore_50'] = (s - s.rolling(50).mean()) / s.rolling(50).std()

        # Trend strength
        f['trend'] = (s.rolling(50).mean() - s.rolling(200).mean()) / s.rolling(200).mean()

        factors[col] = f

    return factors

print("Computing factors...")
factors = compute_factors(close)

# Walk-forward: rank ETFs by predicted forward return, hold top 3
train_window = 252
test_window = 21

# Get common dates
all_dates = close.index
tickers = [t for t in close.columns if t != '^VIX']

# Build panel
portfolio_returns = []
spy_benchmark_returns = []
equal_weight_returns = []

for start_idx in range(train_window + 252, len(all_dates) - test_window, test_window):
    test_start = start_idx
    test_end = min(start_idx + test_window, len(all_dates))
    train_end = start_idx
    train_start = start_idx - train_window

    # For each ticker, compute factor scores and forward returns in training period
    train_X = []
    train_y = []
    train_tickers = []

    for t in tickers:
        if t not in factors:
            continue
        f = factors[t]
        # Align with dates
        f_train = f.loc[all_dates[train_start]:all_dates[train_end-1]].dropna()

        if len(f_train) < 100:
            continue

        # Forward 21d return as target
        fwd_ret = close[t].pct_change(21).shift(-21)
        fwd_train = fwd_ret.loc[f_train.index].dropna()
        f_train = f_train.loc[fwd_train.index]

        if len(f_train) < 50:
            continue

        train_X.append(f_train)
        train_y.append(fwd_train)
        train_tickers.extend([t] * len(f_train))

    if len(train_X) == 0:
        continue

    train_X_all = pd.concat(train_X)
    train_y_all = pd.concat(train_y)

    # Simple approach: compute composite score as weighted average of factors
    # Use rank-based scoring (more robust than regression)

    # Score at test time: for each ticker, get latest factor values
    scores = {}
    for t in tickers:
        if t not in factors:
            continue
        f = factors[t]
        test_date = all_dates[test_start]
        # Get closest available date
        available = f.loc[:test_date].dropna()
        if len(available) < 1:
            continue
        latest = available.iloc[-1]

        # Composite score: momentum + trend (higher = more bullish)
        score = (
            0.2 * latest.get('mom_1m', 0) +
            0.2 * latest.get('mom_3m', 0) +
            0.15 * latest.get('mom_12_1', 0) +
            0.15 * latest.get('rs_200', 0) +
            0.15 * latest.get('trend', 0) +
            0.15 * latest.get('rs_50', 0)
        )

        # Penalize very high volatility (risk adjustment)
        vol = latest.get('vol_21', 0.2)
        if vol > 0:
            score = score / vol  # vol-adjusted momentum

        scores[t] = score

    if len(scores) < 5:
        continue

    # Pick top 3
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top3 = [r[0] for r in ranked[:3]]

    # Compute returns for test period
    test_dates_slice = all_dates[test_start:test_end]

    for dt in test_dates_slice:
        # Top 3 equal weighted
        rets = []
        for t in top3:
            r = close[t].pct_change().get(dt, np.nan)
            if pd.notna(r):
                rets.append(r)
        if rets:
            portfolio_returns.append({'date': dt, 'return': np.mean(rets), 'holdings': top3})

        # SPY benchmark
        spy_r = close['SPY'].pct_change().get(dt, np.nan) if 'SPY' in close.columns else np.nan
        if pd.notna(spy_r):
            spy_benchmark_returns.append({'date': dt, 'return': spy_r})

        # Equal weight all
        all_rets = []
        for t in tickers:
            r = close[t].pct_change().get(dt, np.nan)
            if pd.notna(r):
                all_rets.append(r)
        if all_rets:
            equal_weight_returns.append({'date': dt, 'return': np.mean(all_rets)})

port_df = pd.DataFrame(portfolio_returns)
spy_df = pd.DataFrame(spy_benchmark_returns)
ew_df = pd.DataFrame(equal_weight_returns)

def calc_metrics(returns_series, name):
    returns = returns_series.dropna()
    ann_ret = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0
    cum = (1 + returns).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    years = len(returns) / 252
    cagr = (cum.iloc[-1]) ** (1/years) - 1 if years > 0 else 0

    print(f"\n{name}:")
    print(f"  CAGR: {cagr*100:.1f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Sortino: {sortino:.2f}")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Ann Vol: {ann_vol*100:.1f}%")

    return {'name': name, 'cagr': round(cagr*100,1), 'sharpe': round(sharpe,2),
            'sortino': round(sortino,2), 'max_dd': round(max_dd*100,1)}

results = []
if len(port_df) > 100:
    port_series = port_df.set_index('date')['return']
    results.append(calc_metrics(port_series, "Top-3 Concentrated"))

    # Regime diagnostic
    spy_full = close['SPY'].pct_change() if 'SPY' in close.columns else None
    if spy_full is not None:
        # Green days: SPY > 0 rolling 21d
        spy_21 = close['SPY'].pct_change(21)
        bull_dates = spy_21[spy_21 > 0.02].index
        bear_dates = spy_21[spy_21 < -0.02].index

        bull_rets = port_series[port_series.index.isin(bull_dates)].dropna()
        bear_rets = port_series[port_series.index.isin(bear_dates)].dropna()

        if len(bull_rets) > 20 and len(bear_rets) > 20:
            bull_sharpe = (bull_rets.mean() * 252) / (bull_rets.std() * np.sqrt(252))
            bear_sharpe = (bear_rets.mean() * 252) / (bear_rets.std() * np.sqrt(252))
            gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe)) if max(abs(bull_sharpe), abs(bear_sharpe)) > 0 else 0
            print(f"\n  Regime Gap: {gap:.2f} (Bull Sharpe: {bull_sharpe:.2f}, Bear Sharpe: {bear_sharpe:.2f})")
            results[0]['regime_gap'] = round(gap, 2)

if len(spy_df) > 100:
    results.append(calc_metrics(spy_df.set_index('date')['return'], "SPY Benchmark"))

if len(ew_df) > 100:
    results.append(calc_metrics(ew_df.set_index('date')['return'], "Equal Weight All"))

# Permutation test: random top-3 selection
print("\nPermutation test (100 shuffles)...")
perm_cagrs = []
for _ in range(100):
    perm_rets = []
    for start_idx in range(train_window + 252, len(all_dates) - test_window, test_window):
        test_start = start_idx
        test_end = min(start_idx + test_window, len(all_dates))
        test_dates_slice = all_dates[test_start:test_end]

        # Random top 3
        avail = [t for t in tickers if t in close.columns]
        if len(avail) < 3:
            continue
        random_top3 = list(np.random.choice(avail, 3, replace=False))

        for dt in test_dates_slice:
            rets = []
            for t in random_top3:
                r = close[t].pct_change().get(dt, np.nan)
                if pd.notna(r):
                    rets.append(r)
            if rets:
                perm_rets.append(np.mean(rets))

    if perm_rets:
        perm_cum = np.cumprod(1 + np.array(perm_rets))
        years = len(perm_rets) / 252
        perm_cagr = perm_cum[-1] ** (1/years) - 1
        perm_cagrs.append(perm_cagr * 100)

if results and perm_cagrs:
    actual_cagr = results[0]['cagr']
    perm_p = np.mean([c >= actual_cagr for c in perm_cagrs])
    print(f"  Actual CAGR: {actual_cagr:.1f}%")
    print(f"  Permutation median: {np.median(perm_cagrs):.1f}%")
    print(f"  p-value: {perm_p:.3f}")
    results[0]['perm_p_value'] = round(perm_p, 3)
    results[0]['perm_median_cagr'] = round(np.median(perm_cagrs), 1)

with open(f'{OUT}/approach2_concentrated.json', 'w') as f:
    json.dump({'results': results}, f, indent=2)

print(f"\nSaved to {OUT}/approach2_concentrated.json")
print("\n=== APPROACH 2 COMPLETE ===")
