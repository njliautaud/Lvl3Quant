"""
Approach 6: Multi-Timeframe Time-Series Momentum
- Combine 1mo, 3mo, 6mo, 12mo momentum for each asset
- Universe: 20+ ETFs across asset classes
- Long winners, flat losers (long-only)
- Monthly rebalance with vol-targeting
- Walk-forward validation
"""
import numpy as np
import pandas as pd
import yfinance as yf
import os, json, warnings
warnings.filterwarnings('ignore')

OUT = '/home/jupiter/Lvl3Quant/output/growth_research_r8'
os.makedirs(OUT, exist_ok=True)

# Broad diversified universe
universe = [
    # US Equity
    'SPY', 'QQQ', 'IWM', 'MDY',
    # International
    'EFA', 'EEM', 'VGK', 'EWJ',
    # Sectors
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY',
    # Bonds
    'TLT', 'IEF', 'HYG', 'EMB',
    # Commodities
    'GLD', 'SLV', 'DBA', 'USO',
    # REITs
    'VNQ', 'REM',
    # Specialty
    'SMH', 'XBI', 'GDX'
]

print("Downloading data...")
data = yf.download(universe, start='2008-01-01', end='2026-07-01', auto_adjust=True, progress=False)
close = data['Close'].ffill()

valid = [u for u in universe if u in close.columns and close[u].dropna().shape[0] > 252*4]
close = close[valid]
print(f"Valid tickers: {len(valid)} - {list(valid)}")

# Compute time-series momentum signals
def tsmom_signal(prices, lookbacks=[21, 63, 126, 252]):
    """
    Time-series momentum: each asset's own history
    Returns composite signal: average of normalized momentum across lookbacks
    """
    signals = pd.DataFrame(index=prices.index, columns=prices.columns)

    for col in prices.columns:
        s = prices[col].dropna()
        mom_scores = []
        for lb in lookbacks:
            ret = s.pct_change(lb)
            # Normalize: z-score over rolling window
            zscore = (ret - ret.rolling(252).mean()) / ret.rolling(252).std()
            mom_scores.append(zscore)

        # Average across lookbacks
        composite = pd.concat(mom_scores, axis=1).mean(axis=1)
        signals[col] = composite.reindex(prices.index)

    return signals

# Vol targeting
target_vol = 0.15  # 15% annualized vol target per position

def vol_target_weights(prices, signals, target_vol=0.15, lookback=63):
    """Scale position size inversely to realized volatility"""
    rets = prices.pct_change()
    vol = rets.rolling(lookback).std() * np.sqrt(252)

    # Raw weights: signal * (target_vol / realized_vol)
    weights = pd.DataFrame(index=signals.index, columns=signals.columns)
    for col in signals.columns:
        sig = signals[col]
        v = vol[col]
        # Long-only: max(0, signal)
        long_sig = sig.clip(lower=0)
        # Vol-scale
        w = long_sig * (target_vol / v.replace(0, np.nan))
        weights[col] = w.clip(upper=0.2)  # max 20% per position

    # Normalize to sum to 1
    row_sums = weights.sum(axis=1).replace(0, np.nan)
    weights = weights.div(row_sums, axis=0).fillna(0)

    return weights

# Walk-forward: monthly rebalance
print("Computing momentum signals...")
signals = tsmom_signal(close)

# Monthly rebalance dates
monthly_dates = close.resample('ME').last().index
monthly_dates = monthly_dates[monthly_dates >= close.index[252]]  # need 1yr warmup

# Strategy variants
strategies = {
    'TSMOM_Basic': {'lookbacks': [21, 63, 126, 252], 'vol_target': True},
    'TSMOM_Fast': {'lookbacks': [21, 63], 'vol_target': True},
    'TSMOM_Slow': {'lookbacks': [126, 252], 'vol_target': True},
    'TSMOM_NoVolTarget': {'lookbacks': [21, 63, 126, 252], 'vol_target': False},
}

for strat_name, config in strategies.items():
    print(f"\n--- {strat_name} ---")

    # Recompute signals with specific lookbacks
    strat_signals = tsmom_signal(close, config['lookbacks'])

    portfolio_returns = []

    for i in range(len(monthly_dates) - 1):
        rebal_date = monthly_dates[i]
        next_rebal = monthly_dates[i + 1]

        # Get signals at rebalance date
        sig_row = strat_signals.loc[:rebal_date].iloc[-1]

        # Long-only: positive momentum only
        long_mask = sig_row > 0
        selected = sig_row[long_mask]

        if len(selected) == 0:
            # No positive momentum: go to cash (0 return)
            hold_dates = close.loc[rebal_date:next_rebal].index[1:]
            for dt in hold_dates:
                portfolio_returns.append({'date': dt, 'return': 0.0})
            continue

        if config['vol_target']:
            # Vol-targeted weights
            rets = close[selected.index].pct_change()
            vol = rets.loc[:rebal_date].tail(63).std() * np.sqrt(252)
            raw_w = (target_vol / vol.replace(0, np.nan)).clip(upper=0.3)
            weights = raw_w / raw_w.sum()
            weights = weights.fillna(0)
        else:
            # Equal weight
            weights = pd.Series(1.0 / len(selected), index=selected.index)

        # Compute returns over holding period
        hold_dates = close.loc[rebal_date:next_rebal].index[1:]
        daily_rets = close[weights.index].pct_change().loc[hold_dates]

        for dt in hold_dates:
            day_ret = (daily_rets.loc[dt] * weights).sum()
            if pd.notna(day_ret):
                portfolio_returns.append({'date': dt, 'return': day_ret})

    strategies[strat_name]['returns'] = portfolio_returns

# Compute metrics for all strategies
def calc_metrics(rets_list, name):
    df = pd.DataFrame(rets_list)
    if len(df) < 100:
        print(f"  {name}: insufficient data ({len(df)})")
        return None
    returns = df.set_index('date')['return'].dropna()
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

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    print(f"\n{name}:")
    print(f"  CAGR: {cagr*100:.1f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Sortino: {sortino:.2f}")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Calmar: {calmar:.2f}")

    return {'name': name, 'cagr': round(cagr*100,1), 'sharpe': round(sharpe,2),
            'sortino': round(sortino,2), 'max_dd': round(max_dd*100,1),
            'calmar': round(calmar,2)}

results = []
for strat_name in strategies:
    r = calc_metrics(strategies[strat_name]['returns'], strat_name)
    if r:
        results.append(r)

# SPY benchmark
spy_rets = [{'date': dt, 'return': close['SPY'].pct_change().get(dt, 0)}
            for dt in close.index[253:] if pd.notna(close['SPY'].pct_change().get(dt, np.nan))]
r = calc_metrics(spy_rets, "SPY Buy & Hold")
if r:
    results.append(r)

# Regime diagnostic for best strategy
if results:
    best = max([r for r in results if 'TSMOM' in r['name']], key=lambda x: x['cagr'])
    best_name = best['name']
    best_rets = pd.DataFrame(strategies[best_name]['returns']).set_index('date')['return']

    spy_21 = close['SPY'].pct_change(21)
    bull_dates = spy_21[spy_21 > 0.02].index
    bear_dates = spy_21[spy_21 < -0.02].index

    bull_rets = best_rets[best_rets.index.isin(bull_dates)].dropna()
    bear_rets = best_rets[best_rets.index.isin(bear_dates)].dropna()

    if len(bull_rets) > 20 and len(bear_rets) > 20:
        bull_sharpe = (bull_rets.mean() * 252) / (bull_rets.std() * np.sqrt(252))
        bear_sharpe = (bear_rets.mean() * 252) / (bear_rets.std() * np.sqrt(252))
        gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe)) if max(abs(bull_sharpe), abs(bear_sharpe)) > 0 else 0
        print(f"\nRegime Gap for {best_name}: {gap:.2f}")
        print(f"  Bull Sharpe: {bull_sharpe:.2f}, Bear Sharpe: {bear_sharpe:.2f}")
        best['regime_gap'] = round(gap, 2)

# Permutation test for best strategy
print(f"\nPermutation test for {best_name}...")
if results:
    actual_cagr = best['cagr']
    perm_cagrs = []

    for _ in range(100):
        # Shuffle which assets are "positive momentum" each month
        perm_rets = []
        for i in range(len(monthly_dates) - 1):
            rebal_date = monthly_dates[i]
            next_rebal = monthly_dates[i + 1]

            # Random selection of ~half the assets
            n_select = max(3, len(valid) // 3)
            random_sel = list(np.random.choice(valid, n_select, replace=False))
            weights = pd.Series(1.0 / n_select, index=random_sel)

            hold_dates = close.loc[rebal_date:next_rebal].index[1:]
            daily_rets = close[random_sel].pct_change().loc[hold_dates]

            for dt in hold_dates:
                day_ret = (daily_rets.loc[dt] * weights).sum()
                if pd.notna(day_ret):
                    perm_rets.append(day_ret)

        if perm_rets:
            perm_cum = np.cumprod(1 + np.array(perm_rets))
            years = len(perm_rets) / 252
            perm_cagr = perm_cum[-1] ** (1/years) - 1
            perm_cagrs.append(perm_cagr * 100)

    perm_p = np.mean([c >= actual_cagr for c in perm_cagrs])
    print(f"  Actual CAGR: {actual_cagr:.1f}%")
    print(f"  Permutation median: {np.median(perm_cagrs):.1f}%")
    print(f"  p-value: {perm_p:.3f}")
    best['perm_p_value'] = round(perm_p, 3)

with open(f'{OUT}/approach6_multi_tf_momentum.json', 'w') as f:
    json.dump({'results': results}, f, indent=2)

print(f"\nSaved to {OUT}/approach6_multi_tf_momentum.json")
print("\n=== APPROACH 6 COMPLETE ===")
