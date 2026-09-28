"""
Approach 5: Options Leverage on Predictable Mean-Reversion Events
- After sharp drawdowns (>10% sector), buy the dip with simulated leverage
- After sharp run-ups (>20% in 2mo), reduce/hedge
- Walk-forward: does mean-reversion signal predict post-event returns?

Note: We simulate options payoff profiles (can't get historical options data easily)
- Deep ITM calls ~ 2-3x leverage with limited downside
- We model this as: max(0, leverage * return - premium_cost)
"""
import numpy as np
import pandas as pd
import yfinance as yf
import os, json, warnings
warnings.filterwarnings('ignore')

OUT = '/home/jupiter/Lvl3Quant/output/growth_research_r8'
os.makedirs(OUT, exist_ok=True)

# Universe of ETFs for mean-reversion
etfs = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY',
        'XLP', 'XLB', 'XLRE', 'EFA', 'EEM', 'VNQ', 'SMH', 'XBI', 'GDX']

print("Downloading data...")
data = yf.download(etfs, start='2010-01-01', end='2026-07-01', auto_adjust=True, progress=False)
close = data['Close'].ffill()

valid_etfs = [e for e in etfs if e in close.columns and close[e].dropna().shape[0] > 252*3]
print(f"Valid ETFs: {len(valid_etfs)}")

# Identify mean-reversion events
def find_drawdown_events(prices, threshold=-0.10, lookback=42):
    """Find dates where an ETF has drawn down > threshold over lookback days"""
    ret = prices.pct_change(lookback)
    events = ret[ret < threshold].index
    return events

def find_runup_events(prices, threshold=0.20, lookback=42):
    """Find dates where an ETF has run up > threshold over lookback days"""
    ret = prices.pct_change(lookback)
    events = ret[ret > threshold].index
    return events

# Walk-forward analysis
# For each event, measure forward returns at 5d, 10d, 21d, 42d
horizons = [5, 10, 21, 42]
dd_threshold = -0.10
runup_threshold = 0.20
lookback = 42

print("\nAnalyzing drawdown mean-reversion events...")
all_dd_events = []

for etf in valid_etfs:
    prices = close[etf].dropna()
    ret_lookback = prices.pct_change(lookback)

    for dt in prices.index[lookback + 252:]:  # skip first year for walk-forward context
        drawdown = ret_lookback.get(dt, 0)
        if pd.isna(drawdown):
            continue

        if drawdown < dd_threshold:
            # Mean-reversion buy signal
            fwd_rets = {}
            for h in horizons:
                idx = prices.index.get_loc(dt)
                if idx + h < len(prices):
                    fwd_rets[f'fwd_{h}d'] = prices.iloc[idx + h] / prices.iloc[idx] - 1
                else:
                    fwd_rets[f'fwd_{h}d'] = np.nan

            # Walk-forward feature: was the historical mean-reversion profitable?
            # Use last 252 days of similar events
            hist_start = max(0, prices.index.get_loc(dt) - 252)
            hist_prices = prices.iloc[hist_start:prices.index.get_loc(dt)]
            hist_ret = hist_prices.pct_change(lookback)
            hist_events = hist_ret[hist_ret < dd_threshold]

            hist_success_rate = 0
            if len(hist_events) > 3:
                successes = 0
                for hdt in hist_events.index:
                    hidx = prices.index.get_loc(hdt)
                    if hidx + 21 < len(prices):
                        if prices.iloc[hidx + 21] > prices.iloc[hidx]:
                            successes += 1
                hist_success_rate = successes / len(hist_events)

            all_dd_events.append({
                'date': dt, 'etf': etf, 'drawdown': drawdown,
                'hist_success_rate': hist_success_rate,
                **fwd_rets
            })

dd_df = pd.DataFrame(all_dd_events)
print(f"Total drawdown events: {len(dd_df)}")

if len(dd_df) > 50:
    # Filter for walk-forward: only trade when hist_success_rate > 0.5
    traded = dd_df[dd_df['hist_success_rate'] > 0.5]
    untraded = dd_df[dd_df['hist_success_rate'] <= 0.5]

    print(f"  Events with positive historical edge: {len(traded)}")
    print(f"  Events without edge: {len(untraded)}")

    for h in horizons:
        col = f'fwd_{h}d'
        if col in traded.columns:
            traded_mean = traded[col].dropna().mean() * 100
            all_mean = dd_df[col].dropna().mean() * 100
            wr = (traded[col].dropna() > 0).mean() * 100
            print(f"  {h}d fwd: filtered={traded_mean:.2f}%, all={all_mean:.2f}%, WR={wr:.1f}%")

# Simulate portfolio: enter when drawdown event + positive historical edge
# Use 2x leverage (simulating deep ITM calls)
# Exit after 21 days
# Max 5 concurrent positions

print("\n--- Simulating leveraged mean-reversion portfolio ---")
leverage = 2.0
max_positions = 5
hold_days = 21
premium_cost = 0.02  # 2% options premium (approximate for deep ITM)

# Sort events by date
if len(dd_df) > 0:
    traded_events = dd_df[dd_df['hist_success_rate'] > 0.5].sort_values('date')

    positions = []  # list of (entry_date, etf, entry_price)
    daily_returns = {}
    cash = 1.0

    for _, event in traded_events.iterrows():
        dt = event['date']
        etf = event['etf']

        # Check if we have capacity and not already holding this ETF
        active = [p for p in positions if (dt - p[0]).days <= hold_days]
        positions = active

        if len(active) >= max_positions:
            continue
        if any(p[2] == etf for p in active):
            continue

        # Enter position
        entry_price = close[etf].get(dt, np.nan)
        if pd.notna(entry_price):
            positions.append((dt, entry_price, etf))

            # Track forward returns with leverage
            idx = close.index.get_loc(dt)
            for d in range(1, hold_days + 1):
                if idx + d < len(close):
                    exit_dt = close.index[idx + d]
                    exit_price = close[etf].iloc[idx + d]
                    raw_ret = exit_price / entry_price - 1
                    # Leveraged return minus premium
                    lev_ret = leverage * raw_ret - premium_cost / hold_days

                    if exit_dt not in daily_returns:
                        daily_returns[exit_dt] = []
                    daily_returns[exit_dt].append(lev_ret / hold_days)  # daily portion

    # Aggregate daily returns
    port_daily = {}
    for dt in sorted(daily_returns.keys()):
        rets = daily_returns[dt]
        # Average across active positions, then scale by allocation
        port_daily[dt] = np.mean(rets) * (min(len(rets), max_positions) / max_positions)

    port_series = pd.Series(port_daily).sort_index()

    # Also compute SPY benchmark over same period
    spy_series = close['SPY'].pct_change().reindex(port_series.index).dropna()
    port_series = port_series.reindex(spy_series.index).fillna(0)

def calc_metrics(returns, name):
    returns = returns.dropna()
    if len(returns) < 100:
        print(f"  {name}: insufficient data")
        return None
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

    return {'name': name, 'cagr': round(cagr*100,1), 'sharpe': round(sharpe,2),
            'sortino': round(sortino,2), 'max_dd': round(max_dd*100,1)}

results = []
if len(dd_df) > 50 and len(port_series) > 100:
    r = calc_metrics(port_series, "Leveraged Mean-Reversion")
    if r:
        results.append(r)

    r = calc_metrics(spy_series, "SPY Benchmark")
    if r:
        results.append(r)

    # Permutation: trade at random times instead of drawdown events
    print("\nPermutation test...")
    perm_cagrs = []
    for _ in range(100):
        # Random entry dates
        random_dates = np.random.choice(close.index[252:-42], size=min(len(traded_events), 500), replace=False)
        random_rets = {}
        for dt in random_dates:
            etf = np.random.choice(valid_etfs)
            idx = close.index.get_loc(dt)
            for d in range(1, hold_days + 1):
                if idx + d < len(close):
                    exit_dt = close.index[idx + d]
                    raw_ret = close[etf].iloc[idx + d] / close[etf].iloc[idx] - 1
                    lev_ret = leverage * raw_ret - premium_cost / hold_days
                    if exit_dt not in random_rets:
                        random_rets[exit_dt] = []
                    random_rets[exit_dt].append(lev_ret / hold_days)

        perm_daily = {dt: np.mean(rets) for dt, rets in random_rets.items()}
        if len(perm_daily) > 100:
            ps = pd.Series(perm_daily).sort_index()
            perm_cum = (1 + ps).cumprod()
            years = len(ps) / 252
            perm_cagr = perm_cum.iloc[-1] ** (1/years) - 1
            perm_cagrs.append(perm_cagr * 100)

    if perm_cagrs and results:
        actual_cagr = results[0]['cagr']
        perm_p = np.mean([c >= actual_cagr for c in perm_cagrs])
        print(f"  Actual CAGR: {actual_cagr:.1f}%")
        print(f"  Permutation median: {np.median(perm_cagrs):.1f}%")
        print(f"  p-value: {perm_p:.3f}")
        results[0]['perm_p_value'] = round(perm_p, 3)

# Raw mean-reversion stats
mr_stats = {}
if len(dd_df) > 0:
    for h in horizons:
        col = f'fwd_{h}d'
        vals = dd_df[col].dropna()
        mr_stats[f'{h}d_mean_ret'] = round(vals.mean() * 100, 2)
        mr_stats[f'{h}d_win_rate'] = round((vals > 0).mean() * 100, 1)
        mr_stats[f'{h}d_count'] = int(len(vals))

with open(f'{OUT}/approach5_options_meanrev.json', 'w') as f:
    json.dump({'results': results, 'mean_reversion_stats': mr_stats,
               'n_events': len(dd_df)}, f, indent=2)

print(f"\nSaved to {OUT}/approach5_options_meanrev.json")
print("\n=== APPROACH 5 COMPLETE ===")
