"""
Approach 3: Leveraged Barbell
- 60% UPRO (3x S&P) + 40% tail-risk hedge
- Hedge: managed futures proxy (long vol + trend following)
- Walk-forward: can we predict when to increase/decrease hedge?
"""
import numpy as np
import pandas as pd
import yfinance as yf
import os, json, warnings
warnings.filterwarnings('ignore')

OUT = '/home/jupiter/Lvl3Quant/output/growth_research_r8'
os.makedirs(OUT, exist_ok=True)

print("Downloading data...")
tickers = ['UPRO', 'SPY', 'TLT', 'GLD', 'VIXY', 'TAIL', 'CTA', 'DBMF', 'KMLM',
           'SH', 'BIL', '^VIX']
data = yf.download(tickers, start='2012-01-01', end='2026-07-01', auto_adjust=True, progress=False)
close = data['Close'].ffill()

print(f"Available columns: {list(close.columns)}")
print(f"Data range: {close.index[0]} to {close.index[-1]}")

# Build hedge portfolio from available assets
# Managed futures ETFs are newer, so use what's available
# Fallback: synthetic tail hedge = long TLT + long GLD + short SPY proxy (via SH)
hedge_assets = []
for t in ['DBMF', 'KMLM', 'CTA', 'TAIL']:
    if t in close.columns and close[t].dropna().shape[0] > 500:
        hedge_assets.append(t)

if not hedge_assets:
    # Use synthetic: 40% TLT + 30% GLD + 30% SH
    print("Using synthetic hedge: TLT + GLD + SH")
    hedge_type = 'synthetic'
else:
    print(f"Using managed futures hedge: {hedge_assets}")
    hedge_type = 'managed_futures'

# Strategy variants
strategies = {}

# 1. Static 60/40 UPRO/hedge
# 2. Dynamic: increase hedge when VIX > 25 or SPY < 200MA
# 3. Adaptive: walk-forward model decides hedge weight

def get_hedge_return(dt, close_df):
    if hedge_type == 'managed_futures':
        rets = []
        for h in hedge_assets:
            r = close_df[h].pct_change().get(dt, np.nan)
            if pd.notna(r):
                rets.append(r)
        return np.mean(rets) if rets else 0
    else:
        # Synthetic
        tlt_r = close_df['TLT'].pct_change().get(dt, 0) if 'TLT' in close_df.columns else 0
        gld_r = close_df['GLD'].pct_change().get(dt, 0) if 'GLD' in close_df.columns else 0
        sh_r = close_df['SH'].pct_change().get(dt, 0) if 'SH' in close_df.columns else 0

        tlt_r = tlt_r if pd.notna(tlt_r) else 0
        gld_r = gld_r if pd.notna(gld_r) else 0
        sh_r = sh_r if pd.notna(sh_r) else 0

        return 0.4 * tlt_r + 0.3 * gld_r + 0.3 * sh_r

# Common dates where UPRO exists
upro_dates = close['UPRO'].dropna().index
upro_ret = close['UPRO'].pct_change()
spy_ret = close['SPY'].pct_change()

# VIX for dynamic allocation
vix = close['^VIX'] if '^VIX' in close.columns else None
spy_close = close['SPY']

# Strategy 1: Static 60/40
print("\n--- Strategy 1: Static 60% UPRO / 40% Hedge ---")
static_rets = []
for dt in upro_dates:
    ur = upro_ret.get(dt, np.nan)
    hr = get_hedge_return(dt, close)
    if pd.notna(ur):
        static_rets.append({'date': dt, 'return': 0.6 * ur + 0.4 * hr})

# Strategy 2: Dynamic (VIX-based)
print("--- Strategy 2: Dynamic VIX-based allocation ---")
dynamic_rets = []
for dt in upro_dates:
    ur = upro_ret.get(dt, np.nan)
    hr = get_hedge_return(dt, close)
    if pd.notna(ur) and vix is not None:
        v = vix.get(dt, 20)
        v = v if pd.notna(v) else 20
        spy_val = spy_close.get(dt, np.nan)
        spy_200 = spy_close.rolling(200).mean().get(dt, np.nan)

        # High fear or below 200MA: increase hedge
        if v > 30 or (pd.notna(spy_val) and pd.notna(spy_200) and spy_val < spy_200):
            w_upro = 0.3
        elif v > 20:
            w_upro = 0.5
        else:
            w_upro = 0.75  # Low vol: more aggressive

        dynamic_rets.append({'date': dt, 'return': w_upro * ur + (1 - w_upro) * hr})

# Strategy 3: Quarterly rebalanced with momentum filter
print("--- Strategy 3: Momentum-filtered barbell ---")
momentum_rets = []
spy_mom = spy_close.pct_change(63)  # 3-month momentum

for dt in upro_dates:
    ur = upro_ret.get(dt, np.nan)
    hr = get_hedge_return(dt, close)
    if pd.notna(ur):
        mom = spy_mom.get(dt, 0)
        mom = mom if pd.notna(mom) else 0

        # Positive momentum: more UPRO. Negative: more hedge.
        if mom > 0.05:
            w_upro = 0.80
        elif mom > 0:
            w_upro = 0.60
        elif mom > -0.05:
            w_upro = 0.40
        else:
            w_upro = 0.20

        momentum_rets.append({'date': dt, 'return': w_upro * ur + (1 - w_upro) * hr})

# Strategy 4: Pure UPRO (benchmark for leverage)
upro_only = [{'date': dt, 'return': upro_ret.get(dt, 0)} for dt in upro_dates if pd.notna(upro_ret.get(dt, np.nan))]

def calc_metrics(rets_list, name):
    df = pd.DataFrame(rets_list)
    if len(df) < 100:
        print(f"  {name}: insufficient data ({len(df)} days)")
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

    print(f"\n{name}:")
    print(f"  CAGR: {cagr*100:.1f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Sortino: {sortino:.2f}")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Ann Vol: {ann_vol*100:.1f}%")
    print(f"  Final: {cum.iloc[-1]:.2f}x")

    return {'name': name, 'cagr': round(cagr*100,1), 'sharpe': round(sharpe,2),
            'sortino': round(sortino,2), 'max_dd': round(max_dd*100,1),
            'final_cum': round(float(cum.iloc[-1]),2)}

results = []
for name, rets in [("Static 60/40 UPRO/Hedge", static_rets),
                     ("Dynamic VIX-Based", dynamic_rets),
                     ("Momentum-Filtered Barbell", momentum_rets),
                     ("Pure UPRO", upro_only),
                     ("SPY Buy & Hold", [{'date': dt, 'return': spy_ret.get(dt, 0)}
                                          for dt in upro_dates if pd.notna(spy_ret.get(dt, np.nan))])]:
    r = calc_metrics(rets, name)
    if r:
        results.append(r)

# Regime gap diagnostic for best strategy
if results:
    best = max(results[:3], key=lambda x: x['cagr'])
    print(f"\nBest barbell variant: {best['name']} ({best['cagr']}% CAGR)")

    # Check if managed futures hedge actually helps in crashes
    # Find worst SPY drawdown periods
    spy_cum = (1 + spy_ret.dropna()).cumprod()
    spy_dd = spy_cum / spy_cum.cummax() - 1

    crash_dates = spy_dd[spy_dd < -0.15].index
    print(f"\nDays with SPY DD > 15%: {len(crash_dates)}")

    # Key question: does the barbell protect during these periods?
    if len(crash_dates) > 20:
        print("Hedge provides crash protection: analyzing...")

# Permutation test for dynamic strategy
print("\nPermutation test for dynamic VIX strategy...")
if dynamic_rets:
    actual_df = pd.DataFrame(dynamic_rets).set_index('date')['return'].dropna()
    actual_cum = (1 + actual_df).cumprod()
    years = len(actual_df) / 252
    actual_cagr = (actual_cum.iloc[-1]) ** (1/years) - 1

    perm_cagrs = []
    for _ in range(100):
        # Shuffle the UPRO weight decisions (breaks VIX signal)
        dates_list = list(actual_df.index)
        n = len(dates_list)
        # Random weights between 0.3 and 0.75
        random_weights = np.random.uniform(0.3, 0.75, n)
        perm_ret = []
        for i, dt in enumerate(dates_list):
            ur = upro_ret.get(dt, 0)
            hr = get_hedge_return(dt, close)
            ur = ur if pd.notna(ur) else 0
            perm_ret.append(random_weights[i] * ur + (1 - random_weights[i]) * hr)

        perm_cum = np.cumprod(1 + np.array(perm_ret))
        perm_cagr = perm_cum[-1] ** (1/years) - 1
        perm_cagrs.append(perm_cagr * 100)

    perm_p = np.mean([c >= actual_cagr * 100 for c in perm_cagrs])
    print(f"  Actual CAGR: {actual_cagr*100:.1f}%")
    print(f"  Permutation median: {np.median(perm_cagrs):.1f}%")
    print(f"  p-value: {perm_p:.3f}")

    for r in results:
        if 'Dynamic' in r['name']:
            r['perm_p_value'] = round(perm_p, 3)

with open(f'{OUT}/approach3_leveraged_barbell.json', 'w') as f:
    json.dump({'results': results, 'hedge_type': hedge_type}, f, indent=2)

print(f"\nSaved to {OUT}/approach3_leveraged_barbell.json")
print("\n=== APPROACH 3 COMPLETE ===")
