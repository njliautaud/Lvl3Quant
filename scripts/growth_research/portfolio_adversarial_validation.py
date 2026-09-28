#!/usr/bin/env python3
"""
Adversarial Validation of Combined Portfolio (HC #709 R2)
=========================================================
Tests whether the growth_tilt portfolio result is robust:
1. Sub-period stability (rolling 1-year Sharpe, 3-year sub-periods)
2. Outlier removal (best/worst N days removed)
3. Transaction cost sensitivity
4. Protection signal lag sensitivity (what if signals are 1-5 days late?)
5. Alternative proxy test (different ETFs for the same strategies)
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json, os, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/combined_portfolio'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ADVERSARIAL VALIDATION — Combined Portfolio (HC #709 R2)")
print("=" * 70)

# ── DATA (reuse same setup as portfolio backtest) ──
print("\n[1/6] Loading data...")
tickers = {
    'SPY': 'SPY', 'USMV': 'USMV', 'RSP': 'RSP', 'SVXY': 'SVXY',
    'UPRO': 'UPRO', 'QQQ': 'QQQ', 'VIX': '^VIX',
    'HYG': 'HYG', 'LQD': 'LQD', 'SHV': 'SHV',
    'XLK': 'XLK', 'XLF': 'XLF', 'XLE': 'XLE', 'XLV': 'XLV',
    'XLI': 'XLI', 'XLP': 'XLP', 'XLU': 'XLU', 'XLB': 'XLB',
    'XLRE': 'XLRE', 'XLY': 'XLY', 'XLC': 'XLC',
    # Alternative proxies
    'SPLV': 'SPLV',   # Alt income proxy (low vol)
    'VTV': 'VTV',     # Alt rotation proxy (value)
    'SPHD': 'SPHD',   # Alt income (high div low vol)
}

data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start='2012-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 100:
            data[name] = df['Close']
    except:
        pass

prices = pd.DataFrame(data).ffill()
returns = prices.pct_change()
print(f"  {len(prices)} days, {len(prices.columns)} assets")

# ── BUILD STRATEGY RETURNS (same as backtest) ──
def build_portfolio(prices, returns, weights, protection_lag=0):
    """Build portfolio returns with optional protection signal lag."""
    sectors = [c for c in ['XLK','XLF','XLE','XLV','XLI','XLP','XLU','XLB','XLRE','XLY','XLC'] if c in prices.columns]
    
    # Protection signals
    prot = pd.DataFrame(index=prices.index)
    if 'VIX' in prices.columns:
        prot['vix_ok'] = (prices['VIX'] < 20).astype(int)
    if 'SPY' in prices.columns:
        prot['spy_sma_ok'] = (prices['SPY'] > prices['SPY'].rolling(50).mean()).astype(int)
    if 'HYG' in returns.columns and 'LQD' in returns.columns:
        spread = (returns['HYG'] - returns['LQD']).rolling(20).mean()
        z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        prot['credit_ok'] = (z > -1.0).astype(int)
    if len(sectors) >= 5:
        above_50 = pd.DataFrame({s: (prices[s] > prices[s].rolling(50).mean()).astype(int) for s in sectors})
        prot['breadth_ok'] = (above_50.mean(axis=1) > 0.5).astype(int)
    
    sig_cols = [c for c in prot.columns]
    prot['majority_clear'] = (prot[sig_cols].sum(axis=1) >= 3).astype(int)
    
    # Apply lag to protection signals
    if protection_lag > 0:
        prot['majority_clear'] = prot['majority_clear'].shift(protection_lag)
    
    idx = prices.dropna().index
    idx = idx[idx >= '2013-01-01']
    
    strat_ret = pd.DataFrame(index=idx)
    
    # Income strategies
    income_proxy = weights.get('income_proxy', 'USMV')
    rotation_proxy = weights.get('rotation_proxy', 'RSP')
    
    if income_proxy in returns.columns:
        strat_ret['income_csp'] = returns[income_proxy].reindex(idx).fillna(0)
    if rotation_proxy in returns.columns:
        strat_ret['income_rotation'] = returns[rotation_proxy].reindex(idx).fillna(0)
    if 'SVXY' in returns.columns and 'VIX' in prices.columns:
        vix = prices['VIX'].reindex(idx)
        svxy_scale = pd.Series(0.5, index=idx)
        svxy_scale[vix.between(18, 28)] = 1.0
        svxy_scale[vix > 35] = 0.0
        strat_ret['income_strangle'] = returns['SVXY'].reindex(idx).fillna(0) * svxy_scale
    
    # Growth
    if 'UPRO' in returns.columns and 'VIX' in prices.columns:
        vix = prices['VIX'].reindex(idx)
        p = prot.reindex(idx)
        upro_alloc = pd.Series(0.0, index=idx)
        upro_alloc[vix < 17] = 1.0
        upro_alloc[vix.between(17, 25)] = 0.3
        upro_alloc[vix >= 25] = 0.0
        upro_alloc[p['majority_clear'] == 0] *= 0.5
        strat_ret['growth_upro'] = returns['UPRO'].reindex(idx).fillna(0) * upro_alloc
    
    if 'QQQ' in returns.columns:
        qqq_above_200 = (prices['QQQ'] > prices['QQQ'].rolling(200).mean()).reindex(idx).astype(int)
        strat_ret['growth_megacap'] = returns['QQQ'].reindex(idx).fillna(0) * qqq_above_200
    
    if 'SHV' in returns.columns:
        strat_ret['cash'] = returns['SHV'].reindex(idx).fillna(0)
    
    # Build portfolio
    alloc = {
        'income_csp': weights.get('income_csp', 0.05),
        'income_rotation': weights.get('income_rotation', 0.15),
        'income_strangle': weights.get('income_strangle', 0.15),
        'growth_upro': weights.get('growth_upro', 0.25),
        'growth_megacap': weights.get('growth_megacap', 0.35),
        'cash': weights.get('cash', 0.05),
    }
    
    port_ret = pd.Series(0.0, index=idx)
    for strat, w in alloc.items():
        if strat in strat_ret.columns and w > 0:
            port_ret += strat_ret[strat] * w
    
    return port_ret, idx

def compute_metrics(ret):
    """Compute standard metrics for a return series."""
    if len(ret) < 20 or ret.std() == 0:
        return {'sharpe': 0, 'sortino': 0, 'cagr': 0, 'maxdd': 0, 'wr': 0, 'pf': 0}
    cum = (1 + ret).cumprod()
    n_years = len(ret) / 252
    sharpe = ret.mean() / ret.std() * np.sqrt(252)
    neg = ret[ret < 0]
    sortino = ret.mean() / neg.std() * np.sqrt(252) if len(neg) > 5 and neg.std() > 0 else 0
    cagr = (cum.iloc[-1] ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    dd = cum / cum.cummax() - 1
    maxdd = dd.min() * 100
    wr = (ret > 0).mean() * 100
    gains = ret[ret > 0].sum()
    losses = abs(ret[ret < 0].sum())
    pf = gains / losses if losses > 0 else 0
    return {
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr, 1), 'maxdd': round(maxdd, 1),
        'wr': round(wr, 1), 'pf': round(pf, 3),
    }

# Growth-tilt weights (validated config)
base_weights = {
    'income_csp': 0.05, 'income_rotation': 0.15, 'income_strangle': 0.15,
    'growth_upro': 0.25, 'growth_megacap': 0.35, 'cash': 0.05,
}

port_ret, idx = build_portfolio(prices, returns, base_weights)
baseline = compute_metrics(port_ret)
print(f"\n  Baseline: Sharpe {baseline['sharpe']}, CAGR {baseline['cagr']}%, MaxDD {baseline['maxdd']}%")

# ── TEST 1: SUB-PERIOD STABILITY ──
print("\n" + "="*70)
print("[2/6] Sub-period stability...")

# 3-year sub-periods
sub_periods = []
years = sorted(set(idx.year))
for i in range(0, len(years)-2, 3):
    start_y, end_y = years[i], years[min(i+2, len(years)-1)]
    mask = (idx.year >= start_y) & (idx.year <= end_y)
    sub_ret = port_ret[mask]
    spy_ret = returns['SPY'].reindex(idx)[mask].fillna(0)
    if len(sub_ret) > 100:
        m = compute_metrics(sub_ret)
        sm = compute_metrics(spy_ret)
        sub_periods.append({
            'period': f"{start_y}-{end_y}",
            'sharpe': m['sharpe'], 'cagr': m['cagr'], 'maxdd': m['maxdd'],
            'spy_sharpe': sm['sharpe'], 'beats_spy': m['sharpe'] > sm['sharpe'],
        })
        beats = "✅" if m['sharpe'] > sm['sharpe'] else "❌"
        print(f"  {start_y}-{end_y}: Sharpe {m['sharpe']:.3f} (SPY {sm['sharpe']:.3f}) {beats} | CAGR {m['cagr']:.1f}% | MaxDD {m['maxdd']:.1f}%")

# Rolling 1-year Sharpe
rolling_sharpe = port_ret.rolling(252).apply(lambda x: x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0, raw=True)
min_rolling = rolling_sharpe.min()
max_rolling = rolling_sharpe.max()
pct_positive = (rolling_sharpe > 0).mean() * 100
pct_above_1 = (rolling_sharpe > 1).mean() * 100
print(f"\n  Rolling 1yr Sharpe: min {min_rolling:.2f}, max {max_rolling:.2f}")
print(f"  Positive {pct_positive:.0f}% of the time | Above 1.0: {pct_above_1:.0f}%")

# ── TEST 2: OUTLIER REMOVAL ──
print("\n" + "="*70)
print("[3/6] Outlier removal...")

for n_remove in [3, 5, 10, 20]:
    # Remove best N days
    sorted_ret = port_ret.sort_values(ascending=False)
    without_best = port_ret.drop(sorted_ret.index[:n_remove])
    m_best = compute_metrics(without_best)
    
    # Remove worst N days
    without_worst = port_ret.drop(sorted_ret.index[-n_remove:])
    m_worst = compute_metrics(without_worst)
    
    # Remove both best and worst
    without_both = port_ret.drop(sorted_ret.index[:n_remove]).drop(sorted_ret.index[-n_remove:], errors='ignore')
    m_both = compute_metrics(without_both)
    
    print(f"\n  Remove {n_remove} days:")
    print(f"    Remove best:  Sharpe {m_best['sharpe']:.3f} (Δ{m_best['sharpe']-baseline['sharpe']:+.3f})")
    print(f"    Remove worst: Sharpe {m_worst['sharpe']:.3f} (Δ{m_worst['sharpe']-baseline['sharpe']:+.3f})")
    print(f"    Remove both:  Sharpe {m_both['sharpe']:.3f} (Δ{m_both['sharpe']-baseline['sharpe']:+.3f})")

# ── TEST 3: TRANSACTION COST SENSITIVITY ──
print("\n" + "="*70)
print("[4/6] Transaction cost sensitivity...")

for annual_cost_bps in [5, 10, 20, 50, 100, 200]:
    daily_cost = annual_cost_bps / 10000 / 252
    adj_ret = port_ret - daily_cost
    m = compute_metrics(adj_ret)
    print(f"  {annual_cost_bps:>3}bps/yr: Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1f}% | MaxDD {m['maxdd']:.1f}%")

# ── TEST 4: PROTECTION SIGNAL LAG ──
print("\n" + "="*70)
print("[5/6] Protection signal lag sensitivity...")

for lag in [0, 1, 2, 3, 5, 10]:
    lag_ret, _ = build_portfolio(prices, returns, base_weights, protection_lag=lag)
    m = compute_metrics(lag_ret)
    delta = m['sharpe'] - baseline['sharpe']
    print(f"  {lag:>2}d lag: Sharpe {m['sharpe']:.3f} (Δ{delta:+.3f}) | MaxDD {m['maxdd']:.1f}%")

# ── TEST 5: ALTERNATIVE PROXIES ──
print("\n" + "="*70)
print("[6/6] Alternative proxy sensitivity...")

alt_configs = [
    ('SPLV as income', {'income_proxy': 'SPLV'}),
    ('VTV as rotation', {'rotation_proxy': 'VTV'}),
    ('SPHD as income', {'income_proxy': 'SPHD'}),
]

for name, overrides in alt_configs:
    w = {**base_weights, **overrides}
    try:
        alt_ret, _ = build_portfolio(prices, returns, w)
        m = compute_metrics(alt_ret)
        delta = m['sharpe'] - baseline['sharpe']
        print(f"  {name}: Sharpe {m['sharpe']:.3f} (Δ{delta:+.3f}) | CAGR {m['cagr']:.1f}% | MaxDD {m['maxdd']:.1f}%")
    except Exception as e:
        print(f"  {name}: ERROR — {e}")

# ── VERDICT ──
print("\n" + "="*70)
n_sub_beating = sum(1 for s in sub_periods if s['beats_spy'])
n_sub_total = len(sub_periods)

# Key checks
checks = {
    'sub_period_stability': n_sub_beating >= n_sub_total * 0.6,
    'rolling_mostly_positive': pct_positive > 70,
    'survives_outlier_removal': True,  # will check below
    'cost_robust': True,  # will check
}

# Check outlier removal (Sharpe still > 1.0 after removing 10 best days)
sorted_ret = port_ret.sort_values(ascending=False)
without_10_best = port_ret.drop(sorted_ret.index[:10])
m10 = compute_metrics(without_10_best)
checks['survives_outlier_removal'] = m10['sharpe'] > 1.0

# Check cost robustness (still beats SPY at 100bps)
adj_100 = port_ret - 100/10000/252
m100 = compute_metrics(adj_100)
spy_m = compute_metrics(returns['SPY'].reindex(idx).fillna(0))
checks['cost_robust'] = m100['sharpe'] > spy_m['sharpe']

n_pass = sum(checks.values())
n_total = len(checks)

verdict = f"{'ROBUST' if n_pass == n_total else 'PARTIAL'} — {n_pass}/{n_total} adversarial checks pass"
print(f"VERDICT: {verdict}")
print(f"  Sub-period ({n_sub_beating}/{n_sub_total} beat SPY): {'✅' if checks['sub_period_stability'] else '❌'}")
print(f"  Rolling Sharpe positive {pct_positive:.0f}%: {'✅' if checks['rolling_mostly_positive'] else '❌'}")
print(f"  Survives 10-day outlier removal (Sharpe {m10['sharpe']:.3f}): {'✅' if checks['survives_outlier_removal'] else '❌'}")
print(f"  Cost robust at 100bps (Sharpe {m100['sharpe']:.3f} vs SPY {spy_m['sharpe']:.3f}): {'✅' if checks['cost_robust'] else '❌'}")
print("="*70)

# Save
output = {
    'timestamp': datetime.now().isoformat(),
    'baseline': baseline,
    'sub_periods': sub_periods,
    'rolling_sharpe': {
        'min': round(float(min_rolling), 3),
        'max': round(float(max_rolling), 3),
        'pct_positive': round(float(pct_positive), 1),
        'pct_above_1': round(float(pct_above_1), 1),
    },
    'checks': {k: bool(v) for k, v in checks.items()},
    'verdict': verdict,
}
with open(os.path.join(OUTPUT_DIR, 'adversarial_results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"Saved to {OUTPUT_DIR}/adversarial_results.json")
