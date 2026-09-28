#!/usr/bin/env python3
"""
12-Strategy Portfolio Optimizer v4
===================================
Combines ALL 12 validated strategies.

HC #713: Fixed capital, no DCA.
HC #714: Income + growth dual focus.
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/portfolio_optimizer_v4')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("12-STRATEGY PORTFOLIO OPTIMIZATION v4")
print("=" * 70)

# ─── Strategy parameters (from validated backtests) ───
strategies = {
    'CTA_Trend':      {'sharpe': 2.90, 'vol': 0.068, 'type': 'growth',  'spy_corr': 0.25},
    'Sector_Rot':     {'sharpe': 1.99, 'vol': 0.084, 'type': 'growth',  'spy_corr': 0.45},
    'Stat_Arb':       {'sharpe': 0.81, 'vol': 0.100, 'type': 'income',  'spy_corr': 0.04},
    'Credit_Timing':  {'sharpe': 1.19, 'vol': 0.067, 'type': 'income',  'spy_corr': 0.30},
    'Vol_Breakout':   {'sharpe': 1.11, 'vol': 0.138, 'type': 'growth',  'spy_corr': 0.15},
    'Commodity_Trend':{'sharpe': 2.28, 'vol': 0.241, 'type': 'growth',  'spy_corr': 0.24},
    'Carry_Momentum': {'sharpe': 2.96, 'vol': 0.150, 'type': 'growth',  'spy_corr': 0.24},
    'Tail_Risk':      {'sharpe': 4.12, 'vol': 0.128, 'type': 'hedge',   'spy_corr': 0.20},
    'Currency_Carry':  {'sharpe': 1.99, 'vol': 0.050, 'type': 'income', 'spy_corr': 0.10},
    'Bond_Duration':  {'sharpe': 2.00, 'vol': 0.118, 'type': 'income',  'spy_corr': 0.22},
    'Gold_Silver':    {'sharpe': 1.28, 'vol': 0.195, 'type': 'income',  'spy_corr': 0.12},
    'Yield_Curve':    {'sharpe': 2.01, 'vol': 0.158, 'type': 'income',  'spy_corr': -0.16},
}

names = list(strategies.keys())
n = len(names)

# ─── Cross-correlation matrix ───
# Based on strategy types, asset classes, and empirical relationships
corr_matrix = np.eye(n)
# Encoding: (i, j) -> correlation
# Using 0-indexed: CTA=0, Sector=1, StatArb=2, Credit=3, VolBreak=4,
# Commodity=5, Carry=6, TailRisk=7, CurrCarry=8, BondDur=9
corr_values = {
    # CTA correlations (idx 0)
    (0,1): 0.39,  (0,2): 0.00,  (0,3): 0.03,  (0,4): 0.07,
    (0,5): 0.15,  (0,6): 0.25,  (0,7): 0.10,  (0,8): 0.05,
    (0,9): -0.05, (0,10): 0.08, (0,11): -0.05,
    # Sector correlations (idx 1)
    (1,2): 0.00,  (1,3): 0.01,  (1,4): 0.20,  (1,5): 0.10,
    (1,6): 0.35,  (1,7): 0.15,  (1,8): 0.05,  (1,9): -0.10,
    (1,10): 0.05, (1,11): -0.15,
    # StatArb (idx 2, market-neutral)
    (2,3): 0.05,  (2,4): 0.03,  (2,5): 0.02,  (2,6): 0.05,
    (2,7): -0.05, (2,8): 0.02,  (2,9): 0.03,  (2,10): 0.02, (2,11): 0.00,
    # Credit (idx 3)
    (3,4): 0.08,  (3,5): 0.10,  (3,6): 0.15,  (3,7): 0.10,
    (3,8): 0.08,  (3,9): 0.25,  (3,10): 0.05, (3,11): 0.20,
    # VolBreakout (idx 4)
    (4,5): 0.05,  (4,6): 0.12,  (4,7): 0.08,  (4,8): 0.03,
    (4,9): 0.00,  (4,10): 0.03, (4,11): 0.00,
    # Commodity (idx 5)
    (5,6): 0.10,  (5,7): 0.05,  (5,8): 0.15,  (5,9): 0.05,
    (5,10): 0.30, (5,11): 0.05, # Gold/Silver correlated to commodities
    # Carry (idx 6)
    (6,7): 0.12,  (6,8): 0.08,  (6,9): -0.05,
    (6,10): 0.08, (6,11): -0.08,
    # TailRisk (idx 7)
    (7,8): 0.02,  (7,9): 0.10,  (7,10): 0.05, (7,11): 0.08,
    # CurrCarry (idx 8)
    (8,9): 0.15,  (8,10): 0.10, (8,11): 0.05,
    # BondDur (idx 9)
    (9,10): 0.05, (9,11): 0.35, # Both bond-related strategies
    # Gold/Silver (idx 10)
    (10,11): 0.05,
}

for (i,j), c in corr_values.items():
    corr_matrix[i,j] = c
    corr_matrix[j,i] = c

# ─── Build covariance matrix and synthetic returns ───
vols = np.array([strategies[s]['vol'] for s in names])
cov_matrix = np.outer(vols, vols) * corr_matrix

# Ensure positive semi-definite
eigvals = np.linalg.eigvalsh(cov_matrix)
if np.min(eigvals) < 0:
    cov_matrix += np.eye(n) * (abs(np.min(eigvals)) + 1e-6)

np.random.seed(42)
n_days = 5000
daily_means = np.array([strategies[s]['sharpe'] * strategies[s]['vol'] / 252 for s in names])
L = np.linalg.cholesky(cov_matrix)
Z = np.random.randn(n_days, n) @ (L.T / np.sqrt(252)) + daily_means
returns = pd.DataFrame(Z, columns=names)

print(f"\nSynthetic returns: {n_days} days, {n} strategies")
print(f"\nRealized metrics:")
for s in names:
    r = returns[s]
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    cum = (1 + r).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    stype = strategies[s]['type']
    print(f"  {s:18s} [{stype:6s}]: Sharpe {sharpe:.2f}, Vol {ann_vol:.1%}, MaxDD {dd:.1f}%")

print(f"\nCorrelation matrix:")
print(returns.corr().round(2).to_string())

# ─── Portfolio optimization ───
def portfolio_metrics(weights, returns_df):
    port_ret = returns_df.values @ weights
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = np.cumprod(1 + port_ret)
    peak = np.maximum.accumulate(cum)
    dd = ((cum - peak) / peak).min()
    n_years = len(port_ret) / 252
    cagr = (cum[-1] ** (1/n_years) - 1) * 100 if cum[-1] > 0 else 0
    down = port_ret[port_ret < 0]
    down_vol = down.std() * np.sqrt(252) if len(down) > 1 else ann_vol
    sortino = ann_ret / down_vol if down_vol > 0 else 0
    calmar = cagr / abs(dd * 100) if dd != 0 else 0
    return {
        'sharpe': round(sharpe, 2), 'sortino': round(sortino, 2),
        'cagr': round(cagr, 1), 'max_dd': round(dd * 100, 1),
        'calmar': round(calmar, 2), 'vol': round(ann_vol * 100, 1)
    }

def neg_sharpe(w, returns_df):
    port_ret = returns_df.values @ w
    return -(port_ret.mean() * 252) / (port_ret.std() * np.sqrt(252))

def neg_calmar(w, returns_df):
    port_ret = returns_df.values @ w
    cum = np.cumprod(1 + port_ret)
    peak = np.maximum.accumulate(cum)
    dd = ((cum - peak) / peak).min()
    n_years = len(port_ret) / 252
    cagr = cum[-1] ** (1/n_years) - 1 if cum[-1] > 0 else 0
    return -(cagr / abs(dd)) if dd != 0 else 0

constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
bounds = [(0.02, 0.40) for _ in range(n)]  # min 2%, max 40% each
w0 = np.ones(n) / n

print("\n" + "=" * 70)
print("OPTIMIZATION RESULTS")
print("=" * 70)

results = {}

# Method 1: Max Sharpe
res = minimize(neg_sharpe, w0, args=(returns,), method='SLSQP',
               bounds=bounds, constraints=constraints)
w_sharpe = res.x
m_sharpe = portfolio_metrics(w_sharpe, returns)
results['max_sharpe'] = {'weights': {names[i]: round(w_sharpe[i]*100, 1) for i in range(n)}, **m_sharpe}

print(f"\n1. MAX SHARPE PORTFOLIO:")
print(f"   Sharpe {m_sharpe['sharpe']}, Sortino {m_sharpe['sortino']}, CAGR {m_sharpe['cagr']}%, MaxDD {m_sharpe['max_dd']}%, Calmar {m_sharpe['calmar']}")
print(f"   Weights: ", {names[i]: f"{w_sharpe[i]*100:.0f}%" for i in range(n) if w_sharpe[i] > 0.025})

# Method 2: Max Calmar
res2 = minimize(neg_calmar, w0, args=(returns,), method='SLSQP',
                bounds=bounds, constraints=constraints)
w_calmar = res2.x
m_calmar = portfolio_metrics(w_calmar, returns)
results['max_calmar'] = {'weights': {names[i]: round(w_calmar[i]*100, 1) for i in range(n)}, **m_calmar}

print(f"\n2. MAX CALMAR PORTFOLIO:")
print(f"   Sharpe {m_calmar['sharpe']}, Sortino {m_calmar['sortino']}, CAGR {m_calmar['cagr']}%, MaxDD {m_calmar['max_dd']}%, Calmar {m_calmar['calmar']}")
print(f"   Weights: ", {names[i]: f"{w_calmar[i]*100:.0f}%" for i in range(n) if w_calmar[i] > 0.025})

# Method 3: Risk Parity (inverse vol)
inv_vols = 1.0 / vols
w_rp = inv_vols / inv_vols.sum()
m_rp = portfolio_metrics(w_rp, returns)
results['risk_parity'] = {'weights': {names[i]: round(w_rp[i]*100, 1) for i in range(n)}, **m_rp}

print(f"\n3. RISK PARITY PORTFOLIO:")
print(f"   Sharpe {m_rp['sharpe']}, Sortino {m_rp['sortino']}, CAGR {m_rp['cagr']}%, MaxDD {m_rp['max_dd']}%, Calmar {m_rp['calmar']}")
print(f"   Weights: ", {names[i]: f"{w_rp[i]*100:.0f}%" for i in range(n) if w_rp[i] > 0.025})

# Method 4: Equal Weight
w_ew = np.ones(n) / n
m_ew = portfolio_metrics(w_ew, returns)
results['equal_weight'] = {'weights': {names[i]: round(w_ew[i]*100, 1) for i in range(n)}, **m_ew}

print(f"\n4. EQUAL WEIGHT PORTFOLIO:")
print(f"   Sharpe {m_ew['sharpe']}, Sortino {m_ew['sortino']}, CAGR {m_ew['cagr']}%, MaxDD {m_ew['max_dd']}%, Calmar {m_ew['calmar']}")

# Method 5: Income-Heavy (min 50% income/hedge strategies)
def neg_sharpe_income(w, returns_df):
    # Same as neg_sharpe but income constraint in optimizer
    port_ret = returns_df.values @ w
    return -(port_ret.mean() * 252) / (port_ret.std() * np.sqrt(252))

income_indices = [i for i, s in enumerate(names) if strategies[s]['type'] in ('income', 'hedge')]
income_constraint = {'type': 'ineq', 'fun': lambda w: sum(w[i] for i in income_indices) - 0.50}
res5 = minimize(neg_sharpe_income, w0, args=(returns,), method='SLSQP',
                bounds=bounds, constraints=[constraints[0], income_constraint])
w_inc = res5.x
m_inc = portfolio_metrics(w_inc, returns)
results['income_heavy'] = {'weights': {names[i]: round(w_inc[i]*100, 1) for i in range(n)}, **m_inc}

print(f"\n5. INCOME-HEAVY PORTFOLIO (≥50% income/hedge):")
income_pct = sum(w_inc[i] for i in income_indices) * 100
print(f"   Income/Hedge: {income_pct:.0f}%, Growth: {100-income_pct:.0f}%")
print(f"   Sharpe {m_inc['sharpe']}, Sortino {m_inc['sortino']}, CAGR {m_inc['cagr']}%, MaxDD {m_inc['max_dd']}%, Calmar {m_inc['calmar']}")
print(f"   Weights: ", {names[i]: f"{w_inc[i]*100:.0f}%" for i in range(n) if w_inc[i] > 0.025})

# ─── SPY comparison ───
spy_sharpe = 0.89
spy_maxdd = -33.7
spy_cagr = 14.5

print(f"\n{'='*70}")
print(f"COMPARISON vs SPY (Sharpe {spy_sharpe}, MaxDD {spy_maxdd}%, CAGR {spy_cagr}%)")
print(f"{'='*70}")
for method, m in [('Max Sharpe', m_sharpe), ('Max Calmar', m_calmar),
                   ('Risk Parity', m_rp), ('Equal Weight', m_ew), ('Income Heavy', m_inc)]:
    sr_mult = m['sharpe'] / spy_sharpe
    dd_mult = m['max_dd'] / spy_maxdd if spy_maxdd != 0 else 0
    print(f"  {method:15s}: Sharpe {sr_mult:.1f}x SPY, MaxDD {abs(dd_mult):.0f}% of SPY DD")

# ─── Income/Growth Split ───
print(f"\n{'='*70}")
print("INCOME vs GROWTH ALLOCATION")
print(f"{'='*70}")
for method, w in [('Max Sharpe', w_sharpe), ('Max Calmar', w_calmar),
                   ('Risk Parity', w_rp), ('Income Heavy', w_inc)]:
    growth_pct = sum(w[i] for i in range(n) if strategies[names[i]]['type'] == 'growth') * 100
    income_pct = sum(w[i] for i in range(n) if strategies[names[i]]['type'] == 'income') * 100
    hedge_pct = sum(w[i] for i in range(n) if strategies[names[i]]['type'] == 'hedge') * 100
    print(f"  {method:15s}: Growth {growth_pct:.0f}% | Income {income_pct:.0f}% | Hedge {hedge_pct:.0f}%")

# ─── Save ───
output = {
    'strategy': '10-Strategy Portfolio Optimizer v3',
    'n_strategies': n,
    'strategy_list': {s: {
        'sharpe': strategies[s]['sharpe'],
        'type': strategies[s]['type'],
        'spy_corr': strategies[s]['spy_corr'],
    } for s in names},
    'results': results,
    'spy_benchmark': {'sharpe': spy_sharpe, 'max_dd': spy_maxdd, 'cagr': spy_cagr},
    'timestamp': str(pd.Timestamp.now()),
}

with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved. DONE.")
