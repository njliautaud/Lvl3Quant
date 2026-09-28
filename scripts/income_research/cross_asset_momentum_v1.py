#!/usr/bin/env python3
"""
Cross-Asset Momentum / Risk-Parity v1
======================================
HC #705 adversarial checks built-in.

THESIS: Unlike single-asset-class rotation (which failed), cross-ASSET-CLASS
momentum may work because asset classes are driven by different macro factors.
When stocks crash, bonds rally. When inflation spikes, commodities rally.
The DIVERSIFICATION across asset classes is the edge, not stock-picking.

ALSO: Trend-following across asset classes (managed futures proxy).

ETFs:
  - SPY (US Equities)
  - TLT (Long-term US Treasuries)
  - GLD (Gold)
  - DBC or GSG (Broad Commodities)
  - EFA (International Developed)
  - EEM (Emerging Markets)
  - IEF (Intermediate Treasuries)
  - TIP (TIPS — inflation-protected)
  - VNQ (REITs)

Strategies:
  A) Static Risk-Parity (1/vol weighting, monthly rebal)
  B) Trend-following: long assets above 200d MA, cash if below
  C) Cross-asset momentum: buy top 3-4 by trailing 6-12m return
  D) Defensive: SPY + TLT + GLD only, trend-filtered
  E) All-Weather: equal-weight SPY/TLT/GLD/TIP, rebal monthly

PERMUTATION FIX: Use proper random-date entry test, not return shuffling.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import json
from pathlib import Path
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/cross_asset_momentum_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
ASSETS = ['SPY', 'TLT', 'GLD', 'EFA', 'EEM', 'IEF', 'TIP', 'VNQ']
# DBC/GSG have shorter history, skip for now

print("=" * 70)
print("CROSS-ASSET MOMENTUM v1")
print("=" * 70)

cache_path = Path("/home/jupiter/Lvl3Quant/data/cache/cross_asset_etfs.parquet")
if cache_path.exists():
    print("\nLoading cached data...")
    closes = pd.read_parquet(cache_path)[ASSETS]
else:
    print("\nDownloading data...")
    all_closes = {}
    for t in ASSETS:
        d = yf.download(t, start='2008-01-01', end='2026-07-15', progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            all_closes[t] = d['Close'][t]
        else:
            all_closes[t] = d['Close']
    closes = pd.DataFrame(all_closes).ffill().dropna()
    closes.to_parquet(cache_path)

returns = closes.pct_change().dropna()

print(f"Data: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}, {len(closes)} days")

# ─────────────────────────────────────────────
# PROPER PERMUTATION TEST (fixed)
# ─────────────────────────────────────────────
def proper_permutation_test(portfolio_daily_ret, spy_daily_ret, n_perms=200):
    """
    Proper permutation: keep portfolio returns in order, but randomly assign
    invested/cash days. Tests whether the TIMING adds value over random.

    For multi-asset strategies, we compare against randomly timed equal-weight
    portfolio of all assets.
    """
    real_sharpe = portfolio_daily_ret.mean() / portfolio_daily_ret.std() * np.sqrt(252) if portfolio_daily_ret.std() > 0 else 0

    # For comparison: what if we randomly shifted the signal by 1-252 days?
    n = len(portfolio_daily_ret)
    random_sharpes = []
    for _ in range(n_perms):
        shift = np.random.randint(1, min(252, n))
        shifted = np.roll(portfolio_daily_ret.values, shift)
        s = np.mean(shifted) / np.std(shifted) * np.sqrt(252) if np.std(shifted) > 0 else 0
        random_sharpes.append(s)

    p_value = np.mean([s >= real_sharpe for s in random_sharpes])
    return p_value, real_sharpe, np.mean(random_sharpes)


def regime_test(port_ret, spy_ret_aligned):
    """R1: Check regime balance."""
    green = port_ret[spy_ret_aligned > 0]
    red = port_ret[spy_ret_aligned < 0]
    sg = green.mean() / green.std() * np.sqrt(252) if len(green) > 10 and green.std() > 0 else 0
    sr = red.mean() / red.std() * np.sqrt(252) if len(red) > 10 and red.std() > 0 else 0
    gap = abs(sg - sr) / max(abs(sg), abs(sr), 0.001)
    return gap, sg, sr

# ─────────────────────────────────────────────
# STRATEGIES
# ─────────────────────────────────────────────
spy_ret = returns['SPY']

configs = [
    # A) Equal Weight Buy-and-Hold (baseline)
    {'name': 'equal_weight_all', 'type': 'equal_weight', 'assets': ASSETS},
    {'name': 'equal_weight_stt', 'type': 'equal_weight', 'assets': ['SPY', 'TLT', 'GLD']},
    {'name': 'equal_weight_sttv', 'type': 'equal_weight', 'assets': ['SPY', 'TLT', 'GLD', 'VNQ']},

    # B) Trend-Following: long only when above 200d MA
    {'name': 'trend_200d_all', 'type': 'trend', 'assets': ASSETS, 'ma': 200},
    {'name': 'trend_200d_stt', 'type': 'trend', 'assets': ['SPY', 'TLT', 'GLD'], 'ma': 200},
    {'name': 'trend_100d_all', 'type': 'trend', 'assets': ASSETS, 'ma': 100},
    {'name': 'trend_50d_all', 'type': 'trend', 'assets': ASSETS, 'ma': 50},

    # C) Cross-Asset Momentum: top N by trailing return
    {'name': 'xmom_126d_top3', 'type': 'xmom', 'assets': ASSETS, 'lookback': 126, 'top_n': 3, 'hold': 21},
    {'name': 'xmom_252d_top3', 'type': 'xmom', 'assets': ASSETS, 'lookback': 252, 'top_n': 3, 'hold': 21},
    {'name': 'xmom_63d_top3', 'type': 'xmom', 'assets': ASSETS, 'lookback': 63, 'top_n': 3, 'hold': 21},
    {'name': 'xmom_126d_top4', 'type': 'xmom', 'assets': ASSETS, 'lookback': 126, 'top_n': 4, 'hold': 21},

    # D) Dual Momentum (Antonacci): SPY if SPY mom > 0 AND SPY > bonds, else TLT if TLT mom > 0, else cash
    {'name': 'dual_mom_12m', 'type': 'dual_mom', 'lookback': 252},
    {'name': 'dual_mom_6m', 'type': 'dual_mom', 'lookback': 126},

    # E) Inverse-Vol Weighting
    {'name': 'invvol_all', 'type': 'invvol', 'assets': ASSETS, 'vol_lookback': 63},
    {'name': 'invvol_stt', 'type': 'invvol', 'assets': ['SPY', 'TLT', 'GLD'], 'vol_lookback': 63},

    # F) Trend + Invvol: trend-filter + inverse-vol weight
    {'name': 'trend_invvol_all', 'type': 'trend_invvol', 'assets': ASSETS, 'ma': 200, 'vol_lookback': 63},
    {'name': 'trend_invvol_stt', 'type': 'trend_invvol', 'assets': ['SPY', 'TLT', 'GLD'], 'ma': 200, 'vol_lookback': 63},
]

print(f"\nTesting {len(configs)} configurations...")

def run_strategy(config, closes, returns):
    """Run strategy, return daily portfolio returns."""
    assets = config.get('assets', ASSETS)
    start_idx = 260  # enough for 252d lookback

    port_returns = []
    dates = []

    n_returns = len(returns)
    for i in range(start_idx, min(len(closes), n_returns)):
        date = returns.index[i]

        if config['type'] == 'equal_weight':
            day_ret = returns[assets].iloc[i].mean()

        elif config['type'] == 'trend':
            # Long only assets above MA
            ma = config['ma']
            in_trend = []
            for a in assets:
                if closes[a].iloc[i-1] > closes[a].iloc[i-ma:i].mean():
                    in_trend.append(a)
            if in_trend:
                day_ret = returns[in_trend].iloc[i].mean()
            else:
                day_ret = 0.0  # All cash

        elif config['type'] == 'xmom':
            lb = config['lookback']
            hold = config['hold']
            top_n = config['top_n']

            if (i - start_idx) % hold == 0:
                # Rebalance
                trailing = {}
                for a in assets:
                    trailing[a] = closes[a].iloc[i-1] / closes[a].iloc[i-lb-1] - 1
                ranked = sorted(trailing.items(), key=lambda x: x[1], reverse=True)
                config['_holdings'] = [r[0] for r in ranked[:top_n]]

            holdings = config.get('_holdings', assets[:top_n])
            day_ret = returns[holdings].iloc[i].mean()

        elif config['type'] == 'dual_mom':
            lb = config['lookback']
            spy_mom = closes['SPY'].iloc[i-1] / closes['SPY'].iloc[i-lb-1] - 1
            tlt_mom = closes['TLT'].iloc[i-1] / closes['TLT'].iloc[i-lb-1] - 1

            if spy_mom > 0 and spy_mom > tlt_mom:
                day_ret = returns['SPY'].iloc[i]
            elif tlt_mom > 0:
                day_ret = returns['TLT'].iloc[i]
            else:
                day_ret = 0.0  # Cash

        elif config['type'] == 'invvol':
            vl = config['vol_lookback']
            vols = returns[assets].iloc[i-vl:i].std()
            inv_vols = 1.0 / vols.replace(0, np.nan).dropna()
            weights = inv_vols / inv_vols.sum()
            day_ret = (returns[assets].iloc[i] * weights).sum()

        elif config['type'] == 'trend_invvol':
            ma = config['ma']
            vl = config['vol_lookback']
            in_trend = []
            for a in assets:
                if closes[a].iloc[i-1] > closes[a].iloc[i-ma:i].mean():
                    in_trend.append(a)
            if in_trend:
                vols = returns[in_trend].iloc[i-vl:i].std()
                inv_vols = 1.0 / vols.replace(0, np.nan).dropna()
                if len(inv_vols) > 0:
                    weights = inv_vols / inv_vols.sum()
                    day_ret = (returns[in_trend].iloc[i] * weights).sum()
                else:
                    day_ret = 0.0
            else:
                day_ret = 0.0

        else:
            day_ret = 0.0

        port_returns.append(day_ret)
        dates.append(date)

    return pd.Series(port_returns, index=dates)

# Run all
results = []
for idx, config in enumerate(configs):
    port_ret = run_strategy(config, closes, returns)

    if len(port_ret) < 252:
        continue

    # Metrics
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else 0
    sortino_d = port_ret[port_ret < 0].std()
    sortino = port_ret.mean() / sortino_d * np.sqrt(252) if sortino_d > 0 else 0
    total_ret = (1 + port_ret).prod() - 1
    cagr = (1 + total_ret) ** (252 / len(port_ret)) - 1
    wr = (port_ret[port_ret != 0] > 0).mean() if (port_ret != 0).sum() > 0 else 0

    cum = (1 + port_ret).cumprod()
    peak = cum.cummax()
    max_dd = ((cum - peak) / peak).min()

    gains = port_ret[port_ret > 0].sum()
    losses = abs(port_ret[port_ret < 0].sum())
    pf = gains / losses if losses > 0 else 999

    invested = (port_ret != 0).mean() * 100

    # Adversarial
    perm_p, real_sharpe, rand_sharpe = proper_permutation_test(port_ret, spy_ret)
    spy_aligned = spy_ret.reindex(port_ret.index).fillna(0)
    r1_gap, sg, sr = regime_test(port_ret, spy_aligned)

    mid = len(port_ret) // 2
    h1 = port_ret.iloc[:mid]; h2 = port_ret.iloc[mid:]
    h1s = h1.mean()/h1.std()*np.sqrt(252) if h1.std()>0 else 0
    h2s = h2.mean()/h2.std()*np.sqrt(252) if h2.std()>0 else 0

    gates = {
        'perm': perm_p < 0.05,
        'R1': r1_gap < 0.50,
        'sub': h1s > 0 and h2s > 0,
        'sharpe+': sharpe > 0,
    }
    all_pass = all(gates.values())

    result = {
        'name': config['name'],
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 2),
        'wr': round(wr * 100, 1),
        'pf': round(pf, 2),
        'max_dd': round(max_dd * 100, 2),
        'invested': round(invested, 1),
        'perm_p': round(perm_p, 3),
        'rand_sharpe': round(rand_sharpe, 2),
        'r1_gap': round(r1_gap, 3),
        'sg': round(sg, 2), 'sr': round(sr, 2),
        'h1s': round(h1s, 2), 'h2s': round(h2s, 2),
        'all_pass': all_pass,
        'gates': gates,
    }
    results.append(result)

    status = "✅" if all_pass else "❌"
    print(f"  [{idx+1}/{len(configs)}] {config['name']}: Sharpe {sharpe:.2f}, perm={perm_p:.3f}, R1={r1_gap:.3f}, invested={invested:.0f}% {status}")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("RESULTS")
print(f"{'=' * 70}")

rdf = pd.DataFrame(results).sort_values('sharpe', ascending=False)
passing = rdf[rdf['all_pass']]

print(f"\nTotal: {len(rdf)} configs")
print(f"Passing all gates: {len(passing)}")

if len(passing) > 0:
    print(f"\nPASSING:")
    for _, r in passing.iterrows():
        print(f"  {r['name']}: Sharpe {r['sharpe']:.2f}, Sortino {r['sortino']:.2f}, CAGR {r['cagr']:.1f}%")
        print(f"    WR {r['wr']:.0f}%, PF {r['pf']:.2f}, MaxDD {r['max_dd']:.1f}%, Invested {r['invested']:.0f}%")
        print(f"    Perm p={r['perm_p']:.3f}, R1={r['r1_gap']:.3f} (green={r['sg']:.2f}, red={r['sr']:.2f})")
        print(f"    Sub-period: H1={r['h1s']:.2f}, H2={r['h2s']:.2f}")

print(f"\nTop 5:")
for _, r in rdf.head(5).iterrows():
    gate_str = " ".join([f"{'✅' if v else '❌'}{k}" for k, v in r['gates'].items()])
    print(f"  {r['name']}: Sharpe {r['sharpe']:.2f}, perm={r['perm_p']:.3f}, R1={r['r1_gap']:.3f} | {gate_str}")

# SPY benchmark
spy_full = spy_ret.iloc[260:]
spy_s = spy_full.mean()/spy_full.std()*np.sqrt(252) if spy_full.std()>0 else 0
print(f"\nSPY buy-and-hold Sharpe: {spy_s:.2f}")

# Failure modes
print(f"\nFailure modes:")
for g in ['perm', 'R1', 'sub', 'sharpe+']:
    fails = sum(1 for r in results if not r['gates'][g])
    print(f"  {g}: {fails}/{len(results)} fail")

# Save
rdf.to_csv(OUTPUT_DIR / 'results.csv', index=False)
with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")
if len(passing) > 0:
    best = passing.iloc[0]
    print(f"  🔥 {len(passing)} configs pass all gates!")
    print(f"  Best: {best['name']} — Sharpe {best['sharpe']:.2f}, MaxDD {best['max_dd']:.1f}%")
else:
    print(f"  ❌ No configs pass all gates.")
    # Key insight
    print(f"  Note: Permutation test uses time-shift (keeps autocorrelation structure).")
    perm_fails = sum(1 for r in results if not r['gates']['perm'])
    r1_fails = sum(1 for r in results if not r['gates']['R1'])
    print(f"  Permutation: {perm_fails}/{len(results)} fail | R1: {r1_fails}/{len(results)} fail")
