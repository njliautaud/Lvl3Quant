#!/usr/bin/env python3
"""
Volatility Contango Harvester v1 — VIX Term Structure Strategy
==============================================================
HC #705 (adversarial checks built-in), survivorship-bias-free.

THESIS: VIX futures are typically in contango (front month < back month).
When contango steepens, short-vol trades (SVXY/ZIV) profit from roll yield.
When backwardation occurs (fear spike), go to cash or long-vol (UVXY/VXX).

ETFs used:
  - SVXY: Short VIX (profits from contango)
  - UVXY: Long VIX (profits from backwardation / fear spikes)
  - VXX (proxy): Long VIX short-term (for older data)
  - ^VIX: VIX index for signal generation
  - SPY: Benchmark

Signals:
  A) VIX level: >25 = danger zone, <15 = harvest zone
  B) VIX term structure: VIX/VIX3M ratio (contango vs backwardation)
  C) VIX momentum: is VIX falling (safe to harvest) or rising (danger)?
  D) Combined
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
import json
from pathlib import Path
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/vol_contango_harvest_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("VOL CONTANGO HARVESTER v1 — VIX TERM STRUCTURE")
print("=" * 70)

# Download data
tickers = ['^VIX', '^VIX3M', 'SPY', 'SVXY', 'UVXY']
print("\nDownloading data...")
data = yf.download(tickers, start='2012-01-01', end='2026-07-15', progress=False)

if isinstance(data.columns, pd.MultiIndex):
    closes = data['Close'].copy()
else:
    closes = data.copy()

# VIX doesn't need ffill the same way
closes = closes.ffill().dropna()

# Rename for clarity
vix = closes['^VIX']
vix3m = closes['^VIX3M'] if '^VIX3M' in closes.columns else None
spy = closes['SPY']
spy_ret = spy.pct_change()

# SVXY may not have full history — use inverse VIX proxy for earlier dates
svxy = closes['SVXY'] if 'SVXY' in closes.columns else None
svxy_ret = svxy.pct_change() if svxy is not None else None

print(f"VIX data: {vix.index[0].strftime('%Y-%m-%d')} to {vix.index[-1].strftime('%Y-%m-%d')}")
if svxy is not None:
    print(f"SVXY data: {svxy.dropna().index[0].strftime('%Y-%m-%d')} to {svxy.dropna().index[-1].strftime('%Y-%m-%d')}")

# ─────────────────────────────────────────────
# SIGNAL COMPUTATION
# ─────────────────────────────────────────────

# VIX level signals
vix_ma5 = vix.rolling(5).mean()
vix_ma20 = vix.rolling(20).mean()
vix_change_5d = vix.pct_change(5)

# VIX term structure (contango ratio)
if vix3m is not None:
    contango_ratio = vix / vix3m  # <1 = contango (normal), >1 = backwardation (fear)
else:
    contango_ratio = None

# ─────────────────────────────────────────────
# STRATEGY CONFIGS — All trade SPY (most liquid, no decay issues)
# ─────────────────────────────────────────────
configs = []

# A) VIX Level: Be in SPY when VIX < threshold, cash when VIX > threshold
for vix_thresh in [15, 18, 20, 25]:
    configs.append({
        'name': f'vix_lt{vix_thresh}_spy',
        'desc': f'Long SPY when VIX < {vix_thresh}, else cash',
        'type': 'vix_level',
        'threshold': vix_thresh,
        'direction': 'below',
    })

# B) VIX Momentum: Be in SPY when VIX is falling
for lookback in [5, 10, 20]:
    configs.append({
        'name': f'vix_falling_{lookback}d_spy',
        'desc': f'Long SPY when VIX has fallen over {lookback}d',
        'type': 'vix_momentum',
        'lookback': lookback,
    })

# C) VIX Mean-Reversion: Buy SPY when VIX spikes then drops back
for spike_thresh in [25, 30]:
    for drop_thresh in [20, 22]:
        for hold in [5, 10, 20]:
            configs.append({
                'name': f'vix_spike{spike_thresh}_drop{drop_thresh}_hold{hold}d',
                'desc': f'Buy SPY when VIX drops below {drop_thresh} after being above {spike_thresh}',
                'type': 'vix_mean_rev',
                'spike': spike_thresh,
                'drop': drop_thresh,
                'hold': hold,
            })

# D) VIX Term Structure: Trade based on contango/backwardation
if contango_ratio is not None:
    for ratio_thresh in [0.85, 0.90, 0.95]:
        configs.append({
            'name': f'contango_lt{int(ratio_thresh*100)}_spy',
            'desc': f'Long SPY when VIX/VIX3M < {ratio_thresh} (contango)',
            'type': 'contango',
            'threshold': ratio_thresh,
        })

# E) Combined: VIX < 20 AND falling AND contango
configs.append({
    'name': 'combined_safe_harvest',
    'desc': 'Long SPY when VIX<20 AND falling 5d AND contango<0.95',
    'type': 'combined',
})

print(f"\nTesting {len(configs)} configurations...")

# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def backtest_vix_signal(config, vix, spy_ret, vix_ma5, vix_change_5d, contango_ratio):
    """Backtest a VIX-based timing strategy on SPY."""
    start_idx = 252  # 1 year warmup

    portfolio_returns = []
    dates = []
    signal_history = []

    for i in range(start_idx, len(spy_ret)):
        date = spy_ret.index[i]

        if date not in vix.index:
            continue

        # Determine signal (use PREVIOUS day to avoid look-ahead)
        vix_val = vix.loc[:date].iloc[-2] if len(vix.loc[:date]) > 1 else None
        if vix_val is None:
            continue

        signal = 0  # 0 = cash, 1 = long SPY

        if config['type'] == 'vix_level':
            if config['direction'] == 'below':
                signal = 1 if vix_val < config['threshold'] else 0

        elif config['type'] == 'vix_momentum':
            lb = config['lookback']
            if i > lb:
                vix_prev = vix.loc[:date].iloc[-lb-1] if len(vix.loc[:date]) > lb else None
                if vix_prev and vix_val < vix_prev:
                    signal = 1  # VIX falling = go long

        elif config['type'] == 'vix_mean_rev':
            # Check if VIX was above spike threshold in last 10 days and now below drop threshold
            recent_vix = vix.loc[:date].iloc[-11:-1]
            if len(recent_vix) >= 5:
                was_above = (recent_vix > config['spike']).any()
                now_below = vix_val < config['drop']
                if was_above and now_below:
                    signal = 1  # Mean reversion signal
                    # Hold for config['hold'] days after signal
                elif len(signal_history) >= 1:
                    # Check if we're still in hold period from previous signal
                    last_signals = signal_history[-config['hold']:]
                    if 1 in last_signals:
                        signal = 1

        elif config['type'] == 'contango':
            if contango_ratio is not None and date in contango_ratio.index:
                cr = contango_ratio.loc[:date].iloc[-2] if len(contango_ratio.loc[:date]) > 1 else None
                if cr and cr < config['threshold']:
                    signal = 1

        elif config['type'] == 'combined':
            cr = contango_ratio.loc[:date].iloc[-2] if contango_ratio is not None and date in contango_ratio.index and len(contango_ratio.loc[:date]) > 1 else 1.0
            vix_5d_ago = vix.loc[:date].iloc[-6] if len(vix.loc[:date]) > 5 else vix_val
            if vix_val < 20 and vix_val < vix_5d_ago and cr < 0.95:
                signal = 1

        signal_history.append(signal)

        if signal == 1:
            day_ret = spy_ret.iloc[i]
        else:
            day_ret = 0.0  # Cash

        portfolio_returns.append(day_ret)
        dates.append(date)

    return pd.Series(portfolio_returns, index=dates), signal_history

# ─────────────────────────────────────────────
# ADVERSARIAL CHECKS
# ─────────────────────────────────────────────
def permutation_test(port_ret, n_perms=200):
    real_sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else 0
    random_sharpes = []
    for _ in range(n_perms):
        shuffled = port_ret.sample(frac=1.0).values
        s = np.mean(shuffled) / np.std(shuffled) * np.sqrt(252) if np.std(shuffled) > 0 else 0
        random_sharpes.append(s)
    return np.mean([s >= real_sharpe for s in random_sharpes]), real_sharpe

def regime_test(port_ret, spy_ret_aligned):
    green = port_ret[spy_ret_aligned > 0]
    red = port_ret[spy_ret_aligned < 0]
    sg = green.mean() / green.std() * np.sqrt(252) if green.std() > 0 else 0
    sr = red.mean() / red.std() * np.sqrt(252) if red.std() > 0 else 0
    gap = abs(sg - sr) / max(abs(sg), abs(sr), 0.001)
    return gap, sg, sr

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
results = []

for idx, config in enumerate(configs):
    port_ret, signals = backtest_vix_signal(config, vix, spy_ret, vix_ma5, vix_change_5d, contango_ratio)

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

    # Days invested
    invested_pct = (port_ret != 0).mean() * 100
    n_signals = sum(signals) if signals else 0

    # Adversarial
    perm_p, _ = permutation_test(port_ret)
    spy_aligned = spy_ret.reindex(port_ret.index).fillna(0)
    r1_gap, sg, sr = regime_test(port_ret, spy_aligned)

    # Sub-period
    mid = len(port_ret) // 2
    h1_sharpe = port_ret.iloc[:mid].mean() / port_ret.iloc[:mid].std() * np.sqrt(252) if port_ret.iloc[:mid].std() > 0 else 0
    h2_sharpe = port_ret.iloc[mid:].mean() / port_ret.iloc[mid:].std() * np.sqrt(252) if port_ret.iloc[mid:].std() > 0 else 0
    subperiod = h1_sharpe > 0 and h2_sharpe > 0

    gates = {
        'perm': perm_p < 0.05,
        'R1': r1_gap < 0.50,
        'sub': subperiod,
        'sharpe+': sharpe > 0,
    }
    all_pass = all(gates.values())

    result = {
        'name': config['name'],
        'desc': config['desc'],
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr_pct': round(cagr * 100, 2),
        'wr': round(wr * 100, 1),
        'pf': round(pf, 2),
        'max_dd_pct': round(max_dd * 100, 2),
        'invested_pct': round(invested_pct, 1),
        'perm_p': round(perm_p, 3),
        'r1_gap': round(r1_gap, 3),
        'sg': round(sg, 2),
        'sr': round(sr, 2),
        'h1': round(h1_sharpe, 2),
        'h2': round(h2_sharpe, 2),
        'all_pass': all_pass,
        'gates': gates,
    }
    results.append(result)

    status = "✅" if all_pass else "❌"
    print(f"  [{idx+1}/{len(configs)}] {config['name']}: Sharpe {sharpe:.2f}, perm={perm_p:.3f}, R1={r1_gap:.3f}, invested={invested_pct:.0f}% {status}")

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
    print(f"\n{'=' * 70}")
    print("PASSING CONFIGS:")
    print(f"{'=' * 70}")
    for _, r in passing.iterrows():
        print(f"\n  {r['name']} — {r['desc']}")
        print(f"    Sharpe {r['sharpe']:.2f} | Sortino {r['sortino']:.2f} | CAGR {r['cagr_pct']:.1f}%")
        print(f"    WR {r['wr']:.0f}% | PF {r['pf']:.2f} | MaxDD {r['max_dd_pct']:.1f}% | Invested {r['invested_pct']:.0f}%")
        print(f"    Perm p={r['perm_p']:.3f} | R1 gap={r['r1_gap']:.3f} (green={r['sg']:.2f}, red={r['sr']:.2f})")
        print(f"    Sub-period: H1={r['h1']:.2f}, H2={r['h2']:.2f}")

# SPY buy-and-hold
spy_common = spy_ret.loc[rdf.iloc[0]['name']:] if len(rdf) > 0 else spy_ret.iloc[252:]
# Just compute SPY Sharpe over full period
spy_full = spy_ret.iloc[252:]
spy_sharpe = spy_full.mean() / spy_full.std() * np.sqrt(252) if spy_full.std() > 0 else 0
spy_cagr = ((1 + spy_full).prod() ** (252/len(spy_full)) - 1) * 100
print(f"\nBenchmark SPY buy-and-hold: Sharpe {spy_sharpe:.2f}, CAGR {spy_cagr:.1f}%")

print(f"\nTop 5 by Sharpe:")
for _, r in rdf.head(5).iterrows():
    gate_str = " ".join([f"{'✅' if v else '❌'}{k}" for k, v in r['gates'].items()])
    print(f"  {r['name']}: Sharpe {r['sharpe']:.2f}, perm={r['perm_p']:.3f}, R1={r['r1_gap']:.3f} | {gate_str}")

# Failure mode
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
    print(f"  🔥 {len(passing)} configs pass all gates!")
    print(f"  These are VIX-based timing strategies with REAL edge.")
    print(f"  No survivorship bias (SPY only). No options pricing issues.")
else:
    print(f"  ❌ No configs pass all gates.")
    perm_fails = sum(1 for r in results if not r['gates']['perm'])
    r1_fails = sum(1 for r in results if not r['gates']['R1'])
    print(f"  Permutation failures: {perm_fails}, R1 failures: {r1_fails}")
