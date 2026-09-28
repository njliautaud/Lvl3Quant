#!/usr/bin/env python3
"""
Stat Arb Baseline — Fast Permutation Test
==========================================
Tests whether the pure z-score pairs trading edge is REAL.

The ML stat arb showed ML hurts (0.356 vs baseline 0.807). But is the baseline
itself real? Permutation: shuffle which days get entry signals → if random
entries also profit, the edge is market exposure, not mean reversion.

HC #714: Income + growth research
HC #665: Adversarial validation mandatory
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from itertools import combinations
from scipy import stats
import json, os, sys, warnings
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/output/stat_arb_baseline'
os.makedirs(OUTPUT, exist_ok=True)
np.random.seed(42)

# Parameters
INITIAL_CAPITAL = 100_000
COINT_LOOKBACK = 126
ENTRY_Z = 1.5
EXIT_Z = 0.3
STOP_Z = 4.0
MAX_HOLD = 42
MAX_PAIRS = 8
POS_SIZE = 1.0 / MAX_PAIRS
N_PERM = 50  # Faster — 50 is enough for p-value significance

print("=" * 70)
print("STAT ARB BASELINE — FAST PERMUTATION TEST")
print("Pure z-score pairs trading (no ML)")
print("=" * 70)

# Download data
sectors = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU']
cross = ['GLD','GDX','TLT','IEF','HYG','LQD','SPY','QQQ','IWM','EEM','DIA']
tickers = list(set(sectors + cross))
print(f"\nDownloading {len(tickers)} assets...")
df = yf.download(tickers, start='2005-01-01', progress=False)
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)

close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
close = close.ffill()
valid = close.columns[close.notna().sum() > 2000]
close = close[valid].dropna()
print(f"  {len(close)} days, {len(close.columns)} assets, {close.index[0].date()} to {close.index[-1].date()}")


def rolling_cointegration(price_a, price_b, window=COINT_LOOKBACK):
    """Quick cointegration test."""
    if len(price_a) < window:
        return np.nan, np.nan, np.nan
    a = price_a.values[-window:]
    b = price_b.values[-window:]
    a_norm, b_norm = a / a[0], b / b[0]
    X = np.column_stack([b_norm, np.ones(window)])
    try:
        beta, alpha = np.linalg.lstsq(X, a_norm, rcond=None)[0]
    except:
        return np.nan, np.nan, np.nan
    residual = a_norm - beta * b_norm - alpha
    if np.std(residual) < 1e-10:
        return np.nan, np.nan, np.nan
    residual_lag = residual[:-1]
    residual_diff = np.diff(residual)
    if np.std(residual_lag) < 1e-10:
        return np.nan, np.nan, np.nan
    slope = np.polyfit(residual_lag, residual_diff, 1)[0]
    if slope >= 0:
        return np.nan, np.nan, np.nan
    half_life = -np.log(2) / slope
    lags = range(2, min(20, window // 5))
    tau = [np.std(np.subtract(residual[lag:], residual[:-lag])) for lag in lags]
    if len(tau) < 2 or any(t <= 0 for t in tau):
        return np.nan, beta, half_life
    try:
        hurst = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    except:
        hurst = 0.5
    return hurst, beta, half_life


def run_baseline_backtest(close, shuffle_entries=False):
    """
    Pure z-score pairs trading — no ML.
    If shuffle_entries: randomly permute which days generate entry signals.
    """
    n_days = len(close)
    assets = list(close.columns)
    all_pairs = list(combinations(assets, 2))
    daily_returns = np.zeros(n_days)
    positions = {}
    trade_log = []

    for day in range(COINT_LOOKBACK + 63, n_days):
        # Check exits first
        to_close = []
        for pair_key, pos in positions.items():
            a, b = pair_key.split('|')
            if a not in close.columns or b not in close.columns:
                to_close.append(pair_key)
                continue
            pa = close[a].iloc[max(0, day - COINT_LOOKBACK):day + 1]
            pb = close[b].iloc[max(0, day - COINT_LOOKBACK):day + 1]
            hurst, beta, hl = rolling_cointegration(pa, pb)
            if np.isnan(beta):
                to_close.append(pair_key)
                continue
            spread = pa / pa.iloc[0] - beta * (pb / pb.iloc[0])
            z = (spread.iloc[-1] - spread.rolling(63).mean().iloc[-1]) / max(spread.rolling(63).std().iloc[-1], 1e-8)
            hold_days = day - pos['entry_day']

            if (pos['direction'] == 1 and z <= EXIT_Z) or \
               (pos['direction'] == -1 and z >= -EXIT_Z) or \
               abs(z) >= STOP_Z or hold_days >= MAX_HOLD:
                to_close.append(pair_key)
                reason = 'revert' if abs(z) <= EXIT_Z else ('stop' if abs(z) >= STOP_Z else 'time')
                # Compute P&L
                ret_a = close[a].iloc[day] / close[a].iloc[pos['entry_day']] - 1
                ret_b = close[b].iloc[day] / close[b].iloc[pos['entry_day']] - 1
                pair_ret = pos['direction'] * (ret_a - ret_b) * POS_SIZE
                trade_log.append({'pair': pair_key, 'ret': pair_ret, 'hold': hold_days, 'reason': reason})

        for pk in to_close:
            if pk in positions:
                del positions[pk]

        # Check entries — only if we have capacity
        if len(positions) < MAX_PAIRS:
            # Find best pairs by cointegration
            pair_scores = []
            for a, b in all_pairs:
                pair_key = f"{a}|{b}"
                if pair_key in positions:
                    continue
                pa = close[a].iloc[max(0, day - COINT_LOOKBACK):day + 1]
                pb = close[b].iloc[max(0, day - COINT_LOOKBACK):day + 1]
                hurst, beta, hl = rolling_cointegration(pa, pb)
                if np.isnan(hurst) or hurst >= 0.45 or np.isnan(beta):
                    continue
                spread = pa / pa.iloc[0] - beta * (pb / pb.iloc[0])
                smean = spread.rolling(63).mean().iloc[-1]
                sstd = spread.rolling(63).std().iloc[-1]
                if sstd < 1e-8:
                    continue
                z = (spread.iloc[-1] - smean) / sstd
                if abs(z) >= ENTRY_Z:
                    pair_scores.append((pair_key, a, b, z, hurst))

            # Sort by |z| descending
            pair_scores.sort(key=lambda x: abs(x[3]), reverse=True)

            for pair_key, a, b, z, hurst in pair_scores:
                if len(positions) >= MAX_PAIRS:
                    break
                if shuffle_entries:
                    # Randomly decide whether to enter (same frequency as real)
                    if np.random.random() > 0.5:
                        continue
                    # Random direction
                    direction = np.random.choice([-1, 1])
                else:
                    direction = -1 if z > 0 else 1  # Fade the z-score
                positions[pair_key] = {'direction': direction, 'entry_day': day}

        # Mark-to-market
        day_pnl = 0
        for pair_key, pos in positions.items():
            a, b = pair_key.split('|')
            if day > 0:
                ret_a = close[a].iloc[day] / close[a].iloc[day - 1] - 1
                ret_b = close[b].iloc[day] / close[b].iloc[day - 1] - 1
                day_pnl += pos['direction'] * (ret_a - ret_b) * POS_SIZE
        daily_returns[day] = day_pnl

    return daily_returns[COINT_LOOKBACK + 63:], trade_log


def compute_metrics(rets):
    if len(rets) == 0 or np.std(rets) == 0:
        return {'sharpe': 0, 'sortino': 0, 'cagr': 0, 'maxdd': 0, 'wr': 0, 'pf': 0, 'trades': 0}
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252)
    downside = rets[rets < 0]
    sortino = np.mean(rets) / np.std(downside) * np.sqrt(252) if len(downside) > 0 else sharpe
    cum = (1 + rets).cumprod() if isinstance(rets, pd.Series) else np.cumprod(1 + rets)
    years = len(rets) / 252
    cagr = (cum[-1] ** (1 / max(years, 0.1)) - 1) * 100
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    maxdd = np.min(dd) * 100
    wins = np.sum(rets > 0)
    wr = wins / len(rets[rets != 0]) * 100 if np.sum(rets != 0) > 0 else 0
    gain = np.sum(rets[rets > 0])
    loss = abs(np.sum(rets[rets < 0]))
    pf = gain / loss if loss > 0 else float('inf')
    return {'sharpe': sharpe, 'sortino': sortino, 'cagr': cagr, 'maxdd': maxdd, 'wr': wr, 'pf': pf, 'years': years}


# 1. Run baseline
print("\n[1] Running baseline (pure z-score pairs trading)...")
baseline_rets, trades = run_baseline_backtest(close, shuffle_entries=False)
m = compute_metrics(baseline_rets)
print(f"\n  Baseline Results:")
print(f"    Sharpe: {m['sharpe']:.3f}")
print(f"    Sortino: {m['sortino']:.3f}")
print(f"    CAGR: {m['cagr']:.1f}%")
print(f"    MaxDD: {m['maxdd']:.1f}%")
print(f"    Years: {m['years']:.1f}")
print(f"    Trades: {len(trades)}")
if trades:
    wr_trades = sum(1 for t in trades if t['ret'] > 0) / len(trades) * 100
    avg_hold = np.mean([t['hold'] for t in trades])
    reasons = {}
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    print(f"    WR (trades): {wr_trades:.1f}%")
    print(f"    Avg hold: {avg_hold:.1f} days")
    print(f"    Exit reasons: {reasons}")

# Market correlation
if 'SPY' in close.columns:
    spy_rets = close['SPY'].pct_change().values[COINT_LOOKBACK + 63:]
    if len(spy_rets) == len(baseline_rets):
        corr = np.corrcoef(baseline_rets, spy_rets)[0, 1]
        print(f"    Market correlation (vs SPY): {corr:.3f}")

# 2. Permutation test — shuffle entries (random timing + random direction)
print(f"\n[2] Permutation test ({N_PERM} shuffles)...")
real_sharpe = m['sharpe']
perm_sharpes = []
for i in range(N_PERM):
    if i % 10 == 0:
        print(f"    Perm {i}/{N_PERM}...", flush=True)
    perm_rets, _ = run_baseline_backtest(close, shuffle_entries=True)
    pm = compute_metrics(perm_rets)
    perm_sharpes.append(pm['sharpe'])

perm_mean = np.mean(perm_sharpes)
perm_std = np.std(perm_sharpes) if np.std(perm_sharpes) > 0 else 1
p_value = np.mean([s >= real_sharpe for s in perm_sharpes])
perm_pass = p_value < 0.05

print(f"\n  Permutation Result:")
print(f"    Real Sharpe: {real_sharpe:.3f}")
print(f"    Random Sharpe: {perm_mean:.3f} ± {perm_std:.3f}")
print(f"    p-value: {p_value:.3f}")
print(f"    VERDICT: {'PASS — edge is REAL' if perm_pass else 'FAIL — edge is ARTIFACT'}")

# 3. Sub-period stability
print("\n[3] Sub-period stability...")
n = len(baseline_rets)
q_size = n // 4
q_sharpes = []
for q in range(4):
    start = q * q_size
    end = start + q_size if q < 3 else n
    qm = compute_metrics(baseline_rets[start:end])
    q_sharpes.append(qm['sharpe'])
    print(f"    Q{q+1}: Sharpe {qm['sharpe']:.3f}")

cv = np.std(q_sharpes) / abs(np.mean(q_sharpes)) if abs(np.mean(q_sharpes)) > 0 else 99
subp_pass = cv < 1.0
print(f"    CV: {cv:.3f} {'PASS' if subp_pass else 'FAIL'}")

# 4. Outlier sensitivity
print("\n[4] Outlier sensitivity...")
trimmed = np.sort(baseline_rets)[int(len(baseline_rets)*0.01):int(len(baseline_rets)*0.99)]
tm = compute_metrics(trimmed)
degradation = (1 - tm['sharpe'] / max(abs(real_sharpe), 0.001)) * 100 if real_sharpe != 0 else 0
outlier_pass = abs(degradation) < 50
print(f"    Full Sharpe: {real_sharpe:.3f}")
print(f"    Trimmed (1-99%): {tm['sharpe']:.3f}")
print(f"    Degradation: {degradation:.1f}%")
print(f"    VERDICT: {'PASS' if outlier_pass else 'FAIL'}")

# 5. R1 regime test
print("\n[5] R1 regime test...")
if 'SPY' in close.columns:
    spy_close = close['SPY'].values[COINT_LOOKBACK + 63:]
    if len(spy_close) == len(baseline_rets) + 1:
        spy_close = spy_close[:-1]
    elif len(spy_close) != len(baseline_rets):
        spy_close = spy_close[:len(baseline_rets)]
    spy_daily = np.diff(spy_close) / spy_close[:-1]
    # Align
    min_len = min(len(spy_daily), len(baseline_rets))
    spy_daily = spy_daily[:min_len]
    bl_rets = baseline_rets[:min_len]

    green = bl_rets[spy_daily > 0]
    red = bl_rets[spy_daily <= 0]

    green_sharpe = np.mean(green) / np.std(green) * np.sqrt(252) if len(green) > 10 and np.std(green) > 0 else 0
    red_sharpe = np.mean(red) / np.std(red) * np.sqrt(252) if len(red) > 10 and np.std(red) > 0 else 0

    gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.001)
    r1_pass = gap < 0.50
    print(f"    Green day Sharpe: {green_sharpe:.3f}")
    print(f"    Red day Sharpe: {red_sharpe:.3f}")
    print(f"    Gap: {gap:.3f}")
    print(f"    VERDICT: {'PASS' if r1_pass else 'FAIL'}")
else:
    r1_pass = False
    print("    No SPY data — FAIL by default")

# Final verdict
gates = sum([perm_pass, subp_pass, outlier_pass, r1_pass])
print(f"\n{'='*70}")
print(f"FINAL VERDICT: {gates}/4 adversarial gates")
print(f"  Permutation: {'PASS' if perm_pass else 'FAIL'} (p={p_value:.3f})")
print(f"  Sub-period:  {'PASS' if subp_pass else 'FAIL'} (CV={cv:.3f})")
print(f"  Outlier:     {'PASS' if outlier_pass else 'FAIL'} (deg={degradation:.1f}%)")
print(f"  R1 Regime:   {'PASS' if r1_pass else 'FAIL'}")
print(f"{'='*70}")

# Save results
results = {
    'baseline': m,
    'perm': {'real': real_sharpe, 'random_mean': perm_mean, 'p_value': p_value, 'pass': perm_pass},
    'subperiod': {'cv': cv, 'sharpes': q_sharpes, 'pass': subp_pass},
    'outlier': {'degradation': degradation, 'pass': outlier_pass},
    'r1': {'gap': gap if 'gap' in dir() else None, 'pass': r1_pass},
    'gates_passed': gates,
    'trades': len(trades),
    'timestamp': datetime.now().isoformat()
}

with open(f'{OUTPUT}/baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT}/baseline_results.json")
