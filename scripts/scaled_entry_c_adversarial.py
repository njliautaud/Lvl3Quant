#!/usr/bin/env python3
"""
Adversarial Validation: Volatility-Scaled Mean Reversion Entry (Scaled Entry Variant C)

6 adversarial tests:
1. Inverse Signal Test
2. Random Timing Percentile Test (1000 perms)
3. Sub-Period Stability Test (4 equal sub-periods)
4. Remove Top-3 Tickers Test
5. Parameter Sensitivity Grid
6. Cost Sensitivity Test
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from itertools import product

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
TICKERS = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]
START = '2022-01-01'
END = '2026-07-31'
CAPITAL = 669.0
MAX_CONCURRENT = 3

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading price data...")
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
close_df = raw['Close'].copy()
close_df.columns = [str(c) for c in close_df.columns]
print(f"Data: {close_df.shape[0]} days, {close_df.shape[1]} tickers")


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def precompute_indicators(prices, tickers=None):
    """Precompute all indicators once."""
    if tickers is None:
        tickers = [t for t in TICKERS if t in prices.columns]
    df = prices[tickers].copy()

    high20 = df.rolling(20).max()
    low20 = df.rolling(20).min()
    rsi = pd.DataFrame({t: compute_rsi(df[t]) for t in df.columns})
    vol20 = df.pct_change().rolling(20).std() * np.sqrt(252) * 100

    return df, high20, low20, rsi, vol20


def generate_signals(df, high20, low20, rsi, vol20,
                     dip_pct=5, rsi_thresh=40, vol_high=40, inverse=False):
    """Generate all candidate entry signals (no concurrent limit yet)."""
    signals = []
    for t in df.columns:
        prices_t = df[t].values
        high20_t = high20[t].values
        low20_t = low20[t].values
        rsi_t = rsi[t].values
        vol_t = vol20[t].values

        for i in range(20, len(df)):
            price = prices_t[i]
            if np.isnan(price):
                continue

            if inverse:
                ref = low20_t[i]
                if np.isnan(ref) or ref == 0:
                    continue
                condition = (price - ref) / ref * 100 > dip_pct
            else:
                ref = high20_t[i]
                if np.isnan(ref) or ref == 0:
                    continue
                condition = (ref - price) / ref * 100 > dip_pct

            r = rsi_t[i]
            if np.isnan(r) or r >= rsi_thresh:
                continue
            if not condition:
                continue

            v = vol_t[i]
            if np.isnan(v):
                continue
            pos_size = 100 if v > vol_high else (150 if v > 20 else 200)

            signals.append((i, t, price, pos_size))

    # Sort by date index
    signals.sort(key=lambda x: x[0])
    return signals


def execute_trades(df, signals, hold_days=10, slippage_bps=2):
    """Apply concurrent limit and compute trade PnL."""
    trades = []
    active_exit_indices = []

    for idx, ticker, entry_price, pos_size in signals:
        # Remove expired
        active_exit_indices = [e for e in active_exit_indices if e > idx]

        if len(active_exit_indices) >= MAX_CONCURRENT:
            continue

        exit_idx = idx + hold_days
        if exit_idx >= len(df):
            continue

        exit_price = df[ticker].iloc[exit_idx]
        if np.isnan(exit_price):
            continue

        shares = pos_size / entry_price
        slip = slippage_bps / 10000
        adj_entry = entry_price * (1 + slip)
        adj_exit = exit_price * (1 - slip)
        pnl = shares * (adj_exit - adj_entry)
        ret = (adj_exit - adj_entry) / adj_entry

        trades.append({
            'ticker': ticker,
            'entry_idx': idx,
            'entry_date': str(df.index[idx].date()),
            'exit_date': str(df.index[exit_idx].date()),
            'entry_price': round(entry_price, 4),
            'exit_price': round(exit_price, 4),
            'pos_size': pos_size,
            'pnl': round(pnl, 4),
            'ret': round(ret, 6),
        })
        active_exit_indices.append(exit_idx)

    return trades


def calc_metrics(trades):
    if not trades:
        return {'sharpe': 0.0, 'wr': 0.0, 'total_pnl': 0.0, 'n_trades': 0}

    rets = [t['ret'] for t in trades]
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 1e-9

    n_years = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
    trades_per_year = len(trades) / n_years
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 1e-12 else 0.0

    wr = sum(1 for r in rets if r > 0) / len(rets) * 100

    return {
        'sharpe': round(sharpe, 4),
        'wr': round(wr, 2),
        'total_pnl': round(sum(t['pnl'] for t in trades), 2),
        'n_trades': len(trades),
    }


# ── Precompute ──────────────────────────────────────────────────────────────
df, high20, low20, rsi_df, vol20 = precompute_indicators(close_df)

# ── Baseline ────────────────────────────────────────────────────────────────
print("\n=== BASELINE ===")
baseline_signals = generate_signals(df, high20, low20, rsi_df, vol20)
baseline_trades = execute_trades(df, baseline_signals)
baseline = calc_metrics(baseline_trades)
print(f"Trades: {baseline['n_trades']}, Sharpe: {baseline['sharpe']}, "
      f"WR: {baseline['wr']}%, PnL: ${baseline['total_pnl']}")

results = {
    'strategy': 'Scaled Entry Variant C',
    'baseline': baseline,
    'tests': {},
    'overall_pass': True,
    'timestamp': datetime.now().isoformat(),
}

# ── Test 1: Inverse Signal ──────────────────────────────────────────────────
print("\n=== TEST 1: Inverse Signal ===")
inv_signals = generate_signals(df, high20, low20, rsi_df, vol20, inverse=True)
inv_trades = execute_trades(df, inv_signals)
inv = calc_metrics(inv_trades)
inv_ratio = inv['sharpe'] / baseline['sharpe'] if baseline['sharpe'] != 0 else 999
t1_pass = inv['sharpe'] < 0.5 * baseline['sharpe']
print(f"Inverse Sharpe: {inv['sharpe']} ({inv['n_trades']} trades), ratio: {inv_ratio:.3f}, Pass: {t1_pass}")
results['tests']['1_inverse_signal'] = {
    'inverse_sharpe': inv['sharpe'],
    'inverse_trades': inv['n_trades'],
    'baseline_sharpe': baseline['sharpe'],
    'ratio': round(inv_ratio, 4),
    'pass': t1_pass,
    'threshold': '< 50% of baseline',
}

# ── Test 2: Random Timing Percentile (1000 perms) ──────────────────────────
print("\n=== TEST 2: Random Timing (1000 perms) ===")
# Precompute: we shuffle date indices among signals, then re-sort and apply concurrent limit
sig_indices = np.array([s[0] for s in baseline_signals])
sig_data = [(s[1], s[2], s[3]) for s in baseline_signals]  # ticker, price, pos_size
n_sigs = len(baseline_signals)

perm_sharpes = []
for p in range(1000):
    if p % 200 == 0:
        print(f"  Permutation {p}/1000...")
    rng = np.random.RandomState(p)
    shuffled_idx = sig_indices.copy()
    rng.shuffle(shuffled_idx)

    # Rebuild signals with shuffled dates
    perm_sigs = [(int(shuffled_idx[i]), sig_data[i][0], sig_data[i][1], sig_data[i][2])
                 for i in range(n_sigs)]
    perm_sigs.sort(key=lambda x: x[0])

    perm_trades = execute_trades(df, perm_sigs)
    perm_sharpes.append(calc_metrics(perm_trades)['sharpe'])

perm_sharpes = np.array(perm_sharpes)
pval = float(np.mean(perm_sharpes >= baseline['sharpe']))
percentile = float(np.mean(perm_sharpes < baseline['sharpe'])) * 100
t2_pass = pval < 0.05
print(f"Baseline Sharpe: {baseline['sharpe']}, p-value: {pval:.4f}, "
      f"Percentile: {percentile:.1f}%, Pass: {t2_pass}")
results['tests']['2_random_timing'] = {
    'p_value': round(pval, 4),
    'percentile': round(percentile, 2),
    'perm_mean_sharpe': round(float(np.mean(perm_sharpes)), 4),
    'perm_std_sharpe': round(float(np.std(perm_sharpes)), 4),
    'pass': t2_pass,
    'threshold': 'p < 0.05',
}

# ── Test 3: Sub-Period Stability ────────────────────────────────────────────
print("\n=== TEST 3: Sub-Period Stability ===")
dates = close_df.index
n = len(dates)
quarter = n // 4
sub_results = []
for q in range(4):
    s_idx = q * quarter
    e_idx = (q + 1) * quarter if q < 3 else n
    sub_df = close_df.iloc[s_idx:e_idx]
    sub_tickers = [t for t in TICKERS if t in sub_df.columns]
    sub_prices, sh20, sl20, sr, sv = precompute_indicators(sub_df, sub_tickers)
    sub_sigs = generate_signals(sub_prices, sh20, sl20, sr, sv)
    sub_trades = execute_trades(sub_prices, sub_sigs)
    sub_m = calc_metrics(sub_trades)
    period_str = f"{sub_df.index[0].date()} to {sub_df.index[-1].date()}"
    print(f"  Q{q+1} ({period_str}): Sharpe={sub_m['sharpe']}, "
          f"Trades={sub_m['n_trades']}, WR={sub_m['wr']}%")
    sub_results.append({
        'period': period_str,
        'sharpe': sub_m['sharpe'],
        'n_trades': sub_m['n_trades'],
        'wr': sub_m['wr'],
        'total_pnl': sub_m['total_pnl'],
    })

t3_pass = all(s['sharpe'] > 0 for s in sub_results)
print(f"All positive Sharpe: {t3_pass}")
results['tests']['3_sub_period_stability'] = {
    'sub_periods': sub_results,
    'pass': t3_pass,
    'threshold': 'All 4 sub-periods Sharpe > 0',
}

# ── Test 4: Remove Top-3 Tickers ───────────────────────────────────────────
print("\n=== TEST 4: Remove Top-3 Tickers ===")
from collections import defaultdict
tpnl = defaultdict(float)
for t in baseline_trades:
    tpnl[t['ticker']] += t['pnl']
sorted_tickers = sorted(tpnl.items(), key=lambda x: x[1], reverse=True)
top3 = [t[0] for t in sorted_tickers[:3]]
print(f"Top 3 by PnL: {top3} ({[round(t[1],2) for t in sorted_tickers[:3]]})")

remaining = [t for t in TICKERS if t not in top3]
r_df, rh, rl, rr, rv = precompute_indicators(close_df, remaining)
r_sigs = generate_signals(r_df, rh, rl, rr, rv)
r_trades = execute_trades(r_df, r_sigs)
t4 = calc_metrics(r_trades)
sharpe_drop = 1 - (t4['sharpe'] / baseline['sharpe']) if baseline['sharpe'] != 0 else 1
t4_pass = sharpe_drop < 0.5
print(f"Without top 3: Sharpe={t4['sharpe']}, Drop={sharpe_drop:.1%}, Pass: {t4_pass}")
results['tests']['4_remove_top3'] = {
    'top3_tickers': top3,
    'top3_pnl': [round(t[1], 2) for t in sorted_tickers[:3]],
    'reduced_sharpe': t4['sharpe'],
    'baseline_sharpe': baseline['sharpe'],
    'sharpe_drop_pct': round(sharpe_drop * 100, 2),
    'pass': t4_pass,
    'threshold': 'Sharpe drop < 50%',
}

# ── Test 5: Parameter Sensitivity Grid ──────────────────────────────────────
print("\n=== TEST 5: Parameter Sensitivity Grid ===")
dip_vals = [3, 5, 7, 10]
rsi_vals = [30, 35, 40, 45]
vol_vals = [30, 40, 50]
hold_vals = [5, 7, 10, 15]

total_combos = len(dip_vals) * len(rsi_vals) * len(vol_vals) * len(hold_vals)
print(f"Testing {total_combos} parameter combinations...")

grid_results = []
count = 0
for dip, rsi_t, vol_h, hold in product(dip_vals, rsi_vals, vol_vals, hold_vals):
    count += 1
    if count % 48 == 0:
        print(f"  {count}/{total_combos}...")
    sigs = generate_signals(df, high20, low20, rsi_df, vol20,
                            dip_pct=dip, rsi_thresh=rsi_t, vol_high=vol_h)
    trades = execute_trades(df, sigs, hold_days=hold)
    m = calc_metrics(trades)
    grid_results.append({
        'dip': dip, 'rsi': rsi_t, 'vol_high': vol_h, 'hold': hold,
        'sharpe': m['sharpe'], 'n_trades': m['n_trades'],
    })

sharpe_above_03 = sum(1 for g in grid_results if g['sharpe'] > 0.3)
pct_above = sharpe_above_03 / total_combos * 100
t5_pass = pct_above > 50
print(f"Combos with Sharpe > 0.3: {sharpe_above_03}/{total_combos} = {pct_above:.1f}%, Pass: {t5_pass}")

best = max(grid_results, key=lambda x: x['sharpe'])
worst = min(grid_results, key=lambda x: x['sharpe'])
results['tests']['5_parameter_sensitivity'] = {
    'total_combos': total_combos,
    'pct_sharpe_above_0_3': round(pct_above, 2),
    'best_combo': best,
    'worst_combo': worst,
    'mean_sharpe': round(float(np.mean([g['sharpe'] for g in grid_results])), 4),
    'median_sharpe': round(float(np.median([g['sharpe'] for g in grid_results])), 4),
    'pass': t5_pass,
    'threshold': '> 50% combos with Sharpe > 0.3',
}

# ── Test 6: Cost Sensitivity ───────────────────────────────────────────────
print("\n=== TEST 6: Cost Sensitivity ===")
slip_vals = [0, 2, 5, 10, 20, 50]
cost_results = []
for slip in slip_vals:
    trades = execute_trades(df, baseline_signals, slippage_bps=slip)
    m = calc_metrics(trades)
    print(f"  {slip} bps: Sharpe={m['sharpe']}, PnL=${m['total_pnl']}")
    cost_results.append({
        'slippage_bps': slip,
        'sharpe': m['sharpe'],
        'total_pnl': m['total_pnl'],
        'n_trades': m['n_trades'],
    })

breakeven = None
for i in range(len(cost_results) - 1):
    pnl1 = cost_results[i]['total_pnl']
    pnl2 = cost_results[i + 1]['total_pnl']
    if pnl1 >= 0 and pnl2 < 0:
        bps1 = cost_results[i]['slippage_bps']
        bps2 = cost_results[i + 1]['slippage_bps']
        breakeven = bps1 + (bps2 - bps1) * pnl1 / (pnl1 - pnl2)
        break

if breakeven is None:
    if cost_results[-1]['total_pnl'] > 0:
        breakeven = 999.0
    else:
        breakeven = 0.0

t6_pass = breakeven > 20
print(f"Breakeven slippage: {breakeven:.1f} bps, Pass: {t6_pass}")
results['tests']['6_cost_sensitivity'] = {
    'results': cost_results,
    'breakeven_bps': round(float(breakeven), 2),
    'pass': t6_pass,
    'threshold': 'Breakeven > 20 bps',
}

# ── Summary ─────────────────────────────────────────────────────────────────
all_pass = all(results['tests'][k]['pass'] for k in results['tests'])
results['overall_pass'] = all_pass

n_pass = sum(1 for k in results['tests'] if results['tests'][k]['pass'])
n_total = len(results['tests'])

print(f"\n{'='*60}")
print(f"ADVERSARIAL VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"Strategy: {results['strategy']}")
print(f"Baseline: Sharpe={baseline['sharpe']}, WR={baseline['wr']}%, "
      f"Trades={baseline['n_trades']}, PnL=${baseline['total_pnl']}")
print(f"{'='*60}")
for k, v in results['tests'].items():
    status = "PASS" if v['pass'] else "FAIL"
    print(f"  {k}: {status}")
print(f"{'='*60}")
print(f"Overall: {n_pass}/{n_total} passed — {'PASS' if all_pass else 'FAIL'}")

out_path = '/home/jupiter/Lvl3Quant/data/scaled_entry_c_adversarial.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
