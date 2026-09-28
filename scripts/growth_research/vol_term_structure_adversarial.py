#!/usr/bin/env python3
"""
Adversarial Validation: Strategy F — Volatility Term Structure Trade
6-test battery with extra strictness for small sample (20 trades).
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# ── Configuration ──────────────────────────────────────────────────────
BASKET = ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','JPM','UNH','LLY','AVGO',
          'AMD','HD','ABBV','MRK','COST','CRM','NFLX','ADBE','PG','JNJ']
START = '2020-01-01'
END   = '2026-07-01'
POSITION_SIZE = 300.0
MAX_CONCURRENT = 2
VIX_MULT = 1.15
VIX_LOOKBACK = 60
MAX_HOLD = 21
TP_PCT = 0.10
SL_PCT = -0.15
BASELINE_SHARPE = 3.43

np.random.seed(42)

# ── Data Download (with caching) ───────────────────────────────────────
import os, pickle

CACHE_FILE = '/tmp/vol_term_struct_data_cache.pkl'

if os.path.exists(CACHE_FILE):
    log("Loading cached data...")
    with open(CACHE_FILE, 'rb') as f:
        cached = pickle.load(f)
    vix = cached['vix']
    prices_raw = cached['prices']
else:
    log("Downloading data...")
    vix = yf.download('^VIX', start=START, end=END, progress=False)['Close'].squeeze()
    vix.index = vix.index.tz_localize(None) if vix.index.tz else vix.index

    tickers_to_dl = BASKET + ['SPY']
    prices_raw = yf.download(tickers_to_dl, start=START, end=END, progress=False)['Close']
    prices_raw.index = prices_raw.index.tz_localize(None) if prices_raw.index.tz else prices_raw.index

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'vix': vix, 'prices': prices_raw}, f)
    log("Data cached.")

# Align dates
common = vix.index.intersection(prices_raw.index)
vix = vix.loc[common]
prices = prices_raw.loc[common]

# Pre-compute numpy arrays for speed
vix_arr = vix.values.astype(float)
dates_arr = vix.index.values  # numpy datetime64
n_days = len(common)

# Price matrix: (n_days, n_stocks) for BASKET
price_matrix = prices[BASKET].values.astype(float)  # shape (n_days, 20)
n_stocks = len(BASKET)

log(f"Data: {n_days} trading days, {common[0].date()} to {common[-1].date()}")
log(f"VIX range: {vix_arr.min():.1f} - {vix_arr.max():.1f}")

# ── Core Strategy Engine (numpy-based for speed) ─────────────────────
def run_strategy_fast(vix_a, price_mat, vix_mult=VIX_MULT, vix_lookback=VIX_LOOKBACK,
                      max_hold=MAX_HOLD, tp=TP_PCT, sl=SL_PCT,
                      max_concurrent=MAX_CONCURRENT, exit_mult=1.0,
                      stock_mask=None):
    """
    Fast numpy-based strategy runner.
    stock_mask: boolean array of shape (n_stocks,) — which stocks to include.
    Returns list of trade dicts.
    """
    n = len(vix_a)
    if stock_mask is None:
        stock_mask = np.ones(price_mat.shape[1], dtype=bool)

    # Rolling mean of VIX
    vix_avg = np.full(n, np.nan)
    for i in range(vix_lookback - 1, n):
        vix_avg[i] = np.mean(vix_a[i - vix_lookback + 1:i + 1])

    trades = []
    # Active positions: list of (entry_idx, entry_prices_for_selected_stocks)
    active = []

    for i in range(vix_lookback + 1, n):
        avg = vix_avg[i]
        if np.isnan(avg):
            continue

        cur_vix = vix_a[i]
        prev_vix = vix_a[i - 1]

        # Check exits
        new_active = []
        for entry_idx, entry_p in active:
            hold_days = i - entry_idx
            cur_p = price_mat[i, stock_mask]

            # Valid stocks (non-nan, positive entry)
            valid = ~np.isnan(entry_p) & ~np.isnan(cur_p) & (entry_p > 0)
            if valid.sum() == 0:
                new_active.append((entry_idx, entry_p))
                continue

            rets = cur_p[valid] / entry_p[valid] - 1
            basket_ret = rets.mean()

            exit_triggered = False
            reason = ''
            if cur_vix < exit_mult * avg:
                exit_triggered = True; reason = 'vix_normalized'
            elif hold_days >= max_hold:
                exit_triggered = True; reason = 'max_hold'
            elif basket_ret >= tp:
                exit_triggered = True; reason = 'tp'
            elif basket_ret <= sl:
                exit_triggered = True; reason = 'sl'

            if exit_triggered:
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'hold_days': hold_days,
                    'return': float(basket_ret),
                    'reason': reason
                })
            else:
                new_active.append((entry_idx, entry_p))
        active = new_active

        # Check entry
        if len(active) < max_concurrent:
            spike = cur_vix > vix_mult * avg
            declining = cur_vix < prev_vix
            if spike and declining:
                entry_p = price_mat[i, stock_mask].copy()
                active.append((i, entry_p))

    # Close remaining
    for entry_idx, entry_p in active:
        cur_p = price_mat[-1, stock_mask]
        valid = ~np.isnan(entry_p) & ~np.isnan(cur_p) & (entry_p > 0)
        if valid.sum() > 0:
            basket_ret = float((cur_p[valid] / entry_p[valid] - 1).mean())
        else:
            basket_ret = 0.0
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': n - 1,
            'hold_days': n - 1 - entry_idx,
            'return': basket_ret,
            'reason': 'end_of_period'
        })

    return trades


def compute_sharpe(trades):
    if len(trades) < 2:
        return 0.0
    rets = np.array([t['return'] for t in trades])
    if rets.std() == 0:
        return 0.0
    avg_hold = np.mean([t['hold_days'] for t in trades])
    trades_per_year = 252 / max(avg_hold, 1)
    return float((rets.mean() / rets.std()) * np.sqrt(trades_per_year))


def compute_metrics(trades):
    if not trades:
        return {'sharpe': 0, 'wr': 0, 'pf': 0, 'n_trades': 0, 'avg_ret': 0, 'total_ret': 0}
    rets = np.array([t['return'] for t in trades])
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    wr = len(wins) / len(rets)
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    sharpe = compute_sharpe(trades)
    return {
        'sharpe': sharpe, 'wr': wr, 'pf': pf,
        'n_trades': len(trades), 'avg_ret': float(rets.mean()),
        'total_ret': float(rets.sum()), 'max_dd_trade': float(rets.min())
    }


# ── Run baseline ──────────────────────────────────────────────────────
log("\n" + "="*70)
log("BASELINE: Strategy F — Volatility Term Structure Trade")
log("="*70)

baseline_trades = run_strategy_fast(vix_arr, price_matrix)
baseline_metrics = compute_metrics(baseline_trades)

log(f"Trades: {baseline_metrics['n_trades']}")
log(f"Sharpe: {baseline_metrics['sharpe']:.2f}")
log(f"WR:     {baseline_metrics['wr']:.1%}")
log(f"PF:     {baseline_metrics['pf']:.2f}")
log(f"Avg return: {baseline_metrics['avg_ret']:.2%}")
log(f"Total return: {baseline_metrics['total_ret']:.2%}")

for t in baseline_trades:
    entry_d = pd.Timestamp(dates_arr[t['entry_idx']]).date()
    exit_d = pd.Timestamp(dates_arr[t['exit_idx']]).date()
    log(f"  {entry_d} -> {exit_d} ({t['hold_days']}d) ret={t['return']:+.2%} [{t['reason']}]")

reimpl_sharpe = baseline_metrics['sharpe']

# ══════════════════════════════════════════════════════════════════════
# TEST 1: RE-IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════
log("\n" + "="*70)
log("TEST 1: RE-IMPLEMENTATION")
log("="*70)

test1_threshold = 0.70 * BASELINE_SHARPE
log(f"Re-implemented Sharpe: {reimpl_sharpe:.2f}")
log(f"Threshold (70% of {BASELINE_SHARPE}): {test1_threshold:.2f}")
test1_pass = reimpl_sharpe > test1_threshold
log(f"RESULT: {'PASS' if test1_pass else 'FAIL'} — {reimpl_sharpe:.2f} {'>' if test1_pass else '<='} {test1_threshold:.2f}")


# ══════════════════════════════════════════════════════════════════════
# TEST 2: INVERSE SIGNAL
# ══════════════════════════════════════════════════════════════════════
log("\n" + "="*70)
log("TEST 2: INVERSE SIGNAL")
log("="*70)

def run_inverse_fast(vix_a, price_mat):
    """Buy when VIX < 0.85 * avg AND VIX rising."""
    n = len(vix_a)
    vix_avg = np.full(n, np.nan)
    for i in range(VIX_LOOKBACK - 1, n):
        vix_avg[i] = np.mean(vix_a[i - VIX_LOOKBACK + 1:i + 1])

    trades = []
    active = []

    for i in range(VIX_LOOKBACK + 1, n):
        avg = vix_avg[i]
        if np.isnan(avg):
            continue
        cur_vix = vix_a[i]
        prev_vix = vix_a[i - 1]

        new_active = []
        for entry_idx, entry_p in active:
            hold_days = i - entry_idx
            cur_p = price_mat[i]
            valid = ~np.isnan(entry_p) & ~np.isnan(cur_p) & (entry_p > 0)
            if valid.sum() == 0:
                new_active.append((entry_idx, entry_p))
                continue
            basket_ret = float((cur_p[valid] / entry_p[valid] - 1).mean())

            exit_triggered = False
            # Inverse exit: VIX goes ABOVE avg
            if cur_vix > avg:
                exit_triggered = True
            elif hold_days >= MAX_HOLD:
                exit_triggered = True
            elif basket_ret >= TP_PCT:
                exit_triggered = True
            elif basket_ret <= SL_PCT:
                exit_triggered = True

            if exit_triggered:
                trades.append({'entry_idx': entry_idx, 'hold_days': hold_days, 'return': basket_ret})
            else:
                new_active.append((entry_idx, entry_p))
        active = new_active

        if len(active) < MAX_CONCURRENT:
            if cur_vix < 0.85 * avg and cur_vix > prev_vix:
                active.append((i, price_mat[i].copy()))

    return trades

inverse_trades = run_inverse_fast(vix_arr, price_matrix)
inverse_metrics = compute_metrics(inverse_trades)

log(f"Inverse trades: {inverse_metrics['n_trades']}")
log(f"Inverse Sharpe: {inverse_metrics['sharpe']:.2f}")
log(f"Original Sharpe: {reimpl_sharpe:.2f}")
test2_threshold = 0.50 * reimpl_sharpe
test2_pass = inverse_metrics['sharpe'] < test2_threshold
log(f"RESULT: {'PASS' if test2_pass else 'FAIL'} — Inverse Sharpe {inverse_metrics['sharpe']:.2f} {'<' if test2_pass else '>='} {test2_threshold:.2f} (50% of original)")


# ══════════════════════════════════════════════════════════════════════
# TEST 3: RANDOM TIMING (500 permutations)
# ══════════════════════════════════════════════════════════════════════
log("\n" + "="*70)
log("TEST 3: RANDOM TIMING (500 permutations)")
log("="*70)

n_baseline_trades = len(baseline_trades)
n_perms = 500
random_sharpes = np.zeros(n_perms)

# Pre-compute: for every possible entry day, compute the basket return
# at each holding period 1..MAX_HOLD, and find first exit day
# This avoids repeated price lookups in the permutation loop
log("Pre-computing exit returns for all possible entry dates...")

valid_start = VIX_LOOKBACK
valid_end = n_days - MAX_HOLD - 2
valid_indices = np.arange(valid_start, valid_end)

# For each valid entry index, find exit return using same rules
entry_returns = np.full(n_days, np.nan)  # final return for each entry day
entry_hold = np.full(n_days, np.nan)

for idx in valid_indices:
    entry_p = price_matrix[idx]
    valid_stocks = ~np.isnan(entry_p) & (entry_p > 0)
    if valid_stocks.sum() == 0:
        continue

    for d in range(1, MAX_HOLD + 1):
        if idx + d >= n_days:
            break
        cur_p = price_matrix[idx + d]
        v = valid_stocks & ~np.isnan(cur_p)
        if v.sum() == 0:
            continue
        ret = float((cur_p[v] / entry_p[v] - 1).mean())

        # VIX exit check
        cur_v = vix_arr[idx + d]
        avg_v = vix_arr[max(0, idx + d - VIX_LOOKBACK + 1):idx + d + 1].mean()
        vix_exit = cur_v < avg_v

        if ret >= TP_PCT or ret <= SL_PCT or vix_exit or d == MAX_HOLD:
            entry_returns[idx] = ret
            entry_hold[idx] = d
            break

log("Running 500 random permutations...")
valid_mask = ~np.isnan(entry_returns)
valid_entry_indices = np.where(valid_mask)[0]

for perm in range(n_perms):
    chosen = np.random.choice(valid_entry_indices, size=min(n_baseline_trades, len(valid_entry_indices)), replace=False)
    rets = entry_returns[chosen]
    holds = entry_hold[chosen]
    if len(rets) >= 2 and np.std(rets) > 0:
        avg_h = holds.mean()
        tpy = 252 / max(avg_h, 1)
        random_sharpes[perm] = (rets.mean() / rets.std()) * np.sqrt(tpy)

p_value = float((random_sharpes >= reimpl_sharpe).mean())
log(f"Original Sharpe: {reimpl_sharpe:.2f}")
log(f"Random Sharpe: mean={random_sharpes.mean():.2f}, median={np.median(random_sharpes):.2f}, std={random_sharpes.std():.2f}")
log(f"Random p95: {np.percentile(random_sharpes, 95):.2f}")
log(f"p-value: {p_value:.4f}")
test3_pass = p_value < 0.05
log(f"RESULT: {'PASS' if test3_pass else 'FAIL'} — p={p_value:.4f} {'<' if test3_pass else '>='} 0.05")


# ══════════════════════════════════════════════════════════════════════
# TEST 4: SUB-PERIOD STABILITY
# ══════════════════════════════════════════════════════════════════════
log("\n" + "="*70)
log("TEST 4: SUB-PERIOD STABILITY")
log("="*70)

quarter = n_days // 4
period_bounds = [
    (0, quarter - 1),
    (quarter, 2 * quarter - 1),
    (2 * quarter, 3 * quarter - 1),
    (3 * quarter, n_days - 1)
]

period_sharpes = []
for pi, (p_start, p_end) in enumerate(period_bounds):
    period_trades = [t for t in baseline_trades if p_start <= t['entry_idx'] <= p_end]
    if len(period_trades) >= 2:
        ps = compute_sharpe(period_trades)
    elif len(period_trades) == 1:
        ps = period_trades[0]['return'] / 0.01 if period_trades[0]['return'] != 0 else 0
    else:
        ps = 0.0
    period_sharpes.append(ps)
    sd = pd.Timestamp(dates_arr[p_start]).date()
    ed = pd.Timestamp(dates_arr[p_end]).date()
    log(f"Period {pi+1} ({sd} - {ed}): {len(period_trades)} trades, Sharpe={ps:.2f}")

n_positive = sum(1 for s in period_sharpes if s > 0)
n_bad = sum(1 for s in period_sharpes if s < -0.50)
test4_pass = n_positive >= 3 and n_bad == 0
log(f"Positive periods: {n_positive}/4, Periods < -0.50: {n_bad}")
log(f"RESULT: {'PASS' if test4_pass else 'FAIL'} — {n_positive}>=3 positive AND {n_bad}==0 below -0.50")


# ══════════════════════════════════════════════════════════════════════
# TEST 5: TOP-3 STOCK REMOVAL
# ══════════════════════════════════════════════════════════════════════
log("\n" + "="*70)
log("TEST 5: TOP-3 STOCK REMOVAL")
log("="*70)

# Compute per-stock contribution
stock_contributions = np.zeros(n_stocks)
for t in baseline_trades:
    entry_p = price_matrix[t['entry_idx']]
    exit_p = price_matrix[t['exit_idx']]
    for s in range(n_stocks):
        ep, xp = entry_p[s], exit_p[s]
        if not np.isnan(ep) and not np.isnan(xp) and ep > 0:
            stock_contributions[s] += (xp / ep - 1) / n_stocks

sorted_idx = np.argsort(stock_contributions)[::-1]
top3_idx = sorted_idx[:3]
top3_names = [BASKET[i] for i in top3_idx]
log(f"Top 3 contributors: {top3_names}")
for i in sorted_idx[:5]:
    log(f"  {BASKET[i]}: {stock_contributions[i]:+.4f}")

# Create mask excluding top 3
stock_mask = np.ones(n_stocks, dtype=bool)
stock_mask[top3_idx] = False
reduced_basket_names = [BASKET[i] for i in range(n_stocks) if stock_mask[i]]
log(f"Reduced basket ({sum(stock_mask)} stocks)")

reduced_trades = run_strategy_fast(vix_arr, price_matrix, stock_mask=stock_mask)
reduced_metrics = compute_metrics(reduced_trades)

log(f"Reduced Sharpe: {reduced_metrics['sharpe']:.2f}")
test5_threshold = 0.50 * reimpl_sharpe
test5_pass = reduced_metrics['sharpe'] > test5_threshold
log(f"RESULT: {'PASS' if test5_pass else 'FAIL'} — {reduced_metrics['sharpe']:.2f} {'>' if test5_pass else '<='} {test5_threshold:.2f} (50% of original)")


# ══════════════════════════════════════════════════════════════════════
# TEST 6: PARAMETER SENSITIVITY (200 random combos)
# ══════════════════════════════════════════════════════════════════════
log("\n" + "="*70)
log("TEST 6: PARAMETER SENSITIVITY (200 random combos)")
log("="*70)

n_combos = 200
param_sharpes = np.zeros(n_combos)

for c in range(n_combos):
    vm = np.random.uniform(1.05, 1.30)
    vl = np.random.randint(30, 91)
    em = np.random.uniform(0.90, 1.05)
    mh = np.random.randint(10, 31)
    tp = np.random.uniform(0.05, 0.15)
    sl_val = np.random.uniform(-0.20, -0.10)

    try:
        trades_i = run_strategy_fast(vix_arr, price_matrix,
                                     vix_mult=vm, vix_lookback=vl,
                                     max_hold=mh, tp=tp, sl=sl_val,
                                     exit_mult=em)
        if len(trades_i) >= 2:
            param_sharpes[c] = compute_sharpe(trades_i)
    except:
        pass

    if (c + 1) % 50 == 0:
        log(f"  {c+1}/{n_combos} combos done...")

frac_above = float((param_sharpes > 0.30).mean())
log(f"Parameter variations: {n_combos}")
log(f"Sharpe distribution: mean={param_sharpes.mean():.2f}, median={np.median(param_sharpes):.2f}, std={param_sharpes.std():.2f}")
log(f"Fraction with Sharpe > 0.30: {frac_above:.1%}")
log(f"p10/p25/p50/p75/p90: {np.percentile(param_sharpes,10):.2f} / "
    f"{np.percentile(param_sharpes,25):.2f} / {np.percentile(param_sharpes,50):.2f} / "
    f"{np.percentile(param_sharpes,75):.2f} / {np.percentile(param_sharpes,90):.2f}")
test6_pass = frac_above > 0.50
log(f"RESULT: {'PASS' if test6_pass else 'FAIL'} — {frac_above:.1%} {'>' if test6_pass else '<='} 50%")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
log("\n" + "="*70)
log("ADVERSARIAL VALIDATION SUMMARY: Strategy F — Vol Term Structure")
log("="*70)

results = [
    ("Test 1: Re-implementation", test1_pass, f"Sharpe {reimpl_sharpe:.2f} vs threshold {test1_threshold:.2f}"),
    ("Test 2: Inverse Signal", test2_pass, f"Inverse Sharpe {inverse_metrics['sharpe']:.2f} vs threshold {test2_threshold:.2f}"),
    ("Test 3: Random Timing", test3_pass, f"p={p_value:.4f}"),
    ("Test 4: Sub-period Stability", test4_pass, f"{n_positive}/4 positive, {n_bad} below -0.50"),
    ("Test 5: Top-3 Stock Removal", test5_pass, f"Reduced Sharpe {reduced_metrics['sharpe']:.2f} vs threshold {test5_threshold:.2f}"),
    ("Test 6: Parameter Sensitivity", test6_pass, f"{frac_above:.1%} above 0.30"),
]

n_pass = 0
for name, passed, detail in results:
    status = "PASS" if passed else "FAIL"
    log(f"  {status}  {name}: {detail}")
    if passed:
        n_pass += 1

log(f"\nOVERALL: {n_pass}/6 tests passed")
if n_pass >= 5:
    log("VERDICT: STRONG — strategy passes adversarial validation")
elif n_pass >= 4:
    log("VERDICT: CONDITIONAL — mostly passes but review failures")
elif n_pass >= 3:
    log("VERDICT: WEAK — significant concerns")
else:
    log("VERDICT: REJECT — strategy fails adversarial validation")

log(f"\nCAVEAT: Only {baseline_metrics['n_trades']} trades — all metrics are noisy estimates.")
log("Statistical confidence is inherently limited with this sample size.")
