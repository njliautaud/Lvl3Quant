#!/usr/bin/env python3
"""
Strategy F: Volatility Risk Premium Harvest — 6-Test Adversarial Validation

Signal: VIX > SPY 20-day realized vol by 5+ points
Entry: Equal-weight basket of 20 quality stocks, $300 per entry, max 2 concurrent
Exit: VIX-RV gap < 2 OR 21-day max hold OR +10% TP OR -15% SL
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASKET = ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','JPM','UNH','LLY','AVGO',
          'AMD','HD','ABBV','MRK','COST','CRM','NFLX','ADBE','PG','JNJ']
START = '2019-06-01'  # extra buffer for 200d SMA
END = '2026-07-01'
ANALYSIS_START = '2020-01-01'
ENTRY_AMT = 300.0
MAX_CONCURRENT = 2
BASELINE_SHARPE = 1.93

# ── DATA DOWNLOAD ──────────────────────────────────────────────────────────
print("Downloading data...")
tickers = BASKET + ['^VIX', '^GSPC']
data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
prices = data['Close'].copy()
prices.columns = [c if isinstance(c, str) else c for c in prices.columns]

# Rename index columns
vix = prices['^VIX'].dropna()
spy = prices['^GSPC'].dropna()

# Align dates
common_idx = vix.index.intersection(spy.index)
for t in BASKET:
    common_idx = common_idx.intersection(prices[t].dropna().index)
common_idx = common_idx.sort_values()

vix = vix.loc[common_idx]
spy = spy.loc[common_idx]
stock_prices = prices[BASKET].loc[common_idx].copy()

print(f"Data: {common_idx[0].date()} to {common_idx[-1].date()}, {len(common_idx)} days")


def compute_realized_vol(series, window=20):
    """Annualized realized volatility from log returns."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252) * 100  # as percentage points


def run_backtest(stock_px, vix_s, spy_s, dates,
                 gap_entry=5.0, rv_lookback=20, gap_exit=2.0,
                 max_hold=21, tp_pct=0.10, sl_pct=-0.15,
                 entry_amt=300.0, max_concurrent=2,
                 signal_dates_override=None,
                 excluded_stocks=None,
                 basket=None):
    """
    Core backtest engine.
    Returns dict with trades list, daily returns, sharpe, etc.
    """
    if basket is None:
        basket = BASKET
    if excluded_stocks:
        basket = [s for s in basket if s not in excluded_stocks]

    n_stocks = len(basket)
    per_stock_amt = entry_amt / n_stocks

    rv = compute_realized_vol(spy_s, window=rv_lookback)
    gap = vix_s - rv

    # Filter to analysis period
    mask = dates >= pd.Timestamp(ANALYSIS_START)
    analysis_dates = dates[mask]

    trades = []
    positions = []  # list of active positions: (entry_date, entry_prices_dict, entry_idx)
    daily_pnl = pd.Series(0.0, index=analysis_dates)

    for i, dt in enumerate(analysis_dates):
        dt_loc = dates.get_loc(dt)

        # Check exits for active positions
        new_positions = []
        for pos in positions:
            entry_dt, entry_px, entry_idx = pos
            days_held = (dt - entry_dt).days

            # Compute current basket value
            current_val = 0
            entry_val = 0
            for s in basket:
                if s in entry_px:
                    shares = per_stock_amt / entry_px[s]
                    current_val += shares * stock_px[s].iloc[dt_loc]
                    entry_val += per_stock_amt

            pct_return = (current_val - entry_val) / entry_val if entry_val > 0 else 0
            current_gap = gap.iloc[dt_loc] if not np.isnan(gap.iloc[dt_loc]) else 999

            exit_reason = None
            if current_gap < gap_exit:
                exit_reason = 'gap_close'
            elif days_held >= max_hold:
                exit_reason = 'max_hold'
            elif pct_return >= tp_pct:
                exit_reason = 'tp'
            elif pct_return <= sl_pct:
                exit_reason = 'sl'

            if exit_reason:
                trades.append({
                    'entry_date': entry_dt,
                    'exit_date': dt,
                    'pct_return': pct_return,
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                    'pnl_dollar': pct_return * entry_val,
                    'per_stock_pnl': {}
                })
                # Record per-stock P&L for attribution
                for s in basket:
                    if s in entry_px:
                        shares = per_stock_amt / entry_px[s]
                        s_pnl = shares * (stock_px[s].iloc[dt_loc] - entry_px[s])
                        trades[-1]['per_stock_pnl'][s] = s_pnl
            else:
                new_positions.append(pos)

        positions = new_positions

        # Check entry signal
        if signal_dates_override is not None:
            signal_fire = dt in signal_dates_override
        else:
            current_gap_val = gap.iloc[dt_loc] if dt_loc < len(gap) else np.nan
            signal_fire = (not np.isnan(current_gap_val)) and (current_gap_val >= gap_entry)

        if signal_fire and len(positions) < max_concurrent:
            entry_px_dict = {}
            for s in basket:
                px = stock_px[s].iloc[dt_loc]
                if not np.isnan(px) and px > 0:
                    entry_px_dict[s] = px
            if len(entry_px_dict) > 0:
                positions.append((dt, entry_px_dict, dt_loc))

        # Mark-to-market daily P&L
        total_mtm = 0
        for pos in positions:
            entry_dt, entry_px, entry_idx = pos
            for s in basket:
                if s in entry_px:
                    shares = per_stock_amt / entry_px[s]
                    if dt_loc > 0:
                        prev_loc = dt_loc - 1
                        # Only count today's change
                        today_px = stock_px[s].iloc[dt_loc]
                        yest_px = stock_px[s].iloc[prev_loc] if prev_loc >= entry_idx else entry_px[s]
                        total_mtm += shares * (today_px - yest_px)

        daily_pnl.loc[dt] = total_mtm

    # Close any remaining positions at end
    for pos in positions:
        entry_dt, entry_px, entry_idx = pos
        dt = analysis_dates[-1]
        dt_loc = dates.get_loc(dt)
        current_val = 0
        entry_val = 0
        for s in basket:
            if s in entry_px:
                shares = per_stock_amt / entry_px[s]
                current_val += shares * stock_px[s].iloc[dt_loc]
                entry_val += per_stock_amt
        pct_return = (current_val - entry_val) / entry_val if entry_val > 0 else 0
        trades.append({
            'entry_date': entry_dt,
            'exit_date': dt,
            'pct_return': pct_return,
            'days_held': (dt - entry_dt).days,
            'exit_reason': 'end',
            'pnl_dollar': pct_return * entry_val,
            'per_stock_pnl': {}
        })

    # Compute metrics
    if len(trades) == 0:
        return {'sharpe': 0, 'trades': 0, 'wr': 0, 'pf': 0, 'daily_pnl': daily_pnl,
                'trade_list': [], 'total_pnl': 0}

    trade_returns = [t['pct_return'] for t in trades]
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    wr = len(wins) / len(trade_returns) * 100 if trade_returns else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    # Sharpe from daily P&L
    daily_ret = daily_pnl / entry_amt  # normalize by position size
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    total_pnl = sum(t['pnl_dollar'] for t in trades)

    return {
        'sharpe': sharpe,
        'trades': len(trades),
        'wr': wr,
        'pf': pf,
        'daily_pnl': daily_pnl,
        'trade_list': trades,
        'total_pnl': total_pnl,
        'trade_returns': trade_returns
    }


# ── BASELINE RUN ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BASELINE RUN")
print("="*70)
baseline = run_backtest(stock_prices, vix, spy, common_idx)
print(f"  Sharpe: {baseline['sharpe']:.2f}")
print(f"  Trades: {baseline['trades']}")
print(f"  WR: {baseline['wr']:.1f}%")
print(f"  PF: {baseline['pf']:.2f}")
print(f"  Total P&L: ${baseline['total_pnl']:.2f}")

results = {}

# ── TEST 1: RE-IMPLEMENTATION ──────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 1: RE-IMPLEMENTATION (independent build)")
print("="*70)

# This IS the re-implementation (built from scratch above)
reimpl_sharpe = baseline['sharpe']
threshold_1 = 0.70 * BASELINE_SHARPE
pass_1 = reimpl_sharpe > threshold_1
print(f"  Re-impl Sharpe: {reimpl_sharpe:.2f}")
print(f"  Threshold (70% of {BASELINE_SHARPE}): {threshold_1:.2f}")
print(f"  RESULT: {'PASS' if pass_1 else 'FAIL'}")
results['test1'] = {'pass': pass_1, 'sharpe': reimpl_sharpe, 'threshold': threshold_1}


# ── TEST 2: INVERSE SIGNAL ────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 2: INVERSE SIGNAL (buy when VIX BELOW realized vol by 5+)")
print("="*70)

rv_20 = compute_realized_vol(spy, window=20)
inv_gap = rv_20 - vix  # inverted: realized vol > VIX

# Find dates where realized vol exceeds VIX by 5+
analysis_mask = common_idx >= pd.Timestamp(ANALYSIS_START)
analysis_dates = common_idx[analysis_mask]
inverse_signal_dates = set()
for dt in analysis_dates:
    loc = common_idx.get_loc(dt)
    g = inv_gap.iloc[loc]
    if not np.isnan(g) and g >= 5.0:
        inverse_signal_dates.add(dt)

inverse = run_backtest(stock_prices, vix, spy, common_idx,
                       signal_dates_override=inverse_signal_dates)

inv_sharpe = inverse['sharpe']
threshold_2 = 0.50 * reimpl_sharpe
pass_2 = inv_sharpe < threshold_2
print(f"  Inverse signal dates found: {len(inverse_signal_dates)}")
print(f"  Inverse Sharpe: {inv_sharpe:.2f}")
print(f"  Original Sharpe: {reimpl_sharpe:.2f}")
print(f"  Threshold (50% of original): {threshold_2:.2f}")
print(f"  Inverse trades: {inverse['trades']}")
print(f"  RESULT: {'PASS' if pass_2 else 'FAIL'}")
results['test2'] = {'pass': pass_2, 'sharpe': inv_sharpe, 'threshold': threshold_2}


# ── TEST 3: RANDOM TIMING (200 permutations) ─────────────────────────────
print("\n" + "="*70)
print("TEST 3: RANDOM TIMING (200 permutations)")
print("="*70)

n_orig_trades = baseline['trades']
np.random.seed(42)
random_sharpes = []

for perm in range(200):
    # Random dates from analysis period
    rand_indices = np.random.choice(len(analysis_dates), size=min(n_orig_trades, len(analysis_dates)), replace=False)
    rand_dates = set(analysis_dates[rand_indices])

    r = run_backtest(stock_prices, vix, spy, common_idx,
                     signal_dates_override=rand_dates,
                     gap_exit=2.0)  # still use gap exit
    random_sharpes.append(r['sharpe'])

    if (perm + 1) % 50 == 0:
        print(f"  ... completed {perm+1}/200 permutations")

random_sharpes = np.array(random_sharpes)
p_value = np.mean(random_sharpes >= reimpl_sharpe)
pass_3 = p_value < 0.05
print(f"  Original Sharpe: {reimpl_sharpe:.2f}")
print(f"  Random Sharpe mean: {np.mean(random_sharpes):.2f} (std: {np.std(random_sharpes):.2f})")
print(f"  Random Sharpe max: {np.max(random_sharpes):.2f}")
print(f"  p-value: {p_value:.4f}")
print(f"  RESULT: {'PASS' if pass_3 else 'FAIL'}")
results['test3'] = {'pass': pass_3, 'p_value': p_value}


# ── TEST 4: SUB-PERIOD STABILITY ──────────────────────────────────────────
print("\n" + "="*70)
print("TEST 4: SUB-PERIOD STABILITY (4 equal periods)")
print("="*70)

period_start = pd.Timestamp(ANALYSIS_START)
period_end = pd.Timestamp('2026-07-01')
total_days = (period_end - period_start).days
quarter_days = total_days // 4

sub_sharpes = []
for q in range(4):
    q_start = period_start + timedelta(days=q * quarter_days)
    q_end = period_start + timedelta(days=(q+1) * quarter_days) if q < 3 else period_end

    # Mask data to this sub-period but keep full data for RV calculation
    sub_mask = (common_idx >= q_start) & (common_idx < q_end)
    sub_dates = common_idx[sub_mask]

    if len(sub_dates) < 20:
        sub_sharpes.append(0)
        continue

    # Run backtest but only allow entries in this sub-period
    sub_signal_dates = set()
    rv_sub = compute_realized_vol(spy, window=20)
    gap_sub = vix - rv_sub
    for dt in sub_dates:
        loc = common_idx.get_loc(dt)
        g = gap_sub.iloc[loc]
        if not np.isnan(g) and g >= 5.0:
            sub_signal_dates.add(dt)

    # Use a modified approach: restrict analysis to sub-period
    sub_result = run_backtest(stock_prices, vix, spy, common_idx,
                              signal_dates_override=sub_signal_dates)

    # Compute Sharpe only on sub-period daily P&L
    sub_daily = sub_result['daily_pnl'].loc[sub_dates[0]:sub_dates[-1]]
    sub_ret = sub_daily / ENTRY_AMT
    s = (sub_ret.mean() / sub_ret.std() * np.sqrt(252)) if sub_ret.std() > 0 else 0
    sub_sharpes.append(s)

    print(f"  Period {q+1} ({q_start.date()} to {q_end.date()}): Sharpe={s:.2f}, "
          f"Signals={len(sub_signal_dates)}, Trades={sub_result['trades']}")

n_positive = sum(1 for s in sub_sharpes if s > 0)
n_below_neg05 = sum(1 for s in sub_sharpes if s < -0.50)
pass_4 = (n_positive >= 3) and (n_below_neg05 == 0)
print(f"  Positive periods: {n_positive}/4")
print(f"  Periods below -0.50: {n_below_neg05}")
print(f"  RESULT: {'PASS' if pass_4 else 'FAIL'}")
results['test4'] = {'pass': pass_4, 'sub_sharpes': sub_sharpes}


# ── TEST 5: TOP-3 STOCK REMOVAL ──────────────────────────────────────────
print("\n" + "="*70)
print("TEST 5: TOP-3 STOCK REMOVAL")
print("="*70)

# Compute per-stock P&L attribution from baseline trades
stock_pnl = {s: 0.0 for s in BASKET}
for t in baseline['trade_list']:
    for s, pnl in t.get('per_stock_pnl', {}).items():
        if s in stock_pnl:
            stock_pnl[s] += pnl

# Sort by contribution
sorted_stocks = sorted(stock_pnl.items(), key=lambda x: x[1], reverse=True)
top3 = [s[0] for s in sorted_stocks[:3]]
print(f"  Top 3 contributors: {top3}")
for s, pnl in sorted_stocks[:5]:
    print(f"    {s}: ${pnl:.2f}")

# Rerun without top 3
reduced = run_backtest(stock_prices, vix, spy, common_idx, excluded_stocks=top3)
threshold_5 = 0.50 * reimpl_sharpe
pass_5 = reduced['sharpe'] > threshold_5
print(f"  Reduced basket Sharpe: {reduced['sharpe']:.2f}")
print(f"  Threshold (50% of {reimpl_sharpe:.2f}): {threshold_5:.2f}")
print(f"  RESULT: {'PASS' if pass_5 else 'FAIL'}")
results['test5'] = {'pass': pass_5, 'sharpe': reduced['sharpe'], 'removed': top3}


# ── TEST 6: PARAMETER SENSITIVITY (100 random combos) ────────────────────
print("\n" + "="*70)
print("TEST 6: PARAMETER SENSITIVITY (100 random combos)")
print("="*70)

np.random.seed(123)
param_sharpes = []

for combo in range(100):
    gap_entry = np.random.uniform(3, 8)
    rv_lookback = int(np.random.uniform(10, 30))
    gap_exit = np.random.uniform(0, 4)
    max_hold = int(np.random.uniform(10, 30))
    tp = np.random.uniform(0.05, 0.15)
    sl = np.random.uniform(-0.20, -0.10)

    r = run_backtest(stock_prices, vix, spy, common_idx,
                     gap_entry=gap_entry, rv_lookback=rv_lookback,
                     gap_exit=gap_exit, max_hold=max_hold,
                     tp_pct=tp, sl_pct=sl)
    param_sharpes.append(r['sharpe'])

    if (combo + 1) % 25 == 0:
        print(f"  ... completed {combo+1}/100 combos")

param_sharpes = np.array(param_sharpes)
frac_above_030 = np.mean(param_sharpes > 0.30)
pass_6 = frac_above_030 > 0.50
print(f"  Combos with Sharpe > 0.30: {frac_above_030*100:.1f}%")
print(f"  Mean param Sharpe: {np.mean(param_sharpes):.2f}")
print(f"  Median param Sharpe: {np.median(param_sharpes):.2f}")
print(f"  Threshold: >50% above 0.30")
print(f"  RESULT: {'PASS' if pass_6 else 'FAIL'}")
results['test6'] = {'pass': pass_6, 'frac_above': frac_above_030}


# ── FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL ADVERSARIAL VALIDATION SUMMARY")
print("="*70)
print(f"Strategy F: Volatility Risk Premium Harvest")
print(f"{'='*70}")

test_names = {
    'test1': 'Re-Implementation',
    'test2': 'Inverse Signal',
    'test3': 'Random Timing',
    'test4': 'Sub-Period Stability',
    'test5': 'Top-3 Stock Removal',
    'test6': 'Parameter Sensitivity'
}

n_pass = 0
for key in ['test1','test2','test3','test4','test5','test6']:
    status = 'PASS' if results[key]['pass'] else 'FAIL'
    if results[key]['pass']:
        n_pass += 1

    detail = ''
    if key == 'test1':
        detail = f"Sharpe={results[key]['sharpe']:.2f} vs threshold={results[key]['threshold']:.2f}"
    elif key == 'test2':
        detail = f"Inverse Sharpe={results[key]['sharpe']:.2f} vs threshold={results[key]['threshold']:.2f}"
    elif key == 'test3':
        detail = f"p-value={results[key]['p_value']:.4f}"
    elif key == 'test4':
        detail = f"Sub-Sharpes={[f'{s:.2f}' for s in results[key]['sub_sharpes']]}"
    elif key == 'test5':
        detail = f"Reduced Sharpe={results[key]['sharpe']:.2f}, removed={results[key]['removed']}"
    elif key == 'test6':
        detail = f"{results[key]['frac_above']*100:.1f}% combos above 0.30"

    print(f"  Test {key[-1]}: {test_names[key]:25s} [{status}]  {detail}")

print(f"\n  TOTAL: {n_pass}/6 PASSED")
if n_pass >= 5:
    print(f"  >>> STRATEGY F IS VALIDATED <<<")
else:
    print(f"  >>> STRATEGY F FAILS VALIDATION (need 5/6) <<<")
