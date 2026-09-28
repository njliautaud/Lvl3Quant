#!/usr/bin/env python3
"""
Adversarial Validation: Multi-Timeframe F — Cascading Rate of Change
6 tests: inverse signal, random timing, sub-period stability,
         remove top-3 tickers, parameter sensitivity, cost sensitivity
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
TICKERS = [
    'AAPL','MSFT','AVGO','JPM','JNJ','PG','KO','PEP','HD','COST',
    'UNH','LLY','V','MA','ABBV','MRK','WMT','AMZN','GOOGL','META'
]
LOOKBACK_START = '2021-06-01'
TRADE_START    = '2022-01-01'
TRADE_END      = '2026-07-31'
CAPITAL        = 669.0
MAX_POS_SIZE   = 200.0
MAX_CONCURRENT = 3
DEFAULT_SLIP   = 0.0002  # 2 bps
HOLD_DAYS      = 10
ROC5_THR       = -5.0
ROC10_THR      = -8.0
ROC20_THR      = -10.0

OUTPUT_PATH = Path('/home/jupiter/Lvl3Quant/data/multi_tf_f_adversarial.json')

# ── Data Download ───────────────────────────────────────────────────────
print("Downloading price data...")
raw = yf.download(TICKERS, start=LOOKBACK_START, end=TRADE_END,
                  auto_adjust=True, progress=False)
close = raw['Close'].dropna(how='all')
close = close.ffill()
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days, {len(close.columns)} tickers")


# ── Backtest Engine ─────────────────────────────────────────────────────
def compute_roc(prices, period):
    return (prices / prices.shift(period) - 1) * 100

def backtest(prices, roc5_thr, roc10_thr, roc20_thr, hold_days,
             slippage_bps=2, start_date=TRADE_START, end_date=TRADE_END,
             tickers=None, inverse=False):
    """
    Run the cascading ROC strategy.
    If inverse=True, buy when ROC > +threshold (overbought).
    Returns dict with metrics and per-trade details.
    """
    if tickers is not None:
        prices = prices[[t for t in tickers if t in prices.columns]]

    roc5  = compute_roc(prices, 5)
    roc10 = compute_roc(prices, 10)
    roc20 = compute_roc(prices, 20)

    mask = (prices.index >= pd.Timestamp(start_date)) & (prices.index <= pd.Timestamp(end_date))
    trade_dates = prices.index[mask]

    slip = slippage_bps / 10000.0
    trades = []
    open_positions = []  # list of (ticker, entry_date, entry_price, shares, exit_idx)

    date_list = list(trade_dates)

    for i, date in enumerate(date_list):
        # Close expired positions
        new_open = []
        for pos in open_positions:
            tk, edate, eprice, shares, exit_i = pos
            if i >= exit_i:
                # Find actual exit date
                actual_exit_i = min(exit_i, len(date_list) - 1)
                exit_date = date_list[actual_exit_i]
                exit_price = prices.loc[exit_date, tk] * (1 - slip)
                pnl = (exit_price - eprice) * shares
                trades.append({
                    'ticker': tk, 'entry': str(edate.date()),
                    'exit': str(exit_date.date()), 'pnl': float(pnl),
                    'entry_price': float(eprice), 'exit_price': float(exit_price)
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        # Check for new entries
        if len(open_positions) >= MAX_CONCURRENT:
            continue

        for tk in prices.columns:
            if len(open_positions) >= MAX_CONCURRENT:
                break
            # Skip if already holding this ticker
            if any(p[0] == tk for p in open_positions):
                continue

            r5  = roc5.loc[date, tk]  if date in roc5.index  else np.nan
            r10 = roc10.loc[date, tk] if date in roc10.index else np.nan
            r20 = roc20.loc[date, tk] if date in roc20.index else np.nan

            if pd.isna(r5) or pd.isna(r10) or pd.isna(r20):
                continue

            if inverse:
                signal = (r5 > -roc5_thr) and (r10 > -roc10_thr) and (r20 > -roc20_thr)
            else:
                signal = (r5 < roc5_thr) and (r10 < roc10_thr) and (r20 < roc20_thr)

            if signal:
                entry_price = prices.loc[date, tk] * (1 + slip)
                shares = int(MAX_POS_SIZE / entry_price) if entry_price > 0 else 0
                if shares < 1:
                    shares = 1
                cost = shares * entry_price
                if cost > MAX_POS_SIZE * 1.5:
                    continue
                exit_i = i + hold_days
                open_positions.append((tk, date, entry_price, shares, exit_i))

    # Close remaining positions at last date
    last_date = date_list[-1]
    for pos in open_positions:
        tk, edate, eprice, shares, exit_i = pos
        exit_price = prices.loc[last_date, tk] * (1 - slip)
        pnl = (exit_price - eprice) * shares
        trades.append({
            'ticker': tk, 'entry': str(edate.date()),
            'exit': str(last_date.date()), 'pnl': float(pnl),
            'entry_price': float(eprice), 'exit_price': float(exit_price)
        })

    return compute_metrics(trades)


def compute_metrics(trades):
    if not trades:
        return {'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
                'max_dd_pct': 0, 'n_trades': 0, 'total_pnl': 0, 'trades': []}

    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    wr = len(wins) / n * 100 if n > 0 else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    # Sharpe / Sortino (annualized, assuming ~25 trades/year avg)
    arr = np.array(pnls)
    mu = arr.mean()
    std = arr.std(ddof=1) if n > 1 else 1e-9
    downside = arr[arr < 0]
    down_std = downside.std(ddof=1) if len(downside) > 1 else 1e-9

    trades_per_year = max(n / 4.5, 1)  # ~4.5 year period
    ann_factor = np.sqrt(trades_per_year)
    sharpe = (mu / std) * ann_factor if std > 1e-9 else 0
    sortino = (mu / down_std) * ann_factor if down_std > 1e-9 else 0

    # Max drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = dd.min()
    max_dd_pct = (max_dd / CAPITAL) * 100 if CAPITAL > 0 else 0

    return {
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'win_rate': round(float(wr), 1),
        'profit_factor': round(float(pf), 3),
        'max_dd_pct': round(float(max_dd_pct), 2),
        'n_trades': n,
        'total_pnl': round(float(total_pnl), 2),
        'trades': trades
    }


# ── Run Baseline ────────────────────────────────────────────────────────
print("\n=== BASELINE ===")
baseline = backtest(close, ROC5_THR, ROC10_THR, ROC20_THR, HOLD_DAYS)
print(f"Sharpe={baseline['sharpe']}, Sortino={baseline['sortino']}, "
      f"WR={baseline['win_rate']}%, PF={baseline['profit_factor']}, "
      f"Trades={baseline['n_trades']}, PnL=${baseline['total_pnl']:.2f}")

results = {
    'strategy': 'Multi-Timeframe F — Cascading Rate of Change',
    'timestamp': datetime.now().isoformat(),
    'baseline': {k: v for k, v in baseline.items() if k != 'trades'},
    'tests': {}
}

# ── Test 1: Inverse Signal ──────────────────────────────────────────────
print("\n=== TEST 1: Inverse Signal (Cascading Overbought) ===")
inverse = backtest(close, ROC5_THR, ROC10_THR, ROC20_THR, HOLD_DAYS, inverse=True)
inv_ratio = inverse['sharpe'] / baseline['sharpe'] if baseline['sharpe'] != 0 else 999
passed_1 = inv_ratio < 0.50
print(f"Inverse Sharpe={inverse['sharpe']}, Ratio={inv_ratio:.3f}, "
      f"Trades={inverse['n_trades']}, PASS={passed_1}")

results['tests']['1_inverse_signal'] = {
    'inverse_sharpe': inverse['sharpe'],
    'baseline_sharpe': baseline['sharpe'],
    'ratio': round(inv_ratio, 3),
    'inverse_trades': inverse['n_trades'],
    'passed': passed_1,
    'criterion': 'inverse_sharpe < 50% of baseline'
}

# ── Test 2: Random Timing Percentile ────────────────────────────────────
print("\n=== TEST 2: Random Timing (1000 permutations) ===")
np.random.seed(42)
baseline_trades_list = baseline['trades']
n_baseline = baseline['n_trades']

# Get all valid trading dates
trade_mask = (close.index >= pd.Timestamp(TRADE_START)) & (close.index <= pd.Timestamp(TRADE_END))
all_trade_dates = close.index[trade_mask]

random_sharpes = []
for perm in range(1000):
    if perm % 200 == 0:
        print(f"  Permutation {perm}/1000...")
    # Random entries: pick n_baseline random (date, ticker) pairs
    rand_trades = []
    rand_dates = np.random.choice(len(all_trade_dates) - HOLD_DAYS - 1,
                                   size=min(n_baseline, len(all_trade_dates) - HOLD_DAYS - 1),
                                   replace=False)
    for idx in rand_dates:
        entry_date = all_trade_dates[idx]
        tk = np.random.choice(TICKERS)
        if tk not in close.columns:
            continue
        entry_price = close.loc[entry_date, tk] * (1 + DEFAULT_SLIP)
        exit_idx = min(idx + HOLD_DAYS, len(all_trade_dates) - 1)
        exit_date = all_trade_dates[exit_idx]
        exit_price = close.loc[exit_date, tk] * (1 - DEFAULT_SLIP)
        shares = max(1, int(MAX_POS_SIZE / entry_price)) if entry_price > 0 else 1
        pnl = (exit_price - entry_price) * shares
        rand_trades.append({'pnl': float(pnl)})

    rm = compute_metrics(rand_trades)
    random_sharpes.append(rm['sharpe'])

random_sharpes = np.array(random_sharpes)
p_value = np.mean(random_sharpes >= baseline['sharpe'])
passed_2 = p_value < 0.05
print(f"Baseline Sharpe={baseline['sharpe']}, p-value={p_value:.4f}, "
      f"Random mean={random_sharpes.mean():.3f}, PASS={passed_2}")

results['tests']['2_random_timing'] = {
    'p_value': round(float(p_value), 4),
    'baseline_sharpe': baseline['sharpe'],
    'random_mean_sharpe': round(float(random_sharpes.mean()), 3),
    'random_median_sharpe': round(float(np.median(random_sharpes)), 3),
    'random_p95_sharpe': round(float(np.percentile(random_sharpes, 95)), 3),
    'passed': passed_2,
    'criterion': 'p < 0.05'
}

# ── Test 3: Sub-Period Stability ────────────────────────────────────────
print("\n=== TEST 3: Sub-Period Stability (4 periods) ===")
start_ts = pd.Timestamp(TRADE_START)
end_ts = pd.Timestamp(TRADE_END)
total_days = (end_ts - start_ts).days
quarter = total_days // 4

sub_results = []
all_positive = True
for q in range(4):
    s = start_ts + timedelta(days=quarter * q)
    e = start_ts + timedelta(days=quarter * (q + 1)) if q < 3 else end_ts
    sub = backtest(close, ROC5_THR, ROC10_THR, ROC20_THR, HOLD_DAYS,
                   start_date=str(s.date()), end_date=str(e.date()))
    sub_results.append({
        'period': f"{s.date()} to {e.date()}",
        'sharpe': sub['sharpe'],
        'n_trades': sub['n_trades'],
        'total_pnl': sub['total_pnl'],
        'win_rate': sub['win_rate']
    })
    print(f"  Q{q+1} ({s.date()} to {e.date()}): Sharpe={sub['sharpe']}, "
          f"Trades={sub['n_trades']}, PnL=${sub['total_pnl']:.2f}")
    if sub['sharpe'] <= 0:
        all_positive = False

passed_3 = all_positive
print(f"All positive Sharpe: {passed_3}")

results['tests']['3_sub_period_stability'] = {
    'sub_periods': sub_results,
    'all_positive_sharpe': all_positive,
    'passed': passed_3,
    'criterion': 'all 4 sub-periods have positive Sharpe'
}

# ── Test 4: Remove Top-3 Tickers ────────────────────────────────────────
print("\n=== TEST 4: Remove Top-3 Tickers by PnL ===")
ticker_pnl = {}
for t in baseline['trades']:
    tk = t['ticker']
    ticker_pnl[tk] = ticker_pnl.get(tk, 0) + t['pnl']

sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
top3 = [t[0] for t in sorted_tickers[:3]]
print(f"Top 3 tickers by PnL: {top3}")
print(f"  PnL contributions: {[(t[0], round(t[1],2)) for t in sorted_tickers[:3]]}")

remaining = [t for t in TICKERS if t not in top3]
reduced = backtest(close, ROC5_THR, ROC10_THR, ROC20_THR, HOLD_DAYS, tickers=remaining)
sharpe_drop = 1 - (reduced['sharpe'] / baseline['sharpe']) if baseline['sharpe'] != 0 else 1
passed_4 = sharpe_drop < 0.50
print(f"Reduced Sharpe={reduced['sharpe']}, Drop={sharpe_drop*100:.1f}%, "
      f"Trades={reduced['n_trades']}, PASS={passed_4}")

results['tests']['4_remove_top3'] = {
    'top3_tickers': top3,
    'top3_pnl': {t[0]: round(t[1], 2) for t in sorted_tickers[:3]},
    'baseline_sharpe': baseline['sharpe'],
    'reduced_sharpe': reduced['sharpe'],
    'sharpe_drop_pct': round(sharpe_drop * 100, 1),
    'reduced_trades': reduced['n_trades'],
    'passed': passed_4,
    'criterion': 'Sharpe drop < 50%'
}

# ── Test 5: Parameter Sensitivity Grid ──────────────────────────────────
print("\n=== TEST 5: Parameter Sensitivity Grid ===")
roc5_vals  = [-3, -5, -7, -10]
roc10_vals = [-5, -8, -10, -12]
roc20_vals = [-8, -10, -12, -15]
hold_vals  = [5, 7, 10, 15]

total_combos = len(roc5_vals) * len(roc10_vals) * len(roc20_vals) * len(hold_vals)
good_combos = 0
combo_count = 0
grid_results = []

for r5 in roc5_vals:
    for r10 in roc10_vals:
        for r20 in roc20_vals:
            for hd in hold_vals:
                combo_count += 1
                if combo_count % 50 == 0:
                    print(f"  Combo {combo_count}/{total_combos}...")
                res = backtest(close, r5, r10, r20, hd)
                if res['sharpe'] > 0.3:
                    good_combos += 1
                grid_results.append({
                    'roc5': r5, 'roc10': r10, 'roc20': r20, 'hold': hd,
                    'sharpe': res['sharpe'], 'n_trades': res['n_trades']
                })

pct_good = good_combos / total_combos * 100
passed_5 = pct_good > 50
print(f"Total combos: {total_combos}, Sharpe>0.3: {good_combos} ({pct_good:.1f}%), PASS={passed_5}")

# Top/bottom combos
grid_sorted = sorted(grid_results, key=lambda x: x['sharpe'], reverse=True)
print(f"  Best:  {grid_sorted[0]}")
print(f"  Worst: {grid_sorted[-1]}")

results['tests']['5_parameter_sensitivity'] = {
    'total_combos': total_combos,
    'combos_sharpe_gt_0_3': good_combos,
    'pct_good': round(pct_good, 1),
    'best_combo': grid_sorted[0],
    'worst_combo': grid_sorted[-1],
    'top5': grid_sorted[:5],
    'passed': passed_5,
    'criterion': '>50% of combos have Sharpe > 0.3'
}

# ── Test 6: Cost Sensitivity ───────────────────────────────────────────
print("\n=== TEST 6: Cost Sensitivity ===")
slip_levels = [0, 2, 5, 10, 20, 50]
cost_results = []
breakeven_bps = 0

for sl in slip_levels:
    res = backtest(close, ROC5_THR, ROC10_THR, ROC20_THR, HOLD_DAYS, slippage_bps=sl)
    cost_results.append({
        'slippage_bps': sl,
        'sharpe': res['sharpe'],
        'total_pnl': res['total_pnl'],
        'n_trades': res['n_trades']
    })
    print(f"  {sl:2d} bps: Sharpe={res['sharpe']}, PnL=${res['total_pnl']:.2f}")
    if res['total_pnl'] > 0:
        breakeven_bps = sl

passed_6 = breakeven_bps > 20
print(f"Breakeven > {breakeven_bps} bps, PASS={passed_6}")

results['tests']['6_cost_sensitivity'] = {
    'levels': cost_results,
    'breakeven_bps': breakeven_bps,
    'passed': passed_6,
    'criterion': 'breakeven > 20 bps'
}

# ── Summary ─────────────────────────────────────────────────────────────
tests_passed = sum(1 for t in results['tests'].values() if t['passed'])
total_tests = len(results['tests'])
results['summary'] = {
    'tests_passed': tests_passed,
    'total_tests': total_tests,
    'pass_rate': f"{tests_passed}/{total_tests}",
    'overall_verdict': 'PASS' if tests_passed >= 5 else 'MARGINAL' if tests_passed >= 4 else 'FAIL'
}

# Strip trade details from baseline to keep JSON manageable
results['baseline'].pop('trades', None)

# Save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"ADVERSARIAL VALIDATION RESULTS: {results['summary']['overall_verdict']}")
print(f"{'='*60}")
print(f"Passed: {tests_passed}/{total_tests}")
for name, test in results['tests'].items():
    status = 'PASS' if test['passed'] else 'FAIL'
    print(f"  [{status}] {name}: {test['criterion']}")
print(f"\nResults saved to {OUTPUT_PATH}")
