#!/usr/bin/env python3
"""
Adversarial Validation: Consecutive Dip B — Deepening Losses Pattern
6 adversarial tests to validate edge is real, not overfit/luck.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────
TICKERS = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]
LOOKBACK_START = '2021-06-01'
TRADE_START = '2022-01-01'
TRADE_END = '2026-07-31'
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
DEFAULT_HOLD = 10
DEFAULT_CONSEC = 3
DEFAULT_DIP_PCT = 5.0
DEFAULT_SLIPPAGE_BPS = 2
BASELINE_SHARPE = 1.418

# ── DATA DOWNLOAD ──────────────────────────────────────────────────────
print("Downloading price data...")
data = yf.download(TICKERS, start=LOOKBACK_START, end=TRADE_END, progress=False, auto_adjust=True)
close = data['Close'].dropna(how='all')
# Forward-fill small gaps (weekends already excluded by yfinance)
close = close.ffill().dropna(how='any')
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days, {len(close.columns)} tickers")


def compute_sharpe(returns_series):
    """Annualized Sharpe from daily returns."""
    if len(returns_series) < 2 or returns_series.std() == 0:
        return 0.0
    return float(returns_series.mean() / returns_series.std() * np.sqrt(252))


def run_strategy(prices, tickers=None, hold_days=DEFAULT_HOLD,
                 min_consec=DEFAULT_CONSEC, dip_pct=DEFAULT_DIP_PCT,
                 slippage_bps=DEFAULT_SLIPPAGE_BPS, require_acceleration=True,
                 inverse=False):
    """
    Run the Consecutive Dip B strategy (or its inverse).

    inverse=False: Buy on 3+ consecutive red days with accelerating losses, >5% below 20d high.
    inverse=True:  Buy on 3+ consecutive green days with accelerating gains, above 20d high.

    Returns: dict with sharpe, trades, daily_returns, pnl_by_ticker, trade_list
    """
    if tickers is None:
        tickers = prices.columns.tolist()
    prices = prices[[t for t in tickers if t in prices.columns]].copy()

    trade_start_dt = pd.Timestamp(TRADE_START)
    trade_dates = prices.index[prices.index >= trade_start_dt]

    daily_pnl = pd.Series(0.0, index=trade_dates)
    active_trades = []  # list of dicts: {ticker, entry_date, entry_price, shares, exit_date_idx}
    trade_list = []
    pnl_by_ticker = defaultdict(float)

    slippage_mult = 1.0 + slippage_bps / 10000.0
    slippage_sell = 1.0 - slippage_bps / 10000.0

    for i, date in enumerate(trade_dates):
        date_idx = prices.index.get_loc(date)

        # Close expired trades
        new_active = []
        for tr in active_trades:
            days_held = (date - tr['entry_date']).days
            # Use trading days count instead
            entry_loc = prices.index.get_loc(tr['entry_date'])
            trading_days_held = date_idx - entry_loc
            if trading_days_held >= hold_days:
                exit_price = prices.loc[date, tr['ticker']] * slippage_sell
                pnl = (exit_price - tr['entry_price']) * tr['shares']
                daily_pnl.loc[date] += pnl
                pnl_by_ticker[tr['ticker']] += pnl
                trade_list.append({
                    'ticker': tr['ticker'],
                    'entry_date': str(tr['entry_date'].date()),
                    'exit_date': str(date.date()),
                    'entry_price': tr['entry_price'],
                    'exit_price': exit_price,
                    'shares': tr['shares'],
                    'pnl': pnl
                })
            else:
                new_active.append(tr)
        active_trades = new_active

        # Check for new signals if capacity available
        if len(active_trades) >= MAX_CONCURRENT:
            continue

        if date_idx < 20:
            continue

        for ticker in prices.columns:
            if len(active_trades) >= MAX_CONCURRENT:
                break

            # Skip if already holding this ticker
            if any(t['ticker'] == ticker for t in active_trades):
                continue

            # Get recent prices
            recent = prices[ticker].iloc[max(0, date_idx - 25):date_idx + 1]
            if len(recent) < min_consec + 1:
                continue

            current_price = recent.iloc[-1]
            if pd.isna(current_price) or current_price <= 0:
                continue

            # Compute daily returns for last min_consec days
            returns = recent.pct_change().dropna()
            if len(returns) < min_consec:
                continue

            last_n = returns.iloc[-min_consec:]

            if not inverse:
                # ORIGINAL: consecutive red days with accelerating losses
                all_red = all(r < 0 for r in last_n)
                if not all_red:
                    continue

                if require_acceleration:
                    # Each loss must be LARGER (more negative) than previous
                    accelerating = all(
                        last_n.iloc[j] < last_n.iloc[j - 1]
                        for j in range(1, len(last_n))
                    )
                    if not accelerating:
                        continue

                # Must be >dip_pct% below 20-day high
                high_20d = prices[ticker].iloc[date_idx - 20:date_idx].max()
                if current_price > high_20d * (1 - dip_pct / 100):
                    continue
            else:
                # INVERSE: consecutive green days with accelerating gains
                all_green = all(r > 0 for r in last_n)
                if not all_green:
                    continue

                if require_acceleration:
                    accelerating = all(
                        last_n.iloc[j] > last_n.iloc[j - 1]
                        for j in range(1, len(last_n))
                    )
                    if not accelerating:
                        continue

                # Must be ABOVE 20-day high
                high_20d = prices[ticker].iloc[date_idx - 20:date_idx].max()
                if current_price < high_20d:
                    continue

            # Position sizing
            entry_price = current_price * slippage_mult
            shares = int(MAX_PER_TRADE / entry_price)
            if shares < 1:
                continue

            active_trades.append({
                'ticker': ticker,
                'entry_date': date,
                'entry_price': entry_price,
                'shares': shares
            })

        # Mark-to-market for active trades (for daily returns)
        # We track realized PnL on exit, so daily_pnl already captures that

    # Close remaining trades at end
    last_date = trade_dates[-1]
    last_idx = prices.index.get_loc(last_date)
    for tr in active_trades:
        exit_price = prices.loc[last_date, tr['ticker']] * slippage_sell
        pnl = (exit_price - tr['entry_price']) * tr['shares']
        daily_pnl.loc[last_date] += pnl
        pnl_by_ticker[tr['ticker']] += pnl
        trade_list.append({
            'ticker': tr['ticker'],
            'entry_date': str(tr['entry_date'].date()),
            'exit_date': str(last_date.date()),
            'entry_price': tr['entry_price'],
            'exit_price': exit_price,
            'shares': tr['shares'],
            'pnl': pnl
        })

    daily_returns = daily_pnl / CAPITAL
    sharpe = compute_sharpe(daily_returns)
    total_pnl = daily_pnl.sum()
    n_trades = len(trade_list)

    return {
        'sharpe': sharpe,
        'total_pnl': float(total_pnl),
        'n_trades': n_trades,
        'daily_returns': daily_returns,
        'pnl_by_ticker': dict(pnl_by_ticker),
        'trade_list': trade_list
    }


# ── BASELINE ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BASELINE RUN")
print("="*70)
baseline = run_strategy(close)
print(f"Sharpe: {baseline['sharpe']:.3f}, Trades: {baseline['n_trades']}, PnL: ${baseline['total_pnl']:.2f}")

results = {
    'strategy': 'Consecutive Dip B — Deepening Losses',
    'baseline': {
        'sharpe': baseline['sharpe'],
        'n_trades': baseline['n_trades'],
        'total_pnl': baseline['total_pnl']
    },
    'tests': {}
}


# ── TEST 1: INVERSE SIGNAL ──────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 1: Inverse Signal (buy on accelerating gains above 20d high)")
print("="*70)
inverse_result = run_strategy(close, inverse=True)
inverse_sharpe = inverse_result['sharpe']
ratio = abs(inverse_sharpe / baseline['sharpe']) if baseline['sharpe'] != 0 else 999
passed_1 = inverse_sharpe < 0.5 * baseline['sharpe']
print(f"Inverse Sharpe: {inverse_sharpe:.3f}, Baseline Sharpe: {baseline['sharpe']:.3f}")
print(f"Inverse/Baseline ratio: {ratio:.3f}")
print(f"Inverse trades: {inverse_result['n_trades']}")
print(f"PASS: {passed_1} (inverse < 50% of baseline)")

results['tests']['1_inverse_signal'] = {
    'inverse_sharpe': inverse_sharpe,
    'baseline_sharpe': baseline['sharpe'],
    'ratio': ratio,
    'inverse_trades': inverse_result['n_trades'],
    'threshold': 0.5,
    'passed': passed_1
}


# ── TEST 2: RANDOM TIMING PERCENTILE ────────────────────────────────────
print("\n" + "="*70)
print("TEST 2: Random Timing Percentile (1000 permutations)")
print("="*70)

# Get all trade entry dates from baseline
baseline_entries = [t['entry_date'] for t in baseline['trade_list']]
n_baseline_trades = len(baseline_entries)

trade_start_dt = pd.Timestamp(TRADE_START)
trade_dates_all = close.index[close.index >= trade_start_dt]

random_sharpes = []
for perm in range(1000):
    if perm % 200 == 0:
        print(f"  Permutation {perm}/1000...")

    # Random entry dates, random tickers
    rand_dates = np.random.choice(trade_dates_all, size=n_baseline_trades, replace=True)
    rand_tickers = np.random.choice(TICKERS, size=n_baseline_trades, replace=True)

    daily_pnl = pd.Series(0.0, index=trade_dates_all)
    slippage_mult = 1.0 + DEFAULT_SLIPPAGE_BPS / 10000.0
    slippage_sell = 1.0 - DEFAULT_SLIPPAGE_BPS / 10000.0

    for entry_date, ticker in zip(rand_dates, rand_tickers):
        entry_loc = close.index.get_loc(entry_date)
        exit_loc = min(entry_loc + DEFAULT_HOLD, len(close) - 1)
        exit_date = close.index[exit_loc]

        entry_price = close.loc[entry_date, ticker] * slippage_mult
        if pd.isna(entry_price) or entry_price <= 0:
            continue
        shares = int(MAX_PER_TRADE / entry_price)
        if shares < 1:
            continue

        exit_price = close.iloc[exit_loc][ticker] * slippage_sell
        if pd.isna(exit_price):
            continue
        pnl = (exit_price - entry_price) * shares
        daily_pnl.loc[exit_date] += pnl

    daily_ret = daily_pnl / CAPITAL
    random_sharpes.append(compute_sharpe(daily_ret))

random_sharpes = np.array(random_sharpes)
p_value = float(np.mean(random_sharpes >= baseline['sharpe']))
passed_2 = p_value < 0.05
print(f"Baseline Sharpe: {baseline['sharpe']:.3f}")
print(f"Random Sharpe mean: {np.mean(random_sharpes):.3f}, std: {np.std(random_sharpes):.3f}")
print(f"Random Sharpe 95th pctile: {np.percentile(random_sharpes, 95):.3f}")
print(f"Permutation p-value: {p_value:.4f}")
print(f"PASS: {passed_2} (p < 0.05)")

results['tests']['2_random_timing'] = {
    'baseline_sharpe': baseline['sharpe'],
    'random_sharpe_mean': float(np.mean(random_sharpes)),
    'random_sharpe_std': float(np.std(random_sharpes)),
    'random_sharpe_95pct': float(np.percentile(random_sharpes, 95)),
    'p_value': p_value,
    'n_permutations': 1000,
    'passed': passed_2
}


# ── TEST 3: SUB-PERIOD STABILITY ────────────────────────────────────────
print("\n" + "="*70)
print("TEST 3: Sub-Period Stability (4 equal sub-periods)")
print("="*70)

daily_ret = baseline['daily_returns']
n_days = len(daily_ret)
quarter = n_days // 4
sub_sharpes = []
for q in range(4):
    start = q * quarter
    end = (q + 1) * quarter if q < 3 else n_days
    sub = daily_ret.iloc[start:end]
    s = compute_sharpe(sub)
    sub_sharpes.append(s)
    period_start = sub.index[0].date()
    period_end = sub.index[-1].date()
    print(f"  Q{q+1} ({period_start} to {period_end}): Sharpe={s:.3f}, days={len(sub)}")

passed_3 = all(s > 0 for s in sub_sharpes)
print(f"All positive: {passed_3}")
print(f"PASS: {passed_3}")

results['tests']['3_sub_period_stability'] = {
    'sub_period_sharpes': sub_sharpes,
    'all_positive': passed_3,
    'passed': passed_3
}


# ── TEST 4: REMOVE TOP-3 TICKERS ────────────────────────────────────────
print("\n" + "="*70)
print("TEST 4: Remove Top-3 Tickers by PnL")
print("="*70)

ticker_pnl = baseline['pnl_by_ticker']
sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
print("PnL by ticker (top 5):")
for t, p in sorted_tickers[:5]:
    print(f"  {t}: ${p:.2f}")

top3 = [t for t, _ in sorted_tickers[:3]]
remaining = [t for t in TICKERS if t not in top3]
print(f"\nRemoving top 3: {top3}")
print(f"Remaining: {remaining}")

reduced = run_strategy(close, tickers=remaining)
sharpe_drop = 1.0 - reduced['sharpe'] / baseline['sharpe'] if baseline['sharpe'] != 0 else 1.0
passed_4 = sharpe_drop < 0.5
print(f"Reduced Sharpe: {reduced['sharpe']:.3f} (drop: {sharpe_drop:.1%})")
print(f"Reduced trades: {reduced['n_trades']}")
print(f"PASS: {passed_4} (drop < 50%)")

results['tests']['4_remove_top3'] = {
    'top3_tickers': top3,
    'baseline_sharpe': baseline['sharpe'],
    'reduced_sharpe': reduced['sharpe'],
    'sharpe_drop_pct': sharpe_drop,
    'reduced_trades': reduced['n_trades'],
    'passed': passed_4
}


# ── TEST 5: PARAMETER SENSITIVITY GRID ──────────────────────────────────
print("\n" + "="*70)
print("TEST 5: Parameter Sensitivity Grid")
print("="*70)

consec_vals = [2, 3, 4, 5]
dip_vals = [3, 5, 7, 10]
hold_vals = [5, 7, 10, 15]
accel_vals = [True, False]

total_combos = len(consec_vals) * len(dip_vals) * len(hold_vals) * len(accel_vals)
print(f"Testing {total_combos} parameter combinations...")

grid_results = []
count = 0
above_threshold = 0
threshold = 0.3

for consec in consec_vals:
    for dip in dip_vals:
        for hold in hold_vals:
            for accel in accel_vals:
                count += 1
                if count % 32 == 0:
                    print(f"  {count}/{total_combos}...")
                r = run_strategy(close, min_consec=consec, dip_pct=dip,
                                hold_days=hold, require_acceleration=accel)
                grid_results.append({
                    'min_consec': consec,
                    'dip_pct': dip,
                    'hold_days': hold,
                    'require_acceleration': accel,
                    'sharpe': r['sharpe'],
                    'n_trades': r['n_trades'],
                    'total_pnl': r['total_pnl']
                })
                if r['sharpe'] > threshold:
                    above_threshold += 1

pct_above = above_threshold / total_combos
passed_5 = pct_above > 0.5
print(f"\nCombinations with Sharpe > {threshold}: {above_threshold}/{total_combos} ({pct_above:.1%})")
print(f"PASS: {passed_5} (> 50% above {threshold})")

# Show best and worst
grid_df = pd.DataFrame(grid_results).sort_values('sharpe', ascending=False)
print("\nTop 5 configs:")
for _, row in grid_df.head(5).iterrows():
    print(f"  consec={row['min_consec']}, dip={row['dip_pct']}%, hold={row['hold_days']}, "
          f"accel={row['require_acceleration']}: Sharpe={row['sharpe']:.3f}, trades={row['n_trades']}")
print("Bottom 5 configs:")
for _, row in grid_df.tail(5).iterrows():
    print(f"  consec={row['min_consec']}, dip={row['dip_pct']}%, hold={row['hold_days']}, "
          f"accel={row['require_acceleration']}: Sharpe={row['sharpe']:.3f}, trades={row['n_trades']}")

results['tests']['5_parameter_sensitivity'] = {
    'total_combinations': total_combos,
    'above_threshold': above_threshold,
    'threshold': threshold,
    'pct_above': pct_above,
    'passed': passed_5,
    'grid_summary': grid_results
}


# ── TEST 6: COST SENSITIVITY ────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 6: Cost Sensitivity")
print("="*70)

slippage_vals = [0, 2, 5, 10, 20, 50]
cost_results = []
for slip in slippage_vals:
    r = run_strategy(close, slippage_bps=slip)
    cost_results.append({
        'slippage_bps': slip,
        'sharpe': r['sharpe'],
        'n_trades': r['n_trades'],
        'total_pnl': r['total_pnl']
    })
    print(f"  Slippage {slip:2d} bps: Sharpe={r['sharpe']:.3f}, PnL=${r['total_pnl']:.2f}")

# Find breakeven slippage (where Sharpe crosses 0)
breakeven_bps = 0
for i in range(len(cost_results) - 1):
    if cost_results[i]['sharpe'] > 0 and cost_results[i + 1]['sharpe'] <= 0:
        # Linear interpolation
        s1 = cost_results[i]['sharpe']
        s2 = cost_results[i + 1]['sharpe']
        b1 = cost_results[i]['slippage_bps']
        b2 = cost_results[i + 1]['slippage_bps']
        breakeven_bps = b1 + (b2 - b1) * s1 / (s1 - s2)
        break
else:
    if all(r['sharpe'] > 0 for r in cost_results):
        breakeven_bps = cost_results[-1]['slippage_bps']  # Still positive at max
        print(f"  Still profitable at {breakeven_bps} bps — breakeven > {breakeven_bps}")
        breakeven_bps = breakeven_bps + 1  # Mark as above max tested

passed_6 = breakeven_bps > 20
print(f"\nBreakeven slippage: ~{breakeven_bps:.0f} bps")
print(f"PASS: {passed_6} (breakeven > 20 bps)")

results['tests']['6_cost_sensitivity'] = {
    'cost_curve': cost_results,
    'breakeven_bps': float(breakeven_bps),
    'passed': passed_6
}


# ── SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("ADVERSARIAL VALIDATION SUMMARY")
print("="*70)

test_names = {
    '1_inverse_signal': 'Inverse Signal',
    '2_random_timing': 'Random Timing',
    '3_sub_period_stability': 'Sub-Period Stability',
    '4_remove_top3': 'Remove Top-3 Tickers',
    '5_parameter_sensitivity': 'Parameter Sensitivity',
    '6_cost_sensitivity': 'Cost Sensitivity'
}

n_passed = 0
for key, name in test_names.items():
    p = results['tests'][key]['passed']
    n_passed += p
    print(f"  {'PASS' if p else 'FAIL'} — {name}")

results['summary'] = {
    'tests_passed': n_passed,
    'tests_total': 6,
    'overall_pass': n_passed >= 5,
    'timestamp': datetime.now().isoformat()
}

print(f"\nOverall: {n_passed}/6 passed ({'PASS' if n_passed >= 5 else 'FAIL'} — need 5/6)")

# Save results
output_path = '/home/jupiter/Lvl3Quant/data/consecutive_dip_b_adversarial.json'

# Clean up non-serializable fields
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()
                if k != 'daily_returns'}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj

results_clean = clean_for_json(results)

with open(output_path, 'w') as f:
    json.dump(results_clean, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
