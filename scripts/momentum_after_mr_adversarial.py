#!/usr/bin/env python3
"""
Momentum After MR — Variant D Adversarial Validation
=====================================================
Strategy: "SMA Cross + RSI Exit"
After a quality stock's MR bounce (5%+ dip from 20d high + RSI<35 + recovery),
enter when price crosses above SMA(5), exit when RSI(14)>70 or max 20 days.

6 adversarial tests:
  1. Inverse Signal
  2. Random Entry Timing (1000 iterations)
  3. Sub-Period Stability (4 sub-periods)
  4. Remove Top 3 Tickers
  5. Parameter Sensitivity sweep
  6. Cost Sensitivity
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# ─── Config ───────────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

START = "2021-01-01"  # lookback before OOT
OOT_START = "2022-01-03"
OOT_END = "2026-07-31"
INITIAL_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2  # 0.02% each way = 2 bps
RISK_FREE = 0.04

# Default strategy params
DEFAULT_SMA = 5
DEFAULT_RSI_EXIT = 70
DEFAULT_MAX_HOLD = 20
RSI_PERIOD = 14
DIP_PCT = 0.05        # 5% drop from 20d high
RSI_DIP_THRESH = 35   # RSI must have crossed below this
DIP_LOOKBACK = 20     # look back 20 days for the dip event

# ─── Helpers ──────────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def annualized_sharpe(returns, rf=RISK_FREE):
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    return (ann_ret - rf) / ann_vol

def run_strategy(close_df, tickers, sma_period=DEFAULT_SMA, rsi_exit=DEFAULT_RSI_EXIT,
                 max_hold=DEFAULT_MAX_HOLD, slippage_bps=SLIPPAGE_BPS,
                 entry_dates_override=None, inverse=False):
    """
    Run the Momentum After MR Variant D strategy.
    Returns: list of trade dicts, daily equity series

    If entry_dates_override is provided: dict {ticker: [dates]} — use those entries instead.
    If inverse=True: buy near highs, RSI>65, price below SMA(5).
    """
    slippage = slippage_bps / 10000.0
    oot_mask = close_df.index >= pd.Timestamp(OOT_START)
    oot_dates = close_df.index[oot_mask]

    # Precompute indicators for all tickers
    indicators = {}
    for t in tickers:
        if t not in close_df.columns:
            continue
        px = close_df[t].dropna()
        if len(px) < 50:
            continue
        rsi = compute_rsi(px, RSI_PERIOD)
        sma = px.rolling(sma_period).mean()
        high_20d = px.rolling(20).max()
        indicators[t] = {
            'px': px, 'rsi': rsi, 'sma': sma, 'high_20d': high_20d
        }

    trades = []
    positions = {}  # ticker -> {entry_date, entry_price, shares}
    capital = INITIAL_CAPITAL
    equity_curve = []

    for date in oot_dates:
        # Mark-to-market
        pos_value = 0
        for t, pos in list(positions.items()):
            if t in indicators and date in indicators[t]['px'].index:
                pos_value += pos['shares'] * indicators[t]['px'].loc[date]
        equity_curve.append({'date': date, 'equity': capital + pos_value})

        # Check exits first
        for t in list(positions.keys()):
            if t not in indicators or date not in indicators[t]['px'].index:
                continue
            pos = positions[t]
            ind = indicators[t]
            days_held = (date - pos['entry_date']).days
            rsi_val = ind['rsi'].get(date, 50)

            exit_signal = False
            if rsi_val > rsi_exit:
                exit_signal = True
            if days_held >= max_hold:
                exit_signal = True

            if exit_signal:
                exit_price = ind['px'].loc[date] * (1 - slippage)
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                capital += pos['shares'] * exit_price
                trades.append({
                    'ticker': t,
                    'entry_date': str(pos['entry_date'].date()),
                    'exit_date': str(date.date()),
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'days_held': days_held,
                })
                del positions[t]

        # Check entries
        if len(positions) >= MAX_CONCURRENT:
            continue

        if entry_dates_override is not None:
            # Use override entries
            for t in tickers:
                if len(positions) >= MAX_CONCURRENT:
                    break
                if t in positions or t not in indicators:
                    continue
                if date not in indicators[t]['px'].index:
                    continue
                if t in entry_dates_override and date in entry_dates_override[t]:
                    px_now = indicators[t]['px'].loc[date]
                    entry_price = px_now * (1 + slippage)
                    shares = min(MAX_PER_TRADE, capital * 0.95) / entry_price
                    if shares < 0.01:
                        continue
                    cost = shares * entry_price
                    if cost > capital:
                        continue
                    capital -= cost
                    positions[t] = {
                        'entry_date': date,
                        'entry_price': entry_price,
                        'shares': shares,
                    }
        elif inverse:
            # Inverse signal: near highs, RSI > 65, price below SMA
            for t in tickers:
                if len(positions) >= MAX_CONCURRENT:
                    break
                if t in positions or t not in indicators:
                    continue
                ind = indicators[t]
                if date not in ind['px'].index:
                    continue
                px_now = ind['px'].loc[date]
                high_20d = ind['high_20d'].get(date, np.nan)
                rsi_val = ind['rsi'].get(date, 50)
                sma_val = ind['sma'].get(date, np.nan)

                if pd.isna(high_20d) or pd.isna(sma_val):
                    continue

                # Within 2% of 20d high
                near_high = (high_20d - px_now) / high_20d < 0.02
                rsi_high = rsi_val > 65
                below_sma = px_now < sma_val

                if near_high and rsi_high and below_sma:
                    entry_price = px_now * (1 + slippage)
                    shares = min(MAX_PER_TRADE, capital * 0.95) / entry_price
                    if shares < 0.01:
                        continue
                    cost = shares * entry_price
                    if cost > capital:
                        continue
                    capital -= cost
                    positions[t] = {
                        'entry_date': date,
                        'entry_price': entry_price,
                        'shares': shares,
                    }
        else:
            # Normal signal
            for t in tickers:
                if len(positions) >= MAX_CONCURRENT:
                    break
                if t in positions or t not in indicators:
                    continue
                ind = indicators[t]
                if date not in ind['px'].index:
                    continue

                px_now = ind['px'].loc[date]
                high_20d = ind['high_20d'].get(date, np.nan)
                rsi_val = ind['rsi'].get(date, 50)
                sma_val = ind['sma'].get(date, np.nan)

                if pd.isna(high_20d) or pd.isna(sma_val):
                    continue

                # Check MR dip event in last 20 days:
                # 5% drop from 20d high + RSI crossed below 35
                dip_occurred = False
                lookback_start = max(0, ind['px'].index.get_loc(date) - DIP_LOOKBACK)
                lookback_end = ind['px'].index.get_loc(date)
                for i in range(lookback_start, lookback_end):
                    lb_date = ind['px'].index[i]
                    lb_px = ind['px'].iloc[i]
                    lb_high = ind['high_20d'].get(lb_date, np.nan)
                    lb_rsi = ind['rsi'].get(lb_date, np.nan)
                    if pd.isna(lb_high) or pd.isna(lb_rsi):
                        continue
                    drop_pct = (lb_high - lb_px) / lb_high
                    if drop_pct >= DIP_PCT and lb_rsi < RSI_DIP_THRESH:
                        dip_occurred = True
                        break

                if not dip_occurred:
                    continue

                # Price crosses above SMA(5)
                prev_idx = ind['px'].index.get_loc(date) - 1
                if prev_idx < 0:
                    continue
                prev_date = ind['px'].index[prev_idx]
                prev_px = ind['px'].iloc[prev_idx]
                prev_sma = ind['sma'].get(prev_date, np.nan)
                if pd.isna(prev_sma):
                    continue

                cross_above = prev_px <= prev_sma and px_now > sma_val

                if cross_above:
                    entry_price = px_now * (1 + slippage)
                    shares = min(MAX_PER_TRADE, capital * 0.95) / entry_price
                    if shares < 0.01:
                        continue
                    cost = shares * entry_price
                    if cost > capital:
                        continue
                    capital -= cost
                    positions[t] = {
                        'entry_date': date,
                        'entry_price': entry_price,
                        'shares': shares,
                    }

    # Close remaining positions at last date
    last_date = oot_dates[-1] if len(oot_dates) > 0 else None
    if last_date:
        for t in list(positions.keys()):
            if t in indicators and last_date in indicators[t]['px'].index:
                pos = positions[t]
                exit_price = indicators[t]['px'].loc[last_date] * (1 - slippage)
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                capital += pos['shares'] * exit_price
                trades.append({
                    'ticker': t,
                    'entry_date': str(pos['entry_date'].date()),
                    'exit_date': str(last_date.date()),
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'days_held': (last_date - pos['entry_date']).days,
                })

    # Build daily returns from equity curve
    if len(equity_curve) > 1:
        eq = pd.DataFrame(equity_curve).set_index('date')['equity']
        daily_returns = eq.pct_change().dropna()
    else:
        daily_returns = pd.Series(dtype=float)

    return trades, daily_returns


# ─── Download Data ────────────────────────────────────────────────────────────
flush_print("Downloading price data...")
all_tickers = TICKERS + ["SPY"]
raw = yf.download(all_tickers, start=START, end=OOT_END, auto_adjust=True, progress=False)
close = raw["Close"].copy().ffill()

spy_close = close["SPY"].copy()
spy_sma200 = spy_close.rolling(200).mean()
regime = (spy_close > spy_sma200).astype(int)  # 1=bull, 0=bear

flush_print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} rows")

# ─── Baseline Strategy ───────────────────────────────────────────────────────
flush_print("\n{'='*60}")
flush_print("Running baseline strategy (Variant D: SMA Cross + RSI Exit)...")
baseline_trades, baseline_returns = run_strategy(close, TICKERS)

baseline_sharpe = annualized_sharpe(baseline_returns)
total_pnl = sum(t['pnl'] for t in baseline_trades)
n_trades = len(baseline_trades)
win_rate = sum(1 for t in baseline_trades if t['pnl'] > 0) / max(n_trades, 1) * 100

flush_print(f"Baseline: {n_trades} trades, PnL=${total_pnl:.2f}, WR={win_rate:.1f}%, Sharpe={baseline_sharpe:.3f}")

# ─── Regime breakdown ────────────────────────────────────────────────────────
bull_pnl = sum(t['pnl'] for t in baseline_trades
               if regime.get(pd.Timestamp(t['entry_date']), 0) == 1)
bear_pnl = sum(t['pnl'] for t in baseline_trades
               if regime.get(pd.Timestamp(t['entry_date']), 0) == 0)
flush_print(f"  Bull PnL: ${bull_pnl:.2f}, Bear PnL: ${bear_pnl:.2f}")

results = {
    'strategy': 'Momentum After MR - Variant D (SMA Cross + RSI Exit)',
    'baseline': {
        'n_trades': n_trades,
        'total_pnl': round(total_pnl, 2),
        'win_rate': round(win_rate, 1),
        'sharpe': round(baseline_sharpe, 3),
        'bull_pnl': round(bull_pnl, 2),
        'bear_pnl': round(bear_pnl, 2),
    },
    'tests': {},
}

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Inverse Signal
# ═══════════════════════════════════════════════════════════════════════════════
flush_print("\n" + "="*60)
flush_print("TEST 1: Inverse Signal")
flush_print("  Entry: near 20d high (<2%), RSI>65, price below SMA(5)")

inv_trades, inv_returns = run_strategy(close, TICKERS, inverse=True)
inv_sharpe = annualized_sharpe(inv_returns)
inv_pnl = sum(t['pnl'] for t in inv_trades)

ratio = abs(inv_sharpe) / max(abs(baseline_sharpe), 1e-9)
t1_pass = ratio < 0.50

flush_print(f"  Inverse: {len(inv_trades)} trades, PnL=${inv_pnl:.2f}, Sharpe={inv_sharpe:.3f}")
flush_print(f"  |Inverse Sharpe| / |Baseline Sharpe| = {ratio:.3f}")
flush_print(f"  TEST 1: {'PASS' if t1_pass else 'FAIL'} (threshold: <0.50)")

results['tests']['1_inverse_signal'] = {
    'inverse_sharpe': round(inv_sharpe, 3),
    'baseline_sharpe': round(baseline_sharpe, 3),
    'ratio': round(ratio, 3),
    'threshold': 0.50,
    'pass': t1_pass,
}

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Random Entry Timing (1000 iterations)
# ═══════════════════════════════════════════════════════════════════════════════
flush_print("\n" + "="*60)
flush_print("TEST 2: Random Entry Timing (1000 iterations)")

# Collect real entry info
real_entries_by_ticker = defaultdict(list)
for t in baseline_trades:
    real_entries_by_ticker[t['ticker']].append(pd.Timestamp(t['entry_date']))

oot_mask = close.index >= pd.Timestamp(OOT_START)
oot_dates_list = close.index[oot_mask].tolist()

random_sharpes = []
for iteration in range(1000):
    # Randomly reassign entry dates (same ticker, same count)
    random_entries = {}
    rng = np.random.RandomState(iteration)
    for ticker, dates in real_entries_by_ticker.items():
        n = len(dates)
        if n == 0:
            continue
        rand_idx = rng.choice(len(oot_dates_list), size=n, replace=False)
        random_entries[ticker] = set([oot_dates_list[i] for i in rand_idx])

    _, rand_returns = run_strategy(close, TICKERS, entry_dates_override=random_entries)
    rand_sharpe = annualized_sharpe(rand_returns)
    random_sharpes.append(rand_sharpe)

    if (iteration + 1) % 200 == 0:
        flush_print(f"  ... completed {iteration + 1}/1000 iterations")

random_sharpes = np.array(random_sharpes)
p_value = np.mean(random_sharpes >= baseline_sharpe)
percentile = (1 - p_value) * 100
t2_pass = p_value < 0.05

flush_print(f"  Baseline Sharpe: {baseline_sharpe:.3f}")
flush_print(f"  Random Sharpe: mean={random_sharpes.mean():.3f}, std={random_sharpes.std():.3f}")
flush_print(f"  Percentile rank: {percentile:.1f}th")
flush_print(f"  p-value: {p_value:.4f}")
flush_print(f"  TEST 2: {'PASS' if t2_pass else 'FAIL'} (p_value < 0.05)")

results['tests']['2_random_entry_timing'] = {
    'baseline_sharpe': round(baseline_sharpe, 3),
    'random_mean_sharpe': round(float(random_sharpes.mean()), 3),
    'random_std_sharpe': round(float(random_sharpes.std()), 3),
    'percentile_rank': round(percentile, 1),
    'p_value': round(float(p_value), 4),
    'pass': t2_pass,
}

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Sub-Period Stability
# ═══════════════════════════════════════════════════════════════════════════════
flush_print("\n" + "="*60)
flush_print("TEST 3: Sub-Period Stability (4 sub-periods)")

oot_start_ts = pd.Timestamp(OOT_START)
oot_end_ts = close.index[close.index >= pd.Timestamp(OOT_START)][-1]
total_days = (oot_end_ts - oot_start_ts).days
quarter_days = total_days // 4

sub_periods = []
for i in range(4):
    sp_start = oot_start_ts + pd.Timedelta(days=i * quarter_days)
    if i < 3:
        sp_end = oot_start_ts + pd.Timedelta(days=(i + 1) * quarter_days)
    else:
        sp_end = oot_end_ts
    sub_periods.append((sp_start, sp_end))

sub_sharpes = []
sub_details = []
all_sub_positive = True

for i, (sp_start, sp_end) in enumerate(sub_periods):
    # Filter trades that started in this sub-period
    sp_trades = [t for t in baseline_trades
                 if sp_start <= pd.Timestamp(t['entry_date']) <= sp_end]

    # Filter returns for this period
    sp_returns = baseline_returns[(baseline_returns.index >= sp_start) &
                                  (baseline_returns.index <= sp_end)]
    sp_sharpe = annualized_sharpe(sp_returns)
    sp_pnl = sum(t['pnl'] for t in sp_trades)

    sub_sharpes.append(sp_sharpe)
    sub_details.append({
        'period': f"{sp_start.date()} to {sp_end.date()}",
        'n_trades': len(sp_trades),
        'pnl': round(sp_pnl, 2),
        'sharpe': round(sp_sharpe, 3),
    })

    if sp_sharpe <= 0:
        all_sub_positive = False

    flush_print(f"  Period {i+1} ({sp_start.date()} to {sp_end.date()}): "
                f"{len(sp_trades)} trades, PnL=${sp_pnl:.2f}, Sharpe={sp_sharpe:.3f}")

t3_pass = all_sub_positive
flush_print(f"  All 4 sub-periods positive Sharpe: {all_sub_positive}")
flush_print(f"  TEST 3: {'PASS' if t3_pass else 'FAIL'}")

results['tests']['3_sub_period_stability'] = {
    'sub_periods': sub_details,
    'all_positive_sharpe': all_sub_positive,
    'pass': t3_pass,
}

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Remove Top 3 Tickers
# ═══════════════════════════════════════════════════════════════════════════════
flush_print("\n" + "="*60)
flush_print("TEST 4: Remove Top 3 Tickers by PnL")

ticker_pnl = defaultdict(float)
for t in baseline_trades:
    ticker_pnl[t['ticker']] += t['pnl']

sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
top3 = [t[0] for t in sorted_tickers[:3]]
remaining_tickers = [t for t in TICKERS if t not in top3]

flush_print(f"  Top 3 by PnL: {[(t, f'${p:.2f}') for t, p in sorted_tickers[:3]]}")

reduced_trades, reduced_returns = run_strategy(close, remaining_tickers)
reduced_sharpe = annualized_sharpe(reduced_returns)
reduced_pnl = sum(t['pnl'] for t in reduced_trades)

if abs(baseline_sharpe) > 1e-9:
    sharpe_drop = (baseline_sharpe - reduced_sharpe) / abs(baseline_sharpe)
else:
    sharpe_drop = 0.0

t4_pass = sharpe_drop < 0.50

flush_print(f"  Reduced: {len(reduced_trades)} trades, PnL=${reduced_pnl:.2f}, Sharpe={reduced_sharpe:.3f}")
flush_print(f"  Sharpe drop: {sharpe_drop*100:.1f}%")
flush_print(f"  TEST 4: {'PASS' if t4_pass else 'FAIL'} (drop < 50%)")

results['tests']['4_remove_top3'] = {
    'top3_tickers': top3,
    'top3_pnl': [round(ticker_pnl[t], 2) for t in top3],
    'baseline_sharpe': round(baseline_sharpe, 3),
    'reduced_sharpe': round(reduced_sharpe, 3),
    'sharpe_drop_pct': round(sharpe_drop * 100, 1),
    'pass': t4_pass,
}

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Parameter Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
flush_print("\n" + "="*60)
flush_print("TEST 5: Parameter Sensitivity Sweep")

sma_periods = [3, 5, 7, 10]
rsi_exits = [60, 65, 70, 75]
max_holds = [10, 15, 20, 30]

total_combos = len(sma_periods) * len(rsi_exits) * len(max_holds)
good_combos = 0
combo_results = []
done = 0

for sma_p in sma_periods:
    for rsi_e in rsi_exits:
        for mh in max_holds:
            _, combo_returns = run_strategy(close, TICKERS, sma_period=sma_p,
                                            rsi_exit=rsi_e, max_hold=mh)
            combo_sharpe = annualized_sharpe(combo_returns)
            if combo_sharpe > 0.3:
                good_combos += 1
            combo_results.append({
                'sma': sma_p, 'rsi_exit': rsi_e, 'max_hold': mh,
                'sharpe': round(combo_sharpe, 3),
            })
            done += 1
            if done % 16 == 0:
                flush_print(f"  ... {done}/{total_combos} combinations tested")

pct_good = good_combos / total_combos * 100
t5_pass = pct_good >= 30  # at least 30% of combos should be viable

flush_print(f"  {good_combos}/{total_combos} combos with Sharpe > 0.3 ({pct_good:.1f}%)")
flush_print(f"  TEST 5: {'PASS' if t5_pass else 'FAIL'} (>=30% required)")

# Find best/worst
best_combo = max(combo_results, key=lambda x: x['sharpe'])
worst_combo = min(combo_results, key=lambda x: x['sharpe'])
flush_print(f"  Best:  SMA={best_combo['sma']}, RSI_exit={best_combo['rsi_exit']}, "
            f"hold={best_combo['max_hold']} -> Sharpe={best_combo['sharpe']:.3f}")
flush_print(f"  Worst: SMA={worst_combo['sma']}, RSI_exit={worst_combo['rsi_exit']}, "
            f"hold={worst_combo['max_hold']} -> Sharpe={worst_combo['sharpe']:.3f}")

results['tests']['5_parameter_sensitivity'] = {
    'total_combinations': total_combos,
    'sharpe_above_0_3': good_combos,
    'pct_good': round(pct_good, 1),
    'best_combo': best_combo,
    'worst_combo': worst_combo,
    'pass': t5_pass,
}

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Cost Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
flush_print("\n" + "="*60)
flush_print("TEST 6: Cost Sensitivity")

cost_levels = [5, 10, 20, 50]  # bps
cost_sharpes = {}

for bps in cost_levels:
    _, cost_returns = run_strategy(close, TICKERS, slippage_bps=bps)
    cs = annualized_sharpe(cost_returns)
    cost_sharpes[bps] = cs
    flush_print(f"  {bps} bps: Sharpe={cs:.3f}")

# Find breakeven slippage via interpolation
# Test finer granularity around where Sharpe crosses zero
test_bps_range = list(range(1, 101, 1))
breakeven_bps = None
prev_sharpe = None

for bps in test_bps_range:
    _, cr = run_strategy(close, TICKERS, slippage_bps=bps)
    cs = annualized_sharpe(cr)
    if prev_sharpe is not None and prev_sharpe > 0 and cs <= 0:
        # Linear interpolation
        breakeven_bps = bps - 1 + prev_sharpe / (prev_sharpe - cs + 1e-12)
        break
    prev_sharpe = cs

if breakeven_bps is None:
    # Check if always positive or always negative
    if prev_sharpe is not None and prev_sharpe > 0:
        breakeven_bps = 100  # still positive at 100bps
    else:
        breakeven_bps = 0  # never positive

t6_pass = breakeven_bps > SLIPPAGE_BPS * 3  # at least 3x headroom

flush_print(f"  Breakeven slippage: {breakeven_bps:.1f} bps")
flush_print(f"  Headroom: {breakeven_bps / max(SLIPPAGE_BPS, 0.01):.1f}x operating cost ({SLIPPAGE_BPS} bps)")
flush_print(f"  TEST 6: {'PASS' if t6_pass else 'FAIL'} (need >3x headroom)")

results['tests']['6_cost_sensitivity'] = {
    'cost_sharpes': {str(k): round(v, 3) for k, v in cost_sharpes.items()},
    'breakeven_bps': round(breakeven_bps, 1),
    'operating_bps': SLIPPAGE_BPS,
    'headroom_x': round(breakeven_bps / max(SLIPPAGE_BPS, 0.01), 1),
    'pass': t6_pass,
}

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
flush_print("\n" + "="*60)
flush_print("ADVERSARIAL VALIDATION SUMMARY")
flush_print("="*60)

test_names = {
    '1_inverse_signal': 'Inverse Signal',
    '2_random_entry_timing': 'Random Entry Timing',
    '3_sub_period_stability': 'Sub-Period Stability',
    '4_remove_top3': 'Remove Top 3 Tickers',
    '5_parameter_sensitivity': 'Parameter Sensitivity',
    '6_cost_sensitivity': 'Cost Sensitivity',
}

n_pass = 0
for key, name in test_names.items():
    passed = results['tests'][key]['pass']
    if passed:
        n_pass += 1
    status = "PASS" if passed else "FAIL"
    flush_print(f"  {name}: {status}")

results['summary'] = {
    'tests_passed': n_pass,
    'tests_total': 6,
    'overall': 'PASS' if n_pass >= 5 else 'FAIL',
    'timestamp': datetime.now().isoformat(),
}

flush_print(f"\nOverall: {n_pass}/6 tests passed -> {'PASS' if n_pass >= 5 else 'FAIL'}")

# Save results
out_path = Path("/home/jupiter/Lvl3Quant/data/momentum_after_mr_adversarial.json")
out_path.write_text(json.dumps(results, indent=2, default=str))
flush_print(f"\nResults saved to {out_path}")
