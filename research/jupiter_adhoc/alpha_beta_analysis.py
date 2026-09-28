#!/usr/bin/env python3
"""
Alpha vs Beta Analysis for WF Fill Sim Sweep Results
Determines if CNN model generates real alpha or just rides market beta.
"""

import os
import json
import re
import sys
from collections import defaultdict
import math

RESULTS_DIR = os.path.expanduser('~/Lvl3Quant/alpha_discovery/results/wf_sweep/')

# ── Load all result files ──────────────────────────────────────────────────────

print("=" * 70)
print("ALPHA VS BETA ANALYSIS — WF Fill Sim Sweep")
print("=" * 70)
print()

files = os.listdir(RESULTS_DIR)
json_files = [f for f in files if f.endswith('.json') and re.search(r'\d{4}-\d{2}-\d{2}', f)]
print(f"Total JSON result files: {len(json_files)}")

# Group by date
date_files = defaultdict(list)
for f in json_files:
    m = re.search(r'(\d{4}-\d{2}-\d{2})', f)
    if m:
        date_files[m.group(1)].append(f)

dates = sorted(date_files.keys())
print(f"Trading days in sweep: {len(dates)}")
for d in dates:
    print(f"  {d}: {len(date_files[d])} configs")
print()

# ── Load all trades from all configs ──────────────────────────────────────────

# Per-day data: list of (config_name, pnl, trades)
day_data = {}  # date -> {configs: [...], all_trades: [...], total_pnl_sum, config_count}

total_loaded = 0
total_errors = 0

for date in dates:
    day_data[date] = {
        'configs': [],
        'all_trades': [],
        'total_pnl_sum': 0.0,
        'config_count': 0,
        'n_trades_sum': 0,
    }
    for fname in date_files[date]:
        fpath = os.path.join(RESULTS_DIR, fname)
        try:
            with open(fpath) as f:
                result = json.load(f)
            total_loaded += 1
            pnl = result.get('total_pnl_dollars', 0.0) or 0.0
            trades = result.get('trades', [])
            day_data[date]['configs'].append({
                'file': fname,
                'pnl': pnl,
                'n_trades': len(trades),
                'config': result.get('config', {}),
            })
            day_data[date]['total_pnl_sum'] += pnl
            day_data[date]['config_count'] += 1
            day_data[date]['n_trades_sum'] += len(trades)
            day_data[date]['all_trades'].extend(trades)
        except Exception as e:
            total_errors += 1

print(f"Loaded {total_loaded} configs, {total_errors} errors")
print()

# ── Compute market direction per day from trade entry prices ──────────────────
# ES daily return: use first and last entry price as proxy for open/close

print("=" * 70)
print("MARKET DIRECTION ANALYSIS (from trade entry prices)")
print("=" * 70)
print()

day_stats = {}

for date in dates:
    trades = day_data[date]['all_trades']
    if not trades:
        continue

    # Sort by signal time
    trades_sorted = sorted(trades, key=lambda t: t.get('signal_time_ns', 0))

    # Get all entry prices + timestamps
    entry_prices = [(t['signal_time_ns'], t['entry_price']) for t in trades_sorted if 'entry_price' in t]
    exit_prices = [(t['exit_time_ns'], t['exit_price']) for t in trades_sorted if 'exit_price' in t]

    if not entry_prices:
        continue

    # Day open = first entry price, day close = last exit price
    day_open = entry_prices[0][1]
    day_close = exit_prices[-1][1] if exit_prices else entry_prices[-1][1]
    market_return_ticks = day_close - day_open
    market_return_pct = market_return_ticks / day_open * 100

    # Strategy P&L: average across all configs for this day (mean pnl per config)
    n_configs = day_data[date]['config_count']
    avg_pnl = day_data[date]['total_pnl_sum'] / n_configs if n_configs > 0 else 0.0
    total_pnl = day_data[date]['total_pnl_sum']

    # Naive long-only: buy at open, sell at close, 1 contract
    # ES tick = $12.50, so each point = $50
    naive_long_pnl = market_return_ticks * 50  # dollars per contract

    day_stats[date] = {
        'day_open': day_open,
        'day_close': day_close,
        'market_return_ticks': market_return_ticks,
        'market_return_pct': market_return_pct,
        'market_up': market_return_ticks > 0,
        'avg_config_pnl': avg_pnl,
        'total_pnl_all_configs': total_pnl,
        'naive_long_pnl': naive_long_pnl,
        'n_configs': n_configs,
        'n_trades_total': day_data[date]['n_trades_sum'],
    }

print(f"{'Date':<12} {'Open':>8} {'Close':>8} {'Mkt Ret':>10} {'Mkt Dir':>8} {'AvgCfgPnL':>12} {'Naive Long':>12}")
print("-" * 72)
for date, s in sorted(day_stats.items()):
    direction = 'UP  ' if s['market_up'] else 'DOWN'
    print(f"{date:<12} {s['day_open']:>8.2f} {s['day_close']:>8.2f} {s['market_return_ticks']:>+10.2f} {direction:>8} {s['avg_config_pnl']:>+12.2f} {s['naive_long_pnl']:>+12.2f}")
print()

# ── 1. Market direction vs P&L correlation ───────────────────────────────────

print("=" * 70)
print("TEST 1: MARKET DIRECTION vs STRATEGY P&L CORRELATION")
print("=" * 70)
print()

valid_days = [(d, s) for d, s in day_stats.items() if s['market_return_ticks'] != 0]
if len(valid_days) >= 3:
    xs = [s['market_return_ticks'] for _, s in valid_days]
    ys = [s['avg_config_pnl'] for _, s in valid_days]

    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom_x > 0 and denom_y > 0:
        corr = num / (denom_x * denom_y)
    else:
        corr = 0.0

    print(f"Correlation (market return vs avg config P&L): {corr:+.4f}")
    if abs(corr) > 0.7:
        print(f"  *** WARNING: High correlation ({corr:.2f}) — likely BETA-driven ***")
    elif abs(corr) > 0.4:
        print(f"  MODERATE correlation ({corr:.2f}) — mixed alpha/beta")
    else:
        print(f"  LOW correlation ({corr:.2f}) — suggests alpha (market-direction-independent)")
    print()

# ── 2. Naive long-only benchmark ─────────────────────────────────────────────

print("=" * 70)
print("TEST 2: NAIVE LONG-ONLY BENCHMARK")
print("=" * 70)
print()

print(f"{'Date':<12} {'Mkt Dir':>8} {'Naive L/O':>12} {'Avg Strat':>12} {'Strat Wins':>12}")
print("-" * 56)
strat_beats_naive = 0
days_mkt_up = 0
days_strat_up = 0

for date, s in sorted(day_stats.items()):
    mkt_dir = 'UP  ' if s['market_up'] else 'DOWN'
    strat_pnl = s['avg_config_pnl']
    naive_pnl = s['naive_long_pnl']
    beats = strat_pnl > naive_pnl
    if beats:
        strat_beats_naive += 1
    if s['market_up']:
        days_mkt_up += 1
    if strat_pnl > 0:
        days_strat_up += 1
    beat_str = 'YES' if beats else 'NO '
    print(f"{date:<12} {mkt_dir:>8} {naive_pnl:>+12.2f} {strat_pnl:>+12.2f} {beat_str:>12}")

total_days = len(day_stats)
print()
print(f"Days strategy beats naive long: {strat_beats_naive}/{total_days} ({100*strat_beats_naive/total_days:.1f}%)")
print(f"Market UP days: {days_mkt_up}/{total_days} ({100*days_mkt_up/total_days:.1f}%)")
print(f"Strategy profitable days: {days_strat_up}/{total_days} ({100*days_strat_up/total_days:.1f}%)")
print()

# ── 3. Green day overlap ──────────────────────────────────────────────────────

print("=" * 70)
print("TEST 3: GREEN DAY OVERLAP (market up AND strategy profitable)")
print("=" * 70)
print()

both_green = sum(1 for _, s in day_stats.items() if s['market_up'] and s['avg_config_pnl'] > 0)
mkt_green_strat_red = sum(1 for _, s in day_stats.items() if s['market_up'] and s['avg_config_pnl'] <= 0)
mkt_red_strat_green = sum(1 for _, s in day_stats.items() if not s['market_up'] and s['avg_config_pnl'] > 0)
both_red = sum(1 for _, s in day_stats.items() if not s['market_up'] and s['avg_config_pnl'] <= 0)

print(f"  Market UP   + Strategy PROFIT:  {both_green} days")
print(f"  Market UP   + Strategy LOSS:    {mkt_green_strat_red} days")
print(f"  Market DOWN + Strategy PROFIT:  {mkt_red_strat_green} days  ← KEY METRIC")
print(f"  Market DOWN + Strategy LOSS:    {both_red} days")
print()

down_days = sum(1 for _, s in day_stats.items() if not s['market_up'])
if down_days > 0:
    pct_win_on_down = mkt_red_strat_green / down_days * 100
    print(f"Win rate on DOWN days: {mkt_red_strat_green}/{down_days} = {pct_win_on_down:.1f}%")
    if mkt_red_strat_green == 0:
        print("  *** ALERT: Strategy NEVER profitable on down days — strong beta signal ***")
    elif pct_win_on_down >= 50:
        print("  GOOD: Strategy wins on ≥50% of down days — suggests alpha")
    else:
        print("  CONCERNING: Strategy losing on majority of down days")
else:
    print("  No market DOWN days in sample!")
print()

# ── 4. Beta-adjusted returns (OLS regression) ─────────────────────────────────

print("=" * 70)
print("TEST 4: BETA-ADJUSTED RETURNS (OLS regression)")
print("=" * 70)
print()

if len(valid_days) >= 3:
    xs = [s['market_return_ticks'] for _, s in valid_days]
    ys = [s['avg_config_pnl'] for _, s in valid_days]
    n = len(xs)

    # OLS: y = alpha + beta * x
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))

    beta = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    alpha = (sy - beta * sx) / n

    # R²
    y_pred = [alpha + beta * x for x in xs]
    ss_res = sum((y - yp) ** 2 for y, yp in zip(ys, y_pred))
    ss_tot = sum((y - (sy / n)) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # t-stat for alpha
    if n > 2 and ss_res > 0:
        s2 = ss_res / (n - 2)
        # Var(alpha) = s2 * (1/n + mean_x^2 / Sxx)
        mean_x = sx / n
        Sxx = sxx - n * mean_x ** 2
        if Sxx > 0:
            var_alpha = s2 * (1.0 / n + mean_x ** 2 / Sxx)
            se_alpha = math.sqrt(var_alpha)
            t_alpha = alpha / se_alpha if se_alpha > 0 else 0.0
        else:
            t_alpha = 0.0
    else:
        t_alpha = 0.0

    print(f"OLS Regression: Strategy_PnL = Alpha + Beta * Market_Return")
    print(f"  Alpha (daily edge, $): {alpha:+.4f}")
    print(f"  Beta (market exposure): {beta:+.4f}")
    print(f"  R² (% explained by mkt): {r2:.4f} ({100*r2:.1f}%)")
    print(f"  t-stat for Alpha: {t_alpha:+.3f}")
    print()

    if alpha > 0 and t_alpha > 2.0:
        print("  POSITIVE ALPHA: Statistically significant positive intercept — REAL ALPHA")
    elif alpha > 0:
        print(f"  Positive alpha but t={t_alpha:.2f} — insufficient data for significance")
    else:
        print(f"  *** NEGATIVE ALPHA (alpha={alpha:.2f}) — strategy loses money independent of market ***")

    if abs(r2) > 0.5:
        print(f"  *** HIGH R²={r2:.2f}: >50% of P&L explained by market direction — BETA ***")
    elif abs(r2) > 0.2:
        print(f"  MODERATE R²={r2:.2f}: Some beta exposure, some alpha")
    else:
        print(f"  LOW R²={r2:.2f}: Market direction explains little of strategy P&L — ALPHA")
    print()

# ── 5. Direction-neutral test (contra-trend trades) ──────────────────────────

print("=" * 70)
print("TEST 5: DIRECTION-NEUTRAL (CONTRA-TREND TRADES)")
print("=" * 70)
print()

# For each day, compute market direction from start of day to trade entry time
# (use first entry price as "day open", compare each trade's entry relative to that)

contra_trades_pnl = []
with_trend_trades_pnl = []
contra_wins = 0
with_wins = 0
contra_count = 0
with_count = 0

for date, s in day_stats.items():
    trades = day_data[date]['all_trades']
    if not trades:
        continue

    trades_sorted = sorted(trades, key=lambda t: t.get('signal_time_ns', 0))
    if not trades_sorted:
        continue

    # Day open = first entry price
    day_open_price = s['day_open']

    for trade in trades_sorted:
        if 'entry_price' not in trade or 'pnl_dollars' not in trade or 'side' not in trade:
            continue

        entry = trade['entry_price']
        pnl = trade['pnl_dollars']
        side = trade['side']  # BUY or SELL

        # Market direction at entry: price vs day open
        mkt_up_at_entry = entry > day_open_price

        # Is this trade with-trend or contra-trend?
        # With-trend: BUY on up day, SELL on down day
        # Contra-trend: BUY on down day, SELL on up day
        is_with_trend = (side == 'BUY' and mkt_up_at_entry) or (side == 'SELL' and not mkt_up_at_entry)

        if is_with_trend:
            with_trend_trades_pnl.append(pnl)
            with_count += 1
            if pnl > 0:
                with_wins += 1
        else:
            contra_trades_pnl.append(pnl)
            contra_count += 1
            if pnl > 0:
                contra_wins += 1

print(f"With-trend trades: {with_count}")
if with_count > 0:
    wt_avg = sum(with_trend_trades_pnl) / with_count
    wt_wr = with_wins / with_count * 100
    wt_total = sum(with_trend_trades_pnl)
    print(f"  Total P&L: ${wt_total:+,.2f}")
    print(f"  Avg P&L/trade: ${wt_avg:+.2f}")
    print(f"  Win rate: {wt_wr:.1f}%")
print()

print(f"Contra-trend trades: {contra_count}")
if contra_count > 0:
    ct_avg = sum(contra_trades_pnl) / contra_count
    ct_wr = contra_wins / contra_count * 100
    ct_total = sum(contra_trades_pnl)
    print(f"  Total P&L: ${ct_total:+,.2f}")
    print(f"  Avg P&L/trade: ${ct_avg:+.2f}")
    print(f"  Win rate: {ct_wr:.1f}%")
    print()
    if ct_total > 0 and ct_wr > 50:
        print("  *** REAL ALPHA: Contra-trend trades are PROFITABLE — model adds value beyond beta ***")
    elif ct_total > 0:
        print("  MODERATE: Contra-trend trades profit but below 50% WR — mixed signal")
    else:
        print("  *** BETA CONCERN: Contra-trend trades LOSE — model may be trend-following ***")
else:
    print("  No contra-trend trades found")
print()

# ── 6. Random entry benchmark ─────────────────────────────────────────────────

print("=" * 70)
print("TEST 6: RANDOM ENTRY BENCHMARK")
print("=" * 70)
print()

# Random entry: 50/50 long/short, same hold period, same fills
# Expected P&L for random entry:
#   E[P&L_long_random] = avg_price_move_in_hold_period - cost
#   E[P&L_short_random] = -avg_price_move_in_hold_period - cost
#   E[P&L_random_50/50] = -cost (commission + half-spread)

all_trades_flat = []
for date in dates:
    all_trades_flat.extend(day_data[date]['all_trades'])

if all_trades_flat:
    # Actual trade metrics
    total_actual_pnl = sum(t['pnl_dollars'] for t in all_trades_flat if 'pnl_dollars' in t)
    n_actual = len(all_trades_flat)
    buy_trades = [t for t in all_trades_flat if t.get('side') == 'BUY']
    sell_trades = [t for t in all_trades_flat if t.get('side') == 'SELL']

    # Price moves for each trade
    buy_moves = []
    sell_moves = []
    for t in all_trades_flat:
        if 'entry_price' not in t or 'exit_price' not in t:
            continue
        move = t['exit_price'] - t['entry_price']  # positive = price went up
        if t.get('side') == 'BUY':
            buy_moves.append(move)
        else:
            sell_moves.append(move)

    all_moves = buy_moves + sell_moves
    if all_moves:
        avg_abs_move = sum(abs(m) for m in all_moves) / len(all_moves)
        avg_signed_move = sum(all_moves) / len(all_moves)

        # Random 50/50: E[P&L] per trade = avg_signed_move * 0 (cancels) - commission_cost
        # But we can estimate: if long 50% and short 50%:
        # E[P&L] = 0.5 * (avg_move * 50) + 0.5 * (-avg_move * 50) - commission
        # = -commission
        # Commission is already in pnl_ticks. Let's compute from data.
        avg_commission = 0.24 * 12.5  # 0.24 ticks * $12.5/tick = $3 per trade

        random_expected_pnl_per_trade = -avg_commission  # Pure cost
        random_total_pnl = random_expected_pnl_per_trade * n_actual

        print(f"Actual trades analyzed: {n_actual}")
        print(f"  Buy trades: {len(buy_trades)}, Sell trades: {len(sell_trades)}")
        print(f"  L/S ratio: {len(buy_trades)/n_actual*100:.1f}% longs")
        print(f"  Avg price move per trade: {avg_signed_move:+.4f} pts")
        print(f"  Avg |price move|: {avg_abs_move:.4f} pts")
        print()
        print(f"Actual total P&L (all trades, all configs): ${total_actual_pnl:+,.2f}")
        print(f"Random entry expected P&L (−commission only): ${random_total_pnl:+,.2f}")
        print(f"Edge above random: ${total_actual_pnl - random_total_pnl:+,.2f}")
        print()

        # Key test: L/S ratio vs market direction
        if len(buy_trades) / n_actual > 0.65:
            print("  *** ALERT: Highly skewed to LONG (>65%) — potential long bias / beta ***")
        elif len(buy_trades) / n_actual < 0.35:
            print("  *** ALERT: Highly skewed to SHORT (<35%) — potential short bias ***")
        else:
            print(f"  L/S ratio is balanced ({len(buy_trades)/n_actual*100:.0f}%/{len(sell_trades)/n_actual*100:.0f}%) — direction-balanced model")
        print()

# ── 7. Long-only every-signal benchmark ──────────────────────────────────────

print("=" * 70)
print("TEST 7: LONG-ONLY EVERY SIGNAL BENCHMARK")
print("=" * 70)
print()

if all_trades_flat:
    # What if every signal was taken as BUY (ignore model direction)?
    all_price_moves = []
    actual_model_pnl = []

    for t in all_trades_flat:
        if 'entry_price' not in t or 'exit_price' not in t or 'pnl_ticks' not in t:
            continue
        move_ticks = (t['exit_price'] - t['entry_price']) * 4  # 1 ES point = 4 ticks
        commission = 0.24  # ticks
        spread_cost = 0.0  # already in pnl for limit orders
        if t.get('side') == 'BUY':
            long_pnl_ticks = move_ticks - commission
        else:
            long_pnl_ticks = -move_ticks - commission  # if we went long instead of short
        all_price_moves.append(long_pnl_ticks * 12.5)  # convert to dollars
        actual_model_pnl.append(t['pnl_dollars'])

    if all_price_moves:
        long_only_total = sum(all_price_moves)
        actual_total = sum(actual_model_pnl)
        long_only_wr = sum(1 for p in all_price_moves if p > 0) / len(all_price_moves) * 100
        actual_wr = sum(1 for p in actual_model_pnl if p > 0) / len(actual_model_pnl) * 100

        print(f"Trades with complete data: {len(all_price_moves)}")
        print()
        print(f"  Actual model P&L (correct L/S): ${actual_total:+,.2f}  WR: {actual_wr:.1f}%")
        print(f"  Long-only every signal:          ${long_only_total:+,.2f}  WR: {long_only_wr:.1f}%")
        print(f"  Model advantage over long-only:  ${actual_total - long_only_total:+,.2f}")
        print()

        if actual_total > long_only_total * 1.1:
            print("  GOOD: Model significantly outperforms long-only — directional signal ADDS VALUE")
        elif abs(actual_total - long_only_total) < abs(long_only_total) * 0.1:
            print("  *** CONCERNING: Model performs SIMILARLY to long-only — directional signal WORTHLESS ***")
        elif long_only_total > actual_total:
            print("  *** ALERT: Long-only BEATS the model — model's short calls are destroying value ***")
        else:
            print("  MARGINAL: Model slightly outperforms long-only")
        print()

        # Break down BUY vs SELL performance
        buy_pnl = sum(t['pnl_dollars'] for t in all_trades_flat if t.get('side') == 'BUY' and 'pnl_dollars' in t)
        sell_pnl = sum(t['pnl_dollars'] for t in all_trades_flat if t.get('side') == 'SELL' and 'pnl_dollars' in t)
        buy_n = len([t for t in all_trades_flat if t.get('side') == 'BUY'])
        sell_n = len([t for t in all_trades_flat if t.get('side') == 'SELL'])

        print(f"BUY trades:  n={buy_n}  Total P&L: ${buy_pnl:+,.2f}  Avg: ${buy_pnl/buy_n:+.2f}" if buy_n > 0 else "No BUY trades")
        print(f"SELL trades: n={sell_n}  Total P&L: ${sell_pnl:+,.2f}  Avg: ${sell_pnl/sell_n:+.2f}" if sell_n > 0 else "No SELL trades")
        print()
        if buy_n > 0 and sell_n > 0:
            if buy_pnl > 0 and sell_pnl > 0:
                print("  STRONG ALPHA: Both BUY and SELL trades are profitable — truly directional")
            elif buy_pnl > 0 and sell_pnl <= 0:
                print("  *** BETA CONCERN: Only BUY trades profit — long bias, likely beta ***")
            elif buy_pnl <= 0 and sell_pnl > 0:
                print("  *** UNUSUAL: Only SELL trades profit — short bias ***")
            else:
                print("  *** BOTH SIDES LOSING — edge not there regardless ***")

# ── SUMMARY ──────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("SUMMARY: ALPHA vs BETA VERDICT")
print("=" * 70)
print()

# Collect signals
signals = []

if len(valid_days) >= 3:
    if abs(corr) > 0.7:
        signals.append(("BETA", f"Market correlation = {corr:.2f} (>0.7 threshold)"))
    elif abs(corr) < 0.3:
        signals.append(("ALPHA", f"Market correlation = {corr:.2f} (low — direction-independent)"))
    else:
        signals.append(("MIXED", f"Market correlation = {corr:.2f} (moderate)"))

    if r2 > 0.5:
        signals.append(("BETA", f"R² = {r2:.2f} — market explains >{50}% of P&L"))
    elif r2 < 0.2:
        signals.append(("ALPHA", f"R² = {r2:.2f} — market explains <20% of P&L"))
    else:
        signals.append(("MIXED", f"R² = {r2:.2f}"))

    if alpha > 0 and t_alpha > 1.5:
        signals.append(("ALPHA", f"Positive regression alpha = {alpha:.2f}, t = {t_alpha:.2f}"))
    elif alpha < 0:
        signals.append(("BETA", f"Negative regression alpha = {alpha:.2f}"))

if down_days > 0:
    if mkt_red_strat_green == 0:
        signals.append(("BETA", "0% win rate on down days — pure trend-follower"))
    elif mkt_red_strat_green / down_days >= 0.5:
        signals.append(("ALPHA", f"{mkt_red_strat_green/down_days*100:.0f}% win rate on down days"))

if all_trades_flat and len(buy_trades) / n_actual > 0.65:
    signals.append(("BETA", f"Long-biased ({len(buy_trades)/n_actual*100:.0f}% longs)"))

if 'ct_total' in dir() and contra_count > 0:
    if ct_total > 0:
        signals.append(("ALPHA", f"Contra-trend trades profitable (${ct_total:+,.0f})"))
    else:
        signals.append(("BETA", f"Contra-trend trades lose (${ct_total:+,.0f})"))

alpha_signals = sum(1 for s in signals if s[0] == "ALPHA")
beta_signals = sum(1 for s in signals if s[0] == "BETA")
mixed_signals = sum(1 for s in signals if s[0] == "MIXED")

print(f"Evidence ALPHA: {alpha_signals} signals")
print(f"Evidence BETA:  {beta_signals} signals")
print(f"Mixed:          {mixed_signals} signals")
print()
for verdict, detail in signals:
    icon = "✓" if verdict == "ALPHA" else ("✗" if verdict == "BETA" else "~")
    print(f"  [{verdict}] {detail}")
print()

if beta_signals > alpha_signals:
    print("VERDICT: *** PRIMARILY BETA — strategy is riding market direction, not generating alpha ***")
    print("RECOMMENDATION: Run analysis on down-trending period to confirm. Consider market-neutral strategy.")
elif alpha_signals > beta_signals:
    print("VERDICT: *** PRIMARILY ALPHA — strategy generates returns independent of market direction ***")
    print("RECOMMENDATION: Strategy passes alpha test. Proceed with paper trading.")
else:
    print("VERDICT: MIXED — insufficient evidence to distinguish alpha from beta conclusively")
    print("RECOMMENDATION: Need more data (more trading days) and/or down-market period for definitive test.")

print()
print("=" * 70)
print("Analysis complete.")
