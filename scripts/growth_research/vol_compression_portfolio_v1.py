#!/usr/bin/env python3
"""
Vol Compression Portfolio Simulation v1
========================================
MOTIVATION: vol_compression_magnitude_v1 showed Sharpe 6.06 at 5d hold, 81% WR.
Those are per-trade metrics. Need to validate at PORTFOLIO level:
- Max concurrent positions (capped at 10)
- Proper equity curve with compounding
- Regime classification fixed (scale threshold by hold period)
- Transaction costs (0.1% round trip for stocks)
- Correlation between concurrent positions
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/vol_compression_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("VOL COMPRESSION — PORTFOLIO-LEVEL VALIDATION")
print("="*70)

# Load trades from v1
trades_file = os.path.join(OUTPUT_DIR, 'all_trades.csv')
if not os.path.exists(trades_file):
    # Try Neptune
    trades_file = '/home/nick/Lvl3Quant/output/vol_compression_v1/all_trades.csv'

print(f"\nLoading trades from {trades_file}...")
try:
    trades = pd.read_csv(trades_file, parse_dates=['compression_date', 'entry_date', 'exit_date'])
except Exception as e:
    print(f"ERROR: Could not load trades: {e}")
    print("Run vol_compression_magnitude_v1.py first, then copy output here")
    sys.exit(1)

print(f"  Loaded {len(trades)} trades")

# ─── Config ───
MAX_POSITIONS = 10
COST_PCT = 0.001  # 0.1% round trip (10bps, typical for stocks)
INITIAL_CAPITAL = 100_000
EQUAL_WEIGHT = True  # Equal weight across positions

for hold_period in [5, 10, 21]:
    print(f"\n{'='*60}")
    print(f"  {hold_period}-DAY HOLD PORTFOLIO SIMULATION")
    print(f"{'='*60}")

    hp_trades = trades[trades['hold_period'] == hold_period].copy()
    hp_trades = hp_trades.sort_values('entry_date')
    print(f"  Total trades available: {len(hp_trades)}")

    # ─── Portfolio simulation ───
    capital = INITIAL_CAPITAL
    equity_curve = []
    active_positions = []  # List of (entry_date, exit_date, pnl_pct, ticker, direction)
    trades_taken = []
    skipped_full = 0

    # Get all unique dates where something happens
    all_entry_dates = hp_trades['entry_date'].unique()
    all_exit_dates = hp_trades['exit_date'].unique()
    all_dates = sorted(set(list(all_entry_dates) + list(all_exit_dates)))

    daily_returns = []

    for date in all_dates:
        # Close expired positions
        expired = [p for p in active_positions if p['exit_date'] <= date]
        for p in expired:
            pos_pnl = p['pnl_pct'] - COST_PCT  # Deduct costs
            pos_capital = capital / max(len(active_positions), 1) if EQUAL_WEIGHT else capital
            capital += pos_capital * pos_pnl
            active_positions.remove(p)
            trades_taken.append({**p, 'net_pnl': pos_pnl})

        # Open new positions (up to MAX_POSITIONS)
        new_entries = hp_trades[hp_trades['entry_date'] == date]
        available_slots = MAX_POSITIONS - len(active_positions)

        if available_slots > 0 and len(new_entries) > 0:
            # Prioritize by breakout magnitude (stronger breakouts first)
            new_entries_sorted = new_entries.sort_values('breakout_magnitude', ascending=False)
            for _, trade in new_entries_sorted.head(available_slots).iterrows():
                active_positions.append({
                    'entry_date': trade['entry_date'],
                    'exit_date': trade['exit_date'],
                    'pnl_pct': trade['pnl_pct'],
                    'ticker': trade['ticker'],
                    'direction': trade['direction']
                })
        else:
            skipped_full += len(new_entries)

        equity_curve.append({'date': date, 'capital': capital, 'n_positions': len(active_positions)})

    # Close remaining positions
    for p in active_positions:
        pos_pnl = p['pnl_pct'] - COST_PCT
        pos_capital = capital / max(len(active_positions), 1)
        capital += pos_capital * pos_pnl
        trades_taken.append({**p, 'net_pnl': pos_pnl})

    equity_df = pd.DataFrame(equity_curve)
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    taken_df = pd.DataFrame(trades_taken)

    # ─── Metrics ───
    total_return = (capital / INITIAL_CAPITAL - 1) * 100
    n_years = (equity_df['date'].max() - equity_df['date'].min()).days / 365.25
    cagr = ((capital / INITIAL_CAPITAL) ** (1/n_years) - 1) * 100 if n_years > 0 else 0

    # Daily returns from equity curve
    equity_df['daily_ret'] = equity_df['capital'].pct_change()
    daily_rets = equity_df['daily_ret'].dropna()

    if daily_rets.std() > 0:
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
        neg_std = daily_rets[daily_rets < 0].std()
        sortino = daily_rets.mean() / neg_std * np.sqrt(252) if neg_std > 0 else np.nan
    else:
        sharpe = sortino = 0

    # Drawdown
    equity_df['peak'] = equity_df['capital'].cummax()
    equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak']
    max_dd = equity_df['drawdown'].min() * 100

    # Trade stats
    if len(taken_df) > 0:
        win_rate = (taken_df['net_pnl'] > 0).mean()
        gross_profit = taken_df[taken_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(taken_df[taken_df['net_pnl'] < 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        avg_win = taken_df[taken_df['net_pnl'] > 0]['net_pnl'].mean() * 100
        avg_loss = taken_df[taken_df['net_pnl'] < 0]['net_pnl'].mean() * 100
    else:
        win_rate = profit_factor = 0
        avg_win = avg_loss = 0

    # ─── Regime analysis (FIXED) ───
    # Scale threshold by holding period
    regime_threshold = 0.002 * hold_period  # ~0.2% per day * hold_period
    if len(taken_df) > 0:
        taken_df['spy_ret'] = taken_df.get('spy_ret', pd.Series([np.nan]*len(taken_df)))
        # Re-classify with proper thresholds
        # Use SPY from original trades
        hp_with_spy = hp_trades[['entry_date', 'spy_ret']].drop_duplicates('entry_date')
        taken_df_merged = taken_df.merge(hp_with_spy, on='entry_date', how='left', suffixes=('', '_orig'))
        spy_col = 'spy_ret_orig' if 'spy_ret_orig' in taken_df_merged.columns else 'spy_ret'

        taken_df_merged['regime'] = taken_df_merged[spy_col].apply(
            lambda x: 'GREEN' if x > regime_threshold else ('RED' if x < -regime_threshold else 'FLAT')
        )

        regime_stats = {}
        for regime in ['GREEN', 'RED', 'FLAT']:
            rt = taken_df_merged[taken_df_merged['regime'] == regime]
            if len(rt) > 5:
                wr = (rt['net_pnl'] > 0).mean()
                avg = rt['net_pnl'].mean() * 100
                regime_stats[regime] = {'n': len(rt), 'wr': wr, 'avg_pnl': avg}
    else:
        regime_stats = {}

    # Regime gap
    if 'GREEN' in regime_stats and 'RED' in regime_stats:
        green_wr = regime_stats['GREEN']['wr']
        red_wr = regime_stats['RED']['wr']
        max_wr = max(green_wr, red_wr)
        regime_gap = abs(green_wr - red_wr) / max_wr if max_wr > 0 else 0
    else:
        regime_gap = np.nan

    # ─── Sub-period stability ───
    n_periods = 3
    if len(taken_df) > 30:
        period_size = len(taken_df) // n_periods
        sub_results = []
        for p in range(n_periods):
            sub = taken_df.iloc[p*period_size:(p+1)*period_size]
            sub_wr = (sub['net_pnl'] > 0).mean()
            sub_avg = sub['net_pnl'].mean() * 100
            sub_results.append({'period': p+1, 'n': len(sub), 'wr': sub_wr, 'avg_pnl': sub_avg})
    else:
        sub_results = []

    # ─── Print ───
    print(f"\n  PORTFOLIO METRICS (max {MAX_POSITIONS} concurrent positions):")
    print(f"  Trades taken: {len(taken_df)} (skipped {skipped_full} due to full portfolio)")
    print(f"  Total return: {total_return:.1f}%  |  CAGR: {cagr:.1f}%")
    print(f"  Sharpe: {sharpe:.2f}  |  Sortino: {sortino:.2f}")
    print(f"  Win rate: {win_rate:.0%}  |  PF: {profit_factor:.2f}")
    print(f"  Max DD: {max_dd:.1f}%")
    print(f"  Avg win: +{avg_win:.2f}%  |  Avg loss: {avg_loss:.2f}%")
    print(f"  Calmar: {cagr / abs(max_dd):.2f}" if max_dd != 0 else "  Calmar: inf")

    print(f"\n  REGIME ANALYSIS (threshold: ±{regime_threshold*100:.1f}% SPY over {hold_period}d):")
    for regime, stats in regime_stats.items():
        print(f"    {regime}: {stats['n']} trades, WR {stats['wr']:.0%}, avg PnL {stats['avg_pnl']:.2f}%")
    print(f"  Regime gap: {regime_gap:.2f} ({'PASS' if regime_gap < 0.50 else 'FAIL'})" if not np.isnan(regime_gap) else "  Regime gap: N/A")

    print(f"\n  SUB-PERIOD STABILITY:")
    for sr in sub_results:
        print(f"    P{sr['period']}: {sr['n']} trades, WR {sr['wr']:.0%}, avg PnL {sr['avg_pnl']:.2f}%")

    # ─── Direction analysis ───
    if len(taken_df) > 0:
        for d in ['LONG', 'SHORT']:
            dt = taken_df[taken_df['direction'] == d]
            if len(dt) > 0:
                d_wr = (dt['net_pnl'] > 0).mean()
                d_avg = dt['net_pnl'].mean() * 100
                print(f"  {d}: {len(dt)} trades, WR {d_wr:.0%}, avg {d_avg:.2f}%")

    # Save equity curve
    equity_df.to_csv(f"{OUTPUT_DIR}/equity_curve_{hold_period}d.csv", index=False)

# ─── CONCURRENCY ANALYSIS ───
print(f"\n{'='*60}")
print("CONCURRENCY ANALYSIS")
print(f"{'='*60}")

# Test different max_positions
for max_pos in [3, 5, 10, 20, 50]:
    hp_trades = trades[trades['hold_period'] == 5].sort_values('entry_date')
    capital = INITIAL_CAPITAL
    active = []

    for date in sorted(hp_trades['entry_date'].unique()):
        expired = [p for p in active if p['exit_date'] <= date]
        for p in expired:
            capital += (capital / max(len(active), 1)) * (p['pnl_pct'] - COST_PCT)
            active.remove(p)

        new = hp_trades[hp_trades['entry_date'] == date]
        slots = max_pos - len(active)
        if slots > 0:
            for _, t in new.head(slots).iterrows():
                active.append({'exit_date': t['exit_date'], 'pnl_pct': t['pnl_pct']})

    for p in active:
        capital += (capital / max(len(active), 1)) * (p['pnl_pct'] - COST_PCT)

    total_ret = (capital / INITIAL_CAPITAL - 1) * 100
    n_yrs = (hp_trades['exit_date'].max() - hp_trades['entry_date'].min()).days / 365.25
    cagr = ((capital / INITIAL_CAPITAL) ** (1/n_yrs) - 1) * 100 if n_yrs > 0 else 0
    print(f"  MaxPos={max_pos:2d}: Total {total_ret:7.1f}%, CAGR {cagr:5.1f}%")

print(f"\nCompleted: {datetime.now()}")
