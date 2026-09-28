"""
MAX PROFIT ANALYSIS — ES Futures MBO Feature Cache
====================================================
Computes the theoretical maximum extractable profit from perfect directional
prediction at various trading horizons.

Key parameters:
  - ES futures (full-size), tick size = 0.25 pts, tick value = $12.50/contract
  - Mid price used as synthetic execution price (bid/ask midpoint)
  - Spread cost = 0.25 pts = $12.50 (1-tick market impact per entry)
  - Commission = $3.00 per round-trip
  - Total cost per round-trip = $12.50 + $3.00 = $15.50
  - Bars = 100ms each
  - 198,000 bars/day = 5.5 hours of data per day

Horizon bars:
  - 100ms  → 1 bar     (omniscient every tick)
  - 1s     → 10 bars
  - 3s     → 30 bars
  - 5s     → 50 bars
  - 10s    → 100 bars
  - 30s    → 300 bars
  - 1min   → 600 bars
  - 5min   → 3000 bars
"""

import numpy as np
import os
import sys
from datetime import datetime

# ============================================================
#  PARAMETERS
# ============================================================
TICK_SIZE          = 0.25    # points (ES full-size tick)
TICK_VALUE         = 12.50   # $ per tick per contract (ES full-size: $50/pt × 0.25)
POINT_VALUE        = 50.0    # $ per point per contract
SPREAD_COST_PTS    = 0.0     # HC #290(C) 2026-05-11: NO synthetic spread cost. Fill price IS the price regardless of order type.
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)    # $ per round-trip
SPREAD_COST_DOLLAR = 0.0  # HC #290(C): zero — commission only
TOTAL_COST_RT      = COMMISSION_RT  # HC #290(C): commission only = $4.70 = 0.376 ticks
BAR_MS             = 100     # milliseconds per bar

HORIZONS = {
    '100ms (1bar)' : 1,
    '1s  (10bar)'  : 10,
    '3s  (30bar)'  : 30,
    '5s  (50bar)'  : 50,
    '10s (100bar)' : 100,
    '30s (300bar)' : 300,
    '1min(600bar)' : 600,
    '5min(3000bar)': 3000,
}

# Flat day threshold: if day price range < 1 pt, skip (weekend/holiday)
FLAT_DAY_THRESHOLD_PTS = 1.0

# ============================================================
#  LOAD DATA
# ============================================================
cache_path = 'alpha_discovery/results/feature_cache/features_alldays.npz'
print(f"Loading feature cache: {cache_path}")
data = np.load(cache_path, allow_pickle=True)

mid   = data['mid_prices'].astype(np.float64)   # (9702000,)
db    = data['day_boundaries']                   # (50,) — 49 days
n_days = len(db) - 1

print(f"Total bars: {len(mid):,}")
print(f"Total days in cache: {n_days}")
print(f"Bars per day: {db[1]-db[0]:,}")
print(f"Bar resolution: {BAR_MS}ms")
print()

# ============================================================
#  PER-DAY ANALYSIS
# ============================================================

results = []   # list of dicts, one per trading day

print("="*80)
print("COMPUTING PER-DAY MAX PROFIT...")
print("="*80)

for day_idx in range(n_days):
    start = db[day_idx]
    end   = db[day_idx + 1]
    mid_day = mid[start:end]

    # Skip flat/weekend days
    price_range = mid_day.max() - mid_day.min()
    if price_range < FLAT_DAY_THRESHOLD_PTS:
        results.append({
            'day': day_idx + 1,
            'flat': True,
            'range_pts': price_range,
        })
        continue

    n_bars = len(mid_day)
    diffs  = np.diff(mid_day)           # (n_bars-1,) price changes per bar

    # ----------------------------------------------------------
    # METRIC 1: Omniscient 100ms trader (zero cost)
    # Trade at every bar boundary if direction is non-zero.
    # Raw gross = sum of abs(tick changes).
    # This is the absolute ceiling with zero latency + zero cost.
    # ----------------------------------------------------------
    total_abs_move_pts  = np.sum(np.abs(diffs))
    total_abs_move_dolr = total_abs_move_pts * POINT_VALUE

    # Count direction changes (number of times the trend reverses)
    signs = np.sign(diffs[diffs != 0])
    n_direction_changes = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
    n_nonzero_bars = int(np.sum(diffs != 0))

    # ----------------------------------------------------------
    # METRIC 2: Omniscient with realistic costs
    # Every direction change requires a round-trip close + new open.
    # n_trades ≈ n_direction_changes (each reversal = 1 RT)
    # Plus the initial entry and final exit.
    # ----------------------------------------------------------
    n_trades_100ms = n_direction_changes + 1  # +1 for the initial entry
    gross_100ms = total_abs_move_dolr
    net_100ms   = gross_100ms - n_trades_100ms * TOTAL_COST_RT

    # ----------------------------------------------------------
    # METRIC 3: Horizon-based perfect oracle
    # At each horizon H bars, look exactly H bars ahead.
    # If you know the direction of the next H bars:
    #   - Enter at current mid (pay spread on entry)
    #   - Exit at mid[t+H] (pay commission)
    # Profit per trade = abs(mid[t+H] - mid[t]) * POINT_VALUE
    #                    - TOTAL_COST_RT
    # Strategy: trade every H bars (non-overlapping windows).
    # Only take a trade if the expected move > total cost.
    # ----------------------------------------------------------
    horizon_results = {}
    for h_name, h_bars in HORIZONS.items():
        if h_bars >= n_bars:
            horizon_results[h_name] = {
                'gross': 0, 'net': 0, 'n_trades': 0,
                'n_profitable': 0, 'win_rate': 0, 'avg_move_pts': 0,
            }
            continue

        # Non-overlapping windows: trade at bar 0, h, 2h, ...
        entry_idx = np.arange(0, n_bars - h_bars, h_bars)
        exit_idx  = entry_idx + h_bars

        moves_pts  = np.abs(mid_day[exit_idx] - mid_day[entry_idx])
        gross_each = moves_pts * POINT_VALUE
        net_each   = gross_each - TOTAL_COST_RT

        # Take ALL trades (perfect oracle always knows direction)
        n_trades      = len(entry_idx)
        gross_total   = float(np.sum(gross_each))
        net_total     = float(np.sum(net_each))        # can be negative if many tiny moves
        n_profitable  = int(np.sum(net_each > 0))
        avg_move      = float(np.mean(moves_pts))

        # Alternative: only trade if move > cost threshold
        # (oracle knows which windows are worthwhile)
        net_profitable_only = float(np.sum(net_each[net_each > 0]))
        n_trades_filtered   = int(np.sum(net_each > 0))

        horizon_results[h_name] = {
            'gross'               : gross_total,
            'net'                 : net_total,
            'net_profitable_only' : net_profitable_only,
            'n_trades'            : n_trades,
            'n_profitable'        : n_profitable,
            'n_trades_filtered'   : n_trades_filtered,
            'win_rate'            : n_profitable / n_trades * 100 if n_trades > 0 else 0,
            'avg_move_pts'        : avg_move,
            'cost_per_trade'      : TOTAL_COST_RT,
        }

    results.append({
        'day'                 : day_idx + 1,
        'flat'                : False,
        'range_pts'           : price_range,
        'n_bars'              : n_bars,
        'total_abs_move_pts'  : total_abs_move_pts,
        'total_abs_move_dolr' : total_abs_move_dolr,
        'n_nonzero_bars'      : n_nonzero_bars,
        'n_direction_changes' : n_direction_changes,
        'n_trades_100ms'      : n_trades_100ms,
        'gross_100ms'         : gross_100ms,
        'net_100ms'           : net_100ms,
        'horizons'            : horizon_results,
    })

    pct = (day_idx + 1) / n_days * 100
    print(f"  Day {day_idx+1:2d}/{n_days}: range={price_range:5.2f}pts  "
          f"abs_move={total_abs_move_pts:7.2f}pts  "
          f"gross_oracle=${total_abs_move_dolr:>10,.0f}  "
          f"net_100ms=${net_100ms:>10,.0f}")

# ============================================================
#  AGGREGATE STATISTICS
# ============================================================
trading_days = [r for r in results if not r.get('flat', False)]
flat_days    = [r for r in results if r.get('flat', False)]

print()
print(f"Trading days: {len(trading_days)}, Flat/weekend days: {len(flat_days)}")

# ============================================================
#  BUILD OUTPUT REPORT
# ============================================================
lines = []
def L(s=''):
    lines.append(s)
    print(s)

L("="*80)
L("  MAX PROFIT ANALYSIS — ES Futures MBO Feature Cache")
L(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
L("="*80)
L()
L("PARAMETERS:")
L(f"  Instrument        : ES futures (full-size, $50/pt)")
L(f"  Tick size         : {TICK_SIZE} pts = ${TICK_VALUE:.2f} per tick")
L(f"  Spread cost       : {SPREAD_COST_PTS} pts = ${SPREAD_COST_DOLLAR:.2f} (1-tick market order)")
L(f"  Commission        : ${COMMISSION_RT:.2f} per round-trip")
L(f"  Total cost/trade  : ${TOTAL_COST_RT:.2f} per round-trip")
L(f"  Bar resolution    : {BAR_MS}ms")
L(f"  Total bars        : {len(mid):,}")
L(f"  Total days        : {n_days} ({len(trading_days)} active, {len(flat_days)} flat/weekend)")
L()

# ----------------------------------------------------------
# TABLE 1: Per-day omniscient oracle summary
# ----------------------------------------------------------
L("="*80)
L("TABLE 1: PER-DAY OMNISCIENT ORACLE (100ms bar trading)")
L("="*80)
L(f"{'Day':>4}  {'Range':>6}  {'AbsMove':>8}  {'Gross($)':>12}  {'#Trades':>8}  {'Net($) 0-cost':>14}  {'Net($) w/cost':>14}")
L(f"{'':>4}  {'(pts)':>6}  {'(pts)':>8}  {'':>12}  {'100ms RT':>8}  {'':>14}  {'':>14}")
L("-"*90)

total_gross_100ms = 0
total_net_100ms   = 0
total_trades_100ms = 0
for r in trading_days:
    L(f"{r['day']:4d}  {r['range_pts']:6.2f}  {r['total_abs_move_pts']:8.2f}  "
      f"${r['total_abs_move_dolr']:>11,.0f}  {r['n_trades_100ms']:>8,}  "
      f"${r['total_abs_move_dolr']:>13,.0f}  ${r['net_100ms']:>13,.0f}")
    total_gross_100ms  += r['total_abs_move_dolr']
    total_net_100ms    += r['net_100ms']
    total_trades_100ms += r['n_trades_100ms']

L("-"*90)
n_td = len(trading_days)
L(f"{'TOTAL':>4}  {'':>6}  {'':>8}  ${total_gross_100ms:>11,.0f}  {total_trades_100ms:>8,}  "
  f"${total_gross_100ms:>13,.0f}  ${total_net_100ms:>13,.0f}")
L(f"{'AVG/day':>4}  {'':>6}  {'':>8}  ${total_gross_100ms/n_td:>11,.0f}  {total_trades_100ms/n_td:>8,.0f}  "
  f"${total_gross_100ms/n_td:>13,.0f}  ${total_net_100ms/n_td:>13,.0f}")
L()

# ----------------------------------------------------------
# TABLE 2: Max profit by horizon
# ----------------------------------------------------------
L("="*80)
L("TABLE 2: MAX PROFIT BY TRADING HORIZON (perfect oracle, all trades)")
L("  Strategy: Non-overlapping windows. Oracle always knows direction.")
L(f"  Cost per trade: ${TOTAL_COST_RT:.2f} (${SPREAD_COST_DOLLAR:.2f} spread + ${COMMISSION_RT:.2f} commission)")
L("="*80)

horizon_names = list(HORIZONS.keys())

# Header
hdr_cols = ['Horizon', 'Gross($)', 'Net($) All', 'Net($) Filtered', '#Trades', '#Profitable', 'Win%', 'AvgMove', '% of 100ms Max']
L(f"{'Horizon':<18}  {'Gross($)':>12}  {'Net($)All':>12}  {'Net($)Filter':>13}  {'#Trades':>9}  {'#Profit':>9}  {'Win%':>6}  {'AvgMove(pts)':>13}  {'% Max':>8}")
L("-"*115)

# Aggregate across trading days
for h_name in horizon_names:
    tot_gross      = sum(r['horizons'][h_name]['gross']               for r in trading_days)
    tot_net        = sum(r['horizons'][h_name]['net']                 for r in trading_days)
    tot_net_filt   = sum(r['horizons'][h_name]['net_profitable_only'] for r in trading_days)
    tot_trades     = sum(r['horizons'][h_name]['n_trades']            for r in trading_days)
    tot_profitable = sum(r['horizons'][h_name]['n_profitable']        for r in trading_days)
    avg_move       = np.mean([r['horizons'][h_name]['avg_move_pts']   for r in trading_days])
    win_rate       = tot_profitable / tot_trades * 100 if tot_trades > 0 else 0
    pct_of_max     = tot_net_filt / total_gross_100ms * 100 if total_gross_100ms > 0 else 0

    L(f"{h_name:<18}  ${tot_gross:>11,.0f}  ${tot_net:>11,.0f}  ${tot_net_filt:>12,.0f}  "
      f"{tot_trades:>9,}  {tot_profitable:>9,}  {win_rate:>5.1f}%  {avg_move:>12.4f}  {pct_of_max:>7.2f}%")

L("-"*115)
L(f"{'100ms Ceiling':18}  ${total_gross_100ms:>11,.0f}  "
  f"${total_net_100ms:>11,.0f}  {'---':>13}  "
  f"{'---':>9}  {'---':>9}  {'---':>6}  {'---':>13}  {'100.00%':>8}")
L()

# ----------------------------------------------------------
# TABLE 3: Per-horizon daily averages
# ----------------------------------------------------------
L("="*80)
L("TABLE 3: DAILY AVERAGES BY HORIZON")
L("="*80)
L(f"{'Horizon':<18}  {'Avg Gross/day':>14}  {'Avg Net/day':>13}  {'Avg Net(filt)':>14}  {'Avg #Trades':>12}  {'Avg WinRate':>12}")
L("-"*95)
for h_name in horizon_names:
    avg_gross    = np.mean([r['horizons'][h_name]['gross']               for r in trading_days])
    avg_net      = np.mean([r['horizons'][h_name]['net']                 for r in trading_days])
    avg_net_filt = np.mean([r['horizons'][h_name]['net_profitable_only'] for r in trading_days])
    avg_trades   = np.mean([r['horizons'][h_name]['n_trades']            for r in trading_days])
    avg_wr       = np.mean([r['horizons'][h_name]['win_rate']            for r in trading_days])
    L(f"{h_name:<18}  ${avg_gross:>13,.0f}  ${avg_net:>12,.0f}  ${avg_net_filt:>13,.0f}  "
      f"{avg_trades:>12,.0f}  {avg_wr:>11.1f}%")

L("-"*95)
L(f"{'100ms Ceiling':18}  ${total_gross_100ms/n_td:>13,.0f}  "
  f"${total_net_100ms/n_td:>12,.0f}  {'---':>14}  {'---':>12}  {'---':>12}")
L()

# ----------------------------------------------------------
# TABLE 4: Annualized projections
# ----------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252

L("="*80)
L("TABLE 4: ANNUALIZED PROJECTIONS (252 trading days, 1 contract)")
L("  Note: Assumes same avg daily profit holds throughout the year.")
L("="*80)
L(f"{'Horizon':<18}  {'Annual Gross':>13}  {'Annual Net':>12}  {'Annual Net(filt)':>17}")
L("-"*70)
for h_name in horizon_names:
    avg_gross    = np.mean([r['horizons'][h_name]['gross']               for r in trading_days])
    avg_net      = np.mean([r['horizons'][h_name]['net']                 for r in trading_days])
    avg_net_filt = np.mean([r['horizons'][h_name]['net_profitable_only'] for r in trading_days])
    ann_gross = avg_gross * TRADING_DAYS_PER_YEAR
    ann_net   = avg_net   * TRADING_DAYS_PER_YEAR
    ann_filt  = avg_net_filt * TRADING_DAYS_PER_YEAR
    L(f"{h_name:<18}  ${ann_gross:>12,.0f}  ${ann_net:>11,.0f}  ${ann_filt:>16,.0f}")

L("-"*70)
ann_100ms_gross = (total_gross_100ms / n_td) * TRADING_DAYS_PER_YEAR
ann_100ms_net   = (total_net_100ms   / n_td) * TRADING_DAYS_PER_YEAR
L(f"{'100ms Ceiling':18}  ${ann_100ms_gross:>12,.0f}  ${ann_100ms_net:>11,.0f}  {'---':>17}")
L()

# ----------------------------------------------------------
# TABLE 5: % of Max Theoretical captured by horizon
# ----------------------------------------------------------
L("="*80)
L("TABLE 5: EFFICIENCY vs. THEORETICAL CEILING")
L("  How much of the omniscient 100ms gross profit does each horizon capture?")
L("="*80)
L(f"{'Horizon':<18}  {'Gross Capture%':>15}  {'Net Filt / Gross Max':>21}  {'BreakEven Move(pts)':>20}")
L("-"*80)
for h_name, h_bars in HORIZONS.items():
    tot_gross    = sum(r['horizons'][h_name]['gross']               for r in trading_days)
    tot_net_filt = sum(r['horizons'][h_name]['net_profitable_only'] for r in trading_days)
    gross_cap    = tot_gross    / total_gross_100ms * 100 if total_gross_100ms > 0 else 0
    net_cap      = tot_net_filt / total_gross_100ms * 100 if total_gross_100ms > 0 else 0
    # Break-even move: min move to cover costs
    breakeven_pts = TOTAL_COST_RT / POINT_VALUE
    L(f"{h_name:<18}  {gross_cap:>14.2f}%  {net_cap:>20.2f}%  {breakeven_pts:>20.4f}")
L()

# ----------------------------------------------------------
# TABLE 6: Per-day details for 5s horizon (most relevant signal)
# ----------------------------------------------------------
L("="*80)
L("TABLE 6: PER-DAY DETAIL — 5s Horizon (most relevant for IC=0.135 signal)")
L("="*80)
h_key = '5s  (50bar)'
L(f"{'Day':>4}  {'Range(pts)':>10}  {'Gross($)':>10}  {'Net($)All':>11}  {'Net($)Filt':>12}  {'#Trades':>8}  {'Win%':>6}")
L("-"*75)
for r in trading_days:
    hr = r['horizons'][h_key]
    L(f"{r['day']:4d}  {r['range_pts']:10.2f}  ${hr['gross']:>9,.0f}  "
      f"${hr['net']:>10,.0f}  ${hr['net_profitable_only']:>11,.0f}  "
      f"{hr['n_trades']:>8,}  {hr['win_rate']:>5.1f}%")

tot_g5  = sum(r['horizons'][h_key]['gross']               for r in trading_days)
tot_n5  = sum(r['horizons'][h_key]['net']                 for r in trading_days)
tot_nf5 = sum(r['horizons'][h_key]['net_profitable_only'] for r in trading_days)
tot_t5  = sum(r['horizons'][h_key]['n_trades']            for r in trading_days)
avg_w5  = np.mean([r['horizons'][h_key]['win_rate'] for r in trading_days])
L("-"*75)
L(f"{'TOTAL':>4}  {'':>10}  ${tot_g5:>9,.0f}  ${tot_n5:>10,.0f}  ${tot_nf5:>11,.0f}  {tot_t5:>8,}  {avg_w5:>5.1f}%")
L(f"{'AVG':>4}  {'':>10}  ${tot_g5/n_td:>9,.0f}  ${tot_n5/n_td:>10,.0f}  ${tot_nf5/n_td:>11,.0f}  "
  f"{tot_t5/n_td:>8,.0f}  {avg_w5:>5.1f}%")
L()

# ----------------------------------------------------------
# SUMMARY SECTION
# ----------------------------------------------------------
L("="*80)
L("EXECUTIVE SUMMARY")
L("="*80)
L()
L(f"  Data: {n_days} cache days, {len(trading_days)} active trading days, {len(flat_days)} flat/weekend days")
L(f"  Period: 49 days of ES MBO data (July-Aug 2025), 100ms bars")
L()

# Key numbers at 5s horizon
h5_key = '5s  (50bar)'
tot_g5  = sum(r['horizons'][h5_key]['gross']               for r in trading_days)
tot_n5  = sum(r['horizons'][h5_key]['net']                 for r in trading_days)
tot_nf5 = sum(r['horizons'][h5_key]['net_profitable_only'] for r in trading_days)
avg_n5d = tot_nf5 / n_td

L("  THEORETICAL MAXIMUM (omniscient 100ms oracle, zero cost):")
L(f"    Total across all days:  ${total_gross_100ms:>12,.0f}")
L(f"    Average per day:        ${total_gross_100ms/n_td:>12,.0f}")
L(f"    Annualized (252d):      ${ann_100ms_gross:>12,.0f}")
L()
L("  OMNISCIENT WITH COSTS (100ms, pay spread+commission each reversal):")
L(f"    Total across all days:  ${total_net_100ms:>12,.0f}")
L(f"    Average per day:        ${total_net_100ms/n_td:>12,.0f}")
L(f"    Annualized (252d):      ${ann_100ms_net:>12,.0f}")
L(f"    Total round-trips:      {total_trades_100ms:>12,}")
L()
L("  PERFECT ORACLE AT 5s HORIZON (most relevant for IC=0.135 signal):")
L(f"    Gross (all trades):     ${tot_g5:>12,.0f}")
L(f"    Net all trades:         ${tot_n5:>12,.0f}")
L(f"    Net filtered (> cost):  ${tot_nf5:>12,.0f}")
L(f"    Average per day (filt): ${avg_n5d:>12,.0f}")
L(f"    Annualized (252d filt): ${avg_n5d*252:>12,.0f}")
L(f"    Win rate (profitable):  {np.mean([r['horizons'][h5_key]['win_rate'] for r in trading_days]):.1f}%")
L()

# Key insight: what fraction of max does actual IC capture?
# IC = 0.135 → roughly captures IC^2 of variance explained
# Expected gross per trade = IC * vol * POINT_VALUE
avg_move_5s = np.mean([r['horizons'][h5_key]['avg_move_pts'] for r in trading_days])
n_trades_5s_total = sum(r['horizons'][h5_key]['n_trades'] for r in trading_days)
L("  INTERPRETATION vs. ACTUAL SIGNAL (IC=0.135):")
L(f"    Break-even move needed: {TOTAL_COST_RT/POINT_VALUE:.4f} pts = {TOTAL_COST_RT/TICK_VALUE:.2f} ticks")
L(f"    Avg actual move at 5s:  {avg_move_5s:.4f} pts per window")
L(f"    With IC=0.135, predicted move fraction: ~{0.135:.1%} (rank correlation)")
L(f"    Practical implication: IC=0.135 signal captures only a fraction of this ceiling.")
L(f"    The theoretical 5s max assumes PERFECT knowledge of direction every 5 seconds.")
L()
L(f"  BOTTOM LINE:")
L(f"    5s horizon max (perfect, filtered):  ${tot_nf5:>12,.0f} total / ${avg_n5d:>8,.0f} per day")
L(f"    5s horizon annualized:               ${avg_n5d*252:>12,.0f} per year per contract")
L(f"    Break-even % needed from IC signal:  {TOTAL_COST_RT/(avg_move_5s*POINT_VALUE)*100:.1f}% of moves must exceed cost")
L()
L("="*80)
L(f"Analysis complete. Output saved to: alpha_discovery/results/max_profit_analysis.txt")
L("="*80)

# ============================================================
#  SAVE OUTPUT
# ============================================================
os.makedirs('alpha_discovery/results', exist_ok=True)
out_path = 'alpha_discovery/results/max_profit_analysis.txt'
with open(out_path, 'w') as f:
    f.write('\n'.join(lines))

print(f"\nSaved to: {out_path}")
