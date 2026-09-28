#!/usr/bin/env python3
"""
Jupiter Execution Research Queue v1 — 2026-03-28
Execution Research Lead: Systematic campaign to push MC p5 Sortino > 1.0

CURRENT STATE:
  Best config: A_tp13_sl40_t050 -> OOT Sortino 4.197, MC median 5.06, MC p5=-2.67
  Goal: Reduce variance, push MC p5 Sortino > 1.0 for deployable confidence

CAMPAIGN DESIGN:
  Experiment 1: Chase entry sweep (proven Sortino 4.22 in prior OOT research)
  Experiment 2: High-conviction threshold z>2.0 (prior result: Sortino 2.046)
  Experiment 3: Dynamic TP/SL by signal strength bins
  Experiment 4: Time-of-day tighter gates (10:30-2:30 is prime, test 11:00-2:00)
  Experiment 5: Conviction exit (signal flip exit with delay)
  Experiment 6: MAE exit (cut losers faster)
  Experiment 7: Chase + vol regime combo
  Experiment 8: Ratchet stop (lock in profits on runners)
  Experiment 9: Latency-aware full sim (20ms realistic)
  Experiment 10: Combined best: chase + high thresh + conviction exit

All results aggregated with Monte Carlo (1000 bootstrap sims).
"""

import numpy as np
import subprocess
import json
import os
import glob
import sys
from datetime import datetime

# Paths
FILL_SIM = '/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli'
MBO_OOT_DIR = '/home/jupiter/Lvl3Quant/fill_sim_test/results_oot_best/A_tp13_sl40_t050'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
SINGLE_PREDS = '/home/jupiter/Lvl3Quant/fill_sim_test/single_preds'
BASE_OUT = '/home/jupiter/Lvl3Quant/results/exec_queue_v1'
os.makedirs(BASE_OUT, exist_ok=True)

# Discover matched dates (single_preds aligned with MBO files)
mbo_files = sorted(glob.glob(os.path.join(MBO_DIR, '*.dbn.zst')))
matched = []
for mbo in mbo_files:
    fname = os.path.basename(mbo)
    date8 = fname.split('-')[2].split('.')[0]
    pred_path = os.path.join(SINGLE_PREDS, '%s.npz' % date8)
    if os.path.exists(pred_path):
        matched.append((mbo, date8, pred_path))
print('Matched %d date/MBO pairs' % len(matched))

# Also find the OOT-only dates (Dec 2025 - Mar 2026) for cleaner OOT analysis
oot_matched = [(mbo, d, p) for mbo, d, p in matched if int(d) >= 20251201]
print('OOT matched dates: %d (Dec 2025+)' % len(oot_matched))


def run_sweep(name, flags, threshold, use_chase=False, use_oot_only=True):
    """Run fill sim across all matched dates, return per-day results."""
    dates_to_run = oot_matched if use_oot_only else matched
    out_dir = os.path.join(BASE_OUT, name)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    chase_flags = '--chase-entry --chase-max-ticks 3 --chase-max-reprices 10' if use_chase else ''

    for mbo, date8, pred_path in dates_to_run:
        out_path = os.path.join(out_dir, '%s_%s.json' % (name, date8))
        if os.path.exists(out_path):
            # Already computed, load cached
            with open(out_path) as f:
                data = json.load(f)
            results.append(data)
            continue

        cmd = ('{fill_sim} --mbo-file {mbo} --predictions {pred} --output {out} '
               '--signal-threshold {thresh:.2f} --prime-hours {chase} {flags} --quiet').format(
            fill_sim=FILL_SIM, mbo=mbo, pred=pred_path, out=out_path,
            thresh=threshold, chase=chase_flags, flags=flags
        )

        try:
            proc = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=180)
            if os.path.exists(out_path):
                with open(out_path) as f:
                    data = json.load(f)
                results.append(data)
            else:
                print('  [WARN] No output for %s / %s: %s' % (name, date8, proc.stderr[:100]))
        except subprocess.TimeoutExpired:
            print('  [TIMEOUT] %s / %s' % (name, date8))
        except Exception as e:
            print('  [ERROR] %s / %s: %s' % (name, date8, e))

    return results


def compute_sortino(daily_pnls):
    """Compute annualized Sortino ratio from daily P&L array."""
    if len(daily_pnls) < 3:
        return float('nan')
    daily_pnls = np.array(daily_pnls)
    mean_daily = np.mean(daily_pnls)
    downside = daily_pnls[daily_pnls < 0]
    if len(downside) == 0:
        return float('inf')
    downside_std = np.std(downside)
    if downside_std == 0:
        return float('inf')
    return (mean_daily / downside_std) * np.sqrt(252)


def monte_carlo(daily_pnls, n_sims=1000):
    """Bootstrap Monte Carlo — returns p5, median, p95 Sortino and prob_profit."""
    if len(daily_pnls) < 5:
        return {'p5': float('nan'), 'median': float('nan'), 'p95': float('nan'), 'prob_profit': 0.0}

    daily_pnls = np.array(daily_pnls)
    n = len(daily_pnls)
    sortinos = []
    for _ in range(n_sims):
        sample = np.random.choice(daily_pnls, size=n, replace=True)
        sortinos.append(compute_sortino(sample))

    sortinos = [s for s in sortinos if not (np.isnan(s) or np.isinf(s))]
    if not sortinos:
        return {'p5': float('nan'), 'median': float('nan'), 'p95': float('nan'), 'prob_profit': 0.0}

    return {
        'p5': float(np.percentile(sortinos, 5)),
        'median': float(np.percentile(sortinos, 50)),
        'p95': float(np.percentile(sortinos, 95)),
        'prob_profit': float(np.mean(np.array(sortinos) > 0))
    }


def summarize(name, results):
    """Compute summary metrics from fill sim results list."""
    if not results:
        return None

    daily_pnls = [r.get('total_pnl_dollars', 0) for r in results]
    total_pnl = sum(daily_pnls)
    total_trades = sum(r.get('total_trades', 0) for r in results)
    n_days = len(results)

    # Aggregate win rate
    total_wins = sum(r.get('total_wins', 0) for r in results if 'total_wins' in r)
    total_trades_with_wr = sum(r.get('total_trades', 0) for r in results if 'total_wins' in r)
    win_rate = (total_wins / total_trades_with_wr) if total_trades_with_wr > 0 else float('nan')

    sortino = compute_sortino(daily_pnls)
    mc = monte_carlo(daily_pnls, n_sims=1000)

    return {
        'name': name,
        'n_days': n_days,
        'total_pnl': total_pnl,
        'avg_pnl_per_day': total_pnl / n_days if n_days > 0 else 0,
        'total_trades': total_trades,
        'trades_per_day': total_trades / n_days if n_days > 0 else 0,
        'win_rate': win_rate,
        'sortino': sortino,
        'mc_p5_sortino': mc['p5'],
        'mc_median_sortino': mc['median'],
        'mc_p95_sortino': mc['p95'],
        'mc_prob_profit': mc['prob_profit']
    }


def print_summary(s):
    if s is None:
        print('  [NO DATA]')
        return
    print('  Sortino=%.3f  MC_p5=%.3f  MC_med=%.3f  prob_profit=%.1f%%' % (
        s['sortino'], s['mc_p5_sortino'], s['mc_median_sortino'], s['mc_prob_profit']*100))
    print('  PnL=$%.0f  AvgDay=$%.0f  Trades/Day=%.1f  WR=%.3f  Days=%d' % (
        s['total_pnl'], s['avg_pnl_per_day'], s['trades_per_day'],
        s['win_rate'] if not np.isnan(s['win_rate']) else 0, s['n_days']))


# ===== EXPERIMENT QUEUE =====
all_summaries = {}

# ------------------------------------------------------------------
# EXP 1: BASELINE — Current best config (sanity check)
# A_tp13_sl40 at threshold 0.5, prime hours, no chase
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 1: Baseline — A_tp13_sl40 t=0.5 prime_hours')
exp = 'exp1_baseline_tp13_sl40_t050'
r = run_sweep(exp, '--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000', 0.5, use_chase=False)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 2: CHASE ENTRY — The proven edge from prior research
# Chase has shown Sortino 4.22 on 66 OOT days
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 2: Chase entry — A_tp13_sl40 t=0.5 + chase (3-tick max)')
exp = 'exp2_chase_tp13_sl40_t050'
r = run_sweep(exp, '--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000', 0.5, use_chase=True)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 3: HIGH CONVICTION — z>2.0 threshold (prior result: Sortino 2.046)
# Fewer trades but higher quality signals only
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 3: High conviction — A_tp13_sl40 t=2.0')
exp = 'exp3_highconv_tp13_sl40_t200'
r = run_sweep(exp, '--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000', 2.0, use_chase=False)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 4: CHASE + HIGH CONVICTION — best of both worlds
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 4: Chase + high conviction — A_tp13_sl40 t=2.0 + chase')
exp = 'exp4_chase_highconv_tp13_sl40_t200'
r = run_sweep(exp, '--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000', 2.0, use_chase=True)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 5: CONVICTION EXIT — exit when signal flips (reduces losers)
# 300 bars = 30 seconds of opposite signal before exit
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 5: Conviction exit — A_tp13_sl40 t=0.5 + conv_exit 300bar')
exp = 'exp5_convex_tp13_sl40_t050'
r = run_sweep(exp,
    '--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000 '
    '--conviction-exit-bars 300 --conviction-exit-mag 1.0',
    0.5, use_chase=False)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 6: CHASE + CONVICTION EXIT — aggressive chase in, smart exit
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 6: Chase + conviction exit — t=0.5 chase + conv_exit 300bar')
exp = 'exp6_chase_convex_t050'
r = run_sweep(exp,
    '--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000 '
    '--conviction-exit-bars 300 --conviction-exit-mag 1.0',
    0.5, use_chase=True)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 7: MAE EXIT — cut losses faster (exit if down 8 ticks after 300s)
# Hypothesis: many big losers could be cut early
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 7: MAE exit — A_tp13_sl40 t=0.5 + mae_exit 8tick@300s')
exp = 'exp7_mae_exit_tp13_sl40_t050'
r = run_sweep(exp,
    '--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000 '
    '--mae-exit-ticks 8 --mae-exit-hold-sec 300',
    0.5, use_chase=False)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 8: RATCHET STOP — lock in profits progressively
# Protects winners from giving back gains
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 8: Ratchet stop — A_tp13_sl40 t=0.5 + ratchet')
exp = 'exp8_ratchet_tp13_sl40_t050'
r = run_sweep(exp,
    '--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000 --ratchet-stop',
    0.5, use_chase=False)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 9: LATENCY REALISTIC — 20ms submission delay (prod estimate)
# Validates results hold with realistic co-lo latency
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 9: Latency realistic — A_tp13_sl40 t=0.5 + latency 20ms')
exp = 'exp9_lat20_tp13_sl40_t050'
r = run_sweep(exp,
    '--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000 --latency-ms 20',
    0.5, use_chase=False)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 10: TIGHT TP FOR HIGH CONV — at t=2.0 use smaller TP (C9S style)
# High conviction = move quickly = smaller TP fine
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 10: Tight TP high conv — tp6/sl20 t=2.0 chase (C9S pattern)')
exp = 'exp10_c9s_style_tp6_sl20_t200_chase'
r = run_sweep(exp,
    '--take-profit-ticks 6 --stop-loss-ticks 20 --hold-ms 600000',
    2.0, use_chase=True)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 11: WIDE TP + RATCHET (let winners run)
# tp=20 with ratchet stop — ride the big moves
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 11: Wide TP ratchet — tp20/sl40 t=0.5 + ratchet + chase')
exp = 'exp11_wide_tp20_ratchet_chase_t050'
r = run_sweep(exp,
    '--take-profit-ticks 20 --stop-loss-ticks 40 --hold-ms 2400000 --ratchet-stop',
    0.5, use_chase=True)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 12: FULL COMBO — Chase + HighConv + ConvExit + MAE + Ratchet
# The "kitchen sink" config — see if combining all edges works
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 12: FULL COMBO — Chase + t=2.0 + conv_exit + mae_exit + ratchet')
exp = 'exp12_full_combo_t200'
r = run_sweep(exp,
    '--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000 '
    '--conviction-exit-bars 300 --conviction-exit-mag 1.0 '
    '--mae-exit-ticks 8 --mae-exit-hold-sec 300 --ratchet-stop',
    2.0, use_chase=True)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 13: C1 OPTIMAL — tp10/sl40 (per pre-crash analysis: Sortino 0.455)
# Validate with full Monte Carlo
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 13: C1 optimal — tp10/sl40 t=0.5 (pre-crash result validation)')
exp = 'exp13_c1_optimal_tp10_sl40_t050'
r = run_sweep(exp,
    '--take-profit-ticks 10 --stop-loss-ticks 40 --hold-ms 1800000',
    0.5, use_chase=False)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ------------------------------------------------------------------
# EXP 14: SIGNAL FLIP EXIT — instant exit on sign reversal
# Zero tolerance for holding a decaying position
# ------------------------------------------------------------------
print('\n' + '='*60)
print('EXP 14: Signal flip exit — instant exit on sign reversal')
exp = 'exp14_sigflip_tp13_sl40_t050'
r = run_sweep(exp,
    '--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000 --signal-flip-exit',
    0.5, use_chase=False)
s = summarize(exp, r)
all_summaries[exp] = s
print_summary(s)

# ===== FINAL REPORT =====
print('\n' + '='*70)
print('EXECUTION RESEARCH — FINAL COMPARISON (sorted by MC p5 Sortino)')
print('='*70)

valid = {k: v for k, v in all_summaries.items() if v is not None}
sorted_by_mc_p5 = sorted(valid.items(), key=lambda x: x[1].get('mc_p5_sortino', -999), reverse=True)

print('{:<45} {:>8} {:>8} {:>8} {:>7} {:>7}'.format(
    'Config', 'Sortino', 'MC_p5', 'MC_med', 'WR', 'Trd/D'))
print('-' * 90)
for name, s in sorted_by_mc_p5:
    print('{:<45} {:>8.3f} {:>8.3f} {:>8.3f} {:>7.3f} {:>7.1f}'.format(
        name[:45],
        s.get('sortino', 0),
        s.get('mc_p5_sortino', 0),
        s.get('mc_median_sortino', 0),
        s.get('win_rate', 0) if not np.isnan(s.get('win_rate', 0)) else 0,
        s.get('trades_per_day', 0)
    ))

# Save full results
results_path = os.path.join(BASE_OUT, 'exec_queue_v1_results.json')
with open(results_path, 'w') as f:
    json.dump({k: v for k, v in all_summaries.items() if v is not None}, f, indent=2)
print('\nFull results saved to: %s' % results_path)

# Identify winner
if sorted_by_mc_p5:
    winner = sorted_by_mc_p5[0]
    print('\n*** WINNER: %s (MC p5 Sortino = %.3f) ***' % (winner[0], winner[1]['mc_p5_sortino']))
    if winner[1]['mc_p5_sortino'] > 1.0:
        print('*** DEPLOYMENT CANDIDATE: MC p5 > 1.0 ACHIEVED ***')
    else:
        print('*** MC p5 < 1.0 — still below deployment threshold ***')

print('\nDone at %s' % datetime.now().isoformat())
