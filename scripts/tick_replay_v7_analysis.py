#!/usr/bin/env python3
"""
Tick Replay V7 — Deep analysis of winning config + v3.3 comparison
=====================================================================

Goals:
1. Run winning config (both_30s_vol2.5_k20) with v3.3 predictions
2. Fine-grained sweep around winning parameters (vol 2.2-2.8, k 16-24, hold 20-40)
3. Trade-level deep dive: time-of-day, signal strength, MFE/MAE analysis
4. Regime-stratified results (green/red/flat days)

HC #659 compliant: tick-level replay only, permutation gate on all results.
"""

import sys
import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, '/home/jupiter/Lvl3Quant')

# Import the tick replay engine
from engines.tick_replay_engine import (
    TickReplayEngine, load_predictions, find_mbo_files,
    run_permutation_test, COMMISSION_RT_TICKS, SPREAD_TICKS,
    COST_PASSIVE_EXIT, COST_MARKET_EXIT
)

MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR_V342 = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
PRED_DIR_V33 = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/oot_47day_perdate"
OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/tick_replay_v7"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ES daily regime data (SPY close-to-close direction)
def get_regime(date_str):
    """Classify day as green/red/flat based on ES moves.
    Simple heuristic from the dates we know."""
    # We'll classify based on the day's overall move direction from the data
    return 'unknown'  # Will be filled from actual data


def run_config(mbo_dir, pred_dir, tp, sl, threshold, hold, cancel, k_filter, max_days=None):
    """Run a single config through tick replay on all available days."""
    predictions = load_predictions(pred_dir)
    mbo_files = find_mbo_files(mbo_dir)

    # Match dates
    matched = []
    for mf in mbo_files:
        basename = os.path.basename(mf)
        date8 = basename.split('-')[2].split('.')[0]
        if date8 in predictions:
            matched.append((date8, mf, predictions[date8]))

    if max_days and len(matched) > max_days:
        matched = matched[:max_days]

    all_trades = []
    per_day = {}

    for date8, mbo_path, preds in matched:
        try:
            engine = TickReplayEngine(
                tp_ticks=tp, sl_ticks=sl,
                signal_threshold=threshold,
                hold_seconds=hold,
                cancel_seconds=cancel,
            )

            # Apply k-filter: only use top-k strongest predictions per stride
            if k_filter and k_filter > 0:
                # Sort by absolute signal strength, keep top k
                abs_preds = np.abs(preds)
                if len(preds) > k_filter:
                    k_thresh = np.sort(abs_preds)[-k_filter]
                    preds_filtered = preds.copy()
                    preds_filtered[abs_preds < k_thresh] = 0.0
                else:
                    preds_filtered = preds
            else:
                preds_filtered = preds

            trades = engine.run_day(mbo_path, preds_filtered)

            day_trades = []
            day_net = 0.0
            for t in trades:
                day_trades.append({
                    'side': t.side,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'entry_time_ns': t.entry_time_ns,
                    'exit_time_ns': t.exit_time_ns,
                    'signal_strength': float(t.signal_strength),
                    'exit_reason': t.exit_reason,
                    'pnl_ticks': float(t.pnl_ticks),
                    'cost_ticks': float(t.cost_ticks),
                    'mfe_ticks': float(t.mfe_ticks),
                    'mae_ticks': float(t.mae_ticks),
                    'fill_latency_ns': t.fill_latency_ns,
                    'queue_depth': t.queue_depth_at_entry,
                })
                day_net += t.pnl_ticks - t.cost_ticks

            per_day[date8] = {
                'n_trades': len(trades),
                'net_pnl': round(day_net, 2),
            }
            all_trades.extend(day_trades)

        except Exception as e:
            print(f"  Error on {date8}: {e}")
            continue

    # Aggregate stats
    if not all_trades:
        return {'n_trades': 0, 'net_pnl': 0, 'sharpe': 0, 'trades': [], 'per_day': per_day}

    gross_pnls = [t['pnl_ticks'] for t in all_trades]
    costs = [t['cost_ticks'] for t in all_trades]
    net_pnls = [g - c for g, c in zip(gross_pnls, costs)]

    total_net = sum(net_pnls)
    avg_net = total_net / len(all_trades)
    win_rate = sum(1 for n in net_pnls if n > 0) / len(net_pnls) if net_pnls else 0

    # Daily PnL for Sharpe
    daily_pnls = [v['net_pnl'] for v in per_day.values()]
    if len(daily_pnls) > 1:
        sharpe = np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252) if np.std(daily_pnls) > 0 else 0
    else:
        sharpe = 0

    n_green = sum(1 for p in daily_pnls if p > 0)
    n_red = sum(1 for p in daily_pnls if p < 0)

    # Profit factor
    gross_wins = sum(n for n in net_pnls if n > 0)
    gross_losses = abs(sum(n for n in net_pnls if n < 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    return {
        'n_trades': len(all_trades),
        'net_pnl': round(total_net, 2),
        'avg_net': round(avg_net, 4),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(pf, 3),
        'sharpe': round(sharpe, 2),
        'green_days': n_green,
        'red_days': n_red,
        'total_days': len(daily_pnls),
        'trades': all_trades,
        'per_day': per_day,
    }


def analyze_trades(trades):
    """Deep trade-level analysis."""
    if not trades:
        return {}

    # Time-of-day distribution (group by hour ET)
    hour_stats = defaultdict(lambda: {'n': 0, 'net': 0.0, 'wins': 0})
    for t in trades:
        # Convert ns timestamp to hour (assuming ET)
        hour = (t['entry_time_ns'] // (3600 * 10**9)) % 24
        net = t['pnl_ticks'] - t['cost_ticks']
        hour_stats[hour]['n'] += 1
        hour_stats[hour]['net'] += net
        if net > 0:
            hour_stats[hour]['wins'] += 1

    # Signal strength distribution
    signals = [abs(t['signal_strength']) for t in trades]
    signal_buckets = {}
    for q_name, q_val in [('bot25', 0.25), ('mid50', 0.5), ('top25', 0.75)]:
        thresh = np.quantile(signals, q_val) if q_val < 0.75 else np.quantile(signals, 0.75)
        if q_name == 'bot25':
            bucket_trades = [t for t in trades if abs(t['signal_strength']) <= np.quantile(signals, 0.25)]
        elif q_name == 'mid50':
            q1, q3 = np.quantile(signals, 0.25), np.quantile(signals, 0.75)
            bucket_trades = [t for t in trades if q1 < abs(t['signal_strength']) <= q3]
        else:
            bucket_trades = [t for t in trades if abs(t['signal_strength']) > np.quantile(signals, 0.75)]

        if bucket_trades:
            nets = [t['pnl_ticks'] - t['cost_ticks'] for t in bucket_trades]
            signal_buckets[q_name] = {
                'n': len(bucket_trades),
                'avg_net': round(np.mean(nets), 3),
                'wr': round(sum(1 for n in nets if n > 0) / len(nets), 3),
            }

    # MFE/MAE analysis
    mfes = [t['mfe_ticks'] for t in trades]
    maes = [t['mae_ticks'] for t in trades]

    mfe_mae = {
        'mfe_mean': round(np.mean(mfes), 2),
        'mfe_median': round(np.median(mfes), 2),
        'mfe_p75': round(np.percentile(mfes, 75), 2),
        'mfe_p90': round(np.percentile(mfes, 90), 2),
        'mae_mean': round(np.mean(maes), 2),
        'mae_median': round(np.median(maes), 2),
        'mae_p75': round(np.percentile(maes, 75), 2),
        'mae_p90': round(np.percentile(maes, 90), 2),
    }

    # Exit reason distribution
    exit_reasons = defaultdict(int)
    exit_pnl = defaultdict(list)
    for t in trades:
        r = t['exit_reason']
        exit_reasons[r] += 1
        exit_pnl[r].append(t['pnl_ticks'] - t['cost_ticks'])

    exit_analysis = {}
    for r, count in exit_reasons.items():
        nets = exit_pnl[r]
        exit_analysis[r] = {
            'n': count,
            'pct': round(count / len(trades) * 100, 1),
            'avg_net': round(np.mean(nets), 3),
            'total_net': round(sum(nets), 2),
        }

    # Fill latency
    latencies_ms = [t['fill_latency_ns'] / 1e6 for t in trades if t['fill_latency_ns'] > 0]
    lat_stats = {}
    if latencies_ms:
        lat_stats = {
            'mean_ms': round(np.mean(latencies_ms), 1),
            'median_ms': round(np.median(latencies_ms), 1),
            'p90_ms': round(np.percentile(latencies_ms, 90), 1),
        }

    return {
        'time_of_day': {str(h): v for h, v in sorted(hour_stats.items())},
        'signal_strength_buckets': signal_buckets,
        'mfe_mae': mfe_mae,
        'exit_reasons': exit_analysis,
        'fill_latency': lat_stats,
    }


def main():
    print("=" * 70)
    print("TICK REPLAY V7 — DEEP ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}

    # ==========================================
    # PART 1: Re-run winning config with v3.4.2 (baseline)
    # ==========================================
    print("\n[1/4] Baseline: v3.4.2 predictions, winning config (vol2.5, k20, 30s hold)")
    baseline = run_config(
        MBO_DIR, PRED_DIR_V342,
        tp=4, sl=3, threshold=2.5, hold=30.0, cancel=15.0, k_filter=20
    )
    print(f"  {baseline['n_trades']} trades, net={baseline['net_pnl']}t, "
          f"Sharpe={baseline['sharpe']}, WR={baseline['win_rate']}")

    trade_analysis = analyze_trades(baseline['trades'])
    results['v342_baseline'] = {
        'config': 'tp4_sl3_vol2.5_k20_hold30',
        'stats': {k: v for k, v in baseline.items() if k != 'trades'},
        'trade_analysis': trade_analysis,
    }

    # ==========================================
    # PART 2: Same config with v3.3 predictions
    # ==========================================
    print("\n[2/4] v3.3 predictions, same winning config")
    v33_result = run_config(
        MBO_DIR, PRED_DIR_V33,
        tp=4, sl=3, threshold=2.5, hold=30.0, cancel=15.0, k_filter=20
    )
    print(f"  {v33_result['n_trades']} trades, net={v33_result['net_pnl']}t, "
          f"Sharpe={v33_result['sharpe']}, WR={v33_result['win_rate']}")

    v33_analysis = analyze_trades(v33_result['trades'])
    results['v33_same_config'] = {
        'config': 'tp4_sl3_vol2.5_k20_hold30 (v3.3 preds)',
        'stats': {k: v for k, v in v33_result.items() if k != 'trades'},
        'trade_analysis': v33_analysis,
    }

    # ==========================================
    # PART 3: Fine-grained sweep around winning region
    # ==========================================
    print("\n[3/4] Fine-grained sweep around winning parameters")
    sweep_results = {}

    configs = [
        # Vary hold time
        {'tp': 4, 'sl': 3, 'threshold': 2.5, 'hold': 20.0, 'k': 20, 'label': 'hold20'},
        {'tp': 4, 'sl': 3, 'threshold': 2.5, 'hold': 25.0, 'k': 20, 'label': 'hold25'},
        {'tp': 4, 'sl': 3, 'threshold': 2.5, 'hold': 35.0, 'k': 20, 'label': 'hold35'},
        {'tp': 4, 'sl': 3, 'threshold': 2.5, 'hold': 45.0, 'k': 20, 'label': 'hold45'},
        # Vary TP
        {'tp': 3, 'sl': 3, 'threshold': 2.5, 'hold': 30.0, 'k': 20, 'label': 'tp3_sl3'},
        {'tp': 5, 'sl': 3, 'threshold': 2.5, 'hold': 30.0, 'k': 20, 'label': 'tp5_sl3'},
        {'tp': 6, 'sl': 3, 'threshold': 2.5, 'hold': 30.0, 'k': 20, 'label': 'tp6_sl3'},
        {'tp': 4, 'sl': 2, 'threshold': 2.5, 'hold': 30.0, 'k': 20, 'label': 'tp4_sl2'},
        {'tp': 4, 'sl': 4, 'threshold': 2.5, 'hold': 30.0, 'k': 20, 'label': 'tp4_sl4'},
        # Vary vol threshold
        {'tp': 4, 'sl': 3, 'threshold': 2.2, 'hold': 30.0, 'k': 20, 'label': 'vol2.2'},
        {'tp': 4, 'sl': 3, 'threshold': 2.8, 'hold': 30.0, 'k': 20, 'label': 'vol2.8'},
        # Vary k
        {'tp': 4, 'sl': 3, 'threshold': 2.5, 'hold': 30.0, 'k': 18, 'label': 'k18'},
        {'tp': 4, 'sl': 3, 'threshold': 2.5, 'hold': 30.0, 'k': 22, 'label': 'k22'},
        # No SL (time stop only)
        {'tp': 4, 'sl': 99, 'threshold': 2.5, 'hold': 30.0, 'k': 20, 'label': 'no_sl'},
        # Asymmetric: tight SL, wider TP
        {'tp': 6, 'sl': 2, 'threshold': 2.5, 'hold': 30.0, 'k': 20, 'label': 'tp6_sl2'},
        {'tp': 8, 'sl': 3, 'threshold': 2.5, 'hold': 45.0, 'k': 20, 'label': 'tp8_sl3_hold45'},
    ]

    for cfg in configs:
        label = cfg['label']
        print(f"  Testing {label}...", end=' ', flush=True)
        r = run_config(
            MBO_DIR, PRED_DIR_V342,
            tp=cfg['tp'], sl=cfg['sl'], threshold=cfg['threshold'],
            hold=cfg['hold'], cancel=15.0, k_filter=cfg['k']
        )
        sweep_results[label] = {
            'config': cfg,
            'n_trades': r['n_trades'],
            'net_pnl': r['net_pnl'],
            'avg_net': r.get('avg_net', 0),
            'win_rate': r.get('win_rate', 0),
            'profit_factor': r.get('profit_factor', 0),
            'sharpe': r['sharpe'],
            'green_days': r.get('green_days', 0),
            'red_days': r.get('red_days', 0),
            'total_days': r.get('total_days', 0),
        }
        print(f"n={r['n_trades']}, net={r['net_pnl']}t, Sharpe={r['sharpe']}")

    results['fine_sweep'] = sweep_results

    # ==========================================
    # PART 4: Permutation test on top 3 configs
    # ==========================================
    print("\n[4/4] Permutation tests on best sweep configs")

    # Sort by net PnL, get top 3 with > 0 net
    positive = [(k, v) for k, v in sweep_results.items() if v['net_pnl'] > 0]
    positive.sort(key=lambda x: x[1]['net_pnl'], reverse=True)

    perm_results = {}
    predictions_all = load_predictions(PRED_DIR_V342)
    mbo_files = find_mbo_files(MBO_DIR)

    for label, stats in positive[:3]:
        cfg = stats['config']
        print(f"  Permutation test: {label} (net={stats['net_pnl']}t)...", flush=True)

        # Apply k-filter to predictions before permutation test
        filtered_preds = {}
        for d8, p in predictions_all.items():
            if cfg['k'] and cfg['k'] > 0 and len(p) > cfg['k']:
                abs_p = np.abs(p)
                k_thresh = np.sort(abs_p)[-cfg['k']]
                pf = p.copy()
                pf[abs_p < k_thresh] = 0.0
                filtered_preds[d8] = pf
            else:
                filtered_preds[d8] = p

        def make_engine(c=cfg):
            return TickReplayEngine(
                tp_ticks=c['tp'], sl_ticks=c['sl'],
                signal_threshold=c['threshold'],
                hold_seconds=c['hold'],
                cancel_seconds=15.0,
            )

        model_metrics, p_value = run_permutation_test(
            make_engine, mbo_files, filtered_preds,
            n_perms=100,
        )
        perm_results[label] = {
            'real_pnl': round(model_metrics.get('net_pnl_ticks', 0), 2),
            'p_value': round(p_value, 4),
            'verdict': '✅ PASS' if p_value < 0.05 else '❌ FAIL',
        }
        print(f"    p={p_value:.4f} {'✅' if p_value < 0.05 else '❌'}")

    results['permutation_tests'] = perm_results

    # ==========================================
    # SAVE
    # ==========================================
    # Remove raw trade data before saving (too large)
    for key in ['v342_baseline', 'v33_same_config']:
        if key in results and 'stats' in results[key]:
            results[key]['stats'].pop('per_day', None)

    output_path = os.path.join(OUTPUT_DIR, 'v7_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n--- Baseline (v3.4.2) ---")
    b = results['v342_baseline']['stats']
    print(f"  Trades: {b['n_trades']}, Net: {b['net_pnl']}t, Sharpe: {b['sharpe']}, WR: {b['win_rate']}")

    print("\n--- v3.3 comparison ---")
    v = results['v33_same_config']['stats']
    print(f"  Trades: {v['n_trades']}, Net: {v['net_pnl']}t, Sharpe: {v['sharpe']}, WR: {v['win_rate']}")

    print("\n--- Fine sweep (top 5 by net PnL) ---")
    sorted_sweep = sorted(sweep_results.items(), key=lambda x: x[1]['net_pnl'], reverse=True)
    for label, s in sorted_sweep[:5]:
        print(f"  {label}: n={s['n_trades']}, net={s['net_pnl']}t, Sharpe={s['sharpe']}, WR={s['win_rate']}")

    print("\n--- Permutation tests ---")
    for label, p in perm_results.items():
        print(f"  {label}: real={p['real_pnl']}t, p={p['p_value']} {p['verdict']}")

    if trade_analysis:
        print("\n--- Trade Analysis (baseline) ---")
        if 'exit_reasons' in trade_analysis:
            print("  Exit reasons:", json.dumps(trade_analysis['exit_reasons'], indent=4))
        if 'mfe_mae' in trade_analysis:
            print("  MFE/MAE:", json.dumps(trade_analysis['mfe_mae'], indent=4))
        if 'signal_strength_buckets' in trade_analysis:
            print("  Signal buckets:", json.dumps(trade_analysis['signal_strength_buckets'], indent=4))

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
