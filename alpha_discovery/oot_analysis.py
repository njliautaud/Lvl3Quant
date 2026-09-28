#!/usr/bin/env python3
"""
OOT Analysis — Aggregate analysis of Out-of-Time sweep results
================================================================
Reads all OOT sweep results and generates comprehensive comparison reports.
Includes CNN vs GNN vs Ensemble comparison tables.

Usage:
    python alpha_discovery/oot_analysis.py --results-dir ~/Lvl3Quant/alpha_discovery/results/oot
    python alpha_discovery/oot_analysis.py --results-dir ~/Lvl3Quant/alpha_discovery/results/oot --is-results ~/Lvl3Quant/alpha_discovery/results/cnn_chase_sweep_20260310_171243.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

RESULTS_DIR = Path(__file__).resolve().parent / 'results'

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')


def load_oot_results(results_dir):
    """Load all OOT sweep result files."""
    results_dir = Path(results_dir)
    all_stats = []

    # Look for full results files
    for f in sorted(results_dir.glob('oot_mega_sweep_full_*.json')):
        with open(f) as fh:
            data = json.load(fh)
            all_stats.extend(data)
            print(f"  Loaded {len(data)} configs from {f.name}")

    # Deduplicate by config label (keep latest)
    seen = {}
    for s in all_stats:
        seen[s['config']] = s
    all_stats = list(seen.values())

    print(f"Total unique configs: {len(all_stats)}")
    return all_stats


def load_is_results(is_file):
    """Load IS (in-sample) results for comparison."""
    is_file = Path(is_file)
    if not is_file.exists():
        print(f"  IS results file not found: {is_file}")
        return {}

    with open(is_file) as f:
        data = json.load(f)

    # Index by config name
    results = {}
    for entry in data:
        config = entry.get('config', '')
        results[config] = entry

    print(f"  Loaded {len(results)} IS configs from {is_file.name}")
    return results


def print_sweep_summary(all_stats, sweep_num):
    """Print summary for a specific sweep."""
    sweep_stats = [s for s in all_stats if s.get('sweep') == sweep_num]
    if not sweep_stats:
        return

    sweep_names = {
        1: "CNN CHASE MODE",
        2: "CNN PASSIVE LIMIT",
        3: "CNN ENTRY TIMING",
        4: "CNN EXIT STRATEGIES",
        5: "CNN SIGNAL SENSITIVITY",
        6: "CNN LONG/SHORT ASYMMETRY",
        7: "GNN CHASE MODE",
        8: "GNN PASSIVE LIMIT",
        9: "CNN+GNN ENSEMBLE",
    }

    print(f"\n{'='*130}")
    print(f"SWEEP {sweep_num}: {sweep_names.get(sweep_num, 'UNKNOWN')}")
    print(f"{'='*130}")

    sweep_stats.sort(key=lambda x: x.get('sharpe', 0), reverse=True)

    print(
        f"{'#':>3} {'Config':<55} {'Model':>8} {'P&L':>10} {'Sharpe':>7} {'Sortino':>8} "
        f"{'Trades':>7} {'WR':>6} {'FillR':>6} {'MaxDD':>8} {'PF':>6}"
    )
    print("-" * 130)

    for i, s in enumerate(sweep_stats[:20]):
        pnl = s.get('total_pnl', 0)
        print(
            f"{i+1:>3} {s['config']:<55} "
            f"{s.get('model_type', 'cnn'):>8} "
            f"${pnl:>9,.0f} "
            f"{s.get('sharpe', 0):>7.2f} "
            f"{s.get('sortino', 0):>8.2f} "
            f"{s.get('n_trades', 0):>7} "
            f"{s.get('win_rate', 0):>5.1%} "
            f"{s.get('fill_rate', 0):>5.1%} "
            f"${s.get('max_dd', 0):>7,.0f} "
            f"{s.get('profit_factor', 0):>6.2f}"
        )

    # Stats summary
    sharpes = [s.get('sharpe', 0) for s in sweep_stats]
    pnls = [s.get('total_pnl', 0) for s in sweep_stats]
    profitable = sum(1 for p in pnls if p > 0)

    print(f"\n  Summary: {len(sweep_stats)} configs, {profitable} profitable ({profitable/len(sweep_stats)*100:.0f}%)")
    print(f"  Sharpe: mean={np.mean(sharpes):.2f}, median={np.median(sharpes):.2f}, max={max(sharpes):.2f}")
    print(f"  P&L: mean=${np.mean(pnls):,.0f}, median=${np.median(pnls):,.0f}, max=${max(pnls):,.0f}")


def is_vs_oot_comparison(oot_stats, is_results):
    """Compare IS and OOT performance for matching configs."""
    print(f"\n{'='*130}")
    print("IN-SAMPLE vs OUT-OF-TIME COMPARISON")
    print(f"{'='*130}")

    matched = []

    for oot in oot_stats:
        oot_label = oot['config']
        if oot_label in is_results:
            matched.append((oot, is_results[oot_label]))
            continue

        params = oot.get('config_params', {})
        vol = params.get('vol_gate', 0)
        conv = params.get('signal_threshold', 0)
        hold_ms = params.get('hold_ms', 0)
        chase_t = params.get('chase_max_ticks')
        chase_r = params.get('chase_max_reprices')

        for is_label, is_data in is_results.items():
            is_vol = is_data.get('vol_gate', 0)
            if is_vol == 0:
                if f'vol{vol}' in is_label and f'conv{conv}' in is_label.replace('.', ''):
                    matched.append((oot, is_data))
                    break

    if not matched:
        print("  No matching configs found between IS and OOT results.")
        return

    print(
        f"{'Config':<55} {'IS P&L':>10} {'OOT P&L':>10} {'IS Sharpe':>9} {'OOT Sharpe':>10} "
        f"{'Decay':>7} {'IS WR':>6} {'OOT WR':>7}"
    )
    print("-" * 130)

    decays = []
    for oot, is_data in matched:
        is_sharpe = is_data.get('sharpe_daily', is_data.get('sharpe', 0))
        oot_sharpe = oot.get('sharpe', 0)
        decay = ((oot_sharpe - is_sharpe) / abs(is_sharpe) * 100) if is_sharpe != 0 else 0
        decays.append(decay)

        is_pnl = is_data.get('total_pnl', 0)
        oot_pnl = oot.get('total_pnl', 0)
        is_wr = is_data.get('win_rate', 0)
        oot_wr = oot.get('win_rate', 0)

        print(
            f"{oot['config']:<55} "
            f"${is_pnl:>9,.0f} "
            f"${oot_pnl:>9,.0f} "
            f"{is_sharpe:>9.2f} "
            f"{oot_sharpe:>10.2f} "
            f"{decay:>6.1f}% "
            f"{is_wr:>5.1%} "
            f"{oot_wr:>6.1%}"
        )

    if decays:
        avg_decay = np.mean(decays)
        print(f"\n  Average Sharpe decay IS->OOT: {avg_decay:.1f}%")
        if avg_decay > -30:
            print("  VERDICT: Moderate decay — signal appears REGIME-STABLE")
        elif avg_decay > -60:
            print("  VERDICT: Significant decay — signal is PARTIALLY regime-dependent")
        else:
            print("  VERDICT: Severe decay — signal is REGIME-DEPENDENT (overfitting likely)")


# ─── NEW: CNN vs GNN vs Ensemble Comparison ──────────────────────────────────

def model_comparison(all_stats):
    """Compare CNN, GNN, and Ensemble model performance side-by-side.

    For matched parameter configs (same vol_gate, conv_threshold, hold, entry_mode),
    shows how each model type performs.
    """
    print(f"\n{'='*130}")
    print("CNN vs GNN vs ENSEMBLE MODEL COMPARISON")
    print(f"{'='*130}")

    # Group results by model_type
    by_model = defaultdict(list)
    for s in all_stats:
        model = s.get('model_type', 'cnn')
        by_model[model].append(s)

    # Print model-level summary
    print(f"\n  {'Model':<12} {'Configs':>8} {'Profitable':>11} {'Avg Sharpe':>11} "
          f"{'Med Sharpe':>11} {'Best Sharpe':>12} {'Avg P&L':>12} {'Avg WR':>8}")
    print(f"  {'-'*90}")

    model_summaries = {}
    for model in ['cnn', 'gnn', 'ensemble']:
        stats = by_model.get(model, [])
        if not stats:
            continue

        sharpes = [s.get('sharpe', 0) for s in stats]
        pnls = [s.get('total_pnl', 0) for s in stats]
        wrs = [s.get('win_rate', 0) for s in stats]
        profitable = sum(1 for p in pnls if p > 0)

        model_summaries[model] = {
            'count': len(stats),
            'profitable': profitable,
            'avg_sharpe': np.mean(sharpes),
            'med_sharpe': np.median(sharpes),
            'best_sharpe': max(sharpes),
            'avg_pnl': np.mean(pnls),
            'avg_wr': np.mean(wrs),
        }

        print(
            f"  {model.upper():<12} {len(stats):>8} "
            f"{profitable:>5} ({profitable/len(stats)*100:4.0f}%) "
            f"{np.mean(sharpes):>11.3f} "
            f"{np.median(sharpes):>11.3f} "
            f"{max(sharpes):>12.3f} "
            f"${np.mean(pnls):>11,.0f} "
            f"{np.mean(wrs):>7.1%}"
        )

    # Matched-config comparison: find configs that share the same parameters
    # across model types, differing only in model_type
    print(f"\n  --- Matched-Config Pairwise Comparison ---")
    print(f"  (Same vol/conv/hold/entry params, different model)")

    # Build a parameter fingerprint for each config (excluding model-specific fields)
    def param_fingerprint(config_params):
        """Extract comparable parameter fingerprint."""
        keys = ['vol_gate', 'signal_threshold', 'hold_ms', 'entry_mode',
                'chase_max_ticks', 'chase_max_reprices']
        return tuple(config_params.get(k) for k in keys)

    # Index by fingerprint
    cnn_by_fp = {}
    gnn_by_fp = {}
    ensemble_by_fp = {}

    for s in all_stats:
        model = s.get('model_type', 'cnn')
        params = s.get('config_params', {})
        fp = param_fingerprint(params)

        if model == 'cnn':
            cnn_by_fp[fp] = s
        elif model == 'gnn':
            gnn_by_fp[fp] = s
        elif model == 'ensemble':
            ensemble_by_fp[fp] = s

    # Find common fingerprints (CNN vs GNN)
    common_cnn_gnn = set(cnn_by_fp.keys()) & set(gnn_by_fp.keys())

    if common_cnn_gnn:
        print(f"\n  CNN vs GNN ({len(common_cnn_gnn)} matched configs):")
        print(
            f"  {'Vol':>4} {'Conv':>5} {'Hold':>6} {'Entry':>7} | "
            f"{'CNN Sharpe':>10} {'CNN P&L':>10} {'CNN WR':>7} | "
            f"{'GNN Sharpe':>10} {'GNN P&L':>10} {'GNN WR':>7} | "
            f"{'Winner':>8}"
        )
        print(f"  {'-'*110}")

        cnn_wins = 0
        gnn_wins = 0

        for fp in sorted(common_cnn_gnn):
            cnn = cnn_by_fp[fp]
            gnn = gnn_by_fp[fp]
            params = cnn.get('config_params', {})

            cnn_sharpe = cnn.get('sharpe', 0)
            gnn_sharpe = gnn.get('sharpe', 0)
            winner = 'CNN' if cnn_sharpe > gnn_sharpe else 'GNN'
            if cnn_sharpe > gnn_sharpe:
                cnn_wins += 1
            else:
                gnn_wins += 1

            hold_min = params.get('hold_ms', 0) // 60000
            entry = params.get('entry_mode', 'passive')
            print(
                f"  {params.get('vol_gate', 0):>4} "
                f"{params.get('signal_threshold', 0):>5.1f} "
                f"{hold_min:>4}m "
                f"{entry:>7} | "
                f"{cnn_sharpe:>10.3f} "
                f"${cnn.get('total_pnl', 0):>9,.0f} "
                f"{cnn.get('win_rate', 0):>6.1%} | "
                f"{gnn_sharpe:>10.3f} "
                f"${gnn.get('total_pnl', 0):>9,.0f} "
                f"{gnn.get('win_rate', 0):>6.1%} | "
                f"{winner:>8}"
            )

        print(f"\n  Score: CNN wins {cnn_wins}, GNN wins {gnn_wins} "
              f"(out of {len(common_cnn_gnn)} matched configs)")
        if cnn_wins + gnn_wins > 0:
            cnn_pct = cnn_wins / (cnn_wins + gnn_wins) * 100
            print(f"  CNN win rate: {cnn_pct:.0f}%")
    else:
        print("\n  No matched CNN-GNN config pairs found.")

    # CNN vs Ensemble comparison
    common_cnn_ens = set(cnn_by_fp.keys()) & set(ensemble_by_fp.keys())
    if common_cnn_ens:
        print(f"\n  CNN vs Ensemble ({len(common_cnn_ens)} matched configs):")
        print(
            f"  {'Vol':>4} {'Conv':>5} {'Mode':>12} | "
            f"{'CNN Sharpe':>10} {'CNN P&L':>10} | "
            f"{'Ens Sharpe':>10} {'Ens P&L':>10} | "
            f"{'Improvement':>12}"
        )
        print(f"  {'-'*90}")

        improvements = []
        for fp in sorted(common_cnn_ens):
            cnn = cnn_by_fp[fp]
            ens = ensemble_by_fp[fp]
            params = ens.get('config_params', {})
            agree_mode = params.get('agreement_mode', 'average')

            cnn_sharpe = cnn.get('sharpe', 0)
            ens_sharpe = ens.get('sharpe', 0)
            improvement = ((ens_sharpe - cnn_sharpe) / abs(cnn_sharpe) * 100) if cnn_sharpe != 0 else 0
            improvements.append(improvement)

            print(
                f"  {params.get('vol_gate', 0):>4} "
                f"{params.get('signal_threshold', 0):>5.1f} "
                f"{agree_mode:>12} | "
                f"{cnn_sharpe:>10.3f} "
                f"${cnn.get('total_pnl', 0):>9,.0f} | "
                f"{ens_sharpe:>10.3f} "
                f"${ens.get('total_pnl', 0):>9,.0f} | "
                f"{improvement:>+10.1f}%"
            )

        if improvements:
            avg_imp = np.mean(improvements)
            print(f"\n  Average Sharpe improvement from ensemble: {avg_imp:+.1f}%")
            if avg_imp > 10:
                print("  VERDICT: Ensemble ADDS VALUE over CNN alone")
            elif avg_imp > -10:
                print("  VERDICT: Ensemble roughly EQUAL to CNN")
            else:
                print("  VERDICT: Ensemble HURTS performance (model noise cancellation ineffective)")

    # Best config per model type
    print(f"\n  --- Best Config Per Model Type ---")
    for model in ['cnn', 'gnn', 'ensemble']:
        stats = by_model.get(model, [])
        if not stats:
            continue
        best = max(stats, key=lambda x: x.get('sharpe', 0))
        print(
            f"  {model.upper():<10} "
            f"Sharpe={best.get('sharpe', 0):.3f}  "
            f"P&L=${best.get('total_pnl', 0):,.0f}  "
            f"WR={best.get('win_rate', 0):.1%}  "
            f"Trades={best.get('n_trades', 0)}  "
            f"Config={best['config']}"
        )

    # Ensemble agreement mode comparison
    ensemble_stats = by_model.get('ensemble', [])
    if ensemble_stats:
        print(f"\n  --- Ensemble Agreement Mode Comparison ---")
        by_mode = defaultdict(list)
        for s in ensemble_stats:
            mode = s.get('config_params', {}).get('agreement_mode', 'average')
            by_mode[mode].append(s)

        print(f"  {'Mode':<15} {'N':>4} {'Avg Sharpe':>11} {'Best Sharpe':>12} "
              f"{'Avg P&L':>12} {'Avg WR':>8} {'Profitable':>11}")
        print(f"  {'-'*80}")

        for mode in ['both_agree', 'average', 'max_signal']:
            stats = by_mode.get(mode, [])
            if not stats:
                continue
            sharpes = [s.get('sharpe', 0) for s in stats]
            pnls = [s.get('total_pnl', 0) for s in stats]
            wrs = [s.get('win_rate', 0) for s in stats]
            profitable = sum(1 for p in pnls if p > 0)
            print(
                f"  {mode:<15} {len(stats):>4} "
                f"{np.mean(sharpes):>11.3f} "
                f"{max(sharpes):>12.3f} "
                f"${np.mean(pnls):>11,.0f} "
                f"{np.mean(wrs):>7.1%} "
                f"{profitable:>5} ({profitable/len(stats)*100:4.0f}%)"
            )

    return model_summaries


def fill_rate_comparison(all_stats):
    """Compare fill rates across models — critical for understanding signal quality."""
    print(f"\n{'='*130}")
    print("FILL RATE COMPARISON BY MODEL")
    print(f"{'='*130}")

    by_model = defaultdict(list)
    for s in all_stats:
        model = s.get('model_type', 'cnn')
        by_model[model].append(s)

    # For chase mode only (sweeps 1, 7) — most comparable
    print(f"\n  Chase mode configs only:")
    print(f"  {'Model':<10} {'Avg Fill%':>10} {'Med Fill%':>10} "
          f"{'Avg Trades':>11} {'Avg Signals':>12}")
    print(f"  {'-'*60}")

    for model in ['cnn', 'gnn', 'ensemble']:
        stats = [s for s in by_model.get(model, [])
                 if s.get('config_params', {}).get('entry_mode') == 'chase'
                 and s.get('n_trades', 0) > 0]
        if not stats:
            continue
        fill_rates = [s.get('fill_rate', 0) for s in stats]
        trade_counts = [s.get('n_trades', 0) for s in stats]
        print(
            f"  {model.upper():<10} "
            f"{np.mean(fill_rates):>9.1%} "
            f"{np.median(fill_rates):>9.1%} "
            f"{np.mean(trade_counts):>11.0f} "
            f"{'N/A':>12}"
        )


def regime_stability_analysis(all_stats):
    """Identify configs that are regime-stable vs regime-dependent.
    Uses daily P&L series to check for structural breaks.
    """
    print(f"\n{'='*130}")
    print("REGIME STABILITY ANALYSIS")
    print(f"{'='*130}")

    stable_configs = []
    unstable_configs = []

    for s in all_stats:
        daily_series = s.get('daily_pnl_series', [])
        if len(daily_series) < 20:
            continue

        pnls = [d['pnl'] for d in daily_series]

        # Split into first half and second half
        mid = len(pnls) // 2
        first_half = pnls[:mid]
        second_half = pnls[mid:]

        mean_1 = np.mean(first_half)
        mean_2 = np.mean(second_half)
        std_1 = np.std(first_half) if len(first_half) > 1 else 1e-8
        std_2 = np.std(second_half) if len(second_half) > 1 else 1e-8

        sharpe_1 = (mean_1 / std_1) * np.sqrt(252) if std_1 > 1e-10 else 0
        sharpe_2 = (mean_2 / std_2) * np.sqrt(252) if std_2 > 1e-10 else 0

        # Check rolling 10-day performance
        if len(pnls) >= 10:
            rolling_means = []
            for i in range(len(pnls) - 9):
                rolling_means.append(np.mean(pnls[i:i+10]))
            negative_windows = sum(1 for m in rolling_means if m < 0)
            pct_negative = negative_windows / len(rolling_means)
        else:
            pct_negative = 0

        stability_info = {
            'config': s['config'],
            'model_type': s.get('model_type', 'cnn'),
            'overall_sharpe': s.get('sharpe', 0),
            'first_half_sharpe': round(sharpe_1, 2),
            'second_half_sharpe': round(sharpe_2, 2),
            'pct_negative_windows': round(pct_negative, 3),
            'total_pnl': s.get('total_pnl', 0),
        }

        # Regime-stable: both halves profitable and consistent direction
        if sharpe_1 > 0 and sharpe_2 > 0 and pct_negative < 0.3:
            stable_configs.append(stability_info)
        else:
            unstable_configs.append(stability_info)

    stable_configs.sort(key=lambda x: x['overall_sharpe'], reverse=True)
    unstable_configs.sort(key=lambda x: x['overall_sharpe'], reverse=True)

    print(f"\nREGIME-STABLE configs ({len(stable_configs)}):")
    print(f"  (Both halves Sharpe > 0, < 30% negative 10-day windows)")
    print(f"  {'Config':<55} {'Model':>8} {'Sharpe':>7} {'1H Sharpe':>9} {'2H Sharpe':>9} {'Neg%':>5} {'P&L':>10}")
    print(f"  {'-'*110}")
    for c in stable_configs[:15]:
        print(
            f"  {c['config']:<55} "
            f"{c['model_type']:>8} "
            f"{c['overall_sharpe']:>7.2f} "
            f"{c['first_half_sharpe']:>9.2f} "
            f"{c['second_half_sharpe']:>9.2f} "
            f"{c['pct_negative_windows']:>4.0%} "
            f"${c['total_pnl']:>9,.0f}"
        )

    print(f"\nREGIME-DEPENDENT configs ({len(unstable_configs)}):")
    for c in unstable_configs[:10]:
        print(
            f"  {c['config']:<55} "
            f"{c['model_type']:>8} "
            f"{c['overall_sharpe']:>7.2f} "
            f"{c['first_half_sharpe']:>9.2f} "
            f"{c['second_half_sharpe']:>9.2f} "
            f"{c['pct_negative_windows']:>4.0%} "
            f"${c['total_pnl']:>9,.0f}"
        )


def parameter_sensitivity_report(all_stats):
    """Analyze which parameters matter most."""
    print(f"\n{'='*130}")
    print("PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*130}")

    # Group by parameter values
    param_groups = defaultdict(list)

    for s in all_stats:
        params = s.get('config_params', {})
        sharpe = s.get('sharpe', 0)

        for key, val in params.items():
            if val is not None and key not in ('label', 'sweep', 'time_gate', 'day_gate',
                                                'agreement_mode', 'model_type'):
                param_groups[f"{key}={val}"].append(sharpe)

    # Compute mean Sharpe per parameter value
    print(f"\n  {'Parameter=Value':<40} {'N':>5} {'Mean Sharpe':>12} {'Median':>8} {'Std':>8}")
    print(f"  {'-'*80}")

    param_impact = []
    for param_val, sharpes in sorted(param_groups.items()):
        if len(sharpes) < 3:
            continue
        mean_s = np.mean(sharpes)
        param_impact.append((param_val, len(sharpes), mean_s, np.median(sharpes), np.std(sharpes)))

    param_impact.sort(key=lambda x: x[2], reverse=True)

    for pv, n, mean_s, med_s, std_s in param_impact[:30]:
        print(f"  {pv:<40} {n:>5} {mean_s:>12.3f} {med_s:>8.3f} {std_s:>8.3f}")

    # Model type as a parameter
    print(f"\n  --- Model Type Impact ---")
    model_sharpes = defaultdict(list)
    for s in all_stats:
        model = s.get('model_type', 'cnn')
        model_sharpes[model].append(s.get('sharpe', 0))

    for model, sharpes in sorted(model_sharpes.items()):
        print(f"  model_type={model:<12} N={len(sharpes):>4}  "
              f"mean={np.mean(sharpes):.3f}  med={np.median(sharpes):.3f}  std={np.std(sharpes):.3f}")


def find_optimal_config(all_stats):
    """Find the optimal config using multiple criteria."""
    print(f"\n{'='*130}")
    print("OPTIMAL CONFIG SEARCH")
    print(f"{'='*130}")

    # Score each config on multiple dimensions
    scored = []
    for s in all_stats:
        sharpe = s.get('sharpe', 0)
        sortino = s.get('sortino', 0)
        pf = s.get('profit_factor', 0)
        wr = s.get('win_rate', 0)
        n_trades = s.get('n_trades', 0)
        max_dd = s.get('max_dd', 1e-8)
        total_pnl = s.get('total_pnl', 0)

        # Minimum trade count threshold
        if n_trades < 20:
            continue

        # Composite score: Sharpe-weighted with penalties
        score = (
            sharpe * 0.35 +
            sortino * 0.15 +
            min(pf, 5) * 0.10 +  # cap PF contribution
            (total_pnl / max(max_dd, 1)) * 0.20 +  # recovery factor
            wr * 5.0 * 0.10 +  # win rate contribution
            min(n_trades / 100, 1.0) * 0.10  # trade frequency (penalize too few)
        )

        scored.append({**s, '_composite_score': score})

    scored.sort(key=lambda x: x['_composite_score'], reverse=True)

    print(f"\nTop 10 by composite score (Sharpe 35% + Sortino 15% + PF 10% + Recovery 20% + WR 10% + Freq 10%):")
    print(
        f"{'#':>3} {'Config':<55} {'Model':>8} {'Score':>7} {'Sharpe':>7} {'P&L':>10} "
        f"{'Trades':>7} {'WR':>6} {'MaxDD':>8}"
    )
    print("-" * 120)

    for i, s in enumerate(scored[:10]):
        print(
            f"{i+1:>3} {s['config']:<55} "
            f"{s.get('model_type', 'cnn'):>8} "
            f"{s['_composite_score']:>7.2f} "
            f"{s.get('sharpe', 0):>7.2f} "
            f"${s.get('total_pnl', 0):>9,.0f} "
            f"{s.get('n_trades', 0):>7} "
            f"{s.get('win_rate', 0):>5.1%} "
            f"${s.get('max_dd', 0):>7,.0f}"
        )

    if scored:
        best = scored[0]
        print(f"\n  RECOMMENDED CONFIG: {best['config']} (model: {best.get('model_type', 'cnn')})")
        print(f"  Parameters: {json.dumps(best.get('config_params', {}), indent=4)}")


def long_short_analysis(all_stats):
    """Analyze long vs short performance."""
    print(f"\n{'='*130}")
    print("LONG vs SHORT ANALYSIS")
    print(f"{'='*130}")

    sweep6 = [s for s in all_stats if s.get('sweep') == 6]
    if not sweep6:
        sweep6 = all_stats

    print(
        f"  {'Config':<55} {'Model':>8} {'L Count':>8} {'L WR':>6} {'L Avg$':>8} "
        f"{'S Count':>8} {'S WR':>6} {'S Avg$':>8} {'L-S Gap':>8}"
    )
    print(f"  {'-'*120}")

    for s in sorted(sweep6, key=lambda x: x.get('sharpe', 0), reverse=True)[:20]:
        l_avg = s.get('long_avg_pnl', 0)
        s_avg = s.get('short_avg_pnl', 0)
        gap = l_avg - s_avg

        print(
            f"  {s['config']:<55} "
            f"{s.get('model_type', 'cnn'):>8} "
            f"{s.get('long_count', 0):>8} "
            f"{s.get('long_wr', 0):>5.1%} "
            f"${l_avg:>7,.0f} "
            f"{s.get('short_count', 0):>8} "
            f"{s.get('short_wr', 0):>5.1%} "
            f"${s_avg:>7,.0f} "
            f"${gap:>7,.0f}"
        )


def hourly_heatmap(all_stats):
    """Show hourly P&L distribution for top configs."""
    print(f"\n{'='*130}")
    print("HOURLY P&L HEATMAP (Top 5 configs)")
    print(f"{'='*130}")

    top5 = sorted(all_stats, key=lambda x: x.get('sharpe', 0), reverse=True)[:5]

    for s in top5:
        heatmap = s.get('hourly_pnl_heatmap', {})
        if not heatmap:
            continue

        model = s.get('model_type', 'cnn')
        print(f"\n  {s['config']} [{model.upper()}]  (Sharpe={s.get('sharpe', 0):.2f})")
        for hour in sorted(heatmap.keys(), key=lambda h: int(h)):
            pnl = heatmap[hour]
            bar_len = int(abs(pnl) / max(abs(v) for v in heatmap.values()) * 30) if heatmap.values() else 0
            bar = '+' * bar_len if pnl > 0 else '-' * bar_len
            print(f"    {int(hour):>2}:00 ET  ${pnl:>8,.0f}  {bar}")


def generate_report(all_stats, is_results=None, output_file=None):
    """Generate comprehensive analysis report."""
    # Overall leaderboard
    all_stats.sort(key=lambda x: x.get('sharpe', 0), reverse=True)

    print("\n" + "#" * 130)
    print("#  OOT MEGA SWEEP — COMPREHENSIVE ANALYSIS REPORT (CNN + GNN + ENSEMBLE)")
    print(f"#  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Total configs analyzed: {len(all_stats)}")
    profitable = sum(1 for s in all_stats if s.get('total_pnl', 0) > 0)
    print(f"#  Profitable configs: {profitable}/{len(all_stats)} ({profitable/max(len(all_stats),1)*100:.0f}%)")

    # Model breakdown
    from collections import Counter
    model_counts = Counter(s.get('model_type', 'cnn') for s in all_stats)
    print(f"#  Model breakdown: {dict(model_counts)}")
    print("#" * 130)

    # Per-sweep summaries
    for sweep_num in range(1, 10):
        print_sweep_summary(all_stats, sweep_num)

    # CNN vs GNN vs Ensemble comparison (NEW)
    model_summaries = model_comparison(all_stats)

    # Fill rate comparison (NEW)
    fill_rate_comparison(all_stats)

    # IS vs OOT comparison
    if is_results:
        is_vs_oot_comparison(all_stats, is_results)

    # Regime stability
    regime_stability_analysis(all_stats)

    # Parameter sensitivity
    parameter_sensitivity_report(all_stats)

    # Optimal config
    find_optimal_config(all_stats)

    # Long/short analysis
    long_short_analysis(all_stats)

    # Hourly heatmap
    hourly_heatmap(all_stats)

    # Save report
    if output_file:
        report_data = {
            'generated': datetime.now().isoformat(),
            'total_configs': len(all_stats),
            'profitable_configs': profitable,
            'model_breakdown': dict(model_counts),
            'model_summaries': model_summaries if model_summaries else {},
            'top_10': all_stats[:10],
            'sweep_summaries': {},
        }
        for sweep_num in range(1, 10):
            sweep = [s for s in all_stats if s.get('sweep') == sweep_num]
            if sweep:
                report_data['sweep_summaries'][f'sweep_{sweep_num}'] = {
                    'count': len(sweep),
                    'profitable': sum(1 for s in sweep if s.get('total_pnl', 0) > 0),
                    'best_sharpe': max(s.get('sharpe', 0) for s in sweep),
                    'best_config': max(sweep, key=lambda x: x.get('sharpe', 0))['config'],
                    'model_type': sweep[0].get('model_type', 'cnn'),
                }

        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"\nReport saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='OOT Sweep Analysis (CNN/GNN/Ensemble)')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing OOT sweep results')
    parser.add_argument('--is-results', type=str, default=None,
                        help='IS results JSON file for comparison')
    parser.add_argument('--output', type=str, default=None,
                        help='Output report JSON file')

    args = parser.parse_args()

    print("Loading OOT results...")
    all_stats = load_oot_results(args.results_dir)

    if not all_stats:
        print("No results found. Run oot_mega_sweep.py first.")
        return

    is_results = {}
    if args.is_results:
        print("Loading IS results...")
        is_results = load_is_results(args.is_results)

    output_file = args.output or str(Path(args.results_dir) / f'oot_analysis_report_{_ts}.json')

    generate_report(all_stats, is_results, output_file)


if __name__ == '__main__':
    main()
