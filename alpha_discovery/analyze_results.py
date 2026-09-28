"""
Results Auto-Analyzer — Parse overnight run outputs and generate Discord reports.

Usage:
    python alpha_discovery/analyze_results.py                    # Analyze latest run
    python alpha_discovery/analyze_results.py --file results/overnight_final_*.json
    python alpha_discovery/analyze_results.py --monitor          # Watch for new results
"""

import json
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "alpha_discovery" / "results"

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('analyzer')


def find_latest_result(prefix: str = "overnight_final") -> Optional[Path]:
    """Find the most recent result file with given prefix."""
    files = sorted(RESULTS_DIR.glob(f"{prefix}_*.json"))
    return files[-1] if files else None


def find_latest_phase_result(phase: int) -> Optional[Path]:
    """Find the most recent result for a specific phase."""
    files = sorted(RESULTS_DIR.glob(f"overnight_phase{phase}_*.json"))
    return files[-1] if files else None


def format_phase1_report(data: dict) -> str:
    """Format Phase 1 (honest baseline) results."""
    lines = ["**Phase 1: Honest Baseline**"]
    lines.append(f"Data: {data.get('n_days', '?')} days, {data.get('n_bars', 0):,} bars")
    lines.append(f"Features: {data.get('n_features_used', '?')}")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Horizon':<10} {'IC':>8} {'t-stat':>8} {'ICIR':>8} {'Consist':>8} {'AvgMove':>8} {'Viable':>8}")
    lines.append("-" * 68)

    horizons = data.get('horizons', {})
    for hz in sorted(horizons.keys()):
        r = horizons[hz]
        if 'error' in r:
            lines.append(f"{hz:<10} {'ERROR':>8}")
            continue
        viable = 'YES' if r.get('cost_ratio', 0) > 0.5 and r.get('tstat', 0) > 2 else 'no'
        lines.append(
            f"{hz:<10} {r['ic']:>8.4f} {r['tstat']:>8.1f} {r['icir']:>8.2f} "
            f"{r.get('consistency', 0):>7.0%} {r.get('avg_move_ticks', 0):>7.1f}t "
            f"{'** ' + viable + ' **' if viable == 'YES' else viable:>8}"
        )
    lines.append("```")

    # Top features from best horizon
    best_hz = max(
        ((hz, r) for hz, r in horizons.items() if 'error' not in r),
        key=lambda x: abs(x[1].get('tstat', 0)),
        default=(None, None)
    )
    if best_hz[0]:
        r = best_hz[1]
        lines.append(f"\nBest: **{best_hz[0]}** (IC={r['ic']:.4f}, t={r['tstat']:.1f})")
        if 'top_features' in r:
            lines.append("Top features: " + ", ".join(f[0] for f in r['top_features'][:5]))

    return "\n".join(lines)


def format_phase2_report(data: dict) -> str:
    """Format Phase 2 (multi-timeframe) results."""
    lines = ["**Phase 2: Multi-Timeframe**"]
    lines.append("```")
    lines.append(f"{'Timeframe':<10} {'IC':>8} {'t-stat':>8} {'ICIR':>8} {'AvgMove':>8} {'Bars/Day':>8}")
    lines.append("-" * 58)

    for tf in sorted(data.get('timeframes', {}).keys()):
        r = data['timeframes'][tf]
        if 'error' in r:
            lines.append(f"{tf:<10} {'ERROR':>8}")
            continue
        lines.append(
            f"{tf:<10} {r['ic']:>8.4f} {r['tstat']:>8.1f} {r['icir']:>8.2f} "
            f"{r.get('avg_move_ticks', 0):>7.1f}t {r.get('trades_per_day', 0):>7.0f}"
        )
    lines.append("```")
    return "\n".join(lines)


def format_phase3_report(data: dict) -> str:
    """Format Phase 3 (feature discovery) results."""
    lines = ["**Phase 3: Feature Discovery**"]
    lines.append(f"Forward-dominant: {data.get('n_forward_dominant', '?')} / {data.get('n_features_analyzed', '?')}")

    if data.get('decomposition'):
        lines.append("\nTop 10 by forward IC:")
        lines.append("```")
        for r in data['decomposition'][:10]:
            marker = "FWD" if r['is_forward_dominant'] else "BWD"
            lines.append(f"  {r['feature']:<28s} fwd={r['fwd_ic']:.4f} bwd={r['bwd_ic']:.4f} [{marker}]")
        lines.append("```")

    if data.get('new_features'):
        lines.append("\nNew acceleration features:")
        lines.append("```")
        for r in data['new_features'][:5]:
            marker = "FWD" if r['is_forward_dominant'] else "BWD"
            lines.append(f"  {r['feature']:<32s} fwd={r['fwd_ic']:.4f} [{marker}]")
        lines.append("```")

    return "\n".join(lines)


def format_phase4_report(data: dict) -> str:
    """Format Phase 4 (model robustness) results."""
    lines = ["**Phase 4: Model Robustness**"]
    r = data.get('results', {})

    if 'holdout_split' in r:
        h = r['holdout_split']
        lines.append(f"Holdout split: IC={h['ic']:.4f} (train={h['train_days']}d, test={h['test_days']}d)")

    if 'ridge_benchmark' in r and 'ic' in r['ridge_benchmark']:
        ridge_ic = r['ridge_benchmark']['ic']
        holdout_ic = r.get('holdout_split', {}).get('ic', 0)
        lines.append(f"Ridge benchmark: IC={ridge_ic:.4f} (LightGBM: {holdout_ic:.4f})")

    if 'hyperparam_sweep' in r and r['hyperparam_sweep']:
        hp = r['hyperparam_sweep']
        lines.append(f"HP sweep: IC range [{hp[-1]['ic']:.4f}, {hp[0]['ic']:.4f}]")
        best = hp[0]
        lines.append(f"  Best: depth={best['max_depth']}, min_child={best['min_child']}")

    return "\n".join(lines)


def format_phase5_report(data: dict) -> str:
    """Format Phase 5 (model progression) results."""
    lines = ["**Phase 5: Model Progression (CNN/LSTM/Transformer vs LightGBM)**"]

    for hz, hz_res in data.get('results', {}).items():
        lgbm_ic = hz_res.get('lgbm', {}).get('ic', 0)
        lines.append(f"\n`{hz}` — LightGBM IC={lgbm_ic:.4f}")

        for tr in hz_res.get('temporal', []):
            if 'error' in tr:
                lines.append(f"  {tr.get('model_type', '?')}: ERROR")
                continue
            ic = tr.get('ic', 0)
            delta = ic - lgbm_ic
            marker = "BETTER" if abs(ic) > abs(lgbm_ic) else ""
            lines.append(
                f"  {tr.get('model_type', '?'):15s}: IC={ic:.4f} (delta={delta:+.4f}) "
                f"t={tr.get('tstat', 0):.1f} params={tr.get('n_params', 0):,} {marker}"
            )

    return "\n".join(lines)


def generate_full_report(final_data: Optional[dict] = None) -> str:
    """Generate full Discord report from overnight results.

    If final_data is None, tries to find individual phase results.
    """
    lines = [
        "# Overnight Alpha Discovery Results",
        f"*{datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]

    if final_data and 'phases' in final_data:
        phases = final_data['phases']
    else:
        # Try to load individual phase results
        phases = {}
        for phase_num in range(1, 6):
            path = find_latest_phase_result(phase_num)
            if path:
                with open(str(path)) as f:
                    phases[f'phase_{phase_num}'] = json.load(f)

    formatters = {
        'phase_1': format_phase1_report,
        'phase_2': format_phase2_report,
        'phase_3': format_phase3_report,
        'phase_4': format_phase4_report,
        'phase_5': format_phase5_report,
    }

    for phase_key, formatter in formatters.items():
        if phase_key in phases:
            lines.append(formatter(phases[phase_key]))
            lines.append("")

    # Verdict
    lines.append("---")
    lines.append("**VERDICT:**")

    if 'phase_1' in phases:
        horizons = phases['phase_1'].get('horizons', {})
        viable = [
            (hz, r) for hz, r in horizons.items()
            if 'error' not in r and r.get('tstat', 0) > 2 and r.get('cost_ratio', 0) > 0.3
        ]
        if viable:
            best = max(viable, key=lambda x: x[1]['tstat'])
            lines.append(f"Best viable horizon: **{best[0]}** (IC={best[1]['ic']:.4f}, t={best[1]['tstat']:.1f})")
        else:
            lines.append("No statistically significant viable horizon found.")

    if 'phase_5' in phases:
        # Check if any temporal model beat LightGBM
        for hz, hz_res in phases['phase_5'].get('results', {}).items():
            lgbm_ic = hz_res.get('lgbm', {}).get('ic', 0)
            for tr in hz_res.get('temporal', []):
                if 'error' not in tr and abs(tr.get('ic', 0)) > abs(lgbm_ic):
                    lines.append(
                        f"Temporal model **{tr['model_type']}** beats LightGBM at {hz}: "
                        f"IC={tr['ic']:.4f} vs {lgbm_ic:.4f}"
                    )

    return "\n".join(lines)


def monitor_results(check_interval: int = 60):
    """Watch for new results and print reports as they appear."""
    seen_files = set(RESULTS_DIR.glob("overnight_phase*_*.json"))

    log.info(f"Monitoring {RESULTS_DIR} for new results (every {check_interval}s)...")
    while True:
        current_files = set(RESULTS_DIR.glob("overnight_phase*_*.json"))
        new_files = current_files - seen_files

        for f in sorted(new_files):
            log.info(f"\n{'='*60}")
            log.info(f"NEW RESULT: {f.name}")
            log.info(f"{'='*60}")

            with open(str(f)) as fh:
                data = json.load(fh)

            phase = data.get('phase', '')
            formatter = {
                'honest_baseline': format_phase1_report,
                'multi_timeframe': format_phase2_report,
                'feature_discovery': format_phase3_report,
                'model_robustness': format_phase4_report,
                'model_progression': format_phase5_report,
            }.get(phase)

            if formatter:
                log.info(formatter(data))
            else:
                log.info(json.dumps(data, indent=2, default=str)[:2000])

            seen_files.add(f)

        # Also check for final report
        final_files = set(RESULTS_DIR.glob("overnight_final_*.json")) - seen_files
        for f in sorted(final_files):
            log.info(f"\n{'='*60}")
            log.info(f"FINAL REPORT: {f.name}")
            log.info(f"{'='*60}")
            with open(str(f)) as fh:
                data = json.load(fh)
            log.info(generate_full_report(data))
            seen_files.add(f)

        time.sleep(check_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze overnight alpha results')
    parser.add_argument('--file', type=str, help='Specific result file to analyze')
    parser.add_argument('--monitor', action='store_true', help='Watch for new results')
    parser.add_argument('--phase', type=int, help='Analyze specific phase (1-5)')
    args = parser.parse_args()

    if args.monitor:
        monitor_results()
    elif args.file:
        with open(args.file) as f:
            data = json.load(f)
        print(generate_full_report(data))
    elif args.phase:
        path = find_latest_phase_result(args.phase)
        if path:
            with open(str(path)) as f:
                data = json.load(f)
            formatter = {
                1: format_phase1_report,
                2: format_phase2_report,
                3: format_phase3_report,
                4: format_phase4_report,
                5: format_phase5_report,
            }.get(args.phase)
            if formatter:
                print(formatter(data))
        else:
            print(f"No results found for phase {args.phase}")
    else:
        # Try latest final report
        path = find_latest_result()
        if path:
            with open(str(path)) as f:
                data = json.load(f)
            print(generate_full_report(data))
        else:
            # Try individual phases
            print(generate_full_report())
