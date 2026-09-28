"""
run_mfe_mae_fill_sim.py -- Fill sim sweep for MFE/MAE CNN predictions.

For each fold with both:
  1. MFE/MAE preds npz (from mfe_mae_infer_folds.py)
  2. MBO data for that date

Runs fill_sim_cli with chase entry at multiple signal thresholds:
  - Absolute thresholds: 1.5, 2.0, 2.5, 3.0 (raw mfe/mae ratio)
  - Percentile thresholds: top 10%, top 5%, top 1%

Chase entry, prime hours, hold_ms=10000 (10s), conviction exit 50 bars.

Results saved to: /home/jupiter/Lvl3Quant/results/mfe_mae_fill_sim/

Usage: python3 run_mfe_mae_fill_sim.py [--preds-dir <path>] [--mbo-dir <path>]
"""

import os
import sys
import json
import subprocess
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [fill_sim] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# === Paths ===
LVL3 = Path('/home/jupiter/Lvl3Quant')
FILL_SIM_CLI = LVL3 / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3 / 'data' / 'raw' / 'mbo'
PREDS_DIR = LVL3 / 'data' / 'processed' / 'mfe_mae_fold_preds'
RESULTS_DIR = LVL3 / 'results' / 'mfe_mae_fill_sim'

# Parse args
for i, arg in enumerate(sys.argv):
    if arg == '--preds-dir' and i+1 < len(sys.argv):
        PREDS_DIR = Path(sys.argv[i+1])
    if arg == '--mbo-dir' and i+1 < len(sys.argv):
        MBO_DIR = Path(sys.argv[i+1])

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === Threshold configs ===
# Absolute thresholds on |signal| (= mfe_pred/mae_pred ratio)
ABS_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]

# Percentile thresholds computed per-day from distribution of |signal|
PERCENTILE_THRESHOLDS = [90, 95, 99, 99.5, 99.9]  # top 10%, 5%, 1%, 0.5%, 0.1%

# Fixed execution params (proven from prior fill sim work)
HOLD_MS = 10000        # 10s hold
CONVICTION_BARS = 50   # ~5s conviction exit
STOP_LOSS_TICKS = 8    # 8 tick stop
TAKE_PROFIT_TICKS = 16 # 16 tick TP (2:1 R/R)
LATENCY_MS = 2         # 2ms typical latency


def get_mbo_file(date_str: str) -> Optional[Path]:
    """Find MBO file for date YYYY-MM-DD."""
    compact = date_str.replace('-', '')
    candidates = list(MBO_DIR.glob(f'*{compact}*.dbn.zst')) + list(MBO_DIR.glob(f'*{compact}*.dbn'))
    return candidates[0] if candidates else None


def compute_percentile_threshold(preds_file: Path, percentile: int) -> float:
    """Load preds npz and return the given percentile of |signal|."""
    data = np.load(preds_file)
    sig = np.abs(data['predictions'])
    nonzero = sig[sig > 0]
    if len(nonzero) == 0:
        return 999.0  # no signals
    return float(np.percentile(nonzero, percentile))


def run_fill_sim(mbo_file: Path, preds_file: Path, output_file: Path, threshold: float) -> Optional[dict]:
    """Run fill_sim_cli for one (date, threshold) combo."""
    cmd = [
        str(FILL_SIM_CLI),
        '--mbo-file', str(mbo_file),
        '--predictions', str(preds_file),
        '--output', str(output_file),
        '--signal-threshold', str(threshold),
        '--hold-ms', str(HOLD_MS),
        '--stop-loss-ticks', str(STOP_LOSS_TICKS),
        '--take-profit-ticks', str(TAKE_PROFIT_TICKS),
        '--conviction-exit-bars', str(CONVICTION_BARS),
        '--chase-entry',
        '--chase-max-ticks', '2',
        '--chase-max-reprices', '5',
        '--prime-hours',
        '--latency-ms', str(LATENCY_MS),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.error(f'fill_sim_cli failed: {result.stderr[:300]}')
            return None
        if output_file.exists():
            with open(output_file) as f:
                return json.load(f)
    except subprocess.TimeoutExpired:
        log.error(f'fill_sim_cli timed out for {preds_file.name}')
    except Exception as e:
        log.error(f'Error running fill_sim_cli: {e}')
    return None


def main():
    log.info('=' * 70)
    log.info('MFE/MAE CNN Fill Sim Sweep')
    log.info(f'Preds dir: {PREDS_DIR}')
    log.info(f'MBO dir:   {MBO_DIR}')
    log.info(f'Results:   {RESULTS_DIR}')
    log.info(f'Absolute thresholds: {ABS_THRESHOLDS}')
    log.info(f'Percentile thresholds (top N%): {[100-p for p in PERCENTILE_THRESHOLDS]}')
    log.info('=' * 70)

    if not FILL_SIM_CLI.exists():
        log.error(f'fill_sim_cli not found at {FILL_SIM_CLI}')
        sys.exit(1)

    # Find available prediction files
    pred_files = sorted(PREDS_DIR.glob('*_mfe_mae_preds.npz'))
    if not pred_files:
        log.error(f'No prediction files in {PREDS_DIR}')
        log.info('Run mfe_mae_infer_folds.py on Uranus first, then sync preds here.')
        sys.exit(1)

    log.info(f'Found {len(pred_files)} prediction files')

    all_results = []
    n_run = 0
    n_no_mbo = 0

    for preds_file in pred_files:
        date_str = preds_file.name[:10]
        mbo_file = get_mbo_file(date_str)

        if mbo_file is None:
            log.warning(f'{date_str}: no MBO file found, skipping')
            n_no_mbo += 1
            continue

        log.info(f'\n--- {date_str} ---')

        # Compute percentile thresholds for this day
        pct_thresholds = {}
        for pct in PERCENTILE_THRESHOLDS:
            t = compute_percentile_threshold(preds_file, pct)
            pct_thresholds[f'top_{100-pct}pct'] = t
            log.info(f'  Percentile threshold top {100-pct}%: {t:.4f}')

        # Run all threshold combos
        day_results = []
        for thresh_name, threshold in (
            [(f'abs_{t}', t) for t in ABS_THRESHOLDS] +
            [(k, v) for k, v in pct_thresholds.items()]
        ):
            out_file = RESULTS_DIR / f'{date_str}_{thresh_name}.json'

            if out_file.exists():
                with open(out_file) as f:
                    sim_result = json.load(f)
                log.info(f'  {thresh_name}: loaded from cache')
            else:
                sim_result = run_fill_sim(mbo_file, preds_file, out_file, threshold)

            if sim_result is None:
                continue

            summary = sim_result.get('summary', sim_result)
            n_trades = summary.get('n_trades', 0)
            pnl = summary.get('total_pnl', summary.get('net_pnl', 0))
            sortino = summary.get('sortino', summary.get('sortino_ratio', float('nan')))
            wr = summary.get('win_rate', float('nan'))

            log.info(
                f'  {thresh_name:15s}: threshold={threshold:.3f} '
                f'trades={n_trades:3d} PnL={pnl:+.1f} '
                f'Sortino={sortino:.2f} WR={wr:.1%}'
            )

            day_results.append({
                'date': date_str,
                'threshold_name': thresh_name,
                'threshold_value': threshold,
                'n_trades': n_trades,
                'total_pnl': pnl,
                'sortino': sortino,
                'win_rate': wr,
                'raw': summary,
            })
            n_run += 1

        all_results.extend(day_results)

    # Aggregate by threshold across all days
    log.info('\n' + '=' * 70)
    log.info('AGGREGATED RESULTS BY THRESHOLD')
    log.info('=' * 70)

    if all_results:
        from collections import defaultdict
        by_threshold = defaultdict(list)
        for r in all_results:
            by_threshold[r['threshold_name']].append(r)

        summary_rows = []
        for tname, rows in sorted(by_threshold.items()):
            n_days_traded = len([r for r in rows if r['n_trades'] > 0])
            total_trades = sum(r['n_trades'] for r in rows)
            total_pnl = sum(r['total_pnl'] for r in rows)
            avg_pnl_per_day = total_pnl / max(len(rows), 1)
            valid_sortinos = [r['sortino'] for r in rows if not (r['sortino'] != r['sortino'])]
            avg_sortino = float(np.mean(valid_sortinos)) if valid_sortinos else float('nan')
            valid_wr = [r['win_rate'] for r in rows if not (r['win_rate'] != r['win_rate'])]
            avg_wr = float(np.mean(valid_wr)) if valid_wr else float('nan')
            trades_per_day = total_trades / max(len(rows), 1)

            log.info(
                f'{tname:15s}: days={n_days_traded}/{len(rows)} '
                f'trades={total_trades} ({trades_per_day:.1f}/day) '
                f'total_PnL={total_pnl:+.1f} avg/day={avg_pnl_per_day:+.1f} '
                f'avgSortino={avg_sortino:.2f} avgWR={avg_wr:.1%}'
            )

            summary_rows.append({
                'threshold': tname,
                'n_days': len(rows),
                'days_with_trades': n_days_traded,
                'total_trades': total_trades,
                'trades_per_day': trades_per_day,
                'total_pnl': total_pnl,
                'avg_pnl_per_day': avg_pnl_per_day,
                'avg_sortino': avg_sortino,
                'avg_win_rate': avg_wr,
            })

        final_summary = {
            'run_date': datetime.now().isoformat(),
            'n_days_processed': len(pred_files) - n_no_mbo,
            'n_days_no_mbo': n_no_mbo,
            'threshold_summary': summary_rows,
            'per_day_results': all_results,
        }

        summary_file = RESULTS_DIR / 'mfe_mae_fill_sim_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        log.info(f'\nSummary saved: {summary_file}')

    log.info(f'\nTotal sims run: {n_run} | No MBO: {n_no_mbo}')


if __name__ == '__main__':
    main()
