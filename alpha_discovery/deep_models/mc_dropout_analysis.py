#!/usr/bin/env python3
"""
MC Dropout Analysis — Post-inference analysis and fill_sim integration
=======================================================================
After running mc_dropout_inference.py, use this script to:

1. Load all per-day _mc_preds.npz files
2. Identify the optimal uncertainty threshold for gating
3. Compare IC with vs without uncertainty gating
4. Generate gated prediction files ready for run_wf_fill_sim.py
5. Print recommendations

Usage:
    python mc_dropout_analysis.py \
        --mc-dir results/mc_dropout \
        --output-dir results/mc_dropout/analysis

    # Also run a fill_sim comparison with gated vs ungated:
    python mc_dropout_analysis.py \
        --mc-dir results/mc_dropout \
        --output-dir results/mc_dropout/analysis \
        --run-fillsim \
        --mbo-dir ../../data/raw/mbo
"""

import sys
import os
import gc
import json
import argparse
import logging
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

_THIS = Path(__file__).resolve()
ROOT  = _THIS.parent.parent.parent

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'mc_analysis_{_ts}.log'
    logger = logging.getLogger('mc_analysis')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_file), mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Signal processing helpers (mirror run_wf_fill_sim.py)
# ---------------------------------------------------------------------------

CNN_OFFSET  = 19   # window_size - 1
HORIZON     = 100  # bars for IC (~10s)
BARS_PER_SEC = 10


def zscore_expanding(arr: np.ndarray) -> np.ndarray:
    """Expanding-window z-score (no look-ahead). Requires >=50 non-NaN samples."""
    result = np.full_like(arr, np.nan, dtype=np.float64)
    running_sum = 0.0
    running_sq  = 0.0
    count = 0
    for i in range(len(arr)):
        v = arr[i]
        if np.isnan(v):
            continue
        running_sum += v
        running_sq  += v * v
        count += 1
        if count >= 50:
            mean = running_sum / count
            var  = (running_sq / count) - mean * mean
            std  = max(np.sqrt(var), 1e-8)
            result[i] = (v - mean) / std
    return result


def compute_ic(preds: np.ndarray, mid: np.ndarray, horizon: int = HORIZON) -> float | None:
    n = len(mid)
    fwd_ret = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - horizon):
        if mid[i] > 0:
            fwd_ret[i] = (mid[i + horizon] - mid[i]) / mid[i]
    valid = ~np.isnan(preds) & ~np.isnan(fwd_ret) & (preds != 0) & np.isfinite(fwd_ret)
    if valid.sum() < 50:
        return None
    ic, _ = stats.spearmanr(preds[valid].astype(np.float64), fwd_ret[valid])
    return float(ic)


# ---------------------------------------------------------------------------
# Load all MC output files
# ---------------------------------------------------------------------------

def load_mc_results(mc_dir: Path, logger: logging.Logger) -> list[dict]:
    """Load all per-day *_mc_preds.npz files."""
    files = sorted(mc_dir.glob('*_mc_preds.npz'))
    if not files:
        logger.error(f"No *_mc_preds.npz files found in {mc_dir}")
        return []

    logger.info(f"Loading {len(files)} MC result files from {mc_dir}")
    days = []
    for f in files:
        date = f.name.replace('_mc_preds.npz', '')
        npz  = np.load(str(f))
        days.append({
            'date':       date,
            'pred_mean':  npz['pred_mean'].astype(np.float64),
            'pred_std':   npz['pred_std'].astype(np.float64),
            'mid_prices': npz['mid_prices'].astype(np.float64),
        })
        npz.close()

    logger.info(f"Loaded {len(days)} days ({days[0]['date']} to {days[-1]['date']})")
    return days


# ---------------------------------------------------------------------------
# Threshold sweep — find optimal uncertainty gate
# ---------------------------------------------------------------------------

def sweep_uncertainty_thresholds(
    days: list[dict],
    thresholds: list[float],
    logger: logging.Logger,
) -> list[dict]:
    """
    For each uncertainty threshold, compute:
      - IC (with bars pred_std > thr zeroed out)
      - N bars remaining (fraction of bars kept)
      - IC improvement vs no gating

    Returns list of dicts sorted by gated IC.
    """
    logger.info("\n" + "=" * 75)
    logger.info("UNCERTAINTY THRESHOLD SWEEP")
    logger.info(f"Thresholds tested: {thresholds}")
    logger.info("=" * 75)

    # Baseline (no gating)
    baseline_ics = []
    for d in days:
        ic = compute_ic(d['pred_mean'], d['mid_prices'])
        if ic is not None:
            baseline_ics.append(ic)
    baseline_mean = np.mean(baseline_ics) if baseline_ics else 0.0
    baseline_t    = (baseline_mean / (np.std(baseline_ics) / np.sqrt(len(baseline_ics)))
                     if len(baseline_ics) > 1 else 0.0)

    logger.info(f"\nBaseline (no gating): IC={baseline_mean:+.4f}, "
                f"t={baseline_t:.2f}, {np.mean(np.array(baseline_ics)>0)*100:.0f}% positive "
                f"({len(baseline_ics)} days)")

    results = []

    for thr in thresholds:
        gated_ics = []
        kept_fracs = []

        for d in days:
            pm  = d['pred_mean'].copy()
            ps  = d['pred_std']
            mid = d['mid_prices']

            # Gate: zero out high-uncertainty bars
            high_unc = ps > thr
            pm[high_unc] = 0.0

            # Track fraction kept
            valid_signal = (d['pred_mean'] != 0)
            if valid_signal.sum() > 0:
                kept = (~high_unc & valid_signal).sum() / valid_signal.sum()
                kept_fracs.append(float(kept))

            ic = compute_ic(pm, mid)
            if ic is not None:
                gated_ics.append(ic)

        if not gated_ics:
            continue

        ic_arr    = np.array(gated_ics)
        mean_ic   = float(np.mean(ic_arr))
        std_ic    = float(np.std(ic_arr))
        t_stat    = float(mean_ic / (std_ic / np.sqrt(len(ic_arr)))) if std_ic > 0 else 0.0
        pct_pos   = float(np.mean(ic_arr > 0) * 100)
        mean_kept = float(np.mean(kept_fracs)) if kept_fracs else 1.0
        ic_lift   = mean_ic - baseline_mean

        results.append({
            'threshold': thr,
            'ic_mean':   round(mean_ic, 6),
            'ic_tstat':  round(t_stat, 3),
            'ic_pct_pos': round(pct_pos, 1),
            'ic_lift':   round(ic_lift, 6),
            'mean_kept': round(mean_kept, 4),
            'n_days':    len(gated_ics),
        })

        logger.info(
            f"  thr={thr:.3f} → IC={mean_ic:+.4f} (t={t_stat:.2f}, {pct_pos:.0f}%+) "
            f"| lift={ic_lift:+.4f} | kept={mean_kept:.1%}"
        )

    results.sort(key=lambda x: x['ic_mean'], reverse=True)

    if results:
        best = results[0]
        logger.info(f"\nBest threshold: {best['threshold']:.3f} → "
                    f"IC={best['ic_mean']:+.4f} (lift={best['ic_lift']:+.4f} "
                    f"vs baseline {baseline_mean:+.4f})")

    return results, baseline_mean, baseline_t


# ---------------------------------------------------------------------------
# Generate final gated prediction files for run_wf_fill_sim.py
# ---------------------------------------------------------------------------

def generate_fillsim_predictions(
    days: list[dict],
    output_dir: Path,
    threshold: float,
    logger: logging.Logger,
) -> list[Path]:
    """
    Apply the chosen threshold, then z-score the gated signal,
    and save per-day NPZ files with 'predictions' key — exactly what
    run_wf_fill_sim.py expects.

    Note: run_wf_fill_sim.py applies its own z-score internally, but we also
    apply it here so the threshold is comparable to the vol-gated baseline.
    Files named: {date}_mc_gated_thr{thr}.npz
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for d in days:
        pm  = d['pred_mean'].copy()
        ps  = d['pred_std']
        mid = d['mid_prices']
        n   = len(mid)

        # Gate
        pm[ps > threshold] = 0.0

        # Align with CNN offset
        aligned = np.zeros(n, dtype=np.float64)
        end_idx = min(CNN_OFFSET + len(pm), n)
        aligned[CNN_OFFSET:end_idx] = pm[:end_idx - CNN_OFFSET]

        thr_str = f'{threshold:.3f}'.replace('.', 'p')
        out_f = output_dir / f'{d["date"]}_mc_gated_thr{thr_str}.npz'
        np.savez_compressed(str(out_f), predictions=aligned)
        saved.append(out_f)

    logger.info(f"Saved {len(saved)} gated prediction files to {output_dir}")
    return saved


# ---------------------------------------------------------------------------
# Run fill_sim for gated vs baseline comparison
# ---------------------------------------------------------------------------

def run_fillsim_comparison(
    days: list[dict],
    gated_files_dir: Path,
    mbo_dir: Path,
    output_dir: Path,
    threshold: float,
    logger: logging.Logger,
) -> dict:
    """
    Run Rust fill_sim on gated predictions for the best IS config.
    Returns dict with results.
    """
    binary = ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
    if not binary.exists():
        logger.error(f"fill_sim_cli not found: {binary}")
        return {}

    TICK_VALUE = 12.50
    # Best IS config: vol70/conv2.5/1t/3r/30min
    config = {
        'threshold':         2.5,
        'hold_ms':           1800000,
        'latency_ms':        0,
        'chase_max_ticks':   1,
        'chase_max_reprices':3,
        'chase_interval_ms': 100,
        'label':             f'mc_gated_thr{threshold:.3f}',
    }

    sim_out_dir = output_dir / 'sim_results'
    sim_out_dir.mkdir(parents=True, exist_ok=True)

    thr_str = f'{threshold:.3f}'.replace('.', 'p')
    results_by_day = {}

    for d in days:
        date = d['date']
        nodash = date.replace('-', '')
        mbo_zst = mbo_dir / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
        mbo_dbn = mbo_dir / f'glbx-mdp3-{nodash}.mbo.dbn'
        mbo_file = mbo_zst if mbo_zst.exists() else (mbo_dbn if mbo_dbn.exists() else None)
        if mbo_file is None:
            continue

        pred_file = gated_files_dir / f'{date}_mc_gated_thr{thr_str}.npz'
        if not pred_file.exists():
            continue

        out_file = sim_out_dir / f'{config["label"]}_{date}.json'
        cmd = [
            str(binary),
            '--mbo-file',          str(mbo_file),
            '--predictions',       str(pred_file),
            '--output',            str(out_file),
            '--hold-ms',           str(config['hold_ms']),
            '--signal-threshold',  str(config['threshold']),
            '--latency-ms',        str(config['latency_ms']),
            '--chase-entry',
            '--chase-max-ticks',   str(config['chase_max_ticks']),
            '--chase-max-reprices',str(config['chase_max_reprices']),
            '--chase-interval-ms', str(config['chase_interval_ms']),
            '--quiet',
        ]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and out_file.exists():
                with open(out_file) as f:
                    results_by_day[date] = json.load(f)
        except Exception as e:
            logger.warning(f"Sim error {date}: {e}")

    if not results_by_day:
        logger.warning("No fill_sim results generated.")
        return {}

    # Aggregate
    total_pnl = 0.0
    total_trades = 0
    total_signals = 0
    total_filled = 0
    total_wins = 0
    daily_pnls = []
    all_trade_pnls = []

    for date_str, res in sorted(results_by_day.items()):
        day_pnl = res.get('total_pnl_dollars', 0)
        total_pnl += day_pnl
        total_trades += res.get('total_trades', 0)
        total_signals += res.get('total_signals', 0)
        total_filled += res.get('total_filled', 0)
        daily_pnls.append(day_pnl)
        if 'trades' in res:
            for trade in res['trades']:
                pnl = trade.get('pnl_dollars', 0)
                all_trade_pnls.append(pnl)
                if pnl > 0:
                    total_wins += 1

    n_days = len(results_by_day)
    if total_trades == 0:
        logger.warning("No trades generated.")
        return {}

    win_rate  = total_wins / total_trades
    fill_rate = total_filled / total_signals if total_signals > 0 else 0
    avg_daily = np.mean(daily_pnls)
    daily_std = np.std(daily_pnls) if len(daily_pnls) > 1 else 1e-8
    sharpe    = (avg_daily / daily_std) * np.sqrt(252) if daily_std > 0 else 0
    avg_trade = np.mean(all_trade_pnls) if all_trade_pnls else 0

    result = {
        'config': config['label'],
        'threshold': threshold,
        'n_days': n_days,
        'total_pnl': round(total_pnl, 2),
        'n_trades': total_trades,
        'fill_rate': round(fill_rate, 4),
        'win_rate': round(win_rate, 4),
        'sharpe_daily': round(sharpe, 3),
        'avg_daily_pnl': round(avg_daily, 2),
        'avg_trade_pnl': round(avg_trade, 2),
        'annualized_pnl': round(avg_daily * 252, 0),
    }

    logger.info("\n" + "=" * 65)
    logger.info(f"FILL_SIM RESULT — MC Gated (thr={threshold:.3f})")
    logger.info(f"  Days:         {n_days}")
    logger.info(f"  Total P&L:    ${total_pnl:,.2f}")
    logger.info(f"  Trades:       {total_trades}")
    logger.info(f"  Fill rate:    {fill_rate:.1%}")
    logger.info(f"  Win rate:     {win_rate:.1%}")
    logger.info(f"  Sharpe:       {sharpe:.2f}")
    logger.info(f"  Avg daily:    ${avg_daily:.2f}")
    logger.info(f"  Annualized:   ${avg_daily * 252:,.0f}")
    logger.info("=" * 65)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='MC Dropout Analysis — threshold sweep + fill_sim integration'
    )
    p.add_argument('--mc-dir', type=Path, required=True, metavar='DIR',
                   help='Directory with *_mc_preds.npz files (output of mc_dropout_inference.py)')
    p.add_argument('--output-dir', type=Path, required=True, metavar='DIR',
                   help='Output directory for analysis results')
    p.add_argument(
        '--thresholds', type=str,
        default='0.02,0.03,0.05,0.07,0.10,0.12,0.15,0.20,0.25,0.30',
        metavar='FLOATS',
        help='Comma-separated uncertainty thresholds to sweep (default: fine grid 0.02-0.30)'
    )
    p.add_argument('--run-fillsim', action='store_true',
                   help='Run Rust fill_sim on best-threshold gated predictions')
    p.add_argument('--mbo-dir', type=Path,
                   default=ROOT / 'data' / 'raw' / 'mbo',
                   metavar='DIR',
                   help='MBO data directory (--run-fillsim only)')
    p.add_argument('--best-threshold', type=float, default=None,
                   help='Override best threshold (skip sweep, use this for fill_sim)')
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)

    logger.info("=" * 65)
    logger.info("MC Dropout Analysis")
    logger.info(f"  MC dir:     {args.mc_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info("=" * 65)

    # Load results
    days = load_mc_results(args.mc_dir, logger)
    if not days:
        sys.exit(1)

    # Threshold sweep
    thresholds = [float(x) for x in args.thresholds.split(',')]
    sweep_results, baseline_ic, baseline_t = sweep_uncertainty_thresholds(
        days, thresholds, logger
    )

    # Determine best threshold
    if args.best_threshold is not None:
        best_thr = args.best_threshold
        logger.info(f"\nUsing user-specified threshold: {best_thr}")
    elif sweep_results:
        best_thr = sweep_results[0]['threshold']
        logger.info(f"\nAuto-selected best threshold: {best_thr} "
                    f"(IC={sweep_results[0]['ic_mean']:+.4f})")
    else:
        logger.warning("No sweep results. Defaulting threshold to 0.15.")
        best_thr = 0.15

    # Generate gated prediction files
    gated_dir = args.output_dir / 'gated_predictions'
    gated_files = generate_fillsim_predictions(days, gated_dir, best_thr, logger)

    # Optional fill_sim run
    sim_result = {}
    if args.run_fillsim:
        sim_result = run_fillsim_comparison(
            days, gated_dir, args.mbo_dir, args.output_dir, best_thr, logger
        )

    # Save full analysis JSON
    out_json = args.output_dir / f'mc_analysis_{_ts}.json'
    with open(str(out_json), 'w') as fh:
        json.dump({
            'timestamp': _ts,
            'n_days': len(days),
            'baseline_ic': round(baseline_ic, 6),
            'baseline_t': round(baseline_t, 3),
            'best_threshold': best_thr,
            'sweep': sweep_results,
            'fillsim': sim_result,
        }, fh, indent=2, default=str)

    logger.info(f"\nAnalysis saved: {out_json}")
    logger.info(f"Gated predictions: {gated_dir}/")
    logger.info("\nNext step: Run run_wf_fill_sim.py pointing to the gated_predictions/ dir")
    logger.info("  (or use --run-fillsim flag to do it automatically)")


if __name__ == '__main__':
    main()
