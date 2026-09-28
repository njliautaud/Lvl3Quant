#!/usr/bin/env python3
"""
Meta-Model Gate Fill Sim — Jupiter
Loads meta-model LGBM predictions (folds 16-38, Sep-Nov 2025),
uses them as a gate on the CNN signal, sweeps gate thresholds,
and reports Sortino for each threshold.

Deployment: /home/jupiter/Lvl3Quant/
"""
import sys, json, time, subprocess, logging
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats

logging.basicConfig(
    format='%(asctime)s [meta_gate] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger('meta_gate')

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
# CNN sim predictions — per-day vol-gated npz files
CNN_PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_oot_sim_predictions"
# Meta-model predictions directory (check both locations)
META_DIR = LVL3_ROOT / "alpha_discovery" / "experiments" / "results" / "meta_model_lgbm"
OUT_BASE = LVL3_ROOT / "data" / "processed" / "meta_gate_fill_sim_results"
OUT_BASE.mkdir(parents=True, exist_ok=True)
GATED_PRED_DIR = LVL3_ROOT / "data" / "processed" / "meta_gate_predictions"
GATED_PRED_DIR.mkdir(parents=True, exist_ok=True)

GATE_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
WORKERS = 8

# Default fill sim params (conservative, matching existing sweeps)
TP_TICKS = 8
SL_TICKS = 15
HOLD_MS = 3600000
LATENCY_MS = 0

def load_meta_predictions():
    """Load all meta-model per-fold npz files, return {date_str: array}."""
    if not META_DIR.exists():
        log.warning(f"Meta dir not found: {META_DIR}")
        # Try alternate locations
        alt = LVL3_ROOT / "alpha_discovery" / "results" / "meta_model_lgbm"
        if alt.exists():
            meta_dir = alt
        else:
            log.error("No meta-model predictions found. Cannot proceed.")
            return {}
    else:
        meta_dir = META_DIR

    meta_preds = {}
    for f in sorted(meta_dir.glob("*.npz")):
        try:
            d = np.load(f)
            # Try common key names
            for key in ['meta_predictions', 'predictions', 'meta_pred', 'pred']:
                if key in d:
                    meta_preds[f.stem] = (str(f), key)
                    break
            else:
                log.warning(f"Unknown keys in {f.name}: {list(d.keys())}")
        except Exception as e:
            log.warning(f"Failed to load {f.name}: {e}")

    log.info(f"Loaded {len(meta_preds)} meta-model fold files")
    return meta_preds

def build_meta_date_map():
    """
    Map date -> meta prediction array.
    Meta-model covers Sep-Nov 2025 (folds 16-38).
    Each fold file covers multiple days — parse date from filename or
    use the concatenated meta predictions if available.
    """
    if not META_DIR.exists():
        log.error(f"Meta dir missing: {META_DIR}")
        return {}

    date_map = {}
    files = sorted(META_DIR.glob("*.npz"))
    log.info(f"Found {len(files)} meta-model files in {META_DIR}")

    for f in files:
        try:
            d = np.load(f)
            keys = list(d.keys())
            log.info(f"  {f.name}: keys={keys}")

            # Get predictions array
            pred_arr = None
            for key in ['meta_predictions', 'predictions', 'meta_pred', 'pred', 'y_pred']:
                if key in d:
                    pred_arr = d[key]
                    break

            # Get dates array if present
            dates_arr = None
            for key in ['dates', 'timestamps', 'date']:
                if key in d:
                    dates_arr = d[key]
                    break

            if pred_arr is None:
                log.warning(f"  No prediction array found in {f.name}")
                continue

            # If dates available, map per day
            if dates_arr is not None:
                unique_dates = np.unique(dates_arr)
                for dt in unique_dates:
                    mask = dates_arr == dt
                    date_str = str(dt)[:10].replace('-', '')
                    if date_str not in date_map:
                        date_map[date_str] = []
                    date_map[date_str].append((pred_arr[mask], f.name))
            else:
                # No date info — try to extract from filename
                # e.g. fold_16_meta.npz or 2025-09-01_meta.npz
                fname = f.stem
                if len(fname) >= 10 and fname[:4].isdigit():
                    date_str = fname[:10].replace('-', '')
                    if date_str not in date_map:
                        date_map[date_str] = []
                    date_map[date_str].append((pred_arr, f.name))
                else:
                    # Store as fold-level, will be handled separately
                    date_map[f'fold_{fname}'] = [(pred_arr, f.name)]

        except Exception as e:
            log.error(f"Error loading {f.name}: {e}")

    return date_map

def get_meta_confidence(meta_arr):
    """
    Extract confidence score from meta predictions.
    If binary (0/1), use the value directly.
    If continuous, use abs value as confidence.
    """
    if meta_arr.dtype in [np.float32, np.float64]:
        # Could be probabilities [0,1] or z-scores
        if meta_arr.max() <= 1.0 and meta_arr.min() >= 0.0:
            return meta_arr  # Already probability
        else:
            # Normalize to [0,1] confidence
            return (meta_arr - meta_arr.min()) / (meta_arr.max() - meta_arr.min() + 1e-8)
    return meta_arr.astype(float)

def create_gated_predictions(cnn_pred_file, meta_arr, gate_thresh, out_path):
    """
    Create gated prediction file: zero out CNN signal where meta confidence < threshold.
    """
    d = np.load(cnn_pred_file)
    cnn_preds = d['predictions'].copy()

    # Meta array length must match CNN
    if len(meta_arr) == len(cnn_preds):
        confidence = get_meta_confidence(meta_arr)
        # Gate: zero out signals where meta confidence below threshold
        mask = confidence < gate_thresh
        gated = cnn_preds.copy()
        gated[mask] = 0.0
        n_kept = (gated != 0).sum()
        n_total = (cnn_preds != 0).sum()
        log.debug(f"Gate {gate_thresh}: kept {n_kept}/{n_total} signals ({100*n_kept/(n_total+1):.1f}%)")
    else:
        log.warning(f"Meta array length {len(meta_arr)} != CNN length {len(cnn_preds)}, skipping gate")
        gated = cnn_preds

    np.savez_compressed(str(out_path), predictions=gated)
    return n_kept, n_total

def run_fill_sim(mbo_file, pred_file, out_file, tp=TP_TICKS, sl=SL_TICKS):
    """Run the Rust fill simulator."""
    cmd = [
        str(BINARY),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--hold-ms", str(HOLD_MS),
        "--signal-threshold", "0.1",
        "--take-profit-ticks", str(tp),
        "--stop-loss-ticks", str(sl),
        "--latency-ms", str(LATENCY_MS),
        "--chase-entry",
        "--chase-max-ticks", "2",
        "--chase-max-reprices", "5",
        "--quiet",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and Path(out_file).exists():
            return True
        else:
            log.debug(f"fill_sim failed: {r.stderr[:200]}")
            return False
    except Exception as e:
        log.debug(f"fill_sim exception: {e}")
        return False

def aggregate_results(result_dir):
    """Aggregate fill sim JSON results, compute Sortino and other metrics."""
    daily_pnl = []
    total_trades = 0
    total_wins = 0

    for f in sorted(result_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            pnl = data.get('total_pnl_ticks', data.get('pnl_ticks', 0))
            trades = data.get('total_trades', data.get('n_trades', 0))
            wins = data.get('winning_trades', data.get('n_wins', 0))
            daily_pnl.append(pnl)
            total_trades += trades
            total_wins += wins
        except Exception as e:
            log.debug(f"Error reading {f.name}: {e}")

    if not daily_pnl or total_trades == 0:
        return None

    daily_arr = np.array(daily_pnl)
    mean_pnl = float(np.mean(daily_arr))
    std_pnl = float(np.std(daily_arr))

    # Sortino: use proper semi-deviation (sqrt of mean of squared negative returns)
    downside = float(np.sqrt(np.mean(np.minimum(daily_arr, 0) ** 2)))
    sortino = mean_pnl / max(downside, 1e-8)

    # Sharpe
    sharpe = mean_pnl / (std_pnl + 1e-8)

    win_rate = total_wins / total_trades if total_trades > 0 else 0
    total_pnl = float(np.sum(daily_arr))

    return {
        'n_days': len(daily_pnl),
        'total_trades': total_trades,
        'total_pnl_ticks': total_pnl,
        'mean_daily_pnl': mean_pnl,
        'sortino': sortino,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'pct_positive_days': float((daily_arr > 0).mean()),
    }

def run_baseline():
    """Run fill sim with no gating (baseline = pure CNN signal)."""
    log.info("Running BASELINE (no meta gate)...")
    baseline_dir = OUT_BASE / "baseline"
    baseline_dir.mkdir(exist_ok=True)

    cnn_files = sorted(CNN_PRED_DIR.glob("*.npz"))
    # Use vol70 morning_afternoon as default vol gate
    cnn_files_filtered = [f for f in cnn_files if 'vol70' in f.name]
    if not cnn_files_filtered:
        cnn_files_filtered = cnn_files[:30]  # fallback

    jobs = []
    for pf in cnn_files_filtered:
        stem = pf.stem
        date = stem[:10].replace('-', '')
        mbo = MBO_DIR / f"glbx-mdp3-{date}.mbo.dbn.zst"
        if not mbo.exists():
            continue
        out = baseline_dir / f"{stem}.json"
        if out.exists():
            continue
        jobs.append((mbo, pf, out))

    log.info(f"Baseline: {len(jobs)} jobs to run")
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_fill_sim, m, p, o): o for m, p, o in jobs}
        for fut in as_completed(futs):
            if fut.result():
                done += 1

    log.info(f"Baseline: {done}/{len(jobs)} completed")
    return aggregate_results(baseline_dir)

def main():
    log.info("=" * 60)
    log.info("META-MODEL GATE FILL SIM SWEEP")
    log.info("=" * 60)

    if not BINARY.exists():
        log.error(f"fill_sim binary not found: {BINARY}")
        sys.exit(1)

    if not CNN_PRED_DIR.exists():
        log.error(f"CNN pred dir not found: {CNN_PRED_DIR}")
        sys.exit(1)

    # Load meta-model predictions
    log.info(f"Loading meta-model predictions from {META_DIR}")
    date_map = build_meta_date_map()

    if not date_map:
        log.error("No meta-model predictions found. Cannot run gated sweep.")
        log.info("Available CNN predictions exist — running baseline only.")
        baseline = run_baseline()
        if baseline:
            log.info(f"BASELINE: Sortino={baseline['sortino']:.3f}, Sharpe={baseline['sharpe']:.3f}, "
                     f"Trades={baseline['total_trades']}, WR={baseline['win_rate']:.2%}")
        sys.exit(1)

    # Determine which CNN dates have corresponding meta predictions
    # Meta covers Sep-Nov 2025 (folds 16-38)
    cnn_files = sorted(CNN_PRED_DIR.glob("*.npz"))
    log.info(f"Found {len(cnn_files)} CNN prediction files")

    # Use vol70 for primary sweep (best vol gate from prior research)
    vol_gate = "vol70"
    cnn_files_vg = [f for f in cnn_files if vol_gate in f.name]
    if not cnn_files_vg:
        cnn_files_vg = cnn_files

    log.info(f"Using {len(cnn_files_vg)} CNN files with vol gate={vol_gate}")

    # Run baseline first
    log.info("Running baseline (no meta gate)...")
    baseline_dir = OUT_BASE / "baseline"
    baseline_dir.mkdir(exist_ok=True)

    baseline_jobs = []
    for pf in cnn_files_vg:
        stem = pf.stem
        date_nodash = stem[:10].replace('-', '')
        mbo = MBO_DIR / f"glbx-mdp3-{date_nodash}.mbo.dbn.zst"
        if not mbo.exists():
            continue
        out = baseline_dir / f"{stem}.json"
        if out.exists():
            continue
        baseline_jobs.append((mbo, pf, out))

    log.info(f"Baseline jobs: {len(baseline_jobs)}")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_fill_sim, m, p, o): o for m, p, o in baseline_jobs}
        done = sum(1 for f in as_completed(futs) if f.result())
    log.info(f"Baseline: {done}/{len(baseline_jobs)} done")

    baseline_metrics = aggregate_results(baseline_dir)
    if baseline_metrics:
        log.info(f"BASELINE: Sortino={baseline_metrics['sortino']:.3f}, "
                 f"Trades={baseline_metrics['total_trades']}, "
                 f"WR={baseline_metrics['win_rate']:.2%}, "
                 f"PnL={baseline_metrics['total_pnl_ticks']:.1f}t")

    # Now run gated sweeps
    # Build a combined meta array (per date if possible, or use fold-level)
    # Strategy: if we have per-date meta arrays, use them
    # Otherwise, if meta arrays are fold-level, we need to figure out date assignment

    results = {'baseline': baseline_metrics, 'gated': {}}

    for gate_thresh in GATE_THRESHOLDS:
        log.info(f"\n--- Gate threshold: {gate_thresh} ---")
        gate_dir = OUT_BASE / f"gate_{int(gate_thresh*100)}"
        gate_dir.mkdir(exist_ok=True)
        gate_pred_subdir = GATED_PRED_DIR / f"gate_{int(gate_thresh*100)}"
        gate_pred_subdir.mkdir(exist_ok=True)

        jobs = []
        no_meta_count = 0

        for pf in cnn_files_vg:
            stem = pf.stem
            date_nodash = stem[:10].replace('-', '')
            mbo = MBO_DIR / f"glbx-mdp3-{date_nodash}.mbo.dbn.zst"
            if not mbo.exists():
                continue

            out = gate_dir / f"{stem}_gate{int(gate_thresh*100)}.json"
            if out.exists():
                continue

            # Look up meta predictions for this date
            meta_arr = None
            if date_nodash in date_map:
                # Take first matching meta array
                meta_arr = date_map[date_nodash][0][0]
            else:
                # Try to find fold-level data covering this date
                # Fall back: no gating for this day (skip or use raw CNN)
                no_meta_count += 1
                continue

            # Create gated prediction file
            gated_pred_file = gate_pred_subdir / f"{stem}_gate{int(gate_thresh*100)}.npz"
            if not gated_pred_file.exists():
                n_kept, n_total = create_gated_predictions(pf, meta_arr, gate_thresh, gated_pred_file)

            jobs.append((mbo, gated_pred_file, out))

        if no_meta_count > 0:
            log.info(f"  Skipped {no_meta_count} dates with no meta predictions")

        log.info(f"  Running {len(jobs)} gated fill sims...")
        if jobs:
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = {ex.submit(run_fill_sim, m, p, o): o for m, p, o in jobs}
                done = sum(1 for f in as_completed(futs) if f.result())
            log.info(f"  Gate {gate_thresh}: {done}/{len(jobs)} done")

        metrics = aggregate_results(gate_dir)
        if metrics:
            results['gated'][gate_thresh] = metrics
            log.info(f"  Gate {gate_thresh}: Sortino={metrics['sortino']:.3f}, "
                     f"Sharpe={metrics['sharpe']:.3f}, "
                     f"Trades={metrics['total_trades']}, "
                     f"WR={metrics['win_rate']:.2%}, "
                     f"PnL={metrics['total_pnl_ticks']:.1f}t, "
                     f"PctPosDays={metrics['pct_positive_days']:.1%}")
        else:
            log.info(f"  Gate {gate_thresh}: No results (meta predictions may not cover this date range)")

    # Final summary
    log.info("\n" + "=" * 60)
    log.info("FINAL SUMMARY — META GATE SWEEP")
    log.info("=" * 60)
    if baseline_metrics:
        log.info(f"BASELINE:  Sortino={baseline_metrics['sortino']:.3f}  "
                 f"Trades={baseline_metrics['total_trades']}  "
                 f"WR={baseline_metrics['win_rate']:.2%}  "
                 f"PnL={baseline_metrics['total_pnl_ticks']:.1f}t")

    best_sortino = baseline_metrics['sortino'] if baseline_metrics else -999
    best_thresh = 'baseline'
    for thresh, m in results['gated'].items():
        if m and m['sortino'] > best_sortino:
            best_sortino = m['sortino']
            best_thresh = thresh
        if m:
            log.info(f"Gate {thresh}: Sortino={m['sortino']:.3f}  "
                     f"Trades={m['total_trades']}  "
                     f"WR={m['win_rate']:.2%}  "
                     f"PnL={m['total_pnl_ticks']:.1f}t")

    log.info(f"\nBEST: thresh={best_thresh}, Sortino={best_sortino:.3f}")

    # Save results summary
    summary_file = OUT_BASE / "sweep_summary.json"
    summary_file.write_text(json.dumps(results, indent=2, default=str))
    log.info(f"Results saved to {summary_file}")

if __name__ == "__main__":
    main()
