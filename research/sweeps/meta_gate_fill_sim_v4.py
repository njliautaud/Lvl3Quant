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
# CNN sim predictions — per-day vol-gated npz files (Dec 2025+)
CNN_PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_oot_sim_predictions"
# Meta-model predictions transferred from Neptune (Oct-Nov 2025)
# Each npz has: meta_preds, cnn_preds, meta_targets, date, fold
META_DIR = LVL3_ROOT / "data" / "processed" / "meta_model_lgbm"
OUT_BASE = LVL3_ROOT / "data" / "processed" / "meta_gate_fill_sim_results"
OUT_BASE.mkdir(parents=True, exist_ok=True)
GATED_PRED_DIR = LVL3_ROOT / "data" / "processed" / "meta_gate_predictions"
GATED_PRED_DIR.mkdir(parents=True, exist_ok=True)

GATE_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
WORKERS = 8

# Meta-model gate modes:
# MODE_DIRECT: use meta_preds from Oct-Nov folds directly (needs matching MBO — not available)
# MODE_TRAIN_APPLY: train LGBM on Oct-Nov features, apply to Dec CNN predictions
# MODE_SIGNAL_THRESHOLD: use abs(cnn_pred) as proxy gate (no model needed, pure signal strength)
GATE_MODE = "MODE_TRAIN_APPLY"  # Primary mode

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
            pnl = data.get('total_pnl_dollars',
                  data.get('total_pnl_ticks',
                  data.get('pnl_ticks', 0)))
            trades = data.get('total_trades', data.get('n_trades', 0))
            wr = data.get('win_rate', 0)
            wins = data.get('winning_trades', data.get('n_wins', int(wr * trades)))
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

    # Sortino: use downside deviation
    negative = daily_arr[daily_arr < 0]
    downside_std = float(np.std(negative)) if len(negative) > 0 else 1e-8
    sortino = mean_pnl / (downside_std + 1e-8)

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

def load_meta_npz_files():
    """
    Load meta-model npz files from META_DIR.
    Each file has: meta_preds (gate signal), cnn_preds (raw CNN signal), date.
    Returns: dict {date_nodash: {'meta_preds': array, 'cnn_preds': array}}
    """
    if not META_DIR.exists():
        log.error(f"Meta dir not found: {META_DIR}")
        return {}

    meta_files = sorted(META_DIR.glob("*.npz"))
    log.info(f"Found {len(meta_files)} meta npz files in {META_DIR}")

    date_data = {}
    for f in meta_files:
        try:
            d = np.load(f)
            # Get date from 'date' key or filename
            date_str = str(d['date']) if 'date' in d else None
            if date_str:
                date_nodash = date_str.replace('-', '')
            else:
                # Parse from filename: fold_016_2025-10-02_meta.npz
                parts = f.stem.split('_')
                date_part = next((p for p in parts if len(p) == 10 and p[4] == '-'), None)
                if date_part:
                    date_nodash = date_part.replace('-', '')
                else:
                    log.warning(f"Cannot determine date from {f.name}")
                    continue

            meta_preds = d.get('meta_preds', d.get('meta_predictions', d.get('predictions', None)))
            cnn_preds = d.get('cnn_preds', d.get('cnn_predictions', None))

            if meta_preds is None:
                log.warning(f"No meta_preds key in {f.name}: keys={list(d.keys())}")
                continue

            date_data[date_nodash] = {
                'meta_preds': meta_preds,
                'cnn_preds': cnn_preds,
                'source': f.name,
            }
        except Exception as e:
            log.warning(f"Error loading {f.name}: {e}")

    log.info(f"Loaded meta predictions for {len(date_data)} dates")
    if date_data:
        dates = sorted(date_data.keys())
        log.info(f"Date range: {dates[0]} to {dates[-1]}")
    return date_data


def create_gated_pred_from_meta(date_nodash, meta_data, gate_thresh, out_path):
    """
    Create gated prediction file using meta_preds as gate on cnn_preds.
    meta_preds: continuous signal [-1, 1], use abs value as gate confidence.
    cnn_preds: raw CNN z-score predictions.
    Gate: only trade where abs(meta_preds) > gate_thresh.
    """
    meta_preds = meta_data['meta_preds']  # confidence signal
    cnn_preds_raw = meta_data.get('cnn_preds')  # raw CNN signal

    if cnn_preds_raw is None:
        log.warning(f"No cnn_preds for {date_nodash}, cannot create gated predictions")
        return 0, 0

    # Both should be 233860 bars (meta covers Oct-Nov which is ~same RTH)
    # Align lengths if needed
    n = min(len(meta_preds), len(cnn_preds_raw))
    meta_preds = meta_preds[:n]
    cnn_preds = cnn_preds_raw[:n].copy()

    # Gate confidence = abs(meta_preds) normalized to [0,1]
    max_abs = np.abs(meta_preds).max()
    if max_abs > 0:
        confidence = np.abs(meta_preds) / max_abs
    else:
        confidence = np.zeros(n)

    # Zero out CNN signal where confidence < threshold
    gate_mask = confidence < gate_thresh
    gated = cnn_preds.copy()
    gated[gate_mask] = 0.0

    n_kept = int((gated != 0).sum())
    n_total = int((cnn_preds != 0).sum())

    # Pad to 234000 if needed (fill sim expects exactly 234000)
    if n < 234000:
        padded = np.zeros(234000, dtype=gated.dtype)
        padded[:n] = gated
        gated = padded

    np.savez_compressed(str(out_path), predictions=gated)
    log.debug(f"Gate {gate_thresh} {date_nodash}: kept {n_kept}/{n_total} signals ({100*n_kept/(n_total+1):.1f}%)")
    return n_kept, n_total


def train_meta_gate_model(meta_by_date):
    """
    Train a lightweight meta-gate model from Oct-Nov fold data.
    Features: abs(cnn_pred), cnn_pred (signed), bar time-of-day, running z-score
    Target: sign of meta_preds (1 = gate open, -1 = gate closed)

    Returns: sklearn-compatible model or None
    """
    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
    except ImportError:
        log.warning("sklearn not available for meta gate model training")
        return None

    X_list, y_list = [], []
    n_dates = len(meta_by_date)
    n_bars = 234000

    for date_nodash, data in sorted(meta_by_date.items()):
        cnn_preds = data.get('cnn_preds')
        meta_preds = data.get('meta_preds')
        if cnn_preds is None or meta_preds is None:
            continue

        n = min(len(cnn_preds), len(meta_preds))
        cnn = cnn_preds[:n]
        meta = meta_preds[:n]

        # Only use bars where CNN has a nonzero signal
        active = cnn != 0
        if active.sum() < 10:
            continue

        idx = np.where(active)[0]
        t_of_day = idx / n  # 0 = market open, 1 = close

        X = np.column_stack([
            cnn[idx],           # signed signal
            np.abs(cnn[idx]),   # signal magnitude
            t_of_day,           # time of day
            np.abs(cnn[idx]) ** 2,  # quadratic signal
        ])

        # Binary target: meta_preds confidence > median = keep signal
        meta_conf = np.abs(meta[idx])
        med_conf = np.median(meta_conf)
        y = (meta_conf > med_conf).astype(int)

        X_list.append(X)
        y_list.append(y)

    if not X_list:
        return None

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    log.info(f"Training meta gate model on {len(X_all)} samples from {n_dates} fold days")
    model = LogisticRegression(max_iter=500, C=1.0)
    model.fit(X_all, y_all)

    score = model.score(X_all, y_all)
    log.info(f"Meta gate model training accuracy: {score:.3f}")
    return model


def apply_gate_model(model, cnn_preds_arr, gate_thresh, out_path):
    """Apply trained gate model to CNN predictions array."""
    import numpy as np

    n = len(cnn_preds_arr)
    active = cnn_preds_arr != 0
    idx = np.where(active)[0]

    if len(idx) == 0:
        np.savez_compressed(str(out_path), predictions=cnn_preds_arr)
        return 0, 0

    t_of_day = idx / n
    cnn_sig = cnn_preds_arr[idx]

    X = np.column_stack([
        cnn_sig,
        np.abs(cnn_sig),
        t_of_day,
        np.abs(cnn_sig) ** 2,
    ])

    proba = model.predict_proba(X)[:, 1]

    gated = cnn_preds_arr.copy()
    gated[idx[proba < gate_thresh]] = 0.0

    n_kept = int((gated != 0).sum())
    n_total = int(active.sum())

    np.savez_compressed(str(out_path), predictions=gated)
    return n_kept, n_total


def main():
    log.info("=" * 60)
    log.info("META-MODEL GATE FILL SIM SWEEP")
    log.info("=" * 60)

    if not BINARY.exists():
        log.error(f"fill_sim binary not found: {BINARY}")
        sys.exit(1)

    # Load meta predictions (Oct-Nov 2025, folds 16-38)
    log.info(f"Loading meta-model npz files from {META_DIR}")
    meta_by_date = load_meta_npz_files()

    if not meta_by_date:
        log.error("No meta predictions found in META_DIR. "
                  "Transfer npz files from Neptune: "
                  "C:/Users/Footb/Documents/Github/Lvl3Quant/alpha_discovery/experiments/results/meta_model_lgbm/")
        sys.exit(1)

    # Train gate model from Oct-Nov data
    gate_model = train_meta_gate_model(meta_by_date)
    if gate_model is None:
        log.warning("Gate model training failed — falling back to signal-threshold gating")

    # Use Dec+ CNN oot_sim_predictions as the target for gating
    cnn_files = sorted(CNN_PRED_DIR.glob("*.npz"))
    cnn_files_vg = [f for f in cnn_files if 'vol70' in f.name]
    if not cnn_files_vg:
        cnn_files_vg = cnn_files
    log.info(f"Applying gate to {len(cnn_files_vg)} Dec+ CNN prediction files")

    results = {'baseline': None, 'gated': {}, 'gate_mode': 'train_apply_or_signal_threshold'}

    # --- BASELINE: raw CNN signal on Dec+ dates (no gate) ---
    log.info("\n--- BASELINE (no meta gate, Dec+ CNN predictions) ---")
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
        results['baseline'] = baseline_metrics
        log.info(f"BASELINE: Sortino={baseline_metrics['sortino']:.3f}, "
                 f"Sharpe={baseline_metrics['sharpe']:.3f}, "
                 f"Trades={baseline_metrics['total_trades']}, "
                 f"WR={baseline_metrics['win_rate']:.2%}, "
                 f"PnL={baseline_metrics['total_pnl_ticks']:.1f}")

    # --- GATED SWEEPS: apply gate model to Dec+ CNN predictions ---
    for gate_thresh in GATE_THRESHOLDS:
        log.info(f"\n--- Gate threshold: {gate_thresh} ---")
        gate_dir = OUT_BASE / f"gate_{int(gate_thresh*100)}"
        gate_dir.mkdir(exist_ok=True)
        gate_pred_subdir = GATED_PRED_DIR / f"gate_{int(gate_thresh*100)}"
        gate_pred_subdir.mkdir(exist_ok=True)

        jobs = []
        for pf in cnn_files_vg:
            stem = pf.stem
            date_nodash = stem[:10].replace('-', '')
            mbo = MBO_DIR / f"glbx-mdp3-{date_nodash}.mbo.dbn.zst"
            if not mbo.exists():
                continue

            out = gate_dir / f"{stem}_gate{int(gate_thresh*100)}.json"
            if out.exists():
                continue

            gated_pred_file = gate_pred_subdir / f"{stem}.npz"
            if not gated_pred_file.exists():
                try:
                    d = np.load(pf)
                    cnn_preds = d['predictions']
                except Exception as e:
                    log.debug(f"Error loading {pf.name}: {e}")
                    continue

                if gate_model is not None:
                    # Apply trained gate model
                    n_kept, n_total = apply_gate_model(gate_model, cnn_preds, gate_thresh, gated_pred_file)
                else:
                    # Fallback: signal-strength threshold gate
                    # Gate = only trade when abs(cnn_pred) > gate_thresh * max_abs
                    gated = cnn_preds.copy()
                    max_abs = np.abs(cnn_preds).max()
                    if max_abs > 0:
                        threshold = gate_thresh * max_abs
                        gated[np.abs(gated) < threshold] = 0.0
                    np.savez_compressed(str(gated_pred_file), predictions=gated)
                    n_kept = int((gated != 0).sum())
                    n_total = int((cnn_preds != 0).sum())

                log.debug(f"Gate {gate_thresh} {date_nodash}: kept {n_kept}/{n_total}")

            jobs.append((mbo, gated_pred_file, out))

        log.info(f"  Gate {gate_thresh}: {len(jobs)} jobs")
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
                     f"PnL={metrics['total_pnl_ticks']:.1f}")
        else:
            log.warning(f"  Gate {gate_thresh}: No results")

    # --- SUMMARY ---
    log.info("\n" + "=" * 60)
    log.info("FINAL SUMMARY — META GATE SWEEP")
    log.info("=" * 60)

    bm = results['baseline']
    if bm:
        log.info(f"BASELINE:  Sortino={bm['sortino']:.3f}  Trades={bm['total_trades']}  "
                 f"WR={bm['win_rate']:.2%}  PnL={bm['total_pnl_ticks']:.1f}")

    best_sortino = bm['sortino'] if bm else -999
    best_thresh = 'baseline'
    for thresh, m in results['gated'].items():
        if m:
            if m['sortino'] > best_sortino:
                best_sortino = m['sortino']
                best_thresh = thresh
            log.info(f"Gate {thresh}: Sortino={m['sortino']:.3f}  "
                     f"Trades={m['total_trades']}  "
                     f"WR={m['win_rate']:.2%}  "
                     f"PnL={m['total_pnl_ticks']:.1f}")

    log.info(f"\nBEST: thresh={best_thresh}, Sortino={best_sortino:.3f}")

    summary_file = OUT_BASE / "sweep_summary.json"
    summary_file.write_text(json.dumps(results, indent=2, default=str))
    log.info(f"Results saved to {summary_file}")

if __name__ == "__main__":
    main()
