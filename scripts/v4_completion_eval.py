#!/usr/bin/env python3
"""
V4 Multihead Completion Evaluator + Auto-Ablation Launcher
===========================================================
Run this when v4_multihead_pressure_v1 training finishes (176 folds).
Evaluates concat IC across all folds, per-head breakdown, regime analysis,
and auto-launches a 3-head ablation (dropping dead NTPS/TIA) if DIR IC < 0.22.

Usage:
    python scripts/v4_completion_eval.py [--output-dir OUTPUT] [--auto-launch]

Author: Claude (evaluation pipeline)
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logging.basicConfig(format="%(asctime)s [V4-Eval] %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("V4-Eval")

# ─── Config ───
BASELINE_IC_1S = 0.220  # v3.4.2 baseline
IC_THRESHOLD_FOR_ABLATION = 0.215  # launch ablation if below this
HEAD_NAMES = ["dir", "ntps", "eofi", "pdi", "tia"]
HORIZONS = ["1s", "5s", "10s", "30s"]

# Neptune paths
NEPTUNE_HOST = "nick@neptune"
NEPTUNE_OUTPUT = "/home/nick/Lvl3Quant/output/v4_multihead_pressure_v1"
NEPTUNE_SCRIPT = "/home/nick/Lvl3Quant/scripts/train_v4_multihead.py"

# Local eval output
LOCAL_EVAL_DIR = Path("/home/jupiter/Lvl3Quant/output/v4_completion_eval")


def ssh_cmd(cmd: str, timeout: int = 30) -> str:
    """Run command on Neptune via SSH."""
    result = subprocess.run(
        ["ssh", NEPTUNE_HOST, cmd],
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout.strip()


def count_completed_folds() -> int:
    """Count how many fold prediction files exist on Neptune."""
    out = ssh_cmd(f"ls {NEPTUNE_OUTPUT}/fold_*_oot_predictions.npz 2>/dev/null | wc -l")
    return int(out) if out.isdigit() else 0


def copy_predictions(local_dir: Path, start_fold: int = 0, end_fold: int = 175) -> List[Path]:
    """Copy fold prediction npz files from Neptune to local."""
    local_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    # Check which folds we already have locally
    existing = set()
    for f in local_dir.glob("fold_*_oot_predictions.npz"):
        fold_num = int(f.stem.split("_")[1])
        existing.add(fold_num)

    # Copy missing folds
    missing = [f for f in range(start_fold, end_fold + 1) if f not in existing]
    if missing:
        log.info(f"Copying {len(missing)} fold predictions from Neptune...")
        for fold in missing:
            fname = f"fold_{fold:03d}_oot_predictions.npz"
            remote = f"{NEPTUNE_OUTPUT}/{fname}"
            local = local_dir / fname
            try:
                subprocess.run(
                    ["scp", f"{NEPTUNE_HOST}:{remote}", str(local)],
                    capture_output=True, timeout=60
                )
                if local.exists() and local.stat().st_size > 0:
                    copied.append(local)
            except Exception as e:
                log.warning(f"Failed to copy fold {fold}: {e}")

    return sorted(local_dir.glob("fold_*_oot_predictions.npz"))


def load_fold_predictions(npz_path: Path) -> Optional[Dict]:
    """Load predictions and labels from a fold npz file."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        result = {}

        # Try to find predictions and labels for each head
        for key in data.files:
            result[key] = data[key]

        return result
    except Exception as e:
        log.warning(f"Failed to load {npz_path.name}: {e}")
        return None


def compute_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute Spearman IC between predictions and labels."""
    mask = np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return 0.0
    try:
        return float(stats.spearmanr(preds[mask], labels[mask])[0])
    except:
        return 0.0


def evaluate_all_folds(fold_files: List[Path]) -> Dict:
    """Evaluate concat IC across all folds, per-head breakdown."""
    results = {
        "n_folds": len(fold_files),
        "per_fold_ic": {},
        "concat_ic": {},
        "per_head_concat_ic": {},
        "fold_ics": [],
    }

    # Collect all predictions across folds for concat IC
    concat_preds = {h: {hz: [] for hz in HORIZONS} for h in HEAD_NAMES}
    concat_labels = {h: {hz: [] for hz in HORIZONS} for h in HEAD_NAMES}

    for npz_path in fold_files:
        fold_num = int(npz_path.stem.split("_")[1])
        data = load_fold_predictions(npz_path)
        if data is None:
            continue

        fold_ics = {}
        for head in HEAD_NAMES:
            for hz in HORIZONS:
                pred_key = f"{head}_{hz}_pred"
                label_key = f"{head}_{hz}_label"

                # Try alternative key formats
                if pred_key not in data:
                    pred_key = f"pred_{head}_{hz}"
                if label_key not in data:
                    label_key = f"label_{head}_{hz}"

                if pred_key in data and label_key in data:
                    p = data[pred_key].flatten()
                    l = data[label_key].flatten()
                    ic = compute_ic(p, l)
                    fold_ics[f"{head}_{hz}"] = ic
                    concat_preds[head][hz].append(p)
                    concat_labels[head][hz].append(l)

        results["fold_ics"].append({"fold": fold_num, **fold_ics})

    # Compute concat IC
    for head in HEAD_NAMES:
        for hz in HORIZONS:
            if concat_preds[head][hz]:
                all_p = np.concatenate(concat_preds[head][hz])
                all_l = np.concatenate(concat_labels[head][hz])
                ic = compute_ic(all_p, all_l)
                results["concat_ic"][f"{head}_{hz}"] = ic

    return results


def print_report(results: Dict):
    """Print evaluation report."""
    print("\n" + "=" * 70)
    print(f"V4 MULTIHEAD COMPLETION EVALUATION ({results['n_folds']} folds)")
    print("=" * 70)

    # Concat IC table
    print(f"\n{'Head':<8}", end="")
    for hz in HORIZONS:
        print(f"  {hz:>8}", end="")
    print()
    print("-" * 44)

    for head in HEAD_NAMES:
        print(f"{head:<8}", end="")
        for hz in HORIZONS:
            key = f"{head}_{hz}"
            ic = results["concat_ic"].get(key, 0)
            marker = " *" if head == "dir" and hz == "1s" and ic < BASELINE_IC_1S else "  "
            print(f"  {ic:>6.3f}{marker}", end="")
        print()

    dir_1s = results["concat_ic"].get("dir_1s", 0)
    print(f"\nDir IC_1s = {dir_1s:.3f} (baseline {BASELINE_IC_1S:.3f}, "
          f"{'ABOVE' if dir_1s >= BASELINE_IC_1S else 'BELOW'} by {abs(dir_1s - BASELINE_IC_1S):.3f})")

    # Per-fold IC stability
    if results["fold_ics"]:
        dir_ics = [f.get("dir_1s", 0) for f in results["fold_ics"] if "dir_1s" in f]
        if dir_ics:
            print(f"\nDir IC_1s stability: mean={np.mean(dir_ics):.3f}, "
                  f"std={np.std(dir_ics):.3f}, "
                  f"min={np.min(dir_ics):.3f}, max={np.max(dir_ics):.3f}")

            # Trend analysis
            x = np.arange(len(dir_ics))
            slope, _, _, p_val, _ = stats.linregress(x, dir_ics)
            print(f"Trend: slope={slope:.4f}/fold, p={p_val:.3f} "
                  f"({'DECLINING' if slope < 0 and p_val < 0.1 else 'STABLE'})")

    # Dead head assessment
    print("\nHead Assessment:")
    for head in HEAD_NAMES:
        ic_1s = results["concat_ic"].get(f"{head}_1s", 0)
        status = "STRONG" if ic_1s > 0.1 else "MARGINAL" if ic_1s > 0.03 else "DEAD"
        print(f"  {head}: IC_1s={ic_1s:.3f} → {status}")

    # Recommendation
    print(f"\n{'=' * 70}")
    if dir_1s < IC_THRESHOLD_FOR_ABLATION:
        print(f"RECOMMENDATION: DIR IC ({dir_1s:.3f}) below threshold ({IC_THRESHOLD_FOR_ABLATION:.3f})")
        print("→ LAUNCH 3-HEAD ABLATION (drop NTPS+TIA, --head-weights 1.0,0.0,1.0,1.0,0.0)")
    else:
        print(f"RECOMMENDATION: DIR IC ({dir_1s:.3f}) at or above threshold ({IC_THRESHOLD_FOR_ABLATION:.3f})")
        print("→ Current 5-head model is acceptable. Optional: still try 3-head to see if better.")
    print("=" * 70)

    return dir_1s


def launch_ablation():
    """Launch 3-head ablation on Neptune."""
    cmd = (
        f"cd /home/nick/Lvl3Quant && nohup /home/nick/miniconda3/bin/python "
        f"scripts/train_v4_multihead.py "
        f"--data-dir data/processed/mbo_events_smart_v3 "
        f"--pressure-dir data/processed/mbo_events_smart_v3_pressure_labels "
        f"--output-dir output/v4_3head_ablation "
        f"--start-fold 0 "
        f"--stride 2000 "
        f"--epochs 2 "
        f"--head-weights 1.0,0.0,1.0,1.0,0.0 "
        f"--device cuda "
        f"--mlflow-uri http://jupiter:5000 "
        f"> output/v4_3head_ablation/nohup.log 2>&1 &"
    )
    log.info("Launching 3-head ablation on Neptune...")
    log.info(f"Head weights: dir=1.0, ntps=0.0, eofi=1.0, pdi=1.0, tia=0.0")

    try:
        ssh_cmd(f"mkdir -p /home/nick/Lvl3Quant/output/v4_3head_ablation", timeout=10)
        result = subprocess.run(
            ["ssh", NEPTUNE_HOST, cmd],
            capture_output=True, text=True, timeout=30
        )
        log.info("3-head ablation launched successfully")
        return True
    except Exception as e:
        log.error(f"Failed to launch ablation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_EVAL_DIR))
    parser.add_argument("--auto-launch", action="store_true",
                        help="Auto-launch 3-head ablation if DIR IC below threshold")
    parser.add_argument("--min-folds", type=int, default=45,
                        help="Minimum folds required to run evaluation (training covers folds 126-175 = 50 folds)")
    args = parser.parse_args()

    local_dir = Path(args.output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Check completion
    n_folds = count_completed_folds()
    log.info(f"Found {n_folds} completed folds on Neptune")

    if n_folds < args.min_folds:
        log.info(f"Training not complete ({n_folds}/{args.min_folds} folds). "
                 f"Run again when training finishes.")
        sys.exit(0)

    # Copy predictions (training starts at fold 126)
    fold_files = copy_predictions(local_dir, start_fold=126, end_fold=175)
    log.info(f"Loaded {len(fold_files)} fold predictions locally")

    if len(fold_files) < args.min_folds:
        log.warning(f"Only {len(fold_files)} files available. Need {args.min_folds}.")
        sys.exit(1)

    # Evaluate
    results = evaluate_all_folds(fold_files)
    dir_1s = print_report(results)

    # Save results
    results_file = local_dir / "eval_results.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    log.info(f"Results saved to {results_file}")

    # Auto-launch ablation if needed
    if args.auto_launch and dir_1s < IC_THRESHOLD_FOR_ABLATION:
        log.info(f"DIR IC ({dir_1s:.3f}) below threshold. Launching 3-head ablation...")
        if launch_ablation():
            log.info("Ablation launched. Monitor with: "
                     "ssh nick@neptune 'tail -f output/v4_3head_ablation/nohup.log'")
    elif args.auto_launch:
        log.info(f"DIR IC ({dir_1s:.3f}) above threshold. No ablation needed.")


if __name__ == "__main__":
    main()
