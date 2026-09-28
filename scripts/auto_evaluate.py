#!/usr/bin/env python3
"""Auto-evaluate completed fold predictions across all nodes.

Usage:
    python3 auto_evaluate.py [--local-only] [--node neptune|razer|jupiter]
"""
import argparse, json, os, subprocess, sys, tempfile, glob
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import pearsonr

EVAL_LOG = "/home/jupiter/Lvl3Quant/data/results/evaluation_log.json"
HORIZONS = ["1s", "5s", "10s"]
TIERS = {"All": 1.0, "Top50%": 0.5, "Top25%": 0.25, "Top10%": 0.10, "Top5%": 0.05}

NODES = {
    "jupiter": {"local": True, "path": "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/"},
    "neptune": {"local": False, "ssh": "nick@neptune", "path": "/home/nick/Lvl3Quant/output/"},
    "razer":   {"local": False, "ssh": "claude@razer", "path": r"C:\Users\claude\Lvl3Quant\alpha_discovery\deep_models\results\\"},
}


def calc_ic(preds, labels):
    mask = np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return float("nan")
    r, _ = pearsonr(preds[mask], labels[mask])
    return round(float(r), 6)


def calc_da(preds, labels):
    mask = np.isfinite(preds) & np.isfinite(labels) & (preds != 0) & (labels != 0)
    if mask.sum() < 10:
        return float("nan")
    return round(float(np.mean(np.sign(preds[mask]) == np.sign(labels[mask]))) * 100, 2)


def calc_magcorr(preds, labels):
    mask = np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return float("nan")
    r, _ = pearsonr(np.abs(preds[mask]), np.abs(labels[mask]))
    return round(float(r), 6)


def extract_horizon_data(data, hz):
    """Extract preds/labels for a horizon, handling both NPZ formats."""
    # Format 1: preds_1s / labels_1s keys
    pk, lk = f"preds_{hz}", f"labels_{hz}"
    if pk in data and lk in data:
        return data[pk].ravel().astype(float), data[lk].ravel().astype(float)
    # Format 2: predictions/labels as (N, H) with horizons array
    if "predictions" in data and "labels" in data:
        horizons = list(data["horizons"]) if "horizons" in data else HORIZONS
        if hz in horizons:
            col = horizons.index(hz)
            p = np.array(data["predictions"], dtype=float)
            l = np.array(data["labels"], dtype=float)
            if p.ndim == 2 and col < p.shape[1]:
                return p[:, col], l[:, col]
    return None, None


def evaluate_npz(filepath):
    """Evaluate a single npz file across all horizons and tiers."""
    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        return {"error": str(e)}

    results = {}
    for hz in HORIZONS:
        preds, labels = extract_horizon_data(data, hz)
        if preds is None or len(preds) == 0 or len(preds) != len(labels):
            continue

        hz_results = {"n_samples": int(len(preds))}
        for tier_name, tier_frac in TIERS.items():
            n = max(10, int(len(preds) * tier_frac))
            if tier_frac < 1.0:
                idx = np.argsort(-np.abs(preds))[:n]
                p, l = preds[idx], labels[idx]
            else:
                p, l = preds, labels
            hz_results[tier_name] = {"IC": calc_ic(p, l), "DA": calc_da(p, l), "MagCorr": calc_magcorr(p, l)}
        results[hz] = hz_results
    return results


def find_local_npz(base_path):
    """Find all oot prediction npz files locally."""
    files = []
    for root, _, fnames in os.walk(base_path):
        for f in fnames:
            if f.endswith(".npz") and "oot_prediction" in f.lower():
                files.append(os.path.join(root, f))
    return sorted(files)


def find_remote_npz(node_cfg):
    """Find oot prediction npz files on remote node, copy locally for eval."""
    ssh = node_cfg["ssh"]
    rpath = node_cfg["path"]
    # List remote npz files
    if "claude@" in ssh:  # Windows node
        cmd = f'ssh -o ConnectTimeout=5 {ssh} "dir /s /b {rpath}*.npz 2>NUL"'
    else:
        cmd = f'ssh -o ConnectTimeout=5 {ssh} "find {rpath} -name \'*.npz\' -ipath \'*oot_prediction*\' 2>/dev/null"'
    try:
        out = subprocess.check_output(cmd, shell=True, timeout=15, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return []
    if not out:
        return []

    remote_files = [l.strip() for l in out.splitlines() if l.strip() and "oot_prediction" in l.lower()]
    local_copies = []
    tmpdir = tempfile.mkdtemp(prefix="auto_eval_")
    for rf in remote_files:
        local_f = os.path.join(tmpdir, os.path.basename(rf))
        try:
            scp_path = f"{ssh}:{rf}" if "\\" not in rf else f'{ssh}:"{rf}"'
            subprocess.check_call(f'scp -o ConnectTimeout=5 {scp_path} "{local_f}"',
                                  shell=True, timeout=30, stderr=subprocess.DEVNULL)
            local_copies.append((rf, local_f))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            continue
    return local_copies


def load_eval_log():
    if os.path.exists(EVAL_LOG):
        with open(EVAL_LOG) as f:
            return json.load(f)
    return []


def save_eval_log(entries):
    os.makedirs(os.path.dirname(EVAL_LOG), exist_ok=True)
    with open(EVAL_LOG, "w") as f:
        json.dump(entries, f, indent=2)


def print_summary(all_results):
    """Print a compact summary table."""
    if not all_results:
        print("No results to display.")
        return
    print(f"\n{'='*90}")
    print(f"{'Node':<8} {'Experiment':<30} {'File':<25} {'Hz':<4} {'IC':>8} {'DA%':>7} {'IC_T5%':>8}")
    print(f"{'-'*90}")
    for r in all_results:
        for hz in HORIZONS:
            if hz not in r.get("metrics", {}):
                continue
            m = r["metrics"][hz]
            ic_all = m.get("All", {}).get("IC", "n/a")
            da_all = m.get("All", {}).get("DA", "n/a")
            ic_t5 = m.get("Top5%", {}).get("IC", "n/a")
            ic_s = f"{ic_all:>8.4f}" if isinstance(ic_all, float) else f"{ic_all:>8}"
            da_s = f"{da_all:>7.1f}" if isinstance(da_all, float) else f"{da_all:>7}"
            t5_s = f"{ic_t5:>8.4f}" if isinstance(ic_t5, float) else f"{ic_t5:>8}"
            exp = r.get("experiment", "")[:30]
            fname = os.path.basename(r.get("file", ""))[:25]
            print(f"{r['node']:<8} {exp:<30} {fname:<25} {hz:<4} {ic_s} {da_s} {t5_s}")
    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="Auto-evaluate fold predictions")
    parser.add_argument("--local-only", action="store_true", help="Only scan Jupiter (no SSH)")
    parser.add_argument("--node", choices=["neptune", "razer", "jupiter"], help="Scan specific node only")
    args = parser.parse_args()

    nodes_to_scan = {}
    if args.node:
        nodes_to_scan = {args.node: NODES[args.node]}
    elif args.local_only:
        nodes_to_scan = {"jupiter": NODES["jupiter"]}
    else:
        nodes_to_scan = NODES

    eval_log = load_eval_log()
    seen = {e["file"] for e in eval_log}
    all_results = []

    for node_name, cfg in nodes_to_scan.items():
        print(f"[{node_name}] Scanning {cfg['path']} ...")
        if cfg["local"]:
            npz_files = [(f, f) for f in find_local_npz(cfg["path"])]
        else:
            npz_files = find_remote_npz(cfg)  # returns (remote_path, local_copy)

        if not npz_files:
            print(f"  No prediction files found.")
            continue
        print(f"  Found {len(npz_files)} file(s)")

        for orig_path, local_path in npz_files:
            if orig_path in seen:
                continue
            metrics = evaluate_npz(local_path)
            if "error" in metrics:
                print(f"  ERROR {os.path.basename(orig_path)}: {metrics['error']}")
                continue
            if not metrics:
                continue
            # Infer experiment name from parent dir
            parts = orig_path.replace("\\", "/").split("/")
            exp_name = parts[-2] if len(parts) >= 2 else "unknown"
            is_concat = "concat" in os.path.basename(orig_path).lower()

            entry = {
                "node": node_name, "file": orig_path, "experiment": exp_name,
                "is_concat": is_concat, "evaluated_at": datetime.utcnow().isoformat(),
                "metrics": metrics,
            }
            all_results.append(entry)
            eval_log.append(entry)
            seen.add(orig_path)

    save_eval_log(eval_log)
    print(f"\nEvaluation log: {EVAL_LOG} ({len(eval_log)} total entries)")
    print_summary(all_results if all_results else eval_log[-20:])


if __name__ == "__main__":
    main()
