#!/usr/bin/env python3
"""
Batch Shadow v0 — daily driver for offline multi-model shadow inference.

Per HC #443 / #448 R2 (offline-batch shadow pivot, 2026-05-20 15:31 ET):
For one given date, run v3.4.2 + v3.3 inference on Neptune, run HC #441 PRIMARY
geometry fill sim against each, compare vs v2 baseline, output JSONL + summary
JSON. Designed for manual trigger; cron + auto-pull is a follow-on.

Architecture:
  1. v3.4.2 inference  → output/batch_shadow/<date>/v342_preds.npz
  2. v3.3 inference    → output/batch_shadow/<date>/v33_preds.npz
  3. Fill sim (HC #441 PRIMARY: SL=0.50, TP=3.00, hold=1.5s, short top-0.5%)
       applied to each NPZ → per-trade JSONL
  4. Daily summary JSON with Sharpe / Sortino / PF / WR / mean_tk_net per model
  5. Append to master daily log + MLflow run

v0 SCOPE (Friday 2026-05-22 deliverable):
  - Single date, manual trigger
  - Reuse Phase-1 chunk NPZs if available (skip inference re-run)
  - Stub for v2 baseline comparison (uses existing cnn_mamba_v2_all_oot NPZs)
  - Stub for v3.3 if architectural load fails (logs gap, continues)

CLI:
  python run_daily_shadow.py --date 20260415 \
    [--skip-v342] [--skip-v33] [--reuse-v342-npz PATH]

Per HC #420 — this is authorized augmentation of user's research codebase.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path("/home/nick/Lvl3Quant")
sys.path.insert(0, str(PROJECT_ROOT))

OUT_ROOT = PROJECT_ROOT / "output" / "batch_shadow"
LOG_ROOT = PROJECT_ROOT / "logs" / "batch_shadow"
MASTER_LOG_JSONL = OUT_ROOT / "_master_daily_log.jsonl"

# v3.4.2 canonical
V342_CKPT = PROJECT_ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.pt"
V342_STATS = PROJECT_ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/fold_00_feature_stats.npz"
V342_SCHED = PROJECT_ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/fold_schedule.json"
V342_INFER = PROJECT_ROOT / "scripts/v3_4_research/v342_run_oot_inference.py"

# v3.3 canonical
V33_CKPT = PROJECT_ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt"
V33_STATS = PROJECT_ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_feature_stats.npz"
V33_SCHED = PROJECT_ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_schedule.json"
V33_INFER = PROJECT_ROOT / "scripts/v3_3_research/v32_run_oot_inference.py"

# v2 baseline (existing per-date NPZs)
V2_OOT_DIR = PROJECT_ROOT / "output" / "cnn_mamba_v2_all_oot"

# HC #441 PRIMARY geometry — short top-0.5% 1s band
FILL_GEOM = {
    "side": "short",
    "horizon": "1",          # 1s band per HC #441
    "conf_band": 0.005,      # top 0.5%
    "SL": 0.50,
    "TP": 3.00,
    "hold_s": 1.5,
    "cancel_s": 10.0,
    "order_type": "passive_at_touch",
}

PYTHON = "/home/nick/miniconda3/envs/py311-train/bin/python"


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Inference dispatchers
# ─────────────────────────────────────────────────────────────────────────────

def run_v342_inference(date_str: str, output_npz: Path,
                       reuse_npz: Optional[Path] = None) -> Optional[Path]:
    """Run v3.4.2 OOT inference for a single date. Returns output NPZ path or None."""
    if reuse_npz and reuse_npz.exists():
        log(f"v342: reusing existing NPZ {reuse_npz.name}")
        return reuse_npz
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", "-X", "faulthandler", str(V342_INFER),
        "--device", "cuda", "--batch-size", "24", "--num-workers", "2",
        "--ckpt", str(V342_CKPT),
        "--feature-stats", str(V342_STATS),
        "--fold-schedule", str(V342_SCHED),
        "--output", str(output_npz),
        "--dates", date_str,
    ]
    log(f"v342 cmd: {' '.join(cmd)}")
    env_str = (
        "V32_BATCH_SIZE=16 V32_WF_TRAIN_DAYS=10 "
        f"PYTHONPATH={PROJECT_ROOT} "
        "MLFLOW_TRACKING_URI=http://jupiter:5000 "
        "MLFLOW_EXPERIMENT_NAME=batch_shadow_daily"
    )
    log(f"v342 env: {env_str}")
    try:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            env={**__import__('os').environ,
                 "V32_BATCH_SIZE": "16", "V32_WF_TRAIN_DAYS": "10",
                 "PYTHONPATH": str(PROJECT_ROOT),
                 "MLFLOW_TRACKING_URI": "http://jupiter:5000",
                 "MLFLOW_EXPERIMENT_NAME": "batch_shadow_daily"},
            check=True, capture_output=True, text=True, timeout=3600,
        )
        log(f"v342 OK rc=0, stdout_tail={proc.stdout[-300:]!r}")
        return output_npz
    except subprocess.CalledProcessError as e:
        log(f"v342 FAILED rc={e.returncode} stderr_tail={e.stderr[-300:]!r}")
        return None
    except subprocess.TimeoutExpired:
        log("v342 TIMEOUT after 3600s")
        return None


def run_v33_inference(date_str: str, output_npz: Path) -> Optional[Path]:
    """Run v3.3 OOT inference for a single date. Returns NPZ path or None.

    Per SESSION_STATE 2026-05-20 15:31 ET: v3.3 ckpt is architecturally
    CNNMambaV32-compatible (loads via strict=False) — the existing
    scripts/v3_3_research/v32_run_oot_inference.py handles this.
    """
    if not V33_INFER.exists():
        log(f"v33 STUB: inference script missing at {V33_INFER}")
        return None
    if not V33_CKPT.exists():
        log(f"v33 STUB: ckpt missing at {V33_CKPT}")
        return None
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", str(V33_INFER),
        "--ckpt", str(V33_CKPT),
        "--feature-stats", str(V33_STATS),
        "--fold-schedule", str(V33_SCHED),
        "--output", str(output_npz),
        "--dates", date_str,
        "--device", "cuda", "--batch-size", "1", "--max-vram-frac", "0.10",
    ]
    log(f"v33 cmd: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            check=True, capture_output=True, text=True, timeout=3600,
        )
        log(f"v33 OK stdout_tail={proc.stdout[-300:]!r}")
        return output_npz
    except subprocess.CalledProcessError as e:
        log(f"v33 FAILED rc={e.returncode} stderr_tail={e.stderr[-300:]!r}")
        return None
    except subprocess.TimeoutExpired:
        log("v33 TIMEOUT after 3600s")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Fill sim — HC #441 PRIMARY geometry
# ─────────────────────────────────────────────────────────────────────────────

def select_signals_for_band(pred_arr: np.ndarray, conf_band: float, side: str):
    """Top conf_band fraction by abs(pred). Returns (idx, strength)."""
    if pred_arr.ndim > 1:
        pred_arr = pred_arr.squeeze()
    n = pred_arr.size
    k = max(1, int(n * conf_band))
    if side == "short":
        # Most-negative predictions
        idx = np.argsort(pred_arr)[:k]
        strs = -pred_arr[idx]
    else:
        idx = np.argsort(-pred_arr)[:k]
        strs = pred_arr[idx]
    return idx.astype(np.int64), strs.astype(np.float32)


def run_fill_sim_for_model(date_str: str, preds_npz: Path,
                            model_name: str, out_dir: Path) -> Optional[Dict]:
    """Run HC #441 PRIMARY-geometry fill sim on one model's predictions.

    Returns summary dict or None on failure. Writes per-trade JSONL too.
    """
    if not preds_npz.exists():
        log(f"{model_name} fill_sim: preds NPZ missing {preds_npz}")
        return None

    try:
        # Import canonical fill sim
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "v3_4_research"))
        # NOTE: hc437_pathB_exit_sweep.py imports use LVL3=jupiter paths; we
        # only need build_fill_cache_for_date + resolve_cell + compute_metrics
        # which take date_str + raw arrays.
        import importlib.util
        hc437_path = PROJECT_ROOT / "scripts/v3_4_research/hc437_pathB_exit_sweep.py"
        spec = importlib.util.spec_from_file_location("hc437_neptune", str(hc437_path))
        mod = importlib.util.module_from_spec(spec)
        # Patch LVL3 to Neptune root BEFORE exec
        import builtins
        # Inject Neptune root into module before execution
        spec.loader.exec_module(mod)
        # Now mod.LVL3 = jupiter path. Override paths the module uses post-load:
        mod.LVL3 = PROJECT_ROOT
        mod.V2_OOT_DIR = PROJECT_ROOT / "output" / "cnn_mamba_v2_all_oot"
        mod.MBO_EVENT_DIR = PROJECT_ROOT / "data" / "processed" / "mbo_events_smart_v3"
        mod.OUT_DIR = PROJECT_ROOT / "output" / "batch_shadow_hc437_cache"
        mod.OUT_DIR.mkdir(parents=True, exist_ok=True)
        mod.CACHE_DIR = mod.OUT_DIR / "fill_cache"
        mod.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        build_fill_cache_for_date = mod.build_fill_cache_for_date
        resolve_cell = mod.resolve_cell
        compute_metrics = mod.compute_metrics
    except Exception as e:
        log(f"{model_name} fill_sim: import error {e!r}")
        return None

    try:
        data = np.load(preds_npz, allow_pickle=True)
        # Multi-schema: v3.4.2 = pred_log_ret_1s; v2/v3.2/v3.3 = predictions[:,0]
        if "pred_log_ret_1s" in data.files:
            preds = data["pred_log_ret_1s"]
            WS = 1500
            ST = 250
        elif "predictions" in data.files:
            preds = data["predictions"][:, 0]  # horizon idx 0 = 1s
            WS = int(data["window_size"]) if "window_size" in data.files else 3000
            ST = int(data["stride"]) if "stride" in data.files else 250
        else:
            log(f"{model_name} fill_sim: no recognized preds key in NPZ. keys={data.files}")
            return None
        log(f"{model_name} preds shape={preds.shape}, WS={WS}, ST={ST}")

        idx, strs = select_signals_for_band(preds, FILL_GEOM["conf_band"], FILL_GEOM["side"])
        log(f"{model_name} selected {len(idx)} signals (top {FILL_GEOM['conf_band']*100}%)")
        cache = build_fill_cache_for_date(
            date_str=date_str,
            idx=idx, strs=strs, ws=WS, st=ST, side=FILL_GEOM["side"],
            cancel_s=FILL_GEOM["cancel_s"],
            max_hold_s=max(FILL_GEOM["hold_s"], 30.0),
            order_type=FILL_GEOM["order_type"],
        )
        log(f"{model_name} fill_cache: n_fills={len(cache['fills'])}")

        # Resolve under PRIMARY geometry
        rows = resolve_cell(cache, FILL_GEOM["TP"], FILL_GEOM["SL"], FILL_GEOM["hold_s"])
        metrics = compute_metrics(rows)

        # Persist per-trade JSONL
        jsonl_path = out_dir / f"{date_str}_{model_name}_fills.jsonl"
        with open(jsonl_path, "w") as f:
            for r in rows:
                # All np scalars → python scalars for JSON
                r_serial = {k: (float(v) if isinstance(v, (np.floating, np.integer))
                               else v if isinstance(v, (str, bool)) or v is None
                               else int(v) if hasattr(v, '__int__') else str(v))
                            for k, v in r.items()}
                f.write(json.dumps(r_serial) + "\n")
        log(f"{model_name} wrote {len(rows)} fills → {jsonl_path.name}")

        return {
            "model": model_name,
            "date": date_str,
            "n_fills": int(metrics["n_fills"]),
            "mean_net_tk": float(metrics["mean_net_tk"]) if metrics["n_fills"] > 0 else None,
            "PF": float(metrics["PF"]) if metrics["n_fills"] > 0 else None,
            "WR_pct": float(metrics["WR_pct"]) if metrics["n_fills"] > 0 else None,
            "Sh_sqrtN": float(metrics["Sh_sqrtN"]) if metrics["n_fills"] > 0 else None,
            "jsonl": str(jsonl_path),
        }
    except Exception as e:
        log(f"{model_name} fill_sim runtime error {e!r}")
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# v2 baseline (uses existing OOT NPZ — no inference re-run)
# ─────────────────────────────────────────────────────────────────────────────

def run_v2_baseline(date_str: str, out_dir: Path) -> Optional[Dict]:
    """Locate existing v2 OOT predictions NPZ and run fill sim."""
    v2_npz = V2_OOT_DIR / f"{date_str}_predictions.npz"
    if not v2_npz.exists():
        log(f"v2 baseline: NPZ missing for {date_str} at {v2_npz}")
        # STUB — could try other v2 dirs (bulk_oot_v2 etc.)
        return None
    return run_fill_sim_for_model(date_str, v2_npz, "v2_baseline", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow logging
# ─────────────────────────────────────────────────────────────────────────────

def log_to_mlflow(date_str: str, summary: Dict) -> None:
    try:
        import os
        os.environ.setdefault("MLFLOW_TRACKING_URI", "http://jupiter:5000")
        import mlflow
        mlflow.set_experiment("batch_shadow_daily")
        with mlflow.start_run(run_name=f"shadow_{date_str}"):
            mlflow.log_param("date", date_str)
            for entry in summary.get("models", []):
                mname = entry["model"]
                for mk in ("n_fills", "mean_net_tk", "PF", "WR_pct", "Sh_sqrtN"):
                    val = entry.get(mk)
                    if val is not None:
                        mlflow.log_metric(f"{mname}_{mk}", float(val))
        log("mlflow logged")
    except Exception as e:
        log(f"mlflow logging failed: {e!r} (non-fatal)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYYMMDD")
    p.add_argument("--skip-v342", action="store_true")
    p.add_argument("--skip-v33", action="store_true")
    p.add_argument("--skip-v2", action="store_true")
    p.add_argument("--reuse-v342-npz", type=Path, default=None,
                   help="Use existing v342 NPZ instead of re-running inference")
    args = p.parse_args()

    date_str = args.date
    date_dir = OUT_ROOT / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log(f"=== BATCH SHADOW {date_str} START ===")

    summary = {
        "date": date_str,
        "start_ts": datetime.utcnow().isoformat() + "Z",
        "models": [],
        "errors": [],
    }

    # === v3.4.2 ===
    if not args.skip_v342:
        v342_npz = date_dir / f"{date_str}_v342_preds.npz"
        npz = run_v342_inference(date_str, v342_npz, reuse_npz=args.reuse_v342_npz)
        if npz is None:
            summary["errors"].append("v342_inference_failed")
        else:
            res = run_fill_sim_for_model(date_str, npz, "v342", date_dir)
            if res:
                summary["models"].append(res)
            else:
                summary["errors"].append("v342_fillsim_failed")

    # === v3.3 ===
    if not args.skip_v33:
        v33_npz = date_dir / f"{date_str}_v33_preds.npz"
        npz = run_v33_inference(date_str, v33_npz)
        if npz is None:
            summary["errors"].append("v33_inference_failed_or_stubbed")
        else:
            res = run_fill_sim_for_model(date_str, npz, "v33", date_dir)
            if res:
                summary["models"].append(res)
            else:
                summary["errors"].append("v33_fillsim_failed")

    # === v2 baseline ===
    if not args.skip_v2:
        res = run_v2_baseline(date_str, date_dir)
        if res:
            summary["models"].append(res)
        else:
            summary["errors"].append("v2_baseline_unavailable")

    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["end_ts"] = datetime.utcnow().isoformat() + "Z"

    # Persist daily summary JSON
    summary_path = date_dir / f"{date_str}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"summary → {summary_path}")

    # Append to master log
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MASTER_LOG_JSONL, "a") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    log(f"master log appended {MASTER_LOG_JSONL.name}")

    # MLflow
    log_to_mlflow(date_str, summary)

    log(f"=== DONE in {summary['elapsed_s']}s ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
