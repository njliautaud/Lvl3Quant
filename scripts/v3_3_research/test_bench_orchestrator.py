"""
HC #358 — Jupiter parallel test-bench orchestrator.

Coordinates per-model bench runs (v2, v3.2, v3.3) and the final cross-model
verdict. Polls for v3.3 NPZ if not yet present.

Run with CPU pinning:
    nice -n 18 taskset -c 6,7,8,9,10,11 python3 test_bench_orchestrator.py

Progress is written to:
    output/v_test_bench_20260514/orchestrator.log
    output/v_test_bench_20260514/PROGRESS.json    (machine-readable status)

The orchestrator launches each model's bench as a subprocess (pinned, niced) so
that they execute SEQUENTIALLY by default. Per HC #358 the task is labeled
"parallel test-bench" referring to the OVERALL effort (multiple models in one
sweep). On a 16-core box with v3.3 inference + v2 ALL-OOT already running, we
intentionally run our 3 model benches sequentially to avoid CPU/RAM contention.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path("/home/jupiter/Lvl3Quant")
LABELS_DIR = REPO / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
BENCH_ROOT = REPO / "output" / "v_test_bench_20260514"

V2_NPZ = REPO / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_oot_predictions.npz"
V3_2_NPZ = REPO / "output" / "v3_2_deep_sim_20260512" / "fold_00_oot_predictions.npz"
V3_3_NPZ = REPO / "output" / "v3_3_oot_20260223" / "predictions.npz"

PER_MODEL_SCRIPT = REPO / "scripts" / "v3_3_research" / "test_bench_per_model.py"
CROSS_MODEL_SCRIPT = REPO / "scripts" / "v3_3_research" / "test_bench_cross_model.py"


def log(msg: str, log_path: Path):
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def write_progress(progress: dict, path: Path):
    path.write_text(json.dumps(progress, indent=2, default=str))


def run_model_bench(model: str, npz: Path, dates_csv: str,
                     out_dir: Path, log_path: Path,
                     optuna_trials: int = 25) -> int:
    cmd = [
        "python3", str(PER_MODEL_SCRIPT),
        "--model", model,
        "--npz", str(npz),
        "--labels-dir", str(LABELS_DIR),
        "--out-dir", str(out_dir),
        "--optuna-trials", str(optuna_trials),
    ]
    if dates_csv:
        cmd += ["--dates", dates_csv]

    log(f"LAUNCH {model}: {' '.join(cmd)}", log_path)
    t0 = time.time()
    sub_log = out_dir / "bench.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(sub_log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    log(f"{model} returncode={r.returncode} elapsed={elapsed:.1f}s sublog={sub_log}", log_path)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-v33-wait", action="store_true")
    ap.add_argument("--v33-poll-min", type=int, default=30,
                    help="minutes to poll for v3.3 NPZ before proceeding without it")
    ap.add_argument("--optuna-trials", type=int, default=25)
    args = ap.parse_args()

    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = BENCH_ROOT / "orchestrator.log"
    progress_path = BENCH_ROOT / "PROGRESS.json"
    progress = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "phases": {},
    }
    log("orchestrator starting", log_path)
    log(f"BENCH_ROOT={BENCH_ROOT}", log_path)

    # Phase 1: v2 ---------------------------------------------------------
    progress["phases"]["v2"] = {"status": "running", "started": datetime.utcnow().isoformat()}
    write_progress(progress, progress_path)
    rc_v2 = run_model_bench("v2", V2_NPZ, "20260224", BENCH_ROOT / "v2", log_path,
                             optuna_trials=args.optuna_trials)
    progress["phases"]["v2"]["status"] = "done" if rc_v2 == 0 else f"failed (rc={rc_v2})"
    progress["phases"]["v2"]["ended"] = datetime.utcnow().isoformat()
    write_progress(progress, progress_path)

    # Phase 2: v3.2 -------------------------------------------------------
    progress["phases"]["v3_2"] = {"status": "running", "started": datetime.utcnow().isoformat()}
    write_progress(progress, progress_path)
    rc_v32 = run_model_bench("v3_2", V3_2_NPZ, "", BENCH_ROOT / "v3_2", log_path,
                              optuna_trials=args.optuna_trials)
    progress["phases"]["v3_2"]["status"] = "done" if rc_v32 == 0 else f"failed (rc={rc_v32})"
    progress["phases"]["v3_2"]["ended"] = datetime.utcnow().isoformat()
    write_progress(progress, progress_path)

    # Phase 3: poll for v3.3, then run if available -----------------------
    v33_ready = V3_3_NPZ.exists()
    if not v33_ready and not args.skip_v33_wait:
        deadline = time.time() + args.v33_poll_min * 60
        log(f"v3.3 NPZ not present; polling for up to {args.v33_poll_min} min", log_path)
        while time.time() < deadline:
            time.sleep(60)
            if V3_3_NPZ.exists():
                v33_ready = True
                break
    if v33_ready:
        progress["phases"]["v3_3"] = {"status": "running", "started": datetime.utcnow().isoformat()}
        write_progress(progress, progress_path)
        rc_v33 = run_model_bench("v3_3", V3_3_NPZ, "20260223",
                                  BENCH_ROOT / "v3_3", log_path,
                                  optuna_trials=args.optuna_trials)
        progress["phases"]["v3_3"]["status"] = "done" if rc_v33 == 0 else f"failed (rc={rc_v33})"
        progress["phases"]["v3_3"]["ended"] = datetime.utcnow().isoformat()
    else:
        log("v3.3 NPZ never appeared — proceeding with v2 + v3.2 only", log_path)
        progress["phases"]["v3_3"] = {"status": "skipped (NPZ not available)"}
    write_progress(progress, progress_path)

    # Phase 4: cross-model verdict ---------------------------------------
    progress["phases"]["cross_model"] = {"status": "running", "started": datetime.utcnow().isoformat()}
    write_progress(progress, progress_path)
    models_done = [m for m in ["v2", "v3_2", "v3_3"]
                    if (BENCH_ROOT / m / "recommended_config.json").exists()]
    log(f"models with bench done: {models_done}", log_path)
    cmd = [
        "python3", str(CROSS_MODEL_SCRIPT),
        "--bench-root", str(BENCH_ROOT),
        "--labels-dir", str(LABELS_DIR),
        "--models", ",".join(models_done),
    ]
    log(f"LAUNCH cross-model: {' '.join(cmd)}", log_path)
    rc_x = subprocess.run(cmd).returncode
    progress["phases"]["cross_model"]["status"] = "done" if rc_x == 0 else f"failed (rc={rc_x})"
    progress["phases"]["cross_model"]["ended"] = datetime.utcnow().isoformat()
    progress["finished_at"] = datetime.utcnow().isoformat() + "Z"
    write_progress(progress, progress_path)

    log("orchestrator complete", log_path)


if __name__ == "__main__":
    main()
