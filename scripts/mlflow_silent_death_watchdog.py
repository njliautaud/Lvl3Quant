#!/usr/bin/env python3
"""
HC #456 R3 — Silent-death watchdog for training runs.

Catches the failure mode where an MLflow run is status=RUNNING but the
training process has died (OOM, segfault, SSH disconnect, etc.) and no
metric has been written in N minutes.

Run via cron every 10 min. Fires Discord alert (via autonomy_inject) on
first detection only (de-duped via state file).

Two zombie criteria:
  1. n_metrics > 0 AND latest_metric_age_min > METRIC_STALE_MIN
  2. n_metrics == 0 AND start_age_min > NO_METRIC_GRACE_MIN (training
     never got past dataset build → almost certainly OOM)

When zombie detected:
  - Mark MLflow run as FAILED with end_time = now
  - Append to state file (so we don't re-alert)
  - Inject Discord prompt for Claude to investigate

Author: HC #456 R3 binding rule.
"""
import json
import os
import sys
import time
import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")

# Node SSH details for GPU-alive verification before alerting
NODE_SSH = {
    "neptune": "nick@neptune",
    "razer": "claude@razer",
}
STATE_FILE = Path(os.environ.get(
    "WATCHDOG_STATE",
    "/home/jupiter/Lvl3Quant/logs/mlflow_silent_death_state.json"))
LOG_FILE = Path("/home/jupiter/Lvl3Quant/logs/mlflow_silent_death_watchdog.log")
INJECT_SH = "/home/jupiter/Lvl3Quant/scripts/autonomy_inject.sh"

# Thresholds
METRIC_STALE_MIN = 20      # if metrics exist but last write > 20 min → zombie
NO_METRIC_GRACE_MIN = 360  # if no metrics at all but run open > 360 min → zombie
                           # NOTE: v3.4.2 fold-0 first epoch + OOT validation takes ~75 min.
                           # Was 45 → killed v3.4.2 runs before epoch 1 validation could log.
                           # 120 min covered batch_size=12/96 short-context runs (HC #457 — 2026-05-21).
                           # 2026-05-22 03:03 ET: HC #485 v2 retrain has 22137 batches/ep × 0.9s ≈ 5.5h/ep.
                           # 120 min FP'd run a96f146c at 128.9 min (training healthy). Bumped to 360 min
                           # to cover current trainer's fold-0 ep-1 cadence with margin. Proper fix per
                           # HC #484 R2 (root-cause not suppression).
MAX_RUNS_PER_EXP = 50      # scan recent runs only
MAX_ALERTS_PER_INVOCATION = 3  # safety cap to prevent inject-storms

# 2026-06-09: paper-trading experiments emit metrics on cycle events, not
# heartbeat, and can have multi-hour warmup before first metric. Skip them
# from silent-death detection entirely — these are LIVE long-running engines,
# not bounded training jobs. Crash detection is process-level, not metric-level.
SKIP_EXPERIMENT_NAME_SUBSTRINGS = ("paper-wheel", "paper_wheel", "wheel_paper", "live_trading")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {"alerted": []}
    return {"alerted": []}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def gpu_has_training_process(node_name: str) -> bool:
    """SSH to node and check if a python training process is using the GPU.
    Returns True if GPU is active with training, False if idle/unreachable."""
    ssh_target = NODE_SSH.get(node_name)
    if not ssh_target:
        return False  # unknown node, allow alert
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no",
             ssh_target,
             "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null && "
             "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return False
        lines = result.stdout.strip().split("\n")
        # Last line is GPU utilization
        if lines:
            try:
                gpu_util = int(lines[-1].strip())
                if gpu_util >= 50:
                    log(f"  GPU-alive check for {node_name}: util={gpu_util}% — SUPPRESSING false positive")
                    return True
            except ValueError:
                pass
        return False
    except Exception as e:
        log(f"  GPU-alive check for {node_name} failed: {e}")
        return False  # can't verify, allow alert


def inject_alert(msg: str) -> None:
    """Send Discord alert via the durable autonomy_inject endpoint."""
    try:
        subprocess.run([INJECT_SH, msg], timeout=10, check=False,
                       capture_output=True)
    except Exception as e:
        log(f"inject failed: {e}")


def latest_metric_age_min(client: MlflowClient, run_id: str,
                          metric_keys: list) -> float | None:
    """Return age of the most recent metric write across all keys, in minutes."""
    if not metric_keys:
        return None
    now_ms = int(time.time() * 1000)
    newest_ts = 0
    for k in metric_keys:
        try:
            hist = client.get_metric_history(run_id, k)
        except Exception:
            continue
        for h in hist:
            if h.timestamp > newest_ts:
                newest_ts = h.timestamp
    if newest_ts == 0:
        return None
    return (now_ms - newest_ts) / 1000.0 / 60.0


def main() -> int:
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    state = load_state()
    alerted = set(state.get("alerted", []))

    try:
        exps = client.search_experiments(max_results=200)
    except Exception as e:
        log(f"search_experiments failed: {e}")
        return 1

    exp_ids = [e.experiment_id for e in exps]
    if not exp_ids:
        log("no experiments")
        return 0

    try:
        runs = client.search_runs(
            exp_ids,
            filter_string="attributes.status='RUNNING'",
            max_results=200,
        )
    except Exception as e:
        log(f"search_runs failed: {e}")
        return 1

    now_ms = int(time.time() * 1000)
    n_scanned = 0
    n_zombie = 0
    n_new_alerts = 0

    # Build experiment_id -> experiment_name map for skip filter
    exp_name_by_id = {e.experiment_id: e.name for e in exps}

    for r in runs:
        n_scanned += 1
        rid = r.info.run_id
        run_name = r.data.tags.get("mlflow.runName", "?")
        exp_name = exp_name_by_id.get(r.info.experiment_id, "")
        # Skip paper-trading / live-engine experiments
        if any(s in exp_name.lower() or s in run_name.lower()
               for s in SKIP_EXPERIMENT_NAME_SUBSTRINGS):
            log(f"SKIP run={rid[:8]} name={run_name} exp={exp_name} — paper/live engine, not training")
            continue
        start_age_min = (now_ms - (r.info.start_time or now_ms)) / 1000.0 / 60.0
        metric_keys = list(r.data.metrics.keys())
        age_min = latest_metric_age_min(client, rid, metric_keys)

        # Classify
        zombie = False
        reason = ""
        if age_min is not None and age_min > METRIC_STALE_MIN:
            zombie = True
            reason = f"last metric {age_min:.1f} min ago"
        elif age_min is None and start_age_min > NO_METRIC_GRACE_MIN:
            zombie = True
            reason = f"no metrics in {start_age_min:.1f} min since start"

        if not zombie:
            continue

        n_zombie += 1

        # HC #491 R5 FIX: Before alerting, check if the GPU node actually
        # has training running. Prevents false positives when MLflow metrics
        # are stale but training is healthy (logging to local file instead).
        node_tag = r.data.tags.get("node", "")
        if not node_tag:
            # Try to infer node from experiment name or run name
            rn_lower = (run_name + " " + r.info.experiment_id).lower()
            if "neptune" in rn_lower:
                node_tag = "neptune"
            elif "razer" in rn_lower:
                node_tag = "razer"

        if node_tag and gpu_has_training_process(node_tag):
            log(f"SUPPRESSED run={rid[:8]} name={run_name} reason={reason} — "
                f"GPU on {node_tag} is active, training healthy despite stale MLflow metrics")
            # Mark as known-alive so we don't re-check every 10 min
            alerted.add(rid)
            continue

        log(f"ZOMBIE run={rid[:8]} name={run_name} start_age_min={start_age_min:.1f} "
            f"reason={reason} n_metrics={len(metric_keys)}")

        # Mark FAILED in MLflow
        try:
            client.set_terminated(rid, status="FAILED", end_time=now_ms)
            client.set_tag(rid, "watchdog.killed_at", time.strftime("%Y-%m-%d %H:%M:%S"))
            client.set_tag(rid, "watchdog.reason", reason)
        except Exception as e:
            log(f"  set_terminated failed: {e}")

        # Alert (once per run, capped per invocation)
        if rid not in alerted:
            alerted.add(rid)
            if n_new_alerts < MAX_ALERTS_PER_INVOCATION:
                n_new_alerts += 1
                msg = (f"SILENT_DEATH: training run '{run_name}' "
                       f"({rid[:8]}) marked FAILED by watchdog — {reason}. "
                       f"Investigate: check Neptune/Razer dmesg for OOM, "
                       f"verify GPU idle, decide if relaunch needed per DIRECTIVES.")
                inject_alert(msg)
                log(f"  ALERT sent for {rid[:8]}")
            else:
                log(f"  ALERT suppressed for {rid[:8]} (cap {MAX_ALERTS_PER_INVOCATION} reached)")

    state["alerted"] = sorted(alerted)
    save_state(state)
    log(f"scan complete: scanned={n_scanned} zombies={n_zombie} new_alerts={n_new_alerts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
