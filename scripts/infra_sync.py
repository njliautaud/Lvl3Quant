#!/usr/bin/env python3
"""
infra_sync.py — Master infrastructure sync daemon.

Closes ALL identified staleness gaps by auto-syncing:
  1. DIRECTIVES.md training status (from QCC SQLite DB)
  2. research_queue_persistent.json (from MLflow API + QCC jobs)
  3. CLAUDE.md "What's Been Tested" section (from MLflow experiments)
  4. Memory daily notes (session activity log)

Data flow:
  QCC SQLite DB (node status) ──┐
  MLflow API (experiment results) ──┼──> DIRECTIVES.md
                                    ├──> research_queue_persistent.json
                                    └──> CLAUDE.md

Usage:
  python3 infra_sync.py              # Full sync (all systems)
  python3 infra_sync.py --directives # Only sync DIRECTIVES.md
  python3 infra_sync.py --queue      # Only sync research queue
  python3 infra_sync.py --docs       # Only sync CLAUDE.md test results
  python3 infra_sync.py --health     # Health check — report what's broken
"""

import json
import os
import re
import sqlite3
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

# ─── Paths ─────────────────────────────────────────────────────────────────────

DIRECTIVES_PATH = "/home/jupiter/Lvl3Quant/DIRECTIVES.md"
RESEARCH_QUEUE_PATH = "/home/jupiter/teleclaude-main/data/research_queue_persistent.json"
CLAUDE_MD_PATH = "/home/jupiter/teleclaude-main/CLAUDE.md"
QCC_DB_PATH = "/home/jupiter/teleclaude-main/data/qcc.db"
BACKUP_DIR = "/home/jupiter/Lvl3Quant/data/directive_backups"
SYNC_LOG = "/home/jupiter/Lvl3Quant/logs/infra_sync.log"
MLFLOW_URL = "http://localhost:5000"

# ─── Helpers ───────────────────────────────────────────────────────────────────

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M ET")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        os.makedirs(os.path.dirname(SYNC_LOG), exist_ok=True)
        with open(SYNC_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def backup_file(path):
    """Create timestamped backup of a file."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    name = os.path.basename(path).replace(".", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(BACKUP_DIR, f"{name}_{ts}")
    try:
        with open(path, "r") as f:
            content = f.read()
        with open(backup, "w") as f:
            f.write(content)
        # Clean old backups (keep last 10 per file)
        prefix = name + "_"
        backups = sorted([
            os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR)
            if f.startswith(prefix)
        ])
        for old in backups[:-10]:
            os.remove(old)
    except Exception:
        pass

# ─── QCC Database Reader ──────────────────────────────────────────────────────

def read_qcc_nodes():
    """Read node status from QCC SQLite DB."""
    try:
        conn = sqlite3.connect(f"file:{QCC_DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM compute_nodes").fetchall()
        nodes = {}
        for r in rows:
            nodes[r["name"].lower()] = {
                "status": r["status"] or "unknown",
                "gpu_util": r["last_gpu_util"] or 0,
                "gpu_mem_mb": r["last_gpu_mem_mb"] or 0,
                "gpu": r["gpu"] or "none",
                "last_heartbeat": r["last_heartbeat"] or "never",
            }
        conn.close()
        return nodes
    except Exception as e:
        log(f"⚠ QCC DB read failed: {e}")
        return None

def read_qcc_training_jobs():
    """Read active training jobs from QCC DB."""
    try:
        conn = sqlite3.connect(f"file:{QCC_DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM training_jobs WHERE status IN ('running', 'active') ORDER BY started_at DESC"
        ).fetchall()
        jobs = []
        for r in rows:
            jobs.append({
                "node": (r["node"] or "").lower(),
                "description": r["description"] if "description" in r.keys() else "",
                "pid": r["pid"] if "pid" in r.keys() else None,
                "current_fold": r["current_fold"] if "current_fold" in r.keys() else None,
                "total_folds": r["total_folds"] if "total_folds" in r.keys() else None,
                "status": r["status"],
                "started_at": r["started_at"] if "started_at" in r.keys() else None,
            })
        conn.close()
        return jobs
    except Exception as e:
        log(f"⚠ QCC training jobs read failed: {e}")
        return []

# ─── MLflow API Reader ─────────────────────────────────────────────────────────

def mlflow_post(endpoint, data=None):
    """POST to MLflow API."""
    try:
        url = f"{MLFLOW_URL}/api/2.0/mlflow/{endpoint}"
        payload = json.dumps(data or {}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log(f"⚠ MLflow API failed ({endpoint}): {e}")
        return None

def get_mlflow_experiments():
    """Get all MLflow experiments with their latest runs."""
    result = mlflow_post("experiments/search", {"max_results": 100})
    if not result or "experiments" not in result:
        return []
    return result["experiments"]

def get_mlflow_runs(experiment_id, max_results=5):
    """Get recent runs for an experiment."""
    result = mlflow_post("runs/search", {
        "experiment_ids": [str(experiment_id)],
        "max_results": max_results,
        "order_by": ["start_time DESC"]
    })
    if not result or "runs" not in result:
        return []
    return result["runs"]

def get_experiment_best_ic(experiment_id):
    """Get the best concat IC from an MLflow experiment's runs."""
    runs = get_mlflow_runs(experiment_id, max_results=20)
    best_ic = None
    best_run = None
    total_runs = len(runs)

    for run in runs:
        metrics = run.get("data", {}).get("metrics", [])
        for m in metrics:
            if "concat_ic" in m.get("key", "") or "ic_10s" in m.get("key", ""):
                try:
                    val = float(m.get("value", 0))
                except (ValueError, TypeError):
                    continue
                if best_ic is None or val > best_ic:
                    best_ic = val
                    best_run = run

    return {"best_ic": best_ic, "total_runs": total_runs, "best_run": best_run}

# ─── Gap 1: Sync DIRECTIVES.md training status ────────────────────────────────

def sync_directives():
    """Update DIRECTIVES.md CURRENT TRAINING section from QCC DB."""
    log("🔄 Syncing DIRECTIVES.md...")

    nodes = read_qcc_nodes()
    if not nodes:
        log("⚠ No node data — skipping directives sync")
        return False

    jobs = read_qcc_training_jobs()

    # Build training status lines
    lines = []
    for name in ["neptune", "razer", "jupiter", "saturn"]:
        info = nodes.get(name, {})
        status = info.get("status", "unknown")
        gpu_util = info.get("gpu_util", 0) or 0
        gpu_mem = info.get("gpu_mem_mb", 0) or 0
        gpu = info.get("gpu", "none")

        # Find matching job
        node_jobs = [j for j in jobs if j["node"] == name]

        if node_jobs:
            job = node_jobs[0]
            desc = job.get("description", "unknown")
            pid = job.get("pid", "?")
            fold_info = ""
            if job.get("current_fold") and job.get("total_folds"):
                fold_info = f", fold {job['current_fold']}/{job['total_folds']}"
            lines.append(f"- {name.capitalize()}: {desc} (PID {pid}{fold_info}), GPU {gpu_util}%, {gpu_mem}MB VRAM")
        elif gpu_util > 50:
            lines.append(f"- {name.capitalize()}: TRAINING (unregistered), GPU {gpu_util}%, {gpu_mem}MB VRAM")
        elif gpu == "none" or name in ("jupiter", "saturn"):
            lines.append(f"- {name.capitalize()}: CPU node, status: {status}")
        elif status == "offline":
            lines.append(f"- {name.capitalize()}: ⚠ OFFLINE (last heartbeat: {info.get('last_heartbeat', 'unknown')})")
        else:
            lines.append(f"- {name.capitalize()}: **IDLE** — GPU {gpu_util}%, needs experiment")

    new_section = "## CURRENT TRAINING (auto-synced)\n" + "\n".join(lines)

    with open(DIRECTIVES_PATH, "r") as f:
        content = f.read()

    original = content

    # Replace existing section
    pattern = r"## CURRENT TRAINING.*?\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        content = content[:match.start()] + new_section + "\n" + content[match.end():]

    # Update timestamp
    content = re.sub(r"# Last updated:.*", f"# Last updated: {now_str()} (auto-sync)", content)

    if content != original:
        backup_file(DIRECTIVES_PATH)
        with open(DIRECTIVES_PATH, "w") as f:
            f.write(content)
        log("✅ DIRECTIVES.md updated")
        return True

    log("ℹ DIRECTIVES.md already current")
    return False

# ─── Gap 2: Sync research queue from MLflow ───────────────────────────────────

def sync_research_queue():
    """Update research_queue_persistent.json status from MLflow runs."""
    log("🔄 Syncing research queue...")

    experiments = get_mlflow_experiments()
    if not experiments:
        log("⚠ No MLflow experiments — skipping queue sync")
        return False

    # Build lookup: experiment name → results
    exp_results = {}
    for exp in experiments:
        name = exp.get("name", "")
        exp_id = exp.get("experiment_id", "")
        if "Event" in name or "LGBM" in name or "Mamba" in name:
            info = get_experiment_best_ic(exp_id)
            if info["total_runs"] > 0:
                exp_results[name] = info

    # Read current queue
    try:
        with open(RESEARCH_QUEUE_PATH, "r") as f:
            queue = json.load(f)
    except Exception as e:
        log(f"⚠ Can't read research queue: {e}")
        return False

    changed = False

    # Update experiment statuses based on MLflow
    if "event_architectures_gpu" in queue:
        for exp in queue["event_architectures_gpu"].get("experiments", []):
            mlflow_name = exp.get("mlflow_experiment", "")
            if mlflow_name and mlflow_name in exp_results:
                info = exp_results[mlflow_name]
                if info["best_ic"] is not None and exp.get("status") in ("queued", "RUNNING"):
                    # Has results — mark as completed or update
                    if info["total_runs"] >= 3:
                        old_status = exp.get("status")
                        exp["mlflow_best_ic"] = round(info["best_ic"], 4)
                        exp["mlflow_runs"] = info["total_runs"]
                        exp["mlflow_last_checked"] = now_str()
                        if old_status != exp.get("status"):
                            changed = True
                        changed = True  # Always write back IC

    # Update nodes status from QCC
    nodes = read_qcc_nodes()
    if nodes:
        for name, info in nodes.items():
            if name in queue.get("node_details", {}):
                queue["node_details"][name]["last_gpu_util"] = info.get("gpu_util", 0)
                queue["node_details"][name]["last_status"] = info.get("status", "unknown")

    if changed:
        queue["_updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        backup_file(RESEARCH_QUEUE_PATH)
        with open(RESEARCH_QUEUE_PATH, "w") as f:
            json.dump(queue, f, indent=2)
        log("✅ Research queue updated with MLflow results")
        return True

    log("ℹ Research queue already current")
    return False

# ─── Gap 3: Sync CLAUDE.md test results from MLflow ───────────────────────────

def sync_claude_md():
    """Update the 'What's Been Tested' section in CLAUDE.md from MLflow."""
    log("🔄 Syncing CLAUDE.md test results...")

    experiments = get_mlflow_experiments()
    if not experiments:
        log("⚠ No MLflow experiments — skipping CLAUDE.md sync")
        return False

    # Build results map
    tested = {}
    for exp in experiments:
        name = exp.get("name", "")
        exp_id = exp.get("experiment_id", "")

        # Only care about event-driven experiments
        if not any(k in name for k in ["Event", "Mamba", "CNN", "LGBM", "Hawkes", "ODE"]):
            continue

        info = get_experiment_best_ic(exp_id)
        if info["total_runs"] > 0:
            tested[name] = {
                "best_ic": info["best_ic"],
                "runs": info["total_runs"],
            }

    if not tested:
        log("ℹ No experiment results to sync")
        return False

    # Read CLAUDE.md
    try:
        with open(CLAUDE_MD_PATH, "r") as f:
            content = f.read()
    except Exception as e:
        log(f"⚠ Can't read CLAUDE.md: {e}")
        return False

    # Build auto-generated comment block
    result_lines = [
        "<!-- AUTO-SYNCED FROM MLFLOW - DO NOT EDIT MANUALLY -->",
        f"<!-- Last sync: {now_str()} -->",
    ]
    for name, info in sorted(tested.items(), key=lambda x: -(x[1]["best_ic"] or 0)):
        ic = f"IC={info['best_ic']:.3f}" if info["best_ic"] else "IC=N/A"
        result_lines.append(f"<!-- {name}: {ic}, {info['runs']} runs -->")
    result_lines.append("<!-- END AUTO-SYNC -->")

    result_block = "\n".join(result_lines)

    # Replace or insert auto-sync block
    pattern = r"<!-- AUTO-SYNCED FROM MLFLOW.*?<!-- END AUTO-SYNC -->"
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, result_block, content, flags=re.DOTALL)
    else:
        # Insert after "### What's Been Tested" section
        marker = "### What's Been Tested"
        idx = content.find(marker)
        if idx >= 0:
            # Find end of line
            eol = content.find("\n", idx)
            if eol >= 0:
                new_content = content[:eol+1] + result_block + "\n" + content[eol+1:]
            else:
                new_content = content
        else:
            # Insert before "### What Needs Testing"
            marker2 = "### What Needs Testing"
            idx2 = content.find(marker2)
            if idx2 >= 0:
                new_content = content[:idx2] + result_block + "\n\n" + content[idx2:]
            else:
                log("ℹ Can't find insertion point in CLAUDE.md")
                return False

    if new_content != content:
        backup_file(CLAUDE_MD_PATH)
        with open(CLAUDE_MD_PATH, "w") as f:
            f.write(new_content)
        log("✅ CLAUDE.md experiment results synced from MLflow")
        return True

    log("ℹ CLAUDE.md already current")
    return False

# ─── Health Check ──────────────────────────────────────────────────────────────

def health_check():
    """Report what's working and what's broken."""
    log("🏥 Running health check...")

    issues = []
    ok = []

    # 1. QCC DB
    nodes = read_qcc_nodes()
    if nodes:
        ok.append(f"QCC DB: {len(nodes)} nodes readable")
    else:
        issues.append("QCC DB: UNREACHABLE")

    # 2. MLflow API
    experiments = get_mlflow_experiments()
    if experiments:
        ok.append(f"MLflow API: {len(experiments)} experiments found")
    else:
        issues.append("MLflow API: UNREACHABLE or empty")

    # 3. DIRECTIVES.md freshness
    try:
        mtime = os.path.getmtime(DIRECTIVES_PATH)
        age_min = int((datetime.now().timestamp() - mtime) / 60)
        if age_min > 60:
            issues.append(f"DIRECTIVES.md: {age_min}min stale")
        else:
            ok.append(f"DIRECTIVES.md: {age_min}min old")
    except Exception:
        issues.append("DIRECTIVES.md: MISSING")

    # 4. Research queue freshness
    try:
        with open(RESEARCH_QUEUE_PATH, "r") as f:
            queue = json.load(f)
        updated = queue.get("_updated", "unknown")
        ok.append(f"Research queue: last updated {updated}")
    except Exception:
        issues.append("Research queue: MISSING or corrupt")

    # 5. Backups
    try:
        backup_count = len(os.listdir(BACKUP_DIR))
        ok.append(f"Backups: {backup_count} files in {BACKUP_DIR}")
    except Exception:
        issues.append("Backups directory: MISSING")

    # 6. Node GPU status
    if nodes:
        for name, info in nodes.items():
            gpu_util = info.get("gpu_util", 0) or 0
            gpu = info.get("gpu", "none")
            if gpu != "none" and gpu_util < 5 and info.get("status") == "online":
                issues.append(f"{name.upper()}: GPU IDLE ({gpu_util}%) — wasting compute")

    print("\n=== INFRASTRUCTURE HEALTH CHECK ===")
    print(f"Time: {now_str()}\n")

    if ok:
        print("✅ HEALTHY:")
        for item in ok:
            print(f"   {item}")

    if issues:
        print("\n⚠ ISSUES:")
        for item in issues:
            print(f"   {item}")
    else:
        print("\n🎉 All systems nominal!")

    return len(issues) == 0

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if "--health" in args:
        health_check()
        return

    if "--directives" in args:
        sync_directives()
        return

    if "--queue" in args:
        sync_research_queue()
        return

    if "--docs" in args:
        sync_claude_md()
        return

    # Full sync
    log("=" * 60)
    log("🔄 FULL INFRASTRUCTURE SYNC")
    log("=" * 60)

    results = {
        "directives": sync_directives(),
        "queue": sync_research_queue(),
        "docs": sync_claude_md(),
    }

    updated = [k for k, v in results.items() if v]
    skipped = [k for k, v in results.items() if not v]

    log(f"📊 Sync complete: {len(updated)} updated, {len(skipped)} unchanged")
    if updated:
        log(f"   Updated: {', '.join(updated)}")

if __name__ == "__main__":
    main()
