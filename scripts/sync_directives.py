#!/usr/bin/env python3
"""
sync_directives.py — Auto-sync DIRECTIVES.md with live cluster state.

Layers of protection for DIRECTIVES.md:
  1. BEHAVIORAL: Claude updates immediately when user gives instructions (reinforced by hook)
  2. HOOK: UserPromptSubmit hook injects reminder to check for new user directives
  3. CRON: This script runs every 30min to sync training status from QCC
  4. PRE-COMPACTION: Stop hook triggers this script before context dies

This script ONLY touches the "## CURRENT TRAINING" section.
It NEVER modifies HARD CONSTRAINTS — those are user-set and sacred.

Usage:
  python3 sync_directives.py                  # Full sync from QCC
  python3 sync_directives.py --status-only    # Just print current status
  python3 sync_directives.py --update-field "key" "value"  # Update a specific field
"""

import json
import os
import re
import sqlite3
import sys
import urllib.request
import urllib.error
from datetime import datetime

DIRECTIVES_PATH = "/home/jupiter/Lvl3Quant/DIRECTIVES.md"
QCC_DB_PATHS = [
    "/home/jupiter/teleclaude-main/data/qcc.db",
    "/home/jupiter/teleclaude/data/qcc.db",
    "/home/jupiter/Lvl3Quant/qcc.db",
]
QCC_BASE = "http://localhost:3456"
BACKUP_DIR = "/home/jupiter/Lvl3Quant/data/directive_backups"

# ─── Helpers ───────────────────────────────────────────────────────────────────

def now_et():
    """Current time string in ET."""
    return datetime.now().strftime("%Y-%m-%d %H:%M ET")

def qcc_get(endpoint, timeout=10):
    """GET from QCC daemon."""
    try:
        req = urllib.request.Request(f"{QCC_BASE}{endpoint}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"⚠ QCC request failed ({endpoint}): {e}", file=sys.stderr)
        return None

def read_directives():
    """Read current DIRECTIVES.md."""
    with open(DIRECTIVES_PATH, "r") as f:
        return f.read()

def write_directives(content, reason="auto-sync"):
    """Write DIRECTIVES.md with backup."""
    # Ensure backup dir exists
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Backup current version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"DIRECTIVES_{timestamp}.md")

    try:
        current = read_directives()
        with open(backup_path, "w") as f:
            f.write(current)
    except FileNotFoundError:
        pass  # First run, no backup needed

    # Clean old backups (keep last 20)
    try:
        backups = sorted([
            os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR)
            if f.startswith("DIRECTIVES_") and f.endswith(".md")
        ])
        for old in backups[:-20]:
            os.remove(old)
    except Exception:
        pass

    # Update the "Last updated" line
    content = re.sub(
        r"# Last updated:.*",
        f"# Last updated: {now_et()} (auto-sync: {reason})",
        content
    )

    with open(DIRECTIVES_PATH, "w") as f:
        f.write(content)

    print(f"✅ DIRECTIVES.md updated ({reason}), backup: {backup_path}")

# ─── Node Status Fetcher ──────────────────────────────────────────────────────

def find_qcc_db():
    """Find the QCC SQLite database."""
    for path in QCC_DB_PATHS:
        if os.path.exists(path):
            return path
    return None

def get_node_training_status():
    """Query QCC SQLite DB directly for GPU status on all nodes.
    Falls back to HTTP API if DB not found."""
    nodes_info = {}

    # Try SQLite first (works from cron, no daemon needed)
    db_path = find_qcc_db()
    if db_path:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all nodes
            cursor.execute("SELECT * FROM compute_nodes")
            for row in cursor.fetchall():
                name = row["name"].lower()
                nodes_info[name] = {
                    "status": row["status"] or "unknown",
                    "gpu_util": row["last_gpu_util"] or 0,
                    "gpu_mem_mb": row["last_gpu_mem_mb"] or 0,
                    "gpu_power_w": row["last_gpu_power_w"] or 0,
                    "last_heartbeat": row["last_heartbeat"] or "unknown",
                }

            # Get running jobs
            try:
                cursor.execute("SELECT * FROM training_jobs WHERE status='running'")
                for job in cursor.fetchall():
                    node = (job["node"] or "").lower()
                    if node in nodes_info:
                        nodes_info[node]["job"] = {
                            "name": job["description"] if "description" in job.keys() else "unknown",
                            "pid": job["pid"] if "pid" in job.keys() else None,
                            "started": job["started_at"] if "started_at" in job.keys() else None,
                        }
            except sqlite3.OperationalError:
                pass  # jobs table might not exist

            conn.close()
            print(f"✅ Read node status from QCC DB: {db_path}")
            return nodes_info

        except Exception as e:
            print(f"⚠ SQLite read failed: {e}", file=sys.stderr)

    # Fallback to HTTP API
    print("ℹ Falling back to QCC HTTP API...")
    for node_name in ["neptune", "razer", "jupiter", "saturn"]:
        data = qcc_get(f"/api/nodes/{node_name}")
        if data and "node" in data:
            node = data["node"]
            nodes_info[node_name] = {
                "status": node.get("status", "unknown"),
                "gpu_util": node.get("last_gpu_util", 0),
                "gpu_mem_mb": node.get("last_gpu_mem_mb", 0),
                "gpu_power_w": node.get("last_gpu_power_w", 0),
                "last_heartbeat": node.get("last_heartbeat", "unknown"),
            }
        else:
            nodes_info[node_name] = {"status": "unreachable"}

    # Also check QCC jobs via API
    jobs = qcc_get("/api/jobs?status=running")
    if jobs and "jobs" in jobs:
        for job in jobs["jobs"]:
            node = job.get("node", "").lower()
            if node in nodes_info:
                nodes_info[node]["job"] = {
                    "name": job.get("name", "unknown"),
                    "pid": job.get("pid"),
                    "started": job.get("started_at"),
                }

    return nodes_info

def format_training_line(node_name, info):
    """Format a single training status line."""
    status = info.get("status", "unknown")

    if status == "unreachable":
        return f"- {node_name.capitalize()}: ⚠ UNREACHABLE (QCC can't reach node)"

    gpu_util = info.get("gpu_util", 0) or 0
    gpu_mem = info.get("gpu_mem_mb", 0) or 0

    job = info.get("job")
    if job:
        job_name = job.get("name", "unknown")
        pid = job.get("pid", "?")
        return f"- {node_name.capitalize()}: {job_name} (PID {pid}), GPU {gpu_util}%, {gpu_mem}MB VRAM"
    elif gpu_util > 50:
        return f"- {node_name.capitalize()}: TRAINING (unregistered job), GPU {gpu_util}%, {gpu_mem}MB VRAM"
    elif gpu_util > 5:
        return f"- {node_name.capitalize()}: Low GPU activity ({gpu_util}%), may be idle or finishing"
    else:
        if node_name in ("jupiter", "saturn"):
            return f"- {node_name.capitalize()}: CPU node, status: {status}"
        return f"- {node_name.capitalize()}: **IDLE** — GPU {gpu_util}%, needs experiment dispatched"

# ─── Section Updaters ──────────────────────────────────────────────────────────

def update_training_section(content, nodes_info):
    """Replace the ## CURRENT TRAINING section with live data."""
    lines = []
    for node_name in ["neptune", "razer", "jupiter", "saturn"]:
        if node_name in nodes_info:
            lines.append(format_training_line(node_name, nodes_info[node_name]))

    new_section = "## CURRENT TRAINING (auto-synced)\n" + "\n".join(lines)

    # Replace existing section (everything between ## CURRENT TRAINING and the next ##)
    pattern = r"## CURRENT TRAINING.*?\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, content, re.DOTALL)

    if match:
        content = content[:match.start()] + new_section + "\n" + content[match.end():]
    else:
        # Section doesn't exist yet, add before KNOWN PROBLEMS
        content = content.replace("## KNOWN PROBLEMS", new_section + "\n\n## KNOWN PROBLEMS")

    return content

def append_changelog(content, entry):
    """Append an entry to the CHANGE LOG section."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_entry = f"- {timestamp}: {entry}"

    # Find CHANGE LOG section and insert after the header
    pattern = r"(## CHANGE LOG\n)"
    match = re.search(pattern, content)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + new_entry + "\n" + content[insert_pos:]

    return content

# ─── Main ──────────────────────────────────────────────────────────────────────

def sync_training_status():
    """Main sync: pull live node status and update DIRECTIVES.md."""
    print(f"🔄 Syncing DIRECTIVES.md with live cluster state... ({now_et()})")

    nodes_info = get_node_training_status()

    if not any(n.get("status") != "unreachable" for n in nodes_info.values()):
        print("⚠ All nodes unreachable. QCC daemon may be down. Skipping sync.")
        return False

    content = read_directives()
    original = content

    content = update_training_section(content, nodes_info)

    if content != original:
        write_directives(content, reason="training-status-sync")
        return True
    else:
        print("ℹ No changes needed — DIRECTIVES.md already up to date.")
        return False

def print_status():
    """Print current node status without modifying anything."""
    nodes_info = get_node_training_status()
    for node_name, info in nodes_info.items():
        print(format_training_line(node_name, info))

def update_field(key, value):
    """Update a specific directive field. Used by Claude's hook to log user instructions."""
    content = read_directives()

    # Generic key-value update in HARD CONSTRAINTS
    # Format: key = description from user
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_line = f"- **{key}** — {value} ({timestamp})"

    # Check if this key already exists
    pattern = rf"- \*\*{re.escape(key)}\*\*.*"
    match = re.search(pattern, content)

    if match:
        content = content[:match.start()] + new_line + content[match.end():]
    else:
        # Add to Training Rules section
        pattern = r"(### Training Rules\s*\n(?:.*\n)*?)((?=\n###|\n##))"
        match = re.search(pattern, content)
        if match:
            content = content[:match.end(1)] + new_line + "\n" + content[match.end(1):]

    content = append_changelog(content, f"Auto-update: {key} = {value}")
    write_directives(content, reason=f"field-update:{key}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status-only":
        print_status()
    elif len(sys.argv) > 2 and sys.argv[1] == "--update-field":
        if len(sys.argv) >= 4:
            update_field(sys.argv[2], sys.argv[3])
        else:
            print("Usage: sync_directives.py --update-field KEY VALUE")
            sys.exit(1)
    else:
        # Try full infra sync first, fallback to directives-only
        import subprocess
        infra_sync = "/home/jupiter/Lvl3Quant/scripts/infra_sync.py"
        if os.path.exists(infra_sync):
            result = subprocess.run(
                [sys.executable, infra_sync, "--directives"],
                capture_output=True, timeout=15
            )
            if result.returncode == 0:
                print(result.stdout.decode())
                sys.exit(0)
        # Fallback
        sync_training_status()
