#!/usr/bin/env python3
"""
sync_training_results.py

Syncs walk-forward training results (checkpoints + predictions) between
Neptune (local) and Uranus (remote), merges them, and optionally triggers
fill_sim on Saturn.

Usage:
    python sync_training_results.py                # sync and report
    python sync_training_results.py --run-fillsim  # sync + trigger fill_sim
    python sync_training_results.py --loop 10      # run every 10 minutes
"""

import argparse
import fnmatch
import io
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import paramiko
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEPTUNE = {
    "checkpoints_dir": r"C:\Users\Footb\Documents\Github\Lvl3Quant\alpha_discovery\deep_models\checkpoints",
    "predictions_path": r"C:\Users\Footb\Documents\Github\Lvl3Quant\alpha_discovery\deep_models\results\oot_wf_predictions_incremental.npz",
    "log_path": r"C:\Users\Footb\Documents\Github\Lvl3Quant\alpha_discovery\deep_models\results\walkforward_oot_lean_20260311_233837.log",
}

URANUS = {
    "host": "winnode",
    "port": 22,
    "user": "nick",
    "password": os.environ.get("CLUSTER_SSH_PASSWORD", ""),
    "checkpoints_dir": r"C:\Users\Nick\Documents\Lvl3Quant\alpha_discovery\deep_models\checkpoints",
    "predictions_path": r"C:\Users\Nick\Documents\Lvl3Quant\alpha_discovery\deep_models\results\oot_wf_predictions_incremental.npz",
    "log_path": r"C:\Users\Nick\Documents\Lvl3Quant\alpha_discovery\results\wf_uranus_21_44.log",
}

JUPITER = {
    "host": "jupiter",
    "port": 22,
    "user": "jupiter",
    "password": os.environ.get("CLUSTER_SSH_PASSWORD", ""),
}

SATURN = {
    "host": "saturn",
    "port": 22,
    "user": "saturn",
    "password": os.environ.get("CLUSTER_SSH_PASSWORD", ""),
    "predictions_path": "/home/saturn/Lvl3Quant/data/processed/oot_wf_predictions_incremental.npz",
    "fill_sim_script": "/home/saturn/Lvl3Quant/alpha_discovery/run_fill_sim_oot.sh",
}

CHECKPOINT_PATTERN = re.compile(r"cnn_wf_fold(\d+)_(\d{4}-\d{2}-\d{2})\.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def compute_ic(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rank correlation between predictions and labels."""
    if len(predictions) < 3:
        return float("nan")
    try:
        corr, _ = spearmanr(predictions, labels)
        return float(corr)
    except Exception:
        return float("nan")


def parse_ics_from_log(log_path: str) -> dict:
    """Parse IC values from a training log. Returns {date_str: ic_float}."""
    ics = {}
    fold_date = None
    ic_pattern = re.compile(r"IC=([\d.\-]+)")
    date_pattern = re.compile(r"Fold \d+/\d+.*?(\d{4}-\d{2}-\d{2})")
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                dm = date_pattern.search(line)
                if dm:
                    fold_date = dm.group(1)
                if "IC=" in line and "morning" not in line and fold_date:
                    im = ic_pattern.search(line)
                    if im:
                        ics[fold_date] = float(im.group(1))
    except Exception:
        pass
    return ics


def parse_ics_from_remote_log(client: paramiko.SSHClient, log_path: str) -> dict:
    """Parse IC values from a remote training log via SSH."""
    try:
        cmd = f'type "{log_path}"' if "\\" in log_path else f'cat "{log_path}"'
        out, _, _ = ssh_run(client, cmd, timeout=30)
        ics = {}
        fold_date = None
        ic_pattern = re.compile(r"IC=([\d.\-]+)")
        date_pattern = re.compile(r"Fold \d+/\d+.*?(\d{4}-\d{2}-\d{2})")
        for line in out.split("\n"):
            dm = date_pattern.search(line)
            if dm:
                fold_date = dm.group(1)
            if "IC=" in line and "morning" not in line and fold_date:
                im = ic_pattern.search(line)
                if im:
                    ics[fold_date] = float(im.group(1))
        return ics
    except Exception:
        return {}


def parse_npz_dates(npz_data) -> dict:
    """
    Parse an .npz file into a dict keyed by date string.
    Each entry has 'predictions', 'labels', and optionally 'raw_preds'.
    Returns: { '2025-12-01': {'predictions': arr, 'labels': arr, ...}, ... }
    """
    dates = {}
    for key in npz_data.files:
        # Keys look like '2025-12-01_predictions', '2025-12-01_labels', etc.
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            date_str, field = parts
            # Validate date format
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if date_str not in dates:
                dates[date_str] = {}
            dates[date_str][field] = npz_data[key]
    return dates


def parse_checkpoint_filename(filename: str):
    """Returns (fold_num, date_str) or None."""
    m = CHECKPOINT_PATTERN.match(os.path.basename(filename))
    if m:
        return int(m.group(1)), m.group(2)
    return None


def print_separator(char="-", width=80):
    print(char * width)


# ---------------------------------------------------------------------------
# SSH / SFTP helpers
# ---------------------------------------------------------------------------

def connect_uranus() -> paramiko.SSHClient:
    """Connect directly to Uranus using SSH key (password auth disabled)."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    log(f"Connecting to Uranus ({URANUS['host']})...")
    key_path = os.path.expanduser("~/.ssh/id_ed25519")
    try:
        pkey = paramiko.Ed25519Key.from_private_key_file(key_path)
        client.connect(
            URANUS["host"],
            port=URANUS["port"],
            username=URANUS["user"],
            pkey=pkey,
            timeout=30,
            look_for_keys=False,
            allow_agent=False,
        )
    except Exception:
        # Fallback to password if key fails
        client.connect(
            URANUS["host"],
            port=URANUS["port"],
            username=URANUS["user"],
            password=URANUS["password"],
            timeout=30,
            look_for_keys=False,
            allow_agent=False,
        )
    log("Connected to Uranus.")
    return client


def connect_saturn_via_jupiter() -> tuple:
    """
    Connect to Saturn via Jupiter as a hop.
    Returns (jupiter_client, saturn_client).
    Caller must close both when done.
    """
    # Connect to Jupiter first
    jupiter_client = paramiko.SSHClient()
    jupiter_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    log(f"Connecting to Jupiter ({JUPITER['host']}) as hop...")
    jupiter_client.connect(
        JUPITER["host"],
        port=JUPITER["port"],
        username=JUPITER["user"],
        password=JUPITER["password"],
        timeout=30,
        look_for_keys=False,
        allow_agent=False,
    )
    log("Connected to Jupiter.")

    # Open a channel through Jupiter to Saturn
    transport = jupiter_client.get_transport()
    dest_addr = (SATURN["host"], SATURN["port"])
    src_addr = (JUPITER["host"], 0)
    channel = transport.open_channel("direct-tcpip", dest_addr, src_addr, timeout=30)

    # Connect to Saturn through that channel
    saturn_client = paramiko.SSHClient()
    saturn_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    log(f"Connecting to Saturn ({SATURN['host']}) via Jupiter hop...")
    saturn_client.connect(
        SATURN["host"],
        port=SATURN["port"],
        username=SATURN["user"],
        password=SATURN["password"],
        sock=channel,
        timeout=30,
        look_for_keys=False,
        allow_agent=False,
    )
    log("Connected to Saturn.")
    return jupiter_client, saturn_client


def ssh_run(client: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple:
    """Run a command on an SSH client. Returns (stdout, stderr, exit_code)."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return out, err, code


def sftp_list_files(sftp, remote_dir: str, pattern: str = "*.pt") -> list:
    """List files in a remote directory matching a glob pattern."""
    try:
        entries = sftp.listdir(remote_dir)
    except FileNotFoundError:
        return []
    return [e for e in entries if fnmatch.fnmatch(e, pattern)]


def sftp_download_to_bytes(sftp, remote_path: str) -> bytes:
    """Download a remote file and return its contents as bytes."""
    buf = io.BytesIO()
    sftp.getfo(remote_path, buf)
    buf.seek(0)
    return buf.read()


def sftp_upload_from_bytes(sftp, data: bytes, remote_path: str):
    """Upload bytes to a remote path, creating directories as needed."""
    remote_dir = "/".join(remote_path.replace("\\", "/").rsplit("/", 1)[:-1])
    try:
        sftp.makedirs(remote_dir)
    except Exception:
        pass
    buf = io.BytesIO(data)
    sftp.putfo(buf, remote_path)


# Paramiko's SFTP doesn't have makedirs; add it
def sftp_makedirs(sftp, remote_dir: str):
    """Recursively create remote directories (like mkdir -p)."""
    dirs = []
    path = remote_dir
    while True:
        try:
            sftp.stat(path)
            break
        except FileNotFoundError:
            dirs.append(path)
            parent = "/".join(path.replace("\\", "/").rsplit("/", 1)[:-1])
            if parent == path or not parent:
                break
            path = parent
    for d in reversed(dirs):
        try:
            sftp.mkdir(d)
        except Exception:
            pass


# Monkey-patch sftp.makedirs
paramiko.SFTPClient.makedirs = lambda self, path: sftp_makedirs(self, path)


# ---------------------------------------------------------------------------
# Step 1: Discover completed folds
# ---------------------------------------------------------------------------

def discover_neptune_folds() -> dict:
    """Parse Neptune's local .npz to find completed fold dates."""
    path = NEPTUNE["predictions_path"]
    if not os.path.exists(path):
        log(f"Neptune predictions file not found: {path}")
        return {}
    try:
        data = np.load(path, allow_pickle=True)
        dates = parse_npz_dates(data)
        log(f"Neptune has {len(dates)} fold dates in predictions file.")
        return dates
    except Exception as e:
        log(f"Error reading Neptune predictions: {e}")
        return {}


def discover_neptune_checkpoints() -> dict:
    """List checkpoint files on Neptune. Returns {fold_num: filename}."""
    ckpt_dir = NEPTUNE["checkpoints_dir"]
    if not os.path.exists(ckpt_dir):
        return {}
    result = {}
    for fname in os.listdir(ckpt_dir):
        parsed = parse_checkpoint_filename(fname)
        if parsed:
            fold_num, date_str = parsed
            result[fold_num] = fname
    return result


def discover_uranus_folds(sftp: paramiko.SFTPClient) -> dict:
    """Download and parse Uranus's .npz to find completed fold dates."""
    remote_path = URANUS["predictions_path"].replace("\\", "/")
    try:
        raw = sftp_download_to_bytes(sftp, remote_path)
        data = np.load(io.BytesIO(raw), allow_pickle=True)
        dates = parse_npz_dates(data)
        log(f"Uranus has {len(dates)} fold dates in predictions file.")
        return dates, raw
    except FileNotFoundError:
        log(f"Uranus predictions file not found: {remote_path}")
        return {}, None
    except Exception as e:
        log(f"Error reading Uranus predictions: {e}")
        return {}, None


def discover_uranus_checkpoints(sftp: paramiko.SFTPClient) -> dict:
    """List checkpoint files on Uranus. Returns {fold_num: filename}."""
    remote_dir = URANUS["checkpoints_dir"].replace("\\", "/")
    try:
        files = sftp_list_files(sftp, remote_dir, "cnn_wf_fold*.pt")
    except Exception as e:
        log(f"Error listing Uranus checkpoints: {e}")
        return {}
    result = {}
    for fname in files:
        parsed = parse_checkpoint_filename(fname)
        if parsed:
            fold_num, date_str = parsed
            result[fold_num] = fname
    return result


# ---------------------------------------------------------------------------
# Step 2: Sync checkpoints
# ---------------------------------------------------------------------------

def sync_checkpoints(sftp: paramiko.SFTPClient, neptune_ckpts: dict, uranus_ckpts: dict):
    """
    Sync checkpoints between Neptune and Uranus.
    Download Uranus-only checkpoints to Neptune.
    Upload Neptune-only checkpoints to Uranus.
    """
    neptune_folds = set(neptune_ckpts.keys())
    uranus_folds = set(uranus_ckpts.keys())

    to_download = uranus_folds - neptune_folds  # On Uranus, not Neptune
    to_upload = neptune_folds - uranus_folds    # On Neptune, not Uranus

    if not to_download and not to_upload:
        log("Checkpoints are in sync — no transfers needed.")
        return

    neptune_dir = Path(NEPTUNE["checkpoints_dir"])
    uranus_dir = URANUS["checkpoints_dir"].replace("\\", "/")

    # Download from Uranus → Neptune
    for fold_num in sorted(to_download):
        fname = uranus_ckpts[fold_num]
        remote_path = f"{uranus_dir}/{fname}"
        local_path = neptune_dir / fname
        log(f"Downloading checkpoint fold {fold_num}: {fname} from Uranus...")
        try:
            sftp.get(remote_path, str(local_path))
            log(f"  Downloaded {fname} ({local_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            log(f"  ERROR downloading {fname}: {e}")

    # Upload from Neptune → Uranus
    for fold_num in sorted(to_upload):
        fname = neptune_ckpts[fold_num]
        local_path = neptune_dir / fname
        remote_path = f"{uranus_dir}/{fname}"
        size_mb = local_path.stat().st_size / 1e6
        log(f"Uploading checkpoint fold {fold_num}: {fname} to Uranus ({size_mb:.1f} MB)...")
        try:
            sftp.put(str(local_path), remote_path)
            log(f"  Uploaded {fname}")
        except Exception as e:
            log(f"  ERROR uploading {fname}: {e}")


# ---------------------------------------------------------------------------
# Step 3: Merge predictions
# ---------------------------------------------------------------------------

def merge_predictions(neptune_dates: dict, uranus_dates: dict) -> tuple:
    """
    Merge fold predictions from both machines.
    For dates on both, keep the one with higher IC.
    Returns (merged_dict, merge_report) where merge_report lists what was kept.
    """
    all_dates = sorted(set(neptune_dates.keys()) | set(uranus_dates.keys()))
    merged = {}
    report = []  # list of (date, source, ic)

    for date_str in all_dates:
        in_neptune = date_str in neptune_dates
        in_uranus = date_str in uranus_dates

        if in_neptune and not in_uranus:
            entry = neptune_dates[date_str]
            ic = compute_ic(entry.get("preds", entry.get("predictions", np.array([]))),
                            entry.get("mid", entry.get("labels", np.array([]))))
            merged[date_str] = entry
            report.append((date_str, "Neptune", ic))

        elif in_uranus and not in_neptune:
            entry = uranus_dates[date_str]
            ic = compute_ic(entry.get("preds", entry.get("predictions", np.array([]))),
                            entry.get("mid", entry.get("labels", np.array([]))))
            merged[date_str] = entry
            report.append((date_str, "Uranus", ic))

        else:
            # Both have it — pick higher IC
            n_entry = neptune_dates[date_str]
            u_entry = uranus_dates[date_str]
            n_ic = compute_ic(n_entry.get("preds", n_entry.get("predictions", np.array([]))),
                              n_entry.get("mid", n_entry.get("labels", np.array([]))))
            u_ic = compute_ic(u_entry.get("preds", u_entry.get("predictions", np.array([]))),
                              u_entry.get("mid", u_entry.get("labels", np.array([]))))

            if not np.isnan(u_ic) and (np.isnan(n_ic) or u_ic > n_ic):
                merged[date_str] = u_entry
                report.append((date_str, f"Uranus (IC={u_ic:.4f} > Neptune IC={n_ic:.4f})", u_ic))
            else:
                merged[date_str] = n_entry
                report.append((date_str, f"Neptune (IC={n_ic:.4f})", n_ic))

    return merged, report


def save_npz(merged: dict, output_path: str):
    """Save merged fold data back to an .npz file."""
    arrays = {}
    for date_str, entry in merged.items():
        for field, arr in entry.items():
            arrays[f"{date_str}_{field}"] = arr
    np.savez(output_path, **arrays)
    size_mb = os.path.getsize(output_path) / 1e6
    log(f"Saved merged predictions to {output_path} ({size_mb:.1f} MB, {len(merged)} folds)")


def upload_merged_to_uranus(sftp: paramiko.SFTPClient, local_path: str):
    """Upload the merged .npz file back to Uranus."""
    remote_path = URANUS["predictions_path"].replace("\\", "/")
    size_mb = os.path.getsize(local_path) / 1e6
    log(f"Uploading merged predictions to Uranus ({size_mb:.1f} MB)...")
    try:
        sftp.put(local_path, remote_path)
        log("  Uploaded merged predictions to Uranus.")
    except Exception as e:
        log(f"  ERROR uploading merged predictions to Uranus: {e}")


# ---------------------------------------------------------------------------
# Step 4: Report status
# ---------------------------------------------------------------------------

def infer_fold_number(date_str: str, all_dates: list) -> int:
    """Infer fold number from sorted date list (1-indexed)."""
    try:
        return sorted(all_dates).index(date_str) + 1
    except ValueError:
        return -1


def print_status_report(merge_report: list, neptune_ckpts: dict, uranus_ckpts: dict):
    """Print a formatted table of all folds and summary statistics."""
    print_separator("=")
    print("WALK-FORWARD TRAINING SYNC REPORT")
    print_separator("=")

    if not merge_report:
        print("No folds found on either machine.")
        return

    all_dates = [r[0] for r in merge_report]

    # Header
    print(f"{'Fold':>5}  {'Date':<12}  {'Source':<55}  {'IC':>8}")
    print_separator()

    valid_ics = []
    for i, (date_str, source, ic) in enumerate(merge_report):
        fold_num = i + 1
        ic_str = f"{ic:.4f}" if not np.isnan(ic) else "  N/A"
        print(f"{fold_num:>5}  {date_str:<12}  {source:<55}  {ic_str:>8}")
        if not np.isnan(ic):
            valid_ics.append(ic)

    print_separator()

    # Summary
    total_folds = len(merge_report)
    mean_ic = np.mean(valid_ics) if valid_ics else float("nan")
    pct_positive = (np.sum(np.array(valid_ics) > 0) / len(valid_ics) * 100) if valid_ics else float("nan")

    print(f"Total folds completed: {total_folds}")
    print(f"Mean IC: {mean_ic:.4f}  |  % Positive: {pct_positive:.1f}%")

    # Checkpoint coverage
    all_fold_nums = set(neptune_ckpts.keys()) | set(uranus_ckpts.keys())
    n_ckpts = len(all_fold_nums)
    print(f"Checkpoints available: {n_ckpts} folds (Neptune: {len(neptune_ckpts)}, Uranus: {len(uranus_ckpts)})")

    # Missing folds detection
    if total_folds > 0:
        expected_range = range(1, total_folds + 2)
        # Gaps are any fold nums not covered by predictions
        covered = set(range(1, total_folds + 1))
        # We can't know truly missing without a total target, but flag checkpoint gaps
        pred_folds = set(range(1, total_folds + 1))
        ckpt_only = all_fold_nums - pred_folds
        pred_only = pred_folds - all_fold_nums
        if ckpt_only:
            print(f"Checkpoints without predictions: folds {sorted(ckpt_only)}")
        if pred_only:
            print(f"Predictions without checkpoints: folds {sorted(pred_only)}")

    print_separator("=")


# ---------------------------------------------------------------------------
# Step 5: Trigger fill_sim on Saturn
# ---------------------------------------------------------------------------

def trigger_fill_sim_saturn(merged_npz_path: str, new_folds_added: bool):
    """Upload merged predictions to Saturn and trigger fill_sim."""
    if not new_folds_added:
        log("No new folds added — skipping fill_sim trigger.")
        return

    log("Connecting to Saturn via Jupiter for fill_sim...")
    jupiter_client = None
    saturn_client = None
    try:
        jupiter_client, saturn_client = connect_saturn_via_jupiter()
        sftp = saturn_client.open_sftp()

        # Upload merged predictions
        size_mb = os.path.getsize(merged_npz_path) / 1e6
        log(f"Uploading merged predictions to Saturn ({size_mb:.1f} MB)...")
        sftp_makedirs(sftp, "/".join(SATURN["predictions_path"].rsplit("/", 1)[:-1]))
        sftp.put(merged_npz_path, SATURN["predictions_path"])
        log("Uploaded predictions to Saturn.")

        # Check if fill_sim script exists
        out, err, code = ssh_run(saturn_client, f"ls {SATURN['fill_sim_script']}")
        if code != 0:
            log(f"fill_sim script not found at {SATURN['fill_sim_script']}. Skipping.")
            log(f"  ls error: {err.strip()}")
        else:
            # Launch fill_sim in background (detached from SSH session)
            cmd = (
                f"nohup bash {SATURN['fill_sim_script']} "
                f"--predictions {SATURN['predictions_path']} "
                f"> /tmp/fill_sim_sync_run.log 2>&1 &"
            )
            out, err, code = ssh_run(saturn_client, cmd)
            if code == 0:
                log("fill_sim launched on Saturn (background). Log: /tmp/fill_sim_sync_run.log")
            else:
                log(f"Error launching fill_sim: {err.strip()}")

        sftp.close()
    except Exception as e:
        log(f"ERROR connecting to Saturn or running fill_sim: {e}")
        traceback.print_exc()
    finally:
        if saturn_client:
            saturn_client.close()
        if jupiter_client:
            jupiter_client.close()


# ---------------------------------------------------------------------------
# Main sync routine
# ---------------------------------------------------------------------------

def run_sync(run_fillsim: bool = False) -> int:
    """
    Full sync cycle. Returns number of new folds added to Neptune.
    """
    log("=" * 60)
    log("Starting sync cycle...")
    log("=" * 60)

    # --- Neptune local state ---
    neptune_dates = discover_neptune_folds()
    neptune_ckpts = discover_neptune_checkpoints()
    log(f"Neptune: {len(neptune_dates)} prediction folds, {len(neptune_ckpts)} checkpoints")

    # --- Connect to Uranus ---
    uranus_client = None
    uranus_dates = {}
    uranus_raw_npz = None
    uranus_ckpts = {}

    try:
        uranus_client = connect_uranus()
        sftp = uranus_client.open_sftp()

        uranus_dates, uranus_raw_npz = discover_uranus_folds(sftp)
        uranus_ckpts = discover_uranus_checkpoints(sftp)
        log(f"Uranus: {len(uranus_dates)} prediction folds, {len(uranus_ckpts)} checkpoints")

        # --- Step 2: Sync checkpoints ---
        log("")
        log("--- Step 2: Syncing checkpoints ---")
        sync_checkpoints(sftp, neptune_ckpts, uranus_ckpts)

        # --- Step 3: Merge predictions ---
        log("")
        log("--- Step 3: Merging predictions ---")
        neptune_fold_count_before = len(neptune_dates)
        merged, merge_report = merge_predictions(neptune_dates, uranus_dates)
        new_folds_added = len(merged) - neptune_fold_count_before

        if new_folds_added > 0:
            log(f"Merged: {new_folds_added} new folds added from Uranus.")
        else:
            log("No new folds to merge from Uranus.")

        # Save merged file back to Neptune
        neptune_pred_path = NEPTUNE["predictions_path"]
        save_npz(merged, neptune_pred_path)

        # Upload merged back to Uranus
        upload_merged_to_uranus(sftp, neptune_pred_path)

        sftp.close()

    except Exception as e:
        log(f"ERROR during Uranus operations: {e}")
        traceback.print_exc()
        # Still report what we have locally
        merged, merge_report = merge_predictions(neptune_dates, {})
        new_folds_added = 0
    finally:
        if uranus_client:
            uranus_client.close()
            log("Disconnected from Uranus.")

    # --- Step 4: Parse ICs from training logs and report ---
    log("")
    log("--- Step 4: Parsing ICs from training logs ---")
    log_ics = parse_ics_from_log(NEPTUNE["log_path"])
    log(f"  Neptune log ICs: {len(log_ics)} folds")
    if uranus_client is not None:
        try:
            uc = connect_uranus()
            uranus_log_ics = parse_ics_from_remote_log(uc, URANUS["log_path"])
            uc.close()
            log(f"  Uranus log ICs: {len(uranus_log_ics)} folds")
            log_ics.update({k: v for k, v in uranus_log_ics.items() if k not in log_ics})
        except Exception as e:
            log(f"  Could not parse Uranus log: {e}")
    # Override merge_report ICs with log-based ICs and clean source labels
    updated_report = []
    for date_str, source, _ in merge_report:
        ic = log_ics.get(date_str, float("nan"))
        # Clean source label — remove old wrong IC from display
        clean_source = source.split(" (IC=")[0] if " (IC=" in source else source
        updated_report.append((date_str, clean_source, ic))
    print_status_report(updated_report, neptune_ckpts, uranus_ckpts)

    # --- Step 5: Optional fill_sim ---
    if run_fillsim:
        log("")
        log("--- Step 5: Triggering fill_sim on Saturn ---")
        trigger_fill_sim_saturn(NEPTUNE["predictions_path"], new_folds_added > 0)

    return new_folds_added


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sync walk-forward training results between Neptune and Uranus."
    )
    parser.add_argument(
        "--run-fillsim",
        action="store_true",
        help="After syncing, upload merged predictions to Saturn and trigger fill_sim.",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        metavar="MINUTES",
        help="Run continuously, sleeping MINUTES between each sync cycle.",
    )
    args = parser.parse_args()

    if args.loop > 0:
        log(f"Loop mode: running every {args.loop} minute(s). Ctrl+C to stop.")
        cycle = 0
        while True:
            cycle += 1
            log(f"\n{'='*60}")
            log(f"LOOP CYCLE {cycle} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                run_sync(run_fillsim=args.run_fillsim)
            except KeyboardInterrupt:
                log("Interrupted by user. Exiting.")
                sys.exit(0)
            except Exception as e:
                log(f"Unhandled error in sync cycle: {e}")
                traceback.print_exc()
            log(f"Sleeping {args.loop} minute(s) before next cycle...")
            try:
                time.sleep(args.loop * 60)
            except KeyboardInterrupt:
                log("Interrupted by user. Exiting.")
                sys.exit(0)
    else:
        try:
            run_sync(run_fillsim=args.run_fillsim)
        except KeyboardInterrupt:
            log("Interrupted by user.")
            sys.exit(0)


if __name__ == "__main__":
    main()
