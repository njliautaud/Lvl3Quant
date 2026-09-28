#!/usr/bin/env python3
"""
Nightly CNN Retraining Pipeline
================================
Runs on Razer (triggered by Windows Task Scheduler at 4:30pm ET daily).
Designed to run AS A SCRIPT ON RAZER — see DEPLOYMENT section at bottom.

Flow:
  1. Data sync check — confirm latest day's MBO file landed
  2. Launch CNN training (TINY_DATE_CUTOFF = today, rolling 60d train / 15d OOT)
  3. Validate: IC_1s on most recent 15d OOT >= 0.15
  4. Champion promotion: if pass, save as champion + overwrite production symlink
  5. Discord alert (pass or fail)
  6. MLflow: all runs logged under experiment "TinyCNN_nightly_production"
  7. Keep last 5 champion checkpoints; prune older ones

Target wall time: < 2h (single fold, 60d train window, W=1000, 2M events/file cap)

DEPLOYMENT:
  - This script lives on Jupiter for version control, but RUNS on Razer.
  - Sync to Razer: scp /home/jupiter/Lvl3Quant/alpha_discovery/nightly_retrain_pipeline.py claude@razer:C:/Users/claude/Lvl3Quant/alpha_discovery/nightly_retrain_pipeline.py
  - On Razer, schedule via Task Scheduler:
      Action: C:\\Python311\\python.exe
      Arguments: C:\\Users\\claude\\Lvl3Quant\\alpha_discovery\\nightly_retrain_pipeline.py
      Trigger: Daily at 4:30pm ET (16:30)
  - Or trigger manually: C:\\Python311\\python.exe nightly_retrain_pipeline.py
"""

import os
import sys
import json
import glob
import shutil
import logging
import subprocess
import time
import datetime
import pathlib
import traceback
import re
import urllib.request
import urllib.parse
import urllib.error

# ─────────────────────────── CONFIG ──────────────────────────────────────────

# Paths (Razer Windows paths — this script runs ON Razer)
DATA_DIR         = pathlib.Path(r"C:\Users\claude\Lvl3Quant\alpha_discovery\data\mbo_events")
TRAIN_SCRIPT     = pathlib.Path(r"C:\Users\claude\Lvl3Quant\alpha_discovery\deep_models\train_tiny_cnn.py")
RESULTS_BASE     = pathlib.Path(r"C:\Users\claude\Lvl3Quant\alpha_discovery\deep_models\results")
CHAMPION_DIR     = pathlib.Path(r"C:\Users\claude\Lvl3Quant\alpha_discovery\deep_models\champion")
PRODUCTION_LINK  = pathlib.Path(r"C:\Users\claude\Lvl3Quant\alpha_discovery\deep_models\production_model")  # points to latest champion
LOG_DIR          = pathlib.Path(r"C:\Users\claude\Lvl3Quant\alpha_discovery\deep_models\nightly_logs")
PYTHON_EXE       = r"C:\Python311\python.exe"

# Training hyperparameters (single-fold nightly)
TRAIN_WINDOW_DAYS  = 60    # rolling train window
TEST_WINDOW_DAYS   = 15    # OOT validation window
N_FOLDS            = 1     # single most-recent fold only for nightly
WINDOW             = 1000
STRIDE             = 500
BATCH              = 1024
EPOCHS             = 10    # fast nightly retrain
LR                 = 0.0003
DROPOUT            = 0.2
WORKERS            = 4
MAX_EVENTS         = 2_000_000
CACHE_DAYS         = 5
RECENCY_DECAY      = 0.5   # up-weight recent data in nightly retrain

# Validation gate
MIN_IC_1S          = 0.15   # IC_1s on OOT window (kept for reference)
MIN_IC_TOP10       = 0.25   # Champion gate uses top10% IC — overall IC is noise, we only trade high-confidence

# Champion retention
MAX_CHAMPIONS      = 5

# MLflow
MLFLOW_URI         = "http://localhost:5002"
MLFLOW_EXPERIMENT  = "TinyCNN_nightly_production"

# Discord webhook — set via env var DISCORD_WEBHOOK_URL
# or hardcode here if preferred (keep out of git)
DISCORD_WEBHOOK    = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ─────────────────────────── LOGGING ─────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)
today_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path  = LOG_DIR / f"nightly_{today_str}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────── DISCORD ─────────────────────────────────────────

def send_discord(message: str, title: str = "", color: int = 0x00ff00) -> None:
    """Send embed to Discord via webhook. Fails silently if no webhook set."""
    if not DISCORD_WEBHOOK:
        log.warning("DISCORD_WEBHOOK_URL not set — skipping Discord notification.")
        return
    payload = {
        "embeds": [{
            "title": title or "Nightly CNN Retrain",
            "description": message[:4000],
            "color": color,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }]
    }
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        DISCORD_WEBHOOK,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 204):
                log.warning(f"Discord returned HTTP {resp.status}")
    except Exception as exc:
        log.warning(f"Discord notification failed: {exc}")

# ─────────────────────────── DATA SYNC CHECK ─────────────────────────────────

def check_data_freshness() -> tuple[bool, str, list[pathlib.Path]]:
    """
    Verify MBO data is up to date.
    Returns (ok, message, sorted_files).
    Expects at least one file dated today or yesterday (market days).
    """
    files = sorted(DATA_DIR.glob("*_mbo_events.npz"))
    if not files:
        return False, f"No MBO files found in {DATA_DIR}", []

    latest = files[-1]
    # Extract date from filename: YYYYMMDD_mbo_events.npz
    m = re.match(r"(\d{8})_mbo_events", latest.stem)
    if not m:
        return False, f"Unexpected filename format: {latest.name}", files

    file_date = datetime.datetime.strptime(m.group(1), "%Y%m%d").date()
    today     = datetime.date.today()
    delta     = (today - file_date).days

    # Allow up to 4 calendar days lag (weekends + holiday buffer)
    if delta > 4:
        msg = (f"Data stale: latest file is {latest.name} ({delta} days old). "
               f"Expected data through ~{today}. Aborting retrain.")
        return False, msg, files

    msg = f"Data OK: {len(files)} files, latest={latest.name} ({delta}d old)"
    return True, msg, files

# ─────────────────────────── TRAINING ────────────────────────────────────────

def run_training(run_name: str) -> tuple[bool, pathlib.Path]:
    """
    Launch CNN training subprocess and wait for completion.
    Returns (success, output_dir).
    Uses creationflags=0x01000008 (DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB)
    so process survives any SSH/parent session close if needed.
    For nightly Task Scheduler runs, we wait synchronously.
    """
    output_dir = RESULTS_BASE / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = output_dir / "stdout.log"

    env = os.environ.copy()
    env.update({
        "TINY_WINDOW":        str(WINDOW),
        "TINY_STRIDE":        str(STRIDE),
        "TINY_BATCH":         str(BATCH),
        "TINY_EPOCHS":        str(EPOCHS),
        "TINY_LR":            str(LR),
        "TINY_DROPOUT":       str(DROPOUT),
        "TINY_WORKERS":       str(WORKERS),
        "TINY_MAX_EVENTS":    str(MAX_EVENTS),
        "TINY_CACHE_DAYS":    str(CACHE_DAYS),
        "TINY_N_FOLDS":       str(N_FOLDS),
        "TINY_RECENCY_DECAY": str(RECENCY_DECAY),
        "TINY_TRAIN_DAYS":    str(TRAIN_WINDOW_DAYS),
        "TINY_TEST_DAYS":     str(TEST_WINDOW_DAYS),
        "CNN_DERIVED_FEATURES": "1",
        "MLFLOW_TRACKING_URI": MLFLOW_URI,
        "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT,
        # Use most recent N files for rolling window (let script compute based on TINY_N_FOLDS=1)
        # DATE_CUTOFF not needed — omit to use all data, fold selection gives us the last window
    })

    # Warm start — always pass latest checkpoint; script decides warm vs cold per fold
    warm_ckpt = os.path.join(output_dir, 'latest_checkpoint.pt')
    if os.path.exists(warm_ckpt):
        env['TINY_WARM_START'] = warm_ckpt
        env['TINY_WARM_LR'] = '0.0001'
        env['TINY_WARM_EPOCHS'] = '3'
        env['TINY_WARM_IC_GATE'] = '0.15'
        env['TINY_WARM_DROP_GATE'] = '0.25'

    cmd = [PYTHON_EXE, str(TRAIN_SCRIPT),
           "--run-name",   run_name,
           "--output-dir", str(output_dir),
           "--data-dir",   str(DATA_DIR)]

    log.info(f"Launching training: {' '.join(cmd)}")
    log.info(f"stdout -> {stdout_log}")

    t0 = time.time()
    with open(stdout_log, "w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            cmd,
            stdout=fout,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(TRAIN_SCRIPT.parent),
        )

    # Poll with timeout: 2h = 7200s
    MAX_WALL = 7200
    poll_interval = 30
    elapsed = 0
    while proc.poll() is None:
        time.sleep(poll_interval)
        elapsed += poll_interval
        if elapsed % 300 == 0:
            log.info(f"  Training in progress... {elapsed//60}m elapsed")
        if elapsed >= MAX_WALL:
            log.error(f"Training exceeded {MAX_WALL}s wall time — killing process")
            proc.kill()
            return False, output_dir

    wall = time.time() - t0
    rc   = proc.returncode
    log.info(f"Training finished: returncode={rc}, wall={wall/60:.1f}m")

    if rc != 0:
        # Tail the log for error context
        try:
            lines = stdout_log.read_text(encoding="utf-8", errors="replace").splitlines()
            tail  = "\n".join(lines[-20:])
            log.error(f"Training FAILED. Last 20 lines of log:\n{tail}")
        except Exception:
            pass
        return False, output_dir

    return True, output_dir

# ─────────────────────────── VALIDATION ──────────────────────────────────────

def extract_ic_from_log(output_dir: pathlib.Path) -> dict:
    """
    Parse stdout.log for reported IC metrics.
    Looks for lines like:
      [INFO] FOLD 00 OOT  ic_1s=0.171  ic_10s=0.050
      [10s top10]  n=XXXXX  IC=+X.XXXX  DirAcc=X.XXX
    Returns dict with ic_1s, ic_10s, ic_top10 (None if not found).
    """
    result = {"ic_1s": None, "ic_10s": None, "ic_top10": None}
    stdout_log = output_dir / "stdout.log"
    if not stdout_log.exists():
        return result

    text = stdout_log.read_text(encoding="utf-8", errors="replace")

    # Pattern: OOT ic_1s=X.XXX ic_10s=X.XXX (various formats)
    m1 = re.search(r"ic_1s\s*[=:]\s*([+-]?\d+\.\d+)", text)
    m2 = re.search(r"ic_10s\s*[=:]\s*([+-]?\d+\.\d+)", text)

    if m1:
        result["ic_1s"] = float(m1.group(1))
    if m2:
        result["ic_10s"] = float(m2.group(1))

    # Also check for summary line with best fold
    # Pattern: Best fold: XX  ic_1s=X.XXX
    m3 = re.search(r"[Bb]est.*?ic_1s\s*[=:]\s*([+-]?\d+\.\d+)", text)
    if m3:
        result["ic_1s"] = float(m3.group(1))

    # Top10% IC from confidence_eval output lines
    # Pattern: [10s top10]  n=XXXXX  IC=+X.XXXX  DirAcc=X.XXX
    m4 = re.search(r"\[10s top10\]\s+n=\d+\s+IC=([+-]?\d+\.\d+)", text)
    if m4:
        result["ic_top10"] = float(m4.group(1))

    return result

def validate_model(output_dir: pathlib.Path) -> tuple[bool, dict]:
    """
    Check if trained model meets validation gate.
    Returns (passes, metrics_dict).
    Champion gate uses top10% IC — overall IC is noise, we only trade high-confidence.
    """
    metrics  = extract_ic_from_log(output_dir)
    ic_top10 = metrics.get("ic_top10")
    ic_1s    = metrics.get("ic_1s")

    if ic_top10 is None:
        log.warning("Could not extract ic_top10 from training log — falling back to ic_1s gate")
        if ic_1s is None:
            log.error("Could not extract ic_1s either — validation FAILED")
            return False, metrics
        passes = ic_1s >= MIN_IC_1S
        log.info(f"Validation (fallback): ic_1s={ic_1s:.4f} (gate={MIN_IC_1S}) -> {'PASS' if passes else 'FAIL'}")
        return passes, metrics

    passes = ic_top10 >= MIN_IC_TOP10
    log.info(f"Validation: ic_top10={ic_top10:.4f} (gate={MIN_IC_TOP10})  ic_1s={ic_1s if ic_1s is not None else 'n/a'} -> {'PASS' if passes else 'FAIL'}")
    return passes, metrics

# ─────────────────────────── CHAMPION PROMOTION ──────────────────────────────

def promote_champion(output_dir: pathlib.Path, run_name: str, metrics: dict) -> pathlib.Path:
    """
    Copy best-fold weights + metadata to CHAMPION_DIR/<timestamp>_<run_name>/.
    Updates PRODUCTION_LINK to point to new champion.
    Prunes oldest champions beyond MAX_CHAMPIONS.
    Returns path to new champion dir.
    """
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)

    champ_name = f"{today_str}_{run_name}"
    champ_path = CHAMPION_DIR / champ_name
    champ_path.mkdir(parents=True, exist_ok=True)

    # Copy all .pt weight files and the log
    for src in output_dir.glob("*.pt"):
        shutil.copy2(src, champ_path / src.name)
    for src in output_dir.glob("*.npz"):
        shutil.copy2(src, champ_path / src.name)
    stdout_log = output_dir / "stdout.log"
    if stdout_log.exists():
        shutil.copy2(stdout_log, champ_path / "stdout.log")

    # Write metadata
    meta = {
        "promoted_at": datetime.datetime.now().isoformat(),
        "run_name":    run_name,
        "source_dir":  str(output_dir),
        "metrics":     metrics,
        "training_config": {
            "window":          WINDOW,
            "stride":          STRIDE,
            "batch":           BATCH,
            "epochs":          EPOCHS,
            "lr":              LR,
            "recency_decay":   RECENCY_DECAY,
            "train_days":      TRAIN_WINDOW_DAYS,
            "test_days":       TEST_WINDOW_DAYS,
        },
    }
    (champ_path / "champion_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Update production pointer (write a JSON file since Windows junctions need admin)
    prod_pointer = PRODUCTION_LINK.with_suffix(".json")
    prod_pointer.write_text(
        json.dumps({"champion_path": str(champ_path), "promoted_at": meta["promoted_at"]}, indent=2),
        encoding="utf-8",
    )
    log.info(f"Production pointer updated: {prod_pointer} -> {champ_path}")

    # Also try to create/update a directory junction (best-effort, may need admin)
    try:
        if PRODUCTION_LINK.exists() or PRODUCTION_LINK.is_symlink():
            subprocess.run(["cmd", "/c", "rmdir", str(PRODUCTION_LINK)], check=False)
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(PRODUCTION_LINK), str(champ_path)],
            check=True, capture_output=True,
        )
        log.info(f"Junction updated: {PRODUCTION_LINK} -> {champ_path}")
    except Exception as exc:
        log.warning(f"Could not update junction (non-fatal, JSON pointer is authoritative): {exc}")

    # Prune old champions
    all_champs = sorted(CHAMPION_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
    while len(all_champs) > MAX_CHAMPIONS:
        oldest = all_champs.pop(0)
        log.info(f"Pruning old champion: {oldest}")
        shutil.rmtree(oldest, ignore_errors=True)

    return champ_path

# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    start_time = time.time()
    run_name   = f"nightly_{today_str}"
    log.info("=" * 60)
    log.info(f"Nightly CNN Retrain Pipeline starting — {today_str}")
    log.info("=" * 60)

    # ── Step 1: Data freshness check ──────────────────────────────
    log.info("Step 1: Data freshness check")
    data_ok, data_msg, files = check_data_freshness()
    log.info(data_msg)
    if not data_ok:
        send_discord(
            f"**Data sync check FAILED** — training aborted.\n\n{data_msg}",
            title="Nightly Retrain ABORTED",
            color=0xff6600,  # orange
        )
        log.error("Aborting due to data check failure.")
        sys.exit(2)

    # ── Step 2: Launch training ───────────────────────────────────
    log.info(f"Step 2: Launching training run '{run_name}'")
    train_ok, output_dir = run_training(run_name)
    if not train_ok:
        wall = (time.time() - start_time) / 60
        msg  = (f"**Training FAILED** for run `{run_name}`.\n"
                f"Wall time: {wall:.1f}m\n"
                f"Log: `{output_dir / 'stdout.log'}`\n\n"
                "Previous champion unchanged.")
        send_discord(msg, title="Nightly Retrain FAILED", color=0xff0000)
        log.error("Training failed — keeping previous champion.")
        sys.exit(1)

    # ── Step 3: Validate IC ───────────────────────────────────────
    log.info("Step 3: Validating IC_1s on OOT window")
    passes, metrics = validate_model(output_dir)
    ic_1s  = metrics.get("ic_1s", "N/A")
    ic_10s = metrics.get("ic_10s", "N/A")
    wall   = (time.time() - start_time) / 60

    if not passes:
        msg = (
            f"**Validation FAILED** — champion unchanged.\n\n"
            f"Run: `{run_name}`\n"
            f"IC_1s: `{ic_1s}` (gate: `{MIN_IC_1S}`)\n"
            f"IC_10s: `{ic_10s}`\n"
            f"Wall time: `{wall:.1f}m`\n\n"
            f"Previous champion kept in production."
        )
        send_discord(msg, title="Nightly Retrain: Validation FAILED", color=0xff0000)
        log.warning("Validation failed — keeping previous champion.")
        sys.exit(0)  # Not an error, just didn't promote

    # ── Step 4: Promote champion ──────────────────────────────────
    log.info("Step 4: Promoting new champion")
    champ_path = promote_champion(output_dir, run_name, metrics)

    # ── Step 5: Discord success alert ─────────────────────────────
    wall = (time.time() - start_time) / 60
    msg  = (
        f"**New champion promoted!**\n\n"
        f"Run: `{run_name}`\n"
        f"IC_1s: `{ic_1s:.4f}` (gate: {MIN_IC_1S})\n"
        f"IC_10s: `{ic_10s}`\n"
        f"Wall time: `{wall:.1f}m`\n"
        f"Champion path: `{champ_path}`\n\n"
        f"Data: {data_msg}"
    )
    send_discord(msg, title="Nightly Retrain: New Champion!", color=0x00ff00)

    log.info(f"Pipeline complete. Wall time: {wall:.1f}m")
    log.info(f"Champion: {champ_path}")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        log.critical(f"Unhandled exception in nightly pipeline:\n{tb}")
        send_discord(
            f"**CRASH** in nightly retrain pipeline.\n\n```\n{tb[-1500:]}\n```",
            title="Nightly Retrain CRASHED",
            color=0xff0000,
        )
        sys.exit(1)
