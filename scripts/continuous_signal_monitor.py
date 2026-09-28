#!/usr/bin/env python3
"""
Continuous Signal Monitor (HC #790) — Event-driven signal capture
================================================================
Replaces static 3-window execution model with continuous monitoring.

Runs every 15 min during RTH (9:30-16:00), every 30 min extended hours (7-9:30, 16-20).
Detects significant changes in paper engine states and triggers re-aggregation
+ execution validation when conditions warrant.

Key improvements over static windows:
  1. Detects source count jumps, confidence spikes, direction flips in real-time
  2. Auto-triggers execution pre-validator when any signal crosses 78% confidence
  3. Covers pre-market (7 AM) through after-hours (8 PM)
  4. Only re-runs full aggregator when meaningful changes detected (saves compute)

Usage:
  python3 continuous_signal_monitor.py          # Normal mode
  python3 continuous_signal_monitor.py --force   # Force full re-aggregation
"""

import json
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path("/home/jupiter/Lvl3Quant")
STATE = BASE / "state"
SIGNALS_FILE = STATE / "agentic_signals.json"
SNAPSHOT_FILE = STATE / "signal_monitor_snapshot.json"
ALERT_FILE = STATE / "signal_flip_alert.txt"
MORNING_QUEUE_FILE = STATE / "morning_execution_queue.json"
MONITOR_LOG = BASE / "logs" / "continuous_signal_monitor.log"

# Paper engine state files to watch for changes
ENGINE_STATE_FILES = [
    BASE / "paper_engines" / "state" / "sector_momentum_state.json",
    BASE / "paper_engines" / "state" / "earnings_outperformance_state.json",
    BASE / "paper_engines" / "state" / "vix_contango_state.json",
    BASE / "paper_engines" / "state" / "sector_rank_reversal_state.json",
    BASE / "state" / "subsector_rotation_state.json",
    BASE / "state" / "sector_momentum_ml_state.json",
    BASE / "state" / "iv_rank_data.json",
    BASE / "state" / "timing_score_results.json",
]

# Thresholds for "meaningful change" detection
CONFIDENCE_JUMP_THRESHOLD = 0.10    # 10% confidence increase triggers alert
SOURCE_COUNT_JUMP = 2                # 2+ new confirming sources triggers alert
AUTO_EXECUTE_CONFIDENCE = 0.78       # Auto-trigger execution pre-validator above this
HIGH_CONFIDENCE_THRESHOLD = 0.85     # High-conviction = immediate alert

AGGREGATOR_SCRIPT = BASE / "paper_engines" / "agentic_signal_aggregator.py"
PRE_VALIDATOR_SCRIPT = BASE / "scripts" / "execution_pre_validator.py"
INJECT_SCRIPT = BASE / "scripts" / "autonomy_inject.sh"


def log(msg):
    """Append to monitor log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(MONITOR_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_json(path, data):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        log(f"ERROR saving {path}: {e}")


def compute_engine_fingerprint():
    """Hash all paper engine state files to detect changes."""
    hasher = hashlib.md5()
    for fpath in sorted(ENGINE_STATE_FILES):
        try:
            if fpath.exists():
                mtime = os.path.getmtime(fpath)
                size = fpath.stat().st_size
                hasher.update(f"{fpath}:{mtime}:{size}".encode())
        except Exception:
            continue
    return hasher.hexdigest()


def extract_signal_summary(signals_data):
    """Extract ticker -> {confidence, n_confirming, direction} from current signals."""
    summary = {}
    for sig in signals_data.get("signals", []):
        ticker = sig.get("ticker", "")
        if ticker:
            summary[ticker] = {
                "confidence": sig.get("confidence_score", 0),
                "n_confirming": sig.get("n_confirming", 0),
                "n_conflicting": sig.get("n_conflicting", 0),
                "direction": sig.get("direction", ""),
                "timing_score": sig.get("timing_score", 0),
                "timing_recommendation": sig.get("timing_recommendation", ""),
            }
    return summary


def detect_changes(current_summary, prior_summary):
    """Compare current vs prior signal summaries. Return list of significant changes."""
    changes = []

    for ticker, curr in current_summary.items():
        prior = prior_summary.get(ticker, {})

        # New signal appeared
        if not prior:
            if curr["confidence"] >= 0.70:
                changes.append({
                    "type": "NEW_SIGNAL",
                    "ticker": ticker,
                    "confidence": curr["confidence"],
                    "direction": curr["direction"],
                    "sources": curr["n_confirming"],
                    "msg": f"🆕 {ticker} appeared: {curr['direction']} {curr['confidence']*100:.0f}% conf, {curr['n_confirming']} sources"
                })
            continue

        # Confidence jump
        conf_delta = curr["confidence"] - prior.get("confidence", 0)
        if conf_delta >= CONFIDENCE_JUMP_THRESHOLD:
            changes.append({
                "type": "CONFIDENCE_JUMP",
                "ticker": ticker,
                "confidence": curr["confidence"],
                "prior_confidence": prior.get("confidence", 0),
                "delta": conf_delta,
                "msg": f"📈 {ticker} confidence jumped {prior.get('confidence',0)*100:.0f}% → {curr['confidence']*100:.0f}% (+{conf_delta*100:.0f}%)"
            })

        # Source count jump
        source_delta = curr["n_confirming"] - prior.get("n_confirming", 0)
        if source_delta >= SOURCE_COUNT_JUMP:
            changes.append({
                "type": "SOURCE_JUMP",
                "ticker": ticker,
                "sources": curr["n_confirming"],
                "prior_sources": prior.get("n_confirming", 0),
                "delta": source_delta,
                "msg": f"🔔 {ticker} gained {source_delta} new sources ({prior.get('n_confirming',0)} → {curr['n_confirming']})"
            })

        # Direction flip
        if prior.get("direction") and curr["direction"] != prior["direction"]:
            changes.append({
                "type": "DIRECTION_FLIP",
                "ticker": ticker,
                "direction": curr["direction"],
                "prior_direction": prior.get("direction", ""),
                "msg": f"🔄 {ticker} FLIPPED: {prior.get('direction','')} → {curr['direction']}"
            })

        # Timing upgrade (DEFER → ENTER)
        if prior.get("timing_recommendation", "").startswith("DEFER") and \
           curr.get("timing_recommendation", "").startswith("ENTER"):
            changes.append({
                "type": "TIMING_UPGRADE",
                "ticker": ticker,
                "timing": curr["timing_recommendation"],
                "msg": f"⏰ {ticker} timing upgraded: DEFER → {curr['timing_recommendation']}"
            })

    # Signal disappeared (was strong, now gone)
    for ticker, prior in prior_summary.items():
        if ticker not in current_summary and prior.get("confidence", 0) >= 0.75:
            changes.append({
                "type": "SIGNAL_LOST",
                "ticker": ticker,
                "prior_confidence": prior.get("confidence", 0),
                "msg": f"⚠️ {ticker} signal DISAPPEARED (was {prior['confidence']*100:.0f}%)"
            })

    return changes


def should_trigger_execution(changes, current_summary):
    """Determine if we should auto-trigger the execution pre-validator."""
    for change in changes:
        ticker = change.get("ticker", "")
        curr = current_summary.get(ticker, {})

        # Any signal crossing the execution threshold
        if curr.get("confidence", 0) >= AUTO_EXECUTE_CONFIDENCE:
            if change["type"] in ("CONFIDENCE_JUMP", "SOURCE_JUMP", "NEW_SIGNAL", "TIMING_UPGRADE"):
                return True, f"{ticker} at {curr['confidence']*100:.0f}% confidence"

        # High conviction = always check
        if curr.get("confidence", 0) >= HIGH_CONFIDENCE_THRESHOLD:
            return True, f"{ticker} high conviction at {curr['confidence']*100:.0f}%"

    return False, ""


def is_market_hours():
    """Check if we're in regular trading hours (9:30 AM - 4:00 PM ET)."""
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    return market_open <= now <= market_close and now.weekday() < 5


def is_extended_hours():
    """Check if we're in extended hours (7:00-9:30 or 16:00-20:00 ET)."""
    now = datetime.now()
    pre_market = now.replace(hour=7, minute=0, second=0) <= now < now.replace(hour=9, minute=30, second=0)
    after_hours = now.replace(hour=16, minute=0, second=0) <= now <= now.replace(hour=20, minute=0, second=0)
    return (pre_market or after_hours) and now.weekday() < 5


def run_aggregator():
    """Re-run the signal aggregator."""
    log("Running signal aggregator...")
    try:
        result = subprocess.run(
            ["python3", str(AGGREGATOR_SCRIPT)],
            capture_output=True, text=True, timeout=300,
            cwd=str(BASE)
        )
        if result.returncode == 0:
            log("Aggregator completed successfully")
            return True
        else:
            log(f"Aggregator failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        log(f"Aggregator error: {e}")
        return False


def run_pre_validator():
    """Run the execution pre-validator."""
    log("Running execution pre-validator...")
    try:
        result = subprocess.run(
            ["python3", str(PRE_VALIDATOR_SCRIPT)],
            capture_output=True, text=True, timeout=60,
            cwd=str(BASE)
        )
        output = result.stdout.strip()
        log(f"Pre-validator result: {output[:200]}")
        return output
    except Exception as e:
        log(f"Pre-validator error: {e}")
        return ""


def send_alert(message):
    """Send alert via autonomy_inject for Claude to act on."""
    try:
        if INJECT_SCRIPT.exists():
            subprocess.run(
                [str(INJECT_SCRIPT), f"SIGNAL_CHANGE_ALERT: {message}"],
                timeout=30
            )
            log(f"Alert sent: {message[:100]}")
    except Exception as e:
        log(f"Alert send failed: {e}")


def queue_for_morning(changes):
    """Queue significant after-hours changes for morning execution."""
    queue = load_json(MORNING_QUEUE_FILE)
    if not isinstance(queue, dict):
        queue = {}

    existing_flips = queue.get("flips", [])
    for change in changes:
        if change["type"] in ("CONFIDENCE_JUMP", "SOURCE_JUMP", "NEW_SIGNAL", "DIRECTION_FLIP"):
            existing_flips.append({
                "ticker": change["ticker"],
                "type": change["type"],
                "detail": change["msg"],
                "detected_at": datetime.now().isoformat(),
            })

    queue["flips"] = existing_flips
    queue["queued_at"] = datetime.now().isoformat()
    queue["status"] = "PENDING"
    save_json(MORNING_QUEUE_FILE, queue)
    log(f"Queued {len(changes)} changes for morning execution")


def main():
    force = "--force" in sys.argv

    log("=" * 60)
    log("Continuous Signal Monitor starting")

    # Load prior snapshot
    snapshot = load_json(SNAPSHOT_FILE)
    prior_fingerprint = snapshot.get("engine_fingerprint", "")
    prior_summary = snapshot.get("signal_summary", {})

    # Check if engine states changed
    current_fingerprint = compute_engine_fingerprint()
    engines_changed = current_fingerprint != prior_fingerprint

    if engines_changed or force:
        log(f"Engine states changed (or forced). Re-running aggregator.")
        run_aggregator()
    else:
        log("No engine state changes detected. Checking existing signals.")

    # Load current signals (possibly just re-aggregated)
    current_signals = load_json(SIGNALS_FILE)
    current_summary = extract_signal_summary(current_signals)

    # Detect changes
    changes = detect_changes(current_summary, prior_summary)

    if changes:
        log(f"Detected {len(changes)} significant changes:")
        for c in changes:
            log(f"  {c['msg']}")

        # During market hours: check if we should auto-execute
        if is_market_hours():
            should_exec, reason = should_trigger_execution(changes, current_summary)
            if should_exec:
                log(f"AUTO-EXECUTE triggered: {reason}")
                validator_result = run_pre_validator()

                # If validator found actionable trades, alert Claude
                if validator_result and "pass" in validator_result.lower() and "no trades pass" not in validator_result.lower():
                    alert_msg = f"Signal change detected actionable trade. {reason}. Validator says: {validator_result[:150]}"
                    send_alert(alert_msg)
                else:
                    log(f"Pre-validator: no trades pass gates despite signal change (no alert)")

            # Alert on high-impact changes even if not auto-executing
            high_impact = [c for c in changes if c["type"] in ("DIRECTION_FLIP", "NEW_SIGNAL")
                          and current_summary.get(c["ticker"], {}).get("confidence", 0) >= 0.75]
            if high_impact and not should_exec:
                msgs = "; ".join(c["msg"] for c in high_impact)
                send_alert(msgs)

        elif is_extended_hours():
            # After hours: queue for morning, don't execute
            actionable = [c for c in changes if c["type"] in ("CONFIDENCE_JUMP", "SOURCE_JUMP", "NEW_SIGNAL", "DIRECTION_FLIP")]
            if actionable:
                queue_for_morning(actionable)
                log("Extended hours — queued changes for morning")

        else:
            log("Outside trading hours — logging changes only")
    else:
        log("No significant signal changes detected")

    # Print summary of current signal landscape
    if current_summary:
        log("Current signal landscape:")
        for ticker, info in sorted(current_summary.items(), key=lambda x: -x[1]["confidence"]):
            log(f"  {ticker}: {info['direction']} {info['confidence']*100:.0f}% ({info['n_confirming']} sources) timing={info.get('timing_recommendation', 'N/A')}")

    # Save snapshot for next run
    save_json(SNAPSHOT_FILE, {
        "engine_fingerprint": current_fingerprint,
        "signal_summary": current_summary,
        "last_check": datetime.now().isoformat(),
        "changes_detected": len(changes),
        "last_changes": [c["msg"] for c in changes[:5]],
    })

    log("Monitor complete")


if __name__ == "__main__":
    main()
