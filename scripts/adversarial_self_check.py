#!/usr/bin/env python3
"""
Adversarial Self-Check — Daily Accountability Adversary
========================================================
Runs daily and catches silent failures across the entire quant infrastructure.
Checks paper engines, pipeline components, data freshness, and sanity.

Usage:
    python3 scripts/adversarial_self_check.py
    python3 scripts/adversarial_self_check.py --quiet   # JSON output only, no stdout

Output:
    state/self_check_results.json
    stdout: human-readable summary
"""

import json
import os
import sys
import glob
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

BASE = "/home/jupiter/Lvl3Quant"
PAPER_LOG_DIRS = [
    os.path.join(BASE, "paper_engines", "logs"),
    os.path.join(BASE, "logs"),
    os.path.join(BASE, "logs", "paper_engines"),
]
STATE_DIR = os.path.join(BASE, "state")
OUTPUT_FILE = os.path.join(STATE_DIR, "self_check_results.json")

ERROR_PATTERNS = re.compile(
    r"(ERROR|Traceback|Exception|CRITICAL|FATAL|ModuleNotFoundError|"
    r"KeyError|TypeError|ValueError|FileNotFoundError|ConnectionError|"
    r"PermissionError|ImportError|RuntimeError|ZeroDivisionError)",
    re.IGNORECASE,
)
SUCCESS_PATTERNS = re.compile(
    r"(State saved|Done|SUMMARY|saved state|completed|finished|"
    r"Paper engine complete|wrote.*state|portfolio updated|rebalance complete)",
    re.IGNORECASE,
)

# Pipeline state files to check for freshness
PIPELINE_STATE_FILES = {
    "flow_screener_signals": "state/flow_screener_signals.json",
    "spread_recommendations": "state/spread_recommendations.json",
    "daily_trade_plan": "state/daily_trade_plan.json",
    "agentic_signals": "state/agentic_signals.json",
}

# Pipeline log files to check
PIPELINE_LOGS = {
    "flow_screener": "logs/flow_screener/screener.log",
    "spread_engine": "logs/spread_engine.log",
    "portfolio_brain": "logs/portfolio_brain.log",
    "signal_aggregator": "logs/agentic_signal_aggregator.log",
}

# How many tail lines to scan for errors/success
TAIL_LINES = 30
# Stale equity threshold (days without change)
STALE_EQUITY_DAYS = 5
# IV shift suspicion threshold (%)
IV_SHIFT_SUSPICIOUS = 90.0


# ─── Helpers ─────────────────────────────────────────────────────────────────

def last_trading_day(ref_date=None):
    """Return the most recent trading day (Mon-Fri) on or before ref_date."""
    d = ref_date or datetime.now()
    # If weekend, roll back to Friday
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d.date()


def file_mod_date(path):
    """Return the modification date of a file, or None if missing."""
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).date()
    except (OSError, FileNotFoundError):
        return None


def tail_file(path, n=TAIL_LINES):
    """Read last n lines of a file. Returns list of strings."""
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
            return lines[-n:]
    except (OSError, FileNotFoundError):
        return []


def parse_crontab_paper_engines():
    """Parse crontab to discover all paper engine entries with their log paths."""
    engines = []
    try:
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return engines
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            # Look for paper-related cron entries
            if "paper" not in line.lower() and "Paper" not in line:
                continue
            # Extract the log file path (after >> or > )
            log_match = re.search(r">>\s*(\S+)", line)
            # Extract the script name
            script_match = re.search(r"(?:python3?|/usr/bin/python3)\s+(?:-m\s+)?(\S+\.py\b|\S+)", line)

            log_path = None
            if log_match:
                log_raw = log_match.group(1)
                # Resolve relative paths against BASE
                if not os.path.isabs(log_raw):
                    log_path = os.path.join(BASE, log_raw)
                else:
                    log_path = log_raw

            script_name = ""
            if script_match:
                script_name = script_match.group(1)

            # Derive a readable engine name from script path
            name = script_name.replace("/", ".").replace(".py", "")
            if not name:
                name = f"unknown_cron_{hash(line) % 10000}"

            engines.append({
                "name": name,
                "log_path": log_path,
                "cron_line": line,
            })
    except Exception:
        pass
    return engines


def check_log_health(log_path, engine_name, target_date):
    """Check a log file for recent runs, errors, and success markers.

    Returns (issues_list, is_healthy).
    """
    issues = []

    if log_path is None:
        issues.append({
            "component": engine_name,
            "issue": "No log path configured in crontab",
            "severity": "warning",
        })
        return issues, False

    if not os.path.exists(log_path):
        issues.append({
            "component": engine_name,
            "issue": f"Log file does not exist",
            "severity": "warning",
        })
        return issues, False

    mod_date = file_mod_date(log_path)
    if mod_date is None:
        issues.append({
            "component": engine_name,
            "issue": "Cannot read log file modification date",
            "severity": "warning",
        })
        return issues, False

    # Check if log was updated on the target trading day (or after)
    if mod_date < target_date:
        days_stale = (datetime.now().date() - mod_date).days
        issues.append({
            "component": engine_name,
            "issue": f"Log not updated since {mod_date} ({days_stale}d stale, expected {target_date})",
            "severity": "critical" if days_stale > 3 else "warning",
        })
        return issues, False

    # Check tail for errors
    tail = tail_file(log_path)
    error_lines = [l.strip() for l in tail if ERROR_PATTERNS.search(l)]
    success_lines = [l.strip() for l in tail if SUCCESS_PATTERNS.search(l)]

    if error_lines:
        # Summarize: show first distinct error, truncated
        first_error = error_lines[-1][:200]
        issues.append({
            "component": engine_name,
            "issue": f"Errors in recent log ({len(error_lines)} error lines). Last: {first_error}",
            "severity": "critical",
        })
        # Still might be healthy if there's a success AFTER errors
        if not success_lines:
            return issues, False

    if not success_lines and not error_lines:
        # Log updated but no clear success/error markers
        issues.append({
            "component": engine_name,
            "issue": "Log updated but no success markers found (State saved/Done/SUMMARY)",
            "severity": "warning",
        })
        return issues, len(issues) == 0

    return issues, len(issues) == 0


def check_paper_state_staleness(target_date):
    """Check paper engine state files for stale equity (unchanged for 5+ days)."""
    issues = []
    state_files = glob.glob(os.path.join(STATE_DIR, "*paper_state*.json"))

    for sf in state_files:
        try:
            with open(sf, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            basename = os.path.basename(sf)
            issues.append({
                "component": f"state:{basename}",
                "issue": "State file is corrupt or unreadable",
                "severity": "critical",
            })
            continue

        basename = os.path.basename(sf).replace("_paper_state.json", "")

        # Check for capital/equity field
        equity = data.get("capital") or data.get("equity") or data.get("portfolio_value")
        if equity is not None:
            # Check if file hasn't been modified in 5+ days
            mod_date = file_mod_date(sf)
            if mod_date and (datetime.now().date() - mod_date).days >= STALE_EQUITY_DAYS:
                issues.append({
                    "component": f"paper_state:{basename}",
                    "issue": f"State file unchanged for {(datetime.now().date() - mod_date).days}d (equity={equity:.2f}). Engine may be dead.",
                    "severity": "warning",
                })

        # Check for positions with no recent activity
        positions = data.get("positions", [])
        trades = data.get("trade_history", data.get("trades", []))
        if isinstance(positions, list) and len(positions) > 0 and isinstance(trades, list) and len(trades) == 0:
            issues.append({
                "component": f"paper_state:{basename}",
                "issue": f"Has {len(positions)} positions but zero trade history — possible stale init state",
                "severity": "warning",
            })

    return issues


def check_pipeline_state_freshness(target_date):
    """Check that pipeline state JSON files are from today/last trading day and have real data."""
    issues = []
    healthy = []

    for name, rel_path in PIPELINE_STATE_FILES.items():
        full_path = os.path.join(BASE, rel_path)

        if not os.path.exists(full_path):
            issues.append({
                "component": f"state:{name}",
                "issue": "State file missing entirely",
                "severity": "critical",
            })
            continue

        mod_date = file_mod_date(full_path)
        if mod_date and mod_date < target_date:
            days_stale = (datetime.now().date() - mod_date).days
            issues.append({
                "component": f"state:{name}",
                "issue": f"State file stale (last modified {mod_date}, {days_stale}d ago)",
                "severity": "critical" if days_stale > 2 else "warning",
            })
            continue

        # Try to read and validate content
        try:
            with open(full_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            issues.append({
                "component": f"state:{name}",
                "issue": "State file is corrupt JSON",
                "severity": "critical",
            })
            continue

        # Specific validations
        if name == "daily_trade_plan":
            trades = data.get("trades", [])
            if not trades:
                issues.append({
                    "component": f"state:{name}",
                    "issue": "Trade plan has empty trades list",
                    "severity": "critical",
                })
                continue
            # Check for zero OI/volume
            zero_oi_count = sum(
                1 for t in trades
                if isinstance(t, dict) and t.get("open_interest", 1) == 0 and t.get("volume", 1) == 0
            )
            if zero_oi_count == len(trades) and len(trades) > 0:
                issues.append({
                    "component": f"state:{name}",
                    "issue": f"All {len(trades)} trades have zero OI/volume — suspicious data",
                    "severity": "warning",
                })
                continue

        if name == "flow_screener_signals":
            # Check for suspicious IV shifts
            signals = data.get("signals", data.get("tickers", []))
            if isinstance(signals, list):
                for sig in signals:
                    if isinstance(sig, dict):
                        iv_shift = sig.get("iv_shift", sig.get("iv_change", 0))
                        if isinstance(iv_shift, (int, float)) and abs(iv_shift) > IV_SHIFT_SUSPICIOUS:
                            ticker = sig.get("ticker", sig.get("symbol", "?"))
                            issues.append({
                                "component": f"state:{name}",
                                "issue": f"Suspicious IV shift {iv_shift:.1f}% on {ticker} (> {IV_SHIFT_SUSPICIOUS}%)",
                                "severity": "warning",
                            })

        healthy.append(f"state:{name}")

    return issues, healthy


def check_pipeline_logs(target_date):
    """Check pipeline component log files."""
    issues = []
    healthy = []

    for name, rel_path in PIPELINE_LOGS.items():
        full_path = os.path.join(BASE, rel_path)
        log_issues, is_ok = check_log_health(full_path, f"pipeline:{name}", target_date)
        issues.extend(log_issues)
        if is_ok:
            healthy.append(f"pipeline:{name}")

    return issues, healthy


# ─── Main ────────────────────────────────────────────────────────────────────

def run_self_check():
    quiet = "--quiet" in sys.argv

    now = datetime.now()
    target_date = last_trading_day(now)

    all_issues = []
    all_healthy = []

    # ── 1. Paper Engines from crontab ──
    engines = parse_crontab_paper_engines()
    if not engines:
        all_issues.append({
            "component": "crontab",
            "issue": "No paper engine cron entries found — crontab may be empty or unreadable",
            "severity": "critical",
        })

    for eng in engines:
        log_issues, is_ok = check_log_health(eng["log_path"], f"paper:{eng['name']}", target_date)
        all_issues.extend(log_issues)
        if is_ok:
            all_healthy.append(f"paper:{eng['name']}")

    # ── 2. Pipeline components ──
    pipe_issues, pipe_healthy = check_pipeline_logs(target_date)
    all_issues.extend(pipe_issues)
    all_healthy.extend(pipe_healthy)

    # ── 3. Pipeline state freshness ──
    state_issues, state_healthy = check_pipeline_state_freshness(target_date)
    all_issues.extend(state_issues)
    all_healthy.extend(state_healthy)

    # ── 4. Paper state staleness / dead engines ──
    stale_issues = check_paper_state_staleness(target_date)
    all_issues.extend(stale_issues)

    # ── Deduplicate by component name ──
    seen = set()
    deduped_issues = []
    for iss in all_issues:
        key = (iss["component"], iss["issue"][:80])
        if key not in seen:
            seen.add(key)
            deduped_issues.append(iss)
    all_issues = deduped_issues

    # ── Tally ──
    critical_count = sum(1 for i in all_issues if i["severity"] == "critical")
    warning_count = sum(1 for i in all_issues if i["severity"] == "warning")
    healthy_count = len(all_healthy)
    overall = "FAIL" if critical_count > 0 else ("WARN" if warning_count > 0 else "PASS")

    summary = f"{critical_count} critical, {warning_count} warnings, {healthy_count} healthy"

    result = {
        "check_date": now.strftime("%Y-%m-%d"),
        "check_time": now.strftime("%H:%M:%S"),
        "target_trading_day": str(target_date),
        "overall_status": overall,
        "failures": all_issues,
        "healthy": sorted(set(all_healthy)),
        "summary": summary,
        "engines_discovered": len(engines),
    }

    # ── Write output ──
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    # ── Print human-readable summary ──
    if not quiet:
        print("=" * 70)
        print(f"  ADVERSARIAL SELF-CHECK — {now.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Target trading day: {target_date}")
        print(f"  Paper engines discovered in crontab: {len(engines)}")
        print("=" * 70)
        print()

        if overall == "PASS":
            print(f"  OVERALL: PASS  ({summary})")
        elif overall == "WARN":
            print(f"  OVERALL: WARN  ({summary})")
        else:
            print(f"  OVERALL: FAIL  ({summary})")
        print()

        if all_issues:
            # Print criticals first
            criticals = [i for i in all_issues if i["severity"] == "critical"]
            warnings = [i for i in all_issues if i["severity"] == "warning"]

            if criticals:
                print("  CRITICAL FAILURES:")
                print("  " + "-" * 66)
                for iss in criticals:
                    print(f"    [CRIT] {iss['component']}")
                    # Word-wrap long issues
                    issue_text = iss["issue"]
                    while len(issue_text) > 64:
                        print(f"           {issue_text[:64]}")
                        issue_text = issue_text[64:]
                    print(f"           {issue_text}")
                print()

            if warnings:
                print("  WARNINGS:")
                print("  " + "-" * 66)
                for iss in warnings:
                    print(f"    [WARN] {iss['component']}")
                    issue_text = iss["issue"]
                    while len(issue_text) > 64:
                        print(f"           {issue_text[:64]}")
                        issue_text = issue_text[64:]
                    print(f"           {issue_text}")
                print()

        if all_healthy:
            print(f"  HEALTHY ({healthy_count}):")
            print("  " + "-" * 66)
            for h in sorted(set(all_healthy)):
                print(f"    [OK]   {h}")
            print()

        print(f"  Results saved to: {OUTPUT_FILE}")
        print("=" * 70)

    return result


if __name__ == "__main__":
    result = run_self_check()
    # Exit with non-zero if critical failures found
    criticals = sum(1 for i in result["failures"] if i["severity"] == "critical")
    sys.exit(1 if criticals > 0 else 0)
