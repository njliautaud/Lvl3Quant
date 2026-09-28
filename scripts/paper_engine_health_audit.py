#!/usr/bin/env python3
"""Paper Engine Health Audit

Scans all paper engine state files and classifies each engine's health.
Correlates with PM2 processes and cron jobs.

Usage:
    python3 paper_engine_health_audit.py            # Full table
    python3 paper_engine_health_audit.py --discord   # Short Discord summary
    python3 paper_engine_health_audit.py --cleanup   # Cleanup recommendations
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

STATE_DIR = Path("/home/jupiter/Lvl3Quant/state")
STATE_GLOB = "*paper_state*.json"
STALE_DAYS = 3
UNREALISTIC_MULTIPLIER = 80  # > 80x initial = unrealistic for options paper

# Equity field names used across different engines, in priority order
EQUITY_FIELDS = ["equity", "portfolio_value", "nav", "cash", "capital"]


def get_equity(data: dict) -> float | None:
    """Extract equity from state data, trying multiple field names."""
    for field in EQUITY_FIELDS:
        val = data.get(field)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def get_initial_capital(data: dict) -> float:
    """Determine the initial capital for an engine.

    Engines using 'capital' field started at that value (typically $100K).
    Engines using 'equity' field (sector spreads etc) started at $645.
    Engines using 'nav'/'cash' vary.
    """
    # If the engine has a 'capital' field, that IS its initial capital
    cap = data.get("capital")
    if cap is not None:
        try:
            return float(cap)
        except (TypeError, ValueError):
            pass
    # IC engine starts at ~65K
    if "next_id" in data and data.get("cash", 0) > 50000:
        return 65000.0
    # Cross-asset trend starts at ~30K
    if "nav" in data and data.get("nav", 0) > 20000:
        return 30000.0
    # Large portfolio engines (DL stock ranker, etc) likely started at $100K
    pv = data.get("portfolio_value")
    if pv is not None:
        try:
            pv_f = float(pv)
            if pv_f > 50000:
                return 100000.0
        except (TypeError, ValueError):
            pass
    # Default for sector/spreads engines
    return 645.0


def count_positions(data: dict) -> int:
    """Count open positions from state data."""
    for key in ["positions", "open_positions", "holdings", "eq_holdings"]:
        val = data.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            return len(val)
        if isinstance(val, list):
            return len(val)
    return 0


def count_closed_trades(data: dict) -> int:
    """Count closed trades from state data."""
    for key in ["closed_trades", "trade_log", "trade_history", "rebalance_history"]:
        val = data.get(key)
        if val is None:
            continue
        if isinstance(val, (list, dict)):
            return len(val)
    total = data.get("total_trades", 0)
    if total:
        try:
            return int(total)
        except (TypeError, ValueError):
            pass
    return 0


def get_last_update_from_data(data: dict) -> str | None:
    """Try to get last update timestamp from state data."""
    for key in ["last_update", "last_run", "last_rebalance"]:
        val = data.get(key)
        if val and isinstance(val, str):
            return val
    return None


def get_pm2_processes() -> dict:
    """Get PM2 process list with status. Returns {name: status}."""
    try:
        result = subprocess.run(
            ["pm2", "jlist"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return {}
        procs = json.loads(result.stdout)
        return {
            p["name"]: p.get("pm2_env", {}).get("status", "unknown")
            for p in procs
            if "paper" in p["name"].lower()
        }
    except Exception:
        return {}


def get_cron_entries() -> list[str]:
    """Get cron entries mentioning 'paper'."""
    try:
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return []
        return [
            line.strip()
            for line in result.stdout.splitlines()
            if "paper" in line.lower() and not line.strip().startswith("#")
        ]
    except Exception:
        return []


def classify_engine(
    equity: float | None,
    positions: int,
    closed_trades: int,
    mtime: float,
    has_pm2: bool,
    pm2_status: str | None,
    has_cron: bool,
    initial_capital: float = 645.0,
) -> tuple[str, str]:
    """Classify engine health. Returns (status, reason)."""
    now = time.time()
    days_old = (now - mtime) / 86400

    if equity is not None and equity == 0:
        return "BROKEN", "Zero equity"

    if equity is not None and initial_capital > 0:
        ratio = equity / initial_capital
        if ratio > UNREALISTIC_MULTIPLIER:
            return "UNREALISTIC", f"Equity ${equity:,.0f} ({ratio:.0f}x initial ${initial_capital:,.0f})"

    if positions == 0 and closed_trades == 0:
        return "EMPTY", "No trades ever executed"

    if days_old > STALE_DAYS:
        return "STALE", f"No update in {days_old:.1f} days"

    return "HEALTHY", "Active with trades"


def state_name_to_pm2_name(state_file: str) -> str:
    """Convert state filename to likely PM2 process name."""
    name = state_file.replace("_paper_state.json", "").replace("_", "-")
    return f"{name}-paper"


def audit_engines():
    """Run the full audit and return list of engine reports."""
    state_files = sorted(STATE_DIR.glob(STATE_GLOB))
    pm2_procs = get_pm2_processes()
    cron_entries = get_cron_entries()
    cron_text = "\n".join(cron_entries).lower()

    results = []
    for sf in state_files:
        name = sf.stem.replace("_paper_state", "")
        display_name = name.replace("_", "-")

        # Parse state file
        valid_json = False
        data = {}
        try:
            with open(sf) as f:
                data = json.load(f)
            valid_json = True
        except (json.JSONDecodeError, OSError):
            pass

        equity = get_equity(data) if valid_json else None
        initial_capital = get_initial_capital(data) if valid_json else 645.0
        positions = count_positions(data) if valid_json else 0
        closed_trades = count_closed_trades(data) if valid_json else 0
        last_update_str = get_last_update_from_data(data) if valid_json else None
        mtime = sf.stat().st_mtime

        # PM2 correlation
        pm2_name = state_name_to_pm2_name(sf.name)
        pm2_status = pm2_procs.get(pm2_name)
        # Try alternate names
        if pm2_status is None:
            for pname, pstatus in pm2_procs.items():
                if display_name in pname or name.replace("_", "-") in pname:
                    pm2_status = pstatus
                    pm2_name = pname
                    break

        has_pm2 = pm2_status is not None
        has_cron = display_name in cron_text or name in cron_text

        if not valid_json:
            status, reason = "BROKEN", "Invalid JSON"
        else:
            status, reason = classify_engine(
                equity, positions, closed_trades, mtime, has_pm2, pm2_status, has_cron,
                initial_capital=initial_capital,
            )

        mtime_dt = datetime.fromtimestamp(mtime)
        days_ago = (time.time() - mtime) / 86400

        results.append({
            "name": display_name,
            "status": status,
            "reason": reason,
            "equity": equity,
            "positions": positions,
            "closed_trades": closed_trades,
            "last_modified": mtime_dt.strftime("%Y-%m-%d %H:%M"),
            "days_ago": days_ago,
            "pm2_status": pm2_status or "none",
            "has_cron": has_cron,
            "valid_json": valid_json,
        })

    return results


def print_table(results: list[dict]):
    """Print full audit table to stdout."""
    status_colors = {
        "HEALTHY": "\033[32m",
        "STALE": "\033[33m",
        "BROKEN": "\033[31m",
        "UNREALISTIC": "\033[35m",
        "EMPTY": "\033[90m",
    }
    reset = "\033[0m"

    print("=" * 120)
    print("PAPER ENGINE HEALTH AUDIT")
    print(f"Scanned: {len(results)} engines | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 120)
    print(
        f"{'Engine':<30} {'Status':<14} {'Equity':>12} {'Pos':>4} {'Trades':>7} "
        f"{'Last Update':<18} {'PM2':<10} {'Cron':<5}"
    )
    print("-" * 120)

    for r in sorted(results, key=lambda x: (
        {"BROKEN": 0, "UNREALISTIC": 1, "EMPTY": 2, "STALE": 3, "HEALTHY": 4}.get(x["status"], 5),
        x["name"],
    )):
        color = status_colors.get(r["status"], "")
        eq_str = f"${r['equity']:,.0f}" if r["equity"] is not None else "N/A"
        cron_str = "yes" if r["has_cron"] else "no"
        days_str = f"({r['days_ago']:.1f}d ago)"

        print(
            f"{r['name']:<30} {color}{r['status']:<14}{reset} {eq_str:>12} "
            f"{r['positions']:>4} {r['closed_trades']:>7} "
            f"{r['last_modified']:<18} {r['pm2_status']:<10} {cron_str:<5}"
        )

    # Summary
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print("-" * 120)
    print("SUMMARY:", end="")
    for status in ["HEALTHY", "STALE", "BROKEN", "UNREALISTIC", "EMPTY"]:
        if status in counts:
            color = status_colors.get(status, "")
            print(f"  {color}{status}: {counts[status]}{reset}", end="")
    print(f"  | Total: {len(results)}")
    print("=" * 120)


def print_discord(results: list[dict]):
    """Print short Discord-friendly summary (per HC #433 - plain English, no paths)."""
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    healthy = [r for r in results if r["status"] == "HEALTHY"]
    broken = [r for r in results if r["status"] == "BROKEN"]
    stale = [r for r in results if r["status"] == "STALE"]
    empty = [r for r in results if r["status"] == "EMPTY"]
    unrealistic = [r for r in results if r["status"] == "UNREALISTIC"]

    lines = [f"Paper Engine Audit - {len(results)} engines scanned"]
    lines.append("")

    if healthy:
        lines.append(f"Healthy: {len(healthy)} engines active with trades")
        for r in healthy:
            eq_str = f"${r['equity']:,.0f}" if r['equity'] is not None else "no equity tracked"
            lines.append(f"  {r['name']}: {eq_str}, {r['positions']} open pos, {r['closed_trades']} trades")

    if broken:
        lines.append(f"Broken: {len(broken)} engines need attention")
        for r in broken:
            lines.append(f"  {r['name']}: {r['reason']}")

    if unrealistic:
        lines.append(f"Unrealistic: {len(unrealistic)} engines with implausible returns")
        for r in unrealistic:
            lines.append(f"  {r['name']}: {r['reason']}")

    if stale:
        lines.append(f"Stale: {len(stale)} engines not updated in 3+ days")

    if empty:
        lines.append(f"Empty: {len(empty)} engines never traded")

    running_pm2 = sum(1 for r in results if r["pm2_status"] == "online")
    stopped_pm2 = sum(1 for r in results if r["pm2_status"] not in ("online", "none"))
    if running_pm2 or stopped_pm2:
        lines.append(f"PM2: {running_pm2} running, {stopped_pm2} stopped")

    print("\n".join(lines))


def print_cleanup(results: list[dict]):
    """Print cleanup recommendations."""
    print("=" * 80)
    print("CLEANUP RECOMMENDATIONS")
    print("=" * 80)

    # Engines to stop (running but broken/empty)
    stop_candidates = [
        r for r in results
        if r["pm2_status"] == "online" and r["status"] in ("BROKEN", "EMPTY", "UNREALISTIC")
    ]
    if stop_candidates:
        print("\nSTOP these PM2 processes (running but broken/empty/unrealistic):")
        for r in stop_candidates:
            print(f"  pm2 stop {r['name']}-paper  # {r['reason']}")

    # Engines to remove (stopped + empty/broken)
    remove_candidates = [
        r for r in results
        if r["pm2_status"] not in ("online", "none") and r["status"] in ("BROKEN", "EMPTY")
    ]
    if remove_candidates:
        print("\nREMOVE these PM2 entries (stopped and broken/empty):")
        for r in remove_candidates:
            print(f"  pm2 delete {r['name']}-paper  # {r['reason']}")

    # Stale engines with no process
    stale_orphans = [
        r for r in results
        if r["status"] == "STALE" and r["pm2_status"] != "online" and not r["has_cron"]
    ]
    if stale_orphans:
        print("\nSTALE ORPHANS (no PM2, no cron, stale state file):")
        for r in stale_orphans:
            print(f"  {r['name']}: last updated {r['days_ago']:.1f} days ago")
            print(f"    Consider removing state file if no longer needed")

    # Engines with zero equity
    zero_eq = [r for r in results if r["equity"] is not None and r["equity"] == 0]
    if zero_eq:
        print("\nZERO EQUITY (reset or investigate):")
        for r in zero_eq:
            print(f"  {r['name']}: equity is $0, likely needs reset")

    # PM2 processes with no state file (running but maybe shouldn't be)
    pm2_procs = get_pm2_processes()
    state_names = {r["name"] for r in results}
    orphan_pm2 = {
        name: status for name, status in pm2_procs.items()
        if not any(sn in name for sn in state_names)
    }
    if orphan_pm2:
        print("\nPM2 PROCESSES WITHOUT MATCHING STATE FILE:")
        for name, status in orphan_pm2.items():
            print(f"  {name}: {status}")

    if not any([stop_candidates, remove_candidates, stale_orphans, zero_eq, orphan_pm2]):
        print("\nNo cleanup actions needed - all engines look reasonable.")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Paper Engine Health Audit")
    parser.add_argument("--discord", action="store_true", help="Short Discord-friendly output")
    parser.add_argument("--cleanup", action="store_true", help="Show cleanup recommendations")
    args = parser.parse_args()

    results = audit_engines()

    if args.discord:
        print_discord(results)
    elif args.cleanup:
        print_cleanup(results)
    else:
        print_table(results)


if __name__ == "__main__":
    main()
