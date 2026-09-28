#!/usr/bin/env python3
"""
Completion Checker — Pipeline Integrity Auditor
================================================
Scans all paper engines and checks whether each one is:
  1. Adversarially tested (result in SESSION_STATE.md)
  2. Has a recent state file (updated within 48h on weekdays)
  3. Has a cron entry to run it daily
  4. Is wired into the signal aggregator

Outputs gap report to state/completion_gaps.json and a plain-English
summary to stdout (for autonomy_inject consumption).

Run: python3 scripts/completion_checker.py
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path("/home/jupiter/Lvl3Quant")
PAPER_ENGINES_DIR = BASE / "paper_engines"
STATE_DIR = BASE / "state"
SESSION_STATE = BASE / "SESSION_STATE.md"
AGGREGATOR = PAPER_ENGINES_DIR / "agentic_signal_aggregator.py"
OUTPUT = STATE_DIR / "completion_gaps.json"

# Files that are NOT daily-run paper engines (infrastructure, docs, one-off scripts)
EXCLUDED_FILES = {
    "agentic_signal_aggregator.py",
    "ecosystem.config.js",
    "__init__.py",
    # One-off backtests / adversarial scripts (not daily engines)
    "rotation_confluence_backtest.py",
    "subsector_rotation_adversarial.py",
    "sector_combined_adversarial.py",
    # Signal generators (not paper engines themselves)
    "v10_agentic_signal_generator.py",
    "broad_market_sector_screener.py",
    "subsector_rotation_tracker.py",
    "subsector_rotation_ml.py",
    # Superseded iterative versions — only latest (v10_optimal) is active
    "sector_combined_v7_paper.py",
    "sector_combined_v8_paper.py",
    "sector_combined_v9_paper.py",
    "sector_combined_v91_paper.py",
    "sector_combined_v92_paper.py",
    "sector_combined_v93_paper.py",
    "sector_combined_v10_paper.py",
    # CONFIRMED DEAD — adversarial batch 2026-09-03 classified these as SKIP/DEAD
    # No state files, stale 800+ hours, no crons, no active trading
    "contrarian_sector_reversion_paper.py",  # NO STATE FILE
    "fifo_champion_paper.py",                # NO STATE FILE, no cron
    "integrated_pipeline_paper.py",          # NO STATE FILE, no cron
    "queue_entry_v21_paper.py",              # NO STATE FILE, no cron
    "earnings_gap_halfsize_paper.py",        # STALE 956h, no cron
    "earnings_jade_lizard_paper.py",         # STALE 909h, no cron
    "equity_rotation_paper.py",              # STALE 1011h (test run, never real)
    "market_neutral_ls_paper.py",            # STALE 841h, no cron
    "momentum_options_paper.py",             # STALE 841h, no cron
    # Engines that exist as files but never produced state (non-functional)
    "bond_yield_inflow_paper_engine.py",     # No state file ever produced
    "multi_timeframe_paper.py",              # No state file, no cron
}
EXCLUDED_PREFIXES = ("CONFLUENCE_", "CROSS_TYPE_CONFLUENCE_README", "INDEX_")

# Mapping from engine filename to SESSION_STATE search aliases
# (many strategies are logged under shortened/different names)
ENGINE_ALIASES = {
    "sector_combined_v93_paper.py": ["V93", "SECTOR COMBINED V93", "V9.3"],
    "sector_combined_v92_paper.py": ["V92", "SECTOR COMBINED V92", "V9.2"],
    "sector_combined_v91_paper.py": ["V91", "SECTOR COMBINED V91", "V9.1"],
    "sector_combined_v9_paper.py": ["V9 ", "SECTOR COMBINED V9"],
    "sector_combined_v10_optimal_paper.py": ["V10", "SECTOR COMBINED V10", "V10 OPTIMAL"],
    "sector_combined_v10_paper.py": ["V10", "SECTOR COMBINED V10"],
    "sector_combined_v8_paper.py": ["V8", "SECTOR COMBINED V8"],
    "sector_combined_v7_paper.py": ["V7", "SECTOR COMBINED V7"],
    "cross_type_confluence_paper.py": ["CROSS-TYPE", "CROSS TYPE CONFLUENCE"],
    "subsector_rotation_paper_engine.py": ["SUB-SECTOR ROTATION", "SUBSECTOR ROTATION PAPER"],
    "strategy_rotation_v2f_paper.py": ["STRATEGY ROTATION V2F", "ROTATION V2F"],
    "fifo_champion_paper.py": ["FIFO CHAMPION"],
    "bond_yield_inflow_paper_engine.py": ["BOND YIELD", "BOND YIELD SIGNAL"],
    "iv_rv_gap_paper.py": ["IV-RV GAP", "IV RV GAP", "VOL REGIME F"],
    "vix_mr_spread_paper.py": ["VIX MR", "VIX MEAN REVERSION"],
    "vix_call_spread_paper.py": ["VIX CALL SPREAD"],
    "multi_timeframe_paper.py": ["MULTI-TF", "MULTI TIMEFRAME", "MULTI-TIMEFRAME"],
    "earnings_gap_halfsize_paper.py": ["EARNINGS GAP"],
    "earnings_iv_crush_real_paper.py": ["EARNINGS IV CRUSH", "IV CRUSH", "EARNINGS VOL CRUSH"],
    "earnings_jade_lizard_paper.py": ["JADE LIZARD", "EARNINGS JADE"],
    "factor_etf_rotation_paper.py": ["FACTOR ETF ROTATION"],
    "integrated_pipeline_paper.py": ["INTEGRATED PIPELINE"],
    "market_neutral_ls_paper.py": ["MARKET NEUTRAL"],
    "queue_entry_v21_paper.py": ["QUEUE ENTRY V21"],
    "sector_momentum_spreads_paper.py": ["SECTOR MOMENTUM SPREADS"],
    "weekly_momentum_burst_paper.py": ["WEEKLY MOMENTUM BURST", "MOMENTUM BURST"],
    "vol_crush_paper.py": ["VOL CRUSH", "EARNINGS VOL CRUSH"],
    "gap_fade_spread_paper.py": ["GAP FADE"],
    "extreme_idio_paper.py": ["EXTREME IDIO"],
    "volume_surge_paper.py": ["VOLUME SURGE"],
    "sector_reversal_paper.py": ["SECTOR REVERSAL"],
    "sector_pairs_paper.py": ["SECTOR PAIRS"],
    "sector_earnings_standalone_paper.py": ["SECTOR EARNINGS STANDALONE"],
    "contrarian_sector_reversion_paper.py": ["CONTRARIAN SECTOR REVERSION", "CONTRARIAN REVERSION"],
    "iv_runup_paper.py": ["IV RUNUP", "IV RUN-UP"],
    # Paper engines whose underlying strategies are already adversarially validated
    "subsector_rotation_paper.py": ["SUB-SECTOR ROTATION", "SUBSECTOR ROTATION", "Sub-Sector Rotation Adversarial"],
    "sequential_chain_paper.py": ["SEQUENTIAL CHAIN", "Sequential Chain E"],
    "rsi_divergence_paper.py": ["RSI DIVERGENCE", "RSI Divergence C Adversarial"],
    "signal_aggregator_paper.py": ["SIGNAL AGGREGATION", "Signal Aggregation v1"],
    "signal_scoring_paper.py": ["SIGNAL SCORING", "Signal Aggregation"],
    "liquidity_signal_paper.py": ["LIQUIDITY SIGNAL", "Liquidity Signal F"],
    "sector_equity_rotation_paper.py": ["SECTOR EQUITY ROTATION", "Vol-Adj RS", "EQUITY ROTATION"],
    "sector_etf_momentum_paper.py": ["SECTOR ETF MOMENTUM", "ETF MOMENTUM"],
    "momentum_options_paper.py": ["MOMENTUM OPTIONS"],
    "pead_drift_paper.py": ["PEAD DRIFT", "PEAD"],
    "pead_ml_paper.py": ["PEAD ML", "PEAD"],
    "quality_momentum_paper.py": ["QUALITY MOMENTUM", "Quality Mean Reversion"],
    "sector_combined_adversarial.py": ["SECTOR COMBINED ADVERSARIAL"],
    "vol_term_structure_paper.py": ["VOL TERM STRUCTURE", "VOLATILITY TERM STRUCTURE"],
    "sector_spreads_paper.py": ["SECTOR SPREADS"],
    "unified_portfolio_engine.py": ["UNIFIED PORTFOLIO", "WEEKLY RISK PARITY"],
    "vix_contango_sector_oversold_paper.py": ["VIX CONTANGO", "VIX CONTANGO SECTOR"],
}


def find_paper_engines():
    """Find all Python files in paper_engines/ that look like engines."""
    engines = []
    for f in sorted(PAPER_ENGINES_DIR.glob("*.py")):
        name = f.name
        if name in EXCLUDED_FILES:
            continue
        if any(name.startswith(p) for p in EXCLUDED_PREFIXES):
            continue
        if name.startswith("__"):
            continue
        engines.append(name)
    return engines


def infer_state_filename(engine_name):
    """Guess the state file name for a given engine."""
    # Common patterns:
    #   sector_reversal_paper.py -> sector_reversal_paper_state.json
    #   bond_yield_inflow_paper_engine.py -> bond_yield_paper_state.json (or similar)
    #   subsector_rotation_ml.py -> subsector_rotation_state.json
    base = engine_name.replace(".py", "")

    candidates = [
        f"{base}_state.json",
        f"{base.replace('_paper_engine', '_paper_state')}.json",
        f"{base.replace('_paper', '_paper_state')}.json",
        f"{base.replace('_paper', '_state')}.json",
        f"{base.replace('_engine', '_state')}.json",
        f"{base}_state.json".replace("_paper_state_state", "_paper_state"),
    ]
    # Also check inside paper_engines/state/ directory
    pe_state_dir = PAPER_ENGINES_DIR / "state"

    found = []
    for c in candidates:
        if (STATE_DIR / c).exists():
            found.append(STATE_DIR / c)
        if pe_state_dir.exists() and (pe_state_dir / c).exists():
            found.append(pe_state_dir / c)

    # Deduplicate
    seen = set()
    unique = []
    for p in found:
        if str(p) not in seen:
            seen.add(str(p))
            unique.append(p)

    return unique


def check_adversarial_status(engine_name):
    """Check SESSION_STATE.md AND RUN_HISTORY.md for adversarial validation results."""
    if not SESSION_STATE.exists():
        return "UNKNOWN", "SESSION_STATE.md not found"

    content = SESSION_STATE.read_text(errors="replace")
    # Also search RUN_HISTORY.md where most adversarial results are recorded
    run_history = BASE / "RUN_HISTORY.md"
    if run_history.exists():
        content = content + "\n" + run_history.read_text(errors="replace")

    # Extract a "strategy name" from the engine filename
    # e.g., sector_reversal_paper.py -> "sector reversal", "sector_reversal"
    base = engine_name.replace(".py", "").replace("_paper_engine", "").replace("_paper", "")
    # Also try with spaces and various forms
    search_terms = [
        base.replace("_", " ").upper(),
        base.replace("_", " ").title(),
        base.replace("_", " "),
        base,
        engine_name,
    ]
    # Add manually defined aliases for strategies logged under different names
    if engine_name in ENGINE_ALIASES:
        search_terms = ENGINE_ALIASES[engine_name] + search_terms

    # Look for status markers near mentions of this strategy
    for term in search_terms:
        if len(term) < 5:
            continue
        # Case-insensitive search
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches = list(pattern.finditer(content))
        if not matches:
            continue

        # Check nearby lines for status markers
        for match in matches:
            # Get context: 200 chars before the match
            start = max(0, match.start() - 200)
            context = content[start : match.end() + 200]

            if "ADVERSARIAL" in context.upper():
                if "\u2705" in context or "PASS" in context.upper():
                    return "VALIDATED", "Adversarial pass found"
                if "\u274c" in context or "FAIL" in context.upper() or "REJECT" in context.upper():
                    return "FAILED", "Adversarial test failed/rejected"
                if "\u23f3" in context or "RUNNING" in context.upper():
                    return "IN_PROGRESS", "Adversarial test running"

            # Check for general status
            if "\U0001f3c6" in context:  # trophy
                return "VALIDATED", "Validated (trophy marker)"
            if "\u274c" in context:
                # Only count as failed if it's an adversarial/backtest context
                if any(kw in context.upper() for kw in ["BACKTEST", "ADVERSARIAL", "FAIL", "REJECT", "DEAD"]):
                    return "FAILED", "Strategy failed validation"

    return "NO_RESULT", "No adversarial result found in SESSION_STATE"


def check_state_file(engine_name):
    """Check if engine has a recent state file."""
    state_files = infer_state_filename(engine_name)

    if not state_files:
        return "MISSING", None, "No state file found"

    # Use the most recently modified one
    best = max(state_files, key=lambda p: p.stat().st_mtime)
    mtime = datetime.fromtimestamp(best.stat().st_mtime)
    age = datetime.now() - mtime

    # Check if file is empty or trivial
    size = best.stat().st_size
    if size < 10:
        return "EMPTY", best.name, f"State file exists but empty/trivial ({size} bytes)"

    # 48h threshold for weekdays (be lenient on weekends)
    today = datetime.now()
    is_weekday = today.weekday() < 5
    threshold = timedelta(hours=48) if is_weekday else timedelta(hours=96)

    if age > threshold:
        return "STALE", best.name, f"Last updated {age.total_seconds()/3600:.0f}h ago"

    return "OK", best.name, f"Updated {age.total_seconds()/3600:.1f}h ago"


def check_cron_entry(engine_name):
    """Check if engine has a crontab entry."""
    try:
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, timeout=5
        )
        crontab = result.stdout
    except Exception:
        return "UNKNOWN", "Could not read crontab"

    base = engine_name.replace(".py", "")
    # Search for the engine name in crontab
    if engine_name in crontab or base in crontab:
        # Find the actual line
        for line in crontab.split("\n"):
            if engine_name in line or base in line:
                if not line.strip().startswith("#"):
                    return "OK", line.strip()[:80]
        return "COMMENTED", "Found but commented out"

    return "MISSING", "No cron entry found"


def check_aggregator_wiring(engine_name):
    """Check if the engine's state file is read by the signal aggregator."""
    if not AGGREGATOR.exists():
        return "UNKNOWN", "Aggregator file not found"

    agg_content = AGGREGATOR.read_text(errors="replace")
    state_files = infer_state_filename(engine_name)

    if not state_files:
        return "UNWIRED", "No state file to wire"

    for sf in state_files:
        # Check if the state file name appears in the aggregator
        if sf.name in agg_content or sf.stem in agg_content:
            return "WIRED", f"Found {sf.name} in aggregator"

    # Also check for partial matches (e.g., variable names)
    base = engine_name.replace(".py", "").replace("_paper_engine", "").replace("_paper", "")
    base_upper = base.upper().replace("_", "")
    if base_upper in agg_content.upper().replace("_", ""):
        return "LIKELY_WIRED", "Probable match in aggregator (partial)"

    return "UNWIRED", f"State file(s) not referenced in aggregator"


def severity_for_gap(check_type, status, adversarial_status):
    """Determine gap severity."""
    if check_type == "adversarial":
        if status == "NO_RESULT":
            return "HIGH"
        if status == "IN_PROGRESS":
            return "LOW"
        return None  # VALIDATED or FAILED = no gap

    if check_type == "state_file":
        if status == "MISSING":
            return "MEDIUM"
        if status == "EMPTY":
            return "MEDIUM"
        if status == "STALE":
            # Only a gap if the engine is supposed to be active
            if adversarial_status in ("VALIDATED", "NO_RESULT", "IN_PROGRESS"):
                return "MEDIUM"
            return "LOW"
        return None

    if check_type == "cron":
        if status == "MISSING":
            # Only high if engine is validated
            if adversarial_status == "VALIDATED":
                return "HIGH"
            return "MEDIUM"
        return None

    if check_type == "aggregator":
        if status == "UNWIRED":
            if adversarial_status == "VALIDATED":
                return "HIGH"
            return "LOW"
        return None

    return None


def action_for_gap(check_type, status, engine_name):
    """Suggest action for each gap type."""
    actions = {
        ("adversarial", "NO_RESULT"): "Run adversarial backtest",
        ("adversarial", "IN_PROGRESS"): "Check if adversarial test is still running",
        ("state_file", "MISSING"): "Verify engine runs correctly; check logs",
        ("state_file", "EMPTY"): "Engine may be failing silently; check logs",
        ("state_file", "STALE"): "Verify cron is running and engine isn't crashing",
        ("cron", "MISSING"): "Add cron entry for daily execution",
        ("cron", "COMMENTED"): "Uncomment cron entry or remove if deprecated",
        ("aggregator", "UNWIRED"): "Wire state file into agentic_signal_aggregator.py",
    }
    return actions.get((check_type, status), f"Investigate {check_type} issue")


def run_audit():
    """Main audit function."""
    engines = find_paper_engines()
    gaps = []
    engine_summaries = {}

    for engine in engines:
        summary = {"engine": engine, "checks": {}}

        # 1. Adversarial status
        adv_status, adv_detail = check_adversarial_status(engine)
        summary["checks"]["adversarial"] = {"status": adv_status, "detail": adv_detail}

        # 2. State file
        sf_status, sf_name, sf_detail = check_state_file(engine)
        summary["checks"]["state_file"] = {
            "status": sf_status,
            "file": sf_name,
            "detail": sf_detail,
        }

        # 3. Cron entry
        cr_status, cr_detail = check_cron_entry(engine)
        summary["checks"]["cron"] = {"status": cr_status, "detail": cr_detail}

        # 4. Aggregator wiring
        ag_status, ag_detail = check_aggregator_wiring(engine)
        summary["checks"]["aggregator"] = {"status": ag_status, "detail": ag_detail}

        engine_summaries[engine] = summary

        # Generate gaps
        for check_type, check_data in summary["checks"].items():
            status = check_data["status"]
            sev = severity_for_gap(check_type, status, adv_status)
            if sev:
                gaps.append(
                    {
                        "engine": engine,
                        "issue": check_data.get("detail", f"{check_type}: {status}"),
                        "check_type": check_type,
                        "severity": sev,
                        "action": action_for_gap(check_type, status, engine),
                    }
                )

    # Classify engines
    fully_complete = 0
    for engine, s in engine_summaries.items():
        adv = s["checks"]["adversarial"]["status"]
        sf = s["checks"]["state_file"]["status"]
        cr = s["checks"]["cron"]["status"]
        ag = s["checks"]["aggregator"]["status"]

        if adv in ("VALIDATED", "FAILED") and sf == "OK" and cr == "OK" and ag in ("WIRED", "LIKELY_WIRED"):
            fully_complete += 1
        elif adv == "FAILED":
            # Failed strategies are "complete" in the sense we've decided on them
            fully_complete += 1

    critical = sum(1 for g in gaps if g["severity"] == "HIGH")
    high_gaps = [g for g in gaps if g["severity"] == "HIGH"]

    report = {
        "timestamp": datetime.now().isoformat(),
        "gaps": gaps,
        "summary": {
            "total_engines": len(engines),
            "fully_complete": fully_complete,
            "gaps_found": len(gaps),
            "critical_gaps": critical,
        },
        "engine_details": engine_summaries,
    }

    # Write report
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"=== COMPLETION CHECK ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===")
    print(f"Total engines: {len(engines)}")
    print(f"Fully complete: {fully_complete}")
    print(f"Gaps found: {len(gaps)}")
    print(f"Critical (HIGH): {critical}")
    print()

    if high_gaps:
        print("CRITICAL_GAPS DETECTED:")
        print()
        for g in high_gaps:
            print(f"  [{g['severity']}] {g['engine']}")
            print(f"    Issue: {g['issue']}")
            print(f"    Action: {g['action']}")
            print()

    medium_gaps = [g for g in gaps if g["severity"] == "MEDIUM"]
    if medium_gaps:
        print(f"MEDIUM gaps ({len(medium_gaps)}):")
        for g in medium_gaps:
            print(f"  - {g['engine']}: {g['issue']}")
        print()

    low_gaps = [g for g in gaps if g["severity"] == "LOW"]
    if low_gaps:
        print(f"LOW gaps ({len(low_gaps)}):")
        for g in low_gaps[:10]:
            print(f"  - {g['engine']}: {g['issue']}")
        if len(low_gaps) > 10:
            print(f"  ... and {len(low_gaps) - 10} more")
        print()

    # Plain-English prompt for Claude if critical gaps exist
    if critical > 0:
        print("--- CLAUDE PROMPT ---")
        print(f"The completion checker found {critical} critical gaps in the pipeline.")
        print("Engines with HIGH severity issues need immediate attention:")
        for g in high_gaps:
            print(f"- {g['engine']}: {g['action']}")
        print("Fix these gaps now. Do not wait for user approval (HC #393).")
        print("--- END PROMPT ---")

    print(f"\nFull report: {OUTPUT}")
    return report


if __name__ == "__main__":
    run_audit()
