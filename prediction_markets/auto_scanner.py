#!/usr/bin/env python3
"""
Automated Prediction Market Scanner — Runs on a schedule to build price history.

Designed to be called every 30 minutes via cron/Task Scheduler.
Records price snapshots for the odds scalper and optionally runs LLM edge analysis.

Usage:
    python auto_scanner.py                          # Full scan (scalper + LLM if key available)
    python auto_scanner.py --scalper-only            # Price recording only (no LLM)
    python auto_scanner.py --llm-only                # LLM edge scan only
    python auto_scanner.py --limit 500               # Fetch more markets (paginated)

Cron example (every 30 minutes):
    */30 * * * * cd /path/to/prediction_markets && python3 auto_scanner.py >> /var/log/auto_scanner.log 2>&1

Windows Task Scheduler:
    Program: python
    Arguments: C:\path\to\prediction_markets\auto_scanner.py
    Start in: C:\path\to\prediction_markets
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SUMMARY_DIR = DATA_DIR / "daily_summaries"
SUMMARY_DIR.mkdir(exist_ok=True)

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Minimal logging — one-line format for cron log readability
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto_scanner")


# ---------------------------------------------------------------------------
# Scalper scan — records prices for ALL tradeable markets
# ---------------------------------------------------------------------------

def run_scalper_scan(limit: int = 200) -> dict:
    """
    Run the odds scalper price-recording scan.
    Returns a summary dict with counts and any signals found.
    """
    try:
        from odds_scalper import OddsScalper
        scalper = OddsScalper()
        result = scalper.scan_all(limit=limit)
        return {
            "status": "OK",
            "scanned": result.get("scanned", 0),
            "total_markets": result.get("total_markets", 0),
            "mean_reversion": result.get("mean_reversion_signals", 0),
            "momentum": result.get("momentum_signals", 0),
            "wide_spreads": result.get("wide_spread_markets", 0),
            "opportunities": len(result.get("opportunities", [])),
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# ---------------------------------------------------------------------------
# LLM edge scan — only runs if API key is available
# ---------------------------------------------------------------------------

def run_llm_scan(limit: int = 200) -> dict:
    """
    Run the LLM edge scanner if an API key is available.
    Returns a summary dict.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    # Also check .env file
    env_file = THIS_DIR.parents[1] / ".env"
    if env_file.exists() and not (groq_key or anthropic_key):
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                if k.strip() == "GROQ_API_KEY" and not groq_key:
                    groq_key = v
                    os.environ["GROQ_API_KEY"] = v
                elif k.strip() == "ANTHROPIC_API_KEY" and not anthropic_key:
                    anthropic_key = v
                    os.environ["ANTHROPIC_API_KEY"] = v
        except Exception:
            pass

    if not groq_key and not anthropic_key:
        return {
            "status": "SKIPPED",
            "reason": "No GROQ_API_KEY or ANTHROPIC_API_KEY available",
        }

    try:
        from llm_edge_scanner import LLMEdgeScanner
        scanner = LLMEdgeScanner()
        result = scanner.full_scan(limit=limit)

        if "error" in result and not result.get("confirmed_opportunities"):
            return {
                "status": "NO_KEY",
                "reason": result.get("error", "unknown"),
                "tradeable": result.get("tradeable_count", 0),
            }

        return {
            "status": "OK",
            "tradeable": result.get("tradeable", 0),
            "candidates": result.get("candidates_screened", 0),
            "deep_analyzed": result.get("deep_analyzed", 0),
            "confirmed": result.get("confirmed_opportunities", 0),
            "duration_s": result.get("duration_seconds", 0),
            "categories": result.get("category_breakdown", {}),
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# ---------------------------------------------------------------------------
# Daily summary file — append results for tracking over time
# ---------------------------------------------------------------------------

def append_to_daily_summary(scan_result: dict):
    """Append a scan result to today's summary file (JSONL format)."""
    today = datetime.now().strftime("%Y-%m-%d")
    summary_file = SUMMARY_DIR / f"scan_{today}.jsonl"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **scan_result,
    }

    try:
        with open(summary_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        log.error(f"Failed to write summary: {e}")


def get_daily_stats() -> dict:
    """Read today's summary and compute stats."""
    today = datetime.now().strftime("%Y-%m-%d")
    summary_file = SUMMARY_DIR / f"scan_{today}.jsonl"

    if not summary_file.exists():
        return {"scans_today": 0}

    entries = []
    try:
        with open(summary_file) as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except Exception:
                    continue
    except Exception:
        return {"scans_today": 0}

    scalper_runs = [e for e in entries if e.get("scalper", {}).get("status") == "OK"]
    llm_runs = [e for e in entries if e.get("llm", {}).get("status") == "OK"]

    total_opps = sum(e.get("scalper", {}).get("opportunities", 0) for e in entries)
    total_confirmed = sum(e.get("llm", {}).get("confirmed", 0) for e in entries)

    return {
        "scans_today": len(entries),
        "scalper_runs": len(scalper_runs),
        "llm_runs": len(llm_runs),
        "total_scalp_signals": total_opps,
        "total_llm_confirmed": total_confirmed,
    }


# ---------------------------------------------------------------------------
# One-line summary for logging
# ---------------------------------------------------------------------------

def format_oneline(result: dict) -> str:
    """Format scan result as a single line for log output."""
    ts = datetime.now().strftime("%H:%M")
    parts = [f"[{ts}]"]

    scalper = result.get("scalper", {})
    if scalper.get("status") == "OK":
        parts.append(
            f"SCALP: {scalper['scanned']}/{scalper['total_markets']} mkts"
            f" | mr={scalper['mean_reversion']} mom={scalper['momentum']}"
            f" | opps={scalper['opportunities']}"
        )
    elif scalper.get("status") == "ERROR":
        parts.append(f"SCALP: ERROR ({scalper.get('error', '?')[:50]})")
    elif scalper.get("status") == "SKIPPED":
        parts.append("SCALP: skipped")

    llm = result.get("llm", {})
    if llm.get("status") == "OK":
        parts.append(
            f"LLM: {llm['tradeable']} tradeable"
            f" -> {llm['candidates']} candidates"
            f" -> {llm['confirmed']} confirmed"
            f" ({llm['duration_s']:.0f}s)"
        )
    elif llm.get("status") == "SKIPPED":
        parts.append("LLM: skipped (no key)")
    elif llm.get("status") == "NO_KEY":
        parts.append(f"LLM: no key ({llm.get('tradeable', 0)} tradeable)")
    elif llm.get("status") == "ERROR":
        parts.append(f"LLM: ERROR ({llm.get('error', '?')[:50]})")

    duration = result.get("duration_s", 0)
    parts.append(f"[{duration:.1f}s]")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated prediction market scanner (cron-friendly)",
    )
    parser.add_argument(
        "--scalper-only", action="store_true",
        help="Only run price-recording scalper scan (no LLM)",
    )
    parser.add_argument(
        "--llm-only", action="store_true",
        help="Only run LLM edge scan (no price recording)",
    )
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Max markets to fetch from Polymarket (default: 500, paginated)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show today's scan statistics and exit",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    if args.stats:
        stats = get_daily_stats()
        print(json.dumps(stats, indent=2))
        return

    t_start = time.time()
    result = {}

    # Run scalper scan
    if not args.llm_only:
        try:
            result["scalper"] = run_scalper_scan(limit=args.limit)
        except Exception as e:
            result["scalper"] = {"status": "ERROR", "error": str(e)}
    else:
        result["scalper"] = {"status": "SKIPPED"}

    # Run LLM scan
    if not args.scalper_only:
        try:
            result["llm"] = run_llm_scan(limit=args.limit)
        except Exception as e:
            result["llm"] = {"status": "ERROR", "error": str(e)}
    else:
        result["llm"] = {"status": "SKIPPED"}

    result["duration_s"] = round(time.time() - t_start, 1)

    # Append to daily summary
    append_to_daily_summary(result)

    # Print one-line summary
    print(format_oneline(result))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(130)
    except Exception as e:
        # Never crash silently in cron — always print the error
        print(f"FATAL: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
