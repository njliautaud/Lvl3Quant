#!/usr/bin/env python3
"""
Prediction Markets Execution Scheduler

Automated trading loop that orchestrates:
  1. Morning scan (8:00 AM ET) — Full strategy scan, identify opportunities
  2. Intraday MM cycles (every 5 min during market hours) — Quote, manage inventory
  3. Evening settle (4:30 PM ET) — Mark positions, report daily P&L
  4. Overnight scan (8:00 PM ET) — Scan for overnight opportunities

Can run standalone or be called from the Node.js scheduler.

Usage:
    python execution_scheduler.py --mode paper              # Paper trading
    python execution_scheduler.py --mode paper --scan-only   # Scan only, no trading
    python execution_scheduler.py --mode paper --once        # Single cycle
    python execution_scheduler.py --status                   # Show current status
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [exec_scheduler] %(levelname)s: %(message)s",
)
log = logging.getLogger("exec_scheduler")

# ---------------------------------------------------------------------------
# Imports (with graceful fallbacks)
# ---------------------------------------------------------------------------

from polymarket_client import PolymarketScanner
from market_maker import MarketMaker, MMConfig, load_state

try:
    from daily_scanner import run_scan as run_daily_scan, format_discord_report
    HAS_DAILY_SCANNER = True
except ImportError:
    HAS_DAILY_SCANNER = False

try:
    from strategy_engine import StrategyEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def get_et_hour() -> float:
    """Get current hour in Eastern Time as float (e.g., 14.5 = 2:30 PM)."""
    try:
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        # Fallback: assume UTC-5 (EST) or UTC-4 (EDT)
        utc = datetime.now(timezone.utc)
        # Simple DST check (Mar second Sunday to Nov first Sunday)
        month = utc.month
        is_dst = 3 < month < 11 or (month == 3 and utc.day > 14) or (month == 11 and utc.day < 7)
        offset = timedelta(hours=-4 if is_dst else -5)
        et = utc + offset
    return et.hour + et.minute / 60.0


def is_market_hours() -> bool:
    """Check if US markets are open (9:30 AM - 4:00 PM ET)."""
    h = get_et_hour()
    return 9.5 <= h < 16.0


def is_pre_market() -> bool:
    h = get_et_hour()
    return 7.0 <= h < 9.5


def is_after_hours() -> bool:
    h = get_et_hour()
    return 16.0 <= h < 20.0


# ---------------------------------------------------------------------------
# Scheduler State
# ---------------------------------------------------------------------------

SCHED_STATE_FILE = DATA_DIR / "exec_scheduler_state.json"


def load_sched_state() -> dict:
    if SCHED_STATE_FILE.exists():
        try:
            with open(SCHED_STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "last_morning_scan": None,
        "last_evening_report": None,
        "last_overnight_scan": None,
        "today": None,
        "morning_opportunities": [],
        "daily_trades": 0,
        "daily_pnl": 0.0,
        "status": "idle",
    }


def save_sched_state(state: dict):
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(SCHED_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Discord posting
# ---------------------------------------------------------------------------

def post_discord(message: str, channel: str = "prediction-markets"):
    sidecar = THIS_DIR.parents[1] / ".discord_pending.json"
    try:
        pending = []
        if sidecar.exists():
            try:
                with open(sidecar) as f:
                    pending = json.load(f)
            except Exception:
                pending = []
        pending.append({
            "channel": channel,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        with open(sidecar, "w") as f:
            json.dump(pending, f, indent=2)
    except Exception as e:
        log.warning(f"Discord post failed: {e}")


# ---------------------------------------------------------------------------
# Scheduled tasks
# ---------------------------------------------------------------------------

def morning_scan(sched_state: dict, spx: float = 5900.0) -> dict:
    """Run full morning scan before market open."""
    log.info("=== MORNING SCAN ===")
    results = {"scan": None, "opportunities": [], "errors": []}

    # Run daily scanner if available
    if HAS_DAILY_SCANNER:
        try:
            scan_data = run_daily_scan(spx=spx, hours=6.5, run_engine=True, quiet=True)
            results["scan"] = scan_data

            # Extract actionable opportunities
            poly = scan_data.get("polymarket", {})
            if poly.get("status") == "OK":
                poly_result = poly.get("result", {})
                results["opportunities"].extend(poly_result.get("opportunities", []))

            engine = scan_data.get("engine", {})
            if engine.get("status") == "OK":
                top = engine.get("result", {}).get("top_opportunities", [])
                results["opportunities"].extend(top)

            # Post report
            report = format_discord_report(scan_data)
            post_discord(report)

        except Exception as e:
            log.error(f"Daily scanner failed: {e}")
            results["errors"].append(str(e))
    else:
        # Fallback: just run Polymarket scanner
        try:
            scanner = PolymarketScanner()
            opps = scanner.scan(limit=200, min_net_edge=0.015)
            results["opportunities"] = opps
            post_discord(
                f"**MORNING SCAN** ({datetime.now().strftime('%H:%M ET')})\n"
                f"Found {len(opps)} Polymarket opportunities with edge >1.5c"
            )
        except Exception as e:
            results["errors"].append(str(e))

    sched_state["last_morning_scan"] = datetime.now(timezone.utc).isoformat()
    sched_state["morning_opportunities"] = [
        {k: v for k, v in opp.items() if k in
         ("question", "market_slug", "net_edge", "action", "side", "market_price", "fair_value")}
        for opp in results["opportunities"][:20]
    ]

    log.info(f"Morning scan complete: {len(results['opportunities'])} opportunities")
    return results


def run_mm_cycle(mm: MarketMaker, sched_state: dict) -> dict:
    """Run one market making cycle."""
    report = mm.run_cycle()
    sched_state["daily_trades"] += len([
        a for a in report.get("actions", [])
        if "DIRECTIONAL" in a or "MM_QUOTE" in a
    ])
    sched_state["daily_pnl"] = report.get("daily_pnl", 0)
    return report


def evening_report(mm: MarketMaker, sched_state: dict):
    """Generate end-of-day report."""
    log.info("=== EVENING REPORT ===")

    mm_state = load_state()

    lines = [
        f"**PREDICTION MARKETS EOD REPORT** ({datetime.now().strftime('%Y-%m-%d')})",
        "",
        f"Daily P&L: ${mm_state.daily_pnl:.2f}",
        f"Total P&L: ${mm_state.total_pnl:.2f}",
        f"Positions: {len(mm_state.positions)}",
        f"Total exposure: ${mm_state.total_exposure:.0f}",
        f"Cycles run: {mm_state.cycle_count}",
        f"Trades today: {sched_state.get('daily_trades', 0)}",
    ]

    if mm_state.positions:
        lines.append("\n**Open Positions:**")
        for slug, pos in mm_state.positions.items():
            lines.append(
                f"  {slug[:35]}: {pos.side} ${pos.size_usdc:.0f} "
                f"@ {pos.avg_entry:.3f} (PnL: ${pos.total_pnl:.2f})"
            )

    # Recent trades
    recent = mm_state.trade_log[-5:]
    if recent:
        lines.append("\n**Recent Trades:**")
        for t in recent:
            lines.append(
                f"  {t.get('action', '?')} {t.get('side', '?')} "
                f"'{t.get('question', '?')[:30]}' @ {t.get('price', 0):.3f} "
                f"(edge: {t.get('edge', 0):.3f})"
            )

    report_msg = "\n".join(lines)
    post_discord(report_msg)

    sched_state["last_evening_report"] = datetime.now(timezone.utc).isoformat()
    log.info("Evening report posted")


# ---------------------------------------------------------------------------
# Main scheduler loop
# ---------------------------------------------------------------------------

def run_scheduler(mode: str = "paper", scan_only: bool = False,
                  spx: float = 5900.0, once: bool = False):
    """Main scheduler loop."""
    log.info(f"Execution Scheduler starting (mode={mode}, scan_only={scan_only})")

    sched_state = load_sched_state()
    today = datetime.now().strftime("%Y-%m-%d")

    # Reset daily counters if new day
    if sched_state.get("today") != today:
        sched_state["today"] = today
        sched_state["daily_trades"] = 0
        sched_state["daily_pnl"] = 0.0
        sched_state["morning_opportunities"] = []

    # Initialize market maker
    config = MMConfig(mode=mode)
    mm = MarketMaker(config)

    sched_state["status"] = "running"
    save_sched_state(sched_state)

    # Track what we've done today
    did_morning = (sched_state.get("last_morning_scan") or "")[:10] == today
    did_evening = (sched_state.get("last_evening_report") or "")[:10] == today

    try:
        # In --once mode, always run a scan regardless of time-of-day
        if once and not did_morning:
            log.info("Single-cycle mode: running morning scan regardless of time")
            scan_results = morning_scan(sched_state, spx=spx)
            did_morning = True
            save_sched_state(sched_state)

            # Print a summary to stdout for CLI usage
            opps = scan_results.get("opportunities", [])
            errors = scan_results.get("errors", [])
            print(f"\n{'='*60}")
            print(f"  PREDICTION MARKETS SCAN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            print(f"{'='*60}")
            print(f"  Opportunities found: {len(opps)}")
            if errors:
                print(f"  Errors: {len(errors)}")
                for e in errors:
                    print(f"    - {e[:80]}")
            if opps:
                print(f"\n  Top opportunities:")
                for i, opp in enumerate(opps[:10], 1):
                    q = opp.get("question", opp.get("ticker", opp.get("group", "?")))
                    edge = opp.get("net_edge_after_fees", opp.get("net_edge", 0))
                    strat = opp.get("strategy", "?")
                    action = opp.get("action", "?")
                    print(f"    {i}. {str(q)[:55]}")
                    print(f"       Strategy: {strat} | Edge: {edge:.4f} | Action: {action}")
            else:
                print("  No actionable opportunities — markets appear fairly priced.")
            print(f"{'='*60}\n")

            # Run one MM cycle if not scan-only
            if not scan_only:
                report = run_mm_cycle(mm, sched_state)
                save_sched_state(sched_state)

            log.info("Single cycle complete, exiting.")

        else:
            while True:
                et_hour = get_et_hour()
                today = datetime.now().strftime("%Y-%m-%d")

                # Reset on new day
                if sched_state.get("today") != today:
                    sched_state["today"] = today
                    sched_state["daily_trades"] = 0
                    sched_state["daily_pnl"] = 0.0
                    did_morning = False
                    did_evening = False
                    # Reset MM daily loss
                    mm.state.daily_pnl = 0.0
                    mm.state.is_stopped = False

                # Morning scan (7:30-9:30 AM ET, once per day)
                if 7.5 <= et_hour < 9.5 and not did_morning:
                    morning_scan(sched_state, spx=spx)
                    did_morning = True
                    save_sched_state(sched_state)

                # MM cycles — Polymarket is 24/7, so always run
                if not scan_only:
                    report = run_mm_cycle(mm, sched_state)

                    # Only post to Discord every 6th cycle (every 30 min) to avoid spam
                    if mm.state.cycle_count % 6 == 0:
                        msg = mm.format_cycle_report(report)
                        post_discord(msg)

                    save_sched_state(sched_state)

                # Evening report (4:15-4:45 PM ET, once per day)
                if 16.25 <= et_hour < 16.75 and not did_evening:
                    evening_report(mm, sched_state)
                    did_evening = True
                    save_sched_state(sched_state)

                # Sleep until next cycle — Polymarket is 24/7
                sleep_time = config.cycle_interval_seconds  # Default 5 min

                log.info(f"Next cycle in {sleep_time}s (ET hour: {et_hour:.1f})")
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        log.info("Scheduler interrupted")
    finally:
        sched_state["status"] = "stopped"
        save_sched_state(sched_state)
        log.info("Scheduler stopped")


def show_status():
    """Show current scheduler and MM status."""
    sched = load_sched_state()
    mm = load_state()

    print("\n" + "=" * 60)
    print("  PREDICTION MARKETS EXECUTION STATUS")
    print("=" * 60)
    print(f"  Scheduler: {sched.get('status', 'unknown')}")
    print(f"  Last updated: {sched.get('last_updated', 'never')}")
    print(f"  Today: {sched.get('today', 'N/A')}")
    last_morning = sched.get('last_morning_scan') or ''
    print(f"  Morning scan: {'Done' if last_morning[:10] == datetime.now().strftime('%Y-%m-%d') else 'Pending'}")
    print(f"  Daily trades: {sched.get('daily_trades', 0)}")
    print(f"  Daily P&L: ${sched.get('daily_pnl', 0):.2f}")
    print()
    print(f"  MM Cycles: {mm.cycle_count}")
    print(f"  MM Total P&L: ${mm.total_pnl:.2f}")
    print(f"  MM Exposure: ${mm.total_exposure:.0f}")
    print(f"  Positions: {len(mm.positions)}")
    print(f"  Open orders: {len(mm.open_orders)}")
    print(f"  MM Stopped: {mm.is_stopped} ({mm.stop_reason})" if mm.is_stopped else "  MM Active")

    if mm.positions:
        print("\n  Open Positions:")
        for slug, pos in mm.positions.items():
            print(f"    {slug[:35]}: {pos.side} ${pos.size_usdc:.0f} @ {pos.avg_entry:.3f}")

    opps = sched.get("morning_opportunities", [])
    if opps:
        print(f"\n  Morning Opportunities ({len(opps)}):")
        for opp in opps[:5]:
            print(f"    {opp.get('question', '?')[:45]} | edge={opp.get('net_edge', 0):.3f}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prediction Markets Execution Scheduler")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--scan-only", action="store_true",
                       help="Run scans but don't place trades")
    parser.add_argument("--once", action="store_true",
                       help="Run single cycle then exit")
    parser.add_argument("--status", action="store_true",
                       help="Show current status and exit")
    parser.add_argument("--spx", type=float, default=5900.0,
                       help="Current SPX price for bracket pricing")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    run_scheduler(
        mode=args.mode,
        scan_only=args.scan_only,
        spx=args.spx,
        once=args.once,
    )


if __name__ == "__main__":
    main()
