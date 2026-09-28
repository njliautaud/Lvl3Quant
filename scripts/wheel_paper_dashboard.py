#!/usr/bin/env python3
"""
Wheel Paper Engine Dashboard
==============================
Aggregates all wheel paper engines into a single status report.
Run manually or from cron to track which variant is performing best.

Usage: python3 scripts/wheel_paper_dashboard.py [--json] [--discord]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "live_trading_linux"

ENGINES = {
    "V4 (CSP base)":    "wheel_v4_state",
    "V5 (income)":      "wheel_v5_state",
    "BPS $10":          "wheel_bps_state",
    "IC (iron condor)": "wheel_ic_state",
    "Diversified 20":   "wheel_diversified_state",
    "SPY base":         "wheel_paper_state",
    "Balanced":         "wheel_paper_balanced_state",
}

STARTING_CAPITAL = 100_000.0


def load_engine(name: str, state_dir_name: str) -> dict:
    state_file = STATE_DIR / state_dir_name / "state.json"
    nav_file = STATE_DIR / state_dir_name / "nav_history.json"
    sod_file = STATE_DIR / state_dir_name / "sod_nav.json"

    result = {"name": name, "status": "unknown"}

    if not state_file.exists():
        result["status"] = "NO STATE FILE"
        return result

    try:
        with open(state_file) as f:
            state = json.load(f)
    except Exception as e:
        result["status"] = f"ERROR: {e}"
        return result

    positions = state.get("positions", state.get("open_positions", state.get("spreads", [])))
    realized = state.get("realized_pnl", state.get("total_realized", 0))
    trades = state.get("trade_count", state.get("total_trades", state.get("n_trades", 0)))
    start = state.get("start_date", state.get("inception_date", "unknown"))

    # NAV from nav_history (latest entry)
    nav = None
    if nav_file.exists():
        try:
            with open(nav_file) as f:
                hist = json.load(f)
            if hist:
                nav = hist[-1].get("nav")
        except Exception:
            pass

    # SOD nav for today's P&L
    sod_nav = None
    if sod_file.exists():
        try:
            with open(sod_file) as f:
                sod = json.load(f)
            sod_nav = sod.get("nav")
        except Exception:
            pass

    if nav is None:
        # Approximate from unrealized + realized + starting
        unrealized = state.get("unrealized_pnl", 0) or 0
        nav = STARTING_CAPITAL + realized + unrealized

    n_pos = len(positions) if isinstance(positions, (list, dict)) else 0
    pnl_total = nav - STARTING_CAPITAL
    pnl_pct = pnl_total / STARTING_CAPITAL * 100

    # Days active
    days_active = "?"
    if start and start != "unknown":
        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            days_active = (datetime.now() - start_dt.replace(tzinfo=None)).days
        except Exception:
            pass

    today_pnl = None
    if sod_nav and nav:
        today_pnl = nav - sod_nav

    result.update({
        "status": "RUNNING",
        "nav": nav,
        "pnl_total": pnl_total,
        "pnl_pct": pnl_pct,
        "realized": realized,
        "positions": n_pos,
        "trades": trades,
        "days_active": days_active,
        "today_pnl": today_pnl,
        "start": start,
    })
    return result


def main():
    json_mode = "--json" in sys.argv
    discord_mode = "--discord" in sys.argv

    results = []
    for name, state_dir in ENGINES.items():
        results.append(load_engine(name, state_dir))

    if json_mode:
        print(json.dumps(results, indent=2, default=str))
        return

    # Table output
    now = datetime.now().strftime("%Y-%m-%d %H:%M ET")
    lines = []
    lines.append(f"=== WHEEL PAPER ENGINE DASHBOARD ({now}) ===\n")
    lines.append(f"{'Engine':<20} {'NAV':>10} {'P&L%':>7} {'Realized':>10} {'Pos':>4} {'Trades':>6} {'Days':>5}")
    lines.append("-" * 75)

    for r in sorted(results, key=lambda x: x.get("pnl_pct", -999), reverse=True):
        if r["status"] != "RUNNING":
            lines.append(f"{r['name']:<20} {r['status']}")
            continue

        nav_str = f"${r['nav']:,.0f}" if r['nav'] else "N/A"
        pnl_str = f"{r['pnl_pct']:+.2f}%"
        real_str = f"${r['realized']:,.0f}" if isinstance(r['realized'], (int, float)) else "N/A"
        today_str = f" (today: ${r['today_pnl']:+,.0f})" if r.get('today_pnl') is not None else ""

        lines.append(
            f"{r['name']:<20} {nav_str:>10} {pnl_str:>7} {real_str:>10} "
            f"{r['positions']:>4} {r['trades']:>6} {r['days_active']:>5}{today_str}"
        )

    lines.append("")

    # Summary
    running = [r for r in results if r["status"] == "RUNNING"]
    if running:
        best = max(running, key=lambda x: x.get("pnl_pct", -999))
        worst = min(running, key=lambda x: x.get("pnl_pct", -999))
        lines.append(f"Best: {best['name']} ({best['pnl_pct']:+.2f}%)")
        lines.append(f"Worst: {worst['name']} ({worst['pnl_pct']:+.2f}%)")
        lines.append(f"Active engines: {len(running)}/{len(results)}")

    output = "\n".join(lines)
    print(output)

    # Save to file
    out_dir = ROOT / "output" / "wheel_paper_dashboard"
    out_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    with open(out_dir / f"dashboard_{date_str}.txt", "w") as f:
        f.write(output)

    # Also save latest
    with open(out_dir / "latest.txt", "w") as f:
        f.write(output)

    if discord_mode:
        print("\n[Discord output would go here]")


if __name__ == "__main__":
    main()
