#!/usr/bin/env python3
"""HC #590 P1-4: Daily NAV snapshot for all paper lanes.

State files overwrite in place, so without this no equity time series
accumulates for ETF rotation or megacap K=6. (Wheel keeps its own
equity.csv but we snapshot it too for one unified history.)

Appends one row per lane per run to data/nav_history.csv:
    date,ts_utc,lane,nav_usd,cumulative_realized_pnl
Idempotent per (date, lane): skips if today's row already written.

Cron: 16:30 ET weekdays.
"""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant/live_trading_linux")
OUT = ROOT / "data" / "nav_history.csv"
LOG = Path("/home/jupiter/Lvl3Quant/logs/nav_snapshot.log")

LANES = {
    # Original lanes
    "etf_rotation": ROOT / "data" / "etf_rotation_paper_state.json",
    "megacap_k6": ROOT / "data" / "megacap_paper_state.json",
    "wheel": ROOT / "wheel_paper_state" / "state.json",
    "wheel_v4": ROOT / "wheel_v4_state" / "state.json",
    "wheel_balanced": ROOT / "wheel_paper_balanced_state" / "state.json",
    "wheel_diversified": ROOT / "wheel_diversified_state" / "state.json",
    "lh_2h": ROOT / "lh_2h_paper_state" / "state.json",
    # New lanes (added 2026-07-10)
    "wheel_v5": ROOT / "wheel_v5_state" / "state.json",
    "wheel_ic": ROOT / "wheel_ic_state" / "state.json",
    "wheel_bps": ROOT / "wheel_bps_state" / "state.json",
    "wheel_bps_conservative": ROOT / "wheel_bps_conservative_state" / "state.json",
    "wheel_bps_ga": ROOT / "wheel_bps_ga_state" / "state.json",
    "etf_rotation_v2": ROOT / "etf_rotation_v2_state" / "state.json",
    "etf_rotation_v3": ROOT / "etf_rotation_v3_state" / "state.json",
    # HC #725/#726 research engines (added 2026-07-22)
    "trend_cta": Path("/home/jupiter/Lvl3Quant/data/paper_engines/trend_cta/state.json"),
    "jade_lizard": Path("/home/jupiter/Lvl3Quant/data/paper_engines/jade_lizard/state.json"),
    "ml_stock_ranker": ROOT / "state" / "ml_stock_ranker_state.json",
    "ml_asymmetric_ranker": Path("/home/jupiter/Lvl3Quant/data/paper_engines/ml_asymmetric_ranker/state.json"),
}


def wheel_nav(d: dict) -> float:
    """Wheel state has no nav_usd; derive from equity_curve tail or cash+positions.

    For CSP engines (V4, V5, etc.), NAV = cash + sum(margin_held) for short puts,
    plus share value for assigned positions.
    """
    # 1. Try equity_curve (original wheel format)
    curve = d.get("equity_curve") or []
    if curve:
        last = curve[-1]
        if isinstance(last, (list, tuple)) and len(last) >= 2:
            return float(last[1])
        if isinstance(last, dict):
            for k in ("equity", "nav", "value"):
                if k in last:
                    return float(last[k])
        if isinstance(last, (int, float)):
            return float(last)

    # 2. For CSP engines: NAV = cash + sum(margin_held for short_puts)
    #    This is an approximation since we don't have live prices here.
    nav = float(d.get("cash", 0.0))
    for pos in d.get("positions", []):
        side = pos.get("side", "")
        if side == "short_put":
            nav += float(pos.get("margin_held", 0))
        elif side in ("long_shares", "short_call"):
            # For shares, use strike or share_basis as price proxy
            basis = float(pos.get("share_basis", pos.get("strike", 0)))
            contracts = int(pos.get("contracts", 1))
            nav += basis * 100 * contracts

    # Hedge realized P&L
    hedge = d.get("hedge", {})
    nav += float(hedge.get("realized_pnl", 0.0))

    return nav


def main() -> None:
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if OUT.exists():
        with open(OUT) as f:
            for row in csv.DictReader(f):
                existing.add((row["date"], row["lane"]))

    new_file = not OUT.exists()
    written, errors = [], []
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "ts_utc", "lane", "nav_usd", "cumulative_realized_pnl"])
        for lane, path in LANES.items():
            if (today, lane) in existing:
                continue
            try:
                d = json.loads(path.read_text())

                # Extract NAV based on state format
                if lane.startswith("etf_rotation_v"):
                    # ETF v2/v3: has nav_usd at top level
                    nav = float(d.get("nav_usd", d.get("portfolio_value", 0)))
                    pnl = float(d.get("cumulative_realized_pnl", 0.0))
                elif lane == "wheel_ic":
                    # IC condor: NAV = cash only. Margin is held FROM cash, not additive.
                    # The engine's own compute_nav() and equity.csv use cash as the base,
                    # then subtract unrealized IC liabilities. We use cash as approximation
                    # since we don't have live prices to mark-to-market the open spreads.
                    # DO NOT add margin_held — that double-counts (margin comes from cash).
                    nav = float(d.get("cash", 0.0))
                    pnl = float(d.get("realized_pnl", 0.0))
                elif lane.startswith("wheel"):
                    nav = wheel_nav(d)
                    pnl = float(d.get("realized_pnl", 0.0))
                elif lane == "lh_2h":
                    # 2h ES paper engine: uses 'capital' field
                    nav = float(d.get("capital", 0.0))
                    pnl = float(d.get("total_pnl_dollars", 0.0))
                elif lane in ("trend_cta", "ml_stock_ranker", "ml_asymmetric_ranker"):
                    # Research engines: use 'nav' field
                    nav = float(d.get("nav", d.get("cash", 0.0)))
                    pnl = float(d.get("realized_pnl", 0.0))
                elif lane == "jade_lizard":
                    # Jade lizard: cash + unrealized from open positions
                    nav = float(d.get("cash", 0.0)) + float(d.get("realized_pnl", 0.0))
                    pnl = float(d.get("realized_pnl", 0.0))
                else:
                    nav = float(d["nav_usd"])
                    pnl = float(d.get("cumulative_realized_pnl", 0.0))

                if nav > 0:
                    w.writerow([today, now.strftime("%H:%M:%S"), lane, f"{nav:.2f}", f"{pnl:.2f}"])
                    written.append(lane)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{lane}: {e}")

    with open(LOG, "a") as lf:
        lf.write(f"[{now.isoformat()}] wrote={written} skipped_existing={[l for l in LANES if (today, l) in existing]} errors={errors}\n")
    if errors:
        os.system(
            f"node /home/jupiter/teleclaude-main/utils/webhook_notifier.js "
            f"\"NAV snapshot errors: {'; '.join(errors)}\" 2>/dev/null"
        )


if __name__ == "__main__":
    main()
