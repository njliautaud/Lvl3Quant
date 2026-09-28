#!/usr/bin/env python3
"""
Paper Trading Dashboard — Unified view of all paper engines.
Shows P&L, positions, and status for every active paper strategy.

Usage:
    python3 scripts/paper_dashboard.py [--discord]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

STATE_DIR = Path("/home/jupiter/Lvl3Quant/state")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def format_pnl(val: float) -> str:
    if val >= 0:
        return f"+${val:,.0f}"
    return f"-${abs(val):,.0f}"


def dashboard():
    """Build unified paper trading dashboard."""
    lines = []
    lines.append(f"Paper Trading Dashboard — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    lines.append("=" * 60)

    total_capital = 0
    total_pnl = 0

    # ── ETF Rotation v3 ──
    d = read_json(STATE_DIR / "etf_rotation_v3_paper_state.json")
    if d:
        pv = d.get("portfolio_value", d.get("capital", 0))
        cap = d.get("capital", 100000)
        ret = d.get("total_return_pct", 0)
        positions = d.get("positions", {})
        pos_str = ", ".join(f"{k} ({v.get('pnl_pct', 0):+.1f}%)" for k, v in positions.items()) if positions else "no positions"
        lines.append(f"\n📈 ETF Rotation v3: ${pv:,.0f} ({ret:+.1f}%) — {pos_str}")
        total_capital += cap
        total_pnl += pv - cap

    # ── Iron Condor Paper ──
    d = read_json(STATE_DIR / "ic_paper_state.json")
    if d and d.get("positions"):
        positions = d["positions"]
        cash = d.get("cash", 0)
        unrealized = sum(p.get("unrealized_pnl", 0) for p in positions.values())
        n_pos = len(positions)
        margin = sum(p.get("margin_dollar", 0) for p in positions.values())
        lines.append(f"\n🦅 Iron Condors: {n_pos} positions, {format_pnl(unrealized)} unrealized, ${cash:,.0f} cash")
        # Show worst 3
        sorted_pos = sorted(positions.values(), key=lambda p: p.get("unrealized_pnl", 0))
        for p in sorted_pos[:3]:
            lines.append(f"   {p['ticker']}: {format_pnl(p.get('unrealized_pnl', 0))} ({p.get('pnl_pct', 0):+.1f}%) — {p.get('action', '?')}")
        total_pnl += unrealized

    # ── Contrarian Signals ──
    d = read_json(STATE_DIR / "contrarian_signals_state.json")
    if d and d.get("positions"):
        positions = d["positions"]
        pnl = sum(p.get("pnl_dollar", 0) for p in positions.values())
        n_pos = len(positions)
        lines.append(f"\n🔄 Contrarian Signals: {n_pos} positions, {format_pnl(pnl)} total")
        for p in positions.values():
            lines.append(f"   {p['ticker']} ({p['signal']}): {p.get('days_held', 0)}d/{p['hold_days']}d, {format_pnl(p.get('pnl_dollar', 0))}")
        total_pnl += pnl

    # ── Covered Call Paper ──
    d = read_json(STATE_DIR / "covered_call_paper_state.json")
    if d:
        income = d.get("total_premium_collected", 0)
        lines.append(f"\n📞 Covered Calls: ${income:,.0f} premium collected")

    # ── VIX Contango Paper ──
    d = read_json(STATE_DIR / "vix_contango_paper_state.json")
    if d:
        pnl = d.get("total_pnl", 0)
        trades = d.get("total_trades", 0)
        lines.append(f"\n📊 VIX Contango: {trades} trades, {format_pnl(pnl)}")

    # ── Play Scanner ──
    d = read_json(STATE_DIR / "play_scanner_state.json")
    if d:
        setups = d.get("setups_found", 0)
        top = d.get("top_5", [])
        vix = d.get("vix", {}).get("vix", "?")
        if top:
            top_str = ", ".join(f"{t['ticker']} ({t.get('best_setup', {}).get('type', '?').split('_')[0]})" for t in top[:3])
        else:
            top_str = "none"
        lines.append(f"\n🔍 Scanner: {setups} setups found, VIX {vix} — top: {top_str}")

    # ── Summary ──
    lines.append(f"\n{'=' * 60}")
    lines.append(f"Total paper PnL: {format_pnl(total_pnl)}")

    return "\n".join(lines)


if __name__ == "__main__":
    output = dashboard()
    print(output)

    if "--discord" in sys.argv:
        import re
        import urllib.request
        try:
            wf = Path("/home/jupiter/teleclaude-main/API_KEYS.md")
            if wf.exists():
                for line in wf.read_text().split('\n'):
                    if 'discord' in line.lower() and 'webhook' in line.lower() and 'http' in line:
                        urls = re.findall(r'https://discord\.com/api/webhooks/\S+', line)
                        if urls:
                            url = urls[0].strip('`').strip()
                            data = json.dumps({"content": output[:2000]}).encode()
                            req = urllib.request.Request(url, data=data,
                                                        headers={"Content-Type": "application/json"},
                                                        method="POST")
                            urllib.request.urlopen(req, timeout=10)
                            print("\n[Discord alert sent]")
        except Exception as e:
            print(f"\n[Discord failed: {e}]")
