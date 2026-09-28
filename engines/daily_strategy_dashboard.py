#!/usr/bin/env python3
"""
Daily Strategy Dashboard — Consolidated view of all paper strategies.

Pulls state from all paper engines, signal watcher, and market data.
Generates a plain-English summary suitable for Discord.

Run daily after market close (16:30 ET) or on-demand.
"""

import json, os, sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
LOG_DIR = ROOT / "logs" / "dashboard"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    """Safely load a JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_gp3_status():
    """Get Gameplan v3 paper status."""
    d = load_json(STATE_DIR / "gameplan_v3_state.json")
    if not d:
        return None

    initial = d.get("initial_capital", 500)
    contributed = d.get("total_contributed", initial)
    v3_val = d.get("v3_value", 0)
    v2_val = d.get("v2_value", 0)
    spy_val = d.get("spy_benchmark", 0)

    # Return on invested capital
    v3_ret = ((v3_val / contributed) - 1) * 100 if contributed > 0 else 0
    v2_ret = ((v2_val / contributed) - 1) * 100 if contributed > 0 else 0

    return {
        "name": "Gameplan v3",
        "value": v3_val,
        "contributed": contributed,
        "return_pct": v3_ret,
        "regime": d.get("v3_regime", "?"),
        "holding": d.get("v3_holding", "?"),
        "in_upro": d.get("v3_in_upro", False),
        "switches": d.get("v3_switches", 0),
        "v2_value": v2_val,
        "v2_return_pct": v2_ret,
        "start_date": d.get("start_date", "?"),
    }


def get_vmr_status():
    """Get Vol Mean Reversion paper status."""
    d = load_json(STATE_DIR / "vol_mean_reversion_paper.json")
    if not d:
        return None

    initial = d.get("initial_value", 10000)
    val = d.get("portfolio_value", 0)
    spy = d.get("spy_benchmark", 0)
    upro = d.get("upro_benchmark", 0)

    ret = ((val / initial) - 1) * 100 if initial > 0 else 0
    spy_ret = ((spy / initial) - 1) * 100 if initial > 0 else 0
    upro_ret = ((upro / initial) - 1) * 100 if initial > 0 else 0

    return {
        "name": "Vol Mean Reversion",
        "value": val,
        "initial": initial,
        "return_pct": ret,
        "regime": d.get("current_regime", "?"),
        "spy_benchmark": spy,
        "spy_return_pct": spy_ret,
        "upro_benchmark": upro,
        "upro_return_pct": upro_ret,
        "trades": len(d.get("trades", [])),
        "days": len(d.get("history", [])),
        "start_date": d.get("start_date", "?"),
    }


def get_signal_status():
    """Get signal watcher current readings."""
    d = load_json(STATE_DIR / "signal_watcher_state.json")
    if not d:
        return None

    return {
        "vix_regime": d.get("vix_regime", "?"),
        "confluence_score": d.get("confluence_score", 0),
        "confluence_regime": d.get("confluence_regime", "?"),
        "breadth_pct": d.get("breadth_pct", 0),
        "protection_status": d.get("protection_status", {}),
        "vix_level": d.get("vix_level", 0),
    }


def get_market_snapshot():
    """Get latest market snapshot."""
    d = load_json(STATE_DIR / "market_snapshot.json")
    if not d:
        return None

    return {
        "date": d.get("market_date", "?"),
        "prices": d.get("prices", {}),
        "daily_change": d.get("daily_change_pct", {}),
        "volatility": d.get("volatility", {}),
    }


def format_dashboard():
    """Format the full dashboard as a plain-English string."""
    now = datetime.now()
    lines = []
    lines.append(f"📊 **Daily Strategy Dashboard** — {now.strftime('%B %d, %Y')}")
    lines.append("")

    # Signal watcher
    sig = get_signal_status()
    if sig:
        prot = sig.get("protection_status", {})
        if isinstance(prot, dict):
            green_count = sum(1 for v in prot.values() if v == "OK" or v is True)
            total_prot = len(prot) if prot else 4
        else:
            # Protection status stored as summary string
            green_count = "?"
            total_prot = "?"

        vix_val = sig.get('vix_level', 0)
        vix_str = f"{vix_val:.1f}" if isinstance(vix_val, (int, float)) else str(vix_val)
        lines.append(f"**Market Signals**: VIX {vix_str}, "
                     f"Confluence {sig['confluence_score']}/3.0 ({sig['confluence_regime']}), "
                     f"Protection {green_count}/{total_prot} green")
        lines.append("")

    # GP3 paper
    gp3 = get_gp3_status()
    if gp3:
        upro_str = "✅ IN UPRO" if gp3["in_upro"] else "⏸️ NOT in UPRO"
        lines.append(f"**Gameplan v3**: ${gp3['value']:,.2f} ({gp3['return_pct']:+.1f}% on ${gp3['contributed']:,.0f} invested)")
        lines.append(f"  Regime: {gp3['regime']} | Holding: {gp3['holding']} | {upro_str}")
        lines.append(f"  vs v2: ${gp3['v2_value']:,.2f} ({gp3['v2_return_pct']:+.1f}%) | Switches: {gp3['switches']}")
        lines.append("")

    # VMR paper
    vmr = get_vmr_status()
    if vmr:
        lines.append(f"**Vol Mean Reversion**: ${vmr['value']:,.2f} ({vmr['return_pct']:+.1f}%)")
        lines.append(f"  Regime: {vmr['regime']} | Day {vmr['days']} | {vmr['trades']} trades")
        lines.append(f"  vs SPY: {vmr['spy_return_pct']:+.1f}% | vs UPRO: {vmr['upro_return_pct']:+.1f}%")
        lines.append("")

    # Combined portfolio view
    if gp3 and vmr:
        # Normalized comparison (both started recently, different bases)
        lines.append("**Portfolio Summary**:")
        lines.append(f"  GP3 paper tracking since {gp3['start_date']}")
        lines.append(f"  VMR paper tracking since {vmr['start_date']}")
        lines.append("")

    # Validated strategies scorecard
    lines.append("**Validated Strategies** (backtest): GP3 Sharpe 2.39, VMR Sharpe 1.47")
    lines.append("**Failed strategies**: 35+ tested, all others rejected by adversarial validation")

    return "\n".join(lines)


def main():
    dashboard = format_dashboard()
    print(dashboard)

    # Save to log
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"dashboard_{today}.txt"
    with open(log_file, "w") as f:
        f.write(dashboard)

    # If --discord flag, output in a format suitable for Discord
    if "--discord" in sys.argv:
        print("\n--- DISCORD MESSAGE ---")
        print(dashboard)

    return dashboard


if __name__ == "__main__":
    main()
