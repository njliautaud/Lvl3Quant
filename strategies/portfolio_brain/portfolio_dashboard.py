"""
Portfolio Dashboard — human-readable summary of all state files.

Shows:
  - Current signals and their agreement
  - Today's trade plan
  - Portfolio heat
  - Strategy agreement matrix
  - Earnings warnings
  - Spread recommendations
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_DIR = os.path.join(BASE_DIR, "state")
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load(filename: str, subdir: str = "state") -> Optional[Dict]:
    d = STATE_DIR if subdir == "state" else DATA_DIR
    path = os.path.join(d, filename)
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def hr(char: str = "=", width: int = 70) -> str:
    return char * width


def section(title: str) -> str:
    return f"\n{hr()}\n  {title}\n{hr()}"


def print_signals(agentic: Dict):
    """Print current signals summary."""
    print(section("CURRENT SIGNALS"))

    equity = agentic.get("account_equity", "?")
    vix = agentic.get("vix_level", "?")
    regime = agentic.get("market_regime", "?")
    print(f"  Account: ${equity}  |  VIX: {vix}  |  Regime: {regime}")
    print()

    signals = agentic.get("signals", [])
    if not signals:
        print("  No active signals.")
        return

    print(f"  {'Ticker':<8} {'Dir':<6} {'Conf':>6} {'Sources':>4} {'Reason':<40}")
    print(f"  {'-'*64}")
    for s in signals:
        ticker = s.get("ticker", "?")
        direction = s.get("direction", "?")
        conf = s.get("confidence_score", 0)
        n_src = s.get("n_confirming", 0)
        reason = s.get("reason", "")[:40]
        print(f"  {ticker:<8} {direction:<6} {conf:5.0%} {n_src:>4} {reason:<40}")


def print_trade_plan(plan: Dict):
    """Print today's trade plan."""
    print(section("TODAY'S TRADE PLAN"))

    if not plan:
        print("  No trade plan generated yet. Run unified_brain.py first.")
        return

    date = plan.get("date", "?")
    heat = plan.get("portfolio_heat", 0)
    regime = plan.get("regime", "?")
    n_trades = plan.get("n_trades", 0)

    print(f"  Date: {date}  |  Regime: {regime}  |  Heat: {heat:.0f}%  |  Trades: {n_trades}")

    trades = plan.get("trades", [])
    if not trades:
        print("  No trades today.")
        return

    print()
    print(f"  {'Ticker':<8} {'Dir':<6} {'Conv':<8} {'Conf':>5} {'Size':>7} {'Structure':<20} {'R:R':>5}")
    print(f"  {'-'*62}")
    for t in trades:
        struct_info = t.get("structure", {})
        struct_type = struct_info.get("type", "?")
        rr = struct_info.get("reward_risk", "")
        rr_str = f"{rr:.1f}x" if isinstance(rr, (int, float)) else ""

        print(f"  {t['ticker']:<8} {t['direction']:<6} {t['conviction']:<8} "
              f"{t['confidence']:4.0%} ${t['size']:>6.0f} {struct_type:<20} {rr_str:>5}")

    # Rationale
    print()
    for t in trades:
        rationale = t.get("rationale", "")
        if rationale:
            print(f"  {t['ticker']}: {rationale[:70]}")


def print_portfolio_heat(plan: Dict, agentic: Dict):
    """Print portfolio heat visualization."""
    print(section("PORTFOLIO HEAT"))

    equity = agentic.get("account_equity", 750)
    heat = plan.get("portfolio_heat", 0) if plan else 0
    trades = plan.get("trades", []) if plan else []

    total_risk = sum(t.get("size", 0) for t in trades)

    # Visual bar
    bar_width = 50
    filled = int(heat / 100 * bar_width)
    bar = "[" + "#" * filled + "." * (bar_width - filled) + "]"

    danger = ""
    if heat > 50:
        danger = " <-- HIGH"
    elif heat > 30:
        danger = " <-- moderate"

    print(f"  Capital at risk: ${total_risk:.0f} / ${equity:.0f}")
    print(f"  Heat: {heat:.0f}% {bar}{danger}")

    # Per-trade breakdown
    if trades:
        print()
        for t in trades:
            pct = t['size'] / equity * 100 if equity > 0 else 0
            mini_bar = "#" * int(pct / 2) + "." * (10 - int(pct / 2))
            print(f"  {t['ticker']:<8} ${t['size']:>6.0f}  ({pct:4.1f}%) [{mini_bar}]")


def print_strategy_agreement(plan: Dict):
    """Print strategy agreement matrix."""
    print(section("STRATEGY AGREEMENT MATRIX"))

    agreement = plan.get("strategy_agreement", {}) if plan else {}
    if not agreement:
        print("  No multi-source agreements found.")
        return

    # Sort by number of sources
    sorted_items = sorted(agreement.items(), key=lambda x: -x[1]["n_sources"])

    for ticker, info in sorted_items:
        n = info["n_sources"]
        direction = info["direction"]
        sources = info["sources"]
        conf = info.get("confidence", 0)

        stars = "*" * min(n, 5)
        print(f"  {ticker:<8} {direction:<6} {stars:<6} conf={conf:.0%}  sources: {', '.join(sources[:5])}")

    # Legend
    print(f"\n  Legend: * = 1 confirming source. *** = high conviction (3+).")


def print_earnings_warnings(earnings: Dict):
    """Print earnings proximity warnings."""
    print(section("EARNINGS WARNINGS"))

    if not earnings:
        print("  No earnings data loaded.")
        return

    imminent = earnings.get("imminent", [])
    if not imminent:
        print("  No imminent earnings in quality universe.")
        return

    for e in imminent:
        ticker = e.get("ticker", "?")
        date = e.get("earnings_date", "?")
        days = e.get("days_until", "?")
        alert = " *** AVOID ***" if isinstance(days, int) and days <= 2 else ""
        print(f"  {ticker:<8} reports {date}  ({days} day{'s' if days != 1 else ''} away){alert}")


def print_spread_recommendations(spread_recs: Dict):
    """Print spread recommendations."""
    print(section("SPREAD RECOMMENDATIONS"))

    if not spread_recs:
        print("  No spread recommendations. Run spread_strategy_engine.py first.")
        return

    recs = spread_recs.get("recommendations", [])
    actionable = [r for r in recs if r.get("liquidity_ok")]
    rejected = [r for r in recs if not r.get("liquidity_ok")]

    print(f"  Total: {len(recs)}  |  Actionable: {len(actionable)}  |  Rejected: {len(rejected)}")

    if actionable:
        print()
        print(f"  {'Ticker':<8} {'Type':<22} {'Risk':>7} {'Reward':>7} {'R:R':>5} {'Conf':>5}")
        print(f"  {'-'*56}")
        for r in actionable:
            print(f"  {r['ticker']:<8} {r['spread_type']:<22} "
                  f"${r['max_risk']:>6.0f} ${r['max_reward']:>6.0f} "
                  f"{r['reward_risk_ratio']:4.1f}x {r['confidence']:4.0%}")

    if rejected:
        print(f"\n  Rejected:")
        for r in rejected:
            reason = r.get("rejection_reason", "unknown")
            print(f"  {r['ticker']:<8} — {reason}")


def print_validated_strategies():
    """Print validated strategy scorecard."""
    print(section("VALIDATED STRATEGIES"))

    strats = _load("validated_strategies.json")
    if not strats:
        print("  No validated strategies file found.")
        return

    for category, strategies in strats.items():
        print(f"\n  {category.upper()}:")
        for name, info in strategies.items():
            sharpe = info.get("sharpe", "?")
            wr = info.get("wr", "")
            cagr = info.get("cagr", "")
            note = info.get("note", "")[:50]
            wr_str = f"WR={wr}%" if wr else ""
            cagr_str = f"CAGR={cagr}%" if cagr else ""
            print(f"    {name:<30} Sharpe={sharpe:<6} {wr_str:<10} {cagr_str:<12} {note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{hr('=', 70)}")
    print(f"  PORTFOLIO DASHBOARD — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{hr('=', 70)}")

    # Load all data
    agentic = _load("agentic_signals.json") or {}
    plan = _load("daily_trade_plan.json")
    earnings = _load("earnings_calendar.json")
    spread_recs = _load("spread_recommendations.json")

    # Print each section
    print_signals(agentic)
    print_trade_plan(plan)
    print_portfolio_heat(plan, agentic)
    print_strategy_agreement(plan)
    print_earnings_warnings(earnings)
    print_spread_recommendations(spread_recs)
    print_validated_strategies()

    print(f"\n{hr()}")
    print(f"  Dashboard generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{hr()}\n")


if __name__ == "__main__":
    main()
