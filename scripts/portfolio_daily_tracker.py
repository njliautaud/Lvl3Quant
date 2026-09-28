#!/usr/bin/env python3
"""
Combined Portfolio Daily Tracker
Tracks the 68% V5 / 14% IC / 18% ETF allocation.
Computes portfolio-level NAV, returns, drawdown, and risk metrics.
Runs daily at 4:30 PM ET (after market close) via cron.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = ROOT / "output" / "portfolio_tracker"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical weights per live readiness scorecard (item #126, SESSION_STATE)
WEIGHTS = {
    "v5_csp": 0.68,
    "ic_condors": 0.14,
    "etf_rotation_v3": 0.18,
}

# Starting capital per engine (all $100K paper)
INITIAL_NAV = 100_000.0

STATE_PATHS = {
    "v5_csp": ROOT / "live_trading_linux" / "wheel_v5_state",
    "ic_condors": ROOT / "live_trading_linux" / "wheel_ic_state",
    "etf_rotation_v3": ROOT / "live_trading_linux" / "etf_rotation_v3_state",
}


def load_nav_history(strategy: str) -> list:
    """Load NAV history for a strategy."""
    state_dir = STATE_PATHS[strategy]
    nav_file = state_dir / "nav_history.json"

    if nav_file.exists():
        with open(nav_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data

    # Fallback: try state.json
    state_file = state_dir / "state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        if strategy == "etf_rotation_v3":
            nav = state.get("nav_usd", INITIAL_NAV)
            return [{"date": datetime.now().isoformat(), "nav": nav}]
        else:
            cash = state.get("cash", INITIAL_NAV)
            margin = sum(p.get("margin_held", 0) for p in state.get("positions", []))
            return [{"date": datetime.now().isoformat(), "nav": cash + margin}]

    return []


def get_current_nav(strategy: str) -> float:
    """Get most recent NAV for a strategy."""
    hist = load_nav_history(strategy)
    if hist:
        return hist[-1].get("nav", INITIAL_NAV)
    return INITIAL_NAV


def compute_strategy_return(strategy: str) -> float:
    """Compute total return for a strategy."""
    nav = get_current_nav(strategy)
    return (nav - INITIAL_NAV) / INITIAL_NAV


def get_daily_navs(strategy: str) -> dict:
    """Get NAV by date (use last entry per day)."""
    hist = load_nav_history(strategy)
    daily = {}
    for entry in hist:
        date_str = entry.get("date", "")[:10]
        nav = entry.get("nav", INITIAL_NAV)
        if date_str:
            daily[date_str] = nav  # last entry per day wins
    return daily


def compute_portfolio_metrics():
    """Compute portfolio-level metrics."""
    now = datetime.now()

    result = {
        "timestamp": now.isoformat(),
        "weights": WEIGHTS,
        "strategies": {},
        "portfolio": {},
    }

    # Per-strategy metrics
    for strategy, weight in WEIGHTS.items():
        nav = get_current_nav(strategy)
        ret = compute_strategy_return(strategy)
        daily_navs = get_daily_navs(strategy)

        # Load trade stats
        state_dir = STATE_PATHS[strategy]
        trades_file = state_dir / "trades.jsonl"
        trades = []
        if trades_file.exists():
            with open(trades_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        pnl_trades = [t for t in trades if "pnl" in t]
        wins = sum(1 for t in pnl_trades if t["pnl"] > 0)
        losses = sum(1 for t in pnl_trades if t["pnl"] <= 0)
        total_pnl = sum(t["pnl"] for t in pnl_trades)

        result["strategies"][strategy] = {
            "weight": weight,
            "current_nav": round(nav, 2),
            "return_pct": round(ret * 100, 3),
            "weighted_return_pct": round(ret * weight * 100, 3),
            "n_trading_days": len(daily_navs),
            "total_trades": len(pnl_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
            "realized_pnl": round(total_pnl, 2),
        }

    # Portfolio-level metrics
    portfolio_return = sum(
        compute_strategy_return(s) * w for s, w in WEIGHTS.items()
    )

    # Compute daily portfolio returns using common dates
    all_daily = {s: get_daily_navs(s) for s in WEIGHTS}
    all_dates = sorted(set().union(*[d.keys() for d in all_daily.values()]))

    if len(all_dates) >= 2:
        port_navs = []
        for date in all_dates:
            port_nav = 0
            for strategy, weight in WEIGHTS.items():
                nav = all_daily[strategy].get(date, INITIAL_NAV)
                port_nav += nav * weight
            port_navs.append(port_nav)

        # Daily returns
        daily_returns = []
        for i in range(1, len(port_navs)):
            r = (port_navs[i] - port_navs[i-1]) / port_navs[i-1]
            daily_returns.append(r)

        if daily_returns:
            daily_returns = np.array(daily_returns)
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            downside = daily_returns[daily_returns < 0]
            sortino = np.mean(daily_returns) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else 0

            # Max drawdown
            peak = port_navs[0]
            max_dd = 0
            for nav in port_navs:
                peak = max(peak, nav)
                dd = (nav - peak) / peak
                max_dd = min(max_dd, dd)

            result["portfolio"]["daily_returns"] = len(daily_returns)
            result["portfolio"]["sharpe"] = round(sharpe, 2)
            result["portfolio"]["sortino"] = round(sortino, 2)
            result["portfolio"]["max_drawdown_pct"] = round(max_dd * 100, 2)

    portfolio_nav = sum(get_current_nav(s) * w for s, w in WEIGHTS.items())
    result["portfolio"]["nav"] = round(portfolio_nav, 2)
    result["portfolio"]["return_pct"] = round(portfolio_return * 100, 3)
    result["portfolio"]["initial_nav"] = INITIAL_NAV
    result["portfolio"]["n_dates_tracked"] = len(all_dates)

    # Days to live target
    target_days = 60
    days_tracked = len(all_dates)
    days_remaining = max(0, target_days - days_tracked)
    target_date = now + timedelta(days=int(days_remaining * 7/5))  # approximate business days
    result["portfolio"]["days_tracked"] = days_tracked
    result["portfolio"]["days_to_live_target"] = days_remaining
    result["portfolio"]["estimated_live_date"] = target_date.strftime("%Y-%m-%d")

    return result


def format_report(metrics: dict) -> str:
    """Format metrics as a human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"PORTFOLIO DAILY TRACKER — {metrics['timestamp'][:10]}")
    lines.append("=" * 60)
    lines.append(f"Allocation: V5 68% / IC 14% / ETF 18%")
    lines.append("")

    for strategy, data in metrics["strategies"].items():
        name = strategy.replace("_", " ").title()
        lines.append(f"  {name} ({data['weight']:.0%}):")
        lines.append(f"    NAV: ${data['current_nav']:>12,.2f}  Return: {data['return_pct']:+.2f}%  (weighted: {data['weighted_return_pct']:+.3f}%)")
        lines.append(f"    Trades: {data['total_trades']}  WR: {data['win_rate']:.0f}%  P&L: ${data['realized_pnl']:,.2f}")

    port = metrics["portfolio"]
    lines.append("")
    lines.append(f"  PORTFOLIO:")
    lines.append(f"    NAV: ${port['nav']:>12,.2f}  Return: {port['return_pct']:+.3f}%")
    if "sharpe" in port:
        lines.append(f"    Sharpe: {port['sharpe']:.2f}  Sortino: {port['sortino']:.2f}  MaxDD: {port['max_drawdown_pct']:.2f}%")

    lines.append("")
    lines.append(f"  LIVE READINESS:")
    lines.append(f"    Paper days tracked: {port['days_tracked']} / 60")
    lines.append(f"    Days remaining: {port['days_to_live_target']}")
    lines.append(f"    Estimated go-live: {port['estimated_live_date']}")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    metrics = compute_portfolio_metrics()

    # Save JSON
    date_str = datetime.now().strftime("%Y%m%d")
    json_path = OUTPUT_DIR / f"portfolio_{date_str}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save latest
    with open(OUTPUT_DIR / "latest.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print report
    report = format_report(metrics)
    print(report)

    # Save report
    with open(OUTPUT_DIR / f"report_{date_str}.txt", "w") as f:
        f.write(report)

    return metrics


if __name__ == "__main__":
    main()
