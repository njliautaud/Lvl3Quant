#!/usr/bin/env python3
"""
BPS Drawdown Recovery Analysis — HC #664 R4
=============================================

Critical question for deployment readiness: After a bad drawdown,
how long does it take to recover?

This matters because:
1. A 28% drawdown that recovers in 2 weeks is tolerable
2. A 28% drawdown that takes 6 months to recover is a strategy killer
3. Recovery time determines real-world position sizing

Tests both ungated and VIX-gated versions to see if VIX scaling
speeds recovery (hypothesis: yes, because it reduces losses in high-vol
and lets you compound faster when vol normalizes).

Also computes: underwater chart, recovery curves by drawdown depth,
and compares to SPY buy-and-hold recovery times.

Output: output/bps_drawdown_recovery/recovery_results.json
"""

import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))

OUTPUT = ROOT / "output" / "bps_drawdown_recovery"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load trades and macro."""
    trades = pd.read_parquet(ROOT / "output" / "bps_assignment_risk" / "trades_close_1dte.parquet")
    from higher_returns_study import load_data as ld
    prices, iv, macro, fund, universe, earnings = ld()
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    return trades, macro, prices


def apply_ba_and_vix_scaling(trades, macro, ba_frac=0.05, vix_scale=True):
    """Apply BA costs and optionally VIX-scaled sizing."""
    trades = trades.copy()
    trades["open_date"] = pd.to_datetime(trades["open_date"])
    trades["close_date"] = pd.to_datetime(trades["close_date"])

    # BA cost
    open_cost = trades["net_credit"].abs() * ba_frac * 2
    closed_early = trades["exit_type"].isin(["profit_take", "early_close_1DTE", "loss_stop"])
    close_cost = pd.Series(0.0, index=trades.index)
    close_cost[closed_early] = trades.loc[closed_early, "net_credit"].abs() * ba_frac * 2
    trades["ba_cost"] = open_cost + close_cost
    trades["honest_pnl"] = trades["realized_pnl"] - trades["ba_cost"]

    if vix_scale:
        vix_by_date = macro.set_index("date")["vix"].to_dict()
        trades["open_vix"] = trades["open_date"].map(vix_by_date)

        def scale(vix):
            if pd.isna(vix): return 1.0
            if vix < 15: return 1.0
            if vix < 20: return 0.8
            if vix < 25: return 0.5
            if vix < 30: return 0.25
            return 0.0

        trades["vix_scalar"] = trades["open_vix"].apply(scale)
        trades = trades[trades["vix_scalar"] > 0].copy()
        trades["honest_pnl"] = trades["honest_pnl"] * trades["vix_scalar"]

    return trades


def build_equity_curve(trades, starting_capital=100_000):
    """Build daily equity curve from trades."""
    daily_pnl = trades.groupby("close_date")["honest_pnl"].sum()
    all_dates = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq='B')
    daily_pnl = daily_pnl.reindex(all_dates, fill_value=0.0)
    equity = starting_capital + daily_pnl.cumsum()
    return equity, daily_pnl


def analyze_drawdowns(equity):
    """
    Find all drawdown episodes: peak, trough, recovery date, duration.
    """
    peak = equity.cummax()
    dd_pct = (equity - peak) / peak

    # Find drawdown episodes
    episodes = []
    in_dd = False
    dd_start = None
    dd_peak_val = None

    for date, val in equity.items():
        current_peak = peak[date]
        current_dd = dd_pct[date]

        if not in_dd and current_dd < -0.01:  # >1% drawdown starts episode
            in_dd = True
            dd_start = date
            dd_peak_val = current_peak

        elif in_dd and val >= dd_peak_val:  # recovered
            trough_idx = dd_pct[dd_start:date].idxmin()
            trough_dd = float(dd_pct[trough_idx])
            trough_val = float(equity[trough_idx])

            episodes.append({
                "peak_date": str(dd_start.date()),
                "trough_date": str(trough_idx.date()),
                "recovery_date": str(date.date()),
                "peak_to_trough_days": (trough_idx - dd_start).days,
                "trough_to_recovery_days": (date - trough_idx).days,
                "total_days": (date - dd_start).days,
                "max_dd_pct": round(trough_dd * 100, 1),
                "peak_equity": round(float(dd_peak_val), 0),
                "trough_equity": round(trough_val, 0),
            })
            in_dd = False

    # Handle ongoing drawdown (not yet recovered)
    if in_dd:
        trough_idx = dd_pct[dd_start:].idxmin()
        episodes.append({
            "peak_date": str(dd_start.date()),
            "trough_date": str(trough_idx.date()),
            "recovery_date": "ONGOING",
            "peak_to_trough_days": (trough_idx - dd_start).days,
            "trough_to_recovery_days": None,
            "total_days": None,
            "max_dd_pct": round(float(dd_pct[trough_idx]) * 100, 1),
            "peak_equity": round(float(dd_peak_val), 0),
            "trough_equity": round(float(equity[trough_idx]), 0),
        })

    return episodes


def drawdown_statistics(episodes):
    """Compute summary statistics across all drawdown episodes."""
    if not episodes:
        return {"note": "no drawdowns"}

    completed = [e for e in episodes if e["recovery_date"] != "ONGOING"]
    all_dds = [abs(e["max_dd_pct"]) for e in episodes]

    stats = {
        "total_episodes": len(episodes),
        "completed_recoveries": len(completed),
        "ongoing": len(episodes) - len(completed),
        "deepest_dd_pct": round(max(all_dds), 1),
        "avg_dd_pct": round(np.mean(all_dds), 1),
        "median_dd_pct": round(np.median(all_dds), 1),
    }

    if completed:
        recovery_days = [e["total_days"] for e in completed]
        trough_to_recovery = [e["trough_to_recovery_days"] for e in completed]

        stats.update({
            "avg_recovery_days": round(np.mean(recovery_days), 0),
            "median_recovery_days": round(np.median(recovery_days), 0),
            "max_recovery_days": max(recovery_days),
            "avg_trough_to_recovery_days": round(np.mean(trough_to_recovery), 0),
            "p90_recovery_days": round(np.percentile(recovery_days, 90), 0),
        })

        # Bucket by severity
        minor = [e for e in completed if abs(e["max_dd_pct"]) < 5]
        moderate = [e for e in completed if 5 <= abs(e["max_dd_pct"]) < 15]
        severe = [e for e in completed if abs(e["max_dd_pct"]) >= 15]

        stats["by_severity"] = {
            "minor_lt5pct": {
                "count": len(minor),
                "avg_recovery_days": round(np.mean([e["total_days"] for e in minor]), 0) if minor else None,
            },
            "moderate_5_15pct": {
                "count": len(moderate),
                "avg_recovery_days": round(np.mean([e["total_days"] for e in moderate]), 0) if moderate else None,
            },
            "severe_gt15pct": {
                "count": len(severe),
                "avg_recovery_days": round(np.mean([e["total_days"] for e in severe]), 0) if severe else None,
            },
        }

    return stats


def underwater_analysis(equity):
    """Compute underwater curve (time spent below previous high water mark)."""
    peak = equity.cummax()
    dd_pct = (equity - peak) / peak

    # How much time spent in drawdown?
    total_days = len(dd_pct)
    in_dd = (dd_pct < -0.01).sum()
    in_deep_dd = (dd_pct < -0.10).sum()
    in_severe_dd = (dd_pct < -0.20).sum()

    return {
        "total_trading_days": total_days,
        "days_in_drawdown_gt1pct": int(in_dd),
        "pct_time_in_drawdown": round(float(in_dd / total_days * 100), 1),
        "days_in_deep_dd_gt10pct": int(in_deep_dd),
        "pct_time_in_deep_dd": round(float(in_deep_dd / total_days * 100), 1),
        "days_in_severe_dd_gt20pct": int(in_severe_dd),
        "pct_time_in_severe_dd": round(float(in_severe_dd / total_days * 100), 1),
        "max_consecutive_dd_days": int(max_consecutive_below(dd_pct, -0.01)),
    }


def max_consecutive_below(series, threshold):
    """Find longest consecutive run below threshold."""
    below = (series < threshold).astype(int)
    streaks = below * (below.groupby((below != below.shift()).cumsum()).cumcount() + 1)
    return streaks.max() if len(streaks) > 0 else 0


def compare_spy_recovery(prices, equity):
    """Compare BPS drawdown recovery to SPY buy-and-hold."""
    # Get SPY prices aligned to same dates
    if "ticker" in prices.columns:
        spy = prices[prices["ticker"] == "SPY"].copy()
        spy["date"] = pd.to_datetime(spy["date"])
        spy = spy.set_index("date")["close"]
    else:
        spy = prices.get("SPY", pd.Series(dtype=float))

    if len(spy) < 100:
        return {"note": "insufficient SPY data"}

    # Align dates
    common_dates = equity.index.intersection(spy.index)
    if len(common_dates) < 100:
        # Try reindexing
        spy_reindexed = spy.reindex(equity.index, method='ffill')
        common_dates = equity.index[spy_reindexed.notna()]

    if len(common_dates) < 100:
        return {"note": "insufficient overlapping dates"}

    spy_aligned = spy.reindex(equity.index, method='ffill').dropna()
    eq_aligned = equity.loc[spy_aligned.index]

    # Normalize both to $100K start
    spy_norm = spy_aligned / spy_aligned.iloc[0] * 100000
    eq_norm = eq_aligned

    # Max DD comparison
    spy_peak = spy_norm.cummax()
    spy_dd = ((spy_norm - spy_peak) / spy_peak).min()

    eq_peak = eq_norm.cummax()
    eq_dd = ((eq_norm - eq_peak) / eq_peak).min()

    # Final return
    spy_total_return = float((spy_norm.iloc[-1] / spy_norm.iloc[0] - 1) * 100)
    eq_total_return = float((eq_norm.iloc[-1] / eq_norm.iloc[0] - 1) * 100)

    return {
        "spy_max_dd_pct": round(float(spy_dd * 100), 1),
        "bps_max_dd_pct": round(float(eq_dd * 100), 1),
        "spy_total_return_pct": round(spy_total_return, 1),
        "bps_total_return_pct": round(eq_total_return, 1),
        "period_years": round(len(common_dates) / 252, 1),
    }


def main():
    t0 = time.time()

    print("Loading data...")
    trades, macro, prices = load_data()
    print(f"  {len(trades)} trades")

    results = {}

    for config_name, use_vix_scale in [("ungated", False), ("vix_scaled", True)]:
        print(f"\n{'='*60}")
        print(f"CONFIG: {config_name}")
        print(f"{'='*60}")

        processed = apply_ba_and_vix_scaling(trades, macro, vix_scale=use_vix_scale)
        equity, daily_pnl = build_equity_curve(processed)

        print(f"  Equity: ${equity.iloc[0]:,.0f} → ${equity.iloc[-1]:,.0f}")
        print(f"  Total return: {(equity.iloc[-1]/equity.iloc[0] - 1)*100:.1f}%")

        # Drawdown episodes
        episodes = analyze_drawdowns(equity)
        print(f"  Drawdown episodes: {len(episodes)}")

        # Top 5 worst drawdowns
        worst = sorted(episodes, key=lambda e: e["max_dd_pct"])[:5]
        print(f"\n  Top 5 worst drawdowns:")
        for i, ep in enumerate(worst):
            rec = ep['total_days'] if ep['total_days'] else 'ONGOING'
            print(f"    {i+1}. {ep['max_dd_pct']}% ({ep['peak_date']} → {ep['trough_date']}, "
                  f"recovery: {rec} days)")

        # Statistics
        stats = drawdown_statistics(episodes)
        print(f"\n  Recovery stats:")
        for k, v in stats.items():
            if k != "by_severity":
                print(f"    {k}: {v}")
        if "by_severity" in stats:
            for sev, data in stats["by_severity"].items():
                print(f"    {sev}: {data}")

        # Underwater analysis
        underwater = underwater_analysis(equity)
        print(f"\n  Underwater analysis:")
        for k, v in underwater.items():
            print(f"    {k}: {v}")

        # SPY comparison
        spy_comp = compare_spy_recovery(prices, equity)
        print(f"\n  vs SPY:")
        for k, v in spy_comp.items():
            print(f"    {k}: {v}")

        results[config_name] = {
            "top5_drawdowns": worst,
            "all_episodes": episodes,
            "statistics": stats,
            "underwater": underwater,
            "spy_comparison": spy_comp,
            "equity_start": round(float(equity.iloc[0]), 0),
            "equity_end": round(float(equity.iloc[-1]), 0),
            "total_return_pct": round(float((equity.iloc[-1]/equity.iloc[0] - 1)*100), 1),
        }

    # Summary comparison
    print("\n" + "="*70)
    print("RECOVERY COMPARISON: Ungated vs VIX-Scaled")
    print("="*70)

    for metric in ["deepest_dd_pct", "avg_recovery_days", "median_recovery_days",
                    "max_recovery_days", "p90_recovery_days"]:
        ug = results["ungated"]["statistics"].get(metric, "N/A")
        vs = results["vix_scaled"]["statistics"].get(metric, "N/A")
        print(f"  {metric:<30} Ungated: {ug:<12} VIX-Scaled: {vs}")

    for metric in ["pct_time_in_drawdown", "pct_time_in_deep_dd", "max_consecutive_dd_days"]:
        ug = results["ungated"]["underwater"].get(metric, "N/A")
        vs = results["vix_scaled"]["underwater"].get(metric, "N/A")
        print(f"  {metric:<30} Ungated: {ug:<12} VIX-Scaled: {vs}")

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    with open(OUTPUT / "recovery_results.json", "w") as f:
        json.dump(convert(results), f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Results → {OUTPUT / 'recovery_results.json'}")


if __name__ == "__main__":
    main()
