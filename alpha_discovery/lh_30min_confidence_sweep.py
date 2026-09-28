#!/usr/bin/env python3
"""
30-Min Lean LightGBM — Confidence Threshold Sweep
==================================================

Sweeps confidence thresholds 5%–30% on the lean model's full walk-forward
OOT predictions (26 folds, 135 days, 3584 bars) to find the optimal
threshold for the paper engine.

Current paper engine: 15% threshold.
Champion result: Sharpe 3.90 at top 10% on 37d holdout.

Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission).
Regime gap check per HC #428 R1.
Day concentration cap per HC #344.

Usage:
  cd /home/<user>/Lvl3Quant && python3 -u alpha_discovery/lh_30min_confidence_sweep.py

Author: Claude (autonomous research)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# Try multiple possible npz locations
NPZ_CANDIDATES = [
    ROOT / "output" / "lh_30min_feature_ablation" / "lean_concat_oot.npz",
    Path("/home/nick/Lvl3Quant/output/lh_30min_feature_ablation/lean_concat_oot.npz"),
    Path("/home/jupiter/Lvl3Quant/output/lh_30min_feature_ablation/lean_concat_oot.npz"),
]
OUTPUT_DIR = ROOT / "output" / "lh_30min_confidence_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission


# ═══════════════════════════════════════════════════════════════════
#  TRADE SIMULATION (matches v4_focused.py logic)
# ═══════════════════════════════════════════════════════════════════

def simulate_trades(
    preds: np.ndarray,
    actuals: np.ndarray,
    dates: np.ndarray,
    confidence_pct: float = 0.10,
    cost_ticks: float = COST_RT_TICKS,
    long_only: bool = False,
    long_pct: Optional[float] = None,
    short_pct: Optional[float] = None,
) -> Optional[Dict]:
    """
    Simulate trades with per-side and per-day reporting.

    If long_pct/short_pct are set, use asymmetric thresholds.
    Otherwise use confidence_pct symmetrically.
    """
    valid = ~np.isnan(preds) & ~np.isnan(actuals)
    preds_v = preds[valid]
    actuals_v = actuals[valid]
    dates_v = dates[valid]

    if len(preds_v) < 20:
        return None

    if long_pct is None:
        long_pct = confidence_pct
    if short_pct is None:
        short_pct = confidence_pct

    trades = []
    upper = np.quantile(preds_v, 1 - long_pct)
    lower = np.quantile(preds_v, short_pct)

    for i in range(len(preds_v)):
        if preds_v[i] >= upper:
            pnl = actuals_v[i] - cost_ticks
            trades.append({
                "dir": "long", "pnl": pnl, "raw": float(actuals_v[i]),
                "pred": float(preds_v[i]), "date": str(dates_v[i]),
            })
        elif not long_only and preds_v[i] <= lower:
            pnl = -actuals_v[i] - cost_ticks
            trades.append({
                "dir": "short", "pnl": pnl, "raw": float(-actuals_v[i]),
                "pred": float(preds_v[i]), "date": str(dates_v[i]),
            })

    if not trades:
        return None

    pnl_arr = np.array([t["pnl"] for t in trades])
    cum_pnl = np.cumsum(pnl_arr)

    sharpe = pnl_arr.mean() / max(pnl_arr.std(), 1e-6) * np.sqrt(252)
    downside = np.sqrt(np.mean(np.minimum(pnl_arr, 0) ** 2))
    sortino = pnl_arr.mean() / max(downside, 1e-6) * np.sqrt(252)
    wr = float(np.mean(pnl_arr > 0))
    pf = float(np.sum(pnl_arr[pnl_arr > 0]) / max(-np.sum(pnl_arr[pnl_arr < 0]), 1e-6))
    max_dd = float(np.min(cum_pnl - np.maximum.accumulate(cum_pnl)))

    long_trades = [t for t in trades if t["dir"] == "long"]
    short_trades = [t for t in trades if t["dir"] == "short"]
    long_pnl = np.array([t["pnl"] for t in long_trades]) if long_trades else np.array([])
    short_pnl = np.array([t["pnl"] for t in short_trades]) if short_trades else np.array([])

    # Per-day analysis
    trade_df = pd.DataFrame(trades)
    day_pnl = trade_df.groupby("date")["pnl"].agg(["sum", "count"]).reset_index()
    day_pnl.columns = ["date", "daily_pnl", "daily_trades"]

    n_trading_days = len(day_pnl)
    avg_trades_per_day = len(trades) / max(n_trading_days, 1)

    # Day concentration: max trades in a single day / total trades
    day_conc = float(day_pnl["daily_trades"].max() / len(trades)) if len(trades) > 0 else 1.0

    return {
        "n_trades": len(trades),
        "n_trading_days": n_trading_days,
        "avg_trades_per_day": float(avg_trades_per_day),
        "total_pnl_ticks": float(pnl_arr.sum()),
        "total_pnl_dollars": float(pnl_arr.sum() * ES_TICK_VALUE),
        "avg_pnl_ticks": float(pnl_arr.mean()),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate": wr,
        "profit_factor": pf,
        "max_dd_ticks": max_dd,
        "max_dd_dollars": float(max_dd * ES_TICK_VALUE),
        "day_concentration": day_conc,
        # Long side
        "long_trades": len(long_pnl),
        "long_wr": float(np.mean(long_pnl > 0)) if len(long_pnl) > 0 else 0,
        "long_avg_ticks": float(long_pnl.mean()) if len(long_pnl) > 0 else 0,
        "long_sharpe": float(
            long_pnl.mean() / max(long_pnl.std(), 1e-6) * np.sqrt(252)
        ) if len(long_pnl) > 2 else 0,
        # Short side
        "short_trades": len(short_pnl),
        "short_wr": float(np.mean(short_pnl > 0)) if len(short_pnl) > 0 else 0,
        "short_avg_ticks": float(short_pnl.mean()) if len(short_pnl) > 0 else 0,
        "short_sharpe": float(
            short_pnl.mean() / max(short_pnl.std(), 1e-6) * np.sqrt(252)
        ) if len(short_pnl) > 2 else 0,
        # Daily
        "daily_pnl_list": day_pnl.to_dict("records"),
    }


# ═══════════════════════════════════════════════════════════════════
#  REGIME STRATIFICATION (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════

def compute_regime_gap(
    preds: np.ndarray,
    actuals: np.ndarray,
    dates: np.ndarray,
    day_returns: Dict[str, float],
    confidence_pct: float = 0.10,
    long_pct: Optional[float] = None,
    short_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute regime gap per HC #428 R1.
    green (>0.1%), red (<-0.1%), flat.
    gap = |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|)
    Pass if gap <= 0.50.
    """
    valid = ~np.isnan(preds) & ~np.isnan(actuals)
    preds_v = preds[valid]
    actuals_v = actuals[valid]
    dates_v = dates[valid]

    day_class = {}
    for d, ret in day_returns.items():
        if ret > 0.001:
            day_class[d] = "green"
        elif ret < -0.001:
            day_class[d] = "red"
        else:
            day_class[d] = "flat"

    regime_arr = np.array([day_class.get(str(d), "flat") for d in dates_v])
    regime_sharpes = {}

    for regime in ["green", "red", "flat"]:
        mask = regime_arr == regime
        if mask.sum() < 10:
            continue
        sim = simulate_trades(
            preds_v[mask], actuals_v[mask], dates_v[mask],
            confidence_pct=confidence_pct,
            long_pct=long_pct, short_pct=short_pct,
        )
        if sim is not None:
            regime_sharpes[regime] = sim["sharpe"]

    if "green" in regime_sharpes and "red" in regime_sharpes:
        s_green = regime_sharpes["green"]
        s_red = regime_sharpes["red"]
        denom = max(abs(s_green), abs(s_red), 1e-6)
        gap = abs(s_green - s_red) / denom
        return {
            "gap": float(gap),
            "pass": gap <= 0.50,
            "green_sharpe": float(s_green),
            "red_sharpe": float(s_red),
            "flat_sharpe": regime_sharpes.get("flat", float("nan")),
        }
    else:
        return {"gap": float("nan"), "pass": False, "detail": "insufficient regime data"}


# ═══════════════════════════════════════════════════════════════════
#  COMPUTE DAY RETURNS (proxy from actuals — sum of bar moves per day)
# ═══════════════════════════════════════════════════════════════════

def compute_day_returns(actuals: np.ndarray, dates: np.ndarray) -> Dict[str, float]:
    """
    Compute daily direction from sum of 30-min bar actual moves.
    Normalize by dividing by a rough ES level to get approximate % return.
    We use 5000 pts as approximate ES level — only the sign/magnitude
    classification matters (>0.1% green, <-0.1% red).
    1 tick = 0.25 pts. So sum of actuals in ticks * 0.25 / 5000 = approx pct.
    """
    ES_APPROX_LEVEL = 5000.0  # approximate ES points
    TICK_TO_PTS = 0.25

    day_returns = {}
    unique_dates = sorted(set(dates))
    for d in unique_dates:
        mask = dates == d
        daily_ticks = float(actuals[mask].sum())
        daily_pts = daily_ticks * TICK_TO_PTS
        daily_pct = daily_pts / ES_APPROX_LEVEL
        day_returns[str(d)] = daily_pct

    return day_returns


# ═══════════════════════════════════════════════════════════════════
#  MAIN SWEEP
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  30-Min Lean LightGBM — Confidence Threshold Sweep")
    print("=" * 80)

    # Find npz file
    npz_path = None
    for candidate in NPZ_CANDIDATES:
        if candidate.exists():
            npz_path = candidate
            break

    if npz_path is None:
        print(f"ERROR: Cannot find lean_concat_oot.npz in any of:")
        for c in NPZ_CANDIDATES:
            print(f"  {c}")
        sys.exit(1)

    print(f"\nLoading: {npz_path}")
    data = np.load(str(npz_path), allow_pickle=True)
    preds = data["preds"]
    actuals = data["actuals"].astype(np.float64)
    dates = data["dates"]

    print(f"  Predictions: {len(preds)}")
    print(f"  Unique dates: {len(set(dates))}")
    print(f"  Date range: {sorted(set(dates))[0]} to {sorted(set(dates))[-1]}")
    print(f"  Pred range: [{preds.min():.2f}, {preds.max():.2f}]")
    print(f"  Actual range: [{actuals.min():.1f}, {actuals.max():.1f}] ticks")

    # Compute day returns for regime classification
    day_returns = compute_day_returns(actuals, dates)
    n_green = sum(1 for v in day_returns.values() if v > 0.001)
    n_red = sum(1 for v in day_returns.values() if v < -0.001)
    n_flat = len(day_returns) - n_green - n_red
    print(f"  Regimes: {n_green} green, {n_red} red, {n_flat} flat days")

    # ─────────────────────────────────────────
    #  SYMMETRIC SWEEP: 5% to 30% in 1% steps
    # ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PART 1: SYMMETRIC THRESHOLD SWEEP (same % for long & short)")
    print("=" * 80)

    thresholds = [i / 100.0 for i in range(5, 31)]
    results = []

    for thresh in thresholds:
        sim = simulate_trades(preds, actuals, dates, confidence_pct=thresh)
        if sim is None:
            continue

        regime = compute_regime_gap(preds, actuals, dates, day_returns, confidence_pct=thresh)

        results.append({
            "threshold_pct": int(thresh * 100),
            "n_trades": sim["n_trades"],
            "trades_per_day": sim["avg_trades_per_day"],
            "sharpe": sim["sharpe"],
            "sortino": sim["sortino"],
            "win_rate": sim["win_rate"],
            "profit_factor": sim["profit_factor"],
            "avg_pnl_ticks": sim["avg_pnl_ticks"],
            "total_pnl_ticks": sim["total_pnl_ticks"],
            "total_pnl_dollars": sim["total_pnl_dollars"],
            "max_dd_ticks": sim["max_dd_ticks"],
            "day_concentration": sim["day_concentration"],
            "regime_gap": regime.get("gap", float("nan")),
            "regime_pass": regime.get("pass", False),
            "green_sharpe": regime.get("green_sharpe", float("nan")),
            "red_sharpe": regime.get("red_sharpe", float("nan")),
            # Per-side
            "long_trades": sim["long_trades"],
            "long_wr": sim["long_wr"],
            "long_avg_ticks": sim["long_avg_ticks"],
            "long_sharpe": sim["long_sharpe"],
            "short_trades": sim["short_trades"],
            "short_wr": sim["short_wr"],
            "short_avg_ticks": sim["short_avg_ticks"],
            "short_sharpe": sim["short_sharpe"],
        })

    # ─────────────────────────────────────────
    #  Print table
    # ─────────────────────────────────────────
    print(f"\n{'Thr%':>4} {'N':>5} {'T/D':>5} {'Sharpe':>7} {'Sort':>7} {'WR':>6} "
          f"{'PF':>6} {'AvgPnL':>7} {'TotPnL':>8} {'MaxDD':>7} {'DConc':>5} "
          f"{'RGap':>5} {'Pass':>4}")
    print("-" * 100)

    for r in results:
        rg = f"{r['regime_gap']:.2f}" if not np.isnan(r['regime_gap']) else "  N/A"
        rp = "Y" if r['regime_pass'] else "N"
        print(f"{r['threshold_pct']:>4} {r['n_trades']:>5} {r['trades_per_day']:>5.1f} "
              f"{r['sharpe']:>7.2f} {r['sortino']:>7.2f} {r['win_rate']:>5.1%} "
              f"{r['profit_factor']:>6.2f} {r['avg_pnl_ticks']:>7.2f} "
              f"{r['total_pnl_ticks']:>8.1f} {r['max_dd_ticks']:>7.1f} "
              f"{r['day_concentration']:>5.2f} {rg:>5} {rp:>4}")

    # Per-side detail
    print(f"\n{'Thr%':>4} {'L#':>4} {'LWR':>6} {'LAvg':>6} {'LSh':>6}  "
          f"{'S#':>4} {'SWR':>6} {'SAvg':>6} {'SSh':>6}  "
          f"{'GrSh':>6} {'RdSh':>6}")
    print("-" * 85)

    for r in results:
        gs = f"{r['green_sharpe']:.1f}" if not np.isnan(r['green_sharpe']) else "  N/A"
        rs_str = f"{r['red_sharpe']:.1f}" if not np.isnan(r['red_sharpe']) else "  N/A"
        print(f"{r['threshold_pct']:>4} {r['long_trades']:>4} {r['long_wr']:>5.1%} "
              f"{r['long_avg_ticks']:>6.2f} {r['long_sharpe']:>6.1f}  "
              f"{r['short_trades']:>4} {r['short_wr']:>5.1%} "
              f"{r['short_avg_ticks']:>6.2f} {r['short_sharpe']:>6.1f}  "
              f"{gs:>6} {rs_str:>6}")

    # ─────────────────────────────────────────
    #  Find optimal threshold
    # ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  OPTIMAL THRESHOLD SELECTION")
    print("=" * 80)

    eligible = [r for r in results
                if r['regime_pass']
                and r['trades_per_day'] >= 2.0
                and r['day_concentration'] <= 0.70]

    if eligible:
        best = max(eligible, key=lambda x: x['sharpe'])
        print(f"\n  OPTIMAL (regime-pass, >=2 trades/day, conc<=0.70):")
        print(f"    Threshold: {best['threshold_pct']}%")
        print(f"    Sharpe:    {best['sharpe']:.2f}")
        print(f"    Sortino:   {best['sortino']:.2f}")
        print(f"    WR:        {best['win_rate']:.1%}")
        print(f"    PF:        {best['profit_factor']:.2f}")
        print(f"    N trades:  {best['n_trades']} ({best['trades_per_day']:.1f}/day)")
        print(f"    Avg PnL:   {best['avg_pnl_ticks']:.2f} ticks")
        print(f"    Total PnL: {best['total_pnl_ticks']:.1f} ticks (${best['total_pnl_dollars']:.0f})")
        print(f"    Max DD:    {best['max_dd_ticks']:.1f} ticks (${best['max_dd_ticks'] * ES_TICK_VALUE:.0f})")
        print(f"    Regime gap: {best['regime_gap']:.2f} (PASS)")
        print(f"    Day conc:  {best['day_concentration']:.2f}")
    else:
        print("\n  WARNING: No threshold passes ALL constraints.")
        print("  Relaxing: showing best Sharpe with regime_pass only:")
        eligible_relaxed = [r for r in results if r['regime_pass']]
        if eligible_relaxed:
            best = max(eligible_relaxed, key=lambda x: x['sharpe'])
            print(f"    Best regime-pass: {best['threshold_pct']}% (Sharpe {best['sharpe']:.2f}, "
                  f"{best['trades_per_day']:.1f} T/D, conc {best['day_concentration']:.2f})")
        else:
            print("  No threshold passes regime gap. Showing best Sharpe overall:")
            best = max(results, key=lambda x: x['sharpe'])
            print(f"    Best overall: {best['threshold_pct']}% (Sharpe {best['sharpe']:.2f})")

    # Compare to current 15%
    curr = next((r for r in results if r['threshold_pct'] == 15), None)
    if curr and eligible:
        print(f"\n  CURRENT (15%) vs OPTIMAL ({best['threshold_pct']}%):")
        print(f"    Sharpe:  {curr['sharpe']:.2f} → {best['sharpe']:.2f}")
        print(f"    Sortino: {curr['sortino']:.2f} → {best['sortino']:.2f}")
        print(f"    WR:      {curr['win_rate']:.1%} → {best['win_rate']:.1%}")
        print(f"    PF:      {curr['profit_factor']:.2f} → {best['profit_factor']:.2f}")
        print(f"    T/day:   {curr['trades_per_day']:.1f} → {best['trades_per_day']:.1f}")
        delta_sharpe = best['sharpe'] - curr['sharpe']
        print(f"    Sharpe delta: {delta_sharpe:+.2f}")
        if delta_sharpe > 0.3:
            print(f"    >>> RECOMMEND changing paper engine to {best['threshold_pct']}%")
        elif delta_sharpe > 0:
            print(f"    >>> Marginal improvement. Consider changing to {best['threshold_pct']}%.")
        else:
            print(f"    >>> Current 15% is already optimal or close. KEEP.")

    # ─────────────────────────────────────────
    #  PART 2: ASYMMETRIC THRESHOLDS
    # ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PART 2: ASYMMETRIC THRESHOLDS (different % for long vs short)")
    print("=" * 80)
    print("  Testing: long 5-25%, short 5-25% (5% steps for speed)")

    asym_results = []
    for l_pct in range(5, 26, 5):
        for s_pct in range(5, 26, 5):
            lp = l_pct / 100.0
            sp = s_pct / 100.0
            sim = simulate_trades(preds, actuals, dates,
                                  long_pct=lp, short_pct=sp)
            if sim is None:
                continue

            regime = compute_regime_gap(preds, actuals, dates, day_returns,
                                        long_pct=lp, short_pct=sp)

            asym_results.append({
                "long_pct": l_pct,
                "short_pct": s_pct,
                "n_trades": sim["n_trades"],
                "trades_per_day": sim["avg_trades_per_day"],
                "sharpe": sim["sharpe"],
                "sortino": sim["sortino"],
                "win_rate": sim["win_rate"],
                "profit_factor": sim["profit_factor"],
                "avg_pnl_ticks": sim["avg_pnl_ticks"],
                "total_pnl_dollars": sim["total_pnl_dollars"],
                "max_dd_ticks": sim["max_dd_ticks"],
                "day_concentration": sim["day_concentration"],
                "regime_gap": regime.get("gap", float("nan")),
                "regime_pass": regime.get("pass", False),
                "long_trades": sim["long_trades"],
                "short_trades": sim["short_trades"],
                "long_sharpe": sim["long_sharpe"],
                "short_sharpe": sim["short_sharpe"],
            })

    # Print asymmetric table
    print(f"\n{'L%':>3} {'S%':>3} {'N':>5} {'T/D':>5} {'Sharpe':>7} {'Sort':>7} "
          f"{'WR':>6} {'PF':>6} {'AvgPnL':>7} {'RGap':>5} {'Pass':>4} "
          f"{'L#':>4} {'S#':>4}")
    print("-" * 90)

    for r in sorted(asym_results, key=lambda x: x['sharpe'], reverse=True):
        rg = f"{r['regime_gap']:.2f}" if not np.isnan(r['regime_gap']) else "  N/A"
        rp = "Y" if r['regime_pass'] else "N"
        print(f"{r['long_pct']:>3} {r['short_pct']:>3} {r['n_trades']:>5} "
              f"{r['trades_per_day']:>5.1f} {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
              f"{r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} "
              f"{r['avg_pnl_ticks']:>7.2f} {rg:>5} {rp:>4} "
              f"{r['long_trades']:>4} {r['short_trades']:>4}")

    # Best asymmetric
    asym_eligible = [r for r in asym_results
                     if r['regime_pass']
                     and r['trades_per_day'] >= 2.0
                     and r['day_concentration'] <= 0.70]

    if asym_eligible:
        best_asym = max(asym_eligible, key=lambda x: x['sharpe'])
        print(f"\n  BEST ASYMMETRIC (constrained):")
        print(f"    Long: {best_asym['long_pct']}%, Short: {best_asym['short_pct']}%")
        print(f"    Sharpe: {best_asym['sharpe']:.2f}, Sortino: {best_asym['sortino']:.2f}")
        print(f"    WR: {best_asym['win_rate']:.1%}, PF: {best_asym['profit_factor']:.2f}")
        print(f"    N trades: {best_asym['n_trades']} ({best_asym['trades_per_day']:.1f}/day)")
        print(f"    Regime gap: {best_asym['regime_gap']:.2f}")

        if eligible:
            print(f"\n  SYMMETRIC BEST ({best['threshold_pct']}%) vs ASYMMETRIC BEST "
                  f"(L{best_asym['long_pct']}/S{best_asym['short_pct']}):")
            print(f"    Sharpe: {best['sharpe']:.2f} vs {best_asym['sharpe']:.2f}")
            delta = best_asym['sharpe'] - best['sharpe']
            if delta > 0.3:
                print(f"    >>> Asymmetric is meaningfully better (+{delta:.2f} Sharpe)")
            elif delta > 0:
                print(f"    >>> Asymmetric is marginally better (+{delta:.2f} Sharpe)")
            else:
                print(f"    >>> Symmetric is better or equal. Keep it simple.")

    # ─────────────────────────────────────────
    #  SAVE RESULTS
    # ─────────────────────────────────────────
    output = {
        "symmetric_sweep": results,
        "asymmetric_sweep": asym_results,
        "optimal_symmetric": best if eligible else None,
        "optimal_asymmetric": best_asym if asym_eligible else None,
        "data_info": {
            "npz_path": str(npz_path),
            "n_predictions": len(preds),
            "n_dates": len(set(dates)),
            "date_range": f"{sorted(set(dates))[0]} to {sorted(set(dates))[-1]}",
            "cost_rt_ticks": COST_RT_TICKS,
        },
    }

    # Convert any remaining numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    out_path = OUTPUT_DIR / "confidence_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nResults saved to: {out_path}")

    print("\n" + "=" * 80)
    print("  SWEEP COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
