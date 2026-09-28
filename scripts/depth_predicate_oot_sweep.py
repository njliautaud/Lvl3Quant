#!/usr/bin/env python3
"""
oot_depth_predicate_sweep.py - Apply depth>=N filter post-hoc to A_tp13_sl40_t050 OOT results.

The predicate_execution_analysis.py ran on IS data (results_v2 = 79 days 2025-07-22 to 2026-03-06).
This script applies the SAME post-hoc filter to the OOT results (results_oot_best = 68 days Dec-Mar OOT).

We test depth thresholds: 5, 10, 20, 50, 100 lots.
Output: per-threshold Sharpe, Sortino, PnL, trade count, win rate.
Also runs Monte Carlo for the best threshold.

Data: /home/jupiter/Lvl3Quant/fill_sim_test/results_oot_best/A_tp13_sl40_t050/
      Each JSON has per-trade book_size_at_post field.
"""
import json
import glob
import os
import sys
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("/home/jupiter/Lvl3Quant/fill_sim_test/results_oot_best/A_tp13_sl40_t050")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/oot_depth_predicate")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 25000
N_MC_SIMS = 5000
DEPTH_THRESHOLDS = [0, 5, 10, 20, 50, 100]  # 0 = baseline (no filter)

def load_oot_results(results_dir):
    """Load all OOT result JSONs. Returns list of (date, trades) tuples."""
    files = sorted(results_dir.glob("*.json"))
    days = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        date = f.stem.split("_")[-1]  # e.g. "20251201"
        trades = d.get("trades", [])
        days.append((date, trades))
    print(f"Loaded {len(days)} OOT days from {results_dir}")
    return days

def analyze_threshold(days, min_depth):
    """Apply depth filter and compute performance metrics."""
    daily_pnl = []
    all_trade_pnls = []
    all_depths = []

    for date, trades in days:
        day_trades = []
        for t in trades:
            depth = t.get("book_size_at_post", 999)  # default large if missing
            all_depths.append(depth)
            if depth >= min_depth:
                day_trades.append(t.get("pnl_dollars", 0))
                all_trade_pnls.append(t.get("pnl_dollars", 0))
        daily_pnl.append(sum(day_trades))

    pnl = np.array(daily_pnl)
    trades_arr = np.array(all_trade_pnls)
    wins = trades_arr[trades_arr > 0]
    losses = trades_arr[trades_arr <= 0]

    n_trades = len(trades_arr)
    wr = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0

    total_pnl = float(np.sum(pnl))
    avg_pnl = float(np.mean(pnl))

    # Sharpe
    pnl_std = float(np.std(pnl))
    sharpe = float(avg_pnl / pnl_std * np.sqrt(252)) if pnl_std > 0 else 0

    # Sortino
    neg_pnl = pnl[pnl < 0]
    downside = float(np.sqrt(np.mean(neg_pnl**2))) if len(neg_pnl) > 0 else 1e-6
    sortino = float(avg_pnl / downside * np.sqrt(252))

    # Drawdown
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(dd.min())

    return {
        "min_depth": min_depth,
        "n_trades": n_trades,
        "total_pnl": total_pnl,
        "avg_daily_pnl": avg_pnl,
        "sharpe": sharpe,
        "sortino": sortino,
        "win_rate": wr,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_dd": max_dd,
        "pct_trades_kept": n_trades / max(1, sum(len(t[1]) for t in days)),
        "all_depths": all_depths,
    }

def run_monte_carlo(days, min_depth, n_sims=N_MC_SIMS):
    """Monte Carlo for a given depth threshold."""
    all_trade_pnls = []
    daily_pnl = []

    for date, trades in days:
        day_pnl = 0
        for t in trades:
            depth = t.get("book_size_at_post", 999)
            if depth >= min_depth:
                pnl = t.get("pnl_dollars", 0)
                all_trade_pnls.append(pnl)
                day_pnl += pnl
        daily_pnl.append(day_pnl)

    if len(all_trade_pnls) < 10:
        return {"error": "too few trades", "n_trades": len(all_trade_pnls)}

    pnl_arr = np.array(daily_pnl)
    trade_arr = np.array(all_trade_pnls)
    n_trades = len(trade_arr)

    # Trade-level MC
    final_equity = []
    max_dds_pct = []
    for _ in range(n_sims):
        sampled = np.random.choice(trade_arr, size=n_trades, replace=True)
        cum_eq = INITIAL_CAPITAL + np.cumsum(sampled)
        final_equity.append(float(cum_eq[-1]))
        pk = np.maximum.accumulate(cum_eq)
        dd_pct = (cum_eq - pk) / pk * 100
        max_dds_pct.append(float(dd_pct.min()))

    fe = np.array(final_equity)
    md = np.array(max_dds_pct)

    # Sortino MC
    sortinos = []
    for _ in range(1000):
        sampled_daily = np.random.choice(pnl_arr, size=len(pnl_arr), replace=True)
        neg_s = sampled_daily[sampled_daily < 0]
        ds = float(np.sqrt(np.mean(neg_s**2))) if len(neg_s) > 0 else 1e-6
        sortinos.append(float(np.mean(sampled_daily) / ds * np.sqrt(252)))
    sortinos = np.array(sortinos)

    return {
        "min_depth": min_depth,
        "n_trades": n_trades,
        "final_equity_median": float(np.median(fe)),
        "final_equity_p5": float(np.percentile(fe, 5)),
        "final_equity_p95": float(np.percentile(fe, 95)),
        "max_dd_median_pct": float(np.median(md)),
        "max_dd_p5_pct": float(np.percentile(md, 5)),
        "prob_profit": float(np.mean(fe > INITIAL_CAPITAL)),
        "prob_ruin": float(np.mean(md < -50)),
        "sortino_median": float(np.median(sortinos)),
        "sortino_p5": float(np.percentile(sortinos, 5)),
    }

def main():
    print("=" * 70)
    print("OOT DEPTH PREDICATE SWEEP")
    print(f"Data: {RESULTS_DIR}")
    print("=" * 70)

    days = load_oot_results(RESULTS_DIR)

    if not days:
        print("ERROR: No results found!")
        sys.exit(1)

    # Check book_size_at_post availability
    total_trades = sum(len(t) for _, t in days)
    trades_with_depth = sum(
        1 for _, ts in days for t in ts
        if "book_size_at_post" in t
    )
    print(f"Total trades: {total_trades}, with book_size_at_post: {trades_with_depth}")

    if trades_with_depth == 0:
        print("WARNING: No book_size_at_post field found in trades!")
        # Print a sample trade to debug
        for _, ts in days:
            if ts:
                print("Sample trade fields:", list(ts[0].keys()))
                break

    # Sweep thresholds
    results = []
    print()
    print(f"{'Depth':>8}  {'Trades':>7}  {'%Kept':>6}  {'Sharpe':>8}  {'Sortino':>8}  {'PnL':>10}  {'WR':>7}")
    print("-" * 72)

    for thresh in DEPTH_THRESHOLDS:
        r = analyze_threshold(days, thresh)
        results.append(r)
        kept_pct = r["pct_trades_kept"] * 100
        print(f"  depth>={thresh:3d}  {r['n_trades']:7d}  {kept_pct:5.1f}%  {r['sharpe']:8.3f}  {r['sortino']:8.3f}  ${r['total_pnl']:9.0f}  {r['win_rate']:6.1%}")

    # Find best threshold by Sortino
    best = max(results, key=lambda x: x["sortino"])
    print(f"\nBest by Sortino: depth>={best['min_depth']} (Sortino={best['sortino']:.3f})")

    # Run MC on baseline and best
    print("\n=== Monte Carlo Analysis ===")
    for thresh in [0, best["min_depth"]]:
        print(f"\ndepth>={thresh}:")
        mc = run_monte_carlo(days, thresh)
        if "error" not in mc:
            print(f"  Trades: {mc['n_trades']}")
            print(f"  Final equity: median=${mc['final_equity_median']:.0f}, p5=${mc['final_equity_p5']:.0f}, p95=${mc['final_equity_p95']:.0f}")
            print(f"  Max DD: median={mc['max_dd_median_pct']:.1f}%, p5={mc['max_dd_p5_pct']:.1f}%")
            print(f"  Prob profit: {mc['prob_profit']*100:.1f}%")
            print(f"  Prob ruin (>50% DD): {mc['prob_ruin']*100:.1f}%")
            print(f"  Sortino: median={mc['sortino_median']:.2f}, p5={mc['sortino_p5']:.2f}")
            passed = mc['sortino_p5'] >= 2.0 and mc['prob_ruin'] <= 0.05 and mc['prob_profit'] >= 0.80
            print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")

    # Save results
    output = {
        "config": "A_tp13_sl40_t050",
        "data_dir": str(RESULTS_DIR),
        "n_days": len(days),
        "total_baseline_trades": total_trades,
        "trades_with_depth_field": trades_with_depth,
        "sweep_results": results,
        "best_threshold": best["min_depth"],
    }

    out_path = OUT_DIR / "depth_predicate_oot_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
