#!/usr/bin/env python3
"""
wheel_v5_research.py — Validate three improvements for wheel v5.

1. SECTOR FILTERING: Remove losing sectors (Cannabis, Consumer Cyclical,
   Basic Materials, Healthcare) — keep winners only.
2. SHORTER DTE + HIGHER DELTA: d35/dte10 vs d30/dte14 baseline.
3. IV RANK FLOOR: Only sell when IV rank >= 0.30 (30th percentile vs own 1y).

Runs through the existing wheel_engine walk-forward backtest, computes
CAGR / Sharpe / Sortino / MaxDD / WR / PF for each variant, prints a
comparative summary.

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 wheel_v5_research.py
"""
from __future__ import annotations
import sys
import time
import json
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest.wheel_engine import run_wheel, WheelConfig
from strategy.tier_runner import compute_metrics, _load_inputs, _load_spy_close, _apply_iv_rank_floor

CACHE = ROOT / "data" / "cache"
OUTPUT = Path("/home/jupiter/Lvl3Quant/output/wheel_v5_research")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Losing sectors to exclude (from sector_attribution analysis) ──
LOSING_SECTORS = {
    "Cannabis",         # PF 0.45, avg_pnl -0.45
    "Consumer Cyclical",# PF 0.72, avg_pnl -0.37
    "Basic Materials",  # PF 0.98, avg_pnl -0.02
    "Healthcare",       # PF 0.88, avg_pnl -0.08
}

# ── Baseline config (current v4: d30/dte14/65%PT/full_wheel) ──
BASELINE_CFG = WheelConfig(
    put_delta_target=0.30,
    call_delta_target=0.30,
    dte_min=10,
    dte_max=18,
    profit_take_pct=0.65,
    roll_dte_trigger=1,       # Full wheel (allow assignment)
    max_concurrent_names=20,
    sector_cap_pct=0.25,
    vix_max_gate=35.0,
    naaim_min_gate=-60.0,
    fund_score_floor=35.0,
    share_stop_loss_pct=0.15,  # 15% loss cut on assigned shares
)

# ── Higher delta + shorter DTE config (d35/dte10) ──
HIGH_DELTA_CFG = WheelConfig(
    put_delta_target=0.35,
    call_delta_target=0.30,
    dte_min=7,
    dte_max=14,
    profit_take_pct=0.65,
    roll_dte_trigger=1,
    max_concurrent_names=20,
    sector_cap_pct=0.25,
    vix_max_gate=35.0,
    naaim_min_gate=-60.0,
    fund_score_floor=35.0,
    share_stop_loss_pct=0.15,
)


def load_data():
    """Load all cached data."""
    data = _load_inputs(modeled=True, smoke=False, real_iv=False)
    return data


def filter_sectors(data: dict, exclude_sectors: set) -> dict:
    """Remove tickers belonging to excluded sectors from prices + IV."""
    uni = data["universe"]
    bad_tickers = set(uni.loc[uni["sector"].isin(exclude_sectors), "ticker"])

    filtered = dict(data)
    filtered["prices"] = data["prices"][~data["prices"]["ticker"].isin(bad_tickers)].copy()
    filtered["iv"] = data["iv"][~data["iv"]["ticker"].isin(bad_tickers)].copy()

    n_removed = len(bad_tickers)
    n_remaining = data["prices"]["ticker"].nunique() - n_removed
    print(f"  Sector filter: removed {n_removed} tickers ({', '.join(sorted(exclude_sectors))})")
    print(f"  Remaining: {n_remaining} tickers")

    return filtered


def run_variant(name: str, cfg: WheelConfig, data: dict,
                iv_rank_floor: float = 0.0,
                start: str = "2018-01-01", end: str = None,
                capital: float = 100_000.0) -> dict:
    """Run a single backtest variant and compute metrics."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"  Delta: {cfg.put_delta_target}, DTE: {cfg.dte_min}-{cfg.dte_max}, "
          f"PT: {cfg.profit_take_pct:.0%}, IV floor: {iv_rank_floor:.0%}")
    print(f"{'='*60}")

    px = data["prices"].copy()
    iv = data["iv"].copy()

    # Apply IV rank floor
    if iv_rank_floor > 0:
        iv = _apply_iv_rank_floor(iv, iv_rank_floor)
        print(f"  IV rank floor {iv_rank_floor:.0%} applied: {len(iv):,} IV rows remaining")

    t0 = time.time()
    result = run_wheel(
        cfg=cfg,
        prices=px,
        iv=iv,
        macro=data["macro"],
        fundamentals=data["fundamentals"],
        universe=data["universe"],
        starting_cash=capital,
        start=start,
        end=end,
        verbose=False,
    )
    elapsed = time.time() - t0

    spy_close = _load_spy_close(data)
    metrics = compute_metrics(result, capital, spy_close=spy_close)

    print(f"\n  Results ({elapsed:.0f}s):")
    print(f"  MTM  CAGR={metrics['cagr']*100:.1f}%  Sharpe={metrics['sharpe']:.2f}  "
          f"Sortino={metrics['sortino']:.2f}  MaxDD={metrics['max_dd']*100:.1f}%  "
          f"PF={metrics['pf']:.2f}  WR={metrics['wr']*100:.1f}%  Trades={metrics['n_trades']}")
    print(f"  REAL CAGR={metrics['realized_cagr']*100:.1f}%  Sharpe={metrics['realized_sharpe']:.2f}  "
          f"Sortino={metrics['realized_sortino']:.2f}  MaxDD={metrics['realized_max_dd']*100:.1f}%")

    return {
        "name": name,
        "config": asdict(cfg),
        "iv_rank_floor": iv_rank_floor,
        "metrics": metrics,
        "elapsed_s": elapsed,
        "equity_curve": result["equity_curve"],
        "ledger": result["ledger"],
        "csp_opened": result["csp_opened"],
        "assignment_count": result["assignment_count"],
    }


def main():
    print("=" * 70)
    print("WHEEL V5 RESEARCH — Three Improvement Variants")
    print("=" * 70)

    data = load_data()
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    results = {}

    # ── 1. BASELINE (current v4: d30/dte14/65%PT) ──
    results["baseline"] = run_variant(
        "Baseline (v4: d30/dte10-18/65%PT)",
        BASELINE_CFG, data, iv_rank_floor=0.0, end=end_date,
    )

    # ── 2. SECTOR FILTER ONLY ──
    data_filtered = filter_sectors(data, LOSING_SECTORS)
    results["sector_filter"] = run_variant(
        "Sector Filter (remove Cannabis/ConsCycl/BasicMat/Health)",
        BASELINE_CFG, data_filtered, iv_rank_floor=0.0, end=end_date,
    )

    # ── 3. HIGHER DELTA + SHORTER DTE ONLY ──
    results["high_delta"] = run_variant(
        "Higher Delta (d35/dte7-14)",
        HIGH_DELTA_CFG, data, iv_rank_floor=0.0, end=end_date,
    )

    # ── 4. IV RANK FLOOR ONLY ──
    results["iv_rank_30"] = run_variant(
        "IV Rank Floor >= 30%",
        BASELINE_CFG, data, iv_rank_floor=0.30, end=end_date,
    )

    # ── 5. SECTOR FILTER + HIGHER DELTA (combo) ──
    results["sector_high_delta"] = run_variant(
        "Sector Filter + Higher Delta (d35/dte7-14)",
        HIGH_DELTA_CFG, data_filtered, iv_rank_floor=0.0, end=end_date,
    )

    # ── 6. ALL THREE COMBINED (v5 candidate) ──
    results["v5_combined"] = run_variant(
        "V5 COMBINED (sector + d35/dte7-14 + IV rank 30%)",
        HIGH_DELTA_CFG, data_filtered, iv_rank_floor=0.30, end=end_date,
    )

    # ── 7. SECTOR + IV RANK (without high delta) ──
    results["sector_iv"] = run_variant(
        "Sector Filter + IV Rank >= 30%",
        BASELINE_CFG, data_filtered, iv_rank_floor=0.30, end=end_date,
    )

    # ── Summary ──
    print("\n" + "=" * 100)
    print("COMPARATIVE SUMMARY — ALL VARIANTS")
    print("=" * 100)
    print(f"{'Variant':<55} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'PF':>5} {'WR%':>5} {'Trades':>7}")
    print("-" * 100)

    baseline_cagr = results["baseline"]["metrics"]["cagr"]
    for key, r in results.items():
        m = r["metrics"]
        delta_cagr = (m["cagr"] - baseline_cagr) * 100
        marker = " ***" if key == "v5_combined" else ""
        print(f"{r['name']:<55} {m['cagr']*100:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd']*100:>6.1f}% {m['pf']:>5.2f} {m['wr']*100:>4.1f}% {m['n_trades']:>6d}"
              f"  ({delta_cagr:+.1f}pp){marker}")

    print("\n" + "=" * 100)
    print("REALIZED-CASH METRICS (account cash, no MTM noise)")
    print("=" * 100)
    print(f"{'Variant':<55} {'R.CAGR%':>8} {'R.Sharpe':>9} {'R.Sortino':>10} {'R.MaxDD%':>9}")
    print("-" * 100)
    for key, r in results.items():
        m = r["metrics"]
        marker = " ***" if key == "v5_combined" else ""
        print(f"{r['name']:<55} {m['realized_cagr']*100:>7.1f}% {m['realized_sharpe']:>9.2f} "
              f"{m['realized_sortino']:>10.2f} {m['realized_max_dd']*100:>8.1f}%{marker}")

    # ── Save results ──
    summary = {}
    for key, r in results.items():
        summary[key] = {
            "name": r["name"],
            "config": r["config"],
            "iv_rank_floor": r["iv_rank_floor"],
            "metrics": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                       for k, v in r["metrics"].items()},
            "elapsed_s": r["elapsed_s"],
            "csp_opened": r["csp_opened"],
            "assignment_count": r["assignment_count"],
        }
        # Save equity curves
        if isinstance(r["equity_curve"], pd.DataFrame):
            r["equity_curve"].to_parquet(OUTPUT / f"equity_{key}.parquet", index=False)
        if isinstance(r["ledger"], pd.DataFrame) and not r["ledger"].empty:
            r["ledger"].to_parquet(OUTPUT / f"ledger_{key}.parquet", index=False)

    with open(OUTPUT / "v5_research_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT}/")

    # ── Recommendation ──
    best_key = max(results.keys(),
                   key=lambda k: results[k]["metrics"]["sharpe"]
                   if results[k]["metrics"]["max_dd"] > -0.35 else -999)
    best = results[best_key]
    bm = best["metrics"]

    print(f"\n{'='*70}")
    print(f"RECOMMENDATION: {best['name']}")
    print(f"  CAGR: {bm['cagr']*100:.1f}%  Sharpe: {bm['sharpe']:.2f}  "
          f"Sortino: {bm['sortino']:.2f}  MaxDD: {bm['max_dd']*100:.1f}%")
    print(f"  vs Baseline: CAGR {(bm['cagr']-baseline_cagr)*100:+.1f}pp  "
          f"Sharpe {bm['sharpe']-results['baseline']['metrics']['sharpe']:+.2f}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    main()
