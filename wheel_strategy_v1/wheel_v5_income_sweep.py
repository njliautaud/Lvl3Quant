#!/usr/bin/env python3
"""
wheel_v5_income_sweep.py — Sweep for maximum income configs.

The first research pass showed:
- Higher delta (d35) = +3.3pp CAGR but -0.42 Sharpe, -6pp MaxDD
- Sector filter alone REDUCES CAGR (fewer tickers = fewer opportunities)
- IV rank floor has minimal impact on CAGR but improves realized Sharpe

The user wants HIGHER RETURNS. So we need to push the delta/DTE envelope
further while adding risk controls to compensate.

This sweep tests:
- d35 + d40 delta targets
- DTE 7-14 and 10-18
- Various IV rank floors (0, 20%, 30%, 40%)
- With and without sector filter
- Earnings 2-day buffer always on
"""
from __future__ import annotations
import sys
import time
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest.wheel_engine import run_wheel, WheelConfig
from strategy.tier_runner import compute_metrics, _load_inputs, _load_spy_close, _apply_iv_rank_floor

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/wheel_v5_income_sweep")
OUTPUT.mkdir(parents=True, exist_ok=True)

LOSING_SECTORS = {"Cannabis", "Consumer Cyclical", "Basic Materials", "Healthcare"}


def run_config(name, cfg, data, iv_rank_floor=0.0, sector_filter=False, capital=100_000.0):
    px = data["prices"].copy()
    iv = data["iv"].copy()

    if sector_filter:
        uni = data["universe"]
        bad = set(uni.loc[uni["sector"].isin(LOSING_SECTORS), "ticker"])
        px = px[~px["ticker"].isin(bad)]
        iv = iv[~iv["ticker"].isin(bad)]

    if iv_rank_floor > 0:
        iv = _apply_iv_rank_floor(iv, iv_rank_floor)

    t0 = time.time()
    result = run_wheel(
        cfg=cfg, prices=px, iv=iv, macro=data["macro"],
        fundamentals=data["fundamentals"], universe=data["universe"],
        starting_cash=capital, start="2018-01-01", end=None, verbose=False,
    )
    elapsed = time.time() - t0
    spy_close = _load_spy_close(data)
    metrics = compute_metrics(result, capital, spy_close=spy_close)

    return {
        "name": name,
        "metrics": {k: round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else v
                   for k, v in metrics.items()},
        "elapsed_s": round(elapsed, 1),
        "csp_opened": result["csp_opened"],
        "assignment_count": result["assignment_count"],
    }


def main():
    data = _load_inputs(modeled=True, smoke=False, real_iv=False)

    configs = []

    # Sweep matrix: delta x DTE x IV floor x sector_filter
    for delta in [0.30, 0.35, 0.40]:
        for dte_min, dte_max in [(7, 14), (10, 18)]:
            for iv_floor in [0.0, 0.20, 0.30]:
                for sec_filt in [False, True]:
                    cfg = WheelConfig(
                        put_delta_target=delta,
                        call_delta_target=0.30,
                        dte_min=dte_min,
                        dte_max=dte_max,
                        profit_take_pct=0.65,
                        roll_dte_trigger=1,
                        max_concurrent_names=20,
                        sector_cap_pct=0.25,
                        vix_max_gate=35.0,
                        naaim_min_gate=-60.0,
                        fund_score_floor=35.0,
                        share_stop_loss_pct=0.15,
                    )
                    name = (f"d{int(delta*100)}_dte{dte_min}-{dte_max}_"
                            f"iv{int(iv_floor*100)}_sec{'Y' if sec_filt else 'N'}")
                    configs.append((name, cfg, iv_floor, sec_filt))

    print(f"Running {len(configs)} configs...")
    results = []

    for i, (name, cfg, iv_floor, sec_filt) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {name}", end=" ... ", flush=True)
        r = run_config(name, cfg, data, iv_rank_floor=iv_floor, sector_filter=sec_filt)
        m = r["metrics"]
        print(f"CAGR={m['cagr']*100:.1f}%  Sharpe={m['sharpe']:.2f}  MaxDD={m['max_dd']*100:.1f}%")
        results.append(r)

    # Sort by CAGR descending
    results.sort(key=lambda r: r["metrics"]["cagr"], reverse=True)

    print("\n" + "=" * 120)
    print("TOP CONFIGS BY CAGR (MaxDD > -25%)")
    print("=" * 120)
    print(f"{'Config':<35} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'PF':>5} {'WR%':>5} {'R.CAGR%':>8} {'R.Sharpe':>9}")
    print("-" * 120)

    for r in results:
        m = r["metrics"]
        if m["max_dd"] < -0.25:
            continue
        print(f"{r['name']:<35} {m['cagr']*100:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd']*100:>6.1f}% {m['pf']:>5.2f} {m['wr']*100:>4.1f}% "
              f"{m['realized_cagr']*100:>7.1f}% {m['realized_sharpe']:>9.2f}")

    print("\n" + "=" * 120)
    print("TOP CONFIGS BY SHARPE")
    print("=" * 120)
    results_by_sharpe = sorted(results, key=lambda r: r["metrics"]["sharpe"], reverse=True)
    for r in results_by_sharpe[:10]:
        m = r["metrics"]
        print(f"{r['name']:<35} CAGR={m['cagr']*100:.1f}%  Sharpe={m['sharpe']:.2f}  "
              f"MaxDD={m['max_dd']*100:.1f}%  R.Sharpe={m['realized_sharpe']:.2f}")

    # Save
    with open(OUTPUT / "income_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to {OUTPUT}/income_sweep_results.json")


if __name__ == "__main__":
    main()
