#!/usr/bin/env python3
"""
wheel_v3_delta_sweep.py — Fine-grained delta sweep around the 30-delta sweet spot.
Tests 20/25/28/30/32/35-delta to find optimal premium capture point.
Uses the v3 expanded 230-ticker universe.
"""
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path("/home/jupiter/Lvl3Quant/scripts")))
from wheel_universe_v3_expand import load_all_data, run_portfolio_v3, compute_metrics, regime_analysis

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output" / "wheel_universe_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [DELTA-SWEEP] %(levelname)s %(message)s',
    level=logging.INFO,
)
log = logging.getLogger('DELTA-SWEEP')


def main():
    log.info("Loading data...")
    prices, spy_regime, sector_map = load_all_data()

    deltas = [0.20, 0.25, 0.28, 0.30, 0.32, 0.35]
    profit_takes = [0.50, 0.65, 0.75]

    results = {}
    for delta in deltas:
        for pt in profit_takes:
            name = f"d{int(delta*100)}_pt{int(pt*100)}"
            log.info(f"Running {name}...")
            t0 = time.time()

            daily_eq, trades = run_portfolio_v3(
                prices, spy_regime, sector_map,
                starting_cash=100_000,
                margin_cap=0.40, per_name_pct=0.03,
                put_delta=delta, profit_take=pt,
                bear_mode="liq_csp_only", dte_target=14,
            )

            elapsed = time.time() - t0
            metrics = compute_metrics(daily_eq, 100_000)
            regime = regime_analysis(daily_eq, spy_regime)

            results[name] = {
                "delta": delta,
                "profit_take": pt,
                "metrics": metrics,
                "regime": regime,
                "n_trades": len(trades),
                "elapsed_s": round(elapsed, 1),
            }

            log.info(f"  {name}: CAGR={metrics.get('cagr_pct')}% Sharpe={metrics.get('sharpe')} "
                    f"DD={metrics.get('max_dd_pct')}% Calmar={metrics.get('calmar')} "
                    f"({elapsed:.1f}s)")

    # Sort by Calmar (risk-adjusted return per unit of drawdown)
    ranked = sorted(results.items(), key=lambda x: x[1]["metrics"].get("calmar", 0), reverse=True)

    log.info("\n=== RANKED BY CALMAR ===")
    for name, r in ranked:
        m = r["metrics"]
        log.info(f"  {name:12s}: CAGR={m.get('cagr_pct'):5.1f}% Sharpe={m.get('sharpe'):5.2f} "
                f"DD={m.get('max_dd_pct'):6.1f}% Calmar={m.get('calmar'):5.2f} "
                f"Sortino={m.get('sortino'):5.2f}")

    out_file = OUT_DIR / "delta_sweep_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "generated": pd.Timestamp.now().isoformat(),
            "sweep_type": "delta x profit_take",
            "universe_size": prices["ticker"].nunique(),
            "results": results,
            "ranking_by_calmar": [
                {"name": n, "cagr": r["metrics"].get("cagr_pct"),
                 "sharpe": r["metrics"].get("sharpe"),
                 "max_dd": r["metrics"].get("max_dd_pct"),
                 "calmar": r["metrics"].get("calmar"),
                 "sortino": r["metrics"].get("sortino")}
                for n, r in ranked
            ]
        }, f, indent=2, default=str)

    log.info(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
