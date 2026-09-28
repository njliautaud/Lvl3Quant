#!/usr/bin/env python3
"""
wheel_v3_brake_robustness.py — Robustness test for the equity curve brake.

Tests whether the 60d/5%/0.5x winner is robust across neighboring parameters,
or if it's a lucky overfit to one specific combination.

Grid: lookback × threshold × scale = 5 × 5 × 3 = 75 configs
If the Sharpe/MaxDD improvement is stable across most of the grid,
the overlay is robust. If only 60d/5% works, it's overfit.
"""
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output" / "wheel_v3_brake_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from wheel_universe_v3_expand import load_all_data, compute_metrics, regime_analysis
from wheel_earnings_filter import download_earnings_dates, build_earnings_lookup
from wheel_v3_dd_protection import run_portfolio_v3_with_overlay, per_year_analysis

logging.basicConfig(
    format='%(asctime)s [ROBUST] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('ROBUST')


def main():
    log.info("=" * 60)
    log.info("EQUITY CURVE BRAKE — ROBUSTNESS SWEEP")
    log.info("=" * 60)

    # Load data
    log.info("Loading data...")
    prices, spy_regime, sector_map = load_all_data()
    earnings_raw = download_earnings_dates(prices["ticker"].unique())
    earnings_lookup = build_earnings_lookup(earnings_raw)

    base_params = dict(
        starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
        put_delta=0.30, dte_target=14, profit_take=0.65,
        bear_mode="liq_csp_only", max_assignments_5d=3,
        max_share_positions=5, loss_cut_pct=-0.15,
        min_price=10.0, max_price=500.0,
        earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )

    # Parameter grid
    lookbacks = [20, 40, 60, 80, 100]
    thresholds = [0.03, 0.04, 0.05, 0.06, 0.08]
    scales = [0.25, 0.50, 0.75]

    # First run baseline
    log.info("Running baseline (no overlay)...")
    t0 = time.time()
    eq_base, trades_base, _ = run_portfolio_v3_with_overlay(
        prices, spy_regime, sector_map, **base_params, overlay_type="none"
    )
    baseline_metrics = compute_metrics(eq_base, 100_000)
    baseline_yearly = per_year_analysis(eq_base)
    log.info(f"  Baseline: CAGR {baseline_metrics['cagr_pct']}%, Sharpe {baseline_metrics['sharpe']}, "
             f"MaxDD {baseline_metrics['max_dd_pct']}% ({time.time()-t0:.0f}s)")

    # Sweep
    results = []
    total = len(lookbacks) * len(thresholds) * len(scales)
    done = 0

    for lb in lookbacks:
        for th in thresholds:
            for sc in scales:
                done += 1
                tag = f"lb{lb}_th{int(th*100)}_sc{int(sc*100)}"

                t0 = time.time()
                eq, trades, overlay_log = run_portfolio_v3_with_overlay(
                    prices, spy_regime, sector_map, **base_params,
                    overlay_type="eq_brake",
                    eq_brake_lookback=lb,
                    eq_brake_threshold=th,
                    eq_brake_scale=sc,
                )
                elapsed = time.time() - t0

                m = compute_metrics(eq, 100_000)
                yearly = per_year_analysis(eq)

                n_active = len(overlay_log)
                bear_2022 = yearly.get(2022, {}).get("return_pct", None)

                result = {
                    "lookback": lb,
                    "threshold": th,
                    "scale": sc,
                    "tag": tag,
                    "cagr_pct": m.get("cagr_pct"),
                    "sharpe": m.get("sharpe"),
                    "sortino": m.get("sortino"),
                    "max_dd_pct": m.get("max_dd_pct"),
                    "calmar": m.get("calmar"),
                    "profit_factor": m.get("profit_factor"),
                    "bear_2022_pct": bear_2022,
                    "n_overlay_days": n_active,
                    "n_trades": len([t for t in trades if t["action"] == "sell_csp"]),
                }
                results.append(result)

                # Progress
                if done % 5 == 0 or done == total:
                    log.info(f"  [{done}/{total}] {tag}: Sharpe={m.get('sharpe')}, "
                             f"MaxDD={m.get('max_dd_pct')}%, 2022={bear_2022}% ({elapsed:.0f}s)")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # === ROBUSTNESS ANALYSIS ===
    log.info("\n" + "=" * 80)
    log.info("ROBUSTNESS ANALYSIS")
    log.info("=" * 80)

    base_sharpe = baseline_metrics["sharpe"]
    base_dd = baseline_metrics["max_dd_pct"]
    base_calmar = baseline_metrics["calmar"]

    # How many configs beat baseline?
    n_beat_sharpe = (df["sharpe"] > base_sharpe).sum()
    n_beat_dd = (df["max_dd_pct"] > base_dd).sum()  # less negative = better
    n_beat_calmar = (df["calmar"] > base_calmar).sum()
    n_beat_all = ((df["sharpe"] > base_sharpe) & (df["max_dd_pct"] > base_dd) & (df["calmar"] > base_calmar)).sum()

    log.info(f"Baseline: Sharpe={base_sharpe}, MaxDD={base_dd}%, Calmar={base_calmar}")
    log.info(f"Total configs tested: {len(df)}")
    log.info(f"Beat baseline Sharpe: {n_beat_sharpe}/{len(df)} ({100*n_beat_sharpe/len(df):.0f}%)")
    log.info(f"Beat baseline MaxDD:  {n_beat_dd}/{len(df)} ({100*n_beat_dd/len(df):.0f}%)")
    log.info(f"Beat baseline Calmar: {n_beat_calmar}/{len(df)} ({100*n_beat_calmar/len(df):.0f}%)")
    log.info(f"Beat ALL three:       {n_beat_all}/{len(df)} ({100*n_beat_all/len(df):.0f}%)")

    # Stability by dimension
    log.info("\n--- By Lookback ---")
    for lb in lookbacks:
        sub = df[df["lookback"] == lb]
        log.info(f"  {lb}d: Sharpe {sub['sharpe'].mean():.2f}±{sub['sharpe'].std():.2f}, "
                 f"MaxDD {sub['max_dd_pct'].mean():.1f}±{sub['max_dd_pct'].std():.1f}%, "
                 f"Calmar {sub['calmar'].mean():.2f}±{sub['calmar'].std():.2f}, "
                 f"2022 {sub['bear_2022_pct'].mean():.1f}%")

    log.info("\n--- By Threshold ---")
    for th in thresholds:
        sub = df[df["threshold"] == th]
        log.info(f"  {int(th*100)}%: Sharpe {sub['sharpe'].mean():.2f}±{sub['sharpe'].std():.2f}, "
                 f"MaxDD {sub['max_dd_pct'].mean():.1f}±{sub['max_dd_pct'].std():.1f}%, "
                 f"Calmar {sub['calmar'].mean():.2f}±{sub['calmar'].std():.2f}, "
                 f"2022 {sub['bear_2022_pct'].mean():.1f}%")

    log.info("\n--- By Scale ---")
    for sc in scales:
        sub = df[df["scale"] == sc]
        log.info(f"  {int(sc*100)}%: Sharpe {sub['sharpe'].mean():.2f}±{sub['sharpe'].std():.2f}, "
                 f"MaxDD {sub['max_dd_pct'].mean():.1f}±{sub['max_dd_pct'].std():.2f}, "
                 f"Calmar {sub['calmar'].mean():.2f}±{sub['calmar'].std():.2f}, "
                 f"2022 {sub['bear_2022_pct'].mean():.1f}%")

    # Top 10 configs
    log.info("\n--- Top 10 by Calmar (risk-adjusted) ---")
    top10 = df.nlargest(10, "calmar")
    for _, row in top10.iterrows():
        log.info(f"  lb={int(row['lookback'])}, th={row['threshold']:.2f}, sc={row['scale']:.2f} → "
                 f"Sharpe={row['sharpe']:.2f}, MaxDD={row['max_dd_pct']:.1f}%, "
                 f"Calmar={row['calmar']:.2f}, CAGR={row['cagr_pct']:.1f}%, "
                 f"2022={row['bear_2022_pct']:.1f}%")

    # Bottom 10 (worst performing overlays)
    log.info("\n--- Bottom 5 by Sharpe (worst overlays) ---")
    bot5 = df.nsmallest(5, "sharpe")
    for _, row in bot5.iterrows():
        log.info(f"  lb={int(row['lookback'])}, th={row['threshold']:.2f}, sc={row['scale']:.2f} → "
                 f"Sharpe={row['sharpe']:.2f}, MaxDD={row['max_dd_pct']:.1f}%, "
                 f"CAGR={row['cagr_pct']:.1f}%")

    # VERDICT
    log.info("\n" + "=" * 80)
    log.info("VERDICT")
    log.info("=" * 80)

    if n_beat_all / len(df) >= 0.50:
        log.info("✅ ROBUST — Majority of parameter grid beats baseline on all three metrics.")
        log.info("   The equity curve brake is a genuine improvement, not parameter-specific luck.")
    elif n_beat_all / len(df) >= 0.25:
        log.info("⚠️ PARTIALLY ROBUST — 25-50% of grid beats baseline. Edge exists but is parameter-sensitive.")
        log.info("   Use with caution; prefer the broader sweet spot over one specific combo.")
    else:
        log.info("❌ FRAGILE — Fewer than 25% of configs beat baseline. The 60d/5% result may be overfit.")
        log.info("   Do NOT deploy without further validation.")

    # CAGR cost analysis
    cagr_loss = df["cagr_pct"] - baseline_metrics["cagr_pct"]
    log.info(f"\nCAGR cost of overlay: median {cagr_loss.median():.1f}%, "
             f"mean {cagr_loss.mean():.1f}%, worst {cagr_loss.min():.1f}%")

    # Save results
    df.to_csv(OUT_DIR / "robustness_grid.csv", index=False)
    with open(OUT_DIR / "robustness_summary.json", "w") as f:
        json.dump({
            "baseline": baseline_metrics,
            "n_configs": len(df),
            "n_beat_sharpe": int(n_beat_sharpe),
            "n_beat_dd": int(n_beat_dd),
            "n_beat_calmar": int(n_beat_calmar),
            "n_beat_all": int(n_beat_all),
            "pct_beat_all": round(100 * n_beat_all / len(df), 1),
            "top10_calmar": top10[["lookback", "threshold", "scale", "sharpe",
                                    "max_dd_pct", "calmar", "cagr_pct", "bear_2022_pct"]].to_dict("records"),
        }, f, indent=2, default=str)

    log.info(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
