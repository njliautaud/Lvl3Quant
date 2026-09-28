#!/usr/bin/env python3
"""
Phase 2: Refinement study based on Phase 1 findings.

Phase 1 takeaways:
- Bull Put Spreads $10-wide: Sharpe 2.93, Sortino 3.40 (best) but MaxDD -45.9%
- Weekly rotation 7 DTE: Sharpe 2.10, MaxDD -18.1% (best risk-adjusted for CSP)
- Smart selection filters BROKE the strategy (too restrictive, caused concentration risk)
- Dynamic margin: marginal improvement over baseline

Phase 2 tests:
1. BPS $10 with conservative sizing (lower margin cap to tame DD)
2. BPS $10 + weekly rotation (7 DTE BPS)
3. Smart selection FIX: only use positive momentum, no vol/IV filter (less restrictive)
4. BPS $10 + conservative + weekly rotation (best combo)
5. CSP weekly rotation with higher margin (sweet spot search)
"""
import functools, json, sys, time
import numpy as np, pandas as pd
from pathlib import Path

# Import from phase 1
sys.path.insert(0, str(Path(__file__).parent))
from higher_returns_study import (
    load_data, compute_metrics, run_baseline_csp, run_bull_put_spread, OUTPUT
)

print = functools.partial(print, flush=True)

def main():
    t0 = time.time()
    prices, iv, macro, fund, universe, earnings = load_data()
    results = {}

    # ── 1. BPS $10 Conservative (lower margin cap) ──
    print("\n=== BPS $10 Conservative (40% margin, 40 positions) ===")
    r = run_bull_put_spread(prices, iv, macro, fund, universe, earnings,
        spread_width=10.0, max_concurrent=40, per_name_pct=0.03,
        margin_cap=0.40,  # same as baseline CSP
        label="BPS $10 Conservative (40% margin)")
    results["bps10_conservative"] = r["metrics"]
    r["equity_curve"].to_parquet(OUTPUT / "eq_bps10_cons.parquet")
    print(f"  CAGR: {r['metrics'].get('cagr_pct')}%  Sharpe: {r['metrics'].get('sharpe')}  "
          f"Sortino: {r['metrics'].get('sortino')}  MaxDD: {r['metrics'].get('max_dd_pct')}%")

    # ── 2. BPS $10 Very Conservative ──
    print("\n=== BPS $10 Very Conservative (30% margin, 30 positions) ===")
    r = run_bull_put_spread(prices, iv, macro, fund, universe, earnings,
        spread_width=10.0, max_concurrent=30, per_name_pct=0.025,
        margin_cap=0.30,
        label="BPS $10 Very Conservative (30% margin)")
    results["bps10_very_conservative"] = r["metrics"]
    r["equity_curve"].to_parquet(OUTPUT / "eq_bps10_vcons.parquet")
    print(f"  CAGR: {r['metrics'].get('cagr_pct')}%  Sharpe: {r['metrics'].get('sharpe')}  "
          f"Sortino: {r['metrics'].get('sortino')}  MaxDD: {r['metrics'].get('max_dd_pct')}%")

    # ── 3. BPS $10 Weekly (7 DTE spreads) ──
    print("\n=== BPS $10 Weekly (7 DTE) ===")
    r = run_bull_put_spread(prices, iv, macro, fund, universe, earnings,
        spread_width=10.0, max_concurrent=40, per_name_pct=0.03,
        margin_cap=0.40, dte_target=7, profit_take=0.50,
        label="BPS $10 Weekly (7 DTE, 40% margin)")
    results["bps10_weekly"] = r["metrics"]
    r["equity_curve"].to_parquet(OUTPUT / "eq_bps10_weekly.parquet")
    print(f"  CAGR: {r['metrics'].get('cagr_pct')}%  Sharpe: {r['metrics'].get('sharpe')}  "
          f"Sortino: {r['metrics'].get('sortino')}  MaxDD: {r['metrics'].get('max_dd_pct')}%")

    # ── 4. CSP Weekly with higher margin ──
    print("\n=== CSP Weekly (7 DTE, 50% margin) ===")
    r = run_baseline_csp(prices, iv, macro, fund, universe, earnings,
        dte_override=7, weekly_rotation_pct=0.25, profit_take=0.50,
        max_concurrent=40, margin_cap=0.50,
        label="CSP Weekly (7 DTE, 50% margin)")
    results["csp_weekly_50margin"] = r["metrics"]
    r["equity_curve"].to_parquet(OUTPUT / "eq_csp_weekly_50m.parquet")
    print(f"  CAGR: {r['metrics'].get('cagr_pct')}%  Sharpe: {r['metrics'].get('sharpe')}  "
          f"Sortino: {r['metrics'].get('sortino')}  MaxDD: {r['metrics'].get('max_dd_pct')}%")

    # ── 5. Smart Selection v2: momentum-only filter (no vol/IV restriction) ──
    print("\n=== Smart Selection v2: Momentum Only (3mo ret > 0) ===")
    r = run_baseline_csp(prices, iv, macro, fund, universe, earnings,
        smart_select=True, mom_lookback=63, max_rv=1.0, min_iv_rank=0.0,
        label="Smart Select: Momentum Only")
    results["smart_momentum_only"] = r["metrics"]
    r["equity_curve"].to_parquet(OUTPUT / "eq_smart_mom.parquet")
    print(f"  CAGR: {r['metrics'].get('cagr_pct')}%  Sharpe: {r['metrics'].get('sharpe')}  "
          f"Sortino: {r['metrics'].get('sortino')}  MaxDD: {r['metrics'].get('max_dd_pct')}%")

    # ── 6. Smart Selection v3: Momentum + mild IV rank ──
    print("\n=== Smart Selection v3: Momentum + IV Rank > 20% ===")
    r = run_baseline_csp(prices, iv, macro, fund, universe, earnings,
        smart_select=True, mom_lookback=63, max_rv=1.0, min_iv_rank=0.20,
        label="Smart Select: Mom + IVR>20%")
    results["smart_mom_ivr20"] = r["metrics"]
    r["equity_curve"].to_parquet(OUTPUT / "eq_smart_mom_ivr20.parquet")
    print(f"  CAGR: {r['metrics'].get('cagr_pct')}%  Sharpe: {r['metrics'].get('sharpe')}  "
          f"Sortino: {r['metrics'].get('sortino')}  MaxDD: {r['metrics'].get('max_dd_pct')}%")

    # ── 7. BPS $5 Conservative ──
    print("\n=== BPS $5 Conservative (40% margin) ===")
    r = run_bull_put_spread(prices, iv, macro, fund, universe, earnings,
        spread_width=5.0, max_concurrent=40, per_name_pct=0.03,
        margin_cap=0.40,
        label="BPS $5 Conservative (40% margin)")
    results["bps5_conservative"] = r["metrics"]
    r["equity_curve"].to_parquet(OUTPUT / "eq_bps5_cons.parquet")
    print(f"  CAGR: {r['metrics'].get('cagr_pct')}%  Sharpe: {r['metrics'].get('sharpe')}  "
          f"Sortino: {r['metrics'].get('sortino')}  MaxDD: {r['metrics'].get('max_dd_pct')}%")

    # ── 8. Best Combo: BPS $10 Conservative + Weekly ──
    print("\n=== Best Combo: BPS $10, 7 DTE, 30% margin ===")
    r = run_bull_put_spread(prices, iv, macro, fund, universe, earnings,
        spread_width=10.0, max_concurrent=30, per_name_pct=0.025,
        margin_cap=0.30, dte_target=7, profit_take=0.50,
        label="BPS $10 Weekly 30% margin")
    results["best_combo"] = r["metrics"]
    r["equity_curve"].to_parquet(OUTPUT / "eq_best_combo.parquet")
    print(f"  CAGR: {r['metrics'].get('cagr_pct')}%  Sharpe: {r['metrics'].get('sharpe')}  "
          f"Sortino: {r['metrics'].get('sortino')}  MaxDD: {r['metrics'].get('max_dd_pct')}%")

    # ── 9. CSP Dynamic Margin aggressive (higher cap in calm) ──
    print("\n=== CSP Dynamic Margin Aggressive (VIX<15: 70%, 15-25: 50%, 25-35: 25%) ===")
    r = run_baseline_csp(prices, iv, macro, fund, universe, earnings,
        dynamic_margin=True,
        margin_schedule={15: 0.70, 25: 0.50, 35: 0.25},
        max_concurrent=40,
        label="Dynamic Margin Aggressive")
    results["dyn_margin_aggressive"] = r["metrics"]
    r["equity_curve"].to_parquet(OUTPUT / "eq_dynm_agg.parquet")
    print(f"  CAGR: {r['metrics'].get('cagr_pct')}%  Sharpe: {r['metrics'].get('sharpe')}  "
          f"Sortino: {r['metrics'].get('sortino')}  MaxDD: {r['metrics'].get('max_dd_pct')}%")

    # ── Summary Table ──
    elapsed = time.time() - t0
    print(f"\n{'='*95}")
    print(f"PHASE 2 COMPLETE — {elapsed:.0f}s elapsed")
    print(f"{'='*95}")
    print(f"\n{'Label':<45} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'WR':>6} {'PF':>6}")
    print("-" * 90)
    for key, m in results.items():
        if "error" in m:
            print(f"  {m.get('label','?'):<43} ERROR: {m['error']}")
            continue
        print(f"  {m.get('label','?'):<43} {m.get('cagr_pct',0):>6.1f}% {m.get('sharpe',0):>7.2f} "
              f"{m.get('sortino',0):>8.2f} {m.get('max_dd_pct',0):>6.1f}% {m.get('win_rate_pct',0):>5.1f}% "
              f"{m.get('profit_factor',0):>5.2f}")

    # Save phase 2 results
    with open(OUTPUT / "summary_phase2.json", "w") as f:
        json.dump({"phase": 2, "results": results}, f, indent=2, default=str)

    # Load phase 1 and merge into combined summary
    try:
        with open(OUTPUT / "summary.json") as f:
            p1 = json.load(f)
        combined = {
            "study": "Wheel Higher Returns Study - Combined",
            "phases": {
                "phase1": p1["results"],
                "phase2": results,
            }
        }
        with open(OUTPUT / "summary_combined.json", "w") as f:
            json.dump(combined, f, indent=2, default=str)
    except:
        pass

    print(f"\nPhase 2 results saved to {OUTPUT}/summary_phase2.json")

if __name__ == "__main__":
    main()
