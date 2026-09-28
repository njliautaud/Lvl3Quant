"""
picker_horizon_sweep.py — Run sector_picker_v5 across multiple hold horizons
in parallel, to test the HC #565/#566 hypothesis that 21-day hold is the
bottleneck (last night's v5: Sharpe 0.77, Calmar 0.30, all 11 sectors below
the 1.0 Calmar floor).

For each HOLD_DAYS in {1, 3, 5, 10, 21}, run the v5 picker (same panel
and feature pool) and write report.json/report.md/per_day_pnl.parquet to
a per-horizon subdirectory under research/findings/picker_horizon_sweep/.

Parallel across horizons via concurrent.futures.ProcessPoolExecutor — each
horizon is an independent ~3-min ridge fit, so wallclock should be ~3-5 min
instead of ~15 min sequential. HC #565 R1 mandates parallel compute.
"""
from __future__ import annotations
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
HORIZONS = [1, 3, 5, 10, 21]
OUT_ROOT = ROOT / "research/findings/picker_horizon_sweep"


def run_one_horizon(hold_days: int) -> dict:
    """Run v5-style picker with HOLD_DAYS patched. Must be top-level for pickling."""
    import sys, os
    sys.path.insert(0, str(ROOT / "strategy/macro_picker"))
    import sector_picker_v4 as v4  # type: ignore

    v4.PANEL = ROOT / "data/feature_store/master_panel/master_panel_v2.parquet"
    v4.HOLD_DAYS = hold_days
    v4.OUT_DIR = OUT_ROOT / f"hold_{hold_days:02d}d"
    v4.OUT_DIR.mkdir(parents=True, exist_ok=True)

    FUND_PIT = [
        "fp_revenue_ttm", "fp_fcf_ttm", "fp_ni_ttm", "fp_eps_ttm", "fp_ebitda_ttm",
        "fp_gross_margin", "fp_net_margin", "fp_ebitda_margin",
        "fp_roe", "fp_debt_to_equity", "fp_current_ratio",
        "fp_market_cap_pit", "fp_fcf_yield",
        "fp_rev_yoy_growth", "fp_ni_yoy_growth", "fp_eps_yoy_growth",
        "fp_fcf_yoy_growth", "fp_margin_trend_4q", "fp_beat_rate_4q",
    ]
    SECTOR_FLOWS = [
        "sf_dv_z_20d", "sf_dv_chg_60d",
        "sf_rs_60d_spy", "sf_rs_252d_spy",
        "sf_px_200dma", "sf_vol_252d",
        "sf_corr_tlt_60d", "sf_corr_hyg_60d", "sf_corr_uup_60d", "sf_corr_gld_60d",
    ]
    INSIDER_V2 = [
        "ins_net_insider_usd", "ins_gross_buy_usd", "ins_gross_sell_usd",
        "ins_n_buys", "ins_n_sells", "ins_n_buyers", "ins_n_sellers", "ins_mean_price",
    ]
    EXTRA = FUND_PIT + SECTOR_FLOWS + INSIDER_V2
    v4.FEATURE_POOL = list(dict.fromkeys(v4.FEATURE_POOL_BASE + v4.TEXT_FEATURES + EXTRA))

    # Redirect stdout to a per-horizon log so parallel processes don't interleave.
    log_path = v4.OUT_DIR / "run.log"
    t0 = time.time()
    with open(log_path, "w") as f:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = f
        try:
            v4.main()
            err = None
        except Exception as e:
            err = str(e)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    wall_sec = time.time() - t0

    # Pull headline metrics from the report.json v4.main writes.
    rj_path = v4.OUT_DIR / "report.json"
    summary = {"hold_days": hold_days, "wall_sec": round(wall_sec, 1),
               "err": err, "out_dir": str(v4.OUT_DIR)}
    if rj_path.exists():
        rj = json.loads(rj_path.read_text())
        summary["pooled"] = rj.get("pooled_oot_combined", {})
        summary["spy_1x"] = rj.get("spy_1x", {})
        summary["per_sector_sharpe"] = {
            s: round(m.get("sharpe", float("nan")), 2)
            for s, m in rj.get("per_sector", {}).items()
        }
        summary["n_sectors_calmar_pass"] = sum(
            1 for m in rj.get("per_sector", {}).values()
            if m.get("calmar", -1e9) >= 1.0
        )
    return summary


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[horizon_sweep] launching {len(HORIZONS)} parallel runs: {HORIZONS}")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=len(HORIZONS)) as ex:
        futs = {ex.submit(run_one_horizon, h): h for h in HORIZONS}
        for fut in as_completed(futs):
            r = fut.result()
            print(f"  hold={r['hold_days']:>2}d  wall={r['wall_sec']:.0f}s "
                  f"sharpe={r.get('pooled', {}).get('sharpe', 'n/a')} "
                  f"calmar={r.get('pooled', {}).get('calmar', 'n/a')} "
                  f"deployable_sectors={r.get('n_sectors_calmar_pass', 'n/a')}/11"
                  + (f"  ERR={r['err']}" if r.get('err') else ''))
            results.append(r)

    results.sort(key=lambda r: r["hold_days"])
    out = {
        "horizons_tested": HORIZONS,
        "wall_sec_total": round(time.time() - t0, 1),
        "results": results,
    }
    (OUT_ROOT / "sweep_summary.json").write_text(json.dumps(out, indent=2, default=str))

    # Compact markdown comparison
    md = ["# Picker hold-horizon sweep (HC #565 — 2026-06-08 morning)",
          "",
          "Same panel (master_panel_v2, 124 cols), same feature pool (76 features), "
          "same train/OOT walk-forward windows — only HOLD_DAYS varies.",
          "",
          "| Hold | Pooled Sharpe | Pooled Calmar | Pooled CAGR | MaxDD | Deployable (Calmar≥1) |",
          "|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        p = r.get("pooled", {})
        md.append(f"| {r['hold_days']:>2}d | "
                  f"{p.get('sharpe', 0):.2f} | "
                  f"{p.get('calmar', 0):.2f} | "
                  f"{p.get('cagr', 0)*100:.1f}% | "
                  f"{p.get('max_dd', 0)*100:.1f}% | "
                  f"{r.get('n_sectors_calmar_pass', 0)}/11 |")
    (OUT_ROOT / "sweep_summary.md").write_text("\n".join(md))

    print("\n" + "\n".join(md))
    print(f"\nwall_sec_total: {out['wall_sec_total']}")


if __name__ == "__main__":
    main()
