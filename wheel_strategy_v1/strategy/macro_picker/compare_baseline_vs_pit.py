"""
compare_baseline_vs_pit.py — Side-by-side OOT comparison: GA v2 (static-snapshot
feature shelf) vs GA v2-PIT (point-in-time fundamentals + sector flows shelf).

Reads both formula_v2_<sector>.json and formula_v2_pit_<sector>.json and emits
a delta table + writes results/compare_baseline_vs_pit.md.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1")
FORMS = ROOT / "strategy" / "macro_picker" / "formulas"
OUT_MD = ROOT / "results" / "compare_baseline_vs_pit.md"
OUT_MD.parent.mkdir(parents=True, exist_ok=True)


def _read(tag: str, sector: str) -> dict:
    p = FORMS / f"formula_{tag}_{sector.replace(' ', '_')}.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def main():
    sectors = [
        "Technology", "Basic Materials", "Utilities", "Industrials",         # GA v2 deployable
        "Energy", "Healthcare", "Consumer Defensive", "Communication Services",
        "Consumer Cyclical", "Real Estate", "Financial Services",            # GA v2 failed
    ]
    rows = []
    for s in sectors:
        base = _read("v2", s)
        pit = _read("v2_pit", s)
        bm = (base.get("fitness_oot") or {})
        pm = (pit.get("fitness_oot") or {})
        rows.append({
            "sector": s,
            "base_sharpe": bm.get("sharpe", float("nan")),
            "pit_sharpe":  pm.get("sharpe", float("nan")),
            "delta_sharpe": (pm.get("sharpe", 0) or 0) - (bm.get("sharpe", 0) or 0),
            "base_calmar": bm.get("calmar", float("nan")),
            "pit_calmar":  pm.get("calmar", float("nan")),
            "base_cagr": (bm.get("cagr", 0) or 0) * 100,
            "pit_cagr": (pm.get("cagr", 0) or 0) * 100,
            "base_dd": (bm.get("maxdd", 0) or 0) * 100,
            "pit_dd": (pm.get("maxdd", 0) or 0) * 100,
            "base_deploy": bool(base.get("deployable")),
            "pit_deploy":  bool(pit.get("deployable")),
        })

    df = pd.DataFrame(rows)
    print("\n=== GA v2 BASELINE vs GA v2-PIT — OOT only (2025-04-01 → present) ===")
    print(f"{'Sector':25s} | {'BL_Sh':>6s} {'PIT_Sh':>6s} {'Δ_Sh':>6s} | "
          f"{'BL_Cal':>6s} {'PIT_Cal':>7s} | {'BL_CAGR':>7s} {'PIT_CAGR':>8s} | "
          f"{'BL_Dep':>6s} {'PIT_Dep':>7s}")
    print("-" * 110)
    for _, r in df.iterrows():
        print(f"{r['sector']:25s} | {r['base_sharpe']:>+6.2f} {r['pit_sharpe']:>+6.2f} "
              f"{r['delta_sharpe']:>+6.2f} | {r['base_calmar']:>+6.2f} {r['pit_calmar']:>+7.2f} | "
              f"{r['base_cagr']:>+6.1f}% {r['pit_cagr']:>+7.1f}% | "
              f"{'YES' if r['base_deploy'] else 'no':>6s} {'YES' if r['pit_deploy'] else 'no':>7s}")

    # Aggregate uplift
    delta = df.dropna(subset=["pit_sharpe", "base_sharpe"])
    print("\n--- Aggregate (sectors where both runs exist) ---")
    print(f"  N sectors compared      : {len(delta)}")
    print(f"  Mean ΔSharpe (PIT−base) : {delta['delta_sharpe'].mean():+.2f}")
    print(f"  Median ΔSharpe          : {delta['delta_sharpe'].median():+.2f}")
    print(f"  PIT improves            : {(delta['delta_sharpe'] > 0).sum()}/{len(delta)}")
    print(f"  PIT regresses           : {(delta['delta_sharpe'] < 0).sum()}/{len(delta)}")
    base_dep = int(delta["base_deploy"].sum())
    pit_dep = int(delta["pit_deploy"].sum())
    print(f"  Deployable: base={base_dep}  pit={pit_dep}")

    # Write markdown
    with open(OUT_MD, "w") as f:
        f.write("# GA v2 baseline vs PIT-augmented — OOT comparison\n\n")
        f.write(f"OOT window: 2025-04-01 → present (~14 months trading days, single-slice).\n\n")
        f.write("| Sector | base Sharpe | PIT Sharpe | ΔSharpe | base Calmar | PIT Calmar | base CAGR | PIT CAGR | base deploy | PIT deploy |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|\n")
        for _, r in df.iterrows():
            f.write(f"| {r['sector']} | {r['base_sharpe']:+.2f} | {r['pit_sharpe']:+.2f} | "
                    f"{r['delta_sharpe']:+.2f} | {r['base_calmar']:+.2f} | {r['pit_calmar']:+.2f} | "
                    f"{r['base_cagr']:+.1f}% | {r['pit_cagr']:+.1f}% | "
                    f"{'YES' if r['base_deploy'] else 'no'} | {'YES' if r['pit_deploy'] else 'no'} |\n")
        f.write(f"\n## Aggregate\n\n")
        f.write(f"- N sectors compared: {len(delta)}\n")
        f.write(f"- Mean ΔSharpe (PIT-baseline): {delta['delta_sharpe'].mean():+.2f}\n")
        f.write(f"- PIT improves: {(delta['delta_sharpe'] > 0).sum()}/{len(delta)} sectors\n")
        f.write(f"- Deployable: baseline {base_dep} → PIT {pit_dep}\n")
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
