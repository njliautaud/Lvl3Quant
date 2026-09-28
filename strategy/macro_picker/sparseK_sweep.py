"""
sparseK_sweep.py — Run sector_picker_v6_sparseK over a K-grid in parallel.

Sweeps K in {5, 8, 10, 15} on the master_panel_v2 shelf (76-feature pool) at
the best hold horizon from this morning's sweep (10d). Each K-run is itself
internally parallel across sectors via joblib; here we just give each
ProcessPoolExecutor worker its own K and let them race.

Each K-run is ~30s of CPU (11 sectors x walk-forward), so 4 K-values in
parallel should finish in well under 2 minutes on Jupiter.

Output:
    output/macro_picker/sparseK_sweep_<TS>/
        K05_H10d/  report.json, report.md, ...
        K08_H10d/  ...
        K10_H10d/  ...
        K15_H10d/  ...
        sweep_summary.json
        sweep_summary.md   <-- same format as horizon_sweep_*.log tail
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
K_GRID = [5, 8, 10, 15]
HOLD_DAYS = 10  # winner from horizon sweep this morning

PICKER = ROOT / "strategy/macro_picker/sector_picker_v6_sparseK.py"


def run_one_K(K: int, hold_days: int, out_root: Path) -> dict:
    """Spawn one v6 picker run for a given K. Returns headline metrics."""
    out_dir = out_root / f"K{K:02d}_H{hold_days:02d}d"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    cmd = [
        sys.executable, str(PICKER),
        "--K", str(K),
        "--hold-days", str(hold_days),
        "--out", str(out_dir),
        # leave n-jobs default (-1) — each K-process can use all cores; the
        # K-grid processes will compete but joblib backs off cleanly.
    ]
    t0 = time.time()
    err = None
    with open(log_path, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        err = f"picker exited rc={rc}; see {log_path}"
    wall = time.time() - t0

    summary = {"K": K, "hold_days": hold_days, "wall_sec": round(wall, 1),
               "err": err, "out_dir": str(out_dir)}
    rj_path = out_dir / "report.json"
    if rj_path.exists():
        rj = json.loads(rj_path.read_text())
        summary["pooled"] = rj.get("pooled_oot_combined", {})
        summary["spy_1x"] = rj.get("spy_1x", {})
        summary["n_sectors_calmar_pass"] = rj.get("n_sectors_calmar_pass", 0)
        summary["n_sectors_total"] = len(rj.get("per_sector", {}))
        summary["per_sector_sharpe"] = {
            s: round(m.get("sharpe", float("nan")), 2)
            for s, m in rj.get("per_sector", {}).items()
        }
        # surface top 5 most-picked features for quick eyeballing
        pc = rj.get("feature_pick_counts", {})
        summary["top5_picks"] = dict(list(pc.items())[:5])
    return summary


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / f"output/macro_picker/sparseK_sweep_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[sparseK_sweep] K grid: {K_GRID}  hold_days={HOLD_DAYS}")
    print(f"[sparseK_sweep] out_root: {out_root}")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=len(K_GRID)) as ex:
        futs = {ex.submit(run_one_K, K, HOLD_DAYS, out_root): K for K in K_GRID}
        for fut in as_completed(futs):
            r = fut.result()
            p = r.get("pooled", {})
            print(f"  K={r['K']:>2}  wall={r['wall_sec']:.0f}s  "
                  f"sharpe={p.get('sharpe', 'n/a')}  "
                  f"calmar={p.get('calmar', 'n/a')}  "
                  f"deployable={r.get('n_sectors_calmar_pass', 'n/a')}/"
                  f"{r.get('n_sectors_total', 11)}"
                  + (f"  ERR={r['err']}" if r.get('err') else ''))
            results.append(r)

    results.sort(key=lambda r: r["K"])
    total_wall = round(time.time() - t0, 1)
    out = {
        "K_grid": K_GRID,
        "hold_days": HOLD_DAYS,
        "wall_sec_total": total_wall,
        "results": results,
    }
    (out_root / "sweep_summary.json").write_text(json.dumps(out, indent=2, default=str))

    # Markdown comparison — same shape as horizon_sweep summary
    md = [f"# Sparse-K sweep (HC #565+ — sparse cardinality fix for v5 dense ridge)",
          "",
          f"Same panel (master_panel_v2, 76-feature pool), hold={HOLD_DAYS}d, "
          f"train/OOT walk-forward unchanged. Only K (number of features kept "
          f"per fold by |Spearman IC|) varies.",
          "",
          "| K | Pooled Sharpe | Pooled Calmar | Pooled CAGR | MaxDD | Deployable (Calmar>=1) |",
          "|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        p = r.get("pooled", {})
        md.append(f"| {r['K']:>2} | "
                  f"{p.get('sharpe', 0):.2f} | "
                  f"{p.get('calmar', 0):.2f} | "
                  f"{p.get('cagr', 0)*100:.1f}% | "
                  f"{p.get('max_dd', 0)*100:.1f}% | "
                  f"{r.get('n_sectors_calmar_pass', 0)}/{r.get('n_sectors_total', 11)} |")
    md.append("")
    md.append("## Top-5 picked features per K")
    md.append("")
    for r in results:
        md.append(f"### K={r['K']}")
        for f, c in (r.get("top5_picks") or {}).items():
            md.append(f"- `{f}`: picked {c}x across folds-x-sectors")
        md.append("")
    md.append(f"_total wall: {total_wall}s_")
    (out_root / "sweep_summary.md").write_text("\n".join(md))

    print()
    print("\n".join(md))
    print(f"\nwall_sec_total: {total_wall}")
    print(f"summary: {out_root / 'sweep_summary.md'}")


if __name__ == "__main__":
    main()
