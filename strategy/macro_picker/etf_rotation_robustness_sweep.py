"""
ETF rotation hyperparameter robustness sweep (2026-06-08).

Goal: validate that the deploy-grade leader (hold=21, n-long=2, no-short,
target-vol=0.15, regime-overlay) sits on a ROBUST PLATEAU of hyperparams, not
a sharp peak that would indicate curve-fit. Per HC #428 R1 implication: real
edge survives small perturbations of trade-design knobs.

Neighborhood:
    hold-days       ∈ {14, 21, 28}      (signal-decay sensitivity)
    n-long          ∈ {1, 2, 3}         (concentration sensitivity)
    target-vol      ∈ {0.10, 0.15, 0.20}(sizing sensitivity)

All configs share:
    --no-short --regime-overlay --txn-cost-bps 5 --lev-min 0.25 --lev-max 2.0

Each config runs sequentially (each uses all cores internally via joblib);
total wall ~30-40 min for 27 configs.

Pass criterion for "robust plateau":
    >=70% of configs (>=19/27) have median per-fold Calmar >= 1.0.
    Leader-config Sharpe must be within ±20% of the neighborhood mean Sharpe.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
ETF_SCRIPT = ROOT / "strategy/macro_picker/etf_rotation_v1.py"

HOLD_GRID = [14, 21, 28]
NLONG_GRID = [1, 2, 3]
VOL_GRID = [0.10, 0.15, 0.20]

OUT_ROOT = ROOT / f"output/macro_picker/etf_robustness_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _run_config(hold: int, n_long: int, tgt_vol: float) -> dict:
    label = f"hold{hold:02d}_long{n_long}_vol{int(tgt_vol*100):02d}"
    cfg_dir = OUT_ROOT / label
    cfg_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, str(ETF_SCRIPT),
        "--hold-days", str(hold),
        "--n-long", str(n_long),
        "--no-short",
        "--regime-overlay",
        "--target-vol", str(tgt_vol),
        "--txn-cost-bps", "5",
        "--out", str(cfg_dir),
    ]
    log = cfg_dir / "run.log"
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            cwd=str(ROOT)).returncode
    wall = time.time() - t0
    # Parse metrics.json
    mj = cfg_dir / "metrics.json"
    if not mj.exists():
        return {"label": label, "hold": hold, "n_long": n_long, "tgt_vol": tgt_vol,
                "wall_sec": wall, "rc": rc, "error": "metrics.json missing"}
    m = json.loads(mj.read_text())
    pooled = m.get("pooled_metrics", m)  # depends on script schema
    per_fold = m.get("per_fold", [])
    fold_calmars = [f.get("calmar", 0) for f in per_fold if f.get("calmar") is not None]
    return {
        "label": label,
        "hold": hold, "n_long": n_long, "tgt_vol": tgt_vol,
        "wall_sec": round(wall, 1),
        "rc": rc,
        "sharpe": pooled.get("sharpe"),
        "calmar": pooled.get("calmar"),
        "cagr": pooled.get("cagr"),
        "max_dd": pooled.get("max_dd"),
        "median_fold_calmar": float(pd.Series(fold_calmars).median())
                              if fold_calmars else None,
        "worst_fold_calmar": float(pd.Series(fold_calmars).min())
                              if fold_calmars else None,
        "n_folds": len(per_fold),
        "folds_calmar_ge_1": int(sum(1 for c in fold_calmars if c is not None and c >= 1.0)),
    }


def main():
    print(f"[robustness_sweep] out_root = {OUT_ROOT}")
    print(f"[robustness_sweep] grid size = {len(HOLD_GRID)*len(NLONG_GRID)*len(VOL_GRID)}")

    rows = []
    t0 = time.time()
    for i, (h, nl, tv) in enumerate(product(HOLD_GRID, NLONG_GRID, VOL_GRID), 1):
        print(f"  [{i:02d}] hold={h} n_long={nl} vol={tv} ... ", end="", flush=True)
        row = _run_config(h, nl, tv)
        rows.append(row)
        _s = row.get('sharpe'); _s = float('nan') if _s is None else _s
        _c = row.get('calmar'); _c = float('nan') if _c is None else _c
        _mfc = row.get('median_fold_calmar'); _mfc = float('nan') if _mfc is None else _mfc
        print(f"sharpe={_s:.2f} calmar={_c:.2f} medFoldCal={_mfc:.2f} ({row['wall_sec']}s)")

    wall_total = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "sweep_grid.csv", index=False)

    # Pass criteria
    deploy_configs = df["median_fold_calmar"].apply(
        lambda x: x is not None and x >= 1.0
    ).sum()
    n_total = len(df)
    pct_deployable = 100.0 * deploy_configs / n_total
    leader_row = df[(df["hold"] == 21) & (df["n_long"] == 2) & (df["tgt_vol"] == 0.15)]
    leader_sharpe = leader_row["sharpe"].iloc[0] if not leader_row.empty else None
    mean_sharpe = df["sharpe"].mean()

    summary = {
        "wall_total_sec": round(wall_total, 1),
        "n_configs": n_total,
        "deploy_configs": int(deploy_configs),
        "pct_deployable": pct_deployable,
        "leader_sharpe": leader_sharpe,
        "neighborhood_mean_sharpe": float(mean_sharpe) if pd.notna(mean_sharpe) else None,
        "leader_vs_mean_sharpe_pct": (
            100 * (leader_sharpe - mean_sharpe) / abs(mean_sharpe)
            if leader_sharpe is not None and pd.notna(mean_sharpe) and mean_sharpe != 0
            else None
        ),
        "robust_plateau_pass": (pct_deployable >= 70.0
                                and leader_sharpe is not None
                                and pd.notna(mean_sharpe)
                                and abs(leader_sharpe - mean_sharpe) / max(abs(mean_sharpe), 1e-6) <= 0.20),
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Markdown
    lines = [
        "# ETF rotation — robustness sweep",
        "",
        f"Grid: 3 holds × 3 n_long × 3 target_vol = {n_total} configs.",
        f"Wall: {wall_total:.0f}s.",
        "",
        "## Robust-plateau verdict",
        f"- Configs with median per-fold Calmar ≥ 1.0: **{deploy_configs}/{n_total} ({pct_deployable:.0f}%)**",
        f"- Leader (hold=21, n_long=2, vol=0.15) pooled Sharpe: **{leader_sharpe:.2f}**",
        f"- Neighborhood mean Sharpe: **{mean_sharpe:.2f}**",
        f"- Leader vs mean: **{summary['leader_vs_mean_sharpe_pct']:+.1f}%**" if summary['leader_vs_mean_sharpe_pct'] is not None else "- Leader vs mean: n/a",
        f"- **Plateau pass:** {'YES ✅' if summary['robust_plateau_pass'] else 'NO ❌'}",
        "",
        "## Full grid",
        "",
        "| hold | n_long | vol | Sharpe | Calmar | medFoldCal | worstFoldCal | foldsPass | CAGR | MaxDD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['hold']} | {r['n_long']} | {r['tgt_vol']:.2f} | "
            f"{r.get('sharpe') or float('nan'):.2f} | "
            f"{r.get('calmar') or float('nan'):.2f} | "
            f"{r.get('median_fold_calmar') or float('nan'):.2f} | "
            f"{r.get('worst_fold_calmar') or float('nan'):.2f} | "
            f"{r.get('folds_calmar_ge_1', 0)}/{r.get('n_folds',0)} | "
            f"{(r.get('cagr') or 0)*100:.1f}% | "
            f"{(r.get('max_dd') or 0)*100:.1f}% |"
        )
    (OUT_ROOT / "report.md").write_text("\n".join(lines))

    print(f"\n[robustness_sweep] DONE wall={wall_total:.0f}s")
    print(f"  configs deployable: {deploy_configs}/{n_total} ({pct_deployable:.0f}%)")
    print(f"  leader Sharpe={leader_sharpe} | mean Sharpe={mean_sharpe:.2f}")
    print(f"  robust plateau: {'PASS' if summary['robust_plateau_pass'] else 'FAIL'}")
    print(f"  -> {OUT_ROOT}/report.md")


if __name__ == "__main__":
    main()
