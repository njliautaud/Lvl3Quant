"""
boost_weighted_ensemble.py — HC #427 R5 boosting experiment (c).

Sweep weighted-ensemble predictions across multiple (w_v33, w_v342) weight
splits, replay each against the v3.4.2 LOO-robust configs, and pick the
weight that maximizes robust count + average worst-day Sharpe.

Boost (a) showed:
  - solo v3.4.2 → 12/20 robust (best baseline)
  - 0.5/0.5 ensemble → 11/20 robust (slight dilution)
  - solo v3.3 → 7/20 robust (weakest)
Hypothesis: a v3.4.2-weighted ensemble (e.g. 0.7/0.3 or 0.6/0.4) preserves
most of v3.4.2's edge while pulling in v3.3's complementary information on
configs where v3.4.2 is borderline-robust.

METHOD
  1. For each weight (0.6/0.4, 0.7/0.3, 0.8/0.2, 0.4/0.6), build a weighted-
     ensemble NPZ in memory (predict_w = w33*p33 + w342*p342).
  2. Write each ensemble NPZ to a separate subdir under
     output/boost_weighted_ensemble/.
  3. For each, run existing oot_loo_validate_top_configs.py against v3.4.2's
     top-20 best_configs.
  4. Compare robust counts and best worst-day Sharpes.

OUTPUT
  output/boost_weighted_ensemble/
    w{w33}_{w342}/fold_00_predictions.npz   (per weight)
    w{w33}_{w342}/loo_robust_configs.json
    summary.json                            (winning weight + table)
    verdict.md

HC #420 codebase auth. HC #393 autonomy. HC #427 R5 boosting #3.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJ = Path("/home/jupiter/Lvl3Quant")
V33_NPZ = PROJ / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_predictions.npz"
V342_NPZ = PROJ / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "fold_00_predictions.npz"
V342_SWEEP_DIR = PROJ / "output" / "v342_execution_optuna_20260518"
V33_SWEEP_DIR = PROJ / "output" / "v33_execution_optuna_20260518"
OUT_DIR = PROJ / "output" / "boost_weighted_ensemble"
LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
VALIDATOR = PROJ / "scripts" / "v3_3_research" / "oot_loo_validate_top_configs.py"

WEIGHT_GRID = [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.6, 0.4), (0.1, 0.9)]  # (w33, w342)
TOP_K = 20


def build_weighted_npz(w33: float, w342: float, out_path: Path) -> dict:
    z33 = np.load(V33_NPZ, allow_pickle=True)
    z342 = np.load(V342_NPZ, allow_pickle=True)
    common = set(z33.files) & set(z342.files)
    out = {}
    averaged = 0
    for k in sorted(common):
        a33, a342 = z33[k], z342[k]
        if a33.shape == ():
            out[k] = a33
        elif a33.shape == a342.shape and k.startswith("pred_"):
            ok33 = np.isfinite(a33)
            ok342 = np.isfinite(a342)
            both = ok33 & ok342
            avg = np.full_like(a33, np.nan, dtype=np.float32)
            avg[both] = w33 * a33[both].astype(np.float32) + w342 * a342[both].astype(np.float32)
            avg[ok33 & ~ok342] = a33[ok33 & ~ok342].astype(np.float32)
            avg[ok342 & ~ok33] = a342[ok342 & ~ok33].astype(np.float32)
            out[k] = avg
            averaged += 1
        else:
            out[k] = a33
    np.savez_compressed(out_path, **out)
    return {"n_averaged": averaged, "out_size_mb": out_path.stat().st_size / 1e6}


def run_validator(stage_dir: Path, preds_path: Path) -> dict:
    cmd = [sys.executable, str(VALIDATOR),
           "--sweep-dir", str(stage_dir),
           "--preds", str(preds_path),
           "--labels-dir", str(LABELS_DIR),
           "--top-k", str(TOP_K)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    if proc.returncode != 0:
        return {"ok": False, "stderr": proc.stderr[-1500:]}
    robust_path = stage_dir / "loo_robust_configs.json"
    results_path = stage_dir / "loo_validation_results.json"
    robust = json.loads(robust_path.read_text()) if robust_path.exists() else []
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    return {
        "ok": True,
        "n_robust": len(robust),
        "n_tested": len(results),
        "worst_day_sharpes": [r.get("worst_day_sharpe", 0.0) for r in robust],
        "top_5": [
            {"trial": r.get("trial"),
             "params_brief": f"{r['params'].get('head_horizon')}/{r['params'].get('side')}/{r['params'].get('order_type')}",
             "mean_day_sharpe": round(r.get("mean_day_sharpe", 0), 2),
             "worst_day_sharpe": round(r.get("worst_day_sharpe", 0), 2),
             "n_fills_total": r.get("n_fills_total")}
            for r in robust[:5]
        ],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"weight_grid": WEIGHT_GRID, "results": []}

    for w33, w342 in WEIGHT_GRID:
        tag = f"w{int(w33*100):02d}_{int(w342*100):02d}"
        sub = OUT_DIR / tag
        sub.mkdir(parents=True, exist_ok=True)
        npz = sub / "fold_00_predictions.npz"
        print(f"\n=== weight {w33}/{w342} → {tag} ===")
        build_info = build_weighted_npz(w33, w342, npz)
        print(f"  built NPZ: {build_info}")

        # Stage validator with v3.4.2 best_configs
        shutil.copy2(V342_SWEEP_DIR / "best_configs.json", sub / "best_configs.json")
        r = run_validator(sub, npz)
        print(f"  validator: n_robust={r.get('n_robust')}/{r.get('n_tested')}")
        for t in r.get("top_5", []):
            print(f"    trial={t['trial']} {t['params_brief']} "
                  f"mean_Sh={t['mean_day_sharpe']} worst_Sh={t['worst_day_sharpe']} "
                  f"fills={t['n_fills_total']}")
        summary["results"].append({
            "w33": w33, "w342": w342, "tag": tag,
            "build": build_info, "validator": r,
        })

    # Winning weight: max n_robust, tiebreak by mean(worst_day_sharpes)
    valid = [s for s in summary["results"] if s["validator"].get("ok")]
    if valid:
        winner = max(valid, key=lambda s: (
            s["validator"]["n_robust"],
            float(np.mean(s["validator"]["worst_day_sharpes"])) if s["validator"]["worst_day_sharpes"] else 0.0,
        ))
        summary["winner"] = {
            "w33": winner["w33"], "w342": winner["w342"], "tag": winner["tag"],
            "n_robust": winner["validator"]["n_robust"],
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Verdict markdown
    md = [
        "# Boosting (c) verdict — weighted ensemble v3.3 / v3.4.2",
        "",
        "HC #427 R5 boosting technique #3. Sweep `w33*v3.3 + w342*v3.4.2` weights;",
        "score against v3.4.2 top-20 best_configs.",
        "",
        "## Baselines (from boost (a))",
        "- v3.4.2 SOLO: 12/20 robust",
        "- ensemble 0.5/0.5: 11/20 robust",
        "",
        "## Sweep results",
        "",
        "| w33 | w342 | n_robust | mean(worst_day_Sharpe) |",
        "|----:|-----:|---------:|-----------------------:|",
    ]
    for s in summary["results"]:
        if s["validator"].get("ok"):
            ws = s["validator"]["worst_day_sharpes"]
            mean_w = float(np.mean(ws)) if ws else 0.0
            md.append(f"| {s['w33']} | {s['w342']} | {s['validator']['n_robust']} | {mean_w:.2f} |")
        else:
            md.append(f"| {s['w33']} | {s['w342']} | ERR | — |")
    md.append("")
    if "winner" in summary:
        w = summary["winner"]
        md.append(f"## Winning weight: **w33={w['w33']}, w342={w['w342']}** "
                  f"(n_robust={w['n_robust']})")
    (OUT_DIR / "verdict.md").write_text("\n".join(md))
    print(f"\n[verdict] {OUT_DIR / 'verdict.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
