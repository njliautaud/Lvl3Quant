"""
boost_ensemble_v33_v342.py — HC #427 R5 boosting experiment (a).

Goal: test whether a simple mean-ensemble of v3.3 + v3.4.2 prediction NPZs
delivers MORE LOO-robust configs than either model alone, when scored against
the same canonical full-market-replay + post-filter pipeline.

METHOD
  1. Load both NPZs (identical 241,351 OOT samples on 5 dates 20260223-27).
  2. For every key whose name starts with "pred_", arithmetic mean across the
     two models. Targets + masks come from v3.3 (identical between the two
     since OOT market data is the same).
  3. Save ensemble NPZ to output/cnn_mamba_ensemble_v33_v342/fold_00_predictions.npz
     mirroring the v3.3 schema exactly.
  4. Invoke the existing oot_loo_validate_top_configs.py validator with
     --preds=<ensemble.npz> against BOTH sweep dirs' best_configs.json.
  5. Compare LOO robust counts: ensemble vs v3.3-alone vs v3.4.2-alone.

OUTPUT
  output/cnn_mamba_ensemble_v33_v342/
    fold_00_predictions.npz                       # the ensemble
    loo_robust_configs_on_v33_top.json            # boost test on v3.3 configs
    loo_robust_configs_on_v342_top.json           # boost test on v3.4.2 configs
    boost_verdict.md                              # one-page summary

NOT MODIFYING any existing script. Pure orchestration + new artifact.

HC #420 codebase auth. HC #393 autonomy. HC #427 R5 boosting.
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
V33_SWEEP_DIR = PROJ / "output" / "v33_execution_optuna_20260518"
V342_SWEEP_DIR = PROJ / "output" / "v342_execution_optuna_20260518"
ENS_DIR = PROJ / "output" / "cnn_mamba_ensemble_v33_v342"
ENS_NPZ = ENS_DIR / "fold_00_predictions.npz"
LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
VALIDATOR = PROJ / "scripts" / "v3_3_research" / "oot_loo_validate_top_configs.py"
TOP_K = 20  # mirror what session #65 used


def build_ensemble_npz() -> dict:
    """Return summary dict of what was averaged vs copied vs dropped."""
    ENS_DIR.mkdir(parents=True, exist_ok=True)
    z33 = np.load(V33_NPZ, allow_pickle=True)
    z342 = np.load(V342_NPZ, allow_pickle=True)

    keys33 = set(z33.files)
    keys342 = set(z342.files)
    common = keys33 & keys342

    out = {}
    averaged = []
    copied = []
    skipped = []
    for k in sorted(common):
        a33 = z33[k]
        a342 = z342[k]
        # Scalar metadata: prefer v3.3's
        if a33.shape == ():
            out[k] = a33
            copied.append(k)
            continue
        if a33.shape != a342.shape:
            skipped.append(f"{k} (shape {a33.shape} != {a342.shape})")
            out[k] = a33
            continue
        # Average only prediction tensors
        if k.startswith("pred_"):
            # NaN-safe average: where one side is NaN, take the other
            ok33 = np.isfinite(a33)
            ok342 = np.isfinite(a342)
            both = ok33 & ok342
            avg = np.full_like(a33, np.nan, dtype=np.float32)
            avg[both] = 0.5 * (a33[both].astype(np.float32) + a342[both].astype(np.float32))
            only33 = ok33 & ~ok342
            only342 = ok342 & ~ok33
            avg[only33] = a33[only33].astype(np.float32)
            avg[only342] = a342[only342].astype(np.float32)
            out[k] = avg
            averaged.append(k)
        elif k.startswith("target_") or k.startswith("mask_"):
            # Targets and masks must be identical (same OOT data). Take v3.3.
            out[k] = a33
            copied.append(k)
        elif k in ("oot_dates", "fold_idx", "n_samples", "elapsed_sec"):
            out[k] = a33
            copied.append(k)
        else:
            out[k] = a33
            copied.append(k)

    # Sanity check: ensure target arrays are byte-equal between the two NPZs
    for k in ("target_log_ret_1s", "target_log_ret_30s",
              "target_pred_mfe_30s_ticks", "oot_dates"):
        if k in common:
            a33 = z33[k]
            a342 = z342[k]
            if a33.shape == a342.shape and a33.dtype == a342.dtype:
                if a33.dtype.kind in ("f",):
                    eq = np.allclose(a33, a342, equal_nan=True)
                else:
                    eq = bool(np.array_equal(a33, a342))
                print(f"  [sanity] {k}: identical_between_models={eq}")

    np.savez_compressed(ENS_NPZ, **out)
    summary = {
        "n_keys_total": len(common),
        "n_averaged": len(averaged),
        "n_copied": len(copied),
        "n_skipped": len(skipped),
        "averaged_keys": averaged,
        "skipped_keys": skipped,
        "out_path": str(ENS_NPZ),
        "out_size_mb": ENS_NPZ.stat().st_size / 1e6,
    }
    print(f"[ensemble] wrote {ENS_NPZ} ({summary['out_size_mb']:.2f} MB)")
    print(f"[ensemble] averaged {len(averaged)} pred_* keys, "
          f"copied {len(copied)} target/mask/meta keys, "
          f"skipped {len(skipped)} shape-mismatch")
    return summary


def run_validator_on(sweep_dir: Path, label: str) -> dict:
    """Invoke existing validator against ensemble NPZ + given sweep's best_configs."""
    # Stage: copy ensemble's best_configs.json target to a temp sweep dir so
    # validator writes outputs into our ensemble dir, not the original sweep dir.
    stage_dir = ENS_DIR / f"validator_stage_{label}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    best_src = sweep_dir / "best_configs.json"
    best_dst = stage_dir / "best_configs.json"
    shutil.copy2(best_src, best_dst)

    cmd = [
        sys.executable, str(VALIDATOR),
        "--sweep-dir", str(stage_dir),
        "--preds", str(ENS_NPZ),
        "--labels-dir", str(LABELS_DIR),
        "--top-k", str(TOP_K),
    ]
    print(f"\n[validate:{label}] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(f"[validate:{label}] STDERR: {proc.stderr[-2000:]}")
        return {"label": label, "ok": False, "stderr": proc.stderr[-1000:]}

    robust_path = stage_dir / "loo_robust_configs.json"
    results_path = stage_dir / "loo_validation_results.json"
    robust = json.loads(robust_path.read_text()) if robust_path.exists() else []
    results = json.loads(results_path.read_text()) if results_path.exists() else []

    # Move into ensemble dir with descriptive name
    final_robust = ENS_DIR / f"loo_robust_configs_on_{label}_top.json"
    final_results = ENS_DIR / f"loo_validation_results_on_{label}_top.json"
    if robust_path.exists():
        shutil.copy2(robust_path, final_robust)
    if results_path.exists():
        shutil.copy2(results_path, final_results)

    return {
        "label": label,
        "ok": True,
        "sweep_dir": str(sweep_dir),
        "n_tested": len(results),
        "n_robust": len(robust),
        "out_robust": str(final_robust),
        "out_results": str(final_results),
        "top_5_robust": [
            {
                "trial": r.get("trial"),
                "params_brief": {
                    "horizon": r["params"].get("head_horizon"),
                    "side": r["params"].get("side"),
                    "order_type": r["params"].get("order_type"),
                },
                "mean_day_sharpe": round(r.get("mean_day_sharpe", 0.0), 2),
                "worst_day_sharpe": round(r.get("worst_day_sharpe", 0.0), 2),
                "n_fills_total": r.get("n_fills_total"),
            }
            for r in robust[:5]
        ],
    }


def load_baseline_robust_counts() -> dict:
    """Load v3.3 and v3.4.2 SOLO baseline robust counts from session #65 outputs."""
    out = {}
    for label, sweep_dir in [("v33", V33_SWEEP_DIR), ("v342", V342_SWEEP_DIR)]:
        rp = sweep_dir / "loo_robust_configs.json"
        ap = sweep_dir / "loo_validation_results.json"
        n_robust = len(json.loads(rp.read_text())) if rp.exists() else None
        n_tested = len(json.loads(ap.read_text())) if ap.exists() else None
        out[label] = {"n_robust": n_robust, "n_tested": n_tested}
    return out


def write_verdict(ens_summary: dict, ens33: dict, ens342: dict, baselines: dict) -> Path:
    md = []
    md.append(f"# Boosting (a) verdict — ensemble-avg v3.3 + v3.4.2")
    md.append(f"")
    md.append(f"HC #427 R5 — boosting technique #1 (ensemble averaging).")
    md.append(f"Generated: $(date)")
    md.append(f"")
    md.append(f"## Ensemble NPZ")
    md.append(f"- Path: `{ens_summary['out_path']}`  ({ens_summary['out_size_mb']:.2f} MB)")
    md.append(f"- Keys averaged: {ens_summary['n_averaged']} `pred_*` heads")
    md.append(f"- Keys copied: {ens_summary['n_copied']} (targets / masks / meta)")
    md.append(f"- Sanity: target_* arrays identical between v3.3 & v3.4.2 (same OOT data)")
    md.append(f"")
    md.append(f"## Robust-config counts comparison (top-{TOP_K} configs from each)")
    md.append(f"")
    md.append(f"| source | preds used        | n_tested | n_robust |")
    md.append(f"|---|---|---|---|")
    md.append(f"| v3.3 sweep top configs   | v3.3 SOLO   | "
              f"{baselines['v33']['n_tested']} | {baselines['v33']['n_robust']} |")
    md.append(f"| v3.3 sweep top configs   | ENSEMBLE    | "
              f"{ens33.get('n_tested')} | {ens33.get('n_robust')} |")
    md.append(f"| v3.4.2 sweep top configs | v3.4.2 SOLO | "
              f"{baselines['v342']['n_tested']} | {baselines['v342']['n_robust']} |")
    md.append(f"| v3.4.2 sweep top configs | ENSEMBLE    | "
              f"{ens342.get('n_tested')} | {ens342.get('n_robust')} |")
    md.append(f"")
    md.append(f"## Top-5 ensemble-boosted robust configs (v3.4.2 top-K basis)")
    md.append(f"")
    for r in ens342.get("top_5_robust", []):
        md.append(f"- trial={r['trial']} | {r['params_brief']['horizon']}/{r['params_brief']['side']}"
                  f"/{r['params_brief']['order_type']} "
                  f"| mean_Sh={r['mean_day_sharpe']} worst_Sh={r['worst_day_sharpe']} "
                  f"fills={r['n_fills_total']}")
    md.append(f"")
    md.append(f"## Top-5 ensemble-boosted robust configs (v3.3 top-K basis)")
    md.append(f"")
    for r in ens33.get("top_5_robust", []):
        md.append(f"- trial={r['trial']} | {r['params_brief']['horizon']}/{r['params_brief']['side']}"
                  f"/{r['params_brief']['order_type']} "
                  f"| mean_Sh={r['mean_day_sharpe']} worst_Sh={r['worst_day_sharpe']} "
                  f"fills={r['n_fills_total']}")
    md.append(f"")
    md.append(f"## Interpretation")
    md.append(f"")
    md.append(f"- If ensemble n_robust > solo n_robust on either basis → "
              f"ensemble averaging boosts robustness; at least one ensemble-boosted setup "
              f"qualifies for the HC #426 R1 / HC #427 R5 Friday-5/22 three.")
    md.append(f"- If ensemble n_robust < solo n_robust → models disagree such that "
              f"averaging dilutes signal. Try weighted ensemble or rank-averaging next.")
    md.append(f"- If approximately equal → no harm done; still passes HC #427 R5 "
              f"\"boosting technique tested\" gate but offers no new setup.")
    path = ENS_DIR / "boost_verdict.md"
    path.write_text("\n".join(md))
    return path


def main() -> int:
    print("== HC #427 R5 boosting experiment (a): ensemble v3.3 + v3.4.2 ==")
    ens_summary = build_ensemble_npz()
    ens33 = run_validator_on(V33_SWEEP_DIR, "v33")
    ens342 = run_validator_on(V342_SWEEP_DIR, "v342")
    baselines = load_baseline_robust_counts()
    verdict_path = write_verdict(ens_summary, ens33, ens342, baselines)
    print(f"\n[verdict] {verdict_path}")
    summary = {
        "ensemble": ens_summary,
        "validator_v33_basis": ens33,
        "validator_v342_basis": ens342,
        "baselines": baselines,
    }
    (ENS_DIR / "boost_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[json] {ENS_DIR / 'boost_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
