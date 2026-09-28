"""
hc427_boost_f_confconditional.py — HC #427 R5 boosting experiment (f).

CONFIDENCE-CONDITIONAL ENSEMBLE: use different v3.3 / v3.4.2 mix weights
inside each confidence quartile, so the ensemble adapts to where each model's
edge actually lives.

THEORETICAL CASE (per user-documented signal characteristics + HC #424/#425):
  - Short side has stronger edge than long side at every confidence level.
  - 1s horizon strongest, signal decays past ~30s (MFE/MAE).
  - Top 10% conf signals profitable even with market orders.
  - Top 20% conf works with passive limits.
  - Boost (a) (uniform 0.5/0.5 ensemble) BEAT v3.3 SOLO (+43% n_robust) on v3.3-basis
    but TIED v3.4.2 SOLO on v3.4.2-basis (11 vs 12) — implying the optimal mix
    is NOT uniform across confidence bands.
  - Boost (c) (single global weight sweep) was NULL — confirming a single weight
    cannot resolve regime differences.
  - Therefore: optimal weights likely DIFFER per confidence quartile. v3.3
    (uncertainty-weighted heads, lower variance) may calibrate LOW-conf bucket
    better; v3.4.2 (sharper, higher std) may dominate HIGH-conf bucket.

DESIGN
  1. Load v3.3 + v3.4.2 prediction NPZs (same 241,351 OOT samples × 5 days).
  2. Compute global confidence proxy = |v3.4.2 pred_log_ret_5s| (dominant horizon
     in 12 LOO-robust v3.4.2 configs — 7 of 12 use 5s/10s short).
  3. Bucket every prediction into 4 confidence quartiles (Q1=low … Q4=high).
  4. Define 8 conditional-weight schemes (w33 per quartile, w342 = 1 - w33):
       baseline_v342     : (Q1, Q2, Q3, Q4) = (0.0, 0.0, 0.0, 0.0)
       baseline_uniform  : (0.5, 0.5, 0.5, 0.5)
       monotone_v342     : (0.7, 0.5, 0.3, 0.0)   # low conf → v3.3, high → v3.4.2
       monotone_v33      : (0.0, 0.3, 0.5, 0.7)   # reverse
       extremes_v342     : (0.0, 0.5, 0.5, 0.0)   # only middle blends
       extremes_v33      : (1.0, 0.5, 0.5, 1.0)   # extremes pure v3.3
       q4_pure_v342      : (0.4, 0.4, 0.4, 0.0)   # only Q4 escapes ensemble
       q4_pure_v33       : (0.4, 0.4, 0.4, 1.0)
  5. For each scheme: build a new NPZ with conditional pred_* values (only pred_log_ret_*
     heads adjusted; all other heads keep uniform 0.5/0.5 avg as in boost (a) for
     consistency with the validator inputs).
  6. Run the existing oot_loo_validate_top_configs.py on each scheme NPZ against
     the v3.4.2 sweep top-20 best_configs.json.
  7. Report n_robust per scheme; flag any scheme that BEATS 12 (v3.4.2 SOLO).
  8. LOO-CV is already baked into the validator (5 OOT days, worst-day Sharpe gate).

OUTPUT
  output/hc427_boost_f_<timestamp>/
    npz/<scheme>.npz                                # 8 conditional-ensemble NPZs
    loo_robust_<scheme>.json                        # per-scheme robust configs
    loo_results_<scheme>.json                       # per-scheme full results
    summary.json                                    # n_robust per scheme
    verdict.md                                      # one-pager

NOT MODIFYING any existing script. Pure orchestration + new NPZ artifacts.

HC #420 codebase auth. HC #393 autonomy. HC #426 R4 canonical exec.
HC #427 R5 boosting techniques.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJ = Path("/home/jupiter/Lvl3Quant")
V33_NPZ = PROJ / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_predictions.npz"
V342_NPZ = PROJ / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "fold_00_predictions.npz"
V342_SWEEP_DIR = PROJ / "output" / "v342_execution_optuna_20260518"
LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
VALIDATOR = PROJ / "scripts" / "v3_3_research" / "oot_loo_validate_top_configs.py"
TOP_K = 20

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJ / "output" / f"hc427_boost_f_{TIMESTAMP}"
NPZ_DIR = OUT_DIR / "npz"

# Confidence proxy: dominant horizon among LOO-robust v3.4.2 configs.
CONF_HEAD = "pred_log_ret_5s"

# 8 conditional-weight schemes — w33 per quartile (Q1=lowest conf ... Q4=highest).
SCHEMES = {
    "baseline_v342":    (0.0, 0.0, 0.0, 0.0),   # pure v3.4.2 → matches solo baseline
    "baseline_uniform": (0.5, 0.5, 0.5, 0.5),   # boost (a) replication
    "monotone_v342":    (0.7, 0.5, 0.3, 0.0),   # low conf → v3.3; high → v3.4.2
    "monotone_v33":     (0.0, 0.3, 0.5, 0.7),   # reverse monotone
    "extremes_v342":    (0.0, 0.5, 0.5, 0.0),   # pure v3.4.2 at tails
    "extremes_v33":     (1.0, 0.5, 0.5, 1.0),   # pure v3.3 at tails
    "q4_pure_v342":     (0.4, 0.4, 0.4, 0.0),
    "q4_pure_v33":      (0.4, 0.4, 0.4, 1.0),
}


def _bucket_indices(conf_abs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (bucket_id 0..3 per sample, quartile breakpoints).

    Uses GLOBAL quantiles across full OOT — single set of bins, same for all
    schemes (so we are comparing apples to apples).
    """
    valid = np.isfinite(conf_abs)
    qs = np.quantile(conf_abs[valid], [0.25, 0.50, 0.75])
    # bucket: 0 if x <= q25, 1 if x <= q50, 2 if x <= q75, else 3
    bucket = np.full(conf_abs.shape, -1, dtype=np.int8)
    bucket[valid & (conf_abs <= qs[0])] = 0
    bucket[valid & (conf_abs > qs[0]) & (conf_abs <= qs[1])] = 1
    bucket[valid & (conf_abs > qs[1]) & (conf_abs <= qs[2])] = 2
    bucket[valid & (conf_abs > qs[2])] = 3
    return bucket, qs


def build_conditional_npz(scheme_name: str, w33_per_q: tuple[float, ...]) -> Path:
    """Materialize a conditional-ensemble NPZ for the given scheme.

    Only directional `pred_log_ret_<H>` heads are conditionally weighted by the
    quartile of |v3.4.2 pred_log_ret_5s|. All other pred_* keys use uniform
    0.5/0.5 avg (matches boost (a) NPZ behavior so config thresholds stay valid).
    target_*, mask_*, oot_dates, meta keys are copied from v3.3 NPZ
    (identical to v3.4.2 on the same OOT data — verified in boost (a)).
    """
    z33 = np.load(V33_NPZ, allow_pickle=True)
    z342 = np.load(V342_NPZ, allow_pickle=True)

    conf_proxy = np.abs(z342[CONF_HEAD]).astype(np.float32)
    bucket, qs = _bucket_indices(conf_proxy)

    # Build per-sample w33 lookup vector
    w33_vec = np.full(conf_proxy.shape, 0.5, dtype=np.float32)
    for q_id in range(4):
        w33_vec[bucket == q_id] = w33_per_q[q_id]
    # rows with bucket == -1 (NaN conf) fall back to uniform 0.5 (no surprise)

    out: dict = {}
    keys = sorted(set(z33.files) & set(z342.files))
    for k in keys:
        a33 = z33[k]
        a342 = z342[k]
        if a33.shape == ():
            out[k] = a33
            continue
        if a33.shape != a342.shape:
            out[k] = a33
            continue
        if k.startswith("pred_log_ret_") and not k.endswith(("_q10", "_q50", "_q90")):
            # CONDITIONAL: w33 per quartile
            ok33 = np.isfinite(a33)
            ok342 = np.isfinite(a342)
            both = ok33 & ok342
            avg = np.full_like(a33, np.nan, dtype=np.float32)
            w = w33_vec
            avg[both] = (
                w[both] * a33[both].astype(np.float32)
                + (1.0 - w[both]) * a342[both].astype(np.float32)
            )
            only33 = ok33 & ~ok342
            only342 = ok342 & ~ok33
            avg[only33] = a33[only33].astype(np.float32)
            avg[only342] = a342[only342].astype(np.float32)
            out[k] = avg
        elif k.startswith("pred_"):
            # UNIFORM 0.5/0.5 for non-directional heads (same as boost (a)
            # so MFE/MAE/FIFO heads used by `use_fifo_confluence` etc are stable)
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
        else:
            # target_*, mask_*, meta — copy from v3.3 (= v3.4.2 on same OOT)
            out[k] = a33

    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NPZ_DIR / f"{scheme_name}.npz"
    np.savez_compressed(out_path, **out)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  [{scheme_name}] w33/q = {w33_per_q}  bins q25/q50/q75 = "
          f"{qs[0]:.4f}/{qs[1]:.4f}/{qs[2]:.4f}  → {out_path.name} ({size_mb:.1f} MB)")
    return out_path


def run_validator(scheme_name: str, npz_path: Path) -> dict:
    """Run the existing LOO validator against v3.4.2 sweep top-K configs."""
    stage = OUT_DIR / f"_stage_{scheme_name}"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(V342_SWEEP_DIR / "best_configs.json", stage / "best_configs.json")
    cmd = [
        sys.executable, str(VALIDATOR),
        "--sweep-dir", str(stage),
        "--preds", str(npz_path),
        "--labels-dir", str(LABELS_DIR),
        "--top-k", str(TOP_K),
    ]
    print(f"\n[validate:{scheme_name}] launching validator...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        print(f"[validate:{scheme_name}] FAILED rc={proc.returncode}")
        print(proc.stderr[-1500:])
        return {"scheme": scheme_name, "ok": False, "err": proc.stderr[-500:]}

    # tail of stdout
    print(proc.stdout[-2500:])
    robust_p = stage / "loo_robust_configs.json"
    results_p = stage / "loo_validation_results.json"
    robust = json.loads(robust_p.read_text()) if robust_p.exists() else []
    results = json.loads(results_p.read_text()) if results_p.exists() else []
    # promote
    shutil.copy2(robust_p, OUT_DIR / f"loo_robust_{scheme_name}.json")
    shutil.copy2(results_p, OUT_DIR / f"loo_results_{scheme_name}.json")
    return {
        "scheme": scheme_name,
        "ok": True,
        "n_tested": len(results),
        "n_robust": len(robust),
        "robust_trials": [r.get("trial") for r in robust],
        "top3": [
            {
                "trial": r.get("trial"),
                "horizon": r["params"].get("head_horizon"),
                "side": r["params"].get("side"),
                "order_type": r["params"].get("order_type"),
                "mean_sh": round(r.get("mean_day_sharpe", 0.0), 2),
                "worst_sh": round(r.get("worst_day_sharpe", 0.0), 2),
                "fills": r.get("n_fills_total"),
            }
            for r in robust[:3]
        ],
    }


def write_verdict(all_results: list[dict]) -> Path:
    md: list[str] = []
    md.append("# Boosting (f) verdict — confidence-conditional ensemble v3.3 / v3.4.2")
    md.append("")
    md.append(f"HC #427 R5 boosting technique #5 (f). Generated: {TIMESTAMP}")
    md.append("")
    md.append("## Hypothesis")
    md.append("")
    md.append("Boost (a) (uniform 0.5/0.5) tied v3.4.2 SOLO on v3.4.2-basis (11 vs 12) "
              "but BEAT v3.3 SOLO (+43% on v3.3-basis). Boost (c) (single global weight "
              "sweep) was NULL — no scalar weight beat baseline. Implies the optimal "
              "weight varies across the prediction distribution. **Confidence-conditional "
              "weighting** lets each model take over where its edge is strongest.")
    md.append("")
    md.append("## Confidence proxy")
    md.append("")
    md.append(f"`|v3.4.2 {CONF_HEAD}|` — dominant horizon among the 12 LOO-robust v3.4.2 "
              "configs. Quartile bins computed globally across all 241,351 OOT samples.")
    md.append("")
    md.append("## Scheme results (v3.4.2 sweep top-20 basis, LOO across 5 OOT days)")
    md.append("")
    md.append("| scheme | w33(Q1,Q2,Q3,Q4) | n_robust / 20 | top-trial worst-day Sh | top-trial mean Sh |")
    md.append("|---|---|---:|---:|---:|")
    for r in all_results:
        scheme = r["scheme"]
        w33 = SCHEMES[scheme]
        n_robust = r.get("n_robust", "ERR")
        top = r.get("top3", [])
        worst = top[0]["worst_sh"] if top else "—"
        mean = top[0]["mean_sh"] if top else "—"
        md.append(f"| {scheme} | {w33} | {n_robust} | {worst} | {mean} |")
    md.append("")

    # Find best scheme (highest n_robust)
    valid = [r for r in all_results if r.get("ok")]
    valid.sort(key=lambda r: r.get("n_robust", 0), reverse=True)
    baseline = next((r for r in valid if r["scheme"] == "baseline_v342"), None)
    baseline_n = baseline.get("n_robust", 12) if baseline else 12
    winner = valid[0] if valid else None

    md.append("## Verdict")
    md.append("")
    if winner and winner.get("n_robust", 0) > baseline_n:
        md.append(f"✅ **POSITIVE** — scheme `{winner['scheme']}` produced "
                  f"**{winner['n_robust']}/20 robust** vs baseline v3.4.2 SOLO "
                  f"{baseline_n}/20 (Δ = +{winner['n_robust'] - baseline_n}).")
        md.append("")
        md.append("Top-3 robust configs under winning scheme:")
        for r in winner.get("top3", []):
            md.append(f"- trial={r['trial']} | {r['horizon']}/{r['side']}/{r['order_type']} "
                      f"| mean_Sh={r['mean_sh']} worst_Sh={r['worst_sh']} fills={r['fills']}")
    elif winner and winner.get("n_robust", 0) == baseline_n:
        md.append(f"⚪ **NULL** — best scheme `{winner['scheme']}` matched baseline "
                  f"{baseline_n}/20 robust. Confidence-conditional weighting offers no "
                  "boost on v3.4.2-basis configs. (May still help v3.3-basis — separate run.)")
    else:
        md.append(f"❌ **NEGATIVE** — no scheme beat baseline {baseline_n}/20. "
                  "Confidence-conditional ensembling appears to dilute v3.4.2's edge.")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("If the winning scheme is `monotone_v342` (low-conf → v3.3, high-conf → v3.4.2): "
              "consistent with v3.3 being better calibrated in the tail of weak signals while "
              "v3.4.2's sharper predictions dominate high-conviction trades. If `monotone_v33` "
              "wins: v3.4.2 over-confident in the tails, v3.3 corrects. If extremes-pure wins: "
              "blending hurts at conviction extremes (signal is bimodal). NULL/NEGATIVE → the "
              "two models do not separate edge by confidence regime.")
    md.append("")
    md.append("## HC #427 R5 counter")
    md.append("")
    md.append("Techniques tested so far (per session #67 + this run):")
    md.append("- (a) mean ensemble v3.3+v3.4.2 — ✅ POSITIVE on v3.3 basis (+43% n_robust)")
    md.append("- (b) meta-LGBM gate — ❌ NEGATIVE")
    md.append("- (c) weighted-ensemble sweep — ⚪ NULL")
    md.append(f"- (f) confidence-conditional ensemble — see verdict above")
    md.append("")

    p = OUT_DIR / "verdict.md"
    p.write_text("\n".join(md))
    return p


def main() -> int:
    print(f"== HC #427 R5 boosting (f): confidence-conditional ensemble ==")
    print(f"OUT_DIR = {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build all NPZs first (fast — ~30 sec each)
    print("\n[1/2] Building 8 conditional-ensemble NPZs...")
    npz_paths: dict[str, Path] = {}
    for scheme, w33 in SCHEMES.items():
        npz_paths[scheme] = build_conditional_npz(scheme, w33)

    # Run validator on each (slow — ~5 min each on Jupiter CPU)
    print("\n[2/2] Running LOO validator on each scheme...")
    all_results: list[dict] = []
    for scheme, p in npz_paths.items():
        r = run_validator(scheme, p)
        all_results.append(r)
        # incremental write of running summary
        (OUT_DIR / "summary.json").write_text(json.dumps({
            "schemes": SCHEMES,
            "results": all_results,
        }, indent=2, default=str))

    # Final summary + verdict
    summary = {
        "timestamp": TIMESTAMP,
        "out_dir": str(OUT_DIR),
        "conf_head": CONF_HEAD,
        "schemes": SCHEMES,
        "n_oot_days": 5,
        "top_k_configs_basis": TOP_K,
        "results": all_results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    verdict_p = write_verdict(all_results)
    print(f"\n[done] verdict: {verdict_p}")
    print(f"[done] summary: {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
