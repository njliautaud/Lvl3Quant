"""
hc427_boost_g_horizon_stacking.py — HC #427 R5 boosting experiment (g).

HORIZON-STACKING: blend shorter-horizon prediction heads into longer-horizon
heads (or vice versa) within v3.4.2's own multi-horizon outputs. Tests whether
cross-horizon agreement carries incremental information beyond the per-horizon
trained heads — a self-stacking ensemble (no v3.3 dependency).

WHY THIS / HC #427 R5 MOTIVATION
  Signals decay rapidly past 30s (CLAUDE.md Key Signal Characteristics).
  But the 1s/5s/10s/30s heads are JOINTLY trained on the same shared encoder,
  so they tend to be highly correlated near zero and diverge in tails. The
  stacking hypothesis: a 30s decision is sharper when 1s + 5s + 10s + 30s
  AGREE on direction (consensus = stronger signal); a single-horizon decision
  loses this cross-horizon confluence information. We test by materially
  blending shorter-horizon preds into each target head and re-validating
  against the existing v3.4.2 sweep top-20 best_configs.

CONVENTION (same as boost-f / boost-h)
  - Read-only on v3.4.2 prediction NPZ (241,351 OOT samples × 5 days).
  - Materialize per-scheme stacked NPZs (only the `pred_log_ret_<H>` heads
    are modified; non-directional heads — MFE, MAE, vol, FIFO — copy verbatim
    from v3.4.2). V3.3 is NOT used here (pure self-stacking).
  - Run existing LOO validator against v3.4.2 sweep top-20 best_configs.
  - Same HC #344 deploy gates baked in via validator.

SCHEMES (each describes the new `pred_log_ret_<H>` definition)
  baseline_v342           : no change                                   (control)
  stack_5s_with_1s        : pred_5s   = 0.70·5s   + 0.30·1s
  stack_10s_with_5s_1s    : pred_10s  = 0.60·10s  + 0.25·5s   + 0.15·1s
  stack_30s_with_10s_5s   : pred_30s  = 0.60·30s  + 0.25·10s  + 0.15·5s
  pyramid_10s_full        : pred_10s  = 0.55·10s  + 0.25·5s   + 0.20·1s
  pyramid_30s_full        : pred_30s  = 0.40·30s  + 0.25·10s  + 0.20·5s  + 0.15·1s
  all_short_to_long_mild  : every long head += 0.15·1s, renormalized to 1.0
  consensus_4h_uniform    : each pred_<H> = 0.25·(1s + 5s + 10s + 30s)
                             (replace each with cross-horizon mean)

Only the target horizon's array is overwritten per scheme — the others keep
their stock v3.4.2 values. So the validator picks up the boosted signal only
when its sweep config uses `head_horizon` matching the modified head.

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
V342_NPZ = PROJ / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "fold_00_predictions.npz"
V342_SWEEP_DIR = PROJ / "output" / "v342_execution_optuna_20260518"
LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
VALIDATOR = PROJ / "scripts" / "v3_3_research" / "oot_loo_validate_top_configs.py"
TOP_K = 20

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJ / "output" / f"hc427_boost_g_{TIMESTAMP}"
NPZ_DIR = OUT_DIR / "npz"

# A "stacking recipe" maps target_head_name -> dict of {source_head: weight}.
# Weights are NOT auto-normalized; explicit in the recipe.
# Target heads NOT in the recipe are copied verbatim from v3.4.2.
SCHEMES: dict[str, dict[str, dict[str, float]]] = {
    "baseline_v342": {},  # no changes
    "stack_5s_with_1s": {
        "pred_log_ret_5s": {"pred_log_ret_5s": 0.70, "pred_log_ret_1s": 0.30},
    },
    "stack_10s_with_5s_1s": {
        "pred_log_ret_10s": {"pred_log_ret_10s": 0.60, "pred_log_ret_5s": 0.25,
                              "pred_log_ret_1s": 0.15},
    },
    "stack_30s_with_10s_5s": {
        "pred_log_ret_30s": {"pred_log_ret_30s": 0.60, "pred_log_ret_10s": 0.25,
                              "pred_log_ret_5s": 0.15},
    },
    "pyramid_10s_full": {
        "pred_log_ret_10s": {"pred_log_ret_10s": 0.55, "pred_log_ret_5s": 0.25,
                              "pred_log_ret_1s": 0.20},
    },
    "pyramid_30s_full": {
        "pred_log_ret_30s": {"pred_log_ret_30s": 0.40, "pred_log_ret_10s": 0.25,
                              "pred_log_ret_5s": 0.20, "pred_log_ret_1s": 0.15},
    },
    "all_short_to_long_mild": {
        "pred_log_ret_5s":  {"pred_log_ret_5s":  0.85, "pred_log_ret_1s": 0.15},
        "pred_log_ret_10s": {"pred_log_ret_10s": 0.85, "pred_log_ret_1s": 0.15},
        "pred_log_ret_30s": {"pred_log_ret_30s": 0.85, "pred_log_ret_1s": 0.15},
    },
    "consensus_4h_uniform": {
        "pred_log_ret_1s":  {"pred_log_ret_1s":  0.25, "pred_log_ret_5s": 0.25,
                              "pred_log_ret_10s": 0.25, "pred_log_ret_30s": 0.25},
        "pred_log_ret_5s":  {"pred_log_ret_1s":  0.25, "pred_log_ret_5s": 0.25,
                              "pred_log_ret_10s": 0.25, "pred_log_ret_30s": 0.25},
        "pred_log_ret_10s": {"pred_log_ret_1s":  0.25, "pred_log_ret_5s": 0.25,
                              "pred_log_ret_10s": 0.25, "pred_log_ret_30s": 0.25},
        "pred_log_ret_30s": {"pred_log_ret_1s":  0.25, "pred_log_ret_5s": 0.25,
                              "pred_log_ret_10s": 0.25, "pred_log_ret_30s": 0.25},
    },
}


def build_stacked_npz(scheme_name: str,
                       recipe: dict[str, dict[str, float]]) -> Path:
    """Materialize a horizon-stacked v3.4.2 NPZ."""
    z342 = np.load(V342_NPZ, allow_pickle=True)
    out: dict = {}

    # Copy everything verbatim first, then overwrite the recipe targets.
    for k in z342.files:
        out[k] = z342[k]

    for target, source_weights in recipe.items():
        if target not in z342.files:
            print(f"  [WARN] {scheme_name}: target {target} not in NPZ — skipping")
            continue
        # Sanity: all sources present
        missing = [s for s in source_weights if s not in z342.files]
        if missing:
            print(f"  [WARN] {scheme_name}: missing sources {missing} for {target} — skipping")
            continue

        # Linear blend; nan-aware (mask out non-finite entries per source)
        blended = np.zeros_like(z342[target], dtype=np.float32)
        total_w = np.zeros_like(z342[target], dtype=np.float32)
        for src, w in source_weights.items():
            arr = z342[src].astype(np.float32)
            ok = np.isfinite(arr)
            blended[ok] += w * arr[ok]
            total_w[ok] += w

        # Renormalize where total_w differs from 1 (handles partial NaN coverage)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.where(total_w > 0, blended / total_w, np.nan)
        out[target] = normalized.astype(np.float32)

    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NPZ_DIR / f"{scheme_name}.npz"
    np.savez_compressed(out_path, **out)
    size_mb = out_path.stat().st_size / 1e6
    n_heads_modified = len(recipe)
    print(f"  [{scheme_name}] modified {n_heads_modified} heads → {out_path.name} ({size_mb:.1f} MB)")
    return out_path


def run_validator(scheme_name: str, npz_path: Path) -> dict:
    """Run the LOO validator against v3.4.2 sweep top-K configs."""
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

    print(proc.stdout[-2500:])
    robust_p = stage / "loo_robust_configs.json"
    results_p = stage / "loo_validation_results.json"
    robust = json.loads(robust_p.read_text()) if robust_p.exists() else []
    results = json.loads(results_p.read_text()) if results_p.exists() else []
    shutil.copy2(robust_p, OUT_DIR / f"loo_robust_{scheme_name}.json")
    shutil.copy2(results_p, OUT_DIR / f"loo_results_{scheme_name}.json")

    top3 = []
    for r in robust[:3]:
        top3.append({
            "trial": r.get("trial"),
            "horizon": r["params"].get("head_horizon"),
            "side": r["params"].get("side"),
            "order_type": r["params"].get("order_type"),
            "mean_sh": round(r.get("mean_day_sharpe", 0.0), 2),
            "worst_sh": round(r.get("worst_day_sharpe", 0.0), 2),
            "fills": r.get("n_fills_total"),
            "profitable_days": r.get("n_profitable_days"),
            "pf": round(r.get("pf_total", 0.0), 2),
        })
    return {
        "scheme": scheme_name,
        "ok": True,
        "n_tested": len(results),
        "n_robust": len(robust),
        "robust_trials": [r.get("trial") for r in robust],
        "top3": top3,
    }


def write_verdict(all_results: list[dict],
                  excluded_trials: set[int]) -> Path:
    """Write verdict.md using same structure as boost-h."""
    md: list[str] = []
    md.append("# Boosting (g) verdict — horizon-stacking (v3.4.2 self-ensemble)")
    md.append("")
    md.append(f"HC #427 R5 boosting technique (g). Generated: {TIMESTAMP}")
    md.append("")
    md.append("## Hypothesis")
    md.append("")
    md.append("v3.4.2's 1s/5s/10s/30s prediction heads are jointly trained on a "
              "shared encoder — they share information but produce distinct "
              "horizon-specific outputs. Stacking hypothesis: a longer-horizon "
              "decision is sharper when shorter-horizon heads AGREE on direction "
              "(cross-horizon confluence). We test by linearly blending shorter "
              "heads into each target head and re-running the existing LOO "
              "validator. Pure self-ensemble — v3.3 not used here.")
    md.append("")
    md.append("## Scheme leaderboard (v3.4.2 sweep top-20 basis, LOO across 5 OOT days)")
    md.append("")
    md.append("| scheme | n_robust / 20 | top-trial worst-day Sh | top-trial mean Sh | top fills | top PF |")
    md.append("|---|---:|---:|---:|---:|---:|")
    sorted_results = sorted(all_results,
                             key=lambda r: r.get("n_robust", 0),
                             reverse=True)
    for r in sorted_results:
        scheme = r["scheme"]
        n_robust = r.get("n_robust", "ERR")
        top = r.get("top3", [])
        worst = top[0]["worst_sh"] if top else "—"
        mean = top[0]["mean_sh"] if top else "—"
        fills = top[0]["fills"] if top else "—"
        pf = top[0]["pf"] if top else "—"
        md.append(f"| {scheme} | {n_robust} | {worst} | {mean} | {fills} | {pf} |")
    md.append("")

    valid = [r for r in all_results if r.get("ok")]
    valid.sort(key=lambda r: r.get("n_robust", 0), reverse=True)
    baseline = next((r for r in valid if r["scheme"] == "baseline_v342"), None)
    baseline_n = baseline.get("n_robust", 12) if baseline else 12
    winner = valid[0] if valid else None

    md.append("## Verdict")
    md.append("")
    if winner and winner.get("n_robust", 0) > baseline_n:
        delta = winner["n_robust"] - baseline_n
        md.append(f"✅ **POSITIVE** — scheme `{winner['scheme']}` produced "
                  f"**{winner['n_robust']}/20 robust** vs baseline v3.4.2 SOLO "
                  f"{baseline_n}/20 (Δ = +{delta}).")
        md.append("")
        md.append("Top-3 robust configs under winning scheme:")
        for r in winner.get("top3", []):
            md.append(f"- trial={r['trial']} | {r['horizon']}/{r['side']}/{r['order_type']} "
                      f"| mean_Sh={r['mean_sh']} worst_Sh={r['worst_sh']} "
                      f"fills={r['fills']} pf={r['pf']} prof_days={r['profitable_days']}")
    elif winner and winner.get("n_robust", 0) == baseline_n:
        md.append(f"⚪ **NULL** — best scheme `{winner['scheme']}` matched baseline "
                  f"{baseline_n}/20 robust. Horizon-stacking offers no boost.")
    else:
        md.append(f"❌ **NEGATIVE** — no scheme beat baseline {baseline_n}/20. "
                  "Horizon-stacking dilutes per-horizon edge.")
    md.append("")

    # Net-new robust trials NOT in baseline ∪ already-counted boost-f/boost-h
    baseline_robust = set()
    for r in valid:
        if r["scheme"] == "baseline_v342":
            baseline_robust = set(r.get("robust_trials") or [])
            break
    all_g_robust = set()
    for r in valid:
        all_g_robust.update(r.get("robust_trials") or [])
    net_new = sorted(all_g_robust - baseline_robust - excluded_trials)
    md.append("## Net-new robust trials (boost-g only)")
    md.append("")
    md.append(f"Trials newly LOO-robust under any boost-g scheme but NOT in v3.4.2 "
              f"SOLO ({len(baseline_robust)}) and NOT already counted under "
              f"boost-f/h ({sorted(excluded_trials)}):")
    md.append(f"`{net_new}` (count = {len(net_new)})")
    md.append("")

    md.append("## HC #427 R5 boost-counter (updated)")
    md.append("")
    md.append("- (a) mean ensemble v3.3+v3.4.2 — ✅ POSITIVE on v3.3 basis (+43% n_robust)")
    md.append("- (b) meta-LGBM gate — ❌ NEGATIVE")
    md.append("- (c) weighted-ensemble sweep — ⚪ NULL")
    md.append("- (f) confidence-conditional ensemble — ⚪ NULL (+1 net-new trial 2142)")
    md.append("- (h) volatility-regime gating — ✅ POSITIVE (+4 net-new trials)")
    md.append(f"- (g) horizon-stacking — see verdict above (+{len(net_new)} net-new)")
    md.append("")

    p = OUT_DIR / "verdict.md"
    p.write_text("\n".join(md))
    return p


def write_scheme_summary_csv(all_results: list[dict]) -> Path:
    rows = ["scheme,n_tested,n_robust,top_trial,top_horizon,top_side,top_order,top_worst_sh,top_mean_sh,top_fills,top_pf,top_profitable_days"]
    for r in all_results:
        s = r["scheme"]
        if not r.get("ok"):
            rows.append(f"{s},ERR,ERR,,,,,,,,,")
            continue
        top = r.get("top3", [])
        if top:
            t = top[0]
            rows.append(
                f"{s},{r['n_tested']},{r['n_robust']},"
                f"{t['trial']},{t['horizon']},{t['side']},{t['order_type']},"
                f"{t['worst_sh']},{t['mean_sh']},{t['fills']},{t['pf']},{t['profitable_days']}"
            )
        else:
            rows.append(f"{s},{r['n_tested']},{r['n_robust']},,,,,,,,,")
    p = OUT_DIR / "scheme_summary.csv"
    p.write_text("\n".join(rows) + "\n")
    return p


def main() -> int:
    print("== HC #427 R5 boosting (g): horizon-stacking ==")
    print(f"OUT_DIR = {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build per-scheme stacked NPZs
    print(f"\n[1/2] Building {len(SCHEMES)} horizon-stacked NPZs...")
    npz_paths: dict[str, Path] = {}
    for scheme, recipe in SCHEMES.items():
        npz_paths[scheme] = build_stacked_npz(scheme, recipe)

    # 2. Run validator on each scheme
    print("\n[2/2] Running LOO validator on each scheme...")
    all_results: list[dict] = []
    for scheme, p in npz_paths.items():
        r = run_validator(scheme, p)
        all_results.append(r)
        (OUT_DIR / "summary.json").write_text(json.dumps({
            "schemes": list(SCHEMES.keys()),
            "results": all_results,
        }, indent=2, default=str))

    # Excluded trials = trial 2142 (boost-f net-new) ∪ trials 563/1296/2326 (boost-h net-new)
    excluded = {2142, 563, 1296, 2326}

    # 3. Final outputs
    summary = {
        "timestamp": TIMESTAMP,
        "out_dir": str(OUT_DIR),
        "schemes": {k: v for k, v in SCHEMES.items()},
        "excluded_already_counted_trials": sorted(excluded),
        "n_oot_days": 5,
        "top_k_configs_basis": TOP_K,
        "results": all_results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    verdict_p = write_verdict(all_results, excluded)
    csv_p = write_scheme_summary_csv(all_results)
    print(f"\n[done] verdict: {verdict_p}")
    print(f"[done] summary: {OUT_DIR / 'summary.json'}")
    print(f"[done] scheme_summary: {csv_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
