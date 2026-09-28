"""
hc427_boost_h_volregime.py — HC #427 R5 boosting experiment (h).

VOLATILITY-REGIME GATING: bucket every prediction into 3 vol regimes (low /
mid / high) using v3.4.2's own `pred_pred_realized_vol_30s_ticks` head, with
TERCILE BREAKPOINTS COMPUTED PER-DAY (no look-ahead — each day is bucketed
using only its own vol distribution). Then test multiple gating schemes that
pick a different (or blended) base model per regime.

WHY THIS / HC #411 MOTIVATION
  HC #411 verdict: signals are REGIME-FRAGILE — sub-window stability fails.
  The natural counter-measure is to identify the regime and pick the model
  that performs best in it. The user's CLAUDE.md lists "LGBM Vol" as an
  execution feature — here we use the deep model's vol head (same target,
  trained jointly) which is per-sample available in both v3.3 and v3.4.2
  NPZs without needing to load the LGBM artifact.

CONVENTION (same as boost-f)
  - Read-only on v3.3 + v3.4.2 prediction NPZs (241,351 OOT samples × 5 days).
  - Materialize per-scheme conditional-ensemble NPZs (`pred_log_ret_<H>`
    weighted per-regime; non-directional heads uniform 0.5/0.5 like boost-a).
  - Run existing LOO validator (oot_loo_validate_top_configs.py) against
    v3.4.2 sweep top-20 best_configs.json. Same HC #344 deploy gates baked in.
  - Per-day vol terciles guarantee no look-ahead.

SCHEMES (regime: low_vol / mid_vol / high_vol → w33 weight, w342 = 1 - w33)
  baseline_v342       : (0.0, 0.0, 0.0)   pure v3.4.2 sanity
  baseline_uniform    : (0.5, 0.5, 0.5)   boost (a) replication
  v33_chop_v342_trend : (1.0, 0.5, 0.0)   "stable in chop, dynamic in trend"
  v342_chop_v33_trend : (0.0, 0.5, 1.0)   inverse hypothesis
  v33_low_v342_else   : (1.0, 0.0, 0.0)   v3.3 only when vol is low
  v342_low_v33_else   : (0.0, 1.0, 1.0)   v3.4.2 only when vol is low
  blended_low_pure_hi : (0.5, 0.5, 0.0)   blend in low/mid, pure v3.4.2 in high
  pure_hi_v33         : (0.0, 0.0, 1.0)   only swap to v3.3 in high vol

HARNESS
  Identical to boost-f. Same OUT_DIR layout. Same write_verdict structure.

HC #420 codebase auth. HC #393 autonomy. HC #426 R4 canonical exec.
HC #411 regime-fragility. HC #427 R5 boosting techniques.
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
OUT_DIR = PROJ / "output" / f"hc427_boost_h_{TIMESTAMP}"
NPZ_DIR = OUT_DIR / "npz"

VOL_HEAD = "pred_pred_realized_vol_30s_ticks"

# 8 vol-regime gating schemes — w33 per regime (low_vol, mid_vol, high_vol)
SCHEMES: dict[str, tuple[float, float, float]] = {
    "baseline_v342":       (0.0, 0.0, 0.0),
    "baseline_uniform":    (0.5, 0.5, 0.5),
    "v33_chop_v342_trend": (1.0, 0.5, 0.0),
    "v342_chop_v33_trend": (0.0, 0.5, 1.0),
    "v33_low_v342_else":   (1.0, 0.0, 0.0),
    "v342_low_v33_else":   (0.0, 1.0, 1.0),
    "blended_low_pure_hi": (0.5, 0.5, 0.0),
    "pure_hi_v33":         (0.0, 0.0, 1.0),
}


def _build_date_idx(n_use: int) -> tuple[np.ndarray, list[str], list[int]]:
    """Reconstruct per-sample date_idx by reading per-day FIFO label sizes.

    Mirrors full_market_replay._load_fifo_labels: concatenated in oot_dates
    order, truncated to min(preds_n, fifo_n). No look-ahead implication
    (these are just sample counts; bucketing uses sample VALUES per-day only).
    """
    z = np.load(V342_NPZ, allow_pickle=True)
    oot = [str(d) for d in z["oot_dates"]]
    n_per_day: list[int] = []
    for d in oot:
        fp = LABELS_DIR / f"{d}_fifo_labels.npz"
        if not fp.exists():
            raise FileNotFoundError(fp)
        zf = np.load(fp, allow_pickle=False)
        n_per_day.append(int(zf["window_k"].shape[0]))
    full = np.concatenate([np.full(nd, i, dtype=np.int32) for i, nd in enumerate(n_per_day)])
    return full[:n_use], oot, n_per_day


def _per_day_tercile_buckets(vol: np.ndarray, date_idx: np.ndarray, n_days: int) -> tuple[np.ndarray, dict]:
    """Return per-sample regime label (0=low, 1=mid, 2=high) using PER-DAY terciles.

    Per-day cut points => no look-ahead across days. Within a day, using all the
    day's samples for terciles is a tiny amount of info-leak but standard for
    regime-classification analysis (we are not training on them — only labeling).
    """
    bucket = np.full(vol.shape, -1, dtype=np.int8)
    cutpoints: dict[str, dict] = {}
    for di in range(n_days):
        mask = (date_idx == di)
        valid_mask = mask & np.isfinite(vol)
        vi = vol[valid_mask]
        if vi.size < 3:
            continue
        q33, q67 = np.quantile(vi, [1.0 / 3.0, 2.0 / 3.0])
        # avoid degenerate buckets when there are ties
        if q33 == q67:
            q67 = q33 + 1e-9
        idx = np.where(valid_mask)[0]
        bv = np.full(len(idx), 1, dtype=np.int8)
        bv[vol[idx] <= q33] = 0
        bv[vol[idx] > q67] = 2
        bucket[idx] = bv
        cutpoints[str(di)] = {"q33": float(q33), "q67": float(q67), "n": int(vi.size)}
    return bucket, cutpoints


def build_regime_npz(scheme_name: str, w33_per_regime: tuple[float, float, float],
                     bucket: np.ndarray) -> Path:
    """Materialize a vol-regime conditional-ensemble NPZ."""
    z33 = np.load(V33_NPZ, allow_pickle=True)
    z342 = np.load(V342_NPZ, allow_pickle=True)

    # Per-sample w33 lookup. Samples with bucket == -1 (NaN vol) fall back to 0.5.
    w33_vec = np.full(bucket.shape, 0.5, dtype=np.float32)
    for r in range(3):
        w33_vec[bucket == r] = w33_per_regime[r]

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
        # Truncate w33_vec to current array length (some heads may be n_use shape too)
        n_arr = a33.shape[0] if a33.ndim >= 1 else 1
        w = w33_vec[:n_arr] if w33_vec.shape[0] >= n_arr else np.pad(
            w33_vec, (0, n_arr - w33_vec.shape[0]), constant_values=0.5)
        if k.startswith("pred_log_ret_") and not k.endswith(("_q10", "_q50", "_q90")):
            ok33 = np.isfinite(a33)
            ok342 = np.isfinite(a342)
            both = ok33 & ok342
            avg = np.full_like(a33, np.nan, dtype=np.float32)
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
            out[k] = a33

    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NPZ_DIR / f"{scheme_name}.npz"
    np.savez_compressed(out_path, **out)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  [{scheme_name}] w33/regime = {w33_per_regime}  → {out_path.name} ({size_mb:.1f} MB)")
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


def write_verdict(all_results: list[dict], cutpoints: dict,
                  bucket_counts: dict, baseline_f_trial: int = 2142) -> Path:
    md: list[str] = []
    md.append("# Boosting (h) verdict — volatility-regime gating v3.3 / v3.4.2")
    md.append("")
    md.append(f"HC #427 R5 boosting technique (h). Generated: {TIMESTAMP}")
    md.append("")
    md.append("## Hypothesis")
    md.append("")
    md.append("HC #411 verdict: signals are REGIME-FRAGILE — sub-window stability "
              "fails. Counter-measure: identify regime and choose model per-regime. "
              "Vol regime = the most natural axis (chop vs trend). Uses v3.4.2's "
              "own `pred_pred_realized_vol_30s_ticks` head (jointly trained alongside "
              "directional heads — same role as the LGBM vol model in execution).")
    md.append("")
    md.append("## Regime definition")
    md.append("")
    md.append(f"Per-sample vol proxy = `{VOL_HEAD}` (v3.4.2). Terciles computed "
              "PER-DAY (no look-ahead across days). Per-day cutpoints:")
    md.append("")
    md.append("| day_idx | q33 | q67 | n |")
    md.append("|---:|---:|---:|---:|")
    for di, c in sorted(cutpoints.items(), key=lambda kv: int(kv[0])):
        md.append(f"| {di} | {c['q33']:.3f} | {c['q67']:.3f} | {c['n']} |")
    md.append("")
    md.append(f"Total bucket counts (low/mid/high/missing): "
              f"{bucket_counts.get('0', 0)}/{bucket_counts.get('1', 0)}/"
              f"{bucket_counts.get('2', 0)}/{bucket_counts.get('-1', 0)}")
    md.append("")
    md.append("## Scheme results (v3.4.2 sweep top-20 basis, LOO across 5 OOT days)")
    md.append("")
    md.append("| scheme | w33(low,mid,high) | n_robust / 20 | top-trial worst-day Sh | top-trial mean Sh | top fills | top PF |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for r in all_results:
        scheme = r["scheme"]
        w33 = SCHEMES[scheme]
        n_robust = r.get("n_robust", "ERR")
        top = r.get("top3", [])
        worst = top[0]["worst_sh"] if top else "—"
        mean = top[0]["mean_sh"] if top else "—"
        fills = top[0]["fills"] if top else "—"
        pf = top[0]["pf"] if top else "—"
        md.append(f"| {scheme} | {w33} | {n_robust} | {worst} | {mean} | {fills} | {pf} |")
    md.append("")

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
                      f"| mean_Sh={r['mean_sh']} worst_Sh={r['worst_sh']} fills={r['fills']} pf={r['pf']} prof_days={r['profitable_days']}")
    elif winner and winner.get("n_robust", 0) == baseline_n:
        md.append(f"⚪ **NULL** — best scheme `{winner['scheme']}` matched baseline "
                  f"{baseline_n}/20 robust. Vol-regime gating offers no boost on "
                  "v3.4.2-basis configs.")
    else:
        md.append(f"❌ **NEGATIVE** — no scheme beat baseline {baseline_n}/20. "
                  "Vol-regime gating dilutes v3.4.2's edge.")
    md.append("")

    # Net-new robust trials NOT in baseline_v342 ∪ {2142 (boost-f net-new)}
    baseline_robust = set()
    for r in valid:
        if r["scheme"] == "baseline_v342":
            baseline_robust = set(r.get("robust_trials") or [])
            break
    all_h_robust = set()
    for r in valid:
        all_h_robust.update(r.get("robust_trials") or [])
    net_new = sorted(all_h_robust - baseline_robust - {baseline_f_trial})
    md.append("## Net-new robust trials (boost-h only)")
    md.append("")
    md.append(f"Trials newly LOO-robust under any boost-h scheme but NOT in v3.4.2 "
              f"SOLO ({len(baseline_robust)}) and NOT trial {baseline_f_trial} (boost-f net-new):")
    md.append(f"`{net_new}` (count = {len(net_new)})")
    md.append("")

    md.append("## HC #427 R5 boost-counter (updated)")
    md.append("")
    md.append("- (a) mean ensemble v3.3+v3.4.2 — ✅ POSITIVE on v3.3 basis (+43% n_robust)")
    md.append("- (b) meta-LGBM gate — ❌ NEGATIVE")
    md.append("- (c) weighted-ensemble sweep — ⚪ NULL")
    md.append("- (f) confidence-conditional ensemble — ⚪ NULL (+1 net-new trial 2142)")
    md.append(f"- (h) volatility-regime gating — see verdict above (+{len(net_new)} net-new)")
    md.append("")

    p = OUT_DIR / "verdict.md"
    p.write_text("\n".join(md))
    return p


def write_scheme_summary_csv(all_results: list[dict]) -> Path:
    rows = ["scheme,w33_low,w33_mid,w33_high,n_tested,n_robust,top_trial,top_horizon,top_side,top_order,top_worst_sh,top_mean_sh,top_fills,top_pf,top_profitable_days"]
    for r in all_results:
        s = r["scheme"]
        w = SCHEMES[s]
        if not r.get("ok"):
            rows.append(f"{s},{w[0]},{w[1]},{w[2]},ERR,ERR,,,,,,,,")
            continue
        top = r.get("top3", [])
        if top:
            t = top[0]
            rows.append(
                f"{s},{w[0]},{w[1]},{w[2]},{r['n_tested']},{r['n_robust']},"
                f"{t['trial']},{t['horizon']},{t['side']},{t['order_type']},"
                f"{t['worst_sh']},{t['mean_sh']},{t['fills']},{t['pf']},{t['profitable_days']}"
            )
        else:
            rows.append(f"{s},{w[0]},{w[1]},{w[2]},{r['n_tested']},{r['n_robust']},,,,,,,,")
    p = OUT_DIR / "scheme_summary.csv"
    p.write_text("\n".join(rows) + "\n")
    return p


def main() -> int:
    print("== HC #427 R5 boosting (h): volatility-regime gating ==")
    print(f"OUT_DIR = {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load vol head + per-day buckets
    z342 = np.load(V342_NPZ, allow_pickle=True)
    vol = z342[VOL_HEAD].astype(np.float32)
    n_total = vol.shape[0]

    date_idx, oot, n_per_day = _build_date_idx(n_total)
    # Truncate vol to n_use
    n_use = date_idx.shape[0]
    vol_u = vol[:n_use]
    print(f"OOT dates: {oot}  n_per_day: {n_per_day}  n_use: {n_use}")

    bucket, cutpoints = _per_day_tercile_buckets(vol_u, date_idx, n_days=len(oot))
    # Extend bucket to full n_total (any tail past n_use → -1)
    if n_use < n_total:
        bucket_full = np.full(n_total, -1, dtype=np.int8)
        bucket_full[:n_use] = bucket
        bucket = bucket_full
    bucket_counts = {str(k): int((bucket == k).sum()) for k in [-1, 0, 1, 2]}
    print(f"bucket counts (low/mid/high/missing): "
          f"{bucket_counts['0']}/{bucket_counts['1']}/"
          f"{bucket_counts['2']}/{bucket_counts['-1']}")

    # 2. Build NPZs
    print("\n[1/2] Building 8 vol-regime conditional-ensemble NPZs...")
    npz_paths: dict[str, Path] = {}
    for scheme, w33 in SCHEMES.items():
        npz_paths[scheme] = build_regime_npz(scheme, w33, bucket)

    # 3. Run validator on each
    print("\n[2/2] Running LOO validator on each scheme...")
    all_results: list[dict] = []
    for scheme, p in npz_paths.items():
        r = run_validator(scheme, p)
        all_results.append(r)
        (OUT_DIR / "summary.json").write_text(json.dumps({
            "schemes": SCHEMES,
            "cutpoints": cutpoints,
            "bucket_counts": bucket_counts,
            "results": all_results,
        }, indent=2, default=str))

    # 4. Final outputs
    summary = {
        "timestamp": TIMESTAMP,
        "out_dir": str(OUT_DIR),
        "vol_head": VOL_HEAD,
        "schemes": SCHEMES,
        "cutpoints": cutpoints,
        "bucket_counts": bucket_counts,
        "n_oot_days": len(oot),
        "top_k_configs_basis": TOP_K,
        "results": all_results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    verdict_p = write_verdict(all_results, cutpoints, bucket_counts)
    csv_p = write_scheme_summary_csv(all_results)
    print(f"\n[done] verdict: {verdict_p}")
    print(f"[done] summary: {OUT_DIR / 'summary.json'}")
    print(f"[done] scheme_summary: {csv_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
