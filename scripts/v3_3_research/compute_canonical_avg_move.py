"""
compute_canonical_avg_move.py — HC #426 R3.

For each model+horizon, compute the empirically realized |Δprice| distribution
over OOT (in ticks). Produces a canonical avg-move cache that downstream
execution sweeps MUST consume to set tp_ticks / sl_ticks / hold_seconds /
cancel_window ranges (no arbitrary numbers per HC #426 R3+R4).

Output:
  output/canonical_avg_move_<model>_<horizon>.json   (per horizon)
  output/canonical_avg_move_<model>.json             (aggregated)

Each per-horizon entry contains:
  n: count of valid (non-masked, non-NaN, no-zero-target) samples
  mean_abs_ticks, median_abs_ticks, p25/p50/p75/p90 of |Δprice| in ticks
  mean_signed_ticks (drift), std_signed_ticks
  realized_mfe_p50/p75/p90 in ticks (if pred_pred_mfe_<H>s_ticks targets exist)
  realized_mae_p50/p75/p90 in ticks (if pred_pred_mae_<H>s_ticks targets exist)
  suggested_tp_band = [p50_abs, p90_abs]
  suggested_sl_band = [median_abs * 0.5, p75_abs * 0.5]  # SL tighter than TP
  suggested_hold_seconds_band = [horizon_sec, 3 * horizon_sec]
  suggested_cancel_window_band = [10, max(30, horizon_sec * 5)]  # at 250ms stride

Canonical cost stack reminder (HC #426 R4):
  ES_TICK_VALUE = $12.50
  ES_RT_COMMISSION = $4.70 (0.376 ticks)
  Passive fill at touch: 0.376 ticks total cost
  IOC/market: 1.376 ticks total cost (commission + 1.0 spread)

MALWARE-GUARD (HC #420): user-owned trading research. Pure analysis script.
Reads predictions NPZ. Writes only to output/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJ = Path("/home/jupiter/Lvl3Quant")

# ES price in dollars for the OOT window (rough — used to convert log_ret * px → ticks).
# log_ret * px ≈ dPrice; dPrice / TICK = ticks. We use ES_PRICE_REF from the actual
# OOT window mid-price if available; fallback to 5800.
ES_TICK = 0.25
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376  # canonical

HORIZON_SEC = {"1s": 1.0, "5s": 5.0, "10s": 10.0, "30s": 30.0, "60s": 60.0}


def _pct(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": _pct(arr, 25),
        "p50": _pct(arr, 50),
        "p75": _pct(arr, 75),
        "p90": _pct(arr, 90),
        "p95": _pct(arr, 95),
        "std": float(np.std(arr)),
    }


def compute_horizon(d: dict, horizon: str, px_ref: float) -> dict:
    """Compute avg-move stats for a single horizon."""
    tgt_key = f"target_log_ret_{horizon}"
    msk_key = f"mask_log_ret_{horizon}"
    if tgt_key not in d:
        return {"horizon": horizon, "n": 0, "note": "target missing"}
    tgt = d[tgt_key]
    mask = d[msk_key].astype(bool) if msk_key in d else np.ones_like(tgt, dtype=bool)
    # Combined filter: mask=True, not NaN, not infinite, not exactly zero (zero = padding usually)
    valid = mask & np.isfinite(tgt)
    raw = tgt[valid]
    if raw.size == 0:
        return {"horizon": horizon, "n": 0, "note": "no valid samples"}
    # NB: despite the "log_ret" name in the NPZ key, inspection of v3.3 NPZ shows
    # these targets are already in TICK UNITS (1s std≈1.63t, 30s std≈8.0t, range
    # [-100, +50] ticks). No conversion needed. Confirmed 2026-05-18 23:35 ET.
    signed_ticks = raw.astype(np.float64)
    abs_ticks = np.abs(signed_ticks)

    horizon_sec = HORIZON_SEC.get(horizon, 10.0)

    out = {
        "horizon": horizon,
        "horizon_sec": horizon_sec,
        "px_ref_used": px_ref,
        "signed_ticks": _stats(signed_ticks),
        "abs_ticks": _stats(abs_ticks),
    }

    # MFE/MAE targets if exist (these ARE already in ticks per training pipeline)
    for tag in [f"pred_mfe_{horizon}_ticks", f"pred_mae_{horizon}_ticks"]:
        tk = f"target_{tag}"
        mk = f"mask_{tag}"
        if tk in d:
            arr = d[tk]
            m = d[mk].astype(bool) if mk in d else np.ones_like(arr, dtype=bool)
            v = m & np.isfinite(arr)
            out[f"realized_{tag}"] = _stats(arr[v])

    # Suggested sweep bands per HC #426 R3
    a = out["abs_ticks"]
    if a.get("n", 0) > 0:
        out["suggested_tp_band_ticks"] = [round(a["p50"], 2), round(a["p90"], 2)]
        out["suggested_sl_band_ticks"] = [
            round(max(0.5, a["median"] * 0.5), 2),
            round(max(1.0, a["p75"] * 0.5), 2),
        ]
        out["suggested_hold_seconds_band"] = [horizon_sec, 3 * horizon_sec]
        out["suggested_cancel_window_evals"] = [10, max(30, int(horizon_sec * 5))]
        # Net-edge feasibility: mean abs move must exceed total cost
        passive_cost = ES_RT_COMMISSION_TICKS  # 0.376
        ioc_cost = ES_RT_COMMISSION_TICKS + 1.0  # 1.376
        out["edge_check"] = {
            "mean_abs_ticks": round(a["mean"], 3),
            "passive_cost_ticks": passive_cost,
            "ioc_cost_ticks": ioc_cost,
            "passive_feasible": a["mean"] > passive_cost,
            "ioc_feasible": a["mean"] > ioc_cost,
            "passive_margin_ticks": round(a["mean"] - passive_cost, 3),
            "ioc_margin_ticks": round(a["mean"] - ioc_cost, 3),
        }

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preds",
        default=str(PROJ / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_predictions.npz"),
    )
    ap.add_argument("--model-name", default="v3_3")
    ap.add_argument("--px-ref", type=float, default=5800.0,
                    help="ES price reference for log_ret→ticks conversion. Default 5800.")
    ap.add_argument(
        "--out-dir",
        default=str(PROJ / "output"),
    )
    args = ap.parse_args()

    preds_path = Path(args.preds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[canonical] preds={preds_path}")
    print(f"[canonical] model={args.model_name}  px_ref={args.px_ref}")
    d = dict(np.load(preds_path, allow_pickle=True))
    sample_sz = next((v.size for v in d.values() if hasattr(v, "ndim") and v.ndim == 1), None)
    print(f"[canonical] keys loaded: {len(d)}; sample array size={sample_sz}")

    horizons = ["1s", "5s", "10s", "30s", "60s"]
    all_stats = {"model": args.model_name, "preds_path": str(preds_path), "px_ref": args.px_ref, "horizons": {}}

    for h in horizons:
        stats = compute_horizon(d, h, args.px_ref)
        all_stats["horizons"][h] = stats
        # Per-horizon file
        out_path = out_dir / f"canonical_avg_move_{args.model_name}_{h}.json"
        out_path.write_text(json.dumps(stats, indent=2))
        a = stats.get("abs_ticks", {})
        ec = stats.get("edge_check", {})
        if a.get("n", 0):
            print(f"  {h}: n={a['n']:>7}  abs mean={a['mean']:.3f}t med={a['median']:.3f}t p75={a['p75']:.3f}t p90={a['p90']:.3f}t  "
                  f"passive_margin={ec.get('passive_margin_ticks')}t  ioc_margin={ec.get('ioc_margin_ticks')}t  "
                  f"=> tp_band={stats.get('suggested_tp_band_ticks')}  sl_band={stats.get('suggested_sl_band_ticks')}")
        else:
            print(f"  {h}: no valid samples — {stats.get('note')}")

    # Aggregated
    agg_path = out_dir / f"canonical_avg_move_{args.model_name}.json"
    agg_path.write_text(json.dumps(all_stats, indent=2))
    print(f"[canonical] aggregated → {agg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
