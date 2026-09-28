"""
HC #403 (A) — MFE/MAE Economics Matrix (cross-horizon bug FIXED).

Goal:
  For each (horizon, side, confidence tier), report the gross MFE distribution
  and the fraction of trades whose MFE beats two cost regimes:
    - passive limit (0.376 ticks RT commission only)
    - market / IOC    (1.376 ticks: commission + 1 tick spread cross)

Why this script exists:
  The earlier v33_mfe_mae_economics.csv (hc402_unlock_20260517_000151/)
  produced IDENTICAL numbers across all four horizon heads. Root cause:
  hc402_24h_unlock.py called full_market_replay with
  hold_seconds=5.0 for EVERY horizon. Inside
  full_market_replay._mfe_mae_per_fill the MFE/MAE is built by sweeping all
  horizons up to hold_seconds — so passing hold_seconds=5.0 for every head
  caused the MFE walk to use the same {1s, 5s} subset regardless of the
  signal head being studied. Hence the duplicate numbers.

This script avoids that path entirely. We:
  1) Load the OOT predictions NPZ once.
  2) For each (horizon, side), rank predictions by side-appropriate signed
     value and pick the top tier (Top50 .. Top0.1) percentile sets.
  3) Compute, per selected sample, the SIGNED MFE/MAE in TICKS over
     [t0, t0+horizon] using the per-sample target_log_ret_{h'} arrays (which
     are ALREADY in ticks per the data-reality comment at the top of
     full_market_replay.py). MFE = max over h' <= horizon of
     side_sign*target_log_ret_{h'}; MAE = min over the same set.
  4) For each cell write: n_samples, percentiles of MFE/MAE, fraction beating
     0.376 / 1.376, mean MFE minus each cost, and a 95%-CI lower bound on the
     pct-above-market metric (Wilson interval) and on mean_MFE (normal-approx).

Output dir: output/hc403_a_mfe_<ts>/
Files:
  mfe_economics_matrix.csv
  SUMMARY.json

CPU-only, ~1 min for the full matrix on 673k samples. Read-only on data.
NOT MALWARE. Pure analysis. Writes only to its own output directory.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
NPZ_PATH = PROJ / "output" / "v3_3_extended_oot_20260514" / "extended_oot_predictions.npz"

# Horizons we will TRY. 60s / 5min target arrays were verified empty after mask
# in this OOT NPZ, so they will be flagged "no_data" rather than producing
# noise. (target_pred_mfe_60s_ticks is also empty.)
ALL_HORIZONS = ["1s", "5s", "10s", "30s", "60s"]
HORIZON_ORDER = {"1s": 1, "5s": 2, "10s": 3, "30s": 4, "60s": 5}

# Tiers expressed as fraction of population kept on the chosen side's tail.
TIERS = [
    ("Top50", 0.50),
    ("Top25", 0.25),
    ("Top10", 0.10),
    ("Top5",  0.05),
    ("Top1",  0.01),
    ("Top0.5", 0.005),
    ("Top0.1", 0.001),
]

COST_PASSIVE = 0.376   # commission only
COST_MARKET  = 1.376   # commission + 1 tick spread cross

OUT_DIR = PROJ / "output" / f"hc403_a_mfe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    """Wilson 95% lower bound on a binomial proportion (returns fraction 0..1)."""
    if n <= 0:
        return float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre - half) / denom)


def _mean_ci_low_95(arr: np.ndarray) -> float:
    """Normal-approx 95% lower bound on the mean of `arr`."""
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return float("nan")
    return float(arr.mean() - 1.96 * arr.std(ddof=1) / np.sqrt(n))


def load_npz_horizons(npz_path: Path) -> dict:
    """Load preds + per-horizon target/mask. Returns a dict per horizon plus
    a global ndarray length."""
    print(f"[{_now()}] Loading NPZ: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    n = int(d["n_samples"])
    out = {"_n": n}
    for h in ALL_HORIZONS:
        pk, tk, mk = f"pred_log_ret_{h}", f"target_log_ret_{h}", f"mask_log_ret_{h}"
        if pk not in d.keys() or tk not in d.keys() or mk not in d.keys():
            out[h] = {"available": False, "reason": "missing_keys"}
            continue
        pred = d[pk][:n].astype(np.float64)
        tgt  = d[tk][:n].astype(np.float64)   # ALREADY IN TICKS
        msk  = d[mk][:n].astype(bool) & np.isfinite(pred) & np.isfinite(tgt)
        if int(msk.sum()) == 0:
            out[h] = {"available": False, "reason": "all_masked_out",
                      "n_valid": 0}
            continue
        out[h] = {
            "available": True,
            "pred": pred,
            "tgt":  tgt,
            "mask": msk,
            "n_valid": int(msk.sum()),
        }
    # Build cross-horizon list of which lower horizons are available, ordered.
    out["_available_horizons"] = [h for h in ALL_HORIZONS if out[h].get("available")]
    print(f"[{_now()}] Available horizons (target data present): {out['_available_horizons']}")
    return out


def select_tier(pred: np.ndarray, mask: np.ndarray, side: str,
                tier_frac: float) -> np.ndarray:
    """Return boolean selector for top `tier_frac` of valid `pred` on `side`.
    For side=long the top tier is the most positive predictions.
    For side=short the top tier is the most negative predictions.
    """
    valid = pred[mask]
    if valid.size == 0 or tier_frac <= 0:
        return np.zeros_like(mask)
    if side == "long":
        thr = float(np.quantile(valid, 1.0 - tier_frac))
        return mask & (pred >= thr)
    else:
        thr = float(np.quantile(valid, tier_frac))
        return mask & (pred <= thr)


def mfe_mae_for_cell(npz: dict, horizon: str, side: str, sel_mask: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Compute MFE/MAE in ticks for each selected sample.

    MFE/MAE walk:
      For trade taken with horizon h, look at the SIGNED in-position return
      side_sign * target_log_ret_{h'} for every horizon h' in {1s,5s,...,h}
      that has target data available. MFE = max over those values; MAE = min.

    This is the conservative, data-driven MFE/MAE: we sample the in-position
    P&L at 1s/5s/10s/30s checkpoints inside the trade window and take the best
    and worst. It does NOT capture intra-checkpoint excursions but it is
    horizon-specific by construction (FIXING the prior bug).
    """
    h_order = HORIZON_ORDER[horizon]
    side_sign = 1.0 if side == "long" else -1.0
    idx = np.where(sel_mask)[0]
    n_sel = idx.size
    if n_sel == 0:
        return np.array([]), np.array([])

    # Cross-horizon checkpoint matrix (n_sel, n_checkpoints).
    cols = []
    for h_prime in ALL_HORIZONS:
        if HORIZON_ORDER[h_prime] > h_order:
            break
        info = npz.get(h_prime)
        if not info or not info.get("available"):
            continue
        signed = side_sign * info["tgt"][idx]
        # Where the per-horizon target mask is invalid for this sample,
        # treat as NaN so it doesn't pollute max/min.
        valid_here = info["mask"][idx]
        signed = np.where(valid_here, signed, np.nan)
        cols.append(signed)
    if not cols:
        return np.full(n_sel, np.nan), np.full(n_sel, np.nan)
    M = np.column_stack(cols)
    # Use nanmax / nanmin, but if a whole row is NaN, return NaN.
    with np.errstate(invalid="ignore"):
        all_nan = np.all(np.isnan(M), axis=1)
        mfe = np.where(all_nan, np.nan, np.nanmax(M, axis=1))
        mae = np.where(all_nan, np.nan, np.nanmin(M, axis=1))
    return mfe, mae


def compute_cell_metrics(mfe: np.ndarray, mae: np.ndarray) -> dict:
    """Reduce raw MFE/MAE arrays into per-cell statistics."""
    finite = np.isfinite(mfe)
    n = int(finite.sum())
    if n == 0:
        return {
            "n_samples": 0,
            "median_MFE_ticks": float("nan"),
            "mean_MFE_ticks": float("nan"),
            "p90_MFE_ticks": float("nan"),
            "median_MAE_ticks": float("nan"),
            "pct_above_0.376_passive": float("nan"),
            "pct_above_1.376_market": float("nan"),
            "mean_MFE_minus_passive_cost": float("nan"),
            "mean_MFE_minus_market_cost": float("nan"),
            "ci_low_95_mean_MFE": float("nan"),
            "ci_low_95_pct_above_market": float("nan"),
        }
    mf = mfe[finite]
    ma = mae[np.isfinite(mae)]
    k_passive = int((mf > COST_PASSIVE).sum())
    k_market  = int((mf > COST_MARKET).sum())
    return {
        "n_samples": n,
        "median_MFE_ticks": float(np.median(mf)),
        "mean_MFE_ticks": float(mf.mean()),
        "p90_MFE_ticks": float(np.percentile(mf, 90)),
        "median_MAE_ticks": float(np.median(ma)) if ma.size else float("nan"),
        "pct_above_0.376_passive": float(100.0 * k_passive / n),
        "pct_above_1.376_market":  float(100.0 * k_market  / n),
        "mean_MFE_minus_passive_cost": float(mf.mean() - COST_PASSIVE),
        "mean_MFE_minus_market_cost":  float(mf.mean() - COST_MARKET),
        "ci_low_95_mean_MFE": _mean_ci_low_95(mf),
        "ci_low_95_pct_above_market": 100.0 * _wilson_lower_bound(k_market, n),
    }


def build_matrix(npz: dict) -> pd.DataFrame:
    rows = []
    for horizon in ALL_HORIZONS:
        info = npz.get(horizon)
        if not info or not info.get("available"):
            reason = info.get("reason", "unknown") if info else "missing"
            for side in ("long", "short"):
                for tier_name, _ in TIERS:
                    rows.append({
                        "horizon": horizon, "side": side, "tier": tier_name,
                        "n_samples": 0,
                        "median_MFE_ticks": float("nan"),
                        "mean_MFE_ticks": float("nan"),
                        "p90_MFE_ticks": float("nan"),
                        "median_MAE_ticks": float("nan"),
                        "pct_above_0.376_passive": float("nan"),
                        "pct_above_1.376_market": float("nan"),
                        "mean_MFE_minus_passive_cost": float("nan"),
                        "mean_MFE_minus_market_cost": float("nan"),
                        "ci_low_95_mean_MFE": float("nan"),
                        "ci_low_95_pct_above_market": float("nan"),
                        "note": f"horizon_unavailable:{reason}",
                    })
            continue
        pred = info["pred"]
        mask = info["mask"]
        for side in ("long", "short"):
            for tier_name, tier_frac in TIERS:
                sel = select_tier(pred, mask, side, tier_frac)
                mfe, mae = mfe_mae_for_cell(npz, horizon, side, sel)
                stats = compute_cell_metrics(mfe, mae)
                rows.append({
                    "horizon": horizon, "side": side, "tier": tier_name,
                    **stats,
                    "note": "ok",
                })
                print(f"[{_now()}] {horizon:>4} {side:>5} {tier_name:>7}  "
                      f"n={stats['n_samples']:>6}  "
                      f"medMFE={stats['median_MFE_ticks']:+6.2f}  "
                      f"meanMFE={stats['mean_MFE_ticks']:+6.2f}  "
                      f">0.376={stats['pct_above_0.376_passive']:5.1f}%  "
                      f">1.376={stats['pct_above_1.376_market']:5.1f}%")
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame, npz: dict) -> dict:
    market = df[(df["median_MFE_ticks"] > COST_MARKET) & (df["n_samples"] > 0)]
    passive = df[(df["median_MFE_ticks"] > COST_PASSIVE) & (df["n_samples"] > 0)]
    # Rough trade-frequency framing: tier fraction * total valid samples / 15 OOT days.
    tier_lookup = dict(TIERS)
    def fills_per_day(row) -> float:
        tf = tier_lookup.get(row["tier"], np.nan)
        h_info = npz.get(row["horizon"])
        n_valid = h_info["n_valid"] if h_info and h_info.get("available") else 0
        return float(n_valid * tf / 15.0)
    def cell_dict(row) -> dict:
        return {
            "horizon": row["horizon"], "side": row["side"], "tier": row["tier"],
            "n_samples": int(row["n_samples"]),
            "median_MFE_ticks": float(row["median_MFE_ticks"]),
            "mean_MFE_ticks": float(row["mean_MFE_ticks"]),
            "pct_above_passive": float(row["pct_above_0.376_passive"]),
            "pct_above_market":  float(row["pct_above_1.376_market"]),
            "ci_low_95_pct_above_market": float(row["ci_low_95_pct_above_market"]),
            "approx_fills_per_day": fills_per_day(row),
        }
    summary = {
        "input_npz": str(NPZ_PATH),
        "n_oot_days": 15,
        "cost_passive_ticks": COST_PASSIVE,
        "cost_market_ticks":  COST_MARKET,
        "available_horizons": npz["_available_horizons"],
        "unavailable_horizons": [
            h for h in ALL_HORIZONS if h not in npz["_available_horizons"]
        ],
        "n_cells_total": int(len(df)),
        "n_cells_market_tradeable_median_MFE": int(len(market)),
        "n_cells_passive_tradeable_median_MFE": int(len(passive)),
        "market_tradeable_cells": [cell_dict(r) for _, r in market.iterrows()],
        "passive_tradeable_cells": [cell_dict(r) for _, r in passive.iterrows()],
    }
    return summary


def main() -> None:
    t0 = time.time()
    print(f"[{_now()}] HC #403 (A) — MFE/MAE Economics Matrix")
    print(f"[{_now()}] OUT_DIR = {OUT_DIR}")
    npz = load_npz_horizons(NPZ_PATH)
    df = build_matrix(npz)
    csv_path = OUT_DIR / "mfe_economics_matrix.csv"
    df.to_csv(csv_path, index=False)
    print(f"[{_now()}] Wrote {csv_path} ({len(df)} rows)")

    summary = build_summary(df, npz)
    sum_path = OUT_DIR / "SUMMARY.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"[{_now()}] Wrote {sum_path}")

    print()
    print("=" * 72)
    print(f"STDOUT SUMMARY")
    print("=" * 72)
    print(f"Available horizons (target data present): {summary['available_horizons']}")
    print(f"Cells with median_MFE > {COST_PASSIVE} (passive-tradeable): "
          f"{summary['n_cells_passive_tradeable_median_MFE']} / {summary['n_cells_total']}")
    print(f"Cells with median_MFE > {COST_MARKET} (market-tradeable):  "
          f"{summary['n_cells_market_tradeable_median_MFE']} / {summary['n_cells_total']}")
    if summary["market_tradeable_cells"]:
        print()
        print("MARKET-TRADEABLE CELLS (median MFE > 1.376):")
        for c in summary["market_tradeable_cells"]:
            print(f"  {c['horizon']:>4} {c['side']:>5} {c['tier']:>7}  "
                  f"medMFE={c['median_MFE_ticks']:+6.2f}  "
                  f"meanMFE={c['mean_MFE_ticks']:+6.2f}  "
                  f">market={c['pct_above_market']:5.1f}% (CI_low {c['ci_low_95_pct_above_market']:5.1f}%)  "
                  f"~{c['approx_fills_per_day']:.0f} fills/day")
    if summary["passive_tradeable_cells"]:
        print()
        print(f"PASSIVE-TRADEABLE CELLS (median MFE > 0.376) [showing top 20]:")
        for c in summary["passive_tradeable_cells"][:20]:
            print(f"  {c['horizon']:>4} {c['side']:>5} {c['tier']:>7}  "
                  f"medMFE={c['median_MFE_ticks']:+6.2f}  "
                  f">passive={c['pct_above_passive']:5.1f}%  "
                  f"~{c['approx_fills_per_day']:.0f} fills/day")
    print()
    print(f"[{_now()}] Done in {(time.time()-t0):.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[{_now()}] FATAL: {e}", file=sys.stderr)
        raise
