"""
Confidence-band analyzer for CNN-Mamba OOT predictions (HC #306).

Ingests `fold_NN_oot_predictions.npz` produced by train_cnn_mamba_v3{,_1,_2}.py
(or compatible inference dumps), and emits the canonical band table:

    Band       | DA_all  DA_long DA_short | IC | MagCorr | avg_pred | avg_real | Sharpe
    Top 0.1%   |
    Top 0.5%   |
    Top 1%     |
    Top 5%     |
    Top 10%    |
    Top 20%    |
    Bottom 20% |
    Bottom 10% |
    Bottom 5%  |
    Bottom 1%  |
    Bottom 0.5%|
    Bottom 0.1%|

× per regression horizon (default 1s/5s/10s/30s).

Per HC #306: aggregate IC is SECONDARY; per-band DA + long/short asymmetry are PRIMARY.

Usage:
    python compute_confidence_bands.py \\
        --pred-npz /home/nick/Lvl3Quant/output/cnn_mamba_v3_2_long_context/fold_00_oot_predictions.npz \\
        --horizons log_ret_1s log_ret_5s log_ret_10s log_ret_30s \\
        --out-csv /tmp/v3_2_fold0_bands.csv \\
        --label "v3.2 Ep5 fold0"

The script does NOT mutate or augment any training code. Read-only analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# Canonical bands per HC #306. Stored as (label, top_fraction or None, bottom_fraction or None)
# "Top X%" means |pred| in top X% of magnitudes, then sign-split into long/short.
# "Bottom X%" means lowest-confidence X% (small |pred|) — sanity-check baseline.
BANDS: list[tuple[str, str, float]] = [
    ("Top 0.1%",   "top", 0.001),
    ("Top 0.5%",   "top", 0.005),
    ("Top 1%",     "top", 0.01),
    ("Top 5%",     "top", 0.05),
    ("Top 10%",    "top", 0.10),
    ("Top 20%",    "top", 0.20),
    ("All",        "all", 1.00),
    ("Bottom 20%", "bot", 0.20),
    ("Bottom 10%", "bot", 0.10),
    ("Bottom 5%",  "bot", 0.05),
    ("Bottom 1%",  "bot", 0.01),
]


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson corr robust to NaN/inf/zero-variance."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_sharpe(realized: np.ndarray, pred_sign: np.ndarray) -> float:
    """Trade-Sharpe assuming we trade per pred-sign with unit size.

    Returns float NaN if <10 samples or zero std. Annualized assuming the
    realized log-returns are at the head's native horizon (caller annotates).
    """
    m = np.isfinite(realized) & np.isfinite(pred_sign) & (pred_sign != 0)
    if m.sum() < 10:
        return float("nan")
    pnl = realized[m] * pred_sign[m]
    if pnl.std() < 1e-12:
        return float("nan")
    # Per-trade Sharpe (NOT annualized) — caller can scale by sqrt(N_trades_per_year)
    return float(pnl.mean() / pnl.std())


def analyze_horizon(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    """Compute the HC #306 band table for one horizon.

    Returns: dict { band_label: { da_all, da_long, da_short, n, n_long, n_short,
                                  ic, mag_corr, avg_pred, avg_realized, sharpe } }
    """
    valid = np.isfinite(pred) & np.isfinite(target) & (mask > 0)
    p = pred[valid]
    t = target[valid]
    n_total = p.size
    if n_total < 100:
        return {"_error": f"insufficient valid samples: {n_total}"}

    abs_p = np.abs(p)
    # Pre-compute magnitude thresholds for top-X bands
    rows: dict[str, dict] = {}
    for label, kind, frac in BANDS:
        if kind == "all":
            sel = np.ones_like(p, dtype=bool)
        elif kind == "top":
            k = max(1, int(np.ceil(frac * n_total)))
            thresh = np.partition(abs_p, n_total - k)[n_total - k]
            sel = abs_p >= thresh
        elif kind == "bot":
            k = max(1, int(np.ceil(frac * n_total)))
            thresh = np.partition(abs_p, k - 1)[k - 1]
            sel = abs_p <= thresh
        else:
            raise ValueError(kind)

        pp = p[sel]
        tt = t[sel]
        n = pp.size
        # Directional accuracy: sign(pred) == sign(target), among non-zero-pred samples
        nz = pp != 0
        n_eff = int(nz.sum())
        if n_eff > 0:
            da_all = float((np.sign(pp[nz]) == np.sign(tt[nz])).mean())
        else:
            da_all = float("nan")

        long_mask = pp > 0
        short_mask = pp < 0
        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())
        # Long DA: of long preds, fraction where target > 0 (i.e., we'd profit)
        da_long = (
            float((tt[long_mask] > 0).mean()) if n_long > 0 else float("nan")
        )
        da_short = (
            float((tt[short_mask] < 0).mean()) if n_short > 0 else float("nan")
        )

        ic = safe_corr(pp, tt)
        mag_corr = safe_corr(np.abs(pp), np.abs(tt))
        avg_pred = float(np.mean(pp)) if n > 0 else float("nan")
        avg_real = float(np.mean(tt)) if n > 0 else float("nan")
        pred_sign = np.sign(pp)
        sharpe = safe_sharpe(tt, pred_sign)

        rows[label] = {
            "n": n,
            "n_long": n_long,
            "n_short": n_short,
            "da_all": da_all,
            "da_long": da_long,
            "da_short": da_short,
            "ic": ic,
            "mag_corr": mag_corr,
            "avg_pred": avg_pred,
            "avg_realized": avg_real,
            "sharpe_per_trade": sharpe,
        }
    return rows


def format_table(rows: dict, horizon: str, label: str) -> str:
    """Human-readable table for one horizon."""
    out = []
    out.append(f"\n=== {label} | horizon={horizon} ===")
    header = (
        f"{'Band':<12} | {'N':>7} | {'DA_all':>7} {'DA_long':>7} {'DA_short':>8} | "
        f"{'IC':>7} {'MagCorr':>8} | {'avgPred':>9} {'avgReal':>9} | {'Sharpe':>7}"
    )
    out.append(header)
    out.append("-" * len(header))
    for label_row in [b[0] for b in BANDS]:
        r = rows.get(label_row)
        if r is None:
            continue
        out.append(
            f"{label_row:<12} | {r['n']:>7d} | "
            f"{r['da_all']:>7.4f} {r['da_long']:>7.4f} {r['da_short']:>8.4f} | "
            f"{r['ic']:>7.4f} {r['mag_corr']:>8.4f} | "
            f"{r['avg_pred']:>9.2e} {r['avg_realized']:>9.2e} | "
            f"{r['sharpe_per_trade']:>7.4f}"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-npz", required=True, type=Path)
    ap.add_argument(
        "--horizons",
        nargs="+",
        default=["log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s"],
        help="Head names to analyze (must match keys in the .npz: pred_<H>/target_<H>/mask_<H>)",
    )
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--label", type=str, default="model")
    args = ap.parse_args()

    if not args.pred_npz.exists():
        print(f"ERROR: pred-npz not found: {args.pred_npz}", file=sys.stderr)
        sys.exit(2)

    npz = np.load(args.pred_npz, allow_pickle=True)
    available = set(npz.files)

    all_results: dict[str, dict] = {}
    for h in args.horizons:
        pk, tk, mk = f"pred_{h}", f"target_{h}", f"mask_{h}"
        if pk not in available or tk not in available:
            print(f"SKIP {h}: missing keys ({pk}/{tk}) in npz", file=sys.stderr)
            continue
        pred = np.asarray(npz[pk]).reshape(-1)
        target = np.asarray(npz[tk]).reshape(-1)
        mask = (
            np.asarray(npz[mk]).reshape(-1)
            if mk in available
            else np.ones_like(pred, dtype=bool)
        )
        rows = analyze_horizon(pred, target, mask)
        all_results[h] = rows
        print(format_table(rows, h, args.label))

    # Aggregate IC line (HC #306: secondary, but always reported)
    print("\n--- Aggregate IC (secondary per HC #306) ---")
    for h, rows in all_results.items():
        all_row = rows.get("All", {})
        print(
            f"  {h}: IC={all_row.get('ic', float('nan')):.4f} "
            f"DA={all_row.get('da_all', float('nan')):.4f} "
            f"N={all_row.get('n', 0)}"
        )

    if args.out_csv:
        import csv
        with args.out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["horizon", "band", "n", "n_long", "n_short", "da_all", "da_long",
                 "da_short", "ic", "mag_corr", "avg_pred", "avg_realized",
                 "sharpe_per_trade"]
            )
            for h, rows in all_results.items():
                for band_label in [b[0] for b in BANDS]:
                    r = rows.get(band_label)
                    if r is None:
                        continue
                    w.writerow(
                        [h, band_label, r["n"], r["n_long"], r["n_short"],
                         r["da_all"], r["da_long"], r["da_short"], r["ic"],
                         r["mag_corr"], r["avg_pred"], r["avg_realized"],
                         r["sharpe_per_trade"]]
                    )
        print(f"\n[csv] {args.out_csv}")

    if args.out_json:
        args.out_json.write_text(json.dumps(all_results, indent=2, default=float))
        print(f"[json] {args.out_json}")


if __name__ == "__main__":
    main()
