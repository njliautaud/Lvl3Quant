#!/usr/bin/env python3
"""
Triple Fusion v2 — Constrained Linear Blend Meta-Learner.

Why not LGBM:
  - Branches are highly correlated (cnn_mamba_v2 vs book_cnn vs patchtst all
    predict the same direction most of the time).
  - LGBM tree splits on correlated features over-fit fold idiosyncrasies.
  - 2-branch POC confirmed: meta UNDERPERFORMED both branches everywhere.

What this does instead:
  For each horizon h:
    Find weights w_i ≥ 0, sum(w_i)=1, that maximize OOS Spearman IC of
    sum_i w_i * z(p_i)  on training-fold labels.
    Use a small L2 prior toward the historical-best branch (cnn_mamba_v2)
    so we don't degrade in low-signal regimes.

Inputs: per-fold predictions npz files from each branch
        (must be pixel-aligned: same n_windows per fold).

Walk-forward: leave-one-fold-out. For each test fold, fit weights on remaining folds.

Outputs:
  - blended predictions per fold (npz) — drop-in substitute for execution sim
  - weights_history.json — per fold weights, for inspection / sanity
  - concat IC + top-0.5% IC vs each branch
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def neg_ic(w, P_z, y, lam, w_prior):
    """Objective: -Spearman IC + L2 toward prior. P_z: (N, K) zscored, y: (N,)."""
    blended = P_z @ w
    ic = spearmanr(blended, y).correlation
    if ic is None or np.isnan(ic):
        ic = 0.0
    penalty = lam * np.sum((w - w_prior) ** 2)
    return -ic + penalty


def fit_blend(P_z_train, y_train, k, prior_idx=0, lam=0.05):
    """Fit non-negative simplex weights via SLSQP."""
    w_prior = np.zeros(k); w_prior[prior_idx] = 1.0
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * k
    w0 = np.full(k, 1.0 / k)
    res = minimize(neg_ic, w0, args=(P_z_train, y_train, lam, w_prior),
                   method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-6})
    return res.x


def load_fold_streams(branch_dirs, fold_ids, branch_horizon_idx=2):
    """For each fold, returns dict: branch_name -> preds (N,) at the requested horizon."""
    fold_data = {}   # fold_id -> {"branches": {name: pred}, "labels": (N,)}
    for fid in fold_ids:
        per_branch = {}
        labels = None
        for name, bd in branch_dirs.items():
            f = Path(bd) / f"fold_{fid:02d}_oot_predictions.npz"
            if not f.exists():
                continue
            d = np.load(f)
            per_branch[name] = d["predictions"][:, branch_horizon_idx].astype(np.float32)
            this_labels = d["labels"][:, branch_horizon_idx].astype(np.float32)
            if labels is None:
                labels = this_labels
            elif len(this_labels) != len(labels):
                print(f"  [WARN] fold {fid:02d} {name} length mismatch ({len(this_labels)} vs {len(labels)}), truncating")
                m = min(len(this_labels), len(labels))
                labels = labels[:m]
                per_branch = {k: v[:m] for k, v in per_branch.items()}
                this_labels = this_labels[:m]
        if not per_branch:
            print(f"  [WARN] fold {fid:02d} no branches found")
            continue
        # Force common length across branches
        m = min([len(v) for v in per_branch.values()] + [len(labels)])
        per_branch = {k: v[:m] for k, v in per_branch.items()}
        labels = labels[:m]
        fold_data[fid] = {"branches": per_branch, "labels": labels}
    return fold_data


def top_pct_ic(p, y, q=0.995):
    if len(p) < 10:
        return float("nan")
    thr = np.quantile(np.abs(p), q)
    m = np.abs(p) >= thr
    if m.sum() < 5:
        return float("nan")
    ic = spearmanr(p[m], y[m]).correlation
    return 0.0 if (ic is None or np.isnan(ic)) else float(ic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", required=True,
                    help="Comma-separated 'name=dir' pairs. First is the prior branch (most trusted).")
    ap.add_argument("--folds", default="5,6,7,8,9", help="Comma-separated fold ids")
    ap.add_argument("--horizons", default="0,1,2", help="Indices into 3-horizon predictions (1s,5s,10s)")
    ap.add_argument("--lam", type=float, default=0.05, help="L2 prior strength")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    branch_dirs = {}
    prior_name = None
    for spec in args.branches.split(","):
        name, d = spec.split("=", 1)
        if prior_name is None:
            prior_name = name
        branch_dirs[name] = d.strip()
    fold_ids = [int(x) for x in args.folds.split(",")]
    horizon_idxs = [int(x) for x in args.horizons.split(",")]
    horizon_names = ["1s", "5s", "10s"]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Branches: {list(branch_dirs.keys())}  prior={prior_name}")
    print(f"Folds:    {fold_ids}")
    print(f"Horizons: {[horizon_names[i] for i in horizon_idxs]}")
    print()

    summary = {}

    for h_idx in horizon_idxs:
        h_name = horizon_names[h_idx]
        print(f"=== Horizon {h_name} ===")
        fold_data = load_fold_streams(branch_dirs, fold_ids, branch_horizon_idx=h_idx)
        if not fold_data:
            print("  no fold data, skipping")
            continue

        branch_names = sorted(set().union(*[set(fd["branches"].keys()) for fd in fold_data.values()]))
        prior_idx = branch_names.index(prior_name) if prior_name in branch_names else 0
        print(f"  ordered branches: {branch_names} (prior={branch_names[prior_idx]})")

        # Walk-forward leave-one-fold-out
        weights_history = {}
        meta_concat, label_concat = [], []
        per_branch_concat = {n: [] for n in branch_names}
        for test_fid in fold_ids:
            if test_fid not in fold_data:
                continue
            train_fids = [f for f in fold_ids if f != test_fid and f in fold_data]
            if not train_fids:
                print(f"  fold {test_fid:02d}: no train folds available")
                continue

            # Stack training set from train folds
            P_train_list, y_train_list = [], []
            for tf in train_fids:
                fd = fold_data[tf]
                if not all(n in fd["branches"] for n in branch_names):
                    continue
                P = np.stack([fd["branches"][n] for n in branch_names], axis=1)  # (N, K)
                P_train_list.append(P)
                y_train_list.append(fd["labels"])
            if not P_train_list:
                continue
            P_train = np.concatenate(P_train_list)
            y_train = np.concatenate(y_train_list)
            P_train_z = np.column_stack([zscore(P_train[:, i]) for i in range(P_train.shape[1])])

            w = fit_blend(P_train_z, y_train, k=len(branch_names),
                          prior_idx=prior_idx, lam=args.lam)
            weights_history[test_fid] = {n: float(w[i]) for i, n in enumerate(branch_names)}

            # Apply to test fold
            test_fd = fold_data[test_fid]
            if not all(n in test_fd["branches"] for n in branch_names):
                print(f"  fold {test_fid:02d}: missing branch on test, skipping")
                continue
            P_test = np.stack([test_fd["branches"][n] for n in branch_names], axis=1)
            P_test_z = np.column_stack([zscore(P_test[:, i]) for i in range(P_test.shape[1])])
            blended = P_test_z @ w
            meta_concat.append(blended)
            label_concat.append(test_fd["labels"])
            for n in branch_names:
                per_branch_concat[n].append(test_fd["branches"][n])

            # Per-fold IC
            ic_meta = spearmanr(blended, test_fd["labels"]).correlation or 0.0
            branch_ics = {n: spearmanr(test_fd["branches"][n], test_fd["labels"]).correlation or 0.0
                          for n in branch_names}
            wstr = " ".join(f"{n}={w[i]:.2f}" for i, n in enumerate(branch_names))
            bstr = " ".join(f"{n}={branch_ics[n]:+.3f}" for n in branch_names)
            print(f"  fold {test_fid:02d}  meta_ic={ic_meta:+.4f}  [{bstr}]   w={{{wstr}}}")

        # Concat IC
        if meta_concat:
            P = np.concatenate(meta_concat); Y = np.concatenate(label_concat)
            ic_meta = spearmanr(P, Y).correlation or 0.0
            top_meta = top_pct_ic(P, Y)
            branch_ics = {}
            for n in branch_names:
                Pn = np.concatenate(per_branch_concat[n])
                branch_ics[n] = {
                    "concat": float(spearmanr(Pn, Y).correlation or 0.0),
                    "top05pct": top_pct_ic(Pn, Y),
                }
            best_branch = max(branch_ics, key=lambda x: branch_ics[x]["concat"])
            lift = ic_meta - branch_ics[best_branch]["concat"]
            print(f"  CONCAT  meta_ic={ic_meta:+.4f}  top05pct_meta={top_meta:+.4f}  best_branch={best_branch}({branch_ics[best_branch]['concat']:+.4f})  lift={lift:+.4f}")
            summary[h_name] = {
                "meta_concat_ic": float(ic_meta),
                "meta_top05pct_ic": float(top_meta),
                "branches": branch_ics,
                "weights_history": weights_history,
                "lift_over_best_branch": float(lift),
                "n_samples": int(len(P)),
            }
        print()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
