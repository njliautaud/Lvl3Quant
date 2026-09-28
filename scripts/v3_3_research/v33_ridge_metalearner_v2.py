"""
v33_ridge_metalearner_v2.py — Ridge regression meta-learner on K=2 LONG gated events.

Goal: train a tiny model that learns to allow/block K=2 LONG entries within the
intra_vol30s_lt_1.75 gated universe (50 fills, +1.94 t/fill). Features = 32 v3.3
head predictions + regime features (vol/drift/spread @ 30s/60s/300s + ToD + intraday drift).
Label = net_ticks (regression) and net_ticks>0 (classification).

5-fold leave-one-day-out CV over 5 OOT days. Held-out OOS evaluation only.

Pure NumPy, no sklearn — small dataset (~500 events ungated).
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/k2_metalearner_v2")
OUT.mkdir(parents=True, exist_ok=True)
SIG = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/regime_analysis/k2_long_signals_with_regime_v2.csv")
PRED = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz")


def ridge_fit(X, y, lam=1.0):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    A[-1, -1] = 0  # don't regularize bias
    b = X1.T @ y
    return np.linalg.solve(A, b)


def ridge_predict(coef, X):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    return X1 @ coef


def main():
    t0 = time.time()
    sig = pd.read_csv(SIG)
    print(f"Loaded {len(sig)} K=2 events, {int(sig.filled.sum())} filled")

    # Load all v3.3 head predictions to pull row-aligned to global_idx
    pred = np.load(PRED, allow_pickle=True)
    print(f"  predictions keys: {[k for k in pred.files if k.startswith('pred_')][:5]}…")

    head_names = sorted([k for k in pred.files if k.startswith("pred_")])
    n_heads = len(head_names)
    print(f"  {n_heads} prediction heads available")

    # Build feature matrix
    idx = sig.global_idx.values.astype(int)
    head_X = np.zeros((len(sig), n_heads), dtype=np.float32)
    for i, h in enumerate(head_names):
        head_X[:, i] = pred[h][idx]

    regime_cols = ["vol_30s_ticks", "drift_30s_ticks", "mean_spread_30s_t",
                   "vol_60s_ticks", "drift_60s_ticks",
                   "vol_300s_ticks", "drift_300s_ticks",
                   "intraday_drift_ticks", "min_of_day"]
    regime_X = sig[regime_cols].fillna(0).values.astype(np.float32)
    # ToD bucket one-hot
    buckets = sorted(sig.bucket.unique())
    bucket_X = np.zeros((len(sig), len(buckets)), dtype=np.float32)
    for j, b in enumerate(buckets):
        bucket_X[:, j] = (sig.bucket == b).values
    X = np.hstack([head_X, regime_X, bucket_X])
    feat_names = head_names + regime_cols + [f"bucket_{b}" for b in buckets]
    print(f"  feature matrix: {X.shape}, names total {len(feat_names)}")

    # Labels
    sig["net"] = sig.net_ticks.where(sig.filled, np.nan).values
    # Apply gate: intra_vol30s < 1.75 (the winning regime gate)
    gate_mask = (sig.vol_30s_ticks < 1.75).values
    fill_mask = sig.filled.values
    full_mask = gate_mask & fill_mask
    print(f"  gated (vol30s<1.75) & filled: {int(full_mask.sum())} events")

    if int(full_mask.sum()) < 20:
        print("Too few samples after gate")
        return

    Xg = X[full_mask]
    yg = sig.net.values[full_mask]
    dg = sig.date.values[full_mask]

    # Z-score features over the gated set
    Xg_mu = Xg.mean(0); Xg_sd = Xg.std(0) + 1e-6
    Xg_z = (Xg - Xg_mu) / Xg_sd

    days = sorted(np.unique(dg).tolist())
    print(f"  OOT days in gated set: {days}")
    print(f"  per-day gated counts: {dict((str(d), int((dg==d).sum())) for d in days)}")

    # Leave-one-day-out CV
    fold_results = []
    all_pred = np.zeros(len(yg))
    for test_day in days:
        tr = dg != test_day; te = dg == test_day
        Xtr, ytr = Xg_z[tr], yg[tr]
        Xte, yte = Xg_z[te], yg[te]
        for lam in [0.1, 1.0, 10.0, 100.0]:
            coef = ridge_fit(Xtr, ytr, lam=lam)
            pred_te = ridge_predict(coef, Xte)
            mse = float(np.mean((pred_te - yte) ** 2))
            corr = float(np.corrcoef(pred_te, yte)[0, 1]) if len(yte) > 1 and yte.std() > 0 else 0.0
            # "Take" decision: predict > 0
            take = pred_te > 0
            n_take = int(take.sum())
            if n_take > 0:
                pnl_take = float(yte[take].sum())
                wr_take = float((yte[take] > 0).mean())
                tpf_take = float(yte[take].mean())
            else:
                pnl_take = 0.0; wr_take = 0.0; tpf_take = 0.0
            fold_results.append(dict(
                test_day=str(test_day), lam=lam, n_tr=len(ytr), n_te=len(yte),
                mse=mse, corr=corr, n_take=n_take, pnl_take=pnl_take,
                wr_take=wr_take, tpf_take=tpf_take,
                baseline_tpf=float(yte.mean()), baseline_wr=float((yte>0).mean()),
            ))
        # Choose lam=10 as stable default for held-out prediction store
        coef = ridge_fit(Xtr, ytr, lam=10.0)
        all_pred[te] = ridge_predict(coef, Xte)

    # Aggregate by lam
    agg = pd.DataFrame(fold_results).groupby("lam").agg(
        total_te=("n_te", "sum"),
        total_take=("n_take", "sum"),
        total_pnl=("pnl_take", "sum"),
        mean_mse=("mse", "mean"),
        mean_corr=("corr", "mean"),
    )
    print(f"\n=== Ridge meta-learner held-out OOS (lam sweep) ===")
    print(agg.to_string())

    # Per-day under lam=10 OOS prediction
    take_l10 = all_pred > 0
    per_day = {}
    for d in days:
        m = dg == d
        n = int(m.sum())
        nt = int((m & take_l10).sum())
        pnl_taken = float(yg[m & take_l10].sum()) if nt else 0.0
        pnl_all = float(yg[m].sum())
        per_day[str(d)] = dict(n_filled=n, n_take=nt,
                                pnl_taken=pnl_taken, pnl_all=pnl_all,
                                tpf_taken=(pnl_taken/nt) if nt else 0.0)
    print("\n=== Per-day OOS (lam=10) ===")
    print(json.dumps(per_day, indent=2))

    # Overall metrics
    n_total = len(yg)
    n_taken = int(take_l10.sum())
    pnl_taken_total = float(yg[take_l10].sum())
    pnl_all_total = float(yg.sum())
    wr_taken = float((yg[take_l10] > 0).mean()) if n_taken else 0.0
    wr_all = float((yg > 0).mean())
    print(f"\n=== Overall held-out OOS ===")
    print(f"  Without meta (all gated): n={n_total} pnl={pnl_all_total:+.1f} wr={wr_all:.3f} tpf={pnl_all_total/n_total:+.2f}")
    print(f"  With meta (taken):        n={n_taken} pnl={pnl_taken_total:+.1f} wr={wr_taken:.3f} tpf={pnl_taken_total/n_taken if n_taken else 0:+.2f}")
    pnl_diff = pnl_taken_total - pnl_all_total
    print(f"  Δ from blocking trades: {pnl_diff:+.1f} ticks (net commission already in net_ticks)")

    # Save
    out = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        gate="intra_vol30s_lt_1.75",
        n_events_gated=int(full_mask.sum()),
        per_day_gated_counts={str(d): int((dg==d).sum()) for d in days},
        lam_sweep=agg.reset_index().to_dict("records"),
        per_day_oos_lam10=per_day,
        overall_oos_lam10=dict(
            n=n_total, n_taken=n_taken,
            pnl_all=pnl_all_total, pnl_taken=pnl_taken_total,
            wr_all=wr_all, wr_taken=wr_taken,
            tpf_all=pnl_all_total/n_total, tpf_taken=pnl_taken_total/n_taken if n_taken else 0,
            delta_blocked=pnl_diff,
        ),
        elapsed=round(time.time()-t0, 2),
    )
    json.dump(out, open(OUT / "ridge_results.json", "w"), indent=2, default=str)
    np.savez(OUT / "ridge_oos_preds.npz",
             global_idx=sig.global_idx.values[full_mask],
             date=dg, y=yg, pred=all_pred)
    print(f"\nWrote {OUT/'ridge_results.json'}, elapsed {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
