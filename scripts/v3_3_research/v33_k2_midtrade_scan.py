"""
v33_k2_midtrade_scan.py — Exhaustive mid-trade feature scan.

For each of the 50 K=2 LONG gated events, peek at ALL 32 v3.3 heads at offset events
k in {1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150} after entry. Compute Spearman corr
of each (head, offset) pair against realized net_ticks with permutation p-value.

Goal: find which mid-trade head/lag combinations carry information about exit timing.
Best (negative-corr, low-p) features become candidates for adaptive-exit triggers.

Pure NumPy/pandas/scipy. Internal Lvl3Quant research.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/midtrade_scan")
OUT.mkdir(parents=True, exist_ok=True)
SIG = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/regime_analysis/k2_long_signals_with_regime_v2.csv")
PRED = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz")

np.random.seed(0)


def perm_p(x, y, c, n_perm=300):
    null = np.zeros(n_perm)
    for i in range(n_perm):
        null[i] = spearmanr(x, np.random.permutation(y))[0]
    return float((np.abs(null) >= abs(c)).mean())


def main():
    t0 = time.time()
    sig = pd.read_csv(SIG)
    pred = np.load(PRED, allow_pickle=True)
    head_names = sorted([k for k in pred.files if k.startswith("pred_")])
    n_pred = len(pred['pred_log_ret_1s'])

    gate = (sig.vol_30s_ticks < 1.75).values & sig.filled.values
    sigg = sig[gate].reset_index(drop=True).copy()
    idx = sigg.global_idx.values.astype(int)
    y = sigg.net_ticks.values.astype(np.float32)
    dates = sigg.date.values
    n = len(sigg)
    print(f"Gated set: n={n}, t/fill={y.mean():+.2f}, WR={(y>0).mean():.2%}")

    # Cross-day boundary detection — events that would peek into next day are flagged
    # (cheap proxy: index k must stay within same date in sig.csv, but K=2 sig.csv is filtered.
    # Use the full sig.csv before gating, look up date by global_idx)
    # Build global_idx → date map from the original signals csv
    glob_to_date = dict(zip(sig.global_idx.values.astype(int), sig.date.values))

    offsets = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150]

    rows = []
    for h in head_names:
        arr = pred[h]
        x0 = arr[idx]
        # Entry-time corr
        c0, _ = spearmanr(x0, y)
        p0 = perm_p(x0, y, c0)
        rows.append(dict(head=h, offset=0, kind='entry', corr=float(c0), p=float(p0)))
        for k in offsets:
            oidx = np.clip(idx + k, 0, n_pred - 1)
            # Only valid if next event is still same day (heuristic: check if k-events away in source CSV maps to same date)
            # For simplicity assume valid; warn if cross-day
            xv = arr[oidx]
            # Delta version
            xd = xv - x0
            cv, _ = spearmanr(xv, y)
            cd, _ = spearmanr(xd, y)
            pv = perm_p(xv, y, cv)
            pd_ = perm_p(xd, y, cd)
            rows.append(dict(head=h, offset=k, kind='value', corr=float(cv), p=float(pv)))
            rows.append(dict(head=h, offset=k, kind='delta', corr=float(cd), p=float(pd_)))

    df = pd.DataFrame(rows)
    df['abs_corr'] = df['corr'].abs()
    df = df.sort_values('abs_corr', ascending=False)

    print(f"\n=== TOP 25 (head, offset, kind) by |corr| with net_ticks ===")
    print(df.head(25).to_string(index=False))

    sig_df = df[df.p < 0.10]
    print(f"\n=== Significant @ p<0.10: {len(sig_df)} rows ===")
    print(sig_df.head(40).to_string(index=False))

    # Bucket by kind/offset to see structure
    bucket = df.groupby(['kind', 'offset']).agg(
        n_features=('head', 'count'),
        max_abs_corr=('abs_corr', 'max'),
        n_sig=('p', lambda s: (s < 0.10).sum())
    ).reset_index().sort_values(['kind', 'offset'])
    print(f"\n=== Corr structure by (kind, offset) ===")
    print(bucket.to_string(index=False))

    # Save
    out = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        gate='intra_vol30s_lt_1.75',
        n_events=int(n),
        offsets_tested=offsets,
        n_total_features=int(len(df)),
        n_sig_p10=int((df.p < 0.10).sum()),
        n_sig_p05=int((df.p < 0.05).sum()),
        top_25=df.head(25).to_dict(orient='records'),
        sig_p10=sig_df.to_dict(orient='records'),
        bucket=bucket.to_dict(orient='records'),
        elapsed=round(time.time() - t0, 2),
    )
    json.dump(out, open(OUT / 'midtrade_scan_results.json', 'w'), indent=2, default=str)
    print(f"\nWrote {OUT/'midtrade_scan_results.json'}, elapsed {time.time()-t0:.2f}s")


if __name__ == '__main__':
    main()
