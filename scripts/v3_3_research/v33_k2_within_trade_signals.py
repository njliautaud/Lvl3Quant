"""
v33_k2_within_trade_signals.py — Within-trade signal analysis for adaptive exits.

USER REQUEST (2026-05-15 16:36 ET): "a single snapshot prediction about direction and
holding for a set amount of time without any other confluence mid trade hold is bad.
Some way of using the alpha MODEL itself..."

Goal: For the 50 K=2 LONG events that pass vol30s<1.75 gate, examine whether
WITHIN-TRADE model behavior (cross-head consensus, prediction drift right after entry,
adjacent-event prediction shifts) predicts the realized 30s net_ticks better than
entry-time-only features.

Features tested per event:
  A. Entry-time only (baseline; ridge metalearner v2 already showed corr=0.09-0.17):
     - 32 v3.3 head preds @ global_idx
  B. Cross-head AGREEMENT at entry (directional consensus):
     - Sign-agreement count across short-horizon return heads
     - Sign-agreement weighted by predicted magnitude
     - Quantile head spread (q90-q10) = uncertainty proxy
  C. Adjacent-event drift (proxy for "what happens 1-5 events after entry"):
     - Δ pred_log_ret_1s from global_idx → global_idx+k for k in {1,3,5,10}
     - Same for pred_log_ret_5s, pred_log_ret_10s
     - Direction-flip indicator: does pred sign change in first 5 events?
  D. Cross-horizon AGREEMENT (1s/5s/10s/30s/60s all positive?):
     - Multi-horizon-positive count
     - Multi-horizon mean magnitude

Method: 5-fold leave-one-day-out CV, pure-NumPy ridge + simple gating thresholds.
Pure NumPy. No sklearn. Self-contained. Internal Lvl3Quant research script.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/within_trade_signals")
OUT.mkdir(parents=True, exist_ok=True)
SIG = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/regime_analysis/k2_long_signals_with_regime_v2.csv")
PRED = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz")


def ridge_fit(X, y, lam=1.0):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    A[-1, -1] = 0
    return np.linalg.solve(A, X1.T @ y)


def ridge_predict(coef, X):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    return X1 @ coef


def loo_day_cv(X, y, dates, lam=10.0):
    """Leave-one-day-out CV. Returns OOS preds aligned with input."""
    oos = np.zeros(len(y))
    for d in np.unique(dates):
        tr = dates != d
        te = dates == d
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        coef = ridge_fit(Xtr, y[tr], lam=lam)
        oos[te] = ridge_predict(coef, Xte)
    return oos


def score_take(oos_pred, y, thr=0.0):
    take = oos_pred > thr
    n = int(take.sum())
    if n == 0:
        return dict(n_take=0, pnl=0.0, wr=0.0, tpf=0.0, sharpe=0.0,
                    n_total=len(y), pnl_baseline=float(y.sum()),
                    tpf_baseline=float(y.mean()))
    pnl = float(y[take].sum())
    return dict(
        n_take=n,
        pnl=pnl,
        wr=float((y[take] > 0).mean()),
        tpf=float(y[take].mean()),
        sharpe=float(y[take].mean() / (y[take].std() + 1e-9)),
        n_total=len(y),
        pnl_baseline=float(y.sum()),
        tpf_baseline=float(y.mean()),
        threshold=thr,
    )


def main():
    t0 = time.time()
    sig = pd.read_csv(SIG)
    pred = np.load(PRED, allow_pickle=True)

    head_names = sorted([k for k in pred.files if k.startswith("pred_")])
    n_pred = len(pred['pred_log_ret_1s'])
    print(f"K=2 signals: {len(sig)} | Predictions: {n_pred} across {len(head_names)} heads")

    # Gate to the 50 winning events
    gate = (sig.vol_30s_ticks < 1.75).values & sig.filled.values
    sigg = sig[gate].reset_index(drop=True).copy()
    idx = sigg.global_idx.values.astype(int)
    y = sigg.net_ticks.values.astype(np.float32)
    dates = sigg.date.values
    print(f"Gated set: {len(sigg)} events | t/fill = {y.mean():+.2f} | WR = {(y>0).mean():.2%}")
    print(f"Per-day distribution: {dict(sigg.date.value_counts().sort_index())}")

    # --- Block A: 32 head preds at entry ---
    XA = np.zeros((len(sigg), len(head_names)), dtype=np.float32)
    for i, h in enumerate(head_names):
        arr = pred[h]
        XA[:, i] = arr[idx]
    print(f"Block A (entry-time 32 heads): shape {XA.shape}")

    # --- Block B: cross-head consensus at entry ---
    # Short-horizon return heads
    ret_heads = ['pred_log_ret_1s', 'pred_log_ret_5s', 'pred_log_ret_10s', 'pred_log_ret_30s', 'pred_log_ret_60s']
    ret_idx = [head_names.index(h) for h in ret_heads if h in head_names]
    ret_X = XA[:, ret_idx]  # (n, 5)
    n_pos = (ret_X > 0).sum(axis=1).astype(np.float32)  # 0..5
    n_neg = (ret_X < 0).sum(axis=1).astype(np.float32)
    mag_signed = ret_X.sum(axis=1)  # sum of returns; positive = bullish
    mag_abs_mean = np.abs(ret_X).mean(axis=1)
    consensus_strength = n_pos - n_neg  # -5..+5

    # Quantile uncertainty proxy (q90-q10 of log_ret_10s)
    q10_h = 'pred_log_ret_10s_q10'; q90_h = 'pred_log_ret_10s_q90'
    if q10_h in pred.files and q90_h in pred.files:
        unc_10s = (pred[q90_h][idx] - pred[q10_h][idx])
    else:
        unc_10s = np.zeros(len(idx))
    q10_30 = 'pred_log_ret_30s_q10'; q90_30 = 'pred_log_ret_30s_q90'
    if q10_30 in pred.files and q90_30 in pred.files:
        unc_30s = (pred[q90_30][idx] - pred[q10_30][idx])
    else:
        unc_30s = np.zeros(len(idx))

    XB = np.column_stack([n_pos, n_neg, mag_signed, mag_abs_mean, consensus_strength, unc_10s, unc_30s])
    XB_names = ['n_pos_5h', 'n_neg_5h', 'mag_signed', 'mag_abs_mean', 'consensus_strength', 'unc_10s', 'unc_30s']
    print(f"Block B (consensus): shape {XB.shape}")

    # --- Block C: adjacent-event drift (proxy for mid-trade) ---
    # For each event at idx, peek at idx+k for k in {1,3,5,10}
    # Important: don't cross day boundaries — peek with bounds check
    XC_list = []
    XC_names = []
    for k in [1, 3, 5, 10]:
        for h in ['pred_log_ret_1s', 'pred_log_ret_5s', 'pred_log_ret_10s']:
            arr = pred[h]
            offset_idx = np.clip(idx + k, 0, len(arr) - 1)
            entry_val = arr[idx]
            offset_val = arr[offset_idx]
            delta = offset_val - entry_val
            XC_list.append(delta)
            XC_names.append(f'd{h}_k{k}')
            # also flip indicator
            flip = ((entry_val > 0) & (offset_val < 0)).astype(np.float32) - \
                   ((entry_val < 0) & (offset_val > 0)).astype(np.float32)
            XC_list.append(flip)
            XC_names.append(f'flip_{h}_k{k}')
    XC = np.column_stack(XC_list).astype(np.float32)
    print(f"Block C (adjacent-event drift): shape {XC.shape}")

    # --- Block D: cross-horizon agreement ---
    multi_pos = (ret_X > 0).all(axis=1).astype(np.float32)
    multi_neg = (ret_X < 0).all(axis=1).astype(np.float32)
    horiz_corr_ranks = np.argsort(ret_X, axis=1).mean(axis=1).astype(np.float32)
    XD = np.column_stack([multi_pos, multi_neg, horiz_corr_ranks, ret_X.mean(axis=1), ret_X.std(axis=1)])
    XD_names = ['multi_pos', 'multi_neg', 'horiz_rank_mean', 'ret_mean', 'ret_std']
    print(f"Block D (cross-horizon): shape {XD.shape}")

    # --- Block E: regime (re-use from CSV) ---
    regime_cols = ['vol_30s_ticks', 'drift_30s_ticks', 'mean_spread_30s_t',
                   'vol_60s_ticks', 'drift_60s_ticks',
                   'vol_300s_ticks', 'drift_300s_ticks',
                   'intraday_drift_ticks', 'min_of_day']
    XE = sigg[regime_cols].fillna(0).values.astype(np.float32)
    print(f"Block E (regime): shape {XE.shape}")

    # Evaluate each block separately and combinations
    results = {}
    for name, X, feat_names in [
        ('A_entry32', XA, head_names),
        ('B_consensus', XB, XB_names),
        ('C_adj_drift', XC, XC_names),
        ('D_horizon_agree', XD, XD_names),
        ('E_regime', XE, regime_cols),
        ('B+C+D', np.hstack([XB, XC, XD]), XB_names + XC_names + XD_names),
        ('B+C+D+E', np.hstack([XB, XC, XD, XE]), XB_names + XC_names + XD_names + regime_cols),
        ('ALL', np.hstack([XA, XB, XC, XD, XE]), head_names + XB_names + XC_names + XD_names + regime_cols),
    ]:
        for lam in [0.1, 1.0, 10.0, 100.0]:
            oos = loo_day_cv(X, y, dates, lam=lam)
            corr = float(np.corrcoef(oos, y)[0, 1]) if y.std() > 0 else 0.0
            scored = score_take(oos, y, thr=0.0)
            results[f'{name}|lam={lam}'] = dict(
                corr=corr, n_feat=X.shape[1], lam=lam,
                **scored,
            )

    # Sort by tpf, then by Sharpe
    rows = []
    for k, v in results.items():
        rows.append(dict(model=k, **v))
    rdf = pd.DataFrame(rows).sort_values(['tpf', 'sharpe'], ascending=False)
    print(f"\n=== Within-trade signal evaluation (LOO-day, take if pred>0) ===")
    print(rdf[['model', 'corr', 'n_feat', 'n_take', 'wr', 'tpf', 'sharpe', 'pnl']].to_string(index=False))

    # Threshold sweep on the best model
    best_row = rdf.iloc[0]
    best_model = best_row['model']
    name, lam_str = best_model.split('|')
    lam = float(lam_str.split('=')[1])
    if name == 'A_entry32':
        Xbest = XA
    elif name == 'B_consensus':
        Xbest = XB
    elif name == 'C_adj_drift':
        Xbest = XC
    elif name == 'D_horizon_agree':
        Xbest = XD
    elif name == 'E_regime':
        Xbest = XE
    elif name == 'B+C+D':
        Xbest = np.hstack([XB, XC, XD])
    elif name == 'B+C+D+E':
        Xbest = np.hstack([XB, XC, XD, XE])
    else:
        Xbest = np.hstack([XA, XB, XC, XD, XE])
    best_oos = loo_day_cv(Xbest, y, dates, lam=lam)
    thr_sweep = []
    for thr in np.percentile(best_oos, [10, 25, 50, 75, 90]).tolist() + [0.0]:
        s = score_take(best_oos, y, thr=thr)
        s['threshold'] = float(thr)
        thr_sweep.append(s)
    print(f"\n=== Threshold sweep on best model {best_model} ===")
    print(pd.DataFrame(thr_sweep).to_string(index=False))

    # Save
    out = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        gate='intra_vol30s_lt_1.75',
        n_events=int(len(sigg)),
        baseline_tpf=float(y.mean()),
        baseline_wr=float((y > 0).mean()),
        baseline_pnl=float(y.sum()),
        block_shapes=dict(A=XA.shape, B=XB.shape, C=XC.shape, D=XD.shape, E=XE.shape),
        results=results,
        best_model=best_model,
        thr_sweep=thr_sweep,
        elapsed=round(time.time() - t0, 2),
    )
    json.dump(out, open(OUT / 'within_trade_signal_results.json', 'w'), indent=2, default=str)
    np.savez(OUT / 'best_oos_scores.npz',
             global_idx=sigg.global_idx.values, date=dates,
             y=y, oos_score=best_oos, model_name=best_model)
    print(f"\nWrote {OUT/'within_trade_signal_results.json'}, elapsed {time.time()-t0:.2f}s")


if __name__ == '__main__':
    main()
