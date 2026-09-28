"""
v33_k2_midtrade_gate.py — Combine TOP-significant mid-trade features into a gating model.

Uses the 5 best (head, offset, kind) features identified in midtrade_scan:
  1. pred_pred_time_to_mfe_secs @ value k=150 (corr -0.52)
  2. pred_pred_mae_60s_ticks @ value k=50 (corr -0.49)
  3. pred_fifo_tp8sl5_net @ value k=30 (corr +0.48)
  4. pred_pred_realized_vol_30s_ticks @ delta k=5 (corr -0.45)
  5. pred_fifo_tp8sl5_hit_tp @ delta k=10 (corr -0.45)

CAVEAT: features at offset k>30 (e.g. k=150, k=50) are FUTURE-LOOKING relative to trade
entry but EARLIER than the trade exit ONLY for offsets ≤ 120 (30s @ ~250ms stride).
The k=150 features peek BEYOND the nominal 30s hold — these are useful for understanding
the signal structure but cannot literally be used at trade time. So we test BOTH:
  - Tier A: use only features with offset ≤ 30 (≈7.5s, well within hold) — TRADEABLE
  - Tier B: full set (k≤150) — INFORMATIVE only

Method: LOO-day-CV ridge classifier on net_ticks>0. Gate: take if P(win) > 0.5.

Internal Lvl3Quant research, pure NumPy.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/midtrade_gate")
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


def loo_day(X, y, dates, lam=1.0):
    """Leave-one-day-out OOS prediction. Z-scoring fitted on train, applied to test."""
    oos = np.zeros(len(y))
    for d in np.unique(dates):
        tr = dates != d
        te = dates == d
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        coef = ridge_fit(Xtr, y[tr].astype(np.float32), lam=lam)
        oos[te] = ridge_predict(coef, Xte)
    return oos


def per_day_breakdown(take, y, dates):
    out = {}
    for d in np.unique(dates):
        m = (dates == d) & take
        out[str(d)] = dict(n=int(m.sum()), pnl=float(y[m].sum()) if m.any() else 0.0,
                            wr=float((y[m] > 0).mean()) if m.any() else 0.0,
                            tpf=float(y[m].mean()) if m.any() else 0.0)
    return out


def main():
    t0 = time.time()
    sig = pd.read_csv(SIG)
    pred = np.load(PRED, allow_pickle=True)
    n_pred = len(pred['pred_log_ret_1s'])

    gate = (sig.vol_30s_ticks < 1.75).values & sig.filled.values
    sigg = sig[gate].reset_index(drop=True).copy()
    idx = sigg.global_idx.values.astype(int)
    y = sigg.net_ticks.values.astype(np.float32)
    dates = sigg.date.values
    print(f"Gated: n={len(sigg)}, t/fill={y.mean():+.2f}, pnl={y.sum():+.1f}")

    # TIER A features (offset ≤ 30, tradeable in real-time)
    def at_offset(head, k):
        arr = pred[head]
        oidx = np.clip(idx + k, 0, n_pred - 1)
        return arr[oidx]
    def delta(head, k):
        arr = pred[head]
        oidx = np.clip(idx + k, 0, n_pred - 1)
        return arr[oidx] - arr[idx]

    A_feats = {
        'tp8_net_k30': at_offset('pred_fifo_tp8sl5_net', 30),
        'tp4_net_k30': at_offset('pred_fifo_tp4sl3_net', 30),
        'vol30s_d_k5': delta('pred_pred_realized_vol_30s_ticks', 5),
        'tp8_hit_d_k10': delta('pred_fifo_tp8sl5_hit_tp', 10),
        'p_rev30_d_k30': delta('pred_p_reversal_30s', 30),
        'mae30_d_k10': delta('pred_pred_mae_30s_ticks', 10),
        'mae30_d_k5': delta('pred_pred_mae_30s_ticks', 5),
    }
    XA_names = list(A_feats.keys())
    XA = np.column_stack([A_feats[k] for k in XA_names]).astype(np.float32)

    # TIER B features (full set, informative)
    B_extra = {
        'mae60_k50': at_offset('pred_pred_mae_60s_ticks', 50),
        'time_mfe_k150': at_offset('pred_pred_time_to_mfe_secs', 150),
        'tp4_net_k150': at_offset('pred_fifo_tp4sl3_net', 150),
        'tp8_net_k50': at_offset('pred_fifo_tp8sl5_net', 50),
    }
    XB = np.column_stack([XA, *[B_extra[k] for k in B_extra]]).astype(np.float32)
    XB_names = XA_names + list(B_extra.keys())

    # Regime gate base for comparison
    print(f"\nTier A features ({len(XA_names)}): {XA_names}")
    print(f"Tier B features ({len(XB_names)}): {XB_names}")

    # Run LOO-day for both tiers, classification target (win/loss) AND regression (net_ticks)
    results = []
    for tier, X, names in [('A_tradeable_k≤30', XA, XA_names), ('B_full_k≤150', XB, XB_names)]:
        for target_name, target in [('regression_ticks', y), ('class_win', (y > 0).astype(np.float32))]:
            for lam in [0.1, 1.0, 10.0]:
                oos = loo_day(X, target, dates, lam=lam)
                # Take if oos > median (per-fold threshold) — robust to scale issues
                for pct in [50, 60, 70, 80]:
                    thr = np.percentile(oos, pct)
                    take = oos > thr
                    n_take = int(take.sum())
                    if n_take < 5:
                        continue
                    pnl = float(y[take].sum())
                    wr = float((y[take] > 0).mean())
                    tpf = float(y[take].mean())
                    per_day = per_day_breakdown(take, y, dates)
                    max_day_conc = max((v['n'] for v in per_day.values()), default=0) / n_take
                    results.append(dict(
                        tier=tier, target=target_name, lam=lam, take_pct=pct,
                        n_take=n_take, pnl=pnl, wr=wr, tpf=tpf,
                        max_day_conc=max_day_conc,
                        per_day=per_day,
                    ))

    df = pd.DataFrame(results).sort_values(['tpf', 'wr'], ascending=False)
    print(f"\n=== Mid-trade gating results, sorted by tpf ===")
    print(df[['tier', 'target', 'lam', 'take_pct', 'n_take', 'wr', 'tpf',
              'max_day_conc', 'pnl']].head(25).to_string(index=False))

    # Bottom for sanity (worst gates = anti-edge events confirming signal)
    print(f"\n=== BOTTOM 10 (anti-edge — gates that pick BAD trades, sanity check) ===")
    print(df[['tier', 'target', 'lam', 'take_pct', 'n_take', 'wr', 'tpf',
              'max_day_conc', 'pnl']].tail(10).to_string(index=False))

    # Best per tier
    bestA = df[df.tier == 'A_tradeable_k≤30'].iloc[0]
    bestB = df[df.tier == 'B_full_k≤150'].iloc[0]
    print(f"\n=== Best Tier A (TRADEABLE in real-time): tpf={bestA.tpf:+.3f}, wr={bestA.wr:.3f}, n_take={bestA.n_take}, max_day={bestA.max_day_conc:.2f} ===")
    print(json.dumps(bestA.per_day, indent=2, default=str))

    # Save
    out = dict(
        generated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        gate_base='intra_vol30s_lt_1.75',
        n_events=int(len(sigg)),
        baseline=dict(tpf=float(y.mean()), wr=float((y>0).mean()), pnl=float(y.sum())),
        tier_A_features=XA_names,
        tier_B_features=XB_names,
        top_25=df.head(25).to_dict(orient='records'),
        best_tier_A=bestA.to_dict(),
        best_tier_B=bestB.to_dict(),
        elapsed=round(time.time() - t0, 2),
    )
    json.dump(out, open(OUT / 'midtrade_gate_results.json', 'w'), indent=2, default=str)
    print(f"\nWrote {OUT/'midtrade_gate_results.json'}, elapsed {time.time()-t0:.2f}s")


if __name__ == '__main__':
    main()
