"""
v33_exec_policy_v2.py — HC #374 build #3.

Two improvements over v1:
1. **Permutation null test**: shuffle final P&L labels within day and re-run LOO-day CV.
   How big is the +229 ticks "lift" expected under random? Anything < 95th percentile
   of the null distribution = not statistically significant.
2. **Per-decision-time isolation**: train separately at each decision-time using ONLY
   features available at that t (val_* at decision-time, deltas from entry, regime).
   Goal: see if EARLY decision-times (t=4-10) actually give the same lift as LATE
   (t=80-100), or if late info is what matters.
3. **Non-linear option**: gradient-boosted regression via histogram trees, pure NumPy.
   For speed, fall back to ridge if implementation is too slow.

Internal Lvl3Quant research.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/exec_policy_v2")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/exec_decision_dataset/decision_dataset.parquet")

EXIT_THRESH = -0.376
N_PERM = 50  # permutation null shuffles
np.random.seed(0)


def ridge_fit_predict(Xtr, ytr, Xte, lam=10.0):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr_z = (Xtr - mu) / sd
    Xte_z = (Xte - mu) / sd
    X1 = np.hstack([Xtr_z, np.ones((len(Xtr_z), 1))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    A[-1, -1] = 0
    coef = np.linalg.solve(A, X1.T @ ytr)
    return np.hstack([Xte_z, np.ones((len(Xte_z), 1))]) @ coef


def loo_day_at_t(df_t, feat_cols, lam=10.0):
    """LOO-day for single decision-time t."""
    df_t = df_t.copy()
    df_t['oos_pred'] = np.nan
    for d in sorted(df_t.date.unique()):
        tr = df_t[df_t.date != d]
        te = df_t[df_t.date == d]
        if len(tr) < 20 or len(te) == 0:
            continue
        Xtr = tr[feat_cols].fillna(0).values.astype(np.float32)
        ytr = tr['y_final_net_ticks'].values.astype(np.float32)
        Xte = te[feat_cols].fillna(0).values.astype(np.float32)
        oos = ridge_fit_predict(Xtr, ytr, Xte, lam=lam)
        df_t.loc[te.index, 'oos_pred'] = oos
    return df_t


def policy_pnl(df_oos, thresh):
    """Earliest-exit policy. Returns realized pnl, n_held, held_pnl."""
    df_oos = df_oos.dropna(subset=['oos_pred'])
    realized = 0.0
    n_held = 0; held_pnl = 0.0; n_exit = 0; held_wr_sum = 0
    correctly_blocked = 0; falsely_blocked = 0
    for tid, g in df_oos.groupby('trade_id'):
        g = g.sort_values('t_events')
        actual = g['y_final_net_ticks'].iloc[0]
        triggered = g[g.oos_pred < thresh]
        if len(triggered) > 0:
            realized += 0.0
            n_exit += 1
            if actual < EXIT_THRESH:
                correctly_blocked += 1
            elif actual > 0:
                falsely_blocked += 1
        else:
            realized += actual
            n_held += 1
            held_pnl += actual
            if actual > 0:
                held_wr_sum += 1
    return dict(realized=realized, n_held=n_held, held_pnl=held_pnl, n_exit=n_exit,
                held_wr=held_wr_sum/n_held if n_held else 0,
                held_tpf=held_pnl/n_held if n_held else 0,
                correctly_blocked=correctly_blocked, falsely_blocked=falsely_blocked)


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    print(f"Loaded: {len(df)} rows, {df.trade_id.nunique()} trades")

    feat_cols = [c for c in df.columns if c.startswith(('val_', 'd_'))]
    extras = ['n_pos_5h', 'n_neg_5h', 'mag_signed', 'mag_abs_mean', 'multi_pos_all5',
              'multi_neg_all5', 'ret_mean', 'ret_std',
              'unc_log_ret_10s', 'unc_log_ret_30s', 'unc_log_ret_60s',
              'vol_30s_ticks', 'drift_30s_ticks', 'mean_spread_30s_t',
              'vol_60s_ticks', 'drift_60s_ticks', 'mean_spread_60s_t',
              'vol_300s_ticks', 'drift_300s_ticks', 'mean_spread_300s_t',
              'intraday_drift_ticks', 'min_of_day', 'p60_entry', 'p5m_entry']
    feat_cols += [c for c in extras if c in df.columns]
    print(f"Features: {len(feat_cols)}")

    baseline_pnl = float(df.groupby('trade_id')['y_final_net_ticks'].first().sum())
    n_trades = int(df.trade_id.nunique())
    print(f"Baseline static-hold: pnl={baseline_pnl:+.1f}, n_trades={n_trades}\n")

    # ===== Per-decision-time isolation =====
    print(f"=== Per-decision-time isolated training (LOO-day) ===")
    per_t = []
    for t in sorted(df.t_events.unique()):
        df_t = df[df.t_events == t].copy()
        df_t_oos = loo_day_at_t(df_t, feat_cols, lam=10.0)
        df_t_oos = df_t_oos.dropna(subset=['oos_pred'])
        if len(df_t_oos) == 0:
            continue
        corr = float(np.corrcoef(df_t_oos.oos_pred, df_t_oos.y_final_net_ticks)[0, 1])
        # Best thresh sweep
        best_pnl = -1e9; best_thresh = None; best_stats = None
        for thresh in [-2.0, -1.0, -0.5, -0.376, -0.2, -0.1, 0.0, 0.1, 0.376, 1.0]:
            stats = policy_pnl(df_t_oos.assign(t_events=t), thresh)
            if stats['realized'] > best_pnl:
                best_pnl = stats['realized']
                best_thresh = thresh
                best_stats = stats
        per_t.append(dict(t=t, t_sec=t*0.25, corr=corr,
                          best_thresh=best_thresh, best_pnl=best_pnl, **best_stats))
        print(f"  t={t:3d} ({t*0.25:4.1f}s): corr={corr:+.3f}, best_thr={best_thresh:+.2f}, "
              f"realized={best_pnl:+.1f}, n_held={best_stats['n_held']}, "
              f"held_tpf={best_stats['held_tpf']:+.3f}, held_wr={best_stats['held_wr']:.3f}")

    # ===== Permutation null test =====
    print(f"\n=== Permutation null (shuffle y within day, {N_PERM} reps) ===")
    # Use the t=20 case as representative (mid-trade)
    df_t = df[df.t_events == 20].copy().reset_index(drop=True)
    null_pnls = []
    null_corrs = []
    for rep in range(N_PERM):
        df_perm = df_t.copy()
        # shuffle within each day
        for d in df_perm.date.unique():
            mask = df_perm.date == d
            df_perm.loc[mask, 'y_final_net_ticks'] = \
                np.random.permutation(df_perm.loc[mask, 'y_final_net_ticks'].values)
        df_perm_oos = loo_day_at_t(df_perm, feat_cols, lam=10.0)
        df_perm_oos = df_perm_oos.dropna(subset=['oos_pred'])
        if len(df_perm_oos) == 0:
            continue
        c = float(np.corrcoef(df_perm_oos.oos_pred, df_perm_oos.y_final_net_ticks)[0, 1])
        s = policy_pnl(df_perm_oos.assign(t_events=20), thresh=0.0)
        null_pnls.append(s['realized'])
        null_corrs.append(c)
    null_pnls = np.array(null_pnls)
    null_corrs = np.array(null_corrs)

    # Real result at t=20, thresh=0
    df_t20 = df[df.t_events == 20].copy()
    df_t20_oos = loo_day_at_t(df_t20, feat_cols, lam=10.0)
    df_t20_oos = df_t20_oos.dropna(subset=['oos_pred'])
    real_pnl = policy_pnl(df_t20_oos.assign(t_events=20), thresh=0.0)['realized']
    real_corr = float(np.corrcoef(df_t20_oos.oos_pred, df_t20_oos.y_final_net_ticks)[0, 1])

    p_pnl = float((null_pnls >= real_pnl).mean())
    p_corr = float((null_corrs >= real_corr).mean())
    print(f"  Real t=20, thresh=0: realized={real_pnl:+.1f}, corr={real_corr:+.3f}")
    print(f"  Null mean realized: {null_pnls.mean():+.1f} | std: {null_pnls.std():.1f} | max: {null_pnls.max():+.1f}")
    print(f"  Null mean corr: {null_corrs.mean():+.3f} | std: {null_corrs.std():.3f} | max: {null_corrs.max():+.3f}")
    print(f"  p(real_pnl ≤ null_pnl) = {p_pnl:.3f}")
    print(f"  p(real_corr ≤ null_corr) = {p_corr:.3f}")

    out = dict(
        generated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        n_trades=n_trades,
        baseline_pnl=baseline_pnl,
        per_decision_time=per_t,
        null_test=dict(
            decision_t=20,
            real_pnl=real_pnl, real_corr=real_corr,
            null_mean_pnl=float(null_pnls.mean()), null_std_pnl=float(null_pnls.std()),
            null_max_pnl=float(null_pnls.max()),
            null_mean_corr=float(null_corrs.mean()), null_std_corr=float(null_corrs.std()),
            null_max_corr=float(null_corrs.max()),
            p_pnl=p_pnl, p_corr=p_corr,
            n_perm=int(len(null_pnls)),
        ),
        elapsed=round(time.time() - t0, 2),
    )
    json.dump(out, open(OUT / 'policy_v2_results.json', 'w'), indent=2, default=str)
    print(f"\nSaved {OUT/'policy_v2_results.json'}, elapsed {time.time()-t0:.2f}s")


if __name__ == '__main__':
    main()
