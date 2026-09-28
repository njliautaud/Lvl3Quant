"""
v33_exec_policy_train.py — HC #374 build #2.

Train a per-decision-time regression model to predict final 30s P&L from each
decision-event's full feature set (~85 features). Then evaluate the simple
EXIT-WHEN-PRED-BAD policy:

  At each decision-time t for trade i:
    If pred_final_pnl < exit_threshold (-0.376 = commission cost):
      → assume we exit, realized P&L for this trade = 0 (conservative)
    Else continue holding.

  If never exit, realized = actual final net_ticks.

Compare policy realized P&L vs static-hold baseline (sum of all final net_ticks).

Method: LOO-day CV ridge per decision-time. Pure NumPy.

Internal Lvl3Quant research.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/exec_policy_v1")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/exec_decision_dataset/decision_dataset.parquet")

EXIT_THRESHOLD = -0.376  # commission cost in ticks; if model predicts below this, abort


def ridge_fit_predict(Xtr, ytr, Xte, lam=1.0):
    """Z-score on train, ridge fit, predict on test."""
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr_z = (Xtr - mu) / sd
    Xte_z = (Xte - mu) / sd
    X1 = np.hstack([Xtr_z, np.ones((len(Xtr_z), 1))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    A[-1, -1] = 0
    coef = np.linalg.solve(A, X1.T @ ytr)
    return np.hstack([Xte_z, np.ones((len(Xte_z), 1))]) @ coef


def loo_day_per_decision(df, feat_cols, lam=10.0):
    """For each decision-time, run LOO-day CV ridge. Returns df with oos_pred column."""
    df = df.copy()
    df['oos_pred'] = np.nan
    for t in sorted(df.t_events.unique()):
        sub = df[df.t_events == t]
        for d in sorted(sub.date.unique()):
            tr = sub[sub.date != d]
            te = sub[sub.date == d]
            if len(tr) < 20 or len(te) == 0:
                continue
            Xtr = tr[feat_cols].fillna(0).values.astype(np.float32)
            ytr = tr['y_final_net_ticks'].values.astype(np.float32)
            Xte = te[feat_cols].fillna(0).values.astype(np.float32)
            oos = ridge_fit_predict(Xtr, ytr, Xte, lam=lam)
            df.loc[te.index, 'oos_pred'] = oos
    return df


def evaluate_policy(df, exit_thresh, label_col='y_final_net_ticks'):
    """For each trade_id, find the EARLIEST decision-time where oos_pred < exit_thresh.
    If found, realized = 0 (assume early exit at break-even).
    If not, realized = label_col."""
    out = []
    for tid, g in df.groupby('trade_id'):
        g = g.sort_values('t_events')
        triggered = g[g.oos_pred < exit_thresh]
        date = g.date.iloc[0]
        actual = g[label_col].iloc[0]
        if len(triggered) > 0:
            exit_t = int(triggered.t_events.iloc[0])
            realized = 0.0  # conservative approximation
            decision = 'EXIT'
        else:
            exit_t = -1
            realized = float(actual)
            decision = 'HOLD'
        out.append(dict(trade_id=tid, date=int(date), actual=float(actual),
                         realized=realized, exit_t=exit_t, decision=decision))
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    print(f"Loaded dataset: {len(df)} rows × {df.shape[1]} cols, {df.trade_id.nunique()} trades")

    feat_cols = [c for c in df.columns if c.startswith(('val_', 'd_'))]
    feat_cols += ['n_pos_5h', 'n_neg_5h', 'mag_signed', 'mag_abs_mean', 'multi_pos_all5',
                   'multi_neg_all5', 'ret_mean', 'ret_std',
                   'unc_log_ret_10s', 'unc_log_ret_30s', 'unc_log_ret_60s',
                   'vol_30s_ticks', 'drift_30s_ticks', 'mean_spread_30s_t',
                   'vol_60s_ticks', 'drift_60s_ticks', 'mean_spread_60s_t',
                   'vol_300s_ticks', 'drift_300s_ticks', 'mean_spread_300s_t',
                   'intraday_drift_ticks', 'min_of_day',
                   'p60_entry', 'p5m_entry', 't_seconds']
    feat_cols = [c for c in feat_cols if c in df.columns]
    print(f"Using {len(feat_cols)} features")

    # Run LOO-day CV per decision-time
    best_lam = 10.0
    df_oos = loo_day_per_decision(df, feat_cols, lam=best_lam)
    df_oos = df_oos.dropna(subset=['oos_pred'])
    print(f"\nOOS predictions computed: {df_oos.shape[0]} rows")

    # Per-decision-time corr
    print(f"\n=== Per-decision-time corr(oos_pred, y) ===")
    for t in sorted(df_oos.t_events.unique()):
        sub = df_oos[df_oos.t_events == t]
        c = float(np.corrcoef(sub.oos_pred, sub.y_final_net_ticks)[0, 1]) if len(sub) > 1 else 0
        print(f"  t={t:3d} events ({t*0.25:4.1f}s): n={len(sub)}, corr={c:+.3f}")

    # Try multiple exit thresholds
    print(f"\n=== Exit-threshold policy evaluation (LOO-day, n_trades={df_oos.trade_id.nunique()}) ===")
    baseline_pnl = float(df_oos.groupby('trade_id')['y_final_net_ticks'].first().sum())
    baseline_tpf = float(df_oos.groupby('trade_id')['y_final_net_ticks'].first().mean())
    n_trades_total = int(df_oos.trade_id.nunique())
    baseline_wr = float((df_oos.groupby('trade_id')['y_final_net_ticks'].first() > 0).mean())
    print(f"  BASELINE static-hold all: pnl={baseline_pnl:+.1f}, t/fill={baseline_tpf:+.3f}, WR={baseline_wr:.3f}, n={n_trades_total}")

    pol_rows = []
    for thresh in [-2.0, -1.0, -0.5, -0.376, -0.2, -0.1, 0.0, 0.1, 0.376, 0.5, 1.0]:
        pol = evaluate_policy(df_oos, thresh)
        n_exit = int((pol.decision == 'EXIT').sum())
        n_held = n_trades_total - n_exit
        # Of the EXITs, how many would have been losers anyway? (sanity)
        n_correctly_blocked = int(((pol.decision == 'EXIT') & (pol.actual < EXIT_THRESHOLD)).sum())
        n_falsely_blocked = int(((pol.decision == 'EXIT') & (pol.actual > 0)).sum())
        n_correctly_held = int(((pol.decision == 'HOLD') & (pol.actual > 0)).sum())
        # Realized P&L
        realized_pnl = float(pol.realized.sum())
        realized_tpf_all = realized_pnl / n_trades_total if n_trades_total else 0
        realized_tpf_held = float(pol[pol.decision == 'HOLD'].realized.mean()) if n_held else 0
        held_wr = float((pol[pol.decision == 'HOLD'].actual > 0).mean()) if n_held else 0
        # Per-day breakdown
        per_day = {}
        for d in sorted(pol.date.unique()):
            dpol = pol[pol.date == d]
            per_day[str(int(d))] = dict(
                n_total=int(len(dpol)),
                n_exit=int((dpol.decision == 'EXIT').sum()),
                n_held=int((dpol.decision == 'HOLD').sum()),
                held_pnl=float(dpol[dpol.decision == 'HOLD'].actual.sum()),
                held_wr=float((dpol[dpol.decision == 'HOLD'].actual > 0).mean()) if (dpol.decision == 'HOLD').any() else 0,
            )
        pol_rows.append(dict(
            exit_thresh=thresh, n_exit=n_exit, n_held=n_held,
            n_correctly_blocked=n_correctly_blocked,
            n_falsely_blocked=n_falsely_blocked,
            n_correctly_held=n_correctly_held,
            realized_pnl=realized_pnl, realized_tpf_all=realized_tpf_all,
            realized_tpf_held=realized_tpf_held, held_wr=held_wr,
            per_day=per_day,
        ))

    pol_df = pd.DataFrame(pol_rows)
    print(pol_df[['exit_thresh', 'n_exit', 'n_held', 'n_correctly_blocked', 'n_falsely_blocked',
                  'realized_pnl', 'realized_tpf_all', 'realized_tpf_held', 'held_wr']].to_string(index=False))

    # Save
    out = dict(
        generated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        baseline=dict(pnl=baseline_pnl, tpf=baseline_tpf, wr=baseline_wr, n=n_trades_total),
        lam=best_lam,
        n_features=len(feat_cols),
        decision_times=sorted(df_oos.t_events.unique().tolist()),
        policy_sweep=pol_rows,
        per_decision_corr={int(t): float(np.corrcoef(df_oos[df_oos.t_events==t].oos_pred,
                                                       df_oos[df_oos.t_events==t].y_final_net_ticks)[0,1])
                            for t in sorted(df_oos.t_events.unique())},
        elapsed=round(time.time() - t0, 2),
    )
    json.dump(out, open(OUT / 'policy_v1_results.json', 'w'), indent=2, default=str)
    df_oos.to_parquet(OUT / 'oos_predictions.parquet', index=False)
    print(f"\nSaved {OUT}/policy_v1_results.json")
    print(f"Elapsed: {time.time()-t0:.2f}s")


if __name__ == '__main__':
    main()
