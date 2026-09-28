"""
v33_exec_policy_v3_withinday.py — HC #374 build #4.

The v1/v2 +218t lift was a day-mean phantom: ridge LOO-day-CV just learned
"day 0225 is good, take it; days 0223/0226 are bad, skip them." Confirmed
by permutation null (p=1.000).

THIS BUILD: kill the day-mean confound by CONSTRUCTION via within-day K-fold.
- Train and test sets are sampled from the SAME day.
- Target: rank-within-day of y_final_net_ticks (or its z-score).
  → Model literally cannot win by learning day-mean — every fold's mean is 0.
- For each (date, decision_t), do 5-fold random CV within the day.
- Out-of-fold predictions are then compared to held-out y rank.
- Permutation null: shuffle y WITHIN each fold (so any within-day predictability
  that survives must be real signal, not day or fold leakage).

Two days have enough trades: 0223 (251), 0225 (221). Use both.
0224 (1) and 0226 (17) are too small — drop.

POLICY EVAL (within-day):
  For each held-out trade, the model emits a predicted-rank in [0,1].
  Policy: exit (realized=0) if pred_rank < q-th percentile of fold's pred_rank.
  Compare realized P&L to "take all" baseline within-fold.

Internal Lvl3Quant research. Pure NumPy/pandas. Reads only the existing
decision_dataset.parquet — no model training, no trainer code touched.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/exec_policy_v3_withinday")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/exec_decision_dataset/decision_dataset.parquet")

N_FOLDS = 5
N_PERM = 200
MIN_TRADES_PER_DAY = 50
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


def kfold_within_day(df_day, feat_cols, target_col, n_folds=5, lam=10.0, seed=0):
    """K-fold within a single day. Returns df_day with oos_pred column."""
    rng = np.random.RandomState(seed)
    n = len(df_day)
    perm = rng.permutation(n)
    fold_assign = np.zeros(n, dtype=int)
    for k in range(n_folds):
        lo = k * n // n_folds
        hi = (k + 1) * n // n_folds
        fold_assign[perm[lo:hi]] = k
    df_day = df_day.copy().reset_index(drop=True)
    df_day['fold'] = fold_assign
    df_day['oos_pred'] = np.nan
    for k in range(n_folds):
        tr_mask = df_day.fold != k
        te_mask = df_day.fold == k
        if tr_mask.sum() < 10 or te_mask.sum() == 0:
            continue
        Xtr = df_day.loc[tr_mask, feat_cols].fillna(0).values.astype(np.float32)
        ytr = df_day.loc[tr_mask, target_col].values.astype(np.float32)
        Xte = df_day.loc[te_mask, feat_cols].fillna(0).values.astype(np.float32)
        oos = ridge_fit_predict(Xtr, ytr, Xte, lam=lam)
        df_day.loc[te_mask, 'oos_pred'] = oos
    return df_day


def policy_pnl_within_day(df_day, pred_col='oos_pred', y_col='y_final_net_ticks', q_exit=0.30):
    """Earliest-exit policy at trade level. df_day must be per-trade (one row per trade_id).
    Exit (realized=0) trades whose pred is in the BOTTOM q_exit fraction within the day.
    """
    df_day = df_day.dropna(subset=[pred_col]).copy()
    if len(df_day) == 0:
        return dict(realized=0, n_held=0, n_exit=0, held_tpf=0, held_wr=0)
    thresh = df_day[pred_col].quantile(q_exit)
    df_day['exit'] = df_day[pred_col] < thresh
    held = df_day[~df_day['exit']]
    realized = float(held[y_col].sum())
    n_held = len(held)
    n_exit = int(df_day['exit'].sum())
    correctly_blocked = int(df_day[df_day['exit'] & (df_day[y_col] < -0.376)].shape[0])
    falsely_blocked = int(df_day[df_day['exit'] & (df_day[y_col] > 0)].shape[0])
    return dict(realized=realized, n_held=n_held, n_exit=n_exit,
                held_tpf=float(held[y_col].mean()) if n_held else 0,
                held_wr=float((held[y_col] > 0).mean()) if n_held else 0,
                correctly_blocked=correctly_blocked,
                falsely_blocked=falsely_blocked)


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    print(f"Loaded {len(df)} rows. Decision-times: {sorted(df.t_events.unique())}")

    # Per-trade Y for baseline (one row per trade_id)
    trade_y = df.groupby('trade_id').agg(date=('date', 'first'), y=('y_final_net_ticks', 'first')).reset_index()
    print("\nPer-day baseline:")
    print(trade_y.groupby('date')['y'].agg(['mean', 'std', 'count', 'sum']))

    # Eligible days
    elig_days = trade_y.groupby('date').size()
    elig_days = elig_days[elig_days >= MIN_TRADES_PER_DAY].index.tolist()
    print(f"\nEligible days (≥{MIN_TRADES_PER_DAY} trades): {elig_days}")

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

    # ===== PER-DECISION-T, PER-DAY WITHIN-DAY 5-FOLD CV =====
    print(f"\n=== Per-decision-t per-day within-day 5-fold (n_folds={N_FOLDS}) ===")

    per_t_per_day = []
    aggregated = []  # for the cross-day rollup
    for t in sorted(df.t_events.unique()):
        df_t = df[df.t_events == t].copy()
        for d in elig_days:
            df_td = df_t[df_t.date == d].copy()
            if len(df_td) < MIN_TRADES_PER_DAY:
                continue
            # Within-day rank target (or z-score)
            df_td['y_rank'] = df_td['y_final_net_ticks'].rank(pct=True) - 0.5
            df_td = kfold_within_day(df_td, feat_cols, 'y_rank', n_folds=N_FOLDS, lam=10.0)
            df_td = df_td.dropna(subset=['oos_pred'])
            if len(df_td) == 0:
                continue
            corr_rank = float(np.corrcoef(df_td.oos_pred, df_td.y_rank)[0, 1])
            corr_y = float(np.corrcoef(df_td.oos_pred, df_td.y_final_net_ticks)[0, 1])

            # Policy: exit bottom q
            best_lift = -1e9
            best_q = None; best_stats = None
            baseline_pnl = float(df_td.y_final_net_ticks.sum())
            baseline_tpf = float(df_td.y_final_net_ticks.mean())
            for q in [0.10, 0.20, 0.30, 0.40, 0.50]:
                stats = policy_pnl_within_day(df_td, q_exit=q)
                lift = stats['realized'] - baseline_pnl
                if lift > best_lift:
                    best_lift = lift
                    best_q = q
                    best_stats = stats
            per_t_per_day.append(dict(t=t, date=int(d), n=len(df_td),
                                       corr_rank=corr_rank, corr_y=corr_y,
                                       baseline_pnl=baseline_pnl, baseline_tpf=baseline_tpf,
                                       best_q=best_q, lift=best_lift, **best_stats))
            print(f"  t={t:3d} day={d}: n={len(df_td):3d}, corr_rank={corr_rank:+.3f}, "
                  f"corr_y={corr_y:+.3f}, base_pnl={baseline_pnl:+.1f}, "
                  f"best_q={best_q:.2f}, lift={best_lift:+.1f}, "
                  f"held_tpf={best_stats['held_tpf']:+.3f}, held_wr={best_stats['held_wr']:.3f}")

    # ===== PERMUTATION NULL =====
    print(f"\n=== Permutation null on best (t, day) combos ===")
    # Pick top 3 by lift (sanity) and run within-fold-shuffle null
    per_t_df = pd.DataFrame(per_t_per_day)
    top_combos = per_t_df.sort_values('corr_rank', ascending=False).head(3)
    null_results = []
    for _, row in top_combos.iterrows():
        t = int(row['t']); d = int(row['date'])
        df_td = df[(df.t_events == t) & (df.date == d)].copy()
        df_td['y_rank'] = df_td['y_final_net_ticks'].rank(pct=True) - 0.5

        # Run actual CV
        df_td_real = kfold_within_day(df_td, feat_cols, 'y_rank', n_folds=N_FOLDS, lam=10.0)
        real_corr = float(np.corrcoef(df_td_real.oos_pred.dropna(),
                                        df_td_real.loc[df_td_real.oos_pred.notna(), 'y_rank'])[0, 1])

        # Null: shuffle y_rank within fold assignments (same K-fold structure, shuffled target)
        null_corrs = []
        for rep in range(N_PERM):
            df_td_perm = df_td.copy().reset_index(drop=True)
            df_td_perm['y_rank'] = np.random.permutation(df_td_perm['y_rank'].values)
            df_td_perm = kfold_within_day(df_td_perm, feat_cols, 'y_rank', n_folds=N_FOLDS,
                                           lam=10.0, seed=rep)
            df_td_perm = df_td_perm.dropna(subset=['oos_pred'])
            if len(df_td_perm) > 1:
                c = float(np.corrcoef(df_td_perm.oos_pred, df_td_perm.y_rank)[0, 1])
                null_corrs.append(c)
        null_corrs = np.array(null_corrs)
        p = float((null_corrs >= real_corr).mean())
        null_results.append(dict(t=t, date=d, real_corr=real_corr,
                                   null_mean=float(null_corrs.mean()),
                                   null_std=float(null_corrs.std()),
                                   null_max=float(null_corrs.max()),
                                   p=p, n_perm=int(len(null_corrs))))
        print(f"  t={t} day={d}: real_corr={real_corr:+.3f}, "
              f"null_mean={null_corrs.mean():+.3f} ± {null_corrs.std():.3f}, "
              f"null_max={null_corrs.max():+.3f}, p(real≥null)={p:.3f}")

    # ===== SUMMARY =====
    out = dict(
        generated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        n_total_rows=int(len(df)),
        elig_days=[int(d) for d in elig_days],
        n_folds=N_FOLDS,
        n_features=len(feat_cols),
        per_t_per_day=per_t_per_day,
        null_test=null_results,
        elapsed=round(time.time() - t0, 2),
    )
    json.dump(out, open(OUT / 'policy_v3_withinday_results.json', 'w'), indent=2, default=str)
    per_t_df.to_csv(OUT / 'per_t_per_day.csv', index=False)
    print(f"\nSaved {OUT/'policy_v3_withinday_results.json'} | elapsed {time.time()-t0:.2f}s")

    # Conclusion line
    sig_count = sum(1 for r in null_results if r['p'] < 0.10)
    print(f"\nVERDICT: {sig_count}/{len(null_results)} top (t,day) combos pass p<0.10 within-fold null.")
    if sig_count == 0:
        print("  → No real within-day signal in v3.3 head features at any decision-time.")
        print("  → The phantom was the entire claimed edge. Bottleneck = extended OOT depth.")
    else:
        print("  → REAL within-day signal at the surviving (t,day) combos. Tradeable.")


if __name__ == '__main__':
    main()
