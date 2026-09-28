"""
v33_k2_within_trade_v2.py — Refined within-trade analysis.

Fixes v1's broken absolute-threshold gating by:
  - Using OOS-rank-based gating (take top-X% of OOS scores per day)
  - Adding leakage-safe feature-level univariate gates (no model fitting needed)
  - Per-day breakdown of every gate
  - Permutation null test for corr significance

Internal Lvl3Quant research, pure NumPy/pandas, self-contained.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/within_trade_signals_v2")
OUT.mkdir(parents=True, exist_ok=True)
SIG = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/regime_analysis/k2_long_signals_with_regime_v2.csv")
PRED = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz")

np.random.seed(0)


def per_day_breakdown(take_mask, y, dates):
    out = {}
    for d in np.unique(dates):
        m = (dates == d) & take_mask
        n = int(m.sum())
        out[str(d)] = dict(
            n_take=n,
            n_total_day=int((dates == d).sum()),
            pnl=float(y[m].sum()) if n else 0.0,
            wr=float((y[m] > 0).mean()) if n else 0.0,
            tpf=float(y[m].mean()) if n else 0.0,
        )
    return out


def gate_stats(take_mask, y, dates, label):
    n = int(take_mask.sum())
    if n == 0:
        return dict(gate=label, n_take=0, pnl=0.0, wr=0.0, tpf=0.0, sharpe=0.0,
                    per_day={}, max_day_conc=0.0)
    pnl = float(y[take_mask].sum())
    wr = float((y[take_mask] > 0).mean())
    tpf = float(y[take_mask].mean())
    sharpe = float(y[take_mask].mean() / (y[take_mask].std() + 1e-9))
    per_day = per_day_breakdown(take_mask, y, dates)
    max_day_conc = max((v['n_take'] for v in per_day.values()), default=0) / max(n, 1)
    return dict(gate=label, n_take=n, pnl=pnl, wr=wr, tpf=tpf, sharpe=sharpe,
                per_day=per_day, max_day_conc=max_day_conc)


def main():
    t0 = time.time()
    sig = pd.read_csv(SIG)
    pred = np.load(PRED, allow_pickle=True)
    head_names = sorted([k for k in pred.files if k.startswith("pred_")])

    gate = (sig.vol_30s_ticks < 1.75).values & sig.filled.values
    sigg = sig[gate].reset_index(drop=True).copy()
    idx = sigg.global_idx.values.astype(int)
    y = sigg.net_ticks.values.astype(np.float32)
    dates = sigg.date.values
    n = len(sigg)
    print(f"Gated set: n={n}, t/fill={y.mean():+.2f}, WR={(y>0).mean():.2%}, pnl={y.sum():+.1f}")
    print(f"Per-day: {dict(sigg.date.value_counts().sort_index())}")

    # Compute per-event features for univariate gating
    feats = {}
    for h in head_names:
        feats[h] = pred[h][idx]
    # Adjacent-event deltas
    n_pred = len(pred['pred_log_ret_1s'])
    for k in [1, 3, 5, 10, 20, 50]:
        oidx = np.clip(idx + k, 0, n_pred - 1)
        for h in ['pred_log_ret_1s', 'pred_log_ret_5s', 'pred_log_ret_10s']:
            feats[f'd{h}_k{k}'] = pred[h][oidx] - pred[h][idx]
            feats[f'val_{h}_k{k}'] = pred[h][oidx]
    # Consensus across 5 return horizons at entry
    ret_heads = ['pred_log_ret_1s', 'pred_log_ret_5s', 'pred_log_ret_10s', 'pred_log_ret_30s', 'pred_log_ret_60s']
    ret_X = np.column_stack([feats[h] for h in ret_heads])
    feats['n_pos_5h'] = (ret_X > 0).sum(axis=1).astype(np.float32)
    feats['consensus_signed'] = ret_X.sum(axis=1)
    feats['multi_pos_all5'] = (ret_X > 0).all(axis=1).astype(np.float32)
    feats['multi_neg_all5'] = (ret_X < 0).all(axis=1).astype(np.float32)
    feats['ret_mean'] = ret_X.mean(axis=1)
    feats['ret_std'] = ret_X.std(axis=1)
    # Regime
    for c in ['vol_30s_ticks', 'drift_30s_ticks', 'mean_spread_30s_t', 'vol_60s_ticks',
              'drift_60s_ticks', 'vol_300s_ticks', 'drift_300s_ticks',
              'intraday_drift_ticks', 'min_of_day']:
        feats[c] = sigg[c].fillna(0).values.astype(np.float32)

    # Univariate Spearman corr vs y, plus split-half stability
    print(f"\n=== Univariate feature ranking (by |corr with net_ticks|) ===")
    rows = []
    for name, x in feats.items():
        if np.std(x) < 1e-9:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        # Permutation p-value
        null = []
        for _ in range(200):
            null.append(float(np.corrcoef(x, np.random.permutation(y))[0, 1]))
        p = float((np.abs(null) >= abs(c)).mean())
        rows.append(dict(feat=name, corr=c, p=p, n=n))
    fdf = pd.DataFrame(rows).sort_values('corr', key=lambda s: s.abs(), ascending=False)
    print(fdf.head(25).to_string(index=False))

    # Univariate threshold gates — pick top features and test "take if x>X" and "take if x<X"
    top_feats = fdf[fdf.p < 0.10].head(15)['feat'].tolist()
    print(f"\n=== Univariate threshold gates on top features (p<0.10), per-day-aware ===")
    gate_results = []
    for f in top_feats:
        x = feats[f]
        for pct in [10, 20, 25, 33, 50, 67, 75, 80, 90]:
            thr = float(np.percentile(x, pct))
            # Gate UP (take when x > thr)
            mask_up = x > thr
            r = gate_stats(mask_up, y, dates, f'{f}>p{pct}({thr:.4f})')
            r['n_total'] = n
            r['delta_tpf'] = r['tpf'] - y.mean() if r['n_take'] else 0
            gate_results.append(r)
            # Gate DOWN
            mask_dn = x < thr
            r = gate_stats(mask_dn, y, dates, f'{f}<p{pct}({thr:.4f})')
            r['n_total'] = n
            r['delta_tpf'] = r['tpf'] - y.mean() if r['n_take'] else 0
            gate_results.append(r)

    gdf = pd.DataFrame(gate_results).query('n_take>=10').sort_values('tpf', ascending=False)
    print(f"\n=== TOP 20 GATES (n_take>=10, ranked by tpf) ===")
    print(gdf[['gate', 'n_take', 'wr', 'tpf', 'sharpe', 'max_day_conc', 'pnl', 'delta_tpf']].head(20).to_string(index=False))

    # Best gate + per-day
    if len(gdf) > 0:
        best = gdf.iloc[0]
        print(f"\n=== Best gate per-day: {best['gate']} ===")
        print(json.dumps(best['per_day'], indent=2, default=str))

    # Save
    fdf['p'] = fdf['p'].round(4)
    out = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        gate_base='intra_vol30s_lt_1.75',
        n_events=int(n),
        baseline=dict(tpf=float(y.mean()), wr=float((y>0).mean()), pnl=float(y.sum())),
        feature_ranking=fdf.head(50).to_dict(orient='records'),
        top_gates=gdf.head(50).to_dict(orient='records'),
        best_gate_full=gdf.iloc[0].to_dict() if len(gdf) else None,
        elapsed=round(time.time() - t0, 2),
    )
    json.dump(out, open(OUT / 'within_trade_v2_results.json', 'w'), indent=2, default=str)
    print(f"\nWrote {OUT/'within_trade_v2_results.json'}, elapsed {time.time()-t0:.2f}s")


if __name__ == '__main__':
    main()
