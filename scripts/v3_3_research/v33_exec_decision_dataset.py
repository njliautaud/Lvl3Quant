"""
v33_exec_decision_dataset.py — HC #374 build #1.

For each filled K=2 LONG trade (490 events on 5-day OOT), generate a per-decision-event
dataset capturing the FULL state of the model + regime at every potential exit point
during the 30s hold window.

DECISION TIMES (events after entry; stride ≈ 250ms):
  t_events ∈ {4, 10, 20, 30, 40, 60, 80, 100}  ≈ {1, 2.5, 5, 7.5, 10, 15, 20, 25}s

PER (trade, decision_t) FEATURES (~85):
  - 32 v3.3 head VALUES at global_idx + t_events
  - 32 v3.3 head DELTAS (current value - entry value)
  - Cross-head consensus: n_pos_5h, mag_signed, mag_abs_mean, multi_pos, multi_neg
  - Quantile uncertainty: pred_log_ret_10s_q90 - q10, same for 30s/60s
  - Regime at entry: vol_30/60/300, drift_30/60/300, spread_30/60/300, intraday_drift, min_of_day
  - Decision time: t_events, t_seconds_into_hold, bucket
  - Trade meta: entry_p60, entry_p5m (the K=2 stack values)

LABEL: final net_ticks of the trade (regression target).
       This is the SAME label for every decision-time of a given trade — that's intentional:
       we want the model to predict "what will this trade end up at?" given current state.

CAVEAT (acknowledged): without mid-hold price paths we can't simulate true adaptive-exit P&L.
We approximate "exit now" as P&L = 0 (commission-only, no spread, no slip). This is
conservative for a LONG trade that's going to lose ≥0.376 (commission); aggressive otherwise.
Real HC #357 adaptive replay is the next build (v33_exec_adaptive_replay.py).

Internal Lvl3Quant research, pure NumPy/pandas, self-contained.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/exec_decision_dataset")
OUT.mkdir(parents=True, exist_ok=True)
SIG = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/regime_analysis/k2_long_signals_with_regime_v2.csv")
PRED = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz")

# Stride ≈ 250ms, so t_events = seconds * 4
DECISION_T_EVENTS = [4, 10, 20, 30, 40, 60, 80, 100]
DECISION_T_SECS = [t * 0.25 for t in DECISION_T_EVENTS]


def main():
    t0 = time.time()
    sig = pd.read_csv(SIG)
    pred = np.load(PRED, allow_pickle=True)
    head_names = sorted([k for k in pred.files if k.startswith("pred_")])
    n_pred = len(pred['pred_log_ret_1s'])
    print(f"Loaded {len(sig)} K=2 signals, {n_pred} pred events, {len(head_names)} heads")

    # Use ALL filled K=2 LONG trades, not just the vol30s<1.75 subset — bigger dataset
    sigf = sig[sig.filled].reset_index(drop=True).copy()
    idx_entry = sigf.global_idx.values.astype(int)
    y = sigf.net_ticks.values.astype(np.float32)
    dates = sigf.date.values
    print(f"Filled K=2 LONG fills: {len(sigf)} | t/fill={y.mean():+.3f} | WR={(y>0).mean():.2%}")
    print(f"Per-day: {dict(sigf.date.value_counts().sort_index())}")

    # Preload all head arrays once
    head_arr = {h: pred[h] for h in head_names}

    # Quantile heads for uncertainty
    qh = {}
    for h_base in ['log_ret_10s', 'log_ret_30s', 'log_ret_60s']:
        q10 = f'pred_{h_base}_q10'
        q90 = f'pred_{h_base}_q90'
        if q10 in head_arr and q90 in head_arr:
            qh[h_base] = (head_arr[q10], head_arr[q90])

    ret_heads = ['pred_log_ret_1s', 'pred_log_ret_5s', 'pred_log_ret_10s',
                 'pred_log_ret_30s', 'pred_log_ret_60s']

    rows = []
    regime_cols = ['vol_30s_ticks', 'drift_30s_ticks', 'mean_spread_30s_t',
                   'vol_60s_ticks', 'drift_60s_ticks', 'mean_spread_60s_t',
                   'vol_300s_ticks', 'drift_300s_ticks', 'mean_spread_300s_t',
                   'intraday_drift_ticks', 'min_of_day']

    for i, (gidx, date, p60, p5m, y_i, bucket) in enumerate(zip(
        idx_entry, dates, sigf.p60.values, sigf.p5m.values, y, sigf.bucket.values)):
        # Entry-time head values
        entry_vals = {h: head_arr[h][gidx] for h in head_names}
        regime = {c: float(sigf[c].iloc[i]) if c in sigf.columns and not pd.isna(sigf[c].iloc[i]) else 0.0
                  for c in regime_cols}

        for t_ev, t_s in zip(DECISION_T_EVENTS, DECISION_T_SECS):
            dec_idx = min(gidx + t_ev, n_pred - 1)
            row = {
                'trade_id': int(i),
                'entry_global_idx': int(gidx),
                'date': int(date),
                't_events': int(t_ev),
                't_seconds': float(t_s),
                'bucket': str(bucket),
                'p60_entry': float(p60),
                'p5m_entry': float(p5m),
                'y_final_net_ticks': float(y_i),
            }
            # Head values at decision-time and deltas from entry
            vals = {h: head_arr[h][dec_idx] for h in head_names}
            for h in head_names:
                row[f'val_{h}'] = float(vals[h])
                row[f'd_{h}'] = float(vals[h] - entry_vals[h])

            # Cross-head consensus (at decision time)
            ret_v = np.array([vals[h] for h in ret_heads if h in vals])
            row['n_pos_5h'] = int((ret_v > 0).sum())
            row['n_neg_5h'] = int((ret_v < 0).sum())
            row['mag_signed'] = float(ret_v.sum())
            row['mag_abs_mean'] = float(np.abs(ret_v).mean())
            row['multi_pos_all5'] = int((ret_v > 0).all())
            row['multi_neg_all5'] = int((ret_v < 0).all())
            row['ret_mean'] = float(ret_v.mean())
            row['ret_std'] = float(ret_v.std())

            # Uncertainty (quantile spread) at decision-time
            for h_base, (q10arr, q90arr) in qh.items():
                row[f'unc_{h_base}'] = float(q90arr[dec_idx] - q10arr[dec_idx])

            # Regime (entry-time values, copied to each decision-event)
            for c in regime_cols:
                row[c] = regime[c]

            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nBuilt dataset: {len(df)} (trade, decision_t) rows × {df.shape[1]} cols")
    print(f"Per-trade row count: {len(DECISION_T_EVENTS)} | per-day:")
    print(df.groupby('date')['trade_id'].nunique().to_string())

    # Quick sanity: corr of label with a few key features
    print(f"\nSanity correlations (full dataset, all decision times):")
    for f in ['d_pred_fifo_tp8sl5_hit_tp', 'd_pred_pred_realized_vol_30s_ticks',
              'val_pred_fifo_tp8sl5_net', 'val_pred_pred_mae_60s_ticks',
              'val_pred_log_ret_1s', 'mag_signed', 'mean_spread_30s_t']:
        if f in df.columns:
            c = df[[f, 'y_final_net_ticks']].corr().iloc[0, 1]
            print(f"  corr({f}, y) = {c:+.3f}")

    # Save
    df.to_parquet(OUT / 'decision_dataset.parquet', index=False)
    meta = dict(
        generated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        n_trades=int(sigf.shape[0]),
        n_rows=int(len(df)),
        n_features=int(df.shape[1]),
        decision_t_events=DECISION_T_EVENTS,
        decision_t_seconds=DECISION_T_SECS,
        baseline=dict(tpf=float(y.mean()), wr=float((y>0).mean()), pnl=float(y.sum())),
        elapsed=round(time.time() - t0, 2),
    )
    json.dump(meta, open(OUT / 'meta.json', 'w'), indent=2, default=str)
    print(f"\nSaved: {OUT/'decision_dataset.parquet'}")
    print(f"Saved: {OUT/'meta.json'}")
    print(f"Elapsed: {time.time()-t0:.2f}s")


if __name__ == '__main__':
    main()
