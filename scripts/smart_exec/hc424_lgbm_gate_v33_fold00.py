#!/usr/bin/env python3
"""HC #424 §3 fallback — LGBM execution gate on v3.3 fold_00 predictions.

Mirrors hc424_lgbm_gate_v342_ep3.py and _v2.py exactly. Same multi-head feature
set, same FIFO replay targets, same HC #74/#392/#344/#0/#393/#397B methodology.

Two outputs:
  1. Threshold-based verdicts (thr=0/0.5/1.0 cost-adjusted)
  2. Percentile-based verdicts (top 50/20/10/5/2/1%) — v2-style

Run-time target: ~3-5 min on Jupiter CPU.

HC #420 authorized: user's own quant-trading research codebase.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow

LVL3 = Path("/home/jupiter/Lvl3Quant")
NPZ_PATH = LVL3 / "output/hc424_jupiter_exec_research/inputs/v3_3_fold00/fold_00_predictions.npz"
NPZ_SHA256 = "b455f30373555ae71d991a8962d9f307dadba4c60ecd22c23eb723d9e38e06b0"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
OUT_DIR = LVL3 / "output/hc424_jupiter_exec_research"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
MARKET_COST_TICKS = COMMISSION_RT_TICKS + SPREAD_TICKS

OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]
DAY_CONC_GATE = 0.50

# Same 30 multi-head features as v3.4.2 ep3 work
MULTI_HEAD_FEATURES = [
    "pred_log_ret_1s",
    "pred_log_ret_5s",
    "pred_log_ret_10s",
    "pred_log_ret_30s",
    "pred_log_ret_60s",
    "pred_log_ret_5min",
    "pred_p_up_5s",
    "pred_p_up_10s",
    "pred_p_up_30s",
    "pred_p_up_60s",
    "pred_log_ret_10s_q10", "pred_log_ret_10s_q50", "pred_log_ret_10s_q90",
    "pred_log_ret_30s_q10", "pred_log_ret_30s_q50", "pred_log_ret_30s_q90",
    "pred_log_ret_60s_q10", "pred_log_ret_60s_q50", "pred_log_ret_60s_q90",
    "pred_pred_mfe_30s_ticks",
    "pred_pred_mae_30s_ticks",
    "pred_pred_mfe_60s_ticks",
    "pred_pred_mae_60s_ticks",
    "pred_pred_time_to_mfe_secs",
    "pred_p_reversal_15s",
    "pred_p_reversal_30s",
    "pred_p_reversal_60s",
    "pred_pred_realized_vol_30s_ticks",
    "pred_fifo_tp4sl3_net",
    "pred_fifo_tp4sl3_hit_tp",
]


def proportional_per_day_keep(fifo_n_per_day: dict, n_npz: int) -> dict:
    total = sum(fifo_n_per_day.values())
    diff = total - n_npz
    if diff < 0:
        raise RuntimeError(f"NPZ has more samples than FIFO ({n_npz} > {total})")
    keep = {}
    cum = 0
    dates = list(fifo_n_per_day.keys())
    for i, dt in enumerate(dates):
        if i < len(dates) - 1:
            frac = fifo_n_per_day[dt] / total
            t = int(round(diff * frac))
            keep[dt] = fifo_n_per_day[dt] - t
            cum += t
        else:
            keep[dt] = fifo_n_per_day[dt] - (diff - cum)
    assert sum(keep.values()) == n_npz
    return keep


def load_aligned_data(npz_path=NPZ_PATH):
    print(f"[load] npz: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    n = d["pred_log_ret_1s"].shape[0]
    print(f"[load] n_samples: {n}")

    feats = []
    for k in MULTI_HEAD_FEATURES:
        a = d[k][:n]
        feats.append(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0))
    X = np.stack(feats, axis=1).astype(np.float32)
    print(f"[load] X shape: {X.shape}")

    fifo_n = {dt: int(np.load(LABELS_DIR / f"{dt}_fifo_labels.npz")["window_k"].shape[0])
              for dt in OOT_DATES}
    keep_per_day = proportional_per_day_keep(fifo_n, n)
    print(f"[load] keep_per_day: {keep_per_day}")

    long_net, short_net = [], []
    long_filled, short_filled = [], []
    ts_ns, date_idx = [], []
    for i, dt in enumerate(OOT_DATES):
        z = np.load(LABELS_DIR / f"{dt}_fifo_labels.npz")
        k = keep_per_day[dt]
        long_net.append(z["tp4sl3_long_net_ticks"][:k].astype(np.float32))
        short_net.append(z["tp4sl3_short_net_ticks"][:k].astype(np.float32))
        long_filled.append(z["tp4sl3_long_filled"][:k].astype(bool))
        short_filled.append(z["tp4sl3_short_filled"][:k].astype(bool))
        ts_ns.append(z["ts_ns"][:k].astype(np.int64))
        date_idx.append(np.full(k, i, dtype=np.int32))
    long_net = np.concatenate(long_net)
    short_net = np.concatenate(short_net)
    long_filled = np.concatenate(long_filled)
    short_filled = np.concatenate(short_filled)
    ts_ns = np.concatenate(ts_ns)
    date_idx = np.concatenate(date_idx)

    print(f"[load] long net mean={long_net.mean():.3f} pct_filled={long_filled.mean():.3f}")
    print(f"[load] short net mean={short_net.mean():.3f} pct_filled={short_filled.mean():.3f}")

    return {
        "X": X, "long_net": long_net, "short_net": short_net,
        "long_filled": long_filled, "short_filled": short_filled,
        "ts_ns": ts_ns, "date_idx": date_idx,
        "feature_names": MULTI_HEAD_FEATURES,
    }


def time_split_per_day(date_idx: np.ndarray, holdout_frac: float = 0.20):
    n = date_idx.shape[0]
    train_mask = np.zeros(n, dtype=bool)
    hold_mask = np.zeros(n, dtype=bool)
    for i in np.unique(date_idx):
        rows = np.where(date_idx == i)[0]
        cut = int(len(rows) * (1.0 - holdout_frac))
        train_mask[rows[:cut]] = True
        hold_mask[rows[cut:]] = True
    return train_mask, hold_mask


def train_lgbm(X_tr, y_tr, X_ho):
    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
        verbosity=-1, n_estimators=500, n_jobs=-1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_ho, np.zeros(len(X_ho)))],
              callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    return model


def threshold_verdict(name, side, gate_signal, net_realized, filled_mask, date_idx_ho, threshold):
    take = (gate_signal > threshold) & filled_mask
    n = int(take.sum())
    if n == 0:
        return {"name": name, "side": side, "threshold": threshold,
                "n_signals": 0, "n_fills": 0,
                "ticks_per_signal": 0.0, "ticks_per_fill": 0.0,
                "sharpe": 0.0, "sortino": 0.0, "pf": 0.0, "wr": 0.0,
                "adv_sel_30s_avg": 0.0, "day_conc": 1.0, "pass_hc344": False,
                "total_ticks": 0.0,
                "queue_pos": "N/A", "cancel_window": "N/A"}
    realized = net_realized[take]
    mu = float(realized.mean())
    sd = float(realized.std(ddof=1)) if n > 1 else 0
    sharpe = (mu / sd) * np.sqrt(n) if sd > 0 else 0
    neg = realized[realized < 0]
    nsd = float(neg.std(ddof=1)) if len(neg) > 1 else 0
    sortino = (mu / nsd) * np.sqrt(n) if nsd > 0 else 0
    gwin = float(realized[realized > 0].sum())
    gloss = float(-realized[realized < 0].sum())
    pf = gwin / gloss if gloss > 0 else float("inf")
    wins = float((realized > 0).sum()); losses = float((realized < 0).sum())
    wr = wins / max(1.0, wins + losses)
    days = date_idx_ho[take]
    cnt = np.bincount(days) if len(days) > 0 else np.array([1])
    dc = float(cnt.max() / cnt.sum())
    return {"name": name, "side": side, "threshold": threshold,
            "n_signals": n, "n_fills": n,
            "ticks_per_signal": mu, "ticks_per_fill": mu,
            "sharpe": float(sharpe), "sortino": float(sortino),
            "pf": float(pf), "wr": float(wr),
            "adv_sel_30s_avg": mu, "day_conc": dc,
            "pass_hc344": bool(dc < DAY_CONC_GATE),
            "total_ticks": float(realized.sum()),
            "queue_pos": "N/A (FIFO label realized)",
            "cancel_window": "N/A (FIFO label realized)"}


def percentile_verdict(name, side, pred_score, net_realized, filled_mask, date_idx_ho, top_pct):
    fi = np.where(filled_mask)[0]
    if len(fi) == 0:
        return None
    scores = pred_score[fi]
    cutoff = np.percentile(scores, 100 - top_pct)
    take_local = scores >= cutoff
    take = np.zeros_like(filled_mask)
    take[fi[take_local]] = True
    realized = net_realized[take]
    n = len(realized)
    if n == 0:
        return None
    mu = float(realized.mean())
    sd = float(realized.std(ddof=1)) if n > 1 else 0
    sharpe = mu / sd * np.sqrt(n) if sd > 0 else 0
    neg = realized[realized < 0]
    nsd = float(neg.std(ddof=1)) if len(neg) > 1 else 0
    sortino = mu / nsd * np.sqrt(n) if nsd > 0 else 0
    gwin = float(realized[realized > 0].sum())
    gloss = float(-realized[realized < 0].sum())
    pf = gwin / gloss if gloss > 0 else float("inf")
    wr = (realized > 0).sum() / max(1, ((realized > 0).sum() + (realized < 0).sum()))
    days = date_idx_ho[take]
    cnt = np.bincount(days) if len(days) > 0 else np.array([1])
    dc = float(cnt.max() / cnt.sum())
    return {"name": name, "side": side, "top_pct": top_pct,
            "n_fills": n, "ticks_per_fill": mu, "total_ticks": mu * n,
            "sharpe": float(sharpe), "sortino": float(sortino),
            "pf": float(pf), "wr": float(wr),
            "adv_sel_30s_avg": mu, "day_conc": dc,
            "pass_hc344": bool(dc < DAY_CONC_GATE),
            "queue_pos": "N/A (FIFO label realized)",
            "cancel_window": "N/A (FIFO label realized)"}


def run_full(npz_path, npz_sha256, label_tag, mlflow_uri, experiment):
    t0 = time.time()
    data = load_aligned_data(npz_path)
    X = data["X"]
    train_mask, hold_mask = time_split_per_day(data["date_idx"])
    n_train, n_hold = int(train_mask.sum()), int(hold_mask.sum())
    print(f"[split] train={n_train} hold={n_hold}")

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=f"lgbm_gate_{label_tag}_{int(time.time())}") as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}")
        mlflow.log_params({
            "source_npz": str(npz_path),
            "npz_sha256": npz_sha256,
            "label_tag": label_tag,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_train": n_train, "n_holdout": n_hold,
            "feature_set": "multi_head_30",
            "oot_dates": ",".join(OOT_DATES),
            "commission_ticks": COMMISSION_RT_TICKS,
            "spread_ticks": SPREAD_TICKS,
            "split_method": "per_day_last_20pct (HC #0 sliding, HC #393 holdout)",
            "label_source": "tp4sl3_long/short_net_ticks (FIFO market replay, HC #74)",
        })

        verdicts_thr = []
        verdicts_pct = []
        feature_imp_dict = {}

        for side, y_full, filled_full in [
            ("long", data["long_net"], data["long_filled"]),
            ("short", data["short_net"], data["short_filled"]),
        ]:
            print(f"\n=== {side.upper()} ===")
            X_tr = X[train_mask]; X_ho = X[hold_mask]
            y_tr = y_full[train_mask]; y_ho = y_full[hold_mask]
            filled_ho = filled_full[hold_mask]; date_ho = data["date_idx"][hold_mask]
            net_realized = y_ho - COMMISSION_RT_TICKS

            print(f"[train] multi-head LGBM ({X_tr.shape[1]} features)...")
            model = train_lgbm(X_tr, y_tr, X_ho)
            pred_mh = model.predict(X_ho)
            decision_mh = pred_mh - COMMISSION_RT_TICKS

            print(f"[train] single-head LGBM (pred_log_ret_1s only)...")
            single_idx = data["feature_names"].index("pred_log_ret_1s")
            model_s = train_lgbm(X_tr[:, single_idx:single_idx+1], y_tr, X_ho[:, single_idx:single_idx+1])
            pred_sh = model_s.predict(X_ho[:, single_idx:single_idx+1])
            decision_sh = pred_sh - COMMISSION_RT_TICKS

            raw_1s = X_ho[:, single_idx]
            if side == "short":
                raw_1s = -raw_1s

            # Threshold verdicts (multi-head, single-head)
            for thr in [0.0, 0.5, 1.0]:
                vmh = threshold_verdict(
                    f"multihead_lgbm_{side}", side, decision_mh,
                    net_realized, filled_ho, date_ho, thr)
                verdicts_thr.append(vmh)
                print(f"  MH thr={thr:.1f}: n={vmh['n_fills']} tpf={vmh['ticks_per_fill']:+.3f} "
                      f"sh={vmh['sharpe']:.2f} pf={vmh['pf']:.2f} wr={vmh['wr']:.1%}")
            vsh0 = threshold_verdict(
                f"singlehead_lgbm_{side}", side, decision_sh,
                net_realized, filled_ho, date_ho, 0.0)
            verdicts_thr.append(vsh0)
            print(f"  SH thr=0.0: n={vsh0['n_fills']} tpf={vsh0['ticks_per_fill']:+.3f} "
                  f"sh={vsh0['sharpe']:.2f} pf={vsh0['pf']:.2f} wr={vsh0['wr']:.1%}")

            # Rules baseline (HC #74)
            pred_1s_ho = X_ho[:, single_idx]
            if side == "long":
                rules_take = (pred_1s_ho > 0) & filled_ho
            else:
                rules_take = (pred_1s_ho < 0) & filled_ho
            realized_r = y_ho[rules_take] - COMMISSION_RT_TICKS
            if len(realized_r) > 0:
                mu = realized_r.mean()
                sd = realized_r.std(ddof=1) if len(realized_r) > 1 else 0
                sh_r = mu / sd * np.sqrt(len(realized_r)) if sd > 0 else 0
                neg_r = realized_r[realized_r < 0]
                nsd_r = neg_r.std(ddof=1) if len(neg_r) > 1 else 0
                so_r = mu / nsd_r * np.sqrt(len(realized_r)) if nsd_r > 0 else 0
                gwin = realized_r[realized_r > 0].sum()
                gloss = -realized_r[realized_r < 0].sum()
                pf_r = gwin / gloss if gloss > 0 else float("inf")
                wr_r = (realized_r > 0).sum() / max(1, ((realized_r > 0).sum() + (realized_r < 0).sum()))
                days_r = date_ho[rules_take]
                cnt = np.bincount(days_r) if len(days_r) > 0 else np.array([1])
                dc_r = cnt.max() / cnt.sum()
                vr = {"name": f"rules_baseline_{side}", "side": side, "threshold": 0.0,
                      "n_signals": int(rules_take.sum()), "n_fills": int(rules_take.sum()),
                      "ticks_per_signal": float(mu), "ticks_per_fill": float(mu),
                      "sharpe": float(sh_r), "sortino": float(so_r),
                      "pf": float(pf_r), "wr": float(wr_r),
                      "adv_sel_30s_avg": float(mu), "day_conc": float(dc_r),
                      "pass_hc344": bool(dc_r < DAY_CONC_GATE),
                      "total_ticks": float(realized_r.sum()),
                      "queue_pos": "N/A", "cancel_window": "N/A"}
                verdicts_thr.append(vr)
                print(f"  RULES: n={vr['n_fills']} tpf={vr['ticks_per_fill']:+.3f} "
                      f"sh={vr['sharpe']:.2f} pf={vr['pf']:.2f} wr={vr['wr']:.1%}")

            # Percentile verdicts (multi-head, single-head, raw)
            for top_pct in [50, 20, 10, 5, 2, 1]:
                vmh = percentile_verdict(
                    f"multihead_lgbm_{side}", side, pred_mh,
                    net_realized, filled_ho, date_ho, top_pct)
                if vmh: verdicts_pct.append(vmh)
                vsh = percentile_verdict(
                    f"singlehead_lgbm_{side}", side, pred_sh,
                    net_realized, filled_ho, date_ho, top_pct)
                if vsh: verdicts_pct.append(vsh)
                vraw = percentile_verdict(
                    f"raw_signal_{side}", side, raw_1s,
                    net_realized, filled_ho, date_ho, top_pct)
                if vraw: verdicts_pct.append(vraw)
                if vmh and vsh and vraw:
                    print(f"  top {top_pct}%: MH tpf={vmh['ticks_per_fill']:+.3f} pf={vmh['pf']:.2f}  |  "
                          f"SH tpf={vsh['ticks_per_fill']:+.3f} pf={vsh['pf']:.2f}  |  "
                          f"RAW tpf={vraw['ticks_per_fill']:+.3f} pf={vraw['pf']:.2f}")

            imp = model.feature_importances_
            ranked = sorted(zip(data["feature_names"], imp), key=lambda x: -x[1])
            feature_imp_dict[side] = [(n, int(v)) for n, v in ranked]
            print(f"  Top-5 features ({side}): {ranked[:5]}")

        # MLflow metrics
        for v in verdicts_thr:
            prefix = f"{v['name']}_thr{v['threshold']:.1f}"
            for k in ["n_signals", "n_fills", "ticks_per_signal", "ticks_per_fill",
                      "sharpe", "sortino", "pf", "wr", "adv_sel_30s_avg",
                      "day_conc", "pass_hc344", "total_ticks"]:
                val = v[k]
                if isinstance(val, bool): val = int(val)
                if val == float("inf"): val = 9999.0
                try:
                    mlflow.log_metric(f"{prefix}_{k}", float(val))
                except Exception:
                    pass

        # Best multi-head pct per side as flat metric
        for side in ("long", "short"):
            sub = [v for v in verdicts_pct if v["name"] == f"multihead_lgbm_{side}"]
            if sub:
                best = max(sub, key=lambda v: v["sharpe"])
                for k in ("n_fills", "ticks_per_fill", "sharpe", "sortino", "pf", "wr", "day_conc", "top_pct"):
                    try:
                        mlflow.log_metric(f"best_mh_pct_{side}_{k}", float(best[k]))
                    except Exception:
                        pass

        # Save artifacts
        df_thr = pd.DataFrame(verdicts_thr)
        df_pct = pd.DataFrame(verdicts_pct)
        thr_csv = OUT_DIR / f"lgbm_gate_{label_tag}_threshold_verdicts.csv"
        pct_csv = OUT_DIR / f"lgbm_gate_{label_tag}_percentile_verdicts.csv"
        df_thr.to_csv(thr_csv, index=False)
        df_pct.to_csv(pct_csv, index=False)
        mlflow.log_artifact(str(thr_csv))
        mlflow.log_artifact(str(pct_csv))

        imp_path = OUT_DIR / f"lgbm_gate_{label_tag}_feature_importance.json"
        with open(imp_path, "w") as f:
            json.dump(feature_imp_dict, f, indent=2, default=str)
        mlflow.log_artifact(str(imp_path))

        elapsed = time.time() - t0
        mlflow.log_metric("wall_time_seconds", elapsed)
        print(f"\n[done] wall={elapsed:.1f}s mlflow_run_id={run_id}")

        # Final summary tables
        print("\n" + "=" * 110)
        print("THRESHOLD VERDICTS")
        print(f"{'name':<32} {'side':<5} {'thr':>5} {'n_fills':>8} {'tpf':>8} {'sharpe':>7} "
              f"{'sortino':>8} {'pf':>6} {'wr':>6} {'day_conc':>8} {'hc344':>6}")
        print("=" * 110)
        for v in verdicts_thr:
            pf_s = f"{v['pf']:.2f}" if v['pf'] != float('inf') else "inf"
            print(f"{v['name']:<32} {v['side']:<5} {v['threshold']:>5.1f} "
                  f"{v['n_fills']:>8d} {v['ticks_per_fill']:>+8.3f} "
                  f"{v['sharpe']:>7.2f} {v['sortino']:>8.2f} {pf_s:>6} "
                  f"{v['wr']:>6.2%} {v['day_conc']:>8.3f} {str(v['pass_hc344']):>6}")

        print("\n" + "=" * 110)
        print("PERCENTILE VERDICTS")
        print(f"{'name':<28} {'side':<5} {'top%':>5} {'n_fills':>8} {'tpf':>8} {'sharpe':>8} "
              f"{'sortino':>8} {'pf':>6} {'wr':>7} {'day_conc':>9} {'hc344':>6}")
        print("=" * 110)
        for v in verdicts_pct:
            pf_s = f"{v['pf']:.2f}" if v['pf'] != float('inf') else "inf"
            print(f"{v['name']:<28} {v['side']:<5} {v['top_pct']:>5d} "
                  f"{v['n_fills']:>8d} {v['ticks_per_fill']:>+8.3f} "
                  f"{v['sharpe']:>8.2f} {v['sortino']:>8.2f} {pf_s:>6} "
                  f"{v['wr']:>6.2%} {v['day_conc']:>9.3f} {str(v['pass_hc344']):>6}")
        print("=" * 110)

        return verdicts_thr, verdicts_pct, run_id


def main():
    return run_full(
        npz_path=NPZ_PATH,
        npz_sha256=NPZ_SHA256,
        label_tag="v33_fold00",
        mlflow_uri="http://jupiter:5000",
        experiment="hc424_jupiter_exec_research_v33",
    )


if __name__ == "__main__":
    main()
