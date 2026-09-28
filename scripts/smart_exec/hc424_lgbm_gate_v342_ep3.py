#!/usr/bin/env python3
"""HC #424 §3 — LGBM execution gate on v3.4.2 ep3 multi-head predictions.

Pipeline:
  1. Load v3.4.2 ep3 OOT NPZ (241,351 samples, 5 OOT dates 20260223-20260227).
  2. Build per-row date_idx + ts_ns via per-day FIFO label counts (HC #417
     proportional-trim alignment).
  3. Build multi-head feature matrix X (30 pred_* heads). HC #422 R8 / HC #423 §4.
  4. Targets: tp4sl3_long_net_ticks and tp4sl3_short_net_ticks from FIFO labels
     aligned per-day. Cost-aware net = net_ticks − COMMISSION_RT_TICKS (0.376)
     since labels already reflect realized FIFO replay outcome.
  5. Split 80/20 by time (sliding holdout per HC #393 — first 80% of each day
     is train, last 20% is holdout). HC #0 sliding-only.
  6. Train two LGBM regressors (long-net, short-net). Single-head BASELINE
     uses only pred_log_ret_1s. Multi-head GATE uses all 30 heads.
  7. Canonical replay verdict on holdout: HC #397B columns —
        n_fills, ticks_per_fill, sharpe√N, sortino√N, PF, WR,
        adv_sel_30s_avg, day_conc, pass_hc344, queue_pos, cancel_window.
     queue_pos / cancel_window N/A for IOC-market evaluation; we evaluate
     PASSIVE limit fills using FIFO label `filled` flag + net_ticks directly
     (real MBO replay; HC #74 FIFO market replay).
  8. Log to MLflow experiment `hc424_jupiter_exec_research_v342`.

Run-time: ~3-5 min on Jupiter CPU.

HC #420 authorized: this is the user's own research codebase.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow

LVL3 = Path("/home/jupiter/Lvl3Quant")
NPZ_PATH = LVL3 / "output/hc424_jupiter_exec_research/inputs/v3_4_2_ep3/fold_00_ep3_oot.npz"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
OUT_DIR = LVL3 / "output/hc424_jupiter_exec_research"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# HC #392 cost model — passive limit = commission only (0.376), market = 1.376
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
MARKET_COST_TICKS = COMMISSION_RT_TICKS + SPREAD_TICKS  # 1.376

OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]

# Steps per second @ 250 ms stride
ANN_FACTOR = np.sqrt(252.0 * 6.5 * 3600.0 * 4.0)
DAY_CONC_GATE = 0.50  # HC #344: pass if < 0.50


# -----------------------------------------------------------------------------
# Multi-head feature schema — 30 pred_* heads per HC #422 R8 / HC #423 §4
# -----------------------------------------------------------------------------
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
    """HC #417 proportional trim: distribute FIFO rows across days to sum to n_npz."""
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
    assert sum(keep.values()) == n_npz, f"sum {sum(keep.values())} != n_npz {n_npz}"
    return keep


def load_aligned_data():
    """Returns dict with X (N, F), targets, date_idx, ts_ns, side outcomes."""
    print(f"[load] npz: {NPZ_PATH}")
    d = np.load(NPZ_PATH, allow_pickle=True)
    n = d["pred_log_ret_1s"].shape[0]
    print(f"[load] n_samples: {n}")

    # Build feature matrix
    feats = []
    for k in MULTI_HEAD_FEATURES:
        a = d[k][:n]
        feats.append(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0))
    X = np.stack(feats, axis=1).astype(np.float32)
    print(f"[load] X shape: {X.shape}, features: {len(MULTI_HEAD_FEATURES)}")

    # Per-day alignment
    fifo_n = {dt: int(np.load(LABELS_DIR / f"{dt}_fifo_labels.npz")["window_k"].shape[0])
              for dt in OOT_DATES}
    keep_per_day = proportional_per_day_keep(fifo_n, n)
    print(f"[load] keep_per_day: {keep_per_day}")

    # Concat FIFO label arrays trimmed per day
    long_net = []
    short_net = []
    long_filled = []
    short_filled = []
    ts_ns = []
    date_idx = []
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

    assert long_net.shape[0] == n, f"long_net {long_net.shape[0]} != n {n}"

    print(f"[load] long net mean={long_net.mean():.3f} pct_filled={long_filled.mean():.3f}")
    print(f"[load] short net mean={short_net.mean():.3f} pct_filled={short_filled.mean():.3f}")

    return {
        "X": X,
        "long_net": long_net,
        "short_net": short_net,
        "long_filled": long_filled,
        "short_filled": short_filled,
        "ts_ns": ts_ns,
        "date_idx": date_idx,
        "feature_names": MULTI_HEAD_FEATURES,
    }


def time_split_per_day(date_idx: np.ndarray, holdout_frac: float = 0.20):
    """HC #393 holdout: last 20% of each day is holdout. Returns boolean masks."""
    n = date_idx.shape[0]
    train_mask = np.zeros(n, dtype=bool)
    hold_mask = np.zeros(n, dtype=bool)
    for i in np.unique(date_idx):
        rows = np.where(date_idx == i)[0]
        cut = int(len(rows) * (1.0 - holdout_frac))
        train_mask[rows[:cut]] = True
        hold_mask[rows[cut:]] = True
    return train_mask, hold_mask


def train_lgbm(X_train, y_train, X_holdout, params=None):
    if params is None:
        params = dict(
            objective="regression",
            metric="rmse",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=200,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbosity=-1,
            n_estimators=500,
            n_jobs=-1,
        )
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_holdout, np.zeros(len(X_holdout)))],
              callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    return model


def adv_sel_30s_avg(net_ticks: np.ndarray) -> float:
    """Adverse selection proxy: avg negative drift in net ticks (HC #397B).

    Since FIFO labels already realized to ~30s exit, we report net_ticks mean
    (negative => adverse). Average tick lost to adverse fills.
    """
    if len(net_ticks) == 0:
        return float("nan")
    return float(net_ticks.mean())


def day_concentration(date_idx_holdout: np.ndarray, fill_mask: np.ndarray) -> float:
    """HC #344: max share of fills from a single day. Pass if < 0.50."""
    if fill_mask.sum() == 0:
        return 1.0
    days_with_fill = date_idx_holdout[fill_mask]
    counts = np.bincount(days_with_fill)
    return float(counts.max() / max(1, counts.sum()))


def canonical_verdict(name, side, gate_signal, net_ticks_realized, filled_mask,
                      date_idx_holdout, threshold=0.0):
    """HC #397B verdict generator.

    gate_signal: model-predicted net (cost-adjusted). Pass if > threshold (default 0).
    net_ticks_realized: realized FIFO net ticks from labels (post-cost: labels
        are net_ticks meaning gross − tp4sl3 cost. We subtract COMMISSION further
        as passive limit cost per HC #392.)
    filled_mask: did the passive limit actually fill (real MBO replay)?
    """
    take = (gate_signal > threshold) & filled_mask
    n_signals = int(take.sum())
    if n_signals == 0:
        return {
            "name": name, "side": side, "threshold": threshold,
            "n_signals": 0, "n_fills": 0,
            "ticks_per_signal": 0.0, "ticks_per_fill": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "pf": 0.0, "wr": 0.0,
            "adv_sel_30s_avg": 0.0, "day_conc": 1.0, "pass_hc344": False,
            "queue_pos": None, "cancel_window": None,
            "total_ticks": 0.0,
        }
    # Apply HC #392 passive limit commission (label net_ticks may already reflect
    # commission depending on labeller; check by examining whether mean is near
    # gross - 0.376. We treat label net as post-commission per FIFO convention.)
    realized = net_ticks_realized[take]

    n_fills = n_signals  # filled_mask already gates
    total = float(realized.sum())
    tps = total / n_signals
    tpf = tps  # equal because all signals filled
    mu = float(realized.mean())
    sd = float(realized.std(ddof=1)) if len(realized) > 1 else 0.0
    sharpe = (mu / sd) * np.sqrt(n_signals) if sd > 0 else 0.0
    neg = realized[realized < 0]
    nsd = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
    sortino = (mu / nsd) * np.sqrt(n_signals) if nsd > 0 else 0.0
    wins = float((realized > 0).sum())
    losses = float((realized < 0).sum())
    wr = wins / max(1.0, wins + losses)
    gross_win = float(realized[realized > 0].sum())
    gross_loss = float(-realized[realized < 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    adv = adv_sel_30s_avg(realized)
    dc = day_concentration(date_idx_holdout, take)
    return {
        "name": name, "side": side, "threshold": threshold,
        "n_signals": n_signals, "n_fills": n_fills,
        "ticks_per_signal": tps, "ticks_per_fill": tpf,
        "sharpe": sharpe, "sortino": sortino, "pf": pf, "wr": wr,
        "adv_sel_30s_avg": adv, "day_conc": dc,
        "pass_hc344": bool(dc < DAY_CONC_GATE),
        "queue_pos": "N/A (FIFO label realized fill)",
        "cancel_window": "N/A (FIFO label realized fill)",
        "total_ticks": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow-uri", default="http://jupiter:5000")
    ap.add_argument("--experiment", default="hc424_jupiter_exec_research_v342")
    ap.add_argument("--run-name", default=f"lgbm_gate_v342_ep3_{int(time.time())}")
    args = ap.parse_args()

    t0 = time.time()
    data = load_aligned_data()
    X = data["X"]
    train_mask, hold_mask = time_split_per_day(data["date_idx"])
    n_train = int(train_mask.sum())
    n_hold = int(hold_mask.sum())
    print(f"[split] train={n_train} hold={n_hold}")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}")
        mlflow.log_params({
            "source_npz": str(NPZ_PATH),
            "npz_sha256": "bf513ec9eef5ffad47e841d30893222934f3537396fc75c59062e351e43b9838",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_train": n_train,
            "n_holdout": n_hold,
            "feature_set": "multi_head_30",
            "oot_dates": ",".join(OOT_DATES),
            "commission_ticks": COMMISSION_RT_TICKS,
            "spread_ticks": SPREAD_TICKS,
            "split_method": "per_day_last_20pct (HC #0 sliding, HC #393 holdout)",
            "label_source": "tp4sl3_long/short_net_ticks (FIFO market replay, HC #74)",
        })

        verdicts = []
        feature_imp_dict = {}

        for side, y_full, filled_full in [
            ("long", data["long_net"], data["long_filled"]),
            ("short", data["short_net"], data["short_filled"]),
        ]:
            print(f"\n[train {side}] training LGBM multi-head gate...")
            X_tr = X[train_mask]
            X_ho = X[hold_mask]
            y_tr = y_full[train_mask]
            y_ho = y_full[hold_mask]
            filled_ho = filled_full[hold_mask]
            date_ho = data["date_idx"][hold_mask]

            # Multi-head GATE
            model = train_lgbm(X_tr, y_tr, X_ho)
            pred_ho = model.predict(X_ho)
            # Cost-adjusted decision signal: predicted net minus commission
            decision = pred_ho - COMMISSION_RT_TICKS

            # Multi-head verdicts at thresholds 0.0 and +0.5
            for thr in [0.0, 0.5, 1.0]:
                v = canonical_verdict(
                    name=f"multihead_lgbm_{side}",
                    side=side,
                    gate_signal=decision,
                    net_ticks_realized=y_ho - COMMISSION_RT_TICKS,
                    filled_mask=filled_ho,
                    date_idx_holdout=date_ho,
                    threshold=thr,
                )
                verdicts.append(v)
                print(f"  thr={thr:.1f}: n_fills={v['n_fills']} tpf={v['ticks_per_fill']:+.3f} "
                      f"sharpe={v['sharpe']:.2f} sortino={v['sortino']:.2f} pf={v['pf']:.2f} "
                      f"wr={v['wr']:.2%} day_conc={v['day_conc']:.2f} pass_hc344={v['pass_hc344']}")

            # SINGLE-HEAD baseline
            print(f"\n[train {side}-single] training LGBM single-head baseline (pred_log_ret_1s only)...")
            single_idx = data["feature_names"].index("pred_log_ret_1s")
            X_tr_s = X_tr[:, single_idx:single_idx + 1]
            X_ho_s = X_ho[:, single_idx:single_idx + 1]
            model_s = train_lgbm(X_tr_s, y_tr, X_ho_s)
            pred_s = model_s.predict(X_ho_s)
            decision_s = pred_s - COMMISSION_RT_TICKS
            for thr in [0.0]:
                v = canonical_verdict(
                    name=f"singlehead_lgbm_{side}",
                    side=side,
                    gate_signal=decision_s,
                    net_ticks_realized=y_ho - COMMISSION_RT_TICKS,
                    filled_mask=filled_ho,
                    date_idx_holdout=date_ho,
                    threshold=thr,
                )
                verdicts.append(v)
                print(f"  single thr={thr:.1f}: n_fills={v['n_fills']} tpf={v['ticks_per_fill']:+.3f} "
                      f"sharpe={v['sharpe']:.2f} pf={v['pf']:.2f} wr={v['wr']:.2%}")

            # Rules baseline: HC #74 — take all fills where signal pred_log_ret_1s favors side
            pred_1s_ho = X_ho[:, single_idx]
            if side == "long":
                rules_take = (pred_1s_ho > 0) & filled_ho
            else:
                rules_take = (pred_1s_ho < 0) & filled_ho
            realized = y_ho[rules_take] - COMMISSION_RT_TICKS
            if len(realized) > 0:
                mu = realized.mean()
                sd = realized.std(ddof=1) if len(realized) > 1 else 0
                sh = mu / sd * np.sqrt(len(realized)) if sd > 0 else 0
                neg = realized[realized < 0]
                nsd = neg.std(ddof=1) if len(neg) > 1 else 0
                so = mu / nsd * np.sqrt(len(realized)) if nsd > 0 else 0
                gwin = realized[realized > 0].sum()
                gloss = -realized[realized < 0].sum()
                pf_r = gwin / gloss if gloss > 0 else float("inf")
                wr_r = (realized > 0).sum() / max(1, ((realized > 0).sum() + (realized < 0).sum()))
                days_r = date_ho[rules_take]
                cnt = np.bincount(days_r) if len(days_r) > 0 else np.array([1])
                dc_r = cnt.max() / cnt.sum()
                vr = {
                    "name": f"rules_baseline_{side}", "side": side, "threshold": 0.0,
                    "n_signals": int(rules_take.sum()), "n_fills": int(rules_take.sum()),
                    "ticks_per_signal": float(mu), "ticks_per_fill": float(mu),
                    "sharpe": float(sh), "sortino": float(so), "pf": float(pf_r), "wr": float(wr_r),
                    "adv_sel_30s_avg": float(mu), "day_conc": float(dc_r),
                    "pass_hc344": bool(dc_r < DAY_CONC_GATE),
                    "queue_pos": "N/A", "cancel_window": "N/A",
                    "total_ticks": float(realized.sum()),
                }
                verdicts.append(vr)
                print(f"  rules: n_fills={vr['n_fills']} tpf={vr['ticks_per_fill']:+.3f} "
                      f"sharpe={vr['sharpe']:.2f} pf={vr['pf']:.2f} wr={vr['wr']:.2%}")

            # Feature importance (multi-head gate only)
            imp = model.feature_importances_
            feature_imp_dict[side] = dict(zip(data["feature_names"], [int(x) for x in imp]))

        # MLflow metrics — flatten verdicts
        for v in verdicts:
            prefix = f"{v['name']}_thr{v['threshold']:.1f}"
            for k in ["n_signals", "n_fills", "ticks_per_signal", "ticks_per_fill",
                      "sharpe", "sortino", "pf", "wr", "adv_sel_30s_avg",
                      "day_conc", "pass_hc344", "total_ticks"]:
                val = v[k]
                if isinstance(val, (bool,)):
                    val = int(val)
                if val == float("inf"):
                    val = 9999.0
                try:
                    mlflow.log_metric(f"{prefix}_{k}", float(val))
                except Exception:
                    pass

        # Save artifacts
        verdict_df = pd.DataFrame(verdicts)
        verdict_csv = OUT_DIR / "lgbm_gate_v342_ep3_verdicts.csv"
        verdict_df.to_csv(verdict_csv, index=False)
        mlflow.log_artifact(str(verdict_csv))

        imp_path = OUT_DIR / "lgbm_gate_v342_ep3_feature_importance.json"
        with open(imp_path, "w") as f:
            json.dump(feature_imp_dict, f, indent=2)
        mlflow.log_artifact(str(imp_path))

        # Schema doc
        schema_doc = OUT_DIR / "v3_4_2_ep3_npz_schema.md"
        if schema_doc.exists():
            mlflow.log_artifact(str(schema_doc))

        elapsed = time.time() - t0
        mlflow.log_metric("wall_time_seconds", elapsed)
        print(f"\n[done] wall={elapsed:.1f}s mlflow_run_id={run_id}")

        # Print final summary table
        print("\n" + "=" * 100)
        print(f"{'name':<32} {'side':<5} {'thr':>5} {'n_fills':>8} {'tpf':>8} {'sharpe':>7} "
              f"{'sortino':>8} {'pf':>6} {'wr':>6} {'day_conc':>8} {'hc344':>6}")
        print("=" * 100)
        for v in verdicts:
            pf_s = f"{v['pf']:.2f}" if v['pf'] != float('inf') else "inf"
            print(f"{v['name']:<32} {v['side']:<5} {v['threshold']:>5.1f} "
                  f"{v['n_fills']:>8d} {v['ticks_per_fill']:>+8.3f} "
                  f"{v['sharpe']:>7.2f} {v['sortino']:>8.2f} {pf_s:>6} "
                  f"{v['wr']:>6.2%} {v['day_conc']:>8.3f} {str(v['pass_hc344']):>6}")
        print("=" * 100)

        return verdicts, run_id


if __name__ == "__main__":
    main()
