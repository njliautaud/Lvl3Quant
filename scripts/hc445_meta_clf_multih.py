#!/usr/bin/env python3
"""HC #445 — Meta-classifier v2 with MULTI-HORIZON FEATURES.

The HC #444 verdict was: pre-trade features (time, queue, signal density,
pred_strength alone) cannot separate profitable from unprofitable canonical
fills well enough to overcome cost. The verdict explicitly called for
"book imbalance at signal, recent realized vol, MULTI-MODEL AGREEMENT".

This script adds the *multi-horizon agreement* features that the v2 model
already emits (predictions at 1s, 5s, 10s horizons for the same signal
instant). It then re-trains the LGBM meta-classifier and re-runs the
threshold sweep with strict gating:

  TIME-ORDERED 70/30 OOS HOLDOUT:
    PASS if any prob threshold yields  n >= 50  AND  mean_tk > 0  AND  PF >= 1.10

Inputs:
  - Fills CSV under output/hc432_v342_47day_validation/<config>_fifo_fills.csv
  - Per-day v2 predictions at output/cnn_mamba_v2_bulk_oot_v2/<date>_predictions.npz
  - Per-day MBO event timestamps at data/processed/mbo_events_smart_v3/<date>_mbo_events.npz

Output:
  - output/hc445_meta_clf_multih/<config>/report.md
  - output/hc445_meta_clf_multih/<config>/oos_eval.json
  - output/hc445_meta_clf_multih/_SUMMARY.tsv  (rolling, appended)

Usage:
  python3 scripts/hc445_meta_clf_multih.py --config hc443_band_top5_short
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
FILLS_DIR = LVL3 / "output/hc432_v342_47day_validation"
V2_OOT_DIR = LVL3 / "output/cnn_mamba_v2_bulk_oot_v2"
MBO_EVENT_DIR = LVL3 / "data/processed/mbo_events_smart_v3"
OUT_DIR = LVL3 / "output/hc445_meta_clf_multih"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376  # AMP — already baked into net_ticks of FIFO replay


def load_v2_day(date_str: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return (preds_1s_5s_10s [N,3], mbo_timestamps [M], window, stride)."""
    p = V2_OOT_DIR / f"{date_str}_predictions.npz"
    d = np.load(p, allow_pickle=False)
    preds = d["predictions"][:, :3].astype(np.float64)  # 1s, 5s, 10s
    window = int(d["window_size"])
    stride = int(d["stride"])

    mp = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    mbo = np.load(mp, allow_pickle=False)
    ts = mbo["timestamps"].astype(np.int64)
    return preds, ts, window, stride


def signal_ts_to_idx(ts_signal_ns: np.ndarray, mbo_ts: np.ndarray,
                     window: int, stride: int) -> np.ndarray:
    """Map ts_signal_ns -> idx_in_day in the predictions array.

    Inverse of: event_idx = idx_in_day * stride + window - 1
                ts_signal_ns = mbo_ts[event_idx]
    """
    # Find event_idx for each ts_signal via searchsorted on mbo_ts.
    pos = np.searchsorted(mbo_ts, ts_signal_ns, side="left")
    # Clamp & check exact match (the FIFO engine uses these exact ts already).
    pos = np.clip(pos, 0, len(mbo_ts) - 1)
    # Some signals may not be exact; pick nearest of left/right.
    left_diff = np.abs(mbo_ts[np.clip(pos - 1, 0, len(mbo_ts) - 1)] - ts_signal_ns)
    right_diff = np.abs(mbo_ts[pos] - ts_signal_ns)
    use_left = left_diff < right_diff
    event_idx = np.where(use_left, pos - 1, pos)
    # Recover idx_in_day.
    idx_in_day = (event_idx - (window - 1)) // stride
    return idx_in_day


def extract_multih_features(fills: pd.DataFrame) -> pd.DataFrame:
    """Add pred_1s, pred_5s, pred_10s + engineered multi-h features to each fill."""
    dates = sorted(fills["date"].astype(str).unique())
    rows_out: List[Dict] = []
    n_dropped = 0
    for d in dates:
        sub = fills[fills["date"].astype(str) == d].copy()
        try:
            preds, mbo_ts, window, stride = load_v2_day(d)
        except FileNotFoundError:
            n_dropped += len(sub)
            continue
        ts_sig = sub["ts_signal_ns"].to_numpy(dtype=np.int64)
        idx_in_day = signal_ts_to_idx(ts_sig, mbo_ts, window, stride)
        valid = (idx_in_day >= 0) & (idx_in_day < preds.shape[0])
        sub = sub.loc[valid].copy()
        idx_v = idx_in_day[valid]
        sub["pred_1s"] = preds[idx_v, 0]
        sub["pred_5s"] = preds[idx_v, 1]
        sub["pred_10s"] = preds[idx_v, 2]
        rows_out.append(sub)
    out = pd.concat(rows_out, ignore_index=True)
    print(f"  multi-h feature attach: kept {len(out)}/{len(fills)} fills"
          f"  ({n_dropped} from missing-pred days)")

    # Engineered features
    out["pred_1s_5s_diff"] = out["pred_1s"] - out["pred_5s"]
    out["pred_5s_10s_diff"] = out["pred_5s"] - out["pred_10s"]
    out["pred_abs_1s"] = np.abs(out["pred_1s"])
    out["pred_abs_5s"] = np.abs(out["pred_5s"])
    out["pred_abs_10s"] = np.abs(out["pred_10s"])
    # Sign-agreement: 1 if all three horizons agree on sign with the trade direction
    is_short = (out["direction"] == "short").astype(int).to_numpy()
    sign1 = np.sign(out["pred_1s"]).to_numpy()
    sign5 = np.sign(out["pred_5s"]).to_numpy()
    sign10 = np.sign(out["pred_10s"]).to_numpy()
    # For shorts, want all negative. For longs, want all positive.
    want = np.where(is_short, -1.0, 1.0)
    out["sign_agree_1_5"] = ((sign1 == want) & (sign5 == want)).astype(int)
    out["sign_agree_1_10"] = ((sign1 == want) & (sign10 == want)).astype(int)
    out["sign_agree_all3"] = ((sign1 == want) & (sign5 == want) & (sign10 == want)).astype(int)
    # Magnitude consistency: how flat is the term structure?
    out["pred_h_range"] = out[["pred_1s", "pred_5s", "pred_10s"]].max(axis=1) \
                         - out[["pred_1s", "pred_5s", "pred_10s"]].min(axis=1)
    # Decay slope: is the signal decaying fast (1s big, 10s small) or persisting?
    out["pred_decay_1_10"] = out["pred_abs_1s"] - out["pred_abs_10s"]
    # Within-day z-scores of each horizon's prediction magnitude
    g = out.groupby("date")
    for h in ("1s", "5s", "10s"):
        mu = g[f"pred_abs_{h}"].transform("mean")
        sd = g[f"pred_abs_{h}"].transform("std").replace(0, 1.0)
        out[f"pred_abs_{h}_z_in_day"] = (out[f"pred_abs_{h}"] - mu) / sd
    return out


def build_design_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return X (features) and y (binary: net_ticks > 0)."""
    df = df.copy()
    # Time features
    ts = pd.to_datetime(df["ts_signal_ns"], unit="ns", utc=True).dt.tz_convert("US/Eastern")
    df["hour"] = ts.dt.hour
    df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    df["dow"] = ts.dt.dayofweek
    df["is_open_30m"] = ((df["minute_of_day"] >= 9 * 60 + 30) &
                         (df["minute_of_day"] < 10 * 60 + 0)).astype(int)
    df["is_close_30m"] = ((df["minute_of_day"] >= 15 * 60 + 30) &
                          (df["minute_of_day"] < 16 * 60 + 0)).astype(int)
    df["is_lunch"] = ((df["minute_of_day"] >= 12 * 60) &
                       (df["minute_of_day"] < 13 * 60)).astype(int)
    df["dir_short"] = (df["direction"] == "short").astype(int)
    df["queue_ahead_log1p"] = np.log1p(df["queue_ahead"].clip(lower=0))
    # Signal density: count of signals in same minute-of-day across day
    df["signal_density"] = df.groupby(["date", "minute_of_day"])["ts_signal_ns"].transform("count")

    feature_cols = [
        # original meta-clf v1 features
        "pred_strength", "queue_ahead_log1p",
        "hour", "minute_of_day", "dow",
        "is_open_30m", "is_close_30m", "is_lunch",
        "signal_density", "dir_short",
        # multi-h enrichment
        "pred_1s", "pred_5s", "pred_10s",
        "pred_1s_5s_diff", "pred_5s_10s_diff",
        "pred_abs_1s", "pred_abs_5s", "pred_abs_10s",
        "sign_agree_1_5", "sign_agree_1_10", "sign_agree_all3",
        "pred_h_range", "pred_decay_1_10",
        "pred_abs_1s_z_in_day", "pred_abs_5s_z_in_day", "pred_abs_10s_z_in_day",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    y = (df["net_ticks"] > 0).astype(int).to_numpy()
    return X, y


def time_ordered_split(df: pd.DataFrame, train_frac: float = 0.70):
    """Time-ordered: oldest 70% of unique dates train, newest 30% test."""
    dates = sorted(df["date"].astype(str).unique())
    n_train_d = int(len(dates) * train_frac)
    train_dates = set(dates[:n_train_d])
    test_dates = set(dates[n_train_d:])
    return train_dates, test_dates


def threshold_sweep(probs: np.ndarray, net_ticks: np.ndarray,
                    thresholds: List[float]) -> List[Dict]:
    rows = []
    for thr in thresholds:
        mask = probs >= thr
        n = int(mask.sum())
        if n < 5:
            rows.append({"thr": thr, "n": n, "mean_tk": None, "PF": None,
                         "WR": None, "sharpe": None})
            continue
        sub = net_ticks[mask]
        gains = sub[sub > 0].sum()
        losses = -sub[sub < 0].sum()
        pf = float(gains / losses) if losses > 0 else float("inf")
        sharpe = float((sub.mean() / sub.std() * np.sqrt(n))) if sub.std() > 0 else 0.0
        rows.append({
            "thr": float(thr), "n": n,
            "mean_tk": float(sub.mean()),
            "PF": pf,
            "WR": float((sub > 0).mean() * 100.0),
            "sharpe": sharpe,
        })
    return rows


def run_config(config: str) -> Dict:
    fills_p = FILLS_DIR / f"{config}_fifo_fills.csv"
    if not fills_p.exists():
        return {"config": config, "error": f"missing {fills_p}"}
    fills = pd.read_csv(fills_p, dtype={"date": str})
    print(f"[{config}] loaded {len(fills)} fills across {fills['date'].nunique()} dates")
    if len(fills) < 200:
        return {"config": config, "error": "too few fills (<200)"}

    df = extract_multih_features(fills)
    if len(df) < 200:
        return {"config": config, "error": "post-attach too few fills"}

    X, y = build_design_matrix(df)
    df = df.reset_index(drop=True)
    X = X.reset_index(drop=True)

    train_dates, test_dates = time_ordered_split(df, 0.70)
    train_mask = df["date"].astype(str).isin(train_dates).to_numpy()
    test_mask = df["date"].astype(str).isin(test_dates).to_numpy()
    print(f"  train: {train_mask.sum()} ({len(train_dates)} dates),"
          f"  test: {test_mask.sum()} ({len(test_dates)} dates)")
    print(f"  test base positive rate: {y[test_mask].mean():.4f}")

    try:
        import lightgbm as lgb
    except ImportError:
        return {"config": config, "error": "lightgbm not installed"}

    dtrain = lgb.Dataset(X.iloc[train_mask], label=y[train_mask])
    dtest = lgb.Dataset(X.iloc[test_mask], label=y[test_mask], reference=dtrain)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "verbose": -1,
    }
    model = lgb.train(
        params, dtrain,
        num_boost_round=400,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    test_probs = model.predict(X.iloc[test_mask])
    test_net = df.loc[test_mask, "net_ticks"].to_numpy()
    from sklearn.metrics import roc_auc_score
    auc_oos = float(roc_auc_score(y[test_mask], test_probs))

    sweep = threshold_sweep(test_probs, test_net,
                            [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])

    # Verdict gate (HC #444 R3 carried forward)
    passes = [r for r in sweep
              if r["mean_tk"] is not None
              and r["n"] >= 50
              and r["mean_tk"] > 0
              and r["PF"] is not None and r["PF"] >= 1.10]
    verdict_pass = len(passes) > 0
    best = max(passes, key=lambda r: r["sharpe"]) if passes else None

    # Baseline (no-filter) on test set
    base_n = int(test_mask.sum())
    base_mean = float(test_net.mean())
    base_pf = float(test_net[test_net > 0].sum() / max(1e-9, -test_net[test_net < 0].sum()))
    base_wr = float((test_net > 0).mean() * 100.0)

    fi = sorted(zip(X.columns.tolist(), model.feature_importance(importance_type="gain").tolist()),
                key=lambda x: -x[1])

    out = {
        "config": config,
        "n_total": int(len(df)),
        "n_dates": int(df["date"].nunique()),
        "train": {"n": int(train_mask.sum()), "n_dates": len(train_dates)},
        "test": {"n": base_n, "n_dates": len(test_dates),
                  "baseline_mean_tk": base_mean, "baseline_PF": base_pf, "baseline_WR": base_wr},
        "auc_oos": auc_oos,
        "best_iter": int(model.best_iteration),
        "threshold_sweep": sweep,
        "verdict_pass": verdict_pass,
        "best_threshold": best,
        "feature_importance_gain_top10": fi[:10],
    }

    # Write outputs
    cfg_out = OUT_DIR / config
    cfg_out.mkdir(parents=True, exist_ok=True)
    with open(cfg_out / "oos_eval.json", "w") as f:
        json.dump(out, f, indent=2)

    with open(cfg_out / "report.md", "w") as f:
        f.write(f"# HC #445 Meta-Classifier v2 (multi-h) — {config}\n\n")
        f.write(f"Fills: **{out['n_total']}** across **{out['n_dates']}** dates.\n\n")
        f.write(f"## OOS time-ordered split\n")
        f.write(f"- train: {out['train']['n']} fills ({out['train']['n_dates']} dates)\n")
        f.write(f"- test: {out['test']['n']} fills ({out['test']['n_dates']} dates)\n")
        f.write(f"- test baseline (no filter): mean_tk={base_mean:+.4f}  PF={base_pf:.3f}  WR={base_wr:.2f}%\n")
        f.write(f"- AUC_oos = **{auc_oos:.4f}**   (best_iter={model.best_iteration})\n\n")
        f.write("## Threshold sweep on OOS test\n\n")
        f.write("| thr | n | mean_tk | PF | WR | sharpe |\n|---|---:|---:|---:|---:|---:|\n")
        for r in sweep:
            if r["mean_tk"] is None:
                f.write(f"| {r['thr']:.2f} | {r['n']} | — | — | — | — |\n")
            else:
                f.write(f"| {r['thr']:.2f} | {r['n']} | {r['mean_tk']:+.4f} | "
                        f"{r['PF']:.3f} | {r['WR']:.2f}% | {r['sharpe']:+.3f} |\n")
        f.write("\n## Verdict\n")
        if verdict_pass:
            f.write(f"✅ **PASS** — threshold {best['thr']:.2f} gives n={best['n']}, "
                    f"mean_tk={best['mean_tk']:+.4f}, PF={best['PF']:.3f}, "
                    f"WR={best['WR']:.2f}%, sharpe={best['sharpe']:+.3f}.\n\n")
        else:
            f.write("❌ **FAIL** — no threshold yields n≥50, mean_tk>0, PF≥1.10 on time-ordered OOS test.\n\n")
        f.write("## Top-10 feature importance (gain)\n\n")
        for name, gain in fi[:10]:
            f.write(f"- `{name}`: {gain:.1f}\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="fills config name (matches <name>_fifo_fills.csv)")
    args = ap.parse_args()
    result = run_config(args.config)

    # Append to global summary
    summ_p = OUT_DIR / "_SUMMARY.tsv"
    if not summ_p.exists():
        with open(summ_p, "w") as f:
            f.write("config\tn_total\tn_dates\tauc_oos\tverdict_pass\tbest_thr\tbest_n\tbest_mean_tk\tbest_PF\tbase_mean_tk\tbase_PF\n")
    with open(summ_p, "a") as f:
        b = result.get("best_threshold") or {}
        t = result.get("test", {})
        f.write(f"{result.get('config','?')}\t{result.get('n_total','')}\t{result.get('n_dates','')}\t"
                f"{result.get('auc_oos','')}\t{result.get('verdict_pass','')}\t"
                f"{b.get('thr','')}\t{b.get('n','')}\t{b.get('mean_tk','')}\t{b.get('PF','')}\t"
                f"{t.get('baseline_mean_tk','')}\t{t.get('baseline_PF','')}\n")

    # Print headline to stdout
    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"\n=== {result['config']} ===")
    print(f"AUC_oos: {result['auc_oos']:.4f}")
    print(f"Verdict: {'PASS' if result['verdict_pass'] else 'FAIL'}")
    if result["best_threshold"]:
        b = result["best_threshold"]
        print(f"  best thr={b['thr']:.2f}  n={b['n']}  mean_tk={b['mean_tk']:+.4f}  "
              f"PF={b['PF']:.3f}  WR={b['WR']:.2f}%  sharpe={b['sharpe']:+.3f}")


if __name__ == "__main__":
    main()
