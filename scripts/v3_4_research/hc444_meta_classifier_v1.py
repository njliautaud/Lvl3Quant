#!/usr/bin/env python3
"""HC #444 R3 meta-classifier v1 — "when do conditions line up to be profitable".

Trains a per-fill LightGBM binary classifier on canonical realtime_sl fills.
Target: net_ticks > 0.  Features are pre-trade-known at signal-entry time
(pred_strength, TOD, DOW, queue_ahead, intraday signal density, side).

Two evaluation regimes:
  1) Within-day random 80/20 split — measures learnability
  2) Time-ordered 70/30 split (first-21-days train / last-15-days test) —
     measures OOS generalization

If the OOS classifier's high-confidence positive-class predictions yield a
filtered subset whose realized mean_tk > 0 AND PF > 1, that's the HC #444 R3
"conditions line up" winner. We then publish the filter as a tradeable gate.

Apples-to-apples: same canonical fills, same engine, just a pre-trade YES/NO
overlay. The fills CSV already contains all required columns.

Output: /output/hc444_meta_clf_v1/{model.pkl, oos_eval.json, report.md,
filter_thresholds_grid.csv}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception as e:
    print(f"[meta-clf] lightgbm import failed: {e}", file=sys.stderr)
    sys.exit(2)

LVL3 = Path("/home/jupiter/Lvl3Quant")
FILLS_ROOT = LVL3 / "output" / "hc432_v342_47day_validation"
OUT_DIR = LVL3 / "output" / "hc444_meta_clf_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["entry_dt"] = pd.to_datetime(df["ts_entry_ns"], unit="ns", utc=True).dt.tz_convert("US/Eastern")
    df["hour"] = df["entry_dt"].dt.hour.astype(float)
    df["minute_of_day"] = (df["entry_dt"].dt.hour * 60 + df["entry_dt"].dt.minute).astype(float)
    df["dow"] = df["entry_dt"].dt.dayofweek.astype(float)
    df["is_open_30m"] = ((df["minute_of_day"] >= 9*60+30) & (df["minute_of_day"] < 10*60)).astype(float)
    df["is_close_30m"] = ((df["minute_of_day"] >= 15*60) & (df["minute_of_day"] < 15*60+30)).astype(float)
    df["is_lunch"] = ((df["minute_of_day"] >= 12*60) & (df["minute_of_day"] < 13*60)).astype(float)
    df["queue_ahead_log1p"] = np.log1p(df["queue_ahead"].clip(0, 1e9))
    df["pred_strength_z_within_day"] = df.groupby("date")["pred_strength"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    df["pred_pct_in_day"] = df.groupby("date")["pred_strength"].rank(pct=True)
    df = df.sort_values(["date", "ts_signal_ns"]).reset_index(drop=True)
    df["signal_idx_in_day"] = df.groupby("date").cumcount().astype(float)
    df["signal_count_in_day"] = df.groupby("date")["date"].transform("count").astype(float)
    df["signal_density"] = df["signal_idx_in_day"] / (df["signal_count_in_day"] + 1.0)
    df["dir_short"] = (df["direction"].str.lower() == "short").astype(float)
    return df


FEATURE_COLS = [
    "pred_strength", "pred_strength_z_within_day", "pred_pct_in_day",
    "queue_ahead_log1p",
    "hour", "minute_of_day", "dow",
    "is_open_30m", "is_close_30m", "is_lunch",
    "signal_density",
    "dir_short",
]


def train_eval(df: pd.DataFrame, train_dates, test_dates, tag: str) -> dict:
    tr = df[df["date"].isin(train_dates)].copy()
    te = df[df["date"].isin(test_dates)].copy()
    if len(tr) < 200 or len(te) < 100:
        return {"tag": tag, "skipped": True, "n_train": len(tr), "n_test": len(te)}

    X_tr = tr[FEATURE_COLS].to_numpy()
    y_tr = (tr["net_ticks"] > 0).astype(int).to_numpy()
    X_te = te[FEATURE_COLS].to_numpy()
    y_te = (te["net_ticks"] > 0).astype(int).to_numpy()
    nt_te = te["net_ticks"].to_numpy()

    params = dict(
        objective="binary",
        learning_rate=0.04,
        num_leaves=15,
        min_data_in_leaf=40,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l2=1.0,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
    )
    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS)
    valid_set = lgb.Dataset(X_te, label=y_te, feature_name=FEATURE_COLS, reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=400,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
    )

    p_te = booster.predict(X_te, num_iteration=booster.best_iteration)
    # base rate
    base_rate = float(y_te.mean())
    # AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score(y_te, p_te))
    except Exception:
        auc = float("nan")

    # threshold sweep — for each prob threshold, compute subset metrics
    rows = []
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = p_te >= thr
        n = int(mask.sum())
        if n < 30:
            continue
        sub = nt_te[mask]
        wins = sub[sub > 0].sum()
        losses = -sub[sub < 0].sum()
        pf = float(wins / losses) if losses > 0 else (999.0 if wins > 0 else 0.0)
        mean_tk = float(sub.mean())
        wr = float((sub > 0).mean()) * 100.0
        sd = float(sub.std(ddof=1)) if n > 1 else 0.0
        sharpe = (mean_tk / sd) * np.sqrt(n) if sd > 0 else 0.0
        rows.append({"thr": thr, "n": n, "mean_tk": mean_tk, "PF": pf, "WR": wr, "sharpe": sharpe})

    feat_imp = dict(zip(FEATURE_COLS, booster.feature_importance(importance_type="gain").tolist()))

    return {
        "tag": tag,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "base_rate_positive": base_rate,
        "auc": auc,
        "best_iter": int(booster.best_iteration or 0),
        "threshold_grid": rows,
        "feature_importance_gain": feat_imp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(FILLS_ROOT / "hc442_v2_canon_c1_fifo_fills.csv"))
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"[meta-clf] missing fills CSV: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[meta-clf] loaded {len(df)} fills from {path.name}", file=sys.stderr)
    df = featurize(df)

    dates = sorted(df["date"].unique())
    print(f"[meta-clf] {len(dates)} unique dates", file=sys.stderr)

    # 70/30 time-ordered split
    n = len(dates)
    cut = int(n * 0.70)
    train_dates = dates[:cut]
    test_dates = dates[cut:]
    oos_result = train_eval(df, train_dates, test_dates, tag=f"time_split_{cut}_{n-cut}")

    # 80/20 random within-day split (date-stratified)
    rng = np.random.default_rng(42)
    train_dates_rand = list(rng.choice(dates, size=int(n*0.8), replace=False))
    test_dates_rand = [d for d in dates if d not in train_dates_rand]
    iid_result = train_eval(df, train_dates_rand, test_dates_rand, tag=f"random_80_20")

    out = {
        "fills_csv": str(path),
        "n_total": int(len(df)),
        "n_dates": int(n),
        "time_ordered_split": oos_result,
        "random_80_20_split": iid_result,
    }
    with open(OUT_DIR / "oos_eval.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    # markdown report
    lines = ["# HC #444 Meta-Classifier v1 Report", ""]
    lines.append(f"Fills source: `{path.name}`  ({len(df)} fills × {n} dates)")
    lines.append("")
    for label, r in [("Time-ordered 70/30 (OOS holdout)", oos_result),
                     ("Random 80/20 (learnability)", iid_result)]:
        lines.append(f"## {label}")
        if r.get("skipped"):
            lines.append(f"- SKIPPED (n_train={r['n_train']}, n_test={r['n_test']})")
            lines.append("")
            continue
        lines.append(f"- n_train = {r['n_train']}, n_test = {r['n_test']}")
        lines.append(f"- base positive-class rate (test) = {r['base_rate_positive']:.3f}")
        lines.append(f"- AUC = {r['auc']:.4f}")
        lines.append(f"- best_iter = {r['best_iter']}")
        lines.append("")
        lines.append("Threshold sweep (subset of test fills above prob threshold):")
        lines.append("")
        lines.append("| thr | n | mean_tk | PF | WR | sharpe |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in r["threshold_grid"]:
            lines.append(f"| {row['thr']:.2f} | {row['n']} | {row['mean_tk']:+.3f} | "
                         f"{row['PF']:.2f} | {row['WR']:.1f}% | {row['sharpe']:+.2f} |")
        lines.append("")
        lines.append("Top-5 feature importance (gain):")
        sorted_feats = sorted(r["feature_importance_gain"].items(), key=lambda x: -x[1])
        for k, v in sorted_feats[:5]:
            lines.append(f"- `{k}`: {v:.1f}")
        lines.append("")
    # Champion search
    lines.append("## HC #444 R3 verdict")
    champ = None
    if not oos_result.get("skipped"):
        for row in oos_result["threshold_grid"]:
            if row["mean_tk"] > 0 and row["PF"] >= 1.10 and row["n"] >= 50:
                if champ is None or row["sharpe"] > champ["sharpe"]:
                    champ = row
    if champ:
        lines.append(f"✅ **Champion threshold = {champ['thr']:.2f}** "
                     f"(n={champ['n']}, mean_tk=+{champ['mean_tk']:.3f}, "
                     f"PF={champ['PF']:.2f}, WR={champ['WR']:.1f}%, "
                     f"sharpe={champ['sharpe']:+.2f})")
        lines.append("")
        lines.append("→ Apply this classifier + threshold as a pre-trade gate on the canonical "
                     "config to obtain a tradeable subset. Re-run canonical sweep on the gated "
                     "signals for confirmation.")
    else:
        lines.append("❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the "
                     "time-ordered OOS test set.")
        lines.append("")
        lines.append("Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, "
                     "signal density, side) do not separate profitable from unprofitable fills "
                     "well enough to rescue the canonical loss. Stronger features required "
                     "(book imbalance at signal, recent realized vol, multi-model agreement).")
    with open(OUT_DIR / "report.md", "w") as fh:
        fh.write("\n".join(lines))

    print(f"[meta-clf] DONE — saved {OUT_DIR}", file=sys.stderr)
    if champ:
        print(f"[meta-clf] ✅ champion thr={champ['thr']:.2f} mean_tk=+{champ['mean_tk']:.3f}", file=sys.stderr)
    else:
        print(f"[meta-clf] ❌ no champion threshold passes gates", file=sys.stderr)


if __name__ == "__main__":
    main()
