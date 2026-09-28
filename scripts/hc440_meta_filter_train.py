#!/usr/bin/env python3
"""HC #440 — Meta-filter training: predict whether a top-0.5% short signal
will end as SL-kill vs TP under realtime_sl FIFO, using only decision-time
features. Then evaluate filtered vs unfiltered P&L.

Data sources:
  - Per-trade fills (label source):
    output/hc432_v342_47day_validation/v2_short_1s_top0.5_realtime_sl_HC437_47day_bracketparams_fifo_fills.csv
    Columns include: date, ts_signal_ns, fill_type, net_ticks, pred_strength
  - Per-signal features (feature source):
    output/hc439_deep_mfe_mae/signals/<DATE>.parquet
    Columns include: sig_ts_ns, pred_1s/5s/10s, mfe_*_tk, mae_*_tk (these MFE/MAE
    are realized AFTER the signal so they LEAK — must exclude).
    Decision-time features OK: pred_1s/5s/10s, filter_vol_500ev_tk,
    filter_evt_per_sec_30s, filter_buy_aggr_50, filter_spread_proxy_tk,
    tod_min_et, tod_bucket, dow

Output:
  output/hc440_meta_filter/
    meta_filter_dataset.parquet     (joined dataset)
    lgbm_model.pkl                  (trained model)
    feature_importance.csv
    pnl_by_threshold.csv             (filtered P&L sweep)
    holdout_pnl_by_day.csv
    summary.md
"""
from __future__ import annotations
import os
import sys
import time
import glob
from pathlib import Path
import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
FILLS_CSV = LVL3 / ("output/hc432_v342_47day_validation/"
                    "v2_short_1s_top0.5_realtime_sl_HC437_47day_"
                    "bracketparams_fifo_fills.csv")
SIG_DIR = LVL3 / "output/hc439_deep_mfe_mae/signals"
OUT_DIR = LVL3 / "output/hc440_meta_filter"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cost constants (already baked into net_ticks via the FIFO replay).

DECISION_FEATURES = [
    "pred_1s", "pred_5s", "pred_10s",
    "filter_vol_500ev_tk",
    "filter_evt_per_sec_30s",
    "filter_buy_aggr_50",
    "filter_spread_proxy_tk",
    "tod_min_et",
    "dow",
]
CAT_FEATURES = ["dow"]


def load_fills() -> pd.DataFrame:
    df = pd.read_csv(FILLS_CSV)
    df["date"] = df["date"].astype(str)
    df["ts_signal_ns"] = df["ts_signal_ns"].astype(np.int64)
    return df


def load_signal_features(date: str) -> pd.DataFrame | None:
    p = SIG_DIR / f"{date}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["date", "sig_ts_ns"] + DECISION_FEATURES)
    df["date"] = df["date"].astype(str)
    df["sig_ts_ns"] = df["sig_ts_ns"].astype(np.int64)
    # deduplicate (some files have multi-row per ts due to multi-signal events)
    df = df.drop_duplicates(subset=["date", "sig_ts_ns"], keep="first")
    return df


def build_dataset(fills: pd.DataFrame) -> pd.DataFrame:
    """Join fills with signal-level features by (date, ts_signal_ns)."""
    dates = sorted(fills["date"].unique())
    print(f"Building meta-filter dataset over {len(dates)} days...")
    out = []
    for d in dates:
        sub = fills[fills["date"] == d].copy()
        feats = load_signal_features(d)
        if feats is None:
            print(f"  {d}: no signal parquet, skipping {len(sub)} fills")
            continue
        merged = sub.merge(
            feats.rename(columns={"sig_ts_ns": "ts_signal_ns"}),
            on=["date", "ts_signal_ns"], how="left", suffixes=("", "_sig"))
        n_missing = merged["pred_1s"].isna().sum()
        if n_missing > 0:
            print(f"  {d}: {len(merged)} fills, {n_missing} unmatched")
        out.append(merged)
    df = pd.concat(out, ignore_index=True)
    print(f"Joined dataset: {len(df)} rows, "
          f"{df[DECISION_FEATURES].isna().any(axis=1).sum()} with any NaN feature")
    # Drop rows with NaN in features
    before = len(df)
    df = df.dropna(subset=DECISION_FEATURES)
    print(f"After dropping NaN-feature rows: {len(df)} ({before-len(df)} dropped)")
    return df


def train_eval(df: pd.DataFrame):
    """Train LGBM regressor with time-walk split (train: first 70% of days,
    test: last 30%). Evaluate threshold sweep."""
    import lightgbm as lgb

    # Sort days
    days = sorted(df["date"].unique())
    n_train = int(len(days) * 0.7)
    train_days = set(days[:n_train])
    test_days = set(days[n_train:])
    print(f"Train days: {len(train_days)} ({days[0]}..{days[n_train-1]})")
    print(f"Test days:  {len(test_days)} ({days[n_train]}..{days[-1]})")

    Xtr = df[df["date"].isin(train_days)][DECISION_FEATURES].copy()
    ytr = df[df["date"].isin(train_days)]["net_ticks"].astype(float)
    Xte = df[df["date"].isin(test_days)][DECISION_FEATURES].copy()
    yte = df[df["date"].isin(test_days)]["net_ticks"].astype(float)
    test_meta = df[df["date"].isin(test_days)][
        ["date", "ts_signal_ns", "fill_type", "net_ticks"]].copy()

    print(f"Train fills: {len(Xtr)}  Test fills: {len(Xte)}")
    print(f"Train mean net_tk: {ytr.mean():.4f}  Test mean net_tk: {yte.mean():.4f}")

    # Train LGBM regressor
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(Xtr, ytr, categorical_feature=CAT_FEATURES,
              eval_set=[(Xte, yte)], callbacks=[lgb.early_stopping(50)])

    # Predict on test
    test_meta["pred_net_tk"] = model.predict(Xte)

    # Feature importance
    fi = pd.DataFrame({
        "feature": DECISION_FEATURES,
        "importance": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print("\nFeature importance:")
    print(fi.to_string(index=False))

    # Threshold sweep on HOLDOUT (test) set
    thresholds = np.linspace(-1.0, 1.0, 41)
    rows = []
    for thr in thresholds:
        kept = test_meta[test_meta["pred_net_tk"] >= thr]
        n_kept = len(kept)
        if n_kept == 0:
            continue
        net_tk_total = kept["net_ticks"].sum()
        net_tk_mean = kept["net_ticks"].mean()
        wr = (kept["net_ticks"] > 0).mean()
        sl_rate = (kept["fill_type"] == "sl").mean()
        # Daily Sharpe
        daily = kept.groupby("date")["net_ticks"].sum()
        if len(daily) >= 5 and daily.std() > 0:
            sharpe = daily.mean() / daily.std() * np.sqrt(len(daily))
        else:
            sharpe = np.nan
        rows.append({
            "threshold": float(thr),
            "n_kept": n_kept,
            "frac_kept": n_kept / len(test_meta),
            "net_tk_total": float(net_tk_total),
            "net_tk_mean": float(net_tk_mean),
            "wr": float(wr),
            "sl_rate": float(sl_rate),
            "daily_sharpe_sqrtN": float(sharpe) if not np.isnan(sharpe) else None,
            "n_days_traded": int(daily.shape[0]),
        })
    df_thr = pd.DataFrame(rows)
    df_thr.to_csv(OUT_DIR / "pnl_by_threshold.csv", index=False)

    # Baseline (no filter)
    baseline_tot = test_meta["net_ticks"].sum()
    baseline_mean = test_meta["net_ticks"].mean()
    baseline_wr = (test_meta["net_ticks"] > 0).mean()
    daily_b = test_meta.groupby("date")["net_ticks"].sum()
    baseline_sharpe = (daily_b.mean() / daily_b.std() * np.sqrt(len(daily_b))
                       if daily_b.std() > 0 else np.nan)

    print(f"\nBaseline (no filter) on holdout:")
    print(f"  n_fills: {len(test_meta)}")
    print(f"  mean net_tk: {baseline_mean:.4f}")
    print(f"  total net_tk: {baseline_tot:.2f}")
    print(f"  WR: {baseline_wr:.3f}")
    print(f"  Daily Sharpe(sqrtN): {baseline_sharpe:.2f}")

    # Find best threshold by Sharpe with at least 30 fills
    df_thr_valid = df_thr[df_thr["n_kept"] >= 30].copy()
    if len(df_thr_valid) > 0:
        best = df_thr_valid.sort_values("daily_sharpe_sqrtN",
                                        ascending=False).iloc[0]
        print(f"\nBest threshold (by Sharpe, n>=30):")
        print(best.to_string())

    # Also: best by total net_tk
    if len(df_thr_valid) > 0:
        best_tk = df_thr_valid.sort_values("net_tk_total",
                                           ascending=False).iloc[0]
        print(f"\nBest threshold (by total net_tk, n>=30):")
        print(best_tk.to_string())

    # Save per-day P&L at the best threshold
    if len(df_thr_valid) > 0:
        best_thr = best["threshold"]
        kept = test_meta[test_meta["pred_net_tk"] >= best_thr]
        per_day = kept.groupby("date").agg(
            n_fills=("net_ticks", "count"),
            net_tk=("net_ticks", "sum"),
            mean_tk=("net_ticks", "mean"),
            wr=("net_ticks", lambda x: (x > 0).mean()),
        )
        per_day.to_csv(OUT_DIR / "holdout_pnl_by_day.csv")
        print(f"\nPer-day holdout P&L at best Sharpe threshold "
              f"({best_thr:.3f}):")
        print(per_day.to_string())

    # Save model
    import pickle
    with open(OUT_DIR / "lgbm_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": DECISION_FEATURES,
                     "cat_features": CAT_FEATURES,
                     "train_days": sorted(train_days),
                     "test_days": sorted(test_days)}, f)

    return df_thr, baseline_mean, baseline_sharpe, baseline_tot, len(test_meta)


def write_summary(df_thr, baseline_mean, baseline_sharpe, baseline_tot,
                  n_test):
    lines = []
    lines.append("# HC #440 Meta-Filter — Phase A results")
    lines.append("")
    lines.append("Goal: take the v2 top-0.5% short config (which loses "
                 "-0.27 tk/fill under realtime_sl FIFO) and apply a "
                 "decision-time classifier filter to skip the signals most "
                 "likely to end as SL kills. Filtered P&L vs unfiltered "
                 "baseline measured on a hold-out 30% of days "
                 "(time-walk split).")
    lines.append("")
    lines.append(f"Holdout baseline (no filter):")
    lines.append(f"- fills: {n_test}")
    lines.append(f"- mean net_tk/fill: {baseline_mean:.4f}")
    lines.append(f"- total net_tk: {baseline_tot:.2f}")
    lines.append(f"- daily Sharpe sqrtN: {baseline_sharpe:.2f}")
    lines.append("")
    lines.append("Threshold sweep (filter signals with pred_net_tk >= thr):")
    lines.append("")
    df_show = df_thr.copy()
    df_show = df_show[df_show["n_kept"] >= 20]
    # Format
    lines.append("```")
    cols = ["threshold", "n_kept", "frac_kept", "net_tk_mean", "net_tk_total",
            "wr", "sl_rate", "daily_sharpe_sqrtN"]
    widths = [10, 7, 8, 12, 12, 6, 8, 18]
    hdr = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    lines.append(hdr)
    lines.append("-+-".join("-" * w for w in widths))
    for _, r in df_show.iterrows():
        row = []
        for c, w in zip(cols, widths):
            v = r[c]
            if isinstance(v, float):
                row.append(format(v, ".3f").ljust(w))
            elif v is None:
                row.append("nan".ljust(w))
            else:
                row.append(str(v).ljust(w))
        lines.append(" | ".join(row))
    lines.append("```")
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"\nWrote {OUT_DIR / 'summary.md'}")


def main():
    t0 = time.time()
    fills = load_fills()
    print(f"Loaded {len(fills)} fills across {fills['date'].nunique()} days")
    print(f"SL fills: {(fills['fill_type']=='sl').sum()}, "
          f"TP fills: {(fills['fill_type']=='tp').sum()}, "
          f"max_hold: {(fills['fill_type']=='max_hold').sum()}")

    df = build_dataset(fills)
    df.to_parquet(OUT_DIR / "meta_filter_dataset.parquet")
    print(f"Saved dataset to {OUT_DIR / 'meta_filter_dataset.parquet'}")

    df_thr, baseline_mean, baseline_sharpe, baseline_tot, n_test = \
        train_eval(df)
    write_summary(df_thr, baseline_mean, baseline_sharpe, baseline_tot, n_test)
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
