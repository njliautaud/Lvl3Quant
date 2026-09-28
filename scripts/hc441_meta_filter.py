#!/usr/bin/env python3
"""HC #441 meta-filter — retrain the LGBM confluence classifier on the
NEW champion-geometry fills (SL=0.50, TP=3.00, H=1.5s).

Inputs:
  - output/hc441_champion_fills/fills_R2_STRICT_AGGR_SL0.5_TP3.0_H1.5s.csv
  - output/hc439_deep_mfe_mae/signals/{date}.parquet  (decision-time features)

Output:
  - output/hc441_meta_filter/lgbm_<ts>.pkl
  - output/hc441_meta_filter/threshold_sweep.csv
  - output/hc441_meta_filter/run.log
"""
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
FILLS_CSV = LVL3 / ("output/hc441_champion_fills/"
                    "fills_R2_STRICT_AGGR_SL0.5_TP3.0_H1.5s.csv")
SIG_DIR = LVL3 / "output/hc439_deep_mfe_mae/signals"
OUT_DIR = LVL3 / "output/hc441_meta_filter"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    # hc441 fills CSV uses 'sig_ts_ns'; the signal parquet also uses 'sig_ts_ns'
    df["sig_ts_ns"] = df["sig_ts_ns"].astype(np.int64)
    return df


def load_signal_features(date: str) -> pd.DataFrame | None:
    p = SIG_DIR / f"{date}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["date", "sig_ts_ns"] + DECISION_FEATURES)
    df["date"] = df["date"].astype(str)
    df["sig_ts_ns"] = df["sig_ts_ns"].astype(np.int64)
    df = df.drop_duplicates(subset=["date", "sig_ts_ns"], keep="first")
    return df


def build_dataset(fills: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(fills["date"].unique())
    print(f"Building dataset over {len(dates)} days")
    out = []
    for d in dates:
        sub = fills[fills["date"] == d].copy()
        feats = load_signal_features(d)
        if feats is None:
            print(f"  {d}: no signal parquet, dropping {len(sub)} fills")
            continue
        merged = sub.merge(feats, on=["date", "sig_ts_ns"], how="left",
                           suffixes=("", "_sig"))
        n_missing = merged["pred_1s"].isna().sum()
        if n_missing > 0:
            print(f"  {d}: {len(merged)} fills, {n_missing} unmatched")
        out.append(merged)
    df = pd.concat(out, ignore_index=True)
    print(f"Joined: {len(df)} rows  NaN-feature: "
          f"{df[DECISION_FEATURES].isna().any(axis=1).sum()}")
    before = len(df)
    df = df.dropna(subset=DECISION_FEATURES)
    print(f"After dropping NaN: {len(df)} ({before-len(df)} dropped)")
    return df


def train_eval(df: pd.DataFrame):
    import lightgbm as lgb

    days = sorted(df["date"].unique())
    n_train = int(len(days) * 0.7)
    train_days = set(days[:n_train])
    test_days  = set(days[n_train:])
    print(f"Train days: {len(train_days)} ({days[0]}..{days[n_train-1]})")
    print(f"Test  days: {len(test_days)} ({days[n_train]}..{days[-1]})")

    Xtr = df[df["date"].isin(train_days)][DECISION_FEATURES].copy()
    ytr = df[df["date"].isin(train_days)]["net_ticks"].astype(float)
    Xte = df[df["date"].isin(test_days)][DECISION_FEATURES].copy()
    yte = df[df["date"].isin(test_days)]["net_ticks"].astype(float)
    test_meta = df[df["date"].isin(test_days)][
        ["date", "sig_ts_ns", "exit_reason", "net_ticks"]].copy()

    print(f"Train fills: {len(Xtr)}  Test fills: {len(Xte)}")
    print(f"Train mean net_tk: {ytr.mean():.4f}  "
          f"Test mean net_tk: {yte.mean():.4f}")

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.8,
        bagging_freq=4, verbose=-1)
    model.fit(Xtr, ytr,
              categorical_feature=CAT_FEATURES,
              eval_set=[(Xte, yte)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

    # Feature importance
    fi = pd.DataFrame({
        "feature": DECISION_FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importance:")
    print(fi.to_string(index=False))

    # Threshold sweep
    yhat_te = model.predict(Xte)
    test_meta["pred_net_tk"] = yhat_te
    sweep_rows = []
    for thr in np.linspace(yhat_te.min(), yhat_te.max(), 41):
        kept = test_meta[test_meta["pred_net_tk"] >= thr]
        n = len(kept)
        if n == 0:
            sweep_rows.append({"threshold": thr, "n": 0})
            continue
        mu = kept["net_ticks"].mean()
        sd = kept["net_ticks"].std()
        sh = mu / sd * np.sqrt(n) if sd > 0 else float("nan")
        gw = kept.loc[kept["net_ticks"] > 0, "net_ticks"].sum()
        gl = -kept.loc[kept["net_ticks"] <= 0, "net_ticks"].sum()
        pf = gw / gl if gl > 0 else float("inf")
        per_day = kept.groupby("date")["net_ticks"].sum()
        pos_days = (per_day > 0).sum()
        sweep_rows.append({
            "threshold": thr, "n": n, "net_tk": mu, "Sh": sh, "PF": pf,
            "WR": (kept["net_ticks"] > 0).mean() * 100,
            "n_days": per_day.shape[0], "pos_days": pos_days,
            "frac_kept": n / len(test_meta),
        })
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT_DIR / "threshold_sweep.csv", index=False)

    # Print top by Sharpe with n>=200
    print("\n=== Threshold sweep — top 10 by Sharpe (n>=200) ===")
    s2 = sweep[sweep["n"] >= 200].sort_values("Sh", ascending=False).head(10)
    print(s2.to_string(index=False))

    # Baseline (no filter)
    base_mu = yte.mean(); base_sd = yte.std()
    base_sh = base_mu / base_sd * np.sqrt(len(yte))
    base_gw = yte[yte > 0].sum(); base_gl = -yte[yte <= 0].sum()
    base_pf = base_gw / base_gl if base_gl > 0 else float("inf")
    print(f"\nBASELINE (no filter, full test): n={len(yte)} net={base_mu:+.4f} "
          f"Sh={base_sh:.2f} PF={base_pf:.2f}")

    # Save model + sweep
    ts = time.strftime("%Y%m%d_%H%M%S")
    pkl = OUT_DIR / f"lgbm_{ts}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"model": model, "features": DECISION_FEATURES,
                     "cat_features": CAT_FEATURES,
                     "train_days": sorted(train_days),
                     "test_days": sorted(test_days)}, f)
    print(f"Saved model: {pkl}")
    return model, sweep, test_meta


def main():
    fills = load_fills()
    print(f"Loaded {len(fills)} champion fills "
          f"across {fills['date'].nunique()} dates")
    print(f"  net_tk mean = {fills['net_ticks'].mean():+.4f}  "
          f"std = {fills['net_ticks'].std():.4f}")
    df = build_dataset(fills)
    train_eval(df)


if __name__ == "__main__":
    main()
