#!/usr/bin/env python3
"""
HC #270 — Phase 3: Meta-LGBM gate trainer.

Walk-forward (sliding 30 train days, 1 OOT day, slide 1d) binary classification of
label_winner (FIFO net P&L > 0) on the enriched signal features. Per HC #0 and HC #257.

Inputs:  /home/jupiter/Lvl3Quant/output/meta_lgbm_features/<date>_signals_enriched.parquet
Outputs: /home/jupiter/Lvl3Quant/output/meta_lgbm_gate_v1/
            fold_<NN>_<date>.npz       — per-fold OOT predictions
            concat_oot_predictions.npz — concatenated OOT predictions across all folds
            concat_oot_metrics.json    — summary metrics per fold + overall
            fold_<NN>_model.txt        — saved LGBM booster
            feature_importance.csv     — gain importances
MLflow tracking URI must be Tailscale (neptune-win:5000) — not LAN.
"""

from pathlib import Path
import os
import argparse
import logging
import json
import sys
import numpy as np
import pandas as pd

LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
FEAT_DIR = LVL3_ROOT / "output" / "meta_lgbm_features"
DEFAULT_OUT = LVL3_ROOT / "output" / "meta_lgbm_gate_v1"

# Tailscale Jupiter IP — required per session history (LAN URI = silent failure on Neptune)
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://neptune-win:5000")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("meta_lgbm_train")


# ----- Feature columns -----
def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of LGBM feature columns. Excludes label, identifier, leakage cols."""
    drop = {
        "date", "signal_ts_ns", "pred_idx",
        "is_filled", "pnl_ticks_gross", "pnl_ticks_net", "exit_reason",
        "hold_time_ns", "queue_wait_ns", "mid_at_signal", "spread_at_signal",
        "label_winner", "label_net_ticks",
    }
    feats = []
    for c in df.columns:
        if c in drop:
            continue
        if df[c].dtype == "object" and c != "direction":
            continue
        feats.append(c)
    return feats


def _encode_features(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """direction is the only categorical we use. Encode it as binary 'is_short'."""
    out = df[feat_cols].copy()
    if "direction" in out.columns:
        out["is_short"] = (out["direction"] == "short").astype(np.int8)
        out = out.drop(columns=["direction"])
    return out


def _walkforward_folds(dates: list[str], train_days: int = 30) -> list[tuple]:
    """Yield (fold_idx, train_dates, oot_date) tuples. Sliding window."""
    folds = []
    for i, oot in enumerate(dates):
        if i < train_days:
            continue
        train = dates[i - train_days:i]
        folds.append((len(folds), train, oot))
    return folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--train-days", type=int, default=30,
                   help="Sliding train window size in days (HC #0 sliding only)")
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--min-data-in-leaf", type=int, default=200)
    p.add_argument("--lambda-l2", type=float, default=1.0)
    p.add_argument("--early-stopping", type=int, default=30)
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Tail fraction of training set used for early-stopping val")
    p.add_argument("--filter-direction", type=str, default="both",
                   choices=["both", "long", "short"],
                   help="Restrict training to one side (short-only deploy is HC #267)")
    p.add_argument("--mlflow-experiment", type=str, default="meta_lgbm_gate")
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import lightgbm as lgb
    except ImportError:
        log.error("lightgbm is required: pip install lightgbm")
        sys.exit(1)

    # Discover enriched parquets
    paths = sorted(FEAT_DIR.glob("*_signals_enriched.parquet"))
    if len(paths) < args.train_days + 1:
        log.error(f"Need ≥{args.train_days + 1} enriched dates; found {len(paths)}")
        sys.exit(1)
    dates = [p.name[:8] for p in paths]
    path_by_date = {d: p for d, p in zip(dates, paths)}

    log.info(f"Found {len(dates)} enriched dates: {dates[0]} → {dates[-1]}")

    folds = _walkforward_folds(dates, train_days=args.train_days)
    log.info(f"Walk-forward folds: {len(folds)} (sliding {args.train_days}-day train)")

    # Pre-build feature col list from one date
    sample = pd.read_parquet(paths[0])
    if args.filter_direction != "both":
        sample = sample[sample["direction"] == args.filter_direction]
    feat_cols = _feature_columns(sample)
    log.info(f"Feature columns: {len(feat_cols)} (filter_direction={args.filter_direction})")
    log.info(f"  feats: {feat_cols}")

    # ----- MLflow -----
    mlflow = None
    if not args.no_mlflow:
        try:
            import mlflow as _mlflow
            _mlflow.set_tracking_uri(MLFLOW_URI)
            _mlflow.set_experiment(args.mlflow_experiment)
            mlflow = _mlflow
            log.info(f"MLflow URI={MLFLOW_URI}, experiment={args.mlflow_experiment}")
        except Exception as e:
            log.warning(f"MLflow setup failed ({e}); continuing without MLflow")

    if mlflow:
        mlflow.start_run(run_name=f"meta_lgbm_gate_{args.filter_direction}")
        mlflow.log_params({
            "train_days": args.train_days, "num_leaves": args.num_leaves,
            "learning_rate": args.learning_rate, "n_estimators": args.n_estimators,
            "min_data_in_leaf": args.min_data_in_leaf, "lambda_l2": args.lambda_l2,
            "early_stopping": args.early_stopping, "val_frac": args.val_frac,
            "filter_direction": args.filter_direction, "n_features": len(feat_cols),
            "n_folds": len(folds),
        })

    # ----- Train per-fold -----
    fold_metrics = []
    concat_records = []
    importances = None

    for fold_idx, train_dates, oot_date in folds:
        log.info(f"=== Fold {fold_idx:02d}: train={train_dates[0]}→{train_dates[-1]} OOT={oot_date} ===")

        # Load train+val
        train_dfs = []
        for d in train_dates:
            df = pd.read_parquet(path_by_date[d])
            if args.filter_direction != "both":
                df = df[df["direction"] == args.filter_direction]
            train_dfs.append(df)
        train_full = pd.concat(train_dfs, ignore_index=True)
        # Sort by signal_ts_ns to ensure time order for val split
        train_full = train_full.sort_values("signal_ts_ns").reset_index(drop=True)
        n_train_total = len(train_full)
        n_val = int(n_train_total * args.val_frac)
        n_train = n_train_total - n_val
        train = train_full.iloc[:n_train]
        val = train_full.iloc[n_train:]

        oot = pd.read_parquet(path_by_date[oot_date])
        if args.filter_direction != "both":
            oot = oot[oot["direction"] == args.filter_direction]
        oot = oot.sort_values("signal_ts_ns").reset_index(drop=True)

        X_tr = _encode_features(train, feat_cols)
        X_val = _encode_features(val, feat_cols)
        X_oot = _encode_features(oot, feat_cols)
        y_tr = train["label_winner"].values
        y_val = val["label_winner"].values
        y_oot = oot["label_winner"].values

        # Class weight
        n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
        scale_pos_weight = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0

        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "num_leaves": args.num_leaves,
            "learning_rate": args.learning_rate,
            "min_data_in_leaf": args.min_data_in_leaf,
            "lambda_l2": args.lambda_l2,
            "scale_pos_weight": scale_pos_weight,
            "feature_pre_filter": False,
            "verbosity": -1,
            "force_col_wise": True,
        }
        booster = lgb.train(
            params, lgb_train,
            num_boost_round=args.n_estimators,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(args.early_stopping, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # Predictions
        p_oot = booster.predict(X_oot, num_iteration=booster.best_iteration)
        # Per-fold metrics
        from sklearn.metrics import roc_auc_score, log_loss
        try:
            auc = float(roc_auc_score(y_oot, p_oot)) if len(np.unique(y_oot)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        try:
            ll = float(log_loss(y_oot, np.clip(p_oot, 1e-6, 1 - 1e-6))) if len(y_oot) > 0 else float("nan")
        except Exception:
            ll = float("nan")
        # Brier
        brier = float(np.mean((p_oot - y_oot) ** 2)) if len(y_oot) > 0 else float("nan")
        # Decile lift: top-10% predicted winrate
        order = np.argsort(-p_oot)
        top10 = order[: max(1, len(order) // 10)]
        wr_top10 = float(y_oot[top10].mean()) if len(top10) else float("nan")
        wr_base = float(y_oot.mean())
        # Net-ticks lift: sum of label_net_ticks over filled rows in top-10%
        net_lift = float("nan")
        if "label_net_ticks" in oot.columns:
            net_top10 = oot["label_net_ticks"].iloc[top10].dropna().sum()
            net_lift = float(net_top10)

        log.info(f"  OOT: n={len(oot)} AUC={auc:.4f} logloss={ll:.4f} brier={brier:.4f} "
                 f"WR_top10%={wr_top10:.3f} (base={wr_base:.3f}) net_top10={net_lift:+.1f}t")

        fold_metrics.append({
            "fold": fold_idx, "oot_date": oot_date, "n_oot": int(len(oot)),
            "auc": auc, "logloss": ll, "brier": brier,
            "wr_top10": wr_top10, "wr_base": wr_base, "net_top10_ticks": net_lift,
            "best_iter": int(booster.best_iteration or args.n_estimators),
        })

        # Save fold artifacts
        fold_npz = out_dir / f"fold_{fold_idx:02d}_{oot_date}.npz"
        np.savez(fold_npz,
                 signal_ts_ns=oot["signal_ts_ns"].values,
                 p_win=p_oot.astype(np.float32),
                 label_winner=y_oot.astype(np.int8),
                 label_net_ticks=oot["label_net_ticks"].values.astype(np.float32),
                 direction=oot["direction"].astype(str).values,
                 is_filled=oot["is_filled"].values.astype(bool),
                 date=np.array(oot_date))

        booster.save_model(str(out_dir / f"fold_{fold_idx:02d}_model.txt"))

        # Accumulate concat records
        concat_records.append({
            "signal_ts_ns": oot["signal_ts_ns"].values,
            "p_win": p_oot.astype(np.float32),
            "label_winner": y_oot.astype(np.int8),
            "label_net_ticks": oot["label_net_ticks"].values.astype(np.float32),
            "direction": oot["direction"].astype(str).values,
            "is_filled": oot["is_filled"].values.astype(bool),
            "date": np.array([oot_date] * len(oot)),
        })

        # Aggregate importance
        gain = booster.feature_importance(importance_type="gain")
        names = booster.feature_name()
        if importances is None:
            importances = pd.DataFrame({"feature": names, "gain": gain.astype(np.float64)})
        else:
            tmp = pd.DataFrame({"feature": names, "gain": gain.astype(np.float64)})
            importances = importances.merge(tmp, on="feature", how="outer", suffixes=("", "_x"))
            importances["gain"] = importances[["gain", "gain_x"]].sum(axis=1)
            importances = importances.drop(columns=["gain_x"])

        if mlflow:
            mlflow.log_metrics({
                f"fold{fold_idx:02d}_auc": auc,
                f"fold{fold_idx:02d}_wr_top10": wr_top10,
                f"fold{fold_idx:02d}_net_top10": net_lift,
            }, step=fold_idx)

    # ----- Concat -----
    cat = {k: np.concatenate([r[k] for r in concat_records]) for k in concat_records[0].keys()}
    np.savez(out_dir / "concat_oot_predictions.npz", **cat)
    log.info(f"Wrote concat: {len(cat['signal_ts_ns']):,} rows")

    # Overall metrics
    p_all = cat["p_win"]; y_all = cat["label_winner"]
    from sklearn.metrics import roc_auc_score
    overall_auc = float(roc_auc_score(y_all, p_all)) if len(np.unique(y_all)) > 1 else float("nan")
    folds_pos_top10 = sum(1 for fm in fold_metrics if fm["net_top10_ticks"] > 0)
    pct_folds_pos = folds_pos_top10 / max(1, len(fold_metrics))
    log.info(f"OVERALL: AUC={overall_auc:.4f}  folds_with_top10_positive={folds_pos_top10}/{len(fold_metrics)} ({100*pct_folds_pos:.1f}%)")

    summary = {
        "overall_auc": overall_auc,
        "n_folds": len(fold_metrics),
        "folds_with_positive_top10_net": folds_pos_top10,
        "pct_folds_positive_top10": pct_folds_pos,
        "fold_metrics": fold_metrics,
        "feature_columns": feat_cols + (["is_short"] if "direction" in feat_cols else []),
        "filter_direction": args.filter_direction,
        "params": vars(args),
    }
    with open(out_dir / "concat_oot_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Importances
    importances = importances.sort_values("gain", ascending=False).reset_index(drop=True)
    importances.to_csv(out_dir / "feature_importance.csv", index=False)
    log.info(f"Top 10 features by gain:")
    for _, row in importances.head(10).iterrows():
        log.info(f"  {row['feature']:32s}  gain={row['gain']:.0f}")

    if mlflow:
        mlflow.log_metric("overall_auc", overall_auc)
        mlflow.log_metric("pct_folds_positive_top10", pct_folds_pos)
        mlflow.log_artifact(str(out_dir / "concat_oot_metrics.json"))
        mlflow.log_artifact(str(out_dir / "feature_importance.csv"))
        mlflow.end_run()

    log.info("DONE.")


if __name__ == "__main__":
    main()
