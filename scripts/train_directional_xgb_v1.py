"""
Directional Streaming XGBoost v1 — Signed price change prediction
=================================================================
Same feature pipeline as streaming continuation v1, but predicts
DIRECTIONAL price change (labels_30s from events) instead of
direction-agnostic mfe_minus_mae.

If this achieves IC > 0.15 on direction at 30s, it changes everything
for trade simulation profitability.

Features: smart_v3 events (25) + book features (30) + label-features (4) = 59
Target: labels_30s (signed price change at 30s horizon, in ticks)
Walk-forward: 30d train, 1d OOT, sliding (HC #0)
"""
from __future__ import annotations
import gc, sys, time, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
try:
    import mlflow; HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

DATA_ROOT = Path("/home/nick/Lvl3Quant/data")
EVENTS_DIR = DATA_ROOT / "processed/mbo_events_smart_v3"
BOOK_DIR = DATA_ROOT / "processed/mbo_book_features"
RELABEL_DIR = DATA_ROOT / "relabel"  # only for MFE features
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/directional_xgb_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_URI = "http://localhost:5000"

TRAIN_DAYS = 30
OOT_DAYS = 1
VAL_FRAC = 0.20
TRAIN_SUBSAMPLE_PER_DAY = 50_000
OOT_MAX_EVENTS = 500_000

# Two variants: with and without label-leaking features
# Variant A: use labels_1s/5s/10s as features (upper bound, needs CNN-Mamba preds in live)
# Variant B: events + book only (no label features, standalone model)

HORIZONS = [30, 60]  # 30s and 60s signed price change

XGB_PARAMS = {
    "device": "cuda",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 200,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "verbosity": 0,
}
N_ROUNDS = 800
EARLY_STOP = 30

EVENT_FEAT_NAMES = [f"ev_{i}" for i in range(25)]
BOOK_FEAT_NAMES = [
    "bid_px1","bid_px2","bid_px3","bid_px4","bid_px5",
    "ask_px1","ask_px2","ask_px3","ask_px4","ask_px5",
    "bid_sz1","bid_sz2","bid_sz3","bid_sz4","bid_sz5",
    "ask_sz1","ask_sz2","ask_sz3","ask_sz4","ask_sz5",
    "cum_delta","roll_imbal_100","trade_intensity_100",
    "depth_imbal_5","spread_ticks","bid_sz_chg","ask_sz_chg",
    "mid_px_chg_ticks","spread_chg_ticks","net_order_flow",
]
LABEL_FEAT_NAMES = ["lab_1s","lab_5s","lab_10s","lab_30s"]
FEAT_NAMES_WITH_LABELS = EVENT_FEAT_NAMES + BOOK_FEAT_NAMES + LABEL_FEAT_NAMES
FEAT_NAMES_NO_LABELS = EVENT_FEAT_NAMES + BOOK_FEAT_NAMES


def discover_dates(horizon_s: int) -> list[str]:
    event_dates = {p.stem.split("_")[0] for p in EVENTS_DIR.glob("2026*_mbo_events.npz")}
    book_dates = {p.stem.split("_")[0] for p in BOOK_DIR.glob("2026*_book_features.npz")}
    common = sorted(event_dates & book_dates)
    print(f"  Found {len(common)} dates with events + book features")
    return common


def load_date(date_str: str, horizon_s: int, max_events: int = None,
              use_labels: bool = True) -> tuple | None:
    ev_path = EVENTS_DIR / f"{date_str}_mbo_events.npz"
    bk_path = BOOK_DIR / f"{date_str}_book_features.npz"
    if not ev_path.exists() or not bk_path.exists():
        return None

    ev = np.load(ev_path, allow_pickle=True)
    bk = np.load(bk_path, allow_pickle=True)
    ev_feats = ev["events"]
    bk_feats = bk["features"]
    N = ev_feats.shape[0]
    if bk_feats.shape[0] != N:
        return None

    # Target: signed price change at horizon
    if horizon_s == 30:
        target = ev["labels_30s"].astype(np.float32)
    elif horizon_s == 60:
        # labels_30s is for 30s horizon. For 60s we'd need labels_60s which doesn't exist.
        # Use relabel file for 60s forward return
        rl_path = RELABEL_DIR / f"mfe_mae_h60s_{date_str}.parquet"
        if not rl_path.exists():
            return None
        rl = pd.read_parquet(rl_path, columns=["mid_t_ticks"])
        mid = rl["mid_t_ticks"].values.astype(np.float32)
        if len(mid) != N:
            return None
        # Can't compute forward return without timestamps/event rate
        # Fall back to just using 30s labels for now
        target = ev["labels_30s"].astype(np.float32)
    else:
        target = ev["labels_30s"].astype(np.float32)

    valid_mask = ~np.isnan(target)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return None

    if max_events is not None and len(valid_idx) > max_events:
        rng = np.random.default_rng(hash(date_str) % 2**32)
        chosen = rng.choice(valid_idx, max_events, replace=False)
        chosen.sort()
    else:
        chosen = valid_idx

    if use_labels:
        labels = np.column_stack([
            ev["labels_1s"][chosen].astype(np.float32),
            ev["labels_5s"][chosen].astype(np.float32),
            ev["labels_10s"][chosen].astype(np.float32),
            ev["labels_30s"][chosen].astype(np.float32),
        ])
        X = np.hstack([
            ev_feats[chosen].astype(np.float32),
            bk_feats[chosen].astype(np.float32),
            labels,
        ])
    else:
        X = np.hstack([
            ev_feats[chosen].astype(np.float32),
            bk_feats[chosen].astype(np.float32),
        ])

    y = target[chosen]
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def walk_forward_train(dates: list[str], use_labels: bool, variant_name: str) -> dict:
    horizon_s = 30  # fixed to 30s for now

    feat_names = FEAT_NAMES_WITH_LABELS if use_labels else FEAT_NAMES_NO_LABELS
    n_feats = len(feat_names)

    print(f"\n{'='*70}")
    print(f"DIRECTIONAL XGB — {variant_name}")
    print(f"Target: signed price change at {horizon_s}s | Features: {n_feats}")
    print(f"Walk-forward: {TRAIN_DAYS}d train, {OOT_DAYS}d OOT | {len(dates)} dates")
    print(f"{'='*70}")

    all_preds, all_actuals, all_dates_out = [], [], []
    fold_metrics = []
    feat_imp = np.zeros(n_feats, dtype=np.float64)

    total_folds = len(dates) - TRAIN_DAYS
    if total_folds <= 0:
        print("Not enough dates")
        return {}

    for fold_idx in range(total_folds):
        train_dates = dates[fold_idx:fold_idx + TRAIN_DAYS]
        oot_date = dates[fold_idx + TRAIN_DAYS]
        t0 = time.time()

        oot_data = load_date(oot_date, horizon_s, OOT_MAX_EVENTS, use_labels)
        if oot_data is None:
            continue
        X_oot, y_oot = oot_data

        X_parts, y_parts = [], []
        for td in train_dates:
            td_data = load_date(td, horizon_s, TRAIN_SUBSAMPLE_PER_DAY, use_labels)
            if td_data is not None:
                X_parts.append(td_data[0])
                y_parts.append(td_data[1])

        if len(X_parts) < TRAIN_DAYS * 0.5:
            del X_oot, y_oot
            continue

        X_train = np.vstack(X_parts)
        y_train = np.concatenate(y_parts)
        del X_parts, y_parts; gc.collect()

        val_sz = int(len(y_train) * VAL_FRAC)
        X_val, y_val = X_train[-val_sz:], y_train[-val_sz:]
        X_tr, y_tr = X_train[:-val_sz], y_train[:-val_sz]
        del X_train, y_train; gc.collect()

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_names)
        doot = xgb.DMatrix(X_oot, label=y_oot, feature_names=feat_names)
        del X_tr, y_tr, X_val, y_val; gc.collect()

        model = xgb.train(
            XGB_PARAMS, dtrain, N_ROUNDS,
            evals=[(dval, "val")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=False,
        )

        oot_pred = model.predict(doot)
        rho, _ = spearmanr(oot_pred, y_oot)
        ic = np.corrcoef(oot_pred, y_oot)[0, 1]
        dir_acc = (np.sign(oot_pred) == np.sign(y_oot)).mean()
        
        # Directional profitability: avg actual move for top/bottom decile predictions
        abs_pred = np.abs(oot_pred)
        p90 = np.percentile(abs_pred, 90)
        top_mask = abs_pred >= p90
        top_pred_signs = np.sign(oot_pred[top_mask])
        top_actual = y_oot[top_mask]
        top_gross = (top_pred_signs * top_actual).mean()  # avg gross ticks for top decile

        elapsed = time.time() - t0
        print(f"  Fold {fold_idx:2d} ({oot_date}): "
              f"IC={ic:.4f} rho={rho:.4f} dir_acc={dir_acc:.3f} "
              f"top10%_gross={top_gross:+.3f}t "
              f"n_train={len(dtrain.get_label()):,} n_oot={len(y_oot):,} "
              f"rounds={model.best_iteration} ({elapsed:.0f}s)")

        all_preds.append(oot_pred)
        all_actuals.append(y_oot)
        all_dates_out.extend([oot_date] * len(y_oot))
        fold_metrics.append({"date": oot_date, "ic": ic, "rho": rho,
                             "dir_acc": dir_acc, "top10_gross": top_gross})

        imp = model.get_score(importance_type="gain")
        for i, fn in enumerate(feat_names):
            feat_imp[i] += imp.get(fn, 0)

        del dtrain, dval, doot, model; gc.collect()

    if not all_preds:
        return {}

    # Concat OOS
    concat_preds = np.concatenate(all_preds)
    concat_actual = np.concatenate(all_actuals)
    concat_dates = np.array(all_dates_out)

    # Concat metrics
    rho_c, _ = spearmanr(concat_preds, concat_actual)
    ic_c = np.corrcoef(concat_preds, concat_actual)[0, 1]
    dir_acc_c = (np.sign(concat_preds) == np.sign(concat_actual)).mean()

    print(f"\n{'='*70}")
    print(f"CONCAT OOS RESULTS — {variant_name}")
    print(f"{'='*70}")
    print(f"  Total OOS predictions: {len(concat_preds):,}")
    print(f"  Concat IC: {ic_c:.4f} | Spearman: {rho_c:.4f}")
    print(f"  Directional accuracy: {dir_acc_c:.4f} ({dir_acc_c:.1%})")

    # Decile analysis
    print(f"\n  Decile analysis:")
    cuts = np.percentile(concat_preds, np.arange(0, 101, 10))
    for i in range(10):
        lo, hi = cuts[i], cuts[i+1]
        mask = (concat_preds >= lo) & (concat_preds < hi) if i < 9 else (concat_preds >= lo)
        a = concat_actual[mask]
        gross_if_trade = np.sign((lo+hi)/2) * a.mean()  # gross P&L if we trade the decile's direction
        print(f"    D{i}: pred [{lo:+.3f},{hi:+.3f}) | "
              f"actual_mean={a.mean():+.3f}t | "
              f"WR_correct={(np.sign(a)==np.sign((lo+hi)/2)).mean():.3f} | "
              f"gross_trade={gross_if_trade:+.3f}t | n={mask.sum():,}")

    # Top quantile profitability
    print(f"\n  Top quantile profitability analysis:")
    for q in [1, 2, 5, 10]:
        # Shorts
        thresh = np.percentile(concat_preds, q)
        mask = concat_preds <= thresh
        a = concat_actual[mask]
        short_gross = -a.mean()
        short_wr = (a < 0).mean()
        print(f"    Top {q}% SHORT: n={mask.sum():,} | gross={short_gross:+.3f}t | WR={short_wr:.3f} | "
              f"net@0.752={short_gross-0.752:+.3f}t | net@1.752={short_gross-1.752:+.3f}t")
        
        # Longs
        thresh = np.percentile(concat_preds, 100-q)
        mask = concat_preds >= thresh
        a = concat_actual[mask]
        long_gross = a.mean()
        long_wr = (a > 0).mean()
        print(f"    Top {q}% LONG:  n={mask.sum():,} | gross={long_gross:+.3f}t | WR={long_wr:.3f} | "
              f"net@0.752={long_gross-0.752:+.3f}t | net@1.752={long_gross-1.752:+.3f}t")

    # Feature importance
    feat_imp /= max(len(fold_metrics), 1)
    top_feats = sorted(zip(feat_names, feat_imp), key=lambda x: -x[1])[:15]
    print(f"\n  Top 15 features by gain:")
    for fn, imp in top_feats:
        print(f"    {fn:25s}: {imp:.1f}")

    # Save predictions
    np.savez_compressed(
        OUTPUT_DIR / f"concat_preds_{variant_name}.npz",
        predictions=concat_preds,
        actuals=concat_actual,
        dates=concat_dates,
    )

    # Save fold metrics
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(OUTPUT_DIR / f"fold_metrics_{variant_name}.csv", index=False)

    summary = {
        "variant": variant_name,
        "n_folds": len(fold_metrics),
        "n_predictions": len(concat_preds),
        "concat_ic": ic_c,
        "concat_spearman": rho_c,
        "concat_dir_accuracy": dir_acc_c,
        "mean_fold_ic": fold_df["ic"].mean(),
        "mean_fold_dir_acc": fold_df["dir_acc"].mean(),
        "mean_top10_gross": fold_df["top10_gross"].mean(),
    }
    with open(OUTPUT_DIR / f"summary_{variant_name}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def main():
    t_start = time.time()
    print("=" * 70)
    print("DIRECTIONAL XGBoost v1 — Can we predict DIRECTION at 30s?")
    print("=" * 70)

    dates = discover_dates(30)

    # MLflow
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("directional_xgb_v1")
        except:
            pass

    # Variant A: with label features (upper bound)
    print("\n\n" + "=" * 70)
    print("VARIANT A: Events + Book + Label-features (upper bound)")
    print("Uses labels_1s/5s/10s/30s as features → needs CNN-Mamba preds in live")
    print("=" * 70)
    
    summary_a = walk_forward_train(dates, use_labels=True, variant_name="with_labels")
    
    if HAS_MLFLOW and summary_a:
        with mlflow.start_run(run_name="directional_xgb_with_labels"):
            for k, v in summary_a.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

    # Variant B: without label features (standalone)
    print("\n\n" + "=" * 70)
    print("VARIANT B: Events + Book only (standalone, no label leaking)")
    print("=" * 70)
    
    summary_b = walk_forward_train(dates, use_labels=False, variant_name="no_labels")
    
    if HAS_MLFLOW and summary_b:
        with mlflow.start_run(run_name="directional_xgb_no_labels"):
            for k, v in summary_b.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

    # Compare
    print(f"\n\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    if summary_a and summary_b:
        print(f"  With labels:    IC={summary_a.get('concat_ic',0):.4f} | "
              f"DirAcc={summary_a.get('concat_dir_accuracy',0):.4f} | "
              f"Top10 gross={summary_a.get('mean_top10_gross',0):+.3f}t")
        print(f"  Without labels: IC={summary_b.get('concat_ic',0):.4f} | "
              f"DirAcc={summary_b.get('concat_dir_accuracy',0):.4f} | "
              f"Top10 gross={summary_b.get('mean_top10_gross',0):+.3f}t")
        print(f"\n  Label-feature boost: IC {summary_a.get('concat_ic',0) - summary_b.get('concat_ic',0):+.4f}")

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
