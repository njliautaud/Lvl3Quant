"""
Streaming Continuation Intensity XGBoost — v1
==============================================
Predicts pressure INTENSITY (continuous mfe_minus_mae_ticks) for multi-minute trades.
Walk-forward sliding window: 30-day train, 1-day OOT (HC #0).

Features: smart_v3 events (25 cols) + book features (30 cols) + labels as features = 59 total.
Target: mfe_minus_mae_ticks from relabel parquets (continuous regression).
Horizons: 60s, 120s, 300s.

Memory-efficient: subsample 5% per train day (~1M rows per day → 50k → 1.5M total for 30 days).
OOT: subsample to 500k events max (still statistically robust).

After WF training, runs a streaming backtest simulation.
"""
from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/home/nick/Lvl3Quant/data")
EVENTS_DIR = DATA_ROOT / "processed/mbo_events_smart_v3"
BOOK_DIR = DATA_ROOT / "processed/mbo_book_features"
RELABEL_DIR = DATA_ROOT / "relabel"
OUTPUT_DIR = DATA_ROOT / "models/streaming_continuation_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "streaming_continuation_v1"

# ─── Walk-forward config ──────────────────────────────────────────────────
TRAIN_DAYS = 30
OOT_DAYS = 1
VAL_FRAC = 0.20

# ─── Horizons ─────────────────────────────────────────────────────────────
HORIZONS = [60, 120, 300]

# ─── Subsampling for memory (Neptune has 32GB RAM) ────────────────────────
TRAIN_SUBSAMPLE = 0.05   # 5% of ~13M valid events/day → ~650k/day → ~19.5M total for 30d... still too much
                          # Actually with 30 days: 30 * 650k = 19.5M * 59 * 4 = ~4.6GB. Manageable.
OOT_MAX_EVENTS = 500_000  # Cap OOT at 500k events for speed

# ─── XGBoost GPU params (regression) ─────────────────────────────────────
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

# ─── Cost constants ──────────────────────────────────────────────────────
TAKER_COST_TICKS = 1.376

# ─── Streaming backtest thresholds ────────────────────────────────────────
ENTRY_PCTILES = [90, 95, 99]
EXIT_PCTILES = [30, 40, 50]

# ─── Feature names ───────────────────────────────────────────────────────
EVENT_FEAT_NAMES = [f"ev_{i}" for i in range(25)]
BOOK_FEAT_NAMES = [
    "bid_px1", "bid_px2", "bid_px3", "bid_px4", "bid_px5",
    "ask_px1", "ask_px2", "ask_px3", "ask_px4", "ask_px5",
    "bid_sz1", "bid_sz2", "bid_sz3", "bid_sz4", "bid_sz5",
    "ask_sz1", "ask_sz2", "ask_sz3", "ask_sz4", "ask_sz5",
    "cum_delta", "roll_imbal_100", "trade_intensity_100",
    "depth_imbal_5", "spread_ticks", "bid_sz_chg", "ask_sz_chg",
    "mid_px_chg_ticks", "spread_chg_ticks", "net_order_flow",
]
LABEL_FEAT_NAMES = ["lab_1s", "lab_5s", "lab_10s", "lab_30s"]
ALL_FEAT_NAMES = EVENT_FEAT_NAMES + BOOK_FEAT_NAMES + LABEL_FEAT_NAMES  # 59 total


def discover_dates(horizon_s: int) -> list[str]:
    """Find all 2026 dates where smart_v3 + book + relabel all exist."""
    relabel_dates = {p.stem.split("_")[-1]
                     for p in RELABEL_DIR.glob(f"mfe_mae_h{horizon_s}s_2026*.parquet")}
    event_dates = {p.stem.split("_")[0]
                   for p in EVENTS_DIR.glob("2026*_mbo_events.npz")}
    book_dates = {p.stem.split("_")[0]
                  for p in BOOK_DIR.glob("2026*_book_features.npz")}
    common = sorted(relabel_dates & event_dates & book_dates)
    print(f"  Found {len(common)} dates with all data for h={horizon_s}s")
    return common


def load_date(date_str: str, horizon_s: int, max_events: int | None = None) -> tuple | None:
    """
    Load features + target for one date.
    Returns (X, y) or None. Memory-efficient: loads, subsamples, returns.
    """
    ev_path = EVENTS_DIR / f"{date_str}_mbo_events.npz"
    bk_path = BOOK_DIR / f"{date_str}_book_features.npz"
    rl_path = RELABEL_DIR / f"mfe_mae_h{horizon_s}s_{date_str}.parquet"

    if not ev_path.exists() or not bk_path.exists() or not rl_path.exists():
        return None

    # Load all arrays
    ev = np.load(ev_path, allow_pickle=True)
    bk = np.load(bk_path, allow_pickle=True)

    ev_feats = ev["events"]     # (N, 25) float32
    bk_feats = bk["features"]   # (N, 30) float32
    N = ev_feats.shape[0]

    if bk_feats.shape[0] != N:
        return None

    # Load target
    rl = pd.read_parquet(rl_path, columns=["mfe_minus_mae_ticks"])
    target = rl["mfe_minus_mae_ticks"].values  # (N,) float64
    del rl

    if target.shape[0] != N:
        return None

    # Find valid rows (non-NaN target)
    valid_mask = ~np.isnan(target)
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        return None

    # Subsample if needed
    if max_events is not None and len(valid_idx) > max_events:
        rng = np.random.default_rng(hash(date_str) % 2**32)
        chosen = rng.choice(valid_idx, max_events, replace=False)
        chosen.sort()
    else:
        chosen = valid_idx

    # Build feature matrix for chosen indices only
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
    y = target[chosen].astype(np.float32)

    # Clean inf/nan in features
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


def walk_forward_train(horizon_s: int, dates: list[str]) -> dict | None:
    """Run sliding-window WF training for one horizon."""
    print(f"\n{'='*70}")
    print(f"HORIZON = {horizon_s}s | {len(dates)} dates | train={TRAIN_DAYS}d, oot={OOT_DAYS}d")
    print(f"{'='*70}")

    all_oos_preds = []
    all_oos_actuals = []
    all_oos_dates = []
    fold_metrics = []
    feature_importance_accum = np.zeros(len(ALL_FEAT_NAMES), dtype=np.float64)
    n_folds = 0

    total_folds = len(dates) - TRAIN_DAYS
    if total_folds <= 0:
        print(f"  Not enough dates ({len(dates)} < {TRAIN_DAYS + 1})")
        return None

    # Per-day train subsample: target ~50k events per day
    events_per_day_target = 50_000  # 30 days * 50k = 1.5M total train

    for fold_idx in range(total_folds):
        train_dates = dates[fold_idx : fold_idx + TRAIN_DAYS]
        oot_date = dates[fold_idx + TRAIN_DAYS]

        t_fold = time.time()

        # ── Load OOT (capped at OOT_MAX_EVENTS) ──
        oot_data = load_date(oot_date, horizon_s, max_events=OOT_MAX_EVENTS)
        if oot_data is None:
            continue
        X_oot, y_oot = oot_data

        # ── Load train data day by day ──
        X_parts, y_parts = [], []
        for td in train_dates:
            td_data = load_date(td, horizon_s, max_events=events_per_day_target)
            if td_data is not None:
                X_parts.append(td_data[0])
                y_parts.append(td_data[1])

        if len(X_parts) < TRAIN_DAYS * 0.5:
            print(f"  Fold {fold_idx}: too few train dates ({len(X_parts)}), skip")
            del X_oot, y_oot
            continue

        X_train = np.vstack(X_parts)
        y_train = np.concatenate(y_parts)
        del X_parts, y_parts
        gc.collect()

        # ── Split train/val (last 20%) ──
        val_size = int(len(y_train) * VAL_FRAC)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_tr = X_train[:-val_size]
        y_tr = y_train[:-val_size]
        del X_train, y_train
        gc.collect()

        # ── Build DMatrix and train ──
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=ALL_FEAT_NAMES)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=ALL_FEAT_NAMES)
        doot = xgb.DMatrix(X_oot, label=y_oot, feature_names=ALL_FEAT_NAMES)
        del X_tr, y_tr, X_val, y_val
        gc.collect()

        model = xgb.train(
            XGB_PARAMS,
            dtrain,
            num_boost_round=N_ROUNDS,
            evals=[(dval, "val")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=False,
        )
        del dtrain, dval
        gc.collect()

        # ── Predict OOS ──
        preds = model.predict(doot)
        del doot

        all_oos_preds.append(preds)
        all_oos_actuals.append(y_oot)
        all_oos_dates.extend([oot_date] * len(preds))

        # Fold-level spearman
        if len(np.unique(y_oot)) > 1 and len(np.unique(preds)) > 1:
            sp, _ = spearmanr(preds, y_oot)
        else:
            sp = 0.0

        best_iter = model.best_iteration if hasattr(model, "best_iteration") else N_ROUNDS

        fold_metrics.append({
            "fold": fold_idx, "oot_date": oot_date,
            "n_oot": len(y_oot), "spearman": float(sp), "best_iter": best_iter,
        })

        # Feature importance
        imp = model.get_score(importance_type="gain")
        for fname, gain in imp.items():
            if fname in ALL_FEAT_NAMES:
                feature_importance_accum[ALL_FEAT_NAMES.index(fname)] += gain

        n_folds += 1
        elapsed = time.time() - t_fold

        if fold_idx % 5 == 0 or fold_idx == total_folds - 1:
            print(f"  Fold {fold_idx}/{total_folds-1} | OOT={oot_date} | "
                  f"n_oot={len(y_oot):,} | Sp={sp:.4f} | "
                  f"iters={best_iter} | {elapsed:.0f}s")
            sys.stdout.flush()

        del X_oot, y_oot, preds, model
        gc.collect()

    if n_folds == 0:
        print("  No folds completed!")
        return None

    # ─── Concat OOS analysis ─────────────────────────────────────────────
    all_preds = np.concatenate(all_oos_preds)
    all_actuals = np.concatenate(all_oos_actuals)
    all_dates_arr = np.array(all_oos_dates)

    concat_sp, _ = spearmanr(all_preds, all_actuals)
    print(f"\n  CONCAT Spearman (h={horizon_s}s): {concat_sp:.4f} | "
          f"{n_folds} folds | {len(all_preds):,} OOS events")

    # ─── Decile analysis ─────────────────────────────────────────────────
    decile_edges = np.percentile(all_preds, np.arange(0, 101, 10))
    decile_means = []
    print(f"\n  Decile analysis (predicted → realized mfe_minus_mae):")
    for d in range(10):
        lo, hi = decile_edges[d], decile_edges[d + 1]
        mask = (all_preds >= lo) & (all_preds < hi) if d < 9 else (all_preds >= lo)
        mean_actual = float(all_actuals[mask].mean()) if mask.sum() > 0 else 0.0
        decile_means.append(mean_actual)
        print(f"    D{d}: [{lo:.2f}, {hi:.2f}) → realized={mean_actual:.2f}t | n={mask.sum():,}")

    # ─── Feature importance ──────────────────────────────────────────────
    if n_folds > 0:
        feature_importance_accum /= n_folds
    fi_order = np.argsort(feature_importance_accum)[::-1]
    print(f"\n  Top 15 features by gain:")
    for rank, idx in enumerate(fi_order[:15]):
        print(f"    {rank+1}. {ALL_FEAT_NAMES[idx]}: {feature_importance_accum[idx]:.1f}")
    sys.stdout.flush()

    # ─── Save OOS predictions ────────────────────────────────────────────
    oos_df = pd.DataFrame({"date": all_dates_arr, "pred": all_preds, "actual": all_actuals})
    oos_path = OUTPUT_DIR / f"oos_preds_h{horizon_s}s.parquet"
    oos_df.to_parquet(oos_path, index=False)
    print(f"  Saved: {oos_path}")

    return {
        "horizon_s": horizon_s, "n_folds": n_folds,
        "n_oos_events": len(all_preds),
        "concat_spearman": float(concat_sp),
        "top_decile_mean": float(decile_means[9]),
        "bot_decile_mean": float(decile_means[0]),
        "decile_means": [float(x) for x in decile_means],
        "feature_importance": {ALL_FEAT_NAMES[i]: float(feature_importance_accum[i])
                               for i in fi_order[:20]},
        "fold_metrics": fold_metrics,
        "all_preds": all_preds, "all_actuals": all_actuals, "all_dates": all_dates_arr,
    }


def streaming_backtest(results: dict):
    """
    Simulate streaming trade management using OOS model scores.
    Each event ~250ms apart. Enter on high score, exit when score drops.
    P&L proxy: entry event's realized mfe_minus_mae minus taker cost.
    """
    horizon_s = results["horizon_s"]
    all_preds = results["all_preds"]
    all_actuals = results["all_actuals"]
    all_dates = results["all_dates"]

    print(f"\n{'='*70}")
    print(f"STREAMING BACKTEST | h={horizon_s}s")
    print(f"{'='*70}")

    unique_dates = sorted(set(all_dates))
    n_days = len(unique_dates)

    pctiles = {}
    for p in [30, 40, 50, 90, 95, 99]:
        pctiles[p] = float(np.percentile(all_preds, p))
        print(f"  p{p} = {pctiles[p]:.3f}")

    backtest_results = []

    for entry_pct in ENTRY_PCTILES:
        entry_thresh = pctiles[entry_pct]
        for exit_pct in EXIT_PCTILES:
            exit_thresh = pctiles[exit_pct]

            trades = []
            for date in unique_dates:
                mask = all_dates == date
                day_p = all_preds[mask]
                day_a = all_actuals[mask]

                in_trade = False
                entry_i = 0
                for i in range(len(day_p)):
                    if not in_trade:
                        if day_p[i] > entry_thresh:
                            in_trade = True
                            entry_i = i
                    else:
                        if day_p[i] < exit_thresh:
                            hold_events = i - entry_i
                            pnl = day_a[entry_i] - TAKER_COST_TICKS
                            trades.append({
                                "date": date,
                                "hold_events": hold_events,
                                "entry_score": float(day_p[entry_i]),
                                "exit_score": float(day_p[i]),
                                "entry_actual": float(day_a[entry_i]),
                                "pnl_ticks": float(pnl),
                            })
                            in_trade = False

            if not trades:
                continue

            tdf = pd.DataFrame(trades)
            n_t = len(tdf)
            tpd = n_t / max(n_days, 1)
            mp = tdf["pnl_ticks"].mean()
            sp = tdf["pnl_ticks"].std()
            sharpe = mp / sp * np.sqrt(252) if sp > 0 else 0.0
            wr = (tdf["pnl_ticks"] > 0).mean()
            gw = tdf.loc[tdf["pnl_ticks"] > 0, "pnl_ticks"].sum()
            gl = abs(tdf.loc[tdf["pnl_ticks"] <= 0, "pnl_ticks"].sum())
            pf = gw / gl if gl > 0 else float("inf")

            dpnl = tdf.groupby("date")["pnl_ticks"].sum()
            ds = dpnl.mean() / dpnl.std() * np.sqrt(252) if dpnl.std() > 0 else 0.0

            mh = tdf["hold_events"].mean()

            res = {
                "entry_pct": entry_pct, "exit_pct": exit_pct,
                "entry_thresh": entry_thresh, "exit_thresh": exit_thresh,
                "n_trades": n_t, "trades_per_day": tpd,
                "mean_pnl_ticks": mp, "mean_hold_events": mh,
                "wr": wr, "pf": pf, "trade_sharpe": sharpe, "daily_sharpe": ds,
            }
            backtest_results.append(res)
            print(f"  E=p{entry_pct} X=p{exit_pct} | "
                  f"trades={n_t} ({tpd:.1f}/d) | pnl={mp:.2f}t | "
                  f"WR={wr:.1%} | PF={pf:.2f} | dSharpe={ds:.2f}")

    sys.stdout.flush()
    return backtest_results


def main():
    print("=" * 70)
    print("Streaming Continuation Intensity XGBoost v1")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Horizons: {HORIZONS}")
    print(f"Train: {TRAIN_DAYS}d sliding | Train subsample: {TRAIN_SUBSAMPLE*100:.0f}%/day ({events_per_day_target:,}/day)")
    print(f"XGBoost GPU: hist/cuda | {N_ROUNDS} rounds, ES={EARLY_STOP}")
    print("=" * 70)
    sys.stdout.flush()

    # Setup MLflow
    HAS_MLFLOW_ACTIVE = False
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            HAS_MLFLOW_ACTIVE = True
            print(f"MLflow: {MLFLOW_URI} / {EXPERIMENT_NAME}")
        except Exception as e:
            print(f"MLflow failed: {e}")
    else:
        print("MLflow not available")

    all_results = {}

    for horizon_s in HORIZONS:
        dates = discover_dates(horizon_s)
        if len(dates) < TRAIN_DAYS + 5:
            print(f"  Skip h={horizon_s}s: only {len(dates)} dates")
            continue

        t0 = time.time()
        results = walk_forward_train(horizon_s, dates)
        train_time = time.time() - t0

        if results is None:
            continue

        bt_results = streaming_backtest(results)

        # Log to MLflow
        if HAS_MLFLOW_ACTIVE:
            try:
                with mlflow.start_run(run_name=f"h{horizon_s}s_streaming_v1"):
                    mlflow.log_param("horizon_s", horizon_s)
                    mlflow.log_param("train_days", TRAIN_DAYS)
                    mlflow.log_param("n_folds", results["n_folds"])
                    mlflow.log_param("n_features", 59)
                    mlflow.log_param("train_subsample_per_day", events_per_day_target)
                    mlflow.log_param("model_type", "xgboost_regression")
                    mlflow.log_param("target", "mfe_minus_mae_ticks")

                    mlflow.log_metric("concat_spearman", results["concat_spearman"])
                    mlflow.log_metric("top_decile_mean", results["top_decile_mean"])
                    mlflow.log_metric("bot_decile_mean", results["bot_decile_mean"])
                    mlflow.log_metric("n_oos_events", results["n_oos_events"])
                    mlflow.log_metric("train_time_s", train_time)

                    for d, dm in enumerate(results["decile_means"]):
                        mlflow.log_metric(f"decile_{d}_mean", dm)

                    for fname, gain in list(results["feature_importance"].items())[:10]:
                        mlflow.log_metric(f"fi_{fname}", gain)

                    if bt_results:
                        best = max(bt_results, key=lambda x: x["daily_sharpe"])
                        mlflow.log_metric("best_daily_sharpe", best["daily_sharpe"])
                        mlflow.log_metric("best_pf", best["pf"])
                        mlflow.log_metric("best_wr", best["wr"])
                        mlflow.log_metric("best_trades_per_day", best["trades_per_day"])
                        mlflow.log_metric("best_mean_pnl_ticks", best["mean_pnl_ticks"])
                        mlflow.log_param("best_entry_pct", best["entry_pct"])
                        mlflow.log_param("best_exit_pct", best["exit_pct"])

                print(f"  MLflow logged for h={horizon_s}s")
            except Exception as e:
                print(f"  MLflow logging failed: {e}")

        all_results[horizon_s] = {
            "concat_spearman": results["concat_spearman"],
            "top_decile_mean": results["top_decile_mean"],
            "bot_decile_mean": results["bot_decile_mean"],
            "n_folds": results["n_folds"],
            "n_oos_events": results["n_oos_events"],
            "train_time_s": train_time,
            "backtest": bt_results,
        }

        del results
        gc.collect()

    # ─── Final summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for h, r in all_results.items():
        print(f"\n  h={h}s:")
        print(f"    Concat Spearman: {r['concat_spearman']:.4f}")
        print(f"    Top decile: {r['top_decile_mean']:.2f}t | Bot decile: {r['bot_decile_mean']:.2f}t")
        print(f"    Folds: {r['n_folds']} | OOS events: {r['n_oos_events']:,} | Time: {r['train_time_s']:.0f}s")
        if r["backtest"]:
            best = max(r["backtest"], key=lambda x: x["daily_sharpe"])
            print(f"    Best: E=p{best['entry_pct']} X=p{best['exit_pct']} | "
                  f"{best['trades_per_day']:.1f}/d | pnl={best['mean_pnl_ticks']:.2f}t | "
                  f"WR={best['wr']:.1%} | PF={best['pf']:.2f} | dSharpe={best['daily_sharpe']:.2f}")

    print(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()


# Need these at module level for main() print
events_per_day_target = 50_000

if __name__ == "__main__":
    main()
