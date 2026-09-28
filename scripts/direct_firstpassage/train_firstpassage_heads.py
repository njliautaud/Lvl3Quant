#!/usr/bin/env python
"""
Direct first-passage binary classifier training.

For each (K, S) pair, label = 1 iff TP_K_dt_ns < SL_S_dt_ns within hold cap (15s).
Walk-forward, sliding train window. XGBoost GPU.

Outputs:
- per_head_oos.parquet: OOS predictions for every (head_id, fold, event_id)
- per_head_models/{K}_{S}_fold{N}.json  (optional)
- training_log.json
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HEADS = [(2,1),(3,1),(4,1),(5,1),(3,2),(4,2),(5,2),(5,4)]
HOLD_CAP_NS = 15_000_000_000  # 15s

FEATURE_COLS = ["y_pred_adverse", "y_pred_mfe", "y_pred_toxicity", "side"]
# Optional engineered features from preds
def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["adv_x_side"] = df["y_pred_adverse"] * df["side"]
    df["mfe_x_side"] = df["y_pred_mfe"] * df["side"]
    df["tox_x_side"] = df["y_pred_toxicity"] * df["side"]
    df["mfe_minus_adv"] = df["y_pred_mfe"] - df["y_pred_adverse"]
    df["mfe_minus_adv_x_side"] = df["mfe_minus_adv"] * df["side"]
    return df

ENG_COLS = FEATURE_COLS + ["adv_x_side","mfe_x_side","tox_x_side","mfe_minus_adv","mfe_minus_adv_x_side"]

def label_first_passage(df: pd.DataFrame, K: int, S: int, hold_cap_ns: int = HOLD_CAP_NS) -> np.ndarray:
    tp = df[f"tp{K}_dt_ns"].values
    sl = df[f"sl{S}_dt_ns"].values
    tp_hit = (tp > 0) & (tp <= hold_cap_ns)
    sl_hit = (sl > 0) & (sl <= hold_cap_ns)
    # Win iff TP hit and (no SL hit OR TP_ns < SL_ns)
    win = tp_hit & ((~sl_hit) | (tp < sl))
    return win.astype(np.int8)

def train_head(df_full: pd.DataFrame, K: int, S: int, oot_dates: list[str], device: str,
               feature_cols: list[str], n_estimators: int = 400, save_models_dir: Path | None = None) -> pd.DataFrame:
    """Walk-forward over OOT dates. For OOT date d (idx i), train on all rows with oot_date < d (sliding via fold_id alt: simpler — use all prior OOT-bucket rows).

    Per task: use first 4 OOT dates as burn-in train; evaluate folds 5..end.
    Simpler & robust: at OOT date i (i >= 4), train on rows whose oot_date is among oot_dates[0..i-1]; predict on rows whose oot_date == oot_dates[i].
    This is sliding-equivalent because the per_trade_walks parquet already represents fold-OOT rows (each row is an OOT event from its own fold's training window).
    """
    y_full = label_first_passage(df_full, K, S)
    X_full = df_full[feature_cols].values.astype(np.float32)
    dates_full = df_full["oot_date"].values

    oos_records = []
    burn_in = 4
    pos_weight_cache = []

    for i, d in enumerate(oot_dates):
        if i < burn_in:
            continue
        train_mask = np.isin(dates_full, oot_dates[:i])
        test_mask = dates_full == d
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        Xtr, ytr = X_full[train_mask], y_full[train_mask]
        Xte, yte = X_full[test_mask], y_full[test_mask]

        # Class balance
        pos = ytr.sum()
        neg = len(ytr) - pos
        spw = (neg / max(pos, 1)) if pos > 0 else 1.0
        pos_weight_cache.append(spw)

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        model.fit(Xtr, ytr)
        yhat = model.predict_proba(Xte)[:, 1]

        df_test = df_full.loc[test_mask, ["event_id", "ts_ns", "oot_date", "fold_id", "side"]].copy()
        df_test["K"] = K
        df_test["S"] = S
        df_test["y_true_fp"] = yte
        df_test["y_pred_fp"] = yhat.astype(np.float32)
        df_test["fold_oot_idx"] = i
        oos_records.append(df_test)

        if save_models_dir is not None:
            save_models_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(save_models_dir / f"K{K}_S{S}_oot{i:02d}_{d}.json"))

    if not oos_records:
        return pd.DataFrame()
    out = pd.concat(oos_records, ignore_index=True)
    out.attrs["mean_spw"] = float(np.mean(pos_weight_cache)) if pos_weight_cache else 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk-parquet", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--heads", default="all", help="all | smoke | comma list of K_S e.g. 4_1,3_1")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--engineered", action="store_true", help="Add engineered crosses")
    ap.add_argument("--smoke-folds", type=int, default=0, help="If >0, only run this many OOT folds after burn-in (for smoke test)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading walk parquet...", flush=True)
    df = pd.read_parquet(args.walk_parquet)
    print(f"  rows={len(df)}  cols={len(df.columns)}", flush=True)

    feature_cols = ENG_COLS if args.engineered else FEATURE_COLS
    if args.engineered:
        df = add_engineered(df)
    print(f"  features: {feature_cols}", flush=True)

    oot_dates = sorted(df["oot_date"].unique().tolist())
    print(f"  oot_dates={oot_dates}  n={len(oot_dates)}", flush=True)

    if args.heads == "all":
        head_list = HEADS
    elif args.heads == "smoke":
        head_list = [(4, 1)]
    else:
        head_list = []
        for tok in args.heads.split(","):
            K, S = tok.split("_")
            head_list.append((int(K), int(S)))
    print(f"  heads to train: {head_list}", flush=True)

    if args.smoke_folds > 0:
        oot_dates_used = oot_dates[: 4 + args.smoke_folds]
        print(f"  SMOKE: limiting to first {len(oot_dates_used)} dates ({args.smoke_folds} OOS folds after 4 burn-in)", flush=True)
    else:
        oot_dates_used = oot_dates

    df_used = df[df["oot_date"].isin(oot_dates_used)].reset_index(drop=True)
    print(f"  used rows: {len(df_used)}", flush=True)

    all_oos = []
    log = {"heads": [], "device": args.device, "n_estimators": args.n_estimators,
           "n_rows": len(df_used), "feature_cols": feature_cols, "engineered": args.engineered}

    models_dir = out_dir / "per_head_models" if args.save_models else None

    for K, S in head_list:
        t_h = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] === head (K={K}, S={S}) ===", flush=True)
        oos = train_head(df_used, K, S, oot_dates_used, args.device, feature_cols,
                         n_estimators=args.n_estimators, save_models_dir=models_dir)
        if oos.empty:
            print(f"  EMPTY oos for (K={K},S={S})", flush=True)
            continue
        # Quick metric: AUC-style — rank correlation of pred vs true.
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            auc = roc_auc_score(oos["y_true_fp"], oos["y_pred_fp"])
            ap_ = average_precision_score(oos["y_true_fp"], oos["y_pred_fp"])
        except Exception:
            auc = float("nan"); ap_ = float("nan")
        base = float(oos["y_true_fp"].mean())
        print(f"  done in {time.time()-t_h:.1f}s  n_oos={len(oos)}  base_rate={base:.4f}  AUC={auc:.4f}  AP={ap_:.4f}", flush=True)
        log["heads"].append({"K": K, "S": S, "n_oos": int(len(oos)), "base_rate": base, "auc": float(auc), "ap": float(ap_), "wall_s": time.time()-t_h})
        all_oos.append(oos)

    if all_oos:
        oos_all = pd.concat(all_oos, ignore_index=True)
        oos_path = out_dir / "per_head_oos.parquet"
        oos_all.to_parquet(oos_path, index=False)
        print(f"[{time.strftime('%H:%M:%S')}] wrote {oos_path}  rows={len(oos_all)}", flush=True)
    else:
        print("NO OOS produced!", flush=True)

    log["total_wall_s"] = time.time() - t0
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] total wall {log['total_wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
