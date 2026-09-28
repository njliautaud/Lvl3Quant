#!/usr/bin/env python
"""
Direct first-passage v2 — MLP GPU on extended feature set.

Same data and WF as the XGB sibling. 3-layer MLP (256->128->64), BN, Dropout 0.2, GELU,
AdamW lr=1e-3, weight_decay=1e-4, batch=4096, 30 epochs early-stop patience 5 on val AUC.

Outputs:
- per_head_oos.parquet
- training_log.json
- mlflow runs (experiment: direct_firstpassage_v2_mlp)
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

try:
    import mlflow
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

HEADS = [(2,1),(3,1),(4,1),(5,1),(3,2),(4,2),(5,2),(5,4)]
HOLD_CAP_NS = 15_000_000_000

ENG_COLS = ["y_pred_adverse","y_pred_mfe","y_pred_toxicity","side",
            "adv_x_side","mfe_x_side","tox_x_side","mfe_minus_adv","mfe_minus_adv_x_side"]


def label_first_passage(df: pd.DataFrame, K: int, S: int, hold_cap_ns: int = HOLD_CAP_NS) -> np.ndarray:
    tp = df[f"tp{K}_dt_ns"].values
    sl = df[f"sl{S}_dt_ns"].values
    tp_hit = (tp > 0) & (tp <= hold_cap_ns)
    sl_hit = (sl > 0) & (sl <= hold_cap_ns)
    return (tp_hit & ((~sl_hit) | (tp < sl))).astype(np.int8)


def select_features(Xtr, ytr, top_k, eng_count, rng=42):
    if Xtr.shape[1] <= eng_count:
        return list(range(Xtr.shape[1]))
    eng_idx = list(range(eng_count))
    bf_idx = list(range(eng_count, Xtr.shape[1]))
    take = min(len(ytr), 30000)
    sel = np.random.default_rng(rng).choice(len(ytr), size=take, replace=False)
    mi = mutual_info_classif(Xtr[sel][:, bf_idx], ytr[sel], random_state=rng, n_neighbors=3)
    order = np.argsort(mi)[::-1][:top_k]
    return eng_idx + [bf_idx[i] for i in order]


class MLP(nn.Module):
    def __init__(self, d_in, hidden=(256,128,64), p=0.2):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_fold(Xtr, ytr, Xval, yval, Xte, device, batch=4096, epochs=30, patience=5, lr=1e-3, wd=1e-4, pos_weight=1.0):
    d_in = Xtr.shape[1]
    model = MLP(d_in).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pw = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).float()
    ds = TensorDataset(Xtr_t, ytr_t)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))

    Xval_t = torch.from_numpy(Xval).float().to(device)
    yval_np = yval.astype(np.float32)
    Xte_t = torch.from_numpy(Xte).float().to(device)

    best_auc = -1.0; best_state = None; bad = 0; best_epoch = -1
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(Xval_t).cpu().numpy()
        try:
            v_auc = roc_auc_score(yval_np, val_logits)
        except Exception:
            v_auc = float("nan")
        if v_auc > best_auc:
            best_auc = v_auc; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; bad = 0; best_epoch = ep
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(Xte_t).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-test_logits))
    return probs.astype(np.float32), {"best_epoch": int(best_epoch), "best_val_auc": float(best_auc)}


def train_head(df, K, S, oot_dates, device, feature_cols, top_k_bf, eng_count, batch, epochs, patience, lr, wd):
    y_full = label_first_passage(df, K, S)
    X_full = df[feature_cols].values.astype(np.float32)
    dates = df["oot_date"].values

    burn_in = 4
    records = []; fold_logs = []
    for i, d in enumerate(oot_dates):
        if i < burn_in:
            continue
        train_dates_window = oot_dates[max(0, i - 20):i]
        tr_mask = np.isin(dates, train_dates_window)
        te_mask = dates == d
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue
        Xtr_all, ytr = X_full[tr_mask], y_full[tr_mask]
        Xte_all, yte = X_full[te_mask], y_full[te_mask]

        sel = select_features(Xtr_all, ytr, top_k=top_k_bf, eng_count=eng_count)
        Xtr_sel = Xtr_all[:, sel]; Xte_sel = Xte_all[:, sel]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr_sel).astype(np.float32)
        Xte_s = scaler.transform(Xte_sel).astype(np.float32)
        # nan/inf guard
        Xtr_s = np.nan_to_num(Xtr_s, nan=0.0, posinf=0.0, neginf=0.0)
        Xte_s = np.nan_to_num(Xte_s, nan=0.0, posinf=0.0, neginf=0.0)

        rng = np.random.default_rng(42 + i)
        perm = rng.permutation(len(Xtr_s))
        n_val = max(int(0.15 * len(Xtr_s)), 2000)
        val_idx = perm[:n_val]; tr_idx = perm[n_val:]
        Xtr_f = Xtr_s[tr_idx]; ytr_f = ytr[tr_idx]
        Xval_f = Xtr_s[val_idx]; yval_f = ytr[val_idx]

        pos = ytr_f.sum(); neg = len(ytr_f) - pos
        pw = float(neg / max(pos, 1)) if pos > 0 else 1.0

        probs, info = train_one_fold(Xtr_f, ytr_f, Xval_f, yval_f, Xte_s, device,
                                     batch=batch, epochs=epochs, patience=patience, lr=lr, wd=wd, pos_weight=pw)

        df_te = df.loc[te_mask, ["event_id","ts_ns","oot_date","fold_id","side"]].copy()
        df_te["K"] = K; df_te["S"] = S
        df_te["y_true_fp"] = yte
        df_te["y_pred_fp"] = probs
        df_te["fold_oot_idx"] = i
        records.append(df_te)
        try:
            fauc = roc_auc_score(yte, probs)
        except Exception:
            fauc = float("nan")
        fold_logs.append({"oot_idx": i, "date": d, "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
                          "fold_auc": float(fauc), **info,
                          "selected_features": [feature_cols[k] for k in sel]})

    if not records:
        return pd.DataFrame(), {"folds": fold_logs}
    return pd.concat(records, ignore_index=True), {"folds": fold_logs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks-extended", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--top-k-bf", type=int, default=10)
    ap.add_argument("--mlflow-uri", default="")
    ap.add_argument("--mlflow-experiment", default="direct_firstpassage_v2_mlp")
    ap.add_argument("--heads", default="all")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if HAS_MLFLOW and args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading extended walks...", flush=True)
    df = pd.read_parquet(args.walks_extended)
    bf_cols = [c for c in df.columns if c.startswith("bf_")]
    feature_cols = ENG_COLS + bf_cols
    eng_count = len(ENG_COLS)
    print(f"  rows={len(df)} eng={eng_count} bf={len(bf_cols)}", flush=True)

    oot_dates = sorted(df["oot_date"].unique().tolist())
    print(f"  oot_dates n={len(oot_dates)}", flush=True)

    head_list = HEADS if args.heads == "all" else [tuple(int(x) for x in t.split("_")) for t in args.heads.split(",")]

    all_oos = []
    log = {"heads": [], "device": args.device, "batch": args.batch, "epochs": args.epochs,
           "patience": args.patience, "lr": args.lr, "wd": args.wd, "top_k_bf": args.top_k_bf,
           "n_rows": len(df), "feature_cols_available": feature_cols}

    for K, S in head_list:
        t_h = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] === head K={K} S={S} ===", flush=True)
        run = mlflow.start_run(run_name=f"K{K}_S{S}") if (HAS_MLFLOW and args.mlflow_uri) else None
        try:
            if run is not None:
                mlflow.log_params({"K": K, "S": S, "batch": args.batch, "epochs": args.epochs,
                                   "patience": args.patience, "lr": args.lr, "wd": args.wd,
                                   "top_k_bf": args.top_k_bf, "device": args.device})
            oos, head_log = train_head(df, K, S, oot_dates, args.device, feature_cols,
                                       args.top_k_bf, eng_count, args.batch, args.epochs,
                                       args.patience, args.lr, args.wd)
            if oos.empty:
                print("  EMPTY"); continue
            try:
                auc = roc_auc_score(oos["y_true_fp"], oos["y_pred_fp"])
                ap_ = average_precision_score(oos["y_true_fp"], oos["y_pred_fp"])
            except Exception:
                auc = float("nan"); ap_ = float("nan")
            base = float(oos["y_true_fp"].mean())
            wall = time.time() - t_h
            print(f"  done {wall:.1f}s n={len(oos)} base={base:.4f} AUC={auc:.4f} AP={ap_:.4f}", flush=True)
            log["heads"].append({"K": K, "S": S, "n_oos": int(len(oos)), "base_rate": base,
                                 "auc": float(auc), "ap": float(ap_), "wall_s": wall, "folds": head_log["folds"]})
            all_oos.append(oos)
            if run is not None:
                mlflow.log_metrics({"auc_oos": float(auc), "ap_oos": float(ap_), "base_rate": float(base),
                                    "n_oos": int(len(oos)), "wall_s": float(wall)})
        finally:
            if run is not None:
                mlflow.end_run()

    if all_oos:
        pd.concat(all_oos, ignore_index=True).to_parquet(out_dir / "per_head_oos.parquet", index=False)
    log["total_wall_s"] = time.time() - t0
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2, default=float))
    print(f"[{time.strftime('%H:%M:%S')}] done wall={log['total_wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
