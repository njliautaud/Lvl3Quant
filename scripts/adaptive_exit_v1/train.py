#!/usr/bin/env python3
"""
ADAPTIVE-EXIT POLICY v1 — supervised imitation learning toward hindsight-optimal exit.

Data source: /home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/mfe_mae_analysis/fold_*_mfe_mae_10s.npz
Labels: 6-class discretization of time_to_mfe_s with "never" bucket for losing/marginal trades.
Features: as-of-safe (NO future-leakage): pred_value, |pred_value|, direction, plus rolling
  statistics over the PRIOR N events (mean/std of pred, realized hit-rates from past trades).

Constraints honored:
  HC #0: SLIDING window walk-forward only (13 folds)
  HC #495: as-of features only; rolling stats use STRICTLY past events
  HC #498 R4: honest reporting, no spin on marginal results
"""

import os, sys, json, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import mlflow

# ----- CONFIG -----
DATA_GLOB = "/home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/mfe_mae_analysis/fold_*_mfe_mae_10s.npz"
OUT_DIR = "/home/nick/Lvl3Quant/data/processed/adaptive_exit_v1"
RESULTS_DIR = "/home/nick/Lvl3Quant/output/adaptive_exit_v1"
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

# Label discretization on time_to_mfe_s; bucket "never" if MFE <= MIN_PROFIT_TICKS
# Cost to take a trade: 0.376 ticks RT (HC docs). Profitable MFE needs > 1 tick to be tradeable.
MIN_PROFIT_TICKS = 1.0
TIME_BINS = [0.5, 2.5, 7.5, 15.0, 30.01]   # boundaries: 0=exit_now, 1=~1s, 2=~5s, 3=~10s, 4=~30s
CLASS_NAMES = ["exit_now", "exit_1s", "exit_5s", "exit_10s", "exit_30s", "never"]
N_CLASSES = 6
N_FOLDS = 13
DEVICE = torch.device("cuda:0")
SEED = 1337
torch.manual_seed(SEED); np.random.seed(SEED)

# ----- LOAD ALL DATA, SORT BY TIME -----
print("[load] reading fold files...")
all_files = sorted(glob.glob(DATA_GLOB))
print(f"[load] {len(all_files)} files found")
arrs = {"pred":[], "dir":[], "mfe":[], "mae":[], "t2mfe":[], "t2mae":[], "ts":[], "label_val":[]}
for fp in all_files:
    f = np.load(fp)
    arrs["pred"].append(f["pred_values"].astype(np.float32))
    arrs["dir"].append(f["direction"].astype(np.float32))
    arrs["mfe"].append(f["mfe_ticks"].astype(np.float32))
    arrs["mae"].append(f["mae_ticks"].astype(np.float32))
    arrs["t2mfe"].append(f["time_to_mfe_s"].astype(np.float32))
    arrs["t2mae"].append(f["time_to_mae_s"].astype(np.float32))
    arrs["ts"].append(f["timestamps_ns"].astype(np.int64))
    arrs["label_val"].append(f["label_values"].astype(np.float32))
for k in arrs: arrs[k] = np.concatenate(arrs[k])

# Folds OVERLAP in time (WF CV outputs). De-dup on timestamp_ns + pred sign.
# Easier: sort by ts and keep unique (ts_ns, direction) pairs.
order = np.argsort(arrs["ts"], kind="stable")
for k in arrs: arrs[k] = arrs[k][order]
# de-dup
keys = arrs["ts"] * 4 + ((arrs["dir"].astype(np.int64)+1))
uniq_keys, uniq_idx = np.unique(keys, return_index=True)
uniq_idx.sort()
for k in arrs: arrs[k] = arrs[k][uniq_idx]
N = len(arrs["ts"])
print(f"[load] N events after de-dup & sort: {N}")

# ----- LABELS -----
mfe = arrs["mfe"]; t2mfe = arrs["t2mfe"]
labels = np.full(N, 5, dtype=np.int64)  # default "never"
profitable = mfe > MIN_PROFIT_TICKS
# digitize t2mfe into bins
bin_idx = np.digitize(t2mfe, TIME_BINS).astype(np.int64)  # 0..5
# only profitable trades get a real exit bucket; rest = 5 ("never")
labels[profitable] = bin_idx[profitable]
labels = np.clip(labels, 0, 5)

print("[labels] class balance:")
for c in range(N_CLASSES):
    cnt = (labels==c).sum()
    print(f"  {c} {CLASS_NAMES[c]}: {cnt} ({cnt/N*100:.1f}%)")

# ----- FEATURES (as-of safe) -----
# Features at decision time = available without future info:
#   pred, |pred|, direction
#   pred^2, sign(pred)
#   pred z-score over PRIOR 200 events
#   prior_mean_mfe / prior_mean_mae (window 50, strictly before idx)
#   prior_hit_rate (frac of past 200 events with mfe>1)
#   prior_mean_t2mfe (50)
#   time-of-day features (cyclical)
# Use causal cumulative-style rolling — at index i use [i-W:i] only.

pred = arrs["pred"].astype(np.float64)
direction = arrs["dir"].astype(np.float64)

def causal_rolling_mean(x, w):
    """At index i return mean of x[max(0,i-w):i] (strictly past)."""
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    out = np.zeros_like(x)
    for i in range(len(x)):
        lo = max(0, i-w); hi = i
        if hi > lo:
            out[i] = (cs[hi] - cs[lo]) / (hi - lo)
        else:
            out[i] = 0.0
    return out

def causal_rolling_std(x, w):
    m = causal_rolling_mean(x, w)
    m2 = causal_rolling_mean(x*x, w)
    v = np.maximum(m2 - m*m, 1e-12)
    return np.sqrt(v)

print("[feat] computing causal rolling features (may take ~30s)...")
t0 = time.time()
pred_mean_200 = causal_rolling_mean(pred, 200)
pred_std_200  = causal_rolling_std (pred, 200)
prior_mfe_50  = causal_rolling_mean(mfe.astype(np.float64), 50)
prior_mae_50  = causal_rolling_mean(arrs["mae"].astype(np.float64), 50)
prior_hit_200 = causal_rolling_mean(profitable.astype(np.float64), 200)
prior_t2mfe_50= causal_rolling_mean(t2mfe.astype(np.float64), 50)
prior_mfe_200 = causal_rolling_mean(mfe.astype(np.float64), 200)
prior_pred_mfe_corr_200 = causal_rolling_mean((pred*mfe.astype(np.float64)), 200)
print(f"[feat] rolling computed in {time.time()-t0:.1f}s")

# z-scored pred (causal)
pred_z = (pred - pred_mean_200) / (pred_std_200 + 1e-6)

# time of day from ts_ns
ts_s = arrs["ts"] / 1e9
seconds_of_day = ts_s.astype(np.float64) % 86400.0
tod_sin = np.sin(2*np.pi*seconds_of_day/86400.0)
tod_cos = np.cos(2*np.pi*seconds_of_day/86400.0)

abspred = np.abs(pred)
signpred = np.sign(pred)

# inter-event arrival rate proxy: dt to previous event
dt = np.diff(ts_s, prepend=ts_s[0])
log_dt = np.log1p(np.maximum(dt, 0))

X = np.column_stack([
    pred, abspred, signpred, direction,
    pred*pred,
    pred_z,
    pred_mean_200, pred_std_200,
    prior_mfe_50, prior_mae_50, prior_mfe_200,
    prior_hit_200, prior_t2mfe_50,
    prior_pred_mfe_corr_200,
    tod_sin, tod_cos,
    log_dt,
    pred*direction,           # signed-against-direction interaction
    abspred*prior_hit_200,    # confidence × recent regime
    abspred*prior_mfe_200,    # confidence × recent payoff magnitude
]).astype(np.float32)
N_FEAT = X.shape[1]
print(f"[feat] X shape: {X.shape}, n_features: {N_FEAT}")

# nan/inf check
n_bad = np.isnan(X).sum() + np.isinf(X).sum()
print(f"[feat] bad values: {n_bad}")
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

y = labels.astype(np.int64)
ts_ns = arrs["ts"]

# Save train/val (concat OOT folds) for record
np.savez_compressed(os.path.join(OUT_DIR, "all.npz"),
                    X=X, y=y, ts_ns=ts_ns,
                    mfe=mfe, mae=arrs["mae"], t2mfe=t2mfe,
                    feature_names=np.array([
                        "pred","abspred","signpred","direction","pred_sq",
                        "pred_z","pred_mean_200","pred_std_200",
                        "prior_mfe_50","prior_mae_50","prior_mfe_200",
                        "prior_hit_200","prior_t2mfe_50",
                        "prior_pred_mfe_corr_200",
                        "tod_sin","tod_cos","log_dt",
                        "pred_x_dir","conf_x_hit","conf_x_payoff"
                    ]))
print(f"[save] {OUT_DIR}/all.npz")

# ----- MODEL -----
class ExitMLP(nn.Module):
    def __init__(self, n_in, n_out=6, hidden=(64,32)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden[0]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden[1], n_out),
        )
    def forward(self, x): return self.net(x)

# ----- SLIDING-WINDOW WF CV -----
# Split chronologically into 13 equal blocks. Fold k: train on prior train_blocks blocks, OOT = block k.
# SLIDING window: train_blocks = ceil(N_FOLDS/2) = 7 blocks preceding the OOT block.
TOTAL_BLOCKS = N_FOLDS + 7  # 7 warmup blocks for sliding train, 13 OOT
# Simpler: blocks = N_FOLDS+1; OOT_block_idx in [1..N_FOLDS]; sliding train = max(0, idx-TRAIN_BLOCKS) .. idx-1
TRAIN_BLOCKS = 5  # sliding train window
N_BLOCKS = N_FOLDS + TRAIN_BLOCKS  # so we have enough warmup
block_edges = np.linspace(0, N, N_BLOCKS+1).astype(np.int64)
print(f"[cv] N_BLOCKS={N_BLOCKS}, TRAIN_BLOCKS_SLIDING={TRAIN_BLOCKS}, N_FOLDS={N_FOLDS}")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("adaptive_exit_v1")

all_oot_y = []
all_oot_pred = []
per_fold = []

run_name = f"adaptive_exit_v1_{int(time.time())}"
with mlflow.start_run(run_name=run_name) as parent_run:
    mlflow.log_params({
        "n_features": N_FEAT, "n_classes": N_CLASSES,
        "n_folds": N_FOLDS, "train_blocks_sliding": TRAIN_BLOCKS,
        "n_events_total": N,
        "min_profit_ticks": MIN_PROFIT_TICKS,
        "time_bins": str(TIME_BINS),
        "hidden": "64,32", "dropout": 0.1,
        "seed": SEED,
    })

    for fold in range(N_FOLDS):
        oot_block_idx = TRAIN_BLOCKS + fold  # 5..17 for 13 folds
        train_lo_b = oot_block_idx - TRAIN_BLOCKS  # sliding window
        train_hi_b = oot_block_idx
        tr_lo, tr_hi = block_edges[train_lo_b], block_edges[train_hi_b]
        oot_lo, oot_hi = block_edges[oot_block_idx], block_edges[oot_block_idx+1]

        Xtr, ytr = X[tr_lo:tr_hi], y[tr_lo:tr_hi]
        Xte, yte = X[oot_lo:oot_hi], y[oot_lo:oot_hi]

        # standardize using train stats only
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd<1e-6] = 1.0
        Xtr_n = (Xtr - mu)/sd
        Xte_n = (Xte - mu)/sd

        # class weights (inverse freq) to combat imbalance
        cls_counts = np.bincount(ytr, minlength=N_CLASSES).astype(np.float32)
        cls_w = (cls_counts.sum() / (N_CLASSES * (cls_counts + 1.0)))
        cls_w_t = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)

        Xtr_t = torch.tensor(Xtr_n, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.long)
        Xte_t = torch.tensor(Xte_n, dtype=torch.float32, device=DEVICE)
        yte_t = torch.tensor(yte, dtype=torch.long, device=DEVICE)

        dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=4096, shuffle=True, num_workers=2, pin_memory=True)
        model = ExitMLP(N_FEAT, N_CLASSES, hidden=(64,32)).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss(weight=cls_w_t)

        EPOCHS = 12
        model.train()
        for ep in range(EPOCHS):
            tot_loss = 0.0; n_batch = 0
            for xb, yb in dl:
                xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
                opt.zero_grad()
                logits = model(xb)
                loss = ce(logits, yb)
                loss.backward()
                opt.step()
                tot_loss += loss.item(); n_batch += 1

        # eval
        model.eval()
        with torch.no_grad():
            logits = model(Xte_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

        acc = (preds == yte).mean()
        # baseline: predict majority train class
        maj = int(np.argmax(cls_counts))
        baseline_acc = (yte == maj).mean()
        # top-2 accuracy
        top2 = np.argsort(-probs, axis=1)[:, :2]
        top2_acc = (top2 == yte[:,None]).any(axis=1).mean()

        # predicted class distribution (sanity gate)
        pred_dist = np.bincount(preds, minlength=N_CLASSES) / len(preds)

        per_fold.append({
            "fold": fold,
            "tr_lo": int(tr_lo), "tr_hi": int(tr_hi),
            "oot_lo": int(oot_lo), "oot_hi": int(oot_hi),
            "n_tr": int(tr_hi-tr_lo), "n_oot": int(oot_hi-oot_lo),
            "acc": float(acc),
            "top2_acc": float(top2_acc),
            "baseline_acc_majority": float(baseline_acc),
            "pred_dist": [float(x) for x in pred_dist],
            "true_dist": [float(x) for x in np.bincount(yte, minlength=N_CLASSES)/len(yte)],
            "final_train_loss": float(tot_loss/max(1,n_batch)),
        })
        all_oot_y.append(yte); all_oot_pred.append(preds)

        mlflow.log_metrics({
            f"fold{fold}_acc": float(acc),
            f"fold{fold}_top2_acc": float(top2_acc),
            f"fold{fold}_baseline_acc": float(baseline_acc),
        }, step=fold)
        print(f"[fold {fold}] n_tr={tr_hi-tr_lo} n_oot={oot_hi-oot_lo} acc={acc:.4f} top2={top2_acc:.4f} maj_baseline={baseline_acc:.4f} pred_dist={np.round(pred_dist,3).tolist()}")

    # concat metrics
    y_concat = np.concatenate(all_oot_y)
    p_concat = np.concatenate(all_oot_pred)
    concat_acc = (y_concat == p_concat).mean()
    random_baseline = 1.0/N_CLASSES
    majority_baseline = max(np.bincount(y_concat, minlength=N_CLASSES)/len(y_concat))

    # per-class precision/recall
    rep = classification_report(y_concat, p_concat, labels=list(range(N_CLASSES)),
                                target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_concat, p_concat, labels=list(range(N_CLASSES)))

    # sanity gate
    pred_dist_concat = np.bincount(p_concat, minlength=N_CLASSES) / len(p_concat)
    true_dist_concat = np.bincount(y_concat, minlength=N_CLASSES) / len(y_concat)
    collapsed_class = None
    for c in range(N_CLASSES):
        if pred_dist_concat[c] > 0.70 and true_dist_concat[c] < 0.70:
            collapsed_class = c

    mlflow.log_metrics({
        "concat_acc": float(concat_acc),
        "random_baseline_acc": float(random_baseline),
        "majority_baseline_acc": float(majority_baseline),
    })

    # honest verdict
    skill_gate_pp = (concat_acc - random_baseline) * 100.0
    if collapsed_class is not None:
        verdict = "REJECT_COLLAPSED"
    elif skill_gate_pp >= 10.0:
        verdict = "MEANINGFUL"
    elif skill_gate_pp >= 5.0:
        verdict = "MARGINAL"
    else:
        verdict = "NO_USABLE_SIGNAL"

    summary = {
        "run_name": run_name,
        "n_events_total": int(N),
        "n_features": int(N_FEAT),
        "class_names": CLASS_NAMES,
        "class_balance_overall": [float(x) for x in (np.bincount(y, minlength=N_CLASSES)/len(y))],
        "n_folds": N_FOLDS,
        "train_blocks_sliding": TRAIN_BLOCKS,
        "concat_acc": float(concat_acc),
        "concat_pred_dist": [float(x) for x in pred_dist_concat],
        "concat_true_dist": [float(x) for x in true_dist_concat],
        "random_baseline_acc": float(random_baseline),
        "majority_baseline_acc": float(majority_baseline),
        "skill_vs_random_pp": float(skill_gate_pp),
        "per_class_report": rep,
        "confusion_matrix": cm.tolist(),
        "collapsed_class": collapsed_class,
        "verdict": verdict,
        "per_fold": per_fold,
    }
    out_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(out_path, "w") as f: json.dump(summary, f, indent=2)
    mlflow.log_artifact(out_path)
    print("\n==== CONCAT OOT REPORT ====")
    print(json.dumps({
        "concat_acc": summary["concat_acc"],
        "random_baseline": summary["random_baseline_acc"],
        "majority_baseline": summary["majority_baseline_acc"],
        "skill_vs_random_pp": summary["skill_vs_random_pp"],
        "concat_pred_dist": summary["concat_pred_dist"],
        "concat_true_dist": summary["concat_true_dist"],
        "verdict": summary["verdict"],
    }, indent=2))
    print(f"[done] summary written to {out_path}")
