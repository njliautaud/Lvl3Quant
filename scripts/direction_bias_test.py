"""Test if our direction classification is real signal or just learning market bias."""
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import lightgbm as lgb

cache_dir = Path("data/processed/dl_book_cache")
day_files = sorted(cache_dir.glob("*_book_tensors.npz"))
dates = [f.name.split("_book_tensors")[0] for f in day_files]
train_dates = dates[:60]
oot_dates = dates[60:80]

print(f"Train: {train_dates[0]}..{train_dates[-1]} ({len(train_dates)}d)")
print(f"OOT: {oot_dates[0]}..{oot_dates[-1]} ({len(oot_dates)}d)")

def load_features(date_list):
    all_X, all_mids, all_bounds = [], [], []
    for d in date_list:
        f = cache_dir / f"{d}_book_tensors.npz"
        data = np.load(str(f))
        tensor = data["book_tensors"]
        mid = data["mid_prices"]
        bid = tensor[:, :10, :]
        ask = tensor[:, 10:, :]
        feats = []
        for side in [bid, ask]:
            for ch in range(4):
                feats.append(side[:, :, ch].mean(axis=1))
                feats.append(side[:, :, ch].std(axis=1))
        spread = ask[:, 0, 0] - bid[:, 0, 0]
        feats.append(spread)
        bid_depth = bid[:, :, 1].sum(axis=1)
        ask_depth = ask[:, :, 1].sum(axis=1)
        feats.append((bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-8))
        X = np.column_stack(feats)
        all_X.append(X)
        all_mids.append(mid)
        all_bounds.append(len(mid))
    return np.concatenate(all_X), np.concatenate(all_mids), all_bounds

def compute_mfe(mids, bounds, horizon=100):
    n = len(mids)
    target = np.full(n, np.nan)
    cum = 0
    for b in bounds:
        for i in range(cum, cum + b - horizon):
            fwd = mids[i+1:i+1+horizon]
            mfe_l = max(0, (fwd.max() - mids[i]) / 0.25)
            mfe_s = max(0, (mids[i] - fwd.min()) / 0.25)
            target[i] = mfe_l - mfe_s
        cum += b
    return target

X_train, mids_train, bounds_train = load_features(train_dates)
X_oot, mids_oot, bounds_oot = load_features(oot_dates)

for horizon, label in [(100, "10s"), (300, "30s")]:
    y_train_raw = compute_mfe(mids_train, bounds_train, horizon)
    y_oot_raw = compute_mfe(mids_oot, bounds_oot, horizon)
    mask_tr = np.isfinite(y_train_raw)
    mask_ot = np.isfinite(y_oot_raw)

    y_train_dir = (y_train_raw[mask_tr] > 0).astype(int)
    y_oot_dir = (y_oot_raw[mask_ot] > 0).astype(int)

    # BASE RATE
    train_base = y_train_dir.mean()
    oot_base = y_oot_dir.mean()

    # Train classifier
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
              "num_leaves": 64, "min_data_in_leaf": 1000, "verbose": -1,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1}
    dtrain = lgb.Dataset(X_train[mask_tr], y_train_dir)
    dval = lgb.Dataset(X_oot[mask_ot], y_oot_dir, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=200, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    pred = model.predict(X_oot[mask_ot])
    pred_binary = (pred > 0.5).astype(int)

    acc = accuracy_score(y_oot_dir, pred_binary)
    auc = roc_auc_score(y_oot_dir, pred)
    cm = confusion_matrix(y_oot_dir, pred_binary)
    pred_up_pct = pred_binary.mean()
    naive_acc = max(oot_base, 1 - oot_base)

    up_mask = y_oot_dir == 1
    down_mask = y_oot_dir == 0
    acc_when_up = pred_binary[up_mask].mean() if up_mask.sum() > 0 else 0
    acc_when_down = (1 - pred_binary[down_mask]).mean() if down_mask.sum() > 0 else 0

    print(f"\n=== {label} HORIZON ===")
    print(f"Base rate (pct bullish): Train={train_base:.3f} OOT={oot_base:.3f}")
    print(f"Naive baseline (always predict majority): {naive_acc:.3f}")
    print(f"Model accuracy: {acc:.3f} (lift over naive: {acc - naive_acc:+.3f})")
    print(f"Model AUC: {auc:.3f}")
    print(f"Model predicts UP: {pred_up_pct:.3f} vs actual UP rate: {oot_base:.3f}")
    print(f"Accuracy when market IS up: {acc_when_up:.3f}")
    print(f"Accuracy when market IS down: {acc_when_down:.3f}")
    print(f"Confusion: TN={cm[0][0]:,} FP={cm[0][1]:,} FN={cm[1][0]:,} TP={cm[1][1]:,}")

    if pred_up_pct > 0.9:
        print("VERDICT: Model always says UP - just learning bull bias!")
    elif abs(pred_up_pct - oot_base) < 0.03:
        print("VERDICT: Prediction rate matches base rate - possible bias")
    else:
        print("VERDICT: Prediction rate differs from base rate - may have real signal")
