
import numpy as np
import json
import time
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score

cache_dir = Path("/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot")
day_files = sorted(cache_dir.glob("*_book_tensors.npz"))
dates = [f.name.split("_book_tensors")[0] for f in day_files]

train_dates = dates[:40]
oot_dates = dates[40:55]

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
        for side, name in [(bid, "bid"), (ask, "ask")]:
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

import lightgbm as lgb

X_train, mids_train, bounds_train = load_features(train_dates)
X_oot, mids_oot, bounds_oot = load_features(oot_dates)

results = {}
for horizon, label in [(100, "10s"), (300, "30s"), (600, "1min")]:
    t0 = time.time()
    y_train_raw = compute_mfe(mids_train, bounds_train, horizon)
    y_oot_raw = compute_mfe(mids_oot, bounds_oot, horizon)
    
    # REGRESSION (current approach)
    mask_tr = np.isfinite(y_train_raw)
    mask_ot = np.isfinite(y_oot_raw)
    
    params_reg = {"objective": "regression", "metric": "mse", "learning_rate": 0.05,
                  "num_leaves": 64, "min_data_in_leaf": 1000, "verbose": -1,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1}
    
    dtrain = lgb.Dataset(X_train[mask_tr], y_train_raw[mask_tr])
    dval = lgb.Dataset(X_oot[mask_ot], y_oot_raw[mask_ot], reference=dtrain)
    model_reg = lgb.train(params_reg, dtrain, num_boost_round=200, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    pred_reg = model_reg.predict(X_oot[mask_ot])
    ic = spearmanr(pred_reg, y_oot_raw[mask_ot])[0]
    
    # CLASSIFICATION (direction only)
    y_train_dir = (y_train_raw > 0).astype(int)
    y_oot_dir = (y_oot_raw > 0).astype(int)
    
    params_cls = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
                  "num_leaves": 64, "min_data_in_leaf": 1000, "verbose": -1,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1}
    
    dtrain_c = lgb.Dataset(X_train[mask_tr], y_train_dir[mask_tr])
    dval_c = lgb.Dataset(X_oot[mask_ot], y_oot_dir[mask_ot], reference=dtrain_c)
    model_cls = lgb.train(params_cls, dtrain_c, num_boost_round=200, valid_sets=[dval_c],
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    pred_cls = model_cls.predict(X_oot[mask_ot])
    auc = roc_auc_score(y_oot_dir[mask_ot], pred_cls)
    acc = accuracy_score(y_oot_dir[mask_ot], (pred_cls > 0.5).astype(int))
    
    elapsed = time.time() - t0
    print(f"{label}: Regression IC={ic:.4f} | Classification AUC={auc:.4f} Acc={acc:.3f} | {elapsed:.0f}s")
    results[label] = {"ic": ic, "auc": auc, "accuracy": acc}

print()
print(json.dumps(results, indent=2))
