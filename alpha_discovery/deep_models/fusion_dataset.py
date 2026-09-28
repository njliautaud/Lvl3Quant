import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

OF12_DIR = "/home/jupiter/Lvl3Quant/data/processed/orderflow_features"
OF3_DIR = "/home/jupiter/Lvl3Quant/data/processed/of3_large_order"
OF4_DIR = "/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth"
CTX_DIR = "/home/jupiter/Lvl3Quant/data/processed/fusion_context"

OF12_KEYS = ["book_imbalance","cum_delta","roll_delta_10s","total_bid_size","total_ask_size"]
OF4_KEYS = ["depth_imbalance_5","depth_imbalance_10","bid_depth_5","ask_depth_5","wall_imbalance","poc_dist_ticks"]

N_OF_FEATURES = 5 + 6 + 13

def load_of_features_for_date(date_str):
    d12p = os.path.join(OF12_DIR, date_str+"_orderflow.npz")
    d4p = os.path.join(OF4_DIR, date_str+"_of4.npz")
    ctxp = os.path.join(CTX_DIR, date_str+"_fusion_context.npz")
    if not (os.path.exists(d12p) and os.path.exists(d4p) and os.path.exists(ctxp)):
        return None
    d12 = np.load(d12p)
    d4 = np.load(d4p)
    ctx = np.load(ctxp)["of_context"]
    n = len(np.load(d12p)[OF12_KEYS[0]])
    parts = []
    for k in OF12_KEYS:
        arr = d12[k].astype(np.float32)
        parts.append(arr[:n].reshape(n,1))
    for k in OF4_KEYS:
        arr = d4[k].astype(np.float32)
        parts.append(arr[:n].reshape(n,1))
    ctx_broad = np.tile(ctx, (n,1))
    parts.append(ctx_broad)
    return np.concatenate(parts, axis=1)

def load_cnn_preds_from_npz(npz_path, dates):
    d = np.load(npz_path)
    preds = {}
    for date in dates:
        key = date+"-01_preds" if False else None
        for candidate in [date.replace("/","-")+"_preds", date+"_preds"]:
            if candidate in d.files:
                preds[date] = d[candidate].astype(np.float32)
                break
    return preds

class FusionDataset(Dataset):
    def __init__(self, fold_dates, cnn_preds_by_date, subsample=1000):
        self.X = []
        self.y = []
        for date in fold_dates:
            of_feats = load_of_features_for_date(date)
            if of_feats is None:
                continue
            cnn_pred = cnn_preds_by_date.get(date)
            if cnn_pred is None:
                continue
            n = min(len(cnn_pred), of_feats.shape[0])
            cnn_pred = cnn_pred[:n].reshape(n,1)
            X_day = np.concatenate([cnn_pred, of_feats[:n]], axis=1)
            idx = np.random.choice(n, min(subsample, n), replace=False)
            self.X.append(X_day[idx])
            self.y.append(cnn_pred[idx,0])
        if len(self.X) > 0:
            self.X = np.concatenate(self.X, axis=0).astype(np.float32)
            self.y = np.concatenate(self.y, axis=0).astype(np.float32)
        else:
            self.X = np.zeros((0, 1+N_OF_FEATURES), dtype=np.float32)
            self.y = np.zeros(0, dtype=np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.tensor(self.X[i]), torch.tensor(self.y[i])
from sklearn.preprocessing import StandardScaler
import pickle

def fit_scaler(train_ds):
    scaler = StandardScaler()
    scaler.fit(train_ds.X)
    return scaler

def apply_scaler(ds, scaler):
    ds.X = scaler.transform(ds.X)
    return ds

def save_scaler(scaler, path):
    with open(path, "wb") as f: pickle.dump(scaler, f)

def load_scaler(path):
    with open(path, "rb") as f: return pickle.load(f)
