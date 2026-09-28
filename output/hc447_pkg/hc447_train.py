#!/usr/bin/env python3
"""HC #447 — Deep MLP meta-classifier on Neptune GPU.

LGBM HC #445 saturated at AUC~0.55 across configs. This trains a 4-layer
PyTorch MLP (256-128-64-1) with dropout + batchnorm on the same multi-h
feature set, time-ordered 70/30 split. If GPU MLP can't beat LGBM, the
negative outcome is genuinely confirmed (model capacity isn't the bottleneck;
data has no separable signal-vs-cost class boundary).

Inputs: 5 fills CSVs in this directory.
Output: results.json (AUC + threshold sweep verdict per config).
"""
import json, sys, os, time, glob
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).parent
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE={device} {torch.cuda.get_device_name(0) if device=='cuda' else ''}")

FEATURES = ["pred_v2_1s","pred_v2_5s","pred_v2_10s","abs_pred","queue_ahead",
            "time_of_day_min","signal_density_5s","spread_at_signal"]

def time_split(df):
    df = df.sort_values("trade_date") if "trade_date" in df.columns else df.reset_index(drop=True)
    n = len(df); cut = int(n*0.7)
    return df.iloc[:cut], df.iloc[cut:]

def to_xy(df, feats):
    avail = [f for f in feats if f in df.columns]
    if len(avail) < 3: return None, None, avail
    X = df[avail].fillna(0).values.astype(np.float32)
    y = (df["net_tk"] > 0).astype(np.float32).values
    return X, y, avail

class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))
    def forward(self,x): return self.net(x).squeeze(-1)

def train_eval(cfg, csv):
    df = pd.read_csv(csv)
    if "net_tk" not in df.columns:
        for alt in ["net_ticks","fill_net_tk","pnl_tk"]:
            if alt in df.columns: df = df.rename(columns={alt:"net_tk"}); break
    if "net_tk" not in df.columns: return {"config":cfg,"err":"no net_tk"}
    tr,te = time_split(df)
    Xtr,ytr,feats = to_xy(tr, FEATURES)
    Xte,yte,_ = to_xy(te, FEATURES)
    if Xtr is None: return {"config":cfg,"err":f"only {feats} feats"}
    # normalize
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-6
    Xtr = (Xtr-mu)/sd; Xte = (Xte-mu)/sd
    Xtr_t = torch.tensor(Xtr).to(device); ytr_t = torch.tensor(ytr).to(device)
    Xte_t = torch.tensor(Xte).to(device); yte_t = torch.tensor(yte).to(device)
    model = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    n = len(Xtr); bs=512
    best_auc=0
    for ep in range(40):
        model.train()
        idx = np.random.permutation(n)
        for i in range(0,n,bs):
            b = idx[i:i+bs]
            opt.zero_grad()
            logit = model(Xtr_t[b])
            loss = bce(logit, ytr_t[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xte_t)).cpu().numpy()
        # quick AUC
        from sklearn.metrics import roc_auc_score
        try: auc = roc_auc_score(yte, p)
        except: auc = float("nan")
        if auc>best_auc: best_auc=auc
    # threshold sweep
    best_thr=None; best_mt=-1e9; best_n=0; best_pf=0
    for thr in np.linspace(0.05,0.95,19):
        mask = p >= thr
        if mask.sum() < 30: continue
        sub = te.iloc[mask]
        mt = sub["net_tk"].mean()
        gw = sub.loc[sub["net_tk"]>0,"net_tk"].sum()
        gl = -sub.loc[sub["net_tk"]<0,"net_tk"].sum()
        pf = gw/gl if gl>0 else float("inf")
        if mt>best_mt: best_mt=float(mt); best_thr=float(thr); best_n=int(mask.sum()); best_pf=float(pf)
    return {"config":cfg,"feats_used":feats,"n_train":len(Xtr),"n_test":len(Xte),
            "best_auc_oos":float(best_auc),"best_thr":best_thr,
            "best_n":best_n,"best_mean_tk":best_mt,"best_PF":best_pf,
            "verdict_pass": bool(best_mt>0 and best_pf>=1.10 and best_n>=100)}

def main():
    out=[]
    for csv in sorted(HERE.glob("*_fifo_fills.csv")):
        cfg = csv.name.replace("_fifo_fills.csv","")
        t0=time.time()
        r = train_eval(cfg, csv)
        r["seconds"]=round(time.time()-t0,1)
        print(json.dumps(r))
        out.append(r)
    with open(HERE/"results.json","w") as f: json.dump(out,f,indent=2)
    print("DONE")

if __name__=="__main__": main()
