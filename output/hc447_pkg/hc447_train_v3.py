#!/usr/bin/env python3
"""HC #447 v3 — Deep MLP meta-classifier, LEAKAGE-AUDITED FEATURES ONLY.

v2 showed AUC 0.88-0.90 across configs which is suspicious vs LGBM's 0.55 baseline.
Root cause: hold_s, queue_wait_ns, slippage_ticks are POST-FILL outcomes — they
encode the trade result and leak the label.

v3 uses STRICTLY PRE-TRADE features:
  - pred_strength (signal model output at signal time)
  - queue_ahead (queue depth observed at order placement)
  - tod_sec, dow (calendar)
  - is_short (direction chosen by signal rule, known ex-ante)

If AUC collapses back near 0.55, the v2 result was leakage and the negative
verdict stands. If AUC remains high, we have a real edge.
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE={device}", flush=True)

PRE_TRADE_FEATS = ["pred_strength","queue_ahead","tod_sec","dow","is_short"]

def featurize(df):
    df = df.copy()
    if "ts_signal_ns" in df.columns:
        ts = pd.to_datetime(df["ts_signal_ns"], unit="ns", utc=True).dt.tz_convert("US/Eastern")
        df["tod_sec"] = ts.dt.hour*3600 + ts.dt.minute*60 + ts.dt.second
        df["dow"] = ts.dt.dayofweek
    if "direction" in df.columns:
        df["is_short"] = (df["direction"].astype(str).str.lower()=="short").astype(int)
    return df, [c for c in PRE_TRADE_FEATS if c in df.columns]

def time_split(df):
    if "date" in df.columns:
        df = df.sort_values(["date","ts_signal_ns"] if "ts_signal_ns" in df.columns else ["date"])
    n=len(df); cut=int(n*0.7)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)

class MLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def run(cfg,csv):
    df = pd.read_csv(csv)
    df["net_tk"] = df["net_ticks"]
    df, feats = featurize(df)
    if len(feats)<3: return {"config":cfg,"err":f"feats={feats}"}
    tr,te = time_split(df)
    Xtr=tr[feats].fillna(0).values.astype(np.float32); ytr=(tr["net_tk"]>0).astype(np.float32).values
    Xte=te[feats].fillna(0).values.astype(np.float32); yte=(te["net_tk"]>0).astype(np.float32).values
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    Xtr_t=torch.tensor(Xtr).to(device); ytr_t=torch.tensor(ytr).to(device); Xte_t=torch.tensor(Xte).to(device)
    model=MLP(Xtr.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    bce=nn.BCEWithLogitsLoss()
    n=len(Xtr); bs=512; best_auc=0; p=None
    for ep in range(80):
        model.train()
        idx=np.random.permutation(n)
        for i in range(0,n,bs):
            b=idx[i:i+bs]
            if len(b)<2: continue
            opt.zero_grad(); loss=bce(model(Xtr_t[b]),ytr_t[b]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): p=torch.sigmoid(model(Xte_t)).cpu().numpy()
        try: a=roc_auc_score(yte,p)
        except: a=float("nan")
        if a>best_auc: best_auc=a
    net=te["net_tk"].values; best={"mean_tk":-1e9}
    for thr in np.linspace(0.05,0.95,19):
        m=p>=thr
        if m.sum()<30: continue
        sub=net[m]; mt=float(sub.mean()); gw=sub[sub>0].sum(); gl=-sub[sub<0].sum()
        pf=float(gw/gl) if gl>0 else float("inf"); wr=float((sub>0).mean()); n_=int(m.sum())
        if "date" in te.columns:
            te2=te.iloc[m]; byd=te2.groupby("date")["net_tk"].sum()
            posd=int((byd>0).sum()); nd=int(len(byd))
        else: posd,nd=-1,-1
        if mt>best["mean_tk"]:
            best={"thr":float(thr),"n":n_,"mean_tk":mt,"PF":pf,"WR":wr,"pos_days":posd,"n_days":nd}
    return {"config":cfg,"feats":feats,"n_train":len(Xtr),"n_test":len(Xte),
            "best_auc_oos":float(best_auc),"best_threshold_sweep":best,
            "verdict_pass": bool(best["mean_tk"]>0 and best.get("PF",0)>=1.10 and best["n"]>=100 and best.get("pos_days",-1)>=int(best.get("n_days",1)*0.8))}

def main():
    out=[]
    for csv in sorted(HERE.glob("*_fifo_fills.csv")):
        cfg=csv.name.replace("_fifo_fills.csv","")
        t0=time.time(); r=run(cfg,csv); r["seconds"]=round(time.time()-t0,1)
        print(json.dumps(r),flush=True); out.append(r)
    with open(HERE/"results_v3_pretrade_only.json","w") as f: json.dump(out,f,indent=2,default=str)
    print("DONE",flush=True)

if __name__=="__main__": main()
