#!/usr/bin/env python3
"""
MFE/MAE Analysis — Maximum Favorable/Adverse Excursion for 30-min LightGBM Signals
==================================================================================

PRICE FORMAT: Data prices are in TICK UNITS (1 data unit = 1 ES tick = 0.25 points = $12.50).
All MFE/MAE/SL/TP values are in REAL TICKS.

Author: Claude (autonomous research)
"""

import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
OUTPUT_DIR = ROOT / "output" / "mfe_mae_analysis"
LOG_DIR = ROOT / "logs"
sys.path.insert(0, str(ROOT))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [MFE-MAE] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "mfe_mae_analysis.log")),
    ],
)
log = logging.getLogger("MFE-MAE")

# ─── PRICE/COST CONSTANTS ───
# Data prices are in tick units: 1 data unit = 1 ES tick = $12.50
TICK_SIZE = 1.0            # 1 tick in data price units
TICK_VALUE = 12.50         # $ per tick
RT_COMMISSION_TICKS = 0.376  # $4.70 / $12.50
MARKET_SLIPPAGE_TICKS = 1.0  # 1 tick for market orders

TRAIN_DAYS = 30
SLIDE_DAYS = 5
ENTRY_BAR_SIZE_MIN = 30
TRACK_MINUTES = 30

LGBM_ENTRY_PARAMS = {
    "objective": "regression", "metric": "mae", "learning_rate": 0.05,
    "num_leaves": 63, "min_child_samples": 30, "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 5, "lambda_l1": 0.1,
    "lambda_l2": 1.0, "max_depth": 8, "verbose": -1, "n_jobs": -1,
}

lgb = None
def _import_lightgbm():
    global lgb
    if lgb is not None: return
    import lightgbm as _lgb
    lgb = _lgb


# ═══════════════════════════════════════════════
#  DATA LOADING (from passive_execution_v1.py)
# ═══════════════════════════════════════════════

def load_all_minute_bars(min_date="20250714"):
    files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    frames = []
    for f in files:
        if f.stem < min_date: continue
        try:
            df = pd.read_parquet(f); df["date"] = f.stem; frames.append(df)
        except Exception as e:
            log.warning(f"Skip {f.stem}: {e}")
    if not frames: raise RuntimeError(f"No files in {MINUTE_BAR_DIR}")
    combined = pd.concat(frames, ignore_index=True)
    combined["ts_minute"] = pd.to_datetime(combined["ts_minute"], utc=True)
    combined = combined.sort_values("ts_minute").reset_index(drop=True)
    log.info(f"Loaded {len(combined):,} minute bars across {len(frames)} days")
    return combined

def _safe_polyfit_slope(arr):
    if len(arr) < 2: return 0.0
    try: return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except: return 0.0

def aggregate_to_bars(minute_df, bar_size_min=30):
    df = minute_df.copy()
    df["bar_key"] = df["ts_minute"].dt.floor(f"{bar_size_min}min")
    df["return_1m"] = df.groupby("date")["close"].pct_change()
    df["abs_ofi"] = df["ofi_1min"].abs()
    sv_std = df.groupby("date")["signed_volume"].transform("std").replace(0, 1)
    df["sv_zscore"] = df["signed_volume"] / sv_std
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["close"].clip(lower=1)
    records = []
    for (date_str, bar_key), grp in df.groupby(["date", "bar_key"]):
        if len(grp) < 2: continue
        ca = grp["close"].values; va = grp["volume"].values
        oa = grp["ofi_1min"].values; sa = grp["signed_volume"].values
        ra = grp["return_1m"].fillna(0).values; spa = grp["spread_mean"].values
        ta = grp["trade_count"].values
        rec = {
            "date": date_str, "bar_key": bar_key, "ts": grp["ts_minute"].iloc[0],
            "open": ca[0], "high": ca.max(), "low": ca.min(), "close": ca[-1],
            "return_bar": (ca[-1]/ca[0]-1) if ca[0]>0 else 0,
            "range_ticks": (ca.max()-ca.min())/TICK_SIZE,
            "close_position": (ca[-1]-ca.min())/max(ca.max()-ca.min(), TICK_SIZE),
            "total_volume": va.sum(), "avg_volume": va.mean(),
            "volume_trend": _safe_polyfit_slope(va),
            "volume_concentration": va.max()/max(va.mean(),1),
            "ofi_sum": oa.sum(), "ofi_mean": oa.mean(),
            "ofi_std": oa.std() if len(oa)>1 else 0,
            "ofi_trend": _safe_polyfit_slope(oa),
            "ofi_consistency": np.mean(np.sign(oa)==np.sign(oa.sum())) if oa.sum()!=0 else 0.5,
            "ofi_late_vs_early": oa[len(oa)//2:].sum()-oa[:len(oa)//2].sum(),
            "signed_volume_sum": sa.sum(),
            "signed_volume_ratio": sa.sum()/max(va.sum(),1),
            "buy_volume_frac": float(np.sum(sa[sa>0]))/max(va.sum(),1),
            "sell_volume_frac": float(-np.sum(sa[sa<0]))/max(va.sum(),1),
            "sweep_minutes": int(np.sum(np.abs(grp["sv_zscore"].values)>2)),
            "max_sweep_intensity": float(np.abs(grp["sv_zscore"].values).max()),
            "sweep_direction": float(np.sign(sa[np.abs(grp["sv_zscore"].values).argmax()])) if len(sa)>0 else 0.0,
            "spread_mean": spa.mean(), "spread_max": spa.max(),
            "spread_trend": _safe_polyfit_slope(spa),
            "trade_count_sum": ta.sum(), "trade_count_trend": _safe_polyfit_slope(ta),
            "realized_vol": float(np.std(ra)*np.sqrt(252*(390//bar_size_min))) if len(ra)>1 else 0,
            "vol_of_vol": float(np.std(np.abs(ra))) if len(ra)>1 else 0,
            "up_vol": float(np.std(ra[ra>0])) if np.sum(ra>0)>1 else 0,
            "down_vol": float(np.std(ra[ra<0])) if np.sum(ra<0)>1 else 0,
            "vwap_dev_mean": grp["vwap_dev"].mean(),
            "vwap_dev_trend": _safe_polyfit_slope(grp["vwap_dev"].values),
        }
        if rec["up_vol"]>0 and rec["down_vol"]>0: rec["vol_asymmetry"]=rec["down_vol"]/rec["up_vol"]-1
        elif rec["down_vol"]>0: rec["vol_asymmetry"]=1.0
        elif rec["up_vol"]>0: rec["vol_asymmetry"]=-1.0
        else: rec["vol_asymmetry"]=0.0
        records.append(rec)
    result = pd.DataFrame(records)
    log.info(f"Aggregated {len(result):,} {bar_size_min}min bars")
    return result

def add_rolling_features(df, bar_size_min=30):
    df = df.sort_values("ts").reset_index(drop=True)
    for w in [4,8,16,32]:
        rm = df["ofi_sum"].rolling(w,min_periods=1).mean()
        rs = df["ofi_sum"].rolling(w,min_periods=2).std().fillna(1).replace(0,1)
        df[f"ofi_zscore_{w}bar"] = (df["ofi_sum"]-rm)/rs
        vm = df["total_volume"].rolling(w,min_periods=1).mean()
        df[f"vol_rel_{w}bar"] = df["total_volume"]/vm.clip(lower=1)
        if "sweep_minutes" in df.columns:
            df[f"sweep_pct_{w}bar"] = df["sweep_minutes"].rolling(w,min_periods=1).sum()/(w*bar_size_min)
    for b,l in [(4,"lb_4bar"),(8,"lb_8bar"),(16,"lb_16bar")]:
        df[f"ret_{l}"] = df["close"].pct_change(b)
    for w in [4,8,16]:
        df[f"rvol_{w}bar"] = df["return_bar"].rolling(w,min_periods=2).std()
    df["intraday_cum_ofi"] = df.groupby("date")["ofi_sum"].cumsum()
    df["intraday_cum_sv"] = df.groupby("date")["signed_volume_sum"].cumsum()
    df["ofi_sign_flip"] = (np.sign(df["ofi_sum"])!=np.sign(df["ofi_sum"].shift(1))).astype(np.float32)
    df["absorption"] = df["total_volume"]/df["range_ticks"].clip(lower=1)
    ds = df.groupby("date").agg(day_ofi=("ofi_sum","sum"),day_sv=("signed_volume_sum","sum"),
                                 day_ret=("return_bar","sum"),day_vol=("realized_vol","mean")).reset_index()
    ds["prev_day_ofi"]=ds["day_ofi"].shift(1); ds["prev_day_sv"]=ds["day_sv"].shift(1)
    ds["prev_day_ret"]=ds["day_ret"].shift(1)
    df = df.merge(ds[["date","prev_day_ofi","prev_day_sv","prev_day_ret"]], on="date", how="left")
    h = df["ts"].dt.hour+df["ts"].dt.minute/60.0
    p = ((h-13.5)/6.5).clip(0,1)
    df["tod_sin"]=np.sin(2*np.pi*p); df["tod_cos"]=np.cos(2*np.pi*p)
    df["tod_progress"]=p.values; df["bars_since_open"]=df.groupby("date").cumcount()
    df["regime_ret_16bar"]=df["close"].pct_change(16)
    df["regime_ret_32bar"]=df["close"].pct_change(32)
    rs2=df["return_bar"].rolling(4,min_periods=2).std()
    rl2=df["return_bar"].rolling(16,min_periods=4).std()
    df["regime_vol_ratio"]=rs2/rl2.clip(lower=1e-8)
    df["intraday_direction_strength"]=(
        df["intraday_cum_ofi"].abs()/df.groupby("date")["ofi_sum"].transform(lambda x:x.abs().cumsum()).clip(lower=1))
    return df

def get_feature_columns(df):
    ex = ("fwd_","direction_","trade_quality_","date","ts","bar_key")
    rp = {"open","high","low","close"}
    return [c for c in df.columns if not any(c.startswith(p) for p in ex) and c not in rp
            and df[c].dtype in (np.float64,np.float32,np.int64,np.int32,np.float16,np.int16)]

def add_forward_labels(df, horizon_bars, horizon_label, min_edge_ticks=2.5):
    df = df.sort_values("ts").reset_index(drop=True)
    fc = df["close"].shift(-horizon_bars)
    ft = (fc-df["close"])/TICK_SIZE
    tn = df["ts"].values; tf = df["ts"].shift(-horizon_bars).values
    for i in range(len(df)-horizon_bars):
        if pd.isna(tf[i]): continue
        if (pd.Timestamp(tf[i])-pd.Timestamp(tn[i])).total_seconds()>6*3600:
            ft.iloc[i]=np.nan
    df[f"fwd_ticks_{horizon_label}"]=ft
    df[f"direction_{horizon_label}"]=0
    df.loc[ft>min_edge_ticks,f"direction_{horizon_label}"]=1
    df.loc[ft<-min_edge_ticks,f"direction_{horizon_label}"]=-1
    log.info(f"Forward labels ({horizon_label}): {(~ft.isna()).sum():,} valid")
    return df

def leakage_audit(train_dates, val_dates, feature_cols):
    if set(train_dates)&set(val_dates): return False
    if max(train_dates)>=min(val_dates): return False
    if [c for c in feature_cols if c.startswith(("fwd_","direction_","trade_quality_"))]: return False
    return True

def train_entry_model(X_train, y_train, X_val, y_val, feature_names, fold_idx):
    _import_lightgbm()
    tv=~np.isnan(y_train); vv=~np.isnan(y_val)
    if tv.sum()<50 or vv.sum()<10:
        return None, np.full(len(y_train),np.nan), np.full(len(y_val),np.nan)
    Xt,yt=X_train[tv],y_train[tv]; Xv,yv=X_val[vv],y_val[vv]
    p={**LGBM_ENTRY_PARAMS,"seed":42+fold_idx}
    td=lgb.Dataset(Xt,label=yt,feature_name=feature_names)
    vd=lgb.Dataset(Xv,label=yv,feature_name=feature_names,reference=td)
    cb=[lgb.early_stopping(stopping_rounds=30,verbose=False),lgb.log_evaluation(period=0)]
    m=lgb.train(p,td,num_boost_round=500,valid_sets=[vd],callbacks=cb)
    tp=np.full(len(y_train),np.nan); vp=np.full(len(y_val),np.nan)
    tp[tv]=m.predict(Xt,num_iteration=m.best_iteration)
    vp[vv]=m.predict(Xv,num_iteration=m.best_iteration)
    pv=vp[vv]
    if len(pv)>5:
        ic=np.corrcoef(pv,yv)[0,1]
        log.info(f"  Fold {fold_idx}: IC={ic:.4f}, iter={m.best_iteration}")
    return m,tp,vp


# ═══════════════════════════════════════════════
#  MFE/MAE CORE
# ═══════════════════════════════════════════════

def compute_mfe_mae(bars_30m, minute_df, entry_preds, confidence_pcts, cancel_window_min=10):
    """Track minute-by-minute MFE/MAE after FIFO entry fill."""
    vm = ~np.isnan(entry_preds)
    vp = entry_preds[vm]
    if len(vp)<20: return {}

    minute_lookup = {}
    for d, g in minute_df.groupby("date"):
        minute_lookup[d] = g.sort_values("ts_minute").reset_index(drop=True)

    bts=bars_30m["ts"].values; bd=bars_30m["date"].values; bc=bars_30m["close"].values
    results = {}

    for cpct in confidence_pcts:
        ut = np.nanquantile(entry_preds[vm], 1-cpct)
        lt = np.nanquantile(entry_preds[vm], cpct)
        recs = []; ns=0; nf=0

        for i in range(len(bars_30m)):
            if np.isnan(entry_preds[i]): continue
            d = 0
            if entry_preds[i]>=ut: d=1
            elif entry_preds[i]<=lt: d=-1
            else: continue
            ns += 1
            ds = bd[i]; st = pd.Timestamp(bts[i]); sp = bc[i]
            if ds not in minute_lookup: continue
            dm = minute_lookup[ds]
            dts = dm["ts_minute"].values

            lp = sp
            sbe = st+pd.Timedelta(minutes=ENTRY_BAR_SIZE_MIN)
            cts = st+pd.Timedelta(minutes=cancel_window_min+ENTRY_BAR_SIZE_MIN)
            fm = (dts>=np.datetime64(sbe))&(dts<=np.datetime64(cts))
            fc = dm[fm]
            if len(fc)==0: continue

            filled=False; fp=None; fi=None
            for j,(ri,mb) in enumerate(fc.iterrows()):
                if d==1:
                    if mb["low"]<=lp-TICK_SIZE:
                        filled=True; fp=lp; fi=ri; break
                else:
                    if mb["high"]>=lp+TICK_SIZE:
                        filled=True; fp=lp; fi=ri; break
            if not filled: continue
            nf += 1

            fpos = dm.index.get_loc(fi)
            te = min(fpos+TRACK_MINUTES+1, len(dm))
            tb = dm.iloc[fpos:te]
            if len(tb)<2: continue

            th=tb["high"].values; tl=tb["low"].values; tc=tb["close"].values

            if d==1:
                mfe = float((th.max()-fp)/TICK_SIZE)
                mae = float((fp-tl.min())/TICK_SIZE)
                final = float((tc[-1]-fp)/TICK_SIZE)
            else:
                mfe = float((fp-tl.min())/TICK_SIZE)
                mae = float((th.max()-fp)/TICK_SIZE)
                final = float((fp-tc[-1])/TICK_SIZE)

            # Running MFE/MAE per minute
            rmfe=[]; rmae=[]
            for m in range(len(tb)):
                if d==1:
                    rmfe.append(float((th[:m+1].max()-fp)/TICK_SIZE))
                    rmae.append(float((fp-tl[:m+1].min())/TICK_SIZE))
                else:
                    rmfe.append(float((fp-tl[:m+1].min())/TICK_SIZE))
                    rmae.append(float((th[:m+1].max()-fp)/TICK_SIZE))

            recs.append({"date":ds,"direction":d,"pred":float(entry_preds[i]),
                         "fill_price":fp,"mfe_ticks":mfe,"mae_ticks":mae,
                         "final_ticks":final,"winner":final>0,
                         "track_minutes":len(tb),"running_mfe":rmfe,"running_mae":rmae})

        log.info(f"Conf {cpct:.0%}: signals={ns}, filled={nf}, tracked={len(recs)}")
        if recs:
            results[f"top_{int(cpct*100)}pct"] = pd.DataFrame(recs)
    return results


def analyze_mfe_mae(tdf, label):
    mfe=tdf["mfe_ticks"].values; mae=tdf["mae_ticks"].values
    final=tdf["final_ticks"].values; win=tdf["winner"].values
    pcts=[10,25,50,75,90,95]

    r = {
        "label":label, "n_trades":len(tdf),
        "n_winners":int(win.sum()), "n_losers":int((~win).sum()),
        "win_rate":float(win.mean()),
        "mfe_dist":{f"p{p}":float(np.percentile(mfe,p)) for p in pcts},
        "mfe_mean":float(mfe.mean()), "mfe_std":float(mfe.std()),
        "mae_dist":{f"p{p}":float(np.percentile(mae,p)) for p in pcts},
        "mae_mean":float(mae.mean()), "mae_std":float(mae.std()),
        "final_dist":{f"p{p}":float(np.percentile(final,p)) for p in pcts},
        "final_mean":float(final.mean()),
    }

    if win.sum()>5:
        wm=mae[win]
        r["winner_mae_dist"]={f"p{p}":float(np.percentile(wm,p)) for p in pcts}
        r["winner_mae_mean"]=float(wm.mean())
        r["winner_mfe_mean"]=float(mfe[win].mean())
    if (~win).sum()>5:
        lm=mae[~win]
        r["loser_mae_dist"]={f"p{p}":float(np.percentile(lm,p)) for p in pcts}
        r["loser_mae_mean"]=float(lm.mean())
        r["loser_mfe_mean"]=float(mfe[~win].mean())

    # SL analysis
    sl_analysis = []
    for sl in [1,2,3,4,5,6,8,10,12,15,20]:
        ws = float(np.mean(mae[win]>=sl)) if win.sum()>0 else 0
        ls = float(np.mean(mae[~win]>=sl)) if (~win).sum()>0 else 0
        surv = mae<sl
        sw = float(win[surv].mean()) if surv.sum()>0 else 0
        sf = float(final[surv].mean()) if surv.sum()>0 else 0
        sl_analysis.append({"sl":sl,"win_stop%":ws,"lose_stop%":ls,
                            "n_surv":int(surv.sum()),"n_stop":int((~surv).sum()),
                            "surv_wr":sw,"surv_avg":sf})
    r["sl_analysis"]=sl_analysis

    # TP analysis
    tp_analysis = []
    for tp in [1,2,3,4,5,6,8,10,12,15,20]:
        rr = mfe>=tp
        cm = float(mae[rr].mean()) if rr.sum()>0 else 0
        cp = float(np.percentile(mae[rr],90)) if rr.sum()>0 else 0
        tp_analysis.append({"tp":tp,"reach%":float(rr.mean()),"n":int(rr.sum()),
                            "cond_mae_mean":cm,"cond_mae_p90":cp})
    r["tp_analysis"]=tp_analysis

    # Combined TP/SL grid using running paths
    grid = []
    for tp in [2,3,4,5,6,8,10]:
        for sl in [2,3,4,5,6,8,10]:
            ntp=0; nsl=0; nto=0; psum=0.0
            for _,row in tdf.iterrows():
                rm=row["running_mfe"]; ra=row["running_mae"]
                tpm=None; slm=None
                for m in range(len(rm)):
                    if tpm is None and rm[m]>=tp: tpm=m
                    if slm is None and ra[m]>=sl: slm=m
                if tpm is not None and slm is not None:
                    if tpm<=slm:
                        ntp+=1; psum+=tp-RT_COMMISSION_TICKS
                    else:
                        nsl+=1; psum+=-(sl+MARKET_SLIPPAGE_TICKS+RT_COMMISSION_TICKS)
                elif tpm is not None:
                    ntp+=1; psum+=tp-RT_COMMISSION_TICKS
                elif slm is not None:
                    nsl+=1; psum+=-(sl+MARKET_SLIPPAGE_TICKS+RT_COMMISSION_TICKS)
                else:
                    nto+=1; psum+=row["final_ticks"]-MARKET_SLIPPAGE_TICKS-RT_COMMISSION_TICKS
            tot=ntp+nsl+nto
            if tot>0:
                be_wr = (sl+1.376)/(tp+sl+1.0)
                actual_wr = ntp/tot
                grid.append({"tp":tp,"sl":sl,"tp%":ntp/tot,"sl%":nsl/tot,"to%":nto/tot,
                             "avg_pnl":psum/tot,"total_pnl":psum,"n":tot,
                             "wr":actual_wr,"be_wr":be_wr,"wr_edge":actual_wr-be_wr})
    r["combined_grid"]=grid

    # Per direction
    for dv,dn in [(1,"long"),(-1,"short")]:
        dm = tdf["direction"].values==dv
        if dm.sum()<5: continue
        r[f"{dn}_n"]=int(dm.sum())
        r[f"{dn}_wr"]=float(win[dm].mean())
        r[f"{dn}_mfe_mean"]=float(mfe[dm].mean())
        r[f"{dn}_mae_mean"]=float(mae[dm].mean())
        r[f"{dn}_mfe_p50"]=float(np.median(mfe[dm]))
        r[f"{dn}_mae_p50"]=float(np.median(mae[dm]))
        r[f"{dn}_mfe_p90"]=float(np.percentile(mfe[dm],90))
        r[f"{dn}_mae_p90"]=float(np.percentile(mae[dm],90))
    return r


def clean_for_json(obj):
    if isinstance(obj,dict): return {k:clean_for_json(v) for k,v in obj.items()}
    elif isinstance(obj,list): return [clean_for_json(v) for v in obj]
    elif isinstance(obj,(np.integer,)): return int(obj)
    elif isinstance(obj,(np.floating,)): return float(obj)
    elif isinstance(obj,np.ndarray): return obj.tolist()
    elif isinstance(obj,(np.bool_,)): return bool(obj)
    elif isinstance(obj,pd.Timestamp): return str(obj)
    return obj


def main():
    _import_lightgbm()
    t0 = time.time()

    log.info("="*70)
    log.info("PHASE 1: Load data + build features")
    log.info("="*70)
    minute_df = load_all_minute_bars()
    bars_30m = aggregate_to_bars(minute_df, bar_size_min=30)
    bars_30m = add_rolling_features(bars_30m, bar_size_min=30)
    bars_30m = add_forward_labels(bars_30m, horizon_bars=1, horizon_label="30min", min_edge_ticks=2.5)
    feature_cols = get_feature_columns(bars_30m)
    log.info(f"Features: {len(feature_cols)}")

    log.info("\n"+"="*70)
    log.info("PHASE 2: Walk-forward entry model (30d train, SLIDING)")
    log.info("="*70)
    dates_30m = sorted(bars_30m["date"].unique())
    features_all = bars_30m[feature_cols].values.astype(np.float32)
    labels_all = bars_30m["fwd_ticks_30min"].values.astype(np.float32)
    dates_all = bars_30m["date"].values
    entry_preds = np.full(len(bars_30m), np.nan, dtype=np.float32)
    fi = 0
    for fs in range(TRAIN_DAYS, len(dates_30m)-5+1, SLIDE_DAYS):
        ftd = dates_30m[fs-TRAIN_DAYS:fs]; fvd = dates_30m[fs:fs+5]
        if len(fvd)<5: break
        fi += 1
        tm = np.isin(dates_all,ftd); vm = np.isin(dates_all,fvd)
        if not leakage_audit(list(ftd),list(fvd),feature_cols): continue
        tr=features_all[tm].copy(); vl=features_all[vm].copy()
        med=np.nanmedian(tr,axis=0); q75=np.nanpercentile(tr,75,axis=0)
        q25=np.nanpercentile(tr,25,axis=0); iqr=q75-q25; iqr[iqr<1e-8]=1.0
        tr=np.clip(np.nan_to_num((tr-med)/iqr,nan=0.0),-5,5)
        vl=np.clip(np.nan_to_num((vl-med)/iqr,nan=0.0),-5,5)
        _,_,vp = train_entry_model(tr,labels_all[tm],vl,labels_all[vm],feature_cols,fi)
        entry_preds[vm] = vp
        if fi%5==0: log.info(f"  Progress: fold {fi}, valid={int((~np.isnan(entry_preds)).sum())}")

    nv = (~np.isnan(entry_preds)).sum()
    log.info(f"\nEntry model: {fi} folds, {nv} valid preds, "
             f"{len(set(dates_all[~np.isnan(entry_preds)]))} OOT days")
    vmask = ~np.isnan(entry_preds) & ~np.isnan(labels_all)
    if vmask.sum()>50:
        ic = np.corrcoef(entry_preds[vmask],labels_all[vmask])[0,1]
        ric = stats.spearmanr(entry_preds[vmask],labels_all[vmask]).correlation
        log.info(f"Concat IC: {ic:.4f}, Rank IC: {ric:.4f}")

    log.info("\n"+"="*70)
    log.info("PHASE 3: MFE/MAE Analysis")
    log.info("="*70)
    data = compute_mfe_mae(bars_30m, minute_df, entry_preds, [0.05,0.10,0.15], cancel_window_min=10)

    analyses = {}
    for label, tdf in data.items():
        log.info(f"\n{'='*60}")
        log.info(f"  {label}")
        log.info(f"{'='*60}")
        a = analyze_mfe_mae(tdf, label)
        analyses[label] = a

        log.info(f"\n  Trades: {a['n_trades']} (W:{a['n_winners']}, L:{a['n_losers']}, WR:{a['win_rate']:.1%})")
        log.info(f"\n  MFE (ticks): " + ", ".join(f"{k}={v:.1f}" for k,v in a["mfe_dist"].items()))
        log.info(f"  MFE mean={a['mfe_mean']:.1f}")
        log.info(f"\n  MAE (ticks): " + ", ".join(f"{k}={v:.1f}" for k,v in a["mae_dist"].items()))
        log.info(f"  MAE mean={a['mae_mean']:.1f}")
        log.info(f"\n  Final PnL (ticks): " + ", ".join(f"{k}={v:.1f}" for k,v in a["final_dist"].items()))
        log.info(f"  Final mean={a['final_mean']:.1f}")

        if "winner_mae_dist" in a:
            log.info(f"\n  WINNER MAE: " + ", ".join(f"{k}={v:.1f}" for k,v in a["winner_mae_dist"].items()))
            log.info(f"  Winner MAE mean={a['winner_mae_mean']:.1f}")
        if "loser_mae_dist" in a:
            log.info(f"\n  LOSER MAE: " + ", ".join(f"{k}={v:.1f}" for k,v in a["loser_mae_dist"].items()))
            log.info(f"  Loser MAE mean={a['loser_mae_mean']:.1f}")

        log.info(f"\n  STOP LOSS ANALYSIS:")
        log.info(f"  {'SL':>4} {'WinStop':>8} {'LoseStop':>9} {'Surv':>6} {'Stop':>6} {'SurvWR':>7} {'SurvAvg':>8}")
        log.info(f"  {'-'*52}")
        for s in a["sl_analysis"]:
            log.info(f"  {s['sl']:4d} {s['win_stop%']:8.1%} {s['lose_stop%']:9.1%} "
                     f"{s['n_surv']:6d} {s['n_stop']:6d} {s['surv_wr']:7.1%} {s['surv_avg']:8.2f}")

        log.info(f"\n  TP REACH ANALYSIS:")
        log.info(f"  {'TP':>4} {'Reach':>7} {'N':>6} {'CondMAE':>8} {'MAE_p90':>8}")
        log.info(f"  {'-'*38}")
        for t in a["tp_analysis"]:
            log.info(f"  {t['tp']:4d} {t['reach%']:7.1%} {t['n']:6d} {t['cond_mae_mean']:8.1f} {t['cond_mae_p90']:8.1f}")

        if a.get("combined_grid"):
            gs = sorted(a["combined_grid"], key=lambda x:x["avg_pnl"], reverse=True)
            log.info(f"\n  BEST TP/SL COMBOS (by avg PnL/trade, in real ticks):")
            log.info(f"  {'TP':>3} {'SL':>3} {'TP%':>6} {'SL%':>6} {'TO%':>6} {'AvgPnL':>8} {'TotPnL':>9} "
                     f"{'WR':>6} {'BEWR':>6} {'Edge':>6}")
            log.info(f"  {'-'*65}")
            for g in gs[:20]:
                log.info(f"  {g['tp']:3d} {g['sl']:3d} {g['tp%']:6.1%} {g['sl%']:6.1%} {g['to%']:6.1%} "
                         f"{g['avg_pnl']:8.3f} {g['total_pnl']:9.1f} {g['wr']:6.1%} {g['be_wr']:6.1%} "
                         f"{g['wr_edge']:+6.1%}")

        for dn in ["long","short"]:
            if f"{dn}_n" in a:
                log.info(f"\n  {dn.upper()}: N={a[f'{dn}_n']}, WR={a[f'{dn}_wr']:.1%}, "
                         f"MFE_mean={a[f'{dn}_mfe_mean']:.1f}, MAE_mean={a[f'{dn}_mae_mean']:.1f}, "
                         f"MFE_p90={a[f'{dn}_mfe_p90']:.1f}, MAE_p90={a[f'{dn}_mae_p90']:.1f}")

    # Save
    log.info("\n"+"="*70)
    log.info("PHASE 4: Saving results")
    log.info("="*70)
    np.savez_compressed(str(OUTPUT_DIR/"entry_predictions.npz"),
                        entry_preds=entry_preds, labels_30m=labels_all, dates=dates_all)
    with open(str(OUTPUT_DIR/"mfe_mae_results.json"),"w") as f:
        json.dump(clean_for_json(analyses), f, indent=2, default=str)
    for label, tdf in data.items():
        tdf.drop(columns=["running_mfe","running_mae"],errors="ignore").to_parquet(
            str(OUTPUT_DIR/f"trades_{label}.parquet"), index=False)

    elapsed = time.time()-t0
    log.info(f"\nComplete in {elapsed/60:.1f} minutes")

    log.info("\n"+"="*70)
    log.info("SUMMARY")
    log.info("="*70)
    for label, a in analyses.items():
        log.info(f"\n{label}: N={a['n_trades']}, WR={a['win_rate']:.1%}")
        log.info(f"  MFE p50={a['mfe_dist']['p50']:.1f}, p90={a['mfe_dist']['p90']:.1f}")
        log.info(f"  MAE p50={a['mae_dist']['p50']:.1f}, p90={a['mae_dist']['p90']:.1f}")
        if "winner_mae_mean" in a:
            log.info(f"  Winner MAE={a['winner_mae_mean']:.1f} vs Loser MAE={a.get('loser_mae_mean',0):.1f}")
        if a.get("combined_grid"):
            best = max(a["combined_grid"], key=lambda x:x["avg_pnl"])
            log.info(f"  BEST: TP={best['tp']}, SL={best['sl']}, "
                     f"AvgPnL={best['avg_pnl']:.3f}, WR={best['wr']:.1%}, Edge={best['wr_edge']:+.1%}")

if __name__=="__main__":
    main()
