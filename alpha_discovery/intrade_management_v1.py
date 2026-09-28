#!/usr/bin/env python3
"""
In-Trade Management v1 — Continuous Position Monitoring via LightGBM
=====================================================================

After entry, evaluate the trade every N minutes using a LightGBM classifier
that predicts: will this trade ultimately hit TP or SL?

KEY INSIGHT: Sample weighting per trade. A TP winner with 50 minute samples
and a SL loser with 2 minute samples must contribute equally to the classifier.
Each sample gets weight = 1 / (num_samples_from_that_trade).

Management actions:
  - P(win) < invalidation_threshold → EXIT EARLY at market (cut loser)
  - P(win) > confirmation_threshold AND profitable → TRAIL stop to breakeven
  - Otherwise → hold as normal

Walk-forward: 30d train / 5d slide (SLIDING, HC #0)
FIFO fills: passive entry, passive TP, market SL (HC #74)
Cost: commission = 0.376 ticks RT, market slippage = 1 tick

Base config (champion): TP_long=25, TP_short=25, SL_long=4, SL_short=3,
                         max_hold=60, entry_threshold=0.05, cancel_window=10

Sweep: 3 × 2 × 2 × 3 = 36 management configs

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
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
OUTPUT_DIR = ROOT / "output" / "intrade_management_v1"
MFE_MAE_DIR = ROOT / "output" / "mfe_mae_analysis"
LOG_DIR = ROOT / "logs"
sys.path.insert(0, str(ROOT))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [INTRADE-MGMT] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "intrade_management_v1.log"), mode="w"),
    ],
)
log = logging.getLogger("INTRADE-MGMT")

TICK_SIZE = 1.0
TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_SLIPPAGE_TICKS = 1.0
TRAIN_DAYS = 30
SLIDE_DAYS = 5
ENTRY_BAR_SIZE_MIN = 30

BASE_CONFIG = {
    "tp_long": 25, "tp_short": 25,
    "sl_long": 4, "sl_short": 3,
    "max_hold": 60,
    "entry_threshold": 0.05, "cancel_window": 10,
}

INVALIDATION_PCTILE_SWEEP = [10, 20, 30]
CONFIRMATION_PCTILE_SWEEP = [70, 80]
TRAIL_TO_BREAKEVEN_SWEEP = [True, False]
CHECK_INTERVAL_SWEEP = [1, 2, 5]

TOTAL_MGMT_CONFIGS = (len(INVALIDATION_PCTILE_SWEEP) *
                      len(CONFIRMATION_PCTILE_SWEEP) *
                      len(TRAIL_TO_BREAKEVEN_SWEEP) *
                      len(CHECK_INTERVAL_SWEEP))

LGBM_ENTRY_PARAMS = {
    "objective": "regression", "metric": "mae",
    "learning_rate": 0.05, "num_leaves": 63,
    "min_child_samples": 30, "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "lambda_l1": 0.1, "lambda_l2": 1.0, "max_depth": 8,
    "verbose": -1, "n_jobs": -1,
}

LGBM_MGMT_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.03, "num_leaves": 31,
    "min_child_samples": 10, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "lambda_l1": 0.05, "lambda_l2": 0.5, "max_depth": 5,
    "verbose": -1, "n_jobs": -1,
}

lgb = None

def _import_lightgbm():
    global lgb
    if lgb is None:
        import lightgbm as _lgb
        lgb = _lgb

def _safe_polyfit_slope(arr):
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except:
        return 0.0


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_all_minute_bars(min_date="20250714"):
    files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    frames = []
    for f in files:
        if f.stem < min_date:
            continue
        try:
            df = pd.read_parquet(f)
            df["date"] = f.stem
            frames.append(df)
        except:
            pass
    if not frames:
        raise RuntimeError("No minute bar files")
    combined = pd.concat(frames, ignore_index=True)
    combined["ts_minute"] = pd.to_datetime(combined["ts_minute"], utc=True)
    combined = combined.sort_values("ts_minute").reset_index(drop=True)
    log.info(f"Loaded {len(combined):,} minute bars across {len(frames)} days")
    return combined


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
        if len(grp) < 2:
            continue
        c = grp["close"].values
        v = grp["volume"].values
        o = grp["ofi_1min"].values
        sv = grp["signed_volume"].values
        r = grp["return_1m"].fillna(0).values
        sp = grp["spread_mean"].values
        tc = grp["trade_count"].values
        rec = {
            "date": date_str, "bar_key": bar_key,
            "ts": grp["ts_minute"].iloc[0],
            "open": c[0], "high": c.max(), "low": c.min(), "close": c[-1],
            "return_bar": (c[-1]/c[0]-1) if c[0]>0 else 0,
            "range_ticks": (c.max()-c.min())/TICK_SIZE,
            "close_position": (c[-1]-c.min())/max(c.max()-c.min(), TICK_SIZE),
            "total_volume": v.sum(), "avg_volume": v.mean(),
            "volume_trend": _safe_polyfit_slope(v),
            "volume_concentration": v.max()/max(v.mean(),1),
            "ofi_sum": o.sum(), "ofi_mean": o.mean(),
            "ofi_std": o.std() if len(o)>1 else 0,
            "ofi_trend": _safe_polyfit_slope(o),
            "ofi_consistency": np.mean(np.sign(o)==np.sign(o.sum())) if o.sum()!=0 else 0.5,
            "ofi_late_vs_early": o[len(o)//2:].sum()-o[:len(o)//2].sum(),
            "signed_volume_sum": sv.sum(),
            "signed_volume_ratio": sv.sum()/max(v.sum(),1),
            "buy_volume_frac": float(np.sum(sv[sv>0]))/max(v.sum(),1),
            "sell_volume_frac": float(-np.sum(sv[sv<0]))/max(v.sum(),1),
            "sweep_minutes": int(np.sum(np.abs(grp["sv_zscore"].values)>2)),
            "max_sweep_intensity": float(np.abs(grp["sv_zscore"].values).max()),
            "sweep_direction": float(np.sign(sv[np.abs(grp["sv_zscore"].values).argmax()])) if len(sv)>0 else 0.0,
            "spread_mean": sp.mean(), "spread_max": sp.max(),
            "spread_trend": _safe_polyfit_slope(sp),
            "trade_count_sum": tc.sum(),
            "trade_count_trend": _safe_polyfit_slope(tc),
            "realized_vol": float(np.std(r)*np.sqrt(252*(390//bar_size_min))) if len(r)>1 else 0,
            "vol_of_vol": float(np.std(np.abs(r))) if len(r)>1 else 0,
            "up_vol": float(np.std(r[r>0])) if np.sum(r>0)>1 else 0,
            "down_vol": float(np.std(r[r<0])) if np.sum(r<0)>1 else 0,
            "vwap_dev_mean": grp["vwap_dev"].mean(),
            "vwap_dev_trend": _safe_polyfit_slope(grp["vwap_dev"].values),
        }
        if rec["up_vol"]>0 and rec["down_vol"]>0:
            rec["vol_asymmetry"] = rec["down_vol"]/rec["up_vol"]-1
        elif rec["down_vol"]>0:
            rec["vol_asymmetry"] = 1.0
        elif rec["up_vol"]>0:
            rec["vol_asymmetry"] = -1.0
        else:
            rec["vol_asymmetry"] = 0.0
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
    ds["prev_day_ofi"] = ds["day_ofi"].shift(1)
    ds["prev_day_sv"] = ds["day_sv"].shift(1)
    ds["prev_day_ret"] = ds["day_ret"].shift(1)
    df = df.merge(ds[["date","prev_day_ofi","prev_day_sv","prev_day_ret"]], on="date", how="left")
    sh,eh = 13.5,20.0
    hf = df["ts"].dt.hour+df["ts"].dt.minute/60.0
    p = ((hf-sh)/(eh-sh)).clip(0,1)
    df["tod_sin"] = np.sin(2*np.pi*p)
    df["tod_cos"] = np.cos(2*np.pi*p)
    df["tod_progress"] = p.values
    df["bars_since_open"] = df.groupby("date").cumcount()
    df["regime_ret_16bar"] = df["close"].pct_change(16)
    df["regime_ret_32bar"] = df["close"].pct_change(32)
    rs_s = df["return_bar"].rolling(4,min_periods=2).std()
    rs_l = df["return_bar"].rolling(16,min_periods=4).std()
    df["regime_vol_ratio"] = rs_s/rs_l.clip(lower=1e-8)
    df["intraday_direction_strength"] = (
        df["intraday_cum_ofi"].abs()/
        df.groupby("date")["ofi_sum"].transform(lambda x: x.abs().cumsum()).clip(lower=1))
    return df


def get_feature_columns(df):
    excl = ("fwd_","direction_","trade_quality_","date","ts","bar_key")
    raw = {"open","high","low","close"}
    return [c for c in df.columns
            if not any(c.startswith(p) for p in excl)
            and c not in raw
            and df[c].dtype in (np.float64,np.float32,np.int64,np.int32)]


def add_forward_labels(df, horizon_bars, horizon_label, min_edge_ticks=2.5):
    df = df.sort_values("ts").reset_index(drop=True)
    fc = df["close"].shift(-horizon_bars)
    ft = (fc-df["close"])/TICK_SIZE
    ts_now = df["ts"].values
    ts_fwd = df["ts"].shift(-horizon_bars).values
    for i in range(len(df)-horizon_bars):
        if pd.isna(ts_fwd[i]):
            continue
        if (pd.Timestamp(ts_fwd[i])-pd.Timestamp(ts_now[i])).total_seconds()>6*3600:
            ft.iloc[i] = np.nan
    df[f"fwd_ticks_{horizon_label}"] = ft
    df[f"direction_{horizon_label}"] = 0
    df.loc[ft>min_edge_ticks, f"direction_{horizon_label}"] = 1
    df.loc[ft<-min_edge_ticks, f"direction_{horizon_label}"] = -1
    return df


def leakage_audit(train_dates, val_dates, feature_cols):
    if set(train_dates) & set(val_dates):
        return False
    if max(train_dates) >= min(val_dates):
        return False
    if any(c.startswith(("fwd_","direction_","trade_quality_")) for c in feature_cols):
        return False
    return True


def train_entry_model(X_train, y_train, X_val, y_val, feature_names, fold_idx):
    _import_lightgbm()
    tv = ~np.isnan(y_train)
    vv = ~np.isnan(y_val)
    if tv.sum()<50 or vv.sum()<10:
        return None, np.full(len(y_train),np.nan), np.full(len(y_val),np.nan)
    p = {**LGBM_ENTRY_PARAMS, "seed": 42+fold_idx}
    td = lgb.Dataset(X_train[tv], label=y_train[tv], feature_name=feature_names)
    vd = lgb.Dataset(X_val[vv], label=y_val[vv], feature_name=feature_names, reference=td)
    cb = [lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=0)]
    m = lgb.train(p, td, num_boost_round=500, valid_sets=[vd], callbacks=cb)
    tp = np.full(len(y_train),np.nan)
    vp = np.full(len(y_val),np.nan)
    tp[tv] = m.predict(X_train[tv], num_iteration=m.best_iteration)
    vp[vv] = m.predict(X_val[vv], num_iteration=m.best_iteration)
    return m, tp, vp


# ═══════════════════════════════════════════════════════════════════
#  ENTRY FILL RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════

def reconstruct_entry_fills(bars_30m, minute_df, pred_30m, confidence_pct, cancel_window_min):
    vm = ~np.isnan(pred_30m)
    vp = pred_30m[vm]
    if len(vp)<20:
        return [], {"error":"too few"}
    ut = np.nanquantile(pred_30m[vm], 1-confidence_pct)
    lt = np.nanquantile(pred_30m[vm], confidence_pct)
    ml = {}
    for ds, grp in minute_df.groupby("date"):
        ml[ds] = grp.sort_values("ts_minute").reset_index(drop=True)
    trades = []
    ns, nf, nc = 0, 0, 0
    fd = []
    bts = bars_30m["ts"].values
    bdt = bars_30m["date"].values
    bcl = bars_30m["close"].values
    for i in range(len(bars_30m)):
        if np.isnan(pred_30m[i]):
            continue
        d = 0
        if pred_30m[i]>=ut: d=1
        elif pred_30m[i]<=lt: d=-1
        else: continue
        ns += 1
        ds = bdt[i]
        sts = pd.Timestamp(bts[i])
        sp = bcl[i]
        if ds not in ml:
            continue
        dm = ml[ds]
        dts = dm["ts_minute"].values
        lp = sp
        sbe = sts+pd.Timedelta(minutes=ENTRY_BAR_SIZE_MIN)
        cts = sts+pd.Timedelta(minutes=cancel_window_min+ENTRY_BAR_SIZE_MIN)
        fm = (dts>=np.datetime64(sbe))&(dts<=np.datetime64(cts))
        fc = dm[fm]
        if len(fc)==0:
            nc += 1; continue
        filled = False
        for j,(_,mb) in enumerate(fc.iterrows()):
            if d==1:
                if mb["low"]<=lp-TICK_SIZE:
                    filled=True; fp=lp; fts=mb["ts_minute"]; fd.append(j+1); break
            else:
                if mb["high"]>=lp+TICK_SIZE:
                    filled=True; fp=lp; fts=mb["ts_minute"]; fd.append(j+1); break
        if not filled:
            nc += 1; continue
        nf += 1
        rm = dm[dts>=np.datetime64(fts)]
        if len(rm)<2:
            nf -= 1; nc += 1; continue
        trades.append({
            "idx": i, "date": ds, "signal_ts": sts,
            "fill_ts": pd.Timestamp(fts), "fill_price": fp,
            "fill_delay_minutes": fd[-1], "direction": d,
            "pred_30m": float(pred_30m[i]),
            "prices_close": rm["close"].values.copy(),
            "prices_high": rm["high"].values.copy(),
            "prices_low": rm["low"].values.copy(),
            "times": rm["ts_minute"].values.copy(),
            "volumes": rm["volume"].values.copy(),
            "vwaps": rm["vwap"].values.copy(),
            "signed_volumes": rm["signed_volume"].values.copy(),
            "spread_means": rm["spread_mean"].values.copy(),
            "ofi_1mins": rm["ofi_1min"].values.copy(),
            "n_remaining_minutes": len(rm),
        })
    fs = {"n_signals":ns,"n_filled":nf,"n_cancelled":nc,
          "fill_rate":nf/max(ns,1),"avg_fill_delay":float(np.mean(fd)) if fd else 0}
    log.info(f"  Entry fills: signals={ns}, filled={nf} ({fs['fill_rate']:.1%})")
    return trades, fs


# ═══════════════════════════════════════════════════════════════════
#  BASE STRATEGY SIMULATION
# ═══════════════════════════════════════════════════════════════════

def simulate_base_strategy(trades, tp_long, tp_short, sl_long, sl_short, max_hold_minutes):
    results = []
    for trade in trades:
        d = trade["direction"]
        fp = trade["fill_price"]
        pc = trade["prices_close"]
        ph = trade["prices_high"]
        pl = trade["prices_low"]
        nr = trade["n_remaining_minutes"]
        tpt = tp_long if d==1 else tp_short
        slt = sl_long if d==1 else sl_short
        tpp = fp + d*tpt*TICK_SIZE
        slp = fp - d*slt*TICK_SIZE
        et = None; em = None; tp_pnl = None
        mc = min(max_hold_minutes, nr)
        for m in range(1, mc):
            sh, slh, tph = False, False, False
            if d==1:
                if pl[m]<=slp: slh=True
                if ph[m]>=tpp+TICK_SIZE: tph=True
            else:
                if ph[m]>=slp: slh=True
                if pl[m]<=tpp-TICK_SIZE: tph=True
            if slh and tph: slh,tph = True,False
            if slh:
                tp_pnl = -(slt+MARKET_SLIPPAGE_TICKS+RT_COMMISSION_TICKS)
                et="sl"; em=m; break
            if tph:
                tp_pnl = tpt-RT_COMMISSION_TICKS
                et="tp"; em=m; break
        if et is None:
            em = min(max_hold_minutes, nr-1); em = max(em,1)
            ec = pc[em]
            ef = (ec-TICK_SIZE) if d==1 else (ec+TICK_SIZE)
            tp_pnl = (ef-fp)/TICK_SIZE*d - RT_COMMISSION_TICKS
            et = "time_stop"
        results.append({**trade, "tp_ticks":tpt, "sl_ticks":slt,
                       "exit_type":et, "exit_minute":em, "trade_pnl":tp_pnl,
                       "outcome": 1 if et=="tp" else 0})
    return results


# ═══════════════════════════════════════════════════════════════════
#  MANAGEMENT FEATURES + TRAINING
# ═══════════════════════════════════════════════════════════════════

MGMT_FEATURE_NAMES = [
    "unrealized_pnl_ticks", "pnl_pct_to_tp", "pnl_pct_to_sl",
    "max_favorable_so_far", "max_adverse_so_far",
    "minutes_elapsed", "minutes_remaining", "pct_time_elapsed",
    "ofi_since_entry", "ofi_direction_match",
    "volume_since_entry", "volume_trend_recent",
    "spread_mean_recent", "spread_trend_recent",
    "close_vs_entry_bar_close", "signed_volume_direction",
    "vwap_vs_price", "pnl_velocity", "mfe_mae_ratio",
    "direction_is_short",
]


def _feat_vec(d, fp, tpt, slt, mh, m, pc, ph, pl, vol, vw, sv, sp, ofi, nr, mf, ma, co, cv):
    cc = pc[m]
    u = (cc-fp)/TICK_SIZE*d
    lb = min(5, m)
    rs = sp[max(0,m-lb):m+1]
    rv = vol[max(0,m-lb):m+1]
    rsv = sv[max(0,m-lb):m+1]
    return [
        u, u/tpt if tpt>0 else 0, u/slt if slt>0 else 0,
        mf, ma, float(m), float(mh-m), m/mh,
        co, 1.0 if co*d>0 else -1.0,
        cv, _safe_polyfit_slope(rv),
        float(np.mean(rs)), _safe_polyfit_slope(rs),
        (cc-fp)/TICK_SIZE,
        1.0 if np.sum(rsv)*d>0 else -1.0,
        (cc-vw[m])/TICK_SIZE*d if vw[m]>0 else 0.0,
        u/max(m,1), mf/max(ma,0.1),
        1.0 if d==-1 else 0.0,
    ]


def build_management_samples(trade_results, check_interval, max_hold):
    """
    Build samples with per-trade weighting so each trade contributes
    equal total weight regardless of duration.
    """
    all_features = []
    all_labels = []
    all_weights = []

    for tr in trade_results:
        d = tr["direction"]
        fp = tr["fill_price"]
        tpt = tr["tp_ticks"]
        slt = tr["sl_ticks"]
        em = tr["exit_minute"]
        outcome = tr["outcome"]
        nr = tr["n_remaining_minutes"]

        mf, ma, co, cv = 0.0, 0.0, 0.0, 0.0
        trade_feats = []

        for m in range(1, em):
            if m >= nr:
                break
            if d==1:
                fav = (tr["prices_high"][m]-fp)/TICK_SIZE
                adv = (fp-tr["prices_low"][m])/TICK_SIZE
            else:
                fav = (fp-tr["prices_low"][m])/TICK_SIZE
                adv = (tr["prices_high"][m]-fp)/TICK_SIZE
            mf = max(mf, fav)
            ma = max(ma, adv)
            co += tr["ofi_1mins"][m]
            cv += tr["volumes"][m]

            if m % check_interval != 0:
                continue

            feat = _feat_vec(d, fp, tpt, slt, max_hold, m,
                           tr["prices_close"], tr["prices_high"], tr["prices_low"],
                           tr["volumes"], tr["vwaps"], tr["signed_volumes"],
                           tr["spread_means"], tr["ofi_1mins"],
                           nr, mf, ma, co, cv)
            trade_feats.append(feat)

        if trade_feats:
            n_samples = len(trade_feats)
            w = 1.0 / n_samples  # Each trade contributes total weight of 1.0
            for feat in trade_feats:
                all_features.append(feat)
                all_labels.append(outcome)
                all_weights.append(w)

    if not all_features:
        return np.empty((0,len(MGMT_FEATURE_NAMES))), np.empty(0), np.empty(0)

    return (np.array(all_features, dtype=np.float32),
            np.array(all_labels, dtype=np.float32),
            np.array(all_weights, dtype=np.float32))


# ═══════════════════════════════════════════════════════════════════
#  MANAGED EXIT SIMULATION
# ═══════════════════════════════════════════════════════════════════

def simulate_managed_exit(
    trades, mgmt_model,
    tp_long, tp_short, sl_long, sl_short, max_hold,
    inv_cutoff, conf_cutoff, trail_to_breakeven, check_interval,
):
    pnls, dates, dirs_out, hold_durs = [], [], [], []
    n_tp, n_sl, n_time, n_early, n_trail = 0, 0, 0, 0, 0
    ee_would_sl, ee_would_tp = 0, 0
    ee_saved, ee_pnls = [], []

    for trade in trades:
        d = trade["direction"]
        fp = trade["fill_price"]
        pc = trade["prices_close"]
        ph = trade["prices_high"]
        pl = trade["prices_low"]
        nr = trade["n_remaining_minutes"]
        tpt = tp_long if d==1 else tp_short
        slt = sl_long if d==1 else sl_short
        tpp = fp + d*tpt*TICK_SIZE
        slp = fp - d*slt*TICK_SIZE
        eff_slp = slp
        eff_slt = slt
        trailed = False
        mc = min(max_hold, nr)
        mf, ma, co, cv = 0.0, 0.0, 0.0, 0.0

        # Base outcome for quality tracking
        bet = None
        bep = None
        for bm in range(1, mc):
            bsh, bth = False, False
            if d==1:
                if pl[bm]<=slp: bsh=True
                if ph[bm]>=tpp+TICK_SIZE: bth=True
            else:
                if ph[bm]>=slp: bsh=True
                if pl[bm]<=tpp-TICK_SIZE: bth=True
            if bsh and bth: bsh,bth=True,False
            if bsh:
                bet="sl"; bep=-(slt+MARKET_SLIPPAGE_TICKS+RT_COMMISSION_TICKS); break
            if bth:
                bet="tp"; bep=tpt-RT_COMMISSION_TICKS; break
        if bet is None:
            bet="time_stop"
            bm_e=min(max_hold,nr-1); bm_e=max(bm_e,1)
            bc=pc[bm_e]
            bf=(bc-TICK_SIZE) if d==1 else (bc+TICK_SIZE)
            bep=(bf-fp)/TICK_SIZE*d-RT_COMMISSION_TICKS

        # Managed simulation
        exited = False
        for m in range(1, mc):
            if d==1:
                fav=(ph[m]-fp)/TICK_SIZE; adv=(fp-pl[m])/TICK_SIZE
            else:
                fav=(fp-pl[m])/TICK_SIZE; adv=(ph[m]-fp)/TICK_SIZE
            mf=max(mf,fav); ma=max(ma,adv)
            if m<nr:
                co+=trade["ofi_1mins"][m]; cv+=trade["volumes"][m]

            # SL check
            slh, tph = False, False
            if d==1:
                if pl[m]<=eff_slp: slh=True
                if ph[m]>=tpp+TICK_SIZE: tph=True
            else:
                if ph[m]>=eff_slp: slh=True
                if pl[m]<=tpp-TICK_SIZE: tph=True
            if slh and tph: slh,tph=True,False

            if slh:
                if trailed:
                    tp_pnl = -(MARKET_SLIPPAGE_TICKS+RT_COMMISSION_TICKS)
                else:
                    tp_pnl = -(eff_slt+MARKET_SLIPPAGE_TICKS+RT_COMMISSION_TICKS)
                pnls.append(tp_pnl); dates.append(trade["date"]); dirs_out.append(d)
                hold_durs.append(m); n_sl+=1; exited=True; break

            if tph:
                tp_pnl = tpt-RT_COMMISSION_TICKS
                pnls.append(tp_pnl); dates.append(trade["date"]); dirs_out.append(d)
                hold_durs.append(m); n_tp+=1; exited=True; break

            # Management check
            if m%check_interval==0 and mgmt_model is not None and m<nr:
                feat = np.array([_feat_vec(d,fp,tpt,slt,max_hold,m,pc,ph,pl,
                    trade["volumes"],trade["vwaps"],trade["signed_volumes"],
                    trade["spread_means"],trade["ofi_1mins"],nr,mf,ma,co,cv)],
                    dtype=np.float32)
                pw = mgmt_model.predict(feat, num_iteration=mgmt_model.best_iteration)[0]

                if pw < inv_cutoff:
                    cc = pc[m]
                    ef = (cc-TICK_SIZE) if d==1 else (cc+TICK_SIZE)
                    rp = (ef-fp)/TICK_SIZE*d
                    tp_pnl = rp-RT_COMMISSION_TICKS
                    pnls.append(tp_pnl); dates.append(trade["date"]); dirs_out.append(d)
                    hold_durs.append(m); n_early+=1; exited=True; ee_pnls.append(tp_pnl)
                    if bet=="sl":
                        ee_would_sl+=1; ee_saved.append(tp_pnl-bep)
                    elif bet=="tp":
                        ee_would_tp+=1
                    break

                if trail_to_breakeven and not trailed and pw>conf_cutoff:
                    u = (pc[m]-fp)/TICK_SIZE*d
                    if u>0:
                        eff_slp=fp; eff_slt=0; trailed=True; n_trail+=1

        if not exited:
            em = min(max_hold,nr-1); em=max(em,1)
            ec = pc[em]
            ef = (ec-TICK_SIZE) if d==1 else (ec+TICK_SIZE)
            rp = (ef-fp)/TICK_SIZE*d
            tp_pnl = rp-RT_COMMISSION_TICKS
            pnls.append(tp_pnl); dates.append(trade["date"]); dirs_out.append(d)
            hold_durs.append(em); n_time+=1

    total = n_tp+n_sl+n_time+n_early
    es = {"n_tp":n_tp,"n_sl":n_sl,"n_time_stop":n_time,"n_early_exit":n_early,
          "tp_rate":n_tp/max(total,1),"sl_rate":n_sl/max(total,1),
          "early_exit_rate":n_early/max(total,1),
          "avg_hold_min":float(np.mean(hold_durs)) if hold_durs else 0}
    pa = np.array(pnls)
    if len(pa)>0:
        w=pa[pa>0]; l=pa[pa<0]
        es["avg_winner"]=float(w.mean()) if len(w)>0 else 0
        es["avg_loser"]=float(l.mean()) if len(l)>0 else 0

    ms = {"early_exit_count":n_early,"early_exit_pct":n_early/max(total,1),
          "trailed_count":n_trail,"trailed_pct":n_trail/max(total,1),
          "true_positives":ee_would_sl,"true_positive_rate":ee_would_sl/max(n_early,1),
          "false_positives":ee_would_tp,"false_positive_rate":ee_would_tp/max(n_early,1),
          "avg_ticks_saved_vs_sl":float(np.mean(ee_saved)) if ee_saved else 0,
          "avg_early_exit_pnl":float(np.mean(ee_pnls)) if ee_pnls else 0}

    return pnls, dates, dirs_out, es, ms


# ═══════════════════════════════════════════════════════════════════
#  DAILY STATS + REGIME
# ═══════════════════════════════════════════════════════════════════

def compute_daily_stats(pnl_arr, dates, directions, label, **kw):
    if len(pnl_arr)==0:
        return {"error":"no trades"}
    wr = np.mean(pnl_arr>0)
    pf = np.sum(pnl_arr[pnl_arr>0])/max(-np.sum(pnl_arr[pnl_arr<0]),1e-6)
    lm = directions==1; sm = directions==-1
    lp = pnl_arr[lm] if lm.any() else np.array([])
    sp_arr = pnl_arr[sm] if sm.any() else np.array([])
    tdf = pd.DataFrame({"pnl":pnl_arr,"date":dates})
    dp = tdf.groupby("date")["pnl"].agg(["sum","count"]).reset_index()
    dp.columns = ["date","daily_pnl","daily_trades"]
    ds, dso = 0.0, 0.0
    if len(dp)>2:
        dm = dp["daily_pnl"].mean()
        dd = dp["daily_pnl"].std()
        ds = float(dm/max(dd,1e-6)*np.sqrt(252))
        dds = np.sqrt(np.mean(np.minimum(dp["daily_pnl"].values,0)**2))
        dso = float(dm/max(dds,1e-6)*np.sqrt(252))
    cp = np.cumsum(pnl_arr)
    mdd = float(np.min(cp-np.maximum.accumulate(cp)))
    dc = 0.0
    if len(dp)>0:
        ta = dp["daily_pnl"].abs().sum()
        if ta>0: dc = float(dp["daily_pnl"].abs().max()/ta)
    r = {"label":label,"n_trades":len(pnl_arr),
         "total_pnl_ticks":float(pnl_arr.sum()),"total_pnl_dollars":float(pnl_arr.sum()*TICK_VALUE),
         "avg_pnl_ticks":float(pnl_arr.mean()),"daily_sharpe":ds,"daily_sortino":dso,
         "win_rate":float(wr),"profit_factor":float(pf),
         "max_dd_ticks":mdd,"n_trading_days":int(len(dp)),
         "trades_per_day":float(len(pnl_arr)/max(len(dp),1)),
         "day_concentration":dc,"day_concentration_pass":dc<=0.70,
         "long_trades":int(len(lp)),"long_wr":float(np.mean(lp>0)) if len(lp)>0 else 0,
         "short_trades":int(len(sp_arr)),"short_wr":float(np.mean(sp_arr>0)) if len(sp_arr)>0 else 0}
    for k,v in kw.items():
        if v is not None:
            r[k] = v
    return r


def full_regime_analysis(pnl_arr, dates, directions, day_returns):
    if len(pnl_arr)<20:
        return {"error":"too few","regime_gap_pass":False}
    dc = {}
    for d,ret in day_returns.items():
        if ret>0.001: dc[d]="green"
        elif ret<-0.001: dc[d]="red"
        else: dc[d]="flat"
    tdf = pd.DataFrame({"pnl":pnl_arr,"date":dates})
    da = tdf.groupby("date").agg(daily_pnl=("pnl","sum"),n_trades=("pnl","count")).reset_index()
    da["regime"] = da["date"].map(lambda d: dc.get(d,"flat"))
    rs = {}; rr = {}
    for regime in ["green","red","flat"]:
        mask = da["regime"]==regime
        if mask.sum()<3:
            rr[regime]={"n_days":int(mask.sum()),"skip":True}; continue
        rp = da[mask]["daily_pnl"].values
        s = float(rp.mean()/max(rp.std(),1e-6)*np.sqrt(252))
        dd = np.sqrt(np.mean(np.minimum(rp,0)**2))
        so = float(rp.mean()/max(dd,1e-6)*np.sqrt(252))
        rs[regime] = s
        rr[regime] = {"n_days":int(mask.sum()),"sharpe":s,"sortino":so,
                      "total_pnl":float(rp.sum()),"avg_daily":float(rp.mean())}
    if "green" in rs and "red" in rs:
        sg,sr = rs["green"],rs["red"]
        den = max(abs(sg),abs(sr),1e-6)
        gap = abs(sg-sr)/den
        rr["regime_gap"]=float(gap)
        rr["regime_gap_pass"]=gap<=0.50
        rr["regime_gap_detail"]=f"green={sg:.2f}, red={sr:.2f}, gap={gap:.2f} {'PASS' if gap<=0.50 else 'FAIL'}"
    else:
        rr["regime_gap"]=float("nan"); rr["regime_gap_pass"]=False
    return {"regimes":rr, "n_total_days":len(da)}


def clean_for_json(obj):
    if isinstance(obj,dict): return {k:clean_for_json(v) for k,v in obj.items()}
    if isinstance(obj,list): return [clean_for_json(v) for v in obj]
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return float(obj)
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,(np.bool_,)): return bool(obj)
    if isinstance(obj,pd.Timestamp): return str(obj)
    return obj


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    _import_lightgbm()
    t0 = time.time()

    mlflow_active = False
    mlflow = None
    try:
        import mlflow as _mlflow
        mlflow = _mlflow
        mlflow.set_tracking_uri("http://neptune:5000")
        mlflow.set_experiment("intrade_management_v1")
        mlflow_run = mlflow.start_run(run_name=f"intrade_mgmt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow_active = True
        log.info(f"MLflow run started: {mlflow_run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow unavailable: {e}")

    # Phase 1: Data
    log.info("="*70)
    log.info("PHASE 1: Loading data")
    log.info("="*70)

    minute_df = load_all_minute_bars()
    bars_30m = aggregate_to_bars(minute_df, bar_size_min=30)
    bars_30m = add_rolling_features(bars_30m, bar_size_min=30)
    bars_30m = add_forward_labels(bars_30m, horizon_bars=1, horizon_label="30min", min_edge_ticks=2.5)
    feature_cols = get_feature_columns(bars_30m)
    log.info(f"Features: {len(feature_cols)}")

    day_close = bars_30m.groupby("date")["close"].last()
    day_returns = day_close.pct_change().to_dict()

    # Phase 2: Entry predictions
    log.info("\n"+"="*70)
    log.info("PHASE 2: Entry model predictions")
    log.info("="*70)

    dates_30m = sorted(bars_30m["date"].unique())
    labels_all = bars_30m["fwd_ticks_30min"].values.astype(np.float32)
    dates_all = bars_30m["date"].values

    cached_preds_path = MFE_MAE_DIR / "entry_predictions.npz"
    entry_preds = None
    if cached_preds_path.exists():
        try:
            cached = np.load(str(cached_preds_path), allow_pickle=True)
            if len(cached["entry_preds"])==len(bars_30m) and np.array_equal(cached["dates"],dates_all):
                entry_preds = cached["entry_preds"]
                log.info(f"Loaded cached predictions: {(~np.isnan(entry_preds)).sum()} valid")
        except:
            pass

    if entry_preds is None:
        log.info("Training walk-forward entry model...")
        fa = bars_30m[feature_cols].values.astype(np.float32)
        entry_preds = np.full(len(bars_30m), np.nan, dtype=np.float32)
        fi = 0
        for fs in range(TRAIN_DAYS, len(dates_30m)-5+1, SLIDE_DAYS):
            ftd = dates_30m[fs-TRAIN_DAYS:fs]
            fvd = dates_30m[fs:fs+5]
            if len(fvd)<5: break
            fi += 1
            tm = np.isin(dates_all,ftd); vm = np.isin(dates_all,fvd)
            if not leakage_audit(list(ftd),list(fvd),feature_cols): continue
            tr = fa[tm].copy(); vl = fa[vm].copy()
            med = np.nanmedian(tr,axis=0)
            iqr = np.nanpercentile(tr,75,axis=0)-np.nanpercentile(tr,25,axis=0)
            iqr[iqr<1e-8]=1.0
            tr = np.clip(np.nan_to_num((tr-med)/iqr,nan=0),-5,5)
            vl = np.clip(np.nan_to_num((vl-med)/iqr,nan=0),-5,5)
            _,_,vp = train_entry_model(tr,labels_all[tm],vl,labels_all[vm],feature_cols,fi)
            entry_preds[vm] = vp
        log.info(f"Entry model: {fi} folds, {(~np.isnan(entry_preds)).sum()} valid")

    # Phase 3: Management WF + Sweep
    log.info("\n"+"="*70)
    log.info(f"PHASE 3: Management WF + {TOTAL_MGMT_CONFIGS} configs")
    log.info("="*70)

    all_trades, fill_stats = reconstruct_entry_fills(
        bars_30m, minute_df, entry_preds,
        confidence_pct=BASE_CONFIG["entry_threshold"],
        cancel_window_min=BASE_CONFIG["cancel_window"])
    log.info(f"Total entry fills: {len(all_trades)}")

    if len(all_trades)<30:
        log.error("Too few trades"); return

    # Analyze trade durations
    base_all = simulate_base_strategy(all_trades,
        tp_long=BASE_CONFIG["tp_long"], tp_short=BASE_CONFIG["tp_short"],
        sl_long=BASE_CONFIG["sl_long"], sl_short=BASE_CONFIG["sl_short"],
        max_hold_minutes=BASE_CONFIG["max_hold"])
    exit_mins = [t["exit_minute"] for t in base_all]
    sl_mins = [t["exit_minute"] for t in base_all if t["exit_type"]=="sl"]
    tp_mins = [t["exit_minute"] for t in base_all if t["exit_type"]=="tp"]
    log.info(f"Trade duration stats:")
    log.info(f"  All: median={np.median(exit_mins):.0f}min, mean={np.mean(exit_mins):.0f}min")
    log.info(f"  SL hits: n={len(sl_mins)}, median={np.median(sl_mins):.0f}min, mean={np.mean(sl_mins):.0f}min")
    log.info(f"  TP hits: n={len(tp_mins)}, median={np.median(tp_mins):.0f}min, mean={np.mean(tp_mins):.0f}min")
    log.info(f"  SL<3min: {sum(1 for m in sl_mins if m<3)}/{len(sl_mins)}")
    log.info(f"  Outcome rates: TP={sum(1 for t in base_all if t['exit_type']=='tp')}, "
             f"SL={sum(1 for t in base_all if t['exit_type']=='sl')}, "
             f"time={sum(1 for t in base_all if t['exit_type']=='time_stop')}")

    trade_dates = np.array([t["date"] for t in all_trades])
    unique_trade_dates = sorted(set(trade_dates))

    all_results = []
    base_pnls_all, base_dates_all, base_dirs_all = [], [], []
    all_oot_aucs = []
    feature_importances = []

    for ci_idx, check_interval in enumerate(CHECK_INTERVAL_SWEEP):
        log.info(f"\n{'='*50}")
        log.info(f"Check interval: {check_interval} min")
        log.info(f"{'='*50}")

        fold_data = []
        fold_idx = 0

        for fold_start in range(TRAIN_DAYS, len(unique_trade_dates)-SLIDE_DAYS+1, SLIDE_DAYS):
            ftd = set(unique_trade_dates[max(0,fold_start-TRAIN_DAYS):fold_start])
            fod = set(unique_trade_dates[fold_start:fold_start+SLIDE_DAYS])
            if len(fod)<1: continue
            fold_idx += 1

            train_trades = [t for t in all_trades if t["date"] in ftd]
            oot_trades = [t for t in all_trades if t["date"] in fod]
            if len(train_trades)<10 or len(oot_trades)<1:
                fold_data.append((fod,None,0.5,{},{})); continue

            train_results = simulate_base_strategy(train_trades,
                tp_long=BASE_CONFIG["tp_long"],tp_short=BASE_CONFIG["tp_short"],
                sl_long=BASE_CONFIG["sl_long"],sl_short=BASE_CONFIG["sl_short"],
                max_hold_minutes=BASE_CONFIG["max_hold"])

            X_tr, y_tr, w_tr = build_management_samples(
                train_results, check_interval=check_interval, max_hold=BASE_CONFIG["max_hold"])

            if len(X_tr)<30:
                fold_data.append((fod,None,0.5,{},{})); continue

            # Weighted positive rate (should reflect true WR now)
            wpos = np.sum(w_tr[y_tr==1])
            wtot = np.sum(w_tr)
            wpr = wpos/wtot if wtot>0 else 0.5

            # Train with sample weights
            params = {**LGBM_MGMT_PARAMS, "seed":42+fold_idx}
            n_tr = int(len(X_tr)*0.8)
            td = lgb.Dataset(X_tr[:n_tr], label=y_tr[:n_tr], weight=w_tr[:n_tr],
                           feature_name=MGMT_FEATURE_NAMES)
            vd = lgb.Dataset(X_tr[n_tr:], label=y_tr[n_tr:], weight=w_tr[n_tr:],
                           feature_name=MGMT_FEATURE_NAMES, reference=td)
            cb = [lgb.early_stopping(stopping_rounds=20,verbose=False),lgb.log_evaluation(period=0)]
            mgmt_model = lgb.train(params,td,num_boost_round=300,valid_sets=[vd],callbacks=cb)

            imp = dict(zip(MGMT_FEATURE_NAMES, mgmt_model.feature_importance("gain")))
            feature_importances.append(imp)

            # Train pred distribution for percentile thresholds
            train_preds = mgmt_model.predict(X_tr, num_iteration=mgmt_model.best_iteration)
            inv_cutoffs = {p:float(np.percentile(train_preds,p)) for p in INVALIDATION_PCTILE_SWEEP}
            conf_cutoffs = {p:float(np.percentile(train_preds,p)) for p in CONFIRMATION_PCTILE_SWEEP}

            # OOT AUC
            oot_results = simulate_base_strategy(oot_trades,
                tp_long=BASE_CONFIG["tp_long"],tp_short=BASE_CONFIG["tp_short"],
                sl_long=BASE_CONFIG["sl_long"],sl_short=BASE_CONFIG["sl_short"],
                max_hold_minutes=BASE_CONFIG["max_hold"])
            X_oot, y_oot, w_oot = build_management_samples(
                oot_results, check_interval=check_interval, max_hold=BASE_CONFIG["max_hold"])

            oot_auc = 0.5
            if len(X_oot)>5 and len(np.unique(y_oot))>1:
                op = mgmt_model.predict(X_oot, num_iteration=mgmt_model.best_iteration)
                try:
                    oot_auc = roc_auc_score(y_oot, op, sample_weight=w_oot)
                except:
                    oot_auc = 0.5
                all_oot_aucs.append(oot_auc)

            if fold_idx<=3 or fold_idx%5==0:
                log.info(f"  Fold {fold_idx}: samples={len(X_tr)}, "
                         f"weighted_pos_rate={wpr:.2f}, oot_auc={oot_auc:.3f}, "
                         f"inv10={inv_cutoffs.get(10,0):.3f}, inv30={inv_cutoffs.get(30,0):.3f}, "
                         f"range=[{train_preds.min():.3f},{train_preds.max():.3f}]")

            fold_data.append((fod,mgmt_model,oot_auc,inv_cutoffs,conf_cutoffs))

            # Collect base pnls (once)
            if ci_idx==0:
                for tr in oot_results:
                    base_pnls_all.append(tr["trade_pnl"])
                    base_dates_all.append(tr["date"])
                    base_dirs_all.append(tr["direction"])

        log.info(f"  Trained {fold_idx} folds")

        # Sweep
        for inv_p in INVALIDATION_PCTILE_SWEEP:
            for conf_p in CONFIRMATION_PCTILE_SWEEP:
                for trail in TRAIL_TO_BREAKEVEN_SWEEP:
                    label = f"ci{check_interval}_inv{inv_p}_conf{conf_p}_trail{int(trail)}"
                    amp, amd, amdi = [], [], []
                    aes, ams = [], []

                    for fod,mm,_,ic,cc in fold_data:
                        ot = [t for t in all_trades if t["date"] in fod]
                        if not ot: continue
                        inv_c = ic.get(inv_p,0.0)
                        conf_c = cc.get(conf_p,1.0)
                        p,td2,di,es,ms = simulate_managed_exit(
                            ot,mm,BASE_CONFIG["tp_long"],BASE_CONFIG["tp_short"],
                            BASE_CONFIG["sl_long"],BASE_CONFIG["sl_short"],
                            BASE_CONFIG["max_hold"],inv_c,conf_c,trail,check_interval)
                        amp.extend(p); amd.extend(td2); amdi.extend(di)
                        aes.append(es); ams.append(ms)

                    if len(amp)<20: continue
                    pa = np.array(amp); da = np.array(amd); dia = np.array(amdi)

                    # Aggregate
                    ag_m = {
                        "early_exit_count":sum(s["early_exit_count"] for s in ams),
                        "trailed_count":sum(s["trailed_count"] for s in ams),
                        "true_positives":sum(s["true_positives"] for s in ams),
                        "false_positives":sum(s["false_positives"] for s in ams),
                    }
                    te = ag_m["early_exit_count"]
                    ag_m["early_exit_pct"]=te/max(len(pa),1)
                    ag_m["trailed_pct"]=ag_m["trailed_count"]/max(len(pa),1)
                    ag_m["true_positive_rate"]=ag_m["true_positives"]/max(te,1)
                    ag_m["false_positive_rate"]=ag_m["false_positives"]/max(te,1)
                    sv2 = [s["avg_ticks_saved_vs_sl"] for s in ams if s["avg_ticks_saved_vs_sl"]!=0]
                    ag_m["avg_ticks_saved"]=float(np.mean(sv2)) if sv2 else 0
                    ep2 = [s["avg_early_exit_pnl"] for s in ams if s["avg_early_exit_pnl"]!=0]
                    ag_m["avg_early_exit_pnl"]=float(np.mean(ep2)) if ep2 else 0

                    ag_e = {
                        "n_tp":sum(s["n_tp"] for s in aes),
                        "n_sl":sum(s["n_sl"] for s in aes),
                        "n_time_stop":sum(s["n_time_stop"] for s in aes),
                        "n_early_exit":sum(s["n_early_exit"] for s in aes),
                    }
                    tt = sum(ag_e.values())
                    ag_e["tp_rate"]=ag_e["n_tp"]/max(tt,1)
                    ag_e["sl_rate"]=ag_e["n_sl"]/max(tt,1)
                    ag_e["early_exit_rate"]=ag_e["n_early_exit"]/max(tt,1)

                    result = compute_daily_stats(pa,da,dia,label,
                        exit_stats=ag_e,fill_stats=fill_stats,mgmt_stats=ag_m)
                    if "error" in result: continue

                    regime = full_regime_analysis(pa,da,dia,day_returns)
                    result["regime_analysis"] = regime
                    result["config"] = {"check_interval":check_interval,
                        "invalidation_pctile":inv_p,"confirmation_pctile":conf_p,
                        "trail_to_breakeven":trail,**BASE_CONFIG}
                    all_results.append(result)

                    log.info(f"  {label}: Sharpe={result['daily_sharpe']:.2f}, "
                             f"Sortino={result['daily_sortino']:.2f}, "
                             f"WR={result['win_rate']:.1%}, PF={result['profit_factor']:.2f}, "
                             f"trades={result['n_trades']}, "
                             f"early_exit={ag_m['early_exit_pct']:.1%} ({te}/{len(pa)}), "
                             f"TP_rate={ag_m['true_positive_rate']:.1%}, "
                             f"FP_rate={ag_m['false_positive_rate']:.1%}, "
                             f"saved={ag_m['avg_ticks_saved']:.2f}t")

    # Phase 4: Results
    log.info("\n"+"="*70)
    log.info("PHASE 4: Results")
    log.info("="*70)

    bpa = np.array(base_pnls_all); bda = np.array(base_dates_all); bdia = np.array(base_dirs_all)
    base_stats = compute_daily_stats(bpa,bda,bdia,"BASE")
    base_regime = full_regime_analysis(bpa,bda,bdia,day_returns)
    base_stats["regime_analysis"] = base_regime

    log.info(f"\nBASE: Sharpe={base_stats['daily_sharpe']:.2f}, Sortino={base_stats['daily_sortino']:.2f}, "
             f"WR={base_stats['win_rate']:.1%}, PF={base_stats['profit_factor']:.2f}, trades={base_stats['n_trades']}")
    rg = base_regime.get("regimes",{})
    if "regime_gap_detail" in rg:
        log.info(f"  Regime: {rg['regime_gap_detail']}")

    vr = sorted([r for r in all_results if r.get("daily_sharpe",0)!=0],
                key=lambda x:x["daily_sharpe"], reverse=True)
    rp = [r for r in vr if r.get("regime_analysis",{}).get("regimes",{}).get("regime_gap_pass",False)]

    log.info(f"\nTotal configs: {len(vr)}, Regime-passing: {len(rp)}")

    log.info("\n--- TOP 10 MANAGED (by Sharpe) ---")
    for i,r in enumerate(vr[:10]):
        m = r.get("mgmt_stats",{})
        rr = r.get("regime_analysis",{}).get("regimes",{})
        gp = rr.get("regime_gap_pass",False)
        g = rr.get("regime_gap",0)
        log.info(f"  #{i+1}: {r['label']} | Sharpe={r['daily_sharpe']:.2f} Sortino={r['daily_sortino']:.2f} "
                 f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} n={r['n_trades']} | "
                 f"early={m.get('early_exit_pct',0):.1%} TP_rate={m.get('true_positive_rate',0):.1%} "
                 f"FP_rate={m.get('false_positive_rate',0):.1%} saved={m.get('avg_ticks_saved',0):.2f}t | "
                 f"gap={g:.2f} {'PASS' if gp else 'FAIL'}")

    if rp:
        log.info("\n--- TOP 5 REGIME-PASSING ---")
        for i,r in enumerate(rp[:5]):
            m = r.get("mgmt_stats",{})
            g = r.get("regime_analysis",{}).get("regimes",{}).get("regime_gap",0)
            log.info(f"  #{i+1}: {r['label']} | Sharpe={r['daily_sharpe']:.2f} Sortino={r['daily_sortino']:.2f} "
                     f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} | "
                     f"early={m.get('early_exit_pct',0):.1%} TP_rate={m.get('true_positive_rate',0):.1%} | gap={g:.2f}")

    if feature_importances:
        ai = {}
        for imp in feature_importances:
            for k,v in imp.items(): ai[k]=ai.get(k,0)+v
        for k in ai: ai[k]/=len(feature_importances)
        tf = sorted(ai.items(), key=lambda x:x[1], reverse=True)[:10]
        log.info("\n--- TOP FEATURES ---")
        for n,s in tf:
            log.info(f"  {n}: {s:.1f}")

    bm = vr[0] if vr else None
    brp = rp[0] if rp else None
    if bm:
        imp = bm["daily_sharpe"]-base_stats["daily_sharpe"]
        log.info(f"\nBest managed improvement: {imp:+.2f} ({base_stats['daily_sharpe']:.2f} -> {bm['daily_sharpe']:.2f})")
    if brp:
        imp = brp["daily_sharpe"]-base_stats["daily_sharpe"]
        log.info(f"Best regime-passing improvement: {imp:+.2f} ({base_stats['daily_sharpe']:.2f} -> {brp['daily_sharpe']:.2f})")

    if all_oot_aucs:
        log.info(f"\nMgmt classifier OOT AUC: mean={np.mean(all_oot_aucs):.4f}, "
                 f"median={np.median(all_oot_aucs):.4f}, std={np.std(all_oot_aucs):.4f}, n={len(all_oot_aucs)}")

    # Save
    output = clean_for_json({
        "timestamp":datetime.now().isoformat(),
        "base_config":BASE_CONFIG, "base_stats":base_stats,
        "management_results":vr[:20], "regime_passing_results":rp[:10],
        "management_classifier_auc":{
            "mean":float(np.mean(all_oot_aucs)) if all_oot_aucs else 0.5,
            "median":float(np.median(all_oot_aucs)) if all_oot_aucs else 0.5,
            "std":float(np.std(all_oot_aucs)) if all_oot_aucs else 0,
            "n_folds":len(all_oot_aucs)},
        "feature_importances":ai if feature_importances else {},
        "total_configs":len(vr),"regime_passing_count":len(rp),
        "elapsed_seconds":time.time()-t0})

    op = OUTPUT_DIR/"results.json"
    with open(op,"w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nSaved to {op}")

    if mlflow_active:
        mlflow.log_metric("base_sharpe",base_stats.get("daily_sharpe",0))
        mlflow.log_metric("base_sortino",base_stats.get("daily_sortino",0))
        if all_oot_aucs:
            mlflow.log_metric("mgmt_auc_mean",np.mean(all_oot_aucs))
        if bm: mlflow.log_metric("best_managed_sharpe",bm["daily_sharpe"])
        if brp: mlflow.log_metric("best_regime_pass_sharpe",brp["daily_sharpe"])
        mlflow.log_metric("n_regime_passing",len(rp))
        try: mlflow.log_artifact(str(op))
        except: pass
        mlflow.end_run()

    log.info(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
