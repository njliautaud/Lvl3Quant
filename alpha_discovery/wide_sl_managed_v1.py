#!/usr/bin/env python3
"""
Wide SL + In-Trade Management v1
===================================

HYPOTHESIS: Tight SL (4/3 ticks) fires too fast (~1min) for the management
classifier to work. Widen SL to [6,8,10,12] ticks, giving the management
model time to evaluate trades every 1 minute and cut losers before they
reach the full wider SL.

DESIGN:
  1. Entry: same base strategy — passive limit at bid/ask, top 5% confidence
  2. TP: passive limit, 25 ticks (unchanged)
  3. SL: wider — sweep [6, 8, 10, 12] ticks (market, 1 tick slippage)
  4. Management: every 1 min, LightGBM classifier predicts P(win).
     If P(win) < invalidation threshold → EXIT EARLY at market.

The management model is trained walk-forward ON the wider-SL trades:
  - For each WF fold, run wider-SL strategy on train data to generate trades
  - Every minute during each trade, record 20 features + outcome
  - Train LightGBM on those samples (per-trade weighting)
  - Apply management overlay on OOT fold

SWEEP: 4 SL × 4 invalidation_pctile × 2 trail_breakeven = 32 managed
       + 4 unmanaged wide SL + 1 tight-SL base = 37 total

Walk-forward: 30d train / 5d slide (SLIDING, HC #0)
FIFO fills: passive entry, passive TP, market SL (HC #74)
Cost: commission = 0.376 ticks RT, market slippage = 1 tick

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
OUTPUT_DIR = ROOT / "output" / "wide_sl_managed_v1"
MFE_MAE_DIR = ROOT / "output" / "mfe_mae_analysis"
LOG_DIR = ROOT / "logs"
sys.path.insert(0, str(ROOT))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [WIDE-SL-MGMT] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "wide_sl_managed_v1.log"), mode="w"),
    ],
)
log = logging.getLogger("WIDE-SL-MGMT")

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
TICK_SIZE = 1.0
TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_SLIPPAGE_TICKS = 1.0
TRAIN_DAYS = 30
SLIDE_DAYS = 5
ENTRY_BAR_SIZE_MIN = 30
CHECK_INTERVAL = 1  # management check every 1 minute

BASE_CONFIG = {
    "tp_long": 25, "tp_short": 25,
    "entry_threshold": 0.05, "cancel_window": 10,
    "max_hold": 60,
}

# Tight SL reference (the current champion)
TIGHT_SL_LONG = 4
TIGHT_SL_SHORT = 3

# Wide SL sweep
SL_SWEEP = [6, 8, 10, 12]
INVALIDATION_PCTILE_SWEEP = [10, 20, 30, 40]
TRAIL_BREAKEVEN_SWEEP = [True, False]

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
#  DATA LOADING (reused from intrade_management_v1)
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
#  ENTRY FILL RECONSTRUCTION (with extra fields for management)
# ═══════════════════════════════════════════════════════════════════

def reconstruct_entry_fills(bars_30m, minute_df, pred_30m, confidence_pct, cancel_window_min):
    """Reconstruct FIFO passive entry fills. Stores extra minute-bar arrays
    needed for in-trade management feature computation."""
    vm = ~np.isnan(pred_30m)
    vp = pred_30m[vm]
    if len(vp) < 20:
        return [], {"error": "too few"}
    ut = np.nanquantile(pred_30m[vm], 1 - confidence_pct)
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
        if pred_30m[i] >= ut:
            d = 1
        elif pred_30m[i] <= lt:
            d = -1
        else:
            continue
        ns += 1
        ds = bdt[i]
        sts = pd.Timestamp(bts[i])
        sp = bcl[i]
        if ds not in ml:
            continue
        dm = ml[ds]
        dts = dm["ts_minute"].values
        lp = sp
        sbe = sts + pd.Timedelta(minutes=ENTRY_BAR_SIZE_MIN)
        cts = sts + pd.Timedelta(minutes=cancel_window_min + ENTRY_BAR_SIZE_MIN)
        fm = (dts >= np.datetime64(sbe)) & (dts <= np.datetime64(cts))
        fc = dm[fm]
        if len(fc) == 0:
            nc += 1
            continue
        filled = False
        for j, (_, mb) in enumerate(fc.iterrows()):
            if d == 1:
                if mb["low"] <= lp - TICK_SIZE:
                    filled = True
                    fp = lp
                    fts = mb["ts_minute"]
                    fd.append(j + 1)
                    break
            else:
                if mb["high"] >= lp + TICK_SIZE:
                    filled = True
                    fp = lp
                    fts = mb["ts_minute"]
                    fd.append(j + 1)
                    break
        if not filled:
            nc += 1
            continue
        nf += 1
        rm = dm[dts >= np.datetime64(fts)]
        if len(rm) < 2:
            nf -= 1
            nc += 1
            continue
        trades.append({
            "idx": i, "date": ds, "signal_ts": sts,
            "fill_ts": pd.Timestamp(fts), "fill_price": fp,
            "fill_delay_minutes": fd[-1], "direction": d,
            "pred_30m": float(pred_30m[i]),
            "entry_confidence": float(abs(pred_30m[i])),
            "prices_close": rm["close"].values.copy(),
            "prices_high": rm["high"].values.copy(),
            "prices_low": rm["low"].values.copy(),
            "times": rm["ts_minute"].values.copy(),
            "volumes": rm["volume"].values.copy(),
            "vwaps": rm["vwap"].values.copy(),
            "signed_volumes": rm["signed_volume"].values.copy(),
            "spread_means": rm["spread_mean"].values.copy(),
            "ofi_1mins": rm["ofi_1min"].values.copy(),
            "trade_counts": rm["trade_count"].values.copy(),
            "n_remaining_minutes": len(rm),
        })
    fs = {"n_signals": ns, "n_filled": nf, "n_cancelled": nc,
          "fill_rate": nf / max(ns, 1), "avg_fill_delay": float(np.mean(fd)) if fd else 0}
    log.info(f"  Entry fills: signals={ns}, filled={nf} ({fs['fill_rate']:.1%})")
    return trades, fs


# ═══════════════════════════════════════════════════════════════════
#  BASE STRATEGY SIMULATION (parameterized SL)
# ═══════════════════════════════════════════════════════════════════

def simulate_base_strategy(trades, tp_long, tp_short, sl_long, sl_short, max_hold_minutes):
    """Simulate TP/SL strategy. Returns list of trade result dicts."""
    results = []
    for trade in trades:
        d = trade["direction"]
        fp = trade["fill_price"]
        pc = trade["prices_close"]
        ph = trade["prices_high"]
        pl = trade["prices_low"]
        nr = trade["n_remaining_minutes"]
        tpt = tp_long if d == 1 else tp_short
        slt = sl_long if d == 1 else sl_short
        tpp = fp + d * tpt * TICK_SIZE
        slp = fp - d * slt * TICK_SIZE
        et = None
        em = None
        tp_pnl = None
        mc = min(max_hold_minutes, nr)
        for m in range(1, mc):
            slh, tph = False, False
            if d == 1:
                if pl[m] <= slp:
                    slh = True
                if ph[m] >= tpp + TICK_SIZE:
                    tph = True
            else:
                if ph[m] >= slp:
                    slh = True
                if pl[m] <= tpp - TICK_SIZE:
                    tph = True
            if slh and tph:
                slh, tph = True, False
            if slh:
                tp_pnl = -(slt + MARKET_SLIPPAGE_TICKS + RT_COMMISSION_TICKS)
                et = "sl"
                em = m
                break
            if tph:
                tp_pnl = tpt - RT_COMMISSION_TICKS
                et = "tp"
                em = m
                break
        if et is None:
            em = min(max_hold_minutes, nr - 1)
            em = max(em, 1)
            ec = pc[em]
            ef = (ec - TICK_SIZE) if d == 1 else (ec + TICK_SIZE)
            tp_pnl = (ef - fp) / TICK_SIZE * d - RT_COMMISSION_TICKS
            et = "time_stop"
        results.append({
            **trade,
            "tp_ticks": tpt, "sl_ticks": slt,
            "exit_type": et, "exit_minute": em, "trade_pnl": tp_pnl,
            "outcome": 1 if et == "tp" else 0,
        })
    return results


# ═══════════════════════════════════════════════════════════════════
#  MANAGEMENT FEATURES (20 features)
# ═══════════════════════════════════════════════════════════════════

MGMT_FEATURE_NAMES = [
    "unrealized_pnl_ticks", "pnl_pct_to_tp", "pnl_pct_to_sl",
    "max_favorable_so_far", "max_adverse_so_far",
    "minutes_elapsed", "minutes_remaining",
    "pnl_velocity",
    "ofi_cumulative", "ofi_direction_match",
    "volume_total", "volume_trend_recent",
    "spread_mean_recent",
    "signed_volume_direction_match",
    "vwap_relative",
    "trade_count_recent", "momentum_2min", "momentum_5min",
    "entry_confidence",
    "side",
]


def _compute_mgmt_features(d, fp, tpt, slt, max_hold, m, pc, ph, pl,
                            vol, vw, sv, sp, ofi, tc, nr, mf, ma,
                            cum_ofi, cum_vol, entry_conf):
    """Compute 20 management features at minute m of trade."""
    cc = pc[m]
    u = (cc - fp) / TICK_SIZE * d  # unrealized PnL in ticks

    # Recent lookback (last 3 min or available)
    lb = min(3, m)
    recent_vol = vol[max(0, m - lb):m + 1]
    recent_sp = sp[max(0, m - lb):m + 1]
    recent_sv = sv[max(0, m - lb):m + 1]
    recent_tc = tc[max(0, m - lb):m + 1]

    # Momentum
    mom_2 = (pc[m] - pc[max(0, m - 2)]) / TICK_SIZE * d if m >= 2 else 0.0
    mom_5 = (pc[m] - pc[max(0, m - 5)]) / TICK_SIZE * d if m >= 5 else 0.0

    # Velocity: PnL change over last 2 minutes
    if m >= 2:
        u_prev = (pc[m - 2] - fp) / TICK_SIZE * d
        vel = (u - u_prev) / 2.0
    else:
        vel = u / max(m, 1)

    return [
        u,                                              # unrealized_pnl_ticks
        u / tpt if tpt > 0 else 0,                     # pnl_pct_to_tp
        u / slt if slt > 0 else 0,                     # pnl_pct_to_sl (negative when adverse)
        mf,                                             # max_favorable_so_far
        ma,                                             # max_adverse_so_far
        float(m),                                       # minutes_elapsed
        float(max_hold - m),                            # minutes_remaining
        vel,                                            # pnl_velocity
        cum_ofi,                                        # ofi_cumulative
        1.0 if cum_ofi * d > 0 else -1.0,              # ofi_direction_match
        cum_vol,                                        # volume_total
        _safe_polyfit_slope(recent_vol),                # volume_trend_recent
        float(np.mean(recent_sp)),                      # spread_mean_recent
        1.0 if np.sum(recent_sv) * d > 0 else -1.0,    # signed_volume_direction_match
        (cc - vw[m]) / TICK_SIZE * d if m < len(vw) and vw[m] > 0 else 0.0,  # vwap_relative
        float(np.mean(recent_tc)),                      # trade_count_recent
        mom_2,                                          # momentum_2min
        mom_5,                                          # momentum_5min
        entry_conf,                                     # entry_confidence
        1.0 if d == -1 else 0.0,                        # side (1=short, 0=long)
    ]


def build_management_samples(trade_results, max_hold):
    """Build management training samples with per-trade weighting.
    Each trade contributes total weight = 1.0 regardless of duration."""
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
        entry_conf = tr.get("entry_confidence", 0.0)

        mf, ma, cum_ofi, cum_vol = 0.0, 0.0, 0.0, 0.0
        trade_feats = []

        for m in range(1, em):
            if m >= nr:
                break
            # Track MFE/MAE
            if d == 1:
                fav = (tr["prices_high"][m] - fp) / TICK_SIZE
                adv = (fp - tr["prices_low"][m]) / TICK_SIZE
            else:
                fav = (fp - tr["prices_low"][m]) / TICK_SIZE
                adv = (tr["prices_high"][m] - fp) / TICK_SIZE
            mf = max(mf, fav)
            ma = max(ma, adv)
            cum_ofi += tr["ofi_1mins"][m]
            cum_vol += tr["volumes"][m]

            # Check every minute (CHECK_INTERVAL=1)
            feat = _compute_mgmt_features(
                d, fp, tpt, slt, max_hold, m,
                tr["prices_close"], tr["prices_high"], tr["prices_low"],
                tr["volumes"], tr["vwaps"], tr["signed_volumes"],
                tr["spread_means"], tr["ofi_1mins"], tr["trade_counts"],
                nr, mf, ma, cum_ofi, cum_vol, entry_conf)
            trade_feats.append(feat)

        if trade_feats:
            n_samples = len(trade_feats)
            w = 1.0 / n_samples
            for feat in trade_feats:
                all_features.append(feat)
                all_labels.append(outcome)
                all_weights.append(w)

    if not all_features:
        return np.empty((0, len(MGMT_FEATURE_NAMES))), np.empty(0), np.empty(0)

    return (np.array(all_features, dtype=np.float32),
            np.array(all_labels, dtype=np.float32),
            np.array(all_weights, dtype=np.float32))


# ═══════════════════════════════════════════════════════════════════
#  MANAGED EXIT SIMULATION
# ═══════════════════════════════════════════════════════════════════

def simulate_managed_exit(trades, mgmt_model, tp_long, tp_short,
                          sl_long, sl_short, max_hold,
                          inv_cutoff, conf_cutoff, trail_to_breakeven):
    """Simulate trades with management overlay. Returns PnL arrays + stats."""
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
        entry_conf = trade.get("entry_confidence", 0.0)
        tpt = tp_long if d == 1 else tp_short
        slt = sl_long if d == 1 else sl_short
        tpp = fp + d * tpt * TICK_SIZE
        slp = fp - d * slt * TICK_SIZE
        eff_slp = slp
        eff_slt = slt
        trailed = False
        mc = min(max_hold, nr)
        mf, ma, cum_ofi, cum_vol = 0.0, 0.0, 0.0, 0.0

        # Compute base outcome for quality tracking
        bet = None
        bep = None
        for bm in range(1, mc):
            bsh, bth = False, False
            if d == 1:
                if pl[bm] <= slp: bsh = True
                if ph[bm] >= tpp + TICK_SIZE: bth = True
            else:
                if ph[bm] >= slp: bsh = True
                if pl[bm] <= tpp - TICK_SIZE: bth = True
            if bsh and bth: bsh, bth = True, False
            if bsh:
                bet = "sl"
                bep = -(slt + MARKET_SLIPPAGE_TICKS + RT_COMMISSION_TICKS)
                break
            if bth:
                bet = "tp"
                bep = tpt - RT_COMMISSION_TICKS
                break
        if bet is None:
            bet = "time_stop"
            bm_e = min(max_hold, nr - 1)
            bm_e = max(bm_e, 1)
            bc = pc[bm_e]
            bf = (bc - TICK_SIZE) if d == 1 else (bc + TICK_SIZE)
            bep = (bf - fp) / TICK_SIZE * d - RT_COMMISSION_TICKS

        # Managed simulation
        exited = False
        for m in range(1, mc):
            if d == 1:
                fav = (ph[m] - fp) / TICK_SIZE
                adv = (fp - pl[m]) / TICK_SIZE
            else:
                fav = (fp - pl[m]) / TICK_SIZE
                adv = (ph[m] - fp) / TICK_SIZE
            mf = max(mf, fav)
            ma = max(ma, adv)
            if m < nr:
                cum_ofi += trade["ofi_1mins"][m]
                cum_vol += trade["volumes"][m]

            # SL check
            slh, tph = False, False
            if d == 1:
                if pl[m] <= eff_slp: slh = True
                if ph[m] >= tpp + TICK_SIZE: tph = True
            else:
                if ph[m] >= eff_slp: slh = True
                if pl[m] <= tpp - TICK_SIZE: tph = True
            if slh and tph: slh, tph = True, False

            if slh:
                if trailed:
                    tp_pnl = -(MARKET_SLIPPAGE_TICKS + RT_COMMISSION_TICKS)
                else:
                    tp_pnl = -(eff_slt + MARKET_SLIPPAGE_TICKS + RT_COMMISSION_TICKS)
                pnls.append(tp_pnl)
                dates.append(trade["date"])
                dirs_out.append(d)
                hold_durs.append(m)
                n_sl += 1
                exited = True
                break

            if tph:
                tp_pnl = tpt - RT_COMMISSION_TICKS
                pnls.append(tp_pnl)
                dates.append(trade["date"])
                dirs_out.append(d)
                hold_durs.append(m)
                n_tp += 1
                exited = True
                break

            # Management check every minute
            if mgmt_model is not None and m < nr:
                feat = np.array([_compute_mgmt_features(
                    d, fp, tpt, slt, max_hold, m, pc, ph, pl,
                    trade["volumes"], trade["vwaps"], trade["signed_volumes"],
                    trade["spread_means"], trade["ofi_1mins"], trade["trade_counts"],
                    nr, mf, ma, cum_ofi, cum_vol, entry_conf)],
                    dtype=np.float32)
                pw = mgmt_model.predict(feat, num_iteration=mgmt_model.best_iteration)[0]

                if pw < inv_cutoff:
                    # Early exit at market
                    cc = pc[m]
                    ef = (cc - TICK_SIZE) if d == 1 else (cc + TICK_SIZE)
                    rp = (ef - fp) / TICK_SIZE * d
                    tp_pnl = rp - RT_COMMISSION_TICKS
                    pnls.append(tp_pnl)
                    dates.append(trade["date"])
                    dirs_out.append(d)
                    hold_durs.append(m)
                    n_early += 1
                    exited = True
                    ee_pnls.append(tp_pnl)
                    if bet == "sl":
                        ee_would_sl += 1
                        ee_saved.append(tp_pnl - bep)
                    elif bet == "tp":
                        ee_would_tp += 1
                    break

                if trail_to_breakeven and not trailed and pw > conf_cutoff:
                    u = (pc[m] - fp) / TICK_SIZE * d
                    if u > 0:
                        eff_slp = fp
                        eff_slt = 0
                        trailed = True
                        n_trail += 1

        if not exited:
            em = min(max_hold, nr - 1)
            em = max(em, 1)
            ec = pc[em]
            ef = (ec - TICK_SIZE) if d == 1 else (ec + TICK_SIZE)
            rp = (ef - fp) / TICK_SIZE * d
            tp_pnl = rp - RT_COMMISSION_TICKS
            pnls.append(tp_pnl)
            dates.append(trade["date"])
            dirs_out.append(d)
            hold_durs.append(em)
            n_time += 1

    total = n_tp + n_sl + n_time + n_early
    es = {
        "n_tp": n_tp, "n_sl": n_sl, "n_time_stop": n_time, "n_early_exit": n_early,
        "tp_rate": n_tp / max(total, 1), "sl_rate": n_sl / max(total, 1),
        "early_exit_rate": n_early / max(total, 1),
        "avg_hold_min": float(np.mean(hold_durs)) if hold_durs else 0,
    }
    pa = np.array(pnls)
    if len(pa) > 0:
        w = pa[pa > 0]
        l = pa[pa < 0]
        es["avg_winner"] = float(w.mean()) if len(w) > 0 else 0
        es["avg_loser"] = float(l.mean()) if len(l) > 0 else 0

    ms = {
        "early_exit_count": n_early,
        "early_exit_pct": n_early / max(total, 1),
        "trailed_count": n_trail,
        "trailed_pct": n_trail / max(total, 1),
        "true_positives": ee_would_sl,
        "true_positive_rate": ee_would_sl / max(n_early, 1),
        "false_positives": ee_would_tp,
        "false_positive_rate": ee_would_tp / max(n_early, 1),
        "avg_ticks_saved_vs_sl": float(np.mean(ee_saved)) if ee_saved else 0,
        "avg_early_exit_pnl": float(np.mean(ee_pnls)) if ee_pnls else 0,
    }

    return pnls, dates, dirs_out, es, ms


# ═══════════════════════════════════════════════════════════════════
#  DAILY STATS + REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def compute_daily_stats(pnl_arr, dates, directions, label, **kw):
    if len(pnl_arr) == 0:
        return {"error": "no trades"}
    wr = np.mean(pnl_arr > 0)
    pf = np.sum(pnl_arr[pnl_arr > 0]) / max(-np.sum(pnl_arr[pnl_arr < 0]), 1e-6)
    lm = directions == 1
    sm = directions == -1
    lp = pnl_arr[lm] if lm.any() else np.array([])
    sp_arr = pnl_arr[sm] if sm.any() else np.array([])
    tdf = pd.DataFrame({"pnl": pnl_arr, "date": dates})
    dp = tdf.groupby("date")["pnl"].agg(["sum", "count"]).reset_index()
    dp.columns = ["date", "daily_pnl", "daily_trades"]
    ds, dso = 0.0, 0.0
    if len(dp) > 2:
        dm = dp["daily_pnl"].mean()
        dd = dp["daily_pnl"].std()
        ds = float(dm / max(dd, 1e-6) * np.sqrt(252))
        dds = np.sqrt(np.mean(np.minimum(dp["daily_pnl"].values, 0) ** 2))
        dso = float(dm / max(dds, 1e-6) * np.sqrt(252))
    cp = np.cumsum(pnl_arr)
    mdd = float(np.min(cp - np.maximum.accumulate(cp)))
    dc = 0.0
    if len(dp) > 0:
        ta = dp["daily_pnl"].abs().sum()
        if ta > 0:
            dc = float(dp["daily_pnl"].abs().max() / ta)
    r = {
        "label": label, "n_trades": len(pnl_arr),
        "total_pnl_ticks": float(pnl_arr.sum()),
        "total_pnl_dollars": float(pnl_arr.sum() * TICK_VALUE),
        "avg_pnl_ticks": float(pnl_arr.mean()), "daily_sharpe": ds, "daily_sortino": dso,
        "win_rate": float(wr), "profit_factor": float(pf),
        "max_dd_ticks": mdd, "n_trading_days": int(len(dp)),
        "trades_per_day": float(len(pnl_arr) / max(len(dp), 1)),
        "day_concentration": dc, "day_concentration_pass": dc <= 0.70,
        "long_trades": int(len(lp)),
        "long_wr": float(np.mean(lp > 0)) if len(lp) > 0 else 0,
        "short_trades": int(len(sp_arr)),
        "short_wr": float(np.mean(sp_arr > 0)) if len(sp_arr) > 0 else 0,
    }
    for k, v in kw.items():
        if v is not None:
            r[k] = v
    return r


def full_regime_analysis(pnl_arr, dates, directions, day_returns):
    if len(pnl_arr) < 20:
        return {"error": "too few", "regime_gap_pass": False}
    dc = {}
    for d, ret in day_returns.items():
        if ret > 0.001:
            dc[d] = "green"
        elif ret < -0.001:
            dc[d] = "red"
        else:
            dc[d] = "flat"
    tdf = pd.DataFrame({"pnl": pnl_arr, "date": dates})
    da = tdf.groupby("date").agg(daily_pnl=("pnl", "sum"), n_trades=("pnl", "count")).reset_index()
    da["regime"] = da["date"].map(lambda d: dc.get(d, "flat"))
    rs = {}
    rr = {}
    for regime in ["green", "red", "flat"]:
        mask = da["regime"] == regime
        if mask.sum() < 3:
            rr[regime] = {"n_days": int(mask.sum()), "skip": True}
            continue
        rp = da[mask]["daily_pnl"].values
        s = float(rp.mean() / max(rp.std(), 1e-6) * np.sqrt(252))
        dd = np.sqrt(np.mean(np.minimum(rp, 0) ** 2))
        so = float(rp.mean() / max(dd, 1e-6) * np.sqrt(252))
        rs[regime] = s
        rr[regime] = {
            "n_days": int(mask.sum()), "sharpe": s, "sortino": so,
            "total_pnl": float(rp.sum()), "avg_daily": float(rp.mean()),
        }
    if "green" in rs and "red" in rs:
        sg, sr = rs["green"], rs["red"]
        den = max(abs(sg), abs(sr), 1e-6)
        gap = abs(sg - sr) / den
        rr["regime_gap"] = float(gap)
        rr["regime_gap_pass"] = gap <= 0.50
        rr["regime_gap_detail"] = f"green={sg:.2f}, red={sr:.2f}, gap={gap:.2f} {'PASS' if gap <= 0.50 else 'FAIL'}"
    else:
        rr["regime_gap"] = float("nan")
        rr["regime_gap_pass"] = False
    return {"regimes": rr, "n_total_days": len(da)}


def clean_for_json(obj):
    if isinstance(obj, dict): return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, pd.Timestamp): return str(obj)
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
        mlflow.set_experiment("wide_sl_managed_v1")
        mlflow_run = mlflow.start_run(
            run_name=f"wide_sl_mgmt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow_active = True
        log.info(f"MLflow run started: {mlflow_run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow unavailable: {e}")

    # ══════════════════════════════════════
    #  PHASE 1: Data
    # ══════════════════════════════════════
    log.info("=" * 70)
    log.info("PHASE 1: Loading data")
    log.info("=" * 70)

    minute_df = load_all_minute_bars()
    bars_30m = aggregate_to_bars(minute_df, bar_size_min=30)
    bars_30m = add_rolling_features(bars_30m, bar_size_min=30)
    bars_30m = add_forward_labels(bars_30m, horizon_bars=1, horizon_label="30min",
                                   min_edge_ticks=2.5)
    feature_cols = get_feature_columns(bars_30m)
    log.info(f"Features: {len(feature_cols)}")

    day_close = bars_30m.groupby("date")["close"].last()
    day_returns = day_close.pct_change().to_dict()

    # ══════════════════════════════════════
    #  PHASE 2: Entry predictions (cached or WF)
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: Entry model predictions")
    log.info("=" * 70)

    dates_30m = sorted(bars_30m["date"].unique())
    labels_all = bars_30m["fwd_ticks_30min"].values.astype(np.float32)
    dates_all = bars_30m["date"].values

    cached_preds_path = MFE_MAE_DIR / "entry_predictions.npz"
    entry_preds = None

    if cached_preds_path.exists():
        try:
            cached = np.load(str(cached_preds_path), allow_pickle=True)
            if len(cached["entry_preds"]) == len(bars_30m) and np.array_equal(cached["dates"], dates_all):
                entry_preds = cached["entry_preds"]
                log.info(f"Loaded cached entry predictions: {(~np.isnan(entry_preds)).sum()} valid")
        except:
            pass

    if entry_preds is None:
        log.info("Training walk-forward entry model...")
        fa = bars_30m[feature_cols].values.astype(np.float32)
        entry_preds = np.full(len(bars_30m), np.nan, dtype=np.float32)
        fi = 0
        for fs in range(TRAIN_DAYS, len(dates_30m) - 5 + 1, SLIDE_DAYS):
            ftd = dates_30m[fs - TRAIN_DAYS:fs]
            fvd = dates_30m[fs:fs + 5]
            if len(fvd) < 5:
                break
            fi += 1
            tm = np.isin(dates_all, ftd)
            vm = np.isin(dates_all, fvd)
            if not leakage_audit(list(ftd), list(fvd), feature_cols):
                continue
            tr = fa[tm].copy()
            vl = fa[vm].copy()
            med = np.nanmedian(tr, axis=0)
            iqr = np.nanpercentile(tr, 75, axis=0) - np.nanpercentile(tr, 25, axis=0)
            iqr[iqr < 1e-8] = 1.0
            tr = np.clip(np.nan_to_num((tr - med) / iqr, nan=0), -5, 5)
            vl = np.clip(np.nan_to_num((vl - med) / iqr, nan=0), -5, 5)
            _, _, vp = train_entry_model(tr, labels_all[tm], vl, labels_all[vm], feature_cols, fi)
            entry_preds[vm] = vp
        log.info(f"Entry model: {fi} folds, {(~np.isnan(entry_preds)).sum()} valid")

    # ══════════════════════════════════════
    #  PHASE 3: Entry fills
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: Reconstructing entry fills")
    log.info("=" * 70)

    all_trades, fill_stats = reconstruct_entry_fills(
        bars_30m, minute_df, entry_preds,
        confidence_pct=BASE_CONFIG["entry_threshold"],
        cancel_window_min=BASE_CONFIG["cancel_window"])
    log.info(f"Total entry fills: {len(all_trades)}")

    if len(all_trades) < 30:
        log.error("Too few trades for analysis")
        return

    # ══════════════════════════════════════
    #  PHASE 4: Tight-SL base reference
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 4: Tight-SL base reference (SL_L=4, SL_S=3)")
    log.info("=" * 70)

    tight_results = simulate_base_strategy(
        all_trades,
        tp_long=BASE_CONFIG["tp_long"], tp_short=BASE_CONFIG["tp_short"],
        sl_long=TIGHT_SL_LONG, sl_short=TIGHT_SL_SHORT,
        max_hold_minutes=BASE_CONFIG["max_hold"])

    tight_pnls = np.array([t["trade_pnl"] for t in tight_results])
    tight_dates = np.array([t["date"] for t in tight_results])
    tight_dirs = np.array([t["direction"] for t in tight_results])

    tight_stats = compute_daily_stats(tight_pnls, tight_dates, tight_dirs,
                                       "TIGHT_SL_BASE")
    tight_regime = full_regime_analysis(tight_pnls, tight_dates, tight_dirs, day_returns)
    tight_stats["regime_analysis"] = tight_regime

    # SL timing analysis
    sl_trades = [t for t in tight_results if t["exit_type"] == "sl"]
    sl_mins = [t["exit_minute"] for t in sl_trades]
    log.info(f"Tight SL base: Sharpe={tight_stats['daily_sharpe']:.2f}, "
             f"Sortino={tight_stats['daily_sortino']:.2f}, "
             f"WR={tight_stats['win_rate']:.1%}, PF={tight_stats['profit_factor']:.2f}")
    if sl_mins:
        log.info(f"  SL timing: median={np.median(sl_mins):.0f}min, mean={np.mean(sl_mins):.0f}min, "
                 f"<3min: {sum(1 for m in sl_mins if m < 3)}/{len(sl_mins)}")
    rg = tight_regime.get("regimes", {})
    if "regime_gap_detail" in rg:
        log.info(f"  Regime: {rg['regime_gap_detail']}")

    # ══════════════════════════════════════
    #  PHASE 5: Wide SL unmanaged baselines
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 5: Wide SL unmanaged baselines")
    log.info("=" * 70)

    unmanaged_results = {}
    for sl_width in SL_SWEEP:
        sl_l = sl_width
        sl_s = sl_width
        results = simulate_base_strategy(
            all_trades,
            tp_long=BASE_CONFIG["tp_long"], tp_short=BASE_CONFIG["tp_short"],
            sl_long=sl_l, sl_short=sl_s,
            max_hold_minutes=BASE_CONFIG["max_hold"])

        pnls = np.array([t["trade_pnl"] for t in results])
        dates_arr = np.array([t["date"] for t in results])
        dirs_arr = np.array([t["direction"] for t in results])

        label = f"UNMANAGED_SL{sl_width}"
        stats = compute_daily_stats(pnls, dates_arr, dirs_arr, label)
        regime = full_regime_analysis(pnls, dates_arr, dirs_arr, day_returns)
        stats["regime_analysis"] = regime
        stats["sl_width"] = sl_width

        # SL timing
        sl_trades_w = [t for t in results if t["exit_type"] == "sl"]
        sl_mins_w = [t["exit_minute"] for t in sl_trades_w]

        log.info(f"  SL={sl_width}: Sharpe={stats['daily_sharpe']:.2f}, "
                 f"Sortino={stats['daily_sortino']:.2f}, "
                 f"WR={stats['win_rate']:.1%}, PF={stats['profit_factor']:.2f}, "
                 f"n={stats['n_trades']}")
        if sl_mins_w:
            log.info(f"    SL timing: median={np.median(sl_mins_w):.0f}min, "
                     f"SL hits: {len(sl_mins_w)}/{len(results)}")
        rg = regime.get("regimes", {})
        if "regime_gap_detail" in rg:
            log.info(f"    Regime: {rg['regime_gap_detail']}")

        unmanaged_results[sl_width] = {
            "stats": stats, "regime": regime, "trade_results": results,
        }

    # ══════════════════════════════════════
    #  PHASE 6: Walk-forward management + sweep
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 6: Walk-forward management training + sweep")
    log.info(f"  SL: {SL_SWEEP}, Inv%: {INVALIDATION_PCTILE_SWEEP}, Trail: {TRAIL_BREAKEVEN_SWEEP}")
    log.info(f"  Total managed configs: {len(SL_SWEEP) * len(INVALIDATION_PCTILE_SWEEP) * len(TRAIL_BREAKEVEN_SWEEP)}")
    log.info("=" * 70)

    trade_dates = np.array([t["date"] for t in all_trades])
    unique_trade_dates = sorted(set(trade_dates))

    all_managed_results = []
    all_classifier_aucs = {}

    for sl_width in SL_SWEEP:
        sl_l = sl_width
        sl_s = sl_width
        log.info(f"\n{'─' * 60}")
        log.info(f"SL={sl_width}: Training management classifiers walk-forward")
        log.info(f"{'─' * 60}")

        fold_data = []
        fold_aucs = []
        feature_importances = []
        fold_idx = 0

        for fold_start in range(TRAIN_DAYS, len(unique_trade_dates) - SLIDE_DAYS + 1, SLIDE_DAYS):
            ftd = set(unique_trade_dates[max(0, fold_start - TRAIN_DAYS):fold_start])
            fod = set(unique_trade_dates[fold_start:fold_start + SLIDE_DAYS])
            if len(fod) < 1:
                continue
            fold_idx += 1

            train_trades = [t for t in all_trades if t["date"] in ftd]
            oot_trades = [t for t in all_trades if t["date"] in fod]
            if len(train_trades) < 10 or len(oot_trades) < 1:
                fold_data.append((fod, None, 0.5, {}))
                continue

            # Simulate WIDER SL strategy on training data
            train_results = simulate_base_strategy(
                train_trades,
                tp_long=BASE_CONFIG["tp_long"], tp_short=BASE_CONFIG["tp_short"],
                sl_long=sl_l, sl_short=sl_s,
                max_hold_minutes=BASE_CONFIG["max_hold"])

            # Build management samples from wider-SL trades
            X_tr, y_tr, w_tr = build_management_samples(
                train_results, max_hold=BASE_CONFIG["max_hold"])

            if len(X_tr) < 30:
                fold_data.append((fod, None, 0.5, {}))
                continue

            wpos = np.sum(w_tr[y_tr == 1])
            wtot = np.sum(w_tr)
            wpr = wpos / wtot if wtot > 0 else 0.5

            # Train management classifier
            params = {**LGBM_MGMT_PARAMS, "seed": 42 + fold_idx + sl_width}
            n_tr = int(len(X_tr) * 0.8)
            td = lgb.Dataset(X_tr[:n_tr], label=y_tr[:n_tr], weight=w_tr[:n_tr],
                             feature_name=MGMT_FEATURE_NAMES)
            vd = lgb.Dataset(X_tr[n_tr:], label=y_tr[n_tr:], weight=w_tr[n_tr:],
                             feature_name=MGMT_FEATURE_NAMES, reference=td)
            cb = [lgb.early_stopping(stopping_rounds=20, verbose=False),
                  lgb.log_evaluation(period=0)]
            mgmt_model = lgb.train(params, td, num_boost_round=300, valid_sets=[vd], callbacks=cb)

            imp = dict(zip(MGMT_FEATURE_NAMES, mgmt_model.feature_importance("gain")))
            feature_importances.append(imp)

            # Train pred distribution for percentile thresholds
            train_preds = mgmt_model.predict(X_tr, num_iteration=mgmt_model.best_iteration)
            inv_cutoffs = {p: float(np.percentile(train_preds, p))
                          for p in INVALIDATION_PCTILE_SWEEP}

            # OOT AUC
            oot_results = simulate_base_strategy(
                oot_trades,
                tp_long=BASE_CONFIG["tp_long"], tp_short=BASE_CONFIG["tp_short"],
                sl_long=sl_l, sl_short=sl_s,
                max_hold_minutes=BASE_CONFIG["max_hold"])
            X_oot, y_oot, w_oot = build_management_samples(
                oot_results, max_hold=BASE_CONFIG["max_hold"])

            oot_auc = 0.5
            if len(X_oot) > 5 and len(np.unique(y_oot)) > 1:
                op = mgmt_model.predict(X_oot, num_iteration=mgmt_model.best_iteration)
                try:
                    oot_auc = roc_auc_score(y_oot, op, sample_weight=w_oot)
                except:
                    oot_auc = 0.5
                fold_aucs.append(oot_auc)

            # Confirmation cutoff at p80 of train preds (for trail-to-breakeven)
            conf_cutoff_80 = float(np.percentile(train_preds, 80))

            if fold_idx <= 3 or fold_idx % 5 == 0:
                log.info(f"  Fold {fold_idx}: train_samples={len(X_tr)}, "
                         f"wpos_rate={wpr:.2f}, oot_auc={oot_auc:.3f}, "
                         f"range=[{train_preds.min():.3f},{train_preds.max():.3f}]")

            fold_data.append((fod, mgmt_model, oot_auc, inv_cutoffs, conf_cutoff_80))

        log.info(f"  Trained {fold_idx} folds for SL={sl_width}")
        if fold_aucs:
            all_classifier_aucs[sl_width] = {
                "mean": float(np.mean(fold_aucs)),
                "median": float(np.median(fold_aucs)),
                "std": float(np.std(fold_aucs)),
                "n": len(fold_aucs),
            }
            log.info(f"  Classifier AUC: mean={np.mean(fold_aucs):.3f}, "
                     f"median={np.median(fold_aucs):.3f}")

        # Sweep invalidation + trail
        for inv_p in INVALIDATION_PCTILE_SWEEP:
            for trail in TRAIL_BREAKEVEN_SWEEP:
                label = f"SL{sl_width}_inv{inv_p}_trail{int(trail)}"
                amp, amd, amdi = [], [], []
                aes_list, ams_list = [], []

                for fd_item in fold_data:
                    if len(fd_item) == 4:
                        fod, mm, auc_val, ic = fd_item
                        cc80 = 0.8
                    else:
                        fod, mm, auc_val, ic, cc80 = fd_item

                    ot = [t for t in all_trades if t["date"] in fod]
                    if not ot:
                        continue
                    inv_c = ic.get(inv_p, 0.0) if isinstance(ic, dict) else 0.0
                    conf_c = cc80

                    p, td2, di, es, ms = simulate_managed_exit(
                        ot, mm,
                        BASE_CONFIG["tp_long"], BASE_CONFIG["tp_short"],
                        sl_l, sl_s, BASE_CONFIG["max_hold"],
                        inv_c, conf_c, trail)
                    amp.extend(p)
                    amd.extend(td2)
                    amdi.extend(di)
                    aes_list.append(es)
                    ams_list.append(ms)

                if len(amp) < 20:
                    continue

                pa = np.array(amp)
                da = np.array(amd)
                dia = np.array(amdi)

                # Aggregate exit stats
                ag_e = {
                    "n_tp": sum(s["n_tp"] for s in aes_list),
                    "n_sl": sum(s["n_sl"] for s in aes_list),
                    "n_time_stop": sum(s["n_time_stop"] for s in aes_list),
                    "n_early_exit": sum(s["n_early_exit"] for s in aes_list),
                }
                tt = sum(ag_e.values())
                ag_e["tp_rate"] = ag_e["n_tp"] / max(tt, 1)
                ag_e["sl_rate"] = ag_e["n_sl"] / max(tt, 1)
                ag_e["early_exit_rate"] = ag_e["n_early_exit"] / max(tt, 1)

                # Aggregate management stats
                te = sum(s["early_exit_count"] for s in ams_list)
                tp_count = sum(s["true_positives"] for s in ams_list)
                fp_count = sum(s["false_positives"] for s in ams_list)
                tr_count = sum(s["trailed_count"] for s in ams_list)
                ag_m = {
                    "early_exit_count": te,
                    "early_exit_pct": te / max(len(pa), 1),
                    "trailed_count": tr_count,
                    "trailed_pct": tr_count / max(len(pa), 1),
                    "true_positives": tp_count,
                    "true_positive_rate": tp_count / max(te, 1),
                    "false_positives": fp_count,
                    "false_positive_rate": fp_count / max(te, 1),
                }
                sv2 = [s["avg_ticks_saved_vs_sl"] for s in ams_list if s["avg_ticks_saved_vs_sl"] != 0]
                ag_m["avg_ticks_saved"] = float(np.mean(sv2)) if sv2 else 0
                ep2 = [s["avg_early_exit_pnl"] for s in ams_list if s["avg_early_exit_pnl"] != 0]
                ag_m["avg_early_exit_pnl"] = float(np.mean(ep2)) if ep2 else 0

                result = compute_daily_stats(pa, da, dia, label,
                                              exit_stats=ag_e, mgmt_stats=ag_m)
                if "error" in result:
                    continue

                regime = full_regime_analysis(pa, da, dia, day_returns)
                result["regime_analysis"] = regime
                result["config"] = {
                    "sl_width": sl_width, "sl_long": sl_l, "sl_short": sl_s,
                    "tp_long": BASE_CONFIG["tp_long"], "tp_short": BASE_CONFIG["tp_short"],
                    "max_hold": BASE_CONFIG["max_hold"],
                    "invalidation_pctile": inv_p,
                    "trail_to_breakeven": trail,
                    "check_interval": CHECK_INTERVAL,
                }
                result["classifier_auc"] = all_classifier_aucs.get(sl_width, {})

                all_managed_results.append(result)

                log.info(f"  {label}: Sharpe={result['daily_sharpe']:.2f}, "
                         f"Sortino={result['daily_sortino']:.2f}, "
                         f"WR={result['win_rate']:.1%}, PF={result['profit_factor']:.2f}, "
                         f"n={result['n_trades']}, "
                         f"early={ag_m['early_exit_pct']:.1%} "
                         f"TP_rate={ag_m['true_positive_rate']:.1%} "
                         f"FP_rate={ag_m['false_positive_rate']:.1%} "
                         f"saved={ag_m['avg_ticks_saved']:.2f}t")

        # Feature importances for this SL width
        if feature_importances:
            ai = {}
            for imp in feature_importances:
                for k, v in imp.items():
                    ai[k] = ai.get(k, 0) + v
            for k in ai:
                ai[k] /= len(feature_importances)
            tf = sorted(ai.items(), key=lambda x: x[1], reverse=True)[:5]
            log.info(f"  Top features (SL={sl_width}): " +
                     ", ".join(f"{n}={s:.0f}" for n, s in tf))

        gc.collect()

    # ══════════════════════════════════════
    #  PHASE 7: LEADERBOARD
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 7: LEADERBOARD")
    log.info("=" * 70)

    # Sort all managed by Sharpe
    all_managed_results.sort(key=lambda x: x.get("daily_sharpe", -999), reverse=True)
    passing = [r for r in all_managed_results
               if r.get("regime_analysis", {}).get("regimes", {}).get("regime_gap_pass", False)]

    log.info(f"\nTight-SL base (reference): Sharpe={tight_stats['daily_sharpe']:.2f}")
    log.info(f"Total managed configs: {len(all_managed_results)}, Regime-passing: {len(passing)}")

    # Unmanaged comparison table
    log.info("\n--- UNMANAGED WIDE SL vs TIGHT BASE ---")
    log.info(f"  {'SL':>4} {'Sharpe':>8} {'Sort':>8} {'WR':>6} {'PF':>6} {'N':>5} {'Gap':>5} {'Pass':>5}")
    log.info(f"  {'4/3':>4} {tight_stats['daily_sharpe']:8.2f} {tight_stats['daily_sortino']:8.2f} "
             f"{tight_stats['win_rate']:6.1%} {tight_stats['profit_factor']:6.2f} "
             f"{tight_stats['n_trades']:5d} "
             f"{tight_regime.get('regimes',{}).get('regime_gap',0):5.2f} "
             f"{'PASS' if tight_regime.get('regimes',{}).get('regime_gap_pass',False) else 'FAIL':>5}")
    for sl_w in SL_SWEEP:
        us = unmanaged_results[sl_w]["stats"]
        ur = unmanaged_results[sl_w]["regime"]
        rg = ur.get("regimes", {})
        log.info(f"  {sl_w:4d} {us['daily_sharpe']:8.2f} {us['daily_sortino']:8.2f} "
                 f"{us['win_rate']:6.1%} {us['profit_factor']:6.2f} "
                 f"{us['n_trades']:5d} "
                 f"{rg.get('regime_gap',0):5.2f} "
                 f"{'PASS' if rg.get('regime_gap_pass',False) else 'FAIL':>5}")

    # Top 10 managed
    log.info("\n--- TOP 10 MANAGED (by Sharpe) ---")
    for i, r in enumerate(all_managed_results[:10]):
        m = r.get("mgmt_stats", {})
        rr = r.get("regime_analysis", {}).get("regimes", {})
        gp = rr.get("regime_gap_pass", False)
        g = rr.get("regime_gap", 0)
        log.info(f"  #{i+1}: {r['label']} | Sharpe={r['daily_sharpe']:.2f} "
                 f"Sortino={r['daily_sortino']:.2f} "
                 f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} n={r['n_trades']} | "
                 f"early={m.get('early_exit_pct',0):.1%} "
                 f"TP_rate={m.get('true_positive_rate',0):.1%} "
                 f"FP_rate={m.get('false_positive_rate',0):.1%} "
                 f"saved={m.get('avg_ticks_saved',0):.2f}t | "
                 f"gap={g:.2f} {'PASS' if gp else 'FAIL'}")

    if passing:
        log.info("\n--- TOP 5 REGIME-PASSING MANAGED ---")
        for i, r in enumerate(passing[:5]):
            m = r.get("mgmt_stats", {})
            g = r.get("regime_analysis", {}).get("regimes", {}).get("regime_gap", 0)
            log.info(f"  #{i+1}: {r['label']} | Sharpe={r['daily_sharpe']:.2f} "
                     f"Sortino={r['daily_sortino']:.2f} "
                     f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} | "
                     f"early={m.get('early_exit_pct',0):.1%} "
                     f"TP_rate={m.get('true_positive_rate',0):.1%} | gap={g:.2f}")

    # ══════════════════════════════════════
    #  PHASE 8: Key comparison table
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 8: KEY COMPARISON — Per SL width")
    log.info("=" * 70)

    for sl_w in SL_SWEEP:
        us = unmanaged_results[sl_w]["stats"]
        ur = unmanaged_results[sl_w]["regime"]
        managed_for_sl = [r for r in all_managed_results
                          if r.get("config", {}).get("sl_width") == sl_w]
        best_managed = managed_for_sl[0] if managed_for_sl else None

        log.info(f"\n  SL={sl_w}:")
        log.info(f"    Unmanaged: Sharpe={us['daily_sharpe']:.2f}, "
                 f"WR={us['win_rate']:.1%}, PF={us['profit_factor']:.2f}")
        if best_managed:
            bm = best_managed
            m = bm.get("mgmt_stats", {})
            log.info(f"    Best managed: Sharpe={bm['daily_sharpe']:.2f}, "
                     f"WR={bm['win_rate']:.1%}, PF={bm['profit_factor']:.2f} "
                     f"({bm['label']})")
            log.info(f"      Early exits: {m.get('early_exit_pct',0):.1%}, "
                     f"TP_rate={m.get('true_positive_rate',0):.1%}, "
                     f"FP_rate={m.get('false_positive_rate',0):.1%}, "
                     f"saved={m.get('avg_ticks_saved',0):.2f}t")
            delta = bm['daily_sharpe'] - us['daily_sharpe']
            log.info(f"      Management improvement: {delta:+.2f} Sharpe")
        auc_info = all_classifier_aucs.get(sl_w, {})
        if auc_info:
            log.info(f"    Classifier AUC: {auc_info['mean']:.3f} (n={auc_info['n']})")

    # ══════════════════════════════════════
    #  PHASE 9: Save results
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 9: Saving results")
    log.info("=" * 70)

    # Best managed overall
    bm_overall = all_managed_results[0] if all_managed_results else None
    bm_passing = passing[0] if passing else None

    output = clean_for_json({
        "timestamp": datetime.now().isoformat(),
        "tight_sl_base": tight_stats,
        "unmanaged_baselines": {str(k): v["stats"] for k, v in unmanaged_results.items()},
        "managed_results_top20": all_managed_results[:20],
        "regime_passing_results": passing[:10],
        "classifier_aucs": all_classifier_aucs,
        "total_managed_configs": len(all_managed_results),
        "regime_passing_count": len(passing),
        "best_managed_overall": bm_overall,
        "best_regime_passing": bm_passing,
        "elapsed_seconds": time.time() - t0,
    })

    op = OUTPUT_DIR / "results.json"
    with open(op, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Saved results to {op}")

    if mlflow_active:
        mlflow.log_metric("tight_base_sharpe", tight_stats.get("daily_sharpe", 0))
        for sl_w in SL_SWEEP:
            us = unmanaged_results[sl_w]["stats"]
            mlflow.log_metric(f"unmanaged_sl{sl_w}_sharpe", us.get("daily_sharpe", 0))
        if bm_overall:
            mlflow.log_metric("best_managed_sharpe", bm_overall["daily_sharpe"])
        if bm_passing:
            mlflow.log_metric("best_passing_sharpe", bm_passing["daily_sharpe"])
        mlflow.log_metric("n_regime_passing", len(passing))
        auc_vals = [v["mean"] for v in all_classifier_aucs.values()]
        if auc_vals:
            mlflow.log_metric("avg_classifier_auc", np.mean(auc_vals))
        try:
            mlflow.log_artifact(str(op))
        except:
            pass
        mlflow.end_run()

    # ══════════════════════════════════════
    #  EXECUTIVE SUMMARY
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("EXECUTIVE SUMMARY")
    log.info("=" * 70)
    log.info(f"  Tight-SL base (SL=4/3): Sharpe={tight_stats['daily_sharpe']:.2f}")
    for sl_w in SL_SWEEP:
        us = unmanaged_results[sl_w]["stats"]
        log.info(f"  Unmanaged SL={sl_w}: Sharpe={us['daily_sharpe']:.2f}")
    log.info(f"  Total managed configs: {len(all_managed_results)}")
    log.info(f"  Regime-passing managed: {len(passing)}")
    if bm_overall:
        log.info(f"  Best managed: {bm_overall['label']}, Sharpe={bm_overall['daily_sharpe']:.2f}")
        delta_vs_tight = bm_overall['daily_sharpe'] - tight_stats['daily_sharpe']
        log.info(f"    vs tight base: {delta_vs_tight:+.2f}")
    if bm_passing:
        log.info(f"  Best regime-passing: {bm_passing['label']}, Sharpe={bm_passing['daily_sharpe']:.2f}")
    log.info(f"\n  Elapsed: {(time.time()-t0)/60:.1f} minutes")
    log.info("DONE")


if __name__ == "__main__":
    main()
