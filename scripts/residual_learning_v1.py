#!/usr/bin/env python3
"""
residual_learning_v1.py
=======================
HC #488 creativity-mandate axis #4: post-hoc residual correction of v3.4.2.

Goal
----
Train a fast LGBM per (horizon, side) to predict the RESIDUAL of v3.4.2
OOT preds:
    residual_i = target_log_ret_h_i - pred_log_ret_h_i        (in TICKS, HC #486)
If residuals are predictable from causal features the v3.4.2 model didn't
exploit, then a small LGBM can correct the broken predictions:
    corrected_i = pred_i + predicted_residual_i

If residuals are pure noise -> confirms v3.4.2 has no extractable alpha.

Split: 16 CAL days / 16 TEST days (same as conformal_wrapper_v1).

Feature set (causal, at prediction time)
----------------------------------------
From v3.4.2 npz (head outputs available BEFORE the trade):
    pred_log_ret_{1s,5s,10s,30s}
    pred_log_ret_60s, pred_log_ret_5min
    pred_p_up_{5s,10s,30s,60s}
    pred_log_ret_{10s,30s,60s}_q{10,50,90}          (quantile heads)
    pred_pred_mfe_{30,60}s_ticks, pred_pred_mae_{30,60}s_ticks
    pred_pred_time_to_mfe_secs
    pred_p_reversal_{15,30,60}s
    pred_pred_realized_vol_30s_ticks
    pred_fifo_tp4sl3_net, pred_fifo_tp8sl5_net
    pred_fifo_tp4sl3_hit_tp, pred_fifo_tp8sl5_hit_tp
    abs(pred_h), sign(pred_h)
    quantile spreads (q90 - q10)
Pred-stream features (rolling on pred_log_ret_5s timeline, K=20):
    sign_consistency_K20  -- |mean(sign(pred_5s[i-20..i-1]))|
    drift_K20             -- mean(pred_5s[i-20..i-1])
    flip_rate_K20         -- count of sign changes / K
From OFI features at strided MBO index (i*250):
    ofi_aggressive_{1s,5s,10s,30s}
    ofi_book_{1s,5s,10s,30s}
    trade_signed_flow_{1s,5s,10s,30s}
    spread_ticks_now
From MBO timestamps:
    hour_et (one-hot dropped; we use int)
    minute_of_day_et
    event_rate_5s = MBO event count in 5s before pred_ts (causal)
    trade_rate_5s = trade-event (T) count in 5s before pred_ts (causal)

LGBM hyperparams (fast, small, per task constraints)
----------------------------------------------------
    n_estimators=100, num_leaves=200, learning_rate=0.05
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5
    min_child_samples=200, reg_lambda=1.0
    objective='regression_l2'

Evaluation
----------
1. residual_model_metrics.csv per (horizon, side):
     R^2 (OOS), MAE (OOS), n_cal, n_test, mean|residual_cal|, mean|residual_test|.
2. corrected_preds_summary.csv:
     pooled across all TEST events. For each (horizon, side, conf in {top1pc,top5pc,top10pc}):
       n_trades, net_ticks/trade, sharpe, wr, pf, profitable_days, day_conc, regime gap.
     Confidence is taken on ABS(corrected_pred) within (side, horizon, day).
3. corrected_stratified_summary.csv:
     Same TOD x velocity x horizon x side x conf cells as
     tod_velocity_stratification_v1, but ranked by |corrected_pred|.
     Apply HC #428 deploy gates.
4. feature_importance.csv:
     LGBM gain importance per (horizon, side), top 30 features.

Cost: 0.376 ticks/trade passive limit (HC ES cost table).
Runtime budget: <=30 min wall.

Outputs (output/residual_learning_v1/):
    residual_model_metrics.csv
    corrected_preds_summary.csv
    corrected_stratified_summary.csv
    feature_importance.csv
    REPORT.md
    .regen_complete.json
"""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception as e:
    print(f"LightGBM import failed: {e}", file=sys.stderr)
    sys.exit(1)

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
OOT_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
OFI_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_ofi_features")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/residual_learning_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ES_RT_COMMISSION_TICKS = 0.376       # passive limit round-trip
STRIDE = 250                          # MBO -> pred subsampling stride
SKIP_DATES = {"20260308", "20260315"} # per task spec
N_CAL_DAYS = 16
N_TEST_DAYS = 16

HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES    = ["long", "short"]
CONFIDENCES = [("top1pc", 0.01), ("top5pc", 0.05), ("top10pc", 0.10)]

TOD_BUCKETS = [
    ("open",     9*60+30, 10*60+30),
    ("mid_am",   10*60+30, 12*60),
    ("midday",   12*60,    14*60),
    ("late_pm",  14*60,    15*60),
    ("close",    15*60,    16*60),
]

# HC #428 deploy gates
GATE_NET      = 0.10
GATE_SHARPE   = 0.3
GATE_PF       = 1.1
GATE_PROFDAYS = 0.60
GATE_IMBAL    = 0.50
MIN_TRADES    = 50

# Event-type ids (from mbo_events: action_encoding T=3)
EVENT_TYPE_TRADE = 3
VEL_WIN_NS = 5_000_000_000  # 5s

# LGBM params (fast, small)
LGB_PARAMS = dict(
    objective="regression_l2",
    n_estimators=100,
    num_leaves=200,
    learning_rate=0.05,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    min_child_samples=200,
    reg_lambda=1.0,
    verbosity=-1,
    n_jobs=-1,
    deterministic=True,
    seed=42,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def tod_label(min_of_day):
    out = np.full(min_of_day.shape, "outside", dtype=object)
    for name, lo, hi in TOD_BUCKETS:
        m = (min_of_day >= lo) & (min_of_day < hi)
        out[m] = name
    return out


def rolling_pred_features(pred_5s, K=20):
    """Causal rolling features on the pred-stream. All features use indices i-K..i-1."""
    n = len(pred_5s)
    sign = np.sign(pred_5s).astype(np.float32)
    # signed cumsum -> rolling mean of sign
    cs_sign = np.cumsum(sign)
    cs_val  = np.cumsum(pred_5s.astype(np.float64))
    # diff to detect sign changes
    flips = (sign[1:] != sign[:-1]).astype(np.int32)
    cs_flips = np.zeros(n, dtype=np.int32)
    cs_flips[1:] = np.cumsum(flips)
    sign_consist = np.zeros(n, dtype=np.float32)
    drift        = np.zeros(n, dtype=np.float32)
    flip_rate    = np.zeros(n, dtype=np.float32)
    if n > 1:
        for i in range(n):
            j = max(0, i - K)
            lo_sum_sign = cs_sign[i-1] - (cs_sign[j-1] if j > 0 else 0.0) if i >= 1 else 0.0
            lo_sum_val  = cs_val[i-1]  - (cs_val[j-1]  if j > 0 else 0.0) if i >= 1 else 0.0
            cnt = i - j
            if cnt > 0:
                sign_consist[i] = abs(lo_sum_sign) / cnt
                drift[i]        = lo_sum_val / cnt
            # flips in window
            if i >= 1:
                flips_in_win = cs_flips[i-1] - (cs_flips[j-1] if j > 0 else 0)
                if cnt > 0:
                    flip_rate[i] = flips_in_win / max(cnt, 1)
    return sign_consist, drift, flip_rate


def event_rates(pred_ts_ns, mbo_ts_ns, mbo_type):
    """Causal counts in 5s before each pred_ts. Returns (event_rate_5s, trade_rate_5s)."""
    lo = pred_ts_ns - VEL_WIN_NS
    left_all  = np.searchsorted(mbo_ts_ns, lo,         side="left")
    right_all = np.searchsorted(mbo_ts_ns, pred_ts_ns, side="left")
    event_rate = (right_all - left_all).astype(np.int32)
    trade_ts = mbo_ts_ns[mbo_type == EVENT_TYPE_TRADE]
    left_t  = np.searchsorted(trade_ts, lo,         side="left")
    right_t = np.searchsorted(trade_ts, pred_ts_ns, side="left")
    trade_rate = (right_t - left_t).astype(np.int32)
    return event_rate, trade_rate


def day_decile(values):
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    dec = (ranks * 10) // n
    dec[dec == 10] = 9
    return dec.astype(np.int8)


# -----------------------------------------------------------------------------
# Per-day loader
# -----------------------------------------------------------------------------
def load_day(date):
    pred_path = OOT_DIR / f"oot_{date}.npz"
    mbo_path  = MBO_DIR / f"{date}_mbo_events.npz"
    ofi_path  = OFI_DIR / f"{date}_ofi.npz"
    if not pred_path.exists() or not mbo_path.exists():
        return None
    pred = np.load(pred_path)
    mbo  = np.load(mbo_path)
    has_ofi = ofi_path.exists()
    ofi  = np.load(ofi_path) if has_ofi else None

    n_pred = pred["pred_log_ret_1s"].shape[0]
    mbo_ts = mbo["timestamps"]
    mbo_ev = mbo["events"][:, 1].astype(np.int8)
    pred_ts = mbo_ts[::STRIDE][:n_pred]
    if len(pred_ts) != n_pred:
        return None

    # Build feature dict
    feats = {}

    # v3.4.2 prediction heads
    head_keys_signed = [
        "pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_10s", "pred_log_ret_30s",
        "pred_log_ret_60s", "pred_log_ret_5min",
        "pred_p_up_5s", "pred_p_up_10s", "pred_p_up_30s", "pred_p_up_60s",
        "pred_log_ret_10s_q10", "pred_log_ret_10s_q50", "pred_log_ret_10s_q90",
        "pred_log_ret_30s_q10", "pred_log_ret_30s_q50", "pred_log_ret_30s_q90",
        "pred_log_ret_60s_q10", "pred_log_ret_60s_q50", "pred_log_ret_60s_q90",
        "pred_pred_mfe_30s_ticks", "pred_pred_mae_30s_ticks",
        "pred_pred_mfe_60s_ticks", "pred_pred_mae_60s_ticks",
        "pred_pred_time_to_mfe_secs",
        "pred_p_reversal_15s", "pred_p_reversal_30s", "pred_p_reversal_60s",
        "pred_pred_realized_vol_30s_ticks",
        "pred_fifo_tp4sl3_net", "pred_fifo_tp8sl5_net",
        "pred_fifo_tp4sl3_hit_tp", "pred_fifo_tp8sl5_hit_tp",
    ]
    for k in head_keys_signed:
        if k in pred.files:
            feats[k] = pred[k].astype(np.float32)

    # |pred| and sign(pred) per horizon (give LGBM symmetric info)
    for h in HORIZONS:
        p = pred[f"pred_log_ret_{h}"].astype(np.float32)
        feats[f"abs_pred_{h}"]  = np.abs(p)
        feats[f"sign_pred_{h}"] = np.sign(p).astype(np.float32)

    # quantile spreads
    for h in ["10s", "30s", "60s"]:
        q10 = f"pred_log_ret_{h}_q10"
        q90 = f"pred_log_ret_{h}_q90"
        if q10 in pred.files and q90 in pred.files:
            feats[f"qspread_{h}"] = (pred[q90] - pred[q10]).astype(np.float32)

    # Pred-stream rolling features (on pred_5s)
    sc, dr, fr = rolling_pred_features(pred["pred_log_ret_5s"].astype(np.float32), K=20)
    feats["sign_consist_K20"] = sc
    feats["drift_K20"]        = dr
    feats["flip_rate_K20"]    = fr

    # OFI features at strided index
    if has_ofi:
        idx = np.arange(n_pred, dtype=np.int64) * STRIDE
        idx = np.clip(idx, 0, ofi["ofi_aggressive_1s"].shape[0] - 1)
        for k in ["ofi_aggressive_1s","ofi_aggressive_5s","ofi_aggressive_10s","ofi_aggressive_30s",
                  "ofi_book_1s","ofi_book_5s","ofi_book_10s","ofi_book_30s",
                  "trade_signed_flow_1s","trade_signed_flow_5s","trade_signed_flow_10s","trade_signed_flow_30s",
                  "spread_ticks_now"]:
            if k in ofi.files:
                feats[k] = ofi[k][idx].astype(np.float32)

    # MBO timing features
    ts_et = pd.to_datetime(pred_ts, unit="ns", utc=True).tz_convert("America/New_York")
    minute_of_day = (ts_et.hour * 60 + ts_et.minute).to_numpy().astype(np.int32)
    feats["hour_et"]       = ts_et.hour.to_numpy().astype(np.int32)
    feats["min_of_day_et"] = minute_of_day

    er, tr = event_rates(pred_ts, mbo_ts, mbo_ev)
    feats["event_rate_5s"] = er
    feats["trade_rate_5s"] = tr

    # Targets + masks (per horizon)
    targets, masks = {}, {}
    for h in HORIZONS:
        targets[h] = pred[f"target_log_ret_{h}"].astype(np.float32)
        masks[h]   = pred[f"mask_log_ret_{h}"].astype(np.float32)

    # Build DataFrame
    df = pd.DataFrame(feats)
    df.insert(0, "date", date)
    df.insert(1, "ts_ns", pred_ts)
    df.insert(2, "min_et", minute_of_day)
    df["tod"] = tod_label(df["min_et"].values)
    # Add targets / masks
    for h in HORIZONS:
        df[f"target_{h}"] = targets[h]
        df[f"mask_{h}"]   = masks[h]
    return df


# -----------------------------------------------------------------------------
# Cell metrics + gates
# -----------------------------------------------------------------------------
def cell_metrics(net_ticks, day_ids):
    n = len(net_ticks)
    if n == 0:
        return None
    mean = float(np.mean(net_ticks))
    sd   = float(np.std(net_ticks, ddof=1)) if n > 1 else 0.0
    sharpe = (mean / sd) * np.sqrt(n) if sd > 0 else 0.0
    wins = net_ticks > 0
    wr = float(np.mean(wins))
    gp = float(np.sum(net_ticks[wins])) if wins.any() else 0.0
    gl = float(-np.sum(net_ticks[~wins])) if (~wins).any() else 0.0
    pf = (gp / gl) if gl > 1e-9 else (np.inf if gp > 0 else 0.0)
    days = np.unique(day_ids)
    day_means = np.array([np.mean(net_ticks[day_ids == d]) for d in days])
    day_sums  = np.array([np.sum(net_ticks[day_ids == d])  for d in days])
    prof_days = float(np.mean(day_means > 0)) if len(days) > 0 else 0.0
    abs_sum = float(np.sum(np.abs(day_sums)))
    day_conc = float(np.max(np.abs(day_sums)) / abs_sum) if abs_sum > 1e-9 else np.nan
    return {
        "n_trades":  n,
        "n_days":    int(len(days)),
        "net_ticks": mean,
        "sharpe":    float(sharpe),
        "wr":        wr,
        "pf":        pf,
        "prof_days": prof_days,
        "day_conc":  day_conc,
    }


def classify_regime(df, mask_col, target_col, by_date_col="date"):
    """green/red/flat by cumulative target_1s ticks per day."""
    out = {}
    for d, sub in df.groupby(by_date_col):
        v = sub[mask_col].values > 0.5
        cum = float(np.sum(sub.loc[v, target_col].values))
        if cum > 5:    out[d] = "green"
        elif cum < -5: out[d] = "red"
        else:          out[d] = "flat"
    return out


def regime_imbalance(net_ticks, day_ids, regime_map):
    if len(net_ticks) == 0:
        return np.nan, np.nan, np.nan
    reg = np.array([regime_map.get(d, "flat") for d in day_ids])
    def sh(arr):
        if len(arr) < 2: return 0.0
        s = np.std(arr, ddof=1)
        return float((np.mean(arr)/s)*np.sqrt(len(arr))) if s>0 else 0.0
    sh_g = sh(net_ticks[reg == "green"])
    sh_r = sh(net_ticks[reg == "red"])
    denom = max(abs(sh_g), abs(sh_r))
    imb = abs(sh_g - sh_r) / denom if denom > 1e-9 else np.nan
    return sh_g, sh_r, imb


def gates_failed(net, sh, pf, prof_days, imbal):
    fails = []
    if net <= GATE_NET: fails.append("net")
    if sh  <= GATE_SHARPE: fails.append("sharpe")
    if not np.isfinite(pf) or pf <= GATE_PF: fails.append("pf")
    if prof_days < GATE_PROFDAYS: fails.append("profdays")
    if not np.isfinite(imbal) or imbal >= GATE_IMBAL: fails.append("imbalance")
    return fails


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("[1/6] discovering OOT dates", flush=True)
    files = sorted(OOT_DIR.glob("oot_*.npz"))
    all_dates = [f.stem.replace("oot_", "") for f in files]
    dates = [d for d in all_dates if d not in SKIP_DATES]
    print(f"      {len(dates)} valid OOT dates", flush=True)

    print("[2/6] loading per-day data (preds + OFI + MBO timing)", flush=True)
    per_day = []
    for i, d in enumerate(dates):
        try:
            df_d = load_day(d)
            if df_d is None or df_d.empty:
                print(f"  skip {d}: empty/missing", flush=True)
                continue
            df_d = df_d[df_d["tod"] != "outside"].reset_index(drop=True)
            if df_d.empty:
                continue
            per_day.append(df_d)
            if (i + 1) % 4 == 0 or i + 1 == len(dates):
                print(f"  loaded {i+1}/{len(dates)}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"  ERROR loading {d}: {e}", flush=True)
            traceback.print_exc()

    if not per_day:
        print("ERROR: no days loaded", flush=True)
        sys.exit(1)

    df = pd.concat(per_day, ignore_index=True)
    df["date"] = df["date"].astype(str)
    dates_sorted = sorted(df["date"].unique())
    print(f"      merged: {len(df):,} rows across {len(dates_sorted)} days", flush=True)

    if len(dates_sorted) < N_CAL_DAYS + 1:
        print(f"ERROR: need at least {N_CAL_DAYS+1} days, got {len(dates_sorted)}", flush=True)
        sys.exit(1)

    # Causal split: first N_CAL_DAYS -> CAL, next N_TEST_DAYS -> TEST
    cal_dates  = set(dates_sorted[:N_CAL_DAYS])
    test_dates = set(dates_sorted[N_CAL_DAYS:N_CAL_DAYS + N_TEST_DAYS])
    print(f"      cal days: {len(cal_dates)}  test days: {len(test_dates)}", flush=True)

    is_cal  = df["date"].isin(cal_dates).values
    is_test = df["date"].isin(test_dates).values

    # ------------------------------------------------------------------
    # Feature columns
    # ------------------------------------------------------------------
    non_feat_cols = {"date", "ts_ns", "min_et", "tod"} | \
                    {f"target_{h}" for h in HORIZONS} | \
                    {f"mask_{h}" for h in HORIZONS}
    feat_cols = [c for c in df.columns if c not in non_feat_cols]
    print(f"      feature cols: {len(feat_cols)}", flush=True)

    # ------------------------------------------------------------------
    # Train residual LGBMs per (horizon, side)
    # ------------------------------------------------------------------
    print("[3/6] training LGBM residual models per (horizon, side)", flush=True)
    residual_metrics_rows = []
    feature_importance_rows = []
    # We'll store corrected_pred per (horizon) on TEST rows (one value per side-trained model).
    # corrected_pred_long, corrected_pred_short: per-row predictions of residual using the side-specific model.
    # For trades, we use the model matching the side of raw pred.
    corrected_pred_test = {}   # (h, side) -> array length test sum

    test_idx = np.where(is_test)[0]
    n_test = len(test_idx)
    print(f"      test rows: {n_test:,}", flush=True)

    # Pre-extract feature matrix once
    X_all = df[feat_cols].astype(np.float32).values
    # Replace inf with nan -> LGBM tolerates nan
    X_all[~np.isfinite(X_all)] = np.nan

    for h in HORIZONS:
        targ = df[f"target_{h}"].values.astype(np.float32)
        mask = df[f"mask_{h}"].values
        pred_h_col = f"pred_log_ret_{h}"
        raw_pred = df[pred_h_col].values.astype(np.float32)
        residual = targ - raw_pred  # in ticks

        for side in SIDES:
            side_sign = 1 if side == "long" else -1
            # Filter: side based on raw_pred sign, mask valid, finite residual
            valid = (mask > 0.5) & np.isfinite(residual) & np.isfinite(raw_pred)
            if side == "long":
                side_mask = (raw_pred > 0)
            else:
                side_mask = (raw_pred < 0)
            cal_sel  = valid & side_mask & is_cal
            test_sel = valid & side_mask & is_test
            n_cal  = int(cal_sel.sum())
            n_test_s = int(test_sel.sum())
            if n_cal < 500 or n_test_s < 100:
                print(f"  [skip] h={h} side={side} n_cal={n_cal} n_test={n_test_s}", flush=True)
                continue

            X_cal = X_all[cal_sel]
            y_cal = residual[cal_sel]
            X_te  = X_all[test_sel]
            y_te  = residual[test_sel]

            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(X_cal, y_cal)
            yhat_te = model.predict(X_te).astype(np.float32)

            # Metrics
            ss_res = float(np.sum((y_te - yhat_te) ** 2))
            ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
            r2  = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            mae = float(np.mean(np.abs(y_te - yhat_te)))
            mae_baseline = float(np.mean(np.abs(y_te)))  # vs predicting 0

            residual_metrics_rows.append({
                "horizon": h, "side": side,
                "n_cal": n_cal, "n_test": n_test_s,
                "mean_abs_resid_cal":  float(np.mean(np.abs(y_cal))),
                "mean_abs_resid_test": mae_baseline,
                "r2_test": r2,
                "mae_test": mae,
                "mae_improvement_vs_zero": mae_baseline - mae,
            })

            # Save importance
            imp = model.booster_.feature_importance(importance_type="gain")
            top = np.argsort(imp)[::-1][:30]
            for rank, j in enumerate(top, start=1):
                feature_importance_rows.append({
                    "horizon": h, "side": side, "rank": rank,
                    "feature": feat_cols[j], "gain": float(imp[j]),
                })

            corrected_pred_test[(h, side)] = (test_sel, yhat_te)
            print(f"  h={h:>3s} {side:>5s} n_cal={n_cal:>7,d} n_te={n_test_s:>7,d} "
                  f"R2={r2:+.4f} MAE={mae:.3f} (baseline_MAE={mae_baseline:.3f}, "
                  f"impr={mae_baseline - mae:+.4f})", flush=True)

    metrics_df = pd.DataFrame(residual_metrics_rows)
    metrics_df.to_csv(OUT_DIR / "residual_model_metrics.csv", index=False)
    fi_df = pd.DataFrame(feature_importance_rows)
    fi_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print(f"      wrote residual_model_metrics.csv ({len(metrics_df)} rows) and "
          f"feature_importance.csv ({len(fi_df)} rows)", flush=True)

    # ------------------------------------------------------------------
    # Build corrected predictions on full TEST set, then evaluate deploy gates
    # ------------------------------------------------------------------
    print("[4/6] evaluating CORRECTED-pred deploy gates (pooled)", flush=True)

    # Build per-horizon corrected_pred array aligned with df rows; nan elsewhere
    corrected_df = pd.DataFrame(index=df.index)
    for h in HORIZONS:
        corrected_df[f"corrected_pred_{h}"] = np.full(len(df), np.nan, dtype=np.float32)
    for (h, side), (sel, yhat) in corrected_pred_test.items():
        raw = df[f"pred_log_ret_{h}"].values[sel]
        corrected_df.loc[sel, f"corrected_pred_{h}"] = raw + yhat

    # Regime map (use full TEST window so all-test days classified)
    test_df = df[is_test].copy()
    regime_map = classify_regime(test_df, "mask_1s", "target_1s")
    n_g = sum(v == "green" for v in regime_map.values())
    n_r = sum(v == "red"   for v in regime_map.values())
    n_f = sum(v == "flat"  for v in regime_map.values())
    print(f"      test regime mix: green={n_g} red={n_r} flat={n_f}", flush=True)

    # POOLED deploy-gate scan
    pooled_rows = []
    for h in HORIZONS:
        cp = corrected_df[f"corrected_pred_{h}"].values
        mask = df[f"mask_{h}"].values
        targ = df[f"target_{h}"].values
        dates_arr = df["date"].values
        valid = is_test & (mask > 0.5) & np.isfinite(cp)
        if valid.sum() == 0:
            continue
        for side in SIDES:
            side_sign = 1 if side == "long" else -1
            if side == "long":
                side_mask = cp > 0
            else:
                side_mask = cp < 0
            sel_all = valid & side_mask
            if sel_all.sum() < MIN_TRADES:
                continue
            cp_s = cp[sel_all]
            targ_s = targ[sel_all]
            dates_s = dates_arr[sel_all]
            for conf_name, conf_frac in CONFIDENCES:
                # threshold per-day on |corrected_pred|
                keep = np.zeros(len(cp_s), dtype=bool)
                for d in np.unique(dates_s):
                    dsel = (dates_s == d)
                    if dsel.sum() < 10:
                        continue
                    thr = np.quantile(np.abs(cp_s[dsel]), 1.0 - conf_frac)
                    keep |= dsel & (np.abs(cp_s) >= thr)
                if keep.sum() < MIN_TRADES:
                    continue
                signed = side_sign * targ_s[keep] - ES_RT_COMMISSION_TICKS
                day_ids = dates_s[keep]
                m = cell_metrics(signed, day_ids)
                sh_g, sh_r, imb = regime_imbalance(signed, day_ids, regime_map)
                fails = gates_failed(m["net_ticks"], m["sharpe"], m["pf"], m["prof_days"], imb)
                pooled_rows.append({
                    "scope": "pooled", "horizon": h, "side": side, "conf": conf_name,
                    **m, "sharpe_green": sh_g, "sharpe_red": sh_r,
                    "regime_imbalance": imb,
                    "n_gates_failed": len(fails),
                    "failed_gates": ",".join(fails) if fails else "",
                    "passes_all": (len(fails) == 0),
                })

    pooled_df = pd.DataFrame(pooled_rows).sort_values(
        ["passes_all", "sharpe"], ascending=[False, False]
    )
    pooled_df.to_csv(OUT_DIR / "corrected_preds_summary.csv", index=False)
    n_pooled_wins = int(pooled_df["passes_all"].sum()) if len(pooled_df) else 0
    print(f"      pooled cells: {len(pooled_df)}   winners: {n_pooled_wins}", flush=True)

    # ------------------------------------------------------------------
    # Stratified TOD x velocity scan on CORRECTED preds
    # ------------------------------------------------------------------
    print("[5/6] stratified TOD x velocity on CORRECTED preds", flush=True)
    # Velocity per row: per-day decile of trade_rate_5s within TEST days only
    test_mask_idx = np.where(is_test)[0]
    test_dates_arr = df["date"].values[test_mask_idx]
    test_trade_rate = df["trade_rate_5s"].values[test_mask_idx]
    # day -> decile
    vel_dec = np.zeros(len(test_mask_idx), dtype=np.int8)
    for d in np.unique(test_dates_arr):
        m = (test_dates_arr == d)
        vel_dec[m] = day_decile(test_trade_rate[m])
    # Map back
    vel_dec_full = np.full(len(df), -1, dtype=np.int8)
    vel_dec_full[test_mask_idx] = vel_dec

    strat_rows = []
    for h in HORIZONS:
        cp = corrected_df[f"corrected_pred_{h}"].values
        mask = df[f"mask_{h}"].values
        targ = df[f"target_{h}"].values
        dates_arr = df["date"].values
        tod_arr = df["tod"].values
        valid = is_test & (mask > 0.5) & np.isfinite(cp) & (vel_dec_full >= 0)
        if valid.sum() == 0:
            continue
        for side in SIDES:
            side_sign = 1 if side == "long" else -1
            if side == "long":
                side_mask = cp > 0
            else:
                side_mask = cp < 0
            base = valid & side_mask
            if base.sum() < MIN_TRADES:
                continue
            for conf_name, conf_frac in CONFIDENCES:
                # threshold per-day per-side on |cp|
                cp_b = cp[base]
                targ_b = targ[base]
                tod_b = tod_arr[base]
                vd_b  = vel_dec_full[base]
                dates_b = dates_arr[base]
                keep = np.zeros(len(cp_b), dtype=bool)
                for d in np.unique(dates_b):
                    dsel = (dates_b == d)
                    if dsel.sum() < 10:
                        continue
                    thr = np.quantile(np.abs(cp_b[dsel]), 1.0 - conf_frac)
                    keep |= dsel & (np.abs(cp_b) >= thr)
                if keep.sum() == 0:
                    continue
                cp_k = cp_b[keep]; targ_k = targ_b[keep]
                tod_k = tod_b[keep]; vd_k = vd_b[keep]; dates_k = dates_b[keep]
                for tod_name, _, _ in TOD_BUCKETS:
                    tmask = (tod_k == tod_name)
                    if tmask.sum() == 0:
                        continue
                    for vd in range(10):
                        cmask = tmask & (vd_k == vd)
                        if cmask.sum() < MIN_TRADES:
                            continue
                        signed = side_sign * targ_k[cmask] - ES_RT_COMMISSION_TICKS
                        day_ids = dates_k[cmask]
                        m = cell_metrics(signed, day_ids)
                        sh_g, sh_r, imb = regime_imbalance(signed, day_ids, regime_map)
                        fails = gates_failed(m["net_ticks"], m["sharpe"], m["pf"], m["prof_days"], imb)
                        strat_rows.append({
                            "horizon": h, "side": side, "conf": conf_name,
                            "tod": tod_name, "vel_dec": int(vd),
                            **m, "sharpe_green": sh_g, "sharpe_red": sh_r,
                            "regime_imbalance": imb,
                            "n_gates_failed": len(fails),
                            "failed_gates": ",".join(fails) if fails else "",
                            "passes_all": (len(fails) == 0),
                        })
    strat_df = pd.DataFrame(strat_rows).sort_values(
        ["passes_all", "sharpe"], ascending=[False, False]
    )
    strat_df.to_csv(OUT_DIR / "corrected_stratified_summary.csv", index=False)
    n_strat_wins = int(strat_df["passes_all"].sum()) if len(strat_df) else 0
    print(f"      stratified cells: {len(strat_df)}   winners: {n_strat_wins}", flush=True)

    # Closest-miss table (stratified)
    closest = []
    if len(strat_df):
        misses = strat_df[(~strat_df["passes_all"]) & (strat_df["n_trades"] >= MIN_TRADES)] \
                 .sort_values(["n_gates_failed", "sharpe"], ascending=[True, False]).head(5)
        for _, r in misses.iterrows():
            closest.append({
                "horizon": r["horizon"], "side": r["side"], "conf": r["conf"],
                "tod": r["tod"], "vel_dec": int(r["vel_dec"]),
                "n_trades": int(r["n_trades"]),
                "net_ticks": float(r["net_ticks"]),
                "sharpe": float(r["sharpe"]),
                "pf": float(r["pf"]) if np.isfinite(r["pf"]) else None,
                "wr": float(r["wr"]),
                "prof_days": float(r["prof_days"]),
                "regime_imbalance": (None if not np.isfinite(r["regime_imbalance"])
                                     else float(r["regime_imbalance"])),
                "n_gates_failed": int(r["n_gates_failed"]),
                "failed_gates": r["failed_gates"],
            })

    # ------------------------------------------------------------------
    # REPORT.md
    # ------------------------------------------------------------------
    print("[6/6] writing REPORT.md", flush=True)
    accept = (n_strat_wins > 6)  # task asks: did correction expand beyond 6 baseline?
    # Aggregate residual predictability summary
    if len(metrics_df):
        max_r2 = float(metrics_df["r2_test"].max())
        mean_r2 = float(metrics_df["r2_test"].mean())
        # top 5 features by aggregate gain across all (h, side) models
        if len(fi_df):
            agg = fi_df.groupby("feature")["gain"].sum().sort_values(ascending=False)
            top5_feats = agg.head(5)
        else:
            top5_feats = pd.Series(dtype=float)
    else:
        max_r2 = mean_r2 = float("nan")
        top5_feats = pd.Series(dtype=float)

    lines = []
    lines.append("# Residual Learning v1 - REPORT\n\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n")
    lines.append(f"- HC: #488 creativity-mandate axis #4 (residual stacking on v3.4.2)\n")
    lines.append(f"- Split: {N_CAL_DAYS} CAL days / {N_TEST_DAYS} TEST days (chronological, causal)\n")
    lines.append(f"- Skipped dates: {sorted(SKIP_DATES)}\n")
    lines.append(f"- Total rows: {len(df):,}   CAL rows: {int(is_cal.sum()):,}   TEST rows: {int(is_test.sum()):,}\n")
    lines.append(f"- Feature count: {len(feat_cols)}\n")
    lines.append(f"- LGBM: 100 trees, 200 leaves, lr=0.05\n\n")

    lines.append(f"## VERDICT: {'ACCEPT' if accept else 'REJECT'}\n\n")
    lines.append(f"Stratified winners (corrected preds): **{n_strat_wins}**   "
                 f"vs baseline (raw-pred stratification): **6**\n")
    lines.append(f"Pooled deploy-gate winners (corrected): **{n_pooled_wins}**   "
                 f"vs baseline pooled: **0**\n\n")

    lines.append("## Residual predictability (per horizon x side, OOS R^2)\n\n")
    if len(metrics_df):
        lines.append(metrics_df[["horizon","side","n_cal","n_test","r2_test","mae_test",
                                 "mae_improvement_vs_zero"]]
                     .round(4).to_markdown(index=False))
        lines.append("\n\n")
        lines.append(f"- Best R^2: **{max_r2:+.4f}**   Mean R^2: **{mean_r2:+.4f}**\n")
        if max_r2 < 0.001:
            lines.append("- **Verdict: residuals essentially NOISE.** No model edge to rescue.\n")
        elif max_r2 < 0.01:
            lines.append("- Residuals marginally predictable (R^2 < 0.01); weak correction signal.\n")
        else:
            lines.append("- Residuals show predictable structure - correction may extract edge.\n")
    else:
        lines.append("(no models trained)\n")
    lines.append("\n")

    lines.append("## Top 5 features explaining residual (aggregate gain across models)\n\n")
    if len(top5_feats):
        for f, g in top5_feats.items():
            lines.append(f"- {f}  (sum-gain={g:,.0f})\n")
    else:
        lines.append("(no importance data)\n")
    lines.append("\n")

    lines.append("## Pooled deploy-gate results (corrected preds, top of summary)\n\n")
    if len(pooled_df):
        cols = ["horizon","side","conf","n_trades","n_days","net_ticks","sharpe","wr",
                "pf","prof_days","day_conc","regime_imbalance","n_gates_failed","passes_all"]
        lines.append(pooled_df[cols].head(10).round(4).to_markdown(index=False))
        lines.append("\n\n")
    else:
        lines.append("(no pooled rows)\n\n")

    lines.append("## Stratified winners (corrected preds, HC #428 pass)\n\n")
    if n_strat_wins > 0:
        cols = ["horizon","side","conf","tod","vel_dec","n_trades","n_days","net_ticks",
                "sharpe","wr","pf","prof_days","day_conc","regime_imbalance"]
        lines.append(strat_df[strat_df["passes_all"]][cols].head(20).round(4).to_markdown(index=False))
        lines.append("\n\n")
    else:
        lines.append("(no stratified cells clear HC #428)\n\n")

    lines.append("## Closest miss (stratified, corrected preds)\n\n")
    if closest:
        for cr in closest:
            lines.append(f"- {cr['horizon']:>3s} {cr['conf']:>7s} {cr['side']:>5s} "
                         f"{cr['tod']:>8s} vd={cr['vel_dec']}: n={cr['n_trades']} "
                         f"net={cr['net_ticks']:+.3f}t Sh={cr['sharpe']:.2f} "
                         f"fails=[{cr['failed_gates']}]\n")
    else:
        lines.append("(no closest-miss rows)\n")
    lines.append("\n")

    lines.append("## Interpretation\n\n")
    if accept:
        lines.append("- Residual learning EXPANDED the winning-cell count beyond baseline 6. "
                     "Post-hoc correction extracts edge that pooled / raw-stratified analysis missed.\n")
        lines.append("- Next: lock the corrected-pred configuration and run shadow paper-trade for 5 OOT days.\n")
    else:
        lines.append("- Residual correction does NOT expand the winning-cell count beyond baseline 6. "
                     "v3.4.2 residuals are either pure noise (R^2 ~ 0) or only weakly predictable in a way that "
                     "doesn't translate to deploy-grade cells under HC #428 gates.\n")
        if max_r2 < 0.001:
            lines.append("- The R^2 floor confirms v3.4.2 OOT residuals are noise. There is no extractable "
                         "alpha left in those predictions via post-hoc correction at this feature set.\n")
        lines.append("- This corroborates the conformal-wrapper and TOD x velocity findings: the broken side "
                     "of v3.4.2 is not rescuable by features we have access to.\n")
    lines.append("\n")

    (OUT_DIR / "REPORT.md").write_text("".join(lines))

    # HC #485 R5 regen stamp
    stamp = {
        "script": "scripts/residual_learning_v1.py",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_seconds": round(time.time() - t0, 1),
        "n_oot_days": len(dates_sorted),
        "cal_days": N_CAL_DAYS,
        "test_days": N_TEST_DAYS,
        "skipped_dates": sorted(SKIP_DATES),
        "n_features": len(feat_cols),
        "lgbm_params": LGB_PARAMS,
        "n_pooled_cells": int(len(pooled_df)),
        "n_pooled_winners": n_pooled_wins,
        "n_stratified_cells": int(len(strat_df)),
        "n_stratified_winners": n_strat_wins,
        "baseline_winners_to_beat": 6,
        "max_r2_residual": max_r2,
        "mean_r2_residual": mean_r2,
        "verdict": "ACCEPT" if accept else "REJECT",
    }
    (OUT_DIR / ".regen_complete.json").write_text(json.dumps(stamp, indent=2, default=str))

    print(f"\nDone in {time.time()-t0:.1f}s. Verdict: {'ACCEPT' if accept else 'REJECT'}. "
          f"strat_winners={n_strat_wins} (baseline=6)  pooled_winners={n_pooled_wins} (baseline=0)  "
          f"max_R^2={max_r2:+.4f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
