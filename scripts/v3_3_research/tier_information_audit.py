"""
HC #316 — Tiered input information audit.

Q: How much information do T2 (orderflow) and T3 (session) features carry vs
   v3.2's targets and v3.2's predictions?

This is NOT a full model ablation (which requires GPU inference with each tier
zeroed). Instead, this measures the marginal predictive power of each individual
T2/T3 feature against:
  (a) realized targets (log_ret_1s, log_ret_5s, log_ret_30s in ticks)
  (b) v3.2's own predictions (how much of v3.2's edge is "explained" by these features)

Then trains a simple Ridge regression per-tier on standardized features against
each target → upper-bound on TIER-ONLY predictive IC.

T1 (MBO event stream) is NOT included here — it requires the model to read raw
events. T1's contribution is estimated as IC_v32_full − max(T2_only, T3_only).

OUT: output/v3_2_deep_sim_20260512/tier_information_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as pads
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PREDS = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
T2_ROOT = Path("/home/jupiter/Lvl3Quant/data/derived/tier2_orderflow_features_v1.parquet")
T3_ROOT = Path("/home/jupiter/Lvl3Quant/data/derived/tier3_session_features_v1.parquet")
OUT_JSON = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/tier_information_audit.json")

OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]
STRIDE, WINDOW = 250, 1500


def safe_ic(p, t):
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() < 100:
        return float("nan")
    try:
        r, _ = spearmanr(p[m], t[m])
    except Exception:
        return float("nan")
    return float(r)


def build_pred_times():
    """For each prediction, the timestamp at last event in its window."""
    ts_parts = []
    second_idx_parts = []  # for T3 alignment: seconds since RTH open
    bucket_idx_parts = []  # for T2 alignment: 10-second buckets
    dates_parts = []
    for date in OOT_DATES:
        d = np.load(MBO_DIR / f"{date}_mbo_events.npz", allow_pickle=True)
        ts = d["timestamps"]; n_ev = len(ts)
        n_steps = max(0, (n_ev - WINDOW) // STRIDE + 1)
        sei = WINDOW - 1 + np.arange(n_steps) * STRIDE
        sei = sei[sei < n_ev]
        ts_at = ts[sei]
        ts_parts.append(ts_at)
        # seconds since midnight ET / RTH open inference
        dt = ts_at.astype("datetime64[ns]")
        secs = ((dt.astype("datetime64[s]") - dt.astype("datetime64[D]").astype("datetime64[s]"))
                .astype(np.int64))
        # T3 uses second_idx = seconds since 9:30 ET = seconds since midnight UTC - 14.5*3600
        rth_open_s = (9 * 60 + 30) * 60 + 5 * 3600  # ET 9:30 ≈ UTC 14:30 (winter)
        second_idx = secs - rth_open_s
        second_idx_parts.append(second_idx.astype(np.int32))
        # T2 uses bucket_idx = floor(second_idx / 10) (10-second buckets — assumption)
        bucket_idx = (second_idx // 10).astype(np.int32)
        bucket_idx_parts.append(bucket_idx)
        dates_parts.append(np.array([date] * len(ts_at)))
    return {
        "ts_ns": np.concatenate(ts_parts),
        "second_idx": np.concatenate(second_idx_parts),
        "bucket_idx": np.concatenate(bucket_idx_parts),
        "date": np.concatenate(dates_parts),
    }


def load_t2_for_date(date_str: str):
    iso_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    p = T2_ROOT / f"date={iso_date}"
    if not p.exists():
        return None
    df = pq.read_table(p).to_pandas()
    return df


def load_t3_for_date(date_str: str):
    iso_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    p = T3_ROOT / f"date={iso_date}"
    if not p.exists():
        return None
    df = pq.read_table(p).to_pandas()
    return df


def main():
    pred = np.load(PREDS, allow_pickle=True)
    meta = build_pred_times()
    n = int(pred["n_samples"])
    n = min(n, len(meta["ts_ns"]))
    print(f"n_predictions: {n:,}")

    # Build merged feature dataframe by date
    t2_feats_list = []
    t3_feats_list = []
    found_dates = []
    for date in OOT_DATES:
        t2 = load_t2_for_date(date)
        t3 = load_t3_for_date(date)
        if t2 is None and t3 is None:
            print(f"  {date}: T2={'no' if t2 is None else 'yes'}  T3={'no' if t3 is None else 'yes'}")
            continue
        found_dates.append(date)
        if t2 is not None: t2_feats_list.append((date, t2))
        if t3 is not None: t3_feats_list.append((date, t3))

    # T2 cols (exclude alignment keys)
    t2_cols = ["log_return_in_bucket_bps", "bucket_mfe_ticks", "bucket_mae_ticks",
               "n_trades", "trade_volume", "aggressor_buy_ratio", "signed_volume",
               "n_cancels", "n_adds", "cancel_add_ratio", "n_order_events",
               "avg_order_size", "bucket_range_ticks", "seconds_since_rth_open"]
    t3_cols = ["dist_intraday_high_ticks", "dist_intraday_low_ticks", "dist_session_vwap_ticks",
               "dist_prior_session_close_ticks", "dist_prior_session_vwap_ticks",
               "dist_intraday_vpoc_ticks", "dist_intraday_vah_ticks", "dist_intraday_val_ticks",
               "position_in_value_area", "volume_at_current_price_pctile",
               "dist_prior_session_high_ticks", "dist_prior_session_low_ticks",
               "dist_prior_session_vpoc_ticks", "dist_5d_extreme_ticks",
               "log_return_60s_bps", "log_return_5min_bps", "log_return_15min_bps",
               "realized_vol_5min_ticks", "trend_strength_5min",
               "tod_sin", "tod_cos", "is_lunch_lull", "is_close_hour", "dow_sin", "dow_cos"]

    # Align each prediction to T2 row by date + bucket_idx, T3 row by date + second_idx
    T2_arr = np.full((n, len(t2_cols)), np.nan, dtype=np.float32)
    T3_arr = np.full((n, len(t3_cols)), np.nan, dtype=np.float32)
    for date, df_t2 in t2_feats_list:
        mask_d = meta["date"][:n] == date
        if not mask_d.any(): continue
        df_idx = df_t2.set_index("bucket_idx")
        for c_i, col in enumerate(t2_cols):
            if col not in df_idx.columns: continue
            col_map = df_idx[col]
            bks = meta["bucket_idx"][:n][mask_d]
            vals = col_map.reindex(bks).values
            T2_arr[mask_d, c_i] = vals.astype(np.float32)
    for date, df_t3 in t3_feats_list:
        mask_d = meta["date"][:n] == date
        if not mask_d.any(): continue
        df_idx = df_t3.set_index("second_idx")
        for c_i, col in enumerate(t3_cols):
            if col not in df_idx.columns: continue
            col_map = df_idx[col]
            sids = meta["second_idx"][:n][mask_d]
            vals = col_map.reindex(sids).values
            T3_arr[mask_d, c_i] = vals.astype(np.float32)

    print(f"T2 fill rate: {(~np.isnan(T2_arr)).mean()*100:.1f}%")
    print(f"T3 fill rate: {(~np.isnan(T3_arr)).mean()*100:.1f}%")

    # Targets and v3.2 prediction
    TARGETS = {
        "log_ret_1s":  pred["target_log_ret_1s"][:n],
        "log_ret_5s":  pred["target_log_ret_5s"][:n],
        "log_ret_30s": pred["target_log_ret_30s"][:n],
    }
    V32_PREDS = {
        "log_ret_1s":  pred["pred_log_ret_1s"][:n],
        "log_ret_5s":  pred["pred_log_ret_5s"][:n],
        "log_ret_30s": pred["pred_log_ret_30s"][:n],
    }

    findings = {"setup": {"n": int(n), "oot_dates": OOT_DATES, "stride_window": [STRIDE, WINDOW]},
                "t2_cols": t2_cols, "t3_cols": t3_cols,
                "feature_ic_vs_target": {}, "feature_ic_vs_v32_pred": {},
                "tier_ridge_baseline": {}, "tier_share_of_v32": {}}

    # === Per-feature IC vs targets and v3.2 preds ===
    for h, target in TARGETS.items():
        feat_ic_t = {}
        feat_ic_p = {}
        for c_i, col in enumerate(t2_cols):
            feat_ic_t[f"T2:{col}"] = safe_ic(T2_arr[:, c_i], target)
            feat_ic_p[f"T2:{col}"] = safe_ic(T2_arr[:, c_i], V32_PREDS[h])
        for c_i, col in enumerate(t3_cols):
            feat_ic_t[f"T3:{col}"] = safe_ic(T3_arr[:, c_i], target)
            feat_ic_p[f"T3:{col}"] = safe_ic(T3_arr[:, c_i], V32_PREDS[h])
        findings["feature_ic_vs_target"][h] = feat_ic_t
        findings["feature_ic_vs_v32_pred"][h] = feat_ic_p

    # === Ridge baseline per tier ===
    # For each horizon, fit Ridge on T2-only, T3-only, T2+T3, predict target.
    # 80/20 train/test split (time-ordered).
    n_train = int(n * 0.8)
    for h, target in TARGETS.items():
        target_arr = target.astype(np.float64)
        valid = np.isfinite(target_arr)
        results = {}
        for tier_name, X in [("T2_only", T2_arr), ("T3_only", T3_arr),
                              ("T2+T3", np.concatenate([T2_arr, T3_arr], axis=1))]:
            X = X.astype(np.float64)
            # Impute nans with column medians (per fitted train)
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = target_arr[:n_train], target_arr[n_train:]
            tr_v = valid[:n_train] & np.isfinite(X_train).all(axis=1)
            te_v = valid[n_train:] & np.isfinite(X_test).all(axis=1)
            if tr_v.sum() < 1000 or te_v.sum() < 500:
                # Fall back to median impute
                col_med = np.nanmedian(X_train, axis=0)
                col_med = np.where(np.isfinite(col_med), col_med, 0.0)
                X_train_imp = np.where(np.isfinite(X_train), X_train, col_med)
                X_test_imp = np.where(np.isfinite(X_test), X_test, col_med)
                tr_v = valid[:n_train]; te_v = valid[n_train:]
                X_train, X_test = X_train_imp, X_test_imp
            try:
                sc = StandardScaler()
                Xs_tr = sc.fit_transform(X_train[tr_v])
                Xs_te = sc.transform(X_test[te_v])
                m = Ridge(alpha=1.0)
                m.fit(Xs_tr, y_train[tr_v])
                pred_te = m.predict(Xs_te)
                ic = safe_ic(pred_te, y_test[te_v])
                results[tier_name] = {"n_test": int(te_v.sum()), "IC": ic}
            except Exception as e:
                results[tier_name] = {"error": str(e)}
        findings["tier_ridge_baseline"][h] = results

    # === Tier share of v32 ===
    # IC(tier-only Ridge on TARGET) / IC(v3.2 on TARGET) per horizon
    for h in TARGETS:
        v32_ic = safe_ic(V32_PREDS[h], TARGETS[h])
        shares = {"v32_full_IC": v32_ic}
        for tier_name in ["T2_only", "T3_only", "T2+T3"]:
            tier_ic = findings["tier_ridge_baseline"][h].get(tier_name, {}).get("IC", float("nan"))
            if v32_ic and np.isfinite(v32_ic) and v32_ic != 0:
                shares[f"{tier_name}_share_pct"] = float(tier_ic / v32_ic * 100.0) if np.isfinite(tier_ic) else None
                shares[f"{tier_name}_IC"] = tier_ic
        findings["tier_share_of_v32"][h] = shares

    with open(OUT_JSON, "w") as f:
        json.dump(findings, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)

    print("\n=== TIER-ONLY RIDGE → IC vs TARGET (test split) ===")
    print(f"{'horizon':<12} {'v32_full':>10} {'T2_only':>10} {'T3_only':>10} {'T2+T3':>10} {'T2%':>6} {'T3%':>6} {'T1+CNN_share':>15}")
    for h in TARGETS:
        sh = findings["tier_share_of_v32"][h]
        v32 = sh.get("v32_full_IC", 0)
        t2 = sh.get("T2_only_IC", 0)
        t3 = sh.get("T3_only_IC", 0)
        t23 = sh.get("T2+T3_IC", 0)
        t2_pct = sh.get("T2_only_share_pct", 0) or 0
        t3_pct = sh.get("T3_only_share_pct", 0) or 0
        t23_pct = sh.get("T2+T3_share_pct", 0) or 0
        t1_share = max(0.0, 100.0 - t23_pct)
        print(f"{h:<12} {v32:>10.4f} {t2:>10.4f} {t3:>10.4f} {t23:>10.4f} {t2_pct:>5.1f}% {t3_pct:>5.1f}% {t1_share:>14.1f}%")

    print("\n=== TOP 8 T2+T3 features by |IC| vs log_ret_5s ===")
    feat_ic = findings["feature_ic_vs_target"]["log_ret_5s"]
    ranked = sorted(feat_ic.items(), key=lambda kv: -abs(kv[1] or 0))
    for k, v in ranked[:8]:
        print(f"  {k:<45} {v:>8.4f}")

    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
