#!/usr/bin/env python3
"""
2h ES LGBM — Regime Debiasing Study (HC #693 + R1 Compliance)
=============================================================
Goal: Fix regime gap (currently 0.64, need <0.50) without destroying profitability.

Strategies tested:
1. SYMMETRIC FILTER — same absolute threshold for long and short
2. ASYMMETRIC THRESHOLD — require higher confidence for shorts (the model's strong side)
3. REGIME-AWARE SCALING — reduce position confidence when trend regime detected
4. DIRECTION BALANCE — cap max short fraction per day
5. LONG-ONLY — test if longs alone are profitable
6. SHORT-ONLY CAPPED — limit short trades to match long count

Also: HC #693 cost scenarios (4 entry/exit combos).

Includes permutation test for any config that passes R1.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json
import warnings
warnings.filterwarnings("ignore")

# ─── Constants ───
ROOT = Path("/home/jupiter/Lvl3Quant")
BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
OUT_DIR = ROOT / "output" / "lh_2h_regime_debias"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DAYS = 60
PURGE_DAYS = 5
HORIZON_BARS = 2
SIGNAL_HOURS_UTC = [14, 15, 16, 17, 18]

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "num_leaves": 15,
    "max_depth": 4,
    "learning_rate": 0.02,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 1.0,
    "lambda_l2": 5.0,
    "n_estimators": 500,
}

# HC #693 cost scenarios (ticks RT)
COST_SCENARIOS = {
    "passive_passive": 0.376,   # Commission only
    "passive_market":  0.876,   # Commission + half spread
    "market_passive":  0.876,   # Commission + half spread
    "market_market":   1.376,   # Commission + full spread
}

# Leaky features to exclude (from clean model analysis)
LEAKY_FEATURES = {
    "microprice_dev_mean", "microprice_dev_trend", "microprice_dev_late",
    "vwap_dev_final", "vwap_dev_trend",
    "mpdev_sum_2h", "mpdev_sum_4h", "mpdev_sum_6h",
    "vwapdev_sum_2h", "vwapdev_sum_4h", "vwapdev_sum_6h",
}


def safe_polyfit_slope(arr, deg=1):
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, deg)[0])
    except:
        return 0.0


def build_hourly_features(minute_df):
    """Aggregate minute bars to hourly bars with MBO features."""
    df = minute_df.copy()
    df["hour"] = df["ts_minute"].dt.hour
    df["date_str"] = df["ts_minute"].dt.strftime("%Y-%m-%d")
    df["return_1m"] = df.groupby("date_str")["close"].pct_change()

    records = []
    for (date_str, hour), g in df.groupby(["date_str", "hour"]):
        if len(g) < 5:
            continue
        c = g["close"].values.astype(float)
        v = g["volume"].values.astype(float)
        ofi = g["ofi_1min"].values.astype(float)
        sv = g["signed_volume"].values.astype(float)
        ret = g["return_1m"].fillna(0).values.astype(float)
        sp = g["spread_mean"].values.astype(float)
        tc = g["trade_count"].values.astype(float)
        vwap = g["vwap"].values.astype(float)
        mp = g["microprice_close"].values.astype(float)

        rec = {
            "date": date_str, "hour": hour, "ts": g["ts_minute"].iloc[0],
            "open": c[0], "high": c.max(), "low": c.min(), "close": c[-1],
            "return_1h": (c[-1] / c[0] - 1) if c[0] > 0 else 0,
            "range_ticks": (c.max() - c.min()),
            "close_position": (c[-1] - c.min()) / max(c.max() - c.min(), 1),
            "total_volume": v.sum(),
            "avg_volume": v.mean(),
            "volume_trend": safe_polyfit_slope(v),
            "volume_concentration": v.max() / max(v.mean(), 1),
            "ofi_sum": ofi.sum(),
            "ofi_mean": ofi.mean(),
            "ofi_trend": safe_polyfit_slope(ofi),
            "ofi_consistency": np.mean(np.sign(ofi) == np.sign(ofi.sum())) if ofi.sum() != 0 else 0.5,
            "ofi_late_vs_early": ofi[len(ofi)//2:].sum() - ofi[:len(ofi)//2].sum(),
            "signed_volume_sum": sv.sum(),
            "signed_volume_ratio": sv.sum() / max(v.sum(), 1),
            "buy_volume_fraction": np.sum(sv[sv > 0]) / max(v.sum(), 1),
            "sell_volume_fraction": -np.sum(sv[sv < 0]) / max(v.sum(), 1),
            "sweep_minutes": int(np.sum(np.abs(sv) > 2 * sv.std())) if sv.std() > 0 else 0,
            "spread_mean": sp.mean(),
            "spread_max": sp.max(),
            "trade_count_sum": tc.sum(),
            "trade_intensity": tc.mean(),
            "return_std": ret.std(),
            "return_skew": float(stats.skew(ret)) if len(ret) > 3 else 0,
            "realized_vol": ret.std() * np.sqrt(60),
        }

        # Microprice (kept but will be excluded from features)
        mp_valid = mp[mp > 0]
        c_valid = c[:len(mp_valid)]
        if len(mp_valid) > 0 and len(c_valid) > 0:
            mp_dev = (mp_valid - c_valid[:len(mp_valid)]) / np.maximum(c_valid[:len(mp_valid)], 1)
            rec["microprice_dev_mean"] = mp_dev.mean()
            rec["microprice_dev_trend"] = safe_polyfit_slope(mp_dev)
            rec["microprice_dev_late"] = mp_dev[-len(mp_dev)//3:].mean() if len(mp_dev) >= 3 else mp_dev.mean()
        else:
            rec["microprice_dev_mean"] = rec["microprice_dev_trend"] = rec["microprice_dev_late"] = 0

        # VWAP (kept but excluded from features)
        vwap_valid = vwap[vwap > 0]
        if len(vwap_valid) > 0:
            vwap_dev = (c[:len(vwap_valid)] - vwap_valid) / np.maximum(vwap_valid, 1)
            rec["vwap_dev_final"] = vwap_dev[-1] if len(vwap_dev) > 0 else 0
            rec["vwap_dev_trend"] = safe_polyfit_slope(vwap_dev)
        else:
            rec["vwap_dev_final"] = rec["vwap_dev_trend"] = 0

        rec["vw_return"] = np.sum(ret * v[:len(ret)]) / v[:len(ret)].sum() if v.sum() > 0 and len(ret) <= len(v) else 0

        if len(ret) > 10:
            rec["return_autocorr_1"] = np.corrcoef(ret[:-1], ret[1:])[0,1] if ret[:-1].std() > 0 and ret[1:].std() > 0 else 0
            rec["return_autocorr_5"] = np.corrcoef(ret[:-5], ret[5:])[0,1] if ret[:-5].std() > 0 and ret[5:].std() > 0 else 0
        else:
            rec["return_autocorr_1"] = rec["return_autocorr_5"] = 0

        if tc.sum() > 0 and v.sum() > 0:
            rec["avg_trade_size"] = v.sum() / tc.sum()
            v_sorted = np.sort(v)
            rec["volume_top_half_ratio"] = v_sorted[len(v_sorted)//2:].sum() / max(v.sum(), 1)
        else:
            rec["avg_trade_size"] = 0
            rec["volume_top_half_ratio"] = 0.5

        if len(ofi) > 5:
            rec["ofi_acceleration"] = ofi[len(ofi)//2:].sum() - ofi[:len(ofi)//2].sum()
            rec["ofi_curvature"] = np.polyfit(np.arange(len(ofi)), ofi, 2)[0] if len(ofi) > 3 else 0
        else:
            rec["ofi_acceleration"] = rec["ofi_curvature"] = 0

        rec["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        rec["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        rec["hl_range_position"] = (c[-1] - c.min()) / max(c.max() - c.min(), 1)

        if len(v) > 5 and v.std() > 0 and c.std() > 0:
            rec["vol_price_divergence"] = float(np.sign(c[-1] - c[0]) != np.sign(safe_polyfit_slope(v)))
        else:
            rec["vol_price_divergence"] = 0

        rec["ofi_vol"] = ofi.std() if ofi.std() > 0 else 0
        rec["ofi_vol_normalized"] = ofi.std() / max(abs(ofi.mean()), 1) if ofi.std() > 0 else 0

        records.append(rec)

    hourly = pd.DataFrame(records).sort_values("ts").reset_index(drop=True)

    for w in [2, 4, 6]:
        lbl = f"{w}h"
        hourly[f"ofi_sum_{lbl}"] = hourly["ofi_sum"].rolling(w, min_periods=1).sum()
        hourly[f"ofi_trend_{lbl}"] = hourly["ofi_trend"].rolling(w, min_periods=1).mean()
        hourly[f"sv_sum_{lbl}"] = hourly["signed_volume_sum"].rolling(w, min_periods=1).sum()
        hourly[f"volume_ma_{lbl}"] = hourly["total_volume"].rolling(w, min_periods=1).mean()
        hourly[f"volume_vs_ma_{lbl}"] = hourly["total_volume"] / hourly[f"volume_ma_{lbl}"].clip(lower=1)
        hourly[f"vol_trend_{lbl}"] = hourly["realized_vol"].rolling(w, min_periods=1).apply(
            lambda x: safe_polyfit_slope(x.values), raw=False)
        hourly[f"mpdev_sum_{lbl}"] = hourly["microprice_dev_mean"].rolling(w, min_periods=1).sum()
        hourly[f"vwapdev_sum_{lbl}"] = hourly["vwap_dev_final"].rolling(w, min_periods=1).sum()
        hourly[f"ofi_accel_{lbl}"] = hourly["ofi_acceleration"].rolling(w, min_periods=1).sum()
        hourly[f"autocorr_mean_{lbl}"] = hourly["return_autocorr_1"].rolling(w, min_periods=1).mean()

    hourly["mom_2h"] = hourly["close"].pct_change(2)
    hourly["mom_4h"] = hourly["close"].pct_change(4)
    hourly["mom_6h"] = hourly["close"].pct_change(6)

    hourly["vol_20h"] = hourly["realized_vol"].rolling(20, min_periods=5).mean()
    expanding_rank = hourly["vol_20h"].expanding(min_periods=5).rank(pct=True)
    hourly["vol_regime_f"] = pd.cut(expanding_rank, bins=[0, 1/3, 2/3, 1.0],
                                     labels=[0, 1, 2], include_lowest=True).astype(float)
    hourly["trend_8h"] = hourly["close"].pct_change(8)
    hourly["trend_20h"] = hourly["close"].pct_change(20)
    hourly["trend_regime"] = np.where(hourly["trend_20h"] > 0.005, 1,
                                       np.where(hourly["trend_20h"] < -0.005, -1, 0))

    return hourly


def get_feature_cols(df):
    exclude = {"date", "hour", "ts", "open", "high", "low", "close", "fwd_ticks", "date_str"}
    cols = [c for c in df.columns if c not in exclude and c not in LEAKY_FEATURES
            and df[c].dtype in ["float64", "float32", "int64", "int32"]]
    return cols


def run_walkforward(hourly_clean, hourly_full, feature_cols, dates, trade_filter_fn, label="baseline"):
    """Run WF with a configurable trade filter function.

    trade_filter_fn(pred, direction, row, daily_counts) -> bool
        Returns True if trade should be taken.
    """
    import lightgbm as lgb
    import gc

    all_trades = []

    for i in range(TRAIN_DAYS + PURGE_DAYS, len(dates)):
        oot_date = dates[i]
        train_end = i - PURGE_DAYS
        train_start = max(0, train_end - TRAIN_DAYS)
        train_dates = dates[train_start:train_end]

        train = hourly_clean[hourly_clean["date"].isin(train_dates)].dropna(subset=["fwd_ticks"])
        oot = hourly_clean[hourly_clean["date"] == oot_date]
        oot_signal = oot[oot["hour"].isin(SIGNAL_HOURS_UTC)]

        if len(train) < 100 or len(oot_signal) == 0:
            continue

        X_train = train[feature_cols].fillna(0).values.astype(np.float32)
        y_train = train["fwd_ticks"].values.astype(np.float32)

        split = int(len(X_train) * 0.8)
        try:
            model = lgb.LGBMRegressor(**LGBM_PARAMS, early_stopping_rounds=50, seed=42)
            model.fit(X_train[:split], y_train[:split],
                      eval_set=[(X_train[split:], y_train[split:])],
                      callbacks=[lgb.log_evaluation(0)])
        except:
            continue

        X_oot = oot_signal[feature_cols].fillna(0).values.astype(np.float32)
        preds = model.predict(X_oot)

        daily_counts = {"LONG": 0, "SHORT": 0}

        for j, (idx, row) in enumerate(oot_signal.iterrows()):
            pred = preds[j]
            direction = "LONG" if pred > 0 else "SHORT"

            if not trade_filter_fn(pred, direction, row, daily_counts):
                continue

            daily_counts[direction] += 1
            signal = 1 if direction == "LONG" else -1
            entry_price = row["close"]
            entry_hour = int(row["hour"])
            exit_hour = entry_hour + HORIZON_BARS

            exit_bars = hourly_full[(hourly_full["date"] == oot_date) & (hourly_full["hour"] == exit_hour)]
            if exit_bars.empty:
                continue

            exit_price = exit_bars.iloc[0]["close"]
            gross_ticks = (exit_price - entry_price) * signal

            all_trades.append({
                "date": oot_date,
                "entry_hour": entry_hour,
                "direction": direction,
                "prediction": pred,
                "gross_ticks": gross_ticks,
            })

        del model
        gc.collect()

    return pd.DataFrame(all_trades)


def compute_regime_map(hourly):
    """Compute daily regime (green/red/flat) from ES close-to-close."""
    daily_close = hourly.groupby("date")["close"].last()
    daily_ret = daily_close.pct_change()
    regime_map = {}
    for d in daily_ret.index:
        r = daily_ret.loc[d]
        if pd.isna(r):
            regime_map[d] = "flat"
        elif r > 0.002:
            regime_map[d] = "green"
        elif r < -0.002:
            regime_map[d] = "red"
        else:
            regime_map[d] = "flat"
    return regime_map


def evaluate_trades(trades, regime_map, cost_ticks, label=""):
    """Evaluate trades under a given cost scenario. Returns metrics dict."""
    if trades.empty:
        return {"label": label, "n_trades": 0, "sharpe": 0, "r1_pass": False}

    t = trades.copy()
    t["net_ticks"] = t["gross_ticks"] - cost_ticks
    t["regime"] = t["date"].map(regime_map)

    daily = t.groupby("date")["net_ticks"].sum()
    n_days = len(daily)

    if n_days < 5 or daily.std() == 0:
        return {"label": label, "n_trades": len(t), "n_days": n_days, "sharpe": 0,
                "sortino": 0, "pf": 0, "wr": 0, "day_wr": 0, "max_dd_ticks": 0,
                "total_net_ticks": 0, "total_net_dollars": 0, "long_frac": 0,
                "long_net": 0, "short_net": 0, "green_sharpe": 0, "red_sharpe": 0,
                "regime_gap": 999, "r1_pass": False, "trades_per_day": 0,
                "gross_per_trade": 0, "net_per_trade": 0, "cost_ticks": cost_ticks}

    sharpe = daily.mean() / daily.std() * np.sqrt(252)
    sortino_down = daily[daily < 0].std()
    sortino = daily.mean() / sortino_down * np.sqrt(252) if sortino_down and sortino_down > 0 else float("inf")

    pf = daily[daily > 0].sum() / abs(daily[daily < 0].sum()) if daily[daily < 0].sum() != 0 else float("inf")
    wr = (t["net_ticks"] > 0).mean()
    day_wr = (daily > 0).mean()

    # Direction breakdown
    long_t = t[t["direction"] == "LONG"]
    short_t = t[t["direction"] == "SHORT"]
    long_frac = len(long_t) / len(t) if len(t) > 0 else 0

    # Regime stratification
    regime_sharpes = {}
    for regime in ["green", "red", "flat"]:
        sub = t[t["regime"] == regime]
        if len(sub) > 0:
            daily_r = sub.groupby("date")["net_ticks"].sum()
            if len(daily_r) >= 3 and daily_r.std() > 0:
                regime_sharpes[regime] = daily_r.mean() / daily_r.std() * np.sqrt(252)
            elif len(daily_r) >= 1:
                regime_sharpes[regime] = daily_r.mean()  # Not enough for std

    # R1 gate
    g_sharpe = regime_sharpes.get("green", 0)
    r_sharpe = regime_sharpes.get("red", 0)
    max_sharpe = max(abs(g_sharpe), abs(r_sharpe))
    regime_gap = abs(g_sharpe - r_sharpe) / max_sharpe if max_sharpe > 0 else 0
    r1_pass = regime_gap <= 0.50

    # Cumulative DD
    cum = daily.cumsum()
    dd = cum - cum.cummax()
    max_dd = dd.min()

    return {
        "label": label,
        "n_trades": len(t),
        "n_days": n_days,
        "trades_per_day": len(t) / n_days,
        "gross_per_trade": t["gross_ticks"].mean(),
        "net_per_trade": t["net_ticks"].mean(),
        "cost_ticks": cost_ticks,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "pf": round(pf, 2),
        "wr": round(wr * 100, 1),
        "day_wr": round(day_wr * 100, 1),
        "max_dd_ticks": round(max_dd, 1),
        "total_net_ticks": round(daily.sum(), 1),
        "total_net_dollars": round(daily.sum() * 12.50, 0),
        "long_frac": round(long_frac * 100, 1),
        "long_net": round(long_t["net_ticks"].mean(), 3) if len(long_t) > 0 else 0,
        "short_net": round(short_t["net_ticks"].mean(), 3) if len(short_t) > 0 else 0,
        "green_sharpe": round(g_sharpe, 2),
        "red_sharpe": round(r_sharpe, 2),
        "regime_gap": round(regime_gap, 3),
        "r1_pass": r1_pass,
    }


def run_permutation_test(trades, regime_map, cost_ticks, n_trials=50):
    """Shuffle trade directions, compute Sharpe of shuffled trades."""
    if trades.empty:
        return 1.0, []

    real_t = trades.copy()
    real_t["net_ticks"] = real_t["gross_ticks"] - cost_ticks
    daily = real_t.groupby("date")["net_ticks"].sum()
    if daily.std() == 0:
        return 1.0, []
    real_sharpe = daily.mean() / daily.std() * np.sqrt(252)

    beat_count = 0
    shuffled_sharpes = []

    for trial in range(n_trials):
        t_shuf = trades.copy()
        # Randomly flip direction (negate gross ticks)
        flip = np.random.choice([-1, 1], size=len(t_shuf))
        t_shuf["gross_ticks"] = t_shuf["gross_ticks"] * flip
        t_shuf["net_ticks"] = t_shuf["gross_ticks"] - cost_ticks

        daily_shuf = t_shuf.groupby("date")["net_ticks"].sum()
        if daily_shuf.std() > 0:
            s = daily_shuf.mean() / daily_shuf.std() * np.sqrt(252)
        else:
            s = 0
        shuffled_sharpes.append(s)
        if s >= real_sharpe:
            beat_count += 1

    p_value = beat_count / n_trials
    return p_value, shuffled_sharpes


def main():
    import lightgbm as lgb
    import gc
    import time

    t0 = time.time()
    print("=" * 70)
    print("2h ES LGBM — REGIME DEBIASING STUDY")
    print("HC #693 cost scenarios + R1 compliance attempt")
    print("=" * 70)

    # Load data
    files = sorted(BAR_DIR.glob("*.parquet"))
    print(f"Loading {len(files)} minute bar files...")
    dfs = [pd.read_parquet(f) for f in files]
    minute_df = pd.concat(dfs).sort_values("ts_minute").reset_index(drop=True)
    print(f"Loaded {len(minute_df):,} minute bars")

    print("Building hourly features...")
    hourly = build_hourly_features(minute_df)
    del minute_df
    gc.collect()

    # Forward labels
    hourly["fwd_ticks"] = hourly["close"].shift(-HORIZON_BARS) - hourly["close"]
    for i in range(len(hourly) - HORIZON_BARS):
        ts_now = hourly["ts"].iloc[i]
        ts_fwd = hourly["ts"].iloc[i + HORIZON_BARS]
        if (ts_fwd - ts_now).total_seconds() > 8 * 3600:
            hourly.loc[hourly.index[i], "fwd_ticks"] = np.nan

    hourly_clean = hourly[~hourly["hour"].isin([19, 20])].copy()
    dates = sorted(hourly_clean["date"].unique())
    feature_cols = get_feature_cols(hourly_clean)
    regime_map = compute_regime_map(hourly)

    print(f"Features: {len(feature_cols)} (leaky excluded)")
    print(f"OOT days: {len(dates) - TRAIN_DAYS - PURGE_DAYS}")
    print()

    # ─── Define Trade Filters ───

    # Strategy 1: Baseline (same as clean model — bottom 20% filter)
    def filter_baseline(pred, direction, row, daily_counts):
        return abs(pred) >= 16.0  # Original threshold

    # Strategy 2: Higher symmetric threshold (top 10%)
    def filter_high_confidence(pred, direction, row, daily_counts):
        return abs(pred) >= 24.0

    # Strategy 3: Asymmetric — require higher confidence for shorts
    def filter_asymmetric_short(pred, direction, row, daily_counts):
        if direction == "SHORT":
            return abs(pred) >= 28.0  # Harder to go short
        else:
            return abs(pred) >= 14.0  # Easier to go long

    # Strategy 4: Direction balance — cap shorts to match longs per day
    def filter_balanced(pred, direction, row, daily_counts):
        if abs(pred) < 16.0:
            return False
        if direction == "SHORT" and daily_counts["SHORT"] >= max(daily_counts["LONG"], 2):
            return False
        return True

    # Strategy 5: Long only
    def filter_long_only(pred, direction, row, daily_counts):
        return pred > 16.0  # Only positive predictions

    # Strategy 6: Short only (for diagnostic)
    def filter_short_only(pred, direction, row, daily_counts):
        return pred < -16.0

    # Strategy 7: Regime-aware scaling — skip shorts when strong uptrend
    def filter_regime_aware(pred, direction, row, daily_counts):
        if abs(pred) < 16.0:
            return False
        # If in strong uptrend (trend_20h > 0.01), require higher short confidence
        if direction == "SHORT" and hasattr(row, "trend_20h") and row.get("trend_20h", 0) > 0.01:
            return abs(pred) >= 30.0
        # If in strong downtrend, require higher long confidence
        if direction == "LONG" and hasattr(row, "trend_20h") and row.get("trend_20h", 0) < -0.01:
            return abs(pred) >= 30.0
        return True

    # Strategy 8: Very high confidence only (top 5%)
    def filter_ultra_confidence(pred, direction, row, daily_counts):
        return abs(pred) >= 32.0

    # Strategy 9: Moderate asymmetric (less aggressive than #3)
    def filter_mild_asymmetric(pred, direction, row, daily_counts):
        if direction == "SHORT":
            return abs(pred) >= 22.0  # Slightly harder to go short
        else:
            return abs(pred) >= 14.0

    strategies = {
        "baseline_20pct": filter_baseline,
        "high_confidence_10pct": filter_high_confidence,
        "asymmetric_short_hard": filter_asymmetric_short,
        "direction_balanced": filter_balanced,
        "long_only": filter_long_only,
        "short_only": filter_short_only,
        "regime_aware": filter_regime_aware,
        "ultra_confidence_5pct": filter_ultra_confidence,
        "mild_asymmetric": filter_mild_asymmetric,
    }

    # ─── Run All Strategies ───
    all_results = []
    strategy_trades = {}

    for name, filter_fn in strategies.items():
        print(f"\n{'─'*50}")
        print(f"Strategy: {name}")
        trades = run_walkforward(hourly_clean, hourly, feature_cols, dates, filter_fn, label=name)
        strategy_trades[name] = trades

        if trades.empty:
            print(f"  NO TRADES")
            continue

        print(f"  Trades: {len(trades)}, Days: {trades['date'].nunique()}")
        long_n = len(trades[trades["direction"] == "LONG"])
        short_n = len(trades[trades["direction"] == "SHORT"])
        print(f"  Long: {long_n} ({100*long_n/len(trades):.0f}%), Short: {short_n} ({100*short_n/len(trades):.0f}%)")

        # Evaluate under all 4 HC #693 cost scenarios
        for cost_name, cost_ticks in COST_SCENARIOS.items():
            result = evaluate_trades(trades, regime_map, cost_ticks, label=f"{name}|{cost_name}")
            all_results.append(result)

            # Print summary for primary cost scenario
            if cost_name == "passive_market":  # Most realistic default
                r1_str = "✅ PASS" if result["r1_pass"] else f"❌ FAIL (gap={result['regime_gap']:.3f})"
                print(f"  [{cost_name}] Sharpe={result['sharpe']:.2f}, PF={result['pf']:.2f}, "
                      f"WR={result['wr']:.0f}%, R1={r1_str}")
                print(f"    Green Sharpe={result['green_sharpe']:.2f}, Red Sharpe={result['red_sharpe']:.2f}")

    # ─── Summary Table ───
    print(f"\n{'='*90}")
    print("SUMMARY — All Strategies × passive_market cost (0.876 ticks)")
    print(f"{'='*90}")
    print(f"{'Strategy':<28} {'N':>5} {'Sharpe':>7} {'PF':>5} {'WR%':>5} {'L%':>4} {'G_Sh':>6} {'R_Sh':>6} {'Gap':>6} {'R1':>4}")
    print("-" * 90)

    for r in all_results:
        if "|passive_market" in r["label"]:
            name = r["label"].split("|")[0]
            r1 = "✅" if r["r1_pass"] else "❌"
            print(f"{name:<28} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['pf']:>5.2f} "
                  f"{r['wr']:>5.1f} {r['long_frac']:>4.0f} {r['green_sharpe']:>6.2f} "
                  f"{r['red_sharpe']:>6.2f} {r['regime_gap']:>6.3f} {r1:>4}")

    # ─── Find R1-passing configs ───
    passing = [r for r in all_results if r.get("r1_pass") and r.get("sharpe", 0) > 0.5]

    if passing:
        print(f"\n{'='*70}")
        print(f"R1-PASSING CONFIGS FOUND: {len(passing)}")
        print(f"{'='*70}")

        # Run permutation test on each
        for r in sorted(passing, key=lambda x: -x["sharpe"]):
            strategy_name = r["label"].split("|")[0]
            cost_name = r["label"].split("|")[1]
            cost_ticks = COST_SCENARIOS[cost_name]
            trades = strategy_trades[strategy_name]

            print(f"\nPermutation test: {r['label']} (Sharpe={r['sharpe']:.2f})")
            p_val, shuffled = run_permutation_test(trades, regime_map, cost_ticks, n_trials=50)
            r["permutation_p"] = p_val
            r["permutation_n"] = 50

            p_str = f"p={p_val:.2f}"
            if p_val <= 0.05:
                print(f"  ✅ PASSES permutation test ({p_str})")
            else:
                print(f"  ❌ FAILS permutation test ({p_str})")
    else:
        print(f"\n⚠️ NO configs pass R1 regime gate.")
        print("The 2h model's short bias appears structural — OFI features inherently")
        print("capture selling pressure better than buying pressure.")

    # ─── Full cost scenario table for baseline ───
    print(f"\n{'='*70}")
    print("HC #693 COST ANALYSIS — Baseline strategy, all 4 cost scenarios")
    print(f"{'='*70}")
    for r in all_results:
        if r["label"].startswith("baseline_20pct|"):
            cost = r["label"].split("|")[1]
            print(f"  {cost:<20} cost={r['cost_ticks']:.3f}t  Sharpe={r['sharpe']:>6.2f}  "
                  f"net/trade={r['net_per_trade']:>+.3f}  PF={r['pf']:>5.2f}  total=${r['total_net_dollars']:>8,.0f}")

    # ─── Save results ───
    elapsed = time.time() - t0
    output = {
        "generated": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_oot_days": len(dates) - TRAIN_DAYS - PURGE_DAYS,
        "n_features": len(feature_cols),
        "leaky_excluded": list(LEAKY_FEATURES),
        "results": all_results,
    }

    with open(OUT_DIR / "regime_debias_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR / 'regime_debias_results.json'}")
    print(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
