#!/usr/bin/env python3
"""
Clean Walk-Forward Validation for 2h ES LGBM Model
===================================================
Ground-truth replay — trains fresh on each fold, no data leakage.
Reports honest OOS metrics with per-day and per-regime breakdown.

This is a STANDALONE script — no dependency on the paper engine code.
Every line is auditable.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── Constants ───
ROOT = Path("/home/jupiter/Lvl3Quant")
BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
COST_RT_TICKS = 1.376  # AMP/Rithmic market order RT
TRAIN_DAYS = 60
PURGE_DAYS = 5
HORIZON_BARS = 2
MIN_CONFIDENCE_TICKS = 16.0  # bottom 20% filter
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

        # Microprice
        mp_valid = mp[mp > 0]
        c_valid = c[:len(mp_valid)]
        if len(mp_valid) > 0 and len(c_valid) > 0:
            mp_dev = (mp_valid - c_valid[:len(mp_valid)]) / np.maximum(c_valid[:len(mp_valid)], 1)
            rec["microprice_dev_mean"] = mp_dev.mean()
            rec["microprice_dev_trend"] = safe_polyfit_slope(mp_dev)
            rec["microprice_dev_late"] = mp_dev[-len(mp_dev)//3:].mean() if len(mp_dev) >= 3 else mp_dev.mean()
        else:
            rec["microprice_dev_mean"] = rec["microprice_dev_trend"] = rec["microprice_dev_late"] = 0

        # VWAP
        vwap_valid = vwap[vwap > 0]
        if len(vwap_valid) > 0:
            vwap_dev = (c[:len(vwap_valid)] - vwap_valid) / np.maximum(vwap_valid, 1)
            rec["vwap_dev_final"] = vwap_dev[-1] if len(vwap_dev) > 0 else 0
            rec["vwap_dev_trend"] = safe_polyfit_slope(vwap_dev)
        else:
            rec["vwap_dev_final"] = rec["vwap_dev_trend"] = 0

        # Volume-weighted return
        rec["vw_return"] = np.sum(ret * v[:len(ret)]) / v[:len(ret)].sum() if v.sum() > 0 and len(ret) <= len(v) else 0

        # Return autocorr
        if len(ret) > 10:
            rec["return_autocorr_1"] = np.corrcoef(ret[:-1], ret[1:])[0,1] if ret[:-1].std() > 0 and ret[1:].std() > 0 else 0
            rec["return_autocorr_5"] = np.corrcoef(ret[:-5], ret[5:])[0,1] if ret[:-5].std() > 0 and ret[5:].std() > 0 else 0
        else:
            rec["return_autocorr_1"] = rec["return_autocorr_5"] = 0

        # Trade size
        if tc.sum() > 0 and v.sum() > 0:
            rec["avg_trade_size"] = v.sum() / tc.sum()
            v_sorted = np.sort(v)
            rec["volume_top_half_ratio"] = v_sorted[len(v_sorted)//2:].sum() / max(v.sum(), 1)
        else:
            rec["avg_trade_size"] = 0
            rec["volume_top_half_ratio"] = 0.5

        # OFI acceleration
        if len(ofi) > 5:
            rec["ofi_acceleration"] = ofi[len(ofi)//2:].sum() - ofi[:len(ofi)//2].sum()
            rec["ofi_curvature"] = np.polyfit(np.arange(len(ofi)), ofi, 2)[0] if len(ofi) > 3 else 0
        else:
            rec["ofi_acceleration"] = rec["ofi_curvature"] = 0

        # Time-of-day
        rec["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        rec["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Range position
        rng = c.max() - c.min()
        rec["hl_range_position"] = (c[-1] - c.min()) / max(rng, 1)

        # Vol-price divergence
        if len(v) > 5 and v.std() > 0 and c.std() > 0:
            rec["vol_price_divergence"] = float(np.sign(c[-1] - c[0]) != np.sign(safe_polyfit_slope(v)))
        else:
            rec["vol_price_divergence"] = 0

        # OFI vol
        rec["ofi_vol"] = ofi.std() if ofi.std() > 0 else 0
        rec["ofi_vol_normalized"] = ofi.std() / max(abs(ofi.mean()), 1) if ofi.std() > 0 else 0

        records.append(rec)

    hourly = pd.DataFrame(records).sort_values("ts").reset_index(drop=True)

    # Rolling features
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

    # Momentum
    hourly["mom_2h"] = hourly["close"].pct_change(2)
    hourly["mom_4h"] = hourly["close"].pct_change(4)
    hourly["mom_6h"] = hourly["close"].pct_change(6)

    # Regime context (expanding rank to avoid look-ahead)
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
    return [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "float32", "int64", "int32"]]


def main():
    import lightgbm as lgb
    import gc

    print("=" * 70)
    print("2h ES LGBM — CLEAN Walk-Forward Validation")
    print("=" * 70)

    # Load ALL minute bar data
    files = sorted(BAR_DIR.glob("*.parquet"))
    print(f"Loading {len(files)} minute bar files...")
    dfs = [pd.read_parquet(f) for f in files]
    minute_df = pd.concat(dfs).sort_values("ts_minute").reset_index(drop=True)
    print(f"Loaded {len(minute_df):,} minute bars")

    # Build hourly features
    print("Building hourly features...")
    hourly = build_hourly_features(minute_df)
    del minute_df
    gc.collect()

    # Add forward labels
    hourly["fwd_ticks"] = hourly["close"].shift(-HORIZON_BARS) - hourly["close"]

    # Null overnight gaps
    for i in range(len(hourly) - HORIZON_BARS):
        ts_now = hourly["ts"].iloc[i]
        ts_fwd = hourly["ts"].iloc[i + HORIZON_BARS]
        if (ts_fwd - ts_now).total_seconds() > 8 * 3600:
            hourly.loc[hourly.index[i], "fwd_ticks"] = np.nan

    # Filter intraday-clean (exclude hours 19-20 UTC)
    hourly_clean = hourly[~hourly["hour"].isin([19, 20])].copy()

    dates = sorted(hourly_clean["date"].unique())
    feature_cols = get_feature_cols(hourly_clean)
    print(f"Features: {len(feature_cols)}, Dates: {len(dates)}")
    print(f"OOT window: {dates[TRAIN_DAYS + PURGE_DAYS]} to {dates[-1]}")
    print(f"OOT days: {len(dates) - TRAIN_DAYS - PURGE_DAYS}")
    print()

    # ─── Walk-Forward Loop ───
    all_trades = []
    oos_preds = []

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
        except Exception as e:
            continue

        # OOS predictions
        X_oot = oot_signal[feature_cols].fillna(0).values.astype(np.float32)
        preds = model.predict(X_oot)

        # OOS IC for this fold
        oot_labels = oot_signal["fwd_ticks"].values
        valid_mask = ~np.isnan(oot_labels)
        if valid_mask.sum() > 3:
            fold_ic = float(stats.spearmanr(preds[valid_mask], oot_labels[valid_mask])[0])
            oos_preds.append({"date": oot_date, "ic": fold_ic, "n": int(valid_mask.sum())})

        # Generate trades
        for j, (idx, row) in enumerate(oot_signal.iterrows()):
            pred = preds[j]
            if abs(pred) < MIN_CONFIDENCE_TICKS:
                continue

            signal = 1 if pred > 0 else -1
            entry_price = row["close"]
            entry_hour = int(row["hour"])
            exit_hour = entry_hour + HORIZON_BARS

            # Find exit bar
            exit_bars = hourly[(hourly["date"] == oot_date) & (hourly["hour"] == exit_hour)]
            if exit_bars.empty:
                continue

            exit_price = exit_bars.iloc[0]["close"]

            # VERIFIED: close is in tick units, raw diff = ticks. NO multiplier.
            if signal == 1:
                gross_ticks = exit_price - entry_price
            else:
                gross_ticks = entry_price - exit_price

            net_ticks = gross_ticks - COST_RT_TICKS

            all_trades.append({
                "date": oot_date,
                "entry_hour": entry_hour,
                "direction": "LONG" if signal == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "prediction": pred,
                "gross_ticks": gross_ticks,
                "net_ticks": net_ticks,
            })

        del model
        gc.collect()

        if (i - TRAIN_DAYS - PURGE_DAYS) % 20 == 0:
            print(f"  Processed {i - TRAIN_DAYS - PURGE_DAYS + 1}/{len(dates) - TRAIN_DAYS - PURGE_DAYS} OOT days...")

    # ─── Results ───
    trades = pd.DataFrame(all_trades)
    ic_df = pd.DataFrame(oos_preds)

    print()
    print("=" * 70)
    print("RESULTS — CLEAN WALK-FORWARD (NO LEAKAGE)")
    print("=" * 70)

    if trades.empty:
        print("NO TRADES GENERATED")
        return

    print(f"\n--- Trade-Level ---")
    print(f"Total trades:     {len(trades)}")
    print(f"OOT days traded:  {trades['date'].nunique()}")
    print(f"Trades/day:       {len(trades) / trades['date'].nunique():.1f}")
    print(f"Avg gross ticks:  {trades['gross_ticks'].mean():.2f}")
    print(f"Avg net ticks:    {trades['net_ticks'].mean():.2f}")
    print(f"Win rate (net>0): {(trades['net_ticks'] > 0).mean():.1%}")
    print(f"Gross ticks std:  {trades['gross_ticks'].std():.2f}")

    # Per-direction
    for d in ["LONG", "SHORT"]:
        sub = trades[trades["direction"] == d]
        if len(sub) > 0:
            print(f"  {d}: n={len(sub)}, avg_net={sub['net_ticks'].mean():.2f}, WR={(sub['net_ticks']>0).mean():.1%}")

    # Daily aggregation
    daily = trades.groupby("date")["net_ticks"].sum()
    daily_n = trades.groupby("date").size()
    print(f"\n--- Daily ---")
    print(f"Days traded:      {len(daily)}")
    print(f"Daily mean:       {daily.mean():.2f} ticks")
    print(f"Daily std:        {daily.std():.2f} ticks")
    if daily.std() > 0:
        sharpe = daily.mean() / daily.std() * np.sqrt(252)
        sortino_down = daily[daily < 0].std()
        sortino = daily.mean() / sortino_down * np.sqrt(252) if sortino_down > 0 else float("inf")
        print(f"Sharpe (ann):     {sharpe:.2f}")
        print(f"Sortino (ann):    {sortino:.2f}")
    print(f"Day WR:           {(daily > 0).mean():.1%}")
    print(f"Max daily loss:   {daily.min():.2f} ticks")
    print(f"Max daily gain:   {daily.max():.2f} ticks")
    print(f"Profit factor:    {daily[daily>0].sum() / abs(daily[daily<0].sum()):.2f}" if daily[daily<0].sum() != 0 else "PF: inf")

    # Cumulative
    cum = daily.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    print(f"Max drawdown:     {dd.min():.2f} ticks")
    print(f"Total net:        {daily.sum():.2f} ticks (${daily.sum() * 12.50:,.2f})")

    # OOS IC
    if not ic_df.empty:
        print(f"\n--- OOS IC ---")
        print(f"Mean daily IC:    {ic_df['ic'].mean():.4f}")
        print(f"IC std:           {ic_df['ic'].std():.4f}")
        print(f"IC Sharpe:        {ic_df['ic'].mean() / ic_df['ic'].std():.2f}" if ic_df['ic'].std() > 0 else "")
        print(f"Pct IC > 0:       {(ic_df['ic'] > 0).mean():.1%}")

    # Regime stratification (simple: green/red/flat based on daily close change)
    print(f"\n--- Regime Stratification ---")
    # Get daily ES close-to-close
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

    trades["regime"] = trades["date"].map(regime_map)
    for regime in ["green", "red", "flat"]:
        sub = trades[trades["regime"] == regime]
        if len(sub) > 0:
            daily_r = sub.groupby("date")["net_ticks"].sum()
            r_sharpe = daily_r.mean() / daily_r.std() * np.sqrt(252) if daily_r.std() > 0 else float("inf")
            print(f"  {regime:5s}: n={len(sub):4d}, days={sub['date'].nunique():3d}, "
                  f"avg_net={sub['net_ticks'].mean():+.2f}, WR={(sub['net_ticks']>0).mean():.1%}, "
                  f"daily_Sharpe={r_sharpe:.2f}")

    # Hour-of-day analysis
    print(f"\n--- Hour of Day (UTC) ---")
    for h in SIGNAL_HOURS_UTC:
        sub = trades[trades["entry_hour"] == h]
        if len(sub) > 0:
            print(f"  h={h} ({h-4:02d}:00 ET): n={len(sub):4d}, avg_net={sub['net_ticks'].mean():+.2f}, "
                  f"WR={(sub['net_ticks']>0).mean():.1%}")

    # Save results
    out_dir = ROOT / "output" / "lh_2h_clean_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "trades.csv", index=False)
    ic_df.to_csv(out_dir / "daily_ic.csv", index=False)
    daily.to_csv(out_dir / "daily_pnl.csv")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
