"""
Trade-Test Walk-Forward Predictions
====================================
Tests EventTransformer (IC=0.094) and LightGBM (IC=0.139) predictions
through a trading simulator at various cost structures.

Walk-forward OOS predictions from:
  alpha_discovery/deep_models/results/oos_predictions_event_20260228_123832.npz

LightGBM predictions are re-generated inline using the same walk-forward
methodology (expanding window, 100-bar horizon, 340 MBO features).

Cost structures tested:
  - ES futures: spread=1 tick, commission=$3.00 RT
  - SPY 500sh:  spread=0.008 ticks equiv, commission=0
  - Zero cost:  spread=0, commission=0

OOS protocol:
  - Days 1-40: parameter optimization
  - Days 41+:  fixed holdout, NO re-optimization
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ---- Paths ----
LVL3_ROOT = Path(__file__).parent.parent
FEAT_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
ET_PREDS_FILE = (LVL3_ROOT / "alpha_discovery" / "deep_models" / "results" /
                 "oos_predictions_event_20260228_123832.npz")
CNN_PREDS_FILE = (LVL3_ROOT / "alpha_discovery" / "deep_models" / "results" /
                  "oos_predictions_book_20260303_234725.npz")
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Constants ----
TICK = 0.25
TICK_VAL = 12.50
BARS_PER_SEC = 10

# Cost structures (in tick units)
COST_STRUCTURES = {
    "ES_futures": {
        "spread_ticks": 1.0,        # 1 tick spread
        "comm_ticks": 3.00 / 12.50, # $3.00 RT / $12.50 per tick = 0.24 ticks
        "label": "ES (spread=1.0, comm=$3.00)",
    },
    "SPY_500sh": {
        "spread_ticks": 0.008,      # ~$0.01 spread on SPY -> 0.008 ticks equiv
        "comm_ticks": 0.40,         # IBKR $0.005/sh * 500sh * 2 = $5.00 RT / $12.50 = 0.40 ticks
        "label": "SPY 500sh (spread=0.008, comm=$5.00)",
    },
    "zero_cost": {
        "spread_ticks": 0.0,
        "comm_ticks": 0.0,
        "label": "Zero cost (spread=0, comm=0)",
    },
}

# Parameter grid
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
HOLD_BARS_LIST = [300, 600, 900, 1200, 1800, 3000]  # 30s, 60s, 90s, 120s, 180s, 300s
COOLDOWN = 50

# Train/test split
TRAIN_DAYS = 40


def sim_day(mid, spread, preds, thresh, hold_bars, spread_ticks_cost, comm_ticks,
            cooldown=50):
    """
    Market order simulator.

    Uses actual spread from data for slippage, but costs are parameterized:
      - spread_ticks_cost: additional cost per trade from crossing spread
      - comm_ticks: round-trip commission in tick units

    Returns: (total_pnl_ticks, n_trades, n_wins, trade_pnls_list)
    """
    n = min(len(mid), len(preds))
    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0
    trade_pnls = []

    for i in range(n):
        if in_pos:
            elapsed = i - entry_bar
            if elapsed >= hold_bars:
                # Time-based exit
                raw_pnl = direction * (mid[i] - entry_price) / TICK
                cost = comm_ticks + spread_ticks_cost / 2  # half spread on exit (entry half already in entry_price)
                pnl = raw_pnl - cost
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                trade_pnls.append(pnl)
                in_pos = False
                last_exit = i

        elif i - last_exit >= cooldown and abs(preds[i]) > thresh:
            # Entry: cross the spread
            entry_price = mid[i]
            direction = 1 if preds[i] > 0 else -1
            entry_bar = i
            # Entry cost: half the spread
            entry_price += direction * spread_ticks_cost * TICK / 2
            in_pos = True

    return total_pnl, trades, wins, trade_pnls


def load_mbo_data(dates_needed):
    """Load mid prices and spreads for requested dates."""
    data = {}
    for date_str in dates_needed:
        fpath = FEAT_CACHE / f"{date_str}_mbo_features.npz"
        if not fpath.exists():
            continue
        try:
            feats = np.load(str(fpath))["mbo_features"]
            data[date_str] = {
                "mid": feats[:, 0].astype(np.float32).copy(),
                "spread": feats[:, 1].astype(np.float32).copy(),
                "n_bars": len(feats),
            }
            del feats
        except Exception as e:
            print(f"  Skip {date_str}: {e}")
    return data


def load_et_predictions():
    """Load EventTransformer OOS predictions from NPZ."""
    print("Loading EventTransformer predictions...")
    data = np.load(str(ET_PREDS_FILE), allow_pickle=True)

    preds_by_date = {}
    dates = []

    for key in data.keys():
        if key.endswith("_preds"):
            date_str = key.replace("_preds", "")
            preds_by_date[date_str] = data[key]
            dates.append(date_str)

    dates.sort()
    print(f"  Loaded {len(dates)} days of ET predictions")
    print(f"  Date range: {dates[0]} to {dates[-1]}")

    return preds_by_date, dates


def load_cnn_predictions():
    """Load BookSpatialCNN OOS predictions from NPZ."""
    print("Loading BookSpatialCNN predictions...")
    if not CNN_PREDS_FILE.exists():
        print("  CNN predictions file not found, skipping")
        return {}, []
    data = np.load(str(CNN_PREDS_FILE), allow_pickle=True)

    preds_by_date = {}
    dates = []

    for key in data.keys():
        if key.endswith("_preds"):
            date_str = key.replace("_preds", "")
            preds_by_date[date_str] = data[key]
            dates.append(date_str)

    dates.sort()
    print(f"  Loaded {len(dates)} days of CNN predictions")
    print(f"  Date range: {dates[0]} to {dates[-1]}")

    return preds_by_date, dates


def generate_lgbm_predictions(dates, mbo_data):
    """
    Re-generate LightGBM walk-forward predictions.
    Same methodology as the ET training baseline:
      - Expanding window (min 5 days, max 30 days)
      - 100-bar horizon forward return
      - Subsample train by 5x to reduce autocorrelation
      - Predict ALL bars on test day
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: lightgbm not installed. Skipping LightGBM predictions.")
        return None, None

    print("\nGenerating LightGBM walk-forward predictions...")
    print("  Loading features for all days...")

    # Load all feature data
    day_features = {}
    day_targets = {}

    for date_str in dates:
        fpath = FEAT_CACHE / f"{date_str}_mbo_features.npz"
        if not fpath.exists():
            continue
        try:
            feats = np.load(str(fpath))["mbo_features"]
            mid = feats[:, 0].astype(np.float32)

            # Compute 100-bar forward return in ticks
            horizon = 100
            n = len(mid)
            fwd_ret = np.full(n, np.nan, dtype=np.float32)
            fwd_ret[:n - horizon] = (mid[horizon:] - mid[:n - horizon]) / TICK

            day_features[date_str] = feats.copy()
            day_targets[date_str] = fwd_ret
            del feats
        except Exception as e:
            print(f"  Skip {date_str}: {e}")

    print(f"  Loaded features for {len(day_features)} days")

    # LightGBM params (same as arch_benchmark / run_model_progression)
    lgbm_params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 6,
        "min_child_samples": 200,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    # Walk-forward: expanding window with max 30 days
    min_train_days = 5
    max_train_days = 30
    subsample_step = 5  # match ET training subsample

    preds_by_date = {}
    fold_ics = []

    sorted_dates = sorted(day_features.keys())

    for test_idx in range(min_train_days, len(sorted_dates)):
        test_date = sorted_dates[test_idx]

        # Training window: last max_train_days (or all if fewer)
        train_start = max(0, test_idx - max_train_days)
        train_dates = sorted_dates[train_start:test_idx]

        # Build training set (subsampled)
        X_trains = []
        y_trains = []
        for td in train_dates:
            if td not in day_features:
                continue
            feats = day_features[td]
            tgt = day_targets[td]

            # Subsample + valid mask
            n = len(feats)
            indices = np.arange(n)
            mask = (indices % subsample_step == 0) & np.isfinite(tgt) & (indices >= 5000)

            if mask.sum() > 0:
                X_trains.append(feats[mask])
                y_trains.append(tgt[mask])

        if not X_trains:
            continue

        X_train = np.vstack(X_trains)
        y_train = np.concatenate(y_trains)

        # Remove NaN/inf
        finite = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
        X_train = X_train[finite]
        y_train = y_train[finite]

        if len(X_train) < 1000:
            continue

        # Cap training size
        if len(X_train) > 2_000_000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), 2_000_000, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        # Normalize target
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        if y_std < 1e-10:
            continue
        y_train_norm = (y_train - y_mean) / y_std

        try:
            # Train
            train_ds = lgb.Dataset(X_train, label=y_train_norm, free_raw_data=True)
            model = lgb.train(
                lgbm_params, train_ds, num_boost_round=200,
                valid_sets=[train_ds], callbacks=[lgb.log_evaluation(0)],
            )

            # Predict on full test day
            test_feats = day_features[test_date]
            valid_mask = np.all(np.isfinite(test_feats), axis=1)
            predictions = np.zeros(len(test_feats), dtype=np.float32)
            if np.any(valid_mask):
                preds_raw = model.predict(test_feats[valid_mask])
                predictions[valid_mask] = (preds_raw * y_std + y_mean).astype(np.float32)

            preds_by_date[test_date] = predictions

            # Compute IC on subsampled test
            test_tgt = day_targets[test_date]
            n_test = len(test_feats)
            test_indices = np.arange(n_test)
            test_mask = ((test_indices % subsample_step == 0) &
                        np.isfinite(test_tgt) & (test_indices >= 5000))

            if test_mask.sum() > 100:
                from scipy.stats import spearmanr
                p_sub = predictions[test_mask]
                t_sub = test_tgt[test_mask]
                ic, _ = spearmanr(p_sub, t_sub)
                fold_ics.append(float(ic))
            else:
                fold_ics.append(0.0)

            if (test_idx - min_train_days) % 10 == 0:
                ic_val = fold_ics[-1] if fold_ics else 0.0
                print(f"  [{test_idx+1}/{len(sorted_dates)}] {test_date} "
                      f"IC={ic_val:+.4f}  train={len(X_train):,}")

            del model, train_ds, X_train, y_train

        except Exception as e:
            print(f"  [{test_idx+1}/{len(sorted_dates)}] {test_date}: ERROR: {e}")
            continue

    # Summary
    if fold_ics:
        ic_arr = np.array(fold_ics)
        valid = ic_arr[ic_arr != 0.0]
        if len(valid) > 0:
            print(f"\n  LightGBM Walk-Forward Summary:")
            print(f"    Folds: {len(valid)}")
            print(f"    Mean IC: {np.mean(valid):+.4f}")
            print(f"    Std IC:  {np.std(valid):.4f}")
            print(f"    Pct positive: {100*np.mean(valid > 0):.1f}%")

    lgbm_dates = sorted(preds_by_date.keys())
    print(f"  Generated predictions for {len(lgbm_dates)} days")

    return preds_by_date, lgbm_dates


def run_parameter_scan(signal_preds, dates, mbo_data, cost_key, cost_cfg):
    """
    Run parameter grid scan over thresholds and hold periods.

    Returns dict of (thresh, hold_bars) -> {pnl, trades, wins, sharpe, day_pnls, ...}
    """
    spread_cost = cost_cfg["spread_ticks"]
    comm_cost = cost_cfg["comm_ticks"]

    results = {}

    for thresh in THRESHOLDS:
        for hold_bars in HOLD_BARS_LIST:
            total_pnl = 0.0
            total_trades = 0
            total_wins = 0
            day_pnls = []

            for date in dates:
                if date not in signal_preds or date not in mbo_data:
                    day_pnls.append(0.0)
                    continue

                preds = signal_preds[date]
                mid = mbo_data[date]["mid"]
                spread = mbo_data[date]["spread"]

                # Truncate to min length
                n = min(len(mid), len(preds))

                pnl_ticks, trades, wins, _ = sim_day(
                    mid[:n], spread[:n], preds[:n],
                    thresh, hold_bars, spread_cost, comm_cost, COOLDOWN
                )

                pnl_dollars = pnl_ticks * TICK_VAL
                total_pnl += pnl_dollars
                total_trades += trades
                total_wins += wins
                day_pnls.append(pnl_dollars)

            # Stats
            avg_daily = np.mean(day_pnls) if day_pnls else 0
            std_daily = np.std(day_pnls) if day_pnls else 1
            sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0
            win_rate = total_wins / total_trades if total_trades > 0 else 0
            active_days = [p for p in day_pnls if p != 0]
            pct_pos_days = (np.mean([p > 0 for p in active_days]) * 100
                          if active_days else 0)

            results[(thresh, hold_bars)] = {
                "thresh": thresh,
                "hold_bars": hold_bars,
                "hold_sec": hold_bars / BARS_PER_SEC,
                "total_pnl": round(total_pnl, 2),
                "trades": total_trades,
                "wins": total_wins,
                "win_rate": round(win_rate, 4),
                "sharpe": round(sharpe, 3),
                "pct_pos_days": round(pct_pos_days, 1),
                "avg_daily": round(avg_daily, 2),
                "std_daily": round(std_daily, 2),
                "day_pnls": [round(p, 2) for p in day_pnls],
            }

    return results


def find_best_config(scan_results):
    """Find best config by total PnL."""
    best_key = None
    best_pnl = -np.inf
    for key, res in scan_results.items():
        if res["total_pnl"] > best_pnl:
            best_pnl = res["total_pnl"]
            best_key = key
    return best_key, scan_results[best_key] if best_key else None


def run_oos_test(signal_preds, train_dates, test_dates, mbo_data,
                 cost_key, cost_cfg, signal_name):
    """
    Full OOS test:
    1. Optimize parameters on train_dates
    2. Apply fixed best params to test_dates
    3. Report monthly breakdown
    """
    spread_cost = cost_cfg["spread_ticks"]
    comm_cost = cost_cfg["comm_ticks"]

    # Phase 1: Parameter optimization on train set
    train_results = run_parameter_scan(
        signal_preds, train_dates, mbo_data, cost_key, cost_cfg
    )
    best_key, best_train = find_best_config(train_results)

    if best_train is None:
        return None

    thresh = best_train["thresh"]
    hold_bars = best_train["hold_bars"]

    # Phase 2: Fixed params on OOS
    oos_total_pnl = 0.0
    oos_trades = 0
    oos_wins = 0
    oos_day_pnls = []
    oos_day_details = []

    for date in test_dates:
        if date not in signal_preds or date not in mbo_data:
            oos_day_pnls.append(0.0)
            oos_day_details.append({"date": date, "pnl": 0, "trades": 0})
            continue

        preds = signal_preds[date]
        mid = mbo_data[date]["mid"]
        spread = mbo_data[date]["spread"]
        n = min(len(mid), len(preds))

        pnl_ticks, trades, wins, _ = sim_day(
            mid[:n], spread[:n], preds[:n],
            thresh, hold_bars, spread_cost, comm_cost, COOLDOWN
        )

        pnl_dollars = pnl_ticks * TICK_VAL
        oos_total_pnl += pnl_dollars
        oos_trades += trades
        oos_wins += wins
        oos_day_pnls.append(pnl_dollars)
        oos_day_details.append({
            "date": date, "pnl": round(pnl_dollars, 2), "trades": trades
        })

    # OOS stats
    avg_oos = np.mean(oos_day_pnls) if oos_day_pnls else 0
    std_oos = np.std(oos_day_pnls) if oos_day_pnls else 1
    sharpe_oos = (avg_oos / std_oos * np.sqrt(252)) if std_oos > 0 else 0
    wr_oos = oos_wins / oos_trades if oos_trades > 0 else 0
    active = [p for p in oos_day_pnls if p != 0]
    pct_pos_oos = np.mean([p > 0 for p in active]) * 100 if active else 0

    # Monthly breakdown
    monthly = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "days": 0})
    for date, pnl in zip(test_dates, oos_day_pnls):
        m = date[:7]
        monthly[m]["pnl"] += pnl
        monthly[m]["trades"] += 1
        monthly[m]["days"] += 1

    # Both-halves test
    half = len(test_dates) // 2
    first_half_pnl = sum(oos_day_pnls[:half])
    second_half_pnl = sum(oos_day_pnls[half:])

    # Count positive train configs
    pos_train = sum(1 for r in train_results.values() if r["total_pnl"] > 0)
    total_configs = len(train_results)

    return {
        "signal": signal_name,
        "cost": cost_key,
        "cost_label": cost_cfg["label"],
        "best_config": {
            "thresh": thresh,
            "hold_bars": hold_bars,
            "hold_sec": hold_bars / BARS_PER_SEC,
        },
        "train": {
            "pnl": round(best_train["total_pnl"], 2),
            "sharpe": best_train["sharpe"],
            "trades": best_train["trades"],
            "win_rate": best_train["win_rate"],
            "n_days": len(train_dates),
            "pos_configs": pos_train,
            "total_configs": total_configs,
        },
        "oos": {
            "pnl": round(oos_total_pnl, 2),
            "sharpe": round(sharpe_oos, 3),
            "trades": oos_trades,
            "win_rate": round(wr_oos, 4),
            "pct_pos_days": round(pct_pos_oos, 1),
            "avg_daily": round(avg_oos, 2),
            "std_daily": round(std_oos, 2),
            "first_half_pnl": round(first_half_pnl, 2),
            "second_half_pnl": round(second_half_pnl, 2),
            "n_days": len(test_dates),
            "monthly": {m: {"pnl": round(d["pnl"], 2), "days": d["days"]}
                       for m, d in sorted(monthly.items())},
            "day_pnls": [round(p, 2) for p in oos_day_pnls],
        },
    }


def print_summary_table(all_results):
    """Print a clear comparison table."""
    print("\n" + "=" * 90)
    print("TRADE TEST SUMMARY TABLE")
    print("=" * 90)

    header = (f"{'Signal':<12} {'Cost':<14} {'Config':>12} "
              f"{'Train$':>10} {'OOS$':>10} {'OOS Sharpe':>11} "
              f"{'Trades':>7} {'WR':>6} {'1stH$':>9} {'2ndH$':>9}")
    print(header)
    print("-" * 90)

    for r in all_results:
        if r is None:
            continue
        cfg = f"t={r['best_config']['thresh']:.1f} h={int(r['best_config']['hold_sec'])}s"
        oos = r["oos"]
        trn = r["train"]

        line = (f"{r['signal']:<12} {r['cost']:<14} {cfg:>12} "
                f"${trn['pnl']:>+9,.0f} ${oos['pnl']:>+9,.0f} "
                f"{oos['sharpe']:>+10.2f} "
                f"{oos['trades']:>7d} {oos['win_rate']:>5.1%} "
                f"${oos['first_half_pnl']:>+8,.0f} ${oos['second_half_pnl']:>+8,.0f}")
        print(line)

    print("-" * 90)

    # Print monthly breakdown for key results
    print("\n" + "=" * 70)
    print("MONTHLY OOS BREAKDOWN")
    print("=" * 70)

    for r in all_results:
        if r is None:
            continue
        monthly = r["oos"]["monthly"]
        if not monthly:
            continue

        print(f"\n  {r['signal']} @ {r['cost']}:")
        for m in sorted(monthly.keys()):
            d = monthly[m]
            print(f"    {m}: ${d['pnl']:>+9,.0f}  ({d['days']} days)")
        print(f"    {'TOTAL':>7}: ${r['oos']['pnl']:>+9,.0f}  ({r['oos']['n_days']} days)")


def main():
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("WALK-FORWARD TRADE TEST")
    print("EventTransformer (IC=0.094) + LightGBM (IC~0.139)")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Hold periods (bars): {HOLD_BARS_LIST}")
    print(f"Hold periods (sec): {[h/BARS_PER_SEC for h in HOLD_BARS_LIST]}")
    print(f"Cooldown: {COOLDOWN} bars")
    print(f"Train/OOS split: first {TRAIN_DAYS} days / rest")

    # Step 1: Load ET and CNN predictions
    et_preds, et_dates = load_et_predictions()
    cnn_preds, cnn_dates = load_cnn_predictions()

    # Step 2: Load MBO data for all dates
    all_dates_needed = sorted(set(et_dates) | set(cnn_dates))
    print(f"\nLoading MBO data for {len(all_dates_needed)} days...")
    mbo_data = load_mbo_data(all_dates_needed)
    print(f"  Loaded MBO data for {len(mbo_data)} days")

    # Step 3: Generate LightGBM predictions
    lgbm_preds, lgbm_dates = generate_lgbm_predictions(et_dates, mbo_data)

    # Determine common dates for fair comparison
    if lgbm_preds:
        common_dates = sorted(set(et_dates) & set(lgbm_dates) & set(mbo_data.keys()))
    else:
        common_dates = sorted(set(et_dates) & set(mbo_data.keys()))

    print(f"\nCommon dates for testing: {len(common_dates)}")
    print(f"  Range: {common_dates[0]} to {common_dates[-1]}")

    # Split into train and OOS
    train_dates = common_dates[:TRAIN_DAYS]
    test_dates = common_dates[TRAIN_DAYS:]

    print(f"\n  TRAIN: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"  OOS:   {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

    # Step 4: Normalize predictions to z-scores per day for fair comparison
    # ET predictions are already model output (normalized during training)
    # LightGBM predictions are in raw tick space - need to z-score per day

    print("\nNormalizing predictions to per-day z-scores...")

    et_preds_z = {}
    for date in common_dates:
        if date in et_preds:
            p = et_preds[date].astype(np.float64)
            std = np.std(p)
            if std > 1e-10:
                et_preds_z[date] = ((p - np.mean(p)) / std).astype(np.float32)
            else:
                et_preds_z[date] = np.zeros_like(p)

    lgbm_preds_z = {}
    if lgbm_preds:
        for date in common_dates:
            if date in lgbm_preds:
                p = lgbm_preds[date].astype(np.float64)
                std = np.std(p)
                if std > 1e-10:
                    lgbm_preds_z[date] = ((p - np.mean(p)) / std).astype(np.float32)
                else:
                    lgbm_preds_z[date] = np.zeros_like(p)

    cnn_preds_z = {}
    if cnn_preds:
        for date in common_dates:
            if date in cnn_preds:
                p = cnn_preds[date].astype(np.float64)
                std = np.std(p)
                if std > 1e-10:
                    cnn_preds_z[date] = ((p - np.mean(p)) / std).astype(np.float32)
                else:
                    cnn_preds_z[date] = np.zeros_like(p)

    # Print prediction stats
    print("\n  ET prediction stats (z-scored):")
    sample_date = common_dates[0]
    if sample_date in et_preds_z:
        p = et_preds_z[sample_date]
        print(f"    {sample_date}: mean={np.mean(p):.4f} std={np.std(p):.4f} "
              f"min={np.min(p):.3f} max={np.max(p):.3f}")

    if lgbm_preds_z:
        print("  LightGBM prediction stats (z-scored):")
        if sample_date in lgbm_preds_z:
            p = lgbm_preds_z[sample_date]
            print(f"    {sample_date}: mean={np.mean(p):.4f} std={np.std(p):.4f} "
                  f"min={np.min(p):.3f} max={np.max(p):.3f}")

    # Step 5: Run trade tests
    all_results = []
    signals_to_test = [("ET", et_preds_z)]
    if cnn_preds_z:
        signals_to_test.append(("CNN", cnn_preds_z))
    if lgbm_preds_z:
        signals_to_test.append(("LightGBM", lgbm_preds_z))

    for signal_name, signal_preds in signals_to_test:
        print(f"\n{'='*70}")
        print(f"TESTING: {signal_name}")
        print(f"{'='*70}")

        for cost_key, cost_cfg in COST_STRUCTURES.items():
            print(f"\n  --- {cost_cfg['label']} ---")

            result = run_oos_test(
                signal_preds, train_dates, test_dates, mbo_data,
                cost_key, cost_cfg, signal_name
            )

            if result:
                trn = result["train"]
                oos = result["oos"]
                cfg = result["best_config"]

                print(f"  Best config: thresh={cfg['thresh']:.1f}, "
                      f"hold={int(cfg['hold_sec'])}s")
                print(f"  TRAIN: ${trn['pnl']:+,.0f} | Sharpe={trn['sharpe']:+.2f} | "
                      f"{trn['trades']} trades | "
                      f"pos configs: {trn['pos_configs']}/{trn['total_configs']}")
                print(f"  OOS:   ${oos['pnl']:+,.0f} | Sharpe={oos['sharpe']:+.2f} | "
                      f"{oos['trades']} trades | WR={oos['win_rate']:.1%}")
                print(f"  OOS 1st half: ${oos['first_half_pnl']:+,.0f} | "
                      f"2nd half: ${oos['second_half_pnl']:+,.0f}")

                all_results.append(result)
            else:
                print(f"  No results (insufficient data)")

    # Also test raw (un-normalized) predictions for comparison
    print(f"\n{'='*70}")
    print(f"TESTING: ET RAW (un-normalized predictions)")
    print(f"{'='*70}")

    for cost_key, cost_cfg in COST_STRUCTURES.items():
        print(f"\n  --- {cost_cfg['label']} ---")

        result = run_oos_test(
            et_preds, train_dates, test_dates, mbo_data,
            cost_key, cost_cfg, "ET_raw"
        )

        if result:
            trn = result["train"]
            oos = result["oos"]
            cfg = result["best_config"]

            print(f"  Best config: thresh={cfg['thresh']:.1f}, "
                  f"hold={int(cfg['hold_sec'])}s")
            print(f"  TRAIN: ${trn['pnl']:+,.0f} | Sharpe={trn['sharpe']:+.2f} | "
                  f"{trn['trades']} trades")
            print(f"  OOS:   ${oos['pnl']:+,.0f} | Sharpe={oos['sharpe']:+.2f} | "
                  f"{oos['trades']} trades | WR={oos['win_rate']:.1%}")

            all_results.append(result)

    # Step 6: Print summary
    print_summary_table(all_results)

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Find best ES result
    es_results = [r for r in all_results if r and r["cost"] == "ES_futures"]
    if es_results:
        best_es = max(es_results, key=lambda r: r["oos"]["pnl"])
        if best_es["oos"]["pnl"] > 0 and best_es["oos"]["sharpe"] > 1.0:
            print(f"  >>> PROFITABLE at ES cost! {best_es['signal']}: "
                  f"${best_es['oos']['pnl']:+,.0f} Sharpe={best_es['oos']['sharpe']:+.2f}")
        elif best_es["oos"]["pnl"] > 0:
            print(f"  >>> MARGINALLY POSITIVE at ES cost. {best_es['signal']}: "
                  f"${best_es['oos']['pnl']:+,.0f} Sharpe={best_es['oos']['sharpe']:+.2f}")
        else:
            print(f"  >>> NOT PROFITABLE at ES cost. Best: {best_es['signal']}: "
                  f"${best_es['oos']['pnl']:+,.0f}")

    # Find best zero-cost result
    zero_results = [r for r in all_results if r and r["cost"] == "zero_cost"]
    if zero_results:
        best_zero = max(zero_results, key=lambda r: r["oos"]["pnl"])
        if best_zero["oos"]["pnl"] > 0:
            print(f"  >>> Raw edge exists (zero cost): {best_zero['signal']}: "
                  f"${best_zero['oos']['pnl']:+,.0f} Sharpe={best_zero['oos']['sharpe']:+.2f}")
        else:
            print(f"  >>> NO raw edge even at zero cost! Signal is pure noise for trading.")

    # SPY result
    spy_results = [r for r in all_results if r and r["cost"] == "SPY_500sh"]
    if spy_results:
        best_spy = max(spy_results, key=lambda r: r["oos"]["pnl"])
        if best_spy["oos"]["pnl"] > 0:
            print(f"  >>> SPY viable: {best_spy['signal']}: "
                  f"${best_spy['oos']['pnl']:+,.0f} Sharpe={best_spy['oos']['sharpe']:+.2f}")

    # Step 7: Save results
    save_data = {
        "timestamp": timestamp,
        "methodology": "walk_forward_trade_test",
        "et_ic": 0.094,
        "lgbm_ic_ref": 0.139,
        "horizon_bars": 100,
        "horizon_sec": 10,
        "train_days": TRAIN_DAYS,
        "oos_days": len(test_dates),
        "date_range": {"first": common_dates[0], "last": common_dates[-1]},
        "train_range": {"first": train_dates[0], "last": train_dates[-1]},
        "oos_range": {"first": test_dates[0], "last": test_dates[-1]},
        "thresholds_tested": THRESHOLDS,
        "hold_bars_tested": HOLD_BARS_LIST,
        "cooldown": COOLDOWN,
        "cost_structures": {k: {"spread_ticks": v["spread_ticks"],
                                "comm_ticks": v["comm_ticks"],
                                "label": v["label"]}
                           for k, v in COST_STRUCTURES.items()},
        "results": [],
        "elapsed_sec": round(time.time() - t0, 1),
    }

    for r in all_results:
        if r:
            # Remove day_pnls from saved results to keep file manageable
            r_save = json.loads(json.dumps(r, default=str))
            save_data["results"].append(r_save)

    out_path = RESULTS_DIR / f"trade_test_walkforward_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    print(f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
