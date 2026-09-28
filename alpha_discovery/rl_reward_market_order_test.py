"""
rl_reward OOS Test via Market Order Simulation (NO Rust needed)
================================================================
Walk-forward trains the rl_reward LightGBM model, generates OOS predictions,
then tests them with market orders (spread-crossing execution).

This bypasses the Rust binary and runs entirely in Python.
Market order cost: 1.24 ticks/RT (1.0 spread + 0.24 commission)

Usage:
    python alpha_discovery/rl_reward_market_order_test.py
"""

import gc
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FEAT_CACHE_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RESULTS_DIR / f'rl_reward_mktorder_{_ts}.log'), mode='w'),
    ]
)
log = logging.getLogger('rl_mkt')

TICK = 0.25
TICK_VAL = 12.50
COMM_TICKS = 3.00 / TICK_VAL  # 0.24
SPREAD_TICKS = 1.0
TOTAL_COST = SPREAD_TICKS + COMM_TICKS  # 1.24 ticks
BARS_PER_SEC = 10
HORIZON_30S = 300  # 30s = 300 bars at 100ms
N_IS_DAYS = 48
MIN_IS_DAYS = 20
PURGE_GAP = 1  # 1-day purge between IS and OOS

# LightGBM parameters — conservative to prevent overfitting
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_samples': 200,
    'verbose': -1,
    'n_jobs': -1,
}

# Feature columns to EXCLUDE (raw price/vol that could leak)
EXCLUDE_COLS = list(range(13))  # columns 0-12

# Market order sim sweep parameters
THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
HOLD_BARS = [50, 100, 300, 600, 1200, 3000]  # 5s, 10s, 30s, 60s, 120s, 300s
TRAILS = [0, 4, 8, 12]
COOLDOWNS = [10, 50]  # 1s, 5s cooldown between trades


def compute_rl_reward_targets(mid_prices: np.ndarray, horizon_bars: int = HORIZON_30S) -> np.ndarray:
    """Compute rl_reward combined signal: buy_q - sell_q (positive = long, negative = short)."""
    N = len(mid_prices)
    combined = np.zeros(N, dtype=np.float32)

    for i in range(N - horizon_bars):
        exit_price = mid_prices[i + horizon_bars]
        entry_price = mid_prices[i]
        return_ticks = (exit_price - entry_price) / TICK

        # buy_q = return - cost, sell_q = -return - cost
        # combined = buy_q - sell_q = 2 * return_ticks
        # (costs cancel in the difference)
        combined[i] = 2.0 * return_ticks

    # Normalize to roughly [-1, +1]
    valid = combined[combined != 0]
    if len(valid) > 0:
        std = np.std(valid)
        if std > 0:
            combined = combined / (4 * std)

    return combined


def load_feature_day(date_str: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load features and mid prices for a single date."""
    feat_file = FEAT_CACHE_DIR / f'{date_str}_mbo_features.npz'
    if not feat_file.exists():
        return None
    data = np.load(feat_file)
    features = data['mbo_features']
    mid_prices = features[:, 0].astype(np.float64)
    return features, mid_prices


def sim_day_market_order(mid, spread, preds, thresh, hold_bars, trail, cooldown=10, max_trades=0):
    """Market order sim for one day. Returns (pnl_ticks, trades, wins, daily_pnl_list)."""
    n = min(len(mid), len(preds))
    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0
    peak = 0.0

    for i in range(n):
        if in_pos:
            if direction == 1:
                unrealized = (mid[i] - entry_price) / TICK
            else:
                unrealized = (entry_price - mid[i]) / TICK

            peak = max(peak, unrealized)
            bars = i - entry_bar

            do_exit = False
            if bars >= hold_bars:
                do_exit = True
            elif trail > 0 and (peak - unrealized) >= trail:
                do_exit = True

            if do_exit:
                exit_cost = spread[i] / 2.0 / TICK + COMM_TICKS
                pnl = unrealized - exit_cost
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i

        elif not in_pos and (i - last_exit >= cooldown):
            if max_trades > 0 and trades >= max_trades:
                continue

            p = preds[i]
            if abs(p) > thresh:
                if p > 0:
                    entry_price = mid[i] + spread[i] / 2.0  # buy at ask
                    direction = 1
                else:
                    entry_price = mid[i] - spread[i] / 2.0  # sell at bid
                    direction = -1
                in_pos = True
                entry_bar = i
                peak = 0.0

    # Force close
    if in_pos:
        i = n - 1
        if direction == 1:
            unrealized = (mid[i] - entry_price) / TICK
        else:
            unrealized = (entry_price - mid[i]) / TICK
        exit_cost = spread[i] / 2.0 / TICK + COMM_TICKS
        pnl = unrealized - exit_cost
        total_pnl += pnl
        trades += 1
        if pnl > 0:
            wins += 1

    return total_pnl, trades, wins


def main():
    log.info("=" * 70)
    log.info("rl_reward OOS Test — Market Order Simulation")
    log.info("=" * 70)
    log.info(f"Cost: {TOTAL_COST:.2f} ticks/RT ({SPREAD_TICKS}t spread + {COMM_TICKS:.2f}t comm)")

    # Get all available dates
    all_feat_files = sorted(FEAT_CACHE_DIR.glob('*_mbo_features.npz'))
    all_dates = [f.stem.replace('_mbo_features', '') for f in all_feat_files]
    log.info(f"Available dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")

    # Walk-forward: use first N_IS_DAYS for training, rest for OOS
    oos_start = N_IS_DAYS + PURGE_GAP
    if oos_start >= len(all_dates):
        log.error("Not enough dates for walk-forward test")
        return

    oos_dates = all_dates[oos_start:]
    log.info(f"OOS dates: {len(oos_dates)} ({oos_dates[0]} to {oos_dates[-1]})")

    import lightgbm as lgb

    # Feature columns to use
    sample_feat, _ = load_feature_day(all_dates[0])
    n_features = sample_feat.shape[1]
    feature_cols = [i for i in range(n_features) if i not in EXCLUDE_COLS]
    log.info(f"Using {len(feature_cols)}/{n_features} features (excluded first {len(EXCLUDE_COLS)})")
    del sample_feat

    # ========================================================
    # PHASE 1: Walk-forward training → generate OOS predictions
    # ========================================================
    log.info(f"\n{'='*70}")
    log.info("PHASE 1: Walk-forward rl_reward training")
    log.info(f"{'='*70}")

    oos_predictions = {}
    oos_mid_prices = {}
    oos_spreads = {}
    t0 = time.time()

    for oos_idx in range(oos_start, len(all_dates)):
        oos_date = all_dates[oos_idx]

        # IS window: last N_IS_DAYS before purge gap
        is_end_idx = oos_idx - PURGE_GAP
        is_start_idx = max(0, is_end_idx - N_IS_DAYS)
        is_dates = all_dates[is_start_idx:is_end_idx]

        if len(is_dates) < MIN_IS_DAYS:
            log.warning(f"Skipping {oos_date}: only {len(is_dates)} IS dates")
            continue

        log.info(f"\nFold {len(oos_predictions)+1}: OOS={oos_date}, "
                 f"IS={is_dates[0]}..{is_dates[-1]} ({len(is_dates)}d)")

        # Load IS data
        is_features_list = []
        is_targets_list = []

        for is_date in is_dates:
            result = load_feature_day(is_date)
            if result is None:
                continue
            feat, mid = result
            targets = compute_rl_reward_targets(mid)
            is_features_list.append(feat[:, feature_cols])
            is_targets_list.append(targets)

        if not is_features_list:
            continue

        X_train = np.vstack(is_features_list).astype(np.float32)
        y_train = np.concatenate(is_targets_list).astype(np.float32)

        # Remove zero-target bars (end of day padding)
        mask = y_train != 0
        X_train = X_train[mask]
        y_train = y_train[mask]

        if len(X_train) < 1000:
            continue

        # Subsample for speed (every 3rd sample)
        subsample_idx = np.arange(0, len(X_train), 3)
        X_train_sub = X_train[subsample_idx]
        y_train_sub = y_train[subsample_idx]

        # Train LightGBM
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train_sub, y_train_sub)

        # Load OOS data and predict
        result = load_feature_day(oos_date)
        if result is None:
            continue

        oos_feat, oos_mid = result
        X_oos = oos_feat[:, feature_cols].astype(np.float32)
        preds = model.predict(X_oos).astype(np.float64)

        oos_predictions[oos_date] = preds
        oos_mid_prices[oos_date] = oos_mid
        oos_spreads[oos_date] = oos_feat[:, 1].copy()  # spread is column 1

        # Quick IC check
        from scipy.stats import spearmanr
        fwd_ret = np.zeros(len(oos_mid))
        for i in range(len(oos_mid) - HORIZON_30S):
            fwd_ret[i] = (oos_mid[i + HORIZON_30S] - oos_mid[i]) / TICK
        valid = (fwd_ret != 0) & np.isfinite(preds)
        if valid.sum() > 100:
            ic = spearmanr(preds[valid], fwd_ret[valid])[0]
            log.info(f"  Predictions: range=[{preds.min():.3f}, {preds.max():.3f}], "
                     f"std={preds.std():.4f}, IC={ic:+.4f}")
        else:
            log.info(f"  Predictions: range=[{preds.min():.3f}, {preds.max():.3f}]")

        del is_features_list, is_targets_list, X_train, y_train, model
        gc.collect()

    train_time = time.time() - t0
    log.info(f"\nTraining complete: {len(oos_predictions)} OOS days in {train_time:.0f}s")

    if not oos_predictions:
        log.error("No predictions generated!")
        return

    # ========================================================
    # PHASE 2: Market order simulation sweep
    # ========================================================
    log.info(f"\n{'='*70}")
    log.info("PHASE 2: Market Order Simulation Sweep")
    log.info(f"{'='*70}")

    total_combos = len(THRESHOLDS) * len(HOLD_BARS) * len(TRAILS) * len(COOLDOWNS)
    n_oos = len(oos_predictions)
    log.info(f"Sweep: {total_combos} combos × {n_oos} days = {total_combos * n_oos} sims")

    t0 = time.time()
    all_results = []
    combo_count = 0

    for thresh in THRESHOLDS:
        for hold in HOLD_BARS:
            for trail in TRAILS:
                for cooldown in COOLDOWNS:
                    combo_count += 1
                    day_pnls = []
                    day_trades = []
                    day_wins = []

                    for date in sorted(oos_predictions.keys()):
                        pnl, tr, w = sim_day_market_order(
                            oos_mid_prices[date],
                            oos_spreads[date],
                            oos_predictions[date],
                            thresh, hold, trail, cooldown
                        )
                        day_pnls.append(pnl * TICK_VAL)  # convert to dollars
                        day_trades.append(tr)
                        day_wins.append(w)

                    pnl_arr = np.array(day_pnls)
                    total_pnl = float(pnl_arr.sum())
                    total_trades = sum(day_trades)
                    total_wins = sum(day_wins)
                    mean_daily = float(pnl_arr.mean())
                    std_daily = float(pnl_arr.std())
                    sharpe = float(mean_daily / max(std_daily, 0.01) * np.sqrt(252))
                    win_rate = total_wins / max(total_trades, 1)
                    profit_days = int((pnl_arr > 0).sum())

                    hold_sec = hold / BARS_PER_SEC
                    combo_id = f"t{thresh}_h{hold_sec:.0f}s_tr{trail}_cd{cooldown}"

                    all_results.append({
                        'combo_id': combo_id,
                        'threshold': thresh,
                        'hold_bars': hold,
                        'hold_sec': hold_sec,
                        'trailing': trail,
                        'cooldown': cooldown,
                        'total_pnl': round(total_pnl, 2),
                        'mean_daily_pnl': round(mean_daily, 2),
                        'std_daily_pnl': round(std_daily, 2),
                        'sharpe': round(sharpe, 2),
                        'total_trades': total_trades,
                        'win_rate': round(win_rate, 4),
                        'profitable_days': profit_days,
                        'total_days': n_oos,
                        'daily_pnls': [round(p, 2) for p in day_pnls],
                    })

                    if combo_count % 50 == 0:
                        log.info(f"  [{combo_count}/{total_combos}] combos tested...")

    sim_time = time.time() - t0
    log.info(f"Sim complete: {total_combos} combos in {sim_time:.1f}s")

    # ========================================================
    # PHASE 3: Results
    # ========================================================
    all_results.sort(key=lambda x: x['sharpe'], reverse=True)

    log.info(f"\n{'='*70}")
    log.info("RESULTS — rl_reward Market Order OOS Test")
    log.info(f"{'='*70}")
    log.info(f"OOS dates: {n_oos} ({sorted(oos_predictions.keys())[0]} to {sorted(oos_predictions.keys())[-1]})")
    log.info(f"Cost: {TOTAL_COST:.2f} ticks/RT")

    log.info(f"\nTOP 20 COMBOS BY SHARPE:")
    log.info(f"{'Combo':<30} {'Sharpe':>7} {'PnL':>10} {'$/day':>8} {'Trades':>7} {'Win%':>6} {'Days+':>6}")
    log.info("-" * 80)
    for r in all_results[:20]:
        marker = " ***" if r['total_pnl'] > 0 else ""
        log.info(f"{r['combo_id']:<30} {r['sharpe']:>+7.2f} ${r['total_pnl']:>+9,.0f} "
                 f"${r['mean_daily_pnl']:>+7,.0f} {r['total_trades']:>7} "
                 f"{r['win_rate']:>5.1%} {r['profitable_days']:>3}/{r['total_days']}{marker}")

    profitable = [r for r in all_results if r['total_pnl'] > 0]
    log.info(f"\nProfitable combos: {len(profitable)}/{len(all_results)} "
             f"({len(profitable)/len(all_results)*100:.1f}%)")

    if profitable:
        log.info("\nALL PROFITABLE COMBOS:")
        for r in sorted(profitable, key=lambda x: x['total_pnl'], reverse=True):
            log.info(f"  {r['combo_id']:<30} +${r['total_pnl']:>9,.0f}  "
                     f"Sharpe={r['sharpe']:+.2f}  {r['total_trades']} trades  "
                     f"{r['profitable_days']}/{r['total_days']} days")

    # WORST 5
    log.info(f"\nWORST 5 COMBOS:")
    for r in all_results[-5:]:
        log.info(f"  {r['combo_id']:<30} ${r['total_pnl']:>+9,.0f}  "
                 f"{r['total_trades']} trades")

    # Save results
    output = {
        'experiment': 'rl_reward_market_order_oos',
        'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'n_is_days': N_IS_DAYS,
        'n_oos_days': n_oos,
        'oos_range': f"{sorted(oos_predictions.keys())[0]} to {sorted(oos_predictions.keys())[-1]}",
        'cost_ticks_rt': TOTAL_COST,
        'training_time_secs': round(train_time, 1),
        'sim_time_secs': round(sim_time, 1),
        'results': all_results,
    }

    out_file = RESULTS_DIR / f'rl_reward_mktorder_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved: {out_file}")


if __name__ == '__main__':
    main()
