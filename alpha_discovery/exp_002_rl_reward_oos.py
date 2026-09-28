"""
EXP_002: rl_reward Target Through Rust MBO Sim
================================================
rl_reward achieved IS Sharpe +1.02 (BEST of all targets) but has NEVER
been run through the Rust MBO fill simulator.

rl_reward trains 3 LightGBM models (buy_q, sell_q, hold_q) as Q-values:
- buy_q: expected reward if we place a buy order now
- sell_q: expected reward if we place a sell order now
- hold_q: expected reward of doing nothing

Entry signal = max(buy_q, sell_q, hold_q) with sign:
  +signal if max is buy_q
  -signal if max is sell_q
  0 if max is hold_q (no trade)

Walk-forward training on IS feature cache, generates OOS predictions,
then sweeps through Rust sim.

Usage:
    python alpha_discovery/exp_002_rl_reward_oos.py
    python alpha_discovery/exp_002_rl_reward_oos.py --n-is-days 48 --workers 4
"""

import gc
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'rl_reward_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'rl_reward_sim'
PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)
SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'mbo'
FEAT_CACHE_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RESULTS_DIR / f'exp002_rl_reward_{_ts}.log'), mode='w'),
    ]
)
log = logging.getLogger('exp002')

TICK_VALUE = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24t
BARS_PER_SEC = 10
HORIZON_30S = 300  # bars for 30s at 100ms

OOS_DATES = [
    '2025-09-19', '2025-09-22', '2025-09-23', '2025-09-24', '2025-09-25', '2025-09-26',
    '2025-09-29', '2025-09-30', '2025-10-01', '2025-10-02', '2025-10-03',
    '2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09', '2025-10-10',
    '2025-10-13', '2025-10-14', '2025-10-15', '2025-10-16', '2025-10-17',
]

# Sweep params for Rust sim
SIM_THRESHOLDS = [0.3, 0.5, 0.7]
SIM_HOLD_MS = [10000, 30000, 60000]
SIM_TRAILING = [4, 8, 12]
SIM_LATENCIES = [0, 10]


def compute_rl_reward_targets(mid_prices: np.ndarray, day_boundaries: List[int],
                               horizon_bars: int = HORIZON_30S) -> np.ndarray:
    """
    Compute rl_reward Q-value targets for buy, sell, and hold actions.

    For each bar t:
      buy_q[t]  = net PnL ticks from buying at t, holding horizon_bars, then selling
      sell_q[t] = net PnL ticks from selling at t, holding horizon_bars, then buying back
      hold_q[t] = 0 (doing nothing has zero expected reward)

    With costs:
      Limit entry earns 0.5t spread, costs 0.24t commission = net +0.26t on fill
      Market exit costs 0.5t spread + 0.24t commission = -0.74t
      Net cost per RT: +0.26 - 0.74 = approximately -0.24t if MFE = 0
      Minimum edge needed per RT: ~0.5t to be profitable

    Returns combined signal: buy_q - sell_q (positive = lean long, negative = lean short)
    This provides a direction signal usable by the Rust sim with a threshold.
    """
    N = len(mid_prices)
    buy_q = np.zeros(N, dtype=np.float32)
    sell_q = np.zeros(N, dtype=np.float32)

    for d in range(len(day_boundaries) - 1):
        ds = day_boundaries[d]
        de = day_boundaries[d + 1]
        day_mid = mid_prices[ds:de]
        n_day = len(day_mid)

        for i in range(n_day - horizon_bars):
            exit_price = day_mid[i + horizon_bars]
            entry_price = day_mid[i]

            # Limit order entry: posted at bid/ask (earn ~0.5t half-spread)
            # Market exit: cross spread (lose 0.5t) + commission
            buy_return_ticks = (exit_price - entry_price) / 0.25
            sell_return_ticks = (entry_price - exit_price) / 0.25

            # Add limit entry edge (0.5t for passive fill), subtract exit cost
            entry_edge = 0.5   # earn half-spread on passive fill
            exit_cost = 0.5    # lose half-spread on market exit
            total_comm = COMMISSION_TICKS  # 0.24t commission

            buy_q[ds + i] = buy_return_ticks + entry_edge - exit_cost - total_comm
            sell_q[ds + i] = sell_return_ticks + entry_edge - exit_cost - total_comm

    # Combined signal: long if buy_q > sell_q, short if sell_q > buy_q
    # Scale to be comparable to direction predictions
    combined = buy_q - sell_q  # range roughly [-4, +4] for typical 30s moves

    # Normalize to roughly [-1, +1] range for consistency with Rust sim thresholds
    std = np.std(combined[combined != 0])
    if std > 0:
        combined = combined / (4 * std)  # 4-sigma = ~1.0 signal

    return combined


def load_feature_day(date_str: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load features and mid prices for a single date."""
    feat_file = FEAT_CACHE_DIR / f'{date_str}_mbo_features.npz'
    if not feat_file.exists():
        return None
    data = np.load(feat_file)
    features = data['mbo_features']
    # Mid price is the first column (index 0) per mbo_features.py naming
    mid_prices = features[:, 0].astype(np.float64)
    return features, mid_prices


def run_walk_forward_rl_reward(n_is_days: int = 48):
    """Walk-forward train rl_reward model and generate OOS predictions."""
    import lightgbm as lgb

    log.info("Starting walk-forward rl_reward training...")

    # Get all available feature cache files sorted by date
    all_feat_files = sorted(FEAT_CACHE_DIR.glob('*_mbo_features.npz'))
    all_dates = [f.stem.replace('_mbo_features', '') for f in all_feat_files]
    log.info(f"Available dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")

    # Find OOS start index
    oos_start_idx = None
    for i, d in enumerate(all_dates):
        if d >= '2025-09-19':
            oos_start_idx = i
            break

    if oos_start_idx is None:
        log.error("Could not find OOS start date in feature cache")
        return None

    log.info(f"OOS starts at index {oos_start_idx} ({all_dates[oos_start_idx]})")

    oos_predictions = {}

    for oos_idx in range(oos_start_idx, min(oos_start_idx + len(OOS_DATES), len(all_dates))):
        oos_date = all_dates[oos_idx]
        if oos_date not in OOS_DATES:
            continue

        # IS = last n_is_days before OOS (with 1-day purge gap)
        is_end = oos_idx - 1  # purge gap
        is_start = max(0, is_end - n_is_days)
        is_dates = all_dates[is_start:is_end]

        if len(is_dates) < 5:
            log.warning(f"Skipping {oos_date}: only {len(is_dates)} IS dates")
            continue

        log.info(f"Training for OOS {oos_date} using {len(is_dates)} IS dates "
                 f"({is_dates[0]} to {is_dates[-1]})")

        # Load IS data
        is_features_list = []
        is_targets_list = []
        is_day_boundaries = [0]

        for is_date in is_dates:
            result = load_feature_day(is_date)
            if result is None:
                continue
            feat, mid = result
            targets = compute_rl_reward_targets(mid, [0, len(mid)])
            is_features_list.append(feat)
            is_targets_list.append(targets)
            is_day_boundaries.append(is_day_boundaries[-1] + len(feat))

        if not is_features_list:
            continue

        is_features = np.vstack(is_features_list).astype(np.float32)
        is_targets = np.concatenate(is_targets_list).astype(np.float32)

        # Remove zero-target bars (end of day, horizon beyond day boundary)
        mask = is_targets != 0
        is_features_clean = is_features[mask]
        is_targets_clean = is_targets[mask]

        if len(is_features_clean) < 1000:
            log.warning(f"  Too few training samples: {len(is_features_clean)}")
            continue

        log.info(f"  Training on {len(is_features_clean):,} samples...")

        # Exclude leaky features (same as direction model)
        EXCLUDE_COLS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # raw price/vol columns
        feature_cols = [i for i in range(is_features_clean.shape[1]) if i not in EXCLUDE_COLS]
        X_train = is_features_clean[:, feature_cols]

        # Train LightGBM regression
        params = {
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

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, is_targets_clean, verbose=False)

        # Load OOS data and predict
        result = load_feature_day(oos_date)
        if result is None:
            log.warning(f"  No OOS data for {oos_date}")
            continue

        oos_feat, oos_mid = result
        X_oos = oos_feat[:, feature_cols].astype(np.float32)
        oos_preds = model.predict(X_oos).astype(np.float64)

        oos_predictions[oos_date] = oos_preds
        log.info(f"  Generated predictions for {oos_date}: "
                 f"shape={oos_preds.shape}, "
                 f"range=[{oos_preds.min():.3f}, {oos_preds.max():.3f}], "
                 f"std={oos_preds.std():.4f}")

        del is_features_list, is_features, is_targets_list, is_targets
        del is_features_clean, X_train, model
        gc.collect()

    return oos_predictions


def save_predictions_for_sim(oos_predictions):
    """Save predictions in NPZ format expected by fill_sim_cli."""
    saved_files = {}
    for date_str, preds in oos_predictions.items():
        out_file = PRED_OUT_DIR / f'{date_str}_rl_reward_predictions.npz'
        np.savez_compressed(str(out_file), predictions=preds)
        saved_files[date_str] = out_file
        log.info(f"Saved predictions: {out_file}")
    return saved_files


def run_rust_sim_sweep(prediction_files, workers=6):
    """Run Rust sim sweep on rl_reward predictions."""
    jobs = []
    for thresh in SIM_THRESHOLDS:
        for lat in SIM_LATENCIES:
            for hold in SIM_HOLD_MS:
                for trail in SIM_TRAILING:
                    combo_id = f"rl_t{int(thresh*10)}_l{lat}_h{hold//1000}s_tr{trail}"
                    for date_str in OOS_DATES:
                        if date_str in prediction_files:
                            jobs.append((date_str, combo_id, thresh, lat, hold, trail,
                                        prediction_files[date_str]))

    log.info(f"Running Rust sim: {len(jobs)} jobs with {workers} workers")
    results = {}
    done = 0

    def run_one(date_str, combo_id, thresh, lat, hold, trail, pred_file):
        date_nodash = date_str.replace('-', '')
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_nodash}.mbo.dbn.zst'
        out_file = SIM_OUT_DIR / f'{combo_id}_{date_str}.json'
        if not mbo_file.exists():
            return None
        cmd = [
            str(BINARY), '--mbo-file', str(mbo_file),
            '--predictions', str(pred_file), '--output', str(out_file),
            '--hold-ms', str(hold), '--trailing-ticks', str(trail),
            '--signal-threshold', str(thresh), '--latency-ms', str(lat),
            '--quiet',
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                return None
            with open(out_file) as f:
                return (date_str, combo_id, json.load(f))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, *job): job for job in jobs}
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                if result:
                    date_str, combo_id, data = result
                    if combo_id not in results:
                        results[combo_id] = {}
                    results[combo_id][date_str] = data
            except Exception:
                pass

            if done % 100 == 0:
                log.info(f"  [{done}/{len(jobs)}] sim jobs completed")

    return results


def aggregate_and_report(sim_results):
    """Aggregate sim results and print report."""
    summary = []
    for thresh in SIM_THRESHOLDS:
        for lat in SIM_LATENCIES:
            for hold in SIM_HOLD_MS:
                for trail in SIM_TRAILING:
                    combo_id = f"rl_t{int(thresh*10)}_l{lat}_h{hold//1000}s_tr{trail}"
                    combo_data = sim_results.get(combo_id, {})
                    daily_pnls = [combo_data.get(d, {}).get('total_pnl_dollars', 0) for d in OOS_DATES]
                    dp = np.array(daily_pnls)
                    std = float(np.std(dp)) if len(dp) > 1 else 1.0
                    sharpe = float(np.mean(dp) / max(std, 0.01) * np.sqrt(252))
                    summary.append({
                        'combo_id': combo_id,
                        'threshold': thresh, 'latency_ms': lat,
                        'hold_ms': hold, 'trailing_ticks': trail,
                        'total_pnl': round(float(dp.sum()), 2),
                        'total_trades': sum(combo_data.get(d, {}).get('total_trades', 0) for d in OOS_DATES),
                        'profitable_days': int((dp > 0).sum()),
                        'sharpe': round(sharpe, 2),
                        'daily_pnls': [round(float(p), 2) for p in daily_pnls],
                    })

    summary.sort(key=lambda x: x['sharpe'], reverse=True)

    log.info(f"\n{'='*70}")
    log.info(f"EXP_002 RESULTS — rl_reward through Rust MBO Sim")
    log.info(f"{'='*70}")
    log.info(f"\nTOP 10 COMBOS:")
    for i, s in enumerate(summary[:10]):
        marker = ' ***PROFITABLE***' if s['total_pnl'] > 0 else ''
        log.info(f"  {i+1:2d}. {s['combo_id']:35s}  Sharpe={s['sharpe']:+6.2f}  "
                 f"PnL=${s['total_pnl']:>+9,.2f}  trades={s['total_trades']:4d}  "
                 f"days={s['profitable_days']}/21{marker}")

    profitable = [s for s in summary if s['total_pnl'] > 0]
    log.info(f"\nProfitable combos: {len(profitable)}/{len(summary)}")
    if profitable:
        log.info("PROFITABLE COMBOS:")
        for s in profitable:
            log.info(f"  {s['combo_id']}  +${s['total_pnl']:,.2f}  Sharpe={s['sharpe']:.2f}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-is-days', type=int, default=48)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training if predictions already exist')
    args = parser.parse_args()

    if not BINARY.exists():
        log.error(f"Rust binary not found: {BINARY}")
        exit(1)

    t_start = time.time()

    # Check for existing predictions
    existing_preds = {}
    if args.skip_training:
        for date_str in OOS_DATES:
            pred_file = PRED_OUT_DIR / f'{date_str}_rl_reward_predictions.npz'
            if pred_file.exists():
                existing_preds[date_str] = pred_file
        log.info(f"Found {len(existing_preds)} existing prediction files (skipping training)")

    if not existing_preds:
        # Run walk-forward training
        oos_predictions = run_walk_forward_rl_reward(n_is_days=args.n_is_days)
        if not oos_predictions:
            log.error("Training failed — no predictions generated")
            exit(1)

        prediction_files = save_predictions_for_sim(oos_predictions)
    else:
        prediction_files = existing_preds

    log.info(f"\nPredictions ready for {len(prediction_files)} OOS dates")

    # Run Rust sim sweep
    log.info("Starting Rust sim sweep...")
    sim_results = run_rust_sim_sweep(prediction_files, workers=args.workers)

    # Aggregate and report
    summary = aggregate_and_report(sim_results)

    # Save
    output = {
        'experiment': 'exp_002_rl_reward_oos',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'n_is_days': args.n_is_days,
        'elapsed_secs': round(time.time() - t_start, 1),
        'results': summary,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'exp002_rl_reward_{ts}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved: {out_file}")
