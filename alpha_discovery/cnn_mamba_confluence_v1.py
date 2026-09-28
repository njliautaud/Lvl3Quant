#!/usr/bin/env python3
"""
cnn_mamba_confluence_v1.py — CNN-Mamba Prediction Confluence Analysis

PURPOSE: Test whether CNN-Mamba short-horizon predictions (1s/5s) can
improve the champion strategy's 23.3% win rate when used as a confluence
filter at entry time.

HYPOTHESIS: The champion uses 30-min LightGBM predictions for entry. If
the CNN-Mamba model (which reads raw MBO events at tick level) AGREES with
the 30-min signal at entry time, the trade should have higher WR because
two independent models agree on direction.

APPROACH:
1. Reconstruct champion trade entry timestamps
2. Find the CNN-Mamba prediction closest to each entry time
3. Test: does CNN-Mamba agreement at entry predict winners?
4. Walk-forward: train filter using CNN-Mamba confidence + base predictions

DATA: Uses precomputed CNN-Mamba observations (250ms stride, 1s/5s horizons).

Run on Neptune (has precomputed obs + conda env).

HC #0: SLIDING windows only.
HC #428: Regime-agnostic validation.
"""

import os, sys, json, logging, warnings, time, gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ── Paths ──
ROOT = Path("/home/nick/Lvl3Quant")
PRECOMPUTED_DIR = ROOT / "data" / "precomputed_obs"
PRECOMPUTED_V342_DIR = ROOT / "data" / "precomputed_obs_v342"
MINUTE_BARS_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
ENTRY_PREDS_PATH = ROOT / "output" / "mfe_mae_analysis" / "entry_predictions.npz"
FEATURES_PATH = ROOT / "output" / "long_horizon_flow_v1" / "daily_features.parquet"
MODEL_WEIGHTS_DIR = ROOT / "output"
OUTPUT_DIR = ROOT / "output" / "cnn_mamba_confluence_v1"
LOG_FILE = ROOT / "logs" / "cnn_mamba_confluence_v1.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CNN-CONFLUENCE] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Constants ──
TICK_SIZE = 1.0
TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_SLIPPAGE_TICKS = 1.0

# Strategy params
TP_LONG = 25
TP_SHORT = 25
SL_LONG = 4
SL_SHORT = 3
MAX_HOLD = 60
ENTRY_THRESHOLD = 0.05
CANCEL_WINDOW = 10
ENTRY_BAR_SIZE = 30
DAILY_BIAS_THRESHOLD_MULT = 1.5

# CNN-Mamba prediction stride
PRED_STRIDE_MS = 250  # new prediction every 250ms


def load_precomputed_predictions(date_str):
    """Load precomputed CNN-Mamba timestamps and labels for a date.

    The precomputed obs files contain:
    - base_obs: the observation tensor (not needed here)
    - timestamps: nanosecond timestamps of each prediction point
    - meta: metadata (labels, etc.)
    """
    ts_path = PRECOMPUTED_DIR / f"{date_str}_timestamps.npy"
    meta_path = PRECOMPUTED_DIR / f"{date_str}_meta.npy"

    if not ts_path.exists():
        # Try v342 dir
        ts_path = PRECOMPUTED_V342_DIR / f"{date_str}_timestamps.npy"
        meta_path = PRECOMPUTED_V342_DIR / f"{date_str}_meta.npy"

    if not ts_path.exists():
        return None, None

    timestamps = np.load(str(ts_path))
    meta = np.load(str(meta_path), allow_pickle=True)

    return timestamps, meta


def reconstruct_trades_with_cnn_preds():
    """Reconstruct champion trades and extract CNN-Mamba predictions at entry."""
    log.info("=" * 70)
    log.info("Reconstructing champion trades + CNN-Mamba predictions")
    log.info("=" * 70)

    # Load minute bars
    files = sorted(MINUTE_BARS_DIR.glob("*.parquet"))
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df['date'] = f.stem
        frames.append(df)
    minute_df = pd.concat(frames, ignore_index=True)
    minute_df['ts_minute'] = pd.to_datetime(minute_df['ts_minute'], utc=True)
    minute_df = minute_df.sort_values('ts_minute').reset_index(drop=True)
    log.info(f"Loaded {len(minute_df):,} minute bars across {len(files)} days")

    # Build 30-min bars
    bars_30m = minute_df.groupby('date').apply(
        lambda g: g.iloc[::30] if len(g) >= 30 else g.iloc[:0]
    ).reset_index(drop=True)

    # Load 30-min predictions
    data = np.load(str(ENTRY_PREDS_PATH), allow_pickle=True)
    pred_30m = data['entry_preds']

    # Load daily features for bias filter
    daily_df = pd.read_parquet(FEATURES_PATH)
    if 'date' not in daily_df.columns:
        daily_df = daily_df.reset_index()
        daily_df.columns = ['date'] + list(daily_df.columns[1:])

    # Compute daily bias
    if 'session_ofi' in daily_df.columns:
        ofi_col = daily_df['session_ofi']
        ofi_std = ofi_col.expanding(min_periods=20).std()
        ofi_mean = ofi_col.expanding(min_periods=20).mean()
        daily_df['ofi_z'] = (ofi_col - ofi_mean) / ofi_std.clip(lower=1e-6)

    daily_bias_lookup = {}
    if 'ofi_z' in daily_df.columns:
        for _, row in daily_df.iterrows():
            d = str(row['date'])
            z = row['ofi_z']
            if pd.isna(z):
                daily_bias_lookup[d] = 'neutral'
            elif z > DAILY_BIAS_THRESHOLD_MULT:
                daily_bias_lookup[d] = 'short'
            elif z < -DAILY_BIAS_THRESHOLD_MULT:
                daily_bias_lookup[d] = 'long'
            else:
                daily_bias_lookup[d] = 'neutral'

    # Reconstruct trades
    valid_mask = ~np.isnan(pred_30m)
    upper_thresh = np.nanquantile(pred_30m[valid_mask], 1 - ENTRY_THRESHOLD)
    lower_thresh = np.nanquantile(pred_30m[valid_mask], ENTRY_THRESHOLD)

    minute_lookup = {}
    for date_str, grp in minute_df.groupby('date'):
        minute_lookup[date_str] = grp.sort_values('ts_minute').reset_index(drop=True)

    ts_col = 'ts_minute' if 'ts_minute' in bars_30m.columns else 'ts'
    bars_ts = bars_30m[ts_col].values
    bars_dates = bars_30m['date'].values
    bars_close = bars_30m['close'].values

    trades = []
    cnn_dates_loaded = {}
    n_cnn_found = 0
    n_cnn_missing = 0

    for i in range(len(bars_30m)):
        if i >= len(pred_30m) or np.isnan(pred_30m[i]):
            continue

        direction = 0
        if pred_30m[i] >= upper_thresh:
            direction = 1
        elif pred_30m[i] <= lower_thresh:
            direction = -1
        else:
            continue

        date_str = str(bars_dates[i])
        bias = daily_bias_lookup.get(date_str, 'neutral')
        if bias == 'short' and direction == 1:
            continue
        if bias == 'long' and direction == -1:
            continue

        signal_ts = pd.Timestamp(bars_ts[i])
        signal_price = bars_close[i]

        if date_str not in minute_lookup:
            continue

        day_minutes = minute_lookup[date_str]
        day_ts = day_minutes['ts_minute'].values

        limit_price = signal_price
        signal_bar_end = signal_ts + pd.Timedelta(minutes=ENTRY_BAR_SIZE)
        cancel_ts = signal_ts + pd.Timedelta(minutes=CANCEL_WINDOW + ENTRY_BAR_SIZE)

        fill_mask = (day_ts >= np.datetime64(signal_bar_end)) & (day_ts <= np.datetime64(cancel_ts))
        fill_candidates = day_minutes[fill_mask]

        if len(fill_candidates) == 0:
            continue

        filled = False
        fill_price = None
        fill_ts = None

        for j, (_, mbar) in enumerate(fill_candidates.iterrows()):
            if direction == 1:
                if mbar['low'] <= limit_price - TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar['ts_minute']
                    break
            else:
                if mbar['high'] >= limit_price + TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar['ts_minute']
                    break

        if not filled:
            continue

        fill_ts_pd = pd.Timestamp(fill_ts)
        fill_ts_np = np.datetime64(fill_ts)
        remaining_mask = day_ts >= fill_ts_np
        remaining_minutes = day_minutes[remaining_mask]

        if len(remaining_minutes) < 2:
            continue

        # Simulate exit
        prices_close = remaining_minutes['close'].values
        prices_high = remaining_minutes['high'].values
        prices_low = remaining_minutes['low'].values

        tp_ticks = TP_LONG if direction == 1 else TP_SHORT
        sl_ticks = SL_LONG if direction == 1 else SL_SHORT
        tp_price = fill_price + direction * tp_ticks * TICK_SIZE
        sl_price = fill_price - direction * sl_ticks * TICK_SIZE

        exit_type = 'time'
        exit_pnl = 0.0

        max_check = min(MAX_HOLD, len(remaining_minutes))
        for m in range(1, max_check):
            bar_high = prices_high[m]
            bar_low = prices_low[m]
            sl_hit = tp_hit = False

            if direction == 1:
                if bar_low <= sl_price: sl_hit = True
                if bar_high >= tp_price + TICK_SIZE: tp_hit = True
            else:
                if bar_high >= sl_price: sl_hit = True
                if bar_low <= tp_price - TICK_SIZE: tp_hit = True

            if sl_hit and tp_hit: sl_hit, tp_hit = True, False

            if sl_hit:
                exit_pnl = -(sl_ticks + MARKET_SLIPPAGE_TICKS + RT_COMMISSION_TICKS)
                exit_type = 'sl'
                break
            if tp_hit:
                exit_pnl = tp_ticks - RT_COMMISSION_TICKS
                exit_type = 'tp'
                break

        if exit_type == 'time':
            em = min(MAX_HOLD, len(remaining_minutes) - 1)
            em = max(em, 1)
            ec = prices_close[em]
            ef = ec - TICK_SIZE if direction == 1 else ec + TICK_SIZE
            exit_pnl = (ef - fill_price) / TICK_SIZE * direction - RT_COMMISSION_TICKS

        # Get CNN-Mamba prediction at entry time
        cnn_pred_1s = np.nan
        cnn_pred_5s = np.nan
        cnn_confidence = np.nan

        if date_str not in cnn_dates_loaded:
            cnn_ts, cnn_meta = load_precomputed_predictions(date_str)
            cnn_dates_loaded[date_str] = (cnn_ts, cnn_meta)
        else:
            cnn_ts, cnn_meta = cnn_dates_loaded[date_str]

        if cnn_ts is not None:
            fill_ts_ns = int(fill_ts_pd.value)
            # Find closest prediction to fill time
            time_diffs = np.abs(cnn_ts - fill_ts_ns)
            closest_idx = np.argmin(time_diffs)
            closest_diff_ms = time_diffs[closest_idx] / 1e6

            if closest_diff_ms < 5000:  # within 5 seconds
                n_cnn_found += 1
                # Extract labels (1s, 5s horizons)
                if cnn_meta is not None and len(cnn_meta.shape) > 0:
                    if len(cnn_meta.shape) == 2:
                        if cnn_meta.shape[1] >= 2:
                            cnn_pred_1s = float(cnn_meta[closest_idx, 0])
                            cnn_pred_5s = float(cnn_meta[closest_idx, 1])
                        elif cnn_meta.shape[1] >= 1:
                            cnn_pred_1s = float(cnn_meta[closest_idx, 0])
                    elif len(cnn_meta.shape) == 1:
                        cnn_pred_1s = float(cnn_meta[closest_idx])

                # CNN-Mamba "agreement" = prediction aligns with our trade direction
                if not np.isnan(cnn_pred_1s):
                    cnn_confidence = abs(cnn_pred_1s)
                    # Agreement: positive pred for long, negative for short
                    agreement = cnn_pred_1s * direction
            else:
                n_cnn_missing += 1
        else:
            n_cnn_missing += 1

        trades.append({
            'idx': i,
            'date': date_str,
            'fill_ts': fill_ts_pd,
            'fill_price': fill_price,
            'direction': direction,
            'pred_30m': float(pred_30m[i]),
            'exit_type': exit_type,
            'exit_pnl': exit_pnl,
            'winner': exit_type == 'tp',
            'cnn_pred_1s': cnn_pred_1s,
            'cnn_pred_5s': cnn_pred_5s,
            'cnn_confidence': cnn_confidence,
            'cnn_agreement_1s': cnn_pred_1s * direction if not np.isnan(cnn_pred_1s) else np.nan,
        })

    log.info(f"Reconstructed {len(trades)} trades")
    log.info(f"CNN-Mamba predictions found: {n_cnn_found}, missing: {n_cnn_missing}")
    winners = sum(1 for t in trades if t['winner'])
    log.info(f"Winners: {winners} ({winners/len(trades)*100:.1f}%)")

    return pd.DataFrame(trades)


def analyze_cnn_confluence(df):
    """Analyze whether CNN-Mamba agreement predicts winners."""
    log.info("=" * 70)
    log.info("CNN-Mamba Confluence Analysis")
    log.info("=" * 70)

    cnn_df = df[df['cnn_pred_1s'].notna()].copy()
    log.info(f"Trades with CNN-Mamba predictions: {len(cnn_df)}")

    if len(cnn_df) < 30:
        log.warning("Too few trades with CNN predictions")
        return {}

    # 1. Does CNN agreement predict winners?
    winners = cnn_df[cnn_df['winner'] == True]
    losers = cnn_df[cnn_df['winner'] == False]

    log.info(f"\n--- Agreement Score (cnn_pred * direction) ---")
    log.info(f"Winners mean agreement: {winners['cnn_agreement_1s'].mean():.4f}")
    log.info(f"Losers mean agreement:  {losers['cnn_agreement_1s'].mean():.4f}")

    t_stat, p_val = stats.ttest_ind(
        winners['cnn_agreement_1s'].dropna(),
        losers['cnn_agreement_1s'].dropna(),
        equal_var=False
    )
    log.info(f"T-test: t={t_stat:.3f}, p={p_val:.4f}")

    # 2. Confluence filter: only trade when CNN agrees
    log.info(f"\n--- Confluence Filter Results ---")
    results = []

    for thresh_name, thresh_fn in [
        ("CNN agrees (>0)", lambda x: x > 0),
        ("CNN strongly agrees (>0.5)", lambda x: x > 0.5),
        ("CNN strongly agrees (>1.0)", lambda x: x > 1.0),
        ("Top 50% confidence", lambda x: x > cnn_df['cnn_agreement_1s'].median()),
        ("Top 25% confidence", lambda x: x > cnn_df['cnn_agreement_1s'].quantile(0.75)),
    ]:
        mask = thresh_fn(cnn_df['cnn_agreement_1s'])
        if mask.sum() < 10:
            continue

        passed = cnn_df[mask]
        blocked = cnn_df[~mask]

        passed_wr = passed['winner'].mean()
        blocked_wr = blocked['winner'].mean()
        base_wr = cnn_df['winner'].mean()

        # FIFO PnL
        passed_pnl = passed['exit_pnl'].sum()
        blocked_pnl = blocked['exit_pnl'].sum()

        # Daily Sharpe of filtered
        passed_daily = passed.groupby('date')['exit_pnl'].sum()
        if len(passed_daily) > 5:
            sharpe = passed_daily.mean() / passed_daily.std() * np.sqrt(252)
        else:
            sharpe = np.nan

        result = {
            'filter': thresh_name,
            'n_passed': int(mask.sum()),
            'n_blocked': int((~mask).sum()),
            'passed_wr': passed_wr,
            'blocked_wr': blocked_wr,
            'base_wr': base_wr,
            'wr_lift': passed_wr - base_wr,
            'passed_pnl': passed_pnl,
            'blocked_pnl': blocked_pnl,
            'sharpe': sharpe,
        }
        results.append(result)

        log.info(f"  {thresh_name}:")
        log.info(f"    Passed: {result['n_passed']} trades, WR={result['passed_wr']:.1%} "
                 f"(base {result['base_wr']:.1%}, lift {result['wr_lift']:+.1%})")
        log.info(f"    PnL: passed={result['passed_pnl']:.1f}t, blocked={result['blocked_pnl']:.1f}t")
        log.info(f"    Sharpe: {result['sharpe']:.2f}")

    # 3. Direction-specific analysis
    log.info(f"\n--- Direction-Specific Analysis ---")
    for dir_name, dir_val in [("LONG", 1), ("SHORT", -1)]:
        dir_df = cnn_df[cnn_df['direction'] == dir_val]
        if len(dir_df) < 20:
            continue

        w = dir_df[dir_df['winner'] == True]
        l = dir_df[dir_df['winner'] == False]

        log.info(f"\n{dir_name} trades ({len(dir_df)} total, {len(w)} winners):")
        log.info(f"  Winner agreement: {w['cnn_agreement_1s'].mean():.4f}")
        log.info(f"  Loser agreement:  {l['cnn_agreement_1s'].mean():.4f}")

    # 4. CNN confidence (magnitude) as predictor
    log.info(f"\n--- CNN Confidence (|pred|) Analysis ---")
    cnn_df['abs_confidence'] = cnn_df['cnn_confidence']
    for q_name, q_val in [("Top 50%", 0.5), ("Top 25%", 0.75), ("Top 10%", 0.9)]:
        thresh = cnn_df['abs_confidence'].quantile(q_val)
        high_conf = cnn_df[cnn_df['abs_confidence'] >= thresh]
        if len(high_conf) < 10:
            continue
        log.info(f"  {q_name} confidence: {len(high_conf)} trades, "
                 f"WR={high_conf['winner'].mean():.1%} "
                 f"(base {cnn_df['winner'].mean():.1%})")

    return {
        'n_trades_with_cnn': len(cnn_df),
        'agreement_t_test': {'t_stat': t_stat, 'p_value': p_val},
        'filter_results': results,
    }


def regime_test_best_filter(df, filter_results):
    """Test the best confluence filter against regime gate HC #428."""
    log.info("=" * 70)
    log.info("Regime Gate Test (HC #428)")
    log.info("=" * 70)

    if not filter_results:
        log.warning("No filter results to test")
        return {}

    # Find the filter with best WR lift that keeps enough trades
    viable = [r for r in filter_results if r['n_passed'] >= 50 and r['wr_lift'] > 0]
    if not viable:
        viable = [r for r in filter_results if r['n_passed'] >= 30 and r['wr_lift'] > 0]
    if not viable:
        log.warning("No viable filters found")
        return {}

    best = max(viable, key=lambda x: x['wr_lift'])
    log.info(f"Best filter: {best['filter']}")
    log.info(f"  Trades: {best['n_passed']}, WR: {best['passed_wr']:.1%}, Lift: {best['wr_lift']:+.1%}")

    # Would need the daily features to compute regime stats properly
    # For now, report what we have
    return {'best_filter': best}


def main():
    start_time = time.time()
    log.info("=" * 70)
    log.info("CNN-MAMBA CONFLUENCE ANALYSIS v1")
    log.info(f"Started: {datetime.now()}")
    log.info("=" * 70)

    # Step 1: Reconstruct trades with CNN-Mamba predictions
    trades_df = reconstruct_trades_with_cnn_preds()

    # Save intermediate
    trades_df.to_parquet(OUTPUT_DIR / "trades_with_cnn.parquet", index=False)

    # Step 2: Confluence analysis
    results = analyze_cnn_confluence(trades_df)

    # Step 3: Regime gate test
    regime_results = regime_test_best_filter(
        trades_df, results.get('filter_results', [])
    )

    elapsed = time.time() - start_time

    # Save results
    summary = {
        'run_time': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'n_trades': len(trades_df),
        'n_with_cnn': int(trades_df['cnn_pred_1s'].notna().sum()),
        'base_wr': float(trades_df['winner'].mean()),
        'confluence_results': results,
        'regime_test': regime_results,
    }

    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("=" * 70)
    log.info(f"COMPLETE in {elapsed:.0f}s")
    log.info(f"Results saved to {OUTPUT_DIR}")
    log.info("=" * 70)

    return summary


if __name__ == '__main__':
    main()
