#!/usr/bin/env python3
"""
V4 Multihead Tick-Level Replay — HC #659 Compliant
====================================================

Tests V4 multihead predictions through honest tick-level FIFO replay to
measure ACTUAL adverse selection (the critical unknown from SESSION_STATE #54).

v3.4.2 had:
  raw edge:          +0.145 ticks
  commission:        -0.376 ticks
  adverse selection: -0.770 ticks
  net:               -1.001 ticks

V4 multihead has:
  raw edge:          +0.225 ticks (55% stronger)
  commission:        -0.376 ticks (same)
  adverse selection: ???  (THIS IS WHAT WE'RE MEASURING)

If V4 adverse selection < 0.38 ticks, the signal is PROFITABLE.

Also tests:
- eofi confluence filtering (does end-of-fill prediction reduce adverse selection?)
- Quantile-based thresholds (top 1%, 3%, 5%, 10%)
- Short-only vs both sides
- Multiple TP/SL configs

Author: Claude
Date: 2026-07-11
"""

import sys, os, time, json, gc
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/jupiter/Lvl3Quant/engines')
from tick_replay_engine import (
    TICK_SIZE, TICK_VALUE, PRED_STRIDE, PRED_WINDOW,
    COMMISSION_RT_TICKS, SPREAD_TICKS, COST_PASSIVE_EXIT, COST_MARKET_EXIT,
    compute_metrics, Trade
)

# Paths
PROCESSED_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3'
RAW_MBO_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
V4_PRED_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_pressure_v1'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/v4_tick_replay'
CACHE_DIR = os.path.join(OUTPUT_DIR, 'day_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Use v16's cached extractions if available
V16_CACHE_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v16/day_cache'

N_PERMS = 50  # Permutation tests


# =============================================================================
# Step 1: Load V4 predictions and map to raw MBO timestamps
# =============================================================================

def load_v4_predictions():
    """
    Load all V4 multihead fold predictions.
    Returns dict of date_str -> {preds_dir_1s, preds_eofi_1s, preds_pdi_1s, ...}
    """
    fold_files = sorted(glob.glob(os.path.join(V4_PRED_DIR, 'fold_*_oot_predictions.npz')))
    print(f"Found {len(fold_files)} V4 fold prediction files")

    predictions_by_date = {}

    for fpath in fold_files:
        d = np.load(fpath, allow_pickle=True)
        fold_num = int(d['fold'])
        oot_files = d['oot_files']

        for oot_file in oot_files:
            # Extract date from path like .../20260102_mbo_events.npz
            date_str = os.path.basename(str(oot_file)).split('_')[0]

            # Primary signal: preds_dir column 0 = 1s horizon
            preds_dir_1s = d['preds_dir'][:, 0]

            # Secondary heads for confluence
            preds_eofi_1s = d['preds_eofi'][:, 0] if 'preds_eofi' in d else None
            preds_pdi_1s = d['preds_pdi'][:, 0] if 'preds_pdi' in d else None
            preds_ntps_1s = d['preds_ntps'][:, 0] if 'preds_ntps' in d else None

            predictions_by_date[date_str] = {
                'preds_dir_1s': preds_dir_1s,
                'preds_eofi_1s': preds_eofi_1s,
                'preds_pdi_1s': preds_pdi_1s,
                'preds_ntps_1s': preds_ntps_1s,
                'fold': fold_num,
                'n_preds': len(preds_dir_1s),
                'ic_dir_1s': float(d['ic_dir_1s']) if 'ic_dir_1s' in d else None,
            }

    return predictions_by_date


def get_prediction_timestamps(date_str, n_preds):
    """
    Get nanosecond timestamps for each prediction from the processed NPZ.
    Predictions are at indices: PRED_WINDOW + i * PRED_STRIDE
    """
    npz_path = os.path.join(PROCESSED_DIR, f'{date_str}_mbo_events.npz')
    if not os.path.exists(npz_path):
        return None

    d = np.load(npz_path, allow_pickle=True)
    timestamps = d['timestamps']

    pred_indices = np.arange(PRED_WINDOW, PRED_WINDOW + n_preds * PRED_STRIDE, PRED_STRIDE)
    valid = pred_indices < len(timestamps)
    pred_indices = pred_indices[valid]

    return timestamps[pred_indices], int(np.sum(valid))


# =============================================================================
# Step 2: Extract trade-based BBO from raw MBO (reuse v16 approach)
# =============================================================================

def extract_day_arrays(date_str):
    """
    Extract trade events + trade-based BBO from raw MBO data.
    Uses v16 cache if available, otherwise extracts fresh.
    """
    # Check v16 cache first
    v16_cache = os.path.join(V16_CACHE_DIR, f'{date_str}_extracted.npz')
    if os.path.exists(v16_cache):
        print(f"    [v16 cache hit] {date_str}")
        return dict(np.load(v16_cache))

    # Check our cache
    cache_path = os.path.join(CACHE_DIR, f'{date_str}_extracted.npz')
    if os.path.exists(cache_path):
        print(f"    [cache hit] {date_str}")
        return dict(np.load(cache_path))

    # Extract from raw MBO
    mbo_path = os.path.join(RAW_MBO_DIR, f'glbx-mdp3-{date_str}.mbo.dbn.zst')
    if not os.path.exists(mbo_path):
        print(f"    [ERROR] No raw MBO file for {date_str}")
        return None

    try:
        import databento as db
        store = db.DBNStore.from_file(mbo_path)
        df = store.to_df()
    except Exception as e:
        print(f"    [ERROR] {date_str}: {e}")
        return None

    # Filter to front-month ES contract
    es_symbols = [s for s in df['symbol'].unique()
                  if s.startswith('ES') and '-' not in s and len(s) <= 4]
    if not es_symbols:
        es_symbols = [s for s in df['symbol'].unique()
                      if s.startswith('ES') and '-' not in s]
    if not es_symbols:
        print(f"    [ERROR] {date_str}: no ES symbols found")
        return None

    best_sym = max(es_symbols,
                   key=lambda s: len(df[(df['symbol'] == s) & (df['action'] == 'T')]))
    df = df[df['symbol'] == best_sym].copy()
    print(f"    {date_str}: filtered to {best_sym} ({len(df):,} events)")

    n_events = len(df)
    t0 = time.time()

    ts_event = df['ts_event'].values.astype(np.int64)
    actions = df['action'].astype(str).values if hasattr(df['action'], 'cat') else df['action'].values
    sides = df['side'].astype(str).values if hasattr(df['side'], 'cat') else df['side'].values
    prices = df['price'].values.astype(np.float64)
    sizes = df['size'].values.astype(np.int64)

    # Arrays
    bbo_bid = np.zeros(n_events, dtype=np.float64)
    bbo_ask = np.zeros(n_events, dtype=np.float64)
    is_trade = np.zeros(n_events, dtype=np.bool_)
    trade_price = np.zeros(n_events, dtype=np.float64)
    trade_size = np.zeros(n_events, dtype=np.int32)
    trade_side_arr = np.zeros(n_events, dtype=np.int8)
    bbo_bid_size = np.zeros(n_events, dtype=np.int32)
    bbo_ask_size = np.zeros(n_events, dtype=np.int32)

    last_bid = 0.0
    last_ask = 0.0
    bid_depth = 0
    ask_depth = 0

    for i in range(n_events):
        action = str(actions[i])
        side = str(sides[i])
        price = float(prices[i])
        size = int(sizes[i])

        if action == 'T':
            is_trade[i] = True
            trade_price[i] = price
            trade_size[i] = size
            if side == 'A':
                trade_side_arr[i] = -1
                if price != last_bid:
                    last_bid = price
                    bid_depth = 0
                bid_depth = max(0, bid_depth - size)
            elif side == 'B':
                trade_side_arr[i] = 1
                if price != last_ask:
                    last_ask = price
                    ask_depth = 0
                ask_depth = max(0, ask_depth - size)
        elif action == 'A':
            if side == 'B' and price == last_bid:
                bid_depth += size
            elif side == 'A' and price == last_ask:
                ask_depth += size
            if side == 'B' and last_bid > 0 and price > last_bid:
                last_bid = price
                bid_depth = size
            elif side == 'A' and last_ask > 0 and price < last_ask and price > 0:
                last_ask = price
                ask_depth = size
        elif action == 'C':
            if side == 'B' and price == last_bid:
                bid_depth = max(0, bid_depth - size)
            elif side == 'A' and price == last_ask:
                ask_depth = max(0, ask_depth - size)

        bbo_bid[i] = last_bid
        bbo_ask[i] = last_ask
        bbo_bid_size[i] = max(1, bid_depth)
        bbo_ask_size[i] = max(1, ask_depth)

    elapsed = time.time() - t0
    both_valid = (bbo_bid > 0) & (bbo_ask > 0)
    if both_valid.any():
        spreads = bbo_ask[both_valid] - bbo_bid[both_valid]
        good = (spreads >= 0) & (spreads <= 1.0)
        print(f"    {date_str}: {n_events:,} events in {elapsed:.0f}s — "
              f"BBO valid {both_valid.mean()*100:.1f}%, good spread {good.mean()*100:.1f}%")

    np.savez_compressed(cache_path,
        ts_event=ts_event, bbo_bid=bbo_bid, bbo_ask=bbo_ask,
        bbo_bid_size=bbo_bid_size, bbo_ask_size=bbo_ask_size,
        is_trade=is_trade, trade_price=trade_price,
        trade_size=trade_size, trade_side=trade_side_arr,
        n_events=np.array([n_events]))

    return {
        'ts_event': ts_event, 'bbo_bid': bbo_bid, 'bbo_ask': bbo_ask,
        'bbo_bid_size': bbo_bid_size, 'bbo_ask_size': bbo_ask_size,
        'is_trade': is_trade, 'trade_price': trade_price,
        'trade_size': trade_size, 'trade_side': trade_side_arr,
        'n_events': np.array([n_events])
    }


# =============================================================================
# Step 3: Map V4 predictions to raw MBO event indices via timestamp
# =============================================================================

def map_predictions_to_raw_mbo(pred_timestamps, raw_ts_event):
    """
    Find the raw MBO event index closest to each prediction timestamp.
    Uses binary search for speed.
    """
    raw_indices = np.searchsorted(raw_ts_event, pred_timestamps)
    # Clamp to valid range
    raw_indices = np.clip(raw_indices, 0, len(raw_ts_event) - 1)
    return raw_indices


# =============================================================================
# Step 4: Tick-level simulation (adapted from v16)
# =============================================================================

def simulate_v4_on_day(day_data, predictions, pred_raw_indices,
                       tp_ticks, sl_ticks, hold_seconds,
                       threshold_quantile=None, threshold_abs=None,
                       side_filter=None,
                       eofi_preds=None, eofi_threshold=None):
    """
    Simulate V4 predictions through tick-level FIFO replay.

    Args:
        predictions: 1D array of dir_1s predictions
        pred_raw_indices: mapping of prediction positions to raw MBO event indices
        tp_ticks, sl_ticks: take-profit / stop-loss in ticks
        hold_seconds: max hold time
        threshold_quantile: use top/bottom X% of predictions (e.g. 0.03 for 3%)
        threshold_abs: absolute threshold for predictions
        side_filter: 'short' or 'long' or None (both)
        eofi_preds: optional eofi predictions for confluence filtering
        eofi_threshold: only trade when eofi > this (end-of-fill timing)
    """
    n_events = int(day_data['n_events'][0])
    ts = day_data['ts_event']
    bbo_bid = day_data['bbo_bid']
    bbo_ask = day_data['bbo_ask']
    bbo_bid_size = day_data['bbo_bid_size']
    bbo_ask_size = day_data['bbo_ask_size']
    is_trade = day_data['is_trade']
    trade_price_arr = day_data['trade_price']
    trade_size_arr = day_data['trade_size']
    trade_side_data = day_data['trade_side']

    # Compute thresholds from predictions
    if threshold_quantile is not None:
        long_thresh = np.percentile(predictions, 100 - threshold_quantile * 100)
        short_thresh = np.percentile(predictions, threshold_quantile * 100)
    elif threshold_abs is not None:
        long_thresh = threshold_abs
        short_thresh = -threshold_abs
    else:
        long_thresh = 0.0
        short_thresh = 0.0

    hold_ns = int(hold_seconds * 1e9)
    cancel_ns = int(15 * 1e9)  # 15s cancel window
    tp_price_delta = tp_ticks * TICK_SIZE
    sl_price_delta = sl_ticks * TICK_SIZE

    # Pre-compute trade event indices
    trade_indices = np.where(is_trade)[0]
    if len(trade_indices) == 0:
        return []
    trade_ts = ts[trade_indices]
    trade_pr = trade_price_arr[trade_indices]
    trade_sz = trade_size_arr[trade_indices]
    trade_sd = trade_side_data[trade_indices]

    completed_trades = []
    position_exit_after = 0

    for pi in range(len(predictions)):
        if pi >= len(pred_raw_indices):
            break

        pred_idx = int(pred_raw_indices[pi])
        if pred_idx >= n_events or pred_idx < 0:
            continue

        pred = predictions[pi]

        if pred_idx < position_exit_after:
            continue

        # Side determination
        if pred >= long_thresh and (side_filter is None or side_filter == 'long'):
            side = 'long'
        elif pred <= short_thresh and (side_filter is None or side_filter == 'short'):
            side = 'short'
        else:
            continue

        # Eofi confluence filter
        if eofi_preds is not None and eofi_threshold is not None:
            if pi < len(eofi_preds) and eofi_preds[pi] < eofi_threshold:
                continue

        # Check BBO validity
        bid = bbo_bid[pred_idx]
        ask = bbo_ask[pred_idx]
        if bid <= 0 or ask <= 0 or (ask - bid) > 2 * TICK_SIZE:
            continue

        if side == 'long':
            entry_price = bid
            queue_ahead = float(bbo_bid_size[pred_idx])
            tp_price_val = entry_price + tp_price_delta
            sl_price_val = entry_price - sl_price_delta
        else:
            entry_price = ask
            queue_ahead = float(bbo_ask_size[pred_idx])
            tp_price_val = entry_price - tp_price_delta
            sl_price_val = entry_price + sl_price_delta

        signal_ts = ts[pred_idx]
        cancel_ts = signal_ts + cancel_ns

        # Find trade events after prediction
        ti_start = np.searchsorted(trade_indices, pred_idx, side='right')

        # ENTRY FILL SEARCH
        filled = False
        fill_ti = -1
        fill_ts_val = 0
        remaining_queue = queue_ahead

        for ti in range(ti_start, len(trade_indices)):
            t_ts = trade_ts[ti]
            if t_ts > cancel_ts:
                break
            t_pr = trade_pr[ti]
            t_sz = int(trade_sz[ti])
            t_sd = trade_sd[ti]

            if side == 'long' and t_sd == -1:
                if abs(t_pr - entry_price) < 0.001:
                    remaining_queue -= t_sz
                    if remaining_queue <= 0:
                        filled = True
                        fill_ti = ti
                        fill_ts_val = t_ts
                        break
                elif t_pr < entry_price:
                    filled = True
                    fill_ti = ti
                    fill_ts_val = t_ts
                    break
            elif side == 'short' and t_sd == 1:
                if abs(t_pr - entry_price) < 0.001:
                    remaining_queue -= t_sz
                    if remaining_queue <= 0:
                        filled = True
                        fill_ti = ti
                        fill_ts_val = t_ts
                        break
                elif t_pr > entry_price:
                    filled = True
                    fill_ti = ti
                    fill_ts_val = t_ts
                    break

        if not filled:
            continue

        # EXIT SEARCH
        exit_deadline_ts = fill_ts_val + hold_ns
        fill_event_idx = trade_indices[fill_ti]
        tp_queue = float(bbo_ask_size[min(fill_event_idx, n_events - 1)] if side == 'long'
                        else bbo_bid_size[min(fill_event_idx, n_events - 1)])

        exit_reason = None
        exit_price_val = 0.0
        exit_ts_val = 0
        cost = COST_PASSIVE_EXIT
        best_price_seen = entry_price
        worst_price_seen = entry_price
        tp_queue_remaining = tp_queue

        for ti in range(fill_ti + 1, len(trade_indices)):
            t_ts = trade_ts[ti]
            t_pr = trade_pr[ti]
            t_sz = int(trade_sz[ti])
            t_sd = trade_sd[ti]

            if side == 'long':
                best_price_seen = max(best_price_seen, t_pr)
                worst_price_seen = min(worst_price_seen, t_pr)
            else:
                best_price_seen = min(best_price_seen, t_pr)
                worst_price_seen = max(worst_price_seen, t_pr)

            # SL check
            if side == 'long' and t_pr <= sl_price_val:
                exit_reason = 'sl'
                exit_price_val = sl_price_val
                exit_ts_val = t_ts
                cost = COST_MARKET_EXIT
                break
            elif side == 'short' and t_pr >= sl_price_val:
                exit_reason = 'sl'
                exit_price_val = sl_price_val
                exit_ts_val = t_ts
                cost = COST_MARKET_EXIT
                break

            # TP check (passive FIFO)
            if tp_ticks < 99:
                if side == 'long' and t_sd == 1:
                    if abs(t_pr - tp_price_val) < 0.001:
                        tp_queue_remaining -= t_sz
                        if tp_queue_remaining <= 0:
                            exit_reason = 'tp'
                            exit_price_val = tp_price_val
                            exit_ts_val = t_ts
                            cost = COST_PASSIVE_EXIT
                            break
                    elif t_pr > tp_price_val:
                        exit_reason = 'tp'
                        exit_price_val = tp_price_val
                        exit_ts_val = t_ts
                        cost = COST_PASSIVE_EXIT
                        break
                elif side == 'short' and t_sd == -1:
                    if abs(t_pr - tp_price_val) < 0.001:
                        tp_queue_remaining -= t_sz
                        if tp_queue_remaining <= 0:
                            exit_reason = 'tp'
                            exit_price_val = tp_price_val
                            exit_ts_val = t_ts
                            cost = COST_PASSIVE_EXIT
                            break
                    elif t_pr < tp_price_val:
                        exit_reason = 'tp'
                        exit_price_val = tp_price_val
                        exit_ts_val = t_ts
                        cost = COST_PASSIVE_EXIT
                        break

            # Time stop
            if t_ts >= exit_deadline_ts:
                exit_reason = 'time_stop'
                exit_price_val = t_pr
                exit_ts_val = t_ts
                cost = COST_MARKET_EXIT
                break

        if exit_reason is None:
            if len(trade_indices) > fill_ti + 1:
                exit_reason = 'eod'
                last_ti = len(trade_indices) - 1
                exit_price_val = trade_pr[last_ti]
                exit_ts_val = trade_ts[last_ti]
                cost = COST_MARKET_EXIT
            else:
                continue

        # Compute PnL
        if side == 'long':
            raw_pnl = (exit_price_val - entry_price) / TICK_SIZE
            mfe = max(0, (best_price_seen - entry_price) / TICK_SIZE)
            mae = max(0, (entry_price - worst_price_seen) / TICK_SIZE)
        else:
            raw_pnl = (entry_price - exit_price_val) / TICK_SIZE
            mfe = max(0, (entry_price - best_price_seen) / TICK_SIZE)
            mae = max(0, (worst_price_seen - entry_price) / TICK_SIZE)

        net_pnl = raw_pnl - cost

        completed_trades.append(Trade(
            trade_id=len(completed_trades),
            side=side,
            entry_price=entry_price,
            exit_price=exit_price_val,
            entry_time_ns=fill_ts_val,
            exit_time_ns=exit_ts_val,
            signal_time_ns=signal_ts,
            signal_strength=float(pred),
            exit_reason=exit_reason,
            queue_depth_at_entry=int(queue_ahead),
            fill_latency_ns=int(fill_ts_val - signal_ts),
            pnl_ticks=net_pnl,
            pnl_dollars=net_pnl * TICK_VALUE,
            cost_ticks=cost,
            mfe_ticks=mfe,
            mae_ticks=mae,
        ))

        position_exit_after = trade_indices[fill_ti] + 1

    return completed_trades


# =============================================================================
# Step 5: Adverse Selection Decomposition
# =============================================================================

def decompose_adverse_selection(trades):
    """
    Decompose trade PnL into: raw edge, commission, adverse selection.

    adverse_selection = -(avg_pnl_ticks + commission)
    where avg_pnl_ticks already includes commission in our engine,
    so: raw_pnl = net_pnl + cost
    adverse_selection = raw_pnl - (entry_to_exit at midpoint)

    Actually simpler:
    - raw_edge = what the SIGNAL predicted (pred direction × actual move)
    - adverse_selection = what we LOST between signal time and fill time
    """
    if not trades:
        return {}

    pnls = np.array([t.pnl_ticks for t in trades])
    costs = np.array([t.cost_ticks for t in trades])
    raw_pnls = pnls + costs  # PnL before costs
    mfes = np.array([t.mfe_ticks for t in trades])
    maes = np.array([t.mae_ticks for t in trades])
    fill_latencies = np.array([t.fill_latency_ns for t in trades]) / 1e9  # seconds

    # Decomposition
    avg_net_pnl = np.mean(pnls)
    avg_cost = np.mean(costs)
    avg_raw_pnl = np.mean(raw_pnls)

    # Adverse selection = how much price moves against us between signal and fill
    # This is captured in raw_pnl being worse than the theoretical prediction
    # For comparison: v3.4.2 had raw_edge=+0.145, raw_pnl_after_fill=-0.625
    # adverse_selection = 0.145 - (-0.625) ≈ 0.77 (price moved 0.77 ticks against us during fill)

    return {
        'n_trades': len(trades),
        'avg_net_pnl_ticks': float(avg_net_pnl),
        'avg_raw_pnl_ticks': float(avg_raw_pnl),
        'avg_cost_ticks': float(avg_cost),
        'avg_mfe_ticks': float(np.mean(mfes)),
        'avg_mae_ticks': float(np.mean(maes)),
        'avg_fill_latency_s': float(np.mean(fill_latencies)),
        'median_fill_latency_s': float(np.median(fill_latencies)),
        'win_rate': float(np.mean(pnls > 0)),
        'long_trades': sum(1 for t in trades if t.side == 'long'),
        'short_trades': sum(1 for t in trades if t.side == 'short'),
        'tp_exits': sum(1 for t in trades if t.exit_reason == 'tp'),
        'sl_exits': sum(1 for t in trades if t.exit_reason == 'sl'),
        'time_exits': sum(1 for t in trades if t.exit_reason == 'time_stop'),
        'eod_exits': sum(1 for t in trades if t.exit_reason == 'eod'),
    }


# =============================================================================
# Step 6: Permutation Test
# =============================================================================

def run_permutation_test(all_day_data, all_pred_raw_indices, all_predictions,
                         config, n_perms=50):
    """
    Permutation test: randomly flip prediction signs.
    If random directions also make money, the result is ARTIFACT.
    """
    # Run actual model
    real_trades = []
    for date_str in sorted(all_day_data.keys()):
        day_data = all_day_data[date_str]
        preds = all_predictions[date_str]
        indices = all_pred_raw_indices[date_str]
        trades = simulate_v4_on_day(
            day_data, preds, indices,
            tp_ticks=config['tp'], sl_ticks=config['sl'],
            hold_seconds=config['hold'],
            threshold_quantile=config.get('quantile'),
            threshold_abs=config.get('threshold'),
            side_filter=config.get('side_filter'),
        )
        real_trades.extend(trades)

    if not real_trades:
        return None, 1.0

    real_pnl = np.sum([t.pnl_ticks for t in real_trades])

    # Permutation: randomly flip signs
    perm_better = 0
    for p in range(n_perms):
        perm_trades = []
        for date_str in sorted(all_day_data.keys()):
            day_data = all_day_data[date_str]
            preds = all_predictions[date_str].copy()
            # Random sign flip
            signs = np.random.choice([-1, 1], size=len(preds))
            preds = np.abs(preds) * signs

            indices = all_pred_raw_indices[date_str]
            trades = simulate_v4_on_day(
                day_data, preds, indices,
                tp_ticks=config['tp'], sl_ticks=config['sl'],
                hold_seconds=config['hold'],
                threshold_quantile=config.get('quantile'),
                threshold_abs=config.get('threshold'),
                side_filter=config.get('side_filter'),
            )
            perm_trades.extend(trades)

        if perm_trades:
            perm_pnl = np.sum([t.pnl_ticks for t in perm_trades])
            if perm_pnl >= real_pnl:
                perm_better += 1

    p_value = (perm_better + 1) / (n_perms + 1)
    return real_trades, p_value


# =============================================================================
# Main
# =============================================================================

def main():
    t0_total = time.time()
    print("=" * 70)
    print("V4 MULTIHEAD TICK-LEVEL REPLAY ANALYSIS")
    print("HC #659 Compliant — FIFO Queue, Permutation Tested")
    print("=" * 70)

    # Step 1: Load V4 predictions
    print("\n[1/5] Loading V4 multihead predictions...")
    v4_preds = load_v4_predictions()
    print(f"  Loaded predictions for {len(v4_preds)} dates")

    # Step 2: Check raw MBO availability
    available_dates = []
    for date_str in sorted(v4_preds.keys()):
        mbo_path = os.path.join(RAW_MBO_DIR, f'glbx-mdp3-{date_str}.mbo.dbn.zst')
        if os.path.exists(mbo_path):
            available_dates.append(date_str)
    print(f"  Raw MBO available for {len(available_dates)}/{len(v4_preds)} dates")

    if not available_dates:
        print("ERROR: No dates with both V4 predictions and raw MBO data!")
        return

    # Step 3: Extract day arrays and map predictions
    print(f"\n[2/5] Extracting trade BBO data for {len(available_dates)} days...")
    all_day_data = {}
    all_pred_raw_indices = {}
    all_dir_preds = {}
    all_eofi_preds = {}
    total_preds = 0

    for date_str in available_dates:
        print(f"  Processing {date_str}...")

        # Get prediction timestamps from processed NPZ
        pred_info = v4_preds[date_str]
        n_preds = pred_info['n_preds']
        result = get_prediction_timestamps(date_str, n_preds)
        if result is None:
            print(f"    [SKIP] No processed NPZ for {date_str}")
            continue
        pred_timestamps, valid_count = result

        # Extract trade BBO from raw MBO
        day_data = extract_day_arrays(date_str)
        if day_data is None:
            continue

        # Map prediction timestamps to raw MBO event indices
        raw_indices = map_predictions_to_raw_mbo(pred_timestamps, day_data['ts_event'])

        # Trim predictions to valid count
        dir_preds = pred_info['preds_dir_1s'][:valid_count]
        eofi_preds = pred_info['preds_eofi_1s'][:valid_count] if pred_info['preds_eofi_1s'] is not None else None

        all_day_data[date_str] = day_data
        all_pred_raw_indices[date_str] = raw_indices
        all_dir_preds[date_str] = dir_preds
        if eofi_preds is not None:
            all_eofi_preds[date_str] = eofi_preds
        total_preds += len(dir_preds)

        print(f"    {date_str}: {len(dir_preds)} predictions mapped "
              f"(IC_dir_1s={pred_info.get('ic_dir_1s', '?'):.3f})")

    print(f"\n  Total: {len(all_day_data)} days, {total_preds:,} predictions")

    # Step 4: Run baseline sweep
    print(f"\n[3/5] Running V4 tick replay configs...")

    configs = [
        # Standard configs matching v3.4.2 decomposition
        {'name': 'v4_all_both_tp2sl4', 'tp': 2, 'sl': 4, 'hold': 30, 'quantile': None, 'threshold': 0.0, 'side_filter': None},
        # Short-only (strongest edge historically)
        {'name': 'v4_all_short_tp2sl4', 'tp': 2, 'sl': 4, 'hold': 30, 'threshold': 0.0, 'side_filter': 'short'},
        # Quantile-based (top 3% — best from v3.4.2 analysis)
        {'name': 'v4_q3_short_tp2sl4', 'tp': 2, 'sl': 4, 'hold': 30, 'quantile': 0.03, 'side_filter': 'short'},
        {'name': 'v4_q5_short_tp2sl4', 'tp': 2, 'sl': 4, 'hold': 30, 'quantile': 0.05, 'side_filter': 'short'},
        {'name': 'v4_q10_short_tp2sl4', 'tp': 2, 'sl': 4, 'hold': 30, 'quantile': 0.10, 'side_filter': 'short'},
        {'name': 'v4_q1_short_tp2sl4', 'tp': 2, 'sl': 4, 'hold': 30, 'quantile': 0.01, 'side_filter': 'short'},
        # Tighter TP
        {'name': 'v4_q3_short_tp1sl3', 'tp': 1, 'sl': 3, 'hold': 20, 'quantile': 0.03, 'side_filter': 'short'},
        {'name': 'v4_q3_short_tp1sl2', 'tp': 1, 'sl': 2, 'hold': 15, 'quantile': 0.03, 'side_filter': 'short'},
        # Wider — testing if more room helps V4
        {'name': 'v4_q3_short_tp3sl6', 'tp': 3, 'sl': 6, 'hold': 45, 'quantile': 0.03, 'side_filter': 'short'},
        # Both sides with quantile
        {'name': 'v4_q3_both_tp2sl4', 'tp': 2, 'sl': 4, 'hold': 30, 'quantile': 0.03, 'side_filter': None},
        {'name': 'v4_q5_both_tp2sl4', 'tp': 2, 'sl': 4, 'hold': 30, 'quantile': 0.05, 'side_filter': None},
        # Time-only exit (no TP) — measures pure adverse selection
        {'name': 'v4_q3_short_noTP_sl4', 'tp': 100, 'sl': 4, 'hold': 30, 'quantile': 0.03, 'side_filter': 'short'},
        {'name': 'v4_all_short_noTP_sl4', 'tp': 100, 'sl': 4, 'hold': 30, 'threshold': 0.0, 'side_filter': 'short'},
    ]

    results = []
    for cfg in configs:
        print(f"\n  Config: {cfg['name']}")
        all_trades = []

        for date_str in sorted(all_day_data.keys()):
            trades = simulate_v4_on_day(
                all_day_data[date_str],
                all_dir_preds[date_str],
                all_pred_raw_indices[date_str],
                tp_ticks=cfg['tp'], sl_ticks=cfg['sl'],
                hold_seconds=cfg['hold'],
                threshold_quantile=cfg.get('quantile'),
                threshold_abs=cfg.get('threshold'),
                side_filter=cfg.get('side_filter'),
            )
            all_trades.extend(trades)

        decomp = decompose_adverse_selection(all_trades)
        metrics = compute_metrics(all_trades, cfg['name'])

        result = {**cfg, **metrics, **decomp}
        results.append(result)

        if all_trades:
            print(f"    Trades: {len(all_trades)}, Net PnL: {metrics['avg_pnl_ticks']:+.3f} ticks, "
                  f"WR: {metrics['win_rate']:.1%}, Sharpe: {metrics['sharpe']:.2f}")
            print(f"    Raw PnL: {decomp['avg_raw_pnl_ticks']:+.3f}, Cost: {decomp['avg_cost_ticks']:.3f}, "
                  f"Fill latency: {decomp['avg_fill_latency_s']:.2f}s")
        else:
            print(f"    NO TRADES")

    # Step 5: Eofi confluence test on best configs
    print(f"\n[4/5] Testing eofi confluence filtering...")

    if all_eofi_preds:
        # Combine all eofi predictions to get quantile thresholds
        all_eofi_flat = np.concatenate([v for v in all_eofi_preds.values()])
        eofi_p50 = np.percentile(all_eofi_flat, 50)
        eofi_p75 = np.percentile(all_eofi_flat, 75)
        eofi_p90 = np.percentile(all_eofi_flat, 90)
        print(f"  Eofi distribution: p50={eofi_p50:.4f}, p75={eofi_p75:.4f}, p90={eofi_p90:.4f}")

        eofi_configs = [
            {'name': 'v4_q3_short_tp2sl4_eofi50', 'tp': 2, 'sl': 4, 'hold': 30,
             'quantile': 0.03, 'side_filter': 'short', 'eofi_thresh': eofi_p50},
            {'name': 'v4_q3_short_tp2sl4_eofi75', 'tp': 2, 'sl': 4, 'hold': 30,
             'quantile': 0.03, 'side_filter': 'short', 'eofi_thresh': eofi_p75},
            {'name': 'v4_q3_short_tp2sl4_eofi90', 'tp': 2, 'sl': 4, 'hold': 30,
             'quantile': 0.03, 'side_filter': 'short', 'eofi_thresh': eofi_p90},
        ]

        for cfg in eofi_configs:
            print(f"\n  Config: {cfg['name']}")
            all_trades = []

            for date_str in sorted(all_day_data.keys()):
                eofi = all_eofi_preds.get(date_str)
                trades = simulate_v4_on_day(
                    all_day_data[date_str],
                    all_dir_preds[date_str],
                    all_pred_raw_indices[date_str],
                    tp_ticks=cfg['tp'], sl_ticks=cfg['sl'],
                    hold_seconds=cfg['hold'],
                    threshold_quantile=cfg.get('quantile'),
                    side_filter=cfg.get('side_filter'),
                    eofi_preds=eofi,
                    eofi_threshold=cfg.get('eofi_thresh'),
                )
                all_trades.extend(trades)

            decomp = decompose_adverse_selection(all_trades)
            metrics = compute_metrics(all_trades, cfg['name'])
            result = {**cfg, **metrics, **decomp}
            results.append(result)

            if all_trades:
                print(f"    Trades: {len(all_trades)}, Net PnL: {decomp['avg_net_pnl_ticks']:+.3f}, "
                      f"WR: {metrics['win_rate']:.1%}, Sharpe: {metrics['sharpe']:.2f}")
            else:
                print(f"    NO TRADES")

    # Step 6: Permutation test on any promising configs
    print(f"\n[5/5] Permutation tests on promising configs...")

    promising = [r for r in results if r.get('avg_net_pnl_ticks', -99) > -0.2 and r.get('n_trades', 0) > 20]

    if promising:
        for r in promising[:3]:  # Top 3 by net PnL
            cfg_name = r['name']
            print(f"\n  Permutation test: {cfg_name} (n={N_PERMS})")

            config = {
                'tp': r['tp'], 'sl': r['sl'], 'hold': r['hold'],
                'quantile': r.get('quantile'), 'threshold': r.get('threshold'),
                'side_filter': r.get('side_filter'),
            }

            _, p_value = run_permutation_test(
                all_day_data, all_pred_raw_indices, all_dir_preds,
                config, n_perms=N_PERMS
            )

            r['perm_p_value'] = p_value
            print(f"    p-value: {p_value:.3f} {'✅ PASS' if p_value < 0.05 else '❌ FAIL'}")
    else:
        print("  No configs with avg PnL > -0.2 ticks to test")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — V4 vs v3.4.2 Adverse Selection")
    print("=" * 70)
    print(f"\nv3.4.2 baseline: raw=+0.145, commission=-0.376, adverse=-0.770, net=-1.001")
    print()

    # Sort by net PnL
    results.sort(key=lambda x: x.get('avg_net_pnl_ticks', -99), reverse=True)

    print(f"{'Config':<35} {'N':>5} {'NetPnL':>8} {'RawPnL':>8} {'Cost':>6} {'WR':>6} {'Sharpe':>7} {'p-val':>6}")
    print("-" * 83)
    for r in results:
        n = r.get('n_trades', 0)
        if n == 0:
            continue
        net = r.get('avg_net_pnl_ticks', 0)
        raw = r.get('avg_raw_pnl_ticks', 0)
        cost = r.get('avg_cost_ticks', 0)
        wr = r.get('win_rate', 0)
        sharpe = r.get('sharpe', 0)
        pval = r.get('perm_p_value', '')
        pval_str = f"{pval:.3f}" if isinstance(pval, float) else "  —"

        print(f"{r['name']:<35} {n:>5} {net:>+8.3f} {raw:>+8.3f} {cost:>6.3f} {wr:>5.1%} {sharpe:>7.2f} {pval_str:>6}")

    # Adverse selection comparison
    all_preds_config = [r for r in results if r.get('name') == 'v4_all_short_noTP_sl4']
    if all_preds_config:
        r = all_preds_config[0]
        v4_raw = r.get('avg_raw_pnl_ticks', 0)
        # Implied adverse selection: raw_edge(0.225 from signal analysis) - raw_pnl_after_fill
        # But we need the raw edge from pure signal (no execution)
        print(f"\n--- ADVERSE SELECTION DECOMPOSITION ---")
        print(f"V4 raw PnL after fill (time-only exit): {v4_raw:+.3f} ticks")
        print(f"V4 signal raw edge (from item #54):     +0.225 ticks")
        v4_adverse = 0.225 - v4_raw
        print(f"V4 implied adverse selection:            {v4_adverse:+.3f} ticks")
        print(f"v3.4.2 adverse selection:                +0.770 ticks")
        print(f"Improvement:                             {0.770 - v4_adverse:+.3f} ticks ({(0.770 - v4_adverse)/0.770*100:.0f}%)")

        if v4_adverse < 0.376:
            print(f"\n🟢 V4 adverse selection ({v4_adverse:.3f}) < commission ({COMMISSION_RT_TICKS:.3f})")
            print(f"   → V4 multihead signals MAY be profitable with passive execution!")
        else:
            print(f"\n🔴 V4 adverse selection ({v4_adverse:.3f}) > commission ({COMMISSION_RT_TICKS:.3f})")
            print(f"   → Still not profitable. Need {v4_adverse - 0.376:.3f} more ticks of improvement.")

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'v4_tick_replay_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    elapsed = time.time() - t0_total
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
