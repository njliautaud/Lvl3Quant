#!/usr/bin/env python3
"""
Tick Replay v16.1 — Expanded Config Sweep + Signal Diagnostic
=============================================================

v16 showed 0/108 configs profitable on 3 days with 3 heads × 3 quantiles × 12 configs.
This expands the search:

1. MORE SIGNAL HEADS: log_ret_5s, p_up_5s, p_up_10s, fifo_tp8sl5_net (6 total)
2. SHORT-ONLY configs (signal decay analysis says short side has +1.56 ticks avg, 60.5% WR)
3. WIDER CONFIG SPACE: TP up to 8 ticks, holds up to 60s, asymmetric TP/SL
4. SIGNAL DIAGNOSTIC: Raw IC vs realized execution outcome analysis
5. If any config shows promise → expand to all 34 available days

Uses cached day_cache from v16 (3 days already extracted).

Author: Claude
Date: 2026-07-10
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

MBO_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v16'
CACHE_DIR = os.path.join(OUTPUT_DIR, 'day_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# EXPANDED heads — v16 only tested 3
SIGNAL_HEADS = [
    'pred_log_ret_1s',
    'pred_log_ret_5s',
    'pred_log_ret_10s',
    'pred_p_up_5s',
    'pred_p_up_10s',
    'pred_fifo_tp4sl3_net',
    'pred_fifo_tp8sl5_net',
    'pred_pred_mfe_30s_ticks',
]

# Phase 1 reuse from v16
def extract_day_arrays(mbo_path, date_key):
    """Extract trade arrays + trade-based BBO (same as v16, with caching)."""
    cache_path = os.path.join(CACHE_DIR, f'{date_key}_extracted.npz')
    if os.path.exists(cache_path):
        print(f"    [cache hit] {date_key}")
        return dict(np.load(cache_path))

    try:
        import databento as db
        store = db.DBNStore.from_file(mbo_path)
        df = store.to_df()
    except Exception as e:
        print(f"    [ERROR] {date_key}: {e}")
        return None

    es_symbols = [s for s in df['symbol'].unique()
                  if s.startswith('ES') and '-' not in s and len(s) <= 4]
    if not es_symbols:
        es_symbols = [s for s in df['symbol'].unique()
                      if s.startswith('ES') and '-' not in s]
    if not es_symbols:
        print(f"    [ERROR] {date_key}: no ES symbols found")
        return None

    best_sym = max(es_symbols,
                   key=lambda s: len(df[(df['symbol'] == s) & (df['action'] == 'T')]))
    df = df[df['symbol'] == best_sym].copy()
    print(f"    {date_key}: filtered to {best_sym} ({len(df):,} events)")

    n_events = len(df)
    t0 = time.time()

    ts_event = df['ts_event'].values.astype(np.int64)
    actions = df['action'].astype(str).values if hasattr(df['action'], 'cat') else df['action'].values
    sides = df['side'].astype(str).values if hasattr(df['side'], 'cat') else df['side'].values
    prices = df['price'].values.astype(np.float64)
    sizes = df['size'].values.astype(np.int64)

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
    report_interval = n_events // 10

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

        if report_interval > 0 and i > 0 and i % report_interval == 0:
            elapsed = time.time() - t0
            pct = i / n_events * 100
            eta = elapsed / (i / n_events) - elapsed
            spread = last_ask - last_bid if last_ask > 0 and last_bid > 0 else -1
            print(f"\r    {date_key}: {pct:.0f}% ({elapsed:.0f}s, ETA {eta:.0f}s) "
                  f"bid={last_bid:.2f} ask={last_ask:.2f} spread={spread:.2f}    ",
                  end='', flush=True)

    elapsed = time.time() - t0
    both_valid = (bbo_bid > 0) & (bbo_ask > 0)
    if both_valid.any():
        spreads = bbo_ask[both_valid] - bbo_bid[both_valid]
        good = (spreads >= 0) & (spreads <= 1.0)
        print(f"\r    {date_key}: {n_events:,} events in {elapsed:.0f}s — "
              f"BBO valid {both_valid.mean()*100:.1f}%, "
              f"good spread {good.mean()*100:.1f}%, "
              f"median spread {np.median(spreads):.2f}")

    np.savez_compressed(cache_path,
        ts_event=ts_event, bbo_bid=bbo_bid, bbo_ask=bbo_ask,
        bbo_bid_size=bbo_bid_size, bbo_ask_size=bbo_ask_size,
        is_trade=is_trade, trade_price=trade_price,
        trade_size=trade_size, trade_side=trade_side_arr,
        n_events=np.array([n_events])
    )

    return {
        'ts_event': ts_event, 'bbo_bid': bbo_bid, 'bbo_ask': bbo_ask,
        'bbo_bid_size': bbo_bid_size, 'bbo_ask_size': bbo_ask_size,
        'is_trade': is_trade, 'trade_price': trade_price,
        'trade_size': trade_size, 'trade_side': trade_side_arr,
        'n_events': np.array([n_events])
    }


def simulate_config_on_day(day_data, predictions, tp_ticks, sl_ticks,
                           hold_seconds, threshold, cancel_seconds,
                           side_filter='both'):
    """
    Simulate a config on pre-extracted day arrays.
    side_filter: 'both', 'long_only', 'short_only'
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
    trade_side_arr = day_data['trade_side']

    n_preds = len(predictions)
    pred_indices = np.arange(PRED_WINDOW, PRED_WINDOW + n_preds * PRED_STRIDE, PRED_STRIDE)
    pred_indices = pred_indices[pred_indices < n_events]
    predictions = predictions[:len(pred_indices)]

    hold_ns = int(hold_seconds * 1e9)
    cancel_ns = int(cancel_seconds * 1e9)
    tp_price_delta = tp_ticks * TICK_SIZE
    sl_price_delta = sl_ticks * TICK_SIZE

    trade_indices = np.where(is_trade)[0]
    trade_ts = ts[trade_indices]
    trade_pr = trade_price_arr[trade_indices]
    trade_sz = trade_size_arr[trade_indices]
    trade_sd = trade_side_arr[trade_indices]

    completed_trades = []
    position_exit_after = 0

    for pi, pred_idx in enumerate(pred_indices):
        pred = predictions[pi]
        if pred_idx < position_exit_after:
            continue
        if abs(pred) < threshold:
            continue

        bid = bbo_bid[pred_idx]
        ask = bbo_ask[pred_idx]
        if bid <= 0 or ask <= 0 or (ask - bid) > 2 * TICK_SIZE:
            continue

        # Determine side
        if pred > threshold:
            side = 'long'
        else:
            side = 'short'

        # Apply side filter
        if side_filter == 'long_only' and side != 'long':
            continue
        if side_filter == 'short_only' and side != 'short':
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

        ti_start = np.searchsorted(trade_indices, pred_idx, side='right')

        # --- ENTRY FILL SEARCH ---
        filled = False
        fill_ti = -1
        fill_ts_val = 0
        remaining_queue = queue_ahead

        for ti in range(ti_start, len(trade_indices)):
            t_ts = trade_ts[ti]
            t_pr = trade_pr[ti]
            t_sz = int(trade_sz[ti])
            t_sd = trade_sd[ti]

            if t_ts > cancel_ts:
                break

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

        # --- EXIT SEARCH ---
        exit_deadline_ts = fill_ts_val + hold_ns
        fill_event_idx = trade_indices[fill_ti]
        if side == 'long':
            tp_queue = float(bbo_ask_size[min(fill_event_idx, n_events - 1)])
        else:
            tp_queue = float(bbo_bid_size[min(fill_event_idx, n_events - 1)])

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

            # SL check (market exit)
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

            # Time stop (market exit)
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
            fill_latency_ns=fill_ts_val - signal_ts,
            pnl_ticks=net_pnl,
            pnl_dollars=net_pnl * TICK_VALUE,
            cost_ticks=cost,
            mfe_ticks=mfe,
            mae_ticks=mae,
        ))

        exit_event_idx = trade_indices[ti] if ti < len(trade_indices) else n_events
        position_exit_after = exit_event_idx

    return completed_trades


def load_predictions(pred_dir):
    preds = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, 'oot_*.npz'))):
        d = np.load(f, allow_pickle=True)
        date_str = os.path.basename(f).replace('oot_', '').replace('.npz', '')
        heads = {}
        for head in SIGNAL_HEADS:
            if head in d:
                heads[head] = d[head].astype(np.float32)
        if heads:
            preds[date_str] = heads
    return preds


def compute_quantile_threshold(preds_dict, head, q):
    vals = []
    for d, heads in preds_dict.items():
        if head in heads:
            vals.append(np.abs(heads[head]))
    if not vals:
        return 999.0
    return float(np.percentile(np.concatenate(vals), (1 - q) * 100))


def signal_diagnostic(preds_dict, day_arrays):
    """
    Analyze raw signal properties without execution.
    For each prediction, check what actually happened in the next N seconds.
    """
    print(f"\n{'='*70}")
    print("SIGNAL DIAGNOSTIC — Raw Signal vs Realized Moves")
    print(f"{'='*70}")

    for head in ['pred_log_ret_1s', 'pred_log_ret_5s', 'pred_fifo_tp4sl3_net']:
        print(f"\n  HEAD: {head}")

        all_preds = []
        all_realized_1s = []
        all_realized_5s = []
        all_realized_mfe_5s = []
        all_fill_rates = []

        for date_key, data in day_arrays.items():
            if date_key not in preds_dict or head not in preds_dict[date_key]:
                continue

            preds = preds_dict[date_key][head]
            n_events = int(data['n_events'][0])
            ts = data['ts_event']
            bbo_bid = data['bbo_bid']
            bbo_ask = data['bbo_ask']
            is_trade = data['is_trade']
            trade_price_arr = data['trade_price']
            trade_side_arr = data['trade_side']

            n_preds = len(preds)
            pred_indices = np.arange(PRED_WINDOW, PRED_WINDOW + n_preds * PRED_STRIDE, PRED_STRIDE)
            pred_indices = pred_indices[pred_indices < n_events]

            trade_indices = np.where(is_trade)[0]
            trade_ts = ts[trade_indices]
            trade_pr = trade_price_arr[trade_indices]

            for pi in range(len(pred_indices)):
                pred_idx = pred_indices[pi]
                pred = preds[pi]
                bid = bbo_bid[pred_idx]
                ask = bbo_ask[pred_idx]
                if bid <= 0 or ask <= 0 or (ask - bid) > 2 * TICK_SIZE:
                    continue

                mid = (bid + ask) / 2
                signal_ts = ts[pred_idx]
                ti_start = np.searchsorted(trade_indices, pred_idx, side='right')

                # Find realized move at 1s, 5s
                realized_1s = np.nan
                realized_5s = np.nan
                mfe_5s = 0.0

                for ti in range(ti_start, len(trade_indices)):
                    dt_ns = trade_ts[ti] - signal_ts
                    if dt_ns > 5e9:
                        break
                    move_ticks = (trade_pr[ti] - mid) / TICK_SIZE
                    if pred < 0:
                        move_ticks = -move_ticks
                    mfe_5s = max(mfe_5s, move_ticks)
                    if np.isnan(realized_1s) and dt_ns >= 1e9:
                        realized_1s = move_ticks
                    if np.isnan(realized_5s) and dt_ns >= 5e9:
                        realized_5s = move_ticks

                all_preds.append(float(pred))
                all_realized_1s.append(realized_1s)
                all_realized_5s.append(realized_5s)
                all_realized_mfe_5s.append(mfe_5s)

        all_preds = np.array(all_preds)
        all_realized_1s = np.array(all_realized_1s)
        all_realized_mfe_5s = np.array(all_realized_mfe_5s)

        # Split by signal strength quantiles
        abs_preds = np.abs(all_preds)
        for q_label, q_lo, q_hi in [('Top 5%', 0.95, 1.0), ('Top 10%', 0.90, 0.95),
                                      ('Top 20%', 0.80, 0.90), ('All', 0.0, 1.0)]:
            lo = np.percentile(abs_preds, q_lo * 100)
            hi = np.percentile(abs_preds, q_hi * 100) if q_hi < 1.0 else abs_preds.max() + 1
            mask = (abs_preds >= lo) & (abs_preds <= hi)
            n = mask.sum()
            if n < 10:
                continue

            r1s = all_realized_1s[mask]
            mfe5s = all_realized_mfe_5s[mask]
            valid_1s = ~np.isnan(r1s)

            mean_r1s = np.nanmean(r1s) if valid_1s.any() else 0
            mean_mfe5s = np.mean(mfe5s)
            wr_1s = np.nanmean(r1s[valid_1s] > 0) if valid_1s.any() else 0

            # Short vs long split
            short_mask = mask & (all_preds < 0)
            long_mask = mask & (all_preds > 0)
            short_mfe = np.mean(all_realized_mfe_5s[short_mask]) if short_mask.any() else 0
            long_mfe = np.mean(all_realized_mfe_5s[long_mask]) if long_mask.any() else 0

            print(f"    {q_label:>8s} (n={n:5d}): r1s={mean_r1s:+.3f}t  MFE5s={mean_mfe5s:.2f}t  "
                  f"WR1s={wr_1s:.1%}  short_MFE={short_mfe:.2f}t  long_MFE={long_mfe:.2f}t")

        # Fill rate analysis — how often does a passive order at BBO get filled within cancel window?
        print(f"    Fill rate (passive at BBO, 10s cancel): ", end='')
        n_attempted = 0
        n_filled = 0
        for date_key, data in day_arrays.items():
            if date_key not in preds_dict or head not in preds_dict[date_key]:
                continue
            preds = preds_dict[date_key][head]
            n_events = int(data['n_events'][0])
            ts = data['ts_event']
            bbo_bid = data['bbo_bid']
            bbo_ask = data['bbo_ask']
            bbo_bid_size = data['bbo_bid_size']
            is_trade = data['is_trade']
            trade_price_arr = data['trade_price']
            trade_side_arr = data['trade_side']
            trade_size_arr = data['trade_size']

            n_preds_local = len(preds)
            pred_indices = np.arange(PRED_WINDOW, PRED_WINDOW + n_preds_local * PRED_STRIDE, PRED_STRIDE)
            pred_indices = pred_indices[pred_indices < n_events]
            trade_indices = np.where(is_trade)[0]
            trade_ts = ts[trade_indices]
            trade_pr = trade_price_arr[trade_indices]
            trade_sz = trade_size_arr[trade_indices]
            trade_sd = trade_side_arr[trade_indices]

            # Sample top-20% signals
            abs_p = np.abs(preds[:len(pred_indices)])
            thresh = np.percentile(abs_p, 80)

            for pi in range(min(len(pred_indices), len(preds))):
                if abs(preds[pi]) < thresh:
                    continue
                pred_idx = pred_indices[pi]
                bid = bbo_bid[pred_idx]
                ask = bbo_ask[pred_idx]
                if bid <= 0 or ask <= 0 or (ask - bid) > 2 * TICK_SIZE:
                    continue

                n_attempted += 1
                signal_ts_val = ts[pred_idx]
                cancel_ts_val = signal_ts_val + int(10e9)

                if preds[pi] > 0:
                    entry_price = bid
                    queue = float(bbo_bid_size[pred_idx])
                    ti_s = np.searchsorted(trade_indices, pred_idx, side='right')
                    for ti in range(ti_s, len(trade_indices)):
                        if trade_ts[ti] > cancel_ts_val:
                            break
                        if trade_sd[ti] == -1:
                            if abs(trade_pr[ti] - entry_price) < 0.001:
                                queue -= trade_sz[ti]
                                if queue <= 0:
                                    n_filled += 1
                                    break
                            elif trade_pr[ti] < entry_price:
                                n_filled += 1
                                break
                else:
                    entry_price = ask
                    queue = float(data['bbo_ask_size'][pred_idx])
                    ti_s = np.searchsorted(trade_indices, pred_idx, side='right')
                    for ti in range(ti_s, len(trade_indices)):
                        if trade_ts[ti] > cancel_ts_val:
                            break
                        if trade_sd[ti] == 1:
                            if abs(trade_pr[ti] - entry_price) < 0.001:
                                queue -= trade_sz[ti]
                                if queue <= 0:
                                    n_filled += 1
                                    break
                            elif trade_pr[ti] > entry_price:
                                n_filled += 1
                                break

        fill_rate = n_filled / max(n_attempted, 1) * 100
        print(f"{n_filled}/{n_attempted} = {fill_rate:.1f}%")


def main():
    t0 = time.time()
    print("=" * 70)
    print("TICK REPLAY v16.1 — EXPANDED CONFIG SWEEP + DIAGNOSTIC")
    print("=" * 70)

    # Load predictions
    print("\nLoading predictions...")
    preds_dict = load_predictions(PRED_DIR)
    print(f"  {len(preds_dict)} dates, heads per date: {list(list(preds_dict.values())[0].keys())}")

    # Match MBO files
    mbo_files = sorted(glob.glob(os.path.join(MBO_DIR, 'glbx-mdp3-*.mbo.dbn.zst')))
    all_matched = []
    for mbo_path in mbo_files:
        date8 = os.path.basename(mbo_path).split('-')[2].split('.')[0]
        if date8 in preds_dict:
            all_matched.append((mbo_path, date8))

    print(f"  {len(all_matched)} matched MBO+pred days")

    # PHASE 0: Use existing 3 cached days first for diagnostic + expanded sweep
    # Then if anything promising, extract all 34 days
    screen_dates_initial = ['20260223', '20260313', '20260414']

    print(f"\n{'='*70}")
    print("PHASE 1: LOAD CACHED DAYS + EXTRACT NEW")
    print(f"{'='*70}")

    day_arrays = {}
    for mbo_path, date_key in all_matched:
        if date_key in screen_dates_initial:
            data = extract_day_arrays(mbo_path, date_key)
            if data is not None:
                day_arrays[date_key] = data
                gc.collect()

    phase1_time = time.time() - t0
    print(f"\n  Phase 1: {len(day_arrays)} days loaded in {phase1_time:.0f}s")

    # PHASE 1.5: SIGNAL DIAGNOSTIC
    signal_diagnostic(preds_dict, day_arrays)

    # PHASE 2: EXPANDED CONFIG SWEEP
    print(f"\n{'='*70}")
    print("PHASE 2: EXPANDED CONFIG SWEEP")
    print(f"{'='*70}")

    QUANTILES = [0.03, 0.05, 0.10, 0.20]

    # Expanded configs: (hold_s, tp, sl, cancel_s, side_filter)
    CONFIGS = [
        # --- Original v16 configs (both sides) ---
        (5,  3, 3, 10, 'both'),
        (10, 4, 3, 15, 'both'),
        (15, 5, 3, 15, 'both'),
        (5, 99, 99, 10, 'both'),
        (10, 99, 99, 15, 'both'),
        (30, 99, 99, 20, 'both'),

        # --- SHORT ONLY (signal decay analysis: short side = better edge) ---
        (5,  3, 3, 10, 'short_only'),
        (5,  4, 3, 10, 'short_only'),
        (10, 4, 3, 15, 'short_only'),
        (10, 3, 5, 15, 'short_only'),
        (15, 5, 3, 15, 'short_only'),
        (5, 99, 99, 10, 'short_only'),
        (10, 99, 99, 15, 'short_only'),
        (15, 99, 99, 15, 'short_only'),
        (30, 99, 99, 20, 'short_only'),

        # --- WIDE TP (6-10 ticks, longer holds) ---
        (30, 6, 4, 20, 'both'),
        (30, 8, 5, 20, 'both'),
        (45, 6, 4, 25, 'both'),
        (45, 8, 5, 25, 'both'),
        (60, 8, 5, 30, 'both'),
        (60, 10, 5, 30, 'both'),

        # --- WIDE TP SHORT ONLY ---
        (30, 6, 4, 20, 'short_only'),
        (30, 8, 5, 20, 'short_only'),
        (45, 6, 4, 25, 'short_only'),
        (45, 8, 5, 25, 'short_only'),
        (60, 8, 5, 30, 'short_only'),
        (60, 10, 5, 30, 'short_only'),

        # --- ASYMMETRIC (tight SL, wide TP) ---
        (15, 6, 2, 15, 'both'),
        (15, 6, 2, 15, 'short_only'),
        (20, 8, 3, 20, 'both'),
        (20, 8, 3, 20, 'short_only'),
        (30, 10, 3, 25, 'both'),
        (30, 10, 3, 25, 'short_only'),

        # --- LONG ONLY (control) ---
        (10, 4, 3, 15, 'long_only'),
        (15, 5, 3, 15, 'long_only'),
        (30, 99, 99, 20, 'long_only'),
    ]

    total_combos = len(SIGNAL_HEADS) * len(QUANTILES) * len(CONFIGS)
    print(f"  {len(SIGNAL_HEADS)} heads x {len(QUANTILES)} quantiles x {len(CONFIGS)} configs = {total_combos}")

    all_results = []
    promising = []
    config_count = 0
    sweep_t0 = time.time()

    for head in SIGNAL_HEADS:
        has_head = any(head in preds_dict.get(d, {}) for d in day_arrays)
        if not has_head:
            print(f"\n  Skipping {head} (not in predictions)")
            continue

        print(f"\n{'='*60}")
        print(f"HEAD: {head}")
        print(f"{'='*60}")

        for q in QUANTILES:
            thresh = compute_quantile_threshold(preds_dict, head, q)
            print(f"\n  Q={q*100:.0f}% -> threshold={thresh:.6f}")

            for hold_s, tp, sl, cancel_s, side_filter in CONFIGS:
                config_count += 1
                sf_tag = '' if side_filter == 'both' else f'_{side_filter[:1]}'
                label = f"{head.split('_',1)[1]}|q{q*100:.0f}|h{hold_s}tp{tp}sl{sl}{sf_tag}"

                all_trades = []
                day_pnls = []
                t1 = time.time()

                for date_key, data in day_arrays.items():
                    if date_key not in preds_dict or head not in preds_dict[date_key]:
                        continue

                    trades = simulate_config_on_day(
                        data, preds_dict[date_key][head],
                        tp, sl, hold_s, thresh, cancel_s, side_filter
                    )
                    all_trades.extend(trades)
                    day_pnls.append(sum(t.pnl_ticks for t in trades))

                elapsed = time.time() - t1

                if not all_trades:
                    continue

                m = compute_metrics(all_trades, label)
                n = m.get('n_trades', 0)
                net = m.get('net_pnl_ticks', 0)
                wr = m.get('win_rate', 0)
                sharpe = m.get('sharpe', 0)
                avg = net / max(n, 1)
                green = sum(1 for p in day_pnls if p > 0)
                red = sum(1 for p in day_pnls if p < 0)

                marker = "**" if sharpe > 1.0 else ("Y" if sharpe > 0.5 else ("+" if sharpe > 0 else " "))
                if sharpe > 0 or config_count <= 20:
                    print(f"    [{config_count:4d}] {marker:2s} {label:50s}: "
                          f"{'+' if net > 0 else ''}{net:.0f}t "
                          f"({'+' if avg > 0 else ''}{avg:.3f}t/tr) n={n} WR={wr:.1%} Sh={sharpe:.2f} "
                          f"G/R={green}/{red} MFE={np.mean([t.mfe_ticks for t in all_trades]):.1f} "
                          f"MAE={np.mean([t.mae_ticks for t in all_trades]):.1f}")

                result = {
                    'label': label, 'head': head, 'quantile': q,
                    'threshold': thresh, 'hold_s': hold_s,
                    'tp': tp, 'sl': sl, 'cancel_s': cancel_s,
                    'side_filter': side_filter,
                    'n_trades': n, 'trades_per_day': round(n / max(len(day_pnls), 1), 1),
                    'net_ticks': round(float(net), 1),
                    'per_trade': round(float(avg), 4),
                    'win_rate': round(float(wr), 4),
                    'sharpe': round(float(sharpe), 3),
                    'green_days': green, 'red_days': red,
                    'n_days': len(day_pnls),
                    'exit_reasons': m.get('exit_reasons', {}),
                    'avg_mfe': round(float(np.mean([t.mfe_ticks for t in all_trades])), 2),
                    'avg_mae': round(float(np.mean([t.mae_ticks for t in all_trades])), 2),
                }
                all_results.append(result)

                if sharpe > 0.3 and n >= 8:
                    promising.append(result)

    sweep_time = time.time() - sweep_t0

    # Summary
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"  Phase 1 (extraction): {phase1_time:.0f}s")
    print(f"  Phase 2 (sweep):      {sweep_time:.0f}s ({sweep_time/60:.1f} min)")
    print(f"  Total:                {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    print(f"  Configs tested:       {len(all_results)}")
    print(f"  Positive Sharpe:      {sum(1 for r in all_results if r['sharpe'] > 0)}")
    print(f"  Promising (Sh>0.3):   {len(promising)}")

    # Top 20 by Sharpe
    print(f"\n  TOP 20 BY SHARPE:")
    for i, r in enumerate(sorted(all_results, key=lambda x: -x['sharpe'])[:20]):
        print(f"    {i+1:2d}. {r['label']:50s} Sh={r['sharpe']:+.2f} n={r['n_trades']} "
              f"WR={r['win_rate']:.1%} avg={r['per_trade']:+.3f}t/tr MFE={r['avg_mfe']:.1f} MAE={r['avg_mae']:.1f}")

    # Short-only vs both comparison
    print(f"\n  SHORT-ONLY vs BOTH (same config, averaged over matching pairs):")
    both_results = {r['label'].rstrip('_s'): r for r in all_results if r['side_filter'] == 'both'}
    short_results = {r['label'].rstrip('_s'): r for r in all_results if r['side_filter'] == 'short_only'}
    n_compared = 0
    short_better = 0
    for label in both_results:
        if label in short_results and both_results[label]['n_trades'] >= 5:
            n_compared += 1
            if short_results[label]['sharpe'] > both_results[label]['sharpe']:
                short_better += 1
    if n_compared > 0:
        print(f"    Short-only beats both-sides in {short_better}/{n_compared} configs ({short_better/n_compared:.0%})")

    # PHASE 3: Permutation tests on promising
    validated = []
    if promising:
        print(f"\n{'='*70}")
        print(f"PHASE 3: PERMUTATION TESTS (top 10 of {len(promising)})")
        print(f"{'='*70}")

        N_PERMS = 30

        for cfg in sorted(promising, key=lambda x: -x['sharpe'])[:10]:
            head = cfg['head']
            thresh = cfg['threshold']
            tp_t, sl_t = cfg['tp'], cfg['sl']
            hold_s = cfg['hold_s']
            cancel_s = cfg['cancel_s']
            side_filter = cfg['side_filter']
            real_sharpe = cfg['sharpe']

            print(f"\n  Testing: {cfg['label']} (Sharpe={real_sharpe:.2f}, n={cfg['n_trades']})")

            perm_sharpes = np.zeros(N_PERMS)

            for perm_i in range(N_PERMS):
                perm_trades = []
                for date_key, data in day_arrays.items():
                    if date_key not in preds_dict or head not in preds_dict[date_key]:
                        continue
                    preds_copy = preds_dict[date_key][head].copy()
                    mask = np.random.random(len(preds_copy)) < 0.5
                    preds_copy[mask] *= -1

                    trades = simulate_config_on_day(
                        data, preds_copy, tp_t, sl_t, hold_s, thresh, cancel_s, side_filter
                    )
                    perm_trades.extend(trades)

                if len(perm_trades) >= 5:
                    pnls = [t.pnl_ticks for t in perm_trades]
                    perm_sharpes[perm_i] = np.mean(pnls) / max(np.std(pnls), 1e-10) * np.sqrt(252)

            p_val = float(np.mean(perm_sharpes >= real_sharpe))
            avg_perm = float(np.mean(perm_sharpes))

            status = "PASSES" if p_val < 0.05 else "FAILS"
            print(f"    {status} -- p={p_val:.3f} (real={real_sharpe:.2f} vs perm_mean={avg_perm:.2f})")

            cfg['perm_p_value'] = round(p_val, 3)
            cfg['perm_mean_sharpe'] = round(avg_perm, 3)
            cfg['perm_passes'] = p_val < 0.05

            if p_val < 0.05:
                validated.append(cfg)

    # PHASE 4: If any validated, expand to all days
    if validated:
        print(f"\n{'='*70}")
        print(f"PHASE 4: EXPANDING {len(validated)} VALIDATED CONFIGS TO ALL {len(all_matched)} DAYS")
        print(f"{'='*70}")

        # Extract remaining days
        for mbo_path, date_key in all_matched:
            if date_key not in day_arrays:
                print(f"\n  Extracting {date_key}...")
                data = extract_day_arrays(mbo_path, date_key)
                if data is not None:
                    day_arrays[date_key] = data
                    gc.collect()

        print(f"\n  Now have {len(day_arrays)} days")

        # Re-run validated configs on all days
        full_results = []
        for cfg in validated:
            head = cfg['head']
            thresh = cfg['threshold']
            tp_t, sl_t = cfg['tp'], cfg['sl']
            hold_s = cfg['hold_s']
            cancel_s = cfg['cancel_s']
            side_filter = cfg['side_filter']

            all_trades = []
            day_pnls = []

            for date_key in sorted(day_arrays.keys()):
                if date_key not in preds_dict or head not in preds_dict[date_key]:
                    continue
                trades = simulate_config_on_day(
                    day_arrays[date_key], preds_dict[date_key][head],
                    tp_t, sl_t, hold_s, thresh, cancel_s, side_filter
                )
                all_trades.extend(trades)
                day_pnl = sum(t.pnl_ticks for t in trades)
                day_pnls.append(day_pnl)

            if not all_trades:
                continue

            n = len(all_trades)
            net = sum(t.pnl_ticks for t in all_trades)
            avg = net / n
            wr = sum(1 for t in all_trades if t.pnl_ticks > 0) / n
            daily_pnls_arr = np.array(day_pnls)
            sharpe = float(np.mean(daily_pnls_arr) / max(np.std(daily_pnls_arr), 1e-10) * np.sqrt(252))
            green = sum(1 for p in day_pnls if p > 0)
            red = sum(1 for p in day_pnls if p < 0)

            print(f"\n  FULL: {cfg['label']}: {'+' if net > 0 else ''}{net:.0f}t "
                  f"({'+' if avg > 0 else ''}{avg:.3f}t/tr) n={n} WR={wr:.1%} Sh={sharpe:.2f} "
                  f"G/R={green}/{red}/{len(day_pnls)}")

            full_results.append({
                **cfg,
                'full_n_trades': n,
                'full_net_ticks': round(float(net), 1),
                'full_per_trade': round(float(avg), 4),
                'full_win_rate': round(float(wr), 4),
                'full_sharpe': round(float(sharpe), 3),
                'full_green_days': green,
                'full_red_days': red,
                'full_n_days': len(day_pnls),
            })

    # Save results
    output = {
        'generated': time.strftime('%Y-%m-%d %H:%M'),
        'version': 'v16.1_expanded',
        'dates_screened': sorted(day_arrays.keys()),
        'total_dates_available': len(all_matched),
        'phase1_seconds': round(phase1_time, 1),
        'phase2_seconds': round(sweep_time, 1),
        'total_seconds': round(time.time() - t0, 1),
        'n_configs_tested': len(all_results),
        'n_positive_sharpe': sum(1 for r in all_results if r['sharpe'] > 0),
        'n_promising': len(promising),
        'n_validated': len(validated),
        'top_20': sorted(all_results, key=lambda x: -x['sharpe'])[:20],
        'results': all_results,
        'promising': promising,
        'validated': validated,
    }

    out_path = os.path.join(OUTPUT_DIR, 'v16_1_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Final verdict
    print(f"\n{'='*70}")
    print(f"FINAL VERDICT")
    print(f"{'='*70}")
    if validated:
        print(f"  {len(validated)} configs passed permutation test on screening days")
        print(f"  See full-day expansion results above")
    elif promising:
        print(f"  {len(promising)} configs had Sharpe > 0.3 but NONE passed permutation test")
        print(f"  Signal may have directional edge but it's too weak to overcome costs")
    elif sum(1 for r in all_results if r['sharpe'] > 0) > 0:
        n_pos = sum(1 for r in all_results if r['sharpe'] > 0)
        print(f"  {n_pos} configs had positive Sharpe but none > 0.3")
        print(f"  Signal has marginal edge, might work with better execution or lower costs")
    else:
        print(f"  ZERO configs had positive Sharpe across {len(all_results)} tested")
        print(f"  Model signal does NOT overcome transaction costs in tick-level FIFO execution")
        print(f"  Possible paths forward:")
        print(f"    1. Multi-model confluence (combine heads for higher-conviction entries)")
        print(f"    2. Conditional signals (filter by vol/spread/time-of-day)")
        print(f"    3. Longer horizons (signal may work at 30s+ but not sub-10s)")
        print(f"    4. Accept: model is IC-positive but NOT cost-positive for active trading")


if __name__ == '__main__':
    main()
