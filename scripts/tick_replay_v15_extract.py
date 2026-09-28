#!/usr/bin/env python3
"""
Tick Replay v15 — Extract-Once, Sweep-Many
============================================

Two-phase approach for 100x speedup over v14:

Phase 1 (EXTRACT): Run through MBO events ONCE per day, building the order
book and recording BBO + trade arrays at every event. ~55 min per day, done
only once per day (3 days = ~3 hours). Results cached to disk.

Phase 2 (SWEEP): For each config (head × quantile × params = 108 combos),
use the pre-extracted arrays to simulate trading. Each config takes seconds
(numpy-vectorized forward scanning) instead of 55 minutes.

Total: ~3.5 hours instead of ~13 days (v14).

HC #659: tick-level FIFO replay, permutation test mandatory.

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
    BBOTracker, TICK_SIZE, TICK_VALUE, PRED_STRIDE, PRED_WINDOW,
    COMMISSION_RT_TICKS, SPREAD_TICKS, COST_PASSIVE_EXIT, COST_MARKET_EXIT,
    compute_metrics, Trade
)

MBO_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v15_extract'
CACHE_DIR = os.path.join(OUTPUT_DIR, 'day_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

SIGNAL_HEADS = ['pred_log_ret_1s', 'pred_fifo_tp4sl3_net', 'pred_pred_mfe_30s_ticks']
N_SCREEN_DAYS = 3
N_PERMS = 30


# =============================================================================
# Phase 1: Extract BBO + Trade arrays from MBO data
# =============================================================================

def extract_day_arrays(mbo_path: str, date_key: str) -> dict:
    """
    Run through all MBO events once, recording:
    - bbo_bid[i], bbo_ask[i]: best bid/ask at event i
    - bbo_bid_size[i], bbo_ask_size[i]: depth at BBO
    - ts_event[i]: nanosecond timestamp of event i
    - is_trade[i]: bool, True if event is a trade (T/F action)
    - trade_price[i], trade_size[i], trade_side[i]: trade details (0 for non-trades)

    Returns dict of numpy arrays, or None on failure.
    """
    cache_path = os.path.join(CACHE_DIR, f'{date_key}_extracted.npz')
    if os.path.exists(cache_path):
        print(f"    [cache hit] {date_key}")
        return dict(np.load(cache_path))

    # Parse MBO file
    try:
        import databento as db
        store = db.DBNStore.from_file(mbo_path)
        df = store.to_df()
    except Exception as e:
        print(f"    [ERROR] {date_key}: {e}")
        return None

    # CRITICAL: Filter to front-month ES contract only
    # (MBO files contain multiple contracts: ESH6, ESM6, spreads, etc.)
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
    print(f"    {date_key}: filtered to {best_sym} ({len(df):,} of original events)")

    n_events = len(df)
    print(f"    {date_key}: {n_events:,} events — extracting BBO...", end='', flush=True)
    t0 = time.time()

    # Pre-extract columns as numpy arrays for speed
    ts_event = df['ts_event'].values.astype(np.int64)
    actions = df['action'].values  # categorical or string
    sides = df['side'].values
    prices = df['price'].values.astype(np.float64)
    sizes = df['size'].values.astype(np.int64)
    order_ids = df['order_id'].values.astype(np.int64)

    # Convert to simple string arrays if needed
    if hasattr(actions, 'cat'):
        actions = actions.astype(str).values
    if hasattr(sides, 'cat'):
        sides = sides.astype(str).values

    # Output arrays
    bbo_bid = np.zeros(n_events, dtype=np.float64)
    bbo_ask = np.zeros(n_events, dtype=np.float64)
    bbo_bid_size = np.zeros(n_events, dtype=np.int32)
    bbo_ask_size = np.zeros(n_events, dtype=np.int32)
    is_trade = np.zeros(n_events, dtype=np.bool_)
    trade_price = np.zeros(n_events, dtype=np.float64)
    trade_size = np.zeros(n_events, dtype=np.int32)
    trade_side_arr = np.zeros(n_events, dtype=np.int8)  # 1=buy, -1=sell, 0=none

    # Run book simulation
    book = BBOTracker()
    report_interval = n_events // 10

    for i in range(n_events):
        action = str(actions[i])
        side = str(sides[i])
        price = float(prices[i])
        size = int(sizes[i])
        oid = int(order_ids[i])

        # Update book
        if price == price and price > 0:  # fast NaN check
            book.process_event(action, side, price, size, oid)

        # Record BBO
        bbo_bid[i] = book.best_bid
        bbo_ask[i] = book.best_ask
        bbo_bid_size[i] = book.bid_size
        bbo_ask_size[i] = book.ask_size

        # Record trades
        if action in ('T', 'F') and price == price and price > 0:
            is_trade[i] = True
            trade_price[i] = price
            trade_size[i] = size
            trade_side_arr[i] = 1 if side == 'B' else -1

        if report_interval > 0 and i > 0 and i % report_interval == 0:
            elapsed = time.time() - t0
            pct = i / n_events * 100
            eta = elapsed / (i / n_events) - elapsed
            print(f"\r    {date_key}: {pct:.0f}% ({elapsed:.0f}s, ETA {eta:.0f}s)    ", end='', flush=True)

    elapsed = time.time() - t0
    print(f"\r    {date_key}: {n_events:,} events extracted in {elapsed:.0f}s")

    # Save cache
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


# =============================================================================
# Phase 2: Vectorized Config Sweep
# =============================================================================

def simulate_config_on_day(day_data: dict, predictions: np.ndarray,
                           tp_ticks: int, sl_ticks: int, hold_seconds: float,
                           threshold: float, cancel_seconds: float) -> list:
    """
    Simulate a single config on pre-extracted day arrays.

    Uses forward-scanning on trade arrays instead of rebuilding the book.
    Returns list of Trade-like dicts.
    """
    n_events = int(day_data['n_events'][0])
    ts = day_data['ts_event']
    bbo_bid = day_data['bbo_bid']
    bbo_ask = day_data['bbo_ask']
    bbo_bid_size = day_data['bbo_bid_size']
    bbo_ask_size = day_data['bbo_ask_size']
    is_trade = day_data['is_trade']
    trade_price = day_data['trade_price']
    trade_size = day_data['trade_size']
    trade_side = day_data['trade_side']

    # Build prediction indices
    n_preds = len(predictions)
    pred_indices = np.arange(PRED_WINDOW, PRED_WINDOW + n_preds * PRED_STRIDE, PRED_STRIDE)
    pred_indices = pred_indices[pred_indices < n_events]
    predictions = predictions[:len(pred_indices)]

    # Convert time params to nanoseconds
    hold_ns = int(hold_seconds * 1e9)
    cancel_ns = int(cancel_seconds * 1e9)
    tp_price_delta = tp_ticks * TICK_SIZE
    sl_price_delta = sl_ticks * TICK_SIZE

    # Pre-compute trade event indices for fast forward scanning
    trade_indices = np.where(is_trade)[0]
    trade_ts = ts[trade_indices]
    trade_pr = trade_price[trade_indices]
    trade_sz = trade_size[trade_indices]
    trade_sd = trade_side[trade_indices]

    completed_trades = []
    in_position = False  # Simple: max 1 concurrent position
    position_exit_after = 0  # event index after which we can enter again

    for pi, pred_idx in enumerate(pred_indices):
        pred = predictions[pi]

        if in_position:
            continue
        if pred_idx < position_exit_after:
            continue

        # Check signal
        if abs(pred) < threshold:
            continue

        # Check BBO validity
        bid = bbo_bid[pred_idx]
        ask = bbo_ask[pred_idx]
        if bid <= 0 or ask <= 0 or (ask - bid) > 2 * TICK_SIZE:
            continue

        # Determine side
        if pred > threshold:
            side = 'long'
            entry_price = bid  # passive buy at bid
            queue_ahead = float(bbo_bid_size[pred_idx])
            tp_price_val = entry_price + tp_price_delta
            sl_price_val = entry_price - sl_price_delta
        else:  # pred < -threshold
            side = 'short'
            entry_price = ask  # passive sell at ask
            queue_ahead = float(bbo_ask_size[pred_idx])
            tp_price_val = entry_price - tp_price_delta
            sl_price_val = entry_price + sl_price_delta

        signal_ts = ts[pred_idx]
        cancel_ts = signal_ts + cancel_ns

        # Find trade events after this prediction for entry fill
        ti_start = np.searchsorted(trade_indices, pred_idx, side='right')

        # --- ENTRY FILL SEARCH ---
        filled = False
        fill_ti = -1
        fill_ts_val = 0
        remaining_queue = queue_ahead

        for ti in range(ti_start, len(trade_indices)):
            t_idx = trade_indices[ti]
            t_ts = trade_ts[ti]
            t_pr = trade_pr[ti]
            t_sz = int(trade_sz[ti])
            t_sd = trade_sd[ti]

            # Cancel check
            if t_ts > cancel_ts:
                break

            # Fill check (FIFO queue model)
            if side == 'long' and t_sd == -1:  # Aggressive sell → can fill our buy
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
            elif side == 'short' and t_sd == 1:  # Aggressive buy → can fill our sell
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

        # Get TP queue depth at fill time
        fill_event_idx = trade_indices[fill_ti]
        if side == 'long':
            # TP is on the ask side
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

            # Update MFE/MAE
            if side == 'long':
                best_price_seen = max(best_price_seen, t_pr)
                worst_price_seen = min(worst_price_seen, t_pr)
            else:
                best_price_seen = min(best_price_seen, t_pr)
                worst_price_seen = max(worst_price_seen, t_pr)

            # 1. SL check (market exit)
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

            # 2. TP check (passive, FIFO queue)
            if tp_ticks < 99:  # Skip TP check for pure hold configs
                if side == 'long' and t_sd == 1:  # Buy aggression at TP level (ask)
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
                elif side == 'short' and t_sd == -1:  # Sell aggression at TP level (bid)
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

            # 3. Time stop (market exit)
            if t_ts >= exit_deadline_ts:
                exit_reason = 'time_stop'
                exit_price_val = t_pr
                exit_ts_val = t_ts
                cost = COST_MARKET_EXIT
                break

        if exit_reason is None:
            # EOD or ran out of trades
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
            fill_latency_ns=fill_ts_val - signal_ts,
            pnl_ticks=net_pnl,
            pnl_dollars=net_pnl * TICK_VALUE,
            cost_ticks=cost,
            mfe_ticks=mfe,
            mae_ticks=mae,
        ))

        # Mark position exit — next entry only after this trade exits
        in_position = False
        # Find the event index of the exit
        exit_event_idx = trade_indices[ti] if ti < len(trade_indices) else n_events
        position_exit_after = exit_event_idx

    return completed_trades


def run_permutation_test(day_data: dict, predictions: np.ndarray,
                         tp_ticks: int, sl_ticks: int, hold_seconds: float,
                         threshold: float, cancel_seconds: float,
                         real_sharpe: float, n_perms: int = 30) -> float:
    """Run permutation test: shuffle prediction signs, compare Sharpe."""
    perm_sharpes = []
    for _ in range(n_perms):
        # Randomize direction (keep magnitude)
        shuffled = predictions.copy()
        mask = np.random.random(len(shuffled)) < 0.5
        shuffled[mask] *= -1

        trades = simulate_config_on_day(day_data, shuffled, tp_ticks, sl_ticks,
                                         hold_seconds, threshold, cancel_seconds)
        if len(trades) >= 5:
            pnls = [t.pnl_ticks for t in trades]
            sh = np.mean(pnls) / max(np.std(pnls), 1e-10) * np.sqrt(252)
            perm_sharpes.append(sh)
        else:
            perm_sharpes.append(0.0)

    # p-value: fraction of perms >= real sharpe
    p_val = np.mean([s >= real_sharpe for s in perm_sharpes])
    return p_val


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


def main():
    t0 = time.time()
    print("=" * 70)
    print("TICK REPLAY v15 — EXTRACT-ONCE, SWEEP-MANY")
    print("=" * 70)

    # Load predictions
    print("\nLoading predictions...")
    preds_dict = load_predictions(PRED_DIR)
    print(f"  {len(preds_dict)} dates")

    # Match MBO files
    mbo_files = sorted(glob.glob(os.path.join(MBO_DIR, 'glbx-mdp3-*.mbo.dbn.zst')))
    all_matched = []
    for mbo_path in mbo_files:
        date8 = os.path.basename(mbo_path).split('-')[2].split('.')[0]
        if date8 in preds_dict:
            all_matched.append((mbo_path, date8))

    print(f"  {len(all_matched)} matched MBO+pred days")

    # Select screening days (spread out)
    indices = np.linspace(0, len(all_matched) - 1, N_SCREEN_DAYS, dtype=int)
    screen_matched = [all_matched[i] for i in indices]
    screen_dates = [d for _, d in screen_matched]
    print(f"  Screening: {screen_dates}")

    # ========================================
    # PHASE 1: Extract BBO arrays (once per day)
    # ========================================
    print(f"\n{'='*70}")
    print("PHASE 1: EXTRACT BBO + TRADE ARRAYS")
    print(f"{'='*70}")

    day_arrays = {}
    for mbo_path, date_key in screen_matched:
        data = extract_day_arrays(mbo_path, date_key)
        if data is not None:
            day_arrays[date_key] = data
            gc.collect()

    phase1_time = time.time() - t0
    print(f"\n  Phase 1 complete: {len(day_arrays)} days extracted in {phase1_time:.0f}s ({phase1_time/60:.1f} min)")

    # ========================================
    # PHASE 2: Sweep configs (vectorized)
    # ========================================
    print(f"\n{'='*70}")
    print("PHASE 2: CONFIG SWEEP")
    print(f"{'='*70}")

    QUANTILES = [0.05, 0.10, 0.20]
    CONFIGS = [
        # (hold_s, tp, sl, cancel_s)
        (5,  3, 3, 10),
        (5,  4, 3, 10),
        (10, 4, 3, 15),
        (10, 3, 5, 15),
        (15, 5, 3, 15),
        (5, 99, 99, 10),    # pure hold 5s
        (10, 99, 99, 15),   # pure hold 10s
        (15, 99, 99, 15),   # pure hold 15s
        (30, 99, 99, 20),   # pure hold 30s
        (3,  2, 2, 8),      # very tight
        (7,  3, 4, 12),
        (5,  2, 3, 10),
    ]

    total_combos = len(SIGNAL_HEADS) * len(QUANTILES) * len(CONFIGS)
    print(f"  {len(SIGNAL_HEADS)} heads × {len(QUANTILES)} quantiles × {len(CONFIGS)} configs = {total_combos} combos")
    print(f"  × {len(day_arrays)} days")

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
            print(f"\n  Q={q*100:.0f}% → threshold={thresh:.6f}")

            for hold_s, tp, sl, cancel_s in CONFIGS:
                config_count += 1
                label = f"{head.split('_',1)[1]}|q{q*100:.0f}|h{hold_s}tp{tp}sl{sl}"

                all_trades = []
                day_pnls = []
                t1 = time.time()

                for date_key, data in day_arrays.items():
                    if date_key not in preds_dict or head not in preds_dict[date_key]:
                        continue

                    trades = simulate_config_on_day(
                        data, preds_dict[date_key][head],
                        tp, sl, hold_s, thresh, cancel_s
                    )
                    all_trades.extend(trades)
                    day_pnls.append(sum(t.pnl_ticks for t in trades))

                elapsed = time.time() - t1

                if not all_trades:
                    if config_count <= 10:
                        print(f"    [{config_count:3d}] {label}: NO TRADES ({elapsed:.1f}s)")
                    continue

                m = compute_metrics(all_trades, label)
                n = m.get('n_trades', 0)
                net = m.get('net_pnl_ticks', 0)
                wr = m.get('win_rate', 0)
                sharpe = m.get('sharpe', 0)
                avg = net / max(n, 1)
                green = sum(1 for p in day_pnls if p > 0)
                red = sum(1 for p in day_pnls if p < 0)
                sign = '+' if net > 0 else ''

                marker = "✅" if sharpe > 0.5 else ("➕" if sharpe > 0 else "  ")
                print(f"    [{config_count:3d}] {marker} {label}: {sign}{net:.0f}t "
                      f"({sign}{avg:.3f}t/tr) n={n} WR={wr:.1%} Sh={sharpe:.2f} "
                      f"G/R={green}/{red} MFE={np.mean([t.mfe_ticks for t in all_trades]):.1f} "
                      f"MAE={np.mean([t.mae_ticks for t in all_trades]):.1f} ({elapsed:.1f}s)")

                result = {
                    'label': label, 'head': head, 'quantile': q,
                    'threshold': thresh, 'hold_s': hold_s,
                    'tp': tp, 'sl': sl, 'cancel_s': cancel_s,
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
                    'elapsed_s': round(elapsed, 1),
                }
                all_results.append(result)

                if sharpe > 0.3 and n >= 8:
                    promising.append(result)

    sweep_time = time.time() - sweep_t0

    # Summary
    print(f"\n{'='*60}")
    print(f"SCREENING COMPLETE")
    print(f"{'='*60}")
    print(f"  Phase 1 (extraction): {phase1_time:.0f}s ({phase1_time/60:.1f} min)")
    print(f"  Phase 2 (sweep):      {sweep_time:.0f}s ({sweep_time/60:.1f} min)")
    print(f"  Total:                {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    print(f"  Configs tested:       {len(all_results)}")
    print(f"  Promising (Sh>0.3):   {len(promising)}")

    # ========================================
    # PHASE 3: Permutation tests on promising configs
    # ========================================
    if promising:
        print(f"\n{'='*60}")
        print(f"PHASE 3: PERMUTATION TESTS (top 5 of {len(promising)})")
        print(f"{'='*60}")

        # For permutation, combine predictions across all days
        validated = []
        for cfg in sorted(promising, key=lambda x: -x['sharpe'])[:5]:
            head = cfg['head']
            thresh = cfg['threshold']
            tp_t, sl_t = cfg['tp'], cfg['sl']
            hold_s = cfg['hold_s']
            cancel_s = cfg['cancel_s']
            real_sharpe = cfg['sharpe']

            print(f"\n  Testing: {cfg['label']} (Sharpe={real_sharpe:.2f}, n={cfg['n_trades']})")

            # Run permutation across all days combined
            all_perm_trades_real = []
            all_perm_sharpes = np.zeros(N_PERMS)

            for perm_i in range(N_PERMS):
                perm_trades = []
                for date_key, data in day_arrays.items():
                    if date_key not in preds_dict or head not in preds_dict[date_key]:
                        continue
                    preds_copy = preds_dict[date_key][head].copy()
                    # Shuffle directions
                    mask = np.random.random(len(preds_copy)) < 0.5
                    preds_copy[mask] *= -1

                    trades = simulate_config_on_day(
                        data, preds_copy, tp_t, sl_t, hold_s, thresh, cancel_s
                    )
                    perm_trades.extend(trades)

                if len(perm_trades) >= 5:
                    pnls = [t.pnl_ticks for t in perm_trades]
                    all_perm_sharpes[perm_i] = np.mean(pnls) / max(np.std(pnls), 1e-10) * np.sqrt(252)

            p_val = np.mean(all_perm_sharpes >= real_sharpe)
            avg_perm = np.mean(all_perm_sharpes)

            status = "✅ PASSES" if p_val < 0.05 else "❌ FAILS"
            print(f"    {status} — p={p_val:.3f} (real Sharpe={real_sharpe:.2f} vs perm mean={avg_perm:.2f})")

            cfg['perm_p_value'] = round(float(p_val), 3)
            cfg['perm_mean_sharpe'] = round(float(avg_perm), 3)
            cfg['perm_passes'] = p_val < 0.05

            if p_val < 0.05:
                validated.append(cfg)

        if validated:
            print(f"\n  🎯 {len(validated)} configs VALIDATED (permutation p < 0.05)")
            for v in validated:
                print(f"    {v['label']}: Sharpe={v['sharpe']:.2f}, p={v['perm_p_value']:.3f}, "
                      f"n={v['n_trades']}, WR={v['win_rate']:.1%}")
        else:
            print(f"\n  ⚠️ NO configs passed permutation test")

    # Save all results
    output = {
        'generated': time.strftime('%Y-%m-%d %H:%M'),
        'dates_screened': list(day_arrays.keys()),
        'total_dates_available': len(all_matched),
        'phase1_seconds': round(phase1_time, 1),
        'phase2_seconds': round(sweep_time, 1),
        'total_seconds': round(time.time() - t0, 1),
        'results': all_results,
        'promising': promising,
        'validated': validated if promising else [],
    }

    out_path = os.path.join(OUTPUT_DIR, 'v15_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # If validated configs found, expand to all days
    if promising and any(cfg.get('perm_passes', False) for cfg in promising):
        print(f"\n{'='*60}")
        print(f"PHASE 4: EXPAND WINNERS TO ALL {len(all_matched)} DAYS")
        print(f"{'='*60}")
        print(f"  (Would extract remaining {len(all_matched) - len(day_arrays)} days)")
        print(f"  TODO: implement expansion phase")


if __name__ == '__main__':
    main()
