#!/usr/bin/env python3
"""
Tick Replay v17 — ALIGNMENT-FIXED Sweep
========================================

v16/v16.1 had a CRITICAL alignment bug: predictions indexed by preprocessed
event position were mapped to RAW MBO event position. Since raw MBO files
include overnight/globex data, pred[0] mapped to ~24hrs before the actual
prediction time, giving IC=0.000 and zero profitable configs.

v17 FIX: Load preprocessed timestamps, map each prediction to its correct
raw MBO event via timestamp alignment. IC recovers to 0.10-0.13.

Author: Claude
Date: 2026-07-10
"""

import sys, os, time, json, gc
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/jupiter/Lvl3Quant/engines')
from tick_replay_engine import (
    TICK_SIZE, TICK_VALUE, PRED_STRIDE, PRED_WINDOW,
    COMMISSION_RT_TICKS, SPREAD_TICKS, COST_PASSIVE_EXIT, COST_MARKET_EXIT,
    TickReplayEngine
)

MBO_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate'
PROC_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v17'
CACHE_DIR = os.path.join(OUTPUT_DIR, 'day_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

SIGNAL_HEADS = [
    'pred_log_ret_1s',
    'pred_log_ret_5s',
    'pred_log_ret_10s',
]


def get_aligned_pred_indices(date_key, n_preds, raw_ts):
    """
    Map prediction indices from preprocessed event space to raw MBO event space.

    Returns: array of raw event indices corresponding to each prediction.
    """
    proc_path = os.path.join(PROC_DIR, f'{date_key}_mbo_events.npz')
    if not os.path.exists(proc_path):
        return None

    proc = np.load(proc_path, allow_pickle=True)
    proc_ts = proc['timestamps']
    n_proc = len(proc_ts)

    # Prediction i corresponds to preprocessed event at index (PRED_WINDOW + i * PRED_STRIDE)
    proc_indices = np.arange(PRED_WINDOW, PRED_WINDOW + n_preds * PRED_STRIDE, PRED_STRIDE)
    valid = proc_indices < n_proc
    proc_indices = proc_indices[valid]

    # Get the timestamps for these preprocessed events
    pred_timestamps = proc_ts[proc_indices]

    # Map to nearest raw MBO event by timestamp
    raw_indices = np.searchsorted(raw_ts, pred_timestamps)
    raw_indices = np.clip(raw_indices, 0, len(raw_ts) - 1)

    del proc
    return raw_indices, valid


def extract_day_arrays(mbo_path, date_key):
    """Extract arrays from raw MBO file (with caching)."""
    cache_path = os.path.join(CACHE_DIR, f'{date_key}_extracted.npz')
    if os.path.exists(cache_path):
        return dict(np.load(cache_path))

    # Also check v16 cache
    v16_cache = f'/home/jupiter/Lvl3Quant/output/tick_replay_v16/day_cache/{date_key}_extracted.npz'
    if os.path.exists(v16_cache):
        print(f"    [v16 cache hit] {date_key}")
        data = dict(np.load(v16_cache))
        # Copy to v17 cache
        np.savez_compressed(cache_path, **data)
        return data

    import databento as db

    print(f"  Extracting {date_key}...")
    store = db.DBNStore.from_file(mbo_path)
    df = store.to_df()

    es_symbols = [s for s in df['symbol'].unique()
                  if s.startswith('ES') and '-' not in s and len(s) <= 4]
    if not es_symbols:
        es_symbols = [s for s in df['symbol'].unique()
                      if s.startswith('ES') and '-' not in s]
    if not es_symbols:
        print(f"    [ERROR] {date_key}: no ES symbols")
        return None

    best_sym = max(es_symbols,
                   key=lambda s: len(df[(df.symbol == s) & (df.action == 'T')]))
    df = df[df.symbol == best_sym].copy()
    print(f"    {date_key}: filtered to {best_sym} ({len(df):,} events)")

    ts = df['ts_event'].values.astype('int64')
    actions = df['action'].values.astype('U1')
    sides = df['side'].values.astype('U1')
    prices = df['price'].values.astype('float64')
    sizes = df['size'].values.astype('int64')
    order_ids = df['order_id'].values.astype('int64')

    # Compute BBO from trade prices (v16 approach)
    n = len(df)
    bbo_bid = np.zeros(n, dtype='float64')
    bbo_ask = np.zeros(n, dtype='float64')
    bbo_bid_size = np.zeros(n, dtype='int64')
    bbo_ask_size = np.zeros(n, dtype='int64')
    is_trade = np.zeros(n, dtype='int8')
    trade_price = np.zeros(n, dtype='float64')
    trade_size = np.zeros(n, dtype='int64')
    trade_side = np.zeros(n, dtype='U1')

    last_bid = 0.0
    last_ask = 0.0
    TICK = 0.25

    for i in range(n):
        a = actions[i]
        s = sides[i]
        p = prices[i]
        sz = sizes[i]

        if a == 'T' and p > 0 and not np.isnan(p):
            is_trade[i] = 1
            trade_price[i] = p
            trade_size[i] = sz
            trade_side[i] = s

            if s == 'A':  # Seller aggressor → trade at bid
                last_bid = p
                last_ask = p + TICK
            elif s == 'B':  # Buyer aggressor → trade at ask
                last_ask = p
                last_bid = p - TICK

        bbo_bid[i] = last_bid
        bbo_ask[i] = last_ask

    data = {
        'ts_event': ts,
        'actions': actions,
        'sides': sides,
        'prices': prices,
        'sizes': sizes,
        'order_ids': order_ids,
        'bbo_bid': bbo_bid,
        'bbo_ask': bbo_ask,
        'bbo_bid_size': bbo_bid_size,
        'bbo_ask_size': bbo_ask_size,
        'is_trade': is_trade,
        'trade_price': trade_price,
        'trade_size': trade_size,
        'trade_side': trade_side,
        'n_events': np.array(n),
    }

    np.savez_compressed(cache_path, **data)
    return data


def simulate_config(day_arrays, preds, raw_pred_indices,
                    hold_s, tp, sl, cancel_s, side_filter='both'):
    """
    Fast vectorized simulation using aligned predictions.

    Returns dict with metrics or None if no trades.
    """
    n_ev = int(day_arrays['n_events'])
    ts = day_arrays['ts_event'][:n_ev]
    is_trade = day_arrays['is_trade'][:n_ev].astype(bool)
    trade_px = day_arrays['trade_price'][:n_ev]
    trade_sd = day_arrays['trade_side'][:n_ev]
    bid = day_arrays['bbo_bid'][:n_ev]
    ask = day_arrays['bbo_ask'][:n_ev]

    TICK = 0.25
    hold_ns = int(hold_s * 1e9)
    cancel_ns = int(cancel_s * 1e9)
    tp_price_offset = tp * TICK if tp < 99 else 999999.0
    sl_price_offset = sl * TICK if sl < 99 else 999999.0

    trades = []

    for pi in range(len(raw_pred_indices)):
        pred_idx = raw_pred_indices[pi]
        if pred_idx >= n_ev:
            continue

        pred = preds[pi]
        t_pred = ts[pred_idx]

        # Determine direction
        if pred > 0:
            direction = 'long'
        elif pred < 0:
            direction = 'short'
        else:
            continue

        # Side filter
        if side_filter == 'short_only' and direction == 'long':
            continue
        if side_filter == 'long_only' and direction == 'short':
            continue

        # Entry price (passive limit)
        if direction == 'long':
            entry_price = bid[pred_idx]
        else:
            entry_price = ask[pred_idx]

        if entry_price <= 0:
            continue

        # Set TP/SL prices
        if direction == 'long':
            tp_price = entry_price + tp_price_offset
            sl_price = entry_price - sl_price_offset
        else:
            tp_price = entry_price - tp_price_offset
            sl_price = entry_price + sl_price_offset

        # Simulate queue-based fill:
        # We need a trade at our price on the other side to fill us
        # Simplified: first trade at our level after posting fills us
        filled = False
        fill_idx = None

        for j in range(pred_idx + 1, min(pred_idx + 200000, n_ev)):
            if ts[j] > t_pred + cancel_ns:
                break  # Cancel timeout

            if not is_trade[j]:
                continue

            tp_trade = trade_px[j]
            ts_trade = trade_sd[j]

            # Long entry: filled when aggressive sell (side A) hits our bid
            if direction == 'long' and ts_trade == 'A' and tp_trade <= entry_price:
                filled = True
                fill_idx = j
                break
            # Short entry: filled when aggressive buy (side B) hits our ask
            elif direction == 'short' and ts_trade == 'B' and tp_trade >= entry_price:
                filled = True
                fill_idx = j
                break

        if not filled:
            continue

        fill_time = ts[fill_idx]
        exit_deadline = fill_time + hold_ns

        # Track position for exit
        best_price = entry_price
        worst_price = entry_price
        exit_reason = None
        exit_price = None
        cost = COST_MARKET_EXIT  # default: market exit

        for j in range(fill_idx + 1, min(fill_idx + 500000, n_ev)):
            if not is_trade[j]:
                continue

            tp_j = trade_px[j]
            t_j = ts[j]

            # Update MFE/MAE
            if direction == 'long':
                best_price = max(best_price, tp_j)
                worst_price = min(worst_price, tp_j)
            else:
                best_price = min(best_price, tp_j)
                worst_price = max(worst_price, tp_j)

            # Check SL
            if direction == 'long' and tp_j <= sl_price:
                exit_reason = 'sl'
                exit_price = sl_price
                cost = COST_MARKET_EXIT
                break
            elif direction == 'short' and tp_j >= sl_price:
                exit_reason = 'sl'
                exit_price = sl_price
                cost = COST_MARKET_EXIT
                break

            # Check TP (passive — needs trade through our level)
            if direction == 'long' and tp_j >= tp_price:
                exit_reason = 'tp'
                exit_price = tp_price
                cost = COST_PASSIVE_EXIT
                break
            elif direction == 'short' and tp_j <= tp_price:
                exit_reason = 'tp'
                exit_price = tp_price
                cost = COST_PASSIVE_EXIT
                break

            # Check time stop
            if t_j >= exit_deadline:
                exit_reason = 'time_stop'
                exit_price = tp_j
                cost = COST_MARKET_EXIT
                break

        if exit_reason is None:
            continue  # Never exited (ran out of events)

        # Compute PnL
        if direction == 'long':
            raw_pnl = (exit_price - entry_price) / TICK
            mfe = (best_price - entry_price) / TICK
            mae = (entry_price - worst_price) / TICK
        else:
            raw_pnl = (entry_price - exit_price) / TICK
            mfe = (entry_price - best_price) / TICK
            mae = (worst_price - entry_price) / TICK

        net_pnl = raw_pnl - cost

        trades.append({
            'side': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'raw_pnl': raw_pnl,
            'net_pnl': net_pnl,
            'cost': cost,
            'mfe': mfe,
            'mae': mae,
            'exit_reason': exit_reason,
            'fill_latency_ns': fill_time - t_pred,
            'hold_time_ns': ts[j] - fill_time if exit_reason else 0,
        })

    return trades


def compute_metrics_from_trades(trades_list, n_days):
    """Compute aggregate metrics from trades across days."""
    if not trades_list:
        return None

    all_trades = []
    for day_trades in trades_list:
        all_trades.extend(day_trades)

    if not all_trades:
        return None

    n = len(all_trades)
    net_pnls = [t['net_pnl'] for t in all_trades]
    total_net = sum(net_pnls)
    avg_net = total_net / n

    wins = sum(1 for p in net_pnls if p > 0)
    wr = wins / n if n > 0 else 0

    # Daily P&L for Sharpe
    from collections import defaultdict
    daily_pnl = defaultdict(float)
    for i, trades in enumerate(trades_list):
        daily_pnl[i] = sum(t['net_pnl'] for t in trades)

    daily_vals = list(daily_pnl.values())
    if len(daily_vals) > 1:
        sharpe = np.mean(daily_vals) / (np.std(daily_vals) + 1e-9) * np.sqrt(252)
    else:
        sharpe = 0

    green_days = sum(1 for v in daily_vals if v > 0)
    red_days = sum(1 for v in daily_vals if v < 0)

    exit_reasons = {}
    for t in all_trades:
        r = t['exit_reason']
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    avg_mfe = np.mean([t['mfe'] for t in all_trades])
    avg_mae = np.mean([t['mae'] for t in all_trades])
    avg_fill_lat = np.mean([t['fill_latency_ns'] for t in all_trades]) / 1e9

    return {
        'n_trades': n,
        'trades_per_day': n / n_days,
        'net_ticks': total_net,
        'per_trade': avg_net,
        'win_rate': wr,
        'sharpe': sharpe,
        'green_days': green_days,
        'red_days': red_days,
        'n_days': n_days,
        'exit_reasons': exit_reasons,
        'avg_mfe': avg_mfe,
        'avg_mae': avg_mae,
        'avg_fill_latency_s': avg_fill_lat,
    }


def run_permutation(day_arrays_list, preds_list, raw_indices_list,
                    hold_s, tp, sl, cancel_s, side_filter, n_perms=100):
    """Run permutation test: randomize prediction directions."""
    # Real result
    real_trades = []
    for da, p, ri in zip(day_arrays_list, preds_list, raw_indices_list):
        real_trades.append(simulate_config(da, p, ri, hold_s, tp, sl, cancel_s, side_filter))

    real_metrics = compute_metrics_from_trades(real_trades, len(day_arrays_list))
    if real_metrics is None or real_metrics['n_trades'] < 20:
        return None

    real_sharpe = real_metrics['sharpe']

    # Permutation: flip signs randomly
    n_better = 0
    for _ in range(n_perms):
        perm_trades = []
        for da, p, ri in zip(day_arrays_list, preds_list, raw_indices_list):
            # Random sign flip
            flipped = p * np.random.choice([-1, 1], size=len(p))
            perm_trades.append(simulate_config(da, flipped, ri, hold_s, tp, sl, cancel_s, side_filter))

        perm_metrics = compute_metrics_from_trades(perm_trades, len(day_arrays_list))
        if perm_metrics and perm_metrics['sharpe'] >= real_sharpe:
            n_better += 1

    p_value = (n_better + 1) / (n_perms + 1)
    return p_value, real_metrics


def main():
    t0 = time.time()
    print("=" * 70)
    print("TICK REPLAY v17 — ALIGNMENT-FIXED SWEEP")
    print("=" * 70)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Find matching dates (must have: raw MBO, preprocessed, predictions)
    pred_dates = sorted([f.replace('oot_', '').replace('.npz', '')
                        for f in os.listdir(PRED_DIR) if f.startswith('oot_')])

    matched = []
    for d in pred_dates:
        mbo_path = os.path.join(MBO_DIR, f'glbx-mdp3-{d}.mbo.dbn.zst')
        proc_path = os.path.join(PROC_DIR, f'{d}_mbo_events.npz')
        if os.path.exists(mbo_path) and os.path.exists(proc_path):
            matched.append(d)

    print(f"\nMatched dates: {len(matched)} / {len(pred_dates)}")
    print(f"Dates: {matched}")

    # PHASE 1: Extract + align all days
    print(f"\n{'='*70}")
    print("PHASE 1: EXTRACT + ALIGN")
    print(f"{'='*70}")

    day_data = {}  # date -> {arrays, preds_per_head, raw_indices}
    phase1_t0 = time.time()

    # Screen on first 5 dates, expand if promising
    screen_dates = matched[:5]
    print(f"Screening on {len(screen_dates)} dates: {screen_dates}")

    for date in screen_dates:
        mbo_path = os.path.join(MBO_DIR, f'glbx-mdp3-{date}.mbo.dbn.zst')

        arrays = extract_day_arrays(mbo_path, date)
        if arrays is None:
            continue

        n_ev = int(arrays['n_events'])
        raw_ts = arrays['ts_event'][:n_ev]

        # Load predictions
        pred_data = np.load(os.path.join(PRED_DIR, f'oot_{date}.npz'), allow_pickle=True)

        # Get aligned indices
        preds_dict = {}
        for head in SIGNAL_HEADS:
            if head not in pred_data:
                continue
            preds = pred_data[head]
            n_preds = len(preds)

            result = get_aligned_pred_indices(date, n_preds, raw_ts)
            if result is None:
                continue

            raw_indices, valid_mask = result
            preds_valid = preds[valid_mask][:len(raw_indices)]

            preds_dict[head] = (preds_valid, raw_indices)

        if preds_dict:
            day_data[date] = {'arrays': arrays, 'preds': preds_dict}
            print(f"  {date}: {n_ev:,} events, {len(preds_dict)} heads aligned")

        del pred_data
        gc.collect()

    phase1_time = time.time() - phase1_t0
    print(f"\nPhase 1: {phase1_time:.0f}s ({phase1_time/60:.1f} min)")

    # PHASE 1.5: Quick IC verification
    print(f"\n{'='*70}")
    print("PHASE 1.5: IC VERIFICATION (ALIGNMENT CHECK)")
    print(f"{'='*70}")

    for date, dd in day_data.items():
        arrays = dd['arrays']
        n_ev = int(arrays['n_events'])
        raw_ts = arrays['ts_event'][:n_ev]
        bid = arrays['bbo_bid'][:n_ev]
        ask = arrays['bbo_ask'][:n_ev]

        for head, (preds, raw_indices) in dd['preds'].items():
            # Get mid prices at prediction points and 1s later
            mid_now = (bid[raw_indices] + ask[raw_indices]) / 2
            valid = mid_now > 0

            horizon_ns = int(1e9)  # 1s
            fut_indices = np.searchsorted(raw_ts, raw_ts[raw_indices] + horizon_ns)
            fut_indices = np.clip(fut_indices, 0, n_ev - 1)
            mid_fut = (bid[fut_indices] + ask[fut_indices]) / 2

            both_valid = valid & (mid_fut > 0)
            if both_valid.sum() < 100:
                continue

            ret_ticks = (mid_fut[both_valid] - mid_now[both_valid]) / TICK_SIZE
            ic, _ = spearmanr(preds[both_valid], ret_ticks)
            print(f"  {date} {head}: IC_1s={ic:.3f} (N={both_valid.sum()})")

    # PHASE 2: CONFIG SWEEP
    print(f"\n{'='*70}")
    print("PHASE 2: CONFIG SWEEP")
    print(f"{'='*70}")

    QUANTILES = [0.03, 0.05, 0.10, 0.20]

    CONFIGS = [
        # (hold_s, tp, sl, cancel_s, side_filter)
        # Short-only (strongest signal per decay analysis)
        (5,  3, 3, 10, 'short_only'),
        (5,  4, 3, 10, 'short_only'),
        (10, 3, 5, 15, 'short_only'),
        (10, 4, 3, 15, 'short_only'),
        (10, 5, 3, 15, 'short_only'),
        (15, 5, 3, 15, 'short_only'),
        (15, 6, 3, 15, 'short_only'),
        (30, 6, 4, 20, 'short_only'),
        (30, 8, 5, 20, 'short_only'),
        (5,  99, 99, 10, 'short_only'),
        (10, 99, 99, 15, 'short_only'),
        (15, 99, 99, 15, 'short_only'),
        (30, 99, 99, 20, 'short_only'),

        # Both sides
        (5,  3, 3, 10, 'both'),
        (10, 4, 3, 15, 'both'),
        (15, 5, 3, 15, 'both'),
        (30, 6, 4, 20, 'both'),
        (30, 8, 5, 20, 'both'),
        (5,  99, 99, 10, 'both'),
        (10, 99, 99, 15, 'both'),
        (30, 99, 99, 20, 'both'),

        # Asymmetric (tight SL, wide TP)
        (15, 6, 2, 15, 'short_only'),
        (20, 8, 3, 20, 'short_only'),
        (30, 10, 3, 25, 'short_only'),
        (15, 6, 2, 15, 'both'),
        (20, 8, 3, 20, 'both'),

        # Long only (control)
        (10, 4, 3, 15, 'long_only'),
        (30, 99, 99, 20, 'long_only'),
    ]

    total = len(SIGNAL_HEADS) * len(QUANTILES) * len(CONFIGS)
    print(f"  {len(SIGNAL_HEADS)} heads × {len(QUANTILES)} quantiles × {len(CONFIGS)} configs = {total}")

    all_results = []
    promising = []
    sweep_t0 = time.time()
    config_count = 0

    dates_list = sorted(day_data.keys())
    n_days = len(dates_list)

    for head in SIGNAL_HEADS:
        has_head = any(head in dd['preds'] for dd in day_data.values())
        if not has_head:
            print(f"\n  Skipping {head} (not in predictions)")
            continue

        print(f"\n  Head: {head}")

        for q in QUANTILES:
            # Get threshold from combined predictions
            all_preds_for_q = []
            for date in dates_list:
                if head in day_data[date]['preds']:
                    p, _ = day_data[date]['preds'][head]
                    all_preds_for_q.append(p)

            combined = np.concatenate(all_preds_for_q)
            threshold = np.quantile(np.abs(combined), 1 - q)

            for hold_s, tp, sl, cancel_s, side_filter in CONFIGS:
                config_count += 1

                # Filter predictions by threshold
                day_trades_all = []
                for date in dates_list:
                    if head not in day_data[date]['preds']:
                        day_trades_all.append([])
                        continue

                    p, ri = day_data[date]['preds'][head]

                    # Apply threshold: only trade extreme predictions
                    mask = np.abs(p) >= threshold
                    p_filtered = p[mask]
                    ri_filtered = ri[mask]

                    if len(p_filtered) == 0:
                        day_trades_all.append([])
                        continue

                    trades = simulate_config(
                        day_data[date]['arrays'], p_filtered, ri_filtered,
                        hold_s, tp, sl, cancel_s, side_filter
                    )
                    day_trades_all.append(trades)

                metrics = compute_metrics_from_trades(day_trades_all, n_days)

                if metrics is None:
                    continue

                label = f"{head}|q{q}|h{hold_s}tp{tp}sl{sl}{'_s' if side_filter=='short_only' else '_l' if side_filter=='long_only' else ''}"

                result = {
                    'label': label,
                    'head': head,
                    'quantile': q,
                    'threshold': float(threshold),
                    'hold_s': hold_s,
                    'tp': tp,
                    'sl': sl,
                    'cancel_s': cancel_s,
                    'side_filter': side_filter,
                    **metrics,
                }
                all_results.append(result)

                # Mark promising
                marker = ' '
                if metrics['sharpe'] > 0:
                    marker = '+'
                if metrics['sharpe'] > 0.5 and metrics['win_rate'] > 0.45:
                    marker = 'Y'
                    promising.append(result)

                if config_count % 50 == 0 or marker != ' ':
                    print(f"    [{config_count:>4}] {marker} {label:50s}: "
                          f"{metrics['net_ticks']:+.0f}t ({metrics['per_trade']:+.3f}t/tr) "
                          f"n={metrics['n_trades']} WR={metrics['win_rate']:.1%} "
                          f"Sh={metrics['sharpe']:.2f} G/R={metrics['green_days']}/{metrics['red_days']} "
                          f"MFE={metrics['avg_mfe']:.1f} MAE={metrics['avg_mae']:.1f} "
                          f"fill={metrics['avg_fill_latency_s']:.1f}s")

    sweep_time = time.time() - sweep_t0

    # PHASE 3: Permutation tests on promising configs
    print(f"\n{'='*70}")
    print("PHASE 3: PERMUTATION TESTS")
    print(f"{'='*70}")

    validated = []
    if promising:
        print(f"  Testing {len(promising)} promising configs...")
        for cfg in promising:
            head = cfg['head']
            q = cfg['quantile']

            # Rebuild filtered predictions
            all_preds_for_q = []
            for date in dates_list:
                if head in day_data[date]['preds']:
                    p, _ = day_data[date]['preds'][head]
                    all_preds_for_q.append(p)
            threshold = np.quantile(np.abs(np.concatenate(all_preds_for_q)), 1 - q)

            da_list = []
            p_list = []
            ri_list = []
            for date in dates_list:
                if head not in day_data[date]['preds']:
                    continue
                p, ri = day_data[date]['preds'][head]
                mask = np.abs(p) >= threshold
                da_list.append(day_data[date]['arrays'])
                p_list.append(p[mask])
                ri_list.append(ri[mask])

            result = run_permutation(
                da_list, p_list, ri_list,
                cfg['hold_s'], cfg['tp'], cfg['sl'], cfg['cancel_s'], cfg['side_filter'],
                n_perms=200
            )

            if result is not None:
                p_val, real_metrics = result
                cfg['permutation_p'] = p_val
                print(f"    {cfg['label']}: Sharpe={cfg['sharpe']:.2f}, p={p_val:.3f} "
                      f"{'✅ PASS' if p_val < 0.05 else '❌ FAIL'}")

                if p_val < 0.05:
                    validated.append(cfg)
    else:
        # Check top N by Sharpe even if none passed "promising" threshold
        if all_results:
            sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
            top_n = sorted_results[:10]
            print(f"  No promising configs. Testing top 10 by Sharpe...")
            for cfg in top_n:
                if cfg['n_trades'] < 10:
                    continue
                print(f"    {cfg['label']}: Sharpe={cfg['sharpe']:.2f}, "
                      f"net={cfg['per_trade']:+.3f}t/tr, n={cfg['n_trades']}")

    # PHASE 4: Expand to all dates if validated
    if validated and len(matched) > len(screen_dates):
        print(f"\n{'='*70}")
        print(f"PHASE 4: EXPANDING {len(validated)} CONFIGS TO ALL {len(matched)} DATES")
        print(f"{'='*70}")
        # TODO: expand to remaining dates

    # Save results
    total_time = time.time() - t0

    output = {
        'generated': time.strftime('%Y-%m-%d %H:%M'),
        'version': 'v17_aligned',
        'alignment_fix': 'preprocessed_timestamp_to_raw_event',
        'dates_screened': list(day_data.keys()),
        'total_dates_available': len(matched),
        'phase1_seconds': phase1_time,
        'phase2_seconds': sweep_time,
        'total_seconds': total_time,
        'results': all_results,
        'promising': promising,
        'validated': validated,
    }

    out_path = os.path.join(OUTPUT_DIR, 'v17_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Phase 1 (extract+align): {phase1_time:.0f}s")
    print(f"  Phase 2 (sweep):         {sweep_time:.0f}s")
    print(f"  Total:                   {total_time:.0f}s")
    print(f"  Configs tested:          {len(all_results)}")

    n_pos = sum(1 for r in all_results if r.get('sharpe', -999) > 0)
    print(f"  Positive Sharpe:         {n_pos}")
    print(f"  Promising (Sh>0.5):      {len(promising)}")
    print(f"  Validated (perm p<0.05): {len(validated)}")

    if all_results:
        best = max(all_results, key=lambda x: x.get('sharpe', -999))
        print(f"\n  Best config: {best['label']}")
        print(f"    Sharpe={best['sharpe']:.2f}, per_trade={best['per_trade']:+.3f}t, "
              f"WR={best['win_rate']:.1%}, trades/day={best['trades_per_day']:.1f}")

    print(f"\n  Results: {out_path}")


if __name__ == '__main__':
    main()
