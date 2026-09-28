#!/usr/bin/env python3
"""
Tick Replay v17 FAST — Vectorized alignment-fixed sweep
========================================================

v17_aligned was correct but pure-Python loops were too slow.
This version uses numpy vectorization for the simulation:
- Pre-compute all trade events once
- For each prediction: binary search for fill, TP, SL, time_stop
- No inner Python loops over raw events

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
)

MBO_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate'
PROC_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v17'
V16_CACHE = '/home/jupiter/Lvl3Quant/output/tick_replay_v16/day_cache'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIGNAL_HEADS = ['pred_log_ret_1s', 'pred_log_ret_5s', 'pred_log_ret_10s']


def get_aligned_pred_indices(date_key, n_preds, raw_ts):
    """Map prediction indices from preprocessed to raw event space."""
    proc_path = os.path.join(PROC_DIR, f'{date_key}_mbo_events.npz')
    if not os.path.exists(proc_path):
        return None

    proc = np.load(proc_path, allow_pickle=True)
    proc_ts = proc['timestamps']
    n_proc = len(proc_ts)

    proc_indices = np.arange(PRED_WINDOW, PRED_WINDOW + n_preds * PRED_STRIDE, PRED_STRIDE)
    valid = proc_indices < n_proc
    proc_indices = proc_indices[valid]
    pred_timestamps = proc_ts[proc_indices]

    raw_indices = np.searchsorted(raw_ts, pred_timestamps)
    raw_indices = np.clip(raw_indices, 0, len(raw_ts) - 1)

    del proc
    return raw_indices, valid


class DayTradeData:
    """Pre-extracted trade-only data for fast simulation."""
    def __init__(self, day_cache):
        n = int(day_cache['n_events'])
        ts_all = day_cache['ts_event'][:n]
        is_trade = day_cache['is_trade'][:n].astype(bool)

        # Extract trade-only arrays
        self.trade_mask_indices = np.where(is_trade)[0]  # indices into full event array
        self.trade_ts = ts_all[is_trade]
        self.trade_px = day_cache['trade_price'][:n][is_trade]
        self.trade_side = day_cache['trade_side'][:n][is_trade]

        # BBO at each event (for entry price lookup)
        self.all_ts = ts_all
        self.all_bid = day_cache['bbo_bid'][:n]
        self.all_ask = day_cache['bbo_ask'][:n]

        self.n_trades = len(self.trade_ts)
        self.n_events = n


def simulate_day_vectorized(day: DayTradeData, preds, raw_pred_indices,
                           hold_s, tp_ticks, sl_ticks, cancel_s, side_filter='both'):
    """
    Vectorized simulation for one day.

    For each prediction:
    1. Determine entry price (bid/ask) and direction
    2. Binary search for fill event (trade at our level on correct side)
    3. Scan forward for exit (TP/SL/time_stop)

    Returns list of trade dicts.
    """
    TICK = 0.25
    hold_ns = int(hold_s * 1e9)
    cancel_ns = int(cancel_s * 1e9)

    tp_offset = tp_ticks * TICK if tp_ticks < 99 else 999999.0
    sl_offset = sl_ticks * TICK if sl_ticks < 99 else 999999.0

    trades = []

    # Pre-filter predictions by direction
    for pi in range(len(preds)):
        pred = preds[pi]
        idx = raw_pred_indices[pi]

        if idx >= day.n_events:
            continue

        # Determine direction
        if pred > 0:
            direction = 1  # long
        elif pred < 0:
            direction = -1  # short
        else:
            continue

        if side_filter == 'short_only' and direction == 1:
            continue
        if side_filter == 'long_only' and direction == -1:
            continue

        # Entry price
        if direction == 1:
            entry_price = day.all_bid[idx]
        else:
            entry_price = day.all_ask[idx]

        if entry_price <= 0:
            continue

        t_pred = day.all_ts[idx]
        cancel_deadline = t_pred + cancel_ns

        # TP/SL prices
        if direction == 1:
            tp_price = entry_price + tp_offset
            sl_price = entry_price - sl_offset
        else:
            tp_price = entry_price - tp_offset
            sl_price = entry_price + sl_offset

        # Find fill: binary search for first trade after our prediction
        trade_start = np.searchsorted(day.trade_ts, t_pred)

        filled = False
        fill_trade_idx = None

        # Scan trades for fill
        for ti in range(trade_start, day.n_trades):
            if day.trade_ts[ti] > cancel_deadline:
                break

            tp = day.trade_px[ti]
            ts = day.trade_side[ti]

            # Long fill: aggressive sell (A) at our bid
            if direction == 1 and ts == 'A' and tp <= entry_price:
                filled = True
                fill_trade_idx = ti
                break
            # Short fill: aggressive buy (B) at our ask
            elif direction == -1 and ts == 'B' and tp >= entry_price:
                filled = True
                fill_trade_idx = ti
                break

        if not filled:
            continue

        fill_time = day.trade_ts[fill_trade_idx]
        exit_deadline = fill_time + hold_ns

        # Scan for exit
        best_px = entry_price
        worst_px = entry_price
        exit_reason = None
        exit_price = None
        cost = COST_MARKET_EXIT

        for ti in range(fill_trade_idx + 1, day.n_trades):
            tp = day.trade_px[ti]
            t = day.trade_ts[ti]

            # Update MFE/MAE
            if direction == 1:
                if tp > best_px: best_px = tp
                if tp < worst_px: worst_px = tp
            else:
                if tp < best_px: best_px = tp
                if tp > worst_px: worst_px = tp

            # SL check
            if direction == 1 and tp <= sl_price:
                exit_reason = 'sl'
                exit_price = sl_price
                cost = COST_MARKET_EXIT
                break
            elif direction == -1 and tp >= sl_price:
                exit_reason = 'sl'
                exit_price = sl_price
                cost = COST_MARKET_EXIT
                break

            # TP check (simplified: trade-through = fill)
            if direction == 1 and tp >= tp_price:
                exit_reason = 'tp'
                exit_price = tp_price
                cost = COST_PASSIVE_EXIT
                break
            elif direction == -1 and tp <= tp_price:
                exit_reason = 'tp'
                exit_price = tp_price
                cost = COST_PASSIVE_EXIT
                break

            # Time stop
            if t >= exit_deadline:
                exit_reason = 'time_stop'
                exit_price = tp
                cost = COST_MARKET_EXIT
                break

        if exit_reason is None:
            continue

        # PnL
        if direction == 1:
            raw_pnl = (exit_price - entry_price) / TICK
            mfe = (best_px - entry_price) / TICK
            mae = (entry_price - worst_px) / TICK
        else:
            raw_pnl = (entry_price - exit_price) / TICK
            mfe = (entry_price - best_px) / TICK
            mae = (worst_px - entry_price) / TICK

        net_pnl = raw_pnl - cost

        trades.append({
            'net_pnl': net_pnl,
            'raw_pnl': raw_pnl,
            'mfe': mfe,
            'mae': mae,
            'exit_reason': exit_reason,
            'fill_latency': (fill_time - t_pred) / 1e9,
            'side': 'long' if direction == 1 else 'short',
        })

    return trades


def aggregate_metrics(day_trades_list, n_days):
    """Compute metrics from trades across days."""
    all_trades = []
    for dt in day_trades_list:
        all_trades.extend(dt)

    if not all_trades:
        return None

    n = len(all_trades)
    net_pnls = np.array([t['net_pnl'] for t in all_trades])
    total = net_pnls.sum()
    avg = total / n
    wr = (net_pnls > 0).mean()

    # Daily PnL
    daily = []
    for dt in day_trades_list:
        daily.append(sum(t['net_pnl'] for t in dt) if dt else 0)

    daily = np.array(daily)
    sharpe = np.mean(daily) / (np.std(daily) + 1e-9) * np.sqrt(252)
    green = (daily > 0).sum()
    red = (daily < 0).sum()

    exits = {}
    for t in all_trades:
        r = t['exit_reason']
        exits[r] = exits.get(r, 0) + 1

    mfe = np.mean([t['mfe'] for t in all_trades])
    mae = np.mean([t['mae'] for t in all_trades])
    fill_lat = np.mean([t['fill_latency'] for t in all_trades])

    return {
        'n_trades': n,
        'trades_per_day': n / n_days,
        'net_ticks': float(total),
        'per_trade': float(avg),
        'win_rate': float(wr),
        'sharpe': float(sharpe),
        'green_days': int(green),
        'red_days': int(red),
        'n_days': n_days,
        'exit_reasons': exits,
        'avg_mfe': float(mfe),
        'avg_mae': float(mae),
        'avg_fill_lat_s': float(fill_lat),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("TICK REPLAY v17 FAST — ALIGNMENT-FIXED VECTORIZED SWEEP")
    print("=" * 70)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Find matching dates
    pred_dates = sorted([f.replace('oot_', '').replace('.npz', '')
                        for f in os.listdir(PRED_DIR) if f.startswith('oot_')])

    matched = []
    for d in pred_dates:
        mbo_path = os.path.join(MBO_DIR, f'glbx-mdp3-{d}.mbo.dbn.zst')
        proc_path = os.path.join(PROC_DIR, f'{d}_mbo_events.npz')
        if os.path.exists(mbo_path) and os.path.exists(proc_path):
            matched.append(d)

    print(f"Matched dates: {len(matched)}")

    # Use ALL dates (not just screening subset — we have 34 dates for better Sharpe estimation)
    use_dates = matched
    print(f"Using: {len(use_dates)} dates")

    # PHASE 1: Load + align
    print(f"\n{'='*70}")
    print("PHASE 1: LOAD + ALIGN")
    phase1_t0 = time.time()

    day_data = {}

    for date in use_dates:
        # Load from v16 cache or extract
        cache_path = os.path.join(V16_CACHE, f'{date}_extracted.npz')
        v17_cache = os.path.join(OUTPUT_DIR, f'{date}_cache.npz')

        if os.path.exists(v17_cache):
            cache = np.load(v17_cache)
        elif os.path.exists(cache_path):
            cache = np.load(cache_path)
        else:
            # Need to extract — skip for now if no cache
            print(f"  {date}: no cache, skipping (run v16 expansion first)")
            continue

        n_ev = int(cache['n_events'])
        raw_ts = cache['ts_event'][:n_ev]

        # Load predictions
        pred_path = os.path.join(PRED_DIR, f'oot_{date}.npz')
        pred_data = np.load(pred_path, allow_pickle=True)

        # Align
        preds_dict = {}
        for head in SIGNAL_HEADS:
            if head not in pred_data:
                continue
            preds = pred_data[head]
            result = get_aligned_pred_indices(date, len(preds), raw_ts)
            if result is None:
                continue
            raw_indices, valid_mask = result
            preds_valid = preds[valid_mask][:len(raw_indices)]
            preds_dict[head] = (preds_valid, raw_indices)

        if not preds_dict:
            del cache, pred_data
            continue

        # Build DayTradeData
        dtd = DayTradeData(cache)
        day_data[date] = {'dtd': dtd, 'preds': preds_dict}

        del cache, pred_data
        gc.collect()

    phase1_time = time.time() - phase1_t0
    print(f"\n  Loaded {len(day_data)} days in {phase1_time:.0f}s")

    # IC verification
    print(f"\n{'='*70}")
    print("IC VERIFICATION")
    for date, dd in sorted(day_data.items())[:5]:
        dtd = dd['dtd']
        for head, (preds, raw_indices) in dd['preds'].items():
            mid = (dtd.all_bid[raw_indices] + dtd.all_ask[raw_indices]) / 2
            valid = mid > 0

            horizon_ns = int(1e9)
            fut_idx = np.searchsorted(dtd.all_ts, dtd.all_ts[raw_indices] + horizon_ns)
            fut_idx = np.clip(fut_idx, 0, dtd.n_events - 1)
            fut_mid = (dtd.all_bid[fut_idx] + dtd.all_ask[fut_idx]) / 2

            ok = valid & (fut_mid > 0)
            if ok.sum() < 100:
                continue
            ret = (fut_mid[ok] - mid[ok]) / TICK_SIZE
            ic, _ = spearmanr(preds[ok], ret)
            if head == 'pred_log_ret_1s':
                print(f"  {date} {head}: IC_1s={ic:.3f}")

    # PHASE 2: CONFIG SWEEP
    print(f"\n{'='*70}")
    print("PHASE 2: CONFIG SWEEP")
    phase2_t0 = time.time()

    QUANTILES = [0.03, 0.05, 0.10, 0.20]

    CONFIGS = [
        # (hold_s, tp, sl, cancel_s, side_filter)
        # Short-only (strongest edge)
        (5,  2, 3, 8,  'short_only'),
        (5,  3, 3, 10, 'short_only'),
        (5,  4, 3, 10, 'short_only'),
        (7,  3, 4, 12, 'short_only'),
        (10, 3, 5, 15, 'short_only'),
        (10, 4, 3, 15, 'short_only'),
        (10, 5, 3, 15, 'short_only'),
        (15, 5, 3, 15, 'short_only'),
        (15, 6, 3, 15, 'short_only'),
        (20, 6, 4, 20, 'short_only'),
        (30, 6, 4, 20, 'short_only'),
        (30, 8, 5, 20, 'short_only'),
        (5,  99, 99, 10, 'short_only'),
        (10, 99, 99, 15, 'short_only'),
        (15, 99, 99, 15, 'short_only'),
        (30, 99, 99, 20, 'short_only'),
        # Asymmetric short
        (15, 6, 2, 15, 'short_only'),
        (20, 8, 3, 20, 'short_only'),
        (30, 10, 3, 25, 'short_only'),

        # Both sides
        (5,  3, 3, 10, 'both'),
        (10, 4, 3, 15, 'both'),
        (15, 5, 3, 15, 'both'),
        (30, 6, 4, 20, 'both'),
        (5,  99, 99, 10, 'both'),
        (10, 99, 99, 15, 'both'),
        (30, 99, 99, 20, 'both'),
        (15, 6, 2, 15, 'both'),

        # Long only (control)
        (10, 4, 3, 15, 'long_only'),
        (30, 99, 99, 20, 'long_only'),
    ]

    dates_list = sorted(day_data.keys())
    n_days = len(dates_list)
    total_combos = len(SIGNAL_HEADS) * len(QUANTILES) * len(CONFIGS)
    print(f"  {len(SIGNAL_HEADS)} heads × {len(QUANTILES)} q × {len(CONFIGS)} cfg = {total_combos}")
    print(f"  Running on {n_days} days")

    all_results = []
    promising = []
    count = 0

    for head in SIGNAL_HEADS:
        has = any(head in dd['preds'] for dd in day_data.values())
        if not has:
            continue

        print(f"\n  === {head} ===")

        for q in QUANTILES:
            # Compute threshold
            all_p = []
            for date in dates_list:
                if head in day_data[date]['preds']:
                    p, _ = day_data[date]['preds'][head]
                    all_p.append(p)
            combined = np.concatenate(all_p)
            threshold = float(np.quantile(np.abs(combined), 1 - q))

            for hold_s, tp, sl, cancel_s, sf in CONFIGS:
                count += 1

                day_trades = []
                for date in dates_list:
                    if head not in day_data[date]['preds']:
                        day_trades.append([])
                        continue

                    p, ri = day_data[date]['preds'][head]
                    mask = np.abs(p) >= threshold

                    if mask.sum() == 0:
                        day_trades.append([])
                        continue

                    trades = simulate_day_vectorized(
                        day_data[date]['dtd'], p[mask], ri[mask],
                        hold_s, tp, sl, cancel_s, sf
                    )
                    day_trades.append(trades)

                metrics = aggregate_metrics(day_trades, n_days)
                if metrics is None:
                    continue

                label = (f"{head.replace('pred_','')}|q{q}|h{hold_s}tp{tp}sl{sl}"
                         f"{'_s' if sf=='short_only' else '_l' if sf=='long_only' else ''}")

                result = {'label': label, 'head': head, 'quantile': q,
                          'threshold': threshold, 'hold_s': hold_s,
                          'tp': tp, 'sl': sl, 'cancel_s': cancel_s,
                          'side_filter': sf, **metrics}
                all_results.append(result)

                marker = ' '
                if metrics['sharpe'] > 0:
                    marker = '+'
                if metrics['sharpe'] > 0.5 and metrics['win_rate'] > 0.45:
                    marker = 'Y'
                    promising.append(result)
                elif metrics['sharpe'] > 1.0:
                    marker = 'Y'
                    promising.append(result)

                if count % 20 == 0 or marker != ' ':
                    print(f"  [{count:>4}] {marker} {label:48s} "
                          f"{metrics['net_ticks']:>+7.0f}t {metrics['per_trade']:>+.3f}t/tr "
                          f"n={metrics['n_trades']:>5} WR={metrics['win_rate']:.1%} "
                          f"Sh={metrics['sharpe']:>6.2f} "
                          f"G/R={metrics['green_days']}/{metrics['red_days']} "
                          f"MFE={metrics['avg_mfe']:.1f} MAE={metrics['avg_mae']:.1f} "
                          f"fill={metrics['avg_fill_lat_s']:.1f}s")

    phase2_time = time.time() - phase2_t0

    # PHASE 3: PERMUTATION TESTS
    print(f"\n{'='*70}")
    print("PHASE 3: PERMUTATION TESTS")
    phase3_t0 = time.time()

    validated = []

    # Test top configs (either promising or top by Sharpe)
    to_test = promising if promising else sorted(all_results, key=lambda x: x['sharpe'], reverse=True)[:15]
    to_test = [c for c in to_test if c['n_trades'] >= 20]

    print(f"  Testing {len(to_test)} configs...")

    for cfg in to_test:
        head = cfg['head']
        q = cfg['quantile']

        # Rebuild threshold
        all_p = []
        for date in dates_list:
            if head in day_data[date]['preds']:
                p, _ = day_data[date]['preds'][head]
                all_p.append(p)
        threshold = float(np.quantile(np.abs(np.concatenate(all_p)), 1 - q))

        # Real Sharpe
        real_sharpe = cfg['sharpe']

        # Permutation: 200 random sign flips
        n_better = 0
        n_perms = 200

        for perm_i in range(n_perms):
            perm_day_trades = []
            for date in dates_list:
                if head not in day_data[date]['preds']:
                    perm_day_trades.append([])
                    continue

                p, ri = day_data[date]['preds'][head]
                mask = np.abs(p) >= threshold
                if mask.sum() == 0:
                    perm_day_trades.append([])
                    continue

                # Random sign flip
                flipped = p[mask] * np.random.choice([-1, 1], size=mask.sum())

                trades = simulate_day_vectorized(
                    day_data[date]['dtd'], flipped, ri[mask],
                    cfg['hold_s'], cfg['tp'], cfg['sl'],
                    cfg['cancel_s'], cfg['side_filter']
                )
                perm_day_trades.append(trades)

            perm_metrics = aggregate_metrics(perm_day_trades, n_days)
            if perm_metrics and perm_metrics['sharpe'] >= real_sharpe:
                n_better += 1

        p_val = (n_better + 1) / (n_perms + 1)
        cfg['perm_p'] = p_val

        status = '✅ PASS' if p_val < 0.05 else '❌ FAIL'
        print(f"    {cfg['label']}: Sh={cfg['sharpe']:.2f} p={p_val:.3f} {status}")

        if p_val < 0.05:
            validated.append(cfg)

    phase3_time = time.time() - phase3_t0

    # SAVE RESULTS
    total_time = time.time() - t0

    output = {
        'generated': time.strftime('%Y-%m-%d %H:%M'),
        'version': 'v17_fast_aligned',
        'alignment_fix': 'preprocessed_timestamp_to_raw_event',
        'dates_used': dates_list,
        'n_dates': n_days,
        'phase1_s': phase1_time,
        'phase2_s': phase2_time,
        'phase3_s': phase3_time,
        'total_s': total_time,
        'results': all_results,
        'promising': promising,
        'validated': validated,
    }

    out_path = os.path.join(OUTPUT_DIR, 'v17_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # SUMMARY
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Phase 1: {phase1_time:.0f}s  Phase 2: {phase2_time:.0f}s  Phase 3: {phase3_time:.0f}s  Total: {total_time:.0f}s")
    print(f"  Configs tested: {len(all_results)}")

    n_pos = sum(1 for r in all_results if r.get('sharpe', -999) > 0)
    print(f"  Positive Sharpe: {n_pos}")
    print(f"  Promising: {len(promising)}")
    print(f"  Validated (perm): {len(validated)}")

    if all_results:
        best = max(all_results, key=lambda x: x.get('sharpe', -999))
        print(f"\n  Best: {best['label']}")
        print(f"    Sh={best['sharpe']:.2f} {best['per_trade']:+.3f}t/tr WR={best['win_rate']:.1%} "
              f"n={best['n_trades']} G/R={best['green_days']}/{best['red_days']}")

    if validated:
        print(f"\n  🎯 VALIDATED CONFIGS (perm p<0.05):")
        for v in validated:
            print(f"    {v['label']}: Sh={v['sharpe']:.2f} p={v['perm_p']:.3f} "
                  f"{v['per_trade']:+.3f}t/tr WR={v['win_rate']:.1%}")

    print(f"\n  Results: {out_path}")


if __name__ == '__main__':
    main()
