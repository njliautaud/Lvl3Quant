#!/usr/bin/env python3
"""
Tick Replay v18 FULL — Complete sweep on ALL cached days with permutation test
==============================================================================

Runs v17's vectorized simulation on ALL v16-cached days (30+), not just the 14
that v17 started with. Includes:
- All 3 signal heads (1s, 5s, 10s)
- 4 quantile thresholds (3%, 5%, 10%, 20%)
- 29 execution configs (short-only, both, long-only with varied TP/SL/hold)
- Permutation test on any promising configs (HC #659 mandatory)
- Day-stratified analysis (green/red/flat regime classification)

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
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v18'
V16_CACHE = '/home/jupiter/Lvl3Quant/output/tick_replay_v16/day_cache'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIGNAL_HEADS = ['pred_log_ret_1s', 'pred_log_ret_5s', 'pred_log_ret_10s']

# ES daily close data for regime classification
ES_DAILY_FILE = '/home/jupiter/Lvl3Quant/data/processed/es_daily_close.csv'


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

        self.trade_ts = ts_all[is_trade]
        self.trade_px = day_cache['trade_price'][:n][is_trade]
        self.trade_side = day_cache['trade_side'][:n][is_trade]

        self.all_ts = ts_all
        self.all_bid = day_cache['bbo_bid'][:n]
        self.all_ask = day_cache['bbo_ask'][:n]

        self.n_trades = len(self.trade_ts)
        self.n_events = n


def simulate_day_vectorized(day, preds, raw_pred_indices,
                           hold_s, tp_ticks, sl_ticks, cancel_s, side_filter='both'):
    """Simulate trades for one day given predictions and config."""
    TICK = 0.25
    hold_ns = int(hold_s * 1e9)
    cancel_ns = int(cancel_s * 1e9)

    tp_offset = tp_ticks * TICK if tp_ticks < 99 else 999999.0
    sl_offset = sl_ticks * TICK if sl_ticks < 99 else 999999.0

    trades = []

    for pi in range(len(preds)):
        pred = preds[pi]
        idx = raw_pred_indices[pi]

        if idx >= day.n_events:
            continue

        if pred > 0:
            direction = 1
        elif pred < 0:
            direction = -1
        else:
            continue

        if side_filter == 'short_only' and direction == 1:
            continue
        if side_filter == 'long_only' and direction == -1:
            continue

        if direction == 1:
            entry_price = day.all_bid[idx]
        else:
            entry_price = day.all_ask[idx]

        if entry_price <= 0:
            continue

        t_pred = day.all_ts[idx]
        cancel_deadline = t_pred + cancel_ns

        if direction == 1:
            tp_price = entry_price + tp_offset
            sl_price = entry_price - sl_offset
        else:
            tp_price = entry_price - tp_offset
            sl_price = entry_price + sl_offset

        # Find fill
        trade_start = np.searchsorted(day.trade_ts, t_pred)
        filled = False
        fill_trade_idx = None

        for ti in range(trade_start, day.n_trades):
            if day.trade_ts[ti] > cancel_deadline:
                break
            tp_val = day.trade_px[ti]
            ts_val = day.trade_side[ti]

            if direction == 1 and ts_val == 'A' and tp_val <= entry_price:
                filled = True
                fill_trade_idx = ti
                break
            elif direction == -1 and ts_val == 'B' and tp_val >= entry_price:
                filled = True
                fill_trade_idx = ti
                break

        if not filled:
            continue

        fill_time = day.trade_ts[fill_trade_idx]
        exit_deadline = fill_time + hold_ns

        best_px = entry_price
        worst_px = entry_price
        exit_reason = None
        exit_price = None
        cost = COST_MARKET_EXIT

        for ti in range(fill_trade_idx + 1, day.n_trades):
            tp_val = day.trade_px[ti]
            t = day.trade_ts[ti]

            if direction == 1:
                if tp_val > best_px: best_px = tp_val
                if tp_val < worst_px: worst_px = tp_val
            else:
                if tp_val < best_px: best_px = tp_val
                if tp_val > worst_px: worst_px = tp_val

            # SL check
            if direction == 1 and tp_val <= sl_price:
                exit_reason = 'sl'
                exit_price = sl_price
                cost = COST_MARKET_EXIT
                break
            elif direction == -1 and tp_val >= sl_price:
                exit_reason = 'sl'
                exit_price = sl_price
                cost = COST_MARKET_EXIT
                break

            # TP check
            if direction == 1 and tp_val >= tp_price:
                exit_reason = 'tp'
                exit_price = tp_price
                cost = COST_PASSIVE_EXIT
                break
            elif direction == -1 and tp_val <= tp_price:
                exit_reason = 'tp'
                exit_price = tp_price
                cost = COST_PASSIVE_EXIT
                break

            # Time stop
            if t >= exit_deadline:
                exit_reason = 'time_stop'
                exit_price = tp_val
                cost = COST_MARKET_EXIT
                break

        if exit_reason is None:
            continue

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

    daily = []
    for dt in day_trades_list:
        daily.append(sum(t['net_pnl'] for t in dt) if dt else 0)

    daily = np.array(daily)
    sharpe = np.mean(daily) / (np.std(daily) + 1e-9) * np.sqrt(252)
    sortino_denom = np.sqrt(np.mean(np.minimum(daily, 0)**2) + 1e-12) * np.sqrt(252)
    sortino = np.mean(daily) / sortino_denom if sortino_denom > 0 else 0

    green = (daily > 0).sum()
    red = (daily < 0).sum()

    # Profit factor
    gross_win = net_pnls[net_pnls > 0].sum() if (net_pnls > 0).any() else 0
    gross_loss = abs(net_pnls[net_pnls < 0].sum()) if (net_pnls < 0).any() else 1e-9
    pf = gross_win / gross_loss

    exits = {}
    for t in all_trades:
        r = t['exit_reason']
        exits[r] = exits.get(r, 0) + 1

    mfe = np.mean([t['mfe'] for t in all_trades])
    mae = np.mean([t['mae'] for t in all_trades])
    fill_lat = np.mean([t['fill_latency'] for t in all_trades])

    # Side breakdown
    shorts = [t for t in all_trades if t['side'] == 'short']
    longs = [t for t in all_trades if t['side'] == 'long']
    short_wr = np.mean([t['net_pnl'] > 0 for t in shorts]) if shorts else 0
    long_wr = np.mean([t['net_pnl'] > 0 for t in longs]) if longs else 0

    return {
        'n_trades': n,
        'trades_per_day': n / n_days,
        'net_ticks': float(total),
        'per_trade': float(avg),
        'win_rate': float(wr),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'profit_factor': float(pf),
        'green_days': int(green),
        'red_days': int(red),
        'n_days': n_days,
        'exit_reasons': exits,
        'avg_mfe': float(mfe),
        'avg_mae': float(mae),
        'avg_fill_lat_s': float(fill_lat),
        'n_short': len(shorts),
        'n_long': len(longs),
        'short_wr': float(short_wr),
        'long_wr': float(long_wr),
    }


def run_permutation_test(day_data, dates_list, head, quantile, hold_s, tp, sl,
                         cancel_s, sf, real_sharpe, n_perms=200):
    """Randomize prediction signs and check if real Sharpe beats random."""
    random_sharpes = []

    for perm in range(n_perms):
        rng = np.random.default_rng(perm)
        day_trades = []

        for date in dates_list:
            if head not in day_data[date]['preds']:
                day_trades.append([])
                continue

            p, ri = day_data[date]['preds'][head]
            # Random signs (same magnitude)
            p_rand = np.abs(p) * rng.choice([-1, 1], size=len(p))

            # Compute threshold from original distribution
            all_p = []
            for d in dates_list:
                if head in day_data[d]['preds']:
                    pp, _ = day_data[d]['preds'][head]
                    all_p.append(pp)
            combined = np.concatenate(all_p)
            threshold = float(np.quantile(np.abs(combined), 1 - quantile))

            mask = np.abs(p_rand) >= threshold
            if mask.sum() == 0:
                day_trades.append([])
                continue

            trades = simulate_day_vectorized(
                day_data[date]['dtd'], p_rand[mask], ri[mask],
                hold_s, tp, sl, cancel_s, sf
            )
            day_trades.append(trades)

        metrics = aggregate_metrics(day_trades, len(dates_list))
        if metrics:
            random_sharpes.append(metrics['sharpe'])
        else:
            random_sharpes.append(0)

    random_sharpes = np.array(random_sharpes)
    p_value = (random_sharpes >= real_sharpe).mean()

    return {
        'p_value': float(p_value),
        'real_sharpe': float(real_sharpe),
        'random_mean': float(random_sharpes.mean()),
        'random_std': float(random_sharpes.std()),
        'random_max': float(random_sharpes.max()),
        'n_perms': n_perms,
    }


def classify_day_regime(date_str):
    """Classify day as green/red/flat based on ES close-to-close."""
    # Simple approach: use the BBO data to get open/close from our cache
    # If no daily data file, return 'unknown'
    if os.path.exists(ES_DAILY_FILE):
        import pandas as pd
        df = pd.read_csv(ES_DAILY_FILE)
        row = df[df['date'] == int(date_str)]
        if len(row) > 0:
            ret = row.iloc[0].get('return_pct', 0)
            if ret > 0.3:
                return 'green'
            elif ret < -0.3:
                return 'red'
            else:
                return 'flat'
    return 'unknown'


def main():
    t0 = time.time()
    print("=" * 70)
    print("TICK REPLAY v18 FULL — ALL-DAY SWEEP + PERMUTATION + REGIME")
    print("=" * 70)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check how many days are cached
    cached_files = sorted([f for f in os.listdir(V16_CACHE) if f.endswith('_extracted.npz')])
    cached_dates = [f.replace('_extracted.npz', '') for f in cached_files]
    print(f"  v16 cache: {len(cached_dates)} days available")

    if len(cached_dates) < 20:
        print(f"  WARNING: Only {len(cached_dates)} days. Waiting for v16 to finish extraction.")
        print(f"  Need at least 20 days for meaningful Sharpe estimation.")
        # Wait and poll
        while len(cached_dates) < 20:
            time.sleep(60)
            cached_files = sorted([f for f in os.listdir(V16_CACHE) if f.endswith('_extracted.npz')])
            cached_dates = [f.replace('_extracted.npz', '') for f in cached_files]
            print(f"  ... {len(cached_dates)} days cached")

    print(f"\n  Using {len(cached_dates)} days: {cached_dates[0]} to {cached_dates[-1]}")

    # PHASE 1: Load all cached days + align predictions
    print(f"\n{'='*70}")
    print("PHASE 1: LOAD + ALIGN ALL DAYS")
    phase1_t0 = time.time()

    day_data = {}
    for date in cached_dates:
        cache_path = os.path.join(V16_CACHE, f'{date}_extracted.npz')
        cache = np.load(cache_path)
        n_ev = int(cache['n_events'])
        raw_ts = cache['ts_event'][:n_ev]

        # Load predictions
        pred_path = os.path.join(PRED_DIR, f'oot_{date}.npz')
        if not os.path.exists(pred_path):
            del cache
            continue

        pred_data = np.load(pred_path, allow_pickle=True)

        # Align each head
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

        dtd = DayTradeData(cache)
        day_data[date] = {'dtd': dtd, 'preds': preds_dict, 'regime': classify_day_regime(date)}

        del cache, pred_data
        gc.collect()

    phase1_time = time.time() - phase1_t0
    n_days = len(day_data)
    dates_list = sorted(day_data.keys())
    print(f"\n  Loaded {n_days} days in {phase1_time:.0f}s")
    print(f"  Regime distribution: {sum(1 for d in day_data.values() if d['regime']=='green')} green, "
          f"{sum(1 for d in day_data.values() if d['regime']=='red')} red, "
          f"{sum(1 for d in day_data.values() if d['regime']=='flat')} flat, "
          f"{sum(1 for d in day_data.values() if d['regime']=='unknown')} unknown")

    # IC verification on first 5 days
    print(f"\n{'='*70}")
    print("IC VERIFICATION (first 5 days)")
    for date in dates_list[:5]:
        dd = day_data[date]
        dtd = dd['dtd']
        for head, (preds, raw_indices) in dd['preds'].items():
            if head != 'pred_log_ret_1s':
                continue
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
            print(f"  {date}: IC_1s={ic:.3f}")

    # PHASE 2: CONFIG SWEEP
    print(f"\n{'='*70}")
    print("PHASE 2: CONFIG SWEEP")
    phase2_t0 = time.time()

    QUANTILES = [0.03, 0.05, 0.10, 0.20]

    CONFIGS = [
        # Short-only (strongest edge from decay analysis)
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
        (5,  99, 99, 10, 'short_only'),   # time-stop only
        (10, 99, 99, 15, 'short_only'),
        (15, 99, 99, 15, 'short_only'),
        (30, 99, 99, 20, 'short_only'),
        # Asymmetric short (wide TP, tight SL)
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
        # Long only (control — should be WORSE per decay analysis)
        (10, 4, 3, 15, 'long_only'),
        (30, 99, 99, 20, 'long_only'),
    ]

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
                          'threshold': float(threshold), 'hold_s': hold_s,
                          'tp': tp, 'sl': sl, 'cancel_s': cancel_s,
                          'side_filter': sf, **metrics}
                all_results.append(result)

                # Promising criteria: Sharpe > 0.5 AND WR > 45% AND enough trades
                marker = ' '
                if metrics['sharpe'] > 0:
                    marker = '+'
                if (metrics['sharpe'] > 0.5 and metrics['win_rate'] > 0.45 and
                    metrics['n_trades'] >= 50):
                    marker = 'Y'
                    promising.append(result)
                elif metrics['sharpe'] > 1.0 and metrics['n_trades'] >= 30:
                    marker = 'Y'
                    promising.append(result)

                if count % 20 == 0 or marker == 'Y':
                    print(f"  [{count:>4}/{total_combos}] {marker} {label:48s} "
                          f"{metrics['net_ticks']:>+7.0f}t {metrics['per_trade']:>+.3f}t/tr "
                          f"n={metrics['n_trades']:>5} WR={metrics['win_rate']:.1%} "
                          f"Sh={metrics['sharpe']:>6.2f} So={metrics['sortino']:>5.2f} "
                          f"PF={metrics['profit_factor']:.2f} "
                          f"G/R={metrics['green_days']}/{metrics['red_days']} "
                          f"MFE={metrics['avg_mfe']:.1f} MAE={metrics['avg_mae']:.1f} "
                          f"fill={metrics['avg_fill_lat_s']:.1f}s")

    phase2_time = time.time() - phase2_t0
    print(f"\n  Sweep complete: {phase2_time:.0f}s ({phase2_time/60:.1f} min)")
    print(f"  {len(promising)} promising configs found (of {len(all_results)} tested)")

    # Sort by Sharpe
    all_results.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"\n  TOP 20 BY SHARPE:")
    for i, r in enumerate(all_results[:20]):
        print(f"  {i+1:>2}. {r['label']:48s} Sh={r['sharpe']:>6.2f} So={r['sortino']:>5.2f} "
              f"PF={r['profit_factor']:.2f} WR={r['win_rate']:.1%} n={r['n_trades']:>5} "
              f"+{r['net_ticks']:.0f}t G/R={r['green_days']}/{r['red_days']}")

    # PHASE 3: PERMUTATION TESTS (HC #659 MANDATORY)
    print(f"\n{'='*70}")
    print("PHASE 3: PERMUTATION TESTS")
    phase3_t0 = time.time()

    to_test = promising if promising else all_results[:15]
    to_test = [c for c in to_test if c['n_trades'] >= 20][:15]  # cap at 15

    print(f"  Testing {len(to_test)} configs with 200 permutations each...")
    validated = []

    for i, cfg in enumerate(to_test):
        print(f"\n  [{i+1}/{len(to_test)}] {cfg['label']} (Sharpe={cfg['sharpe']:.2f})")

        perm_result = run_permutation_test(
            day_data, dates_list, cfg['head'], cfg['quantile'],
            cfg['hold_s'], cfg['tp'], cfg['sl'], cfg['cancel_s'],
            cfg['side_filter'], cfg['sharpe'], n_perms=200
        )

        cfg['permutation'] = perm_result
        print(f"    p={perm_result['p_value']:.3f} | "
              f"real={perm_result['real_sharpe']:.2f} vs "
              f"random={perm_result['random_mean']:.2f}±{perm_result['random_std']:.2f} "
              f"(max={perm_result['random_max']:.2f})")

        if perm_result['p_value'] < 0.05:
            print(f"    ✅ PASSES PERMUTATION TEST (p < 0.05)")
            validated.append(cfg)
        else:
            print(f"    ❌ FAILS — random achieves similar/better Sharpe")

    phase3_time = time.time() - phase3_t0
    print(f"\n  Permutation tests: {phase3_time:.0f}s ({phase3_time/60:.1f} min)")
    print(f"  {len(validated)} of {len(to_test)} configs PASS (p < 0.05)")

    # PHASE 4: REGIME STRATIFICATION (HC #428 R1)
    print(f"\n{'='*70}")
    print("PHASE 4: REGIME STRATIFICATION")

    for cfg in validated:
        head = cfg['head']
        q = cfg['quantile']

        # Re-run day-by-day to get per-regime metrics
        green_pnl = []
        red_pnl = []
        flat_pnl = []

        for date in dates_list:
            regime = day_data[date]['regime']
            if head not in day_data[date]['preds']:
                pnl = 0
            else:
                p, ri = day_data[date]['preds'][head]
                all_p = []
                for d in dates_list:
                    if head in day_data[d]['preds']:
                        pp, _ = day_data[d]['preds'][head]
                        all_p.append(pp)
                combined = np.concatenate(all_p)
                threshold = float(np.quantile(np.abs(combined), 1 - q))
                mask = np.abs(p) >= threshold
                if mask.sum() == 0:
                    pnl = 0
                else:
                    trades = simulate_day_vectorized(
                        day_data[date]['dtd'], p[mask], ri[mask],
                        cfg['hold_s'], cfg['tp'], cfg['sl'], cfg['cancel_s'], cfg['side_filter']
                    )
                    pnl = sum(t['net_pnl'] for t in trades)

            if regime == 'green':
                green_pnl.append(pnl)
            elif regime == 'red':
                red_pnl.append(pnl)
            else:
                flat_pnl.append(pnl)

        green_sharpe = (np.mean(green_pnl) / (np.std(green_pnl) + 1e-9) * np.sqrt(252)
                       if green_pnl else 0)
        red_sharpe = (np.mean(red_pnl) / (np.std(red_pnl) + 1e-9) * np.sqrt(252)
                     if red_pnl else 0)

        # HC #428 R1: reject if |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) > 0.50
        max_sh = max(abs(green_sharpe), abs(red_sharpe), 1e-9)
        regime_gap = abs(green_sharpe - red_sharpe) / max_sh

        cfg['regime'] = {
            'green_sharpe': float(green_sharpe),
            'red_sharpe': float(red_sharpe),
            'regime_gap': float(regime_gap),
            'n_green': len(green_pnl),
            'n_red': len(red_pnl),
            'n_flat': len(flat_pnl),
            'r1_pass': regime_gap <= 0.50,
        }

        status = "✅ PASS" if regime_gap <= 0.50 else "❌ FAIL"
        print(f"\n  {cfg['label']}:")
        print(f"    Green Sharpe: {green_sharpe:.2f} ({len(green_pnl)} days)")
        print(f"    Red Sharpe:   {red_sharpe:.2f} ({len(red_pnl)} days)")
        print(f"    Regime gap:   {regime_gap:.2f} {status}")

    # FINAL SUMMARY
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    fully_validated = [c for c in validated if c.get('regime', {}).get('r1_pass', False)]

    print(f"\n  Total configs tested: {len(all_results)}")
    print(f"  Promising (Sharpe>0.5, WR>45%): {len(promising)}")
    print(f"  Pass permutation test: {len(validated)}")
    print(f"  Pass R1 regime gate: {len(fully_validated)}")
    print(f"\n  Days used: {n_days}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

    if fully_validated:
        print(f"\n  🎯 PRODUCTION CANDIDATES:")
        for c in fully_validated:
            print(f"    {c['label']}")
            print(f"      Sharpe={c['sharpe']:.2f} Sortino={c['sortino']:.2f} PF={c['profit_factor']:.2f}")
            print(f"      WR={c['win_rate']:.1%} Trades/day={c['trades_per_day']:.1f}")
            print(f"      Permutation p={c['permutation']['p_value']:.3f}")
            print(f"      Regime gap={c['regime']['regime_gap']:.2f} "
                  f"(green={c['regime']['green_sharpe']:.2f}, red={c['regime']['red_sharpe']:.2f})")
    elif validated:
        print(f"\n  ⚠️ {len(validated)} configs pass permutation but FAIL R1 regime gate.")
        print(f"     These are regime-dependent (work only in one direction).")
    else:
        print(f"\n  ❌ NO configs pass permutation test. Signal may not be tradeable")
        print(f"     at these execution parameters, or edge is too small after costs.")

    # Save results
    output = {
        'run_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_days': n_days,
        'dates': dates_list,
        'total_configs': len(all_results),
        'promising_count': len(promising),
        'validated_count': len(validated),
        'fully_validated_count': len(fully_validated),
        'top_20': all_results[:20],
        'validated': validated,
        'fully_validated': fully_validated,
        'total_time_s': time.time() - t0,
    }

    out_path = os.path.join(OUTPUT_DIR, 'v18_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
