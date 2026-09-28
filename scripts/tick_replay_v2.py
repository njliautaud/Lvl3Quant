#!/usr/bin/env python3
"""
Tick-Level FIFO Replay Engine v2
=================================
Key improvements over v1:
  - Loads MBO data ONCE per date, replays N times (10x faster)
  - Signal selection uses per-direction percentile (not broken z-score)
  - Reports longs/shorts separately
  - Proper permutation test
  - Numpy-vectorized where possible

Author: Claude (autonomous build, 2026-07-02)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tick_level_replay import (
    ROOT, RAW_MBO_DIR, SMART_V3_DIR, OOT_PRED_DIR, OUTPUT_DIR,
    ES_TICK_SIZE, ES_TICK_VALUE, ES_RT_COMMISSION_TICKS,
    COST_TP_EXIT, COST_SL_EXIT, COST_TIMEOUT_EXIT,
    WINDOW_SIZE, STRIDE,
    MAX_HOLD_SECONDS, CANCEL_WINDOW_SECONDS,
    compute_metrics, get_oot_dates,
)
import databento as dbn

logging.basicConfig(
    format="%(asctime)s [REPLAY-V2] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("REPLAY-V2")

CANCEL_NS = int(CANCEL_WINDOW_SECONDS * 1e9)
MAX_HOLD_NS = int(MAX_HOLD_SECONDS * 1e9)


# ─────────────────────────────────────────────
#  DATA LOADING (one-time per date)
# ─────────────────────────────────────────────

class DayData:
    """Pre-loaded data for one trading day. Load once, replay many times."""
    __slots__ = ['date_str', 'trade_ts', 'trade_prices', 'trade_bid', 'trade_ask',
                 'signal_ts', 'signal_dirs', 'signal_conf', 'n_signals',
                 'n_long_signals', 'n_short_signals']

    def __init__(self, date_str: str, top_pct: float = 0.10):
        self.date_str = date_str
        self._load_trades(date_str)
        self._load_predictions(date_str, top_pct)

    def _load_trades(self, date_str: str):
        """Load and cache trade tick data."""
        fname = f"glbx-mdp3-{date_str}.mbo.dbn.zst"
        fpath = RAW_MBO_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Raw MBO file not found: {fpath}")

        store = dbn.DBNStore.from_file(str(fpath))
        df = store.to_df()

        # Filter to ES front-month
        es_mask = df['symbol'].str.match(r'^ES[A-Z]\d$', na=False)
        es = df[es_mask]
        inst_counts = es['instrument_id'].value_counts()
        dominant = inst_counts.index[0]
        es = es[es['instrument_id'] == dominant]

        # RTH filter
        es_ts_ns = es['ts_event'].astype(np.int64)
        ts_utc = es['ts_event'].dt.tz_convert('UTC')
        hour = ts_utc.dt.hour
        minute = ts_utc.dt.minute
        rth_mask = ((hour > 13) | ((hour == 13) & (minute >= 30))) & (hour < 21)
        es_rth = es[rth_mask]
        rth_ts = es_ts_ns[rth_mask].values

        # Extract trades only
        actions = es_rth['action'].values
        trade_mask = actions == 'T'

        self.trade_ts = rth_ts[trade_mask]
        self.trade_prices = es_rth['price'].values[trade_mask].astype(np.float64)
        sides = es_rth['side'].values[trade_mask]

        # Build bid/ask from trade aggressor side
        n = len(self.trade_prices)
        self.trade_bid = np.empty(n, dtype=np.float64)
        self.trade_ask = np.empty(n, dtype=np.float64)
        for i in range(n):
            if sides[i] == 'A':
                self.trade_ask[i] = self.trade_prices[i]
                self.trade_bid[i] = self.trade_prices[i] - ES_TICK_SIZE
            elif sides[i] == 'B':
                self.trade_bid[i] = self.trade_prices[i]
                self.trade_ask[i] = self.trade_prices[i] + ES_TICK_SIZE
            else:
                self.trade_bid[i] = self.trade_prices[i] - ES_TICK_SIZE
                self.trade_ask[i] = self.trade_prices[i] + ES_TICK_SIZE

        # Free the big dataframe
        del df, es, es_rth, store
        gc.collect()

        log.info(f"  {date_str}: {n} trades loaded")

    def _load_predictions(self, date_str: str, top_pct: float):
        """Load predictions, select top signals per direction."""
        pred_path = OOT_PRED_DIR / f"oot_{date_str}.npz"
        mbo_path = SMART_V3_DIR / f"{date_str}_mbo_events.npz"

        pred_data = np.load(str(pred_path), allow_pickle=True)
        mbo_data = np.load(str(mbo_path), allow_pickle=True)

        pred_1s = pred_data['pred_log_ret_1s'].astype(np.float64)
        n_pred = len(pred_1s)
        mbo_ts = mbo_data['timestamps']
        n_events = len(mbo_ts)

        # Map predictions to event timestamps
        pred_indices = np.arange(WINDOW_SIZE - 1, n_events, STRIDE)[:n_pred]
        pred_timestamps = mbo_ts[pred_indices]

        direction = np.sign(pred_1s)
        confidence = np.abs(pred_1s)

        # Select top signals PER DIRECTION (balanced long/short)
        long_mask = pred_1s > 0
        short_mask = pred_1s < 0
        signal_mask = np.zeros(n_pred, dtype=bool)

        n_long = np.sum(long_mask)
        n_short = np.sum(short_mask)

        if n_long > 0:
            long_thresh = np.percentile(confidence[long_mask], 100 * (1 - top_pct))
            signal_mask[long_mask & (confidence >= long_thresh)] = True

        if n_short > 0:
            short_thresh = np.percentile(confidence[short_mask], 100 * (1 - top_pct))
            signal_mask[short_mask & (confidence >= short_thresh)] = True

        self.signal_ts = pred_timestamps[signal_mask]
        self.signal_dirs = direction[signal_mask].astype(np.int8)
        self.signal_conf = confidence[signal_mask]
        self.n_signals = int(np.sum(signal_mask))
        self.n_long_signals = int(np.sum(signal_mask & long_mask))
        self.n_short_signals = int(np.sum(signal_mask & short_mask))

        log.info(f"  {date_str}: {self.n_signals} signals (L:{self.n_long_signals}, S:{self.n_short_signals})")


# ─────────────────────────────────────────────
#  REPLAY ENGINE (pure computation, no I/O)
# ─────────────────────────────────────────────

def replay(day: DayData, tp_ticks: int, sl_ticks: int,
           override_dirs: np.ndarray = None) -> List[Dict]:
    """
    Replay one day using pre-loaded data.
    If override_dirs is provided, use those instead of model directions (for permutation test).
    """
    signal_ts = day.signal_ts
    signal_dirs = override_dirs if override_dirs is not None else day.signal_dirs

    if len(signal_ts) == 0:
        return []

    trade_ts = day.trade_ts
    trade_prices = day.trade_prices
    trade_bid = day.trade_bid
    trade_ask = day.trade_ask
    n_trades = len(trade_ts)

    tp_pts = tp_ticks * ES_TICK_SIZE
    sl_pts = sl_ticks * ES_TICK_SIZE

    completed = []

    # Active trade state
    active = False
    filled = False
    sig_dir = 0
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    cancel_ts = 0
    max_hold_ts = 0
    fill_ts = 0
    signal_t = 0

    signal_ptr = 0

    for ti in range(n_trades):
        t = trade_ts[ti]
        p = trade_prices[ti]

        if active:
            if not filled:
                # Check cancel
                if t >= cancel_ts:
                    active = False
                else:
                    # Check fill
                    if sig_dir == 1 and p <= entry_price:
                        filled = True
                        fill_ts = t
                        max_hold_ts = t + MAX_HOLD_NS
                    elif sig_dir == -1 and p >= entry_price:
                        filled = True
                        fill_ts = t
                        max_hold_ts = t + MAX_HOLD_NS

            if active and filled:
                exit_type = ''
                pnl = 0.0
                cost = 0.0
                exit_price = 0.0

                if sig_dir == 1:
                    if p >= tp_price:
                        exit_type = 'TP'
                        pnl = tp_ticks
                        cost = COST_TP_EXIT
                        exit_price = tp_price
                    elif p <= sl_price:
                        exit_type = 'SL'
                        pnl = -sl_ticks
                        cost = COST_SL_EXIT
                        exit_price = sl_price
                    elif t >= max_hold_ts:
                        exit_type = 'TIMEOUT'
                        mid = (trade_bid[ti] + trade_ask[ti]) / 2
                        pnl = (mid - entry_price) / ES_TICK_SIZE
                        cost = COST_TIMEOUT_EXIT
                        exit_price = p
                else:
                    if p <= tp_price:
                        exit_type = 'TP'
                        pnl = tp_ticks
                        cost = COST_TP_EXIT
                        exit_price = tp_price
                    elif p >= sl_price:
                        exit_type = 'SL'
                        pnl = -sl_ticks
                        cost = COST_SL_EXIT
                        exit_price = sl_price
                    elif t >= max_hold_ts:
                        exit_type = 'TIMEOUT'
                        mid = (trade_bid[ti] + trade_ask[ti]) / 2
                        pnl = (entry_price - mid) / ES_TICK_SIZE
                        cost = COST_TIMEOUT_EXIT
                        exit_price = p

                if exit_type:
                    net = pnl - cost
                    completed.append({
                        'date': day.date_str,
                        'signal_ts': int(signal_t),
                        'direction': int(sig_dir),
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_type': exit_type,
                        'pnl_ticks': round(pnl, 4),
                        'cost_ticks': round(cost, 4),
                        'net_ticks': round(net, 4),
                        'net_dollars': round(net * ES_TICK_VALUE, 2),
                        'fill_time_ns': int(fill_ts - signal_t),
                        'hold_time_ns': int(t - fill_ts),
                    })
                    active = False
                    filled = False

        # New signal
        if not active:
            while signal_ptr < len(signal_ts) and signal_ts[signal_ptr] <= t:
                sd = signal_dirs[signal_ptr]
                st = signal_ts[signal_ptr]
                signal_ptr += 1

                if sd == 0:
                    continue

                sig_dir = int(sd)
                signal_t = st
                if sig_dir == 1:
                    entry_price = trade_bid[ti]
                    tp_price = entry_price + tp_pts
                    sl_price = entry_price - sl_pts
                else:
                    entry_price = trade_ask[ti]
                    tp_price = entry_price - tp_pts
                    sl_price = entry_price + sl_pts

                cancel_ts = st + CANCEL_NS
                active = True
                filled = False
                break

            # Skip remaining signals at this timestamp
            if active:
                while signal_ptr < len(signal_ts) and signal_ts[signal_ptr] <= t:
                    signal_ptr += 1

    # Force-close EOD
    if active and filled and n_trades > 0:
        ti = n_trades - 1
        p = trade_prices[ti]
        mid = (trade_bid[ti] + trade_ask[ti]) / 2
        if sig_dir == 1:
            pnl = (mid - entry_price) / ES_TICK_SIZE
        else:
            pnl = (entry_price - mid) / ES_TICK_SIZE
        net = pnl - COST_TIMEOUT_EXIT
        completed.append({
            'date': day.date_str,
            'signal_ts': int(signal_t),
            'direction': int(sig_dir),
            'entry_price': entry_price,
            'exit_price': p,
            'exit_type': 'EOD',
            'pnl_ticks': round(pnl, 4),
            'cost_ticks': round(COST_TIMEOUT_EXIT, 4),
            'net_ticks': round(net, 4),
            'net_dollars': round(net * ES_TICK_VALUE, 2),
            'fill_time_ns': int(fill_ts - signal_t),
            'hold_time_ns': int(trade_ts[ti] - fill_ts),
        })

    return completed


# ─────────────────────────────────────────────
#  EXPERIMENT RUNNER
# ─────────────────────────────────────────────

def run_experiment(days: List[DayData], tp: int, sl: int,
                   n_perms: int = 20) -> Dict:
    """Run real + permutation replays on pre-loaded data."""
    config = f"TP{tp}_SL{sl}"
    log.info(f"\n--- {config} ---")

    # Real trades
    all_trades = []
    for day in days:
        trades = replay(day, tp, sl)
        all_trades.extend(trades)

    real_m = compute_metrics(all_trades, f"REAL {config}")
    real_pnl = real_m.get('total_pnl_ticks', 0)

    # Per-direction
    long_trades = [t for t in all_trades if t['direction'] == 1]
    short_trades = [t for t in all_trades if t['direction'] == -1]
    long_m = compute_metrics(long_trades, f"LONG")
    short_m = compute_metrics(short_trades, f"SHORT")

    # Per-day breakdown
    day_pnls = {}
    for t in all_trades:
        d = t['date']
        day_pnls[d] = day_pnls.get(d, 0) + t['net_ticks']
    green = sum(1 for v in day_pnls.values() if v > 0)
    red = sum(1 for v in day_pnls.values() if v < 0)

    # Permutation test
    perm_pnls = []
    perm_long_pnls = []
    perm_short_pnls = []

    for pi in range(n_perms):
        rng = np.random.default_rng(seed=pi * 1000 + 42)
        pt = []
        for day in days:
            dirs = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(day.signal_dirs))
            trades = replay(day, tp, sl, override_dirs=dirs)
            pt.extend(trades)

        perm_pnls.append(sum(t['net_ticks'] for t in pt))
        perm_long_pnls.append(sum(t['net_ticks'] for t in pt if t['direction'] == 1))
        perm_short_pnls.append(sum(t['net_ticks'] for t in pt if t['direction'] == -1))

    mean_perm = np.mean(perm_pnls)
    edge = real_pnl - mean_perm
    p_val = np.mean([p >= real_pnl for p in perm_pnls])

    real_long = long_m.get('total_pnl_ticks', 0)
    real_short = short_m.get('total_pnl_ticks', 0)
    long_edge = real_long - np.mean(perm_long_pnls) if perm_long_pnls else 0
    short_edge = real_short - np.mean(perm_short_pnls) if perm_short_pnls else 0
    p_long = np.mean([p >= real_long for p in perm_long_pnls]) if perm_long_pnls else 1
    p_short = np.mean([p >= real_short for p in perm_short_pnls]) if perm_short_pnls else 1

    log.info(f"  {config}: {real_m['n_trades']} trades, PnL={real_pnl:+.1f}t, "
             f"WR={real_m.get('win_rate',0):.1%}, Sharpe={real_m.get('sharpe',0):.2f}, "
             f"Edge={edge:+.1f}t, p={p_val:.3f}")

    return {
        'config': config,
        'tp': tp, 'sl': sl,
        'n_trades': real_m['n_trades'],
        'real_pnl': round(float(real_pnl), 2),
        'win_rate': real_m.get('win_rate', 0),
        'profit_factor': real_m.get('profit_factor', 0),
        'sharpe': real_m.get('sharpe', 0),
        'sortino': real_m.get('sortino', 0),
        'max_dd': real_m.get('max_dd_ticks', 0),
        'exit_types': real_m.get('exit_types', {}),
        'green_days': green, 'red_days': red,
        'n_longs': long_m.get('n_trades', 0),
        'n_shorts': short_m.get('n_trades', 0),
        'long_pnl': round(float(real_long), 2),
        'short_pnl': round(float(real_short), 2),
        'long_wr': long_m.get('win_rate', 0),
        'short_wr': short_m.get('win_rate', 0),
        'mean_perm_pnl': round(float(mean_perm), 2),
        'model_edge': round(float(edge), 2),
        'p_value': round(float(p_val), 3),
        'long_edge': round(float(long_edge), 2),
        'short_edge': round(float(short_edge), 2),
        'p_long': round(float(p_long), 3),
        'p_short': round(float(p_short), 3),
        'perm_pnls': [round(float(p), 2) for p in perm_pnls],
        'trades': all_trades,  # keep for per-day analysis
    }


def main():
    parser = argparse.ArgumentParser(description="Tick-Level Replay v2")
    parser.add_argument('--n-dates', type=int, default=None)
    parser.add_argument('--tp', type=int, nargs='+', default=[2, 3, 4, 5, 8])
    parser.add_argument('--sl', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--top-pct', type=float, default=0.10)
    parser.add_argument('--permutations', type=int, default=20)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    dates = get_oot_dates()
    if args.n_dates:
        dates = dates[:args.n_dates]

    log.info(f"Loading {len(dates)} days of data (this is the slow part — one-time)...")

    # LOAD ALL DATA ONCE
    t0 = time.time()
    days = []
    for d in dates:
        try:
            day = DayData(d, top_pct=args.top_pct)
            days.append(day)
        except Exception as e:
            log.error(f"  Skip {d}: {e}")
    load_time = time.time() - t0
    log.info(f"Data loaded in {load_time:.0f}s ({len(days)} days)")

    # RUN ALL CONFIGS (fast — no I/O)
    all_results = []
    t1 = time.time()

    for tp in args.tp:
        for sl in args.sl:
            if tp <= sl:
                continue
            try:
                r = run_experiment(days, tp, sl, n_perms=args.permutations)
                all_results.append(r)
            except Exception as e:
                log.error(f"TP{tp}_SL{sl}: {e}")
                traceback.print_exc()

    replay_time = time.time() - t1

    # SUMMARY
    print(f"\n{'='*110}")
    print(f"SWEEP SUMMARY | {len(days)} days | top {args.top_pct:.0%} per direction | {args.permutations} permutations")
    print(f"Load: {load_time:.0f}s | Replay: {replay_time:.0f}s | Total: {load_time+replay_time:.0f}s")
    print(f"{'='*110}")
    print(f"{'Config':<12} {'N':>5} {'L/S':>7} {'PnL(t)':>8} {'WR':>6} {'PF':>5} "
          f"{'Sharpe':>7} {'Sort':>7} {'G/R':>5} {'Bias':>8} {'Edge':>8} {'p':>6} {'Sig':>3}")
    print("-" * 110)

    for r in sorted(all_results, key=lambda x: x.get('model_edge', 0), reverse=True):
        sig = '✅' if r['p_value'] < 0.05 else '❌'
        print(f"  {r['config']:<12} {r['n_trades']:>4} {r['n_longs']:>3}/{r['n_shorts']:<3} "
              f"{r['real_pnl']:>+8.1f} {r['win_rate']:>5.1%} {r['profit_factor']:>5.2f} "
              f"{r['sharpe']:>+7.2f} {r['sortino']:>+7.2f} "
              f"{r['green_days']}/{r['red_days']:<2} "
              f"{r['mean_perm_pnl']:>+8.1f} {r['model_edge']:>+8.1f} "
              f"{r['p_value']:>6.3f} {sig:>3}")

    # Direction breakdown
    print(f"\n{'='*80}")
    print(f"DIRECTION BREAKDOWN")
    print(f"{'='*80}")
    print(f"{'Config':<12} {'L_PnL':>8} {'L_WR':>6} {'L_Edge':>8} {'L_p':>6} | "
          f"{'S_PnL':>8} {'S_WR':>6} {'S_Edge':>8} {'S_p':>6}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: x.get('model_edge', 0), reverse=True):
        print(f"  {r['config']:<12} {r['long_pnl']:>+8.1f} {r['long_wr']:>5.1%} "
              f"{r['long_edge']:>+8.1f} {r['p_long']:>6.3f} | "
              f"{r['short_pnl']:>+8.1f} {r['short_wr']:>5.1%} "
              f"{r['short_edge']:>+8.1f} {r['p_short']:>6.3f}")

    # Save (without raw trade lists)
    output_path = args.output or str(OUTPUT_DIR / "tick_replay_v2_results.json")
    save_data = []
    for r in all_results:
        r2 = {k: v for k, v in r.items() if k != 'trades'}
        # Add per-day breakdown
        day_pnls = {}
        for t in r['trades']:
            d = t['date']
            if d not in day_pnls:
                day_pnls[d] = {'net_ticks': 0, 'n_trades': 0, 'n_tp': 0, 'n_sl': 0, 'n_timeout': 0}
            day_pnls[d]['net_ticks'] += t['net_ticks']
            day_pnls[d]['n_trades'] += 1
            day_pnls[d][f"n_{t['exit_type'].lower()}"] = day_pnls[d].get(f"n_{t['exit_type'].lower()}", 0) + 1
        r2['per_day'] = day_pnls
        save_data.append(r2)

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    log.info(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
