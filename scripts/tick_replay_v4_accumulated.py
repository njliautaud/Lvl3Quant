#!/usr/bin/env python3
"""
Tick-Level Replay v4 — Accumulated Signal
==========================================
Instead of trading each individual prediction, ACCUMULATE predictions over a
rolling window and trade only when the accumulated signal exceeds a threshold.

Rationale: Individual predictions have IC ~0.14 but per-trade edge is ~0.05-0.25 ticks.
If we accumulate N predictions pointing the same way, the expected edge scales with N
while reducing false signals.

Approach:
  - Compute rolling mean of last N predictions (e.g., last 20 = 5 seconds)
  - Trade when rolling signal exceeds threshold (e.g., > 2 std from neutral)
  - Enter at market when signal triggers
  - Exit when: (a) signal reverses, (b) max hold time, or (c) time-based
  - This naturally reduces trade frequency while concentrating on strongest periods

Also tests:
  - Different rolling windows (5, 10, 20, 40 predictions = 1.25s to 10s)
  - Different entry thresholds (1, 2, 3 std)
  - Short-only vs both directions
  - Signal-reversal exit vs time-based exit

Author: Claude (autonomous build, 2026-07-02)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tick_level_replay import (
    ROOT, RAW_MBO_DIR, SMART_V3_DIR, OOT_PRED_DIR, OUTPUT_DIR,
    ES_TICK_SIZE, ES_TICK_VALUE, ES_RT_COMMISSION_TICKS,
    WINDOW_SIZE, STRIDE,
    compute_metrics, get_oot_dates,
)
import databento as dbn

logging.basicConfig(
    format="%(asctime)s [REPLAY-V4] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("REPLAY-V4")

# Cost for market entry + market exit = commission + 1 tick spread RT
COST_MARKET_RT = ES_RT_COMMISSION_TICKS + 1.0  # 1.376 ticks


class DayData:
    """Pre-loaded data for one trading day."""

    def __init__(self, date_str: str):
        self.date_str = date_str
        self._load_trades(date_str)
        self._load_predictions(date_str)

    def _load_trades(self, date_str: str):
        fname = f"glbx-mdp3-{date_str}.mbo.dbn.zst"
        fpath = RAW_MBO_DIR / fname
        store = dbn.DBNStore.from_file(str(fpath))
        df = store.to_df()

        es_mask = df['symbol'].str.match(r'^ES[A-Z]\d$', na=False)
        es = df[es_mask]
        inst_counts = es['instrument_id'].value_counts()
        dominant = inst_counts.index[0]
        es = es[es['instrument_id'] == dominant]

        es_ts_ns = es['ts_event'].astype(np.int64)
        ts_utc = es['ts_event'].dt.tz_convert('UTC')
        hour = ts_utc.dt.hour
        minute = ts_utc.dt.minute
        rth_mask = ((hour > 13) | ((hour == 13) & (minute >= 30))) & (hour < 21)
        es_rth = es[rth_mask]

        actions = es_rth['action'].values
        trade_mask = actions == 'T'
        rth_ts = es_ts_ns[rth_mask].values

        self.trade_ts = rth_ts[trade_mask]
        self.trade_prices = es_rth['price'].values[trade_mask].astype(np.float64)
        sides = es_rth['side'].values[trade_mask]

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

        del df, es, es_rth, store
        gc.collect()
        log.info(f"  {date_str}: {n} trades loaded")

    def _load_predictions(self, date_str: str):
        """Load ALL predictions (we'll compute rolling signal ourselves)."""
        pred_path = OOT_PRED_DIR / f"oot_{date_str}.npz"
        mbo_path = SMART_V3_DIR / f"{date_str}_mbo_events.npz"

        pred_data = np.load(str(pred_path), allow_pickle=True)
        mbo_data = np.load(str(mbo_path), allow_pickle=True)

        self.pred_1s = pred_data['pred_log_ret_1s'].astype(np.float64)
        self.pred_5s = pred_data['pred_log_ret_5s'].astype(np.float64)
        self.pred_10s = pred_data['pred_log_ret_10s'].astype(np.float64)

        n_pred = len(self.pred_1s)
        mbo_ts = mbo_data['timestamps']
        n_events = len(mbo_ts)

        pred_indices = np.arange(WINDOW_SIZE - 1, n_events, STRIDE)[:n_pred]
        self.pred_ts = mbo_ts[pred_indices]

        log.info(f"  {date_str}: {n_pred} predictions loaded")


def compute_rolling_signal(preds: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling mean of predictions over last `window` samples.
    Returns array of same length as preds.
    """
    n = len(preds)
    rolling = np.full(n, np.nan)
    cumsum = np.cumsum(preds)
    for i in range(window - 1, n):
        if i < window:
            rolling[i] = cumsum[i] / (i + 1)
        else:
            rolling[i] = (cumsum[i] - cumsum[i - window]) / window
    return rolling


def replay_accumulated(day: DayData, window: int = 20,
                       entry_threshold_std: float = 2.0,
                       max_hold_seconds: float = 30.0,
                       mode: str = 'both',
                       use_reversal_exit: bool = True,
                       reversal_threshold: float = 0.0,
                       shuffle: bool = False,
                       rng: np.random.Generator = None) -> List[Dict]:
    """
    Accumulated signal replay.

    - Compute rolling mean of predictions over `window` predictions
    - Normalize by rolling std to get z-score
    - Enter when z-score exceeds entry_threshold_std
    - Exit when: signal reverses past reversal_threshold, or max hold time
    """
    preds = day.pred_1s.copy()

    if shuffle:
        if rng is None:
            rng = np.random.default_rng(42)
        rng.shuffle(preds)  # shuffle in-place

    # Compute rolling signal
    rolling_mean = compute_rolling_signal(preds, window)

    # Compute rolling std for normalization
    n = len(preds)
    rolling_std = np.full(n, np.nan)
    # Use expanding std for the first `window` samples
    for i in range(window - 1, n):
        start = max(0, i - window + 1)
        rolling_std[i] = np.std(preds[start:i+1])

    # Z-score the rolling signal
    valid_mask = ~np.isnan(rolling_mean) & ~np.isnan(rolling_std) & (rolling_std > 1e-10)
    z_signal = np.full(n, 0.0)
    z_signal[valid_mask] = rolling_mean[valid_mask] / rolling_std[valid_mask]

    pred_ts = day.pred_ts
    trade_ts = day.trade_ts
    trade_prices = day.trade_prices
    trade_bid = day.trade_bid
    trade_ask = day.trade_ask

    max_hold_ns = int(max_hold_seconds * 1e9)

    results = []
    in_position = False
    position_dir = 0
    entry_price = 0.0
    entry_ts = 0
    entry_signal = 0.0

    pred_ptr = 0
    current_z = 0.0

    for ti in range(len(trade_ts)):
        t = trade_ts[ti]
        p = trade_prices[ti]
        mid = (trade_bid[ti] + trade_ask[ti]) / 2

        # Update signal from predictions that have arrived
        while pred_ptr < n and pred_ts[pred_ptr] <= t:
            if valid_mask[pred_ptr]:
                current_z = z_signal[pred_ptr]
            pred_ptr += 1

        if in_position:
            # Check exit conditions
            should_exit = False
            exit_reason = ''

            # Time-based exit
            if t >= entry_ts + max_hold_ns:
                should_exit = True
                exit_reason = 'TIMEOUT'

            # Signal reversal exit
            if use_reversal_exit and not should_exit:
                if position_dir == -1 and current_z > reversal_threshold:
                    should_exit = True
                    exit_reason = 'REVERSAL'
                elif position_dir == 1 and current_z < -reversal_threshold:
                    should_exit = True
                    exit_reason = 'REVERSAL'

            if should_exit:
                # Exit at market (mid price)
                if position_dir == 1:
                    pnl_ticks = (mid - entry_price) / ES_TICK_SIZE
                else:
                    pnl_ticks = (entry_price - mid) / ES_TICK_SIZE

                net = pnl_ticks - COST_MARKET_RT
                results.append({
                    'date': day.date_str,
                    'direction': position_dir,
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(mid, 2),
                    'exit_type': exit_reason,
                    'pnl_ticks': round(pnl_ticks, 4),
                    'cost_ticks': round(COST_MARKET_RT, 4),
                    'net_ticks': round(net, 4),
                    'net_dollars': round(net * ES_TICK_VALUE, 2),
                    'hold_time_ns': int(t - entry_ts),
                    'fill_time_ns': 0,
                    'entry_z': round(entry_signal, 4),
                    'exit_z': round(current_z, 4),
                })
                in_position = False

        if not in_position:
            # Check entry conditions
            if current_z <= -entry_threshold_std and mode in ('both', 'short_only'):
                # Strong short signal
                in_position = True
                position_dir = -1
                entry_price = mid
                entry_ts = t
                entry_signal = current_z
            elif current_z >= entry_threshold_std and mode in ('both', 'long_only'):
                # Strong long signal
                in_position = True
                position_dir = 1
                entry_price = mid
                entry_ts = t
                entry_signal = current_z

    # Force close EOD
    if in_position and len(trade_ts) > 0:
        ti = len(trade_ts) - 1
        mid = (trade_bid[ti] + trade_ask[ti]) / 2
        if position_dir == 1:
            pnl_ticks = (mid - entry_price) / ES_TICK_SIZE
        else:
            pnl_ticks = (entry_price - mid) / ES_TICK_SIZE
        net = pnl_ticks - COST_MARKET_RT
        results.append({
            'date': day.date_str,
            'direction': position_dir,
            'entry_price': round(entry_price, 2),
            'exit_price': round(mid, 2),
            'exit_type': 'EOD',
            'pnl_ticks': round(pnl_ticks, 4),
            'cost_ticks': round(COST_MARKET_RT, 4),
            'net_ticks': round(net, 4),
            'net_dollars': round(net * ES_TICK_VALUE, 2),
            'hold_time_ns': int(trade_ts[ti] - entry_ts),
            'fill_time_ns': 0,
            'entry_z': round(entry_signal, 4),
            'exit_z': round(current_z, 4),
        })

    return results


def run_sweep(days: List[DayData], windows: List[int],
              thresholds: List[float], modes: List[str],
              max_holds: List[float],
              use_reversal: bool = True,
              n_perms: int = 20) -> List[Dict]:
    """Run full parameter sweep."""
    all_results = []

    for mode in modes:
        for window in windows:
            for thresh in thresholds:
                for max_hold in max_holds:
                    label = f"{mode}_w{window}_z{thresh}_h{int(max_hold)}s"

                    # Real trades
                    real_trades = []
                    for day in days:
                        trades = replay_accumulated(
                            day, window=window, entry_threshold_std=thresh,
                            max_hold_seconds=max_hold, mode=mode,
                            use_reversal_exit=use_reversal
                        )
                        real_trades.extend(trades)

                    if not real_trades:
                        continue

                    real_m = compute_metrics(real_trades, label)
                    real_pnl = real_m.get('total_pnl_ticks', 0)
                    gross_pnl = sum(t['pnl_ticks'] for t in real_trades)
                    avg_gross = gross_pnl / len(real_trades)

                    # Day stats
                    day_pnls = {}
                    for t in real_trades:
                        d = t['date']
                        day_pnls[d] = day_pnls.get(d, 0) + t['net_ticks']
                    green = sum(1 for v in day_pnls.values() if v > 0)
                    red = sum(1 for v in day_pnls.values() if v < 0)

                    # Avg hold time
                    avg_hold_s = np.mean([t['hold_time_ns'] for t in real_trades]) / 1e9

                    # Exit type breakdown
                    rev_exits = sum(1 for t in real_trades if t['exit_type'] == 'REVERSAL')
                    timeout_exits = sum(1 for t in real_trades if t['exit_type'] == 'TIMEOUT')

                    # Permutation test
                    perm_pnls = []
                    for pi in range(n_perms):
                        rng = np.random.default_rng(seed=pi * 1000 + 42)
                        pt = []
                        for day in days:
                            trades = replay_accumulated(
                                day, window=window, entry_threshold_std=thresh,
                                max_hold_seconds=max_hold, mode=mode,
                                use_reversal_exit=use_reversal,
                                shuffle=True, rng=rng
                            )
                            pt.extend(trades)
                        perm_pnls.append(sum(t['net_ticks'] for t in pt))

                    mean_perm = np.mean(perm_pnls) if perm_pnls else 0
                    edge = real_pnl - mean_perm
                    p_val = np.mean([p >= real_pnl for p in perm_pnls]) if perm_pnls else 1.0

                    log.info(f"  {label}: {len(real_trades)} trades, "
                             f"gross={gross_pnl:+.1f}t ({avg_gross:+.3f}/tr), "
                             f"net={real_pnl:+.1f}t, edge={edge:+.1f}t, p={p_val:.3f}, "
                             f"hold={avg_hold_s:.1f}s")

                    all_results.append({
                        'label': label,
                        'mode': mode,
                        'window': window,
                        'threshold': thresh,
                        'max_hold': max_hold,
                        'n_trades': len(real_trades),
                        'gross_pnl': round(float(gross_pnl), 2),
                        'avg_gross': round(float(avg_gross), 4),
                        'net_pnl': round(float(real_pnl), 2),
                        'win_rate': real_m.get('win_rate', 0),
                        'profit_factor': real_m.get('profit_factor', 0),
                        'sharpe': real_m.get('sharpe', 0),
                        'sortino': real_m.get('sortino', 0),
                        'green_days': green,
                        'red_days': red,
                        'avg_hold_s': round(avg_hold_s, 1),
                        'rev_exits': rev_exits,
                        'timeout_exits': timeout_exits,
                        'mean_perm_pnl': round(float(mean_perm), 2),
                        'model_edge': round(float(edge), 2),
                        'p_value': round(float(p_val), 3),
                        'cost_per_trade': round(COST_MARKET_RT, 3),
                    })

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Tick-Level Replay v4 — Accumulated Signal")
    parser.add_argument('--n-dates', type=int, default=None)
    parser.add_argument('--windows', type=int, nargs='+', default=[5, 10, 20, 40])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[1.5, 2.0, 2.5, 3.0])
    parser.add_argument('--max-holds', type=float, nargs='+', default=[15, 30, 60])
    parser.add_argument('--modes', nargs='+', default=['short_only', 'both'])
    parser.add_argument('--permutations', type=int, default=20)
    parser.add_argument('--no-reversal', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    dates = get_oot_dates()
    if args.n_dates:
        dates = dates[:args.n_dates]

    log.info(f"Loading {len(dates)} days...")
    t0 = time.time()
    days = []
    for d in dates:
        try:
            day = DayData(d)
            days.append(day)
        except Exception as e:
            log.error(f"  Skip {d}: {e}")
    load_time = time.time() - t0
    log.info(f"Loaded in {load_time:.0f}s")

    t1 = time.time()
    results = run_sweep(days, args.windows, args.thresholds, args.modes,
                        args.max_holds, use_reversal=not args.no_reversal,
                        n_perms=args.permutations)
    sweep_time = time.time() - t1

    # Summary
    print(f"\n{'='*130}")
    print(f"ACCUMULATED SIGNAL SWEEP | {len(days)} days | {args.permutations} perms | cost={COST_MARKET_RT:.3f}t/RT")
    print(f"Load: {load_time:.0f}s | Sweep: {sweep_time:.0f}s")
    print(f"{'='*130}")
    print(f"{'Label':<30} {'N':>5} {'Gross':>8} {'$/tr':>7} {'Net':>8} {'WR':>6} "
          f"{'Sharpe':>7} {'Hold':>5} {'G/R':>5} {'Bias':>8} {'Edge':>8} {'p':>6} {'Sig':>3}")
    print("-" * 130)

    for r in sorted(results, key=lambda x: x.get('avg_gross', 0), reverse=True):
        sig = '✅' if r['p_value'] < 0.05 else '❌'
        print(f"  {r['label']:<30} {r['n_trades']:>4} {r['gross_pnl']:>+8.1f} "
              f"{r['avg_gross']:>+7.3f} {r['net_pnl']:>+8.1f} {r['win_rate']:>5.1%} "
              f"{r['sharpe']:>+7.2f} {r['avg_hold_s']:>5.1f} "
              f"{r['green_days']}/{r['red_days']:<2} "
              f"{r['mean_perm_pnl']:>+8.1f} {r['model_edge']:>+8.1f} "
              f"{r['p_value']:>6.3f} {sig:>3}")

    # Highlight profitable configs
    profitable = [r for r in results if r['net_pnl'] > 0]
    if profitable:
        print(f"\n  🟢 {len(profitable)} NET PROFITABLE configs!")
        for r in sorted(profitable, key=lambda x: x['net_pnl'], reverse=True)[:5]:
            print(f"    {r['label']}: net={r['net_pnl']:+.1f}t (${r['net_pnl']*ES_TICK_VALUE:+,.0f}), "
                  f"edge={r['model_edge']:+.1f}t, p={r['p_value']:.3f}")

    # Save
    output_path = args.output or str(OUTPUT_DIR / "tick_replay_v4_accumulated_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
