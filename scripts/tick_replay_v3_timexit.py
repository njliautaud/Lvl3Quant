#!/usr/bin/env python3
"""
Tick-Level Replay v3 — Time-Based Exit Only
=============================================
Tests the raw directional prediction accuracy by:
  - Market entry (immediate fill at current price, pay the spread)
  - Time-based exit only (hold for exactly N seconds, exit at market)
  - No TP/SL mechanics that confound the signal

This answers: "Does the model predict direction correctly?"
If time-exit shows real edge, we can THEN optimize execution around it.

Also tests:
  - SHORT-ONLY (since model shows real short signal)
  - INVERTED longs (reverse long signals to go short)
  - Multiple hold periods (1s, 5s, 10s, 30s)
  - Multiple signal selectivity levels (top 1%, 5%, 10%, 20%)

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
    WINDOW_SIZE, STRIDE,
    compute_metrics, get_oot_dates,
)
import databento as dbn

logging.basicConfig(
    format="%(asctime)s [REPLAY-V3] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("REPLAY-V3")

# Market order cost: commission + spread crossing on BOTH sides
MARKET_RT_COST_TICKS = ES_RT_COMMISSION_TICKS + 2.0  # 0.376 + 2.0 = 2.376 ticks
# (1 tick spread crossing on entry + 1 tick on exit)

# More conservative: 1 tick on entry only, passive exit
MARKET_ENTRY_PASSIVE_EXIT_COST = ES_RT_COMMISSION_TICKS + 1.0  # 1.376 ticks


class DayData:
    """Pre-loaded trade data for one day."""

    def __init__(self, date_str: str, top_pct: float = 0.10):
        self.date_str = date_str
        self._load_trades(date_str)
        self._load_all_predictions(date_str, top_pct)

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
        rth_ts = es_ts_ns[rth_mask].values

        actions = es_rth['action'].values
        trade_mask = actions == 'T'

        self.trade_ts = rth_ts[trade_mask]
        self.trade_prices = es_rth['price'].values[trade_mask].astype(np.float64)
        sides = es_rth['side'].values[trade_mask]

        n = len(self.trade_prices)
        self.trade_mid = np.empty(n, dtype=np.float64)
        for i in range(n):
            if sides[i] == 'A':
                self.trade_mid[i] = self.trade_prices[i] - ES_TICK_SIZE / 2
            elif sides[i] == 'B':
                self.trade_mid[i] = self.trade_prices[i] + ES_TICK_SIZE / 2
            else:
                self.trade_mid[i] = self.trade_prices[i]

        del df, es, es_rth, store
        gc.collect()
        log.info(f"  {date_str}: {n} trades loaded")

    def _load_all_predictions(self, date_str: str, top_pct: float):
        """Load predictions at multiple selectivity levels."""
        pred_path = OOT_PRED_DIR / f"oot_{date_str}.npz"
        mbo_path = SMART_V3_DIR / f"{date_str}_mbo_events.npz"

        pred_data = np.load(str(pred_path), allow_pickle=True)
        mbo_data = np.load(str(mbo_path), allow_pickle=True)

        self.raw_preds = pred_data['pred_log_ret_1s'].astype(np.float64)
        n_pred = len(self.raw_preds)
        mbo_ts = mbo_data['timestamps']
        n_events = len(mbo_ts)

        pred_indices = np.arange(WINDOW_SIZE - 1, n_events, STRIDE)[:n_pred]
        self.pred_ts = mbo_ts[pred_indices]
        self.pred_dirs = np.sign(self.raw_preds).astype(np.int8)
        self.pred_conf = np.abs(self.raw_preds)

        # Pre-compute signal masks at various selectivity levels
        self.signal_masks = {}
        for pct in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
            long_mask = self.raw_preds > 0
            short_mask = self.raw_preds < 0
            sel_mask = np.zeros(n_pred, dtype=bool)

            if np.sum(long_mask) > 0:
                lt = np.percentile(self.pred_conf[long_mask], 100 * (1 - pct))
                sel_mask[long_mask & (self.pred_conf >= lt)] = True
            if np.sum(short_mask) > 0:
                st = np.percentile(self.pred_conf[short_mask], 100 * (1 - pct))
                sel_mask[short_mask & (self.pred_conf >= st)] = True

            n_l = int(np.sum(sel_mask & long_mask))
            n_s = int(np.sum(sel_mask & short_mask))
            self.signal_masks[pct] = (sel_mask, n_l, n_s)

        log.info(f"  {date_str}: {n_pred} predictions, masks computed")


def time_exit_replay(day: DayData, hold_seconds: float, top_pct: float = 0.10,
                     mode: str = 'both', shuffle: bool = False,
                     rng: np.random.Generator = None,
                     cooldown_seconds: float = 0.0) -> List[Dict]:
    """
    Time-based exit replay.

    Args:
        mode: 'both' (L+S), 'short_only', 'long_only', 'invert_longs' (L→S, keep S)
        cooldown_seconds: minimum seconds between trades
    """
    sel_mask, _, _ = day.signal_masks.get(top_pct, (np.zeros(len(day.raw_preds), dtype=bool), 0, 0))
    if not np.any(sel_mask):
        return []

    sig_ts = day.pred_ts[sel_mask]
    sig_dirs = day.pred_dirs[sel_mask].copy()
    sig_conf = day.pred_conf[sel_mask]

    # Apply mode filter
    if mode == 'short_only':
        keep = sig_dirs == -1
        sig_ts = sig_ts[keep]
        sig_dirs = sig_dirs[keep]
        sig_conf = sig_conf[keep]
    elif mode == 'long_only':
        keep = sig_dirs == 1
        sig_ts = sig_ts[keep]
        sig_dirs = sig_dirs[keep]
        sig_conf = sig_conf[keep]
    elif mode == 'invert_longs':
        # Reverse long signals to short, keep original shorts
        sig_dirs[sig_dirs == 1] = -1

    if shuffle:
        if rng is None:
            rng = np.random.default_rng(42)
        sig_dirs = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(sig_dirs))

    if len(sig_ts) == 0:
        return []

    trade_ts = day.trade_ts
    trade_prices = day.trade_prices
    trade_mid = day.trade_mid
    n_trades_total = len(trade_ts)

    hold_ns = int(hold_seconds * 1e9)
    cooldown_ns = int(cooldown_seconds * 1e9)

    results = []
    signal_ptr = 0
    last_exit_ts = 0

    for ti in range(n_trades_total):
        t = trade_ts[ti]

        # Find next signal at or before this timestamp
        while signal_ptr < len(sig_ts) and sig_ts[signal_ptr] <= t:
            sig_t = sig_ts[signal_ptr]
            sig_d = sig_dirs[signal_ptr]
            signal_ptr += 1

            if sig_d == 0:
                continue

            # Cooldown check
            if sig_t < last_exit_ts + cooldown_ns:
                continue

            # MARKET ENTRY: fill at current price
            entry_price = trade_mid[ti]

            # Find exit: first trade after hold_seconds
            exit_ts_target = sig_t + hold_ns
            # Binary search for exit trade
            exit_idx = np.searchsorted(trade_ts[ti:], exit_ts_target) + ti

            if exit_idx >= n_trades_total:
                exit_idx = n_trades_total - 1

            exit_price = trade_mid[exit_idx]
            actual_hold_ns = int(trade_ts[exit_idx] - sig_t)

            # PnL in ticks
            if sig_d == 1:  # LONG
                pnl_ticks = (exit_price - entry_price) / ES_TICK_SIZE
            else:  # SHORT
                pnl_ticks = (entry_price - exit_price) / ES_TICK_SIZE

            # Cost: market entry + market exit = 2 ticks spread + commission
            cost = MARKET_RT_COST_TICKS
            net = pnl_ticks - cost

            results.append({
                'date': day.date_str,
                'direction': int(sig_d),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'exit_type': 'TIMEOUT',
                'pnl_ticks': round(pnl_ticks, 4),
                'cost_ticks': round(cost, 4),
                'net_ticks': round(net, 4),
                'net_dollars': round(net * ES_TICK_VALUE, 2),
                'hold_ns': actual_hold_ns,
                'fill_time_ns': 0,
                'hold_time_ns': actual_hold_ns,
            })

            last_exit_ts = trade_ts[exit_idx]
            break  # one trade at a time

    return results


def run_time_exit_sweep(days: List[DayData], hold_seconds: List[float],
                        top_pcts: List[float], modes: List[str],
                        n_perms: int = 20, cooldown: float = 0.0) -> List[Dict]:
    """Run full sweep of time-exit experiments."""
    all_results = []

    for mode in modes:
        for top_pct in top_pcts:
            for hold_s in hold_seconds:
                label = f"{mode}_top{int(top_pct*100)}%_hold{hold_s}s"
                log.info(f"\n--- {label} ---")

                # Real trades
                real_trades = []
                for day in days:
                    trades = time_exit_replay(day, hold_s, top_pct=top_pct,
                                              mode=mode, cooldown_seconds=cooldown)
                    real_trades.extend(trades)

                if not real_trades:
                    log.info(f"  {label}: NO TRADES")
                    continue

                real_m = compute_metrics(real_trades, label)
                real_pnl = real_m.get('total_pnl_ticks', 0)

                # Day-level stats
                day_pnls = {}
                for t in real_trades:
                    d = t['date']
                    day_pnls[d] = day_pnls.get(d, 0) + t['net_ticks']
                green = sum(1 for v in day_pnls.values() if v > 0)
                red = sum(1 for v in day_pnls.values() if v < 0)

                # Permutation test
                perm_pnls = []
                for pi in range(n_perms):
                    rng = np.random.default_rng(seed=pi * 1000 + 42)
                    pt = []
                    for day in days:
                        trades = time_exit_replay(day, hold_s, top_pct=top_pct,
                                                  mode=mode, shuffle=True, rng=rng,
                                                  cooldown_seconds=cooldown)
                        pt.extend(trades)
                    perm_pnls.append(sum(t['net_ticks'] for t in pt))

                mean_perm = np.mean(perm_pnls) if perm_pnls else 0
                edge = real_pnl - mean_perm
                p_val = np.mean([p >= real_pnl for p in perm_pnls]) if perm_pnls else 1.0

                log.info(f"  {label}: {real_m['n_trades']} trades, "
                         f"PnL={real_pnl:+.1f}t, WR={real_m.get('win_rate',0):.1%}, "
                         f"Edge={edge:+.1f}t, p={p_val:.3f}")

                # Gross PnL (before costs)
                gross_pnl = sum(t['pnl_ticks'] for t in real_trades)
                avg_gross = gross_pnl / len(real_trades) if real_trades else 0

                all_results.append({
                    'label': label,
                    'mode': mode,
                    'top_pct': top_pct,
                    'hold_seconds': hold_s,
                    'n_trades': real_m['n_trades'],
                    'real_pnl': round(float(real_pnl), 2),
                    'gross_pnl': round(float(gross_pnl), 2),
                    'avg_gross_per_trade': round(float(avg_gross), 4),
                    'win_rate': real_m.get('win_rate', 0),
                    'profit_factor': real_m.get('profit_factor', 0),
                    'sharpe': real_m.get('sharpe', 0),
                    'sortino': real_m.get('sortino', 0),
                    'green_days': green,
                    'red_days': red,
                    'mean_perm_pnl': round(float(mean_perm), 2),
                    'model_edge': round(float(edge), 2),
                    'p_value': round(float(p_val), 3),
                    'cost_per_trade': round(MARKET_RT_COST_TICKS, 3),
                })

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Tick-Level Replay v3 — Time Exit")
    parser.add_argument('--n-dates', type=int, default=None)
    parser.add_argument('--hold', type=float, nargs='+', default=[1, 5, 10, 30])
    parser.add_argument('--top-pct', type=float, nargs='+', default=[0.01, 0.05, 0.10])
    parser.add_argument('--modes', nargs='+', default=['both', 'short_only', 'invert_longs'])
    parser.add_argument('--permutations', type=int, default=20)
    parser.add_argument('--cooldown', type=float, default=5.0, help='Min seconds between trades')
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
            day = DayData(d, top_pct=max(args.top_pct))
            days.append(day)
        except Exception as e:
            log.error(f"  Skip {d}: {e}")
    load_time = time.time() - t0
    log.info(f"Loaded in {load_time:.0f}s")

    t1 = time.time()
    results = run_time_exit_sweep(days, args.hold, args.top_pct, args.modes,
                                  n_perms=args.permutations,
                                  cooldown=args.cooldown)
    sweep_time = time.time() - t1

    # Summary
    print(f"\n{'='*120}")
    print(f"TIME-EXIT SWEEP | {len(days)} days | {args.permutations} permutations | cooldown={args.cooldown}s")
    print(f"Cost model: market entry + market exit = {MARKET_RT_COST_TICKS:.3f} ticks/RT")
    print(f"Load: {load_time:.0f}s | Sweep: {sweep_time:.0f}s")
    print(f"{'='*120}")
    print(f"{'Label':<35} {'N':>5} {'Gross':>8} {'Net':>8} {'$/tr':>7} {'WR':>6} "
          f"{'Sharpe':>7} {'G/R':>5} {'Bias':>8} {'Edge':>8} {'p':>6} {'Sig':>3}")
    print("-" * 120)

    for r in sorted(results, key=lambda x: x.get('model_edge', 0), reverse=True):
        sig = '✅' if r['p_value'] < 0.05 else '❌'
        print(f"  {r['label']:<35} {r['n_trades']:>4} {r['gross_pnl']:>+8.1f} "
              f"{r['real_pnl']:>+8.1f} {r['avg_gross_per_trade']:>+7.4f} "
              f"{r['win_rate']:>5.1%} {r['sharpe']:>+7.2f} "
              f"{r['green_days']}/{r['red_days']:<2} "
              f"{r['mean_perm_pnl']:>+8.1f} {r['model_edge']:>+8.1f} "
              f"{r['p_value']:>6.3f} {sig:>3}")

    # Save
    output_path = args.output or str(OUTPUT_DIR / "tick_replay_v3_timexit_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved to {output_path}")

    # Key findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")

    sig_results = [r for r in results if r['p_value'] < 0.05]
    if sig_results:
        print(f"\n  {len(sig_results)} significant results (p < 0.05):")
        for r in sorted(sig_results, key=lambda x: x['model_edge'], reverse=True)[:10]:
            print(f"    {r['label']}: edge={r['model_edge']:+.1f}t, "
                  f"gross_avg={r['avg_gross_per_trade']:+.4f}t/trade, p={r['p_value']:.3f}")
    else:
        print("\n  ❌ NO significant results found.")

    # Check if gross PnL is positive (edge exists before costs)
    gross_pos = [r for r in results if r['gross_pnl'] > 0]
    if gross_pos:
        print(f"\n  {len(gross_pos)} configs with positive GROSS PnL (edge before costs):")
        for r in sorted(gross_pos, key=lambda x: x['avg_gross_per_trade'], reverse=True)[:5]:
            print(f"    {r['label']}: gross={r['gross_pnl']:+.1f}t "
                  f"({r['avg_gross_per_trade']:+.4f}t/trade), "
                  f"net={r['real_pnl']:+.1f}t (need >{r['cost_per_trade']:.3f}t/trade to profit)")


if __name__ == '__main__':
    main()
