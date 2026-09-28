"""
Fast Market Order Simulator — preloads all data once, then sweeps combos.

Tests signals with market (spread-crossing) execution to bypass adverse selection.
Cost: 1 tick spread + 0.24 tick commission = 1.24 ticks per RT.

Usage:
    python alpha_discovery/market_order_fast.py --signals cancel_asym volgated novel_risk_adjusted_return
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(str(RESULTS_DIR / f"market_order_fast_{_ts}.log"), mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("mkt_fast")

TICK = 0.25
TICK_VAL = 12.50
COMM_TICKS = 3.00 / TICK_VAL  # 0.24
SPREAD_TICKS = 1.0  # full RT spread cost for market orders
TOTAL_COST = SPREAD_TICKS + COMM_TICKS  # 1.24 ticks
BARS_SEC = 10

FEAT_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
SIG_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"


def preload_days(n_days: int = 50) -> Dict[str, Dict]:
    """Preload mid, spread for all days. Returns {date: {mid, spread}}."""
    days = {}
    for f in sorted(FEAT_CACHE.glob("*_mbo_features.npz"))[:n_days]:
        date = f.stem.replace("_mbo_features", "")
        data = np.load(str(f))
        feats = data['mbo_features']
        days[date] = {
            'mid': feats[:, 0].copy(),
            'spread': feats[:, 1].copy(),
            'n_bars': len(feats),
        }
        del feats, data
    return days


def load_signal(signal_name: str, date: str) -> np.ndarray:
    """Load signal predictions for a day."""
    path = SIG_DIR / f"{signal_name}_{date}.npz"
    if not path.exists():
        return None
    return np.load(str(path))['predictions']


def sim_day(mid, spread, preds, thresh, hold_bars, trail, cooldown=10, max_trades=0):
    """Fast single-day market order simulation. Returns (pnl_ticks, trades, wins)."""
    n = min(len(mid), len(preds))
    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0
    peak = 0.0

    for i in range(n):
        if in_pos:
            if direction == 1:
                unrealized = (mid[i] - entry_price) / TICK
            else:
                unrealized = (entry_price - mid[i]) / TICK

            peak = max(peak, unrealized)
            bars = i - entry_bar

            do_exit = False
            if bars >= hold_bars:
                do_exit = True
            elif trail > 0 and (peak - unrealized) >= trail:
                do_exit = True

            if do_exit:
                exit_cost = spread[i] / 2.0 / TICK + COMM_TICKS
                pnl = unrealized - exit_cost
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i

        elif not in_pos and (i - last_exit >= cooldown):
            if max_trades > 0 and trades >= max_trades:
                continue

            p = preds[i]
            if abs(p) > thresh:
                if p > 0:
                    entry_price = mid[i] + spread[i] / 2.0  # buy at ask
                    direction = 1
                else:
                    entry_price = mid[i] - spread[i] / 2.0  # sell at bid
                    direction = -1
                in_pos = True
                entry_bar = i
                peak = 0.0

    # Force close
    if in_pos:
        i = n - 1
        if direction == 1:
            unrealized = (mid[i] - entry_price) / TICK
        else:
            unrealized = (entry_price - mid[i]) / TICK
        exit_cost = spread[i] / 2.0 / TICK + COMM_TICKS
        pnl = unrealized - exit_cost
        total_pnl += pnl
        trades += 1
        if pnl > 0:
            wins += 1

    return total_pnl, trades, wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signals', nargs='+', required=True)
    parser.add_argument('--n-days', type=int, default=50)
    args = parser.parse_args()

    log.info(f"Market Order Fast Simulator")
    log.info(f"  Cost: {TOTAL_COST:.2f} ticks/RT ({SPREAD_TICKS}t spread + {COMM_TICKS:.2f}t comm)")
    log.info(f"  Signals: {args.signals}")

    # Preload all day data ONCE
    log.info(f"\nPreloading {args.n_days} days of mid/spread...")
    t0 = time.time()
    days = preload_days(args.n_days)
    log.info(f"  Loaded {len(days)} days in {time.time()-t0:.1f}s")

    # Sweep parameters
    thresholds = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    holds_bars = [100, 300, 600, 1200]  # 10s, 30s, 60s, 120s
    trails = [0, 4, 8, 12]
    max_trades_list = [0, 50]  # unlimited, or max 50/day

    total_combos = len(thresholds) * len(holds_bars) * len(trails) * len(max_trades_list)

    for signal_name in args.signals:
        log.info(f"\n{'='*80}")
        log.info(f"SIGNAL: {signal_name}")
        log.info(f"{'='*80}")

        # Load signal for all days
        sig_data = {}
        for date in days:
            s = load_signal(signal_name, date)
            if s is not None:
                sig_data[date] = s
        log.info(f"  Matched {len(sig_data)}/{len(days)} days with signal files")

        if not sig_data:
            log.info("  SKIPPING — no signal files")
            continue

        log.info(f"  Sweep: {total_combos} combos × {len(sig_data)} days = {total_combos * len(sig_data)} sims")
        t0 = time.time()

        results = []
        combo_count = 0

        for thresh in thresholds:
            for hold in holds_bars:
                for trail in trails:
                    for max_t in max_trades_list:
                        combo_count += 1
                        day_pnls = []
                        day_trades = []
                        day_wins = []

                        for date in sorted(sig_data.keys()):
                            pnl, tr, w = sim_day(
                                days[date]['mid'],
                                days[date]['spread'],
                                sig_data[date],
                                thresh, hold, trail,
                                cooldown=10,
                                max_trades=max_t,
                            )
                            day_pnls.append(pnl * TICK_VAL)
                            day_trades.append(tr)
                            day_wins.append(w)

                        total_pnl = sum(day_pnls)
                        total_tr = sum(day_trades)
                        total_w = sum(day_wins)
                        n = len(day_pnls)
                        pos_days = sum(1 for p in day_pnls if p > 0)
                        active_days = sum(1 for t in day_trades if t > 0)

                        results.append({
                            'thresh': thresh,
                            'hold_s': hold / BARS_SEC,
                            'trail': trail,
                            'max_t': max_t,
                            'total_pnl': total_pnl,
                            'mean_daily_pnl': total_pnl / max(active_days, 1),
                            'total_trades': total_tr,
                            'mean_trades_day': total_tr / max(active_days, 1),
                            'win_rate': total_w / max(total_tr, 1),
                            'pos_days': pos_days,
                            'active_days': active_days,
                            'pct_pos': 100 * pos_days / max(active_days, 1),
                            'sharpe': (np.mean(day_pnls) / max(np.std(day_pnls), 1e-10)) * np.sqrt(252)
                                       if active_days > 1 else 0,
                        })

                        if combo_count % 50 == 0:
                            log.info(f"  [{combo_count}/{total_combos}] "
                                    f"{time.time()-t0:.0f}s elapsed")

        elapsed = time.time() - t0
        log.info(f"\n  Sweep completed in {elapsed:.1f}s ({combo_count} combos)")

        # Sort by mean daily PnL
        results.sort(key=lambda x: x['mean_daily_pnl'], reverse=True)

        # Print top 30
        log.info(f"\n  {'thresh':>6} {'hold':>6} {'trail':>5} {'mt':>3} {'mean_pnl':>10} "
                 f"{'total_pnl':>10} {'wr':>6} {'tr/d':>6} {'%pos':>6} {'sharpe':>7} {'act_d':>5}")
        log.info("  " + "-" * 85)

        for r in results[:30]:
            log.info(f"  {r['thresh']:>6.2f} {r['hold_s']:>5.0f}s {r['trail']:>5.0f} "
                    f"{r['max_t']:>3} {r['mean_daily_pnl']:>+10.2f} "
                    f"{r['total_pnl']:>+10.0f} {r['win_rate']*100:>5.1f}% "
                    f"{r['mean_trades_day']:>5.1f} {r['pct_pos']:>5.1f}% "
                    f"{r['sharpe']:>+6.2f} {r['active_days']:>5}")

        log.info(f"\n  Bottom 5:")
        for r in results[-5:]:
            log.info(f"  {r['thresh']:>6.2f} {r['hold_s']:>5.0f}s {r['trail']:>5.0f} "
                    f"{r['max_t']:>3} {r['mean_daily_pnl']:>+10.2f} "
                    f"{r['total_pnl']:>+10.0f} {r['win_rate']*100:>5.1f}% "
                    f"{r['mean_trades_day']:>5.1f} {r['pct_pos']:>5.1f}% "
                    f"{r['sharpe']:>+6.2f} {r['active_days']:>5}")

        # Aggregate by dimension
        from collections import defaultdict
        for dim_name, dim_key in [('threshold', 'thresh'), ('hold', 'hold_s'), ('trail', 'trail')]:
            by_dim = defaultdict(list)
            for r in results:
                by_dim[r[dim_key]].append(r)
            log.info(f"\n  By {dim_name}:")
            for k in sorted(by_dim.keys()):
                rs = by_dim[k]
                mp = np.mean([r['mean_daily_pnl'] for r in rs])
                wr = np.mean([r['win_rate'] for r in rs])
                pp = np.mean([r['pct_pos'] for r in rs])
                log.info(f"    {dim_name}={k:>6}: mean_pnl={mp:>+8.2f}  wr={wr*100:>5.1f}%  pct_pos={pp:>5.1f}%")

        # Save JSON (convert numpy types)
        def to_native(obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            return obj

        out = RESULTS_DIR / f"market_order_fast_{signal_name}_{_ts}.json"
        clean_results = [{k: to_native(v) for k, v in r.items()} for r in results[:50]]
        with open(str(out), 'w') as f:
            json.dump({
                'signal': signal_name,
                'cost_ticks_rt': float(TOTAL_COST),
                'n_days': len(sig_data),
                'results': clean_results,
            }, f, indent=2)
        log.info(f"\n  Saved: {out}")


if __name__ == '__main__':
    main()
