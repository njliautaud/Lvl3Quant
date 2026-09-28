"""
Market Order Simulator — Test signals with MARKET execution (cross the spread).

The Rust fill_sim_cli only supports limit orders with FIFO queue. This simulator
tests what happens if we use market orders instead:
- Entry: cross the spread (pay best ask for buy, best bid for sell)
- Exit: market order OR trailing stop (also crosses spread)
- Cost: 1 tick spread + 0.24 tick commission = 1.24 ticks per RT

This bypasses the adverse selection problem in limit orders but requires
the signal to be strong enough to overcome the spread cost.

Usage:
    python alpha_discovery/market_order_sim.py --signal cancel_asym --n-days 50
    python alpha_discovery/market_order_sim.py --signal cancel_asym --n-days 50 --thresh 1.5 --hold 600
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

# Setup logging
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(str(RESULTS_DIR / f"market_order_sim_{_ts}.log"), mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("mkt_sim")

# Constants
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24 ticks
SPREAD_COST_TICKS = 1.0  # market order crosses 1 tick spread (entry + exit)
TOTAL_COST_TICKS = SPREAD_COST_TICKS + COMMISSION_TICKS  # 1.24 ticks
BARS_PER_SEC = 10

FEATURE_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
SIGNAL_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"


def discover_days(n_days: int = 0) -> List[Tuple[str, Path, Optional[Path]]]:
    """Find days with both feature cache and signal predictions."""
    days = []
    for f in sorted(FEATURE_CACHE.glob("*_mbo_features.npz")):
        date_str = f.stem.replace("_mbo_features", "")
        days.append((date_str, f))
    if n_days > 0:
        days = days[:n_days]
    return days


def load_features_and_signal(feat_path: Path, signal_path: Path):
    """Load features (for mid/spread) and pre-computed signal predictions."""
    feat_data = np.load(str(feat_path))
    features = feat_data['mbo_features']
    mid = features[:, 0].copy()     # col 0 = mid
    spread = features[:, 1].copy()  # col 1 = spread

    sig_data = np.load(str(signal_path))
    predictions = sig_data['predictions']

    n = min(len(mid), len(predictions))
    return mid[:n], spread[:n], predictions[:n]


def simulate_market_orders(
    mid: np.ndarray,
    spread: np.ndarray,
    predictions: np.ndarray,
    signal_threshold: float = 1.0,
    hold_bars: int = 600,
    trailing_stop_ticks: float = 8.0,
    take_profit_ticks: float = 0.0,
    cooldown_bars: int = 10,
    max_signals_per_day: int = 0,
) -> Dict:
    """
    Simulate market order execution.

    Entry: when |prediction| > signal_threshold
      - prediction > thresh → BUY at mid + spread/2
      - prediction < -thresh → SELL at mid - spread/2

    Exit: whichever comes first:
      - hold_bars reached → market exit
      - trailing stop hit → market exit
      - take profit hit → market exit

    Market exit: pay the other half of the spread.
    Total cost: entry_spread/2 + exit_spread/2 + commission = ~1.24 ticks
    """
    n = len(mid)
    trades = []
    position = 0  # 0=flat, 1=long, -1=short
    entry_bar = 0
    entry_price = 0.0
    entry_direction = 0
    peak_pnl_ticks = 0.0
    signal_count = 0
    last_exit_bar = -cooldown_bars

    for i in range(n):
        if position != 0:
            # In a position — check exits
            if entry_direction == 1:  # long
                current_pnl_ticks = (mid[i] - entry_price) / TICK_SIZE
            else:  # short
                current_pnl_ticks = (entry_price - mid[i]) / TICK_SIZE

            peak_pnl_ticks = max(peak_pnl_ticks, current_pnl_ticks)
            bars_held = i - entry_bar

            exit_reason = None

            # Time exit
            if bars_held >= hold_bars:
                exit_reason = 'time'

            # Trailing stop: exit if drawdown from peak > trailing_stop_ticks
            if trailing_stop_ticks > 0 and (peak_pnl_ticks - current_pnl_ticks) >= trailing_stop_ticks:
                exit_reason = 'trailing'

            # Take profit
            if take_profit_ticks > 0 and current_pnl_ticks >= take_profit_ticks:
                exit_reason = 'tp'

            if exit_reason:
                # Market exit: pay half spread at exit
                exit_spread_cost = spread[i] / 2.0 / TICK_SIZE  # in ticks
                if entry_direction == 1:
                    exit_price = mid[i] - spread[i] / 2.0  # sell at bid
                else:
                    exit_price = mid[i] + spread[i] / 2.0  # buy at ask

                pnl_ticks = current_pnl_ticks - exit_spread_cost - COMMISSION_TICKS
                # Entry spread was already subtracted from entry_price

                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'direction': entry_direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_ticks': pnl_ticks,
                    'pnl_dollars': pnl_ticks * TICK_VALUE,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason,
                    'peak_pnl_ticks': peak_pnl_ticks,
                    'signal_strength': abs(predictions[entry_bar]),
                })

                position = 0
                last_exit_bar = i
                continue

        else:
            # Flat — check for entry signals
            if i - last_exit_bar < cooldown_bars:
                continue

            if max_signals_per_day > 0 and signal_count >= max_signals_per_day:
                continue

            pred = predictions[i]
            if abs(pred) > signal_threshold:
                signal_count += 1

                if pred > 0:
                    # BUY: cross the spread, pay ask
                    entry_price = mid[i] + spread[i] / 2.0
                    entry_direction = 1
                else:
                    # SELL: cross the spread, pay bid
                    entry_price = mid[i] - spread[i] / 2.0
                    entry_direction = -1

                # Subtract entry half-spread from price (already in entry_price)
                # The cost is: entry_spread/2 is embedded in entry_price
                position = 1
                entry_bar = i
                peak_pnl_ticks = 0.0

    # Force close at end of day
    if position != 0:
        i = n - 1
        if entry_direction == 1:
            current_pnl_ticks = (mid[i] - entry_price) / TICK_SIZE
        else:
            current_pnl_ticks = (entry_price - mid[i]) / TICK_SIZE

        exit_spread_cost = spread[i] / 2.0 / TICK_SIZE
        pnl_ticks = current_pnl_ticks - exit_spread_cost - COMMISSION_TICKS

        trades.append({
            'entry_bar': entry_bar,
            'exit_bar': i,
            'direction': entry_direction,
            'entry_price': entry_price,
            'exit_price': mid[i],
            'pnl_ticks': pnl_ticks,
            'pnl_dollars': pnl_ticks * TICK_VALUE,
            'bars_held': i - entry_bar,
            'exit_reason': 'eod',
            'peak_pnl_ticks': peak_pnl_ticks,
            'signal_strength': abs(predictions[entry_bar]),
        })

    return summarize_trades(trades, signal_count)


def summarize_trades(trades: List[Dict], signal_count: int) -> Dict:
    """Summarize trade list into metrics."""
    if not trades:
        return {
            'total_trades': 0,
            'total_pnl_dollars': 0.0,
            'total_pnl_ticks': 0.0,
            'signal_count': signal_count,
        }

    pnls = [t['pnl_dollars'] for t in trades]
    pnl_ticks = [t['pnl_ticks'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    return {
        'total_trades': len(trades),
        'total_pnl_dollars': sum(pnls),
        'total_pnl_ticks': sum(pnl_ticks),
        'win_rate': len(wins) / len(trades) if trades else 0,
        'avg_pnl_per_trade': sum(pnls) / len(trades),
        'avg_win': sum(wins) / len(wins) if wins else 0,
        'avg_loss': sum(losses) / len(losses) if losses else 0,
        'max_win': max(pnls) if pnls else 0,
        'max_loss': min(pnls) if pnls else 0,
        'avg_bars_held': np.mean([t['bars_held'] for t in trades]),
        'avg_peak_pnl': np.mean([t['peak_pnl_ticks'] for t in trades]),
        'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0,
        'signal_count': signal_count,
        'exit_reasons': {
            'time': sum(1 for t in trades if t['exit_reason'] == 'time'),
            'trailing': sum(1 for t in trades if t['exit_reason'] == 'trailing'),
            'tp': sum(1 for t in trades if t['exit_reason'] == 'tp'),
            'eod': sum(1 for t in trades if t['exit_reason'] == 'eod'),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Market Order Simulator")
    parser.add_argument('--signal', type=str, required=True,
                       help='Signal name (prefix of NPZ files in signal_predictions/)')
    parser.add_argument('--n-days', type=int, default=50)
    parser.add_argument('--sweep', action='store_true',
                       help='Run parameter sweep instead of single config')

    # Single config params
    parser.add_argument('--thresh', type=float, default=1.0)
    parser.add_argument('--hold', type=int, default=600, help='Hold bars (600=60s at 10Hz)')
    parser.add_argument('--trail', type=float, default=8.0)
    parser.add_argument('--tp', type=float, default=0.0)
    parser.add_argument('--cooldown', type=int, default=10)
    parser.add_argument('--max-trades', type=int, default=0,
                       help='Max signals per day (0=unlimited)')

    args = parser.parse_args()

    logger.info(f"Market Order Simulator — {args.signal}")
    logger.info(f"  Cost model: {TOTAL_COST_TICKS:.2f} ticks/RT "
                f"({SPREAD_COST_TICKS:.1f}t spread + {COMMISSION_TICKS:.2f}t commission)")

    # Find days
    all_days = discover_days(args.n_days)
    logger.info(f"  Available days: {len(all_days)}")

    # Match signal files
    matched_days = []
    for date_str, feat_path in all_days:
        sig_path = SIGNAL_DIR / f"{args.signal}_{date_str}.npz"
        if sig_path.exists():
            matched_days.append((date_str, feat_path, sig_path))

    logger.info(f"  Matched signal files: {len(matched_days)}")

    if not matched_days:
        logger.error("No matching signal files found!")
        return

    if args.sweep:
        # Parameter sweep
        thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        holds = [100, 300, 600, 1200]  # 10s, 30s, 60s, 120s
        trails = [4, 8, 12, 0]
        cooldowns = [10]
        max_trades_list = [0, 50, 100]

        total_combos = len(thresholds) * len(holds) * len(trails) * len(max_trades_list)
        logger.info(f"\n  SWEEP MODE: {total_combos} parameter combos × {len(matched_days)} days")
        logger.info(f"  Thresholds: {thresholds}")
        logger.info(f"  Holds: {holds} bars ({[h/BARS_PER_SEC for h in holds]}s)")
        logger.info(f"  Trails: {trails}")
        logger.info(f"  Max trades/day: {max_trades_list}")

        all_combo_results = {}
        t0 = time.time()

        for thresh in thresholds:
            for hold in holds:
                for trail in trails:
                    for max_t in max_trades_list:
                        combo_key = f"st{thresh}_h{hold}_tr{trail}_mt{max_t}"
                        day_results = []

                        for date_str, feat_path, sig_path in matched_days:
                            try:
                                mid, spread, preds = load_features_and_signal(feat_path, sig_path)
                                result = simulate_market_orders(
                                    mid, spread, preds,
                                    signal_threshold=thresh,
                                    hold_bars=hold,
                                    trailing_stop_ticks=trail,
                                    cooldown_bars=10,
                                    max_signals_per_day=max_t,
                                )
                                result['date'] = date_str
                                day_results.append(result)
                                del mid, spread, preds
                            except Exception as e:
                                logger.error(f"  {date_str} {combo_key}: {e}")

                        # Aggregate
                        active_days = [d for d in day_results if d['total_trades'] > 0]
                        if active_days:
                            total_pnl = sum(d['total_pnl_dollars'] for d in active_days)
                            total_trades = sum(d['total_trades'] for d in active_days)
                            mean_wr = np.mean([d['win_rate'] for d in active_days])
                            pos_days = sum(1 for d in active_days if d['total_pnl_dollars'] > 0)

                            all_combo_results[combo_key] = {
                                'thresh': thresh,
                                'hold': hold,
                                'trail': trail,
                                'max_trades': max_t,
                                'n_days': len(active_days),
                                'total_pnl': total_pnl,
                                'mean_daily_pnl': total_pnl / len(active_days),
                                'total_trades': total_trades,
                                'mean_wr': float(mean_wr),
                                'pos_days': pos_days,
                                'pct_pos_days': 100 * pos_days / len(active_days),
                                'mean_trades_per_day': total_trades / len(active_days),
                                'per_day': [{
                                    'date': d['date'],
                                    'pnl': d['total_pnl_dollars'],
                                    'trades': d['total_trades'],
                                    'wr': d['win_rate'],
                                } for d in active_days],
                            }

                        elapsed = time.time() - t0
                        done_combos = len(all_combo_results)
                        if done_combos % 20 == 0 and done_combos > 0:
                            logger.info(f"  [{done_combos}/{total_combos}] {elapsed:.0f}s elapsed")

        # Print sweep summary
        logger.info(f"\n{'='*100}")
        logger.info(f"MARKET ORDER SWEEP SUMMARY — {args.signal}")
        logger.info(f"Cost: {TOTAL_COST_TICKS:.2f} ticks/RT | Days: {len(matched_days)}")
        logger.info(f"{'='*100}")

        sorted_combos = sorted(all_combo_results.items(),
                               key=lambda x: x[1].get('mean_daily_pnl', -9999), reverse=True)

        logger.info(f"\n{'Combo':>35} {'n_days':>6} {'mean_pnl':>10} {'total_pnl':>10} "
                     f"{'wr':>6} {'trades/d':>9} {'%pos':>6}")
        logger.info("-" * 90)

        for name, r in sorted_combos[:30]:
            logger.info(f"{name:>35} {r['n_days']:>6} {r['mean_daily_pnl']:>+10.2f} "
                       f"{r['total_pnl']:>+10.2f} {r['mean_wr']*100:>5.1f}% "
                       f"{r['mean_trades_per_day']:>8.1f} {r['pct_pos_days']:>5.1f}%")

        # Save
        out_file = RESULTS_DIR / f"market_order_sweep_{args.signal}_{_ts}.json"
        with open(str(out_file), 'w') as f:
            json.dump({
                'signal': args.signal,
                'timestamp': _ts,
                'cost_ticks_rt': TOTAL_COST_TICKS,
                'n_days': len(matched_days),
                'combos': all_combo_results,
            }, f, indent=2, default=str)
        logger.info(f"\nSaved: {out_file}")
        logger.info(f"Total time: {time.time()-t0:.0f}s")

    else:
        # Single config
        logger.info(f"  Config: thresh={args.thresh}, hold={args.hold} bars "
                    f"({args.hold/BARS_PER_SEC:.0f}s), trail={args.trail}t, "
                    f"cooldown={args.cooldown}, max_trades={args.max_trades}")

        all_results = []
        for date_str, feat_path, sig_path in matched_days:
            try:
                mid, spread, preds = load_features_and_signal(feat_path, sig_path)
                result = simulate_market_orders(
                    mid, spread, preds,
                    signal_threshold=args.thresh,
                    hold_bars=args.hold,
                    trailing_stop_ticks=args.trail,
                    take_profit_ticks=args.tp,
                    cooldown_bars=args.cooldown,
                    max_signals_per_day=args.max_trades,
                )
                result['date'] = date_str
                all_results.append(result)

                if len(all_results) % 5 == 0 or len(all_results) == 1:
                    logger.info(f"  [{len(all_results)}/{len(matched_days)}] {date_str}  "
                              f"trades={result['total_trades']}  "
                              f"pnl=${result['total_pnl_dollars']:+.2f}  "
                              f"wr={result['win_rate']*100:.1f}%")

                del mid, spread, preds
            except Exception as e:
                logger.error(f"  {date_str}: {e}")

        # Summary
        active = [r for r in all_results if r['total_trades'] > 0]
        if active:
            total_pnl = sum(r['total_pnl_dollars'] for r in active)
            total_trades = sum(r['total_trades'] for r in active)
            mean_wr = np.mean([r['win_rate'] for r in active])
            pos_days = sum(1 for r in active if r['total_pnl_dollars'] > 0)

            logger.info(f"\n{'='*70}")
            logger.info(f"MARKET ORDER SIM — {args.signal}")
            logger.info(f"  thresh={args.thresh}, hold={args.hold/BARS_PER_SEC:.0f}s, "
                       f"trail={args.trail}t")
            logger.info(f"  Cost: {TOTAL_COST_TICKS:.2f} ticks/RT")
            logger.info(f"{'='*70}")
            logger.info(f"  Days: {len(active)}")
            logger.info(f"  Total PnL: ${total_pnl:+,.2f}")
            logger.info(f"  Mean daily PnL: ${total_pnl/len(active):+,.2f}")
            logger.info(f"  Total trades: {total_trades}")
            logger.info(f"  Mean trades/day: {total_trades/len(active):.1f}")
            logger.info(f"  Win rate: {mean_wr*100:.1f}%")
            logger.info(f"  Positive days: {pos_days}/{len(active)} "
                       f"({100*pos_days/len(active):.1f}%)")
            logger.info(f"{'='*70}")
        else:
            logger.info("No trades generated!")


if __name__ == '__main__':
    main()
