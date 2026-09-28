"""
Market Making with Alpha-Skewed Quotes — Simulation

Instead of paying the spread as a directional trader, this simulates EARNING the
spread as a market maker who uses our alpha signals to reduce adverse selection.

Strategies tested:
  1. Naive MM: always quote at best, earn spread on fills
  2. Direction-Skewed: pull quotes on the side of predicted moves
  3. Magnitude-Width: widen spread during high-magnitude (volatile) bars
  4. Combined: direction skew + magnitude width
  5. Selective: only quote during low-magnitude bars (calm markets)

Fill model (simplified but conservative):
  - Bid fill: mid moves DOWN through our bid quote in the next bar
  - Ask fill: mid moves UP through our ask quote in the next bar
  - Commission: 0.248 ticks per round-trip (half per side = 0.124 per fill)

Uses pre-computed direction + magnitude predictions from the walk-forward pipeline.

Usage:
    python alpha_discovery/mm_alpha_sim.py [--predictions PATH] [--n-days N] [--oos-split N]
"""

import sys
import gc
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("mm_alpha")

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE = 0.25            # ES tick size in points
TICK_VALUE = 12.50          # $ per tick per contract
COMMISSION_TICKS_RT = 0.376  # $4.70 RT / $12.50 tick (HC #52)
COMMISSION_PER_FILL = COMMISSION_TICKS_RT / 2.0  # Half per side
ES_SPREAD_TICKS = 1.0       # ES typical spread = 1 tick during RTH
BARS_PER_SEC = 10           # 100ms bars

MAX_INVENTORY = 3           # Max contracts before halting quotes
MIN_FILL_GAP = 5            # Min bars between fills (0.5s) to avoid double-counting
FLAT_EOD = True             # Force flatten at end of day


# ============================================================================
# LOAD PREDICTIONS
# ============================================================================

def load_predictions(path: str) -> Dict:
    """Load pre-computed walk-forward predictions."""
    npz = np.load(path, allow_pickle=True)
    data = {
        'mid_prices': npz['mid_prices'].astype(np.float64),
        'dir_preds': npz['direction_preds'].astype(np.float32),
        'mag_preds': npz['magnitude_preds'].astype(np.float32),
        'dir_target': npz['direction_target'].astype(np.float32),
        'mag_target': npz['magnitude_target'].astype(np.float32),
        'day_boundaries': npz['day_boundaries'].astype(np.int64),
    }
    N = len(data['mid_prices'])
    n_days = len(data['day_boundaries']) - 1
    n_valid = int(np.isfinite(data['dir_preds']).sum())
    logger.info(f"Loaded {N:,} bars, {n_days} days, {n_valid:,} valid predictions")
    return data


# ============================================================================
# CORE MM SIMULATION ENGINE
# ============================================================================

def simulate_mm(
    mid_prices: np.ndarray,
    day_boundaries: np.ndarray,
    bid_offsets: np.ndarray,    # Per-bar bid offset in ticks (0 = at best bid)
    ask_offsets: np.ndarray,    # Per-bar ask offset in ticks (0 = at best ask)
    quoting_mask: np.ndarray,   # Per-bar bool: True = actively quoting
    label: str = "Strategy",
) -> Dict:
    """
    Core MM simulation engine.

    At each bar where quoting_mask is True:
      bid_quote = mid - half_spread - bid_offset * tick_size
      ask_quote = mid + half_spread + ask_offset * tick_size

    A fill occurs when next-bar mid crosses our quote level.
    Inventory tracked with position limits. Force flat at EOD.

    Args:
        bid_offsets: additional ticks BELOW best bid (0 = at best, 1 = 1 tick deeper)
        ask_offsets: additional ticks ABOVE best ask (0 = at best, 1 = 1 tick deeper)
        quoting_mask: whether we're actively quoting at this bar
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    half_spread = ES_SPREAD_TICKS * TICK_SIZE / 2.0  # 0.125 points

    # Pre-compute quote levels
    bid_quote = mid_prices - half_spread - bid_offsets * TICK_SIZE
    ask_quote = mid_prices + half_spread + ask_offsets * TICK_SIZE

    # Detect potential fills (next-bar mid crosses quote)
    mid_next = np.roll(mid_prices, -1)
    mid_next[-1] = np.nan

    can_bid_fill = mid_next <= bid_quote  # mid drops to our bid
    can_ask_fill = mid_next >= ask_quote  # mid rises to our ask

    # Walk through bars
    inventory = 0
    avg_entry_price = 0.0
    total_gross_spread = 0.0
    total_adverse_sel = 0.0
    total_commission = 0.0
    total_realized_pnl = 0.0
    total_inventory_pnl = 0.0

    n_bid_fills = 0
    n_ask_fills = 0
    n_round_trips = 0
    last_fill_bar = -MIN_FILL_GAP

    daily_pnl = np.zeros(n_days)
    daily_fills = np.zeros(n_days)
    daily_rt = np.zeros(n_days)  # round trips per day

    # Pre-compute day index per bar for fast lookup
    bar_to_day = np.zeros(N, dtype=np.int32)
    for d in range(n_days):
        bar_to_day[day_boundaries[d]:day_boundaries[d + 1]] = d

    for i in range(N - 1):
        if not np.isfinite(mid_prices[i]) or not np.isfinite(mid_prices[i + 1]):
            continue
        if not quoting_mask[i]:
            continue
        if i - last_fill_bar < MIN_FILL_GAP:
            continue

        d = bar_to_day[i]
        current_mid = mid_prices[i]
        next_mid = mid_prices[i + 1]

        # Check for bid fill (we BUY)
        if can_bid_fill[i] and inventory < MAX_INVENTORY:
            fill_px = float(bid_quote[i])
            # Adverse selection: how much did mid move against us after fill?
            adv = max(0.0, fill_px - next_mid)  # We bought, price went down further

            if inventory <= 0 and abs(inventory) > 0:
                # Closing short position — this is a round trip
                rt_pnl = (avg_entry_price - fill_px) * abs(inventory) * (TICK_VALUE / TICK_SIZE)
                total_realized_pnl += rt_pnl
                total_gross_spread += abs(avg_entry_price - fill_px) * (TICK_VALUE / TICK_SIZE)
                n_round_trips += 1
                daily_rt[d] += 1
                inventory = 0
                avg_entry_price = 0.0

            # Open/add to long
            if inventory == 0:
                avg_entry_price = fill_px
            else:
                avg_entry_price = (avg_entry_price * inventory + fill_px) / (inventory + 1)
            inventory += 1

            total_adverse_sel += adv * (TICK_VALUE / TICK_SIZE)
            total_commission += COMMISSION_PER_FILL * TICK_VALUE
            n_bid_fills += 1
            last_fill_bar = i
            daily_fills[d] += 1

        # Check for ask fill (we SELL)
        elif can_ask_fill[i] and inventory > -MAX_INVENTORY:
            fill_px = float(ask_quote[i])
            adv = max(0.0, next_mid - fill_px)  # We sold, price went up further

            if inventory >= 1:
                # Closing long position — round trip
                rt_pnl = (fill_px - avg_entry_price) * inventory * (TICK_VALUE / TICK_SIZE)
                total_realized_pnl += rt_pnl
                total_gross_spread += abs(fill_px - avg_entry_price) * (TICK_VALUE / TICK_SIZE)
                n_round_trips += 1
                daily_rt[d] += 1
                inventory = 0
                avg_entry_price = 0.0

            # Open/add to short
            if inventory == 0:
                avg_entry_price = fill_px
            else:
                avg_entry_price = (avg_entry_price * abs(inventory) + fill_px) / (abs(inventory) + 1)
            inventory -= 1

            total_adverse_sel += adv * (TICK_VALUE / TICK_SIZE)
            total_commission += COMMISSION_PER_FILL * TICK_VALUE
            n_ask_fills += 1
            last_fill_bar = i
            daily_fills[d] += 1

        # EOD flatten
        if FLAT_EOD and d < n_days - 1:
            day_end = day_boundaries[d + 1] - 1
            if i == day_end and inventory != 0:
                close_px = current_mid
                if inventory > 0:
                    mtm = (close_px - avg_entry_price) * inventory * (TICK_VALUE / TICK_SIZE)
                else:
                    mtm = (avg_entry_price - close_px) * abs(inventory) * (TICK_VALUE / TICK_SIZE)
                total_inventory_pnl += mtm
                daily_pnl[d] += mtm
                inventory = 0
                avg_entry_price = 0.0

    # Aggregate
    n_fills = n_bid_fills + n_ask_fills
    net_pnl = total_realized_pnl + total_inventory_pnl - total_commission
    daily_pnl_with_rt = daily_pnl + (total_realized_pnl / max(n_days, 1))  # approximate daily

    # Sharpe
    if n_days > 2:
        daily_net = net_pnl / n_days
        daily_std = max(np.std(daily_pnl), 1.0)
        sharpe = daily_net / daily_std * np.sqrt(252)
    else:
        sharpe = 0.0

    results = {
        'label': label,
        'n_days': n_days,
        'n_bid_fills': n_bid_fills,
        'n_ask_fills': n_ask_fills,
        'n_fills': n_fills,
        'fills_per_day': round(n_fills / max(n_days, 1), 1),
        'n_round_trips': n_round_trips,
        'rt_per_day': round(n_round_trips / max(n_days, 1), 1),
        'total_realized_pnl': round(total_realized_pnl, 2),
        'total_inventory_pnl': round(total_inventory_pnl, 2),
        'total_commission': round(total_commission, 2),
        'net_pnl': round(net_pnl, 2),
        'pnl_per_day': round(net_pnl / max(n_days, 1), 2),
        'pnl_per_rt': round(net_pnl / max(n_round_trips, 1), 2),
        'total_adverse_sel': round(total_adverse_sel, 2),
        'adverse_pct': round(total_adverse_sel / max(total_gross_spread, 1) * 100, 1),
        'sharpe': round(sharpe, 2),
    }
    return results


# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

def run_all_strategies(data: Dict, start_day: int = 0, end_day: Optional[int] = None) -> List[Dict]:
    """Run all MM strategies on the given data window."""
    mid = data['mid_prices']
    dir_p = data['dir_preds']
    mag_p = data['mag_preds']
    bounds = data['day_boundaries']
    N = len(mid)

    if end_day is None:
        end_day = len(bounds) - 1

    # Slice to requested day range
    bar_start = bounds[start_day]
    bar_end = bounds[end_day]
    mid_slice = mid[bar_start:bar_end]
    dir_slice = dir_p[bar_start:bar_end]
    mag_slice = mag_p[bar_start:bar_end]
    n_bars = len(mid_slice)

    # Recompute boundaries relative to slice
    new_bounds = bounds[start_day:end_day + 1] - bar_start
    n_days = len(new_bounds) - 1

    # Masks: valid predictions
    valid = np.isfinite(dir_slice) & np.isfinite(mag_slice) & np.isfinite(mid_slice)

    # Compute percentiles for thresholds (from valid bars only)
    dir_abs = np.abs(dir_slice)
    dir_abs_valid = dir_abs[valid]
    mag_valid = mag_slice[valid]

    dir_p70 = np.percentile(dir_abs_valid, 70) if len(dir_abs_valid) > 100 else 0.3
    dir_p90 = np.percentile(dir_abs_valid, 90) if len(dir_abs_valid) > 100 else 0.5
    mag_p50 = np.percentile(mag_valid, 50) if len(mag_valid) > 100 else 1.0
    mag_p75 = np.percentile(mag_valid, 75) if len(mag_valid) > 100 else 2.0
    mag_p90 = np.percentile(mag_valid, 90) if len(mag_valid) > 100 else 3.0

    logger.info(f"  Thresholds: |dir| P70={dir_p70:.3f} P90={dir_p90:.3f}, "
                f"mag P50={mag_p50:.2f} P75={mag_p75:.2f} P90={mag_p90:.2f}")

    results = []

    # ---- Strategy 1: Naive MM (always quote at best) ----
    bid_off = np.zeros(n_bars)
    ask_off = np.zeros(n_bars)
    mask = valid.copy()
    r = simulate_mm(mid_slice, new_bounds, bid_off, ask_off, mask, "1. Naive MM (best bid/ask)")
    results.append(r)
    logger.info(f"  {r['label']}: fills/day={r['fills_per_day']}, RT/day={r['rt_per_day']}, "
                f"PnL/day=${r['pnl_per_day']}, Sharpe={r['sharpe']}")

    # ---- Strategy 2: Direction-Skewed MM ----
    # When model predicts UP: widen bid (harder for us to buy into uptrend)
    # When model predicts DOWN: widen ask (harder for us to sell into downtrend)
    bid_off_dir = np.zeros(n_bars)
    ask_off_dir = np.zeros(n_bars)
    strong_up = valid & (dir_slice > dir_p70)
    strong_down = valid & (dir_slice < -dir_p70)
    bid_off_dir[strong_up] = 1.0     # Widen bid when predicting up (avoid adverse fill)
    ask_off_dir[strong_down] = 1.0   # Widen ask when predicting down
    r = simulate_mm(mid_slice, new_bounds, bid_off_dir, ask_off_dir, valid.copy(),
                    "2. Direction-Skewed (P70)")
    results.append(r)
    logger.info(f"  {r['label']}: fills/day={r['fills_per_day']}, RT/day={r['rt_per_day']}, "
                f"PnL/day=${r['pnl_per_day']}, Sharpe={r['sharpe']}")

    # ---- Strategy 2b: Aggressive direction skew (P90) ----
    bid_off_dir2 = np.zeros(n_bars)
    ask_off_dir2 = np.zeros(n_bars)
    very_strong_up = valid & (dir_slice > dir_p90)
    very_strong_down = valid & (dir_slice < -dir_p90)
    bid_off_dir2[very_strong_up] = 2.0
    ask_off_dir2[very_strong_down] = 2.0
    r = simulate_mm(mid_slice, new_bounds, bid_off_dir2, ask_off_dir2, valid.copy(),
                    "2b. Direction-Skewed (P90, 2-tick)")
    results.append(r)
    logger.info(f"  {r['label']}: fills/day={r['fills_per_day']}, RT/day={r['rt_per_day']}, "
                f"PnL/day=${r['pnl_per_day']}, Sharpe={r['sharpe']}")

    # ---- Strategy 3: Magnitude-Width MM ----
    # High magnitude predicted → widen spread (protect from big moves)
    # Low magnitude → tight spread (collect spread during calm)
    bid_off_mag = np.zeros(n_bars)
    ask_off_mag = np.zeros(n_bars)
    high_mag = valid & (mag_slice > mag_p75)
    very_high_mag = valid & (mag_slice > mag_p90)
    bid_off_mag[high_mag] = 1.0
    ask_off_mag[high_mag] = 1.0
    bid_off_mag[very_high_mag] = 2.0
    ask_off_mag[very_high_mag] = 2.0
    r = simulate_mm(mid_slice, new_bounds, bid_off_mag, ask_off_mag, valid.copy(),
                    "3. Magnitude-Width (widen on high mag)")
    results.append(r)
    logger.info(f"  {r['label']}: fills/day={r['fills_per_day']}, RT/day={r['rt_per_day']}, "
                f"PnL/day=${r['pnl_per_day']}, Sharpe={r['sharpe']}")

    # ---- Strategy 4: Combined (direction skew + magnitude width) ----
    bid_off_combo = np.zeros(n_bars)
    ask_off_combo = np.zeros(n_bars)
    # Magnitude widens both sides
    bid_off_combo[high_mag] += 1.0
    ask_off_combo[high_mag] += 1.0
    bid_off_combo[very_high_mag] += 1.0  # total 2 for very high
    ask_off_combo[very_high_mag] += 1.0
    # Direction skews one side
    bid_off_combo[strong_up] += 1.0
    ask_off_combo[strong_down] += 1.0
    r = simulate_mm(mid_slice, new_bounds, bid_off_combo, ask_off_combo, valid.copy(),
                    "4. Combined (dir skew + mag width)")
    results.append(r)
    logger.info(f"  {r['label']}: fills/day={r['fills_per_day']}, RT/day={r['rt_per_day']}, "
                f"PnL/day=${r['pnl_per_day']}, Sharpe={r['sharpe']}")

    # ---- Strategy 5: Selective MM (only quote during LOW magnitude) ----
    # Don't quote at all when magnitude is high — avoid the storm
    selective_mask = valid & (mag_slice < mag_p50)
    r = simulate_mm(mid_slice, new_bounds, np.zeros(n_bars), np.zeros(n_bars),
                    selective_mask, "5. Selective (only low-mag bars, P<50)")
    results.append(r)
    logger.info(f"  {r['label']}: fills/day={r['fills_per_day']}, RT/day={r['rt_per_day']}, "
                f"PnL/day=${r['pnl_per_day']}, Sharpe={r['sharpe']}")

    # ---- Strategy 5b: Selective + direction skew during quoting ----
    bid_off_sel = np.zeros(n_bars)
    ask_off_sel = np.zeros(n_bars)
    sel_up = selective_mask & (dir_slice > dir_p70)
    sel_down = selective_mask & (dir_slice < -dir_p70)
    bid_off_sel[sel_up] = 1.0
    ask_off_sel[sel_down] = 1.0
    r = simulate_mm(mid_slice, new_bounds, bid_off_sel, ask_off_sel, selective_mask,
                    "5b. Selective + Dir Skew")
    results.append(r)
    logger.info(f"  {r['label']}: fills/day={r['fills_per_day']}, RT/day={r['rt_per_day']}, "
                f"PnL/day=${r['pnl_per_day']}, Sharpe={r['sharpe']}")

    # ---- Strategy 6: Magnitude-gated directional (our current approach baseline) ----
    # For comparison: only trade when mag>3t AND strong direction signal
    # This is the directional approach we've been testing
    dir_gate = valid & (mag_slice > 3.0) & (dir_abs > dir_p90)
    n_dir_trades = dir_gate.sum()
    logger.info(f"  [Baseline] Directional mag>3t+P90: {n_dir_trades} qualifying bars "
                f"({n_dir_trades / n_bars * 100:.1f}%)")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Market Making with Alpha-Skewed Quotes')
    parser.add_argument('--predictions', type=str, default=None,
                        help='Path to predictions NPZ file')
    parser.add_argument('--oos-split', type=int, default=70,
                        help='Day index to split IS/OOS')
    args = parser.parse_args()

    # Find predictions file
    if args.predictions:
        pred_path = args.predictions
    else:
        # Auto-find most recent predictions file
        results_dir = Path(__file__).parent / 'results'
        candidates = sorted(results_dir.glob('predictions_mfe_path_*.npz'), reverse=True)
        if not candidates:
            logger.error("No predictions file found. Run edge analysis first.")
            return
        pred_path = str(candidates[0])
        logger.info(f"Auto-detected predictions: {pred_path}")

    data = load_predictions(pred_path)
    n_days = len(data['day_boundaries']) - 1
    oos_split = min(args.oos_split, n_days - 1)

    logger.info(f"\nIS: days 0..{oos_split-1} ({oos_split} days)")
    logger.info(f"OOS: days {oos_split}..{n_days-1} ({n_days - oos_split} days)")

    # ---- IN-SAMPLE ----
    logger.info(f"\n{'='*70}")
    logger.info(f"IN-SAMPLE MARKET MAKING ({oos_split} days)")
    logger.info(f"{'='*70}")
    is_results = run_all_strategies(data, start_day=0, end_day=oos_split)

    # ---- OUT-OF-SAMPLE ----
    if n_days > oos_split:
        logger.info(f"\n{'='*70}")
        logger.info(f"OUT-OF-SAMPLE MARKET MAKING ({n_days - oos_split} days)")
        logger.info(f"{'='*70}")
        oos_results = run_all_strategies(data, start_day=oos_split, end_day=n_days)
    else:
        oos_results = []

    # ---- SUMMARY TABLE ----
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Strategy':<40} {'Fills/d':>8} {'RT/d':>6} {'$/day':>10} {'$/RT':>8} {'Sharpe':>7} {'Adv%':>6}")
    logger.info("-" * 85)

    for label_prefix, results_list in [("IS", is_results), ("OOS", oos_results)]:
        if not results_list:
            continue
        for r in results_list:
            logger.info(
                f"[{label_prefix}] {r['label']:<36} {r['fills_per_day']:>8} "
                f"{r['rt_per_day']:>6} {r['pnl_per_day']:>10} {r['pnl_per_rt']:>8} "
                f"{r['sharpe']:>7} {r['adverse_pct']:>5}%"
            )
        logger.info("-" * 85)

    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = {
        'timestamp': timestamp,
        'predictions_file': str(pred_path),
        'oos_split': oos_split,
        'is_results': is_results,
        'oos_results': oos_results,
    }
    out_path = RESULTS_DIR / f'mm_alpha_{timestamp}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
