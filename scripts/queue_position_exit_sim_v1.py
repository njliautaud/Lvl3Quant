#!/usr/bin/env python3
"""
Queue Position Exit Simulation v1

Simulates realistic passive exit fill rates for top-percentile short signals
using queue position modeling.

KEY INSIGHT: The MBO labels (labels_1s, 5s, 10s, 30s) are ENDPOINT returns,
not MFE (Max Favorable Excursion). MFE is always >= endpoint return.
We approximate MFE by taking the maximum favorable return across all
available horizons as a lower bound.

For shorts: MFE_approx = max(-label_1s, -label_5s, -label_10s, -label_30s)
This is a LOWER BOUND on true MFE (price path may have been even more favorable
between these snapshots).

Queue position model:
- After entry, exit limit placed 1 tick below entry (for shorts)
- We're LAST in queue. Need volume to trade through our level.
- P(fill | MFE >= 1 tick) depends on how FAR past our level price went
  and how LONG it stayed there (more depth = more fills)
- Conservative: assume we need price to exceed our level by enough
  that the volume through clears the queue ahead of us.

Commission: 0.376 ticks RT
Passive exit net: +1 tick - 0.376 = +0.624 ticks
"""

import numpy as np
import json
import os
import glob
from datetime import datetime

# Constants
COMMISSION_RT_TICKS = 0.376
PASSIVE_EXIT_NET = 1.0 - COMMISSION_RT_TICKS  # +0.624 ticks

PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2/'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3/'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/queue_position_sim_v1/'

QUEUE_AHEAD_VALUES = [50, 100, 200, 300, 500, 1000]
PERCENTILE_THRESHOLDS = [1, 2, 5, 10]

# ES typical volume characteristics during RTH
# When price moves through a level, ~200-500 contracts trade at that level
# per tick of penetration depth. This is based on typical ES book depth.
VOLUME_PER_TICK_DEPTH = 300  # Conservative estimate


def get_common_dates():
    pred_dates = {os.path.basename(f).replace('_predictions.npz', '')
                  for f in glob.glob(PRED_DIR + '*_predictions.npz')}
    mbo_dates = {os.path.basename(f).replace('_mbo_events.npz', '')
                 for f in glob.glob(MBO_DIR + '*_mbo_events.npz')}
    return sorted(pred_dates & mbo_dates)


def load_day_data(date_str):
    """Load predictions and compute approximate MFE for shorts."""
    pred_data = np.load(f'{PRED_DIR}{date_str}_predictions.npz', allow_pickle=True)
    mbo_data = np.load(f'{MBO_DIR}{date_str}_mbo_events.npz')

    predictions = pred_data['predictions']  # (N, 3) for 1s/5s/10s
    pred_labels = pred_data['labels']        # (N, 3) aligned 1s/5s/10s labels
    window_size = int(pred_data['window_size'])
    stride = int(pred_data['stride'])
    n_pred = predictions.shape[0]

    # Get event indices for each prediction
    event_indices = np.arange(n_pred) * stride + window_size - 1

    # Get all horizon labels aligned to predictions
    labels_1s = pred_labels[:, 0]   # Already aligned
    labels_5s = pred_labels[:, 1]
    labels_10s = pred_labels[:, 2]
    labels_30s = mbo_data['labels_30s'][event_indices]

    # For short positions, favorable move = negative label (price drops)
    # MFE for short = max favorable excursion downward
    # Approximate MFE: max of favorable returns across horizons
    # This is a LOWER BOUND (true MFE could be higher between snapshots)
    short_favorable = np.stack([
        -labels_1s, -labels_5s, -labels_10s, -labels_30s
    ], axis=1)

    # MFE_approx = max favorable return across all horizons
    # Handle NaN: use nanmax
    mfe_approx = np.nanmax(short_favorable, axis=1)

    # Also get the endpoint return at 30s for market exit P&L
    endpoint_30s = labels_30s  # For shorts: profit = -endpoint

    # 1s prediction for signal ranking
    pred_1s = predictions[:, 0]

    return {
        'pred_1s': pred_1s,
        'mfe_approx': mfe_approx,
        'endpoint_30s': endpoint_30s,
        'labels_1s': labels_1s,
        'labels_5s': labels_5s,
        'labels_10s': labels_10s,
        'labels_30s': labels_30s,
        'n_signals': n_pred,
    }


def compute_fill_prob(mfe_ticks, queue_ahead):
    """
    Compute probability of passive exit fill given MFE and queue position.

    Model:
    - If MFE < 1 tick: price never reaches our exit level -> P(fill) = 0
    - If MFE >= 1 tick: price reaches our level. Volume traded at our level
      increases with penetration depth beyond 1 tick.
    - Each tick of additional depth ~ VOLUME_PER_TICK_DEPTH contracts trade
      at our level (as price sweeps through the book).
    - P(fill) = min(1, volume_at_level / queue_ahead)

    For a 1-tick touch (MFE exactly 1):
    - Price just barely reaches our level
    - Maybe 30-50% of queue fills (price doesn't sweep through)
    - Model: 0.3 * VOLUME_PER_TICK_DEPTH contracts trade

    For deeper penetration:
    - Linear scaling: (mfe - 0.7) * VOLUME_PER_TICK_DEPTH
    """
    if mfe_ticks < 1.0:
        return 0.0

    # Effective volume that trades at our price level
    # Conservative: use mfe_ticks - 0.7 (need some penetration beyond touch)
    effective_depth = max(0, mfe_ticks - 0.7)
    volume_at_level = effective_depth * VOLUME_PER_TICK_DEPTH

    return min(1.0, volume_at_level / max(queue_ahead, 1))


def run_simulation():
    print("=" * 70)
    print("Queue Position Exit Simulation v1")
    print("Uses approximate MFE from multi-horizon labels (lower bound)")
    print("=" * 70)

    dates = get_common_dates()
    print(f"\n{len(dates)} OOT dates: {dates[0]} to {dates[-1]}")

    # Load all data
    all_pred_1s = []
    all_mfe = []
    all_endpoint_30s = []
    all_labels = {h: [] for h in ['1s', '5s', '10s', '30s']}

    for date_str in dates:
        try:
            data = load_day_data(date_str)
            all_pred_1s.append(data['pred_1s'])
            all_mfe.append(data['mfe_approx'])
            all_endpoint_30s.append(data['endpoint_30s'])
            all_labels['1s'].append(data['labels_1s'])
            all_labels['5s'].append(data['labels_5s'])
            all_labels['10s'].append(data['labels_10s'])
            all_labels['30s'].append(data['labels_30s'])
        except Exception as e:
            print(f"  Skip {date_str}: {e}")

    pred_1s = np.concatenate(all_pred_1s)
    mfe_approx = np.concatenate(all_mfe)
    endpoint_30s = np.concatenate(all_endpoint_30s)
    labels = {h: np.concatenate(all_labels[h]) for h in all_labels}

    # Filter valid
    valid = ~np.isnan(mfe_approx) & ~np.isnan(pred_1s) & ~np.isnan(endpoint_30s)
    pred_1s = pred_1s[valid]
    mfe_approx = mfe_approx[valid]
    endpoint_30s = endpoint_30s[valid]
    for h in labels:
        labels[h] = labels[h][valid]

    n_total = len(pred_1s)
    print(f"Total valid predictions: {n_total:,}")

    # Baseline MFE stats (all signals)
    print(f"\nBaseline MFE stats (all signals, approximate lower bound):")
    print(f"  P(MFE >= 1 tick): {np.mean(mfe_approx >= 1.0)*100:.1f}%")
    print(f"  P(MFE >= 2 ticks): {np.mean(mfe_approx >= 2.0)*100:.1f}%")
    print(f"  Mean MFE: {np.nanmean(mfe_approx):.2f} ticks")
    print(f"  Median MFE: {np.nanmedian(mfe_approx):.2f} ticks")

    all_results = {}

    for pct in PERCENTILE_THRESHOLDS:
        # Short signals: most negative 1s predictions
        threshold = np.percentile(pred_1s, pct)
        short_mask = pred_1s <= threshold
        n_short = np.sum(short_mask)

        short_mfe = mfe_approx[short_mask]
        short_endpoint = endpoint_30s[short_mask]

        # Natural retrace rate for this percentile
        retrace_rate = np.mean(short_mfe >= 1.0)

        print(f"\n{'='*70}")
        print(f"TOP {pct}% SHORT SIGNALS ({n_short:,} signals)")
        print(f"  Prediction threshold: {threshold:.6f}")
        print(f"  Natural 1-tick retrace rate (MFE approx): {retrace_rate*100:.1f}%")
        print(f"  Mean MFE: {np.nanmean(short_mfe):.2f} ticks")
        print(f"  MFE distribution: p25={np.percentile(short_mfe, 25):.1f}, "
              f"p50={np.percentile(short_mfe, 50):.1f}, "
              f"p75={np.percentile(short_mfe, 75):.1f}, "
              f"p90={np.percentile(short_mfe, 90):.1f}")

        # Market exit P&L for unfilled trades
        # For shorts: profit = -endpoint_30s
        short_market_exit_pnl = -short_endpoint - COMMISSION_RT_TICKS
        avg_market_exit = np.mean(short_market_exit_pnl)
        median_market_exit = np.median(short_market_exit_pnl)
        print(f"  Market exit (30s timeout): mean={avg_market_exit:.2f}, median={median_market_exit:.2f} ticks")

        pct_results = {
            'n_signals': int(n_short),
            'prediction_threshold': float(threshold),
            'natural_retrace_rate': float(retrace_rate),
            'mean_mfe_ticks': float(np.nanmean(short_mfe)),
            'median_mfe_ticks': float(np.nanmedian(short_mfe)),
            'avg_market_exit_pnl': float(avg_market_exit),
            'queue_results': {},
        }

        print(f"\n  {'Queue':>6} {'P(fill)':>8} {'Fill%':>8} "
              f"{'PnL/Trade':>10} {'$/Trade':>10} {'Total$':>12} {'Status':>12}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")

        for qa in QUEUE_AHEAD_VALUES:
            # Compute fill probability for each signal
            fill_probs = np.array([compute_fill_prob(m, qa) for m in short_mfe])

            # Monte Carlo for stability
            np.random.seed(42)
            n_mc = 20
            mc_pnls = []
            mc_fills = []

            for mc_i in range(n_mc):
                np.random.seed(42 + mc_i)
                filled = np.random.random(n_short) < fill_probs

                # Filled trades: passive exit at +1 tick TP
                pnl_filled = np.sum(filled) * PASSIVE_EXIT_NET

                # Unfilled trades: market exit at 30s endpoint
                unfilled_pnl = np.sum(short_market_exit_pnl[~filled])

                total_pnl = pnl_filled + unfilled_pnl
                mc_pnls.append(total_pnl / n_short)
                mc_fills.append(np.mean(filled))

            avg_pnl = np.mean(mc_pnls)
            avg_fill = np.mean(mc_fills)
            effective_fill_rate = avg_fill
            total_pnl_dollars = avg_pnl * n_short * 12.50
            per_trade_dollars = avg_pnl * 12.50

            status = "PROFITABLE" if avg_pnl > 0 else "UNPROFITABLE"

            print(f"  {qa:6d} {np.mean(fill_probs)*100:7.1f}% {effective_fill_rate*100:7.1f}% "
                  f"{avg_pnl:+9.4f} ${per_trade_dollars:+9.2f} "
                  f"${total_pnl_dollars:+11,.0f} {status:>12}")

            pct_results['queue_results'][str(qa)] = {
                'queue_ahead': qa,
                'avg_fill_prob': float(np.mean(fill_probs)),
                'effective_fill_rate': float(effective_fill_rate),
                'avg_pnl_per_trade_ticks': float(avg_pnl),
                'avg_pnl_per_trade_dollars': float(per_trade_dollars),
                'total_pnl_dollars': float(total_pnl_dollars),
                'profitable': bool(avg_pnl > 0),
            }

        # Binary search for breakeven queue_ahead
        lo, hi = 1, 10000
        breakeven_qa = 0
        for _ in range(50):
            if lo > hi:
                break
            mid = (lo + hi) // 2
            fill_probs = np.array([compute_fill_prob(m, mid) for m in short_mfe])
            np.random.seed(42)
            filled = np.random.random(n_short) < fill_probs
            pnl = np.sum(filled) * PASSIVE_EXIT_NET + np.sum(short_market_exit_pnl[~filled])
            if pnl / n_short > 0:
                breakeven_qa = mid
                lo = mid + 1
            else:
                hi = mid - 1

        pct_results['breakeven_queue_ahead'] = breakeven_qa
        print(f"\n  BREAKEVEN queue position: {breakeven_qa} contracts ahead")
        print(f"  (Profitable if queue ahead < {breakeven_qa})")

        all_results[f'top{pct}pct'] = pct_results

    # KEY DIAGNOSTIC: opportunity cost analysis
    print(f"\n{'='*70}")
    print("OPPORTUNITY COST ANALYSIS (Top 2% shorts)")
    print(f"{'='*70}")

    threshold_2 = np.percentile(pred_1s, 2)
    mask_2 = pred_1s <= threshold_2
    short_mfe_2 = mfe_approx[mask_2]
    short_ep_2 = endpoint_30s[mask_2]
    short_return = -short_ep_2  # Short perspective: positive = profit

    retrace_mask_2 = short_mfe_2 >= 1.0
    retrace_ep = short_return[retrace_mask_2]
    no_retrace_ep = short_return[~retrace_mask_2]

    print(f"\n  HOLD-TO-30s BASELINE (no TP):")
    print(f"    Avg return: {np.mean(short_return):.3f} ticks")
    print(f"    Net commission: {np.mean(short_return) - COMMISSION_RT_TICKS:.3f} ticks/trade")
    print(f"    Profitable: {'YES' if np.mean(short_return) - COMMISSION_RT_TICKS > 0 else 'NO'}")
    print(f"\n  CONDITIONAL RETURNS:")
    print(f"    E[return | MFE >= 1 tick]: {np.mean(retrace_ep):.3f} ticks ({np.sum(retrace_mask_2):,} signals)")
    print(f"    E[return | MFE < 1 tick]:  {np.mean(no_retrace_ep):.3f} ticks ({np.sum(~retrace_mask_2):,} signals)")
    print(f"\n  PASSIVE EXIT vs HOLD:")
    print(f"    Passive exit net: +{PASSIVE_EXIT_NET:.3f} ticks")
    print(f"    Hold-to-30s net (MFE>=1): {np.mean(retrace_ep) - COMMISSION_RT_TICKS:.3f} ticks")
    print(f"    OPPORTUNITY COST per filled exit: {np.mean(retrace_ep) - COMMISSION_RT_TICKS - PASSIVE_EXIT_NET:.3f} ticks")
    print(f"\n  CONCLUSION: Passive 1-tick TP LOSES {np.mean(retrace_ep) - COMMISSION_RT_TICKS - PASSIVE_EXIT_NET:.3f} ticks/trade vs holding.")
    print(f"  The signal's average favorable move ({np.mean(retrace_ep):.1f} ticks) far exceeds the 1-tick TP.")
    print(f"  Passive exit is suboptimal -- it caps winners while still taking full losers.")

    # Also test different TP levels
    print(f"\n{'='*70}")
    print("TP LEVEL OPTIMIZATION (Top 2% shorts, perfect fill assumption)")
    print(f"{'='*70}")
    print(f"\n  {'TP':>4} {'Fill%':>8} {'Net/Fill':>9} {'Avg PnL':>9} {'Status':>12}")
    print(f"  {'-'*4} {'-'*8} {'-'*9} {'-'*9} {'-'*12}")

    for tp_ticks in [1, 2, 3, 4, 5, 7, 10]:
        # What fraction of trades have MFE >= tp_ticks?
        fills = short_mfe_2 >= tp_ticks
        fill_rate = np.mean(fills)
        net_per_fill = tp_ticks - COMMISSION_RT_TICKS
        # Filled trades get TP, unfilled get 30s market exit
        avg_pnl = fill_rate * net_per_fill + (1 - fill_rate) * (np.mean(short_return[~fills]) - COMMISSION_RT_TICKS) if np.any(~fills) else fill_rate * net_per_fill
        status = "PROFITABLE" if avg_pnl > 0 else "UNPROFITABLE"
        print(f"  {tp_ticks:4d} {fill_rate*100:7.1f}% {net_per_fill:+8.3f} {avg_pnl:+8.3f} {status:>12}")

    # Additional analysis: fill rate sensitivity
    print(f"\n{'='*70}")
    print("FILL RATE SENSITIVITY (Top 2% shorts, 1-tick TP)")
    print("What fill rate is needed for profitability?")
    print(f"{'='*70}")

    short_mfe_2 = mfe_approx[mask_2]
    short_ep_2 = endpoint_30s[mask_2]
    mkt_exit_2 = -short_ep_2 - COMMISSION_RT_TICKS
    n_2 = np.sum(mask_2)

    # Only consider signals where MFE >= 1 (retrace happens)
    retrace_mask = short_mfe_2 >= 1.0
    n_retrace = np.sum(retrace_mask)
    no_retrace_pnl = np.sum(mkt_exit_2[~retrace_mask])  # These always market exit

    print(f"\n  Signals with MFE >= 1 tick: {n_retrace:,} / {n_2:,} ({n_retrace/n_2*100:.1f}%)")
    print(f"  Signals with MFE < 1 tick: {n_2 - n_retrace:,} (always market exit)")
    print(f"  Market exit PnL for no-retrace signals: {no_retrace_pnl/n_2:.4f} ticks/signal avg impact")

    fill_rate_results = {}
    print(f"\n  {'Fill%':>8} {'PnL/Trade':>10} {'$/Trade':>10} {'Status':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*12}")

    breakeven_fill_rate = None
    for fill_pct in range(50, 101, 5):
        fill_rate = fill_pct / 100.0
        # Among signals where retrace happens, this fraction gets filled
        n_filled = int(n_retrace * fill_rate)
        # Sort by MFE descending (higher MFE = more likely to fill)
        retrace_indices = np.where(retrace_mask)[0]
        mfe_order = np.argsort(-short_mfe_2[retrace_indices])
        filled_idx = retrace_indices[mfe_order[:n_filled]]
        unfilled_retrace_idx = retrace_indices[mfe_order[n_filled:]]

        pnl_filled = n_filled * PASSIVE_EXIT_NET
        pnl_unfilled_retrace = np.sum(mkt_exit_2[unfilled_retrace_idx])
        pnl_no_retrace = no_retrace_pnl

        total_pnl = pnl_filled + pnl_unfilled_retrace + pnl_no_retrace
        avg_pnl = total_pnl / n_2
        dollars = avg_pnl * 12.50

        status = "PROFITABLE" if avg_pnl > 0 else "UNPROFITABLE"
        print(f"  {fill_pct:7d}% {avg_pnl:+9.4f} ${dollars:+9.2f} {status:>12}")

        fill_rate_results[fill_pct] = {
            'fill_rate_pct': fill_pct,
            'avg_pnl_ticks': float(avg_pnl),
            'profitable': bool(avg_pnl > 0),
        }

        if avg_pnl > 0 and breakeven_fill_rate is None:
            # Refine
            for fine_pct in range(fill_pct - 5, fill_pct + 1):
                fr = fine_pct / 100.0
                nf = int(n_retrace * fr)
                fi = retrace_indices[mfe_order[:nf]]
                ufi = retrace_indices[mfe_order[nf:]]
                tp = nf * PASSIVE_EXIT_NET + np.sum(mkt_exit_2[ufi]) + no_retrace_pnl
                if tp / n_2 > 0:
                    breakeven_fill_rate = fine_pct
                    break

    if breakeven_fill_rate:
        print(f"\n  BREAKEVEN fill rate: ~{breakeven_fill_rate}% of retrace signals")
        print(f"  (= {breakeven_fill_rate/100 * np.mean(retrace_mask)*100:.1f}% of ALL signals)")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'commission_rt_ticks': COMMISSION_RT_TICKS,
            'passive_exit_net_ticks': PASSIVE_EXIT_NET,
            'volume_per_tick_depth': VOLUME_PER_TICK_DEPTH,
            'n_dates': len(dates),
            'date_range': f'{dates[0]} to {dates[-1]}',
            'total_predictions': int(n_total),
            'mfe_method': 'approximate_lower_bound_from_multi_horizon_labels',
        },
        'baseline': {
            'all_signals_mfe_gte_1tick': float(np.mean(mfe_approx >= 1.0)),
            'all_signals_mfe_gte_2ticks': float(np.mean(mfe_approx >= 2.0)),
            'all_signals_mean_mfe': float(np.nanmean(mfe_approx)),
        },
        'results_by_percentile': all_results,
        'fill_rate_sensitivity': fill_rate_results,
        'breakeven_fill_rate_pct': breakeven_fill_rate,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print("EXECUTIVE SUMMARY")
    print(f"{'='*70}")
    top2 = all_results.get('top2pct', {})
    if top2:
        print(f"\nTop 2% short signals ({top2['n_signals']:,} signals across {len(dates)} days):")
        print(f"  Approximate MFE retrace rate (>= 1 tick): {top2['natural_retrace_rate']*100:.1f}%")
        print(f"  Breakeven queue position: {top2['breakeven_queue_ahead']} contracts")
        print(f"  Market exit avg PnL: {top2['avg_market_exit_pnl']:.2f} ticks")
        if breakeven_fill_rate:
            print(f"  Breakeven fill rate: {breakeven_fill_rate}% of retrace signals")
        print(f"\n  NOTE: MFE is approximated using max favorable endpoint across")
        print(f"  1s/5s/10s/30s horizons. True MFE is HIGHER (price path between")
        print(f"  snapshots). Real fill rates should be BETTER than shown here.")

    print(f"\nResults saved to {OUTPUT_DIR}results.json")


if __name__ == '__main__':
    run_simulation()
