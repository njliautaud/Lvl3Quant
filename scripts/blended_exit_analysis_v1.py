#!/usr/bin/env python3
"""
Blended Exit Analysis v1
========================
Definitive profitability analysis for top-3% short signals, hold=5s, stop=2 ticks.

Five cost scenarios:
  A) Passive-passive (best case): 0.376 ticks commission only
  B) Passive entry, market exit: 0.376 + 1.0 = 1.376 ticks
  C) Market entry, passive exit: same 1.376 ticks
  D) Market both sides (worst): 0.376 + 2.0 = 2.376 ticks
  E) Blended (realistic): passive entry, attempt passive exit for 5 extra seconds,
     market exit if passive doesn't fill. Sweep fill rates 50-100%.

Labels are in TICKS (verified: std ~3.2 for 5s, consistent with all prior scripts).
Short P&L in ticks = -labels_5s (positive when price drops).
Stop-loss: triggered if labels_1s >= 2.0 (price moved 2+ ticks against short = 0.50 pts).
  Stop cost: 3 ticks loss (2 tick stop + 1 tick slippage) + 0.376 commission, always market exit.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# === CONSTANTS ===
COMMISSION_RT_TICKS = 0.376   # $4.70 / $12.50
SPREAD_TICKS = 1.0            # 1 tick wide book during RTH
STOP_TICKS = 2.0              # stop at 2 ticks against
STOP_SLIPPAGE = 1.0           # 1 tick slippage on stop market exit
STOP_LOSS_TOTAL = STOP_TICKS + STOP_SLIPPAGE  # 3 ticks realized loss on stop
HOLD_SECONDS = 5
SHORT_PERCENTILE = 3          # top 3% short signals

PRED_DIR = Path('/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2/')
MBO_DIR  = Path('/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3/')
OUT_DIR  = Path('/home/jupiter/Lvl3Quant/output/blended_exit_v1/')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_day(pred_file):
    """Load predictions and align with MBO labels for one OOT day."""
    pred_data = np.load(pred_file, allow_pickle=True)
    date_str = str(pred_data['date'])
    preds = pred_data['predictions']        # (n_windows, 3) -> [1s, 5s, 10s]
    n_windows = int(pred_data['n_windows'])
    window_size = int(pred_data['window_size'])
    stride = int(pred_data['stride'])

    mbo_file = MBO_DIR / f'{date_str}_mbo_events.npz'
    if not mbo_file.exists():
        return None

    mbo = np.load(mbo_file, allow_pickle=True)
    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = len(mbo['labels_1s']) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds = preds[valid]

    return {
        'date': date_str,
        'pred_5s': preds[:, 1],
        'label_1s': mbo['labels_1s'][indices],
        'label_5s': mbo['labels_5s'][indices],
    }


def compute_stop_adjusted_pnl(label_1s, label_5s, cost_ticks):
    """
    For each trade compute realized P&L in ticks:
      - If stop triggered (label_1s >= 2.0 ticks, i.e. price moved 2+ ticks up against short):
          realized = -(STOP_LOSS_TOTAL) - COMMISSION_RT_TICKS  (always market exit on stop)
      - Else: realized = -label_5s - cost_ticks
    Returns array of per-trade P&L in ticks.
    """
    n = len(label_1s)
    pnl = np.empty(n)

    stop_hit = label_1s >= STOP_TICKS  # price moved up by 2+ ticks within 1s (against short)
    # Stop-hit trades: fixed loss
    pnl[stop_hit] = -STOP_LOSS_TOTAL - COMMISSION_RT_TICKS
    # Non-stopped trades: hold to 5s, pay specified cost
    pnl[~stop_hit] = -label_5s[~stop_hit] - cost_ticks

    return pnl


def compute_blended_pnl(label_1s, label_5s, passive_exit_fill_rate):
    """
    Scenario E: passive entry + attempt passive exit.
    - All entries are passive (no spread cost on entry).
    - passive_exit_fill_rate fraction of non-stopped exits are passive (commission only).
    - Remaining exits are market (commission + 1 tick spread).
    - Stops are always market exit at 3-tick loss + commission.
    """
    n = len(label_1s)
    pnl = np.empty(n)

    stop_hit = label_1s >= STOP_TICKS
    n_stopped = stop_hit.sum()
    n_non_stopped = (~stop_hit).sum()

    # Stopped trades
    pnl[stop_hit] = -STOP_LOSS_TOTAL - COMMISSION_RT_TICKS

    # Non-stopped: split by fill rate
    if n_non_stopped > 0:
        non_stop_labels = label_5s[~stop_hit]
        # Deterministic split: fraction that fill passively
        n_passive = int(round(n_non_stopped * passive_exit_fill_rate))
        n_market = n_non_stopped - n_passive

        # Sort by how favorable: most favorable exits (largest drop = most negative label)
        # get passive fills first (they're the ones that retrace to our limit)
        sort_idx = np.argsort(non_stop_labels)  # most negative first = best for shorts
        non_stop_pnl = np.empty(n_non_stopped)

        # Passive exits: commission only
        non_stop_pnl[sort_idx[:n_passive]] = -non_stop_labels[sort_idx[:n_passive]] - COMMISSION_RT_TICKS
        # Market exits: commission + 1 tick spread
        non_stop_pnl[sort_idx[n_passive:]] = -non_stop_labels[sort_idx[n_passive:]] - COMMISSION_RT_TICKS - SPREAD_TICKS

        pnl[~stop_hit] = non_stop_pnl

    return pnl


def daily_metrics(dates, pnl_per_trade, dates_arr):
    """Compute per-day P&L, then Sharpe, green day %, etc."""
    unique_dates = sorted(set(dates_arr))
    daily_pnl = []
    daily_trades = []
    for d in unique_dates:
        mask = dates_arr == d
        day_pnl = pnl_per_trade[mask]
        daily_pnl.append(day_pnl.sum())
        daily_trades.append(len(day_pnl))

    daily_pnl = np.array(daily_pnl)
    daily_trades = np.array(daily_trades)

    n_days = len(daily_pnl)
    if n_days < 2:
        return {'n_days': n_days, 'sharpe': float('nan'), 'green_pct': float('nan')}

    mean_daily = daily_pnl.mean()
    std_daily = daily_pnl.std(ddof=1)
    sharpe = mean_daily / std_daily * math.sqrt(252) if std_daily > 0 else float('nan')

    green_days = (daily_pnl > 0).sum()
    green_pct = green_days / n_days * 100

    return {
        'n_days': n_days,
        'total_pnl_ticks': float(daily_pnl.sum()),
        'mean_daily_pnl': float(mean_daily),
        'std_daily_pnl': float(std_daily),
        'sharpe_annualized': float(sharpe),
        'green_days': int(green_days),
        'green_pct': float(green_pct),
        'mean_trades_per_day': float(daily_trades.mean()),
    }


def main():
    # Load all OOT days
    pred_files = sorted(PRED_DIR.glob('*_predictions.npz'))
    print(f'Found {len(pred_files)} prediction files')

    all_data = []
    for pf in pred_files:
        d = load_day(pf)
        if d is not None:
            all_data.append(d)
    print(f'Loaded {len(all_data)} days with matched MBO data')

    # Concatenate
    all_pred_5s = np.concatenate([d['pred_5s'] for d in all_data])
    all_label_1s = np.concatenate([d['label_1s'] for d in all_data])
    all_label_5s = np.concatenate([d['label_5s'] for d in all_data])
    all_dates = np.concatenate([np.full(len(d['pred_5s']), d['date']) for d in all_data])

    # Remove NaNs
    valid = ~(np.isnan(all_label_1s) | np.isnan(all_label_5s) | np.isnan(all_pred_5s))
    all_pred_5s = all_pred_5s[valid]
    all_label_1s = all_label_1s[valid]
    all_label_5s = all_label_5s[valid]
    all_dates = all_dates[valid]

    print(f'Total valid predictions: {len(all_pred_5s):,}')

    # Select top 3% short signals (bottom 3 percentile of pred_5s)
    threshold = np.percentile(all_pred_5s, SHORT_PERCENTILE)
    short_mask = all_pred_5s <= threshold
    n_short = short_mask.sum()

    s_label_1s = all_label_1s[short_mask]
    s_label_5s = all_label_5s[short_mask]
    s_dates = all_dates[short_mask]

    print(f'\nTop {SHORT_PERCENTILE}% short signals: {n_short:,} trades')
    print(f'  Prediction threshold: {threshold:.6f}')
    print(f'  Mean label_5s: {s_label_5s.mean():.4f} ticks')
    print(f'  Mean gross short P&L: {-s_label_5s.mean():.4f} ticks (before costs)')

    # Check stop rate
    stop_rate = (s_label_1s >= STOP_TICKS).mean()
    print(f'  Stop-loss trigger rate (label_1s >= {STOP_TICKS}): {stop_rate*100:.2f}%')

    # === SCENARIOS A-D ===
    scenarios = {}

    # A: Passive-passive
    cost_a = COMMISSION_RT_TICKS
    pnl_a = compute_stop_adjusted_pnl(s_label_1s, s_label_5s, cost_a)
    metrics_a = daily_metrics(None, pnl_a, s_dates)
    scenarios['A_passive_passive'] = {
        'cost_ticks': cost_a,
        'mean_pnl_per_trade': float(pnl_a.mean()),
        'win_rate': float((pnl_a > 0).mean() * 100),
        'n_trades': int(n_short),
        **metrics_a,
    }

    # B: Passive entry, market exit
    cost_b = COMMISSION_RT_TICKS + SPREAD_TICKS
    pnl_b = compute_stop_adjusted_pnl(s_label_1s, s_label_5s, cost_b)
    metrics_b = daily_metrics(None, pnl_b, s_dates)
    scenarios['B_passive_entry_market_exit'] = {
        'cost_ticks': cost_b,
        'mean_pnl_per_trade': float(pnl_b.mean()),
        'win_rate': float((pnl_b > 0).mean() * 100),
        'n_trades': int(n_short),
        **metrics_b,
    }

    # C: Market entry, passive exit (same cost as B)
    cost_c = COMMISSION_RT_TICKS + SPREAD_TICKS
    pnl_c = compute_stop_adjusted_pnl(s_label_1s, s_label_5s, cost_c)
    metrics_c = daily_metrics(None, pnl_c, s_dates)
    scenarios['C_market_entry_passive_exit'] = {
        'cost_ticks': cost_c,
        'mean_pnl_per_trade': float(pnl_c.mean()),
        'win_rate': float((pnl_c > 0).mean() * 100),
        'n_trades': int(n_short),
        **metrics_c,
    }

    # D: Market both sides
    cost_d = COMMISSION_RT_TICKS + 2 * SPREAD_TICKS
    pnl_d = compute_stop_adjusted_pnl(s_label_1s, s_label_5s, cost_d)
    metrics_d = daily_metrics(None, pnl_d, s_dates)
    scenarios['D_market_both'] = {
        'cost_ticks': cost_d,
        'mean_pnl_per_trade': float(pnl_d.mean()),
        'win_rate': float((pnl_d > 0).mean() * 100),
        'n_trades': int(n_short),
        **metrics_d,
    }

    # === SCENARIO E: BLENDED ===
    fill_rates = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    blended_results = {}

    for fr in fill_rates:
        pnl_e = compute_blended_pnl(s_label_1s, s_label_5s, fr)
        metrics_e = daily_metrics(None, pnl_e, s_dates)
        key = f'E_blended_fill{int(fr*100)}pct'
        blended_results[key] = {
            'passive_exit_fill_rate': fr,
            'mean_pnl_per_trade': float(pnl_e.mean()),
            'win_rate': float((pnl_e > 0).mean() * 100),
            'n_trades': int(n_short),
            **metrics_e,
        }

    scenarios.update(blended_results)

    # === SUMMARY TABLE ===
    print('\n' + '='*100)
    print(f'BLENDED EXIT ANALYSIS — Top {SHORT_PERCENTILE}% Short Signals, Hold={HOLD_SECONDS}s, Stop={STOP_TICKS} ticks')
    print(f'OOT days: {metrics_a["n_days"]}, Trades: {n_short:,}, Stop rate: {stop_rate*100:.1f}%')
    print('='*100)
    print(f'{"Scenario":<35} {"Cost":>6} {"Net/Trade":>10} {"WR%":>6} {"Sharpe":>8} {"Green%":>7} {"Tot P&L":>10}')
    print('-'*100)

    for name, m in scenarios.items():
        cost_str = f'{m.get("cost_ticks", "-"):>6.3f}' if 'cost_ticks' in m else f'{"blend":>6}'
        label = name.replace('_', ' ')
        print(f'{label:<35} {cost_str} {m["mean_pnl_per_trade"]:>+10.4f} '
              f'{m["win_rate"]:>6.1f} {m["sharpe_annualized"]:>8.2f} '
              f'{m["green_pct"]:>6.1f}% {m["total_pnl_ticks"]:>+10.1f}')

    print('='*100)

    # === VERDICT ===
    print('\n--- VERDICT ---')
    b_net = scenarios['B_passive_entry_market_exit']['mean_pnl_per_trade']
    e70 = blended_results['E_blended_fill70pct']['mean_pnl_per_trade']
    e80 = blended_results['E_blended_fill80pct']['mean_pnl_per_trade']

    if b_net > 0:
        print('PROFITABLE even with passive entry + market exit (Scenario B).')
    elif e70 > 0:
        print('Profitable IF passive exit fills >=70% of the time (Scenario E, 70%+ fill rate).')
    elif e80 > 0:
        print('Marginally profitable only with >=80% passive exit fill rate.')
    else:
        print('NOT PROFITABLE under realistic exit assumptions. Edge is consumed by costs.')

    # Save results
    output = {
        'config': {
            'short_percentile': SHORT_PERCENTILE,
            'hold_seconds': HOLD_SECONDS,
            'stop_ticks': STOP_TICKS,
            'stop_slippage': STOP_SLIPPAGE,
            'commission_rt_ticks': COMMISSION_RT_TICKS,
            'spread_ticks': SPREAD_TICKS,
            'n_oot_days': metrics_a['n_days'],
            'n_trades': int(n_short),
            'stop_rate_pct': float(stop_rate * 100),
            'labels_unit': 'ticks (verified: labels are in tick units, not points)',
        },
        'scenarios': scenarios,
    }

    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f'\nResults saved to {OUT_DIR / "results.json"}')


if __name__ == '__main__':
    main()
