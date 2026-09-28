#!/usr/bin/env python3
"""
Market Order Viability Analysis v1
===================================
Analyze whether CNN-Mamba v2 predictions at extreme confidence levels
can profitably execute via MARKET ORDERS (crossing the spread).

Cost model:
  - Commission RT: $4.70 = 0.376 ticks
  - Spread crossing RT: 1.0 tick (enter at ask/bid, exit at bid/ask)
  - Total market order RT cost: 1.376 ticks

Data: CNN-Mamba v2 OOT predictions (per-date NPZ files)
  - Labels are in TICKS (forward price change / 0.25)
  - Predictions are model output (log-ret scale, not ticks)
  - Horizons: 1s, 5s, 10s

Author: Claude (autonomous execution research)
Date: 2026-05-23
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Constants ──────────────────────────────────────────────────────────
ES_TICK_POINTS = 0.25       # 1 tick = 0.25 ES points
ES_TICK_VALUE  = 12.50      # $12.50 per tick
COMMISSION_RT_TICKS = 0.376 # $4.70 / $12.50
SPREAD_CROSSING_RT  = 1.0   # 1 tick each way for market order
MKT_ORDER_COST_TICKS = COMMISSION_RT_TICKS + SPREAD_CROSSING_RT  # 1.376

CONFIDENCE_PERCENTILES = [1, 2, 3, 5, 10, 15, 20]
HORIZONS = ['1s', '5s', '10s']

# ── Data Sources ───────────────────────────────────────────────────────
# bulk_oot_v2 has IC=0.168 at 1s (48 dates, window_size=1000)
# all_oot has near-zero IC (window_size=3000, bad model) — DO NOT USE
DATA_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/market_order_viability_v1'

def load_all_predictions():
    """Load all per-date OOT prediction files."""
    pattern = os.path.join(DATA_DIR, '2*_predictions.npz')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"ERROR: No prediction files found at {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} per-date prediction files")

    all_preds = []
    all_labels = []
    all_dates = []

    for fpath in files:
        f = np.load(fpath, allow_pickle=True)
        preds = f['predictions']   # shape (N, 3) — model predictions
        labels = f['labels']       # shape (N, 3) — realized forward returns in TICKS
        date_str = str(f['date'])

        # Filter out NaN labels
        valid_mask = ~np.isnan(labels).any(axis=1)
        preds = preds[valid_mask]
        labels = labels[valid_mask]

        n = len(preds)
        dates = np.full(n, date_str, dtype='U10')

        all_preds.append(preds)
        all_labels.append(labels)
        all_dates.append(dates)

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    dates = np.concatenate(all_dates, axis=0)

    print(f"Total events: {len(preds):,}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Unique dates: {len(np.unique(dates))}")

    return preds, labels, dates


def analyze_side(preds_h, labels_h, dates, side, horizon_name):
    """
    Analyze market-order viability for one side (short or long) at one horizon.

    For SHORT: we want price to go DOWN.
      - Filter: pred < 0 (model predicts down)
      - Realized P&L in ticks: -labels (negative label = price went down = profit for short)
      - Net: realized_pnl - MKT_ORDER_COST_TICKS

    For LONG: we want price to go UP.
      - Filter: pred > 0 (model predicts up)
      - Realized P&L in ticks: +labels (positive label = price went up = profit for long)
      - Net: realized_pnl - MKT_ORDER_COST_TICKS
    """
    results = []

    if side == 'short':
        side_mask = preds_h < 0
        confidence_values = -preds_h[side_mask]  # absolute prediction magnitude
        realized_ticks = -labels_h[side_mask]     # short profits when price drops
    else:
        side_mask = preds_h > 0
        confidence_values = preds_h[side_mask]
        realized_ticks = labels_h[side_mask]      # long profits when price rises

    side_dates = dates[side_mask]
    n_side = side_mask.sum()

    if n_side == 0:
        return results

    for pct in CONFIDENCE_PERCENTILES:
        # Top pct% by confidence = above (100-pct) percentile
        threshold = np.percentile(confidence_values, 100 - pct)
        conf_mask = confidence_values >= threshold

        filtered_realized = realized_ticks[conf_mask]
        filtered_dates = side_dates[conf_mask]
        n_events = len(filtered_realized)

        if n_events < 10:
            continue

        avg_realized = filtered_realized.mean()
        net_after_cost = avg_realized - MKT_ORDER_COST_TICKS

        # Win rate: fraction where realized > cost
        win_rate = (filtered_realized > MKT_ORDER_COST_TICKS).mean()

        # Per-trade net P&L for Sharpe
        net_pnl_per_trade = filtered_realized - MKT_ORDER_COST_TICKS
        sharpe = net_pnl_per_trade.mean() / net_pnl_per_trade.std() if net_pnl_per_trade.std() > 0 else 0
        # Annualize: assume ~80k events/day, scale by sqrt
        # Actually report per-trade Sharpe (more meaningful here)

        # Per-day analysis
        unique_dates = np.unique(filtered_dates)
        n_days = len(unique_dates)
        profitable_days = 0
        daily_pnls = []
        for d in unique_dates:
            day_mask = filtered_dates == d
            day_pnl = net_pnl_per_trade[day_mask].sum()
            daily_pnls.append(day_pnl)
            if day_pnl > 0:
                profitable_days += 1

        profitable_days_frac = profitable_days / n_days if n_days > 0 else 0
        daily_pnls = np.array(daily_pnls)
        daily_sharpe = daily_pnls.mean() / daily_pnls.std() if daily_pnls.std() > 0 else 0

        # Avg events per day
        events_per_day = n_events / n_days if n_days > 0 else 0

        # Avg daily $ P&L
        avg_daily_pnl_dollars = daily_pnls.mean() * ES_TICK_VALUE

        # Median realized (more robust than mean for fat tails)
        median_realized = np.median(filtered_realized)

        results.append({
            'side': side,
            'horizon': horizon_name,
            'confidence_pct': pct,
            'threshold': threshold,
            'n_events': n_events,
            'n_days': n_days,
            'events_per_day': events_per_day,
            'avg_realized_ticks': avg_realized,
            'median_realized_ticks': median_realized,
            'net_after_mkt_order': net_after_cost,
            'win_rate': win_rate,
            'per_trade_sharpe': sharpe,
            'daily_sharpe': daily_sharpe,
            'profitable_days_frac': profitable_days_frac,
            'avg_daily_pnl_ticks': daily_pnls.mean(),
            'avg_daily_pnl_dollars': avg_daily_pnl_dollars,
            'std_daily_pnl_ticks': daily_pnls.std(),
        })

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("MARKET ORDER VIABILITY ANALYSIS v1")
    print(f"Cost model: {MKT_ORDER_COST_TICKS:.3f} ticks RT "
          f"(commission {COMMISSION_RT_TICKS:.3f} + spread {SPREAD_CROSSING_RT:.1f})")
    print("=" * 70)
    print()

    preds, labels, dates = load_all_predictions()

    # Basic stats
    print("\n── Label Statistics (realized forward moves in ticks) ──")
    for i, h in enumerate(HORIZONS):
        col = labels[:, i]
        print(f"  {h}: mean={col.mean():.3f}, std={col.std():.3f}, "
              f"skew={pd.Series(col).skew():.3f}, "
              f"p1={np.percentile(col,1):.1f}, p99={np.percentile(col,99):.1f}")

    print("\n── Prediction Statistics ──")
    for i, h in enumerate(HORIZONS):
        col = preds[:, i]
        print(f"  {h}: mean={col.mean():.4f}, std={col.std():.4f}, "
              f"p1={np.percentile(col,1):.3f}, p99={np.percentile(col,99):.3f}")
        # Pred-label correlation (IC)
        valid = ~np.isnan(labels[:, i])
        ic = np.corrcoef(preds[valid, i], labels[valid, i])[0, 1]
        print(f"       IC (pred vs realized): {ic:.4f}")

    # ── Run analysis ──
    print("\n── Sweeping confidence percentiles ──")
    all_results = []

    for i, h in enumerate(HORIZONS):
        for side in ['short', 'long']:
            results = analyze_side(preds[:, i], labels[:, i], dates, side, h)
            all_results.extend(results)

    # ── Create DataFrame ──
    df = pd.DataFrame(all_results)
    df = df.sort_values('net_after_mkt_order', ascending=False)

    # ── Save CSV ──
    csv_path = os.path.join(OUTPUT_DIR, 'market_order_viability.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nCSV saved: {csv_path}")

    # ── Print Summary Table ──
    print("\n" + "=" * 120)
    print("RESULTS: sorted by net ticks after market-order cost")
    print("=" * 120)

    fmt = "{:<6} {:<8} {:>6} {:>8} {:>8} {:>10} {:>12} {:>10} {:>8} {:>12} {:>14} {:>10}"
    header = fmt.format(
        'Side', 'Horizon', 'Top%', 'N_evts', 'N_days', 'Evts/day',
        'AvgReal(tk)', 'MedReal', 'Net(tk)', 'WinRate', 'DailySharpe', 'ProfDays'
    )
    print(header)
    print("-" * 120)

    for _, row in df.iterrows():
        line = fmt.format(
            row['side'],
            row['horizon'],
            f"{row['confidence_pct']}%",
            f"{row['n_events']:,}",
            f"{row['n_days']}",
            f"{row['events_per_day']:.0f}",
            f"{row['avg_realized_ticks']:.3f}",
            f"{row['median_realized_ticks']:.3f}",
            f"{row['net_after_mkt_order']:.3f}",
            f"{row['win_rate']:.1%}",
            f"{row['daily_sharpe']:.3f}",
            f"{row['profitable_days_frac']:.1%}",
        )
        # Highlight profitable configs
        marker = " <<<" if row['net_after_mkt_order'] > 0 else ""
        print(line + marker)

    # ── Generate Report ──
    report_lines = []
    report_lines.append("# Market Order Viability Analysis — CNN-Mamba v2 OOT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"")
    report_lines.append(f"## Setup")
    report_lines.append(f"- Model: CNN-Mamba v2 (all_oot per-date predictions)")
    report_lines.append(f"- OOT dates: {len(np.unique(dates))} days ({dates[0]} to {np.unique(dates)[-1]})")
    report_lines.append(f"- Total events: {len(preds):,}")
    report_lines.append(f"- Market order RT cost: {MKT_ORDER_COST_TICKS:.3f} ticks "
                        f"(${MKT_ORDER_COST_TICKS * ES_TICK_VALUE:.2f})")
    report_lines.append(f"- Horizons analyzed: {', '.join(HORIZONS)}")
    report_lines.append(f"")

    # Key findings
    profitable = df[df['net_after_mkt_order'] > 0]
    report_lines.append(f"## Key Findings")
    report_lines.append(f"")

    if len(profitable) > 0:
        report_lines.append(f"### PROFITABLE configurations ({len(profitable)} found):")
        report_lines.append(f"")
        for _, row in profitable.iterrows():
            report_lines.append(
                f"- **{row['side'].upper()} {row['horizon']} top {row['confidence_pct']}%**: "
                f"net +{row['net_after_mkt_order']:.3f} ticks/trade, "
                f"WR {row['win_rate']:.1%}, "
                f"daily Sharpe {row['daily_sharpe']:.2f}, "
                f"{row['profitable_days_frac']:.0%} profitable days, "
                f"~{row['events_per_day']:.0f} events/day, "
                f"avg daily P&L ${row['avg_daily_pnl_dollars']:.0f}"
            )
        report_lines.append(f"")
    else:
        report_lines.append(f"### NO profitable market-order configurations found.")
        report_lines.append(f"")
        report_lines.append(f"The best configurations are:")
        best5 = df.head(5)
        for _, row in best5.iterrows():
            report_lines.append(
                f"- {row['side'].upper()} {row['horizon']} top {row['confidence_pct']}%: "
                f"net {row['net_after_mkt_order']:.3f} ticks/trade "
                f"(avg realized {row['avg_realized_ticks']:.3f}, need >{MKT_ORDER_COST_TICKS:.3f}), "
                f"WR {row['win_rate']:.1%}, {row['events_per_day']:.0f} events/day"
            )
        report_lines.append(f"")

    # Break-even analysis
    report_lines.append(f"## Break-Even Analysis")
    report_lines.append(f"")
    report_lines.append(f"Market order cost = {MKT_ORDER_COST_TICKS:.3f} ticks. "
                        f"Need avg realized move > {MKT_ORDER_COST_TICKS:.3f} ticks to profit.")
    report_lines.append(f"")

    # Best short and long
    for side in ['short', 'long']:
        side_df = df[df['side'] == side]
        if len(side_df) > 0:
            best = side_df.iloc[0]
            report_lines.append(
                f"Best {side}: {best['horizon']} top {best['confidence_pct']}% — "
                f"avg {best['avg_realized_ticks']:.3f} ticks, "
                f"gap to breakeven: {best['net_after_mkt_order']:.3f} ticks"
            )

    report_lines.append(f"")
    report_lines.append(f"## Implications")
    report_lines.append(f"")
    if len(profitable) > 0:
        report_lines.append(f"Market orders ARE viable at extreme confidence levels.")
        report_lines.append(f"Focus on the profitable configs above for live implementation.")
    else:
        report_lines.append(f"Pure market orders are NOT viable even at extreme confidence.")
        report_lines.append(f"Alternative paths:")
        report_lines.append(f"1. Passive limit orders with adverse-selection-aware placement")
        report_lines.append(f"2. Hybrid: passive entry + market exit (or vice versa)")
        report_lines.append(f"3. Smart execution: RL/MLP to learn optimal order type per signal")
        report_lines.append(f"4. Multi-signal confluence to boost confidence further")

    report_path = os.path.join(OUTPUT_DIR, 'REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nReport saved: {report_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
