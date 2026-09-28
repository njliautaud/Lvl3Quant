#!/usr/bin/env python3
"""
Paper Expiry Scorer — Compares Actual vs Predicted
=====================================================

Run this after Friday expiry to score how paper engine outcomes
compare to the baseline predictions from paper_vs_backtest_baseline.py.

Reads:
  - Baseline predictions (output/paper_validation/week_YYYY_MM_DD_baseline.json)
  - Current paper engine state (trades.jsonl for completed trades)
  - Equity curve (equity.csv for NAV trajectory)

Outputs:
  - Actual vs predicted comparison
  - BS pricing accuracy assessment
  - Recommendations for model calibration

Author: Claude (2026-07-06)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/paper_validation')

def load_baseline(week_date='2026_07_06'):
    path = OUTPUT_DIR / f'week_{week_date}_baseline.json'
    if not path.exists():
        print(f"No baseline found at {path}")
        return None
    with open(path) as f:
        return json.load(f)

def load_trades(state_dir):
    """Load completed trades from trades.jsonl."""
    trades_path = Path(state_dir) / 'trades.jsonl'
    if not trades_path.exists():
        return []
    trades = []
    with open(trades_path) as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))
    return trades

def load_equity(state_dir):
    """Load equity curve."""
    eq_path = Path(state_dir) / 'equity.csv'
    if not eq_path.exists():
        return None
    df = pd.read_csv(eq_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp')

def score_engine(name, state_dir, baseline_positions, strategy_type):
    """Score an engine's actual results vs predictions."""
    print(f"\n{'='*50}")
    print(f"  {name} — ACTUAL vs PREDICTED")
    print(f"{'='*50}")

    trades = load_trades(state_dir)
    equity = load_equity(state_dir)

    if equity is None or len(equity) < 2:
        print("  Insufficient data")
        return None

    # Current state
    state_path = Path(state_dir) / 'state.json'
    with open(state_path) as f:
        state = json.load(f)

    active = state.get('spreads', state.get('positions', []))
    cash = state.get('cash', 0)

    # NAV trajectory this week
    week_start = equity['nav'].iloc[0]
    current_nav = equity['nav'].iloc[-1]
    peak_nav = equity['nav'].max()
    actual_return_pct = (current_nav - 100000) / 100000 * 100

    print(f"  NAV: ${week_start:,.0f} → ${current_nav:,.0f} ({actual_return_pct:+.2f}%)")
    print(f"  Active positions: {len(active)}")
    print(f"  Completed trades: {len(trades)}")

    # Compare to baseline prediction
    if baseline_positions:
        predicted_pnl = sum(p['mc_prediction']['mean_pnl'] for p in baseline_positions)
        actual_pnl = current_nav - 100000

        print(f"\n  Predicted total PnL: ${predicted_pnl:,.0f}")
        print(f"  Actual PnL so far:   ${actual_pnl:,.0f}")

        if predicted_pnl != 0:
            accuracy = actual_pnl / predicted_pnl * 100
            print(f"  Prediction accuracy: {accuracy:.0f}%")

        # Per-position comparison (if trades completed)
        if trades:
            print(f"\n  Per-position results:")
            for t in trades:
                ticker = t.get('ticker', '?')
                pnl = t.get('pnl', t.get('profit', 0))
                reason = t.get('exit_reason', t.get('reason', '?'))

                # Find matching baseline prediction
                pred = next((p for p in baseline_positions if p['ticker'] == ticker), None)
                if pred:
                    pred_pnl = pred['mc_prediction']['mean_pnl']
                    pred_wr = pred['mc_prediction']['win_rate']
                    print(f"    {ticker}: actual ${pnl:,.0f} vs predicted ${pred_pnl:,.0f} "
                          f"({reason}, pred WR {pred_wr:.0%})")
                else:
                    print(f"    {ticker}: actual ${pnl:,.0f} ({reason}) — no baseline prediction")

    # BS pricing quality indicators
    print(f"\n  BS Pricing Quality Indicators:")
    if len(active) > 0:
        # Check if positions are behaving as BS would predict
        # (theta decay should be positive for premium sellers)
        if len(equity) > 10:
            recent = equity.tail(10)
            nav_trend = recent['nav'].iloc[-1] - recent['nav'].iloc[0]
            print(f"    Recent NAV trend (last 10 snapshots): ${nav_trend:+,.0f}")
            print(f"    (Positive = theta decay working as expected)")

    return {
        'name': name,
        'actual_nav': current_nav,
        'actual_return_pct': actual_return_pct,
        'active_positions': len(active),
        'completed_trades': len(trades),
    }


def main():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M ET')
    print(f"\n{'#'*60}")
    print(f"  PAPER ENGINE EXPIRY SCORER — {timestamp}")
    print(f"{'#'*60}")

    baseline = load_baseline()

    results = {}

    # IC engine
    ic_baseline = baseline.get('ic_positions', []) if baseline else []
    r = score_engine('Iron Condor',
                     '/home/jupiter/Lvl3Quant/live_trading_linux/wheel_ic_state',
                     ic_baseline, 'ic')
    if r: results['ic'] = r

    # BPS engine
    bps_baseline = baseline.get('bps_positions', []) if baseline else []
    r = score_engine('Bull Put Spread',
                     '/home/jupiter/Lvl3Quant/live_trading_linux/wheel_bps_state',
                     bps_baseline, 'bps')
    if r: results['bps'] = r

    # V4 and V5 (no baseline predictions, just track performance)
    for name, path in [('V4 CSP', 'wheel_v4_state'), ('V5 CSP', 'wheel_v5_state')]:
        r = score_engine(name,
                        f'/home/jupiter/Lvl3Quant/live_trading_linux/{path}',
                        [], 'csp')
        if r: results[name.lower().replace(' ', '_')] = r

    # Summary
    print(f"\n{'='*50}")
    print(f"  WEEKLY SUMMARY")
    print(f"{'='*50}")

    for k, v in results.items():
        print(f"  {v['name']}: NAV ${v['actual_nav']:,.0f} ({v['actual_return_pct']:+.2f}%)")

    # BS pricing verdict
    if baseline:
        ic_pred = baseline.get('ic_portfolio_prediction', {})
        bps_pred = baseline.get('bps_portfolio_prediction', {})

        print(f"\n  BS Pricing Verdict:")
        if 'ic' in results and ic_pred:
            actual = results['ic']['actual_return_pct']
            predicted = ic_pred.get('predicted_yield', 0) / 100 * ic_pred.get('total_premium', 0) / 1000
            print(f"    IC: Predicted ${ic_pred.get('predicted_mean_pnl',0):,.0f} yield")
            print(f"    IC: Actual so far ${results['ic']['actual_nav'] - 100000:,.0f}")

        if 'bps' in results and bps_pred:
            print(f"    BPS: Predicted ${bps_pred.get('predicted_mean_pnl',0):,.0f} yield")
            print(f"    BPS: Actual so far ${results['bps']['actual_nav'] - 100000:,.0f}")

    # Save
    out = {
        'scored_at': timestamp,
        'results': results,
        'baseline_file': 'week_2026_07_06_baseline.json',
    }
    out_path = OUTPUT_DIR / f'week_2026_07_06_scored.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == '__main__':
    main()
