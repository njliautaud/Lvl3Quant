#!/usr/bin/env python3
"""
Risk-Adjusted Validator - Proper evaluation with Sharpe, Sortino, drawdown
Not just P&L - risk-adjusted returns matter
"""

import numpy as np
import sys
from pathlib import Path
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)

PRED_DIR = Path("/home/jupiter/Lvl3Quant/alpha_discovery/results/lgbm_confidence_ic")
DEPLOY_THRESHOLDS = {
    'sharpe': 2.0,      # Minimum Sharpe ratio
    'sortino': 2.5,     # Minimum Sortino ratio
    'max_dd': 0.15,     # Maximum drawdown (15%)
    'win_rate': 0.60,   # Minimum win rate for top signals
}

def analyze_predictions(pred_file):
    """Full risk-adjusted analysis"""
    data = np.load(pred_file)
    preds = data['preds']
    labels = data['labels']

    print(f"\n{'='*70}")
    print(f"RISK-ADJUSTED ANALYSIS: {pred_file.name}")
    print(f"{'='*70}\n")

    abs_p = np.abs(preds)

    # Top 5% signals (what we'd actually trade)
    top5_mask = abs_p >= np.percentile(abs_p, 95)
    top_preds = preds[top5_mask]
    top_labels = labels[top5_mask]

    # Direction accuracy
    correct = np.sign(top_preds) == np.sign(top_labels)
    win_rate = correct.mean()

    # Realized returns per trade (simplified)
    # If prediction correct: get label move - costs
    # If wrong: lose label move + costs
    costs_ticks = 1.2
    returns_ticks = np.where(correct,
                             np.abs(top_labels) * 1e4 - costs_ticks,  # Scale labels to ticks
                             -np.abs(top_labels) * 1e4 - costs_ticks)

    # Filter out NaN/inf
    returns_ticks = returns_ticks[np.isfinite(returns_ticks)]

    if len(returns_ticks) < 100:
        print("ERROR: Not enough valid trades for analysis")
        return None

    # Risk metrics
    mean_return = returns_ticks.mean()
    std_return = returns_ticks.std()
    downside_returns = returns_ticks[returns_ticks < 0]
    downside_std = np.sqrt((downside_returns**2).mean()) if len(downside_returns) > 0 else 1e-9

    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    sortino = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    # Drawdown
    cumulative = np.cumsum(returns_ticks)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-9)
    max_dd = abs(drawdown.min())

    # Profit factor
    wins = returns_ticks[returns_ticks > 0].sum()
    losses = abs(returns_ticks[returns_ticks < 0].sum())
    profit_factor = wins / losses if losses > 0 else float('inf')

    print(f"Top 5% Signals (N={len(top_preds):,}):")
    print(f"  Win Rate:       {win_rate:.1%}")
    print(f"  Mean Return:    {mean_return:.2f} ticks/trade")
    print(f"  Std Dev:        {std_return:.2f} ticks")
    print(f"  Sharpe Ratio:   {sharpe:.2f}")
    print(f"  Sortino Ratio:  {sortino:.2f}")
    print(f"  Max Drawdown:   {max_dd:.1%}")
    print(f"  Profit Factor:  {profit_factor:.2f}")
    print()

    # Daily P&L estimate (100 trades/day, $12.50/tick)
    daily_pnl = mean_return * 100 * 12.5
    print(f"Est. Daily P&L: ${daily_pnl:,.2f} ({mean_return:.2f} ticks × 100 trades × $12.50)")

    return {
        'win_rate': win_rate,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'profit_factor': profit_factor,
        'daily_pnl': daily_pnl,
        'mean_return_ticks': mean_return
    }

def make_deploy_decision(results_1s, results_10s):
    """Autonomous deploy decision based on risk-adjusted metrics"""
    print(f"\n{'='*70}")
    print("DEPLOY DECISION (RISK-ADJUSTED)")
    print(f"{'='*70}\n")

    print("Thresholds:")
    print(f"  Sharpe:     ≥ {DEPLOY_THRESHOLDS['sharpe']}")
    print(f"  Sortino:    ≥ {DEPLOY_THRESHOLDS['sortino']}")
    print(f"  Max DD:     ≤ {DEPLOY_THRESHOLDS['max_dd']:.0%}")
    print(f"  Win Rate:   ≥ {DEPLOY_THRESHOLDS['win_rate']:.0%}")
    print()

    # Check both horizons
    for horizon, results in [('1s', results_1s), ('10s', results_10s)]:
        if results is None:
            continue

        print(f"{horizon} Predictions:")
        print(f"  Sharpe:     {results['sharpe']:.2f} {'✅' if results['sharpe'] >= DEPLOY_THRESHOLDS['sharpe'] else '❌'}")
        print(f"  Sortino:    {results['sortino']:.2f} {'✅' if results['sortino'] >= DEPLOY_THRESHOLDS['sortino'] else '❌'}")
        print(f"  Max DD:     {results['max_dd']:.1%} {'✅' if results['max_dd'] <= DEPLOY_THRESHOLDS['max_dd'] else '❌'}")
        print(f"  Win Rate:   {results['win_rate']:.1%} {'✅' if results['win_rate'] >= DEPLOY_THRESHOLDS['win_rate'] else '❌'}")
        print(f"  Daily P&L:  ${results['daily_pnl']:,.2f}")

        passes = (
            results['sharpe'] >= DEPLOY_THRESHOLDS['sharpe'] and
            results['sortino'] >= DEPLOY_THRESHOLDS['sortino'] and
            results['max_dd'] <= DEPLOY_THRESHOLDS['max_dd'] and
            results['win_rate'] >= DEPLOY_THRESHOLDS['win_rate']
        )

        if passes:
            print(f"\n✅ DEPLOY ON {horizon} PREDICTIONS")
            print(f"All risk-adjusted metrics pass thresholds.")
            return True, horizon, results
        print()

    print("❌ NO DEPLOY")
    print("Neither horizon passes all risk-adjusted thresholds.")
    print("Continue research. Deploy simple baseline Monday.")
    return False, None, None

def main():
    print("🎯 RISK-ADJUSTED VALIDATOR - Autonomous Mode\n")

    # Check both 1s and 10s predictions
    fold00_1s = PRED_DIR / "fold00_labels_1s.npz"
    fold00_10s = PRED_DIR / "fold00_labels_10s.npz"

    results_1s = analyze_predictions(fold00_1s) if fold00_1s.exists() else None
    results_10s = analyze_predictions(fold00_10s) if fold00_10s.exists() else None

    # Make decision
    deploy, horizon, results = make_deploy_decision(results_1s, results_10s)

    # Save decision
    import json
    decision = {
        'deploy': deploy,
        'horizon': horizon,
        'results_1s': results_1s,
        'results_10s': results_10s,
        'timestamp': __import__('time').time()
    }

    decision_file = PRED_DIR / "risk_adjusted_decision.json"
    json.dump(decision, open(decision_file, 'w'), indent=2, default=float)
    print(f"\n💾 Decision saved: {decision_file}")

    return deploy

if __name__ == "__main__":
    deploy = main()
    sys.exit(0 if deploy else 1)
