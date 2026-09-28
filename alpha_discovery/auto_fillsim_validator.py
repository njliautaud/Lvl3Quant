#!/usr/bin/env python3
"""
Auto Fill Sim Validator - Runs immediately when LGBM predictions saved
Validates profitability and makes autonomous deploy decision
"""

import numpy as np
import sys
import time
from pathlib import Path
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)

PRED_DIR = Path("/home/jupiter/Lvl3Quant/alpha_discovery/results/lgbm_confidence_ic")
DEPLOY_THRESHOLD_USD = 100  # $100/day minimum for deploy

def wait_for_predictions():
    """Wait for fold00_labels_10s.npz to be created"""
    target = PRED_DIR / "fold00_labels_10s.npz"
    print(f"Waiting for predictions: {target}")

    while not target.exists():
        time.sleep(10)

    # Wait extra 5 sec to ensure file is fully written
    time.sleep(5)
    print(f"✅ Predictions found: {target}")
    return target

def analyze_predictions(pred_file):
    """Analyze direction accuracy and filter for top signals"""
    data = np.load(pred_file)
    preds = data['preds']
    labels = data['labels']

    print(f"\n{'='*70}")
    print("PREDICTION ANALYSIS (10s labels)")
    print(f"{'='*70}")

    abs_p = np.abs(preds)
    pred_sign = np.sign(preds)
    label_sign = np.sign(labels)
    correct = (pred_sign == label_sign)

    # Overall stats
    overall_ic = spearmanr(preds, labels)[0]
    overall_dir = correct.mean()

    print(f"Overall IC: {overall_ic:+.4f}")
    print(f"Overall DirAcc: {overall_dir:.4f} ({overall_dir*100:.1f}%)")
    print()

    # Top percentile analysis
    results = {}
    for pct in [50, 25, 10, 5, 1]:
        thresh = np.percentile(abs_p, 100 - pct)
        mask = abs_p >= thresh

        dir_acc = correct[mask].mean()
        ic = spearmanr(preds[mask], labels[mask])[0]
        n = mask.sum()

        results[f"top{pct}"] = {
            'dir_acc': dir_acc,
            'ic': ic,
            'count': n,
            'threshold': thresh
        }

        print(f"Top {pct:2d}%: DirAcc={dir_acc:.4f} ({dir_acc*100:.1f}%), IC={ic:+.4f}, N={n:,}")

    return results

def estimate_pnl(results):
    """Estimate P&L based on direction accuracy

    Simplified model:
    - Avg move: 1.5 ticks when right
    - Avg loss: 1.0 tick when wrong
    - Cost: 1.2 ticks RT (bid-ask + fees)
    - Trade only top 5% signals
    - Assume 100 trades/day
    """

    top5_dir = results['top5']['dir_acc']
    trades_per_day = 100

    win_ticks = 1.5
    loss_ticks = 1.0
    cost_ticks = 1.2

    wins = trades_per_day * top5_dir
    losses = trades_per_day * (1 - top5_dir)

    gross_pnl_ticks = (wins * win_ticks) - (losses * loss_ticks)
    costs_ticks = trades_per_day * cost_ticks
    net_pnl_ticks = gross_pnl_ticks - costs_ticks

    # $12.50 per tick for ES
    net_pnl_usd = net_pnl_ticks * 12.5

    print(f"\n{'='*70}")
    print("ESTIMATED DAILY P&L (Top 5% signals, 100 trades/day)")
    print(f"{'='*70}")
    print(f"Win rate: {top5_dir*100:.1f}%")
    print(f"Expected wins: {wins:.1f} trades × {win_ticks} ticks = {wins*win_ticks:.1f} ticks")
    print(f"Expected losses: {losses:.1f} trades × {loss_ticks} ticks = {losses*loss_ticks:.1f} ticks")
    print(f"Gross P&L: {gross_pnl_ticks:.1f} ticks")
    print(f"Costs: {costs_ticks:.1f} ticks")
    print(f"Net P&L: {net_pnl_ticks:.1f} ticks = ${net_pnl_usd:.2f}/day")
    print(f"{'='*70}")

    return net_pnl_usd

def make_deploy_decision(pnl_usd):
    """Autonomous deploy decision"""
    print(f"\n{'='*70}")
    print("DEPLOY DECISION (AUTONOMOUS)")
    print(f"{'='*70}")
    print(f"Threshold: ${DEPLOY_THRESHOLD_USD}/day")
    print(f"Estimated P&L: ${pnl_usd:.2f}/day")
    print()

    if pnl_usd >= DEPLOY_THRESHOLD_USD:
        print(f"✅ DEPLOY MONDAY")
        print(f"P&L ${pnl_usd:.2f} > threshold ${DEPLOY_THRESHOLD_USD}")
        print(f"Model passes profitability test.")
        return True
    else:
        print(f"❌ NO DEPLOY")
        print(f"P&L ${pnl_usd:.2f} < threshold ${DEPLOY_THRESHOLD_USD}")
        print(f"Continue research mode. Deploy simple OFI baseline instead.")
        return False

def main():
    print("🤖 AUTO FILL SIM VALIDATOR - Autonomous Mode")
    print(f"Deploy threshold: ${DEPLOY_THRESHOLD_USD}/day\n")

    # Wait for predictions
    pred_file = wait_for_predictions()

    # Analyze
    results = analyze_predictions(pred_file)

    # Estimate P&L
    pnl_usd = estimate_pnl(results)

    # Make decision
    deploy = make_deploy_decision(pnl_usd)

    # Save decision
    decision_file = PRED_DIR / "deploy_decision.json"
    import json
    json.dump({
        'deploy': deploy,
        'pnl_usd_per_day': pnl_usd,
        'threshold': DEPLOY_THRESHOLD_USD,
        'timestamp': time.time()
    }, open(decision_file, 'w'), indent=2)

    print(f"\n💾 Decision saved: {decision_file}")
    return deploy

if __name__ == "__main__":
    deploy = main()
    sys.exit(0 if deploy else 1)
