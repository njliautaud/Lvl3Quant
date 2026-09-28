#!/usr/bin/env python3
"""
edge_validator.py — HC #492: Automated edge validation
========================================================

Standalone script that validates whether a prediction set has genuine edge.
Run after any model change, retrain, or new fold to verify edge persists.

Tests:
1. Concat correlation > 0.05 minimum
2. Per-fold positive rate > 60%
3. Top 30% meta-filtered gross > 0.5 ticks
4. Top 30% PF > 1.3
5. Adversarial shuffle test (p < 0.01)
6. No single-fold dominance (max fold contribution < 40%)

Usage:
    python3 edge_validator.py <predictions.npz> [--results results.json]
    python3 edge_validator.py output/meta_production_v1/concat_predictions.npz

Returns: 0 if all pass, 1 if any fail, 2 if critical failure.

Author: Claude (HC #492)
Date: 2026-05-24
"""

import json
import math
import sys
from pathlib import Path
import numpy as np


def validate_predictions(pred_path: str, results_path: str = None):
    """Run all validation checks on a prediction set."""
    pred_path = Path(pred_path)

    if not pred_path.exists():
        print(f"CRITICAL: Predictions file not found: {pred_path}")
        return 2

    data = np.load(pred_path, allow_pickle=True)
    preds = data['predictions']
    actuals = data['actuals']

    print(f"Validating: {pred_path}")
    print(f"  N = {len(preds):,}")
    print()

    results = {}
    all_pass = True

    # 1. Concat correlation
    corr = np.corrcoef(preds, actuals)[0, 1]
    passed = corr > 0.05
    results['concat_corr'] = {'value': round(float(corr), 4), 'threshold': 0.05, 'pass': passed}
    print(f"  [{'PASS' if passed else 'FAIL'}] Concat correlation: {corr:.4f} (min 0.05)")
    if not passed:
        all_pass = False

    # 2. Per-fold positive rate (need results.json)
    if results_path:
        rpath = Path(results_path)
        if rpath.exists():
            with open(rpath) as f:
                res = json.load(f)
            per_fold = res.get('per_fold', [])
            if per_fold:
                pos_folds = sum(1 for pf in per_fold if pf.get('corr', 0) > 0)
                total_folds = len(per_fold)
                rate = pos_folds / total_folds
                passed = rate > 0.60
                results['pos_fold_rate'] = {'value': round(rate, 3), 'threshold': 0.60, 'pass': passed,
                                             'detail': f'{pos_folds}/{total_folds} positive'}
                print(f"  [{'PASS' if passed else 'FAIL'}] Positive fold rate: {pos_folds}/{total_folds} = {rate:.1%} (min 60%)")
                if not passed:
                    all_pass = False

    # 3. Top 30% meta-filtered gross
    threshold_30 = np.percentile(preds, 70)
    mask_30 = preds >= threshold_30
    gross_30 = float(actuals[mask_30].mean())
    passed = gross_30 > 0.50
    results['top30_gross'] = {'value': round(gross_30, 4), 'threshold': 0.50, 'pass': passed}
    print(f"  [{'PASS' if passed else 'FAIL'}] Top 30% gross: {gross_30:+.4f} ticks (min +0.50)")
    if not passed:
        all_pass = False

    # 4. Top 30% PF
    top30_pnl = actuals[mask_30]
    wins = float(top30_pnl[top30_pnl > 0].sum())
    losses = float(abs(top30_pnl[top30_pnl < 0].sum()))
    pf = wins / losses if losses > 0 else float('inf')
    passed = pf > 1.3
    results['top30_pf'] = {'value': round(pf, 3), 'threshold': 1.3, 'pass': passed}
    print(f"  [{'PASS' if passed else 'FAIL'}] Top 30% PF: {pf:.3f} (min 1.3)")
    if not passed:
        all_pass = False

    # 5. Adversarial shuffle test
    n_shuffles = 1000
    real_top30_gross = gross_30
    shuffle_grosses = []
    for _ in range(n_shuffles):
        shuffled = np.random.permutation(actuals)
        shuffle_grosses.append(float(shuffled[mask_30].mean()))
    shuffle_arr = np.array(shuffle_grosses)
    p_value = float((shuffle_arr >= real_top30_gross).mean())
    z_score = (real_top30_gross - shuffle_arr.mean()) / (shuffle_arr.std() + 1e-8)
    passed = p_value < 0.01
    results['shuffle_test'] = {'p_value': round(p_value, 4), 'z_score': round(float(z_score), 2),
                                'threshold': 0.01, 'pass': passed}
    print(f"  [{'PASS' if passed else 'FAIL'}] Shuffle test: p={p_value:.4f}, z={z_score:.2f} (need p<0.01)")
    if not passed:
        all_pass = False

    # 6. Single-fold dominance check
    if results_path:
        rpath = Path(results_path)
        if rpath.exists():
            with open(rpath) as f:
                res = json.load(f)
            per_fold = res.get('per_fold', [])
            if per_fold:
                fold_pnls = [pf.get('mean_pnl', 0) * pf.get('n_test', 0) for pf in per_fold]
                total_pnl = sum(abs(p) for p in fold_pnls)
                if total_pnl > 0:
                    max_conc = max(abs(p) for p in fold_pnls) / total_pnl
                    passed = max_conc < 0.40
                    results['fold_concentration'] = {'value': round(max_conc, 3), 'threshold': 0.40, 'pass': passed}
                    print(f"  [{'PASS' if passed else 'FAIL'}] Fold concentration: {max_conc:.3f} (max 0.40)")
                    if not passed:
                        all_pass = False

    # Summary
    print()
    n_pass = sum(1 for r in results.values() if r.get('pass', False))
    n_total = len(results)
    overall = "PASS" if all_pass else "FAIL"
    print(f"  OVERALL: {overall} ({n_pass}/{n_total} checks passed)")

    return 0 if all_pass else 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 edge_validator.py <predictions.npz> [--results results.json]")
        sys.exit(1)

    pred_path = sys.argv[1]
    results_path = None
    if '--results' in sys.argv:
        idx = sys.argv.index('--results')
        if idx + 1 < len(sys.argv):
            results_path = sys.argv[idx + 1]

    # Auto-detect results.json in same directory
    if results_path is None:
        auto = Path(pred_path).parent / "results.json"
        if auto.exists():
            results_path = str(auto)

    exit_code = validate_predictions(pred_path, results_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
