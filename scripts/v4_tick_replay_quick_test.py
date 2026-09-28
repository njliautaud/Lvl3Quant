#!/usr/bin/env python3
"""
V4 Multihead — Quick Tick-Level Validation
Tests new v4 OOT predictions against MBO tick-level data.
Runs permutation test + confidence filtering.

Usage:
    python3 v4_tick_replay_quick_test.py           # test all available folds
    python3 v4_tick_replay_quick_test.py --fold 140  # test specific fold
"""
import argparse
import json
import logging
import numpy as np
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [V4-TEST] %(message)s')
log = logging.getLogger(__name__)

ROOT = Path("/home/jupiter/Lvl3Quant")
V4_PRED_DIR = ROOT / "output" / "v4_multihead_pressure_v1"
TICK_REPLAY_DIR = ROOT / "output" / "tick_level_replay"
TICK_REPLAY_DIR.mkdir(exist_ok=True)

COST_PASSIVE = 0.376  # ticks RT
N_PERMS = 100


def test_fold(fold_path: Path, n_perms: int = N_PERMS) -> dict:
    """Test a single fold's predictions using label-based PnL + permutation test."""
    d = np.load(str(fold_path), allow_pickle=True)
    fold = int(d['fold'])
    date_str = str(d['oot_files'][0]).split('/')[-1].split('_')[0]

    pred_1s = d['preds_dir'][:, 0]
    label_1s = d['labels_dir'][:, 0]

    valid = np.isfinite(pred_1s) & np.isfinite(label_1s) & (label_1s != 0)
    n_valid = valid.sum()

    if n_valid < 50:
        return {'fold': fold, 'date': date_str, 'status': 'SKIP', 'reason': f'only {n_valid} valid'}

    p = pred_1s[valid]
    t = label_1s[valid]
    ic = float(np.corrcoef(p, t)[0, 1])

    results = {
        'fold': fold, 'date': date_str, 'ic': ic, 'n_total': int(n_valid),
        'status': 'OK',
    }

    # Test at different confidence levels
    for pct_name, pct in [('all', 0), ('top50', 50), ('top20', 80), ('top10', 90)]:
        abs_p = np.abs(p)
        if pct > 0:
            thresh = np.percentile(abs_p, pct)
            mask = abs_p >= thresh
        else:
            mask = np.ones(len(p), dtype=bool)

        direction = np.sign(p[mask])
        pnl = direction * t[mask]
        net = pnl - COST_PASSIVE

        n = int(mask.sum())
        total_net = float(net.sum())
        avg_net = float(net.mean())
        wr = float((net > 0).mean())
        gross = float(pnl.mean())

        results[f'{pct_name}_n'] = n
        results[f'{pct_name}_gross'] = round(gross, 4)
        results[f'{pct_name}_net'] = round(avg_net, 4)
        results[f'{pct_name}_total_net'] = round(total_net, 1)
        results[f'{pct_name}_wr'] = round(wr, 3)

    # Permutation test (top 20% confidence)
    abs_p = np.abs(p)
    thresh = np.percentile(abs_p, 80)
    strong = abs_p >= thresh
    real_pnl = (np.sign(p[strong]) * t[strong] - COST_PASSIVE).sum()

    rng = np.random.default_rng(42)
    perm_totals = []
    for _ in range(n_perms):
        p_shuf = p.copy()
        rng.shuffle(p_shuf)
        abs_shuf = np.abs(p_shuf)
        thresh_shuf = np.percentile(abs_shuf, 80)
        strong_shuf = abs_shuf >= thresh_shuf
        perm_pnl = (np.sign(p_shuf[strong_shuf]) * t[strong_shuf] - COST_PASSIVE).sum()
        perm_totals.append(perm_pnl)

    perm_arr = np.array(perm_totals)
    p_value = float((perm_arr >= real_pnl).mean())
    z_score = float((real_pnl - perm_arr.mean()) / perm_arr.std()) if perm_arr.std() > 0 else 0

    results['perm_p'] = round(p_value, 4)
    results['perm_z'] = round(z_score, 1)
    results['perm_n'] = n_perms

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, help='Test specific fold')
    parser.add_argument('--perms', type=int, default=N_PERMS, help='Number of permutations')
    args = parser.parse_args()

    folds = sorted(V4_PRED_DIR.glob("fold_*_oot_predictions.npz"))

    if args.fold:
        folds = [f for f in folds if f'fold_{args.fold:03d}' in f.name or f'fold_{args.fold}' in f.name]

    log.info(f"Testing {len(folds)} folds with {args.perms} permutations each")

    all_results = []
    for f in folds:
        t0 = time.time()
        result = test_fold(f, n_perms=args.perms)
        elapsed = time.time() - t0

        if result['status'] == 'SKIP':
            log.info(f"Fold {result['fold']} ({result['date']}): SKIPPED ({result['reason']})")
            continue

        status = 'NET+' if result.get('top20_net', 0) > 0 else 'NET-'
        log.info(f"Fold {result['fold']} ({result['date']}): IC={result['ic']:.3f}, "
                f"top20={result.get('top20_net', 0):+.4f}t/tr, "
                f"p={result.get('perm_p', 1):.4f}, z={result.get('perm_z', 0):.1f}, "
                f"[{status}] ({elapsed:.0f}s)")

        all_results.append(result)

    # Save results
    output_path = TICK_REPLAY_DIR / "v4_multihead_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"Saved {len(all_results)} results to {output_path}")

    # Summary
    if all_results:
        profitable = [r for r in all_results if r.get('top20_net', 0) > 0]
        total_net = sum(r.get('top20_total_net', 0) for r in all_results)
        total_n = sum(r.get('top20_n', 0) for r in all_results)
        avg_net = total_net / total_n if total_n > 0 else 0

        log.info(f"\n{'='*60}")
        log.info(f"SUMMARY: {len(profitable)}/{len(all_results)} days profitable (top 20%)")
        log.info(f"Total: {total_net:+.0f}t net across {total_n} trades ({avg_net:+.4f}t/tr)")

        # Overall permutation significance
        passing = [r for r in all_results if r.get('perm_p', 1) < 0.05]
        log.info(f"Permutation test: {len(passing)}/{len(all_results)} pass p<0.05")


if __name__ == '__main__':
    main()
