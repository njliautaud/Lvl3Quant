#!/usr/bin/env python3
"""
V4 Multihead Deep Validation
=============================
Validate the top-5% directional edge with:
1. Per-day PnL breakdown (day concentration check)
2. Long vs short split
3. Permutation test (100 shuffles)
4. Regime analysis (green/red/flat days)
5. Net profitability after costs

Author: Claude (autonomous research, 2026-07-02)
"""

import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(
    format="%(asctime)s [V4-DEEP] %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("V4-DEEP")

ROOT = Path("/home/jupiter/Lvl3Quant")
V4_DIR = ROOT / "output" / "v4_multihead_pressure_v1"
SMART_V3_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
COST_PASSIVE = 0.376  # ticks
COST_MARKET = 1.376   # ticks (commission + 1 tick spread)

# Top percentile thresholds to test
PCT_THRESHOLDS = [1, 2, 3, 5, 10, 20, 50, 100]


def main():
    # Load all fold predictions with OOT date tracking
    fold_files = sorted(V4_DIR.glob("fold_*_oot_predictions.npz"))
    log.info(f"Found {len(fold_files)} fold prediction files")

    # Per-fold data with date tracking
    fold_data = []
    for fp in fold_files:
        try:
            data = np.load(str(fp), allow_pickle=True)
            fold_num = int(data['fold'])
            oot_file = str(data['oot_files'][0]) if 'oot_files' in data else 'unknown'
            # Extract date from filename like "20260102_mbo_events"
            date_str = Path(oot_file).stem.split('_')[0]

            preds_dir = data['preds_dir']  # (N, 3) = 1s, 5s, 10s
            labels_dir = data['labels_dir']  # (N, 3)

            fold_data.append({
                'fold': fold_num,
                'date': date_str,
                'preds_dir': preds_dir,
                'labels_dir': labels_dir,
                'n': len(preds_dir),
            })
            log.info(f"  Fold {fold_num}: {len(preds_dir)} preds, OOT={date_str}")

        except Exception as e:
            log.error(f"  {fp.name}: {e}")

    if not fold_data:
        log.error("No data!")
        return

    # Horizons
    horizons = {0: '1s', 1: '5s', 2: '10s'}

    for h_idx, h_name in horizons.items():
        print(f"\n{'#'*80}")
        print(f"# HORIZON: {h_name} (column {h_idx})")
        print(f"{'#'*80}")

        # Concatenate all data for this horizon
        all_preds = []
        all_labels = []
        all_dates = []

        for fd in fold_data:
            p = fd['preds_dir'][:, h_idx]
            l = fd['labels_dir'][:, h_idx]
            valid = ~np.isnan(p) & ~np.isnan(l)
            all_preds.append(p[valid])
            all_labels.append(l[valid])
            all_dates.extend([fd['date']] * valid.sum())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        dates = np.array(all_dates)
        N = len(preds)

        abs_p = np.abs(preds)
        dirs = np.sign(preds)

        # ================================================================
        # 1. OVERALL EDGE BY PERCENTILE THRESHOLD
        # ================================================================
        print(f"\n{'='*70}")
        print(f"1. EDGE BY PERCENTILE THRESHOLD ({h_name})")
        print(f"{'='*70}")
        print(f"{'Pct':>5} {'N':>8} {'Gross':>8} {'NetPass':>8} {'NetMkt':>8} "
              f"{'WR':>6} {'L%':>5} {'S%':>5}")
        print("-" * 62)

        for pct in PCT_THRESHOLDS:
            if pct == 100:
                mask = np.ones(N, dtype=bool)
            else:
                cutoff = np.percentile(abs_p, 100 - pct)
                mask = abs_p >= cutoff

            d = dirs[mask]
            pnl = d * labels[mask]
            gross = np.mean(pnl)
            net_p = gross - COST_PASSIVE
            net_m = gross - COST_MARKET
            wr = np.mean(pnl > 0)
            long_pct = np.mean(d > 0)

            prof = '✓' if net_p > 0 else ''
            print(f"  {pct:>3}% {mask.sum():>7} {gross:>+7.3f}t {net_p:>+7.3f}t "
                  f"{net_m:>+7.3f}t {wr:>5.1%} {long_pct:>4.0%} {1-long_pct:>4.0%} {prof}")

        # ================================================================
        # 2. PER-DAY BREAKDOWN (top 5%)
        # ================================================================
        print(f"\n{'='*70}")
        print(f"2. PER-DAY BREAKDOWN — TOP 5% ({h_name})")
        print(f"{'='*70}")

        cutoff_5 = np.percentile(abs_p, 95)
        top5_mask = abs_p >= cutoff_5
        top5_dirs = dirs[top5_mask]
        top5_labels = labels[top5_mask]
        top5_dates = dates[top5_mask]
        top5_pnl = top5_dirs * top5_labels

        unique_dates = sorted(set(top5_dates))
        day_pnls = {}
        print(f"{'Date':>12} {'N':>5} {'Gross':>8} {'Net':>8} {'WR':>6} {'L':>3} {'S':>3}")
        print("-" * 55)

        for d in unique_dates:
            dmask = top5_dates == d
            dpnl = top5_pnl[dmask]
            dgross = np.sum(dpnl)
            dnet = dgross - COST_PASSIVE * dmask.sum()
            dwr = np.mean(dpnl > 0)
            dlong = np.sum(top5_dirs[dmask] > 0)
            dshort = np.sum(top5_dirs[dmask] < 0)
            day_pnls[d] = dgross
            print(f"  {d:>10} {dmask.sum():>4} {dgross:>+7.1f}t {dnet:>+7.1f}t "
                  f"{dwr:>5.1%} {dlong:>3} {dshort:>3}")

        total_gross = np.sum(top5_pnl)
        total_net = total_gross - COST_PASSIVE * len(top5_pnl)
        print(f"  {'TOTAL':>10} {len(top5_pnl):>4} {total_gross:>+7.1f}t {total_net:>+7.1f}t "
              f"{np.mean(top5_pnl > 0):>5.1%}")

        # Day concentration
        sorted_days = sorted(day_pnls.items(), key=lambda x: x[1], reverse=True)
        if total_gross > 0 and len(sorted_days) >= 2:
            top2_gross = sorted_days[0][1] + sorted_days[1][1]
            conc = top2_gross / total_gross
            print(f"\n  Day concentration: top-2 days = {top2_gross:+.1f}t = {conc:.0%} of total")
            print(f"  Top day: {sorted_days[0][0]} ({sorted_days[0][1]:+.1f}t)")
            if conc > 0.70:
                print(f"  ⚠ FAILS day-conc cap (0.70)")
            else:
                print(f"  ✓ Passes day-conc cap (0.70)")

        # ================================================================
        # 3. LONG vs SHORT SPLIT (top 5%)
        # ================================================================
        print(f"\n{'='*70}")
        print(f"3. LONG vs SHORT SPLIT — TOP 5% ({h_name})")
        print(f"{'='*70}")

        long_mask_5 = top5_dirs > 0
        short_mask_5 = top5_dirs < 0

        for name, mask in [("LONG", long_mask_5), ("SHORT", short_mask_5)]:
            if mask.sum() == 0:
                print(f"  {name}: 0 trades")
                continue
            pnl_side = top5_pnl[mask]
            gross = np.mean(pnl_side)
            net = gross - COST_PASSIVE
            wr = np.mean(pnl_side > 0)
            print(f"  {name}: n={mask.sum()}, gross={gross:+.3f}t, net={net:+.3f}t, WR={wr:.1%}")

        # ================================================================
        # 4. PERMUTATION TEST (top 5%)
        # ================================================================
        print(f"\n{'='*70}")
        print(f"4. PERMUTATION TEST — TOP 5% ({h_name})")
        print(f"{'='*70}")

        real_gross = np.mean(top5_pnl)
        n_perms = 200
        perm_grosses = []

        rng = np.random.RandomState(42)
        for i in range(n_perms):
            shuffled = rng.permutation(top5_labels)
            perm_pnl = top5_dirs * shuffled
            perm_grosses.append(np.mean(perm_pnl))

        perm_grosses = np.array(perm_grosses)
        p_value = np.mean(perm_grosses >= real_gross)
        bonf_threshold = 0.05 / len(PCT_THRESHOLDS)

        print(f"  Real gross: {real_gross:+.4f} ticks/trade")
        print(f"  Perm mean:  {np.mean(perm_grosses):+.4f}")
        print(f"  Perm std:   {np.std(perm_grosses):.4f}")
        print(f"  p-value:    {p_value:.4f} ({n_perms} permutations)")
        print(f"  Bonferroni: {bonf_threshold:.4f} (8 tests)")
        if p_value < bonf_threshold:
            print(f"  ✓ PASSES Bonferroni correction")
        elif p_value < 0.05:
            print(f"  ~ Passes nominal 5%, FAILS Bonferroni")
        else:
            print(f"  ✗ FAILS even nominal 5%")

        # ================================================================
        # 5. REGIME ANALYSIS
        # ================================================================
        print(f"\n{'='*70}")
        print(f"5. REGIME ANALYSIS — TOP 5% ({h_name})")
        print(f"{'='*70}")

        # Classify days by net label direction (proxy for ES daily return)
        # Use mean of ALL labels that day as a regime indicator
        day_regime = {}
        for d in unique_dates:
            day_mask_all = dates == d
            day_labels = labels[day_mask_all]
            daily_mean = np.mean(day_labels[~np.isnan(day_labels)])
            if daily_mean > 0.5:
                day_regime[d] = 'GREEN'
            elif daily_mean < -0.5:
                day_regime[d] = 'RED'
            else:
                day_regime[d] = 'FLAT'

        for regime in ['GREEN', 'RED', 'FLAT']:
            regime_dates = [d for d in unique_dates if day_regime.get(d) == regime]
            if not regime_dates:
                continue
            regime_mask = np.isin(top5_dates, regime_dates)
            if regime_mask.sum() == 0:
                continue
            rpnl = top5_pnl[regime_mask]
            rgross = np.mean(rpnl)
            rnet = rgross - COST_PASSIVE
            rwr = np.mean(rpnl > 0)
            print(f"  {regime:>5}: {len(regime_dates)} days, {regime_mask.sum()} trades, "
                  f"gross={rgross:+.3f}t, net={rnet:+.3f}t, WR={rwr:.1%}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'#'*80}")
    print("FINAL SUMMARY")
    print(f"{'#'*80}")

    for h_idx, h_name in horizons.items():
        all_p = np.concatenate([fd['preds_dir'][:, h_idx] for fd in fold_data])
        all_l = np.concatenate([fd['labels_dir'][:, h_idx] for fd in fold_data])
        valid = ~np.isnan(all_p) & ~np.isnan(all_l)
        p = all_p[valid]
        l = all_l[valid]
        abs_p = np.abs(p)

        for pct in [5, 10]:
            cutoff = np.percentile(abs_p, 100 - pct)
            mask = abs_p >= cutoff
            d = np.sign(p[mask])
            pnl = d * l[mask]
            gross = np.mean(pnl)
            net_p = gross - COST_PASSIVE
            net_m = gross - COST_MARKET
            wr = np.mean(pnl > 0)
            ic = np.corrcoef(p[mask], l[mask])[0, 1]

            print(f"  {h_name} top-{pct}%: n={mask.sum()}, gross={gross:+.3f}t, "
                  f"net_passive={net_p:+.3f}t, net_market={net_m:+.3f}t, "
                  f"WR={wr:.1%}, IC={ic:.3f}")

    # Save results
    output_path = ROOT / "output" / "v4_deep_validation.json"
    results = {
        'n_folds': len(fold_data),
        'n_total': sum(fd['n'] for fd in fold_data),
        'n_dates': len(set(d['date'] for d in fold_data)),
        'date_range': f"{fold_data[0]['date']}-{fold_data[-1]['date']}",
    }
    with open(str(output_path), 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
