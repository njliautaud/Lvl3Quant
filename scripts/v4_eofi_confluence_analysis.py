#!/usr/bin/env python3
"""
V4 Multihead — EOFI Confluence Analysis
========================================
Tests HC #514 hypothesis: does conditioning entries on EOFI (order flow
imbalance) agreement with the directional head improve trade quality?

The v4 multihead IC analysis showed:
  - Directional head: IC_1s=0.211 (strong)
  - EOFI head: IC_1s=0.061 (weak but positive in 82% of folds)
  - NTPS/PDI/TIA: dead

Question: when EOFI AGREES with direction, does the directional signal
have higher effective IC? Does filtering to EOFI-aligned predictions
concentrate alpha?

Uses v4 OOT fold predictions (126-136) with actual MBO event data for
ground truth price paths.
"""

import json
import logging
import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("v4_eofi_confluence")

BASE = Path("/home/jupiter/Lvl3Quant")
V4_DIR = BASE / "output" / "v4_multihead_pressure_v1"
EVENTS_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3"
OUTPUT_DIR = BASE / "output" / "v4_eofi_confluence_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

COMMISSION_RT_TICKS = 0.376
TICK_SIZE = 0.25

def load_fold(fold_num):
    """Load v4 predictions and match to MBO events for ground truth."""
    pred_file = V4_DIR / f"fold_{fold_num}_oot_predictions.npz"
    if not pred_file.exists():
        return None

    d = np.load(str(pred_file), allow_pickle=True)

    # Get predictions
    preds_dir = d['preds_dir']     # (N, 3) = [1s, 5s, 10s]
    preds_eofi = d['preds_eofi']   # (N, 4) = [1s, 5s, 10s, 30s]
    labels_dir = d['labels_dir']   # (N, 3) — may have NaN

    # Get OOT date from filename
    oot_files = d.get('oot_files', np.array([]))
    if len(oot_files) > 0:
        oot_file = str(oot_files[0])
        # Extract date from path like '.../20260115_mbo_events.npz'
        date_str = oot_file.split('/')[-1].split('_')[0]
        if '\\' in oot_file:
            date_str = oot_file.split('\\')[-1].split('_')[0]
    else:
        return None

    # Load MBO events for ground truth
    event_file = EVENTS_DIR / f"{date_str}_mbo_events.npz"
    if not event_file.exists():
        log.warning(f"No events file for {date_str}")
        return None

    events = np.load(str(event_file), allow_pickle=True)

    return {
        'fold': fold_num,
        'date': date_str,
        'preds_dir': preds_dir,
        'preds_eofi': preds_eofi,
        'labels_dir': labels_dir,
        'events': events,
    }


def analyze_confluence(fold_data, horizons=None):
    """Analyze whether EOFI confluence improves directional signal quality."""
    if horizons is None:
        horizons = [(0, "1s"), (1, "5s"), (2, "10s")]

    preds_dir = fold_data['preds_dir']
    preds_eofi = fold_data['preds_eofi']
    labels_dir = fold_data['labels_dir']

    results = {}

    for h_idx, h_name in horizons:
        dir_preds = preds_dir[:, h_idx]

        # EOFI prediction at same horizon (EOFI has 4 horizons: 1s,5s,10s,30s)
        eofi_idx = min(h_idx, preds_eofi.shape[1] - 1)
        eofi_preds = preds_eofi[:, eofi_idx]

        labels = labels_dir[:, h_idx]
        valid = ~np.isnan(labels) & ~np.isnan(dir_preds) & ~np.isnan(eofi_preds)

        if valid.sum() < 100:
            continue

        dir_v = dir_preds[valid]
        eofi_v = eofi_preds[valid]
        lab_v = labels[valid]

        # Baseline IC (all predictions)
        ic_all, _ = spearmanr(dir_v, lab_v)

        # EOFI agreement: both predict same sign
        agree_mask = (np.sign(dir_v) == np.sign(eofi_v)) & (np.abs(dir_v) > 0.01) & (np.abs(eofi_v) > 0.001)
        disagree_mask = (np.sign(dir_v) != np.sign(eofi_v)) & (np.abs(dir_v) > 0.01) & (np.abs(eofi_v) > 0.001)

        # IC when EOFI agrees
        if agree_mask.sum() > 50:
            ic_agree, _ = spearmanr(dir_v[agree_mask], lab_v[agree_mask])
        else:
            ic_agree = float('nan')

        # IC when EOFI disagrees
        if disagree_mask.sum() > 50:
            ic_disagree, _ = spearmanr(dir_v[disagree_mask], lab_v[disagree_mask])
        else:
            ic_disagree = float('nan')

        # Mean absolute realized move when agree vs disagree
        mean_move_agree = np.mean(np.abs(lab_v[agree_mask])) if agree_mask.sum() > 0 else 0
        mean_move_disagree = np.mean(np.abs(lab_v[disagree_mask])) if disagree_mask.sum() > 0 else 0

        # Directional accuracy (does the prediction sign match the label sign?)
        dir_correct_all = np.mean(np.sign(dir_v) == np.sign(lab_v))
        dir_correct_agree = np.mean(np.sign(dir_v[agree_mask]) == np.sign(lab_v[agree_mask])) if agree_mask.sum() > 50 else float('nan')
        dir_correct_disagree = np.mean(np.sign(dir_v[disagree_mask]) == np.sign(lab_v[disagree_mask])) if disagree_mask.sum() > 50 else float('nan')

        # Confidence-stratified: top 10% and top 20% by |dir_pred|
        abs_dir = np.abs(dir_v)
        p90 = np.percentile(abs_dir, 90)
        p80 = np.percentile(abs_dir, 80)

        top10_mask = abs_dir >= p90
        top20_mask = abs_dir >= p80
        top10_agree = top10_mask & agree_mask
        top20_agree = top20_mask & agree_mask

        # Net ticks analysis for top confidence + EOFI agreement
        # Signed realized move in label direction
        def net_ticks_analysis(mask, label_vals, pred_vals):
            if mask.sum() < 20:
                return {}
            sides = np.sign(pred_vals[mask])
            moves = label_vals[mask] / TICK_SIZE  # convert to ticks
            gross = sides * moves
            net = gross - COMMISSION_RT_TICKS
            return {
                'n_trades': int(mask.sum()),
                'mean_gross_ticks': round(float(np.mean(gross)), 4),
                'mean_net_ticks': round(float(np.mean(net)), 4),
                'win_rate': round(float(np.mean(net > 0)), 4),
                'sharpe': round(float(np.mean(net) / (np.std(net) + 1e-8) * np.sqrt(252)), 2),
            }

        results[h_name] = {
            'n_total': int(valid.sum()),
            'n_agree': int(agree_mask.sum()),
            'n_disagree': int(disagree_mask.sum()),
            'pct_agree': round(float(agree_mask.sum()) / valid.sum() * 100, 1),
            'ic_all': round(float(ic_all), 4),
            'ic_agree': round(float(ic_agree), 4) if not np.isnan(ic_agree) else None,
            'ic_disagree': round(float(ic_disagree), 4) if not np.isnan(ic_disagree) else None,
            'ic_lift': round(float(ic_agree - ic_all), 4) if not np.isnan(ic_agree) else None,
            'dir_accuracy_all': round(float(dir_correct_all), 4),
            'dir_accuracy_agree': round(float(dir_correct_agree), 4) if not np.isnan(dir_correct_agree) else None,
            'dir_accuracy_disagree': round(float(dir_correct_disagree), 4) if not np.isnan(dir_correct_disagree) else None,
            'mean_move_agree_ticks': round(float(mean_move_agree / TICK_SIZE), 2),
            'mean_move_disagree_ticks': round(float(mean_move_disagree / TICK_SIZE), 2),
            'top10_all': net_ticks_analysis(top10_mask, lab_v, dir_v),
            'top10_agree': net_ticks_analysis(top10_agree, lab_v, dir_v),
            'top20_all': net_ticks_analysis(top20_mask, lab_v, dir_v),
            'top20_agree': net_ticks_analysis(top20_agree, lab_v, dir_v),
        }

    return results


def main():
    log.info("V4 Multihead EOFI Confluence Analysis")
    log.info("=" * 60)

    # Load all available folds
    fold_files = sorted(V4_DIR.glob("fold_*_oot_predictions.npz"))
    fold_nums = [int(f.stem.split("_")[1]) for f in fold_files]

    all_results = []
    combined = {h: {'ic_all': [], 'ic_agree': [], 'ic_disagree': [],
                     'dir_acc_all': [], 'dir_acc_agree': [], 'dir_acc_disagree': [],
                     'ic_lift': [], 'n_agree_pct': []}
                for h in ["1s", "5s", "10s"]}

    for fold_num in fold_nums:
        log.info(f"\n--- Fold {fold_num} ---")
        fold_data = load_fold(fold_num)
        if fold_data is None:
            log.warning(f"Could not load fold {fold_num}")
            continue

        log.info(f"Date: {fold_data['date']}, Predictions: {fold_data['preds_dir'].shape[0]}")

        result = analyze_confluence(fold_data)
        result['fold'] = fold_num
        result['date'] = fold_data['date']
        all_results.append(result)

        for h in ["1s", "5s", "10s"]:
            if h in result:
                r = result[h]
                combined[h]['ic_all'].append(r['ic_all'])
                if r['ic_agree'] is not None:
                    combined[h]['ic_agree'].append(r['ic_agree'])
                if r['ic_disagree'] is not None:
                    combined[h]['ic_disagree'].append(r['ic_disagree'])
                combined[h]['dir_acc_all'].append(r['dir_accuracy_all'])
                if r['dir_accuracy_agree'] is not None:
                    combined[h]['dir_acc_agree'].append(r['dir_accuracy_agree'])
                if r['dir_accuracy_disagree'] is not None:
                    combined[h]['dir_acc_disagree'].append(r['dir_accuracy_disagree'])
                if r['ic_lift'] is not None:
                    combined[h]['ic_lift'].append(r['ic_lift'])
                combined[h]['n_agree_pct'].append(r['pct_agree'])

                log.info(f"  {h}: IC_all={r['ic_all']:.4f} IC_agree={r.get('ic_agree','N/A')} "
                        f"IC_disagree={r.get('ic_disagree','N/A')} "
                        f"DirAcc: all={r['dir_accuracy_all']:.3f} agree={r.get('dir_accuracy_agree','N/A')} "
                        f"disagree={r.get('dir_accuracy_disagree','N/A')}")

    # Summary
    print("\n" + "=" * 100)
    print("COMBINED RESULTS ACROSS ALL FOLDS")
    print("=" * 100)

    summary = {}
    for h in ["1s", "5s", "10s"]:
        c = combined[h]
        s = {
            'mean_ic_all': round(float(np.mean(c['ic_all'])), 4),
            'mean_ic_agree': round(float(np.mean(c['ic_agree'])), 4) if c['ic_agree'] else None,
            'mean_ic_disagree': round(float(np.mean(c['ic_disagree'])), 4) if c['ic_disagree'] else None,
            'mean_ic_lift': round(float(np.mean(c['ic_lift'])), 4) if c['ic_lift'] else None,
            'mean_dir_acc_all': round(float(np.mean(c['dir_acc_all'])), 4),
            'mean_dir_acc_agree': round(float(np.mean(c['dir_acc_agree'])), 4) if c['dir_acc_agree'] else None,
            'mean_dir_acc_disagree': round(float(np.mean(c['dir_acc_disagree'])), 4) if c['dir_acc_disagree'] else None,
            'mean_pct_agree': round(float(np.mean(c['n_agree_pct'])), 1),
        }
        summary[h] = s

        print(f"\n--- {h} Horizon ---")
        print(f"  IC (all predictions):      {s['mean_ic_all']:.4f}")
        print(f"  IC (EOFI agrees):          {s['mean_ic_agree']}")
        print(f"  IC (EOFI disagrees):       {s['mean_ic_disagree']}")
        print(f"  IC LIFT from confluence:   {s['mean_ic_lift']}")
        print(f"  Dir accuracy (all):        {s['mean_dir_acc_all']:.1%}")
        print(f"  Dir accuracy (EOFI agree): {s['mean_dir_acc_agree']}")
        print(f"  Dir accuracy (disagree):   {s['mean_dir_acc_disagree']}")
        print(f"  % predictions with agree:  {s['mean_pct_agree']:.1f}%")

    # Verdict
    print("\n" + "=" * 100)
    best_h = max(summary.keys(), key=lambda h: (summary[h].get('mean_ic_lift') or 0))
    best_lift = summary[best_h].get('mean_ic_lift', 0) or 0

    if best_lift > 0.02:
        verdict = f"EOFI CONFLUENCE ADDS VALUE: +{best_lift:.4f} IC lift at {best_h}. Worth pursuing as trade filter."
    elif best_lift > 0:
        verdict = f"EOFI CONFLUENCE MARGINAL: +{best_lift:.4f} IC lift at {best_h}. Weak but positive — test in sim."
    else:
        verdict = f"EOFI CONFLUENCE DEAD: No IC lift. EOFI head doesn't help filter directional trades."

    print(f"VERDICT: {verdict}")
    print("=" * 100)

    # Save
    output = {
        'summary': summary,
        'verdict': verdict,
        'per_fold': all_results,
    }
    out_file = OUTPUT_DIR / "eofi_confluence_results.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
