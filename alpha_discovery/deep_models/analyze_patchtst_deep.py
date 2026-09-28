"""
Deep analysis of PatchTST predictions: IC vs DA disconnect, magnitude/volatility prediction.
"""
import numpy as np
import json
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/patchtst_sliding60d_smart_v2")
MAMBA_PATH = Path("/home/jupiter/Lvl3Quant/output/mamba_v7_tiny_smart_v3_mar_apr/concat_oot_predictions.npz")
SKIP_FOLDS = {4, 6}  # Very few samples
HORIZONS = ['1s', '5s', '10s']
TIERS = {
    'All': 1.0, 'Top50pct': 0.5, 'Top25pct': 0.25,
    'Top10pct': 0.10, 'Top5pct': 0.05, 'Top1pct': 0.01
}

def compute_metrics(preds, labels, tier_frac=1.0):
    """Compute IC, DA, MagCorr for a given tier."""
    if len(preds) < 10:
        return {'ic': None, 'da': None, 'magcorr': None, 'n': len(preds)}

    if tier_frac < 1.0:
        abs_preds = np.abs(preds)
        threshold = np.percentile(abs_preds, 100 * (1 - tier_frac))
        mask = abs_preds >= threshold
        preds = preds[mask]
        labels = labels[mask]

    if len(preds) < 10:
        return {'ic': None, 'da': None, 'magcorr': None, 'n': len(preds)}

    # IC (rank correlation)
    ic = stats.spearmanr(preds, labels)[0]

    # DA (direction accuracy) - exclude zero labels
    nonzero = labels != 0
    if nonzero.sum() > 10:
        da = np.mean(np.sign(preds[nonzero]) == np.sign(labels[nonzero]))
    else:
        da = None

    # MagCorr (correlation of magnitudes)
    magcorr = stats.spearmanr(np.abs(preds), np.abs(labels))[0]

    return {'ic': float(ic) if not np.isnan(ic) else None,
            'da': float(da) if da is not None else None,
            'magcorr': float(magcorr) if not np.isnan(magcorr) else None,
            'n': int(len(preds))}

def analyze_ic_da_disconnect(preds, labels):
    """Why IC rises but DA stays low."""
    results = {}

    # 1. Prediction distribution analysis
    results['pred_mean'] = float(np.mean(preds))
    results['pred_std'] = float(np.std(preds))
    results['pred_abs_mean'] = float(np.mean(np.abs(preds)))
    results['pred_near_zero_pct'] = float(np.mean(np.abs(preds) < np.std(preds) * 0.1))
    results['label_mean'] = float(np.mean(labels))
    results['label_std'] = float(np.std(labels))

    # 2. Signed vs unsigned decomposition
    nonzero = labels != 0
    p, l = preds[nonzero], labels[nonzero]

    # Direction component: sign agreement
    sign_agree = np.sign(p) == np.sign(l)
    results['direction_accuracy'] = float(np.mean(sign_agree))

    # Among correct-direction predictions
    if sign_agree.sum() > 10:
        results['ic_correct_direction'] = float(stats.spearmanr(p[sign_agree], l[sign_agree])[0])
    # Among wrong-direction predictions
    wrong = ~sign_agree
    if wrong.sum() > 10:
        results['ic_wrong_direction'] = float(stats.spearmanr(p[wrong], l[wrong])[0])

    # 3. Magnitude ranking when direction is wrong
    # Key question: when direction is wrong, is |pred| still correlated with |label|?
    if wrong.sum() > 10:
        results['magcorr_wrong_direction'] = float(stats.spearmanr(np.abs(p[wrong]), np.abs(l[wrong]))[0])
    if sign_agree.sum() > 10:
        results['magcorr_correct_direction'] = float(stats.spearmanr(np.abs(p[sign_agree]), np.abs(l[sign_agree]))[0])

    # 4. Quintile analysis: split predictions into 5 quintiles, check mean label in each
    quintiles = np.percentile(preds, [20, 40, 60, 80])
    bins = np.digitize(preds, quintiles)
    quintile_means = []
    for q in range(5):
        mask = bins == q
        if mask.sum() > 0:
            quintile_means.append({
                'quintile': q,
                'pred_mean': float(np.mean(preds[mask])),
                'label_mean': float(np.mean(labels[mask])),
                'label_abs_mean': float(np.mean(np.abs(labels[mask]))),
                'da': float(np.mean(np.sign(preds[mask][labels[mask]!=0]) == np.sign(labels[mask][labels[mask]!=0]))) if (labels[mask]!=0).sum() > 0 else None,
                'n': int(mask.sum())
            })
    results['quintile_analysis'] = quintile_means

    # 5. Confidence vs correctness: does higher |pred| mean more likely correct direction?
    abs_pred = np.abs(p)
    for pct_name, pct_val in [('top50', 50), ('top25', 75), ('top10', 90), ('top5', 95)]:
        thresh = np.percentile(abs_pred, pct_val)
        high_conf = abs_pred >= thresh
        if high_conf.sum() > 10:
            results[f'da_{pct_name}_confidence'] = float(np.mean(sign_agree[high_conf]))

    return results

def analyze_volatility_prediction(preds, labels):
    """Can PatchTST predict volatility (|movement|) regardless of direction?"""
    abs_pred = np.abs(preds)
    abs_label = np.abs(labels)

    results = {}

    # Overall vol correlation
    results['vol_spearman'] = float(stats.spearmanr(abs_pred, abs_label)[0])
    results['vol_pearson'] = float(stats.pearsonr(abs_pred, abs_label)[0])

    # Quintile vol analysis
    quintiles = np.percentile(abs_pred, [20, 40, 60, 80])
    bins = np.digitize(abs_pred, quintiles)
    vol_quintiles = []
    for q in range(5):
        mask = bins == q
        if mask.sum() > 0:
            vol_quintiles.append({
                'quintile': q,
                'pred_abs_mean': float(np.mean(abs_pred[mask])),
                'label_abs_mean': float(np.mean(abs_label[mask])),
                'label_abs_median': float(np.median(abs_label[mask])),
                'label_std': float(np.std(labels[mask])),
                'n': int(mask.sum())
            })
    results['vol_quintile_analysis'] = vol_quintiles

    # MFE/MAE at confidence tiers (for long predictions)
    long_mask = preds > 0
    short_mask = preds < 0

    for direction, mask, name in [(True, long_mask, 'long'), (False, short_mask, 'short')]:
        if mask.sum() < 10:
            continue
        p_sub = preds[mask]
        l_sub = labels[mask]
        # MFE = max favorable excursion ~ mean label when direction correct
        correct = np.sign(p_sub) == np.sign(l_sub)
        wrong = ~correct & (l_sub != 0)
        results[f'{name}_pct_correct'] = float(correct.sum() / max(1, (l_sub != 0).sum()))
        if correct.sum() > 0:
            results[f'{name}_mfe_mean'] = float(np.mean(np.abs(l_sub[correct])))
        if wrong.sum() > 0:
            results[f'{name}_mae_mean'] = float(np.mean(np.abs(l_sub[wrong])))

    return results

def analyze_long_short(preds, labels):
    """Long vs short performance."""
    results = {}
    for name, mask_fn in [('long', lambda p: p > 0), ('short', lambda p: p < 0)]:
        mask = mask_fn(preds)
        if mask.sum() < 10:
            continue
        p, l = preds[mask], labels[mask]
        results[name] = {
            'n': int(mask.sum()),
            'pct_of_total': float(mask.sum() / len(preds)),
            'ic': float(stats.spearmanr(p, l)[0]),
            'da': float(np.mean(np.sign(p[l!=0]) == np.sign(l[l!=0]))) if (l!=0).sum() > 10 else None,
            'magcorr': float(stats.spearmanr(np.abs(p), np.abs(l))[0]),
            'mean_pred': float(np.mean(p)),
            'mean_label': float(np.mean(l)),
            'mean_abs_label': float(np.mean(np.abs(l))),
        }
    return results


def main():
    all_results = {
        'model': 'PatchTST_smart_v2',
        'skipped_folds': list(SKIP_FOLDS),
    }

    # ========== PER-FOLD ANALYSIS ==========
    fold_results = []
    concat_preds = {h: [] for h in range(3)}
    concat_labels = {h: [] for h in range(3)}

    for fold_idx in range(7):
        if fold_idx in SKIP_FOLDS:
            continue

        d = np.load(OUTPUT_DIR / f'fold_0{fold_idx}_oot_predictions.npz')
        preds = d['predictions']  # (N, 3)
        labels = d['labels']      # (N, 3)

        fold_data = {'fold': fold_idx, 'n_samples': int(preds.shape[0])}

        for h_idx, h_name in enumerate(HORIZONS):
            p = preds[:, h_idx]
            l = labels[:, h_idx]

            concat_preds[h_idx].append(p)
            concat_labels[h_idx].append(l)

            tier_results = {}
            for tier_name, tier_frac in TIERS.items():
                tier_results[tier_name] = compute_metrics(p, l, tier_frac)

            fold_data[h_name] = tier_results

        fold_results.append(fold_data)

    all_results['per_fold'] = fold_results

    # ========== CONCAT ANALYSIS ==========
    concat_data = {}
    for h_idx, h_name in enumerate(HORIZONS):
        p = np.concatenate(concat_preds[h_idx])
        l = np.concatenate(concat_labels[h_idx])

        tier_results = {}
        for tier_name, tier_frac in TIERS.items():
            tier_results[tier_name] = compute_metrics(p, l, tier_frac)

        concat_data[h_name] = {
            'tiers': tier_results,
            'n_total': int(len(p)),
            'ic_da_disconnect': analyze_ic_da_disconnect(p, l),
            'volatility_prediction': analyze_volatility_prediction(p, l),
            'long_short': analyze_long_short(p, l),
        }

    all_results['concat'] = concat_data

    # ========== PER-FOLD TREND ==========
    trend = {}
    for h_name in HORIZONS:
        fold_ics = []
        fold_das = []
        fold_magcorrs = []
        for fr in fold_results:
            m = fr[h_name]['All']
            fold_ics.append(m['ic'])
            fold_das.append(m['da'])
            fold_magcorrs.append(m['magcorr'])
        trend[h_name] = {
            'fold_ids': [fr['fold'] for fr in fold_results],
            'ic_trend': fold_ics,
            'da_trend': fold_das,
            'magcorr_trend': fold_magcorrs,
            'ic_slope': float(np.polyfit(range(len(fold_ics)), fold_ics, 1)[0]) if len(fold_ics) > 1 else None,
            'da_slope': float(np.polyfit(range(len(fold_das)), fold_das, 1)[0]) if len(fold_das) > 1 else None,
            'magcorr_slope': float(np.polyfit(range(len(fold_magcorrs)), fold_magcorrs, 1)[0]) if len(fold_magcorrs) > 1 else None,
        }
    all_results['fold_trend'] = trend

    # ========== COMPARISON WITH MAMBA V7 ==========
    try:
        md = np.load(str(MAMBA_PATH))
        comparison = {}
        for h_name in HORIZONS:
            mp = md[f'preds_{h_name}']
            ml = md[f'labels_{h_name}']

            mamba_tiers = {}
            for tier_name, tier_frac in TIERS.items():
                mamba_tiers[tier_name] = compute_metrics(mp, ml, tier_frac)

            # PatchTST concat for this horizon
            p = np.concatenate(concat_preds[HORIZONS.index(h_name)])
            l = np.concatenate(concat_labels[HORIZONS.index(h_name)])
            ptst_tiers = {}
            for tier_name, tier_frac in TIERS.items():
                ptst_tiers[tier_name] = compute_metrics(p, l, tier_frac)

            comparison[h_name] = {
                'mamba_v7': mamba_tiers,
                'patchtst_smart_v2': ptst_tiers,
                'mamba_vol_analysis': analyze_volatility_prediction(mp, ml),
            }
        all_results['comparison_mamba_v7'] = comparison
    except Exception as e:
        all_results['comparison_mamba_v7'] = f'Error: {str(e)}'

    # ========== SAVE ==========
    out_path = OUTPUT_DIR / 'patchtst_deep_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ========== PRINT SUMMARY ==========
    print("=" * 80)
    print("PATCHTST SMART_V2 DEEP ANALYSIS")
    print("=" * 80)

    print(f"\nFolds analyzed: {[fr['fold'] for fr in fold_results]}")
    print(f"Total concat samples: {sum(fr['n_samples'] for fr in fold_results)}")

    print("\n" + "=" * 80)
    print("1. PER-FOLD IC / DA / MAGCORR (All samples, 10s horizon)")
    print("=" * 80)
    print(f"{'Fold':>5} {'N':>8} {'IC_1s':>8} {'IC_5s':>8} {'IC_10s':>8} {'DA_1s':>8} {'DA_5s':>8} {'DA_10s':>8} {'Mag_10s':>8}")
    for fr in fold_results:
        print(f"{fr['fold']:>5} {fr['n_samples']:>8} "
              f"{fr['1s']['All']['ic']:>8.4f} {fr['5s']['All']['ic']:>8.4f} {fr['10s']['All']['ic']:>8.4f} "
              f"{fr['1s']['All']['da']:>8.4f} {fr['5s']['All']['da']:>8.4f} {fr['10s']['All']['da']:>8.4f} "
              f"{fr['10s']['All']['magcorr']:>8.4f}")

    print("\n" + "=" * 80)
    print("2. CONCAT TIER BREAKDOWN (10s horizon)")
    print("=" * 80)
    c10 = all_results['concat']['10s']['tiers']
    print(f"{'Tier':>12} {'N':>8} {'IC':>8} {'DA':>8} {'MagCorr':>8}")
    for tier_name in TIERS:
        m = c10[tier_name]
        ic_s = f"{m['ic']:.4f}" if m['ic'] is not None else "N/A"
        da_s = f"{m['da']:.4f}" if m['da'] is not None else "N/A"
        mc_s = f"{m['magcorr']:.4f}" if m['magcorr'] is not None else "N/A"
        print(f"{tier_name:>12} {m['n']:>8} {ic_s:>8} {da_s:>8} {mc_s:>8}")

    print("\n" + "=" * 80)
    print("3. IC vs DA DISCONNECT ANALYSIS (10s horizon)")
    print("=" * 80)
    disc = all_results['concat']['10s']['ic_da_disconnect']
    print(f"  Direction accuracy (all):       {disc['direction_accuracy']:.4f}")
    for k in ['da_top50_confidence', 'da_top25_confidence', 'da_top10_confidence', 'da_top5_confidence']:
        if k in disc:
            print(f"  DA at {k.replace('da_','').replace('_confidence','')} confidence: {disc[k]:.4f}")
    print(f"  IC among correct-direction:     {disc.get('ic_correct_direction', 'N/A')}")
    print(f"  IC among wrong-direction:       {disc.get('ic_wrong_direction', 'N/A')}")
    print(f"  MagCorr correct-direction:      {disc.get('magcorr_correct_direction', 'N/A')}")
    print(f"  MagCorr wrong-direction:        {disc.get('magcorr_wrong_direction', 'N/A')}")
    print(f"  Pred near-zero pct:             {disc['pred_near_zero_pct']:.4f}")
    print(f"  Pred std:                       {disc['pred_std']:.4f}")
    print(f"  Label std:                      {disc['label_std']:.4f}")

    print("\n  Quintile analysis (pred quintile -> mean label):")
    for q in disc['quintile_analysis']:
        da_s = f"{q['da']:.4f}" if q['da'] is not None else "N/A"
        print(f"    Q{q['quintile']}: pred_mean={q['pred_mean']:>8.4f}  label_mean={q['label_mean']:>8.4f}  |label|_mean={q['label_abs_mean']:>8.4f}  DA={da_s}  N={q['n']}")

    print("\n" + "=" * 80)
    print("4. VOLATILITY PREDICTION ANALYSIS (10s horizon)")
    print("=" * 80)
    vol = all_results['concat']['10s']['volatility_prediction']
    print(f"  Vol Spearman (|pred| vs |label|):  {vol['vol_spearman']:.4f}")
    print(f"  Vol Pearson  (|pred| vs |label|):  {vol['vol_pearson']:.4f}")
    print(f"\n  Volatility quintile analysis:")
    for q in vol['vol_quintile_analysis']:
        print(f"    Q{q['quintile']}: |pred|_mean={q['pred_abs_mean']:>8.4f}  |label|_mean={q['label_abs_mean']:>8.4f}  label_std={q['label_std']:>8.4f}  N={q['n']}")

    for side in ['long', 'short']:
        if f'{side}_pct_correct' in vol:
            print(f"\n  {side.upper()} side:")
            print(f"    Pct correct:  {vol[f'{side}_pct_correct']:.4f}")
            if f'{side}_mfe_mean' in vol:
                print(f"    MFE (mean):   {vol[f'{side}_mfe_mean']:.4f}")
            if f'{side}_mae_mean' in vol:
                print(f"    MAE (mean):   {vol[f'{side}_mae_mean']:.4f}")

    print("\n" + "=" * 80)
    print("5. LONG vs SHORT BREAKDOWN (10s horizon)")
    print("=" * 80)
    ls = all_results['concat']['10s']['long_short']
    for side in ['long', 'short']:
        if side in ls:
            s = ls[side]
            print(f"  {side.upper()}: N={s['n']:,} ({s['pct_of_total']:.1%})  IC={s['ic']:.4f}  DA={s['da']:.4f}  MagCorr={s['magcorr']:.4f}  mean_pred={s['mean_pred']:.4f}  mean_label={s['mean_label']:.4f}")

    print("\n" + "=" * 80)
    print("6. PER-FOLD TREND (slope of IC/DA/MagCorr across folds)")
    print("=" * 80)
    for h in HORIZONS:
        t = all_results['fold_trend'][h]
        print(f"  {h}: IC_slope={t['ic_slope']:+.5f}  DA_slope={t['da_slope']:+.5f}  MagCorr_slope={t['magcorr_slope']:+.5f}")
        print(f"       ICs: {['%.4f' % x for x in t['ic_trend']]}")
        print(f"       DAs: {['%.4f' % x for x in t['da_trend']]}")

    print("\n" + "=" * 80)
    print("7. COMPARISON: PatchTST vs Mamba v7 (concat, 10s)")
    print("=" * 80)
    if isinstance(all_results.get('comparison_mamba_v7'), dict):
        comp = all_results['comparison_mamba_v7']['10s']
        print(f"{'Tier':>12} {'PatchTST IC':>12} {'Mamba IC':>12} {'PatchTST DA':>12} {'Mamba DA':>12} {'PatchTST Mag':>12} {'Mamba Mag':>12}")
        for tier in TIERS:
            pt = comp['patchtst_smart_v2'][tier]
            mb = comp['mamba_v7'][tier]
            def fmt(v): return f"{v:.4f}" if v is not None else "N/A"
            print(f"{tier:>12} {fmt(pt['ic']):>12} {fmt(mb['ic']):>12} {fmt(pt['da']):>12} {fmt(mb['da']):>12} {fmt(pt['magcorr']):>12} {fmt(mb['magcorr']):>12}")

        # Mamba vol analysis
        mvol = comp['mamba_vol_analysis']
        print(f"\n  Mamba v7 Vol Spearman: {mvol['vol_spearman']:.4f} vs PatchTST: {vol['vol_spearman']:.4f}")
    else:
        print(f"  {all_results.get('comparison_mamba_v7')}")

    # ========== ALL HORIZONS CONCAT SUMMARY ==========
    print("\n" + "=" * 80)
    print("8. ALL HORIZONS CONCAT SUMMARY")
    print("=" * 80)
    print(f"{'Horizon':>8} {'IC':>8} {'DA':>8} {'MagCorr':>8} {'VolCorr':>8}")
    for h in HORIZONS:
        c = all_results['concat'][h]
        m = c['tiers']['All']
        vc = c['volatility_prediction']['vol_spearman']
        print(f"{h:>8} {m['ic']:.4f} {m['da']:.4f} {m['magcorr']:.4f} {vc:.4f}")

    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
