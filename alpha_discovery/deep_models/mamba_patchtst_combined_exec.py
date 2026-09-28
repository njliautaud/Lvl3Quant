#!/usr/bin/env python3
"""
MAMBA + PatchTST COMBINED EXECUTION ENGINE
===========================================
Uses Mamba v7 as backbone (magnitude/IC) + PatchTST as DA filter (direction).
Tests whether PatchTST confirmation improves Mamba-only strategies.

Strategies:
1. Mamba-only baseline (top confidence tiers)
2. PatchTST DA filter: only trade when PatchTST agrees on direction
3. PatchTST embedding similarity: use PatchTST embeddings to find
   "high-confidence microstructure states"
4. Ensemble: average Mamba + PatchTST predictions, then apply tiers
5. MLP gate on combined embeddings (Mamba 96d + PatchTST 256d = 352d)

Requires overlapping OOT dates between Mamba and PatchTST predictions.
"""

import numpy as np
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
MAMBA_DIR = Path("/home/jupiter/Lvl3Quant/output/mamba_v7_tiny_smart_v3_mar_apr")
PATCHTST_DIR = Path("/home/jupiter/Lvl3Quant/output/patchtst_smart_v2_mar")
RESULTS_DIR = Path("/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COST_TICKS = 0.376  # $4.70 RT


def extract_date_from_fold(npz_path):
    """Extract date string from fold file's oot_files metadata."""
    d = np.load(npz_path, allow_pickle=True)
    oot_files = d.get('oot_files', None)
    if oot_files is not None:
        if hasattr(oot_files, 'item'):
            oot_files = oot_files.item()
        if isinstance(oot_files, (list, np.ndarray)) and len(oot_files) > 0:
            fname = str(oot_files[0]).split('/')[-1].split('\\')[-1]
            return fname[:8]  # YYYYMMDD
        return str(oot_files).split('/')[-1].split('\\')[-1][:8]
    return None


def load_model_data(pred_dir, model_name):
    """Load all fold predictions with date mapping."""
    folds = {}
    for f in sorted(pred_dir.glob("fold_*_oot_predictions.npz")):
        fold_num = int(f.name.split("_")[1])
        date = extract_date_from_fold(f)
        if date is None:
            continue

        d = np.load(f, allow_pickle=True)
        folds[date] = {
            'fold': fold_num,
            'preds': d['predictions'],
            'labels': d['labels'],
            'embeddings': d.get('embeddings', None),
            'n': len(d['predictions']),
        }
        log.info(f"  {model_name} {date}: {len(d['predictions']):,} events, fold {fold_num}")

    return folds


def align_predictions(mamba_data, patchtst_data):
    """
    Find overlapping dates and align predictions.
    Both models must have same OOT date AND same number of events.
    If event counts differ (different stride/window), truncate to minimum.
    """
    common_dates = sorted(set(mamba_data.keys()) & set(patchtst_data.keys()))

    if not common_dates:
        log.warning("NO overlapping dates between Mamba and PatchTST!")
        log.info(f"  Mamba dates: {sorted(mamba_data.keys())}")
        log.info(f"  PatchTST dates: {sorted(patchtst_data.keys())}")
        return None

    aligned = []
    for date in common_dates:
        m = mamba_data[date]
        p = patchtst_data[date]

        n_min = min(m['n'], p['n'])

        aligned.append({
            'date': date,
            'mamba_preds': m['preds'][:n_min],
            'mamba_labels': m['labels'][:n_min],
            'mamba_embeds': m['embeddings'][:n_min] if m['embeddings'] is not None else None,
            'patchtst_preds': p['preds'][:n_min],
            'patchtst_labels': p['labels'][:n_min],
            'patchtst_embeds': p['embeddings'][:n_min] if p['embeddings'] is not None else None,
            'n': n_min,
        })

        if m['n'] != p['n']:
            log.warning(f"  {date}: Mamba={m['n']} vs PatchTST={p['n']} events — truncated to {n_min}")
        else:
            log.info(f"  {date}: {n_min:,} events aligned")

    return aligned


def compute_pnl(labels, mask, direction, horizon=0):
    """Compute PnL for selected trades."""
    if mask.sum() == 0:
        return {'n': 0, 'pnl': 0, 'da': 0, 'wr': 0, 'avg_pnl': 0, 'pf': 0, 'sortino': 0}

    sel_labels = labels[mask, horizon]
    sel_dir = direction[mask]

    pnl_ticks = sel_labels * sel_dir - COST_TICKS
    pnl_dollars = pnl_ticks * TICK_VALUE

    actual_dir = np.sign(sel_labels)
    correct = (sel_dir == actual_dir) | (sel_labels == 0)
    da = correct.mean()
    wr = (pnl_dollars > 0).mean()

    winners = pnl_dollars[pnl_dollars > 0]
    losers = pnl_dollars[pnl_dollars <= 0]
    pf = winners.sum() / abs(losers.sum()) if losers.sum() != 0 else float('inf')

    neg = pnl_dollars[pnl_dollars < 0]
    downside = np.sqrt(np.mean(neg**2)) if len(neg) > 0 else 1
    sortino = pnl_dollars.mean() / downside if downside > 0 else 0

    return {
        'n': int(mask.sum()),
        'pnl': float(pnl_dollars.sum()),
        'da': float(da),
        'wr': float(wr),
        'avg_pnl': float(pnl_dollars.mean()),
        'pf': float(pf),
        'sortino': float(sortino),
    }


def strategy_mamba_only(mamba_preds, labels, percentile=99):
    """Baseline: Mamba confidence tier only."""
    strength = np.abs(mamba_preds[:, 0])
    thresh = np.percentile(strength, percentile)
    mask = strength >= thresh
    direction = np.sign(mamba_preds[:, 0])
    return mask, direction


def strategy_patchtst_filter(mamba_preds, patchtst_preds, labels, percentile=99):
    """Mamba confidence + PatchTST direction agreement."""
    strength = np.abs(mamba_preds[:, 0])
    thresh = np.percentile(strength, percentile)
    mamba_mask = strength >= thresh

    mamba_dir = np.sign(mamba_preds[:, 0])
    patchtst_dir = np.sign(patchtst_preds[:, 0])
    agree = mamba_dir == patchtst_dir

    mask = mamba_mask & agree
    direction = mamba_dir
    return mask, direction


def strategy_patchtst_strong_filter(mamba_preds, patchtst_preds, labels,
                                      mamba_pct=99, patchtst_pct=75):
    """Mamba top confidence + PatchTST also confident (not just direction)."""
    m_strength = np.abs(mamba_preds[:, 0])
    m_thresh = np.percentile(m_strength, mamba_pct)
    mamba_mask = m_strength >= m_thresh

    p_strength = np.abs(patchtst_preds[:, 0])
    p_thresh = np.percentile(p_strength, patchtst_pct)
    patchtst_confident = p_strength >= p_thresh

    mamba_dir = np.sign(mamba_preds[:, 0])
    patchtst_dir = np.sign(patchtst_preds[:, 0])
    agree = mamba_dir == patchtst_dir

    mask = mamba_mask & patchtst_confident & agree
    direction = mamba_dir
    return mask, direction


def strategy_ensemble(mamba_preds, patchtst_preds, labels, percentile=99):
    """Average predictions from both models, then apply confidence tier."""
    # Normalize both to similar scales before averaging
    m_std = mamba_preds[:, 0].std()
    p_std = patchtst_preds[:, 0].std()

    if m_std > 0 and p_std > 0:
        m_norm = mamba_preds[:, 0] / m_std
        p_norm = patchtst_preds[:, 0] / p_std
        ensemble = (m_norm + p_norm) / 2
    else:
        ensemble = mamba_preds[:, 0]

    strength = np.abs(ensemble)
    thresh = np.percentile(strength, percentile)
    mask = strength >= thresh
    direction = np.sign(ensemble)

    # Use mamba labels (should be identical to patchtst labels)
    return mask, direction


def strategy_mlp_combined_embeddings(mamba_embeds, patchtst_embeds, labels,
                                       horizon=0, threshold=0.6):
    """Train tiny MLP on combined embeddings to predict profitability."""
    if mamba_embeds is None or patchtst_embeds is None:
        return np.zeros(len(labels), dtype=bool), np.ones(len(labels))

    # Combine embeddings
    combined = np.concatenate([mamba_embeds, patchtst_embeds], axis=1)  # (N, 352)

    # Target: was this event profitable at horizon?
    # Use mamba predictions for direction
    # (In practice we'd use a proper train/test split per fold)

    n = len(labels)
    if n < 200:
        return np.zeros(n, dtype=bool), np.ones(n)

    # Simple approach: train on first 70%, test on last 30%
    split = int(0.7 * n)

    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    # Target: profitable trade (direction * label > cost)
    direction_all = np.sign(combined[:, :96].mean(axis=1))  # Use mamba embed mean as proxy
    # Actually, we need predictions for direction. Use a simpler approach:
    # Train MLP to predict: is |label| > cost_ticks? (i.e., worth trading)
    target = (np.abs(labels[:, horizon]) > COST_TICKS * 1.5).astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(combined[:split])
    X_test = scaler.transform(combined[split:])
    y_train = target[:split]

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                        random_state=42, early_stopping=True,
                        validation_fraction=0.15)
    try:
        mlp.fit(X_train, y_train)
        probs = mlp.predict_proba(X_test)[:, 1]
    except Exception as e:
        log.warning(f"MLP training failed: {e}")
        return np.zeros(n, dtype=bool), np.ones(n)

    # Select high-probability entries
    test_mask = np.zeros(n, dtype=bool)
    test_direction = np.sign(labels[:, 0])  # Cheating on direction — placeholder

    # Use mamba predictions for actual direction in test set
    high_prob = probs >= threshold
    test_indices = np.arange(split, n)[high_prob]
    test_mask[test_indices] = True

    # Direction from mamba embedding mean (proxy)
    test_direction = np.sign(combined[:, :96].mean(axis=1))

    return test_mask, test_direction


def run_analysis():
    """Main analysis: compare Mamba-only vs Mamba+PatchTST strategies."""
    log.info("=" * 70)
    log.info("MAMBA + PatchTST COMBINED EXECUTION ENGINE")
    log.info("=" * 70)
    log.info(f"Cost: {COST_TICKS} ticks (${COST_TICKS * TICK_VALUE:.2f} RT)")
    log.info("")

    # Load both models
    log.info("Loading Mamba v7 predictions...")
    mamba_data = load_model_data(MAMBA_DIR, "Mamba")

    log.info("\nLoading PatchTST predictions...")
    if not PATCHTST_DIR.exists():
        log.error(f"PatchTST dir not found: {PATCHTST_DIR}")
        log.info("PatchTST is still training on Razer. Run this again when it has March predictions.")
        log.info("\nFalling back to Mamba-only analysis with additional strategies...")
        run_mamba_only_extended(mamba_data)
        return

    patchtst_data = load_model_data(PATCHTST_DIR, "PatchTST")

    if not patchtst_data:
        log.warning("No PatchTST predictions found yet.")
        log.info("PatchTST is still training. Running Mamba-only extended analysis...")
        run_mamba_only_extended(mamba_data)
        return

    # Align predictions
    log.info("\nAligning predictions by date...")
    aligned = align_predictions(mamba_data, patchtst_data)

    if aligned is None or len(aligned) == 0:
        log.warning("No overlapping dates. Running Mamba-only extended analysis...")
        run_mamba_only_extended(mamba_data)
        return

    log.info(f"\n{len(aligned)} overlapping days found!")

    # Run all strategies
    strategies = {
        'mamba_top1': lambda m, p, l: strategy_mamba_only(m, l, 99),
        'mamba_top05': lambda m, p, l: strategy_mamba_only(m, l, 99.5),
        'ptst_filter_top1': lambda m, p, l: strategy_patchtst_filter(m, p, l, 99),
        'ptst_filter_top05': lambda m, p, l: strategy_patchtst_filter(m, p, l, 99.5),
        'ptst_strong_top1': lambda m, p, l: strategy_patchtst_strong_filter(m, p, l, 99, 75),
        'ptst_strong_top05': lambda m, p, l: strategy_patchtst_strong_filter(m, p, l, 99.5, 75),
        'ensemble_top1': lambda m, p, l: strategy_ensemble(m, p, l, 99),
        'ensemble_top05': lambda m, p, l: strategy_ensemble(m, p, l, 99.5),
    }

    all_results = {}

    for horizon_idx, horizon_name in [(0, '1s'), (1, '5s'), (2, '10s')]:
        log.info(f"\n{'='*70}")
        log.info(f"HORIZON: {horizon_name}")
        log.info(f"{'='*70}")

        for strat_name, strat_fn in strategies.items():
            total_pnl = 0
            total_n = 0
            daily_pnls = []
            daily_das = []

            for day in aligned:
                mask, direction = strat_fn(day['mamba_preds'], day['patchtst_preds'],
                                           day['mamba_labels'])
                result = compute_pnl(day['mamba_labels'], mask, direction, horizon_idx)
                total_pnl += result['pnl']
                total_n += result['n']
                daily_pnls.append(result['pnl'])
                if result['n'] > 0:
                    daily_das.append(result['da'])

            avg_da = np.mean(daily_das) if daily_das else 0
            avg_pnl = total_pnl / total_n if total_n > 0 else 0

            key = f"{strat_name}_{horizon_name}"
            all_results[key] = {
                'strategy': strat_name,
                'horizon': horizon_name,
                'total_n': total_n,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'avg_da': avg_da,
                'daily_pnls': daily_pnls,
            }

            log.info(f"  {strat_name:<25} n={total_n:>6} DA={avg_da:.1%} "
                    f"PnL=${total_pnl:>10,.0f} $/trade=${avg_pnl:>8.2f}")

    # Summary comparison
    log.info(f"\n\n{'='*90}")
    log.info("COMPARISON: Does PatchTST filter improve Mamba?")
    log.info(f"{'='*90}")

    for horizon_name in ['1s', '5s', '10s']:
        log.info(f"\n--- {horizon_name} hold ---")
        baseline_key = f"mamba_top1_{horizon_name}"
        filter_key = f"ptst_filter_top1_{horizon_name}"
        strong_key = f"ptst_strong_top1_{horizon_name}"
        ensemble_key = f"ensemble_top1_{horizon_name}"

        for key, label in [(baseline_key, "Mamba only"),
                           (filter_key, "+ PatchTST direction"),
                           (strong_key, "+ PatchTST strong"),
                           (ensemble_key, "Ensemble avg")]:
            if key in all_results:
                r = all_results[key]
                improvement = ""
                if key != baseline_key and baseline_key in all_results:
                    base_pnl = all_results[baseline_key]['avg_pnl']
                    if base_pnl != 0:
                        pct = (r['avg_pnl'] - base_pnl) / abs(base_pnl) * 100
                        improvement = f" ({'+' if pct > 0 else ''}{pct:.0f}%)"
                log.info(f"  {label:<25} n={r['total_n']:>6} DA={r['avg_da']:.1%} "
                        f"${r['avg_pnl']:>8.2f}/trade{improvement}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"mamba_patchtst_combined_{ts}.json"
    save_data = {k: {kk: vv for kk, vv in v.items() if kk != 'daily_pnls'}
                 for k, v in all_results.items()}
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")


def run_mamba_only_extended(mamba_data):
    """Extended Mamba-only analysis while waiting for PatchTST."""
    log.info("\n" + "=" * 70)
    log.info("EXTENDED MAMBA-ONLY ANALYSIS (PatchTST not yet available)")
    log.info("=" * 70)

    # Aggregate all folds
    all_preds = []
    all_labels = []
    all_embeds = []

    for date in sorted(mamba_data.keys()):
        d = mamba_data[date]
        all_preds.append(d['preds'])
        all_labels.append(d['labels'])
        if d['embeddings'] is not None:
            all_embeds.append(d['embeddings'])

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    embeds = np.concatenate(all_embeds) if all_embeds else None

    log.info(f"Total: {len(preds):,} events across {len(mamba_data)} days")

    # Strategy: Adaptive threshold (different threshold per day based on volatility)
    log.info("\n--- ADAPTIVE THRESHOLD (vol-adjusted) ---")
    for date in sorted(mamba_data.keys()):
        d = mamba_data[date]
        vol = np.std(d['labels'][:, 0])  # 1s volatility
        strength = np.abs(d['preds'][:, 0])

        # Higher vol → stricter threshold
        base_pct = 99
        vol_adj = min(vol / np.std(labels[:, 0]), 2.0)  # Relative vol
        adj_pct = min(base_pct + (vol_adj - 1) * 0.5, 99.9)

        thresh = np.percentile(strength, adj_pct)
        mask = strength >= thresh
        direction = np.sign(d['preds'][:, 0])

        result = compute_pnl(d['labels'], mask, direction, horizon=0)
        log.info(f"  {date}: vol={vol:.3f} adj_pct={adj_pct:.1f}% "
                f"n={result['n']:>4} DA={result['da']:.1%} PnL=${result['pnl']:>8,.0f}")

    # Strategy: Time-of-day analysis
    log.info("\n--- POSITION IN DAY (early/mid/late session) ---")
    for date in sorted(mamba_data.keys()):
        d = mamba_data[date]
        n = d['n']
        thirds = [slice(0, n//3), slice(n//3, 2*n//3), slice(2*n//3, n)]
        labels_t = ['Early', 'Mid  ', 'Late ']

        for label_t, s in zip(labels_t, thirds):
            p = d['preds'][s]
            l = d['labels'][s]
            strength = np.abs(p[:, 0])
            if len(strength) < 10:
                continue
            thresh = np.percentile(strength, 99)
            mask = strength >= thresh
            direction = np.sign(p[:, 0])
            result = compute_pnl(l, mask, direction, horizon=0)
            if result['n'] > 0:
                log.info(f"  {date} {label_t}: n={result['n']:>3} "
                        f"DA={result['da']:.1%} PnL=${result['pnl']:>6,.0f}")

    # Strategy: Momentum regime (recent IC as regime filter)
    log.info("\n--- ROLLING IC REGIME FILTER ---")
    log.info("(Skip events where recent IC is low → model uncertain)")

    window = 500  # Rolling window for IC estimation
    for date in sorted(mamba_data.keys()):
        d = mamba_data[date]
        if d['n'] < window * 2:
            continue

        # Compute rolling IC
        n = d['n']
        rolling_ic = np.zeros(n)
        for i in range(window, n):
            p_win = d['preds'][i-window:i, 0]
            l_win = d['labels'][i-window:i, 0]
            if p_win.std() > 0 and l_win.std() > 0:
                rolling_ic[i] = np.corrcoef(p_win, l_win)[0, 1]

        # Only trade when rolling IC > threshold
        for ic_thresh in [0.1, 0.15, 0.2]:
            high_ic = rolling_ic > ic_thresh
            strength = np.abs(d['preds'][:, 0])
            top1 = strength >= np.percentile(strength, 99)
            mask = high_ic & top1
            direction = np.sign(d['preds'][:, 0])
            result = compute_pnl(d['labels'], mask, direction, horizon=0)
            if result['n'] > 0:
                log.info(f"  {date} IC>{ic_thresh}: n={result['n']:>3} "
                        f"DA={result['da']:.1%} PnL=${result['pnl']:>6,.0f} "
                        f"$/trade=${result['avg_pnl']:>6.2f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"mamba_extended_analysis_{ts}.json"
    log.info(f"\nAnalysis complete. Results logged above.")
    log.info(f"Waiting for PatchTST March predictions to run combined analysis.")


if __name__ == '__main__':
    run_analysis()
