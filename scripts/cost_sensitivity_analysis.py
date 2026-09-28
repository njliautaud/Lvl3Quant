#!/usr/bin/env python3
"""
Cost Sensitivity Analysis for LGBM smart_v3 and Mamba v7 smart_v3 predictions.

Tests profitability at multiple transaction cost levels across confidence tiers.
"""

import numpy as np
import json
import glob
from collections import defaultdict

# === CONFIG ===
COST_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # in ticks (round-trip)
CONFIDENCE_TIERS = {
    'All':    (0.0, 1.0),
    'Top50%': (0.50, 1.0),
    'Top25%': (0.75, 1.0),
    'Top10%': (0.90, 1.0),
    'Top5%':  (0.95, 1.0),
    'Top1%':  (0.99, 1.0),
}
TICK_VALUE = 12.50  # ES futures: 1 tick = $12.50

# === LOAD LGBM DATA ===
def load_lgbm():
    """Load all LGBM fold predictions. Binary classifier with probs/confidence."""
    all_probs = []
    all_labels = []
    all_confidence = []

    files = sorted(glob.glob('/home/jupiter/Lvl3Quant/output/lgbm_da_smart_v3_1d_oot/fold*_preds.npz'))
    print(f"LGBM: Loading {len(files)} folds...")

    for f in files:
        d = np.load(f)
        all_probs.append(d['probs'])
        all_labels.append(d['labels'])
        all_confidence.append(d['confidence'])

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    confidence = np.concatenate(all_confidence)

    print(f"  Total samples: {len(probs):,}")
    print(f"  Label balance: {labels.mean():.3f} (up fraction)")
    print(f"  Confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]")

    return probs, labels, confidence


def load_mamba():
    """Load all Mamba fold predictions. Regression with multi-horizon predictions."""
    all_preds = []
    all_labels = []

    # October folds
    files_oct = sorted(glob.glob('/home/jupiter/Lvl3Quant/output/mamba_v7_tiny_smart_v3/fold_*_oot_predictions.npz'))
    # March/April folds
    files_mar = sorted(glob.glob('/home/jupiter/Lvl3Quant/output/mamba_v7_tiny_smart_v3_mar_apr/fold_*_oot_predictions.npz'))

    all_files = files_oct + files_mar
    print(f"Mamba: Loading {len(all_files)} folds ({len(files_oct)} Oct + {len(files_mar)} Mar/Apr)...")

    for f in all_files:
        d = np.load(f, allow_pickle=True)
        # Use 10s horizon (column 2)
        all_preds.append(d['predictions'][:, 2])
        all_labels.append(d['labels'][:, 2])

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    print(f"  Total samples: {len(preds):,}")
    print(f"  Pred range: [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"  Label range: [{labels.min():.2f}, {labels.max():.2f}]")

    # Confidence = |prediction| (stronger prediction = higher confidence)
    confidence = np.abs(preds)

    return preds, labels, confidence


def compute_pnl_lgbm(probs, labels, confidence, cost_ticks):
    """
    LGBM PnL computation.

    Strategy: Go long when prob > 0.5, short when prob < 0.5.
    PnL per trade: direction * actual_move - cost

    Since we don't have actual tick moves (only binary labels), we estimate:
    - If label=1 (up) and we're long: we earn the average up-move
    - If label=0 (down) and we're short: we earn the average down-move

    But we need actual move sizes. For a binary classifier on smart_v3:
    - label=1 means price went up by at least the threshold
    - We approximate: correct prediction = +1 tick, wrong = -1 tick
    - This is conservative but standard for DA-based P&L estimation

    Better approach: use confidence-weighted PnL
    - PnL = (2*correct - 1) * 1 tick - cost
    - Where correct = (pred_dir matches label)
    """
    direction = np.where(probs > 0.5, 1.0, -1.0)  # +1 long, -1 short
    actual_dir = np.where(labels > 0.5, 1.0, -1.0)  # +1 up, -1 down

    # Correct prediction = directions match
    correct = (direction == actual_dir).astype(float)

    # PnL per trade in ticks: correct gets +1 tick, wrong gets -1 tick, minus cost
    # This assumes average move size of 1 tick (conservative)
    pnl_per_trade = (2 * correct - 1) * 1.0 - cost_ticks  # in ticks

    return pnl_per_trade


def compute_pnl_mamba(preds, labels, confidence, cost_ticks):
    """
    Mamba PnL computation.

    Strategy: Go long when pred > 0, short when pred < 0.
    PnL per trade = direction * actual_move - cost

    Labels are actual tick moves, so we can compute realistic PnL.
    """
    direction = np.sign(preds)
    # Filter out zero predictions (no signal)
    mask = direction != 0

    # PnL = direction * actual_move - cost (both in ticks)
    pnl_per_trade = np.full_like(preds, np.nan)
    pnl_per_trade[mask] = direction[mask] * labels[mask] - cost_ticks

    return pnl_per_trade


def compute_metrics(pnl, n_trades):
    """Compute trading metrics from PnL array."""
    if n_trades == 0 or len(pnl) == 0:
        return {'net_pnl_per_trade': 0, 'profit_factor': 0, 'sortino': 0,
                'win_rate': 0, 'n_trades': 0, 'total_pnl': 0}

    valid = pnl[~np.isnan(pnl)]
    if len(valid) == 0:
        return {'net_pnl_per_trade': 0, 'profit_factor': 0, 'sortino': 0,
                'win_rate': 0, 'n_trades': 0, 'total_pnl': 0}

    net_pnl = valid.mean()
    total_pnl = valid.sum()

    wins = valid[valid > 0]
    losses = valid[valid < 0]

    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    # Sortino: mean / downside_std
    downside = valid[valid < 0]
    if len(downside) > 1:
        downside_std = downside.std()
        sortino = net_pnl / downside_std if downside_std > 0 else float('inf')
    else:
        sortino = float('inf') if net_pnl > 0 else 0

    win_rate = (valid > 0).mean()

    return {
        'net_pnl_per_trade_ticks': round(float(net_pnl), 4),
        'net_pnl_per_trade_usd': round(float(net_pnl * TICK_VALUE), 2),
        'total_pnl_ticks': round(float(total_pnl), 2),
        'total_pnl_usd': round(float(total_pnl * TICK_VALUE), 2),
        'profit_factor': round(float(min(profit_factor, 999)), 3),
        'sortino': round(float(min(sortino, 999)), 3),
        'win_rate': round(float(win_rate), 4),
        'n_trades': int(len(valid)),
    }


def analyze_model(name, pnl_func, signal, labels, confidence):
    """Run full cost sensitivity analysis for a model."""
    print(f"\n{'='*80}")
    print(f"  {name} — Cost Sensitivity Analysis")
    print(f"{'='*80}")

    results = {}

    # Compute confidence percentiles for tier filtering
    conf_percentiles = {}
    for tier_name, (lo_pct, hi_pct) in CONFIDENCE_TIERS.items():
        threshold = np.percentile(confidence, lo_pct * 100)
        conf_percentiles[tier_name] = threshold

    for cost in COST_LEVELS:
        pnl_all = pnl_func(signal, labels, confidence, cost)
        results[str(cost)] = {}

        for tier_name, (lo_pct, hi_pct) in CONFIDENCE_TIERS.items():
            threshold = conf_percentiles[tier_name]
            mask = confidence >= threshold

            tier_pnl = pnl_all[mask]
            metrics = compute_metrics(tier_pnl, mask.sum())
            results[str(cost)][tier_name] = metrics

    # Print results table
    print(f"\n{'Cost':>6} | {'Tier':>7} | {'N Trades':>8} | {'PnL/Trade':>10} | {'$/Trade':>8} | {'PF':>6} | {'Sortino':>7} | {'WinRate':>7}")
    print('-' * 85)

    for cost in COST_LEVELS:
        for tier_name in CONFIDENCE_TIERS:
            m = results[str(cost)][tier_name]
            pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 999 else "inf"
            sort_str = f"{m['sortino']:.3f}" if m['sortino'] < 999 else "inf"
            print(f"{cost:>6.2f} | {tier_name:>7} | {m['n_trades']:>8,} | {m['net_pnl_per_trade_ticks']:>10.4f} | {m['net_pnl_per_trade_usd']:>8.2f} | {pf_str:>6} | {sort_str:>7} | {m['win_rate']:>7.1%}")
        print('-' * 85)

    # Find break-even cost for each tier
    print(f"\n--- Break-Even Cost Analysis ---")
    breakeven = {}
    for tier_name in CONFIDENCE_TIERS:
        # Find where PnL crosses zero by interpolating
        costs_arr = np.array(COST_LEVELS)
        pnls_arr = np.array([results[str(c)][tier_name]['net_pnl_per_trade_ticks'] for c in COST_LEVELS])

        be_cost = None
        if pnls_arr[0] > 0:
            # Find first crossover
            for i in range(len(pnls_arr) - 1):
                if pnls_arr[i] > 0 and pnls_arr[i+1] <= 0:
                    # Linear interpolation
                    frac = pnls_arr[i] / (pnls_arr[i] - pnls_arr[i+1])
                    be_cost = costs_arr[i] + frac * (costs_arr[i+1] - costs_arr[i])
                    break
            if be_cost is None and pnls_arr[-1] > 0:
                be_cost = float('inf')  # profitable at all tested costs
        else:
            be_cost = 0.0  # never profitable

        breakeven[tier_name] = round(float(be_cost), 3) if be_cost != float('inf') else "inf"
        status = f"{be_cost:.3f} ticks" if be_cost != float('inf') and be_cost > 0 else ("never profitable" if be_cost == 0 else "profitable at all tested costs")
        n = results[str(COST_LEVELS[0])][tier_name]['n_trades']
        print(f"  {tier_name:>7}: break-even at {status:>30} ({n:,} trades)")

    results['breakeven_cost'] = breakeven
    return results


def compute_breakeven_da(cost_ticks):
    """
    For binary classification (LGBM), what DA is needed to break even?
    PnL = (2*DA - 1) * avg_move - cost >= 0
    Assuming avg_move = 1 tick:
    DA >= (1 + cost) / 2
    """
    return (1 + cost_ticks) / 2


def main():
    # Load data
    lgbm_probs, lgbm_labels, lgbm_confidence = load_lgbm()
    mamba_preds, mamba_labels, mamba_confidence = load_mamba()

    # Run analysis
    lgbm_results = analyze_model(
        "LGBM smart_v3 (Binary Classifier, 1-tick avg move assumption)",
        compute_pnl_lgbm, lgbm_probs, lgbm_labels, lgbm_confidence
    )

    mamba_results = analyze_model(
        "Mamba v7 smart_v3 (Regression, actual tick moves)",
        compute_pnl_mamba, mamba_preds, mamba_labels, mamba_confidence
    )

    # Break-even DA thresholds for LGBM
    print(f"\n{'='*80}")
    print(f"  Break-Even Directional Accuracy by Cost Level (LGBM)")
    print(f"{'='*80}")
    da_thresholds = {}
    for cost in COST_LEVELS:
        be_da = compute_breakeven_da(cost)
        da_thresholds[str(cost)] = round(be_da, 4)
        print(f"  Cost={cost:.2f} ticks → Need DA >= {be_da:.2%}")

    # Actual DA by tier for LGBM
    print(f"\n--- Actual LGBM Directional Accuracy by Tier ---")
    lgbm_direction = np.where(lgbm_probs > 0.5, 1.0, -1.0)
    lgbm_actual_dir = np.where(lgbm_labels > 0.5, 1.0, -1.0)
    lgbm_correct = (lgbm_direction == lgbm_actual_dir)

    conf_percentiles = {name: np.percentile(lgbm_confidence, lo*100) for name, (lo, _) in CONFIDENCE_TIERS.items()}

    lgbm_da = {}
    for tier_name, (lo_pct, _) in CONFIDENCE_TIERS.items():
        mask = lgbm_confidence >= conf_percentiles[tier_name]
        da = lgbm_correct[mask].mean()
        n = mask.sum()
        lgbm_da[tier_name] = round(float(da), 4)

        # What's the max cost this DA supports?
        max_cost = 2 * da - 1  # from DA = (1+cost)/2
        print(f"  {tier_name:>7}: DA = {da:.4f} ({n:,} trades) → supports up to {max_cost:.3f} ticks cost")

    # Summary comparison
    print(f"\n{'='*80}")
    print(f"  SUMMARY: Profitability Thresholds")
    print(f"{'='*80}")
    print(f"\n  Model         | Top1% BE Cost | Top5% BE Cost | Top10% BE Cost | Top25% BE Cost")
    print(f"  {'-'*80}")

    l_be = lgbm_results['breakeven_cost']
    m_be = mamba_results['breakeven_cost']

    def fmt_be(v):
        if v == "inf":
            return ">2.0 ticks"
        elif v == 0:
            return "never"
        else:
            return f"{v:.3f} ticks"

    print(f"  LGBM          | {fmt_be(l_be['Top1%']):>13} | {fmt_be(l_be['Top5%']):>13} | {fmt_be(l_be['Top10%']):>14} | {fmt_be(l_be['Top25%']):>14}")
    print(f"  Mamba v7      | {fmt_be(m_be['Top1%']):>13} | {fmt_be(m_be['Top5%']):>13} | {fmt_be(m_be['Top10%']):>14} | {fmt_be(m_be['Top25%']):>14}")

    # Save to JSON
    output = {
        'analysis_date': '2026-04-25',
        'cost_levels_ticks': COST_LEVELS,
        'tick_value_usd': TICK_VALUE,
        'confidence_tiers': {k: {'min_percentile': v[0], 'max_percentile': v[1]} for k, v in CONFIDENCE_TIERS.items()},
        'lgbm_smart_v3': lgbm_results,
        'mamba_v7_smart_v3': mamba_results,
        'lgbm_breakeven_da_by_cost': da_thresholds,
        'lgbm_actual_da_by_tier': lgbm_da,
    }

    with open('/home/jupiter/Lvl3Quant/output/cost_sensitivity_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to /home/jupiter/Lvl3Quant/output/cost_sensitivity_analysis.json")


if __name__ == '__main__':
    main()
