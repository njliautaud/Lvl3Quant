#!/usr/bin/env python3
"""
3-Tier Analysis of Mamba v7 Tiny Smart_v3 Predictions
=====================================================
Tier 1 (Signal Quality): IC, DA, MagCorr at confidence tiers
Tier 2 (Tradeability): MFE/MAE, win/loss, long vs short, profit factor
Tier 3 (Profit Translation): Simulated P&L after costs, Sortino, profit factor
"""

import numpy as np
import json
import os
import glob
from collections import OrderedDict

# ============================================================
# Configuration
# ============================================================
OCT_DIR = "/home/jupiter/Lvl3Quant/output/mamba_v7_tiny_smart_v3"
MAR_DIR = "/home/jupiter/Lvl3Quant/output/mamba_v7_tiny_smart_v3_mar_apr"
OUTPUT_FILE = "/home/jupiter/Lvl3Quant/output/mamba_v7_smart_v3_3tier_analysis.json"

HORIZONS = ['1s', '5s', '10s']
HORIZON_IDX = {h: i for i, h in enumerate(HORIZONS)}

# Confidence tiers: top X% by absolute prediction magnitude
CONF_TIERS = ['All', '50%', '25%', '10%', '5%', '1%']
CONF_THRESHOLDS = {'All': 1.0, '50%': 0.50, '25%': 0.25, '10%': 0.10, '5%': 0.05, '1%': 0.01}

# Cost model: 2 ticks spread + slippage (in label units = ticks)
COST_PER_TRADE = 2.0  # 2 ticks round-trip spread + slippage


def load_fold_data(directory, fold_range=None):
    """Load and concatenate all fold prediction files from a directory."""
    all_preds = []
    all_labels = []
    fold_info = []

    files = sorted(glob.glob(os.path.join(directory, "fold_*_oot_predictions.npz")))

    for f in files:
        fold_num = int(os.path.basename(f).split('_')[1])
        if fold_range is not None and fold_num not in fold_range:
            continue

        d = np.load(f, allow_pickle=True)
        preds = d['predictions']
        labels = d['labels']

        oot_files = d['oot_files'] if 'oot_files' in d else ['unknown']
        date_str = 'unknown'
        for of in oot_files:
            # Extract date from filename like 20251013_mbo_events.npz
            bn = os.path.basename(str(of))
            if bn[:8].isdigit():
                date_str = bn[:8]

        fold_info.append({
            'fold': fold_num,
            'date': date_str,
            'n_samples': int(preds.shape[0]),
            'ic_1s': float(d['ic_1s']) if 'ic_1s' in d else None,
            'ic_5s': float(d['ic_5s']) if 'ic_5s' in d else None,
            'ic_10s': float(d['ic_10s']) if 'ic_10s' in d else None,
        })

        all_preds.append(preds)
        all_labels.append(labels)

    if not all_preds:
        return None, None, fold_info

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return preds, labels, fold_info


def compute_ic(preds, labels):
    """Pearson correlation (Information Coefficient)."""
    if len(preds) < 10:
        return 0.0
    # Remove NaN/inf
    mask = np.isfinite(preds) & np.isfinite(labels)
    p, l = preds[mask], labels[mask]
    if len(p) < 10 or np.std(p) < 1e-10 or np.std(l) < 1e-10:
        return 0.0
    return float(np.corrcoef(p, l)[0, 1])


def compute_rank_ic(preds, labels):
    """Spearman rank correlation."""
    if len(preds) < 10:
        return 0.0
    mask = np.isfinite(preds) & np.isfinite(labels)
    p, l = preds[mask], labels[mask]
    if len(p) < 10:
        return 0.0
    from scipy.stats import spearmanr
    rho, _ = spearmanr(p, l)
    return float(rho) if np.isfinite(rho) else 0.0


def compute_da(preds, labels):
    """Directional Accuracy - % of predictions with correct sign."""
    mask = (preds != 0) & (labels != 0) & np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return 0.5
    return float(np.mean(np.sign(preds[mask]) == np.sign(labels[mask])))


def compute_mag_corr(preds, labels):
    """Magnitude correlation - correlation of |pred| with |label|."""
    mask = np.isfinite(preds) & np.isfinite(labels)
    p, l = np.abs(preds[mask]), np.abs(labels[mask])
    if len(p) < 10 or np.std(p) < 1e-10 or np.std(l) < 1e-10:
        return 0.0
    return float(np.corrcoef(p, l)[0, 1])


def get_confidence_mask(preds, tier):
    """Get mask for top X% predictions by absolute magnitude."""
    frac = CONF_THRESHOLDS[tier]
    if frac >= 1.0:
        return np.ones(len(preds), dtype=bool)
    abs_preds = np.abs(preds)
    threshold = np.percentile(abs_preds, (1 - frac) * 100)
    return abs_preds >= threshold


def tier1_analysis(preds, labels, horizon_idx):
    """Tier 1: Signal Quality metrics at each confidence tier."""
    p = preds[:, horizon_idx]
    l = labels[:, horizon_idx]

    results = {}
    for tier in CONF_TIERS:
        mask = get_confidence_mask(p, tier)
        n = int(mask.sum())
        if n < 10:
            results[tier] = {'n': n, 'ic': None, 'da': None, 'mag_corr': None}
            continue

        pp, ll = p[mask], l[mask]
        results[tier] = {
            'n': n,
            'ic': round(compute_ic(pp, ll), 6),
            'rank_ic': round(compute_rank_ic(pp, ll), 6),
            'da': round(compute_da(pp, ll), 4),
            'mag_corr': round(compute_mag_corr(pp, ll), 6),
            'pred_std': round(float(np.std(pp)), 4),
            'label_std': round(float(np.std(ll)), 4),
        }
    return results


def tier2_analysis(preds, labels, horizon_idx):
    """Tier 2: Tradeability metrics at each confidence tier."""
    p = preds[:, horizon_idx]
    l = labels[:, horizon_idx]

    results = {}
    for tier in CONF_TIERS:
        mask = get_confidence_mask(p, tier)
        n = int(mask.sum())
        if n < 10:
            results[tier] = {'n': n}
            continue

        pp, ll = p[mask], l[mask]

        # Signed returns: positive if pred direction matches label direction
        signed_returns = np.sign(pp) * ll

        winners = signed_returns > 0
        losers = signed_returns < 0
        flat = signed_returns == 0

        n_winners = int(winners.sum())
        n_losers = int(losers.sum())
        n_flat = int(flat.sum())

        avg_winner = float(np.mean(signed_returns[winners])) if n_winners > 0 else 0.0
        avg_loser = float(np.mean(np.abs(signed_returns[losers]))) if n_losers > 0 else 0.0

        win_rate = n_winners / max(n_winners + n_losers, 1)

        # MFE (Max Favorable Excursion) = average of winning trades
        # MAE (Max Adverse Excursion) = average of losing trades
        mfe = avg_winner
        mae = avg_loser

        # Profit factor = gross profits / gross losses
        gross_profit = float(np.sum(signed_returns[winners])) if n_winners > 0 else 0.0
        gross_loss = float(np.sum(np.abs(signed_returns[losers]))) if n_losers > 0 else 0.001
        profit_factor = gross_profit / gross_loss

        # Long vs Short breakdown
        long_mask = pp > 0
        short_mask = pp < 0

        long_pnl = float(np.mean(ll[long_mask])) if long_mask.sum() > 0 else 0.0
        short_pnl = float(np.mean(-ll[short_mask])) if short_mask.sum() > 0 else 0.0

        long_da = compute_da(pp[long_mask], ll[long_mask]) if long_mask.sum() > 10 else None
        short_da = compute_da(pp[short_mask], ll[short_mask]) if short_mask.sum() > 10 else None

        results[tier] = {
            'n': n,
            'win_rate': round(win_rate, 4),
            'n_winners': n_winners,
            'n_losers': n_losers,
            'n_flat': n_flat,
            'avg_winner_ticks': round(mfe, 3),
            'avg_loser_ticks': round(mae, 3),
            'mfe_mae_ratio': round(mfe / max(mae, 0.001), 3),
            'profit_factor': round(profit_factor, 3),
            'avg_signed_return': round(float(np.mean(signed_returns)), 4),
            'long_avg_pnl': round(long_pnl, 4),
            'short_avg_pnl': round(short_pnl, 4),
            'long_da': round(long_da, 4) if long_da is not None else None,
            'short_da': round(short_da, 4) if short_da is not None else None,
            'n_long': int(long_mask.sum()),
            'n_short': int(short_mask.sum()),
        }
    return results


def tier3_analysis(preds, labels, horizon_idx, cost=COST_PER_TRADE):
    """Tier 3: Profit Translation with realistic costs."""
    p = preds[:, horizon_idx]
    l = labels[:, horizon_idx]

    results = {}
    for tier in CONF_TIERS:
        mask = get_confidence_mask(p, tier)
        n = int(mask.sum())
        if n < 10:
            results[tier] = {'n': n}
            continue

        pp, ll = p[mask], l[mask]

        # PnL per trade = direction * label - cost
        # If pred > 0, go long: pnl = label - cost
        # If pred < 0, go short: pnl = -label - cost
        raw_pnl = np.sign(pp) * ll  # ticks gained from direction
        net_pnl = raw_pnl - cost  # subtract round-trip cost

        total_pnl = float(np.sum(net_pnl))
        avg_pnl = float(np.mean(net_pnl))

        # Sortino ratio (annualized, assuming ~25000 trades/day, 252 days)
        # Daily Sortino = mean(daily_pnl) / downside_std(daily_pnl)
        # We'll compute per-trade Sortino and scale
        downside = net_pnl[net_pnl < 0]
        downside_std = float(np.std(downside)) if len(downside) > 1 else 1.0
        per_trade_sortino = avg_pnl / max(downside_std, 0.001)

        # Scale to daily: multiply by sqrt(trades_per_day)
        # Approximate trades per day from the data
        trades_per_day = n / max(1, 8)  # rough: 8 OOT days in March
        daily_sortino = per_trade_sortino * np.sqrt(trades_per_day)
        annual_sortino = daily_sortino * np.sqrt(252)

        # Winners/losers after costs
        winners_after = net_pnl > 0
        losers_after = net_pnl <= 0

        win_rate_after = float(np.mean(winners_after))

        gross_profit_after = float(np.sum(net_pnl[winners_after])) if winners_after.sum() > 0 else 0.0
        gross_loss_after = float(np.sum(np.abs(net_pnl[losers_after]))) if losers_after.sum() > 0 else 0.001
        pf_after = gross_profit_after / gross_loss_after

        # Cumulative PnL stats
        cum_pnl = np.cumsum(net_pnl)
        max_drawdown = float(np.max(np.maximum.accumulate(cum_pnl) - cum_pnl))

        results[tier] = {
            'n_trades': n,
            'cost_per_trade_ticks': cost,
            'total_pnl_ticks': round(total_pnl, 1),
            'avg_pnl_per_trade': round(avg_pnl, 4),
            'win_rate_after_costs': round(win_rate_after, 4),
            'profit_factor_after_costs': round(pf_after, 3),
            'per_trade_sortino': round(per_trade_sortino, 4),
            'annualized_sortino': round(annual_sortino, 2),
            'max_drawdown_ticks': round(max_drawdown, 1),
            'total_pnl_div_maxdd': round(total_pnl / max(max_drawdown, 1), 3),
            'avg_winner_after_costs': round(float(np.mean(net_pnl[winners_after])), 3) if winners_after.sum() > 0 else 0,
            'avg_loser_after_costs': round(float(np.mean(net_pnl[losers_after])), 3) if losers_after.sum() > 0 else 0,
        }
    return results


def per_fold_ic_table(fold_info):
    """Create per-fold IC summary."""
    rows = []
    for fi in fold_info:
        rows.append({
            'fold': fi['fold'],
            'date': fi['date'],
            'n_samples': fi['n_samples'],
            'ic_1s': round(fi['ic_1s'], 4) if fi['ic_1s'] is not None else None,
            'ic_5s': round(fi['ic_5s'], 4) if fi['ic_5s'] is not None else None,
            'ic_10s': round(fi['ic_10s'], 4) if fi['ic_10s'] is not None else None,
        })
    return rows


def analyze_period(preds, labels, fold_info, period_name):
    """Run full 3-tier analysis for a period."""
    if preds is None:
        return {'error': 'No data'}

    result = {
        'period': period_name,
        'total_samples': int(preds.shape[0]),
        'n_folds': len(fold_info),
        'per_fold_ic': per_fold_ic_table(fold_info),
    }

    # Concat IC across all samples
    for h in HORIZONS:
        hi = HORIZON_IDX[h]
        ic = compute_ic(preds[:, hi], labels[:, hi])
        result[f'concat_ic_{h}'] = round(ic, 6)

    # Per-horizon 3-tier analysis
    for h in HORIZONS:
        hi = HORIZON_IDX[h]
        result[f'tier1_{h}'] = tier1_analysis(preds, labels, hi)
        result[f'tier2_{h}'] = tier2_analysis(preds, labels, hi)
        result[f'tier3_{h}'] = tier3_analysis(preds, labels, hi)

    return result


def format_summary(results, period_name):
    """Print a human-readable summary."""
    r = results
    print(f"\n{'='*70}")
    print(f"  {period_name}")
    print(f"  Samples: {r['total_samples']:,}  |  Folds: {r['n_folds']}")
    print(f"{'='*70}")

    # Concat IC
    print(f"\n  Concat IC:  1s={r['concat_ic_1s']:.4f}  5s={r['concat_ic_5s']:.4f}  10s={r['concat_ic_10s']:.4f}")

    # Per-fold IC table
    print(f"\n  Per-Fold IC:")
    print(f"  {'Fold':>4} {'Date':>10} {'Samples':>8} {'IC_1s':>8} {'IC_5s':>8} {'IC_10s':>8}")
    for fi in r['per_fold_ic']:
        ic1 = f"{fi['ic_1s']:.4f}" if fi['ic_1s'] is not None else "N/A"
        ic5 = f"{fi['ic_5s']:.4f}" if fi['ic_5s'] is not None else "N/A"
        ic10 = f"{fi['ic_10s']:.4f}" if fi['ic_10s'] is not None else "N/A"
        print(f"  {fi['fold']:>4} {fi['date']:>10} {fi['n_samples']:>8} {ic1:>8} {ic5:>8} {ic10:>8}")

    for h in HORIZONS:
        print(f"\n  --- Horizon: {h} ---")

        # Tier 1
        print(f"  TIER 1 (Signal Quality):")
        print(f"  {'Tier':>6} {'N':>8} {'IC':>8} {'RankIC':>8} {'DA':>7} {'MagCorr':>8}")
        t1 = r[f'tier1_{h}']
        for tier in CONF_TIERS:
            d = t1[tier]
            if d.get('ic') is None:
                print(f"  {tier:>6} {d['n']:>8} {'N/A':>8} {'N/A':>8} {'N/A':>7} {'N/A':>8}")
            else:
                print(f"  {tier:>6} {d['n']:>8} {d['ic']:>8.4f} {d['rank_ic']:>8.4f} {d['da']:>7.1%} {d['mag_corr']:>8.4f}")

        # Tier 2
        print(f"\n  TIER 2 (Tradeability):")
        print(f"  {'Tier':>6} {'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>8} {'MFE/MAE':>8} {'PF':>7} {'LongDA':>7} {'ShortDA':>7}")
        t2 = r[f'tier2_{h}']
        for tier in CONF_TIERS:
            d = t2[tier]
            if 'win_rate' not in d:
                continue
            lda = f"{d['long_da']:.1%}" if d['long_da'] is not None else "N/A"
            sda = f"{d['short_da']:.1%}" if d['short_da'] is not None else "N/A"
            print(f"  {tier:>6} {d['win_rate']:>7.1%} {d['avg_winner_ticks']:>8.2f} {d['avg_loser_ticks']:>8.2f} {d['mfe_mae_ratio']:>8.2f} {d['profit_factor']:>7.2f} {lda:>7} {sda:>7}")

        # Tier 3
        print(f"\n  TIER 3 (Profit Translation, cost={COST_PER_TRADE} ticks):")
        print(f"  {'Tier':>6} {'Trades':>7} {'TotalPnL':>9} {'AvgPnL':>8} {'WR_net':>7} {'PF_net':>7} {'Sortino':>8} {'MaxDD':>8} {'PnL/DD':>7}")
        t3 = r[f'tier3_{h}']
        for tier in CONF_TIERS:
            d = t3[tier]
            if 'total_pnl_ticks' not in d:
                continue
            print(f"  {tier:>6} {d['n_trades']:>7} {d['total_pnl_ticks']:>9.0f} {d['avg_pnl_per_trade']:>8.3f} {d['win_rate_after_costs']:>6.1%} {d['profit_factor_after_costs']:>7.2f} {d['annualized_sortino']:>8.1f} {d['max_drawdown_ticks']:>8.0f} {d['total_pnl_div_maxdd']:>7.2f}")


def main():
    print("Loading data...")

    # October OOT: original run, folds 0-9
    oct_preds, oct_labels, oct_folds = load_fold_data(OCT_DIR, fold_range=range(0, 10))
    print(f"  October: {oct_preds.shape[0]:,} samples from {len(oct_folds)} folds")

    # March OOT: mar_apr run, folds 5-13
    mar_preds, mar_labels, mar_folds = load_fold_data(MAR_DIR, fold_range=range(5, 14))
    print(f"  March:   {mar_preds.shape[0]:,} samples from {len(mar_folds)} folds")

    # Combined
    combined_preds = np.concatenate([oct_preds, mar_preds], axis=0)
    combined_labels = np.concatenate([oct_labels, mar_labels], axis=0)
    combined_folds = oct_folds + mar_folds
    print(f"  Combined: {combined_preds.shape[0]:,} samples")

    # Run analyses
    oct_results = analyze_period(oct_preds, oct_labels, oct_folds, "October 2025 OOT (folds 0-9)")
    mar_results = analyze_period(mar_preds, mar_labels, mar_folds, "March 2026 OOT (folds 5-13)")
    combined_results = analyze_period(combined_preds, combined_labels, combined_folds, "Combined (Oct+Mar)")

    # Temporal comparison
    temporal = {
        'october_concat_ic': {h: oct_results[f'concat_ic_{h}'] for h in HORIZONS},
        'march_concat_ic': {h: mar_results[f'concat_ic_{h}'] for h in HORIZONS},
        'ic_change': {h: round(mar_results[f'concat_ic_{h}'] - oct_results[f'concat_ic_{h}'], 6) for h in HORIZONS},
        'ic_change_pct': {h: round((mar_results[f'concat_ic_{h}'] - oct_results[f'concat_ic_{h}']) / max(abs(oct_results[f'concat_ic_{h}']), 0.0001) * 100, 1) for h in HORIZONS},
    }

    # Add Tier 3 comparison at 10% confidence for 10s horizon
    for tier in ['All', '10%', '5%']:
        for h in ['10s']:
            oct_t3 = oct_results.get(f'tier3_{h}', {}).get(tier, {})
            mar_t3 = mar_results.get(f'tier3_{h}', {}).get(tier, {})
            if oct_t3 and mar_t3 and 'avg_pnl_per_trade' in oct_t3:
                temporal[f'pnl_comparison_{h}_{tier}'] = {
                    'oct_avg_pnl': oct_t3.get('avg_pnl_per_trade'),
                    'mar_avg_pnl': mar_t3.get('avg_pnl_per_trade'),
                    'oct_sortino': oct_t3.get('annualized_sortino'),
                    'mar_sortino': mar_t3.get('annualized_sortino'),
                    'oct_pf': oct_t3.get('profit_factor_after_costs'),
                    'mar_pf': mar_t3.get('profit_factor_after_costs'),
                }

    # Print summaries
    format_summary(oct_results, "OCTOBER 2025 OOT")
    format_summary(mar_results, "MARCH 2026 OOT")
    format_summary(combined_results, "COMBINED (OCT + MAR)")

    # Temporal comparison summary
    print(f"\n{'='*70}")
    print(f"  TEMPORAL COMPARISON: OCTOBER vs MARCH")
    print(f"{'='*70}")
    print(f"\n  Concat IC:")
    print(f"  {'Horizon':>8} {'October':>10} {'March':>10} {'Change':>10} {'Change%':>10}")
    for h in HORIZONS:
        print(f"  {h:>8} {temporal['october_concat_ic'][h]:>10.4f} {temporal['march_concat_ic'][h]:>10.4f} {temporal['ic_change'][h]:>+10.4f} {temporal['ic_change_pct'][h]:>+9.1f}%")

    # Save to JSON
    output = {
        'model': 'mamba_v7_tiny_smart_v3',
        'analysis_date': '2026-04-25',
        'cost_model': f'{COST_PER_TRADE} ticks (spread + slippage)',
        'october_oot': oct_results,
        'march_oot': mar_results,
        'combined': combined_results,
        'temporal_comparison': temporal,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
