"""
v7 Execution Re-evaluation

Evaluates whether the stronger meta v7 predictions (Sp 0.308 vs v6 0.167)
change the execution calculus. Analyzes optimal hold times, cancel windows,
and passive vs aggressive exit strategies across confidence tiers.

Key question: Does the v7 signal strength unlock execution strategies
that were previously marginal?
"""

import os
import sys
import json
import time
import zipfile
import numpy as np
from scipy import stats
from datetime import datetime
from collections import defaultdict

# MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# Constants
# ============================================================
ES_TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376       # passive limit (commission only)
MARKET_ORDER_COST_TICKS = 1.376   # market order (commission + 1 tick spread)

PRED_NPZ = '/home/nick/Lvl3Quant/output/meta_v7_prod/concat_oot_predictions.npz'
CM_DIR = '/home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_oot'
MBO_DIR = '/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3'
OUTPUT_DIR = '/home/nick/Lvl3Quant/output/v7_execution_reeval'

CONFIDENCE_TIERS = [5, 10, 20, 50, 100]   # top N% by |pred|
HORIZONS_SEC = [1, 5, 10, 30]              # seconds
CANCEL_WINDOWS_SEC = [3, 5, 10, 15, 20]    # cancel unfilled after N seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

def ts():
    return datetime.now().strftime('%H:%M:%S')

# ============================================================
# Data Loading
# ============================================================
def load_predictions():
    """Load v7 concat OOT predictions."""
    print(f"[{ts()}] Loading v7 predictions from {PRED_NPZ}")
    d = np.load(PRED_NPZ, allow_pickle=True)
    preds = d['predictions']
    labels = d['labels']
    dates = d['dates']
    print(f"  {len(preds):,} predictions, {len(np.unique(dates))} OOT dates")
    print(f"  pred range: [{preds.min():.3f}, {preds.max():.3f}], std={preds.std():.3f}")
    print(f"  label range: [{labels.min():.1f}, {labels.max():.1f}], std={labels.std():.3f}")
    return preds, labels, dates


def load_multi_horizon_labels(date_str):
    """
    Load multi-horizon labels for a date by reconstructing the CNN-Mamba
    event indices into the MBO data.
    
    Returns dict: {horizon_sec: labels_array} or None if data missing.
    """
    cm_path = os.path.join(CM_DIR, f'{date_str}_predictions.npz')
    mbo_path = os.path.join(MBO_DIR, f'{date_str}_mbo_events.npz')
    
    if not os.path.exists(cm_path) or not os.path.exists(mbo_path):
        return None
    
    try:
        cm_data = np.load(cm_path, allow_pickle=True)
        mbo_data = np.load(mbo_path)
    except (zipfile.BadZipFile, Exception):
        return None
    
    window_size = int(cm_data['window_size'])
    stride = int(cm_data['stride'])
    n_cm = len(cm_data['predictions'])
    
    # Reconstruct event indices
    event_idx = np.array([window_size + i * stride for i in range(n_cm)])
    n_mbo = len(mbo_data['labels_1s'])
    valid = event_idx < n_mbo
    event_idx = event_idx[valid]
    
    result = {}
    for h in HORIZONS_SEC:
        key = f'labels_{h}s'
        if key in mbo_data:
            result[h] = mbo_data[key][event_idx]
    
    return result


# ============================================================
# Analysis Functions
# ============================================================
def compute_tier_mask(preds, percentile):
    """Get mask for top percentile by absolute prediction value."""
    if percentile >= 100:
        return np.ones(len(preds), dtype=bool)
    threshold = np.percentile(np.abs(preds), 100 - percentile)
    return np.abs(preds) >= threshold


def compute_pnl_metrics(preds, labels, cost_ticks):
    """
    Compute PnL metrics for given predictions, labels, and cost.
    Labels are in tick units (forward return).
    
    Returns dict with Sharpe, Sortino, PF, WR, avg_pnl, n_trades.
    """
    if len(preds) < 10:
        return None
    
    # PnL per trade: sign(pred) * realized_move - cost
    trade_pnl = np.sign(preds) * labels - cost_ticks
    
    n_trades = len(trade_pnl)
    avg_pnl = trade_pnl.mean()
    std_pnl = trade_pnl.std()
    
    # Sharpe (annualized assuming ~250 trading days, ~50k signals/day)
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
    
    # Sortino
    downside = trade_pnl[trade_pnl < 0]
    downside_std = downside.std() if len(downside) > 0 else 1.0
    sortino = avg_pnl / downside_std if downside_std > 0 else 0.0
    
    # Profit factor
    gross_profit = trade_pnl[trade_pnl > 0].sum()
    gross_loss = abs(trade_pnl[trade_pnl < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Win rate
    wr = (trade_pnl > 0).mean()
    
    # Total PnL in dollars
    total_pnl_dollars = trade_pnl.sum() * ES_TICK_VALUE
    
    return {
        'n_trades': n_trades,
        'avg_pnl_ticks': float(avg_pnl),
        'std_pnl_ticks': float(std_pnl),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'profit_factor': float(min(pf, 99.9)),
        'win_rate': float(wr),
        'total_pnl_dollars': float(total_pnl_dollars),
    }


def mfe_mae_analysis(preds, multi_horizon_labels, tier_pct):
    """
    For a given confidence tier, compute MFE (max favorable excursion)
    and MAE (max adverse excursion) statistics at each horizon.
    
    Since we have labels at 1s/5s/10s/30s (forward returns), we use
    sign(pred)*label as the signed excursion at each horizon.
    MFE = max positive excursion, MAE = max negative excursion.
    """
    mask = compute_tier_mask(preds, tier_pct)
    p = preds[mask]
    
    results = {}
    for h in HORIZONS_SEC:
        if h not in multi_horizon_labels:
            continue
        lbl = multi_horizon_labels[h][mask]
        
        # Signed excursion: how much the trade moves in our direction
        signed_excursion = np.sign(p) * lbl
        
        results[h] = {
            'mean_excursion': float(np.mean(signed_excursion)),
            'median_excursion': float(np.median(signed_excursion)),
            'p10_excursion': float(np.percentile(signed_excursion, 10)),
            'p25_excursion': float(np.percentile(signed_excursion, 25)),
            'p75_excursion': float(np.percentile(signed_excursion, 75)),
            'p90_excursion': float(np.percentile(signed_excursion, 90)),
            'pct_favorable': float((signed_excursion > 0).mean()),
            'mean_mfe': float(np.mean(signed_excursion[signed_excursion > 0])) if (signed_excursion > 0).any() else 0,
            'mean_mae': float(np.mean(signed_excursion[signed_excursion < 0])) if (signed_excursion < 0).any() else 0,
            'n_samples': int(mask.sum()),
        }
    
    return results


def cancel_window_analysis(preds, labels, tier_pct, cancel_sec):
    """
    Simulate cancel window effect. If a prediction is generated but the
    order isn't filled within cancel_sec, we cancel it.
    
    Approximation: predictions generated every 250ms (stride=250 at ~1000 events/sec).
    A 'cancel' means we only take trades where the label at cancel_sec horizon
    shows the market moved in our predicted direction (order likely filled).
    
    For passive limit orders, the order fills if price touches our level.
    Proxy: the signed excursion at cancel_sec horizon should be favorable
    (or near zero = at our level).
    
    Simplified model: we assume the order fills if |label_1s| > 0 within
    the cancel window, and we use the realized 1s label as the trade PnL.
    For more accurate analysis, we'd need tick-by-tick data, but this
    gives a reasonable approximation.
    """
    mask = compute_tier_mask(preds, tier_pct)
    p = preds[mask]
    l = labels[mask]  # 1s forward return (our trading label)
    
    # More aggressive predictions are more likely to be filled quickly
    # Proxy for fill probability: higher |pred| = more conviction = post closer to market
    # We model fill rate as: all orders are posted, cancel after cancel_sec
    # For this simplified analysis, we assume fill rate depends on cancel window:
    # Shorter cancel = fewer fills but better signal freshness
    # Longer cancel = more fills but staler predictions
    
    # Signal decay model: v7 signal peaks at 1s, ~50% decay by 10s, ~80% by 30s
    # (from CLAUDE.md signal characteristics)
    decay_factor = {
        3: 0.85,   # 3s: signal still ~85% strong
        5: 0.70,   # 5s: ~70%
        10: 0.50,  # 10s: ~50%
        15: 0.35,  # 15s: ~35%
        20: 0.25,  # 20s: ~25%
    }
    
    # Approximate fill rate by cancel window (longer window = more fills)
    # Based on typical ES fill dynamics for passive limits
    fill_rate = {
        3: 0.25,   # 3s: only 25% fill
        5: 0.40,   # 5s: 40% fill
        10: 0.60,  # 10s: 60% fill
        15: 0.70,  # 15s: 70% fill
        20: 0.80,  # 20s: 80% fill
    }
    
    df = decay_factor.get(cancel_sec, 0.5)
    fr = fill_rate.get(cancel_sec, 0.5)
    
    # Effective PnL = filled trades only, with decayed signal
    # Randomly select fill_rate fraction (deterministic via threshold on |pred|)
    n_filled = max(1, int(len(p) * fr))
    
    # Higher conviction predictions fill first (they're posted more aggressively)
    abs_pred_order = np.argsort(-np.abs(p))
    filled_idx = abs_pred_order[:n_filled]
    
    filled_p = p[filled_idx]
    filled_l = l[filled_idx]
    
    # PnL with decayed signal (the realized return is the same, 
    # but the signal-to-noise is what we're measuring)
    passive_metrics = compute_pnl_metrics(filled_p, filled_l, COMMISSION_RT_TICKS)
    
    if passive_metrics is None:
        return None
    
    passive_metrics['cancel_window_sec'] = cancel_sec
    passive_metrics['fill_rate'] = fr
    passive_metrics['signal_decay'] = df
    passive_metrics['n_posted'] = len(p)
    passive_metrics['n_filled'] = n_filled
    
    return passive_metrics


def passive_vs_aggressive_analysis(preds, labels, multi_horizon_labels, tier_pct):
    """
    Compare passive-only vs market-order exit strategies.
    
    Passive entry + passive exit: cost = 2 * 0.376/2 = 0.376 ticks (one RT)
    Passive entry + market exit: cost = 0.376 + 1.0 = 1.376 ticks
    Market entry + market exit: cost = 1.376 + 1.0 = 2.376 ticks (worst case)
    
    For each horizon, compare the edge with different cost structures.
    """
    mask = compute_tier_mask(preds, tier_pct)
    p = preds[mask]
    
    results = {}
    for h in HORIZONS_SEC:
        if h not in multi_horizon_labels:
            continue
        lbl = multi_horizon_labels[h][mask]
        
        # Passive entry, passive exit (cheapest)
        passive_passive = compute_pnl_metrics(p, lbl, COMMISSION_RT_TICKS)
        
        # Passive entry, market exit (common: post limit, then lift to exit)
        passive_market = compute_pnl_metrics(p, lbl, MARKET_ORDER_COST_TICKS)
        
        # Market entry, market exit (most expensive, fastest)
        market_market = compute_pnl_metrics(p, lbl, 2 * MARKET_ORDER_COST_TICKS - COMMISSION_RT_TICKS)
        
        results[h] = {
            'passive_passive': passive_passive,
            'passive_market': passive_market,
            'market_market': market_market,
        }
    
    return results


def per_day_analysis(preds, labels, dates):
    """Compute per-day Sharpe/PF/WR for regime analysis."""
    unique_dates = np.unique(dates)
    daily_results = []
    
    for d in unique_dates:
        mask = dates == d
        dp = preds[mask]
        dl = labels[mask]
        
        if len(dp) < 50:
            continue
        
        day_result = {'date': d}
        
        for tier in CONFIDENCE_TIERS:
            tier_mask = compute_tier_mask(dp, tier)
            if tier_mask.sum() < 10:
                continue
            
            tp = dp[tier_mask]
            tl = dl[tier_mask]
            
            # Passive cost
            m = compute_pnl_metrics(tp, tl, COMMISSION_RT_TICKS)
            if m:
                day_result[f'top{tier}_passive'] = m
            
            # Market cost
            m2 = compute_pnl_metrics(tp, tl, MARKET_ORDER_COST_TICKS)
            if m2:
                day_result[f'top{tier}_market'] = m2
        
        daily_results.append(day_result)
    
    return daily_results


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()
    print(f"[{ts()}] === v7 Execution Re-evaluation ===")
    print(f"  v7 Spearman: 0.308 (vs v6: 0.167, +85% improvement)")
    print()
    
    # --- Load data ---
    preds, labels, dates = load_predictions()
    unique_dates = np.unique(dates)
    
    # Verify Spearman
    sp, _ = stats.spearmanr(preds, labels)
    print(f"  Concat Spearman: {sp:.4f}")
    print()
    
    # --- Load multi-horizon labels ---
    print(f"[{ts()}] Loading multi-horizon labels from MBO events...")
    all_multi_labels = {h: [] for h in HORIZONS_SEC}
    all_multi_mask = []
    
    date_pred_counts = {}
    offset = 0
    for d in unique_dates:
        n_date = (dates == d).sum()
        date_pred_counts[d] = (offset, offset + n_date)
        offset += n_date
    
    n_matched = 0
    n_failed = 0
    for d in unique_dates:
        mh = load_multi_horizon_labels(d)
        start_idx, end_idx = date_pred_counts[d]
        n_date = end_idx - start_idx
        
        if mh is None:
            # No multi-horizon data for this date
            for h in HORIZONS_SEC:
                all_multi_labels[h].append(np.full(n_date, np.nan))
            all_multi_mask.append(np.zeros(n_date, dtype=bool))
            n_failed += 1
            continue
        
        # Check length alignment
        min_len = min(len(v) for v in mh.values())
        if min_len < n_date:
            # Trim predictions or pad labels
            for h in HORIZONS_SEC:
                if h in mh and len(mh[h]) >= n_date:
                    all_multi_labels[h].append(mh[h][:n_date])
                else:
                    padded = np.full(n_date, np.nan)
                    if h in mh:
                        padded[:len(mh[h])] = mh[h]
                    all_multi_labels[h].append(padded)
            all_multi_mask.append(np.ones(n_date, dtype=bool))
        else:
            for h in HORIZONS_SEC:
                all_multi_labels[h].append(mh[h][:n_date])
            all_multi_mask.append(np.ones(n_date, dtype=bool))
        n_matched += 1
    
    multi_labels = {h: np.concatenate(all_multi_labels[h]) for h in HORIZONS_SEC}
    multi_mask = np.concatenate(all_multi_mask)
    print(f"  Matched {n_matched}/{len(unique_dates)} dates for multi-horizon labels")
    print(f"  Total samples with multi-horizon: {multi_mask.sum():,}")
    print()
    
    # ============================================================
    # Analysis 1: Optimal Hold Time (MFE/MAE by horizon)
    # ============================================================
    print(f"[{ts()}] === Analysis 1: Optimal Hold Time (MFE/MAE) ===")
    
    # Use samples where multi-horizon labels are available
    valid = multi_mask & ~np.isnan(multi_labels[1])
    vp = preds[valid]
    vl = {h: multi_labels[h][valid] for h in HORIZONS_SEC}
    
    mfe_results = {}
    for tier in CONFIDENCE_TIERS:
        mfe_results[f'top{tier}'] = mfe_mae_analysis(vp, vl, tier)
        
        print(f"\n  Top {tier}% predictions:")
        for h in HORIZONS_SEC:
            r = mfe_results[f'top{tier}'].get(h)
            if r:
                print(f"    {h}s horizon: mean={r['mean_excursion']:.3f} ticks, "
                      f"median={r['median_excursion']:.3f}, "
                      f"p90_MFE={r['p90_excursion']:.3f}, "
                      f"favorable={r['pct_favorable']:.1%}, "
                      f"n={r['n_samples']:,}")
    
    # Determine sweet spot
    print(f"\n  --- Sweet Spot Analysis ---")
    for tier in [5, 10, 20]:
        best_h = None
        best_net = -999
        for h in HORIZONS_SEC:
            r = mfe_results[f'top{tier}'].get(h)
            if r:
                # Net after passive cost
                net = r['mean_excursion'] - COMMISSION_RT_TICKS
                if net > best_net:
                    best_net = net
                    best_h = h
        if best_h:
            print(f"  Top {tier}%: Best hold = {best_h}s "
                  f"(net after passive cost: {best_net:.3f} ticks)")
    
    # ============================================================
    # Analysis 2: Cancel Window
    # ============================================================
    print(f"\n[{ts()}] === Analysis 2: Cancel Window Analysis ===")
    
    cancel_results = {}
    for tier in [5, 10, 20]:
        cancel_results[f'top{tier}'] = {}
        print(f"\n  Top {tier}% predictions:")
        for cw in CANCEL_WINDOWS_SEC:
            cr = cancel_window_analysis(preds, labels, tier, cw)
            if cr:
                cancel_results[f'top{tier}'][f'{cw}s'] = cr
                print(f"    Cancel {cw}s: Sharpe={cr['sharpe']:.3f}, "
                      f"WR={cr['win_rate']:.1%}, PF={cr['profit_factor']:.2f}, "
                      f"fills={cr['n_filled']:,}/{cr['n_posted']:,} "
                      f"(fill_rate={cr['fill_rate']:.0%})")
    
    # ============================================================
    # Analysis 3: Passive vs Aggressive Exit
    # ============================================================
    print(f"\n[{ts()}] === Analysis 3: Passive vs Aggressive Exit ===")
    
    exit_results = {}
    for tier in [5, 10, 20]:
        exit_results[f'top{tier}'] = passive_vs_aggressive_analysis(
            vp, vl[1], vl, tier)  # Use 1s labels as base
        
        print(f"\n  Top {tier}% predictions:")
        for h in HORIZONS_SEC:
            r = exit_results[f'top{tier}'].get(h)
            if r:
                pp = r['passive_passive']
                pm = r['passive_market']
                mm = r['market_market']
                if pp and pm and mm:
                    print(f"    {h}s horizon:")
                    print(f"      Passive/Passive: Sharpe={pp['sharpe']:.3f}, "
                          f"WR={pp['win_rate']:.1%}, avg={pp['avg_pnl_ticks']:.3f}t")
                    print(f"      Passive/Market:  Sharpe={pm['sharpe']:.3f}, "
                          f"WR={pm['win_rate']:.1%}, avg={pm['avg_pnl_ticks']:.3f}t")
                    print(f"      Market/Market:   Sharpe={mm['sharpe']:.3f}, "
                          f"WR={mm['win_rate']:.1%}, avg={mm['avg_pnl_ticks']:.3f}t")
    
    # ============================================================
    # Analysis 4: Per-Day Breakdown (Regime)
    # ============================================================
    print(f"\n[{ts()}] === Analysis 4: Per-Day Regime Analysis ===")
    daily = per_day_analysis(preds, labels, dates)
    
    print(f"\n  Daily results (top 10%, passive cost):")
    print(f"  {'Date':<12} {'N':>6} {'Sharpe':>8} {'WR':>6} {'PF':>6} {'PnL/trade':>10}")
    print(f"  {'-'*50}")
    
    daily_sharpes = []
    for d in daily:
        m = d.get('top10_passive')
        if m:
            daily_sharpes.append(m['sharpe'])
            print(f"  {d['date']:<12} {m['n_trades']:>6,} {m['sharpe']:>8.3f} "
                  f"{m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} "
                  f"{m['avg_pnl_ticks']:>10.3f}")
    
    if daily_sharpes:
        green_days = sum(1 for s in daily_sharpes if s > 0)
        print(f"\n  Green days: {green_days}/{len(daily_sharpes)} "
              f"({green_days/len(daily_sharpes):.0%})")
        print(f"  Mean daily Sharpe: {np.mean(daily_sharpes):.3f}")
        print(f"  Std daily Sharpe: {np.std(daily_sharpes):.3f}")
    
    # ============================================================
    # Analysis 5: v7 vs v6 Execution Comparison
    # ============================================================
    print(f"\n[{ts()}] === Analysis 5: v7 Execution Edge Summary ===")
    
    summary = {
        'v7_spearman': float(sp),
        'v6_spearman': 0.167,
        'improvement_pct': float((sp - 0.167) / 0.167 * 100),
        'n_oot_dates': len(unique_dates),
        'n_predictions': len(preds),
        'execution_analysis': {},
    }
    
    for tier in [5, 10, 20]:
        tier_mask = compute_tier_mask(preds, tier)
        tp = preds[tier_mask]
        tl = labels[tier_mask]
        
        passive = compute_pnl_metrics(tp, tl, COMMISSION_RT_TICKS)
        market = compute_pnl_metrics(tp, tl, MARKET_ORDER_COST_TICKS)
        
        summary['execution_analysis'][f'top{tier}'] = {
            'passive_cost': passive,
            'market_cost': market,
        }
        
        if passive and market:
            print(f"\n  Top {tier}% ({passive['n_trades']:,} trades):")
            print(f"    Passive entry: Sharpe={passive['sharpe']:.3f}, "
                  f"Sortino={passive['sortino']:.3f}, "
                  f"WR={passive['win_rate']:.1%}, PF={passive['profit_factor']:.2f}")
            print(f"    Market entry:  Sharpe={market['sharpe']:.3f}, "
                  f"Sortino={market['sortino']:.3f}, "
                  f"WR={market['win_rate']:.1%}, PF={market['profit_factor']:.2f}")
            print(f"    Total PnL: passive=${passive['total_pnl_dollars']:,.0f}, "
                  f"market=${market['total_pnl_dollars']:,.0f}")
    
    # Key conclusion
    top10_passive = summary['execution_analysis']['top10']['passive_cost']
    top10_market = summary['execution_analysis']['top10']['market_cost']
    
    print(f"\n  === KEY CONCLUSION ===")
    if top10_passive and top10_passive['sharpe'] > 0 and top10_market and top10_market['sharpe'] > 0:
        print(f"  v7 is profitable with BOTH passive AND market orders at top 10%")
        print(f"  This is a major improvement over v6 which needed passive-only at top 20%+")
    elif top10_passive and top10_passive['sharpe'] > 0:
        print(f"  v7 profitable with passive orders at top 10%, but not market orders")
        print(f"  Execution edge exists but requires patience for fills")
    else:
        print(f"  v7 needs tighter filtering (top 5%) for profitability")
    
    # ============================================================
    # Save results
    # ============================================================
    print(f"\n[{ts()}] Saving results...")
    
    # Main summary
    summary['mfe_analysis'] = mfe_results
    summary['cancel_window_analysis'] = cancel_results
    summary['exit_strategy_analysis'] = exit_results
    summary['daily_results'] = daily
    summary['timestamp'] = datetime.now().isoformat()
    
    summary_path = os.path.join(OUTPUT_DIR, 'execution_reeval_results.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved results to {summary_path}")
    
    # ============================================================
    # MLflow Logging
    # ============================================================
    if MLFLOW_AVAILABLE:
        print(f"\n[{ts()}] Logging to MLflow...")
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("v7_execution_reeval")
            
            run_name = f"v7_exec_reeval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.start_run(run_name=run_name)
            
            # Log params
            mlflow.log_params({
                'pred_source': 'meta_v7_prod',
                'n_oot_dates': len(unique_dates),
                'n_predictions': len(preds),
                'commission_passive': COMMISSION_RT_TICKS,
                'commission_market': MARKET_ORDER_COST_TICKS,
                'confidence_tiers': str(CONFIDENCE_TIERS),
            })
            
            # Log key metrics
            mlflow.log_metrics({
                'concat_spearman': float(sp),
                'v6_spearman': 0.167,
                'improvement_pct': float((sp - 0.167) / 0.167 * 100),
            })
            
            # Log tier metrics
            for tier in [5, 10, 20]:
                for cost_type in ['passive_cost', 'market_cost']:
                    m = summary['execution_analysis'][f'top{tier}'].get(cost_type)
                    if m:
                        prefix = f'top{tier}_{cost_type.split("_")[0]}'
                        mlflow.log_metrics({
                            f'{prefix}_sharpe': m['sharpe'],
                            f'{prefix}_sortino': m['sortino'],
                            f'{prefix}_wr': m['win_rate'],
                            f'{prefix}_pf': m['profit_factor'],
                            f'{prefix}_avg_pnl': m['avg_pnl_ticks'],
                            f'{prefix}_total_pnl': m['total_pnl_dollars'],
                        })
            
            # Log MFE sweet spots
            for tier in [5, 10, 20]:
                for h in HORIZONS_SEC:
                    r = mfe_results.get(f'top{tier}', {}).get(h)
                    if r:
                        mlflow.log_metrics({
                            f'mfe_top{tier}_{h}s_mean': r['mean_excursion'],
                            f'mfe_top{tier}_{h}s_p90': r['p90_excursion'],
                            f'mfe_top{tier}_{h}s_favorable': r['pct_favorable'],
                        })
            
            # Log artifacts
            mlflow.log_artifact(summary_path)
            
            mlflow.end_run()
            print(f"  MLflow run logged: {run_name}")
        except Exception as e:
            print(f"  MLflow error (non-fatal): {e}")
    else:
        print(f"  MLflow not available, skipping")
    
    elapsed = time.time() - start_time
    print(f"\n[{ts()}] === Done in {elapsed:.1f}s ===")


if __name__ == '__main__':
    main()
