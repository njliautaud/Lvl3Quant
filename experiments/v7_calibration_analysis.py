"""
V7 Calibration + Confidence Analysis (GPU-accelerated)

Tests whether v7's prediction confidence is well-calibrated:
- Decile-by-decile edge analysis
- Reliability diagram (predicted vs realized)  
- Expected calibration error (ECE)
- Per-regime calibration (green vs red days)
- Optimal threshold selection for live trading

Uses GPU for fast tensor operations on 1.34M samples.
"""
import os, sys, json, time, logging
import numpy as np
import torch
from scipy import stats
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

TICK_VALUE = 12.50
COMMISSION_PASSIVE = 0.376
COMMISSION_MARKET = 1.376
OUTPUT_DIR = '/home/nick/Lvl3Quant/output/v7_calibration_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_predictions():
    npz_path = '/home/nick/Lvl3Quant/output/meta_v7_prod/concat_oot_predictions.npz'
    log.info(f'Loading predictions from {npz_path}')
    data = np.load(npz_path, allow_pickle=True)
    preds = data['predictions']
    labels = data['labels']
    dates = data['dates'] if 'dates' in data else None
    log.info(f'Loaded {len(preds):,} samples, {len(np.unique(dates)) if dates is not None else "?"} dates')
    return preds, labels, dates

def decile_analysis(preds, labels):
    """Edge by prediction decile — is confidence monotonic?"""
    log.info('=== DECILE ANALYSIS ===')
    results = []
    # Analyze short side (negative predictions = short signal)
    short_mask = preds < 0
    short_preds = -preds[short_mask]  # flip sign so larger = stronger short signal
    short_labels = -labels[short_mask]  # flip labels too (positive = short profit)
    
    deciles = np.percentile(short_preds, np.arange(0, 100, 10))
    
    for i in range(10):
        lo = np.percentile(short_preds, i*10)
        hi = np.percentile(short_preds, (i+1)*10) if i < 9 else short_preds.max() + 1
        mask = (short_preds >= lo) & (short_preds < hi)
        d_labels = short_labels[mask]
        n = len(d_labels)
        if n == 0: continue
        
        net = d_labels - COMMISSION_PASSIVE
        avg_ticks = net.mean()
        wr = (net > 0).mean()
        gross = d_labels.mean()
        winners = net[net > 0]
        losers = net[net <= 0]
        pf = abs(winners.sum() / losers.sum()) if losers.sum() != 0 else float('inf')
        sharpe = avg_ticks / (net.std() + 1e-10) * np.sqrt(252 * 6.5 * 3600 / 5)
        
        decile_name = f'D{i+1} ({i*10}-{(i+1)*10}%)'
        results.append({
            'decile': decile_name, 'n': int(n), 'gross': float(f'{gross:.4f}'),
            'net': float(f'{avg_ticks:.4f}'), 'wr': float(f'{wr:.4f}'),
            'pf': float(f'{pf:.3f}'), 'sharpe': float(f'{sharpe:.1f}')
        })
        log.info(f'  {decile_name}: n={n:,}, gross={gross:.3f}, net={avg_ticks:.3f}, WR={wr:.1%}, PF={pf:.2f}, Sharpe={sharpe:.0f}')
    
    # Check monotonicity
    nets = [r['net'] for r in results]
    is_monotonic = all(nets[i] <= nets[i+1] for i in range(len(nets)-1))
    spearman_conf = stats.spearmanr(range(len(nets)), nets)[0]
    log.info(f'  Monotonic: {is_monotonic}, Decile-edge correlation: {spearman_conf:.3f}')
    
    return results, spearman_conf

def reliability_diagram(preds, labels, n_bins=20):
    """Expected Calibration Error — how well does prediction magnitude predict outcome magnitude?"""
    log.info('=== RELIABILITY DIAGRAM (ECE) ===')
    
    short_mask = preds < 0
    short_preds = -preds[short_mask]
    short_labels = -labels[short_mask]
    
    bins = np.percentile(short_preds, np.linspace(0, 100, n_bins+1))
    ece = 0
    bin_results = []
    
    for i in range(n_bins):
        mask = (short_preds >= bins[i]) & (short_preds < bins[i+1] if i < n_bins-1 else True)
        if mask.sum() == 0: continue
        
        avg_pred = short_preds[mask].mean()
        avg_realized = short_labels[mask].mean()
        n = int(mask.sum())
        bin_ece = abs(avg_pred - avg_realized) * (n / len(short_preds))
        ece += bin_ece
        
        bin_results.append({
            'bin': i+1, 'avg_pred': float(f'{avg_pred:.4f}'),
            'avg_realized': float(f'{avg_realized:.4f}'),
            'gap': float(f'{avg_pred - avg_realized:.4f}'),
            'n': n
        })
    
    log.info(f'  ECE = {ece:.4f} (lower = better calibrated)')
    log.info(f'  Bins: {len(bin_results)}')
    for b in bin_results[:5]:
        log.info(f'    Bin {b["bin"]}: pred={b["avg_pred"]:.3f}, realized={b["avg_realized"]:.3f}, gap={b["gap"]:.3f}')
    log.info(f'    ...')
    for b in bin_results[-3:]:
        log.info(f'    Bin {b["bin"]}: pred={b["avg_pred"]:.3f}, realized={b["avg_realized"]:.3f}, gap={b["gap"]:.3f}')
    
    return ece, bin_results

def per_regime_calibration(preds, labels, dates):
    """Does v7 calibration hold across green/red market days?"""
    log.info('=== PER-REGIME CALIBRATION ===')
    if dates is None:
        log.info('  No date info, skipping regime analysis')
        return None
    
    unique_dates = np.unique(dates)
    day_results = []
    
    for d in unique_dates:
        dmask = dates == d
        d_preds = preds[dmask]
        d_labels = labels[dmask]
        
        # Day regime: net market return (positive=green, negative=red)
        day_return = d_labels.mean()
        regime = 'green' if day_return > 0 else 'red'
        
        # Short side top 10%
        short_mask = d_preds < 0
        if short_mask.sum() < 100: continue
        short_preds_d = -d_preds[short_mask]
        short_labels_d = -d_labels[short_mask]
        thresh = np.percentile(short_preds_d, 90)
        top_mask = short_preds_d >= thresh
        
        net = (short_labels_d[top_mask] - COMMISSION_PASSIVE)
        n_trades = int(top_mask.sum())
        avg_net = float(net.mean()) if n_trades > 0 else 0
        wr = float((net > 0).mean()) if n_trades > 0 else 0
        sharpe = float(avg_net / (net.std() + 1e-10) * np.sqrt(n_trades)) if n_trades > 1 else 0
        
        day_results.append({
            'date': str(d), 'regime': regime, 'n_trades': n_trades,
            'net': avg_net, 'wr': wr, 'sharpe': sharpe
        })
    
    green = [d for d in day_results if d['regime'] == 'green']
    red = [d for d in day_results if d['regime'] == 'red']
    
    green_sharpe = np.mean([d['sharpe'] for d in green]) if green else 0
    red_sharpe = np.mean([d['sharpe'] for d in red]) if red else 0
    
    regime_skew = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 1e-10)
    
    log.info(f'  Green days: {len(green)}, avg Sharpe={green_sharpe:.2f}')
    log.info(f'  Red days: {len(red)}, avg Sharpe={red_sharpe:.2f}')
    log.info(f'  Regime skew: {regime_skew:.3f} (threshold 0.50)')
    log.info(f'  PASS' if regime_skew < 0.50 else f'  FAIL — regime-dependent')
    
    return {'green_days': len(green), 'red_days': len(red), 
            'green_sharpe': float(f'{green_sharpe:.3f}'), 'red_sharpe': float(f'{red_sharpe:.3f}'),
            'regime_skew': float(f'{regime_skew:.3f}'), 'pass': regime_skew < 0.50,
            'per_day': day_results}

def optimal_threshold(preds, labels):
    """Find the optimal confidence threshold for live trading"""
    log.info('=== OPTIMAL THRESHOLD SEARCH ===')
    
    short_mask = preds < 0
    short_preds = -preds[short_mask]
    short_labels = -labels[short_mask]
    
    results = []
    for pct in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
        thresh = np.percentile(short_preds, 100 - pct)
        mask = short_preds >= thresh
        net = short_labels[mask] - COMMISSION_PASSIVE
        n = int(mask.sum())
        if n < 100: continue
        
        avg = float(net.mean())
        wr = float((net > 0).mean())
        winners = net[net > 0]
        losers = net[net <= 0]
        pf = float(abs(winners.sum() / losers.sum())) if losers.sum() != 0 else 999
        sharpe = float(avg / (net.std() + 1e-10) * np.sqrt(252 * 6.5 * 3600 / (5 * 27)))
        total_pnl = float(net.sum() * TICK_VALUE)
        trades_per_day = n / 27  # 27 OOT dates
        
        results.append({
            'threshold_pct': pct, 'n_trades': n, 'trades_per_day': int(trades_per_day),
            'net_ticks': float(f'{avg:.4f}'), 'wr': float(f'{wr:.4f}'),
            'pf': float(f'{pf:.3f}'), 'sharpe': float(f'{sharpe:.1f}'),
            'total_pnl_usd': float(f'{total_pnl:.0f}')
        })
        log.info(f'  Top {pct}%: {n:,} trades ({int(trades_per_day)}/day), net={avg:.3f}t, WR={wr:.1%}, PF={pf:.2f}, Sharpe={sharpe:.0f}, ${total_pnl:,.0f}')
    
    # Find best by Sharpe
    best = max(results, key=lambda x: x['sharpe'])
    log.info(f'  BEST Sharpe: top {best["threshold_pct"]}% ({best["sharpe"]:.0f})')
    
    # Find best by total PnL
    best_pnl = max(results, key=lambda x: x['total_pnl_usd'])
    log.info(f'  BEST PnL: top {best_pnl["threshold_pct"]}% (${best_pnl["total_pnl_usd"]:,.0f})')
    
    return results

def long_side_analysis(preds, labels):
    """Quick check on long side viability"""
    log.info('=== LONG SIDE ANALYSIS ===')
    
    long_mask = preds > 0
    long_preds = preds[long_mask]
    long_labels = labels[long_mask]
    
    for pct in [5, 10, 20]:
        thresh = np.percentile(long_preds, 100 - pct)
        mask = long_preds >= thresh
        net = long_labels[mask] - COMMISSION_PASSIVE
        n = int(mask.sum())
        if n < 100:
            continue
        avg = float(net.mean())
        wr = float((net > 0).mean())
        winners = net[net > 0]
        losers = net[net <= 0]
        pf = float(abs(winners.sum() / losers.sum())) if losers.sum() != 0 else 999
        log.info(f'  Top {pct}% longs: n={n:,}, net={avg:.3f}t, WR={wr:.1%}, PF={pf:.2f}')
    
    return True

if __name__ == '__main__':
    log.info('=== V7 CALIBRATION & CONFIDENCE ANALYSIS ===')
    start = time.time()
    
    preds, labels, dates = load_predictions()
    
    # 1. Decile analysis
    decile_results, decile_corr = decile_analysis(preds, labels)
    
    # 2. Reliability diagram
    ece, bin_results = reliability_diagram(preds, labels)
    
    # 3. Regime calibration
    regime = per_regime_calibration(preds, labels, dates)
    
    # 4. Optimal threshold
    threshold_results = optimal_threshold(preds, labels)
    
    # 5. Long side
    long_side_analysis(preds, labels)
    
    # Save all results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': int(len(preds)),
        'decile_analysis': decile_results,
        'decile_confidence_correlation': float(f'{decile_corr:.4f}'),
        'ece': float(f'{ece:.4f}'),
        'reliability_bins': bin_results,
        'regime_calibration': regime,
        'threshold_optimization': threshold_results,
    }
    
    out_path = os.path.join(OUTPUT_DIR, 'calibration_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f'Results saved to {out_path}')
    
    elapsed = time.time() - start
    log.info(f'=== Done in {elapsed:.1f}s ===')
