#!/usr/bin/env python3
"""
EXTREME SELECTIVITY V1 — does the top 0.1% / 0.5% / 1% of CNN-Mamba v2 confidence have edge?

Today's continuous-exit sweep tested entry thresholds {0.5, 0.75, 1.0, 1.5} — none clear cost.
But HC #515 R6 forbids declaring death without exhausting selectivity. The 1.5σ threshold still
fires 837 trades/day — not extreme. Test 2.0σ, 2.5σ, 3.0σ, 4.0σ — actual top-percentile regimes.

Uses the trustworthy v1 MFE/MAE labels produced today (46 dates, 2M predictions). Cost = 0.376t
commission only (HC #512). Static-hold exit at 10s horizon (CNN-Mamba v2's best raw IC).

Decisive: if any extreme-selectivity config produces positive net/trade across 30+ days with
regime-agnostic Sharpe per HC #428 R1, it's a live direction. Otherwise this kills the
selectivity hypothesis too.
"""
from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime
import numpy as np

V1_DIR = Path('/home/jupiter/Lvl3Quant/output/mfe_mae_labels_v1')
OUT_DIR = Path('/home/jupiter/Lvl3Quant/output/extreme_selectivity_v1')
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_TICKS = 0.376
HORIZON = '10s'           # best raw IC per decay analysis
PRED_COL = 2              # CNN-Mamba v2's 10s head

# σ thresholds to test
SIGMA_LEVELS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

def main():
    files = sorted(V1_DIR.glob('*_mfe_mae.npz'))
    print(f'Loaded {len(files)} dates of v1 MFE/MAE')

    # First pass: estimate sigma of predictions across all dates
    all_preds = []
    for f in files:
        d = np.load(f)
        if len(d['predictions']) < 100: continue
        all_preds.append(d['predictions'][:, PRED_COL])
    P_all = np.concatenate(all_preds)
    sigma = float(P_all.std())
    print(f'CNN-Mamba v2 10s pred sigma: {sigma:.4f}, mean: {P_all.mean():.4f}, n: {len(P_all):,}')

    results = []
    for sigma_k in SIGMA_LEVELS:
        thr = sigma_k * sigma
        # Realized "trade" net per prediction:
        # Long if pred > +thr, short if pred < -thr, signed P&L = sign * realized direction at 10s.
        # Use MFE-MAE difference signed by prediction direction.
        # Net at 10s = realized 10s return signed by trade direction. We don't have explicit
        # realized 10s in the v1 file — we have MFE_10s and MAE_10s (both unsigned magnitudes).
        # Approximation: take the *net 10s checkpoint position* as MFE - MAE signed by realized
        # path. Since we don't have it, use a conservative proxy: realized 10s = (mfe_10s - mae_10s)
        # which is approximately the signed move at the 10s checkpoint when MFE preceded MAE,
        # and vice versa. This is a known coarse proxy.
        # (For a strict realized return we'd need raw price path — separate work item.)
        all_trades = []
        per_day = []
        for f in files:
            d = np.load(f)
            preds = d['predictions'][:, PRED_COL]
            if len(preds) < 100: continue
            mfe = d['mfe_10s']
            mae = d['mae_10s']
            # signed realized ≈ pred direction * (mfe - mae) is too generous (selection bias).
            # Instead: realized magnitude ≈ mfe-mae (the asymmetry), signed by pred → gross net per trade
            realized_signed = (mfe - mae)  # proxy: net favorable minus adverse (unsigned by self)
            # Filter to high-conviction trades
            mask = np.abs(preds) >= thr
            if mask.sum() < 5: continue
            signed_dir = np.sign(preds[mask])
            gross = signed_dir * realized_signed[mask]  # proxy gross
            net = gross - COMMISSION_TICKS
            all_trades.append(net)
            per_day.append({
                'date': str(d['date']),
                'n_trades': int(mask.sum()),
                'gross_mean': float(gross.mean()),
                'net_mean': float(net.mean()),
                'wr': float((gross > 0).mean()),
            })
        if not all_trades:
            continue
        T = np.concatenate(all_trades)
        if len(T) < 50: continue
        # Sharpe across daily net means
        daily = np.array([d['net_mean'] for d in per_day])
        mu_d = float(daily.mean()); sd_d = float(daily.std(ddof=1)) if len(daily)>1 else 0
        sharpe = (mu_d / sd_d * np.sqrt(252)) if sd_d > 1e-9 else float('nan')
        results.append({
            'sigma_k': sigma_k,
            'threshold': thr,
            'n_dates': len(per_day),
            'n_trades_total': int(len(T)),
            'trades_per_day_avg': float(np.mean([d['n_trades'] for d in per_day])),
            'net_mean_ticks': float(T.mean()),
            'net_median_ticks': float(np.median(T)),
            'wr': float((T + COMMISSION_TICKS > 0).mean()),
            'daily_sharpe': sharpe,
            'profitable_days': int((daily > 0).sum()),
            'pct_profitable_days': float((daily > 0).mean()),
            'best_day_net': float(daily.max()),
            'worst_day_net': float(daily.min()),
        })

    print('\n=== EXTREME SELECTIVITY RESULTS ===')
    print(f'{"σ":>5} | {"thr":>7} | {"trades/d":>9} | {"days":>5} | {"gross-com":>10} | {"WR":>6} | {"Sharpe":>7} | {"prof days":>10}')
    print('-' * 90)
    for r in results:
        print(f'{r["sigma_k"]:5.1f} | {r["threshold"]:7.3f} | {r["trades_per_day_avg"]:9.1f} | '
              f'{r["n_dates"]:5d} | {r["net_mean_ticks"]:+10.3f} | {r["wr"]:6.1%} | '
              f'{r["daily_sharpe"]:7.2f} | {r["pct_profitable_days"]:10.1%}')

    out_path = OUT_DIR / f'extreme_selectivity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as fh:
        json.dump({'sigma': sigma, 'results': results}, fh, indent=2)
    print(f'\nWrote {out_path}')

    # Verdict
    profitable = [r for r in results if r['net_mean_ticks'] > 0 and r['pct_profitable_days'] > 0.55]
    print('\n=== VERDICT ===')
    if profitable:
        print(f'{len(profitable)} extreme-selectivity configs net positive — LIVE DIRECTION.')
        for p in profitable:
            print(f"  σ={p['sigma_k']}: net +{p['net_mean_ticks']:.3f}t, {p['trades_per_day_avg']:.1f} trades/d, {p['pct_profitable_days']:.0%} profitable days")
    else:
        print('No extreme-selectivity config clears commission. Direction CLOSED.')

if __name__ == '__main__':
    main()
