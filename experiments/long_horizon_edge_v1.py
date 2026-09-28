#!/usr/bin/env python3
"""
LONG-HORIZON EDGE V1 — does CNN-Mamba v2 have any predictive edge at 60s or 5min?

Today's verdict (2026-06-03): at horizons ≤30s the signal is +0.18t gross vs 0.376t commission
(structural ~0.20t gap, 0/99 continuous-exit configs profitable). The remaining open
question per HC #515 R6 + HC #432 R2 is whether longer horizons retain any edge.

This script aligns CNN-Mamba v2 OOT predictions (from the broken-MFE/MAE v2 job —
predictions themselves are valid, only the MFE/MAE values were broken) to alpha_labels_v4
which already contains pre-computed log_ret_60s and log_ret_5min for every smart_v3 event.

Output: IC at 60s and 5min, top-decile realized returns (in ticks) with regime stratification
per HC #428 R1, and a clean pass/fail verdict against the 0.376t commission floor.

NO new model training. Pure analysis on existing data. Decisive: positive → train long-horizon
model on Razer. Null → kill the long-horizon direction.
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np

# Cost constants (HC #512: 0.376t commission only, no spread cost)
COMMISSION_TICKS = 0.376
TICK_VALUE = 12.50  # $ per tick (informational only)

V2_DIR    = Path('/home/jupiter/Lvl3Quant/data/mfe_mae_labels_v2')
ALPHA_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_alpha_labels_v4')
OUT_DIR   = Path('/home/jupiter/Lvl3Quant/output/long_horizon_edge_v1')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per HC #428 R1: stratify by ES close-to-close regime. We don't have an ES daily file
# wired in here, so we proxy regime per-date via mean signed return across all predictions
# on that day (green = mean > +0.5t, red = mean < -0.5t, flat = otherwise). This is a
# reasonable proxy because it captures whether the underlying tape was trending up/down/range.
def classify_regime(mean_signed_return_ticks: float) -> str:
    if mean_signed_return_ticks > 0.5: return 'green'
    if mean_signed_return_ticks < -0.5: return 'red'
    return 'flat'

def pearson_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Concat IC (Pearson) with NaN-safe handling."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100: return float('nan')
    x, y = x[m], y[m]
    if x.std() < 1e-9 or y.std() < 1e-9: return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

def daily_sharpe(daily_returns_ticks: np.ndarray) -> float:
    """Annualized Sharpe-like ratio from daily mean returns."""
    if len(daily_returns_ticks) < 3: return float('nan')
    mu = float(np.mean(daily_returns_ticks))
    sd = float(np.std(daily_returns_ticks, ddof=1))
    if sd < 1e-9: return float('nan')
    return mu / sd * np.sqrt(252)

def overlapping_dates() -> list[str]:
    v2 = {f.stem.split('_')[0] for f in V2_DIR.glob('*_mfe_mae.npz')}
    al = {f.stem.split('_')[0] for f in ALPHA_DIR.glob('*_alpha_labels.npz')}
    return sorted(v2 & al)

def horizon_to_ticks(log_ret: np.ndarray, ref_price: float = 6000.0) -> np.ndarray:
    """Convert log returns to tick moves. ES front-month ~6000, tick=0.25. Approx delta = log_ret * price / 0.25."""
    # log(1+r) ≈ r for small r → price * r = $ move → / $12.50 = ticks
    # log_ret here is float32 — already small. Multiply by price then divide by 0.25 to get points*4=ticks.
    return log_ret.astype(np.float64) * ref_price / 0.25

def analyze_horizon(dates: list[str], pred_col: int, ret_key: str, horizon_label: str) -> dict:
    """For one horizon, build aligned (pred, realized) pairs across all dates and compute metrics.

    pred_col: which CNN-Mamba v2 prediction head to use (0=1s, 1=5s, 2=10s — closest-horizon proxy).
              For 60s and 5min we use col 2 (10s head, the model's longest available) — testing
              whether the model's longest horizon retains correlation with even-longer realized moves.
    ret_key:  'log_ret_60s' or 'log_ret_5min'.
    """
    all_preds: list[np.ndarray] = []
    all_realised_ticks: list[np.ndarray] = []
    per_day = []

    for d in dates:
        v2 = np.load(V2_DIR / f'{d}_mfe_mae.npz')
        al = np.load(ALPHA_DIR / f'{d}_alpha_labels.npz')

        preds_full = v2['predictions']           # (N, 3)
        ev_idx     = v2['event_indices']         # (N,)
        if len(preds_full) < 100:                # skip near-empty days
            continue
        ret_full = al[ret_key]                   # full-day length, indexed by event index
        if ev_idx.max() >= len(ret_full):        # alignment guard
            continue

        preds = preds_full[:, pred_col].astype(np.float64)
        realised = horizon_to_ticks(ret_full[ev_idx])  # tick moves

        all_preds.append(preds)
        all_realised_ticks.append(realised)

        # Per-day metrics
        # Top decile by |pred|, signed by pred direction
        absp = np.abs(preds)
        if absp.max() == 0: continue
        thr = np.quantile(absp, 0.9)
        mask = absp >= thr
        if mask.sum() < 20: continue
        signed = np.sign(preds[mask])
        gross = signed * realised[mask]
        net = gross - COMMISSION_TICKS
        per_day.append({
            'date': d,
            'n_total': int(len(preds)),
            'n_top10': int(mask.sum()),
            'top10_gross_mean_ticks': float(gross.mean()),
            'top10_net_mean_ticks': float(net.mean()),
            'top10_wr': float((gross > 0).mean()),
            'mean_signed_return_ticks': float(realised.mean()),
        })

    if not all_preds:
        return {'horizon': horizon_label, 'error': 'no aligned data'}

    P = np.concatenate(all_preds)
    R = np.concatenate(all_realised_ticks)

    # Concat IC (signed)
    ic = pearson_ic(P, R)

    # Top-decile (signed): mean gross/net per trade across all top-decile predictions
    absp = np.abs(P)
    thr = np.quantile(absp, 0.9)
    mask = absp >= thr
    signed = np.sign(P[mask])
    gross = signed * R[mask]
    net = gross - COMMISSION_TICKS

    # Regime-stratified daily Sharpe (HC #428 R1)
    by_regime = {'green': [], 'red': [], 'flat': []}
    for d in per_day:
        regime = classify_regime(d['mean_signed_return_ticks'])
        by_regime[regime].append(d['top10_net_mean_ticks'])
    sharpe_by_regime = {r: daily_sharpe(np.array(v)) for r, v in by_regime.items()}
    days_by_regime = {r: len(v) for r, v in by_regime.items()}

    overall_sharpe = daily_sharpe(np.array([d['top10_net_mean_ticks'] for d in per_day]))

    # Day concentration (HC #344): does a single day carry >70% of positive contribution?
    daily_nets = np.array([d['top10_net_mean_ticks'] for d in per_day])
    positives = daily_nets[daily_nets > 0]
    day_conc = float(positives.max() / positives.sum()) if positives.sum() > 0 else 0.0

    # Regime-agnostic gate: |s_green - s_red| / max(|.|, |.|) <= 0.50 (HC #428 R1)
    sg = sharpe_by_regime.get('green') or 0.0
    sr = sharpe_by_regime.get('red') or 0.0
    if max(abs(sg), abs(sr)) > 1e-9:
        regime_gap = abs(sg - sr) / max(abs(sg), abs(sr))
    else:
        regime_gap = 0.0
    regime_pass = regime_gap <= 0.50

    # Verdict
    profitable_after_cost = float(net.mean()) > 0
    pass_all = profitable_after_cost and regime_pass and (day_conc <= 0.70)

    return {
        'horizon': horizon_label,
        'pred_col_used': pred_col,
        'n_dates': len(per_day),
        'n_total_preds': int(len(P)),
        'n_top10_preds': int(mask.sum()),
        'concat_ic': ic,
        'top10_gross_mean_ticks': float(gross.mean()),
        'top10_net_mean_ticks': float(net.mean()),
        'top10_wr': float((gross > 0).mean()),
        'overall_daily_sharpe': overall_sharpe,
        'sharpe_by_regime': sharpe_by_regime,
        'days_by_regime': days_by_regime,
        'regime_gap': regime_gap,
        'regime_agnostic_pass': regime_pass,
        'day_concentration': day_conc,
        'day_conc_pass': day_conc <= 0.70,
        'profitable_after_cost': profitable_after_cost,
        'overall_pass': pass_all,
        'per_day': per_day,
    }

def main():
    dates = overlapping_dates()
    print(f'Overlap dates: {len(dates)}  ({dates[0]} … {dates[-1]})')
    if len(dates) < 10:
        raise SystemExit('Insufficient overlap (<10 dates). Abort.')

    results = {}
    for ret_key, label in [('log_ret_60s', '60s'), ('log_ret_5min', '5min')]:
        print(f'\n=== Horizon {label} ===')
        # Use 10s prediction head (col 2) — model's longest — as the extrapolation probe
        r = analyze_horizon(dates, pred_col=2, ret_key=ret_key, horizon_label=label)
        for k, v in r.items():
            if k != 'per_day':
                print(f'  {k}: {v}')
        results[label] = r

    out_path = OUT_DIR / f'long_horizon_edge_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f'\nWrote {out_path}')

    print('\n=== HEADLINE VERDICTS ===')
    for label, r in results.items():
        if 'error' in r:
            print(f'{label}: {r["error"]}'); continue
        ok = 'PASS' if r['overall_pass'] else 'FAIL'
        print(f'{label}: IC {r["concat_ic"]:.4f} | gross {r["top10_gross_mean_ticks"]:+.3f}t | '
              f'net {r["top10_net_mean_ticks"]:+.3f}t | WR {r["top10_wr"]:.1%} | '
              f'Sharpe {r["overall_daily_sharpe"]:.2f} | regime_gap {r["regime_gap"]:.2f} | {ok}')

if __name__ == '__main__':
    main()
