#!/usr/bin/env python3
"""
Confluence Analysis: CNN-Mamba v2 + EOFI Pressure Agreement
==========================================================
Tests whether filtering CNN-Mamba entries by EOFI pressure agreement
improves trade quality. EOFI predictions are at event-level; CNN-Mamba
predictions are at stride=250 intervals. We align by event index.

Key question: When CNN-Mamba says "short" AND EOFI pressure confirms
(negative pressure), do we get better win rates and net PnL?
"""

import json
import logging
import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger('confluence')

BASE = Path("/home/jupiter/Lvl3Quant")
EVENTS_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3"
EOFI_PREDS_FILE = BASE / "output" / "eofi_pressure_xgb_v1" / "predictions_eofi_10s.npz"
EOFI_SUMMARY = BASE / "output" / "eofi_pressure_xgb_v1" / "summary_eofi_10s.json"
CM_PRED_DIRS = [
    BASE / "output" / "cnn_mamba_v2_all_oot",
    BASE / "output" / "cnn_mamba_v2_bulk_oot_v2",
    BASE / "output" / "cnn_mamba_v2_bulk_oot",
]
OUTPUT_DIR = BASE / "output" / "confluence_eofi_cnnmamba"

TICK_SIZE = 0.25
STRIDE = 250
COMMISSION_RT_TICKS = 0.376


def load_cnn_mamba_preds(date_str: str) -> dict:
    """Load CNN-Mamba predictions for a date."""
    for d in CM_PRED_DIRS:
        f = d / f"{date_str}_predictions.npz"
        if f.exists():
            data = np.load(str(f), allow_pickle=True)
            preds = data['predictions']  # (N, 3) — pred_1s, pred_5s, pred_10s
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            stride = int(data.get('stride', STRIDE))
            window = int(data.get('window_size', 3000))
            return {'predictions': preds, 'stride': stride, 'window': window, 'n_preds': len(preds)}
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load EOFI predictions and split by date
    log.info("Loading EOFI predictions...")
    with open(EOFI_SUMMARY) as f:
        summary = json.load(f)
    fold_details = summary['fold_details']

    eofi_data = np.load(str(EOFI_PREDS_FILE))
    all_eofi_preds = eofi_data['predictions']

    eofi_by_date = {}
    offset = 0
    for fd in fold_details:
        n = fd['n_test']
        eofi_by_date[fd['fold_date']] = all_eofi_preds[offset:offset + n]
        offset += n
    log.info(f"EOFI predictions split into {len(eofi_by_date)} dates")

    # Find overlapping dates with CNN-Mamba
    cm_dates = set()
    for d in CM_PRED_DIRS:
        if d.exists():
            for f in d.glob("*_predictions.npz"):
                dt = f.stem.replace("_predictions", "")
                if len(dt) == 8 and dt.isdigit():
                    cm_dates.add(dt)

    overlap_dates = sorted(cm_dates & set(eofi_by_date.keys()))
    log.info(f"Overlapping dates: {len(overlap_dates)}")

    # For each date: align CNN-Mamba and EOFI, then compute entry-level metrics
    results_by_quantile = {q: {'pnl_10s': [], 'pnl_5s': [], 'pnl_1s': [], 'n': 0}
                           for q in ['all', 'eofi_agree', 'eofi_disagree',
                                     'top10_all', 'top10_agree', 'top10_disagree',
                                     'top20_all', 'top20_agree', 'top20_disagree']}

    daily_results = []

    for di, date_str in enumerate(overlap_dates):
        # Load CNN-Mamba
        cm = load_cnn_mamba_preds(date_str)
        if cm is None:
            continue

        # Load events (just for mid prices and timestamps via mmap)
        event_file = EVENTS_DIR / f"{date_str}_mbo_events.npz"
        if not event_file.exists():
            continue

        try:
            ev = np.load(str(event_file), mmap_mode='r')
            n_events = len(ev['timestamps'])
            bid_ask = ev['events'][:, [5, 6]]  # (N, 2)
        except Exception as e:
            log.warning(f"Failed to load events for {date_str}: {e}")
            continue

        eofi_preds = eofi_by_date[date_str]
        cm_preds = cm['predictions']
        stride = cm['stride']
        window = cm['window']
        n_cm = cm['n_preds']

        # CNN-Mamba predictions correspond to events at indices:
        # window-1, window-1+stride, window-1+2*stride, ...
        cm_event_indices = np.arange(n_cm) * stride + (window - 1)

        # Filter to valid indices (within event range and EOFI range)
        n_eofi = len(eofi_preds)
        valid_mask = (cm_event_indices < n_events) & (cm_event_indices < n_eofi)
        cm_event_indices = cm_event_indices[valid_mask]
        cm_valid_preds = cm_preds[valid_mask]

        if len(cm_event_indices) < 100:
            continue

        # Get EOFI predictions at CNN-Mamba evaluation points
        eofi_at_cm = eofi_preds[cm_event_indices]

        # Get mid prices at evaluation points
        bid = bid_ask[cm_event_indices, 0].copy().astype(np.float64)
        ask = bid_ask[cm_event_indices, 1].copy().astype(np.float64)
        mid = (bid + ask) / 2.0

        # Get forward returns at multiple horizons (using future mid prices)
        # For 10s: ~40 strides at 250 events/stride ≈ 10s at RTH density
        for horizon_name, horizon_strides in [('1s', 4), ('5s', 20), ('10s', 40)]:
            # Forward mid at horizon
            n_valid = len(cm_event_indices)
            fwd_indices = cm_event_indices + horizon_strides * stride
            fwd_valid = fwd_indices < n_events

            if fwd_valid.sum() < 100:
                continue

            fwd_bid = bid_ask[fwd_indices[fwd_valid], 0].copy().astype(np.float64)
            fwd_ask = bid_ask[fwd_indices[fwd_valid], 1].copy().astype(np.float64)
            fwd_mid = (fwd_bid + fwd_ask) / 2.0

            # CNN-Mamba directional prediction (use 10s horizon pred, col 2)
            cm_col = min(2, cm_valid_preds.shape[1] - 1)  # 10s horizon
            cm_dir = cm_valid_preds[fwd_valid, cm_col]

            # EOFI pressure at entry
            eofi_entry = eofi_at_cm[fwd_valid]

            # Current mid
            cur_mid = mid[fwd_valid]

            # Forward return in ticks (signed by prediction direction)
            fwd_ret_ticks = (fwd_mid - cur_mid) / TICK_SIZE

            # CNN-Mamba predicted direction
            cm_long = cm_dir > 0
            cm_short = cm_dir < 0

            # Signed PnL: if CNN-Mamba says long, PnL = fwd_ret; if short, PnL = -fwd_ret
            signed_pnl = np.where(cm_long, fwd_ret_ticks, np.where(cm_short, -fwd_ret_ticks, 0))
            # After cost (taker)
            net_pnl = signed_pnl - COMMISSION_RT_TICKS  # only commission, spread handled by bid/ask

            # EOFI agreement: CNN-Mamba long AND EOFI positive, OR CNN-Mamba short AND EOFI negative
            eofi_agrees = (cm_long & (eofi_entry > 0)) | (cm_short & (eofi_entry < 0))
            eofi_disagrees = (cm_long & (eofi_entry < 0)) | (cm_short & (eofi_entry > 0))

            # Signal strength (|CNN-Mamba pred|)
            cm_strength = np.abs(cm_dir)
            p90 = np.percentile(cm_strength, 90)
            p80 = np.percentile(cm_strength, 80)
            top10 = cm_strength >= p90
            top20 = cm_strength >= p80

            key_suffix = horizon_name.replace('s', 's')

            # Store results
            has_signal = cm_long | cm_short

            def store(key, mask):
                m = mask & has_signal
                if m.sum() > 0:
                    results_by_quantile[key][f'pnl_{horizon_name}'].extend(net_pnl[m].tolist())

            store('all', np.ones(len(net_pnl), dtype=bool))
            store('eofi_agree', eofi_agrees)
            store('eofi_disagree', eofi_disagrees)
            store('top10_all', top10)
            store('top10_agree', top10 & eofi_agrees)
            store('top10_disagree', top10 & eofi_disagrees)
            store('top20_all', top20)
            store('top20_agree', top20 & eofi_agrees)
            store('top20_disagree', top20 & eofi_disagrees)

        if (di + 1) % 10 == 0:
            log.info(f"  Processed {di+1}/{len(overlap_dates)} dates...")

    # Compute summary statistics
    log.info("\n" + "=" * 90)
    log.info("CONFLUENCE ANALYSIS: CNN-Mamba v2 + EOFI Pressure Agreement")
    log.info("=" * 90)
    log.info(f"Dates analyzed: {len(overlap_dates)}")
    log.info(f"Cost model: {COMMISSION_RT_TICKS} ticks commission RT (taker)\n")

    summary_out = {}
    for key in ['all', 'eofi_agree', 'eofi_disagree',
                'top10_all', 'top10_agree', 'top10_disagree',
                'top20_all', 'top20_agree', 'top20_disagree']:
        log.info(f"--- {key} ---")
        for hz in ['1s', '5s', '10s']:
            pnls = np.array(results_by_quantile[key][f'pnl_{hz}'])
            if len(pnls) < 10:
                log.info(f"  {hz}: insufficient data ({len(pnls)} samples)")
                continue
            n = len(pnls)
            mean_pnl = np.mean(pnls)
            wr = np.mean(pnls > 0)
            total = np.sum(pnls)
            std = np.std(pnls)
            wins = np.sum(pnls[pnls > 0])
            losses = -np.sum(pnls[pnls <= 0])
            pf = wins / losses if losses > 0 else float('inf')

            log.info(f"  {hz}: n={n:,} mean={mean_pnl:+.4f}t WR={wr:.1%} "
                     f"PF={pf:.2f} total={total:+.1f}t")

            summary_out[f'{key}_{hz}'] = {
                'n': n, 'mean_pnl_ticks': float(mean_pnl), 'win_rate': float(wr),
                'profit_factor': float(pf), 'total_pnl_ticks': float(total),
                'std': float(std),
            }
        log.info("")

    # Key comparison
    log.info("=" * 90)
    log.info("KEY COMPARISONS (10s horizon, net of commission):")
    log.info("=" * 90)
    for prefix in ['', 'top10_', 'top20_']:
        label = prefix.replace('_', '') if prefix else 'all signals'
        agree = summary_out.get(f'{prefix}eofi_agree_10s', {})
        disagree = summary_out.get(f'{prefix}eofi_disagree_10s', {})
        all_sigs = summary_out.get(f'{prefix}all_10s' if prefix else 'all_10s', {})

        if agree and disagree and all_sigs:
            log.info(f"\n{label}:")
            log.info(f"  All:      mean={all_sigs['mean_pnl_ticks']:+.4f}t WR={all_sigs['win_rate']:.1%} n={all_sigs['n']:,}")
            log.info(f"  Agree:    mean={agree['mean_pnl_ticks']:+.4f}t WR={agree['win_rate']:.1%} n={agree['n']:,}")
            log.info(f"  Disagree: mean={disagree['mean_pnl_ticks']:+.4f}t WR={disagree['win_rate']:.1%} n={disagree['n']:,}")
            if agree['mean_pnl_ticks'] > disagree['mean_pnl_ticks']:
                lift = agree['mean_pnl_ticks'] - disagree['mean_pnl_ticks']
                log.info(f"  → EOFI agreement LIFT: +{lift:.4f} ticks/trade")
            else:
                log.info(f"  → NO lift from EOFI agreement")

    out_file = OUTPUT_DIR / "confluence_results.json"
    with open(out_file, 'w') as f:
        json.dump(summary_out, f, indent=2)
    log.info(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
