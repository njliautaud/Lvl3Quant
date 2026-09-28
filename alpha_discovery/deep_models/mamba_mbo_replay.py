#!/usr/bin/env python3
"""
MAMBA v7 — REAL MBO Order Book Replay
======================================
NO MORE MID-PRICE PNL.

Uses the actual Databento MBO data to:
1. Reconstruct the full FIFO order book event by event
2. Place simulated limit orders at the best bid/ask
3. Track queue position, fill probability, adverse selection
4. Compute REAL PnL after spread, commission, queue, slippage

Signal source: Mamba v7 OOT predictions (top confidence tiers)
Order types: limit (passive), market (aggressive), chase (reprice)

ES futures: tick=$12.50, commission=$4.70 RT = 0.376 ticks
"""

import numpy as np
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# Import the existing MBO replay engine
sys.path.insert(0, str(Path(__file__).parent))
from mbo_replay_server import run_fill_sim, TICK_USD, COMMISSION_TICKS

# ── Config ──────────────────────────────────────────────────────────────────
PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/mamba_v7_tiny_smart_v3_mar_apr")
DATA_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
RESULTS_DIR = Path("/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ES instrument IDs (changes with contract roll)
ES_INSTRUMENT_IDS = {
    # Try common ES instrument IDs — the replay engine will filter by this
    '20260301': 42140878,  # ESH6
    '20260302': 42140878,
    '20260303': 42140878,
    '20260304': 42140878,
    '20260305': 42140878,
    '20260306': 42140878,
    '20260308': 42140878,
    '20260309': 42140878,
    '20260310': 42140878,
    '20260311': 42140878,
    '20260312': 42140878,
    '20260313': 42140878,
}
DEFAULT_INSTR_ID = 42140878  # ESH6

# Prediction windowing params (must match training)
WINDOW_SIZE = 1000
STRIDE = 500


def load_predictions_with_timestamps(fold_path, data_dir):
    """
    Load OOT predictions and map each to its nanosecond timestamp.

    Each prediction i was made on a window of events [i*stride : i*stride + window].
    The prediction corresponds to the LAST event in the window: event[i*stride + window - 1].
    """
    d = np.load(fold_path, allow_pickle=True)
    preds = d['predictions']   # (N, 3)
    labels = d['labels']       # (N, 3)
    embeds = d.get('embeddings', None)

    # Get OOT date
    oot_files = d.get('oot_files', None)
    if oot_files is not None:
        if hasattr(oot_files, 'item'):
            oot_files = oot_files.item()
        if isinstance(oot_files, (list, np.ndarray)) and len(oot_files) > 0:
            fname = str(oot_files[0]).split('/')[-1].split('\\')[-1]
        else:
            fname = str(oot_files).split('/')[-1].split('\\')[-1]
        date_str = fname[:8]
    else:
        return None

    # Load timestamps from smart_v3 data
    sv3_path = data_dir / f"{date_str}_mbo_events.npz"
    if not sv3_path.exists():
        log.warning(f"  No smart_v3 file for {date_str}")
        return None

    sv3 = np.load(sv3_path, allow_pickle=True)
    timestamps = sv3['timestamps']
    n_events = len(timestamps)

    # Map prediction index to timestamp
    # Prediction i → event at index min(i * STRIDE + WINDOW - 1, n_events - 1)
    n_preds = len(preds)
    pred_timestamps = np.zeros(n_preds, dtype=np.int64)
    for i in range(n_preds):
        event_idx = min(i * STRIDE + WINDOW_SIZE - 1, n_events - 1)
        pred_timestamps[i] = timestamps[event_idx]

    return {
        'date': date_str,
        'preds': preds,
        'labels': labels,
        'embeddings': embeds,
        'timestamps_ns': pred_timestamps,
        'n_preds': n_preds,
        'n_events': n_events,
    }


def select_signals(data, strategy='top1pct', horizon=0):
    """
    Select trade signals based on confidence strategy.
    Returns list of {'ts_ns': int, 'direction': 'long'|'short'} dicts.
    """
    preds = data['preds']
    timestamps = data['timestamps_ns']

    strength = np.abs(preds[:, 0])  # Use 1s prediction for signal strength

    if strategy == 'top1pct':
        thresh = np.percentile(strength, 99)
    elif strategy == 'top05pct':
        thresh = np.percentile(strength, 99.5)
    elif strategy == 'top5pct':
        thresh = np.percentile(strength, 95)
    elif strategy == 'top10pct':
        thresh = np.percentile(strength, 90)
    elif strategy == 'agree_top1':
        agree = (np.sign(preds[:, 0]) == np.sign(preds[:, 1])) & \
                (np.sign(preds[:, 1]) == np.sign(preds[:, 2]))
        thresh = np.percentile(strength, 99)
        mask = agree & (strength >= thresh)
        indices = np.where(mask)[0]
        signals = []
        for idx in indices:
            direction = 'long' if preds[idx, horizon] > 0 else 'short'
            signals.append({'ts_ns': int(timestamps[idx]), 'direction': direction})
        return signals
    else:
        thresh = np.percentile(strength, 99)

    mask = strength >= thresh
    indices = np.where(mask)[0]

    signals = []
    for idx in indices:
        direction = 'long' if preds[idx, horizon] > 0 else 'short'
        signals.append({'ts_ns': int(timestamps[idx]), 'direction': direction})

    return signals


def run_mbo_replay():
    """Run Mamba predictions through real MBO order book replay."""
    log.info("=" * 80)
    log.info("MAMBA v7 — REAL MBO ORDER BOOK REPLAY")
    log.info("=" * 80)
    log.info("NO MID-PRICE PNL. Using actual bid/ask, FIFO queue, fill probability.")
    log.info(f"ES: tick=$12.50, commission=$4.70 RT ({COMMISSION_TICKS:.3f} ticks)")
    log.info("")

    # Load all folds
    fold_files = sorted(PRED_DIR.glob("fold_*_oot_predictions.npz"))
    if not fold_files:
        log.error(f"No prediction files in {PRED_DIR}")
        return

    all_data = []
    for f in fold_files:
        data = load_predictions_with_timestamps(f, DATA_DIR)
        if data:
            all_data.append(data)
            log.info(f"  {data['date']}: {data['n_preds']:,} predictions from {data['n_events']:,} events")

    if not all_data:
        log.error("No valid data loaded")
        return

    log.info(f"\nLoaded {len(all_data)} days with predictions")

    # Strategy x order_type x TP/SL combos
    strategies = ['top1pct', 'top05pct', 'top5pct', 'agree_top1']
    order_types = ['limit', 'market', 'chase']
    tp_sl_combos = [
        (2.0, 1.0, '2:1'),     # 2 tick TP, 1 tick SL
        (4.0, 2.0, '4:2'),     # 4 tick TP, 2 tick SL
        (2.0, 2.0, '2:2'),     # Symmetric
        (1.0, 1.0, '1:1'),     # Tight
        (8.0, 4.0, '8:4'),     # Wide
    ]

    all_results = {}

    for strat in strategies:
        for otype in order_types:
            for tp, sl, ratio_name in tp_sl_combos:
                key = f"{strat}_{otype}_{ratio_name}"
                log.info(f"\n{'='*60}")
                log.info(f"Strategy: {strat} | Order: {otype} | TP/SL: {ratio_name}")
                log.info(f"{'='*60}")

                day_results = []
                total_signals = 0
                total_fills = 0
                total_pnl_ticks = 0

                for data in all_data:
                    date = data['date']
                    instr_id = ES_INSTRUMENT_IDS.get(date, DEFAULT_INSTR_ID)

                    signals = select_signals(data, strat, horizon=0)
                    total_signals += len(signals)

                    if not signals:
                        log.info(f"  {date}: 0 signals")
                        continue

                    try:
                        result = run_fill_sim(
                            date=date,
                            instrument_id=instr_id,
                            signal_ts_ns=[s['ts_ns'] for s in signals],
                            directions=[s['direction'] for s in signals],
                            tp_ticks=tp,
                            sl_ticks=sl,
                            order_type=otype,
                            cancel_after_ns=30_000_000_000,  # 30s cancel
                            max_reprices=3,
                            reprice_after_ns=1_000_000_000,  # 1s reprice for chase
                        )

                        c = result['combined']
                        n_fills = c['n']
                        total_fills += n_fills

                        if n_fills > 0:
                            day_pnl = c['mean_pnl_net_ticks'] * n_fills
                            total_pnl_ticks += day_pnl

                            log.info(f"  {date}: {len(signals)} signals → {n_fills} fills "
                                    f"({c['fill_rate']:.1%} fill rate) | "
                                    f"WR={c['win_rate']:.1%} | "
                                    f"PnL={day_pnl:.1f} ticks (${day_pnl * TICK_USD:.0f}) | "
                                    f"TP={c['tp_rate']:.0%} SL={c['sl_rate']:.0%} TO={c['timeout_rate']:.0%} | "
                                    f"Avg queue={c['mean_queue_ahead']:.0f} wait={c['mean_queue_wait_ms']:.0f}ms "
                                    f"slip={c['mean_slippage_ticks']:.2f}t")
                        else:
                            log.info(f"  {date}: {len(signals)} signals → 0 fills")

                        day_results.append({
                            'date': date,
                            'n_signals': len(signals),
                            'long': result['long'],
                            'short': result['short'],
                            'combined': result['combined'],
                        })

                    except FileNotFoundError as e:
                        log.warning(f"  {date}: {e}")
                    except Exception as e:
                        log.error(f"  {date}: Error — {e}")

                # Aggregate
                fill_rate = total_fills / max(1, total_signals)
                pnl_dollars = total_pnl_ticks * TICK_USD

                all_results[key] = {
                    'strategy': strat,
                    'order_type': otype,
                    'tp_sl': ratio_name,
                    'total_signals': total_signals,
                    'total_fills': total_fills,
                    'fill_rate': fill_rate,
                    'total_pnl_ticks': total_pnl_ticks,
                    'total_pnl_dollars': pnl_dollars,
                    'daily': day_results,
                }

                log.info(f"\n  TOTAL: {total_signals} signals → {total_fills} fills "
                        f"({fill_rate:.1%}) | PnL: {total_pnl_ticks:.1f} ticks "
                        f"(${pnl_dollars:,.0f})")

    # ── Summary table ───────────────────────────────────────────────────────
    log.info(f"\n\n{'='*120}")
    log.info("REAL MBO REPLAY — STRATEGY COMPARISON (actual fills, queue, spread, commission)")
    log.info(f"{'='*120}")
    log.info(f"{'Strategy':<20} {'Order':<7} {'TP/SL':<6} {'Signals':>8} {'Fills':>7} "
             f"{'Fill%':>6} {'PnL($)':>10} {'$/Fill':>8} {'$/Signal':>9}")
    log.info("-" * 120)

    # Sort by PnL
    sorted_keys = sorted(all_results.keys(),
                         key=lambda k: all_results[k]['total_pnl_dollars'],
                         reverse=True)

    for key in sorted_keys:
        r = all_results[key]
        per_fill = r['total_pnl_dollars'] / max(1, r['total_fills'])
        per_signal = r['total_pnl_dollars'] / max(1, r['total_signals'])
        log.info(f"{r['strategy']:<20} {r['order_type']:<7} {r['tp_sl']:<6} "
                f"{r['total_signals']:>8} {r['total_fills']:>7} "
                f"{r['fill_rate']:>5.1%} ${r['total_pnl_dollars']:>9,.0f} "
                f"${per_fill:>7.2f} ${per_signal:>8.2f}")

    log.info(f"{'='*120}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"mamba_mbo_replay_{ts}.json"

    # Make JSON-serializable (remove raw_results)
    save_data = {}
    for k, v in all_results.items():
        sv = dict(v)
        sv['daily'] = [{kk: vv for kk, vv in d.items()
                        if kk != 'raw_results'} for d in sv['daily']]
        save_data[k] = sv

    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")


if __name__ == '__main__':
    run_mbo_replay()
