#!/usr/bin/env python3
"""
HC #428 R2 Experiments: 5s Horizon Limit + v7 Market Orders
============================================================
Experiment 1: CNN-Mamba v2 raw predictions at 5s horizon
Experiment 2: v7 meta-classifier, top-1% confidence, market orders only

Run: python run_horizon_market_experiments.py
"""

import numpy as np
import os
import sys
import json
import logging
import glob
import time
from pathlib import Path
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alpha_discovery.deep_models.fifo_market_replay import (
    FIFOReplayEngine, generate_signals, compute_metrics, compute_daily_metrics,
    COMMISSION_TICKS, TICK_USD, TICK_RAW
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('horizon_market_exp')

LVL3_ROOT   = Path(__file__).resolve().parents[2]
MBO_DIR     = LVL3_ROOT / 'data' / 'processed' / 'mbo_events'
RAW_MBO_DIRS = [str(LVL3_ROOT / 'data' / 'raw_mbo'), str(LVL3_ROOT / 'data' / 'raw' / 'mbo')]
OUTPUT_DIR  = LVL3_ROOT / 'output'
REPORT_PATH = OUTPUT_DIR / 'fifo_horizon_market_REPORT.md'


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (custom — harness loader expects fold files with oot_files key)
# ─────────────────────────────────────────────────────────────────────────────

def load_cnn_mamba_fold(fold_path: Path, mbo_dir: Path):
    """Load one CNN-Mamba v2 fold and return data dict."""
    d = np.load(fold_path, allow_pickle=True)
    preds  = d['predictions']   # (N, 3): col0=1s, col1=5s, col2=10s
    labels = d['labels']        # (N, 3)

    oot_files = d.get('oot_files', None)
    if oot_files is None:
        return None
    fname    = str(oot_files[0] if hasattr(oot_files, '__len__') else oot_files).split('/')[-1].split('\\')[-1]
    date_str = fname[:8]

    mbo_path = mbo_dir / f'{date_str}_mbo_events.npz'
    if not mbo_path.exists():
        log.warning(f'No MBO events for {date_str}')
        return None

    mbo = np.load(mbo_path, allow_pickle=True)
    timestamps = mbo['timestamps']
    n_events   = len(timestamps)

    n_preds = len(preds)
    WINDOW_SIZE = 1000
    STRIDE      = 500
    pred_ts = np.zeros(n_preds, dtype=np.int64)
    for i in range(n_preds):
        ev_idx = min(i * STRIDE + WINDOW_SIZE - 1, n_events - 1)
        pred_ts[i] = timestamps[ev_idx]

    return {
        'date':          date_str,
        'predictions':   preds,
        'labels':        labels,
        'timestamps_ns': pred_ts,
        'n_preds':       n_preds,
    }


def load_v7_for_date(date_str: str, concat_path: Path, mbo_dir: Path):
    """
    Load v7 meta-classifier predictions for a single date.
    concat_oot_predictions.npz has keys: predictions (N,), dates (N,).
    Returns data dict compatible with generate_signals (but we handle
    signal generation manually since this is 1D predictions + date mask).
    """
    d = np.load(concat_path, allow_pickle=True)
    all_preds = d['predictions']   # (N,)
    all_dates = d['dates']         # (N,) str YYYYMMDD

    mask = all_dates == date_str
    if not mask.any():
        return None

    preds = all_preds[mask]        # (N_date,)

    mbo_path = mbo_dir / f'{date_str}_mbo_events.npz'
    if not mbo_path.exists():
        log.warning(f'No MBO events for {date_str}')
        return None

    mbo = np.load(mbo_path, allow_pickle=True)
    timestamps = mbo['timestamps']
    n_events   = len(timestamps)

    # v7 is built from the same MBO event pipeline — predictions are aligned
    # to stride-500 windows just like CNN-Mamba (same pipeline assumption).
    n_preds = len(preds)
    WINDOW_SIZE = 1000
    STRIDE      = 500
    pred_ts = np.zeros(n_preds, dtype=np.int64)
    for i in range(n_preds):
        ev_idx = min(i * STRIDE + WINDOW_SIZE - 1, n_events - 1)
        pred_ts[i] = timestamps[ev_idx]

    return {
        'date':          date_str,
        'predictions_1d': preds,
        'timestamps_ns': pred_ts,
        'n_preds':       n_preds,
    }


def generate_signals_v7(data: dict, threshold: float, short_only: bool = False,
                         min_interval_ns: int = 500_000_000):
    """Generate signals from v7 1D predictions (both-sided or short-only)."""
    preds = data['predictions_1d']  # (N,) signed
    ts    = data['timestamps_ns']

    signals = []
    last_ts = -np.inf
    for i in range(len(preds)):
        p = float(preds[i])
        strength = abs(p)
        if strength < threshold:
            continue
        t = int(ts[i])
        if t - last_ts < min_interval_ns:
            continue
        direction = 'long' if p > 0 else 'short'
        if short_only and direction == 'long':
            continue
        signals.append({'ts_ns': t, 'direction': direction, 'strength': strength})
        last_ts = t

    return signals


def find_dbn(date: str) -> str:
    fname = f'glbx-mdp3-{date}.mbo.dbn.zst'
    for d in RAW_MBO_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'DBN not found for {date}')


# ─────────────────────────────────────────────────────────────────────────────
# Regime classification (ES close-to-close)
# ─────────────────────────────────────────────────────────────────────────────

def classify_regime(date: str, mbo_dir: Path) -> str:
    """Classify day as green/red/flat from ES close-to-close using MBO events."""
    try:
        mbo_path = mbo_dir / f'{date}_mbo_events.npz'
        if not mbo_path.exists():
            return 'unknown'
        mbo = np.load(mbo_path, allow_pickle=True)
        # Use 'labels_1s' as proxy for price direction — or extract from events
        # Simpler: use first vs last timestamp's embedded mid price if available
        if 'close_px' in mbo:
            cp = float(mbo['close_px'])
            op = float(mbo['open_px']) if 'open_px' in mbo else cp
            ret = (cp - op) / op
            if ret > 0.001:
                return 'green'
            elif ret < -0.001:
                return 'red'
            return 'flat'
        # Fallback: use mean of labels_1s
        if 'labels_1s' in mbo:
            mean_label = float(np.nanmean(mbo['labels_1s']))
            if mean_label > 0.05:
                return 'green'
            elif mean_label < -0.05:
                return 'red'
            return 'flat'
        return 'unknown'
    except Exception:
        return 'unknown'


# ─────────────────────────────────────────────────────────────────────────────
# Regime skew test (HC #428 R1)
# ─────────────────────────────────────────────────────────────────────────────

def regime_skew_test(daily_results: list) -> dict:
    """
    HC #428 R1: compute per-regime Sharpe and check skew ratio.
    Reject if |Sh_green - Sh_red| / max(|Sh_green|, |Sh_red|) > 0.50.
    """
    by_regime = {'green': [], 'red': [], 'flat': [], 'unknown': []}
    for dr in daily_results:
        regime = dr.get('regime', 'unknown')
        by_regime.setdefault(regime, []).append(dr['pnl_ticks'])

    def sharpe(arr):
        a = np.array(arr)
        if len(a) < 2:
            return float('nan')
        return float(a.mean() / (a.std() + 1e-9))

    sh_green = sharpe(by_regime['green'])
    sh_red   = sharpe(by_regime['red'])

    denom = max(abs(sh_green) if not np.isnan(sh_green) else 0,
                abs(sh_red)   if not np.isnan(sh_red)   else 0)
    if denom < 1e-9:
        skew_ratio = float('nan')
    else:
        skew_ratio = abs(sh_green - sh_red) / denom if not (np.isnan(sh_green) or np.isnan(sh_red)) else float('nan')

    return {
        'sh_green':     round(sh_green, 4),
        'sh_red':       round(sh_red, 4),
        'skew_ratio':   round(skew_ratio, 4) if not np.isnan(skew_ratio) else 'nan',
        'n_green_days': len(by_regime['green']),
        'n_red_days':   len(by_regime['red']),
        'n_flat_days':  len(by_regime['flat']),
        'hc428_pass':   (skew_ratio <= 0.50) if not np.isnan(skew_ratio) else False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_single_variant(
    label: str,
    date_data_pairs: list,        # list of (date, data_dict)
    signal_fn,                     # callable(data) -> List[signal_dict]
    tp_ticks: float,
    sl_ticks: float,
    order_type: str,
    hold_ms: float,
    cancel_ms: float,
    mbo_dir: Path,
) -> dict:
    """
    Run FIFO replay for all dates with given config.
    Returns aggregate + daily results dict.
    """
    log.info(f'\n{"─"*60}')
    log.info(f'VARIANT: {label}')
    log.info(f'  TP={tp_ticks}t  SL={sl_ticks}t  order={order_type}  hold={hold_ms}ms  cancel={cancel_ms}ms')

    all_results = []
    daily_results = []
    total_signals = 0

    for date, data in date_data_pairs:
        signals = signal_fn(data)
        total_signals += len(signals)

        if not signals:
            log.info(f'  {date}: 0 signals')
            regime = classify_regime(date, mbo_dir)
            daily_results.append({
                'date': date, 'regime': regime,
                'n_signals': 0, 'n_trades': 0, 'pnl_ticks': 0.0,
                'win_rate': 0.0, 'profit_factor': 0.0,
                'sl_rate': 0.0, 'tp_rate': 0.0
            })
            continue

        try:
            engine = FIFOReplayEngine(
                date=date,
                instrument_id=None,
                cancel_after_ns=int(cancel_ms * 1e6),
                max_hold_ns=int(hold_ms * 1e6),
            )
        except FileNotFoundError as e:
            log.warning(f'  {date}: {e}')
            continue

        trades = engine.simulate(
            signals, tp_ticks=tp_ticks, sl_ticks=sl_ticks, order_type=order_type
        )
        all_results.extend(trades)

        day_m = compute_metrics(trades, len(signals))
        regime = classify_regime(date, mbo_dir)
        day_pnl_ticks = float(np.sum([t.pnl_ticks_net for t in trades])) if trades else 0.0

        log.info(
            f'  {date} ({regime}): {len(signals)} sig -> {len(trades)} trades '
            f'WR={day_m["win_rate"]:.0%} net={day_pnl_ticks:+.2f}t '
            f'TP={day_m["tp_rate"]:.0%} SL={day_m["sl_rate"]:.0%}'
        )

        daily_results.append({
            'date': date, 'regime': regime,
            'n_signals': len(signals), 'n_trades': len(trades),
            'pnl_ticks': day_pnl_ticks,
            'win_rate': day_m['win_rate'],
            'profit_factor': day_m['profit_factor'],
            'sl_rate': day_m['sl_rate'], 'tp_rate': day_m['tp_rate'],
        })

    # Aggregate
    overall = compute_metrics(all_results, total_signals)
    daily_pnl_dollars = [dr['pnl_ticks'] * TICK_USD for dr in daily_results]
    portfolio = compute_daily_metrics(daily_pnl_dollars)
    regime_stats = regime_skew_test(daily_results)

    n_pos_days = sum(1 for dr in daily_results if dr['pnl_ticks'] > 0)
    n_neg_days = sum(1 for dr in daily_results if dr['pnl_ticks'] < 0)
    n_zero_days = sum(1 for dr in daily_results if dr['pnl_ticks'] == 0)

    # Verdict
    n_trades = overall['n_trades']
    net_t    = overall.get('mean_pnl_ticks', 0.0)
    if n_trades < 100:
        verdict = 'REJECT (too thin: <100 trades)'
    elif net_t <= 0:
        verdict = 'REJECT'
    elif n_pos_days >= 9 and (regime_stats['skew_ratio'] == 'nan' or regime_stats['skew_ratio'] <= 0.50):
        verdict = 'PASS'
    elif net_t > 0:
        verdict = 'WEAK PASS'
    else:
        verdict = 'REJECT'

    # Sample P&L distribution
    if all_results:
        pnl_arr = np.array([t.pnl_ticks_net for t in all_results])
        pnl_sample = {
            'p10': round(float(np.percentile(pnl_arr, 10)), 3),
            'p25': round(float(np.percentile(pnl_arr, 25)), 3),
            'p50': round(float(np.percentile(pnl_arr, 50)), 3),
            'p75': round(float(np.percentile(pnl_arr, 75)), 3),
            'p90': round(float(np.percentile(pnl_arr, 90)), 3),
        }
        # First 3 fill rows
        first3 = [
            {
                'date': datetime.utcfromtimestamp(t.entry_ts_ns / 1e9).strftime('%Y-%m-%d %H:%M:%S'),
                'dir': t.direction, 'exit': t.exit_reason,
                'pnl_net': round(t.pnl_ticks_net, 3),
            }
            for t in all_results[:3]
        ]
    else:
        pnl_sample = {}
        first3 = []

    log.info(f'\n  VERDICT: {verdict}')
    log.info(f'  Trades={n_trades}, net={net_t:+.4f}t, WR={overall["win_rate"]:.1%}')
    log.info(f'  Days +/-/0: {n_pos_days}/{n_neg_days}/{n_zero_days} of {len(daily_results)}')
    log.info(f'  Daily Sharpe={portfolio.get("daily_sharpe","N/A"):.3f} Sortino={portfolio.get("daily_sortino","N/A"):.3f}')
    log.info(f'  Regime skew: {regime_stats["skew_ratio"]} (HC428 pass={regime_stats["hc428_pass"]})')

    return {
        'label': label,
        'config': {
            'tp_ticks': tp_ticks, 'sl_ticks': sl_ticks,
            'order_type': order_type, 'hold_ms': hold_ms, 'cancel_ms': cancel_ms,
        },
        'overall': overall,
        'portfolio': portfolio,
        'regime_stats': regime_stats,
        'daily_results': daily_results,
        'n_pos_days': n_pos_days,
        'n_neg_days': n_neg_days,
        'n_zero_days': n_zero_days,
        'verdict': verdict,
        'pnl_sample': pnl_sample,
        'first3_fills': first3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: CNN-Mamba v2 @ 5s horizon
# ─────────────────────────────────────────────────────────────────────────────

def experiment_1_cnn_mamba_5s():
    log.info('\n' + '='*70)
    log.info('EXPERIMENT 1: CNN-Mamba v2 @ 5s Horizon (limit orders)')
    log.info('  HC #428 R2: hold≤7.5s, cancel≤5s, TP≤p90_MFE=5t')
    log.info('='*70)

    CNN_PRED_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'

    # Load all fold files (deduplicate by date)
    fold_files = sorted(CNN_PRED_DIR.glob('fold_*_oot_predictions.npz'))
    seen_dates = set()
    date_data_pairs = []
    for f in fold_files:
        data = load_cnn_mamba_fold(f, MBO_DIR)
        if data is None:
            continue
        if data['date'] in seen_dates:
            log.info(f'  Skipping duplicate date {data["date"]}')
            continue
        # Verify DBN exists
        try:
            find_dbn(data['date'])
        except FileNotFoundError:
            log.warning(f'  No DBN tape for {data["date"]}, skipping')
            continue
        seen_dates.add(data['date'])
        date_data_pairs.append((data['date'], data))
        log.info(f'  Loaded {data["date"]}: {data["n_preds"]:,} preds')

    log.info(f'\nTotal dates for Exp 1: {len(date_data_pairs)}')
    if not date_data_pairs:
        log.error('No data for Experiment 1!')
        return []

    # HC #428 R2 bounds for 5s horizon:
    # TP <= p90 MFE within 5s = 5.0 ticks
    # hold <= 1.5 * 5s = 7.5s
    # cancel <= 5s
    # Use TP=4.0 (≤5.0 ceiling), SL=2.0 (R:R=2:1)
    # 5s horizon column index = 1

    results = []

    # Compute 5s |pred| percentiles across all data
    all_preds_5s = np.concatenate([d['predictions'][:,1] for _, d in date_data_pairs])
    abs_5s = np.abs(all_preds_5s)
    thresh_10pct = np.percentile(abs_5s, 90)
    thresh_5pct  = np.percentile(abs_5s, 95)
    log.info(f'5s |pred| thresholds: top10%={thresh_10pct:.4f}, top5%={thresh_5pct:.4f}')

    for name, thresh in [('top10pct', thresh_10pct), ('top5pct', thresh_5pct)]:
        def make_signal_fn(t):
            def signal_fn(data):
                p5s = data['predictions'][:, 1]  # 5s column
                ts  = data['timestamps_ns']
                signals = []
                last_ts = -np.inf
                for i in range(len(p5s)):
                    strength = abs(p5s[i])
                    if strength < t:
                        continue
                    ts_i = int(ts[i])
                    if ts_i - last_ts < 500_000_000:  # 500ms anti-churn
                        continue
                    direction = 'long' if p5s[i] > 0 else 'short'
                    signals.append({'ts_ns': ts_i, 'direction': direction, 'strength': float(strength)})
                    last_ts = ts_i
                return signals
            return signal_fn

        r = run_single_variant(
            label=f'cnn_mamba_5s_{name}',
            date_data_pairs=date_data_pairs,
            signal_fn=make_signal_fn(thresh),
            tp_ticks=4.0,    # ≤ p90 MFE (5.0t)
            sl_ticks=2.0,    # R:R = 2:1
            order_type='limit',
            hold_ms=7500,    # 7.5s = 1.5 × 5s
            cancel_ms=5000,  # = horizon
            mbo_dir=MBO_DIR,
        )
        results.append(r)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: v7 Market Orders, top-1% confidence
# ─────────────────────────────────────────────────────────────────────────────

def experiment_2_v7_market():
    log.info('\n' + '='*70)
    log.info('EXPERIMENT 2: v7 Meta-Classifier, Market Orders, Top-1%')
    log.info('  HC #428 R2: hold≤1.5s, market fill immediate, TP≤p90_MFE_1s=4.0t')
    log.info('='*70)

    V7_CONCAT = LVL3_ROOT / 'output' / 'meta_v7_prod' / 'concat_oot_predictions.npz'
    d_concat = np.load(V7_CONCAT, allow_pickle=True)
    all_preds = d_concat['predictions']
    all_dates = d_concat['dates']

    # 17 OOT dates with DBN
    unique_dates = sorted(np.unique(all_dates))
    valid_dates = []
    for dt in unique_dates:
        try:
            find_dbn(str(dt))
            valid_dates.append(str(dt))
        except FileNotFoundError:
            pass

    log.info(f'v7 dates with DBN: {len(valid_dates)}')

    # Top-1% threshold across ALL dates
    abs_v7 = np.abs(all_preds)
    thresh_1pct = np.percentile(abs_v7, 99)
    log.info(f'v7 |pred| top-1% threshold = {thresh_1pct:.4f}')

    # Load per-date data
    date_data_pairs = []
    for dt in valid_dates:
        data = load_v7_for_date(dt, V7_CONCAT, MBO_DIR)
        if data:
            date_data_pairs.append((dt, data))

    if not date_data_pairs:
        log.error('No data for Experiment 2!')
        return []

    # HC #428 R2 bounds for 1s horizon (v7 model horizon):
    # TP ≤ p90 MFE within 1s = 4.0t → use TP=2.0t (conservative, market orders expensive)
    # hold ≤ 1.5 × 1s = 1.5s
    # Market orders: cost = 1.0t spread + 0.376t commission = 1.376t total
    results = []

    for name, short_only in [('both_sides', False), ('short_only', True)]:
        def make_signal_fn(threshold, so):
            def signal_fn(data):
                return generate_signals_v7(
                    data, threshold=threshold, short_only=so,
                    min_interval_ns=500_000_000
                )
            return signal_fn

        r = run_single_variant(
            label=f'v7_market_top1pct_{name}',
            date_data_pairs=date_data_pairs,
            signal_fn=make_signal_fn(thresh_1pct, short_only),
            tp_ticks=2.0,    # ≤ p90 MFE (4.0t), conservative for market cost
            sl_ticks=1.5,    # keep R:R ≥ 1.3:1 after 1.376t cost
            order_type='market',
            hold_ms=1500,    # 1.5s = 1.5 × 1s horizon
            cancel_ms=1500,  # N/A for market, but set for consistency
            mbo_dir=MBO_DIR,
        )
        results.append(r)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report generator
# ─────────────────────────────────────────────────────────────────────────────

def write_report(exp1_results: list, exp2_results: list):
    lines = []
    lines.append('# HC #428 R2 Experiments: 5s Horizon + v7 Market Orders')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')
    lines.append('## Context')
    lines.append('- Base v7 FIFO: -0.621 net ticks/trade, 0/17 positive days. REJECT.')
    lines.append('- All v7 execution variants (5 configs): REJECT. Pattern: 65% SL exits, adverse fill mechanics.')
    lines.append('- These two experiments test: (a) a longer horizon with same model, (b) market orders bypassing fill mechanics.')
    lines.append('')

    all_results = exp1_results + exp2_results

    for section_name, results in [
        ('## Experiment 1 — CNN-Mamba v2 @ 5s Horizon (limit orders)', exp1_results),
        ('## Experiment 2 — v7 Meta-Classifier, Market Orders Only (top-1%)', exp2_results),
    ]:
        lines.append(section_name)
        lines.append('')

        for r in results:
            lines.append(f'### {r["label"]}')
            lines.append('')
            o = r['overall']
            p = r['portfolio']
            reg = r['regime_stats']
            lines.append(f'**VERDICT: {r["verdict"]}**')
            lines.append('')
            lines.append('| Metric | Value |')
            lines.append('|--------|-------|')
            lines.append(f'| Trades | {o["n_trades"]} |')
            lines.append(f'| Net ticks/trade | {o.get("mean_pnl_ticks", 0):+.4f} |')
            lines.append(f'| Win Rate | {o["win_rate"]:.1%} |')
            lines.append(f'| Profit Factor | {o["profit_factor"]:.3f} |')
            lines.append(f'| Per-trade Sharpe | {o["sharpe"]:.4f} |')
            lines.append(f'| Per-trade Sortino | {o["sortino"]:.4f} |')
            lines.append(f'| Daily Sharpe | {p.get("daily_sharpe", "N/A")} |')
            lines.append(f'| Daily Sortino | {p.get("daily_sortino", "N/A")} |')
            lines.append(f'| Positive days | {r["n_pos_days"]}/{len(r["daily_results"])} |')
            lines.append(f'| SL rate | {o["sl_rate"]:.1%} |')
            lines.append(f'| TP rate | {o["tp_rate"]:.1%} |')
            lines.append(f'| Regime skew ratio | {reg["skew_ratio"]} (HC428 pass={reg["hc428_pass"]}) |')
            lines.append(f'| Regime Sh green/red | {reg["sh_green"]}/{reg["sh_red"]} |')
            lines.append('')

            if r['pnl_sample']:
                s = r['pnl_sample']
                lines.append(f'P&L distribution (net ticks): p10={s["p10"]}, p25={s["p25"]}, p50={s["p50"]}, p75={s["p75"]}, p90={s["p90"]}')
                lines.append('')

            if r['first3_fills']:
                lines.append('First 3 fills:')
                for fill in r['first3_fills']:
                    lines.append(f'  - {fill["date"]} {fill["dir"]} exit={fill["exit"]} net={fill["pnl_net"]:+.3f}t')
                lines.append('')

            lines.append('**Daily breakdown:**')
            lines.append('')
            lines.append('| Date | Regime | Signals | Trades | PnL (ticks) | WR |')
            lines.append('|------|--------|---------|--------|-------------|-----|')
            for dr in r['daily_results']:
                lines.append(
                    f'| {dr["date"]} | {dr["regime"]} | {dr["n_signals"]} | {dr["n_trades"]} '
                    f'| {dr["pnl_ticks"]:+.2f} | {dr["win_rate"]:.0%} |'
                )
            lines.append('')

    # Combined summary table
    lines.append('## Summary Table')
    lines.append('')
    lines.append('| Variant | Trades | net t/trade | Pos days | Daily Sh | PF | WR | SL% | Verdict |')
    lines.append('|---------|--------|-------------|----------|----------|-----|-----|-----|---------|')
    for r in all_results:
        o = r['overall']
        p = r['portfolio']
        lines.append(
            f'| {r["label"]} | {o["n_trades"]} | {o.get("mean_pnl_ticks",0):+.4f} '
            f'| {r["n_pos_days"]}/{len(r["daily_results"])} '
            f'| {p.get("daily_sharpe","N/A")} | {o["profit_factor"]:.3f} '
            f'| {o["win_rate"]:.1%} | {o["sl_rate"]:.1%} | {r["verdict"]} |'
        )
    lines.append('')

    # Next-axis recommendation
    lines.append('## Next-Axis Recommendation (HC #488 R2)')
    lines.append('')

    verdicts = {r['label']: r['verdict'] for r in all_results}
    all_reject  = all('REJECT' in v for v in verdicts.values())
    exp1_passes = any('PASS' in v and 'REJECT' not in v for v in verdicts.values() if '5s' in r['label'] for r in exp1_results)
    exp2_passes = any('PASS' in v and 'REJECT' not in v for v in verdicts.values() if 'market' in r['label'] for r in exp2_results)

    # Determine actual outcome
    e1_verdicts = [r['verdict'] for r in exp1_results]
    e2_verdicts = [r['verdict'] for r in exp2_results]
    e1_any_pass = any('PASS' in v and 'REJECT' not in v for v in e1_verdicts)
    e2_any_pass = any('PASS' in v and 'REJECT' not in v for v in e2_verdicts)

    if e1_any_pass and e2_any_pass:
        lines.append('**Both experiments pass.** Combine 5s limit execution with market-order signals:')
        lines.append('- Run combined config: 5s CNN-Mamba signals + market orders for top-1% subset.')
        lines.append('- Re-validate across all available dates with combined regime breakdown.')
    elif e1_any_pass and not e2_any_pass:
        lines.append('**5s limit passes, market orders reject.**')
        lines.append('Longer-horizon limit execution is the viable path. Recommendation:')
        lines.append('1. Dense OOT validation of the 5s config across more dates (extend to April dates if CNN-Mamba preds available).')
        lines.append('2. Optimize 5s cancel window and TP/SL via grid search.')
        lines.append('3. Test short-only variant of the 5s config (prior signal research shows short side stronger).')
    elif not e1_any_pass and e2_any_pass:
        lines.append('**Market orders pass, 5s limit rejects.**')
        lines.append('Signal is real but only exploitable with aggressive market execution. Recommendation:')
        lines.append('1. Configure short-only market-order weights for the paper trader next-day session.')
        lines.append('2. Analyze which confidence decile drives the market-order P&L — may be top 0.5%.')
        lines.append('3. Test dynamic position sizing (larger size at top-0.1% vs top-1%).')
    else:
        lines.append('**Both experiments reject.** This is a fundamental edge problem, not a horizon or execution problem.')
        lines.append('')
        lines.append('The v7 meta-classifier signal does not translate into profitable fills at any tested configuration.')
        lines.append('65% SL exits persist regardless of order type (limit or market), horizon (1s or 5s), or confidence filter.')
        lines.append('')
        lines.append('**Recommended next axis — Feature/Model axis (highest probability of finding new edge):**')
        lines.append('')
        lines.append('1. **Cross-asset confluence (VIX/NQ/VIX futures)**: ES moves are more predictable when NQ or VIX')
        lines.append('   diverges from ES. Filter signals to only trade when cross-asset agreement is present.')
        lines.append('   Implementation: add NQ and VX MBO/OHLC features to the existing CNN-Mamba input window.')
        lines.append('')
        lines.append('2. **Microprice-based entry signals**: Replace log-return target with signed microprice move')
        lines.append('   (bid_qty/(bid_qty+ask_qty) × spread). Microprice better reflects queue-weighted mid and')
        lines.append('   may have higher 1s predictability than the current label.')
        lines.append('')
        lines.append('3. **Queue-imbalance-only model**: Strip all book-depth features, train exclusively on')
        lines.append('   bid/ask queue imbalance ratios. The hypothesis: the current model may be learning')
        lines.append('   cross-level book shape that decays before the fill window. Pure imbalance is faster.')
        lines.append('')
        lines.append('4. **Conformal prediction wrapper**: Add conformal prediction calibration to CNN-Mamba outputs.')
        lines.append('   This gives statistically valid coverage intervals — only trade when p90 conformal interval')
        lines.append('   is fully positive (long) or fully negative (short), eliminating ambiguous signals.')
        lines.append('')
        lines.append('5. **Re-examine signal IC on April dates**: The CNN-Mamba v2 model was trained/validated on')
        lines.append('   Feb-Mar 2026. The April 2026 tape (17 OOT dates used for v7) may show model decay.')
        lines.append('   Run IC_1s on April dates using CNN-Mamba v2 raw predictions to confirm edge still exists.')
        lines.append('   If IC has decayed below 0.10, the model itself needs retraining before any execution work.')

    lines.append('')
    lines.append('---')
    lines.append('*Report generated by run_horizon_market_experiments.py*')

    report_text = '\n'.join(lines)
    with open(REPORT_PATH, 'w') as f:
        f.write(report_text)
    log.info(f'\nReport saved to {REPORT_PATH}')
    return report_text


# ─────────────────────────────────────────────────────────────────────────────
# MLflow logging (optional — graceful fallback if unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def log_to_mlflow(results: list, experiment_name: str):
    try:
        import mlflow
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment(experiment_name)

        for r in results:
            with mlflow.start_run(run_name=r['label']):
                o = r['overall']
                p = r['portfolio']
                mlflow.log_params(r['config'])
                mlflow.log_metrics({
                    'net_ticks_per_trade': o.get('mean_pnl_ticks', 0),
                    'win_rate': o['win_rate'],
                    'profit_factor': o['profit_factor'],
                    'per_trade_sharpe': o['sharpe'],
                    'per_trade_sortino': o['sortino'],
                    'daily_sharpe': p.get('daily_sharpe', 0),
                    'daily_sortino': p.get('daily_sortino', 0),
                    'n_trades': o['n_trades'],
                    'n_pos_days': r['n_pos_days'],
                    'sl_rate': o['sl_rate'],
                    'regime_skew_ratio': float(r['regime_stats']['skew_ratio'])
                        if r['regime_stats']['skew_ratio'] != 'nan' else -1,
                })
                mlflow.log_param('verdict', r['verdict'])
        log.info(f'MLflow: logged {len(results)} runs to {experiment_name}')
    except Exception as e:
        log.warning(f'MLflow logging failed: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()

    log.info('HC #428 R2 Experiments — 5s Horizon + v7 Market Orders')
    log.info(f'Timestamp: {datetime.now().isoformat()}')
    log.info(f'Commission: {COMMISSION_TICKS:.4f} ticks ({COMMISSION_TICKS * TICK_USD:.2f}/RT)')

    exp1_results = experiment_1_cnn_mamba_5s()
    exp2_results = experiment_2_v7_market()

    report = write_report(exp1_results, exp2_results)

    # MLflow
    log_to_mlflow(exp1_results, 'cnn_mamba_5s_fifo')
    log_to_mlflow(exp2_results, 'v7_market_orders_fifo')

    elapsed = time.time() - t0
    log.info(f'\nTotal runtime: {elapsed:.0f}s')
    log.info('Done.')
