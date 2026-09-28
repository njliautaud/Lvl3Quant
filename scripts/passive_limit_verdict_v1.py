#!/usr/bin/env python3
"""
Passive Limit Order Fill-Rate & P&L Verdict — v1
=================================================
Key question: For top-1% confidence CNN-Mamba v2 short signals, how often would
passive limit sell orders at the ask actually fill? And what's the net P&L?

Uses CONSOLIDATED predictions (hc417_v2_full_oot_56d.npz) — single best checkpoint,
IC_1s=0.237. Maps to MBO event data for 46 overlapping dates.

Prediction mapping: offset=999, stride=250 events.

Entry model:
  - Place sell limit AT the ask. Fill if price touches our ask within cancel_window.
  - Using MBO 1s/5s/10s/30s labels to detect if price goes UP from current level.
  - CONSERVATIVE: require label >= +0.5 tick (clear uptick to our level).
  - MODERATE: require label >= 0 (flat = still active at our level).

Exit model after entry fill:
  - Place buy limit at bid. Track price path using chained 1s labels.
  - Passive exit: mid drops >= 1.0 tick from entry (conservative FIFO fill).
  - Market exit at hold timeout: P&L = -(mid change) - 0.376 commission.

Cost: 0.376 ticks RT (passive fills, no spread cost per HC #512).

Output: /home/jupiter/Lvl3Quant/output/passive_limit_verdict_v1.json
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
CONS_FILE = ROOT / "output/hc417_v2_full_oot_56d.npz"
MBO_DIR = ROOT / "data/processed/mbo_events_smart_v3"
OUT_FILE = ROOT / "output/passive_limit_verdict_v1.json"

# ── Constants ────────────────────────────────────────────────────────────────
PRED_OFFSET = 999       # window_size - 1
PRED_STRIDE = 250       # Events between consecutive predictions
COMMISSION_TICKS = 0.376
TICK_VALUE_USD = 12.50
MIN_SIGNAL_GAP_S = 5.0  # Minimum gap between signals

# ── Sweep parameters ────────────────────────────────────────────────────────
CONFIDENCE_PCTS = [1, 2, 5]
CANCEL_WINDOWS_S = [5, 10, 30]
HOLD_TIMEOUTS_S = [10, 30, 60]

# Entry fill modes: minimum uptick in labels for fill
ENTRY_MODES = {
    "conservative": 0.5,   # Need +0.5 tick move (price clearly at our ask)
    "moderate": 0.0,       # Need >= 0 (flat = trading at our level)
    "aggressive": -1.0,    # Even mild drops have intra-bar upticks
}

# Passive exit: need mid to drop this much for FIFO bid fill
PASSIVE_EXIT_DROP = 1.0  # ticks


def flush(msg):
    print(msg, flush=True)


def load_consolidated():
    """Load consolidated predictions and split by date."""
    cons = np.load(CONS_FILE, allow_pickle=True)

    pred_1s = cons['pred_log_ret_1s']
    target_1s = cons['target_log_ret_1s']
    mask_1s = cons['mask_log_ret_1s']
    day_index = cons['day_index']
    oot_dates = list(cons['oot_dates'])
    per_day_n = cons['per_day_n_windows']

    flush(f"Consolidated: {len(pred_1s):,} samples, IC_1s={float(cons['metric_ic_log_ret_1s']):.4f}")
    flush(f"Dates: {len(oot_dates)} OOT, {sum(per_day_n > 0)} with data")

    # Split by date
    days = {}
    cum = 0
    for i, dt in enumerate(oot_dates):
        n = int(per_day_n[i])
        if n == 0:
            continue

        sl = slice(cum, cum + n)
        days[dt] = {
            'date': dt,
            'preds': pred_1s[sl],
            'targets': target_1s[sl],
            'masks': mask_1s[sl],
            'n_preds': n,
        }
        cum += n

    return days


def load_mbo_for_date(date_str):
    """Load MBO event data for a date."""
    event_file = MBO_DIR / f"{date_str}_mbo_events.npz"
    if not event_file.exists():
        return None
    events = np.load(event_file)
    return {
        'timestamps': events['timestamps'],
        'labels_1s': events['labels_1s'],
        'labels_5s': events['labels_5s'],
        'labels_10s': events['labels_10s'],
        'labels_30s': events['labels_30s'],
        'n_events': len(events['timestamps']),
    }


def check_entry_fill(l1, l5, l10, l30, cancel_s, min_uptick):
    """Check if passive sell limit at ask would fill within cancel_window."""
    if l1 >= min_uptick:
        return True, 0.5
    if cancel_s >= 5 and l5 >= min_uptick:
        return True, 2.5
    if cancel_s >= 10 and l10 >= min_uptick:
        return True, 5.0
    if cancel_s >= 30 and l30 >= min_uptick:
        return True, 15.0
    return False, 0.0


def simulate_exit(mbo, ev_idx, fill_delay_s, hold_timeout_s):
    """Simulate passive/market exit after entry fill."""
    timestamps = mbo['timestamps']
    labels_1s = mbo['labels_1s']
    n_events = mbo['n_events']

    fill_t_ns = timestamps[ev_idx] + int(fill_delay_s * 1e9)
    fill_ev = min(int(np.searchsorted(timestamps, fill_t_ns)), n_events - 1)

    cum_price = 0.0
    min_price = 0.0
    exit_type = 'market'
    exit_time_s = hold_timeout_s
    exit_price_change = 0.0

    cur_ev = fill_ev
    for s in range(1, int(hold_timeout_s) + 1):
        if cur_ev >= n_events:
            break
        l1_val = labels_1s[cur_ev]
        if not np.isnan(l1_val):
            cum_price += l1_val

        if cum_price < min_price:
            min_price = cum_price

        if cum_price <= -PASSIVE_EXIT_DROP:
            exit_type = 'passive'
            exit_time_s = s
            exit_price_change = cum_price
            break

        target_t = fill_t_ns + int(s * 1e9)
        cur_ev = min(int(np.searchsorted(timestamps, target_t)), n_events - 1)

    if exit_type == 'market':
        exit_price_change = cum_price

    if exit_type == 'passive':
        pnl_ticks = 1.0 - COMMISSION_TICKS  # +0.624
    else:
        pnl_ticks = -exit_price_change - COMMISSION_TICKS

    return {
        'exit_type': exit_type,
        'pnl_ticks': float(pnl_ticks),
        'exit_price_change': float(exit_price_change),
        'exit_time_s': exit_time_s,
        'mfe_ticks': float(-min_price),
    }


def simulate_day(day_data, mbo, pred_threshold, cancel_s, hold_s, entry_mode_thresh):
    """Simulate all trades for one day."""
    preds = day_data['preds']
    masks = day_data['masks']
    n_preds = day_data['n_preds']

    # Map to event indices
    event_indices = PRED_OFFSET + np.arange(n_preds) * PRED_STRIDE
    valid_event = event_indices < mbo['n_events']

    # Get labels at prediction points
    timestamps = mbo['timestamps']
    labels_1s = mbo['labels_1s']
    labels_5s = mbo['labels_5s']
    labels_10s = mbo['labels_10s']
    labels_30s = mbo['labels_30s']

    # Only use valid predictions (mask > 0) with valid event mapping
    use_mask = (masks > 0) & valid_event
    use_indices = np.where(use_mask)[0]

    if len(use_indices) == 0:
        return [], 0, 0

    use_preds = preds[use_indices]
    use_ev_idx = event_indices[use_indices]

    # Get MBO labels at prediction events
    pred_l1 = np.nan_to_num(labels_1s[use_ev_idx], nan=0.0)
    pred_l5 = np.nan_to_num(labels_5s[use_ev_idx], nan=0.0)
    pred_l10 = np.nan_to_num(labels_10s[use_ev_idx], nan=0.0)
    pred_l30 = np.nan_to_num(labels_30s[use_ev_idx], nan=0.0)
    pred_ts = timestamps[use_ev_idx]

    # Find short signals (most negative predictions)
    signal_mask = use_preds <= pred_threshold
    signal_local_indices = np.where(signal_mask)[0]
    n_raw_signals = int(signal_mask.sum())

    if len(signal_local_indices) == 0:
        return [], n_raw_signals, 0

    # Enforce minimum gap
    sig_ts_s = pred_ts[signal_local_indices].astype(np.float64) / 1e9
    keep = np.ones(len(signal_local_indices), dtype=bool)
    last_ts = -1e18
    for i in range(len(signal_local_indices)):
        if sig_ts_s[i] - last_ts >= MIN_SIGNAL_GAP_S:
            last_ts = sig_ts_s[i]
        else:
            keep[i] = False
    signal_local_indices = signal_local_indices[keep]
    n_deduped = len(signal_local_indices)

    if n_deduped == 0:
        return [], n_raw_signals, 0

    trades = []
    for sl_i in signal_local_indices:
        ev_idx = use_ev_idx[sl_i]

        filled, fill_delay = check_entry_fill(
            pred_l1[sl_i], pred_l5[sl_i], pred_l10[sl_i], pred_l30[sl_i],
            cancel_s, entry_mode_thresh
        )
        if not filled:
            continue

        trade = simulate_exit(mbo, ev_idx, fill_delay, hold_s)
        trades.append(trade)

    return trades, n_raw_signals, n_deduped


def compute_regime(mbo):
    """Classify day as green/red/flat."""
    labels_30s = mbo['labels_30s']
    valid = ~np.isnan(labels_30s)
    if valid.sum() == 0:
        return 'flat'
    total = float(labels_30s[valid].mean())
    if total > 0.05:
        return 'green'
    elif total < -0.05:
        return 'red'
    return 'flat'


def compute_metrics(trades_by_day, regimes, config):
    """Compute comprehensive metrics with regime stratification."""
    all_trades = []
    daily_pnls = []
    regime_daily_pnls = {'green': [], 'red': [], 'flat': []}

    for date in sorted(trades_by_day.keys()):
        trades = trades_by_day[date]
        day_pnl = sum(t['pnl_ticks'] for t in trades) if trades else 0.0
        daily_pnls.append(day_pnl)
        all_trades.extend(trades)
        regime = regimes.get(date, 'flat')
        regime_daily_pnls[regime].append(day_pnl)

    n_trades = len(all_trades)
    n_days = len(daily_pnls)
    if n_trades == 0 or n_days == 0:
        return None

    pnl_arr = np.array([t['pnl_ticks'] for t in all_trades])
    daily_arr = np.array(daily_pnls)

    wins = int(np.sum(pnl_arr > 0))
    win_rate = wins / n_trades

    gross_profit = float(np.sum(pnl_arr[pnl_arr > 0])) if (pnl_arr > 0).any() else 0.0
    gross_loss = float(np.abs(np.sum(pnl_arr[pnl_arr < 0]))) if (pnl_arr < 0).any() else 0.001
    pf = gross_profit / gross_loss

    def daily_sharpe(arr):
        if len(arr) > 1 and np.std(arr) > 0:
            return float(np.mean(arr) / np.std(arr) * np.sqrt(252))
        return 0.0

    def daily_sortino(arr):
        if len(arr) < 2:
            return 0.0
        down = arr[arr < 0]
        if len(down) == 0:
            return 999.0 if np.mean(arr) > 0 else 0.0
        ds = float(np.sqrt(np.mean(down ** 2)))
        return float(np.mean(arr) / ds * np.sqrt(252)) if ds > 0 else 0.0

    sharpe = daily_sharpe(daily_arr)
    sortino = daily_sortino(daily_arr)

    regime_sharpes = {}
    for regime in ['green', 'red', 'flat']:
        rarr = np.array(regime_daily_pnls[regime])
        regime_sharpes[regime] = {
            'n_days': len(rarr),
            'sharpe': daily_sharpe(rarr) if len(rarr) > 1 else 0.0,
            'mean_pnl': float(np.mean(rarr)) if len(rarr) > 0 else 0.0,
        }

    gs = regime_sharpes['green']['sharpe']
    rs = regime_sharpes['red']['sharpe']
    denom = max(abs(gs), abs(rs), 0.001)
    regime_gap = abs(gs - rs) / denom

    passive_exits = sum(1 for t in all_trades if t['exit_type'] == 'passive')
    passive_rate = passive_exits / n_trades

    cum = np.cumsum(daily_arr)
    peak = np.maximum.accumulate(cum)
    max_dd = float((peak - cum).max()) if len(cum) > 0 else 0.0

    entry_fill_rate = config.get('entry_fill_rate', 0)
    avg_daily_pnl_ticks = float(np.mean(daily_arr))
    avg_daily_pnl_usd = avg_daily_pnl_ticks * TICK_VALUE_USD

    # Per-day breakdown
    day_details = []
    for date in sorted(trades_by_day.keys()):
        trades = trades_by_day[date]
        dpnl = sum(t['pnl_ticks'] for t in trades) if trades else 0.0
        n_passive = sum(1 for t in trades if t['exit_type'] == 'passive')
        day_details.append({
            'date': date,
            'regime': regimes.get(date, 'flat'),
            'n_trades': len(trades),
            'pnl_ticks': round(dpnl, 2),
            'pnl_usd': round(dpnl * TICK_VALUE_USD, 2),
            'passive_exits': n_passive,
        })

    return {
        'confidence_pct': config['confidence_pct'],
        'entry_mode': config['entry_mode'],
        'cancel_window_s': config['cancel_window_s'],
        'hold_timeout_s': config['hold_timeout_s'],
        'n_trades': n_trades,
        'n_days': n_days,
        'trades_per_day': round(n_trades / n_days, 1),
        'entry_fill_rate': round(entry_fill_rate, 4),
        'passive_exit_rate': round(passive_rate, 4),
        'total_pnl_ticks': round(float(pnl_arr.sum()), 2),
        'total_pnl_usd': round(float(pnl_arr.sum() * TICK_VALUE_USD), 2),
        'avg_pnl_per_trade_ticks': round(float(pnl_arr.mean()), 4),
        'avg_daily_pnl_ticks': round(avg_daily_pnl_ticks, 2),
        'avg_daily_pnl_usd': round(avg_daily_pnl_usd, 2),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(min(pf, 999.0), 3),
        'daily_sharpe': round(sharpe, 3),
        'daily_sortino': round(min(sortino, 999.0), 3),
        'max_drawdown_ticks': round(max_dd, 2),
        'green_days': int(np.sum(daily_arr > 0)),
        'red_days': int(np.sum(daily_arr < 0)),
        'flat_days': int(np.sum(daily_arr == 0)),
        'regime_sharpes': regime_sharpes,
        'regime_gap': round(regime_gap, 3),
        'regime_gap_pass': regime_gap <= 0.50,
        'day_details': day_details,
    }


def main():
    flush("=" * 80)
    flush("PASSIVE LIMIT ORDER FILL-RATE & P&L VERDICT — v1")
    flush("  CNN-Mamba v2 (best checkpoint, IC_1s=0.237)")
    flush("  Consolidated OOT predictions | FIFO cost model")
    flush("=" * 80)

    # Load consolidated predictions
    day_preds = load_consolidated()

    # Find overlapping dates with MBO data
    mbo_dates = set()
    for f in os.listdir(MBO_DIR):
        if f.endswith('_mbo_events.npz'):
            mbo_dates.add(f[:8])

    overlap = sorted(set(day_preds.keys()) & mbo_dates)
    flush(f"\nPrediction dates: {len(day_preds)}")
    flush(f"MBO dates: {len(mbo_dates)}")
    flush(f"Overlap: {len(overlap)}")

    # Load MBO data and verify IC
    flush("\nLoading MBO data and verifying predictions...")
    day_list = []
    regimes = {}
    ics = []

    for dt in overlap:
        mbo = load_mbo_for_date(dt)
        if mbo is None:
            continue

        dp = day_preds[dt]
        n = dp['n_preds']
        event_indices = PRED_OFFSET + np.arange(n) * PRED_STRIDE
        valid_ev = event_indices < mbo['n_events']
        valid_mask = (dp['masks'] > 0) & valid_ev

        if valid_mask.sum() < 100:
            flush(f"  {dt}: too few valid samples ({valid_mask.sum()}), skipping")
            continue

        # Verify IC
        p = dp['preds'][valid_mask]
        ei = event_indices[valid_mask]
        el = mbo['labels_1s'][ei]
        bv = ~np.isnan(el)
        if bv.sum() > 100:
            ic = float(spearmanr(p[bv], el[bv])[0])
        else:
            ic = 0.0
        ics.append(ic)

        regime = compute_regime(mbo)
        regimes[dt] = regime

        day_list.append((dt, dp, mbo))
        flush(f"  {dt}: {valid_mask.sum():>6} valid preds, IC_1s={ic:.3f}, regime={regime}")

    flush(f"\nLoaded {len(day_list)} days with valid data")
    flush(f"Mean IC_1s: {np.mean(ics):.4f}")
    n_green = sum(1 for r in regimes.values() if r == 'green')
    n_red = sum(1 for r in regimes.values() if r == 'red')
    n_flat = sum(1 for r in regimes.values() if r == 'flat')
    flush(f"Regimes: {n_green} green, {n_red} red, {n_flat} flat")

    # Compute global thresholds using ALL valid predictions
    flush("\nComputing prediction thresholds...")
    all_preds_list = []
    for dt, dp, mbo in day_list:
        n = dp['n_preds']
        ei = PRED_OFFSET + np.arange(n) * PRED_STRIDE
        valid = (dp['masks'] > 0) & (ei < mbo['n_events'])
        all_preds_list.append(dp['preds'][valid])
    all_preds = np.concatenate(all_preds_list)
    flush(f"Total valid predictions: {len(all_preds):,}")

    thresholds = {}
    for pct in CONFIDENCE_PCTS:
        thresholds[pct] = float(np.percentile(all_preds, pct))
        n = int(np.sum(all_preds <= thresholds[pct]))
        flush(f"  Top {pct}% short: threshold={thresholds[pct]:.4f}, "
              f"{n:,} signals ({n/len(day_list):.0f}/day)")

    # ── FILL RATE ANALYSIS ──────────────────────────────────────────────────
    flush("\n" + "=" * 80)
    flush("ENTRY FILL RATE ANALYSIS — Top 1% Short Signals")
    flush("=" * 80)

    top1_thresh = thresholds[1]
    for mode_name, mode_thresh in ENTRY_MODES.items():
        for cancel_s in CANCEL_WINDOWS_S:
            total_deduped = 0
            total_filled = 0

            for dt, dp, mbo in day_list:
                trades, n_raw, n_dedup = simulate_day(
                    dp, mbo, top1_thresh, cancel_s, 30, mode_thresh
                )
                total_deduped += n_dedup
                total_filled += len(trades)

            fill_rate = total_filled / total_deduped if total_deduped > 0 else 0
            flush(f"  {mode_name:>12} | cancel={cancel_s:>2}s | "
                  f"deduped={total_deduped:>5} filled={total_filled:>5} | "
                  f"fill_rate={100*fill_rate:.1f}% ({total_filled/len(day_list):.1f}/day)")

    # ── FULL SWEEP ──────────────────────────────────────────────────────────
    flush("\n" + "=" * 80)
    flush("FULL P&L SWEEP")
    flush("=" * 80)

    all_results = []
    t0 = time.time()

    for conf_pct in CONFIDENCE_PCTS:
        thresh = thresholds[conf_pct]
        for mode_name, mode_thresh in ENTRY_MODES.items():
            for cancel_s in CANCEL_WINDOWS_S:
                for hold_s in HOLD_TIMEOUTS_S:
                    trades_by_day = {}
                    total_attempts = 0
                    total_filled = 0

                    for dt, dp, mbo in day_list:
                        trades, n_raw, n_dedup = simulate_day(
                            dp, mbo, thresh, cancel_s, hold_s, mode_thresh
                        )
                        trades_by_day[dt] = trades
                        total_attempts += n_dedup
                        total_filled += len(trades)

                    efr = total_filled / total_attempts if total_attempts > 0 else 0
                    config = {
                        'confidence_pct': conf_pct,
                        'entry_mode': mode_name,
                        'cancel_window_s': cancel_s,
                        'hold_timeout_s': hold_s,
                        'entry_fill_rate': efr,
                    }
                    metrics = compute_metrics(trades_by_day, regimes, config)
                    if metrics is not None:
                        all_results.append(metrics)

    elapsed = time.time() - t0
    flush(f"\nSweep completed in {elapsed:.0f}s ({len(all_results)} configs)")

    # Sort by Sharpe
    all_results.sort(key=lambda x: x['daily_sharpe'], reverse=True)

    # ── RESULTS TABLE ───────────────────────────────────────────────────────
    flush("\n" + "=" * 145)
    flush("TOP 25 CONFIGS (by Daily Sharpe)")
    flush("=" * 145)
    header = (f"{'Conf%':>5} {'Mode':>12} {'CW':>3} {'HT':>4} | "
              f"{'Trades':>6} {'T/D':>5} {'Fill%':>5} {'PsxEx':>5} | "
              f"{'AvgPnL':>7} {'WR%':>5} {'PF':>6} | "
              f"{'Sharpe':>7} {'Sortino':>8} | "
              f"{'$/day':>7} {'$total':>9} | "
              f"{'RGap':>5} {'Pass':>4}")
    flush(header)
    flush("-" * 145)

    for r in all_results[:25]:
        sort_s = f"{r['daily_sortino']:>8.2f}" if r['daily_sortino'] < 900 else "     inf"
        pf_s = f"{r['profit_factor']:>6.2f}" if r['profit_factor'] < 900 else "   inf"
        pass_s = "YES" if r['regime_gap_pass'] else "NO"
        flush(f"{r['confidence_pct']:>5} {r['entry_mode']:>12} {r['cancel_window_s']:>3} {r['hold_timeout_s']:>4} | "
              f"{r['n_trades']:>6} {r['trades_per_day']:>5.1f} {100*r['entry_fill_rate']:>4.0f}% {100*r['passive_exit_rate']:>4.0f}% | "
              f"{r['avg_pnl_per_trade_ticks']:>+7.3f} {100*r['win_rate']:>5.1f} {pf_s} | "
              f"{r['daily_sharpe']:>7.2f} {sort_s} | "
              f"{r['avg_daily_pnl_usd']:>+7.0f} {r['total_pnl_usd']:>+9.0f} | "
              f"{r['regime_gap']:>5.2f} {pass_s:>4}")

    # ── DETAILED TOP 5 ──────────────────────────────────────────────────────
    flush("\n" + "=" * 80)
    flush("TOP 5 CONFIGS — DETAILED BREAKDOWN")
    flush("=" * 80)

    for i, r in enumerate(all_results[:5]):
        sort_v = 'inf' if r['daily_sortino'] >= 900 else f"{r['daily_sortino']:.2f}"
        pf_v = 'inf' if r['profit_factor'] >= 900 else f"{r['profit_factor']:.3f}"
        flush(f"\n--- #{i+1}: top{r['confidence_pct']}% | {r['entry_mode']} | "
              f"cancel={r['cancel_window_s']}s | hold={r['hold_timeout_s']}s ---")
        flush(f"  Trades: {r['n_trades']} ({r['trades_per_day']:.1f}/day over {r['n_days']} days)")
        flush(f"  Entry fill rate: {100*r['entry_fill_rate']:.1f}%")
        flush(f"  Passive exit rate: {100*r['passive_exit_rate']:.1f}%")
        flush(f"  Avg P&L: {r['avg_pnl_per_trade_ticks']:+.4f} ticks "
              f"(${r['avg_pnl_per_trade_ticks']*TICK_VALUE_USD:+.2f}/trade)")
        flush(f"  Win rate: {100*r['win_rate']:.1f}%")
        flush(f"  Profit factor: {pf_v}")
        flush(f"  Daily Sharpe: {r['daily_sharpe']:.3f}")
        flush(f"  Daily Sortino: {sort_v}")
        flush(f"  Avg daily P&L: {r['avg_daily_pnl_ticks']:+.2f} ticks (${r['avg_daily_pnl_usd']:+.0f}/day)")
        flush(f"  Total P&L: {r['total_pnl_ticks']:+.1f} ticks (${r['total_pnl_usd']:+,.0f})")
        flush(f"  Green/Red/Flat days: {r['green_days']}/{r['red_days']}/{r['flat_days']}")
        flush(f"  Max drawdown: {r['max_drawdown_ticks']:.1f} ticks (${r['max_drawdown_ticks']*TICK_VALUE_USD:.0f})")
        flush(f"  Regime gap: {r['regime_gap']:.3f} {'PASS' if r['regime_gap_pass'] else 'FAIL'}")
        for regime in ['green', 'red', 'flat']:
            rs = r['regime_sharpes'][regime]
            flush(f"    {regime:>5}: {rs['n_days']} days, Sharpe={rs['sharpe']:.2f}, "
                  f"avg={rs['mean_pnl']:+.2f} ticks/day")

    # ── FOCUSED VERDICT ─────────────────────────────────────────────────────
    flush("\n" + "=" * 80)
    flush("VERDICT: Top 1% Short Signals")
    flush("=" * 80)

    # Best top-1% config by entry mode
    for mode in ['conservative', 'moderate', 'aggressive']:
        configs = [r for r in all_results if r['confidence_pct'] == 1 and r['entry_mode'] == mode]
        if configs:
            best = max(configs, key=lambda x: x['daily_sharpe'])
            pass_str = "PASS" if best['regime_gap_pass'] else "FAIL"
            flush(f"\n  {mode.upper()} entry: cancel={best['cancel_window_s']}s, hold={best['hold_timeout_s']}s")
            flush(f"    Fill rate: {100*best['entry_fill_rate']:.1f}%  |  "
                  f"Passive exit: {100*best['passive_exit_rate']:.1f}%")
            flush(f"    Trades/day: {best['trades_per_day']:.1f}  |  "
                  f"Win rate: {100*best['win_rate']:.1f}%")
            flush(f"    Sharpe: {best['daily_sharpe']:.2f}  |  "
                  f"Sortino: {best['daily_sortino']:.2f}")
            flush(f"    PF: {best['profit_factor']:.2f}  |  "
                  f"Regime gap: {best['regime_gap']:.3f} {pass_str}")
            flush(f"    Est. daily: ${best['avg_daily_pnl_usd']:+.0f}/day  |  "
                  f"Total: ${best['total_pnl_usd']:+,.0f}")

    # ── SAVE ────────────────────────────────────────────────────────────────
    passing = [r for r in all_results if r['regime_gap_pass']]
    best_overall = all_results[0] if all_results else None
    best_passing = max(passing, key=lambda x: x['daily_sharpe']) if passing else None

    # Remove day_details from all_results to keep file manageable
    # but keep it for best config
    top1_with_details = [r for r in all_results if r['confidence_pct'] == 1]
    for r in all_results:
        if r not in top1_with_details[:3]:
            r.pop('day_details', None)

    output = {
        'analysis': 'passive_limit_verdict_v1',
        'description': 'CNN-Mamba v2 (best checkpoint IC_1s=0.237): passive limit fill rate and P&L',
        'date_run': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_summary': {
            'n_dates': len(day_list),
            'n_green': n_green,
            'n_red': n_red,
            'n_flat': n_flat,
            'mean_ic_1s': round(float(np.mean(ics)), 4),
            'total_valid_predictions': int(len(all_preds)),
            'date_range': f"{overlap[0]} to {overlap[-1]}",
            'model': 'CNN-Mamba v2 fold_10_best.pt',
            'model_ic_1s': 0.237,
        },
        'cost_model': {
            'commission_rt_ticks': COMMISSION_TICKS,
            'spread_cost': 'zero (passive fill at bid/ask)',
            'passive_exit_net': f'+1.0 - {COMMISSION_TICKS} = +{1.0 - COMMISSION_TICKS} ticks',
        },
        'thresholds': {f'top_{pct}pct': round(float(thresholds[pct]), 4) for pct in CONFIDENCE_PCTS},
        'best_overall': best_overall,
        'best_regime_passing': best_passing,
        'all_results': all_results,
    }

    with open(OUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    flush(f"\nResults saved to {OUT_FILE}")

    # ── BOTTOM LINE ─────────────────────────────────────────────────────────
    flush("\n" + "=" * 80)
    flush("BOTTOM LINE")
    flush("=" * 80)
    if best_passing:
        flush(f"\n  Best regime-passing config:")
        flush(f"    Confidence: top {best_passing['confidence_pct']}%")
        flush(f"    Entry: {best_passing['entry_mode']}, cancel={best_passing['cancel_window_s']}s")
        flush(f"    Hold: {best_passing['hold_timeout_s']}s")
        flush(f"    Fill rate: {100*best_passing['entry_fill_rate']:.1f}%")
        flush(f"    Trades/day: {best_passing['trades_per_day']:.1f}")
        flush(f"    Sharpe: {best_passing['daily_sharpe']:.2f}")
        flush(f"    Sortino: {best_passing['daily_sortino']:.2f}")
        flush(f"    Win rate: {100*best_passing['win_rate']:.1f}%")
        flush(f"    PF: {best_passing['profit_factor']:.2f}")
        flush(f"    Est. daily: ${best_passing['avg_daily_pnl_usd']:+.0f}/day")
        flush(f"    Regime gap: {best_passing['regime_gap']:.3f} PASS")
    flush("\nDone.")


if __name__ == '__main__':
    main()
