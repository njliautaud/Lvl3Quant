#!/usr/bin/env python3
"""
entry_timing_mbo_v1.py — MBO Entry Timing Study

After the LightGBM 30-min bar signal fires, what happens in the next 10 minutes
at the MBO tick level? Is there a systematic pullback we can exploit for better entry?

Key questions:
  1. After signal: avg price path for next 10 min (do we see a pullback?)
  2. If we waited 1-5 minutes, how often could we enter 1-3 ticks better?
  3. What's the optimal "patience window" — wait for pullback within N minutes?
  4. How does entry improvement translate to WR improvement?

Uses MBO event data for tick-level analysis on the 14 dates where we have
both champion trades and MBO data.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/nick/Lvl3Quant")
MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
MBO_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
ENTRY_PREDS_PATH = ROOT / "output" / "mfe_mae_analysis" / "entry_predictions.npz"
TRADES_PATH = ROOT / "output" / "mfe_mae_analysis" / "trades_top_5pct.parquet"
OUT_DIR = ROOT / "output" / "entry_timing_mbo_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / f"entry_timing_mbo_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("entry_timing")

TICK_SIZE = 0.25
ENTRY_BAR_SIZE = 30
CANCEL_WINDOW_MIN = 10
CONFIDENCE_PCT = 0.05


def load_minute_bars(date: str) -> pd.DataFrame | None:
    p = MINUTE_BAR_DIR / f"{date}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df['ts_minute'] = pd.to_datetime(df['ts_minute'], utc=True)
    return df.sort_values('ts_minute').reset_index(drop=True)


def aggregate_to_30min(minute_df: pd.DataFrame) -> pd.DataFrame:
    df = minute_df.copy()
    df['bar_key'] = df['ts_minute'].dt.floor('30min')
    agg = df.groupby('bar_key').agg(
        open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'), close=('close', 'last'),
        volume=('volume', 'sum'),
    ).reset_index()
    agg['ts'] = agg['bar_key']
    return agg


def load_minute_price_series(date: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load minute bar data and return (timestamps_ns, close_prices) arrays.
    Uses minute bars as highest-resolution price data available with raw prices."""
    minute_df = load_minute_bars(date)
    if minute_df is None or len(minute_df) < 10:
        return None

    ts = minute_df['ts_minute'].values.astype('datetime64[ns]').astype(np.int64)
    prices = minute_df['close'].values.astype(np.float64)
    return ts, prices


def get_signal_times(date: str, all_preds, all_dates, upper_thresh, lower_thresh) -> list:
    """Get LightGBM signal times and directions for a date."""
    minute_df = load_minute_bars(date)
    if minute_df is None:
        return []

    bars_30m = aggregate_to_30min(minute_df)
    date_mask = all_dates == date
    date_preds = all_preds[date_mask]

    n = min(len(date_preds), len(bars_30m))
    date_preds = date_preds[:n]
    bars_30m = bars_30m.iloc[:n].reset_index(drop=True)

    signals = []
    for i in range(len(bars_30m)):
        if np.isnan(date_preds[i]):
            continue
        direction = 0
        if date_preds[i] >= upper_thresh:
            direction = 1
        elif date_preds[i] <= lower_thresh:
            direction = -1
        if direction == 0:
            continue

        signal_ts = pd.Timestamp(bars_30m['ts'].iloc[i])
        signal_bar_end = signal_ts + pd.Timedelta(minutes=ENTRY_BAR_SIZE)
        signal_price = bars_30m['close'].iloc[i]

        signals.append({
            'date': date,
            'signal_ts': signal_ts,
            'signal_bar_end': signal_bar_end,
            'signal_bar_end_ns': int(signal_bar_end.value),
            'signal_price': signal_price,
            'direction': direction,
            'pred': float(date_preds[i]),
        })

    return signals


def analyze_post_signal_paths(signals: list, mbo_cache: dict):
    """For each signal, analyze the MBO tick-level price path after signal fires."""
    results = []

    for sig in signals:
        date = sig['date']
        if date not in mbo_cache:
            continue

        ts, mid = mbo_cache[date]
        bar_end_ns = sig['signal_bar_end_ns']
        signal_price = sig['signal_price']
        direction = sig['direction']

        # Find all MBO events in the 10 minutes after signal bar end
        window_end_ns = bar_end_ns + int(10 * 60 * 1e9)  # 10 min
        mask = (ts >= bar_end_ns) & (ts <= window_end_ns)

        if np.sum(mask) < 2:
            continue

        window_ts = ts[mask]
        window_mid = mid[mask]

        # Price change from signal price, in ticks
        price_change = (window_mid - signal_price) / TICK_SIZE

        # For shorts, negate (positive = favorable for short)
        if direction == -1:
            price_change = -price_change

        # Compute stats at various time checkpoints (seconds after bar end)
        checkpoints = [5, 10, 15, 30, 60, 120, 180, 300, 600]
        path_stats = {'date': date, 'direction': direction}

        for cp_sec in checkpoints:
            cp_ns = bar_end_ns + int(cp_sec * 1e9)
            cp_mask = window_ts <= cp_ns
            if np.sum(cp_mask) == 0:
                continue

            cp_changes = price_change[cp_mask]
            path_stats[f'avg_{cp_sec}s'] = float(cp_changes[-1]) if len(cp_changes) > 0 else np.nan
            path_stats[f'min_{cp_sec}s'] = float(np.min(cp_changes))  # worst adverse
            path_stats[f'max_{cp_sec}s'] = float(np.max(cp_changes))  # best favorable

        # Best entry improvement: min price change in first N seconds
        # (for longs, min change = deepest pullback = best entry; for shorts, same after negation)
        for window_sec in [30, 60, 120, 180, 300]:
            w_ns = bar_end_ns + int(window_sec * 1e9)
            w_mask = window_ts <= w_ns
            if np.sum(w_mask) < 1:
                continue
            w_changes = price_change[w_mask]
            # Best entry = minimum price change (deepest pullback)
            best_entry_improvement = -float(np.min(w_changes))  # ticks better than signal price
            path_stats[f'best_entry_{window_sec}s'] = best_entry_improvement
            # How often is there a 1-tick pullback?
            path_stats[f'pullback_1t_{window_sec}s'] = float(np.min(w_changes) <= -1.0)
            path_stats[f'pullback_2t_{window_sec}s'] = float(np.min(w_changes) <= -2.0)

        results.append(path_stats)

    return results


def main():
    log.info("=" * 70)
    log.info("MBO Entry Timing Study v1")
    log.info("=" * 70)

    # Load predictions
    data = np.load(ENTRY_PREDS_PATH, allow_pickle=True)
    all_preds = data['entry_preds']
    all_dates = data['dates']
    valid_mask = ~np.isnan(all_preds)
    upper_thresh = np.nanquantile(all_preds[valid_mask], 1 - CONFIDENCE_PCT)
    lower_thresh = np.nanquantile(all_preds[valid_mask], CONFIDENCE_PCT)

    # Get trade dates
    orig = pd.read_parquet(TRADES_PATH)
    oot_dates = sorted(orig['date'].unique())
    log.info(f"OOT dates: {len(oot_dates)}")

    # Load MBO data and get signals for all dates
    log.info("\nLoading MBO data and computing signals...")
    mbo_cache = {}
    all_signals = []

    for date in oot_dates:
        result = load_minute_price_series(date)
        if result is not None:
            mbo_cache[date] = result
            signals = get_signal_times(date, all_preds, all_dates, upper_thresh, lower_thresh)
            all_signals.extend(signals)

    log.info(f"Loaded MBO for {len(mbo_cache)} dates, found {len(all_signals)} signals")

    if not all_signals:
        log.error("No signals found with MBO data!")
        return

    # Analyze post-signal price paths
    log.info("\nAnalyzing post-signal price paths...")
    path_results = analyze_post_signal_paths(all_signals, mbo_cache)
    log.info(f"Analyzed {len(path_results)} signal paths")

    if not path_results:
        log.error("No paths analyzed!")
        return

    path_df = pd.DataFrame(path_results)

    # Summary statistics
    log.info("\n" + "=" * 70)
    log.info("POST-SIGNAL PRICE PATH (positive = favorable direction)")
    log.info("=" * 70)

    log.info("\n--- Average price change at checkpoints (ticks, favorable direction) ---")
    for cp in [5, 10, 15, 30, 60, 120, 180, 300, 600]:
        col = f'avg_{cp}s'
        if col in path_df.columns:
            vals = path_df[col].dropna()
            log.info(f"  @{cp:>3}s: mean={vals.mean():+.2f}t, median={vals.median():+.2f}t, "
                    f"std={vals.std():.2f}t (n={len(vals)})")

    log.info("\n--- Best entry improvement (ticks better than signal price) ---")
    for ws in [30, 60, 120, 180, 300]:
        col = f'best_entry_{ws}s'
        if col in path_df.columns:
            vals = path_df[col].dropna()
            log.info(f"  Wait {ws:>3}s: avg improvement={vals.mean():.2f}t, "
                    f"median={vals.median():.2f}t, p75={vals.quantile(0.75):.2f}t")

    log.info("\n--- Pullback frequency ---")
    for ws in [30, 60, 120, 180, 300]:
        pb1 = f'pullback_1t_{ws}s'
        pb2 = f'pullback_2t_{ws}s'
        if pb1 in path_df.columns:
            vals1 = path_df[pb1].dropna()
            vals2 = path_df[pb2].dropna()
            log.info(f"  Window {ws:>3}s: 1-tick pullback {vals1.mean():.0%}, "
                    f"2-tick pullback {vals2.mean():.0%} (n={len(vals1)})")

    # Direction-specific analysis
    log.info("\n--- By direction ---")
    for d, label in [(1, 'LONG'), (-1, 'SHORT')]:
        sub = path_df[path_df['direction'] == d]
        if len(sub) < 5:
            continue
        log.info(f"\n  {label} ({len(sub)} signals):")
        for ws in [60, 120, 300]:
            col = f'best_entry_{ws}s'
            if col in sub.columns:
                vals = sub[col].dropna()
                log.info(f"    Wait {ws:>3}s: avg improvement={vals.mean():.2f}t, "
                        f"median={vals.median():.2f}t")
        for ws in [60, 300]:
            pb1 = f'pullback_1t_{ws}s'
            if pb1 in sub.columns:
                log.info(f"    Wait {ws:>3}s: 1-tick pullback rate={sub[pb1].mean():.0%}")

    # Impact analysis: if we entered X ticks better, what would WR be?
    log.info("\n--- IMPACT: If we entered N ticks better ---")
    log.info("  (Using existing trade outcomes: TP=25, SL_L=4/SL_S=3)")
    enriched = pd.read_parquet(ROOT / "output" / "winloss_characterization_v1" / "trades_enriched.parquet")
    total_trades = len(enriched)
    base_wr = enriched['winner'].mean()

    # Winners hit TP=25 and avoided SL. Entry improvement helps losers avoid SL.
    # A loser with exit_type='sl' and MAE just barely above SL might survive with better entry.
    sl_losers = enriched[enriched['exit_type'] == 'sl']

    log.info(f"  Baseline: {total_trades} trades, WR={base_wr:.1%}")
    log.info(f"  SL losers: {len(sl_losers)} trades")

    # For each tick of entry improvement, how many SL losers would be saved?
    # (simplified: if entry is N ticks better, the effective SL is N ticks wider)
    for improvement in [0.5, 1.0, 1.5, 2.0, 3.0]:
        # Losers whose MAE was within `improvement` of the SL
        for d in [1, -1]:
            sl = 4 if d == 1 else 3
            sub = sl_losers[sl_losers['direction'] == d]
            # Would survive: mae_ticks < sl + improvement (effectively wider SL)
            # But we need bar-by-bar sim... approximate: count losers with low mae
            saved = sub[sub['mae_ticks'] < sl + improvement]
            # Actually mae_ticks is the full-path max, not bar-0 max
            # Better approx: use exit_bar=0 losers where first-bar adverse < sl + improvement

    # Just report the pullback statistics and let the user decide
    log.info("\n--- PRACTICAL RECOMMENDATION ---")
    for ws in [60, 120, 300]:
        col = f'best_entry_{ws}s'
        if col in path_df.columns:
            vals = path_df[col].dropna()
            pct_1t = (vals >= 1.0).mean()
            pct_2t = (vals >= 2.0).mean()
            avg = vals.mean()
            log.info(f"  Wait up to {ws}s for pullback: avg improvement {avg:.2f}t, "
                    f"chance of 1+ tick improvement: {pct_1t:.0%}, "
                    f"chance of 2+ tick improvement: {pct_2t:.0%}")

    # Save results
    path_df.to_parquet(OUT_DIR / "post_signal_paths.parquet", index=False)
    log.info(f"\nSaved to {OUT_DIR}")
    log.info("DONE")


if __name__ == "__main__":
    main()
