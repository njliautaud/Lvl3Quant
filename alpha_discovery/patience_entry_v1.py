#!/usr/bin/env python3
"""
patience_entry_v1.py — Patience entry backtest

Instead of entering at the 30-min bar close price, wait for a pullback of N ticks
within a patience window of M minutes. If the pullback happens, enter at a better
price (effectively wider SL). If not, enter at market or skip.

Sweep:
  - pullback_ticks: [1, 2, 3, 4] — how much better than bar close we demand
  - patience_min: [1, 2, 3, 5] — how long to wait for pullback
  - miss_action: ['market', 'skip'] — what to do if pullback doesn't happen

Uses existing enriched trade data + minute bars for fill simulation.
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
ENTRY_PREDS_PATH = ROOT / "output" / "mfe_mae_analysis" / "entry_predictions.npz"
TRADES_PATH = ROOT / "output" / "mfe_mae_analysis" / "trades_top_5pct.parquet"
OUT_DIR = ROOT / "output" / "patience_entry_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / f"patience_entry_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("patience")

# Champion params
TICK_SIZE = 0.25
ENTRY_BAR_SIZE = 30
CONFIDENCE_PCT = 0.05
TP_TICKS = 25
SL_LONG = 4
SL_SHORT = 3
MAX_HOLD_BARS = 60
RT_COMMISSION_TICKS = 0.376
MARKET_SL_EXTRA = 1.0

# Sweep params
PULLBACK_TICKS = [0, 1, 2, 3, 4, 5]  # 0 = original baseline (no patience)
PATIENCE_MINUTES = [1, 2, 3, 5, 10]
MISS_ACTIONS = ['market', 'skip']


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


def simulate_patience_entry(direction, signal_price, signal_bar_end, minute_df,
                            pullback_ticks, patience_min, miss_action):
    """
    Simulate patience entry: wait for a pullback_ticks improvement within patience_min.

    For LONG: limit = signal_price - pullback_ticks * TICK_SIZE (buy lower)
              fill when bar low <= limit - TICK_SIZE (passive fill)
    For SHORT: limit = signal_price + pullback_ticks * TICK_SIZE (sell higher)
              fill when bar high >= limit + TICK_SIZE (passive fill)

    If pullback doesn't happen within patience window:
      - 'market': enter at signal_price (same as baseline)
      - 'skip': skip the trade

    Returns: (fill_price, fill_ts, was_patience_fill, was_market_fill) or None
    """
    if pullback_ticks == 0:
        # Baseline: original entry logic (no patience)
        limit_price = signal_price
    else:
        # Patience entry: demand N ticks better
        if direction == 1:
            limit_price = signal_price - pullback_ticks * TICK_SIZE
        else:
            limit_price = signal_price + pullback_ticks * TICK_SIZE

    day_ts = minute_df['ts_minute'].values
    signal_bar_end_np = np.datetime64(signal_bar_end)
    patience_end = signal_bar_end + pd.Timedelta(minutes=patience_min)
    patience_end_np = np.datetime64(patience_end)

    # Original cancel window (10 min from bar end)
    cancel_ts = signal_bar_end + pd.Timedelta(minutes=10 + ENTRY_BAR_SIZE - ENTRY_BAR_SIZE)  # 10 min
    cancel_ts_np = np.datetime64(cancel_ts)

    # Phase 1: Look for patience fill within patience window
    patience_mask = (day_ts >= signal_bar_end_np) & (day_ts <= patience_end_np)
    patience_bars = minute_df[patience_mask]

    for _, mbar in patience_bars.iterrows():
        if direction == 1:
            if mbar['low'] <= limit_price - TICK_SIZE:
                return limit_price, mbar['ts_minute'], True, False
        else:
            if mbar['high'] >= limit_price + TICK_SIZE:
                return limit_price, mbar['ts_minute'], True, False

    # Patience failed — handle miss
    if miss_action == 'skip':
        return None

    # miss_action == 'market': enter at original signal price after patience window
    # Use remaining cancel window after patience expires
    market_mask = (day_ts > patience_end_np) & (day_ts <= cancel_ts_np)
    market_bars = minute_df[market_mask]

    for _, mbar in market_bars.iterrows():
        if direction == 1:
            if mbar['low'] <= signal_price - TICK_SIZE:
                return signal_price, mbar['ts_minute'], False, True
        else:
            if mbar['high'] >= signal_price + TICK_SIZE:
                return signal_price, mbar['ts_minute'], False, True

    return None  # Cancel window expired


def simulate_exit(direction, fill_price, fill_ts, minute_df):
    """Simulate TP/SL/max-hold exit from fill."""
    day_ts = minute_df['ts_minute'].values
    fill_ts_np = np.datetime64(fill_ts)
    remaining_mask = day_ts >= fill_ts_np
    remaining = minute_df[remaining_mask].head(MAX_HOLD_BARS)

    if len(remaining) < 2:
        return None

    sl_ticks = SL_LONG if direction == 1 else SL_SHORT

    for j in range(len(remaining)):
        bar = remaining.iloc[j]
        if direction == 1:
            bar_mfe = (bar['high'] - fill_price) / TICK_SIZE
            bar_mae = (fill_price - bar['low']) / TICK_SIZE
        else:
            bar_mfe = (fill_price - bar['low']) / TICK_SIZE
            bar_mae = (bar['high'] - fill_price) / TICK_SIZE

        if bar_mfe >= TP_TICKS:
            return TP_TICKS - RT_COMMISSION_TICKS  # passive TP
        if bar_mae >= sl_ticks:
            return -sl_ticks - RT_COMMISSION_TICKS - MARKET_SL_EXTRA  # market SL

    # Max hold exit
    if direction == 1:
        final = (remaining['close'].iloc[-1] - fill_price) / TICK_SIZE
    else:
        final = (fill_price - remaining['close'].iloc[-1]) / TICK_SIZE
    return final - RT_COMMISSION_TICKS


def compute_metrics(exits, dates):
    """Compute Sharpe/WR/PF from trade results."""
    if len(exits) == 0:
        return {'n': 0, 'sharpe': np.nan, 'wr': np.nan, 'pf': np.nan,
                'avg': np.nan, 'total': 0.0}

    exits = np.array(exits)
    n = len(exits)
    wr = float(np.mean(exits > 0))

    gross_win = float(np.sum(exits[exits > 0]))
    gross_loss = float(np.abs(np.sum(exits[exits < 0])))
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf

    # Day-level Sharpe
    df_temp = pd.DataFrame({'date': dates, 'exit': exits})
    day_pnl = df_temp.groupby('date')['exit'].sum()
    sharpe = float(day_pnl.mean() / day_pnl.std() * np.sqrt(252)) if day_pnl.std() > 0 else np.nan

    return {
        'n': n, 'sharpe': sharpe, 'wr': wr, 'pf': pf,
        'avg': float(np.mean(exits)), 'total': float(np.sum(exits)),
    }


def main():
    log.info("=" * 70)
    log.info("Patience Entry Backtest v1")
    log.info("=" * 70)

    # Load predictions
    data = np.load(ENTRY_PREDS_PATH, allow_pickle=True)
    all_preds = data['entry_preds']
    all_dates = data['dates']
    valid_mask = ~np.isnan(all_preds)
    upper_thresh = np.nanquantile(all_preds[valid_mask], 1 - CONFIDENCE_PCT)
    lower_thresh = np.nanquantile(all_preds[valid_mask], CONFIDENCE_PCT)

    # Get OOT dates
    orig = pd.read_parquet(TRADES_PATH)
    oot_dates = sorted(orig['date'].unique())
    log.info(f"OOT dates: {len(oot_dates)}")

    # Pre-load all minute bars
    minute_cache = {}
    bar30_cache = {}
    for date in oot_dates:
        mb = load_minute_bars(date)
        if mb is not None:
            minute_cache[date] = mb
            bar30_cache[date] = aggregate_to_30min(mb)
    log.info(f"Loaded minute bars for {len(minute_cache)} dates")

    # Pre-compute signals
    signals = []
    for date in oot_dates:
        if date not in bar30_cache:
            continue
        bars_30m = bar30_cache[date]
        date_mask = all_dates == date
        date_preds = all_preds[date_mask]

        n = min(len(date_preds), len(bars_30m))
        for i in range(n):
            if np.isnan(date_preds[i]):
                continue
            direction = 0
            if date_preds[i] >= upper_thresh:
                direction = 1
            elif date_preds[i] <= lower_thresh:
                direction = -1
            if direction == 0:
                continue

            signals.append({
                'date': date,
                'signal_ts': pd.Timestamp(bars_30m['ts'].iloc[i]),
                'signal_bar_end': pd.Timestamp(bars_30m['ts'].iloc[i]) + pd.Timedelta(minutes=ENTRY_BAR_SIZE),
                'signal_price': bars_30m['close'].iloc[i],
                'direction': direction,
                'pred': float(date_preds[i]),
            })
    log.info(f"Signals: {len(signals)}")

    # Run sweep
    results = []

    for pullback in PULLBACK_TICKS:
        for patience in PATIENCE_MINUTES:
            for miss in MISS_ACTIONS:
                # Skip redundant combos
                if pullback == 0 and patience > 1:
                    continue  # baseline doesn't need patience sweep
                if pullback == 0 and miss == 'skip':
                    continue

                arm_name = f"pb={pullback}t|wait={patience}m|miss={miss}"

                exits = []
                dates = []
                n_patience_fills = 0
                n_market_fills = 0
                n_skipped = 0

                for sig in signals:
                    date = sig['date']
                    if date not in minute_cache:
                        continue

                    result = simulate_patience_entry(
                        sig['direction'], sig['signal_price'], sig['signal_bar_end'],
                        minute_cache[date], pullback, patience, miss
                    )

                    if result is None:
                        n_skipped += 1
                        continue

                    fill_price, fill_ts, was_patience, was_market = result
                    if was_patience:
                        n_patience_fills += 1
                    elif was_market:
                        n_market_fills += 1

                    exit_ticks = simulate_exit(
                        sig['direction'], fill_price, fill_ts, minute_cache[date]
                    )

                    if exit_ticks is not None:
                        exits.append(exit_ticks)
                        dates.append(date)

                m = compute_metrics(exits, dates)
                total_attempts = n_patience_fills + n_market_fills + n_skipped
                patience_rate = n_patience_fills / max(1, total_attempts)

                log.info(f"\n  {arm_name}:")
                log.info(f"    Trades: {m['n']}, Sharpe: {m['sharpe']:.2f}, "
                        f"WR: {m['wr']:.1%}, PF: {m['pf']:.2f}, Avg: {m['avg']:.2f}t")
                log.info(f"    Patience fills: {n_patience_fills} ({patience_rate:.0%}), "
                        f"Market fills: {n_market_fills}, Skipped: {n_skipped}")

                results.append({
                    'arm': arm_name,
                    'pullback': pullback,
                    'patience_min': patience,
                    'miss_action': miss,
                    'n_trades': m['n'],
                    'sharpe': m['sharpe'],
                    'wr': m['wr'],
                    'pf': m['pf'],
                    'avg': m['avg'],
                    'total': m['total'],
                    'patience_fills': n_patience_fills,
                    'market_fills': n_market_fills,
                    'skipped': n_skipped,
                    'patience_rate': patience_rate,
                })

    # Summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "patience_sweep.csv", index=False)

    log.info("\n\n" + "=" * 70)
    log.info("SUMMARY — ALL ARMS")
    log.info("=" * 70)
    log.info(f"\n{'ARM':<35} {'N':>4} {'Sharpe':>7} {'WR':>6} {'PF':>5} {'Avg':>6} {'Pat%':>5}")
    log.info(f"{'-'*35} {'-'*4} {'-'*7} {'-'*6} {'-'*5} {'-'*6} {'-'*5}")

    for _, row in results_df.sort_values('sharpe', ascending=False).iterrows():
        if row['n_trades'] >= 30:
            log.info(f"{row['arm']:<35} {row['n_trades']:>4} {row['sharpe']:>7.2f} "
                    f"{row['wr']:>6.1%} {row['pf']:>5.2f} {row['avg']:>6.2f} "
                    f"{row['patience_rate']:>5.0%}")

    # Best arm
    valid = results_df[results_df['n_trades'] >= 50].sort_values('sharpe', ascending=False)
    if len(valid) > 0:
        best = valid.iloc[0]
        baseline = results_df[results_df['pullback'] == 0].iloc[0]
        log.info(f"\n  BASELINE: {baseline['n_trades']} trades, Sharpe {baseline['sharpe']:.2f}, "
                f"WR {baseline['wr']:.1%}")
        log.info(f"  BEST:     {best['arm']}: {best['n_trades']} trades, Sharpe {best['sharpe']:.2f}, "
                f"WR {best['wr']:.1%} (Sharpe lift: {best['sharpe']-baseline['sharpe']:+.2f})")

    log.info(f"\nSaved to {OUT_DIR}")
    log.info("DONE")


if __name__ == "__main__":
    main()
