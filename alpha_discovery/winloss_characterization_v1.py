#!/usr/bin/env python3
"""
winloss_characterization_v1.py — Deep dive into what separates winning vs losing trades
in the champion 30-min LightGBM strategy.

Goal: Find features at entry time that predict whether a trade will win or lose,
so we can build a smarter entry filter to boost WR from 23%.

Analyzes ALL 60 OOT days (no CNN-Mamba dependency).

Features examined at entry time:
  1. Time of day (morning vs afternoon vs close)
  2. LightGBM prediction magnitude (higher confidence = better?)
  3. Recent volatility (last 30 min realized vol)
  4. Recent OFI (order flow imbalance direction/magnitude)
  5. Direction (long vs short performance)
  6. MFE trajectory (how quickly do winners reach TP?)
  7. MAE trajectory (how quickly do losers hit SL?)
  8. Day of week effects
  9. Entry fill delay (immediate vs delayed fills)
  10. Volume at entry
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
OUT_DIR = ROOT / "output" / "winloss_characterization_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / f"winloss_char_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("winloss")

# Champion params
TICK_SIZE = 0.25
ENTRY_BAR_SIZE = 30
CANCEL_WINDOW_MIN = 10
CONFIDENCE_PCT = 0.05
TP_TICKS = 25
SL_LONG = 4
SL_SHORT = 3
MAX_HOLD_BARS = 60
RT_COMMISSION_TICKS = 0.376
MARKET_SL_EXTRA = 1.0


def reconstruct_all_trades() -> pd.DataFrame:
    """Reconstruct champion trades with full context features for all 60 OOT days."""
    data = np.load(ENTRY_PREDS_PATH, allow_pickle=True)
    all_preds = data['entry_preds']
    all_dates = data['dates']

    valid_mask = ~np.isnan(all_preds)
    valid_preds = all_preds[valid_mask]
    upper_thresh = np.nanquantile(valid_preds, 1 - CONFIDENCE_PCT)
    lower_thresh = np.nanquantile(valid_preds, CONFIDENCE_PCT)
    log.info(f"Thresholds: upper={upper_thresh:.4f}, lower={lower_thresh:.4f}")

    # Load original trades to know which dates to cover
    orig_trades = pd.read_parquet(TRADES_PATH)
    oot_dates = sorted(orig_trades['date'].unique())
    log.info(f"OOT dates: {len(oot_dates)}")

    trades = []

    for date in oot_dates:
        minute_df = load_minute_bars(date)
        if minute_df is None:
            continue

        # Aggregate to 30-min
        bars_30m = aggregate_to_30min(minute_df)

        # Get predictions for this date
        date_mask = all_dates == date
        date_preds = all_preds[date_mask]

        n = min(len(date_preds), len(bars_30m))
        date_preds = date_preds[:n]
        bars_30m = bars_30m.iloc[:n].reset_index(drop=True)

        # Pre-compute daily features
        daily_ofi = compute_daily_ofi(minute_df)
        daily_vol = compute_rolling_vol(minute_df)

        minute_lookup = minute_df.set_index('ts_minute')

        for i in range(len(bars_30m)):
            if np.isnan(date_preds[i]):
                continue

            direction = 0
            if date_preds[i] >= upper_thresh:
                direction = 1
            elif date_preds[i] <= lower_thresh:
                direction = -1
            else:
                continue

            signal_ts = pd.Timestamp(bars_30m['ts'].iloc[i])
            signal_price = bars_30m['close'].iloc[i]
            limit_price = signal_price

            # Entry fill
            signal_bar_end = signal_ts + pd.Timedelta(minutes=ENTRY_BAR_SIZE)
            cancel_ts = signal_ts + pd.Timedelta(minutes=CANCEL_WINDOW_MIN + ENTRY_BAR_SIZE)

            day_ts = minute_df['ts_minute'].values
            fill_mask = (day_ts >= np.datetime64(signal_bar_end)) & (day_ts <= np.datetime64(cancel_ts))
            fill_candidates = minute_df[fill_mask]

            if len(fill_candidates) == 0:
                continue

            filled = False
            fill_price = None
            fill_ts = None
            fill_delay = 0

            for j, (_, mbar) in enumerate(fill_candidates.iterrows()):
                if direction == 1:
                    if mbar['low'] <= limit_price - TICK_SIZE:
                        filled = True
                        fill_price = limit_price
                        fill_ts = mbar['ts_minute']
                        fill_delay = j + 1
                        break
                else:
                    if mbar['high'] >= limit_price + TICK_SIZE:
                        filled = True
                        fill_price = limit_price
                        fill_ts = mbar['ts_minute']
                        fill_delay = j + 1
                        break

            if not filled:
                continue

            fill_ts_np = np.datetime64(fill_ts)
            remaining_mask = day_ts >= fill_ts_np
            remaining = minute_df[remaining_mask].head(MAX_HOLD_BARS)

            if len(remaining) < 2:
                continue

            # MFE/MAE
            if direction == 1:
                excursion = (remaining['high'].values - fill_price) / TICK_SIZE
                adverse = (fill_price - remaining['low'].values) / TICK_SIZE
            else:
                excursion = (fill_price - remaining['low'].values) / TICK_SIZE
                adverse = (remaining['high'].values - fill_price) / TICK_SIZE

            mfe = float(np.max(excursion))
            mae = float(np.max(adverse))

            # MFE/MAE timing (bars to peak)
            mfe_bar = int(np.argmax(excursion))
            mae_bar = int(np.argmax(adverse))

            # Cumulative MFE at various checkpoints
            mfe_5bar = float(np.max(excursion[:min(5, len(excursion))]))
            mfe_10bar = float(np.max(excursion[:min(10, len(excursion))]))
            mfe_20bar = float(np.max(excursion[:min(20, len(excursion))]))

            # Apply TP/SL exit
            sl_ticks = SL_LONG if direction == 1 else SL_SHORT
            exit_ticks = None
            exit_bar = len(remaining) - 1
            exit_type = 'max_hold'

            for j in range(len(remaining)):
                bar = remaining.iloc[j]
                if direction == 1:
                    bar_mfe = (bar['high'] - fill_price) / TICK_SIZE
                    bar_mae = (fill_price - bar['low']) / TICK_SIZE
                else:
                    bar_mfe = (fill_price - bar['low']) / TICK_SIZE
                    bar_mae = (bar['high'] - fill_price) / TICK_SIZE

                if bar_mfe >= TP_TICKS:
                    exit_ticks = TP_TICKS - RT_COMMISSION_TICKS
                    exit_bar = j
                    exit_type = 'tp'
                    break
                if bar_mae >= sl_ticks:
                    exit_ticks = -sl_ticks - RT_COMMISSION_TICKS - MARKET_SL_EXTRA
                    exit_bar = j
                    exit_type = 'sl'
                    break

            if exit_ticks is None:
                final = (remaining['close'].iloc[-1] - fill_price) / TICK_SIZE
                if direction == -1:
                    final = -final
                exit_ticks = final - RT_COMMISSION_TICKS
                exit_type = 'max_hold'

            # Context features at entry time
            fill_ts_pd = pd.Timestamp(fill_ts)
            hour = fill_ts_pd.hour
            minute = fill_ts_pd.minute
            time_of_day_minutes = (hour - 9) * 60 + minute  # minutes since 9:00 ET (approx)

            # Recent vol (30-min realized vol before signal)
            pre_signal_mask = day_ts < np.datetime64(signal_ts)
            pre_bars = minute_df[pre_signal_mask].tail(30)
            if len(pre_bars) > 5:
                recent_vol = float(pre_bars['close'].pct_change().std() * np.sqrt(len(pre_bars)))
                recent_range = float((pre_bars['high'].max() - pre_bars['low'].min()) / TICK_SIZE)
            else:
                recent_vol = np.nan
                recent_range = np.nan

            # Recent OFI (last 30 bars)
            if len(pre_bars) > 5 and 'volume' in pre_bars.columns:
                price_chg = pre_bars['close'].diff()
                recent_ofi = float((price_chg * pre_bars['volume']).sum())
            else:
                recent_ofi = np.nan

            # Volume at signal bar
            signal_volume = float(bars_30m['volume'].iloc[i]) if 'volume' in bars_30m.columns else np.nan

            # Day of week (0=Mon, 4=Fri)
            try:
                day_of_week = pd.Timestamp(date, tz='UTC').dayofweek
            except Exception:
                day_of_week = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]}").dayofweek

            trades.append({
                'date': date,
                'direction': direction,
                'pred_magnitude': abs(float(date_preds[i])),
                'pred_raw': float(date_preds[i]),
                'fill_price': fill_price,
                'fill_delay_bars': fill_delay,
                'time_of_day_min': time_of_day_minutes,
                'hour': hour,
                'day_of_week': day_of_week,
                'recent_vol': recent_vol,
                'recent_range': recent_range,
                'recent_ofi': recent_ofi,
                'signal_volume': signal_volume,
                'mfe_ticks': mfe,
                'mae_ticks': mae,
                'mfe_bar': mfe_bar,
                'mae_bar': mae_bar,
                'mfe_5bar': mfe_5bar,
                'mfe_10bar': mfe_10bar,
                'mfe_20bar': mfe_20bar,
                'exit_ticks': exit_ticks,
                'exit_bar': exit_bar,
                'exit_type': exit_type,
                'winner': exit_ticks > 0,
            })

    df = pd.DataFrame(trades)
    log.info(f"Reconstructed {len(df)} trades with full context over {df['date'].nunique()} dates")
    return df


def load_minute_bars(date: str) -> pd.DataFrame | None:
    p = MINUTE_BAR_DIR / f"{date}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df['ts_minute'] = pd.to_datetime(df['ts_minute'], utc=True)
    df = df.sort_values('ts_minute').reset_index(drop=True)
    return df


def aggregate_to_30min(minute_df: pd.DataFrame) -> pd.DataFrame:
    df = minute_df.copy()
    df['bar_key'] = df['ts_minute'].dt.floor('30min')
    agg = df.groupby('bar_key').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).reset_index()
    agg['ts'] = agg['bar_key']
    return agg


def compute_daily_ofi(minute_df: pd.DataFrame) -> float:
    if 'volume' not in minute_df.columns or len(minute_df) < 10:
        return 0.0
    price_chg = minute_df['close'].diff()
    return float((price_chg * minute_df['volume']).sum())


def compute_rolling_vol(minute_df: pd.DataFrame) -> float:
    if len(minute_df) < 10:
        return 0.0
    return float(minute_df['close'].pct_change().std())


def analyze_trades(df: pd.DataFrame):
    """Run comprehensive win/loss analysis."""
    log.info("\n" + "=" * 70)
    log.info("WIN/LOSS CHARACTERIZATION")
    log.info("=" * 70)

    winners = df[df['winner']]
    losers = df[~df['winner']]
    log.info(f"\nTotal: {len(df)} trades, {len(winners)} winners ({len(winners)/len(df):.1%}), "
             f"{len(losers)} losers ({len(losers)/len(df):.1%})")
    log.info(f"Exit types: {df['exit_type'].value_counts().to_dict()}")

    # 1. Direction analysis
    log.info("\n--- 1. DIRECTION ---")
    for d, label in [(1, 'LONG'), (-1, 'SHORT')]:
        sub = df[df['direction'] == d]
        w = sub['winner'].mean() if len(sub) > 0 else 0
        avg = sub['exit_ticks'].mean() if len(sub) > 0 else 0
        log.info(f"  {label}: {len(sub)} trades, WR {w:.1%}, avg {avg:.2f} t/trade")

    # 2. Prediction magnitude
    log.info("\n--- 2. PREDICTION MAGNITUDE ---")
    df['pred_quintile'] = pd.qcut(df['pred_magnitude'], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
    for q in sorted(df['pred_quintile'].unique()):
        sub = df[df['pred_quintile'] == q]
        log.info(f"  {q}: {len(sub)} trades, WR {sub['winner'].mean():.1%}, "
                f"avg {sub['exit_ticks'].mean():.2f}, pred_mag {sub['pred_magnitude'].mean():.2f}")

    # 3. Time of day
    log.info("\n--- 3. TIME OF DAY ---")
    df['session'] = pd.cut(df['hour'], bins=[0, 10, 12, 14, 24],
                          labels=['early(9-10)', 'mid(10-12)', 'afternoon(12-14)', 'late(14+)'],
                          right=False)
    for s in df['session'].cat.categories:
        sub = df[df['session'] == s]
        if len(sub) > 0:
            log.info(f"  {s}: {len(sub)} trades, WR {sub['winner'].mean():.1%}, "
                    f"avg {sub['exit_ticks'].mean():.2f}")

    # 4. Recent volatility
    log.info("\n--- 4. RECENT VOLATILITY (30-min pre-signal) ---")
    valid_vol = df.dropna(subset=['recent_vol'])
    if len(valid_vol) > 20:
        valid_vol['vol_tercile'] = pd.qcut(valid_vol['recent_vol'], 3,
                                          labels=['low_vol', 'mid_vol', 'high_vol'], duplicates='drop')
        for t in sorted(valid_vol['vol_tercile'].unique()):
            sub = valid_vol[valid_vol['vol_tercile'] == t]
            log.info(f"  {t}: {len(sub)} trades, WR {sub['winner'].mean():.1%}, "
                    f"avg {sub['exit_ticks'].mean():.2f}, vol {sub['recent_vol'].mean():.6f}")

    # 5. Recent range (ticks)
    log.info("\n--- 5. RECENT RANGE (30-min pre-signal, ticks) ---")
    valid_range = df.dropna(subset=['recent_range'])
    if len(valid_range) > 20:
        valid_range['range_tercile'] = pd.qcut(valid_range['recent_range'], 3,
                                              labels=['narrow', 'medium', 'wide'], duplicates='drop')
        for t in sorted(valid_range['range_tercile'].unique()):
            sub = valid_range[valid_range['range_tercile'] == t]
            log.info(f"  {t}: {len(sub)} trades, WR {sub['winner'].mean():.1%}, "
                    f"avg {sub['exit_ticks'].mean():.2f}, range {sub['recent_range'].mean():.1f}t")

    # 6. Fill delay
    log.info("\n--- 6. FILL DELAY ---")
    df['fill_speed'] = pd.cut(df['fill_delay_bars'], bins=[-1, 1, 3, 100],
                             labels=['instant(1bar)', 'fast(2-3bar)', 'slow(4+bar)'])
    for s in df['fill_speed'].cat.categories:
        sub = df[df['fill_speed'] == s]
        if len(sub) > 0:
            log.info(f"  {s}: {len(sub)} trades, WR {sub['winner'].mean():.1%}, "
                    f"avg {sub['exit_ticks'].mean():.2f}")

    # 7. Day of week
    log.info("\n--- 7. DAY OF WEEK ---")
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    for d in sorted(df['day_of_week'].unique()):
        sub = df[df['day_of_week'] == d]
        log.info(f"  {dow_names.get(d, d)}: {len(sub)} trades, WR {sub['winner'].mean():.1%}, "
                f"avg {sub['exit_ticks'].mean():.2f}")

    # 8. MFE trajectory (how fast do winners move?)
    log.info("\n--- 8. MFE TRAJECTORY ---")
    log.info(f"  Winners MFE@5bar: {winners['mfe_5bar'].mean():.1f}t, "
            f"@10bar: {winners['mfe_10bar'].mean():.1f}t, "
            f"@20bar: {winners['mfe_20bar'].mean():.1f}t, "
            f"peak: {winners['mfe_ticks'].mean():.1f}t at bar {winners['mfe_bar'].mean():.0f}")
    log.info(f"  Losers  MFE@5bar: {losers['mfe_5bar'].mean():.1f}t, "
            f"@10bar: {losers['mfe_10bar'].mean():.1f}t, "
            f"@20bar: {losers['mfe_20bar'].mean():.1f}t, "
            f"peak: {losers['mfe_ticks'].mean():.1f}t at bar {losers['mfe_bar'].mean():.0f}")

    # 9. Exit analysis
    log.info("\n--- 9. EXIT TYPE BREAKDOWN ---")
    for et in ['tp', 'sl', 'max_hold']:
        sub = df[df['exit_type'] == et]
        if len(sub) > 0:
            log.info(f"  {et}: {len(sub)} trades ({len(sub)/len(df):.1%}), "
                    f"avg exit {sub['exit_ticks'].mean():.2f}t")

    # 10. Winner prediction for potential filter
    log.info("\n--- 10. POTENTIAL FILTERS (single-feature) ---")

    # Test various single-feature filters
    filters_to_test = [
        ('pred_magnitude > median', df['pred_magnitude'] > df['pred_magnitude'].median()),
        ('pred_magnitude > p75', df['pred_magnitude'] > df['pred_magnitude'].quantile(0.75)),
        ('fill_delay == 1 bar', df['fill_delay_bars'] == 1),
        ('fill_delay <= 2 bars', df['fill_delay_bars'] <= 2),
        ('hour <= 11 (morning)', df['hour'] <= 11),
        ('hour >= 12 (afternoon)', df['hour'] >= 12),
    ]

    # Add vol-based filters if available
    valid_vol = df.dropna(subset=['recent_vol'])
    if len(valid_vol) > 20:
        vol_med = valid_vol['recent_vol'].median()
        filters_to_test.append(('low_vol (below median)', df['recent_vol'] < vol_med))
        filters_to_test.append(('high_vol (above median)', df['recent_vol'] >= vol_med))

    # Add range-based filters if available
    valid_range = df.dropna(subset=['recent_range'])
    if len(valid_range) > 20:
        range_med = valid_range['recent_range'].median()
        filters_to_test.append(('narrow_range (below median)', df['recent_range'] < range_med))
        filters_to_test.append(('wide_range (above median)', df['recent_range'] >= range_med))

    log.info(f"  {'Filter':<35} {'N':>4} {'WR':>7} {'Avg':>7} {'Sharpe':>8}")
    log.info(f"  {'-'*35} {'-'*4} {'-'*7} {'-'*7} {'-'*8}")
    log.info(f"  {'BASELINE (all trades)':<35} {len(df):>4} {df['winner'].mean():>7.1%} "
            f"{df['exit_ticks'].mean():>7.2f} {'--':>8}")

    for name, mask in filters_to_test:
        sub = df[mask]
        if len(sub) >= 10:
            wr = sub['winner'].mean()
            avg = sub['exit_ticks'].mean()
            day_pnl = sub.groupby('date')['exit_ticks'].sum()
            if len(day_pnl) > 1 and day_pnl.std() > 0:
                sharpe = day_pnl.mean() / day_pnl.std() * np.sqrt(252)
            else:
                sharpe = np.nan
            log.info(f"  {name:<35} {len(sub):>4} {wr:>7.1%} {avg:>7.2f} {sharpe:>8.2f}")

    # 11. Two-feature combinations
    log.info("\n--- 11. TWO-FEATURE COMBINATIONS ---")
    combos = [
        ('morning + high pred', (df['hour'] <= 11) & (df['pred_magnitude'] > df['pred_magnitude'].median())),
        ('morning + instant fill', (df['hour'] <= 11) & (df['fill_delay_bars'] == 1)),
        ('high pred + instant fill', (df['pred_magnitude'] > df['pred_magnitude'].median()) & (df['fill_delay_bars'] == 1)),
        ('afternoon + short only', (df['hour'] >= 12) & (df['direction'] == -1)),
        ('morning + long only', (df['hour'] <= 11) & (df['direction'] == 1)),
    ]

    if len(valid_vol) > 20:
        vol_med = valid_vol['recent_vol'].median()
        combos.append(('low_vol + high pred', (df['recent_vol'] < vol_med) & (df['pred_magnitude'] > df['pred_magnitude'].median())))
        combos.append(('high_vol + high pred', (df['recent_vol'] >= vol_med) & (df['pred_magnitude'] > df['pred_magnitude'].median())))

    log.info(f"  {'Combo':<35} {'N':>4} {'WR':>7} {'Avg':>7} {'Sharpe':>8}")
    log.info(f"  {'-'*35} {'-'*4} {'-'*7} {'-'*7} {'-'*8}")

    for name, mask in combos:
        sub = df[mask]
        if len(sub) >= 10:
            wr = sub['winner'].mean()
            avg = sub['exit_ticks'].mean()
            day_pnl = sub.groupby('date')['exit_ticks'].sum()
            if len(day_pnl) > 1 and day_pnl.std() > 0:
                sharpe = day_pnl.mean() / day_pnl.std() * np.sqrt(252)
            else:
                sharpe = np.nan
            log.info(f"  {name:<35} {len(sub):>4} {wr:>7.1%} {avg:>7.2f} {sharpe:>8.2f}")

    # Save enriched trades
    df.to_parquet(OUT_DIR / "trades_enriched.parquet", index=False)
    log.info(f"\nSaved enriched trades to {OUT_DIR / 'trades_enriched.parquet'}")


def main():
    log.info("=" * 70)
    log.info("Win/Loss Characterization Study v1")
    log.info("=" * 70)

    df = reconstruct_all_trades()

    if len(df) == 0:
        log.error("No trades reconstructed!")
        return

    analyze_trades(df)
    log.info("\nDONE")


if __name__ == "__main__":
    main()
