#!/usr/bin/env python3
"""
combined_refinement_v1.py — Combine patience entries + time/day filters

Tests the combination of two independent improvements:
1. Patience entries (wait for pullback) — from patience_entry_v1
2. Time/day filters (morning only, no Friday) — from strategy_refinement_v1

Also adds regime validation per the 40-day regime-agnostic rule.

Best single improvements found:
- Patience pb=2t, wait=1m, miss=market: Sharpe 16.8, WR 56%
- Morning + No Friday: WR 33.3%, regime gap 0.08

Run on Neptune.
"""

import logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/nick/Lvl3Quant")
MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
ENTRY_PREDS_PATH = ROOT / "output" / "mfe_mae_analysis" / "entry_predictions.npz"
TRADES_PATH = ROOT / "output" / "mfe_mae_analysis" / "trades_top_5pct.parquet"
OUT_DIR = ROOT / "output" / "combined_refinement_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / "combined_refinement_v1.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [COMBINED] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("combined")

# Champion params
TICK_SIZE = 1.0  # Data is in tick units (1 unit = 1 tick = 0.25 ES points)
ENTRY_BAR_SIZE = 30
CONFIDENCE_PCT = 0.05
TP_TICKS = 25
SL_LONG = 4
SL_SHORT = 3
MAX_HOLD_BARS = 60
RT_COMMISSION_TICKS = 0.376
MARKET_SL_EXTRA = 1.0

# Configs to test
CONFIGS = [
    # Baseline
    {'name': 'BASE', 'pb': 0, 'patience': 1, 'miss': 'market',
     'time_filter': None, 'day_filter': None},
    # Patience only (best arms)
    {'name': 'PATIENCE_2t', 'pb': 2, 'patience': 1, 'miss': 'market',
     'time_filter': None, 'day_filter': None},
    {'name': 'PATIENCE_5t', 'pb': 5, 'patience': 2, 'miss': 'skip',
     'time_filter': None, 'day_filter': None},
    # Time/day only
    {'name': 'MORNING_NO_FRI', 'pb': 0, 'patience': 1, 'miss': 'market',
     'time_filter': 13, 'day_filter': [4]},
    # Combined: patience + time/day
    {'name': 'COMBO_2t_MORNING', 'pb': 2, 'patience': 1, 'miss': 'market',
     'time_filter': 13, 'day_filter': None},
    {'name': 'COMBO_2t_NO_FRI', 'pb': 2, 'patience': 1, 'miss': 'market',
     'time_filter': None, 'day_filter': [4]},
    {'name': 'COMBO_2t_MORNING_NO_FRI', 'pb': 2, 'patience': 1, 'miss': 'market',
     'time_filter': 13, 'day_filter': [4]},
    {'name': 'COMBO_5t_MORNING_NO_FRI', 'pb': 5, 'patience': 2, 'miss': 'skip',
     'time_filter': 13, 'day_filter': [4]},
    # Mon-Wed only + patience
    {'name': 'COMBO_2t_MON_WED', 'pb': 2, 'patience': 1, 'miss': 'market',
     'time_filter': None, 'day_filter': [3, 4]},
    {'name': 'COMBO_2t_MORNING_MON_WED', 'pb': 2, 'patience': 1, 'miss': 'market',
     'time_filter': 13, 'day_filter': [3, 4]},
    # Lower TP + patience + filters
    {'name': 'COMBO_2t_MORNING_NO_FRI_TP22', 'pb': 2, 'patience': 1, 'miss': 'market',
     'time_filter': 13, 'day_filter': [4], 'tp_override': 22},
    {'name': 'COMBO_2t_MORNING_NO_FRI_TP20', 'pb': 2, 'patience': 1, 'miss': 'market',
     'time_filter': 13, 'day_filter': [4], 'tp_override': 20},
]


def load_minute_bars(date):
    p = MINUTE_BAR_DIR / f"{date}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df['ts_minute'] = pd.to_datetime(df['ts_minute'], utc=True)
    return df.sort_values('ts_minute').reset_index(drop=True)


def aggregate_to_30min(minute_df):
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
    if pullback_ticks == 0:
        limit_price = signal_price
    else:
        if direction == 1:
            limit_price = signal_price - pullback_ticks * TICK_SIZE
        else:
            limit_price = signal_price + pullback_ticks * TICK_SIZE

    day_ts = minute_df['ts_minute'].values
    signal_bar_end_np = np.datetime64(signal_bar_end)
    patience_end = signal_bar_end + pd.Timedelta(minutes=patience_min)
    patience_end_np = np.datetime64(patience_end)
    cancel_ts = signal_bar_end + pd.Timedelta(minutes=10)
    cancel_ts_np = np.datetime64(cancel_ts)

    # Phase 1: patience fill
    patience_mask = (day_ts >= signal_bar_end_np) & (day_ts <= patience_end_np)
    patience_bars = minute_df[patience_mask]

    for _, mbar in patience_bars.iterrows():
        if direction == 1:
            if mbar['low'] <= limit_price - TICK_SIZE:
                return limit_price, mbar['ts_minute'], True, False
        else:
            if mbar['high'] >= limit_price + TICK_SIZE:
                return limit_price, mbar['ts_minute'], True, False

    if miss_action == 'skip':
        return None

    # Phase 2: market fallback
    market_mask = (day_ts > patience_end_np) & (day_ts <= cancel_ts_np)
    market_bars = minute_df[market_mask]

    for _, mbar in market_bars.iterrows():
        if direction == 1:
            if mbar['low'] <= signal_price - TICK_SIZE:
                return signal_price, mbar['ts_minute'], False, True
        else:
            if mbar['high'] >= signal_price + TICK_SIZE:
                return signal_price, mbar['ts_minute'], False, True

    return None


def simulate_exit(direction, fill_price, fill_ts, minute_df, tp_ticks=TP_TICKS):
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

        if bar_mfe >= tp_ticks:
            return tp_ticks - RT_COMMISSION_TICKS
        if bar_mae >= sl_ticks:
            return -sl_ticks - RT_COMMISSION_TICKS - MARKET_SL_EXTRA

    if direction == 1:
        final = (remaining['close'].iloc[-1] - fill_price) / TICK_SIZE
    else:
        final = (fill_price - remaining['close'].iloc[-1]) / TICK_SIZE
    return final - RT_COMMISSION_TICKS


def get_et_hour(ts):
    """Convert UTC timestamp to approximate ET hour."""
    # EDT = UTC - 4
    return (pd.Timestamp(ts).hour - 4) % 24


def compute_regime_metrics(exits, dates, daily_returns):
    """Compute regime-stratified Sharpe."""
    if len(exits) < 10:
        return np.nan, np.nan, np.nan

    df_temp = pd.DataFrame({'date': dates, 'exit': exits})
    day_pnl = df_temp.groupby('date')['exit'].sum().reset_index()
    day_pnl['regime'] = day_pnl['date'].map(
        lambda d: 'green' if daily_returns.get(d, 0) > 4
        else ('red' if daily_returns.get(d, 0) < -4 else 'flat')
    )

    green = day_pnl[day_pnl['regime'] == 'green']['exit'].values
    red = day_pnl[day_pnl['regime'] == 'red']['exit'].values

    g_sharpe = green.mean() / green.std() * np.sqrt(252) if len(green) > 3 and green.std() > 0 else np.nan
    r_sharpe = red.mean() / red.std() * np.sqrt(252) if len(red) > 3 and red.std() > 0 else np.nan

    if not np.isnan(g_sharpe) and not np.isnan(r_sharpe):
        gap = abs(g_sharpe - r_sharpe) / max(abs(g_sharpe), abs(r_sharpe), 1e-6)
    else:
        gap = np.nan

    return g_sharpe, r_sharpe, gap


def main():
    log.info("=" * 70)
    log.info("COMBINED REFINEMENT v1 — Patience + Time/Day Filters")
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

    # Pre-load minute bars and compute daily returns for regime
    minute_cache = {}
    bar30_cache = {}
    daily_returns = {}
    for date in oot_dates:
        mb = load_minute_bars(date)
        if mb is not None:
            minute_cache[date] = mb
            bar30_cache[date] = aggregate_to_30min(mb)
            # Daily return in ticks
            daily_returns[date] = (mb['close'].iloc[-1] - mb['close'].iloc[0]) / TICK_SIZE
    log.info(f"Loaded {len(minute_cache)} dates, regime: "
             f"{sum(1 for v in daily_returns.values() if v > 4)} green, "
             f"{sum(1 for v in daily_returns.values() if v < -4)} red")

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

            signal_ts = pd.Timestamp(bars_30m['ts'].iloc[i])
            signals.append({
                'date': date,
                'signal_ts': signal_ts,
                'signal_bar_end': signal_ts + pd.Timedelta(minutes=ENTRY_BAR_SIZE),
                'signal_price': bars_30m['close'].iloc[i],
                'direction': direction,
                'pred': float(date_preds[i]),
                'et_hour': get_et_hour(signal_ts),
                'dow': signal_ts.dayofweek,
            })
    log.info(f"Total signals: {len(signals)}")

    # Run each config
    all_results = []

    for config in CONFIGS:
        name = config['name']
        pb = config['pb']
        patience = config['patience']
        miss = config['miss']
        time_filter = config.get('time_filter')
        day_filter = config.get('day_filter')
        tp = config.get('tp_override', TP_TICKS)

        exits = []
        dates = []
        n_filtered_time = 0
        n_filtered_day = 0
        n_patience_fills = 0
        n_market_fills = 0
        n_skipped = 0

        for sig in signals:
            # Time filter
            if time_filter is not None and sig['et_hour'] >= time_filter:
                n_filtered_time += 1
                continue

            # Day filter
            if day_filter is not None and sig['dow'] in day_filter:
                n_filtered_day += 1
                continue

            date = sig['date']
            if date not in minute_cache:
                continue

            result = simulate_patience_entry(
                sig['direction'], sig['signal_price'], sig['signal_bar_end'],
                minute_cache[date], pb, patience, miss
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
                sig['direction'], fill_price, fill_ts, minute_cache[date], tp_ticks=tp
            )

            if exit_ticks is not None:
                exits.append(exit_ticks)
                dates.append(date)

        if len(exits) < 10:
            log.info(f"\n{name}: too few trades ({len(exits)})")
            all_results.append({'name': name, 'n_trades': len(exits), 'error': 'too_few'})
            continue

        exits_arr = np.array(exits)
        n = len(exits_arr)
        wr = float(np.mean(exits_arr > 0))
        gross_win = float(np.sum(exits_arr[exits_arr > 0]))
        gross_loss = float(np.abs(np.sum(exits_arr[exits_arr < 0])))
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

        df_temp = pd.DataFrame({'date': dates, 'exit': exits})
        day_pnl = df_temp.groupby('date')['exit'].sum()
        sharpe = float(day_pnl.mean() / day_pnl.std() * np.sqrt(252)) if day_pnl.std() > 0 else np.nan
        sortino_denom = day_pnl[day_pnl < 0].std()
        sortino = float(day_pnl.mean() / sortino_denom * np.sqrt(252)) if sortino_denom and sortino_denom > 0 else np.nan

        total_pnl = float(exits_arr.sum())
        max_dd = float((np.maximum.accumulate(np.cumsum(exits_arr)) - np.cumsum(exits_arr)).max())

        g_sharpe, r_sharpe, regime_gap = compute_regime_metrics(exits, dates, daily_returns)
        regime_pass = regime_gap < 0.50 if not np.isnan(regime_gap) else False

        result = {
            'name': name,
            'n_trades': n,
            'wr': wr,
            'sharpe': sharpe,
            'sortino': sortino,
            'pf': pf,
            'total_pnl_ticks': total_pnl,
            'total_pnl_dollars': total_pnl * 12.50,  # 1 tick = $12.50
            'max_dd_ticks': max_dd,
            'avg_pnl': float(exits_arr.mean()),
            'green_sharpe': g_sharpe,
            'red_sharpe': r_sharpe,
            'regime_gap': regime_gap,
            'regime_pass': regime_pass,
            'n_filtered_time': n_filtered_time,
            'n_filtered_day': n_filtered_day,
            'n_patience_fills': n_patience_fills,
            'n_market_fills': n_market_fills,
            'n_skipped': n_skipped,
        }
        all_results.append(result)

        gap_str = f"{regime_gap:.3f}" if not np.isnan(regime_gap) else "N/A"
        pass_str = "PASS" if regime_pass else "FAIL"
        log.info(f"\n{name}:")
        log.info(f"  Trades={n}, WR={wr:.1%}, Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, PF={pf:.2f}")
        log.info(f"  PnL={total_pnl:.1f}t (${total_pnl * 12.50 / TICK_SIZE:.0f}), MaxDD={max_dd:.1f}t")
        log.info(f"  Green Sharpe={g_sharpe:.2f}, Red Sharpe={r_sharpe:.2f}, Gap={gap_str} {pass_str}")
        log.info(f"  Filtered: time={n_filtered_time}, day={n_filtered_day}, patience={n_patience_fills}, market={n_market_fills}, skip={n_skipped}")

    # Summary table
    log.info("\n" + "=" * 100)
    log.info("SUMMARY — ALL CONFIGS")
    log.info("=" * 100)
    log.info(f"{'Config':<35s} {'N':>4s} {'WR':>6s} {'Sharpe':>7s} {'Sortino':>8s} {'PF':>5s} "
             f"{'PnL$':>7s} {'G_Sh':>6s} {'R_Sh':>6s} {'Gap':>6s} {'Pass':>5s}")
    log.info("-" * 100)

    passing = []
    for r in sorted(all_results, key=lambda x: x.get('sharpe', 0), reverse=True):
        if 'error' in r:
            continue
        gap_str = f"{r['regime_gap']:.3f}" if not np.isnan(r.get('regime_gap', np.nan)) else "N/A"
        pass_str = "PASS" if r.get('regime_pass', False) else "FAIL"
        g_str = f"{r['green_sharpe']:.2f}" if not np.isnan(r.get('green_sharpe', np.nan)) else "N/A"
        r_str = f"{r['red_sharpe']:.2f}" if not np.isnan(r.get('red_sharpe', np.nan)) else "N/A"
        log.info(f"{r['name']:<35s} {r['n_trades']:>4d} {r['wr']:>6.1%} {r['sharpe']:>7.2f} "
                 f"{r['sortino']:>8.2f} {r['pf']:>5.2f} "
                 f"${r['total_pnl_dollars']:>6.0f} {g_str:>6s} {r_str:>6s} {gap_str:>6s} {pass_str:>5s}")
        if r.get('regime_pass', False):
            passing.append(r)

    if passing:
        best = max(passing, key=lambda x: x['sharpe'])
        base = next((r for r in all_results if r['name'] == 'BASE'), None)
        log.info(f"\nBEST REGIME-PASSING CONFIG: {best['name']}")
        log.info(f"  Sharpe: {best['sharpe']:.2f}, WR: {best['wr']:.1%}, PF: {best['pf']:.2f}")
        if base:
            log.info(f"  vs BASE: Sharpe {base.get('sharpe', 0):.2f}, WR {base.get('wr', 0):.1%}")

    # Save
    import json
    with open(OUT_DIR / "results.json", 'w') as f:
        json.dump({
            'results': all_results,
            'best_passing': best if passing else None,
        }, f, indent=2, default=str)

    log.info(f"\nDone. Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
