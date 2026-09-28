#!/usr/bin/env python3
"""
vol_gated_champion_v1.py — Volatility-gated champion strategy

Adds a walk-forward volatility gate to the champion 30-min LightGBM strategy.
The gate filters OUT trades in low-volatility environments where the 25-tick TP
is unlikely to be reached before the tight SL fires.

Key: The vol threshold is computed WALK-FORWARD (rolling 20-day lookback) to
avoid in-sample contamination. We do NOT use a single fixed threshold.

Sweep parameters:
  - vol_metric: 'range_ticks' (30-min pre-signal high-low range) or 'realized_vol'
  - vol_percentile: [30, 40, 50, 60] — minimum percentile of recent vol to accept trade
  - lookback_days: [10, 20, 30] — days to compute rolling vol distribution

HC #428: Reports per-regime Sharpe, regime gap. Uses all 60 OOT days.
HC #432: Vol gate is at entry time only, no future information.
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
ENRICHED_PATH = ROOT / "output" / "winloss_characterization_v1" / "trades_enriched.parquet"
OUT_DIR = ROOT / "output" / "vol_gated_champion_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / f"vol_gated_champion_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("vol_gated")

# Cost constants
RT_COMMISSION_TICKS = 0.376
MARKET_SL_EXTRA = 1.0

# Sweep params
VOL_METRICS = ['recent_range']  # ticks range in 30-min pre-signal window
VOL_PERCENTILES = [30, 40, 50, 60, 70]
LOOKBACK_DAYS = [10, 20, 30]


def load_enriched_trades() -> pd.DataFrame:
    """Load the enriched trade data from winloss characterization."""
    df = pd.read_parquet(ENRICHED_PATH)
    log.info(f"Loaded {len(df)} enriched trades over {df['date'].nunique()} dates")
    return df


def compute_daily_vol_stats(df: pd.DataFrame) -> dict:
    """Compute per-date volatility statistics for walk-forward thresholds."""
    daily_stats = {}
    for date in df['date'].unique():
        sub = df[df['date'] == date]
        daily_stats[date] = {
            'mean_range': sub['recent_range'].mean(),
            'mean_vol': sub['recent_vol'].mean(),
            'median_range': sub['recent_range'].median(),
            'n_trades': len(sub),
        }
    return daily_stats


def walk_forward_vol_gate(df: pd.DataFrame, vol_metric: str, percentile: int,
                          lookback: int) -> pd.Series:
    """
    Compute walk-forward vol gate: for each trade, is the vol_metric
    above the `percentile`-th percentile of the vol_metric over the
    prior `lookback` trading days?

    Returns boolean mask (True = trade passes gate).
    """
    dates = sorted(df['date'].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # Collect per-trade vol values indexed by date
    trade_vol_by_date = {}
    for _, row in df.iterrows():
        d = row['date']
        if d not in trade_vol_by_date:
            trade_vol_by_date[d] = []
        trade_vol_by_date[d].append(row[vol_metric])

    # Compute rolling threshold for each date
    date_thresholds = {}
    for i, date in enumerate(dates):
        # Lookback: gather vol values from prior `lookback` dates
        start_idx = max(0, i - lookback)
        lookback_dates = dates[start_idx:i]  # EXCLUDE current date (walk-forward)

        if len(lookback_dates) < 5:
            # Not enough history — use a lenient threshold (pass everything)
            date_thresholds[date] = -np.inf
            continue

        lookback_vols = []
        for ld in lookback_dates:
            if ld in trade_vol_by_date:
                lookback_vols.extend(trade_vol_by_date[ld])

        if len(lookback_vols) < 10:
            date_thresholds[date] = -np.inf
            continue

        date_thresholds[date] = np.percentile(lookback_vols, percentile)

    # Apply gate
    mask = np.zeros(len(df), dtype=bool)
    for i, (_, row) in enumerate(df.iterrows()):
        thresh = date_thresholds.get(row['date'], -np.inf)
        val = row[vol_metric]
        mask[i] = (not np.isnan(val)) and (val >= thresh)

    return pd.Series(mask, index=df.index)


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute risk-adjusted metrics."""
    if len(df) < 5:
        return {'n': 0, 'n_dates': 0, 'sharpe': np.nan, 'sortino': np.nan,
                'wr': np.nan, 'pf': np.nan, 'avg': np.nan, 'total': 0.0,
                'day_conc': np.nan}

    exits = df['exit_ticks'].values
    n = len(exits)
    n_dates = df['date'].nunique()

    day_pnl = df.groupby('date')['exit_ticks'].sum()
    sharpe = float(day_pnl.mean() / day_pnl.std() * np.sqrt(252)) if day_pnl.std() > 0 else np.nan

    downside = day_pnl[day_pnl < 0]
    ds_std = downside.std() if len(downside) > 1 else 0.001
    sortino = float(day_pnl.mean() / ds_std * np.sqrt(252)) if ds_std > 0 else np.nan

    wr = float(np.mean(exits > 0))
    gross_win = float(np.sum(exits[exits > 0]))
    gross_loss = float(np.abs(np.sum(exits[exits < 0])))
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf

    day_counts = df.groupby('date').size()
    day_conc = float(day_counts.max() / n) if n > 0 else 0.0

    return {
        'n': n, 'n_dates': n_dates,
        'sharpe': sharpe, 'sortino': sortino,
        'wr': wr, 'pf': pf,
        'avg': float(np.mean(exits)), 'total': float(np.sum(exits)),
        'day_conc': day_conc,
    }


def compute_regime_metrics(df: pd.DataFrame) -> dict:
    """
    Compute regime-stratified metrics.
    Classify each date as green/red based on day's net PnL of ES
    (approximate: use our own day net as proxy — imperfect but directional).
    """
    if len(df) < 10:
        return {'sharpe_green': np.nan, 'sharpe_red': np.nan, 'regime_gap': np.nan}

    day_pnl = df.groupby('date')['exit_ticks'].sum()
    green_dates = set(day_pnl[day_pnl > 0].index)
    red_dates = set(day_pnl[day_pnl <= 0].index)

    green_df = df[df['date'].isin(green_dates)]
    red_df = df[df['date'].isin(red_dates)]

    if len(green_dates) < 3 or len(red_dates) < 3:
        return {'sharpe_green': np.nan, 'sharpe_red': np.nan, 'regime_gap': np.nan}

    green_day_pnl = green_df.groupby('date')['exit_ticks'].sum()
    red_day_pnl = red_df.groupby('date')['exit_ticks'].sum()

    sg = float(green_day_pnl.mean() / green_day_pnl.std() * np.sqrt(252)) if green_day_pnl.std() > 0 else np.nan
    sr = float(red_day_pnl.mean() / red_day_pnl.std() * np.sqrt(252)) if red_day_pnl.std() > 0 else np.nan

    if np.isnan(sg) or np.isnan(sr):
        gap = np.nan
    else:
        denom = max(abs(sg), abs(sr))
        gap = abs(sg - sr) / denom if denom > 0 else 0.0

    return {'sharpe_green': sg, 'sharpe_red': sr, 'regime_gap': gap}


def main():
    log.info("=" * 70)
    log.info("Volatility-Gated Champion Strategy v1")
    log.info("Walk-Forward Vol Gate Sweep")
    log.info("=" * 70)

    df = load_enriched_trades()

    # Baseline
    baseline = compute_metrics(df)
    baseline_regime = compute_regime_metrics(df)
    log.info(f"\n--- BASELINE ---")
    log.info(f"  {baseline['n']} trades, {baseline['n_dates']} dates")
    log.info(f"  Sharpe: {baseline['sharpe']:.2f}, Sortino: {baseline['sortino']:.2f}")
    log.info(f"  WR: {baseline['wr']:.1%}, PF: {baseline['pf']:.2f}")
    log.info(f"  Avg: {baseline['avg']:.2f}, Total: {baseline['total']:.1f}")
    log.info(f"  Regime — Green: {baseline_regime['sharpe_green']:.2f}, "
             f"Red: {baseline_regime['sharpe_red']:.2f}, Gap: {baseline_regime['regime_gap']:.2f}")
    log.info(f"  Day conc: {baseline['day_conc']:.2f}")

    # Direction-split baseline
    for d, label in [(1, 'LONG'), (-1, 'SHORT')]:
        sub = df[df['direction'] == d]
        m = compute_metrics(sub)
        log.info(f"  {label}: {m['n']} trades, WR {m['wr']:.1%}, Sharpe {m['sharpe']:.2f}")

    results = []

    # Sweep
    for vol_metric in VOL_METRICS:
        for lookback in LOOKBACK_DAYS:
            for pct in VOL_PERCENTILES:
                arm_name = f"{vol_metric}|lb={lookback}d|pct>={pct}"

                mask = walk_forward_vol_gate(df, vol_metric, pct, lookback)
                gated = df[mask]
                blocked = df[~mask]

                gm = compute_metrics(gated)
                bm = compute_metrics(blocked)
                gr = compute_regime_metrics(gated)

                log.info(f"\n  ARM: {arm_name}")
                log.info(f"    Passed: {gm['n']} trades ({gm['n']/baseline['n']:.0%}), "
                        f"Sharpe: {gm['sharpe']:.2f}, WR: {gm['wr']:.1%}, PF: {gm['pf']:.2f}")
                log.info(f"    Blocked: {bm['n']} trades, Sharpe: {bm['sharpe']:.2f}, WR: {bm['wr']:.1%}")
                log.info(f"    Regime: Green {gr['sharpe_green']:.2f}, Red {gr['sharpe_red']:.2f}, "
                        f"Gap: {gr['regime_gap']:.2f}")

                results.append({
                    'arm': arm_name,
                    'vol_metric': vol_metric,
                    'lookback': lookback,
                    'percentile': pct,
                    'n_passed': gm['n'],
                    'n_dates': gm['n_dates'],
                    'pass_rate': gm['n'] / max(1, baseline['n']),
                    'sharpe': gm['sharpe'],
                    'sortino': gm['sortino'],
                    'wr': gm['wr'],
                    'pf': gm['pf'],
                    'avg': gm['avg'],
                    'total': gm['total'],
                    'day_conc': gm['day_conc'],
                    'sharpe_green': gr['sharpe_green'],
                    'sharpe_red': gr['sharpe_red'],
                    'regime_gap': gr['regime_gap'],
                    'blocked_n': bm['n'],
                    'blocked_sharpe': bm['sharpe'],
                    'blocked_wr': bm['wr'],
                    'sharpe_lift': gm['sharpe'] - baseline['sharpe'],
                    'wr_lift': gm['wr'] - baseline['wr'],
                })

    # Also test direction filter (long-only) + vol
    log.info("\n\n--- DIRECTION + VOL COMBOS ---")
    for vol_metric in VOL_METRICS:
        for lookback in [20]:  # Just test the middle lookback
            for pct in [40, 50]:
                arm_name = f"long_only+{vol_metric}|lb={lookback}d|pct>={pct}"
                vol_mask = walk_forward_vol_gate(df, vol_metric, pct, lookback)
                long_mask = df['direction'] == 1
                combo_mask = vol_mask & long_mask

                gated = df[combo_mask]
                gm = compute_metrics(gated)
                gr = compute_regime_metrics(gated)

                log.info(f"\n  ARM: {arm_name}")
                log.info(f"    Passed: {gm['n']} trades ({gm['n']/baseline['n']:.0%}), "
                        f"Sharpe: {gm['sharpe']:.2f}, WR: {gm['wr']:.1%}, PF: {gm['pf']:.2f}")
                log.info(f"    Regime: Green {gr['sharpe_green']:.2f}, Red {gr['sharpe_red']:.2f}, "
                        f"Gap: {gr['regime_gap']:.2f}")

                results.append({
                    'arm': arm_name,
                    'vol_metric': vol_metric,
                    'lookback': lookback,
                    'percentile': pct,
                    'n_passed': gm['n'],
                    'n_dates': gm['n_dates'],
                    'pass_rate': gm['n'] / max(1, baseline['n']),
                    'sharpe': gm['sharpe'],
                    'sortino': gm['sortino'],
                    'wr': gm['wr'],
                    'pf': gm['pf'],
                    'avg': gm['avg'],
                    'total': gm['total'],
                    'day_conc': gm['day_conc'],
                    'sharpe_green': gr['sharpe_green'],
                    'sharpe_red': gr['sharpe_red'],
                    'regime_gap': gr['regime_gap'],
                    'blocked_n': baseline['n'] - gm['n'],
                    'blocked_sharpe': np.nan,
                    'blocked_wr': np.nan,
                    'sharpe_lift': gm['sharpe'] - baseline['sharpe'],
                    'wr_lift': gm['wr'] - baseline['wr'],
                })

    # Summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "vol_gate_sweep.csv", index=False)

    log.info("\n\n" + "=" * 70)
    log.info("SUMMARY — BEST ARMS (min 50 trades, regime gap <= 0.50)")
    log.info("=" * 70)

    valid = results_df[(results_df['n_passed'] >= 50) & (results_df['regime_gap'] <= 0.50)].copy()
    if len(valid) > 0:
        valid = valid.sort_values('sharpe', ascending=False)
        log.info(f"\n{'ARM':<50} {'N':>4} {'Sharpe':>7} {'WR':>6} {'PF':>5} {'Gap':>5}")
        log.info(f"{'-'*50} {'-'*4} {'-'*7} {'-'*6} {'-'*5} {'-'*5}")
        for _, row in valid.head(10).iterrows():
            log.info(f"{row['arm']:<50} {row['n_passed']:>4} {row['sharpe']:>7.2f} "
                    f"{row['wr']:>6.1%} {row['pf']:>5.2f} {row['regime_gap']:>5.2f}")
    else:
        # Relax regime gate constraint
        log.info("\nNo arms pass regime gap <= 0.50 with >= 50 trades. Showing all with >= 50 trades:")
        valid = results_df[results_df['n_passed'] >= 50].sort_values('sharpe', ascending=False)
        log.info(f"\n{'ARM':<50} {'N':>4} {'Sharpe':>7} {'WR':>6} {'PF':>5} {'Gap':>5}")
        log.info(f"{'-'*50} {'-'*4} {'-'*7} {'-'*6} {'-'*5} {'-'*5}")
        for _, row in valid.head(10).iterrows():
            log.info(f"{row['arm']:<50} {row['n_passed']:>4} {row['sharpe']:>7.2f} "
                    f"{row['wr']:>6.1%} {row['pf']:>5.2f} {row['regime_gap']:>5.2f}")

    # Also show the regime-gap-constrained winner regardless of n
    all_valid = results_df[results_df['regime_gap'] <= 0.50].sort_values('sharpe', ascending=False)
    if len(all_valid) > 0:
        log.info(f"\nBest regime-balanced arm (gap <= 0.50): {all_valid.iloc[0]['arm']}")
        best = all_valid.iloc[0]
        log.info(f"  Trades: {best['n_passed']}, Sharpe: {best['sharpe']:.2f}, "
                f"WR: {best['wr']:.1%}, PF: {best['pf']:.2f}, "
                f"Green: {best['sharpe_green']:.2f}, Red: {best['sharpe_red']:.2f}, "
                f"Gap: {best['regime_gap']:.2f}")

    log.info(f"\nSaved sweep results to {OUT_DIR}")
    log.info("DONE")


if __name__ == "__main__":
    main()
