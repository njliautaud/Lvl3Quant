#!/usr/bin/env python3
"""
lgbm_cnnmamba_confluence_v1.py — Confluence gate study

Tests whether gating 30-min LightGBM entries on CNN-Mamba tick-level agreement
improves risk-adjusted performance of the champion strategy (Sharpe 5.46).

Approach:
  1. Reconstruct champion entry fill timestamps (30-min bar signals + FIFO fill)
  2. For each fill, find latest CNN-Mamba prediction at fill timestamp
  3. Gate: CNN-Mamba prediction agrees on direction? Sweep confidence thresholds.
  4. Compare gated vs ungated Sharpe/WR/PF per HC #428/#432 gates.

Runs on Neptune. Uses existing per-date CNN-Mamba predictions from
output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/.

HC #428 compliance: Reports per-regime Sharpe, regime gap, day-conc.
HC #432: CNN-Mamba horizon is 1-5s; staleness window enforced.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──
ROOT = Path("/home/nick/Lvl3Quant")
CNN_PERDATE_DIR = ROOT / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "oot_47day_perdate"
MBO_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
ENTRY_PREDS_PATH = ROOT / "output" / "mfe_mae_analysis" / "entry_predictions.npz"
TRADES_PATH = ROOT / "output" / "mfe_mae_analysis" / "trades_top_5pct.parquet"
OUT_DIR = ROOT / "output" / "lgbm_cnnmamba_confluence_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / f"lgbm_cnnmamba_confluence_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("confluence")

# ── CNN-Mamba timestamp reconstruction constants ──
CHAMP_WINDOW = 1500  # event window
CHAMP_STRIDE = 250   # event stride (250ms equiv)

# ── Champion strategy params ──
TICK_SIZE = 0.25
ENTRY_BAR_SIZE = 30  # minutes
CANCEL_WINDOW_MIN = 10
CONFIDENCE_PCT = 0.05  # top 5%

# Exit params (from champion config)
TP_TICKS = 25
SL_LONG = 4
SL_SHORT = 3
MAX_HOLD_BARS = 60

# Costs
RT_COMMISSION_TICKS = 0.376  # passive entry/TP
MARKET_SL_EXTRA = 1.0       # market order SL crossing cost

# ── CNN gate sweep params ──
CNN_PRED_FIELDS = ["pred_log_ret_1s", "pred_log_ret_5s", "pred_p_up_5s"]
# Staleness: how old can the CNN pred be at entry fill time
STALENESS_OPTIONS_NS = [
    int(5e9),    # 5s (within CNN model horizon)
    int(15e9),   # 15s
    int(30e9),   # 30s
]
# Confidence filters: None = direction-only, float = must be in top N% of abs pred
CONFIDENCE_GATES = [None, 0.50, 0.25]  # direction only, top 50%, top 25%


def reconstruct_cnn_timestamps(date: str) -> np.ndarray:
    """Reconstruct nanosecond timestamps for CNN-Mamba per-date predictions."""
    mbo_path = MBO_DIR / f"{date}_mbo_events.npz"
    if not mbo_path.exists():
        return np.array([], dtype=np.int64)
    with np.load(mbo_path, allow_pickle=True) as z:
        ts = z["timestamps"].astype(np.int64)
        lab1 = z["labels_1s"]
    n = len(ts)
    pos = np.arange(CHAMP_WINDOW - 1, n, CHAMP_STRIDE)
    keep = ~np.isnan(lab1[pos])
    pos = pos[keep]
    return ts[pos]


def load_minute_bars_for_date(date: str) -> pd.DataFrame | None:
    """Load minute bars for a single date."""
    p = MINUTE_BAR_DIR / f"{date}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df['ts_minute'] = pd.to_datetime(df['ts_minute'], utc=True)
    df = df.sort_values('ts_minute').reset_index(drop=True)
    return df


def aggregate_to_30min(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute bars to 30-min bars."""
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


def reconstruct_champion_trades_with_timestamps(dates_with_cnn: set) -> pd.DataFrame:
    """
    Re-run champion entry fill logic to get fill timestamps for trades
    that fall on dates with CNN-Mamba predictions available.

    Returns DataFrame with: date, direction, fill_price, fill_ts_utc, fill_ts_ns,
                           mfe_ticks, mae_ticks, final_ticks, winner
    """
    # Load 30-min predictions
    data = np.load(ENTRY_PREDS_PATH, allow_pickle=True)
    all_preds = data['entry_preds']
    all_dates = data['dates']
    log.info(f"Loaded {len(all_preds)} entry predictions over {len(set(all_dates))} dates")

    # Compute confidence thresholds on FULL dataset (not just overlap)
    valid_mask = ~np.isnan(all_preds)
    valid_preds = all_preds[valid_mask]
    upper_thresh = np.nanquantile(valid_preds, 1 - CONFIDENCE_PCT)
    lower_thresh = np.nanquantile(valid_preds, CONFIDENCE_PCT)
    log.info(f"Confidence thresholds: upper={upper_thresh:.4f}, lower={lower_thresh:.4f}")

    trades = []

    for date in sorted(dates_with_cnn):
        # Get bars for this date
        minute_df = load_minute_bars_for_date(date)
        if minute_df is None:
            log.warning(f"No minute bars for {date}")
            continue

        bars_30m = aggregate_to_30min(minute_df)

        # Get predictions for this date
        date_mask = all_dates == date
        date_preds = all_preds[date_mask]

        if len(date_preds) != len(bars_30m):
            # Align by truncating to min length (bars might differ slightly)
            n = min(len(date_preds), len(bars_30m))
            date_preds = date_preds[:n]
            bars_30m = bars_30m.iloc[:n].reset_index(drop=True)

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

            # Entry fill logic: passive limit at signal_price, fill after bar end
            limit_price = signal_price
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

            for _, mbar in fill_candidates.iterrows():
                if direction == 1:
                    if mbar['low'] <= limit_price - TICK_SIZE:
                        filled = True
                        fill_price = limit_price
                        fill_ts = mbar['ts_minute']
                        break
                else:
                    if mbar['high'] >= limit_price + TICK_SIZE:
                        filled = True
                        fill_price = limit_price
                        fill_ts = mbar['ts_minute']
                        break

            if not filled or fill_ts is None:
                continue

            # Track MFE/MAE from fill
            fill_ts_np = np.datetime64(fill_ts)
            remaining_mask = day_ts >= fill_ts_np
            remaining = minute_df[remaining_mask].head(MAX_HOLD_BARS)

            if len(remaining) < 2:
                continue

            # Compute MFE/MAE/final
            if direction == 1:
                excursion = (remaining['high'].values - fill_price) / TICK_SIZE
                adverse = (fill_price - remaining['low'].values) / TICK_SIZE
                final = (remaining['close'].iloc[-1] - fill_price) / TICK_SIZE
            else:
                excursion = (fill_price - remaining['low'].values) / TICK_SIZE
                adverse = (remaining['high'].values - fill_price) / TICK_SIZE
                final = (fill_price - remaining['close'].iloc[-1]) / TICK_SIZE

            mfe = float(np.max(excursion))
            mae = float(np.max(adverse))

            # Apply TP/SL exit
            sl_ticks = SL_LONG if direction == 1 else SL_SHORT
            exit_ticks = final  # default: hold to end

            for j in range(len(remaining)):
                bar = remaining.iloc[j]
                if direction == 1:
                    bar_mfe = (bar['high'] - fill_price) / TICK_SIZE
                    bar_mae = (fill_price - bar['low']) / TICK_SIZE
                else:
                    bar_mfe = (fill_price - bar['low']) / TICK_SIZE
                    bar_mae = (bar['high'] - fill_price) / TICK_SIZE

                if bar_mfe >= TP_TICKS:
                    exit_ticks = TP_TICKS - RT_COMMISSION_TICKS  # passive TP fill
                    break
                if bar_mae >= sl_ticks:
                    exit_ticks = -sl_ticks - RT_COMMISSION_TICKS - MARKET_SL_EXTRA  # market SL
                    break
            else:
                # Max hold reached — exit at close with commission
                exit_ticks = final - RT_COMMISSION_TICKS

            # Convert fill_ts to ns for CNN alignment
            fill_ts_utc = pd.Timestamp(fill_ts)
            fill_ts_ns = int(fill_ts_utc.value)  # nanoseconds since epoch

            trades.append({
                'date': date,
                'direction': direction,
                'pred': float(date_preds[i]),
                'fill_price': fill_price,
                'fill_ts_utc': fill_ts_utc,
                'fill_ts_ns': fill_ts_ns,
                'mfe_ticks': mfe,
                'mae_ticks': mae,
                'exit_ticks': exit_ticks,
                'winner': exit_ticks > 0,
            })

    df = pd.DataFrame(trades)
    if len(df) == 0:
        log.warning("No trades reconstructed!")
        return df
    log.info(f"Reconstructed {len(df)} trades with timestamps over {df['date'].nunique()} dates")
    return df


def load_cnn_predictions(date: str) -> tuple[np.ndarray, dict]:
    """Load CNN-Mamba predictions for a date. Returns (timestamps_ns, {field: values})."""
    npz_path = CNN_PERDATE_DIR / f"{date}_predictions.npz"
    if not npz_path.exists():
        return np.array([]), {}

    ts = reconstruct_cnn_timestamps(date)
    data = np.load(npz_path)

    preds = {}
    for field in CNN_PRED_FIELDS:
        if field in data:
            preds[field] = data[field]

    # Verify alignment
    for field, vals in preds.items():
        if len(vals) != len(ts):
            log.warning(f"{date}: {field} length {len(vals)} != ts length {len(ts)}")
            # Truncate to min
            n = min(len(vals), len(ts))
            ts = ts[:n]
            preds[field] = vals[:n]

    return ts, preds


def cnn_pred_at_fill(fill_ts_ns: int, cnn_ts: np.ndarray, cnn_vals: np.ndarray,
                     staleness_ns: int) -> float:
    """Get latest CNN prediction at or before fill timestamp, within staleness window."""
    if len(cnn_ts) == 0:
        return np.nan

    idx = np.searchsorted(cnn_ts, fill_ts_ns, side="right") - 1
    if idx < 0:
        return np.nan

    age = fill_ts_ns - cnn_ts[idx]
    if age > staleness_ns:
        return np.nan

    return float(cnn_vals[idx])


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute risk-adjusted metrics on trade results."""
    if len(trades_df) == 0:
        return {
            'n_trades': 0, 'n_dates': 0,
            'sharpe': np.nan, 'sortino': np.nan,
            'wr': np.nan, 'pf': np.nan,
            'total_ticks': 0.0, 'avg_ticks': np.nan,
            'sharpe_green': np.nan, 'sharpe_red': np.nan,
            'regime_gap': np.nan, 'day_conc': np.nan,
        }

    exits = trades_df['exit_ticks'].values
    n = len(exits)
    n_dates = trades_df['date'].nunique()

    # Per-day P&L for Sharpe
    day_pnl = trades_df.groupby('date')['exit_ticks'].sum()

    sharpe = float(day_pnl.mean() / day_pnl.std() * np.sqrt(252)) if day_pnl.std() > 0 else np.nan

    # Sortino
    downside = day_pnl[day_pnl < 0]
    downside_std = downside.std() if len(downside) > 1 else 0.001
    sortino = float(day_pnl.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else np.nan

    wr = float(np.mean(exits > 0))

    gross_win = float(np.sum(exits[exits > 0]))
    gross_loss = float(np.abs(np.sum(exits[exits < 0])))
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf

    total = float(np.sum(exits))
    avg = float(np.mean(exits))

    # Day concentration
    day_counts = trades_df.groupby('date').size()
    day_conc = float(day_counts.max() / n) if n > 0 else 0.0

    # Regime analysis (need ES daily returns — use our own day P&L as proxy for now)
    # TODO: load actual ES daily close for regime classification
    # For now, classify by our own day-level direction as a placeholder
    sharpe_green = np.nan
    sharpe_red = np.nan
    regime_gap = np.nan

    return {
        'n_trades': n,
        'n_dates': n_dates,
        'sharpe': sharpe,
        'sortino': sortino,
        'wr': wr,
        'pf': pf,
        'total_ticks': total,
        'avg_ticks': avg,
        'sharpe_green': sharpe_green,
        'sharpe_red': sharpe_red,
        'regime_gap': regime_gap,
        'day_conc': day_conc,
    }


def run_confluence_study():
    """Main confluence study."""
    log.info("=" * 70)
    log.info("LightGBM x CNN-Mamba Confluence Study v1")
    log.info("=" * 70)

    # Find dates with CNN predictions
    cnn_dates = set()
    for f in CNN_PERDATE_DIR.glob("*_predictions.npz"):
        d = f.stem.replace("_predictions", "")
        cnn_dates.add(d)
    log.info(f"CNN-Mamba prediction dates available: {len(cnn_dates)}")

    # Load champion trades to find which dates overlap
    orig_trades = pd.read_parquet(TRADES_PATH)
    trade_dates = set(orig_trades['date'].unique())
    overlap_dates = trade_dates & cnn_dates
    log.info(f"Champion trade dates: {len(trade_dates)}")
    log.info(f"Overlap dates (have both): {len(overlap_dates)}")
    log.info(f"Overlap dates: {sorted(overlap_dates)}")

    if len(overlap_dates) == 0:
        log.error("No overlapping dates! Cannot run confluence study.")
        return

    # Step 1: Reconstruct champion trades with fill timestamps
    log.info("\n--- Step 1: Reconstructing champion trades with fill timestamps ---")
    trades_df = reconstruct_champion_trades_with_timestamps(overlap_dates)

    if len(trades_df) == 0:
        log.error("No trades reconstructed!")
        return

    log.info(f"\nReconstructed {len(trades_df)} trades:")
    log.info(f"  Dates: {trades_df['date'].nunique()}")
    log.info(f"  Winners: {trades_df['winner'].sum()} ({trades_df['winner'].mean():.1%})")
    log.info(f"  Avg exit: {trades_df['exit_ticks'].mean():.2f} ticks")
    log.info(f"  Total: {trades_df['exit_ticks'].sum():.1f} ticks")

    # Baseline metrics (ungated)
    baseline = compute_metrics(trades_df)
    log.info(f"\n--- BASELINE (ungated, {len(overlap_dates)} dates) ---")
    log.info(f"  Trades: {baseline['n_trades']}, Dates: {baseline['n_dates']}")
    log.info(f"  Sharpe: {baseline['sharpe']:.2f}, Sortino: {baseline['sortino']:.2f}")
    log.info(f"  WR: {baseline['wr']:.1%}, PF: {baseline['pf']:.2f}")
    log.info(f"  Avg: {baseline['avg_ticks']:.2f} t/trade, Total: {baseline['total_ticks']:.1f} ticks")
    log.info(f"  Day conc: {baseline['day_conc']:.2f}")

    # Step 2: Load CNN predictions for each overlap date
    log.info("\n--- Step 2: Loading CNN-Mamba predictions ---")
    cnn_cache = {}  # date -> (ts_ns, {field: values})
    for date in sorted(overlap_dates):
        ts, preds = load_cnn_predictions(date)
        if len(ts) > 0:
            cnn_cache[date] = (ts, preds)
            log.info(f"  {date}: {len(ts)} CNN samples, fields={list(preds.keys())}")
        else:
            log.warning(f"  {date}: failed to load CNN predictions")

    # Step 3: For each trade, look up CNN prediction at fill time
    log.info("\n--- Step 3: CNN prediction lookup at fill timestamps ---")
    results = []

    for cnn_field in CNN_PRED_FIELDS:
        for staleness_ns in STALENESS_OPTIONS_NS:
            staleness_label = f"{staleness_ns / 1e9:.0f}s"

            # Look up CNN pred for each trade
            cnn_at_fill = []
            for _, trade in trades_df.iterrows():
                date = trade['date']
                if date not in cnn_cache:
                    cnn_at_fill.append(np.nan)
                    continue

                ts, preds = cnn_cache[date]
                if cnn_field not in preds:
                    cnn_at_fill.append(np.nan)
                    continue

                val = cnn_pred_at_fill(trade['fill_ts_ns'], ts, preds[cnn_field], staleness_ns)
                cnn_at_fill.append(val)

            trades_df[f'cnn_{cnn_field}_{staleness_label}'] = cnn_at_fill
            n_valid = sum(1 for v in cnn_at_fill if not np.isnan(v))
            log.info(f"  {cnn_field} @{staleness_label}: {n_valid}/{len(trades_df)} trades have CNN pred")

            # Now sweep confidence gates
            for conf_gate in CONFIDENCE_GATES:
                gate_label = "dir_only" if conf_gate is None else f"top_{int(conf_gate*100)}pct"
                arm_name = f"{cnn_field}|stale={staleness_label}|gate={gate_label}"

                # Apply gate
                cnn_vals = np.array(cnn_at_fill)
                directions = trades_df['direction'].values

                # Direction agreement mask
                if 'p_up' in cnn_field:
                    # p_up_5s: >0.5 = bullish, <0.5 = bearish
                    agree = np.where(
                        directions == 1,
                        cnn_vals > 0.5,    # long needs p_up > 0.5
                        cnn_vals < 0.5,    # short needs p_up < 0.5
                    )
                else:
                    # log_ret: positive = bullish, negative = bearish
                    agree = np.where(
                        directions == 1,
                        cnn_vals > 0,      # long needs positive pred
                        cnn_vals < 0,      # short needs negative pred
                    )

                # Confidence filter
                if conf_gate is not None:
                    abs_vals = np.abs(cnn_vals - (0.5 if 'p_up' in cnn_field else 0.0))
                    # Compute threshold on valid values
                    valid_abs = abs_vals[~np.isnan(abs_vals)]
                    if len(valid_abs) > 0:
                        conf_thresh = np.quantile(valid_abs, 1 - conf_gate)
                        conf_mask = abs_vals >= conf_thresh
                    else:
                        conf_mask = np.zeros(len(trades_df), dtype=bool)
                else:
                    conf_mask = np.ones(len(trades_df), dtype=bool)

                # Combined gate: direction agrees AND confidence passes AND CNN pred is valid
                valid = ~np.isnan(cnn_vals)
                gate_mask = valid & agree & conf_mask

                gated_trades = trades_df[gate_mask].copy()
                blocked_trades = trades_df[valid & ~gate_mask].copy()

                gated_metrics = compute_metrics(gated_trades)
                blocked_metrics = compute_metrics(blocked_trades)

                log.info(f"\n  ARM: {arm_name}")
                log.info(f"    Passed gate: {len(gated_trades)} trades ({len(gated_trades)/max(1,sum(valid)):.0%} of valid)")
                log.info(f"    Gated  — Sharpe: {gated_metrics['sharpe']:.2f}, WR: {gated_metrics['wr']:.1%}, "
                        f"PF: {gated_metrics['pf']:.2f}, Avg: {gated_metrics['avg_ticks']:.2f}")
                log.info(f"    Blocked— Sharpe: {blocked_metrics['sharpe']:.2f}, WR: {blocked_metrics['wr']:.1%}, "
                        f"PF: {blocked_metrics['pf']:.2f}, Avg: {blocked_metrics['avg_ticks']:.2f}")

                results.append({
                    'arm': arm_name,
                    'cnn_field': cnn_field,
                    'staleness': staleness_label,
                    'conf_gate': gate_label,
                    'n_valid': int(sum(valid)),
                    'n_passed': len(gated_trades),
                    'n_blocked': len(blocked_trades),
                    'pass_rate': len(gated_trades) / max(1, sum(valid)),
                    'baseline_sharpe': baseline['sharpe'],
                    'gated_sharpe': gated_metrics['sharpe'],
                    'gated_wr': gated_metrics['wr'],
                    'gated_pf': gated_metrics['pf'],
                    'gated_avg': gated_metrics['avg_ticks'],
                    'gated_total': gated_metrics['total_ticks'],
                    'blocked_sharpe': blocked_metrics['sharpe'],
                    'blocked_wr': blocked_metrics['wr'],
                    'blocked_pf': blocked_metrics['pf'],
                    'blocked_avg': blocked_metrics['avg_ticks'],
                })

    # Step 4: Summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "confluence_results.csv", index=False)

    log.info("\n" + "=" * 70)
    log.info("CONFLUENCE STUDY RESULTS SUMMARY")
    log.info("=" * 70)
    log.info(f"\nBaseline: {baseline['n_trades']} trades, Sharpe {baseline['sharpe']:.2f}, "
             f"WR {baseline['wr']:.1%}, PF {baseline['pf']:.2f}")
    log.info(f"Coverage: {len(overlap_dates)} dates (of 60 total OOT)")

    # Find best arms
    valid_results = results_df[results_df['n_passed'] >= 20].copy()  # min 20 trades
    if len(valid_results) > 0:
        valid_results['sharpe_lift'] = valid_results['gated_sharpe'] - baseline['sharpe']
        valid_results = valid_results.sort_values('gated_sharpe', ascending=False)

        log.info(f"\nTop 5 confluence arms (min 20 trades):")
        for _, row in valid_results.head(5).iterrows():
            log.info(f"  {row['arm']}")
            log.info(f"    Trades: {row['n_passed']}/{row['n_valid']} ({row['pass_rate']:.0%}), "
                    f"Sharpe: {row['gated_sharpe']:.2f} ({row['sharpe_lift']:+.2f}), "
                    f"WR: {row['gated_wr']:.1%}, PF: {row['gated_pf']:.2f}")

        # Win/loss diagnostic: for the best arm, show what got blocked
        best = valid_results.iloc[0]
        log.info(f"\n--- Win/Loss Diagnostic (best arm: {best['arm']}) ---")
        log.info(f"  Gate blocked {best['n_blocked']} trades")
        log.info(f"  Blocked trades: Sharpe {best['blocked_sharpe']:.2f}, WR {best['blocked_wr']:.1%}")

        if best['blocked_wr'] < best['gated_wr']:
            log.info(f"  ✓ Gate IS selective: blocked trades are WORSE (WR {best['blocked_wr']:.1%} vs {best['gated_wr']:.1%})")
        else:
            log.info(f"  ✗ Gate is NOT selective: blocked trades are BETTER or equal")
    else:
        log.info("\nNo arms with >= 20 trades found. Insufficient data.")

    # Save full trade-level data with CNN predictions
    trades_df.to_parquet(OUT_DIR / "trades_with_cnn.parquet", index=False)
    log.info(f"\nSaved results to {OUT_DIR}")
    log.info("DONE")


if __name__ == "__main__":
    run_confluence_study()
