#!/usr/bin/env python3
"""
Pressure Exit v2: Test pressure-based exits on buy_afternoon FIFO trades.

Uses the 46-date extended OOT validation dataset.
For each filled trade, monitors the CNN-Mamba prediction stream to decide
whether to exit early based on fading or reversing pressure.

Requires mid price bars to estimate P&L at early exit points.
Falls back to linear interpolation if mid price cache is unavailable.

Run on Neptune: python3 /home/nick/Lvl3Quant/scripts/pressure_exit_v2.py
"""

import os
import sys
import json
import glob
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── Paths (Neptune) ──────────────────────────────────────────────────────────
BASE = Path("/home/nick/Lvl3Quant")
FILLSIM_DIR = BASE / "output/extended_oot_validation/fillsim_results"
PRED_DIR = BASE / "output/extended_oot_validation/pred_npzs"
OUTPUT_DIR = BASE / "output/pressure_exit_v2"

# Multiple possible mid price cache locations
MID_CACHE_CANDIDATES = [
    BASE / "data/derived/mid_price_cache_hc439",
    BASE / "data/derived/mid_price_cache",
    BASE / "data/derived/mid_prices",
    BASE / "output/mid_price_bars",
]

# ── Cost constants (ES futures, AMP/Rithmic) ────────────────────────────────
COMMISSION_TICKS = 0.376       # $4.70 / $12.50
MARKET_EXIT_SPREAD = 0.5       # half-spread for market exit during RTH
TICK_SIZE = 0.25               # ES tick = 0.25 pts
TICK_VALUE = 12.50

# ── Pressure parameter sweep ────────────────────────────────────────────────
# NOTE: NPZ files only have a single 'predictions' array (10s horizon).
# No multi-horizon data available, so we use the single prediction as pressure.
# Sweep: 4 fade_thresh × 5 fade_n × 3 rev_thresh × 3 rev_n = 180 configs
CONFIGS = []
for fade_thresh in [-0.02, -0.05, -0.1, -0.2]:
    for fade_n in [3, 6, 10, 20, 40]:       # consecutive bars (100ms each) = 0.3s to 4s
        for rev_thresh in [-0.1, -0.3, -0.5]:
            for rev_n in [2, 4, 8]:
                CONFIGS.append({
                    'fade_thresh': fade_thresh,
                    'fade_n': fade_n,
                    'rev_thresh': rev_thresh,
                    'rev_n': rev_n,
                })


def find_mid_cache():
    """Find the mid price cache directory, checking multiple candidates."""
    for candidate in MID_CACHE_CANDIDATES:
        if candidate.exists() and any(candidate.glob("*.npz")):
            return candidate
    return None


def load_trades_and_preds(date_str, mid_cache_dir):
    """Load fill sim trades, predictions, and mid prices for a date.

    Returns (trades, preds_dict, mid_prices_array) or (None, None, None).
    """
    # Load trades
    trade_file = FILLSIM_DIR / f"buy_afternoon_{date_str}.json"
    if not trade_file.exists():
        return None, None, None

    try:
        with open(trade_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  WARNING: Cannot read {trade_file.name}: {e}")
        return None, None, None

    trades = data.get('trades', [])
    if not trades:
        return None, None, None

    # Load predictions
    pred_file = PRED_DIR / f"{date_str}_unfiltered.npz"
    if not pred_file.exists():
        return None, None, None

    try:
        npz = np.load(pred_file)
        # NPZ has single 'predictions' array (10s horizon CNN-Mamba v2)
        # and optionally 'ts_ns' for timestamps
        if 'predictions' in npz:
            pred_arr = npz['predictions']
        elif 'pred_10s' in npz:
            pred_arr = npz['pred_10s']
        else:
            pred_arr = npz[npz.files[0]]

        # Generate ts_ns if not present (100ms bars starting at 9:30 ET)
        if 'ts_ns' in npz:
            ts_ns = npz['ts_ns']
        else:
            # Approximate: 234000 bars at 100ms = 23400s = 6.5h (9:30-16:00 ET)
            # Use bar index as proxy since we match by fill_time_ns
            ts_ns = None

        preds = {
            'predictions': pred_arr,
            'ts_ns': ts_ns,
        }
    except Exception as e:
        print(f"  WARNING: Cannot read {pred_file.name}: {e}")
        return None, None, None

    # Load mid prices if cache available
    mid_prices = None
    if mid_cache_dir is not None:
        # Try several naming conventions
        for pattern in [f"{date_str}.npz", f"mid_{date_str}.npz",
                        f"glbx-mdp3-{date_str}.npz"]:
            mid_file = mid_cache_dir / pattern
            if mid_file.exists():
                try:
                    mid_npz = np.load(mid_file)
                    # Check common key names
                    for key in ['mid', 'mid_price', 'mid_prices', 'close',
                                'last', 'price']:
                        if key in mid_npz:
                            mid_prices = mid_npz[key]
                            break
                    # Fall back to first array in the file
                    if mid_prices is None and len(mid_npz.files) > 0:
                        mid_prices = mid_npz[mid_npz.files[0]]
                except Exception as e:
                    print(f"  WARNING: Cannot read mid cache {mid_file.name}: {e}")
                break

    return trades, preds, mid_prices


def ns_to_bar(target_ns, rth_open_ns):
    """Convert epoch nanosecond timestamp to bar index.
    RTH has 234000 bars at 100ms (23400s from 9:30-16:00 ET)."""
    offset_ns = target_ns - rth_open_ns
    bar = int(offset_ns / 100_000_000)  # 100ms per bar
    return max(0, min(bar, 233999))


def compute_rth_open_ns(date_str):
    """Compute RTH open (9:30 AM ET) in epoch nanoseconds for a given date.
    EDT (Mar-Nov): RTH open = 13:30 UTC. EST (Nov-Mar): RTH open = 14:30 UTC.
    All our dates (March-April 2026) are EDT."""
    from datetime import datetime as dt, timezone, timedelta
    d = dt.strptime(date_str, "%Y%m%d")
    # EDT: UTC-4, so 9:30 ET = 13:30 UTC
    midnight_utc = dt(d.year, d.month, d.day, tzinfo=timezone.utc)
    rth_open_utc = midnight_utc + timedelta(hours=13, minutes=30)
    return int(rth_open_utc.timestamp() * 1_000_000_000)


def simulate_pressure_exit(trade, preds, mid_prices, config, rth_open_ns):
    """Simulate pressure-based exit for a single trade.

    Returns dict with pnl_ticks, exit_reason, hold_bars, pressure_triggered.
    """
    fill_ns = trade['fill_time_ns']
    exit_ns = trade['exit_time_ns']
    entry_price = trade['entry_price']
    original_pnl = trade['pnl_ticks']

    pred_arr = preds['predictions']
    n_bars = len(pred_arr)

    fill_bar = ns_to_bar(fill_ns, rth_open_ns)
    exit_bar = ns_to_bar(exit_ns, rth_open_ns)
    max_bar = min(exit_bar + 1, n_bars)

    fade_thresh = config['fade_thresh']
    fade_n = config['fade_n']
    rev_thresh = config['rev_thresh']
    rev_n = config['rev_n']

    fade_count = 0
    rev_count = 0
    pressure_exit_bar = None
    pressure_exit_reason = None

    for bar in range(fill_bar + 1, max_bar):
        pressure = float(pred_arr[bar]) if bar < n_bars else 0.0

        # Fade detection: pressure dropped below threshold for N consecutive bars
        if pressure < fade_thresh:
            fade_count += 1
        else:
            fade_count = 0

        # Reversal detection: pressure deeply negative for M consecutive bars
        if pressure < rev_thresh:
            rev_count += 1
        else:
            rev_count = 0

        if fade_count >= fade_n:
            pressure_exit_bar = bar
            pressure_exit_reason = 'PressureFade'
            break

        if rev_count >= rev_n:
            pressure_exit_bar = bar
            pressure_exit_reason = 'PressureReversal'
            break

    # No pressure exit triggered -- use original exit
    if pressure_exit_bar is None:
        return {
            'pnl_ticks': original_pnl,
            'exit_reason': trade.get('exit_reason', 'Original'),
            'hold_bars': exit_bar - fill_bar,
            'pressure_triggered': False,
        }

    # ── Estimate P&L at pressure exit ────────────────────────────────────
    if mid_prices is not None and pressure_exit_bar < len(mid_prices):
        exit_price = float(mid_prices[pressure_exit_bar])
        # BUY trade: pnl = (exit - entry) / tick_size
        raw_ticks = (exit_price - entry_price) / TICK_SIZE
        pnl = raw_ticks - COMMISSION_TICKS - MARKET_EXIT_SPREAD
    else:
        # Fallback: linear interpolation between entry and realized exit
        hold_total = max(exit_bar - fill_bar, 1)
        hold_frac = (pressure_exit_bar - fill_bar) / hold_total
        # Conservative: assume linear price path from 0 to original_pnl
        raw_ticks = original_pnl * hold_frac
        pnl = raw_ticks - MARKET_EXIT_SPREAD  # extra spread for market exit

    return {
        'pnl_ticks': pnl,
        'exit_reason': pressure_exit_reason,
        'hold_bars': pressure_exit_bar - fill_bar,
        'pressure_triggered': True,
    }


def compute_metrics(daily_pnl_dict, dates_ordered, n_trades, n_wins,
                    gross_win, gross_loss):
    """Compute risk-adjusted metrics from daily P&L."""
    wr = n_wins / n_trades if n_trades > 0 else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    daily_vals = [daily_pnl_dict.get(d, 0.0) for d in dates_ordered]
    mean_d = np.mean(daily_vals)
    std_d = np.std(daily_vals)
    sharpe = mean_d / std_d * np.sqrt(252) if std_d > 0 else 0

    neg_daily = [v for v in daily_vals if v < 0]
    downside_std = np.std(neg_daily) if len(neg_daily) > 1 else 0
    sortino = mean_d / downside_std * np.sqrt(252) if downside_std > 0 else 0

    return wr, pf, sharpe, sortino, daily_vals


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Discover dates ───────────────────────────────────────────────────
    date_files = sorted(glob.glob(str(FILLSIM_DIR / "buy_afternoon_*.json")))
    if not date_files:
        print(f"ERROR: No trade files found in {FILLSIM_DIR}")
        sys.exit(1)
    dates = [Path(f).stem.replace("buy_afternoon_", "") for f in date_files]
    print(f"Found {len(dates)} date files")

    # ── Find mid price cache ─────────────────────────────────────────────
    mid_cache = find_mid_cache()
    if mid_cache is not None:
        mid_files = list(mid_cache.glob("*.npz"))
        print(f"Mid price cache: {mid_cache} ({len(mid_files)} files)")
    else:
        print("WARNING: No mid price cache found. Using linear interpolation "
              "for pressure-exit P&L estimates.")

    # ── Load all data ────────────────────────────────────────────────────
    all_data = {}
    total_trades = 0
    mid_hit = 0
    mid_miss = 0
    for date in dates:
        trades, preds, mid_prices = load_trades_and_preds(date, mid_cache)
        if trades is not None:
            all_data[date] = (trades, preds, mid_prices)
            total_trades += len(trades)
            has_mid = mid_prices is not None
            if has_mid:
                mid_hit += 1
            else:
                mid_miss += 1
            print(f"  {date}: {len(trades):3d} trades, "
                  f"mid_prices={'yes' if has_mid else 'NO'}")

    if not all_data:
        print("ERROR: No valid data loaded.")
        sys.exit(1)

    dates_ordered = sorted(all_data.keys())
    print(f"\nLoaded {total_trades} trades across {len(all_data)} dates "
          f"(mid: {mid_hit} yes / {mid_miss} no)")
    print(f"Sweeping {len(CONFIGS)} configs...\n")

    # ── Baseline metrics ─────────────────────────────────────────────────
    bl_daily = defaultdict(float)
    bl_trades = 0
    bl_wins = 0
    bl_total_pnl = 0.0
    bl_gross_win = 0.0
    bl_gross_loss = 0.0
    for date, (trades, _, _) in all_data.items():
        for t in trades:
            pnl = t['pnl_ticks']
            bl_daily[date] += pnl
            bl_total_pnl += pnl
            bl_trades += 1
            if pnl > 0:
                bl_wins += 1
                bl_gross_win += pnl
            else:
                bl_gross_loss += abs(pnl)

    bl_wr, bl_pf, bl_sharpe, bl_sortino, _ = compute_metrics(
        bl_daily, dates_ordered, bl_trades, bl_wins,
        bl_gross_win, bl_gross_loss)

    print(f"BASELINE: PF={bl_pf:.3f}  WR={bl_wr:.1%}  "
          f"Net={bl_total_pnl:.1f}t  Sharpe={bl_sharpe:.1f}  "
          f"Sortino={bl_sortino:.1f}  Trades={bl_trades}\n")

    # ── Sweep configs ────────────────────────────────────────────────────
    results = []
    for i, cfg in enumerate(CONFIGS):
        daily_pnl = defaultdict(float)
        n_trades = 0
        n_wins = 0
        n_pressure_exits = 0
        total_pnl = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        hold_bars_list = []
        exit_reasons = defaultdict(int)

        for date, (trades, preds, mid_prices) in all_data.items():
            rth_open_ns = compute_rth_open_ns(date)
            for t in trades:
                result = simulate_pressure_exit(
                    t, preds, mid_prices, cfg, rth_open_ns)
                pnl = result['pnl_ticks']
                daily_pnl[date] += pnl
                total_pnl += pnl
                n_trades += 1
                if pnl > 0:
                    n_wins += 1
                    gross_win += pnl
                else:
                    gross_loss += abs(pnl)
                if result['pressure_triggered']:
                    n_pressure_exits += 1
                hold_bars_list.append(result['hold_bars'])
                exit_reasons[result['exit_reason']] += 1

        wr, pf, sharpe, sortino, _ = compute_metrics(
            daily_pnl, dates_ordered, n_trades, n_wins,
            gross_win, gross_loss)

        avg_hold_sec = np.mean(hold_bars_list) * 0.1 if hold_bars_list else 0

        cfg_name = (f"fade{cfg['fade_thresh']}_n{cfg['fade_n']}_"
                    f"rev{cfg['rev_thresh']}_m{cfg['rev_n']}")

        results.append({
            'name': cfg_name,
            'config': cfg,
            'pf': round(pf, 4),
            'wr': round(wr, 4),
            'net_ticks': round(total_pnl, 1),
            'sharpe': round(sharpe, 2),
            'sortino': round(sortino, 2),
            'n_trades': n_trades,
            'n_pressure_exits': n_pressure_exits,
            'pct_pressure_exits': (
                round(n_pressure_exits / n_trades * 100, 1)
                if n_trades else 0),
            'avg_hold_sec': round(avg_hold_sec, 1),
            'exit_reasons': dict(exit_reasons),
            'daily_pnl': {d: round(v, 1) for d, v in daily_pnl.items()},
            'improvement_vs_baseline': round(total_pnl - bl_total_pnl, 1),
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(CONFIGS)}] configs done  "
                  f"({elapsed:.0f}s elapsed)")

    # ── Sort by PF descending ────────────────────────────────────────────
    results.sort(key=lambda x: x['pf'], reverse=True)

    # ── Print top 10 ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TOP 10 CONFIGS (vs baseline PF={bl_pf:.3f}, "
          f"Sharpe={bl_sharpe:.1f}, Sortino={bl_sortino:.1f})")
    print(f"{'='*70}")
    for r in results[:10]:
        print(f"  {r['name']}")
        print(f"    PF={r['pf']:.3f}  WR={r['wr']:.1%}  "
              f"Net={r['net_ticks']:.0f}t  Sharpe={r['sharpe']:.1f}  "
              f"Sortino={r['sortino']:.1f}")
        print(f"    PressureExits={r['n_pressure_exits']}/{r['n_trades']} "
              f"({r['pct_pressure_exits']:.0f}%)  "
              f"AvgHold={r['avg_hold_sec']:.0f}s")
        print(f"    vs baseline: {r['improvement_vs_baseline']:+.0f} ticks")
        print()

    # ── Print worst 5 (sanity check) ─────────────────────────────────────
    print(f"\nBOTTOM 5 CONFIGS (sanity check)")
    print(f"{'-'*70}")
    for r in results[-5:]:
        print(f"  {r['name']}")
        print(f"    PF={r['pf']:.3f}  WR={r['wr']:.1%}  "
              f"Net={r['net_ticks']:.0f}t  "
              f"PressureExits={r['pct_pressure_exits']:.0f}%")

    # ── Regime analysis on top config ────────────────────────────────────
    if results:
        best = results[0]
        daily = best['daily_pnl']
        green_days = [v for v in daily.values() if v > 0]
        red_days = [v for v in daily.values() if v < 0]
        flat_days = [v for v in daily.values() if v == 0]
        print(f"\nBEST CONFIG DAILY BREAKDOWN:")
        print(f"  Green days: {len(green_days)} "
              f"(avg {np.mean(green_days):.1f}t)" if green_days else
              "  Green days: 0")
        print(f"  Red days:   {len(red_days)} "
              f"(avg {np.mean(red_days):.1f}t)" if red_days else
              "  Red days:   0")
        print(f"  Flat days:  {len(flat_days)}")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        'run_timestamp': datetime.utcnow().isoformat() + 'Z',
        'baseline': {
            'pf': round(bl_pf, 4),
            'wr': round(bl_wr, 4),
            'net_ticks': round(bl_total_pnl, 1),
            'sharpe': round(bl_sharpe, 2),
            'sortino': round(bl_sortino, 2),
            'n_trades': bl_trades,
        },
        'sweep_info': {
            'configs_tested': len(CONFIGS),
            'total_trades_per_config': total_trades,
            'dates': len(all_data),
            'has_mid_prices': mid_cache is not None,
            'mid_price_dates': mid_hit,
            'interpolation_dates': mid_miss,
        },
        'top_configs': results[:20],
        'all_configs': results,
    }

    outfile = OUTPUT_DIR / "pressure_exit_v2_results.json"
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Saved to {outfile}")


if __name__ == '__main__':
    main()
