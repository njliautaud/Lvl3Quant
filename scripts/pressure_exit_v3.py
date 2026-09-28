#!/usr/bin/env python3
"""
Pressure Exit v3: Full bar-by-bar simulation with hold extension.

Builds on v2 (pressure-based early exits) with two major additions:
  1. Mid price support: loads actual mid prices for bar-by-bar P&L
     (falls back to interpolation if unavailable)
  2. Hold extension: when pressure is strong at TP hit, override TP
     and trail a stop to ride continuation moves

Run on Neptune: python3 /home/nick/Lvl3Quant/scripts/pressure_exit_v3.py
"""

import os
import sys
import json
import glob
import time
import numpy as np
from pathlib import Path
from itertools import product
from collections import defaultdict
from datetime import datetime, timezone, timedelta

# ── Paths (Neptune) ──────────────────────────────────────────────────────────
BASE = Path("/home/nick/Lvl3Quant")
FILLSIM_DIR = BASE / "output/extended_oot_validation/fillsim_results"
PRED_DIR = BASE / "output/extended_oot_validation/pred_npzs"
MID_PRICE_DIR = BASE / "data/derived/mid_price_bars"
OUTPUT_DIR = BASE / "output/pressure_exit_v3"

# Legacy mid price cache locations (v2 compat)
MID_CACHE_CANDIDATES = [
    MID_PRICE_DIR,
    BASE / "data/derived/mid_price_cache_hc439",
    BASE / "data/derived/mid_price_cache",
    BASE / "data/derived/mid_prices",
    BASE / "output/mid_price_bars",
]

# ── Cost constants (ES futures, AMP/Rithmic) ────────────────────────────────
COMMISSION_TICKS = 0.376       # $4.70 / $12.50
MARKET_EXIT_SPREAD = 0.5       # half-spread for market exit during RTH
TICK = 0.25                    # ES tick = 0.25 pts
TICK_VALUE = 12.50

# ── Trade parameters ────────────────────────────────────────────────────────
TP_TICKS = 8
SL_TICKS = 16
MAX_HOLD_BARS = 18000          # 30 min = 1800s = 18000 bars at 100ms
N_BARS_PER_DAY = 234000

# ── Parameter sweep ─────────────────────────────────────────────────────────
# Pressure early exit params (v2 best range)
FADE_THRESH_VALS = [-0.05, -0.1, -0.2]
FADE_N_VALS = [20, 40]
REV_THRESH_VALS = [-0.3, -0.5]
REV_N_VALS = [4, 8]

# Hold extension params (NEW in v3)
EXTEND_THRESH_VALS = [0.2, 0.5, 1.0, 999.0]  # 999 = never extend (pressure-only)
EXTEND_FADE_N_VALS = [10, 20, 40]
MAX_EXTEND_SEC_VALS = [15, 30, 60]
TRAIL_LOCK_TICKS_VALS = [4, 6]

# Config budget: 24 pressure × (1 no-ext + 36 ext) = 888 < 1000
# Achieved by using 2 extend_fade_n values instead of 3


def build_configs():
    """Build config list, keeping total under 1000.

    Strategy: full cross of pressure params (24 combos) × extension params.
    For extend_threshold=999 (no extension), extension params don't matter,
    so only emit one config per pressure combo.
    """
    configs = []
    pressure_combos = list(product(
        FADE_THRESH_VALS, FADE_N_VALS, REV_THRESH_VALS, REV_N_VALS))

    for ft, fn, rt, rn in pressure_combos:
        base = {
            'fade_thresh': ft, 'fade_n': fn,
            'rev_thresh': rt, 'rev_n': rn,
        }

        # No-extension config (pressure exit only, v2-style)
        configs.append({
            **base,
            'extend_threshold': 999.0,
            'extend_fade_n': 20,       # irrelevant
            'max_extend_sec': 30,       # irrelevant
            'trail_lock_ticks': 4,      # irrelevant
        })

        # Extension configs (use subset of extend_fade_n to stay under 1000)
        for et, efn, mes, tlt in product(
                [0.2, 0.5, 1.0], [10, 40],
                MAX_EXTEND_SEC_VALS, TRAIL_LOCK_TICKS_VALS):
            configs.append({
                **base,
                'extend_threshold': et,
                'extend_fade_n': efn,
                'max_extend_sec': mes,
                'trail_lock_ticks': tlt,
            })

    print(f"Built {len(configs)} configs "
          f"({len(pressure_combos)} pressure × extension combos)")
    return configs


def find_mid_cache():
    """Find the mid price cache directory."""
    for candidate in MID_CACHE_CANDIDATES:
        if candidate.exists() and any(candidate.glob("*.npz")):
            return candidate
    return None


def compute_rth_open_ns(date_str):
    """9:30 AM ET = 13:30 UTC for EDT dates."""
    d = datetime.strptime(date_str, "%Y%m%d")
    midnight_utc = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    rth_open_utc = midnight_utc + timedelta(hours=13, minutes=30)
    return int(rth_open_utc.timestamp() * 1_000_000_000)


def ns_to_bar(target_ns, rth_open_ns):
    """Convert epoch ns to bar index (100ms bars, 234000 per RTH day)."""
    offset_ns = target_ns - rth_open_ns
    bar = int(offset_ns / 100_000_000)
    return max(0, min(bar, N_BARS_PER_DAY - 1))


def load_trades_and_preds(date_str, mid_cache_dir):
    """Load fill sim trades, predictions, and mid prices for a date."""
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
        if 'predictions' in npz:
            pred_arr = npz['predictions'].astype(np.float32)
        elif 'pred_10s' in npz:
            pred_arr = npz['pred_10s'].astype(np.float32)
        else:
            pred_arr = npz[npz.files[0]].astype(np.float32)
    except Exception as e:
        print(f"  WARNING: Cannot read {pred_file.name}: {e}")
        return None, None, None

    # Load mid prices — try primary location then cache candidates
    mid_prices = None
    if mid_cache_dir is not None:
        for pattern in [f"{date_str}.npz", f"mid_{date_str}.npz",
                        f"glbx-mdp3-{date_str}.npz"]:
            mid_file = mid_cache_dir / pattern
            if mid_file.exists():
                try:
                    mid_npz = np.load(mid_file)
                    for key in ['mid_prices', 'mid', 'mid_price', 'close',
                                'last', 'price']:
                        if key in mid_npz:
                            mid_prices = mid_npz[key].astype(np.float32)
                            break
                    if mid_prices is None and len(mid_npz.files) > 0:
                        mid_prices = mid_npz[mid_npz.files[0]].astype(np.float32)
                except Exception as e:
                    print(f"  WARNING: Cannot read mid cache {mid_file.name}: {e}")
                break

    return trades, pred_arr, mid_prices


def exit_cost_ticks(exit_reason):
    """Total cost in ticks depending on exit type.

    TP exit = passive limit → commission only (0.376t).
    All other exits = market order → commission + spread (0.876t).
    """
    if exit_reason == 'TakeProfit':
        return COMMISSION_TICKS
    return COMMISSION_TICKS + MARKET_EXIT_SPREAD


def simulate_trade(trade, pred_arr, mid_prices, config, rth_open_ns):
    """Full bar-by-bar simulation of a single trade with pressure management.

    Returns dict: pnl_ticks, exit_reason, hold_bars, used_mid_prices, extended.
    """
    entry_price = trade['entry_price']
    fill_ns = trade['fill_time_ns']
    exit_ns = trade['exit_time_ns']

    fill_bar = ns_to_bar(fill_ns, rth_open_ns)
    original_exit_bar = ns_to_bar(exit_ns, rth_open_ns)

    tp_price = entry_price + TP_TICKS * TICK
    sl_price = entry_price - SL_TICKS * TICK

    # Config params
    fade_thresh = config['fade_thresh']
    fade_n = config['fade_n']
    rev_thresh = config['rev_thresh']
    rev_n = config['rev_n']
    extend_threshold = config['extend_threshold']
    extend_fade_n = config['extend_fade_n']
    max_extend_bars = int(config['max_extend_sec'] * 10)  # sec → 100ms bars
    trail_lock_ticks = config['trail_lock_ticks']

    # State
    fade_count = 0
    rev_count = 0
    tp_hit = False
    tp_hit_bar = None
    trail_stop_price = None
    best_price_seen = entry_price
    ext_fade_count = 0  # bars where pressure < extend_threshold during extension

    # Decide whether we can use mid prices or must interpolate
    has_mid = (mid_prices is not None and len(mid_prices) > fill_bar)

    # For interpolation fallback: need entry→exit price path
    original_pnl_ticks = trade['pnl_ticks']

    end_bar = min(fill_bar + MAX_HOLD_BARS, N_BARS_PER_DAY)

    for bar in range(fill_bar + 1, end_bar):
        # ── Get current price ────────────────────────────────────────────
        if has_mid and bar < len(mid_prices) and mid_prices[bar] > 0:
            current_price = float(mid_prices[bar])
        else:
            # Interpolation fallback: linear from entry to original exit
            hold_total = max(original_exit_bar - fill_bar, 1)
            frac = min((bar - fill_bar) / hold_total, 1.0)
            current_price = entry_price + original_pnl_ticks * TICK * frac

        unrealized_ticks = (current_price - entry_price) / TICK
        pressure = float(pred_arr[bar]) if bar < len(pred_arr) else 0.0

        # Track best price seen (for trailing stop ratchet)
        if current_price > best_price_seen:
            best_price_seen = current_price

        # ── 1. Hard SL — always active ───────────────────────────────────
        if unrealized_ticks <= -SL_TICKS:
            raw_ticks = -SL_TICKS
            cost = exit_cost_ticks('HardSL')
            return _result('HardSL', bar, fill_bar, raw_ticks - cost,
                           has_mid, tp_hit)

        # ── 2. Extension trailing stop ───────────────────────────────────
        if tp_hit:
            # Ratchet trail stop up
            new_trail = best_price_seen - trail_lock_ticks * TICK
            if trail_stop_price is None or new_trail > trail_stop_price:
                trail_stop_price = new_trail

            if current_price <= trail_stop_price:
                raw_ticks = (trail_stop_price - entry_price) / TICK
                cost = exit_cost_ticks('ExtensionTrail')
                return _result('ExtensionTrail', bar, fill_bar,
                               raw_ticks - cost, has_mid, True)

            # Max extension time
            if bar - tp_hit_bar >= max_extend_bars:
                raw_ticks = unrealized_ticks
                cost = exit_cost_ticks('ExtensionTimeout')
                return _result('ExtensionTimeout', bar, fill_bar,
                               raw_ticks - cost, has_mid, True)

            # Extension fade: pressure dropped below extend_threshold
            if pressure < extend_threshold:
                ext_fade_count += 1
            else:
                ext_fade_count = 0

            if ext_fade_count >= extend_fade_n:
                raw_ticks = unrealized_ticks
                cost = exit_cost_ticks('ExtensionFade')
                return _result('ExtensionFade', bar, fill_bar,
                               raw_ticks - cost, has_mid, True)

            # During extension, skip pressure early-exit and TP checks
            continue

        # ── 3. TP check (may be overridden by extension) ─────────────────
        if unrealized_ticks >= TP_TICKS:
            if pressure > extend_threshold:
                # Override TP — start extension
                tp_hit = True
                tp_hit_bar = bar
                trail_stop_price = entry_price + trail_lock_ticks * TICK
                ext_fade_count = 0
                continue
            else:
                # Normal TP exit (passive limit)
                raw_ticks = TP_TICKS
                cost = exit_cost_ticks('TakeProfit')
                return _result('TakeProfit', bar, fill_bar,
                               raw_ticks - cost, has_mid, False)

        # ── 4. Pressure early exit (before TP) ──────────────────────────
        if pressure < fade_thresh:
            fade_count += 1
        else:
            fade_count = 0

        if pressure < rev_thresh:
            rev_count += 1
        else:
            rev_count = 0

        if fade_count >= fade_n:
            raw_ticks = unrealized_ticks
            cost = exit_cost_ticks('PressureFade')
            return _result('PressureFade', bar, fill_bar,
                           raw_ticks - cost, has_mid, False)

        if rev_count >= rev_n:
            raw_ticks = unrealized_ticks
            cost = exit_cost_ticks('PressureReversal')
            return _result('PressureReversal', bar, fill_bar,
                           raw_ticks - cost, has_mid, False)

    # ── Max hold reached ─────────────────────────────────────────────────
    final_bar = min(fill_bar + MAX_HOLD_BARS, N_BARS_PER_DAY) - 1
    if has_mid and final_bar < len(mid_prices) and mid_prices[final_bar] > 0:
        raw_ticks = (float(mid_prices[final_bar]) - entry_price) / TICK
    else:
        raw_ticks = original_pnl_ticks  # best guess
    cost = exit_cost_ticks('MaxHold')
    return _result('MaxHold', final_bar, fill_bar, raw_ticks - cost,
                   has_mid, tp_hit)


def _result(reason, bar, fill_bar, net_pnl_ticks, used_mid, extended):
    """Build a trade result dict."""
    return {
        'pnl_ticks': net_pnl_ticks,
        'exit_reason': reason,
        'hold_bars': bar - fill_bar,
        'used_mid_prices': used_mid,
        'extended': extended,
    }


def compute_metrics(daily_pnl_dict, dates_ordered):
    """Compute risk-adjusted metrics from daily P&L."""
    daily_vals = np.array([daily_pnl_dict.get(d, 0.0) for d in dates_ordered])
    n_days = len(daily_vals)

    mean_d = float(np.mean(daily_vals))
    std_d = float(np.std(daily_vals))
    sharpe = mean_d / std_d * np.sqrt(252) if std_d > 0 else 0.0

    neg = daily_vals[daily_vals < 0]
    ds = float(np.std(neg)) if len(neg) > 1 else 0.0
    sortino = mean_d / ds * np.sqrt(252) if ds > 0 else 0.0

    return sharpe, sortino


def run_config(cfg, all_data, dates_ordered):
    """Run one config across all dates. Returns summary dict."""
    daily_pnl = defaultdict(float)
    n_trades = 0
    n_wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    hold_bars_total = 0
    exit_reasons = defaultdict(int)
    n_extended = 0
    n_mid = 0

    for date, (trades, pred_arr, mid_prices, rth_open_ns) in all_data.items():
        for t in trades:
            res = simulate_trade(t, pred_arr, mid_prices, cfg, rth_open_ns)
            pnl = res['pnl_ticks']
            daily_pnl[date] += pnl
            n_trades += 1
            if pnl > 0:
                n_wins += 1
                gross_win += pnl
            else:
                gross_loss += abs(pnl)
            hold_bars_total += res['hold_bars']
            exit_reasons[res['exit_reason']] += 1
            if res['extended']:
                n_extended += 1
            if res['used_mid_prices']:
                n_mid += 1

    total_pnl = sum(daily_pnl.values())
    wr = n_wins / n_trades if n_trades > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    sharpe, sortino = compute_metrics(daily_pnl, dates_ordered)
    avg_hold_sec = (hold_bars_total / n_trades * 0.1) if n_trades > 0 else 0.0

    has_extension = cfg['extend_threshold'] < 999.0

    return {
        'config': cfg,
        'pf': round(pf, 4),
        'wr': round(wr, 4),
        'net_ticks': round(total_pnl, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'n_trades': n_trades,
        'n_wins': n_wins,
        'avg_hold_sec': round(avg_hold_sec, 1),
        'exit_reasons': dict(exit_reasons),
        'n_extended': n_extended,
        'pct_extended': round(n_extended / n_trades * 100, 1) if n_trades else 0,
        'has_extension': has_extension,
        'daily_pnl': {d: round(v, 2) for d, v in daily_pnl.items()},
    }


def compute_baseline(all_data, dates_ordered):
    """Compute baseline metrics (original fixed TP/SL exits)."""
    daily_pnl = defaultdict(float)
    n_trades = 0
    n_wins = 0
    gross_win = 0.0
    gross_loss = 0.0

    for date, (trades, _, _, _) in all_data.items():
        for t in trades:
            pnl = t['pnl_ticks']
            daily_pnl[date] += pnl
            n_trades += 1
            if pnl > 0:
                n_wins += 1
                gross_win += pnl
            else:
                gross_loss += abs(pnl)

    total_pnl = sum(daily_pnl.values())
    wr = n_wins / n_trades if n_trades > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    sharpe, sortino = compute_metrics(daily_pnl, dates_ordered)

    return {
        'pf': round(pf, 4),
        'wr': round(wr, 4),
        'net_ticks': round(total_pnl, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'n_trades': n_trades,
        'daily_pnl': {d: round(v, 2) for d, v in daily_pnl.items()},
    }


def regime_breakdown(daily_pnl):
    """Split daily P&L into March (early) vs April+ (late) regime."""
    march = {}
    april_plus = {}
    for d, v in daily_pnl.items():
        month = int(d[4:6])
        if month <= 3:
            march[d] = v
        else:
            april_plus[d] = v

    def _stats(vals):
        if not vals:
            return {'days': 0, 'net': 0, 'avg': 0, 'green': 0, 'red': 0}
        v = list(vals.values())
        return {
            'days': len(v),
            'net': round(sum(v), 1),
            'avg': round(float(np.mean(v)), 1),
            'green': sum(1 for x in v if x > 0),
            'red': sum(1 for x in v if x < 0),
        }

    return {'march': _stats(march), 'april_plus': _stats(april_plus)}


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

    # ── Find mid price source ────────────────────────────────────────────
    mid_cache = find_mid_cache()
    if mid_cache is not None:
        mid_files = list(mid_cache.glob("*.npz"))
        print(f"Mid price source: {mid_cache} ({len(mid_files)} files)")
    else:
        print("WARNING: No mid price source found. "
              "Using interpolation fallback for all dates.")

    # ── Load all data ────────────────────────────────────────────────────
    all_data = {}  # date → (trades, pred_arr, mid_prices, rth_open_ns)
    total_trades = 0
    mid_hit = 0
    mid_miss = 0

    for date in dates:
        trades, pred_arr, mid_prices = load_trades_and_preds(date, mid_cache)
        if trades is None:
            continue

        rth_open_ns = compute_rth_open_ns(date)
        all_data[date] = (trades, pred_arr, mid_prices, rth_open_ns)
        total_trades += len(trades)

        has_mid = mid_prices is not None
        if has_mid:
            mid_hit += 1
        else:
            mid_miss += 1
        print(f"  {date}: {len(trades):3d} trades, "
              f"mid={'yes' if has_mid else 'NO'}")

    if not all_data:
        print("ERROR: No valid data loaded.")
        sys.exit(1)

    dates_ordered = sorted(all_data.keys())
    print(f"\nLoaded {total_trades} trades across {len(all_data)} dates "
          f"(mid: {mid_hit} yes / {mid_miss} no)")

    # ── Build configs ────────────────────────────────────────────────────
    configs = build_configs()

    # ── Baseline ─────────────────────────────────────────────────────────
    baseline = compute_baseline(all_data, dates_ordered)
    print(f"\nBASELINE: PF={baseline['pf']:.3f}  WR={baseline['wr']:.1%}  "
          f"Net={baseline['net_ticks']:.0f}t  Sharpe={baseline['sharpe']:.1f}  "
          f"Sortino={baseline['sortino']:.1f}  Trades={baseline['n_trades']}")
    print(f"\nSweeping {len(configs)} configs...\n")

    # ── Sweep ────────────────────────────────────────────────────────────
    results = []
    for i, cfg in enumerate(configs):
        r = run_config(cfg, all_data, dates_ordered)
        r['improvement_ticks'] = round(r['net_ticks'] - baseline['net_ticks'], 1)
        results.append(r)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(configs)}] configs done  "
                  f"({elapsed:.0f}s elapsed)")

    # ── Sort by PF ───────────────────────────────────────────────────────
    results.sort(key=lambda x: x['pf'], reverse=True)

    # ── Separate pressure-only vs extension results ──────────────────────
    pressure_only = [r for r in results if not r['has_extension']]
    with_extension = [r for r in results if r['has_extension']]

    # ── Print comparison ─────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("COMPARISON: Baseline vs Best Pressure-Only vs Best Pressure+Extension")
    print(f"{'='*75}")

    print(f"\n  BASELINE (fixed TP/SL):")
    print(f"    PF={baseline['pf']:.3f}  WR={baseline['wr']:.1%}  "
          f"Net={baseline['net_ticks']:.0f}t  "
          f"Sharpe={baseline['sharpe']:.1f}  Sortino={baseline['sortino']:.1f}")

    if pressure_only:
        best_po = pressure_only[0]
        print(f"\n  BEST PRESSURE-ONLY (no extension):")
        _print_result(best_po, baseline)

    if with_extension:
        best_ext = with_extension[0]
        print(f"\n  BEST PRESSURE+EXTENSION:")
        _print_result(best_ext, baseline)

    # ── Top 10 overall ───────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"TOP 10 CONFIGS (vs baseline PF={baseline['pf']:.3f})")
    print(f"{'='*75}")
    for rank, r in enumerate(results[:10], 1):
        tag = "[EXT]" if r['has_extension'] else "[P/O]"
        print(f"\n  #{rank} {tag}")
        _print_result(r, baseline)

    # ── Regime analysis on top overall ───────────────────────────────────
    if results:
        best = results[0]
        regime = regime_breakdown(best['daily_pnl'])
        print(f"\n{'='*75}")
        print(f"REGIME BREAKDOWN (top config)")
        print(f"{'='*75}")
        for name, stats in regime.items():
            if stats['days'] > 0:
                print(f"  {name}: {stats['days']} days, "
                      f"net={stats['net']:.0f}t, avg={stats['avg']:.1f}t/day, "
                      f"{stats['green']}G/{stats['red']}R")

        # Daily P&L list
        daily = best['daily_pnl']
        green = [v for v in daily.values() if v > 0]
        red = [v for v in daily.values() if v < 0]
        print(f"\n  Green days: {len(green)}"
              f" (avg {np.mean(green):.1f}t)" if green else
              "\n  Green days: 0")
        print(f"  Red days:   {len(red)}"
              f" (avg {np.mean(red):.1f}t)" if red else
              "  Red days:   0")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        'run_timestamp': datetime.now(timezone.utc).isoformat(),
        'version': 'v3',
        'baseline': baseline,
        'sweep_info': {
            'configs_tested': len(configs),
            'pressure_only_configs': len(pressure_only),
            'extension_configs': len(with_extension),
            'total_trades_per_config': total_trades,
            'dates': len(all_data),
            'dates_with_mid_prices': mid_hit,
            'dates_interpolated': mid_miss,
        },
        'best_pressure_only': pressure_only[0] if pressure_only else None,
        'best_with_extension': with_extension[0] if with_extension else None,
        'top_20_configs': results[:20],
        'all_configs': results,
    }

    outfile = OUTPUT_DIR / "results.json"
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Saved to {outfile}")
    print(f"Total configs: {len(configs)} "
          f"({len(pressure_only)} pressure-only + "
          f"{len(with_extension)} with extension)")


def _print_result(r, baseline):
    """Pretty-print a single result."""
    cfg = r['config']
    print(f"    PF={r['pf']:.3f}  WR={r['wr']:.1%}  "
          f"Net={r['net_ticks']:.0f}t  Sharpe={r['sharpe']:.1f}  "
          f"Sortino={r['sortino']:.1f}")
    print(f"    AvgHold={r['avg_hold_sec']:.0f}s  "
          f"vs baseline: {r['improvement_ticks']:+.0f}t")

    # Exit reason breakdown
    reasons = r['exit_reasons']
    reason_str = ", ".join(f"{k}={v}" for k, v in
                           sorted(reasons.items(), key=lambda x: -x[1]))
    print(f"    Exits: {reason_str}")

    if r['has_extension']:
        print(f"    Extended: {r['n_extended']}/{r['n_trades']} "
              f"({r['pct_extended']:.1f}%)")
        print(f"    ext_thresh={cfg['extend_threshold']}  "
              f"ext_fade_n={cfg['extend_fade_n']}  "
              f"max_ext={cfg['max_extend_sec']}s  "
              f"trail_lock={cfg['trail_lock_ticks']}t")

    print(f"    fade={cfg['fade_thresh']}/n{cfg['fade_n']}  "
          f"rev={cfg['rev_thresh']}/n{cfg['rev_n']}")


if __name__ == '__main__':
    main()
