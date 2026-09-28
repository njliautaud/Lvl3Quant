#!/usr/bin/env python3
"""
Extended OOT Validation — ALL 46 dates, Regime-Agnostic
========================================================
Generates bar-level prediction NPZs for all CNN-Mamba v2 OOT dates,
then runs 4 fill sim configs across all of them for proper regime-agnostic
validation per HC #428.

Configs:
  both_baseline:  TP8/SL16, hold 30m, signal 0.3, latency 10ms
  buy_allday:     buy-only TP8/SL16, hold 30m, signal 0.3
  buy_afternoon:  buy-only TP8/SL16, hold 30m, signal 0.3, 14:00-16:00 ET
  both_afternoon: TP8/SL16, hold 30m, signal 0.3, 14:00-16:00 ET
"""

import os
import sys
import json
import subprocess
import datetime
import time
import numpy as np
from pathlib import Path

# ============ PATHS ============
BASE = Path("/home/nick/Lvl3Quant")
CNN_PRED_DIR = BASE / "output" / "cnn_mamba_v2_bulk_oot"
EVENTS_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3"
MBO_DIR = BASE / "data" / "raw" / "mbo"
FILL_SIM = BASE / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
OUTPUT_DIR = BASE / "output" / "extended_oot_validation"
PRED_NPZ_DIR = OUTPUT_DIR / "pred_npzs"
BUY_NPZ_DIR = OUTPUT_DIR / "pred_npzs_buy_only"
RESULTS_DIR = OUTPUT_DIR / "fillsim_results"

for d in [OUTPUT_DIR, PRED_NPZ_DIR, BUY_NPZ_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N_RTH_BARS = 234000  # 6.5 hrs * 3600 s/hr * 10 bars/s

# ============ FILL SIM CONFIGS ============
BASE_CFG = {
    "signal_threshold": 0.3,
    "take_profit_ticks": 8,
    "stop_loss_ticks": 16,
    "hold_ms": 1800000,   # 30 minutes
    "latency_ms": 10,
}

CONFIGS = {
    "both_baseline": {
        "pred_dir": "both",
        "extra_flags": [],
    },
    "buy_allday": {
        "pred_dir": "buy",
        "extra_flags": [],
    },
    "buy_afternoon": {
        "pred_dir": "buy",
        "extra_flags": ["--time-window-start", "14:00", "--time-window-end", "16:00"],
    },
    "both_afternoon": {
        "pred_dir": "both",
        "extra_flags": ["--time-window-start", "14:00", "--time-window-end", "16:00"],
    },
}


def get_rth_bounds(date_str):
    """RTH = 9:30 AM - 4:00 PM ET = 13:30-20:00 UTC."""
    y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    rth_start = datetime.datetime(y, m, d, 13, 30, 0)
    rth_end = datetime.datetime(y, m, d, 20, 0, 0)
    return int(rth_start.timestamp() * 1e9), int(rth_end.timestamp() * 1e9)


def discover_dates():
    """Find all dates that have CNN predictions + MBO events + MBO raw."""
    cnn_dates = set()
    for f in CNN_PRED_DIR.glob("*_predictions.npz"):
        date = f.name.split("_")[0]
        if len(date) == 8 and date.isdigit():
            cnn_dates.add(date)

    valid = []
    for date in sorted(cnn_dates):
        events_path = EVENTS_DIR / f"{date}_mbo_events.npz"
        mbo_path = MBO_DIR / f"glbx-mdp3-{date}.mbo.dbn.zst"
        if events_path.exists() and mbo_path.exists():
            valid.append(date)
        else:
            missing = []
            if not events_path.exists():
                missing.append("events")
            if not mbo_path.exists():
                missing.append("mbo_raw")
            print(f"  SKIP {date}: missing {', '.join(missing)}")

    return valid


def cnn_event_preds_to_bars(date_str):
    """Convert CNN-Mamba event-window predictions to bar-level (100ms) predictions."""
    cnn_path = CNN_PRED_DIR / f"{date_str}_predictions.npz"
    events_path = EVENTS_DIR / f"{date_str}_mbo_events.npz"

    if not cnn_path.exists() or not events_path.exists():
        return None

    d = np.load(str(cnn_path))
    preds_3h = d['predictions']
    stride = int(d['stride'])
    ws = int(d['window_size'])

    ev = np.load(str(events_path))
    ts = ev['timestamps']

    pred_10s = preds_3h[:, 2]  # 10s horizon
    n_win = len(pred_10s)

    # Prediction timestamp = ts at end of window
    pred_ts = np.array([ts[min(i * stride + ws - 1, len(ts) - 1)] for i in range(n_win)])

    rth_start, rth_end = get_rth_bounds(date_str)
    rth_mask = (pred_ts >= rth_start) & (pred_ts < rth_end)
    rth_pred_ts = pred_ts[rth_mask]
    rth_pred_vals = pred_10s[rth_mask]

    bar_indices = ((rth_pred_ts - rth_start) // 100_000_000).astype(int)

    bar_preds = np.zeros(N_RTH_BARS, dtype=np.float32)
    for idx, val in zip(bar_indices, rth_pred_vals):
        if 0 <= idx < N_RTH_BARS:
            bar_preds[idx] = val

    # Forward-fill
    last_val = np.float32(0.0)
    for i in range(N_RTH_BARS):
        if bar_preds[i] != 0:
            last_val = bar_preds[i]
        else:
            bar_preds[i] = last_val

    return bar_preds


def create_prediction_npzs(valid_dates):
    """Create both-side and buy-only bar-level NPZs for all dates."""
    print(f"\n{'='*70}")
    print(f"STEP 1: Creating bar-level prediction NPZs for {len(valid_dates)} dates")
    print(f"{'='*70}")

    success_dates = []
    for date in valid_dates:
        bar_preds = cnn_event_preds_to_bars(date)
        if bar_preds is None:
            print(f"  SKIP {date}: conversion failed")
            continue

        # Both-side
        both_path = PRED_NPZ_DIR / f"{date}_unfiltered.npz"
        np.savez(str(both_path), predictions=bar_preds)

        # Buy-only: zero out negative (sell) signals
        buy_preds = bar_preds.copy()
        buy_preds[buy_preds < 0] = 0.0
        buy_path = BUY_NPZ_DIR / f"{date}_unfiltered.npz"
        np.savez(str(buy_path), predictions=buy_preds.astype(np.float32))

        n_buy = (buy_preds > 0.3).sum()
        n_sell = (bar_preds < -0.3).sum()
        n_total = (np.abs(bar_preds) > 0.3).sum()
        print(f"  {date}: {n_total} active bars (buy={n_buy}, sell={n_sell})")
        success_dates.append(date)

    print(f"\nCreated NPZs for {len(success_dates)}/{len(valid_dates)} dates")
    return success_dates


def run_fillsim(mbo_path, pred_npz_path, output_path, extra_flags):
    """Run the Rust fill_sim_cli and return parsed JSON results."""
    cmd = [
        str(FILL_SIM),
        "--mbo-file", str(mbo_path),
        "--predictions", str(pred_npz_path),
        "--output", str(output_path),
        "--signal-threshold", str(BASE_CFG["signal_threshold"]),
        "--take-profit-ticks", str(BASE_CFG["take_profit_ticks"]),
        "--stop-loss-ticks", str(BASE_CFG["stop_loss_ticks"]),
        "--hold-ms", str(BASE_CFG["hold_ms"]),
        "--latency-ms", str(BASE_CFG["latency_ms"]),
        "--quiet",
    ] + extra_flags

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[:300]}")
            return None
        if os.path.exists(output_path):
            with open(output_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"    EXCEPTION: {e}")
    return None


def run_all_configs(valid_dates):
    """Run all 4 configs across all valid dates."""
    print(f"\n{'='*70}")
    print(f"STEP 2: Running fill sim — {len(CONFIGS)} configs x {len(valid_dates)} dates = {len(CONFIGS)*len(valid_dates)} runs")
    print(f"{'='*70}")

    pred_dirs = {
        "both": PRED_NPZ_DIR,
        "buy": BUY_NPZ_DIR,
    }

    all_results = {}  # config_name -> [{date, result}, ...]
    start_time = time.time()

    for cfg_name, cfg in CONFIGS.items():
        pred_dir = pred_dirs[cfg["pred_dir"]]
        extra_flags = cfg["extra_flags"]
        print(f"\n--- {cfg_name} ---")

        day_results = []
        for date in valid_dates:
            pred_path = pred_dir / f"{date}_unfiltered.npz"
            mbo_path = MBO_DIR / f"glbx-mdp3-{date}.mbo.dbn.zst"

            if not pred_path.exists() or not mbo_path.exists():
                continue

            out_path = RESULTS_DIR / f"{cfg_name}_{date}.json"
            r = run_fillsim(mbo_path, pred_path, out_path, extra_flags)
            if r is not None:
                day_results.append({"date": date, "result": r})
                pnl = r.get("total_pnl_dollars", 0)
                trades = r.get("total_trades", 0)
                wr = r.get("win_rate", 0)
                print(f"  {date}: ${pnl:+.0f}  trades={trades}  WR={wr*100:.1f}%")
            else:
                print(f"  {date}: FAILED")

        all_results[cfg_name] = day_results

    elapsed = time.time() - start_time
    print(f"\nAll fill sim runs completed in {elapsed:.0f}s")
    return all_results


def compute_metrics(day_results):
    """Compute aggregate metrics from a list of day results."""
    if not day_results:
        return None

    all_trade_pnls = []
    daily_pnls = []
    daily_dates = []
    total_pnl = 0
    total_trades = 0
    total_signals = 0
    total_filled = 0
    total_gross_win = 0
    total_gross_loss = 0

    for dr in day_results:
        r = dr["result"]
        date = dr["date"]
        pnl = r.get("total_pnl_dollars", 0)
        total_pnl += pnl
        total_trades += r.get("total_trades", 0)
        total_signals += r.get("total_signals", 0)
        total_filled += r.get("total_filled", 0)
        daily_pnls.append(pnl)
        daily_dates.append(date)

        for t in r.get("trades", []):
            tp = t.get("pnl_dollars", 0)
            all_trade_pnls.append(tp)
            if tp > 0:
                total_gross_win += tp
            else:
                total_gross_loss += abs(tp)

    if not all_trade_pnls:
        return None

    arr = np.array(all_trade_pnls)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    wr = len(wins) / len(arr) if len(arr) > 0 else 0
    pf = total_gross_win / total_gross_loss if total_gross_loss > 0 else float('inf')

    mean_pnl = arr.mean()
    std_pnl = arr.std() if len(arr) > 1 else 1.0
    sharpe_trade = mean_pnl / std_pnl if std_pnl > 0 else 0

    # Daily Sharpe (annualized)
    d_arr = np.array(daily_pnls)
    daily_sharpe = (d_arr.mean() / d_arr.std()) * np.sqrt(252) if len(d_arr) > 1 and d_arr.std() > 0 else 0

    # Sortino (annualized, daily)
    neg_daily = d_arr[d_arr < 0]
    downside_std = neg_daily.std() if len(neg_daily) > 1 else d_arr.std()
    daily_sortino = (d_arr.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    # Day concentration
    if total_pnl != 0:
        day_conc = max(abs(p) for p in daily_pnls) / abs(total_pnl) if daily_pnls else 0
    else:
        day_conc = 0

    return {
        "total_pnl_dollars": round(total_pnl, 2),
        "n_trades": total_trades,
        "n_signals": total_signals,
        "n_filled": total_filled,
        "fill_rate": round(total_filled / total_signals, 4) if total_signals > 0 else 0,
        "win_rate": round(wr, 4),
        "mean_pnl_per_trade": round(mean_pnl, 2),
        "mean_net_ticks": round(mean_pnl / 12.50, 4),
        "sharpe_per_trade": round(sharpe_trade, 4),
        "daily_sharpe_ann": round(daily_sharpe, 2),
        "daily_sortino_ann": round(daily_sortino, 2),
        "profit_factor": round(pf, 4),
        "avg_win": round(wins.mean(), 2) if len(wins) > 0 else 0,
        "avg_loss": round(losses.mean(), 2) if len(losses) > 0 else 0,
        "n_days": len(day_results),
        "day_concentration": round(day_conc, 4),
        "daily_pnls": {dr["date"]: round(dr["result"].get("total_pnl_dollars", 0), 2) for dr in day_results},
    }


def classify_regime(daily_pnls_dict):
    """
    Classify each date as green/red/flat based on ES close-to-close direction.
    Since we don't have ES OHLCV data readily, we use a proxy: if the fill sim's
    net market movement (from first to last trade prices) is positive -> green day.
    This is approximate but useful for regime stratification.

    For now, return None if we can't determine. The per-day PnL is still reported.
    """
    # We'll attempt to extract regime info from the fill sim results themselves
    # by looking at first/last trade entry prices as a proxy for ES direction.
    return None


def print_summary(all_results):
    """Print comprehensive summary."""
    print(f"\n{'='*90}")
    print("EXTENDED OOT VALIDATION — COMPREHENSIVE SUMMARY")
    print(f"{'='*90}")

    summary = {}
    best_config = None
    best_sharpe = -999

    # Aggregate table
    header = f"{'Config':<20} {'Days':>5} {'Trades':>7} {'WR':>7} {'PF':>7} {'Sharpe':>8} {'Sortino':>8} {'TotalPnL':>10} {'AvgTicks':>9} {'DayConc':>8}"
    print(header)
    print("-" * len(header))

    for cfg_name in CONFIGS:
        day_results = all_results.get(cfg_name, [])
        metrics = compute_metrics(day_results)
        if not metrics:
            print(f"  {cfg_name}: NO RESULTS")
            continue

        summary[cfg_name] = metrics

        print(f"{cfg_name:<20} "
              f"{metrics['n_days']:>5} "
              f"{metrics['n_trades']:>7} "
              f"{metrics['win_rate']*100:>6.1f}% "
              f"{metrics['profit_factor']:>7.3f} "
              f"{metrics['daily_sharpe_ann']:>8.2f} "
              f"{metrics['daily_sortino_ann']:>8.2f} "
              f"${metrics['total_pnl_dollars']:>+9.0f} "
              f"{metrics['mean_net_ticks']:>+8.3f} "
              f"{metrics['day_concentration']:>8.2f}")

        if metrics['daily_sharpe_ann'] > best_sharpe:
            best_sharpe = metrics['daily_sharpe_ann']
            best_config = cfg_name

    # Per-day PnL for best config
    if best_config and best_config in summary:
        print(f"\n{'='*90}")
        print(f"PER-DAY PnL — BEST CONFIG: {best_config} (Daily Sharpe: {best_sharpe:.2f})")
        print(f"{'='*90}")

        daily = summary[best_config]["daily_pnls"]
        dates_sorted = sorted(daily.keys())

        green_days = []
        red_days = []
        flat_days = []

        print(f"{'Date':<12} {'PnL':>10} {'Cum PnL':>10}")
        print("-" * 34)
        cum_pnl = 0
        for date in dates_sorted:
            pnl = daily[date]
            cum_pnl += pnl
            marker = "+" if pnl > 0 else "-" if pnl < 0 else "="
            print(f"  {date:<10} ${pnl:>+9.0f} ${cum_pnl:>+9.0f}  {marker}")

            if pnl > 50:
                green_days.append(pnl)
            elif pnl < -50:
                red_days.append(pnl)
            else:
                flat_days.append(pnl)

        print(f"\n  Winning days: {sum(1 for p in daily.values() if p > 0)}/{len(daily)}")
        print(f"  Losing days:  {sum(1 for p in daily.values() if p < 0)}/{len(daily)}")
        print(f"  Flat days:    {sum(1 for p in daily.values() if p == 0)}/{len(daily)}")

        # Simple regime proxy: positive PnL days vs negative PnL days
        # (not ES direction, but still useful for consistency check)
        pos_days = [p for p in daily.values() if p > 0]
        neg_days = [p for p in daily.values() if p < 0]
        if pos_days:
            print(f"\n  Avg winning day: ${np.mean(pos_days):+.0f}")
        if neg_days:
            print(f"  Avg losing day:  ${np.mean(neg_days):+.0f}")

        # Day concentration check (HC #344: cap <= 0.70)
        day_conc = summary[best_config]["day_concentration"]
        conc_status = "PASS" if day_conc <= 0.70 else "FAIL"
        print(f"\n  Day concentration: {day_conc:.2%} [{conc_status} — HC #344 cap: 70%]")

        # Regime analysis using first/second half split as proxy
        first_half = [daily[d] for d in dates_sorted[:len(dates_sorted)//2]]
        second_half = [daily[d] for d in dates_sorted[len(dates_sorted)//2:]]
        if first_half and second_half:
            sh1 = (np.mean(first_half) / np.std(first_half)) * np.sqrt(252) if np.std(first_half) > 0 else 0
            sh2 = (np.mean(second_half) / np.std(second_half)) * np.sqrt(252) if np.std(second_half) > 0 else 0
            print(f"\n  First-half Sharpe (dates 1-{len(first_half)}):  {sh1:.2f}")
            print(f"  Second-half Sharpe (dates {len(first_half)+1}-{len(dates_sorted)}): {sh2:.2f}")
            regime_gap = abs(sh1 - sh2) / max(abs(sh1), abs(sh2), 0.01)
            gap_status = "PASS" if regime_gap <= 0.50 else "WARN"
            print(f"  Regime gap: {regime_gap:.2%} [{gap_status} — HC #428 cap: 50%]")

    # HC #428 / #432 validation checks
    print(f"\n{'='*90}")
    print("HC #428 / HC #432 VALIDATION CHECKS")
    print(f"{'='*90}")
    for cfg_name, metrics in summary.items():
        print(f"\n  {cfg_name}:")
        # Day concentration
        dc = metrics['day_concentration']
        print(f"    Day concentration: {dc:.2%} {'PASS' if dc <= 0.70 else 'FAIL'}")
        # Number of OOT days (must be >= 40)
        nd = metrics['n_days']
        print(f"    OOT days: {nd} {'PASS' if nd >= 40 else 'WARN (< 40)'}")
        # Sharpe positive?
        ds = metrics['daily_sharpe_ann']
        print(f"    Daily Sharpe (ann): {ds:.2f} {'PASS' if ds > 0 else 'FAIL'}")
        # Sortino positive?
        so = metrics['daily_sortino_ann']
        print(f"    Daily Sortino (ann): {so:.2f} {'PASS' if so > 0 else 'FAIL'}")

    # Save full summary
    save_summary = {}
    for cfg_name, metrics in summary.items():
        save_summary[cfg_name] = metrics

    summary_path = OUTPUT_DIR / "extended_oot_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(save_summary, f, indent=2, default=str)
    print(f"\nFull results saved to {summary_path}")

    return summary


def main():
    print(f"Extended OOT Validation — {datetime.datetime.now().isoformat()}")
    print(f"Fill Sim: {FILL_SIM}")
    print()

    # Discover valid dates
    print("Discovering valid dates (need CNN preds + MBO events + MBO raw)...")
    valid_dates = discover_dates()
    print(f"\nFound {len(valid_dates)} valid dates: {valid_dates[0]} to {valid_dates[-1]}")

    # Step 1: Create bar-level NPZs
    success_dates = create_prediction_npzs(valid_dates)

    # Step 2: Run all configs
    all_results = run_all_configs(success_dates)

    # Step 3: Summary
    summary = print_summary(all_results)

    if not summary:
        print("ERROR: No results produced!")
        sys.exit(1)

    print(f"\n{'='*90}")
    print("DONE — Extended OOT validation complete.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
