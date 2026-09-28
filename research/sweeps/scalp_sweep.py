#!/usr/bin/env python3
"""Scalping-appropriate sweep for CNN 10-second predictions.

Tests TP 1-6 ticks, hold 10s/30s/60s/120s, signal threshold 0.1/0.3/0.5
across two signal types: Card1/2 (conv1.5_vol50) and Card4 (conv2.0_vol70).

72 configs per signal × 54 dates = 3,888 jobs per signal, 7,776 total.
14 workers.
"""
import sys, json, time, subprocess, os, math, statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

WORKERS = 14
LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_DIR = LVL3_ROOT / "data" / "processed" / "scalp_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_RT = 4.70

# Sweep parameters
SIGNALS = {
    "card12": "book_predstdExit_conv1.5_vol50",
    "card4":  "book_predstdExit_conv2.0_vol70",
}
TP_TICKS = [1, 2, 3, 4, 5, 6]
HOLD_MS  = [10000, 30000, 60000, 120000]
SIG_THRESH = [0.1, 0.3, 0.5]

# Progress file for partial results
PROGRESS_FILE = OUT_DIR / "sweep_progress.json"
RESULTS_FILE = OUT_DIR / "scalp_sweep_results.json"


def find_dates(signal_name):
    """Find intersection of prediction dates and MBO dates."""
    pred_dates = {}
    mbo_dates = {}

    for f in sorted(PRED_DIR.glob(f"*_{signal_name}.npz")):
        date = f.name[:10]  # 2025-12-01
        pred_dates[date] = f

    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
        nodash = f.name.split("-")[2].split(".")[0]  # 20251201
        date = f"{nodash[:4]}-{nodash[4:6]}-{nodash[6:8]}"
        mbo_dates[date] = f

    common = sorted(set(pred_dates.keys()) & set(mbo_dates.keys()))
    return common, pred_dates, mbo_dates


def config_key(signal_tag, tp, hold, sig):
    return f"{signal_tag}_tp{tp}_hold{hold}_sig{sig}"


def run_sim(args):
    """Run a single simulation. Returns (config_key, date, result_dict or None)."""
    cfg_key, date, cmd, out_path = args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and Path(out_path).exists():
            with open(out_path) as f:
                data = json.load(f)
            # Clean up individual file to save disk
            os.remove(out_path)
            return (cfg_key, date, data)
        return (cfg_key, date, None)
    except Exception as e:
        return (cfg_key, date, None)


def aggregate_config(results_list):
    """Aggregate results across dates for a single config."""
    valid = [r for r in results_list if r is not None]
    if not valid:
        return None

    total_pnl = 0
    total_trades = 0
    total_wins = 0
    total_fills = 0
    total_signals = 0
    daily_pnls = []
    all_mae = []
    all_mfe = []
    all_pnl_per_trade = []

    for r in valid:
        trades = r.get("trades", [])
        summary = r.get("summary", r)  # some formats nest it

        n_trades = len(trades) if trades else summary.get("total_trades", 0)
        total_trades += n_trades
        total_fills += summary.get("total_fills", n_trades)
        total_signals += summary.get("total_signals", 0)

        day_pnl = summary.get("total_pnl_ticks", 0) * TICK_VALUE
        day_pnl -= n_trades * COMMISSION_RT
        daily_pnls.append(day_pnl)
        total_pnl += day_pnl

        wins = summary.get("winning_trades", 0)
        if wins == 0 and trades:
            wins = sum(1 for t in trades if t.get("pnl_ticks", 0) > 0)
        total_wins += wins

        # MAE/MFE from trades
        for t in trades:
            if "mae_ticks" in t:
                all_mae.append(t["mae_ticks"])
            if "mfe_ticks" in t:
                all_mfe.append(t["mfe_ticks"])
            pnl_t = t.get("pnl_ticks", 0) * TICK_VALUE - COMMISSION_RT
            all_pnl_per_trade.append(pnl_t)

    n_days = len(valid)
    sharpe = 0
    if daily_pnls and len(daily_pnls) > 1:
        mean_d = statistics.mean(daily_pnls)
        std_d = statistics.stdev(daily_pnls)
        if std_d > 0:
            sharpe = (mean_d / std_d) * math.sqrt(252)

    fill_rate = (total_fills / total_signals * 100) if total_signals > 0 else 0
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_pnl_trade_ticks = (statistics.mean(all_pnl_per_trade) / TICK_VALUE) if all_pnl_per_trade else 0
    avg_pnl_trade_usd = statistics.mean(all_pnl_per_trade) if all_pnl_per_trade else 0

    return {
        "sharpe": round(sharpe, 3),
        "total_pnl": round(total_pnl, 2),
        "total_trades": total_trades,
        "trades_per_day": round(total_trades / n_days, 1) if n_days > 0 else 0,
        "win_rate": round(wr, 1),
        "avg_pnl_per_trade_ticks": round(avg_pnl_trade_ticks, 3),
        "avg_pnl_per_trade_usd": round(avg_pnl_trade_usd, 2),
        "avg_mae": round(statistics.mean(all_mae), 2) if all_mae else 0,
        "avg_mfe": round(statistics.mean(all_mfe), 2) if all_mfe else 0,
        "fill_rate": round(fill_rate, 1),
        "n_days": n_days,
        "daily_mean_pnl": round(statistics.mean(daily_pnls), 2) if daily_pnls else 0,
        "daily_std_pnl": round(statistics.stdev(daily_pnls), 2) if len(daily_pnls) > 1 else 0,
    }


def main():
    print(f"=== SCALP SWEEP — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Workers: {WORKERS}")

    all_jobs = []
    config_results = {}  # cfg_key -> [results per date]

    for signal_tag, signal_name in SIGNALS.items():
        dates, pred_map, mbo_map = find_dates(signal_name)
        print(f"\n{signal_tag} ({signal_name}): {len(dates)} dates")

        for tp in TP_TICKS:
            for hold in HOLD_MS:
                for sig in SIG_THRESH:
                    cfg_key = config_key(signal_tag, tp, hold, sig)
                    config_results[cfg_key] = []

                    for date in dates:
                        nodash = date.replace("-", "")
                        mbo_path = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"
                        pred_path = PRED_DIR / f"{date}_{signal_name}.npz"
                        out_path = OUT_DIR / f"{cfg_key}_{date}.json"

                        cmd = [
                            str(BINARY),
                            "--mbo-file", str(mbo_path),
                            "--predictions", str(pred_path),
                            "--output", str(out_path),
                            "--signal-threshold", str(sig),
                            "--take-profit-ticks", str(tp),
                            "--hold-ms", str(hold),
                            "--max-wait-bars", "30",
                            "--latency-ms", "5",
                        ]

                        all_jobs.append((cfg_key, date, cmd, str(out_path)))

    total_configs = len(config_results)
    total_jobs = len(all_jobs)
    print(f"\nTotal configs: {total_configs}")
    print(f"Total jobs: {total_jobs}")
    print(f"Starting sweep with {WORKERS} workers...\n")

    completed = 0
    failed = 0
    start_time = time.time()
    last_report = start_time

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_sim, job): job for job in all_jobs}

        for future in as_completed(futures):
            cfg_key, date, result = future.result()
            config_results[cfg_key].append(result)
            completed += 1
            if result is None:
                failed += 1

            now = time.time()
            # Progress report every 60 seconds or every 500 jobs
            if now - last_report > 60 or completed % 500 == 0:
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_jobs - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total_jobs}] {completed/total_jobs*100:.1f}% | "
                      f"{rate:.1f} jobs/s | ETA {eta/60:.1f}min | "
                      f"failed: {failed}", flush=True)
                last_report = now

                # Save partial aggregation
                if completed > total_jobs * 0.1:  # After 10%
                    save_partial(config_results, completed, total_jobs)

    elapsed = time.time() - start_time
    print(f"\n=== SWEEP COMPLETE in {elapsed/60:.1f} min ===")
    print(f"Completed: {completed}, Failed: {failed}\n")

    # Final aggregation
    final_results = {}
    for cfg_key, results in config_results.items():
        agg = aggregate_config(results)
        if agg:
            final_results[cfg_key] = agg

    # Sort by Sharpe
    ranked = sorted(final_results.items(), key=lambda x: x[1]["sharpe"], reverse=True)

    # Save full results
    with open(RESULTS_FILE, "w") as f:
        json.dump({k: v for k, v in ranked}, f, indent=2)

    # Print top 20 for each signal
    for signal_tag in SIGNALS:
        print(f"\n{'='*80}")
        print(f"TOP 20 — {signal_tag.upper()}")
        print(f"{'='*80}")
        print(f"{'Rank':>4} {'Config':<40} {'Sharpe':>7} {'PnL':>10} {'Trades':>7} "
              f"{'T/Day':>6} {'WR%':>5} {'$/Trade':>8} {'MAE':>5} {'MFE':>5} {'Fill%':>6}")
        print("-" * 120)

        signal_ranked = [(k, v) for k, v in ranked if k.startswith(signal_tag)]
        for i, (cfg, m) in enumerate(signal_ranked[:20]):
            # Parse config for display
            parts = cfg.replace(f"{signal_tag}_", "")
            print(f"{i+1:>4} {parts:<40} {m['sharpe']:>7.2f} {m['total_pnl']:>10.0f} "
                  f"{m['total_trades']:>7} {m['trades_per_day']:>6.1f} {m['win_rate']:>5.1f} "
                  f"{m['avg_pnl_per_trade_usd']:>8.2f} {m['avg_mae']:>5.1f} {m['avg_mfe']:>5.1f} "
                  f"{m['fill_rate']:>6.1f}")

    # Overall top 20
    print(f"\n{'='*80}")
    print(f"OVERALL TOP 20 (BOTH SIGNALS)")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'Config':<45} {'Sharpe':>7} {'PnL':>10} {'Trades':>7} "
          f"{'T/Day':>6} {'WR%':>5} {'$/Trade':>8} {'MAE':>5} {'MFE':>5} {'Fill%':>6}")
    print("-" * 125)
    for i, (cfg, m) in enumerate(ranked[:20]):
        print(f"{i+1:>4} {cfg:<45} {m['sharpe']:>7.2f} {m['total_pnl']:>10.0f} "
              f"{m['total_trades']:>7} {m['trades_per_day']:>6.1f} {m['win_rate']:>5.1f} "
              f"{m['avg_pnl_per_trade_usd']:>8.2f} {m['avg_mae']:>5.1f} {m['avg_mfe']:>5.1f} "
              f"{m['fill_rate']:>6.1f}")

    print(f"\nFull results saved to: {RESULTS_FILE}")


def save_partial(config_results, completed, total):
    """Save partial aggregation for early inspection."""
    try:
        partial = {}
        for cfg_key, results in config_results.items():
            valid = [r for r in results if r is not None]
            if len(valid) >= 5:  # At least 5 dates
                agg = aggregate_config(results)
                if agg:
                    partial[cfg_key] = agg

        if partial:
            ranked = sorted(partial.items(), key=lambda x: x[1]["sharpe"], reverse=True)
            with open(PROGRESS_FILE, "w") as f:
                json.dump({
                    "completed": completed,
                    "total": total,
                    "pct": round(completed / total * 100, 1),
                    "top_configs": {k: v for k, v in ranked[:30]},
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
