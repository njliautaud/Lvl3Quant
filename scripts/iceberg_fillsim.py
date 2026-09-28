#!/usr/bin/env python3
"""
iceberg_fillsim.py — Fill-sim validation of iceberg signal.

KEY FINDING: iceberg entry + iceberg exit = Sortino=0.363, WR=84.6%, n=50,218
              (signal_combo_entry_exit.py, 173 days, 40M bars)

Iceberg definition: large cumvol print at best bid/ask (>3x rolling mean).
  - Large bid iceberg  -> support -> LONG
  - Large ask iceberg  -> resistance -> SHORT

This script:
1. Computes iceberg signal per-day from OOT book tensors
2. Runs fill_sim binary with realistic execution (passive, chase, market)
3. Reports aggregate Sortino, WR, PnL across 68+ OOT days

Output: /home/jupiter/Lvl3Quant/data/processed/iceberg_fillsim/
"""

import glob, json, os, sys, subprocess, time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
BOOK_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/iceberg_fillsim")
PRED_DIR = OUTPUT_DIR / "preds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

TICK_SIZE = 0.25
WINDOW = 100  # rolling window for cumvol baseline (~10s at 100ms bars)

# ─────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────

def rolling_mean(arr, w):
    cs = np.cumsum(arr.astype(np.float64))
    out = np.empty(len(arr), dtype=np.float32)
    out[:w] = cs[:w] / np.arange(1, w+1)
    out[w:] = (cs[w:] - cs[:-w]) / w
    return out

def compute_iceberg_signal(book, cv_thresh=3.0, window=WINDOW):
    """
    Iceberg signal from book tensors (N, 20, 4).
    Features: [price_offset, bid_qty, ask_qty, cumvol]
    Levels 0-9: bid side, 10-19: ask side (based on price_offset sign).

    Returns signal array: +1 (bid iceberg=support=long), -1 (ask iceberg=resist=short), 0.
    """
    N = book.shape[0]

    # Best bid is level with most negative offset (level 0 in most implementations)
    # Best ask is level with most positive offset (level 10)
    # cumvol at best bid
    best_bid_cv = book[:, 0, 3].astype(np.float32)
    # cumvol at best ask
    best_ask_cv = book[:, 10, 3].astype(np.float32) if book.shape[1] > 10 else book[:, 0, 3].astype(np.float32)

    # Rolling mean baseline
    roll_bid_cv = rolling_mean(best_bid_cv, window)
    roll_ask_cv = rolling_mean(best_ask_cv, window)

    # Ratio: >3x rolling mean = iceberg print
    bid_ratio = best_bid_cv / (roll_bid_cv + 1e-6)
    ask_ratio = best_ask_cv / (roll_ask_cv + 1e-6)

    sig = np.zeros(N, dtype=np.float32)
    sig[bid_ratio >= cv_thresh] = 1.0   # bid iceberg -> support -> LONG
    sig[ask_ratio >= cv_thresh] = -1.0  # ask iceberg -> resistance -> SHORT

    # If both trigger same bar, clear (ambiguous)
    both = (bid_ratio >= cv_thresh) & (ask_ratio >= cv_thresh)
    sig[both] = 0.0

    return sig


def gen_signal(book_file, cv_thresh=3.0):
    """Load book, compute iceberg signal. Returns (date, signal) or None."""
    fname = os.path.basename(book_file)
    date = fname.replace("_book_tensors.npz", "")

    try:
        z = np.load(book_file, allow_pickle=False)
        book = z.get("book_tensors", z.get("book", None))
        mid = z.get("mid_prices", z.get("mid_price", z.get("mid", None)))
        if book is None or mid is None or book.ndim != 3 or book.shape[1] < 11:
            print("  SKIP %s: bad shape %s" % (date, str(book.shape) if book is not None else "None"))
            return None
    except Exception as e:
        print("  SKIP %s: load error %s" % (date, str(e)))
        return None

    sig = compute_iceberg_signal(book, cv_thresh=cv_thresh)

    n_long  = int(np.sum(sig > 0))
    n_short = int(np.sum(sig < 0))
    total = n_long + n_short

    if total < 5:
        print("  SKIP %s: only %d signals (iceberg too rare)" % (date, total))
        return None

    freq_pct = 100.0 * total / len(mid)
    print("  %s: %d bars, +%d/-%d signals (%.3f%%)" % (
        date, len(mid), n_long, n_short, freq_pct))
    return date, sig


def save_pred(date, sig, suffix=""):
    name = "%s_iceberg_preds%s.npz" % (date, suffix)
    pred_file = PRED_DIR / name
    np.savez_compressed(str(pred_file), predictions=sig)
    return pred_file


def run_fillsim(date, pred_file, cfg):
    date_nodash = date.replace("-", "")
    mbo_file = MBO_DIR / ("glbx-mdp3-%s.mbo.dbn.zst" % date_nodash)
    out_file = OUTPUT_DIR / ("%s_%s.json" % (date, cfg["name"]))

    if not mbo_file.exists():
        return {"date": date, "config": cfg["name"], "error": "no_mbo", "n_trades": 0}

    # Reuse existing result
    if out_file.exists():
        try:
            with open(out_file) as f:
                d = json.load(f)
                d["date"] = date
                d["config"] = cfg["name"]
                return d
        except:
            pass

    cmd = [
        FILL_SIM,
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", "0.5",
        "--latency-ms", "10",
    ]

    if cfg.get("hold_ms"):
        cmd += ["--hold-ms", str(cfg["hold_ms"])]
    if cfg.get("tp"):
        cmd += ["--take-profit-ticks", str(cfg["tp"])]
    if cfg.get("sl"):
        cmd += ["--stop-loss-ticks", str(cfg["sl"])]
    if cfg.get("prime_hours"):
        cmd += ["--prime-hours"]
    if cfg.get("chase"):
        cmd += ["--chase-entry"]
    if cfg.get("market_entry"):
        cmd += ["--market-entry"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {"date": date, "config": cfg["name"], "error": result.stderr[:200], "n_trades": 0}
        with open(out_file) as f:
            d = json.load(f)
            d["date"] = date
            d["config"] = cfg["name"]
            return d
    except Exception as e:
        return {"date": date, "config": cfg["name"], "error": str(e), "n_trades": 0}


def aggregate_results(all_results, config_name):
    """Compute aggregate stats for one config across all dates."""
    res = [r for r in all_results if r.get("config") == config_name and r.get("n_trades", 0) > 0]
    if not res:
        return None

    n_days = len(res)
    total_trades = sum(r.get("n_trades", 0) for r in res)
    total_pnl_ticks = sum(r.get("total_pnl_ticks", r.get("pnl_ticks", 0)) for r in res)

    # Per-trade returns
    per_trade_rets = []
    for r in res:
        nt = r.get("n_trades", 0)
        if nt > 0:
            # Use per_trade_pnl_ticks if available, else compute
            pt = r.get("per_trade_pnl_ticks", r.get("total_pnl_ticks", r.get("pnl_ticks", 0)) / nt)
            per_trade_rets.append(pt)

    avg_per_trade = total_pnl_ticks / max(total_trades, 1)

    # Sortino from daily P&L
    daily_pnls = [r.get("total_pnl_ticks", r.get("pnl_ticks", 0)) for r in res]
    neg = [p for p in daily_pnls if p < 0]
    ds = float(np.sqrt(np.mean([p**2 for p in neg]))) if neg else 1e-9
    mean_daily = float(np.mean(daily_pnls))
    sortino = mean_daily / ds if ds > 0 else 0.0

    # Win rate
    win_rates = [r.get("win_rate", 0.0) for r in res if "win_rate" in r]
    avg_wr = float(np.mean(win_rates)) if win_rates else 0.0

    return {
        "config": config_name,
        "n_days": n_days,
        "n_trades": total_trades,
        "trades_per_day": total_trades / n_days,
        "total_pnl_ticks": total_pnl_ticks,
        "avg_daily_pnl_ticks": mean_daily,
        "avg_per_trade_ticks": avg_per_trade,
        "sortino": sortino,
        "win_rate": avg_wr,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("ICEBERG FILL-SIM VALIDATION")
    print("Signal: cumvol at best bid/ask > 3x rolling mean (100-bar window)")
    print("=" * 70)

    book_files = sorted(glob.glob(str(BOOK_DIR / "*_book_tensors.npz")))
    print("Found %d OOT book files in %s" % (len(book_files), BOOK_DIR))

    if not book_files:
        print("ERROR: No book files found. Check BOOK_DIR path.")
        sys.exit(1)

    # Generate signals
    print("\n=== Generating iceberg signals ===")
    day_sigs = []
    for bf in book_files:
        result = gen_signal(bf, cv_thresh=3.0)
        if result is not None:
            day_sigs.append(result)

    print("\nGenerated signals for %d/%d days" % (len(day_sigs), len(book_files)))

    if not day_sigs:
        print("ERROR: No valid signal days. Check book tensor format.")
        sys.exit(1)

    # Save pred NPZs
    print("\n=== Saving prediction NPZs ===")
    pred_files = {}
    for date, sig in day_sigs:
        pf = save_pred(date, sig)
        pred_files[date] = pf

    # Configs — test the full range of execution approaches
    configs = [
        # Passive limit (best fill quality, lowest fill rate)
        {"name": "hold1000ms_passive",  "hold_ms": 1000},
        {"name": "hold5s_passive",      "hold_ms": 5000},
        {"name": "hold10s_passive",     "hold_ms": 10000},
        {"name": "hold30s_passive",     "hold_ms": 30000},
        # With TP/SL targets (let winners run, cut losers)
        {"name": "hold10s_tp3_sl6",     "hold_ms": 10000, "tp": 3, "sl": 6},
        {"name": "hold10s_tp4_sl8",     "hold_ms": 10000, "tp": 4, "sl": 8},
        {"name": "hold10s_tp6_sl12",    "hold_ms": 10000, "tp": 6, "sl": 12},
        {"name": "hold30s_tp4_sl8",     "hold_ms": 30000, "tp": 4, "sl": 8},
        # Chase entry (aggressive passive — improves fill rate at small cost)
        {"name": "hold10s_chase",       "hold_ms": 10000, "chase": True},
        {"name": "hold10s_chase_tp4",   "hold_ms": 10000, "chase": True, "tp": 4},
        # Market entry (guarantees fill, pays spread)
        {"name": "hold10s_market",      "hold_ms": 10000, "market_entry": True},
        # Prime hours (best execution conditions)
        {"name": "hold10s_prime",       "hold_ms": 10000, "prime_hours": True},
        {"name": "hold10s_prime_chase", "hold_ms": 10000, "prime_hours": True, "chase": True},
    ]

    # Run fill_sim
    print("\n=== Running fill_sim (%d days x %d configs) ===" % (len(day_sigs), len(configs)))

    all_results = []
    tasks = []
    for date, _ in day_sigs:
        if date not in pred_files:
            continue
        pf = pred_files[date]
        for cfg in configs:
            tasks.append((date, pf, cfg))

    print("Total tasks: %d" % len(tasks))

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_fillsim, date, pf, cfg): (date, cfg["name"])
                   for date, pf, cfg in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            all_results.append(result)
            if done % 30 == 0 or done == len(tasks):
                print("  %d/%d done" % (done, len(tasks)))

    # Save raw results
    raw_out = OUTPUT_DIR / "iceberg_fillsim_raw.json"
    with open(str(raw_out), "w") as f:
        json.dump(all_results, f, indent=2)

    # Aggregate and report
    print("\n=== RESULTS BY CONFIG ===")
    print("%-30s %6s %7s %9s %9s %9s %8s" % (
        "Config", "Days", "Trades", "T/Day", "PnL/T", "Sortino", "WR"))
    print("-" * 80)

    agg_results = []
    for cfg in configs:
        agg = aggregate_results(all_results, cfg["name"])
        if agg:
            agg_results.append(agg)
            flag = " *** POSITIVE" if agg["sortino"] > 0.1 else (" * " if agg["sortino"] > 0 else "")
            print("%-30s %6d %7d %9.1f %9.3f %9.3f %8.1f%%%s" % (
                agg["config"], agg["n_days"], agg["n_trades"],
                agg["trades_per_day"], agg["avg_per_trade_ticks"],
                agg["sortino"], 100*agg["win_rate"], flag))

    # Best config
    best = max(agg_results, key=lambda x: x["sortino"]) if agg_results else None
    if best:
        print("\n=== BEST CONFIG ===")
        print("Config:         %s" % best["config"])
        print("Sortino:        %.3f" % best["sortino"])
        print("Win Rate:       %.1f%%" % (100*best["win_rate"]))
        print("Trades/Day:     %.1f" % best["trades_per_day"])
        print("Avg PnL/Trade:  %.3f ticks (%.2f USD per trade)" % (
            best["avg_per_trade_ticks"], best["avg_per_trade_ticks"] * 12.5))
        print("Total Days:     %d" % best["n_days"])

    # Save summary
    summary_out = OUTPUT_DIR / "iceberg_fillsim_summary.json"
    with open(str(summary_out), "w") as f:
        json.dump({
            "meta": {
                "n_days": len(day_sigs),
                "signal": "iceberg_cv_3x",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "configs": agg_results,
            "best": best,
        }, f, indent=2)

    elapsed = time.time() - t0
    print("\nCompleted in %.0fs. Summary: %s" % (elapsed, summary_out))
    print("Report saved to: %s" % raw_out)


if __name__ == "__main__":
    main()
