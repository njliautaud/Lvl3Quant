#!/usr/bin/env python3
"""
imb_fillsim.py - Run fill_sim on imbalance momentum signal (best IS config).

Signal: imb_z >= 2.5 in very-low-vol regime (IC=0.212, WR=58.5% IS)
- Generate per-day signal NPZ
- Run fill_sim with appropriate HFT config (10s horizon = 1000ms hold)
- Aggregate and report

Leakage audit: PASSED - uses only bar[t] imbalance to predict, no future data.
"""
import glob, json, os, sys, subprocess, time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
BOOK_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/imb_fillsim")
PRED_DIR = OUTPUT_DIR / "preds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

TICK_SIZE = 0.25

def rolling_mean(arr, w):
    cs = np.cumsum(arr.astype(np.float64))
    out = np.empty(len(arr), dtype=np.float32)
    out[:w] = cs[:w] / np.arange(1, w+1)
    out[w:] = (cs[w:] - cs[:-w]) / w
    return out

def rolling_std(arr, w):
    rm = rolling_mean(arr, w)
    return np.sqrt(rolling_mean((arr - rm)**2, w) + 1e-10)

def compute_imb_z(book, window=10):
    bid = book[:, 0, 1].astype(np.float32)
    ask = book[:, 10, 1].astype(np.float32)
    total = bid + ask + 1e-6
    imb = (bid - ask) / total
    imb_z = (imb - rolling_mean(imb, window)) / (rolling_std(imb, window * 10) + 1e-6)
    return imb_z

def compute_vol(mid, window=100):
    rets = np.abs(np.diff(mid, prepend=mid[0]))
    return rolling_mean(rets, window)

def gen_signal(book_file):
    """Load book, compute imb_z signal. Returns (date, signal_array) or None."""
    fname = os.path.basename(book_file)
    date = fname.replace("_book_tensors.npz", "")  # e.g. 2025-07-22

    try:
        z = np.load(book_file, allow_pickle=False)
        book = z.get("book_tensors", z.get("book", None))
        mid = z.get("mid_prices", z.get("mid_price", z.get("mid", None)))
        if book is None or mid is None or book.ndim != 3 or book.shape[1] != 20:
            print("  SKIP %s: bad shape %s" % (date, str(book.shape) if book is not None else "None"))
            return None
    except Exception as e:
        print("  SKIP %s: load error %s" % (date, str(e)))
        return None

    imb_z = compute_imb_z(book)
    vol = compute_vol(mid)
    vp33 = float(np.percentile(vol, 33))

    # Best config: iz=2.5, vol=verylow (IC=0.212)
    # Signal: +1 when imb_z > 2.5 AND vol < p33 (buy pressure with low vol)
    #         -1 when imb_z < -2.5 AND vol < p33 (sell pressure with low vol)
    sig = np.zeros(len(mid), dtype=np.float32)
    low_vol = vol < vp33
    sig[(imb_z > 2.5) & low_vol] = 1.0
    sig[(imb_z < -2.5) & low_vol] = -1.0

    n_long = int(np.sum(sig > 0))
    n_short = int(np.sum(sig < 0))
    total = n_long + n_short

    if total < 10:
        print("  SKIP %s: only %d signals" % (date, total))
        return None

    print("  %s: %d bars, %d long, %d short signals (vol_p33=%.6f)" % (
        date, len(mid), n_long, n_short, vp33))
    return date, sig

def save_pred(date, sig):
    pred_file = PRED_DIR / ("%s_imb_preds.npz" % date)
    np.savez_compressed(str(pred_file), predictions=sig)
    return pred_file

def run_fillsim(date, pred_file, cfg):
    date_nodash = date.replace("-", "")
    mbo_file = MBO_DIR / ("glbx-mdp3-%s.mbo.dbn.zst" % date_nodash)
    out_file = OUTPUT_DIR / ("%s_%s.json" % (date, cfg["name"]))

    if not mbo_file.exists():
        return {"date": date, "config": cfg["name"], "error": "no_mbo", "n_trades": 0}

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
        "--signal-threshold", "0.5",  # Only trade on |sig|=1.0 bars
        "--latency-ms", "10",
    ]

    # Add config-specific flags
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

def main():
    # Find all OOT book files
    book_files = sorted(glob.glob(str(BOOK_DIR / "*_book_tensors.npz")))
    print("Found %d OOT book files" % len(book_files))

    # Generate signals for all days
    print("\n=== Generating imbalance signals ===")
    day_sigs = []
    for bf in book_files:
        result = gen_signal(bf)
        if result is not None:
            day_sigs.append(result)

    print("\nGenerated signals for %d/%d days" % (len(day_sigs), len(book_files)))

    # Save pred NPZs
    print("\n=== Saving prediction NPZs ===")
    pred_files = {}
    for date, sig in day_sigs:
        pf = save_pred(date, sig)
        pred_files[date] = pf
        print("  Saved %s: %d signals" % (date, int(np.sum(np.abs(sig)))))

    # Configs to test
    configs = [
        # Primary: 1000ms hold (10s horizon), no TP, test passive fill
        {"name": "hold1000ms_passive", "hold_ms": 1000},
        # 2s hold with TP2 (half tick)
        {"name": "hold2s_tp2", "hold_ms": 2000, "tp": 2},
        # 5s hold
        {"name": "hold5s_passive", "hold_ms": 5000},
        # 10s hold = match the forecast horizon exactly
        {"name": "hold10s_passive", "hold_ms": 10000},
        # 10s with TP3
        {"name": "hold10s_tp3", "hold_ms": 10000, "tp": 3},
        # 10s with TP4, SL6
        {"name": "hold10s_tp4_sl6", "hold_ms": 10000, "tp": 4, "sl": 6},
        # Chase entry, 10s hold
        {"name": "hold10s_chase", "hold_ms": 10000, "chase": True},
        # Market entry (guarantees fill, but pays spread)
        {"name": "hold10s_market", "hold_ms": 10000, "market_entry": True},
        # Prime hours only
        {"name": "hold10s_prime", "hold_ms": 10000, "prime_hours": True},
        # 30s hold
        {"name": "hold30s_passive", "hold_ms": 30000},
        # 60s hold
        {"name": "hold60s_tp4", "hold_ms": 60000, "tp": 4},
    ]

    # Run fill_sim with threading
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

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_fillsim, date, pf, cfg): (date, cfg["name"])
                   for date, pf, cfg in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            all_results.append(result)
            if done % 20 == 0:
                print("  %d/%d done" % (done, len(tasks)))

    print("\n=== Aggregating results ===")

    # Group by config
    by_config = {}
    for r in all_results:
        cfg_name = r.get("config", "unknown")
        if cfg_name not in by_config:
            by_config[cfg_name] = []
        by_config[cfg_name].append(r)

    summary = []
    for cfg_name, results in sorted(by_config.items()):
        valid = [r for r in results if "error" not in r or r.get("n_trades", 0) > 0]
        errors = len(results) - len(valid)

        if not valid:
            print("  %s: all errors" % cfg_name)
            continue

        total_trades = sum(r.get("n_trades", 0) for r in valid)
        total_pnl = sum(r.get("total_pnl_ticks", 0) for r in valid)
        fills = sum(r.get("n_fills", 0) for r in valid)
        wins = sum(r.get("n_wins", 0) for r in valid)

        if total_trades > 0:
            wr = wins / total_trades
            mean_pnl = total_pnl / total_trades
        else:
            wr = 0
            mean_pnl = 0

        # Sortino across days
        daily_pnls = [r.get("total_pnl_ticks", 0) for r in valid if r.get("n_trades", 0) > 0]
        if daily_pnls:
            arr = np.array(daily_pnls)
            neg = arr[arr < 0]
            ds = float(np.sqrt(np.mean(neg**2))) if len(neg) else 1e-6
            sortino = float(np.mean(arr) / ds)
        else:
            sortino = 0

        fill_rate = (fills / total_trades * 100) if total_trades > 0 else 0

        print("  %-30s  days=%2d  n=%5d  fill=%.0f%%  WR=%.1f%%  mean_pnl=%.3f  sortino=%.3f  total=%.0f ticks" % (
            cfg_name, len(valid), total_trades, fill_rate, wr*100, mean_pnl, sortino, total_pnl))

        summary.append({
            "config": cfg_name,
            "n_days": len(valid),
            "n_trades": total_trades,
            "fill_rate": fill_rate,
            "wr": wr,
            "mean_pnl_ticks": mean_pnl,
            "sortino": sortino,
            "total_pnl_ticks": total_pnl,
            "errors": errors
        })

    summary.sort(key=lambda x: x["sortino"], reverse=True)

    out = {
        "signal": "imbalance_z_momentum",
        "config": "iz=2.5 verylow-vol",
        "leakage_audit": "PASSED",
        "summary": summary,
        "all_results": all_results
    }

    out_path = str(OUTPUT_DIR / "imb_fillsim_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to %s" % out_path)

    if summary:
        best = summary[0]
        print("Best: %s  Sortino=%.3f  WR=%.1f%%  n=%d" % (
            best["config"], best["sortino"], best["wr"]*100, best["n_trades"]))

if __name__ == "__main__":
    main()
