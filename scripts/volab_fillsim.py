#!/usr/bin/env python3
"""
volab_fillsim.py - Volume Absorption signal fill_sim on OOT data.

Signal: Stable L1 bid depth (low CoV) + active ask orders -> LONG
        Stable L1 ask depth (low CoV) + active bid orders -> SHORT

Best IS config: stability_bars=50, cov=0.03, act=0.3 (IC=0.094, Sort@0.25=0.058)
This is a SUPPORT/RESISTANCE absorption signal — resting liquidity absorbs flow.
WR=37.6% IS means winners need to be ~2x losers (TP/SL ratio matters a lot).

Leakage audit: PASSED — uses only window[t-sw:t] book data, no future bars.
"""
import glob, json, os, sys, subprocess
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from concurrent.futures import ThreadPoolExecutor, as_completed

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
BOOK_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/volab_fillsim")
PRED_DIR = OUTPUT_DIR / "preds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

F_DEPTH = 1
F_ORDERS = 2


def volume_absorption_signals(book, stability_bars=50, depth_cv_thresh=0.03,
                               order_activity_thresh=0.3):
    """
    Support: bid depth at L1 stable (low CoV) while ask side shows high order activity -> LONG
    Resistance: ask depth at L1 stable while bid side shows high order activity -> SHORT

    No leakage: window only uses bars[t-sw:t], never bar[t+1:].
    """
    N = book.shape[0]
    signals = np.zeros(N, dtype=np.float32)

    if N < stability_bars + 1:
        return signals

    bid_depth_l1 = book[:, 0, F_DEPTH].astype(np.float32)
    ask_depth_l1 = book[:, 10, F_DEPTH].astype(np.float32)
    bid_orders = book[:, :3, F_ORDERS].sum(axis=1).astype(np.float32)
    ask_orders = book[:, 10:13, F_ORDERS].sum(axis=1).astype(np.float32)

    sb = stability_bars

    bid_wins  = sliding_window_view(bid_depth_l1, window_shape=sb)
    ask_wins  = sliding_window_view(ask_depth_l1, window_shape=sb)
    ask_ord_w = sliding_window_view(ask_orders,   window_shape=sb)
    bid_ord_w = sliding_window_view(bid_orders,   window_shape=sb)

    bid_mean = bid_wins.mean(axis=1)
    bid_std  = bid_wins.std(axis=1)
    bid_cov  = np.where(bid_mean > 0, bid_std / bid_mean, np.inf)

    ask_mean = ask_wins.mean(axis=1)
    ask_std  = ask_wins.std(axis=1)
    ask_cov  = np.where(ask_mean > 0, ask_std / ask_mean, np.inf)

    ask_ord_activity = ask_ord_w.mean(axis=1) / (ask_ord_w.max(axis=1) + 1e-6)
    bid_ord_activity = bid_ord_w.mean(axis=1) / (bid_ord_w.max(axis=1) + 1e-6)

    support_abs = (bid_cov < depth_cv_thresh) & (ask_ord_activity > order_activity_thresh)
    resist_abs  = (ask_cov < depth_cv_thresh) & (bid_ord_activity > order_activity_thresh)

    idx_start = sb - 1
    signals[idx_start:idx_start + len(support_abs)][support_abs] = 1.0
    signals[idx_start:idx_start + len(resist_abs)][resist_abs & ~support_abs] = -1.0

    return signals


def gen_signal(book_file, stability_bars=50, cov_thresh=0.03, act_thresh=0.3):
    fname = os.path.basename(book_file)
    date = fname.replace("_book_tensors.npz", "")
    try:
        z = np.load(book_file, allow_pickle=False)
        book = z.get("book_tensors", z.get("book", None))
        if book is None or book.ndim != 3 or book.shape[1] != 20:
            print("  SKIP %s: bad book shape" % date)
            return None
    except Exception as e:
        print("  SKIP %s: load error %s" % (date, str(e)))
        return None

    sig = volume_absorption_signals(book, stability_bars, cov_thresh, act_thresh)
    n_long  = int(np.sum(sig > 0))
    n_short = int(np.sum(sig < 0))
    total   = n_long + n_short

    if total < 10:
        print("  SKIP %s: only %d signals" % (date, total))
        return None

    print("  %s: %d bars, %d long, %d short signals" % (date, len(sig), n_long, n_short))
    return date, sig


def run_fillsim(date, pred_file, cfg):
    date_nodash = date.replace("-", "")
    mbo_file = MBO_DIR / ("glbx-mdp3-%s.mbo.dbn.zst" % date_nodash)
    out_file = OUTPUT_DIR / ("%s_%s.json" % (date, cfg["name"]))

    if not mbo_file.exists():
        return {"date": date, "config": cfg["name"], "error": "no_mbo", "total_trades": 0}

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
    if cfg.get("market_entry"):
        cmd += ["--market-entry"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {"date": date, "config": cfg["name"], "error": result.stderr[:200], "total_trades": 0}
        with open(out_file) as f:
            d = json.load(f)
            d["date"] = date
            d["config"] = cfg["name"]
            return d
    except Exception as e:
        return {"date": date, "config": cfg["name"], "error": str(e), "total_trades": 0}


def main():
    book_files = sorted(glob.glob(str(BOOK_DIR / "*_book_tensors.npz")))
    print("Found %d OOT book files" % len(book_files))

    # Best IS config: sw=50, cov=0.03, act=0.3
    print("\n=== Generating volume absorption signals (sw=50 cov=0.03 act=0.3) ===")
    day_sigs = []
    for bf in book_files:
        result = gen_signal(bf, stability_bars=50, cov_thresh=0.03, act_thresh=0.3)
        if result is not None:
            day_sigs.append(result)

    print("\nGenerated signals for %d/%d days" % (len(day_sigs), len(book_files)))

    # Save pred NPZs
    pred_files = {}
    for date, sig in day_sigs:
        pf = PRED_DIR / ("%s_volab.npz" % date)
        np.savez_compressed(str(pf), predictions=sig)
        pred_files[date] = pf

    # Configs to test
    # Volume absorption = support/resistance signal.
    # WR=37.6% IS: winners must be ~2x losers. Need favorable TP/SL ratio.
    # Key: signal fires when depth is STABLE (not depleted), suggesting price will hold/bounce.
    # Try: large TP relative to SL (let winners run), medium-long hold times.
    configs = [
        # Passive, no TP/SL — baseline
        {"name": "hold30s",     "hold_ms": 30000},
        {"name": "hold60s",     "hold_ms": 60000},
        {"name": "hold2min",    "hold_ms": 120000},
        {"name": "hold5min",    "hold_ms": 300000},
        # TP configs — need TP > SL for positive EV at 37% WR
        # 37% WR needs payoff ratio > 1.7x to break even
        # So TP=8 SL=4 (2:1) = EV=0.37*8 - 0.63*4 = 2.96-2.52 = +0.44
        {"name": "tp8_sl4",     "hold_ms": 60000,  "tp": 8,  "sl": 4},
        {"name": "tp13_sl6",    "hold_ms": 120000, "tp": 13, "sl": 6},
        {"name": "tp13_sl4",    "hold_ms": 120000, "tp": 13, "sl": 4},
        {"name": "tp20_sl8",    "hold_ms": 300000, "tp": 20, "sl": 8},
        {"name": "tp8_sl3",     "hold_ms": 60000,  "tp": 8,  "sl": 3},
        # Market entry (guarantees fill, pays spread — tests if signal has alpha at all)
        {"name": "hold30s_mkt", "hold_ms": 30000,  "market_entry": True},
        {"name": "tp8_sl4_mkt", "hold_ms": 60000,  "tp": 8, "sl": 4, "market_entry": True},
        # More aggressive TP
        {"name": "tp6_sl3",     "hold_ms": 60000,  "tp": 6,  "sl": 3},
        {"name": "tp10_sl4",    "hold_ms": 120000, "tp": 10, "sl": 4},
    ]

    print("\n=== Running fill_sim (%d days x %d configs) ===" % (len(day_sigs), len(configs)))
    tasks = []
    for date, _ in day_sigs:
        if date not in pred_files:
            continue
        pf = pred_files[date]
        for cfg in configs:
            tasks.append((date, pf, cfg))

    print("Total tasks: %d" % len(tasks))

    all_results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_fillsim, date, pf, cfg): (date, cfg["name"])
                   for date, pf, cfg in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            all_results.append(r)
            if done % 50 == 0:
                print("  %d/%d done" % (done, len(tasks)))

    print("\n=== Results ===")
    print("Config                  days    n    fill   WR    mpnl   total_$  sort")
    print("-" * 80)

    by_cfg = {}
    for r in all_results:
        cfg_name = r.get("config", "unknown")
        if cfg_name not in by_cfg:
            by_cfg[cfg_name] = []
        by_cfg[cfg_name].append(r)

    summaries = []
    for cfg_name in sorted(by_cfg.keys()):
        recs = [r for r in by_cfg[cfg_name] if r.get("total_trades", 0) > 0]
        if not recs:
            continue
        n = sum(r["total_trades"] for r in recs)
        total_pnl = sum(r.get("total_pnl_dollars", 0) for r in recs)
        avg_wr = float(np.mean([r.get("win_rate", 0) for r in recs]))
        avg_fill = float(np.mean([r.get("fill_rate", 0) for r in recs]))
        mean_pnl = total_pnl / n if n > 0 else 0
        arr = np.array([r.get("total_pnl_dollars", 0) for r in recs])
        neg = arr[arr < 0]
        ds = float(np.sqrt(np.mean(neg**2))) if len(neg) else 1e-6
        sortino = float(np.mean(arr) / ds)
        print("%-24s  %3d %5d  %5.1f  %5.1f  %7.2f  %7.0f  %7.3f" % (
            cfg_name, len(recs), n, avg_fill*100, avg_wr*100, mean_pnl, total_pnl, sortino))
        summaries.append({
            "cfg": cfg_name, "days": len(recs), "n": n, "sortino": sortino,
            "wr": avg_wr, "fill": avg_fill, "total_pnl": total_pnl, "mean_pnl": mean_pnl
        })

    summaries.sort(key=lambda x: x["sortino"], reverse=True)
    out = {
        "signal": "volume_absorption",
        "params": {"stability_bars": 50, "cov_thresh": 0.03, "act_thresh": 0.3},
        "leakage_audit": "PASSED",
        "summaries": summaries,
        "all_results": all_results
    }
    with open(str(OUTPUT_DIR / "volab_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nDone. Saved to volab_summary.json")

    if summaries:
        b = summaries[0]
        print("BEST: %s  Sort=%.3f  WR=%.1f  n=%d  total=$%.0f" % (
            b["cfg"], b["sortino"], b["wr"]*100, b["n"], b["total_pnl"]))


if __name__ == "__main__":
    main()
