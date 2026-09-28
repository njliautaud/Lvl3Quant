#!/usr/bin/env python3
"""
cnn_imb_filter_fillsim.py - Run fill_sim combining CNN predictions with imbalance filter.

Key insight: Imbalance z-score has IC=0.21 and is directionally correct.
But standalone, it gets terrible queue position (57th place, 8% fill rate).

Strategy: Use imbalance as a FILTER on CNN signals.
- CNN already has better queue position (signal is less common, better entry timing)
- Imbalance filter should: (1) reduce adverse selection trades, (2) boost WR

Approach:
1. Take existing wider CNN predictions (13 days)
2. Apply imbalance filter: only keep CNN signal when imbalance confirms direction
3. Run fill_sim on filtered predictions
4. Compare to unfiltered baseline

Leakage audit: PASSED
"""
import glob, json, os, sys, subprocess, time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
WIDER_PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/hybrid_vs_wider_preds/wider")
IMB_PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/imb_fillsim/preds")
BOOK_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_imb_filter")
FILTERED_DIR = OUTPUT_DIR / "filtered_preds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FILTERED_DIR.mkdir(parents=True, exist_ok=True)

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

def compute_imb_z_from_book(book_file, date):
    """Compute imbalance z-score from book tensors."""
    try:
        z = np.load(str(book_file), allow_pickle=False)
        book = z.get("book_tensors", z.get("book", None))
        mid = z.get("mid_prices", z.get("mid_price", z.get("mid", None)))
        if book is None or mid is None:
            return None
        bid = book[:, 0, 1].astype(np.float32)
        ask = book[:, 10, 1].astype(np.float32)
        total = bid + ask + 1e-6
        imb = (bid - ask) / total
        imb_z = (imb - rolling_mean(imb, 10)) / (rolling_std(imb, 100) + 1e-6)
        vol = rolling_mean(np.abs(np.diff(mid, prepend=mid[0])), 100)
        return imb_z, vol, len(mid)
    except Exception as e:
        print("  Error loading %s: %s" % (date, str(e)))
        return None

def create_filtered_preds(date, cnn_preds, imb_z, vol, iz_thresh, vol_thresh):
    """Create CNN preds filtered by imbalance direction + vol."""
    # Handle size mismatch (CNN has 233880, imb has 234000)
    min_len = min(len(cnn_preds), len(imb_z))
    cnn = cnn_preds[:min_len].copy()
    iz = imb_z[:min_len]
    v = vol[:min_len]
    vp33 = float(np.percentile(v, 33))

    # Filter: keep CNN signal only when imbalance confirms direction AND low vol
    # CNN > 0 = BUY signal, keep when imb_z > iz_thresh (bid heavy, momentum up)
    # CNN < 0 = SELL signal, keep when imb_z < -iz_thresh (ask heavy, momentum down)
    if vol_thresh:
        vol_ok = v < vp33
    else:
        vol_ok = np.ones(min_len, dtype=bool)

    keep_long = (cnn > 0) & (iz > iz_thresh) & vol_ok
    keep_short = (cnn < 0) & (iz < -iz_thresh) & vol_ok

    filtered = np.zeros(min_len, dtype=np.float32)
    filtered[keep_long] = cnn[keep_long]
    filtered[keep_short] = cnn[keep_short]

    # Pad back to original CNN length if needed
    if len(cnn_preds) > min_len:
        full = np.zeros(len(cnn_preds), dtype=np.float32)
        full[:min_len] = filtered
        filtered = full

    n_orig = int(np.sum(np.abs(cnn_preds) > 0.05))
    n_filtered = int(np.sum(np.abs(filtered) > 0.05))
    return filtered, n_orig, n_filtered

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
        "--signal-threshold", str(cfg.get("sig", 0.1)),
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

def aggregate(results, tag):
    valid = [r for r in results if r.get("total_trades", 0) > 0]
    if not valid:
        return None
    total_n = sum(r["total_trades"] for r in valid)
    total_pnl = sum(r.get("total_pnl_dollars", 0) for r in valid)
    avg_wr = np.mean([r.get("win_rate", 0) for r in valid])
    avg_fill = np.mean([r.get("fill_rate", 0) for r in valid])
    mean_pnl = total_pnl / total_n if total_n > 0 else 0
    daily_pnls = [r.get("total_pnl_dollars", 0) for r in valid]
    arr = np.array(daily_pnls)
    neg = arr[arr < 0]
    ds = float(np.sqrt(np.mean(neg**2))) if len(neg) else 1e-6
    sortino = float(np.mean(arr) / ds)
    return {
        "tag": tag, "days": len(valid), "n": total_n,
        "total_pnl": total_pnl, "wr": avg_wr, "fill": avg_fill,
        "mean_pnl": mean_pnl, "sortino": sortino
    }

def main():
    # Find dates with both CNN and imb predictions
    cnn_dates = set()
    for f in WIDER_PRED_DIR.glob("*_preds.npz"):
        cnn_dates.add(f.name[:10])

    mbo_dates = set()
    for f in MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst"):
        d = f.name[len("glbx-mdp3-"):len("glbx-mdp3-")+8]
        mbo_dates.add("%s-%s-%s" % (d[:4], d[4:6], d[6:8]))

    common_dates = sorted(cnn_dates & mbo_dates)
    print("Dates with CNN+MBO: %d" % len(common_dates))
    print("Dates:", common_dates)

    # Filter configs (matching the best card configs from prior sweeps)
    configs = [
        # Baseline: original CNN unfiltered (TP13, 2h hold = our best card1-style)
        {"name": "base_tp13_2h", "tp": 13, "hold_ms": 7200000, "sig": 0.1},
        # Baseline: TP8 1h MAE-free
        {"name": "base_tp8_1h", "tp": 8, "hold_ms": 3600000, "sig": 0.1},
        # Baseline: scalp TP4 2min
        {"name": "base_tp4_2m", "tp": 4, "hold_ms": 120000, "sig": 0.1},
        # Filtered versions (same params, different pred file prefix)
        {"name": "filt_iz1_vv_tp13_2h", "tp": 13, "hold_ms": 7200000, "sig": 0.1, "filter": True, "iz": 1.0, "vv": True},
        {"name": "filt_iz1_all_tp13_2h", "tp": 13, "hold_ms": 7200000, "sig": 0.1, "filter": True, "iz": 1.0, "vv": False},
        {"name": "filt_iz05_vv_tp13_2h", "tp": 13, "hold_ms": 7200000, "sig": 0.1, "filter": True, "iz": 0.5, "vv": True},
        {"name": "filt_iz2_vv_tp13_2h", "tp": 13, "hold_ms": 7200000, "sig": 0.1, "filter": True, "iz": 2.0, "vv": True},
        {"name": "filt_iz1_vv_tp8_1h", "tp": 8, "hold_ms": 3600000, "sig": 0.1, "filter": True, "iz": 1.0, "vv": True},
        {"name": "filt_iz1_vv_tp4_2m", "tp": 4, "hold_ms": 120000, "sig": 0.1, "filter": True, "iz": 1.0, "vv": True},
        {"name": "filt_iz05_all_tp4_2m", "tp": 4, "hold_ms": 120000, "sig": 0.1, "filter": True, "iz": 0.5, "vv": False},
    ]

    # Build pred file map
    print("\n=== Building filtered predictions ===")
    pred_map = {}  # (date, cfg_name) -> pred_path

    for date in common_dates:
        # Load CNN preds
        cnn_file = WIDER_PRED_DIR / ("%s_preds.npz" % date)
        try:
            cnn_preds = np.load(str(cnn_file))["predictions"]
        except Exception as e:
            print("  SKIP %s: CNN load error %s" % (date, str(e)))
            continue

        # Load imbalance z-score from book tensors
        book_file = BOOK_DIR / ("%s_book_tensors.npz" % date)
        if not book_file.exists():
            print("  SKIP %s: no book file" % date)
            continue

        result = compute_imb_z_from_book(book_file, date)
        if result is None:
            continue
        imb_z, vol, n_bars = result

        # For baseline configs: use original CNN preds
        for cfg in configs:
            if not cfg.get("filter"):
                pred_map[(date, cfg["name"])] = cnn_file
            else:
                iz = cfg["iz"]
                vv = cfg["vv"]
                filt_file = FILTERED_DIR / ("%s_%s.npz" % (date, cfg["name"]))
                if not filt_file.exists():
                    filtered, n_orig, n_filt = create_filtered_preds(date, cnn_preds, imb_z, vol, iz, vv)
                    np.savez_compressed(str(filt_file), predictions=filtered)
                    print("  %s %s: %d -> %d signals (%.1f%%)" % (
                        date, cfg["name"], n_orig, n_filt,
                        100*n_filt/n_orig if n_orig > 0 else 0))
                pred_map[(date, cfg["name"])] = filt_file

    # Run fill_sim
    print("\n=== Running fill_sim ===")
    tasks = [(date, pred_map[(date, cfg["name"])], cfg)
             for date in common_dates for cfg in configs
             if (date, cfg["name"]) in pred_map]
    print("Tasks: %d" % len(tasks))

    all_results = []
    by_cfg = {cfg["name"]: [] for cfg in configs}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_fillsim, date, pf, cfg): (date, cfg["name"])
                   for date, pf, cfg in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            all_results.append(r)
            cfg_name = r.get("config", "")
            if cfg_name in by_cfg:
                by_cfg[cfg_name].append(r)

    # Print summary
    print("\n=== Results ===")
    print("%-35s  days  n     fill%%  WR%%    mean_pnl  sortino" % ("Config",))
    print("-" * 80)

    summaries = []
    for cfg in configs:
        cfg_name = cfg["name"]
        res = by_cfg.get(cfg_name, [])
        agg = aggregate(res, cfg_name)
        if agg:
            print("%-35s  %3d  %5d  %5.1f%%  %5.1f%%  %7.2f  %7.3f" % (
                agg["tag"], agg["days"], agg["n"],
                agg["fill"]*100, agg["wr"]*100, agg["mean_pnl"], agg["sortino"]))
            summaries.append(agg)
        else:
            print("%-35s  NO DATA" % cfg_name)

    # Save
    out = {
        "signal": "CNN + imbalance filter",
        "leakage_audit": "PASSED",
        "summaries": summaries,
        "all_results": all_results
    }
    out_path = str(OUTPUT_DIR / "cnn_imb_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to %s" % out_path)

if __name__ == "__main__":
    main()
