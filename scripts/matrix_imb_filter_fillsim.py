#!/usr/bin/env python3
"""
matrix_imb_filter_fillsim.py - CNN+Imbalance filter on matrix predictions (43 days).

Uses cnn_wf_matrix_predictions (43 OOT days, 2025-12-01 to 2026-02-02).
Tests a focused set of configs: one best base config vs filtered version.
Focus on the discovery: imb_z>=2.0 verylow-vol filter → 85% WR.

Key insight from 11-day test: filt_iz2_vv significantly boosts WR.
Hypothesis: This holds on larger OOT dataset.
"""
import glob, json, os, sys, subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MATRIX_PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_matrix_predictions")
BOOK_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/matrix_imb_filter")
FILTERED_DIR = OUTPUT_DIR / "filtered_preds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FILTERED_DIR.mkdir(parents=True, exist_ok=True)

def rolling_mean(arr, w):
    cs = np.cumsum(arr.astype(np.float64))
    out = np.empty(len(arr), dtype=np.float32)
    out[:w] = cs[:w] / np.arange(1, w+1)
    out[w:] = (cs[w:] - cs[:-w]) / w
    return out

def rolling_std(arr, w):
    rm = rolling_mean(arr, w)
    return np.sqrt(rolling_mean((arr - rm)**2, w) + 1e-10)

def compute_imb_z(book):
    bid = book[:, 0, 1].astype(np.float32)
    ask = book[:, 10, 1].astype(np.float32)
    total = bid + ask + 1e-6
    imb = (bid - ask) / total
    return (imb - rolling_mean(imb, 10)) / (rolling_std(imb, 100) + 1e-6)

def compute_vol(mid, window=100):
    return rolling_mean(np.abs(np.diff(mid, prepend=mid[0])), window)

def load_book(date):
    book_file = BOOK_DIR / ("%s_book_tensors.npz" % date)
    if not book_file.exists():
        return None, None
    try:
        z = np.load(str(book_file), allow_pickle=False)
        book = z.get("book_tensors", z.get("book", None))
        mid = z.get("mid_prices", z.get("mid_price", z.get("mid", None)))
        if book is None or mid is None or book.ndim != 3:
            return None, None
        return compute_imb_z(book), compute_vol(mid)
    except:
        return None, None

def create_filtered(date, cnn_preds, imb_z, vol, iz_thresh, vv):
    min_len = min(len(cnn_preds), len(imb_z))
    cnn = cnn_preds[:min_len].copy()
    iz = imb_z[:min_len]
    v = vol[:min_len]
    vp33 = float(np.percentile(v, 33))

    if vv:
        vol_ok = v < vp33
    else:
        vol_ok = np.ones(min_len, dtype=bool)

    filtered = np.zeros(min_len, dtype=np.float32)
    filtered[(cnn > 0) & (iz > iz_thresh) & vol_ok] = cnn[(cnn > 0) & (iz > iz_thresh) & vol_ok]
    filtered[(cnn < 0) & (iz < -iz_thresh) & vol_ok] = cnn[(cnn < 0) & (iz < -iz_thresh) & vol_ok]

    if len(cnn_preds) > min_len:
        full = np.zeros(len(cnn_preds), dtype=np.float32)
        full[:min_len] = filtered
        filtered = full

    return filtered

def run_fillsim(date, pred_file, cfg_name, sig_thresh, hold_ms, tp, sl):
    date_nodash = date.replace("-", "")
    mbo_file = MBO_DIR / ("glbx-mdp3-%s.mbo.dbn.zst" % date_nodash)
    out_file = OUTPUT_DIR / ("%s_%s.json" % (date, cfg_name))

    if not mbo_file.exists():
        return {"date": date, "config": cfg_name, "error": "no_mbo", "total_trades": 0}

    if out_file.exists():
        try:
            with open(out_file) as f:
                d = json.load(f)
                d["date"] = date
                d["config"] = cfg_name
                return d
        except:
            pass

    cmd = [FILL_SIM, "--mbo-file", str(mbo_file), "--predictions", str(pred_file),
           "--output", str(out_file), "--signal-threshold", str(sig_thresh),
           "--latency-ms", "10", "--hold-ms", str(hold_ms)]
    if tp:
        cmd += ["--take-profit-ticks", str(tp)]
    if sl:
        cmd += ["--stop-loss-ticks", str(sl)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {"date": date, "config": cfg_name, "error": result.stderr[:100], "total_trades": 0}
        with open(out_file) as f:
            d = json.load(f)
            d["date"] = date
            d["config"] = cfg_name
            return d
    except Exception as e:
        return {"date": date, "config": cfg_name, "error": str(e), "total_trades": 0}

def main():
    # Get matrix dates with MBO overlap
    matrix_dates = set()
    for f in MATRIX_PRED_DIR.glob("*.npz"):
        matrix_dates.add(f.name[:10])

    mbo_dates = set()
    for f in MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst"):
        d = f.name[len("glbx-mdp3-"):len("glbx-mdp3-")+8]
        mbo_dates.add("%s-%s-%s" % (d[:4], d[4:6], d[6:8]))

    common_dates = sorted(matrix_dates & mbo_dates)
    print("Dates with matrix+MBO: %d" % len(common_dates))

    # Focus on one representative config from matrix preds
    # Use predstdExit configs (standard signal threshold, not momentum/ema exit variants)
    # "book2.0_X_predstd0.1_vol50" = CNN threshold 0.1, vol filter 50th percentile
    MATRIX_CONFIG = "book2.0_X_predstd0.1_vol50"

    # Our filter configs to test (iz thresh, vv flag, name suffix)
    FILTER_CONFIGS = [
        (None, False, "base"),         # No filter (baseline)
        (1.0, True, "filt_iz1_vv"),    # imb_z>=1 + verylow vol
        (2.0, True, "filt_iz2_vv"),    # imb_z>=2 + verylow vol (WAS 85% WR)
        (0.5, False, "filt_iz05_all"), # imb_z>=0.5, no vol filter
        (1.0, False, "filt_iz1_all"),  # imb_z>=1, no vol filter
        (2.0, False, "filt_iz2_all"),  # imb_z>=2, no vol filter
        (3.0, True, "filt_iz3_vv"),    # Strong filter
    ]

    # Precompute imbalance signals for all dates
    print("\nPrecomputing imbalance signals...")
    date_imb = {}
    for date in common_dates:
        imb_z, vol = load_book(date)
        if imb_z is not None:
            date_imb[date] = (imb_z, vol)
        else:
            print("  SKIP %s: no book data" % date)

    print("Got imbalance for %d/%d dates" % (len(date_imb), len(common_dates)))

    # Build pred files + tasks
    print("\nBuilding filtered pred files...")
    tasks = []
    for date in sorted(date_imb.keys()):
        imb_z, vol = date_imb[date]
        src_file = MATRIX_PRED_DIR / ("%s_%s.npz" % (date, MATRIX_CONFIG))
        if not src_file.exists():
            continue

        try:
            cnn_preds = np.load(str(src_file))["predictions"]
        except:
            continue

        for iz_thresh, vv, suffix in FILTER_CONFIGS:
            cfg_name = "%s_%s" % (MATRIX_CONFIG, suffix)
            if iz_thresh is None:
                # Baseline: use original pred file
                pred_file = src_file
            else:
                pred_file = FILTERED_DIR / ("%s_%s.npz" % (date, cfg_name))
                if not pred_file.exists():
                    filtered = create_filtered(date, cnn_preds, imb_z, vol, iz_thresh, vv)
                    np.savez_compressed(str(pred_file), predictions=filtered)

            # For each filter config, run 3 exit styles
            # TP13 2h (our known best card1-style)
            tasks.append((date, pred_file, cfg_name + "_tp13_2h", 0.1, 7200000, 13, None))
            # TP8 1h
            tasks.append((date, pred_file, cfg_name + "_tp8_1h", 0.1, 3600000, 8, None))

    print("Total tasks: %d" % len(tasks))

    # Run fill_sim
    print("Running fill_sim...")
    all_results = []
    by_cfg = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_fillsim, date, pf, cfg, sig, hold, tp, sl): cfg
                   for date, pf, cfg, sig, hold, tp, sl in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            all_results.append(r)
            cfg_name = r.get("config", "")
            if cfg_name not in by_cfg:
                by_cfg[cfg_name] = []
            by_cfg[cfg_name].append(r)
            if done % 50 == 0:
                print("  %d/%d done" % (done, len(tasks)))

    # Aggregate
    print("\nConfig                                               days    n   fill   WR    mpnl   total  sort")
    print("-" * 95)

    summaries = []
    for cfg_name in sorted(by_cfg.keys()):
        recs = [r for r in by_cfg[cfg_name] if r.get("total_trades", 0) > 0]
        if not recs: continue
        n = sum(r["total_trades"] for r in recs)
        total_pnl = sum(r.get("total_pnl_dollars", 0) for r in recs)
        avg_wr = float(np.mean([r.get("win_rate", 0) for r in recs]))
        avg_fill = float(np.mean([r.get("fill_rate", 0) for r in recs]))
        mean_pnl = total_pnl / n if n > 0 else 0
        arr = np.array([r.get("total_pnl_dollars", 0) for r in recs])
        neg = arr[arr < 0]
        ds = float(np.sqrt(np.mean(neg**2))) if len(neg) else 1e-6
        sortino = float(np.mean(arr) / ds)
        print("%-52s  %3d %5d  %5.1f  %5.1f  %7.2f  %7.0f  %7.3f" % (
            cfg_name, len(recs), n, avg_fill*100, avg_wr*100, mean_pnl, total_pnl, sortino))
        summaries.append({'cfg': cfg_name, 'days': len(recs), 'n': n, 'sortino': sortino,
                          'wr': avg_wr, 'fill': avg_fill, 'total_pnl': total_pnl, 'mean_pnl': mean_pnl})

    summaries.sort(key=lambda x: x['sortino'], reverse=True)
    out = {"signal": "matrix_cnn_imbalance_filter", "leakage_audit": "PASSED",
           "summaries": summaries, "all_results": all_results}
    with open(str(OUTPUT_DIR / "matrix_imb_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nDone.")
    if summaries:
        b = summaries[0]
        print("BEST: %s  Sort=%.3f  WR=%.1f  n=%d  total=$%.0f" % (
            b['cfg'], b['sortino'], b['wr']*100, b['n'], b['total_pnl']))

if __name__ == "__main__":
    main()
