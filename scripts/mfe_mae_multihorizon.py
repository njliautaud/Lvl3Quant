#!/usr/bin/env python3
"""
MFE/MAE Multi-Horizon Analysis — Jupiter
==========================================
Computes MFE/MAE/edge_ratio/IC for each signal at 6 horizons.
Reads from dl_book_cache_oot (same format as signal_combo_entry_exit.py).

Book tensor format: (N, 20, 4)
  [:, :, 0] = price_offsets
  [:, :, 1] = bid_qty (levels 0-9)
  [:, :, 2] = ask_qty (levels 10-19)
  [:, :, 3] = cum_vol

Signals computed (same as signal_combo_entry_exit.py):
  imbalance, queue_fade, iceberg, sweep, momentum

Output: /home/jupiter/Lvl3Quant/data/processed/mfe_mae_multihorizon_results.json
"""

import os, sys, glob, json, time, gc
import numpy as np
from pathlib import Path

BOOK_DIR = "/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot"
OUT_JSON = "/home/jupiter/Lvl3Quant/data/processed/mfe_mae_multihorizon_results.json"
LOG_FILE = "/home/jupiter/Lvl3Quant/logs/mfe_mae_multihorizon.log"
BATCH_SIZE = 15  # days per batch

TICK_SIZE = 0.25
SIGNALS = ["imbalance", "queue_fade", "iceberg", "sweep", "momentum"]
HORIZONS = [
    ("10s",   100),
    ("30s",   300),
    ("1min",  600),
    ("5min",  3000),
    ("15min", 9000),
    ("30min", 18000),
]

WINDOW = 50   # rolling window for queue/iceberg signals
IB_THRESH = 0.2
SHORT_W = 20
LONG_W = 100


def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def compute_signals(tensors, mid):
    N = len(tensors)
    price_offsets = tensors[:, :, 0]
    bid_qty       = tensors[:, :, 1]
    ask_qty       = tensors[:, :, 2]
    cum_vol       = tensors[:, :, 3]

    # Bid/ask masks (first 10 levels = bid, next 10 = ask)
    bid_mask = np.zeros((1, 20), dtype=np.float32)
    bid_mask[0, :10] = 1.0
    ask_mask = np.zeros((1, 20), dtype=np.float32)
    ask_mask[0, 10:] = 1.0

    bid_qty_total = np.sum(bid_qty * bid_mask, axis=1)
    ask_qty_total = np.sum(ask_qty * ask_mask, axis=1)
    total_qty = bid_qty_total + ask_qty_total + 1e-6

    # Imbalance
    imbalance_raw = (bid_qty_total - ask_qty_total) / total_qty
    imbalance_sig = np.where(imbalance_raw > IB_THRESH, 1,
                    np.where(imbalance_raw < -IB_THRESH, -1, 0)).astype(np.int8)

    # Queue fade (bid/ask ratio vs rolling mean)
    best_bid_qty = bid_qty[:, 0]
    best_ask_qty = ask_qty[:, 10] if tensors.shape[1] > 10 else ask_qty[:, 0]
    roll_bid = np.convolve(best_bid_qty, np.ones(WINDOW)/WINDOW, mode='same')
    roll_ask = np.convolve(best_ask_qty, np.ones(WINDOW)/WINDOW, mode='same')
    bid_ratio = best_bid_qty / (roll_bid + 1e-6)
    ask_ratio = best_ask_qty / (roll_ask + 1e-6)
    qf_sig = np.where((bid_ratio < 0.5) & (ask_ratio > 1.5), -1,
             np.where((ask_ratio < 0.5) & (bid_ratio > 1.5), 1, 0)).astype(np.int8)

    # Iceberg (cumvol ratio spike at best bid)
    best_bid_cumvol = cum_vol[:, 0]
    roll_bid_cv = np.convolve(best_bid_cumvol, np.ones(WINDOW)/WINDOW, mode='same')
    bid_cv_ratio = best_bid_cumvol / (roll_bid_cv + 1e-6)
    best_ask_cumvol = cum_vol[:, 10] if tensors.shape[1] > 10 else cum_vol[:, 0]
    roll_ask_cv = np.convolve(best_ask_cumvol, np.ones(WINDOW)/WINDOW, mode='same')
    ask_cv_ratio = best_ask_cumvol / (roll_ask_cv + 1e-6)
    iceberg_sig = np.where(bid_cv_ratio > 3.0, 1,
                  np.where(ask_cv_ratio > 3.0, -1, 0)).astype(np.int8)

    # Sweep (bid/ask concentration)
    top3_bid_qty = bid_qty[:, :3].sum(axis=1)
    bid_conc = bid_qty[:, 0] / (top3_bid_qty + 1e-6)
    ask_top3 = ask_qty[:, 10:13].sum(axis=1) if ask_qty.shape[1] > 10 else ask_qty[:, :3].sum(axis=1)
    ask_conc = ask_qty[:, 10] / (ask_top3 + 1e-6) if ask_qty.shape[1] > 10 else bid_conc
    sweep_sig = np.where((bid_conc < 0.4) & (ask_conc > 0.7), -1,
                np.where((ask_conc < 0.4) & (bid_conc > 0.7), 1, 0)).astype(np.int8)

    # Momentum (dual-horizon price change)
    p_change_short = np.zeros(N, dtype=np.float32)
    p_change_long  = np.zeros(N, dtype=np.float32)
    p_change_short[SHORT_W:] = mid[SHORT_W:] - mid[:-SHORT_W]
    p_change_long[LONG_W:]   = mid[LONG_W:] - mid[:-LONG_W]
    mom_sig = np.where((p_change_short > TICK_SIZE) & (p_change_long > TICK_SIZE), 1,
              np.where((p_change_short < -TICK_SIZE) & (p_change_long < -TICK_SIZE), -1, 0)).astype(np.int8)

    return {
        "imbalance":  imbalance_sig,
        "queue_fade": qf_sig,
        "iceberg":    iceberg_sig,
        "sweep":      sweep_sig,
        "momentum":   mom_sig,
    }


def compute_mfe_mae_batch(prices, sigs, horizon_bars):
    results = {}
    for sig_name in SIGNALS:
        if sig_name not in sigs:
            continue
        sig = sigs[sig_name].astype(np.float32)
        prices_f = prices.astype(np.float32)
        active = np.where(sig != 0)[0]
        active = active[active + horizon_bars < len(prices_f)]
        if len(active) == 0:
            results[sig_name] = {"n": 0}
            continue
        if len(active) > 200000:
            rng = np.random.default_rng(42)
            active = rng.choice(active, 200000, replace=False)
        dirs = np.sign(sig[active]).astype(np.float32)
        CHUNK = 5000
        mfes, maes, rets, sig_mags = [], [], [], []
        for start in range(0, len(active), CHUNK):
            idx = active[start:start+CHUNK]
            d = dirs[start:start+CHUNK]
            paths = np.stack([prices_f[i:i+horizon_bars] - prices_f[i] for i in idx], axis=0)
            signed = paths * d[:, None]
            mfes.append(signed.max(axis=1))
            maes.append((-signed).max(axis=1))
            rets.append(signed[:, -1])
            sig_mags.append(np.abs(sig[idx]))
            del paths, signed
        mfes = np.concatenate(mfes)
        maes = np.concatenate(maes)
        rets = np.concatenate(rets)
        sig_mags = np.concatenate(sig_mags)

        mfe_m = float(np.mean(mfes))
        mae_m = float(np.mean(maes))
        edge = mfe_m / (mfe_m + mae_m + 1e-9)

        # IC: correlation of signal magnitude vs return
        corr = float(np.corrcoef(sig_mags, rets)[0, 1]) if len(sig_mags) > 1 else 0.0
        if np.isnan(corr):
            corr = 0.0

        # Sortino: mean return / downside std
        neg = rets[rets < 0]
        dstd = float(np.std(neg)) if len(neg) > 1 else 1e-9
        sortino = float(np.mean(rets)) / (dstd + 1e-9)

        results[sig_name] = {
            "n": len(active),
            "mfe": mfe_m,
            "mae": mae_m,
            "edge": edge,
            "ic": corr,
            "sortino": sortino,
        }
    return results


def main():
    log("=== MFE/MAE Multi-Horizon (Jupiter, compute-from-book-tensors) ===")
    files = sorted(glob.glob(os.path.join(BOOK_DIR, "*_book_tensors.npz")))
    log(f"Found {len(files)} files, batch_size={BATCH_SIZE}")

    from collections import defaultdict
    acc = {sig: {h_name: {"n": 0, "mfe_wsum": 0.0, "mae_wsum": 0.0, "edge_wsum": 0.0,
                           "ic_wsum": 0.0, "sortino_wsum": 0.0}
                 for h_name, _ in HORIZONS}
           for sig in SIGNALS}

    n_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_i in range(n_batches):
        batch_files = files[batch_i*BATCH_SIZE:(batch_i+1)*BATCH_SIZE]
        log(f"Batch {batch_i+1}/{n_batches}: files {batch_i*BATCH_SIZE+1}-{batch_i*BATCH_SIZE+len(batch_files)}")

        tensors_list, mids_list = [], []
        for f in batch_files:
            try:
                z = np.load(f)
                t = z["book_tensors"].astype(np.float32)    # (N, 20, 4)
                m = z["mid_prices"].astype(np.float32)
                tensors_list.append(t)
                mids_list.append(m)
                z.close()
            except Exception as e:
                log(f"  Skip {os.path.basename(f)}: {e}")

        if not tensors_list:
            continue

        tensors_cat = np.concatenate(tensors_list, axis=0)
        mids_cat = np.concatenate(mids_list, axis=0)
        min_len = min(len(tensors_cat), len(mids_cat))
        tensors_cat = tensors_cat[:min_len]
        mids_cat = mids_cat[:min_len]
        log(f"  Loaded {min_len:,} bars")

        sigs = compute_signals(tensors_cat, mids_cat)

        for h_name, h_bars in HORIZONS:
            log(f"  Horizon {h_name}...")
            r = compute_mfe_mae_batch(mids_cat, sigs, h_bars)
            for sig, rv in r.items():
                if rv.get("n", 0) > 0:
                    a = acc[sig][h_name]
                    n = rv["n"]
                    a["n"] += n
                    a["mfe_wsum"] += rv["mfe"] * n
                    a["mae_wsum"] += rv["mae"] * n
                    a["edge_wsum"] += rv["edge"] * n
                    a["ic_wsum"] += rv["ic"] * n
                    a["sortino_wsum"] += rv["sortino"] * n

        del tensors_list, mids_list, tensors_cat, mids_cat, sigs
        gc.collect()

    # Build output
    out = {"completed": time.strftime("%Y-%m-%d %H:%M:%S"), "signals": {}}
    for sig in SIGNALS:
        out["signals"][sig] = {}
        for h_name, _ in HORIZONS:
            a = acc[sig][h_name]
            n = a["n"]
            if n > 0:
                out["signals"][sig][h_name] = {
                    "n": n,
                    "mfe_ticks": round(a["mfe_wsum"]/n / TICK_SIZE, 4),
                    "mae_ticks": round(a["mae_wsum"]/n / TICK_SIZE, 4),
                    "edge_ratio": round(a["edge_wsum"]/n, 4),
                    "ic": round(a["ic_wsum"]/n, 4),
                    "sortino": round(a["sortino_wsum"]/n, 4),
                }
            else:
                out["signals"][sig][h_name] = {"n": 0}

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    log(f"COMPLETE: {OUT_JSON}")

    log("\n=== SUMMARY ===")
    for sig in SIGNALS:
        for h_name, _ in HORIZONS:
            r = out["signals"][sig].get(h_name, {})
            if r.get("n", 0) > 0:
                log(f"  {sig:15s} {h_name:5s}: MFE={r['mfe_ticks']:.3f}t MAE={r['mae_ticks']:.3f}t edge={r['edge_ratio']:.3f} IC={r['ic']:.4f} Sortino={r['sortino']:.3f} n={r['n']}")


if __name__ == "__main__":
    main()
