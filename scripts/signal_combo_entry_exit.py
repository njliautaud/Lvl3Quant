#!/usr/bin/env python3
"""
signal_combo_entry_exit.py — Signal Combination Entry/Exit Pair Tester

Tests the hypothesis that different signals are optimal for entry vs exit.
e.g., "iceberg at entry for timing precision, momentum for exit condition"

Architecture:
- Entry signal: defines WHEN to enter (entry timing)
- Exit signal: defines WHEN to exit (exit condition, separate from TP/SL)
- Fallback: max hold = 3000 bars (~5min) if exit signal doesn't trigger

Also tests:
- Single signals at multiple horizons (baseline)
- Regime-conditional: signal only active when vol_percentile > threshold
- Horizon-conditional: only hold if favorable signal persists

Signals: iceberg, queue_fade, imbalance, sweep, momentum, CNN_prediction
Data: /home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot (173 days)
CNN preds: /home/jupiter/Lvl3Quant/data/processed/cnn_wf_matrix_predictions

Output: /home/jupiter/Lvl3Quant/data/processed/signal_combo_results.json
        /home/jupiter/Lvl3Quant/data/processed/signal_combo_report.txt

Deploy: nohup python3 /home/jupiter/Lvl3Quant/scripts/signal_combo_entry_exit.py > /home/jupiter/Lvl3Quant/logs/signal_combo.log 2>&1 &
"""

import glob
import json
import os
import time
import sys
from itertools import product
import numpy as np
from scipy import stats as ss

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
BOOK_DIR  = "/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot"
PRED_DIR  = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_matrix_predictions"
OUT_JSON  = "/home/jupiter/Lvl3Quant/data/processed/signal_combo_results.json"
OUT_TXT   = "/home/jupiter/Lvl3Quant/data/processed/signal_combo_report.txt"
LOG_DIR   = "/home/jupiter/Lvl3Quant/logs"

TICK_SIZE    = 0.25
MAX_HOLD     = 3000    # ~5 min fallback hold
TC_TICKS     = 0.5     # transaction cost: 2 ticks round trip = 0.5 ticks per side
MIN_TRADES   = 200     # minimum trade count for valid result

# Horizons tested for single-signal baseline
HORIZONS = {"10s": 100, "30s": 300, "1min": 600, "5min": 3000}

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────
def load_book_data():
    """Load all OOT book tensors. Returns (tensors, mids, dates)."""
    files = sorted(glob.glob(os.path.join(BOOK_DIR, "*_book_tensors.npz")))
    log(f"Loading {len(files)} book tensor files...")

    tensors_list, mids_list, dates_list = [], [], []
    for i, f in enumerate(files):
        try:
            z = np.load(f, allow_pickle=False)
            t = z["book_tensors"]   # (N, 20, 4)
            m = z["mid_prices"].astype(np.float32)
            date = os.path.basename(f)[:10]
            tensors_list.append(t)
            mids_list.append(m)
            dates_list.append((date, len(t)))
            if (i+1) % 30 == 0:
                log(f"  {i+1}/{len(files)} loaded...")
        except Exception as e:
            log(f"  Skip {os.path.basename(f)}: {e}")

    tensors = np.concatenate(tensors_list, axis=0)
    mids    = np.concatenate(mids_list, axis=0)
    log(f"Total: {len(mids):,} bars, {len(tensors_list)} days")
    return tensors, mids, dates_list


def load_cnn_predictions(total_len):
    """
    Load CNN WF matrix predictions. Picks the most recent config variant.
    Returns float32 array of shape (total_len,), padded with 0 for missing.
    """
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*_book2.0_X_emaExit0.0_vol50.npz")))
    if not pred_files:
        pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.npz")))
        # Deduplicate by date: take first config per date
        seen_dates = set()
        unique_files = []
        for f in pred_files:
            date = os.path.basename(f)[:10]
            if date not in seen_dates:
                seen_dates.add(date)
                unique_files.append(f)
        pred_files = unique_files

    log(f"Loading {len(pred_files)} CNN prediction files...")
    preds_list = []
    for f in pred_files:
        try:
            z = np.load(f, allow_pickle=False)
            preds_list.append(z["predictions"])
        except Exception as e:
            log(f"  Skip pred {os.path.basename(f)}: {e}")

    if not preds_list:
        log("WARNING: No CNN predictions found, using zeros")
        return np.zeros(total_len, dtype=np.float32)

    preds = np.concatenate(preds_list)
    # Align to total_len
    if len(preds) > total_len:
        preds = preds[:total_len]
    elif len(preds) < total_len:
        preds = np.concatenate([preds, np.zeros(total_len - len(preds), dtype=np.float32)])

    log(f"CNN preds: {len(preds):,} bars, mean={preds.mean():.4f}, std={preds.std():.4f}")
    return preds.astype(np.float32)


# ─────────────────────────────────────────────────────────
# SIGNAL COMPUTATION (same as Razer script for consistency)
# ─────────────────────────────────────────────────────────
def compute_signals(tensors, mid, cnn_preds):
    """Compute all signals. Returns dict of int8 arrays {-1, 0, 1}."""
    N = len(tensors)
    price_offsets = tensors[:, :, 0]
    bid_qty       = tensors[:, :, 1]
    ask_qty       = tensors[:, :, 2]
    cum_vol       = tensors[:, :, 3]

    bid_mask = price_offsets < 0
    ask_mask = price_offsets > 0

    bid_qty_total = np.sum(bid_qty * bid_mask, axis=1)
    ask_qty_total = np.sum(ask_qty * ask_mask, axis=1)

    # 1. IMBALANCE
    total_qty = bid_qty_total + ask_qty_total + 1e-6
    imbalance_raw = (bid_qty_total - ask_qty_total) / total_qty
    ib_thresh = 0.3
    imbalance_sig = np.where(imbalance_raw > ib_thresh, 1,
                    np.where(imbalance_raw < -ib_thresh, -1, 0)).astype(np.int8)

    # 2. QUEUE_FADE
    best_bid_qty = bid_qty[:, 0]
    best_ask_qty = ask_qty[:, 10] if tensors.shape[1] > 10 else ask_qty[:, 0]
    window = 100
    roll_bid = np.convolve(best_bid_qty, np.ones(window)/window, mode='same')
    roll_ask = np.convolve(best_ask_qty, np.ones(window)/window, mode='same')
    bid_ratio = best_bid_qty / (roll_bid + 1e-6)
    ask_ratio = best_ask_qty / (roll_ask + 1e-6)
    qf_sig = np.where((bid_ratio < 0.5) & (ask_ratio > 1.5), -1,
             np.where((ask_ratio < 0.5) & (bid_ratio > 1.5), 1, 0)).astype(np.int8)

    # 3. ICEBERG
    best_bid_cumvol = cum_vol[:, 0]
    best_ask_cumvol = cum_vol[:, 10] if cum_vol.shape[1] > 10 else cum_vol[:, 0]
    roll_bid_cv = np.convolve(best_bid_cumvol, np.ones(window)/window, mode='same')
    roll_ask_cv = np.convolve(best_ask_cumvol, np.ones(window)/window, mode='same')
    bid_cv_ratio = best_bid_cumvol / (roll_bid_cv + 1e-6)
    ask_cv_ratio = best_ask_cumvol / (roll_ask_cv + 1e-6)
    iceberg_sig = np.where(bid_cv_ratio > 3.0, 1,
                  np.where(ask_cv_ratio > 3.0, -1, 0)).astype(np.int8)

    # 4. SWEEP
    top3_bid_qty = bid_qty[:, :3].sum(axis=1)
    bid_conc = bid_qty[:, 0] / (top3_bid_qty + 1e-6)
    ask_top3 = ask_qty[:, 10:13].sum(axis=1) if ask_qty.shape[1] > 12 else ask_qty[:, 0]
    ask_conc = ask_qty[:, 10] / (ask_top3 + 1e-6) if ask_qty.shape[1] > 10 else bid_conc
    sweep_sig = np.where((bid_conc < 0.4) & (ask_conc > 0.7), -1,
                np.where((ask_conc < 0.4) & (bid_conc > 0.7), 1, 0)).astype(np.int8)

    # 5. MOMENTUM
    p_change_short = np.zeros(N, dtype=np.float32)
    p_change_long  = np.zeros(N, dtype=np.float32)
    short_w, long_w = 50, 300
    p_change_short[short_w:] = mid[short_w:] - mid[:-short_w]
    p_change_long[long_w:]   = mid[long_w:] - mid[:-long_w]
    mom_sig = np.where((p_change_short > TICK_SIZE) & (p_change_long > TICK_SIZE), 1,
              np.where((p_change_short < -TICK_SIZE) & (p_change_long < -TICK_SIZE), -1, 0)).astype(np.int8)

    # 6. CNN PREDICTION
    if cnn_preds is not None and len(cnn_preds) == N:
        cnn_mean = float(np.mean(cnn_preds))
        cnn_std  = float(np.std(cnn_preds)) + 1e-9
        # Use 1-sigma threshold
        cnn_sig = np.where(cnn_preds > cnn_mean + 0.5 * cnn_std, 1,
                  np.where(cnn_preds < cnn_mean - 0.5 * cnn_std, -1, 0)).astype(np.int8)
    else:
        cnn_sig = np.zeros(N, dtype=np.int8)

    # 7. VOLATILITY (for regime conditioning)
    roll_std = np.zeros(N, dtype=np.float32)
    vw = 600   # 1-min window for vol
    p_diff = np.abs(np.diff(mid, prepend=mid[0]))
    roll_std_v = np.convolve(p_diff, np.ones(vw)/vw, mode='same')
    vol_pct = np.zeros(N, dtype=np.float32)
    if roll_std_v.max() > 0:
        vol_pct = (np.argsort(np.argsort(roll_std_v)) / N).astype(np.float32)

    return {
        "imbalance":  imbalance_sig,
        "queue_fade": qf_sig,
        "iceberg":    iceberg_sig,
        "sweep":      sweep_sig,
        "momentum":   mom_sig,
        "cnn":        cnn_sig,
        "_vol_pct":   vol_pct,   # internal, for regime filtering
    }


# ─────────────────────────────────────────────────────────
# BACKTESTING ENGINE (vectorized)
# ─────────────────────────────────────────────────────────
def _find_first_exit(entry_idx, exit_mask_long, exit_mask_short, directions, max_hold, N):
    """
    Vectorized: for each entry, find the first exit signal within max_hold.
    exit_mask_long = where exit_sig == 1 (fires for short entries to exit)
    exit_mask_short = where exit_sig == -1 (fires for long entries to exit)

    Returns hold_bars array (int32) — max_hold if no exit found.
    """
    hold_bars = np.full(len(entry_idx), max_hold, dtype=np.int32)

    # Precompute cumulative arrays for fast "next signal" lookup
    # For each position i, next_exit_long[i] = next bar where exit_sig == 1
    # Use suffix scan approach
    next_long  = np.full(N + 1, N, dtype=np.int32)  # next bar with exit==1
    next_short = np.full(N + 1, N, dtype=np.int32)  # next bar with exit==-1

    # Scan backwards to build "next exit" arrays
    long_idx  = np.where(exit_mask_long)[0]
    short_idx = np.where(exit_mask_short)[0]

    if len(long_idx) > 0:
        ptr = len(long_idx) - 1
        for i in range(N - 1, -1, -1):
            while ptr >= 0 and long_idx[ptr] > i:
                ptr -= 1
            # Find first long_idx > i
            hi = np.searchsorted(long_idx, i + 1)
            next_long[i] = long_idx[hi] if hi < len(long_idx) else N

    if len(short_idx) > 0:
        for i in range(N - 1, -1, -1):
            hi = np.searchsorted(short_idx, i + 1)
            next_short[i] = short_idx[hi] if hi < len(short_idx) else N

    # For each entry, lookup first opposing exit
    for k, (ei, d) in enumerate(zip(entry_idx, directions)):
        if d > 0:  # long: exit when exit_sig == -1
            nx = next_short[ei]
        else:      # short: exit when exit_sig == 1
            nx = next_long[ei]

        hold = min(nx - ei, max_hold)
        hold_bars[k] = max(hold, 1)

    return hold_bars


def backtest_entry_exit(entry_sig, exit_sig, mid, max_hold=MAX_HOLD, tc=TC_TICKS):
    """
    Vectorized entry/exit backtest.
    Enter on entry_sig, exit when exit_sig fires in opposing direction or max_hold.

    Uses numpy searchsorted for O(n log n) instead of O(n*max_hold) event loop.
    """
    N = len(mid)

    # Sub-sample entries to max 50k to keep runtime feasible
    entry_idx_all = np.where(entry_sig != 0)[0]
    entry_idx_all = entry_idx_all[entry_idx_all < N - max_hold - 1]

    MAX_ENTRIES = 50000
    if len(entry_idx_all) > MAX_ENTRIES:
        step = len(entry_idx_all) // MAX_ENTRIES
        entry_idx = entry_idx_all[::step]
    else:
        entry_idx = entry_idx_all

    if len(entry_idx) < MIN_TRADES // 5:
        return {"n": len(entry_idx), "sortino": 0.0, "win_rate": 0.0, "ret_mean": 0.0,
                "edge_ratio": 0.5, "avg_hold": 0.0, "avg_hold_sec": 0.0, "signal_exit_pct": 0.0}

    directions = entry_sig[entry_idx].astype(np.float32)

    # Build "next exit" lookup using searchsorted
    exit_long  = np.where(exit_sig == 1)[0]    # exit fires long (used when short)
    exit_short = np.where(exit_sig == -1)[0]   # exit fires short (used when long)

    # For each entry: find first opposing exit within max_hold
    hold_arr = np.full(len(entry_idx), max_hold, dtype=np.int32)
    sig_exit_flag = np.zeros(len(entry_idx), dtype=np.int8)

    for k, (ei, d) in enumerate(zip(entry_idx, directions)):
        if d > 0:  # long trade: exit on short signal
            idx_in_arr = np.searchsorted(exit_short, ei + 1)
            if idx_in_arr < len(exit_short):
                dist = exit_short[idx_in_arr] - ei
                if dist <= max_hold:
                    hold_arr[k] = dist
                    sig_exit_flag[k] = 1
        else:      # short trade: exit on long signal
            idx_in_arr = np.searchsorted(exit_long, ei + 1)
            if idx_in_arr < len(exit_long):
                dist = exit_long[idx_in_arr] - ei
                if dist <= max_hold:
                    hold_arr[k] = dist
                    sig_exit_flag[k] = 1

    # Compute returns
    exit_idx = np.minimum(entry_idx + hold_arr, N - 1)
    exit_prices  = mid[exit_idx]
    entry_prices = mid[entry_idx]
    rets = (exit_prices - entry_prices) / TICK_SIZE * directions - tc

    # MFE/MAE via vectorized future price scan (sample up to 5000)
    sample_n = min(5000, len(entry_idx))
    s_idx = np.random.choice(len(entry_idx), sample_n, replace=False)
    mfe_arr = np.zeros(sample_n, dtype=np.float32)
    mae_arr = np.zeros(sample_n, dtype=np.float32)

    # Use a fixed horizon for MFE/MAE (300 bars = 30s)
    H_mfemae = min(300, max_hold)
    chunk_size = 1000
    for ci in range(0, sample_n, chunk_size):
        ce = min(ci + chunk_size, sample_n)
        idx_c = entry_idx[s_idx[ci:ce]]
        dir_c = directions[s_idx[ci:ce]]
        fut_idx = idx_c[:, None] + np.arange(1, H_mfemae + 1)[None, :]
        fut_idx = np.clip(fut_idx, 0, N - 1)
        fut_p = mid[fut_idx]
        ep = mid[idx_c][:, None]
        fav = (fut_p - ep) / TICK_SIZE * dir_c[:, None]
        mfe_arr[ci:ce] = np.max(fav, axis=1)
        mae_arr[ci:ce] = -np.min(fav, axis=1)

    mae_arr = np.maximum(mae_arr, 0.0)
    mfe_mean = float(np.mean(mfe_arr))
    mae_mean = float(np.mean(mae_arr))
    edge_ratio = mfe_mean / (mfe_mean + mae_mean + 1e-9)

    neg_rets = rets[rets < 0]
    ds = float(np.sqrt(np.mean(neg_rets**2))) if len(neg_rets) > 0 else 1e-9
    sortino = float(np.mean(rets) / ds) if ds > 0 else 0.0
    sig_exit_pct = float(np.mean(sig_exit_flag))

    return {
        "n":               int(len(entry_idx)),
        "n_total_signals": int(len(entry_idx_all)),
        "sortino":         sortino,
        "win_rate":        float(np.mean(rets > 0)),
        "ret_mean":        float(np.mean(rets)),
        "edge_ratio":      float(edge_ratio),
        "mfe_mean":        float(mfe_mean),
        "mae_mean":        float(mae_mean),
        "avg_hold":        float(np.mean(hold_arr)),
        "avg_hold_sec":    float(np.mean(hold_arr) / 10),
        "signal_exit_pct": float(sig_exit_pct),
    }


def backtest_fixed_horizon(signal, mid, horizon_bars, tc=TC_TICKS):
    """Simple fixed-horizon backtest for baseline comparison."""
    N = len(mid)
    H = horizon_bars
    entry_idx = np.where(signal != 0)[0]
    entry_idx = entry_idx[entry_idx < N - H - 1]

    if len(entry_idx) < MIN_TRADES // 5:
        return {"n": 0, "sortino": 0.0, "win_rate": 0.0, "ret_mean": 0.0}

    directions = signal[entry_idx].astype(np.float32)
    exit_prices = mid[entry_idx + H]
    entry_prices = mid[entry_idx]
    rets = (exit_prices - entry_prices) / TICK_SIZE * directions - tc

    neg = rets[rets < 0]
    ds  = float(np.sqrt(np.mean(neg**2))) if len(neg) > 0 else 1e-9
    sortino = float(np.mean(rets) / ds) if ds > 0 else 0.0

    return {
        "n":        int(len(entry_idx)),
        "sortino":  sortino,
        "win_rate": float(np.mean(rets > 0)),
        "ret_mean": float(np.mean(rets)),
    }


def backtest_regime_conditional(signal, mid, vol_pct, vol_threshold, horizon_bars, tc=TC_TICKS):
    """Only trade when vol_pct > threshold (high vol regime)."""
    regime_mask = (vol_pct > vol_threshold).astype(np.int8)
    filtered_sig = np.where(regime_mask == 1, signal, 0).astype(np.int8)
    return backtest_fixed_horizon(filtered_sig, mid, horizon_bars, tc)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log("=" * 70)
    log("Signal Combination Entry/Exit Pair Tester")
    log("=" * 70)

    # Load data
    tensors, mid, dates = load_book_data()
    N = len(mid)

    # Load CNN predictions
    cnn_preds = load_cnn_predictions(N)

    # Compute signals
    log("Computing signals...")
    all_sigs = compute_signals(tensors, mid, cnn_preds)
    vol_pct  = all_sigs.pop("_vol_pct")

    signal_names = list(all_sigs.keys())
    for name, sig in all_sigs.items():
        n_active = int(np.sum(sig != 0))
        log(f"  {name:15s}: {n_active:8,} active ({100*n_active/N:.2f}%)")

    results = {"meta": {"total_bars": N, "days": len(dates)}, "combos": {}, "baselines": {}, "regime": {}}

    # ── 1. BASELINE: Each signal at each fixed horizon ──────────────────────
    log("\n[1/3] Baseline: each signal × fixed horizon...")
    for sig_name, sig in all_sigs.items():
        results["baselines"][sig_name] = {}
        for hz_name, hz_bars in HORIZONS.items():
            r = backtest_fixed_horizon(sig, mid, hz_bars)
            results["baselines"][sig_name][hz_name] = r
            log(f"  baseline {sig_name:12s} @{hz_name:5s}: n={r['n']:5,}  sort={r['sortino']:.3f}  wr={r['win_rate']:.2%}")

    # ── 2. ENTRY/EXIT COMBOS ────────────────────────────────────────────────
    log("\n[2/3] Entry/Exit signal combination pairs...")
    combo_count = 0
    for entry_name, exit_name in product(signal_names, repeat=2):
        combo_key = f"{entry_name}__exit__{exit_name}"
        entry_sig = all_sigs[entry_name]
        exit_sig  = all_sigs[exit_name]

        r = backtest_entry_exit(entry_sig, exit_sig, mid)
        results["combos"][combo_key] = r
        combo_count += 1

        if r["n"] > MIN_TRADES // 5:
            log(f"  COMBO {entry_name:12s} → exit:{exit_name:12s}: "
                f"n={r['n']:4,}  sort={r['sortino']:.3f}  wr={r['win_rate']:.2%}  "
                f"sig_exit={r['signal_exit_pct']:.0%}  hold={r['avg_hold_sec']:.0f}s")

    log(f"  Total combos tested: {combo_count}")

    # ── 3. REGIME-CONDITIONAL ───────────────────────────────────────────────
    log("\n[3/3] Regime-conditional (signal only when high-vol)...")
    for sig_name, sig in all_sigs.items():
        results["regime"][sig_name] = {}
        for vol_thresh in [0.5, 0.7, 0.85]:
            for hz_name, hz_bars in [("30s", 300), ("1min", 600)]:
                key = f"vol>{vol_thresh:.0%}_{hz_name}"
                r = backtest_regime_conditional(sig, mid, vol_pct, vol_thresh, hz_bars)
                results["regime"][sig_name][key] = r
                if r["n"] > 50:
                    log(f"  regime {sig_name:12s} vol>{vol_thresh:.0%} @{hz_name}: "
                        f"n={r['n']:4,}  sort={r['sortino']:.3f}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to: {OUT_JSON}")

    # ── Generate Report ──────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 90)
    lines.append("SIGNAL COMBINATION ENTRY/EXIT REPORT")
    lines.append(f"Data: {N:,} bars, {len(dates)} days")
    lines.append("=" * 90)

    # Best combos by Sortino
    lines.append("\n--- TOP 20 ENTRY/EXIT COMBOS (by Sortino, min 100 trades) ---\n")
    valid_combos = [(k, v) for k, v in results["combos"].items() if v.get("n", 0) >= MIN_TRADES // 2]
    top_combos = sorted(valid_combos, key=lambda x: x[1]["sortino"], reverse=True)[:20]

    if top_combos:
        lines.append(f"{'Entry':15s}  {'Exit':15s}  {'n':6s}  {'Sortino':8s}  "
                    f"{'WinRate':8s}  {'SigExit':8s}  {'AvgHold':8s}  {'EdgeRatio':10s}")
        lines.append("-" * 85)
        for k, v in top_combos:
            parts = k.split("__exit__")
            entry_n, exit_n = parts[0], parts[1]
            lines.append(
                f"{entry_n:15s}  {exit_n:15s}  {v['n']:6,}  {v['sortino']:8.3f}  "
                f"{v['win_rate']:8.2%}  {v['signal_exit_pct']:8.0%}  "
                f"{v['avg_hold_sec']:6.0f}s   {v['edge_ratio']:10.4f}"
            )
    else:
        lines.append("  No valid combos found (all below min trade count)")

    # Baseline comparison
    lines.append("\n--- BASELINE (single signal × fixed horizon) ---\n")
    lines.append(f"{'Signal':15s}  {'Horizon':8s}  {'n':6s}  {'Sortino':8s}  {'WinRate':8s}  {'RetMean':8s}")
    lines.append("-" * 65)
    for sig_name, hz_results in results["baselines"].items():
        for hz_name, r in hz_results.items():
            if r["n"] > 0:
                lines.append(f"{sig_name:15s}  {hz_name:8s}  {r['n']:6,}  {r['sortino']:8.3f}  "
                            f"{r['win_rate']:8.2%}  {r['ret_mean']:8.4f}")

    # Regime analysis
    lines.append("\n--- REGIME-CONDITIONAL (high-vol filter) ---\n")
    lines.append(f"{'Signal':15s}  {'Condition':20s}  {'n':6s}  {'Sortino':8s}  {'WinRate':8s}")
    lines.append("-" * 65)
    for sig_name, regime_results in results["regime"].items():
        for cond_name, r in regime_results.items():
            if r.get("n", 0) > 50:
                lines.append(f"{sig_name:15s}  {cond_name:20s}  {r['n']:6,}  "
                            f"{r['sortino']:8.3f}  {r['win_rate']:8.2%}")

    # Key findings
    lines.append("\n--- KEY FINDINGS ---")
    if top_combos:
        best_k, best_v = top_combos[0]
        parts = best_k.split("__exit__")
        lines.append(f"\nBest combo: {parts[0]} entry → {parts[1]} exit")
        lines.append(f"  Sortino={best_v['sortino']:.3f}, WinRate={best_v['win_rate']:.2%}, "
                    f"n={best_v['n']:,}, AvgHold={best_v['avg_hold_sec']:.0f}s")
        lines.append(f"  Signal exit triggered {best_v['signal_exit_pct']:.0%} of trades")
        lines.append(f"  Edge ratio: {best_v['edge_ratio']:.4f}")

        # Compare best combo vs best baseline
        best_baseline_sortino = max(
            (r["sortino"] for hz_results in results["baselines"].values()
             for r in hz_results.values() if r["n"] > 50),
            default=0.0
        )
        if best_v["sortino"] > best_baseline_sortino:
            improvement = best_v["sortino"] - best_baseline_sortino
            lines.append(f"\nCOMBO IMPROVEMENT: {improvement:+.3f} Sortino over best baseline ({best_baseline_sortino:.3f})")
        else:
            lines.append(f"\nNo improvement over fixed-horizon baseline (best={best_baseline_sortino:.3f})")

    elapsed = time.time() - t0
    lines.append(f"\n\nCompleted in {elapsed:.0f}s at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(OUT_TXT, "w") as f:
        f.write(report_text)
    log(f"Report saved to: {OUT_TXT}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
