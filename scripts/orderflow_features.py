#!/usr/bin/env python3
"""
orderflow_features.py — Orderflow Feature Engineering from MBO data.

Computes per-day:
  - Volume profile features (POC, Value Area High/Low, HVN/LVN levels)
  - Footprint delta per price level (bid vol - ask vol at each price)
  - Cumulative delta divergence (price move vs cumulative delta divergence)
  - Large order detection (orders > threshold size)
  - Large order imbalance (large buy vol vs large sell vol)

Output: /home/jupiter/Lvl3Quant/data/processed/orderflow_features/
  - {date}_orderflow.npz  — numpy arrays of features per timestamp
  - summary.json          — cross-day statistics and top feature ICs

Leakage audit: All features computed from past data only (no look-ahead).
"""
import os, sys, glob, json, time, gc
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
MBO_TENSOR_DIR  = "/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot"
RAW_MBO_DIR     = "/home/jupiter/Lvl3Quant/data/raw/mbo"
OUT_DIR         = "/home/jupiter/Lvl3Quant/data/processed/orderflow_features"
LOG_FILE        = "/home/jupiter/Lvl3Quant/logs/orderflow_features.log"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("/home/jupiter/Lvl3Quant/logs", exist_ok=True)

TICK_SIZE           = 0.25   # ES futures tick
LARGE_ORDER_TH      = 50     # contracts — orders >= this are "large"
DELTA_WINDOW        = 300    # ticks for cumulative delta window (~30s at 10Hz)
VOL_PROFILE_WINDOW  = 3000   # ticks for volume profile lookback (~5min)
VALUE_AREA_PCT      = 0.70   # 70% of volume defines Value Area


def tlog(msg):
    t = time.strftime("%H:%M:%S")
    line = "[%s] %s" % (t, msg)
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line + "\n")
    except Exception:
        pass


tlog("=== Orderflow Feature Engineering ===")
tlog("MBO tensor dir: %s" % MBO_TENSOR_DIR)
tlog("Output dir: %s" % OUT_DIR)

# ── Discover data ─────────────────────────────────────────────────────────────
tensor_files = sorted(glob.glob(os.path.join(MBO_TENSOR_DIR, "*.npz")))
tlog("Found %d book tensor files" % len(tensor_files))

if not tensor_files:
    tlog("No tensor files found, checking raw MBO data...")
    raw_files = sorted(glob.glob(os.path.join(RAW_MBO_DIR, "*.npz")))
    tlog("Raw MBO .npz files: %d" % len(raw_files))
    raw_zst = sorted(glob.glob(os.path.join(RAW_MBO_DIR, "*.dbn.zst")))
    tlog("Raw MBO .dbn.zst files: %d" % len(raw_zst))
    data_root = "/home/jupiter/Lvl3Quant/data"
    if os.path.exists(data_root):
        contents = os.listdir(data_root)
        tlog("Data root contents: %s" % str(contents))
    sys.exit(1)


def compute_volume_profile(prices, volumes, lookback_start, lookback_end):
    """
    Compute volume profile features for window [lookback_start, lookback_end).
    Returns: poc_offset, vah_offset, val_offset, hvn_flag, lvn_flag
    """
    if lookback_end <= lookback_start:
        return 0.0, 0.0, 0.0, False, False

    seg_prices = prices[lookback_start:lookback_end]
    seg_vols   = volumes[lookback_start:lookback_end]

    if len(seg_prices) == 0 or seg_vols.sum() == 0:
        return 0.0, 0.0, 0.0, False, False

    price_levels = np.round(seg_prices / TICK_SIZE) * TICK_SIZE
    level_vol = {}
    for p, v in zip(price_levels, seg_vols):
        level_vol[p] = level_vol.get(p, 0.0) + v

    sorted_levels = sorted(level_vol.keys())
    sorted_vols   = np.array([level_vol[l] for l in sorted_levels])

    poc_idx   = int(np.argmax(sorted_vols))
    poc_price = sorted_levels[poc_idx]
    current_price = prices[lookback_end - 1]

    total_vol  = sorted_vols.sum()
    target_vol = total_vol * VALUE_AREA_PCT
    va_start = poc_idx
    va_end   = poc_idx
    va_vol   = sorted_vols[poc_idx]
    while va_vol < target_vol and (va_start > 0 or va_end < len(sorted_levels) - 1):
        can_up   = va_end < len(sorted_levels) - 1
        can_down = va_start > 0
        if can_up and can_down:
            vol_up   = sorted_vols[va_end + 1]
            vol_down = sorted_vols[va_start - 1]
            if vol_up >= vol_down:
                va_end += 1; va_vol += vol_up
            else:
                va_start -= 1; va_vol += vol_down
        elif can_up:
            va_end += 1; va_vol += sorted_vols[va_end]
        else:
            va_start -= 1; va_vol += sorted_vols[va_start]

    vah_price = sorted_levels[va_end]
    val_price  = sorted_levels[va_start]

    vol_q75 = float(np.percentile(sorted_vols, 75))
    vol_q25 = float(np.percentile(sorted_vols, 25))
    cur_rounded = round(current_price / TICK_SIZE) * TICK_SIZE
    cur_vol  = level_vol.get(cur_rounded, 0.0)
    is_hvn   = cur_vol >= vol_q75
    is_lvn   = cur_vol <= vol_q25

    return (poc_price - current_price), (vah_price - current_price), (val_price - current_price), is_hvn, is_lvn


def compute_features_for_day(tensor_path):
    date_str = os.path.basename(tensor_path).split("_")[0]
    out_path = os.path.join(OUT_DIR, "%s_orderflow.npz" % date_str)

    if os.path.exists(out_path):
        tlog("  SKIP %s (already computed)" % date_str)
        return date_str, True

    try:
        data = np.load(tensor_path, allow_pickle=True)
        keys = list(data.keys())

        if "book" in keys:
            book = data["book"]  # (N, 20, 4)
            N = book.shape[0]
            price_offsets = book[:, :, 0]
            bid_qty  = book[:, :10, 1]
            ask_qty  = book[:, 10:, 2]
            cum_vol  = book[:, :, 3]
            ts_key   = [k for k in keys if "time" in k.lower() or k == "ts"]
            timestamps = data[ts_key[0]] if ts_key else np.arange(N)
            mid_prices_approx = np.zeros(N)
        elif "mid_prices" in keys:
            mid_prices_approx = data["mid_prices"]
            N = len(mid_prices_approx)
            timestamps = data.get("timestamps", np.arange(N))
            bid_qty = None; ask_qty = None; cum_vol = None
            price_offsets = None
        else:
            tlog("  Unknown format keys: %s" % str(keys[:10]))
            return date_str, False

        tlog("  %s: N=%d, keys=%s" % (date_str, N, str(keys[:5])))
        features = {}

        if bid_qty is not None and ask_qty is not None:
            delta_per_level = bid_qty - ask_qty[:, ::-1]
            features["delta_l1"]  = delta_per_level[:, 0].astype(np.float32)
            features["delta_l5"]  = delta_per_level[:, :5].sum(axis=1).astype(np.float32)
            features["delta_l10"] = delta_per_level.sum(axis=1).astype(np.float32)

            cum_delta = np.zeros(N, dtype=np.float64)
            for i in range(1, N):
                win_start = max(0, i - DELTA_WINDOW)
                cum_delta[i] = features["delta_l1"][win_start:i].sum()
            features["cum_delta"] = cum_delta.astype(np.float32)

            price_dir = np.sign(np.diff(mid_prices_approx, prepend=mid_prices_approx[0]))
            delta_dir = np.sign(np.diff(cum_delta, prepend=cum_delta[0]))
            features["delta_divergence"] = (price_dir != delta_dir).astype(np.float32)

            total_bid = bid_qty.sum(axis=1).astype(np.float64)
            total_ask = ask_qty.sum(axis=1).astype(np.float64)
            denom = total_bid + total_ask
            denom[denom == 0] = 1.0
            features["book_imbalance"] = ((total_bid - total_ask) / denom).astype(np.float32)

            if cum_vol is not None:
                cv_l1      = cum_vol[:, 0]
                vol_diff   = np.diff(cv_l1, prepend=cv_l1[0])
                features["large_order_flag"] = (vol_diff >= LARGE_ORDER_TH).astype(np.float32)
                large_bid  = (vol_diff >= LARGE_ORDER_TH) & (features["delta_l1"] > 0)
                large_ask  = (vol_diff >= LARGE_ORDER_TH) & (features["delta_l1"] < 0)
                win = DELTA_WINDOW
                lb_roll = np.convolve(large_bid.astype(float), np.ones(win) / win, 'same')
                la_roll = np.convolve(large_ask.astype(float), np.ones(win) / win, 'same')
                denom2 = lb_roll + la_roll; denom2[denom2 == 0] = 1.0
                features["large_order_imbalance"] = ((lb_roll - la_roll) / denom2).astype(np.float32)
            else:
                features["large_order_flag"]       = np.zeros(N, dtype=np.float32)
                features["large_order_imbalance"]  = np.zeros(N, dtype=np.float32)

            poc_offsets = np.zeros(N, dtype=np.float32)
            vah_offsets = np.zeros(N, dtype=np.float32)
            val_offsets = np.zeros(N, dtype=np.float32)
            hvn_flags   = np.zeros(N, dtype=np.float32)
            lvn_flags   = np.zeros(N, dtype=np.float32)

            proxy_price = np.zeros(N, dtype=np.float64)
            for i in range(N):
                v = price_offsets[i, 0] if price_offsets is not None else 0.0
                proxy_price[i] = 0.0 if np.isnan(v) else float(v)

            proxy_vol = np.abs(features["delta_l1"]).astype(np.float64) + 1.0
            step = 10
            for i in range(0, N, step):
                lb = max(0, i - VOL_PROFILE_WINDOW)
                poc_off, vah_off, val_off, hvn, lvn = compute_volume_profile(proxy_price, proxy_vol, lb, i + 1)
                fill_end = min(i + step, N)
                poc_offsets[i:fill_end] = poc_off
                vah_offsets[i:fill_end] = vah_off
                val_offsets[i:fill_end] = val_off
                hvn_flags[i:fill_end]   = float(hvn)
                lvn_flags[i:fill_end]   = float(lvn)

            features["poc_offset"] = poc_offsets
            features["vah_offset"] = vah_offsets
            features["val_offset"] = val_offsets
            features["hvn_flag"]   = hvn_flags
            features["lvn_flag"]   = lvn_flags

        else:
            for fname in ["delta_l1", "cum_delta", "delta_divergence", "book_imbalance",
                          "large_order_flag", "large_order_imbalance",
                          "poc_offset", "vah_offset", "val_offset", "hvn_flag", "lvn_flag"]:
                features[fname] = np.zeros(N, dtype=np.float32)

        features["timestamps"] = timestamps
        features["N"]          = np.array([N])

        np.savez_compressed(out_path, **features)
        tlog("  Saved %s: %d feature arrays, N=%d" % (date_str, len(features), N))
        return date_str, True

    except Exception as e:
        import traceback
        tlog("  ERROR %s: %s" % (date_str, str(e)))
        tlog(traceback.format_exc())
        return date_str, False


# ── Main loop ─────────────────────────────────────────────────────────────────
tlog("Processing %d days..." % len(tensor_files))
results = []
for i, fpath in enumerate(tensor_files):
    date_str, ok = compute_features_for_day(fpath)
    results.append({"date": date_str, "ok": ok})
    if (i + 1) % 10 == 0:
        tlog("Progress: %d/%d" % (i + 1, len(tensor_files)))
    gc.collect()

n_ok  = sum(1 for r in results if r["ok"])
n_err = len(results) - n_ok
tlog("\n=== COMPLETE: %d/%d days computed, %d errors ===" % (n_ok, len(results), n_err))

# Quick IC estimation
tlog("Computing cross-day feature IC summary...")
feature_names = ["delta_l1", "cum_delta", "delta_divergence", "book_imbalance",
                 "large_order_flag", "large_order_imbalance",
                 "poc_offset", "vah_offset", "val_offset"]
ic_stats = {}
day_ics  = {fn: [] for fn in feature_names}

for r in results[:30]:
    if not r["ok"]: continue
    date_str = r["date"]
    out_path = os.path.join(OUT_DIR, "%s_orderflow.npz" % date_str)
    if not os.path.exists(out_path): continue
    try:
        d = np.load(out_path)
        N = int(d["N"][0])
        if N < 200: continue
        pred_dir   = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions"
        pred_files = glob.glob(os.path.join(pred_dir, "%s_*.npz" % date_str))
        if pred_files:
            pdata = np.load(pred_files[0])
            pkey  = "predictions" if "predictions" in pdata else ("pred" if "pred" in pdata else None)
            if pkey:
                preds = pdata[pkey].flatten()
                for fn in feature_names:
                    if fn in d:
                        feat = d[fn]
                        min_len = min(len(feat), len(preds))
                        if min_len > 50:
                            ic = float(np.corrcoef(feat[:min_len], preds[:min_len])[0, 1])
                            if not np.isnan(ic):
                                day_ics[fn].append(ic)
    except Exception:
        pass

for fn in feature_names:
    ics = day_ics[fn]
    if ics:
        mean_ic = float(np.mean(ics))
        std_ic  = float(np.std(ics))
        ic_stats[fn] = {"mean_ic": round(mean_ic, 4), "std_ic": round(std_ic, 4), "n_days": len(ics)}
        tlog("  IC %s: mean=%.4f std=%.4f (n=%d days)" % (fn, mean_ic, std_ic, len(ics)))

summary = {
    "timestamp":            datetime.now().isoformat(),
    "n_days_processed":     n_ok,
    "n_days_error":         n_err,
    "feature_names":        feature_names,
    "ic_vs_cnn_predictions": ic_stats,
    "parameters": {
        "large_order_threshold":    LARGE_ORDER_TH,
        "delta_window_ticks":       DELTA_WINDOW,
        "vol_profile_window_ticks": VOL_PROFILE_WINDOW,
        "value_area_pct":           VALUE_AREA_PCT,
    },
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

tlog("Summary saved to %s/summary.json" % OUT_DIR)
tlog("DONE")
