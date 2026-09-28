#!/usr/bin/env python3
"""
Tick Path Analysis — Signal vs Random
=======================================
For every model signal, track what ACTUALLY happens to price over the next
1s, 5s, 10s, 30s, 1min, 5min horizons. Compare signal-triggered entries to
random entries at same TP/SL. Test confidence thresholds. Symmetric TP=SL=10 test.

Models analyzed:
  1. Wider CNN (13 days book_cache overlap)
  2. OOT CNN predictions (204 days)

Deployed on: Jupiter (or any node with book cache + predictions)
Output: data/processed/tick_path_results/tick_path_analysis.json
"""

import os
import sys
import glob
import json
import time
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─── Auto-detect data root ───────────────────────────────────────────────────
POSSIBLE_ROOTS = [
    r"C:\Users\Footb\Documents\Github\Lvl3Quant",   # Neptune
    r"C:\Users\claude\Lvl3Quant",                    # Razer
    r"C:\Users\nick\Lvl3Quant",                      # Uranus
    "/home/jupiter/Lvl3Quant",                        # Jupiter (WSL/Linux)
]
LVL3_ROOT = None
for r in POSSIBLE_ROOTS:
    if os.path.isdir(r):
        LVL3_ROOT = r
        break
if LVL3_ROOT is None:
    print("[ERROR] Cannot find Lvl3Quant root!")
    sys.exit(1)
print(f"[ROOT] {LVL3_ROOT}")

DATA_DIR  = os.path.join(LVL3_ROOT, "data", "processed")
OUT_DIR   = os.path.join(DATA_DIR, "tick_path_results")
os.makedirs(OUT_DIR, exist_ok=True)

BOOK_CACHE_DIR = os.path.join(DATA_DIR, "dl_book_cache")

# Prediction directories to analyze
PRED_CONFIGS = [
    {
        "name": "wider_cnn",
        "dir": os.path.join(DATA_DIR, "hybrid_vs_wider_preds", "wider"),
        "pattern": "*_preds.npz",
        "date_fmt": lambda f: os.path.basename(f).split("_")[0],  # 2025-07-22_preds.npz
        "book_pattern": lambda date: os.path.join(BOOK_CACHE_DIR, f"{date}_book_tensors.npz"),
    },
    {
        "name": "oot_cnn",
        "dir": os.path.join(DATA_DIR, "cnn_oot_sim_predictions"),
        "pattern": "*_vol50_morning_afternoon.npz",  # use vol50 as representative
        "date_fmt": lambda f: os.path.basename(f).split("_")[0],
        "book_pattern": lambda date: os.path.join(BOOK_CACHE_DIR, f"{date}_book_tensors.npz"),
    },
]

# Analysis config
TICK_SIZE = 0.25      # ES tick = $12.50
BAR_MS    = 100       # 100ms per bar
HORIZONS  = {
    "1s":   10,
    "5s":   50,
    "10s":  100,
    "30s":  300,
    "1min": 600,
    "5min": 3000,
}
CONF_PCTS = [50, 75, 90, 95, 99]    # top X% confidence thresholds
TP_SL_SYMMETRIC = 10                 # ticks for symmetric TP=SL test
N_RANDOM_SEED   = 42
MAX_DAYS        = 50                 # cap for memory


# ─── Helper: compute MFE/MAE/outcomes for a set of signal bars ──────────────

def analyze_signal_set(mid_prices, signal_bars, directions, label="signals"):
    """
    For each signal bar index + direction (+1 long, -1 short):
      - Track price path at each horizon
      - Compute MFE (max favorable excursion), MAE (max adverse excursion)
      - Hit rate for symmetric TP=SL=10 ticks
    """
    N = len(mid_prices)
    max_h = max(HORIZONS.values())

    results_by_horizon = {h: [] for h in HORIZONS}
    mfe_list, mae_list = [], []
    sym_tp_hits, sym_sl_hits, sym_timeouts = 0, 0, 0
    n_signals = len(signal_bars)

    for idx, direction in zip(signal_bars, directions):
        entry_price = mid_prices[idx]
        # Price path for next max_h bars
        end_idx = min(idx + max_h, N - 1)
        if end_idx <= idx:
            continue
        path = (mid_prices[idx+1:end_idx+1] - entry_price) * direction / TICK_SIZE

        # Horizon returns
        for hname, hbars in HORIZONS.items():
            if len(path) >= hbars:
                results_by_horizon[hname].append(path[hbars - 1])

        # MFE/MAE over full path
        if len(path) > 0:
            mfe_list.append(float(np.nanmax(path)))
            mae_list.append(float(np.nanmin(path)))

        # Symmetric TP=SL=10 test
        tp_tick = TP_SL_SYMMETRIC
        sl_tick = -TP_SL_SYMMETRIC
        hit = False
        for pnl in path:
            if pnl >= tp_tick:
                sym_tp_hits += 1
                hit = True
                break
            elif pnl <= sl_tick:
                sym_sl_hits += 1
                hit = True
                break
        if not hit:
            sym_timeouts += 1

    summary = {
        "n_signals": n_signals,
        "horizons": {},
        "mfe_mean": float(np.mean(mfe_list)) if mfe_list else None,
        "mae_mean": float(np.mean(mae_list)) if mae_list else None,
        "mfe_p50": float(np.percentile(mfe_list, 50)) if mfe_list else None,
        "mae_p50": float(np.percentile(mae_list, 50)) if mae_list else None,
        "sym_tp_rate": sym_tp_hits / n_signals if n_signals > 0 else 0,
        "sym_sl_rate": sym_sl_hits / n_signals if n_signals > 0 else 0,
        "sym_timeout_rate": sym_timeouts / n_signals if n_signals > 0 else 0,
    }
    for hname, vals in results_by_horizon.items():
        if vals:
            arr = np.array(vals)
            summary["horizons"][hname] = {
                "mean_pnl_ticks": float(np.mean(arr)),
                "median_pnl_ticks": float(np.median(arr)),
                "win_rate": float(np.mean(arr > 0)),
                "n": len(arr),
            }

    return summary


# ─── Main analysis per model ─────────────────────────────────────────────────

def run_model_analysis(config):
    name     = config["name"]
    pred_dir = config["dir"]
    if not os.path.isdir(pred_dir):
        print(f"[SKIP] {name}: dir not found {pred_dir}")
        return None

    pred_files = sorted(glob.glob(os.path.join(pred_dir, config["pattern"])))
    print(f"\n[{name}] Found {len(pred_files)} prediction files")

    all_results = []
    days_processed = 0

    for pred_file in pred_files:
        if days_processed >= MAX_DAYS:
            break

        date_str = config["date_fmt"](pred_file)
        book_file = config["book_pattern"](date_str)

        if not os.path.exists(book_file):
            continue

        try:
            pred_data = np.load(pred_file, allow_pickle=True)
            book_data = np.load(book_file, allow_pickle=True)
        except Exception as e:
            print(f"  [WARN] Load error {date_str}: {e}")
            continue

        preds = pred_data["predictions"].astype(np.float32)
        mid_prices = book_data["mid_prices"].astype(np.float64)

        # Align lengths
        n = min(len(preds), len(mid_prices))
        preds = preds[:n]
        mid_prices = mid_prices[:n]

        # Filter zero preds (inactive bars)
        active_mask = preds != 0
        n_active = active_mask.sum()
        if n_active < 100:
            continue

        active_preds = preds[active_mask]
        active_bars = np.where(active_mask)[0]

        # Ensure buffer for max horizon
        max_h = max(HORIZONS.values())
        valid = active_bars < (n - max_h)
        if valid.sum() < 50:
            continue
        active_preds = active_preds[valid]
        active_bars = active_bars[valid]
        directions = np.sign(active_preds)

        day_result = {
            "date": date_str,
            "n_active_bars": int(n_active),
            "n_valid_signals": int(len(active_bars)),
            "pred_mean": float(np.mean(active_preds)),
            "pred_std": float(np.std(active_preds)),
        }

        # ── 1. All signals (no threshold) ──
        day_result["all_signals"] = analyze_signal_set(
            mid_prices, active_bars, directions, "all"
        )

        # ── 2. Confidence threshold tiers ──
        abs_preds = np.abs(active_preds)
        thresholds = {}
        for pct in CONF_PCTS:
            thresh = np.percentile(abs_preds, pct)
            mask_hi = abs_preds >= thresh
            if mask_hi.sum() < 10:
                continue
            thresholds[f"top_{100-pct}pct"] = analyze_signal_set(
                mid_prices,
                active_bars[mask_hi],
                directions[mask_hi],
                f"top_{100-pct}pct"
            )
        day_result["confidence_tiers"] = thresholds

        # ── 3. Random baseline (same count, random direction) ──
        rng = np.random.RandomState(N_RANDOM_SEED)
        rand_bars = rng.choice(len(mid_prices) - max_h, size=min(len(active_bars), 2000), replace=False)
        rand_dirs = rng.choice([-1.0, 1.0], size=len(rand_bars))
        day_result["random_baseline"] = analyze_signal_set(
            mid_prices, rand_bars, rand_dirs, "random"
        )

        all_results.append(day_result)
        days_processed += 1
        print(f"  [{date_str}] {len(active_bars)} valid signals, "
              f"pred_std={active_preds.std():.3f}, "
              f"sym_tp={day_result['all_signals']['sym_tp_rate']:.3f}")

    if not all_results:
        print(f"[{name}] No results produced")
        return None

    # ── Aggregate across all days ──
    agg = aggregate_results(all_results)
    print(f"\n[{name}] AGGREGATE SUMMARY:")
    print_summary(agg)

    return {"model": name, "days": len(all_results), "per_day": all_results, "aggregate": agg}


def aggregate_results(day_results):
    """Combine per-day stats into overall summary."""
    def agg_key(key_path):
        vals = []
        for day in day_results:
            obj = day
            try:
                for k in key_path:
                    obj = obj[k]
                if obj is not None:
                    vals.append(float(obj))
            except (KeyError, TypeError):
                pass
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    agg = {
        "total_signals": sum(d.get("n_valid_signals", 0) for d in day_results),
        "days_analyzed": len(day_results),
        "all_signals": {},
        "confidence_tiers": {},
        "random_baseline": {},
        "edge_vs_random": {},  # signal - random
    }

    for hname in HORIZONS:
        for tier in ["all_signals", "random_baseline"]:
            sig_mean = agg_key([tier, "horizons", hname, "mean_pnl_ticks"])
            win_rate = agg_key([tier, "horizons", hname, "win_rate"])
            if tier not in agg:
                agg[tier] = {}
            if hname not in agg[tier]:
                agg[tier][hname] = {}
            agg[tier][hname]["mean_pnl_ticks"] = sig_mean
            agg[tier][hname]["win_rate"] = win_rate

        # Edge = signal - random
        if hname not in agg["edge_vs_random"]:
            agg["edge_vs_random"][hname] = {}
        s = agg["all_signals"].get(hname, {}).get("mean_pnl_ticks")
        r = agg["random_baseline"].get(hname, {}).get("mean_pnl_ticks")
        if s and r and s.get("mean") is not None and r.get("mean") is not None:
            agg["edge_vs_random"][hname]["mean_edge_ticks"] = s["mean"] - r["mean"]

    agg["sym_tp_rate"]      = agg_key(["all_signals", "sym_tp_rate"])
    agg["sym_sl_rate"]      = agg_key(["all_signals", "sym_sl_rate"])
    agg["random_tp_rate"]   = agg_key(["random_baseline", "sym_tp_rate"])
    agg["mfe_mean"]         = agg_key(["all_signals", "mfe_mean"])
    agg["mae_mean"]         = agg_key(["all_signals", "mae_mean"])

    for pct in CONF_PCTS:
        key = f"top_{100-pct}pct"
        tier_agg = {}
        for hname in HORIZONS:
            sig = agg_key(["confidence_tiers", key, "horizons", hname, "mean_pnl_ticks"])
            wr  = agg_key(["confidence_tiers", key, "horizons", hname, "win_rate"])
            if sig:
                tier_agg[hname] = {"mean_pnl_ticks": sig, "win_rate": wr}
        if tier_agg:
            agg["confidence_tiers"][key] = tier_agg

    return agg


def print_summary(agg):
    print(f"  Total signals: {agg.get('total_signals', 0):,}")
    print(f"  Days: {agg.get('days_analyzed', 0)}")
    print(f"  Sym TP=SL=10 → TP rate: {agg.get('sym_tp_rate', {}).get('mean', 0):.3f}  "
          f"(random: {agg.get('random_tp_rate', {}).get('mean', 0):.3f})")
    print(f"  MFE avg: {(agg.get('mfe_mean') or {}).get('mean', 0):.2f} ticks  "
          f"MAE avg: {(agg.get('mae_mean') or {}).get('mean', 0):.2f} ticks")
    print("  Horizon mean PnL (signal vs random):")
    for hname in HORIZONS:
        s = (agg.get("all_signals", {}).get(hname, {}).get("mean_pnl_ticks") or {}).get("mean")
        r = (agg.get("random_baseline", {}).get(hname, {}).get("mean_pnl_ticks") or {}).get("mean")
        edge = agg.get("edge_vs_random", {}).get(hname, {}).get("mean_edge_ticks")
        if s is not None:
            print(f"    {hname:>6}: signal={s:+.4f}  random={r:+.4f}  edge={edge:+.4f}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    print(f"[START] Tick Path Analysis — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[CONFIG] Horizons: {list(HORIZONS.keys())}, TP/SL symmetric: {TP_SL_SYMMETRIC} ticks")

    all_model_results = []

    for config in PRED_CONFIGS:
        result = run_model_analysis(config)
        if result:
            all_model_results.append(result)

    elapsed = time.time() - start_time
    out = {
        "run_time": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "models": all_model_results,
        "config": {
            "horizons_bars": HORIZONS,
            "conf_pcts": CONF_PCTS,
            "tp_sl_symmetric_ticks": TP_SL_SYMMETRIC,
            "bar_ms": BAR_MS,
            "tick_size": TICK_SIZE,
        }
    }

    out_path = os.path.join(OUT_DIR, "tick_path_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[DONE] {elapsed:.1f}s → {out_path}")

    # Print a condensed final table
    print("\n" + "="*60)
    print("FINAL SUMMARY — SIGNAL vs RANDOM EDGE")
    print("="*60)
    for m in all_model_results:
        print(f"\nModel: {m['model']} ({m['days']} days, {m['aggregate'].get('total_signals',0):,} signals)")
        agg = m["aggregate"]
        # Horizon table
        print(f"{'Horizon':>8} {'Sig(ticks)':>12} {'Rand(ticks)':>12} {'Edge':>8} {'WinRate':>8}")
        print("-"*52)
        for hname in HORIZONS:
            s  = (agg.get("all_signals", {}).get(hname, {}).get("mean_pnl_ticks") or {}).get("mean", 0)
            r  = (agg.get("random_baseline", {}).get(hname, {}).get("mean_pnl_ticks") or {}).get("mean", 0)
            wr = (agg.get("all_signals", {}).get(hname, {}).get("win_rate") or {}).get("mean", 0)
            edge = s - r if s and r else 0
            print(f"{hname:>8} {s:>12.4f} {r:>12.4f} {edge:>8.4f} {wr:>8.3f}")
        # Symmetric test
        tp = (agg.get("sym_tp_rate") or {}).get("mean", 0)
        rtp = (agg.get("random_tp_rate") or {}).get("mean", 0)
        print(f"\n  TP=SL=10 TP rate: {tp:.3f}  Random TP rate: {rtp:.3f}  "
              f"→ {'EDGE EXISTS' if tp > rtp + 0.01 else 'NO EDGE'}")
        # Confidence tiers
        print("\n  Confidence tiers (10s horizon mean PnL):")
        for pct in CONF_PCTS:
            key = f"top_{100-pct}pct"
            tier = agg.get("confidence_tiers", {}).get(key, {})
            val = (tier.get("10s", {}).get("mean_pnl_ticks") or {}).get("mean")
            if val is not None:
                print(f"    {key}: {val:+.4f} ticks")

    return out


if __name__ == "__main__":
    result = main()
    sys.exit(0)
