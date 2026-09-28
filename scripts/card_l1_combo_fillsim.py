#!/usr/bin/env python3
"""
card_l1_combo_fillsim.py -- L1 imbalance filter on per-card CNN signals (Jupiter).

Tests: per-card CNN predictions (already conviction-filtered) vs L1-direction-agree subset.
This is the CORRECT test: card predictions have realistic signal frequency (~50-500/day)
vs raw CNN output which fires 100K+/day.

Cards tested:
  C1: book_predstdExit_conv1.5_vol50  (median ~2K/day, higher frequency)
  C4: book_predstdExit_conv2.0_vol70  (median ~200/day, more realistic)
  C7: smooth_smoothExit_conv1.5_ethr0.0_vol70 (check if nonzero)
  C2.5_vol70: book_predstdExit_conv2.5_vol70 (most selective)

Data (Jupiter):
  - CNN preds: /home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions/
  - Book tensors: /home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot/
  - MBO data: /home/jupiter/Lvl3Quant/data/raw/mbo/
  - fill_sim_cli: /home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli
"""

import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("card_l1")

BASE       = Path("/home/jupiter/Lvl3Quant")
PRED_DIR   = BASE / "data" / "processed" / "cnn_wf_stacked_predictions"
TENSOR_DIR = BASE / "data" / "processed" / "dl_book_cache_oot"
MBO_DIR    = BASE / "data" / "raw" / "mbo"
FILL_SIM   = BASE / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
OUT_FILE   = BASE / "data" / "processed" / "card_l1_combo_fillsim_results.json"

OOT_START  = date(2025, 12, 1)
OOT_END    = date(2026, 3, 6)
WORKERS    = 4  # conservative to avoid memory pressure

# Cards to test
CARDS = {
    "c1_conv1.5_vol50": "book_predstdExit_conv1.5_vol50",
    "c4_conv2.0_vol70": "book_predstdExit_conv2.0_vol70",
    "c25_conv2.5_vol70":"book_predstdExit_conv2.5_vol70",
    "c7_smooth_vol70":  "smooth_smoothExit_conv1.5_ethr0.0_vol70",
}

# Fill sim configs
FILL_CONFIGS = [
    ("chase_tp8_sl20",  ["--chase-entry", "--take-profit-ticks", "8",  "--stop-loss-ticks", "20", "--prime-hours"]),
    ("chase_tp12_sl30", ["--chase-entry", "--take-profit-ticks", "12", "--stop-loss-ticks", "30", "--prime-hours"]),
    ("passive_tp8",     ["--take-profit-ticks", "8",  "--stop-loss-ticks", "20", "--prime-hours"]),
]

L1_THRESH = 0.02


def compute_l1_agree_pred(book_tensors, pred):
    """Return pred array with non-L1-agree signals zeroed out."""
    bid_qty = book_tensors[:, 0, 1].astype(np.float32)
    ask_qty = book_tensors[:, 0, 2].astype(np.float32)
    denom   = bid_qty + ask_qty
    l1_imb  = np.where(denom > 0, (bid_qty - ask_qty) / denom, 0.0)
    l1_dem  = l1_imb - float(np.nanmedian(l1_imb))

    active_mask = pred != 0.0
    agree_mask  = active_mask & (np.sign(pred) == np.sign(l1_dem)) & (np.abs(l1_dem) > L1_THRESH)

    pred_agree = np.where(agree_mask, pred, 0.0).astype(np.float32)
    return pred_agree, int(active_mask.sum()), int(agree_mask.sum())


def run_fillsim(mbo_path, pred_npz, out_json, extra_args):
    cmd = [
        str(FILL_SIM),
        "--mbo-file",    str(mbo_path),
        "--predictions", str(pred_npz),
        "--output",      str(out_json),
        "--latency-ms",  "5",
        "--quiet",
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return {"error": result.stderr[:200]}
    try:
        with open(out_json) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def process_day(args):
    ds, tensor_path, card_pred_paths, mbo_path = args
    date_str = ds.strftime("%Y-%m-%d")
    # Load data within worker process
    try:
        book = np.load(tensor_path)["book_tensors"]
    except Exception as e:
        return {"date": date_str, "error": str(e)}
    card_preds = {}
    for card_key, pred_path in card_pred_paths.items():
        try:
            card_preds[card_key] = np.load(pred_path)["predictions"].astype(np.float32)
        except Exception:
            pass
    results = {"date": date_str}

    with tempfile.TemporaryDirectory() as tmpdir:
        for card_key, pred in card_preds.items():
            pred_agree, n_base, n_agree = compute_l1_agree_pred(book, pred)

            results[f"{card_key}_n_base"]  = n_base
            results[f"{card_key}_n_agree"] = n_agree

            base_npz  = Path(tmpdir) / f"{date_str}_{card_key}_base.npz"
            agree_npz = Path(tmpdir) / f"{date_str}_{card_key}_agree.npz"
            np.savez(base_npz,  predictions=pred)
            np.savez(agree_npz, predictions=pred_agree)

            for cfg_name, cfg_args in FILL_CONFIGS:
                for variant, npz_path in [("base", base_npz), ("agree", agree_npz)]:
                    out_json = Path(tmpdir) / f"{date_str}_{card_key}_{cfg_name}_{variant}.json"
                    r = run_fillsim(mbo_path, npz_path, out_json, cfg_args)
                    key = f"{card_key}_{variant}_{cfg_name}"
                    results[key] = {
                        "pnl":   r.get("total_pnl_dollars"),
                        "n":     r.get("total_trades", r.get("total_filled")),
                        "wr":    r.get("win_rate"),
                        "fill": r.get("total_filled", 0) / max(r.get("total_signals", 1), 1),
                        "error": r.get("error"),
                    }

    return results


def main():
    # Discover available dates
    days_args = []
    d = OOT_START
    while d <= OOT_END:
        ds       = d.strftime("%Y-%m-%d")
        mbo_path = MBO_DIR / f"glbx-mdp3-{d.strftime('%Y%m%d')}.mbo.dbn.zst"
        tensor_p = TENSOR_DIR / f"{ds}_book_tensors.npz"

        if mbo_path.exists() and tensor_p.exists():
            # Collect paths (DON'T load data here — let workers load in parallel)
            card_pred_paths = {}
            for card_key, suffix in CARDS.items():
                pred_path = PRED_DIR / f"{ds}_{suffix}.npz"
                if pred_path.exists():
                    card_pred_paths[card_key] = pred_path

            if card_pred_paths:
                days_args.append((d, tensor_p, card_pred_paths, mbo_path))

        d += timedelta(days=1)

    log.info("Running on %d days, %d workers...", len(days_args), WORKERS)

    all_results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_day, args): args[0] for args in days_args}
        for fut in as_completed(futures):
            day = futures[fut]
            try:
                r = fut.result()
                all_results.append(r)
                # Log C4 summary
                c4_nb = r.get("c4_conv2.0_vol70_n_base", 0)
                c4_na = r.get("c4_conv2.0_vol70_n_agree", 0)
                c4_bp = (r.get("c4_conv2.0_vol70_base_chase_tp8_sl20") or {}).get("pnl", float("nan"))
                c4_ap = (r.get("c4_conv2.0_vol70_agree_chase_tp8_sl20") or {}).get("pnl", float("nan"))
                log.info("Day %s | C4: n=%d agree=%d(%.0f%%) base_pnl=%.0f agree_pnl=%.0f",
                         day.strftime("%Y-%m-%d"), c4_nb, c4_na,
                         100 * c4_na / max(c4_nb, 1), c4_bp or 0, c4_ap or 0)
            except Exception as e:
                log.error("Day %s failed: %s", day.strftime("%Y-%m-%d"), e)

    all_results.sort(key=lambda x: x["date"])

    # Aggregate
    summary = {}
    for card_key in CARDS:
        for cfg_name, _ in FILL_CONFIGS:
            for variant in ["base", "agree"]:
                key = f"{card_key}_{variant}_{cfg_name}"
                pnls = [r[key]["pnl"] for r in all_results
                        if r.get(key) and r[key].get("pnl") is not None]
                ns   = [r[key]["n"] for r in all_results
                        if r.get(key) and r[key].get("n")]
                wrs  = [r[key]["wr"] for r in all_results
                        if r.get(key) and r[key].get("wr")]
                n_sig_mean = float(np.mean([r.get(f"{card_key}_n_base", 0) for r in all_results])) \
                    if variant == "base" else \
                    float(np.mean([r.get(f"{card_key}_n_agree", 0) for r in all_results]))

                summary[key] = {
                    "pnl_per_day":   float(sum(pnls) / len(pnls)) if pnls else float("nan"),
                    "total_pnl":     float(sum(pnls)) if pnls else float("nan"),
                    "n_trades_day":  float(np.mean(ns)) if ns else 0.0,
                    "wr":            float(np.mean(wrs)) if wrs else float("nan"),
                    "n_signal_day":  n_sig_mean,
                    "n_days":        len(pnls),
                }

    output = {
        "n_days":  len(all_results),
        "l1_thresh": L1_THRESH,
        "cards":   list(CARDS.keys()),
        "configs": [c[0] for c in FILL_CONFIGS],
        "summary": summary,
        "per_day": all_results,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    log.info("\n=== CARD L1 FILTER SUMMARY ===")
    log.info("%-25s %-7s %-20s %8s %7s %8s %8s",
             "CARD_KEY", "VARIANT", "CONFIG", "PNL/DAY", "WR", "N_TRADE", "N_SIG")
    for card_key in CARDS:
        for cfg_name, _ in FILL_CONFIGS:
            for variant in ["base", "agree"]:
                key = f"{card_key}_{variant}_{cfg_name}"
                s = summary.get(key, {})
                log.info("%-25s %-7s %-20s %8.0f %6.1f%% %8.0f %8.0f",
                         card_key, variant, cfg_name,
                         s.get("pnl_per_day", float("nan")),
                         (s.get("wr") or 0) * 100,
                         s.get("n_trades_day", 0),
                         s.get("n_signal_day", 0))

    log.info("Saved to %s", OUT_FILE)


if __name__ == "__main__":
    main()
