#!/usr/bin/env python3
"""
conviction_exit_analysis.py — Conviction Exit Signal Analysis

For each baseline trade, examines the model's prediction z-score DURING the hold
period. If the z-score flips opposite to the trade direction and PERSISTS above a
magnitude threshold for a duration threshold, that constitutes a "conviction exit"
signal — the model is saying the trade is now wrong.

Sweeps over (duration_threshold, magnitude_threshold) combos to find the best
conviction exit parameters that cut losers early without killing winners.

Reads:
  - Baseline trade results: .../smart_exit_validation/baseline/c{1,2,4}_YYYY-MM-DD.json
  - Prediction NPZ files:   .../cnn_wf_stacked_predictions/YYYY-MM-DD_*.npz

Output:
  - Per-card JSON with all threshold combos: .../conviction_exit_analysis/
  - Compact summary table to stdout

Run on Jupiter:
  python3 conviction_exit_analysis.py [--cards 1 2 4] [--workers 8]
"""

import os
import sys
import json
import glob
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Configuration ─────────────────────────────────────────────────────────────

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")

BASELINE_DIR = LVL3_ROOT / "data" / "processed" / "smart_exit_validation" / "baseline"
OOT_VALIDATION_DIR = LVL3_ROOT / "data" / "processed" / "card_oot_validation"
OUTPUT_DIR = LVL3_ROOT / "data" / "processed" / "conviction_exit_analysis"

# Prediction directories to search (in priority order)
PRED_DIRS = [
    LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions",
    LVL3_ROOT / "data" / "processed" / "cnn_wf_sim_predictions",
    LVL3_ROOT / "data" / "processed" / "cnn_wf_norm_sweep_predictions",
]

# Card definitions: card_id -> config
# "oot_subdir": name of best config subdir in card_oot_validation (used for cards 2-7)
# "baseline_prefix": prefix in smart_exit_validation/baseline (used for card 1)
CARD_DEFS = {
    1: {
        "suffix": "book_predstdExit_conv1.5_vol50",
        "tp": 8,
        "desc": "Card1: TP8, book predstdExit conv1.5 vol50",
        "baseline_prefix": "c1",
        "oot_subdir": "c1_book_c15_v50_tp8_s01",
    },
    2: {
        "suffix": "book_predstdExit_conv1.5_vol50",
        "tp": 15,
        "desc": "Card2: TP15, book predstdExit conv1.5 vol50",
        "oot_subdir": None,  # Will auto-discover
    },
    3: {
        "suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "tp": 15,
        "desc": "Card3: TP15, raw rawExit conv0.15 vol70",
        "oot_subdir": None,
    },
    4: {
        "suffix": "book_predstdExit_conv2.0_vol70",
        "tp": 20,
        "desc": "Card4: TP20, book predstdExit conv2.0 vol70",
        "oot_subdir": "c4_book_c20_v70_tp20_s01",
    },
    5: {
        "suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "tp": None,
        "desc": "Card5: raw rawExit conv0.05 ethr0.5 vol0",
        "oot_subdir": "c5_raw_e05_v0_tpN_slN_s01",
    },
    6: {
        "suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "tp": 20,
        "desc": "Card6: TP20, raw rawExit conv0.15 vol70",
        "oot_subdir": "c6_raw_c015_e00_v70_tp20_sl25_s01",
    },
    7: {
        "suffix": "smooth_smoothExit_conv1.5_ethr0.0_vol70",
        "tp": None,
        "desc": "Card7: smooth smoothExit conv1.5 vol70",
        "oot_subdir": "c7_smooth_c15_e00_v70_tpN_sl20_s01",
    },
}

# Sweep grid
DURATION_THRESHOLDS_SEC = [5, 10, 15, 30, 60, 120]
MAGNITUDE_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# RTH constants
RTH_START_HOUR = 9
RTH_START_MIN = 30
RTH_END_HOUR = 16
RTH_END_MIN = 0
RTH_DURATION_SEC = (RTH_END_HOUR * 3600 + RTH_END_MIN * 60) - (
    RTH_START_HOUR * 3600 + RTH_START_MIN * 60
)  # 23400 seconds
RTH_NUM_BARS = RTH_DURATION_SEC * 10  # 234000 bars at 100ms each
BAR_NS = 100_000_000  # 100ms in nanoseconds

# ET offset: US/Eastern is UTC-5 (EST) or UTC-4 (EDT)
# We'll compute dynamically per date
ET_OFFSETS = {
    # Rough EDT/EST boundary: EDT Mar second Sun -> Nov first Sun
    # For simplicity, assume EDT (UTC-4) from March-November
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("conviction_exit")


# ── Helpers ───────────────────────────────────────────────────────────────────


def is_edt(date_str: str) -> bool:
    """Rough check if date falls in EDT (UTC-4) vs EST (UTC-5).
    EDT: second Sunday of March through first Sunday of November.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    # Second Sunday of March
    mar1 = datetime(year, 3, 1)
    # Days until Sunday (6 - weekday, mod 7)
    days_to_sun = (6 - mar1.weekday()) % 7
    edt_start = datetime(year, 3, 1 + days_to_sun + 7)  # second Sunday
    # First Sunday of November
    nov1 = datetime(year, 11, 1)
    days_to_sun = (6 - nov1.weekday()) % 7
    est_start = datetime(year, 11, 1 + days_to_sun)  # first Sunday
    return edt_start <= dt < est_start


def rth_start_ns(date_str: str) -> int:
    """Compute RTH start (9:30 AM ET) as nanoseconds since Unix epoch."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    utc_offset_hours = -4 if is_edt(date_str) else -5
    # 9:30 AM ET in UTC
    utc_hour = RTH_START_HOUR - utc_offset_hours  # e.g., 9:30 AM EDT = 13:30 UTC
    rth_utc = datetime(
        dt.year, dt.month, dt.day, utc_hour, RTH_START_MIN, 0, tzinfo=timezone.utc
    )
    return int(rth_utc.timestamp() * 1_000_000_000)


def ns_to_bar_index(timestamp_ns: int, rth_start: int) -> int:
    """Convert nanosecond timestamp to bar index (0-based from RTH start)."""
    delta_ns = timestamp_ns - rth_start
    if delta_ns < 0:
        return 0
    idx = delta_ns // BAR_NS
    return min(int(idx), RTH_NUM_BARS - 1)


def find_prediction_file(date_str: str, suffix: str) -> str | None:
    """Search prediction directories for the NPZ file matching date+suffix."""
    filename = f"{date_str}_{suffix}.npz"
    for pred_dir in PRED_DIRS:
        candidate = pred_dir / filename
        if candidate.exists():
            return str(candidate)
    # Fallback: glob search in all pred dirs
    for pred_dir in PRED_DIRS:
        if not pred_dir.exists():
            continue
        matches = list(pred_dir.glob(f"{date_str}*{suffix}*.npz"))
        if matches:
            return str(matches[0])
    return None


def load_predictions(npz_path: str) -> np.ndarray | None:
    """Load predictions from NPZ file. Handles various key names."""
    try:
        npz = np.load(npz_path, allow_pickle=True)
        keys = list(npz.keys())
        # Try common key names in priority order
        for candidate_key in ["predictions", "preds", "pred", "y_pred", "signal", "signals"]:
            if candidate_key in keys:
                arr = npz[candidate_key]
                if hasattr(arr, "shape"):
                    return arr.flatten().astype(np.float64)
        # If only one key, use it
        if len(keys) == 1:
            arr = npz[keys[0]]
            if hasattr(arr, "shape"):
                return arr.flatten().astype(np.float64)
        # If multiple keys, try the first array-like one
        for k in keys:
            arr = npz[k]
            if hasattr(arr, "shape") and arr.ndim >= 1 and arr.size > 1000:
                return arr.flatten().astype(np.float64)
        log.warning(f"NPZ {npz_path} has keys {keys} but no suitable prediction array found")
        return None
    except Exception as e:
        log.error(f"Failed to load {npz_path}: {e}")
        return None


def load_baseline_trades(card_id: int, date_str: str) -> tuple[list[dict] | None, str | None]:
    """Load baseline trade results for a card on a specific date.

    Searches in order:
    1. smart_exit_validation/baseline/ (has c1_ files)
    2. card_oot_validation/card{N}/{subdir}/ (has per-date JSON files)

    Returns (trades, pred_file_hint) tuple.
    """
    card_def = CARD_DEFS.get(card_id, {})

    # Source 1: smart_exit_validation/baseline/
    baseline_prefix = card_def.get("baseline_prefix", f"c{card_id}")
    filename = f"{baseline_prefix}_{date_str}.json"
    filepath = BASELINE_DIR / filename
    if filepath.exists():
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            trades = data["trades"] if isinstance(data, dict) and "trades" in data else (data if isinstance(data, list) else None)
            pred_hint = data.get("predictions_file") if isinstance(data, dict) else None
            if trades:
                return trades, pred_hint
        except Exception as e:
            log.error(f"Failed to load {filepath}: {e}")

    # Source 2: card_oot_validation/card{N}/{subdir}/
    oot_subdir = card_def.get("oot_subdir")
    card_dir = OOT_VALIDATION_DIR / f"card{card_id}"

    if oot_subdir:
        # Use specific subdir
        filepath = card_dir / oot_subdir / f"{date_str}.json"
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                trades = data["trades"] if isinstance(data, dict) and "trades" in data else (data if isinstance(data, list) else None)
                pred_hint = data.get("predictions_file") if isinstance(data, dict) else None
                if trades:
                    return trades, pred_hint
            except Exception as e:
                log.error(f"Failed to load {filepath}: {e}")
    else:
        # Auto-discover: try first available subdir
        if card_dir.exists():
            subdirs = sorted([d for d in card_dir.iterdir() if d.is_dir()])
            for subdir in subdirs:
                filepath = subdir / f"{date_str}.json"
                if filepath.exists():
                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f)
                        trades = data["trades"] if isinstance(data, dict) and "trades" in data else None
                        pred_hint = data.get("predictions_file") if isinstance(data, dict) else None
                        if trades:
                            log.info(f"  Auto-discovered subdir for card {card_id}: {subdir.name}")
                            card_def["oot_subdir"] = subdir.name  # Cache for future dates
                            return trades, pred_hint
                    except Exception:
                        continue

    return None, None


def analyze_trade_conviction(
    trade: dict,
    preds: np.ndarray,
    rth_start: int,
    duration_thresholds: list[int],
    magnitude_thresholds: list[float],
) -> dict:
    """Analyze a single trade for conviction exit signals across all threshold combos.

    Returns dict with:
      - trade summary fields
      - per (dur, mag) combo: would_exit, exit_bar_offset, exit_pnl_estimate
    """
    fill_ns = trade["fill_time_ns"]
    exit_ns = trade["exit_time_ns"]
    side = trade["side"]  # "BUY" or "SELL"
    pnl = trade["pnl_dollars"]
    signal_strength = trade.get("signal_strength", 0.0)
    exit_reason = trade.get("exit_reason", "unknown")
    mae_ticks = trade.get("mae_ticks", 0)
    mfe_ticks = trade.get("mfe_ticks", 0)
    hold_duration_ns = trade.get("hold_duration_ns", exit_ns - fill_ns)

    # Compute bar indices for the hold period
    fill_bar = ns_to_bar_index(fill_ns, rth_start)
    exit_bar = ns_to_bar_index(exit_ns, rth_start)

    if fill_bar >= exit_bar or fill_bar >= len(preds) or exit_bar > len(preds):
        # Trade outside prediction range or zero-length
        return None

    # Clamp to prediction array bounds
    exit_bar = min(exit_bar, len(preds))
    hold_preds = preds[fill_bar:exit_bar]

    if len(hold_preds) == 0:
        return None

    # Determine "opposite direction" for this trade
    # BUY trade: opposite signal is NEGATIVE prediction (model says sell)
    # SELL trade: opposite signal is POSITIVE prediction (model says buy)
    if side == "BUY":
        opposite_mask = hold_preds < 0
        opposite_magnitudes = np.abs(hold_preds)
    else:  # SELL
        opposite_mask = hold_preds > 0
        opposite_magnitudes = np.abs(hold_preds)

    is_winner = pnl > 0
    hold_bars = len(hold_preds)
    hold_sec = hold_bars / 10.0  # 100ms bars -> seconds

    result = {
        "fill_bar": int(fill_bar),
        "exit_bar": int(exit_bar),
        "hold_bars": hold_bars,
        "hold_sec": round(hold_sec, 1),
        "side": side,
        "pnl": round(pnl, 2),
        "is_winner": is_winner,
        "signal_strength": round(signal_strength, 4),
        "exit_reason": exit_reason,
        "mae_ticks": mae_ticks,
        "mfe_ticks": mfe_ticks,
        "max_opposite_zscore": round(float(np.max(opposite_magnitudes * opposite_mask)), 4),
        "pct_bars_opposite": round(float(np.mean(opposite_mask)) * 100, 1),
        "combos": {},
    }

    # For each (duration, magnitude) threshold combo, check if conviction exit fires
    for dur_sec in duration_thresholds:
        dur_bars = int(dur_sec * 10)  # Convert seconds to 100ms bars
        if dur_bars > hold_bars:
            # Threshold longer than entire hold — can never fire
            for mag in magnitude_thresholds:
                key = f"d{dur_sec}_m{mag}"
                result["combos"][key] = {
                    "would_exit": False,
                    "reason": "hold_too_short",
                }
            continue

        for mag in magnitude_thresholds:
            key = f"d{dur_sec}_m{mag}"

            # Build mask: bars where prediction is opposite AND magnitude >= threshold
            if mag > 0:
                trigger_mask = opposite_mask & (opposite_magnitudes >= mag)
            else:
                trigger_mask = opposite_mask

            # Find continuous streaks of True in trigger_mask
            # We need a streak of >= dur_bars consecutive True values
            would_exit = False
            exit_offset_bar = None

            if dur_bars == 0:
                # Duration 0 means any single bar triggers — always fires if any opposite bar
                if np.any(trigger_mask):
                    would_exit = True
                    exit_offset_bar = int(np.argmax(trigger_mask))
            else:
                # Efficient streak detection using convolution
                # A streak of N consecutive 1s in a binary array can be found by
                # convolving with a kernel of N ones and checking for sum == N
                if np.any(trigger_mask):
                    trigger_int = trigger_mask.astype(np.int8)
                    # Use cumsum trick for O(n) streak detection
                    # Reset cumsum at every 0
                    streak_counts = np.zeros(len(trigger_int), dtype=np.int32)
                    current_streak = 0
                    for i in range(len(trigger_int)):
                        if trigger_int[i]:
                            current_streak += 1
                            streak_counts[i] = current_streak
                        else:
                            current_streak = 0

                    # Find first bar where streak reaches dur_bars
                    streak_hits = np.where(streak_counts >= dur_bars)[0]
                    if len(streak_hits) > 0:
                        would_exit = True
                        exit_offset_bar = int(streak_hits[0])

            if would_exit and exit_offset_bar is not None:
                exit_time_sec = exit_offset_bar / 10.0
                # Estimate what PnL would have been at conviction exit point
                # We can use the prediction at exit point as a proxy direction
                # but actual PnL requires tick-level data we don't have.
                # Instead, flag whether this was a winner or loser that got cut.
                result["combos"][key] = {
                    "would_exit": True,
                    "exit_offset_bar": exit_offset_bar,
                    "exit_time_in_hold_sec": round(exit_time_sec, 1),
                    "pct_hold_elapsed": round(exit_offset_bar / hold_bars * 100, 1),
                    "pred_at_exit": round(float(hold_preds[exit_offset_bar]), 4),
                }
            else:
                result["combos"][key] = {
                    "would_exit": False,
                }

    return result


def process_date(
    card_id: int,
    date_str: str,
    pred_suffix: str,
) -> dict | None:
    """Process all trades for one card on one date. Returns analysis dict or None."""
    # Load baseline trades
    trades, pred_hint = load_baseline_trades(card_id, date_str)
    if not trades or len(trades) == 0:
        return None

    # Find predictions: use hint from trade file first, then fall back to card suffix
    npz_path = None
    if pred_hint:
        # pred_hint is like "2025-12-01_book_predstdExit_conv2.0_vol70.npz"
        for pred_dir in PRED_DIRS:
            candidate = pred_dir / pred_hint
            if candidate.exists():
                npz_path = str(candidate)
                break
    if npz_path is None:
        npz_path = find_prediction_file(date_str, pred_suffix)
    if npz_path is None:
        log.warning(f"  No prediction file for c{card_id} {date_str}")
        return None

    preds = load_predictions(npz_path)
    if preds is None:
        return None

    # Compute RTH start for this date
    rth_start = rth_start_ns(date_str)

    # Analyze each trade
    trade_results = []
    for trade in trades:
        result = analyze_trade_conviction(
            trade,
            preds,
            rth_start,
            DURATION_THRESHOLDS_SEC,
            MAGNITUDE_THRESHOLDS,
        )
        if result is not None:
            trade_results.append(result)

    if not trade_results:
        return None

    return {
        "date": date_str,
        "card_id": card_id,
        "npz_path": npz_path,
        "num_preds": len(preds),
        "num_trades": len(trades),
        "num_analyzed": len(trade_results),
        "trades": trade_results,
    }


def aggregate_results(all_date_results: list[dict], card_id: int) -> dict:
    """Aggregate per-trade results into summary statistics per threshold combo."""
    combos = {}
    for dur in DURATION_THRESHOLDS_SEC:
        for mag in MAGNITUDE_THRESHOLDS:
            key = f"d{dur}_m{mag}"
            combos[key] = {
                "duration_sec": dur,
                "magnitude": mag,
                "total_trades": 0,
                "winners_total": 0,
                "losers_total": 0,
                "winners_cut": 0,         # False positives: winners killed
                "losers_cut": 0,          # True positives: losers saved
                "winners_kept": 0,
                "losers_kept": 0,
                "pnl_winners_cut": 0.0,   # PnL lost from killing winners
                "pnl_losers_cut": 0.0,    # PnL saved from cutting losers (positive = savings)
                "total_pnl_original": 0.0,
                "avg_exit_pct_winners": [],
                "avg_exit_pct_losers": [],
            }

    for date_result in all_date_results:
        for trade in date_result["trades"]:
            pnl = trade["pnl"]
            is_winner = trade["is_winner"]
            for key, combo_data in trade["combos"].items():
                if key not in combos:
                    continue
                c = combos[key]
                c["total_trades"] += 1
                c["total_pnl_original"] += pnl
                if is_winner:
                    c["winners_total"] += 1
                else:
                    c["losers_total"] += 1

                if combo_data["would_exit"]:
                    if is_winner:
                        c["winners_cut"] += 1
                        c["pnl_winners_cut"] += pnl  # This is positive PnL we'd lose
                        if "pct_hold_elapsed" in combo_data:
                            c["avg_exit_pct_winners"].append(combo_data["pct_hold_elapsed"])
                    else:
                        c["losers_cut"] += 1
                        c["pnl_losers_cut"] += abs(pnl)  # This is negative PnL we'd save
                        if "pct_hold_elapsed" in combo_data:
                            c["avg_exit_pct_losers"].append(combo_data["pct_hold_elapsed"])
                else:
                    if is_winner:
                        c["winners_kept"] += 1
                    else:
                        c["losers_kept"] += 1

    # Compute derived metrics
    summary = []
    for key, c in combos.items():
        total = c["total_trades"]
        if total == 0:
            continue

        losers_total = c["losers_total"]
        winners_total = c["winners_total"]
        losers_cut = c["losers_cut"]
        winners_cut = c["winners_cut"]

        # Net PnL improvement = money saved from cutting losers - money lost from killing winners
        # pnl_losers_cut is already positive (abs of losses saved)
        # pnl_winners_cut is positive (wins we'd forfeit)
        net_pnl_improvement = c["pnl_losers_cut"] - c["pnl_winners_cut"]

        # False positive rate: what fraction of winners get killed
        fpr = winners_cut / winners_total if winners_total > 0 else 0.0

        # True positive rate: what fraction of losers get cut
        tpr = losers_cut / losers_total if losers_total > 0 else 0.0

        # Precision: of all trades cut, what fraction were actually losers
        total_cut = losers_cut + winners_cut
        precision = losers_cut / total_cut if total_cut > 0 else 0.0

        avg_exit_pct_w = (
            round(np.mean(c["avg_exit_pct_winners"]), 1)
            if c["avg_exit_pct_winners"]
            else None
        )
        avg_exit_pct_l = (
            round(np.mean(c["avg_exit_pct_losers"]), 1)
            if c["avg_exit_pct_losers"]
            else None
        )

        entry = {
            "key": key,
            "duration_sec": c["duration_sec"],
            "magnitude": c["magnitude"],
            "total_trades": total,
            "winners_total": winners_total,
            "losers_total": losers_total,
            "winners_cut": winners_cut,
            "losers_cut": losers_cut,
            "winners_kept": c["winners_kept"],
            "losers_kept": c["losers_kept"],
            "pnl_winners_cut": round(c["pnl_winners_cut"], 2),
            "pnl_losers_cut": round(c["pnl_losers_cut"], 2),
            "net_pnl_improvement": round(net_pnl_improvement, 2),
            "total_pnl_original": round(c["total_pnl_original"], 2),
            "false_positive_rate": round(fpr, 4),
            "true_positive_rate": round(tpr, 4),
            "precision": round(precision, 4),
            "avg_exit_pct_hold_winners": avg_exit_pct_w,
            "avg_exit_pct_hold_losers": avg_exit_pct_l,
        }
        summary.append(entry)

    # Sort by net PnL improvement descending
    summary.sort(key=lambda x: x["net_pnl_improvement"], reverse=True)

    return {
        "card_id": card_id,
        "card_desc": CARD_DEFS.get(card_id, {}).get("desc", f"Card {card_id}"),
        "total_dates": len(all_date_results),
        "total_trades": sum(d["num_analyzed"] for d in all_date_results),
        "duration_thresholds_sec": DURATION_THRESHOLDS_SEC,
        "magnitude_thresholds": MAGNITUDE_THRESHOLDS,
        "combos": summary,
        "per_date": [
            {
                "date": d["date"],
                "num_trades": d["num_analyzed"],
                "npz_path": d["npz_path"],
            }
            for d in all_date_results
        ],
    }


def print_summary_table(agg: dict):
    """Print compact summary table to stdout."""
    card_id = agg["card_id"]
    print(f"\n{'='*90}")
    print(f"  CONVICTION EXIT ANALYSIS — Card {card_id}: {agg['card_desc']}")
    print(f"  {agg['total_dates']} dates, {agg['total_trades']} trades analyzed")
    print(f"{'='*90}")

    # Header
    print(
        f"{'Dur(s)':>7} {'Mag':>5} | "
        f"{'Cut':>5} {'W_Cut':>6} {'L_Cut':>6} | "
        f"{'FPR':>6} {'TPR':>6} {'Prec':>6} | "
        f"{'$Saved':>10} {'$Lost':>10} {'$Net':>10}"
    )
    print("-" * 90)

    # Only show combos with at least one trade cut
    shown = 0
    for c in agg["combos"]:
        total_cut = c["winners_cut"] + c["losers_cut"]
        if total_cut == 0 and c["magnitude"] > 0:
            continue
        print(
            f"{c['duration_sec']:>7} {c['magnitude']:>5.1f} | "
            f"{total_cut:>5} {c['winners_cut']:>6} {c['losers_cut']:>6} | "
            f"{c['false_positive_rate']:>6.3f} {c['true_positive_rate']:>6.3f} {c['precision']:>6.3f} | "
            f"{c['pnl_losers_cut']:>10.2f} {c['pnl_winners_cut']:>10.2f} {c['net_pnl_improvement']:>10.2f}"
        )
        shown += 1
        if shown >= 42:  # Max rows in table
            break

    # Highlight best combo
    if agg["combos"]:
        best = agg["combos"][0]
        print(f"\n  BEST: d{best['duration_sec']}s m{best['magnitude']} — "
              f"Net PnL improvement: ${best['net_pnl_improvement']:,.2f} "
              f"(saved ${best['pnl_losers_cut']:,.2f} from {best['losers_cut']} losers, "
              f"killed ${best['pnl_winners_cut']:,.2f} from {best['winners_cut']} winners)")
        print(f"        FPR={best['false_positive_rate']:.3f}, "
              f"TPR={best['true_positive_rate']:.3f}, "
              f"Precision={best['precision']:.3f}")

    print()


def discover_dates(card_id: int) -> list[str]:
    """Find all dates with trade files for a card (searches both data sources)."""
    dates = set()
    card_def = CARD_DEFS.get(card_id, {})

    # Source 1: smart_exit_validation/baseline/
    baseline_prefix = card_def.get("baseline_prefix", f"c{card_id}")
    pattern = str(BASELINE_DIR / f"{baseline_prefix}_*.json")
    for f in glob.glob(pattern):
        basename = os.path.basename(f)
        parts = basename.replace(".json", "").split("_", 1)
        if len(parts) == 2:
            dates.add(parts[1])

    # Source 2: card_oot_validation/card{N}/{subdir}/
    card_dir = OOT_VALIDATION_DIR / f"card{card_id}"
    if card_dir.exists():
        oot_subdir = card_def.get("oot_subdir")
        if oot_subdir:
            subdir_path = card_dir / oot_subdir
            if subdir_path.exists():
                for f in subdir_path.glob("*.json"):
                    # Format: 2025-12-01.json
                    date_str = f.stem
                    if len(date_str) == 10 and date_str[4] == '-':
                        dates.add(date_str)
        else:
            # Auto-discover: use first subdir
            subdirs = sorted([d for d in card_dir.iterdir() if d.is_dir()])
            if subdirs:
                for f in subdirs[0].glob("*.json"):
                    date_str = f.stem
                    if len(date_str) == 10 and date_str[4] == '-':
                        dates.add(date_str)

    return sorted(dates)


def run_card_analysis(card_id: int, workers: int = 1) -> dict | None:
    """Run full conviction exit analysis for one card."""
    card_def = CARD_DEFS[card_id]
    pred_suffix = card_def["suffix"]

    log.info(f"=== Card {card_id}: {card_def['desc']} ===")

    # Discover available dates
    dates = discover_dates(card_id)
    if not dates:
        log.warning(f"No baseline trade files found for card {card_id}")
        return None

    log.info(f"Found {len(dates)} dates: {dates[0]} to {dates[-1]}")

    # Process each date
    all_results = []
    t0 = time.time()

    for i, date_str in enumerate(dates):
        result = process_date(card_id, date_str, pred_suffix)
        if result is not None:
            all_results.append(result)

        # Progress logging every 10 dates
        if (i + 1) % 10 == 0 or (i + 1) == len(dates):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            trades_so_far = sum(r["num_analyzed"] for r in all_results)
            log.info(
                f"  Card {card_id}: {i+1}/{len(dates)} dates processed "
                f"({trades_so_far} trades, {elapsed:.1f}s, {rate:.1f} dates/s)"
            )

    if not all_results:
        log.warning(f"No analyzable results for card {card_id}")
        return None

    # Aggregate
    agg = aggregate_results(all_results, card_id)

    # Add per-trade detail for JSON output
    agg["trade_details"] = []
    for date_result in all_results:
        for trade in date_result["trades"]:
            detail = {
                "date": date_result["date"],
                "fill_bar": trade["fill_bar"],
                "exit_bar": trade["exit_bar"],
                "hold_sec": trade["hold_sec"],
                "side": trade["side"],
                "pnl": trade["pnl"],
                "is_winner": trade["is_winner"],
                "signal_strength": trade["signal_strength"],
                "exit_reason": trade["exit_reason"],
                "mae_ticks": trade["mae_ticks"],
                "mfe_ticks": trade["mfe_ticks"],
                "max_opposite_zscore": trade["max_opposite_zscore"],
                "pct_bars_opposite": trade["pct_bars_opposite"],
                "conviction_exits": {},
            }
            for key, combo_data in trade["combos"].items():
                if combo_data["would_exit"]:
                    detail["conviction_exits"][key] = {
                        "exit_time_in_hold_sec": combo_data.get("exit_time_in_hold_sec"),
                        "pct_hold_elapsed": combo_data.get("pct_hold_elapsed"),
                        "pred_at_exit": combo_data.get("pred_at_exit"),
                    }
            agg["trade_details"].append(detail)

    return agg


def probe_npz_structure():
    """Probe a sample NPZ file to report its structure. Run this first to verify."""
    log.info("Probing NPZ file structure...")
    for pred_dir in PRED_DIRS:
        if not pred_dir.exists():
            log.info(f"  {pred_dir} — does not exist")
            continue
        npz_files = sorted(pred_dir.glob("*.npz"))
        if not npz_files:
            log.info(f"  {pred_dir} — no NPZ files")
            continue
        sample = npz_files[0]
        log.info(f"  {pred_dir} — {len(npz_files)} files")
        try:
            npz = np.load(str(sample), allow_pickle=True)
            keys = list(npz.keys())
            log.info(f"    Sample: {sample.name}")
            log.info(f"    Keys: {keys}")
            for k in keys:
                arr = npz[k]
                if hasattr(arr, "shape"):
                    log.info(f"    '{k}': shape={arr.shape}, dtype={arr.dtype}")
                    if arr.size > 0:
                        flat = arr.flatten()
                        log.info(
                            f"      min={flat.min():.4f}, max={flat.max():.4f}, "
                            f"mean={flat.mean():.4f}, std={flat.std():.4f}"
                        )
                else:
                    log.info(f"    '{k}': {type(arr)} = {arr}")
        except Exception as e:
            log.error(f"    Failed to probe {sample}: {e}")
        break  # Only probe first available directory


def main():
    parser = argparse.ArgumentParser(
        description="Conviction Exit Analysis — sweep duration+magnitude thresholds"
    )
    parser.add_argument(
        "--cards",
        type=int,
        nargs="+",
        default=[1, 3, 4, 5, 6, 7],
        help="Card IDs to analyze (default: 1 3 4 5 6 7)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, dates are processed serially per card)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Only probe NPZ file structure and exit",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR

    # Probe NPZ structure first
    probe_npz_structure()

    if args.probe:
        return

    # Validate paths
    if not BASELINE_DIR.exists():
        log.error(f"Baseline directory not found: {BASELINE_DIR}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_dir}")

    # Run analysis per card
    all_aggs = {}
    for card_id in args.cards:
        if card_id not in CARD_DEFS:
            log.warning(f"Unknown card {card_id}, skipping")
            continue

        agg = run_card_analysis(card_id, workers=args.workers)
        if agg is None:
            continue

        all_aggs[card_id] = agg

        # Save per-card JSON
        out_file = output_dir / f"conviction_exit_c{card_id}.json"
        with open(out_file, "w") as f:
            json.dump(agg, f, indent=2)
        log.info(f"Saved: {out_file}")

        # Print summary
        print_summary_table(agg)

    # Save combined summary (without per-trade details for compactness)
    if all_aggs:
        combined = {
            "analysis_time": datetime.now().isoformat(),
            "duration_thresholds_sec": DURATION_THRESHOLDS_SEC,
            "magnitude_thresholds": MAGNITUDE_THRESHOLDS,
            "cards": {},
        }
        for card_id, agg in all_aggs.items():
            combined["cards"][str(card_id)] = {
                "desc": agg["card_desc"],
                "total_dates": agg["total_dates"],
                "total_trades": agg["total_trades"],
                "best_combos": agg["combos"][:5] if agg["combos"] else [],
            }
        summary_file = output_dir / "conviction_exit_summary.json"
        with open(summary_file, "w") as f:
            json.dump(combined, f, indent=2)
        log.info(f"Saved combined summary: {summary_file}")

    # Final cross-card comparison
    print(f"\n{'='*90}")
    print("  CROSS-CARD BEST CONVICTION EXIT PARAMETERS")
    print(f"{'='*90}")
    for card_id, agg in all_aggs.items():
        if agg["combos"]:
            best = agg["combos"][0]
            print(
                f"  Card {card_id}: d={best['duration_sec']}s, mag={best['magnitude']} "
                f"→ Net ${best['net_pnl_improvement']:+,.2f} "
                f"(FPR={best['false_positive_rate']:.3f}, TPR={best['true_positive_rate']:.3f})"
            )
    print()


if __name__ == "__main__":
    main()
