#!/usr/bin/env python3
"""
Queue Time Exit Analysis
=========================
When we place a limit order, how does fill probability change over time?
- If not filled in 3s vs 5s vs 10s -> what's optimal cancel time?
- Does queue depth at entry predict fill probability?
- Order lifetime distributions by price level, side, time of day

Data sources:
  1. rithmic JSONL (A/C/M events, no fills -> infer fills from A+no-C pairs)
  2. databento MBO dbn.zst (A/C/M/T/F events — T=trade, F=fill)

Deployed on: Jupiter or Razer (CPU)
Output: data/processed/queue_time_results/queue_exit_analysis.json
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
    "/home/jupiter/Lvl3Quant",                        # Jupiter (WSL/Linux)
    r"C:\Users\Footb\Documents\Github\Lvl3Quant",    # Neptune
    r"C:\Users\claude\Lvl3Quant",                    # Razer
    r"C:\Users\nick\Lvl3Quant",                      # Uranus
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

DATA_DIR        = os.path.join(LVL3_ROOT, "data")
RITHMIC_DIR     = os.path.join(DATA_DIR, "raw", "rithmic_mbo")
DATABENTO_DIR   = os.path.join(DATA_DIR, "raw", "mbo")

# Also search alternate rithmic locations
RITHMIC_ALT_DIRS = [
    os.path.join(DATA_DIR, "raw", "rithmic"),
    os.path.join(DATA_DIR, "rithmic_mbo"),
]
BOOK_CACHE_DIR  = os.path.join(DATA_DIR, "processed", "dl_book_cache")
OUT_DIR         = os.path.join(DATA_DIR, "processed", "queue_time_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Config ──────────────────────────────────────────────────────────────────
TICK_SIZE    = 0.25                    # ES tick = $12.50
NS_TO_MS     = 1e-6                    # nanoseconds to milliseconds
MAX_LIFETIME_MS = 300_000              # 5 minutes cap
CANCEL_WINDOWS_MS = [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 30000]
MAX_FILES    = 10                      # limit for speed


# ─── Rithmic JSONL analysis ───────────────────────────────────────────────────

def analyze_rithmic_file(filepath):
    """
    Analyze one day of rithmic MBO JSONL.
    Fields: ts(ns), a(A/C/M), s(B/A), p(price), sz, pri(priority), oid, seq, t

    Since rithmic data has no T/F events, we infer:
      - 'filled' = order was Added but never Cancelled (within session)
      - 'cancelled' = order was Added then Cancelled
    """
    print(f"  Reading {os.path.basename(filepath)}...")

    # Build order lifecycle map
    orders = {}   # oid -> {add_ts, cancel_ts, price, side, size, pri}

    with open(filepath, "r") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue

            oid = ev.get("oid")
            if not oid:
                continue

            a  = ev.get("a", "")
            ts = ev.get("ts", 0)

            if a == "A":
                orders[oid] = {
                    "add_ts":    ts,
                    "cancel_ts": None,
                    "price":     ev.get("p", 0),
                    "side":      ev.get("s", ""),
                    "size":      ev.get("sz", 0),
                    "pri":       ev.get("pri", 0),
                }
            elif a == "C" and oid in orders:
                if orders[oid]["cancel_ts"] is None:
                    orders[oid]["cancel_ts"] = ts
            elif a == "M" and oid in orders:
                # Modify — update price/size but keep original add_ts
                orders[oid]["price"] = ev.get("p", orders[oid]["price"])
                orders[oid]["size"]  = ev.get("sz", orders[oid]["size"])

    # Build lifetime stats
    lifetimes_filled   = []   # orders that survived (never cancelled)
    lifetimes_cancelled = []  # orders that were cancelled

    for oid, od in orders.items():
        if od["add_ts"] == 0:
            continue

        if od["cancel_ts"] is not None and od["cancel_ts"] > od["add_ts"]:
            lt_ms = (od["cancel_ts"] - od["add_ts"]) * NS_TO_MS
            if lt_ms < MAX_LIFETIME_MS:
                lifetimes_cancelled.append({
                    "lifetime_ms": lt_ms,
                    "price":       od["price"],
                    "side":        od["side"],
                    "size":        od["size"],
                })
        else:
            # Not cancelled — assume filled or still on book
            lifetimes_filled.append({
                "price": od["price"],
                "side":  od["side"],
                "size":  od["size"],
            })

    if not lifetimes_cancelled:
        return None

    # Compute cancel window analysis
    lt_arr = np.array([x["lifetime_ms"] for x in lifetimes_cancelled])

    cancel_window_stats = {}
    for w_ms in CANCEL_WINDOWS_MS:
        n_cancel_before = (lt_arr <= w_ms).sum()
        pct_cancel = n_cancel_before / len(lt_arr) if len(lt_arr) > 0 else 0
        cancel_window_stats[f"{w_ms}ms"] = {
            "n_cancelled_by_window": int(n_cancel_before),
            "pct_of_all_cancelled":  float(pct_cancel),
        }

    # Fill probability by cancel window
    # If we cancel at time W, we miss fills that would have happened after W
    # Approx: orders alive past W are potentially fillable
    fill_prob_by_window = {}
    total_orders = len(orders)
    n_cancelled   = len(lifetimes_cancelled)
    n_survived    = len(lifetimes_filled)

    for w_ms in CANCEL_WINDOWS_MS:
        # Orders we'd cancel = those cancelled within W
        would_cancel = (lt_arr <= w_ms).sum()
        # Orders we'd keep = cancelled after W + survived (filled)
        would_keep = (lt_arr > w_ms).sum() + n_survived
        # Among those we'd keep, fill rate is n_survived / (n_survived + len(cancelled after W))
        denom = (lt_arr > w_ms).sum() + n_survived
        fill_rate = n_survived / denom if denom > 0 else 0
        fill_prob_by_window[f"{w_ms}ms"] = {
            "would_cancel_pct": float(would_cancel / n_cancelled) if n_cancelled > 0 else 0,
            "fill_rate_if_kept": float(fill_rate),
            "opportunity_cost_pct": float(would_cancel / total_orders) if total_orders > 0 else 0,
        }

    # Priority distribution: low pri = near front of queue = higher fill prob
    # pri is order priority (lower = better position)
    all_pris = [od["pri"] for od in orders.values() if od["pri"] > 0]

    result = {
        "source":             "rithmic",
        "date":               os.path.basename(filepath).split("_")[0],
        "n_orders_total":     total_orders,
        "n_orders_cancelled": n_cancelled,
        "n_orders_survived":  n_survived,
        "cancel_rate":        float(n_cancelled / total_orders) if total_orders > 0 else 0,
        "survive_rate":       float(n_survived / total_orders) if total_orders > 0 else 0,
        "lifetime_ms": {
            "median":  float(np.median(lt_arr)),
            "p25":     float(np.percentile(lt_arr, 25)),
            "p75":     float(np.percentile(lt_arr, 75)),
            "p95":     float(np.percentile(lt_arr, 95)),
            "mean":    float(lt_arr.mean()),
            "n":       len(lt_arr),
        },
        "cancel_window_stats": cancel_window_stats,
        "fill_prob_by_window": fill_prob_by_window,
        "pct_cancel_by_window": {
            f"{w}ms": float((lt_arr <= w).mean()) for w in CANCEL_WINDOWS_MS
        },
    }

    print(f"    Orders: {total_orders:,} total, {n_cancelled:,} cancelled ({100*n_cancelled/total_orders:.1f}%)")
    print(f"    Cancel lifetimes: median={np.median(lt_arr):.0f}ms, p75={np.percentile(lt_arr,75):.0f}ms")

    return result


# ─── Databento MBO analysis (richer: has T/F events) ─────────────────────────

def analyze_databento_file(filepath):
    """
    Analyze one databento MBO dbn file. Has actual T/F (trade/fill) events.
    Actions: A=add, C=cancel, M=modify, T=trade, F=fill, R=reset

    Supports both databento API styles:
      - Old (>=0.7): rec.hd.ts_event
      - New (0.71+): rec.ts_event directly
    """
    try:
        import databento as db
    except ImportError:
        return None

    print(f"  Reading {os.path.basename(filepath)}...")

    try:
        store = db.DBNStore.from_file(filepath)
    except Exception as e:
        print(f"    [WARN] Cannot open: {e}")
        return None

    # Detect API version for timestamp access
    def get_ts(rec):
        if hasattr(rec, 'ts_event'):
            return rec.ts_event
        elif hasattr(rec, 'hd') and hasattr(rec.hd, 'ts_event'):
            return rec.hd.ts_event
        return 0

    orders     = {}    # order_id -> {add_ts, fill_ts, cancel_ts, price, side, size}
    trade_prices = []  # actual trade prices for queue depth analysis
    fills      = []    # fill events

    from collections import Counter
    action_counts = Counter()

    for rec in store:
        a   = rec.action
        oid = rec.order_id
        ts  = get_ts(rec)

        action_counts[a] += 1

        price_val = float(rec.price) / 1e9 if hasattr(rec, "price") and rec.price > 1e9 else (float(rec.price) if hasattr(rec, "price") else 0)
        size_val  = rec.size if hasattr(rec, "size") else 0
        side_val  = str(rec.side) if hasattr(rec, "side") else ""

        if a == "A":
            orders[oid] = {
                "add_ts":    ts,
                "fill_ts":   None,
                "cancel_ts": None,
                "price":     price_val,
                "side":      side_val,
                "size":      size_val,
            }
        elif a == "C" and oid in orders:
            if orders[oid]["cancel_ts"] is None:
                orders[oid]["cancel_ts"] = ts
        elif a == "F" and oid in orders:
            orders[oid]["fill_ts"] = ts
            fills.append({
                "ts":    ts,
                "price": price_val,
                "side":  side_val,
                "size":  size_val,
            })
        elif a == "T":
            trade_prices.append(price_val)
        elif a == "M" and oid in orders:
            orders[oid]["price"] = price_val

    print(f"    Actions: {dict(action_counts)}")

    # Build lifetime arrays by outcome
    filled_lifetimes   = []
    cancelled_lifetimes = []
    ambiguous_lifetimes = []

    for oid, od in orders.items():
        if od["add_ts"] == 0:
            continue

        if od["fill_ts"] is not None and od["fill_ts"] > od["add_ts"]:
            lt = (od["fill_ts"] - od["add_ts"]) * NS_TO_MS
            if lt < MAX_LIFETIME_MS:
                filled_lifetimes.append(lt)
        elif od["cancel_ts"] is not None and od["cancel_ts"] > od["add_ts"]:
            lt = (od["cancel_ts"] - od["add_ts"]) * NS_TO_MS
            if lt < MAX_LIFETIME_MS:
                cancelled_lifetimes.append(lt)
        else:
            ambiguous_lifetimes.append(0)  # neither filled nor cancelled

    if not cancelled_lifetimes and not filled_lifetimes:
        return None

    n_filled    = len(filled_lifetimes)
    n_cancelled = len(cancelled_lifetimes)
    n_total     = n_filled + n_cancelled + len(ambiguous_lifetimes)

    fill_lt  = np.array(filled_lifetimes) if filled_lifetimes else np.array([0.0])
    cancel_lt = np.array(cancelled_lifetimes) if cancelled_lifetimes else np.array([0.0])

    # Fill probability by cancel window (ACTUAL fills, not inferred)
    fill_prob_by_window = {}
    for w_ms in CANCEL_WINDOWS_MS:
        # Would we fill if we cancel at W?
        fills_before_w = (fill_lt <= w_ms).sum()
        fills_after_w  = (fill_lt > w_ms).sum()
        cancels_before_w = (cancel_lt <= w_ms).sum()

        # Fill rate among orders alive past W = fills_after_w / (fills_after_w + cancels_after_w)
        cancels_after_w = (cancel_lt > w_ms).sum()
        denom = fills_after_w + cancels_after_w
        fill_rate_if_kept = fills_after_w / denom if denom > 0 else 0

        # Opportunity cost: fills we'd miss by cancelling at W
        opp_cost = fills_before_w / n_filled if n_filled > 0 else 0

        # Optimal: maximize (fill_rate_if_kept) while minimizing (adverse moves)
        fill_prob_by_window[f"{w_ms}ms"] = {
            "fill_rate_if_kept":  float(fill_rate_if_kept),
            "opp_cost_pct":       float(opp_cost),    # pct of fills missed
            "cancel_rate_at_w":   float((cancel_lt <= w_ms).mean()) if len(cancel_lt) > 0 else 0,
        }

    result = {
        "source":          "databento",
        "file":            os.path.basename(filepath),
        "n_orders":        n_total,
        "n_filled":        n_filled,
        "n_cancelled":     n_cancelled,
        "fill_rate":       float(n_filled / n_total) if n_total > 0 else 0,
        "cancel_rate":     float(n_cancelled / n_total) if n_total > 0 else 0,
        "n_trades":        len(trade_prices),
        "n_fills_logged":  len(fills),
        "fill_lifetime_ms": {
            "median": float(np.median(fill_lt)),
            "p25":    float(np.percentile(fill_lt, 25)),
            "p75":    float(np.percentile(fill_lt, 75)),
            "p95":    float(np.percentile(fill_lt, 95)),
            "mean":   float(fill_lt.mean()),
        } if n_filled > 0 else None,
        "cancel_lifetime_ms": {
            "median": float(np.median(cancel_lt)),
            "p25":    float(np.percentile(cancel_lt, 25)),
            "p75":    float(np.percentile(cancel_lt, 75)),
            "p95":    float(np.percentile(cancel_lt, 95)),
            "mean":   float(cancel_lt.mean()),
        } if n_cancelled > 0 else None,
        "fill_prob_by_window": fill_prob_by_window,
        "pct_cancel_by_window_actual": {
            f"{w}ms": float((cancel_lt <= w).mean()) if n_cancelled > 0 else 0
            for w in CANCEL_WINDOWS_MS
        },
    }

    print(f"    Fills: {n_filled}, Cancels: {n_cancelled}, Fill rate: {100*n_filled/n_total:.1f}%")
    if n_filled > 0:
        print(f"    Fill lifetime: median={np.median(fill_lt):.0f}ms, p75={np.percentile(fill_lt,75):.0f}ms")

    return result


# ─── Compute optimal cancel window ───────────────────────────────────────────

def compute_optimal_cancel_window(results_list):
    """
    Aggregate fill_prob_by_window across days.
    Optimal cancel window = maximize fill rate while catching >80% of would-be cancels.

    If fill_rate_if_kept drops significantly after time W and most cancels happen before W,
    then cancel at W.
    """
    if not results_list:
        return {}

    # Average across days
    agg = {}
    for w_ms in CANCEL_WINDOWS_MS:
        key = f"{w_ms}ms"
        rates = []
        for r in results_list:
            fp = r.get("fill_prob_by_window", {}).get(key, {})
            if fp:
                rates.append(fp)
        if rates:
            agg[key] = {
                k: float(np.mean([r[k] for r in rates if k in r]))
                for k in rates[0]
            }

    # Find optimal: where fill_rate_if_kept plateaus (derivative < 0.01)
    keys   = sorted(CANCEL_WINDOWS_MS)
    frates = [agg.get(f"{w}ms", {}).get("fill_rate_if_kept", 0) for w in keys]

    optimal_ms = None
    for i in range(1, len(frates)):
        delta = frates[i] - frates[i-1]
        if abs(delta) < 0.005 and frates[i] > 0.1:
            optimal_ms = keys[i]
            break

    return {
        "agg_by_window": agg,
        "optimal_cancel_ms": optimal_ms,
        "recommendation": (
            f"Cancel at {optimal_ms}ms — fill rate plateaus after this window"
            if optimal_ms else "Cancel window inconclusive from available data"
        ),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    print(f"[START] Queue Time Exit Analysis — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {"rithmic": [], "databento": []}

    # ── Source 1: Databento (preferred, has T/F events) ──
    dbn_files = sorted(glob.glob(os.path.join(DATABENTO_DIR, "*.mbo.dbn.zst")))
    print(f"\n[DATABENTO] {len(dbn_files)} files found")
    for f in dbn_files[:MAX_FILES]:
        try:
            result = analyze_databento_file(f)
            if result:
                all_results["databento"].append(result)
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(f)}: {e}")

    # ── Source 2: Rithmic JSONL ──
    rithmic_files = sorted(glob.glob(os.path.join(RITHMIC_DIR, "*.jsonl")))
    print(f"\n[RITHMIC] {len(rithmic_files)} files found")
    for f in rithmic_files[:MAX_FILES]:
        try:
            result = analyze_rithmic_file(f)
            if result:
                all_results["rithmic"].append(result)
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(f)}: {e}")

    # ── Compute optimal cancel windows ──
    opt_dbn     = compute_optimal_cancel_window(all_results["databento"])
    opt_rithmic = compute_optimal_cancel_window(all_results["rithmic"])

    elapsed = time.time() - start_time

    out = {
        "run_time": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "databento_days": len(all_results["databento"]),
        "rithmic_days":   len(all_results["rithmic"]),
        "per_day":        all_results,
        "optimal_cancel_databento": opt_dbn,
        "optimal_cancel_rithmic":   opt_rithmic,
        "cancel_windows_tested_ms": CANCEL_WINDOWS_MS,
    }

    out_path = os.path.join(OUT_DIR, "queue_exit_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[SAVED] -> {out_path}")

    # ── Summary ──
    print("\n" + "="*60)
    print("QUEUE TIME EXIT ANALYSIS — SUMMARY")
    print("="*60)

    for source_key, opt_obj in [("DATABENTO", opt_dbn), ("RITHMIC", opt_rithmic)]:
        results = all_results[source_key.lower()]
        if not results:
            print(f"\n{source_key}: No data")
            continue
        print(f"\n{source_key} ({len(results)} days):")

        # Aggregate fill stats
        fill_rates = [r.get("fill_rate", 0) for r in results]
        cancel_rates = [r.get("cancel_rate", 0) for r in results]
        if fill_rates:
            print(f"  Fill rate: {np.mean(fill_rates):.3f} | Cancel rate: {np.mean(cancel_rates):.3f}")

        # Cancel window table
        agg = opt_obj.get("agg_by_window", {})
        if agg:
            print(f"  {'Window':>10} {'FillRate(kept)':>16} {'OppCost':>10} {'CancelPct':>10}")
            print("  " + "-"*50)
            for w_ms in CANCEL_WINDOWS_MS:
                key = f"{w_ms}ms"
                a = agg.get(key, {})
                fr  = a.get("fill_rate_if_kept", 0)
                opp = a.get("opp_cost_pct", 0)
                cp  = a.get("cancel_rate_at_w", 0)
                print(f"  {key:>10} {fr:>16.3f} {opp:>10.3f} {cp:>10.3f}")

        opt_ms = opt_obj.get("optimal_cancel_ms")
        print(f"\n  RECOMMENDATION: {opt_obj.get('recommendation', 'N/A')}")

    print(f"\n[DONE] {elapsed:.1f}s")
    return out


if __name__ == "__main__":
    result = main()
    sys.exit(0)
