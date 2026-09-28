#!/usr/bin/env python3
"""
tick_sweep_v12_timestop_only.py — Pure time-stop strategy (NO TP/SL).

Rationale: All v8-v11 configs used TP/SL which creates asymmetric exit bias
(SL hit rate > TP hit rate due to random walk + spread). A time-stop removes
this bias entirely and tests whether the model's directional edge (IC=0.25)
can overcome costs on average.

Strategy:
- Enter on strong signal (top N% by |prediction|)
- Hold for exactly T seconds
- Exit at market (or passive if possible)
- PnL = signed price move in predicted direction - costs

If average signed PnL > cost_per_trade → profitable.
Cost: passive entry + passive exit = 0.752 ticks RT
      passive entry + market exit = 1.376 ticks RT (forced exit at time-stop)

HC #659 compliant: uses tick-level MBO replay, not bar sim.
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
MBO_DIR = ROOT / "data" / "preprocessed_mbo"
ALIGNED_DIR = ROOT / "data" / "preprocessed_mbo" / "pred_indices_aligned"
PRED_DIR = ROOT / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "oot_47day_perdate"
OUT_DIR = ROOT / "output" / "tick_replay_v12_timestop"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_SIZE = 0.25  # ES tick
COST_PASSIVE_RT = 0.376 * 2  # 0.752 ticks — commission only both sides
COST_MARKET_EXIT = 0.376 + 1.376  # 1.752 ticks — passive entry + market exit (1 tick crossing + commission)


def load_day(date_str, pred_head="pred_log_ret_1s"):
    """Load MBO events, aligned indices, and predictions for a date."""
    mbo_file = MBO_DIR / f"mbo_{date_str}.npz"
    aligned_file = ALIGNED_DIR / f"aligned_{date_str}.npz"
    pred_file = PRED_DIR / f"oot_{date_str}.npz"

    if not all(f.exists() for f in [mbo_file, aligned_file, pred_file]):
        return None, None, None, None

    mbo = np.load(mbo_file)
    aligned = np.load(aligned_file)
    preds_data = np.load(pred_file)

    event_indices = aligned["event_indices"]
    in_range = aligned["in_mbo_range"]

    if pred_head not in preds_data:
        return None, None, None, None

    predictions = preds_data[pred_head].astype(np.float64)

    return mbo, predictions, event_indices, in_range


def simulate_timestop_day(mbo, preds, event_indices, in_range, hold_seconds, threshold_pct, side="S"):
    """
    Simulate pure time-stop trades for one day.

    For each prediction above threshold:
    1. Find entry price (next trade after signal)
    2. Find exit price (first trade after hold_seconds elapsed)
    3. PnL = signed move in predicted direction

    Returns: list of trade dicts
    """
    ts_ns = mbo["ts_ns"]
    action = mbo["action"]
    side_arr = mbo["side"]
    price = mbo["price"]

    # Filter predictions in range
    valid_mask = in_range & (event_indices < len(ts_ns))
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return []

    valid_preds = preds[valid_indices].copy()
    valid_event_idx = event_indices[valid_indices]

    # Threshold: top N% by absolute prediction value
    abs_preds = np.abs(valid_preds)
    threshold = np.percentile(abs_preds, 100 - threshold_pct)

    # Direction filter
    if side == "S":
        signal_mask = valid_preds < -threshold  # Short signals (predict down)
    elif side == "L":
        signal_mask = valid_preds > threshold   # Long signals (predict up)
    else:  # "B" — both sides
        signal_mask = abs_preds >= threshold

    signal_indices = valid_indices[signal_mask]
    signal_event_idx = valid_event_idx[signal_mask]
    signal_preds_vals = valid_preds[signal_mask]

    if len(signal_indices) == 0:
        return []

    hold_ns = int(hold_seconds * 1e9)

    # Find TRADE events for entry/exit pricing
    # Action: 0=ADD, 1=CANCEL, 2=MODIFY, 3=TRADE, 4=FILL
    trade_mask = (action == 3) | (action == 4)
    trade_indices = np.where(trade_mask)[0]
    trade_ts = ts_ns[trade_indices]
    trade_prices = price[trade_indices]

    if len(trade_indices) < 10:
        return []

    trades = []
    last_exit_idx = 0  # Prevent overlapping trades

    for i in range(len(signal_indices)):
        evt_idx = signal_event_idx[i]
        pred_val = signal_preds_vals[i]
        pred_dir = -1 if pred_val < 0 else 1  # -1=short, +1=long

        signal_ts = ts_ns[evt_idx]

        # Find next trade after signal for entry
        entry_search = np.searchsorted(trade_ts, signal_ts, side='right')
        if entry_search >= len(trade_indices):
            continue

        entry_trade_idx = trade_indices[entry_search]
        if entry_trade_idx <= last_exit_idx:
            continue  # Skip if overlapping with previous trade

        entry_price = trade_prices[entry_search]
        entry_ts = trade_ts[entry_search]

        # Find first trade after hold period for exit
        exit_deadline = entry_ts + hold_ns
        exit_search = np.searchsorted(trade_ts, exit_deadline, side='left')

        if exit_search >= len(trade_indices):
            continue  # No exit available

        exit_price = trade_prices[exit_search]
        exit_ts = trade_ts[exit_search]
        exit_trade_idx = trade_indices[exit_search]

        # Calculate signed PnL in ticks
        price_move_ticks = (exit_price - entry_price) / TICK_SIZE

        # PnL = move in predicted direction
        if pred_dir == -1:  # Short
            signed_pnl = -price_move_ticks  # Profit if price went down
        else:  # Long
            signed_pnl = price_move_ticks  # Profit if price went up

        # Net PnL after costs (assume market exit at time-stop)
        net_pnl = signed_pnl - COST_MARKET_EXIT

        actual_hold_s = (exit_ts - entry_ts) / 1e9

        trades.append({
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pred_dir": pred_dir,
            "gross_pnl_ticks": signed_pnl,
            "net_pnl_ticks": net_pnl,
            "hold_seconds": actual_hold_s,
            "pred_val": pred_val,
        })

        last_exit_idx = exit_trade_idx

    return trades


def run_config(dates, hold_seconds, threshold_pct, side, label):
    """Run a single config across all dates."""
    all_trades = []
    daily_pnls = []

    for date_str in dates:
        mbo, preds, event_indices, in_range = load_day(date_str)
        if mbo is None or preds is None:
            continue

        day_trades = simulate_timestop_day(mbo, preds, event_indices, in_range,
                                           hold_seconds, threshold_pct, side)

        if day_trades:
            day_pnl = sum(t["net_pnl_ticks"] for t in day_trades)
            daily_pnls.append(day_pnl)
            all_trades.extend(day_trades)
        else:
            daily_pnls.append(0.0)

    if not all_trades:
        return None

    n_trades = len(all_trades)
    n_days = len(daily_pnls)
    trades_per_day = n_trades / max(n_days, 1)

    gross_pnls = [t["gross_pnl_ticks"] for t in all_trades]
    net_pnls = [t["net_pnl_ticks"] for t in all_trades]

    avg_gross = np.mean(gross_pnls)
    avg_net = np.mean(net_pnls)
    total_net = sum(net_pnls)

    # Win rate (on net PnL)
    wr = np.mean([p > 0 for p in net_pnls])

    # Profit factor
    gains = sum(p for p in net_pnls if p > 0)
    losses = abs(sum(p for p in net_pnls if p < 0))
    pf = gains / losses if losses > 0 else float('inf')

    # Daily Sharpe
    daily_arr = np.array(daily_pnls)
    if daily_arr.std() > 0:
        sharpe = daily_arr.mean() / daily_arr.std() * np.sqrt(252)
    else:
        sharpe = 0

    green_days = sum(1 for d in daily_pnls if d > 0)
    red_days = sum(1 for d in daily_pnls if d < 0)

    avg_hold = np.mean([t["hold_seconds"] for t in all_trades])

    return {
        "label": label,
        "n_trades": n_trades,
        "trades_per_day": trades_per_day,
        "avg_gross_ticks": avg_gross,
        "avg_net_ticks": avg_net,
        "total_net_ticks": total_net,
        "wr": wr,
        "pf": pf,
        "sharpe": sharpe,
        "green_days": green_days,
        "red_days": red_days,
        "avg_hold_s": avg_hold,
    }


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("TICK REPLAY v12 — PURE TIME-STOP (NO TP/SL)", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print("Rationale: TP/SL creates asymmetric exit bias.", flush=True)
    print("Time-stop tests raw directional edge vs costs.", flush=True)
    print(f"Costs: passive RT = {COST_PASSIVE_RT:.3f}t, market exit = {COST_MARKET_EXIT:.3f}t", flush=True)
    print(flush=True)

    # Discover dates (use aligned dir for date list, cross-check with pred dir)
    aligned_files = sorted(ALIGNED_DIR.glob("aligned_*.npz"))
    dates = [f.stem.replace("aligned_", "") for f in aligned_files
             if (PRED_DIR / f"oot_{f.stem.replace('aligned_', '')}.npz").exists()]
    print(f"Found {len(dates)} dates with aligned predictions", flush=True)
    print(flush=True)

    # Configs to test
    # Vary: hold time (1s, 3s, 5s, 10s, 20s), selectivity (top 5%, 10%, 20%), side (S, L, B)
    configs = [
        # Ultra-short hold — is the 1s edge capturable?
        (1.0, 10, "S", "1s_S_top10"),
        (1.0, 5, "S", "1s_S_top5"),
        (1.0, 2, "S", "1s_S_top2"),

        # 3s hold — slightly longer for price discovery
        (3.0, 10, "S", "3s_S_top10"),
        (3.0, 5, "S", "3s_S_top5"),
        (3.0, 2, "S", "3s_S_top2"),

        # 5s hold
        (5.0, 10, "S", "5s_S_top10"),
        (5.0, 5, "S", "5s_S_top5"),
        (5.0, 2, "S", "5s_S_top2"),

        # 10s hold — test signal persistence
        (10.0, 5, "S", "10s_S_top5"),
        (10.0, 2, "S", "10s_S_top2"),

        # 20s hold — beyond signal decay
        (20.0, 5, "S", "20s_S_top5"),

        # Both sides (strongest signals only)
        (3.0, 5, "B", "3s_B_top5"),
        (5.0, 5, "B", "5s_B_top5"),

        # Long side only (known weaker)
        (3.0, 5, "L", "3s_L_top5"),
        (5.0, 5, "L", "5s_L_top5"),
    ]

    print(f"Testing {len(configs)} configs", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    results = []
    for i, (hold, top_pct, side, label) in enumerate(configs):
        t1 = time.time()
        r = run_config(dates, hold, top_pct, side, label)
        elapsed = time.time() - t1

        if r is None:
            print(f"[{i+1}/{len(configs)}] ✗ {label} — no trades [{elapsed:.1f}s]", flush=True)
            continue

        status = "✓" if r["avg_net_ticks"] > 0 else "✗"
        print(f"[{i+1}/{len(configs)}] {status} {label}", flush=True)
        print(f"  {r['n_trades']} trades ({r['trades_per_day']:.0f}/day), "
              f"avg_gross={r['avg_gross_ticks']:+.3f}t, avg_net={r['avg_net_ticks']:+.3f}t, "
              f"WR={r['wr']:.3f}, PF={r['pf']:.3f}, Sharpe={r['sharpe']:.2f}", flush=True)
        print(f"  Days: {r['green_days']}G/{r['red_days']}R, avg_hold={r['avg_hold_s']:.1f}s [{elapsed:.1f}s]", flush=True)

        results.append(r)

    # Summary
    print(flush=True)
    print("=" * 70, flush=True)
    print("SUMMARY — sorted by avg net PnL per trade", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Config':<20} {'N':>6} {'Gross':>7} {'Net':>7} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'G/R':>5}", flush=True)
    print("-" * 70, flush=True)

    results.sort(key=lambda x: x["avg_net_ticks"], reverse=True)
    for r in results:
        print(f"{r['label']:<20} {r['n_trades']:>6} {r['avg_gross_ticks']:>+6.3f} "
              f"{r['avg_net_ticks']:>+6.3f} {r['wr']:>6.3f} {r['pf']:>6.3f} "
              f"{r['sharpe']:>7.2f} {r['green_days']:>2}/{r['red_days']}", flush=True)

    # Key question: does gross PnL exceed costs for ANY config?
    print(flush=True)
    best = results[0] if results else None
    if best and best["avg_gross_ticks"] > 0:
        print(f"BEST gross edge: {best['label']} → {best['avg_gross_ticks']:+.3f} ticks/trade", flush=True)
        print(f"  Costs: passive RT = {COST_PASSIVE_RT:.3f}t, market exit = {COST_MARKET_EXIT:.3f}t", flush=True)
        if best["avg_gross_ticks"] > COST_PASSIVE_RT:
            print(f"  ✓ PROFITABLE with passive entry+exit ({best['avg_gross_ticks']:.3f} > {COST_PASSIVE_RT:.3f})", flush=True)
        elif best["avg_gross_ticks"] > 0:
            print(f"  ✗ Positive gross but below passive RT cost. Need {COST_PASSIVE_RT:.3f}t, have {best['avg_gross_ticks']:.3f}t", flush=True)
    else:
        print("❌ No config has positive gross edge. Model directional signal insufficient.", flush=True)

    # Save
    out_path = OUT_DIR / "v12_timestop_results.json"
    with open(out_path, "w") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "results": results},
                  f, indent=2, default=str)
    print(f"\nSaved to {out_path}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
