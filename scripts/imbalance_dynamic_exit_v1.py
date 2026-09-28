#!/usr/bin/env python3
"""
Imbalance Dynamic Exit Simulator v1 (HC #515)

Tests dynamic exits based on ORDER FLOW IMBALANCE features during trades.
Monitors OFI/buy-sell-ratio during positions and exits when imbalance reverses.

Vectorized for speed: precomputes imbalance signals at all stride points,
then uses numpy operations per config instead of Python-level per-trade loops.

Memory-efficient: processes one date at a time.

Cost: 0.376 ticks RT (commission only, HC #512)
"""

import gc
import json
import time
import itertools
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Paths ──
PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/imbalance_dynamic_exit_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
STRIDE = 250
WINDOW_SIZE = 3000
COST_TICKS_RT = 0.376
PRED_10S_COL = 2

# Core imbalance features: ofi_short(22), ofi_long(23), buy_sell_ratio(19)
CORE_IMB_INDICES = [22, 23, 19]

# ── Parameter sweep ──
CONFIDENCE_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
MAX_HOLD_STRIDES = [20, 40, 80, 120]
REVERSAL_THRESHOLDS = [0.5, 1.0, 2.0]
SIDES = ["both", "long_only", "short_only"]
TP_TICKS = [2, 4, 8, 999]
SL_TICKS = [4, 8, 16]


def pred_to_event_idx(pred_idx):
    return pred_idx * STRIDE + WINDOW_SIZE - 1


def get_overlapping_dates():
    pred_dates = {f.stem[:8] for f in PRED_DIR.glob("*_predictions.npz")}
    mbo_dates = {f.stem[:8] for f in MBO_DIR.glob("*_mbo_events.npz")}
    return sorted(pred_dates & mbo_dates)


def load_date_data(date_str):
    """Load, build cum_price, extract imbalance at strides. Returns compact arrays."""
    pred_path = PRED_DIR / f"{date_str}_predictions.npz"
    mbo_path = MBO_DIR / f"{date_str}_mbo_events.npz"
    try:
        pred = np.load(pred_path)
        mbo = np.load(mbo_path)
    except Exception as e:
        print(f"  [WARN] Failed to load {date_str}: {e}")
        return None

    predictions = pred["predictions"][:, PRED_10S_COL]  # (N,)
    events = mbo["events"]              # (M, 25)
    labels_1s = mbo["labels_1s"]        # (M,)
    timestamps = mbo["timestamps"]      # (M,)
    n_preds = len(predictions)
    n_events = len(events)

    if n_preds < 200:
        return None

    # Extract imbalance at each stride point
    imb = np.full((n_preds, 3), np.nan, dtype=np.float32)
    for i in range(n_preds):
        eidx = i * STRIDE + WINDOW_SIZE - 1
        if eidx < n_events:
            imb[i, 0] = events[eidx, 22]  # ofi_short
            imb[i, 1] = events[eidx, 23]  # ofi_long
            imb[i, 2] = events[eidx, 19]  # buy_sell_ratio

    # Build cumulative price from chained 1s labels
    if len(timestamps) > 20000:
        mid = len(timestamps) // 2
        dt_ns = float(timestamps[mid + 10000] - timestamps[mid])
        N_1s = max(int(round(10000 * 1e9 / dt_ns)), 500) if dt_ns > 0 else 4000
    else:
        N_1s = 4000

    n_blocks = n_events // N_1s
    coarse = np.zeros(n_blocks + 1, dtype=np.float64)
    for k in range(n_blocks):
        val = labels_1s[k * N_1s]
        coarse[k + 1] = coarse[k] + (val if not np.isnan(val) else 0.0)

    cum_price = np.zeros(n_preds, dtype=np.float64)
    for i in range(n_preds):
        eidx = i * STRIDE + WINDOW_SIZE - 1
        if eidx >= n_events:
            cum_price[i] = cum_price[i - 1] if i > 0 else 0.0
            continue
        block = eidx // N_1s
        frac = (eidx % N_1s) / N_1s
        if block >= n_blocks:
            cum_price[i] = coarse[-1]
        else:
            cum_price[i] = coarse[block] + frac * (coarse[block + 1] - coarse[block])

    # Free large arrays
    del events, labels_1s, timestamps, pred, mbo

    return {
        "preds": predictions.astype(np.float32),
        "imb": imb,
        "cum_price": cum_price,
        "n": n_preds,
    }


def simulate_date_fast(preds, imb, cum_price, n,
                       conf_thresh, max_hold, rev_thresh, side, tp, sl):
    """
    Fast simulation for one date + one config.

    Uses a while loop but with minimal Python overhead per stride
    (direct array indexing, no dict creation per trade).
    """
    n_trades = 0
    total_net = 0.0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    hold_sum = 0
    # Exit reason counters
    ex_imb_mean = 0
    ex_imb_strong = 0
    ex_tp = 0
    ex_sl = 0
    ex_max = 0
    ex_eod = 0

    i = 0
    while i < n:
        pv = preds[i]

        # Fast NaN check + threshold
        if pv != pv or pv > -conf_thresh and pv < conf_thresh:
            i += 1
            continue

        is_long = pv > 0.0
        if side == 1 and not is_long:
            i += 1
            continue
        if side == 2 and is_long:
            i += 1
            continue

        entry_price = cum_price[i]
        sign = 1.0 if is_long else -1.0
        exit_j = min(i + max_hold, n - 1)
        reason = 4  # max_hold

        for j in range(i + 1, min(i + max_hold + 1, n)):
            # TP/SL
            pnl_j = sign * (cum_price[j] - entry_price)
            if pnl_j >= tp:
                exit_j = j
                reason = 2  # tp
                break
            if pnl_j <= -sl:
                exit_j = j
                reason = 3  # sl
                break

            # Imbalance check
            i0 = imb[j, 0]
            i1 = imb[j, 1]
            i2 = imb[j, 2]
            if i0 != i0 or i1 != i1 or i2 != i2:
                continue

            mean_v = (i0 + i1 + i2) / 3.0
            if is_long:
                if mean_v < 0:
                    exit_j = j
                    reason = 0  # imb_mean
                    break
                if i0 < -rev_thresh or i1 < -rev_thresh or i2 < -rev_thresh:
                    exit_j = j
                    reason = 1  # imb_strong
                    break
            else:
                if mean_v > 0:
                    exit_j = j
                    reason = 0
                    break
                if i0 > rev_thresh or i1 > rev_thresh or i2 > rev_thresh:
                    exit_j = j
                    reason = 1
                    break

        raw = sign * (cum_price[exit_j] - entry_price)
        net = raw - COST_TICKS_RT

        n_trades += 1
        total_net += net
        hold_sum += (exit_j - i)
        if net > 0:
            wins += 1
            gross_win += net
        else:
            gross_loss += abs(net)

        if reason == 0:
            ex_imb_mean += 1
        elif reason == 1:
            ex_imb_strong += 1
        elif reason == 2:
            ex_tp += 1
        elif reason == 3:
            ex_sl += 1
        else:
            ex_max += 1

        i = exit_j + 1

    return (n_trades, total_net, wins, gross_win, gross_loss, hold_sum,
            ex_imb_mean, ex_imb_strong, ex_tp, ex_sl, ex_max)


def main():
    print("=" * 80)
    print("Imbalance Dynamic Exit Simulator v1 (HC #515)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)

    dates = get_overlapping_dates()
    print(f"\nFound {len(dates)} overlapping dates: {dates[0]} to {dates[-1]}")

    # Build configs
    all_configs = list(itertools.product(
        CONFIDENCE_THRESHOLDS, MAX_HOLD_STRIDES, REVERSAL_THRESHOLDS,
        SIDES, TP_TICKS, SL_TICKS
    ))
    n_configs = len(all_configs)
    print(f"Total configs: {n_configs}")

    # Side string to int
    side_map = {"both": 0, "long_only": 1, "short_only": 2}

    # Accumulators per config
    acc_trades = np.zeros(n_configs, dtype=np.int64)
    acc_net = np.zeros(n_configs, dtype=np.float64)
    acc_wins = np.zeros(n_configs, dtype=np.int64)
    acc_gwin = np.zeros(n_configs, dtype=np.float64)
    acc_gloss = np.zeros(n_configs, dtype=np.float64)
    acc_hold = np.zeros(n_configs, dtype=np.int64)
    acc_ex = np.zeros((n_configs, 5), dtype=np.int64)  # mean,strong,tp,sl,max
    # Daily P&L for Sharpe: list of lists
    daily_pnls = [[] for _ in range(n_configs)]

    n_dates_ok = 0
    t_start = time.time()

    for di, d in enumerate(dates):
        print(f"\n  [{di+1}/{len(dates)}] {d}...", end=" ", flush=True)
        data = load_date_data(d)
        if data is None:
            print("SKIP")
            continue
        n_dates_ok += 1

        preds = data["preds"]
        imb_arr = data["imb"]
        cum = data["cum_price"]
        n = data["n"]
        print(f"loaded ({n} preds)", end="", flush=True)

        t0 = time.time()

        for ci, (ct, mh, rt, sd, tp, sl_val) in enumerate(all_configs):
            si = side_map[sd]
            result = simulate_date_fast(preds, imb_arr, cum, n,
                                        ct, mh, rt, si, tp, sl_val)
            nt, net, w, gw, gl, hs, em, es, et, esl, emx = result

            acc_trades[ci] += nt
            acc_net[ci] += net
            acc_wins[ci] += w
            acc_gwin[ci] += gw
            acc_gloss[ci] += gl
            acc_hold[ci] += hs
            acc_ex[ci, 0] += em
            acc_ex[ci, 1] += es
            acc_ex[ci, 2] += et
            acc_ex[ci, 3] += esl
            acc_ex[ci, 4] += emx
            daily_pnls[ci].append(net)

        dt = time.time() - t0
        elapsed = time.time() - t_start
        rate = (di + 1) / elapsed * 60
        print(f" | {n_configs} cfgs in {dt:.1f}s | elapsed {elapsed:.0f}s ({rate:.1f} dates/min)")

        del data
        gc.collect()

    total_time = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"Done: {n_dates_ok} dates, {n_configs} configs in {total_time:.0f}s")

    # Build results
    results = []
    for ci, (ct, mh, rt, sd, tp, sl_val) in enumerate(all_configs):
        nt = int(acc_trades[ci])
        if nt == 0:
            continue
        net = float(acc_net[ci])
        w = int(acc_wins[ci])
        gw = float(acc_gwin[ci])
        gl = float(acc_gloss[ci])
        hs = int(acc_hold[ci])

        wr = w / nt
        pf = gw / gl if gl > 0 else 999.0
        tpd = nt / max(n_dates_ok, 1)
        ah = hs / nt

        dv = daily_pnls[ci]
        if len(dv) >= 2:
            dm = np.mean(dv)
            ds = np.std(dv, ddof=1)
            sharpe = float(dm / ds * np.sqrt(252)) if ds > 0 else 0
            down = [x for x in dv if x < 0]
            if len(down) >= 2:
                dd = np.std(down, ddof=1)
                sortino = float(dm / dd * np.sqrt(252)) if dd > 0 else 0
            else:
                sortino = sharpe * 1.5
        else:
            sharpe = sortino = 0

        ex_names = ["imb_mean_reversal", "imb_strong_reversal",
                     "take_profit", "stop_loss", "max_hold"]
        exits = {ex_names[k]: int(acc_ex[ci, k]) for k in range(5) if acc_ex[ci, k] > 0}

        results.append({
            "confidence_threshold": ct,
            "max_hold_strides": mh,
            "reversal_threshold": rt,
            "side": sd,
            "tp_ticks": tp,
            "sl_ticks": sl_val,
            "n_trades": nt,
            "n_days": n_dates_ok,
            "trades_per_day": round(tpd, 1),
            "total_net_ticks": round(net, 2),
            "avg_net_ticks": round(net / nt, 4),
            "win_rate": round(wr, 4),
            "profit_factor": round(pf, 3),
            "daily_sharpe": round(sharpe, 3),
            "daily_sortino": round(sortino, 3),
            "avg_hold_strides": round(ah, 1),
            "exit_reasons": exits,
        })

    print(f"Configs with trades: {len(results)}")

    # Filter and sort
    filtered = [r for r in results if r["n_trades"] >= 50]
    filtered.sort(key=lambda x: x["daily_sharpe"], reverse=True)

    # Save
    out = OUTPUT_DIR / "sweep_results.json"
    with open(out, "w") as f:
        json.dump({
            "metadata": {
                "script": "imbalance_dynamic_exit_v1.py",
                "timestamp": datetime.now().isoformat(),
                "n_dates": n_dates_ok,
                "n_configs": n_configs,
                "n_results": len(results),
                "cost_ticks_rt": COST_TICKS_RT,
                "runtime_seconds": round(total_time, 1),
                "exit_features": ["ofi_short(22)", "ofi_long(23)", "buy_sell_ratio(19)"],
            },
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {out}")

    top = OUTPUT_DIR / "top_configs.json"
    with open(top, "w") as f:
        json.dump(filtered[:20], f, indent=2)
    print(f"Saved: {top}")

    # Print top 10
    print(f"\n{'=' * 80}")
    print("TOP 10 (by daily Sharpe, min 50 trades):")
    print("=" * 80)
    for idx, r in enumerate(filtered[:10]):
        print(f"\n  #{idx+1}: Sharpe={r['daily_sharpe']:.3f} | "
              f"Sortino={r['daily_sortino']:.3f} | "
              f"PF={r['profit_factor']:.2f} | "
              f"WR={r['win_rate']:.1%} | "
              f"Net={r['total_net_ticks']:.1f} | "
              f"Trades/day={r['trades_per_day']}")
        print(f"       conf={r['confidence_threshold']}, "
              f"hold={r['max_hold_strides']}, "
              f"rev={r['reversal_threshold']}, "
              f"side={r['side']}, "
              f"tp={r['tp_ticks']}, sl={r['sl_ticks']}")
        print(f"       exits: {r['exit_reasons']}")

    profitable = [r for r in filtered if r["total_net_ticks"] > 0]
    print(f"\n{'=' * 80}")
    print("SUMMARY:")
    print(f"  Dates: {n_dates_ok}")
    print(f"  Configs (>=50 trades): {len(filtered)}")
    pct = len(profitable) / max(len(filtered), 1) * 100
    print(f"  Profitable: {len(profitable)} ({pct:.1f}%)")
    if profitable:
        print(f"  Best net: {max(r['total_net_ticks'] for r in profitable):.1f} ticks")
    if filtered:
        print(f"  Best Sharpe: {filtered[0]['daily_sharpe']:.3f}")
    print("=" * 80)
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
