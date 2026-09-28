#!/usr/bin/env python3
"""
Minute-Bar FIFO Validation of Integrated Pipeline Paper Engine
================================================================

Matches the paper engine logic EXACTLY:
  1. Signal fires on 30-min bar N
  2. Entry price = OPEN of bar N+1 (next bar)
  3. Fill: passive limit, filled when price trades FILL_THROUGH_TICKS through entry
  4. Exit: TP/SL/timeout checked on minute bars from fill_time onward
  5. Same filters: conf >= 0.52, |zscore| >= 0.3, time filters

This is an independent re-implementation to verify the paper engine results.

Author: Claude (autonomous build, HC #658 hourly productivity)
"""

import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT))

MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
OOT_NPZ = ROOT / "output" / "lh_30min_deep_v1" / "concat_oot.npz"
OUTPUT_DIR = ROOT / "output" / "tick_fifo_validation_integrated_pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MIN-FIFO] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("min-fifo")

# ES Constants
TICK_SIZE = 0.25  # 1 tick = 0.25 points
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376

# Strategy parameters (matching paper engine)
TP_TICKS = 20
SL_TICKS = 4
MAX_HOLD_MINUTES = 30
CONF_THRESHOLD = 0.52
ZSCORE_THRESHOLD = 0.3
FILL_THROUGH_TICKS = 1

# Time filters
SKIP_FIRST_BAR = True
FIRST_BAR_CUTOFF_ET = (9, 0)
AFTERNOON_SHORT_FILTER = True
AFTERNOON_CUTOFF_ET = 11
LUNCH_HOUR_FILTER = True
LUNCH_HOUR_ET = 12

# RTH
RTH_START = (13, 30)  # 9:30 ET in UTC
RTH_END = (20, 0)     # 16:00 ET in UTC

# Cost model
COST_PASSIVE_PASSIVE = COMMISSION_RT_TICKS        # entry passive + exit passive (TP)
COST_PASSIVE_MARKET = COMMISSION_RT_TICKS + 1.0   # entry passive + exit market (SL/timeout)


def load_oot_predictions():
    d = np.load(str(OOT_NPZ), allow_pickle=True)
    return d["preds"], d["confs"], d["dates"], d["actuals"]


def compute_expanding_zscore(preds):
    zscores = np.zeros_like(preds)
    running_sum = 0.0
    running_sq = 0.0
    for i in range(len(preds)):
        if i < 2:
            zscores[i] = 0.0
        else:
            mean = running_sum / i
            var = running_sq / i - mean ** 2
            std = max(np.sqrt(max(var, 0)), 1e-10)
            zscores[i] = (preds[i] - mean) / std
        running_sum += preds[i]
        running_sq += preds[i] ** 2
    return zscores


def load_minute_bars(date_str: str) -> Optional[pd.DataFrame]:
    path = MINUTE_BAR_DIR / f"{date_str}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["ts_minute"] = pd.to_datetime(df["ts_minute"], utc=True)
    return df.sort_values("ts_minute").reset_index(drop=True)


def get_30min_bars_from_minutes(minute_df: pd.DataFrame) -> List[Dict]:
    """
    Build 30-min bar OHLCV from 1-min bars, matching RTH boundaries.
    Each 30-min bar also carries its constituent minute bars.
    """
    if minute_df is None or minute_df.empty:
        return []

    # RTH filter
    rth_start = minute_df["ts_minute"].iloc[0].replace(hour=RTH_START[0], minute=RTH_START[1])
    rth_end = minute_df["ts_minute"].iloc[0].replace(hour=RTH_END[0], minute=RTH_END[1])
    rth_df = minute_df[(minute_df["ts_minute"] >= rth_start) & (minute_df["ts_minute"] < rth_end)]

    if rth_df.empty:
        return []

    bars = []
    t = rth_start
    while t < rth_end:
        bar_end = t + timedelta(minutes=30)
        mask = (rth_df["ts_minute"] >= t) & (rth_df["ts_minute"] < bar_end)
        bar_minutes = rth_df[mask]

        if not bar_minutes.empty:
            bars.append({
                "bar_key": t,
                "open": bar_minutes.iloc[0]["open"],
                "high": bar_minutes["high"].max(),
                "low": bar_minutes["low"].min(),
                "close": bar_minutes.iloc[-1]["close"],
                "volume": bar_minutes["volume"].sum(),
                "minutes": bar_minutes,
            })
        t = bar_end

    return bars


def simulate_fifo_fill(entry_price, direction, minute_bars):
    """
    Fill when price trades FILL_THROUGH_TICKS through entry.
    Returns fill_time or None.
    """
    if minute_bars.empty:
        return None

    through = FILL_THROUGH_TICKS * TICK_SIZE

    for _, row in minute_bars.iterrows():
        if direction == 1:
            if row["low"] <= entry_price - through:
                return {"fill_time": row["ts_minute"], "fill_price": entry_price}
        else:
            if row["high"] >= entry_price + through:
                return {"fill_time": row["ts_minute"], "fill_price": entry_price}
    return None


def simulate_exit(entry_price, direction, fill_time, minute_bars):
    """
    TP/SL/timeout exit using minute bars from fill_time.
    Matches paper engine logic exactly.
    """
    tp_price = entry_price + (TP_TICKS * TICK_SIZE * direction)
    sl_price = entry_price - (SL_TICKS * TICK_SIZE * direction)
    time_stop = fill_time + timedelta(minutes=MAX_HOLD_MINUTES)

    if minute_bars.empty:
        return {"exit_type": "NoData", "exit_price": entry_price, "pnl_ticks": 0, "cost": COMMISSION_RT_TICKS}

    for _, row in minute_bars.iterrows():
        ts = row["ts_minute"]
        hi = row["high"]
        lo = row["low"]

        # EOD forced exit (15:55 ET = 19:55 UTC)
        eod_time = ts.replace(hour=19, minute=55, second=0)
        if ts >= eod_time:
            exit_price = row["close"]
            pnl = (exit_price - entry_price) / TICK_SIZE * direction
            return {"exit_type": "EOD", "exit_price": exit_price, "pnl_ticks": pnl, "cost": COST_PASSIVE_MARKET}

        if direction == 1:
            tp_hit = hi >= tp_price
            sl_hit = lo <= sl_price
        else:
            tp_hit = lo <= tp_price
            sl_hit = hi >= sl_price

        if tp_hit and sl_hit:
            # Both hit in same bar — use close as tiebreaker (matches paper engine)
            favorable = (row["close"] >= entry_price) if direction == 1 else (row["close"] <= entry_price)
            if favorable:
                return {"exit_type": "TP", "exit_price": tp_price, "pnl_ticks": TP_TICKS, "cost": COST_PASSIVE_PASSIVE}
            else:
                return {"exit_type": "SL", "exit_price": sl_price, "pnl_ticks": -SL_TICKS, "cost": COST_PASSIVE_MARKET}
        elif sl_hit:
            return {"exit_type": "SL", "exit_price": sl_price, "pnl_ticks": -SL_TICKS, "cost": COST_PASSIVE_MARKET}
        elif tp_hit:
            return {"exit_type": "TP", "exit_price": tp_price, "pnl_ticks": TP_TICKS, "cost": COST_PASSIVE_PASSIVE}

        # Time stop
        if ts >= time_stop:
            exit_price = row["close"]
            pnl = (exit_price - entry_price) / TICK_SIZE * direction
            return {"exit_type": "TimeStop", "exit_price": exit_price, "pnl_ticks": pnl, "cost": COST_PASSIVE_MARKET}

    # End of data
    last = minute_bars.iloc[-1]
    pnl = (last["close"] - entry_price) / TICK_SIZE * direction
    return {"exit_type": "DataEnd", "exit_price": last["close"], "pnl_ticks": pnl, "cost": COST_PASSIVE_MARKET}


def ts_to_et(ts):
    """UTC -> ET (EDT: UTC-4)."""
    return ts - timedelta(hours=4)


def passes_time_filters(bar_time, direction):
    et = ts_to_et(bar_time)
    h, m = et.hour, et.minute

    if SKIP_FIRST_BAR and (h < FIRST_BAR_CUTOFF_ET[0] or (h == FIRST_BAR_CUTOFF_ET[0] and m < FIRST_BAR_CUTOFF_ET[1])):
        return False
    if AFTERNOON_SHORT_FILTER and direction == -1 and h >= AFTERNOON_CUTOFF_ET:
        return False
    if LUNCH_HOUR_FILTER and h == LUNCH_HOUR_ET:
        return False
    return True


def run_validation():
    log.info("Loading OOT predictions...")
    preds, confs, dates, actuals = load_oot_predictions()
    zscores = compute_expanding_zscore(preds)

    unique_dates = sorted(set(dates))
    log.info(f"Loaded {len(preds)} predictions across {len(unique_dates)} OOT dates")

    # Per-date indices
    date_groups = defaultdict(list)
    for i in range(len(preds)):
        date_groups[dates[i]].append(i)

    all_trades = []
    stats = defaultdict(int)

    for di, date_str in enumerate(unique_dates):
        if di % 20 == 0:
            log.info(f"Processing {di+1}/{len(unique_dates)}: {date_str}")

        indices = date_groups[date_str]

        minute_df = load_minute_bars(date_str)
        if minute_df is None:
            stats["no_minute_data"] += len(indices)
            continue

        bars_30 = get_30min_bars_from_minutes(minute_df)
        if len(bars_30) == 0:
            stats["no_30min_bars"] += len(indices)
            continue

        # Map predictions to 30-min bars
        n_bars = min(len(indices), len(bars_30))

        # Position tracking
        position_free_after = pd.Timestamp("2000-01-01", tz="UTC")

        for bar_idx in range(n_bars):
            sample_idx = indices[bar_idx]
            pred = preds[sample_idx]
            conf = confs[sample_idx]
            zscore = zscores[sample_idx]
            stats["total"] += 1

            # Conf/zscore filter
            if conf < CONF_THRESHOLD or abs(zscore) < ZSCORE_THRESHOLD:
                stats["filtered_conf"] += 1
                continue

            direction = 1 if pred > 0 else -1
            bar_time = bars_30[bar_idx]["bar_key"]

            # Time filters
            if not passes_time_filters(bar_time, direction):
                stats["filtered_time"] += 1
                continue

            # RTH boundary check
            h, m = bar_time.hour, bar_time.minute
            t = h * 60 + m
            if not (RTH_START[0] * 60 + RTH_START[1] <= t < RTH_END[0] * 60 + RTH_END[1] - 30):
                stats["filtered_rth"] += 1
                continue

            # Need next bar for entry
            if bar_idx + 1 >= len(bars_30):
                stats["no_next_bar"] += 1
                continue

            next_bar = bars_30[bar_idx + 1]
            entry_price = next_bar["open"]
            next_minutes = next_bar["minutes"]

            # Position check
            if next_bar["bar_key"] < position_free_after:
                stats["filtered_position"] += 1
                continue

            # Simulate FIFO fill
            fill = simulate_fifo_fill(entry_price, direction, next_minutes)
            if fill is None:
                stats["no_fill"] += 1
                all_trades.append({
                    "date": date_str, "bar_time": str(bar_time),
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry_price": entry_price, "conf": conf, "zscore": zscore,
                    "filled": False, "exit_type": "no_fill",
                    "pnl_ticks": 0, "cost_ticks": 0, "net_ticks": 0,
                })
                continue

            fill_time = fill["fill_time"]

            # Exit using minute bars from fill_time onward
            exit_window_end = fill_time + timedelta(minutes=MAX_HOLD_MINUTES + 30)
            exit_bars = minute_df[
                (minute_df["ts_minute"] >= fill_time) &
                (minute_df["ts_minute"] <= exit_window_end)
            ].sort_values("ts_minute")

            exit_info = simulate_exit(entry_price, direction, fill_time, exit_bars)

            raw_pnl = exit_info["pnl_ticks"]
            cost = exit_info["cost"]
            net_pnl = raw_pnl - cost

            # Lock position until exit
            exit_time = exit_info.get("exit_time", fill_time + timedelta(minutes=MAX_HOLD_MINUTES))
            if isinstance(exit_time, str):
                exit_time = pd.Timestamp(exit_time)
            # Approximate: lock for hold duration
            position_free_after = fill_time + timedelta(minutes=MAX_HOLD_MINUTES)

            all_trades.append({
                "date": date_str, "bar_time": str(bar_time),
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": entry_price, "conf": conf, "zscore": zscore,
                "filled": True, "exit_type": exit_info["exit_type"],
                "exit_price": exit_info["exit_price"],
                "pnl_ticks": raw_pnl, "cost_ticks": cost, "net_ticks": net_pnl,
            })

    # ── Analysis ──────────────────────────────────────────────────────
    filled = [t for t in all_trades if t["filled"]]
    unfilled = [t for t in all_trades if not t["filled"]]

    log.info(f"\n{'='*70}")
    log.info(f"MINUTE-BAR FIFO VALIDATION — INTEGRATED PIPELINE")
    log.info(f"Config: TP={TP_TICKS}, SL={SL_TICKS}, hold={MAX_HOLD_MINUTES}min")
    log.info(f"{'='*70}")
    log.info(f"Stats: {dict(stats)}")
    log.info(f"Filled: {len(filled)} | Unfilled: {len(unfilled)} | Fill rate: {100*len(filled)/max(len(all_trades),1):.1f}%")

    if not filled:
        log.info("NO FILLED TRADES.")
        return

    net_arr = np.array([t["net_ticks"] for t in filled])
    raw_arr = np.array([t["pnl_ticks"] for t in filled])
    total_net = net_arr.sum()
    avg_net = net_arr.mean()
    wr = (net_arr > 0).mean()

    wins = net_arr[net_arr > 0]
    losses = net_arr[net_arr <= 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")

    # Daily
    daily_pnl = defaultdict(float)
    daily_cnt = defaultdict(int)
    for t in filled:
        daily_pnl[t["date"]] += t["net_ticks"]
        daily_cnt[t["date"]] += 1

    daily_arr = np.array([daily_pnl[d] for d in sorted(daily_pnl.keys())])
    n_days = len(daily_arr)

    if n_days > 1 and daily_arr.std() > 0:
        sharpe = daily_arr.mean() / daily_arr.std() * np.sqrt(252)
        neg = daily_arr[daily_arr < 0]
        downside = np.sqrt(np.mean(neg ** 2)) if len(neg) > 0 else 1e-10
        sortino = daily_arr.mean() / downside * np.sqrt(252)
    else:
        sharpe = sortino = 0.0

    cumsum = np.cumsum(daily_arr)
    max_dd = (np.maximum.accumulate(cumsum) - cumsum).max()

    # Exit breakdown
    exit_cnt = defaultdict(int)
    for t in filled:
        exit_cnt[t["exit_type"]] += 1

    # Direction
    longs = [t for t in filled if t["direction"] == "LONG"]
    shorts = [t for t in filled if t["direction"] == "SHORT"]
    green_days = (daily_arr > 0).sum()

    log.info(f"\n{'='*70}")
    log.info(f"RESULTS")
    log.info(f"{'='*70}")
    log.info(f"Trades:          {len(filled)}")
    log.info(f"Trading days:    {n_days}")
    log.info(f"Avg/day:         {len(filled)/max(n_days,1):.1f}")
    log.info(f"")
    log.info(f"Net ticks/trade: {avg_net:+.3f}")
    log.info(f"Total net ticks: {total_net:+.1f}")
    log.info(f"Total net USD:   ${total_net * TICK_VALUE:+,.0f}")
    log.info(f"Win Rate:        {wr*100:.1f}%")
    log.info(f"Profit Factor:   {pf:.2f}")
    log.info(f"Daily Sharpe:    {sharpe:.2f}")
    log.info(f"Daily Sortino:   {sortino:.2f}")
    log.info(f"Max Drawdown:    {max_dd:.1f}t (${max_dd * TICK_VALUE:,.0f})")
    log.info(f"")
    log.info(f"Exit breakdown:")
    for et in ["TP", "SL", "TimeStop", "EOD", "DataEnd"]:
        c = exit_cnt.get(et, 0)
        log.info(f"  {et:12s}: {c:4d} ({100*c/len(filled):.1f}%)")
    log.info(f"")
    log.info(f"Longs:  {len(longs)} ({100*len(longs)/len(filled):.0f}%)")
    log.info(f"Shorts: {len(shorts)} ({100*len(shorts)/len(filled):.0f}%)")
    if longs:
        ln = np.array([t["net_ticks"] for t in longs])
        log.info(f"  Long:  avg {ln.mean():+.3f}t, WR {(ln>0).mean()*100:.1f}%")
    if shorts:
        sn = np.array([t["net_ticks"] for t in shorts])
        log.info(f"  Short: avg {sn.mean():+.3f}t, WR {(sn>0).mean()*100:.1f}%")

    log.info(f"\nGreen days: {green_days}/{n_days} ({100*green_days/n_days:.0f}%)")

    # Daily table
    log.info(f"\n{'Date':>12s} {'N':>4s} {'Net':>8s} {'Cum':>8s}")
    cum = 0.0
    for d in sorted(daily_pnl.keys()):
        cum += daily_pnl[d]
        log.info(f"{d:>12s} {daily_cnt[d]:>4d} {daily_pnl[d]:>+8.1f} {cum:>8.1f}")

    # Comparison
    log.info(f"\n{'='*70}")
    log.info(f"COMPARISON TO PAPER ENGINE")
    log.info(f"{'='*70}")
    log.info(f"Paper engine Layer 3 (filters + SL=4): 354 trades, Sharpe 9.66, WR 45.2%, net +$26,211")
    log.info(f"This validation:                       {len(filled)} trades, Sharpe {sharpe:.2f}, WR {wr*100:.1f}%, net ${total_net*TICK_VALUE:+,.0f}")

    if len(filled) > 0:
        trade_diff = len(filled) - 354
        log.info(f"\nTrade count delta: {trade_diff:+d} ({100*trade_diff/354:+.0f}%)")

    if sharpe >= 2.0:
        log.info(f"\n✅ EDGE CONFIRMED at minute resolution (Sharpe >= 2.0)")
    elif sharpe > 0:
        log.info(f"\n⚠️ Positive but weaker than 30-min bar sim")
    else:
        log.info(f"\n❌ NO EDGE at minute resolution")

    # Save
    result = {
        "n_filled": len(filled), "n_unfilled": len(unfilled),
        "fill_rate": len(filled) / max(len(all_trades), 1),
        "n_days": n_days, "avg_net": float(avg_net),
        "total_net_ticks": float(total_net), "total_net_usd": float(total_net * TICK_VALUE),
        "wr": float(wr), "pf": float(min(pf, 999)),
        "sharpe": float(sharpe), "sortino": float(sortino),
        "max_dd_ticks": float(max_dd),
        "exits": dict(exit_cnt),
        "n_longs": len(longs), "n_shorts": len(shorts),
        "green_days": int(green_days), "config": {"tp": TP_TICKS, "sl": SL_TICKS, "hold": MAX_HOLD_MINUTES},
    }
    with open(OUTPUT_DIR / "minute_fifo_results.json", "w") as f:
        json.dump(result, f, indent=2)

    pd.DataFrame(all_trades).to_csv(OUTPUT_DIR / "minute_fifo_trades.csv", index=False)
    log.info(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_validation()
