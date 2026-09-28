#!/usr/bin/env python3
"""HC #539 R1(b)+(c) + HC #540 R1 — Classical longer-horizon ES strategies on minute bars.

Lanes (b) classical minute-bar strategies and (c) calendar/time-of-day effects,
evaluated honestly with market-order execution costs, on BOTH:
  - OLD window: 2025-07-14 -> 2026-01-30 (~115 trading days, never tested before)
  - NEW window: 2026-02-23 -> 2026-04-14 (the prior 29-day OOT window, for comparability)
  - FULL sample: all 197 days

DATA: data/processed/mbo_minute_bars_v1/YYYYMMDD.parquet (RTH 13:30-21:00 UTC,
prices in TICKS, 1 tick = 0.25 ES index points = $12.50).

SIGNAL FAMILIES (grid PRE-REGISTERED here; no tuning on eval data):
  Lane B:
    B1 MA-cross momentum: EMA fast/slow {(5,20),(10,50),(20,100)}, trade cross events.
    B2 Breakout: close > rolling N-bar high / < N-bar low, N in {30,60,120}.
    B3 Mean-reversion: z-score of close vs rolling 60m mean/std, fade |z| > {1.5, 2.0, 2.5}.
    B4 Vol breakout (range expansion): bar range > k*ATR(30), k in {2.0, 3.0},
       directional close -> follow.
    Holds: {1, 5, 15, 30, 60} minutes + EOD. Non-overlapping trades per cell.
  Lane C:
    C1 Open-drive: sign of first-30-min return, enter 10:00 ET, exits {30m, 60m, EOD}.
       Magnitude filter: |first-30m ret| > {0, 8, 16} ticks.
    C2 Close reversion: at {15:00, 15:30} ET fade the day's open-to-now move, exit at close.
       Magnitude filter: |move| > {0, 16} ticks.
    C3 Overnight gap fade: gap = today RTH open - yesterday RTH close; fade if
       |gap| > {8, 20, 40} ticks; exits {30m, 60m, EOD}.

EXECUTION (screening pass — bar-sim, NOT canonical FIFO-MBO replay; any survivor
must be confirmed on canonical replay before any further claim):
  Signal at bar t close -> ENTER at bar t+1 OPEN (market). EXIT at open of bar
  t+1+hold (market), or at the 15:59 ET bar close for EOD.
  COST: market RT = 1.376 ticks = $17.20 (HC #74: $4.70 commission + 1 tick spread).
  NO passive fills anywhere (HC #539 R3).

WALK-FORWARD (HC #0 SLIDING): in addition to per-cell fixed-rule stats (each cell
is itself a fixed pre-registered rule), a WF meta-layer per family x hold picks
the param set with the best trailing 60-day per-day Sharpe (sliding window) and
trades the NEXT day with it -> one honest OOS stream per family x hold (days 61+).
This addresses the multiple-testing concern of reading the best grid cell.

REPORTING (HC #428 R1): per cell per window: n_trades, n_days, net ticks/trade,
net $/trade, PF, WR (trade-weighted), per-day Sharpe (ann sqrt(252)), Sortino,
max drawdown ($, on daily cum PnL), green/red/flat day stratification (ES RTH
close-to-close +-0.10%), regime gap, day concentration.
GATES: pdSharpe>1.5, PF>1.4, WR>0.55, regime_gap<=0.50, day_conc<=0.70,
n_days>=20, n_trades>=50.

OUTPUTS: output/long_horizon_classical_v1/{cells_<window>.csv, wf_meta_<window>.csv,
per_day_<window>.csv, summary.json}, MLflow exp long_horizon_classical_v1
(Jupiter http://localhost:5000, best-effort).
"""
from __future__ import annotations
import glob, json, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
BARS_DIR = ROOT / "data/processed/mbo_minute_bars_v1"
OUT_DIR = ROOT / "output/long_horizon_classical_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_USD = 12.50
COST_RT_TICKS = 1.376  # market exec: $4.70 commission + 1 tick spread crossing
COST_RT_USD = COST_RT_TICKS * TICK_USD  # $17.20

RTH_END_UTC_MIN = 20 * 60  # 20:00 UTC = 16:00 ET strict RTH close
HOLDS_MIN = [1, 5, 15, 30, 60]  # plus "eod"

WINDOWS = {
    "old_jul25_jan26": ("20250714", "20260130"),
    "new_feb_apr26": ("20260223", "20260414"),
    "full": ("20250714", "20260429"),
}

GATES = dict(sharpe=1.5, pf=1.4, wr=0.55, regime_gap=0.50, day_conc=0.70,
             n_days=20, n_trades=50)

WF_LOOKBACK_DAYS = 60  # sliding (HC #0)
WF_MIN_TRADES = 30

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_all_days():
    days = {}
    for fp in sorted(glob.glob(str(BARS_DIR / "*.parquet"))):
        d = os.path.basename(fp)[:8]
        df = pd.read_parquet(fp, columns=["ts_minute", "open", "high", "low", "close", "volume"])
        if len(df) < 100:
            continue
        df = df.reset_index(drop=True)
        mins = df.ts_minute.dt.hour * 60 + df.ts_minute.dt.minute
        df["min_utc"] = mins
        # strict RTH: 13:30 <= t < 20:00 UTC
        df = df[(mins >= 13 * 60 + 30) & (mins < RTH_END_UTC_MIN)].reset_index(drop=True)
        if len(df) < 300:
            continue
        days[d] = df
    return days


def classify_days(days):
    """Green/red/flat by ES RTH close-to-close (+-0.10%)."""
    dates = sorted(days)
    closes = {d: float(days[d]["close"].iloc[-1]) for d in dates}
    regime = {}
    for i, d in enumerate(dates):
        if i == 0:
            regime[d] = "flat"
            continue
        prev = closes[dates[i - 1]]
        ret = (closes[d] - prev) / prev
        regime[d] = "green" if ret > 0.0010 else ("red" if ret < -0.0010 else "flat")
    return regime

# ----------------------------------------------------------------------------
# Signal generators: return integer array sig per bar in {-1,0,+1}
# (signal evaluated at bar close; entry at NEXT bar open)
# ----------------------------------------------------------------------------

def sig_ma_cross(df, fast, slow):
    c = df["close"]
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    above = (ef > es).astype(int)
    cross = above.diff().fillna(0)
    sig = np.zeros(len(df), dtype=int)
    sig[cross.values > 0] = 1
    sig[cross.values < 0] = -1
    sig[:slow] = 0  # warm-up
    return sig


def sig_breakout(df, n):
    c = df["close"].values
    hi = pd.Series(df["high"]).rolling(n).max().shift(1).values
    lo = pd.Series(df["low"]).rolling(n).min().shift(1).values
    sig = np.zeros(len(df), dtype=int)
    sig[c > hi] = 1
    sig[c < lo] = -1
    return sig


def sig_meanrev(df, zthr, lookback=60):
    c = df["close"]
    m = c.rolling(lookback).mean()
    s = c.rolling(lookback).std()
    z = ((c - m) / s).values
    sig = np.zeros(len(df), dtype=int)
    sig[z > zthr] = -1   # fade up-move
    sig[z < -zthr] = 1   # fade down-move
    return sig


def sig_volbreak(df, k, atr_n=30):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    o = df["open"].values
    rng = h - l
    atr = pd.Series(rng).rolling(atr_n).mean().shift(1).values
    expand = rng > k * atr
    sig = np.zeros(len(df), dtype=int)
    up = expand & (c > o)
    dn = expand & (c < o)
    sig[up] = 1
    sig[dn] = -1
    return sig

LANE_B = []
for f, s in [(5, 20), (10, 50), (20, 100)]:
    LANE_B.append((f"B1_macross_{f}_{s}", lambda df, f=f, s=s: sig_ma_cross(df, f, s)))
for n in [30, 60, 120]:
    LANE_B.append((f"B2_breakout_{n}", lambda df, n=n: sig_breakout(df, n)))
for z in [1.5, 2.0, 2.5]:
    LANE_B.append((f"B3_meanrev_z{z}", lambda df, z=z: sig_meanrev(df, z)))
for k in [2.0, 3.0]:
    LANE_B.append((f"B4_volbreak_k{k}", lambda df, k=k: sig_volbreak(df, k)))

# ----------------------------------------------------------------------------
# Trade simulation (per day, non-overlapping)
# ----------------------------------------------------------------------------

def sim_day(df, sig, hold):
    """hold: int minutes or 'eod'. Entry next-bar open, exit at open of
    entry+hold bar (or last-bar close for eod). Returns list of (side, net_ticks)."""
    o = df["open"].values
    c = df["close"].values
    n = len(df)
    last_close = c[-1]
    trades = []
    i = 0
    while i < n - 2:
        s = sig[i]
        if s == 0:
            i += 1
            continue
        e_idx = i + 1
        entry = o[e_idx]
        if hold == "eod":
            exit_px = last_close
            x_idx = n - 1
        else:
            x_idx = e_idx + hold
            if x_idx >= n:
                exit_px = last_close
                x_idx = n - 1
            else:
                exit_px = o[x_idx]
        gross = s * (exit_px - entry)  # already in ticks
        trades.append((s, gross - COST_RT_TICKS))
        i = x_idx  # non-overlapping: next signal considered after exit
    return trades

# Lane C — day-level strategies ------------------------------------------------

def lane_c_trades(dates, days, prev_close_map):
    """Returns dict cell_name -> {date: [(side, net_ticks), ...]}"""
    out = {}

    def add(cell, d, t):
        out.setdefault(cell, {}).setdefault(d, []).append(t)

    for d in dates:
        df = days[d]
        o = df["open"].values
        c = df["close"].values
        mins = df["min_utc"].values
        n = len(df)
        last_close = c[-1]
        # index of 14:00 UTC (10:00 ET) bar
        idx10 = np.searchsorted(mins, 14 * 60)
        if idx10 >= n - 5:
            continue
        first30 = c[idx10 - 1] - o[0]  # ticks, 9:30->10:00 move
        # C1 open-drive
        for mag in [0, 8, 16]:
            if abs(first30) > mag:
                side = 1 if first30 > 0 else -1
                entry = o[idx10]
                for ex_name, ex_min in [("30m", 14 * 60 + 30), ("60m", 15 * 60), ("eod", None)]:
                    if ex_min is None:
                        exit_px = last_close
                    else:
                        xi = np.searchsorted(mins, ex_min)
                        exit_px = o[xi] if xi < n else last_close
                    add(f"C1_opendrive_mag{mag}_{ex_name}", d,
                        (side, side * (exit_px - entry) - COST_RT_TICKS))
        # C2 close reversion: fade open->now move at 15:00 / 15:30 ET (19:00/19:30 UTC)
        for et, label in [(19 * 60, "1500"), (19 * 60 + 30, "1530")]:
            xi = np.searchsorted(mins, et)
            if xi >= n - 2:
                continue
            move = c[xi - 1] - o[0]
            for mag in [0, 16]:
                if abs(move) > mag:
                    side = -1 if move > 0 else 1
                    entry = o[xi]
                    add(f"C2_closerev_{label}_mag{mag}", d,
                        (side, side * (last_close - entry) - COST_RT_TICKS))
        # C3 overnight gap fade
        pc = prev_close_map.get(d)
        if pc is not None:
            gap = o[0] - pc
            for mag in [8, 20, 40]:
                if abs(gap) > mag:
                    side = -1 if gap > 0 else 1
                    entry = o[1] if n > 1 else o[0]  # enter at 9:31 open
                    for ex_name, ex_min in [("30m", 14 * 60), ("60m", 14 * 60 + 30), ("eod", None)]:
                        if ex_min is None:
                            exit_px = last_close
                        else:
                            xi = np.searchsorted(mins, ex_min)
                            exit_px = o[xi] if xi < n else last_close
                        add(f"C3_gapfade_mag{mag}_{ex_name}", d,
                            (side, side * (exit_px - entry) - COST_RT_TICKS))
    return out

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def cell_metrics(day_trades, regime, dates_in_window):
    """day_trades: {date: [(side, net_ticks)]}. Returns metrics dict or None."""
    rows = [(d, s, t) for d, lst in day_trades.items() if d in dates_in_window for (s, t) in lst]
    if not rows:
        return None
    tk = np.array([r[2] for r in rows])
    n_trades = len(tk)
    daily = {}
    for d, s, t in rows:
        daily[d] = daily.get(d, 0.0) + t
    # include zero days? Per-day Sharpe over TRADED days (consistent w/ prior runs)
    dvals = np.array(list(daily.values()))
    n_days = len(dvals)
    mean_d, std_d = dvals.mean(), dvals.std(ddof=1) if n_days > 1 else np.nan
    sharpe = (mean_d / std_d * np.sqrt(252)) if (std_d and std_d > 0) else np.nan
    downside = dvals[dvals < 0]
    dstd = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (mean_d / dstd * np.sqrt(252)) if (dstd and dstd > 0) else np.nan
    wins = tk[tk > 0].sum()
    losses = -tk[tk <= 0].sum()
    pf = wins / losses if losses > 0 else np.inf
    wr = (tk > 0).mean()
    total = tk.sum()
    # day concentration: largest positive day / total (if total>0)
    day_conc = (dvals.max() / total) if total > 0 and dvals.max() > 0 else np.nan
    # regime stratification
    strat = {}
    for rg in ("green", "red", "flat"):
        rv = np.array([v for d, v in daily.items() if regime.get(d) == rg])
        if len(rv) > 1 and rv.std(ddof=1) > 0:
            strat[rg] = float(rv.mean() / rv.std(ddof=1) * np.sqrt(252))
        else:
            strat[rg] = np.nan
        strat[rg + "_ndays"] = int(len(rv))
    sg, sr = strat["green"], strat["red"]
    if np.isfinite(sg) and np.isfinite(sr) and max(abs(sg), abs(sr)) > 0:
        regime_gap = abs(sg - sr) / max(abs(sg), abs(sr))
    else:
        regime_gap = np.nan
    # max drawdown on daily cum pnl ($)
    cum = np.cumsum([daily[d] for d in sorted(daily)]) * TICK_USD
    peak = np.maximum.accumulate(cum)
    mdd = float((cum - peak).min()) if len(cum) else np.nan
    m = dict(
        n_trades=n_trades, n_days=n_days,
        net_ticks_per_trade=float(tk.mean()),
        net_usd_per_trade=float(tk.mean() * TICK_USD),
        total_net_usd=float(total * TICK_USD),
        pf=float(pf) if np.isfinite(pf) else 999.0,
        wr=float(wr),
        day_sharpe=float(sharpe) if np.isfinite(sharpe) else np.nan,
        sortino=float(sortino) if np.isfinite(sortino) else np.nan,
        max_dd_usd=mdd,
        day_conc=float(day_conc) if np.isfinite(day_conc) else np.nan,
        sharpe_green=strat["green"], sharpe_red=strat["red"], sharpe_flat=strat["flat"],
        ndays_green=strat["green_ndays"], ndays_red=strat["red_ndays"], ndays_flat=strat["flat_ndays"],
        regime_gap=float(regime_gap) if np.isfinite(regime_gap) else np.nan,
        mean_daily_usd=float(mean_d * TICK_USD),
    )
    g = GATES
    m["pass_all_gates"] = bool(
        np.isfinite(m["day_sharpe"]) and m["day_sharpe"] > g["sharpe"]
        and m["pf"] > g["pf"] and m["wr"] > g["wr"]
        and np.isfinite(m["regime_gap"]) and m["regime_gap"] <= g["regime_gap"]
        and np.isfinite(m["day_conc"]) and m["day_conc"] <= g["day_conc"]
        and m["n_days"] >= g["n_days"] and m["n_trades"] >= g["n_trades"]
    )
    m["reject_day_conc"] = bool(np.isfinite(m["day_conc"]) and m["day_conc"] > g["day_conc"])
    m["reject_regime_gap"] = bool(np.isfinite(m["regime_gap"]) and m["regime_gap"] > g["regime_gap"])
    return m

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("Loading minute bars ...", flush=True)
    days = load_all_days()
    dates = sorted(days)
    print(f"  {len(dates)} days: {dates[0]} -> {dates[-1]}  ({time.time()-t0:.0f}s)", flush=True)
    regime = classify_days(days)
    prev_close_map = {}
    for i, d in enumerate(dates):
        if i > 0:
            prev_close_map[d] = float(days[dates[i - 1]]["close"].iloc[-1])

    # ---- Lane B: per-cell trades over all days ----
    # cell key: (strat_name, hold_label) -> {date: [(side, net)]}
    all_cells = {}
    print("Simulating Lane B ...", flush=True)
    for d in dates:
        df = days[d]
        for name, fn in LANE_B:
            sig = fn(df)
            for hold in HOLDS_MIN + ["eod"]:
                trades = sim_day(df, sig, hold)
                if trades:
                    key = (name, str(hold))
                    all_cells.setdefault(key, {}).setdefault(d, []).extend(trades)
    print(f"  lane B done ({time.time()-t0:.0f}s)", flush=True)

    # ---- Lane C ----
    print("Simulating Lane C ...", flush=True)
    lane_c = lane_c_trades(dates, days, prev_close_map)
    for cell, dt in lane_c.items():
        all_cells[(cell, "fixed")] = dt
    print(f"  lane C done, total cells={len(all_cells)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- Per-window cell metrics ----
    summary = {"data": {"n_days": len(dates), "first": dates[0], "last": dates[-1],
                        "bars_source": str(BARS_DIR),
                        "older_2024_2025_data": "NONE on Jupiter or Neptune — earliest ES MBO on disk is 2025-07-14",
                        "execution": "bar-sim screening pass (next-bar-open market entry/exit), NOT canonical FIFO-MBO replay",
                        "cost_rt_ticks": COST_RT_TICKS, "cost_rt_usd": COST_RT_USD},
               "gates": GATES, "windows": {}}
    for wname, (d0, d1) in WINDOWS.items():
        win_dates = set(d for d in dates if d0 <= d <= d1)
        rows = []
        for (name, hold), dt in sorted(all_cells.items()):
            m = cell_metrics(dt, regime, win_dates)
            if m is None:
                continue
            m.update(strategy=name, hold=hold, window=wname)
            rows.append(m)
        cdf = pd.DataFrame(rows)
        cdf.to_csv(OUT_DIR / f"cells_{wname}.csv", index=False)
        n_pass = int(cdf["pass_all_gates"].sum()) if len(cdf) else 0
        # regime mix
        mix = {rg: sum(1 for d in win_dates if regime[d] == rg) for rg in ("green", "red", "flat")}
        top = (cdf[cdf.n_trades >= GATES["n_trades"]]
               .sort_values("day_sharpe", ascending=False).head(5)
               [["strategy", "hold", "n_trades", "n_days", "net_ticks_per_trade",
                 "net_usd_per_trade", "pf", "wr", "day_sharpe", "regime_gap", "day_conc"]]
               .to_dict("records")) if len(cdf) else []
        summary["windows"][wname] = dict(
            n_days=len(win_dates), day_mix=mix, n_cells=len(cdf),
            n_pass_all_gates=n_pass, top5_by_day_sharpe=top)
        print(f"  window {wname}: {len(win_dates)} days {mix}, {len(cdf)} cells, {n_pass} pass all gates", flush=True)

    # ---- Walk-forward meta-layer (sliding 60d) per family x hold, Lane B ----
    print("Walk-forward meta-selection (sliding 60d) ...", flush=True)
    families = {"B1": [n for n, _ in LANE_B if n.startswith("B1")],
                "B2": [n for n, _ in LANE_B if n.startswith("B2")],
                "B3": [n for n, _ in LANE_B if n.startswith("B3")],
                "B4": [n for n, _ in LANE_B if n.startswith("B4")]}
    # precompute per-cell per-day pnl
    cell_day_pnl = {}
    cell_day_trades_n = {}
    for (name, hold), dt in all_cells.items():
        if hold == "fixed":
            continue
        cell_day_pnl[(name, hold)] = {d: sum(t for _, t in lst) for d, lst in dt.items()}
        cell_day_trades_n[(name, hold)] = {d: len(lst) for d, lst in dt.items()}
    wf_rows = []
    wf_daily = {}  # (family, hold) -> {date: pnl}
    for fam, members in families.items():
        for hold in [str(h) for h in HOLDS_MIN] + ["eod"]:
            cands = [(m, hold) for m in members if (m, hold) in cell_day_pnl]
            if not cands:
                continue
            stream = {}
            for i in range(WF_LOOKBACK_DAYS, len(dates)):
                lb = dates[i - WF_LOOKBACK_DAYS:i]
                best, best_sh = None, -np.inf
                for ck in cands:
                    pnl = [cell_day_pnl[ck].get(d, 0.0) for d in lb]
                    ntr = sum(cell_day_trades_n[ck].get(d, 0) for d in lb)
                    if ntr < WF_MIN_TRADES:
                        continue
                    arr = np.array(pnl)
                    sd = arr.std(ddof=1)
                    sh = arr.mean() / sd if sd > 0 else -np.inf
                    if sh > best_sh:
                        best_sh, best = sh, ck
                d = dates[i]
                if best is not None and d in cell_day_pnl[best]:
                    stream[d] = cell_day_pnl[best][d]
            if stream:
                wf_daily[(fam, hold)] = stream
    for (fam, hold), stream in wf_daily.items():
        for wname, (d0, d1) in WINDOWS.items():
            win = {d: v for d, v in stream.items() if d0 <= d <= d1}
            if len(win) < 10:
                continue
            vals = np.array(list(win.values()))
            sd = vals.std(ddof=1)
            sh = vals.mean() / sd * np.sqrt(252) if sd > 0 else np.nan
            wins = vals[vals > 0].sum(); losses = -vals[vals <= 0].sum()
            wf_rows.append(dict(family=fam, hold=hold, window=wname, n_days=len(win),
                                mean_daily_usd=float(vals.mean() * TICK_USD),
                                day_sharpe=float(sh) if np.isfinite(sh) else np.nan,
                                pf=float(wins / losses) if losses > 0 else 999.0,
                                pos_day_frac=float((vals > 0).mean()),
                                total_net_usd=float(vals.sum() * TICK_USD)))
    wfdf = pd.DataFrame(wf_rows)
    wfdf.to_csv(OUT_DIR / "wf_meta.csv", index=False)
    best_wf = (wfdf[wfdf.window == "full"].sort_values("day_sharpe", ascending=False)
               .head(5).to_dict("records")) if len(wfdf) else []
    summary["wf_meta_top5_full"] = best_wf
    n_wf_pass = int((wfdf.day_sharpe > GATES["sharpe"]).sum()) if len(wfdf) else 0
    summary["wf_meta_n_sharpe_gt_1p5"] = n_wf_pass

    # per-day breakdown for top cells (full window)
    cdf_full = pd.read_csv(OUT_DIR / "cells_full.csv")
    top_keys = (cdf_full[cdf_full.n_trades >= GATES["n_trades"]]
                .sort_values("day_sharpe", ascending=False).head(10)[["strategy", "hold"]]
                .itertuples(index=False))
    pd_rows = []
    for name, hold in top_keys:
        dt = all_cells.get((name, str(hold))) or all_cells.get((name, hold))
        if not dt:
            continue
        for d, lst in sorted(dt.items()):
            pd_rows.append(dict(strategy=name, hold=hold, date=d, regime=regime[d],
                                n_trades=len(lst),
                                day_net_usd=sum(t for _, t in lst) * TICK_USD))
    pd.DataFrame(pd_rows).to_csv(OUT_DIR / "per_day_top_cells.csv", index=False)

    # verdict
    any_pass = any(w["n_pass_all_gates"] > 0 for w in summary["windows"].values())
    summary["verdict"] = {
        "any_cell_passes_all_gates": any_pass,
        "honest_caveats": [
            "bar-sim screening pass; survivors require canonical FIFO-MBO confirmation",
            "grid cells are pre-registered but reading the best of ~%d cells per window is implicit multiple testing; WF meta-layer is the honest selection-free stream" % len(all_cells),
            "no 2024-2025 (pre-Jul-2025) ES data exists on disk; HC #539 R2 older-window request satisfied only back to Jul 2025",
            "old window day-mix vs new window day-mix reported in summary for regime-tilt context",
        ],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in summary["windows"].items()}, indent=2, default=str)[:2000])
    print(f"DONE in {time.time()-t0:.0f}s -> {OUT_DIR}", flush=True)

    # MLflow best-effort
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("long_horizon_classical_v1")
        with mlflow.start_run(run_name="classical_lanes_BC_v1"):
            mlflow.log_params(dict(n_days=len(dates), cost_rt_ticks=COST_RT_TICKS,
                                   n_cells=len(all_cells), wf_lookback=WF_LOOKBACK_DAYS))
            for wname, w in summary["windows"].items():
                mlflow.log_metric(f"n_pass_{wname}", w["n_pass_all_gates"])
                if w["top5_by_day_sharpe"]:
                    mlflow.log_metric(f"best_day_sharpe_{wname}", w["top5_by_day_sharpe"][0]["day_sharpe"])
            mlflow.log_artifacts(str(OUT_DIR))
        print("MLflow logged.", flush=True)
    except Exception as e:
        print(f"MLflow skip: {e}", flush=True)


if __name__ == "__main__":
    main()
