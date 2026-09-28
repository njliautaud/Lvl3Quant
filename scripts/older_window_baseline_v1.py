#!/usr/bin/env python3
"""HC #540 — Older-window simple baseline (July 2025 – Jan 2026).

GOAL: settle window-vs-strategy question.
- Run the SAME 3 non-ML baselines from simple_baseline_v1 on the OLDER ES MBO
  window (July 2025 – Jan 2026, ~125 trading days).
- v3.4.2 ML is NOT tested here (window is IN-SAMPLE for v3.4.2 → lookahead bias).
- Compare cell-by-cell to Feb-Apr 2026 simple_baseline_v1 results.

If older window passes gates → Feb-Apr 2026 is anomalous, strategies are real.
If older window also fails → strategies are dead, escalate asset-class pivot.

Front-month detection: pick the instrument_id with the highest trade volume
on the day (auto-detected per file, no external cache needed).

Cost model + signal definitions IDENTICAL to simple_baseline_v1.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import math
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import databento as db
from sortedcontainers import SortedDict

ROOT = Path("/home/jupiter/Lvl3Quant")
RAW_DIR = ROOT / "data/raw/mbo"
OUT_DIR = ROOT / "output/older_window_baseline_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FRONT_CACHE_DIR = OUT_DIR / "front_cache"
FRONT_CACHE_DIR.mkdir(exist_ok=True)
GRID_CACHE_DIR = OUT_DIR / "grid_cache"
GRID_CACHE_DIR.mkdir(exist_ok=True)

# Cost model (round-trip, ES-tick units) — same as simple_baseline_v1
COSTS = {
    "ES_mkt":  1.376,
    "ES_psv":  0.376,
    "MES_mkt": 2.20,
    "MES_psv": 1.20,
}
PASSIVE_FILL_PROB = 0.50
TICK_DOLLARS_ES = 12.50

# MBO action / side bytes
A_ADD, A_CANCEL, A_MODIFY, A_TRADE, A_FILL = ord('A'), ord('C'), ord('M'), ord('T'), ord('F')
S_BID, S_ASK = ord('B'), ord('A')

# Grid
GRID_NS = 250_000_000
ONE_S_NS = 1_000_000_000
HORIZONS_NS = {"1s": 1_000_000_000, "5s": 5_000_000_000, "30s": 30_000_000_000}
OFI_WIN_NS = 1_000_000_000
TRADE_FLOW_WIN_NS = 500_000_000
ZSCORE_WIN_NS = 30_000_000_000


def gen_older_dates() -> List[str]:
    start = pd.Timestamp("2025-07-14")
    end = pd.Timestamp("2026-01-30")
    out = []
    for d in pd.date_range(start, end, freq="B"):
        ds = d.strftime("%Y%m%d")
        if (RAW_DIR / f"glbx-mdp3-{ds}.mbo.dbn.zst").exists():
            out.append(ds)
    return out


def rth_window_ns(date_str: str) -> Tuple[int, int]:
    d = pd.Timestamp(date_str)
    open_et = pd.Timestamp(f"{d.strftime('%Y-%m-%d')} 09:30:00", tz="America/New_York")
    close_et = pd.Timestamp(f"{d.strftime('%Y-%m-%d')} 16:00:00", tz="America/New_York")
    return int(open_et.value), int(close_et.value)


def detect_front_id(date_str: str, arr=None) -> Optional[int]:
    """Detect front-month instrument by trade volume during the day.
    Cached to disk for re-runs.
    """
    cache_p = FRONT_CACHE_DIR / f"{date_str}_front.json"
    if cache_p.exists():
        try:
            d = json.loads(cache_p.read_text())
            return int(d["front_id"])
        except Exception:
            pass
    if arr is None:
        dbn = RAW_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
        if not dbn.exists():
            return None
        store = db.DBNStore.from_file(str(dbn))
        arr = store.to_ndarray()
    action = arr["action"].view(np.uint8)
    trade_mask = action == A_TRADE
    if trade_mask.sum() == 0:
        return None
    iid = arr["instrument_id"][trade_mask]
    sz = arr["size"][trade_mask].astype(np.int64)
    # Aggregate volume per instrument
    uniq, inv = np.unique(iid, return_inverse=True)
    vol = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(vol, inv, sz)
    front = int(uniq[int(np.argmax(vol))])
    top_vol = int(vol.max())
    cache_p.write_text(json.dumps({"front_id": front, "top_vol": top_vol,
                                   "date": date_str}))
    return front


def build_day_grid(date_str: str, verbose: bool = False) -> Optional[Dict]:
    dbn = RAW_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    if not dbn.exists():
        return None
    rth_open, rth_close = rth_window_ns(date_str)

    t0 = time.time()
    store = db.DBNStore.from_file(str(dbn))
    arr = store.to_ndarray()

    front_id = detect_front_id(date_str, arr=arr)
    if front_id is None:
        return None

    iid_all = arr["instrument_id"]
    sel = (iid_all == front_id)
    arr = arr[sel]

    ts = arr["ts_event"].astype(np.int64)
    action = arr["action"].view(np.uint8)
    side = arr["side"].view(np.uint8)
    px = arr["price"].astype(np.int64)
    sz = arr["size"].astype(np.int64)
    oid = arr["order_id"].astype(np.int64)

    bids = SortedDict()
    asks = SortedDict()
    orders: Dict[int, Tuple[int, int, int]] = {}

    grid_ts_start = rth_open
    n_grid = (rth_close - rth_open) // GRID_NS + 1
    g_mid = np.full(n_grid, np.nan, dtype=np.float64)
    g_bid_sz = np.zeros(n_grid, dtype=np.int64)
    g_ask_sz = np.zeros(n_grid, dtype=np.int64)
    g_bid_add_sz = np.zeros(n_grid, dtype=np.int64)
    g_bid_cancel_sz = np.zeros(n_grid, dtype=np.int64)
    g_ask_add_sz = np.zeros(n_grid, dtype=np.int64)
    g_ask_cancel_sz = np.zeros(n_grid, dtype=np.int64)
    g_buy_trade_sz = np.zeros(n_grid, dtype=np.int64)
    g_sell_trade_sz = np.zeros(n_grid, dtype=np.int64)

    N = len(arr)
    next_grid_t = grid_ts_start
    gi = 0
    for i in range(N):
        t = ts[i]
        if t < rth_open:
            a = action[i]; s = side[i]; p = px[i]; q = sz[i]; o = oid[i]
            if a == A_ADD and (s == S_BID or s == S_ASK):
                book = bids if s == S_BID else asks
                book[p] = book.get(p, 0) + q
                orders[o] = (s, p, q)
            elif a == A_CANCEL:
                if o in orders:
                    so, po, qo = orders[o]
                    book = bids if so == S_BID else asks
                    cur = book.get(po, 0) - q
                    if cur <= 0:
                        if po in book: del book[po]
                    else:
                        book[po] = cur
                    new_q = qo - q
                    if new_q <= 0:
                        del orders[o]
                    else:
                        orders[o] = (so, po, new_q)
            elif a == A_MODIFY:
                if o in orders:
                    so, po, qo = orders[o]
                    book_old = bids if so == S_BID else asks
                    cur_old = book_old.get(po, 0) - qo
                    if cur_old <= 0:
                        if po in book_old: del book_old[po]
                    else:
                        book_old[po] = cur_old
                if s == S_BID or s == S_ASK:
                    book_new = bids if s == S_BID else asks
                    book_new[p] = book_new.get(p, 0) + q
                    orders[o] = (s, p, q)
            elif a == A_FILL or a == A_TRADE:
                if a == A_FILL and o in orders:
                    so, po, qo = orders[o]
                    book = bids if so == S_BID else asks
                    cur = book.get(po, 0) - q
                    if cur <= 0:
                        if po in book: del book[po]
                    else:
                        book[po] = cur
                    new_q = qo - q
                    if new_q <= 0:
                        del orders[o]
                    else:
                        orders[o] = (so, po, new_q)
            continue
        if t > rth_close:
            break

        while t >= next_grid_t and gi < n_grid:
            if len(bids) > 0 and len(asks) > 0:
                bb_p, bb_s = bids.peekitem(-1)
                aa_p, aa_s = asks.peekitem(0)
                g_mid[gi] = (bb_p + aa_p) / 2.0 / 1e9
                g_bid_sz[gi] = bb_s
                g_ask_sz[gi] = aa_s
            gi += 1
            next_grid_t += GRID_NS

        a = action[i]; s = side[i]; p = px[i]; q = sz[i]; o = oid[i]
        bucket = (t - rth_open) // GRID_NS
        if 0 <= bucket < n_grid:
            if a == A_ADD:
                if s == S_BID:
                    bids[p] = bids.get(p, 0) + q
                    orders[o] = (s, p, q)
                    g_bid_add_sz[bucket] += q
                elif s == S_ASK:
                    asks[p] = asks.get(p, 0) + q
                    orders[o] = (s, p, q)
                    g_ask_add_sz[bucket] += q
            elif a == A_CANCEL:
                if o in orders:
                    so, po, qo = orders[o]
                    book = bids if so == S_BID else asks
                    cur = book.get(po, 0) - q
                    if cur <= 0:
                        if po in book: del book[po]
                    else:
                        book[po] = cur
                    new_q = qo - q
                    if new_q <= 0:
                        del orders[o]
                    else:
                        orders[o] = (so, po, new_q)
                    if so == S_BID:
                        g_bid_cancel_sz[bucket] += q
                    else:
                        g_ask_cancel_sz[bucket] += q
            elif a == A_MODIFY:
                if o in orders:
                    so, po, qo = orders[o]
                    book_old = bids if so == S_BID else asks
                    cur_old = book_old.get(po, 0) - qo
                    if cur_old <= 0:
                        if po in book_old: del book_old[po]
                    else:
                        book_old[po] = cur_old
                if s == S_BID:
                    bids[p] = bids.get(p, 0) + q
                    orders[o] = (s, p, q)
                elif s == S_ASK:
                    asks[p] = asks.get(p, 0) + q
                    orders[o] = (s, p, q)
            elif a == A_TRADE:
                if s == S_ASK:
                    g_sell_trade_sz[bucket] += q
                elif s == S_BID:
                    g_buy_trade_sz[bucket] += q
            elif a == A_FILL:
                if o in orders:
                    so, po, qo = orders[o]
                    book = bids if so == S_BID else asks
                    cur = book.get(po, 0) - q
                    if cur <= 0:
                        if po in book: del book[po]
                    else:
                        book[po] = cur
                    new_q = qo - q
                    if new_q <= 0:
                        del orders[o]
                    else:
                        orders[o] = (so, po, new_q)

    while gi < n_grid:
        if len(bids) > 0 and len(asks) > 0:
            bb_p, bb_s = bids.peekitem(-1)
            aa_p, aa_s = asks.peekitem(0)
            g_mid[gi] = (bb_p + aa_p) / 2.0 / 1e9
            g_bid_sz[gi] = bb_s
            g_ask_sz[gi] = aa_s
        gi += 1

    elapsed = time.time() - t0

    w_1s = OFI_WIN_NS // GRID_NS
    w_500ms = TRADE_FLOW_WIN_NS // GRID_NS
    w_30s = ZSCORE_WIN_NS // GRID_NS

    def rolling_sum(x, w):
        c = np.cumsum(np.concatenate([[0], x]))
        out = np.zeros_like(x, dtype=np.float64)
        out[w-1:] = c[w:] - c[:-w]
        for i in range(min(w-1, len(x))):
            out[i] = c[i+1]
        return out

    bid_add_1s = rolling_sum(g_bid_add_sz, w_1s)
    bid_cnl_1s = rolling_sum(g_bid_cancel_sz, w_1s)
    ask_add_1s = rolling_sum(g_ask_add_sz, w_1s)
    ask_cnl_1s = rolling_sum(g_ask_cancel_sz, w_1s)
    ofi_1s = bid_add_1s - bid_cnl_1s - ask_add_1s + ask_cnl_1s

    buy_500ms = rolling_sum(g_buy_trade_sz, w_500ms)
    sell_500ms = rolling_sum(g_sell_trade_sz, w_500ms)
    trade_flow_500ms = buy_500ms - sell_500ms

    denom = (g_bid_sz + g_ask_sz).astype(np.float64)
    book_imb = np.where(denom > 0, (g_bid_sz - g_ask_sz) / np.maximum(denom, 1), 0.0)

    log_mid = np.log(np.maximum(g_mid, 1e-9))
    ret_1s = np.full(n_grid, np.nan, dtype=np.float64)
    ret_1s[w_1s:] = log_mid[w_1s:] - log_mid[:-w_1s]
    valid_ret = np.where(np.isfinite(ret_1s), ret_1s, 0.0)
    valid_mask = np.isfinite(ret_1s).astype(np.float64)
    csum = np.cumsum(np.concatenate([[0], valid_ret]))
    csum2 = np.cumsum(np.concatenate([[0], valid_ret**2]))
    cmask = np.cumsum(np.concatenate([[0], valid_mask]))
    zmean = np.zeros(n_grid)
    zstd = np.ones(n_grid)
    for i in range(n_grid):
        lo = max(0, i - w_30s + 1)
        n = cmask[i+1] - cmask[lo]
        if n < 10:
            zmean[i] = 0.0
            zstd[i] = 1.0
            continue
        s = csum[i+1] - csum[lo]
        s2 = csum2[i+1] - csum2[lo]
        m = s / n
        v = max(s2/n - m*m, 1e-20)
        zmean[i] = m
        zstd[i] = math.sqrt(v)
    z_ret = (ret_1s - zmean) / np.where(zstd > 0, zstd, 1.0)
    z_ret = np.where(np.isfinite(z_ret), z_ret, 0.0)

    g_ts = grid_ts_start + np.arange(n_grid, dtype=np.int64) * GRID_NS
    mid_filled = pd.Series(g_mid).ffill().bfill().values

    if verbose:
        print(f"  {date_str}: replay {elapsed:.1f}s front_id={front_id} valid_mid={np.isfinite(g_mid).mean():.3f}",
              flush=True)

    return {
        "date": date_str,
        "ts": g_ts,
        "mid": mid_filled,
        "ofi_1s": ofi_1s,
        "book_imb": book_imb,
        "trade_flow_500ms": trade_flow_500ms,
        "z_ret_1s": z_ret,
        "elapsed_s": elapsed,
        "valid_mid_pct": float(np.isfinite(g_mid).mean()),
        "front_id": front_id,
    }


def classify_es_regime(date_str: str, day_grid: Dict) -> str:
    mid = day_grid["mid"]
    valid = np.isfinite(mid)
    if valid.sum() < 100:
        return "flat"
    open_mid = mid[valid][0]
    close_mid = mid[valid][-1]
    move = (close_mid - open_mid) / open_mid
    if move > 0.0015:
        return "green"
    if move < -0.0015:
        return "red"
    return "flat"


def evaluate_signal(day_grids: List[Dict], signal_name: str, horizon_ns: int,
                    pct: float, side_choice: str, exec_mode: str,
                    horizon_label: str, pct_label: str) -> Tuple[Dict, List[Dict]]:
    cost_rt = COSTS[exec_mode]
    contract = "MES" if exec_mode.startswith("MES") else "ES"
    is_passive = exec_mode.endswith("_psv")

    per_day = []
    h_grid = horizon_ns // GRID_NS

    for dg in day_grids:
        mid = dg["mid"]
        n = len(mid)
        if signal_name == "A":
            sig_val = dg["ofi_1s"]
            sig_sign = np.sign(sig_val)
        elif signal_name == "B":
            book = dg["book_imb"]
            flow = dg["trade_flow_500ms"]
            fires = (np.sign(book) == np.sign(flow)) & (np.abs(book) > 0.3) & (flow != 0)
            sig_val = np.where(fires, book, 0.0)
            sig_sign = np.where(fires, np.sign(book), 0).astype(np.int64)
        elif signal_name == "C":
            z = dg["z_ret_1s"]
            fires = np.abs(z) > 2.0
            sig_val = np.where(fires, -z, 0.0)
            sig_sign = np.where(fires, -np.sign(z), 0).astype(np.int64)
        else:
            raise ValueError(signal_name)

        abs_sig = np.abs(sig_val)
        nz_mask = abs_sig > 0
        nz_vals = abs_sig[nz_mask]
        if len(nz_vals) == 0:
            per_day.append({"date": dg["date"], "n_trades": 0, "pnl_ticks": 0.0,
                            "pnl_dollars": 0.0, "wins": 0, "losses": 0,
                            "regime": classify_es_regime(dg["date"], dg)})
            continue
        thresh = np.quantile(nz_vals, 1.0 - pct)
        active = (abs_sig >= thresh) & (sig_sign != 0)

        if side_choice == "long":
            active = active & (sig_sign > 0)
        elif side_choice == "short":
            active = active & (sig_sign < 0)

        idx = np.where(active)[0]
        if len(idx) == 0:
            per_day.append({"date": dg["date"], "n_trades": 0, "pnl_ticks": 0.0,
                            "pnl_dollars": 0.0, "wins": 0, "losses": 0,
                            "regime": classify_es_regime(dg["date"], dg)})
            continue

        idx = idx[idx + h_grid < n]
        if len(idx) == 0:
            per_day.append({"date": dg["date"], "n_trades": 0, "pnl_ticks": 0.0,
                            "pnl_dollars": 0.0, "wins": 0, "losses": 0,
                            "regime": classify_es_regime(dg["date"], dg)})
            continue

        entry_mid = mid[idx]
        exit_mid = mid[idx + h_grid]
        side_vec = sig_sign[idx]
        pnl_pts = side_vec * (exit_mid - entry_mid)
        pnl_ticks = pnl_pts / 0.25
        pnl_ticks_net = pnl_ticks - cost_rt
        if is_passive:
            rng = np.random.default_rng(seed=int(dg["date"]))
            keep = rng.random(len(pnl_ticks_net)) < PASSIVE_FILL_PROB
            pnl_ticks_net = pnl_ticks_net[keep]
        if contract == "ES":
            pnl_dollars = pnl_ticks_net * TICK_DOLLARS_ES
        else:
            pnl_dollars = pnl_ticks_net * TICK_DOLLARS_ES / 5.0

        wins = int((pnl_ticks_net > 0).sum())
        losses = int((pnl_ticks_net < 0).sum())
        per_day.append({
            "date": dg["date"],
            "n_trades": int(len(pnl_ticks_net)),
            "pnl_ticks": float(pnl_ticks_net.sum()),
            "pnl_dollars": float(pnl_dollars.sum()),
            "wins": wins,
            "losses": losses,
            "regime": classify_es_regime(dg["date"], dg),
        })

    df = pd.DataFrame(per_day)
    df_nz = df[df["n_trades"] > 0]
    n_days_traded = len(df_nz)
    n_trades = int(df["n_trades"].sum())

    # Relaxed gates per HC #540: N days >= 50, N trades >= 200
    if n_days_traded < 5 or n_trades < 10:
        return ({
            "signal": signal_name, "horizon": horizon_label, "pct": pct_label,
            "side": side_choice, "exec": exec_mode,
            "n_days_traded": n_days_traded, "n_trades": n_trades,
            "pdShr": np.nan, "Sortino": np.nan, "PF": np.nan, "WR": np.nan,
            "pnl_dollars_total": float(df["pnl_dollars"].sum()),
            "pnl_dollars_per_day": float(df["pnl_dollars"].mean()),
            "Sharpe_green": np.nan, "Sharpe_red": np.nan, "Sharpe_flat": np.nan,
            "day_conc": np.nan, "pass_R1": False, "pass_gates": False,
        }, per_day)

    daily_pnl = df_nz["pnl_dollars"].values
    mean_p = daily_pnl.mean()
    std_p = daily_pnl.std(ddof=1) if len(daily_pnl) > 1 else 0.0
    pdShr = (mean_p / std_p * math.sqrt(252)) if std_p > 0 else np.nan
    downside = daily_pnl[daily_pnl < 0]
    dstd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = (mean_p / dstd * math.sqrt(252)) if dstd > 0 else np.nan
    total_wins = int(df["wins"].sum())
    total_losses = int(df["losses"].sum())
    wr = total_wins / max(total_wins + total_losses, 1)
    gross_w = df["pnl_dollars"].clip(lower=0).sum()
    gross_l = -df["pnl_dollars"].clip(upper=0).sum()
    pf = gross_w / gross_l if gross_l > 0 else np.nan

    abs_p = np.abs(daily_pnl)
    day_conc = (abs_p.max() / abs_p.sum()) if abs_p.sum() > 0 else 0.0

    def sharpe_subset(sub):
        if len(sub) < 3:
            return np.nan
        m = sub.mean(); s = sub.std(ddof=1)
        return (m/s*math.sqrt(252)) if s > 0 else np.nan
    s_green = sharpe_subset(df_nz[df_nz["regime"] == "green"]["pnl_dollars"].values)
    s_red = sharpe_subset(df_nz[df_nz["regime"] == "red"]["pnl_dollars"].values)
    s_flat = sharpe_subset(df_nz[df_nz["regime"] == "flat"]["pnl_dollars"].values)

    if np.isfinite(s_green) and np.isfinite(s_red):
        denom_s = max(abs(s_green), abs(s_red))
        regime_skew = abs(s_green - s_red) / denom_s if denom_s > 0 else 0.0
        pass_R1 = (regime_skew <= 0.50) and (day_conc <= 0.70)
    else:
        pass_R1 = False
    if side_choice == "short":
        traded_regimes = df_nz["regime"].unique().tolist()
        if set(traded_regimes) <= {"red"}:
            pass_R1 = False

    # HC #540 relaxed gates
    pass_basic = (n_days_traded >= 50) and (n_trades >= 200)
    pass_perf = (np.isfinite(pdShr) and pdShr > 1.5) and (np.isfinite(pf) and pf > 1.4) and (wr > 0.55)
    pass_gates = pass_basic and pass_perf and pass_R1

    return ({
        "signal": signal_name, "horizon": horizon_label, "pct": pct_label,
        "side": side_choice, "exec": exec_mode,
        "n_days_traded": n_days_traded, "n_trades": n_trades,
        "pdShr": pdShr, "Sortino": sortino, "PF": pf, "WR": wr,
        "pnl_dollars_total": float(df["pnl_dollars"].sum()),
        "pnl_dollars_per_day": float(df["pnl_dollars"].mean()),
        "Sharpe_green": s_green, "Sharpe_red": s_red, "Sharpe_flat": s_flat,
        "day_conc": float(day_conc), "pass_R1": bool(pass_R1),
        "pass_gates": bool(pass_gates),
    }, per_day)


def build_comparison(older_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Side-by-side: same cells, this run vs simple_baseline_v1 (Feb-Apr 2026)."""
    prior_path = ROOT / "output/simple_baseline_v1/results_table.csv"
    if not prior_path.exists():
        return None
    prior = pd.read_csv(prior_path)
    keys = ["signal", "horizon", "pct", "side", "exec"]
    suf_o = "_older"
    suf_p = "_febapr"
    merged = older_df.merge(prior, on=keys, suffixes=(suf_o, suf_p), how="left")
    sub = merged[keys + [f"pdShr{suf_o}", f"pdShr{suf_p}",
                         f"PF{suf_o}", f"PF{suf_p}",
                         f"WR{suf_o}", f"WR{suf_p}",
                         f"n_trades{suf_o}", f"n_trades{suf_p}",
                         f"pass_gates{suf_o}", f"pass_gates{suf_p}"]].copy()
    # Sign-reversal flag (on pdShr where both finite)
    def reversed_(r):
        a = r[f"pdShr{suf_o}"]; b = r[f"pdShr{suf_p}"]
        if pd.isna(a) or pd.isna(b):
            return False
        return (a > 0) != (b > 0)
    sub["sign_reversed"] = sub.apply(reversed_, axis=1)
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=None)
    ap.add_argument("--limit-days", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cache-grids", action="store_true", default=True)
    ap.add_argument("--max-build-time-min", type=float, default=None,
                    help="abort grid build phase after N minutes, sweep with what we have")
    args = ap.parse_args()

    dates = args.dates if args.dates else gen_older_dates()
    if args.limit_days:
        dates = dates[:args.limit_days]

    print(f"[start] {len(dates)} dates  ({dates[0]} → {dates[-1]})", flush=True)

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("older_window_baseline_v1")
        mlflow_run = mlflow.start_run(run_name=f"older_baseline_{len(dates)}d")
        mlflow.log_param("n_dates", len(dates))
        mlflow.log_param("date_start", dates[0])
        mlflow.log_param("date_end", dates[-1])
    except Exception as e:
        print(f"[mlflow disabled] {e}", flush=True)
        mlflow = None
        mlflow_run = None

    day_grids = []
    build_start = time.time()
    for ds in dates:
        if args.max_build_time_min is not None:
            elapsed_min = (time.time() - build_start) / 60.0
            if elapsed_min > args.max_build_time_min:
                print(f"[build budget hit] {elapsed_min:.1f}min > {args.max_build_time_min}min, stopping early at {len(day_grids)} days",
                      flush=True)
                break
        cache_p = GRID_CACHE_DIR / f"{ds}_grid.npz"
        if args.cache_grids and cache_p.exists():
            try:
                d = np.load(cache_p, allow_pickle=False)
                day_grids.append({
                    "date": ds, "ts": d["ts"], "mid": d["mid"],
                    "ofi_1s": d["ofi_1s"], "book_imb": d["book_imb"],
                    "trade_flow_500ms": d["trade_flow_500ms"], "z_ret_1s": d["z_ret_1s"],
                })
                if args.verbose:
                    print(f"  {ds}: cache hit", flush=True)
                continue
            except Exception as e:
                print(f"  {ds}: cache load failed ({e}), rebuilding", flush=True)
        try:
            dg = build_day_grid(ds, verbose=args.verbose)
            if dg is None:
                print(f"  {ds}: skipped (no data / no front)", flush=True)
                continue
            day_grids.append(dg)
            if args.cache_grids:
                np.savez_compressed(cache_p,
                                    ts=dg["ts"], mid=dg["mid"],
                                    ofi_1s=dg["ofi_1s"], book_imb=dg["book_imb"],
                                    trade_flow_500ms=dg["trade_flow_500ms"],
                                    z_ret_1s=dg["z_ret_1s"])
        except Exception as e:
            print(f"  {ds}: ERROR {e}\n{traceback.format_exc()[:500]}", flush=True)

    print(f"[grids built] {len(day_grids)}/{len(dates)} days", flush=True)
    if len(day_grids) == 0:
        print("No grids built; abort.")
        return

    # Regime mix
    regimes = Counter()
    for dg in day_grids:
        regimes[classify_es_regime(dg["date"], dg)] += 1
    regime_mix = {"green": regimes["green"], "red": regimes["red"], "flat": regimes["flat"],
                  "total": sum(regimes.values()),
                  "feb_apr_2026_reference": {"green": 15, "red": 10, "flat": 3, "total": 28}}
    (OUT_DIR / "regime_mix.json").write_text(json.dumps(regime_mix, indent=2))
    print(f"[regime mix] green={regimes['green']} red={regimes['red']} flat={regimes['flat']}  "
          f"(Feb-Apr 2026 was 15/10/3)", flush=True)

    # Sweep
    signals = ["A", "B", "C"]
    horizons = [("1s", HORIZONS_NS["1s"]), ("5s", HORIZONS_NS["5s"]), ("30s", HORIZONS_NS["30s"])]
    pcts = [("p5", 0.05), ("p10", 0.10), ("p20", 0.20), ("p50", 0.50)]
    sides = ["short", "long", "both"]
    execs = ["ES_mkt", "ES_psv", "MES_mkt", "MES_psv"]

    results = []
    all_per_day = []
    total_cells = len(signals)*len(horizons)*len(pcts)*len(sides)*len(execs)
    cell_i = 0
    sweep_t0 = time.time()
    for sig in signals:
        for hl, h_ns in horizons:
            for pl, p in pcts:
                for side in sides:
                    for ex in execs:
                        cell_i += 1
                        try:
                            agg, per_day = evaluate_signal(day_grids, sig, h_ns, p,
                                                           side, ex, hl, pl)
                            results.append(agg)
                            for pd_row in per_day:
                                pd_row.update({
                                    "signal": sig, "horizon": hl, "pct": pl,
                                    "side": side, "exec": ex
                                })
                                all_per_day.append(pd_row)
                            if cell_i % 50 == 0 or cell_i == total_cells:
                                print(f"  cell {cell_i}/{total_cells} ({(time.time()-sweep_t0):.1f}s)",
                                      flush=True)
                        except Exception as e:
                            print(f"  ERROR {sig}/{hl}/{pl}/{side}/{ex}: {e}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "results_table.csv", index=False)
    pd.DataFrame(all_per_day).to_csv(OUT_DIR / "per_day_breakdown.csv", index=False)

    surv = df[df["pass_gates"]].sort_values("pdShr", ascending=False) if "pass_gates" in df.columns else pd.DataFrame()
    surv_path = OUT_DIR / "survivors.txt"
    with open(surv_path, "w") as f:
        f.write(f"# Older window: {dates[0]} → {dates[-1]}, {len(day_grids)} days processed\n")
        f.write(f"# Regime mix: green={regimes['green']} red={regimes['red']} flat={regimes['flat']}\n")
        f.write(f"# Gates: pdShr>1.5, PF>1.4, WR>55%, N_days>=50, N_trades>=200, R1 regime gate\n\n")
        if len(surv) == 0:
            f.write("NO CELLS PASS ALL GATES\n")
        else:
            for _, r in surv.iterrows():
                f.write(f"sig={r['signal']} h={r['horizon']} pct={r['pct']} side={r['side']} exec={r['exec']} | "
                        f"pdShr={r['pdShr']:.2f} PF={r['PF']:.2f} WR={r['WR']:.2%} "
                        f"N={r['n_trades']} days={r['n_days_traded']} "
                        f"$/day={r['pnl_dollars_per_day']:.2f} day_conc={r['day_conc']:.2f}\n")

    # Comparison vs Feb-Apr 2026
    comp = build_comparison(df)
    if comp is not None:
        comp.to_csv(OUT_DIR / "comparison_vs_febapr.csv", index=False)

    # MLflow
    if mlflow is not None:
        try:
            mlflow.log_artifact(str(OUT_DIR / "results_table.csv"))
            mlflow.log_artifact(str(OUT_DIR / "per_day_breakdown.csv"))
            mlflow.log_artifact(str(surv_path))
            mlflow.log_artifact(str(OUT_DIR / "regime_mix.json"))
            if comp is not None:
                mlflow.log_artifact(str(OUT_DIR / "comparison_vs_febapr.csv"))
            mlflow.log_metric("n_days", len(day_grids))
            mlflow.log_metric("n_green", regimes["green"])
            mlflow.log_metric("n_red", regimes["red"])
            mlflow.log_metric("n_flat", regimes["flat"])
            mlflow.log_metric("n_survivors", int(len(surv)))
            if len(df) > 0 and df["pdShr"].notna().any():
                mlflow.log_metric("best_pdShr", float(df["pdShr"].max()))
            mlflow.end_run()
        except Exception as e:
            print(f"[mlflow log err] {e}")

    # Print verdict
    verdict_lines = []
    verdict_lines.append("=" * 78)
    verdict_lines.append("HC #540 OLDER-WINDOW BASELINE — VERDICT")
    verdict_lines.append("=" * 78)
    verdict_lines.append(f"Days processed: {len(day_grids)} / {len(dates)} requested  "
                         f"({dates[0]} → {dates[-1]})")
    verdict_lines.append(f"Regime mix: green={regimes['green']} red={regimes['red']} "
                         f"flat={regimes['flat']}   (Feb-Apr 2026 was 15/10/3)")
    verdict_lines.append("")

    # Top 5 market-exec by pdShr (ES_mkt + MES_mkt only)
    mkt = df[df["exec"].isin(["ES_mkt", "MES_mkt"])].dropna(subset=["pdShr"]) \
            .sort_values("pdShr", ascending=False).head(5)
    verdict_lines.append("TOP 5 cells by pdShr (market exec only):")
    for _, r in mkt.iterrows():
        verdict_lines.append(f"  sig={r['signal']} h={r['horizon']} pct={r['pct']} "
                             f"side={r['side']} exec={r['exec']} | pdShr={r['pdShr']:.2f} "
                             f"PF={r['PF']:.2f} WR={r['WR']:.2%} N={int(r['n_trades'])}")
    verdict_lines.append("")

    # Cells passing gates per signal
    for sig in signals:
        sdf = df[df["signal"] == sig]
        npass = int(sdf["pass_gates"].sum())
        verdict_lines.append(f"Signal {sig}: cells passing all gates = {npass}")
    verdict_lines.append("")
    verdict_lines.append(f"Global survivors: {len(surv)}")
    verdict_lines.append("")

    # Verdict
    if len(surv) > 0:
        verdict_lines.append("=== VERDICT: OLDER WINDOW PASSES GATES ===")
        verdict_lines.append("→ Simple baselines extract tradeable edge on July 2025 – Jan 2026.")
        verdict_lines.append("→ Feb-Apr 2026 was an ANOMALOUS window. Strategies are real.")
        verdict_lines.append("→ Recommendation: re-buy targeted historical data, build real validation suite.")
    else:
        verdict_lines.append("=== VERDICT: OLDER WINDOW ALSO FAILS GATES ===")
        verdict_lines.append("→ No simple non-ML baseline produces a tradeable strategy on either window.")
        verdict_lines.append("→ Strategies are dead in general (not a window artifact).")
        verdict_lines.append("→ Recommendation: ESCALATE asset-class pivot to user.")

    # Comparison
    if comp is not None:
        verdict_lines.append("")
        verdict_lines.append("--- Comparison to Feb-Apr 2026 (same cells) ---")
        rev = comp[comp["sign_reversed"] == True]
        verdict_lines.append(f"Cells with pdShr sign reversal vs Feb-Apr 2026: {len(rev)} / {len(comp)}")
        cnt_pass_older = int(comp["pass_gates_older"].fillna(False).astype(bool).sum())
        cnt_pass_fa = int(comp["pass_gates_febapr"].fillna(False).astype(bool).sum())
        verdict_lines.append(f"Cells passing gates — older: {cnt_pass_older}, Feb-Apr 2026: {cnt_pass_fa}")

    verdict_lines.append("")
    verdict_lines.append(f"Results: {OUT_DIR/'results_table.csv'}")
    verdict_lines.append(f"Survivors: {surv_path}")
    if comp is not None:
        verdict_lines.append(f"Comparison: {OUT_DIR/'comparison_vs_febapr.csv'}")

    verdict_text = "\n".join(verdict_lines)
    print(verdict_text, flush=True)
    (OUT_DIR / "VERDICT.txt").write_text(verdict_text)


if __name__ == "__main__":
    main()
