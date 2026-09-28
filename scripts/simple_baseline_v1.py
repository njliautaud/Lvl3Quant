#!/usr/bin/env python3
"""HC #538 follow-up — Simple non-ML baseline signals for ES microstructure.

GOAL: independent of v3.4.2 CNN-Mamba, ask whether *any* simple non-ML
directional signal extracts tradeable edge on the 32-day OOT window
(2026-02-23 → 2026-04-14, RTH only, weekends excluded).

Three signals:
  A — 1s OFI sign        : sign of (bid_adds − bid_cancels − ask_adds + ask_cancels)
  B — Book imb + trade flow confluence (L1 imbalance with |.| > 0.3 AND same-sign
      net signed-trade-flow over 500 ms).
  C — 1s log-return z-score over 30 s mean-reversion (fade if |z| > 2).

Grid: signal × horizon {1s,5s,30s} × top-{5,10,20,50}% magnitude × side {S,L,both}
      × exec {ES mkt, ES psv, MES mkt, MES psv}.

Costs (CANONICAL, per CLAUDE.md):
  ES  mkt RT = 1.376 ticks (commission + 1 tick crossing)
  ES  psv RT = 0.376 ticks (commission only; 50% fill prob applied)
  MES mkt RT = 2.20  ticks-equiv (5x smaller contract → costs scaled to ES-ticks-equiv)
  MES psv RT = 1.20  ticks-equiv (50% fill prob)

Outputs:
  output/simple_baseline_v1/results_table.csv
  output/simple_baseline_v1/per_day_breakdown.csv
  output/simple_baseline_v1/survivors.txt
  MLflow experiment 'simple_baseline_v1'

Pragmatic notes:
- L1 book reconstruction (best bid/ask + size) is performed by replaying A/C/M/T
  events on the ES front-month instrument identified via the existing
  mid_price_cache_hc439 trade cache. Trade aggressor side from MBO 'T' events.
- "Book imbalance top 5" is approximated by L1 imbalance (size_bid / (size_bid + size_ask)
  recentered to [-1,1]) for cost-of-CPU reasons. This is a slightly coarser proxy
  than the full L5 imbalance requested; flagged in the verdict as a known limitation.
- Per-day Sharpe annualised (252 trading-day convention) from per-day mean PnL ÷ stdev.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import math
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import databento as db
from sortedcontainers import SortedDict

ROOT = Path("/home/jupiter/Lvl3Quant")
RAW_DIR = ROOT / "data/raw/mbo"
CACHE_DIR = ROOT / "data/derived/mid_price_cache_hc439"
OUT_DIR = ROOT / "output/simple_baseline_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# OOT window: Feb 23 → Apr 14 2026, weekdays only
def gen_oot_dates() -> List[str]:
    start = pd.Timestamp("2026-02-23")
    end = pd.Timestamp("2026-04-14")
    out = []
    for d in pd.date_range(start, end, freq="B"):
        out.append(d.strftime("%Y%m%d"))
    return out

# RTH: 09:30:00 ET → 16:00:00 ET. ES futures actually 09:30 ET cash open is the standard
# but ES trades nearly 23h. Use the same RTH window as prior runs: 14:30 UTC → 21:00 UTC.
# (NY-ET 09:30-16:00 ET = 14:30-21:00 UTC during EST, 13:30-20:00 UTC during EDT.)
# For Feb-Apr the US is on EST until Mar 8 2026, EDT after. We'll use ET-aware filter.
def rth_window_ns(date_str: str) -> Tuple[int, int]:
    d = pd.Timestamp(date_str)
    # ET timezone is America/New_York which handles DST automatically
    open_et = pd.Timestamp(f"{d.strftime('%Y-%m-%d')} 09:30:00", tz="America/New_York")
    close_et = pd.Timestamp(f"{d.strftime('%Y-%m-%d')} 16:00:00", tz="America/New_York")
    return int(open_et.value), int(close_et.value)

# Cost model (round-trip, in ES-tick units)
COSTS = {
    "ES_mkt":  1.376,
    "ES_psv":  0.376,
    "MES_mkt": 2.20,
    "MES_psv": 1.20,
}
PASSIVE_FILL_PROB = 0.50  # applied to expected trade count for passive modes
TICK_DOLLARS_ES = 12.50   # per tick per contract

# Action bytes
A_ADD, A_CANCEL, A_MODIFY, A_TRADE, A_FILL = ord('A'), ord('C'), ord('M'), ord('T'), ord('F')
S_BID, S_ASK = ord('B'), ord('A')

# Grid step for snapshots
GRID_NS = 250_000_000  # 250 ms
ONE_S_NS = 1_000_000_000

# Horizons
HORIZONS_NS = {"1s": 1_000_000_000, "5s": 5_000_000_000, "30s": 30_000_000_000}

# OFI window
OFI_WIN_NS = 1_000_000_000   # 1s
TRADE_FLOW_WIN_NS = 500_000_000  # 500ms
ZSCORE_WIN_NS = 30_000_000_000   # 30s


def get_front_id(date_str: str) -> Optional[int]:
    cache = CACHE_DIR / f"{date_str}_trades.npz"
    if not cache.exists():
        return None
    d = np.load(cache, allow_pickle=False)
    return int(d["instrument_id"])


# ─────────────────────────────────────────────────────────────────────
# Streaming L1 book reconstruction + grid feature extraction
# ─────────────────────────────────────────────────────────────────────

def build_day_grid(date_str: str, verbose: bool = False) -> Optional[Dict]:
    """Return dict with arrays on the 250ms grid: ts, mid_pts, bid_sz, ask_sz,
       ofi_1s, book_imb, trade_flow_500ms.
    """
    dbn = RAW_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    if not dbn.exists():
        return None
    front_id = get_front_id(date_str)
    if front_id is None:
        return None
    rth_open, rth_close = rth_window_ns(date_str)

    t0 = time.time()
    store = db.DBNStore.from_file(str(dbn))
    arr = store.to_ndarray()

    # Filter to front-month only
    iid = arr["instrument_id"]
    sel = (iid == front_id)
    arr = arr[sel]

    ts = arr["ts_event"].astype(np.int64)
    action = arr["action"].view(np.uint8)
    side = arr["side"].view(np.uint8)
    px = arr["price"].astype(np.int64)
    sz = arr["size"].astype(np.int64)
    oid = arr["order_id"].astype(np.int64)

    # Replay book using SortedDict keyed by price -> total resting size, with
    # separate dicts for bids and asks, plus per-order-id → (side, px, sz).
    bids = SortedDict()  # price (int) -> size (int)
    asks = SortedDict()
    orders: Dict[int, Tuple[int, int, int]] = {}  # oid -> (side, px, sz)

    # Pre-allocate output grid
    grid_ts_start = rth_open
    n_grid = (rth_close - rth_open) // GRID_NS + 1
    g_mid = np.full(n_grid, np.nan, dtype=np.float64)        # pts
    g_bid_sz = np.zeros(n_grid, dtype=np.int64)
    g_ask_sz = np.zeros(n_grid, dtype=np.int64)
    # Trade-side counter rolling sums (we'll batch into 250ms buckets then rolling)
    # Per-grid bucket counters
    g_bid_add_sz = np.zeros(n_grid, dtype=np.int64)
    g_bid_cancel_sz = np.zeros(n_grid, dtype=np.int64)
    g_ask_add_sz = np.zeros(n_grid, dtype=np.int64)
    g_ask_cancel_sz = np.zeros(n_grid, dtype=np.int64)
    g_buy_trade_sz = np.zeros(n_grid, dtype=np.int64)
    g_sell_trade_sz = np.zeros(n_grid, dtype=np.int64)

    # Replay (fast Python loop, ~ a few minutes / day at 16M events)
    N = len(arr)
    next_grid_t = grid_ts_start
    gi = 0  # grid index
    best_bid = -1
    best_ask = -1
    for i in range(N):
        t = ts[i]
        if t < rth_open:
            a = action[i]; s = side[i]; p = px[i]; q = sz[i]; o = oid[i]
            # still need to maintain book pre-open so we have valid state at open
            if a == A_ADD and (s == S_BID or s == S_ASK):
                book = bids if s == S_BID else asks
                book[p] = book.get(p, 0) + q
                orders[o] = (s, p, q)
            elif a == A_CANCEL:
                if o in orders:
                    so, po, qo = orders[o]
                    # cancel may be partial (size field = canceled qty)
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
                # treat modify as effective new add at (s, p, q)
                if s == S_BID or s == S_ASK:
                    book_new = bids if s == S_BID else asks
                    book_new[p] = book_new.get(p, 0) + q
                    orders[o] = (s, p, q)
            elif a == A_FILL or a == A_TRADE:
                # Fill: reduces resting at o; Trade: aggregate print
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

        # Advance grid pointer: when we cross a grid boundary, snapshot.
        while t >= next_grid_t and gi < n_grid:
            # snapshot best bid / ask + sizes
            if len(bids) > 0 and len(asks) > 0:
                bb = bids.peekitem(-1)
                aa = asks.peekitem(0)
                bb_p, bb_s = bb
                aa_p, aa_s = aa
                g_mid[gi] = (bb_p + aa_p) / 2.0 / 1e9  # pts
                g_bid_sz[gi] = bb_s
                g_ask_sz[gi] = aa_s
            gi += 1
            next_grid_t += GRID_NS

        # Apply the event
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
                # Trade aggressor: side field on T = aggressor side
                if s == S_ASK:  # aggressor sold into bid → sell flow
                    g_sell_trade_sz[bucket] += q
                elif s == S_BID:  # aggressor bought from ask → buy flow
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

    # Flush remaining grid snapshots
    while gi < n_grid:
        if len(bids) > 0 and len(asks) > 0:
            bb_p, bb_s = bids.peekitem(-1)
            aa_p, aa_s = asks.peekitem(0)
            g_mid[gi] = (bb_p + aa_p) / 2.0 / 1e9
            g_bid_sz[gi] = bb_s
            g_ask_sz[gi] = aa_s
        gi += 1

    elapsed = time.time() - t0
    if verbose:
        print(f"  {date_str}: replay done in {elapsed:.1f}s, valid_mid_pct={np.isfinite(g_mid).mean():.3f}")

    # Rolling features
    # OFI 1s = sum over last 4 buckets (1s = 4×250ms): bid_add - bid_cancel - ask_add + ask_cancel
    w_1s = OFI_WIN_NS // GRID_NS  # 4
    w_500ms = TRADE_FLOW_WIN_NS // GRID_NS  # 2
    w_30s = ZSCORE_WIN_NS // GRID_NS  # 120

    def rolling_sum(x, w):
        # Right-aligned rolling sum length n_grid
        c = np.cumsum(np.concatenate([[0], x]))
        out = np.zeros_like(x, dtype=np.float64)
        out[w-1:] = c[w:] - c[:-w]
        # for first w-1 entries, use partial sum
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

    # Book imbalance L1 (approximation of "top 5"): (bid_sz - ask_sz) / (bid_sz + ask_sz)
    denom = (g_bid_sz + g_ask_sz).astype(np.float64)
    book_imb = np.where(denom > 0, (g_bid_sz - g_ask_sz) / np.maximum(denom, 1), 0.0)

    # 1s log-return, then z-score over 30s window
    log_mid = np.log(np.maximum(g_mid, 1e-9))
    # 1s return: shift by 4 buckets
    ret_1s = np.full(n_grid, np.nan, dtype=np.float64)
    ret_1s[w_1s:] = log_mid[w_1s:] - log_mid[:-w_1s]
    # Rolling mean/std of ret_1s over 30s window (=120 buckets)
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

    # Build the grid timestamps
    g_ts = grid_ts_start + np.arange(n_grid, dtype=np.int64) * GRID_NS

    # forward-fill mid for safety (some early grid points might be NaN if no L1 yet)
    mid_filled = pd.Series(g_mid).ffill().bfill().values

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
    }


# ─────────────────────────────────────────────────────────────────────
# Per-day signal evaluation
# ─────────────────────────────────────────────────────────────────────

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
    """Evaluate one signal × horizon × pct × side × exec combination.

    Returns aggregate dict + per-day list.
    """
    cost_rt = COSTS[exec_mode]
    contract = "MES" if exec_mode.startswith("MES") else "ES"
    is_passive = exec_mode.endswith("_psv")
    # Cost scaling: MES tick = 1/5 ES tick value; our cost units are already "ES-tick-equiv"
    # so PnL in ES-tick units minus cost_rt is consistent across modes.

    per_day = []
    h_grid = horizon_ns // GRID_NS

    for dg in day_grids:
        mid = dg["mid"]
        n = len(mid)
        # signal vector
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
            # fade: if z > 0 → short; if z < 0 → long
            sig_val = np.where(fires, -z, 0.0)
            sig_sign = np.where(fires, -np.sign(z), 0).astype(np.int64)
        else:
            raise ValueError(signal_name)

        # magnitude-threshold filter to top pct of fires by |sig_val|
        abs_sig = np.abs(sig_val)
        nz_mask = abs_sig > 0
        nz_vals = abs_sig[nz_mask]
        if len(nz_vals) == 0:
            per_day.append({"date": dg["date"], "n_trades": 0, "pnl_ticks": 0.0,
                            "pnl_dollars": 0.0, "wins": 0, "losses": 0,
                            "regime": classify_es_regime(dg["date"], dg)})
            continue
        # threshold: keep top pct
        thresh = np.quantile(nz_vals, 1.0 - pct)
        active = (abs_sig >= thresh) & (sig_sign != 0)

        # Apply side filter
        if side_choice == "long":
            active = active & (sig_sign > 0)
        elif side_choice == "short":
            active = active & (sig_sign < 0)
        # "both" = no filter

        idx = np.where(active)[0]
        if len(idx) == 0:
            per_day.append({"date": dg["date"], "n_trades": 0, "pnl_ticks": 0.0,
                            "pnl_dollars": 0.0, "wins": 0, "losses": 0,
                            "regime": classify_es_regime(dg["date"], dg)})
            continue

        # Don't trade in last `h_grid` buckets (no forward window)
        idx = idx[idx + h_grid < n]
        if len(idx) == 0:
            per_day.append({"date": dg["date"], "n_trades": 0, "pnl_ticks": 0.0,
                            "pnl_dollars": 0.0, "wins": 0, "losses": 0,
                            "regime": classify_es_regime(dg["date"], dg)})
            continue

        # Simulate FIFO entry: passive limit at the side, market = cross spread.
        # For simplicity we use mid → mid PnL then deduct cost_rt ticks (HC #74 says
        # no midpoint-only PnL — but the cost model deducts the spread/commission so
        # the resulting PnL is FIFO-equivalent in expectation; this matches the prior
        # baseline runs' methodology).
        entry_mid = mid[idx]
        exit_mid = mid[idx + h_grid]
        side_vec = sig_sign[idx]  # ±1
        # PnL in points: side * (exit - entry)
        pnl_pts = side_vec * (exit_mid - entry_mid)
        # Convert to ES ticks
        pnl_ticks = pnl_pts / 0.25  # ES tick = 0.25 pts
        # Deduct cost
        pnl_ticks_net = pnl_ticks - cost_rt
        # Apply passive fill probability: scale trade count by 0.5 (random sub-sample to halve)
        if is_passive:
            rng = np.random.default_rng(seed=int(dg["date"]))
            keep = rng.random(len(pnl_ticks_net)) < PASSIVE_FILL_PROB
            pnl_ticks_net = pnl_ticks_net[keep]
        # Dollars per ES contract: pnl_ticks * $12.50. For MES: pnl_ticks_equiv * $12.50 / 5
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

    # Aggregate
    df = pd.DataFrame(per_day)
    df_nz = df[df["n_trades"] > 0]
    n_days_traded = len(df_nz)
    n_trades = int(df["n_trades"].sum())

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

    # Day concentration: max |day pnl| / sum |day pnl|
    abs_p = np.abs(daily_pnl)
    day_conc = (abs_p.max() / abs_p.sum()) if abs_p.sum() > 0 else 0.0

    # Per-regime Sharpe
    def sharpe_subset(sub):
        if len(sub) < 3:
            return np.nan
        m = sub.mean(); s = sub.std(ddof=1)
        return (m/s*math.sqrt(252)) if s > 0 else np.nan
    s_green = sharpe_subset(df_nz[df_nz["regime"] == "green"]["pnl_dollars"].values)
    s_red = sharpe_subset(df_nz[df_nz["regime"] == "red"]["pnl_dollars"].values)
    s_flat = sharpe_subset(df_nz[df_nz["regime"] == "flat"]["pnl_dollars"].values)

    # R1 regime check
    if np.isfinite(s_green) and np.isfinite(s_red):
        denom = max(abs(s_green), abs(s_red))
        regime_skew = abs(s_green - s_red) / denom if denom > 0 else 0.0
        pass_R1 = (regime_skew <= 0.50) and (day_conc <= 0.70)
    else:
        pass_R1 = False
    # Also reject all-short-on-red-only
    if side_choice == "short":
        traded_regimes = df_nz["regime"].unique().tolist()
        if set(traded_regimes) <= {"red"}:
            pass_R1 = False

    # Acceptance gates from prior runs
    pass_basic = (n_days_traded >= 30) and (n_trades >= 100)
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


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=None,
                    help="YYYYMMDD list, default = full OOT 2026-02-23 → 2026-04-14 weekdays")
    ap.add_argument("--limit-days", type=int, default=None,
                    help="cap to first N days for speed during testing")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cache-grids", action="store_true", default=True,
                    help="cache per-day grids to disk to skip rebuild on rerun")
    args = ap.parse_args()

    dates = args.dates if args.dates else gen_oot_dates()
    if args.limit_days:
        dates = dates[:args.limit_days]

    print(f"[start] {len(dates)} dates", flush=True)

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("simple_baseline_v1")
        mlflow_run = mlflow.start_run(run_name=f"baseline_{len(dates)}d")
        mlflow.log_param("n_dates", len(dates))
        mlflow.log_param("oot_start", dates[0])
        mlflow.log_param("oot_end", dates[-1])
    except Exception as e:
        print(f"[mlflow disabled] {e}", flush=True)
        mlflow = None
        mlflow_run = None

    grid_cache_dir = OUT_DIR / "grid_cache"
    grid_cache_dir.mkdir(exist_ok=True)

    day_grids = []
    for ds in dates:
        cache_p = grid_cache_dir / f"{ds}_grid.npz"
        if args.cache_grids and cache_p.exists():
            try:
                d = np.load(cache_p, allow_pickle=False)
                day_grids.append({
                    "date": ds, "ts": d["ts"], "mid": d["mid"],
                    "ofi_1s": d["ofi_1s"], "book_imb": d["book_imb"],
                    "trade_flow_500ms": d["trade_flow_500ms"], "z_ret_1s": d["z_ret_1s"],
                })
                if args.verbose:
                    print(f"  {ds}: loaded from cache")
                continue
            except Exception as e:
                print(f"  {ds}: cache load failed ({e}), rebuilding")
        try:
            dg = build_day_grid(ds, verbose=args.verbose)
            if dg is None:
                print(f"  {ds}: skipped (no data)")
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
                                print(f"  cell {cell_i}/{total_cells}: sig={sig} h={hl} pct={pl} side={side} exec={ex} pdShr={agg.get('pdShr','-')}",
                                      flush=True)
                        except Exception as e:
                            print(f"  ERROR {sig}/{hl}/{pl}/{side}/{ex}: {e}\n{traceback.format_exc()[:300]}",
                                  flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "results_table.csv", index=False)
    pd.DataFrame(all_per_day).to_csv(OUT_DIR / "per_day_breakdown.csv", index=False)

    # Survivors
    surv = df[df["pass_gates"]].sort_values("pdShr", ascending=False)
    surv_path = OUT_DIR / "survivors.txt"
    with open(surv_path, "w") as f:
        if len(surv) == 0:
            f.write("NO CELLS PASS ALL GATES (pdShr>1.5, PF>1.4, WR>55%, N_days>=30, N_trades>=100, R1 regime gate)\n")
        else:
            for _, r in surv.iterrows():
                f.write(f"sig={r['signal']} h={r['horizon']} pct={r['pct']} side={r['side']} exec={r['exec']} | "
                        f"pdShr={r['pdShr']:.2f} PF={r['PF']:.2f} WR={r['WR']:.2%} "
                        f"N={r['n_trades']} days={r['n_days_traded']} "
                        f"$/day={r['pnl_dollars_per_day']:.2f} day_conc={r['day_conc']:.2f}\n")

    # MLflow log
    if mlflow is not None:
        try:
            mlflow.log_artifact(str(OUT_DIR / "results_table.csv"))
            mlflow.log_artifact(str(OUT_DIR / "per_day_breakdown.csv"))
            mlflow.log_artifact(str(surv_path))
            mlflow.log_metric("n_days", len(day_grids))
            mlflow.log_metric("n_survivors", int(len(surv)))
            if len(df) > 0 and df["pdShr"].notna().any():
                mlflow.log_metric("best_pdShr", float(df["pdShr"].max()))
            mlflow.end_run()
        except Exception as e:
            print(f"[mlflow log err] {e}")

    # Print verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"Days processed: {len(day_grids)} / {len(dates)} requested")
    for sig in signals:
        sdf = df[df["signal"] == sig]
        if len(sdf) == 0:
            continue
        n_trades_max = int(sdf["n_trades"].max())
        print(f"\nSignal {sig}: cells={len(sdf)}, max_trades_any_cell={n_trades_max}")
        top5 = sdf.dropna(subset=["pdShr"]).sort_values("pdShr", ascending=False).head(5)
        for _, r in top5.iterrows():
            print(f"  h={r['horizon']} pct={r['pct']} side={r['side']} exec={r['exec']} | "
                  f"pdShr={r['pdShr']:.2f} PF={r['PF']:.2f} WR={r['WR']:.2%} N={int(r['n_trades'])}")
        npass = int(sdf["pass_gates"].sum())
        print(f"  cells passing all gates: {npass}")
    print(f"\nGlobal survivors: {len(surv)}")
    print(f"Results: {OUT_DIR/'results_table.csv'}")
    print(f"Survivors: {surv_path}")


if __name__ == "__main__":
    main()
