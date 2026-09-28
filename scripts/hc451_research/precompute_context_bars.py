#!/usr/bin/env python3
"""HC #451 R1 — Multi-resolution context-bar precompute (per-event).

For every MBO event in a day's DBN, attach causal (no-look-ahead) context
features at four timescales (1s, 10s, 60s, 300s) plus three intraday
distance-to features. Output is keyed by ts_ns matching the salience
parquet row order so downstream joins are positional.

Per HC #451 R1 the four windowed features are:
  1. net_signed_volume      — sum of signed front-month trade volume in window
                              (aggressor BUY: +size if side=='A'; SELL: -size if 'B')
  2. book_imbalance_top     — mean of (bid_sz - ask_sz)/(bid_sz + ask_sz) over window
                              sampled at every front-month book update inside window
  3. mid_price_slope        — (mid(t) - mid(t-W))/W  in ticks/sec, from front-month top-of-book
  4. realized_vol           — stddev of 1-sec log-mid returns inside window (per-sec units)

Distance-to features (single scalars per row, NaN if not yet defined intraday):
  - dist_to_vpoc_ticks       — (mid - intraday VPOC price)/tick, VPOC = price with max
                              cumulative front-month traded volume so far today
  - dist_to_on_high_ticks    — (mid - overnight-session high)/tick. Frozen at 09:30 ET.
                              "Overnight" = all front-month trades observed in this file
                              BEFORE the first trade timestamp at/after 13:30 UTC.
                              NaN before the freeze.
  - dist_to_prior_close_ticks- (mid - prior session close)/tick. Prior close is read from
                              the previous trading day's DBN if available; else NaN.

DESIGN ASSUMPTIONS (HC #451 R1 ambiguity calls — documented per task instructions):
  A1. Front-month instrument is picked via max trade count (same as precompute_salience_tags).
  A2. Tick size = 0.25 pts; databento price fixed-point divisor = 1e9; so 1 tick = 250_000_000.
  A3. ts_event ordering in raw DBN can be slightly out-of-order. We stable-sort by ts_event
      for the feature timeline, compute features per event in time-order, then scatter
      results back to original event order (matches how salience parquet aligns).
  A4. Non-front-month events still receive a row; the feature values are the *current*
      front-month feature state at that event's ts (since features are defined on the
      front-month book/tape). Their ts_ns matches the source ts_event.
  A5. book_imbalance_top window aggregation = arithmetic mean of every observed instantaneous
      top-of-book imbalance during the window. Snapshots happen on every front-month L1 change.
  A6. realized_vol uses 1-Hz mid-price samples (causal forward-fill) inside the window.
      Reported as stddev of log-returns (dimensionless per second). NaN if window has <3 samples.
  A7. mid_price_slope: if no mid available at t-W, NaN.
  A8. Overnight high/low: we cannot tell session boundaries from the file alone, so we
      use "trades before first trade with ts_event_utc hour>=13 (RTH 13:30 UTC = 09:30 ET DST,
      14:30 UTC = 09:30 ET EST)" — practical proxy. Set frozen at 13:30 UTC if that ts exists,
      else 14:30 UTC; if neither (rare), NaN.
  A9. Prior session close = last front-month trade price in the prior trading day's DBN.
      If prior file missing, NaN — but feature still emitted as NaN column.
  A10. VPOC accumulator is per-day. Reset implicitly because each day is a separate process.
       VPOC bin = exact traded price level (no bucketing — ES is 0.25-tick granular and
       intraday VPOC at the tick level is what microstructure expects).

Output schema (per date):
  parquet columns (all f32 except ts_ns):
    ts_ns: i64
    nsv_1s, nsv_10s, nsv_60s, nsv_300s             — net signed volume
    imb_1s, imb_10s, imb_60s, imb_300s             — book imbalance mean
    slope_1s, slope_10s, slope_60s, slope_300s     — mid slope (ticks/sec)
    rv_1s, rv_10s, rv_60s, rv_300s                 — realized vol (stddev log-ret)
    dist_to_vpoc_ticks, dist_to_on_high_ticks,
    dist_to_on_low_ticks, dist_to_prior_close_ticks

Run:
  python precompute_context_bars.py --date 20250714 \
      --output-dir /home/jupiter/Lvl3Quant/output/hc451_context_bars/per_day
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import databento as db

RAW_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
TICK_RAW = 250_000_000           # 1 tick in databento int price units
PRICE_DIV = 1_000_000_000        # price / PRICE_DIV = float points

A_TRADE  = ord('T')
A_ADD    = ord('A')
A_CANCEL = ord('C')
A_MODIFY = ord('M')
A_FILL   = ord('F')
A_CLEAR  = ord('R')

S_ASK = ord('A')
S_BID = ord('B')
S_NONE = ord('N')

WINDOWS_NS = np.array([
    1_000_000_000,
    10_000_000_000,
    60_000_000_000,
    300_000_000_000,
], dtype=np.int64)

# RTH cutoff for overnight freeze. ES RTH open = 09:30 ET = 13:30 UTC (DST) or 14:30 UTC.
# We use 13:30 UTC if present, else 14:30 UTC.
RTH_DST_UTC_NS = 13 * 3600 * 1_000_000_000 + 30 * 60 * 1_000_000_000
RTH_EST_UTC_NS = 14 * 3600 * 1_000_000_000 + 30 * 60 * 1_000_000_000


def _pick_front_month(arr: np.ndarray) -> int:
    actions = arr['action'].view(np.uint8)
    t_mask = actions == A_TRADE
    px = arr['price'][t_mask].astype(np.int64)
    iid = arr['instrument_id'][t_mask]
    es_mask = (px > 5_000_000_000_000) & (px < 8_000_000_000_000)
    iid = iid[es_mask]
    if len(iid) == 0:
        raise RuntimeError("no ES trades in file")
    uniq, cnt = np.unique(iid, return_counts=True)
    return int(uniq[np.argmax(cnt)])


try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*a, **kw):
        def deco(f):
            return f
        return deco


@njit(cache=True, fastmath=False)
def _build_top_of_book_and_trades(
    sorted_idx,             # int64[:]   indices into source arrays, sorted by ts
    ts_event,               # int64[:]
    action,                 # uint8[:]
    side,                   # uint8[:]
    instrument_id,          # int64[:]
    price,                  # int64[:]
    size,                   # int64[:]
    front_id,               # int64
    rth_freeze_ns,          # int64
    prior_close_price,      # int64 (raw units, -1 if NaN)
    out_mid,                # int64[:]    raw mid (-1 = NaN), per sorted event
    out_bid_sz,             # int64[:]
    out_ask_sz,             # int64[:]
    out_is_trade,           # uint8[:]    1 if front trade
    out_trade_signed_size,  # int64[:]    +size buy aggressor, -size sell
    out_trade_price,        # int64[:]    raw price
    out_vpoc_price,         # int64[:]    running VPOC price (-1 if none)
    out_on_high,            # int64[:]    overnight high (-1 if not frozen)
    out_on_low,             # int64[:]    overnight low  (-1 if not frozen)
):
    """Single pass: maintain top-of-book + VPOC + overnight extremes.

    We approximate top-of-book using a simple multi-level book restricted to a
    bounded window around the best bid/ask. Since this is a HOT path that runs
    on millions of events per day per process, we use a bounded-size price
    ladder represented as two arrays (bid side, ask side) with prices as
    int64 keys (tick-aligned). We use linear scans; for ES, active book depth
    is small (top ~5 levels carry most volume), so linear is OK on 200ms windows.

    For top-of-book tracking we only need best_bid_px / best_bid_sz /
    best_ask_px / best_ask_sz, which we maintain via aggregate size per price
    level. The ladders we use are HASHMAPS implemented as two sorted-key
    parallel arrays of fixed capacity.
    """
    # Hash-like price -> size maps using open-addressed int64 arrays.
    # Capacity tuned for ES (~ few thousand distinct active price levels per day).
    CAP = 16384
    bid_px = -np.ones(CAP, dtype=np.int64)
    bid_sz = np.zeros(CAP, dtype=np.int64)
    ask_px = -np.ones(CAP, dtype=np.int64)
    ask_sz = np.zeros(CAP, dtype=np.int64)

    best_bid = np.int64(-1)
    best_ask = np.int64(-1)
    best_bid_sz_val = np.int64(0)
    best_ask_sz_val = np.int64(0)

    # VPOC accumulator: price (int64) -> cumulative size. Open-addressed.
    VCAP = 16384
    vp_px = -np.ones(VCAP, dtype=np.int64)
    vp_sz = np.zeros(VCAP, dtype=np.int64)
    vpoc_price = np.int64(-1)
    vpoc_size  = np.int64(0)

    on_high = np.int64(-(1 << 62))
    on_low  = np.int64(1 << 62)
    on_frozen_high = np.int64(-1)
    on_frozen_low  = np.int64(-1)
    rth_seen = False

    n = sorted_idx.shape[0]
    for k in range(n):
        i = sorted_idx[k]
        a = action[i]
        s = side[i]
        iid = instrument_id[i]
        p = price[i]
        sz = size[i]
        ts = ts_event[i]

        if iid == front_id:
            # ---- Update overnight freeze if RTH boundary crossed ----
            # rth_freeze_ns is ABSOLUTE epoch ns of RTH open for this session.
            if not rth_seen and ts >= rth_freeze_ns:
                rth_seen = True
                if on_high != np.int64(-(1 << 62)):
                    on_frozen_high = on_high
                if on_low != np.int64(1 << 62):
                    on_frozen_low = on_low

            # ---- Apply book update ----
            if a == A_CLEAR:
                # full clear
                for c in range(CAP):
                    bid_px[c] = -1
                    bid_sz[c] = 0
                    ask_px[c] = -1
                    ask_sz[c] = 0
                best_bid = -1
                best_ask = -1
                best_bid_sz_val = 0
                best_ask_sz_val = 0

            elif a == A_ADD or a == A_MODIFY:
                if s == S_BID:
                    # open-address insert/accumulate
                    h = (p ^ (p >> 17)) & (CAP - 1)
                    placed = False
                    for probe in range(CAP):
                        slot = (h + probe) & (CAP - 1)
                        if bid_px[slot] == p:
                            bid_sz[slot] += sz
                            placed = True
                            break
                        elif bid_px[slot] == -1:
                            bid_px[slot] = p
                            bid_sz[slot] = sz
                            placed = True
                            break
                    if placed and p > best_bid:
                        best_bid = p
                        best_bid_sz_val = sz if best_bid != p else best_bid_sz_val
                elif s == S_ASK:
                    h = (p ^ (p >> 17)) & (CAP - 1)
                    placed = False
                    for probe in range(CAP):
                        slot = (h + probe) & (CAP - 1)
                        if ask_px[slot] == p:
                            ask_sz[slot] += sz
                            placed = True
                            break
                        elif ask_px[slot] == -1:
                            ask_px[slot] = p
                            ask_sz[slot] = sz
                            placed = True
                            break
                    if placed and (best_ask == -1 or p < best_ask):
                        best_ask = p
                        best_ask_sz_val = sz if best_ask != p else best_ask_sz_val

            elif a == A_CANCEL or a == A_FILL:
                if s == S_BID:
                    h = (p ^ (p >> 17)) & (CAP - 1)
                    for probe in range(CAP):
                        slot = (h + probe) & (CAP - 1)
                        if bid_px[slot] == p:
                            bid_sz[slot] -= sz
                            if bid_sz[slot] <= 0:
                                bid_sz[slot] = 0
                                bid_px[slot] = -2   # tombstone
                            break
                        elif bid_px[slot] == -1:
                            break
                elif s == S_ASK:
                    h = (p ^ (p >> 17)) & (CAP - 1)
                    for probe in range(CAP):
                        slot = (h + probe) & (CAP - 1)
                        if ask_px[slot] == p:
                            ask_sz[slot] -= sz
                            if ask_sz[slot] <= 0:
                                ask_sz[slot] = 0
                                ask_px[slot] = -2
                            break
                        elif ask_px[slot] == -1:
                            break

            # Recompute best-of-book by scanning ladders (bounded CAP)
            # We do this opportunistically: only when current best was touched.
            # For correctness on cancels/fills at best, re-scan.
            need_rescan_bid = (a == A_CANCEL or a == A_FILL or a == A_CLEAR) and s == S_BID
            need_rescan_ask = (a == A_CANCEL or a == A_FILL or a == A_CLEAR) and s == S_ASK
            # Always rescan to keep correct (cost OK; CAP=16384, scan ~us)
            new_best_bid = np.int64(-1)
            new_best_bid_sz = np.int64(0)
            for c in range(CAP):
                px_c = bid_px[c]
                if px_c > 0 and bid_sz[c] > 0:
                    if px_c > new_best_bid:
                        new_best_bid = px_c
                        new_best_bid_sz = bid_sz[c]
            best_bid = new_best_bid
            best_bid_sz_val = new_best_bid_sz

            new_best_ask = np.int64(-1)
            new_best_ask_sz = np.int64(0)
            for c in range(CAP):
                px_c = ask_px[c]
                if px_c > 0 and ask_sz[c] > 0:
                    if new_best_ask == -1 or px_c < new_best_ask:
                        new_best_ask = px_c
                        new_best_ask_sz = ask_sz[c]
            best_ask = new_best_ask
            best_ask_sz_val = new_best_ask_sz

            # ---- Trade handling ----
            if a == A_TRADE:
                # Aggressor sign: side=='A' means trade hit ASK -> buyer aggressor (+)
                # side=='B' means trade hit BID -> seller aggressor (-).
                if s == S_ASK:
                    out_trade_signed_size[k] = sz
                elif s == S_BID:
                    out_trade_signed_size[k] = -sz
                else:
                    out_trade_signed_size[k] = 0
                out_trade_price[k] = p
                out_is_trade[k] = 1

                # Overnight extremes pre-freeze
                if not rth_seen:
                    if p > on_high:
                        on_high = p
                    if p < on_low:
                        on_low = p

                # VPOC update
                h = (p ^ (p >> 17)) & (VCAP - 1)
                for probe in range(VCAP):
                    slot = (h + probe) & (VCAP - 1)
                    if vp_px[slot] == p:
                        vp_sz[slot] += sz
                        if vp_sz[slot] > vpoc_size:
                            vpoc_size = vp_sz[slot]
                            vpoc_price = p
                        break
                    elif vp_px[slot] == -1:
                        vp_px[slot] = p
                        vp_sz[slot] = sz
                        if sz > vpoc_size:
                            vpoc_size = sz
                            vpoc_price = p
                        break

        # ---- Emit snapshot for THIS event (whether front or not) ----
        if best_bid > 0 and best_ask > 0:
            out_mid[k] = (best_bid + best_ask) // 2
            out_bid_sz[k] = best_bid_sz_val
            out_ask_sz[k] = best_ask_sz_val
        else:
            out_mid[k] = -1
            out_bid_sz[k] = 0
            out_ask_sz[k] = 0

        out_vpoc_price[k] = vpoc_price
        out_on_high[k] = on_frozen_high
        out_on_low[k]  = on_frozen_low


@njit(cache=True, fastmath=False)
def _compute_windowed_features(
    ts_sorted,              # int64[:]
    mid_sorted,             # int64[:]   -1 = NaN
    bid_sz_sorted,          # int64[:]
    ask_sz_sorted,          # int64[:]
    is_trade,               # uint8[:]
    trade_signed_size,      # int64[:]
    windows_ns,             # int64[:]   length W
    # outputs (n_events x W)
    nsv_out,                # float32[:, :]
    imb_out,                # float32[:, :]
    slope_out,              # float32[:, :]
    rv_out,                 # float32[:, :]
):
    n = ts_sorted.shape[0]
    W = windows_ns.shape[0]

    # We compute each window with a two-pointer over events (NSV) and over a
    # uniformly-spaced 1Hz mid sample buffer (RV / slope).
    # For imb_out we accumulate sum and count of valid imbalance over window.

    # --- Step 1: build 1Hz mid samples (causal forward-fill of mid_sorted) ---
    # Time grid: from ts_sorted[0] to ts_sorted[-1] at 1s spacing.
    if n == 0:
        return
    t_start = ts_sorted[0]
    t_end   = ts_sorted[-1]
    n_grid = int((t_end - t_start) // 1_000_000_000) + 2
    mid_grid = np.full(n_grid, -1, dtype=np.int64)

    # Forward-fill mid at each grid second using a single pass over events
    last_mid = np.int64(-1)
    g = 0
    for k in range(n):
        # Bring grid up to (and including) this event time
        while g < n_grid and t_start + np.int64(g) * 1_000_000_000 <= ts_sorted[k]:
            mid_grid[g] = last_mid
            g += 1
        if mid_sorted[k] > 0:
            last_mid = mid_sorted[k]
    while g < n_grid:
        mid_grid[g] = last_mid
        g += 1

    # --- Step 2: per-window two-pointer accumulators ---
    for w_idx in range(W):
        win = windows_ns[w_idx]
        # NSV: sum of trade_signed_size in (t-win, t]
        left = 0
        nsv = np.float64(0.0)
        for k in range(n):
            # advance left
            lim = ts_sorted[k] - win
            while left <= k and ts_sorted[left] < lim:
                if is_trade[left] == 1:
                    nsv -= trade_signed_size[left]
                left += 1
            if is_trade[k] == 1:
                nsv += trade_signed_size[k]
            nsv_out[k, w_idx] = np.float32(nsv)

        # IMB: mean of (bid_sz - ask_sz)/(bid_sz + ask_sz) over window
        # We accumulate sum + count using a sliding two-pointer.
        left = 0
        s_imb = np.float64(0.0)
        c_imb = 0
        for k in range(n):
            lim = ts_sorted[k] - win
            while left <= k and ts_sorted[left] < lim:
                b = bid_sz_sorted[left]
                a = ask_sz_sorted[left]
                tot = b + a
                if tot > 0:
                    s_imb -= (b - a) / tot
                    c_imb -= 1
                left += 1
            b = bid_sz_sorted[k]
            a = ask_sz_sorted[k]
            tot = b + a
            if tot > 0:
                s_imb += (b - a) / tot
                c_imb += 1
            if c_imb > 0:
                imb_out[k, w_idx] = np.float32(s_imb / c_imb)
            else:
                imb_out[k, w_idx] = np.float32(np.nan)

    # --- Step 3: slope + RV from 1Hz mid_grid ---
    # For each event, compute mid(t) and mid(t-W). slope = (m_t - m_tw)/W_sec.
    # RV: stddev of log returns over the window of 1Hz samples ending at floor(t).
    for w_idx in range(W):
        win_ns = windows_ns[w_idx]
        win_s = int(win_ns // 1_000_000_000)
        # Pre-compute log mids on grid for RV
        # walk events
        for k in range(n):
            t = ts_sorted[k]
            g_now = int((t - t_start) // 1_000_000_000)
            if g_now < 0:
                g_now = 0
            if g_now >= n_grid:
                g_now = n_grid - 1
            g_prev = g_now - win_s
            m_now = mid_grid[g_now]
            if g_prev < 0:
                slope_out[k, w_idx] = np.float32(np.nan)
                rv_out[k, w_idx]    = np.float32(np.nan)
                continue
            m_prev = mid_grid[g_prev]
            if m_now <= 0 or m_prev <= 0:
                slope_out[k, w_idx] = np.float32(np.nan)
            else:
                d_ticks = (m_now - m_prev) / TICK_RAW
                slope_out[k, w_idx] = np.float32(d_ticks / win_s)

            # RV across [g_prev .. g_now]
            n_samp = g_now - g_prev + 1
            if n_samp < 3:
                rv_out[k, w_idx] = np.float32(np.nan)
                continue
            # compute mean of log returns
            mean_r = 0.0
            cnt_r  = 0
            prev_lm = 0.0
            have_prev = False
            for gi in range(g_prev, g_now + 1):
                m = mid_grid[gi]
                if m <= 0:
                    have_prev = False
                    continue
                lm = np.log(m / 1.0)  # constant offset cancels in diffs
                if have_prev:
                    mean_r += (lm - prev_lm)
                    cnt_r += 1
                prev_lm = lm
                have_prev = True
            if cnt_r < 2:
                rv_out[k, w_idx] = np.float32(np.nan)
                continue
            mean_r /= cnt_r
            var_r = 0.0
            prev_lm = 0.0
            have_prev = False
            cnt_v = 0
            for gi in range(g_prev, g_now + 1):
                m = mid_grid[gi]
                if m <= 0:
                    have_prev = False
                    continue
                lm = np.log(m / 1.0)
                if have_prev:
                    d = (lm - prev_lm) - mean_r
                    var_r += d * d
                    cnt_v += 1
                prev_lm = lm
                have_prev = True
            if cnt_v < 2:
                rv_out[k, w_idx] = np.float32(np.nan)
                continue
            rv_out[k, w_idx] = np.float32(np.sqrt(var_r / (cnt_v - 1)))


def _rth_freeze_absolute_ns(date_str: str) -> int:
    """Return absolute ts_ns at which RTH opens on date_str (09:30 ET).

    During DST (mid-Mar to early-Nov) ET-UTC offset is -4 → 09:30 ET = 13:30 UTC.
    Outside DST it's -5 → 09:30 ET = 14:30 UTC.
    Cheap approximation: use month-based DST window (Mar 10–Nov 5 roughly).
    """
    from datetime import datetime, timezone, timedelta
    d = datetime.strptime(date_str, "%Y%m%d")
    # Rough DST: 2nd Sun in March → 1st Sun in November.
    is_dst = 3 <= d.month <= 11
    if d.month == 3:
        # 2nd Sunday in March
        sundays = [day for day in range(1, 32)
                   if datetime(d.year, 3, day).weekday() == 6]
        is_dst = d.day >= sundays[1]
    elif d.month == 11:
        sundays = [day for day in range(1, 31)
                   if datetime(d.year, 11, day).weekday() == 6]
        is_dst = d.day < sundays[0]
    open_utc_hour = 13 if is_dst else 14
    open_dt = datetime(d.year, d.month, d.day, open_utc_hour, 30, 0,
                       tzinfo=timezone.utc)
    return int(open_dt.timestamp() * 1_000_000_000)


def _read_prior_close(date_str: str) -> int:
    """Find the previous trading day's DBN and return last front-month trade price (raw int).
    Returns -1 if not found.
    """
    from datetime import datetime, timedelta
    d = datetime.strptime(date_str, "%Y%m%d").date()
    for back in range(1, 6):  # search up to 5 calendar days back (skip weekends)
        d_prev = d - timedelta(days=back)
        prev_str = d_prev.strftime("%Y%m%d")
        prev_path = RAW_DIR / f"glbx-mdp3-{prev_str}.mbo.dbn.zst"
        if prev_path.exists():
            try:
                store = db.DBNStore.from_file(str(prev_path))
                arr = store.to_ndarray()
                acts = arr['action'].view(np.uint8)
                tmask = acts == A_TRADE
                if not tmask.any():
                    continue
                front_id = _pick_front_month(arr)
                iids = arr['instrument_id']
                fmask = tmask & (iids == front_id)
                if not fmask.any():
                    continue
                prices = arr['price'][fmask].astype(np.int64)
                ts_tr = arr['ts_event'][fmask].astype(np.int64)
                # last by time
                last_idx = int(np.argmax(ts_tr))
                return int(prices[last_idx])
            except Exception:
                continue
    return -1


def process_date(date_str: str, output_dir: Path, force: bool = False) -> dict:
    out_path = output_dir / f"{date_str}_context_bars.parquet"
    stats_path = output_dir / f"{date_str}_stats.json"
    if out_path.exists() and not force:
        return {"date": date_str, "status": "skipped_exists"}

    dbn_path = RAW_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    if not dbn_path.exists():
        return {"date": date_str, "status": "missing_dbn"}

    t0 = time.time()
    store = db.DBNStore.from_file(str(dbn_path))
    arr = store.to_ndarray()
    t_load = time.time() - t0

    n_events = len(arr)
    action_a = arr['action'].view(np.uint8).copy()
    side_a   = arr['side'].view(np.uint8).copy()
    iid_a    = arr['instrument_id'].astype(np.int64)
    ts_a     = arr['ts_event'].astype(np.int64)
    price_a  = arr['price'].astype(np.int64)
    size_a   = arr['size'].astype(np.int64)

    front_id = _pick_front_month(arr)
    prior_close = _read_prior_close(date_str)
    rth_freeze_ns = _rth_freeze_absolute_ns(date_str)

    # Stable sort by ts_event (raw is mostly sorted but not strictly).
    if (np.diff(ts_a) >= 0).all():
        sorted_idx = np.arange(n_events, dtype=np.int64)
    else:
        sorted_idx = np.argsort(ts_a, kind='stable').astype(np.int64)
    ts_sorted = ts_a[sorted_idx]

    # Allocate sorted-order intermediate buffers
    out_mid     = np.full(n_events, -1, dtype=np.int64)
    out_bid_sz  = np.zeros(n_events, dtype=np.int64)
    out_ask_sz  = np.zeros(n_events, dtype=np.int64)
    out_is_trade = np.zeros(n_events, dtype=np.uint8)
    out_trade_signed_size = np.zeros(n_events, dtype=np.int64)
    out_trade_price = np.zeros(n_events, dtype=np.int64)
    out_vpoc_price = np.full(n_events, -1, dtype=np.int64)
    out_on_high = np.full(n_events, -1, dtype=np.int64)
    out_on_low  = np.full(n_events, -1, dtype=np.int64)

    t1 = time.time()
    _build_top_of_book_and_trades(
        sorted_idx, ts_a, action_a, side_a, iid_a, price_a, size_a,
        np.int64(front_id), np.int64(rth_freeze_ns), np.int64(prior_close),
        out_mid, out_bid_sz, out_ask_sz, out_is_trade,
        out_trade_signed_size, out_trade_price,
        out_vpoc_price, out_on_high, out_on_low,
    )
    t_book = time.time() - t1

    # Window features (in sorted order)
    W = WINDOWS_NS.shape[0]
    nsv_out   = np.zeros((n_events, W), dtype=np.float32)
    imb_out   = np.zeros((n_events, W), dtype=np.float32)
    slope_out = np.zeros((n_events, W), dtype=np.float32)
    rv_out    = np.zeros((n_events, W), dtype=np.float32)

    t2 = time.time()
    _compute_windowed_features(
        ts_sorted, out_mid, out_bid_sz, out_ask_sz,
        out_is_trade, out_trade_signed_size,
        WINDOWS_NS,
        nsv_out, imb_out, slope_out, rv_out,
    )
    t_feat = time.time() - t2

    # Distance-to-features per sorted event
    dist_vpoc = np.full(n_events, np.nan, dtype=np.float32)
    dist_high = np.full(n_events, np.nan, dtype=np.float32)
    dist_low  = np.full(n_events, np.nan, dtype=np.float32)
    dist_close = np.full(n_events, np.nan, dtype=np.float32)
    valid_mid = out_mid > 0
    midf = out_mid.astype(np.float64)
    if prior_close > 0:
        dist_close = np.where(valid_mid,
                              ((midf - prior_close) / TICK_RAW).astype(np.float32),
                              np.float32(np.nan))
    valid_vp = (out_vpoc_price > 0) & valid_mid
    dist_vpoc = np.where(valid_vp,
                         ((midf - out_vpoc_price.astype(np.float64)) / TICK_RAW).astype(np.float32),
                         np.float32(np.nan))
    valid_h = (out_on_high > 0) & valid_mid
    dist_high = np.where(valid_h,
                         ((midf - out_on_high.astype(np.float64)) / TICK_RAW).astype(np.float32),
                         np.float32(np.nan))
    valid_l = (out_on_low > 0) & valid_mid
    dist_low = np.where(valid_l,
                        ((midf - out_on_low.astype(np.float64)) / TICK_RAW).astype(np.float32),
                        np.float32(np.nan))

    # Unscatter sorted-order outputs back to source-order to match salience parquet row order.
    # sorted_idx[k] = source_index of sorted position k. We want arr_in_source_order[i] = value at k where sorted_idx[k]==i.
    inv = np.empty(n_events, dtype=np.int64)
    inv[sorted_idx] = np.arange(n_events, dtype=np.int64)

    def src(arr_sorted):
        return arr_sorted[inv]

    ts_src = ts_a  # already in source order
    df_dict = {"ts_ns": ts_src}
    names_w = ["1s", "10s", "60s", "300s"]
    for w_idx, w in enumerate(names_w):
        df_dict[f"nsv_{w}"]   = src(nsv_out[:, w_idx])
        df_dict[f"imb_{w}"]   = src(imb_out[:, w_idx])
        df_dict[f"slope_{w}"] = src(slope_out[:, w_idx])
        df_dict[f"rv_{w}"]    = src(rv_out[:, w_idx])
    df_dict["dist_to_vpoc_ticks"]       = src(dist_vpoc)
    df_dict["dist_to_on_high_ticks"]    = src(dist_high)
    df_dict["dist_to_on_low_ticks"]     = src(dist_low)
    df_dict["dist_to_prior_close_ticks"] = src(dist_close)

    t3 = time.time()
    df = pl.DataFrame(df_dict)
    df.write_parquet(out_path, compression="zstd")
    t_write = time.time() - t3

    elapsed = time.time() - t0
    stats = {
        "date": date_str,
        "status": "ok",
        "front_instrument_id": int(front_id),
        "prior_close_raw": int(prior_close),
        "n_events": int(n_events),
        "n_trades_front": int(out_is_trade.sum()),
        "elapsed_total_s": round(elapsed, 2),
        "elapsed_load_s": round(t_load, 2),
        "elapsed_book_s": round(t_book, 2),
        "elapsed_feat_s": round(t_feat, 2),
        "elapsed_write_s": round(t_write, 2),
        "used_numba": HAS_NUMBA,
        "windows_ns": WINDOWS_NS.tolist(),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = process_date(args.date, out_dir, force=args.force)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
