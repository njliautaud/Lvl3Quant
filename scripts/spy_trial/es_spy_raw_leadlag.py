#!/usr/bin/env python3
"""es_spy_raw_leadlag.py — Raw price-to-price ES vs SPY cross-correlation lead-lag.

Question: at what lag does ES log-return information arrive at SPY?
Method:   build mid grids at 50ms, compute log-returns, cross-correlate at lags
          -2000ms..+2000ms in 50ms steps. Per-day and overall.

Sign convention:
  lag > 0  => ES leads SPY by `lag` ms  (ES at time t correlates with SPY at t+lag)
  lag < 0  => SPY leads ES

Output: output/es_spy_leadlag_v1/
  lag_scan.csv     — lag_ms, corr_overall, corr_YYYYMMDD per day
  per_window.csv   — intraday window (morning/midday/eod) breakdown
  REPORT.md        — plain-English verdict
"""
from __future__ import annotations
import os, sys, json, time, logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

DATES = [
    "20260302", "20260303", "20260304", "20260305", "20260306",
    "20260309", "20260310", "20260311", "20260312",
]

ES_RAW_DIR  = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
SPY_GRID_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/spy_mid_grid")
OUT_DIR     = Path("/home/jupiter/Lvl3Quant/output/es_spy_leadlag_v1")
CACHE_DIR   = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# RTH (UTC) per date — DST started Sun Mar 8 2026 (US).
RTH_BY_DATE = {
    "20260302": (14*3600+30*60, 21*3600),
    "20260303": (14*3600+30*60, 21*3600),
    "20260304": (14*3600+30*60, 21*3600),
    "20260305": (14*3600+30*60, 21*3600),
    "20260306": (14*3600+30*60, 21*3600),
    "20260309": (13*3600+30*60, 20*3600),
    "20260310": (13*3600+30*60, 20*3600),
    "20260311": (13*3600+30*60, 20*3600),
    "20260312": (13*3600+30*60, 20*3600),
}

GRID_MS = 50
GRID_NS = GRID_MS * 1_000_000
LAG_MIN_MS, LAG_MAX_MS, LAG_STEP_MS = -2000, 2000, 50
LAGS_MS = list(range(LAG_MIN_MS, LAG_MAX_MS + LAG_STEP_MS, LAG_STEP_MS))

ES_INVALID_PRICE = 9_223_372_036_854_775_807
ES_TICK_FIXED = 250_000_000  # 0.25 USD in 1e-9 fixed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("es_spy_ll")


# ────────────────────────────────────────────────────────────────────────
# ES mid grid construction
# ────────────────────────────────────────────────────────────────────────
def build_es_mid_grid(date_str: str) -> Dict[str, np.ndarray]:
    cache = CACHE_DIR / f"es_mid_{date_str}_{GRID_MS}ms.npz"
    if cache.exists():
        d = np.load(cache)
        log.info(f"[{date_str}] ES grid cached: n={len(d['mid'])}")
        return {"ts": d["ts"], "mid": d["mid"]}

    import databento as db
    from sortedcontainers import SortedDict

    raw_path = ES_RAW_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    log.info(f"[{date_str}] decoding ES DBN: {raw_path}")

    rth_s, rth_e = RTH_BY_DATE[date_str]
    # base midnight UTC for this date
    yy = int(date_str[:4]); mm = int(date_str[4:6]); dd = int(date_str[6:8])
    import datetime as dt
    base = int(dt.datetime(yy, mm, dd, tzinfo=dt.timezone.utc).timestamp() * 1_000_000_000)
    rth_start_ns = base + rth_s * 1_000_000_000
    rth_end_ns   = base + rth_e * 1_000_000_000

    n_grid = (rth_end_ns - rth_start_ns) // GRID_NS
    ts_grid = rth_start_ns + np.arange(n_grid, dtype=np.int64) * GRID_NS
    mid_grid = np.zeros(n_grid, dtype=np.float64)
    valid_grid = np.zeros(n_grid, dtype=bool)

    bid_levels = SortedDict()
    ask_levels = SortedDict()
    last_trade = 0
    # ES MBO has multiple symbols (all contracts). We need to pick FRONT MONTH.
    # The MDP-3 DBN file contains many instruments. Filter by instrument_id of front.
    # Simplest: track per-instrument book and pick the one with most msgs (front).
    # To keep memory bounded, do TWO passes:
    #   pass 1: count msgs per instrument_id during RTH
    #   pass 2: replay only that instrument_id
    store = db.DBNStore.from_file(str(raw_path))
    log.info(f"[{date_str}] pass 1: counting front-month instrument...")
    counts: Dict[int, int] = {}
    t0 = time.time()
    n = 0
    for r in store:
        n += 1
        ts = int(r.ts_event)
        if ts < rth_start_ns or ts >= rth_end_ns:
            continue
        iid = int(r.instrument_id)
        counts[iid] = counts.get(iid, 0) + 1
        if n % 5_000_000 == 0:
            log.info(f"  pass1 scanned {n:,} elapsed={time.time()-t0:.1f}s")
    if not counts:
        raise RuntimeError(f"no ES events in RTH for {date_str}")
    front_iid = max(counts, key=counts.get)
    log.info(f"[{date_str}] front-month iid={front_iid} ({counts[front_iid]:,} msgs); "
             f"total distinct iids={len(counts)}")

    # pass 2: build book for front_iid, sample to grid
    log.info(f"[{date_str}] pass 2: replaying front-month book...")
    store2 = db.DBNStore.from_file(str(raw_path))
    grid_idx = 0  # cursor into ts_grid
    cur_mid = 0.0

    def refresh_best():
        nonlocal cur_mid
        while bid_levels and bid_levels.peekitem(-1)[1] <= 0:
            bid_levels.popitem(-1)
        while ask_levels and ask_levels.peekitem(0)[1] <= 0:
            ask_levels.popitem(0)
        bb = bid_levels.peekitem(-1)[0] if bid_levels else 0
        ba = ask_levels.peekitem(0)[0]  if ask_levels else 0
        if bb > 0 and ba > 0 and ba >= bb:
            cur_mid = (bb + ba) / 2.0
        elif last_trade > 0:
            cur_mid = float(last_trade)

    t1 = time.time()
    n = 0
    for r in store2:
        n += 1
        if n % 5_000_000 == 0:
            log.info(f"  pass2 scanned {n:,} grid_idx={grid_idx}/{n_grid} "
                     f"elapsed={time.time()-t1:.1f}s")
        if int(r.instrument_id) != front_iid:
            continue
        ts = int(r.ts_event)
        act = chr(r.action) if isinstance(r.action, int) else str(r.action)
        if act == 'R':
            bid_levels.clear(); ask_levels.clear()
            last_trade = 0
            continue
        if ts < rth_start_ns:
            # still update book pre-RTH so we have state at open
            pass
        if ts >= rth_end_ns:
            break

        sid = chr(r.side) if isinstance(r.side, int) else str(r.side)
        price = int(r.price) if r.price != ES_INVALID_PRICE else 0
        qty = int(r.size)

        if price > 0:
            if act == 'A':
                if sid == 'B': bid_levels[price] = bid_levels.get(price, 0) + qty
                elif sid == 'A': ask_levels[price] = ask_levels.get(price, 0) + qty
            elif act == 'C':
                if sid == 'B' and price in bid_levels: bid_levels[price] -= qty
                elif sid == 'A' and price in ask_levels: ask_levels[price] -= qty
            elif act == 'M':
                if sid == 'B': bid_levels[price] = bid_levels.get(price, 0) + qty
                elif sid == 'A': ask_levels[price] = ask_levels.get(price, 0) + qty
            elif act in ('T', 'F'):
                last_trade = price
                if sid == 'B' and price in ask_levels: ask_levels[price] -= qty
                elif sid == 'A' and price in bid_levels: bid_levels[price] -= qty
            refresh_best()

        # Advance grid cursor up to ts
        if ts >= rth_start_ns and cur_mid > 0:
            while grid_idx < n_grid and ts_grid[grid_idx] <= ts:
                mid_grid[grid_idx] = cur_mid
                valid_grid[grid_idx] = True
                grid_idx += 1

    # Fill any trailing grid cells with last known mid
    if grid_idx < n_grid and cur_mid > 0:
        mid_grid[grid_idx:] = cur_mid
        valid_grid[grid_idx:] = True

    n_valid = int(valid_grid.sum())
    log.info(f"[{date_str}] ES grid built: n_grid={n_grid} n_valid={n_valid} "
             f"elapsed={time.time()-t0:.1f}s")

    np.savez(cache, ts=ts_grid, mid=mid_grid, valid=valid_grid)
    return {"ts": ts_grid, "mid": mid_grid}


# ────────────────────────────────────────────────────────────────────────
# SPY mid grid loader -> resample to 50ms
# ────────────────────────────────────────────────────────────────────────
def load_spy_grid_50ms(date_str: str) -> Dict[str, np.ndarray]:
    """Load SPY 250ms grid, forward-fill to 50ms grid aligned to ES grid."""
    f = SPY_GRID_DIR / f"{date_str}_mid_250ms.npz"
    d = np.load(f)
    ts_spy = d["grid_ts_ns"]
    mid_spy = d["mid_price"]

    rth_s, rth_e = RTH_BY_DATE[date_str]
    yy = int(date_str[:4]); mm = int(date_str[4:6]); dd = int(date_str[6:8])
    import datetime as dt
    base = int(dt.datetime(yy, mm, dd, tzinfo=dt.timezone.utc).timestamp() * 1_000_000_000)
    rth_start_ns = base + rth_s * 1_000_000_000
    rth_end_ns   = base + rth_e * 1_000_000_000

    n_grid = (rth_end_ns - rth_start_ns) // GRID_NS
    ts_grid_50 = rth_start_ns + np.arange(n_grid, dtype=np.int64) * GRID_NS

    # For each 50ms point, find the latest SPY 250ms sample at or before it.
    idx = np.searchsorted(ts_spy, ts_grid_50, side='right') - 1
    idx = np.clip(idx, 0, len(ts_spy) - 1)
    mid_50 = mid_spy[idx]
    return {"ts": ts_grid_50, "mid": mid_50}


# ────────────────────────────────────────────────────────────────────────
# Lag scan
# ────────────────────────────────────────────────────────────────────────
def log_returns(mid: np.ndarray) -> np.ndarray:
    safe = np.where(mid > 0, mid, np.nan)
    lr = np.diff(np.log(safe))
    lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
    return lr.astype(np.float64)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 100: return float('nan')
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x*x).sum() * (y*y).sum())
    if denom == 0: return float('nan')
    return float((x*y).sum() / denom)


def scan_lags(ret_es: np.ndarray, ret_spy: np.ndarray, lags_steps: List[int]) -> Dict[int, float]:
    """For each lag k (in grid steps), corr( ret_es[t], ret_spy[t+k] ).
    k>0 => ES leads SPY. k<0 => SPY leads ES.
    """
    out = {}
    N = len(ret_es)
    for k in lags_steps:
        if k >= 0:
            x = ret_es[:N-k] if k > 0 else ret_es
            y = ret_spy[k:]  if k > 0 else ret_spy
        else:
            kk = -k
            x = ret_es[kk:]
            y = ret_spy[:N-kk]
        out[k] = pearson(x, y)
    return out


# ────────────────────────────────────────────────────────────────────────
# Per-window (intraday regime) scan
# ────────────────────────────────────────────────────────────────────────
def intraday_segments(ts_grid: np.ndarray, date_str: str) -> Dict[str, np.ndarray]:
    """Return masks over grid points for intraday windows.

    Times are ET (RTH = 09:30-16:00 ET regardless of DST since session is fixed).
    Map UTC -> ET by inverting RTH_BY_DATE.
    """
    rth_s, _ = RTH_BY_DATE[date_str]
    yy = int(date_str[:4]); mm = int(date_str[4:6]); dd = int(date_str[6:8])
    import datetime as dt
    base = int(dt.datetime(yy, mm, dd, tzinfo=dt.timezone.utc).timestamp() * 1_000_000_000)
    rth_start_ns = base + rth_s * 1_000_000_000
    # Offset within session (sec)
    off = (ts_grid - rth_start_ns) / 1e9
    # Session is 6.5h = 23400s.
    return {
        "open_30m":   (off >= 0)      & (off < 1800),     # 09:30-10:00 ET
        "morning":    (off >= 1800)   & (off < 7200),     # 10:00-11:30 ET
        "midday":     (off >= 7200)   & (off < 14400),    # 11:30-13:30 ET
        "afternoon":  (off >= 14400)  & (off < 21600),    # 13:30-15:30 ET
        "close_30m":  (off >= 21600)  & (off < 23400),    # 15:30-16:00 ET
    }


def main():
    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("es_spy_leadlag_v1")
        run = mlflow.start_run(run_name="es_spy_leadlag_v1")
        mlflow.log_params({
            "grid_ms": GRID_MS,
            "lag_min_ms": LAG_MIN_MS, "lag_max_ms": LAG_MAX_MS, "lag_step_ms": LAG_STEP_MS,
            "n_dates": len(DATES), "dates": ",".join(DATES),
        })
        mlflow_on = True
    except Exception as e:
        log.warning(f"MLflow off: {e}")
        mlflow_on = False; run = None

    lag_steps = [ms // GRID_MS for ms in LAGS_MS]

    per_day_corr: Dict[str, Dict[int, float]] = {}
    per_day_window_corr: Dict[str, Dict[str, Dict[int, float]]] = {}
    # For pooled overall correlation: concat returns across days, do single scan
    concat_ret_es: List[np.ndarray] = []
    concat_ret_spy: List[np.ndarray] = []

    for d in DATES:
        t0 = time.time()
        try:
            es = build_es_mid_grid(d)
            spy = load_spy_grid_50ms(d)
        except Exception as e:
            log.exception(f"[{d}] data load failed: {e}")
            continue

        # Align lengths (defensive)
        n = min(len(es["mid"]), len(spy["mid"]))
        es_mid = es["mid"][:n]
        spy_mid = spy["mid"][:n]
        ts = es["ts"][:n]

        # Drop initial zeros in ES (book hadn't formed)
        m = (es_mid > 0) & (spy_mid > 0)
        if not m.any():
            log.warning(f"[{d}] no overlap")
            continue
        first = int(np.argmax(m))
        es_mid = es_mid[first:]
        spy_mid = spy_mid[first:]
        ts = ts[first:]

        ret_es  = log_returns(es_mid)
        ret_spy = log_returns(spy_mid)
        ts_r = ts[1:]  # returns are diff -> one fewer

        # Drop bins with zero variance bursts (zeros only)
        # (kept simple — correlation handles zeros fine)

        # Full-day scan
        day_corr = scan_lags(ret_es, ret_spy, lag_steps)
        per_day_corr[d] = day_corr
        log.info(f"[{d}] full-day scan done. peak corr "
                 f"{max(v for v in day_corr.values() if np.isfinite(v)):.4f} "
                 f"at lag {max(day_corr, key=lambda k: day_corr[k] if np.isfinite(day_corr[k]) else -1)*GRID_MS}ms "
                 f"({time.time()-t0:.1f}s)")

        # Intraday window scans
        segs = intraday_segments(ts_r, d)
        window_corr: Dict[str, Dict[int, float]] = {}
        for wname, wmask in segs.items():
            if wmask.sum() < 1000:
                continue
            wre = ret_es[wmask]
            wrs = ret_spy[wmask]
            window_corr[wname] = scan_lags(wre, wrs, lag_steps)
        per_day_window_corr[d] = window_corr

        # Accumulate for pooled scan
        concat_ret_es.append(ret_es)
        concat_ret_spy.append(ret_spy)

        if mlflow_on:
            peak_lag = max(day_corr, key=lambda k: day_corr[k] if np.isfinite(day_corr[k]) else -1)
            mlflow.log_metric(f"day_{d}_peak_corr", day_corr[peak_lag])
            mlflow.log_metric(f"day_{d}_peak_lag_ms", peak_lag * GRID_MS)

    if not concat_ret_es:
        log.error("No usable days. Aborting.")
        return

    # Overall pooled scan
    big_es  = np.concatenate(concat_ret_es)
    big_spy = np.concatenate(concat_ret_spy)
    overall_corr = scan_lags(big_es, big_spy, lag_steps)
    overall_peak_lag = max(overall_corr, key=lambda k: overall_corr[k] if np.isfinite(overall_corr[k]) else -1)
    overall_peak_ms = overall_peak_lag * GRID_MS
    overall_peak_corr = overall_corr[overall_peak_lag]

    log.info(f"=== OVERALL: peak corr {overall_peak_corr:.4f} at lag {overall_peak_ms}ms ===")

    # ─── Save CSV: lag_ms, corr_overall, corr_<date> ───
    csv_path = OUT_DIR / "lag_scan.csv"
    with open(csv_path, "w") as f:
        days_avail = list(per_day_corr.keys())
        f.write("lag_ms,corr_overall," + ",".join(f"corr_{d}" for d in days_avail) + "\n")
        for k in lag_steps:
            lag_ms = k * GRID_MS
            row = [str(lag_ms), f"{overall_corr[k]:.6f}"]
            for d in days_avail:
                v = per_day_corr[d].get(k, float('nan'))
                row.append(f"{v:.6f}")
            f.write(",".join(row) + "\n")
    log.info(f"Saved {csv_path}")

    # ─── Save per-window CSV ───
    win_csv = OUT_DIR / "lag_scan_per_window.csv"
    with open(win_csv, "w") as f:
        f.write("date,window,lag_ms,corr\n")
        for d, wmap in per_day_window_corr.items():
            for wname, lc in wmap.items():
                for k, v in lc.items():
                    f.write(f"{d},{wname},{k*GRID_MS},{v:.6f}\n")
    log.info(f"Saved {win_csv}")

    # ─── Find any negative-lag (SPY leads ES) peaks ───
    neg_lags = {k: overall_corr[k] for k in lag_steps if k * GRID_MS < 0 and np.isfinite(overall_corr[k])}
    pos_lags = {k: overall_corr[k] for k in lag_steps if k * GRID_MS > 0 and np.isfinite(overall_corr[k])}
    neg_peak_lag = max(neg_lags, key=lambda k: neg_lags[k]) if neg_lags else None
    pos_peak_lag = max(pos_lags, key=lambda k: pos_lags[k]) if pos_lags else None
    neg_peak_ms = neg_peak_lag * GRID_MS if neg_peak_lag is not None else None
    pos_peak_ms = pos_peak_lag * GRID_MS if pos_peak_lag is not None else None

    # Per-window SPY-leads check
    spy_leads_windows = []  # (date, window, lag_ms, corr) where lag<0 corr is unusually high
    for d, wmap in per_day_window_corr.items():
        for wname, lc in wmap.items():
            window_neg = {k: lc[k] for k in lc if k*GRID_MS < 0 and np.isfinite(lc[k])}
            window_pos = {k: lc[k] for k in lc if k*GRID_MS > 0 and np.isfinite(lc[k])}
            if not window_neg or not window_pos:
                continue
            n_peak_k = max(window_neg, key=lambda k: window_neg[k])
            p_peak_k = max(window_pos, key=lambda k: window_pos[k])
            n_peak = window_neg[n_peak_k]
            p_peak = window_pos[p_peak_k]
            # SPY-leads window of interest = negative-lag peak >= 50% of positive-lag peak
            if p_peak > 0 and n_peak / p_peak > 0.5:
                spy_leads_windows.append({
                    "date": d, "window": wname,
                    "neg_peak_lag_ms": n_peak_k * GRID_MS, "neg_peak_corr": n_peak,
                    "pos_peak_lag_ms": p_peak_k * GRID_MS, "pos_peak_corr": p_peak,
                    "ratio": n_peak / p_peak,
                })

    # ─── REPORT ───
    lines = []
    lines.append("# ES->SPY Lead-Lag (raw cross-correlation)\n")
    lines.append(f"**Grid**: {GRID_MS}ms  **Lag range**: {LAG_MIN_MS}..{LAG_MAX_MS}ms step {LAG_STEP_MS}ms  "
                 f"**Days**: {len(per_day_corr)}/{len(DATES)}\n")
    lines.append("## Headline\n")
    lines.append(f"- Overall peak correlation **{overall_peak_corr:.4f}** at lag **{overall_peak_ms} ms** "
                 f"({'ES leads SPY' if overall_peak_ms>0 else ('SPY leads ES' if overall_peak_ms<0 else 'simultaneous')}).")
    if pos_peak_lag is not None:
        lines.append(f"- Best positive (ES-leads-SPY) lag: **{pos_peak_ms} ms**, corr **{pos_lags[pos_peak_lag]:.4f}**.")
    if neg_peak_lag is not None:
        lines.append(f"- Best negative (SPY-leads-ES) lag: **{neg_peak_ms} ms**, corr **{neg_lags[neg_peak_lag]:.4f}**.")
    lines.append("")
    lines.append("## Per-day peak ES->SPY lag\n")
    lines.append("| date | peak_lag_ms | peak_corr |\n|---|---|---|")
    for d, dc in per_day_corr.items():
        finite = {k: v for k, v in dc.items() if np.isfinite(v)}
        if not finite: continue
        pk = max(finite, key=lambda k: finite[k])
        lines.append(f"| {d} | {pk*GRID_MS} | {finite[pk]:.4f} |")
    lines.append("")
    lines.append("## Top-10 lags by overall correlation\n")
    lines.append("| rank | lag_ms | corr |\n|---|---|---|")
    sorted_lags = sorted(overall_corr.items(), key=lambda x: -x[1] if np.isfinite(x[1]) else 1)[:10]
    for i, (k, v) in enumerate(sorted_lags, 1):
        lines.append(f"| {i} | {k*GRID_MS} | {v:.4f} |")
    lines.append("")
    lines.append("## SPY-leads-ES windows of interest (neg-lag corr > 50% of pos-lag corr)\n")
    if not spy_leads_windows:
        lines.append("None found. SPY does NOT consistently lead ES in any intraday window. "
                     "Negative-lag correlations are uniformly far weaker than positive-lag.\n")
    else:
        lines.append("| date | window | neg_lag_ms | neg_corr | pos_lag_ms | pos_corr | ratio |\n"
                     "|---|---|---|---|---|---|---|")
        for w in sorted(spy_leads_windows, key=lambda x: -x["ratio"])[:30]:
            lines.append(f"| {w['date']} | {w['window']} | {w['neg_peak_lag_ms']} | {w['neg_peak_corr']:.4f} | "
                         f"{w['pos_peak_lag_ms']} | {w['pos_peak_corr']:.4f} | {w['ratio']:.2f} |")
    lines.append("")
    lines.append("## Verdict\n")
    if overall_peak_ms > 0 and overall_peak_corr > 0.05:
        verdict = (f"ES leads SPY by ~{overall_peak_ms} ms with peak correlation {overall_peak_corr:.3f}. "
                   "This matches the prior that ES futures are the S&P price-discovery venue. ")
    elif overall_peak_ms == 0:
        verdict = (f"Peak correlation {overall_peak_corr:.3f} occurs at zero lag. ES and SPY move "
                   "simultaneously at the 50ms scale measured here — either lead time is shorter than 50ms or "
                   "the SPY grid forward-fill obscures sub-250ms structure. ")
    else:
        verdict = (f"Surprising: peak correlation {overall_peak_corr:.3f} at lag {overall_peak_ms} ms suggests "
                   "SPY leads ES overall. Check data quality (front-month roll, grid alignment) before trusting. ")
    if spy_leads_windows:
        top = sorted(spy_leads_windows, key=lambda x: -x["ratio"])[0]
        verdict += (f"SPY appears to lead ES in `{top['window']}` on {top['date']} "
                    f"(neg-lag corr {top['neg_peak_corr']:.3f} vs pos-lag {top['pos_peak_corr']:.3f}). "
                    "Flag for tradeable-inefficiency follow-up.")
    else:
        verdict += "No window shows SPY leading ES nontrivially — typical positive-lag asymmetry is preserved everywhere."
    lines.append(verdict + "\n")

    report = OUT_DIR / "REPORT.md"
    report.write_text("\n".join(lines))
    log.info(f"Saved {report}")

    # JSON dump
    summary = {
        "overall_peak_lag_ms": overall_peak_ms,
        "overall_peak_corr": overall_peak_corr,
        "pos_peak_lag_ms": pos_peak_ms,
        "neg_peak_lag_ms": neg_peak_ms,
        "neg_peak_corr": neg_lags[neg_peak_lag] if neg_peak_lag is not None else None,
        "pos_peak_corr": pos_lags[pos_peak_lag] if pos_peak_lag is not None else None,
        "days_analyzed": list(per_day_corr.keys()),
        "spy_leads_windows": spy_leads_windows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    if mlflow_on:
        mlflow.log_metric("overall_peak_lag_ms", overall_peak_ms)
        mlflow.log_metric("overall_peak_corr", overall_peak_corr)
        if neg_peak_lag is not None:
            mlflow.log_metric("neg_peak_lag_ms", neg_peak_ms)
            mlflow.log_metric("neg_peak_corr", neg_lags[neg_peak_lag])
        for ap in [csv_path, win_csv, report, OUT_DIR / "summary.json"]:
            try: mlflow.log_artifact(str(ap))
            except Exception as e: log.warning(f"mlflow artifact failed: {e}")
        mlflow.end_run()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
