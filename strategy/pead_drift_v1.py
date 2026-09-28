#!/usr/bin/env python3
"""
pead_drift_v1 — Post-Earnings Announcement Drift in STOCK (long-short).
================================================================================
PRE-REGISTERED DESIGN (written 2026-06-11/12 BEFORE any results were computed).
Lane opened under HC #603 R1 (quant experimentation, non-ES, free/on-disk data).
NEW family — RUN_HISTORY grep for PEAD / post-earnings / drift = zero hits.

WHY THIS LANE
-------------
The last two closures (earnings_vrp_v1, earnings_longvol_v1) both died on the
single-name OPTION spread (~0.9% notional round-trip) eating a real-but-small
gross edge. PEAD is the canonical event-driven anomaly that trades STOCK,
where the cost floor on liquid names is ~10-25bps round trip — 30-90x lower.
It is also cross-sectional LONG-SHORT (dollar-neutral), the only structural
shape that has come near the HC #428 R1 regime gate in this program (every
long-beta lane failed it). HC #586 post-mortem explicitly recommended
"post-earnings drift" as the regime-orthogonal next frontier.

QUESTION
--------
Does continuation of the earnings-announcement-day reaction (classic PEAD /
Chan-Jegadeesh-Lakonishok abnormal-announcement-return sort) survive realistic
stock costs, regime-agnostically, on the ~2,900-name FMP universe 2015-2026
(calendar bounded by SPY series in prices_v2.parquet)?

DATA (all on disk, zero spend)
------------------------------
- Events: teleclaude-main/data/fmp_archive/financials/<T>/income_quarter.json
  (3,240 tickers; acceptedDate of 10-Q/10-K + actual eps).
- Prices: teleclaude-main/data/fmp_archive/prices/<T>_daily.json (4,708
  tickers, split-adjusted OHLCV 2010-01 .. 2026-03-09).
- SPY + calendar: wheel_strategy_v1/data/cache/prices_v2.parquet.
- Secondary SUE: fmp_archive/earnings/<T>/analyst_estimates_quarter.json
  (epsAvg consensus). CAVEAT (stated upfront): estimate vintage is not
  verifiably point-in-time -> SUE cells are DIAGNOSTIC ONLY and can never
  drive a PASS verdict.

EVENT / SIGNAL CONSTRUCTION (fixed before running)
--------------------------------------------------
- Event day E = trading day with max |open/prev_close - 1| (overnight gap) in
  [acceptedDate - 10d, acceptedDate + 1d] (same refinement convention as
  earnings_vrp_v1; NO minimum-gap threshold here — low-reaction events simply
  land in middle quintiles and are not traded).
- Dedup: one event per (ticker, E); drop events within 5 trading days of the
  ticker's previous event.
- PRIMARY signal: reaction = (close_E / close_{E-1} - 1) - SPY same-day ret.
  Observable at the close of E. Entry at close(E) primary; entry at
  open(E+1) as mandatory robustness (G5).
- Ranking is PAST-ONLY: each event's reaction is ranked against all events
  with E' in the trailing 60 trading days (min 300 reference events, else
  event skipped — warmup). Long if rank >= 0.80, short if rank <= 0.20.
- SECONDARY signal (diagnostic): SUE = (eps_actual - epsAvg) / close_E,
  same past-only ranking.

UNIVERSE / LIQUIDITY TIERS (at E, past-only)
--------------------------------------------
- close_E >= $5 AND median dollar volume over previous 60 trading days:
  * tier adv10: >= $10M (primary)
  * tier adv50: >= $50M (robustness — megacap-ish)
- Data sanity: ticker dropped entirely if any |daily close ret| > 200%
  (unadjusted-split artifact); per-position daily returns winsorized +/-50%.

PORTFOLIO / COSTS (stated upfront — retail, zero commission broker)
-------------------------------------------------------------------
- Holds H in {5, 10, 21} trading days (exit at close(E+H), ticker calendar;
  truncated exit at last available close if delisted — no survivorship drop).
- Book: equal-weight 50% long Q5 / 50% short Q1 of active events; $100k.
  Daily book ret = 0.5*mean(long daily rets) - 0.5*mean(short daily rets);
  zero on days with no positions; Sharpe over all calendar-span trading days.
- Costs PRIMARY: 10 bps per side (spread+impact, small size on >=$10M ADV).
  STRESS: 25 bps per side. Charged on entry day and exit day of each leg.
- Dividends not in price series (split-adjusted only) — long-short largely
  nets this; noted as caveat.

GATES (PRE-REGISTERED — pass ALL or verdict = CLOSED NEGATIVE)
--------------------------------------------------------------
- G1: PRIMARY cells only (signal=reaction, entry=closeE, cost=10bps):
      >= 1 (tier, H) cell with n_events >= 2,000 AND book Sharpe >= 1.0,
      AND plateau: every adjacent-H cell (same tier) has Sharpe >= 0.5.
- G2: that cell has positive total P&L in >= 70% of calendar years with
      activity.
- G3: regime gap on daily book returns split by SPY green/red day:
      |Sh_g - Sh_r| / max(|Sh_g|, |Sh_r|) <= 0.50 (HC #428 R1).
- G4: max single-day |P&L| share <= 0.70 (HC #344); max single-event share
      of gross profits <= 10%.
- G5: robustness, same (tier,H): Sharpe >= 0.5 at 25 bps/side AND
      Sharpe >= 0.5 with open(E+1) entry @10bps (no knife-edge on cost or
      entry timing).
- No parameter fitting anywhere; ranking + liquidity are expanding/rolling
  past-only -> entire 2015-2026 stream is OOT. No training => sliding-window
  rules (HC #0) not triggered.

MLflow: experiment pead_drift_v1 @ http://localhost:5000 (mandatory).
Artifacts: /home/jupiter/Lvl3Quant/results/pead_drift_v1/
CPU only (Jupiter). Wheel/live paper engines untouched.
"""
import json
import os
import sys
import time
import bisect
from collections import deque

import numpy as np
import pandas as pd

FMP = "/home/jupiter/teleclaude-main/data/fmp_archive"
PRICES_V2 = "/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/prices_v2.parquet"
OUT = "/home/jupiter/Lvl3Quant/results/pead_drift_v1"
MLFLOW_URI = "http://localhost:5000"
EXP = "pead_drift_v1"

HOLDS = [5, 10, 21]
TIERS = {"adv10": 10e6, "adv50": 50e6}
COSTS = {"c10": 0.0010, "c25": 0.0025}  # per side
RANK_LO, RANK_HI = 0.20, 0.80
RANK_WINDOW_TD = 60
MIN_REF = 300
MIN_PRICE = 5.0
ADV_LOOKBACK = 60
WINSOR = 0.50
BOOK = 100_000.0


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- calendar
def load_calendar_and_spy():
    df = pd.read_parquet(PRICES_V2)
    spy = df[df.ticker == "SPY"][["date", "close"]].sort_values("date")
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.drop_duplicates("date")
    cal = spy["date"].dt.strftime("%Y-%m-%d").tolist()
    spy_ret = spy["close"].pct_change().fillna(0.0).values
    return cal, dict(zip(cal, range(len(cal)))), spy_ret


# ---------------------------------------------------------------- prices
def load_ticker_prices(t, cal_idx):
    fp = f"{FMP}/prices/{t}_daily.json"
    if not os.path.exists(fp):
        return None
    try:
        with open(fp) as f:
            rows = json.load(f)
    except Exception:
        return None
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["date"])
    d, o, c, v = [], [], [], []
    for r in rows:
        gi = cal_idx.get(r["date"])
        if gi is None:
            continue
        cl = r.get("close")
        op = r.get("open")
        if not cl or cl <= 0 or not op or op <= 0:
            continue
        d.append(gi)
        o.append(op)
        c.append(cl)
        v.append((r.get("volume") or 0) * cl)
    if len(d) < ADV_LOOKBACK + 30:
        return None
    d = np.array(d, dtype=np.int64)
    keep = np.concatenate([[True], np.diff(d) > 0])
    d, o, c, v = d[keep], np.array(o)[keep], np.array(c)[keep], np.array(v)[keep]
    rets = np.diff(c) / c[:-1]
    if len(rets) and np.max(np.abs(rets)) > 2.0:  # unadjusted-split artifact
        return None
    return {"gidx": d, "open": o, "close": c, "dvol": v}


# ---------------------------------------------------------------- events
def load_events(cal, cal_idx, spy_ret):
    fin_dir = f"{FMP}/financials"
    tickers = sorted(os.listdir(fin_dir))
    events = []
    px_cache = {}
    n_px = 0
    drops = {"no_prices": 0, "no_inc": 0, "no_window": 0, "dup": 0}
    for ti, t in enumerate(tickers):
        if ti % 500 == 0:
            log(f"events: {ti}/{len(tickers)} tickers, {len(events)} events")
        inc_fp = f"{fin_dir}/{t}/income_quarter.json"
        if not os.path.exists(inc_fp):
            drops["no_inc"] += 1
            continue
        px = load_ticker_prices(t, cal_idx)
        if px is None:
            drops["no_prices"] += 1
            continue
        try:
            with open(inc_fp) as f:
                inc = json.load(f)
        except Exception:
            continue
        # estimates (secondary)
        est = {}
        est_fp = f"{FMP}/earnings/{t}/analyst_estimates_quarter.json"
        if os.path.exists(est_fp):
            try:
                with open(est_fp) as f:
                    for r in json.load(f):
                        if r.get("epsAvg") is not None:
                            est[r["date"]] = r["epsAvg"]
            except Exception:
                pass
        gidx = px["gidx"]
        seen_e = set()
        last_e = -10**9
        evs_t = []
        for r in inc:
            acc = r.get("acceptedDate") or r.get("filingDate")
            if not acc:
                continue
            acc_d = acc[:10]
            if acc_d < "2010-07-01" or acc_d > "2026-03-01":
                continue
            try:
                a = pd.Timestamp(acc_d)
            except Exception:
                continue
            lo = (a - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            hi = (a + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            lo_g = cal_idx.get(lo) or bisect.bisect_left(cal, lo)
            hi_g = cal_idx.get(hi) or bisect.bisect_right(cal, hi) - 1
            # ticker positions within [lo_g, hi_g], need prev close -> pos>=1
            p_lo = np.searchsorted(gidx, lo_g)
            p_hi = np.searchsorted(gidx, hi_g, side="right") - 1
            p_lo = max(p_lo, 1)
            if p_hi < p_lo:
                drops["no_window"] += 1
                continue
            sl = slice(p_lo, p_hi + 1)
            gaps = np.abs(px["open"][sl] / px["close"][p_lo - 1:p_hi] - 1.0)
            p_e = p_lo + int(np.argmax(gaps))
            if p_e in seen_e or p_e - last_e < 5:
                drops["dup"] += 1
                continue
            evs_t.append((p_e, r.get("eps"), est.get(r.get("date"))))
        evs_t.sort(key=lambda x: x[0])
        for p_e, eps, eps_est in evs_t:
            if p_e in seen_e or (seen_e and p_e - max(seen_e) < 5):
                continue
            seen_e.add(p_e)
            g_e = gidx[p_e]
            close_e = px["close"][p_e]
            if close_e < MIN_PRICE:
                continue
            if p_e < ADV_LOOKBACK:
                continue
            adv = float(np.median(px["dvol"][p_e - ADV_LOOKBACK:p_e]))
            if adv < TIERS["adv10"]:
                continue
            reaction = (close_e / px["close"][p_e - 1] - 1.0) - spy_ret[g_e]
            sue = None
            if eps is not None and eps_est is not None:
                sue = (eps - eps_est) / close_e
            events.append({
                "ticker": t, "p_e": p_e, "g_e": int(g_e),
                "reaction": float(reaction), "sue": sue, "adv": adv,
                "date_e": cal[g_e],
            })
        if t not in px_cache:
            px_cache[t] = px
            n_px += 1
    log(f"events built: {len(events)} | px tickers {n_px} | drops {drops}")
    return events, px_cache, drops


# ---------------------------------------------------------------- ranking
def assign_ranks(events, key):
    """Past-only rank vs trailing RANK_WINDOW_TD trading-day event cohort."""
    evs = sorted([e for e in events if e.get(key) is not None],
                 key=lambda e: e["g_e"])
    ref = deque()          # (g_e, value) in arrival order
    sorted_vals = []       # sorted list of ref values
    out = []
    i = 0
    by_day = {}
    for e in evs:
        by_day.setdefault(e["g_e"], []).append(e)
    for g in sorted(by_day):
        while ref and ref[0][0] < g - RANK_WINDOW_TD:
            _, v = ref.popleft()
            sorted_vals.pop(bisect.bisect_left(sorted_vals, v))
        n = len(sorted_vals)
        for e in by_day[g]:
            if n >= MIN_REF:
                rk = bisect.bisect_left(sorted_vals, e[key]) / n
                e[f"rank_{key}"] = rk
                out.append(e)
        for e in by_day[g]:
            ref.append((g, e[key]))
            bisect.insort(sorted_vals, e[key])
    return out


# ---------------------------------------------------------------- book sim
def run_cell(events, px_cache, spy_ret, n_days, *, key, tier_adv, hold,
             cost_side, entry):
    """Build daily L/S book. Returns metrics dict."""
    long_s = np.zeros(n_days); long_n = np.zeros(n_days)
    short_s = np.zeros(n_days); short_n = np.zeros(n_days)
    ev_pnl = []          # per-event net L/S-signed return (for event share)
    n_long = n_short = 0
    first_g, last_g = None, 0
    for e in events:
        rk = e.get(f"rank_{key}")
        if rk is None or e["adv"] < tier_adv:
            continue
        if rk >= RANK_HI:
            side = 1
        elif rk <= RANK_LO:
            side = -1
        else:
            continue
        px = px_cache[e["ticker"]]
        p0 = e["p_e"]
        if entry == "openE1":
            if p0 + 1 >= len(px["close"]):
                continue
            # day-1 ret = close(E+1)/open(E+1); then closes
            base = px["open"][p0 + 1]
            path_p = list(range(p0 + 1, min(p0 + 1 + hold, len(px["close"]))))
            closes = px["close"][path_p]
            days = px["gidx"][path_p]
            rets = np.diff(np.concatenate([[base], closes])) / \
                np.concatenate([[base], closes[:-1]])
        else:  # closeE
            p_end = min(p0 + hold, len(px["close"]) - 1)
            if p_end <= p0:
                continue
            closes = px["close"][p0:p_end + 1]
            days = px["gidx"][p0 + 1:p_end + 1]
            rets = np.diff(closes) / closes[:-1]
        if len(rets) == 0:
            continue
        rets = np.clip(rets, -WINSOR, WINSOR)
        contrib = side * rets
        contrib[0] -= cost_side
        contrib[-1] -= cost_side
        if side == 1:
            n_long += 1
            np.add.at(long_s, days, contrib)
            np.add.at(long_n, days, 1)
        else:
            n_short += 1
            np.add.at(short_s, days, contrib)
            np.add.at(short_n, days, 1)
        ev_pnl.append(float(contrib.sum()))
        fg, lg = int(days[0]), int(days[-1])
        first_g = fg if first_g is None else min(first_g, fg)
        last_g = max(last_g, lg)
    n_ev = n_long + n_short
    if n_ev < 50 or first_g is None:
        return {"n": n_ev, "skip": True}
    lm = np.divide(long_s, long_n, out=np.zeros_like(long_s), where=long_n > 0)
    sm = np.divide(short_s, short_n, out=np.zeros_like(short_s),
                   where=short_n > 0)
    book = 0.5 * lm + 0.5 * sm
    sl = slice(first_g, last_g + 1)
    br = book[sl]
    pnl = br * BOOK
    spy_sl = spy_ret[sl]
    mu, sd = br.mean(), br.std()
    sharpe = float(mu / sd * np.sqrt(252)) if sd > 0 else 0.0
    dn = br[br < 0].std()
    sortino = float(mu / dn * np.sqrt(252)) if dn and dn > 0 else 0.0
    gp = pnl[pnl > 0].sum(); gl = -pnl[pnl < 0].sum()
    pf = float(gp / gl) if gl > 0 else float("inf")
    act = pnl != 0
    wr = float((pnl[act] > 0).mean()) if act.any() else 0.0

    def _sh(x):
        return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 5 and x.std() > 0 else 0.0
    shg, shr = _sh(br[spy_sl > 0]), _sh(br[spy_sl <= 0])
    den = max(abs(shg), abs(shr))
    gap = float(abs(shg - shr) / den) if den > 0 else 0.0
    tot_abs = np.abs(pnl).sum()
    day_conc = float(np.abs(pnl).max() / tot_abs) if tot_abs > 0 else 0.0
    ev_pnl = np.array(ev_pnl)
    gp_ev = ev_pnl[ev_pnl > 0].sum()
    ev_share = float(ev_pnl.max() / gp_ev) if gp_ev > 0 else 1.0
    return {
        "n": n_ev, "n_long": n_long, "n_short": n_short,
        "total_pnl": float(pnl.sum()), "ann_ret_pct": float(mu * 252 * 100),
        "sharpe": sharpe, "sortino": sortino, "pf": pf, "wr_days": wr,
        "sharpe_green": shg, "sharpe_red": shr, "regime_gap": gap,
        "day_conc": day_conc, "max_event_share": ev_share,
        "first_day": first_g, "last_day": last_g,
        "_book": br, "_first_g": first_g,
    }


def per_year(m, cal):
    br = m.pop("_book"); fg = m.pop("_first_g")
    yrs = {}
    for i, r in enumerate(br):
        y = cal[fg + i][:4]
        yrs[y] = yrs.get(y, 0.0) + r * BOOK
    m["per_year"] = {y: round(v, 1) for y, v in sorted(yrs.items())}
    pos = sum(1 for v in yrs.values() if v > 0)
    m["years_pos"], m["years_total"] = pos, len(yrs)
    return m


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cal, cal_idx, spy_ret = load_calendar_and_spy()
    n_days = len(cal)
    log(f"calendar {cal[0]}..{cal[-1]} ({n_days} days)")
    events, px_cache, drops = load_events(cal, cal_idx, spy_ret)
    ranked = assign_ranks(events, "reaction")
    ranked_sue = assign_ranks(events, "sue")
    log(f"ranked reaction={len(ranked)} sue={len(ranked_sue)}")
    pd.DataFrame([{k: v for k, v in e.items()} for e in ranked]).to_parquet(
        f"{OUT}/events.parquet")

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXP)

    cells = {}
    grid = []
    for tier, adv in TIERS.items():
        for h in HOLDS:
            grid.append(("reaction", tier, adv, h, "c10", "closeE"))   # primary
            grid.append(("reaction", tier, adv, h, "c25", "closeE"))   # G5 cost
            grid.append(("reaction", tier, adv, h, "c10", "openE1"))   # G5 entry
            grid.append(("sue", tier, adv, h, "c10", "closeE"))        # diagnostic
    for key, tier, adv, h, ck, entry in grid:
        evs = ranked if key == "reaction" else ranked_sue
        name = f"{key}|{tier}|H{h}|{ck}|{entry}"
        m = run_cell(evs, px_cache, spy_ret, n_days, key=key, tier_adv=adv,
                     hold=h, cost_side=COSTS[ck], entry=entry)
        if not m.get("skip"):
            m = per_year(m, cal)
        cells[name] = m
        log(f"{name}: n={m.get('n')} sharpe={m.get('sharpe', 0):.2f} "
            f"pf={m.get('pf', 0):.2f} gap={m.get('regime_gap', 9):.2f} "
            f"yrs+{m.get('years_pos', 0)}/{m.get('years_total', 0)}")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(dict(signal=key, tier=tier, hold=h, cost=ck,
                                   entry=entry, lane="pead_drift_v1"))
            for mk in ("n", "sharpe", "sortino", "pf", "wr_days",
                       "regime_gap", "day_conc", "max_event_share",
                       "total_pnl", "ann_ret_pct", "years_pos", "years_total"):
                if mk in m and np.isfinite(m.get(mk, np.nan)):
                    mlflow.log_metric(mk, float(m[mk]))

    # ---- verdict per pre-registered gates (PRIMARY: reaction/closeE/c10)
    verdict = "CLOSED_NEGATIVE"
    pass_cells = []
    for tier in TIERS:
        for h in HOLDS:
            c = cells.get(f"reaction|{tier}|H{h}|c10|closeE", {})
            if c.get("skip") or c.get("n", 0) < 2000 or c.get("sharpe", 0) < 1.0:
                continue
            # plateau: adjacent holds >= 0.5
            hs = HOLDS
            i = hs.index(h)
            adj = [hs[j] for j in (i - 1, i + 1) if 0 <= j < len(hs)]
            if not all(cells.get(f"reaction|{tier}|H{a}|c10|closeE",
                                 {}).get("sharpe", 0) >= 0.5 for a in adj):
                continue
            g2 = c["years_pos"] >= 0.70 * c["years_total"]
            g3 = c["regime_gap"] <= 0.50
            g4 = c["day_conc"] <= 0.70 and c["max_event_share"] <= 0.10
            c25 = cells.get(f"reaction|{tier}|H{h}|c25|closeE", {})
            oe1 = cells.get(f"reaction|{tier}|H{h}|c10|openE1", {})
            g5 = c25.get("sharpe", 0) >= 0.5 and oe1.get("sharpe", 0) >= 0.5
            gates = dict(G1=True, G2=bool(g2), G3=bool(g3), G4=bool(g4),
                         G5=bool(g5))
            c["gates"] = gates
            if all(gates.values()):
                pass_cells.append(f"reaction|{tier}|H{h}|c10|closeE")
    if pass_cells:
        verdict = "PASS_CANDIDATE"

    summary = {
        "design": "see strategy/pead_drift_v1.py docstring (pre-registered)",
        "n_events_total": len(events), "n_ranked": len(ranked),
        "drops": drops,
        "cells": {k: {kk: vv for kk, vv in v.items()
                      if not kk.startswith("_")} for k, v in cells.items()},
        "passing_cells": pass_cells, "verdict": verdict,
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    with mlflow.start_run(run_name="SUMMARY"):
        mlflow.log_param("verdict", verdict)
        mlflow.log_param("passing_cells", ",".join(pass_cells) or "none")
        mlflow.log_artifact(f"{OUT}/summary.json")
    log("VERDICT:", verdict, pass_cells)


if __name__ == "__main__":
    main()
