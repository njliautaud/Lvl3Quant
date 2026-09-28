#!/usr/bin/env python3
"""
earnings_vrp_v1 — Single-name earnings IV-crush harvest on REAL option chains.
================================================================================
PRE-REGISTERED DESIGN (written 2026-06-11 BEFORE any results were computed).
Lane opened under HC #603 R1 (quant experimentation reopened, non-ES data).

QUESTION
--------
Does the classic retail "sell vol into earnings, buy back after the crush"
trade survive REAL single-name bid/ask spreads + fees, regime-agnostically,
on 68 megacap names 2019-2026?

DATA (all already on disk, zero spend)
--------------------------------------
- Real chains: wheel_strategy_v1/data/cache/options_real/chains/<T>.parquet
  (68 tickers, 2019-02 .. 2026-06, ~3 snapshots/week pre-2024, near-daily
  2025+, real bid/ask + greeks + IV).
- Earnings dates: FMP financials income_quarter.json acceptedDate (10-Q/K
  filing, typically morning after the call) refined by max |overnight gap|
  in [accepted-10d, accepted+1d] from prices_v2.parquet; event REQUIRES
  |gap| >= 1.5% (drops mislabeled filings; conservative).
- Underlying + SPY daily prices: wheel_strategy_v1/data/cache/prices_v2.parquet.

EVENT / TRADE CONSTRUCTION (fixed before running)
-------------------------------------------------
- gap_day E = trading day with max |open/prev_close - 1| in filing window.
- Entry snapshot D_in = last chain date < E with (E - D_in) <= 4 calendar
  days. Exit snapshot D_out = first chain date >= E with (D_out - E) <= 4
  calendar days. Both must exist else event dropped.
- Expiry: smallest expiration with expiration > D_out and DTE at entry <= 30.
- Structures (1-lot, sell-to-open at D_in, buy-to-close at D_out):
  * straddle  : short ATM call + ATM put (strike argmin |K - spot|).
  * strangle25: short call closest to delta +0.25 and put closest to -0.25
                (require |delta - target| <= 0.10).
  * ironfly   : straddle + long wings at |delta| closest to 0.10 (defined risk).
- Quote sanity per leg: bid > 0, ask >= bid, (ask-bid)/mid <= 0.60; any leg
  failing at entry OR exit drops the event for that structure (counted).

COSTS (stated upfront — retail, Schwab-style)
---------------------------------------------
- PRIMARY  "cross": shorts open @ bid / close @ ask; longs open @ ask /
  close @ bid (full spread cross both ways — worst case).
- SECONDARY "mid": fills at mid +/- 25% of half-spread adverse (realistic
  resting limit near mid).
- Fees: $0.65 per contract per leg per side, both models.

METRICS / AGGREGATION
---------------------
- Per-event net P&L ($, 1-lot) and return-on-notional = pnl / (100*spot_in).
- Book Sharpe: $100k book, 5% of equity notional per event, daily P&L
  series over ALL trading days in span (zeros on no-exit days),
  Sharpe = mean/std * sqrt(252). Sortino analog with downside std.
- WR, PF, per-year totals, regime split.

GATES (PRE-REGISTERED — pass ALL or verdict = CLOSED NEGATIVE)
--------------------------------------------------------------
- G1: at PRIMARY (cross) costs, >=1 structure with n >= 300 events has
      mean net return-on-notional > 0 AND book Sharpe >= 1.0.
- G2: that structure's net total P&L positive in >= 5 of the 8 calendar
      years 2019..2026 (all-regime, HC #428 R1 spirit).
- G3: regime gap (events split green/red by SPY return D_in->D_out):
      |Sharpe_g - Sharpe_r| / max(|Sharpe_g|,|Sharpe_r|) <= 0.50
      (per-event-return Sharpe by group).
- G4: concentration: max single-event share of gross profit <= 10%; max
      single exit-day share of total |P&L| <= 0.70 (HC #344 analog).
- G5: plateau: richness filter sweep tau in {none,1.0,1.1,1.2,1.3,1.4}
      where richness = implied move (ATM straddle mid / spot at entry)
      / expanding past-only median realized |gap| of the ticker (min 4
      events, else global expanding median). If G1 passes ONLY at one tau,
      neighbors +/-0.1 must also pass G1 → otherwise THRESHOLD_LUCK, reject.
- No parameter fitting anywhere; all conditioning is expanding past-only,
  so the whole 2019-2026 stream is OOT. Sliding-window rules (HC #0) not
  triggered (no training).

MLflow: experiment earnings_vrp_v1 @ http://localhost:5000 (mandatory).
Artifacts: /home/jupiter/Lvl3Quant/results/earnings_vrp_v1/
DO NOT touch wheel engines / paper traders. CPU only (Jupiter).
"""
import json
import os
import sys
import glob
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = "/home/jupiter/Lvl3Quant"
CHAINS_DIR = f"{ROOT}/wheel_strategy_v1/data/cache/options_real/chains"
PRICES_PQ = f"{ROOT}/wheel_strategy_v1/data/cache/prices_v2.parquet"
FIN_DIR = "/home/jupiter/teleclaude-main/data/fmp_archive/financials"
OUT_DIR = f"{ROOT}/results/earnings_vrp_v1"
os.makedirs(OUT_DIR, exist_ok=True)

FEE_PER_CONTRACT = 0.65
MAX_SPREAD_FRAC = 0.60
GAP_MIN = 0.015
ENTRY_MAX_CAL_DAYS = 4
EXIT_MAX_CAL_DAYS = 4
DTE_MAX = 30
BOOK_NOTIONAL_FRAC = 0.05  # 5% of $100k book per event
TAUS = [None, 1.0, 1.1, 1.2, 1.3, 1.4]
STRUCTURES = ["straddle", "strangle25", "ironfly"]
COST_MODELS = ["cross", "mid"]


def load_prices():
    df = pd.read_parquet(PRICES_PQ)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_events(tickers, prices):
    """Earnings events: filing acceptedDate refined to max-|gap| trading day."""
    events = []
    px = {t: g.sort_values("date").reset_index(drop=True)
          for t, g in prices[prices.ticker.isin(tickers)].groupby("ticker")}
    for t in tickers:
        fp = f"{FIN_DIR}/{t}/income_quarter.json"
        if not os.path.exists(fp) or t not in px:
            continue
        g = px[t]
        g = g.assign(prev_close=g["close"].shift(1))
        g["gap"] = g["open"] / g["prev_close"] - 1.0
        try:
            rows = json.load(open(fp))
        except Exception:
            continue
        for r in rows:
            acc = r.get("acceptedDate")
            if not acc:
                continue
            acc = pd.Timestamp(acc[:10])
            lo, hi = acc - timedelta(days=10), acc + timedelta(days=1)
            w = g[(g.date >= lo) & (g.date <= hi)].dropna(subset=["gap"])
            if w.empty:
                continue
            i = w["gap"].abs().idxmax()
            gap = w.loc[i, "gap"]
            if abs(gap) < GAP_MIN:
                continue
            events.append(dict(ticker=t, gap_day=w.loc[i, "date"], gap=gap,
                               accepted=acc, period=r.get("period"),
                               fiscal=r.get("date")))
    ev = pd.DataFrame(events)
    if ev.empty:
        return ev
    ev = (ev.sort_values(["ticker", "gap_day"])
            .drop_duplicates(subset=["ticker", "gap_day"])
            .reset_index(drop=True))
    # collapse events <30d apart per ticker (same announcement matched twice)
    keep = []
    for t, g in ev.groupby("ticker"):
        last = None
        for _, r in g.iterrows():
            if last is not None and (r.gap_day - last).days < 30:
                continue
            keep.append(r)
            last = r.gap_day
    return pd.DataFrame(keep).reset_index(drop=True)


def fill_price(bid, ask, side_open):
    """Return (cross_fill, mid_fill) for opening; side_open='sell' or 'buy'."""
    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0
    if side_open == "sell":
        return bid, mid - 0.25 * half
    return ask, mid + 0.25 * half


def leg_ok(row):
    if row is None:
        return False
    b, a = row["bid"], row["ask"]
    if not (b > 0 and a >= b):
        return False
    mid = (a + b) / 2.0
    return mid > 0 and (a - b) / mid <= MAX_SPREAD_FRAC


def get_leg(chain, strike, typ):
    r = chain[(chain.strike == strike) & (chain.type == typ)]
    return None if r.empty else r.iloc[0]


def pick_by_delta(chain, typ, target, tol=0.10):
    c = chain[chain.type == typ].dropna(subset=["delta"])
    if c.empty:
        return None
    i = (c["delta"] - target).abs().idxmin()
    r = c.loc[i]
    return r if abs(r["delta"] - target) <= tol else None


def simulate():
    prices = load_prices()
    tickers = sorted(os.path.basename(f)[:-8]
                     for f in glob.glob(f"{CHAINS_DIR}/*.parquet"))
    ev = build_events(tickers, prices)
    print(f"events detected: {len(ev)} across {ev.ticker.nunique()} tickers")

    spot_map = {(r.ticker, r.date): r.close for r in
                prices[prices.ticker.isin(tickers)].itertuples()}
    spy = prices[prices.ticker == "SPY"].set_index("date")["close"]

    drops = {"no_entry_snap": 0, "no_exit_snap": 0, "no_expiry": 0,
             "no_spot": 0}
    leg_drops = {s: 0 for s in STRUCTURES}
    rows = []
    for t, tev in ev.groupby("ticker"):
        ch = pd.read_parquet(f"{CHAINS_DIR}/{t}.parquet")
        ch["date"] = pd.to_datetime(ch["date"])
        ch["expiration"] = pd.to_datetime(ch["expiration"])
        snap_dates = np.sort(ch["date"].unique())
        for _, e in tev.iterrows():
            E = e.gap_day
            pre = snap_dates[snap_dates < np.datetime64(E)]
            post = snap_dates[snap_dates >= np.datetime64(E)]
            if len(pre) == 0 or (E - pd.Timestamp(pre[-1])).days > ENTRY_MAX_CAL_DAYS:
                drops["no_entry_snap"] += 1
                continue
            if len(post) == 0 or (pd.Timestamp(post[0]) - E).days > EXIT_MAX_CAL_DAYS:
                drops["no_exit_snap"] += 1
                continue
            d_in, d_out = pd.Timestamp(pre[-1]), pd.Timestamp(post[0])
            spot = spot_map.get((t, d_in))
            if spot is None or not np.isfinite(spot) or spot <= 0:
                drops["no_spot"] += 1
                continue
            cin_all = ch[ch.date == d_in]
            cout_all = ch[ch.date == d_out]
            exps = np.sort(np.intersect1d(cin_all.expiration.unique(),
                                          cout_all.expiration.unique()))
            exps = [x for x in exps
                    if pd.Timestamp(x) > d_out
                    and (pd.Timestamp(x) - d_in).days <= DTE_MAX]
            if not exps:
                drops["no_expiry"] += 1
                continue
            exp = pd.Timestamp(exps[0])
            cin = cin_all[cin_all.expiration == exp]
            cout = cout_all[cout_all.expiration == exp]

            # ATM strike (both types present at entry)
            ks = np.intersect1d(cin[cin.type == "c"].strike.unique(),
                                cin[cin.type == "p"].strike.unique())
            if len(ks) == 0:
                drops["no_expiry"] += 1
                continue
            k_atm = ks[np.argmin(np.abs(ks - spot))]

            atm_c, atm_p = get_leg(cin, k_atm, "c"), get_leg(cin, k_atm, "p")
            implied_move = np.nan
            if atm_c is not None and atm_p is not None and atm_c.bid > 0 and atm_p.bid > 0:
                implied_move = ((atm_c.bid + atm_c.ask) / 2 +
                                (atm_p.bid + atm_p.ask) / 2) / spot

            spy_in = spy.asof(d_in)
            spy_out = spy.asof(d_out)
            spy_ret = spy_out / spy_in - 1.0 if spy_in and spy_out else np.nan

            base = dict(ticker=t, gap_day=E, gap=e.gap, d_in=d_in, d_out=d_out,
                        expiration=exp, spot_in=spot, dte_in=(exp - d_in).days,
                        implied_move=implied_move, spy_ret=spy_ret,
                        year=E.year)

            for struct in STRUCTURES:
                if struct == "straddle":
                    legs = [(k_atm, "c", -1), (k_atm, "p", -1)]
                elif struct == "strangle25":
                    lc = pick_by_delta(cin, "c", 0.25)
                    lp = pick_by_delta(cin, "p", -0.25)
                    if lc is None or lp is None:
                        leg_drops[struct] += 1
                        continue
                    legs = [(lc.strike, "c", -1), (lp.strike, "p", -1)]
                else:  # ironfly
                    wc = pick_by_delta(cin, "c", 0.10)
                    wp = pick_by_delta(cin, "p", -0.10)
                    if wc is None or wp is None or wc.strike <= k_atm or wp.strike >= k_atm:
                        leg_drops[struct] += 1
                        continue
                    legs = [(k_atm, "c", -1), (k_atm, "p", -1),
                            (wc.strike, "c", +1), (wp.strike, "p", +1)]

                ok = True
                pnl = {m: 0.0 for m in COST_MODELS}
                for k, typ, pos in legs:
                    lin, lout = get_leg(cin, k, typ), get_leg(cout, k, typ)
                    if not (leg_ok(lin) and leg_ok(lout)):
                        ok = False
                        break
                    side_open = "sell" if pos < 0 else "buy"
                    side_close = "buy" if pos < 0 else "sell"
                    oc, om = fill_price(lin.bid, lin.ask, side_open)
                    cc, cm = fill_price(lout.bid, lout.ask, side_close)
                    sgn = -pos  # short: +open -close ; long: -open +close
                    pnl["cross"] += sgn * (oc - cc)
                    pnl["mid"] += sgn * (om - cm)
                if not ok:
                    leg_drops[struct] += 1
                    continue
                n_legs = len(legs)
                fees = FEE_PER_CONTRACT * n_legs * 2
                for m in COST_MODELS:
                    dollars = pnl[m] * 100.0 - fees
                    rows.append(dict(base, structure=struct, cost_model=m,
                                     pnl_dollars=dollars,
                                     ret_notional=dollars / (100.0 * spot)))
    res = pd.DataFrame(rows)
    print(f"trade rows: {len(res)}  drops: {drops}  leg_drops: {leg_drops}")
    return ev, res, drops, leg_drops


def add_richness(res):
    """Expanding past-only richness per (ticker, gap_day)."""
    ev = (res[["ticker", "gap_day", "gap", "implied_move"]]
          .drop_duplicates(subset=["ticker", "gap_day"])
          .sort_values("gap_day").reset_index(drop=True))
    ev["abs_gap"] = ev["gap"].abs()
    rich = {}
    hist = {}
    glob_hist = []
    for _, r in ev.iterrows():
        h = hist.setdefault(r.ticker, [])
        if len(h) >= 4:
            base = np.median(h)
        elif len(glob_hist) >= 20:
            base = np.median(glob_hist)
        else:
            base = np.nan
        rich[(r.ticker, r.gap_day)] = (r.implied_move / base
                                       if base and np.isfinite(base) and base > 0
                                       and np.isfinite(r.implied_move) else np.nan)
        h.append(r.abs_gap)
        glob_hist.append(r.abs_gap)
    res["richness"] = [rich.get((t, d), np.nan)
                       for t, d in zip(res.ticker, res.gap_day)]
    return res


def book_sharpe(sub):
    """Daily P&L on $100k book, 5% notional/event, all weekdays incl zeros."""
    if sub.empty:
        return np.nan, np.nan
    daily = sub.groupby("d_out").apply(
        lambda g: (BOOK_NOTIONAL_FRAC * g["ret_notional"]).sum())
    idx = pd.bdate_range(sub.d_in.min(), sub.d_out.max())
    daily = daily.reindex(idx, fill_value=0.0)
    if daily.std() == 0:
        return np.nan, np.nan
    sh = daily.mean() / daily.std() * np.sqrt(252)
    dn = daily[daily < 0].std()
    so = daily.mean() / dn * np.sqrt(252) if dn and dn > 0 else np.nan
    return float(sh), float(so)


def cell_metrics(sub):
    if len(sub) == 0:
        return dict(n=0)
    r = sub["ret_notional"]
    wins, losses = sub.pnl_dollars[sub.pnl_dollars > 0], sub.pnl_dollars[sub.pnl_dollars <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    sh, so = book_sharpe(sub)
    yr = sub.groupby("year")["pnl_dollars"].sum()
    g = sub[sub.spy_ret > 0]
    rd = sub[sub.spy_ret <= 0]
    def evsh(x):
        return (x["ret_notional"].mean() / x["ret_notional"].std()
                if len(x) > 5 and x["ret_notional"].std() > 0 else np.nan)
    sg, sr = evsh(g), evsh(rd)
    gap = (abs(sg - sr) / max(abs(sg), abs(sr))
           if np.isfinite(sg) and np.isfinite(sr) and max(abs(sg), abs(sr)) > 0
           else np.nan)
    gross_profit = wins.sum()
    max_ev_share = (sub.pnl_dollars.max() / gross_profit
                    if gross_profit > 0 else np.nan)
    dabs = sub.groupby("d_out")["pnl_dollars"].sum().abs()
    day_conc = dabs.max() / dabs.sum() if dabs.sum() > 0 else np.nan
    return dict(
        n=int(len(sub)), mean_ret=float(r.mean()), median_ret=float(r.median()),
        total_pnl=float(sub.pnl_dollars.sum()),
        mean_pnl=float(sub.pnl_dollars.mean()),
        wr=float((sub.pnl_dollars > 0).mean()), pf=float(pf),
        book_sharpe=sh, book_sortino=so,
        years_pos=int((yr > 0).sum()), years_total=int(len(yr)),
        per_year={int(k): float(v) for k, v in yr.items()},
        sharpe_green=None if not np.isfinite(sg) else float(sg),
        sharpe_red=None if not np.isfinite(sr) else float(sr),
        regime_gap=None if not np.isfinite(gap) else float(gap),
        n_green=int(len(g)), n_red=int(len(rd)),
        max_event_share=None if not np.isfinite(max_ev_share) else float(max_ev_share),
        day_conc=None if not np.isfinite(day_conc) else float(day_conc),
    )


def gates(m):
    if m.get("n", 0) < 300:
        return dict(G1=False, reason="n<300")
    g1 = m["mean_ret"] > 0 and (m["book_sharpe"] or -9) >= 1.0
    g2 = m["years_pos"] >= 5
    g3 = m["regime_gap"] is not None and m["regime_gap"] <= 0.50
    g4 = ((m["max_event_share"] or 1) <= 0.10 and (m["day_conc"] or 1) <= 0.70)
    return dict(G1=bool(g1), G2=bool(g2), G3=bool(g3), G4=bool(g4))


def main():
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("earnings_vrp_v1")

    ev, res, drops, leg_drops = simulate()
    res = add_richness(res)
    res.to_parquet(f"{OUT_DIR}/events_trades.parquet")
    ev.to_parquet(f"{OUT_DIR}/earnings_events.parquet")

    summary = dict(design="see strategy/earnings_vrp_v1.py docstring (pre-registered)",
                   n_events=int(len(ev)), drops=drops, leg_drops=leg_drops,
                   cells={})
    with mlflow.start_run(run_name="earnings_vrp_v1_main"):
        mlflow.log_params(dict(fee=FEE_PER_CONTRACT, gap_min=GAP_MIN,
                               dte_max=DTE_MAX, max_spread=MAX_SPREAD_FRAC,
                               n_events=len(ev)))
        for struct in STRUCTURES:
            for cm in COST_MODELS:
                for tau in TAUS:
                    sub = res[(res.structure == struct) & (res.cost_model == cm)]
                    tname = "none" if tau is None else f"{tau:.1f}"
                    if tau is not None:
                        sub = sub[sub.richness >= tau]
                    key = f"{struct}|{cm}|tau_{tname}"
                    m = cell_metrics(sub)
                    m["gates"] = gates(m) if m.get("n", 0) else {}
                    summary["cells"][key] = m
                    if m.get("n", 0):
                        with mlflow.start_run(run_name=key, nested=True):
                            mlflow.log_params(dict(structure=struct,
                                                   cost_model=cm, tau=tname))
                            for mk in ["n", "mean_ret", "total_pnl", "wr", "pf",
                                       "book_sharpe", "book_sortino",
                                       "years_pos", "regime_gap",
                                       "max_event_share", "day_conc"]:
                                v = m.get(mk)
                                if v is not None and np.isfinite(v) if isinstance(v, float) else v is not None:
                                    try:
                                        mlflow.log_metric(mk, float(v))
                                    except Exception:
                                        pass

        # verdict per pre-registered gates (PRIMARY = cross cost model)
        passing = []
        for struct in STRUCTURES:
            for tau in TAUS:
                tname = "none" if tau is None else f"{tau:.1f}"
                m = summary["cells"][f"{struct}|cross|tau_{tname}"]
                gt = m.get("gates", {})
                if gt and all(gt.get(k) for k in ["G1", "G2", "G3", "G4"]):
                    passing.append((struct, tname))
        # G5 plateau: if pass exists only via tau-filter, neighbors must pass G1
        verdict = "CLOSED_NEGATIVE"
        plateau_notes = []
        for struct, tname in passing:
            if tname == "none":
                verdict = "PASS_CANDIDATE"
                continue
            tau = float(tname)
            neigh = [round(tau - 0.1, 1), round(tau + 0.1, 1)]
            ok = True
            for nt in neigh:
                if nt in [t for t in TAUS if t is not None]:
                    nm = summary["cells"][f"{struct}|cross|tau_{nt:.1f}"]
                    if not nm.get("gates", {}).get("G1"):
                        ok = False
            if ok:
                verdict = "PASS_CANDIDATE"
            else:
                plateau_notes.append(f"{struct} tau={tname}: THRESHOLD_LUCK")
        summary["passing_cells_cross"] = passing
        summary["plateau_notes"] = plateau_notes
        summary["verdict"] = verdict
        mlflow.log_param("verdict", verdict)
        json.dump(summary, open(f"{OUT_DIR}/summary.json", "w"), indent=1,
                  default=str)
        mlflow.log_artifact(f"{OUT_DIR}/summary.json")
    print("VERDICT:", verdict)
    print("passing (cross):", passing, "| plateau:", plateau_notes)


if __name__ == "__main__":
    main()
