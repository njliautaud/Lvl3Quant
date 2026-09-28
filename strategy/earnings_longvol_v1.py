#!/usr/bin/env python3
"""
earnings_longvol_v1 — LONG-VOL PRE-EARNINGS on REAL single-name chains.
================================================================================
PRE-REGISTERED DESIGN (written 2026-06-11 BEFORE any results were computed).
Lane: HC #603 R1. Direct follow-up to earnings_vrp_v1 (CLOSED NEGATIVE: short
earnings vol gross edge ~= +0.9% notional == half-spread cost). This is the
ONE same-data follow-up that experiment left unexplored: buy the documented
pre-earnings IV run-up, exit BEFORE the announcement, never hold the crush.

QUESTION
--------
Does buying single-name vol N trading days before earnings and selling it at
T-1 (the IV peak) harvest enough IV-ramp + gamma to cover theta bleed + the
full retail bid/ask spread, regime-agnostically, 2019-2026?

KNOWN FAILURE MODE TO MEASURE EXPLICITLY: long premium bleeds theta. We
decompose gross (pure-mid, no fee) edge vs spread+fee cost, and report mean
ATM IV change entry->exit plus theta-bleed estimate. If gross edge < spread
cost again, the family closes plainly.

DATA (all on disk, zero spend — identical to sibling)
-----------------------------------------------------
- Real chains: wheel_strategy_v1/data/cache/options_real/chains/<T>.parquet
  (68 megacaps usable, 2019-02..2026-06, ~M/W/F snapshots pre-2024,
  near-daily 2025+, real bid/ask + greeks + IV in column `vol`).
- Earnings events: REUSED VERBATIM from sibling —
  results/earnings_vrp_v1/earnings_events.parquet (1,872 events; FMP
  acceptedDate refined to max-|overnight gap| day, |gap|>=1.5%, point-in-time
  announcement timing). No event re-derivation = no new dataset df.
- Underlying + SPY daily prices: wheel_strategy_v1/data/cache/prices_v2.parquet.

TRADE CONSTRUCTION (fixed before running)
-----------------------------------------
- E = gap_day (first trading day reflecting the announcement). The trade must
  be CLOSED strictly before E.
- Entry lag L in {10, 7, 5} TRADING days (ticker's own price calendar):
  target = calendar[idx(E) - L]. Entry snapshot D_in = last chain date
  <= target with (target - D_in) <= 4 calendar days.
- Exit snapshot D_out = LAST chain date < E with (E - D_out) <= 4 calendar
  days (T-1 close proxy under snapshot sparsity; never >= E). Require
  D_out > D_in (else event dropped for that lag).
- Expiry: smallest expiration present at BOTH snapshots with
  expiration > E (the earnings-premium expiry) and DTE at entry <= 45.
- Structures (1-lot, BUY-to-open at D_in, SELL-to-close at D_out):
  * straddle  : long ATM call + ATM put (strike argmin |K - spot_in|).
  * strangle25: long call closest to delta +0.25 and put closest to -0.25
                (require |delta - target| <= 0.10).
  * atm_call  : long ATM call only (drift + ramp, half the spread bill).
- Quote sanity per leg at entry AND exit: bid > 0, ask >= bid,
  (ask-bid)/mid <= 0.60; failing leg drops the event for that structure.

COSTS (stated upfront — retail, identical to sibling)
-----------------------------------------------------
- PRIMARY  "cross": longs open @ ask / close @ bid (full spread both ways).
- SECONDARY "mid": fills at mid +/- 25% of half-spread adverse.
- DECOMPOSITION-ONLY "midpure": exact mid, zero fees — NOT a gate input,
  used solely to split gross edge vs cost (the sibling's closure insight).
- Fees: $0.65 per contract per leg per side (cross & mid models).

FILTER (expanding past-only, whole stream stays OOT)
----------------------------------------------------
- richness = implied move at D_in (ATM straddle mid / spot) / expanding
  past-only median realized |gap| of the ticker (min 4 past events, else
  global expanding median, else NaN -> excluded when filter active).
  Long vol wants CHEAP vol: trade only if richness <= tau,
  tau in {none, 1.3, 1.2, 1.1, 1.0, 0.9}.

METRICS / AGGREGATION — identical machinery to sibling
------------------------------------------------------
- Per-event net P&L ($, 1-lot), ret_notional = pnl / (100*spot_in).
- Book Sharpe: $100k book, 5% notional/event, daily P&L over all weekdays
  in span (zeros included), annualized; Sortino analog.
- Diagnostics per event: ATM IV entry/exit (d_iv), theta-bleed estimate
  (sum theta_in * hold_days * 100 per leg), hold_days.

GATES (PRE-REGISTERED — pass ALL or verdict = CLOSED NEGATIVE)
--------------------------------------------------------------
- G1: at PRIMARY (cross) costs, cell (structure, lag, tau) with n >= 300:
      mean net ret_notional > 0 AND book Sharpe >= 1.0.
- G2: net total P&L positive in >= 5 of 8 calendar years 2019..2026.
- G3: regime gap (SPY return D_in->D_out, green/red):
      |Sh_g - Sh_r| / max(|Sh_g|,|Sh_r|) <= 0.50.
- G4: max single-event share of gross profit <= 10%; max single exit-day
      share of total |P&L| <= 0.70.
- G5: PLATEAU across BOTH knobs: a passing cell must have (a) at least one
      NEIGHBOR LAG (same structure, same tau) also passing G1, and (b) if
      tau != none, both tau neighbors (+/-0.1 within sweep) passing G1.
      Single-cell pass = THRESHOLD_LUCK, reject.
- No fitting anywhere. Honest negative is a valuable outcome: if gross
  midpure edge < spread cost, say so and close the long-vol family too.

MLflow: experiment earnings_longvol_v1 @ http://localhost:5000 (mandatory).
Artifacts: /home/jupiter/Lvl3Quant/results/earnings_longvol_v1/
DO NOT touch wheel engines / paper traders. CPU only (Jupiter). Wall cap 4h.
"""
import json
import os
import glob
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = "/home/jupiter/Lvl3Quant"
CHAINS_DIR = f"{ROOT}/wheel_strategy_v1/data/cache/options_real/chains"
PRICES_PQ = f"{ROOT}/wheel_strategy_v1/data/cache/prices_v2.parquet"
EVENTS_PQ = f"{ROOT}/results/earnings_vrp_v1/earnings_events.parquet"
OUT_DIR = f"{ROOT}/results/earnings_longvol_v1"
os.makedirs(OUT_DIR, exist_ok=True)

FEE_PER_CONTRACT = 0.65
MAX_SPREAD_FRAC = 0.60
ENTRY_TARGET_TOL_CAL_DAYS = 4
EXIT_MAX_CAL_DAYS = 4
DTE_MAX = 45
BOOK_NOTIONAL_FRAC = 0.05
LAGS = [10, 7, 5]
TAUS = [None, 1.3, 1.2, 1.1, 1.0, 0.9]
STRUCTURES = ["straddle", "strangle25", "atm_call"]
COST_MODELS = ["cross", "mid", "midpure"]  # midpure = decomposition only
GATED_MODELS = ["cross", "mid"]


def fill_price(bid, ask, side_open):
    """(cross, mid+-25%half, pure mid) for a fill; side='sell'|'buy'."""
    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0
    if side_open == "sell":
        return bid, mid - 0.25 * half, mid
    return ask, mid + 0.25 * half, mid


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
    prices = pd.read_parquet(PRICES_PQ)
    prices["date"] = pd.to_datetime(prices["date"])
    ev = pd.read_parquet(EVENTS_PQ)
    ev["gap_day"] = pd.to_datetime(ev["gap_day"])
    tickers = sorted(os.path.basename(f)[:-8]
                     for f in glob.glob(f"{CHAINS_DIR}/*.parquet"))
    ev = ev[ev.ticker.isin(tickers)].reset_index(drop=True)
    print(f"events reused: {len(ev)} across {ev.ticker.nunique()} tickers")

    cal = {t: g["date"].sort_values().reset_index(drop=True)
           for t, g in prices[prices.ticker.isin(tickers)].groupby("ticker")}
    spot_map = {(r.ticker, r.date): r.close for r in
                prices[prices.ticker.isin(tickers)].itertuples()}
    spy = prices[prices.ticker == "SPY"].set_index("date")["close"]

    drops = {"no_cal": 0, "no_entry_snap": 0, "no_exit_snap": 0,
             "hold_nonpos": 0, "no_expiry": 0, "no_spot": 0}
    leg_drops = {s: 0 for s in STRUCTURES}
    rows = []
    for t, tev in ev.groupby("ticker"):
        ch = pd.read_parquet(f"{CHAINS_DIR}/{t}.parquet")
        ch["date"] = pd.to_datetime(ch["date"])
        ch["expiration"] = pd.to_datetime(ch["expiration"])
        snap_dates = np.sort(ch["date"].unique())
        tcal = cal.get(t)
        if tcal is None:
            drops["no_cal"] += len(tev) * len(LAGS)
            continue
        tcal_idx = pd.Series(np.arange(len(tcal)), index=tcal.values)
        for _, e in tev.iterrows():
            E = e.gap_day
            if E not in tcal_idx.index:
                drops["no_cal"] += len(LAGS)
                continue
            ei = int(tcal_idx[E])
            # exit snapshot: last chain date strictly < E, within tolerance
            pre = snap_dates[snap_dates < np.datetime64(E)]
            if len(pre) == 0 or (E - pd.Timestamp(pre[-1])).days > EXIT_MAX_CAL_DAYS:
                drops["no_exit_snap"] += len(LAGS)
                continue
            d_out = pd.Timestamp(pre[-1])
            cout_all = ch[ch.date == d_out]
            spy_out = spy.asof(d_out)

            for lag in LAGS:
                if ei - lag < 0:
                    drops["no_cal"] += 1
                    continue
                target = tcal.iloc[ei - lag]
                pe = snap_dates[snap_dates <= np.datetime64(target)]
                if len(pe) == 0 or (target - pd.Timestamp(pe[-1])).days > ENTRY_TARGET_TOL_CAL_DAYS:
                    drops["no_entry_snap"] += 1
                    continue
                d_in = pd.Timestamp(pe[-1])
                if d_out <= d_in:
                    drops["hold_nonpos"] += 1
                    continue
                spot = spot_map.get((t, d_in))
                if spot is None or not np.isfinite(spot) or spot <= 0:
                    drops["no_spot"] += 1
                    continue
                cin_all = ch[ch.date == d_in]
                exps = np.sort(np.intersect1d(cin_all.expiration.unique(),
                                              cout_all.expiration.unique()))
                exps = [x for x in exps
                        if pd.Timestamp(x) > E
                        and (pd.Timestamp(x) - d_in).days <= DTE_MAX]
                if not exps:
                    drops["no_expiry"] += 1
                    continue
                exp = pd.Timestamp(exps[0])
                cin = cin_all[cin_all.expiration == exp]
                cout = cout_all[cout_all.expiration == exp]

                ks = np.intersect1d(cin[cin.type == "c"].strike.unique(),
                                    cin[cin.type == "p"].strike.unique())
                if len(ks) == 0:
                    drops["no_expiry"] += 1
                    continue
                k_atm = ks[np.argmin(np.abs(ks - spot))]

                atm_c, atm_p = get_leg(cin, k_atm, "c"), get_leg(cin, k_atm, "p")
                implied_move = np.nan
                iv_in = np.nan
                if atm_c is not None and atm_p is not None \
                        and atm_c.bid > 0 and atm_p.bid > 0:
                    implied_move = ((atm_c.bid + atm_c.ask) / 2 +
                                    (atm_p.bid + atm_p.ask) / 2) / spot
                    iv_in = np.nanmean([atm_c["vol"], atm_p["vol"]])
                oc_atm_c, oc_atm_p = get_leg(cout, k_atm, "c"), get_leg(cout, k_atm, "p")
                iv_out = np.nan
                if oc_atm_c is not None and oc_atm_p is not None:
                    iv_out = np.nanmean([oc_atm_c["vol"], oc_atm_p["vol"]])

                spy_in = spy.asof(d_in)
                spy_ret = (spy_out / spy_in - 1.0
                           if spy_in and spy_out else np.nan)
                hold_days = (d_out - d_in).days

                base = dict(ticker=t, gap_day=E, gap=e.gap, lag=lag,
                            d_in=d_in, d_out=d_out, expiration=exp,
                            spot_in=spot, dte_in=(exp - d_in).days,
                            hold_days=hold_days, implied_move=implied_move,
                            iv_in=iv_in, iv_out=iv_out,
                            d_iv=iv_out - iv_in, spy_ret=spy_ret,
                            year=E.year)

                for struct in STRUCTURES:
                    if struct == "straddle":
                        legs = [(k_atm, "c", +1), (k_atm, "p", +1)]
                    elif struct == "strangle25":
                        lc = pick_by_delta(cin, "c", 0.25)
                        lp = pick_by_delta(cin, "p", -0.25)
                        if lc is None or lp is None:
                            leg_drops[struct] += 1
                            continue
                        legs = [(lc.strike, "c", +1), (lp.strike, "p", +1)]
                    else:  # atm_call
                        legs = [(k_atm, "c", +1)]

                    ok = True
                    pnl = {m: 0.0 for m in COST_MODELS}
                    theta_bleed = 0.0
                    for k, typ, pos in legs:
                        lin, lout = get_leg(cin, k, typ), get_leg(cout, k, typ)
                        if not (leg_ok(lin) and leg_ok(lout)):
                            ok = False
                            break
                        side_open = "buy" if pos > 0 else "sell"
                        side_close = "sell" if pos > 0 else "buy"
                        oc, om, op = fill_price(lin.bid, lin.ask, side_open)
                        cc, cm, cp = fill_price(lout.bid, lout.ask, side_close)
                        sgn = -pos  # long: -open +close
                        pnl["cross"] += sgn * (oc - cc)
                        pnl["mid"] += sgn * (om - cm)
                        pnl["midpure"] += sgn * (op - cp)
                        th = lin.get("theta")
                        if th is not None and np.isfinite(th):
                            theta_bleed += th * hold_days * 100.0 * pos
                    if not ok:
                        leg_drops[struct] += 1
                        continue
                    n_legs = len(legs)
                    fees = FEE_PER_CONTRACT * n_legs * 2
                    for m in COST_MODELS:
                        f = 0.0 if m == "midpure" else fees
                        dollars = pnl[m] * 100.0 - f
                        rows.append(dict(base, structure=struct, cost_model=m,
                                         n_legs=n_legs,
                                         theta_bleed=theta_bleed,
                                         pnl_dollars=dollars,
                                         ret_notional=dollars / (100.0 * spot)))
    res = pd.DataFrame(rows)
    print(f"trade rows: {len(res)}  drops: {drops}  leg_drops: {leg_drops}")
    return ev, res, drops, leg_drops


def add_richness(res):
    """Expanding past-only richness per (ticker, gap_day) — sibling's recipe.
    Computed on the event stream in gap_day order (lag-independent)."""
    ev = (res[["ticker", "gap_day", "gap", "implied_move"]]
          .drop_duplicates(subset=["ticker", "gap_day"])
          .sort_values("gap_day").reset_index(drop=True))
    ev["abs_gap"] = ev["gap"].abs()
    rich, hist, glob_hist = {}, {}, []
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
    wins = sub.pnl_dollars[sub.pnl_dollars > 0]
    losses = sub.pnl_dollars[sub.pnl_dollars <= 0]
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
           if np.isfinite(sg) and np.isfinite(sr)
           and max(abs(sg), abs(sr)) > 0 else np.nan)
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
        mean_d_iv=float(sub["d_iv"].mean()),
        mean_theta_bleed=float(sub["theta_bleed"].mean()),
        mean_hold_days=float(sub["hold_days"].mean()),
    )


def gates(m):
    if m.get("n", 0) < 300:
        return dict(G1=False, reason="n<300")
    g1 = m["mean_ret"] > 0 and (m["book_sharpe"] or -9) >= 1.0
    g2 = m["years_pos"] >= 5
    g3 = m["regime_gap"] is not None and m["regime_gap"] <= 0.50
    g4 = ((m["max_event_share"] or 1) <= 0.10 and (m["day_conc"] or 1) <= 0.70)
    return dict(G1=bool(g1), G2=bool(g2), G3=bool(g3), G4=bool(g4))


def cell_key(struct, lag, cm, tau):
    tname = "none" if tau is None else f"{tau:.1f}"
    return f"{struct}|lag{lag}|{cm}|tau_{tname}"


def select(res, struct, lag, cm, tau):
    sub = res[(res.structure == struct) & (res.lag == lag) &
              (res.cost_model == cm)]
    if tau is not None:
        sub = sub[sub.richness <= tau]
    return sub


def main():
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("earnings_longvol_v1")

    ev, res, drops, leg_drops = simulate()
    res = add_richness(res)
    res.to_parquet(f"{OUT_DIR}/events_trades.parquet")

    summary = dict(design="see strategy/earnings_longvol_v1.py docstring "
                          "(pre-registered)",
                   n_events_reused=int(len(ev)), drops=drops,
                   leg_drops=leg_drops, cells={})

    with mlflow.start_run(run_name="earnings_longvol_v1_main"):
        mlflow.log_params(dict(fee=FEE_PER_CONTRACT, dte_max=DTE_MAX,
                               max_spread=MAX_SPREAD_FRAC,
                               lags=str(LAGS), taus=str(TAUS),
                               n_events=len(ev),
                               events_source="earnings_vrp_v1 (reused)"))
        for struct in STRUCTURES:
            for lag in LAGS:
                for cm in COST_MODELS:
                    for tau in TAUS:
                        sub = select(res, struct, lag, cm, tau)
                        key = cell_key(struct, lag, cm, tau)
                        m = cell_metrics(sub)
                        m["gates"] = (gates(m)
                                      if cm in GATED_MODELS and m.get("n", 0)
                                      else {})
                        summary["cells"][key] = m
                        if cm == "cross" and m.get("n", 0):
                            with mlflow.start_run(run_name=key, nested=True):
                                mlflow.log_params(dict(
                                    structure=struct, lag=lag,
                                    cost_model=cm,
                                    tau="none" if tau is None else tau))
                                for mk in ["n", "mean_ret", "total_pnl", "wr",
                                           "pf", "book_sharpe", "book_sortino",
                                           "years_pos", "regime_gap",
                                           "max_event_share", "day_conc",
                                           "mean_d_iv", "mean_theta_bleed"]:
                                    v = m.get(mk)
                                    if v is None:
                                        continue
                                    try:
                                        v = float(v)
                                        if np.isfinite(v):
                                            mlflow.log_metric(mk, v)
                                    except Exception:
                                        pass

        # ---- gross-edge vs cost decomposition (closure insight machinery)
        decomp = {}
        for struct in STRUCTURES:
            for lag in LAGS:
                sp = select(res, struct, lag, "midpure", None)
                sc = select(res, struct, lag, "cross", None)
                if len(sp) == 0:
                    continue
                decomp[f"{struct}|lag{lag}"] = dict(
                    n=int(len(sp)),
                    gross_mid_ret=float(sp.ret_notional.mean()),
                    net_cross_ret=float(sc.ret_notional.mean()),
                    cost_ret=float(sp.ret_notional.mean()
                                   - sc.ret_notional.mean()),
                    gross_mid_pnl=float(sp.pnl_dollars.mean()),
                    net_cross_pnl=float(sc.pnl_dollars.mean()),
                    mean_d_iv=float(sp.d_iv.mean()),
                    mean_theta_bleed=float(sp.theta_bleed.mean()),
                    mean_hold_days=float(sp.hold_days.mean()),
                )
        summary["decomposition"] = decomp

        # ---- verdict per pre-registered gates (PRIMARY = cross)
        def g1_pass(struct, lag, tau):
            m = summary["cells"][cell_key(struct, lag, "cross", tau)]
            return bool(m.get("gates", {}).get("G1"))

        passing, plateau_notes = [], []
        verdict = "CLOSED_NEGATIVE"
        tau_vals = [t for t in TAUS if t is not None]
        for struct in STRUCTURES:
            for lag in LAGS:
                for tau in TAUS:
                    m = summary["cells"][cell_key(struct, lag, "cross", tau)]
                    gt = m.get("gates", {})
                    if not (gt and all(gt.get(k)
                                       for k in ["G1", "G2", "G3", "G4"])):
                        continue
                    # G5(a): neighbor lag must pass G1
                    li = LAGS.index(lag)
                    neigh_lags = [LAGS[j] for j in (li - 1, li + 1)
                                  if 0 <= j < len(LAGS)]
                    lag_ok = any(g1_pass(struct, nl, tau)
                                 for nl in neigh_lags)
                    # G5(b): tau neighbors must pass G1
                    tau_ok = True
                    if tau is not None:
                        ti = tau_vals.index(tau)
                        for j in (ti - 1, ti + 1):
                            if 0 <= j < len(tau_vals):
                                if not g1_pass(struct, lag, tau_vals[j]):
                                    tau_ok = False
                    if lag_ok and tau_ok:
                        passing.append(cell_key(struct, lag, "cross", tau))
                        verdict = "PASS_CANDIDATE"
                    else:
                        plateau_notes.append(
                            f"{cell_key(struct, lag, 'cross', tau)}: "
                            f"THRESHOLD_LUCK (lag_ok={lag_ok} tau_ok={tau_ok})")
        summary["passing_cells_cross"] = passing
        summary["plateau_notes"] = plateau_notes
        summary["verdict"] = verdict
        mlflow.log_param("verdict", verdict)
        json.dump(summary, open(f"{OUT_DIR}/summary.json", "w"), indent=1,
                  default=str)
        mlflow.log_artifact(f"{OUT_DIR}/summary.json")
    print("VERDICT:", verdict)
    print("passing (cross):", passing)
    print("plateau notes:", plateau_notes)
    print("decomposition:", json.dumps(decomp, indent=1))


if __name__ == "__main__":
    main()
