#!/usr/bin/env python3
"""
tsmom_etf_v1 — Cross-asset TIME-SERIES MOMENTUM on a diversified ETF basket.
================================================================================
PRE-REGISTERED DESIGN (written 2026-06-12 BEFORE any results were computed).
Lane opened under HC #603 R1 (quant experimentation, free data, Jupiter CPU).
RUN_HISTORY grep for TSMOM / time-series momentum / trend-following ETF /
managed futures = zero prior hits. NOT a re-run of the SHELVED equity-only
ETF-rotation lane (cross-sectional, long-only, equity sectors — regime gap
1.57-1.61): this lane is TIME-SERIES sign momentum, LONG AND SHORT, on a
multi-asset-class basket where ~3/4 of risk slots are non-equity.

WHY THIS LANE
-------------
Every closure in this program since June (earnings vol x2, PEAD, index VRP,
equity rotation) died one of two deaths: (1) gross edge below the cost floor,
or (2) equity-beta-in-disguise (regime gap >> 0.50). Cross-asset TSMOM
(Moskowitz-Ooi-Pedersen 2012) is the canonical published strategy whose shape
attacks BOTH failure modes:
  - Gross edge historically large vs costs: monthly rebalance on liquid ETFs
    (spreads 1-5 bps) => turnover ~2-4x/yr per leg vs 10 bps/side assumed.
  - NOT structurally long equity: it shorts falling assets (crisis alpha
    2008/2022 in the literature) and the basket is 4 equity / 4 bond /
    4 commodity / 3 FX ETFs, vol-weighted.
HONEST RISK, stated upfront: post-2009 replications on ETFs show degraded
Sharpe (~0.3-0.6). The Sharpe>=1.0 house gate is a HIGH bar for this family;
a rigorous closed-negative is an acceptable outcome and will close the
diversified-trend direction on free daily data.

DATA (all free, zero spend)
---------------------------
yfinance daily adj_close (auto handles splits/distributions — total-return
proxy), cached to results/tsmom_etf_v1/data/. FIXED pre-registered basket,
all liquid US ETFs with inception <= mid-2007:
  Equity:      SPY, EFA, EEM, IWM
  Bonds:       TLT, IEF, LQD, HYG
  Commodities: GLD, SLV, DBC, USO
  FX:          UUP, FXE, FXY
An asset enters the book once it has >= 280 trading days of history (12m
lookback + vol estimate). Effective portfolio start ~2008 (includes GFC,
2011, taper, 2015, COVID, 2022 bear, 2024 unwind). Cash earns 0
(conservative: ignores T-bill yield on unencumbered cash and short proceeds).

SIGNAL + SIZING (canonical MOP, no fitting)
-------------------------------------------
At each month-end close t (last trading day of month):
  sign_i = sign(P_t / P_{t-L} - 1),  L in {252 (PRIMARY), 126, 63} td.
  sigma_i = EWMA(span 60) daily-return vol, annualized, FLOORED at 5%
            (caps per-asset leverage at 2x).
  w_i = sign_i * (0.10 / sigma_i) / N_active   (per-asset 10% vol target,
        equal risk slots; typical gross leverage ~1-2x, no other scaling).
Weights held constant until next rebalance (drift ignored, standard).
Row-t portfolio return = sum_i w_i(t) * r_i(t->t+1) — position decided at
close t earns the NEXT day's return; mandatory lag1 gate on top (G5).

CELLS
-----
  M12  TSMOM L=252 (PRIMARY)        M06  L=126 (plateau neighbor)
  M03  L=63  (plateau neighbor)     ENS  equal-weight average of the three
                                         sign signals (standard ensemble)
Each at 10 bps/side PRIMARY cost, 25 bps/side STRESS, lag1 robustness.
Costs charged on |delta w| at every weight change.
DIAGNOSTICS (never drive a PASS): per-asset-class sleeve Sharpes for M12;
OLS of M12 daily net returns on SPY (alpha t-stat NW lag 21, beta, R^2);
rebalance-day shift +10 td variant of M12.

GATES (PRE-REGISTERED — pass ALL or verdict = CLOSED NEGATIVE)
--------------------------------------------------------------
- G1: M12 or ENS net Sharpe >= 1.0 at 10 bps, n_days >= 2,500; PLATEAU
      required: BOTH lookback neighbors (M06, M03) net Sharpe >= 0.5.
- G2: >= 70% of calendar years positive (first full year -> 2026).
- G3: HC #428 R1 regime gate: stratify daily net returns by SPY
      close-to-close on the P&L-REALIZATION day (green > +0.2%, red <
      -0.2%); gap = |Sh_g - Sh_r| / max(|Sh_g|,|Sh_r|) <= 0.50.
- G4: day-concentration: best day net P&L <= 0.70 of total (HC #344);
      report MaxDD, worst day, Calmar.
- G5: the G1 cell keeps net Sharpe >= 0.5 BOTH at 25 bps AND with
      1-trading-day signal lag (at 10 bps).
Report per cell: Sharpe, Sortino, PF, WR, CAGR, MaxDD, Calmar, regime
split, per-year Sharpe, day-conc, avg gross leverage, ann turnover.

OUTPUT: results/tsmom_etf_v1/{summary.json, cells.csv, daily_*.parquet,
data/*.parquet, run.log}; MLflow exp tsmom_etf_v1 (http://localhost:5000).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/home/jupiter/Lvl3Quant"
OUT = f"{ROOT}/results/tsmom_etf_v1"
DATA = f"{OUT}/data"
MLFLOW_URI = "http://localhost:5000"
EXP = "tsmom_etf_v1"
START, END = "2004-01-01", "2026-06-11"
ANN = 252.0
COST_PRIMARY = 0.0010   # 10 bps per side
COST_STRESS = 0.0025    # 25 bps per side
LOOKBACKS = {"M12": 252, "M06": 126, "M03": 63}
PRIMARY = "M12"
VOL_TARGET = 0.10
VOL_FLOOR = 0.05
MIN_HISTORY = 280
BASKET = {
    "SPY": "equity", "EFA": "equity", "EEM": "equity", "IWM": "equity",
    "TLT": "bond", "IEF": "bond", "LQD": "bond", "HYG": "bond",
    "GLD": "commodity", "SLV": "commodity", "DBC": "commodity", "USO": "commodity",
    "UUP": "fx", "FXE": "fx", "FXY": "fx",
}

os.makedirs(DATA, exist_ok=True)


def log(msg):
    print(msg, flush=True)
    with open(f"{OUT}/run.log", "a") as f:
        f.write(msg + "\n")


# ----------------------------------------------------------------- data
def fetch(ticker):
    import yfinance as yf
    path = f"{DATA}/{ticker}_daily.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        log(f"[data] cached {ticker}: {len(df)} rows {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
        return df
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
            if df is None or len(df) == 0:
                raise RuntimeError("empty")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index()
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            df.to_parquet(path)
            log(f"[data] fetched {ticker}: {len(df)} rows {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
            return df
        except Exception as e:
            log(f"[data] {ticker} attempt {attempt+1} failed: {e}")
            time.sleep(3)
    raise RuntimeError(f"cannot fetch {ticker}")


def build_panel():
    panel = None
    for tk in BASKET:
        df = fetch(tk)[["date", "adj_close"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df.rename(columns={"adj_close": tk})
        panel = df if panel is None else panel.merge(df, on="date", how="outer")
    panel = panel.sort_values("date").reset_index(drop=True)
    return panel


# ----------------------------------------------------------------- metrics
def sharpe(r):
    r = np.asarray(r, float)
    if len(r) < 20 or np.std(r) == 0:
        return np.nan
    return float(np.mean(r) / np.std(r) * np.sqrt(ANN))


def metrics(ret, dates, spy_ret):
    ret = np.asarray(ret, float)
    n = len(ret)
    eq = np.cumprod(1 + ret)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    maxdd = float(dd.min())
    yrs = n / ANN
    cagr = float(eq[-1] ** (1 / yrs) - 1) if yrs > 0.5 and eq[-1] > 0 else np.nan
    neg = ret[ret < 0]
    sortino = float(np.mean(ret) / np.std(neg) * np.sqrt(ANN)) if len(neg) > 10 and np.std(neg) > 0 else np.nan
    act = ret[ret != 0]
    wins, losses = act[act > 0], act[act < 0]
    pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else np.nan
    wr = float(len(wins) / len(act)) if len(act) else np.nan
    g = spy_ret > 0.002
    r_ = spy_ret < -0.002
    fl = ~g & ~r_
    sh_g, sh_r, sh_f = sharpe(ret[g]), sharpe(ret[r_]), sharpe(ret[fl])
    denom = max(abs(sh_g), abs(sh_r)) if np.isfinite(sh_g) and np.isfinite(sh_r) else np.nan
    gap = float(abs(sh_g - sh_r) / denom) if denom and denom > 0 else np.nan
    yr = pd.Series(ret, index=pd.DatetimeIndex(dates)).groupby(lambda d: d.year)
    per_year = {int(y): {"sharpe": sharpe(v.values), "ret": float((1 + v).prod() - 1)} for y, v in yr}
    full_years = sorted(y for y in per_year if y < 2026)[1:]  # drop partial first year
    years = full_years + ([2026] if 2026 in per_year else [])
    yrs_pos = sum(per_year[y]["ret"] > 0 for y in years)
    tot = ret.sum()
    day_conc = float(ret.max() / tot) if tot > 0 else np.nan
    return dict(
        n_days=int(n), sharpe=sharpe(ret), sortino=sortino, pf=pf, wr=wr,
        cagr=cagr, maxdd=maxdd, calmar=float(cagr / -maxdd) if maxdd < 0 and np.isfinite(cagr) else np.nan,
        worst_day=float(ret.min()), best_day=float(ret.max()), day_conc=day_conc,
        sharpe_green=sh_g, sharpe_red=sh_r, sharpe_flat=sh_f, regime_gap=gap,
        years_pos=int(yrs_pos), years_total=len(years),
        per_year={k: per_year[k] for k in sorted(per_year)},
    )


# ----------------------------------------------------------------- engine
def build_weights(px, rets, lookback, rebal_idx, shift_td=0):
    """Target weights decided at each rebalance close; held until next rebalance.
    shift_td>0 moves every rebalance day forward by shift_td trading days (diagnostic)."""
    n = len(px)
    tickers = list(BASKET)
    sigma = rets.ewm(span=60, min_periods=40).std() * np.sqrt(ANN)
    hist_ok = px.notna().rolling(MIN_HISTORY, min_periods=MIN_HISTORY).count() >= MIN_HISTORY
    mom = px / px.shift(lookback) - 1.0
    W = pd.DataFrame(0.0, index=px.index, columns=tickers)
    idx = [min(i + shift_td, n - 1) for i in rebal_idx]
    cur = pd.Series(0.0, index=tickers)
    prev_i = None
    for i in idx:
        active = [t for t in tickers
                  if hist_ok[t].iloc[i] and np.isfinite(mom[t].iloc[i]) and np.isfinite(sigma[t].iloc[i])]
        new = pd.Series(0.0, index=tickers)
        if active:
            na = len(active)
            for t in active:
                sg = np.sign(mom[t].iloc[i])
                vol = max(float(sigma[t].iloc[i]), VOL_FLOOR)
                new[t] = sg * (VOL_TARGET / vol) / na
        if prev_i is not None:
            W.iloc[prev_i:i] = cur.values
        cur, prev_i = new, i
    if prev_i is not None:
        W.iloc[prev_i:] = cur.values
    return W


def build_weights_ens(px, rets, rebal_idx):
    Ws = [build_weights(px, rets, L, rebal_idx) for L in LOOKBACKS.values()]
    return sum(Ws) / len(Ws)


def run_cell(W, rets, cost, lag=0):
    Wl = W.shift(lag).fillna(0.0)
    gross = (Wl * rets.shift(-1)).sum(axis=1, skipna=True)
    turn = Wl.diff().abs().sum(axis=1)
    turn.iloc[0] = Wl.iloc[0].abs().sum()
    net = (gross - turn * cost).iloc[:-1]
    live = (Wl.abs().sum(axis=1) > 0).iloc[:-1]
    first = np.argmax(live.values) if live.any() else len(net)
    stats = dict(
        avg_gross_lev=float(Wl.abs().sum(axis=1).iloc[first:].mean()),
        ann_turnover=float(turn.iloc[first:].mean() * ANN),
    )
    return net.iloc[first:].fillna(0.0), stats


def main():
    open(f"{OUT}/run.log", "w").close()
    log("=== tsmom_etf_v1 START ===")
    df = build_panel()
    px = df[list(BASKET)]
    rets = px.pct_change()
    log(f"[panel] {len(df)} rows {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    for t in BASKET:
        s = px[t].dropna()
        log(f"[panel] {t} ({BASKET[t]}): {df['date'].iloc[s.index[0]].date()} -> {df['date'].iloc[s.index[-1]].date()}")

    # month-end rebalance indices
    d = pd.DatetimeIndex(df["date"])
    rebal_idx = [i for i in range(len(d) - 1) if (d[i].month != d[i + 1].month)]
    log(f"[rebal] {len(rebal_idx)} month-end rebalances")

    spy_ret_all = px["SPY"].pct_change().shift(-1).fillna(0.0)  # P&L-realization-day SPY ret
    cells, extras = {}, {}

    def add(name, W):
        for ck, c in [("c10", COST_PRIMARY), ("c25", COST_STRESS)]:
            for lk, lag in [("lag0", 0), ("lag1", 1)]:
                if (ck, lk) == ("c25", "lag1"):
                    continue
                net, stats = run_cell(W, rets, c, lag)
                m = metrics(net.values, df["date"].iloc[net.index].values, spy_ret_all.iloc[net.index].values)
                m.update(stats)
                cells[f"{name}_{ck}_{lk}"] = m
                if (ck, lk) == ("c10", "lag0"):
                    pd.DataFrame({"date": df["date"].iloc[net.index].values, "net": net.values}).to_parquet(
                        f"{OUT}/daily_{name}.parquet")
                log(f"[cell] {name} {ck} {lk}: Sharpe {m['sharpe']:.2f} Sortino {m['sortino']:.2f} "
                    f"PF {m['pf']:.2f} WR {m['wr']:.2f} CAGR {m['cagr']:.1%} MaxDD {m['maxdd']:.1%} "
                    f"gap {m['regime_gap']:.2f} yrs+ {m['years_pos']}/{m['years_total']} "
                    f"lev {m['avg_gross_lev']:.2f} turn {m['ann_turnover']:.1f}x")

    weights = {}
    for name, L in LOOKBACKS.items():
        weights[name] = build_weights(px, rets, L, rebal_idx)
        add(name, weights[name])
    weights["ENS"] = build_weights_ens(px, rets, rebal_idx)
    add("ENS", weights["ENS"])

    # ---------- diagnostics (never drive a PASS)
    Wp = weights[PRIMARY]
    sleeve = {}
    for cls in ["equity", "bond", "commodity", "fx"]:
        cols = [t for t in BASKET if BASKET[t] == cls]
        Wc = Wp.copy()
        Wc[[t for t in BASKET if BASKET[t] != cls]] = 0.0
        net, _ = run_cell(Wc, rets, COST_PRIMARY, 0)
        sleeve[cls] = sharpe(net.values)
        log(f"[diag] sleeve {cls}: Sharpe {sleeve[cls]:.2f}")
    # SPY OLS on primary net returns (NW lag 21)
    netp = pd.read_parquet(f"{OUT}/daily_{PRIMARY}.parquet")
    spy_d = spy_ret_all.iloc[: len(spy_ret_all)]
    y = netp["net"].values
    x = spy_ret_all.reindex(netp.index if netp.index.equals(pd.RangeIndex(len(netp))) else None)
    # align by date
    spy_df = pd.DataFrame({"date": df["date"], "spy": spy_ret_all})
    mrg = netp.merge(spy_df, on="date", how="left").dropna()
    X = np.column_stack([np.ones(len(mrg)), mrg["spy"].values])
    beta_hat = np.linalg.lstsq(X, mrg["net"].values, rcond=None)[0]
    resid = mrg["net"].values - X @ beta_hat
    nn = len(resid)
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, 22):
        w = 1 - L / 22.0
        G = u[:-L].T @ u[L:]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    alpha_ann = float(beta_hat[0] * ANN)
    alpha_t = float(beta_hat[0] / np.sqrt(cov[0, 0]))
    diag = dict(sleeve_sharpes=sleeve, spy_beta=float(beta_hat[1]),
                alpha_ann=alpha_ann, alpha_nw_t=alpha_t,
                r2=float(1 - resid.var() / mrg["net"].values.var()) if mrg["net"].values.var() > 0 else np.nan)
    log(f"[diag] SPY OLS: beta {diag['spy_beta']:.2f} alpha {alpha_ann:.1%}/yr (NW-t {alpha_t:.2f}) R2 {diag['r2']:.2f}")
    # rebalance-day shift +10td
    Wsh = build_weights(px, rets, LOOKBACKS[PRIMARY], rebal_idx, shift_td=10)
    net_sh, _ = run_cell(Wsh, rets, COST_PRIMARY, 0)
    diag["rebal_shift10_sharpe"] = sharpe(net_sh.values)
    log(f"[diag] rebal shift +10td: Sharpe {diag['rebal_shift10_sharpe']:.2f}")

    # ---------- gates
    verdict_rows, passing = {}, []
    for name in [PRIMARY, "ENS"]:
        m = cells[f"{name}_c10_lag0"]
        nb = [cells["M06_c10_lag0"]["sharpe"], cells["M03_c10_lag0"]["sharpe"]]
        g1 = m["sharpe"] >= 1.0 and m["n_days"] >= 2500 and all(s >= 0.5 for s in nb)
        g2 = m["years_pos"] >= int(np.ceil(0.7 * m["years_total"]))
        g3 = np.isfinite(m["regime_gap"]) and m["regime_gap"] <= 0.50
        g4 = (not np.isfinite(m["day_conc"])) or m["day_conc"] <= 0.70
        m25 = cells[f"{name}_c25_lag0"]
        mlag = cells[f"{name}_c10_lag1"]
        g5 = m25["sharpe"] >= 0.5 and mlag["sharpe"] >= 0.5
        ok = g1 and g2 and g3 and g4 and g5
        verdict_rows[name] = dict(G1=bool(g1), G2=bool(g2), G3=bool(g3), G4=bool(g4), G5=bool(g5), PASS=bool(ok))
        if ok:
            passing.append(name)
        log(f"[gate] {name}: G1={g1} G2={g2} G3={g3} G4={g4} G5={g5} -> {'PASS' if ok else 'FAIL'}")

    verdict = "PASS" if passing else "CLOSED_NEGATIVE"
    summary = dict(
        verdict=verdict, passing_cells=passing, gates=verdict_rows, diagnostics=diag,
        cells={k: {kk: (vv if not isinstance(vv, float) or np.isfinite(vv) else None)
                   for kk, vv in v.items()} for k, v in cells.items()},
        design="see strategy/tsmom_etf_v1.py docstring (pre-registered)",
        caveats=[
            "adj_close total-return proxy (dividends reinvested at close)",
            "cash/short proceeds earn 0 — conservative",
            "weights held constant between month-ends (drift ignored, standard)",
            "borrow cost on ETF shorts not modeled (liquid ETFs, GC ~25-50bps/yr on shorted fraction)",
        ],
    )
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    pd.DataFrame(cells).T.to_csv(f"{OUT}/cells.csv")
    log(f"=== VERDICT: {verdict} (passing: {passing or 'none'}) ===")

    # ---------- MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXP)
        for key, m in cells.items():
            with mlflow.start_run(run_name=key):
                mlflow.log_params(dict(cell=key, cost_primary=COST_PRIMARY, cost_stress=COST_STRESS,
                                       vol_target=VOL_TARGET, basket_size=len(BASKET)))
                for mk in ["sharpe", "sortino", "pf", "wr", "cagr", "maxdd", "calmar",
                           "regime_gap", "sharpe_green", "sharpe_red", "day_conc",
                           "years_pos", "n_days", "avg_gross_lev", "ann_turnover"]:
                    v = m.get(mk)
                    if v is not None and np.isfinite(v):
                        mlflow.log_metric(mk, float(v))
        with mlflow.start_run(run_name="SUMMARY"):
            mlflow.log_param("verdict", verdict)
            mlflow.log_param("passing_cells", ",".join(passing) or "none")
            mlflow.log_metric("spy_beta", diag["spy_beta"])
            mlflow.log_metric("alpha_nw_t", diag["alpha_nw_t"])
            mlflow.log_artifact(f"{OUT}/summary.json")
            mlflow.log_artifact(f"{OUT}/cells.csv")
        log("[mlflow] logged")
    except Exception as e:
        log(f"[mlflow] FAILED: {e}")
    log("=== tsmom_etf_v1 DONE ===")


if __name__ == "__main__":
    sys.exit(main())
