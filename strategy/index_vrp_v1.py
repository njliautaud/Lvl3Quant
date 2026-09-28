#!/usr/bin/env python3
"""
index_vrp_v1 — Index-level Variance Risk Premium harvest via VIX-futures ETPs.
================================================================================
PRE-REGISTERED DESIGN (written 2026-06-12 BEFORE any results were computed).
Lane opened under HC #603 R1 (quant experimentation, free data, CPU only).
Follow-up to earnings_vrp_v1 closure note: "index-level VRP — needs a free
SPX/SPY option chain source first". RUN_HISTORY grep for index VRP / SVXY /
VIX ETP / contango = zero prior hits.

WHY THIS LANE
-------------
earnings_vrp_v1 proved single-name VRP is real gross but equals the option
spread. The INDEX variance premium is the largest, most-documented vol premium
and trades through instruments with 30-100x lower cost floors than single-name
option spreads. No free historical SPX/SPY option chain exists (verified:
DOLT clone has 68 single names only, docs/vix_data_sourcing_20260512.md
surveyed paid sources). The honest free implementation is therefore:
  (a) MEASURE the premium with the canonical proxy VRP = VIX^2 - realized var
      (non-traded diagnostic), and
  (b) TRADE it through VIX-futures ETPs (SVXY short-vol / VIXY long-vol),
      whose daily prices embed REAL roll yield, REAL Feb-2018 Volmageddon
      losses, REAL expense ratios — no simulated option fills anywhere.

HONESTY CAVEATS (stated upfront, part of any verdict)
-----------------------------------------------------
1. ETP P&L harvests the VIX FUTURES roll/term premium — a close cousin of,
   but not identical to, the SPX option variance premium. Labeled as such.
2. SVXY changed leverage -1x -> -0.5x on 2018-02-28 (post-Volmageddon).
   Both eras kept; per-era Sharpe reported. Sharpe is leverage-invariant so
   pooling is legitimate for risk-adjusted gates; CAGR is not pooled-honest.
3. History starts 2011-10 (SVXY inception). Includes Volmageddon (Feb-2018),
   COVID (Mar-2020), 2022 bear, Aug-2024 unwind — all real tail events.
4. Signals use index CLOSE values to trade the SAME close (standard but
   mildly optimistic: VIX settles 16:15 vs equity 16:00). Mandatory 1-day-lag
   robustness gate (G5) controls this.
5. yfinance adj_close used for ETPs (handles SVXY/VIXY reverse splits).

DATA (all free, zero spend)
---------------------------
- On disk: output/macro_swing_v1/{vix,vix3m,spy}_daily.parquet (2010->2026-06).
- yfinance refresh + new daily series: ^VIX, ^VIX3M, ^VVIX, SVXY, VIXY, SPY
  (2010-01-01 -> 2026-06-11). Cached to results/index_vrp_v1/data/.

DIAGNOSTIC (non-traded, establishes gross premium)
--------------------------------------------------
VRP_t = (VIX_t/100)^2 - RV2_{t->t+21}  where RV2 = annualized realized
variance of SPY close-to-close over the NEXT 21 trading days. Report mean,
t-stat (HAC lag 21), % positive, by year. This cell can never drive a PASS
(it is not tradeable as-is without an option chain); it only frames magnitude.

TRADED CELLS (daily close-to-close, $100k notional, fully invested or cash)
---------------------------------------------------------------------------
Position decided at close t from data <= close t; earns the ETP's
adj-close-to-adj-close return t -> t+1. Cash earns 0 (conservative).
  S0  hold_short_vol : always long SVXY (pure-harvest baseline).
  S1  contango_K     : long SVXY iff VIX3M/VIX - 1 > K, else cash.
                       K grid pre-registered {0.00, 0.02, 0.05} — a PLATEAU
                       check, not an optimization: pass requires neighbors hold.
  S2  contango+rich  : S1(K=0.02) AND (VIX/100)^2 - trailing-21d RV2 > 0
                       (past-only realized variance — no lookahead).
  S3  symmetric      : long SVXY iff contango (K=0.02), long VIXY iff
                       backwardation (ratio < -0.02), else cash. Exists to
                       test whether the signal is regime-symmetric or pure
                       short-vol beta.
COSTS: 5 bps per side PRIMARY (ETP spread 1-3bps + slippage, $0 commission),
15 bps per side STRESS. Charged on every position change (incl. entry/exit
and S3 flips = 2 sides). No fitting anywhere; whole 2011-2026 stream is OOT.

GATES (PRE-REGISTERED — pass ALL or verdict = CLOSED NEGATIVE)
--------------------------------------------------------------
- G1: >= 1 traded cell among {S0, S1(K=0.02), S2, S3} with net Sharpe >= 1.0
      at 5 bps over the FULL sample (n_days >= 2,500). If the cell is S1/S2,
      plateau required: both K-neighbors net Sharpe >= 0.5.
- G2: >= 70% of calendar years positive (2012-2026 -> >= 11/15).
- G3: HC #428 R1 regime gate: stratify daily net returns by SPY same-day
      close-to-close sign (green/red); gap = |Sh_g - Sh_r| / max(|Sh_g|,
      |Sh_r|) <= 0.50. Also report flat-day Sharpe (|SPY ret| < 0.2%).
- G4: day-concentration: best single day's net P&L <= 0.70 of total net
      P&L (house HC #344 convention); report MaxDD, worst day, Calmar.
- G5: robustness: the G1 cell must keep net Sharpe >= 0.5 BOTH at 15 bps
      AND with 1-trading-day signal lag (at 5 bps).
Report per cell: Sharpe, Sortino, PF, WR, CAGR, MaxDD, Calmar, regime split,
per-year Sharpe, per-era (pre/post 2018-02-28) Sharpe.

OUTPUT: results/index_vrp_v1/{summary.json, cells.csv, daily_returns.parquet,
data/*.parquet}; MLflow exp index_vrp_v1 (http://localhost:5000).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/home/jupiter/Lvl3Quant"
OUT = f"{ROOT}/results/index_vrp_v1"
DATA = f"{OUT}/data"
MLFLOW_URI = "http://localhost:5000"
EXP = "index_vrp_v1"
START, END = "2010-01-01", "2026-06-11"
ANN = 252.0
COST_PRIMARY = 0.0005   # 5 bps per side
COST_STRESS = 0.0015    # 15 bps per side
K_GRID = [0.00, 0.02, 0.05]
K_MAIN = 0.02

os.makedirs(DATA, exist_ok=True)


def log(msg):
    print(msg, flush=True)
    with open(f"{OUT}/run.log", "a") as f:
        f.write(msg + "\n")


# ----------------------------------------------------------------- data
def fetch(ticker, name):
    import yfinance as yf
    path = f"{DATA}/{name}_daily.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        log(f"[data] cached {name}: {len(df)} rows {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
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
            log(f"[data] fetched {ticker} -> {name}: {len(df)} rows {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
            return df
        except Exception as e:
            log(f"[data] {ticker} attempt {attempt+1} failed: {e}")
            time.sleep(3)
    raise RuntimeError(f"cannot fetch {ticker}")


def build_panel():
    series = {
        "vix": ("^VIX", "close"),
        "vix3m": ("^VIX3M", "close"),
        "vvix": ("^VVIX", "close"),
        "spy": ("SPY", "adj_close"),
        "svxy": ("SVXY", "adj_close"),
        "vixy": ("VIXY", "adj_close"),
    }
    panel = None
    for name, (tk, col) in series.items():
        df = fetch(tk, name)[["date", "close", "adj_close"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df.rename(columns={col: name})[["date", name]]
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
    # regime stratification (HC #428 R1): SPY same-day c2c
    g = spy_ret > 0.002
    r_ = spy_ret < -0.002
    fl = ~g & ~r_
    sh_g, sh_r, sh_f = sharpe(ret[g]), sharpe(ret[r_]), sharpe(ret[fl])
    denom = max(abs(sh_g), abs(sh_r)) if np.isfinite(sh_g) and np.isfinite(sh_r) else np.nan
    gap = float(abs(sh_g - sh_r) / denom) if denom and denom > 0 else np.nan
    # per-year
    yr = pd.Series(ret, index=pd.DatetimeIndex(dates)).groupby(lambda d: d.year)
    per_year = {int(y): {"sharpe": sharpe(v.values), "ret": float((1 + v).prod() - 1)} for y, v in yr}
    years = [y for y in per_year if 2012 <= y <= 2026]
    yrs_pos = sum(per_year[y]["ret"] > 0 for y in years)
    # day concentration
    tot = ret.sum()
    day_conc = float(ret.max() / tot) if tot > 0 else np.nan
    # era split (SVXY deleverage 2018-02-28)
    era = pd.DatetimeIndex(dates) < "2018-02-28"
    return dict(
        n_days=int(n), sharpe=sharpe(ret), sortino=sortino, pf=pf, wr=wr,
        cagr=cagr, maxdd=maxdd, calmar=float(cagr / -maxdd) if maxdd < 0 and np.isfinite(cagr) else np.nan,
        worst_day=float(ret.min()), best_day=float(ret.max()), day_conc=day_conc,
        sharpe_green=sh_g, sharpe_red=sh_r, sharpe_flat=sh_f, regime_gap=gap,
        years_pos=int(yrs_pos), years_total=len(years),
        sharpe_pre2018=sharpe(ret[era]), sharpe_post2018=sharpe(ret[~era]),
        per_year={k: per_year[k] for k in sorted(per_year)},
    )


# ----------------------------------------------------------------- engine
def run_cell(pos_short, pos_long, df, cost, lag=0):
    """pos_short/pos_long: target weight series in SVXY / VIXY decided at close t.
    Earns weight_t * etp_ret_{t+1}; costs charged on weight changes."""
    ps = pos_short.shift(lag).fillna(0.0)
    pl = pos_long.shift(lag).fillna(0.0) if pos_long is not None else pd.Series(0.0, index=ps.index)
    r_s = df["svxy"].pct_change().shift(-1)
    r_l = df["vixy"].pct_change().shift(-1)
    gross = ps * r_s + pl * r_l
    turn = ps.diff().abs().fillna(ps.abs()) + pl.diff().abs().fillna(pl.abs())
    net = gross - turn * cost
    net = net.iloc[:-1]  # last day has no next-day return
    valid = df["svxy"].notna() & df["vix3m"].notna() & df["vix"].notna()
    valid = valid.iloc[:-1]
    return net[valid].fillna(0.0)


def main():
    open(f"{OUT}/run.log", "w").close()
    log("=== index_vrp_v1 START ===")
    df = build_panel()
    df = df[df["date"] >= "2011-10-04"].reset_index(drop=True)  # SVXY inception
    df["spy_ret"] = df["spy"].pct_change()
    df["ratio"] = df["vix3m"] / df["vix"] - 1.0
    # trailing realized variance (past-only, annualized) and trailing VRP
    lr = np.log(df["spy"] / df["spy"].shift(1))
    df["rv2_trail"] = (lr ** 2).rolling(21).mean() * ANN
    df["vrp_trail"] = (df["vix"] / 100.0) ** 2 - df["rv2_trail"]
    log(f"[panel] {len(df)} rows {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")

    # ---------- DIAGNOSTIC: forward VRP (non-traded)
    rv2_fwd = (lr ** 2).rolling(21).mean().shift(-21) * ANN
    vrp_fwd = ((df["vix"] / 100.0) ** 2 - rv2_fwd).dropna()
    vmean = float(vrp_fwd.mean())
    # HAC t-stat, lag 21
    x = vrp_fwd.values - vmean
    nn = len(x)
    var = x.var()
    for L in range(1, 22):
        w = 1 - L / 22.0
        var += 2 * w * np.mean(x[:-L] * x[L:])
    tstat = float(vmean / np.sqrt(var / nn))
    diag = dict(mean_vrp=vmean, hac_tstat=tstat, pct_positive=float((vrp_fwd > 0).mean()), n=nn)
    log(f"[diag] forward VRP mean={vmean:.5f} (var units), HAC-t={tstat:.2f}, %pos={diag['pct_positive']:.1%}")

    # Regime stratification must use the SPY return of the day the P&L is
    # REALIZED. Row t's net return covers day t -> t+1 (r.shift(-1)), so the
    # matching SPY return is spy_ret.shift(-1). (Bugfix before first commit;
    # gates unchanged.)
    spy_ret = df["spy_ret"].shift(-1).fillna(0.0)
    one = pd.Series(1.0, index=df.index)
    cells = {}

    def add(name, ps, pl=None):
        for ck, c in [("c5", COST_PRIMARY), ("c15", COST_STRESS)]:
            for lk, lag in [("lag0", 0), ("lag1", 1)]:
                if (ck, lk) == ("c15", "lag1"):
                    continue
                net = run_cell(ps, pl, df, c, lag)
                m = metrics(net.values, df["date"].iloc[net.index].values, spy_ret.iloc[net.index].values)
                cells[f"{name}_{ck}_{lk}"] = m
                if (ck, lk) == ("c5", "lag0"):
                    pd.DataFrame({"date": df["date"].iloc[net.index].values, "net": net.values}).to_parquet(
                        f"{OUT}/daily_{name}.parquet")
                log(f"[cell] {name} {ck} {lk}: Sharpe {m['sharpe']:.2f} Sortino {m['sortino']:.2f} "
                    f"PF {m['pf']:.2f} WR {m['wr'] if m['wr']==m['wr'] else float('nan'):.2f} CAGR {m['cagr']:.1%} "
                    f"MaxDD {m['maxdd']:.1%} gap {m['regime_gap']:.2f} yrs+ {m['years_pos']}/{m['years_total']}")

    add("S0_hold", one)
    for K in K_GRID:
        add(f"S1_contango_K{int(K*100):02d}", (df["ratio"] > K).astype(float))
    add("S2_contango_rich", ((df["ratio"] > K_MAIN) & (df["vrp_trail"] > 0)).astype(float))
    add("S3_symmetric", (df["ratio"] > K_MAIN).astype(float), (df["ratio"] < -K_MAIN).astype(float))

    # ---------- gates
    main_cells = ["S0_hold", f"S1_contango_K{int(K_MAIN*100):02d}", "S2_contango_rich", "S3_symmetric"]
    verdict_rows = {}
    passing = []
    for name in main_cells:
        m = cells[f"{name}_c5_lag0"]
        g1 = m["sharpe"] >= 1.0 and m["n_days"] >= 2500
        if g1 and name.startswith(("S1", "S2")):
            nb = [cells[f"S1_contango_K{int(K*100):02d}_c5_lag0"]["sharpe"] for K in K_GRID]
            g1 = g1 and all(s >= 0.5 for s in nb)
        g2 = m["years_pos"] >= int(np.ceil(0.7 * m["years_total"]))
        g3 = np.isfinite(m["regime_gap"]) and m["regime_gap"] <= 0.50
        g4 = (not np.isfinite(m["day_conc"])) or m["day_conc"] <= 0.70
        if m["sharpe"] is not None and not np.isfinite(m.get("day_conc", np.nan)):
            g4 = True  # negative total P&L -> day_conc undefined; cell fails G1 anyway
        m15 = cells[f"{name}_c15_lag0"]
        mlag = cells[f"{name}_c5_lag1"]
        g5 = m15["sharpe"] >= 0.5 and mlag["sharpe"] >= 0.5
        ok = g1 and g2 and g3 and g4 and g5
        verdict_rows[name] = dict(G1=bool(g1), G2=bool(g2), G3=bool(g3), G4=bool(g4), G5=bool(g5), PASS=bool(ok))
        if ok:
            passing.append(name)
        log(f"[gate] {name}: G1={g1} G2={g2} G3={g3} G4={g4} G5={g5} -> {'PASS' if ok else 'FAIL'}")

    verdict = "PASS" if passing else "CLOSED_NEGATIVE"
    summary = dict(
        verdict=verdict, passing_cells=passing, gates=verdict_rows,
        diagnostic_forward_vrp=diag,
        cells={k: {kk: (vv if not isinstance(vv, float) or np.isfinite(vv) else None)
                   for kk, vv in v.items()} for k, v in cells.items()},
        design="see strategy/index_vrp_v1.py docstring (pre-registered)",
        caveats=[
            "ETP P&L = VIX futures roll premium, cousin of (not identical to) SPX option VRP",
            "SVXY leverage -1x->-0.5x on 2018-02-28; Sharpe pooled (leverage-invariant), CAGR not",
            "signal-at-close trades same close (VIX settles 16:15); lag1 robustness gate controls",
            "history 2011-10 onward only (SVXY inception); includes Volmageddon/COVID/2022/Aug-2024",
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
                mlflow.log_params(dict(cell=key, cost_primary=COST_PRIMARY, cost_stress=COST_STRESS))
                for mk in ["sharpe", "sortino", "pf", "wr", "cagr", "maxdd", "calmar",
                           "regime_gap", "sharpe_green", "sharpe_red", "day_conc",
                           "years_pos", "n_days", "sharpe_pre2018", "sharpe_post2018"]:
                    v = m.get(mk)
                    if v is not None and np.isfinite(v):
                        mlflow.log_metric(mk, float(v))
        with mlflow.start_run(run_name="SUMMARY"):
            mlflow.log_param("verdict", verdict)
            mlflow.log_param("passing_cells", ",".join(passing) or "none")
            mlflow.log_metric("diag_vrp_hac_tstat", tstat)
            mlflow.log_metric("diag_vrp_pct_positive", diag["pct_positive"])
            mlflow.log_artifact(f"{OUT}/summary.json")
            mlflow.log_artifact(f"{OUT}/cells.csv")
        log("[mlflow] logged")
    except Exception as e:
        log(f"[mlflow] FAILED: {e}")
    log("=== index_vrp_v1 DONE ===")


if __name__ == "__main__":
    sys.exit(main())
