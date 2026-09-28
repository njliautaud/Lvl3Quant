#!/usr/bin/env python3
"""
crypto_funding_v1 — Crypto PERPETUAL FUNDING-RATE harvest (delta-neutral basis).
================================================================================
PRE-REGISTERED DESIGN (written 2026-06-12 BEFORE any results were computed).
Lane opened under HC #603 R1 (quant experimentation, free data, Jupiter CPU).
RUN_HISTORY grep for crypto / funding / perp / Binance = zero prior strategy
hits — NEW lane. Direct follow-up to tsmom_etf_v1 closure note: funding-rate
harvest is the one remaining strong free-data direction; stated risk upfront
that carry shapes tend to fail the HC #428 R1 regime gate.

WHY THIS LANE
-------------
The premium source is structurally DIFFERENT from everything closed so far:
perp funding is paid BY leveraged longs TO shorts (when positive) to pin the
perp to spot. It is a leverage-demand premium, not an equity risk premium.
The honest baseline — long spot / short perp, SAME notional — is delta-neutral
by construction: price P&L cancels (up to basis drift) and the book collects
funding. If this is still secretly long-risk-beta (carry crowding unwinds on
red days), the day-level regime gates will catch it; that is the test.

DATA (all free, zero spend — data.binance.vision public archive; REST is
geo-blocked 451 from this host, archive verified 200)
-----------------------------------------------------
- Funding: data/futures/um/monthly/fundingRate/{SYM}/ monthly CSVs,
  2020-01 -> 2026-05 (archive floor is 2020-01; perps launched late 2019).
  3 payments/day (00/08/16 UTC), rate applied to perp notional.
- Prices: daily klines (close), BOTH spot (data/spot/monthly/klines/{SYM}/1d)
  and USDT-M perp (data/futures/um/monthly/klines/{SYM}/1d), same range.
- SPY: results/tsmom_etf_v1/data/SPY_daily.parquet (yfinance cache, ->2026-06).
FIXED pre-registered basket, 10 most-liquid USDT perps with inception by
late 2020 (no survivorship cherry-pick — these were the liquid set then and
remain it): BTC ETH BNB XRP ADA LTC LINK DOGE DOT SOL (USDT pairs).
A name enters the book once spot+perp+funding all have >=10 days of history.
UTC calendar, 7 days/week, ANN=365.

CARRY UNIT + RETURNS (per 1.0 of carry notional)
------------------------------------------------
Position = long 1.0 spot + short 1.0 perp. Daily return for day d held from
close(d-1):  r_i(d) = spot_ret_i(d) - perp_ret_i(d) + F_i(d),
where F_i(d) = sum of funding rates with calc_time in UTC day d (short
RECEIVES positive funding, PAYS negative). Book return = sum_i w_i(d-1)*r_i(d).
Weights recomputed at each UTC close from PAST-ONLY data; position decided at
close t earns day t+1 (mandatory lag1 robustness on top, G5).

CELLS (all pre-registered; no other variants will be run)
---------------------------------------------------------
  C0_BTC  always-on BTC-only carry, w=1.
  C0_EW   PRIMARY: always-on equal weight 1/N_active across active names.
  C1_K0 / C1_K5 / C1_K10  conditional EW: include name i iff trailing 3-day
          mean daily funding, annualized (x365), > K in {0%, 5%, 10%};
          w_i = 1/N_active (de-levered when few qualify; never levered up).
  CS_2 / CS_3 / CS_4  cross-sectional: top-k names by trailing 3-day funding
          (require >0), w=1/k. CS_3 is the named CS primary.
No reverse-carry leg anywhere (long perp / short spot needs spot borrow —
not cleanly retail; harvest-only book).

COSTS (charged on sum_i |w_i(d) - w_i(d-1)| at each close)
----------------------------------------------------------
One unit of carry traded = 1.0 spot + 1.0 perp simultaneously.
PRIMARY 15 bps per side per carry unit = spot taker 7.5 (BNB-discount tier
10.0 undiscounted) + perp taker 4.5 + ~2-3 bps combined half-spreads on
BTC/ETH-class books. STRESS 30 bps per side. Funding accrues fee-free.
RETAIL-IMPLEMENTABLE means here: same-exchange long spot + short USDT-M perp,
fully collateralized (no liquidation risk at 1x), capital usage ~1.2x carry
notional (1.0 spot + ~0.2 perp margin) -> realized return on capital ~0.83x
of reported per-carry-unit numbers; Sharpe is scale-invariant so gates are
applied to per-carry-unit daily returns. US retail cannot trade Binance
perps directly (would use Kraken/Bybit-equivalent venues at similar fees) —
stated honestly; Binance is the DATA source with the longest free history.

GATES (PRE-REGISTERED — pass ALL or verdict = CLOSED NEGATIVE)
--------------------------------------------------------------
- G1: a NAMED primary cell (C0_EW, C1_K5, or CS_3) net Sharpe >= 1.0 at
      15 bps, n_days >= 1,500. PLATEAU: if C1_K5 passes, C1_K0 AND C1_K10
      must be >= 0.5; if CS_3 passes, CS_2 AND CS_4 must be >= 0.5;
      C0_EW passing requires C0_BTC >= 0.5 (composition plateau).
- G2: >= 70% of calendar years positive (first full year 2021 -> 2026).
- G3: HC #428 R1 day-level regime gate, BOTH benchmarks must pass:
      (a) BTC c2c on the P&L-REALIZATION day (green > +1%, red < -1%);
      (b) SPY c2c on the P&L day (green > +0.2%, red < -0.2%), SPY trading
          days only (crypto weekend P&L excluded from the SPY split).
      gap = |Sh_g - Sh_r| / max(|Sh_g|,|Sh_r|) <= 0.50 for each.
      (A carry harvest that is long-BTC-beta or long-SPY-beta in disguise
      must fail here — that is the point of the lane.)
- G4: day-concentration: best day net P&L <= 0.70 of total (HC #344).
- G5: the G1 cell keeps net Sharpe >= 0.5 BOTH at 30 bps AND with
      1-day signal lag (at 15 bps).
DIAGNOSTICS (never drive a PASS): OLS of primary net returns on BTC ret
(beta, R^2, ann alpha w/ NW-t lag 21); funding vs basis P&L decomposition;
per-year Sharpe + mean annualized funding per name per year (premium
compression check); ann turnover; avg N_active.

Report per cell: Sharpe, Sortino, PF, WR, CAGR, MaxDD, Calmar, both regime
gaps, per-year returns, day-conc, ann turnover.

OUTPUT: results/crypto_funding_v1/{summary.json, cells.csv, daily_*.parquet,
data/*.parquet, run.log}; MLflow exp crypto_funding_v1 (http://localhost:5000).
"""
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = "/home/jupiter/Lvl3Quant"
OUT = f"{ROOT}/results/crypto_funding_v1"
DATA = f"{OUT}/data"
MLFLOW_URI = "http://localhost:5000"
EXP = "crypto_funding_v1"
VISION = "https://data.binance.vision/data"
SYMS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "LTCUSDT", "LINKUSDT", "DOGEUSDT", "DOTUSDT", "SOLUSDT"]
MONTHS = pd.period_range("2020-01", "2026-05", freq="M").strftime("%Y-%m").tolist()
ANN = 365.0
COST_PRIMARY = 0.0015   # 15 bps per side per carry unit (spot+perp legs)
COST_STRESS = 0.0030
K_GRID = {"C1_K0": 0.00, "C1_K5": 0.05, "C1_K10": 0.10}   # annualized funding thresholds
CS_GRID = {"CS_2": 2, "CS_3": 3, "CS_4": 4}
TRAIL = 3               # trailing days for funding signal
MIN_HIST = 10
SPY_PARQUET = f"{ROOT}/results/tsmom_etf_v1/data/SPY_daily.parquet"

os.makedirs(DATA, exist_ok=True)


def log(msg):
    print(msg, flush=True)
    with open(f"{OUT}/run.log", "a") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------- download
def _fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception:
        return None


def _read_zip_csv(raw, names):
    z = zipfile.ZipFile(io.BytesIO(raw))
    with z.open(z.namelist()[0]) as f:
        first = f.readline().decode()
        has_header = not first.split(",")[0].strip().isdigit()
        f2 = z.open(z.namelist()[0])
        df = pd.read_csv(f2, header=0 if has_header else None)
        if not has_header:
            df.columns = names[: df.shape[1]] + list(df.columns[len(names):])
        else:
            df.columns = [c.strip().lower() for c in df.columns]
        return df


KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "tb_base", "tb_quote", "ignore"]
FUND_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def _ts_to_date(ts):
    ts = np.asarray(ts, dtype="float64")
    ts = np.where(ts > 1e14, ts / 1000.0, ts)   # guard microsecond timestamps
    return pd.to_datetime(ts, unit="ms", utc=True).date


def download_symbol(sym):
    """Fetch + cache funding, spot 1d, perp 1d for one symbol. Returns dict of DFs."""
    out = {}
    specs = {
        "funding": (f"{VISION}/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-%s.zip", FUND_COLS),
        "spot": (f"{VISION}/spot/monthly/klines/{sym}/1d/{sym}-1d-%s.zip", KLINE_COLS),
        "perp": (f"{VISION}/futures/um/monthly/klines/{sym}/1d/{sym}-1d-%s.zip", KLINE_COLS),
    }
    for kind, (tmpl, names) in specs.items():
        cache = f"{DATA}/{sym}_{kind}.parquet"
        if os.path.exists(cache):
            out[kind] = pd.read_parquet(cache)
            continue
        frames = []
        with ThreadPoolExecutor(12) as ex:
            raws = list(ex.map(lambda m: _fetch(tmpl % m), MONTHS))
        for raw in raws:
            if raw is None:
                continue
            try:
                frames.append(_read_zip_csv(raw, names))
            except Exception:
                pass
        if not frames:
            out[kind] = None
            continue
        df = pd.concat(frames, ignore_index=True)
        if kind == "funding":
            df["date"] = _ts_to_date(df["calc_time"])
            df = (df.groupby("date")["last_funding_rate"].sum()
                    .rename("funding").reset_index())
        else:
            df["date"] = _ts_to_date(df["open_time"])
            df = df[["date", "close"]].copy()
            df["close"] = df["close"].astype(float)
            df = df.drop_duplicates("date").sort_values("date")
        df.to_parquet(cache, index=False)
        out[kind] = df
    return out


# ---------------------------------------------------------------- panel build
def build_panel():
    spot_px, perp_px, fund = {}, {}, {}
    for sym in SYMS:
        d = download_symbol(sym)
        if any(d.get(k) is None for k in ("funding", "spot", "perp")):
            log(f"WARN {sym}: missing dataset(s) {[k for k in d if d[k] is None]} — EXCLUDED")
            continue
        spot_px[sym] = d["spot"].set_index("date")["close"]
        perp_px[sym] = d["perp"].set_index("date")["close"]
        fund[sym] = d["funding"].set_index("date")["funding"]
        log(f"{sym}: spot {len(spot_px[sym])}d {spot_px[sym].index.min()}->{spot_px[sym].index.max()}, "
            f"perp {len(perp_px[sym])}d, funding {len(fund[sym])}d "
            f"(mean ann {fund[sym].mean()*365*100:.1f}%)")
    spot = pd.DataFrame(spot_px).sort_index()
    perp = pd.DataFrame(perp_px).sort_index()
    fr = pd.DataFrame(fund).reindex(spot.index).fillna(0.0)
    spot_ret = spot.pct_change()
    perp_ret = perp.pct_change()
    # carry unit daily return per name (long spot, short perp, +funding)
    carry = (spot_ret - perp_ret + fr)
    # active mask: both legs + funding history >= MIN_HIST days
    active = (spot.notna() & perp.notna()).astype(float)
    active = (active.cumsum() >= MIN_HIST) & spot.notna() & perp.notna()
    carry = carry.where(active)
    return carry, fr, spot_ret, active, spot


# ---------------------------------------------------------------- weights
def weights_for_cell(cell, fr, active):
    """Weights decided at close t (past-only signals), applied to return t+1."""
    n_active = active.sum(axis=1).replace(0, np.nan)
    if cell == "C0_BTC":
        w = pd.DataFrame(0.0, index=active.index, columns=active.columns)
        w["BTCUSDT"] = active["BTCUSDT"].astype(float)
        return w
    if cell == "C0_EW":
        return active.div(n_active, axis=0).fillna(0.0)
    sig = fr.rolling(TRAIL).mean() * ANN          # trailing 3d funding, annualized
    sig = sig.where(active)
    if cell in K_GRID:
        qual = (sig > K_GRID[cell]) & active
        return qual.div(n_active, axis=0).fillna(0.0)
    if cell in CS_GRID:
        k = CS_GRID[cell]
        pos = sig.where(sig > 0)
        rank = pos.rank(axis=1, ascending=False)
        sel = (rank <= k)
        return sel.div(float(k)).fillna(0.0).astype(float)
    raise ValueError(cell)


# ---------------------------------------------------------------- metrics
def perf(daily, ann=ANN):
    d = daily.dropna()
    if len(d) < 30 or d.std() == 0:
        return dict(sharpe=np.nan, n=len(d))
    mu, sd = d.mean(), d.std()
    downside = d[d < 0].std()
    eq = (1 + d).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (ann / len(d)) - 1
    gains, losses = d[d > 0].sum(), -d[d < 0].sum()
    tot = d.sum()
    return dict(
        sharpe=mu / sd * np.sqrt(ann),
        sortino=(mu / downside * np.sqrt(ann)) if downside and downside > 0 else np.nan,
        pf=gains / losses if losses > 0 else np.inf,
        wr=(d > 0).mean(),
        cagr=cagr, maxdd=dd,
        calmar=cagr / abs(dd) if dd < 0 else np.nan,
        worst_day=d.min(), best_day=d.max(),
        day_conc=(d.max() / tot) if tot > 0 else np.nan,
        n=len(d), total=tot,
    )


def regime_gap(daily, bench, thr):
    df = pd.concat([daily.rename("p"), bench.rename("b")], axis=1).dropna()
    g, r = df[df.b > thr].p, df[df.b < -thr].p
    if len(g) < 30 or len(r) < 30 or g.std() == 0 or r.std() == 0:
        return np.nan, np.nan, np.nan
    shg = g.mean() / g.std() * np.sqrt(ANN)
    shr = r.mean() / r.std() * np.sqrt(ANN)
    gap = abs(shg - shr) / max(abs(shg), abs(shr))
    return shg, shr, gap


def nw_alpha_beta(y, x, lags=21):
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    X = np.column_stack([np.ones(len(df)), df.x.values])
    Y = df.y.values
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, lags + 1):
        wgt = 1 - l / (lags + 1)
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S += wgt * (G + G.T)
    V = XtXi @ S @ XtXi
    r2 = 1 - e.var() / Y.var()
    return dict(alpha_ann=b[0] * ANN, alpha_t=b[0] / np.sqrt(V[0, 0]),
                beta=b[1], r2=r2)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    log(f"=== crypto_funding_v1 start {pd.Timestamp.utcnow()} ===")
    carry, fr, spot_ret, active, spot = build_panel()
    idx = carry.index
    btc_ret = spot_ret["BTCUSDT"]
    spy = pd.read_parquet(SPY_PARQUET)
    spy["date"] = pd.to_datetime(spy["date"]).dt.date
    spy_ret = spy.set_index("date")["adj_close"].pct_change().reindex(idx)

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXP)

    cells = ["C0_BTC", "C0_EW", "C1_K0", "C1_K5", "C1_K10", "CS_2", "CS_3", "CS_4"]
    rows, daily_store = [], {}
    for cell in cells:
        w = weights_for_cell(cell, fr, active)
        for variant, (cost, lag) in {
            "primary": (COST_PRIMARY, 1),
            "stress30": (COST_STRESS, 1),
            "lag1": (COST_PRIMARY, 2),
        }.items():
            wl = w.shift(lag)
            gross = (wl * carry).sum(axis=1)
            turn = (wl - wl.shift(1)).abs().sum(axis=1)
            net = gross - cost * turn
            net = net[wl.notna().any(axis=1)]
            m = perf(net)
            shg_b, shr_b, gap_btc = regime_gap(net, btc_ret, 0.01)
            shg_s, shr_s, gap_spy = regime_gap(net, spy_ret, 0.002)
            yr = net.groupby(pd.Series(net.index, index=net.index).map(lambda d: d.year)).sum()
            yrs_pos = (yr.loc[[y for y in yr.index if y >= 2021]] > 0)
            m.update(cell=cell, variant=variant, cost_bps=cost * 1e4,
                     gap_btc=gap_btc, sh_btc_green=shg_b, sh_btc_red=shr_b,
                     gap_spy=gap_spy, sh_spy_green=shg_s, sh_spy_red=shr_s,
                     years_pos=int(yrs_pos.sum()), years_n=int(len(yrs_pos)),
                     ann_turnover=turn.mean() * ANN,
                     avg_gross=wl.abs().sum(axis=1).mean())
            rows.append(m)
            if variant == "primary":
                daily_store[cell] = net
                net.rename("ret").to_frame().assign(date=net.index).to_parquet(
                    f"{OUT}/daily_{cell}.parquet", index=False)
                with mlflow.start_run(run_name=f"{cell}_primary"):
                    mlflow.log_params(dict(cell=cell, cost_bps=15, lag=1, trail=TRAIL))
                    mlflow.log_metrics({k: float(v) for k, v in m.items()
                                        if isinstance(v, (int, float, np.floating))
                                        and np.isfinite(v)})
            log(f"{cell:7s} {variant:9s} Sh {m.get('sharpe', np.nan):6.2f} "
                f"Sor {m.get('sortino', np.nan):6.2f} PF {m.get('pf', np.nan):5.2f} "
                f"WR {m.get('wr', np.nan):.2f} CAGR {m.get('cagr', np.nan)*100:6.1f}% "
                f"DD {m.get('maxdd', np.nan)*100:6.1f}% gapBTC {gap_btc:5.2f} "
                f"gapSPY {gap_spy:5.2f} yrs+ {m['years_pos']}/{m['years_n']} "
                f"turn {m['ann_turnover']:.1f}x")

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/cells.csv", index=False)

    # ----------------------------------------------------------- gates
    prim = res[res.variant == "primary"].set_index("cell")
    def cell_g1(c):
        if not (prim.loc[c, "sharpe"] >= 1.0 and prim.loc[c, "n"] >= 1500):
            return False
        if c == "C1_K5":
            return prim.loc["C1_K0", "sharpe"] >= 0.5 and prim.loc["C1_K10", "sharpe"] >= 0.5
        if c == "CS_3":
            return prim.loc["CS_2", "sharpe"] >= 0.5 and prim.loc["CS_4", "sharpe"] >= 0.5
        if c == "C0_EW":
            return prim.loc["C0_BTC", "sharpe"] >= 0.5
        return False
    g1_cells = [c for c in ["C0_EW", "C1_K5", "CS_3"] if cell_g1(c)]
    gates, passing = {}, []
    for c in g1_cells:
        p = prim.loc[c]
        s = res[(res.cell == c) & (res.variant == "stress30")].iloc[0]
        l = res[(res.cell == c) & (res.variant == "lag1")].iloc[0]
        g = dict(
            G1=True,
            G2=p.years_pos / max(p.years_n, 1) >= 0.70,
            G3=(p.gap_btc <= 0.50) and (p.gap_spy <= 0.50),
            G4=p.day_conc <= 0.70,
            G5=(s.sharpe >= 0.5) and (l.sharpe >= 0.5),
        )
        gates[c] = g
        if all(g.values()):
            passing.append(c)

    # ----------------------------------------------------------- diagnostics
    diag = {}
    ew = daily_store["C0_EW"]
    diag["ols_vs_btc"] = nw_alpha_beta(ew, btc_ret)
    # funding vs basis decomposition for C0_EW
    w_ew = weights_for_cell("C0_EW", fr, active).shift(1)
    fund_pnl = (w_ew * fr).sum(axis=1)
    basis_pnl = (w_ew * (spot_ret - spot_ret)).sum(axis=1)  # placeholder shape
    basis_pnl = (w_ew * carry).sum(axis=1) - fund_pnl
    diag["decomp_ann"] = dict(funding=float(fund_pnl.mean() * ANN),
                              basis=float(basis_pnl.mean() * ANN))
    diag["funding_ann_by_year"] = {
        str(y): {s: float(v) for s, v in (g.mean() * ANN).dropna().items()}
        for y, g in fr.where(active).groupby(
            pd.Series(fr.index, index=fr.index).map(lambda d: d.year))}
    diag["per_year_sharpe_C0_EW"] = {
        str(y): float(g.mean() / g.std() * np.sqrt(ANN)) if g.std() > 0 else None
        for y, g in ew.groupby(pd.Series(ew.index, index=ew.index).map(lambda d: d.year))}

    verdict = "PASS" if passing else "CLOSED_NEGATIVE"
    summary = dict(
        lane="crypto_funding_v1", verdict=verdict, passing_cells=passing,
        gates={c: {k: bool(v) for k, v in g.items()} for c, g in gates.items()},
        g1_eligible=g1_cells,
        coverage=dict(start=str(idx.min()), end=str(idx.max()), days=int(len(idx)),
                      symbols=list(carry.columns)),
        diagnostics=diag,
        runtime_s=round(time.time() - t0, 1),
    )
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with mlflow.start_run(run_name="SUMMARY"):
        mlflow.log_params(dict(verdict=verdict, passing=",".join(passing) or "none",
                               n_days=int(len(idx))))
        mlflow.log_artifact(f"{OUT}/summary.json")
        mlflow.log_artifact(f"{OUT}/cells.csv")

    log(f"=== VERDICT: {verdict} (passing: {passing}) runtime {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
