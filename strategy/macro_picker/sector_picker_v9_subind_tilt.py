"""
Sector picker v9 — sub-industry TILT overlay on v6 sparse-K (HC #573 R2).

v8 lesson (failed): treating each sub-industry as an independent tradeable bucket
gave 11 active buckets but pooled Sharpe collapsed to 0.09. The sub-buckets are
too small/noisy individually.

v9 idea: keep v6's sector-level sparse-K ridge unchanged, but TILT the per-name
score by how "hot" the name's sub-industry is in the recent past, PIT-safely.

    score_tilted = score + LAMBDA * subind_momentum_z

where subind_momentum_z is computed from the trailing 20-day SUB-INDUSTRY-AVERAGE
return of all panel members in that sub-industry, cross-sectionally Z-scored
within the sector at the rebalance date.

If sub-industry rotation has any edge as a SIGNAL, this should bump the picker's
Sharpe up over v6's. If not, LAMBDA=0 reverts to v6 exactly.

Run:
    cd /home/jupiter/Lvl3Quant
    python3 strategy/macro_picker/sector_picker_v9_subind_tilt.py \
        --K 10 --hold-days 10 --lam 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))
import sector_picker_v4 as v4  # noqa: E402
import sector_picker_v6_sparseK as v6  # noqa: E402
from sub_industry_taxonomy import build_ticker_to_subindustries  # noqa: E402

PANEL_V2 = ROOT / "data/feature_store/master_panel/master_panel_v2.parquet"
T2S = build_ticker_to_subindustries()
# Single primary sub-industry per ticker (first-listed).
T2_PRIMARY = {t: ss[0] for t, ss in T2S.items() if ss}

TOP_N = v6.TOP_N
BOT_N = v6.BOT_N
TXN_COST_BPS = v6.TXN_COST_BPS


def _attach_subind_momentum(sp: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Add 'subind_mom' = trailing N-day avg ret of all panel members in
    same sub-industry, evaluated at each (date, ticker). PIT-safe: uses
    only ret data with date < d (rolling, no peeking).
    """
    sp = sp.copy()
    sp["subind"] = sp["ticker"].map(T2_PRIMARY).fillna("__none__")
    # Sub-industry mean return per date (cross-section of names that day)
    sub_daily = (sp.groupby(["date", "subind"])["ret"]
                   .mean().reset_index().rename(columns={"ret": "sub_ret"}))
    sub_daily = sub_daily.sort_values(["subind", "date"]).reset_index(drop=True)
    sub_daily["sub_mom"] = (sub_daily.groupby("subind")["sub_ret"]
                            .rolling(lookback, min_periods=10).mean()
                            .shift(1)  # PIT: don't include today
                            .reset_index(level=0, drop=True))
    sp = sp.merge(sub_daily[["date", "subind", "sub_mom"]],
                  on=["date", "subind"], how="left")
    return sp


def portfolio_for_sector_v9(
    sp: pd.DataFrame,
    feats: list[str],
    K: int,
    hold_days: int,
    lam: float,
):
    sp = sp.sort_values(["date", "ticker"]).reset_index(drop=True)
    sp["ret_raw"] = sp["ret"].astype(float)
    sp = _attach_subind_momentum(sp, lookback=20)

    start, end = sp["date"].min(), sp["date"].max()
    cursor = start
    daily_pnl = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    folds = []

    while True:
        tr_start = cursor
        tr_end = tr_start + pd.DateOffset(months=36)
        oot_start = tr_end
        oot_end = oot_start + pd.DateOffset(months=12)
        if oot_end > end + pd.Timedelta(days=1):
            break

        train = sp[(sp["date"] >= tr_start) & (sp["date"] < tr_end)]
        oot = sp[(sp["date"] >= oot_start) & (sp["date"] < oot_end)]
        if len(train) < 1000 or len(oot) < 100:
            cursor = cursor + pd.DateOffset(months=6); continue

        train_z = v4._xs_zscore(train, feats)
        oot_z = v4._xs_zscore(oot, feats)
        for f in feats:
            train_z[f] = train_z[f].fillna(0.0)
            oot_z[f] = oot_z[f].fillna(0.0)
        train_z = train_z.dropna(subset=["y_fwd"])
        if train_z.empty:
            cursor = cursor + pd.DateOffset(months=6); continue

        chosen, _ic = v6._train_fold_top_k(train_z, feats, K)
        X_tr = train_z[chosen].values
        y_tr = train_z["y_fwd"].values
        coef, intercept, _ = v4._fit_ridge(X_tr, y_tr)
        if coef is None:
            cursor = cursor + pd.DateOffset(months=6); continue

        X_oot = oot_z[chosen].fillna(0.0).values
        oot_z = oot_z.copy()
        oot_z["score_raw"] = X_oot @ coef + intercept

        # ---- V9 TILT: add lam * sub-industry momentum z (within sector/day) ----
        # sub_mom is already on oot_z (carried through from sp via .copy()
        # in _xs_zscore). Just compute the date-wise cross-section Z.
        if "sub_mom" not in oot_z.columns:
            oot_z["sub_mom"] = 0.0
        g_date = oot_z.groupby("date")["sub_mom"]
        mu = g_date.transform("mean")
        sd = g_date.transform("std")
        oot_z["sub_mom_z"] = ((oot_z["sub_mom"] - mu) / sd.replace(0, np.nan)).fillna(0.0)
        oot_z["score"] = oot_z["score_raw"] + lam * oot_z["sub_mom_z"]

        unique_dates = sorted(oot_z["date"].unique())
        rebal_dates = unique_dates[::hold_days]
        fold_daily = []
        for rd in rebal_dates:
            snap = oot_z[oot_z["date"] == rd].dropna(subset=["score"])
            if len(snap) < (TOP_N + BOT_N):
                continue
            longs = snap.nlargest(TOP_N, "score")["ticker"].tolist()
            shorts = snap.nsmallest(BOT_N, "score")["ticker"].tolist()
            hold_win = oot_z[(oot_z["date"] > rd)
                             & (oot_z["date"] <= rd + pd.Timedelta(days=hold_days))]
            for d, g in hold_win.groupby("date"):
                lret = g[g["ticker"].isin(longs)]["ret_raw"].mean() if longs else 0.0
                sret = g[g["ticker"].isin(shorts)]["ret_raw"].mean() if shorts else 0.0
                day_ret = float(np.clip(
                    0.5 * (lret if pd.notna(lret) else 0)
                    - 0.5 * (sret if pd.notna(sret) else 0),
                    -0.25, 0.25))
                fold_daily.append((d, day_ret))
            tc = TXN_COST_BPS / 10000.0
            fold_daily.append((rd, -tc))

        if fold_daily:
            s = pd.Series(dict(fold_daily))
            s.index = pd.to_datetime(s.index)
            s = s.groupby(level=0).sum()
            daily_pnl = pd.concat([daily_pnl, s])

        folds.append({"tr_start": str(tr_start.date()),
                      "tr_end": str(tr_end.date()),
                      "oot_start": str(oot_start.date()),
                      "oot_end": str(oot_end.date()),
                      "chosen": chosen})
        cursor = cursor + pd.DateOffset(months=6)

    return {"daily_pnl": daily_pnl, "folds": folds}


def _run_one_sector(sector, panel, feats, K, hold_days, lam):
    sp = panel[panel["sector"] == sector].copy()
    if sp.empty:
        return sector, None
    res = portfolio_for_sector_v9(sp, feats, K, hold_days, lam)
    if res["daily_pnl"].empty:
        return sector, None
    return sector, res


def _vol_target_series(daily: pd.Series, target_ann_vol: float = 0.15,
                       lookback: int = 20, max_lev: float = 2.0) -> pd.Series:
    """Scale each day's return by target_vol / trailing_realized_vol.
    PIT: vol estimated on data UP TO (not including) day t.
    """
    if daily.empty:
        return daily
    daily = daily.sort_index()
    rv = daily.rolling(lookback, min_periods=10).std().shift(1) * np.sqrt(252)
    lev = (target_ann_vol / rv).clip(upper=max_lev).fillna(1.0)
    return daily * lev


def _regime_gate(daily: pd.Series, spy_ret: pd.Series,
                 lookback: int = 20, shrink: float = 0.5) -> pd.Series:
    """Halve position size on days when SPY 20d realized vol is in the
    top half of its trailing 252d distribution (high-vol regime).
    """
    if daily.empty or spy_ret.empty:
        return daily
    spy_rv = spy_ret.rolling(lookback, min_periods=10).std().shift(1) * np.sqrt(252)
    spy_rv_med = spy_rv.rolling(252, min_periods=60).median().shift(1)
    high_vol = (spy_rv > spy_rv_med).reindex(daily.index).fillna(False)
    return daily * np.where(high_vol, shrink, 1.0)


def _metrics(daily: pd.Series) -> dict:
    if daily.empty:
        return {"sharpe": 0, "sortino": 0, "cagr": 0,
                "max_dd": 0, "calmar": 0, "wr": 0, "pf": 0, "n": 0}
    eq = (1 + daily).cumprod()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() else 0
    down = daily[daily < 0].std()
    sortino = daily.mean() / down * np.sqrt(252) if down else 0
    n_days = (eq.index[-1] - eq.index[0]).days
    cagr = eq.iloc[-1] ** (365.25 / n_days) - 1 if n_days > 0 else 0
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd else 0
    wr = (daily > 0).mean()
    pf = daily[daily > 0].sum() / -daily[daily < 0].sum() if (daily < 0).any() else 0
    return {"sharpe": float(sharpe), "sortino": float(sortino),
            "cagr": float(cagr), "max_dd": float(max_dd),
            "calmar": float(calmar), "wr": float(wr), "pf": float(pf),
            "n": int(len(daily))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--hold-days", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.5,
                    help="sub-industry momentum tilt weight (0 = pure v6)")
    ap.add_argument("--vol-target", type=float, default=0.0,
                    help="annualized vol target per sector before pooling (e.g. 0.15). 0 disables.")
    ap.add_argument("--regime-gate", action="store_true",
                    help="halve size on high-vol SPY regime days")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out or (
        ROOT / f"output/macro_picker/v9_subind_tilt_{stamp}/K{args.K}_H{args.hold_days}d_L{args.lam}"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v9 tilt] K={args.K} H={args.hold_days}d  lam={args.lam}  -> {out_dir}")

    print("loading panel ...")
    panel = pd.read_parquet(PANEL_V2)
    panel["date"] = pd.to_datetime(panel["date"])
    feats = [c for c in panel.columns if c not in (
        "ticker", "date", "sector", "ret", "y_fwd", "open", "high", "low",
        "close", "volume", "log_ret",
    )]
    # build y_fwd (forward hold-days return)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["y_fwd"] = (panel.groupby("ticker")["ret"]
                      .rolling(args.hold_days).sum().shift(-args.hold_days)
                      .reset_index(level=0, drop=True))
    sectors = sorted([s for s in panel["sector"].dropna().unique() if s != "ETF"])
    print(f"sectors: {sectors}")

    t0 = time.time()
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_run_one_sector)(s, panel, feats, args.K, args.hold_days, args.lam)
        for s in sectors
    )
    print(f"  parallel wall: {time.time()-t0:.1f}s")

    # Pre-pool SPY returns for regime gate
    spy_ret = pd.Series(dtype=float)
    if args.regime_gate:
        spy_slice = panel[panel["ticker"] == "SPY"][["date", "ret"]].dropna()
        if not spy_slice.empty:
            spy_ret = spy_slice.set_index("date")["ret"].sort_index()
            spy_ret = spy_ret[~spy_ret.index.duplicated(keep="first")]

    pooled = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    per_sec = {}
    for sec, r in results:
        if r is None:
            continue
        sec_daily = r["daily_pnl"].copy()
        # Per-sector vol target (applied BEFORE pooling)
        if args.vol_target and args.vol_target > 0:
            sec_daily = _vol_target_series(sec_daily, target_ann_vol=args.vol_target)
        m = _metrics(sec_daily)
        per_sec[sec] = m
        pooled = pd.concat([pooled, sec_daily])

    pooled = pooled.groupby(level=0).sum()
    # Equal-weight across active sectors (sum / N)
    n_active = max(1, len(per_sec))
    pooled = pooled / n_active
    # Regime gate on pooled
    if args.regime_gate and not spy_ret.empty:
        pooled = _regime_gate(pooled, spy_ret)
    pooled_m = _metrics(pooled)

    rep = {"K": args.K, "hold_days": args.hold_days, "lam": args.lam,
           "pooled": pooled_m, "per_sector": per_sec,
           "n_sectors_deployable": sum(1 for m in per_sec.values() if m["sharpe"] >= 1.0)}
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    pooled.to_frame("ret").to_parquet(out_dir / "pooled_daily.parquet")

    print("\n=== POOLED ===")
    print(f"  Sharpe   {pooled_m['sharpe']:.2f}")
    print(f"  Sortino  {pooled_m['sortino']:.2f}")
    print(f"  CAGR     {pooled_m['cagr']*100:.1f}%")
    print(f"  MaxDD    {pooled_m['max_dd']*100:.1f}%")
    print(f"  Calmar   {pooled_m['calmar']:.2f}")
    print(f"  Deploy sectors (Sharpe>=1.0): {rep['n_sectors_deployable']}/{len(per_sec)}")
    print("\n=== PER SECTOR (top 5) ===")
    sorted_secs = sorted(per_sec.items(), key=lambda x: -x[1]["sharpe"])
    for sec, m in sorted_secs[:5]:
        print(f"  {sec:25s}  Sharpe {m['sharpe']:.2f}  CAGR {m['cagr']*100:.1f}%  "
              f"DD {m['max_dd']*100:.1f}%  PF {m['pf']:.2f}")
    print(f"\nwrote {out_dir}/report.json")


if __name__ == "__main__":
    main()
