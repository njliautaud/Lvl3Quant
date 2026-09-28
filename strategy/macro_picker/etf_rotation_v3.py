"""
ETF rotation v3 — sector-SPDR cross-sectional ridge rotation + MACRO REGIME GATE.

PATH-A DESIGN (HC #428 R1 backlog P2-3 follow-up):
Sibling agent's v2 finding:
  (1) Broadcast macro features get zeroed by `_xs_zscore()` (cross-sectional
      normalisation across 11 ETFs → identical value across ETFs → z=0 →
      ridge coef ≈ 0). So adding macro AS PICKER INPUTS is dead on arrival.
  (2) v1 itself FAILS HC #428 R1 regime symmetry:
      Sharpe_green +8.63 vs Sharpe_red -5.04  →  skew ratio 1.58  (cap 0.50).
      The SPY-MA60 overlay is doing all the regime work, and it's still skewed.

v3 sidesteps the z-score wipeout by using macro as a SECOND-STAGE REGIME GATE
that runs AFTER the SPY-MA60 bull-pass. On every rebalance + intra-hold-check
day where SPY > MA60 (bull pass), apply a macro check. KEEP_FLAT (sit cash)
when ANY of:
  - vix_chg_20d   >  THRESH_VIX_CHG     (vol spiking)
  - vix_term      >  THRESH_VIX_TERM    (VIX/VIX3M term inverted)
  - dxy_chg_20d   >  THRESH_DXY_CHG     (dollar shock)
  - sector_disp   <  THRESH_DISP_PCT-th trailing-252d percentile
                                          (low cross-sectional opportunity)

This targets red-regime bleed by sitting out when macro says "stress" while
the SPY-MA60 says "still technically bull".

EVERYTHING ELSE IS IDENTICAL TO v1 (same picker, same sizing, same WF).

CLI:
  python3 etf_rotation_v3.py [--hold-days 21] [--n-long 2] [--no-short]
                             [--regime-overlay] [--out DIR] [--screen|--single]
                             [--thresh-vix-chg 5] [--thresh-vix-term 1.10]
                             [--thresh-dxy-chg 2.5] [--thresh-disp-pct 10]
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
from walk_forward import _metrics  # type: ignore  # noqa: E402

# -----------------------------------------------------------------------------
# Data paths
# -----------------------------------------------------------------------------
ETF_FLOWS_PATH = ROOT / "data/feature_store/sector_etf_flows/daily.parquet"
SECTOR_ROT_PATH = ROOT / "data/feature_store/sector_rotation/daily.parquet"
CROSS_ASSET_PATH = ROOT / "data/feature_store/cross_asset/daily.parquet"
MACRO_EXTRA_PATH = ROOT / "wheel_strategy_v1/data/cache/macro_extra.parquet"
MASTER_PANEL_PATH = ROOT / "data/feature_store/master_panel/master_panel_v2.parquet"
SPY_PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"
VIX_PATH = ROOT / "wheel_strategy_v1/data/cache/vix_history.parquet"
# Optional VIX3M parquet for term-structure check. May not exist on disk; if
# missing we fall back to a NaN vix_term that never triggers KEEP_FLAT.
VIX3M_PATH = ROOT / "wheel_strategy_v1/data/cache/vix3m_history.parquet"

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLU", "XLI", "XLV", "XLB", "XLC", "XLRE"]
BENCHMARK = "SPY"

ETF_TO_SECTOR_LABEL = {
    "XLK": "Information Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLI": "Industrials",
    "XLV": "Health Care",
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLRE": "Real Estate",
}

TRADING_DAYS = 252
TXN_COST_BPS = 5

FUND_FEATURES = [
    "fp_gross_margin", "fp_net_margin", "fp_ebitda_margin",
    "fp_roe", "fp_debt_to_equity", "fp_current_ratio",
    "fp_fcf_yield", "fp_rev_yoy_growth", "fp_ni_yoy_growth",
    "fp_eps_yoy_growth", "fp_fcf_yoy_growth", "fp_margin_trend_4q",
    "fp_beat_rate_4q",
]

FLOW_FEATURES_RAW = [
    "ret_1d", "ret_20d", "ret_60d", "rel_strength_spy",
    "momentum_cross_20_60", "rs_rank_among_sectors",
]

# v1 macro features (kept as picker inputs — same as v1, they're ~0 coef but
# preserved for harness identity).
MACRO_FEATURES_V1 = [
    "macro_ust_10y", "macro_ust_10y_chg_20d",
    "macro_yc_2s10s",
    "macro_dxy_z", "macro_dxy_ret_20d",
    "macro_oil_ret_20d",
    "macro_gold_ret_20d",
]


# -----------------------------------------------------------------------------
# Utilities (mirror v1)
# -----------------------------------------------------------------------------
def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _xs_zscore(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = panel.copy()
    for f in feats:
        if f not in out.columns:
            out[f] = 0.0
            continue
        x = pd.to_numeric(out[f], errors="coerce").astype(float)
        x = _winsorize(x, 0.01)
        out[f] = x
        mu = out.groupby("date")[f].transform("mean")
        sd = out.groupby("date")[f].transform("std")
        z = (x - mu) / sd.replace(0.0, np.nan)
        out[f] = z.replace([np.inf, -np.inf], np.nan)
    return out


def _fit_ridge(X: np.ndarray, y: np.ndarray,
               alphas=(0.1, 1.0, 10.0, 100.0)) -> tuple:
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    n, p = Xc.shape
    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    best = (None, None, None, float("inf"))
    for a in alphas:
        try:
            beta = np.linalg.solve(XtX + a * np.eye(p), Xty)
            resid = yc - Xc @ beta
            mse = float((resid ** 2).mean())
            if mse < best[3]:
                best = (beta, y.mean() - X.mean(axis=0) @ beta, a, mse)
        except np.linalg.LinAlgError:
            continue
    return best[0], best[1], best[2]


# -----------------------------------------------------------------------------
# Panel builders
# -----------------------------------------------------------------------------
def _load_flow_panel() -> pd.DataFrame:
    flows = pd.read_parquet(ETF_FLOWS_PATH)
    flows["date"] = pd.to_datetime(flows["date"])
    flows = flows[flows["etf"].isin(SECTOR_ETFS + [BENCHMARK])].copy()

    rot = pd.read_parquet(SECTOR_ROT_PATH)
    rot["date"] = pd.to_datetime(rot["date"])
    rot = rot[rot["etf"].isin(SECTOR_ETFS + [BENCHMARK])].copy()

    merged = flows.merge(
        rot[["etf", "date", "momentum_cross_20_60",
             "rs_rank_among_sectors", "lead_lag_score_5d"]],
        on=["etf", "date"], how="left",
    )
    return merged


def _load_sector_fundamentals() -> pd.DataFrame:
    cols = ["date", "sector"] + FUND_FEATURES
    mp = pd.read_parquet(MASTER_PANEL_PATH, columns=cols)
    mp["date"] = pd.to_datetime(mp["date"])
    mp = mp.dropna(subset=["sector"])
    mp = mp[mp["sector"] != "ETF"]
    agg = mp.groupby(["date", "sector"], as_index=False)[FUND_FEATURES].mean()
    agg = agg.rename(columns={f: f"{f}_sec" for f in FUND_FEATURES})
    return agg


def _load_macro_panel_v1() -> pd.DataFrame:
    """v1's macro features ONLY (broadcast). v3 picker stays identical to v1."""
    macro_extra = pd.read_parquet(MACRO_EXTRA_PATH)
    macro_extra["date"] = pd.to_datetime(macro_extra["date"])
    cross = pd.read_parquet(CROSS_ASSET_PATH)
    cross["date"] = pd.to_datetime(cross["date"])

    out = pd.DataFrame({"date": macro_extra["date"]})
    out["macro_ust_10y"] = macro_extra["ust_10y"].astype(float).values
    out["macro_yc_2s10s"] = macro_extra["yc_2s10s"].astype(float).values
    out = out.sort_values("date").reset_index(drop=True)
    out["macro_ust_10y_chg_20d"] = out["macro_ust_10y"].diff(20)

    def _pivot(asset: str, col: str, out_col: str) -> pd.DataFrame:
        sub = cross[cross["asset"] == asset][["date", col]].rename(columns={col: out_col})
        return sub

    dxy_z = _pivot("DXY", "zscore_60d", "macro_dxy_z")
    dxy_r = _pivot("DXY", "ret_20d", "macro_dxy_ret_20d")
    oil_r = _pivot("OIL", "ret_20d", "macro_oil_ret_20d")
    gold_r = _pivot("GOLD", "ret_20d", "macro_gold_ret_20d")

    out = (out.merge(dxy_z, on="date", how="left")
              .merge(dxy_r, on="date", how="left")
              .merge(oil_r, on="date", how="left")
              .merge(gold_r, on="date", how="left"))
    out = out.sort_values("date").ffill()
    return out


def _load_spy_for_benchmark() -> pd.Series:
    px = pd.read_parquet(SPY_PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    return spy.pct_change().dropna()


def _load_spy_regime(ma_days: int = 60) -> pd.Series:
    px = pd.read_parquet(SPY_PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    ma = spy.rolling(ma_days, min_periods=max(20, ma_days // 2)).mean()
    out = pd.Series(np.where(spy > ma, "bull", "bear"), index=spy.index, name="spy_regime")
    return out.dropna()


# -----------------------------------------------------------------------------
# v3: macro REGIME GATE (separate from picker — not z-scored)
# -----------------------------------------------------------------------------
def _load_macro_regime_signals() -> pd.DataFrame:
    """Build per-date macro regime signals used by the v3 KEEP_FLAT gate.

    Columns:
      date
      vix_chg_20d        — VIX(t) - VIX(t-20)   (vol pts)
      vix_term           — VIX / VIX3M          (>1.0 = term inverted)
      dxy_chg_20d        — DXY 20d return * 100 (percent)
      sector_disp_raw    — stdev of sector ret_20d across 11 SPDRs (per date)
      sector_disp_pct_252— trailing-252d cross-time percentile of sector_disp_raw
    """
    # --- VIX
    vix = pd.read_parquet(VIX_PATH)
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.sort_values("date").reset_index(drop=True)
    vix["vix_chg_20d"] = vix["close"].diff(20)

    # --- VIX3M (optional)
    if VIX3M_PATH.exists():
        vix3m = pd.read_parquet(VIX3M_PATH)
        vix3m["date"] = pd.to_datetime(vix3m["date"])
        vix3m = vix3m.sort_values("date").reset_index(drop=True)
        merged = vix.merge(vix3m[["date", "close"]].rename(columns={"close": "vix3m"}),
                            on="date", how="left")
        merged["vix_term"] = merged["close"] / merged["vix3m"]
    else:
        merged = vix.copy()
        merged["vix_term"] = np.nan  # will never trigger KEEP_FLAT

    out = merged[["date", "vix_chg_20d", "vix_term"]].copy()

    # --- DXY 20d % return
    cross = pd.read_parquet(CROSS_ASSET_PATH)
    cross["date"] = pd.to_datetime(cross["date"])
    dxy = cross[cross["asset"] == "DXY"][["date", "ret_20d"]].copy()
    dxy = dxy.rename(columns={"ret_20d": "dxy_chg_20d"})
    # ret_20d in cross_asset is fractional (e.g. 0.025 = +2.5%). Convert to pct.
    dxy["dxy_chg_20d"] = dxy["dxy_chg_20d"].astype(float) * 100.0
    out = out.merge(dxy, on="date", how="left")

    # --- sector dispersion (stdev of sector ret_20d) + trailing-252d percentile
    flow = _load_flow_panel()
    sec = flow[flow["etf"].isin(SECTOR_ETFS)][["date", "etf", "ret_20d"]].copy()
    sec["ret_20d"] = pd.to_numeric(sec["ret_20d"], errors="coerce")
    disp = sec.groupby("date")["ret_20d"].std(ddof=1).reset_index()
    disp = disp.rename(columns={"ret_20d": "sector_disp_raw"})
    disp = disp.sort_values("date").reset_index(drop=True)

    # trailing-252d percentile of the dispersion value (0..100).
    # rolling rank with min_periods=60 so early-history doesn't crash the gate.
    def _rolling_pct(s: pd.Series, window: int = 252, min_p: int = 60) -> pd.Series:
        return s.rolling(window=window, min_periods=min_p).apply(
            lambda w: 100.0 * (w.rank(pct=True).iloc[-1]) if len(w) else np.nan,
            raw=False,
        )

    disp["sector_disp_pct_252"] = _rolling_pct(disp["sector_disp_raw"])
    out = out.merge(disp[["date", "sector_disp_raw", "sector_disp_pct_252"]],
                     on="date", how="left")
    out = out.sort_values("date").ffill()
    return out


def _macro_keep_flat(macro_sig: pd.DataFrame, d: pd.Timestamp,
                      thr_vix_chg: float, thr_vix_term: float,
                      thr_dxy_chg: float, thr_disp_pct: float) -> tuple:
    """Return (keep_flat: bool, reason: str|None).

    keep_flat=True if ANY of the threshold conditions trip on date d.
    NaN values do NOT trigger keep_flat (be permissive — sit in cash only on
    POSITIVE evidence of stress).
    """
    row = macro_sig[macro_sig["date"] == pd.Timestamp(d)]
    if row.empty:
        # as-of fallback: last known macro snapshot ≤ d
        prior = macro_sig[macro_sig["date"] <= pd.Timestamp(d)]
        if prior.empty:
            return (False, None)
        row = prior.tail(1)
    r = row.iloc[0]

    vc = r.get("vix_chg_20d", np.nan)
    if pd.notna(vc) and vc > thr_vix_chg:
        return (True, f"vix_chg_20d={vc:.2f}>thr{thr_vix_chg}")

    vt = r.get("vix_term", np.nan)
    if pd.notna(vt) and vt > thr_vix_term:
        return (True, f"vix_term={vt:.3f}>thr{thr_vix_term}")

    dx = r.get("dxy_chg_20d", np.nan)
    if pd.notna(dx) and dx > thr_dxy_chg:
        return (True, f"dxy_chg_20d={dx:.2f}%>thr{thr_dxy_chg}%")

    dp = r.get("sector_disp_pct_252", np.nan)
    if pd.notna(dp) and dp < thr_disp_pct:
        return (True, f"sector_disp_pct={dp:.1f}<thr{thr_disp_pct}")

    return (False, None)


def build_panel(hold_days: int) -> tuple[pd.DataFrame, list[str]]:
    flow = _load_flow_panel()
    fund = _load_sector_fundamentals()
    macro = _load_macro_panel_v1()

    flow["sector_label"] = flow["etf"].map(ETF_TO_SECTOR_LABEL)
    fund_renamed = fund.rename(columns={"sector": "sector_label"})

    panel = flow.merge(fund_renamed, on=["date", "sector_label"], how="left")
    panel = panel.merge(macro, on="date", how="left")

    panel = panel.sort_values(["etf", "date"]).reset_index(drop=True)
    panel["close"] = panel["close"].astype(float)
    panel["y_fwd"] = (panel.groupby("etf")["close"].shift(-hold_days) / panel["close"] - 1.0)

    trade_panel = panel[panel["etf"].isin(SECTOR_ETFS)].copy()

    fund_cols = [f"{f}_sec" for f in FUND_FEATURES]
    feats_all = FLOW_FEATURES_RAW + fund_cols + MACRO_FEATURES_V1
    feats_present = [f for f in feats_all if f in trade_panel.columns]

    return trade_panel, feats_present


# -----------------------------------------------------------------------------
# Walk-forward fold engine (v3 — adds macro KEEP_FLAT gate)
# -----------------------------------------------------------------------------
def _estimate_book_vol(panel: pd.DataFrame, rd: pd.Timestamp,
                       longs: list[str], shorts: list[str],
                       lookback_days: int = 60) -> float:
    cutoff_lo = rd - pd.Timedelta(days=lookback_days * 2 + 10)
    hist = panel[(panel["date"] < rd) & (panel["date"] >= cutoff_lo)]
    hist = hist[hist["etf"].isin(set(longs) | set(shorts))]
    if hist.empty:
        return 0.0
    by_date = hist.groupby("date").apply(
        lambda g: (g[g["etf"].isin(longs)]["ret_1d"].mean() if longs else 0.0)
                  - (g[g["etf"].isin(shorts)]["ret_1d"].mean() if shorts else 0.0)
    )
    by_date = by_date.dropna().tail(lookback_days)
    if len(by_date) < 20:
        return 0.0
    daily_sd = float(by_date.std(ddof=1))
    if not np.isfinite(daily_sd):
        return 0.0
    return daily_sd * np.sqrt(TRADING_DAYS)


def _wf_fold(panel: pd.DataFrame, feats: list[str], tr_start: pd.Timestamp,
             tr_end: pd.Timestamp, oot_start: pd.Timestamp, oot_end: pd.Timestamp,
             hold_days: int, n_long: int, n_short: int,
             allow_short: bool,
             target_vol: float = 0.15,
             lev_min: float = 0.25, lev_max: float = 2.0,
             txn_cost_bps: float = 5.0,
             regime_filter: pd.Series | None = None,
             macro_sig: pd.DataFrame | None = None,
             thr_vix_chg: float = 5.0, thr_vix_term: float = 1.10,
             thr_dxy_chg: float = 2.5, thr_disp_pct: float = 10.0) -> dict:
    train = panel[(panel["date"] >= tr_start) & (panel["date"] < tr_end)].copy()
    oot = panel[(panel["date"] >= oot_start) & (panel["date"] < oot_end)].copy()

    if len(train) < 200 or len(oot) < 20:
        return {"daily_pnl": pd.Series(dtype=float), "coef": {}, "alpha": None,
                "oot_start": str(oot_start.date()), "oot_end": str(oot_end.date()),
                "n_rebal": 0, "n_macro_flat_days": 0, "n_traded_days": 0,
                "n_spy_bear_skips": 0}

    train_z = _xs_zscore(train, feats)
    oot_z = _xs_zscore(oot, feats)
    for f in feats:
        train_z[f] = train_z[f].fillna(0.0)
        oot_z[f] = oot_z[f].fillna(0.0)
    train_z = train_z.dropna(subset=["y_fwd"])
    if train_z.empty:
        return {"daily_pnl": pd.Series(dtype=float), "coef": {}, "alpha": None,
                "oot_start": str(oot_start.date()), "oot_end": str(oot_end.date()),
                "n_rebal": 0, "n_macro_flat_days": 0, "n_traded_days": 0,
                "n_spy_bear_skips": 0}

    X_tr = train_z[feats].values
    y_tr = train_z["y_fwd"].values
    coef, intercept, alpha = _fit_ridge(X_tr, y_tr)
    if coef is None:
        return {"daily_pnl": pd.Series(dtype=float), "coef": {}, "alpha": None,
                "oot_start": str(oot_start.date()), "oot_end": str(oot_end.date()),
                "n_rebal": 0, "n_macro_flat_days": 0, "n_traded_days": 0,
                "n_spy_bear_skips": 0}

    X_oot = oot_z[feats].values
    oot_z = oot_z.copy()
    oot_z["score"] = X_oot @ coef + intercept
    oot_z["ret_raw"] = pd.to_numeric(oot["ret_1d"], errors="coerce").astype(float).values

    unique_dates = sorted(oot_z["date"].unique())
    rebal_dates = unique_dates[::hold_days]

    daily = []
    n_rebal_actual = 0
    n_macro_flat_days = 0
    n_traded_days = 0
    n_spy_bear_skips = 0
    for rd in rebal_dates:
        # STAGE 1 — SPY-MA60 bull/bear (v1 overlay)
        if regime_filter is not None:
            rg = regime_filter.get(pd.Timestamp(rd))
            if rg is None:
                prior = regime_filter.loc[:pd.Timestamp(rd)]
                rg = prior.iloc[-1] if len(prior) else "bull"
            if rg != "bull":
                n_spy_bear_skips += 1
                continue

        # STAGE 2 — MACRO REGIME GATE (v3 new)
        if macro_sig is not None:
            kf, _reason = _macro_keep_flat(macro_sig, pd.Timestamp(rd),
                                             thr_vix_chg, thr_vix_term,
                                             thr_dxy_chg, thr_disp_pct)
            if kf:
                # entire rebal cycle = cash. Record explicit zero-ret marker
                # days for the hold window so per-regime stats see them.
                hold_win_dates = [d for d in unique_dates
                                   if d > rd and d <= rd + pd.Timedelta(days=hold_days)]
                for d in hold_win_dates:
                    daily.append((d, 0.0, 0.0))
                    n_macro_flat_days += 1
                continue

        snap = oot_z[oot_z["date"] == rd].dropna(subset=["score"])
        if len(snap) < (n_long + (n_short if allow_short else 0)):
            continue
        med = snap["score"].median()
        top_score = snap["score"].max()
        if top_score <= med:
            continue

        longs = snap.nlargest(n_long, "score")["etf"].tolist()
        shorts = snap.nsmallest(n_short, "score")["etf"].tolist() if allow_short else []

        realised_vol = _estimate_book_vol(panel, rd, longs, shorts)
        if realised_vol <= 1e-6:
            gross_lev = 1.0
        else:
            gross_lev = float(np.clip(target_vol / realised_vol, lev_min, lev_max))

        hold_win = oot_z[(oot_z["date"] > rd)
                        & (oot_z["date"] <= rd + pd.Timedelta(days=hold_days))]
        for d, g in hold_win.groupby("date"):
            # STAGE 1 intra-hold — SPY-MA bear flip
            if regime_filter is not None:
                rg = regime_filter.get(pd.Timestamp(d))
                if rg is None:
                    prior = regime_filter.loc[:pd.Timestamp(d)]
                    rg = prior.iloc[-1] if len(prior) else "bull"
                if rg != "bull":
                    daily.append((d, 0.0, 0.0))
                    continue
            # STAGE 2 intra-hold — macro KEEP_FLAT mid-cycle
            if macro_sig is not None:
                kf, _reason = _macro_keep_flat(macro_sig, pd.Timestamp(d),
                                                 thr_vix_chg, thr_vix_term,
                                                 thr_dxy_chg, thr_disp_pct)
                if kf:
                    daily.append((d, 0.0, 0.0))
                    n_macro_flat_days += 1
                    continue
            lret = g[g["etf"].isin(longs)]["ret_raw"].mean() if longs else 0.0
            sret = g[g["etf"].isin(shorts)]["ret_raw"].mean() if shorts else 0.0
            if allow_short:
                book_ret = (lret if pd.notna(lret) else 0.0) \
                          - (sret if pd.notna(sret) else 0.0)
            else:
                book_ret = (lret if pd.notna(lret) else 0.0)
            day_ret = gross_lev * book_ret
            day_ret = float(np.clip(day_ret, -0.20, 0.20))
            daily.append((d, day_ret, gross_lev))
            n_traded_days += 1

        tc = (txn_cost_bps / 10000.0) * gross_lev
        daily.append((rd, -tc, gross_lev))
        n_rebal_actual += 1

    if daily:
        df_daily = pd.DataFrame(daily, columns=["date", "ret", "lev"])
        df_daily["date"] = pd.to_datetime(df_daily["date"])
        df_daily = df_daily.groupby("date", as_index=True).agg(
            ret=("ret", "sum"), lev=("lev", "max"))
        s = df_daily["ret"]
        lev_s = df_daily["lev"]
    else:
        s = pd.Series(dtype=float)
        lev_s = pd.Series(dtype=float)

    return {
        "daily_pnl": s,
        "daily_lev": lev_s,
        "coef": dict(zip(feats, [float(c) for c in coef])),
        "alpha": float(alpha) if alpha is not None else None,
        "oot_start": str(oot_start.date()),
        "oot_end": str(oot_end.date()),
        "n_rebal": n_rebal_actual,
        "n_macro_flat_days": n_macro_flat_days,
        "n_traded_days": n_traded_days,
        "n_spy_bear_skips": n_spy_bear_skips,
    }


def _iter_wf_windows(start: pd.Timestamp, end: pd.Timestamp,
                     train_months: int = 24, oot_months: int = 6,
                     step_months: int = 3) -> list[tuple]:
    out = []
    cursor = start
    while True:
        tr_start = cursor
        tr_end = tr_start + pd.DateOffset(months=train_months)
        oot_start = tr_end
        oot_end = oot_start + pd.DateOffset(months=oot_months)
        if oot_end > end + pd.Timedelta(days=1):
            break
        out.append((tr_start, tr_end, oot_start, oot_end))
        cursor = cursor + pd.DateOffset(months=step_months)
    return out


# -----------------------------------------------------------------------------
# Per-regime stratification (HC #428 R1)
# -----------------------------------------------------------------------------
def _classify_es_regime(thr_bp: float = 25.0) -> pd.Series:
    px = pd.read_parquet(SPY_PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    r = spy.pct_change()
    thr = thr_bp / 1e4
    lbl = pd.Series("flat", index=r.index, dtype=object)
    lbl[r > thr] = "green"
    lbl[r < -thr] = "red"
    return lbl.dropna()


def _per_regime_metrics(daily_pnl: pd.Series) -> dict:
    if daily_pnl.empty:
        return {}
    reg = _classify_es_regime()
    aligned = pd.DataFrame({"ret": daily_pnl})
    aligned["regime"] = reg.reindex(aligned.index).fillna("flat")
    out = {"per_regime_n": {}, "per_regime_sharpe": {},
           "per_regime_wr": {}, "per_regime_mean_ret_bps": {}}
    for r in ("green", "red", "flat"):
        sub = aligned[aligned["regime"] == r]["ret"]
        out["per_regime_n"][r] = int(len(sub))
        if len(sub) >= 5 and sub.std(ddof=1) > 0:
            sh = float(sub.mean() / sub.std(ddof=1) * np.sqrt(TRADING_DAYS))
        else:
            sh = float("nan")
        out["per_regime_sharpe"][r] = sh
        out["per_regime_wr"][r] = float((sub > 0).mean()) if len(sub) else float("nan")
        out["per_regime_mean_ret_bps"][r] = float(sub.mean() * 1e4) if len(sub) else float("nan")

    sg = out["per_regime_sharpe"].get("green", float("nan"))
    sr = out["per_regime_sharpe"].get("red", float("nan"))
    if np.isfinite(sg) and np.isfinite(sr):
        denom = max(abs(sg), abs(sr), 1e-9)
        skew = abs(sg - sr) / denom
        out["regime_skew_ratio"] = float(skew)
        out["regime_skew_pass"] = bool(skew <= 0.50)
    else:
        out["regime_skew_ratio"] = float("nan")
        out["regime_skew_pass"] = False
    return out


def _day_concentration(daily_pnl: pd.Series) -> float:
    if daily_pnl.empty:
        return float("nan")
    total = daily_pnl.abs().sum()
    if total <= 0:
        return float("nan")
    return float(daily_pnl.abs().max() / total)


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------
def run(hold_days: int, n_long: int, n_short: int, allow_short: bool,
        out_dir: Path, n_jobs: int = -1,
        target_vol: float = 0.15, lev_min: float = 0.25, lev_max: float = 2.0,
        train_months: int = 24, oot_months: int = 6, step_months: int = 3,
        txn_cost_bps: float = 5.0,
        regime_overlay: bool = True, regime_ma_days: int = 60,
        thr_vix_chg: float = 5.0, thr_vix_term: float = 1.10,
        thr_dxy_chg: float = 2.5, thr_disp_pct: float = 10.0,
        mlflow_experiment: str = "etf_rotation_v3_macro_regime_gate",
        macro_gate_on: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(mlflow_experiment)
        mlflow_run = mlflow.start_run(run_name=(
            f"v3_h{hold_days}_L{n_long}S{n_short}_vc{thr_vix_chg}_vt{thr_vix_term}"
            f"_dx{thr_dxy_chg}_dp{thr_disp_pct}"))
        mlflow.log_params({
            "hold_days": hold_days, "n_long": n_long, "n_short": n_short,
            "allow_short": allow_short, "txn_cost_bps": txn_cost_bps,
            "target_vol": target_vol, "lev_min": lev_min, "lev_max": lev_max,
            "train_months": train_months, "oot_months": oot_months,
            "step_months": step_months, "regime_overlay": regime_overlay,
            "regime_ma_days": regime_ma_days,
            "thr_vix_chg": thr_vix_chg, "thr_vix_term": thr_vix_term,
            "thr_dxy_chg": thr_dxy_chg, "thr_disp_pct": thr_disp_pct,
            "macro_gate_on": macro_gate_on,
        })
        use_mlflow = True
    except Exception as e:
        print(f"[etf_rotation_v3] MLflow unavailable: {e}")
        use_mlflow = False
        mlflow_run = None

    print(f"[etf_rotation_v3] building panel (hold={hold_days})...")
    panel, feats = build_panel(hold_days)
    print(f"[etf_rotation_v3] panel rows={len(panel)} etfs={panel['etf'].nunique()} "
          f"feats={len(feats)} date={panel['date'].min().date()}->{panel['date'].max().date()}")

    macro_sig = _load_macro_regime_signals() if macro_gate_on else None
    if macro_sig is not None:
        for col in ("vix_chg_20d", "vix_term", "dxy_chg_20d", "sector_disp_pct_252"):
            n_nan = int(macro_sig[col].isna().sum())
            n_tot = len(macro_sig)
            print(f"[etf_rotation_v3] macro_sig {col}: {n_nan}/{n_tot} NaN "
                  f"({100*n_nan/n_tot:.1f}%)")

    windows = _iter_wf_windows(panel["date"].min(), panel["date"].max(),
                               train_months=train_months,
                               oot_months=oot_months,
                               step_months=step_months)
    print(f"[etf_rotation_v3] WF={train_months}m/{oot_months}m/{step_months}m → "
          f"{len(windows)} folds")
    if not windows:
        raise RuntimeError("No WF folds — not enough history.")

    regime_filter = None
    if regime_overlay:
        regime_filter = _load_spy_regime(ma_days=regime_ma_days)

    fold_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_wf_fold)(panel, feats, ts, te, os_, oe,
                          hold_days, n_long, n_short, allow_short,
                          target_vol, lev_min, lev_max, txn_cost_bps,
                          regime_filter, macro_sig,
                          thr_vix_chg, thr_vix_term, thr_dxy_chg, thr_disp_pct)
        for (ts, te, os_, oe) in windows
    )

    all_pnl = pd.concat([r["daily_pnl"] for r in fold_results if not r["daily_pnl"].empty])
    all_pnl = all_pnl.sort_index()
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="last")]

    pooled = _metrics(all_pnl) if not all_pnl.empty else {}
    per_fold_metrics = []
    for r in fold_results:
        m = _metrics(r["daily_pnl"]) if not r["daily_pnl"].empty else {}
        per_fold_metrics.append({
            "oot_start": r["oot_start"], "oot_end": r["oot_end"],
            "alpha": r["alpha"], "n_rebal": r["n_rebal"],
            "n_macro_flat_days": r["n_macro_flat_days"],
            "n_traded_days": r["n_traded_days"],
            "n_spy_bear_skips": r["n_spy_bear_skips"],
            **m,
        })

    total_rebal = sum(r["n_rebal"] for r in fold_results)
    total_macro_flat = sum(r["n_macro_flat_days"] for r in fold_results)
    total_traded = sum(r["n_traded_days"] for r in fold_results)
    n_legs = n_long + (n_short if allow_short else 0)
    if not all_pnl.empty:
        n_years = max(len(all_pnl) / TRADING_DAYS, 1e-6)
        annual_turnover = (total_rebal * 2 * n_legs) / n_years
    else:
        annual_turnover = 0.0

    coef_rows = []
    for r in fold_results:
        if not r["coef"]:
            continue
        row = {"oot_start": r["oot_start"], "oot_end": r["oot_end"],
               "alpha": r["alpha"]}
        row.update(r["coef"])
        coef_rows.append(row)
    coef_df = pd.DataFrame(coef_rows)

    all_lev = pd.concat([r.get("daily_lev", pd.Series(dtype=float))
                         for r in fold_results if not r["daily_pnl"].empty])
    if not all_lev.empty:
        all_lev = all_lev.sort_index()
        all_lev = all_lev[~all_lev.index.duplicated(keep="last")]
        all_lev = all_lev.reindex(all_pnl.index).ffill()
    book = pd.DataFrame({"date": all_pnl.index,
                         "daily_ret": all_pnl.values,
                         "gross_lev": all_lev.values if not all_lev.empty else np.nan})
    book.to_parquet(out_dir / "book.parquet", index=False)
    if not coef_df.empty:
        coef_df.to_parquet(out_dir / "coefficients.parquet", index=False)

    regime_stats = _per_regime_metrics(all_pnl)
    day_conc = _day_concentration(all_pnl)
    day_conc_pass = bool(np.isnan(day_conc) or day_conc <= 0.70)

    fold_calmars = [m.get("calmar", float("nan")) for m in per_fold_metrics
                    if "calmar" in m and m["calmar"] is not None
                    and np.isfinite(m.get("calmar", float("nan")))]
    fold_maxdds = [m.get("max_dd", float("nan")) for m in per_fold_metrics
                   if "max_dd" in m and m["max_dd"] is not None
                   and np.isfinite(m.get("max_dd", float("nan")))]
    fold_sharpes = [m.get("sharpe", float("nan")) for m in per_fold_metrics
                    if "sharpe" in m and m["sharpe"] is not None
                    and np.isfinite(m.get("sharpe", float("nan")))]

    median_calmar = float(np.median(fold_calmars)) if fold_calmars else float("nan")
    mean_calmar = float(np.mean(fold_calmars)) if fold_calmars else float("nan")
    worst_maxdd = float(np.min(fold_maxdds)) if fold_maxdds else float("nan")
    median_sharpe = float(np.median(fold_sharpes)) if fold_sharpes else float("nan")
    n_folds_eval = len(fold_calmars)
    n_passing_folds = int(sum(1 for c in fold_calmars if c >= 1.0)) if fold_calmars else 0
    frac_passing = n_passing_folds / n_folds_eval if n_folds_eval > 0 else 0.0

    gate_median_calmar = median_calmar >= 1.0 if np.isfinite(median_calmar) else False
    gate_worst_maxdd = worst_maxdd > -0.25 if np.isfinite(worst_maxdd) else False
    gate_majority_folds = frac_passing >= 0.5
    deploy = gate_median_calmar and gate_worst_maxdd and gate_majority_folds

    metrics = {
        "version": "v3_macro_regime_gate",
        "config": {
            "hold_days": hold_days, "n_long": n_long, "n_short": n_short,
            "allow_short": allow_short, "txn_cost_bps": txn_cost_bps,
            "target_vol": target_vol, "lev_min": lev_min, "lev_max": lev_max,
            "train_months": train_months, "oot_months": oot_months,
            "step_months": step_months,
            "regime_overlay": regime_overlay,
            "macro_gate_on": macro_gate_on,
            "thresholds": {
                "thr_vix_chg": thr_vix_chg, "thr_vix_term": thr_vix_term,
                "thr_dxy_chg": thr_dxy_chg, "thr_disp_pct": thr_disp_pct,
            },
            "universe": SECTOR_ETFS,
            "features": feats,
            "n_folds": len(fold_results),
        },
        "pooled": pooled,
        "per_fold": per_fold_metrics,
        "annual_turnover_est": annual_turnover,
        "regime_stratification_hc428r1": regime_stats,
        "day_concentration_hc344": {
            "value": day_conc, "cap": 0.70, "PASS": day_conc_pass,
        },
        "gating_counts": {
            "total_rebal": int(total_rebal),
            "total_macro_flat_days": int(total_macro_flat),
            "total_traded_days": int(total_traded),
        },
        "deploy_gate": {
            "median_calmar": median_calmar,
            "mean_calmar": mean_calmar,
            "median_sharpe": median_sharpe,
            "worst_fold_maxdd": worst_maxdd,
            "n_folds": n_folds_eval,
            "n_passing_folds": n_passing_folds,
            "frac_passing": frac_passing,
            "gate_median_calmar_ge_1": gate_median_calmar,
            "gate_worst_maxdd_gt_neg25": gate_worst_maxdd,
            "gate_majority_folds": gate_majority_folds,
            "PASS": deploy,
        },
        "date_range": [str(all_pnl.index.min().date()) if not all_pnl.empty else None,
                        str(all_pnl.index.max().date()) if not all_pnl.empty else None],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    if use_mlflow and mlflow_run is not None:
        try:
            if pooled:
                for k, v in pooled.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        mlflow.log_metric(f"pooled_{k}", float(v))
            mlflow.log_metric("median_calmar", median_calmar if np.isfinite(median_calmar) else 0.0)
            mlflow.log_metric("median_sharpe", median_sharpe if np.isfinite(median_sharpe) else 0.0)
            mlflow.log_metric("worst_fold_maxdd", worst_maxdd if np.isfinite(worst_maxdd) else 0.0)
            mlflow.log_metric("frac_folds_passing", frac_passing)
            mlflow.log_metric("annual_turnover", annual_turnover)
            mlflow.log_metric("day_conc", day_conc if np.isfinite(day_conc) else 0.0)
            mlflow.log_metric("total_macro_flat_days", int(total_macro_flat))
            mlflow.log_metric("total_traded_days", int(total_traded))
            if regime_stats:
                for r, sh in regime_stats.get("per_regime_sharpe", {}).items():
                    if np.isfinite(sh):
                        mlflow.log_metric(f"sharpe_{r}", float(sh))
                rsr = regime_stats.get("regime_skew_ratio")
                if rsr is not None and np.isfinite(rsr):
                    mlflow.log_metric("regime_skew_ratio", float(rsr))
            mlflow.log_artifact(str(out_dir / "metrics.json"))
            mlflow.log_artifact(str(out_dir / "book.parquet"))
            if (out_dir / "coefficients.parquet").exists():
                mlflow.log_artifact(str(out_dir / "coefficients.parquet"))
            mlflow.end_run()
        except Exception as e:
            print(f"[etf_rotation_v3] MLflow logging failed: {e}")

    print(f"[etf_rotation_v3] wrote {out_dir}")
    print(f"[etf_rotation_v3] pooled Sharpe={pooled.get('sharpe', float('nan')):.2f} "
          f"Sortino={pooled.get('sortino', float('nan')):.2f} "
          f"Calmar={pooled.get('calmar', float('nan')):.2f} "
          f"MaxDD={pooled.get('max_dd', float('nan'))*100:.1f}%")
    print(f"[etf_rotation_v3] gating: traded={total_traded} macro_flat={total_macro_flat}")
    print(f"[etf_rotation_v3] REGIME-SKEW: ratio="
          f"{regime_stats.get('regime_skew_ratio', float('nan')):.3f} "
          f"(PASS={regime_stats.get('regime_skew_pass')})")
    return metrics


# -----------------------------------------------------------------------------
# Screening loop — vary each threshold one at a time around starting point,
# keep best per axis, then combine.
# -----------------------------------------------------------------------------
def _summary(metrics: dict) -> dict:
    p = metrics.get("pooled", {})
    rs = metrics.get("regime_stratification_hc428r1", {}) or {}
    return {
        "sharpe": p.get("sharpe", float("nan")),
        "sortino": p.get("sortino", float("nan")),
        "calmar": p.get("calmar", float("nan")),
        "cagr": p.get("cagr", float("nan")),
        "max_dd": p.get("max_dd", float("nan")),
        "pf": p.get("pf", float("nan")),
        "wr": p.get("wr", float("nan")),
        "day_conc": metrics.get("day_concentration_hc344", {}).get("value", float("nan")),
        "regime_skew_ratio": rs.get("regime_skew_ratio", float("nan")),
        "regime_skew_pass": rs.get("regime_skew_pass", False),
        "sharpe_green": rs.get("per_regime_sharpe", {}).get("green", float("nan")),
        "sharpe_red": rs.get("per_regime_sharpe", {}).get("red", float("nan")),
        "sharpe_flat": rs.get("per_regime_sharpe", {}).get("flat", float("nan")),
        "n_traded_days": metrics.get("gating_counts", {}).get("total_traded_days", 0),
        "n_macro_flat_days": metrics.get("gating_counts", {}).get("total_macro_flat_days", 0),
    }


def screen(hold_days: int, n_long: int, n_short: int, allow_short: bool,
            base_out_dir: Path, n_jobs: int) -> dict:
    """Axis-by-axis screening (NOT full grid). 12 configs total (3 per axis × 4
    axes — keeping the other 3 axes at their middle starting value)."""
    axes = {
        "thr_vix_chg":  [3.0, 5.0, 7.0],     # middle = 5.0
        "thr_vix_term": [1.05, 1.10, 1.15],   # middle = 1.10
        "thr_dxy_chg":  [1.5, 2.5, 3.5],     # middle = 2.5
        "thr_disp_pct": [5.0, 10.0, 15.0],   # middle = 10.0
    }
    middle = {k: v[1] for k, v in axes.items()}

    all_results = []
    for axis_name, values in axes.items():
        for val in values:
            cfg = dict(middle)
            cfg[axis_name] = val
            run_tag = f"{axis_name}={val}"
            print(f"\n=== SCREEN {run_tag} :: cfg={cfg} ===")
            sub_out = base_out_dir / f"screen_{axis_name}_{val}"
            m = run(hold_days=hold_days, n_long=n_long, n_short=n_short,
                    allow_short=allow_short, out_dir=sub_out, n_jobs=n_jobs,
                    regime_overlay=True, regime_ma_days=60,
                    thr_vix_chg=cfg["thr_vix_chg"],
                    thr_vix_term=cfg["thr_vix_term"],
                    thr_dxy_chg=cfg["thr_dxy_chg"],
                    thr_disp_pct=cfg["thr_disp_pct"],
                    mlflow_experiment="etf_rotation_v3_macro_regime_gate",
                    macro_gate_on=True)
            s = _summary(m)
            s.update({"axis": axis_name, "value": val, "cfg": cfg})
            all_results.append(s)

    # Pick best per axis: prefer regime_skew_pass=True; tiebreak by Sharpe.
    best_per_axis = {}
    for axis_name in axes:
        sub = [r for r in all_results if r["axis"] == axis_name]
        passing = [r for r in sub if r["regime_skew_pass"]]
        pool = passing if passing else sub
        best = max(pool, key=lambda r: (r["regime_skew_pass"],
                                            r["sharpe"] if np.isfinite(r["sharpe"]) else -99))
        best_per_axis[axis_name] = best["value"]

    # Combined config
    combined_cfg = best_per_axis
    print(f"\n=== SCREEN combined {combined_cfg} ===")
    combined_out = base_out_dir / "combined_best"
    m_combined = run(hold_days=hold_days, n_long=n_long, n_short=n_short,
                      allow_short=allow_short, out_dir=combined_out, n_jobs=n_jobs,
                      regime_overlay=True, regime_ma_days=60,
                      thr_vix_chg=combined_cfg["thr_vix_chg"],
                      thr_vix_term=combined_cfg["thr_vix_term"],
                      thr_dxy_chg=combined_cfg["thr_dxy_chg"],
                      thr_disp_pct=combined_cfg["thr_disp_pct"],
                      mlflow_experiment="etf_rotation_v3_macro_regime_gate",
                      macro_gate_on=True)
    combined_summary = _summary(m_combined)
    combined_summary.update({"axis": "combined", "value": None, "cfg": combined_cfg})
    all_results.append(combined_summary)

    # Also run a BASELINE: macro_gate_on=False, regime_overlay=True (== v1 with overlay)
    print(f"\n=== SCREEN baseline (v1-overlay, macro_gate_OFF) ===")
    base_out = base_out_dir / "baseline_v1_overlay"
    m_base = run(hold_days=hold_days, n_long=n_long, n_short=n_short,
                  allow_short=allow_short, out_dir=base_out, n_jobs=n_jobs,
                  regime_overlay=True, regime_ma_days=60,
                  mlflow_experiment="etf_rotation_v3_macro_regime_gate",
                  macro_gate_on=False)
    base_summary = _summary(m_base)
    base_summary.update({"axis": "baseline_v1_overlay", "value": None, "cfg": {}})
    all_results.append(base_summary)

    # ---- Write findings.md
    write_findings(base_out_dir, all_results, combined_summary, base_summary,
                    combined_cfg)

    return {
        "all_results": all_results,
        "combined_cfg": combined_cfg,
        "combined_summary": combined_summary,
        "baseline_summary": base_summary,
    }


def write_findings(out_dir: Path, all_results: list, combined: dict,
                    baseline: dict, combined_cfg: dict) -> None:
    md = []
    md.append("# ETF rotation v3 — macro regime gate findings")
    md.append("")
    md.append(f"Run dir: `{out_dir}`")
    md.append("")
    md.append("## v1 (overlay-only baseline) vs v3-best — side by side")
    md.append("")
    md.append("| Metric | v1 (overlay) | v3 (best macro gate) |")
    md.append("|---|---|---|")
    for k, label in [("sharpe", "Sharpe"), ("sortino", "Sortino"),
                     ("calmar", "Calmar"), ("max_dd", "MaxDD"),
                     ("cagr", "CAGR"), ("pf", "PF"), ("wr", "WR"),
                     ("day_conc", "day-conc")]:
        bv = baseline.get(k, float("nan"))
        cv = combined.get(k, float("nan"))
        if k in ("max_dd", "cagr", "wr"):
            md.append(f"| {label} | {bv*100:.2f}% | {cv*100:.2f}% |")
        else:
            md.append(f"| {label} | {bv:.3f} | {cv:.3f} |")
    md.append("")
    md.append("## Regime split (HC #428 R1)")
    md.append("")
    md.append("| | v1 (overlay) | v3 (best) |")
    md.append("|---|---|---|")
    md.append(f"| Sharpe_green | {baseline.get('sharpe_green', float('nan')):.3f} "
              f"| {combined.get('sharpe_green', float('nan')):.3f} |")
    md.append(f"| Sharpe_red   | {baseline.get('sharpe_red', float('nan')):.3f} "
              f"| {combined.get('sharpe_red', float('nan')):.3f} |")
    md.append(f"| Sharpe_flat  | {baseline.get('sharpe_flat', float('nan')):.3f} "
              f"| {combined.get('sharpe_flat', float('nan')):.3f} |")
    md.append(f"| skew ratio   | {baseline.get('regime_skew_ratio', float('nan')):.3f} "
              f"| {combined.get('regime_skew_ratio', float('nan')):.3f} |")
    md.append(f"| skew ≤ 0.50  | {baseline.get('regime_skew_pass', False)} "
              f"| {combined.get('regime_skew_pass', False)} |")
    md.append("")
    md.append("## v3 gating accounting")
    md.append("")
    md.append(f"- macro-flat days (sat in cash): {combined.get('n_macro_flat_days', 0)}")
    md.append(f"- traded days: {combined.get('n_traded_days', 0)}")
    md.append("")
    md.append("## Best thresholds")
    md.append("")
    for k, v in combined_cfg.items():
        md.append(f"- `{k}` = {v}")
    md.append("")
    md.append("## Screening table (all configs)")
    md.append("")
    md.append("| axis | val | Sharpe | Sortino | Calmar | MaxDD | "
              "Sharpe_g | Sharpe_r | skew | skew_ok |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in all_results:
        md.append(
            f"| {r['axis']} | {r['value']} | "
            f"{r['sharpe']:.2f} | {r['sortino']:.2f} | {r['calmar']:.2f} | "
            f"{r['max_dd']*100:.1f}% | "
            f"{r['sharpe_green']:.2f} | {r['sharpe_red']:.2f} | "
            f"{r['regime_skew_ratio']:.2f} | {r['regime_skew_pass']} |")
    md.append("")
    md.append("## Verdict")
    md.append("")
    # ACCEPT iff: Sharpe ≥ 1.5 AND regime_skew_pass=True AND MaxDD < 25%
    sharpe_ok = np.isfinite(combined.get("sharpe", float("nan"))) \
                  and combined["sharpe"] >= 1.5
    skew_ok = bool(combined.get("regime_skew_pass", False))
    dd_ok = combined.get("max_dd", -1.0) > -0.25
    if sharpe_ok and skew_ok and dd_ok:
        verdict = "ACCEPT"
    elif (sharpe_ok and dd_ok) or (skew_ok and np.isfinite(combined.get("sharpe", float("nan"))) and combined["sharpe"] >= 1.0):
        verdict = "MARGINAL"
    else:
        verdict = "REJECT"
    md.append(f"**{verdict}**")
    md.append("")
    md.append(f"- Sharpe ≥ 1.5: {sharpe_ok}  (got {combined.get('sharpe', float('nan')):.2f})")
    md.append(f"- regime_skew ≤ 0.50: {skew_ok}  (got {combined.get('regime_skew_ratio', float('nan')):.3f})")
    md.append(f"- MaxDD > -25%: {dd_ok}  (got {combined.get('max_dd', float('nan'))*100:.1f}%)")
    md.append("")
    if verdict == "ACCEPT":
        md.append("## Live-paper wiring plan")
        md.append("")
        md.append("1. Wire macro gate into `live_paper_engine` adjacent to existing "
                  "SPY-MA60 overlay check. Same as offline harness: STAGE 1 SPY-MA, "
                  "STAGE 2 macro KEEP_FLAT.")
        md.append("2. Macro signal pipeline: pull VIX, VIX3M, DXY, sector dispersion "
                  "from `feature_store/` nightly at 16:30 ET. Snapshot to a "
                  "`macro_regime_snapshot.parquet` keyed by trade date.")
        md.append("3. Live gate: at 09:30 ET rebal moment, read latest "
                  "`macro_regime_snapshot.parquet`. If ANY threshold trips, "
                  "skip rebalance and stay in cash.")
        md.append("4. Intra-hold check: re-evaluate macro gate at each EOD; if "
                  "trip mid-hold, liquidate to cash at next open.")
        md.append("5. Initial paper sizing: 25% of intended notional for 2 weeks of "
                  "live confirmation before full size.")
    (out_dir / "findings.md").write_text("\n".join(md))
    print(f"[etf_rotation_v3] findings → {out_dir / 'findings.md'}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETF rotation v3 — macro regime gate")
    p.add_argument("--hold-days", type=int, default=21)
    p.add_argument("--n-long", type=int, default=2)
    p.add_argument("--n-short", type=int, default=2)
    p.add_argument("--no-short", action="store_true")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--target-vol", type=float, default=0.15)
    p.add_argument("--lev-min", type=float, default=0.25)
    p.add_argument("--lev-max", type=float, default=2.0)
    p.add_argument("--train-months", type=int, default=24)
    p.add_argument("--oot-months", type=int, default=6)
    p.add_argument("--step-months", type=int, default=3)
    p.add_argument("--txn-cost-bps", type=float, default=5.0)
    p.add_argument("--regime-overlay", action="store_true",
                   help="SPY-MA bull/bear overlay (default ON for v3)")
    p.add_argument("--regime-ma-days", type=int, default=60)
    # Macro gate thresholds
    p.add_argument("--thr-vix-chg", type=float, default=5.0)
    p.add_argument("--thr-vix-term", type=float, default=1.10)
    p.add_argument("--thr-dxy-chg", type=float, default=2.5)
    p.add_argument("--thr-disp-pct", type=float, default=10.0)
    # Modes
    p.add_argument("--single", action="store_true",
                   help="Run ONE config with current thresholds (no sweep).")
    p.add_argument("--screen", action="store_true",
                   help="Axis-by-axis screening loop (default).")
    p.add_argument("--macro-gate-off", action="store_true",
                   help="Disable v3 macro gate (== v1 baseline w/ overlay).")
    p.add_argument("--mlflow-experiment", type=str,
                   default="etf_rotation_v3_macro_regime_gate")
    return p.parse_args()


def main():
    args = _parse_args()
    allow_short = not args.no_short
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / f"output/macro_picker/etf_rotation_v3_{ts}"
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.single:
        run(
            hold_days=args.hold_days,
            n_long=args.n_long,
            n_short=args.n_short if allow_short else 0,
            allow_short=allow_short,
            out_dir=out_dir,
            n_jobs=args.n_jobs,
            target_vol=args.target_vol,
            lev_min=args.lev_min,
            lev_max=args.lev_max,
            train_months=args.train_months,
            oot_months=args.oot_months,
            step_months=args.step_months,
            txn_cost_bps=args.txn_cost_bps,
            regime_overlay=True,  # always on for v3
            regime_ma_days=args.regime_ma_days,
            thr_vix_chg=args.thr_vix_chg,
            thr_vix_term=args.thr_vix_term,
            thr_dxy_chg=args.thr_dxy_chg,
            thr_disp_pct=args.thr_disp_pct,
            mlflow_experiment=args.mlflow_experiment,
            macro_gate_on=(not args.macro_gate_off),
        )
    else:
        # default = screen
        screen(hold_days=args.hold_days, n_long=args.n_long,
                n_short=args.n_short if allow_short else 0,
                allow_short=allow_short,
                base_out_dir=out_dir, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
