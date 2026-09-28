"""
ETF rotation v1 — sector-SPDR cross-sectional ridge rotation.

Pivot from name-picking (v4/v5/v6 capped at Sharpe ~0.77) to trading the 11
sector SPDR ETFs themselves. Lower noise floor, same feature stack, no per-name
attribution problem.

UNIVERSE: XLK XLF XLE XLY XLP XLU XLI XLV XLB XLC XLRE  (+ SPY benchmark/cash)
TARGET: forward N-day return per ETF (N=10 default)
MODEL: one cross-sectional ridge per WF fold over (ETF panel features) -> y_fwd
PORTFOLIO: long top-K, short bot-K, equal-weight, rebalance every N days,
           5bps txn cost, "no-trade" floor (top forecast must exceed median).
WF: 36mo train / 12mo OOT / 6mo step  (reuses research/walk_forward._metrics).
DEPLOY GATE: Calmar >= 1.0.

CLI:
  python3 etf_rotation_v1.py [--hold-days 10] [--n-long 2 --n-short 2]
                             [--no-short] [--out DIR]

Outputs (in --out, default output/macro_picker/etf_rotation_<TS>/):
  metrics.json, report.md, coefficients.parquet, book.parquet
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

# 11 sector SPDRs (universe), SPY used as benchmark only.
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLU", "XLI", "XLV", "XLB", "XLC", "XLRE"]
BENCHMARK = "SPY"

# Mapping sector_etf -> the GICS sector label used in master_panel_v2 'sector' col.
# (Master panel labels stocks by sector; we groupby sector to get per-ETF fund agg.)
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

# Fundamentals to aggregate to sector mean from master_panel_v2.
FUND_FEATURES = [
    "fp_gross_margin", "fp_net_margin", "fp_ebitda_margin",
    "fp_roe", "fp_debt_to_equity", "fp_current_ratio",
    "fp_fcf_yield", "fp_rev_yoy_growth", "fp_ni_yoy_growth",
    "fp_eps_yoy_growth", "fp_fcf_yoy_growth", "fp_margin_trend_4q",
    "fp_beat_rate_4q",
]

# Sector flow features (already at ETF level in sector_etf_flows + sector_rotation).
# NOTE 2026-06-08: dropped `lead_lag_score_5d` — upstream builder shifts other
# sectors by -1 (uses t+1 returns) to compute the score. That is a forward
# look-ahead leak baked into the feature, even though same-day cross-section
# is otherwise allowed. Until the upstream is rewritten to use t-1 (causal
# lead-lag), this feature MUST NOT enter the predictive panel.
FLOW_FEATURES_RAW = [
    "ret_1d", "ret_20d", "ret_60d", "rel_strength_spy",  # from sector_etf_flows
    "momentum_cross_20_60", "rs_rank_among_sectors",  # sector_rotation (no lead_lag)
]

# Macro features (broadcast — same value across all ETFs on a date).
# Pulled from macro_extra + cross_asset.
MACRO_FEATURES = [
    "macro_ust_10y", "macro_ust_10y_chg_20d",
    "macro_yc_2s10s",
    "macro_dxy_z", "macro_dxy_ret_20d",
    "macro_oil_ret_20d",
    "macro_gold_ret_20d",
    # TODO(HC-pivot): VIX level + 20d change not in current parquets.
    #   Need to ingest CBOE VIX into feature_store/cross_asset or macro_extra
    #   before enabling 'macro_vix_level' / 'macro_vix_chg_20d'.
]


# -----------------------------------------------------------------------------
# Reused utilities (mirrors of sector_picker_v4 helpers — kept local so this
# file is self-contained, behaviour identical).
# -----------------------------------------------------------------------------
def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _xs_zscore(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Per-date cross-sectional z-score across the 11 ETFs."""
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
    """Per-ETF daily flow / rotation features for the 11 sector SPDRs + SPY."""
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
    """Aggregate master_panel_v2 fundamentals to sector mean per date.

    Done ONCE with groupby — avoids loading 682k row-per-ticker data into the
    ETF panel directly. Returns (date, sector_label, fp_*_sector_mean).
    """
    # Read only the columns we need to keep memory sane.
    cols = ["date", "sector"] + FUND_FEATURES
    mp = pd.read_parquet(MASTER_PANEL_PATH, columns=cols)
    mp["date"] = pd.to_datetime(mp["date"])
    mp = mp.dropna(subset=["sector"])
    # Drop the 'ETF' pseudo-sector (those are SPY/sector ETFs themselves).
    mp = mp[mp["sector"] != "ETF"]

    agg = mp.groupby(["date", "sector"], as_index=False)[FUND_FEATURES].mean()
    agg = agg.rename(columns={f: f"{f}_sec" for f in FUND_FEATURES})
    return agg


def _load_macro_panel() -> pd.DataFrame:
    """Macro features broadcast across all ETFs (same value per date)."""
    macro_extra = pd.read_parquet(MACRO_EXTRA_PATH)
    macro_extra["date"] = pd.to_datetime(macro_extra["date"])
    cross = pd.read_parquet(CROSS_ASSET_PATH)
    cross["date"] = pd.to_datetime(cross["date"])

    out = pd.DataFrame({"date": macro_extra["date"]})
    out["macro_ust_10y"] = macro_extra["ust_10y"].astype(float).values
    out["macro_yc_2s10s"] = macro_extra["yc_2s10s"].astype(float).values
    out = out.sort_values("date").reset_index(drop=True)
    out["macro_ust_10y_chg_20d"] = out["macro_ust_10y"].diff(20)

    # DXY, OIL, GOLD from cross_asset (already z-scored / returns available).
    def _pivot(asset: str, col: str, out_col: str) -> pd.Series:
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
    return out


def _load_spy_for_benchmark() -> pd.Series:
    """SPY daily returns indexed by date, used for benchmark / cash sub."""
    px = pd.read_parquet(SPY_PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    return spy.pct_change().dropna()


def _load_spy_regime(ma_days: int = 60) -> pd.Series:
    """SPY close vs `ma_days` simple-MA → 'bull' if above, 'bear' if below.
    Returned as a Series of {'bull','bear'} indexed by date — used as a
    real-time-tradeable regime filter (no look-ahead — uses today's close vs
    a backward-looking MA, both available at end-of-day).
    """
    px = pd.read_parquet(SPY_PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    ma = spy.rolling(ma_days, min_periods=max(20, ma_days // 2)).mean()
    out = pd.Series(np.where(spy > ma, "bull", "bear"), index=spy.index, name="spy_regime")
    return out.dropna()


def build_panel(hold_days: int) -> tuple[pd.DataFrame, list[str]]:
    """Build the per-ETF daily panel with all features + forward-N return target.

    Returns (panel_df, feature_columns_present).
    Panel columns include at minimum: etf, date, close, ret_1d, y_fwd, + features.
    """
    flow = _load_flow_panel()
    fund = _load_sector_fundamentals()
    macro = _load_macro_panel()

    # Map ETF -> sector label so we can join the sector-mean fundamentals.
    flow["sector_label"] = flow["etf"].map(ETF_TO_SECTOR_LABEL)
    # SPY won't have a sector — leave NaN, will be filled at z-score stage.
    fund_renamed = fund.rename(columns={"sector": "sector_label"})

    panel = flow.merge(fund_renamed, on=["date", "sector_label"], how="left")
    panel = panel.merge(macro, on="date", how="left")

    # forward N-day return target per ETF
    panel = panel.sort_values(["etf", "date"]).reset_index(drop=True)
    panel["close"] = panel["close"].astype(float)
    panel["y_fwd"] = (panel.groupby("etf")["close"].shift(-hold_days) / panel["close"] - 1.0)

    # Restrict to the rotation universe (drop SPY from the trading panel; we
    # use it later only for benchmarking / cash-substitute).
    trade_panel = panel[panel["etf"].isin(SECTOR_ETFS)].copy()

    # Assemble feature set: flow + sector fundamentals (renamed) + macro.
    fund_cols = [f"{f}_sec" for f in FUND_FEATURES]
    feats_all = FLOW_FEATURES_RAW + fund_cols + MACRO_FEATURES
    feats_present = [f for f in feats_all if f in trade_panel.columns]

    return trade_panel, feats_present


# -----------------------------------------------------------------------------
# Walk-forward fold engine (one fold)
# -----------------------------------------------------------------------------
def _estimate_book_vol(panel: pd.DataFrame, rd: pd.Timestamp,
                       longs: list[str], shorts: list[str],
                       lookback_days: int = 60) -> float:
    """Estimate realised annualised vol of the equal-weight long/short book
    using the last `lookback_days` trading days of daily returns BEFORE rd.

    Returns 0.0 if not enough history.
    """
    cutoff_lo = rd - pd.Timedelta(days=lookback_days * 2 + 10)  # calendar buffer
    hist = panel[(panel["date"] < rd) & (panel["date"] >= cutoff_lo)]
    hist = hist[hist["etf"].isin(set(longs) | set(shorts))]
    if hist.empty:
        return 0.0
    # Per-date book return = mean(long_ret) - mean(short_ret) when allow_short,
    # else mean(long_ret).
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
             regime_filter: pd.Series | None = None) -> dict:
    """Run one walk-forward fold; returns dict with daily pnl + coefs + dates.

    Vol-targets each rebalance: scale gross leverage so that the realised vol
    of the equal-weight long/short book over the prior 60 days matches
    `target_vol` (annualised), clipped to [lev_min, lev_max].
    """
    train = panel[(panel["date"] >= tr_start) & (panel["date"] < tr_end)].copy()
    oot = panel[(panel["date"] >= oot_start) & (panel["date"] < oot_end)].copy()

    if len(train) < 200 or len(oot) < 20:
        return {"daily_pnl": pd.Series(dtype=float), "coef": {}, "alpha": None,
                "oot_start": str(oot_start.date()), "oot_end": str(oot_end.date()),
                "n_rebal": 0}

    train_z = _xs_zscore(train, feats)
    oot_z = _xs_zscore(oot, feats)
    for f in feats:
        train_z[f] = train_z[f].fillna(0.0)
        oot_z[f] = oot_z[f].fillna(0.0)
    train_z = train_z.dropna(subset=["y_fwd"])
    if train_z.empty:
        return {"daily_pnl": pd.Series(dtype=float), "coef": {}, "alpha": None,
                "oot_start": str(oot_start.date()), "oot_end": str(oot_end.date()),
                "n_rebal": 0}

    X_tr = train_z[feats].values
    y_tr = train_z["y_fwd"].values
    coef, intercept, alpha = _fit_ridge(X_tr, y_tr)
    if coef is None:
        return {"daily_pnl": pd.Series(dtype=float), "coef": {}, "alpha": None,
                "oot_start": str(oot_start.date()), "oot_end": str(oot_end.date()),
                "n_rebal": 0}

    X_oot = oot_z[feats].values
    oot_z = oot_z.copy()
    oot_z["score"] = X_oot @ coef + intercept
    # CRITICAL: pull ret_raw from the PRE-zscored oot panel, not oot_z.
    # _xs_zscore overwrites ret_1d with the cross-sectional z-score, which
    # destroys the actual daily return needed for P&L. Earlier versions
    # P&L'd on z-scores, producing a "Sharpe 1.78 / MaxDD -98%" fantasy.
    oot_z["ret_raw"] = pd.to_numeric(oot["ret_1d"], errors="coerce").astype(float).values

    unique_dates = sorted(oot_z["date"].unique())
    rebal_dates = unique_dates[::hold_days]

    daily = []
    n_rebal_actual = 0
    for rd in rebal_dates:
        # REGIME OVERLAY (HC #428 R1 fix): if SPY-MA regime says 'bear' at
        # rebal date, sit in cash for this rebal cycle (no positions, no
        # txn cost). Bull-regime-only filter — addresses validation #3
        # finding that bear-regime Sharpe was -0.76 vs bull +3.15.
        if regime_filter is not None:
            rg = regime_filter.get(pd.Timestamp(rd))
            if rg is None:
                # asof lookback to most-recent known regime
                prior = regime_filter.loc[:pd.Timestamp(rd)]
                rg = prior.iloc[-1] if len(prior) else "bull"
            if rg != "bull":
                continue
        snap = oot_z[oot_z["date"] == rd].dropna(subset=["score"])
        if len(snap) < (n_long + (n_short if allow_short else 0)):
            continue
        # "no-trade" floor: skip when top score not above median.
        med = snap["score"].median()
        top_score = snap["score"].max()
        if top_score <= med:
            continue

        longs = snap.nlargest(n_long, "score")["etf"].tolist()
        shorts = snap.nsmallest(n_short, "score")["etf"].tolist() if allow_short else []

        # VOL-TARGET sizing: scale gross leverage so prior-60d realised vol
        # of this book matches target_vol. Clip to [lev_min, lev_max].
        realised_vol = _estimate_book_vol(panel, rd, longs, shorts)
        if realised_vol <= 1e-6:
            gross_lev = 1.0
        else:
            gross_lev = float(np.clip(target_vol / realised_vol, lev_min, lev_max))

        hold_win = oot_z[(oot_z["date"] > rd)
                        & (oot_z["date"] <= rd + pd.Timedelta(days=hold_days))]
        for d, g in hold_win.groupby("date"):
            # INTRA-HOLD REGIME GATE (fix to original rebal-only gate):
            # if SPY regime flipped to 'bear' mid-hold, sit in cash for that
            # day (book_ret = 0). Prevents the 21-day hold from bleeding
            # through bull→bear regime transitions.
            if regime_filter is not None:
                rg = regime_filter.get(pd.Timestamp(d))
                if rg is None:
                    prior = regime_filter.loc[:pd.Timestamp(d)]
                    rg = prior.iloc[-1] if len(prior) else "bull"
                if rg != "bull":
                    daily.append((d, 0.0, 0.0))
                    continue
            lret = g[g["etf"].isin(longs)]["ret_raw"].mean() if longs else 0.0
            sret = g[g["etf"].isin(shorts)]["ret_raw"].mean() if shorts else 0.0
            if allow_short:
                book_ret = (lret if pd.notna(lret) else 0.0) \
                          - (sret if pd.notna(sret) else 0.0)
            else:
                book_ret = (lret if pd.notna(lret) else 0.0)
            day_ret = gross_lev * book_ret
            # Soft outlier guard — with vol-target sizing this should almost
            # never bind. ±20% is a sanity rail for data-quality glitches.
            day_ret = float(np.clip(day_ret, -0.20, 0.20))
            daily.append((d, day_ret, gross_lev))

        # txn cost on rebalance day, scaled by gross leverage
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
    }


def _iter_wf_windows(start: pd.Timestamp, end: pd.Timestamp,
                     train_months: int = 24, oot_months: int = 6,
                     step_months: int = 3) -> list[tuple]:
    """Generator of (tr_start, tr_end, oot_start, oot_end) tuples."""
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
# Main runner
# -----------------------------------------------------------------------------
def run(hold_days: int, n_long: int, n_short: int, allow_short: bool,
        out_dir: Path, n_jobs: int = -1,
        target_vol: float = 0.15, lev_min: float = 0.25, lev_max: float = 2.0,
        train_months: int = 24, oot_months: int = 6, step_months: int = 3,
        txn_cost_bps: float = 5.0,
        regime_overlay: bool = False, regime_ma_days: int = 60) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[etf_rotation] building panel (hold={hold_days})...")
    panel, feats = build_panel(hold_days)
    print(f"[etf_rotation] panel rows={len(panel)} etfs={panel['etf'].nunique()} "
          f"feats={len(feats)} date={panel['date'].min().date()}->{panel['date'].max().date()}")

    windows = _iter_wf_windows(panel["date"].min(), panel["date"].max(),
                               train_months=train_months,
                               oot_months=oot_months,
                               step_months=step_months)
    print(f"[etf_rotation] WF={train_months}m/{oot_months}m/{step_months}m → "
          f"{len(windows)} folds, vol_target={target_vol:.2f} "
          f"lev_clip=[{lev_min:.2f},{lev_max:.2f}], n_jobs={n_jobs}")

    if not windows:
        raise RuntimeError("No WF folds — not enough history.")

    # Load regime filter once (bull/bear based on SPY vs MA), pass to each fold.
    regime_filter = None
    if regime_overlay:
        regime_filter = _load_spy_regime(ma_days=regime_ma_days)
        bull_frac = float((regime_filter == "bull").mean())
        print(f"[etf_rotation] regime overlay ON (SPY-MA{regime_ma_days}d): "
              f"bull_days_frac={bull_frac:.2f}")

    # Parallel across folds, per HC #565 R1 (11 ETFs too small to parallelise cross-section).
    fold_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_wf_fold)(panel, feats, ts, te, os_, oe,
                          hold_days, n_long, n_short, allow_short,
                          target_vol, lev_min, lev_max, txn_cost_bps,
                          regime_filter)
        for (ts, te, os_, oe) in windows
    )

    # Aggregate pooled-OOT pnl.
    all_pnl = pd.concat([r["daily_pnl"] for r in fold_results if not r["daily_pnl"].empty])
    all_pnl = all_pnl.sort_index()
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="last")]

    pooled = _metrics(all_pnl) if not all_pnl.empty else {}
    # Per-fold metrics for the per-fold table.
    per_fold_metrics = []
    for r in fold_results:
        m = _metrics(r["daily_pnl"]) if not r["daily_pnl"].empty else {}
        per_fold_metrics.append({
            "oot_start": r["oot_start"], "oot_end": r["oot_end"],
            "alpha": r["alpha"], "n_rebal": r["n_rebal"],
            **m,
        })

    # Turnover: rough proxy = txn-cost days / total rebalances * 2 * (n_long+n_short)
    total_rebal = sum(r["n_rebal"] for r in fold_results)
    n_legs = n_long + (n_short if allow_short else 0)
    # turnover_pct: fraction of book rebalanced per year on avg
    if not all_pnl.empty:
        n_years = max(len(all_pnl) / TRADING_DAYS, 1e-6)
        annual_turnover = (total_rebal * 2 * n_legs) / n_years
    else:
        annual_turnover = 0.0

    # Coefficient frame
    coef_rows = []
    for r in fold_results:
        if not r["coef"]:
            continue
        row = {"oot_start": r["oot_start"], "oot_end": r["oot_end"],
               "alpha": r["alpha"]}
        row.update(r["coef"])
        coef_rows.append(row)
    coef_df = pd.DataFrame(coef_rows)

    # Book parquet (date + pnl + leverage)
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

    # Avg coef across folds (for headline)
    avg_coef = {}
    if not coef_df.empty:
        for f in feats:
            if f in coef_df.columns:
                avg_coef[f] = float(coef_df[f].mean())
    feat_imp_sorted = sorted(avg_coef.items(), key=lambda x: -abs(x[1]))

    # === HONEST DEPLOY GATE ===
    # Median per-fold Calmar (NOT the pooled-CAGR/pooled-MaxDD artifact).
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
    n_folds = len(fold_calmars)
    n_passing_folds = int(sum(1 for c in fold_calmars if c >= 1.0)) if fold_calmars else 0
    frac_passing = n_passing_folds / n_folds if n_folds > 0 else 0.0

    gate_median_calmar = median_calmar >= 1.0 if np.isfinite(median_calmar) else False
    gate_worst_maxdd = worst_maxdd > -0.25 if np.isfinite(worst_maxdd) else False
    gate_majority_folds = frac_passing >= 0.5
    deploy = gate_median_calmar and gate_worst_maxdd and gate_majority_folds

    metrics = {
        "config": {
            "hold_days": hold_days, "n_long": n_long, "n_short": n_short,
            "allow_short": allow_short, "txn_cost_bps": txn_cost_bps,
            "target_vol": target_vol, "lev_min": lev_min, "lev_max": lev_max,
            "train_months": train_months, "oot_months": oot_months,
            "step_months": step_months,
            "universe": SECTOR_ETFS,
            "features": feats,
            "n_folds": len(fold_results),
        },
        "pooled": pooled,
        "per_fold": per_fold_metrics,
        "annual_turnover_est": annual_turnover,
        "feature_importance_avg_coef": dict(feat_imp_sorted),
        "deploy_gate": {
            "median_calmar": median_calmar,
            "mean_calmar": mean_calmar,
            "median_sharpe": median_sharpe,
            "worst_fold_maxdd": worst_maxdd,
            "n_folds": n_folds,
            "n_passing_folds": n_passing_folds,
            "frac_passing": frac_passing,
            "gate_median_calmar_ge_1": gate_median_calmar,
            "gate_worst_maxdd_gt_neg25": gate_worst_maxdd,
            "gate_majority_folds": gate_majority_folds,
            "PASS": deploy,
            "description": "Median per-fold Calmar >= 1.0 AND worst-fold MaxDD > -25% "
                          "AND >= 50% of folds individually pass Calmar 1.0",
        },
        "date_range": [str(all_pnl.index.min().date()) if not all_pnl.empty else None,
                        str(all_pnl.index.max().date()) if not all_pnl.empty else None],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # Markdown report
    md = []
    md.append(f"# ETF rotation v1 — sector SPDR cross-sectional ridge")
    md.append("")
    md.append(f"**Config**: hold={hold_days}d, long top-{n_long}, "
              f"{'short bot-' + str(n_short) if allow_short else 'long-only'}, "
              f"txn={txn_cost_bps}bps, vol_target={target_vol:.2f}, "
              f"lev_clip=[{lev_min:.2f},{lev_max:.2f}]")
    md.append("")
    md.append(f"**HONEST DEPLOY GATE**: median per-fold Calmar **{median_calmar:.2f}** "
              f"(need ≥ 1.0), worst-fold MaxDD **{worst_maxdd*100:.1f}%** (need > -25%), "
              f"folds passing Calmar≥1: **{n_passing_folds}/{n_folds}** (need ≥ 50%).")
    md.append(f"**RESULT: {'PASS' if deploy else 'FAIL'}**")
    md.append("")
    if pooled:
        md.append(f"_(For reference — pooled-PnL stats, NOT the deploy gate)_: "
                  f"Sharpe {pooled['sharpe']:.2f} | "
                  f"pooled-Calmar {pooled['calmar']:.2f} | "
                  f"CAGR {pooled['cagr']*100:.1f}% | "
                  f"pooled-MaxDD {pooled['max_dd']*100:.1f}% | "
                  f"WR {pooled['wr']*100:.1f}%")
    md.append(f"**Folds**: {len(fold_results)} | "
              f"**Annualised turnover (legs/yr)**: {annual_turnover:.1f}")
    md.append("")
    md.append("## Per-fold OOT")
    md.append("")
    md.append("| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in per_fold_metrics:
        sh = r.get("sharpe", float("nan"))
        cm = r.get("calmar", float("nan"))
        cg = r.get("cagr", float("nan"))
        dd = r.get("max_dd", float("nan"))
        wr = r.get("wr", float("nan"))
        md.append(f"| {r['oot_start']} → {r['oot_end']} | "
                  f"{sh:.2f} | {cm:.2f} | {cg*100:.1f}% | {dd*100:.1f}% | "
                  f"{wr*100:.1f}% | {r['n_rebal']} | {r['alpha']} |")
    md.append("")
    md.append("## Feature importance (avg coef across folds)")
    md.append("")
    md.append("| Feature | Avg coef |")
    md.append("|---|---|")
    for f, c in feat_imp_sorted[:15]:
        md.append(f"| `{f}` | {c:+.4f} |")
    md.append("")
    (out_dir / "report.md").write_text("\n".join(md))

    print(f"[etf_rotation] wrote {out_dir}/metrics.json, report.md, "
          f"book.parquet, coefficients.parquet")
    print(f"[etf_rotation] GATE: median_calmar={median_calmar:.2f} "
          f"worst_maxdd={worst_maxdd*100:.1f}% "
          f"passing_folds={n_passing_folds}/{n_folds} | "
          f"DEPLOY={'PASS' if deploy else 'FAIL'}")
    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETF rotation v1")
    p.add_argument("--hold-days", type=int, default=10)
    p.add_argument("--n-long", type=int, default=2)
    p.add_argument("--n-short", type=int, default=2)
    p.add_argument("--no-short", action="store_true",
                   help="Long-only variant (ignore --n-short).")
    p.add_argument("--out", type=str, default=None,
                   help="Output dir; defaults to output/macro_picker/etf_rotation_<TS>/")
    p.add_argument("--n-jobs", type=int, default=-1)
    # Vol-target sizing knobs
    p.add_argument("--target-vol", type=float, default=0.15,
                   help="Target annualised portfolio vol (default 0.15 = 15%%)")
    p.add_argument("--lev-min", type=float, default=0.25)
    p.add_argument("--lev-max", type=float, default=2.0)
    # WF window knobs
    p.add_argument("--train-months", type=int, default=24)
    p.add_argument("--oot-months", type=int, default=6)
    p.add_argument("--step-months", type=int, default=3)
    p.add_argument("--txn-cost-bps", type=float, default=5.0,
                   help="Round-trip transaction cost in bps applied at each rebalance.")
    p.add_argument("--regime-overlay", action="store_true",
                   help="Bull-only filter: skip rebalance when SPY < MA (sit cash).")
    p.add_argument("--regime-ma-days", type=int, default=60,
                   help="MA lookback for SPY regime classifier (default 60d).")
    return p.parse_args()


def main():
    args = _parse_args()
    allow_short = not args.no_short
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / f"output/macro_picker/etf_rotation_{ts}"
    else:
        out_dir = Path(args.out)
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
        regime_overlay=args.regime_overlay,
        regime_ma_days=args.regime_ma_days,
    )


if __name__ == "__main__":
    main()
