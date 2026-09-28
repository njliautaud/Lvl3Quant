"""
Master panel builder (HC #563 R5).

Joins every available feature family onto the (ticker, date) anchor from
prices_v2.parquet, producing a single wide parquet that downstream consumers
(GA sector fit, wheel name-selector, ranking models) read instead of opening
ten separate parquets.

OUTPUT
    data/feature_store/master_panel/daily.parquet

JOIN STRATEGY
    anchor          : prices_v2 (ticker, date) — 247 tickers, ~3000 daily rows each
    realized_vol    : merge on (ticker, date)                         [direct]
    intraday        : merge on (ticker, date)                         [direct]
    analyst_revs    : as-of forward fill per ticker (period_end <= date) [as-of]
    cross_asset     : pivot wide on `asset`, merge on date              [broadcast]
    regime dial     : merge on date                                     [broadcast]
    sector_rotation : ticker -> sector -> SPDR ETF, merge on (etf,date) [via map]
    form4 insider   : merge on (ticker, date); NaN -> 0 means no insider activity
    13F flow_per_cusip : SKIPPED — would need cusip->ticker map (TODO)
    stocktwits      : SKIPPED — only 17 daily rows for 10 megacaps (too thin v1)
    gdelt news      : will be merged once the background ingest finishes

PIT-safety: every merge uses date <= D semantics. as-of joins clip to the
right side's most recent row with period_end <= date so we never peek ahead.

Schema sketch (final wide table — columns vary by family availability):
    ticker, date,
    # price block (from prices_v2)
    open, high, low, close, volume, ret, log_ret, rv_20, rv_60, rv_252,
    # realized_vol block
    rv_cc_5d, rv_cc_20d, rv_cc_60d, rv_cc_252d,
    rv_pk_20d, rv_yz_20d, rv_yz_60d,
    # intraday block
    overnight_gap, intraday_range_pct, open_to_close_ret,
    close_minus_typprice, upper_shadow_pct, lower_shadow_pct,
    max_intraday_dd_pct, dollar_volume,
    # analyst revisions
    ar_strong_buy, ar_buy, ar_hold, ar_sell, ar_strong_sell,
    ar_net_score, ar_net_score_delta_qoq,
    # cross-asset (wide)
    xa_GOLD_close, xa_GOLD_ret_20d, xa_GOLD_zscore_60d, ... (per asset),
    # regime
    regime_state, risk_dial, severity,
    # sector rotation
    sr_etf, sr_rel_strength_spy, sr_momentum_cross_20_60,
    sr_rs_rank_among_sectors, sr_lead_lag_score_5d,
    # insider activity
    ins_net_share_change, ins_n_buys, ins_n_sells, ...
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
FS = ROOT / "data/feature_store"
OUT_DIR = FS / "master_panel"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "daily.parquet"

# ---------------------------------------------------------------------------
# Yahoo-Finance sector label -> SPDR sector ETF
# Universe_v2 uses Yahoo's broad sector vocabulary. Map to the 11 SPDRs.
# ---------------------------------------------------------------------------
SECTOR_TO_ETF = {
    "Technology":              "XLK",
    "Financial Services":      "XLF",
    "Financials":              "XLF",
    "Energy":                  "XLE",
    "Healthcare":              "XLV",
    "Health Care":             "XLV",
    "Consumer Cyclical":       "XLY",
    "Consumer Discretionary":  "XLY",
    "Consumer Defensive":      "XLP",
    "Consumer Staples":        "XLP",
    "Industrials":             "XLI",
    "Basic Materials":         "XLB",
    "Materials":               "XLB",
    "Utilities":               "XLU",
    "Real Estate":             "XLRE",
    "Communication Services":  "XLC",
    "Communications":          "XLC",
}


def _print_stage(name: str, df: pd.DataFrame):
    n_t = df["ticker"].nunique() if "ticker" in df.columns else 0
    print(f"  after {name:25s} shape={df.shape}  tickers={n_t}")


def _safe_read(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  SKIP {label}: {path.name} not on disk")
        return None
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    print(f"  read {label:25s} rows={len(df):>9,} cols={len(df.columns):2d}")
    return df


def _asof_per_ticker(left: pd.DataFrame, right: pd.DataFrame,
                     right_key: str, by: str = "ticker") -> pd.DataFrame:
    """As-of merge keyed by (ticker, date) where right is at (ticker, right_key).
       Forward-fills right's most recent row with right_key <= date, per ticker.
       Implemented via explicit groupby/merge to sidestep merge_asof dtype edge
       cases (timezone, sort detection, etc.)."""
    R = right.dropna(subset=[right_key]).copy()
    R[right_key] = pd.to_datetime(R[right_key])
    if getattr(R[right_key].dt, "tz", None) is not None:
        R[right_key] = R[right_key].dt.tz_localize(None)
    R = R.rename(columns={right_key: "date"})
    value_cols = [c for c in R.columns if c not in (by, "date")]

    # Per-ticker: concat sparse R rows into L's date axis, sort, ffill.
    # This handles period_end falling on non-trading days (no exact-date match).
    pieces = []
    L = left.sort_values([by, "date"]).reset_index(drop=True)
    for tk, sub_L in L.groupby(by, sort=False):
        sub_R = R[R[by] == tk][["date"] + value_cols].copy()
        if sub_R.empty:
            sub_out = sub_L.copy()
            for c in value_cols:
                sub_out[c] = np.nan
            pieces.append(sub_out)
            continue
        # add value_cols to L as NaN; mark rows L vs R; concat & ffill.
        sub_L = sub_L.copy()
        for c in value_cols:
            sub_L[c] = np.nan
        sub_L["_src"] = "L"
        sub_R["_src"] = "R"
        # ensure R has all L columns
        for c in sub_L.columns:
            if c not in sub_R.columns:
                sub_R[c] = np.nan
        merged = pd.concat([sub_L, sub_R[sub_L.columns.tolist()]], ignore_index=True)
        merged = merged.sort_values("date", kind="mergesort").reset_index(drop=True)
        merged[value_cols] = merged[value_cols].ffill()
        merged = merged[merged["_src"] == "L"].drop(columns=["_src"]).reset_index(drop=True)
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def main():
    print("[master_panel] building (ticker, date) panel")

    # ----- anchor -----
    anchor = _safe_read(ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet", "anchor prices_v2")
    if anchor is None:
        raise FileNotFoundError("anchor prices_v2 missing")
    panel = anchor.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    _print_stage("anchor", panel)

    # ----- universe sector tag -----
    univ = _safe_read(ROOT / "wheel_strategy_v1/data/cache/universe_v2.parquet", "universe_v2")
    if univ is not None:
        univ = univ[["ticker", "sector"]].copy()
        univ["sr_etf"] = univ["sector"].map(SECTOR_TO_ETF)
        panel = panel.merge(univ, on="ticker", how="left")
        _print_stage("universe sector tag", panel)

    # ----- realized_vol -----
    rv = _safe_read(FS / "realized_vol/daily.parquet", "realized_vol")
    if rv is not None:
        rv = rv.drop(columns=[c for c in ["ret_d"] if c in rv.columns])
        panel = panel.merge(rv, on=["ticker", "date"], how="left")
        _print_stage("realized_vol", panel)

    # ----- intraday -----
    intra = _safe_read(FS / "intraday/daily_proxies.parquet", "intraday")
    if intra is not None:
        # drop the NaN-only placeholder columns to keep panel slim
        drop_cols = ["rv_5m_intraday", "dollar_volume_first30m", "dollar_volume_last30m"]
        intra = intra.drop(columns=[c for c in drop_cols if c in intra.columns])
        panel = panel.merge(intra, on=["ticker", "date"], how="left")
        _print_stage("intraday", panel)

    # ----- analyst_revisions (as-of per ticker) -----
    ar = _safe_read(FS / "analyst_revisions/daily.parquet", "analyst_revisions")
    if ar is not None and not ar.empty:
        ar = ar.rename(columns={
            "strong_buy": "ar_strong_buy",
            "buy": "ar_buy",
            "hold": "ar_hold",
            "sell": "ar_sell",
            "strong_sell": "ar_strong_sell",
            "net_score": "ar_net_score",
            "net_score_delta_qoq": "ar_net_score_delta_qoq",
        })
        # only tickers that appear in panel
        ar = ar[ar["ticker"].isin(panel["ticker"].unique())].copy()
        if not ar.empty:
            panel = _asof_per_ticker(panel, ar, right_key="period_end", by="ticker")
        _print_stage("analyst_revisions (as-of)", panel)

    # ----- cross_asset (pivot wide, broadcast on date) -----
    xa = _safe_read(FS / "cross_asset/daily.parquet", "cross_asset")
    if xa is not None:
        cols_of_interest = ["close", "ret_5d", "ret_20d", "ret_60d", "zscore_60d"]
        cols_of_interest = [c for c in cols_of_interest if c in xa.columns]
        xa_w = xa.pivot_table(index="date", columns="asset", values=cols_of_interest)
        xa_w.columns = [f"xa_{a}_{c}" for c, a in xa_w.columns]
        xa_w = xa_w.reset_index()
        panel = panel.merge(xa_w, on="date", how="left")
        _print_stage("cross_asset", panel)

    # ----- regime dial -----
    rg = _safe_read(FS / "macro_regime/dial_daily.parquet", "regime dial")
    if rg is not None:
        rg_keep = [c for c in ["date", "state", "risk_dial", "severity"] if c in rg.columns]
        rg2 = rg[rg_keep].rename(columns={"state": "regime_state"})
        panel = panel.merge(rg2, on="date", how="left")
        # forward-fill within tickers for the few NaT days at start
        panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
        panel[["regime_state", "risk_dial", "severity"]] = (
            panel.groupby("ticker")[["regime_state", "risk_dial", "severity"]].ffill()
        )
        _print_stage("regime dial", panel)

    # ----- sector_rotation (merge on etf + date) -----
    sr = _safe_read(FS / "sector_rotation/daily.parquet", "sector_rotation")
    if sr is not None and "sr_etf" in panel.columns:
        sr = sr.rename(columns={
            "etf": "sr_etf",
            "rel_strength_spy": "sr_rel_strength_spy",
            "momentum_cross_20_60": "sr_momentum_cross_20_60",
            "rs_rank_among_sectors": "sr_rs_rank_among_sectors",
            "lead_lag_score_5d": "sr_lead_lag_score_5d",
        })
        panel = panel.merge(sr, on=["sr_etf", "date"], how="left")
        _print_stage("sector_rotation", panel)

    # ----- form4 insider activity -----
    f4 = _safe_read(FS / "edgar_form4/insider_daily.parquet", "edgar_form4")
    if f4 is not None and not f4.empty:
        rename = {c: f"ins_{c}" for c in f4.columns if c not in ("ticker", "date")}
        f4r = f4.rename(columns=rename)
        panel = panel.merge(f4r, on=["ticker", "date"], how="left")
        # no insider activity on a given day = 0, not NaN
        for c in [c for c in panel.columns if c.startswith("ins_")]:
            if pd.api.types.is_numeric_dtype(panel[c]):
                panel[c] = panel[c].fillna(0)
        _print_stage("form4 insider", panel)

    # ----- google_trends (weekly Sunday series → as-of forward fill) -----
    gt = _safe_read(FS / "google_trends/_all.parquet", "google_trends")
    if gt is not None and not gt.empty:
        # Source is weekly on Sundays; daily trading panel needs as-of forward fill.
        keep = [c for c in gt.columns if c not in ("ticker", "date")]
        gt_use = gt[["ticker", "date"] + keep].rename(columns={c: f"gt_{c}" for c in keep})
        before_cols = panel.shape[1]
        panel = _asof_per_ticker(panel, gt_use, right_key="date", by="ticker")
        print(f"    +google_trends added {panel.shape[1] - before_cols} cols")
        _print_stage("google_trends", panel)

    # ----- social_hype_reddit (as-of by available_from, PIT-safe) -----
    rh = _safe_read(FS / "social_hype_reddit/_all.parquet", "reddit_hype")
    if rh is not None and not rh.empty and "available_from" in rh.columns:
        # Drop top_subreddit string col (non-numeric); keep numeric features only.
        keep = [c for c in rh.columns if c not in
                ("ticker", "date", "available_from", "top_subreddit")]
        rh_use = rh[["ticker", "available_from"] + keep].copy()
        rh_use = rh_use.rename(columns={c: f"rh_{c}" for c in keep})
        before_cols = panel.shape[1]
        panel = _asof_per_ticker(panel, rh_use, right_key="available_from", by="ticker")
        # treat absence of activity as 0 mentions/upvotes (not NaN)
        for c in [c for c in panel.columns if c.startswith("rh_")]:
            if pd.api.types.is_numeric_dtype(panel[c]):
                panel[c] = panel[c].fillna(0)
        print(f"    +reddit_hype added {panel.shape[1] - before_cols} cols")
        _print_stage("reddit_hype", panel)

    # ----- final sort/order -----
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"\n[master_panel] final shape: {panel.shape}")
    print(f"[master_panel] tickers: {panel['ticker'].nunique()}")
    print(f"[master_panel] date range: {panel['date'].min()} -> {panel['date'].max()}")

    # null-rate per column (informational)
    print("\n[master_panel] null-rate by column (>5%):")
    null_rate = panel.isna().mean().sort_values(ascending=False)
    for c, r in null_rate.items():
        if r > 0.05:
            print(f"  {c:35s} {r*100:5.1f}%")

    panel.to_parquet(OUT, index=False)
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"\nOK master_panel: wrote {len(panel):,} rows x {len(panel.columns)} cols "
          f"({size_mb:.1f} MB) -> {OUT}")
    return OUT


if __name__ == "__main__":
    main()
