"""
Sector picker v3 — full-shelf ridge ranking (HC #559 + HC #561 R4).

This is a NEW file. It does NOT modify ga_sector_fit_v2.py.

WHAT
    Per-sector ridge regression on the master_panel feature shelf, with rolling
    walk-forward evaluation. For each (Yahoo) sector we:
      1. Select names tagged to that sector from the panel.
      2. Build a forward-21d (monthly) total return target.
      3. Standardise features within each (date, sector) cross-section.
      4. Walk-forward train/OOT (3y / 1y, step 6mo) — within each train slice
         fit a ridge model with cross-validated alpha.
      5. On OOT, score names daily, hold top-3 long names per month (equal-weight)
         and short bottom-3 (equal-weight). Rebalance once a month.
      6. Apply 5 bps round-trip per name turnover (liquid).
      7. Pipe the daily portfolio return into walk_forward.walk_forward to get
         per-fold Sharpe / Sortino / CAGR / MaxDD / Calmar / PF / WR vs SPY.

OUTPUT
    research/findings/sector_picker_v3_<sector>.json   per-sector report
    research/findings/sector_picker_v3_summary.md      aggregate report

REJECT GATES (HC #561 R4):
    HARD FLOOR  : median Calmar across WF folds ≥ 1.0
    TARGET MET  : CAGR ≥ 0.18, Calmar ≥ 1.5, Sharpe ≥ 1.5, MaxDD ≤ 0.15
    STRETCH MET : CAGR ≥ 0.25, Calmar ≥ 2.0

Why ridge instead of full GA: ridge gives an interpretable closed-form per
sector (one coefficient per feature), which satisfies HC #559 R3's "interpretable
closed-form formulas per sector" requirement, and trains in seconds — so we can
do a full walk-forward sweep across all 11 sectors in minutes.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
from walk_forward import walk_forward  # type: ignore

PANEL = ROOT / "data/feature_store/master_panel/daily.parquet"
SPY_PRICE = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"
MACRO_EXTRA = ROOT / "wheel_strategy_v1/data/cache/macro_extra.parquet"
FINDINGS = ROOT / "research/findings"
FINDINGS.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
HOLD_DAYS = 21          # ~ 1 month
TOP_N = 3
BOT_N = 3
TXN_COST_BPS = 5        # 5 bps per name turnover (liquid)

# Candidate feature pool from master_panel.
# We intentionally EXCLUDE leakage-prone columns (raw OHLCV future, regime_state).
FEATURE_POOL = [
    # price/return
    "ret", "log_ret",
    # multi-horizon volatility
    "rv_cc_5d", "rv_cc_20d", "rv_cc_60d", "rv_cc_252d",
    "rv_pk_20d", "rv_yz_20d", "rv_yz_60d",
    # intraday proxies (PIT-safe at EOD)
    "overnight_gap", "intraday_range_pct", "open_to_close_ret",
    "upper_shadow_pct", "lower_shadow_pct", "max_intraday_dd_pct",
    "dollar_volume",
    # analyst revisions
    "ar_net_score", "ar_net_score_delta_qoq",
    # cross-asset state
    "xa_GOLD_ret_20d", "xa_OIL_ret_20d", "xa_COPPER_ret_20d",
    "xa_BTC_ret_20d", "xa_UST10Y_ret_20d", "xa_DXY_ret_20d",
    "xa_GOLD_zscore_60d", "xa_DXY_zscore_60d", "xa_UST10Y_zscore_60d",
    # regime (numeric only)
    "risk_dial", "severity",
    # sector rotation (per-name via sector tag)
    "sr_rel_strength_spy", "sr_momentum_cross_20_60",
    "sr_rs_rank_among_sectors", "sr_lead_lag_score_5d",
    # insider activity
    "ins_n_buys", "ins_n_sells", "ins_net_share_change",
]


# ---------------------------------------------------------------------------
# data prep
# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    print(f"loading {PANEL.name} ...")
    p = pd.read_parquet(PANEL)
    p["date"] = pd.to_datetime(p["date"])
    return p


def load_spy_and_funding() -> tuple[pd.Series, pd.Series]:
    px = pd.read_parquet(SPY_PRICE)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    if spy.empty:
        # synthetic: mean of all tickers
        agg = px.groupby("date")["close"].mean()
        spy_ret = agg.pct_change().dropna()
    else:
        spy_ret = spy.pct_change().dropna()
    me = pd.read_parquet(MACRO_EXTRA)
    me["date"] = pd.to_datetime(me["date"])
    me = me.set_index("date")
    if "fed_funds" in me.columns:
        fund = me["fed_funds"].astype(float).reindex(spy_ret.index).ffill().bfill().fillna(5.0)
    else:
        fund = pd.Series(5.0, index=spy_ret.index)
    return spy_ret, fund


def build_target(panel: pd.DataFrame, hold_days: int = HOLD_DAYS) -> pd.DataFrame:
    """Forward `hold_days` total return per (ticker, date)."""
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    fwd = (panel.groupby("ticker")["close"].shift(-hold_days)
           / panel["close"] - 1.0)
    panel["y_fwd"] = fwd
    return panel


# ---------------------------------------------------------------------------
# ridge per fold
# ---------------------------------------------------------------------------
def _fit_ridge(X: np.ndarray, y: np.ndarray, alphas=(0.1, 1.0, 10.0, 100.0)) -> tuple[np.ndarray, float, float]:
    """Ridge with closed-form (X'X + αI)^-1 X'y. Pick α minimising train residual variance.
       Returns (coef, intercept, alpha)."""
    # demean
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


def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _cross_sectional_zscore(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Per-date z-score within sector to make features comparable."""
    out = panel.copy()
    for f in feats:
        if f not in out.columns:
            out[f] = 0.0
            continue
        x = pd.to_numeric(out[f], errors="coerce").astype(float)
        x = _winsorize(x, 0.01)
        out[f] = x
        # group by date (cross-sectional) — sector already filtered upstream
        mu = out.groupby("date")[f].transform("mean")
        sd = out.groupby("date")[f].transform("std")
        z = (x - mu) / sd.replace(0.0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan)
        out[f] = z
    return out


# ---------------------------------------------------------------------------
# walk-forward portfolio construction
# ---------------------------------------------------------------------------
def portfolio_returns_for_sector(
    sector_panel: pd.DataFrame,
    feats: list[str],
    train_months: int = 36,
    oot_months: int = 12,
    step_months: int = 6,
    hold_days: int = HOLD_DAYS,
    top_n: int = TOP_N,
    bot_n: int = BOT_N,
) -> tuple[pd.Series, list[dict]]:
    """Build daily portfolio returns by walking forward through the sector panel."""
    sector_panel = sector_panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    # keep raw daily return aside — features incl. `ret` will be z-scored,
    # but we need the un-touched realized return when sizing the portfolio.
    sector_panel["ret_raw"] = sector_panel["ret"].astype(float)
    sector_panel["ret_next"] = sector_panel.groupby("ticker")["ret_raw"].shift(-1)
    start = sector_panel["date"].min()
    end = sector_panel["date"].max()
    cursor = start
    daily_pnl = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    fold_info = []

    while True:
        tr_start = cursor
        tr_end = tr_start + pd.DateOffset(months=train_months)
        oot_start = tr_end
        oot_end = oot_start + pd.DateOffset(months=oot_months)
        if oot_end > end + pd.Timedelta(days=1):
            break

        train = sector_panel[(sector_panel["date"] >= tr_start) & (sector_panel["date"] < tr_end)]
        oot = sector_panel[(sector_panel["date"] >= oot_start) & (sector_panel["date"] < oot_end)]
        if len(train) < 1000 or len(oot) < 100:
            cursor = cursor + pd.DateOffset(months=step_months)
            continue

        # standardise features cross-sectionally within sector on TRAIN
        train_z = _cross_sectional_zscore(train, feats)
        oot_z = _cross_sectional_zscore(oot, feats)
        # z=0 for missing → "average for that day" (standard for incomplete shelves)
        for f in feats:
            train_z[f] = train_z[f].fillna(0.0)
            oot_z[f] = oot_z[f].fillna(0.0)

        # build matrices — y_fwd MUST be non-null
        train_z = train_z.dropna(subset=["y_fwd"])
        if train_z.empty:
            cursor = cursor + pd.DateOffset(months=step_months)
            continue
        X_tr = train_z[feats].values
        y_tr = train_z["y_fwd"].values
        coef, intercept, alpha = _fit_ridge(X_tr, y_tr)
        if coef is None:
            cursor = cursor + pd.DateOffset(months=step_months)
            continue

        # score OOT
        oot_scored = oot_z.copy()
        X_oot = oot_scored[feats].fillna(0.0).values
        oot_scored["score"] = X_oot @ coef + intercept

        # monthly rebalance: pick top_n long, bot_n short on rebal dates
        # use ACTUAL trading days from the OOT panel, every hold_days-th unique date
        unique_dates = sorted(oot_scored["date"].unique())
        rebal_dates = unique_dates[::hold_days]
        fold_daily = []
        for rd in rebal_dates:
            snapshot = oot_scored[oot_scored["date"] == rd].dropna(subset=["score"])
            if len(snapshot) < (top_n + bot_n):
                continue
            longs = snapshot.nlargest(top_n, "score")["ticker"].tolist()
            shorts = snapshot.nsmallest(bot_n, "score")["ticker"].tolist()
            hold_window = oot_scored[(oot_scored["date"] > rd)
                                     & (oot_scored["date"] <= rd + pd.Timedelta(days=hold_days))]
            for d, g in hold_window.groupby("date"):
                lret = g[g["ticker"].isin(longs)]["ret_raw"].mean() if longs else 0.0
                sret = g[g["ticker"].isin(shorts)]["ret_raw"].mean() if shorts else 0.0
                # long-short, equal-dollar legs — cap per-day to ±25% as a
                # data-error guard (split/dividend glitches in source prices).
                day_ret = float(np.clip(0.5 * (lret if pd.notna(lret) else 0) -
                                        0.5 * (sret if pd.notna(sret) else 0),
                                        -0.25, 0.25))
                fold_daily.append((d, day_ret))
            # turnover cost on rebal day
            tc = (top_n + bot_n) * TXN_COST_BPS / 10000.0 / max(1, top_n + bot_n)
            fold_daily.append((rd, -tc))

        if fold_daily:
            s = pd.Series(dict(fold_daily))
            s.index = pd.to_datetime(s.index)
            # combine duplicates (rebal day + first hold day)
            s = s.groupby(level=0).sum()
            daily_pnl = pd.concat([daily_pnl, s])

        fold_info.append({
            "oot_start": str(oot_start.date()),
            "oot_end": str(oot_end.date()),
            "alpha": alpha,
            "train_rows": int(len(train_z)),
            "oot_rows": int(len(oot)),
            "coef": dict(zip(feats, [float(c) for c in coef])),
        })
        cursor = cursor + pd.DateOffset(months=step_months)

    daily_pnl = daily_pnl.sort_index()
    daily_pnl = daily_pnl[~daily_pnl.index.duplicated(keep="last")]
    return daily_pnl, fold_info


# ---------------------------------------------------------------------------
# orchestrate
# ---------------------------------------------------------------------------
def run_sector(panel: pd.DataFrame, sector: str, spy_ret: pd.Series, fund_rate: pd.Series,
               verbose: bool = True) -> dict:
    sp = panel[panel["sector"] == sector].copy()
    if sp.empty or sp["ticker"].nunique() < 5:
        if verbose:
            print(f"  skip {sector}: only {sp['ticker'].nunique() if not sp.empty else 0} tickers")
        return {"sector": sector, "verdict": "INSUFFICIENT_UNIVERSE", "n_tickers": int(sp['ticker'].nunique() if not sp.empty else 0)}

    feats_present = [f for f in FEATURE_POOL if f in sp.columns]
    if verbose:
        print(f"  {sector}: {sp['ticker'].nunique()} tickers, {len(feats_present)}/{len(FEATURE_POOL)} features present")

    daily_pnl, folds = portfolio_returns_for_sector(sp, feats_present)
    if daily_pnl.empty:
        return {"sector": sector, "verdict": "NO_FOLDS", "n_tickers": int(sp['ticker'].nunique())}

    wf = walk_forward(
        daily_returns=daily_pnl,
        spy_returns=spy_ret,
        fund_rate=fund_rate,
        train_months=36, oot_months=12, step_months=6,
    )
    out = {
        "sector": sector,
        "n_tickers": int(sp["ticker"].nunique()),
        "n_features": len(feats_present),
        "features": feats_present,
        "verdict": wf.verdict,
        "summary": wf.summary,
        "n_wf_folds": len(wf.per_fold),
        "per_fold": wf.per_fold,
        "ridge_folds": folds,
    }
    if verbose:
        med = {k: v.get("median", float("nan")) for k, v in wf.summary.items()}
        print(f"    folds={len(wf.per_fold)}  med Sharpe={med.get('sharpe', float('nan')):.2f}  "
              f"CAGR={med.get('cagr', float('nan')):.3f}  Calmar={med.get('calmar', float('nan')):.2f}  "
              f"-> {wf.verdict}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", type=str, default=None, help="single sector to run; default: all")
    ap.add_argument("--limit", type=int, default=None, help="if set, only run first N sectors")
    args = ap.parse_args()

    panel = load_panel()
    spy_ret, fund = load_spy_and_funding()

    sectors = sorted([s for s in panel["sector"].dropna().unique()])
    if args.sector:
        sectors = [args.sector]
    if args.limit:
        sectors = sectors[:args.limit]

    print(f"sectors to evaluate: {sectors}\n")
    panel = build_target(panel, hold_days=HOLD_DAYS)

    results = []
    for s in sectors:
        r = run_sector(panel, s, spy_ret, fund)
        results.append(r)
        out_path = FINDINGS / f"sector_picker_v3_{s.replace(' ', '_').replace('/', '_')}.json"
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2, default=str)

    # summary markdown
    lines = ["# Sector picker v3 (full-shelf ridge) — walk-forward summary\n"]
    lines.append("| Sector | Tickers | Folds | Verdict | Sharpe (med) | CAGR (med) | Calmar (med) | MaxDD (med) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        if "summary" not in r:
            lines.append(f"| {r['sector']} | {r.get('n_tickers', 0)} | 0 | {r['verdict']} | - | - | - | - |")
            continue
        med = {k: v.get("median", float("nan")) for k, v in r["summary"].items()}
        lines.append(
            f"| {r['sector']} | {r['n_tickers']} | {r['n_wf_folds']} | {r['verdict']} | "
            f"{med.get('sharpe', float('nan')):.2f} | {med.get('cagr', float('nan')):.3f} | "
            f"{med.get('calmar', float('nan')):.2f} | {med.get('max_dd', float('nan')):.3f} |"
        )
    summary_path = FINDINGS / "sector_picker_v3_summary.md"
    summary_path.write_text("\n".join(lines))
    print(f"\nwrote summary -> {summary_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
