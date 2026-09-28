"""
Sector picker v4 — full-shelf ridge + 10-K text features (HC #564 R6(a)).

Same methodology as v3 (per-sector ridge over forward 21d returns, top/bot 3
long-short monthly rebalance, 5bps per-name turnover cost), but:
  - Input panel is master_panel_with_text.parquet (text features merged in).
  - Feature pool extended with lm_tone_score_z, rf_delta_z, flag_going_concern,
    flag_accounting_change, flag_restatement, days_since_filing (decay marker).
  - Reports POOLED-ALL-OOT metrics (single 2018-2025 series) — the honest
    test that matched the 4:43 PM verdict (Sharpe 0.69 / CAGR 7.0%).

For the pooled comparison we walk-forward across ALL 11 sectors, equal-weight
the per-sector long-short books on each trading day, then compute one set of
metrics over the entire OOT span.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))
from walk_forward import walk_forward, _metrics  # type: ignore

PANEL = ROOT / "data/feature_store/master_panel/master_panel_with_text.parquet"
SPY_PRICE = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"
MACRO_EXTRA = ROOT / "wheel_strategy_v1/data/cache/macro_extra.parquet"
OUT_DIR = ROOT / "research/findings/sector_picker_v4_with_10k_text"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
HOLD_DAYS = 21
TOP_N = 3
BOT_N = 3
TXN_COST_BPS = 5

# Reuse v3 pool + text features.
FEATURE_POOL_BASE = [
    "ret", "log_ret",
    "rv_cc_5d", "rv_cc_20d", "rv_cc_60d", "rv_cc_252d",
    "rv_pk_20d", "rv_yz_20d", "rv_yz_60d",
    "overnight_gap", "intraday_range_pct", "open_to_close_ret",
    "upper_shadow_pct", "lower_shadow_pct", "max_intraday_dd_pct",
    "dollar_volume",
    "ar_net_score", "ar_net_score_delta_qoq",
    "xa_GOLD_ret_20d", "xa_OIL_ret_20d", "xa_COPPER_ret_20d",
    "xa_BTC_ret_20d", "xa_UST10Y_ret_20d", "xa_DXY_ret_20d",
    "xa_GOLD_zscore_60d", "xa_DXY_zscore_60d", "xa_UST10Y_zscore_60d",
    "risk_dial", "severity",
    "sr_rel_strength_spy", "sr_momentum_cross_20_60",
    "sr_rs_rank_among_sectors", "sr_lead_lag_score_5d",
    "ins_n_buys", "ins_n_sells", "ins_net_share_change",
]
TEXT_FEATURES = [
    "lm_tone_score_z",
    "rf_delta_z",
    "flag_going_concern",
    "flag_accounting_change",
    "flag_restatement",
]
FEATURE_POOL = FEATURE_POOL_BASE + TEXT_FEATURES


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PANEL)
    p["date"] = pd.to_datetime(p["date"])
    # alias for missing ins_net_share_change (panel has ins_net_insider_usd)
    if "ins_net_share_change" not in p.columns:
        p["ins_net_share_change"] = p.get("ins_net_insider_usd", 0.0)
    return p


def load_spy_and_funding():
    px = pd.read_parquet(SPY_PRICE)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    if spy.empty:
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
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    fwd = (panel.groupby("ticker")["close"].shift(-hold_days) / panel["close"] - 1.0)
    panel["y_fwd"] = fwd
    return panel


def _fit_ridge(X, y, alphas=(0.1, 1.0, 10.0, 100.0)):
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


def _winsorize(s, p=0.01):
    lo = s.quantile(p); hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _xs_zscore(panel, feats):
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


def portfolio_for_sector(sp, feats):
    sp = sp.sort_values(["date", "ticker"]).reset_index(drop=True)
    sp["ret_raw"] = sp["ret"].astype(float)

    start, end = sp["date"].min(), sp["date"].max()
    cursor = start
    daily_pnl = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    folds = []
    avg_coefs = {f: [] for f in feats}

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
            cursor = cursor + pd.DateOffset(months=6)
            continue

        train_z = _xs_zscore(train, feats)
        oot_z = _xs_zscore(oot, feats)
        for f in feats:
            train_z[f] = train_z[f].fillna(0.0)
            oot_z[f] = oot_z[f].fillna(0.0)
        train_z = train_z.dropna(subset=["y_fwd"])
        if train_z.empty:
            cursor = cursor + pd.DateOffset(months=6); continue
        X_tr = train_z[feats].values
        y_tr = train_z["y_fwd"].values
        coef, intercept, alpha = _fit_ridge(X_tr, y_tr)
        if coef is None:
            cursor = cursor + pd.DateOffset(months=6); continue
        for f, c in zip(feats, coef):
            avg_coefs[f].append(float(c))

        X_oot = oot_z[feats].fillna(0.0).values
        oot_z = oot_z.copy()
        oot_z["score"] = X_oot @ coef + intercept

        unique_dates = sorted(oot_z["date"].unique())
        rebal_dates = unique_dates[::HOLD_DAYS]
        fold_daily = []
        for rd in rebal_dates:
            snap = oot_z[oot_z["date"] == rd].dropna(subset=["score"])
            if len(snap) < (TOP_N + BOT_N):
                continue
            longs = snap.nlargest(TOP_N, "score")["ticker"].tolist()
            shorts = snap.nsmallest(BOT_N, "score")["ticker"].tolist()
            hold_win = oot_z[(oot_z["date"] > rd)
                             & (oot_z["date"] <= rd + pd.Timedelta(days=HOLD_DAYS))]
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

        folds.append({
            "oot_start": str(oot_start.date()),
            "oot_end": str(oot_end.date()),
            "alpha": alpha,
            "coef": dict(zip(feats, [float(c) for c in coef])),
        })
        cursor = cursor + pd.DateOffset(months=6)

    daily_pnl = daily_pnl.sort_index()
    daily_pnl = daily_pnl[~daily_pnl.index.duplicated(keep="last")]
    # mean coefficient across folds
    avg_coef = {f: (float(np.mean(v)) if v else 0.0) for f, v in avg_coefs.items()}
    return daily_pnl, folds, avg_coef


def _spy_levered(spy, lev, fund_rate, spread_bps=150):
    daily_rate = ((fund_rate / 100.0 + spread_bps / 10000.0) / TRADING_DAYS).reindex(spy.index).ffill()
    excess_lev = max(lev - 1.0, 0.0)
    return lev * spy - excess_lev * daily_rate


def main():
    print("loading panel ...")
    panel = load_panel()
    print(f"  shape {panel.shape}, cols {len(panel.columns)}")
    spy_ret, fund = load_spy_and_funding()
    panel = build_target(panel)

    sectors = sorted([s for s in panel["sector"].dropna().unique() if s != "ETF"])
    print(f"sectors: {sectors}")

    per_sector_pnl = {}
    per_sector_meta = {}
    for s in sectors:
        sp = panel[panel["sector"] == s].copy()
        if sp.empty or sp["ticker"].nunique() < 5:
            print(f"  skip {s} (n={sp['ticker'].nunique()})")
            continue
        feats_present = [f for f in FEATURE_POOL if f in sp.columns]
        n_text = sum(1 for f in TEXT_FEATURES if f in feats_present)
        print(f"  {s}: {sp['ticker'].nunique()} tickers, {len(feats_present)} feats ({n_text} text)")
        pnl, folds, avg_coef = portfolio_for_sector(sp, feats_present)
        if pnl.empty:
            print(f"    no pnl"); continue
        per_sector_pnl[s] = pnl
        per_sector_meta[s] = {
            "n_tickers": int(sp["ticker"].nunique()),
            "n_features": len(feats_present),
            "n_folds": len(folds),
            "avg_coef": avg_coef,
            "folds": folds,
        }
        m = _metrics(pnl)
        print(f"    pooled-OOT: Sharpe={m['sharpe']:.2f} CAGR={m['cagr']:.3f} "
              f"Calmar={m['calmar']:.2f} MaxDD={m['max_dd']:.3f}")

    # Combined book — equal-weight across sectors per day
    print("\nbuilding combined book ...")
    if not per_sector_pnl:
        print("no sector pnl, aborting"); return
    combined_df = pd.concat(per_sector_pnl.values(), axis=1, keys=per_sector_pnl.keys())
    combined_pnl = combined_df.mean(axis=1, skipna=True).dropna()
    print(f"  {len(combined_pnl)} trading days, {combined_pnl.index.min()} -> {combined_pnl.index.max()}")

    # Pooled OOT metrics
    strat_m = _metrics(combined_pnl)
    aligned = pd.concat([combined_pnl.rename("s"),
                         spy_ret.rename("spy"),
                         fund.rename("f")], axis=1, join="inner").dropna(subset=["s", "spy"])
    aligned["f"] = aligned["f"].ffill().bfill()
    spy1 = _spy_levered(aligned["spy"], 1.0, aligned["f"])
    spy15 = _spy_levered(aligned["spy"], 1.5, aligned["f"])
    spy2 = _spy_levered(aligned["spy"], 2.0, aligned["f"])
    spy1_m = _metrics(spy1)
    spy15_m = _metrics(spy15)
    spy2_m = _metrics(spy2)

    # Per-sector pooled metrics
    per_sector_metrics = {s: _metrics(p) for s, p in per_sector_pnl.items()}

    # Feature importance — average |coef| across all sectors that ran
    feat_imp = {}
    for s, meta in per_sector_meta.items():
        for f, c in meta["avg_coef"].items():
            feat_imp.setdefault(f, []).append(abs(c))
    feat_imp_avg = {f: float(np.mean(v)) for f, v in feat_imp.items()}
    feat_imp_sorted = sorted(feat_imp_avg.items(), key=lambda x: -x[1])

    # 4:43 PM baseline (from sector_picker_v3_honest_verdict.md)
    baseline = {
        "sharpe": 0.69, "sortino": 0.98, "cagr": 0.070,
        "max_dd": -0.182, "calmar": 0.39,
    }

    # Build report
    out = {
        "baseline_443pm": baseline,
        "pooled_oot_combined": strat_m,
        "spy_1x": spy1_m,
        "spy_15x": spy15_m,
        "spy_2x": spy2_m,
        "per_sector": per_sector_metrics,
        "feature_importance_avg_abs_coef": dict(feat_imp_sorted),
        "n_trading_days": int(len(combined_pnl)),
        "date_range": [str(combined_pnl.index.min().date()),
                       str(combined_pnl.index.max().date())],
        "sectors_used": list(per_sector_pnl.keys()),
        "weights_per_sector": {s: m["avg_coef"] for s, m in per_sector_meta.items()},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(out, indent=2, default=str))

    # per-day P&L parquet
    pnl_df = pd.DataFrame({"date": combined_pnl.index,
                           "combined_pnl": combined_pnl.values})
    for s, p in per_sector_pnl.items():
        pnl_df = pnl_df.merge(
            pd.DataFrame({"date": p.index, s: p.values}), on="date", how="left")
    pnl_df.to_parquet(OUT_DIR / "per_day_pnl.parquet", index=False)

    # Markdown report
    top5 = feat_imp_sorted[:5]
    bot5 = feat_imp_sorted[-5:]
    text_ranks = {f: i + 1 for i, (f, _) in enumerate(feat_imp_sorted) if f in TEXT_FEATURES}
    delta_sharpe = strat_m["sharpe"] - baseline["sharpe"]
    delta_cagr = strat_m["cagr"] - baseline["cagr"]

    md = []
    md.append("# Sector picker v4 — with 10-K text features (HC #564 R6(a))")
    md.append("")
    md.append(f"**Pooled-OOT**: {len(combined_pnl)} trading days, "
              f"{combined_pnl.index.min().date()} -> {combined_pnl.index.max().date()}")
    md.append(f"**Sectors used**: {len(per_sector_pnl)} ({', '.join(per_sector_pnl.keys())})")
    md.append("")
    md.append("## Pooled-OOT vs 4:43 PM baseline (v3 shelf, no text)")
    md.append("")
    md.append("| Metric | v4 (with text) | v3 baseline (4:43 PM) | Delta |")
    md.append("|---|---|---|---|")
    md.append(f"| Sharpe | {strat_m['sharpe']:.2f} | {baseline['sharpe']:.2f} | {delta_sharpe:+.2f} |")
    md.append(f"| Sortino | {strat_m['sortino']:.2f} | {baseline['sortino']:.2f} | {strat_m['sortino']-baseline['sortino']:+.2f} |")
    md.append(f"| CAGR | {strat_m['cagr']*100:.1f}% | {baseline['cagr']*100:.1f}% | {delta_cagr*100:+.1f} pp |")
    md.append(f"| MaxDD | {strat_m['max_dd']*100:.1f}% | {baseline['max_dd']*100:.1f}% | {(strat_m['max_dd']-baseline['max_dd'])*100:+.1f} pp |")
    md.append(f"| Calmar | {strat_m['calmar']:.2f} | {baseline['calmar']:.2f} | {strat_m['calmar']-baseline['calmar']:+.2f} |")
    md.append("")
    md.append("## vs SPY (levered, with financing)")
    md.append("")
    md.append("| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |")
    md.append("|---|---|---|---|---|")
    md.append(f"| Sharpe | {strat_m['sharpe']:.2f} | {spy1_m['sharpe']:.2f} | {spy15_m['sharpe']:.2f} | {spy2_m['sharpe']:.2f} |")
    md.append(f"| CAGR | {strat_m['cagr']*100:.1f}% | {spy1_m['cagr']*100:.1f}% | {spy15_m['cagr']*100:.1f}% | {spy2_m['cagr']*100:.1f}% |")
    md.append(f"| MaxDD | {strat_m['max_dd']*100:.1f}% | {spy1_m['max_dd']*100:.1f}% | {spy15_m['max_dd']*100:.1f}% | {spy2_m['max_dd']*100:.1f}% |")
    md.append(f"| Calmar | {strat_m['calmar']:.2f} | {spy1_m['calmar']:.2f} | {spy15_m['calmar']:.2f} | {spy2_m['calmar']:.2f} |")
    md.append("")
    md.append("## Per-sector pooled-OOT")
    md.append("")
    md.append("| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |")
    md.append("|---|---|---|---|---|---|---|")
    for s, m in sorted(per_sector_metrics.items(), key=lambda x: -x[1]["sharpe"]):
        md.append(f"| {s} | {m['sharpe']:.2f} | {m['cagr']*100:.1f}% | {m['max_dd']*100:.1f}% | "
                  f"{m['calmar']:.2f} | {m['pf']:.2f} | {m['wr']*100:.1f}% |")
    md.append("")
    md.append("## Feature importance (avg |coef| across sectors)")
    md.append("")
    md.append("### Top 5")
    for f, c in top5:
        tag = "  <-- TEXT" if f in TEXT_FEATURES else ""
        md.append(f"- `{f}`: {c:.4f}{tag}")
    md.append("")
    md.append("### Bottom 5")
    for f, c in bot5:
        tag = "  <-- TEXT" if f in TEXT_FEATURES else ""
        md.append(f"- `{f}`: {c:.4f}{tag}")
    md.append("")
    md.append("### Where text features landed")
    for f in TEXT_FEATURES:
        r = text_ranks.get(f, None)
        c = feat_imp_avg.get(f, 0.0)
        md.append(f"- `{f}`: rank {r}/{len(feat_imp_sorted)}, avg |coef|={c:.4f}")
    md.append("")

    # verdict
    md.append("## Verdict")
    md.append("")
    calmar_floor_pass = strat_m["calmar"] >= 1.0
    beats_spy1x_sharpe = strat_m["sharpe"] > spy1_m["sharpe"]
    beats_spy15x_sharpe = strat_m["sharpe"] > spy15_m["sharpe"]
    beats_spy15x_cagr = strat_m["cagr"] > spy15_m["cagr"]
    moved = abs(delta_sharpe) >= 0.05 or abs(delta_cagr) >= 0.01

    md.append(f"- Calmar floor (>=1.0): {'PASS' if calmar_floor_pass else 'FAIL'} ({strat_m['calmar']:.2f})")
    md.append(f"- Beats SPY 1x on Sharpe: {'YES' if beats_spy1x_sharpe else 'NO'}")
    md.append(f"- Beats SPY 1.5x on Sharpe: {'YES' if beats_spy15x_sharpe else 'NO'}")
    md.append(f"- Beats SPY 1.5x on CAGR: {'YES' if beats_spy15x_cagr else 'NO'}")
    md.append(f"- 10-K text meaningfully moved the result: {'YES' if moved else 'NO'} "
              f"(dSharpe={delta_sharpe:+.2f}, dCAGR={delta_cagr*100:+.1f}pp)")
    md.append("")

    (OUT_DIR / "report.md").write_text("\n".join(md))
    print("\nWROTE:")
    print(f"  {OUT_DIR / 'report.json'}")
    print(f"  {OUT_DIR / 'report.md'}")
    print(f"  {OUT_DIR / 'per_day_pnl.parquet'}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
