"""
Tech (B) — Sub-industry rotation (semis / software / cyber / cloud).

HC #581 R2(b). Same picker shape as the leader (n_long picks from a tech ETF
universe based on cross-sectional momentum/quality score, hold21, longonly,
SPY-MA regime gate, 5bps txn, vol-targeted, sliding WF).

Universe (target): {XLK, XSW, SOXX, IGV, CIBR, SKYY, HACK, SMH}
Available in repo flow store: XLK, IGV, SMH. Missing tickers are fetched via
yfinance for the WF window; any that still fail are dropped (logged).

WF: sliding 24m / 6m / 3m (HC #0 — never expanding).

Outputs to output/macro_picker/tech_sub_industry_rotation_<TS>/:
  book.parquet, report.json, rebal_picks.parquet, regime_stratification.csv
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
from walk_forward import _metrics  # type: ignore

ETF_FLOWS_PATH = ROOT / "data/feature_store/sector_etf_flows/daily.parquet"
SPY_PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"

TARGET_UNIVERSE = ["XLK", "XSW", "SOXX", "IGV", "CIBR", "SKYY", "HACK", "SMH"]

TRADING_DAYS = 252
TXN_COST_BPS = 5.0
TARGET_VOL = 0.15
LEV_MIN, LEV_MAX = 0.25, 2.0
HOLD_DAYS = 21
REGIME_MA_DAYS = 60
N_LONG = 2  # top-2 from the tech basket each rebal (mirrors leader)

WF_START_OVERRIDE = pd.Timestamp("2021-06-07")  # earliest in flow store
WF_END_OVERRIDE = pd.Timestamp("2026-02-27")    # leader's upper bound
TRAIN_MONTHS, OOT_MONTHS, STEP_MONTHS = 24, 6, 3


def _spy_close_series() -> pd.Series:
    px = pd.read_parquet(SPY_PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    return px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]


def _build_features_from_close(df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    """Build ret_1d, ret_20d, ret_60d, rel_strength_spy from close series."""
    df = df.sort_values("date").copy()
    df["close"] = df["close"].astype(float)
    df["ret_1d"] = df["close"].pct_change()
    df["ret_20d"] = df["close"].pct_change(20)
    df["ret_60d"] = df["close"].pct_change(60)
    # Rel strength vs SPY: ratio of trailing 60d ret
    spy_aligned = spy_close.reindex(df["date"].values).values
    spy_60d = pd.Series(spy_aligned).pct_change(60).values
    df["rel_strength_spy"] = df["ret_60d"].values - spy_60d
    return df


def _fetch_yfinance(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(start=start.strftime("%Y-%m-%d"),
                         end=(end + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                         auto_adjust=True)
        if hist is None or hist.empty:
            return None
        hist = hist.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
        hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
        hist["etf"] = ticker
        return hist[["etf", "date", "close"]]
    except Exception as e:
        print(f"[tech_subind] yfinance fetch failed for {ticker}: {e}")
        return None


def build_panel() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Returns (panel, kept_universe, dropped_universe)."""
    flows = pd.read_parquet(ETF_FLOWS_PATH)
    flows["date"] = pd.to_datetime(flows["date"])
    in_flows = set(flows["etf"].unique())

    spy_close = _spy_close_series()

    kept = []
    dropped = []
    frames = []

    for t in TARGET_UNIVERSE:
        if t in in_flows:
            sub = flows[flows["etf"] == t][
                ["etf", "date", "close", "ret_1d", "ret_20d", "ret_60d", "rel_strength_spy"]
            ].copy()
            sub = sub[(sub["date"] >= WF_START_OVERRIDE) & (sub["date"] <= WF_END_OVERRIDE)]
            if len(sub) > 200:
                frames.append(sub)
                kept.append(t)
                continue
        # Try yfinance fetch
        fetched = _fetch_yfinance(t, WF_START_OVERRIDE - pd.Timedelta(days=120), WF_END_OVERRIDE)
        if fetched is not None and len(fetched) > 250:
            fetched = _build_features_from_close(fetched, spy_close)
            fetched = fetched[(fetched["date"] >= WF_START_OVERRIDE) & (fetched["date"] <= WF_END_OVERRIDE)]
            fetched = fetched[["etf", "date", "close", "ret_1d", "ret_20d", "ret_60d", "rel_strength_spy"]]
            if len(fetched) > 200:
                frames.append(fetched)
                kept.append(t)
                continue
        dropped.append(t)

    if not frames:
        raise RuntimeError("No tech ETFs available — cannot build panel.")

    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "etf"]).reset_index(drop=True)
    print(f"[tech_subind] kept={kept} dropped={dropped}")
    print(f"[tech_subind] panel rows={len(panel)} date={panel['date'].min().date()} → {panel['date'].max().date()}")
    return panel, kept, dropped


def _load_spy_regime(ma_days: int = REGIME_MA_DAYS) -> pd.Series:
    px = pd.read_parquet(SPY_PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    ma = spy.rolling(ma_days, min_periods=max(20, ma_days // 2)).mean()
    return pd.Series(np.where(spy > ma, "bull", "bear"), index=spy.index, name="reg").dropna()


def _load_spy_daily_ret() -> pd.Series:
    px = pd.read_parquet(SPY_PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    return spy.pct_change().dropna()


def _xs_zscore(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = panel.copy()
    for f in feats:
        if f not in out.columns:
            out[f] = 0.0
            continue
        x = pd.to_numeric(out[f], errors="coerce").astype(float)
        # winsorize 1/99
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        x = x.clip(lower=lo, upper=hi)
        out[f] = x
        mu = out.groupby("date")[f].transform("mean")
        sd = out.groupby("date")[f].transform("std")
        z = (x - mu) / sd.replace(0.0, np.nan)
        out[f] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _estimate_book_vol(panel: pd.DataFrame, rd: pd.Timestamp, longs: list[str],
                      lookback: int = 60) -> float:
    cutoff = rd - pd.Timedelta(days=lookback * 2 + 10)
    hist = panel[(panel["date"] < rd) & (panel["date"] >= cutoff) & (panel["etf"].isin(longs))]
    if hist.empty:
        return 0.0
    by_date = hist.groupby("date")["ret_1d"].mean().dropna().tail(lookback)
    if len(by_date) < 20:
        return 0.0
    sd = float(by_date.std(ddof=1))
    if not np.isfinite(sd):
        return 0.0
    return sd * np.sqrt(TRADING_DAYS)


def _iter_wf(start, end, train_m=TRAIN_MONTHS, oot_m=OOT_MONTHS, step_m=STEP_MONTHS):
    out, cursor = [], start
    while True:
        tr_s = cursor
        tr_e = tr_s + pd.DateOffset(months=train_m)
        os_ = tr_e
        oe = os_ + pd.DateOffset(months=oot_m)
        if oe > end + pd.Timedelta(days=1):
            break
        out.append((tr_s, tr_e, os_, oe))
        cursor = cursor + pd.DateOffset(months=step_m)
    return out


def run_wf(panel: pd.DataFrame, regime: pd.Series, n_long: int = N_LONG) -> tuple[pd.DataFrame, list]:
    """Sliding WF rotation: each fold, fit cross-sectional ridge on train,
    use it to score in OOT, pick top-n_long on each rebal date, hold N days,
    bull-only regime gate, vol-target sizing on the book."""
    feats = ["ret_20d", "ret_60d", "rel_strength_spy"]

    panel = panel.sort_values(["etf", "date"]).reset_index(drop=True)
    panel["close"] = panel["close"].astype(float)
    panel["y_fwd"] = panel.groupby("etf")["close"].shift(-HOLD_DAYS) / panel["close"] - 1.0

    windows = _iter_wf(panel["date"].min(), panel["date"].max())
    daily_rows = []
    rebal_rows = []

    for (tr_s, tr_e, os_, oe) in windows:
        train = panel[(panel["date"] >= tr_s) & (panel["date"] < tr_e)].copy()
        oot = panel[(panel["date"] >= os_) & (panel["date"] < oe)].copy()
        if len(train) < 100 or len(oot) < 20:
            continue
        # Cross-sectional z-score on train then OOT
        train_z = _xs_zscore(train, feats).dropna(subset=["y_fwd"])
        oot_z = _xs_zscore(oot, feats)
        if train_z.empty or oot_z.empty:
            continue

        # Fit ridge alpha=1.0 (simple, robust on small universe)
        X = train_z[feats].values
        y = train_z["y_fwd"].values
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        alpha = 1.0
        try:
            beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(len(feats)), Xc.T @ yc)
        except np.linalg.LinAlgError:
            beta = np.zeros(len(feats))
        intercept = y.mean() - X.mean(axis=0) @ beta
        oot_z["score"] = oot_z[feats].values @ beta + intercept
        # Pull raw ret_1d from un-z'd oot
        oot_z["ret_raw"] = pd.to_numeric(oot["ret_1d"], errors="coerce").astype(float).values

        unique_dates = sorted(oot_z["date"].unique())
        rebal_dates = unique_dates[::HOLD_DAYS]

        prev_longs: set[str] = set()
        for rd in rebal_dates:
            rd_ts = pd.Timestamp(rd)
            reg = regime.get(rd_ts)
            if reg is None:
                prior = regime.loc[:rd_ts]
                reg = prior.iloc[-1] if len(prior) else "bear"
            snap = oot_z[oot_z["date"] == rd].dropna(subset=["score"])
            if len(snap) < n_long:
                continue

            # No-trade floor: top score must beat median
            med = snap["score"].median()
            top = snap["score"].max()
            if top <= med or reg != "bull":
                longs: list[str] = []
                gross_lev = 0.0
            else:
                longs = snap.nlargest(n_long, "score")["etf"].tolist()
                book_vol = _estimate_book_vol(panel, rd_ts, longs)
                if book_vol <= 1e-6:
                    gross_lev = 1.0
                else:
                    gross_lev = float(np.clip(TARGET_VOL / book_vol, LEV_MIN, LEV_MAX))

            rebal_rows.append({
                "rebal_date": rd_ts,
                "regime": reg,
                "longs": ",".join(longs),
                "n_longs": len(longs),
                "gross_lev": gross_lev,
                "fold_oot_start": str(os_.date()),
            })

            hold_win = oot_z[(oot_z["date"] > rd) & (oot_z["date"] <= rd + pd.Timedelta(days=int(HOLD_DAYS * 1.6)))]
            for d, g in hold_win.groupby("date"):
                d_ts = pd.Timestamp(d)
                rg = regime.get(d_ts)
                if rg is None:
                    prior = regime.loc[:d_ts]
                    rg = prior.iloc[-1] if len(prior) else "bear"
                if not longs or rg != "bull":
                    daily_rows.append({"date": d_ts, "ret": 0.0, "lev": 0.0})
                    continue
                lret = g[g["etf"].isin(longs)]["ret_raw"].mean()
                lret = float(lret) if pd.notna(lret) else 0.0
                day_ret = gross_lev * lret
                day_ret = float(np.clip(day_ret, -0.20, 0.20))
                daily_rows.append({"date": d_ts, "ret": day_ret, "lev": gross_lev})

            # Txn cost on rebal date proportional to turnover
            new_longs = set(longs)
            turnover_legs = len(prev_longs.symmetric_difference(new_longs))
            if turnover_legs > 0:
                tc = (TXN_COST_BPS / 10000.0) * gross_lev * (turnover_legs / max(2 * n_long, 1))
                daily_rows.append({"date": rd_ts, "ret": -tc, "lev": gross_lev})
            prev_longs = new_longs

    if not daily_rows:
        return pd.DataFrame(columns=["date", "daily_ret", "gross_lev"]), rebal_rows

    df = pd.DataFrame(daily_rows)
    df["date"] = pd.to_datetime(df["date"])
    book = df.groupby("date", as_index=False).agg(daily_ret=("ret", "sum"), gross_lev=("lev", "max"))
    book = book.sort_values("date").reset_index(drop=True)
    return book, rebal_rows


def regime_stratification(book: pd.DataFrame, spy_ret: pd.Series, thresh: float = 0.001) -> pd.DataFrame:
    b = book.set_index("date")["daily_ret"]
    spy = spy_ret.reindex(b.index).fillna(0.0)
    cls = pd.Series("flat", index=b.index, dtype=object)
    cls[spy > thresh] = "green"
    cls[spy < -thresh] = "red"
    rows = []
    for r in ["green", "red", "flat"]:
        sub = b[cls == r].dropna()
        if len(sub) < 5:
            rows.append({"regime": r, "n_days": int(len(sub)), "mean_ret": float("nan"),
                         "sharpe": float("nan")})
            continue
        sd = sub.std(ddof=1)
        sh = float(sub.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
        rows.append({"regime": r, "n_days": int(len(sub)),
                     "mean_ret": float(sub.mean()), "sharpe": sh})
    return pd.DataFrame(rows)


def deploy_verdict(m: dict, strat_df: pd.DataFrame, book: pd.DataFrame) -> dict:
    calmar = m.get("calmar", float("nan"))
    sharpe = m.get("sharpe", float("nan"))
    n_days = int(book["daily_ret"].dropna().shape[0])
    g = strat_df[strat_df["regime"] == "green"]["sharpe"].iloc[0] if not strat_df.empty else float("nan")
    r = strat_df[strat_df["regime"] == "red"]["sharpe"].iloc[0] if not strat_df.empty else float("nan")
    if np.isfinite(g) and np.isfinite(r) and max(abs(g), abs(r)) > 0:
        regime_imbalance = abs(g - r) / max(abs(g), abs(r))
        regime_ok = regime_imbalance <= 0.50
    else:
        regime_imbalance = float("nan")
        regime_ok = False
    abs_pnl = book["daily_ret"].abs()
    if abs_pnl.sum() > 0:
        day_conc = float(abs_pnl.max() / abs_pnl.sum())
    else:
        day_conc = float("nan")
    day_conc_ok = np.isfinite(day_conc) and day_conc <= 0.70
    oot_days_ok = n_days >= 40
    calmar_ok = np.isfinite(calmar) and calmar >= 1.5
    sharpe_ok = np.isfinite(sharpe) and sharpe >= 1.0
    PASS = bool(calmar_ok and sharpe_ok and regime_ok and day_conc_ok and oot_days_ok)
    return {
        "calmar_ge_1_5": bool(calmar_ok),
        "sharpe_ge_1_0": bool(sharpe_ok),
        "regime_imbalance": float(regime_imbalance) if np.isfinite(regime_imbalance) else None,
        "regime_balance_ok": bool(regime_ok),
        "day_concentration": float(day_conc) if np.isfinite(day_conc) else None,
        "day_concentration_ok": bool(day_conc_ok),
        "n_oot_days": n_days,
        "oot_days_ge_40": bool(oot_days_ok),
        "PASSES_DEPLOY_GATES": PASS,
    }


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / f"output/macro_picker/tech_sub_industry_rotation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel, kept, dropped = build_panel()
    regime = _load_spy_regime()
    spy_ret = _load_spy_daily_ret()

    book, rebal_rows = run_wf(panel, regime)
    book.to_parquet(out_dir / "book.parquet", index=False)
    pd.DataFrame(rebal_rows).to_parquet(out_dir / "rebal_picks.parquet", index=False)

    s = book.set_index("date")["daily_ret"]
    m = _metrics(s) if not s.empty else {}
    strat_df = regime_stratification(book, spy_ret)
    strat_df.to_csv(out_dir / "regime_stratification.csv", index=False)
    gates = deploy_verdict(m, strat_df, book)

    report = {
        "strategy": "tech_sub_industry_rotation",
        "config": {
            "universe_target": TARGET_UNIVERSE,
            "universe_kept": kept,
            "universe_dropped": dropped,
            "n_long": N_LONG,
            "hold_days": HOLD_DAYS,
            "longonly": True,
            "txn_cost_bps": TXN_COST_BPS,
            "target_vol": TARGET_VOL,
            "lev_clip": [LEV_MIN, LEV_MAX],
            "regime_filter": f"SPY > MA{REGIME_MA_DAYS}d (bull-only)",
            "wf_train_months": TRAIN_MONTHS,
            "wf_oot_months": OOT_MONTHS,
            "wf_step_months": STEP_MONTHS,
            "wf_window_type": "SLIDING (HC #0)",
        },
        "date_range": [str(book["date"].min().date()) if not book.empty else None,
                       str(book["date"].max().date()) if not book.empty else None],
        "headline": {
            "cagr_pct": (m.get("cagr") or 0) * 100,
            "sharpe": m.get("sharpe"),
            "sortino": m.get("sortino"),
            "calmar": m.get("calmar"),
            "max_dd_pct": (m.get("max_dd") or 0) * 100,
            "pf": m.get("pf"),
            "wr_pct": (m.get("wr") or 0) * 100,
            "n_oot_days": int(s.dropna().shape[0]) if not s.empty else 0,
        },
        "regime_stratification": strat_df.to_dict(orient="records"),
        "deploy_gates": gates,
        "n_rebalances": len(rebal_rows),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"[tech_subind] wrote {out_dir}")
    print(f"[tech_subind] CAGR={report['headline']['cagr_pct']:.1f}% "
          f"Sharpe={report['headline']['sharpe']:.2f} "
          f"Calmar={report['headline']['calmar']:.2f} "
          f"MaxDD={report['headline']['max_dd_pct']:.1f}% "
          f"PASS={gates['PASSES_DEPLOY_GATES']}")
    return out_dir, report


if __name__ == "__main__":
    main()
