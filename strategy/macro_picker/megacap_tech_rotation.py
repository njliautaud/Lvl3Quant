"""
HC #581 R2(c) — Megacap-tech rotation.

Basket of single-stock megacap tech names with a regime-gated momentum picker.
Same picker SHAPE as tech_sub_industry_rotation.py (cross-sectional ridge on
momentum/relative-strength features), but the universe is single stocks not
ETFs. Picker is fit per WF fold (no expanding windows — HC #0 sliding).

Universe: AAPL, MSFT, GOOGL, NVDA, META, AMZN, AVGO, TSLA  (TSLA subs for TSM —
TSM is ADR with sparse data in the cache; TSLA is the closest liquid megacap
alternative).

Strategy spec (per HC #586 R1 dispatch):
  - top-K = 1, 2, 3 (three runs)
  - weekly rebalance (HOLD_DAYS=5)
  - equal weight within holdings, no leverage (1.0x sizing — NOT vol-targeted)
  - regime gate at ENTRY: SPY > 50d MA AND VIX < 25, else cash
  - intra-hold regime gate: same — flip to cash if either condition fails
  - costs: $0.005/share commission + 1bp slippage per trade

Outputs to output/macro_picker/megacap_tech_rotation_v1_<TS>/:
  book_k{K}.parquet, rebal_picks_k{K}.parquet, regime_stratification_k{K}.csv,
  report.json (combined for all K)
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

PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "AVGO", "TSLA"]
BENCH_SPY = "SPY"
BENCH_QQQ = "QQQ"

TRADING_DAYS = 252
HOLD_DAYS = 5  # weekly rebalance
REGIME_MA_DAYS = 50  # SPY > 50d MA
VIX_THRESH = 25.0

# Costs
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 1.0  # 1bp per trade

# Walk-forward (sliding — HC #0)
TRAIN_MONTHS = 24
OOT_MONTHS = 6
STEP_MONTHS = 3

# Test window: 2018-2025 (covers prior leader/tech-blend window)
WF_START = pd.Timestamp("2018-01-01")
WF_END = pd.Timestamp("2025-12-31")


def _load_prices() -> pd.DataFrame:
    """Load close prices for universe + benchmarks."""
    p = pd.read_parquet(PRICE_PATH)
    p["date"] = pd.to_datetime(p["date"])
    tickers = UNIVERSE + [BENCH_SPY, BENCH_QQQ]
    sub = p[p["ticker"].isin(tickers)][["ticker", "date", "close"]].copy()
    sub["close"] = sub["close"].astype(float)
    sub = sub[(sub["date"] >= WF_START - pd.Timedelta(days=400))
              & (sub["date"] <= WF_END + pd.Timedelta(days=2))]
    return sub.sort_values(["ticker", "date"]).reset_index(drop=True)


def _load_vix() -> pd.Series:
    """Load VIX close via yfinance. Returns series indexed by date."""
    import yfinance as yf
    v = yf.Ticker("^VIX").history(
        start=(WF_START - pd.Timedelta(days=400)).strftime("%Y-%m-%d"),
        end=(WF_END + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        auto_adjust=True,
    )
    v = v.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "vix"})
    v["date"] = pd.to_datetime(v["date"]).dt.tz_localize(None).dt.normalize()
    return v.set_index("date")["vix"].sort_index()


def _build_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Build per-stock daily features for the trading universe (excludes bench)."""
    spy = prices[prices["ticker"] == BENCH_SPY].set_index("date")["close"].sort_index()
    spy_ret60 = spy.pct_change(60)

    rows = []
    for t in UNIVERSE:
        sub = prices[prices["ticker"] == t].sort_values("date").copy()
        sub["ret_1d"] = sub["close"].pct_change()
        sub["ret_20d"] = sub["close"].pct_change(20)
        sub["ret_60d"] = sub["close"].pct_change(60)
        # Rel-strength vs SPY (60d)
        spy_aligned = spy_ret60.reindex(sub["date"].values).values
        sub["rel_strength_spy"] = sub["ret_60d"].values - spy_aligned
        rows.append(sub)
    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def _spy_regime(prices: pd.DataFrame, ma_days: int = REGIME_MA_DAYS) -> pd.Series:
    """SPY > MA50 → 'bull', else 'bear'."""
    spy = prices[prices["ticker"] == BENCH_SPY].set_index("date")["close"].sort_index()
    ma = spy.rolling(ma_days, min_periods=20).mean()
    return pd.Series(np.where(spy > ma, "bull", "bear"), index=spy.index).dropna()


def _vix_ok(vix: pd.Series, thresh: float = VIX_THRESH) -> pd.Series:
    """VIX < thresh → True."""
    return (vix < thresh).rename("vix_ok")


def _combined_regime(spy_reg: pd.Series, vix_ok: pd.Series) -> pd.Series:
    """'bull' iff SPY-MA bull AND VIX < thresh; else 'cash'."""
    idx = spy_reg.index.union(vix_ok.index)
    spy_r = spy_reg.reindex(idx).ffill()
    v_ok = vix_ok.reindex(idx).ffill()
    out = pd.Series("cash", index=idx)
    out[(spy_r == "bull") & (v_ok == True)] = "bull"
    return out


def _xs_zscore(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = panel.copy()
    for f in feats:
        if f not in out.columns:
            out[f] = 0.0
            continue
        x = pd.to_numeric(out[f], errors="coerce").astype(float)
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        x = x.clip(lower=lo, upper=hi)
        out[f] = x
        mu = out.groupby("date")[f].transform("mean")
        sd = out.groupby("date")[f].transform("std")
        z = (x - mu) / sd.replace(0.0, np.nan)
        out[f] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


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


def _estimate_share_price_for_costs(panel: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float:
    """Get close price on date (or asof) for the ticker."""
    sub = panel[(panel["ticker"] == ticker) & (panel["date"] <= date)]
    if sub.empty:
        return 100.0
    return float(sub.iloc[-1]["close"])


def _trade_cost_pct(price: float) -> float:
    """Per-leg trade cost as fraction of notional: commission + slippage."""
    # Commission is per share, slippage is 1bp on notional.
    # For an equal-weight position at notional N, shares = N/price.
    # commission = shares * $0.005 = N * 0.005/price.
    # So commission as fraction of notional = 0.005/price.
    comm_frac = 0.005 / max(price, 1.0)
    slip_frac = SLIPPAGE_BPS / 10000.0
    return comm_frac + slip_frac


def run_wf(panel: pd.DataFrame, regime: pd.Series, n_long: int) -> tuple[pd.DataFrame, list]:
    """Sliding WF rotation: fit ridge picker on train, rotate top-K on OOT."""
    feats = ["ret_20d", "ret_60d", "rel_strength_spy"]

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["close"] = panel["close"].astype(float)
    panel["y_fwd"] = panel.groupby("ticker")["close"].shift(-HOLD_DAYS) / panel["close"] - 1.0

    windows = _iter_wf(WF_START, WF_END)
    daily_rows = []
    rebal_rows = []
    prev_longs: set[str] = set()

    for (tr_s, tr_e, os_, oe) in windows:
        train = panel[(panel["date"] >= tr_s) & (panel["date"] < tr_e)].copy()
        oot = panel[(panel["date"] >= os_) & (panel["date"] < oe)].copy()
        if len(train) < 100 or len(oot) < 20:
            continue

        # Cross-sectional z-score on train then OOT.
        train_z = _xs_zscore(train, feats).dropna(subset=["y_fwd"])
        oot_z = _xs_zscore(oot, feats)
        if train_z.empty or oot_z.empty:
            continue

        # Fit ridge alpha=1.0
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
        oot_z["ret_raw"] = pd.to_numeric(oot["ret_1d"], errors="coerce").astype(float).values

        unique_dates = sorted(oot_z["date"].unique())
        rebal_dates = unique_dates[::HOLD_DAYS]

        for rd in rebal_dates:
            rd_ts = pd.Timestamp(rd)
            reg = regime.get(rd_ts)
            if reg is None:
                prior = regime.loc[:rd_ts]
                reg = prior.iloc[-1] if len(prior) else "cash"

            snap = oot_z[oot_z["date"] == rd].dropna(subset=["score"])
            if len(snap) < n_long:
                continue

            if reg != "bull":
                longs: list[str] = []
            else:
                # Top-K by score
                longs = snap.nlargest(n_long, "score")["ticker"].tolist()

            rebal_rows.append({
                "rebal_date": rd_ts,
                "regime": reg,
                "longs": ",".join(longs),
                "n_longs": len(longs),
                "fold_oot_start": str(os_.date()),
            })

            # Compute turnover cost on this rebal
            new_longs = set(longs)
            entered = new_longs - prev_longs
            exited = prev_longs - new_longs
            tc = 0.0
            for t in entered.union(exited):
                # Cost as a fraction of THAT LEG's notional; portfolio weight
                # of one leg is 1/n_long. So portfolio impact = (1/n_long) * tc_pct
                price = _estimate_share_price_for_costs(panel, t, rd_ts)
                leg_tc = _trade_cost_pct(price)
                weight = 1.0 / max(n_long, 1) if t in new_longs else 1.0 / max(len(prev_longs), 1)
                tc += weight * leg_tc
            if tc > 0:
                daily_rows.append({"date": rd_ts, "ret": -tc})
            prev_longs = new_longs

            # Hold period: next HOLD_DAYS trading days
            hold_win = oot_z[(oot_z["date"] > rd) & (oot_z["date"] <= rd + pd.Timedelta(days=int(HOLD_DAYS * 1.5)))]
            for d, g in hold_win.groupby("date"):
                d_ts = pd.Timestamp(d)
                rg = regime.get(d_ts)
                if rg is None:
                    prior = regime.loc[:d_ts]
                    rg = prior.iloc[-1] if len(prior) else "cash"
                # Intra-hold gate: if regime flips to cash, sit out the day
                if not longs or rg != "bull":
                    daily_rows.append({"date": d_ts, "ret": 0.0})
                    continue
                lret = g[g["ticker"].isin(longs)]["ret_raw"].mean()
                lret = float(lret) if pd.notna(lret) else 0.0
                # No leverage — equal weight, 1.0x sizing
                daily_rows.append({"date": d_ts, "ret": lret})

    if not daily_rows:
        return pd.DataFrame(columns=["date", "daily_ret"]), rebal_rows

    df = pd.DataFrame(daily_rows)
    df["date"] = pd.to_datetime(df["date"])
    book = df.groupby("date", as_index=False).agg(daily_ret=("ret", "sum"))
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
        "regime_green_sharpe": float(g) if np.isfinite(g) else None,
        "regime_red_sharpe": float(r) if np.isfinite(r) else None,
        "day_concentration": float(day_conc) if np.isfinite(day_conc) else None,
        "day_concentration_ok": bool(day_conc_ok),
        "n_oot_days": n_days,
        "oot_days_ge_40": bool(oot_days_ok),
        "PASSES_DEPLOY_GATES": PASS,
    }


def name_concentration(rebal_rows: list, n_long: int) -> dict:
    """Compute max % of portfolio in any one ticker over the test window."""
    if not rebal_rows:
        return {}
    # Count weight-days for each name
    df = pd.DataFrame(rebal_rows)
    df = df[df["n_longs"] > 0]
    name_days = {}
    for _, r in df.iterrows():
        if not r["longs"]:
            continue
        names = r["longs"].split(",")
        w = 1.0 / len(names)
        for n in names:
            name_days[n] = name_days.get(n, 0) + w
    total = sum(name_days.values())
    if total == 0:
        return {}
    return {n: v / total for n, v in sorted(name_days.items(), key=lambda x: -x[1])}


def earnings_event_check(book: pd.DataFrame, rebal_rows: list, panel: pd.DataFrame) -> list:
    """Find days where book daily_ret <= -10%; for each, identify which holding drove it."""
    bad_days = book[book["daily_ret"] <= -0.10].copy()
    if bad_days.empty:
        return []
    # Build hold map from rebal_rows
    df = pd.DataFrame(rebal_rows).sort_values("rebal_date")
    out = []
    for _, bd in bad_days.iterrows():
        d = bd["date"]
        # Find the last rebal before this date
        prev = df[df["rebal_date"] <= d]
        if prev.empty:
            continue
        last = prev.iloc[-1]
        names = last["longs"].split(",") if last["longs"] else []
        # Find single-name returns on that day
        rets = {}
        for n in names:
            r = panel[(panel["ticker"] == n) & (panel["date"] == d)]["ret_1d"]
            if not r.empty:
                rets[n] = float(r.iloc[0])
        worst = min(rets.items(), key=lambda x: x[1]) if rets else (None, None)
        out.append({
            "date": str(pd.Timestamp(d).date()),
            "book_ret": float(bd["daily_ret"]),
            "holdings": names,
            "name_returns": rets,
            "worst_name": worst[0],
            "worst_name_ret": worst[1],
        })
    return out


def bench_metrics(prices: pd.DataFrame, ticker: str, dates: pd.DatetimeIndex) -> dict:
    p = prices[prices["ticker"] == ticker].sort_values("date").set_index("date")["close"]
    p = p.reindex(dates).ffill()
    rets = p.pct_change().dropna()
    if rets.empty:
        return {}
    m = _metrics(rets)
    return {
        "cagr_pct": (m.get("cagr") or 0) * 100,
        "sharpe": m.get("sharpe"),
        "sortino": m.get("sortino"),
        "max_dd_pct": (m.get("max_dd") or 0) * 100,
    }


def equity_curve(book: pd.DataFrame, anchor: float = 20000.0) -> pd.DataFrame:
    b = book.copy().sort_values("date")
    b["equity"] = anchor * (1.0 + b["daily_ret"]).cumprod()
    return b


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / f"output/macro_picker/megacap_tech_rotation_v1_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[megacap] out_dir={out_dir}")

    print("[megacap] loading prices…")
    prices = _load_prices()
    print(f"[megacap] price rows={len(prices)}")

    print("[megacap] loading VIX (yfinance)…")
    vix = _load_vix()
    print(f"[megacap] vix rows={len(vix)}")

    print("[megacap] building panel + regime…")
    panel = _build_panel(prices)
    spy_reg = _spy_regime(prices)
    vix_ok = _vix_ok(vix)
    regime = _combined_regime(spy_reg, vix_ok)
    bull_frac = float((regime == "bull").mean())
    cash_frac = float((regime == "cash").mean())
    print(f"[megacap] regime: bull_frac={bull_frac:.2f} cash_frac={cash_frac:.2f}")

    spy_close = prices[prices["ticker"] == BENCH_SPY].sort_values("date").set_index("date")["close"]
    spy_ret = spy_close.pct_change().dropna()

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("megacap_tech_rotation_v1")
        mlflow_ok = True
    except Exception as e:
        print(f"[megacap] MLflow unavailable: {e}")
        mlflow_ok = False

    combined_report = {
        "strategy": "megacap_tech_rotation_v1",
        "config": {
            "universe": UNIVERSE,
            "universe_note": "TSLA substitutes for TSM (TSM ADR not in price cache).",
            "hold_days_weekly": HOLD_DAYS,
            "regime_gate": f"SPY > {REGIME_MA_DAYS}d MA AND VIX < {VIX_THRESH}",
            "leverage": "1.0x (no leverage, equal-weight)",
            "costs": {"commission_per_share_usd": COMMISSION_PER_SHARE,
                      "slippage_bps": SLIPPAGE_BPS},
            "wf": {"train_months": TRAIN_MONTHS, "oot_months": OOT_MONTHS,
                   "step_months": STEP_MONTHS, "window_type": "SLIDING (HC #0)"},
            "test_window": [str(WF_START.date()), str(WF_END.date())],
            "anchor_usd": 20000,
        },
        "regime_summary": {"bull_frac": bull_frac, "cash_frac": cash_frac},
        "by_K": {},
    }

    bench_dates_collected = None

    for K in [1, 2, 3]:
        print(f"\n[megacap] === K={K} ===")
        book, rebal_rows = run_wf(panel, regime, n_long=K)
        if book.empty:
            print(f"[megacap] K={K} produced empty book")
            continue

        book.to_parquet(out_dir / f"book_k{K}.parquet", index=False)
        pd.DataFrame(rebal_rows).to_parquet(out_dir / f"rebal_picks_k{K}.parquet", index=False)

        s = book.set_index("date")["daily_ret"]
        m = _metrics(s)
        strat_df = regime_stratification(book, spy_ret)
        strat_df.to_csv(out_dir / f"regime_stratification_k{K}.csv", index=False)
        gates = deploy_verdict(m, strat_df, book)
        concentration = name_concentration(rebal_rows, K)
        earnings_events = earnings_event_check(book, rebal_rows, panel)

        # Time-in-market: days where any holding was active vs total OOT days
        rebal_df = pd.DataFrame(rebal_rows)
        active_rebals = int((rebal_df["n_longs"] > 0).sum()) if not rebal_df.empty else 0
        total_rebals = len(rebal_df) if not rebal_df.empty else 0
        time_in_market_pct = float(active_rebals / total_rebals * 100) if total_rebals > 0 else 0.0
        # More precise: days with nonzero return
        nonzero_days = int((s.abs() > 1e-9).sum())
        time_in_market_pct_days = float(nonzero_days / len(s) * 100) if len(s) > 0 else 0.0

        # Benchmarks on same date span
        bench_dates = pd.DatetimeIndex(s.index)
        bench_dates_collected = bench_dates
        spy_bench = bench_metrics(prices, BENCH_SPY, bench_dates)
        qqq_bench = bench_metrics(prices, BENCH_QQQ, bench_dates)

        # Equity curve
        eq = equity_curve(book)
        eq.to_csv(out_dir / f"equity_k{K}.csv", index=False)
        final_equity = float(eq["equity"].iloc[-1])

        report = {
            "K": K,
            "headline": {
                "cagr_pct": (m.get("cagr") or 0) * 100,
                "sharpe": m.get("sharpe"),
                "sortino": m.get("sortino"),
                "calmar": m.get("calmar"),
                "max_dd_pct": (m.get("max_dd") or 0) * 100,
                "pf": m.get("pf"),
                "wr_pct": (m.get("wr") or 0) * 100,
                "n_oot_days": int(s.dropna().shape[0]),
                "final_equity_usd": final_equity,
            },
            "regime_stratification": strat_df.to_dict(orient="records"),
            "deploy_gates": gates,
            "name_concentration_pct": {k: v * 100 for k, v in concentration.items()},
            "max_single_name_pct": max(concentration.values()) * 100 if concentration else 0.0,
            "earnings_event_days": earnings_events,
            "n_severe_loss_days": len(earnings_events),
            "time_in_market_pct_days": time_in_market_pct_days,
            "n_rebalances": total_rebals,
            "n_active_rebalances": active_rebals,
            "bench_spy_buy_hold": spy_bench,
            "bench_qqq_buy_hold": qqq_bench,
        }
        combined_report["by_K"][f"K{K}"] = report

        # MLflow log
        if mlflow_ok:
            try:
                with mlflow.start_run(run_name=f"megacap_tech_rotation_K{K}"):
                    mlflow.log_param("K", K)
                    mlflow.log_param("hold_days", HOLD_DAYS)
                    mlflow.log_param("regime_gate", f"SPY>MA{REGIME_MA_DAYS} AND VIX<{VIX_THRESH}")
                    mlflow.log_param("universe", ",".join(UNIVERSE))
                    mlflow.log_param("anchor_usd", 20000)
                    for k, v in report["headline"].items():
                        if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                            mlflow.log_metric(f"headline_{k}", float(v))
                    mlflow.log_metric("regime_imbalance",
                                      gates.get("regime_imbalance") or 0.0)
                    mlflow.log_metric("day_concentration",
                                      gates.get("day_concentration") or 0.0)
                    mlflow.log_metric("max_single_name_pct", report["max_single_name_pct"])
                    mlflow.log_metric("time_in_market_pct", report["time_in_market_pct_days"])
                    mlflow.log_metric("n_severe_loss_days", report["n_severe_loss_days"])
                    mlflow.log_metric("passes_deploy_gates", int(gates["PASSES_DEPLOY_GATES"]))
                    mlflow.log_artifact(str(out_dir / f"book_k{K}.parquet"))
                    mlflow.log_artifact(str(out_dir / f"regime_stratification_k{K}.csv"))
            except Exception as e:
                print(f"[megacap] MLflow log failed for K={K}: {e}")

        print(f"[megacap] K={K} CAGR={report['headline']['cagr_pct']:.1f}% "
              f"Sharpe={report['headline']['sharpe']:.2f} "
              f"Calmar={report['headline']['calmar']:.2f} "
              f"MaxDD={report['headline']['max_dd_pct']:.1f}% "
              f"green_sh={gates.get('regime_green_sharpe')} "
              f"red_sh={gates.get('regime_red_sharpe')} "
              f"gap={gates.get('regime_imbalance')} "
              f"PASS={gates['PASSES_DEPLOY_GATES']}")

    (out_dir / "report.json").write_text(json.dumps(combined_report, indent=2, default=str))
    print(f"\n[megacap] wrote {out_dir}/report.json")
    return out_dir, combined_report


if __name__ == "__main__":
    main()
