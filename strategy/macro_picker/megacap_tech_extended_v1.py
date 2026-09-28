"""
Megacap-tech rotation EXTENDED sweep (extension of megacap_tech_rotation.py).

Pre-stages variant configs in case the user (a) relaxes HC #428 R1 to a tail-DD
gate, (b) prefers a higher-K variant, (c) prefers volatility-targeted weighting,
or (d) adopts a tail-risk overlay (long-put hedge).

Experiments:
  1. K-sweep: K=4,5,6 (on same 8-ticker universe).
  2. Momentum-window sweep at K=3: 1m, 6m, 12m (3m already done).
  3. Volatility-targeted weighting at K=3 (inverse 20d realized vol).
  4. Tail-DD gate evaluation across all variants (worst SPY-regime quarter DD <= 25%).
  5. Tail-risk overlay: K=3 plus rolling -10d 90-DTE SPY put hedge (rolled at DTE<30).
     The hedge is approximated via a Black-Scholes-priced synthetic SPY put using
     a flat VIX-derived IV, since this codebase doesn't have a live options chain
     archive for ALL test dates back to 2018.

Outputs:
  output/macro_picker/megacap_tech_extended_v1_<TS>/
    book_<variant>.parquet, regime_strat_<variant>.csv,
    tail_dd_<variant>.csv, report.json
  research/findings/megacap_tech_extended_v1.md  (final comparison report)

If hedged-K3 passes HC #428 R1: writes paper engine to
  live_trading_linux/megacap_hedged_paper_engine.py (entries_paused=True)
If a higher-K passes HC #428 R1: writes paper engine to
  live_trading_linux/megacap_paper_engine.py (entries_paused=True)
Else if tail-DD gate cleanly passes for K=3 family: pre-writes
  live_trading_linux/megacap_paper_engine.py (entries_paused=True) for user
  to flip after relaxing HC #428 R1.
"""
from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "strategy" / "macro_picker"))

from walk_forward import _metrics  # type: ignore
from megacap_tech_rotation import (  # type: ignore
    UNIVERSE, BENCH_SPY, BENCH_QQQ, HOLD_DAYS, REGIME_MA_DAYS, VIX_THRESH,
    COMMISSION_PER_SHARE, SLIPPAGE_BPS, TRAIN_MONTHS, OOT_MONTHS, STEP_MONTHS,
    WF_START, WF_END,
    _load_prices, _load_vix, _spy_regime, _vix_ok, _combined_regime,
    _xs_zscore, _iter_wf, _estimate_share_price_for_costs, _trade_cost_pct,
    regime_stratification, bench_metrics, equity_curve, name_concentration,
    deploy_verdict,
)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Panel builder parameterized by momentum window
# ---------------------------------------------------------------------------
def build_panel_with_window(prices: pd.DataFrame, mom_days: int) -> pd.DataFrame:
    """Build panel where ret_60d is replaced by ret_<mom_days>."""
    spy = prices[prices["ticker"] == BENCH_SPY].set_index("date")["close"].sort_index()
    spy_mom = spy.pct_change(mom_days)
    rows = []
    for t in UNIVERSE:
        sub = prices[prices["ticker"] == t].sort_values("date").copy()
        sub["ret_1d"] = sub["close"].pct_change()
        sub["ret_20d"] = sub["close"].pct_change(20)
        sub["ret_mom"] = sub["close"].pct_change(mom_days)
        # 20-day realized vol (daily stdev, annualized in pct terms)
        sub["rv_20"] = sub["ret_1d"].rolling(20, min_periods=5).std() * math.sqrt(TRADING_DAYS)
        spy_aligned = spy_mom.reindex(sub["date"].values).values
        sub["rel_strength_spy"] = sub["ret_mom"].values - spy_aligned
        rows.append(sub)
    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Walk-forward (parameterized: K, mom_days, vol_target)
# ---------------------------------------------------------------------------
def run_wf(panel: pd.DataFrame, regime: pd.Series, n_long: int,
           vol_target: bool = False) -> tuple[pd.DataFrame, list]:
    feats = ["ret_20d", "ret_mom", "rel_strength_spy"]
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["close"] = panel["close"].astype(float)
    panel["y_fwd"] = panel.groupby("ticker")["close"].shift(-HOLD_DAYS) / panel["close"] - 1.0

    windows = _iter_wf(WF_START, WF_END)
    daily_rows = []
    rebal_rows = []
    prev_longs_weights: dict[str, float] = {}

    for (tr_s, tr_e, os_, oe) in windows:
        train = panel[(panel["date"] >= tr_s) & (panel["date"] < tr_e)].copy()
        oot = panel[(panel["date"] >= os_) & (panel["date"] < oe)].copy()
        if len(train) < 100 or len(oot) < 20:
            continue
        train_z = _xs_zscore(train, feats).dropna(subset=["y_fwd"])
        oot_z = _xs_zscore(oot, feats)
        if train_z.empty or oot_z.empty:
            continue
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
        oot_z["rv_20"] = pd.to_numeric(oot["rv_20"], errors="coerce").astype(float).values

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
                longs_w: dict[str, float] = {}
            else:
                top = snap.nlargest(n_long, "score")
                if vol_target:
                    rv = top["rv_20"].replace(0.0, np.nan).fillna(top["rv_20"].mean())
                    inv = 1.0 / rv.clip(lower=0.05)
                    w = (inv / inv.sum()).values
                else:
                    w = np.full(n_long, 1.0 / n_long)
                longs_w = dict(zip(top["ticker"].tolist(), w))

            rebal_rows.append({
                "rebal_date": rd_ts,
                "regime": reg,
                "longs": ",".join(list(longs_w.keys())),
                "weights": ",".join(f"{v:.4f}" for v in longs_w.values()),
                "n_longs": len(longs_w),
                "fold_oot_start": str(os_.date()),
            })

            new_names = set(longs_w.keys())
            old_names = set(prev_longs_weights.keys())
            entered = new_names - old_names
            exited = old_names - new_names
            held_changed = {t for t in (new_names & old_names)
                            if abs(longs_w.get(t, 0) - prev_longs_weights.get(t, 0)) > 1e-4}

            tc = 0.0
            for t in entered:
                price = _estimate_share_price_for_costs(panel, t, rd_ts)
                tc += longs_w[t] * _trade_cost_pct(price)
            for t in exited:
                price = _estimate_share_price_for_costs(panel, t, rd_ts)
                tc += prev_longs_weights[t] * _trade_cost_pct(price)
            for t in held_changed:
                price = _estimate_share_price_for_costs(panel, t, rd_ts)
                delta = abs(longs_w[t] - prev_longs_weights[t])
                tc += delta * _trade_cost_pct(price)
            if tc > 0:
                daily_rows.append({"date": rd_ts, "ret": -tc})
            prev_longs_weights = longs_w

            hold_win = oot_z[(oot_z["date"] > rd) & (oot_z["date"] <= rd + pd.Timedelta(days=int(HOLD_DAYS * 1.5)))]
            for d, g in hold_win.groupby("date"):
                d_ts = pd.Timestamp(d)
                rg = regime.get(d_ts)
                if rg is None:
                    prior = regime.loc[:d_ts]
                    rg = prior.iloc[-1] if len(prior) else "cash"
                if not longs_w or rg != "bull":
                    daily_rows.append({"date": d_ts, "ret": 0.0})
                    continue
                lret = 0.0
                for t, w in longs_w.items():
                    row = g[g["ticker"] == t]["ret_raw"]
                    if not row.empty and pd.notna(row.iloc[0]):
                        lret += w * float(row.iloc[0])
                daily_rows.append({"date": d_ts, "ret": lret})

    if not daily_rows:
        return pd.DataFrame(columns=["date", "daily_ret"]), rebal_rows
    df = pd.DataFrame(daily_rows)
    df["date"] = pd.to_datetime(df["date"])
    book = df.groupby("date", as_index=False).agg(daily_ret=("ret", "sum"))
    return book.sort_values("date").reset_index(drop=True), rebal_rows


# ---------------------------------------------------------------------------
# Tail-DD gate: worst SPY-regime quarter DD <= 25%
# ---------------------------------------------------------------------------
def tail_dd_metric(book: pd.DataFrame, spy_close: pd.Series) -> dict:
    """For each calendar quarter, classify as 'green'/'red' based on SPY quarterly
    return sign, then compute the strategy's intra-quarter max drawdown.
    Return the worst-quarter DD on red quarters (the punishing side).
    """
    b = book.copy().sort_values("date").set_index("date")["daily_ret"]
    eq = (1.0 + b).cumprod()
    df = pd.DataFrame({"daily_ret": b, "eq": eq})
    df["quarter"] = df.index.to_period("Q")
    spy = spy_close.reindex(df.index).ffill()
    quarter_dds = []
    for q, sub in df.groupby("quarter"):
        if len(sub) < 5:
            continue
        # Quarter DD: peak-to-trough on equity curve normalized at start of quarter
        eq_q = (1.0 + sub["daily_ret"]).cumprod()
        peak = eq_q.cummax()
        dd = ((eq_q - peak) / peak).min()
        spy_q = spy.loc[sub.index]
        spy_qret = float(spy_q.iloc[-1] / spy_q.iloc[0] - 1.0) if len(spy_q) >= 2 else 0.0
        cls = "green" if spy_qret > 0 else "red"
        quarter_dds.append({"quarter": str(q), "cls": cls,
                            "spy_qret_pct": spy_qret * 100,
                            "dd_pct": dd * 100})
    qdf = pd.DataFrame(quarter_dds)
    if qdf.empty:
        return {"worst_red_quarter_dd_pct": float("nan"), "worst_any_quarter_dd_pct": float("nan"),
                "passes_tail_dd_25pct": False, "per_quarter": []}
    red = qdf[qdf["cls"] == "red"]
    worst_red = float(red["dd_pct"].min()) if not red.empty else float("nan")
    worst_any = float(qdf["dd_pct"].min())
    passes = bool(np.isfinite(worst_red) and worst_red >= -25.0)
    return {
        "worst_red_quarter_dd_pct": worst_red,
        "worst_any_quarter_dd_pct": worst_any,
        "passes_tail_dd_25pct": passes,
        "per_quarter": qdf.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Black-Scholes synthetic put hedge for K=3 overlay
# ---------------------------------------------------------------------------
def _bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price (no dividends)."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    from math import log, sqrt, exp
    try:
        from scipy.stats import norm
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    except Exception:
        # Fallback: rough approximation
        return max(K - S, 0.0) * 0.5 + sigma * S * sqrt(T) * 0.4


def _bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    from math import log, sqrt
    try:
        from scipy.stats import norm
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        return float(norm.cdf(d1) - 1.0)
    except Exception:
        return -0.5


def _find_strike_for_delta(S: float, T: float, r: float, sigma: float,
                           target_delta: float = -0.10) -> float:
    """Brute-force search for the strike that gives target put delta."""
    # Put delta ranges from -1 (deep ITM) to 0 (deep OTM).
    # -10 delta puts are OTM (strike < S).
    best_k = S * 0.85
    best_diff = 1e9
    for frac in np.linspace(0.70, 1.00, 31):
        k = S * frac
        d = _bs_put_delta(S, k, T, r, sigma)
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_k = k
    return best_k


def hedge_overlay(book: pd.DataFrame, spy_close: pd.Series, vix: pd.Series,
                  target_delta: float = -0.10,
                  dte_open: int = 90, dte_roll: int = 30,
                  hedge_notional_pct: float = 1.0,
                  r: float = 0.02) -> tuple[pd.DataFrame, dict]:
    """Apply long-put hedge overlay to the equity book.

    Hedge notional = 1.0x of portfolio NAV (full SPY underlying notional covered by puts).
    Cost: pay put premium at open, mark to model BS price daily, sell at roll.

    Returns hedged book + diagnostics.
    """
    b = book.copy().sort_values("date").reset_index(drop=True)
    b["date"] = pd.to_datetime(b["date"])
    dates = b["date"].tolist()
    spy = spy_close.reindex(b["date"]).ffill()
    iv = (vix.reindex(b["date"]).ffill() / 100.0).clip(lower=0.08, upper=0.80)

    # Track one put position at a time
    nav = 1.0  # normalized
    hedged_ret = []
    hedge_log = []
    open_date = None
    open_strike = None
    open_expiry = None
    open_premium_pct = 0.0  # premium paid as fraction of NAV at open
    contracts_per_nav = 0.0  # synthetic: 1 put per X units of NAV
    last_put_value_pct = 0.0

    for i, d in enumerate(dates):
        S = float(spy.iloc[i])
        sigma = float(iv.iloc[i])
        # Underlying daily return (already in book.daily_ret for equity sleeve)
        equity_r = float(b["daily_ret"].iloc[i])

        # Roll condition: open new put if no open OR DTE < dte_roll
        need_roll = False
        if open_date is None:
            need_roll = True
        else:
            dte = (open_expiry - d).days
            if dte <= dte_roll:
                need_roll = True

        roll_pnl_pct = 0.0
        if need_roll:
            # Close existing put (if any) at current value
            if open_date is not None:
                T_remain = max((open_expiry - d).days / 365.25, 1e-6)
                cur_put = _bs_put_price(S, open_strike, T_remain, r, sigma)
                # PnL on the leg = (cur_put - premium_paid_per_contract) * contracts
                # In NAV-normalized terms: (cur_put_pct - open_premium_pct_at_open)
                # but we tracked open_premium_pct as fraction of NAV at open.
                # cur_put_pct in current NAV terms:
                cur_put_pct = (cur_put / S) * hedge_notional_pct  # value as frac of underlying notional
                # Simpler: we model the hedge as a daily mark-to-market PnL stream.
                # Here we just realize the change since last mark.
                roll_pnl_pct = (cur_put / S - last_put_value_pct) * hedge_notional_pct
                last_put_value_pct = 0.0
            # Open new put
            open_date = d
            open_expiry = d + pd.Timedelta(days=dte_open)
            T_new = dte_open / 365.25
            open_strike = _find_strike_for_delta(S, T_new, r, sigma, target_delta)
            premium = _bs_put_price(S, open_strike, T_new, r, sigma)
            # Premium pct of underlying notional = premium / S; cost in NAV terms = (premium/S) * hedge_notional_pct
            premium_pct = (premium / S) * hedge_notional_pct
            # The open is a cash outflow: subtract premium_pct from today's return
            roll_pnl_pct -= premium_pct
            last_put_value_pct = premium / S  # mark
            open_premium_pct = premium_pct
            hedge_log.append({"date": str(d.date()), "spot": S, "strike": open_strike,
                              "expiry": str(open_expiry.date()), "premium_pct_nav": premium_pct,
                              "iv": sigma})

        # Daily MTM PnL of the open put (after any roll above)
        T_remain = max((open_expiry - d).days / 365.25, 1e-6)
        cur_put = _bs_put_price(S, open_strike, T_remain, r, sigma)
        cur_put_pct_underlying = cur_put / S
        # Change in put value per unit underlying notional:
        delta_put_pct = (cur_put_pct_underlying - last_put_value_pct) * hedge_notional_pct
        last_put_value_pct = cur_put_pct_underlying

        total_r = equity_r + roll_pnl_pct + delta_put_pct
        hedged_ret.append({"date": d, "daily_ret": total_r,
                           "equity_ret": equity_r,
                           "hedge_pnl": roll_pnl_pct + delta_put_pct})
    hedged = pd.DataFrame(hedged_ret)

    # Compute hedge cost (annualized drag): sum of all premiums paid over years
    n_years = max((dates[-1] - dates[0]).days / 365.25, 1e-6)
    total_premium = sum(h["premium_pct_nav"] for h in hedge_log)
    drag_pct = (total_premium / n_years) * 100  # rough — doesn't account for value at sale
    # More accurate: realized total hedge PnL
    total_hedge_pnl = hedged["hedge_pnl"].sum()

    diagnostics = {
        "n_puts_opened": len(hedge_log),
        "total_premium_paid_pct_nav": total_premium * 100,
        "total_hedge_pnl_pct": total_hedge_pnl * 100,
        "approx_annual_cost_pct": (total_premium / n_years) * 100,
        "approx_annual_net_drag_pct": ((total_premium - total_hedge_pnl) / n_years) * 100,
    }
    return hedged[["date", "daily_ret"]], diagnostics


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / f"output/macro_picker/megacap_tech_extended_v1_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ext] out_dir={out_dir}")

    print("[ext] loading prices…")
    prices = _load_prices()
    print(f"[ext] price rows={len(prices)}")
    print("[ext] loading VIX…")
    vix = _load_vix()
    spy_reg = _spy_regime(prices)
    vix_ok = _vix_ok(vix)
    regime = _combined_regime(spy_reg, vix_ok)
    spy_close = prices[prices["ticker"] == BENCH_SPY].sort_values("date").set_index("date")["close"]
    spy_ret = spy_close.pct_change().dropna()

    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("megacap_tech_extended_v1")
        mlflow_ok = True
    except Exception as e:
        print(f"[ext] MLflow unavailable: {e}")
        mlflow_ok = False

    variants = []

    # Exp 1: K-sweep K=4/5/6 at mom=60
    for K in [4, 5, 6]:
        variants.append({"name": f"K{K}_mom60", "K": K, "mom_days": 60, "vol_target": False, "hedged": False})

    # Exp 2: momentum-window sweep at K=3 (3m already done as baseline; include for completeness)
    for mom in [20, 60, 120, 252]:
        if mom == 60:
            variants.append({"name": f"K3_mom{mom}_baseline", "K": 3, "mom_days": mom, "vol_target": False, "hedged": False})
        else:
            variants.append({"name": f"K3_mom{mom}", "K": 3, "mom_days": mom, "vol_target": False, "hedged": False})

    # Exp 3: vol-targeted K=3, mom=60
    variants.append({"name": "K3_mom60_voltarget", "K": 3, "mom_days": 60, "vol_target": True, "hedged": False})

    # Exp 5: hedged K=3 mom=60
    variants.append({"name": "K3_mom60_hedged", "K": 3, "mom_days": 60, "vol_target": False, "hedged": True})

    results = []
    panels_cache = {}

    for v in variants:
        name = v["name"]
        print(f"\n[ext] === variant {name} ===")
        if v["mom_days"] not in panels_cache:
            panels_cache[v["mom_days"]] = build_panel_with_window(prices, v["mom_days"])
        panel = panels_cache[v["mom_days"]]

        book, rebal_rows = run_wf(panel, regime, n_long=v["K"], vol_target=v["vol_target"])
        if book.empty:
            print(f"[ext] {name} EMPTY book")
            continue

        hedge_diag = None
        if v["hedged"]:
            book_unhedged = book.copy()
            book_hedged, hedge_diag = hedge_overlay(book, spy_close, vix)
            book = book_hedged
            book_unhedged.to_parquet(out_dir / f"book_{name}_unhedged.parquet", index=False)

        book.to_parquet(out_dir / f"book_{name}.parquet", index=False)
        if rebal_rows:
            pd.DataFrame(rebal_rows).to_parquet(out_dir / f"rebal_{name}.parquet", index=False)

        s = book.set_index("date")["daily_ret"]
        m = _metrics(s)
        strat_df = regime_stratification(book, spy_ret)
        strat_df.to_csv(out_dir / f"regime_strat_{name}.csv", index=False)
        gates = deploy_verdict(m, strat_df, book)
        tail = tail_dd_metric(book, spy_close)
        pd.DataFrame(tail["per_quarter"]).to_csv(out_dir / f"tail_dd_{name}.csv", index=False)
        concentration = name_concentration(rebal_rows, v["K"])
        eq = equity_curve(book)
        eq.to_csv(out_dir / f"equity_{name}.csv", index=False)
        final_equity = float(eq["equity"].iloc[-1])

        row = {
            "variant": name,
            "K": v["K"],
            "mom_days": v["mom_days"],
            "vol_target": v["vol_target"],
            "hedged": v["hedged"],
            "cagr_pct": (m.get("cagr") or 0) * 100,
            "sharpe": m.get("sharpe"),
            "sortino": m.get("sortino"),
            "calmar": m.get("calmar"),
            "max_dd_pct": (m.get("max_dd") or 0) * 100,
            "pf": m.get("pf"),
            "wr_pct": (m.get("wr") or 0) * 100,
            "regime_gap": gates.get("regime_imbalance"),
            "regime_green_sh": gates.get("regime_green_sharpe"),
            "regime_red_sh": gates.get("regime_red_sharpe"),
            "day_conc": gates.get("day_concentration"),
            "n_oot_days": int(s.dropna().shape[0]),
            "passes_hc428_r1": bool(gates["PASSES_DEPLOY_GATES"]),
            "worst_red_quarter_dd_pct": tail["worst_red_quarter_dd_pct"],
            "worst_any_quarter_dd_pct": tail["worst_any_quarter_dd_pct"],
            "passes_tail_dd_25pct": tail["passes_tail_dd_25pct"],
            "final_equity_usd": final_equity,
            "max_name_pct": (max(concentration.values()) * 100) if concentration else 0.0,
        }
        if hedge_diag:
            row["hedge_diag"] = hedge_diag
        results.append(row)

        if mlflow_ok:
            try:
                with mlflow.start_run(run_name=name):
                    mlflow.log_param("K", v["K"])
                    mlflow.log_param("mom_days", v["mom_days"])
                    mlflow.log_param("vol_target", v["vol_target"])
                    mlflow.log_param("hedged", v["hedged"])
                    for k_, v_ in row.items():
                        if isinstance(v_, (int, float)) and v_ is not None and np.isfinite(v_):
                            mlflow.log_metric(k_, float(v_))
                    if hedge_diag:
                        for k_, v_ in hedge_diag.items():
                            if isinstance(v_, (int, float)) and np.isfinite(v_):
                                mlflow.log_metric(f"hedge_{k_}", float(v_))
                    mlflow.log_artifact(str(out_dir / f"book_{name}.parquet"))
                    mlflow.log_artifact(str(out_dir / f"regime_strat_{name}.csv"))
                    mlflow.log_artifact(str(out_dir / f"tail_dd_{name}.csv"))
            except Exception as e:
                print(f"[ext] MLflow log failed for {name}: {e}")

        print(f"[ext] {name}: CAGR={row['cagr_pct']:.1f}% Sharpe={row['sharpe']:.2f} "
              f"Calmar={row['calmar']:.2f} MaxDD={row['max_dd_pct']:.1f}% "
              f"gap={row['regime_gap']} tail_red_dd={row['worst_red_quarter_dd_pct']:.1f}% "
              f"HC428={row['passes_hc428_r1']} tailDD={row['passes_tail_dd_25pct']}")

    # Write combined report
    rep = {
        "strategy": "megacap_tech_extended_v1",
        "test_window": [str(WF_START.date()), str(WF_END.date())],
        "universe": UNIVERSE,
        "variants": results,
        "anchor_usd": 20000,
    }
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"\n[ext] wrote {out_dir}/report.json")
    return out_dir, results


if __name__ == "__main__":
    main()
