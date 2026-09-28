#!/usr/bin/env python3
"""
Haircut Sensitivity Analysis v1
================================
Tests how sensitive the production v3 strategy (GRU regime>0.4 + LGBM 18 features
+ hold-to-expiry bull call spreads) is to the entry haircut assumption.

Motivation: Our pricing model underestimates IV by ~50% on individual options.
For spreads the error partially cancels, but we need to know how much the
Sharpe degrades as the effective haircut rises.

Haircut levels tested: 10%, 15% (current), 20%, 25%, 30%, 40%, 50%

Key specs:
  - Walk-forward LGBM with 18 legacy features (bi-weekly rebalance)
  - Bull call spreads, 3% width, DTE=21, hold to expiry (intrinsic only)
  - Haircut applied on ENTRY only (no exit haircut — automatic exercise at expiry)
  - $645 starting capital, $2.60 commission per spread RT
  - GRU regime filter > 0.4 (regime_scores from regime_predictions_v1.npz)
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
)

BASE = Path("/home/jupiter/Lvl3Quant")
RESULTS_DIR = BASE / "research" / "findings"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "haircut_sensitivity_v1_results.json"

def fprint(*a, **kw):
    print(*a, **kw, flush=True)


# ─── MLflow setup ──────────────────────────────────────────────────────
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen("http://jupiter:5000/", timeout=2)
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable — results saved to JSON only")


# ─── Constants ─────────────────────────────────────────────────────────
UNIVERSE = [
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
]
CAP = 645.0
SPREAD_PCT = 3.0
DTE = 21
TOP_K = 3
HAIRCUT_LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# 18 legacy features
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture", "dn_capture",
    "trend_r2_63d",
]


# ─── Regime data ───────────────────────────────────────────────────────
def load_regime_data():
    """Load GRU regime predictions. Returns dict date_str -> regime_score."""
    path = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
    if not path.exists():
        fprint(f"WARNING: Regime file not found at {path}")
        return {}
    d = np.load(path, allow_pickle=True)
    dates = d["dates"]  # string dates
    scores = d["regime_scores"]  # these are the bull-regime probabilities
    return dict(zip(dates, scores))


# ─── Data download ─────────────────────────────────────────────────────
def download_data():
    import yfinance as yf
    fprint("Downloading price data...")
    tickers = list(set(UNIVERSE + ["SPY", "^VIX"]))
    raw = yf.download(tickers, start="2012-01-01", end="2026-07-26", progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    high = raw["High"] if mi else raw
    low = raw["Low"] if mi else raw
    volume = raw["Volume"] if mi else raw

    vc = "^VIX" if "^VIX" in close.columns else "VIX"
    vix = close[vc].dropna()
    spy = close["SPY"].dropna()

    avail = [c for c in UNIVERSE if c in close.columns and close[c].dropna().shape[0] > 500]
    sc = close[avail].dropna(how="all")
    sh = high[[c for c in avail if c in high.columns]].dropna(how="all")
    sl = low[[c for c in avail if c in low.columns]].dropna(how="all")
    sv = volume[[c for c in avail if c in volume.columns]].dropna(how="all")

    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index).intersection(sl.index)
    fprint(f"Data: {len(ix)} days, {len(avail)} assets")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix]


# ─── Feature engineering ───────────────────────────────────────────────
def compute_features(px, vol_data=None):
    """Compute the 18 legacy features for a single asset at a point in time."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2

    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0

    pk63 = px.iloc[-63:].cummax()
    f["maxdd_63d"] = float(((px.iloc[-63:] / pk63) - 1).min())
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5

    dr = r63[r63 < 0]
    f["sortino_63d"] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0

    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f["calmar_1y"] = cagr / (abs(mdd) + 1e-10)

    up = rets[rets > 0]
    dn = rets[rets < 0]
    f["up_capture"] = float(up.iloc[-63:].mean() / (up.mean() + 1e-10)) if len(up) > 10 else 1.0
    f["dn_capture"] = float(dn.iloc[-63:].mean() / (dn.mean() + 1e-10)) if len(dn) > 10 else 1.0

    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
    else:
        f["trend_r2_63d"] = 0.0

    return f


# ─── Walk-forward LGBM ranking ─────────────────────────────────────────
def build_rankings(sc, sv, rebal_dates, train_periods=12, fwd_days=DTE):
    """Walk-forward LGBM: predict forward returns, rank assets."""
    import lightgbm as lgb

    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[: idx + 1].dropna()
            vol_d = sv[tk].iloc[: idx + 1] if tk in sv.columns else None
            feats = compute_features(px, vol_d)
            if not feats:
                continue
            fi = min(idx + fwd_days, len(sc) - 1)
            feats.update({
                "date": dt,
                "ticker": tk,
                "fwd_ret": float(sc[tk].iloc[fi] / sc[tk].iloc[idx] - 1),
            })
            records.append(feats)

    df = pd.DataFrame(records)
    for c in LEGACY_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[LEGACY_FEATURES] = df[LEGACY_FEATURES].fillna(0.0)

    if len(df) < 100:
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}

    for i in range(train_periods, len(dates)):
        td = dates[max(0, i - train_periods) : i]
        test_date = dates[i]
        tr = df[df["date"].isin(td)]
        te = df[df["date"] == test_date].copy()
        if len(te) < 3 or len(tr) < 50:
            continue

        Xt = np.nan_to_num(tr[LEGACY_FEATURES].values.astype(np.float32))
        yt = tr["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(te[LEGACY_FEATURES].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            te["score"] = m.predict(Xe)
            rankings[test_date] = dict(zip(te["ticker"], te["score"]))
        except Exception:
            continue

    return rankings


# ─── ATR-based premium (simplified, for cost estimation) ───────────────
def compute_atr_series(h, l, c, period=14):
    tr = pd.DataFrame({"hl": h - l, "hc": abs(h - c.shift(1)), "lc": abs(l - c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()


# ─── Simulation (hold to expiry, entry haircut only) ───────────────────
def simulate(rankings, sc, sh, sl, spy, vix, regime_data, haircut=0.15):
    """
    Simulate bull call spread strategy with:
    - GRU regime filter > 0.4
    - Entry haircut applied (variable)
    - Hold to expiry — exit at intrinsic value (no exit haircut)
    - $2.60 commission RT
    """
    atr_d = {
        tk: compute_atr_series(sh[tk], sl[tk], sc[tk])
        for tk in sc.columns
        if tk in sh.columns and tk in sl.columns
    }

    equity = CAP
    trades = []
    eq_curve = [CAP]

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index:
            continue

        # GRU regime filter: require regime_score > 0.4 (bull regime)
        dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
        regime_score = regime_data.get(dt_str, 0.0)
        if regime_score <= 0.4:
            continue

        cv = float(vix.loc[dt])
        scores = rankings[dt]
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            eq_curve.append(equity)
            continue

        n_ent = 0
        for tk, _ in ranked:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3:
                continue

            S = float(sc[tk].loc[dt])
            if pd.isna(S) or S <= 0:
                continue

            # ATR for IV estimation
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S * 0.015

            # Strikes: ATM + 3% width
            K1 = round(S, 2)
            K2 = round(S * (1 + SPREAD_PCT / 100), 2)
            if K2 <= K1:
                K2 = K1 + 0.01

            # Price the spread using standardized pricer with variable haircut
            entry_cost_ps, max_profit_ps = price_bull_call_spread(
                S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv, haircut=haircut
            )

            # Per-contract cost
            entry_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
            max_profit = max_profit_ps * 100 - COMMISSION_RT_SPREAD

            if entry_cost <= 0 or entry_cost > max_pos or entry_cost > equity * 0.40:
                continue

            # Hold to expiry — intrinsic value at expiry date
            di = sc.index.get_loc(dt)
            ei = min(di + DTE, len(sc) - 1)
            Se = float(sc[tk].iloc[ei])

            # Intrinsic value at expiry (no haircut on exit — automatic exercise)
            intrinsic_long = max(Se - K1, 0.0)
            intrinsic_short = max(Se - K2, 0.0)
            exit_value_ps = intrinsic_long - intrinsic_short

            pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

            equity += pnl
            n_ent += 1

            # Determine regime for reporting
            spy_di = spy.index.get_loc(dt) if dt in spy.index else None
            spy_ei = min(spy_di + DTE, len(spy) - 1) if spy_di is not None else None
            spy_ret = float(spy.iloc[spy_ei] / spy.iloc[spy_di] - 1) if spy_di and spy_ei else 0.0

            trades.append({
                "pnl": pnl,
                "win": pnl > 0,
                "entry_cost": entry_cost,
                "regime": "bull" if spy_ret >= 0 else "bear",
                "date": dt_str,
                "ticker": tk,
                "haircut": haircut,
                "regime_score": regime_score,
            })

        eq_curve.append(equity)

    return trades, equity, eq_curve


# ─── Metrics ───────────────────────────────────────────────────────────
def compute_metrics(trades, final_eq, eq_curve, haircut):
    """Compute risk-adjusted metrics using honest equity-based returns."""
    if not trades or len(trades) < 10:
        return None

    n = len(trades)
    wr = sum(1 for t in trades if t["win"]) / n * 100
    pnls = [t["pnl"] for t in trades]
    avg_pnl = np.mean(pnls)
    entry_costs = [t["entry_cost"] for t in trades]
    avg_cost = np.mean(entry_costs)

    # Monthly aggregation — honest equity-based returns
    tdf = pd.DataFrame(trades)
    tdf["_date"] = pd.to_datetime(tdf["date"])
    tdf["_month"] = tdf["_date"].dt.to_period("M")

    equity_track = CAP
    month_start_eq = {}
    month_pnl = {}
    cur_month = None
    for _, row in tdf.iterrows():
        m = row["_month"]
        if m != cur_month:
            month_start_eq[m] = equity_track
            cur_month = m
            month_pnl[m] = 0
        month_pnl[m] += row["pnl"]
        equity_track += row["pnl"]

    months = sorted(month_pnl.keys())
    mr = np.array([month_pnl[m] / max(month_start_eq[m], 1.0) for m in months])
    ny = max(len(mr) / 12, 0.5)

    sharpe = (np.mean(mr) * 12) / (np.std(mr) * np.sqrt(12) + 1e-10) if len(mr) > 3 else 0
    cagr = (final_eq / CAP) ** (1 / ny) - 1

    eq = np.array(eq_curve)
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / (pk + 1e-10)).min())

    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gp / (gl + 1e-10)

    # Sortino
    dr_monthly = mr[mr < 0]
    sortino = (np.mean(mr) * 12) / (np.std(dr_monthly) * np.sqrt(12) + 1e-10) if len(dr_monthly) > 2 else sharpe

    # Permutation test (500 trials)
    rs = np.mean(mr) / (np.std(mr) + 1e-10)
    pp = sum(
        1 for _ in range(500)
        if np.mean(mr * np.random.choice([-1, 1], len(mr))) / (np.std(mr) + 1e-10) >= rs
    ) / 500

    return {
        "haircut_pct": round(haircut * 100),
        "n_trades": n,
        "wr": round(wr, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "cagr_pct": round(cagr * 100, 1),
        "mdd_pct": round(mdd * 100, 1),
        "pf": round(pf, 2),
        "final_equity": round(final_eq, 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "avg_cost_per_spread": round(avg_cost, 2),
        "total_cost_dollar_impact": round(avg_cost * n, 2),
        "perm_p": round(pp, 3),
        "pass_perm": pp < 0.05,
    }


# ─── Main ──────────────────────────────────────────────────────────────
def main():
    t0 = datetime.now()
    fprint(f"Haircut Sensitivity Analysis v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Strategy: GRU regime>0.4 + WF LGBM (18 features) + Bull Call Spread 3%/DTE21")
    fprint(f"Hold-to-expiry (intrinsic only). Entry haircut varied. $645 starting capital.")
    fprint(f"Commission: ${COMMISSION_RT_SPREAD:.2f}/spread RT")
    fprint(f"Haircut levels: {[f'{h*100:.0f}%' for h in HAIRCUT_LEVELS]}")
    fprint("=" * 80)

    # Load regime data
    fprint("\nLoading GRU regime predictions...")
    regime_data = load_regime_data()
    fprint(f"  Loaded {len(regime_data)} daily regime scores")
    if regime_data:
        scores_arr = np.array(list(regime_data.values()))
        fprint(f"  Score range: {scores_arr.min():.3f} - {scores_arr.max():.3f}")
        fprint(f"  Dates > 0.4 threshold: {(scores_arr > 0.4).sum()} ({(scores_arr > 0.4).mean()*100:.1f}%)")

    # Download data
    sc, sh, sl, sv, spy, vix = download_data()

    # Build walk-forward rankings (shared across all haircut levels)
    fprint("\nBuilding walk-forward LGBM rankings (bi-weekly rebalance, 12-period lookback)...")
    bd_bw = pd.DatetimeIndex(sc.index.to_series().resample("2W-FRI").last().dropna().values)
    rankings = build_rankings(sc, sv, bd_bw, train_periods=12, fwd_days=DTE)
    fprint(f"  Generated rankings for {len(rankings)} dates")

    if not rankings:
        fprint("ERROR: No rankings generated. Cannot proceed.")
        return

    # MLflow experiment
    if MLFLOW_OK:
        try:
            mlflow.set_experiment("haircut_sensitivity_v1")
        except Exception as e:
            fprint(f"MLflow experiment setup failed: {e}")

    # Run simulation for each haircut level
    fprint("\n" + "=" * 80)
    fprint(f"{'Haircut':>8} | {'Sharpe':>7} | {'Sortino':>8} | {'WR':>6} | {'PF':>6} | {'CAGR':>7} | {'MDD':>7} | {'Trades':>7} | {'AvgPnL':>8} | {'AvgCost':>8}")
    fprint("-" * 100)

    all_results = {}
    viable_cutoff = None

    for hc in HAIRCUT_LEVELS:
        trades, final_eq, eq_curve = simulate(rankings, sc, sh, sl, spy, vix, regime_data, haircut=hc)
        metrics = compute_metrics(trades, final_eq, eq_curve, hc)

        if metrics is None:
            fprint(f"  {hc*100:5.0f}%   | INSUFFICIENT DATA")
            all_results[f"haircut_{int(hc*100)}pct"] = {"haircut_pct": int(hc * 100), "error": "insufficient_data"}
            continue

        all_results[f"haircut_{int(hc*100)}pct"] = metrics

        viable = "  OK" if metrics["sharpe"] >= 1.0 else " <1!"
        fprint(
            f"  {hc*100:5.0f}%  | {metrics['sharpe']:7.2f} | {metrics['sortino']:8.2f} | "
            f"{metrics['wr']:5.1f}% | {metrics['pf']:5.2f} | {metrics['cagr_pct']:6.1f}% | "
            f"{metrics['mdd_pct']:6.1f}% | {metrics['n_trades']:7d} | "
            f"${metrics['avg_pnl_per_trade']:7.2f} | ${metrics['avg_cost_per_spread']:7.2f}{viable}"
        )

        # Track where Sharpe drops below 1.0
        if metrics["sharpe"] < 1.0 and viable_cutoff is None:
            viable_cutoff = hc

        # Log to MLflow
        if MLFLOW_OK:
            try:
                with mlflow.start_run(run_name=f"haircut_{int(hc*100)}pct"):
                    mlflow.log_params({
                        "haircut_pct": int(hc * 100),
                        "spread_width_pct": SPREAD_PCT,
                        "dte": DTE,
                        "starting_capital": CAP,
                        "commission_rt": COMMISSION_RT_SPREAD,
                        "regime_threshold": 0.4,
                        "top_k": TOP_K,
                        "strategy": "GRU_regime_LGBM18_BCS_holdexpiry",
                    })
                    mlflow.log_metrics({
                        "sharpe": metrics["sharpe"],
                        "sortino": metrics["sortino"],
                        "win_rate": metrics["wr"],
                        "profit_factor": metrics["pf"],
                        "cagr_pct": metrics["cagr_pct"],
                        "max_drawdown_pct": metrics["mdd_pct"],
                        "n_trades": metrics["n_trades"],
                        "avg_pnl_per_trade": metrics["avg_pnl_per_trade"],
                        "avg_cost_per_spread": metrics["avg_cost_per_spread"],
                        "final_equity": metrics["final_equity"],
                        "perm_p_value": metrics["perm_p"],
                    })
            except Exception as e:
                fprint(f"  MLflow logging failed for {hc*100:.0f}%: {e}")

    # Summary
    fprint("\n" + "=" * 80)
    fprint("SUMMARY")
    fprint("=" * 80)

    # Dollar impact table
    fprint(f"\n{'Haircut':>8} | {'Eff. Cost/Spread':>17} | {'Cost vs 15%':>12} | {'Sharpe':>7} | {'Sharpe vs 15%':>14}")
    fprint("-" * 70)

    baseline_cost = None
    baseline_sharpe = None
    for hc in HAIRCUT_LEVELS:
        key = f"haircut_{int(hc*100)}pct"
        m = all_results.get(key, {})
        if "error" in m:
            continue
        cost = m.get("avg_cost_per_spread", 0)
        sharpe = m.get("sharpe", 0)
        if hc == 0.15:
            baseline_cost = cost
            baseline_sharpe = sharpe
        cost_delta = f"+${cost - baseline_cost:.2f}" if baseline_cost is not None else "baseline"
        sharpe_delta = f"{sharpe - baseline_sharpe:+.2f}" if baseline_sharpe is not None else "baseline"
        if hc == 0.15:
            cost_delta = "baseline"
            sharpe_delta = "baseline"
        fprint(f"  {hc*100:5.0f}%  | ${cost:16.2f} | {cost_delta:>12} | {sharpe:7.2f} | {sharpe_delta:>14}")

    if viable_cutoff is not None:
        fprint(f"\n*** Strategy becomes non-viable (Sharpe < 1.0) at {viable_cutoff*100:.0f}% haircut ***")
    else:
        fprint(f"\n*** Strategy remains viable (Sharpe >= 1.0) at ALL tested haircut levels ***")

    # Marginal Sharpe per 5% haircut increase
    sharpes = []
    for hc in HAIRCUT_LEVELS:
        key = f"haircut_{int(hc*100)}pct"
        m = all_results.get(key, {})
        if "error" not in m:
            sharpes.append((hc, m.get("sharpe", 0)))

    if len(sharpes) >= 2:
        # Linear regression: Sharpe vs haircut
        hc_arr = np.array([s[0] for s in sharpes])
        sh_arr = np.array([s[1] for s in sharpes])
        slope, intercept, r_val, _, _ = stats.linregress(hc_arr, sh_arr)
        fprint(f"\nSharpe sensitivity: {slope:.2f} per 1.0 haircut (R²={r_val**2:.3f})")
        fprint(f"  i.e., each +5% haircut costs ~{abs(slope*0.05):.2f} Sharpe")
        if slope < 0:
            breakeven_hc = -intercept / slope if slope != 0 else float("inf")
            sharpe1_hc = (1.0 - intercept) / slope if slope != 0 else float("inf")
            fprint(f"  Linear extrapolation: Sharpe=1.0 at ~{sharpe1_hc*100:.0f}% haircut")
            fprint(f"  Linear extrapolation: Sharpe=0.0 at ~{breakeven_hc*100:.0f}% haircut")

    # Save results
    all_results["_meta"] = {
        "timestamp": t0.isoformat(),
        "strategy": "GRU regime>0.4 + WF LGBM 18 features + BCS 3% DTE21 hold-to-expiry",
        "starting_capital": CAP,
        "commission_rt": COMMISSION_RT_SPREAD,
        "regime_threshold": 0.4,
        "universe": UNIVERSE,
        "viable_cutoff_pct": viable_cutoff * 100 if viable_cutoff else None,
        "runtime_seconds": (datetime.now() - t0).total_seconds(),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # Adversarial validation on baseline (15%) trades
    fprint("\nRunning adversarial validation on 15% baseline...")
    try:
        from research.tools.adversarial_validator import validate_trades
        baseline_trades_raw, baseline_eq, baseline_curve = simulate(
            rankings, sc, sh, sl, spy, vix, regime_data, haircut=0.15
        )
        if baseline_trades_raw and len(baseline_trades_raw) >= 10:
            result = validate_trades(
                trades=baseline_trades_raw,
                initial_capital=CAP,
                spy_prices=spy,
            )
            result.print_summary()
    except Exception as e:
        fprint(f"Adversarial validation skipped: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
