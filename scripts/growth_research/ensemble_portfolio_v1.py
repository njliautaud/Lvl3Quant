"""
ENSEMBLE PORTFOLIO OPTIMIZER v1
================================
Combines validated strategy building blocks into an optimal allocation.

Building blocks:
  1. Risk Parity 3x     — UPRO/TMF/UGL/DBC inverse-vol monthly rebalance
  2. Cross-Asset Momentum — 13 ETFs, Variant D (best Sharpe among tested)
  3. SPY + Risk Overlay  — SPY with drawdown-predictor position scaling
  4. Stock Picker Proxy  — Top-quintile equal-weight momentum (ML proxy)
  5. Income Proxy        — 6% annualized constant + stress haircuts

Constraints:
  - HC #428 R1: regime gap ≤ 0.50
  - Walk-forward: train on first 60%, test on last 40%
  - Commission-free per HC #694
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import permutation_test
from sklearn.preprocessing import StandardScaler
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── PATHS ───────────────────────────────────────────────────────────────────
OUT_DIR = "/home/jupiter/Lvl3Quant/output/ensemble_portfolio_v1"
BARBELL_D_CSV = "/home/jupiter/Lvl3Quant/output/barbell_strategy_v1/variant_D_nav.csv"
os.makedirs(OUT_DIR, exist_ok=True)

START = "2015-01-01"
END   = "2026-07-01"
TRAIN_FRAC = 0.60   # walk-forward split

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def ann_sharpe(r, rf=0.0):
    """Annualised Sharpe from daily returns."""
    r = np.asarray(r, dtype=float)
    ex = r - rf / 252
    if ex.std() == 0:
        return 0.0
    return np.sqrt(252) * ex.mean() / ex.std()

def ann_sortino(r, rf=0.0):
    r = np.asarray(r, dtype=float)
    ex = r - rf / 252
    neg = ex[ex < 0]
    if len(neg) == 0 or neg.std() == 0:
        return np.inf
    return np.sqrt(252) * ex.mean() / neg.std()

def cagr(r):
    r = np.asarray(r, dtype=float)
    cum = np.prod(1 + r)
    years = len(r) / 252
    if years == 0:
        return 0.0
    return cum ** (1 / years) - 1

def max_drawdown(r):
    r = np.asarray(r, dtype=float)
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    return dd.min()

def calmar(r):
    c = cagr(r)
    md = max_drawdown(r)
    if md == 0:
        return np.inf
    return c / abs(md)

def metrics(r, label=""):
    r = np.asarray(r, dtype=float)
    return {
        "Label": label,
        "CAGR_%": round(cagr(r) * 100, 2),
        "AnnVol_%": round(r.std() * np.sqrt(252) * 100, 2),
        "Sharpe": round(ann_sharpe(r), 3),
        "Sortino": round(ann_sortino(r), 3),
        "MaxDD_%": round(max_drawdown(r) * 100, 2),
        "Calmar": round(calmar(r), 3),
    }

def regime_classify(spy_returns):
    """
    Day classification: GREEN/RED/FLAT using SPY close-to-close returns.
    GREEN  >  +0.2%
    RED    < -0.2%
    FLAT   else
    """
    labels = pd.Series("FLAT", index=spy_returns.index)
    labels[spy_returns >  0.002] = "GREEN"
    labels[spy_returns < -0.002] = "RED"
    return labels

def regime_sharpes(r, spy_or_regimes):
    """
    Sharpe per regime bucket.
    `spy_or_regimes` can be either:
      - a pd.Series of SPY returns (will be classified internally)
      - a pd.Series of labels "GREEN"/"RED"/"FLAT"
    `r` is the portfolio return series (same index).
    """
    # Detect if we received raw returns or labels
    if spy_or_regimes.dtype == object or spy_or_regimes.dtype.name == "object":
        regimes = spy_or_regimes
    else:
        regimes = regime_classify(spy_or_regimes)

    # Align index
    r_s = pd.Series(np.asarray(r), index=pd.DatetimeIndex(r.index) if hasattr(r, 'index') else regimes.index)
    common = r_s.index.intersection(regimes.index)
    r_s   = r_s.reindex(common)
    reg_s = regimes.reindex(common)

    out = {}
    for reg in ["GREEN", "RED", "FLAT"]:
        mask = reg_s == reg
        sub = r_s[mask].dropna()
        out[reg] = ann_sharpe(sub) if len(sub) > 5 else np.nan
        out[f"n_{reg}"] = int(mask.sum())
    # gap metric: |green - red| / max(|green|, |red|)
    g  = out.get("GREEN") or 0
    rd = out.get("RED")   or 0
    if g is np.nan: g = 0
    if rd is np.nan: rd = 0
    denom = max(abs(g), abs(rd))
    out["regime_gap"] = abs(g - rd) / denom if denom > 0 else 0
    out["passes_R1"] = bool(out["regime_gap"] <= 0.50)
    return out

def fetch_prices(tickers, start, end):
    """Download adjusted close prices via yfinance."""
    print(f"  Downloading {tickers}...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    prices = prices.ffill().dropna(how="all")
    return prices

# ─── STRATEGY 1: RISK PARITY 3x ─────────────────────────────────────────────

def build_risk_parity_3x(start, end):
    """
    UPRO/TMF/UGL/DBC: inverse-vol monthly rebalance.
    Weights ~ 1/vol (20-day rolling).
    ~22% CAGR validated standalone.
    """
    print("[1] Building Risk Parity 3x returns...")
    tickers = ["UPRO", "TMF", "UGL", "DBC"]
    # UPRO inception 2009-06, TMF 2009-04, UGL 2008-12
    # Use 2015 start as requested
    px = fetch_prices(tickers, start, end)
    # drop any ticker with >5% NaN
    px = px.dropna(thresh=int(len(px)*0.95), axis=1)
    px = px.ffill().dropna()

    rets = px.pct_change().dropna()

    # Monthly rebalance: at start of each month compute inverse-vol weights
    port_rets = []

    # Build rebal dates: first trading day of each month
    rebal_dates = rets.resample("MS").first().index.tolist()

    for i, d in enumerate(rebal_dates):
        # vol lookback: 20 trading days prior to rebal
        hist = rets.loc[:d].iloc[-20:]
        if len(hist) < 5:
            continue
        vol = hist.std()
        if (vol == 0).any():
            continue
        inv_vol = 1.0 / vol
        w = inv_vol / inv_vol.sum()

        # apply weights from this rebal to the next
        next_d = rebal_dates[i+1] if i+1 < len(rebal_dates) else rets.index[-1]
        period = rets.loc[d:next_d].iloc[:-1]
        if len(period) == 0:
            continue
        pr = (period * w.values).sum(axis=1)
        port_rets.append(pr)

    if not port_rets:
        return pd.Series(dtype=float)

    series = pd.concat(port_rets).sort_index()
    # deduplicate
    series = series[~series.index.duplicated(keep='first')]
    print(f"    RP3x: {len(series)} days, CAGR={cagr(series)*100:.1f}%, Sharpe={ann_sharpe(series):.2f}")
    return series


# ─── STRATEGY 2: CROSS-ASSET MOMENTUM (Variant D) ────────────────────────────

def build_cross_asset_momentum(start, end):
    """
    Replicate Variant D cross-asset momentum.
    13 ETFs across 5 asset classes: equities, bonds, commodities, real estate, USD.
    Signal: 12-1 month momentum. Go long top tercile, short bottom tercile.
    Monthly rebalance.
    """
    print("[2] Building Cross-Asset Momentum (Variant D) returns...")

    tickers = ["SPY", "QQQ", "IWM", "EFA", "EEM",
               "TLT", "IEF", "HYG",
               "GLD", "SLV", "USO", "DBA",
               "VNQ"]
    px = fetch_prices(tickers, "2010-01-01", end)  # need history for signal
    px = px.ffill().dropna(thresh=int(len(px)*0.90), axis=1)
    px = px.ffill().dropna()

    rets = px.pct_change().dropna()

    port_rets = []
    rebal_dates = rets.resample("MS").first().index.tolist()

    for i, d in enumerate(rebal_dates):
        # 12-1 month momentum
        hist_12 = rets.loc[:d]
        if len(hist_12) < 260:
            continue
        # 12m return minus last 1m (skip last month)
        ret_12m = px.loc[:d].iloc[-252:].iloc[[0, -1]].pct_change().iloc[-1]
        ret_1m  = px.loc[:d].iloc[-21:].iloc[[0, -1]].pct_change().iloc[-1]
        mom = ret_12m - ret_1m
        mom = mom.dropna()
        if len(mom) < 3:
            continue
        n = max(1, len(mom) // 3)
        longs  = mom.nlargest(n).index
        shorts = mom.nsmallest(n).index

        next_d = rebal_dates[i+1] if i+1 < len(rebal_dates) else rets.index[-1]
        period = rets.loc[d:next_d].iloc[:-1]
        if len(period) == 0:
            continue

        long_ret  = period[longs].mean(axis=1)
        short_ret = period[shorts].mean(axis=1)
        pr = 0.5 * long_ret - 0.5 * short_ret  # market-neutral variant D
        port_rets.append(pr)

    series = pd.concat(port_rets).sort_index()
    series = series[~series.index.duplicated(keep='first')]
    # Filter to requested start
    series = series.loc[start:]
    print(f"    XMom: {len(series)} days, CAGR={cagr(series)*100:.1f}%, Sharpe={ann_sharpe(series):.2f}")
    return series


# ─── STRATEGY 3: SPY + RISK OVERLAY ─────────────────────────────────────────

def build_spy_risk_overlay(start, end):
    """
    SPY with drawdown-predictor risk overlay.
    Proxy: when SPY 21-day realized vol > 80th percentile rolling → scale to 0.5.
    When SPY 63-day return < -10% → scale to 0.25.
    Validated improvement: Sharpe 0.89→1.71, MaxDD -33.7%→-11.9%.
    """
    print("[3] Building SPY + Risk Overlay returns...")
    px = fetch_prices(["SPY"], "2010-01-01", end)
    spy_px = px["SPY"] if "SPY" in px.columns else px.iloc[:, 0]
    spy_ret = spy_px.pct_change().dropna()

    # Rolling 21-day vol
    vol21 = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_pct = vol21.rolling(252).rank(pct=True)

    # Rolling 63-day return
    ret63 = spy_px.pct_change(63)

    # Position scale
    scale = pd.Series(1.0, index=spy_ret.index)
    scale[vol_pct > 0.80] = 0.50
    scale[(ret63 < -0.10)] = 0.25
    scale[(ret63 < -0.20)] = 0.10

    scaled_ret = spy_ret * scale.reindex(spy_ret.index).ffill()
    scaled_ret = scaled_ret.loc[start:]
    print(f"    SPY+RO: {len(scaled_ret)} days, CAGR={cagr(scaled_ret)*100:.1f}%, Sharpe={ann_sharpe(scaled_ret):.2f}")
    return scaled_ret


# ─── STRATEGY 4: STOCK PICKER PROXY ─────────────────────────────────────────

def build_stock_picker_proxy(start, end):
    """
    Proxy for Stock Predictor v3 (LGBM+XGB, 193 stocks).
    Method: equal-weight top quintile by 3-month momentum, monthly rebalance.
    Use a representative liquid universe.
    Note: This is a PROXY — actual ML model has Sharpe 0.42 with corrected costs.
    """
    print("[4] Building Stock Picker Proxy (momentum quintile) returns...")

    # 40 liquid large-caps as proxy universe
    tickers = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JNJ",
        "JPM", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "INTC", "NFLX",
        "ADBE", "CRM", "PFE", "TMO", "ABBV", "KO", "PEP", "MRK", "WMT", "BAC",
        "VZ", "T", "XOM", "CVX", "LLY", "COST", "MCD", "NKE", "ACN", "TXN",
        "QCOM", "AVGO"
    ]
    px = fetch_prices(tickers, "2012-01-01", end)
    px = px.ffill().dropna(thresh=int(len(px)*0.70), axis=1)
    px = px.ffill().bfill()

    rets = px.pct_change().dropna()
    port_rets = []
    rebal_dates = rets.resample("MS").first().index.tolist()

    for i, d in enumerate(rebal_dates):
        hist = px.loc[:d]
        if len(hist) < 65:
            continue
        # 3-month momentum
        mom3 = hist.iloc[-63:].iloc[[0, -1]].pct_change().iloc[-1]
        mom3 = mom3.dropna()
        q80 = mom3.quantile(0.80)
        top = mom3[mom3 >= q80].index.tolist()
        if not top:
            continue

        next_d = rebal_dates[i+1] if i+1 < len(rebal_dates) else rets.index[-1]
        period = rets.loc[d:next_d].iloc[:-1]
        if len(period) == 0:
            continue

        cols = [t for t in top if t in period.columns]
        if not cols:
            continue
        pr = period[cols].mean(axis=1)
        port_rets.append(pr)

    series = pd.concat(port_rets).sort_index()
    series = series[~series.index.duplicated(keep='first')]
    series = series.loc[start:]
    print(f"    StockPicker: {len(series)} days, CAGR={cagr(series)*100:.1f}%, Sharpe={ann_sharpe(series):.2f}")
    return series


# ─── STRATEGY 5: INCOME PROXY ────────────────────────────────────────────────

def build_income_proxy(index, spy_ret=None):
    """
    Income proxy: ~6% annualized constant return.
    Stress: during SPY drawdowns >15%, apply -30% DD haircut proportionally.

    NOTE: This is an approximation. Real wheel/condor income is not backtestable
    with public data. Actual IC Condors showed +$5,180 in 9 days (paper, one regime).
    Validated CAGR honest estimate: 6-8%.
    """
    print("[5] Building Income Proxy returns...")
    daily_ret = (1 + 0.065) ** (1/252) - 1  # 6.5% base

    income = pd.Series(daily_ret, index=index)

    if spy_ret is not None:
        spy_aligned = spy_ret.reindex(index).fillna(0)
        spy_cum = (1 + spy_aligned).cumprod()
        spy_peak = spy_cum.expanding().max()
        spy_dd = (spy_cum - spy_peak) / spy_peak

        # During SPY drawdown > 15%, scale income down (options lose in big crashes)
        stress_scale = pd.Series(1.0, index=index)
        stress_scale[spy_dd < -0.15] = 0.3
        stress_scale[spy_dd < -0.25] = 0.0  # -30% income, rough 0 in severe crash
        income = income * stress_scale

    print(f"    Income: {len(income)} days, CAGR={cagr(income)*100:.1f}%, Sharpe={ann_sharpe(income):.2f}")
    return income


# ─── PORTFOLIO CONSTRUCTION ───────────────────────────────────────────────────

def build_portfolio(weights, strat_df):
    """Combine strategies (DataFrame, rows=dates, cols=strategies) with given weights."""
    w = np.array(weights)
    if isinstance(strat_df, pd.DataFrame):
        port = (strat_df * w).sum(axis=1)
    else:
        # numpy array: (n_days, n_strats)
        port = pd.Series((strat_df * w).sum(axis=1))
    return port

def max_sharpe_weights(strat_df, bounds=None):
    """MVO: maximize Sharpe ratio.
    Income proxy (last col) capped at 15% — its near-zero realized vol is a proxy artifact.
    """
    n = strat_df.shape[1]
    mu = strat_df.mean().values * 252
    cov = strat_df.cov().values * 252

    def neg_sharpe(w):
        p_ret = w @ mu
        p_vol = np.sqrt(w @ cov @ w)
        return -p_ret / p_vol if p_vol > 0 else 0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if bounds is None:
        # Income proxy (index 4) capped at 15% — not backtestable across regimes
        bounds = [(0.05, 0.60)] * (n-1) + [(0.02, 0.15)]

    best = None
    best_val = np.inf
    for _ in range(50):
        x0 = np.random.dirichlet(np.ones(n))
        res = minimize(neg_sharpe, x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-10})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best = res.x
    return best if best is not None else np.ones(n) / n

def risk_parity_weights(strat_df):
    """Equal risk contribution."""
    n = strat_df.shape[1]
    cov = strat_df.cov().values * 252

    def risk_contr_obj(w):
        port_var = w @ cov @ w
        mrc = cov @ w  # marginal risk contribution
        rc = w * mrc / port_var  # risk contributions
        target = np.ones(n) / n
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    # Income proxy (index 4) capped at 15% — vol is near-zero by construction
    bounds = [(0.02, 0.80)] * (n-1) + [(0.02, 0.15)]
    best = None
    best_val = np.inf
    for _ in range(30):
        x0 = np.random.dirichlet(np.ones(n))
        res = minimize(risk_contr_obj, x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 500})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best = res.x
    return best if best is not None else np.ones(n) / n

def min_variance_weights(strat_df):
    """Minimum variance portfolio. Income proxy capped at 15%."""
    n = strat_df.shape[1]
    cov = strat_df.cov().values * 252

    def port_var(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.05, 0.60)] * (n-1) + [(0.02, 0.15)]
    best = None
    best_val = np.inf
    for _ in range(30):
        x0 = np.random.dirichlet(np.ones(n))
        res = minimize(port_var, x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 500})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best = res.x
    return best if best is not None else np.ones(n) / n


# ─── RISK OVERLAY ────────────────────────────────────────────────────────────

def apply_risk_overlay(port_ret, spy_ret, cash_ret=None):
    """
    Portfolio-level risk overlay:
      Normal:   100% target allocation
      Elevated: 80% target + 20% cash/TLT (SPY vol >60th pct OR DD > 8%)
      High:     50% target + 50% cash/TLT (SPY vol >80th pct OR DD > 15%)
    """
    if cash_ret is None:
        # Use 3-month T-Bill proxy: ~4.5% annualized
        cash_ret = pd.Series((1 + 0.045) ** (1/252) - 1, index=port_ret.index)

    spy_aligned = spy_ret.reindex(port_ret.index).ffill()
    vol21 = spy_aligned.rolling(21).std() * np.sqrt(252)
    vol_pct = vol21.rolling(252).rank(pct=True).fillna(0.5)

    spy_cum = (1 + spy_aligned).cumprod()
    spy_peak = spy_cum.expanding().max()
    spy_dd = (spy_cum - spy_peak) / spy_peak

    # Overlay mode
    mode = pd.Series("normal", index=port_ret.index)
    mode[(vol_pct > 0.60) | (spy_dd < -0.08)] = "elevated"
    mode[(vol_pct > 0.80) | (spy_dd < -0.15)] = "high"

    equity_frac = pd.Series(1.0, index=port_ret.index)
    equity_frac[mode == "elevated"] = 0.80
    equity_frac[mode == "high"]     = 0.50

    cash_aligned = cash_ret.reindex(port_ret.index).ffill().fillna(0)
    overlay_ret = port_ret * equity_frac + cash_aligned * (1 - equity_frac)
    return overlay_ret, mode


# ─── WALK-FORWARD VALIDATION ─────────────────────────────────────────────────

def walk_forward_optimize(strat_df, spy_ret, labels, train_frac=0.60):
    """
    Split at train_frac. Optimize weights on train, evaluate on test.
    Returns test-set portfolio returns for each method.
    """
    n = len(strat_df)
    split_idx = int(n * train_frac)
    train = strat_df.iloc[:split_idx]
    test  = strat_df.iloc[split_idx:]

    print(f"\n  Walk-forward split: Train {train.index[0].date()} → {train.index[-1].date()}")
    print(f"                      Test  {test.index[0].date()} → {test.index[-1].date()}")

    # Optimize on train
    w_mvo  = max_sharpe_weights(train)
    w_rp   = risk_parity_weights(train)
    w_mv   = min_variance_weights(train)

    # Fixed allocations (RP3x / XMom / SPY+RO / StockPick / Income)
    fixed_allocs = {
        "50/20/15/10/5":   [0.50, 0.20, 0.15, 0.10, 0.05],
        "40/25/15/10/10":  [0.40, 0.25, 0.15, 0.10, 0.10],
        "40/20/20/10/10":  [0.40, 0.20, 0.20, 0.10, 0.10],
        "30/25/20/15/10":  [0.30, 0.25, 0.20, 0.15, 0.10],
        "35/15/20/20/10":  [0.35, 0.15, 0.20, 0.20, 0.10],
        "Equal":           [0.20, 0.20, 0.20, 0.20, 0.20],
    }

    spy_test = spy_ret.reindex(test.index)

    results = {}

    for name, w in [("MaxSharpe_MVO", w_mvo),
                    ("RiskParity_ERC", w_rp),
                    ("MinVariance", w_mv)]:
        port = build_portfolio(w, test)
        port_ov, mode = apply_risk_overlay(port, spy_test)
        m = metrics(port.values, f"{name} (no overlay)")
        m_ov = metrics(port_ov.values, f"{name} (with overlay)")
        m["weights"] = {lab: round(float(ww), 4) for lab, ww in zip(labels, w)}
        m_ov["weights"] = m["weights"]
        m["regime"] = regime_sharpes(pd.Series(port.values, index=test.index), spy_test.reindex(test.index))
        m_ov["regime"] = regime_sharpes(pd.Series(port_ov.values, index=test.index), spy_test.reindex(test.index))
        results[name] = {"returns": port, "returns_overlay": port_ov, "metrics": m, "metrics_overlay": m_ov, "weights": w}

    for name, w in fixed_allocs.items():
        port = build_portfolio(w, test)
        port_ov, mode = apply_risk_overlay(port, spy_test)
        m = metrics(port.values, name)
        m_ov = metrics(port_ov.values, f"{name} (overlay)")
        m["weights"] = {lab: round(float(ww), 4) for lab, ww in zip(labels, w)}
        m_ov["weights"] = m["weights"]
        m["regime"] = regime_sharpes(pd.Series(port.values, index=test.index), spy_test.reindex(test.index))
        m_ov["regime"] = regime_sharpes(pd.Series(port_ov.values, index=test.index), spy_test.reindex(test.index))
        results[name] = {"returns": port, "returns_overlay": port_ov, "metrics": m, "metrics_overlay": m_ov, "weights": w}

    return results, test


# ─── PERMUTATION TEST ────────────────────────────────────────────────────────

def permutation_sharpe_test(port_ret, strat_df=None, n_perm=1000):
    """
    Permutation test on the WEIGHTS, not the returns.
    If strat_df is provided: draw random weight vectors and build portfolios.
    This tests whether the optimized weights beat chance allocation.
    Otherwise: block-bootstrap (moving blocks of 21 days) to preserve autocorrelation.
    """
    observed = ann_sharpe(port_ret)
    rng = np.random.default_rng(42)
    null_sharpes = []

    if strat_df is not None:
        # Random weight permutation test
        n = strat_df.shape[1]
        for _ in range(n_perm):
            w_rand = rng.dirichlet(np.ones(n))
            port_rand = build_portfolio(w_rand, strat_df)
            null_sharpes.append(ann_sharpe(port_rand.values))
    else:
        # Moving block bootstrap (block size=21 trading days)
        arr = np.asarray(port_ret).copy()
        n   = len(arr)
        block = 21
        for _ in range(n_perm):
            indices = []
            while len(indices) < n:
                start = rng.integers(0, max(1, n - block))
                indices.extend(range(start, min(start + block, n)))
            bootstrapped = arr[indices[:n]]
            null_sharpes.append(ann_sharpe(bootstrapped))

    null_arr = np.array(null_sharpes)
    p_val = np.mean(null_arr >= observed)
    return observed, null_arr, p_val


# ─── YEAR-BY-YEAR RETURNS ────────────────────────────────────────────────────

def yearly_returns(ret_series):
    """Annual returns by calendar year."""
    return ret_series.resample("YE").apply(lambda x: float((1 + x).prod() - 1)) * 100


# ─── TRAVEL INCOME CALCULATION ───────────────────────────────────────────────

def travel_income_number(port_ret, target_monthly_low=3000, target_monthly_high=5000, max_dd_threshold=-0.15):
    """
    How much capital needed to generate $3-5K/month with MaxDD < 15%?
    """
    c = cagr(port_ret)
    md = max_drawdown(port_ret)

    # Check MaxDD constraint
    passes = md > max_dd_threshold  # md is negative

    if c <= 0:
        return {"error": "Strategy has negative CAGR", "passes_maxdd": passes}

    # Monthly return
    monthly_ret = (1 + c) ** (1/12) - 1

    capital_low  = target_monthly_low  / monthly_ret if monthly_ret > 0 else np.inf
    capital_high = target_monthly_high / monthly_ret if monthly_ret > 0 else np.inf

    return {
        "CAGR_%": round(c * 100, 2),
        "MaxDD_%": round(md * 100, 2),
        "passes_maxdd_15pct": bool(passes),
        "monthly_ret_%": round(monthly_ret * 100, 3),
        "capital_for_3k_month": round(capital_low),
        "capital_for_5k_month": round(capital_high),
        "note": "Income requires steady withdrawal; MaxDD must be < 15% to be viable."
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  ENSEMBLE PORTFOLIO OPTIMIZER v1")
    print(f"  Period: {START} → {END}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 70)

    # ── Build individual strategy returns ────────────────────────────────────
    rp3x   = build_risk_parity_3x(START, END)
    xmom   = build_cross_asset_momentum(START, END)
    spy_ro = build_spy_risk_overlay(START, END)
    stock  = build_stock_picker_proxy(START, END)

    # Get SPY for regime classification and income proxy stress
    print("[SPY] Downloading SPY for benchmarks/regime classification...")
    spy_px = fetch_prices(["SPY"], START, END)
    spy_ret = spy_px.pct_change().dropna().squeeze()

    # Common index: intersection of all strategy dates
    common_idx = rp3x.index \
        .intersection(xmom.index) \
        .intersection(spy_ro.index) \
        .intersection(stock.index) \
        .intersection(spy_ret.index)
    common_idx = pd.DatetimeIndex(sorted(common_idx))
    print(f"\nCommon period: {common_idx[0].date()} → {common_idx[-1].date()} ({len(common_idx)} days)")

    income = build_income_proxy(common_idx, spy_ret.reindex(common_idx))

    # Align all to common index
    rp3x_c   = rp3x.reindex(common_idx)
    xmom_c   = xmom.reindex(common_idx)
    spy_ro_c = spy_ro.reindex(common_idx)
    stock_c  = stock.reindex(common_idx)
    income_c = income.reindex(common_idx)
    spy_c    = spy_ret.reindex(common_idx)

    labels = ["RP3x", "XMom", "SPY_RO", "StockPick", "Income"]
    strat_df = pd.DataFrame({
        "RP3x":      rp3x_c,
        "XMom":      xmom_c,
        "SPY_RO":    spy_ro_c,
        "StockPick": stock_c,
        "Income":    income_c,
    }).dropna()

    print(f"\nAligned dataset: {len(strat_df)} rows × {len(labels)} strategies")

    # ── Individual strategy performance ──────────────────────────────────────
    print("\n── Individual Strategy Metrics (full period) ──")
    indiv_metrics = {}
    for col in strat_df.columns:
        m = metrics(strat_df[col].values, col)
        m["regime"] = regime_sharpes(strat_df[col], spy_c.reindex(strat_df.index))
        indiv_metrics[col] = m
        print(f"  {col:<12}: CAGR={m['CAGR_%']:6.1f}%  Sharpe={m['Sharpe']:.2f}  "
              f"MaxDD={m['MaxDD_%']:.1f}%  gap={m['regime']['regime_gap']:.2f} "
              f"{'✓' if m['regime']['passes_R1'] else '✗'}")

    spy_m = metrics(spy_c.values, "SPY B&H")
    spy_m["regime"] = regime_sharpes(spy_c, spy_c)
    print(f"  {'SPY B&H':<12}: CAGR={spy_m['CAGR_%']:6.1f}%  Sharpe={spy_m['Sharpe']:.2f}  "
          f"MaxDD={spy_m['MaxDD_%']:.1f}%  (benchmark)")

    # ── Correlation matrix ────────────────────────────────────────────────────
    corr = strat_df.corr()
    print("\n── Correlation Matrix ──")
    print(corr.round(3).to_string())

    # ── Walk-forward optimization ─────────────────────────────────────────────
    print("\n── Walk-Forward Optimization ──")
    wf_results, test_df = walk_forward_optimize(
        strat_df, spy_c, labels, train_frac=TRAIN_FRAC
    )

    # Identify best allocation (by Sharpe with overlay, must pass R1)
    print("\n── Allocation Results (Test Period) ──")
    print(f"  {'Allocation':<22} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} "
          f"{'MaxDD%':>8} {'Calmar':>7} {'Gap':>6} {'R1':>4}")
    print("  " + "-"*72)

    best_name = None
    best_sharpe = -np.inf
    all_results_list = []

    for name, res in wf_results.items():
        m  = res["metrics"]
        mo = res["metrics_overlay"]
        rg = mo["regime"]
        passes = rg["passes_R1"]
        sharpe = mo["Sharpe"]
        row = {
            "name": name,
            "sharpe_overlay": sharpe,
            "passes_R1": passes,
            "metrics": mo,
            "weights": res["weights"] if isinstance(res["weights"], list) else list(res["weights"]),
        }
        all_results_list.append(row)
        flag = "✓" if passes else "✗"
        print(f"  {name:<22} {sharpe:>7.3f} {mo['Sortino']:>8.3f} {mo['CAGR_%']:>7.1f}% "
              f"{mo['MaxDD_%']:>8.1f}% {mo['Calmar']:>7.3f} {rg['regime_gap']:>6.3f} {flag:>4}")
        if passes and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_name = name

    if best_name is None:
        # Relax: pick best regardless of R1
        best_name = max(wf_results.keys(),
                        key=lambda k: wf_results[k]["metrics_overlay"]["Sharpe"])
        print(f"\n  WARNING: No allocation passed R1. Using best Sharpe: {best_name}")
    else:
        print(f"\n  BEST (R1-passing + max Sharpe): {best_name}")

    best_res = wf_results[best_name]
    best_port_ret = best_res["returns_overlay"]
    best_weights  = best_res["weights"]

    # ── Permutation test on best ──────────────────────────────────────────────
    obs_sharpe, null_dist, p_val = permutation_sharpe_test(best_port_ret, strat_df=test_df)
    print(f"\n── Permutation Test (n=1000, random weight vs. optimized weight) ──")
    print(f"  Observed Sharpe: {obs_sharpe:.3f}  |  Null median: {np.median(null_dist):.3f}  |  p-value: {p_val:.4f}")

    # ── Year-by-year ──────────────────────────────────────────────────────────
    print("\n── Year-by-Year Returns (Best Allocation vs SPY) ──")
    yr_best = yearly_returns(best_port_ret)
    yr_spy  = yearly_returns(spy_c.reindex(best_port_ret.index))
    yr_df = pd.DataFrame({"Best_Ensemble": yr_best, "SPY_BnH": yr_spy}).dropna()
    print(yr_df.round(2).to_string())

    # ── Regime-stratified ─────────────────────────────────────────────────────
    spy_aligned = spy_c.reindex(best_port_ret.index)
    regimes = regime_classify(spy_aligned)
    best_regime = regime_sharpes(best_port_ret, spy_aligned)
    print(f"\n── Regime Analysis (Best Allocation) ──")
    print(f"  GREEN days Sharpe: {best_regime['GREEN']:.3f}  (n={best_regime['n_GREEN']})")
    print(f"  RED   days Sharpe: {best_regime['RED']:.3f}  (n={best_regime['n_RED']})")
    print(f"  FLAT  days Sharpe: {best_regime['FLAT']:.3f}  (n={best_regime['n_FLAT']})")
    print(f"  Regime gap: {best_regime['regime_gap']:.3f}  {'PASSES R1 ✓' if best_regime['passes_R1'] else 'FAILS R1 ✗'}")

    # ── Travel income number ──────────────────────────────────────────────────
    tin = travel_income_number(best_port_ret)
    print(f"\n── Travel Income Number ──")
    print(f"  CAGR: {tin.get('CAGR_%', 'N/A')}%  MaxDD: {tin.get('MaxDD_%', 'N/A')}%")
    print(f"  Capital for $3K/month: ${tin.get('capital_for_3k_month', 'N/A'):,}")
    print(f"  Capital for $5K/month: ${tin.get('capital_for_5k_month', 'N/A'):,}")
    print(f"  Passes MaxDD < 15%: {tin.get('passes_maxdd_15pct', False)}")

    # ── Build full-period best portfolio for comparison ───────────────────────
    w_best = np.array(best_weights if isinstance(best_weights, list) else list(best_weights))
    port_full = build_portfolio(w_best, strat_df)
    port_full_ov, _ = apply_risk_overlay(port_full, spy_c.reindex(strat_df.index))

    # Also build best individual strategy benchmark
    best_indiv_name = max(indiv_metrics.keys(), key=lambda k: indiv_metrics[k]["Sharpe"])
    best_indiv_ret = strat_df[best_indiv_name]

    print(f"\n── Comparison: Best Ensemble vs Best Individual vs SPY ──")
    m_ens = metrics(port_full_ov.values, "Ensemble (full period)")
    m_ind = metrics(best_indiv_ret.values, f"Best Indiv ({best_indiv_name})")
    m_spy = metrics(spy_c.values, "SPY B&H")
    for m in [m_ens, m_ind, m_spy]:
        print(f"  {m['Label']:<30}: CAGR={m['CAGR_%']:6.1f}%  Sharpe={m['Sharpe']:.2f}  "
              f"MaxDD={m['MaxDD_%']:.1f}%  Calmar={m['Calmar']:.3f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n── Saving outputs ──")

    # 1. Full results JSON
    output = {
        "run_date": datetime.now().isoformat(),
        "period": {"start": START, "end": END},
        "walk_forward": {"train_frac": TRAIN_FRAC,
                         "train_end": str(strat_df.index[int(len(strat_df)*TRAIN_FRAC)].date()),
                         "test_start": str(strat_df.index[int(len(strat_df)*TRAIN_FRAC)+1].date())},
        "individual_strategies": {k: {kk: vv for kk, vv in v.items() if kk != "regime"}
                                   for k, v in indiv_metrics.items()},
        "best_allocation": {
            "name": best_name,
            "weights": {lab: round(float(ww), 4) for lab, ww in zip(labels, w_best)},
            "metrics_test_period": best_res["metrics_overlay"],
            "regime_analysis": best_regime,
            "permutation_test": {"observed_sharpe": round(obs_sharpe, 4),
                                 "p_value": round(p_val, 4),
                                 "null_median": round(float(np.median(null_dist)), 4)},
        },
        "travel_income": tin,
        "correlation_matrix": corr.round(4).to_dict(),
        "all_allocations": [
            {"name": r["name"],
             "sharpe_test": round(r["sharpe_overlay"], 3),
             "passes_R1": bool(r["passes_R1"]),
             "CAGR_%": r["metrics"]["CAGR_%"],
             "MaxDD_%": r["metrics"]["MaxDD_%"],
             "weights": {lab: round(float(ww), 4) for lab, ww in zip(labels, r["weights"])}}
            for r in all_results_list
        ],
        "hc_compliance": {
            "HC428_R1_regime_gap": best_regime["regime_gap"],
            "HC428_R1_passes": bool(best_regime["passes_R1"]),
            "walk_forward_used": True,
            "commission_free": True,
        }
    }

    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUT_DIR}/results.json")

    # 2. Year-by-year CSV
    yr_df.to_csv(f"{OUT_DIR}/yearly_returns.csv")
    print(f"  Saved: {OUT_DIR}/yearly_returns.csv")

    # 3. Daily returns CSV
    ret_csv = pd.DataFrame({
        "ensemble_overlay": port_full_ov,
        "spy_bnh": spy_c.reindex(port_full_ov.index),
        "rp3x": rp3x_c.reindex(port_full_ov.index),
        "xmom": xmom_c.reindex(port_full_ov.index),
        "spy_ro": spy_ro_c.reindex(port_full_ov.index),
        "stock_pick": stock_c.reindex(port_full_ov.index),
        "income": income_c.reindex(port_full_ov.index),
    })
    ret_csv.to_csv(f"{OUT_DIR}/daily_returns.csv")
    print(f"  Saved: {OUT_DIR}/daily_returns.csv")

    # ── CHARTS ───────────────────────────────────────────────────────────────
    print("\n── Generating charts ──")

    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.40, wspace=0.30)

    # 1. Equity curves — all strategies
    ax1 = fig.add_subplot(gs[0, :])
    for col in strat_df.columns:
        cum = (1 + strat_df[col]).cumprod()
        ax1.plot(cum.index, cum.values, alpha=0.7, label=col, linewidth=1.2)
    cum_spy = (1 + spy_c.reindex(strat_df.index)).cumprod()
    ax1.plot(cum_spy.index, cum_spy.values, "k--", alpha=0.5, label="SPY B&H", linewidth=1)
    cum_ens = (1 + port_full_ov).cumprod()
    ax1.plot(cum_ens.index, cum_ens.values, "r-", linewidth=2.5, label=f"Ensemble ({best_name})")
    ax1.set_title("Equity Curves — All Strategies + Best Ensemble", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_ylabel("Growth of $1")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(pd.Timestamp(output["walk_forward"]["train_end"]),
                color="orange", linestyle="--", alpha=0.7, label="Train/Test Split")

    # 2. Correlation heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    im = ax2.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax2.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax2.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax2, shrink=0.8)
    ax2.set_title("Strategy Correlation Matrix", fontsize=11, fontweight="bold")

    # 3. Weight comparison across allocations
    ax3 = fig.add_subplot(gs[1, 1])
    names_short = list(wf_results.keys())[:8]
    wmat = np.array([
        list(wf_results[n]["weights"]) if isinstance(wf_results[n]["weights"], list)
        else list(wf_results[n]["weights"])
        for n in names_short
    ])
    x = np.arange(len(names_short))
    width = 0.15
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for i, lab in enumerate(labels):
        ax3.bar(x + i*width, wmat[:, i], width, label=lab, color=colors[i], alpha=0.8)
    ax3.set_xticks(x + width*2)
    ax3.set_xticklabels(names_short, rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Weight")
    ax3.set_title("Allocation Weights by Method", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Year-by-year bar chart
    ax4 = fig.add_subplot(gs[2, 0])
    yr_both = pd.DataFrame({"Ensemble": yr_best, "SPY": yr_spy}).dropna()
    yrs = [str(y.year) for y in yr_both.index]
    x_pos = np.arange(len(yrs))
    ax4.bar(x_pos - 0.2, yr_both["Ensemble"], 0.4, label="Ensemble", color="#2196F3", alpha=0.8)
    ax4.bar(x_pos + 0.2, yr_both["SPY"],      0.4, label="SPY B&H",  color="#FF9800", alpha=0.8)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(yrs, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Annual Return (%)")
    ax4.set_title("Year-by-Year Returns: Ensemble vs SPY", fontsize=11, fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Drawdown comparison
    ax5 = fig.add_subplot(gs[2, 1])
    def drawdown_series(r):
        w = (1 + r).cumprod()
        pk = w.expanding().max()
        return ((w - pk) / pk) * 100
    dd_ens = drawdown_series(port_full_ov)
    dd_spy = drawdown_series(spy_c.reindex(port_full_ov.index))
    ax5.fill_between(dd_ens.index, dd_ens.values, 0, alpha=0.6, color="#2196F3", label="Ensemble")
    ax5.fill_between(dd_spy.index, dd_spy.values, 0, alpha=0.4, color="#FF5722", label="SPY B&H")
    ax5.set_ylabel("Drawdown (%)")
    ax5.set_title("Drawdown: Ensemble vs SPY", fontsize=11, fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Permutation test null distribution
    ax6 = fig.add_subplot(gs[3, 0])
    rng_null = null_dist[np.isfinite(null_dist)]
    if rng_null.max() - rng_null.min() < 1e-6:
        rng_null = rng_null + np.random.default_rng(0).normal(0, 0.001, len(rng_null))
    n_bins = min(40, max(5, len(np.unique(rng_null.round(2)))))
    ax6.hist(rng_null, bins=n_bins, alpha=0.7, color="#78909C", edgecolor="white", label="Null Sharpes")
    ax6.axvline(obs_sharpe, color="red", linewidth=2, label=f"Observed: {obs_sharpe:.2f}")
    ax6.set_xlabel("Sharpe Ratio")
    ax6.set_ylabel("Frequency")
    ax6.set_title(f"Permutation Test  (p = {p_val:.4f})", fontsize=11, fontweight="bold")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Sharpe by allocation (test period)
    ax7 = fig.add_subplot(gs[3, 1])
    all_names = list(wf_results.keys())
    all_sharpes = [wf_results[n]["metrics_overlay"]["Sharpe"] for n in all_names]
    all_pass = [regime_sharpes(wf_results[n]["returns_overlay"],
                               spy_c.reindex(wf_results[n]["returns_overlay"].index))["passes_R1"]
                for n in all_names]
    bar_colors = ["#4CAF50" if p else "#F44336" for p in all_pass]
    y_pos = np.arange(len(all_names))
    ax7.barh(y_pos, all_sharpes, color=bar_colors, alpha=0.8)
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels(all_names, fontsize=8)
    ax7.set_xlabel("Sharpe Ratio (Test Period, with Overlay)")
    ax7.set_title("Allocation Sharpe Comparison\n(Green=passes R1, Red=fails)", fontsize=11, fontweight="bold")
    ax7.axvline(spy_m["Sharpe"], color="orange", linestyle="--", label=f"SPY {spy_m['Sharpe']:.2f}")
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3, axis="x")

    fig.suptitle("ENSEMBLE PORTFOLIO OPTIMIZER v1\n" +
                 f"Best Allocation: {best_name}  |  Test Sharpe: {best_sharpe:.2f}  |  R1 gap: {best_regime['regime_gap']:.3f}",
                 fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(f"{OUT_DIR}/ensemble_portfolio_v1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/ensemble_portfolio_v1.png")

    # ── Final summary print ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Best Allocation: {best_name}")
    w_labels = list(zip(labels, [round(float(ww)*100, 1) for ww in w_best]))
    for lab, pct in w_labels:
        print(f"    {lab:<12}: {pct:.1f}%")
    print(f"\n  Test Period Performance (with risk overlay):")
    bm = best_res["metrics_overlay"]
    print(f"    CAGR:     {bm['CAGR_%']:.1f}%")
    print(f"    Sharpe:   {bm['Sharpe']:.3f}")
    print(f"    Sortino:  {bm['Sortino']:.3f}")
    print(f"    MaxDD:    {bm['MaxDD_%']:.1f}%")
    print(f"    Calmar:   {bm['Calmar']:.3f}")
    print(f"\n  HC #428 R1: regime gap = {best_regime['regime_gap']:.3f}  "
          f"{'PASSES ✓' if best_regime['passes_R1'] else 'FAILS ✗'}")
    print(f"  Permutation test p-value: {p_val:.4f}  "
          f"({'significant' if p_val < 0.05 else 'not significant'})")
    print(f"\n  Travel Income Number:")
    if "capital_for_3k_month" in tin:
        print(f"    $3K/month → need ${tin['capital_for_3k_month']:>10,.0f}")
        print(f"    $5K/month → need ${tin['capital_for_5k_month']:>10,.0f}")
        print(f"    MaxDD constraint (<15%) passed: {tin['passes_maxdd_15pct']}")
    print(f"\n  Outputs: {OUT_DIR}/")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
