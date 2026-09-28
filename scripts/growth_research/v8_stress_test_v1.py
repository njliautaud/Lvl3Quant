#!/usr/bin/env python3
"""
V8 Stress Test v1 — Comprehensive Robustness Analysis of V8 Config
====================================================================

V8 is our best sector options config: Sharpe 3.23, 5/5 gates, MDD -8.0%
V8 params: DTE=14, OTM=2%, weekly rebalance, 17 features,
           LGBM (100 trees, depth 4, lr 0.05), pairs mode (VIX<20 bull+bear)

Stress tests:
  1. Monte Carlo bootstrap (1000 iterations) — CI on Sharpe, CAGR, MDD
  2. Cost sensitivity — commission at 1x, 1.5x, 2x, 3x
  3. Haircut sensitivity — 15%, 20%, 25%, 30%
  4. IV multiplier sensitivity — 1.0x, 1.2x, 1.4x, 1.6x
  5. Training window sensitivity — 8, 10, 12, 14, 16 periods
  6. Structural edge test — random rankings (50 iterations)

All functions inlined (no project imports). Runs standalone on any node.

Honest pricing rules:
  - HOLD TO EXPIRY ONLY
  - Intrinsic value only at expiry
  - Entry haircut on BS fair value
  - ATR-based IV estimation
  - Sliding window walk-forward (NEVER expanding)
  - Commission $2.60/spread default
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════
# DETECT NODE AND SET PATHS
# ══════════════════════════════════════════════════════════════

HOSTNAME = os.uname().nodename.lower()
if "neptune" in HOSTNAME or "nick" in os.path.expanduser("~"):
    BASE = Path("/home/nick/Lvl3Quant")
else:
    BASE = Path("/home/jupiter/Lvl3Quant")

OUTPUT_DIR = BASE / "output" / "growth_research" / "v8_stress_test_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# V8 CONFIG (CANONICAL)
# ══════════════════════════════════════════════════════════════

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]

CAP = 645.0
DTE = 14
OTM_PCT = 0.02        # 2% OTM
SPREAD_PCT = 3.0       # 3% width
TOP_K = 3
REBAL_FREQ = "W-FRI"   # weekly
WF_TRAIN_PERIODS = 12  # default, varied in sensitivity
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# Pricing constants
RISK_FREE_RATE = 0.045
DEFAULT_HAIRCUT = 0.15
COMMISSION_PER_LEG = 0.65
COMMISSION_RT_SPREAD = 2.60   # $0.65 x 4 legs

# V6/V8 17 features (V4 minus 4 redundant vol features)
V8_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d",
    "cross_sector_dispersion",
]
assert len(V8_FEATURES) == 17

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v8_stress_test"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- results saved to disk only")


# ══════════════════════════════════════════════════════════════
# INLINED: BLACK-SCHOLES PRICING
# ══════════════════════════════════════════════════════════════

def bs_call_price(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    """Black-Scholes European call option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put_price(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    """Black-Scholes European put option price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def estimate_iv(atr, spot, vix=20.0, atr_period=14, iv_multiplier=None):
    """
    Estimate implied volatility from ATR and VIX.
    iv_multiplier overrides the default VIX-based calculation if provided.
    """
    if spot <= 0 or atr <= 0:
        return 0.25
    realized_vol = (atr / spot) * np.sqrt(252 / atr_period)
    if iv_multiplier is not None:
        sigma = realized_vol * iv_multiplier
    else:
        iv_mult = 1.2 + 0.01 * max(vix - 20.0, 0.0)
        sigma = realized_vol * iv_mult
    return max(sigma, 0.10)


def price_bull_call_spread(S, K1, K2, dte, atr, vix=20.0,
                           haircut=DEFAULT_HAIRCUT, iv_multiplier=None):
    """Price a bull call spread with entry haircut."""
    if K2 <= K1:
        raise ValueError(f"K2 ({K2}) must be > K1 ({K1})")
    T = dte / 365.0
    sigma = estimate_iv(atr, S, vix, iv_multiplier=iv_multiplier)
    fair_value = bs_call_price(S, K1, T, RISK_FREE_RATE, sigma) - bs_call_price(S, K2, T, RISK_FREE_RATE, sigma)
    fair_value = max(fair_value, 0.001)
    entry_cost = fair_value * (1.0 + haircut)
    spread_width = K2 - K1
    max_profit = spread_width - entry_cost
    return float(entry_cost), float(max_profit)


def price_bear_put_spread(S, K1, K2, dte, atr, vix=20.0,
                          haircut=DEFAULT_HAIRCUT, iv_multiplier=None):
    """Price a bear put spread with entry haircut."""
    if K2 <= K1:
        raise ValueError(f"K2 ({K2}) must be > K1 ({K1})")
    T = dte / 365.0
    sigma = estimate_iv(atr, S, vix, iv_multiplier=iv_multiplier)
    fair_value = bs_put_price(S, K2, T, RISK_FREE_RATE, sigma) - bs_put_price(S, K1, T, RISK_FREE_RATE, sigma)
    fair_value = max(fair_value, 0.001)
    entry_cost = fair_value * (1.0 + haircut)
    spread_width = K2 - K1
    max_profit = spread_width - entry_cost
    return float(entry_cost), float(max_profit)


# ══════════════════════════════════════════════════════════════
# INLINED: ADVERSARIAL VALIDATION (5-GATE)
# ══════════════════════════════════════════════════════════════

def compute_honest_sharpe(equity_series, annualization=12.0):
    """Compute HONEST Sharpe and Sortino from equity time series (equity-based pct_change)."""
    if len(equity_series) < 2:
        return 0.0, 0.0, pd.Series(dtype=float)
    monthly_equity = equity_series.resample("ME").last().dropna()
    if len(monthly_equity) < 2:
        return 0.0, 0.0, pd.Series(dtype=float)
    monthly_returns = monthly_equity.pct_change().dropna()
    if len(monthly_returns) < 2 or monthly_returns.std() == 0:
        return 0.0, 0.0, monthly_returns
    mean_ret = monthly_returns.mean()
    std_ret = monthly_returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(annualization)
    downside = monthly_returns[monthly_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (mean_ret / downside.std()) * np.sqrt(annualization)
    else:
        sortino = sharpe * 1.5
    return float(sharpe), float(sortino), monthly_returns


def compute_metrics(trades, initial_capital):
    """Compute honest metrics from a trade list."""
    if not trades:
        return {
            "sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "max_dd": -1.0,
            "win_rate": 0.0, "profit_factor": 0.0, "n_trades": 0,
            "final_equity": initial_capital, "monthly_returns": pd.Series(dtype=float),
        }
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("exit_date")
    pnls = df["pnl"].values

    equity_values = [initial_capital]
    for pnl in pnls:
        equity_values.append(equity_values[-1] + pnl)

    dates = [df["entry_date"].iloc[0] - pd.Timedelta(days=1)]
    dates.extend(df["exit_date"].tolist())
    equity_series = pd.Series(equity_values, index=pd.DatetimeIndex(dates))
    equity_series = equity_series.groupby(equity_series.index).last()

    sharpe, sortino, monthly_rets = compute_honest_sharpe(equity_series)

    final_eq = equity_values[-1]
    total_days = (dates[-1] - dates[0]).days
    years = max(total_days / 365.25, 0.1)
    if final_eq > 0 and initial_capital > 0:
        cagr = (final_eq / initial_capital) ** (1 / years) - 1
    else:
        cagr = -1.0

    eq_arr = np.array(equity_values)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.min())

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

    return {
        "sharpe": sharpe, "sortino": sortino, "cagr": cagr,
        "max_dd": max_dd, "win_rate": win_rate, "profit_factor": profit_factor,
        "n_trades": len(pnls), "final_equity": final_eq,
        "monthly_returns": monthly_rets,
    }


def validate_trades_simple(trades, initial_capital=CAP, spy_prices=None,
                           strategy_name="Strategy", n_perms=2000):
    """Simplified 5-gate validation returning a dict."""
    if len(trades) < 10:
        return {"error": f"Too few trades ({len(trades)})", "sharpe": 0.0,
                "sortino": 0.0, "cagr": 0.0, "max_dd": -1.0,
                "win_rate": 0.0, "profit_factor": 0.0,
                "n_trades": len(trades), "final_equity": initial_capital,
                "gates_passed": 0, "gates_total": 5}

    metrics = compute_metrics(trades, initial_capital)
    gates = []

    # Gate 1: Sign-flip permutation
    pnls = np.array([t["pnl"] for t in trades])
    exit_dates = pd.to_datetime([t["exit_date"] for t in trades])
    beat_count = 0
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(pnls))
        flipped = pnls * signs
        eq_vals = [initial_capital]
        for p in flipped:
            eq_vals.append(eq_vals[-1] + p)
        eq_s = pd.Series(eq_vals, index=pd.DatetimeIndex(
            [exit_dates[0] - pd.Timedelta(days=1)] + list(exit_dates)))
        eq_s = eq_s.groupby(eq_s.index).last()
        ps, _, _ = compute_honest_sharpe(eq_s)
        if ps >= metrics["sharpe"]:
            beat_count += 1
    p_val = beat_count / n_perms
    gates.append(("Sign-Flip", p_val < 0.05))

    # Gate 2: Regime balance
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    if spy_prices is not None and len(spy_prices) > 0:
        bull_pnls, bear_pnls = [], []
        spy_sorted = spy_prices.sort_index()
        for _, row in df.iterrows():
            ep = spy_sorted[spy_sorted.index <= row["entry_date"]]
            xp = spy_sorted[spy_sorted.index <= row["exit_date"]]
            if len(ep) == 0 or len(xp) == 0:
                continue
            if float(xp.iloc[-1]) >= float(ep.iloc[-1]):
                bull_pnls.append(row["pnl"])
            else:
                bear_pnls.append(row["pnl"])
        if len(bull_pnls) >= 5 and len(bear_pnls) >= 5:
            bull_wr = np.mean([1 if p > 0 else 0 for p in bull_pnls])
            bear_wr = np.mean([1 if p > 0 else 0 for p in bear_pnls])
            gates.append(("Regime Balance", abs(bull_wr - bear_wr) < 0.50))
        else:
            gates.append(("Regime Balance", True))
    else:
        gates.append(("Regime Balance", True))

    # Gate 3: Sub-period stability
    df_sorted = pd.DataFrame(trades).sort_values("exit_date", key=pd.to_datetime)
    mid = len(df_sorted) // 2
    h1 = df_sorted.iloc[:mid]["pnl"].sum()
    h2 = df_sorted.iloc[mid:]["pnl"].sum()
    gates.append(("Sub-Period", h1 > 0 and h2 > 0))

    # Gate 4: Outlier removal
    mr = metrics["monthly_returns"]
    if len(mr) >= 3:
        trimmed = mr.drop(mr.idxmax())
        if len(trimmed) >= 2 and trimmed.std() > 0:
            ts = (trimmed.mean() / trimmed.std()) * np.sqrt(12)
            gates.append(("Outlier Removal", trimmed.sum() > 0 and ts > 0))
        else:
            gates.append(("Outlier Removal", False))
    else:
        gates.append(("Outlier Removal", False))

    # Gate 5: Yearly consistency
    df_y = pd.DataFrame(trades)
    df_y["year"] = pd.to_datetime(df_y["exit_date"]).dt.year
    yearly = df_y.groupby("year")["pnl"].sum()
    pct_profitable = (yearly > 0).sum() / len(yearly) if len(yearly) > 0 else 0
    gates.append(("Yearly Consistency", pct_profitable >= 0.60))

    passed = sum(1 for _, p in gates if p)

    return {
        "strategy_name": strategy_name,
        "sharpe": round(metrics["sharpe"], 3),
        "sortino": round(metrics["sortino"], 3),
        "cagr": round(metrics["cagr"], 4),
        "max_dd": round(metrics["max_dd"], 4),
        "win_rate": round(metrics["win_rate"], 4),
        "profit_factor": round(metrics["profit_factor"], 3),
        "n_trades": metrics["n_trades"],
        "final_equity": round(metrics["final_equity"], 2),
        "gates_passed": passed,
        "gates_total": 5,
        "gates": {name: passed_flag for name, passed_flag in gates},
    }


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download all required tickers via yfinance."""
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start="2008-01-01", progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw[["Close"]]
    high = raw["High"] if mi else raw[["High"]]
    low = raw["Low"] if mi else raw[["Low"]]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)
    close = close.ffill()
    high = high.ffill()
    low = low.ffill()
    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)
    for t in ["SPY", "VIX"]:
        if t not in close.columns:
            raise ValueError(f"Missing critical ticker: {t}")
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low


# ══════════════════════════════════════════════════════════════
# REGIME LOADING
# ══════════════════════════════════════════════════════════════

def load_regime_predictions():
    """Load GRU regime predictions; fall back to VIX proxy."""
    if not REGIME_FILE.exists():
        fprint("WARNING: Regime file not found, using VIX-based proxy")
        return None
    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    fprint(f"Regime predictions: {len(regime_series)} days, "
           f"mean={regime_series.mean():.3f}, >0.4={int((regime_series > 0.4).sum())}")
    return regime_series


def get_regime_score(regime_series, dt):
    """Get regime score at date with nearest-date fallback."""
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def compute_legacy_features(px, spy_slice):
    """Compute the 15 legacy quality-momentum features (V8 subset)."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()
    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
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
    up_days = rets[rets > 0]
    f["up_capture"] = float(up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)) if len(up_days) > 10 else 1.0

    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
        f["trend_slope_63d"] = slope * 252
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0

    return f


def compute_cross_asset_features(sector_ticker, dt_idx, close_df):
    """Compute the 2 cross-asset features used in V8."""
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {"sector_spy_beta_63d": 1.0, "cross_sector_dispersion": 0.01}

    spy_ret = spy.pct_change().dropna()

    # Sector-SPY beta 63d
    if sector_px is not None and len(sector_px) > 63:
        sec_ret = sector_px.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            sr = sec_ret.loc[common].iloc[-63:]
            mr = spy_ret.loc[common].iloc[-63:]
            cov = np.cov(sr.values, mr.values)
            beta = cov[0, 1] / (cov[1, 1] + 1e-10)
            f["sector_spy_beta_63d"] = float(beta)
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0

    # Cross-sector dispersion
    sector_cols = [c for c in SECTORS if c in close_df.columns]
    if len(sector_cols) > 3:
        sector_rets = close_df[sector_cols].iloc[:dt_idx + 1].pct_change()
        daily_disp = sector_rets.std(axis=1)
        if len(daily_disp) > 21:
            f["cross_sector_dispersion"] = float(daily_disp.rolling(21).mean().iloc[-1])
        else:
            f["cross_sector_dispersion"] = 0.01
    else:
        f["cross_sector_dispersion"] = 0.01

    return f


# ══════════════════════════════════════════════════════════════
# ATR COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_atr_series(high, low, close, period=14):
    """Compute ATR series for all sectors."""
    atr_dict = {}
    for tk in SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h = high[tk].dropna()
            l = low[tk].dropna()
            c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM RANKING (SLIDING WINDOW)
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """Build feature + target records for walk-forward LGBM."""
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        rscore = get_regime_score(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue  # regime filter

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            cross = {}
            for col in feature_cols:
                if col in ["sector_spy_beta_63d", "cross_sector_dispersion"]:
                    cross = compute_cross_asset_features(tk, idx, close)
                    break

            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)
    fprint(f"  Records: {len(df)}, dates: {len(df['date'].unique())}")
    return df


def walk_forward_lgbm_rank(df, feature_cols, train_periods=WF_TRAIN_PERIODS):
    """Walk-forward LGBM ranking with SLIDING window."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"  Insufficient data ({len(df)} records)")
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}

    for i in range(train_periods, len(dates)):
        # SLIDING window (not expanding)
        train_dates = dates[max(0, i - train_periods):i]
        test_date = dates[i]

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()

        if len(test_df) < 3 or len(train_df) < 50:
            continue

        Xt = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[feature_cols].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception:
            continue

    fprint(f"  Rankings: {len(rankings)} dates")
    return rankings


# ══════════════════════════════════════════════════════════════
# STRIKE COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction, otm_pct, spread_pct):
    """Compute strike prices for a spread. Returns (K1, K2) where K1 < K2."""
    if direction == "bull":
        if otm_pct > 0:
            K1 = round(S * (1 + otm_pct), 2)
            K2 = round(K1 * (1 + spread_pct / 100), 2)
        else:
            K1 = round(S, 2)
            K2 = round(S * (1 + spread_pct / 100), 2)
    else:
        if otm_pct > 0:
            K2 = round(S * (1 - otm_pct), 2)
            K1 = round(K2 * (1 - spread_pct / 100), 2)
        else:
            K1 = round(S * (1 - spread_pct / 100), 2)
            K2 = round(S, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION (V8 CONFIG)
# ══════════════════════════════════════════════════════════════

def simulate_v8_trades(rankings, close, high, low, atr_dict,
                       commission=COMMISSION_RT_SPREAD,
                       haircut=DEFAULT_HAIRCUT,
                       iv_multiplier_override=None):
    """
    Simulate V8 config trades.
    Pairs mode: VIX<20 => top-3 bull + bottom-3 bear; VIX>=20 => bull only.
    Hold to expiry, intrinsic value only, entry haircut.
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        # Pairs mode: VIX < 20 => bull+bear, else bull only
        if cv < 20.0:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        if trade_mode == "pairs":
            max_pos = min(100, equity / 6)
        else:
            max_pos = min(200, equity / 3)

        if max_pos < 30:
            continue

        all_picks = [(tk, "bull") for tk in bull_picks] + [(tk, "bear") for tk in bear_picks]

        for tk, direction in all_picks:
            if tk not in close.columns or tk not in atr_dict:
                continue

            S = float(close[tk].loc[dt])
            di = close.index.get_loc(dt)
            ei = min(di + DTE, len(close) - 1)
            if ei <= di:
                continue

            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                av = float(atr_dict[tk].loc[dt])
            else:
                av = S * 0.015

            K1, K2 = compute_strikes(S, direction, OTM_PCT, SPREAD_PCT)

            try:
                if direction == "bull":
                    entry_cost_ps, _ = price_bull_call_spread(
                        S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv,
                        haircut=haircut, iv_multiplier=iv_multiplier_override)
                else:
                    entry_cost_ps, _ = price_bear_put_spread(
                        S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv,
                        haircut=haircut, iv_multiplier=iv_multiplier_override)
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + commission
            if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
                continue

            Se = float(close[tk].iloc[ei])
            if direction == "bull":
                intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
            else:
                intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

            pnl = (intrinsic - entry_cost_ps) * 100 - commission
            equity += pnl

            sv = float(spy.loc[dt])
            se = float(spy.iloc[ei]) if ei < len(spy) else sv
            spy_regime = "bull" if se >= sv else "bear"

            trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": spy_regime,
                "direction": direction,
                "vix": round(cv, 1),
                "win": pnl > 0,
            })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# STRESS TEST 1: MONTE CARLO BOOTSTRAP
# ══════════════════════════════════════════════════════════════

def monte_carlo_bootstrap(trades, n_iterations=1000):
    """Resample trade P&Ls with replacement, compute CI on Sharpe, CAGR, MDD."""
    fprint(f"\n{'='*70}")
    fprint(f"STRESS TEST 1: Monte Carlo Bootstrap ({n_iterations} iterations)")
    fprint(f"{'='*70}")

    pnls = np.array([t["pnl"] for t in trades])
    n_trades = len(pnls)

    sharpes, cagrs, mdds = [], [], []
    entry_dates = pd.to_datetime([t["entry_date"] for t in trades])
    exit_dates = pd.to_datetime([t["exit_date"] for t in trades])
    total_days = (exit_dates.max() - entry_dates.min()).days
    years = max(total_days / 365.25, 0.1)

    for i in range(n_iterations):
        np.random.seed(i)
        idx = np.random.choice(n_trades, size=n_trades, replace=True)
        sampled_pnls = pnls[idx]

        eq = [CAP]
        for p in sampled_pnls:
            eq.append(eq[-1] + p)
        eq = np.array(eq)

        # Sharpe from equity curve
        if len(eq) > 1:
            rets = np.diff(eq) / eq[:-1]
            if np.std(rets) > 0:
                sharpes.append(float(np.mean(rets) / np.std(rets) * np.sqrt(252)))
            else:
                sharpes.append(0.0)
        else:
            sharpes.append(0.0)

        # CAGR
        final = eq[-1]
        if final > 0:
            cagrs.append(float((final / CAP) ** (1 / years) - 1))
        else:
            cagrs.append(-1.0)

        # MDD
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.where(peak > 0, peak, 1)
        mdds.append(float(dd.min()))

    sharpes = np.array(sharpes)
    cagrs = np.array(cagrs)
    mdds = np.array(mdds)

    result = {
        "sharpe_mean": round(float(np.mean(sharpes)), 3),
        "sharpe_median": round(float(np.median(sharpes)), 3),
        "sharpe_5th": round(float(np.percentile(sharpes, 5)), 3),
        "sharpe_95th": round(float(np.percentile(sharpes, 95)), 3),
        "sharpe_pct_positive": round(float(np.mean(sharpes > 0) * 100), 1),
        "cagr_mean": round(float(np.mean(cagrs)), 4),
        "cagr_5th": round(float(np.percentile(cagrs, 5)), 4),
        "cagr_95th": round(float(np.percentile(cagrs, 95)), 4),
        "mdd_mean": round(float(np.mean(mdds)), 4),
        "mdd_5th": round(float(np.percentile(mdds, 5)), 4),
        "mdd_95th": round(float(np.percentile(mdds, 95)), 4),
    }

    fprint(f"  Sharpe:  mean={result['sharpe_mean']:.3f}, "
           f"5th={result['sharpe_5th']:.3f}, 95th={result['sharpe_95th']:.3f}, "
           f"{result['sharpe_pct_positive']:.0f}% positive")
    fprint(f"  CAGR:    mean={result['cagr_mean']*100:.1f}%, "
           f"5th={result['cagr_5th']*100:.1f}%, 95th={result['cagr_95th']*100:.1f}%")
    fprint(f"  MaxDD:   mean={result['mdd_mean']*100:.1f}%, "
           f"5th={result['mdd_5th']*100:.1f}%, 95th={result['mdd_95th']*100:.1f}%")

    return result


# ══════════════════════════════════════════════════════════════
# STRESS TEST 2: COST SENSITIVITY
# ══════════════════════════════════════════════════════════════

def cost_sensitivity(rankings, close, high, low, atr_dict, spy_prices):
    """Test performance at different commission levels."""
    fprint(f"\n{'='*70}")
    fprint(f"STRESS TEST 2: Cost Sensitivity")
    fprint(f"{'='*70}")

    multipliers = [1.0, 1.5, 2.0, 3.0]
    base_comm = COMMISSION_RT_SPREAD  # $2.60

    results = {}
    for mult in multipliers:
        comm = base_comm * mult
        trades, final_eq = simulate_v8_trades(
            rankings, close, high, low, atr_dict, commission=comm)
        if trades and len(trades) >= 10:
            m = validate_trades_simple(trades, CAP, spy_prices, f"Cost_{mult}x")
            results[f"{mult}x (${comm:.2f})"] = m
            fprint(f"  {mult}x (${comm:.2f}/spread): Sharpe={m['sharpe']:.2f}, "
                   f"WR={m['win_rate']*100:.1f}%, PF={m['profit_factor']:.2f}, "
                   f"${CAP:.0f}->${m['final_equity']:.0f}, gates={m['gates_passed']}/5")
        else:
            fprint(f"  {mult}x (${comm:.2f}/spread): insufficient trades")
            results[f"{mult}x (${comm:.2f})"] = {"error": "insufficient trades"}

    return results


# ══════════════════════════════════════════════════════════════
# STRESS TEST 3: HAIRCUT SENSITIVITY
# ══════════════════════════════════════════════════════════════

def haircut_sensitivity(rankings, close, high, low, atr_dict, spy_prices):
    """Test performance at different entry haircut levels."""
    fprint(f"\n{'='*70}")
    fprint(f"STRESS TEST 3: Haircut Sensitivity")
    fprint(f"{'='*70}")

    haircuts = [0.15, 0.20, 0.25, 0.30]
    results = {}
    for hc in haircuts:
        trades, final_eq = simulate_v8_trades(
            rankings, close, high, low, atr_dict, haircut=hc)
        if trades and len(trades) >= 10:
            m = validate_trades_simple(trades, CAP, spy_prices, f"Haircut_{hc*100:.0f}pct")
            results[f"{hc*100:.0f}%"] = m
            fprint(f"  {hc*100:.0f}% haircut: Sharpe={m['sharpe']:.2f}, "
                   f"WR={m['win_rate']*100:.1f}%, PF={m['profit_factor']:.2f}, "
                   f"${CAP:.0f}->${m['final_equity']:.0f}, gates={m['gates_passed']}/5")
        else:
            fprint(f"  {hc*100:.0f}% haircut: insufficient trades")
            results[f"{hc*100:.0f}%"] = {"error": "insufficient trades"}

    return results


# ══════════════════════════════════════════════════════════════
# STRESS TEST 4: IV MULTIPLIER SENSITIVITY
# ══════════════════════════════════════════════════════════════

def iv_multiplier_sensitivity(rankings, close, high, low, atr_dict, spy_prices):
    """Test performance at different IV multiplier levels."""
    fprint(f"\n{'='*70}")
    fprint(f"STRESS TEST 4: IV Multiplier Sensitivity")
    fprint(f"{'='*70}")

    multipliers = [1.0, 1.2, 1.4, 1.6]
    results = {}
    for iv_mult in multipliers:
        trades, final_eq = simulate_v8_trades(
            rankings, close, high, low, atr_dict, iv_multiplier_override=iv_mult)
        if trades and len(trades) >= 10:
            m = validate_trades_simple(trades, CAP, spy_prices, f"IVMult_{iv_mult}x")
            results[f"{iv_mult}x"] = m
            fprint(f"  IV mult={iv_mult}x: Sharpe={m['sharpe']:.2f}, "
                   f"WR={m['win_rate']*100:.1f}%, PF={m['profit_factor']:.2f}, "
                   f"${CAP:.0f}->${m['final_equity']:.0f}, gates={m['gates_passed']}/5")
        else:
            fprint(f"  IV mult={iv_mult}x: insufficient trades")
            results[f"{iv_mult}x"] = {"error": "insufficient trades"}

    return results


# ══════════════════════════════════════════════════════════════
# STRESS TEST 5: TRAINING WINDOW SENSITIVITY
# ══════════════════════════════════════════════════════════════

def training_window_sensitivity(df_records, feature_cols, close, high, low,
                                atr_dict, spy_prices):
    """Test performance with different LGBM training window sizes."""
    fprint(f"\n{'='*70}")
    fprint(f"STRESS TEST 5: Training Window Sensitivity")
    fprint(f"{'='*70}")

    windows = [8, 10, 12, 14, 16]
    results = {}
    for tw in windows:
        fprint(f"  Training with window={tw}...")
        rankings = walk_forward_lgbm_rank(df_records, feature_cols, train_periods=tw)
        if not rankings:
            fprint(f"    No rankings produced")
            results[f"{tw}_periods"] = {"error": "no rankings"}
            continue

        trades, final_eq = simulate_v8_trades(rankings, close, high, low, atr_dict)
        if trades and len(trades) >= 10:
            m = validate_trades_simple(trades, CAP, spy_prices, f"Window_{tw}")
            results[f"{tw}_periods"] = m
            fprint(f"    window={tw}: Sharpe={m['sharpe']:.2f}, "
                   f"WR={m['win_rate']*100:.1f}%, PF={m['profit_factor']:.2f}, "
                   f"trades={m['n_trades']}, gates={m['gates_passed']}/5")
        else:
            fprint(f"    window={tw}: insufficient trades")
            results[f"{tw}_periods"] = {"error": "insufficient trades"}

    return results


# ══════════════════════════════════════════════════════════════
# STRESS TEST 6: STRUCTURAL EDGE TEST (RANDOM RANKINGS)
# ══════════════════════════════════════════════════════════════

def structural_edge_test(rankings, close, high, low, atr_dict, spy_prices,
                         n_trials=50):
    """Quantify ML alpha vs structure alpha using random rankings."""
    fprint(f"\n{'='*70}")
    fprint(f"STRESS TEST 6: Structural Edge Test ({n_trials} random trials)")
    fprint(f"{'='*70}")

    random_sharpes = []
    random_wrs = []
    random_pfs = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_v8_trades(
            rand_rankings, close, high, low, atr_dict)

        if trades and len(trades) >= 10:
            m = compute_metrics(trades, CAP)
            random_sharpes.append(m["sharpe"])
            random_wrs.append(m["win_rate"])
            random_pfs.append(m["profit_factor"])

        if (trial + 1) % 10 == 0:
            fprint(f"  Completed {trial + 1}/{n_trials} trials...")

    random_sharpes = np.array(random_sharpes)
    random_wrs = np.array(random_wrs)
    random_pfs = np.array(random_pfs)

    result = {
        "n_trials": len(random_sharpes),
        "random_sharpe_mean": round(float(np.mean(random_sharpes)), 3) if len(random_sharpes) > 0 else 0.0,
        "random_sharpe_std": round(float(np.std(random_sharpes)), 3) if len(random_sharpes) > 0 else 0.0,
        "random_sharpe_median": round(float(np.median(random_sharpes)), 3) if len(random_sharpes) > 0 else 0.0,
        "random_sharpe_max": round(float(np.max(random_sharpes)), 3) if len(random_sharpes) > 0 else 0.0,
        "random_wr_mean": round(float(np.mean(random_wrs)), 4) if len(random_wrs) > 0 else 0.0,
        "random_pf_mean": round(float(np.mean(random_pfs)), 3) if len(random_pfs) > 0 else 0.0,
    }

    if len(random_sharpes) > 0:
        fprint(f"  Random Sharpe: mean={result['random_sharpe_mean']:.3f}, "
               f"std={result['random_sharpe_std']:.3f}, "
               f"median={result['random_sharpe_median']:.3f}, "
               f"max={result['random_sharpe_max']:.3f}")
        fprint(f"  Random WR:     mean={result['random_wr_mean']*100:.1f}%")
        fprint(f"  Random PF:     mean={result['random_pf_mean']:.2f}")
    else:
        fprint(f"  No valid random trials completed")

    return result


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 70)
    fprint(f"V8 STRESS TEST v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)
    fprint(f"V8 Config: DTE={DTE}, OTM={OTM_PCT*100:.0f}%, weekly rebal, "
           f"17 features, LGBM(100t/d4/lr0.05), pairs(VIX<20)")
    fprint(f"Capital: ${CAP:.0f} | Haircut: {DEFAULT_HAIRCUT:.0%} entry only | "
           f"Comm: ${COMMISSION_RT_SPREAD:.2f}/spread")
    fprint(f"Output: {OUTPUT_DIR}")
    fprint()

    # 1. Download data
    close, high, low = download_data()
    spy_prices = close["SPY"]

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Build rebalance dates (weekly)
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(REBAL_FREQ).last().dropna().values)
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # 5. Build feature records
    fprint("\nBuilding feature records...")
    df_records = build_feature_records(
        close, high, low, rebal_dates, V8_FEATURES, regime_series)

    # 6. Train walk-forward LGBM (baseline with 12-period window)
    fprint("\nTraining baseline LGBM (12-period sliding window)...")
    baseline_rankings = walk_forward_lgbm_rank(df_records, V8_FEATURES, train_periods=12)

    if not baseline_rankings:
        fprint("ERROR: No rankings produced. Cannot proceed.")
        return

    # 7. Run baseline
    fprint("\nRunning baseline V8 simulation...")
    baseline_trades, baseline_eq = simulate_v8_trades(
        baseline_rankings, close, high, low, atr_dict)

    if not baseline_trades or len(baseline_trades) < 10:
        fprint(f"ERROR: Only {len(baseline_trades) if baseline_trades else 0} baseline trades.")
        return

    baseline_result = validate_trades_simple(
        baseline_trades, CAP, spy_prices, "V8_Baseline", n_perms=2000)
    fprint(f"\nBASELINE V8: Sharpe={baseline_result['sharpe']:.2f}, "
           f"Sortino={baseline_result['sortino']:.2f}, "
           f"WR={baseline_result['win_rate']*100:.1f}%, "
           f"PF={baseline_result['profit_factor']:.2f}, "
           f"MDD={baseline_result['max_dd']*100:.1f}%, "
           f"CAGR={baseline_result['cagr']*100:.1f}%, "
           f"${CAP:.0f}->${baseline_result['final_equity']:.0f}, "
           f"gates={baseline_result['gates_passed']}/5, "
           f"trades={baseline_result['n_trades']}")

    # ── Run all stress tests ──
    all_results = {"baseline": baseline_result}

    # Stress Test 1: Monte Carlo
    all_results["monte_carlo"] = monte_carlo_bootstrap(baseline_trades, n_iterations=1000)

    # Stress Test 2: Cost Sensitivity
    all_results["cost_sensitivity"] = cost_sensitivity(
        baseline_rankings, close, high, low, atr_dict, spy_prices)

    # Stress Test 3: Haircut Sensitivity
    all_results["haircut_sensitivity"] = haircut_sensitivity(
        baseline_rankings, close, high, low, atr_dict, spy_prices)

    # Stress Test 4: IV Multiplier Sensitivity
    all_results["iv_multiplier_sensitivity"] = iv_multiplier_sensitivity(
        baseline_rankings, close, high, low, atr_dict, spy_prices)

    # Stress Test 5: Training Window Sensitivity
    all_results["training_window_sensitivity"] = training_window_sensitivity(
        df_records, V8_FEATURES, close, high, low, atr_dict, spy_prices)

    # Stress Test 6: Structural Edge Test
    all_results["structural_edge"] = structural_edge_test(
        baseline_rankings, close, high, low, atr_dict, spy_prices, n_trials=50)

    # ══════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'='*70}")
    fprint(f"COMPREHENSIVE SUMMARY")
    fprint(f"{'='*70}")

    fprint(f"\n--- Baseline V8 ---")
    fprint(f"  Sharpe: {baseline_result['sharpe']:.2f} | Sortino: {baseline_result['sortino']:.2f} | "
           f"WR: {baseline_result['win_rate']*100:.1f}% | PF: {baseline_result['profit_factor']:.2f}")
    fprint(f"  CAGR: {baseline_result['cagr']*100:.1f}% | MDD: {baseline_result['max_dd']*100:.1f}% | "
           f"Gates: {baseline_result['gates_passed']}/5 | Trades: {baseline_result['n_trades']}")

    mc = all_results["monte_carlo"]
    fprint(f"\n--- Monte Carlo Bootstrap (1000x) ---")
    fprint(f"  Sharpe 90% CI: [{mc['sharpe_5th']:.2f}, {mc['sharpe_95th']:.2f}] "
           f"(median={mc['sharpe_median']:.2f})")
    fprint(f"  CAGR 90% CI:   [{mc['cagr_5th']*100:.1f}%, {mc['cagr_95th']*100:.1f}%]")
    fprint(f"  MDD 90% CI:    [{mc['mdd_95th']*100:.1f}%, {mc['mdd_5th']*100:.1f}%]")
    fprint(f"  P(Sharpe>0):   {mc['sharpe_pct_positive']:.0f}%")

    fprint(f"\n--- Cost Sensitivity ---")
    fprint(f"  {'Level':<20} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'Final$':>8} {'Gates':>6}")
    fprint(f"  {'-'*56}")
    for label, m in all_results["cost_sensitivity"].items():
        if "error" not in m:
            fprint(f"  {label:<20} {m['sharpe']:>7.2f} {m['win_rate']*100:>5.1f}% "
                   f"{m['profit_factor']:>5.2f} ${m['final_equity']:>7,.0f} "
                   f"{m['gates_passed']}/{m['gates_total']}")

    fprint(f"\n--- Haircut Sensitivity ---")
    fprint(f"  {'Level':<20} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'Final$':>8} {'Gates':>6}")
    fprint(f"  {'-'*56}")
    for label, m in all_results["haircut_sensitivity"].items():
        if "error" not in m:
            fprint(f"  {label:<20} {m['sharpe']:>7.2f} {m['win_rate']*100:>5.1f}% "
                   f"{m['profit_factor']:>5.2f} ${m['final_equity']:>7,.0f} "
                   f"{m['gates_passed']}/{m['gates_total']}")

    fprint(f"\n--- IV Multiplier Sensitivity ---")
    fprint(f"  {'Level':<20} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'Final$':>8} {'Gates':>6}")
    fprint(f"  {'-'*56}")
    for label, m in all_results["iv_multiplier_sensitivity"].items():
        if "error" not in m:
            fprint(f"  {label:<20} {m['sharpe']:>7.2f} {m['win_rate']*100:>5.1f}% "
                   f"{m['profit_factor']:>5.2f} ${m['final_equity']:>7,.0f} "
                   f"{m['gates_passed']}/{m['gates_total']}")

    fprint(f"\n--- Training Window Sensitivity ---")
    fprint(f"  {'Window':<20} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'Trades':>7} {'Gates':>6}")
    fprint(f"  {'-'*56}")
    for label, m in all_results["training_window_sensitivity"].items():
        if "error" not in m:
            fprint(f"  {label:<20} {m['sharpe']:>7.2f} {m['win_rate']*100:>5.1f}% "
                   f"{m['profit_factor']:>5.2f} {m['n_trades']:>7} "
                   f"{m['gates_passed']}/{m['gates_total']}")

    se = all_results["structural_edge"]
    fprint(f"\n--- Structural Edge (ML vs Random) ---")
    fprint(f"  ML Sharpe:     {baseline_result['sharpe']:.2f}")
    fprint(f"  Random mean:   {se['random_sharpe_mean']:.2f} +/- {se['random_sharpe_std']:.2f}")
    fprint(f"  Random max:    {se['random_sharpe_max']:.2f}")
    if se['random_sharpe_mean'] > 0:
        fprint(f"  ML/Random:     {baseline_result['sharpe'] / se['random_sharpe_mean']:.2f}x")
    structural_pct = se['random_sharpe_mean'] / max(baseline_result['sharpe'], 0.01) * 100
    ml_pct = max(100 - structural_pct, 0)
    fprint(f"  Structure alpha: ~{structural_pct:.0f}% | ML alpha: ~{ml_pct:.0f}%")

    # ── Robustness verdict ──
    fprint(f"\n{'='*70}")
    fprint(f"ROBUSTNESS VERDICT")
    fprint(f"{'='*70}")

    # Check robustness criteria
    robust_checks = []

    # MC: 5th percentile Sharpe > 0
    mc_ok = mc['sharpe_5th'] > 0
    robust_checks.append(("MC 5th pctile Sharpe > 0", mc_ok, f"{mc['sharpe_5th']:.2f}"))

    # Cost: still profitable at 2x commission
    cost_2x = all_results["cost_sensitivity"].get("2.0x ($5.20)", {})
    cost_ok = cost_2x.get("sharpe", 0) > 0 if "error" not in cost_2x else False
    robust_checks.append(("Profitable at 2x commission", cost_ok,
                          f"Sharpe={cost_2x.get('sharpe', 'N/A')}"))

    # Haircut: still profitable at 25%
    hc_25 = all_results["haircut_sensitivity"].get("25%", {})
    hc_ok = hc_25.get("sharpe", 0) > 0 if "error" not in hc_25 else False
    robust_checks.append(("Profitable at 25% haircut", hc_ok,
                          f"Sharpe={hc_25.get('sharpe', 'N/A')}"))

    # ML alpha: ML Sharpe > 1.5x random mean
    ml_ok = baseline_result['sharpe'] > se['random_sharpe_mean'] * 1.5
    robust_checks.append(("ML > 1.5x random", ml_ok,
                          f"{baseline_result['sharpe']:.2f} vs {se['random_sharpe_mean']*1.5:.2f}"))

    # Training stability: all windows produce Sharpe > 1
    tw_results = all_results["training_window_sensitivity"]
    tw_sharpes = [m.get("sharpe", 0) for m in tw_results.values() if "error" not in m]
    tw_ok = all(s > 1.0 for s in tw_sharpes) if tw_sharpes else False
    robust_checks.append(("All training windows Sharpe > 1", tw_ok,
                          f"min={min(tw_sharpes):.2f}" if tw_sharpes else "N/A"))

    for check_name, passed, detail in robust_checks:
        status = "PASS" if passed else "FAIL"
        fprint(f"  [{status}] {check_name}: {detail}")

    n_passed = sum(1 for _, p, _ in robust_checks if p)
    fprint(f"\n  OVERALL: {n_passed}/{len(robust_checks)} robustness checks passed")
    if n_passed == len(robust_checks):
        fprint(f"  VERDICT: V8 is ROBUST — ready for production deployment")
    elif n_passed >= 3:
        fprint(f"  VERDICT: V8 is MOSTLY ROBUST — review failed checks")
    else:
        fprint(f"  VERDICT: V8 has ROBUSTNESS CONCERNS — investigate further")

    # ── Save results ──
    results_path = OUTPUT_DIR / "v8_stress_test_results.json"

    # Clean up non-serializable items
    save_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            cleaned = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    cleaned[k2] = {k3: v3 for k3, v3 in v2.items()
                                   if not isinstance(v3, (pd.Series, pd.DataFrame))}
                elif not isinstance(v2, (pd.Series, pd.DataFrame)):
                    cleaned[k2] = v2
            save_results[k] = cleaned
        else:
            save_results[k] = v

    save_results["robustness_checks"] = [
        {"name": n, "passed": p, "detail": d}
        for n, p, d in robust_checks
    ]

    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"v8_stress_{t0.strftime('%Y%m%d_%H%M')}"):
                # Baseline metrics
                mlflow.log_metric("baseline_sharpe", baseline_result["sharpe"])
                mlflow.log_metric("baseline_sortino", baseline_result["sortino"])
                mlflow.log_metric("baseline_cagr", baseline_result["cagr"])
                mlflow.log_metric("baseline_max_dd", baseline_result["max_dd"])
                mlflow.log_metric("baseline_win_rate", baseline_result["win_rate"])
                mlflow.log_metric("baseline_profit_factor", baseline_result["profit_factor"])
                mlflow.log_metric("baseline_n_trades", baseline_result["n_trades"])
                mlflow.log_metric("baseline_gates", baseline_result["gates_passed"])

                # MC metrics
                mlflow.log_metric("mc_sharpe_5th", mc["sharpe_5th"])
                mlflow.log_metric("mc_sharpe_95th", mc["sharpe_95th"])
                mlflow.log_metric("mc_sharpe_pct_positive", mc["sharpe_pct_positive"])
                mlflow.log_metric("mc_cagr_mean", mc["cagr_mean"])
                mlflow.log_metric("mc_mdd_mean", mc["mdd_mean"])

                # Structural edge
                mlflow.log_metric("random_sharpe_mean", se["random_sharpe_mean"])
                mlflow.log_metric("ml_alpha_pct", ml_pct)

                # Robustness
                mlflow.log_metric("robustness_passed", n_passed)
                mlflow.log_metric("robustness_total", len(robust_checks))

                mlflow.log_params({
                    "dte": DTE, "otm_pct": OTM_PCT, "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT, "commission": COMMISSION_RT_SPREAD,
                    "wf_train_periods": WF_TRAIN_PERIODS, "rebal_freq": REBAL_FREQ,
                    "n_features": len(V8_FEATURES), "top_k": TOP_K,
                    "capital": CAP, "mc_iterations": 1000, "random_trials": 50,
                })

                mlflow.log_artifact(str(results_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
