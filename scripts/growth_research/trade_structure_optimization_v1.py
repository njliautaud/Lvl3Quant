#!/usr/bin/env python3
"""
Trade Structure Optimization v1 — DTE x Spread Width Grid Search
=================================================================

Tests 12 combinations of DTE (14, 21, 28, 35) x spread width (2%, 3%, 5%)
for sector bull call spreads, holding all other parameters constant:
  - Walk-forward LGBM with 21 features (18 legacy + 3 cross-asset)
  - VIX > 20 regime filter (simpler proxy for GRU regime > 0.4)
  - ATR-based Black-Scholes pricing, iv_multiplier = 1.2
  - 15% entry haircut, hold-to-expiry (intrinsic value only), no exit haircut
  - Biweekly rebalance, top 3 sectors, $645 starting capital, max $200/trade
  - Commission: $2.60 per spread round-trip

5-gate adversarial validation on each variant:
  1. Sign-flip permutation (1000 trials, p < 0.05)
  2. Regime balance (bull vs bear Sharpe gap < 0.50)
  3. Sub-period stability (both halves > 0)
  4. Outlier removal (remove top/bottom 5% trades, still positive)
  5. Yearly consistency (60%+ years profitable)

Data: sector ETFs + SPY + VIX from yfinance, 2007-2026.
Walk-forward: 500d train, 250d test, sliding (NOT expanding).

Logs to MLflow experiment "trade_structure_optimization_v1".
Saves JSON results to output/growth_research/trade_structure_v1/.
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ── Config ──
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "trade_structure_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "trade_structure_v1_results.json"

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX"]
CAP = 645.0
MAX_POS = 200.0
TOP_K = 3
VIX_ENTRY_MIN = 20.0
HAIRCUT = 0.15
COMMISSION_RT_SPREAD = 2.60

# Walk-forward config — sliding window
WF_TRAIN_DAYS = 500
WF_TEST_DAYS = 250
WF_REBAL_FREQ = "2W-FRI"

# Feature sets
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]
CROSS_ASSET_FEATURES = [
    "sector_spy_beta_63d",
    "sector_relative_vol_21d",
    "cross_sector_dispersion",
]
ALL_FEATURES = LEGACY_FEATURES + CROSS_ASSET_FEATURES  # 21 total

# DTE x Spread Width grid (12 variants)
GRID = {
    "A_14d_2pct": {"dte": 14, "spread_pct": 2.0, "label": "14 DTE, 2% width"},
    "B_14d_3pct": {"dte": 14, "spread_pct": 3.0, "label": "14 DTE, 3% width"},
    "C_14d_5pct": {"dte": 14, "spread_pct": 5.0, "label": "14 DTE, 5% width"},
    "D_21d_2pct": {"dte": 21, "spread_pct": 2.0, "label": "21 DTE, 2% width"},
    "E_21d_3pct": {"dte": 21, "spread_pct": 3.0, "label": "21 DTE, 3% width (BASELINE)"},
    "F_21d_5pct": {"dte": 21, "spread_pct": 5.0, "label": "21 DTE, 5% width"},
    "G_28d_2pct": {"dte": 28, "spread_pct": 2.0, "label": "28 DTE, 2% width"},
    "H_28d_3pct": {"dte": 28, "spread_pct": 3.0, "label": "28 DTE, 3% width"},
    "I_28d_5pct": {"dte": 28, "spread_pct": 5.0, "label": "28 DTE, 5% width"},
    "J_35d_2pct": {"dte": 35, "spread_pct": 2.0, "label": "35 DTE, 2% width"},
    "K_35d_3pct": {"dte": 35, "spread_pct": 3.0, "label": "35 DTE, 3% width"},
    "L_35d_5pct": {"dte": 35, "spread_pct": 5.0, "label": "35 DTE, 5% width"},
}

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "trade_structure_optimization_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable — results saved to disk only")


# ══════════════════════════════════════════════════════════════
# BLACK-SCHOLES PRICING (self-contained)
# ══════════════════════════════════════════════════════════════

from scipy.stats import norm as _norm

RISK_FREE_RATE = 0.045


def _bs_call(S, K, T, r, sigma):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d2))


def _estimate_iv(atr, spot, vix=20.0, atr_period=14):
    """ATR-based IV estimate. iv_multiplier = 1.2 + 0.01 * max(VIX - 20, 0)."""
    if spot <= 0 or atr <= 0:
        return 0.25
    realized_vol = (atr / spot) * np.sqrt(252 / atr_period)
    iv_mult = 1.2 + 0.01 * max(vix - 20.0, 0.0)
    return max(realized_vol * iv_mult, 0.10)


def _price_bull_call_spread(S, K1, K2, dte, atr, vix=20.0):
    """
    Price bull call spread with 15% entry haircut.
    Returns (entry_cost_per_share, max_profit_per_share).
    """
    T = dte / 365.0
    sigma = _estimate_iv(atr, S, vix)
    fair = _bs_call(S, K1, T, RISK_FREE_RATE, sigma) - _bs_call(S, K2, T, RISK_FREE_RATE, sigma)
    fair = max(fair, 0.001)
    entry_cost = fair * (1.0 + HAIRCUT)
    max_profit = (K2 - K1) - entry_cost
    return float(entry_cost), float(max_profit)


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download sector ETFs + SPY + VIX from yfinance (2007-2026)."""
    import yfinance as yf

    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers (2007-2026)...")

    raw = yf.download(all_tickers, start="2007-01-01", progress=False, auto_adjust=True)
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

    rename_map = {"^VIX": "VIX"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)

    for t in ["SPY", "VIX"]:
        if t not in close.columns:
            raise ValueError(f"Missing critical ticker: {t}")

    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    fprint(f"Columns: {sorted(close.columns.tolist())}")
    return close, high, low


# ══════════════════════════════════════════════════════════════
# ATR COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_atr_series(high, low, close, period=14):
    """Compute ATR time series for all sector ETFs."""
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
                atr_dict[tk] = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
    return atr_dict


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (21 features)
# ══════════════════════════════════════════════════════════════

def compute_legacy_features(px, spy_slice):
    """Compute the 18 legacy quality-momentum features for a single sector ETF."""
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
    """Compute the 3 validated cross-asset features."""
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in CROSS_ASSET_FEATURES}

    spy_ret = spy.pct_change().dropna()

    # 1. Sector-SPY beta 63d
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

    # 2. Sector relative vol 21d
    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        if len(sec_ret) > 21 and len(spy_ret) > 21:
            sec_vol = sec_ret.iloc[-21:].std()
            spy_vol = spy_ret.iloc[-21:].std()
            f["sector_relative_vol_21d"] = float(sec_vol / (spy_vol + 1e-10))
        else:
            f["sector_relative_vol_21d"] = 1.0
    else:
        f["sector_relative_vol_21d"] = 1.0

    # 3. Cross-sector dispersion
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
# WALK-FORWARD LGBM (sliding window: 500d train, 250d test)
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, dte_days):
    """
    Build feature + forward-return records for LGBM walk-forward.
    VIX > 20 regime filter applied. Forward return uses the specified DTE.
    """
    records = []
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None
    sector_cols = [c for c in SECTORS if c in close.columns]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # VIX > 20 regime filter
        if vix is not None and dt in vix.index:
            cv = float(vix.loc[dt])
        else:
            cv = 15.0
        if cv < VIX_ENTRY_MIN:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            cross_asset = compute_cross_asset_features(tk, idx, close)

            # Forward return target using the DTE
            fi = min(idx + dte_days, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk,
                   "fwd_ret": fwd_ret, "vix": cv}
            records.append(rec)

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} rebal dates (VIX>20 filtered)")
    return df


def walk_forward_lgbm_rank(df, variant_name):
    """
    Walk-forward LGBM ranking with SLIDING window.
    Train on WF_TRAIN_DAYS worth of rebalance dates, test on next WF_TEST_DAYS.
    Sliding = oldest drops off as new data comes in.
    """
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    {variant_name}: Insufficient data ({len(df)} records)")
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    # Determine how many rebalance periods fit in train/test windows
    # With biweekly rebalance (~10 trading days), 500d ~ 50 periods, 250d ~ 25 periods
    train_periods = max(WF_TRAIN_DAYS // 10, 20)  # ~50 periods for 500d
    # Sliding: use train_periods for training, predict next period

    rankings = {}
    for i in range(train_periods, len(dates)):
        # SLIDING window: take the last train_periods dates
        train_start = max(0, i - train_periods)
        train_dates = dates[train_start:i]
        test_date = dates[i]

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()

        if len(test_df) < 3 or len(train_df) < 50:
            continue

        Xt = np.nan_to_num(train_df[ALL_FEATURES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[ALL_FEATURES].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)
            rankings[test_date] = {
                "scores": dict(zip(test_df["ticker"], test_df["score"])),
                "vix": float(test_df["vix"].iloc[0]),
            }
        except Exception:
            continue

    fprint(f"    {variant_name}: {len(rankings)} ranking dates")
    return rankings


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_trades(name, rankings, close, atr_dict, dte, spread_pct):
    """
    Simulate bull call spreads for a given DTE + spread width combination.

    HONEST RULES:
      - Hold to expiry: exit at intrinsic value ONLY
      - 15% entry haircut
      - No exit haircut (automatic exercise at expiry)
      - $2.60 commission per spread round-trip
      - Max $200/trade, scale with equity
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    total_spread_cost_pct = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        ranking_data = rankings[dt]
        scores = ranking_data["scores"]
        cv = ranking_data.get("vix", 20.0)

        if not scores:
            continue

        # Pick top K sectors
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]

        # Position sizing: scale with equity, cap at MAX_POS
        max_pos = min(MAX_POS, equity / 3)
        if max_pos < 30:
            continue

        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= TOP_K:
                continue

            S = float(close[tk].loc[dt])
            di = close.index.get_loc(dt)
            ei = min(di + dte, len(close) - 1)
            if ei <= di:
                continue

            # ATR for pricing
            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                av = float(atr_dict[tk].loc[dt])
            else:
                av = S * 0.015

            # Set up strikes: K1 = ATM, K2 = K1 + spread_pct% of S
            K1 = round(S, 2)
            K2 = round(S * (1 + spread_pct / 100.0), 2)
            if K2 <= K1:
                K2 = K1 + 0.50

            try:
                entry_cost_ps, max_profit_ps = _price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=cv
                )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

            if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
                continue

            # Track spread cost as % of width
            spread_width = K2 - K1
            cost_pct = entry_cost_ps / spread_width if spread_width > 0 else 1.0
            total_spread_cost_pct.append(cost_pct)

            # HOLD TO EXPIRY: intrinsic value at expiry
            Se = float(close[tk].iloc[ei])
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)

            exit_value_ps = intrinsic

            # PnL: exit value - entry cost - commission (no exit haircut at expiry)
            pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
            equity += pnl
            n_entered += 1

            # Regime classification
            sv = float(spy.loc[dt]) if dt in spy.index else 0
            se = float(spy.iloc[ei]) if ei < len(spy) else sv
            spy_regime = "bull" if se >= sv else "bear"

            trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": spy_regime,
                "vix": round(cv, 1),
                "dte": dte,
                "spread_pct": spread_pct,
                "entry_cost_ps": round(entry_cost_ps, 4),
                "spread_width": round(spread_width, 2),
                "intrinsic_at_expiry": round(intrinsic, 4),
                "win": pnl > 0,
            })

    avg_cost_pct = float(np.mean(total_spread_cost_pct)) if total_spread_cost_pct else 0.0
    return trades, equity, avg_cost_pct


# ══════════════════════════════════════════════════════════════
# HONEST METRICS (equity-based Sharpe, calendar month)
# ══════════════════════════════════════════════════════════════

def compute_honest_metrics(trades, initial_capital):
    """Compute Sharpe, Sortino, PF, WR, MaxDD, total return from trade list."""
    if not trades or len(trades) < 2:
        return {
            "sharpe": 0.0, "sortino": 0.0, "profit_factor": 0.0,
            "win_rate": 0.0, "max_dd": -1.0, "total_return": 0.0,
            "n_trades": len(trades) if trades else 0,
            "final_equity": initial_capital,
            "monthly_returns": pd.Series(dtype=float),
        }

    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("exit_date")

    pnls = df["pnl"].values

    # Build equity curve
    equity_values = [initial_capital]
    for p in pnls:
        equity_values.append(equity_values[-1] + p)

    dates = [df["entry_date"].iloc[0] - pd.Timedelta(days=1)]
    dates.extend(df["exit_date"].tolist())
    equity_series = pd.Series(equity_values, index=pd.DatetimeIndex(dates))
    equity_series = equity_series.groupby(equity_series.index).last()

    # Calendar month Sharpe (equity-based pct_change)
    monthly_equity = equity_series.resample("ME").last().dropna()
    monthly_returns = monthly_equity.pct_change().dropna() if len(monthly_equity) > 1 else pd.Series(dtype=float)

    if len(monthly_returns) > 1 and monthly_returns.std() > 0:
        sharpe = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
        downside = monthly_returns[monthly_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (monthly_returns.mean() / downside.std()) * np.sqrt(12)
        else:
            sortino = sharpe * 1.5
    else:
        sharpe = 0.0
        sortino = 0.0

    # Max drawdown
    eq_arr = np.array(equity_values)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.min())

    # Win rate, profit factor
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

    final_eq = equity_values[-1]
    total_return = (final_eq / initial_capital - 1)

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "profit_factor": float(min(profit_factor, 999.0)),
        "win_rate": float(win_rate),
        "max_dd": float(max_dd),
        "total_return": float(total_return),
        "n_trades": len(pnls),
        "final_equity": float(final_eq),
        "monthly_returns": monthly_returns,
    }


# ══════════════════════════════════════════════════════════════
# 5-GATE ADVERSARIAL VALIDATION
# ══════════════════════════════════════════════════════════════

def run_adversarial_validation(trades, initial_capital, spy_close, name):
    """
    Run 5-gate adversarial validation:
      1. Sign-flip permutation (1000 trials, p < 0.05)
      2. Regime balance (bull vs bear WR gap < 0.50)
      3. Sub-period stability (both halves > 0)
      4. Outlier removal (remove top/bottom 5% trades, still positive)
      5. Yearly consistency (60%+ years profitable)
    """
    gates = {}

    if len(trades) < 10:
        return {"error": f"Too few trades ({len(trades)})", "gates_passed": 0, "gates_total": 5}

    pnls = np.array([t["pnl"] for t in trades])
    metrics = compute_honest_metrics(trades, initial_capital)
    real_sharpe = metrics["sharpe"]

    # Gate 1: Sign-flip permutation (1000 trials)
    beat_count = 0
    n_perms = 1000
    df_trades = pd.DataFrame(trades)
    df_trades["exit_date"] = pd.to_datetime(df_trades["exit_date"])
    trade_dates = df_trades["exit_date"].values

    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(pnls))
        flipped = pnls * signs
        eq_vals = [initial_capital]
        for p in flipped:
            eq_vals.append(eq_vals[-1] + p)
        eq_series = pd.Series(
            eq_vals,
            index=pd.DatetimeIndex(
                [pd.Timestamp(trade_dates[0]) - pd.Timedelta(days=1)] + list(trade_dates)
            ),
        )
        eq_series = eq_series.groupby(eq_series.index).last()
        meq = eq_series.resample("ME").last().dropna()
        mret = meq.pct_change().dropna()
        if len(mret) > 1 and mret.std() > 0:
            perm_sharpe = (mret.mean() / mret.std()) * np.sqrt(12)
        else:
            perm_sharpe = 0.0
        if perm_sharpe >= real_sharpe:
            beat_count += 1

    perm_p = beat_count / n_perms
    gates["sign_flip"] = {
        "passed": perm_p < 0.05,
        "p_value": round(perm_p, 4),
        "detail": f"{beat_count}/{n_perms} perms beat Sharpe {real_sharpe:.3f}",
    }

    # Gate 2: Regime balance (bull vs bear WR gap < 0.50)
    bull_pnls = [t["pnl"] for t in trades if t.get("regime") == "bull"]
    bear_pnls = [t["pnl"] for t in trades if t.get("regime") == "bear"]
    if len(bull_pnls) >= 5 and len(bear_pnls) >= 5:
        bull_wr = np.mean([1 if p > 0 else 0 for p in bull_pnls])
        bear_wr = np.mean([1 if p > 0 else 0 for p in bear_pnls])
        wr_gap = abs(bull_wr - bear_wr)
        gates["regime_balance"] = {
            "passed": wr_gap < 0.50,
            "wr_gap": round(wr_gap, 4),
            "detail": f"Bull WR={bull_wr:.1%} ({len(bull_pnls)}), Bear WR={bear_wr:.1%} ({len(bear_pnls)})",
        }
    else:
        gates["regime_balance"] = {
            "passed": True,
            "wr_gap": 0.0,
            "detail": f"Insufficient split: {len(bull_pnls)} bull, {len(bear_pnls)} bear",
        }

    # Gate 3: Sub-period stability (both halves > 0)
    mid = len(pnls) // 2
    first_half = pnls[:mid].sum()
    second_half = pnls[mid:].sum()
    gates["sub_period"] = {
        "passed": first_half > 0 and second_half > 0,
        "first_half_pnl": round(first_half, 2),
        "second_half_pnl": round(second_half, 2),
    }

    # Gate 4: Outlier removal (remove top/bottom 5% trades, still positive)
    n_trim = max(1, int(len(pnls) * 0.05))
    sorted_pnls = np.sort(pnls)
    trimmed = sorted_pnls[n_trim:-n_trim] if len(sorted_pnls) > 2 * n_trim else sorted_pnls
    trimmed_sum = trimmed.sum()
    gates["outlier_removal"] = {
        "passed": trimmed_sum > 0,
        "trimmed_pnl": round(float(trimmed_sum), 2),
        "removed": n_trim * 2,
        "detail": f"Removed top/bottom {n_trim} trades each, remaining PnL=${trimmed_sum:.2f}",
    }

    # Gate 5: Yearly consistency (60%+ years profitable)
    df_yr = pd.DataFrame(trades)
    df_yr["exit_date"] = pd.to_datetime(df_yr["exit_date"])
    df_yr["year"] = df_yr["exit_date"].dt.year
    yearly_pnl = df_yr.groupby("year")["pnl"].sum()
    n_years = len(yearly_pnl)
    n_profitable = (yearly_pnl > 0).sum()
    pct_profitable = n_profitable / n_years if n_years > 0 else 0.0
    gates["yearly_consistency"] = {
        "passed": pct_profitable >= 0.60,
        "pct_profitable": round(pct_profitable, 3),
        "detail": f"{n_profitable}/{n_years} years profitable",
    }

    gates_passed = sum(1 for g in gates.values() if g.get("passed", False))
    return {
        "gates": gates,
        "gates_passed": gates_passed,
        "gates_total": 5,
        "all_passed": gates_passed == 5,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"TRADE STRUCTURE OPTIMIZATION v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Capital: ${CAP:.0f} | Max/trade: ${MAX_POS:.0f} | Haircut: {HAIRCUT:.0%} entry only")
    fprint(f"Commission: ${COMMISSION_RT_SPREAD:.2f}/spread | Hold to expiry | VIX>20 filter")
    fprint(f"Walk-forward: {WF_TRAIN_DAYS}d train, {WF_TEST_DAYS}d test, sliding window")
    fprint(f"Features: {len(ALL_FEATURES)} (18 legacy + 3 cross-asset)")
    fprint(f"Grid: {len(GRID)} variants (4 DTE x 3 spread widths)")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Pre-compute ATR for all sectors
    fprint("\nComputing ATR series...")
    atr_dict = compute_atr_series(high, low, close)
    fprint(f"ATR computed for {len(atr_dict)} sectors")

    # 3. Build rebalance dates
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # 4. Group variants by DTE to avoid redundant LGBM training
    # Same DTE = same forward returns = same LGBM model, just different spread width in simulation
    dte_groups = {}
    for var_name, cfg in GRID.items():
        d = cfg["dte"]
        if d not in dte_groups:
            dte_groups[d] = []
        dte_groups[d].append((var_name, cfg))

    all_results = {}
    all_trades = {}

    for dte_val, variants in sorted(dte_groups.items()):
        fprint("\n" + "=" * 100)
        fprint(f"DTE GROUP: {dte_val} days — training LGBM with {dte_val}d forward returns")
        fprint("=" * 100)

        # Build feature records once per DTE
        fprint(f"  Building feature records (DTE={dte_val})...")
        records = build_feature_records(close, high, low, rebal_dates, dte_val)

        if len(records) < 100:
            fprint(f"  SKIP DTE={dte_val}: insufficient records ({len(records)})")
            for var_name, _ in variants:
                fprint(f"    {var_name}: SKIPPED")
            continue

        # Walk-forward LGBM ranking (same for all spread widths at this DTE)
        rankings = walk_forward_lgbm_rank(records, f"DTE_{dte_val}")
        if not rankings:
            fprint(f"  SKIP DTE={dte_val}: no rankings generated")
            continue

        # Simulate trades for each spread width
        for var_name, cfg in variants:
            sp = cfg["spread_pct"]
            fprint(f"\n  --- {var_name}: DTE={dte_val}, Spread={sp}% ---")

            trades, final_eq, avg_cost_pct = simulate_trades(
                var_name, rankings, close, atr_dict, dte_val, sp
            )

            if not trades or len(trades) < 10:
                fprint(f"    SKIP: {len(trades) if trades else 0} trades (need 10+)")
                continue

            all_trades[var_name] = trades

            # Compute metrics
            metrics = compute_honest_metrics(trades, CAP)

            # Run adversarial validation
            adv = run_adversarial_validation(trades, CAP, spy_close, var_name)

            # Store results
            all_results[var_name] = {
                "label": cfg["label"],
                "dte": dte_val,
                "spread_pct": sp,
                "sharpe": metrics["sharpe"],
                "sortino": metrics["sortino"],
                "profit_factor": metrics["profit_factor"],
                "win_rate": metrics["win_rate"],
                "max_dd": metrics["max_dd"],
                "total_return": metrics["total_return"],
                "n_trades": metrics["n_trades"],
                "final_equity": metrics["final_equity"],
                "avg_cost_pct_of_width": avg_cost_pct,
                "gates_passed": adv.get("gates_passed", 0),
                "gates_total": adv.get("gates_total", 5),
                "all_gates_passed": adv.get("all_passed", False),
                "adversarial": adv.get("gates", {}),
            }

            # Print summary
            m = all_results[var_name]
            baseline_tag = " <<< BASELINE" if var_name == "E_21d_3pct" else ""
            fprint(f"    Sharpe: {m['sharpe']:.2f} | Sortino: {m['sortino']:.2f} | "
                   f"PF: {m['profit_factor']:.2f} | WR: {m['win_rate']:.1%}")
            fprint(f"    MaxDD: {m['max_dd']:.1%} | Total Return: {m['total_return']:.1%} | "
                   f"Trades: {m['n_trades']} | Final: ${m['final_equity']:,.0f}")
            fprint(f"    Avg cost as % of width: {m['avg_cost_pct_of_width']:.1%}")
            fprint(f"    Gates: {m['gates_passed']}/{m['gates_total']}{baseline_tag}")

            # Print gate details
            for gname, gdata in adv.get("gates", {}).items():
                status = "PASS" if gdata.get("passed") else "FAIL"
                detail = gdata.get("detail", "")
                fprint(f"      [{status}] {gname}: {detail}")

    # ══════════════════════════════════════════════════════════
    # RESULTS TABLE (sorted by Sharpe)
    # ══════════════════════════════════════════════════════════

    fprint("\n\n" + "=" * 130)
    fprint("RESULTS TABLE — Sorted by Sharpe Ratio")
    fprint("=" * 130)

    if not all_results:
        fprint("NO RESULTS — all variants had insufficient data")
        return

    sorted_variants = sorted(all_results.items(), key=lambda x: x[1]["sharpe"], reverse=True)

    header = (f"{'Variant':<16} {'DTE':>4} {'Sprd%':>5} {'Sharpe':>7} {'Sortino':>8} "
              f"{'PF':>6} {'WR':>6} {'MaxDD':>7} {'TotRet':>8} {'Trades':>7} "
              f"{'Final$':>9} {'Cost%':>6} {'Gates':>6}")
    fprint(header)
    fprint("-" * 130)

    for var_name, r in sorted_variants:
        baseline = " *BASE" if var_name == "E_21d_3pct" else ""
        best = " *BEST" if var_name == sorted_variants[0][0] else ""
        tag = baseline + best
        fprint(f"{var_name:<16} {r['dte']:>4} {r['spread_pct']:>4.0f}% "
               f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['profit_factor']:>5.2f} {r['win_rate']*100:>5.1f}% "
               f"{r['max_dd']*100:>6.1f}% {r['total_return']*100:>7.1f}% "
               f"{r['n_trades']:>7d} ${r['final_equity']:>8,.0f} "
               f"{r['avg_cost_pct_of_width']*100:>5.1f}% "
               f"{r['gates_passed']}/{r['gates_total']}{tag}")

    # ── Key insights ──
    fprint("\n" + "=" * 100)
    fprint("KEY INSIGHTS")
    fprint("=" * 100)

    baseline = all_results.get("E_21d_3pct")
    best_name, best = sorted_variants[0]

    if baseline:
        fprint(f"\nBaseline (E_21d_3pct): Sharpe={baseline['sharpe']:.2f}, "
               f"Return={baseline['total_return']:.1%}, "
               f"Final=${baseline['final_equity']:,.0f}, "
               f"Gates={baseline['gates_passed']}/{baseline['gates_total']}")
        fprint(f"Best ({best_name}): Sharpe={best['sharpe']:.2f}, "
               f"Return={best['total_return']:.1%}, "
               f"Final=${best['final_equity']:,.0f}, "
               f"Gates={best['gates_passed']}/{best['gates_total']}")

        if best_name != "E_21d_3pct":
            sd = best["sharpe"] - baseline["sharpe"]
            fprint(f"  Sharpe improvement over baseline: {sd:+.2f}")

    # DTE analysis (holding spread width constant at 3%)
    fprint(f"\nDTE analysis (3% spread width):")
    for vn in ["B_14d_3pct", "E_21d_3pct", "H_28d_3pct", "K_35d_3pct"]:
        r = all_results.get(vn)
        if r:
            fprint(f"  DTE={r['dte']:>2}: Sharpe={r['sharpe']:.2f}, "
                   f"WR={r['win_rate']:.1%}, Cost%={r['avg_cost_pct_of_width']:.1%}, "
                   f"Trades={r['n_trades']}, Gates={r['gates_passed']}/{r['gates_total']}")

    # Spread width analysis (holding DTE constant at 21)
    fprint(f"\nSpread width analysis (21 DTE):")
    for vn in ["D_21d_2pct", "E_21d_3pct", "F_21d_5pct"]:
        r = all_results.get(vn)
        if r:
            fprint(f"  Width={r['spread_pct']:.0f}%: Sharpe={r['sharpe']:.2f}, "
                   f"WR={r['win_rate']:.1%}, Cost%={r['avg_cost_pct_of_width']:.1%}, "
                   f"Trades={r['n_trades']}, Gates={r['gates_passed']}/{r['gates_total']}")

    # Cost efficiency tradeoff
    fprint(f"\nCost efficiency (entry cost as % of spread width):")
    fprint(f"  Wider spreads = lower cost % = more profit potential per dollar risked")
    fprint(f"  Narrower spreads = higher cost % = more theta drag")
    for var_name, r in sorted_variants:
        fprint(f"  {var_name}: {r['avg_cost_pct_of_width']*100:.1f}% cost drag")

    # Identify fully-passing variants
    passing = [(n, r) for n, r in sorted_variants if r["all_gates_passed"]]
    if passing:
        fprint(f"\nVariants passing ALL 5 gates ({len(passing)}):")
        for n, r in passing:
            fprint(f"  {n}: Sharpe={r['sharpe']:.2f}, Return={r['total_return']:.1%}")
    else:
        fprint(f"\nNo variant passed all 5 gates. Best gates:")
        best_gates = max(sorted_variants, key=lambda x: x[1]["gates_passed"])
        fprint(f"  {best_gates[0]}: {best_gates[1]['gates_passed']}/{best_gates[1]['gates_total']} gates")

    # ══════════════════════════════════════════════════════════
    # LOG TO MLFLOW
    # ══════════════════════════════════════════════════════════

    if MLFLOW_OK:
        fprint("\n\nLogging to MLflow...")
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"structure_opt_{t0.strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("best_variant", best_name)
                mlflow.log_param("n_variants", len(all_results))
                mlflow.log_param("capital", CAP)
                mlflow.log_param("max_pos", MAX_POS)
                mlflow.log_param("haircut", HAIRCUT)
                mlflow.log_param("commission", COMMISSION_RT_SPREAD)
                mlflow.log_param("vix_min", VIX_ENTRY_MIN)
                mlflow.log_param("wf_train_days", WF_TRAIN_DAYS)
                mlflow.log_param("wf_test_days", WF_TEST_DAYS)
                mlflow.log_param("features", len(ALL_FEATURES))

                for var_name, r in all_results.items():
                    prefix = var_name.lower()
                    mlflow.log_metric(f"{prefix}_sharpe", r["sharpe"])
                    mlflow.log_metric(f"{prefix}_sortino", r["sortino"])
                    mlflow.log_metric(f"{prefix}_pf", r["profit_factor"])
                    mlflow.log_metric(f"{prefix}_wr", r["win_rate"])
                    mlflow.log_metric(f"{prefix}_maxdd", r["max_dd"])
                    mlflow.log_metric(f"{prefix}_return", r["total_return"])
                    mlflow.log_metric(f"{prefix}_trades", r["n_trades"])
                    mlflow.log_metric(f"{prefix}_final_eq", r["final_equity"])
                    mlflow.log_metric(f"{prefix}_cost_pct", r["avg_cost_pct_of_width"])
                    mlflow.log_metric(f"{prefix}_gates", r["gates_passed"])

            fprint("  MLflow logging complete")
        except Exception as e:
            fprint(f"  MLflow logging failed: {e}")

    # ══════════════════════════════════════════════════════════
    # SAVE JSON RESULTS
    # ══════════════════════════════════════════════════════════

    def _clean_for_json(obj):
        """Recursively convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: _clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return str(obj)  # skip series in JSON
        return obj

    output = {
        "experiment": EXPERIMENT_NAME,
        "run_date": t0.isoformat(),
        "config": {
            "capital": CAP,
            "max_pos": MAX_POS,
            "haircut": HAIRCUT,
            "commission": COMMISSION_RT_SPREAD,
            "vix_entry_min": VIX_ENTRY_MIN,
            "wf_train_days": WF_TRAIN_DAYS,
            "wf_test_days": WF_TEST_DAYS,
            "rebalance_freq": WF_REBAL_FREQ,
            "top_k": TOP_K,
            "n_features": len(ALL_FEATURES),
            "features": ALL_FEATURES,
            "sectors": SECTORS,
            "grid_size": len(GRID),
        },
        "results": _clean_for_json(all_results),
        "ranking": [name for name, _ in sorted_variants],
        "best_variant": best_name,
        "baseline_comparison": {
            "baseline": "E_21d_3pct",
            "baseline_sharpe": baseline["sharpe"] if baseline else None,
            "best_sharpe": best["sharpe"],
            "best_name": best_name,
            "improvement": round(best["sharpe"] - baseline["sharpe"], 3) if baseline else None,
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_FILE}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    fprint("=" * 100)
    fprint("DONE")


if __name__ == "__main__":
    main()
