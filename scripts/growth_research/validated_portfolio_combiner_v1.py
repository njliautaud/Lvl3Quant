#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""
Validated Portfolio Combiner v1
================================
Combines ONLY adversarial-validated strategies into a proper portfolio backtest.

Strategies (all passed adversarial audit):
  1. Market-Neutral L/S Rotation — Long top-3 + short bottom-3 LGBM-ranked
     sector ETFs, monthly rebalance. Sharpe 2.64, MDD -6.9%.
  2. Equity Rotation Top-2 — Buy top-2 LGBM-ranked sectors, weekly rebalance.
     Sharpe 1.40, CAGR 24.1%.
  3. SPY Iron Condor Income — Non-directional premium selling on SPY.
     10-delta short strikes, $5 wide, DTE=30. Sharpe 3.55, WR 94.7%.

Portfolio Combination Variants:
  A: Equal weight (33/33/34)
  B: Risk parity (inverse vol weighted)
  C: Best two only (market-neutral + equity rotation, skip options)
  D: Growth-weighted (50% equity rotation + 30% market-neutral + 20% income)

Capital: $100K starting.
Walk-forward: SLIDING 500d train for LGBM.
Adversarial: 5-gate on each combo.
Output: stdout + JSON + MLflow.
"""

import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from scipy.stats import norm, percentileofscore
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ── Unbuffered print ─────────────────────────────────────────────────────────
_print = print
def fprint(*a, **kw):
    kw['flush'] = True
    _print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, **kw)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]  # auto-detect from script location
OUTPUT_DIR = BASE / "output" / "growth_research" / "validated_portfolio_combiner_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUTPUT_DIR / "results.json"

# ── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_OK = False
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "validated_portfolio_combiner_v1"
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MLFLOW_OK = True
    fprint("MLflow connected")
except Exception as e:
    fprint(f"MLflow unavailable ({e}), continuing without tracking")

# ── Constants ────────────────────────────────────────────────────────────────
START_DATE = "2008-01-01"
CAPITAL = 100_000.0

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "TLT", "SHY", "HYG", "GLD"]
VIX_TICKERS = ["^VIX", "^VIX3M"]
ALL_EQUITY_TICKERS = SECTOR_ETFS + EXTRA_TICKERS

LGBM_TRAIN_DAYS = 500  # sliding window
N_PERMUTATIONS = 300

# 21 LGBM features: 18 legacy quality-momentum + 3 cross-asset
FEATURE_COLS = [
    # 18 legacy quality-momentum
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high",
    "mom_accel", "pct_pos_months_12m", "sortino_63d", "calmar_1y",
    "trend_r2_63d", "trend_slope_63d", "rsi_14",
    # 3 cross-asset
    "sector_spy_beta_63d", "sector_relative_vol_21d", "cross_sector_dispersion",
]
assert len(FEATURE_COLS) == 21, f"Expected 21 features, got {len(FEATURE_COLS)}"

# Iron condor parameters
IC_DELTA = 0.10
IC_WIDTH = 5
IC_DTE = 30
IC_HAIRCUT = 0.15      # 15% BS haircut
IC_COMMISSION = 2.60   # per IC (4 legs)


# ═══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES PRICING (self-contained)
# ═══════════════════════════════════════════════════════════════════════════════

def bs_price(S, K, T, sigma, r=0.04, opt='call'):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if opt == 'call' else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def strike_at_delta(S, T, sigma, target_delta, r=0.04, opt='call'):
    """Find strike for a given delta via bisection."""
    if T <= 0:
        return S
    lo, hi = (S * 0.7, S * 1.5) if opt == 'call' else (S * 0.5, S * 1.3)
    for _ in range(60):
        mid = (lo + hi) / 2
        d1 = (np.log(S / mid) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if opt == 'call':
            if norm.cdf(d1) > target_delta:
                lo = mid
            else:
                hi = mid
        else:
            if abs(-norm.cdf(-d1)) > target_delta:
                hi = mid
            else:
                lo = mid
    return round((lo + hi) / 2)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_data():
    """Download all required price data."""
    fprint(f"Downloading data from {START_DATE}...")
    all_tickers = ALL_EQUITY_TICKERS + VIX_TICKERS
    data = yf.download(all_tickers, start=START_DATE, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        volume = data["Volume"] if "Volume" in data.columns.get_level_values(0) else pd.DataFrame(index=data.index)
        high = data["High"] if "High" in data.columns.get_level_values(0) else pd.DataFrame(index=data.index)
        low = data["Low"] if "Low" in data.columns.get_level_values(0) else pd.DataFrame(index=data.index)
    else:
        close = data[["Close"]].copy()
        volume = data[["Volume"]].copy() if "Volume" in data.columns else pd.DataFrame(index=data.index)
        high = data[["High"]].copy() if "High" in data.columns else pd.DataFrame(index=data.index)
        low = data[["Low"]].copy() if "Low" in data.columns else pd.DataFrame(index=data.index)

    close.columns = [str(c).strip() for c in close.columns]
    volume.columns = [str(c).strip() for c in volume.columns]
    high.columns = [str(c).strip() for c in high.columns]
    low.columns = [str(c).strip() for c in low.columns]
    close = close.ffill()

    # Extract VIX series
    vix = close["^VIX"].copy() if "^VIX" in close.columns else None
    vix3m = close["^VIX3M"].copy() if "^VIX3M" in close.columns else None

    fprint(f"Data: {len(close)} days, {close.shape[1]} tickers, "
           f"{close.index[0].date()} to {close.index[-1].date()}")
    return close, volume, high, low, vix, vix3m


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (21 features)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_features(close, volume, high, low, spy_close):
    """Compute 21 features for each sector ETF. Returns dict: ticker -> DataFrame."""
    fprint("Computing 21 LGBM features...")
    features = {}
    spy_ret = spy_close.pct_change()
    spy_log_ret = np.log(spy_close / spy_close.shift(1))

    # Cross-sector dispersion (same for all sectors on a given day)
    sector_rets = pd.DataFrame({t: close[t].pct_change() for t in SECTOR_ETFS if t in close.columns})
    cross_dispersion = sector_rets.std(axis=1).rolling(21).mean()  # cross-sectional std, rolling avg

    for ticker in SECTOR_ETFS:
        if ticker not in close.columns:
            fprint(f"  SKIP {ticker} — not in data")
            continue
        p = close[ticker].dropna()
        v = volume[ticker] if ticker in volume.columns else pd.Series(0, index=p.index)
        h = high[ticker] if ticker in high.columns else p
        lo_s = low[ticker] if ticker in low.columns else p
        if len(p) < 300:
            fprint(f"  SKIP {ticker} — only {len(p)} days")
            continue

        ret = p.pct_change()
        log_ret = np.log(p / p.shift(1))
        df = pd.DataFrame(index=p.index)

        # --- 18 legacy quality-momentum features ---
        for w in [5, 10, 21, 63, 126, 252]:
            df[f"ret_{w}d"] = p.pct_change(w)

        df["vol_21d"] = log_ret.rolling(21).std() * np.sqrt(252)
        df["vol_63d"] = log_ret.rolling(63).std() * np.sqrt(252)
        df["sharpe_63d"] = (log_ret.rolling(63).mean() * 252) / (df["vol_63d"] + 1e-8)

        roll_max = p.rolling(63).max()
        dd = (p - roll_max) / roll_max
        df["maxdd_63d"] = dd.rolling(63).min()

        df["pct_52w_high"] = p / p.rolling(252).max()
        df["mom_accel"] = df["ret_21d"] - df["ret_63d"] / 3

        monthly_ret = p.pct_change(21)
        df["pct_pos_months_12m"] = monthly_ret.rolling(12).apply(
            lambda x: (x > 0).sum() / len(x), raw=True
        )

        downside = log_ret.copy()
        downside[downside > 0] = 0
        downside_std = downside.rolling(63).std() * np.sqrt(252)
        df["sortino_63d"] = (log_ret.rolling(63).mean() * 252) / (downside_std + 1e-8)

        ann_ret_1y = df["ret_252d"]
        roll_max_1y = p.rolling(252).max()
        dd_1y = ((p - roll_max_1y) / roll_max_1y).rolling(252).min()
        df["calmar_1y"] = ann_ret_1y / (dd_1y.abs() + 1e-8)

        # Trend R2 and slope (63d linear regression of log price)
        log_p = np.log(p)
        x_vals = np.arange(63)
        r2_arr = np.full(len(log_p), np.nan)
        slope_arr = np.full(len(log_p), np.nan)
        log_vals = log_p.values
        for i in range(62, len(log_vals)):
            y = log_vals[i - 62:i + 1]
            if len(y) == 63 and not np.any(np.isnan(y)):
                slope, _, r_value, _, _ = sp_stats.linregress(x_vals, y)
                r2_arr[i] = r_value ** 2
                slope_arr[i] = slope
        df["trend_r2_63d"] = pd.Series(r2_arr, index=log_p.index)
        df["trend_slope_63d"] = pd.Series(slope_arr, index=log_p.index)

        # RSI 14
        delta_p = ret.copy()
        gain = delta_p.clip(lower=0).rolling(14).mean()
        loss = (-delta_p.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # --- 3 cross-asset features ---
        # sector_spy_beta_63d: rolling beta to SPY
        cov_63 = log_ret.rolling(63).cov(spy_log_ret)
        var_spy_63 = spy_log_ret.rolling(63).var()
        df["sector_spy_beta_63d"] = cov_63 / (var_spy_63 + 1e-12)

        # sector_relative_vol_21d: sector vol / SPY vol
        spy_vol_21 = spy_log_ret.rolling(21).std() * np.sqrt(252)
        df["sector_relative_vol_21d"] = df["vol_21d"] / (spy_vol_21 + 1e-8)

        # cross_sector_dispersion: already computed, align
        df["cross_sector_dispersion"] = cross_dispersion.reindex(df.index)

        features[ticker] = df

    fprint(f"  Features computed for {len(features)} sectors")
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# LGBM WALK-FORWARD RANKING
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_panel(features, close, tickers, train_dates):
    """Build LGBM training panel from feature dict."""
    rows = []
    for dt in train_dates:
        fwd_rets = {}
        feat_row = {}
        valid = True
        for tkr in tickers:
            if tkr not in features:
                valid = False
                break
            fdf = features[tkr]
            if dt not in fdf.index:
                valid = False
                break
            dt_loc = close.index.get_loc(dt)
            if dt_loc + 21 >= len(close):
                valid = False
                break
            fwd_price = close[tkr].iloc[dt_loc + 21]
            cur_price = close[tkr].iloc[dt_loc]
            if pd.isna(fwd_price) or pd.isna(cur_price) or cur_price <= 0:
                valid = False
                break
            fwd_rets[tkr] = fwd_price / cur_price - 1
            row_feats = fdf.loc[dt, FEATURE_COLS]
            if row_feats.isna().any():
                valid = False
                break
            feat_row[tkr] = row_feats.values
        if not valid or len(fwd_rets) < len(tickers):
            continue
        # Rank forward returns
        sorted_tkrs = sorted(fwd_rets.keys(), key=lambda t: fwd_rets[t])
        ranks = {t: i for i, t in enumerate(sorted_tkrs)}
        for tkr in tickers:
            row = list(feat_row[tkr]) + [ranks[tkr], dt, tkr]
            rows.append(row)
    cols = FEATURE_COLS + ["target", "date", "ticker"]
    return pd.DataFrame(rows, columns=cols)


def lgbm_rank_at_date(features, close, rebal_date, tickers):
    """Train LGBM on sliding 500d window, predict rankings for rebal_date."""
    rebal_loc = close.index.get_loc(rebal_date)
    train_start = max(0, rebal_loc - LGBM_TRAIN_DAYS - 21)
    train_end = rebal_loc - 21

    if train_end - train_start < 100:
        return None

    train_dates = close.index[train_start:train_end]
    # Sample every 5th day for speed (vs every 21st — more data but manageable)
    train_dates_sampled = train_dates[::5]

    panel = build_training_panel(features, close, tickers, train_dates_sampled)
    if len(panel) < 50:
        return None

    X_train = panel[FEATURE_COLS].values
    y_train = panel["target"].values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    callbacks = [lgb.log_evaluation(period=-1)]
    model = lgb.train(params, dtrain, num_boost_round=200, callbacks=callbacks)

    scores = {}
    for tkr in tickers:
        if tkr not in features:
            continue
        fdf = features[tkr]
        if rebal_date not in fdf.index:
            continue
        row = fdf.loc[rebal_date, FEATURE_COLS]
        if row.isna().any():
            continue
        vals = np.nan_to_num(row.values.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
        scores[tkr] = model.predict(vals)[0]

    return scores if len(scores) == len(tickers) else None


def generate_all_rankings(features, close):
    """Generate LGBM rankings for all rebalance dates (weekly)."""
    fprint("Generating walk-forward LGBM rankings...")
    # Need features warmup + train window
    start_idx = LGBM_TRAIN_DAYS + 260 + 21  # feature warmup (252) + train + gap
    all_dates = close.index

    # Weekly rebalance dates (every 5 trading days)
    weekly_dates = []
    for i in range(start_idx, len(all_dates) - 21, 5):
        weekly_dates.append(all_dates[i])

    # Monthly rebalance dates (every 21 trading days)
    monthly_dates = []
    for i in range(start_idx, len(all_dates) - 21, 21):
        monthly_dates.append(all_dates[i])

    # Union of all needed dates (weekly is superset of monthly approx)
    all_rebal = sorted(set(weekly_dates + monthly_dates))

    fprint(f"  Total rebalance dates to rank: {len(all_rebal)}")
    fprint(f"  Range: {all_rebal[0].date()} to {all_rebal[-1].date()}")

    rankings = {}  # date -> {ticker: score}
    cache_hits = 0

    for idx, dt in enumerate(all_rebal):
        if idx % 50 == 0:
            fprint(f"  Ranking {idx}/{len(all_rebal)} ({dt.date()})...")
        scores = lgbm_rank_at_date(features, close, dt, SECTOR_ETFS)
        if scores is not None:
            rankings[dt] = scores

    fprint(f"  Generated {len(rankings)} valid rankings")
    return rankings, weekly_dates, monthly_dates


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: MARKET-NEUTRAL L/S ROTATION (monthly)
# ═══════════════════════════════════════════════════════════════════════════════

def run_market_neutral(close, rankings, monthly_dates, capital):
    """Long top-3 + short bottom-3, monthly rebalance, equal weight per side."""
    fprint("Strategy 1: Market-Neutral L/S Rotation...")
    equity = capital
    monthly_returns = []
    dates_out = []

    valid_dates = [d for d in monthly_dates if d in rankings]
    for i in range(len(valid_dates) - 1):
        dt = valid_dates[i]
        next_dt = valid_dates[i + 1]
        scores = rankings[dt]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        longs = [t for t, _ in ranked[:3]]
        shorts = [t for t, _ in ranked[-3:]]

        period_ret = 0.0
        # Long side (50% capital, 3 positions)
        for tkr in longs:
            p0 = close.loc[dt, tkr]
            p1 = close.loc[next_dt, tkr]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                period_ret += (p1 / p0 - 1) / 6  # 1/6 per leg
        # Short side (50% capital, 3 positions)
        for tkr in shorts:
            p0 = close.loc[dt, tkr]
            p1 = close.loc[next_dt, tkr]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                period_ret += -(p1 / p0 - 1) / 6  # short

        equity *= (1 + period_ret)
        monthly_returns.append(period_ret)
        dates_out.append(dt)

    return pd.Series(monthly_returns, index=pd.DatetimeIndex(dates_out), name="MktNeutral")


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: EQUITY ROTATION TOP-2 (weekly)
# ═══════════════════════════════════════════════════════════════════════════════

def run_equity_rotation(close, rankings, weekly_dates, capital):
    """Buy top-2 LGBM-ranked sectors, weekly rebalance, equal weight."""
    fprint("Strategy 2: Equity Rotation Top-2...")
    equity = capital
    weekly_returns = []
    dates_out = []

    valid_dates = [d for d in weekly_dates if d in rankings]
    for i in range(len(valid_dates) - 1):
        dt = valid_dates[i]
        next_dt = valid_dates[i + 1]
        scores = rankings[dt]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top2 = [t for t, _ in ranked[:2]]

        period_ret = 0.0
        for tkr in top2:
            p0 = close.loc[dt, tkr]
            p1 = close.loc[next_dt, tkr]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                period_ret += (p1 / p0 - 1) / 2

        equity *= (1 + period_ret)
        weekly_returns.append(period_ret)
        dates_out.append(dt)

    # Aggregate to monthly for portfolio combination
    weekly_series = pd.Series(weekly_returns, index=pd.DatetimeIndex(dates_out), name="EqRotation")
    return weekly_series


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: SPY IRON CONDOR INCOME (monthly)
# ═══════════════════════════════════════════════════════════════════════════════

def run_iron_condor(spy_close, vix_close, capital):
    """Simplified SPY iron condor: sell 10-delta strangles monthly, $5 wide, BS priced with 15% haircut."""
    fprint("Strategy 3: SPY Iron Condor Income...")

    monthly_returns = []
    dates_out = []

    # Need at least 252 days for VIX rank
    start_idx = max(252, 0)
    trade_dates = spy_close.index[start_idx:]

    # Monthly trade dates (approx every 21 trading days)
    monthly_trade_dates = trade_dates[::21]

    for i in range(len(monthly_trade_dates) - 2):
        entry_date = monthly_trade_dates[i]
        # Find exit ~DTE days later
        entry_loc = spy_close.index.get_loc(entry_date)
        exit_loc = min(entry_loc + IC_DTE, len(spy_close) - 1)
        exit_date = spy_close.index[exit_loc]

        S_entry = float(spy_close.loc[entry_date])
        S_exit = float(spy_close.loc[exit_date])

        if entry_date not in vix_close.index:
            continue
        iv_entry = float(vix_close.loc[entry_date]) / 100.0
        if np.isnan(iv_entry) or iv_entry <= 0:
            continue

        T_entry = IC_DTE / 252.0

        # Find strikes at target delta
        call_short_K = strike_at_delta(S_entry, T_entry, iv_entry, IC_DELTA, opt='call')
        put_short_K = strike_at_delta(S_entry, T_entry, iv_entry, IC_DELTA, opt='put')
        call_long_K = call_short_K + IC_WIDTH
        put_long_K = put_short_K - IC_WIDTH

        # Premium received (short legs - long legs) with haircut
        prem_call_short = bs_price(S_entry, call_short_K, T_entry, iv_entry, opt='call')
        prem_call_long = bs_price(S_entry, call_long_K, T_entry, iv_entry, opt='call')
        prem_put_short = bs_price(S_entry, put_short_K, T_entry, iv_entry, opt='put')
        prem_put_long = bs_price(S_entry, put_long_K, T_entry, iv_entry, opt='put')

        credit = (prem_call_short - prem_call_long + prem_put_short - prem_put_long)
        credit *= (1.0 - IC_HAIRCUT)  # 15% haircut on BS prices

        # Max risk per IC = width - credit (per share, *100 for contract)
        max_risk_per_contract = (IC_WIDTH - credit) * 100
        if max_risk_per_contract <= 0:
            continue

        # Position size: risk max 5% of allocated capital per trade
        n_contracts = max(1, int((capital * 0.05) / max_risk_per_contract))

        # Exit: compute value at expiry
        iv_exit = float(vix_close.loc[exit_date]) / 100.0 if exit_date in vix_close.index else iv_entry
        T_exit = max(1 / 252, 0.001)  # near expiry

        cost_call_short = bs_price(S_exit, call_short_K, T_exit, iv_exit, opt='call')
        cost_call_long = bs_price(S_exit, call_long_K, T_exit, iv_exit, opt='call')
        cost_put_short = bs_price(S_exit, put_short_K, T_exit, iv_exit, opt='put')
        cost_put_long = bs_price(S_exit, put_long_K, T_exit, iv_exit, opt='put')

        close_cost = (cost_call_short - cost_call_long + cost_put_short - cost_put_long)

        pnl_per_contract = (credit - close_cost) * 100 - IC_COMMISSION
        total_pnl = pnl_per_contract * n_contracts

        period_ret = total_pnl / capital
        monthly_returns.append(period_ret)
        dates_out.append(entry_date)

    return pd.Series(monthly_returns, index=pd.DatetimeIndex(dates_out), name="IronCondor")


# ═══════════════════════════════════════════════════════════════════════════════
# RETURN ALIGNMENT + PORTFOLIO COMBINATION
# ═══════════════════════════════════════════════════════════════════════════════

def align_to_monthly(series_dict):
    """Align all strategy returns to calendar monthly frequency."""
    monthly_dict = {}
    for name, s in series_dict.items():
        if s is None or len(s) == 0:
            continue
        s_df = s.to_frame(name)
        # Resample to calendar month-end, compounding within-month returns
        monthly = (1 + s_df[name]).resample('ME').prod() - 1
        monthly_dict[name] = monthly

    # Align to common date range
    combined = pd.DataFrame(monthly_dict)
    combined = combined.dropna()
    return combined


def combine_portfolio(monthly_df, weights, name):
    """Combine strategy monthly returns with given weights into portfolio."""
    cols = list(weights.keys())
    missing = [c for c in cols if c not in monthly_df.columns]
    if missing:
        fprint(f"  WARNING: Missing strategies for {name}: {missing}")
        return None

    port_ret = sum(monthly_df[c] * w for c, w in weights.items())
    return port_ret


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(returns, name=""):
    """Compute Sharpe, Sortino, CAGR, MaxDD, Calmar, Win Rate from monthly returns."""
    if returns is None or len(returns) < 12:
        return {"name": name, "error": "insufficient data"}

    r = returns.values
    n_months = len(r)
    n_years = n_months / 12.0

    # Annualize monthly
    mean_m = np.mean(r)
    std_m = np.std(r, ddof=1)
    sharpe = (mean_m / (std_m + 1e-10)) * np.sqrt(12)

    # Sortino
    downside = r[r < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
    sortino = (mean_m / (downside_std + 1e-10)) * np.sqrt(12)

    # CAGR
    cum = np.cumprod(1 + r)
    total_ret = cum[-1]
    cagr = total_ret ** (1 / n_years) - 1 if n_years > 0 else 0

    # Max drawdown
    cum_series = pd.Series(cum)
    peak = cum_series.cummax()
    dd = (cum_series - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = cagr / (abs(max_dd) + 1e-10)

    # Win rate
    win_rate = np.mean(r > 0) * 100

    return {
        "name": name,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr_pct": round(cagr * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate_pct": round(win_rate, 1),
        "n_months": n_months,
        "n_years": round(n_years, 1),
        "total_return_pct": round((total_ret - 1) * 100, 2),
        "final_equity": round(CAPITAL * total_ret, 0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL VALIDATION (5-gate)
# ═══════════════════════════════════════════════════════════════════════════════

def adversarial_5gate(returns, spy_monthly, name=""):
    """
    5-gate adversarial validation:
      1. Permutation test (300 trials)
      2. Regime balance (bull vs bear)
      3. Sub-period (4 quarters all > 0.5 Sharpe)
      4. Outlier removal (top 5% months)
      5. Yearly consistency (60%+ years profitable)
    """
    fprint(f"  Adversarial 5-gate for {name}...")
    r = returns.values
    n = len(r)
    gates = {}

    # Gate 1: Permutation test (300 trials)
    real_sharpe = (np.mean(r) / (np.std(r, ddof=1) + 1e-10)) * np.sqrt(12)
    perm_sharpes = []
    rng = np.random.RandomState(42)
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(r)
        s = (np.mean(shuffled) / (np.std(shuffled, ddof=1) + 1e-10)) * np.sqrt(12)
        perm_sharpes.append(s)
    perm_pval = np.mean(np.array(perm_sharpes) >= real_sharpe)
    gates["permutation_pval"] = round(perm_pval, 4)
    gates["permutation_pass"] = perm_pval < 0.05

    # Gate 2: Regime balance (bull vs bear months based on SPY)
    if spy_monthly is not None and len(spy_monthly) >= len(r):
        spy_aligned = spy_monthly.reindex(returns.index)
        bull_mask = spy_aligned > 0
        bear_mask = spy_aligned <= 0

        if bull_mask.sum() > 3 and bear_mask.sum() > 3:
            r_bull = r[bull_mask.values[:n]]
            r_bear = r[bear_mask.values[:n]]
            sharpe_bull = (np.mean(r_bull) / (np.std(r_bull, ddof=1) + 1e-10)) * np.sqrt(12)
            sharpe_bear = (np.mean(r_bear) / (np.std(r_bear, ddof=1) + 1e-10)) * np.sqrt(12)
            regime_gap = abs(sharpe_bull - sharpe_bear) / (max(abs(sharpe_bull), abs(sharpe_bear)) + 1e-10)
            gates["sharpe_bull"] = round(sharpe_bull, 3)
            gates["sharpe_bear"] = round(sharpe_bear, 3)
            gates["regime_gap"] = round(regime_gap, 3)
            gates["regime_pass"] = regime_gap < 0.50
        else:
            gates["regime_pass"] = True
            gates["regime_gap"] = 0.0
    else:
        gates["regime_pass"] = True
        gates["regime_gap"] = 0.0

    # Gate 3: Sub-period (split into 4 quarters, all > 0.5 Sharpe)
    quarter_size = n // 4
    quarter_sharpes = []
    all_pass = True
    for q in range(4):
        start = q * quarter_size
        end = start + quarter_size if q < 3 else n
        rq = r[start:end]
        sq = (np.mean(rq) / (np.std(rq, ddof=1) + 1e-10)) * np.sqrt(12) if len(rq) > 3 else 0
        quarter_sharpes.append(round(sq, 3))
        if sq < 0.5:
            all_pass = False
    gates["quarter_sharpes"] = quarter_sharpes
    gates["subperiod_pass"] = all_pass

    # Gate 4: Outlier removal (remove top 5% months, re-check Sharpe > 0)
    threshold = np.percentile(r, 95)
    r_trimmed = r[r <= threshold]
    if len(r_trimmed) > 3:
        sharpe_trimmed = (np.mean(r_trimmed) / (np.std(r_trimmed, ddof=1) + 1e-10)) * np.sqrt(12)
    else:
        sharpe_trimmed = 0
    gates["sharpe_trimmed"] = round(sharpe_trimmed, 3)
    gates["outlier_pass"] = sharpe_trimmed > 0

    # Gate 5: Yearly consistency (60%+ years profitable)
    yearly_ret = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    n_years = len(yearly_ret)
    n_profitable = (yearly_ret > 0).sum()
    pct_profitable = n_profitable / n_years if n_years > 0 else 0
    gates["yearly_profitable_pct"] = round(pct_profitable * 100, 1)
    gates["yearly_pass"] = pct_profitable >= 0.60

    # Overall
    gates["all_pass"] = all(gates.get(f"{g}_pass", False) for g in
                           ["permutation", "regime", "subperiod", "outlier", "yearly"])
    gates["gates_passed"] = sum(1 for g in ["permutation", "regime", "subperiod", "outlier", "yearly"]
                                if gates.get(f"{g}_pass", False))

    return gates


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("VALIDATED PORTFOLIO COMBINER v1")
    fprint("=" * 70)

    # --- Download data ---
    close, volume, high, low, vix, vix3m = download_data()

    spy_close = close["SPY"].dropna()

    # --- Compute features ---
    features = compute_features(close, volume, high, low, spy_close)

    # --- Generate rankings ---
    rankings, weekly_dates, monthly_dates = generate_all_rankings(features, close)

    # --- Run individual strategies ---
    mn_returns = run_market_neutral(close, rankings, monthly_dates, CAPITAL)
    er_returns = run_equity_rotation(close, rankings, weekly_dates, CAPITAL)
    ic_returns = run_iron_condor(spy_close, vix, CAPITAL)

    fprint(f"\nIndividual strategy stats:")
    fprint(f"  Market-Neutral: {len(mn_returns)} periods")
    fprint(f"  Equity Rotation: {len(er_returns)} periods")
    fprint(f"  Iron Condor: {len(ic_returns)} periods")

    # --- Align to monthly ---
    strategy_returns = {
        "MktNeutral": mn_returns,
        "EqRotation": er_returns,
        "IronCondor": ic_returns,
    }
    monthly_df = align_to_monthly(strategy_returns)
    fprint(f"\nAligned monthly data: {len(monthly_df)} months, "
           f"{monthly_df.index[0].date()} to {monthly_df.index[-1].date()}")

    # --- SPY monthly for regime analysis ---
    spy_monthly = spy_close.pct_change().resample('ME').apply(lambda x: (1 + x).prod() - 1)

    # --- Individual strategy metrics ---
    fprint("\n" + "=" * 70)
    fprint("INDIVIDUAL STRATEGY METRICS (monthly)")
    fprint("=" * 70)
    individual_metrics = {}
    for sname in ["MktNeutral", "EqRotation", "IronCondor"]:
        if sname in monthly_df.columns:
            m = compute_metrics(monthly_df[sname], sname)
            individual_metrics[sname] = m
            fprint(f"\n  {sname}:")
            for k, v in m.items():
                if k != "name":
                    fprint(f"    {k}: {v}")

    # --- Correlation matrix ---
    fprint("\n" + "=" * 70)
    fprint("STRATEGY RETURN CORRELATIONS")
    fprint("=" * 70)
    corr = monthly_df.corr()
    fprint(f"\n{corr.round(3).to_string()}")
    corr_dict = {f"{r}_{c}": round(corr.loc[r, c], 4)
                 for r in corr.index for c in corr.columns if r < c}

    # --- Portfolio combinations ---
    VARIANTS = {
        "A_equal": {"MktNeutral": 1/3, "EqRotation": 1/3, "IronCondor": 1/3},
        "D_growth": {"MktNeutral": 0.30, "EqRotation": 0.50, "IronCondor": 0.20},
        "C_best_two": {"MktNeutral": 0.50, "EqRotation": 0.50},
    }

    # Variant B: Risk parity (inverse vol)
    vols = {}
    for sname in ["MktNeutral", "EqRotation", "IronCondor"]:
        if sname in monthly_df.columns:
            v = monthly_df[sname].std() * np.sqrt(12)
            vols[sname] = v if v > 0 else 1e-6
    inv_vols = {k: 1.0 / v for k, v in vols.items()}
    total_inv = sum(inv_vols.values())
    VARIANTS["B_risk_parity"] = {k: v / total_inv for k, v in inv_vols.items()}

    fprint("\n" + "=" * 70)
    fprint("PORTFOLIO COMBINATION RESULTS")
    fprint("=" * 70)

    all_results = {}
    all_adversarial = {}

    for vname, weights in VARIANTS.items():
        fprint(f"\n--- Variant {vname} ---")
        fprint(f"  Weights: {', '.join(f'{k}={v:.1%}' for k, v in weights.items())}")

        port_ret = combine_portfolio(monthly_df, weights, vname)
        if port_ret is None:
            continue

        metrics = compute_metrics(port_ret, vname)
        all_results[vname] = metrics

        fprint(f"  Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
               f"CAGR={metrics['cagr_pct']}%, MDD={metrics['max_dd_pct']}%, "
               f"Calmar={metrics['calmar']}, WR={metrics['win_rate_pct']}%")
        fprint(f"  Final equity: ${metrics['final_equity']:,.0f}")

        # Adversarial validation
        gates = adversarial_5gate(port_ret, spy_monthly, vname)
        all_adversarial[vname] = gates
        fprint(f"  Adversarial: {gates['gates_passed']}/5 gates passed | "
               f"{'PASS' if gates['all_pass'] else 'FAIL'}")
        if not gates['all_pass']:
            failed = []
            for g in ["permutation", "regime", "subperiod", "outlier", "yearly"]:
                if not gates.get(f"{g}_pass", False):
                    failed.append(g)
            fprint(f"  Failed gates: {', '.join(failed)}")

    # --- Diversification benefit ---
    fprint("\n" + "=" * 70)
    fprint("DIVERSIFICATION BENEFIT ANALYSIS")
    fprint("=" * 70)
    for vname in all_results:
        weights = VARIANTS[vname]
        # Weighted avg of individual Sharpes
        weighted_sharpe = sum(
            individual_metrics.get(s, {}).get("sharpe", 0) * w
            for s, w in weights.items()
        )
        combo_sharpe = all_results[vname]["sharpe"]
        benefit = combo_sharpe - weighted_sharpe
        fprint(f"  {vname}: weighted avg Sharpe={weighted_sharpe:.3f}, "
               f"combo Sharpe={combo_sharpe:.3f}, benefit={benefit:+.3f}")

    # --- Summary table sorted by Sharpe ---
    fprint("\n" + "=" * 70)
    fprint("RESULTS SORTED BY SHARPE")
    fprint("=" * 70)
    sorted_results = sorted(all_results.values(), key=lambda x: x.get("sharpe", 0), reverse=True)
    fprint(f"\n{'Variant':<18} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MDD%':>7} "
           f"{'Calmar':>7} {'WR%':>6} {'Final$':>12} {'Gates':>6}")
    fprint("-" * 85)
    for r in sorted_results:
        gates_str = f"{all_adversarial[r['name']]['gates_passed']}/5"
        fprint(f"{r['name']:<18} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
               f"{r['cagr_pct']:>6.1f}% {r['max_dd_pct']:>6.1f}% "
               f"{r['calmar']:>7.3f} {r['win_rate_pct']:>5.1f}% "
               f"${r['final_equity']:>10,.0f} {gates_str:>6}")

    # --- Save JSON ---
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "start_date": START_DATE,
            "capital": CAPITAL,
            "lgbm_train_days": LGBM_TRAIN_DAYS,
            "n_features": len(FEATURE_COLS),
            "n_permutations": N_PERMUTATIONS,
            "ic_delta": IC_DELTA,
            "ic_width": IC_WIDTH,
            "ic_dte": IC_DTE,
            "ic_haircut_pct": IC_HAIRCUT * 100,
            "ic_commission": IC_COMMISSION,
        },
        "individual_metrics": individual_metrics,
        "correlations": corr_dict,
        "portfolio_variants": all_results,
        "adversarial": all_adversarial,
        "variant_weights": {k: {s: round(w, 4) for s, w in v.items()} for k, v in VARIANTS.items()},
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # --- MLflow logging ---
    if MLFLOW_OK:
        try:
            with mlflow.start_run(run_name="portfolio_combiner_v1"):
                # Log config
                mlflow.log_params({
                    "start_date": START_DATE,
                    "capital": CAPITAL,
                    "lgbm_train_days": LGBM_TRAIN_DAYS,
                    "n_features": len(FEATURE_COLS),
                    "n_permutations": N_PERMUTATIONS,
                    "ic_delta": IC_DELTA,
                    "ic_width": IC_WIDTH,
                    "ic_dte": IC_DTE,
                })

                # Log best variant metrics
                if sorted_results:
                    best = sorted_results[0]
                    mlflow.log_metrics({
                        "best_sharpe": best["sharpe"],
                        "best_sortino": best["sortino"],
                        "best_cagr_pct": best["cagr_pct"],
                        "best_max_dd_pct": best["max_dd_pct"],
                        "best_calmar": best["calmar"],
                        "best_win_rate_pct": best["win_rate_pct"],
                    })
                    mlflow.log_param("best_variant", best["name"])

                # Log all variant metrics
                for vname, m in all_results.items():
                    mlflow.log_metrics({
                        f"{vname}_sharpe": m["sharpe"],
                        f"{vname}_sortino": m["sortino"],
                        f"{vname}_cagr_pct": m["cagr_pct"],
                        f"{vname}_max_dd_pct": m["max_dd_pct"],
                    })

                # Log individual strategy metrics
                for sname, m in individual_metrics.items():
                    mlflow.log_metrics({
                        f"ind_{sname}_sharpe": m.get("sharpe", 0),
                        f"ind_{sname}_cagr_pct": m.get("cagr_pct", 0),
                    })

                # Log correlations
                mlflow.log_metrics(corr_dict)

                # Log adversarial results
                for vname, gates in all_adversarial.items():
                    mlflow.log_metric(f"{vname}_gates_passed", gates["gates_passed"])

                # Log JSON artifact
                mlflow.log_artifact(str(RESULTS_PATH))

                fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging error: {e}")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    fprint("DONE.")


if __name__ == "__main__":
    main()
