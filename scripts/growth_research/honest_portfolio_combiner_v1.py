#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""
Honest Portfolio Combiner v1
============================
Combines ONLY honestly-validated strategies (no inflated options pricing)
into a unified portfolio for a $645 starting account.

Strategies:
  1. Equity Rotation Top-2: Long top-2 LGBM-ranked sector ETFs monthly
  2. Market-Neutral L/S Rotation: Long top-3, short bottom-3 monthly
  3. Momentum Burst: Single-leg momentum plays on 14 ETFs

Allocation variants A-F tested with walk-forward validation.
Full adversarial: regime-stratified Sharpe, permutation tests, R1 gap check.

Capital: $645
Output: output/growth_research/honest_portfolio_v1/
MLflow: honest_portfolio_combiner_v1
"""

import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings("ignore")

# ── Unbuffered print ─────────────────────────────────────────────────────────
_print = print
def fprint(*a, **kw):
    kw['flush'] = True
    _print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, **kw)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "honest_portfolio_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fprint(f"Output dir: {OUTPUT_DIR}")

# ── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen("http://jupiter:5000/", timeout=3)
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
    fprint("MLflow connected")
except Exception as e:
    fprint(f"MLflow unavailable ({e})")

# ── Constants ────────────────────────────────────────────────────────────────
START_DATE = "2018-01-01"
CAPITAL = 645.0

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
BROAD_ETFS = ["SPY", "QQQ", "IWM"]
ALL_TICKERS = SECTOR_ETFS + BROAD_ETFS
VIX_TICKER = "^VIX"

LGBM_TRAIN_DAYS = 500  # sliding window
N_PERMUTATIONS = 100

# ── Data Download ────────────────────────────────────────────────────────────
def download_data():
    """Download all required price data from yfinance."""
    fprint("Downloading price data...")
    tickers = ALL_TICKERS + [VIX_TICKER]
    data = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        volume = data["Volume"] if "Volume" in data.columns.get_level_values(0) else None
    else:
        close = data[["Close"]].copy()
        volume = None

    # Ensure column names are clean strings
    close.columns = [str(c).strip() for c in close.columns]
    close = close.ffill()

    # Separate VIX
    vix = close["^VIX"].copy() if "^VIX" in close.columns else None
    price = close[[c for c in close.columns if c != "^VIX"]].copy()

    fprint(f"Data: {len(price)} days, {price.shape[1]} tickers, "
           f"{price.index[0].date()} to {price.index[-1].date()}")
    return price, vix


# ── Feature Engineering for LGBM ─────────────────────────────────────────────
def compute_features(price_df, tickers):
    """
    Compute 17 features for each ticker at each date.
    Returns dict: ticker -> DataFrame of features (index=date).
    """
    features = {}
    for tkr in tickers:
        if tkr not in price_df.columns:
            continue
        p = price_df[tkr].dropna()
        if len(p) < 260:
            continue

        df = pd.DataFrame(index=p.index)

        # Return features
        for w in [5, 10, 21, 63, 126, 252]:
            df[f"ret_{w}d"] = p.pct_change(w)

        # Volatility
        log_ret = np.log(p / p.shift(1))
        df["vol_21d"] = log_ret.rolling(21).std() * np.sqrt(252)
        df["vol_63d"] = log_ret.rolling(63).std() * np.sqrt(252)

        # Sharpe 63d
        df["sharpe_63d"] = (log_ret.rolling(63).mean() * 252) / (df["vol_63d"] + 1e-8)

        # Max drawdown 63d
        roll_max = p.rolling(63).max()
        dd = (p - roll_max) / roll_max
        df["maxdd_63d"] = dd.rolling(63).min()

        # Pct of 52-week high
        df["pct_52w_high"] = p / p.rolling(252).max()

        # Momentum acceleration (21d ret - 63d ret normalized)
        df["mom_accel"] = df["ret_21d"] - df["ret_63d"] / 3

        # Pct positive months in last 12m (approx using 21d blocks)
        monthly_ret = p.pct_change(21)
        df["pct_pos_months_12m"] = monthly_ret.rolling(12).apply(
            lambda x: (x > 0).sum() / len(x), raw=True
        )

        # Sortino 63d
        downside = log_ret.copy()
        downside[downside > 0] = 0
        downside_std = downside.rolling(63).std() * np.sqrt(252)
        df["sortino_63d"] = (log_ret.rolling(63).mean() * 252) / (downside_std + 1e-8)

        # Calmar 1y
        ann_ret_1y = df["ret_252d"]
        roll_max_1y = p.rolling(252).max()
        dd_1y = ((p - roll_max_1y) / roll_max_1y).rolling(252).min()
        df["calmar_1y"] = ann_ret_1y / (dd_1y.abs() + 1e-8)

        # Trend R2 and slope (63d linear regression of log price)
        log_p = np.log(p)
        def _trend_stats(window):
            r2s = []
            slopes = []
            x = np.arange(63)
            for i in range(len(log_p)):
                if i < 62:
                    r2s.append(np.nan)
                    slopes.append(np.nan)
                else:
                    y = log_p.iloc[i-62:i+1].values
                    if len(y) == 63 and not np.any(np.isnan(y)):
                        slope, intercept, r_value, _, _ = stats.linregress(x, y)
                        r2s.append(r_value ** 2)
                        slopes.append(slope)
                    else:
                        r2s.append(np.nan)
                        slopes.append(np.nan)
            return pd.Series(r2s, index=log_p.index), pd.Series(slopes, index=log_p.index)

        df["trend_r2_63d"], df["trend_slope_63d"] = _trend_stats(63)

        features[tkr] = df

    return features


def build_lgbm_panel(features, price_df, tickers, date_idx):
    """
    Build training panel for a given date range.
    Target: rank of 21-day forward return across sectors.
    """
    rows = []
    feature_cols = [
        "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
        "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high",
        "mom_accel", "pct_pos_months_12m", "sortino_63d", "calmar_1y",
        "trend_r2_63d", "trend_slope_63d"
    ]

    for dt in date_idx:
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

            # Forward 21-day return
            dt_loc = price_df.index.get_loc(dt)
            if dt_loc + 21 >= len(price_df):
                valid = False
                break
            fwd_price = price_df[tkr].iloc[dt_loc + 21]
            cur_price = price_df[tkr].iloc[dt_loc]
            if pd.isna(fwd_price) or pd.isna(cur_price) or cur_price <= 0:
                valid = False
                break
            fwd_rets[tkr] = fwd_price / cur_price - 1

            row_feats = fdf.loc[dt, feature_cols]
            if row_feats.isna().any():
                valid = False
                break
            feat_row[tkr] = row_feats.values

        if not valid or len(fwd_rets) < len(tickers):
            continue

        # Rank forward returns (0 = worst, N-1 = best)
        sorted_tkrs = sorted(fwd_rets.keys(), key=lambda t: fwd_rets[t])
        ranks = {t: i for i, t in enumerate(sorted_tkrs)}

        for tkr in tickers:
            row = list(feat_row[tkr]) + [ranks[tkr], dt, tkr]
            rows.append(row)

    cols = feature_cols + ["target", "date", "ticker"]
    return pd.DataFrame(rows, columns=cols)


# ── LGBM Ranking ─────────────────────────────────────────────────────────────
def lgbm_rank_sectors(features, price_df, rebal_date, tickers):
    """
    Train LGBM on sliding 500-day window, predict rankings for rebal_date.
    Returns dict: ticker -> predicted_score (higher = better).
    """
    feature_cols = [
        "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
        "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high",
        "mom_accel", "pct_pos_months_12m", "sortino_63d", "calmar_1y",
        "trend_r2_63d", "trend_slope_63d"
    ]

    # Get training dates: LGBM_TRAIN_DAYS before rebal_date
    rebal_loc = price_df.index.get_loc(rebal_date)
    train_start = max(0, rebal_loc - LGBM_TRAIN_DAYS - 21)
    train_end = rebal_loc - 21  # leave gap for forward returns

    if train_end - train_start < 100:
        return None

    # Sample monthly dates from training window for efficiency
    train_dates = price_df.index[train_start:train_end]
    # Use every 21st day (roughly monthly) to keep training fast
    train_dates_sampled = train_dates[::21]

    panel = build_lgbm_panel(features, price_df, tickers, train_dates_sampled)
    if len(panel) < 50:
        return None

    X_train = panel[feature_cols].values
    y_train = panel["target"].values

    # Train LightGBM regressor
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
    model = lgb.train(params, dtrain, num_boost_round=100)

    # Predict for rebal_date
    scores = {}
    for tkr in tickers:
        if tkr not in features:
            continue
        fdf = features[tkr]
        if rebal_date not in fdf.index:
            continue
        row = fdf.loc[rebal_date, feature_cols]
        if row.isna().any():
            continue
        pred = model.predict(row.values.reshape(1, -1))[0]
        scores[tkr] = pred

    return scores if len(scores) == len(tickers) else None


# ── Strategy Implementations ─────────────────────────────────────────────────
def run_equity_rotation(price_df, features, rebal_dates, capital):
    """
    Strategy 1: Equity Rotation Top-2
    Long top-2 LGBM-ranked sector ETFs, monthly rebalance, equal weight.
    """
    fprint("Running Equity Rotation Top-2...")
    equity = capital
    monthly_returns = []
    monthly_dates = []

    for i in range(len(rebal_dates) - 1):
        dt = rebal_dates[i]
        next_dt = rebal_dates[i + 1]

        scores = lgbm_rank_sectors(features, price_df, dt, SECTOR_ETFS)
        if scores is None:
            monthly_returns.append(0.0)
            monthly_dates.append(dt)
            continue

        # Top 2 sectors
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top2 = [t for t, _ in ranked[:2]]

        # Calculate return over holding period
        period_ret = 0.0
        for tkr in top2:
            p_start = price_df.loc[dt, tkr]
            p_end = price_df.loc[next_dt, tkr]
            if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                period_ret += (p_end / p_start - 1) / 2  # equal weight

        equity *= (1 + period_ret)
        monthly_returns.append(period_ret)
        monthly_dates.append(dt)

    return np.array(monthly_returns), monthly_dates, equity


def run_market_neutral(price_df, features, rebal_dates, capital):
    """
    Strategy 2: Market-Neutral Long/Short Rotation
    Long top-3, short bottom-3, monthly rebalance, equal weight.
    """
    fprint("Running Market-Neutral L/S Rotation...")
    equity = capital
    monthly_returns = []
    monthly_dates = []

    for i in range(len(rebal_dates) - 1):
        dt = rebal_dates[i]
        next_dt = rebal_dates[i + 1]

        scores = lgbm_rank_sectors(features, price_df, dt, SECTOR_ETFS)
        if scores is None:
            monthly_returns.append(0.0)
            monthly_dates.append(dt)
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        longs = [t for t, _ in ranked[:3]]
        shorts = [t for t, _ in ranked[-3:]]

        period_ret = 0.0
        for tkr in longs:
            p_start = price_df.loc[dt, tkr]
            p_end = price_df.loc[next_dt, tkr]
            if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                period_ret += (p_end / p_start - 1) / 6  # 1/6 each leg

        for tkr in shorts:
            p_start = price_df.loc[dt, tkr]
            p_end = price_df.loc[next_dt, tkr]
            if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                period_ret += -(p_end / p_start - 1) / 6  # short side

        equity *= (1 + period_ret)
        monthly_returns.append(period_ret)
        monthly_dates.append(dt)

    return np.array(monthly_returns), monthly_dates, equity


def run_momentum_burst(price_df, rebal_dates_daily, capital):
    """
    Strategy 3: Momentum Burst (simplified, underlying price moves)
    Signal: RSI(14) < 30 AND price > SMA(20) AND ret_5d > 3%
    Exit: +10% TP or -8% SL or 5 trading days hold max
    Max 1 position at a time. Use underlying price (not options).
    """
    fprint("Running Momentum Burst...")

    tickers = ALL_TICKERS  # 14 ETFs

    # Precompute RSI(14) and SMA(20) for all tickers
    rsi_dict = {}
    sma_dict = {}
    for tkr in tickers:
        if tkr not in price_df.columns:
            continue
        p = price_df[tkr].dropna()
        # RSI
        delta = p.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_dict[tkr] = 100 - (100 / (1 + rs))
        sma_dict[tkr] = p.rolling(20).mean()

    # Walk through daily, tracking trades
    equity = capital
    trades = []
    in_trade = False
    entry_price = 0
    entry_date = None
    entry_ticker = None
    days_held = 0

    # Use the daily price index
    dates = price_df.index
    # Start after enough warmup
    start_idx = max(252, 30)  # need RSI + SMA warmup

    # Monthly return tracking: align to rebal_dates
    # We'll track daily equity and then resample to monthly
    daily_equity = pd.Series(index=dates[start_idx:], dtype=float)
    eq = capital

    for i in range(start_idx, len(dates)):
        dt = dates[i]

        if in_trade:
            cur_price = price_df.loc[dt, entry_ticker]
            if pd.isna(cur_price):
                daily_equity.loc[dt] = eq
                continue

            pct_change = cur_price / entry_price - 1
            days_held += 1

            # Check exit conditions
            exit_trade = False
            if pct_change >= 0.10:  # TP +10%
                exit_trade = True
            elif pct_change <= -0.08:  # SL -8%
                exit_trade = True
            elif days_held >= 5:  # Max hold
                exit_trade = True

            if exit_trade:
                trade_ret = pct_change
                eq *= (1 + trade_ret)
                trades.append({
                    "entry": entry_date.strftime("%Y-%m-%d"),
                    "exit": dt.strftime("%Y-%m-%d"),
                    "ticker": entry_ticker,
                    "ret": trade_ret,
                    "days": days_held
                })
                in_trade = False

        if not in_trade:
            # Scan for entry signal
            for tkr in tickers:
                if tkr not in rsi_dict or tkr not in sma_dict:
                    continue
                if dt not in rsi_dict[tkr].index or dt not in sma_dict[tkr].index:
                    continue
                if tkr not in price_df.columns:
                    continue

                rsi_val = rsi_dict[tkr].get(dt, np.nan)
                sma_val = sma_dict[tkr].get(dt, np.nan)
                cur_p = price_df.loc[dt, tkr]

                if pd.isna(rsi_val) or pd.isna(sma_val) or pd.isna(cur_p):
                    continue

                # 5d return
                if i >= 5:
                    p5 = price_df[tkr].iloc[i - 5]
                    if pd.notna(p5) and p5 > 0:
                        ret5 = cur_p / p5 - 1
                    else:
                        continue
                else:
                    continue

                # Signal: RSI < 30, price > SMA20, 5d ret > 3%
                if rsi_val < 30 and cur_p > sma_val and ret5 > 0.03:
                    in_trade = True
                    entry_price = cur_p
                    entry_date = dt
                    entry_ticker = tkr
                    days_held = 0
                    break

        daily_equity.loc[dt] = eq

    # Convert to monthly returns aligned with rebal_dates
    monthly_returns = []
    monthly_dates = []
    for i in range(len(rebal_dates_daily) - 1):
        d1 = rebal_dates_daily[i]
        d2 = rebal_dates_daily[i + 1]
        if d1 in daily_equity.index and d2 in daily_equity.index:
            e1 = daily_equity.loc[d1]
            e2 = daily_equity.loc[d2]
            if pd.notna(e1) and pd.notna(e2) and e1 > 0:
                monthly_returns.append(e2 / e1 - 1)
            else:
                monthly_returns.append(0.0)
        else:
            monthly_returns.append(0.0)
        monthly_dates.append(d1)

    fprint(f"  Momentum Burst: {len(trades)} trades, final equity ${eq:.2f}")
    return np.array(monthly_returns), monthly_dates, eq


# ── Performance Metrics ──────────────────────────────────────────────────────
def compute_metrics(monthly_rets, rf_annual=0.05):
    """Compute all required performance metrics from monthly returns."""
    rets = np.array(monthly_rets)
    n = len(rets)
    if n < 2:
        return {}

    rf_monthly = (1 + rf_annual) ** (1/12) - 1
    excess = rets - rf_monthly

    # Sharpe (annualized)
    sharpe = np.mean(excess) / (np.std(excess, ddof=1) + 1e-10) * np.sqrt(12)

    # Sortino
    downside = excess[excess < 0]
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-10
    sortino = np.mean(excess) / (downside_std + 1e-10) * np.sqrt(12)

    # Profit Factor
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / (losses + 1e-10)

    # Win Rate
    wr = (rets > 0).sum() / n if n > 0 else 0

    # CAGR
    cum = np.prod(1 + rets)
    years = n / 12
    cagr = cum ** (1 / max(years, 0.1)) - 1 if cum > 0 else -1.0

    # Max Drawdown
    cum_series = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum_series)
    dd = (cum_series - peak) / peak
    maxdd = dd.min()

    # Calmar
    calmar = cagr / (abs(maxdd) + 1e-10)

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 3),
        "cagr": round(cagr, 4),
        "max_dd": round(maxdd, 4),
        "calmar": round(calmar, 3),
        "n_months": n,
        "total_return": round(cum - 1, 4),
    }


def regime_sharpes(monthly_rets, monthly_dates, spy_prices):
    """
    Compute Sharpe by bull/bear/flat regime.
    Bull: SPY > 200d SMA, Bear: SPY < 200d SMA * 0.95, Flat: in between.
    """
    sma200 = spy_prices.rolling(200).mean()

    bull_rets, bear_rets, flat_rets = [], [], []

    for ret, dt in zip(monthly_rets, monthly_dates):
        if dt not in spy_prices.index or dt not in sma200.index:
            flat_rets.append(ret)
            continue
        spy_p = spy_prices.loc[dt]
        sma_v = sma200.loc[dt]
        if pd.isna(spy_p) or pd.isna(sma_v):
            flat_rets.append(ret)
            continue

        if spy_p > sma_v:
            bull_rets.append(ret)
        elif spy_p < sma_v * 0.95:
            bear_rets.append(ret)
        else:
            flat_rets.append(ret)

    def _sharpe(r):
        r = np.array(r)
        if len(r) < 3:
            return np.nan
        rf_m = (1.05) ** (1/12) - 1
        excess = r - rf_m
        return np.mean(excess) / (np.std(excess, ddof=1) + 1e-10) * np.sqrt(12)

    s_bull = _sharpe(bull_rets)
    s_bear = _sharpe(bear_rets)
    s_flat = _sharpe(flat_rets)

    # R1 regime gap check
    if not np.isnan(s_bull) and not np.isnan(s_bear):
        gap = abs(s_bull - s_bear) / (max(abs(s_bull), abs(s_bear)) + 1e-10)
        r1_pass = gap < 0.50
    else:
        gap = np.nan
        r1_pass = False

    return {
        "sharpe_bull": round(s_bull, 3) if not np.isnan(s_bull) else None,
        "sharpe_bear": round(s_bear, 3) if not np.isnan(s_bear) else None,
        "sharpe_flat": round(s_flat, 3) if not np.isnan(s_flat) else None,
        "n_bull": len(bull_rets),
        "n_bear": len(bear_rets),
        "n_flat": len(flat_rets),
        "regime_gap": round(gap, 3) if not np.isnan(gap) else None,
        "r1_pass": r1_pass,
    }


def permutation_test(monthly_rets, n_perms=100):
    """
    Shuffle monthly returns 100 times, compute fraction of shuffled Sharpe >= real Sharpe.
    """
    real_sharpe = compute_metrics(monthly_rets).get("sharpe", 0)
    count_ge = 0
    rets = np.array(monthly_rets)

    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        shuffled = rng.permutation(rets)
        shuf_sharpe = compute_metrics(shuffled).get("sharpe", 0)
        if shuf_sharpe >= real_sharpe:
            count_ge += 1

    p_value = count_ge / n_perms
    return round(p_value, 3)


# ── Portfolio Combination ────────────────────────────────────────────────────
def combine_strategies(rets_dict, weights, common_dates):
    """
    Combine strategy returns with given weights.
    rets_dict: {name: {date: ret}}
    weights: {name: weight}
    """
    combined = []
    dates_out = []

    for dt in common_dates:
        port_ret = 0.0
        valid = True
        for name, w in weights.items():
            if dt in rets_dict[name]:
                port_ret += w * rets_dict[name][dt]
            else:
                valid = False
                break
        if valid:
            combined.append(port_ret)
            dates_out.append(dt)

    return np.array(combined), dates_out


def get_rebalance_dates(price_index):
    """Get first trading day of each month."""
    dates = price_index.to_series()
    monthly = dates.groupby([dates.index.year, dates.index.month]).first()
    return list(monthly.values)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("HONEST PORTFOLIO COMBINER v1")
    fprint(f"Capital: ${CAPITAL:.2f} | Start: {START_DATE}")
    fprint("=" * 70)

    # ── Download Data ────────────────────────────────────────────────────
    price_df, vix = download_data()
    spy_prices = price_df["SPY"] if "SPY" in price_df.columns else None

    # ── Rebalance Dates ──────────────────────────────────────────────────
    rebal_dates = get_rebalance_dates(price_df.index)
    fprint(f"Rebalance dates: {len(rebal_dates)} months")

    # ── Compute Features ─────────────────────────────────────────────────
    fprint("Computing features for LGBM...")
    features = compute_features(price_df, SECTOR_ETFS)
    fprint(f"Features computed for {len(features)} tickers")

    # ── Run Individual Strategies ────────────────────────────────────────
    fprint("\n" + "=" * 70)
    fprint("RUNNING INDIVIDUAL STRATEGIES")
    fprint("=" * 70)

    # Strategy 1: Equity Rotation
    eq_rot_rets, eq_rot_dates, eq_rot_final = run_equity_rotation(
        price_df, features, rebal_dates, CAPITAL
    )
    fprint(f"  Equity Rotation: {len(eq_rot_rets)} months, "
           f"final ${eq_rot_final:.2f}")

    # Strategy 2: Market Neutral
    mn_rets, mn_dates, mn_final = run_market_neutral(
        price_df, features, rebal_dates, CAPITAL
    )
    fprint(f"  Market Neutral: {len(mn_rets)} months, "
           f"final ${mn_final:.2f}")

    # Strategy 3: Momentum Burst
    mb_rets, mb_dates, mb_final = run_momentum_burst(
        price_df, rebal_dates, CAPITAL
    )
    fprint(f"  Momentum Burst: {len(mb_rets)} months, "
           f"final ${mb_final:.2f}")

    # ── Build return dicts for combination ───────────────────────────────
    eq_rot_ret_dict = dict(zip(eq_rot_dates, eq_rot_rets))
    mn_ret_dict = dict(zip(mn_dates, mn_rets))
    mb_ret_dict = dict(zip(mb_dates, mb_rets))

    rets_all = {
        "equity_rotation": eq_rot_ret_dict,
        "market_neutral": mn_ret_dict,
        "momentum_burst": mb_ret_dict,
    }

    # Common dates across strategies
    common_all = sorted(set(eq_rot_dates) & set(mn_dates) & set(mb_dates))
    common_eq_mn = sorted(set(eq_rot_dates) & set(mn_dates))
    common_mn_mb = sorted(set(mn_dates) & set(mb_dates))

    fprint(f"\nCommon dates: all3={len(common_all)}, eq+mn={len(common_eq_mn)}, "
           f"mn+mb={len(common_mn_mb)}")

    # ── Compute inverse-vol weights for Risk Parity ──────────────────────
    eq_vol = np.std(eq_rot_rets) if len(eq_rot_rets) > 1 else 1.0
    mn_vol = np.std(mn_rets) if len(mn_rets) > 1 else 1.0
    mb_vol = np.std(mb_rets) if len(mb_rets) > 1 else 1.0

    inv_vol = np.array([1/(eq_vol+1e-10), 1/(mn_vol+1e-10), 1/(mb_vol+1e-10)])
    inv_vol_w = inv_vol / inv_vol.sum()
    fprint(f"Risk Parity weights: EqRot={inv_vol_w[0]:.3f}, MN={inv_vol_w[1]:.3f}, "
           f"MomBurst={inv_vol_w[2]:.3f}")

    # ── Define Allocation Variants ───────────────────────────────────────
    variants = {
        "A) Equity Rotation Only": {
            "weights": {"equity_rotation": 1.0},
            "dates": eq_rot_dates,
            "rets": eq_rot_rets,
        },
        "B) Market-Neutral Only": {
            "weights": {"market_neutral": 1.0},
            "dates": mn_dates,
            "rets": mn_rets,
        },
        "C) 50/50 EqRot + MN": {
            "weights": {"equity_rotation": 0.5, "market_neutral": 0.5},
            "dates": common_eq_mn,
            "rets": None,
        },
        "D) Risk Parity (inv-vol)": {
            "weights": {
                "equity_rotation": inv_vol_w[0],
                "market_neutral": inv_vol_w[1],
                "momentum_burst": inv_vol_w[2],
            },
            "dates": common_all,
            "rets": None,
        },
        "E) 70% MN + 30% MomBurst": {
            "weights": {"market_neutral": 0.7, "momentum_burst": 0.3},
            "dates": common_mn_mb,
            "rets": None,
        },
        "F) Equal-weight All Three": {
            "weights": {
                "equity_rotation": 1/3,
                "market_neutral": 1/3,
                "momentum_burst": 1/3,
            },
            "dates": common_all,
            "rets": None,
        },
    }

    # Compute combined returns for multi-strategy variants
    for name, v in variants.items():
        if v["rets"] is None:
            combined_rets, combined_dates = combine_strategies(
                rets_all, v["weights"], v["dates"]
            )
            v["rets"] = combined_rets
            v["dates"] = combined_dates

    # ── Evaluate All Variants ────────────────────────────────────────────
    fprint("\n" + "=" * 70)
    fprint("PORTFOLIO EVALUATION")
    fprint("=" * 70)

    results = {}
    for name, v in variants.items():
        fprint(f"\n--- {name} ---")
        rets = v["rets"]
        dates = v["dates"]

        # Core metrics
        metrics = compute_metrics(rets)
        fprint(f"  Sharpe={metrics.get('sharpe','N/A')}, "
               f"Sortino={metrics.get('sortino','N/A')}, "
               f"CAGR={metrics.get('cagr','N/A')}, "
               f"MaxDD={metrics.get('max_dd','N/A')}")

        # Final equity
        final_eq = CAPITAL * np.prod(1 + rets)
        metrics["final_equity"] = round(final_eq, 2)
        fprint(f"  Final equity: ${final_eq:.2f} from ${CAPITAL:.2f}")

        # Regime analysis
        if spy_prices is not None:
            regime = regime_sharpes(rets, dates, spy_prices)
            metrics["regime"] = regime
            fprint(f"  Regime: Bull={regime['sharpe_bull']}, "
                   f"Bear={regime['sharpe_bear']}, "
                   f"Flat={regime['sharpe_flat']} | "
                   f"R1 pass={regime['r1_pass']} (gap={regime['regime_gap']})")
        else:
            metrics["regime"] = None

        # Permutation test
        fprint(f"  Running {N_PERMUTATIONS}-shuffle permutation test...")
        p_val = permutation_test(rets, N_PERMUTATIONS)
        metrics["perm_p_value"] = p_val
        fprint(f"  Permutation p-value: {p_val}")

        # Weights
        metrics["weights"] = {k: round(w, 4) for k, w in v["weights"].items()}

        results[name] = metrics

    # ── Correlation Matrix ───────────────────────────────────────────────
    fprint("\n" + "=" * 70)
    fprint("STRATEGY CORRELATION (monthly returns)")
    fprint("=" * 70)

    # Align returns to common dates
    corr_data = {}
    for name, ret_dict in rets_all.items():
        corr_data[name] = pd.Series(ret_dict)

    corr_df = pd.DataFrame(corr_data).dropna()
    if len(corr_df) > 3:
        corr_matrix = corr_df.corr()
        fprint(f"\n{corr_matrix.round(3).to_string()}")
        corr_result = corr_matrix.to_dict()
    else:
        corr_result = {}
        fprint("  Not enough common data for correlation")

    # ── Summary Table ────────────────────────────────────────────────────
    fprint("\n" + "=" * 70)
    fprint("SORTED RESULTS (by Sharpe ratio)")
    fprint("=" * 70)

    sorted_variants = sorted(results.items(),
                              key=lambda x: x[1].get("sharpe", 0),
                              reverse=True)

    header = f"{'Variant':<32} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} " \
             f"{'WR':>5} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7} " \
             f"{'Final$':>8} {'p-val':>6} {'R1':>4}"
    fprint(header)
    fprint("-" * len(header))

    for name, m in sorted_variants:
        regime = m.get("regime", {}) or {}
        r1 = "PASS" if regime.get("r1_pass") else "FAIL"
        fprint(f"{name:<32} {m.get('sharpe',''):>7} {m.get('sortino',''):>8} "
               f"{m.get('profit_factor',''):>6} {m.get('win_rate',''):>5} "
               f"{m.get('cagr',''):>7} {m.get('max_dd',''):>7} "
               f"{m.get('calmar',''):>7} {m.get('final_equity',''):>8} "
               f"{m.get('perm_p_value',''):>6} {r1:>4}")

    # ── Save Results ─────────────────────────────────────────────────────
    output = {
        "run_time": datetime.now().isoformat(),
        "capital": CAPITAL,
        "start_date": START_DATE,
        "n_permutations": N_PERMUTATIONS,
        "lgbm_train_days": LGBM_TRAIN_DAYS,
        "variants": results,
        "correlation": corr_result,
        "risk_parity_weights": {
            "equity_rotation": round(inv_vol_w[0], 4),
            "market_neutral": round(inv_vol_w[1], 4),
            "momentum_burst": round(inv_vol_w[2], 4),
        },
    }

    # Convert numpy types for JSON
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj

    def _deep_convert(obj):
        if isinstance(obj, dict):
            return {str(k): _deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_deep_convert(v) for v in obj]
        return _convert(obj)

    output = _deep_convert(output)

    json_path = OUTPUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {json_path}")

    # Save monthly returns for each variant
    returns_out = {}
    for name, v in variants.items():
        rets_list = v["rets"].tolist() if hasattr(v["rets"], "tolist") else list(v["rets"])
        dates_list = [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in v["dates"]]
        returns_out[name] = {"dates": dates_list, "returns": rets_list}

    returns_path = OUTPUT_DIR / "monthly_returns.json"
    with open(returns_path, "w") as f:
        json.dump(returns_out, f, indent=2, default=str)
    fprint(f"Monthly returns saved to {returns_path}")

    # ── MLflow Logging ───────────────────────────────────────────────────
    if MLFLOW_OK:
        fprint("\nLogging to MLflow...")
        try:
            mlflow.set_experiment("honest_portfolio_combiner_v1")
            with mlflow.start_run(run_name=f"combiner_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("capital", CAPITAL)
                mlflow.log_param("start_date", START_DATE)
                mlflow.log_param("lgbm_train_days", LGBM_TRAIN_DAYS)
                mlflow.log_param("n_permutations", N_PERMUTATIONS)
                mlflow.log_param("n_sectors", len(SECTOR_ETFS))

                for name, m in results.items():
                    prefix = name.split(")")[0].strip() + ")"
                    prefix = prefix.replace(" ", "_").replace("/", "_")
                    for k in ["sharpe", "sortino", "profit_factor", "win_rate",
                              "cagr", "max_dd", "calmar", "final_equity", "perm_p_value"]:
                        if k in m and m[k] is not None:
                            mlflow.log_metric(f"{prefix}_{k}", float(m[k]))

                    regime = m.get("regime", {}) or {}
                    for rk in ["sharpe_bull", "sharpe_bear", "sharpe_flat", "regime_gap"]:
                        if rk in regime and regime[rk] is not None:
                            mlflow.log_metric(f"{prefix}_{rk}", float(regime[rk]))

                mlflow.log_artifact(str(json_path))
                mlflow.log_artifact(str(returns_path))
            fprint("MLflow logging complete")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    fprint("DONE")


if __name__ == "__main__":
    main()
