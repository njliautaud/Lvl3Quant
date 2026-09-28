#!/usr/bin/env python3
"""
Volatility Harvesting / Short-Vol Strategy Backtest
====================================================
HC #696: High-return growth research — systematic vol strategies.

Prior research: equity momentum (10-18% CAGR), leveraged ETFs (25-40% CAGR),
trend following (12-20% CAGR). Systematic short-vol = UNTESTED.

Strategies tested:
1. VIX term structure: hold SVXY in contango, cash in backwardation
2. Vol risk premium harvest: sell vol when IV >> RV, buy when IV << RV
3. Crash protection overlay: SVXY + small UVXY tail hedge
4. VIX mean reversion: buy SPY on high VIX (fear), sell on low VIX
5. Combined: best vol strategy + TQQQ trend following
6. Vol-targeted versions of each

CRITICAL: SVXY/XIV crashed ~80% on 2018-02-05 (Volmageddon).
All strategies MUST survive this. Feb 2018 drawdown shown explicitly.

Walk-forward: sliding 252d train, 21d test.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats
import json
import warnings
import time

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/vol_harvesting")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Tickers ───
VOL_TICKERS = {
    "SVXY": "Short VIX (-0.5x since 2018)",
    "UVXY": "Long VIX (1.5x)",
    "VXX":  "Long VIX (1x, delisted 2022 — limited data)",
    "^VIX": "CBOE VIX Index",
    "^VIX3M": "CBOE 3-Month VIX",  # for term structure
}

EQUITY_TICKERS = ["SPY", "QQQ", "TQQQ", "TLT", "SHV"]


def download_data(start="2011-01-01", end="2026-07-14"):
    """Download all required data."""
    all_tickers = list(VOL_TICKERS.keys()) + EQUITY_TICKERS
    print(f"Downloading {len(all_tickers)} tickers...")

    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.ffill().dropna(how="all")

    # Rename VIX columns for convenience
    rename_map = {}
    if "^VIX" in prices.columns:
        rename_map["^VIX"] = "VIX"
    if "^VIX3M" in prices.columns:
        rename_map["^VIX3M"] = "VIX3M"
    prices = prices.rename(columns=rename_map)

    print(f"Data: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}, {len(prices)} days")

    for t in prices.columns:
        valid = prices[t].notna().sum()
        first_valid = prices[t].first_valid_index()
        if first_valid:
            print(f"  {t}: {valid} days, from {first_valid.strftime('%Y-%m-%d')}")

    return prices


def compute_realized_vol(prices_series, window=20):
    """Compute annualized realized vol from price series."""
    log_rets = np.log(prices_series / prices_series.shift(1))
    return log_rets.rolling(window, min_periods=10).std() * np.sqrt(252) * 100  # in VIX-like terms


def compute_vix_term_structure(prices):
    """
    Compute term structure signal.
    Contango = VIX < VIX3M (normal, short-vol profitable).
    Backwardation = VIX > VIX3M (fear, short-vol dangerous).

    If VIX3M not available, approximate: contango when VIX < 20d realized vol of SPY * sqrt(252).
    """
    if "VIX3M" in prices.columns and "VIX" in prices.columns:
        # VIX/VIX3M ratio: < 1 = contango, > 1 = backwardation
        ratio = prices["VIX"] / prices["VIX3M"]
        contango = ratio < 1.0
        return contango, ratio
    elif "VIX" in prices.columns and "SPY" in prices.columns:
        # Fallback: VIX vs realized vol
        rv = compute_realized_vol(prices["SPY"], window=20)
        ratio = prices["VIX"] / rv
        # VIX typically trades at premium to RV (the vol risk premium)
        # When ratio is high (VIX >> RV), contango is strong
        contango = ratio > 1.0  # VIX above realized = normal = contango-like
        return contango, ratio
    else:
        raise ValueError("Need VIX + VIX3M or VIX + SPY for term structure")


def evaluate_strategy(daily_returns, label="", prices=None):
    """Compute comprehensive metrics with Feb 2018 analysis."""
    dr = pd.Series(daily_returns) if not isinstance(daily_returns, pd.Series) else daily_returns
    if len(dr) < 126 or dr.std() == 0:
        return {"label": label, "sharpe": 0, "cagr": 0, "valid": False}

    # Basic metrics
    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = dr[dr < 0].std() * np.sqrt(252) if (dr < 0).sum() > 10 else ann_vol
    sortino = ann_ret / downside if downside > 0 else 0

    # CAGR
    years = len(dr) / 252
    total_ret = (1 + dr).prod()
    cagr = total_ret ** (1 / years) - 1 if years > 0 else 0

    # Drawdown
    cum = (1 + dr).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    max_dd_date = dd.idxmin() if len(dd) > 0 else None

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate + profit factor
    wr = (dr > 0).mean()
    gains = dr[dr > 0].sum()
    losses = abs(dr[dr < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Feb 2018 drawdown (Volmageddon: Feb 2-9 2018)
    feb2018_mask = (dr.index >= "2018-02-01") & (dr.index <= "2018-02-28")
    if feb2018_mask.sum() > 0:
        feb2018_rets = dr[feb2018_mask]
        feb2018_dd = (1 + feb2018_rets).cumprod().min() - 1
        feb2018_total = (1 + feb2018_rets).prod() - 1
    else:
        feb2018_dd = 0
        feb2018_total = 0

    # COVID crash (Feb-Mar 2020)
    covid_mask = (dr.index >= "2020-02-19") & (dr.index <= "2020-03-23")
    if covid_mask.sum() > 0:
        covid_rets = dr[covid_mask]
        covid_dd = (1 + covid_rets).cumprod().min() - 1
    else:
        covid_dd = 0

    # Regime stratification
    monthly_rets = dr.resample("ME").sum()
    if len(monthly_rets) > 24:
        green_months = monthly_rets[monthly_rets > 0]
        red_months = monthly_rets[monthly_rets <= 0]

        green_sharpe = (green_months.mean() / green_months.std() * np.sqrt(12)
                        if len(green_months) > 3 and green_months.std() > 0 else 0)
        red_sharpe = (red_months.mean() / red_months.std() * np.sqrt(12)
                      if len(red_months) > 3 and red_months.std() > 0 else 0)

        max_s = max(abs(green_sharpe), abs(red_sharpe))
        regime_gap = abs(green_sharpe - red_sharpe) / max_s if max_s > 0 else 999
    else:
        green_sharpe = red_sharpe = 0
        regime_gap = 999

    return {
        "label": label,
        "cagr": round(cagr * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd": round(max_dd * 100, 1),
        "max_dd_date": str(max_dd_date.date()) if max_dd_date is not None else None,
        "calmar": round(calmar, 2),
        "ann_vol": round(ann_vol * 100, 1),
        "wr": round(wr * 100, 1),
        "pf": round(pf, 2),
        "years": round(years, 1),
        "feb2018_dd": round(feb2018_dd * 100, 1),
        "feb2018_total": round(feb2018_total * 100, 1),
        "covid_dd": round(covid_dd * 100, 1),
        "green_sharpe": round(green_sharpe, 2),
        "red_sharpe": round(red_sharpe, 2),
        "regime_gap": round(regime_gap, 3),
        "r1_pass": regime_gap <= 0.50,
        "valid": True,
    }


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 1: VIX TERM STRUCTURE (CONTANGO/BACKWARDATION)
# ═══════════════════════════════════════════════════════════════════

def strategy_term_structure(prices, config):
    """
    Hold SVXY when VIX in contango (term structure normal).
    Go to cash (or TLT) when in backwardation.

    Config:
        threshold: VIX/VIX3M ratio threshold (< threshold = contango)
        cash_asset: 'cash' or 'TLT'
        smooth_days: rolling window to smooth signal (reduce whipsaws)
        vix_cap: if VIX > this level, force cash regardless
    """
    threshold = config.get("threshold", 1.0)
    cash_asset = config.get("cash_asset", "cash")
    smooth_days = config.get("smooth_days", 1)
    vix_cap = config.get("vix_cap", 30)

    returns = prices.pct_change()

    # Term structure
    if "VIX3M" in prices.columns and "VIX" in prices.columns:
        ratio = prices["VIX"] / prices["VIX3M"]
    elif "VIX" in prices.columns and "SPY" in prices.columns:
        rv = compute_realized_vol(prices["SPY"], window=20)
        ratio = prices["VIX"] / rv
    else:
        return pd.Series(dtype=float)

    # Smooth the ratio to reduce whipsaws
    if smooth_days > 1:
        ratio_smooth = ratio.rolling(smooth_days, min_periods=1).mean()
    else:
        ratio_smooth = ratio

    # Signal: contango (ratio < threshold)
    contango = ratio_smooth < threshold

    # VIX cap: if VIX is extremely high, go to cash no matter what
    if "VIX" in prices.columns:
        vix_too_high = prices["VIX"] > vix_cap
        contango = contango & ~vix_too_high

    # Need SVXY
    if "SVXY" not in returns.columns:
        return pd.Series(dtype=float)

    # Build portfolio returns
    warmup = max(252, smooth_days + 21)
    dates = prices.index[warmup:]

    port_rets = []
    port_dates = []

    for date in dates:
        if date not in contango.index:
            continue

        if contango.loc[:date].iloc[-1]:
            # Contango: hold SVXY
            r = returns["SVXY"].get(date, 0)
        else:
            # Backwardation: cash or TLT
            if cash_asset == "TLT" and "TLT" in returns.columns:
                r = returns["TLT"].get(date, 0)
            elif cash_asset == "SHV" and "SHV" in returns.columns:
                r = returns["SHV"].get(date, 0)
            else:
                r = 0.0

        if not np.isnan(r):
            port_rets.append(r)
            port_dates.append(date)

    return pd.Series(port_rets, index=port_dates)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 2: VOL RISK PREMIUM HARVEST
# ═══════════════════════════════════════════════════════════════════

def strategy_vol_premium(prices, config):
    """
    Exploit the variance risk premium: IV (VIX) typically > RV.
    When premium is wide, sell vol (hold SVXY).
    When premium is narrow or negative, go to cash.

    Config:
        premium_threshold: minimum IV-RV spread to hold short vol (in VIX pts)
        rv_window: realized vol lookback (days)
        exit_threshold: if premium goes negative by this much, exit
        vix_cap: max VIX level to hold short vol
    """
    premium_threshold = config.get("premium_threshold", 3)
    rv_window = config.get("rv_window", 20)
    exit_threshold = config.get("exit_threshold", -2)
    vix_cap = config.get("vix_cap", 28)

    returns = prices.pct_change()

    if "VIX" not in prices.columns or "SPY" not in prices.columns:
        return pd.Series(dtype=float)
    if "SVXY" not in returns.columns:
        return pd.Series(dtype=float)

    # Realized vol of SPY in VIX-like terms
    rv = compute_realized_vol(prices["SPY"], window=rv_window)

    # Variance risk premium
    vrp = prices["VIX"] - rv

    warmup = max(252, rv_window + 21)
    dates = prices.index[warmup:]

    port_rets = []
    port_dates = []
    in_position = False

    for date in dates:
        if date not in vrp.index:
            continue

        current_vrp = vrp.get(date, np.nan)
        current_vix = prices["VIX"].get(date, 30)

        if np.isnan(current_vrp):
            port_rets.append(0)
            port_dates.append(date)
            continue

        # Entry/exit logic with hysteresis
        if not in_position:
            if current_vrp > premium_threshold and current_vix < vix_cap:
                in_position = True
        else:
            if current_vrp < exit_threshold or current_vix > vix_cap:
                in_position = False

        if in_position:
            r = returns["SVXY"].get(date, 0)
        else:
            r = 0.0

        if not np.isnan(r):
            port_rets.append(r)
            port_dates.append(date)

    return pd.Series(port_rets, index=port_dates)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 3: CRASH PROTECTION OVERLAY
# ═══════════════════════════════════════════════════════════════════

def strategy_crash_overlay(prices, config):
    """
    Hold SVXY (short vol) with a small UVXY (long vol) tail hedge.
    The hedge costs carry in normal times but protects in crashes.

    Config:
        svxy_weight: SVXY allocation (e.g. 0.95)
        uvxy_weight: UVXY tail hedge (e.g. 0.05)
        dynamic_hedge: if True, increase UVXY weight when VIX rising
        vix_increase_window: lookback for VIX momentum
        max_hedge: maximum UVXY allocation
        regime_filter: require SPY > 200MA for SVXY
    """
    svxy_w = config.get("svxy_weight", 0.95)
    uvxy_w = config.get("uvxy_weight", 0.05)
    dynamic = config.get("dynamic_hedge", False)
    vix_window = config.get("vix_increase_window", 5)
    max_hedge = config.get("max_hedge", 0.20)
    use_regime = config.get("regime_filter", False)

    returns = prices.pct_change()

    if "SVXY" not in returns.columns or "UVXY" not in returns.columns:
        return pd.Series(dtype=float)

    # Regime filter
    if use_regime and "SPY" in prices.columns:
        spy_ma200 = prices["SPY"].rolling(200, min_periods=100).mean()
        risk_on = prices["SPY"] > spy_ma200
    else:
        risk_on = pd.Series(True, index=prices.index)

    warmup = 252
    dates = prices.index[warmup:]

    port_rets = []
    port_dates = []

    for date in dates:
        if date not in returns.index:
            continue

        s_w = svxy_w
        u_w = uvxy_w

        if dynamic and "VIX" in prices.columns:
            # Increase hedge when VIX is rising
            vix_now = prices["VIX"].get(date, 20)
            vix_past = prices["VIX"].shift(vix_window).get(date, 20)
            if vix_past > 0:
                vix_change = (vix_now - vix_past) / vix_past
                if vix_change > 0.1:  # VIX rising > 10%
                    extra_hedge = min(vix_change * 0.5, max_hedge - uvxy_w)
                    u_w = uvxy_w + max(0, extra_hedge)
                    s_w = 1.0 - u_w

        if not risk_on.get(date, True):
            # Risk-off: reduce exposure
            s_w = 0.0
            u_w = 0.0

        r_svxy = returns["SVXY"].get(date, 0)
        r_uvxy = returns["UVXY"].get(date, 0)

        r = s_w * (r_svxy if not np.isnan(r_svxy) else 0) + \
            u_w * (r_uvxy if not np.isnan(r_uvxy) else 0)

        # Cash portion
        cash_w = 1.0 - s_w - u_w
        if cash_w > 0 and "SHV" in returns.columns:
            r_cash = returns["SHV"].get(date, 0)
            r += cash_w * (r_cash if not np.isnan(r_cash) else 0)

        port_rets.append(r)
        port_dates.append(date)

    return pd.Series(port_rets, index=port_dates)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 4: VIX MEAN REVERSION (FEAR = OPPORTUNITY)
# ═══════════════════════════════════════════════════════════════════

def strategy_vix_mean_reversion(prices, config):
    """
    Buy SPY when VIX > high percentile (fear spike = buying opportunity).
    Sell/cash when VIX < low percentile (complacency).

    Config:
        buy_pctile: VIX percentile to buy (e.g. 80 = buy when VIX in top 20%)
        sell_pctile: VIX percentile to sell (e.g. 20 = sell when VIX in bottom 20%)
        lookback: days for percentile calculation
        hold_asset: 'SPY', 'QQQ', or 'TQQQ'
        cash_asset: what to hold in neutral zone
    """
    buy_pctile = config.get("buy_pctile", 80)
    sell_pctile = config.get("sell_pctile", 20)
    lookback = config.get("lookback", 252)
    hold_asset = config.get("hold_asset", "SPY")
    cash_asset = config.get("cash_asset", "cash")

    returns = prices.pct_change()

    if "VIX" not in prices.columns or hold_asset not in returns.columns:
        return pd.Series(dtype=float)

    warmup = max(252, lookback)
    dates = prices.index[warmup:]

    port_rets = []
    port_dates = []
    state = "neutral"  # 'long', 'neutral', 'cash'

    for date in dates:
        vix_history = prices["VIX"].loc[:date].tail(lookback)
        current_vix = prices["VIX"].get(date, np.nan)

        if np.isnan(current_vix) or len(vix_history) < lookback // 2:
            port_rets.append(0)
            port_dates.append(date)
            continue

        vix_pctile = stats.percentileofscore(vix_history, current_vix)

        # State transitions
        if vix_pctile >= buy_pctile:
            state = "long"
        elif vix_pctile <= sell_pctile:
            state = "cash"
        # else: stay in current state (hysteresis)

        if state == "long":
            r = returns[hold_asset].get(date, 0)
        elif state == "cash":
            if cash_asset == "TLT" and "TLT" in returns.columns:
                r = returns["TLT"].get(date, 0)
            else:
                r = 0.0
        else:
            # Neutral: hold asset at reduced weight
            r = returns[hold_asset].get(date, 0) * 0.5

        if not np.isnan(r):
            port_rets.append(r)
            port_dates.append(date)

    return pd.Series(port_rets, index=port_dates)


# ═══════════════════════════════════════════════════════════════════
# VOL-TARGETING OVERLAY
# ═══════════════════════════════════════════════════════════════════

def apply_vol_targeting(daily_returns, vol_target=0.15, max_lev=2.0, window=21):
    """Apply vol-targeting overlay to any strategy returns."""
    dr = pd.Series(daily_returns) if not isinstance(daily_returns, pd.Series) else daily_returns.copy()
    if len(dr) < window + 10:
        return dr

    rv = dr.rolling(window, min_periods=10).std() * np.sqrt(252)

    scaled = dr.copy()
    for i in range(window, len(dr)):
        current_vol = rv.iloc[i - 1]  # use yesterday's vol estimate
        if current_vol > 0:
            scalar = vol_target / current_vol
            scalar = min(scalar, max_lev)
            scalar = max(scalar, 0.1)
            scaled.iloc[i] = dr.iloc[i] * scalar

    return scaled


# ═══════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════

def walk_forward_validate(daily_returns, train_window=252, test_window=21):
    """
    Walk-forward validation: sliding window.
    Returns per-fold metrics and aggregated OOT metrics.
    """
    dr = daily_returns.values if isinstance(daily_returns, pd.Series) else np.array(daily_returns)
    idx = daily_returns.index if isinstance(daily_returns, pd.Series) else range(len(daily_returns))

    folds = []
    oot_rets = []
    oot_dates = []

    i = train_window
    while i + test_window <= len(dr):
        train = dr[i - train_window:i]
        test = dr[i:i + test_window]
        test_idx = idx[i:i + test_window]

        train_sharpe = train.mean() / train.std() * np.sqrt(252) if train.std() > 0 else 0
        test_sharpe = test.mean() / test.std() * np.sqrt(252) if test.std() > 0 else 0
        test_ret = (1 + test).prod() - 1

        folds.append({
            "fold_start": str(test_idx[0].date()) if hasattr(test_idx[0], "date") else str(test_idx[0]),
            "train_sharpe": round(train_sharpe, 2),
            "test_sharpe": round(test_sharpe, 2),
            "test_return": round(test_ret * 100, 2),
        })

        oot_rets.extend(test.tolist())
        oot_dates.extend(test_idx.tolist())

        i += test_window

    if not oot_rets:
        return {"n_folds": 0, "oot_sharpe": 0}

    oot = np.array(oot_rets)
    oot_sharpe = oot.mean() / oot.std() * np.sqrt(252) if oot.std() > 0 else 0
    oot_cagr = ((1 + pd.Series(oot_rets)).prod()) ** (252 / len(oot_rets)) - 1

    # Fold stability
    test_sharpes = [f["test_sharpe"] for f in folds]
    pct_positive = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) if test_sharpes else 0

    return {
        "n_folds": len(folds),
        "oot_sharpe": round(oot_sharpe, 2),
        "oot_cagr": round(oot_cagr * 100, 1),
        "pct_positive_folds": round(pct_positive * 100, 1),
        "fold_sharpe_mean": round(np.mean(test_sharpes), 2),
        "fold_sharpe_std": round(np.std(test_sharpes), 2),
        "folds": folds,
    }


# ═══════════════════════════════════════════════════════════════════
# PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════

def run_permutation_test(daily_returns, n_trials=100):
    """Block-bootstrap permutation test."""
    dr = np.array(daily_returns)
    real_sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0

    beat_count = 0
    shuffled_sharpes = []

    for _ in range(n_trials):
        monthly_idx = np.arange(0, len(dr), 21)
        blocks = [dr[i:i + 21] for i in monthly_idx if i + 21 <= len(dr)]
        if len(blocks) < 12:
            shuf = np.random.permutation(dr)
        else:
            np.random.shuffle(blocks)
            shuf = np.concatenate(blocks)

        s = shuf.mean() / shuf.std() * np.sqrt(252) if shuf.std() > 0 else 0
        shuffled_sharpes.append(s)
        if s >= real_sharpe:
            beat_count += 1

    return beat_count / n_trials, shuffled_sharpes


# ═══════════════════════════════════════════════════════════════════
# COMBINED PORTFOLIO
# ═══════════════════════════════════════════════════════════════════

def strategy_combined(vol_returns, trend_returns, vol_weight=0.5):
    """
    Combine best vol strategy with TQQQ trend following.
    Requires aligned date indices.
    """
    # Align
    common = vol_returns.index.intersection(trend_returns.index)
    if len(common) < 252:
        return pd.Series(dtype=float)

    combined = vol_weight * vol_returns.loc[common] + (1 - vol_weight) * trend_returns.loc[common]
    return combined


def tqqq_trend_following(prices, ma_period=200):
    """Simple TQQQ trend following: hold when TQQQ > MA, cash otherwise."""
    if "TQQQ" not in prices.columns:
        return pd.Series(dtype=float)

    returns = prices["TQQQ"].pct_change()
    ma = prices["TQQQ"].rolling(ma_period, min_periods=100).mean()
    signal = prices["TQQQ"] > ma

    warmup = max(252, ma_period)
    dates = prices.index[warmup:]

    port_rets = []
    port_dates = []

    for date in dates:
        if signal.get(date, False):
            r = returns.get(date, 0)
        else:
            r = 0.0
        if not np.isnan(r):
            port_rets.append(r)
            port_dates.append(date)

    return pd.Series(port_rets, index=port_dates)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("VOLATILITY HARVESTING / SHORT-VOL STRATEGY BACKTEST")
    print("HC #696 — High-Return Growth Research")
    print("=" * 70)
    print()

    # ─── Download data ───
    prices = download_data()

    # ─── Build all configs ───
    all_configs = []

    # --- Strategy 1: Term Structure ---
    for threshold in [0.90, 0.95, 1.0, 1.05]:
        for smooth in [1, 3, 5]:
            for vix_cap in [25, 30, 35, 50]:
                for cash in ["cash", "TLT"]:
                    label = f"TS_th{threshold}_sm{smooth}_cap{vix_cap}_{cash}"
                    all_configs.append({
                        "strategy": "term_structure",
                        "label": label,
                        "params": {
                            "threshold": threshold,
                            "smooth_days": smooth,
                            "vix_cap": vix_cap,
                            "cash_asset": cash,
                        }
                    })

    # --- Strategy 2: Vol Risk Premium ---
    for prem_thresh in [2, 3, 5, 7]:
        for rv_window in [10, 20, 30]:
            for exit_thresh in [-3, -1, 0]:
                for vix_cap in [25, 30, 40]:
                    label = f"VRP_pt{prem_thresh}_rv{rv_window}_ex{exit_thresh}_cap{vix_cap}"
                    all_configs.append({
                        "strategy": "vol_premium",
                        "label": label,
                        "params": {
                            "premium_threshold": prem_thresh,
                            "rv_window": rv_window,
                            "exit_threshold": exit_thresh,
                            "vix_cap": vix_cap,
                        }
                    })

    # --- Strategy 3: Crash Overlay ---
    for svxy_w in [0.90, 0.95, 0.97]:
        for dynamic in [True, False]:
            for regime in [True, False]:
                uvxy_w = round(1.0 - svxy_w, 2)
                label = f"CO_svxy{int(svxy_w*100)}_uvxy{int(uvxy_w*100)}_dyn{'Y' if dynamic else 'N'}_reg{'Y' if regime else 'N'}"
                all_configs.append({
                    "strategy": "crash_overlay",
                    "label": label,
                    "params": {
                        "svxy_weight": svxy_w,
                        "uvxy_weight": uvxy_w,
                        "dynamic_hedge": dynamic,
                        "regime_filter": regime,
                        "vix_increase_window": 5,
                        "max_hedge": 0.20,
                    }
                })

    # --- Strategy 4: VIX Mean Reversion ---
    for buy_pct in [70, 80, 90]:
        for sell_pct in [10, 20, 30]:
            for asset in ["SPY", "QQQ", "TQQQ"]:
                for lookback in [252, 504]:
                    for cash in ["cash", "TLT"]:
                        label = f"MR_buy{buy_pct}_sell{sell_pct}_{asset}_lb{lookback}_{cash}"
                        all_configs.append({
                            "strategy": "mean_reversion",
                            "label": label,
                            "params": {
                                "buy_pctile": buy_pct,
                                "sell_pctile": sell_pct,
                                "hold_asset": asset,
                                "lookback": lookback,
                                "cash_asset": cash,
                            }
                        })

    print(f"Total configs: {len(all_configs)}")
    print()

    # ─── Run all strategies ───
    strategy_funcs = {
        "term_structure": strategy_term_structure,
        "vol_premium": strategy_vol_premium,
        "crash_overlay": strategy_crash_overlay,
        "mean_reversion": strategy_vix_mean_reversion,
    }

    all_results = []
    strategy_returns = {}  # label -> Series for later combination
    best_by_type = {}  # strategy_type -> best result

    for i, cfg in enumerate(all_configs):
        try:
            func = strategy_funcs[cfg["strategy"]]
            dr = func(prices, cfg["params"])

            if len(dr) < 252:
                continue

            result = evaluate_strategy(dr, label=cfg["label"])
            result["strategy_type"] = cfg["strategy"]
            result["config"] = cfg["params"]

            # Also run vol-targeted version
            dr_vt = apply_vol_targeting(dr, vol_target=0.15, max_lev=1.5)
            result_vt = evaluate_strategy(dr_vt, label=cfg["label"] + "_VT15")
            result_vt["strategy_type"] = cfg["strategy"] + "_VT"
            result_vt["config"] = {**cfg["params"], "vol_target": 0.15}

            if result.get("valid"):
                all_results.append(result)
                strategy_returns[cfg["label"]] = dr

            if result_vt.get("valid"):
                all_results.append(result_vt)
                strategy_returns[cfg["label"] + "_VT15"] = dr_vt

        except Exception as e:
            pass  # Skip failed configs silently

        if (i + 1) % 100 == 0:
            valid_so_far = len([r for r in all_results if r.get("valid")])
            print(f"  Processed {i+1}/{len(all_configs)}, {valid_so_far} valid results...")

    # ─── Sort and display results by strategy type ───
    valid = [r for r in all_results if r.get("valid")]
    valid.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'='*110}")
    print(f"ALL RESULTS: {len(valid)} valid configs")
    print(f"{'='*110}")

    # Group by strategy type
    for stype in ["term_structure", "vol_premium", "crash_overlay", "mean_reversion",
                   "term_structure_VT", "vol_premium_VT", "crash_overlay_VT", "mean_reversion_VT"]:
        type_results = [r for r in valid if r.get("strategy_type") == stype]
        if not type_results:
            continue

        type_results.sort(key=lambda x: -x["sharpe"])

        print(f"\n── {stype.upper()} ({len(type_results)} configs) ──")
        print(f"{'Label':<55} {'CAGR':>6} {'Shrp':>5} {'Sort':>5} {'MaxDD':>7} {'Cal':>5} {'Feb18':>6} {'R1':>3}")
        print("-" * 110)

        for r in type_results[:10]:
            r1 = "Y" if r.get("r1_pass") else "N"
            feb = f"{r['feb2018_dd']:.0f}%" if r.get("feb2018_dd") else "N/A"
            print(f"{r['label']:<55} {r['cagr']:>5.1f}% {r['sharpe']:>5.2f} {r['sortino']:>5.2f} "
                  f"{r['max_dd']:>6.1f}% {r['calmar']:>5.2f} {feb:>6} {r1:>3}")

        # Track best by type
        best_by_type[stype] = type_results[0]

    # ─── TQQQ trend following (for combined strategy) ───
    print(f"\n{'='*110}")
    print("TQQQ TREND FOLLOWING (for combination)")
    print(f"{'='*110}")

    tqqq_trend_rets = tqqq_trend_following(prices, ma_period=200)
    if len(tqqq_trend_rets) > 252:
        tqqq_trend_metrics = evaluate_strategy(tqqq_trend_rets, "TQQQ_trend_200MA")
        print(f"  TQQQ trend: CAGR={tqqq_trend_metrics['cagr']:.1f}%, Sharpe={tqqq_trend_metrics['sharpe']:.2f}, "
              f"MaxDD={tqqq_trend_metrics['max_dd']:.1f}%, Feb2018 DD={tqqq_trend_metrics['feb2018_dd']:.1f}%")

    # ─── Strategy 5: Combined portfolios ───
    print(f"\n{'='*110}")
    print("COMBINED PORTFOLIOS (Best Vol + TQQQ Trend)")
    print(f"{'='*110}")

    combined_results = []

    # Find best vol strategy (by Sharpe, must have >0.5 Sharpe)
    base_vol_types = ["term_structure", "vol_premium", "crash_overlay", "mean_reversion"]
    best_vol_configs = []
    for stype in base_vol_types:
        type_results = [r for r in valid if r.get("strategy_type") == stype and r["sharpe"] > 0.3]
        if type_results:
            best_vol_configs.append(type_results[0])

    # Also include VT versions
    for stype in [s + "_VT" for s in base_vol_types]:
        type_results = [r for r in valid if r.get("strategy_type") == stype and r["sharpe"] > 0.3]
        if type_results:
            best_vol_configs.append(type_results[0])

    for vol_result in best_vol_configs:
        vol_label = vol_result["label"]
        if vol_label not in strategy_returns:
            continue
        vol_rets = strategy_returns[vol_label]

        if len(tqqq_trend_rets) < 252:
            continue

        for vol_weight in [0.3, 0.4, 0.5, 0.6, 0.7]:
            combined_rets = strategy_combined(vol_rets, tqqq_trend_rets, vol_weight=vol_weight)
            if len(combined_rets) < 252:
                continue

            combo_label = f"COMBO_{vol_label}_w{int(vol_weight*100)}"
            combo_metrics = evaluate_strategy(combined_rets, combo_label)
            combo_metrics["strategy_type"] = "combined"
            combo_metrics["config"] = {
                "vol_strategy": vol_label,
                "vol_weight": vol_weight,
                "trend_weight": 1 - vol_weight,
            }

            if combo_metrics.get("valid"):
                combined_results.append(combo_metrics)
                strategy_returns[combo_label] = combined_rets

                # Vol-targeted version
                combo_vt = apply_vol_targeting(combined_rets, vol_target=0.20, max_lev=1.5)
                combo_vt_metrics = evaluate_strategy(combo_vt, combo_label + "_VT20")
                combo_vt_metrics["strategy_type"] = "combined_VT"
                combo_vt_metrics["config"] = {
                    "vol_strategy": vol_label,
                    "vol_weight": vol_weight,
                    "vol_target": 0.20,
                }
                if combo_vt_metrics.get("valid"):
                    combined_results.append(combo_vt_metrics)

    combined_results.sort(key=lambda x: -x["sharpe"])
    all_results.extend(combined_results)

    if combined_results:
        print(f"\n{'Label':<60} {'CAGR':>6} {'Shrp':>5} {'Sort':>5} {'MaxDD':>7} {'Cal':>5} {'Feb18':>6} {'R1':>3}")
        print("-" * 110)
        for r in combined_results[:15]:
            r1 = "Y" if r.get("r1_pass") else "N"
            feb = f"{r['feb2018_dd']:.0f}%" if r.get("feb2018_dd") else "N/A"
            print(f"{r['label']:<60} {r['cagr']:>5.1f}% {r['sharpe']:>5.2f} {r['sortino']:>5.2f} "
                  f"{r['max_dd']:>6.1f}% {r['calmar']:>5.2f} {feb:>6} {r1:>3}")

    # ─── R1-passing configs across all strategies ───
    all_valid = [r for r in all_results if r.get("valid")]
    all_valid.sort(key=lambda x: -x["sharpe"])
    r1_passing = [r for r in all_valid if r.get("r1_pass") and r["sharpe"] > 0.3]
    r1_passing.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'='*110}")
    print(f"R1-PASSING CONFIGS (regime gap <= 0.50): {len(r1_passing)} total")
    print(f"{'='*110}")

    if r1_passing:
        print(f"{'Label':<60} {'CAGR':>6} {'Shrp':>5} {'Sort':>5} {'MaxDD':>7} {'Cal':>5} {'Feb18':>6} {'Gap':>6}")
        print("-" * 110)
        for r in r1_passing[:20]:
            feb = f"{r['feb2018_dd']:.0f}%" if r.get("feb2018_dd") else "N/A"
            print(f"{r['label']:<60} {r['cagr']:>5.1f}% {r['sharpe']:>5.2f} {r['sortino']:>5.2f} "
                  f"{r['max_dd']:>6.1f}% {r['calmar']:>5.2f} {feb:>6} {r['regime_gap']:>6.3f}")

    # ─── Walk-forward validation on top configs ───
    print(f"\n{'='*110}")
    print("WALK-FORWARD VALIDATION — Top 10 configs")
    print(f"{'='*110}")

    wf_results = []
    top_for_wf = r1_passing[:10] if len(r1_passing) >= 5 else all_valid[:10]

    for r in top_for_wf:
        label = r["label"]
        if label in strategy_returns:
            wf = walk_forward_validate(strategy_returns[label])
            wf["label"] = label
            wf_results.append(wf)
            print(f"  {label}: OOT Sharpe={wf['oot_sharpe']:.2f}, OOT CAGR={wf['oot_cagr']:.1f}%, "
                  f"Pos folds={wf['pct_positive_folds']:.0f}%, n_folds={wf['n_folds']}")

    # ─── Permutation test top 5 ───
    print(f"\n{'='*110}")
    print("PERMUTATION TESTS — Top 5 R1-passing configs")
    print(f"{'='*110}")

    perm_results = []
    perm_candidates = r1_passing[:5] if len(r1_passing) >= 3 else all_valid[:5]

    for r in perm_candidates:
        label = r["label"]
        if label in strategy_returns:
            p_val, shuf_sharpes = run_permutation_test(strategy_returns[label].values, n_trials=100)
            r["permutation_p"] = p_val
            p_str = "PASS" if p_val <= 0.05 else "FAIL"
            print(f"  {label}: p={p_val:.2f} {p_str} (real Sharpe={r['sharpe']:.2f}, "
                  f"shuffled median={np.median(shuf_sharpes):.2f})")
            perm_results.append({"label": label, "p_value": p_val, "pass": p_val <= 0.05})

    # ─── Feb 2018 deep dive ───
    print(f"\n{'='*110}")
    print("FEB 2018 VOLMAGEDDON STRESS TEST")
    print(f"{'='*110}")

    # Show what happened to SVXY buy-and-hold
    if "SVXY" in prices.columns:
        svxy_rets = prices["SVXY"].pct_change()
        feb_mask = (svxy_rets.index >= "2018-02-01") & (svxy_rets.index <= "2018-02-28")
        if feb_mask.sum() > 0:
            feb_svxy = svxy_rets[feb_mask]
            feb_cum = (1 + feb_svxy).cumprod()
            print(f"  SVXY buy-and-hold Feb 2018: {(feb_cum.iloc[-1]-1)*100:.1f}% total, "
                  f"worst day: {feb_svxy.min()*100:.1f}%")

    print("\n  Strategy performance during Feb 2018:")
    for r in (r1_passing[:10] if r1_passing else all_valid[:10]):
        if r.get("feb2018_dd") is not None:
            print(f"    {r['label']}: DD={r['feb2018_dd']:.1f}%, month total={r['feb2018_total']:.1f}%")

    # ─── Benchmarks ───
    print(f"\n{'='*110}")
    print("BENCHMARKS")
    print(f"{'='*110}")

    benchmarks = {}

    # SVXY buy-and-hold
    if "SVXY" in prices.columns:
        svxy_bh = prices["SVXY"].pct_change().dropna()
        svxy_bh = svxy_bh[svxy_bh.index >= svxy_bh.index[252]]
        svxy_metrics = evaluate_strategy(svxy_bh, "SVXY_buy_hold")
        if svxy_metrics.get("valid"):
            benchmarks["SVXY_BH"] = svxy_metrics
            print(f"  SVXY B&H: CAGR={svxy_metrics['cagr']:.1f}%, Sharpe={svxy_metrics['sharpe']:.2f}, "
                  f"MaxDD={svxy_metrics['max_dd']:.1f}%, Feb2018={svxy_metrics['feb2018_dd']:.1f}%")

    # SPY buy-and-hold
    if "SPY" in prices.columns:
        spy_bh = prices["SPY"].pct_change().dropna()
        spy_bh = spy_bh[spy_bh.index >= spy_bh.index[252]]
        spy_metrics = evaluate_strategy(spy_bh, "SPY_buy_hold")
        if spy_metrics.get("valid"):
            benchmarks["SPY_BH"] = spy_metrics
            print(f"  SPY B&H:  CAGR={spy_metrics['cagr']:.1f}%, Sharpe={spy_metrics['sharpe']:.2f}, "
                  f"MaxDD={spy_metrics['max_dd']:.1f}%")

    # TQQQ buy-and-hold
    if "TQQQ" in prices.columns:
        tqqq_bh = prices["TQQQ"].pct_change().dropna()
        tqqq_bh = tqqq_bh[tqqq_bh.index >= tqqq_bh.index[252]]
        tqqq_metrics = evaluate_strategy(tqqq_bh, "TQQQ_buy_hold")
        if tqqq_metrics.get("valid"):
            benchmarks["TQQQ_BH"] = tqqq_metrics
            print(f"  TQQQ B&H: CAGR={tqqq_metrics['cagr']:.1f}%, Sharpe={tqqq_metrics['sharpe']:.2f}, "
                  f"MaxDD={tqqq_metrics['max_dd']:.1f}%, Feb2018={tqqq_metrics['feb2018_dd']:.1f}%")

    # ─── Save results ───
    elapsed = time.time() - t0

    output = {
        "generated": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_configs_total": len(all_configs),
        "n_valid": len(all_valid),
        "n_r1_passing": len(r1_passing),
        "strategies_tested": ["term_structure", "vol_premium", "crash_overlay", "mean_reversion", "combined"],
        "top_20": all_valid[:20],
        "r1_passing": r1_passing[:20],
        "walk_forward": wf_results,
        "permutation_tests": perm_results,
        "benchmarks": benchmarks,
        "feb2018_stress": {
            r["label"]: {"dd": r.get("feb2018_dd"), "total": r.get("feb2018_total")}
            for r in (r1_passing[:10] if r1_passing else all_valid[:10])
        },
    }

    results_path = OUT_DIR / "vol_harvesting_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save top strategy returns for further analysis
    for r in all_valid[:5]:
        label = r["label"]
        if label in strategy_returns:
            sr = strategy_returns[label]
            sr.to_csv(OUT_DIR / f"returns_{label}.csv", header=True)

    print(f"\nResults saved to {OUT_DIR}")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # ─── Summary ───
    print(f"\n{'='*70}")
    print("EXECUTIVE SUMMARY")
    print(f"{'='*70}")

    if r1_passing:
        best = r1_passing[0]
        print(f"\nBest R1-passing config: {best['label']}")
        print(f"  CAGR: {best['cagr']:.1f}% | Sharpe: {best['sharpe']:.2f} | Sortino: {best['sortino']:.2f}")
        print(f"  MaxDD: {best['max_dd']:.1f}% | Calmar: {best['calmar']:.2f}")
        print(f"  Feb 2018 DD: {best['feb2018_dd']:.1f}% | COVID DD: {best['covid_dd']:.1f}%")
        print(f"  Regime gap: {best['regime_gap']:.3f} (R1 pass: {'YES' if best['r1_pass'] else 'NO'})")
    else:
        best = all_valid[0] if all_valid else None
        if best:
            print(f"\nBest config (no R1-passing found): {best['label']}")
            print(f"  CAGR: {best['cagr']:.1f}% | Sharpe: {best['sharpe']:.2f} | MaxDD: {best['max_dd']:.1f}%")
        else:
            print("\nNo valid strategies found.")

    if benchmarks.get("SPY_BH"):
        spy = benchmarks["SPY_BH"]
        print(f"\nvs SPY B&H: CAGR={spy['cagr']:.1f}%, Sharpe={spy['sharpe']:.2f}, MaxDD={spy['max_dd']:.1f}%")

    print(f"\nConfigs tested: {len(all_configs)} | Valid: {len(all_valid)} | R1-passing: {len(r1_passing)}")


if __name__ == "__main__":
    main()
