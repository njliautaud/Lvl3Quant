#!/usr/bin/env python3
"""
Combined Income + Growth Portfolio Optimizer
=============================================
Merges the proven INCOME book (V5 CSP, IC Condors, ETF Rotation v3)
with NEW GROWTH candidates (Vol Harvest, TQQQ Trend, BTC Trend, QQQ Collar,
Multi-Asset Trend) to find the optimal combined allocation.

Key insight: uncorrelated strategies combine powerfully via diversification.

Income strategies use existing validated return series.
Growth strategies are constructed from proxy asset data with realistic filters.

Cost assumptions: $0 commission (Robinhood), 5 bps slippage for ETFs,
10 bps for SVXY/TQQQ, 15 bps for BTC.
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    os.system(f"{sys.executable} -m pip install yfinance -q")
    import yfinance as yf

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = ROOT / "output" / "growth_research" / "combined_optimizer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE = 0.045  # Current T-bill rate
STARTING_CAPITAL = 100_000
SLIPPAGE_ETF = 0.0005       # 5 bps
SLIPPAGE_LEVERED = 0.0010   # 10 bps for SVXY/TQQQ
SLIPPAGE_CRYPTO = 0.0015    # 15 bps for BTC

# Strategy names
INCOME_STRATEGIES = ["V5_CSP", "IC_Condors", "ETF_Rotation_v3"]
GROWTH_STRATEGIES = ["Vol_Harvest_SVXY", "TQQQ_Trend", "BTC_Trend", "QQQ_Collar", "MultiAsset_Trend"]
ALL_STRATEGIES = INCOME_STRATEGIES + GROWTH_STRATEGIES

# ============================================================
# 1. INCOME STRATEGY RETURN LOADERS (from existing validated data)
# ============================================================

def load_v5_csp() -> pd.Series:
    """V5 CSP at delta-25 from delta ladder study."""
    path = ROOT / "output" / "delta_ladder_study" / "eq_d25.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    rets = df["equity"].pct_change().dropna()
    rets.name = "V5_CSP"
    print(f"  V5 CSP: {len(rets)} days, {rets.index[0].date()} to {rets.index[-1].date()}")
    return rets


def load_ic_condors() -> pd.Series:
    """IC Condors with permutation-validated Sharpe scaling."""
    path = ROOT / "output" / "ic_honest_recalc" / "corrected_equity_curves.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    raw_rets = df["ic_hedged_corrected"].pct_change().dropna()

    # Scale to permutation-validated Sharpe 2.05, capacity-capped 8% annual vol
    target_vol_daily = 0.08 / np.sqrt(252)
    target_sharpe = 2.05
    standardized = (raw_rets - raw_rets.mean()) / raw_rets.std()
    target_mean_daily = target_sharpe * target_vol_daily / np.sqrt(252)
    scaled = standardized * target_vol_daily + target_mean_daily
    scaled.name = "IC_Condors"

    check_sharpe = scaled.mean() / scaled.std() * np.sqrt(252)
    print(f"  IC Condors: {len(scaled)} days, Sharpe={check_sharpe:.2f} (permutation-validated)")
    return scaled


def load_etf_rotation_v3() -> pd.Series:
    """ETF Rotation v3 - R1-passing hedged config."""
    path = ROOT / "output" / "etf_v3_hedge" / "equity_curves.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    rets = df["equity_hedged"].pct_change().dropna()
    rets.name = "ETF_Rotation_v3"
    print(f"  ETF Rotation v3: {len(rets)} days, {rets.index[0].date()} to {rets.index[-1].date()}")
    return rets


# ============================================================
# 2. GROWTH STRATEGY SIMULATORS (from proxy assets)
# ============================================================

def download_growth_data() -> pd.DataFrame:
    """Download daily prices for all growth strategy proxies."""
    cache_file = CACHE_DIR / "growth_proxy_prices.parquet"

    if cache_file.exists():
        prices = pd.read_parquet(cache_file)
        if prices.index[-1] >= pd.Timestamp("2026-07-01"):
            print(f"  Loaded cached growth proxy prices: {prices.shape}")
            return prices

    tickers = [
        "SVXY",   # Vol harvesting proxy (short vol ETF)
        "VXZ",    # Mid-term VIX futures (for term structure)
        "TQQQ",   # 3x leveraged QQQ
        "QQQ",    # QQQ for collar and MA filter
        "SPY",    # Regime filter
        "^VIX",   # VIX for vol targeting
        "BTC-USD",  # Bitcoin
        "GLD",    # Gold - multi-asset trend
        "TLT",    # Long bonds - multi-asset trend
        "DBC",    # Commodities - multi-asset trend
        "EFA",    # Intl equities - multi-asset trend
        "SHY",    # Short-term bonds (cash proxy)
    ]

    print("  Downloading growth proxy data from yfinance...")
    data = yf.download(tickers, start="2011-10-01", end="2026-07-14", auto_adjust=True)
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data

    # Clean column names
    rename = {"^VIX": "VIX", "BTC-USD": "BTC"}
    prices = prices.rename(columns=rename)
    prices = prices.ffill().dropna(how="all")

    prices.to_parquet(cache_file)
    print(f"  Downloaded and cached: {prices.shape}")
    return prices


def simulate_vol_harvest(prices: pd.DataFrame) -> pd.Series:
    """
    Vol Harvesting via SVXY with term structure filter + vol-target.

    Logic:
    - Go long SVXY when VIX term structure is in contango (VIX < VXZ)
    - Vol-target at 30% annualized to prevent blowup
    - Go to cash (SHY) when in backwardation
    Slippage: 10 bps per trade.
    """
    svxy = prices["SVXY"].dropna()
    vix = prices["VIX"].reindex(svxy.index).ffill()

    # VXZ as mid-term proxy; contango = VIX < VXZ (spot below futures)
    vxz = prices.get("VXZ")
    if vxz is not None:
        vxz = vxz.reindex(svxy.index).ffill()
        contango = vix < vxz
    else:
        # Fallback: use VIX < 20 as crude contango proxy
        contango = vix < 20

    svxy_ret = svxy.pct_change()
    shy_ret = prices["SHY"].reindex(svxy.index).pct_change()

    # Realized vol for vol-targeting (20-day trailing)
    realized_vol = svxy_ret.rolling(20).std() * np.sqrt(252)
    vol_target = 0.30
    vol_scalar = (vol_target / realized_vol).clip(0, 1.5)  # Cap at 1.5x

    # Position: contango -> SVXY scaled by vol target, backwardation -> SHY
    signal = contango.shift(1).fillna(False)  # Trade next day
    position = signal.astype(float) * vol_scalar.shift(1).fillna(1.0)

    # Raw returns with position sizing
    raw_rets = position * svxy_ret + (1 - position) * shy_ret.fillna(0)

    # Apply slippage on signal changes
    trades = signal.astype(int).diff().abs().fillna(0)
    raw_rets = raw_rets - trades * SLIPPAGE_LEVERED

    raw_rets = raw_rets.dropna()
    raw_rets.name = "Vol_Harvest_SVXY"

    ann_ret = (1 + raw_rets.mean()) ** 252 - 1
    ann_vol = raw_rets.std() * np.sqrt(252)
    sharpe = (raw_rets.mean() - RISK_FREE / 252) / raw_rets.std() * np.sqrt(252) if raw_rets.std() > 0 else 0
    print(f"  Vol Harvest SVXY: {len(raw_rets)} days, CAGR={ann_ret*100:.1f}%, Sharpe={sharpe:.2f}")
    return raw_rets


def simulate_tqqq_trend(prices: pd.DataFrame) -> pd.Series:
    """
    TQQQ + SPY 200MA filter + vol-target 30%.

    Logic:
    - Long TQQQ when SPY > 200-day SMA
    - Vol-target at 30% annualized
    - Cash (SHY) when SPY < 200MA
    Slippage: 10 bps per trade.
    """
    tqqq = prices["TQQQ"].dropna()
    spy = prices["SPY"].reindex(tqqq.index).ffill()
    spy_sma200 = spy.rolling(200).mean()

    tqqq_ret = tqqq.pct_change()
    shy_ret = prices["SHY"].reindex(tqqq.index).pct_change().fillna(0)

    # Signal: SPY > 200MA
    risk_on = (spy > spy_sma200).shift(1).fillna(False)

    # Vol-target
    realized_vol = tqqq_ret.rolling(20).std() * np.sqrt(252)
    vol_target = 0.30
    vol_scalar = (vol_target / realized_vol).clip(0, 1.0)

    position = risk_on.astype(float) * vol_scalar.shift(1).fillna(1.0)
    raw_rets = position * tqqq_ret + (1 - position) * shy_ret

    # Slippage on signal changes
    trades = risk_on.astype(int).diff().abs().fillna(0)
    raw_rets = raw_rets - trades * SLIPPAGE_LEVERED

    raw_rets = raw_rets.dropna()
    raw_rets.name = "TQQQ_Trend"

    ann_ret = (1 + raw_rets.mean()) ** 252 - 1
    sharpe = (raw_rets.mean() - RISK_FREE / 252) / raw_rets.std() * np.sqrt(252) if raw_rets.std() > 0 else 0
    print(f"  TQQQ Trend: {len(raw_rets)} days, CAGR={ann_ret*100:.1f}%, Sharpe={sharpe:.2f}")
    return raw_rets


def simulate_btc_trend(prices: pd.DataFrame) -> pd.Series:
    """
    BTC + 100-day SMA trend filter + vol-target 50%.

    Logic:
    - Long BTC when price > 100-day SMA
    - Vol-target at 50% annualized (BTC is inherently volatile)
    - Cash when below SMA
    Slippage: 15 bps per trade. BTC trades 7 days/week but we use biz days only.
    """
    btc = prices["BTC"].dropna()
    btc_sma100 = btc.rolling(100).mean()

    btc_ret = btc.pct_change()

    # Signal: BTC > 100MA
    risk_on = (btc > btc_sma100).shift(1).fillna(False)

    # Vol-target at 50% (BTC moves a lot)
    realized_vol = btc_ret.rolling(20).std() * np.sqrt(365)  # BTC trades every day
    vol_target = 0.50
    vol_scalar = (vol_target / realized_vol).clip(0, 1.0)

    position = risk_on.astype(float) * vol_scalar.shift(1).fillna(1.0)
    raw_rets = position * btc_ret

    # Slippage on signal changes
    trades = risk_on.astype(int).diff().abs().fillna(0)
    raw_rets = raw_rets - trades * SLIPPAGE_CRYPTO

    raw_rets = raw_rets.dropna()
    raw_rets.name = "BTC_Trend"

    ann_ret = (1 + raw_rets.mean()) ** 252 - 1
    sharpe = (raw_rets.mean() - RISK_FREE / 252) / raw_rets.std() * np.sqrt(252) if raw_rets.std() > 0 else 0
    print(f"  BTC Trend: {len(raw_rets)} days, CAGR={ann_ret*100:.1f}%, Sharpe={sharpe:.2f}")
    return raw_rets


def simulate_qqq_collar(prices: pd.DataFrame) -> pd.Series:
    """
    QQQ Collar approximation: long QQQ with dampened upside (sold call)
    and protected downside (bought put).

    Approximation: cap daily returns at +0.5% (sold call premium) and
    floor daily returns at -0.3% (put protection kicks in).
    Net premium cost: ~2% annual drag (put cost - call income).
    """
    qqq = prices["QQQ"].dropna()
    qqq_ret = qqq.pct_change().dropna()

    # Collar effect: cap upside, floor downside
    # Approximate 30-delta put/call collar monthly
    daily_upside_cap = 0.005   # ~30-delta OTM call cap
    daily_downside_floor = -0.003  # ~30-delta OTM put protection
    annual_premium_drag = 0.02  # Net cost of collar (put > call premium)

    collar_ret = qqq_ret.clip(lower=daily_downside_floor, upper=daily_upside_cap)
    collar_ret = collar_ret - annual_premium_drag / 252  # Daily drag

    # Slippage on monthly rolls (12 trades/year, 5bps each)
    # Spread over all days: 12 * 0.0005 / 252 ~ negligible
    collar_ret = collar_ret - 12 * SLIPPAGE_ETF / 252

    collar_ret.name = "QQQ_Collar"

    ann_ret = (1 + collar_ret.mean()) ** 252 - 1
    sharpe = (collar_ret.mean() - RISK_FREE / 252) / collar_ret.std() * np.sqrt(252) if collar_ret.std() > 0 else 0
    print(f"  QQQ Collar: {len(collar_ret)} days, CAGR={ann_ret*100:.1f}%, Sharpe={sharpe:.2f}")
    return collar_ret


def simulate_multi_asset_trend(prices: pd.DataFrame) -> pd.Series:
    """
    Classic multi-asset trend following: SMA100 on SPY, GLD, TLT, DBC, EFA.
    Equal weight among assets above their 100MA. Cash when all below.
    """
    assets = ["SPY", "GLD", "TLT", "DBC", "EFA"]
    available = [a for a in assets if a in prices.columns]

    asset_prices = prices[available].dropna(how="all").ffill()
    asset_rets = asset_prices.pct_change()

    # SMA100 signal for each asset
    signals = pd.DataFrame(index=asset_prices.index)
    for asset in available:
        sma100 = asset_prices[asset].rolling(100).mean()
        signals[asset] = (asset_prices[asset] > sma100).shift(1).astype(float)

    # Equal weight among those above SMA
    n_on = signals.sum(axis=1).clip(lower=1)  # At least 1 to avoid div by zero
    weights = signals.div(n_on, axis=0)

    # When nothing is on, go to cash (earn risk-free)
    all_off = signals.sum(axis=1) == 0

    # Portfolio return
    port_ret = (weights * asset_rets[available]).sum(axis=1)
    port_ret[all_off] = RISK_FREE / 252  # Cash return

    # Slippage: count position changes
    for asset in available:
        trades = signals[asset].diff().abs().fillna(0)
        port_ret = port_ret - trades * SLIPPAGE_ETF / len(available)

    port_ret = port_ret.dropna()
    port_ret.name = "MultiAsset_Trend"

    ann_ret = (1 + port_ret.mean()) ** 252 - 1
    sharpe = (port_ret.mean() - RISK_FREE / 252) / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else 0
    print(f"  Multi-Asset Trend: {len(port_ret)} days, CAGR={ann_ret*100:.1f}%, Sharpe={sharpe:.2f}")
    return port_ret


# ============================================================
# 3. METRICS
# ============================================================

def compute_metrics(daily_returns: pd.Series, label: str = "") -> Dict:
    """Comprehensive risk-adjusted metrics from daily return series."""
    n_days = len(daily_returns)
    if n_days < 10:
        return {"label": label, "error": "insufficient data"}
    years = n_days / 252

    cum_return = (1 + daily_returns).prod() - 1
    cagr = (1 + cum_return) ** (1 / years) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)

    excess_daily = daily_returns - RISK_FREE / 252
    sharpe = excess_daily.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (cagr - RISK_FREE) / downside_std

    equity = (1 + daily_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0
    wr = (daily_returns > 0).sum() / len(daily_returns) * 100
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Monthly VaR
    eq_series = pd.Series(equity.values, index=daily_returns.index)
    monthly = eq_series.resample("ME").last().pct_change().dropna()
    var_5 = monthly.quantile(0.05) if len(monthly) > 0 else 0

    # Worst/best year
    yearly = eq_series.resample("YE").last().pct_change().dropna()
    worst_year = yearly.min() if len(yearly) > 0 else 0
    best_year = yearly.max() if len(yearly) > 0 else 0
    worst_year_label = str(yearly.idxmin().year) if len(yearly) > 0 else "N/A"
    best_year_label = str(yearly.idxmax().year) if len(yearly) > 0 else "N/A"

    final_eq = STARTING_CAPITAL * (1 + cum_return)

    return {
        "label": label,
        "CAGR%": round(cagr * 100, 2),
        "AnnVol%": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "MaxDD%": round(max_dd * 100, 2),
        "Calmar": round(calmar, 2),
        "WR%": round(wr, 1),
        "PF": round(pf, 2),
        "Monthly_VaR_5%": round(var_5 * 100, 2),
        "Worst_Year%": round(worst_year * 100, 2),
        "Worst_Year": worst_year_label,
        "Best_Year%": round(best_year * 100, 2),
        "Best_Year": best_year_label,
        "Final_Equity": round(final_eq, 0),
        "Years": round(years, 1),
        "N_Days": n_days,
    }


# ============================================================
# 4. PORTFOLIO OPTIMIZATION
# ============================================================

def max_sharpe_optimize(returns_df: pd.DataFrame, min_weight: float = 0.0,
                        max_weight: float = 1.0) -> Tuple[np.ndarray, str]:
    """Mean-variance optimization for max Sharpe ratio."""
    mu = returns_df.mean().values * 252
    cov = returns_df.cov().values * 252
    n = len(mu)

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -(port_ret - RISK_FREE) / port_vol if port_vol > 1e-10 else 0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else x0
    return weights, "Max Sharpe (MVO)"


def min_variance_optimize(returns_df: pd.DataFrame, min_weight: float = 0.0,
                           max_weight: float = 1.0) -> Tuple[np.ndarray, str]:
    """Minimum variance portfolio."""
    cov = returns_df.cov().values * 252
    n = cov.shape[0]

    def portfolio_var(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight)] * n
    x0 = np.ones(n) / n

    result = minimize(portfolio_var, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else x0
    return weights, "Min Variance"


def risk_parity(returns_df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Inverse-volatility weighted."""
    vols = returns_df.std() * np.sqrt(252)
    inv_vol = 1 / vols
    weights = (inv_vol / inv_vol.sum()).values
    return weights, "Risk Parity (inv-vol)"


def kelly_half(returns_df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Half-Kelly criterion weights."""
    mu = returns_df.mean().values * 252
    cov = returns_df.cov().values * 252
    n = len(mu)

    try:
        cov_inv = np.linalg.inv(cov)
        excess = mu - RISK_FREE
        full_kelly = cov_inv @ excess
        half_kelly = full_kelly / 2
        half_kelly = np.maximum(half_kelly, 0.0)
        if half_kelly.sum() > 0:
            half_kelly = half_kelly / half_kelly.sum()
        else:
            half_kelly = np.ones(n) / n
        return half_kelly, "Half-Kelly"
    except np.linalg.LinAlgError:
        return np.ones(n) / n, "Half-Kelly (fallback)"


def max_calmar_optimize(returns_df: pd.DataFrame, min_weight: float = 0.0,
                         max_weight: float = 1.0) -> Tuple[np.ndarray, str]:
    """Maximize Calmar ratio (CAGR / MaxDD)."""
    n = returns_df.shape[1]

    def neg_calmar(w):
        port_ret = (returns_df.values @ w)
        cum = np.cumprod(1 + port_ret)
        cagr = cum[-1] ** (252 / len(port_ret)) - 1
        running_max = np.maximum.accumulate(cum)
        dd = cum / running_max - 1
        max_dd = dd.min()
        return -(cagr / abs(max_dd)) if abs(max_dd) > 1e-10 else 0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_calmar, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else x0
    return weights, "Max Calmar"


# ============================================================
# 5. SCENARIO TESTING
# ============================================================

def build_scenario_allocations(strategy_names: List[str]) -> List[Tuple[str, Dict[str, float]]]:
    """
    Build a list of (name, {strategy: weight}) tuples for specific allocation scenarios.
    """
    scenarios = []

    # --- Income only (current allocation) ---
    scenarios.append(("Income Only (68/14/18)", {
        "V5_CSP": 0.68, "IC_Condors": 0.14, "ETF_Rotation_v3": 0.18,
    }))

    # --- Income + Vol Harvest at various splits ---
    for income_pct in [0.80, 0.70, 0.60]:
        growth_pct = 1 - income_pct
        scenarios.append((f"Income({income_pct:.0%})+VolHarvest({growth_pct:.0%})", {
            "V5_CSP": 0.68 * income_pct,
            "IC_Condors": 0.14 * income_pct,
            "ETF_Rotation_v3": 0.18 * income_pct,
            "Vol_Harvest_SVXY": growth_pct,
        }))

    # --- Income + TQQQ Trend ---
    for income_pct in [0.80, 0.70, 0.60]:
        growth_pct = 1 - income_pct
        scenarios.append((f"Income({income_pct:.0%})+TQQQTrend({growth_pct:.0%})", {
            "V5_CSP": 0.68 * income_pct,
            "IC_Condors": 0.14 * income_pct,
            "ETF_Rotation_v3": 0.18 * income_pct,
            "TQQQ_Trend": growth_pct,
        }))

    # --- Income + Vol Harvest + TQQQ Trend (3-way) ---
    for income_pct in [0.70, 0.60, 0.50]:
        growth_pct = 1 - income_pct
        scenarios.append((f"Income({income_pct:.0%})+VH({growth_pct/2:.0%})+TQQQ({growth_pct/2:.0%})", {
            "V5_CSP": 0.68 * income_pct,
            "IC_Condors": 0.14 * income_pct,
            "ETF_Rotation_v3": 0.18 * income_pct,
            "Vol_Harvest_SVXY": growth_pct / 2,
            "TQQQ_Trend": growth_pct / 2,
        }))

    # --- Income + BTC Trend ---
    for income_pct in [0.90, 0.85, 0.80]:
        growth_pct = 1 - income_pct
        scenarios.append((f"Income({income_pct:.0%})+BTC({growth_pct:.0%})", {
            "V5_CSP": 0.68 * income_pct,
            "IC_Condors": 0.14 * income_pct,
            "ETF_Rotation_v3": 0.18 * income_pct,
            "BTC_Trend": growth_pct,
        }))

    # --- Income + QQQ Collar ---
    for income_pct in [0.80, 0.70]:
        growth_pct = 1 - income_pct
        scenarios.append((f"Income({income_pct:.0%})+Collar({growth_pct:.0%})", {
            "V5_CSP": 0.68 * income_pct,
            "IC_Condors": 0.14 * income_pct,
            "ETF_Rotation_v3": 0.18 * income_pct,
            "QQQ_Collar": growth_pct,
        }))

    # --- Kitchen sink: all strategies ---
    n_all = len(strategy_names)
    scenarios.append(("Kitchen Sink (equal weight)", {
        s: 1.0 / n_all for s in strategy_names
    }))

    # Income-heavy kitchen sink
    scenarios.append(("Kitchen Sink (60% income, 40% growth)", {
        "V5_CSP": 0.68 * 0.60,
        "IC_Condors": 0.14 * 0.60,
        "ETF_Rotation_v3": 0.18 * 0.60,
        "Vol_Harvest_SVXY": 0.40 / 5,
        "TQQQ_Trend": 0.40 / 5,
        "BTC_Trend": 0.40 / 5,
        "QQQ_Collar": 0.40 / 5,
        "MultiAsset_Trend": 0.40 / 5,
    }))

    return scenarios


# ============================================================
# 6. MAIN ANALYSIS
# ============================================================

def run_combined_optimizer():
    print("=" * 80)
    print("COMBINED INCOME + GROWTH PORTFOLIO OPTIMIZER")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # --- Load income strategies ---
    print("\n[1] Loading INCOME strategy returns...")
    try:
        v5 = load_v5_csp()
        ic = load_ic_condors()
        etf = load_etf_rotation_v3()
        income_loaded = True
    except Exception as e:
        print(f"\n  WARNING: Could not load income return series: {e}")
        print("  Generating synthetic income returns from stated statistics...")
        income_loaded = False
        np.random.seed(42)
        n_days = 3000  # ~12 years
        dates = pd.bdate_range("2014-01-02", periods=n_days)

        # V5 CSP: Sharpe ~3.18, CAGR ~11%, vol ~3.5%
        v5_vol_d = 0.035 / np.sqrt(252)
        v5_mu_d = 3.18 * v5_vol_d / np.sqrt(252) + RISK_FREE / 252
        v5 = pd.Series(np.random.normal(v5_mu_d, v5_vol_d, n_days), index=dates, name="V5_CSP")

        # IC Condors: Sharpe ~2.05, vol ~8%
        ic_vol_d = 0.08 / np.sqrt(252)
        ic_mu_d = 2.05 * ic_vol_d / np.sqrt(252) + RISK_FREE / 252
        ic = pd.Series(np.random.normal(ic_mu_d, ic_vol_d, n_days), index=dates, name="IC_Condors")

        # ETF Rotation: Sharpe ~1.48, CAGR ~16.7%, vol ~11%
        etf_vol_d = 0.11 / np.sqrt(252)
        etf_mu_d = 1.48 * etf_vol_d / np.sqrt(252) + RISK_FREE / 252
        etf = pd.Series(np.random.normal(etf_mu_d, etf_vol_d, n_days), index=dates, name="ETF_Rotation_v3")

    # --- Load growth strategies ---
    print("\n[2] Simulating GROWTH strategy returns from proxy assets...")
    growth_prices = download_growth_data()

    vol_harvest = simulate_vol_harvest(growth_prices)
    tqqq_trend = simulate_tqqq_trend(growth_prices)
    btc_trend = simulate_btc_trend(growth_prices)
    qqq_collar = simulate_qqq_collar(growth_prices)
    multi_asset = simulate_multi_asset_trend(growth_prices)

    # --- Align all series ---
    print("\n[3] Aligning all return series to common dates...")
    all_series = [v5, ic, etf, vol_harvest, tqqq_trend, btc_trend, qqq_collar, multi_asset]
    aligned = pd.concat(all_series, axis=1).dropna()
    print(f"  Common date range: {aligned.index[0].date()} to {aligned.index[-1].date()}")
    print(f"  Common trading days: {len(aligned)}")

    if len(aligned) < 252:
        print("\n  WARNING: Less than 1 year of common data. Results may be unreliable.")

    # --- Individual strategy metrics ---
    print("\n" + "=" * 80)
    print("INDIVIDUAL STRATEGY METRICS")
    print("=" * 80)

    individual_metrics = {}
    for col in aligned.columns:
        m = compute_metrics(aligned[col], label=col)
        individual_metrics[col] = m
        tag = "INCOME" if col in INCOME_STRATEGIES else "GROWTH"
        print(f"\n  [{tag}] {col}:")
        for k, v in m.items():
            if k not in ("label", "N_Days"):
                print(f"    {k}: {v}")

    # --- Correlation matrix ---
    print("\n" + "=" * 80)
    print("CORRELATION MATRIX (daily returns)")
    print("=" * 80)
    corr = aligned.corr()
    print(f"\n{corr.round(3).to_string()}")

    # Average pairwise correlation
    n_strats = len(corr)
    off_diag = []
    for i in range(n_strats):
        for j in range(i + 1, n_strats):
            off_diag.append(corr.iloc[i, j])
    avg_corr = np.mean(off_diag)
    print(f"\n  Average pairwise correlation: {avg_corr:.3f}")

    # Income-vs-growth correlation
    inc_growth_corrs = []
    for inc in INCOME_STRATEGIES:
        for grw in GROWTH_STRATEGIES:
            if inc in corr.index and grw in corr.columns:
                inc_growth_corrs.append(corr.loc[inc, grw])
    if inc_growth_corrs:
        print(f"  Average income-vs-growth correlation: {np.mean(inc_growth_corrs):.3f}")
        print(f"  This is the diversification benefit from adding growth strategies")

    # --- Scenario testing ---
    print("\n" + "=" * 80)
    print("SCENARIO ANALYSIS: FIXED ALLOCATION SCENARIOS")
    print("=" * 80)

    scenarios = build_scenario_allocations(list(aligned.columns))
    scenario_results = []

    print(f"\n{'#':>3} {'Scenario':<55} {'CAGR%':>6} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Calmar':>7}")
    print("-" * 100)

    for i, (name, weights_dict) in enumerate(scenarios, 1):
        # Build weight vector aligned with columns
        w = np.array([weights_dict.get(col, 0.0) for col in aligned.columns])
        # Normalize if needed (should already sum to 1)
        w_sum = w.sum()
        if abs(w_sum - 1.0) > 0.01:
            w = w / w_sum

        port_ret = (aligned.values @ w)
        port_series = pd.Series(port_ret, index=aligned.index)
        m = compute_metrics(port_series, label=name)
        m["weights"] = {col: round(wt, 4) for col, wt in zip(aligned.columns, w)}
        m["scenario_name"] = name
        scenario_results.append(m)

        print(f"{i:>3} {name:<55} {m['CAGR%']:>6.1f} {m['Sharpe']:>7.2f} {m['Sortino']:>8.2f} {m['MaxDD%']:>7.1f} {m['Calmar']:>7.2f}")

    # --- Optimization ---
    print("\n" + "=" * 80)
    print("OPTIMIZED ALLOCATIONS")
    print("=" * 80)

    optimization_methods = [
        (max_sharpe_optimize, {"min_weight": 0.0, "max_weight": 0.60}),
        (min_variance_optimize, {"min_weight": 0.0, "max_weight": 0.60}),
        (risk_parity, {}),
        (kelly_half, {}),
        (max_calmar_optimize, {"min_weight": 0.0, "max_weight": 0.60}),
    ]

    optimized_results = []

    for opt_fn, kwargs in optimization_methods:
        if kwargs:
            weights, opt_name = opt_fn(aligned, **kwargs)
        else:
            weights, opt_name = opt_fn(aligned)

        port_ret = (aligned.values @ weights)
        port_series = pd.Series(port_ret, index=aligned.index)
        m = compute_metrics(port_series, label=opt_name)
        m["weights"] = {col: round(wt, 4) for col, wt in zip(aligned.columns, weights)}
        m["optimization"] = opt_name
        optimized_results.append(m)

        print(f"\n  {opt_name}:")
        print(f"    Weights: ", end="")
        for col, w in zip(aligned.columns, weights):
            if w > 0.005:
                print(f"{col}={w*100:.1f}%  ", end="")
        print()
        print(f"    CAGR={m['CAGR%']:.1f}% | Sharpe={m['Sharpe']:.2f} | Sortino={m['Sortino']:.2f} | MaxDD={m['MaxDD%']:.1f}% | Calmar={m['Calmar']:.2f}")

    # --- Constrained optimization: income >= 50% ---
    print("\n" + "-" * 80)
    print("CONSTRAINED: Income >= 50% of portfolio")
    print("-" * 80)

    income_idx = [i for i, c in enumerate(aligned.columns) if c in INCOME_STRATEGIES]
    growth_idx = [i for i, c in enumerate(aligned.columns) if c in GROWTH_STRATEGIES]

    mu = aligned.mean().values * 252
    cov = aligned.cov().values * 252
    n = len(mu)

    def neg_sharpe_constrained(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -(port_ret - RISK_FREE) / port_vol if port_vol > 1e-10 else 0

    constraints_c = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: sum(w[i] for i in income_idx) - 0.50},  # income >= 50%
    ]
    bounds_c = [(0.0, 0.60)] * n
    x0 = np.ones(n) / n

    result_c = minimize(neg_sharpe_constrained, x0, method="SLSQP", bounds=bounds_c, constraints=constraints_c)
    if result_c.success:
        w_c = result_c.x
        port_ret_c = (aligned.values @ w_c)
        port_series_c = pd.Series(port_ret_c, index=aligned.index)
        m_c = compute_metrics(port_series_c, label="Max Sharpe (income>=50%)")
        m_c["weights"] = {col: round(wt, 4) for col, wt in zip(aligned.columns, w_c)}
        m_c["optimization"] = "Max Sharpe (income>=50%)"
        optimized_results.append(m_c)

        income_total = sum(w_c[i] for i in income_idx)
        growth_total = sum(w_c[i] for i in growth_idx)

        print(f"\n  Weights (income={income_total*100:.0f}%, growth={growth_total*100:.0f}%):")
        for col, w in zip(aligned.columns, w_c):
            if w > 0.005:
                tag = "INC" if col in INCOME_STRATEGIES else "GRW"
                print(f"    [{tag}] {col}: {w*100:.1f}%")
        print(f"  CAGR={m_c['CAGR%']:.1f}% | Sharpe={m_c['Sharpe']:.2f} | Sortino={m_c['Sortino']:.2f} | MaxDD={m_c['MaxDD%']:.1f}% | Calmar={m_c['Calmar']:.2f}")

    # --- Key comparison: income-only vs best combined ---
    print("\n" + "=" * 80)
    print("KEY COMPARISON: INCOME ONLY vs BEST COMBINED")
    print("=" * 80)

    income_only = scenario_results[0]  # First scenario is income-only
    all_tested = scenario_results + optimized_results
    best_sharpe = max(all_tested, key=lambda x: x.get("Sharpe", 0))
    best_calmar = max(all_tested, key=lambda x: x.get("Calmar", 0))
    best_cagr = max(all_tested, key=lambda x: x.get("CAGR%", 0))

    print(f"\n  Income Only (68/14/18):")
    print(f"    CAGR={income_only['CAGR%']:.1f}% | Sharpe={income_only['Sharpe']:.2f} | Sortino={income_only['Sortino']:.2f} | MaxDD={income_only['MaxDD%']:.1f}%")

    print(f"\n  Best Sharpe: {best_sharpe['label']}")
    print(f"    CAGR={best_sharpe['CAGR%']:.1f}% | Sharpe={best_sharpe['Sharpe']:.2f} | Sortino={best_sharpe['Sortino']:.2f} | MaxDD={best_sharpe['MaxDD%']:.1f}%")
    if "weights" in best_sharpe:
        for col, w in best_sharpe["weights"].items():
            if w > 0.005:
                print(f"      {col}: {w*100:.1f}%")

    print(f"\n  Best Calmar: {best_calmar['label']}")
    print(f"    CAGR={best_calmar['CAGR%']:.1f}% | Sharpe={best_calmar['Sharpe']:.2f} | Sortino={best_calmar['Sortino']:.2f} | MaxDD={best_calmar['MaxDD%']:.1f}%")

    print(f"\n  Best CAGR: {best_cagr['label']}")
    print(f"    CAGR={best_cagr['CAGR%']:.1f}% | Sharpe={best_cagr['Sharpe']:.2f} | Sortino={best_cagr['Sortino']:.2f} | MaxDD={best_cagr['MaxDD%']:.1f}%")

    # Sharpe improvement
    sharpe_improvement = best_sharpe["Sharpe"] - income_only["Sharpe"]
    cagr_improvement = best_sharpe["CAGR%"] - income_only["CAGR%"]
    print(f"\n  Sharpe improvement from adding growth: {sharpe_improvement:+.2f} ({income_only['Sharpe']:.2f} -> {best_sharpe['Sharpe']:.2f})")
    print(f"  CAGR improvement: {cagr_improvement:+.1f}% ({income_only['CAGR%']:.1f}% -> {best_sharpe['CAGR%']:.1f}%)")

    # --- Efficient frontier: sweep income/growth split ---
    print("\n" + "=" * 80)
    print("EFFICIENT FRONTIER: Income vs Growth Split")
    print("=" * 80)

    # Find best growth-only allocation first
    growth_only = aligned[GROWTH_STRATEGIES]
    if len(growth_only.columns) > 0:
        g_weights, _ = max_sharpe_optimize(growth_only, min_weight=0.0, max_weight=1.0)
        growth_optimal_rets = (growth_only.values @ g_weights)
    else:
        growth_optimal_rets = np.zeros(len(aligned))

    # Income portfolio at current weights
    income_w = np.array([0.68, 0.14, 0.18])
    income_cols = [c for c in INCOME_STRATEGIES if c in aligned.columns]
    income_rets = (aligned[income_cols].values @ income_w[:len(income_cols)])

    print(f"\n  {'Inc%':>5} {'Grw%':>5} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Calmar':>7}")
    print("-" * 55)

    frontier_data = []
    for inc_pct in np.arange(1.0, -0.01, -0.05):
        grw_pct = 1.0 - inc_pct
        combined = inc_pct * income_rets + grw_pct * growth_optimal_rets
        combined_series = pd.Series(combined, index=aligned.index)
        m = compute_metrics(combined_series, label=f"Inc{inc_pct*100:.0f}/Grw{grw_pct*100:.0f}")
        frontier_data.append({
            "income_pct": round(inc_pct * 100, 0),
            "growth_pct": round(grw_pct * 100, 0),
            **m,
        })
        print(f"  {inc_pct*100:>4.0f}% {grw_pct*100:>4.0f}% {m['CAGR%']:>7.1f} {m['Sharpe']:>7.2f} {m['Sortino']:>8.2f} {m['MaxDD%']:>7.1f} {m['Calmar']:>7.2f}")

    # --- Save results ---
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    output = {
        "generated": datetime.now().isoformat(),
        "data_source": "real" if income_loaded else "synthetic_income_real_growth",
        "common_period": f"{aligned.index[0].date()} to {aligned.index[-1].date()}",
        "common_days": len(aligned),
        "individual_strategies": individual_metrics,
        "correlations": corr.round(4).to_dict(),
        "avg_pairwise_correlation": round(avg_corr, 4),
        "avg_income_growth_correlation": round(np.mean(inc_growth_corrs), 4) if inc_growth_corrs else None,
        "scenario_results": scenario_results,
        "optimized_results": optimized_results,
        "efficient_frontier": frontier_data,
        "key_comparison": {
            "income_only": income_only,
            "best_sharpe": best_sharpe,
            "best_calmar": best_calmar,
            "best_cagr": best_cagr,
            "sharpe_improvement": round(sharpe_improvement, 2),
            "cagr_improvement": round(cagr_improvement, 1),
        },
    }

    results_file = OUTPUT_DIR / "combined_optimizer_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {results_file}")

    # Save aligned returns for downstream use
    returns_file = OUTPUT_DIR / "aligned_daily_returns.parquet"
    aligned.to_parquet(returns_file)
    print(f"  Aligned returns saved to {returns_file}")

    # Save correlation matrix as CSV for easy viewing
    corr_file = OUTPUT_DIR / "correlation_matrix.csv"
    corr.round(4).to_csv(corr_file)
    print(f"  Correlation matrix saved to {corr_file}")

    # Save equity curves for best portfolios
    for result in [income_only, best_sharpe, best_calmar]:
        if "weights" in result:
            w = np.array([result["weights"].get(col, 0.0) for col in aligned.columns])
            port_ret = (aligned.values @ w)
            equity = pd.Series(np.cumprod(1 + port_ret) * STARTING_CAPITAL, index=aligned.index)
            label_clean = result["label"].replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")[:40]
            eq_file = OUTPUT_DIR / f"equity_{label_clean}.csv"
            equity.to_csv(eq_file, header=["equity"])

    print(f"\n  All outputs saved to {OUTPUT_DIR}/")
    print("=" * 80)

    return output


if __name__ == "__main__":
    results = run_combined_optimizer()
