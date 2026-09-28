#!/usr/bin/env python3
"""
High-Return Growth Strategy Backtest v2
=======================================
Comprehensive test of leveraged ETF rotation, dual momentum, concentrated
best-of-breed, and crypto+equity tactical strategies.

FIXES vs prior scripts (leveraged_etf_rotation.py, aggressive_growth_strategies.py):
- Vol-target scalar capped at 1.0 (NO extra leverage beyond ETF's built-in 3x)
- NAV computed correctly: NAV_t = NAV_{t-1} * (1 + port_ret_t)
- MaxDD always in [-100%, 0%] range
- Regime gate uses SPY > 200MA (not monthly-return buckets)
- CAGR = (final/initial)^(252/n_days) - 1
- Rebalance cost = 0.01% per trade (bid-ask proxy, no commission on RH)

Author: Claude (HC #696)
Date: 2026-07-14
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import json
import warnings
import time
import sys

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/high_return_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_NAV = 100_000.0
REBAL_COST_BPS = 1.0  # 0.01% per rebalance (bid-ask on liquid ETFs)

# ─── Universe ───
LEVERAGED_ETFS = ["TQQQ", "SOXL", "TECL", "UPRO", "FAS", "TNA", "FNGU"]
CRYPTO = ["BTC-USD", "ETH-USD"]
BENCHMARKS = ["QQQ", "SPY", "SHV"]
ALL_TICKERS = LEVERAGED_ETFS + CRYPTO + BENCHMARKS


def download_data():
    """Download daily adjusted close prices for all tickers."""
    print("Downloading price data...")
    data = yf.download(ALL_TICKERS, start="2010-01-01", end="2026-07-14",
                       auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.ffill().dropna(how="all")
    # Drop tickers with < 1 year of data
    valid = [t for t in prices.columns if prices[t].notna().sum() > 252]
    prices = prices[valid]
    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")
    print(f"Available tickers: {list(prices.columns)}")
    return prices


def compute_spy_regime(prices):
    """SPY > 200-day MA = risk-on (True), else risk-off (False)."""
    spy = prices["SPY"].copy()
    ma200 = spy.rolling(200).mean()
    regime = (spy > ma200).astype(bool)
    return regime


def compute_metrics(nav_series, label="", spy_regime=None):
    """
    Compute comprehensive metrics from a NAV series (not returns).
    nav_series: pd.Series indexed by date, starting at INITIAL_NAV.
    """
    nav = nav_series.dropna()
    if len(nav) < 252:
        return {"label": label, "valid": False}

    daily_ret = nav.pct_change().dropna()
    if len(daily_ret) < 252 or daily_ret.std() == 0:
        return {"label": label, "valid": False}

    n_days = len(daily_ret)
    n_years = n_days / 252.0

    # CAGR
    final = nav.iloc[-1]
    initial = nav.iloc[0]
    if initial <= 0 or final <= 0:
        return {"label": label, "valid": False}
    cagr = (final / initial) ** (1.0 / n_years) - 1.0

    # Sharpe, Sortino
    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside_ret = daily_ret[daily_ret < 0]
    downside_vol = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 10 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0.0

    # Max Drawdown (from NAV, always between -100% and 0%)
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min()
    # Sanity: clamp
    max_dd = max(max_dd, -1.0)
    max_dd = min(max_dd, 0.0)

    calmar = cagr / abs(max_dd) if abs(max_dd) > 0.001 else 0.0

    # Win rate, profit factor
    wr = (daily_ret > 0).mean()
    gains = daily_ret[daily_ret > 0].sum()
    losses = abs(daily_ret[daily_ret < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Regime gate (R1): Sharpe on green days vs red days
    green_sharpe = 0.0
    red_sharpe = 0.0
    regime_gap = 999.0
    r1_pass = False

    if spy_regime is not None:
        # Align regime to daily_ret index
        regime_aligned = spy_regime.reindex(daily_ret.index).fillna(False)
        green_days = daily_ret[regime_aligned]
        red_days = daily_ret[~regime_aligned]

        if len(green_days) > 60 and green_days.std() > 0:
            green_sharpe = green_days.mean() / green_days.std() * np.sqrt(252)
        if len(red_days) > 60 and red_days.std() > 0:
            red_sharpe = red_days.mean() / red_days.std() * np.sqrt(252)

        max_abs = max(abs(green_sharpe), abs(red_sharpe))
        if max_abs > 0:
            regime_gap = abs(green_sharpe - red_sharpe) / max_abs
        else:
            regime_gap = 0.0
        r1_pass = regime_gap <= 0.50

    return {
        "label": label,
        "valid": True,
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "ann_vol": round(ann_vol * 100, 1),
        "wr": round(wr * 100, 1),
        "pf": round(pf, 3),
        "n_years": round(n_years, 1),
        "n_days": n_days,
        "green_sharpe": round(green_sharpe, 3),
        "red_sharpe": round(red_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "r1_pass": r1_pass,
    }


def permutation_test(nav_series, n_trials=100):
    """
    Permutation test: shuffle daily returns (block-bootstrap, 21-day blocks),
    recompute Sharpe. p-value = fraction of shuffled Sharpe >= real Sharpe.
    """
    daily_ret = nav_series.pct_change().dropna().values
    if len(daily_ret) < 63:
        return 1.0  # Not enough data

    real_sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

    # Create 21-day blocks
    block_size = 21
    n_blocks = len(daily_ret) // block_size
    if n_blocks < 3:
        # Fall back to individual shuffle
        beat = 0
        for _ in range(n_trials):
            shuf = np.random.permutation(daily_ret)
            s = shuf.mean() / shuf.std() * np.sqrt(252) if shuf.std() > 0 else 0
            if s >= real_sharpe:
                beat += 1
        return beat / n_trials

    blocks = [daily_ret[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    beat = 0
    for _ in range(n_trials):
        idx = np.random.choice(n_blocks, size=n_blocks, replace=True)
        shuf = np.concatenate([blocks[i] for i in idx])
        s = shuf.mean() / shuf.std() * np.sqrt(252) if shuf.std() > 0 else 0
        if s >= real_sharpe:
            beat += 1
    return beat / n_trials


def build_nav(daily_returns, index):
    """
    Build NAV series from daily portfolio returns.
    daily_returns: list/array of daily returns (same length as index).
    """
    nav = np.empty(len(daily_returns) + 1)
    nav[0] = INITIAL_NAV
    for i, r in enumerate(daily_returns):
        nav[i + 1] = nav[i] * (1.0 + r)
        # NAV can't go below 0
        nav[i + 1] = max(nav[i + 1], 0.0)
    # Return NAV from day 1 onward (day 0 = initial, no trade yet)
    full_index = index
    return pd.Series(nav[1:], index=full_index)


def realized_vol(returns_series, window=21):
    """Trailing realized annualized vol."""
    return returns_series.rolling(window).std() * np.sqrt(252)


# ═══════════════════════════════════════════════════════════════
# STRATEGY A: Leveraged ETF Rotation
# ═══════════════════════════════════════════════════════════════
def strategy_rotation(prices, regime, etfs, lookback_days, top_n,
                      vol_target=0, rebal_freq="monthly", label=""):
    """
    Monthly/weekly rotation among leveraged ETFs by momentum.
    Regime filter: risk-off → SHV.
    Vol-target: scale weight so portfolio vol ≈ target. Cap weight at 1.0.
    """
    returns = prices.pct_change()
    start_idx = max(lookback_days, 200) + 1  # Need 200 for regime
    dates = prices.index[start_idx:]

    daily_rets = []
    current_weights = {}  # etf -> weight
    last_rebal = None
    n_trades = 0

    for date in dates:
        # Determine if rebalance day
        do_rebal = False
        if last_rebal is None:
            do_rebal = True
        elif rebal_freq == "monthly" and date.month != last_rebal.month:
            do_rebal = True
        elif rebal_freq == "weekly" and date.isocalendar()[1] != last_rebal.isocalendar()[1]:
            do_rebal = True

        if do_rebal:
            last_rebal = date

            # Risk-off check
            if not regime.loc[:date].iloc[-1]:
                if current_weights:
                    n_trades += len(current_weights)
                current_weights = {"SHV": 1.0}
            else:
                # Rank ETFs by momentum
                mom = {}
                for etf in etfs:
                    if etf not in prices.columns:
                        continue
                    hist = prices[etf].loc[:date].dropna()
                    if len(hist) < lookback_days + 1:
                        continue
                    p_now = hist.iloc[-1]
                    p_past = hist.iloc[-lookback_days]
                    if p_past > 0:
                        mom[etf] = p_now / p_past - 1.0

                if len(mom) == 0:
                    current_weights = {"SHV": 1.0}
                else:
                    ranked = sorted(mom, key=mom.get, reverse=True)[:top_n]
                    base_weight = 1.0 / len(ranked)

                    # Vol-targeting
                    if vol_target > 0:
                        # Compute realized vol of equal-weighted portfolio
                        port_ret = sum(
                            returns[e].loc[:date].tail(42) * (1.0 / len(ranked))
                            for e in ranked if e in returns.columns
                        )
                        rv = port_ret.std() * np.sqrt(252) if len(port_ret) > 10 else 0.5
                        if rv > 0:
                            scalar = min(vol_target / rv, 1.0)  # Cap at 1.0!
                            scalar = max(scalar, 0.05)
                        else:
                            scalar = 1.0
                    else:
                        scalar = 1.0

                    old_holdings = set(current_weights.keys()) - {"SHV"}
                    new_holdings = set(ranked)
                    n_trades += len(old_holdings.symmetric_difference(new_holdings))

                    current_weights = {e: base_weight * scalar for e in ranked}
                    # Remainder to cash
                    invested = sum(current_weights.values())
                    if invested < 1.0:
                        current_weights["SHV"] = 1.0 - invested

        # Compute daily portfolio return
        port_ret = 0.0
        for etf, w in current_weights.items():
            if etf in returns.columns:
                r = returns[etf].loc[date]
                if not np.isnan(r):
                    port_ret += w * r

        # Rebalance cost (applied on rebal days only)
        if do_rebal and len(current_weights) > 0:
            port_ret -= REBAL_COST_BPS / 10000.0

        daily_rets.append(port_ret)

    nav = build_nav(daily_rets, dates)
    return nav, n_trades


# ═══════════════════════════════════════════════════════════════
# STRATEGY B: Dual Momentum on Leveraged ETFs
# ═══════════════════════════════════════════════════════════════
def strategy_dual_momentum(prices, regime, etfs, lookback_days,
                           top_n=1, vol_target=0, label=""):
    """
    Absolute + Relative momentum.
    Absolute: ETF 1m return > SHV 1m return (or > 0).
    Relative: pick top N by momentum.
    If nothing passes absolute → 100% SHV.
    Monthly rebalance. Regime filter optional.
    """
    returns = prices.pct_change()
    start_idx = max(lookback_days, 200) + 1
    dates = prices.index[start_idx:]

    daily_rets = []
    current_weights = {}
    last_rebal = None
    n_trades = 0

    # SHV return for absolute threshold
    shv_avail = "SHV" in prices.columns

    for date in dates:
        do_rebal = False
        if last_rebal is None:
            do_rebal = True
        elif date.month != last_rebal.month:
            do_rebal = True

        if do_rebal:
            last_rebal = date

            # Regime filter
            if not regime.loc[:date].iloc[-1]:
                if current_weights:
                    n_trades += len(current_weights)
                current_weights = {"SHV": 1.0}
            else:
                # Compute SHV return as absolute threshold
                shv_mom = 0.0
                if shv_avail:
                    shv_hist = prices["SHV"].loc[:date].dropna()
                    if len(shv_hist) > lookback_days:
                        shv_mom = shv_hist.iloc[-1] / shv_hist.iloc[-lookback_days] - 1.0

                # Rank ETFs by momentum, filter by absolute
                mom = {}
                for etf in etfs:
                    if etf not in prices.columns:
                        continue
                    hist = prices[etf].loc[:date].dropna()
                    if len(hist) < lookback_days + 1:
                        continue
                    m = hist.iloc[-1] / hist.iloc[-lookback_days] - 1.0
                    # Absolute filter: must beat SHV
                    if m > shv_mom:
                        mom[etf] = m

                if len(mom) == 0:
                    current_weights = {"SHV": 1.0}
                else:
                    ranked = sorted(mom, key=mom.get, reverse=True)[:top_n]
                    base_w = 1.0 / len(ranked)

                    if vol_target > 0:
                        port_ret_hist = sum(
                            returns[e].loc[:date].tail(42) * (1.0 / len(ranked))
                            for e in ranked if e in returns.columns
                        )
                        rv = port_ret_hist.std() * np.sqrt(252) if len(port_ret_hist) > 10 else 0.5
                        scalar = min(vol_target / rv, 1.0) if rv > 0 else 1.0
                        scalar = max(scalar, 0.05)
                    else:
                        scalar = 1.0

                    old_h = set(current_weights.keys()) - {"SHV"}
                    new_h = set(ranked)
                    n_trades += len(old_h.symmetric_difference(new_h))

                    current_weights = {e: base_w * scalar for e in ranked}
                    invested = sum(current_weights.values())
                    if invested < 1.0:
                        current_weights["SHV"] = 1.0 - invested

        port_ret = 0.0
        for etf, w in current_weights.items():
            if etf in returns.columns:
                r = returns[etf].loc[date]
                if not np.isnan(r):
                    port_ret += w * r

        if do_rebal:
            port_ret -= REBAL_COST_BPS / 10000.0

        daily_rets.append(port_ret)

    nav = build_nav(daily_rets, dates)
    return nav, n_trades


# ═══════════════════════════════════════════════════════════════
# STRATEGY C: Concentrated Best-of-Breed
# ═══════════════════════════════════════════════════════════════
def strategy_concentrated(prices, regime, etfs, lookback_days,
                          vol_target=0.40, label=""):
    """
    Single best-momentum leveraged ETF. Aggressive vol-targeting.
    Regime filter mandatory.
    """
    return strategy_rotation(prices, regime, etfs, lookback_days,
                             top_n=1, vol_target=vol_target,
                             rebal_freq="monthly", label=label)


# ═══════════════════════════════════════════════════════════════
# STRATEGY D: Crypto + Equity Tactical
# ═══════════════════════════════════════════════════════════════
def strategy_crypto_equity(prices, regime, equity_etfs, crypto_tickers,
                           equity_pct=0.70, crypto_pct=0.30,
                           lookback_days=21, label=""):
    """
    70% equity (TQQQ or QQQ by regime), 30% crypto (BTC/ETH by momentum).
    Monthly rebalance.
    """
    returns = prices.pct_change()
    start_idx = max(lookback_days, 200) + 1
    dates = prices.index[start_idx:]

    daily_rets = []
    current_weights = {}
    last_rebal = None
    n_trades = 0

    for date in dates:
        do_rebal = False
        if last_rebal is None:
            do_rebal = True
        elif date.month != last_rebal.month:
            do_rebal = True

        if do_rebal:
            last_rebal = date
            current_weights = {}

            # Equity portion
            risk_on = regime.loc[:date].iloc[-1]
            if risk_on:
                # Pick best equity ETF by momentum
                best_eq = None
                best_mom = -999
                for etf in equity_etfs:
                    if etf not in prices.columns:
                        continue
                    hist = prices[etf].loc[:date].dropna()
                    if len(hist) < lookback_days + 1:
                        continue
                    m = hist.iloc[-1] / hist.iloc[-lookback_days] - 1.0
                    if m > best_mom:
                        best_mom = m
                        best_eq = etf
                if best_eq and best_mom > 0:
                    current_weights[best_eq] = equity_pct
                else:
                    current_weights["SHV"] = equity_pct
            else:
                current_weights["SHV"] = equity_pct

            # Crypto portion
            crypto_avail = [c for c in crypto_tickers if c in prices.columns]
            if crypto_avail:
                best_crypto = None
                best_cmom = -999
                for c in crypto_avail:
                    hist = prices[c].loc[:date].dropna()
                    if len(hist) < lookback_days + 1:
                        continue
                    m = hist.iloc[-1] / hist.iloc[-lookback_days] - 1.0
                    if m > best_cmom:
                        best_cmom = m
                        best_crypto = c
                if best_crypto and best_cmom > 0:
                    current_weights[best_crypto] = crypto_pct
                else:
                    # Crypto negative momentum → cash
                    current_weights["SHV"] = current_weights.get("SHV", 0) + crypto_pct
            else:
                current_weights["SHV"] = current_weights.get("SHV", 0) + crypto_pct

            n_trades += 2  # equity + crypto rebal

        port_ret = 0.0
        for asset, w in current_weights.items():
            if asset in returns.columns:
                r = returns[asset].loc[date]
                if not np.isnan(r):
                    port_ret += w * r

        if do_rebal:
            port_ret -= REBAL_COST_BPS / 10000.0

        daily_rets.append(port_ret)

    nav = build_nav(daily_rets, dates)
    return nav, n_trades


# ═══════════════════════════════════════════════════════════════
# STRATEGY E: TQQQ Enhanced (baseline comparison)
# ═══════════════════════════════════════════════════════════════
def strategy_tqqq_enhanced(prices, regime, vol_target=0, label=""):
    """
    TQQQ + 200MA regime filter. Optional vol-targeting.
    This is our current growth v2 baseline.
    """
    returns = prices.pct_change()
    start_idx = 201  # Need 200 for regime
    dates = prices.index[start_idx:]

    daily_rets = []
    n_trades = 0
    prev_in = None

    for date in dates:
        risk_on = regime.loc[:date].iloc[-1]

        if risk_on:
            if "TQQQ" in returns.columns:
                r = returns["TQQQ"].loc[date]
                if np.isnan(r):
                    r = 0.0

                # Vol-targeting
                if vol_target > 0:
                    rv = returns["TQQQ"].loc[:date].tail(42).std() * np.sqrt(252)
                    if rv > 0:
                        scalar = min(vol_target / rv, 1.0)
                        scalar = max(scalar, 0.05)
                        cash_frac = 1.0 - scalar
                        # Blended return: scalar * TQQQ + cash_frac * SHV
                        shv_r = 0.0
                        if "SHV" in returns.columns:
                            shv_r = returns["SHV"].loc[date]
                            if np.isnan(shv_r):
                                shv_r = 0.0
                        r = scalar * r + cash_frac * shv_r
                    # else: full position
            else:
                r = 0.0
            in_market = True
        else:
            # Risk-off: SHV
            r = 0.0
            if "SHV" in returns.columns:
                r = returns["SHV"].loc[date]
                if np.isnan(r):
                    r = 0.0
            in_market = False

        if prev_in is not None and prev_in != in_market:
            n_trades += 1
        prev_in = in_market

        daily_rets.append(r)

    nav = build_nav(daily_rets, dates)
    return nav, n_trades


# ═══════════════════════════════════════════════════════════════
# MAIN: Run all configurations
# ═══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    prices = download_data()
    regime = compute_spy_regime(prices)

    # Available leveraged ETFs in data
    avail_lev = [e for e in LEVERAGED_ETFS if e in prices.columns]
    avail_crypto = [c for c in CRYPTO if c in prices.columns]
    print(f"\nAvailable leveraged ETFs: {avail_lev}")
    print(f"Available crypto: {avail_crypto}")
    print(f"Regime filter: SPY > 200MA")
    print()

    all_results = []
    all_navs = {}  # label -> nav series

    # ─── Strategy A: Rotation configs ───
    print("=" * 60)
    print("STRATEGY A: Leveraged ETF Rotation")
    print("=" * 60)

    rotation_configs = []
    for lookback in [21, 63, 126]:  # 1m, 3m, 6m
        for top_n in [1, 2, 3]:
            for vt in [0, 0.20, 0.30, 0.40]:
                for freq in ["monthly"]:
                    lb_name = {21: "1m", 63: "3m", 126: "6m"}[lookback]
                    vt_name = f"vt{int(vt*100)}" if vt > 0 else "novt"
                    label = f"A_Rot_{lb_name}_top{top_n}_{vt_name}_{freq}"
                    rotation_configs.append({
                        "lookback": lookback, "top_n": top_n,
                        "vol_target": vt, "freq": freq, "label": label
                    })

    for i, cfg in enumerate(rotation_configs):
        nav, nt = strategy_rotation(
            prices, regime, avail_lev,
            lookback_days=cfg["lookback"], top_n=cfg["top_n"],
            vol_target=cfg["vol_target"], rebal_freq=cfg["freq"],
            label=cfg["label"]
        )
        m = compute_metrics(nav, label=cfg["label"], spy_regime=regime)
        if m["valid"]:
            m["n_trades"] = nt
            all_results.append(m)
            all_navs[cfg["label"]] = nav
        if (i + 1) % 12 == 0:
            print(f"  ... {i+1}/{len(rotation_configs)} rotation configs done")

    print(f"  Rotation: {len([r for r in all_results if r['label'].startswith('A_')])} valid configs")

    # ─── Strategy B: Dual Momentum ───
    print("\n" + "=" * 60)
    print("STRATEGY B: Dual Momentum")
    print("=" * 60)

    dm_configs = []
    for lookback in [21, 42, 63]:
        for top_n in [1, 2]:
            for vt in [0, 0.30]:
                lb_name = {21: "1m", 42: "2m", 63: "3m"}[lookback]
                vt_name = f"vt{int(vt*100)}" if vt > 0 else "novt"
                label = f"B_DualMom_{lb_name}_top{top_n}_{vt_name}"
                dm_configs.append({
                    "lookback": lookback, "top_n": top_n,
                    "vol_target": vt, "label": label
                })

    for cfg in dm_configs:
        nav, nt = strategy_dual_momentum(
            prices, regime, avail_lev,
            lookback_days=cfg["lookback"], top_n=cfg["top_n"],
            vol_target=cfg["vol_target"], label=cfg["label"]
        )
        m = compute_metrics(nav, label=cfg["label"], spy_regime=regime)
        if m["valid"]:
            m["n_trades"] = nt
            all_results.append(m)
            all_navs[cfg["label"]] = nav

    print(f"  Dual Momentum: {len([r for r in all_results if r['label'].startswith('B_')])} valid configs")

    # ─── Strategy C: Concentrated ───
    print("\n" + "=" * 60)
    print("STRATEGY C: Concentrated Best-of-Breed")
    print("=" * 60)

    conc_configs = []
    for lookback in [21, 63, 126]:
        for vt in [0.30, 0.40, 0.50]:
            lb_name = {21: "1m", 63: "3m", 126: "6m"}[lookback]
            label = f"C_Conc_{lb_name}_vt{int(vt*100)}"
            conc_configs.append({
                "lookback": lookback, "vol_target": vt, "label": label
            })

    for cfg in conc_configs:
        nav, nt = strategy_concentrated(
            prices, regime, avail_lev,
            lookback_days=cfg["lookback"], vol_target=cfg["vol_target"],
            label=cfg["label"]
        )
        m = compute_metrics(nav, label=cfg["label"], spy_regime=regime)
        if m["valid"]:
            m["n_trades"] = nt
            all_results.append(m)
            all_navs[cfg["label"]] = nav

    print(f"  Concentrated: {len([r for r in all_results if r['label'].startswith('C_')])} valid configs")

    # ─── Strategy D: Crypto + Equity ───
    print("\n" + "=" * 60)
    print("STRATEGY D: Crypto + Equity Tactical")
    print("=" * 60)

    if avail_crypto:
        d_configs = [
            {"equity_pct": 0.70, "crypto_pct": 0.30, "lookback": 21,
             "equity_etfs": ["TQQQ", "UPRO"], "label": "D_CryptoEq_70_30_1m"},
            {"equity_pct": 0.70, "crypto_pct": 0.30, "lookback": 63,
             "equity_etfs": ["TQQQ", "UPRO"], "label": "D_CryptoEq_70_30_3m"},
            {"equity_pct": 0.50, "crypto_pct": 0.50, "lookback": 21,
             "equity_etfs": ["TQQQ", "UPRO"], "label": "D_CryptoEq_50_50_1m"},
            {"equity_pct": 0.80, "crypto_pct": 0.20, "lookback": 21,
             "equity_etfs": ["TQQQ", "QQQ"], "label": "D_CryptoEq_80_20_1m"},
        ]

        for cfg in d_configs:
            eq_etfs = [e for e in cfg["equity_etfs"] if e in prices.columns]
            nav, nt = strategy_crypto_equity(
                prices, regime, eq_etfs, avail_crypto,
                equity_pct=cfg["equity_pct"], crypto_pct=cfg["crypto_pct"],
                lookback_days=cfg["lookback"], label=cfg["label"]
            )
            m = compute_metrics(nav, label=cfg["label"], spy_regime=regime)
            if m["valid"]:
                m["n_trades"] = nt
                all_results.append(m)
                all_navs[cfg["label"]] = nav

        print(f"  Crypto+Equity: {len([r for r in all_results if r['label'].startswith('D_')])} valid configs")
    else:
        print("  Skipped (no crypto data available)")

    # ─── Strategy E: TQQQ Enhanced (baselines) ───
    print("\n" + "=" * 60)
    print("STRATEGY E: TQQQ Enhanced (baselines)")
    print("=" * 60)

    e_configs = [
        {"vol_target": 0, "label": "E_TQQQ_200MA"},
        {"vol_target": 0.20, "label": "E_TQQQ_200MA_vt20"},
        {"vol_target": 0.30, "label": "E_TQQQ_200MA_vt30"},
        {"vol_target": 0.40, "label": "E_TQQQ_200MA_vt40"},
    ]

    for cfg in e_configs:
        nav, nt = strategy_tqqq_enhanced(
            prices, regime, vol_target=cfg["vol_target"], label=cfg["label"]
        )
        m = compute_metrics(nav, label=cfg["label"], spy_regime=regime)
        if m["valid"]:
            m["n_trades"] = nt
            all_results.append(m)
            all_navs[cfg["label"]] = nav

    print(f"  TQQQ Enhanced: {len([r for r in all_results if r['label'].startswith('E_')])} valid configs")

    # Also add buy-and-hold benchmarks
    print("\n" + "=" * 60)
    print("BENCHMARKS: Buy-and-Hold")
    print("=" * 60)
    for ticker in ["SPY", "QQQ", "TQQQ"]:
        if ticker in prices.columns:
            p = prices[ticker].dropna()
            nav = (p / p.iloc[0]) * INITIAL_NAV
            m = compute_metrics(nav, label=f"BH_{ticker}", spy_regime=regime)
            if m["valid"]:
                m["n_trades"] = 0
                all_results.append(m)
                all_navs[f"BH_{ticker}"] = nav

    # ─── Permutation tests on all valid results ───
    print("\n" + "=" * 60)
    print("PERMUTATION TESTS (100 trials each)")
    print("=" * 60)

    for i, r in enumerate(all_results):
        if not r["valid"]:
            continue
        label = r["label"]
        if label in all_navs:
            pval = permutation_test(all_navs[label], n_trials=100)
            r["perm_p"] = round(pval, 3)
            r["perm_pass"] = pval < 0.05
        else:
            r["perm_p"] = 1.0
            r["perm_pass"] = False
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(all_results)} permutation tests done")

    print(f"  All {len(all_results)} permutation tests complete")

    # ─── Sanity checks ───
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    issues = 0
    for r in all_results:
        if not r["valid"]:
            continue
        if r["max_dd"] < -100 or r["max_dd"] > 0:
            print(f"  BUG: {r['label']} MaxDD={r['max_dd']}% — out of range!")
            issues += 1
        if r["cagr"] > 200:
            print(f"  WARNING: {r['label']} CAGR={r['cagr']}% — suspiciously high")
            issues += 1
        if r["cagr"] < -90:
            print(f"  WARNING: {r['label']} CAGR={r['cagr']}% — suspiciously low")
            issues += 1
    if issues == 0:
        print("  All checks passed — no impossible values detected")

    # ─── Sort by Sharpe and display ───
    valid_results = [r for r in all_results if r.get("valid", False)]
    valid_results.sort(key=lambda x: x.get("sharpe", 0), reverse=True)

    print("\n" + "=" * 80)
    print("RESULTS RANKED BY SHARPE (top 30)")
    print("=" * 80)
    print(f"{'Rank':>4} {'Label':<40} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD%':>7} {'Calmar':>7} {'R1':>4} {'Perm':>5}")
    print("-" * 100)

    for i, r in enumerate(valid_results[:30]):
        r1 = "PASS" if r.get("r1_pass") else "FAIL"
        perm = "PASS" if r.get("perm_pass") else "FAIL"
        print(f"{i+1:>4} {r['label']:<40} {r['cagr']:>7.1f} {r['sharpe']:>7.3f} "
              f"{r['sortino']:>8.3f} {r['max_dd']:>7.1f} {r['calmar']:>7.2f} "
              f"{r1:>4} {perm:>5}")

    # ─── Configs that pass BOTH gates ───
    both_pass = [r for r in valid_results if r.get("r1_pass") and r.get("perm_pass")]
    print(f"\n{'='*80}")
    print(f"CONFIGS PASSING BOTH GATES (regime + permutation): {len(both_pass)}")
    print(f"{'='*80}")
    if both_pass:
        print(f"{'Rank':>4} {'Label':<40} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} "
              f"{'MaxDD%':>7} {'GreenSh':>8} {'RedSh':>7} {'Gap':>6} {'PermP':>6}")
        print("-" * 110)
        for i, r in enumerate(both_pass[:20]):
            print(f"{i+1:>4} {r['label']:<40} {r['cagr']:>7.1f} {r['sharpe']:>7.3f} "
                  f"{r['sortino']:>8.3f} {r['max_dd']:>7.1f} {r.get('green_sharpe',0):>8.3f} "
                  f"{r.get('red_sharpe',0):>7.3f} {r.get('regime_gap',0):>6.3f} "
                  f"{r.get('perm_p',1):>6.3f}")
    else:
        print("  None — all configs fail at least one gate.")

    # ─── Save results ───
    results_path = OUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(valid_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save top 5 NAV series as CSV
    for i, r in enumerate(valid_results[:5]):
        label = r["label"]
        if label in all_navs:
            nav_df = all_navs[label].to_frame(name="NAV")
            nav_df["daily_return"] = nav_df["NAV"].pct_change()
            csv_path = OUT_DIR / f"top{i+1}_{label}_nav.csv"
            nav_df.to_csv(csv_path)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Total configs tested: {len(valid_results)}")
    print(f"Passing both gates: {len(both_pass)}")

    # Final summary
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    if both_pass:
        best = both_pass[0]
        print(f"Best strategy (both gates passed): {best['label']}")
        print(f"  CAGR: {best['cagr']:.1f}%  |  Sharpe: {best['sharpe']:.2f}  |  "
              f"Sortino: {best['sortino']:.2f}  |  MaxDD: {best['max_dd']:.1f}%")
        print(f"  Calmar: {best['calmar']:.2f}  |  Win Rate: {best['wr']:.1f}%  |  "
              f"Profit Factor: {best['pf']:.2f}")
        print(f"  Regime gap: {best['regime_gap']:.3f} (PASS)  |  "
              f"Perm p-value: {best['perm_p']:.3f} (PASS)")
    else:
        best = valid_results[0] if valid_results else None
        if best:
            print(f"Best strategy by Sharpe (but NOT fully validated): {best['label']}")
            print(f"  CAGR: {best['cagr']:.1f}%  |  Sharpe: {best['sharpe']:.2f}  |  "
                  f"MaxDD: {best['max_dd']:.1f}%")


if __name__ == "__main__":
    main()
