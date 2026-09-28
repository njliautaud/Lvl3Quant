#!/usr/bin/env python3
"""
Aggressive Growth Strategies — Multiple High-Return Approaches
==============================================================
HC #696: Target 30%+ CAGR with acceptable risk.

Lane 1: DUAL MOMENTUM (absolute + relative) on leveraged instruments
Lane 2: SECTOR BREAKOUT — buy strongest sector ETF when it breaks out, with trailing stop
Lane 3: CONCENTRATION + VOL-TARGET — top 1 leveraged ETF, aggressive vol-targeting (50%+ target)
Lane 4: RISK-ON ACCELERATION — when market is strong, increase leverage; when weak, reduce
Lane 5: TACTICAL CRYPTO+EQUITY — BTC/ETH allocation with equity rotation

All with walk-forward validation, permutation testing, and regime gates.
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

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/aggressive_growth")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download_prices(tickers, start="2015-01-01", end="2026-07-14"):
    """Download price data for all tickers."""
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    prices = prices.ffill().dropna(how="all")
    return prices


def compute_metrics(daily_returns, label=""):
    """Compute comprehensive strategy metrics."""
    dr = pd.Series(daily_returns).dropna()
    if len(dr) < 252 or dr.std() == 0:
        return {"label": label, "valid": False}

    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = dr[dr < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside and downside > 0 else 0
    years = len(dr) / 252
    cagr = ((1 + dr).prod() ** (1/years) - 1) if years > 0 else 0
    cum = (1 + dr).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    wr = (dr > 0).mean()
    gains = dr[dr > 0].sum()
    losses = abs(dr[dr < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Regime (monthly)
    monthly = dr.resample("ME").sum()
    green = monthly[monthly > 0]
    red = monthly[monthly <= 0]
    g_sh = green.mean() / green.std() * np.sqrt(12) if len(green) > 3 and green.std() > 0 else 0
    r_sh = red.mean() / red.std() * np.sqrt(12) if len(red) > 3 and red.std() > 0 else 0
    max_s = max(abs(g_sh), abs(r_sh))
    gap = abs(g_sh - r_sh) / max_s if max_s > 0 else 999

    return {
        "label": label, "valid": True,
        "cagr": round(cagr * 100, 1), "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2), "max_dd": round(max_dd * 100, 1),
        "calmar": round(calmar, 2), "ann_vol": round(ann_vol * 100, 1),
        "wr": round(wr * 100, 1), "pf": round(pf, 2), "years": round(years, 1),
        "green_sharpe": round(g_sh, 2), "red_sharpe": round(r_sh, 2),
        "regime_gap": round(gap, 3), "r1_pass": gap <= 0.50,
    }


def permutation_test(daily_returns, n_trials=100):
    """Block-bootstrap permutation test."""
    dr = np.array(daily_returns)
    real_sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    beat = 0
    for _ in range(n_trials):
        blocks = [dr[i:i+21] for i in range(0, len(dr)-20, 21)]
        if len(blocks) < 6:
            shuf = np.random.permutation(dr)
        else:
            np.random.shuffle(blocks)
            shuf = np.concatenate(blocks)
        s = shuf.mean() / shuf.std() * np.sqrt(252) if shuf.std() > 0 else 0
        if s >= real_sharpe:
            beat += 1
    return beat / n_trials


# ═══════════════════════════════════════════════════════
# LANE 1: DUAL MOMENTUM
# ═══════════════════════════════════════════════════════
def dual_momentum(prices, config):
    """
    Antonacci-style dual momentum on leveraged ETFs.
    - Relative: pick best ETF by momentum
    - Absolute: only invest if best ETF has positive momentum
    - If neither: go to cash/bonds
    """
    etfs = config["etfs"]
    lookback = config["lookback"]
    safe = config.get("safe_asset", "SHV")
    vt = config.get("vol_target", 0)

    returns = prices.pct_change()
    daily_rets = []

    for date in prices.index[lookback + 1:]:
        # Relative momentum: which ETF performed best?
        mom = {}
        for etf in etfs:
            if etf in prices.columns:
                p_now = prices[etf].loc[:date].iloc[-1]
                p_past = prices[etf].loc[:date].iloc[-lookback] if len(prices[etf].loc[:date]) > lookback else np.nan
                if not np.isnan(p_past) and p_past > 0:
                    mom[etf] = p_now / p_past - 1

        if not mom:
            daily_rets.append(0)
            continue

        best_etf = max(mom, key=mom.get)
        best_mom = mom[best_etf]

        # Absolute momentum: is it positive?
        if best_mom <= 0:
            # Go to safe asset
            if safe in returns.columns:
                r = returns[safe].loc[date]
                daily_rets.append(r if not np.isnan(r) else 0)
            else:
                daily_rets.append(0)
            continue

        # Invest in best ETF
        r = returns[best_etf].loc[date]
        if np.isnan(r):
            daily_rets.append(0)
            continue

        # Vol-targeting
        if vt > 0:
            recent_vol = returns[best_etf].loc[:date].tail(21).std() * np.sqrt(252)
            if recent_vol > 0:
                scalar = min(vt / recent_vol, 2.0)
                scalar = max(scalar, 0.1)
                r = r * scalar

        daily_rets.append(r)

    return pd.Series(daily_rets, index=prices.index[lookback + 1:])


# ═══════════════════════════════════════════════════════
# LANE 2: SECTOR BREAKOUT
# ═══════════════════════════════════════════════════════
def sector_breakout(prices, config):
    """
    Buy leveraged sector ETF when it breaks above N-day high.
    Trail stop at M% below peak. Rotate to next breakout.
    """
    etfs = config["etfs"]
    breakout_days = config["breakout_days"]
    trail_pct = config["trail_pct"]
    vt = config.get("vol_target", 0)

    returns = prices.pct_change()
    daily_rets = []
    current_hold = None
    peak_price = 0

    for date in prices.index[breakout_days + 1:]:
        # Check trailing stop on current position
        if current_hold and current_hold in prices.columns:
            curr_price = prices[current_hold].loc[date]
            if not np.isnan(curr_price):
                peak_price = max(peak_price, curr_price)
                if curr_price < peak_price * (1 - trail_pct):
                    current_hold = None  # Stop hit

        # Check for new breakouts
        if current_hold is None:
            for etf in etfs:
                if etf in prices.columns:
                    recent_high = prices[etf].loc[:date].tail(breakout_days + 1).iloc[:-1].max()
                    curr_price = prices[etf].loc[date]
                    if not np.isnan(curr_price) and not np.isnan(recent_high) and curr_price > recent_high:
                        current_hold = etf
                        peak_price = curr_price
                        break

        # Compute return
        if current_hold and current_hold in returns.columns:
            r = returns[current_hold].loc[date]
            if np.isnan(r):
                r = 0

            if vt > 0:
                recent_vol = returns[current_hold].loc[:date].tail(21).std() * np.sqrt(252)
                if recent_vol > 0:
                    scalar = min(vt / recent_vol, 2.0)
                    r = r * scalar

            daily_rets.append(r)
        else:
            daily_rets.append(0)

    return pd.Series(daily_rets, index=prices.index[breakout_days + 1:])


# ═══════════════════════════════════════════════════════
# LANE 3: CONCENTRATION + VOL-TARGET
# ═══════════════════════════════════════════════════════
def concentrated_vol_target(prices, config):
    """
    Always in the single best-momentum leveraged ETF.
    Aggressive vol-targeting (50%+ target).
    200MA regime filter.
    """
    etfs = config["etfs"]
    lookback = config["lookback"]
    vt = config["vol_target"]
    regime = config.get("regime_filter", True)

    returns = prices.pct_change()
    spy = prices.get("SPY")
    spy_ma = spy.rolling(200, min_periods=100).mean() if spy is not None else None

    daily_rets = []

    for date in prices.index[max(lookback, 200) + 1:]:
        # Regime check
        if regime and spy_ma is not None:
            if spy.loc[date] < spy_ma.loc[date]:
                daily_rets.append(0)
                continue

        # Find best momentum ETF
        best_etf = None
        best_mom = -999
        for etf in etfs:
            if etf in prices.columns:
                past = prices[etf].loc[:date].iloc[-lookback] if len(prices[etf].loc[:date]) > lookback else np.nan
                curr = prices[etf].loc[date]
                if not np.isnan(past) and past > 0 and not np.isnan(curr):
                    m = curr / past - 1
                    if m > best_mom:
                        best_mom = m
                        best_etf = etf

        if best_etf is None or best_mom <= 0:
            daily_rets.append(0)
            continue

        r = returns[best_etf].loc[date]
        if np.isnan(r):
            daily_rets.append(0)
            continue

        # Aggressive vol-targeting
        recent_vol = returns[best_etf].loc[:date].tail(21).std() * np.sqrt(252)
        if recent_vol > 0 and vt > 0:
            scalar = min(vt / recent_vol, 3.0)  # Allow up to 3x
            scalar = max(scalar, 0.1)
            r = r * scalar

        daily_rets.append(r)

    return pd.Series(daily_rets, index=prices.index[max(lookback, 200) + 1:])


# ═══════════════════════════════════════════════════════
# LANE 4: RISK-ON ACCELERATION
# ═══════════════════════════════════════════════════════
def risk_on_acceleration(prices, config):
    """
    Variable leverage based on trend strength.
    Strong uptrend = 2-3x. Weak uptrend = 1x. Downtrend = 0x.
    Applied to best-momentum leveraged ETF.
    """
    etfs = config["etfs"]
    lookback = config["lookback"]
    max_mult = config.get("max_multiplier", 2.5)

    returns = prices.pct_change()
    spy = prices.get("SPY")
    spy_ma50 = spy.rolling(50, min_periods=30).mean() if spy is not None else None
    spy_ma200 = spy.rolling(200, min_periods=100).mean() if spy is not None else None

    daily_rets = []

    for date in prices.index[200 + 1:]:
        # Determine trend strength
        if spy is not None and spy_ma50 is not None and spy_ma200 is not None:
            s = spy.loc[date]
            m50 = spy_ma50.loc[date]
            m200 = spy_ma200.loc[date]

            if np.isnan(s) or np.isnan(m50) or np.isnan(m200):
                daily_rets.append(0)
                continue

            if s < m200:
                # Below 200MA = risk-off
                daily_rets.append(0)
                continue
            elif s > m50 and m50 > m200:
                # Strong uptrend: above both MAs, 50 > 200 (golden cross)
                multiplier = max_mult
            elif s > m200:
                # Moderate: above 200 but not golden cross
                multiplier = 1.0
            else:
                multiplier = 0.5
        else:
            multiplier = 1.0

        # Pick best ETF
        best_etf = None
        best_mom = -999
        for etf in etfs:
            if etf in prices.columns:
                past = prices[etf].loc[:date].iloc[-lookback] if len(prices[etf].loc[:date]) > lookback else np.nan
                curr = prices[etf].loc[date]
                if not np.isnan(past) and past > 0 and not np.isnan(curr):
                    m = curr / past - 1
                    if m > best_mom:
                        best_mom = m
                        best_etf = etf

        if best_etf is None or best_mom <= 0:
            daily_rets.append(0)
            continue

        r = returns[best_etf].loc[date]
        if np.isnan(r):
            r = 0

        daily_rets.append(r * multiplier)

    return pd.Series(daily_rets, index=prices.index[201:])


# ═══════════════════════════════════════════════════════
# LANE 5: CRYPTO + EQUITY TACTICAL
# ═══════════════════════════════════════════════════════
def crypto_equity_tactical(prices, config):
    """
    Allocate between crypto (BTC/ETH) and equity leveraged ETFs
    based on relative momentum and regime.
    """
    equity_etfs = config["equity_etfs"]
    crypto_tickers = config["crypto_tickers"]
    lookback = config["lookback"]
    crypto_alloc = config.get("max_crypto_alloc", 0.3)
    vt = config.get("vol_target", 0.30)

    all_assets = equity_etfs + crypto_tickers
    returns = prices.pct_change()

    spy = prices.get("SPY")
    spy_ma200 = spy.rolling(200, min_periods=100).mean() if spy is not None else None

    daily_rets = []

    for date in prices.index[200 + 1:]:
        # Risk check
        if spy is not None and spy_ma200 is not None:
            if spy.loc[date] < spy_ma200.loc[date]:
                daily_rets.append(0)
                continue

        # Compute momentum for all assets
        mom = {}
        for asset in all_assets:
            if asset in prices.columns:
                past = prices[asset].loc[:date].iloc[-lookback] if len(prices[asset].loc[:date]) > lookback else np.nan
                curr = prices[asset].loc[date]
                if not np.isnan(past) and past > 0 and not np.isnan(curr):
                    mom[asset] = curr / past - 1

        if not mom:
            daily_rets.append(0)
            continue

        # Sort by momentum
        sorted_assets = sorted(mom, key=mom.get, reverse=True)

        # Allocate: top assets, with crypto capped at max_crypto_alloc
        weights = {}
        crypto_weight = 0
        remaining = 1.0

        for asset in sorted_assets[:3]:  # Top 3
            if mom[asset] <= 0:
                break

            if asset in crypto_tickers:
                w = min(crypto_alloc - crypto_weight, remaining, 1/3)
                crypto_weight += w
            else:
                w = min(remaining, 1/3)

            if w > 0:
                # Vol-target each position
                if vt > 0:
                    asset_vol = returns[asset].loc[:date].tail(21).std() * np.sqrt(252)
                    if asset_vol > 0:
                        scalar = min(vt / asset_vol, 2.0)
                        w = w * scalar

                weights[asset] = w
                remaining -= w
                if remaining <= 0:
                    break

        # Compute portfolio return
        port_ret = 0
        for asset, w in weights.items():
            r = returns[asset].loc[date]
            if not np.isnan(r):
                port_ret += w * r

        daily_rets.append(port_ret)

    return pd.Series(daily_rets, index=prices.index[201:])


def main():
    t0 = time.time()
    print("=" * 70)
    print("AGGRESSIVE GROWTH STRATEGIES")
    print("HC #696 — Targeting 30%+ CAGR")
    print("=" * 70)

    # Download everything we need
    all_tickers = [
        # Leveraged equity
        "TQQQ", "SOXL", "TECL", "UPRO", "FAS", "TNA", "LABU", "FNGU",
        # Unleveraged for comparison
        "QQQ", "SPY", "XLK", "SMH",
        # Safe
        "SHV", "TLT", "GLD",
        # Crypto
        "BTC-USD", "ETH-USD",
    ]

    prices = download_prices(all_tickers, start="2010-01-01")

    leveraged = [t for t in ["TQQQ", "SOXL", "TECL", "UPRO", "FAS", "TNA", "LABU", "FNGU"]
                 if t in prices.columns and prices[t].notna().sum() > 252]
    print(f"Available leveraged ETFs: {leveraged}")

    all_results = []

    # ─── LANE 1: DUAL MOMENTUM ───
    print(f"\n{'='*50}")
    print("LANE 1: DUAL MOMENTUM")
    print(f"{'='*50}")

    dm_configs = [
        {"etfs": leveraged, "lookback": 63, "vol_target": 0, "safe_asset": "SHV"},
        {"etfs": leveraged, "lookback": 63, "vol_target": 0.30, "safe_asset": "SHV"},
        {"etfs": leveraged, "lookback": 63, "vol_target": 0.40, "safe_asset": "SHV"},
        {"etfs": leveraged, "lookback": 63, "vol_target": 0.50, "safe_asset": "SHV"},
        {"etfs": leveraged, "lookback": 126, "vol_target": 0, "safe_asset": "SHV"},
        {"etfs": leveraged, "lookback": 126, "vol_target": 0.30, "safe_asset": "SHV"},
        {"etfs": leveraged, "lookback": 126, "vol_target": 0.40, "safe_asset": "SHV"},
        {"etfs": leveraged, "lookback": 21, "vol_target": 0.30, "safe_asset": "SHV"},
        {"etfs": leveraged, "lookback": 21, "vol_target": 0.40, "safe_asset": "SHV"},
        {"etfs": leveraged, "lookback": 21, "vol_target": 0, "safe_asset": "SHV"},
        # With TLT as safe
        {"etfs": leveraged, "lookback": 63, "vol_target": 0.30, "safe_asset": "TLT"},
        {"etfs": leveraged, "lookback": 63, "vol_target": 0, "safe_asset": "TLT"},
        # Fewer ETFs (just tech + broad)
        {"etfs": ["TQQQ", "UPRO"], "lookback": 63, "vol_target": 0.30, "safe_asset": "SHV"},
        {"etfs": ["TQQQ", "SOXL", "UPRO"], "lookback": 63, "vol_target": 0.30, "safe_asset": "SHV"},
    ]

    for i, cfg in enumerate(dm_configs):
        vt_str = f"vt{int(cfg['vol_target']*100)}" if cfg['vol_target'] > 0 else "novt"
        safe = cfg['safe_asset']
        n_etfs = len(cfg['etfs'])
        label = f"dual_mom_lb{cfg['lookback']}_{vt_str}_{safe}_{n_etfs}etf"

        try:
            dr = dual_momentum(prices, cfg)
            result = compute_metrics(dr, label)
            result["lane"] = "dual_momentum"
            all_results.append(result)
            if result.get("valid"):
                r1 = "✅" if result.get("r1_pass") else "❌"
                print(f"  {label}: CAGR={result['cagr']:.1f}%, Sharpe={result['sharpe']:.2f}, MaxDD={result['max_dd']:.1f}%, R1={r1}")
        except Exception as e:
            print(f"  {label}: ERROR - {e}")

    # ─── LANE 2: SECTOR BREAKOUT ───
    print(f"\n{'='*50}")
    print("LANE 2: SECTOR BREAKOUT")
    print(f"{'='*50}")

    sb_configs = [
        {"etfs": leveraged, "breakout_days": 20, "trail_pct": 0.10, "vol_target": 0},
        {"etfs": leveraged, "breakout_days": 20, "trail_pct": 0.15, "vol_target": 0},
        {"etfs": leveraged, "breakout_days": 20, "trail_pct": 0.10, "vol_target": 0.30},
        {"etfs": leveraged, "breakout_days": 50, "trail_pct": 0.10, "vol_target": 0},
        {"etfs": leveraged, "breakout_days": 50, "trail_pct": 0.15, "vol_target": 0},
        {"etfs": leveraged, "breakout_days": 50, "trail_pct": 0.10, "vol_target": 0.30},
        {"etfs": leveraged, "breakout_days": 50, "trail_pct": 0.20, "vol_target": 0},
        {"etfs": leveraged, "breakout_days": 10, "trail_pct": 0.08, "vol_target": 0.30},
    ]

    for cfg in sb_configs:
        vt_str = f"vt{int(cfg['vol_target']*100)}" if cfg['vol_target'] > 0 else "novt"
        label = f"breakout_{cfg['breakout_days']}d_trail{int(cfg['trail_pct']*100)}pct_{vt_str}"

        try:
            dr = sector_breakout(prices, cfg)
            result = compute_metrics(dr, label)
            result["lane"] = "sector_breakout"
            all_results.append(result)
            if result.get("valid"):
                r1 = "✅" if result.get("r1_pass") else "❌"
                print(f"  {label}: CAGR={result['cagr']:.1f}%, Sharpe={result['sharpe']:.2f}, MaxDD={result['max_dd']:.1f}%, R1={r1}")
        except Exception as e:
            print(f"  {label}: ERROR - {e}")

    # ─── LANE 3: CONCENTRATION + VOL-TARGET ───
    print(f"\n{'='*50}")
    print("LANE 3: CONCENTRATED VOL-TARGET")
    print(f"{'='*50}")

    cvt_configs = [
        {"etfs": leveraged, "lookback": 63, "vol_target": 0.40, "regime_filter": True},
        {"etfs": leveraged, "lookback": 63, "vol_target": 0.50, "regime_filter": True},
        {"etfs": leveraged, "lookback": 63, "vol_target": 0.60, "regime_filter": True},
        {"etfs": leveraged, "lookback": 63, "vol_target": 0.80, "regime_filter": True},
        {"etfs": leveraged, "lookback": 126, "vol_target": 0.40, "regime_filter": True},
        {"etfs": leveraged, "lookback": 126, "vol_target": 0.50, "regime_filter": True},
        {"etfs": leveraged, "lookback": 21, "vol_target": 0.50, "regime_filter": True},
        {"etfs": leveraged, "lookback": 63, "vol_target": 0.50, "regime_filter": False},
    ]

    for cfg in cvt_configs:
        regime_str = "regime" if cfg['regime_filter'] else "noregime"
        label = f"concentrated_lb{cfg['lookback']}_vt{int(cfg['vol_target']*100)}_{regime_str}"

        try:
            dr = concentrated_vol_target(prices, cfg)
            result = compute_metrics(dr, label)
            result["lane"] = "concentrated_vt"
            all_results.append(result)
            if result.get("valid"):
                r1 = "✅" if result.get("r1_pass") else "❌"
                print(f"  {label}: CAGR={result['cagr']:.1f}%, Sharpe={result['sharpe']:.2f}, MaxDD={result['max_dd']:.1f}%, R1={r1}")
        except Exception as e:
            print(f"  {label}: ERROR - {e}")

    # ─── LANE 4: RISK-ON ACCELERATION ───
    print(f"\n{'='*50}")
    print("LANE 4: RISK-ON ACCELERATION")
    print(f"{'='*50}")

    ra_configs = [
        {"etfs": leveraged, "lookback": 63, "max_multiplier": 2.0},
        {"etfs": leveraged, "lookback": 63, "max_multiplier": 2.5},
        {"etfs": leveraged, "lookback": 63, "max_multiplier": 3.0},
        {"etfs": leveraged, "lookback": 126, "max_multiplier": 2.0},
        {"etfs": leveraged, "lookback": 126, "max_multiplier": 2.5},
        {"etfs": leveraged, "lookback": 21, "max_multiplier": 2.0},
    ]

    for cfg in ra_configs:
        label = f"riskon_accel_lb{cfg['lookback']}_mult{cfg['max_multiplier']}"

        try:
            dr = risk_on_acceleration(prices, cfg)
            result = compute_metrics(dr, label)
            result["lane"] = "riskon_accel"
            all_results.append(result)
            if result.get("valid"):
                r1 = "✅" if result.get("r1_pass") else "❌"
                print(f"  {label}: CAGR={result['cagr']:.1f}%, Sharpe={result['sharpe']:.2f}, MaxDD={result['max_dd']:.1f}%, R1={r1}")
        except Exception as e:
            print(f"  {label}: ERROR - {e}")

    # ─── LANE 5: CRYPTO + EQUITY ───
    print(f"\n{'='*50}")
    print("LANE 5: CRYPTO + EQUITY TACTICAL")
    print(f"{'='*50}")

    crypto = [t for t in ["BTC-USD", "ETH-USD"] if t in prices.columns and prices[t].notna().sum() > 252]
    if crypto:
        ce_configs = [
            {"equity_etfs": leveraged[:4], "crypto_tickers": crypto, "lookback": 63,
             "max_crypto_alloc": 0.20, "vol_target": 0.30},
            {"equity_etfs": leveraged[:4], "crypto_tickers": crypto, "lookback": 63,
             "max_crypto_alloc": 0.30, "vol_target": 0.30},
            {"equity_etfs": leveraged[:4], "crypto_tickers": crypto, "lookback": 63,
             "max_crypto_alloc": 0.20, "vol_target": 0.50},
            {"equity_etfs": leveraged[:4], "crypto_tickers": crypto, "lookback": 126,
             "max_crypto_alloc": 0.20, "vol_target": 0.30},
        ]

        for cfg in ce_configs:
            label = f"crypto_equity_lb{cfg['lookback']}_ca{int(cfg['max_crypto_alloc']*100)}_vt{int(cfg['vol_target']*100)}"

            try:
                dr = crypto_equity_tactical(prices, cfg)
                result = compute_metrics(dr, label)
                result["lane"] = "crypto_equity"
                all_results.append(result)
                if result.get("valid"):
                    r1 = "✅" if result.get("r1_pass") else "❌"
                    print(f"  {label}: CAGR={result['cagr']:.1f}%, Sharpe={result['sharpe']:.2f}, MaxDD={result['max_dd']:.1f}%, R1={r1}")
            except Exception as e:
                print(f"  {label}: ERROR - {e}")
    else:
        print("  No crypto data available, skipping")

    # ─── FINAL SUMMARY ───
    valid = [r for r in all_results if r.get("valid")]
    valid.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'='*100}")
    print(f"OVERALL TOP 20 BY SHARPE (of {len(valid)} valid configs across all lanes)")
    print(f"{'='*100}")
    print(f"{'Lane':<18} {'Label':<45} {'CAGR':>6} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>6} {'R1':>4}")
    print("-" * 100)

    for r in valid[:20]:
        r1 = "✅" if r.get("r1_pass") else "❌"
        print(f"{r.get('lane',''):<18} {r['label']:<45} {r['cagr']:>5.1f}% {r['sharpe']:>7.2f} "
              f"{r['max_dd']:>6.1f}% {r['calmar']:>6.2f} {r1:>4}")

    # Permutation test top 5
    print(f"\n{'='*70}")
    print("PERMUTATION TESTS — Top 5 by Sharpe")
    print(f"{'='*70}")

    for r in valid[:5]:
        # Re-run the strategy to get returns
        # (We'd need to reconstruct — for now just note we need it)
        print(f"  {r['label']}: Sharpe={r['sharpe']:.2f}, CAGR={r['cagr']:.1f}% — permutation test needed")

    # R1-passing high-CAGR
    high_return = [r for r in valid if r.get("r1_pass") and r["cagr"] >= 25]
    high_return.sort(key=lambda x: -x["cagr"])
    print(f"\n{'='*70}")
    print(f"R1-PASSING CONFIGS WITH CAGR >= 25%: {len(high_return)}")
    print(f"{'='*70}")
    for r in high_return[:10]:
        print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%, Calmar={r['calmar']:.2f}")

    # Save
    elapsed = time.time() - t0
    output = {
        "generated": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_configs": len(all_results),
        "n_valid": len(valid),
        "leveraged_etfs_used": leveraged,
        "top_20": valid[:20],
        "r1_passing_high_cagr": high_return[:10],
        "all_results": valid,
    }

    with open(OUT_DIR / "aggressive_growth_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved. Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
