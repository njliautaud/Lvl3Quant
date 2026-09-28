#!/usr/bin/env python3
"""
covered_call_income_v1.py — Covered Call Income Optimizer with Risk Overlay

THESIS: Covered calls on quality ETFs, timed by the validated risk overlay.
  - Normal markets: sell slightly OTM calls (delta ~0.30-0.40) for max income
  - Elevated risk: sell deeper OTM calls (delta ~0.20) to keep upside
  - High risk: stop selling calls, buy protective puts

UNIVERSE: SPY, QQQ, XLK, XLF, XLE, XLI, XLV (no leveraged ETFs)

PREMIUM ESTIMATION: Black-Scholes with realized vol as IV proxy.
  *** CAVEAT: Option premiums are ESTIMATED, not from real chains. ***
  Realized vol typically understates IV (variance risk premium), so income
  estimates are CONSERVATIVE relative to real trading.

BACKTEST: 2015-2026, walk-forward sliding 252d vol lookback (HC #0).
COSTS: Commission-free (HC #694).

VALIDATION:
  - HC #428 R1: regime-agnostic OOT (40+ days, all regimes, stratified Sharpe)
  - Permutation test: shuffle overlay timing
  - Income reliability: % months > $3K on $100K capital
  - Compare: buy-hold, CC no overlay, CC with overlay

Output: /home/jupiter/Lvl3Quant/output/covered_call_income_v1/
"""
from __future__ import annotations

import json
import os
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/covered_call_income_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUT_DIR / "backtest.log"),
    ],
)
log = logging.getLogger(__name__)

# ─── UNIVERSE ───────────────────────────────────────────────────────────────
UNIVERSE = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLI", "XLV"]

# Risk overlay tickers
OVERLAY_TICKERS = {
    "vix": "^VIX",
    "vix3m": "^VIX3M",
    "hyg": "HYG",
    "tlt": "TLT",
    "lqd": "LQD",
    "ief": "IEF",
    "tnx": "^TNX",
}

START = "2014-01-01"   # buffer for warm-up
END = "2026-07-15"
BACKTEST_START = "2015-01-02"

STARTING_CAPITAL = 100_000.0
RISK_FREE_RATE = 0.04
CALL_EXPIRY_DAYS = 30       # ~1 month
VOL_LOOKBACK = 252           # sliding window (HC #0)

# Delta targets per risk regime
DELTA_TARGETS = {
    "normal":   0.35,    # slightly OTM, high premium
    "elevated": 0.20,    # deeper OTM, keep upside
    "high":     None,    # no calls, buy protective put instead
}


# ─── BLACK-SCHOLES ──────────────────────────────────────────────────────────

def bs_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * sp_stats.norm.cdf(d1) - K * np.exp(-r * T) * sp_stats.norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * sp_stats.norm.cdf(-d2) - S * sp_stats.norm.cdf(-d1)

def bs_call_delta(S, K, T, r, sigma):
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return float(sp_stats.norm.cdf(d1))

def find_strike_for_delta(S, T, r, sigma, target_delta, option_type="call"):
    """Find strike K such that BS delta ~ target_delta. Binary search."""
    if option_type == "call":
        # Call delta decreases with K, so search upward
        lo, hi = S * 0.80, S * 1.40
        for _ in range(50):
            mid = (lo + hi) / 2
            d = bs_call_delta(S, mid, T, r, sigma)
            if d > target_delta:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2, 2)
    else:
        # Put delta (negative), target_delta should be positive magnitude
        lo, hi = S * 0.60, S * 1.20
        for _ in range(50):
            mid = (lo + hi) / 2
            d1 = bs_d1(S, mid, T, r, sigma)
            put_d = abs(sp_stats.norm.cdf(d1) - 1)
            if put_d < target_delta:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2, 2)


# ─── DATA DOWNLOAD ──────────────────────────────────────────────────────────

def download_data() -> tuple[pd.DataFrame, dict]:
    """Download ETF prices and overlay data. Returns (prices_df, overlay_series_dict)."""
    import yfinance as yf

    log.info("Downloading ETF prices ...")
    all_tickers = UNIVERSE + list(OVERLAY_TICKERS.values())
    raw = yf.download(all_tickers, start=START, end=END, auto_adjust=True, progress=False)

    # Extract close prices
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].copy()

    # ETF prices
    etf_prices = pd.DataFrame()
    for tk in UNIVERSE:
        if tk in prices.columns:
            etf_prices[tk] = prices[tk]
        else:
            log.warning("Missing ticker: %s", tk)

    # Overlay data
    overlay = {}
    for key, ticker in OVERLAY_TICKERS.items():
        col = ticker
        if col in prices.columns:
            overlay[key] = prices[col].dropna()
        else:
            log.warning("Missing overlay ticker: %s (%s)", key, ticker)

    # Also need SPY in overlay for feature computation
    overlay["spy"] = etf_prices["SPY"]

    # Drop rows where ANY ETF price is NaN (e.g. today's incomplete data)
    etf_prices = etf_prices.dropna()

    log.info("ETF prices: %d rows x %d tickers, range %s to %s",
             len(etf_prices), len(etf_prices.columns),
             etf_prices.index[0].strftime("%Y-%m-%d"),
             etf_prices.index[-1].strftime("%Y-%m-%d"))

    return etf_prices, overlay


# ─── RISK OVERLAY (vectorized, from risk_overlay_backtest.py) ───────────────

def compute_risk_overlay(overlay: dict) -> pd.DataFrame:
    """Compute risk overlay signal for every trading day. Returns risk_level and position_scale."""
    spy = overlay["spy"]
    log_ret = np.log(spy / spy.shift(1))

    feats = pd.DataFrame(index=spy.index)

    # SPY momentum
    for w in [5, 21, 63]:
        feats[f"spy_ret_{w}d"] = spy.pct_change(w, fill_method=None)

    # SPY realized vol
    for w in [21, 63]:
        feats[f"spy_rvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # Drawdown
    feats["spy_dd_63d"] = spy / spy.rolling(63).max() - 1
    feats["spy_dd_252d"] = spy / spy.rolling(252).max() - 1

    # Moving averages
    ma200 = spy.rolling(200).mean()
    feats["spy_vs_ma200"] = spy / ma200 - 1
    feats["spy_above_ma50"] = (spy > spy.rolling(50).mean()).astype(int)
    feats["spy_above_ma200"] = (spy > ma200).astype(int)

    # Skew / kurtosis
    for w in [21, 63]:
        feats[f"spy_skew_{w}d"] = log_ret.rolling(w).skew()
        feats[f"spy_kurt_{w}d"] = log_ret.rolling(w).kurt()

    # VIX features
    if "vix" in overlay:
        vix = overlay["vix"].reindex(spy.index, method="ffill")
        feats["vix_level"] = vix
        feats["vix_change_5d"] = vix.pct_change(5, fill_method=None)
        feats["vix_vs_ma20"] = vix / vix.rolling(20).mean() - 1

        if "vix3m" in overlay:
            vix3m = overlay["vix3m"].reindex(spy.index, method="ffill")
            feats["vix_term_ratio"] = vix / (vix3m + 1e-6)
            feats["vix3m_level"] = vix3m

        feats["iv_rv_gap_21d"] = vix / 100 - feats["spy_rvol_21d"]

    # Credit spreads
    if "hyg" in overlay and "tlt" in overlay:
        hyg = overlay["hyg"].reindex(spy.index, method="ffill")
        tlt = overlay["tlt"].reindex(spy.index, method="ffill")
        ratio = hyg / tlt
        feats["hyg_tlt_ratio"] = ratio
        feats["hyg_tlt_change_21d"] = ratio.pct_change(21, fill_method=None)
        feats["hyg_tlt_vs_ma63"] = ratio / ratio.rolling(63).mean() - 1

    if "lqd" in overlay and "tlt" in overlay:
        lqd = overlay["lqd"].reindex(spy.index, method="ffill")
        tlt = overlay["tlt"].reindex(spy.index, method="ffill")
        ratio_ig = lqd / tlt
        feats["lqd_tlt_ratio"] = ratio_ig
        feats["lqd_tlt_change_21d"] = ratio_ig.pct_change(21, fill_method=None)

    if "tlt" in overlay:
        tlt = overlay["tlt"].reindex(spy.index, method="ffill")
        feats["tlt_ret_21d"] = tlt.pct_change(21, fill_method=None)
        feats["tlt_ret_63d"] = tlt.pct_change(63, fill_method=None)

    if "tnx" in overlay:
        tnx = overlay["tnx"].reindex(spy.index, method="ffill")
        feats["yield_10y"] = tnx
        feats["yield_10y_chg_21d"] = tnx.diff(21)

    feats["month"] = pd.to_datetime(feats.index).month
    feats["is_sept_oct"] = feats["month"].isin([9, 10]).astype(int)

    # ── Rules-based scoring (mirrors risk_overlay.py) ──
    score = pd.Series(0.0, index=feats.index)
    max_score = pd.Series(0.0, index=feats.index)

    def add(col, condition, weight):
        has_data = feats[col].notna() if col in feats.columns else pd.Series(False, index=feats.index)
        max_score[has_data] += weight
        score[has_data & condition] += weight

    if "vix_level" in feats.columns:
        add("vix_level", feats["vix_level"] > 25.0, 3.0)
        add("vix_level", feats["vix_level"] > 20.0, 1.5)
    if "vix_term_ratio" in feats.columns:
        add("vix_term_ratio", feats["vix_term_ratio"] > 1.0, 3.0)
        add("vix_term_ratio", feats["vix_term_ratio"] > 1.10, 1.5)
    if "spy_kurt_21d" in feats.columns:
        add("spy_kurt_21d", feats["spy_kurt_21d"] > 3.0, 2.5)
    if "spy_rvol_63d" in feats.columns:
        add("spy_rvol_63d", feats["spy_rvol_63d"] > 0.20, 2.0)
    if "hyg_tlt_vs_ma63" in feats.columns:
        add("hyg_tlt_vs_ma63", feats["hyg_tlt_vs_ma63"] < -0.03, 2.5)
    if "lqd_tlt_change_21d" in feats.columns:
        add("lqd_tlt_change_21d", feats["lqd_tlt_change_21d"] < -0.02, 2.0)
    if "tlt_ret_63d" in feats.columns:
        add("tlt_ret_63d", feats["tlt_ret_63d"] > 0.05, 1.5)
    if "spy_vs_ma200" in feats.columns:
        add("spy_vs_ma200", feats["spy_vs_ma200"] < 0.0, 2.0)
        add("spy_vs_ma200", feats["spy_vs_ma200"] < -0.05, 1.5)
    if "spy_ret_63d" in feats.columns:
        add("spy_ret_63d", feats["spy_ret_63d"] < 0.0, 1.5)
        add("spy_ret_63d", feats["spy_ret_63d"] < -0.05, 1.5)
    if "spy_dd_63d" in feats.columns:
        add("spy_dd_63d", feats["spy_dd_63d"] < -0.03, 1.5)
    if "spy_skew_63d" in feats.columns:
        add("spy_skew_63d", feats["spy_skew_63d"] < -0.5, 1.0)
    max_score += 0.5
    score[feats["is_sept_oct"].astype(bool)] += 0.5

    prob = score / max_score.replace(0, np.nan)
    prob = prob.fillna(0.0).clip(0.0, 1.0)

    risk_level = pd.Series("normal", index=feats.index)
    risk_level[prob >= 0.30] = "elevated"
    risk_level[prob >= 0.50] = "high"

    result = pd.DataFrame({
        "risk_prob": prob,
        "risk_level": risk_level,
    }, index=feats.index)

    return result


# ─── REALIZED VOL (sliding 252d, walk-forward) ─────────────────────────────

def compute_realized_vol(prices: pd.DataFrame, lookback: int = VOL_LOOKBACK) -> pd.DataFrame:
    """Compute annualized realized vol per ticker using sliding window."""
    log_ret = np.log(prices / prices.shift(1))
    rvol = log_ret.rolling(lookback).std() * np.sqrt(252)
    return rvol


# ─── IEF YIELD PROXY FOR RISK-FREE RATE ────────────────────────────────────

def get_risk_free_series(overlay: dict) -> pd.Series:
    """Use 10Y yield as risk-free rate proxy. Fallback to constant 4%."""
    if "tnx" in overlay:
        # TNX is quoted in percent
        rf = overlay["tnx"].reindex(overlay["spy"].index, method="ffill") / 100.0
        rf = rf.clip(0.001, 0.10)
        return rf
    return pd.Series(RISK_FREE_RATE, index=overlay["spy"].index)


# ─── COVERED CALL BACKTEST ENGINE ───────────────────────────────────────────

def backtest_covered_call(
    prices: pd.DataFrame,
    rvol: pd.DataFrame,
    risk_free: pd.Series,
    risk_overlay: Optional[pd.DataFrame] = None,
    capital: float = STARTING_CAPITAL,
    label: str = "CC",
) -> dict:
    """
    Backtest covered call strategy.

    If risk_overlay is None, always sell calls at delta 0.35 (no overlay).
    If risk_overlay is provided, adjust delta per regime.

    Returns dict with daily equity, monthly income, metrics.
    """
    # Align all data
    idx = prices.index[prices.index >= BACKTEST_START]
    n_tickers = len(prices.columns)

    # Equal-weight allocation across tickers
    alloc_per_ticker = capital / n_tickers

    # Track state per ticker
    class Position:
        def __init__(self, ticker, init_alloc):
            self.ticker = ticker
            self.shares = 0
            self.cost_basis = 0.0
            self.call_strike = None
            self.call_premium = 0.0
            self.call_expiry_idx = None
            self.cash = init_alloc
            self.total_premium_collected = 0.0
            self.n_calls_sold = 0
            self.n_assignments = 0

    positions = {tk: Position(tk, alloc_per_ticker) for tk in prices.columns}

    daily_equity = []
    daily_dates = []
    monthly_income = {}  # YYYY-MM -> income
    daily_income = []

    for i, date in enumerate(idx):
        date_str = date.strftime("%Y-%m-%d")
        month_key = date.strftime("%Y-%m")
        day_income = 0.0

        # Get risk regime
        if risk_overlay is not None and date in risk_overlay.index:
            regime = risk_overlay.loc[date, "risk_level"]
        else:
            regime = "normal"

        target_delta = DELTA_TARGETS.get(regime, 0.35)

        for tk in prices.columns:
            pos = positions[tk]
            price = prices.loc[date, tk]
            if pd.isna(price) or price <= 0:
                continue

            vol = rvol.loc[date, tk] if date in rvol.index and tk in rvol.columns else 0.20
            if pd.isna(vol) or vol <= 0:
                vol = 0.20
            rf = risk_free.loc[date] if date in risk_free.index else RISK_FREE_RATE

            T = CALL_EXPIRY_DAYS / 365.0

            # ── Check call expiry ──
            if pos.call_strike is not None and pos.call_expiry_idx is not None:
                if i >= pos.call_expiry_idx:
                    if price >= pos.call_strike:
                        # Called away: sell shares at strike, keep premium
                        proceeds = pos.shares * pos.call_strike
                        pos.cash += proceeds
                        pos.n_assignments += 1
                        pos.shares = 0
                        pos.cost_basis = 0.0
                    # else: expired OTM, keep shares and premium
                    pos.call_strike = None
                    pos.call_premium = 0.0
                    pos.call_expiry_idx = None

# ── HIGH RISK: liquidate shares, go to cash ──
            if regime == "high" and pos.shares > 0 and pos.call_strike is None:
                # Sell all shares -- defensive cash position
                proceeds = pos.shares * price
                pos.cash += proceeds
                pos.shares = 0
                pos.cost_basis = 0.0

            # ── ELEVATED RISK: reduce to 50% position if no active call ──
            if regime == "elevated" and pos.shares > 0 and pos.call_strike is None:
                target_shares = int(pos.shares * 0.5)
                if target_shares < pos.shares:
                    sell_shares = pos.shares - target_shares
                    pos.cash += sell_shares * price
                    pos.shares = target_shares

            # ── Buy shares if not holding and regime allows ──
            if pos.shares == 0 and pos.cash > price and regime != "high":
                shares_to_buy = int(pos.cash / price)
                if shares_to_buy > 0:
                    pos.shares = shares_to_buy
                    pos.cost_basis = price
                    pos.cash -= shares_to_buy * price

            # ── Sell covered call (if we have shares and no active call) ──
            if pos.shares > 0 and pos.call_strike is None and target_delta is not None:
                K = find_strike_for_delta(price, T, rf, vol, target_delta, "call")
                premium = bs_call_price(price, K, T, rf, vol)

                # Scale premium proportionally to shares
                total_premium = premium * pos.shares

                pos.call_strike = K
                pos.call_premium = total_premium
                pos.call_expiry_idx = i + CALL_EXPIRY_DAYS
                pos.cash += total_premium
                pos.total_premium_collected += total_premium
                pos.n_calls_sold += 1
                day_income += total_premium

        # ── Compute daily equity ──
        total_equity = 0.0
        for tk in prices.columns:
            pos = positions[tk]
            price = prices.loc[date, tk]
            if pd.isna(price):
                price = 0.0
            stock_value = pos.shares * price
            total_equity += pos.cash + stock_value

        daily_equity.append(total_equity)
        daily_dates.append(date)
        daily_income.append(day_income)

        if month_key not in monthly_income:
            monthly_income[month_key] = 0.0
        monthly_income[month_key] += day_income

    # Build returns
    eq = pd.Series(daily_equity, index=daily_dates)
    returns = eq.pct_change().fillna(0.0)

    # Aggregate stats
    total_premium_in = sum(p.total_premium_collected for p in positions.values())
    total_calls = sum(p.n_calls_sold for p in positions.values())
    total_assignments = sum(p.n_assignments for p in positions.values())

    monthly_df = pd.Series(monthly_income).sort_index()

    return {
        "label": label,
        "equity": eq,
        "returns": returns,
        "monthly_income": monthly_df,
        "total_premium_collected": total_premium_in,
        "total_calls_sold": total_calls,
        "total_assignments": total_assignments,
        "final_equity": daily_equity[-1] if daily_equity else capital,
    }


# ─── BUY AND HOLD BENCHMARK ────────────────────────────────────────────────

def backtest_buy_hold(
    prices: pd.DataFrame,
    capital: float = STARTING_CAPITAL,
    label: str = "Buy-Hold",
) -> dict:
    """Simple equal-weight buy-and-hold."""
    idx = prices.index[prices.index >= BACKTEST_START]
    n_tickers = len(prices.columns)
    alloc = capital / n_tickers

    # Buy at first available price
    shares = {}
    cash = 0.0
    for tk in prices.columns:
        p0 = prices.loc[idx[0], tk]
        if pd.notna(p0) and p0 > 0:
            s = int(alloc / p0)
            shares[tk] = s
            cash += alloc - s * p0
        else:
            shares[tk] = 0
            cash += alloc

    daily_equity = []
    daily_dates = []
    for date in idx:
        eq = cash
        for tk in prices.columns:
            p = prices.loc[date, tk]
            if pd.notna(p):
                eq += shares[tk] * p
        daily_equity.append(eq)
        daily_dates.append(date)

    eq_s = pd.Series(daily_equity, index=daily_dates)
    returns = eq_s.pct_change().fillna(0.0)

    return {
        "label": label,
        "equity": eq_s,
        "returns": returns,
        "monthly_income": pd.Series(dtype=float),
        "final_equity": daily_equity[-1] if daily_equity else capital,
    }


# ─── PERFORMANCE METRICS ───────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, label: str = "") -> dict:
    """Standard risk-adjusted metrics."""
    r = returns.dropna()
    if len(r) < 50:
        return {"label": label, "error": "insufficient data"}

    ann = 252
    cagr = (1 + r).prod() ** (ann / len(r)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else 0.0
    downside = r[r < 0]
    sortino = r.mean() / downside.std() * np.sqrt(ann) if len(downside) > 0 and downside.std() > 0 else 0.0

    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Monthly return stats
    monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    pf_up = monthly[monthly > 0].sum()
    pf_dn = abs(monthly[monthly < 0].sum())
    profit_factor = pf_up / pf_dn if pf_dn > 0 else float("inf")

    wr = (monthly > 0).mean()

    return {
        "label": label,
        "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "max_dd": round(float(max_dd), 4),
        "calmar": round(float(calmar), 4),
        "profit_factor": round(float(profit_factor), 2),
        "win_rate_monthly": round(float(wr), 4),
        "total_return": round(float((1 + r).prod() - 1), 4),
        "n_days": len(r),
    }


# ─── HC #428 R1: REGIME-AGNOSTIC VALIDATION ────────────────────────────────

def regime_validation(returns: pd.Series, spy_returns: pd.Series, label: str) -> dict:
    """
    HC #428 R1: Stratify returns by regime (green/red/flat day using SPY close-to-close).
    Reject if |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) > 0.50.
    """
    # Classify SPY days
    aligned = pd.DataFrame({"strat": returns, "spy": spy_returns}).dropna()
    if len(aligned) < 50:
        return {"label": label, "error": "insufficient data"}

    green = aligned["spy"] > 0.002    # > +0.2%
    red = aligned["spy"] < -0.002     # < -0.2%
    flat = ~green & ~red

    def sharpe(r):
        if len(r) < 10 or r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))

    s_green = sharpe(aligned.loc[green, "strat"])
    s_red = sharpe(aligned.loc[red, "strat"])
    s_flat = sharpe(aligned.loc[flat, "strat"])

    denom = max(abs(s_green), abs(s_red), 1e-6)
    regime_gap = abs(s_green - s_red) / denom

    passed = regime_gap <= 0.50

    return {
        "label": label,
        "sharpe_green": round(s_green, 4),
        "sharpe_red": round(s_red, 4),
        "sharpe_flat": round(s_flat, 4),
        "n_green": int(green.sum()),
        "n_red": int(red.sum()),
        "n_flat": int(flat.sum()),
        "regime_gap": round(float(regime_gap), 4),
        "hc428_r1_passed": passed,
    }


# ─── PERMUTATION TEST ──────────────────────────────────────────────────────

def permutation_test(
    prices: pd.DataFrame,
    rvol: pd.DataFrame,
    risk_free: pd.Series,
    risk_overlay: pd.DataFrame,
    actual_sharpe: float,
    n_perms: int = 100,
) -> dict:
    """Shuffle overlay timing and compare Sharpe. P-value = fraction of shuffled >= actual."""
    log.info("Running permutation test (%d shuffles) ...", n_perms)

    shuffled_sharpes = []
    overlay_values = risk_overlay["risk_level"].values.copy()

    for p in range(n_perms):
        # Shuffle the risk level assignments
        shuffled = risk_overlay.copy()
        rng = np.random.RandomState(seed=p)
        shuffled_levels = overlay_values.copy()
        rng.shuffle(shuffled_levels)
        shuffled["risk_level"] = shuffled_levels

        result = backtest_covered_call(prices, rvol, risk_free, shuffled,
                                       label=f"perm_{p}")
        m = compute_metrics(result["returns"])
        shuffled_sharpes.append(m.get("sharpe", 0.0))

        if (p + 1) % 25 == 0:
            log.info("  Permutation %d/%d done", p + 1, n_perms)

    shuffled_sharpes = np.array(shuffled_sharpes)
    p_value = (shuffled_sharpes >= actual_sharpe).mean()

    return {
        "actual_sharpe": round(actual_sharpe, 4),
        "shuffled_mean": round(float(shuffled_sharpes.mean()), 4),
        "shuffled_std": round(float(shuffled_sharpes.std()), 4),
        "shuffled_p95": round(float(np.percentile(shuffled_sharpes, 95)), 4),
        "p_value": round(float(p_value), 4),
        "n_perms": n_perms,
        "significant": p_value < 0.05,
    }


# ─── INCOME RELIABILITY ANALYSIS ───────────────────────────────────────────

def income_reliability(monthly_income: pd.Series, capital: float) -> dict:
    """What % of months generate >$3K, >$4K, >$5K on given capital?"""
    if len(monthly_income) == 0:
        return {"error": "no income data"}

    mi = monthly_income[monthly_income.index >= BACKTEST_START[:7]]

    return {
        "n_months": len(mi),
        "mean_monthly": round(float(mi.mean()), 2),
        "median_monthly": round(float(mi.median()), 2),
        "std_monthly": round(float(mi.std()), 2),
        "min_monthly": round(float(mi.min()), 2),
        "max_monthly": round(float(mi.max()), 2),
        "pct_above_3k": round(float((mi > 3000).mean()), 4),
        "pct_above_4k": round(float((mi > 4000).mean()), 4),
        "pct_above_5k": round(float((mi > 5000).mean()), 4),
        "avg_annualized_yield": round(float(mi.sum() / len(mi) * 12 / capital), 4),
    }


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 80)
    log.info("COVERED CALL INCOME OPTIMIZER v1")
    log.info("=" * 80)
    log.info("Universe: %s", ", ".join(UNIVERSE))
    log.info("Capital: $%s", f"{STARTING_CAPITAL:,.0f}")
    log.info("Backtest: %s to %s", BACKTEST_START, END)
    log.info("Walk-forward: sliding %dd vol lookback (HC #0)", VOL_LOOKBACK)
    log.info("Commission-free (HC #694)")
    log.info("*** CAVEAT: Option premiums are ESTIMATED via Black-Scholes ***")
    log.info("")

    # ── 1. Download data ──
    prices, overlay = download_data()

    # ── 2. Compute risk overlay ──
    log.info("Computing risk overlay ...")
    risk_df = compute_risk_overlay(overlay)
    regime_counts = risk_df["risk_level"].value_counts()
    log.info("Regime distribution: %s", dict(regime_counts))

    # ── 3. Realized vol (sliding window) ──
    log.info("Computing realized vol (sliding %dd) ...", VOL_LOOKBACK)
    rvol = compute_realized_vol(prices, VOL_LOOKBACK)

    # ── 4. Risk-free rate ──
    risk_free = get_risk_free_series(overlay)

    # ── 5. Run backtests ──
    log.info("")
    log.info("─── BACKTESTING ───")

    # A) Buy and hold
    log.info("  [1/3] Buy-and-hold ...")
    bh = backtest_buy_hold(prices, STARTING_CAPITAL, "Buy-Hold")

    # B) Covered calls WITHOUT overlay
    log.info("  [2/3] Covered calls (no overlay) ...")
    cc_no = backtest_covered_call(prices, rvol, risk_free, None,
                                  STARTING_CAPITAL, "CC-NoOverlay")

    # C) Covered calls WITH overlay
    log.info("  [3/3] Covered calls (with overlay) ...")
    cc_ov = backtest_covered_call(prices, rvol, risk_free, risk_df,
                                  STARTING_CAPITAL, "CC-WithOverlay")

    # ── 6. Compute metrics ──
    log.info("")
    log.info("─── RESULTS ───")

    results = {}
    for bt in [bh, cc_no, cc_ov]:
        m = compute_metrics(bt["returns"], bt["label"])
        results[bt["label"]] = m
        log.info("  %s:", bt["label"])
        for k, v in m.items():
            if k != "label":
                log.info("    %-20s %s", k, v)
        log.info("")

    # ── 7. Income analysis ──
    log.info("─── INCOME ANALYSIS ───")
    for bt in [cc_no, cc_ov]:
        if "monthly_income" in bt and len(bt["monthly_income"]) > 0:
            inc = income_reliability(bt["monthly_income"], STARTING_CAPITAL)
            results[bt["label"]]["income"] = inc
            log.info("  %s:", bt["label"])
            log.info("    Total premium collected: $%s",
                     f"{bt.get('total_premium_collected', 0):,.0f}")
            log.info("    Calls sold: %d, Assignments: %d",
                     bt.get("total_calls_sold", 0),
                     bt.get("total_assignments", 0))
            for k, v in inc.items():
                log.info("    %-25s %s", k, v)
            log.info("")

    # ── 8. HC #428 R1: Regime validation ──
    log.info("─── HC #428 R1 REGIME VALIDATION ───")
    spy_returns = prices["SPY"].pct_change().fillna(0.0)
    spy_returns = spy_returns[spy_returns.index >= BACKTEST_START]

    for bt in [cc_no, cc_ov]:
        rv = regime_validation(bt["returns"], spy_returns, bt["label"])
        results[bt["label"]]["regime_validation"] = rv
        log.info("  %s:", bt["label"])
        for k, v in rv.items():
            if k != "label":
                log.info("    %-20s %s", k, v)
        passed = "PASSED" if rv.get("hc428_r1_passed") else "FAILED"
        log.info("    >>> HC #428 R1: %s <<<", passed)
        log.info("")

    # ── 9. Permutation test ──
    cc_ov_sharpe = results["CC-WithOverlay"].get("sharpe", 0.0)
    perm = permutation_test(prices, rvol, risk_free, risk_df, cc_ov_sharpe, n_perms=100)
    results["permutation_test"] = perm
    log.info("─── PERMUTATION TEST ───")
    for k, v in perm.items():
        log.info("  %-20s %s", k, v)
    sig = "SIGNIFICANT" if perm.get("significant") else "NOT SIGNIFICANT"
    log.info("  >>> Overlay timing: %s (p=%.4f) <<<", sig, perm.get("p_value", 1.0))
    log.info("")

    # ── 10. Comparison summary ──
    log.info("═" * 80)
    log.info("COMPARISON SUMMARY ($%s capital)", f"{STARTING_CAPITAL:,.0f}")
    log.info("═" * 80)
    header = f"{'Strategy':<25} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'PF':>6} {'WR':>6}"
    log.info(header)
    log.info("─" * 80)
    for key in ["Buy-Hold", "CC-NoOverlay", "CC-WithOverlay"]:
        m = results[key]
        log.info(
            f"{key:<25} {m.get('cagr','')!s:>8} {m.get('sharpe','')!s:>8} "
            f"{m.get('sortino','')!s:>8} {m.get('max_dd','')!s:>8} "
            f"{m.get('profit_factor','')!s:>6} {m.get('win_rate_monthly','')!s:>6}"
        )
    log.info("")

    # Income comparison
    if "income" in results.get("CC-WithOverlay", {}):
        inc = results["CC-WithOverlay"]["income"]
        log.info("INCOME RELIABILITY (CC-WithOverlay):")
        log.info("  Mean monthly income:   $%.0f", inc.get("mean_monthly", 0))
        log.info("  Months > $3K income:   %.1f%%", inc.get("pct_above_3k", 0) * 100)
        log.info("  Months > $4K income:   %.1f%%", inc.get("pct_above_4k", 0) * 100)
        log.info("  Months > $5K income:   %.1f%%", inc.get("pct_above_5k", 0) * 100)
        log.info("  Annualized yield:      %.1f%%", inc.get("avg_annualized_yield", 0) * 100)
    log.info("")

    # ── 11. Save results ──
    # Convert non-serializable items
    save_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save_results[k] = v

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    log.info("Results saved to %s", OUT_DIR / "results.json")

    # Save equity curves
    eq_df = pd.DataFrame({
        "Buy-Hold": bh["equity"],
        "CC-NoOverlay": cc_no["equity"],
        "CC-WithOverlay": cc_ov["equity"],
    })
    eq_df.to_csv(OUT_DIR / "equity_curves.csv")
    log.info("Equity curves saved to %s", OUT_DIR / "equity_curves.csv")

    # Save monthly income
    mi_df = pd.DataFrame({
        "CC-NoOverlay": cc_no.get("monthly_income", pd.Series(dtype=float)),
        "CC-WithOverlay": cc_ov.get("monthly_income", pd.Series(dtype=float)),
    })
    mi_df.to_csv(OUT_DIR / "monthly_income.csv")
    log.info("Monthly income saved to %s", OUT_DIR / "monthly_income.csv")

    # Save risk overlay regime distribution
    regime_by_month = risk_df["risk_level"].resample("ME").apply(
        lambda x: x.value_counts().to_dict()
    )
    with open(OUT_DIR / "regime_distribution.json", "w") as f:
        json.dump({str(k): v for k, v in regime_by_month.items()}, f, indent=2, default=str)

    log.info("")
    log.info("All output saved to %s", OUT_DIR)
    log.info("DONE.")

    return results


if __name__ == "__main__":
    results = main()
