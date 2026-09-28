#!/usr/bin/env python3
"""
pmcc_income_v1.py — Poor Man's Covered Call (PMCC) / LEAPS-Based Income Strategy

THESIS: Buy deep-ITM LEAPS calls (70-delta, ~365 DTE) as stock substitute,
  then sell short-term OTM calls (~30-delta, ~30 DTE) against them monthly.
  Capital efficient covered call — costs 10-15% of owning shares.

  Normal markets: sell 30-delta short calls
  Elevated risk: sell closer-to-ATM (25-delta) for more premium / protection
  High risk: reduce LEAPS positions by 50%, stop selling short calls on reduced portion

UNIVERSE: SPY, XLK, XLF, XLV, XLI, XLE

PREMIUM ESTIMATION: Black-Scholes with realized vol as IV proxy.
  *** CAVEAT: Option premiums are ESTIMATED, not from real chains. ***
  Realized vol typically understates IV (variance risk premium), so income
  estimates are CONSERVATIVE relative to real trading.

BACKTEST: 2012-2026, monthly rebalance, sliding 252d vol lookback (HC #0).
COSTS: Commission-free equity ETFs (HC #694). Options: $0.65/contract.

VALIDATION:
  - HC #428 R1: regime-agnostic OOT (40+ days, all regimes, stratified Sharpe)
  - Permutation test: shuffle overlay timing (500 shuffles)
  - Income reliability: % months > $3K on $100K capital
  - Compare: PMCC vs regular covered call vs buy-hold vs 60/40
  - Capital efficiency ratio
  - Crisis periods analysis

Output: /home/jupiter/Lvl3Quant/output/pmcc_income_v1/
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

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/pmcc_income_v1")
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

# ─── UNIVERSE ─────────────────────────────────────────────────────────────
UNIVERSE = ["SPY", "XLK", "XLF", "XLV", "XLI", "XLE"]

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

START = "2011-01-01"   # buffer for warm-up (252d vol lookback)
END = "2026-07-15"
BACKTEST_START = "2012-01-03"

STARTING_CAPITAL = 100_000.0
RISK_FREE_RATE = 0.04
SHORT_CALL_EXPIRY_DAYS = 30      # ~1 month calendar
LEAPS_TARGET_DTE = 365           # ~1 year LEAPS
LEAPS_ROLL_DTE = 60              # roll when DTE < 60
VOL_LOOKBACK = 252               # sliding window (HC #0)
MAX_ALLOC_PER_ETF = 0.25         # max 25% per ETF
LEAPS_CAPITAL_PCT = 0.80         # 80% of capital to LEAPS
OPTION_COMMISSION = 0.65         # per contract

# Delta targets
LEAPS_DELTA_TARGET = 0.70        # deep ITM LEAPS
SHORT_CALL_DELTA = {
    "normal":   0.30,             # standard 30-delta OTM
    "elevated": 0.25,             # closer to ATM, more premium
    "high":     None,             # no short calls on reduced position
}


# ─── BLACK-SCHOLES ──────────────────────────────────────────────────────────

def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
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

def bs_call_delta(S, K, T, r, sigma):
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return float(sp_stats.norm.cdf(d1))

def find_strike_for_delta(S, T, r, sigma, target_delta):
    """Find call strike K such that BS delta ~ target_delta. Binary search."""
    lo, hi = S * 0.50, S * 1.60
    for _ in range(60):
        mid = (lo + hi) / 2
        d = bs_call_delta(S, mid, T, r, sigma)
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


# ─── DATA DOWNLOAD ──────────────────────────────────────────────────────────

def download_data() -> tuple[pd.DataFrame, dict]:
    """Download ETF prices and overlay data."""
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
            log.warning("Missing ticker: %s — excluded from universe", tk)

    # Drop rows where ALL ETF prices are NaN
    all_nan_mask = etf_prices.isna().all(axis=1)
    n_dropped = all_nan_mask.sum()
    if n_dropped > 0:
        log.info("Dropping %d rows with all-NaN prices", n_dropped)
        etf_prices = etf_prices[~all_nan_mask]

    etf_prices = etf_prices.ffill(limit=3)

    # Overlay data
    overlay = {}
    for key, ticker in OVERLAY_TICKERS.items():
        col = ticker
        if col in prices.columns:
            overlay[key] = prices[col].dropna()
        else:
            log.warning("Missing overlay ticker: %s (%s)", key, ticker)

    overlay["spy"] = etf_prices["SPY"]

    log.info("ETF prices: %d rows x %d tickers, range %s to %s",
             len(etf_prices), len(etf_prices.columns),
             etf_prices.index[0].strftime("%Y-%m-%d"),
             etf_prices.index[-1].strftime("%Y-%m-%d"))

    return etf_prices, overlay


# ─── RISK OVERLAY (vectorized) ────────────────────────────────────────────

def compute_risk_overlay(overlay: dict) -> pd.DataFrame:
    """Compute risk overlay signal for every trading day."""
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

    # ── Rules-based scoring ──
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


# ─── REALIZED VOL ──────────────────────────────────────────────────────────

def compute_realized_vol(prices: pd.DataFrame, lookback: int = VOL_LOOKBACK) -> pd.DataFrame:
    """Compute annualized realized vol per ticker using sliding window."""
    log_ret = np.log(prices / prices.shift(1))
    rvol = log_ret.rolling(lookback).std() * np.sqrt(252)
    return rvol


def get_risk_free_series(overlay: dict) -> pd.Series:
    """Use 10Y yield as risk-free rate proxy. Fallback to constant 4%."""
    if "tnx" in overlay:
        rf = overlay["tnx"].reindex(overlay["spy"].index, method="ffill") / 100.0
        rf = rf.clip(0.001, 0.10)
        return rf
    return pd.Series(RISK_FREE_RATE, index=overlay["spy"].index)


# ─── PMCC BACKTEST ENGINE ──────────────────────────────────────────────────

def backtest_pmcc(
    prices: pd.DataFrame,
    rvol: pd.DataFrame,
    risk_free: pd.Series,
    risk_overlay: Optional[pd.DataFrame] = None,
    capital: float = STARTING_CAPITAL,
    label: str = "PMCC",
) -> dict:
    """
    Backtest Poor Man's Covered Call strategy.

    For each ETF:
      1. Buy a deep-ITM LEAPS call (70-delta, ~365 DTE)
      2. Sell short-term OTM calls (30-delta, ~30 DTE) against LEAPS
      3. Roll LEAPS when DTE < 60
      4. Close short call at 50% profit or expiry
      5. If short call is ITM at expiry: close both, reopen

    Position sizing:
      - Allocate capital to LEAPS (costs ~10-15% of shares)
      - Each LEAPS contract controls 100 shares
      - Sell 1 short call per LEAPS contract
    """
    idx = prices.index[prices.index >= BACKTEST_START]
    tickers = list(prices.columns)
    n_tickers = len(tickers)

    # Max allocation per ETF = min(equal weight, 25% cap)
    alloc_per_ticker = min(capital * LEAPS_CAPITAL_PCT / n_tickers, capital * MAX_ALLOC_PER_ETF)
    cash_reserve = capital - alloc_per_ticker * n_tickers

    class LEAPSPosition:
        """Tracks one ETF's PMCC position."""
        def __init__(self, ticker, init_alloc):
            self.ticker = ticker
            self.alloc = init_alloc
            self.init_alloc = init_alloc  # never changes — for position size cap
            # LEAPS state
            self.leaps_strike = None
            self.leaps_expiry_date = None
            self.leaps_cost = 0.0       # what we paid per share for LEAPS
            self.n_contracts = 0         # number of LEAPS contracts
            self.active = False
            # Short call state
            self.short_strike = None
            self.short_expiry_date = None
            self.short_premium = 0.0    # per share premium received
            self.short_active = False
            # Tracking
            self.cash = init_alloc
            self.total_premium_collected = 0.0
            self.total_leaps_pnl = 0.0
            self.n_short_calls_sold = 0
            self.n_assignments = 0
            self.n_leaps_rolls = 0
            self.total_commissions = 0.0
            # For reduced-risk mode
            self.risk_reduced = False

    positions = {tk: LEAPSPosition(tk, alloc_per_ticker) for tk in tickers}

    daily_equity = []
    monthly_income = []
    current_month = None
    month_income = 0.0
    month_start_equity = capital

    for i, date in enumerate(idx):
        # Get risk level
        if risk_overlay is not None and date in risk_overlay.index:
            rl = risk_overlay.loc[date, "risk_level"]
        else:
            rl = "normal"

        total_equity = cash_reserve

        for tk in tickers:
            pos = positions[tk]
            price = prices.loc[date, tk] if date in prices.index else np.nan
            if np.isnan(price):
                total_equity += pos.cash
                continue

            vol = rvol.loc[date, tk] if date in rvol.index else np.nan
            if np.isnan(vol) or vol <= 0.01:
                vol = 0.20  # fallback vol

            rf = risk_free.get(date, RISK_FREE_RATE)

            # ── LEAPS Management ──

            # Open LEAPS if not active and we have cash
            if not pos.active and pos.cash > 0:
                T_leaps = LEAPS_TARGET_DTE / 365.0
                leaps_strike = find_strike_for_delta(price, T_leaps, rf, vol, LEAPS_DELTA_TARGET)
                leaps_price = bs_call_price(price, leaps_strike, T_leaps, rf, vol)

                if leaps_price > 0:
                    # How many contracts can we buy? Cap to initial allocation
                    cost_per_contract = leaps_price * 100 + OPTION_COMMISSION
                    n_affordable = int(pos.cash / cost_per_contract)
                    n_from_alloc = max(1, int(pos.init_alloc / cost_per_contract))
                    n_contracts = min(n_affordable, n_from_alloc)
                    if n_contracts >= 1:
                        pos.n_contracts = n_contracts
                        pos.leaps_strike = leaps_strike
                        pos.leaps_cost = leaps_price
                        pos.leaps_expiry_date = date + timedelta(days=LEAPS_TARGET_DTE)
                        total_cost = n_contracts * cost_per_contract
                        pos.cash -= total_cost
                        pos.total_commissions += n_contracts * OPTION_COMMISSION
                        pos.active = True

            # Roll LEAPS if DTE < 60
            if pos.active and pos.leaps_expiry_date is not None:
                days_left = (pos.leaps_expiry_date - date).days
                if days_left <= LEAPS_ROLL_DTE:
                    # Close old LEAPS at current value
                    T_old = max(days_left / 365.0, 1/365.0)
                    old_value = bs_call_price(price, pos.leaps_strike, T_old, rf, vol)
                    proceeds = old_value * 100 * pos.n_contracts
                    pos.total_leaps_pnl += (old_value - pos.leaps_cost) * 100 * pos.n_contracts
                    pos.cash += proceeds - pos.n_contracts * OPTION_COMMISSION
                    pos.total_commissions += pos.n_contracts * OPTION_COMMISSION

                    # Open new LEAPS (cap to initial allocation size)
                    T_new = LEAPS_TARGET_DTE / 365.0
                    new_strike = find_strike_for_delta(price, T_new, rf, vol, LEAPS_DELTA_TARGET)
                    new_price = bs_call_price(price, new_strike, T_new, rf, vol)
                    cost_per_contract = new_price * 100 + OPTION_COMMISSION
                    n_affordable = int(pos.cash / cost_per_contract)
                    n_from_alloc = max(1, int(pos.init_alloc / cost_per_contract))
                    n_contracts = min(n_affordable, n_from_alloc)
                    if n_contracts >= 1:
                        pos.n_contracts = n_contracts
                        pos.leaps_strike = new_strike
                        pos.leaps_cost = new_price
                        pos.leaps_expiry_date = date + timedelta(days=LEAPS_TARGET_DTE)
                        pos.cash -= n_contracts * cost_per_contract
                        pos.total_commissions += n_contracts * OPTION_COMMISSION
                        pos.n_leaps_rolls += 1
                    else:
                        # Can't afford new LEAPS, go to cash
                        pos.active = False
                        pos.n_contracts = 0

            # ── Risk reduction: in "high" regime, halve LEAPS exposure ──
            if rl == "high" and pos.active and not pos.risk_reduced:
                # Close half the contracts
                n_close = pos.n_contracts // 2
                if n_close >= 1:
                    days_left = max((pos.leaps_expiry_date - date).days, 1)
                    T_cur = days_left / 365.0
                    cur_value = bs_call_price(price, pos.leaps_strike, T_cur, rf, vol)
                    proceeds = cur_value * 100 * n_close
                    pos.total_leaps_pnl += (cur_value - pos.leaps_cost) * 100 * n_close
                    pos.cash += proceeds - n_close * OPTION_COMMISSION
                    pos.total_commissions += n_close * OPTION_COMMISSION
                    pos.n_contracts -= n_close
                    pos.risk_reduced = True

            # Restore if risk drops back to normal
            if rl == "normal" and pos.risk_reduced:
                pos.risk_reduced = False
                # Will re-open on next available cycle if cash allows

            # ── Short Call Management ──

            # Check short call expiry/profit-take
            if pos.short_active and pos.short_expiry_date is not None:
                days_left_sc = (pos.short_expiry_date - date).days

                if days_left_sc <= 0:
                    # Expiry day
                    if price > pos.short_strike:
                        # Short call is ITM — "assignment"
                        # Close LEAPS at profit, realize gain, reopen everything
                        days_left_leaps = max((pos.leaps_expiry_date - date).days, 1)
                        T_leaps = days_left_leaps / 365.0
                        leaps_value = bs_call_price(price, pos.leaps_strike, T_leaps, rf, vol)
                        proceeds = leaps_value * 100 * pos.n_contracts
                        pos.total_leaps_pnl += (leaps_value - pos.leaps_cost) * 100 * pos.n_contracts
                        pos.cash += proceeds - pos.n_contracts * OPTION_COMMISSION
                        pos.total_commissions += pos.n_contracts * OPTION_COMMISSION
                        pos.active = False
                        pos.n_contracts = 0
                        pos.n_assignments += 1
                    # Short call expires — premium fully captured (already booked)
                    pos.short_active = False
                    pos.short_strike = None
                    pos.short_expiry_date = None

                else:
                    # Check 50% profit take
                    T_sc = days_left_sc / 365.0
                    sc_value = bs_call_price(price, pos.short_strike, T_sc, rf, vol)
                    if sc_value <= pos.short_premium * 0.50:
                        # Buy back at 50% profit
                        buyback_cost = sc_value * 100 * pos.n_contracts
                        pos.cash -= buyback_cost + pos.n_contracts * OPTION_COMMISSION
                        pos.total_commissions += pos.n_contracts * OPTION_COMMISSION
                        # Net premium = original - buyback (original already booked, so add back buyback)
                        # Actually: premium was added to cash when sold. Now we subtract buyback.
                        pos.short_active = False
                        pos.short_strike = None
                        pos.short_expiry_date = None

            # Sell new short call if not active and LEAPS active
            target_delta = SHORT_CALL_DELTA.get(rl, 0.30)
            if (pos.active and not pos.short_active and target_delta is not None
                    and pos.n_contracts > 0):
                T_sc = SHORT_CALL_EXPIRY_DAYS / 365.0
                sc_strike = find_strike_for_delta(price, T_sc, rf, vol, target_delta)
                # Only sell if strike > LEAPS strike (proper diagonal)
                if sc_strike > pos.leaps_strike:
                    sc_premium = bs_call_price(price, sc_strike, T_sc, rf, vol)
                    if sc_premium > 0.05:  # minimum premium filter
                        pos.short_strike = sc_strike
                        pos.short_premium = sc_premium
                        pos.short_expiry_date = date + timedelta(days=SHORT_CALL_EXPIRY_DAYS)
                        pos.short_active = True
                        # Collect premium
                        premium_received = sc_premium * 100 * pos.n_contracts
                        pos.cash += premium_received - pos.n_contracts * OPTION_COMMISSION
                        pos.total_commissions += pos.n_contracts * OPTION_COMMISSION
                        pos.total_premium_collected += premium_received
                        pos.n_short_calls_sold += 1

            # ── Mark to Market ──
            if pos.active and pos.n_contracts > 0:
                days_left_leaps = max((pos.leaps_expiry_date - date).days, 1)
                T_leaps = days_left_leaps / 365.0
                leaps_mtm = bs_call_price(price, pos.leaps_strike, T_leaps, rf, vol)
                leaps_equity = leaps_mtm * 100 * pos.n_contracts

                short_liability = 0.0
                if pos.short_active and pos.short_expiry_date is not None:
                    days_left_sc = max((pos.short_expiry_date - date).days, 1)
                    T_sc = days_left_sc / 365.0
                    sc_mtm = bs_call_price(price, pos.short_strike, T_sc, rf, vol)
                    short_liability = sc_mtm * 100 * pos.n_contracts

                total_equity += pos.cash + leaps_equity - short_liability
            else:
                total_equity += pos.cash

        daily_equity.append({"date": date, "equity": total_equity})

        # Track monthly income
        dt_month = (date.year, date.month)
        if current_month is None:
            current_month = dt_month
            month_start_equity = total_equity
        elif dt_month != current_month:
            # Month changed — record income
            month_return = total_equity - month_start_equity
            monthly_income.append({
                "year": current_month[0],
                "month": current_month[1],
                "income": month_return,
                "equity_start": month_start_equity,
                "equity_end": total_equity,
                "return_pct": month_return / month_start_equity * 100 if month_start_equity > 0 else 0,
            })
            current_month = dt_month
            month_start_equity = total_equity

    # Final month
    if current_month is not None and len(daily_equity) > 0:
        final_eq = daily_equity[-1]["equity"]
        month_return = final_eq - month_start_equity
        monthly_income.append({
            "year": current_month[0],
            "month": current_month[1],
            "income": month_return,
            "equity_start": month_start_equity,
            "equity_end": final_eq,
            "return_pct": month_return / month_start_equity * 100 if month_start_equity > 0 else 0,
        })

    eq_df = pd.DataFrame(daily_equity).set_index("date")
    inc_df = pd.DataFrame(monthly_income)

    # Position summary
    pos_summary = {}
    for tk in tickers:
        p = positions[tk]
        pos_summary[tk] = {
            "total_premium_collected": round(p.total_premium_collected, 2),
            "total_leaps_pnl": round(p.total_leaps_pnl, 2),
            "n_short_calls_sold": p.n_short_calls_sold,
            "n_assignments": p.n_assignments,
            "n_leaps_rolls": p.n_leaps_rolls,
            "total_commissions": round(p.total_commissions, 2),
            "final_cash": round(p.cash, 2),
        }

    return {
        "label": label,
        "equity": eq_df,
        "monthly_income": inc_df,
        "positions": pos_summary,
        "capital": capital,
    }


# ─── COVERED CALL BENCHMARK ──────────────────────────────────────────────

def backtest_covered_call(
    prices: pd.DataFrame,
    rvol: pd.DataFrame,
    risk_free: pd.Series,
    risk_overlay: Optional[pd.DataFrame] = None,
    capital: float = STARTING_CAPITAL,
    label: str = "CC",
) -> dict:
    """Simple covered call benchmark: own shares + sell 30-delta monthly calls."""
    idx = prices.index[prices.index >= BACKTEST_START]
    tickers = list(prices.columns)
    n_tickers = len(tickers)
    alloc_per_ticker = capital / n_tickers

    class CCPosition:
        def __init__(self, ticker, init_alloc):
            self.ticker = ticker
            self.shares = 0
            self.cost_basis = 0.0
            self.call_strike = None
            self.call_expiry_date = None
            self.call_premium = 0.0
            self.n_contracts = 0
            self.cash = init_alloc
            self.total_premium = 0.0
            self.n_calls_sold = 0

    positions_cc = {tk: CCPosition(tk, alloc_per_ticker) for tk in tickers}
    daily_equity = []
    monthly_income = []
    current_month = None
    month_start_equity = capital

    for i, date in enumerate(idx):
        if risk_overlay is not None and date in risk_overlay.index:
            rl = risk_overlay.loc[date, "risk_level"]
        else:
            rl = "normal"

        total_equity = 0.0

        for tk in tickers:
            pos = positions_cc[tk]
            price = prices.loc[date, tk] if date in prices.index else np.nan
            if np.isnan(price):
                total_equity += pos.cash + pos.shares * pos.cost_basis
                continue

            vol = rvol.loc[date, tk] if date in rvol.index else np.nan
            if np.isnan(vol) or vol <= 0.01:
                vol = 0.20
            rf = risk_free.get(date, RISK_FREE_RATE)

            # Buy shares if not owned
            if pos.shares == 0 and pos.cash > 0:
                n_shares = int(pos.cash / price / 100) * 100  # round to lots of 100
                if n_shares >= 100:
                    pos.shares = n_shares
                    pos.cost_basis = price
                    pos.cash -= n_shares * price

            # Manage short call
            if pos.call_expiry_date is not None:
                days_left = (pos.call_expiry_date - date).days
                if days_left <= 0:
                    if price > pos.call_strike:
                        # Assignment: sell shares at strike, rebuy
                        pos.cash += pos.shares * pos.call_strike
                        pos.shares = 0
                    pos.call_strike = None
                    pos.call_expiry_date = None

            # Sell new call
            delta_target = SHORT_CALL_DELTA.get(rl, 0.30)
            if pos.shares >= 100 and pos.call_strike is None and delta_target is not None:
                T_sc = SHORT_CALL_EXPIRY_DAYS / 365.0
                sc_strike = find_strike_for_delta(price, T_sc, rf, vol, delta_target)
                sc_premium = bs_call_price(price, sc_strike, T_sc, rf, vol)
                if sc_premium > 0.05:
                    n_contracts = pos.shares // 100
                    pos.call_strike = sc_strike
                    pos.call_expiry_date = date + timedelta(days=SHORT_CALL_EXPIRY_DAYS)
                    pos.call_premium = sc_premium
                    pos.n_contracts = n_contracts
                    premium = sc_premium * 100 * n_contracts
                    pos.cash += premium
                    pos.total_premium += premium
                    pos.n_calls_sold += 1

            total_equity += pos.cash + pos.shares * price

        daily_equity.append({"date": date, "equity": total_equity})

        dt_month = (date.year, date.month)
        if current_month is None:
            current_month = dt_month
            month_start_equity = total_equity
        elif dt_month != current_month:
            month_return = total_equity - month_start_equity
            monthly_income.append({
                "year": current_month[0],
                "month": current_month[1],
                "income": month_return,
                "equity_start": month_start_equity,
                "equity_end": total_equity,
                "return_pct": month_return / month_start_equity * 100 if month_start_equity > 0 else 0,
            })
            current_month = dt_month
            month_start_equity = total_equity

    if current_month is not None and len(daily_equity) > 0:
        final_eq = daily_equity[-1]["equity"]
        month_return = final_eq - month_start_equity
        monthly_income.append({
            "year": current_month[0],
            "month": current_month[1],
            "income": month_return,
            "equity_start": month_start_equity,
            "equity_end": final_eq,
            "return_pct": month_return / month_start_equity * 100 if month_start_equity > 0 else 0,
        })

    eq_df = pd.DataFrame(daily_equity).set_index("date")
    inc_df = pd.DataFrame(monthly_income)

    return {
        "label": label,
        "equity": eq_df,
        "monthly_income": inc_df,
        "capital": capital,
    }


# ─── BUY-HOLD + 60/40 BENCHMARKS ────────────────────────────────────────

def backtest_buy_hold(prices: pd.DataFrame, capital: float = STARTING_CAPITAL) -> dict:
    """SPY buy-and-hold benchmark."""
    spy = prices["SPY"].dropna()
    idx = spy.index[spy.index >= BACKTEST_START]
    shares = int(capital / spy.loc[idx[0]])
    cash = capital - shares * spy.loc[idx[0]]
    eq = pd.DataFrame({"equity": cash + shares * spy.loc[idx]}, index=idx)

    monthly_income = []
    current_month = None
    month_start = capital
    for date in idx:
        e = eq.loc[date, "equity"]
        dt_month = (date.year, date.month)
        if current_month is None:
            current_month = dt_month
            month_start = e
        elif dt_month != current_month:
            monthly_income.append({
                "year": current_month[0],
                "month": current_month[1],
                "income": e - month_start,
                "return_pct": (e - month_start) / month_start * 100,
            })
            current_month = dt_month
            month_start = e
    if current_month:
        e = eq.iloc[-1]["equity"]
        monthly_income.append({
            "year": current_month[0],
            "month": current_month[1],
            "income": e - month_start,
            "return_pct": (e - month_start) / month_start * 100,
        })

    return {
        "label": "SPY Buy-Hold",
        "equity": eq,
        "monthly_income": pd.DataFrame(monthly_income),
        "capital": capital,
    }


def backtest_60_40(prices: pd.DataFrame, overlay: dict, capital: float = STARTING_CAPITAL) -> dict:
    """60% SPY / 40% TLT, monthly rebalance."""
    spy = prices["SPY"].dropna()
    if "tlt" not in overlay:
        log.warning("No TLT data for 60/40 benchmark")
        return {"label": "60/40", "equity": pd.DataFrame(), "monthly_income": pd.DataFrame(), "capital": capital}

    tlt = overlay["tlt"]
    common_idx = spy.index.intersection(tlt.index)
    common_idx = common_idx[common_idx >= BACKTEST_START]

    spy_shares = int(capital * 0.60 / spy.loc[common_idx[0]])
    tlt_shares = int(capital * 0.40 / tlt.loc[common_idx[0]])
    cash = capital - spy_shares * spy.loc[common_idx[0]] - tlt_shares * tlt.loc[common_idx[0]]

    daily_equity = []
    current_month = None
    monthly_income = []
    month_start = capital

    for date in common_idx:
        eq = cash + spy_shares * spy.loc[date] + tlt_shares * tlt.loc[date]

        # Monthly rebalance
        dt_month = (date.year, date.month)
        if current_month is not None and dt_month != current_month:
            # Rebalance
            total = eq
            spy_shares = int(total * 0.60 / spy.loc[date])
            tlt_shares = int(total * 0.40 / tlt.loc[date])
            cash = total - spy_shares * spy.loc[date] - tlt_shares * tlt.loc[date]
            eq = cash + spy_shares * spy.loc[date] + tlt_shares * tlt.loc[date]

            monthly_income.append({
                "year": current_month[0],
                "month": current_month[1],
                "income": eq - month_start,
                "return_pct": (eq - month_start) / month_start * 100,
            })
            month_start = eq

        if current_month is None:
            current_month = dt_month
            month_start = eq
        else:
            current_month = dt_month

        daily_equity.append({"date": date, "equity": eq})

    if current_month and len(daily_equity) > 0:
        e = daily_equity[-1]["equity"]
        monthly_income.append({
            "year": current_month[0],
            "month": current_month[1],
            "income": e - month_start,
            "return_pct": (e - month_start) / month_start * 100,
        })

    eq_df = pd.DataFrame(daily_equity).set_index("date")
    return {
        "label": "60/40",
        "equity": eq_df,
        "monthly_income": pd.DataFrame(monthly_income),
        "capital": capital,
    }


# ─── PERFORMANCE METRICS ──────────────────────────────────────────────────

def compute_metrics(result: dict) -> dict:
    """Compute Sharpe, Sortino, CAGR, MaxDD, etc."""
    eq = result["equity"]["equity"]
    if len(eq) < 2:
        return {"label": result["label"], "error": "insufficient data"}

    daily_ret = eq.pct_change().dropna()

    # Annualized return
    n_years = len(daily_ret) / 252.0
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Sharpe
    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(252)
    sortino = (ann_ret - RISK_FREE_RATE) / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Monthly income stats
    inc = result.get("monthly_income", pd.DataFrame())
    inc_stats = {}
    if len(inc) > 0 and "income" in inc.columns:
        incomes = inc["income"]
        inc_stats = {
            "mean_monthly_income": round(incomes.mean(), 2),
            "median_monthly_income": round(incomes.median(), 2),
            "pct_months_positive": round((incomes > 0).mean() * 100, 1),
            "pct_months_gt_3k": round((incomes > 3000).mean() * 100, 1),
            "pct_months_gt_5k": round((incomes > 5000).mean() * 100, 1),
            "worst_month": round(incomes.min(), 2),
            "best_month": round(incomes.max(), 2),
        }

    return {
        "label": result["label"],
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "start_equity": round(eq.iloc[0], 2),
        "end_equity": round(eq.iloc[-1], 2),
        "n_years": round(n_years, 1),
        **inc_stats,
    }


# ─── YEAR-BY-YEAR RETURNS ────────────────────────────────────────────────

def yearly_returns(result: dict) -> pd.DataFrame:
    """Compute year-by-year returns."""
    eq = result["equity"]["equity"]
    yearly = eq.resample("YE").last()
    yearly_ret = yearly.pct_change().dropna()
    df = pd.DataFrame({
        "year": yearly_ret.index.year,
        "return_pct": (yearly_ret.values * 100).round(2),
    })
    return df


# ─── CRISIS PERIOD ANALYSIS ──────────────────────────────────────────────

CRISIS_PERIODS = {
    "2015 China Devaluation": ("2015-08-10", "2015-09-30"),
    "2016 Brexit": ("2016-06-23", "2016-07-15"),
    "2018 Vol-mageddon": ("2018-01-29", "2018-03-23"),
    "2018 Q4 Selloff": ("2018-10-01", "2018-12-24"),
    "COVID Crash 2020": ("2020-02-19", "2020-03-23"),
    "COVID Recovery": ("2020-03-23", "2020-06-30"),
    "2022 Bear Market": ("2022-01-03", "2022-10-12"),
    "2023 Banking Crisis": ("2023-03-08", "2023-03-24"),
    "2024 Aug VIX Spike": ("2024-07-16", "2024-08-05"),
    "2025 Tariff Crash": ("2025-02-19", "2025-04-08"),
}


def crisis_analysis(results: list[dict]) -> pd.DataFrame:
    """Compute returns during crisis periods for all strategies."""
    rows = []
    for crisis_name, (start, end) in CRISIS_PERIODS.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        for res in results:
            eq = res["equity"]["equity"]
            mask = (eq.index >= start_dt) & (eq.index <= end_dt)
            crisis_eq = eq[mask]
            if len(crisis_eq) >= 2:
                ret = (crisis_eq.iloc[-1] / crisis_eq.iloc[0] - 1) * 100
                rows.append({
                    "crisis": crisis_name,
                    "strategy": res["label"],
                    "return_pct": round(ret, 2),
                })
    return pd.DataFrame(rows)


# ─── HC #428 R1: REGIME-AGNOSTIC VALIDATION ─────────────────────────────

def regime_validation(result: dict, spy_prices: pd.Series) -> dict:
    """
    HC #428 R1: stratify by green/red/flat SPY days.
    Check |Sharpe_green - Sharpe_red| / max(|Sharpe_green|,|Sharpe_red|) <= 0.50.
    """
    eq = result["equity"]["equity"]
    daily_ret = eq.pct_change().dropna()

    spy_ret = spy_prices.pct_change().dropna()
    common = daily_ret.index.intersection(spy_ret.index)

    strat_ret = daily_ret.loc[common]
    spy_r = spy_ret.loc[common]

    green = spy_r > 0.001
    red = spy_r < -0.001
    flat = ~green & ~red

    def sharpe_subset(rets):
        if len(rets) < 5:
            return np.nan
        ann_r = rets.mean() * 252
        ann_v = rets.std() * np.sqrt(252)
        return (ann_r - RISK_FREE_RATE) / ann_v if ann_v > 0 else 0

    sharpe_green = sharpe_subset(strat_ret[green])
    sharpe_red = sharpe_subset(strat_ret[red])
    sharpe_flat = sharpe_subset(strat_ret[flat])

    max_abs = max(abs(sharpe_green) if not np.isnan(sharpe_green) else 0,
                  abs(sharpe_red) if not np.isnan(sharpe_red) else 0)
    regime_gap = abs(sharpe_green - sharpe_red) / max_abs if max_abs > 0 else 0
    passes = regime_gap <= 0.50

    return {
        "label": result["label"],
        "sharpe_green": round(sharpe_green, 3) if not np.isnan(sharpe_green) else None,
        "sharpe_red": round(sharpe_red, 3) if not np.isnan(sharpe_red) else None,
        "sharpe_flat": round(sharpe_flat, 3) if not np.isnan(sharpe_flat) else None,
        "regime_gap": round(regime_gap, 3),
        "passes_r1": passes,
        "n_green": int(green.sum()),
        "n_red": int(red.sum()),
        "n_flat": int(flat.sum()),
    }


# ─── PERMUTATION TEST ──────────────────────────────────────────────────────

def permutation_test(
    prices: pd.DataFrame,
    rvol: pd.DataFrame,
    risk_free: pd.Series,
    risk_overlay: pd.DataFrame,
    n_shuffles: int = 100,
) -> dict:
    """Shuffle overlay timing to test if overlay adds genuine value."""
    log.info("Running permutation test with %d shuffles ...", n_shuffles)

    # Real result
    real = backtest_pmcc(prices, rvol, risk_free, risk_overlay, label="PMCC_real")
    real_sharpe = compute_metrics(real)["sharpe"]

    # Shuffled results
    shuffled_sharpes = []
    overlay_values = risk_overlay["risk_level"].values.copy()

    for i in range(n_shuffles):
        shuffled = risk_overlay.copy()
        np.random.shuffle(overlay_values)
        shuffled["risk_level"] = overlay_values
        overlay_values = risk_overlay["risk_level"].values.copy()  # reset

        res = backtest_pmcc(prices, rvol, risk_free, shuffled, label=f"shuffle_{i}")
        s = compute_metrics(res)["sharpe"]
        shuffled_sharpes.append(s)

        if (i + 1) % 50 == 0:
            log.info("  Permutation %d/%d done", i + 1, n_shuffles)

    shuffled_sharpes = np.array(shuffled_sharpes)
    p_value = (shuffled_sharpes >= real_sharpe).mean()

    return {
        "real_sharpe": round(real_sharpe, 3),
        "mean_shuffled_sharpe": round(shuffled_sharpes.mean(), 3),
        "std_shuffled_sharpe": round(shuffled_sharpes.std(), 3),
        "p_value": round(p_value, 4),
        "pct_better_than_real": round(p_value * 100, 1),
        "n_shuffles": n_shuffles,
    }


# ─── CAPITAL NEEDED CALCULATOR ───────────────────────────────────────────

def capital_needed(monthly_income_df: pd.DataFrame, capital: float) -> dict:
    """Calculate capital needed for $3K/month and $5K/month targets."""
    if len(monthly_income_df) == 0 or "income" not in monthly_income_df.columns:
        return {}

    median_monthly = monthly_income_df["income"].median()
    mean_monthly = monthly_income_df["income"].mean()

    if median_monthly <= 0:
        return {
            "warning": "Median monthly income is negative — strategy not viable for income",
            "median_monthly": round(median_monthly, 2),
        }

    scale_3k_median = 3000 / median_monthly * capital
    scale_5k_median = 5000 / median_monthly * capital
    scale_3k_mean = 3000 / mean_monthly * capital if mean_monthly > 0 else float("inf")
    scale_5k_mean = 5000 / mean_monthly * capital if mean_monthly > 0 else float("inf")

    return {
        "median_monthly_income_at_100k": round(median_monthly, 2),
        "mean_monthly_income_at_100k": round(mean_monthly, 2),
        "capital_for_3k_month_median": round(scale_3k_median, 0),
        "capital_for_5k_month_median": round(scale_5k_median, 0),
        "capital_for_3k_month_mean": round(scale_3k_mean, 0),
        "capital_for_5k_month_mean": round(scale_5k_mean, 0),
    }


# ─── CAPITAL EFFICIENCY ─────────────────────────────────────────────────

def capital_efficiency(pmcc_metrics: dict, cc_metrics: dict) -> dict:
    """Compare return per dollar invested: PMCC vs covered call."""
    pmcc_cagr = pmcc_metrics.get("cagr_pct", 0)
    cc_cagr = cc_metrics.get("cagr_pct", 0)

    # PMCC uses ~80% of capital for LEAPS (which cost ~10-15% of shares)
    # So effective capital deployed is ~10-15% of what CC uses
    # But we use 80% of the same $100K, so actual $ deployed is same
    # The efficiency is in the RETURN relative to max risk

    pmcc_end = pmcc_metrics.get("end_equity", 100000)
    cc_end = cc_metrics.get("end_equity", 100000)
    pmcc_dd = abs(pmcc_metrics.get("max_dd_pct", 1))
    cc_dd = abs(cc_metrics.get("max_dd_pct", 1))

    return {
        "pmcc_cagr": pmcc_cagr,
        "cc_cagr": cc_cagr,
        "cagr_ratio": round(pmcc_cagr / cc_cagr, 3) if cc_cagr != 0 else None,
        "pmcc_sharpe": pmcc_metrics.get("sharpe", 0),
        "cc_sharpe": cc_metrics.get("sharpe", 0),
        "pmcc_max_dd": pmcc_metrics.get("max_dd_pct", 0),
        "cc_max_dd": cc_metrics.get("max_dd_pct", 0),
        "dd_ratio": round(pmcc_dd / cc_dd, 3) if cc_dd != 0 else None,
        "pmcc_calmar": pmcc_metrics.get("calmar", 0),
        "cc_calmar": cc_metrics.get("calmar", 0),
    }


# ─── MAIN ────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("PMCC Income Strategy Backtest v1")
    log.info("=" * 70)
    log.info("Universe: %s", UNIVERSE)
    log.info("Period: %s to %s", BACKTEST_START, END)
    log.info("Capital: $%d", STARTING_CAPITAL)

    # ── Download data ──
    prices, overlay = download_data()
    risk_overlay = compute_risk_overlay(overlay)
    rvol = compute_realized_vol(prices)
    risk_free = get_risk_free_series(overlay)

    log.info("Risk overlay computed: normal=%.1f%%, elevated=%.1f%%, high=%.1f%%",
             (risk_overlay["risk_level"] == "normal").mean() * 100,
             (risk_overlay["risk_level"] == "elevated").mean() * 100,
             (risk_overlay["risk_level"] == "high").mean() * 100)

    # ── Run backtests ──
    log.info("\n--- Running PMCC with overlay ---")
    pmcc_overlay = backtest_pmcc(prices, rvol, risk_free, risk_overlay, label="PMCC + Overlay")

    log.info("\n--- Running PMCC without overlay ---")
    pmcc_no_overlay = backtest_pmcc(prices, rvol, risk_free, None, label="PMCC (no overlay)")

    log.info("\n--- Running Covered Call benchmark ---")
    cc_result = backtest_covered_call(prices, rvol, risk_free, risk_overlay, label="Covered Call")

    log.info("\n--- Running Buy-Hold benchmark ---")
    bh_result = backtest_buy_hold(prices)

    log.info("\n--- Running 60/40 benchmark ---")
    sixfour_result = backtest_60_40(prices, overlay)

    all_results = [pmcc_overlay, pmcc_no_overlay, cc_result, bh_result, sixfour_result]

    # ── Compute metrics ──
    log.info("\n" + "=" * 70)
    log.info("PERFORMANCE COMPARISON")
    log.info("=" * 70)

    all_metrics = {}
    for res in all_results:
        m = compute_metrics(res)
        all_metrics[res["label"]] = m
        log.info("\n--- %s ---", m["label"])
        for k, v in m.items():
            if k != "label":
                log.info("  %s: %s", k, v)

    # ── Year-by-year ──
    log.info("\n" + "=" * 70)
    log.info("YEAR-BY-YEAR RETURNS")
    log.info("=" * 70)

    yearly_data = {}
    for res in all_results:
        yr = yearly_returns(res)
        yearly_data[res["label"]] = yr
        log.info("\n%s:", res["label"])
        for _, row in yr.iterrows():
            log.info("  %d: %.2f%%", row["year"], row["return_pct"])

    # ── Crisis analysis ──
    log.info("\n" + "=" * 70)
    log.info("CRISIS PERIOD ANALYSIS")
    log.info("=" * 70)

    crisis_df = crisis_analysis(all_results)
    for crisis_name in CRISIS_PERIODS:
        subset = crisis_df[crisis_df["crisis"] == crisis_name]
        if len(subset) > 0:
            log.info("\n%s:", crisis_name)
            for _, row in subset.iterrows():
                log.info("  %s: %.2f%%", row["strategy"], row["return_pct"])

    # ── HC #428 R1: Regime validation ──
    log.info("\n" + "=" * 70)
    log.info("HC #428 R1: REGIME-AGNOSTIC VALIDATION")
    log.info("=" * 70)

    spy_prices = prices["SPY"]
    regime_results = {}
    for res in [pmcc_overlay, pmcc_no_overlay, cc_result]:
        rv = regime_validation(res, spy_prices)
        regime_results[res["label"]] = rv
        log.info("\n%s:", rv["label"])
        log.info("  Sharpe (green days): %s", rv["sharpe_green"])
        log.info("  Sharpe (red days):   %s", rv["sharpe_red"])
        log.info("  Sharpe (flat days):  %s", rv["sharpe_flat"])
        log.info("  Regime gap:          %.3f", rv["regime_gap"])
        log.info("  PASSES R1:           %s", rv["passes_r1"])

    # ── Capital efficiency ──
    log.info("\n" + "=" * 70)
    log.info("CAPITAL EFFICIENCY")
    log.info("=" * 70)

    cap_eff = capital_efficiency(all_metrics["PMCC + Overlay"], all_metrics["Covered Call"])
    for k, v in cap_eff.items():
        log.info("  %s: %s", k, v)

    # ── Capital needed ──
    log.info("\n" + "=" * 70)
    log.info("CAPITAL NEEDED FOR TARGET INCOME")
    log.info("=" * 70)

    cap_needed = capital_needed(pmcc_overlay["monthly_income"], STARTING_CAPITAL)
    for k, v in cap_needed.items():
        log.info("  %s: %s", k, v)

    # ── Position details ──
    log.info("\n" + "=" * 70)
    log.info("PMCC POSITION DETAILS")
    log.info("=" * 70)

    for tk, ps in pmcc_overlay.get("positions", {}).items():
        log.info("\n%s:", tk)
        for k, v in ps.items():
            log.info("  %s: %s", k, v)

    # ── Permutation test ──
    log.info("\n" + "=" * 70)
    log.info("PERMUTATION TEST (100 shuffles)")
    log.info("=" * 70)

    perm = permutation_test(prices, rvol, risk_free, risk_overlay, n_shuffles=100)
    for k, v in perm.items():
        log.info("  %s: %s", k, v)

    # ── Save results ──
    log.info("\n" + "=" * 70)
    log.info("SAVING RESULTS")
    log.info("=" * 70)

    # Save equity curves
    for res in all_results:
        safe_label = res["label"].replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("/", "_")
        res["equity"].to_csv(OUT_DIR / f"equity_{safe_label}.csv")
        if len(res.get("monthly_income", pd.DataFrame())) > 0:
            res["monthly_income"].to_csv(OUT_DIR / f"monthly_income_{safe_label}.csv", index=False)

    # Save crisis analysis
    crisis_df.to_csv(OUT_DIR / "crisis_analysis.csv", index=False)

    # Save comparison metrics
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(OUT_DIR / "comparison_metrics.csv")

    # Save yearly returns
    for label, yr in yearly_data.items():
        safe_label = label.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("/", "_")
        yr.to_csv(OUT_DIR / f"yearly_{safe_label}.csv", index=False)

    # Save full results as JSON
    full_results = {
        "metrics": all_metrics,
        "capital_efficiency": cap_eff,
        "capital_needed": cap_needed,
        "regime_validation": regime_results,
        "permutation_test": perm,
        "positions": pmcc_overlay.get("positions", {}),
        "risk_overlay_distribution": {
            "normal_pct": round((risk_overlay["risk_level"] == "normal").mean() * 100, 1),
            "elevated_pct": round((risk_overlay["risk_level"] == "elevated").mean() * 100, 1),
            "high_pct": round((risk_overlay["risk_level"] == "high").mean() * 100, 1),
        },
        "config": {
            "universe": UNIVERSE,
            "backtest_start": BACKTEST_START,
            "backtest_end": END,
            "starting_capital": STARTING_CAPITAL,
            "leaps_delta": LEAPS_DELTA_TARGET,
            "short_call_deltas": SHORT_CALL_DELTA,
            "leaps_target_dte": LEAPS_TARGET_DTE,
            "leaps_roll_dte": LEAPS_ROLL_DTE,
            "short_call_expiry_days": SHORT_CALL_EXPIRY_DAYS,
            "max_alloc_per_etf": MAX_ALLOC_PER_ETF,
            "leaps_capital_pct": LEAPS_CAPITAL_PCT,
        },
    }

    with open(OUT_DIR / "full_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    log.info("\nAll results saved to %s", OUT_DIR)
    log.info("DONE.")


if __name__ == "__main__":
    main()
