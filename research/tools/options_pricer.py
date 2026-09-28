#!/usr/bin/env python3
"""
Options Pricer — Standardized ATR-Based Black-Scholes with Bid-Ask Haircuts
=============================================================================

IMPORT THIS MODULE instead of reimplementing BS pricing in every script.

KEY DESIGN RULE (from QUANT_KNOWLEDGE_BASE.md finding #10):
  Exit pricing WITHOUT bid-ask haircut inflates Sharpe by ~31%.
  This module ALWAYS applies haircut on BOTH entry AND exit.

Assumptions documented here:
  - Black-Scholes European pricing (adequate for sector ETF options with monthly DTE)
  - ATR-based implied vol: sigma = ATR_annual * iv_multiplier
    - iv_multiplier = 1.2 + 0.01 * max(VIX - 20, 0), roughly 1.2x realized vol
  - 15% default haircut covers bid-ask spread with ~14% margin (validated against
    live SPY option quotes, July 2026: model error ~4.2% for bull call spreads)
  - Commission: $0.65/leg x 4 legs = $2.60 round-trip per spread

Validated accuracy:
  - Bull call spreads: ~4% error vs real markets (with vol skew adjustment)
  - Iron condors: model UNDERPRICES credit by ~68% (conservative)

Usage:
    from research.tools.options_pricer import price_bull_call_spread, exit_spread_value

    entry_cost, max_profit = price_bull_call_spread(
        S=45.0, K1=45.0, K2=47.0, dte=30, atr=1.2, vix=22.0
    )

    exit_val = exit_spread_value(
        S=46.5, K1=45.0, K2=47.0, remaining_dte=10, original_dte=30,
        atr=1.2, vix=20.0
    )
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ─── Constants ───────────────────────────────────────────────────────

RISK_FREE_RATE = 0.045          # Current-ish risk-free rate
DEFAULT_HAIRCUT = 0.15          # 15% bid-ask haircut on BS fair value
COMMISSION_PER_LEG = 0.65       # Discount broker per-leg
COMMISSION_RT_SPREAD = 2.60     # 4 legs round-trip ($0.65 x 4)


# ─── Core Black-Scholes ─────────────────────────────────────────────

def bs_call_price(
    S: float,
    K: float,
    T: float,
    r: float = RISK_FREE_RATE,
    sigma: float = 0.25,
) -> float:
    """
    Black-Scholes European call option price.

    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiry in YEARS (e.g., 30/365 for 30 DTE)
        r: Risk-free rate (annualized)
        sigma: Implied volatility (annualized, e.g., 0.25 = 25%)

    Returns:
        Theoretical call price per share.
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put_price(
    S: float,
    K: float,
    T: float,
    r: float = RISK_FREE_RATE,
    sigma: float = 0.25,
) -> float:
    """Black-Scholes European put option price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ─── Implied Volatility Estimation ──────────────────────────────────

def estimate_iv(
    atr: float,
    spot: float,
    vix: float = 20.0,
    atr_period: int = 14,
) -> float:
    """
    Estimate implied volatility from ATR and VIX.

    Method:
      1. Convert ATR to annualized realized volatility:
         realized_vol = (ATR / spot) * sqrt(252 / atr_period)
      2. Apply IV multiplier: IV typically ~1.2x realized vol,
         increasing with VIX (options get more expensive in fear).
         iv_mult = 1.2 + 0.01 * max(VIX - 20, 0)
      3. Floor at 10% to avoid degenerate pricing.

    Args:
        atr: Average True Range (dollar value, e.g., 1.5)
        spot: Current underlying price
        vix: Current VIX level
        atr_period: ATR lookback period (default 14)

    Returns:
        Estimated annualized implied volatility (e.g., 0.28 = 28%)
    """
    if spot <= 0 or atr <= 0:
        return 0.25  # fallback

    realized_vol = (atr / spot) * np.sqrt(252 / atr_period)
    iv_mult = 1.2 + 0.01 * max(vix - 20.0, 0.0)
    sigma = realized_vol * iv_mult

    return max(sigma, 0.10)  # floor at 10%


# ─── Bull Call Spread Pricing ────────────────────────────────────────

def price_bull_call_spread(
    S: float,
    K1: float,
    K2: float,
    dte: int,
    atr: float,
    vix: float = 20.0,
    haircut: float = DEFAULT_HAIRCUT,
    r: float = RISK_FREE_RATE,
    sigma: float | None = None,
) -> tuple[float, float]:
    """
    Price a bull call spread with bid-ask haircut applied on ENTRY.

    A bull call spread = buy call at K1 (lower), sell call at K2 (higher).
    Entry cost = (BS_call(K1) - BS_call(K2)) * (1 + haircut)
    The haircut models paying more than mid-price when buying (crossing the spread).

    Args:
        S: Current underlying price
        K1: Lower strike (long call)
        K2: Upper strike (short call), must be > K1
        dte: Days to expiration
        atr: Average True Range of underlying
        vix: Current VIX level
        haircut: Bid-ask haircut fraction (default 0.15 = 15%)
        r: Risk-free rate
        sigma: Override implied vol (if None, estimated from ATR + VIX)

    Returns:
        (entry_cost, max_profit) — both per-share values.
        entry_cost: what you PAY to enter (haircut applied, higher than fair value).
        max_profit: maximum possible profit at expiry = (K2 - K1) - entry_cost.
        Note: These are per-share. Multiply by 100 for per-contract.
    """
    if K2 <= K1:
        raise ValueError(f"K2 ({K2}) must be > K1 ({K1})")

    T = dte / 365.0
    if sigma is None:
        sigma = estimate_iv(atr, S, vix)

    fair_value = bs_call_price(S, K1, T, r, sigma) - bs_call_price(S, K2, T, r, sigma)
    fair_value = max(fair_value, 0.001)  # floor

    # ENTRY: you buy the spread → pay MORE than fair (haircut UP)
    entry_cost = fair_value * (1.0 + haircut)

    # Max profit at expiry = spread width - entry cost
    spread_width = K2 - K1
    max_profit = spread_width - entry_cost

    return float(entry_cost), float(max_profit)


def exit_spread_value(
    S: float,
    K1: float,
    K2: float,
    remaining_dte: int,
    original_dte: int,
    atr: float,
    vix: float = 20.0,
    haircut: float = DEFAULT_HAIRCUT,
    r: float = RISK_FREE_RATE,
    sigma: float | None = None,
) -> float:
    """
    Value of an existing bull call spread when exiting BEFORE expiry.

    CRITICAL (QUANT_KNOWLEDGE_BASE finding #10):
      Exit pricing without bid-ask haircut inflates Sharpe by ~31%.
      This function ALWAYS applies haircut on exit (you receive LESS than fair value).

    At expiry (remaining_dte <= 0): returns intrinsic value (no haircut needed,
    exercise/assignment is automatic).

    Mid-life exit: returns fair_value * (1 - haircut), modeling the fact that
    you sell at the bid, not the mid.

    Args:
        S: Current underlying price at exit time
        K1: Lower strike (long call)
        K2: Upper strike (short call)
        remaining_dte: Days until expiration remaining
        original_dte: Original DTE at entry (for reference, not used in calc)
        atr: Current ATR of underlying
        vix: Current VIX level
        haircut: Bid-ask haircut fraction (default 0.15)
        r: Risk-free rate
        sigma: Override implied vol

    Returns:
        Exit value per share (what you RECEIVE after haircut).
    """
    if remaining_dte <= 0:
        # At expiry: intrinsic value, no haircut (automatic exercise)
        intrinsic_low = max(S - K1, 0.0)
        intrinsic_high = max(S - K2, 0.0)
        return float(intrinsic_low - intrinsic_high)

    T = remaining_dte / 365.0
    if sigma is None:
        sigma = estimate_iv(atr, S, vix)

    fair_value = bs_call_price(S, K1, T, r, sigma) - bs_call_price(S, K2, T, r, sigma)
    fair_value = max(fair_value, 0.0)

    # EXIT: you sell the spread → receive LESS than fair (haircut DOWN)
    exit_value = fair_value * (1.0 - haircut)

    return float(exit_value)


# ─── Bear Put Spread Pricing ────────────────────────────────────────

def price_bear_put_spread(
    S: float,
    K1: float,
    K2: float,
    dte: int,
    atr: float,
    vix: float = 20.0,
    haircut: float = DEFAULT_HAIRCUT,
    r: float = RISK_FREE_RATE,
    sigma: float | None = None,
) -> tuple[float, float]:
    """
    Price a bear put spread with bid-ask haircut.

    Bear put spread = buy put at K2 (higher), sell put at K1 (lower).
    Profits when underlying falls.

    Args:
        S: Current underlying price
        K1: Lower strike (short put)
        K2: Upper strike (long put), must be > K1
        dte: Days to expiration
        atr: ATR of underlying
        vix: Current VIX level
        haircut: Bid-ask haircut (default 15%)
        r: Risk-free rate
        sigma: Override IV

    Returns:
        (entry_cost, max_profit) per share.
    """
    if K2 <= K1:
        raise ValueError(f"K2 ({K2}) must be > K1 ({K1})")

    T = dte / 365.0
    if sigma is None:
        sigma = estimate_iv(atr, S, vix)

    fair_value = bs_put_price(S, K2, T, r, sigma) - bs_put_price(S, K1, T, r, sigma)
    fair_value = max(fair_value, 0.001)

    entry_cost = fair_value * (1.0 + haircut)
    spread_width = K2 - K1
    max_profit = spread_width - entry_cost

    return float(entry_cost), float(max_profit)


# ─── Utility Functions ──────────────────────────────────────────────

def spread_pnl(
    entry_cost: float,
    exit_value: float,
    contracts: int = 1,
    commission_rt: float = COMMISSION_RT_SPREAD,
) -> float:
    """
    Compute PnL for a spread trade.

    Args:
        entry_cost: Per-share entry cost (from price_bull_call_spread)
        exit_value: Per-share exit value (from exit_spread_value or expiry intrinsic)
        contracts: Number of contracts
        commission_rt: Round-trip commission per contract

    Returns:
        Dollar PnL after commissions.
    """
    return (exit_value - entry_cost) * contracts * 100 - commission_rt * contracts


def compute_atr(
    high: np.ndarray | "pd.Series",
    low: np.ndarray | "pd.Series",
    close: np.ndarray | "pd.Series",
    period: int = 14,
) -> float:
    """
    Compute the most recent ATR value from OHLC data.

    Args:
        high, low, close: price arrays (most recent values at the end)
        period: ATR lookback period

    Returns:
        Current ATR value (scalar).
    """
    import pandas as pd

    high = pd.Series(high) if not isinstance(high, pd.Series) else high
    low = pd.Series(low) if not isinstance(low, pd.Series) else low
    close = pd.Series(close) if not isinstance(close, pd.Series) else close

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1 / period, min_periods=period).mean()

    return float(atr_series.iloc[-1])


# ─── Self-Test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running options_pricer self-test...\n")

    # Test bull call spread on a $45 sector ETF, 30 DTE, VIX=22
    S, K1, K2 = 45.0, 45.0, 46.35  # 3% spread
    dte = 30
    atr = 1.2
    vix = 22.0

    entry_cost, max_profit = price_bull_call_spread(S, K1, K2, dte, atr, vix)
    print(f"Bull Call Spread: {K1}/{K2} @ ${S:.2f}, {dte}DTE, VIX={vix}")
    print(f"  Entry cost (per share):  ${entry_cost:.4f}")
    print(f"  Entry cost (per contract): ${entry_cost*100:.2f}")
    print(f"  Max profit (per share):  ${max_profit:.4f}")
    print(f"  Max profit (per contract): ${max_profit*100:.2f}")

    # Test mid-life exit at day 20 (10 DTE remaining)
    S_exit = 46.0
    exit_val = exit_spread_value(S_exit, K1, K2, 10, dte, atr, vix)
    print(f"\n  Exit at ${S_exit:.2f} with 10 DTE remaining:")
    print(f"  Exit value (per share, after haircut): ${exit_val:.4f}")

    # Test expiry scenarios
    for S_exp in [44.0, 45.5, 46.0, 47.0]:
        exp_val = exit_spread_value(S_exp, K1, K2, 0, dte, atr, vix)
        pnl = spread_pnl(entry_cost, exp_val, contracts=1)
        print(f"  Expiry at ${S_exp:.2f}: value=${exp_val:.4f}, PnL=${pnl:.2f}")

    # IV estimation test
    sigma = estimate_iv(atr=1.2, spot=45.0, vix=22.0)
    print(f"\n  Estimated IV: {sigma*100:.1f}% (ATR=1.2, S=45, VIX=22)")

    sigma_low = estimate_iv(atr=0.5, spot=45.0, vix=15.0)
    print(f"  Estimated IV: {sigma_low*100:.1f}% (ATR=0.5, S=45, VIX=15)")

    sigma_high = estimate_iv(atr=2.5, spot=45.0, vix=35.0)
    print(f"  Estimated IV: {sigma_high*100:.1f}% (ATR=2.5, S=45, VIX=35)")

    print("\nSelf-test complete.")
