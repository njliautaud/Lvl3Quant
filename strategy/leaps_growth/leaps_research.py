#!/usr/bin/env python3
"""
LEAPS Growth Strategy Research
==============================
For a $441 Robinhood account: identifies affordable LEAPS candidates,
models return profiles, and analyzes Poor Man's Covered Call (PMCC) setups.

Usage:
    python leaps_research.py              # Full scan + analysis
    python leaps_research.py --quick      # Quick scan, top 10 only
    python leaps_research.py --ticker SOFI # Analyze a specific ticker
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/leaps_growth")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNT_SIZE = 441.0
MAX_LEAPS_COST = 500.0  # Max we'd pay for a single LEAPS contract
TARGET_DELTA = (0.70, 0.80)  # Deep ITM range
LEAPS_MIN_DTE = 180  # At least 6 months out
LEAPS_MAX_DTE = 730  # Up to 2 years

# Risk-free rate assumption
RISK_FREE_RATE = 0.043  # ~4.3% as of mid-2026

# ─────────────────────────────────────────────────────
# Black-Scholes helpers
# ─────────────────────────────────────────────────────

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, r, sigma):
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def bs_theta(S, K, T, r, sigma):
    """Black-Scholes call theta (per day, negative = decay)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = (
        -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    ) / 365.0
    return theta


def bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega (per 1% IV change)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100.0


def find_strike_for_delta(S, T, r, sigma, target_delta):
    """Binary search for strike that gives target delta."""
    lo, hi = S * 0.3, S * 1.5
    for _ in range(100):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, r, sigma)
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return mid


# ─────────────────────────────────────────────────────
# Candidate screening
# ─────────────────────────────────────────────────────

# Universe: liquid large/mid-cap stocks with options on Robinhood,
# priced low enough that LEAPS might be under $500
SCREENING_UNIVERSE = [
    # Tech / Growth
    "SOFI", "PLTR", "AMD", "MARA", "RIOT", "HOOD", "RKLB", "IONQ",
    "GRAB", "NU", "SE", "SHOP", "SNAP", "PINS", "UBER", "LYFT",
    "DKNG", "COIN", "SQ", "AFRM", "UPST", "PATH", "CRWD",
    "NET", "DDOG", "SNOW", "ROKU", "RIVN", "LCID", "NIO",
    "XPEV", "LI", "BABA", "JD", "PDD", "KWEB",
    # Biotech / Healthcare
    "MRNA", "DNA", "CRISPR", "BEAM", "NTLA",
    # Energy / Materials
    "FSLR", "ENPH", "RUN", "CLF", "MP", "LAC",
    # Consumer / Other
    "CHWY", "ETSY", "DIS", "NCLH", "CCL", "RCL",
    # Broad market ETFs (for comparison)
    "SPY", "QQQ", "IWM", "ARKK", "XLK",
    # Additional affordable growth
    "OPEN", "WISH", "CLOV", "TLRY", "CGC", "ACB",
    "FUBO", "GSAT", "BB", "NOK", "F", "GM",
    "AAL", "DAL", "UAL", "BA",
    "T", "VZ", "INTC", "MU", "ON", "SMCI",
]


def get_stock_data(ticker, period="1y"):
    """Fetch stock price + fundamentals."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period)
        if hist.empty or len(hist) < 60:
            return None

        info = tk.info or {}
        current_price = hist["Close"].iloc[-1]

        # Momentum metrics
        price_6m_ago = hist["Close"].iloc[-min(126, len(hist))]
        price_3m_ago = hist["Close"].iloc[-min(63, len(hist))]
        mom_6m = (current_price / price_6m_ago - 1) * 100
        mom_3m = (current_price / price_3m_ago - 1) * 100

        # Volatility (annualized)
        returns = hist["Close"].pct_change().dropna()
        hist_vol = returns.std() * np.sqrt(252) * 100

        # Average volume
        avg_volume = hist["Volume"].mean()

        # Earnings growth (from info)
        earnings_growth = info.get("earningsGrowth", None)
        revenue_growth = info.get("revenueGrowth", None)
        forward_pe = info.get("forwardPE", None)
        market_cap = info.get("marketCap", None)

        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "mom_3m": round(mom_3m, 1),
            "mom_6m": round(mom_6m, 1),
            "hist_vol": round(hist_vol, 1),
            "avg_volume": int(avg_volume),
            "earnings_growth": earnings_growth,
            "revenue_growth": revenue_growth,
            "forward_pe": forward_pe,
            "market_cap": market_cap,
            "sector": info.get("sector", "Unknown"),
        }
    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")
        return None


def score_candidate(data):
    """Score a stock for LEAPS suitability (higher = better)."""
    score = 0

    # Must be affordable: price * 100 * ~0.15 (time value ratio for LEAPS) < MAX_LEAPS_COST
    # Rough filter: stock price should be under ~$50 for affordable LEAPS
    if data["price"] > 60:
        return -999  # Too expensive for $441 account

    # Momentum (positive is good for LEAPS calls)
    if data["mom_6m"] > 20:
        score += 3
    elif data["mom_6m"] > 10:
        score += 2
    elif data["mom_6m"] > 0:
        score += 1
    elif data["mom_6m"] < -20:
        score -= 2

    if data["mom_3m"] > 10:
        score += 2
    elif data["mom_3m"] > 0:
        score += 1

    # Volatility: moderate is ideal (enough movement, not crazy premium)
    if 30 <= data["hist_vol"] <= 60:
        score += 2  # Sweet spot
    elif 20 <= data["hist_vol"] < 30:
        score += 1  # Low vol = less leverage benefit
    elif data["hist_vol"] > 80:
        score -= 1  # Too volatile = expensive premiums

    # Liquidity
    if data["avg_volume"] > 10_000_000:
        score += 2
    elif data["avg_volume"] > 5_000_000:
        score += 1

    # Earnings growth
    eg = data.get("earnings_growth")
    if eg is not None:
        if eg > 0.20:
            score += 3
        elif eg > 0.10:
            score += 2
        elif eg > 0:
            score += 1
        else:
            score -= 1

    # Revenue growth
    rg = data.get("revenue_growth")
    if rg is not None:
        if rg > 0.15:
            score += 2
        elif rg > 0.05:
            score += 1

    # Forward PE: lower is better value for growth
    fpe = data.get("forward_pe")
    if fpe is not None:
        if 5 < fpe < 20:
            score += 2
        elif 20 <= fpe < 40:
            score += 1
        elif fpe > 100:
            score -= 1

    return score


def screen_candidates(tickers=None, top_n=20):
    """Screen universe for LEAPS candidates."""
    if tickers is None:
        tickers = SCREENING_UNIVERSE

    print(f"\n{'='*60}")
    print(f"LEAPS CANDIDATE SCREENING")
    print(f"Universe: {len(tickers)} tickers | Budget: ${ACCOUNT_SIZE}")
    print(f"{'='*60}\n")

    results = []
    for i, ticker in enumerate(tickers):
        if (i + 1) % 10 == 0:
            print(f"  Scanning {i+1}/{len(tickers)}...")
        data = get_stock_data(ticker)
        if data is None:
            continue
        data["score"] = score_candidate(data)
        if data["score"] > -999:
            results.append(data)

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\nFound {len(results)} viable candidates")
    print(f"\nTop {min(top_n, len(results))} by composite score:")
    print(f"{'Ticker':<8} {'Price':>8} {'3M%':>7} {'6M%':>7} {'Vol%':>6} {'Score':>6}")
    print("-" * 50)
    for r in results[:top_n]:
        print(
            f"{r['ticker']:<8} ${r['price']:>7.2f} {r['mom_3m']:>6.1f}% "
            f"{r['mom_6m']:>6.1f}% {r['hist_vol']:>5.1f}% {r['score']:>5}"
        )

    return results[:top_n]


# ─────────────────────────────────────────────────────
# LEAPS return modeling
# ─────────────────────────────────────────────────────

def model_leaps_returns(ticker_data, dte=365):
    """
    Model LEAPS return profiles for a given stock.
    Uses Black-Scholes theoretical pricing.
    """
    S = ticker_data["price"]
    sigma = ticker_data["hist_vol"] / 100.0  # Convert to decimal
    T = dte / 365.0
    r = RISK_FREE_RATE

    # Find strikes for target delta range
    strike_lo = find_strike_for_delta(S, T, r, sigma, TARGET_DELTA[1])  # 0.80 delta = deeper ITM
    strike_hi = find_strike_for_delta(S, T, r, sigma, TARGET_DELTA[0])  # 0.70 delta = less ITM
    strike_mid = (strike_lo + strike_hi) / 2

    # Round to nearest $0.50 or $1
    if S < 20:
        strike_mid = round(strike_mid * 2) / 2
    else:
        strike_mid = round(strike_mid)

    K = strike_mid
    call_price = bs_call_price(S, K, T, r, sigma)
    delta = bs_delta(S, K, T, r, sigma)
    theta_day = bs_theta(S, K, T, r, sigma)
    vega_val = bs_vega(S, K, T, r, sigma)

    contract_cost = call_price * 100  # 1 contract = 100 shares
    intrinsic = max(S - K, 0) * 100
    time_value = contract_cost - intrinsic
    leverage = (S * 100) / contract_cost if contract_cost > 0 else 0

    result = {
        "ticker": ticker_data["ticker"],
        "stock_price": S,
        "strike": K,
        "dte": dte,
        "call_price": round(call_price, 2),
        "contract_cost": round(contract_cost, 2),
        "intrinsic_value": round(intrinsic, 2),
        "time_value": round(time_value, 2),
        "delta": round(delta, 3),
        "theta_per_day": round(theta_day, 4),
        "theta_pct_per_day": round(theta_day / call_price * 100, 3) if call_price > 0 else 0,
        "vega": round(vega_val, 4),
        "leverage_ratio": round(leverage, 2),
        "implied_vol": round(sigma * 100, 1),
        "affordable": contract_cost <= ACCOUNT_SIZE,
    }

    # Model return scenarios
    scenarios = {}
    stock_moves = [-30, -20, -10, 0, 10, 20, 30, 50]
    hold_periods = [30, 90, 180, dte]

    for move_pct in stock_moves:
        S_new = S * (1 + move_pct / 100)
        for hold_days in hold_periods:
            if hold_days > dte:
                continue
            T_remaining = (dte - hold_days) / 365.0
            new_call = bs_call_price(S_new, K, T_remaining, r, sigma)
            leaps_return_pct = (new_call / call_price - 1) * 100
            stock_return_pct = move_pct

            key = f"move_{move_pct:+d}pct_hold_{hold_days}d"
            scenarios[key] = {
                "stock_move_pct": move_pct,
                "hold_days": hold_days,
                "stock_return_pct": round(stock_return_pct, 1),
                "leaps_return_pct": round(leaps_return_pct, 1),
                "leaps_value": round(new_call * 100, 2),
                "stock_value_100sh": round(S_new * 100, 2),
                "leverage_realized": round(leaps_return_pct / stock_return_pct, 2)
                if stock_return_pct != 0 else 0,
            }

    result["scenarios"] = scenarios

    # Break-even analysis: how much does stock need to rise to offset theta?
    # After holding the full DTE, option expires at intrinsic
    breakeven_price = K + call_price
    breakeven_move_pct = (breakeven_price / S - 1) * 100
    result["breakeven_price"] = round(breakeven_price, 2)
    result["breakeven_move_pct"] = round(breakeven_move_pct, 1)

    return result


def print_leaps_analysis(analysis):
    """Pretty print LEAPS analysis for a single ticker."""
    a = analysis
    ticker = a["ticker"]

    print(f"\n{'='*60}")
    print(f"LEAPS ANALYSIS: {ticker}")
    print(f"{'='*60}")
    print(f"  Stock Price:     ${a['stock_price']:.2f}")
    print(f"  Strike:          ${a['strike']:.2f} ({a['dte']} DTE)")
    print(f"  Call Price:      ${a['call_price']:.2f} per share")
    print(f"  Contract Cost:   ${a['contract_cost']:.2f}")
    print(f"  Intrinsic:       ${a['intrinsic_value']:.2f}")
    print(f"  Time Value:      ${a['time_value']:.2f}")
    print(f"  Delta:           {a['delta']:.3f}")
    print(f"  Theta/Day:       ${a['theta_per_day']:.4f} ({a['theta_pct_per_day']:.3f}%/day)")
    print(f"  Vega:            ${a['vega']:.4f} per 1% IV")
    print(f"  Leverage:        {a['leverage_ratio']:.1f}x")
    print(f"  Break-even:      ${a['breakeven_price']:.2f} ({a['breakeven_move_pct']:+.1f}%)")
    affordable = "YES" if a["affordable"] else f"NO (need ${a['contract_cost']:.0f})"
    print(f"  Affordable:      {affordable}")

    # Scenario table
    print(f"\n  Return Scenarios (LEAPS vs Stock):")
    print(f"  {'Move':>8} | {'30d LEAPS':>12} | {'90d LEAPS':>12} | {'180d LEAPS':>12} | {'Expiry':>12}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for move in [-30, -20, -10, 0, 10, 20, 30, 50]:
        row = f"  {move:+d}%".ljust(10) + " |"
        for hold in [30, 90, 180, a["dte"]]:
            key = f"move_{move:+d}pct_hold_{hold}d"
            if key in a["scenarios"]:
                val = a["scenarios"][key]["leaps_return_pct"]
                row += f" {val:+.1f}%".rjust(13) + " |"
            else:
                row += "         N/A |"
        print(row)

    return a


# ─────────────────────────────────────────────────────
# Poor Man's Covered Call (PMCC) Analysis
# ─────────────────────────────────────────────────────

def analyze_pmcc(ticker_data, leaps_dte=365, short_dte=30):
    """
    Model a Poor Man's Covered Call:
    - BUY LEAPS call (0.75 delta, 12 months)
    - SELL monthly call (0.30 delta, 30 DTE)

    Returns monthly income estimate and total strategy metrics.
    """
    S = ticker_data["price"]
    sigma = ticker_data["hist_vol"] / 100.0
    T_long = leaps_dte / 365.0
    T_short = short_dte / 365.0
    r = RISK_FREE_RATE

    # Long LEAPS (0.75 delta)
    K_long = find_strike_for_delta(S, T_long, r, sigma, 0.75)
    if S < 20:
        K_long = round(K_long * 2) / 2
    else:
        K_long = round(K_long)
    long_price = bs_call_price(S, K_long, T_long, r, sigma)
    long_delta = bs_delta(S, K_long, T_long, r, sigma)
    long_cost = long_price * 100

    # Short monthly call (0.30 delta)
    K_short = find_strike_for_delta(S, T_short, r, sigma, 0.30)
    if S < 20:
        K_short = round(K_short * 2) / 2
    else:
        K_short = round(K_short)
    short_price = bs_call_price(S, K_short, T_short, r, sigma)
    short_delta = bs_delta(S, K_short, T_short, r, sigma)
    short_income = short_price * 100

    # Monthly income as % of LEAPS cost
    monthly_yield = (short_income / long_cost * 100) if long_cost > 0 else 0

    # Net debit
    net_debit = long_cost  # We receive short premium but it's ongoing

    # How many months of income to break even on time value?
    time_value = long_cost - max(S - K_long, 0) * 100
    months_to_breakeven = time_value / short_income if short_income > 0 else 999

    # Model 6 months of PMCC
    total_premium_collected = 0
    months = min(int(leaps_dte / 30), 12)
    monthly_premiums = []

    for month in range(months):
        days_elapsed = month * 30
        T_remaining = (leaps_dte - days_elapsed) / 365.0
        if T_remaining <= 0:
            break

        # Re-price short call each month (stock flat scenario)
        T_short_remaining = short_dte / 365.0
        K_short_month = find_strike_for_delta(S, T_short_remaining, r, sigma, 0.30)
        if S < 20:
            K_short_month = round(K_short_month * 2) / 2
        else:
            K_short_month = round(K_short_month)
        premium = bs_call_price(S, K_short_month, T_short_remaining, r, sigma) * 100
        monthly_premiums.append(premium)
        total_premium_collected += premium

    # LEAPS value at end (stock flat)
    T_end = max((leaps_dte - months * 30) / 365.0, 0)
    leaps_end_value = bs_call_price(S, K_long, T_end, r, sigma) * 100
    leaps_decay = long_cost - leaps_end_value

    # Net P&L (stock flat)
    net_pnl_flat = total_premium_collected - leaps_decay

    result = {
        "ticker": ticker_data["ticker"],
        "stock_price": S,
        "leaps_strike": K_long,
        "leaps_dte": leaps_dte,
        "leaps_delta": round(long_delta, 3),
        "leaps_cost": round(long_cost, 2),
        "short_strike": K_short,
        "short_dte": short_dte,
        "short_delta": round(short_delta, 3),
        "short_premium": round(short_income, 2),
        "monthly_yield_pct": round(monthly_yield, 2),
        "months_to_breakeven_time_value": round(months_to_breakeven, 1),
        "total_premium_collected": round(total_premium_collected, 2),
        "leaps_theta_decay": round(leaps_decay, 2),
        "net_pnl_flat_scenario": round(net_pnl_flat, 2),
        "annualized_yield_pct": round(monthly_yield * 12, 1),
        "affordable": long_cost <= ACCOUNT_SIZE,
        "max_risk": round(long_cost, 2),  # Can only lose the LEAPS cost
        "monthly_premiums": [round(p, 2) for p in monthly_premiums],
    }

    return result


def print_pmcc_analysis(pmcc):
    """Pretty print PMCC analysis."""
    p = pmcc
    print(f"\n{'='*60}")
    print(f"POOR MAN'S COVERED CALL: {p['ticker']}")
    print(f"{'='*60}")
    print(f"  LONG LEG (LEAPS):")
    print(f"    Strike: ${p['leaps_strike']:.2f} | {p['leaps_dte']} DTE | Delta: {p['leaps_delta']:.3f}")
    print(f"    Cost:   ${p['leaps_cost']:.2f}")
    print()
    print(f"  SHORT LEG (Monthly):")
    print(f"    Strike: ${p['short_strike']:.2f} | {p['short_dte']} DTE | Delta: {p['short_delta']:.3f}")
    print(f"    Premium: ${p['short_premium']:.2f} per month")
    print()
    print(f"  STRATEGY METRICS:")
    print(f"    Monthly Yield:       {p['monthly_yield_pct']:.2f}%")
    print(f"    Annualized Yield:    {p['annualized_yield_pct']:.1f}%")
    print(f"    Time Value Breakeven: {p['months_to_breakeven_time_value']:.1f} months")
    print(f"    Max Risk:            ${p['max_risk']:.2f}")
    affordable = "YES" if p["affordable"] else f"NO (need ${p['leaps_cost']:.0f})"
    print(f"    Affordable ($441):   {affordable}")
    print()
    print(f"  FLAT STOCK SCENARIO ({len(p['monthly_premiums'])} months):")
    print(f"    Total Premiums:      ${p['total_premium_collected']:.2f}")
    print(f"    LEAPS Theta Decay:  -${p['leaps_theta_decay']:.2f}")
    print(f"    Net P&L:             ${p['net_pnl_flat_scenario']:+.2f}")


# ─────────────────────────────────────────────────────
# LEAPS vs Shares comparison
# ─────────────────────────────────────────────────────

def leaps_vs_shares_comparison(ticker_data, dte=365):
    """Compare LEAPS ownership vs buying shares with the same $441."""
    S = ticker_data["price"]
    sigma = ticker_data["hist_vol"] / 100.0
    T = dte / 365.0
    r = RISK_FREE_RATE

    # Shares approach: buy fractional shares with $441
    shares_bought = ACCOUNT_SIZE / S
    share_exposure = ACCOUNT_SIZE

    # LEAPS approach
    K = find_strike_for_delta(S, T, r, sigma, 0.75)
    if S < 20:
        K = round(K * 2) / 2
    else:
        K = round(K)
    call_price = bs_call_price(S, K, T, r, sigma)
    contract_cost = call_price * 100

    if contract_cost > ACCOUNT_SIZE:
        return {
            "ticker": ticker_data["ticker"],
            "feasible": False,
            "reason": f"LEAPS costs ${contract_cost:.0f}, exceeds ${ACCOUNT_SIZE:.0f} budget",
        }

    leaps_exposure = S * 100  # Controls 100 shares
    leverage = leaps_exposure / contract_cost
    remaining_cash = ACCOUNT_SIZE - contract_cost

    comparison = {
        "ticker": ticker_data["ticker"],
        "feasible": True,
        "stock_price": S,
        "shares_approach": {
            "shares_bought": round(shares_bought, 2),
            "exposure": round(share_exposure, 2),
            "leverage": 1.0,
        },
        "leaps_approach": {
            "strike": K,
            "dte": dte,
            "contract_cost": round(contract_cost, 2),
            "exposure": round(leaps_exposure, 2),
            "leverage": round(leverage, 2),
            "remaining_cash": round(remaining_cash, 2),
        },
        "scenarios": {},
    }

    for move_pct in [-20, -10, 0, 10, 20, 30, 50]:
        S_new = S * (1 + move_pct / 100)

        # Shares P&L
        shares_pnl = shares_bought * (S_new - S)
        shares_return = move_pct

        # LEAPS P&L at expiry
        leaps_value_expiry = max(S_new - K, 0) * 100
        leaps_pnl = leaps_value_expiry - contract_cost
        leaps_return = (leaps_pnl / contract_cost) * 100

        # LEAPS P&L at 6 months (still has time value)
        T_half = T / 2
        leaps_value_6m = bs_call_price(S_new, K, T_half, r, sigma) * 100
        leaps_pnl_6m = leaps_value_6m - contract_cost
        leaps_return_6m = (leaps_pnl_6m / contract_cost) * 100

        comparison["scenarios"][f"{move_pct:+d}%"] = {
            "shares_pnl": round(shares_pnl, 2),
            "shares_return_pct": round(shares_return, 1),
            "leaps_pnl_expiry": round(leaps_pnl, 2),
            "leaps_return_expiry_pct": round(leaps_return, 1),
            "leaps_pnl_6m": round(leaps_pnl_6m, 2),
            "leaps_return_6m_pct": round(leaps_return_6m, 1),
        }

    return comparison


def print_comparison(comp):
    """Pretty print LEAPS vs shares comparison."""
    if not comp.get("feasible"):
        print(f"\n  {comp['ticker']}: {comp.get('reason', 'Not feasible')}")
        return

    c = comp
    print(f"\n{'='*60}")
    print(f"LEAPS vs SHARES: {c['ticker']} (${ACCOUNT_SIZE} budget)")
    print(f"{'='*60}")
    print(f"  SHARES: Buy {c['shares_approach']['shares_bought']:.1f} shares @ ${c['stock_price']:.2f}")
    print(f"  LEAPS:  Buy 1 contract @ ${c['leaps_approach']['contract_cost']:.2f} "
          f"(${c['leaps_approach']['strike']:.2f} strike, {c['leaps_approach']['dte']} DTE)")
    print(f"  LEAPS exposure: ${c['leaps_approach']['exposure']:.0f} "
          f"({c['leaps_approach']['leverage']:.1f}x leverage)")
    print(f"  Cash remaining: ${c['leaps_approach']['remaining_cash']:.2f}")
    print()
    print(f"  {'Move':>8} | {'Shares P&L':>12} {'Shares %':>10} | {'LEAPS 6m':>12} {'LEAPS %':>10} | {'LEAPS Exp':>12} {'LEAPS %':>10}")
    print(f"  {'-'*8}-+-{'-'*12}-{'-'*10}-+-{'-'*12}-{'-'*10}-+-{'-'*12}-{'-'*10}")

    for move in ["-20%", "-10%", "+0%", "+10%", "+20%", "+30%", "+50%"]:
        s = c["scenarios"][move]
        print(
            f"  {move:>8} | "
            f"${s['shares_pnl']:>+10.2f} {s['shares_return_pct']:>+8.1f}% | "
            f"${s['leaps_pnl_6m']:>+10.2f} {s['leaps_return_6m_pct']:>+8.1f}% | "
            f"${s['leaps_pnl_expiry']:>+10.2f} {s['leaps_return_expiry_pct']:>+8.1f}%"
        )


# ─────────────────────────────────────────────────────
# Actionable recommendations
# ─────────────────────────────────────────────────────

def generate_recommendations(candidates, leaps_analyses, pmcc_analyses, comparisons):
    """Generate final actionable recommendations."""
    recs = []

    for ticker_data in candidates:
        ticker = ticker_data["ticker"]
        leaps = next((a for a in leaps_analyses if a["ticker"] == ticker), None)
        pmcc = next((p for p in pmcc_analyses if p["ticker"] == ticker), None)
        comp = next((c for c in comparisons if c["ticker"] == ticker), None)

        if leaps is None:
            continue

        affordable = leaps["contract_cost"] <= ACCOUNT_SIZE

        rec = {
            "ticker": ticker,
            "stock_price": ticker_data["price"],
            "score": ticker_data["score"],
            "sector": ticker_data.get("sector", "Unknown"),
            "momentum_3m": ticker_data["mom_3m"],
            "momentum_6m": ticker_data["mom_6m"],
            "hist_vol": ticker_data["hist_vol"],
            "leaps_strike": leaps["strike"],
            "leaps_cost": leaps["contract_cost"],
            "leaps_delta": leaps["delta"],
            "leverage": leaps["leverage_ratio"],
            "breakeven_move": leaps["breakeven_move_pct"],
            "theta_pct_day": leaps["theta_pct_per_day"],
            "affordable": affordable,
            "pmcc_monthly_yield": pmcc["monthly_yield_pct"] if pmcc else None,
            "pmcc_annualized": pmcc["annualized_yield_pct"] if pmcc else None,
            "pmcc_affordable": pmcc["affordable"] if pmcc else False,
        }

        # Recommendation tier
        if affordable and ticker_data["score"] >= 5:
            rec["tier"] = "A - Strong Buy LEAPS"
        elif affordable and ticker_data["score"] >= 3:
            rec["tier"] = "B - Buy LEAPS"
        elif not affordable and ticker_data["score"] >= 5:
            rec["tier"] = "C - Watch (save up)"
        else:
            rec["tier"] = "D - Monitor"

        recs.append(rec)

    # Sort by tier then score
    tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    recs.sort(key=lambda x: (tier_order.get(x["tier"][0], 9), -x["score"]))

    return recs


def print_recommendations(recs):
    """Print final recommendations."""
    print(f"\n{'='*70}")
    print(f"ACTIONABLE LEAPS RECOMMENDATIONS (Budget: ${ACCOUNT_SIZE})")
    print(f"{'='*70}")

    current_tier = None
    for r in recs:
        if r["tier"] != current_tier:
            current_tier = r["tier"]
            print(f"\n--- {current_tier} ---")

        affordable_tag = "OK" if r["affordable"] else f"${r['leaps_cost']:.0f}"
        pmcc_tag = f"{r['pmcc_annualized']:.0f}%/yr" if r.get("pmcc_annualized") else "N/A"

        print(
            f"  {r['ticker']:<7} ${r['stock_price']:>6.2f} | "
            f"LEAPS ${r['leaps_cost']:>6.0f} (d={r['leaps_delta']:.2f}) | "
            f"Lev: {r['leverage']:.1f}x | "
            f"BE: {r['breakeven_move']:+.1f}% | "
            f"PMCC: {pmcc_tag} | "
            f"Budget: {affordable_tag}"
        )

    # Summary stats
    affordable_recs = [r for r in recs if r["affordable"]]
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total candidates analyzed: {len(recs)}")
    print(f"  Affordable (< ${ACCOUNT_SIZE}): {len(affordable_recs)}")

    if affordable_recs:
        best = affordable_recs[0]
        print(f"\n  TOP PICK: {best['ticker']}")
        print(f"    Stock: ${best['stock_price']:.2f}")
        print(f"    LEAPS: ${best['leaps_cost']:.0f} ({best['leaps_delta']:.2f} delta)")
        print(f"    Leverage: {best['leverage']:.1f}x your capital")
        print(f"    Break-even: stock needs {best['breakeven_move']:+.1f}%")
        if best.get("pmcc_annualized"):
            print(f"    PMCC yield: ~{best['pmcc_annualized']:.0f}% annualized")
    else:
        print(f"\n  No affordable LEAPS under ${ACCOUNT_SIZE}.")
        print(f"  Cheapest options (save up or wait for pullback):")
        cheapest = sorted(recs, key=lambda x: x["leaps_cost"])[:3]
        for c in cheapest:
            print(f"    {c['ticker']}: ${c['leaps_cost']:.0f}")


# ─────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────

def run_full_analysis(tickers=None, top_n=20, specific_ticker=None):
    """Run the complete LEAPS research pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if specific_ticker:
        # Single ticker deep dive
        print(f"\nDeep dive: {specific_ticker}")
        data = get_stock_data(specific_ticker)
        if data is None:
            print(f"Could not fetch data for {specific_ticker}")
            return

        data["score"] = score_candidate(data)
        candidates = [data]
    else:
        # Full screening
        candidates = screen_candidates(tickers, top_n=top_n)

    if not candidates:
        print("No candidates found.")
        return

    # LEAPS analysis for each candidate
    print(f"\n{'='*60}")
    print("MODELING LEAPS RETURNS...")
    print(f"{'='*60}")

    leaps_analyses = []
    pmcc_analyses = []
    comparisons = []

    for cand in candidates:
        try:
            # LEAPS at 12 months
            leaps = model_leaps_returns(cand, dte=365)
            leaps_analyses.append(leaps)
            print_leaps_analysis(leaps)

            # PMCC
            pmcc = analyze_pmcc(cand, leaps_dte=365, short_dte=30)
            pmcc_analyses.append(pmcc)
            print_pmcc_analysis(pmcc)

            # Comparison
            comp = leaps_vs_shares_comparison(cand, dte=365)
            comparisons.append(comp)
            print_comparison(comp)

        except Exception as e:
            print(f"  [ERROR] {cand['ticker']}: {e}")

    # Generate recommendations
    recs = generate_recommendations(candidates, leaps_analyses, pmcc_analyses, comparisons)
    print_recommendations(recs)

    # Save results
    output = {
        "timestamp": timestamp,
        "account_size": ACCOUNT_SIZE,
        "candidates": candidates,
        "leaps_analyses": leaps_analyses,
        "pmcc_analyses": pmcc_analyses,
        "comparisons": comparisons,
        "recommendations": recs,
    }

    output_file = OUTPUT_DIR / f"leaps_research_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    # Also save a latest symlink-style file
    latest_file = OUTPUT_DIR / "leaps_research_latest.json"
    with open(latest_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save recommendations as CSV for easy viewing
    if recs:
        recs_df = pd.DataFrame(recs)
        csv_file = OUTPUT_DIR / f"leaps_recommendations_{timestamp}.csv"
        recs_df.to_csv(csv_file, index=False)
        print(f"Recommendations CSV: {csv_file}")

        latest_csv = OUTPUT_DIR / "leaps_recommendations_latest.csv"
        recs_df.to_csv(latest_csv, index=False)

    return output


def main():
    parser = argparse.ArgumentParser(description="LEAPS Growth Strategy Research")
    parser.add_argument("--quick", action="store_true", help="Quick scan, top 10 only")
    parser.add_argument("--ticker", type=str, help="Analyze a specific ticker")
    parser.add_argument("--top", type=int, default=20, help="Number of top candidates")
    args = parser.parse_args()

    if args.ticker:
        run_full_analysis(specific_ticker=args.ticker.upper())
    elif args.quick:
        run_full_analysis(top_n=10)
    else:
        run_full_analysis(top_n=args.top)


if __name__ == "__main__":
    main()
