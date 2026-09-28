#!/usr/bin/env python3
"""
Robinhood Real-Price Validation Pipeline — V10 Bull Call Spread Pricing
========================================================================

PURPOSE
-------
Compare Black-Scholes estimated option prices against real Robinhood market
prices for the V10 strategy universe (sector ETFs + SPY).

CONTEXT
-------
Our calibration study (bs_pricing_calibration_v1) found BS underprices real
option spreads by ~73% median (market_mid / bs_price = 1.73). This script
provides the framework to validate that finding using live Robinhood MCP data
and measure the current gap for each sector ETF.

V10 STRATEGY PARAMETERS
------------------------
  - Tickers: 11 sector ETFs + SPY
  - Structure: Bull call spread, 4% OTM long leg, ~3% spread width
  - Target DTE: ~28 days (tolerance ±5)
  - Entry cost guard: < 50% of spread width

USAGE
-----
This script is a framework — the Robinhood MCP tools are called by the Claude
agent, not directly. Feed in chain data from MCP calls.

  # Step 1: Claude fetches chain data via MCP (see FETCH_INSTRUCTIONS below)
  # Step 2: Build chain_data dict from MCP responses
  # Step 3: Run:
      from scripts.rh_price_validator import run_full_validation
      results = run_full_validation(chain_data_dict, spot_price_dict)

FETCH_INSTRUCTIONS
------------------
For each ticker in V10_TICKERS, the agent should call:
  1. get_equity_quotes(ticker) → spot_price
  2. get_option_chains(ticker) → expiry dates and strikes
  3. get_option_instruments(ticker, expiry_date, option_type="call") → filter
  4. get_option_quotes(instrument_urls) → real bid/ask/mid prices

OUTPUTS
-------
  - Console: formatted comparison table
  - state/rh_price_validation.json: machine-readable results

Author: Claude (Sonnet 4.6)
Date: 2026-07-28
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
BASE = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(BASE))

from research.tools.options_pricer import (
    RISK_FREE_RATE,
    bs_call_price,
    estimate_iv,
)

OUTPUT_PATH = BASE / "state" / "rh_price_validation.json"

# ── V10 Strategy Constants ────────────────────────────────────────────────────
V10_TICKERS = [
    "XLK", "XLV", "XLE", "XLF", "XLI",
    "XLC", "XLY", "XLP", "XLU", "XLB",
    "XLRE", "SPY",
]

# V10 structural parameters (must match backtest)
OTM_PCT        = 0.04    # Long call is 4% OTM
WIDTH_FLOOR_PCT = 0.03   # Spread width is ~3% of spot, floored at $3
WIDTH_FLOOR_ABS = 3.0    # Minimum spread width in dollars
TARGET_DTE     = 28      # Target days to expiration
DTE_TOLERANCE  = 5       # Accept DTE in [23, 33]
MAX_CWR        = 0.50    # Max cost-to-width ratio (entry cost / width)
COMMISSION_RT  = 2.60    # Round-trip commission per spread ($0.65 x 4 legs)

# From calibration study: prior known BS underpricing ratio (median)
CALIBRATION_PRIOR_RATIO = 1.727  # market_mid / bs_price from bs_pricing_calibration_v1


# =============================================================================
# CORE BS FUNCTIONS
# =============================================================================

def bs_estimate_spread(
    spot: float,
    strike_long: float,
    strike_short: float,
    dte: int,
    iv: float,
    r: float = RISK_FREE_RATE,
) -> dict[str, float]:
    """
    Estimate bull call spread price using Black-Scholes.

    Parameters
    ----------
    spot          : Current underlying price
    strike_long   : Long call strike (lower, typically ATM + 4%)
    strike_short  : Short call strike (upper, strike_long + spread_width)
    dte           : Days to expiration
    iv            : Implied volatility (annualized, e.g., 0.25 = 25%)
    r             : Risk-free rate

    Returns
    -------
    dict with:
        bs_long_leg    : BS price of long call leg
        bs_short_leg   : BS price of short call leg
        bs_spread_fair : BS fair value of spread (long - short)
        bs_spread_15pct: BS spread with current 15% haircut (what backtest uses)
        spread_width   : K2 - K1
        moneyness_long : (K1 - spot) / spot
        sigma_used     : IV passed in
    """
    T = dte / 365.0
    bs_long  = bs_call_price(spot, strike_long,  T, r, iv)
    bs_short = bs_call_price(spot, strike_short, T, r, iv)

    fair_value  = max(bs_long - bs_short, 0.001)
    entry_15pct = fair_value * 1.15   # current backtest haircut

    return {
        "bs_long_leg":     round(bs_long,      4),
        "bs_short_leg":    round(bs_short,     4),
        "bs_spread_fair":  round(fair_value,   4),
        "bs_spread_15pct": round(entry_15pct,  4),
        "spread_width":    round(strike_short - strike_long, 4),
        "moneyness_long":  round((strike_long - spot) / spot, 4),
        "sigma_used":      round(iv, 4),
    }


def estimate_iv_from_atr(atr: float, spot: float, vix: float = 20.0) -> float:
    """Wrapper around the canonical ATR-based IV estimator."""
    return estimate_iv(atr=atr, spot=spot, vix=vix)


def v10_strikes(spot: float) -> tuple[float, float]:
    """
    Compute V10 target strikes for a bull call spread.

    Returns (K1, K2):
        K1 = round(spot * 1.04, 2)       — 4% OTM long leg
        K2 = K1 + max($3, K1 * 3%)      — ~3% spread width from K1
    """
    K1 = round(spot * (1.0 + OTM_PCT), 2)
    width = max(WIDTH_FLOOR_ABS, K1 * WIDTH_FLOOR_PCT)
    K2 = round(K1 + width, 2)
    return K1, K2


# =============================================================================
# CHAIN DATA PARSING (handles Robinhood MCP response format)
# =============================================================================

def parse_rh_option_quote(quote: dict[str, Any]) -> dict[str, Any] | None:
    """
    Parse a single option quote from Robinhood MCP response.

    Robinhood get_option_quotes returns records with fields:
        ask_price, bid_price, last_trade_price, implied_volatility,
        delta, gamma, theta, vega, rho,
        instrument (URL with strike/expiry encoded),
        expiration_date, strike_price, type (call/put)

    Returns a cleaned dict or None if the quote is invalid.
    """
    try:
        bid = float(quote.get("bid_price") or 0)
        ask = float(quote.get("ask_price") or 0)

        # RH sometimes returns null for very OTM options during off-hours
        if bid <= 0 or ask <= 0:
            return None

        mid = (bid + ask) / 2.0

        # Filter out absurdly wide spreads (>3x mid) — data quality guard
        if (ask - bid) > mid * 3.0:
            return None

        iv_raw = quote.get("implied_volatility")
        iv = float(iv_raw) if iv_raw else None

        strike = float(quote.get("strike_price") or 0)
        expiry = quote.get("expiration_date", "")  # "YYYY-MM-DD"

        if strike <= 0 or not expiry:
            return None

        return {
            "bid":             round(bid, 4),
            "ask":             round(ask, 4),
            "mid":             round(mid, 4),
            "iv":              round(iv, 4) if iv else None,
            "strike":          strike,
            "expiration_date": expiry,
            "option_type":     quote.get("type", "call"),
        }
    except (TypeError, ValueError):
        return None


def find_best_expiry(
    available_expiries: list[str],
    target_dte: int = TARGET_DTE,
    tolerance: int = DTE_TOLERANCE,
    as_of: date | None = None,
) -> str | None:
    """
    Find the expiry closest to target_dte within tolerance.

    Parameters
    ----------
    available_expiries : list of "YYYY-MM-DD" strings from RH chain
    target_dte         : desired DTE (default 28)
    tolerance          : accept DTE in [target ± tolerance]
    as_of              : reference date (default today)

    Returns best expiry string, or None if none within tolerance.
    """
    today = as_of or date.today()
    best_expiry = None
    best_diff = float("inf")

    for exp_str in available_expiries:
        try:
            exp_date = date.fromisoformat(exp_str)
            dte = (exp_date - today).days
            if dte < 0:
                continue  # already expired
            if abs(dte - target_dte) <= tolerance:
                if abs(dte - target_dte) < best_diff:
                    best_diff = abs(dte - target_dte)
                    best_expiry = exp_str
        except (ValueError, TypeError):
            continue

    return best_expiry


def find_closest_strike(
    quotes: list[dict],
    target_strike: float,
    option_type: str = "call",
    max_pct_diff: float = 0.05,  # reject if >5% away from target
) -> dict | None:
    """
    From a list of parsed option quotes, find the one closest to target_strike.

    Parameters
    ----------
    quotes         : list of dicts from parse_rh_option_quote()
    target_strike  : desired strike price
    option_type    : "call" or "put"
    max_pct_diff   : reject if best match is farther than this fraction from target

    Returns closest valid quote or None.
    """
    matching = [q for q in quotes if q and q.get("option_type") == option_type]
    if not matching:
        return None

    best = min(matching, key=lambda q: abs(q["strike"] - target_strike))
    pct_diff = abs(best["strike"] - target_strike) / max(target_strike, 1.0)

    if pct_diff > max_pct_diff:
        return None  # no strike close enough — likely wrong expiry or thin chain

    return best


# =============================================================================
# CORE COMPARISON FUNCTION
# =============================================================================

def validate_spread_pricing(
    ticker: str,
    chain_data: dict[str, Any],
    spot_price: float,
    vix: float = 20.0,
    atr: float | None = None,
    as_of: date | None = None,
) -> dict[str, Any] | None:
    """
    Compare BS estimated spread price to real Robinhood market price.

    This is the primary function to call with MCP data.

    Parameters
    ----------
    ticker      : e.g. "XLK"
    chain_data  : dict from Robinhood MCP — expected structure:
        {
            "expiration_dates": ["2026-08-15", "2026-08-22", ...],
            "quotes": [
                {
                    "bid_price": "1.23", "ask_price": "1.27",
                    "strike_price": "45.00", "expiration_date": "2026-08-15",
                    "type": "call", "implied_volatility": "0.2134", ...
                },
                ...
            ]
        }
        OR for pre-filtered single-expiry chains:
        {
            "expiration_dates": ["2026-08-15"],
            "quotes": [...]  # all already filtered to one expiry
        }
    spot_price  : Current stock price from get_equity_quotes()
    vix         : Current VIX level (default 20, use live value when available)
    atr         : 14-day ATR of underlying (optional; used if market IV unavailable)
    as_of       : Reference date for DTE calculation (default today)

    Returns
    -------
    dict with full comparison, or None if spread cannot be constructed.
    Keys:
        ticker, spot, expiry, dte, strike_long, strike_short, spread_width,
        rh_long_mid, rh_short_mid, rh_spread_mid,
        bs_fair, bs_15pct, bs_iv_used,
        rh_long_iv, rh_short_iv,
        ratio_fair (rh / bs_fair), ratio_15pct (rh / bs_15pct),
        pct_error_fair, pct_error_15pct,
        cost_width_ratio_rh,
        is_tradeable (bool: RH spread passes CWR filter),
        status ("ok" | error reason string)
    """
    today = as_of or date.today()
    result_base = {"ticker": ticker, "spot": spot_price, "as_of": str(today)}

    if spot_price <= 0:
        return {**result_base, "status": "error: invalid spot price"}

    # ── 1. Find best expiry ────────────────────────────────────────────
    available_expiries = chain_data.get("expiration_dates", [])
    if not available_expiries:
        # Try to infer from quotes
        available_expiries = list({
            q.get("expiration_date", "")
            for q in chain_data.get("quotes", [])
            if q.get("expiration_date")
        })

    best_expiry = find_best_expiry(available_expiries, TARGET_DTE, DTE_TOLERANCE, today)
    if not best_expiry:
        return {**result_base, "status": f"error: no expiry within {DTE_TOLERANCE}d of {TARGET_DTE} DTE"}

    dte = (date.fromisoformat(best_expiry) - today).days

    # ── 2. Compute V10 target strikes ─────────────────────────────────
    K1_target, K2_target = v10_strikes(spot_price)

    # ── 3. Parse and filter quotes to chosen expiry ───────────────────
    raw_quotes = chain_data.get("quotes", [])
    parsed = []
    for q in raw_quotes:
        # Filter to chosen expiry (some chain_data may be pre-filtered)
        if q.get("expiration_date") and q["expiration_date"] != best_expiry:
            continue
        cleaned = parse_rh_option_quote(q)
        if cleaned:
            parsed.append(cleaned)

    if not parsed:
        return {**result_base, "status": "error: no valid quotes after parsing",
                "expiry": best_expiry, "dte": dte}

    # ── 4. Find strikes closest to V10 targets ────────────────────────
    long_quote  = find_closest_strike(parsed, K1_target, "call")
    short_quote = find_closest_strike(parsed, K2_target, "call")

    if long_quote is None:
        return {**result_base, "status": f"error: no long call near K1={K1_target:.2f}",
                "expiry": best_expiry, "dte": dte}
    if short_quote is None:
        return {**result_base, "status": f"error: no short call near K2={K2_target:.2f}",
                "expiry": best_expiry, "dte": dte}

    K1_actual = long_quote["strike"]
    K2_actual = short_quote["strike"]

    if K2_actual <= K1_actual:
        return {**result_base, "status": "error: K2 <= K1 after strike selection",
                "expiry": best_expiry, "dte": dte}

    # ── 5. Real market spread price ───────────────────────────────────
    rh_long_mid  = long_quote["mid"]
    rh_short_mid = short_quote["mid"]
    rh_spread_mid = max(rh_long_mid - rh_short_mid, 0.0)

    if rh_spread_mid <= 0:
        return {**result_base, "status": "error: RH spread mid <= 0 (inverted quotes?)",
                "expiry": best_expiry, "dte": dte,
                "rh_long_mid": rh_long_mid, "rh_short_mid": rh_short_mid}

    # ── 6. Determine IV to use for BS pricing ─────────────────────────
    # Prefer market IV from long leg (most liquid); fall back to ATR estimation
    iv_source = "market_iv"
    rh_long_iv  = long_quote.get("iv")
    rh_short_iv = short_quote.get("iv")

    if rh_long_iv and rh_long_iv > 0.05:
        bs_iv = rh_long_iv
        iv_source = "rh_long_iv"
    elif rh_short_iv and rh_short_iv > 0.05:
        bs_iv = rh_short_iv
        iv_source = "rh_short_iv"
    elif atr and atr > 0:
        bs_iv = estimate_iv_from_atr(atr, spot_price, vix)
        iv_source = "atr_estimate"
    else:
        # Last resort: rough ATR estimate assuming 1.5% daily range
        atr_est = spot_price * 0.015
        bs_iv = estimate_iv_from_atr(atr_est, spot_price, vix)
        iv_source = "fallback_1.5pct"

    # ── 7. BS estimate ────────────────────────────────────────────────
    bs_result = bs_estimate_spread(
        spot=spot_price,
        strike_long=K1_actual,
        strike_short=K2_actual,
        dte=dte,
        iv=bs_iv,
    )

    bs_fair  = bs_result["bs_spread_fair"]
    bs_15pct = bs_result["bs_spread_15pct"]

    # ── 8. Comparison ratios ──────────────────────────────────────────
    ratio_fair  = rh_spread_mid / max(bs_fair,  0.001)
    ratio_15pct = rh_spread_mid / max(bs_15pct, 0.001)

    pct_error_fair  = (rh_spread_mid - bs_fair)  / rh_spread_mid * 100.0
    pct_error_15pct = (rh_spread_mid - bs_15pct) / rh_spread_mid * 100.0

    spread_width = K2_actual - K1_actual
    cwr_rh = rh_spread_mid / max(spread_width, 0.01)
    is_tradeable = cwr_rh <= MAX_CWR

    # ── 9. Contextual interpretation ─────────────────────────────────
    if ratio_fair < 0.9:
        interp = "BS OVERPRICES (unusual — check IV)"
    elif ratio_fair <= 1.20:
        interp = "BS CLOSE TO MARKET (~0-20% gap)"
    elif ratio_fair <= 1.50:
        interp = "BS UNDERPRICES moderate (~20-50% gap)"
    elif ratio_fair <= 2.00:
        interp = "BS UNDERPRICES significant (~50-100% gap)"
    else:
        interp = "BS UNDERPRICES severely (>100% gap)"

    return {
        "ticker":             ticker,
        "spot":               round(spot_price, 4),
        "as_of":              str(today),
        "expiry":             best_expiry,
        "dte":                dte,
        "strike_long":        K1_actual,
        "strike_short":       K2_actual,
        "spread_width":       round(spread_width, 4),
        "moneyness_long_pct": round((K1_actual / spot_price - 1) * 100, 2),

        # Real Robinhood prices
        "rh_long_mid":        round(rh_long_mid,  4),
        "rh_short_mid":       round(rh_short_mid, 4),
        "rh_spread_mid":      round(rh_spread_mid, 4),
        "rh_long_bid":        long_quote.get("bid"),
        "rh_long_ask":        long_quote.get("ask"),
        "rh_short_bid":       short_quote.get("bid"),
        "rh_short_ask":       short_quote.get("ask"),

        # BS estimates
        "bs_fair":            round(bs_fair,   4),
        "bs_15pct":           round(bs_15pct,  4),
        "bs_iv_used":         round(bs_iv,     4),
        "iv_source":          iv_source,
        "rh_long_iv":         round(rh_long_iv,  4) if rh_long_iv  else None,
        "rh_short_iv":        round(rh_short_iv, 4) if rh_short_iv else None,

        # Comparison
        "ratio_fair":         round(ratio_fair,  3),   # rh / bs_fair
        "ratio_15pct":        round(ratio_15pct, 3),   # rh / bs_15pct
        "pct_error_fair":     round(pct_error_fair,   1),
        "pct_error_15pct":    round(pct_error_15pct,  1),
        "prior_calib_ratio":  CALIBRATION_PRIOR_RATIO,
        "ratio_vs_prior":     round(ratio_fair / CALIBRATION_PRIOR_RATIO, 3),

        # Tradeability
        "cost_width_ratio_rh": round(cwr_rh, 3),
        "is_tradeable":        is_tradeable,

        "interpretation": interp,
        "status": "ok",
    }


# =============================================================================
# BATCH VALIDATION
# =============================================================================

def run_full_validation(
    chain_data_dict: dict[str, dict],
    spot_price_dict: dict[str, float],
    vix: float = 20.0,
    atr_dict: dict[str, float] | None = None,
    as_of: date | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """
    Run validation across all tickers and produce comparison report.

    Parameters
    ----------
    chain_data_dict : {ticker: chain_data} — one entry per V10 ticker
    spot_price_dict : {ticker: spot_price}
    vix             : Current VIX level
    atr_dict        : Optional {ticker: ATR} for IV estimation fallback
    as_of           : Reference date (default today)
    save            : Write results to OUTPUT_PATH

    Returns
    -------
    Full results dict with per-ticker comparisons + aggregate stats
    """
    today = as_of or date.today()
    atr_dict = atr_dict or {}

    results_per_ticker = {}
    errors = []

    print("\n" + "=" * 90)
    print(f"RH REAL-PRICE VALIDATION — V10 Sector ETFs — {today}")
    print(f"VIX: {vix:.1f} | Target DTE: {TARGET_DTE} ± {DTE_TOLERANCE} | OTM: {OTM_PCT*100:.0f}%")
    print("=" * 90)

    for ticker in V10_TICKERS:
        chain = chain_data_dict.get(ticker)
        spot  = spot_price_dict.get(ticker, 0.0)
        atr   = atr_dict.get(ticker)

        if chain is None:
            errors.append({"ticker": ticker, "status": "error: no chain data provided"})
            print(f"  {ticker:6s}  SKIP — no chain data")
            continue

        result = validate_spread_pricing(
            ticker=ticker,
            chain_data=chain,
            spot_price=spot,
            vix=vix,
            atr=atr,
            as_of=today,
        )

        if result is None or result.get("status", "").startswith("error"):
            msg = result.get("status", "unknown error") if result else "null result"
            errors.append({"ticker": ticker, "status": msg})
            print(f"  {ticker:6s}  ERROR — {msg}")
        else:
            results_per_ticker[ticker] = result

    # ── Print comparison table ─────────────────────────────────────────
    ok_results = [r for r in results_per_ticker.values() if r["status"] == "ok"]

    if ok_results:
        print("\n" + "-" * 90)
        print(f"{'Ticker':6s}  {'Spot':>8s}  {'Expiry':12s}  {'DTE':>4s}  "
              f"{'K1':>7s}  {'K2':>7s}  {'RH Mid':>7s}  {'BS Fair':>7s}  "
              f"{'BS+15%':>7s}  {'Ratio':>6s}  {'Error%':>7s}  {'Trade?':>6s}")
        print("-" * 90)

        for r in sorted(ok_results, key=lambda x: x["ticker"]):
            tradeable = "YES" if r["is_tradeable"] else "NO"
            print(
                f"  {r['ticker']:6s}  "
                f"${r['spot']:>7.2f}  "
                f"{r['expiry']:12s}  "
                f"{r['dte']:>4d}  "
                f"${r['strike_long']:>6.2f}  "
                f"${r['strike_short']:>6.2f}  "
                f"${r['rh_spread_mid']:>6.3f}  "
                f"${r['bs_fair']:>6.3f}  "
                f"${r['bs_15pct']:>6.3f}  "
                f"{r['ratio_fair']:>6.2f}x  "
                f"{r['pct_error_fair']:>+6.1f}%  "
                f"{tradeable:>6s}"
            )

        print("-" * 90)

        # ── Aggregate statistics ──────────────────────────────────────
        ratios = [r["ratio_fair"] for r in ok_results]
        errors_pct = [r["pct_error_fair"] for r in ok_results]
        tradeable_count = sum(1 for r in ok_results if r["is_tradeable"])

        print(f"\n  Tickers validated:    {len(ok_results)}/{len(V10_TICKERS)}")
        print(f"  Median RH/BS ratio:   {np.median(ratios):.3f}x  "
              f"(prior calibration: {CALIBRATION_PRIOR_RATIO:.3f}x)")
        print(f"  Mean RH/BS ratio:     {np.mean(ratios):.3f}x")
        print(f"  Range:                {min(ratios):.2f}x — {max(ratios):.2f}x")
        print(f"  BS underprices by:    {np.median(errors_pct):.1f}% median")
        print(f"  Tradeable spreads:    {tradeable_count}/{len(ok_results)} "
              f"(CWR ≤ {MAX_CWR})")

        # Diagnosis
        median_ratio = np.median(ratios)
        print(f"\n  DIAGNOSIS:")
        if abs(median_ratio - CALIBRATION_PRIOR_RATIO) / CALIBRATION_PRIOR_RATIO < 0.15:
            print(f"  Consistent with prior calibration ({CALIBRATION_PRIOR_RATIO:.2f}x).")
            print(f"  BS haircut should be ~{(median_ratio - 1)*100:.0f}%, not 15%.")
        elif median_ratio > CALIBRATION_PRIOR_RATIO * 1.15:
            print(f"  WORSE than prior calibration — gap has WIDENED.")
            print(f"  Consider updating recommended haircut to {(median_ratio - 1)*100:.0f}%.")
        else:
            print(f"  BETTER than prior calibration — gap has NARROWED.")
            print(f"  BS haircut still needs ~{(median_ratio - 1)*100:.0f}%.")

        aggregate = {
            "n_tickers_ok":           len(ok_results),
            "n_tickers_total":        len(V10_TICKERS),
            "median_ratio_fair":      round(float(np.median(ratios)), 4),
            "mean_ratio_fair":        round(float(np.mean(ratios)),   4),
            "min_ratio":              round(float(min(ratios)),        4),
            "max_ratio":              round(float(max(ratios)),        4),
            "median_pct_error_fair":  round(float(np.median(errors_pct)), 2),
            "mean_pct_error_fair":    round(float(np.mean(errors_pct)),   2),
            "prior_calib_ratio":      CALIBRATION_PRIOR_RATIO,
            "tradeable_count":        tradeable_count,
            "implied_optimal_haircut_pct": round((float(np.median(ratios)) - 1.0) * 100, 1),
        }
    else:
        aggregate = {"n_tickers_ok": 0, "status": "no valid results"}
        print("\n  WARNING: No valid results produced.")

    # ── Build and optionally save output ─────────────────────────────────
    output = {
        "timestamp":     datetime.now().isoformat(),
        "as_of":         str(today),
        "vix_used":      vix,
        "target_dte":    TARGET_DTE,
        "otm_pct":       OTM_PCT,
        "aggregate":     aggregate,
        "per_ticker":    results_per_ticker,
        "errors":        errors,
        "v10_tickers":   V10_TICKERS,
        "bs_constants": {
            "risk_free_rate":       RISK_FREE_RATE,
            "current_haircut":      0.15,
            "prior_calib_ratio":    CALIBRATION_PRIOR_RATIO,
            "commission_rt_spread": COMMISSION_RT,
        },
    }

    if save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {OUTPUT_PATH}")

    return output


# =============================================================================
# MCP DATA HELPERS — utilities for building chain_data from MCP responses
# =============================================================================

def build_chain_data_from_mcp(
    instruments: list[dict],
    quotes: list[dict],
    expiration_dates: list[str] | None = None,
) -> dict[str, Any]:
    """
    Assemble chain_data dict from separate MCP responses.

    Typical MCP workflow:
        instruments = mcp.get_option_instruments(ticker, expiry, "call")
        quotes      = mcp.get_option_quotes([i["url"] for i in instruments])
        chain_data  = build_chain_data_from_mcp(instruments, quotes, [expiry])

    The function merges instrument metadata (strike, expiry) with quote data
    (bid, ask, IV) into the unified format expected by validate_spread_pricing().

    Parameters
    ----------
    instruments      : list of instrument dicts from get_option_instruments()
    quotes           : list of quote dicts from get_option_quotes()
    expiration_dates : list of expiry strings (optional; inferred from data if omitted)

    Returns
    -------
    chain_data dict ready for validate_spread_pricing()
    """
    # Build URL → instrument lookup for merging
    url_to_instr = {}
    for instr in instruments:
        url = instr.get("url") or instr.get("id") or ""
        if url:
            url_to_instr[url] = instr

    merged_quotes = []
    for q in quotes:
        # Find matching instrument for strike/expiry metadata
        instr_url = q.get("instrument") or q.get("instrument_url") or ""
        instr = url_to_instr.get(instr_url, {})

        # Merge: instrument data takes priority for static fields
        merged = {**q}
        if "strike_price" not in merged or not merged["strike_price"]:
            merged["strike_price"] = instr.get("strike_price")
        if "expiration_date" not in merged or not merged["expiration_date"]:
            merged["expiration_date"] = instr.get("expiration_date")
        if "type" not in merged or not merged["type"]:
            merged["type"] = instr.get("type", "call")

        merged_quotes.append(merged)

    # Infer expiration dates if not provided
    if not expiration_dates:
        expiration_dates = sorted(list({
            q.get("expiration_date", "") for q in merged_quotes
            if q.get("expiration_date")
        }))

    return {
        "expiration_dates": expiration_dates,
        "quotes": merged_quotes,
    }


def build_chain_from_flat_quotes(quotes: list[dict]) -> dict[str, Any]:
    """
    Convenience wrapper when MCP returns fully-populated quote objects
    (i.e., strike_price and expiration_date are already in each quote).

    Parameters
    ----------
    quotes : list of dicts, each with bid_price, ask_price, strike_price,
             expiration_date, type, implied_volatility, etc.

    Returns
    -------
    chain_data dict ready for validate_spread_pricing()
    """
    expiration_dates = sorted(list({
        q.get("expiration_date", "") for q in quotes
        if q.get("expiration_date")
    }))
    return {
        "expiration_dates": expiration_dates,
        "quotes": quotes,
    }


# =============================================================================
# DEMO / EXAMPLE
# =============================================================================

def _demo_with_synthetic_data() -> None:
    """
    Demonstrate the pipeline with synthetic data mimicking RH MCP responses.
    Replace with real MCP data in production.
    """
    print("\n" + "=" * 70)
    print("DEMO: Synthetic RH data (replace with live MCP calls)")
    print("=" * 70)

    # Synthetic spot prices (rough July 2026 levels)
    DEMO_SPOTS = {
        "XLK":  220.0, "XLV":  145.0, "XLE":   90.0, "XLF":   52.0,
        "XLI":  135.0, "XLC":   95.0, "XLY":  200.0, "XLP":   78.0,
        "XLU":   70.0, "XLB":   95.0, "XLRE":  43.0, "SPY":  560.0,
    }

    # Build synthetic chain data for each ticker
    # In production, these come from:
    #   get_equity_quotes(ticker) → spot
    #   get_option_chains(ticker) + get_option_instruments() + get_option_quotes()
    chain_data_dict = {}
    spot_price_dict = {}
    vix_demo = 19.2  # approximate current VIX

    target_expiry = "2026-08-22"  # ~25 DTE from 2026-07-28

    for ticker, spot in DEMO_SPOTS.items():
        K1_target, K2_target = v10_strikes(spot)
        # Simulate real market pricing: BS fair * ~1.73 (calibration ratio)
        # Use a realistic IV for sector ETFs (~20-25%)
        demo_iv = 0.22
        T = 25 / 365.0
        bs_long_fair  = bs_call_price(spot, K1_target, T, RISK_FREE_RATE, demo_iv)
        bs_short_fair = bs_call_price(spot, K2_target, T, RISK_FREE_RATE, demo_iv)

        # Simulate real market = BS * ~1.73 (as found in calibration)
        real_long_mid  = bs_long_fair  * 1.73
        real_short_mid = bs_short_fair * 1.73

        # Add realistic bid-ask spread (~5-8% for ETF options)
        ba_long  = real_long_mid  * 0.06
        ba_short = real_short_mid * 0.08

        quotes = [
            {
                "bid_price":         str(round(real_long_mid  - ba_long  / 2, 2)),
                "ask_price":         str(round(real_long_mid  + ba_long  / 2, 2)),
                "strike_price":      str(K1_target),
                "expiration_date":   target_expiry,
                "type":              "call",
                "implied_volatility": str(round(demo_iv * 1.15, 4)),  # slight IV premium
            },
            {
                "bid_price":         str(round(real_short_mid - ba_short / 2, 2)),
                "ask_price":         str(round(real_short_mid + ba_short / 2, 2)),
                "strike_price":      str(K2_target),
                "expiration_date":   target_expiry,
                "type":              "call",
                "implied_volatility": str(round(demo_iv * 1.20, 4)),
            },
        ]

        chain_data_dict[ticker] = build_chain_from_flat_quotes(quotes)
        spot_price_dict[ticker] = spot

    results = run_full_validation(
        chain_data_dict=chain_data_dict,
        spot_price_dict=spot_price_dict,
        vix=vix_demo,
        save=True,
    )

    print(f"\nDemo complete. {results['aggregate'].get('n_tickers_ok', 0)} tickers validated.")
    print(f"Implied optimal haircut: "
          f"{results['aggregate'].get('implied_optimal_haircut_pct', '?')}%")


# =============================================================================
# MAIN — Agent workflow instructions
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="V10 Real-Price Validation Pipeline (BS vs Robinhood MCP)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic data to verify the framework works"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to JSON file containing pre-fetched MCP data "
             "({tickers: {chain_data: ..., spot: ...}})"
    )
    parser.add_argument(
        "--vix", type=float, default=20.0,
        help="Current VIX level (default 20.0)"
    )
    args = parser.parse_args()

    if args.demo:
        _demo_with_synthetic_data()

    elif args.input:
        # Load pre-fetched MCP data from JSON file
        with open(args.input) as f:
            mcp_data = json.load(f)

        chain_data_dict = {}
        spot_price_dict = {}
        atr_dict = {}

        for ticker, tdata in mcp_data.items():
            chain_data_dict[ticker] = tdata.get("chain_data", {})
            spot_price_dict[ticker] = float(tdata.get("spot", 0))
            if "atr" in tdata:
                atr_dict[ticker] = float(tdata["atr"])

        run_full_validation(
            chain_data_dict=chain_data_dict,
            spot_price_dict=spot_price_dict,
            vix=args.vix,
            atr_dict=atr_dict if atr_dict else None,
            save=True,
        )

    else:
        print(__doc__)
        print("\nAgent workflow:")
        print("  1. For each ticker in V10_TICKERS, call MCP tools:")
        print("       spot  = get_equity_quotes(ticker)['last_trade_price']")
        print("       chain = get_option_chains(ticker)  → expiration_dates")
        print("       instr = get_option_instruments(ticker, expiry, 'call')")
        print("       qts   = get_option_quotes([i['url'] for i in instr])")
        print("       chain_data = build_chain_data_from_mcp(instr, qts, [expiry])")
        print("  2. Assemble dicts and call run_full_validation()")
        print("  3. Results auto-saved to state/rh_price_validation.json")
        print("\nQuick test: python scripts/rh_price_validator.py --demo")
