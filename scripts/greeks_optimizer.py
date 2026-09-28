#!/usr/bin/env python3
"""
Greeks Optimizer for Agentic Options Account
=============================================
For each signal that fires, calculates the optimal Greeks profile and recommends
the best strike/expiry combination.

Key metrics:
  - Theta/day as % of premium: penalize if > 3% daily bleed (2-5 day holds)
  - Delta efficiency: $/delta spent, target delta 0.35-0.40
  - Vega exposure: quantify vega profit if IV mean-reverts toward historical avg
  - Gamma value: higher gamma = more profit from intraday moves on 2-5 day holds

Output: Greeks score (0-100) combining all factors, recommended strike, expected
costs and upside over the hold period.

Usage:
  python3 greeks_optimizer.py --ticker XLF --direction call --dte 30
  python3 greeks_optimizer.py --ticker XLE --direction call --dte 35
  python3 greeks_optimizer.py  # analyze all current actionable signals

Data sources:
  1. Local parquet chain snapshots in data/options_chains/YYYY-MM-DD/
  2. IV rank data from state/iv_rank_data.json
  3. Falls back to Black-Scholes estimates if no chain data available
"""

import argparse
import json
import math
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/home/jupiter/Lvl3Quant")
CHAINS_DIR = BASE / "data" / "options_chains"
# Fresh Dolt-backed chain store (updated daily by scripts/data/daily_options_downloader.py).
# One parquet per ticker holding full history; filter to the latest `date`.
FRESH_CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
# Any snapshot older than this is refused outright. Stale chains silently produced
# recommendations for already-expired contracts (see 2026-09-23 stale-expiry bug).
MAX_CHAIN_AGE_DAYS = 5
IV_RANK_FILE = BASE / "state" / "iv_rank_data.json"
SIGNALS_FILE = BASE / "state" / "agentic_signals.json"
OUTPUT_FILE = BASE / "state" / "greeks_analysis.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACCOUNT_EQUITY = 706.0
MAX_POSITION_COST = 200.0
HOLD_DAYS_DEFAULT = 3       # average hold for momentum burst variant F
HOLD_DAYS_MAX = 5           # max hold
RISK_FREE_RATE = 0.043      # ~4.3% (current T-bill rate approx)

# Greeks score weights (sum to 1.0)
W_THETA_EFFICIENCY = 0.30   # penalize high theta bleed
W_DELTA_EFFICIENCY = 0.25   # reward optimal delta range
W_VEGA_UPSIDE = 0.25        # reward cheap vega (IV mean-reversion potential)
W_GAMMA_VALUE = 0.20        # reward gamma for intraday capture


# ---------------------------------------------------------------------------
# Black-Scholes pricing (fallback when no chain data)
# ---------------------------------------------------------------------------
def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call") -> float:
    """Black-Scholes option price."""
    if T <= 0:
        if option_type == "call":
            return max(0, S - K)
        return max(0, K - S)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> dict:
    """Calculate Black-Scholes Greeks."""
    if T <= 0.0001:
        T = 0.0001

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Price
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0

    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)

    # Theta per calendar day
    theta_annual = (
        -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
        - r * K * math.exp(-r * T) * (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2))
    )
    theta_daily = theta_annual / 365.0

    # Vega (per 1% IV change)
    vega = S * sqrt_T * norm.pdf(d1) / 100.0

    return {
        "price": max(0.01, price),
        "delta": delta,
        "gamma": gamma,
        "theta_daily": theta_daily,
        "vega": vega,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _normalize_chain(df: pd.DataFrame, snapshot_date: date, ticker: str) -> pd.DataFrame:
    """Map a raw chain snapshot onto the column names optimize_greeks expects.

    Handles both the fresh Dolt store (expiration/type/vol/dte) and the legacy
    dated-directory format (expiry/option_type/iv). DTE is always recomputed from
    today rather than trusted from the file -- a stored dte is only valid on the
    day the snapshot was taken.
    """
    df = df.copy()

    if "expiry" not in df.columns and "expiration" in df.columns:
        df["expiry"] = df["expiration"]

    if "option_type" not in df.columns and "type" in df.columns:
        df["option_type"] = (
            df["type"].astype(str).str.lower().str[0].map({"c": "call", "p": "put"})
        )

    if "iv" not in df.columns and "vol" in df.columns:
        df["iv"] = df["vol"]

    # Always recompute DTE against today.
    today = date.today()
    df["dte_days"] = pd.to_datetime(df["expiry"]).dt.date.map(lambda d: (d - today).days)
    df = df[df["dte_days"] > 0]

    if "underlying_price" not in df.columns:
        df["underlying_price"] = _get_approx_price(ticker)

    # The fresh store carries no OI/volume. Mark unknown (-1) rather than 0 so
    # downstream liquidity gates can tell "no data" from "no interest".
    for col in ("open_interest", "volume"):
        if col not in df.columns:
            df[col] = -1

    # Keep expiry as a plain YYYY-MM-DD string; callers stringify it directly.
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.strftime("%Y-%m-%d")

    df["snapshot_timestamp"] = str(snapshot_date)
    return df


def load_latest_chain(ticker: str) -> Optional[pd.DataFrame]:
    """Load the most recent options chain snapshot for a ticker.

    Prefers the fresh Dolt-backed store, falling back to the legacy dated
    directories. Returns None if the newest snapshot found is staler than
    MAX_CHAIN_AGE_DAYS, so callers drop to clearly-labelled synthetic pricing
    instead of quoting prices and expiries that no longer exist.
    """
    today = date.today()

    # Preferred source: fresh per-ticker parquet, filtered to its latest date.
    fresh_file = FRESH_CHAINS_DIR / f"{ticker}.parquet"
    if fresh_file.exists():
        try:
            df = pd.read_parquet(fresh_file)
            if len(df) > 0 and "date" in df.columns:
                dates = pd.to_datetime(df["date"]).dt.date
                snapshot_date = dates.max()
                age = (today - snapshot_date).days
                if age > MAX_CHAIN_AGE_DAYS:
                    print(
                        f"[chain] {ticker}: freshest snapshot {snapshot_date} is {age}d old "
                        f"(max {MAX_CHAIN_AGE_DAYS}) -- refusing stale chain",
                        file=sys.stderr,
                    )
                    return None
                subset = df[dates == snapshot_date]
                if len(subset) > 0:
                    return _normalize_chain(subset, snapshot_date, ticker)
        except Exception as e:
            print(f"[chain] {ticker}: failed reading fresh store: {e}", file=sys.stderr)

    # Legacy source: dated snapshot directories, newest first.
    if not CHAINS_DIR.exists():
        return None

    date_dirs = sorted(
        [d for d in CHAINS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")],
        reverse=True,
    )

    for date_dir in date_dirs:
        try:
            snapshot_date = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
        except ValueError:
            continue

        age = (today - snapshot_date).days
        if age > MAX_CHAIN_AGE_DAYS:
            # Directories are newest-first, so everything below is older still.
            print(
                f"[chain] {ticker}: newest legacy snapshot {snapshot_date} is {age}d old "
                f"(max {MAX_CHAIN_AGE_DAYS}) -- refusing stale chain",
                file=sys.stderr,
            )
            return None

        chain_file = date_dir / f"{ticker}.parquet"
        if chain_file.exists():
            df = pd.read_parquet(chain_file)
            if len(df) > 0:
                return _normalize_chain(df, snapshot_date, ticker)

        all_file = date_dir / "_all_tickers.parquet"
        if all_file.exists():
            df = pd.read_parquet(all_file)
            subset = df[df["ticker"] == ticker]
            if len(subset) > 0:
                return _normalize_chain(subset, snapshot_date, ticker)

    return None


def load_iv_rank_data(ticker: str) -> Optional[dict]:
    """Load IV rank data for a ticker from the IV rank tracker."""
    if not IV_RANK_FILE.exists():
        return None
    try:
        with open(IV_RANK_FILE) as f:
            data = json.load(f)
        iv_data = data.get("iv_data", data.get("tickers", {}))
        return iv_data.get(ticker)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Greeks analysis for a single strike
# ---------------------------------------------------------------------------
def analyze_strike(
    row: dict,
    underlying_price: float,
    hold_days: int = HOLD_DAYS_DEFAULT,
    iv_rank: Optional[float] = None,
    iv_year_low: Optional[float] = None,
    iv_current: Optional[float] = None,
) -> dict:
    """
    Analyze a single option strike for Greeks quality.

    Returns dict with component scores and overall greeks_score.
    """
    premium = row.get("mid") or row.get("price", 0)
    if premium <= 0.01:
        return {"greeks_score": 0, "reason": "zero premium"}

    delta = abs(row.get("delta", 0))
    gamma = abs(row.get("gamma", 0))
    theta = row.get("theta_daily", row.get("theta", 0))
    vega = abs(row.get("vega", 0))
    strike = row.get("strike", 0)
    iv = row.get("iv", 0.25)

    # Per-contract cost (premium * 100 shares)
    contract_cost = premium * 100

    # ── 1. THETA EFFICIENCY (30% weight) ──
    # Theta/day as % of premium. Lower = better.
    # Threshold: > 3% daily = too expensive for 2-5 day holds
    theta_per_day = abs(theta)
    theta_pct_of_premium = (theta_per_day / premium * 100) if premium > 0 else 99

    # Total theta cost over hold period
    theta_cost_hold = theta_per_day * hold_days
    theta_cost_pct = (theta_cost_hold / premium * 100) if premium > 0 else 99

    # Score: 100 if theta < 1% of premium/day, 0 if > 5%
    if theta_pct_of_premium <= 1.0:
        theta_score = 100
    elif theta_pct_of_premium <= 2.0:
        theta_score = 80
    elif theta_pct_of_premium <= 3.0:
        theta_score = 60
    elif theta_pct_of_premium <= 4.0:
        theta_score = 30
    elif theta_pct_of_premium <= 5.0:
        theta_score = 10
    else:
        theta_score = 0

    # ── 2. DELTA EFFICIENCY (25% weight) ──
    # Target delta 0.35-0.40 for optimal R:R on momentum trades
    # Also measure: delta per dollar spent (bang for buck)
    delta_per_dollar = (delta / contract_cost * 100) if contract_cost > 0 else 0

    # Delta range score
    if 0.35 <= delta <= 0.45:
        delta_range_score = 100  # sweet spot
    elif 0.30 <= delta <= 0.50:
        delta_range_score = 75  # acceptable
    elif 0.25 <= delta <= 0.55:
        delta_range_score = 50  # workable
    elif 0.20 <= delta <= 0.60:
        delta_range_score = 30  # suboptimal
    else:
        delta_range_score = 10  # too deep ITM or far OTM

    # Budget fit bonus: if contract is affordable and still good delta
    budget_score = 100 if contract_cost <= MAX_POSITION_COST else max(0, 100 - (contract_cost - MAX_POSITION_COST) / 2)

    delta_score = delta_range_score * 0.6 + min(100, delta_per_dollar * 200) * 0.2 + budget_score * 0.2

    # ── 3. VEGA UPSIDE (25% weight) ──
    # In cheap IV environments (which we target), vega is our friend
    # Quantify: if IV mean-reverts toward historical average, how much do we gain?
    vega_score = 50  # neutral default

    if iv_rank is not None and iv_current is not None:
        # Cheap IV = high vega score (IV likely to expand)
        if iv_rank < 20:
            # Very cheap IV — vega is a strong tailwind
            # Estimate vega profit: if IV reverts to ~30th percentile
            iv_expansion_pts = max(0, (iv_current * 1.3) - iv_current)  # ~30% expansion
            vega_profit_per_contract = vega * iv_expansion_pts * 100
            vega_profit_pct = (vega_profit_per_contract / contract_cost * 100) if contract_cost > 0 else 0
            vega_score = min(100, 70 + vega_profit_pct * 2)
        elif iv_rank < 35:
            # Cheap IV
            iv_expansion_pts = max(0, (iv_current * 1.15) - iv_current)
            vega_profit_per_contract = vega * iv_expansion_pts * 100
            vega_profit_pct = (vega_profit_per_contract / contract_cost * 100) if contract_cost > 0 else 0
            vega_score = min(100, 55 + vega_profit_pct * 2)
        elif iv_rank > 70:
            # Expensive IV — vega is a headwind (IV likely to contract)
            iv_contraction_pts = max(0, iv_current - iv_current * 0.85)
            vega_loss_per_contract = vega * iv_contraction_pts * 100
            vega_loss_pct = (vega_loss_per_contract / contract_cost * 100) if contract_cost > 0 else 0
            vega_score = max(0, 40 - vega_loss_pct * 2)
        else:
            # Normal IV
            vega_score = 50
    elif iv is not None and iv > 0:
        # No IV rank data — use raw IV level as proxy
        # Sector ETFs: IV < 15% is cheap, > 30% is expensive
        if iv < 0.15:
            vega_score = 75
        elif iv < 0.20:
            vega_score = 60
        elif iv > 0.35:
            vega_score = 25
        elif iv > 0.30:
            vega_score = 35

    # ── 4. GAMMA VALUE (20% weight) ──
    # Higher gamma = more P&L capture from intraday moves
    # Gamma is highest for ATM, short-dated options
    # Normalize: gamma * underlying_price gives dollar gamma
    dollar_gamma = gamma * underlying_price * 100  # per contract, per $1 move
    gamma_per_dollar = (dollar_gamma / contract_cost) if contract_cost > 0 else 0

    # Expected gamma profit over hold period from random moves
    # Assume ~0.5% daily moves for sector ETFs
    daily_move = underlying_price * 0.005
    expected_gamma_capture = 0.5 * gamma * (daily_move ** 2) * 100 * hold_days
    gamma_capture_pct = (expected_gamma_capture / contract_cost * 100) if contract_cost > 0 else 0

    if gamma_per_dollar > 0.05:
        gamma_score = 100
    elif gamma_per_dollar > 0.03:
        gamma_score = 75
    elif gamma_per_dollar > 0.02:
        gamma_score = 60
    elif gamma_per_dollar > 0.01:
        gamma_score = 40
    else:
        gamma_score = 20

    # ── COMPOSITE SCORE ──
    greeks_score = (
        W_THETA_EFFICIENCY * theta_score
        + W_DELTA_EFFICIENCY * delta_score
        + W_VEGA_UPSIDE * vega_score
        + W_GAMMA_VALUE * gamma_score
    )

    return {
        "strike": strike,
        "premium": round(premium, 2),
        "contract_cost": round(contract_cost, 0),
        "affordable": contract_cost <= MAX_POSITION_COST,
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta_daily": round(theta, 4),
        "vega": round(vega, 4),
        "iv": round(iv, 4) if iv else None,
        # Component scores
        "theta_score": round(theta_score, 1),
        "theta_pct_per_day": round(theta_pct_of_premium, 2),
        "theta_cost_hold_period": round(theta_cost_hold, 4),
        "theta_cost_hold_pct": round(theta_cost_pct, 1),
        "delta_score": round(delta_score, 1),
        "delta_per_dollar": round(delta_per_dollar, 4),
        "vega_score": round(vega_score, 1),
        "gamma_score": round(gamma_score, 1),
        "dollar_gamma": round(dollar_gamma, 2),
        "gamma_per_dollar": round(gamma_per_dollar, 4),
        "expected_gamma_capture_pct": round(gamma_capture_pct, 2),
        # Composite
        "greeks_score": round(greeks_score, 1),
    }


# ---------------------------------------------------------------------------
# Main optimizer: find best strike for a given ticker/direction
# ---------------------------------------------------------------------------
def optimize_greeks(
    ticker: str,
    direction: str = "call",
    target_dte_min: int = 21,
    target_dte_max: int = 45,
    hold_days: int = HOLD_DAYS_DEFAULT,
    max_cost: float = MAX_POSITION_COST,
) -> dict:
    """
    Find the optimal strike for a ticker given direction and DTE range.

    Returns analysis with recommended strike, Greeks breakdown, and score.
    """
    option_type = "call" if direction in ("call", "bull") else "put"

    # Load IV rank data
    iv_data = load_iv_rank_data(ticker)
    iv_rank = iv_data.get("iv_rank") if iv_data else None
    iv_current = iv_data.get("iv_current") if iv_data else None
    iv_year_low = iv_data.get("iv_year_low") if iv_data else None
    iv_year_high = iv_data.get("iv_year_high") if iv_data else None
    iv_trend = iv_data.get("iv_trend") if iv_data else None

    # Try loading real chain data
    chain = load_latest_chain(ticker)
    results = []

    if chain is not None and len(chain) > 0:
        # Filter for the right option type and DTE range
        df = chain[chain["option_type"] == option_type].copy()

        # Get underlying price
        underlying_price = float(df["underlying_price"].iloc[0]) if "underlying_price" in df.columns else None

        if "dte_days" in df.columns:
            df = df[(df["dte_days"] >= target_dte_min) & (df["dte_days"] <= target_dte_max)]
        elif "expiry" in df.columns:
            today = date.today()
            df["_dte"] = df["expiry"].apply(
                lambda x: (pd.Timestamp(x).date() - today).days if pd.notna(x) else -1
            )
            df = df[(df["_dte"] >= target_dte_min) & (df["_dte"] <= target_dte_max)]

        if len(df) == 0:
            # Relax DTE range
            df = chain[chain["option_type"] == option_type].copy()
            if "dte_days" in df.columns:
                df = df[(df["dte_days"] >= 14) & (df["dte_days"] <= 60)]
            elif "_dte" in df.columns:
                df = df[(df["_dte"] >= 14) & (df["_dte"] <= 60)]

        # Filter to reasonable strikes (delta 0.15-0.70 range or price-based)
        if "delta" in df.columns:
            df = df[df["delta"].abs().between(0.15, 0.70)]
        elif underlying_price:
            if option_type == "call":
                df = df[(df["strike"] >= underlying_price * 0.93) & (df["strike"] <= underlying_price * 1.10)]
            else:
                df = df[(df["strike"] >= underlying_price * 0.90) & (df["strike"] <= underlying_price * 1.07)]

        # Filter to affordable options
        if "mid" in df.columns:
            df = df[df["mid"] * 100 <= max_cost * 1.5]  # allow 50% over for comparison

        # Analyze each strike
        for _, row in df.iterrows():
            row_dict = {
                "strike": float(row.get("strike", 0)),
                "mid": float(row.get("mid", 0)),
                "delta": float(row.get("delta", 0)),
                "gamma": float(row.get("gamma", 0)),
                "theta_daily": float(row.get("theta", 0)),
                "vega": float(row.get("vega", 0)),
                "iv": float(row.get("iv", 0)),
            }
            analysis = analyze_strike(
                row_dict,
                underlying_price=underlying_price or 50,
                hold_days=hold_days,
                iv_rank=iv_rank,
                iv_year_low=iv_year_low,
                iv_current=iv_current,
            )
            analysis["dte"] = int(row.get("dte_days", row.get("_dte", 30)))
            analysis["expiry"] = str(row.get("expiry", "unknown"))
            analysis["open_interest"] = int(row.get("open_interest", 0))
            analysis["volume"] = int(row.get("volume", 0))
            analysis["bid"] = float(row.get("bid", 0))
            analysis["ask"] = float(row.get("ask", 0))
            analysis["data_source"] = "chain_snapshot"
            results.append(analysis)

        chain_date = str(chain["snapshot_timestamp"].iloc[0])[:10] if "snapshot_timestamp" in chain.columns else "unknown"
    else:
        # Fallback: generate synthetic analysis using BS model
        underlying_price = _get_approx_price(ticker)
        chain_date = "synthetic"

        # Use IV from rank tracker or default
        sigma = (iv_current / 100.0) if iv_current else 0.22

        # Generate candidate strikes
        if option_type == "call":
            strikes = [underlying_price * (1 + pct) for pct in [-0.02, 0, 0.02, 0.04, 0.06, 0.08]]
        else:
            strikes = [underlying_price * (1 - pct) for pct in [-0.02, 0, 0.02, 0.04, 0.06, 0.08]]

        for K in strikes:
            # Round to standard strike increments
            if underlying_price < 100:
                K = round(K)
            else:
                K = round(K / 5) * 5

            T = 35 / 365.0  # ~35 DTE
            greeks = bs_greeks(underlying_price, K, T, RISK_FREE_RATE, sigma, option_type)

            row_dict = {
                "strike": K,
                "mid": greeks["price"],
                "price": greeks["price"],
                "delta": greeks["delta"],
                "gamma": greeks["gamma"],
                "theta_daily": greeks["theta_daily"],
                "vega": greeks["vega"],
                "iv": sigma,
            }
            analysis = analyze_strike(
                row_dict,
                underlying_price=underlying_price,
                hold_days=hold_days,
                iv_rank=iv_rank,
                iv_year_low=iv_year_low,
                iv_current=iv_current,
            )
            analysis["dte"] = 35
            analysis["expiry"] = (date.today() + timedelta(days=35)).strftime("%Y-%m-%d")
            analysis["open_interest"] = -1  # unknown
            analysis["volume"] = -1
            analysis["data_source"] = "bs_synthetic"
            results.append(analysis)

    if not results:
        return {
            "ticker": ticker,
            "direction": direction,
            "option_type": option_type,
            "error": "No suitable options found",
            "greeks_score": 0,
        }

    # Sort by greeks_score, prefer affordable
    results.sort(key=lambda x: (x.get("affordable", False), x["greeks_score"]), reverse=True)

    best = results[0]
    top3 = results[:3]

    # Summarize vega opportunity
    vega_upside_note = "N/A"
    if iv_rank is not None and iv_current is not None and best.get("vega", 0) > 0:
        if iv_rank < 20:
            # Estimate: if IV goes from current to ~30th percentile
            iv_target = iv_current * 1.30
            iv_move_pts = iv_target - iv_current
            vega_profit = best["vega"] * iv_move_pts * 100
            vega_upside_note = (
                f"IV rank {iv_rank:.0f}% is very cheap. If IV normalizes to ~{iv_target:.0f}% "
                f"(+{iv_move_pts:.1f} pts), vega adds ~${vega_profit:.0f}/contract "
                f"({vega_profit / best['contract_cost'] * 100:.0f}% of premium)"
            )
        elif iv_rank < 35:
            iv_target = iv_current * 1.15
            iv_move_pts = iv_target - iv_current
            vega_profit = best["vega"] * iv_move_pts * 100
            vega_upside_note = (
                f"IV rank {iv_rank:.0f}% is cheap. Vega upside: ~${vega_profit:.0f}/contract "
                f"if IV normalizes"
            )
        elif iv_rank > 70:
            iv_target = iv_current * 0.85
            iv_move_pts = iv_current - iv_target
            vega_loss = best["vega"] * iv_move_pts * 100
            vega_upside_note = (
                f"WARNING: IV rank {iv_rank:.0f}% is expensive. Vega HEADWIND: "
                f"~${vega_loss:.0f}/contract if IV normalizes down"
            )

    # Build output
    output = {
        "ticker": ticker,
        "direction": direction,
        "option_type": option_type,
        "underlying_price": round(underlying_price, 2) if underlying_price else None,
        "chain_data_date": chain_date,
        "iv_environment": {
            "iv_rank": iv_rank,
            "iv_current": iv_current,
            "iv_year_high": iv_year_high,
            "iv_year_low": iv_year_low,
            "iv_trend": iv_trend,
            "assessment": (
                "VERY_CHEAP" if iv_rank is not None and iv_rank < 15
                else "CHEAP" if iv_rank is not None and iv_rank < 30
                else "NORMAL" if iv_rank is not None and iv_rank < 65
                else "EXPENSIVE" if iv_rank is not None and iv_rank < 85
                else "VERY_EXPENSIVE" if iv_rank is not None
                else "UNKNOWN"
            ),
        },
        "recommended": {
            "strike": best["strike"],
            "expiry": best.get("expiry", "unknown"),
            "dte": best.get("dte", 35),
            "premium": best["premium"],
            "contract_cost": best["contract_cost"],
            "affordable": best["affordable"],
            "greeks_score": best["greeks_score"],
        },
        "greeks_breakdown": {
            "delta": best["delta"],
            "gamma": best["gamma"],
            "theta_daily": best["theta_daily"],
            "vega": best["vega"],
            "iv": best.get("iv"),
        },
        "cost_analysis": {
            "theta_cost_per_day": round(abs(best["theta_daily"]) * 100, 2),
            "theta_cost_hold_period": round(abs(best["theta_daily"]) * 100 * hold_days, 2),
            "theta_pct_per_day": best["theta_pct_per_day"],
            "theta_cost_hold_pct": best["theta_cost_hold_pct"],
            "hold_days_assumed": hold_days,
            "theta_warning": best["theta_pct_per_day"] > 3.0,
        },
        "vega_analysis": {
            "vega_score": best["vega_score"],
            "vega_upside_note": vega_upside_note,
        },
        "gamma_analysis": {
            "gamma_score": best["gamma_score"],
            "dollar_gamma": best["dollar_gamma"],
            "gamma_per_dollar_invested": best["gamma_per_dollar"],
            "expected_gamma_capture_pct": best["expected_gamma_capture_pct"],
        },
        "component_scores": {
            "theta_efficiency": best["theta_score"],
            "delta_efficiency": best["delta_score"],
            "vega_upside": best["vega_score"],
            "gamma_value": best["gamma_score"],
        },
        "alternatives": [
            {
                "strike": a["strike"],
                "premium": a["premium"],
                "contract_cost": a["contract_cost"],
                "delta": a["delta"],
                "greeks_score": a["greeks_score"],
                "affordable": a["affordable"],
            }
            for a in top3[1:]
        ],
        "data_source": best.get("data_source", "unknown"),
    }

    return output


def _get_approx_price(ticker: str) -> float:
    """Get approximate current price. Tries yfinance, then hardcoded fallbacks."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass

    # Hardcoded fallback prices for common sector ETFs (approx Aug 2026)
    fallbacks = {
        "XLF": 57.0, "XLE": 82.0, "XLK": 250.0, "XLV": 155.0,
        "XLY": 210.0, "XLI": 130.0, "XLP": 82.0, "XLU": 78.0,
        "XLRE": 43.0, "XLB": 90.0, "XLC": 95.0, "SPY": 585.0,
        "QQQ": 520.0, "IWM": 225.0, "SMH": 280.0, "IBB": 140.0,
    }
    return fallbacks.get(ticker, 100.0)


# ---------------------------------------------------------------------------
# Batch analysis: enrich all current actionable signals
# ---------------------------------------------------------------------------
def analyze_all_signals() -> list[dict]:
    """
    Load current actionable signals from agentic_signals.json and enrich
    each with Greeks analysis.
    """
    if not SIGNALS_FILE.exists():
        print("[WARN] No agentic_signals.json found. Run aggregator first.", file=sys.stderr)
        return []

    with open(SIGNALS_FILE) as f:
        signals = json.load(f)

    actionable = signals.get("actionable_signals", signals.get("signals", []))
    results = []

    for sig in actionable:
        ticker = sig.get("ticker", "")
        direction = sig.get("recommended_option", sig.get("direction", "call"))
        if direction in ("bull",):
            direction = "call"
        elif direction in ("bear",):
            direction = "put"

        print(f"\n  Analyzing {ticker} ({direction})...")
        analysis = optimize_greeks(ticker, direction)
        analysis["signal_confidence"] = sig.get("confidence_score", 0)
        analysis["signal_sources"] = sig.get("confirming_sources", [])

        # Combined score: Greeks score weighted with signal confidence
        if analysis.get("greeks_score", 0) > 0:
            combined = analysis["greeks_score"] * 0.4 + sig.get("confidence_score", 0.5) * 100 * 0.6
            analysis["combined_score"] = round(combined, 1)

        results.append(analysis)

    # Sort by combined score
    results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

    return results


# ---------------------------------------------------------------------------
# Integration function: called from aggregator to enrich a single signal
# ---------------------------------------------------------------------------
def enrich_signal_with_greeks(signal: dict) -> dict:
    """
    Enrich a signal dict with Greeks analysis.
    Called from agentic_signal_aggregator.py.

    Adds: greeks_analysis sub-dict with score, recommended strike adjustment,
    theta warning, vega opportunity assessment.
    """
    ticker = signal.get("ticker", "")
    direction = signal.get("recommended_option", signal.get("direction", "call"))
    if direction in ("bull",):
        direction = "call"
    elif direction in ("bear",):
        direction = "put"

    if not ticker:
        return signal

    try:
        analysis = optimize_greeks(ticker, direction)
    except Exception as e:
        signal["greeks_analysis"] = {"error": str(e), "greeks_score": 0}
        return signal

    greeks_score = analysis.get("recommended", {}).get("greeks_score", 0)

    signal["greeks_analysis"] = {
        "greeks_score": greeks_score,
        "recommended_strike": analysis.get("recommended", {}).get("strike"),
        "recommended_premium": analysis.get("recommended", {}).get("premium"),
        "recommended_contract_cost": analysis.get("recommended", {}).get("contract_cost"),
        "theta_pct_per_day": analysis.get("cost_analysis", {}).get("theta_pct_per_day", 0),
        "theta_cost_hold_period": analysis.get("cost_analysis", {}).get("theta_cost_hold_period", 0),
        "theta_warning": analysis.get("cost_analysis", {}).get("theta_warning", False),
        "vega_score": analysis.get("vega_analysis", {}).get("vega_score", 50),
        "vega_note": analysis.get("vega_analysis", {}).get("vega_upside_note", ""),
        "gamma_score": analysis.get("gamma_analysis", {}).get("gamma_score", 50),
        "iv_assessment": analysis.get("iv_environment", {}).get("assessment", "UNKNOWN"),
        "data_source": analysis.get("data_source", "unknown"),
    }

    # Adjust confidence based on Greeks quality
    if greeks_score >= 70:
        # Excellent Greeks — small confidence boost
        signal["confidence_score"] = round(min(0.95, signal.get("confidence_score", 0.5) + 0.03), 2)
        signal["greeks_analysis"]["adjustment"] = "+0.03 (excellent Greeks)"
    elif greeks_score >= 50:
        # Acceptable — no change
        signal["greeks_analysis"]["adjustment"] = "none (acceptable Greeks)"
    elif greeks_score >= 30:
        # Poor Greeks — mild penalty
        signal["confidence_score"] = round(max(0.1, signal.get("confidence_score", 0.5) - 0.03), 2)
        signal["greeks_analysis"]["adjustment"] = "-0.03 (poor Greeks)"
    else:
        # Very poor — stronger penalty
        signal["confidence_score"] = round(max(0.1, signal.get("confidence_score", 0.5) - 0.07), 2)
        signal["greeks_analysis"]["adjustment"] = "-0.07 (very poor Greeks)"

    # Override strike recommendation if Greeks optimizer found a better one
    if analysis.get("recommended", {}).get("strike"):
        rec = analysis["recommended"]
        if rec.get("affordable", True):
            signal["recommended_strike"] = rec["strike"]
            signal["recommended_expiry"] = rec.get("expiry", signal.get("recommended_expiry"))
            signal["estimated_cost"] = rec["contract_cost"]

    return signal


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------
def print_analysis(analysis: dict):
    """Pretty-print a single ticker analysis."""
    ticker = analysis.get("ticker", "?")
    opt_type = analysis.get("option_type", "?")
    score = analysis.get("recommended", {}).get("greeks_score", 0)

    print(f"\n{'='*60}")
    print(f"  {ticker} {opt_type.upper()} — Greeks Score: {score:.0f}/100")
    print(f"{'='*60}")

    price = analysis.get("underlying_price")
    if price:
        print(f"  Underlying: ${price:.2f}")

    iv_env = analysis.get("iv_environment", {})
    if iv_env.get("iv_rank") is not None:
        print(f"  IV Rank: {iv_env['iv_rank']:.0f}% ({iv_env.get('assessment', '?')})")
        print(f"  IV Current: {iv_env.get('iv_current', '?')}% | "
              f"Range: {iv_env.get('iv_year_low', '?')}-{iv_env.get('iv_year_high', '?')}%")
        if iv_env.get("iv_trend"):
            print(f"  IV Trend: {iv_env['iv_trend']}")

    rec = analysis.get("recommended", {})
    print(f"\n  RECOMMENDED:")
    print(f"    Strike: ${rec.get('strike', '?')}")
    print(f"    Expiry: {rec.get('expiry', '?')} ({rec.get('dte', '?')} DTE)")
    print(f"    Premium: ${rec.get('premium', '?'):.2f} (${rec.get('contract_cost', '?'):.0f}/contract)")
    print(f"    Affordable: {'YES' if rec.get('affordable') else 'NO — over budget'}")

    greeks = analysis.get("greeks_breakdown", {})
    print(f"\n  GREEKS:")
    print(f"    Delta: {greeks.get('delta', '?'):.4f}")
    print(f"    Gamma: {greeks.get('gamma', '?'):.6f}")
    print(f"    Theta: ${abs(greeks.get('theta_daily', 0)) * 100:.2f}/day "
          f"({analysis.get('cost_analysis', {}).get('theta_pct_per_day', '?'):.1f}% of premium)")
    print(f"    Vega:  {greeks.get('vega', '?'):.4f}")

    cost = analysis.get("cost_analysis", {})
    if cost:
        print(f"\n  THETA COST ({cost.get('hold_days_assumed', 3)}-day hold):")
        print(f"    Daily bleed: ${cost.get('theta_cost_per_day', 0):.2f} "
              f"({cost.get('theta_pct_per_day', 0):.1f}% of premium)")
        print(f"    Total hold cost: ${cost.get('theta_cost_hold_period', 0):.2f} "
              f"({cost.get('theta_cost_hold_pct', 0):.1f}% of premium)")
        if cost.get("theta_warning"):
            print(f"    *** WARNING: Theta > 3%/day — expensive to hold ***")

    vega = analysis.get("vega_analysis", {})
    if vega.get("vega_upside_note") and vega["vega_upside_note"] != "N/A":
        print(f"\n  VEGA OPPORTUNITY:")
        print(f"    {vega['vega_upside_note']}")

    scores = analysis.get("component_scores", {})
    print(f"\n  COMPONENT SCORES:")
    print(f"    Theta Efficiency: {scores.get('theta_efficiency', 0):.0f}/100 (weight 30%)")
    print(f"    Delta Efficiency: {scores.get('delta_efficiency', 0):.0f}/100 (weight 25%)")
    print(f"    Vega Upside:      {scores.get('vega_upside', 0):.0f}/100 (weight 25%)")
    print(f"    Gamma Value:      {scores.get('gamma_value', 0):.0f}/100 (weight 20%)")
    print(f"    TOTAL:            {score:.0f}/100")

    alts = analysis.get("alternatives", [])
    if alts:
        print(f"\n  ALTERNATIVES:")
        for alt in alts:
            print(f"    ${alt['strike']}: premium ${alt['premium']:.2f} "
                  f"(${alt['contract_cost']:.0f}), delta {alt['delta']:.3f}, "
                  f"score {alt['greeks_score']:.0f}")

    print(f"\n  Data: {analysis.get('data_source', '?')} "
          f"(chain date: {analysis.get('chain_data_date', '?')})")


def main():
    parser = argparse.ArgumentParser(description="Options Greeks Optimizer")
    parser.add_argument("--ticker", "-t", help="Ticker to analyze")
    parser.add_argument("--direction", "-d", default="call",
                        choices=["call", "put", "bull", "bear"],
                        help="Option direction")
    parser.add_argument("--dte", type=int, default=35, help="Target DTE")
    parser.add_argument("--hold-days", type=int, default=HOLD_DAYS_DEFAULT,
                        help="Expected hold period in days")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all current actionable signals")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of pretty-print")
    args = parser.parse_args()

    if args.all:
        print("=" * 60)
        print("  GREEKS OPTIMIZER — ALL ACTIONABLE SIGNALS")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        results = analyze_all_signals()

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            for r in results:
                print_analysis(r)

            if results:
                print(f"\n{'='*60}")
                print(f"  RANKING (by combined score):")
                print(f"{'='*60}")
                for i, r in enumerate(results, 1):
                    ticker = r.get("ticker", "?")
                    gs = r.get("recommended", {}).get("greeks_score", 0)
                    cs = r.get("combined_score", 0)
                    cost = r.get("recommended", {}).get("contract_cost", 0)
                    print(f"  {i}. {ticker}: Greeks {gs:.0f} | Combined {cs:.0f} | "
                          f"Cost ${cost:.0f}")

        # Save to file
        output = {
            "timestamp": datetime.now().isoformat(),
            "hold_days": args.hold_days,
            "analyses": results,
        }
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {OUTPUT_FILE.name}")

    elif args.ticker:
        analysis = optimize_greeks(
            args.ticker, args.direction,
            target_dte_min=max(14, args.dte - 10),
            target_dte_max=args.dte + 10,
            hold_days=args.hold_days,
        )

        if args.json:
            print(json.dumps(analysis, indent=2, default=str))
        else:
            print("=" * 60)
            print(f"  GREEKS OPTIMIZER — {args.ticker}")
            print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            print_analysis(analysis)
    else:
        # Default: analyze XLF and XLE (planned entries for tomorrow)
        print("=" * 60)
        print("  GREEKS OPTIMIZER — DEFAULT: XLF + XLE (planned entries)")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        for ticker, direction in [("XLF", "call"), ("XLE", "call")]:
            analysis = optimize_greeks(ticker, direction)
            print_analysis(analysis)


if __name__ == "__main__":
    main()
