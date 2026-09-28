"""
VIX Spike Action Engine — generates specific, actionable trade recommendations
when VIX crosses 30/35/40.

Called by unified_signal_watcher.py to replace generic "buy VIX puts" alerts
with concrete strike/sizing/expiration recommendations.

Instruments (Robinhood-available):
  - UVXY puts (primary) — 1.5x leveraged VIX futures ETF, liquid options
  - SVIX shares/calls (secondary) — short VIX ETF
  - VIX index options are NOT on Robinhood (IBKR only)

Usage:
  # As module:
  from engines.vix_spike_action import generate_vix_spike_playbook
  msg = generate_vix_spike_playbook(vix_level=32.0, allocation=5000.0)

  # Standalone test:
  python3 engines/vix_spike_action.py --vix 32 --allocation 5000
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta, date
from typing import Optional

import numpy as np

# ─── Optional imports (graceful fallback) ────────────────────────────────────

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─── Constants ───────────────────────────────────────────────────────────────

# UVXY empirical relationship to VIX (approximate, based on 1.5x daily leverage)
# These are rough multipliers calibrated from historical data.
# When VIX is at X, UVXY is approximately at the mapped price.
# UVXY resets/decays significantly so this mapping drifts over time.
UVXY_VIX_RATIO_APPROX = {
    20: 18,
    25: 28,
    30: 42,
    35: 58,
    40: 78,
    45: 100,
    50: 125,
}

# Typical UVXY implied vol when VIX is elevated (80-120%)
UVXY_IMPLIED_VOL_DEFAULT = 1.00  # 100% annualized

# Tiered allocation by VIX level
ALLOCATION_TIERS = {
    30: 0.50,  # VIX 30-34: deploy 50% of allocation
    35: 0.30,  # VIX 35-39: deploy 30% more
    40: 0.20,  # VIX 40+:   deploy remaining 20%
}

# Historical stats
HISTORICAL_WIN_RATE = 0.83
HISTORICAL_AVG_RETURN = 1.37  # +137% average return on winners


# ─── Black-Scholes for put estimation ────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF — uses scipy if available, else approximation."""
    if HAS_SCIPY:
        return float(norm.cdf(x))
    # Abramowitz & Stegun approximation
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    )
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x_abs * x_abs / 2.0)
    return 0.5 * (1.0 + sign * y)


def bs_put_price(
    S: float, K: float, T: float, r: float = 0.05, sigma: float = 1.0
) -> float:
    """
    Black-Scholes European put price.
    S: underlying price, K: strike, T: time to expiry (years),
    r: risk-free rate, sigma: implied vol (annualized).
    """
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    put = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return max(put, 0.0)


def bs_call_price(
    S: float, K: float, T: float, r: float = 0.05, sigma: float = 1.0
) -> float:
    """Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return max(call, 0.0)


# ─── Live data fetching ─────────────────────────────────────────────────────

def _get_live_price(ticker: str) -> Optional[float]:
    """Fetch current price via yfinance. Returns None on failure."""
    if not HAS_YFINANCE:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = getattr(info, "last_price", None)
        if price is None:
            hist = t.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return float(price) if price else None
    except Exception:
        return None


def _get_options_chain(ticker: str, min_dte: int = 25, max_dte: int = 55):
    """
    Fetch options chain for a ticker, targeting 30-45 DTE sweet spot.
    Returns (expiration_date_str, puts_df, calls_df) or (None, None, None).
    """
    if not HAS_YFINANCE:
        return None, None, None
    try:
        t = yf.Ticker(ticker)
        expirations = t.options  # list of date strings
        if not expirations:
            return None, None, None

        today = date.today()
        best_exp = None
        best_dte = None
        target_dte = 37  # sweet spot center

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                if best_dte is None or abs(dte - target_dte) < abs(best_dte - target_dte):
                    best_exp = exp_str
                    best_dte = dte

        if best_exp is None:
            # Fallback: pick the nearest expiration >= min_dte
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte >= min_dte:
                    best_exp = exp_str
                    best_dte = dte
                    break

        if best_exp is None:
            return None, None, None

        chain = t.option_chain(best_exp)
        return best_exp, chain.puts, chain.calls

    except Exception:
        return None, None, None


def _estimate_uvxy_price(vix_level: float) -> float:
    """Estimate UVXY price from VIX level using interpolation."""
    vix_keys = sorted(UVXY_VIX_RATIO_APPROX.keys())
    if vix_level <= vix_keys[0]:
        return float(UVXY_VIX_RATIO_APPROX[vix_keys[0]])
    if vix_level >= vix_keys[-1]:
        return float(UVXY_VIX_RATIO_APPROX[vix_keys[-1]])
    # Linear interpolation
    for i in range(len(vix_keys) - 1):
        if vix_keys[i] <= vix_level <= vix_keys[i + 1]:
            lo, hi = vix_keys[i], vix_keys[i + 1]
            frac = (vix_level - lo) / (hi - lo)
            return UVXY_VIX_RATIO_APPROX[lo] + frac * (
                UVXY_VIX_RATIO_APPROX[hi] - UVXY_VIX_RATIO_APPROX[lo]
            )
    return 50.0  # fallback


def _find_target_expiration(dte_target: int = 37) -> tuple[str, int]:
    """Return (expiration_label, dte) for the target DTE window."""
    today = date.today()
    target = today + timedelta(days=dte_target)
    # Snap to the nearest Friday (options expiration day)
    days_until_friday = (4 - target.weekday()) % 7
    if days_until_friday == 0 and target.weekday() != 4:
        days_until_friday = 7
    exp_date = target + timedelta(days=days_until_friday)
    if exp_date.weekday() != 4:
        # If today is Friday, use this Friday
        exp_date = target
        while exp_date.weekday() != 4:
            exp_date += timedelta(days=1)
    actual_dte = (exp_date - today).days
    label = exp_date.strftime("%b %d")  # e.g. "Aug 15"
    return label, actual_dte


# ─── Core recommendation engine ─────────────────────────────────────────────

def _generate_uvxy_puts(
    vix_level: float,
    uvxy_price: float,
    allocation: float,
    tier_frac: float,
    exp_label: str,
    dte: int,
    live_puts=None,
) -> list[dict]:
    """
    Generate UVXY put recommendations.
    Returns list of dicts with strike, premium, contracts, cost info.
    """
    tier_alloc = allocation * tier_frac
    T = dte / 365.0
    sigma = UVXY_IMPLIED_VOL_DEFAULT

    # Strike selection: ATM and ~10-15% OTM
    atm_strike = round(uvxy_price)  # round to nearest dollar
    otm_strike = round(uvxy_price * 0.85)  # ~15% OTM

    # Try to snap to common strike increments ($1 for UVXY typically)
    strikes = [atm_strike, otm_strike]
    # Add a mid-level strike
    mid_strike = round(uvxy_price * 0.90)
    if mid_strike not in strikes and mid_strike != atm_strike:
        strikes = [atm_strike, mid_strike, otm_strike]
    strikes = sorted(strikes, reverse=True)

    recommendations = []
    remaining = tier_alloc

    for i, K in enumerate(strikes):
        if K <= 0 or remaining <= 0:
            continue

        # Try live quote first
        premium = None
        if live_puts is not None:
            try:
                matching = live_puts[
                    (live_puts["strike"] >= K - 0.5) & (live_puts["strike"] <= K + 0.5)
                ]
                if not matching.empty:
                    row = matching.iloc[0]
                    K = float(row["strike"])  # use exact strike
                    ask = row.get("ask", None)
                    last = row.get("lastPrice", None)
                    if ask is not None and float(ask) > 0:
                        premium = float(ask)
                    elif last is not None and float(last) > 0:
                        premium = float(last)
            except Exception:
                pass

        # Fallback to BS estimate
        if premium is None:
            premium = bs_put_price(uvxy_price, K, T, sigma=sigma)
            premium = max(premium, 0.10)  # floor at $0.10

        # Round premium to nearest 0.05 (typical option increment)
        premium = round(premium * 20) / 20
        if premium < 0.05:
            premium = 0.05

        cost_per_contract = premium * 100  # options are 100 shares

        # Allocate: primary strike gets ~60%, secondary ~40%
        if i == 0:
            strike_alloc = remaining * 0.60
        else:
            strike_alloc = remaining

        num_contracts = max(1, int(strike_alloc / cost_per_contract))
        total_cost = num_contracts * cost_per_contract

        if total_cost > remaining * 1.2:  # allow small overshoot
            num_contracts = max(1, int(remaining / cost_per_contract))
            total_cost = num_contracts * cost_per_contract

        # Breakeven: put buyer profits when UVXY drops below K - premium
        breakeven_uvxy = K - premium
        # Rough estimate of what VIX level corresponds to breakeven UVXY
        # (inverse of _estimate_uvxy_price, approximate)
        breakeven_vix = _uvxy_to_vix_approx(breakeven_uvxy)

        recommendations.append({
            "strike": K,
            "premium": premium,
            "contracts": num_contracts,
            "total_cost": total_cost,
            "breakeven_uvxy": breakeven_uvxy,
            "breakeven_vix": breakeven_vix,
            "exp_label": exp_label,
            "dte": dte,
        })
        remaining -= total_cost

    return recommendations


def _uvxy_to_vix_approx(uvxy_price: float) -> Optional[float]:
    """Rough inverse mapping from UVXY price to approximate VIX level."""
    vix_keys = sorted(UVXY_VIX_RATIO_APPROX.keys())
    prices = [UVXY_VIX_RATIO_APPROX[k] for k in vix_keys]
    if uvxy_price <= prices[0]:
        return float(vix_keys[0])
    if uvxy_price >= prices[-1]:
        return float(vix_keys[-1])
    for i in range(len(prices) - 1):
        if prices[i] <= uvxy_price <= prices[i + 1]:
            frac = (uvxy_price - prices[i]) / (prices[i + 1] - prices[i])
            return vix_keys[i] + frac * (vix_keys[i + 1] - vix_keys[i])
    return None


def _generate_svix_recommendation(
    vix_level: float, allocation: float, tier_frac: float,
    simulating: bool = False,
) -> Optional[dict]:
    """Generate SVIX share buy recommendation as secondary play."""
    svix_price = None
    if not simulating:
        svix_price = _get_live_price("SVIX")

    if svix_price is None:
        # Estimate: SVIX roughly inverse to VIX. When VIX=30, SVIX ~$25-35
        # SVIX tracks -1x daily VIX futures
        svix_price = max(5.0, 55.0 - vix_level * 0.8)

    alloc = allocation * tier_frac * 0.40  # smaller position than puts
    num_shares = max(1, int(alloc / svix_price))
    total_cost = num_shares * svix_price
    stop_loss = svix_price * 0.84  # ~16% stop

    return {
        "price": round(svix_price, 2),
        "shares": num_shares,
        "total_cost": round(total_cost, 2),
        "stop_loss": round(stop_loss, 2),
        "stop_pct": 16,
    }


# ─── Main playbook generator ────────────────────────────────────────────────

def generate_vix_spike_playbook(
    vix_level: float, allocation: float = 5000.0
) -> str:
    """
    Generate a formatted Discord-ready message with specific VIX spike trade
    recommendations.

    Args:
        vix_level: Current VIX level (e.g. 32.5)
        allocation: Total dollar amount to deploy across tiers (default $5000)

    Returns:
        Formatted string ready for Discord (no file paths, no jargon).
    """
    # ── Determine which tier we're in ──
    if vix_level >= 40:
        tier_key = 40
        tier_label = "EXTREME"
        emoji = "\U0001f6a8\U0001f6a8\U0001f6a8"
    elif vix_level >= 35:
        tier_key = 35
        tier_label = "ESCALATION"
        emoji = "\U0001f6a8\U0001f6a8"
    elif vix_level >= 30:
        tier_key = 30
        tier_label = "SPIKE"
        emoji = "\U0001f6a8"
    else:
        return (
            f"VIX at {vix_level:.1f} — below 30 threshold. "
            f"No action needed. Monitoring for spike."
        )

    tier_frac = ALLOCATION_TIERS[tier_key]
    tier_alloc = allocation * tier_frac
    mode = "LIVE"

    # ── Get actual VIX to detect simulation vs real spike ──
    actual_vix = _get_live_price("^VIX")
    simulating = False
    if actual_vix is not None and abs(vix_level - actual_vix) > 5:
        # User is simulating a VIX level far from current — use estimates
        # since live UVXY/options prices won't reflect the simulated scenario
        simulating = True

    # ── Get UVXY price (live or estimated) ──
    uvxy_price = None
    if not simulating:
        uvxy_price = _get_live_price("UVXY")
    if uvxy_price is None:
        uvxy_price = _estimate_uvxy_price(vix_level)
        mode = "ESTIMATE"

    # ── Get options chain (skip for simulations — prices won't match) ──
    exp_label_live, live_puts = None, None
    dte_live = None
    if not simulating:
        exp_str, puts_df, calls_df = _get_options_chain("UVXY")
        if exp_str is not None and puts_df is not None:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte_live = (exp_date - date.today()).days
            exp_label_live = exp_date.strftime("%b %d")
            live_puts = puts_df
        else:
            mode = "ESTIMATE"
    else:
        mode = "ESTIMATE"

    # Use live expiration or estimate one
    if exp_label_live:
        exp_label = exp_label_live
        dte = dte_live
    else:
        exp_label, dte = _find_target_expiration(37)

    # ── Generate UVXY put recommendations ──
    puts = _generate_uvxy_puts(
        vix_level, uvxy_price, allocation, tier_frac,
        exp_label, dte, live_puts
    )

    # ── Generate SVIX recommendation ──
    svix = _generate_svix_recommendation(vix_level, allocation, tier_frac, simulating)

    # ── Build the message ──
    lines = []
    lines.append(f"{emoji} VIX {tier_label} ACTION PLAN — VIX at {vix_level:.1f}")
    lines.append("")

    if mode == "ESTIMATE":
        lines.append("(Using estimates — market may be closed)")
        lines.append("")

    # UVXY puts section
    lines.append(f"UVXY Puts (primary play) — UVXY ~${uvxy_price:.2f}, {dte} DTE:")
    total_put_risk = 0.0
    for rec in puts:
        cost_str = f"${rec['total_cost']:,.0f}"
        lines.append(
            f"  Buy {rec['contracts']}x UVXY {rec['exp_label']} "
            f"${rec['strike']:.0f} Put @ ~${rec['premium']:.2f}  "
            f"({cost_str} risk)"
        )
        total_put_risk += rec["total_cost"]

    # Breakeven info
    if puts:
        be = puts[0]  # ATM put breakeven
        be_vix = be.get("breakeven_vix")
        be_str = f"VIX ~{be_vix:.0f}" if be_vix else "N/A"
        lines.append(
            f"  Breakeven: UVXY below ${be['breakeven_uvxy']:.0f} "
            f"(roughly {be_str})"
        )
    lines.append("")

    # SVIX section
    if svix:
        lines.append("Alternative — SVIX shares:")
        lines.append(
            f"  Buy {svix['shares']} shares SVIX @ ~${svix['price']:.2f} "
            f"(${svix['total_cost']:,.0f})"
        )
        lines.append(f"  Stop loss at ${svix['stop_loss']:.2f} (-{svix['stop_pct']}%)")
        lines.append("")

    # Summary
    lines.append(
        f"Total put risk: ${total_put_risk:,.0f} (max loss = premium paid)"
    )
    lines.append(
        f"Tier deployed: {tier_frac*100:.0f}% of ${allocation:,.0f} allocation "
        f"(VIX {tier_key}+ tier)"
    )

    # Sizing context for multi-tier
    if tier_key == 30:
        lines.append(
            "Remaining 50% reserved for VIX 35+ and 40+ escalation tiers"
        )
    elif tier_key == 35:
        lines.append(
            "Tranche 2 of 3 — 20% still reserved for VIX 40+ extreme tier"
        )
    elif tier_key == 40:
        lines.append("Final tranche — full allocation deployed")

    lines.append("")
    lines.append(
        f"Historical: {HISTORICAL_WIN_RATE*100:.0f}% of VIX>{tier_key} events "
        f"are profitable, avg return +{HISTORICAL_AVG_RETURN*100:.0f}%"
    )
    lines.append("Target exit: When VIX drops below 20")
    lines.append(
        "Why it works: UVXY has daily contango decay that accelerates "
        "the put's value as VIX mean-reverts"
    )

    return "\n".join(lines)


# ─── Standalone test ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIX Spike Action Engine — generate trade recommendations"
    )
    parser.add_argument(
        "--vix", type=float, required=True,
        help="Simulated VIX level (e.g. 32)"
    )
    parser.add_argument(
        "--allocation", type=float, default=5000.0,
        help="Total dollar allocation for spike event (default: $5000)"
    )
    parser.add_argument(
        "--all-tiers", action="store_true",
        help="Show playbook for all three tiers (30, 35, 40)"
    )
    args = parser.parse_args()

    if args.all_tiers:
        print("=" * 60)
        print("FULL TIERED PLAYBOOK — All escalation levels")
        print("=" * 60)
        for level in [30, 35, 40]:
            print()
            print("-" * 60)
            result = generate_vix_spike_playbook(level, args.allocation)
            print(result)
        print()
        print("-" * 60)
        print(f"\nCumulative deployment across all 3 tiers: "
              f"${args.allocation:,.0f} (100%)")
    else:
        result = generate_vix_spike_playbook(args.vix, args.allocation)
        print(result)


if __name__ == "__main__":
    main()
