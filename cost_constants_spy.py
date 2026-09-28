"""
CANONICAL COST CONSTANTS FOR SPY (US EQUITIES — RETAIL)
========================================================
Companion to constants.py (which holds ES futures costs).
Single source of truth for SPY paper trading and backtests.
Import from here — NEVER hardcode costs.

SPY (ETF tracking S&P 500) via Alpaca / IBKR Lite / equivalent retail:
  - Commission: $0.00 per share (zero-commission retail brokers)
  - Tick size: 0.01 (NMS minimum for stocks >= $1.00)
  - Tick value: $0.01 per share
  - Sub-penny pricing IS allowed for some venues (NMS Rule 612 carve-outs),
    but our paper trader treats 0.01 as the canonical grid.

HC #420: Codebase authorization confirmed.
HC #433: This file is a constants module — not a Discord message.
Today (2026-06-04): FINRA killed the $25k PDT rule; retail equities now
fully accessible for subsecond strategies.

Spread is VARIABLE and must be measured from live data.
Typical SPY spread during regular trading hours: 0.01 (1 cent = 1 tick).
"""

# === INSTRUMENT: SPY (SPDR S&P 500 ETF) ===
INSTRUMENT = "SPY"
TICK_SIZE = 0.01           # USD per tick (NMS minimum for price >= $1.00)
TICK_VALUE = 0.01          # USD per share per tick
POINT_VALUE = 1.0          # USD per $1.00 move per share

# Position sizing on SPY is in SHARES, not contracts. A "lot" is 100 shares
# only by convention; round lots are not required for paper trading.
SHARES_PER_LOT_CONVENTION = 100

# === COMMISSION (retail zero-commission brokers) ===
# Alpaca, Robinhood, Fidelity, Schwab, IBKR Lite all charge $0 on SPY.
# IBKR Pro / institutional charge tiered fees; capture here if we ever upgrade.
COMMISSION_PER_SHARE = 0.00      # USD
COMMISSION_RT_PER_SHARE = 0.00   # USD round-trip
COMMISSION_TICKS = 0.0           # in tick units, since commission=0

# === REGULATORY FEES (these are NOT zero — apply on SELL only) ===
# SEC Section 31 fee + FINRA TAF (Trading Activity Fee). Rates as of 2026.
# Both are billed per share/notional on the SELL side.
SEC_FEE_PER_DOLLAR = 8.0e-6       # ~$8 per $1,000,000 of sell-side notional (2024-2025 rate; revisits annually)
FINRA_TAF_PER_SHARE = 0.000166    # USD per share, sell-side, capped per trade
FINRA_TAF_CAP_USD = 8.30          # max per trade

def sell_side_regulatory_fees(shares: int, price: float) -> float:
    """Total regulatory fees on a sell, in USD. Apply ONCE per sell-leg."""
    notional = abs(shares) * price
    sec = notional * SEC_FEE_PER_DOLLAR
    taf = min(FINRA_TAF_CAP_USD, abs(shares) * FINRA_TAF_PER_SHARE)
    return sec + taf

# === SPREAD (VARIABLE — measure from data) ===
TYPICAL_SPREAD_TICKS = 1.0   # SPY in RTH: 1 cent ~99% of the time
# Market-order cost ≈ 0.5 * spread (cross half) + reg fees on sell.
# Limit at NBBO: 0 spread cross, but adverse-selection risk.

# === PDT RULE — REPEALED 2026-06-04 ===
# FINRA killed the $25,000 minimum equity for pattern day traders today.
# We can now trade SPY intraday at any account size. Documenting for the
# record so anyone reading this file knows why the SPY pivot happened.
PDT_RULE_REPEALED_DATE = "2026-06-04"
PDT_MIN_EQUITY_USD = 0.0  # was $25,000 pre-repeal

# === DO NOT USE THESE VALUES ===
# 0.005 tick — WRONG, that's a sub-penny carve-out, not the SPY grid
# 1.0 cent fixed spread — WRONG, spread is variable; measure it
# $0.005/share IBKR Pro tier — only if/when we upgrade; document via env

# === SLIPPAGE MODEL (placeholder — calibrate from paper fills) ===
# Until we have actual paper-trade fill data, assume:
#   - passive limit at NBBO: 0 slippage, fill probability per execution model
#   - marketable limit / IOC at NBBO: cross half-spread = 0.5 cents
#   - market order: cross full spread + 0-1 ticks slippage on size > BBO
DEFAULT_PASSIVE_SLIPPAGE_TICKS = 0.0
DEFAULT_AGGRESSIVE_SLIPPAGE_TICKS = 0.5  # half-spread cross
DEFAULT_MARKET_SLIPPAGE_TICKS = 1.0      # full-spread cross


def round_trip_cost_ticks(order_style: str = "aggressive") -> float:
    """Total RT cost in TICKS (=cents for SPY) for a given execution style.

    Returns:
      - "passive":     0.0 (no spread cross, no commission)
      - "aggressive":  1.0 (half-spread cross each leg = full spread RT)
      - "market":      2.0 (full-spread cross each leg = 2x spread RT)
    """
    if order_style == "passive":
        return 2 * DEFAULT_PASSIVE_SLIPPAGE_TICKS + 2 * COMMISSION_TICKS
    if order_style == "aggressive":
        return 2 * DEFAULT_AGGRESSIVE_SLIPPAGE_TICKS + 2 * COMMISSION_TICKS
    if order_style == "market":
        return 2 * DEFAULT_MARKET_SLIPPAGE_TICKS + 2 * COMMISSION_TICKS
    raise ValueError(f"Unknown order_style: {order_style}")


if __name__ == "__main__":
    print("=== SPY cost constants ===")
    print(f"  TICK_SIZE          = ${TICK_SIZE}")
    print(f"  TICK_VALUE         = ${TICK_VALUE} per share")
    print(f"  COMMISSION_RT      = ${COMMISSION_RT_PER_SHARE} per share (retail)")
    print(f"  Typical spread     = {TYPICAL_SPREAD_TICKS} ticks (1 cent)")
    print(f"  RT cost (passive)  = {round_trip_cost_ticks('passive')} ticks")
    print(f"  RT cost (aggro)    = {round_trip_cost_ticks('aggressive')} ticks (1 cent)")
    print(f"  RT cost (market)   = {round_trip_cost_ticks('market')} ticks (2 cents)")
    print(f"  Reg fees on $100k sell of 200 SPY @ $580:"
          f" ${sell_side_regulatory_fees(200, 580):.4f}")
    print(f"  PDT rule repealed: {PDT_RULE_REPEALED_DATE}")
