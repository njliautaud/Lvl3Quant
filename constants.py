"""
CANONICAL COST CONSTANTS FOR ES FUTURES TRADING
=================================================
Single source of truth. Import from here — NEVER hardcode costs.

ES (E-mini S&P 500) via AMP Futures:
  - Commission: $4.70 round-trip ($2.35 per side)
  - Tick size: 0.25 points
  - Tick value: $12.50 per tick
  - Commission in ticks: $4.70 / $12.50 = 0.376 ticks RT

Spread is VARIABLE and must be measured from live data.
Do NOT assume a fixed spread cost. Measure it.

HC #52: "ES TRADING COST IS $4.70 RT = 0.376 TICKS. STOP MAKING UP COSTS."
"""

# === INSTRUMENT: ES (E-mini S&P 500) ===
INSTRUMENT = "ES"
TICK_SIZE = 0.25          # points per tick
TICK_VALUE = 12.50        # USD per tick
POINT_VALUE = 50.0        # USD per point (= TICK_VALUE / TICK_SIZE)

# === COMMISSION (AMP Futures) ===
COMMISSION_RT = 4.70      # USD round-trip per contract
COMMISSION_PER_SIDE = 2.35  # USD per side per contract
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.376 ticks RT

# === COST ALIASES (for backward compatibility) ===
DEFAULT_COST_TICKS = COMMISSION_TICKS  # 0.376 ticks — commission only
RT_COST_USD = COMMISSION_RT
COST_PER_TICK = TICK_VALUE

# === SPREAD (VARIABLE — measure from data) ===
# These are typical values for reference only.
# Always measure actual spread from your data.
TYPICAL_SPREAD_TICKS = 1.0  # ES typically 1 tick during liquid hours
# Total cost for market orders ≈ spread + commission
# But spread depends on time-of-day, volatility, etc.

# === DO NOT USE THESE VALUES ===
# 2.0 ticks — WRONG, this was never correct
# 1.24 ticks — WRONG, used old $3.00 commission
# 1.376 ticks — WRONG, assumed 1 tick spread as fixed cost
# 0.248 ticks — WRONG, used old $3.10 commission
# 0.24 ticks — WRONG, used old $3.00 commission
