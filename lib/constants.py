"""
SINGLE SOURCE OF TRUTH for all trading constants.
Every script MUST import from here. No hardcoded values elsewhere.

If you're tempted to write a number in another file, STOP and add it here.
"""
import os

# ──────────────────────────────────────────────────────────────
# ACCOUNT
# ──────────────────────────────────────────────────────────────
RH_ACCOUNT = os.environ.get("ROBINHOOD_ACCOUNT_NUMBER", "")
MAX_POSITIONS = 4          # Max concurrent option positions
MAX_COST_PER_TRADE = 200   # Dollars, single trade budget cap
MAX_PORTFOLIO_PCT = 0.60   # Don't risk more than 60% of equity in options

# ──────────────────────────────────────────────────────────────
# EXIT PARAMETERS — AVO-VALIDATED (HC #808, 4-yr walk-forward)
# These are the ONLY valid exit params. Old values are WRONG.
# ──────────────────────────────────────────────────────────────

# Low-vol regime (VIX < 25)
LV_TP_PCT = 0.43           # Take profit at +43%
LV_SL_PCT = 0.17           # Stop loss at -17% (realizes ~-20% after spread)
LV_TRAIL_ACTIVATE = 0.08   # Trailing stop activates at +8% from entry
LV_TRAIL_GIVEBACK = 0.20   # Give back 20% of peak gain

# High-vol regime (VIX >= 25)
HV_TP_PCT = 0.25           # Take profit at +25%
HV_SL_PCT = 0.12           # Stop loss at -12%
HV_TRAIL_ACTIVATE = 0.10   # Trailing stop activates at +10%
HV_TRAIL_GIVEBACK = 0.35   # Give back 35% of peak gain
HV_DAY1_EARLY_EXIT = 0.05  # If day 1 and loss > -5%, close (failed bounce)

VIX_REGIME_THRESHOLD = 25  # VIX >= this = high-vol mode

# ──────────────────────────────────────────────────────────────
# SECTOR-SPECIFIC HOLD PERIODS (AVO-validated)
# ──────────────────────────────────────────────────────────────
SECTOR_MAX_HOLD_DAYS = {
    "XLU": 10, "XLB": 9, "XLRE": 11,
    "XLK": 4, "XLF": 4,
    "XLE": 5, "XLI": 5,
}
DEFAULT_MAX_HOLD_DAYS = 5
HV_HOLD_REDUCTION = 3      # Reduce all holds by 3 days in high-vol
MIN_HOLD_DAYS = 2           # Never hold fewer than 2 days

# ──────────────────────────────────────────────────────────────
# BRACKET ORDER PARAMS (HC #810)
# ──────────────────────────────────────────────────────────────
SL_LIMIT_DISCOUNT = 0.05   # Set limit 5% below stop_price to ensure fill
BRACKET_PLACE_TIMEOUT = 300 # 5 min to place brackets after entry

# ──────────────────────────────────────────────────────────────
# ENTRY GATES
# ──────────────────────────────────────────────────────────────
MIN_DELTA = 0.30            # HC #804: no lottery tickets
MAX_THETA_DAILY_PCT = 0.05  # HC #804: theta < 5% of premium/day
MIN_CONFLUENCE = 3          # HC #805: minimum confirming signals
MIN_CONFIDENCE = 0.75       # Minimum confidence score
MIN_DTE_ENTRY = 14          # Don't enter options with < 14 DTE
THETA_EXIT_DTE = 7          # Exit if flat/negative and within 7 DTE
THESIS_BREAK_PCT = 0.05     # Underlying moved >5% against = thesis broken
DEAD_MONEY_LOW = -0.05      # Dead money range lower bound
DEAD_MONEY_HIGH = 0.05      # Dead money range upper bound
NO_REENTRY_SAME_STRIKE = 10 # Days before re-entering same ticker+strike (HC #809)
NO_REENTRY_SAME_TICKER = 5  # Days before re-entering same ticker, different strike

# ──────────────────────────────────────────────────────────────
# MACRO REGIME
# ──────────────────────────────────────────────────────────────
MACRO_REGIMES = {
    "RISK_ON": {"sl_override": None, "entry_allowed": True, "note": "Full sizing"},
    "RISK_OFF": {"sl_override": 0.12, "entry_allowed": False, "note": "No new calls. SL tightened to -12%"},
    "TRANSITION": {"sl_override": 0.15, "entry_allowed": True, "note": "Require 0.85+ confidence. SL -15%"},
    "INFLATIONARY": {"sl_override": None, "entry_allowed": True, "note": "Favor energy/materials"},
    "DEFLATIONARY": {"sl_override": None, "entry_allowed": True, "note": "Favor defensives"},
}

# ──────────────────────────────────────────────────────────────
# ES FUTURES (for quant stack, not options)
# ──────────────────────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
import os
from pathlib import Path

BASE = Path("/home/jupiter/Lvl3Quant")
ACTIVE_OPTIONS_FILE = BASE / "data" / "active_options.json"
POSITION_STATE_FILE = BASE / "data" / "rh_position_state.json"
MACRO_SUMMARY_FILE = BASE / "data" / "macro" / "macro_summary.json"
TRADE_JOURNAL_FILE = BASE / "data" / "trade_journal.json"
EXECUTION_LOG_FILE = BASE / "state" / "execution_log.json"
INJECT_SCRIPT = str(BASE / "scripts" / "autonomy_inject.sh")
