"""
Standardized Quant Research Tools
=================================
Central modules for adversarial validation, options pricing, backtesting,
and research experiment management.

All future research scripts should IMPORT from these modules
instead of reimplementing validation, pricing, and Sharpe calculations.

Modules:
    adversarial_validator - 5-gate adversarial validation of trade streams
    options_pricer        - ATR-based Black-Scholes with bid-ask haircuts
    sector_backtest       - Standardized sector bull call spread backtesting
    research_launcher     - Hypothesis-driven research experiment runner
"""

from .adversarial_validator import validate_trades, ValidationResult
from .options_pricer import (
    price_bull_call_spread,
    exit_spread_value,
    bs_call_price,
)
from .sector_backtest import run_sector_backtest
from .research_launcher import run_research

__all__ = [
    "validate_trades",
    "ValidationResult",
    "price_bull_call_spread",
    "exit_spread_value",
    "bs_call_price",
    "run_sector_backtest",
    "run_research",
]
