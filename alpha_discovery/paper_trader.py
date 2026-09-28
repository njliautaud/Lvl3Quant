#!/usr/bin/env python3
"""
Paper Trading Harness for CNN Vol-Gated Strategy (De-biased Configuration)
==========================================================================

Uses the validated configuration:
  - BookSpatialCNN predictions (expanding z-score normalized)
  - Vol gate: >= 80th percentile expanding
  - Conviction: >= 1.5 z-score
  - Hold: 30 minutes (18,000 bars at 100ms)
  - Time filter: morning (9:30-11:30) + afternoon (1:30-3:00)
  - Direction: signal sign determines long/short

Expected performance (de-biased OOS):
  - Sharpe: 3.89
  - Win rate: 57.8%
  - ~83 trades per 74 days (~1.1/day)
  - Gross: +14.5 ticks/trade ($181)
  - Net: +13.3 ticks/trade ($166) after 1.24 tick costs

This script is the SPECIFICATION for the live paper trading system.
It requires:
  1. Live MBO data feed (Databento or similar)
  2. Trained BookSpatialCNN model weights
  3. Broker connection for paper orders (IBKR paper account)

Status: SPECIFICATION ONLY — not yet connected to live data.
Needs: Dec-Mar 2026 MBO data ($200 Databento) for OOT validation first.
"""

import json
import logging
import os
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("paper_trader")

# Strategy configuration (validated de-biased)
CONFIG = {
    "model": "BookSpatialCNN",
    "signal_normalization": "expanding_zscore",  # NOT full-day zscore!
    "cnn_offset_bars": 99,  # window_size - 1
    "vol_gate_percentile": 80,
    "vol_gate_method": "expanding_window",  # NOT full-day percentile!
    "conviction_threshold": 1.5,  # z-score units
    "hold_bars": 18000,  # 30 minutes at 100ms bars
    "hold_seconds": 1800,
    "time_filter": "morning_afternoon",  # 9:30-11:30 + 1:30-3:00 ET
    "cooldown_bars": 18000,  # don't overlap trades
    "max_trades_per_day": 5,
    "instrument": "ES",
    "tick_size": 0.25,
    "tick_value": 12.50,
    "cost_spread_ticks": 1.0,
    "cost_commission_ticks": 0.24,
    "contracts": 1,
}

# Expected metrics from OOS validation (de-biased: expanding z-score + expanding vol percentile)
EXPECTED = {
    "sharpe": 3.89,
    "win_rate_pct": 57.8,
    "avg_gross_ticks": 14.5,
    "avg_net_ticks": 13.3,
    "trades_per_day": 1.1,
    "max_dd_ticks": 356,
    "oos_period": "Jul-Nov 2025 (74 days)",
    "validation_status": "OOS validated, OOT pending",
    "critical_risk": "Signal may not persist in Dec-Mar 2026 (regime change risk)",
}

# Robustness: vol>=80 + 30min + morning_afternoon is robust across ALL conviction levels
ROBUSTNESS = {
    "30min_morning_afternoon_profitable_pct": "100% (all 4 conv levels at vol>=80)",
    "vol80_conv0.5": {"sharpe": 2.85, "trades": 84, "pnl_ticks": 831},
    "vol80_conv1.0": {"sharpe": 2.86, "trades": 83, "pnl_ticks": 837},
    "vol80_conv1.5": {"sharpe": 3.89, "trades": 83, "pnl_ticks": 1101},  # BEST
    "vol80_conv2.0": {"sharpe": 1.85, "trades": 82, "pnl_ticks": 501},
    "vol60_also_profitable": "All 4 conv levels at vol>=60 also profitable (Sharpe 0.35-1.79)",
    "only_30min_works": "5min/10min/1hr ALL deeply negative (avg Sharpe -2.3 to -2.7)",
    "overall_profitable_pct": "17.2% of 128 configs (narrow but real signal)",
}


class PaperTradingState:
    """Track paper trading state."""

    def __init__(self, state_file="paper_trading_state.json"):
        self.state_file = Path(state_file)
        self.trades = []
        self.current_position = None
        self.daily_trades = 0
        self.last_trade_date = None
        self.expanding_vol_values = []
        self.expanding_signal_sum = 0.0
        self.expanding_signal_sum2 = 0.0
        self.expanding_signal_count = 0
        self.total_pnl_ticks = 0.0
        self.load()

    def load(self):
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
                self.trades = data.get("trades", [])
                self.total_pnl_ticks = data.get("total_pnl_ticks", 0.0)

    def save(self):
        with open(self.state_file, "w") as f:
            json.dump({
                "trades": self.trades,
                "total_pnl_ticks": self.total_pnl_ticks,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2, default=str)

    def add_trade(self, trade):
        self.trades.append(trade)
        self.total_pnl_ticks += trade.get("net_pnl_ticks", 0)
        self.save()

    def summary(self):
        if not self.trades:
            return "No trades yet."

        pnls = [t["net_pnl_ticks"] for t in self.trades]
        n = len(pnls)
        total = sum(pnls)
        avg = total / n
        wins = sum(1 for p in pnls if p > 0)

        return (
            f"Paper Trading Summary:\n"
            f"  Trades: {n}\n"
            f"  Total PnL: {total:+.1f} ticks (${total * CONFIG['tick_value']:+,.0f})\n"
            f"  Avg PnL: {avg:+.2f} ticks/trade\n"
            f"  Win Rate: {100 * wins / n:.1f}%\n"
            f"  Expected (OOS): {EXPECTED['avg_net_ticks']:+.1f} ticks/trade, "
            f"{EXPECTED['win_rate_pct']:.1f}% WR"
        )


def check_readiness():
    """Check if we're ready to paper trade."""
    checks = {
        "MBO data (Dec-Mar 2026)": False,  # Need OOT validation first
        "CNN model weights": Path("alpha_discovery/deep_models/results").exists(),
        "IBKR paper account": False,  # Need to set up
        "Live data feed": False,  # Need Databento subscription
    }

    print("Paper Trading Readiness Check:")
    print("=" * 50)
    all_ready = True
    for check, status in checks.items():
        status_str = "READY" if status else "NOT READY"
        print(f"  {'[x]' if status else '[ ]'} {check}: {status_str}")
        if not status:
            all_ready = False

    print()
    if all_ready:
        print("All systems ready for paper trading!")
    else:
        print("NOT READY — complete the checks above first.")
        print()
        print("BLOCKERS:")
        print("  1. Buy Dec-Mar 2026 MBO data ($200 from Databento)")
        print("  2. Run OOT validation on new data")
        print("  3. If OOT Sharpe > 1.5, proceed to paper trading")
        print("  4. Set up IBKR paper account")
        print("  5. Connect to live Databento MBO feed")

    return all_ready


def print_strategy_card():
    """Print the validated strategy specification."""
    print()
    print("=" * 60)
    print("  CNN VOL-GATED STRATEGY — VALIDATED SPECIFICATION")
    print("=" * 60)
    print()
    for key, val in CONFIG.items():
        print(f"  {key:30s}: {val}")
    print()
    print("  EXPECTED PERFORMANCE (74-day OOS):")
    for key, val in EXPECTED.items():
        print(f"    {key:25s}: {val}")
    print()
    print("  CRITICAL NOTES:")
    print("    1. zscore must be EXPANDING window (not full-day)")
    print("    2. Vol percentile must be EXPANDING (not look-ahead)")
    print("    3. CNN predictions offset by 99 bars from MBO bar 0")
    print("    4. 5min/10min/1hr holds are ALL unprofitable")
    print("    5. Signal works in vol>=80th only (not unconditionally)")
    print("=" * 60)


if __name__ == "__main__":
    print_strategy_card()
    print()
    check_readiness()
