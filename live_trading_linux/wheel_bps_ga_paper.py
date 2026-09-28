#!/usr/bin/env python3
"""
Bull Put Spread (BPS) Paper Engine — GA-OPTIMIZED Variant
==========================================================
Uses the genetically-evolved 20-ticker portfolio from the GA optimizer
(Neptune, 2026-07-10). Walk-forward OOS Sharpe 2.59 vs baseline 1.37.

Imports and runs the conservative engine's logic with a fixed GA universe.
"""
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Override state directory BEFORE importing the conservative module
import live_trading_linux.wheel_bps_conservative_paper as engine

# ── Override paths ──
STATE_DIR = ROOT / "live_trading_linux" / "wheel_bps_ga_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

engine.STATE_DIR = STATE_DIR
engine.STATE_FILE = STATE_DIR / "state.json"
engine.TRADES_FILE = STATE_DIR / "trades.jsonl"
engine.EQUITY_FILE = STATE_DIR / "equity_curve.csv"
engine.FILL_QUALITY_FILE = STATE_DIR / "fill_quality.jsonl"
engine.NAV_HISTORY_FILE = STATE_DIR / "nav_history.json"

# ── Override logging ──
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

for handler in engine.log.handlers[:]:
    engine.log.removeHandler(handler)
engine.log.addHandler(logging.StreamHandler())
engine.log.addHandler(logging.FileHandler(str(LOG_DIR / "wheel_bps_ga_paper.log")))
for handler in engine.log.handlers:
    handler.setFormatter(logging.Formatter('%(asctime)s [BPS-GA] %(levelname)s %(message)s'))

# ── Override universe to GA-optimized 20 tickers ──
GA_UNIVERSE = [
    "NFLX", "NVDA", "PLTR", "WMT", "GM", "PFE", "LLY", "MCD",
    "CL", "T", "SMCI", "OXY", "HOOD", "JNJ", "F", "TGT",
    "VZ", "TSLA", "PANW", "TMUS",
]

# Override MAX_CONCURRENT to fit 20-ticker portfolio
engine.MAX_CONCURRENT = 20

# Monkey-patch load_universe to return GA tickers
_original_load_universe = engine.load_universe
engine.load_universe = lambda: GA_UNIVERSE

if __name__ == '__main__':
    engine.log.info("=" * 60)
    engine.log.info("BPS GA-OPTIMIZED PAPER ENGINE (A/B test)")
    engine.log.info(f"Universe: {len(GA_UNIVERSE)} tickers (GA-evolved)")
    engine.log.info(f"Tickers: {', '.join(sorted(GA_UNIVERSE))}")
    engine.log.info("=" * 60)
    engine.main()
