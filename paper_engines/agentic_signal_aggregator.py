#!/usr/bin/env python3
"""
Agentic Signal Aggregator
=========================
Reads state from ALL validated paper engines and produces a unified signal
dashboard for the agentic trading account ($645, Level 2 options = single-leg only).

Outputs:
  - /home/jupiter/Lvl3Quant/state/agentic_signals.json
  - Plain-English summary to stdout

Usage:
  python3 agentic_signal_aggregator.py

ROBINHOOD MCP TOOLS INTEGRATION
================================

This aggregator can be ENHANCED with Robinhood MCP tools for real-time validation
and technical confirmation. Available tools (see knowledge_base/rh_tools_integration.md):

1. MARKET DATA:
   - get_equity_quotes(symbols=[...]) — current price, bid/ask (replaces yfinance)
   - get_equity_historicals(symbols=[...], start_time=..., interval=...) — OHLCV bars
   - get_earnings_calendar(start_date=..., days=...) — earnings events (replaces yfinance)

2. TECHNICAL INDICATORS:
   - get_equity_technical_indicators(symbol=..., type="rsi"/"macd"/"adx"/..., interval="day")
   - Use to cross-reference LGBM signals with RSI/MACD/Bollinger confluence
   - Example: LGBM bull + RSI>60 + above SMA(50) = HIGH confidence boost

3. SAVED SCANS (run real-time market filters):
   - run_scan(scan_id="a121153d-2123-4a3d-8b68-54b882070f74")  # High Options Volume
   - run_scan(scan_id="<upcoming_earnings_id>")  # Upcoming earnings (get ID via get_scans())
   - Use to flag signals also appearing in high-vol scanner for confluence

4. OPTIONS LIQUIDITY CHECKS:
   - get_option_chains(underlying_symbol="XLK") — available expirations
   - get_option_instruments(chain_symbol="XLK", expiration_dates="2026-08-15", strike_price="185", type="call")
   - get_option_quotes(instrument_ids=[...]) — bid/ask/IV/OI per contract
   - REJECT any recommended option with OI < 50 or bid/ask spread > $0.50

5. ORDER SIMULATION:
   - review_option_order(account_number=..., legs=[...], quantity=..., price=..., chain_symbol=..., underlying_type="equity")
   - Validates buying power, PDT rules, current quote before placing live order
   - ALWAYS call before place_option_order for real money

INTEGRATION ROADMAP (Additive — does not break existing flow):
  Phase 1: Replace yfinance VIX with get_equity_quotes; add real-time price validation
  Phase 2: Add RSI(14) + MACD technical confluence checks (boost confidence if aligned)
  Phase 3: Add options liquidity gating (reject if OI < 50)
  Phase 4: Add pre-trade review_option_order simulation before placing orders
  Phase 5: Replace earnings filter with get_earnings_calendar (more reliable)

See knowledge_base/rh_tools_integration.md for full integration examples and error handling.
"""

import calendar
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any

# Greeks optimization (HC — options entry timing)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
try:
    from greeks_optimizer import enrich_signal_with_greeks
    GREEKS_OPTIMIZER_AVAILABLE = True
except ImportError:
    GREEKS_OPTIMIZER_AVAILABLE = False

# Entry timing score (macro + technical + market structure)
try:
    from timing_score import compute_timing_score
    TIMING_SCORE_AVAILABLE = True
except ImportError:
    TIMING_SCORE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/home/jupiter/Lvl3Quant")
STATE = BASE / "state"
LOGS  = BASE / "paper_engines" / "logs"

SECTOR_SPREADS_STATE   = STATE / "sector_spreads_paper_state.json"
QUALITY_MOMENTUM_STATE = STATE / "quality_momentum_paper_state.json"
PEAD_DRIFT_STATE       = STATE / "pead_drift_paper_state.json"
VIX_CALL_SPREAD_STATE  = STATE / "vix_call_spread_paper_state.json"
SECTOR_ETF_MOM_STATE   = STATE / "sector_etf_momentum_paper_state.json"
SIGNAL_WATCHER_STATE   = STATE / "signal_watcher_state.json"

# New engines (Session 24)
PEAD_ML_STATE          = STATE / "pead_ml_paper_state.json"
IV_RUNUP_STATE         = STATE / "iv_runup_paper_state.json"
EARNINGS_GAP_STATE     = STATE / "earnings_gap_signals.json"
PEAD_ML_SCORES_STATE   = STATE / "pead_ml_scores.json"

# V9.1/V9.3/V10 — our BEST validated strategies (HC #753: must feed agentic signals)
SECTOR_V91_STATE       = STATE / "sector_combined_v91_paper_state.json"
SECTOR_V93_STATE       = STATE / "sector_combined_v93_paper_state.json"
SECTOR_V10_STATE       = STATE / "sector_combined_v10_optimal_paper_state.json"

MOMENTUM_OPTIONS_STATE = STATE / "momentum_options_paper_state.json"
EQUITY_ROTATION_STATE  = STATE / "sector_equity_rotation_paper_state.json"

# Validated strategies #8-#12 (HC #773 gap fix — previously missing from aggregator)
RSI_DIVERGENCE_STATE   = STATE / "rsi_divergence_paper_state.json"
BOND_YIELD_STATE       = STATE / "bond_yield_paper_state.json"
IV_RV_GAP_STATE        = STATE / "iv_rv_gap_paper_state.json"
LIQUIDITY_SIGNAL_STATE = STATE / "liquidity_signal_paper_state.json"
VOL_TERM_STRUCT_STATE  = STATE / "vol_term_structure_paper_state.json"
CROSS_TYPE_STATE       = STATE / "cross_type_confluence_paper_state.json"

# Event-driven catalyst engines (Session 25)
SECTOR_MOM_SPREADS_STATE = STATE / "sector_momentum_spreads_paper_state.json"
GAP_FADE_SPREAD_STATE    = STATE / "gap_fade_spread_paper_state.json"

# Previously unwired VALIDATED engines (completion gap fix)
EXTREME_IDIO_STATE       = STATE / "extreme_idio_paper_state.json"
VOLUME_SURGE_STATE       = STATE / "volume_surge_paper_state.json"
VOL_CRUSH_STATE          = STATE / "vol_crush_paper_state.json"
VIX_MR_SPREAD_STATE      = STATE / "vix_mr_spread_state.json"
BOND_YIELD_INFLOW_STATE  = STATE / "paper_engine_bond_yield_inflow.json"

# Running but not yet adversarial-validated engines (lower weight)
FACTOR_ETF_ROTATION_STATE = STATE / "factor_etf_rotation_paper_state.json"
SECTOR_REVERSAL_STATE     = STATE / "sector_reversal_paper_state.json"
STRATEGY_ROTATION_STATE   = STATE / "strategy_rotation_paper_state.json"
STRATEGY_ROTATION_V2F_STATE = STATE / "strategy_rotation_v2f_paper_state.json"
SECTOR_PAIRS_STATE        = STATE / "sector_pairs_paper_state.json"
SECTOR_V7_STATE           = STATE / "sector_combined_v7_paper_state.json"
SECTOR_V8_STATE           = STATE / "sector_combined_v8_paper_state.json"
SECTOR_V92_STATE          = STATE / "sector_combined_v92_paper_state.json"
SECTOR_EARNINGS_STANDALONE_STATE = STATE / "sector_earnings_standalone_paper_state.json"
WEEKLY_MOMENTUM_BURST_STATE = STATE / "weekly_momentum_burst_state.json"

# HC #775 — Broad market sector screener (mining, healthcare, small caps, industrials, defense)
BROAD_MARKET_SIGNALS     = STATE / "broad_market_signals.json"

# Sub-sector rotation tracker — deep sub-industry rotation signals
SUBSECTOR_ROTATION_STATE = STATE / "subsector_rotation_state.json"
SUBSECTOR_ROTATION_PREDS = STATE / "subsector_rotation_predictions.json"
SUBSECTOR_ROTATION_PAPER = STATE / "subsector_rotation_paper_state.json"

# Validated sub-sector pair rotation (5 adversarial-validated pairs, LGBM)
SUBSECTOR_VALIDATED_PAIRS_STATE = STATE / "subsector_rotation_validated_pairs_state.json"

# Strategy #13 — Signal Scoring (6/6 adversarial, Sharpe 1.556)
SIGNAL_SCORING_STATE     = STATE / "signal_scoring_paper.json"

# Strategy #15 — Sequential Chain RSI Div→Bond Yield (6/6 adversarial, Sharpe 1.528)
SEQUENTIAL_CHAIN_STATE   = STATE / "sequential_chain_paper.json"

# Strategy #16 — VIX Contango + Sector Oversold (6/6 adversarial, Sharpe 1.67, WR 62%)
VIX_CONTANGO_OVERSOLD_STATE = STATE / "vix_contango_sector_oversold_paper_state.json"

# Strategy #17 — Sector Rank Reversal B (6/6 adversarial, Sharpe 2.16, WR 56.9%)
SECTOR_RANK_REVERSAL_B_STATE = STATE / "sector_rank_reversal_b_paper_state.json"

# Strategy #18 — Earnings Outperformance E (6/6 adversarial, Sharpe 2.91, WR 62%)
EARNINGS_OUTPERF_E_STATE = STATE / "earnings_outperformance_e_paper_state.json"

# High-growth strategies (momentum growth rotation + flow reversal 2x)
PE_STATE = BASE / "paper_engines" / "state"
MOMENTUM_GROWTH_STATE          = PE_STATE / "momentum_growth_paper_state.json"
FLOW_REVERSAL_2X_STATE         = PE_STATE / "flow_reversal_2x_paper_state.json"

# AVO-evolved lockbox-validated engines (HC #799 — tradable systems)
SENTIMENT_CONTRARIAN_AVO_STATE = PE_STATE / "sentiment_contrarian_avo_state.json"
VOL_COMPRESSION_AVO_STATE      = PE_STATE / "vol_compression_avo_state.json"
UNIFIED_PORTFOLIO_STATE        = STATE / "unified_portfolio_state.json"

# Put-Call Contrarian AVO — momentum sector rotation + fear overlay (lockbox validated Sharpe 3.78)
PUT_CALL_CONTRARIAN_STATE      = PE_STATE / "put_call_contrarian_state.json"

# Macro Regime Rotation AVO — defensive dip-buy with macro dial (score 8.00, lockbox validated)
MACRO_REGIME_ROTATION_STATE    = PE_STATE / "macro_regime_rotation_state.json"

# Gold-Bond Divergence AVO — GLD/TLT z-score divergence (score 6.80, lockbox Sharpe 3.83)
GOLD_BOND_DIVERGENCE_STATE     = PE_STATE / "gold_bond_divergence_state.json"

# Insider Momentum AVO — defensive dip-buy with insider gate (score 6.50, lockbox validated)
INSIDER_MOMENTUM_STATE         = PE_STATE / "insider_momentum_state.json"

# Vol Regime Mean-Revert AVO — vol spike/compression mean-reversion (score 7.09, lockbox Sharpe 1.18)
VOL_REGIME_MEAN_REVERT_STATE   = PE_STATE / "vol_regime_mean_revert_state.json"

# Treasury Curve Steepener AVO — yield curve steepening via TLT/SHY (score 7.32, lockbox Sharpe 4.09)
TREASURY_CURVE_STEEPENER_STATE = PE_STATE / "treasury_curve_steepener_state.json"

# Trend Dip Reversion AVO — regime-adaptive sector dip buying (score 4.49, lockbox Sharpe 0.95)
TREND_DIP_REVERSION_STATE      = STATE / "trend_dip_reversion_paper_state.json"

# Breadth Momentum Regime AVO — breadth regime sector rotation (score 5.81, lockbox Sharpe 2.12)
BREADTH_MOMENTUM_REGIME_STATE  = PE_STATE / "breadth_momentum_regime_state.json"

# Cross-Asset Macro AVO — defensive sector dip-buying with macro filters (score 7.28, lockbox Sharpe 3.61)
CROSS_ASSET_MACRO_STATE        = PE_STATE / "cross_asset_macro_state.json"

# Size Rotation AVO — IWM/SPY ratio momentum (score 4.56, lockbox Sharpe 2.30)
SIZE_ROTATION_STATE            = PE_STATE / "size_rotation_state.json"

# Calendar Momentum AVO — turn-of-month rotation (score 3.64, lockbox Sharpe 2.58)
CALENDAR_MOMENTUM_STATE        = PE_STATE / "calendar_momentum_state.json"

# Credit Spread Momentum AVO — credit spread regime rotation (score 9.78, lockbox Sharpe 7.78)
CREDIT_SPREAD_MOMENTUM_STATE   = PE_STATE / "credit_spread_momentum_state.json"

# Options Execution AVO — ML execution timing (score 531.27, lockbox Sharpe 2.48)
OPTIONS_EXECUTION_AVO_STATE    = PE_STATE / "options_execution_avo_state.json"

# Options Overlay Paper Engine — amplifies equity signals via ATM calls
OPTIONS_OVERLAY_STATE          = PE_STATE / "options_overlay_paper_state.json"

# IV Rank Tracker — per-ticker IV rank/percentile for options timing
IV_RANK_DATA             = STATE / "iv_rank_data.json"

QUALITY_MOMENTUM_LOG   = LOGS / "quality_momentum_paper.log"
PEAD_DRIFT_LOG         = LOGS / "pead_drift_paper.log"
VIX_CALL_SPREAD_LOG    = LOGS / "vix_call_spread_paper.log"
SECTOR_ETF_MOM_LOG     = LOGS / "sector_etf_momentum_paper.log"

OUTPUT_FILE = STATE / "agentic_signals.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC"]
ACCOUNT_EQUITY = 645.0
MAX_POSITION_COST = 200.0
MAX_CONCURRENT = 3
MIN_CONFIDENCE = 0.6
TARGET_DTE_MIN = 21
TARGET_DTE_MAX = 45

# ---------------------------------------------------------------------------
# ROBINHOOD MCP TOOLS CONFIGURATION
# ---------------------------------------------------------------------------
# Set ENABLE_RH_TOOLS = True when MCP tools are available and credentials configured
# (see knowledge_base/rh_tools_integration.md for setup)
ENABLE_RH_TOOLS = True

# Robinhood scan IDs (retrieve with get_scans() or from Robinhood Legend UI)
RH_SCANS = {
    "high_options_volume": "a121153d-2123-4a3d-8b68-54b882070f74",
    "small_mid_cap_momentum": "b90c7262-c431-4cbf-bc4a-62ad15629112",   # HC #775: $300M-$10B, rel vol >1.5x
    "healthcare_active": "45231a64-a717-452c-b848-10637235e4ca",         # HC #775: Healthcare sector, rel vol >1.2x
    "mining_industrials_active": "dbe2f930-3907-412c-ab74-e2b9b5d2366a", # HC #775: Basic Materials + Industrials, rel vol >1.2x
}

# Technical indicator settings for RH tool enhancement
RH_TECHNICAL_INDICATORS = {
    "rsi_period": 14,
    "rsi_bullish_threshold": 60,  # RSI > 60 confirms bull signals
    "rsi_bearish_threshold": 40,  # RSI < 40 confirms bear signals
    "macd_enabled": True,
    "bollinger_enabled": True,
    "bollinger_period": 20,
    "use_technical_confluence": True,  # If True, boost confidence when indicators align
    "technical_boost_pct": 15,  # Boost confidence by 15% if technical indicators align
}

# Options liquidity requirements
RH_OPTIONS_LIMITS = {
    "min_open_interest": 50,  # Reject if OI < 50
    "max_bid_ask_spread": 0.50,  # Reject if spread > $0.50
    "use_liquidity_gate": True,  # If False, skip OI/spread checks
}

# Account info (populate when using review_option_order)
RH_ACCOUNT = {
    "account_number": os.environ.get("ROBINHOOD_ACCOUNT_NUMBER", ""),  # Agentic account — options only (HC #778)
    "test_mode": False,  # Live execution enabled — signals → options trades
}

# Earnings filter: use get_earnings_calendar if True (more reliable), yfinance if False
RH_USE_EARNINGS_CALENDAR = False  # Set to True when ready to replace yfinance


def load_json(path: Path) -> dict | None:
    """Safely load a JSON file, return None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not read {path.name}: {e}", file=sys.stderr)
        return None


def parse_log_signals(log_path: Path) -> dict:
    """Extract recent signal info from a paper engine log file."""
    result = {"last_run": None, "holdings": [], "signals": [], "status": "unknown"}
    if not log_path.exists():
        result["status"] = "missing"
        return result

    try:
        # Read last 100 lines
        with open(log_path) as f:
            lines = f.readlines()[-100:]
    except Exception:
        result["status"] = "error"
        return result

    text = "".join(lines)

    # Extract last run timestamp
    ts_matches = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
    if ts_matches:
        result["last_run"] = ts_matches[-1]

    # Holdings
    hold_match = re.findall(r"Holdings:\s*(.+)", text)
    if hold_match:
        result["holdings"] = [t.strip() for t in hold_match[-1].split(",")]

    # Top selections with scores
    score_matches = re.findall(r"\s+(\w+):\s+score=([\d.]+)", text)
    for ticker, score in score_matches:
        result["signals"].append({"ticker": ticker, "score": float(score)})

    # Check for weekend skip or no-signal states
    if "Weekend" in text or "skipping" in text:
        result["status"] = "weekend_skip"
    elif "No PEAD signals" in text:
        result["status"] = "no_signals"
    else:
        result["status"] = "active"

    return result


def get_market_data() -> dict:
    """Fetch current market data via yfinance. Returns dict with VIX, sector data, etc."""
    try:
        import yfinance as yf
    except ImportError:
        print("  [WARN] yfinance not installed, using cached signal watcher data", file=sys.stderr)
        return {"available": False}

    data: dict[str, Any] = {"available": True, "sectors": {}, "vix": None}

    # VIX
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if not hist.empty:
            data["vix"] = float(hist["Close"].iloc[-1])
    except Exception as e:
        print(f"  [WARN] VIX fetch failed: {e}", file=sys.stderr)

    # VIX3M for term structure
    try:
        vix3m = yf.Ticker("^VIX3M")
        hist3m = vix3m.history(period="5d")
        if not hist3m.empty:
            data["vix3m"] = float(hist3m["Close"].iloc[-1])
            if data["vix"] is not None:
                data["vix_term_structure"] = "contango" if data["vix"] < data["vix3m"] else "backwardation"
                data["vix_term_ratio"] = round(data["vix"] / data["vix3m"], 3)
    except Exception as e:
        print(f"  [WARN] VIX3M fetch failed: {e}", file=sys.stderr)

    # Credit Spread Velocity (HYG-TLT) — confluence filter for financials/RE
    # When credit widens fast, penalize bullish XLF/XLRE/XLY signals
    try:
        hyg = yf.Ticker("HYG")
        tlt = yf.Ticker("TLT")
        hyg_hist = hyg.history(period="15d")
        tlt_hist = tlt.history(period="15d")
        if not hyg_hist.empty and not tlt_hist.empty and len(hyg_hist) >= 4 and len(tlt_hist) >= 4:
            hyg_ret = hyg_hist["Close"].pct_change().dropna()
            tlt_ret = tlt_hist["Close"].pct_change().dropna()
            # Align on common dates
            common = hyg_ret.index.intersection(tlt_ret.index)
            if len(common) >= 3:
                csv_3d = float((hyg_ret.loc[common] - tlt_ret.loc[common]).iloc[-3:].sum())
                data["credit_spread_velocity_3d"] = round(csv_3d * 100, 3)  # in pct
                # Negative = credit widening (HYG underperforming TLT) = risk-off
                data["credit_stress"] = "widening" if csv_3d < -0.01 else ("tightening" if csv_3d > 0.01 else "stable")
    except Exception as e:
        print(f"  [WARN] Credit spread velocity fetch failed: {e}", file=sys.stderr)

    # Yield Curve (3m10s) — market-level risk filter
    # Rapid 3m10s steepening is bearish for all equities
    try:
        tnx = yf.Ticker("^TNX")  # 10Y yield
        irx = yf.Ticker("^IRX")  # 13-week T-bill yield
        tnx_hist = tnx.history(period="15d")
        irx_hist = irx.history(period="15d")
        if not tnx_hist.empty and not irx_hist.empty and len(tnx_hist) >= 6 and len(irx_hist) >= 6:
            spread_now = float(tnx_hist["Close"].iloc[-1] - irx_hist["Close"].iloc[-1])
            spread_5d_ago = float(tnx_hist["Close"].iloc[-6] - irx_hist["Close"].iloc[-6])
            curve_5d_change = spread_now - spread_5d_ago
            data["yield_curve_3m10s"] = round(spread_now, 3)
            data["yield_curve_5d_change"] = round(curve_5d_change, 3)
            # Rapid steepening (>0.10) = bearish caution flag
            data["curve_stress"] = "steepening_fast" if curve_5d_change > 0.10 else ("flattening_fast" if curve_5d_change < -0.10 else "stable")
    except Exception as e:
        print(f"  [WARN] Yield curve fetch failed: {e}", file=sys.stderr)

    # SPY for relative strength baseline
    spy_data = None
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="65d")
        if not spy_hist.empty and len(spy_hist) > 1:
            spy_data = spy_hist
    except Exception:
        pass

    # Sector ETFs
    for ticker in SECTOR_ETFS:
        try:
            etf = yf.Ticker(ticker)
            hist = etf.history(period="65d")
            if hist.empty or len(hist) < 10:
                continue

            close = hist["Close"]
            current = float(close.iloc[-1])

            # Momentum: 21-day return
            mom_21d = float((close.iloc[-1] / close.iloc[-22] - 1) * 100) if len(close) >= 22 else 0.0

            # RSI-14
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss.replace(0, float("nan"))
            rsi = float(100 - (100 / (1 + rs.iloc[-1]))) if not rs.empty else 50.0

            # Relative strength vs SPY (21d)
            rel_str = 0.0
            if spy_data is not None and len(spy_data) >= 22:
                spy_ret = float((spy_data["Close"].iloc[-1] / spy_data["Close"].iloc[-22] - 1) * 100)
                rel_str = mom_21d - spy_ret

            # 50-day SMA trend
            sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else current
            above_sma50 = current > sma50

            # Momentum acceleration (5d vs 21d)
            mom_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0.0

            data["sectors"][ticker] = {
                "price": round(current, 2),
                "mom_21d": round(mom_21d, 2),
                "mom_5d": round(mom_5d, 2),
                "rsi": round(rsi, 1),
                "rel_str_vs_spy": round(rel_str, 2),
                "above_sma50": above_sma50,
            }
        except Exception as e:
            print(f"  [WARN] {ticker} data fetch failed: {e}", file=sys.stderr)

    return data


def determine_regime(vix_level: float | None, signal_watcher: dict | None) -> str:
    """Determine market regime: bull / bear / neutral."""
    if vix_level is None:
        # Fall back to signal watcher
        if signal_watcher:
            vr = signal_watcher.get("vix_regime", "NORMAL")
            if vr == "ELEVATED":
                return "bear"
            vix_level = signal_watcher.get("vix_level") or signal_watcher.get("vmr", {}).get("vix")
        if vix_level is None:
            return "neutral"

    # VIX >= 20 = elevated vol = bearish tilt; < 15 = very calm = bullish; 15-20 = neutral
    if vix_level >= 25:
        return "bear"
    elif vix_level >= 20:
        return "neutral"  # elevated but not panic
    elif vix_level < 15:
        return "bull"
    else:
        return "neutral"


def build_ticker_signals(
    sector_spreads: dict | None,
    quality_momentum: dict | None,
    sector_etf_mom: dict | None,
    pead_drift: dict | None,
    vix_state: dict | None,
    signal_watcher: dict | None,
    quality_log: dict | None,
    sector_log: dict | None,
    market_data: dict,
    sector_v91: dict | None = None,
    sector_v93: dict | None = None,
    sector_v10: dict | None = None,
) -> dict[str, dict]:
    """
    Build a per-ticker signal map.
    Each ticker -> {direction, sources: [list of confirming engines], details: {}}
    """
    signals: dict[str, dict] = {}

    def ensure(ticker: str):
        if ticker not in signals:
            signals[ticker] = {
                "bull_sources": [],
                "bear_sources": [],
                "details": {},
                "scores": [],
            }

    # --- 1. Sector Spreads Paper Engine ---
    if sector_spreads and "open_positions" in sector_spreads:
        for pos in sector_spreads["open_positions"]:
            t = pos.get("ticker", "")
            if t not in SECTOR_ETFS:
                continue
            ensure(t)
            mode = pos.get("mode", "")
            n_sig = pos.get("n_signals", 0)
            lgbm = pos.get("lgbm_score", 0)
            sig_list = pos.get("signals", [])

            source_name = "sector_spreads"
            if mode == "bear":
                signals[t]["bear_sources"].append(source_name)
                signals[t]["scores"].append(-lgbm if lgbm else -0.5)
            elif mode == "bull":
                signals[t]["bull_sources"].append(source_name)
                signals[t]["scores"].append(lgbm if lgbm else 0.5)
            signals[t]["details"]["sector_spreads"] = {
                "mode": mode, "n_signals": n_sig, "lgbm_score": lgbm,
                "signals": sig_list,
            }

    # --- 2. Quality Momentum (stock-level — map stocks to sector ETFs) ---
    stock_to_sector = {
        "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "INTC": "XLK",
        "TXN": "XLK", "CSCO": "XLK", "AMD": "XLK", "CRM": "XLK", "ADBE": "XLK",
        "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
        "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "EOG": "XLE",
        "UNH": "XLV", "JNJ": "XLV", "PFE": "XLV", "ABBV": "XLV", "MRK": "XLV",
        "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
        "CAT": "XLI", "HON": "XLI", "UPS": "XLI", "GE": "XLI", "BA": "XLI",
        "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP",
        "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU", "AEP": "XLU",
        "AMT": "XLRE", "PLD": "XLRE", "CCI": "XLRE", "SPG": "XLRE", "O": "XLRE",
        "LIN": "XLB", "APD": "XLB", "SHW": "XLB", "FCX": "XLB", "NEM": "XLB",
        "META": "XLC", "GOOG": "XLC", "GOOGL": "XLC", "DIS": "XLC", "NFLX": "XLC",
    }
    if quality_momentum and "holdings" in quality_momentum:
        for stock in quality_momentum["holdings"]:
            sector = stock_to_sector.get(stock)
            if sector and sector in SECTOR_ETFS:
                ensure(sector)
                score = 0
                if quality_momentum.get("rebalance_history"):
                    last_reb = quality_momentum["rebalance_history"][-1]
                    score = last_reb.get("scores", {}).get(stock, 0)
                signals[sector]["bull_sources"].append(f"quality_momentum({stock})")
                signals[sector]["scores"].append(score if score else 0.3)
                signals[sector]["details"].setdefault("quality_momentum_stocks", []).append(
                    {"stock": stock, "score": score}
                )

    # --- 3. Sector ETF Momentum ---
    if sector_etf_mom and "holdings" in sector_etf_mom:
        for etf in sector_etf_mom["holdings"]:
            if etf not in SECTOR_ETFS:
                continue
            ensure(etf)
            score = 0
            if sector_etf_mom.get("rebalance_history"):
                last_reb = sector_etf_mom["rebalance_history"][-1]
                score = last_reb.get("scores", {}).get(etf, 0)
            signals[etf]["bull_sources"].append("sector_etf_momentum")
            signals[etf]["scores"].append(score if score else 0.3)
            signals[etf]["details"]["sector_etf_momentum"] = {"score": score}

    # --- 4. PEAD Drift ---
    if pead_drift and pead_drift.get("positions"):
        for pos in pead_drift["positions"]:
            t = pos.get("ticker", "")
            direction = pos.get("direction", "long")
            sector = stock_to_sector.get(t)
            if sector and sector in SECTOR_ETFS:
                ensure(sector)
                if direction == "long":
                    signals[sector]["bull_sources"].append(f"pead_drift({t})")
                else:
                    signals[sector]["bear_sources"].append(f"pead_drift({t})")

    # --- 5. VIX Call Spread State — DISABLED (adversarial 2/6, Sharpe -2.09, losses overwhelm wins) ---
    # if vix_state and vix_state.get("positions"):
    #     for etf in SECTOR_ETFS:
    #         ensure(etf)
    #         signals[etf]["bear_sources"].append("vix_elevated")

    # --- 6. Signal Watcher ---
    if signal_watcher:
        protection = signal_watcher.get("protection", {})
        if not protection.get("spy_above_50sma", True):
            # SPY below 50 SMA = bearish signal for all sectors
            for etf in SECTOR_ETFS:
                ensure(etf)
                signals[etf]["bear_sources"].append("spy_below_50sma")

        cta_uptrend = signal_watcher.get("cta", {}).get("uptrend_tickers", [])
        # Map CTA trends: USO -> XLE, UUP -> bearish equities, DBA -> XLB
        cta_sector_map = {"USO": "XLE", "DBA": "XLB"}
        for cta_tick in cta_uptrend:
            mapped = cta_sector_map.get(cta_tick)
            if mapped:
                ensure(mapped)
                signals[mapped]["bull_sources"].append(f"cta_trend({cta_tick})")

    # --- 7. Fresh Market Data Overlay ---
    if market_data.get("available"):
        for etf, md in market_data.get("sectors", {}).items():
            if etf not in SECTOR_ETFS:
                continue
            ensure(etf)

            # Strong momentum confirmation
            if md["mom_21d"] > 3.0 and md["rel_str_vs_spy"] > 1.0:
                signals[etf]["bull_sources"].append("strong_momentum")
                signals[etf]["scores"].append(0.3)
            elif md["mom_21d"] < -3.0 and md["rel_str_vs_spy"] < -1.0:
                signals[etf]["bear_sources"].append("weak_momentum")
                signals[etf]["scores"].append(-0.3)

            # RSI confirmation
            if md["rsi"] > 60 and md["above_sma50"]:
                signals[etf]["bull_sources"].append("rsi_bullish")
            elif md["rsi"] < 40 and not md["above_sma50"]:
                signals[etf]["bear_sources"].append("rsi_bearish")

            # Momentum acceleration
            if md["mom_5d"] > 1.5 and md["mom_21d"] > 0:
                signals[etf]["bull_sources"].append("mom_accelerating")
            elif md["mom_5d"] < -1.5 and md["mom_21d"] < 0:
                signals[etf]["bear_sources"].append("mom_decelerating")

            signals[etf]["details"]["market_data"] = md

    # --- 8. V10/V9.x Paper Engine Positions (HIGHEST WEIGHT — best strategies) ---
    # These are our highest-Sharpe validated strategies. Their current positions
    # are strong directional signals for the agentic account.
    for state_name, state_data, weight in [
        ("v10_optimal", sector_v10, 0.85),    # V10: Sharpe 6.14 — highest weight
        ("v93_profit_target", sector_v93, 0.7),  # V9.3: Sharpe 5.12
        ("v91_monthly", sector_v91, 0.6),     # V9.1: Sharpe 2.64
    ]:
        if not state_data or "open_positions" not in state_data:
            continue
        for pos in state_data["open_positions"]:
            t = pos.get("ticker", "")
            if t not in SECTOR_ETFS:
                continue
            ensure(t)
            mode = pos.get("mode", "")
            lgbm = pos.get("lgbm_score", 0)
            source = state_name
            if mode == "bear":
                signals[t]["bear_sources"].append(source)
                signals[t]["scores"].append(-weight)
            elif mode == "bull":
                signals[t]["bull_sources"].append(source)
                signals[t]["scores"].append(weight)
            signals[t]["details"][state_name] = {
                "mode": mode, "lgbm_score": lgbm,
                "entry_date": pos.get("entry_date"),
                "vix_mode": pos.get("vix_mode"),
            }

    # --- 9. Momentum Options Paper Engine (KB #281 — validated, Sharpe 1.28) ---
    momentum_opts = load_json(MOMENTUM_OPTIONS_STATE)
    if momentum_opts and "open_positions" in momentum_opts:
        for pos in momentum_opts["open_positions"]:
            t = pos.get("ticker", "")
            if t not in SECTOR_ETFS:
                continue
            ensure(t)
            opt_type = pos.get("option_type", "call")
            lgbm = pos.get("lgbm_score", 0)
            if opt_type == "put":
                signals[t]["bear_sources"].append("momentum_burst")
                signals[t]["scores"].append(-0.65)
            else:
                signals[t]["bull_sources"].append("momentum_burst")
                signals[t]["scores"].append(0.65)
            signals[t]["details"]["momentum_burst"] = {
                "option_type": opt_type, "lgbm_score": lgbm,
                "entry_date": pos.get("entry_date"),
            }

    # --- 10. Equity Rotation Paper Engine (KB #285 — validated, Sharpe 1.40) ---
    equity_rot = load_json(EQUITY_ROTATION_STATE)
    if equity_rot:
        rankings = equity_rot.get("rankings", {})
        positions = equity_rot.get("positions", [])
        held_tickers = {p.get("ticker") for p in positions}
        for etf in SECTOR_ETFS:
            if etf in rankings:
                ensure(etf)
                rank_score = rankings[etf]
                # Top-ranked sectors (>0.6) = bull signal, bottom (<0.4) = bear
                if etf in held_tickers or rank_score > 0.6:
                    signals[etf]["bull_sources"].append("equity_rotation_rank")
                    signals[etf]["scores"].append(rank_score * 0.7)
                elif rank_score < 0.4:
                    signals[etf]["bear_sources"].append("equity_rotation_rank")
                    signals[etf]["scores"].append(-0.3)
                signals[etf]["details"]["equity_rotation"] = {
                    "lgbm_rank_score": rank_score,
                    "in_portfolio": etf in held_tickers,
                }

    # --- 11. PEAD ML Paper Engine (Session 24 — Sharpe 1.51, WR 52%) ---
    pead_ml = load_json(PEAD_ML_STATE)
    if pead_ml and pead_ml.get("open_positions"):
        for pos in pead_ml["open_positions"]:
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            opt_type = pos.get("option_type", "call")
            conf = pos.get("ml_confidence", 0.6)
            gap = pos.get("gap_pct", 0)
            if opt_type == "call" or gap > 0:
                signals[t]["bull_sources"].append("pead_ml")
                signals[t]["scores"].append(conf * 0.8)
            else:
                signals[t]["bear_sources"].append("pead_ml")
                signals[t]["scores"].append(-conf * 0.8)
            signals[t]["details"]["pead_ml"] = {
                "option_type": opt_type,
                "ml_confidence": conf,
                "gap_pct": gap,
                "earnings_date": pos.get("earnings_date"),
            }

    # Also check PEAD ML scores (real-time scoring output)
    pead_scores = load_json(PEAD_ML_SCORES_STATE)
    if pead_scores and isinstance(pead_scores, list):
        for score in pead_scores:
            t = score.get("ticker", "")
            if not t or not score.get("trade_recommended"):
                continue
            ensure(t)
            conf = score.get("ml_confidence", 0.5)
            gap = score.get("gap_pct", 0)
            if gap and gap > 0:
                signals[t]["bull_sources"].append("pead_ml_score")
                signals[t]["scores"].append(conf * 0.75)
            elif gap and gap < 0:
                signals[t]["bear_sources"].append("pead_ml_score")
                signals[t]["scores"].append(-conf * 0.75)

    # --- 12. IV Run-Up Paper Engine (Session 24 — Sharpe 2.60, WR 77%) ---
    iv_runup = load_json(IV_RUNUP_STATE)
    if iv_runup and iv_runup.get("open_positions"):
        for pos in iv_runup["open_positions"]:
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            # IV Run-Up is non-directional (straddle) — add as both bull and bear
            # This signals "high expected vol" for this ticker
            signals[t]["details"]["iv_runup"] = {
                "option_type": "straddle",
                "earnings_date": pos.get("earnings_date"),
                "entry_iv": pos.get("entry_iv"),
                "current_pnl_pct": pos.get("current_pnl_pct", 0),
            }
            # Don't add to bull/bear sources since straddle is non-directional
            # But flag it as an active earnings play
            signals[t]["bull_sources"].append("iv_runup_active")
            signals[t]["bear_sources"].append("iv_runup_active")

    # --- 13. Earnings Gap Alert System ---
    gap_signals = load_json(EARNINGS_GAP_STATE)
    if gap_signals and isinstance(gap_signals, list):
        for sig in gap_signals:
            t = sig.get("ticker", "")
            if not t:
                continue
            ensure(t)
            gap = sig.get("gap_pct", 0)
            if gap > 5:
                signals[t]["bull_sources"].append("earnings_gap_alert")
                signals[t]["scores"].append(0.6)
            elif gap < -5:
                signals[t]["bear_sources"].append("earnings_gap_alert")
                signals[t]["scores"].append(-0.6)
            signals[t]["details"]["earnings_gap"] = {
                "gap_pct": gap,
                "signal_date": sig.get("date"),
            }

    # --- 14. Sector Momentum Spreads — DISABLED (adversarial 2/6, Sharpe -4.13, 15 trades, no edge) ---
    # sector_mom_spreads = load_json(SECTOR_MOM_SPREADS_STATE)
    # if sector_mom_spreads and sector_mom_spreads.get("open_positions"):
    #     for pos in sector_mom_spreads["open_positions"]:
    #         t = pos.get("ticker", "")
    #         if t not in SECTOR_ETFS:
    #             continue
    #         ensure(t)
    #         spread_type = pos.get("spread_type", "")
    #         rank_score = pos.get("rank_score", 0)
    #         if spread_type == "bull_call":
    #             signals[t]["bull_sources"].append("sector_rotation_leader")
    #             signals[t]["scores"].append(0.55)
    #         elif spread_type == "bear_put":
    #             signals[t]["bear_sources"].append("sector_rotation_laggard")
    #             signals[t]["scores"].append(-0.55)
    #         signals[t]["details"]["sector_mom_spread"] = {
    #             "spread_type": spread_type,
    #             "rank_score": rank_score,
    #             "pair_id": pos.get("pair_id"),
    #             "entry_date": pos.get("entry_date"),
    #         }

    # --- 15. Gap Fade Spread Paper Engine (Session 25 — mean-reversion on gap-downs) ---
    gap_fade = load_json(GAP_FADE_SPREAD_STATE)
    if gap_fade and gap_fade.get("open_positions"):
        for pos in gap_fade["open_positions"]:
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            gap_pct = pos.get("gap_pct", 0)
            signals[t]["bull_sources"].append("gap_fade_reversion")
            signals[t]["scores"].append(0.60)
            signals[t]["details"]["gap_fade"] = {
                "gap_pct": gap_pct,
                "volume_ratio": pos.get("volume_ratio"),
                "entry_date": pos.get("entry_date"),
            }

    # --- 16. Bond Yield + Sub-Sector Inflow Confluence ---
    # Validated: Sharpe 2.25 -> 2.91 with inflow filter (perm p=0.018)
    # Boost bond_yield signals when sub-sector is in INFLOW, penalize on OUTFLOW
    bond_yield_state = load_json(BOND_YIELD_STATE)
    subsector_rot_state = load_json(SUBSECTOR_ROTATION_STATE)
    if bond_yield_state and bond_yield_state.get("positions"):
        # Check staleness of rotation data (>2 days old = neutral)
        rotation_fresh = False
        if subsector_rot_state and subsector_rot_state.get("generated_at"):
            try:
                gen_time = datetime.fromisoformat(subsector_rot_state["generated_at"])
                age_hours = (datetime.now() - gen_time).total_seconds() / 3600
                rotation_fresh = age_hours < 48  # 2 days
            except Exception:
                pass

        # Build ticker -> subsector lookup from rotation state
        ticker_to_subsector = {}
        if rotation_fresh and subsector_rot_state and subsector_rot_state.get("subsectors"):
            for sub_name, sub_data in subsector_rot_state["subsectors"].items():
                if sub_data.get("tickers"):
                    for tk in sub_data["tickers"]:
                        ticker_to_subsector[tk] = sub_name

        for pos in bond_yield_state["positions"]:
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            signals[t]["bull_sources"].append("bond_yield")
            signals[t]["scores"].append(0.55)
            signals[t]["details"]["bond_yield"] = {
                "entry_price": pos.get("entry_price"),
                "entry_date": pos.get("entry_date"),
                "reason": pos.get("reason", ""),
            }

            # Apply sub-sector rotation confluence adjustment
            if rotation_fresh and t in ticker_to_subsector:
                sub_name = ticker_to_subsector[t]
                sub_data = subsector_rot_state["subsectors"].get(sub_name, {})
                phase = sub_data.get("rotation_phase", "NEUTRAL")
                rot_score = sub_data.get("rotation_score", 0)

                if phase in ("INFLOW", "ACCUMULATING") and rot_score > 0:
                    # INFLOW: boost confidence +0.05, add rotation_confluence source
                    signals[t]["scores"].append(0.05)
                    signals[t]["bull_sources"].append("rotation_confluence")
                    signals[t]["details"]["bond_yield"]["rotation_boost"] = {
                        "subsector": sub_name,
                        "phase": phase,
                        "rotation_score": rot_score,
                        "adjustment": "+0.05",
                    }
                elif phase in ("OUTFLOW", "DISTRIBUTING") and rot_score < 0:
                    # OUTFLOW: mild penalty -0.03
                    signals[t]["scores"].append(-0.03)
                    signals[t]["details"]["bond_yield"]["rotation_penalty"] = {
                        "subsector": sub_name,
                        "phase": phase,
                        "rotation_score": rot_score,
                        "adjustment": "-0.03",
                    }
                # NEUTRAL or stale: no adjustment

    # --- 17. Extreme Idiosyncratic Moves (VALIDATED — adversarial pass) ---
    extreme_idio = load_json(EXTREME_IDIO_STATE)
    if extreme_idio and extreme_idio.get("positions"):
        for pos in extreme_idio["positions"]:
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            direction = pos.get("direction", "long")
            if direction == "long":
                signals[t]["bull_sources"].append("extreme_idio")
                signals[t]["scores"].append(0.50)
            else:
                signals[t]["bear_sources"].append("extreme_idio")
                signals[t]["scores"].append(-0.50)
            signals[t]["details"]["extreme_idio"] = {
                "direction": direction,
                "entry_date": pos.get("entry_date"),
                "entry_price": pos.get("entry_price"),
                "regime": pos.get("regime"),
            }

    # --- 18. Volume Surge (VALIDATED — adversarial pass) ---
    volume_surge = load_json(VOLUME_SURGE_STATE)
    if volume_surge and volume_surge.get("positions"):
        for pos in volume_surge["positions"]:
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            direction = pos.get("direction", "long")
            if direction == "long":
                signals[t]["bull_sources"].append("volume_surge")
                signals[t]["scores"].append(0.45)
            else:
                signals[t]["bear_sources"].append("volume_surge")
                signals[t]["scores"].append(-0.45)
            signals[t]["details"]["volume_surge"] = {
                "direction": direction,
                "entry_date": pos.get("entry_date"),
                "volume_ratio": pos.get("volume_ratio"),
            }

    # --- 19. Vol Crush — DISABLED (adversarial 2/6, 0% WR, Sharpe -36, commission > credit) ---
    # vol_crush = load_json(VOL_CRUSH_STATE)
    # if vol_crush and vol_crush.get("positions"):
    #     for pos in vol_crush["positions"]:
    #         t = pos.get("ticker", "")
    #         if not t:
    #             continue
    #         ensure(t)
    #         signals[t]["bull_sources"].append("vol_crush")
    #         signals[t]["scores"].append(0.40)
    #         signals[t]["details"]["vol_crush"] = {
    #             "ticker": t,
    #             "entry_date": pos.get("entry_date"),
    #             "iv_rank": pos.get("iv_rank"),
    #             "earnings_date": pos.get("earnings_date"),
    #         }

    # --- 20. VIX Mean-Reversion Spread (VALIDATED — adversarial pass) ---
    vix_mr = load_json(VIX_MR_SPREAD_STATE)
    if vix_mr and vix_mr.get("positions"):
        for pos in vix_mr["positions"]:
            # VIX MR positions signal elevated vol is expected to revert
            # This is bullish for equities
            for etf in SECTOR_ETFS:
                ensure(etf)
                signals[etf]["bull_sources"].append("vix_mr_revert")
                signals[etf]["scores"].append(0.25)
            break  # Only apply once (VIX is market-wide)

    # --- 21. Bond Yield + Inflow Confluence (VALIDATED — adversarial pass) ---
    bond_inflow = load_json(BOND_YIELD_INFLOW_STATE)
    if bond_inflow and bond_inflow.get("positions"):
        for pos in bond_inflow["positions"]:
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            signals[t]["bull_sources"].append("bond_yield_inflow")
            signals[t]["scores"].append(0.55)
            signals[t]["details"]["bond_yield_inflow"] = {
                "entry_date": pos.get("entry_date"),
                "entry_price": pos.get("entry_price"),
            }

    # --- 23. Signal Scoring #13 (6/6 adversarial, Sharpe 1.556 — HIGH WEIGHT) ---
    sig_scoring = load_json(SIGNAL_SCORING_STATE)
    if sig_scoring and sig_scoring.get("positions"):
        for pos in sig_scoring["positions"]:
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            signals[t]["bull_sources"].append("signal_scoring_13")
            signals[t]["scores"].append(0.60)  # High weight — 6/6 adversarial
            signals[t]["details"]["signal_scoring_13"] = {
                "score": pos.get("score"),
                "entry_date": pos.get("entry_date"),
                "entry_price": pos.get("entry_price"),
            }

    # --- 24. Sequential Chain #15 (6/6 adversarial, Sharpe 1.528 — HIGH WEIGHT) ---
    seq_chain = load_json(SEQUENTIAL_CHAIN_STATE)
    if seq_chain and seq_chain.get("positions"):
        for pos in seq_chain["positions"]:
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            signals[t]["bull_sources"].append("sequential_chain_15")
            signals[t]["scores"].append(0.55)  # High weight — 6/6 adversarial
            signals[t]["details"]["sequential_chain_15"] = {
                "setup_date": pos.get("setup_date"),
                "entry_date": pos.get("entry_date"),
                "entry_price": pos.get("entry_price"),
            }

    # --- 25. VIX Contango + Sector Oversold #16 (6/6 adversarial, Sharpe 1.67, WR 62%) ---
    vix_contango_oversold = load_json(VIX_CONTANGO_OVERSOLD_STATE)
    if vix_contango_oversold:
        # Read positions
        for pos in vix_contango_oversold.get("positions", []):
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            strength = pos.get("sector_sharpe", 1.0)
            signals[t]["bull_sources"].append("vix_contango_oversold_16")
            signals[t]["scores"].append(0.40)
            signals[t]["details"]["vix_contango_oversold_16"] = {
                "entry_date": pos.get("entry_date"),
                "entry_price": pos.get("entry_price"),
                "rsi_at_entry": pos.get("rsi_at_entry"),
                "vix_at_entry": pos.get("vix_at_entry"),
            }
        # Read daily signals (even if no open position, the signal matters)
        for sig in vix_contango_oversold.get("signals", []):
            t = sig.get("ticker", "")
            if not t or sig.get("direction") != "long":
                continue
            if t not in [p.get("ticker") for p in vix_contango_oversold.get("positions", [])]:
                ensure(t)
                signals[t]["bull_sources"].append("vix_contango_oversold_16_signal")
                signals[t]["scores"].append(0.40 * sig.get("strength", 0.5))

    # --- 26. Sector Rank Reversal B #17 (6/6 adversarial, Sharpe 2.16, WR 56.9%) ---
    rank_reversal_b = load_json(SECTOR_RANK_REVERSAL_B_STATE)
    if rank_reversal_b:
        for pos in rank_reversal_b.get("positions", []):
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            signals[t]["bull_sources"].append("rank_reversal_b_17")
            signals[t]["scores"].append(0.40)
            signals[t]["details"]["rank_reversal_b_17"] = {
                "entry_date": pos.get("entry_date"),
                "entry_price": pos.get("entry_price"),
                "rank_at_entry": pos.get("rank_at_entry"),
                "ret_20d_at_entry": pos.get("ret_20d_at_entry"),
                "mom_3d_at_entry": pos.get("mom_3d_at_entry"),
            }
        for sig in rank_reversal_b.get("signals", []):
            t = sig.get("ticker", "")
            if not t or sig.get("direction") != "long":
                continue
            if t not in [p.get("ticker") for p in rank_reversal_b.get("positions", [])]:
                ensure(t)
                signals[t]["bull_sources"].append("rank_reversal_b_17_signal")
                signals[t]["scores"].append(0.40 * sig.get("strength", 0.5))

    # --- 27. Earnings Outperformance E #18 (6/6 adversarial, Sharpe 2.91, WR 62%) ---
    earnings_outperf_e = load_json(EARNINGS_OUTPERF_E_STATE)
    if earnings_outperf_e:
        for pos in earnings_outperf_e.get("positions", []):
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            signals[t]["bull_sources"].append("earnings_outperf_e_18")
            signals[t]["scores"].append(0.40)
            signals[t]["details"]["earnings_outperf_e_18"] = {
                "entry_date": pos.get("entry_date"),
                "entry_price": pos.get("entry_price"),
                "excess_return_at_entry": pos.get("excess_return_at_entry"),
            }
        for sig in earnings_outperf_e.get("signals", []):
            t = sig.get("ticker", "")
            if not t or sig.get("direction") != "long":
                continue
            if t not in [p.get("ticker") for p in earnings_outperf_e.get("positions", [])]:
                ensure(t)
                signals[t]["bull_sources"].append("earnings_outperf_e_18_signal")
                signals[t]["scores"].append(0.40 * sig.get("strength", 0.5))

    # --- 22. Factor ETF Rotation (running, not yet adversarial-validated — lower weight) ---
    factor_etf = load_json(FACTOR_ETF_ROTATION_STATE)
    if factor_etf and factor_etf.get("positions"):
        positions = factor_etf["positions"]
        # Handle both dict (ticker->data) and list formats
        if isinstance(positions, dict):
            tickers = list(positions.keys())
        elif isinstance(positions, list):
            tickers = [p.get("ticker", "") for p in positions if isinstance(p, dict)]
        else:
            tickers = []
        for t in tickers:
            if not t:
                continue
            ensure(t)
            signals[t]["bull_sources"].append("factor_etf_rotation")
            signals[t]["scores"].append(0.30)

    # --- 23. Sector Reversal (running, not yet validated — lower weight) ---
    sector_rev = load_json(SECTOR_REVERSAL_STATE)
    if sector_rev and sector_rev.get("positions"):
        for pos in sector_rev["positions"]:
            t = pos.get("ticker", "")
            if not t or t not in SECTOR_ETFS:
                continue
            ensure(t)
            mode = pos.get("mode", pos.get("direction", "long"))
            if mode in ("bear", "short"):
                signals[t]["bear_sources"].append("sector_reversal")
                signals[t]["scores"].append(-0.30)
            else:
                signals[t]["bull_sources"].append("sector_reversal")
                signals[t]["scores"].append(0.30)

    # --- ML SIGNAL QUALITY ADJUSTMENTS (Session 83, 97-trade analysis) ---
    # Statistically significant losers get half-weight scores
    # strong_momentum: 23% WR (p=0.018), sector_etf_momentum: 36% WR (p=0.037),
    # pead_drift: 14% WR (p=0.044)
    ML_PENALIZED_SOURCES = {"strong_momentum", "sector_etf_momentum", "pead_drift"}
    # Toxic combinations — when both present, reduce confidence further
    TOXIC_COMBOS = [
        {"cta_trend", "strong_momentum"},      # 14% WR together
        {"cta_trend", "mom_accelerating"},      # 14% WR together
        # {"pead_drift", "vol_crush"},  # vol_crush disabled (adversarial fail)            # 33% WR together
    ]
    for t in signals:
        all_sources = signals[t]["bull_sources"] + signals[t]["bear_sources"]
        source_set = set(all_sources)
        # Halve the contribution of ML-penalized sources
        for i, score in enumerate(signals[t]["scores"]):
            # Find corresponding source
            if i < len(signals[t]["bull_sources"]):
                src = signals[t]["bull_sources"][i]
            elif i < len(signals[t]["bull_sources"]) + len(signals[t]["bear_sources"]):
                src = signals[t]["bear_sources"][i - len(signals[t]["bull_sources"])]
            else:
                continue
            if src in ML_PENALIZED_SOURCES:
                signals[t]["scores"][i] = score * 0.5
        # Flag toxic combinations
        for combo in TOXIC_COMBOS:
            if combo.issubset(source_set):
                signals[t].setdefault("warnings", []).append(
                    f"TOXIC COMBO: {' + '.join(combo)} — historically poor WR when combined"
                )

    # --- 24. Sector Pairs (running, not yet validated — lower weight) ---
    sector_pairs = load_json(SECTOR_PAIRS_STATE)
    if sector_pairs and sector_pairs.get("positions"):
        for pos in sector_pairs["positions"]:
            t = pos.get("ticker", "")
            if not t:
                continue
            ensure(t)
            direction = pos.get("direction", pos.get("side", "long"))
            if direction in ("short", "bear"):
                signals[t]["bear_sources"].append("sector_pairs")
                signals[t]["scores"].append(-0.25)
            else:
                signals[t]["bull_sources"].append("sector_pairs")
                signals[t]["scores"].append(0.25)

    return signals


def compute_confidence(bull_sources: list, bear_sources: list) -> tuple[str, float]:
    """
    Compute direction and confidence from confirming sources.
    Returns (direction, confidence_score 0-1).
    """
    n_bull = len(bull_sources)
    n_bear = len(bear_sources)

    if n_bull == 0 and n_bear == 0:
        return "neutral", 0.0

    if n_bull > n_bear:
        direction = "bull"
        confirming = n_bull
        conflicting = n_bear
    elif n_bear > n_bull:
        direction = "bear"
        confirming = n_bear
        conflicting = n_bull
    else:
        # Tied — low confidence
        return "neutral", 0.2

    # Base confidence from number of confirming sources
    # 1 source = 0.3, 2 = 0.55, 3 = 0.7, 4 = 0.8, 5+ = 0.9
    base_map = {1: 0.30, 2: 0.55, 3: 0.70, 4: 0.80}
    base = base_map.get(confirming, 0.90)

    # Penalty for conflicting sources
    penalty = conflicting * 0.15
    confidence = max(0.1, base - penalty)

    return direction, round(confidence, 2)


def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given month/year (standard monthly expiry)."""
    # calendar.monthcalendar returns weeks; Friday is index 4
    cal = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
    return date(year, month, fridays[2])  # 3rd Friday (0-indexed: [2])


def find_standard_expiry(ref_date: date | None = None,
                         dte_min: int = 21, dte_max: int = 45) -> date:
    """
    Find the nearest STANDARD monthly option expiry (3rd Friday) within
    dte_min..dte_max days from ref_date.

    RULES (liquidity-first):
      1. Prefer standard monthly (3rd Friday) within dte_min..dte_max.
      2. If none in range, pick the nearest monthly with >= 21 DTE (even
         if slightly outside dte_max — monthly liquidity beats DTE precision).
      3. Only fall back to a weekly Friday if no monthly has >= 21 DTE
         (should never happen in practice).
      4. NEVER return a non-Friday date.
    """
    if ref_date is None:
        ref_date = date.today()

    # Collect 3rd-Friday monthlies for the next 4 months
    monthlies: list[tuple[date, int]] = []
    for month_offset in range(0, 5):
        y = ref_date.year + (ref_date.month + month_offset - 1) // 12
        m = (ref_date.month + month_offset - 1) % 12 + 1
        tf = _third_friday(y, m)
        dte = (tf - ref_date).days
        if dte >= 7:  # skip if less than a week out
            monthlies.append((tf, dte))

    # 1. Standard monthly in the preferred DTE window
    in_window = [(tf, dte) for tf, dte in monthlies if dte_min <= dte <= dte_max]
    if in_window:
        return in_window[0][0]

    # 2. Nearest monthly with >= 21 DTE (slightly outside window is fine)
    viable = [(tf, dte) for tf, dte in monthlies if dte >= 21]
    if viable:
        target_dte = (dte_min + dte_max) // 2
        viable.sort(key=lambda x: abs(x[1] - target_dte))
        return viable[0][0]

    # 3. Any monthly with >= 7 DTE (emergency — very rare)
    if monthlies:
        return monthlies[0][0]

    # 4. Last resort: nearest Friday with >= 21 DTE (should never reach here)
    start = ref_date + timedelta(days=21)
    days_to_friday = (4 - start.weekday()) % 7
    if days_to_friday == 0 and start.weekday() != 4:
        days_to_friday = 7
    return start + timedelta(days=days_to_friday)


def recommend_option(
    ticker: str, direction: str, price: float | None
) -> dict:
    """
    Recommend a single-leg option for Level 2 account.
    Bull -> buy calls (ATM or slightly ITM), Bear -> buy puts.
    30-45 DTE standard monthly expiry, max $200.
    """
    now = datetime.now()
    expiry = find_standard_expiry(now.date(), TARGET_DTE_MIN, TARGET_DTE_MAX)
    expiry_str = expiry.strftime("%Y-%m-%d")

    if price is None:
        return {
            "type": "call" if direction == "bull" else "put",
            "strike": "ATM (need price data)",
            "expiry": expiry_str,
            "estimated_cost": "unknown",
            "note": "Could not fetch current price",
        }

    # Strike selection: Target delta 0.35-0.40 for optimal R:R on 2-5 day momentum trades
    # Research (2025-2026 backtest): delta 0.40 → Sharpe 0.119, WR 45%, best risk-adjusted
    # Approximation: delta ~0.40 ≈ 1-2% OTM for 30-45 DTE
    if price < 100:
        step = 1.0
    else:
        step = 5.0

    # Target ~0.35-0.40 delta: slightly OTM for leverage with reasonable theta
    otm_pct = 0.02  # 2% OTM targets ~0.38-0.42 delta at 30-45 DTE

    if direction == "bull":
        # Slightly OTM call for delta ~0.40
        target_strike = price * (1 + otm_pct)
        strike = round(target_strike / step) * step
        # Estimate cost: ~2.5-3.5% of underlying for slightly OTM 35 DTE
        est_pct = 0.03
        est_cost = round(price * est_pct * 100, 0)  # per contract
    else:
        # Slightly OTM put for delta ~0.40
        target_strike = price * (1 - otm_pct)
        strike = round(target_strike / step) * step
        est_pct = 0.025
        est_cost = round(price * est_pct * 100, 0)

    # Check budget constraint
    affordable = est_cost <= MAX_POSITION_COST
    note = ""
    if not affordable:
        # Try OTM to reduce cost
        if direction == "bull":
            strike = strike + step * 2  # go OTM
            est_cost = round(price * 0.02 * 100, 0)
        else:
            strike = strike - step * 2
            est_cost = round(price * 0.018 * 100, 0)
        if est_cost > MAX_POSITION_COST:
            note = f"WARNING: estimated cost ${est_cost} exceeds $200 budget. Consider smaller position or different ticker."
        else:
            note = f"Moved OTM to fit budget (est ${est_cost})"

    return {
        "type": "call" if direction == "bull" else "put",
        "strike": strike,
        "expiry": expiry_str,
        "estimated_cost": est_cost,
        "affordable": est_cost <= MAX_POSITION_COST,
        "note": note,
    }


# ---------------------------------------------------------------------------
# ETF Top Holdings (static — more reliable than API lookups)
# ---------------------------------------------------------------------------
ETF_TOP_HOLDINGS = {
    'XLE': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'WMB'],
    'XLU': ['NEE', 'SO', 'DUK', 'D', 'SRE', 'AEP', 'EXC', 'XEL', 'ED', 'WEC'],
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM', 'AMD', 'ADBE', 'ACN', 'CSCO', 'ORCL'],
    'XLF': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI', 'BLK'],
    'XLV': ['UNH', 'LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'AMGN', 'PFE'],
    'XLI': ['GE', 'CAT', 'UNP', 'HON', 'RTX', 'DE', 'LMT', 'BA', 'UPS', 'ADP'],
    'XLP': ['PG', 'COST', 'WMT', 'KO', 'PEP', 'PM', 'MDLZ', 'MO', 'CL', 'TGT'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'ABNB'],
    'XLB': ['LIN', 'SHW', 'APD', 'ECL', 'FCX', 'NEM', 'NUE', 'VMC', 'MLM', 'DD'],
    'XLRE': ['PLD', 'AMT', 'EQIX', 'SPG', 'PSA', 'DLR', 'O', 'WELL', 'CCI', 'AVB'],
    'XLC': ['META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'CMCSA', 'T', 'TMUS', 'VZ', 'CHTR'],
    'SMH': ['NVDA', 'TSM', 'AVGO', 'ASML', 'AMD', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'MU'],
    'IBB': ['VRTX', 'GILD', 'AMGN', 'REGN', 'MRNA', 'ILMN', 'SGEN', 'ALNY', 'BMRN', 'BIIB'],
}


def _fetch_earnings_yfinance(symbols: list[str], horizon_days: int) -> set[str]:
    """Check which symbols have earnings within horizon_days using yfinance."""
    reporting = set()
    try:
        import yfinance as yf
    except ImportError:
        print("  [WARN] yfinance not available for earnings check", file=sys.stderr)
        return reporting

    now = datetime.now()
    cutoff = now + timedelta(days=horizon_days)

    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            # earnings_dates returns a DataFrame with index = earnings date
            ed = tk.earnings_dates
            if ed is not None and not ed.empty:
                for dt in ed.index:
                    # dt is a Timestamp (possibly tz-aware)
                    dt_naive = dt.tz_localize(None) if dt.tzinfo else dt
                    if now <= dt_naive <= cutoff:
                        reporting.add(sym)
                        break
        except Exception:
            # Silently skip — we'll miss this one but won't crash
            pass
    return reporting


def earnings_filter(
    signals: list[dict],
    horizon_days: int = 5,
) -> list[dict]:
    """
    Filter signals based on upcoming earnings of ETF components.

    Rules:
      - >= 5 of top-10 holdings reporting within hold window -> SKIP entirely
      - >= 3 of top-10 holdings reporting within hold window -> FLAG as HIGH
        earnings risk and reduce confidence by 20%

    Returns the filtered list (some entries removed, some confidence-adjusted).
    """
    if not signals:
        return signals

    # Collect all unique component symbols we need to check
    tickers_to_check = set()
    etf_tickers_in_signals = set()
    for sig in signals:
        ticker = sig.get("ticker", "")
        if ticker in ETF_TOP_HOLDINGS:
            etf_tickers_in_signals.add(ticker)
            tickers_to_check.update(ETF_TOP_HOLDINGS[ticker])

    if not tickers_to_check:
        return signals  # No ETF signals to filter

    # Fetch earnings for all relevant component stocks in one pass
    print(f"\n  [EARNINGS FILTER] Checking {len(tickers_to_check)} component stocks "
          f"for earnings within {horizon_days} days...")
    reporting_symbols = _fetch_earnings_yfinance(list(tickers_to_check), horizon_days)
    if reporting_symbols:
        print(f"  [EARNINGS FILTER] Components reporting soon: {', '.join(sorted(reporting_symbols))}")
    else:
        print("  [EARNINGS FILTER] No component earnings found in window (or data unavailable)")
        return signals

    # Apply filter per signal
    filtered = []
    for sig in signals:
        ticker = sig.get("ticker", "")
        holdings = ETF_TOP_HOLDINGS.get(ticker)

        if holdings is None:
            # Not an ETF we track — pass through unchanged
            filtered.append(sig)
            continue

        reporting_in_etf = [h for h in holdings if h in reporting_symbols]
        n_reporting = len(reporting_in_etf)

        if n_reporting >= 5:
            # SKIP entirely
            print(f"  [EARNINGS FILTER] SKIPPED {ticker}: {n_reporting}/10 top holdings "
                  f"reporting ({', '.join(reporting_in_etf)}) — heavy_earnings_week")
            continue  # Do not append — signal is dropped

        if n_reporting >= 3:
            # FLAG and reduce confidence
            original_conf = sig["confidence_score"]
            reduced_conf = round(max(0.05, original_conf - 0.20), 2)
            sig = dict(sig)  # shallow copy so we don't mutate original
            sig["confidence_score"] = reduced_conf
            sig["earnings_risk"] = "HIGH"
            sig["earnings_reporting"] = reporting_in_etf
            sig["earnings_note"] = (
                f"{n_reporting}/10 top holdings reporting within {horizon_days}d "
                f"({', '.join(reporting_in_etf)}). Confidence reduced "
                f"{original_conf:.0%} -> {reduced_conf:.0%}."
            )
            print(f"  [EARNINGS FILTER] FLAGGED {ticker}: {n_reporting}/10 reporting "
                  f"({', '.join(reporting_in_etf)}) — confidence {original_conf:.0%} -> {reduced_conf:.0%}")

        filtered.append(sig)

    n_removed = len(signals) - len(filtered)
    if n_removed:
        print(f"  [EARNINGS FILTER] Removed {n_removed} signal(s) due to heavy earnings week")

    return filtered


def enhance_signal_with_rh_tools(signal: dict) -> dict:
    """
    ENHANCEMENT STUB: Add Robinhood MCP tools for technical confluence + liquidity checks.

    When ENABLE_RH_TOOLS = True and phase 2-4 are implemented:
    1. Fetch RSI(14), MACD, Bollinger Bands
    2. Boost confidence if technical indicators align with direction
    3. Fetch option liquidity (OI, bid/ask spread)
    4. Call review_option_order to simulate trade

    Args:
        signal: dict with keys: ticker, direction, confidence_score, recommended_option, etc.

    Returns:
        signal: enhanced with RH data (rsi_14, option_bid, option_ask, option_oi, review_status, etc.)

    See knowledge_base/rh_tools_integration.md for full example.
    """
    if not ENABLE_RH_TOOLS:
        return signal

    # TODO: Phase 2 — Add technical indicators
    # TODO: Phase 3 — Add options liquidity checks
    # TODO: Phase 4 — Add review_option_order simulation

    # Placeholder: return signal unchanged
    return signal


def validate_signal_with_scanner(signal: dict) -> dict:
    """
    ENHANCEMENT STUB: Check if signal ticker appears in high-options-volume scan.

    If enabled, run_scan for high-options-flow and cross-reference with current signals.
    Boost confidence if ticker is in high-vol scanner (confluence).

    See knowledge_base/rh_tools_integration.md section 1.
    """
    if not ENABLE_RH_TOOLS or not RH_SCANS.get("high_options_volume"):
        return signal

    # TODO: Call run_scan to get current high-vol tickers
    # TODO: If signal["ticker"] in results, boost confidence by 10%

    return signal


def run() -> dict:
    """Main aggregation logic. Returns the output dict."""
    # Market open guard — skip on weekends/holidays
    try:
        import sys as _sys
        _sys.path.insert(0, "/home/jupiter/Lvl3Quant/scripts")
        from market_status import get_market_status
        ms = get_market_status()
        if not ms.get("is_trading_day", True):
            print(f"Market closed: {ms.get('reason')}. Skipping aggregation.")
            return {"status": "MARKET_CLOSED", "reason": ms.get("reason")}
    except Exception:
        pass  # Fail-open

    print("=" * 60)
    print("  AGENTIC SIGNAL AGGREGATOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # 1. Load all state files
    # ------------------------------------------------------------------
    print("[1/4] Loading paper engine states...")
    sector_spreads   = load_json(SECTOR_SPREADS_STATE)
    quality_momentum = load_json(QUALITY_MOMENTUM_STATE)
    pead_drift       = load_json(PEAD_DRIFT_STATE)
    vix_state        = load_json(VIX_CALL_SPREAD_STATE)
    sector_etf_mom   = load_json(SECTOR_ETF_MOM_STATE)
    signal_watcher   = load_json(SIGNAL_WATCHER_STATE)

    # V9.1/V9.3/V10 — our BEST strategies (HC #753: must feed agentic)
    sector_v91       = load_json(SECTOR_V91_STATE)
    sector_v93       = load_json(SECTOR_V93_STATE)
    sector_v10       = load_json(SECTOR_V10_STATE)

    # Load ML-enhanced PEAD scores (from pead_ml_live_scorer)
    pead_ml_scores   = load_json(STATE / "pead_ml_scores.json")

    # Validated strategies #8-#12 (HC #773 gap fix)
    rsi_divergence   = load_json(RSI_DIVERGENCE_STATE)
    bond_yield       = load_json(BOND_YIELD_STATE)
    iv_rv_gap        = load_json(IV_RV_GAP_STATE)
    liquidity_signal = load_json(LIQUIDITY_SIGNAL_STATE)
    vol_term_struct  = load_json(VOL_TERM_STRUCT_STATE)
    cross_type_conf  = load_json(CROSS_TYPE_STATE)

    # Parse log files for additional context
    quality_log = parse_log_signals(QUALITY_MOMENTUM_LOG)
    sector_log  = parse_log_signals(SECTOR_ETF_MOM_LOG)
    pead_log    = parse_log_signals(PEAD_DRIFT_LOG)
    vix_log     = parse_log_signals(VIX_CALL_SPREAD_LOG)

    # Event-driven catalyst engines (Session 25)
    sector_mom_spreads = load_json(SECTOR_MOM_SPREADS_STATE)
    gap_fade_spread    = load_json(GAP_FADE_SPREAD_STATE)

    # Previously unwired validated engines (completion gap fix)
    extreme_idio     = load_json(EXTREME_IDIO_STATE)
    volume_surge     = load_json(VOLUME_SURGE_STATE)
    vol_crush        = load_json(VOL_CRUSH_STATE)
    vix_mr           = load_json(VIX_MR_SPREAD_STATE)
    bond_inflow      = load_json(BOND_YIELD_INFLOW_STATE)
    sig_scoring      = load_json(SIGNAL_SCORING_STATE)

    # Newly validated strategies #16-#18 (6/6 adversarial each)
    vix_contango_oversold = load_json(VIX_CONTANGO_OVERSOLD_STATE)
    rank_reversal_b       = load_json(SECTOR_RANK_REVERSAL_B_STATE)
    earnings_outperf_e    = load_json(EARNINGS_OUTPERF_E_STATE)

    # High-growth strategies (momentum growth rotation + flow reversal 2x)
    momentum_growth      = load_json(MOMENTUM_GROWTH_STATE)
    flow_reversal_2x     = load_json(FLOW_REVERSAL_2X_STATE)

    loaded_count = sum(1 for x in [sector_spreads, quality_momentum, pead_drift,
                                    vix_state, sector_etf_mom, signal_watcher,
                                    sector_v91, sector_v93, sector_v10,
                                    rsi_divergence, bond_yield, iv_rv_gap,
                                    liquidity_signal, vol_term_struct, cross_type_conf,
                                    sector_mom_spreads, gap_fade_spread,
                                    extreme_idio, volume_surge, vol_crush,
                                    vix_mr, bond_inflow, sig_scoring,
                                    vix_contango_oversold, rank_reversal_b,
                                    earnings_outperf_e,
                                    momentum_growth, flow_reversal_2x] if x)
    print(f"  Loaded {loaded_count}/28 state files")

    # Extract V10/V9.x + validated strategy #8-#12 open positions as signal sources
    # These are our highest-Sharpe strategies — their picks are high-weight signals
    for state_name, state_data in [("v10_optimal", sector_v10),
                                    ("v93_profit_target", sector_v93),
                                    ("v91_monthly", sector_v91),
                                    ("rsi_divergence", rsi_divergence),
                                    ("bond_yield_signal", bond_yield),
                                    ("iv_rv_gap", iv_rv_gap),
                                    ("liquidity_signal", liquidity_signal),
                                    ("vol_term_structure", vol_term_struct),
                                    ("cross_type_confluence", cross_type_conf),
                                    ("vix_contango_oversold_16", vix_contango_oversold),
                                    ("rank_reversal_b_17", rank_reversal_b),
                                    ("earnings_outperf_e_18", earnings_outperf_e)]:
        if state_data and ("open_positions" in state_data or "positions" in state_data):
            pos_key = "open_positions" if "open_positions" in state_data else "positions"
            n_pos = len(state_data[pos_key])
            equity = state_data.get("equity", 645)
            wr = 0
            total = state_data.get("wins", 0) + state_data.get("losses", 0)
            if total > 0:
                wr = state_data["wins"] / total * 100
            print(f"  {state_name}: {n_pos} positions, equity ${equity:.0f}, WR {wr:.0f}%")

    # ------------------------------------------------------------------
    # 2. Fetch fresh market data
    # ------------------------------------------------------------------
    print("\n[2/4] Fetching fresh market data...")
    market_data = get_market_data()
    vix_level = market_data.get("vix")

    # Fallback VIX from signal watcher
    if vix_level is None and signal_watcher:
        vix_level = signal_watcher.get("vix_level") or signal_watcher.get("vmr", {}).get("vix")

    if vix_level:
        print(f"  VIX: {vix_level:.2f}")
        if market_data.get("vix3m"):
            ts = market_data.get("vix_term_structure", "unknown")
            ratio = market_data.get("vix_term_ratio", 0)
            print(f"  VIX3M: {market_data['vix3m']:.2f} | Term structure: {ts.upper()} (ratio: {ratio})")
            if ts == "backwardation":
                print("  ⚠️  Backwardation — caution on buying options (IV may fall)")
            else:
                print("  ✅ Contango — favorable for buying options")
    else:
        print("  VIX: unavailable")

    if market_data.get("available"):
        n_sectors = len(market_data.get("sectors", {}))
        print(f"  Fetched data for {n_sectors} sector ETFs")
    else:
        print("  Using cached data only (yfinance unavailable)")

    # ------------------------------------------------------------------
    # 3. Build signals and compute confluence
    # ------------------------------------------------------------------
    print("\n[3/4] Computing confluence signals...")
    regime = determine_regime(vix_level, signal_watcher)
    print(f"  Market regime: {regime.upper()}")

    ticker_signals = build_ticker_signals(
        sector_spreads, quality_momentum, sector_etf_mom, pead_drift,
        vix_state, signal_watcher, quality_log, sector_log, market_data,
        sector_v91=sector_v91, sector_v93=sector_v93, sector_v10=sector_v10,
    )

    # --- HC #775: Inject broad market sector screener signals ---
    broad_mkt = load_json(BROAD_MARKET_SIGNALS)
    if broad_mkt and broad_mkt.get("actionable_signals"):
        print(f"  [HC#775] Broad market screener: {len(broad_mkt['actionable_signals'])} actionable signals")
        for bsig in broad_mkt["actionable_signals"]:
            t = bsig.get("ticker", "")
            if not t:
                continue
            if t not in ticker_signals:
                ticker_signals[t] = {
                    "bull_sources": [], "bear_sources": [],
                    "details": {}, "scores": [],
                }
            direction = bsig.get("signal", "NEUTRAL").lower()
            source_name = f"broad_sector_screener({bsig.get('sector', 'unknown')})"
            conf = bsig.get("confidence", 0.5)
            if direction == "bullish":
                ticker_signals[t]["bull_sources"].append(source_name)
                ticker_signals[t]["scores"].append(conf)
            elif direction == "bearish":
                ticker_signals[t]["bear_sources"].append(source_name)
                ticker_signals[t]["scores"].append(-conf)
            ticker_signals[t]["details"]["broad_market_screener"] = {
                "score": bsig.get("total_score", 0),
                "confidence": conf,
                "sector": bsig.get("sector", ""),
                "rsi_5": bsig.get("indicators", {}).get("rsi_5", 0),
                "ret_5d_pct": bsig.get("indicators", {}).get("ret_5d_pct", 0),
            }

    # --- Sub-Sector Rotation: inject deep sub-industry rotation signals ---
    subsector_rot = load_json(SUBSECTOR_ROTATION_STATE)
    subsector_preds = load_json(SUBSECTOR_ROTATION_PREDS)
    if subsector_rot and subsector_rot.get("rotation_pattern"):
        pattern = subsector_rot["rotation_pattern"]
        print(f"  [SubSector] Rotation pattern: {pattern.get('pattern', 'UNKNOWN')} — "
              f"{pattern.get('inflow_count', 0)} inflow, {pattern.get('outflow_count', 0)} outflow")

        # Inject ETF-level signals from sub-sector rotation
        if subsector_rot.get("subsectors"):
            for sub_name, sub_data in subsector_rot["subsectors"].items():
                phase = sub_data.get("rotation_phase", "NEUTRAL")
                score = sub_data.get("rotation_score", 0)
                etf = sub_data.get("etf_proxy", "")

                if not etf or phase == "NEUTRAL":
                    continue

                if etf not in ticker_signals:
                    ticker_signals[etf] = {
                        "bull_sources": [], "bear_sources": [],
                        "details": {}, "scores": [],
                    }

                source_name = f"subsector_rotation({sub_name})"
                if phase in ("INFLOW", "ACCUMULATING"):
                    ticker_signals[etf]["bull_sources"].append(source_name)
                    ticker_signals[etf]["scores"].append(min(0.7, abs(score) * 0.2))
                elif phase in ("OUTFLOW", "DISTRIBUTING"):
                    ticker_signals[etf]["bear_sources"].append(source_name)
                    ticker_signals[etf]["scores"].append(-min(0.7, abs(score) * 0.2))

                ticker_signals[etf]["details"].setdefault("subsector_rotation", []).append({
                    "subsector": sub_name,
                    "phase": phase,
                    "score": score,
                    "rel_str_21d": sub_data.get("metrics", {}).get("rel_str_vs_spy_21d", 0),
                })

    # Inject ML rotation predictions (sub-sector overweight/underweight)
    if subsector_preds and subsector_preds.get("top_overweight"):
        for pred in subsector_preds["top_overweight"]:
            etf = pred.get("etf_proxy", "")
            if not etf:
                continue
            if etf not in ticker_signals:
                ticker_signals[etf] = {
                    "bull_sources": [], "bear_sources": [],
                    "details": {}, "scores": [],
                }
            ticker_signals[etf]["bull_sources"].append(f"subsector_ml_overweight({pred['subsector']})")
            pred_ret = pred.get("predicted_return_pct", 0)
            ticker_signals[etf]["scores"].append(min(0.6, abs(pred_ret) * 0.1))
            ticker_signals[etf]["details"]["subsector_ml_prediction"] = {
                "subsector": pred["subsector"],
                "predicted_return_pct": pred_ret,
                "rank": pred.get("rank", 0),
            }

    if subsector_preds and subsector_preds.get("bottom_underweight"):
        for pred in subsector_preds["bottom_underweight"]:
            etf = pred.get("etf_proxy", "")
            if not etf:
                continue
            if etf not in ticker_signals:
                ticker_signals[etf] = {
                    "bull_sources": [], "bear_sources": [],
                    "details": {}, "scores": [],
                }
            ticker_signals[etf]["bear_sources"].append(f"subsector_ml_underweight({pred['subsector']})")
            pred_ret = pred.get("predicted_return_pct", 0)
            ticker_signals[etf]["scores"].append(-min(0.6, abs(pred_ret) * 0.1))

    # --- Validated Sub-Sector Pair Rotation (5 adversarial-validated pairs) ---
    # Weight 0.40 — lower than core equity (0.60) since this is a new category
    SUBSECTOR_PAIR_WEIGHT = 0.40
    subsector_pair_state = load_json(SUBSECTOR_VALIDATED_PAIRS_STATE)
    if subsector_pair_state and subsector_pair_state.get("signals"):
        pair_signals = subsector_pair_state["signals"]
        n_firing = sum(1 for s in pair_signals if s.get("fires"))
        print(f"  [SubSecPairs] {len(pair_signals)} pairs checked, {n_firing} firing")

        for sig in pair_signals:
            if not sig.get("fires"):
                continue

            lagging_etf = sig.get("lagging_etf", "")
            if not lagging_etf:
                continue

            if lagging_etf not in ticker_signals:
                ticker_signals[lagging_etf] = {
                    "bull_sources": [], "bear_sources": [],
                    "details": {}, "scores": [],
                }

            prob = sig.get("reversal_prob", 0.5)
            pair_name = sig.get("pair", "unknown")
            source = f"subsector_pair_rotation({pair_name})"

            # Long the lagging sub-sector ETF (mean-reversion signal)
            ticker_signals[lagging_etf]["bull_sources"].append(source)
            # Score = weight * probability, capped at 0.70
            score = min(0.70, prob * SUBSECTOR_PAIR_WEIGHT)
            ticker_signals[lagging_etf]["scores"].append(score)

            ticker_signals[lagging_etf]["details"].setdefault("subsector_pair_rotation", []).append({
                "pair": pair_name,
                "reversal_prob": prob,
                "rel_ret_21d": sig.get("rel_ret_21d", 0),
                "horizon": sig.get("horizon", 5),
                "backtest_sharpe": sig.get("backtest_sharpe", 0),
                "lagging_label": sig.get("lagging_label", ""),
            })

    # --- AVO-Evolved Lockbox-Validated Engines ---
    # Sentiment Contrarian AVO (score 8.25, lockbox Sharpe 2.73 @ MAX_CONCURRENT=3)
    AVO_WEIGHT = 0.55  # Higher weight — lockbox validated with strong Sharpe
    sc_avo_state = load_json(SENTIMENT_CONTRARIAN_AVO_STATE)
    if sc_avo_state and sc_avo_state.get("positions"):
        for pos in sc_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("sentiment_contrarian_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Put-Call Contrarian AVO (score 6.76, lockbox Sharpe 3.78, CAGR 26.2%)
    pc_avo_state = load_json(PUT_CALL_CONTRARIAN_STATE)
    if pc_avo_state and pc_avo_state.get("positions"):
        for pos in pc_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("put_call_contrarian_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Vol Compression AVO (score 7.94, lockbox Sharpe 1.28)
    vc_avo_state = load_json(VOL_COMPRESSION_AVO_STATE)
    if vc_avo_state and vc_avo_state.get("positions"):
        for pos in vc_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("vol_compression_avo")
            ticker_signals[tk]["scores"].append(0.40)

    # Macro Regime Rotation AVO (score 8.00, lockbox validated — highest scoring strategy)
    mr_avo_state = load_json(MACRO_REGIME_ROTATION_STATE)
    if mr_avo_state and mr_avo_state.get("positions"):
        for pos in mr_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("macro_regime_rotation_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Gold-Bond Divergence AVO (score 6.80, lockbox Sharpe 3.83)
    gb_avo_state = load_json(GOLD_BOND_DIVERGENCE_STATE)
    if gb_avo_state and gb_avo_state.get("positions"):
        for pos in gb_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("gold_bond_divergence_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Insider Momentum AVO (score 6.50, lockbox validated)
    im_avo_state = load_json(INSIDER_MOMENTUM_STATE)
    if im_avo_state and im_avo_state.get("positions"):
        for pos in im_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("insider_momentum_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Vol Regime Mean-Revert AVO (score 7.09, lockbox Sharpe 1.18)
    vr_avo_state = load_json(VOL_REGIME_MEAN_REVERT_STATE)
    if vr_avo_state and vr_avo_state.get("positions"):
        for pos in vr_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("vol_regime_mean_revert_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Treasury Curve Steepener AVO (score 7.32, lockbox Sharpe 4.09 — 2nd best lockbox ever)
    tc_avo_state = load_json(TREASURY_CURVE_STEEPENER_STATE)
    if tc_avo_state and tc_avo_state.get("positions"):
        for pos in tc_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("treasury_curve_steepener_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Trend Dip Reversion AVO (score 4.49, lockbox Sharpe 0.95)
    td_avo_state = load_json(TREND_DIP_REVERSION_STATE)
    if td_avo_state and td_avo_state.get("positions"):
        for pos in td_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("trend_dip_reversion_avo")
            ticker_signals[tk]["scores"].append(0.40)  # Lower weight — conditional lockbox pass

    # Breadth Momentum Regime AVO (score 5.81, lockbox Sharpe 2.12)
    bm_avo_state = load_json(BREADTH_MOMENTUM_REGIME_STATE)
    if bm_avo_state and bm_avo_state.get("positions"):
        for pos in bm_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("breadth_momentum_regime_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Cross-Asset Macro AVO (score 7.28, lockbox Sharpe 3.61)
    cam_avo_state = load_json(CROSS_ASSET_MACRO_STATE)
    if cam_avo_state and cam_avo_state.get("positions"):
        for pos in cam_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("cross_asset_macro_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Size Rotation AVO (score 4.56, lockbox Sharpe 2.30)
    sr_avo_state = load_json(SIZE_ROTATION_STATE)
    if sr_avo_state and sr_avo_state.get("positions"):
        for pos in sr_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("size_rotation_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Calendar Momentum AVO (score 3.64, lockbox Sharpe 2.58)
    cm_avo_state = load_json(CALENDAR_MOMENTUM_STATE)
    if cm_avo_state and cm_avo_state.get("positions"):
        for pos in cm_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("calendar_momentum_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Credit Spread Momentum AVO (score 9.78, lockbox Sharpe 7.78)
    csm_avo_state = load_json(CREDIT_SPREAD_MOMENTUM_STATE)
    if csm_avo_state and csm_avo_state.get("positions"):
        for pos in csm_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("credit_spread_momentum_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Options Execution AVO (score 531.27, lockbox Sharpe 2.48)
    oea_avo_state = load_json(OPTIONS_EXECUTION_AVO_STATE)
    if oea_avo_state and oea_avo_state.get("positions"):
        for pos in oea_avo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("options_execution_avo")
            ticker_signals[tk]["scores"].append(AVO_WEIGHT)

    # Options Overlay Paper Engine (ATM calls on multi-strategy confluence signals)
    OPTIONS_OVERLAY_WEIGHT = 0.50  # Backtest 34% CAGR, 11x amplification
    oo_state = load_json(OPTIONS_OVERLAY_STATE)
    if oo_state and oo_state.get("positions"):
        for pos in oo_state["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("options_overlay")
            n_src = pos.get("n_sources", 1)
            # Scale weight by confluence (more strategy sources = stronger signal)
            weight = OPTIONS_OVERLAY_WEIGHT * min(1.0 + 0.1 * (n_src - 1), 1.5)
            ticker_signals[tk]["scores"].append(weight)
            ticker_signals[tk]["details"]["options_overlay"] = {
                "strike": pos.get("strike"),
                "delta": pos.get("delta"),
                "contracts": pos.get("contracts"),
                "dte_remaining": pos.get("dte_remaining"),
                "sources": pos.get("sources", []),
            }

    # Unified Portfolio Engine (meta-allocator across all strategies)
    up_state = load_json(UNIFIED_PORTFOLIO_STATE)
    if up_state and up_state.get("recommendations"):
        for rec in up_state["recommendations"]:
            tk = rec.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            direction = rec.get("direction", "bull")
            source = "unified_portfolio_engine"
            if direction == "bear":
                ticker_signals[tk]["bear_sources"].append(source)
                ticker_signals[tk]["scores"].append(-0.35)
            else:
                ticker_signals[tk]["bull_sources"].append(source)
                ticker_signals[tk]["scores"].append(0.35)

    # --- High-Growth: Momentum Growth Rotation ---
    # Top momentum picks generate bullish call signals on the held stocks
    MOMENTUM_GROWTH_WEIGHT = 0.45  # Validated backtest, vol-weighted RS
    if momentum_growth and momentum_growth.get("positions"):
        n_mg = len(momentum_growth["positions"])
        mg_equity = momentum_growth.get("capital", 10000)
        print(f"  [MomentumGrowth] {n_mg} positions, equity ${mg_equity:.0f}")
        for pos in momentum_growth["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            weight = pos.get("weight", 0.2)
            # Higher weight positions get stronger signal score
            score = min(0.70, MOMENTUM_GROWTH_WEIGHT * (1 + weight))
            ticker_signals[tk]["bull_sources"].append("momentum_growth_rotation")
            ticker_signals[tk]["scores"].append(score)
            ticker_signals[tk]["details"]["momentum_growth"] = {
                "entry_price": pos.get("entry_price", 0),
                "entry_date": pos.get("entry_date", ""),
                "weight": weight,
                "strategy": "RS vol-weighted top-5 rotation",
            }

    # --- High-Growth: Flow Reversal 2x (volume-spike dip buying) ---
    # Active positions = dip-buy signals on ETFs with volume spikes
    FLOW_REVERSAL_WEIGHT = 0.50  # Lockbox Sharpe 3.37, strong signal
    if flow_reversal_2x and flow_reversal_2x.get("positions"):
        n_fr = len(flow_reversal_2x["positions"])
        fr_equity = flow_reversal_2x.get("capital", 10000)
        print(f"  [FlowReversal2x] {n_fr} positions, equity ${fr_equity:.0f}")
        for pos in flow_reversal_2x["positions"]:
            tk = pos.get("ticker", "")
            if not tk:
                continue
            if tk not in ticker_signals:
                ticker_signals[tk] = {"bull_sources": [], "bear_sources": [], "details": {}, "scores": []}
            ticker_signals[tk]["bull_sources"].append("flow_reversal_2x")
            ticker_signals[tk]["scores"].append(FLOW_REVERSAL_WEIGHT)
            ticker_signals[tk]["details"]["flow_reversal_2x"] = {
                "entry_price": pos.get("entry_price", 0),
                "entry_date": pos.get("entry_date", ""),
                "volume_zscore": pos.get("volume_zscore", 0),
                "dip_pct": pos.get("dip_pct", 0),
                "strategy": "Volume-spike dip reversal (2x leveraged)",
                "max_hold_days": 3,
            }

    # Compute confidence and filter
    output_signals = []
    all_candidates = []

    for ticker, sig in ticker_signals.items():
        direction, confidence = compute_confidence(sig["bull_sources"], sig["bear_sources"])
        if direction == "neutral":
            continue

        confirming = sig["bull_sources"] if direction == "bull" else sig["bear_sources"]
        conflicting = sig["bear_sources"] if direction == "bull" else sig["bull_sources"]

        # ML-informed adjustments (Session 83, 97-trade analysis)
        all_src = set(sig["bull_sources"] + sig["bear_sources"])
        ml_warnings = sig.get("warnings", [])
        # Toxic combo penalty: -5% per toxic combo detected
        TOXIC_COMBOS_CHECK = [
            {"cta_trend", "strong_momentum"},
            {"cta_trend", "mom_accelerating"},
            # {"pead_drift", "vol_crush"},  # vol_crush disabled (adversarial fail)
        ]
        n_toxic = sum(1 for combo in TOXIC_COMBOS_CHECK if combo.issubset(all_src))
        if n_toxic:
            confidence = round(max(0.05, confidence * (1 - 0.05 * n_toxic)), 2)
            ml_warnings.append(f"{n_toxic} toxic combo(s) detected — confidence reduced")
        # Ticker quality penalty: XLK/XLV/XLF historically 0-29% WR
        ML_WEAK_TICKERS = {"XLK", "XLV", "XLF"}
        if ticker in ML_WEAK_TICKERS:
            confidence = round(max(0.05, confidence * 0.90), 2)  # 10% penalty
            ml_warnings.append(f"{ticker} historically weak in our system — confidence reduced 10%")

        # Meta-analysis adjustments (Session 84, regime-strategy analysis)
        # 1. Bearish direction penalty: bear plays only 27% WR (3W/8L) in current regime
        if direction == "bear":
            confidence = round(max(0.05, confidence * 0.80), 2)  # 20% penalty on bear plays
            ml_warnings.append("Bear direction historically 27% WR — confidence reduced 20%")

        # 2. Strategy source reliability adjustments from meta-analysis
        META_PENALIZED_SOURCES = {"pead_drift", "pead_ml", "momentum_options"}
        META_BOOSTED_SOURCES = {"v93_profit_target", "v93_daily"}
        penalized_present = META_PENALIZED_SOURCES.intersection(all_src)
        boosted_present = META_BOOSTED_SOURCES.intersection(all_src)
        if penalized_present and not boosted_present:
            confidence = round(max(0.05, confidence * 0.90), 2)  # 10% penalty if only weak sources
            ml_warnings.append(f"Weak sources present ({', '.join(penalized_present)}) — confidence reduced 10%")

        # 3. Volume divergence confluence boost (Session 84)
        # When volume_surge signal confirms our direction, small confidence boost
        # (accumulation sub-signal: Sharpe 0.815, 56.7% WR — not standalone-worthy but useful as confluence)
        if "volume_surge" in all_src and direction == "bull" and len(confirming) >= 3:
            confidence = round(min(0.95, confidence * 1.03), 2)  # 3% boost when volume confirms
            ml_warnings.append("Volume divergence confirms bull thesis — small confidence boost")

        # 4. Signal freshness boost (Session 84+88, sequence pattern mining + rigorous backtest)
        # Burst signals outperform persistent but ONLY on specific tickers (Session 88 backtest):
        #   BURST_POSITIVE: XLE (Sharpe 1.84), XLU (2.63), XLK (2.20), XLI (1.52)
        #   BURST_NEGATIVE: XLV (-1.90), XLP (-2.09), XLY (-2.32), XLC (-1.30), XLB (-2.02)
        #   BURST_NEUTRAL: XLF (0.33), XLRE (-0.17)
        # The burst boost should be ticker-aware — boost on proven tickers, no boost on harmful ones.
        BURST_POSITIVE_TICKERS = {"XLE", "XLU", "XLK", "XLI"}
        BURST_NEGATIVE_TICKERS = {"XLV", "XLP", "XLY", "XLC", "XLB"}
        try:
            prev_signals_path = OUTPUT_FILE.parent / "agentic_signals_prev.json"
            if prev_signals_path.exists():
                import json as _json
                with open(prev_signals_path) as _f:
                    prev = _json.load(_f)
                prev_tickers = {s.get("ticker") for s in prev.get("signals", [])}
                if ticker not in prev_tickers and len(confirming) >= 3:
                    if ticker in BURST_POSITIVE_TICKERS and len(confirming) >= 5:
                        # Strong burst on proven ticker: 15% boost
                        confidence = round(min(0.95, confidence * 1.15), 2)
                        ml_warnings.append(f"STRONG BURST on {ticker} (proven burst ticker, 5+ sources) — confidence boosted 15%")
                    elif ticker in BURST_POSITIVE_TICKERS:
                        # Regular burst on proven ticker: 8% boost
                        confidence = round(min(0.95, confidence * 1.08), 2)
                        ml_warnings.append(f"BURST on {ticker} (proven burst ticker) — confidence boosted 8%")
                    elif ticker not in BURST_NEGATIVE_TICKERS:
                        # Neutral ticker burst: small boost
                        confidence = round(min(0.95, confidence * 1.05), 2)
                        ml_warnings.append("FRESH signal (new today, 3+ sources) — confidence boosted 5%")
                    # No boost for BURST_NEGATIVE_TICKERS — burst pattern loses money on them
        except Exception:
            pass  # Don't break aggregator if prev file missing/corrupt

        # Get price from market data
        price = None
        if market_data.get("available") and ticker in market_data.get("sectors", {}):
            price = market_data["sectors"][ticker]["price"]

        # Option recommendation
        option_rec = recommend_option(ticker, direction, price)

        # Build reason string
        md = sig["details"].get("market_data", {})
        reason_parts = []
        if md:
            reason_parts.append(f"21d mom {md.get('mom_21d', 0):+.1f}%")
            reason_parts.append(f"RSI {md.get('rsi', 0):.0f}")
            reason_parts.append(f"rel str {md.get('rel_str_vs_spy', 0):+.1f}%")
        if sig["details"].get("sector_spreads"):
            ss = sig["details"]["sector_spreads"]
            reason_parts.append(f"LGBM {ss['lgbm_score']:.3f} ({ss['n_signals']} sigs)")
        reason = "; ".join(reason_parts) if reason_parts else "Multiple source agreement"

        entry = {
            "ticker": ticker,
            "direction": direction,
            "confidence_score": confidence,
            "confirming_sources": confirming,
            "conflicting_sources": conflicting,
            "n_confirming": len(confirming),
            "n_conflicting": len(conflicting),
            "recommended_option": option_rec["type"],
            "recommended_strike": option_rec["strike"],
            "recommended_expiry": option_rec["expiry"],
            "estimated_cost": option_rec["estimated_cost"],
            "affordable": option_rec.get("affordable", True),
            "reason": reason,
        }
        if option_rec.get("note"):
            entry["note"] = option_rec["note"]
        if ml_warnings:
            entry["ml_warnings"] = ml_warnings

        # KB #280/#281: Single-leg monthly holds FAIL. Short-term momentum WORKS.
        # Variant F (trailing stop) is champion: Sharpe 1.28, perm p=0.000.
        entry["exit_guidance"] = {
            "take_profit_pct": 0.30,  # Exit when option is up 30%
            "stop_loss_pct": 0.25,    # Exit when option is down 25%
            "max_hold_days": 5,       # Never hold more than 5 trading days
            "trailing_stop": 0.50,    # Exit if gives back 50% of peak unrealized
            "strategy": "momentum_burst_variant_F",
            "avg_hold_days": 2.4,
            "warning": "SINGLE-LEG: Do NOT hold to expiry. Exit within days, not weeks."
        }

        all_candidates.append(entry)

        # Only include high-confidence signals
        if confidence >= MIN_CONFIDENCE:
            output_signals.append(entry)

        # ENHANCEMENT: Apply RH tools enhancements (when enabled)
        # This is where Phase 2-4 integration hooks in
        if ENABLE_RH_TOOLS and confidence >= MIN_CONFIDENCE:
            entry = enhance_signal_with_rh_tools(entry)
            entry = validate_signal_with_scanner(entry)

    # Sort by confidence descending
    output_signals.sort(key=lambda x: x["confidence_score"], reverse=True)
    all_candidates.sort(key=lambda x: x["confidence_score"], reverse=True)

    # ------------------------------------------------------------------
    # Earnings filter — skip or flag signals with heavy component earnings
    # ------------------------------------------------------------------
    hold_days = 5  # default from exit_guidance.max_hold_days
    if output_signals:
        hold_days = output_signals[0].get("exit_guidance", {}).get("max_hold_days", 5)

    pre_filter_count = len(output_signals)
    output_signals = earnings_filter(output_signals, horizon_days=hold_days)

    # ------------------------------------------------------------------
    # IV Rank enrichment — SECTOR-RELATIVE IV assessment (upgraded 2026-08-06)
    # ------------------------------------------------------------------
    # Load sector baselines for sector-relative IV penalties
    sector_baselines = None
    sector_baselines_path = BASE / "config" / "sector_iv_baselines.json"
    if sector_baselines_path.exists():
        sector_baselines = load_json(sector_baselines_path)

    iv_rank_data = load_json(IV_RANK_DATA)
    if iv_rank_data and (iv_rank_data.get("iv_data") or iv_rank_data.get("tickers")):
        iv_tickers = iv_rank_data.get("iv_data") or iv_rank_data.get("tickers") or {}
        for sig in output_signals:
            t = sig["ticker"]
            if t in iv_tickers:
                iv_info = iv_tickers[t]
                iv_rank = iv_info.get("iv_rank")
                iv_class = iv_info.get("classification", "NORMAL")
                iv_trend = iv_info.get("iv_trend", "unknown")
                sig["iv_rank"] = iv_rank
                sig["iv_classification"] = iv_class
                sig["iv_trend"] = iv_trend

                # ── Sector-relative IV assessment ──
                # Determine the sector-specific expensive/cheap thresholds
                sector = iv_info.get("sector")
                sector_name = iv_info.get("sector_name", "Unknown")
                sector_expensive_rank = iv_info.get("sector_expensive_rank")
                sector_cheap_rank = iv_info.get("sector_cheap_rank")

                # Fall back to baselines config if IV tracker didn't embed sector info
                if sector_expensive_rank is None and sector_baselines:
                    ticker_to_sector = sector_baselines.get("ticker_to_sector", {})
                    sector = ticker_to_sector.get(t)
                    if sector:
                        sector_cfg = sector_baselines.get("sectors", {}).get(sector, {})
                        sector_expensive_rank = sector_cfg.get("expensive_rank", 70)
                        sector_cheap_rank = sector_cfg.get("cheap_rank", 25)
                        sector_name = sector_cfg.get("name", sector)

                # Default flat thresholds if no sector data available
                if sector_expensive_rank is None:
                    sector_expensive_rank = 70
                if sector_cheap_rank is None:
                    sector_cheap_rank = 25

                sig["iv_sector"] = sector
                sig["iv_sector_name"] = sector_name
                sig["iv_sector_expensive_threshold"] = sector_expensive_rank

                # ── Sector-relative confidence adjustment ──
                if sig.get("recommended_option") in ("call", "put") and iv_rank is not None:
                    if iv_rank < sector_cheap_rank:
                        # CHEAP relative to sector — boost confidence slightly
                        boost = 0.05
                        if iv_trend == "falling":
                            boost += 0.03  # IV still falling = even better for buying
                        sig["confidence_score"] = round(min(0.95, sig["confidence_score"] + boost), 2)
                        sig["iv_timing_note"] = (
                            f"FAVORABLE — IV rank {iv_rank:.0f}% is cheap for {sector_name} "
                            f"(sector threshold: <{sector_cheap_rank}%). IV {iv_trend}."
                        )

                    elif iv_rank > sector_expensive_rank:
                        # EXPENSIVE relative to sector — graduated penalty
                        # Penalty scales: 5% at threshold, up to 25% at IV rank 100%
                        penalty_schedule = {}
                        if sector_baselines:
                            penalty_schedule = sector_baselines.get("penalty_schedule", {})

                        base_penalty = penalty_schedule.get("base_penalty_at_threshold", 0.05)
                        max_penalty = penalty_schedule.get("max_penalty_at_100_rank", 0.25)
                        trend_adj = penalty_schedule.get("trend_adjustments", {})

                        # Linear interpolation: penalty grows from base at threshold to max at 100%
                        overshoot_pct = (iv_rank - sector_expensive_rank) / (100 - sector_expensive_rank)
                        overshoot_pct = min(1.0, max(0.0, overshoot_pct))
                        penalty = base_penalty + (max_penalty - base_penalty) * overshoot_pct

                        # IV trend adjustment: rising IV = extra penalty, falling = reduced
                        trend_delta = trend_adj.get(iv_trend, 0.0)
                        penalty = max(0.02, penalty + trend_delta)

                        sig["confidence_score"] = round(max(0.05, sig["confidence_score"] * (1 - penalty)), 2)
                        sig["iv_timing_note"] = (
                            f"UNFAVORABLE — IV rank {iv_rank:.0f}% is elevated for {sector_name} "
                            f"(sector threshold: >{sector_expensive_rank}%). "
                            f"Penalty: -{penalty:.0%}. IV {iv_trend}."
                        )
                        sig["iv_penalty_applied"] = round(penalty, 3)

                    else:
                        # Normal range for this sector — no penalty
                        # But add trend context
                        trend_note = ""
                        if iv_trend == "rising":
                            trend_note = " IV rising — watch for expansion."
                        elif iv_trend == "falling":
                            trend_note = " IV falling — favorable direction."
                        sig["iv_timing_note"] = (
                            f"NEUTRAL — IV rank {iv_rank:.0f}% is normal for {sector_name} "
                            f"(sector range: {sector_cheap_rank}-{sector_expensive_rank}%).{trend_note}"
                        )

        n_enriched = sum(1 for s in output_signals if "iv_rank" in s)
        n_penalized = sum(1 for s in output_signals if s.get("iv_penalty_applied"))
        n_boosted = sum(1 for s in output_signals if s.get("iv_timing_note", "").startswith("FAVORABLE"))
        print(f"  [IV RANK] Enriched {n_enriched}/{len(output_signals)} signals with sector-relative IV data")
        if n_penalized:
            print(f"  [IV RANK] {n_penalized} signal(s) penalized for sector-elevated IV")
        if n_boosted:
            print(f"  [IV RANK] {n_boosted} signal(s) boosted for sector-cheap IV")
    all_candidates = earnings_filter(all_candidates, horizon_days=hold_days)

    if len(output_signals) < pre_filter_count:
        print(f"  Earnings filter removed {pre_filter_count - len(output_signals)} "
              f"signal(s) from actionable list")

    # ------------------------------------------------------------------
    # Greeks optimization enrichment — analyze theta/delta/vega/gamma
    # ------------------------------------------------------------------
    if GREEKS_OPTIMIZER_AVAILABLE:
        print("\n  [GREEKS] Enriching signals with Greeks optimization...")
        n_greeks = 0
        for sig in output_signals:
            try:
                enrich_signal_with_greeks(sig)
                gs = sig.get("greeks_analysis", {}).get("greeks_score", 0)
                adj = sig.get("greeks_analysis", {}).get("adjustment", "none")
                if gs > 0:
                    n_greeks += 1
                    ticker = sig.get("ticker", "?")
                    print(f"    {ticker}: Greeks score {gs:.0f}/100, {adj}")
            except Exception as e:
                print(f"    [WARN] Greeks analysis failed for {sig.get('ticker', '?')}: {e}",
                      file=sys.stderr)
        print(f"  [GREEKS] Enriched {n_greeks}/{len(output_signals)} signals")
    else:
        print("\n  [GREEKS] Greeks optimizer not available (import failed)")

    # ------------------------------------------------------------------
    # Entry timing score enrichment — macro + technical + market structure
    # ------------------------------------------------------------------
    if TIMING_SCORE_AVAILABLE:
        print("\n  [TIMING] Computing entry timing scores...")
        n_timed = 0
        for sig in output_signals:
            try:
                ticker = sig.get("ticker", "")
                direction = sig.get("recommended_option", "call")
                strike = sig.get("recommended_strike")
                ts_result = compute_timing_score(ticker, direction, strike)
                sig["timing_score"] = ts_result["total_score"]
                sig["timing_recommendation"] = ts_result["recommendation"]
                sig["timing_action"] = ts_result["action"]
                sig["timing_details"] = {
                    "macro": ts_result["macro"]["score"],
                    "technical": ts_result["technical"]["score"],
                    "structure": ts_result["market_structure"]["score"],
                    "strongest": ts_result["strongest_layer"],
                    "weakest": ts_result["weakest_layer"],
                }
                n_timed += 1
                print(f"    {ticker}: Timing {ts_result['total_score']:.0f}/100"
                      f" → {ts_result['recommendation']}")
            except Exception as e:
                print(f"    [WARN] Timing score failed for {sig.get('ticker', '?')}: {e}",
                      file=sys.stderr)
        print(f"  [TIMING] Scored {n_timed}/{len(output_signals)} signals")
    else:
        print("\n  [TIMING] Timing score not available (import failed)")

    # ------------------------------------------------------------------
    # Macro confluence filters — credit spread velocity + yield curve
    # ------------------------------------------------------------------
    credit_stress = market_data.get("credit_stress", "stable")
    curve_stress = market_data.get("curve_stress", "stable")
    csv_3d = market_data.get("credit_spread_velocity_3d", 0)
    curve_5d = market_data.get("yield_curve_5d_change", 0)

    CREDIT_SENSITIVE = {"XLF", "XLRE", "XLY"}
    n_macro_adj = 0

    for sig in output_signals:
        ticker = sig.get("ticker", "")
        direction = sig.get("direction", "")
        adjustments = []

        # Credit spread widening → penalize bullish financials/RE/consumer disc
        if credit_stress == "widening" and ticker in CREDIT_SENSITIVE and direction == "bull":
            penalty = 0.05  # 5% confidence penalty
            sig["confidence_score"] = round(max(0.05, sig["confidence_score"] * (1 - penalty)), 2)
            adjustments.append(f"Credit widening ({csv_3d:+.2f}% 3d) — {ticker} bull penalized -5%")
            n_macro_adj += 1

        # Yield curve steepening fast → penalize ALL bullish signals mildly
        if curve_stress == "steepening_fast" and direction == "bull":
            penalty = 0.03  # 3% confidence penalty (market-level, lighter)
            sig["confidence_score"] = round(max(0.05, sig["confidence_score"] * (1 - penalty)), 2)
            adjustments.append(f"Curve steepening fast ({curve_5d:+.3f} 5d) — bull caution -3%")
            n_macro_adj += 1

        if adjustments:
            sig["macro_confluence_notes"] = adjustments

    if n_macro_adj:
        print(f"\n  [MACRO] {n_macro_adj} signal(s) adjusted by credit/curve confluence filters")
    if credit_stress != "stable":
        print(f"  [MACRO] Credit spread: {credit_stress} (3d velocity: {csv_3d:+.2f}%)")
    if curve_stress != "stable":
        print(f"  [MACRO] Yield curve: {curve_stress} (5d change: {curve_5d:+.3f})")

    # Re-sort after confidence adjustments from earnings filter + Greeks + macro
    output_signals.sort(key=lambda x: x["confidence_score"], reverse=True)
    all_candidates.sort(key=lambda x: x["confidence_score"], reverse=True)

    # Limit to affordable + max concurrent
    actionable = [s for s in output_signals if s.get("affordable", True)][:MAX_CONCURRENT]

    # ------------------------------------------------------------------
    # 4. Build output
    # ------------------------------------------------------------------
    output = {
        "timestamp": datetime.now().isoformat(),
        "account_equity": ACCOUNT_EQUITY,
        "market_regime": regime,
        "vix_level": round(vix_level, 2) if vix_level else None,
        "market_context": {
            "vix": round(vix_level, 2) if vix_level else None,
            "vix3m": round(market_data.get("vix3m", 0), 2) if market_data.get("vix3m") else None,
            "vix_term_structure": market_data.get("vix_term_structure", "unknown"),
            "vix_term_ratio": market_data.get("vix_term_ratio"),
            "iv_environment": "cheap" if (vix_level and vix_level < 15) else ("expensive" if (vix_level and vix_level > 22) else "normal"),
            "options_timing_note": (
                "Contango — favorable for buying options (IV stable/rising)"
                if market_data.get("vix_term_structure") == "contango"
                else "Backwardation — caution buying options (IV may fall, theta/vega headwind)"
                if market_data.get("vix_term_structure") == "backwardation"
                else "VIX term structure unavailable"
            ),
            "credit_spread_velocity_3d": market_data.get("credit_spread_velocity_3d"),
            "credit_stress": market_data.get("credit_stress", "stable"),
            "yield_curve_3m10s": market_data.get("yield_curve_3m10s"),
            "yield_curve_5d_change": market_data.get("yield_curve_5d_change"),
            "curve_stress": market_data.get("curve_stress", "stable"),
        },
        "signals": output_signals,
        "actionable_signals": actionable,
        "earnings_filtered": pre_filter_count - len(output_signals),
        "below_threshold": [c for c in all_candidates if c["confidence_score"] < MIN_CONFIDENCE],
        "engines_status": {
            "sector_spreads": "loaded" if sector_spreads else "missing",
            "quality_momentum": "loaded" if quality_momentum else "missing",
            "pead_drift": "loaded" if pead_drift else "missing",
            "vix_call_spread": "loaded" if vix_state else "missing",
            "sector_etf_momentum": "loaded" if sector_etf_mom else "missing",
            "signal_watcher": "loaded" if signal_watcher else "missing",
            "sector_v91": "loaded" if sector_v91 else "missing",
            "sector_v93": "loaded" if sector_v93 else "missing",
            "sector_v10_optimal": "loaded" if sector_v10 else "missing",
            "sector_mom_spreads": "loaded" if sector_mom_spreads else "missing",
            "gap_fade_spread": "loaded" if gap_fade_spread else "missing",
            "subsector_rotation": "loaded" if subsector_rot else "missing",
            "subsector_ml_predictions": "loaded" if subsector_preds else "missing",
            "subsector_validated_pairs": "loaded" if subsector_pair_state else "missing",
            "extreme_idio": "loaded" if extreme_idio else "missing",
            "volume_surge": "loaded" if volume_surge else "missing",
            "vol_crush": "loaded" if vol_crush else "missing",
            "vix_mr_spread": "loaded" if vix_mr else "missing",
            "bond_yield_inflow": "loaded" if bond_inflow else "missing",
            "vix_contango_oversold_16": "loaded" if vix_contango_oversold else "missing",
            "rank_reversal_b_17": "loaded" if rank_reversal_b else "missing",
            "earnings_outperf_e_18": "loaded" if earnings_outperf_e else "missing",
            "momentum_growth_rotation": "loaded" if momentum_growth else "missing",
            "flow_reversal_2x": "loaded" if flow_reversal_2x else "missing",
            "macro_regime_rotation_avo": "loaded" if mr_avo_state else "missing",
            "gold_bond_divergence_avo": "loaded" if gb_avo_state else "missing",
            "insider_momentum_avo": "loaded" if im_avo_state else "missing",
            "vol_regime_mean_revert_avo": "loaded" if vr_avo_state else "missing",
            "treasury_curve_steepener_avo": "loaded" if tc_avo_state else "missing",
            "trend_dip_reversion_avo": "loaded" if td_avo_state else "missing",
            "breadth_momentum_regime_avo": "loaded" if bm_avo_state else "missing",
            "cross_asset_macro_avo": "loaded" if cam_avo_state else "missing",
        },
    }

    # Save previous signals for freshness detection (Session 84)
    try:
        prev_path = OUTPUT_FILE.parent / "agentic_signals_prev.json"
        if OUTPUT_FILE.exists():
            import shutil
            shutil.copy2(OUTPUT_FILE, prev_path)
    except Exception:
        pass  # Non-critical

    # Write output
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Output written to {OUTPUT_FILE.name}")
    except Exception as e:
        print(f"\n  [ERROR] Could not write output: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Plain-English Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SIGNAL DASHBOARD SUMMARY")
    print("=" * 60)
    print(f"\n  Account: ${ACCOUNT_EQUITY:.0f} | Level 2 (single-leg only)")
    print(f"  VIX: {vix_level:.2f}" if vix_level else "  VIX: unavailable")
    print(f"  Regime: {regime.upper()}")
    print(f"  Engines online: {loaded_count}/6")

    if actionable:
        print(f"\n  HIGH-CONFIDENCE ACTIONABLE SIGNALS ({len(actionable)}):")
        print("  " + "-" * 56)
        for s in actionable:
            arrow = "BULL" if s["direction"] == "bull" else "BEAR"
            print(f"\n  {s['ticker']} -> {arrow} (confidence: {s['confidence_score']:.0%})")
            print(f"    Confirming: {', '.join(s['confirming_sources'])}")
            if s["conflicting_sources"]:
                print(f"    Conflicting: {', '.join(s['conflicting_sources'])}")
            print(f"    Recommendation: Buy {s['recommended_option'].upper()} "
                  f"@ ${s['recommended_strike']} exp {s['recommended_expiry']}")
            cost = s["estimated_cost"]
            if isinstance(cost, (int, float)):
                print(f"    Estimated cost: ${cost:.0f}/contract")
            print(f"    Reason: {s['reason']}")
            if s.get("earnings_risk"):
                print(f"    ⚠️  EARNINGS RISK: {s['earnings_note']}")
            if s.get("note"):
                print(f"    Note: {s['note']}")
            print(f"    ⚡ EXIT PLAN: +30% take profit, -25% stop loss, 5-day max hold")
    else:
        print("\n  NO HIGH-CONFIDENCE SIGNALS (>= 60% confidence)")
        print("  All signals below threshold — no action recommended.")

    if actionable:
        print("\n  ⚠️  SINGLE-LEG WARNING (KB #280): Do NOT hold to expiry.")
        print("     Sector rotation options lose ~65-94% if held monthly.")
        print("     Exit within 3-5 days. Treat as momentum trades only.")

    if output["below_threshold"]:
        print(f"\n  BELOW-THRESHOLD SIGNALS ({len(output['below_threshold'])}):")
        for s in output["below_threshold"]:
            arrow = "BULL" if s["direction"] == "bull" else "BEAR"
            print(f"    {s['ticker']}: {arrow} ({s['confidence_score']:.0%}) "
                  f"- {s['n_confirming']} confirming, {s['n_conflicting']} conflicting")

    # Budget check
    if actionable:
        total_cost = sum(
            s["estimated_cost"] for s in actionable
            if isinstance(s["estimated_cost"], (int, float))
        )
        print(f"\n  BUDGET CHECK: ${total_cost:.0f} needed / ${ACCOUNT_EQUITY:.0f} available "
              f"({'OK' if total_cost <= ACCOUNT_EQUITY else 'OVER BUDGET'})")

    print("\n" + "=" * 60)
    return output


if __name__ == "__main__":
    run()
