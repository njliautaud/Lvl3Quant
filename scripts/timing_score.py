#!/usr/bin/env python3
"""
Entry Timing Score System
=========================
Computes a 0-100 timing score for options entries by combining:
  - Macro Layer (30%): VIX level/trend, term structure, treasury yields, SPY regime
  - Technical Layer (40%): RSI, MACD, Bollinger Bands, 5-day momentum
  - Market Structure Layer (30%): Sector rotation, relative strength, volume, OI

Timing matters enormously for options because theta decays every day you hold
a position with unfavorable conditions. Direction alone is not enough.

Scoring thresholds:
  80-100: "Enter immediately at open"
  60-79:  "Enter but use limit order, be patient"
  40-59:  "Defer to afternoon check, conditions mixed"
  0-39:   "Skip today, timing unfavorable despite direction"

Usage:
  python3 timing_score.py                          # score all signals from agentic_signals.json
  python3 timing_score.py --ticker XLF --dir call  # score a single trade
  python3 timing_score.py --targets XLF:call:58,XLE:call:60,XLU:put:44

Output:
  /home/jupiter/Lvl3Quant/state/timing_score_results.json
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    from price_action_structure import score_price_structure
    PRICE_STRUCTURE_AVAILABLE = True
except ImportError:
    PRICE_STRUCTURE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/home/jupiter/Lvl3Quant")
STATE = BASE / "state"
OUTPUT_FILE = STATE / "timing_score_results.json"
AGENTIC_SIGNALS = STATE / "agentic_signals.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TimingScore] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TimingScore")

# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

def fetch_ticker_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    if not YF_AVAILABLE:
        log.warning("yfinance not available, returning empty DataFrame")
        return pd.DataFrame()
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df.empty:
            log.warning(f"No data returned for {ticker}")
        return df
    except Exception as e:
        log.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def fetch_vix_data() -> dict:
    """Fetch VIX and VIX3M for term structure analysis."""
    result = {"vix": None, "vix3m": None, "vix_5d_ago": None, "term_structure": "unknown"}

    # Try to read from existing agentic_signals first
    try:
        with open(AGENTIC_SIGNALS) as f:
            sig = json.load(f)
        mc = sig.get("market_context", {})
        result["vix"] = mc.get("vix")
        result["vix3m"] = mc.get("vix3m")
        result["term_structure"] = mc.get("vix_term_structure", "unknown")
    except Exception:
        pass

    # Fetch VIX history for trend
    vix_df = fetch_ticker_data("^VIX", period="1mo")
    if not vix_df.empty and len(vix_df) >= 5:
        result["vix"] = result["vix"] or float(vix_df["Close"].iloc[-1])
        result["vix_5d_ago"] = float(vix_df["Close"].iloc[-5])
        result["vix_sma10"] = float(vix_df["Close"].tail(10).mean())

    # Fetch VIX3M for term structure
    if result["vix3m"] is None:
        vix3m_df = fetch_ticker_data("^VIX3M", period="5d")
        if not vix3m_df.empty:
            result["vix3m"] = float(vix3m_df["Close"].iloc[-1])

    if result["vix"] and result["vix3m"]:
        result["term_structure"] = "contango" if result["vix"] < result["vix3m"] else "backwardation"
        result["term_ratio"] = result["vix"] / result["vix3m"]

    return result


def fetch_treasury_data() -> dict:
    """Fetch 10Y and 2Y treasury yield for trend + yield curve analysis."""
    result = {
        "yield_10y": None, "yield_2y": None,
        "yield_5d_change": None, "yield_trend": "flat",
        "curve_2s10s": None, "curve_regime": "unknown"
    }
    tnx_df = fetch_ticker_data("^TNX", period="1mo")
    if not tnx_df.empty and len(tnx_df) >= 5:
        result["yield_10y"] = float(tnx_df["Close"].iloc[-1])
        result["yield_5d_change"] = float(tnx_df["Close"].iloc[-1] - tnx_df["Close"].iloc[-5])
        if result["yield_5d_change"] > 0.05:
            result["yield_trend"] = "rising"
        elif result["yield_5d_change"] < -0.05:
            result["yield_trend"] = "falling"

    # Fetch 2Y yield for yield curve slope (2s10s spread)
    two_y_df = fetch_ticker_data("^IRX", period="1mo")  # 3-month T-bill as proxy
    if not two_y_df.empty and result["yield_10y"] is not None:
        result["yield_2y"] = float(two_y_df["Close"].iloc[-1])
        result["curve_2s10s"] = result["yield_10y"] - result["yield_2y"]
        if result["curve_2s10s"] > 0.5:
            result["curve_regime"] = "steep"       # Normal, healthy
        elif result["curve_2s10s"] > 0:
            result["curve_regime"] = "flat_normal"  # Flat but positive
        elif result["curve_2s10s"] > -0.2:
            result["curve_regime"] = "flat_inverted" # Slightly inverted
        else:
            result["curve_regime"] = "inverted"     # Fully inverted
    return result


def fetch_spy_regime() -> dict:
    """SPY trend regime: above/below 20-SMA and 50-SMA."""
    result = {"price": None, "sma20": None, "sma50": None, "regime": "unknown"}
    spy_df = fetch_ticker_data("SPY", period="3mo")
    if spy_df.empty or len(spy_df) < 50:
        return result
    result["price"] = float(spy_df["Close"].iloc[-1])
    result["sma20"] = float(spy_df["Close"].tail(20).mean())
    result["sma50"] = float(spy_df["Close"].tail(50).mean())

    above_20 = result["price"] > result["sma20"]
    above_50 = result["price"] > result["sma50"]

    if above_20 and above_50:
        result["regime"] = "bull"
    elif above_50 and not above_20:
        result["regime"] = "pullback"
    elif not above_50 and above_20:
        result["regime"] = "recovery"
    else:
        result["regime"] = "bear"
    return result


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Compute RSI for the last value."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if not np.isnan(val) else 50.0


def compute_macd(series: pd.Series) -> dict:
    """MACD line, signal line, histogram."""
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": float(macd_line.iloc[-1]),
        "signal": float(signal_line.iloc[-1]),
        "histogram": float(histogram.iloc[-1]),
        "hist_prev": float(histogram.iloc[-2]) if len(histogram) >= 2 else 0.0,
        "accelerating": float(histogram.iloc[-1]) > float(histogram.iloc[-2]) if len(histogram) >= 2 else False,
    }


def compute_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0) -> dict:
    """Bollinger Band position: 0 = at lower band, 1 = at upper band."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    price = series.iloc[-1]
    band_width = upper.iloc[-1] - lower.iloc[-1]
    if band_width > 0:
        pct_b = (price - lower.iloc[-1]) / band_width
    else:
        pct_b = 0.5
    return {
        "pct_b": float(pct_b),
        "upper": float(upper.iloc[-1]),
        "lower": float(lower.iloc[-1]),
        "sma": float(sma.iloc[-1]),
    }


def compute_momentum_5d(series: pd.Series) -> float:
    """5-day price momentum as percentage."""
    if len(series) < 6:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-5] - 1) * 100)


def compute_volume_profile(df: pd.DataFrame) -> dict:
    """Volume relative to 20-day average."""
    if df.empty or "Volume" not in df.columns or len(df) < 20:
        return {"vol_ratio": 1.0, "above_avg": False}
    avg_vol = df["Volume"].tail(20).mean()
    last_vol = df["Volume"].iloc[-1]
    ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
    return {"vol_ratio": float(ratio), "above_avg": ratio > 1.0}


def compute_relative_strength(ticker_df: pd.DataFrame, spy_df: pd.DataFrame, days: int) -> float:
    """Relative strength vs SPY over N days."""
    if ticker_df.empty or spy_df.empty or len(ticker_df) < days or len(spy_df) < days:
        return 0.0
    ticker_ret = (ticker_df["Close"].iloc[-1] / ticker_df["Close"].iloc[-days] - 1) * 100
    spy_ret = (spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-days] - 1) * 100
    return float(ticker_ret - spy_ret)


# ---------------------------------------------------------------------------
# Sector -> ETF mapping
# ---------------------------------------------------------------------------
SECTOR_ETF_MAP = {
    "XLF": {"sector": "Financials", "rate_sensitive": "positive"},
    "XLE": {"sector": "Energy", "rate_sensitive": "neutral", "oil_corr": True},
    "XLU": {"sector": "Utilities", "rate_sensitive": "negative"},
    "XLK": {"sector": "Technology", "rate_sensitive": "negative"},
    "XLC": {"sector": "Communication Services", "rate_sensitive": "neutral"},
    "XLV": {"sector": "Healthcare", "rate_sensitive": "neutral"},
    "XLI": {"sector": "Industrials", "rate_sensitive": "neutral"},
    "XLP": {"sector": "Consumer Staples", "rate_sensitive": "neutral"},
    "XLY": {"sector": "Consumer Discretionary", "rate_sensitive": "negative"},
    "XLB": {"sector": "Materials", "rate_sensitive": "neutral"},
    "XLRE": {"sector": "Real Estate", "rate_sensitive": "negative"},
}

# ---------------------------------------------------------------------------
# Scoring functions — each returns a 0-100 sub-score
# ---------------------------------------------------------------------------

def score_macro(direction: str, ticker: str, vix_data: dict, treasury: dict, spy_regime: dict) -> dict:
    """
    Macro layer: 30% of total score.
    Components:
      - VIX level + trend (35% of macro)
      - VIX term structure (20% of macro)
      - Treasury yield movement (20% of macro)
      - SPY regime (25% of macro)
    """
    is_call = direction == "call"
    details = {}
    scores = {}

    # --- VIX level + trend (35%) ---
    vix = vix_data.get("vix")
    vix_5d = vix_data.get("vix_5d_ago")
    if vix is not None:
        # VIX < 15 and falling = bullish (good for calls, bad for puts)
        # VIX > 20 and rising = bearish (good for puts, bad for calls)
        if vix < 13:
            vix_env = 90 if is_call else 30
        elif vix < 15:
            vix_env = 75 if is_call else 40
        elif vix < 18:
            vix_env = 55  # neutral zone
        elif vix < 22:
            vix_env = 35 if is_call else 70
        else:
            vix_env = 20 if is_call else 85

        # Trend adjustment
        if vix_5d is not None:
            vix_change = vix - vix_5d
            if vix_change < -1.5:  # falling fast
                vix_env += 10 if is_call else -10
            elif vix_change > 1.5:  # rising fast
                vix_env += -10 if is_call else 10

        scores["vix_level_trend"] = max(0, min(100, vix_env))
        details["vix"] = f"{vix:.1f}" + (f" (was {vix_5d:.1f} 5d ago)" if vix_5d else "")
    else:
        scores["vix_level_trend"] = 50

    # --- VIX term structure (20%) ---
    ts = vix_data.get("term_structure", "unknown")
    if ts == "contango":
        # Contango = stable, good for buying options (lower near-term IV)
        scores["term_structure"] = 70
    elif ts == "backwardation":
        # Backwardation = stress, expensive options, calls risky, puts may work
        scores["term_structure"] = 30 if is_call else 65
    else:
        scores["term_structure"] = 50
    details["term_structure"] = ts

    # --- Treasury yield movement + yield curve regime (20%) ---
    rate_sens = SECTOR_ETF_MAP.get(ticker, {}).get("rate_sensitive", "neutral")
    yield_trend = treasury.get("yield_trend", "flat")
    yield_change = treasury.get("yield_5d_change", 0)
    curve_regime = treasury.get("curve_regime", "unknown")
    curve_2s10s = treasury.get("curve_2s10s")

    if rate_sens == "positive":
        # Financials benefit from rising rates AND steep curve (net interest margin)
        if yield_trend == "rising":
            scores["treasury"] = 80 if is_call else 30
        elif yield_trend == "falling":
            scores["treasury"] = 30 if is_call else 70
        else:
            scores["treasury"] = 55
        # Yield curve adjustment for financials (XLF)
        if curve_regime == "inverted":
            scores["treasury"] += -15 if is_call else 10  # Inverted curve kills bank margins
        elif curve_regime == "steep":
            scores["treasury"] += 10 if is_call else -10  # Steep curve = bank tailwind
    elif rate_sens == "negative":
        # Utilities, RE, Tech hurt by rising rates
        if yield_trend == "rising":
            scores["treasury"] = 30 if is_call else 75
        elif yield_trend == "falling":
            scores["treasury"] = 75 if is_call else 30
        else:
            scores["treasury"] = 55
        # Yield curve adjustment: inverted curve = flight to safety = supports XLU/XLRE calls
        if curve_regime == "inverted" and ticker in ("XLU", "XLRE"):
            scores["treasury"] += 10 if is_call else -5  # Defensive sectors benefit
    else:
        scores["treasury"] = 55  # Neutral sectors

    scores["treasury"] = max(0, min(100, scores["treasury"]))
    details["treasury"] = (
        f"10Y {treasury.get('yield_10y', '?')}%, trend={yield_trend}, "
        f"rate_sens={rate_sens}, curve={curve_regime}"
        + (f" (2s10s={curve_2s10s:+.2f}%)" if curve_2s10s is not None else "")
    )

    # --- SPY regime (25%) ---
    regime = spy_regime.get("regime", "unknown")
    if regime == "bull":
        scores["spy_regime"] = 80 if is_call else 25
    elif regime == "pullback":
        scores["spy_regime"] = 50  # Could be dip-buy or breakdown
    elif regime == "recovery":
        scores["spy_regime"] = 60 if is_call else 45
    elif regime == "bear":
        scores["spy_regime"] = 20 if is_call else 80
    else:
        scores["spy_regime"] = 50
    details["spy_regime"] = regime

    # Weighted macro score
    macro_score = (
        scores.get("vix_level_trend", 50) * 0.35
        + scores.get("term_structure", 50) * 0.20
        + scores.get("treasury", 50) * 0.20
        + scores.get("spy_regime", 50) * 0.25
    )

    return {
        "score": round(macro_score, 1),
        "components": scores,
        "details": details,
    }


def score_technical(direction: str, ticker_df: pd.DataFrame) -> dict:
    """
    Technical layer: 40% of total score.
    Components:
      - RSI(14) positioning (30% of tech)
      - MACD alignment (25% of tech)
      - Bollinger Band position (25% of tech)
      - 5-day momentum (20% of tech)
    """
    is_call = direction == "call"
    scores = {}
    details = {}

    if ticker_df.empty or len(ticker_df) < 26:
        return {"score": 50.0, "components": {}, "details": {"error": "Insufficient data"}}

    close = ticker_df["Close"]

    # --- RSI(14) (30%) ---
    rsi = compute_rsi(close)
    details["rsi"] = round(rsi, 1)

    if is_call:
        # Buying calls: RSI 40-60 = sweet spot (not overbought, room to run)
        # RSI > 70 = bad (overbought), RSI < 30 = contrarian play (risky)
        if rsi > 75:
            scores["rsi"] = 10  # Extremely overbought, terrible call timing
        elif rsi > 70:
            scores["rsi"] = 20
        elif rsi > 60:
            scores["rsi"] = 55  # Getting hot but still ok
        elif rsi > 45:
            scores["rsi"] = 80  # Sweet spot
        elif rsi > 35:
            scores["rsi"] = 65  # Pullback, good for mean reversion call
        else:
            scores["rsi"] = 40  # Oversold — could bounce but risky
    else:
        # Buying puts: RSI > 65 = good (overbought, room to fall)
        # RSI < 30 = bad (oversold, bounce risk)
        if rsi < 25:
            scores["rsi"] = 10  # Extremely oversold, terrible put timing
        elif rsi < 30:
            scores["rsi"] = 20
        elif rsi < 40:
            scores["rsi"] = 55
        elif rsi < 55:
            scores["rsi"] = 70
        elif rsi < 65:
            scores["rsi"] = 80  # Room to fall
        else:
            scores["rsi"] = 85  # Overbought, good put timing

    # --- MACD alignment (25%) ---
    macd = compute_macd(close)
    details["macd_hist"] = round(macd["histogram"], 4)
    details["macd_accelerating"] = macd["accelerating"]

    if is_call:
        if macd["histogram"] > 0 and macd["accelerating"]:
            scores["macd"] = 90  # Bullish and accelerating
        elif macd["histogram"] > 0:
            scores["macd"] = 65  # Bullish but decelerating
        elif macd["histogram"] < 0 and not macd["accelerating"]:
            scores["macd"] = 25  # Bearish and getting worse
        else:
            scores["macd"] = 45  # Bearish but improving (potential crossover)
    else:
        if macd["histogram"] < 0 and not macd["accelerating"]:
            scores["macd"] = 85  # Bearish and accelerating down
        elif macd["histogram"] < 0:
            scores["macd"] = 60  # Bearish but decelerating
        elif macd["histogram"] > 0 and macd["accelerating"]:
            scores["macd"] = 20  # Bullish and getting stronger — bad for puts
        else:
            scores["macd"] = 45  # Bullish but weakening

    # --- Bollinger Band position (25%) ---
    bb = compute_bollinger(close)
    pct_b = bb["pct_b"]
    details["bb_pct_b"] = round(pct_b, 3)

    if is_call:
        # For calls, best entry near lower band (oversold bounce) or middle (trend continuation)
        # Near upper band = already extended
        if pct_b < 0.2:
            scores["bb"] = 85  # Near lower band — great for calls if trend intact
        elif pct_b < 0.4:
            scores["bb"] = 75
        elif pct_b < 0.6:
            scores["bb"] = 65  # Middle — neutral
        elif pct_b < 0.8:
            scores["bb"] = 45  # Upper half — less upside room
        else:
            scores["bb"] = 25  # At upper band — overbought
    else:
        # For puts, best entry near upper band
        if pct_b > 0.8:
            scores["bb"] = 85
        elif pct_b > 0.6:
            scores["bb"] = 75
        elif pct_b > 0.4:
            scores["bb"] = 60
        elif pct_b > 0.2:
            scores["bb"] = 40
        else:
            scores["bb"] = 20  # At lower band — bad put timing

    # --- 5-day momentum (15%) ---
    mom5d = compute_momentum_5d(close)
    details["momentum_5d_pct"] = round(mom5d, 2)

    if is_call:
        if mom5d > 5:
            scores["momentum"] = 40
        elif mom5d > 2:
            scores["momentum"] = 80
        elif mom5d > 0:
            scores["momentum"] = 70
        elif mom5d > -2:
            scores["momentum"] = 55
        else:
            scores["momentum"] = 30
    else:
        if mom5d < -5:
            scores["momentum"] = 40
        elif mom5d < -2:
            scores["momentum"] = 80
        elif mom5d < 0:
            scores["momentum"] = 70
        elif mom5d < 2:
            scores["momentum"] = 55
        else:
            scores["momentum"] = 30

    # --- Momentum acceleration quality (10%) — NEW ---
    # Is momentum speeding up or slowing down? Second derivative of returns.
    # Risk-adjusted: compare recent 5d return/vol vs prior 5d return/vol
    mom_accel_score = 50  # neutral default
    if len(close) >= 15:
        try:
            ret_5d = float((close.iloc[-1] / close.iloc[-5] - 1))
            ret_prior_5d = float((close.iloc[-5] / close.iloc[-10] - 1))
            vol_5d = float(close.pct_change().tail(5).std()) or 0.01
            vol_prior_5d = float(close.pct_change().iloc[-10:-5].std()) or 0.01

            ra_recent = ret_5d / vol_5d  # Risk-adjusted recent momentum
            ra_prior = ret_prior_5d / vol_prior_5d

            if is_call:
                # For calls: accelerating positive momentum = great
                if ra_recent > ra_prior and ret_5d > 0:
                    mom_accel_score = 85  # Accelerating in our direction
                elif ra_recent > 0 and ra_recent < ra_prior:
                    mom_accel_score = 45  # Positive but decelerating — late entry risk
                elif ra_recent < 0:
                    mom_accel_score = 30  # Momentum against us
                else:
                    mom_accel_score = 55
            else:
                # For puts: accelerating negative momentum = great
                if ra_recent < ra_prior and ret_5d < 0:
                    mom_accel_score = 85
                elif ra_recent < 0 and ra_recent > ra_prior:
                    mom_accel_score = 45  # Negative but slowing
                elif ra_recent > 0:
                    mom_accel_score = 30
                else:
                    mom_accel_score = 55
            details["mom_accel_ra_recent"] = round(ra_recent, 3)
            details["mom_accel_ra_prior"] = round(ra_prior, 3)
        except Exception:
            pass
    scores["mom_accel"] = mom_accel_score
    details["mom_accel_score"] = mom_accel_score

    # Weighted technical score (rebalanced to include momentum acceleration)
    tech_score = (
        scores.get("rsi", 50) * 0.25
        + scores.get("macd", 50) * 0.25
        + scores.get("bb", 50) * 0.20
        + scores.get("momentum", 50) * 0.15
        + scores.get("mom_accel", 50) * 0.15
    )

    return {
        "score": round(tech_score, 1),
        "components": scores,
        "details": details,
    }


def score_market_structure(
    direction: str,
    ticker: str,
    ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
) -> dict:
    """
    Market structure layer: 30% of total score.
    Components (rebalanced Aug 2026 to include price action structure):
      - Price action structure (30% of structure) — swing highs/lows, S/R, range position
      - Sector rotation / relative strength (25% of structure)
      - Volume profile (15% of structure)
      - Multi-timeframe relative strength (15% of structure)
      - Sector flow direction from existing signals (15% of structure)
    """
    is_call = direction == "call"
    scores = {}
    details = {}

    if ticker_df.empty:
        return {"score": 50.0, "components": {}, "details": {"error": "No data"}}

    # --- Relative strength vs SPY: 5d, 10d, 21d (35%) ---
    rs_5 = compute_relative_strength(ticker_df, spy_df, 5)
    rs_10 = compute_relative_strength(ticker_df, spy_df, 10)
    rs_21 = compute_relative_strength(ticker_df, spy_df, 21)
    details["rel_str_5d"] = round(rs_5, 2)
    details["rel_str_10d"] = round(rs_10, 2)
    details["rel_str_21d"] = round(rs_21, 2)

    # Average relative strength
    avg_rs = (rs_5 + rs_10 + rs_21) / 3

    if is_call:
        # For calls: positive RS = money flowing in
        if avg_rs > 3:
            scores["rel_strength"] = 90
        elif avg_rs > 1:
            scores["rel_strength"] = 75
        elif avg_rs > -1:
            scores["rel_strength"] = 55
        elif avg_rs > -3:
            scores["rel_strength"] = 35
        else:
            scores["rel_strength"] = 15
    else:
        # For puts: negative RS = money flowing out
        if avg_rs < -3:
            scores["rel_strength"] = 90
        elif avg_rs < -1:
            scores["rel_strength"] = 75
        elif avg_rs < 1:
            scores["rel_strength"] = 55
        elif avg_rs < 3:
            scores["rel_strength"] = 35
        else:
            scores["rel_strength"] = 15

    # --- Volume profile (25%) ---
    vol = compute_volume_profile(ticker_df)
    details["vol_ratio"] = round(vol["vol_ratio"], 2)

    # High volume = conviction in move direction
    if vol["vol_ratio"] > 1.5:
        scores["volume"] = 80  # Strong conviction
    elif vol["vol_ratio"] > 1.0:
        scores["volume"] = 65  # Above average
    elif vol["vol_ratio"] > 0.7:
        scores["volume"] = 45  # Below average — less conviction
    else:
        scores["volume"] = 30  # Very low volume — illiquid / no interest

    # --- Sector rotation phase from existing signals (25%) ---
    sector_score = 50  # default neutral
    try:
        with open(AGENTIC_SIGNALS) as f:
            sig_data = json.load(f)
        for sig in sig_data.get("signals", []):
            if sig.get("ticker") == ticker:
                n_conf = sig.get("n_confirming", 0)
                n_confl = sig.get("n_conflicting", 0)
                conf_score = sig.get("confidence_score", 0.5)

                # More confirming sources = stronger sector alignment
                if n_conf >= 5 and conf_score > 0.7:
                    sector_score = 85
                elif n_conf >= 3:
                    sector_score = 65
                elif n_confl > n_conf:
                    sector_score = 30
                break
    except Exception:
        pass
    scores["sector_flow"] = sector_score
    details["sector_flow_score"] = sector_score

    # --- Multi-timeframe consistency (15%) ---
    # Check if 5d, 10d, 21d relative strength all agree in direction
    if is_call:
        agree = sum(1 for r in [rs_5, rs_10, rs_21] if r > 0)
    else:
        agree = sum(1 for r in [rs_5, rs_10, rs_21] if r < 0)

    if agree == 3:
        scores["mtf_consistency"] = 90  # All timeframes agree
    elif agree == 2:
        scores["mtf_consistency"] = 60
    elif agree == 1:
        scores["mtf_consistency"] = 35
    else:
        scores["mtf_consistency"] = 15  # All disagree with our direction
    details["mtf_agreement"] = f"{agree}/3 timeframes"

    # --- Price action structure (30%) ---
    # Uses fractal swing highs/lows, multi-TF range position, pivot points, S/R levels
    if PRICE_STRUCTURE_AVAILABLE:
        try:
            pa_result = score_price_structure(ticker, direction, df=ticker_df)
            scores["price_action"] = pa_result["score"]
            details["price_action_raw"] = pa_result.get("raw_score")
            details["price_action_components"] = pa_result.get("components", {})
            pa_det = pa_result.get("details", {})
            if "dist_to_support_pct" in pa_det:
                details["dist_to_support_pct"] = pa_det["dist_to_support_pct"]
            if "dist_to_resistance_pct" in pa_det:
                details["dist_to_resistance_pct"] = pa_det["dist_to_resistance_pct"]
            if "pct_from_52w_high" in pa_det:
                details["pct_from_52w_high"] = pa_det["pct_from_52w_high"]
        except Exception as e:
            log.warning(f"Price action structure failed for {ticker}: {e}")
            scores["price_action"] = 50
    else:
        scores["price_action"] = 50

    # Weighted structure score (rebalanced to include price action)
    struct_score = (
        scores.get("price_action", 50) * 0.30
        + scores.get("rel_strength", 50) * 0.25
        + scores.get("volume", 50) * 0.15
        + scores.get("sector_flow", 50) * 0.15
        + scores.get("mtf_consistency", 50) * 0.15
    )

    return {
        "score": round(struct_score, 1),
        "components": scores,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def compute_timing_score(ticker: str, direction: str, strike: float = None) -> dict:
    """
    Compute the full timing score for a single trade.
    Returns dict with total_score, layer scores, recommendation, and details.
    """
    log.info(f"Computing timing score for {ticker} {direction}" + (f" ${strike}" if strike else ""))

    # Fetch all data upfront
    ticker_df = fetch_ticker_data(ticker)
    spy_df = fetch_ticker_data("SPY", period="3mo")
    vix_data = fetch_vix_data()
    treasury = fetch_treasury_data()
    spy_regime = fetch_spy_regime()

    # Score each layer
    macro = score_macro(direction, ticker, vix_data, treasury, spy_regime)
    technical = score_technical(direction, ticker_df)
    structure = score_market_structure(direction, ticker, ticker_df, spy_df)

    # Weighted total: Structure 40%, Macro 35%, Technical 25%
    # HC #782: Standard indicators (RSI/MACD/BB) are fully arbitraged.
    # Edge is in proprietary signals (structure: sector rotation, flow, rel strength)
    # and macro context (rates, VIX, regime). Technicals are confirmation only.
    total = (
        macro["score"] * 0.35
        + technical["score"] * 0.25
        + structure["score"] * 0.40
    )

    # --- Day-of-week adjustment: DISABLED ---
    # Originally wired +3 Wed/Thu, -3 Mon for RSI<35 dip entries.
    # REVERTED 2026-08-21: Reconciliation test proved the edge comes from
    # overnight gaps (close-to-close), NOT intraday (open-to-close).
    # Since we enter at market open, we've already missed the overnight gap.
    # Wed RSI<35 open-to-close 5d return is -0.02% (no edge), p=0.31.
    # See research/rsi_reconciliation.txt for full analysis.
    dow_adj = 0
    total = round(total + dow_adj, 1)

    # Recommendation
    if total >= 80:
        recommendation = "ENTER_AT_OPEN"
        action = "Enter immediately at open — strong timing across all layers"
    elif total >= 60:
        recommendation = "ENTER_LIMIT"
        action = "Enter but use limit order, be patient for a good fill"
    elif total >= 40:
        recommendation = "DEFER"
        action = "Defer to afternoon check — conditions are mixed"
    else:
        recommendation = "SKIP"
        action = "Skip today — timing is unfavorable despite directional thesis"

    # Identify strongest and weakest layer
    layer_scores = {"macro": macro["score"], "technical": technical["score"], "structure": structure["score"]}
    strongest = max(layer_scores, key=layer_scores.get)
    weakest = min(layer_scores, key=layer_scores.get)

    result = {
        "ticker": ticker,
        "direction": direction,
        "strike": strike,
        "total_score": total,
        "recommendation": recommendation,
        "action": action,
        "strongest_layer": strongest,
        "weakest_layer": weakest,
        "dow_adjustment": dow_adj,
        "macro": macro,
        "technical": technical,
        "market_structure": structure,
    }

    log.info(f"  {ticker} {direction}: SCORE={total} → {recommendation}")
    return result


def get_recommendation_text(score: float) -> str:
    if score >= 80:
        return "ENTER_AT_OPEN"
    elif score >= 60:
        return "ENTER_LIMIT"
    elif score >= 40:
        return "DEFER"
    else:
        return "SKIP"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Entry Timing Score System")
    parser.add_argument("--ticker", help="Single ticker to score")
    parser.add_argument("--dir", choices=["call", "put"], help="Direction")
    parser.add_argument("--strike", type=float, help="Strike price")
    parser.add_argument(
        "--targets",
        help="Comma-separated targets: TICKER:DIR:STRIKE (e.g., XLF:call:58,XLE:call:60)",
    )
    args = parser.parse_args()

    targets = []

    if args.targets:
        for t in args.targets.split(","):
            parts = t.strip().split(":")
            if len(parts) >= 2:
                ticker = parts[0].upper()
                direction = parts[1].lower()
                strike = float(parts[2]) if len(parts) > 2 else None
                targets.append((ticker, direction, strike))
    elif args.ticker and args.dir:
        targets.append((args.ticker.upper(), args.dir, args.strike))
    else:
        # Default: read from agentic_signals.json
        try:
            with open(AGENTIC_SIGNALS) as f:
                sig_data = json.load(f)
            for sig in sig_data.get("signals", []):
                if sig.get("affordable", True) and sig.get("confidence_score", 0) > 0.5:
                    ticker = sig["ticker"]
                    direction = sig.get("recommended_option", "call")
                    strike = sig.get("recommended_strike")
                    targets.append((ticker, direction, strike))
        except Exception as e:
            log.error(f"Could not read agentic signals: {e}")
            # Fallback to the Aug 7 planned trades
            targets = [
                ("XLF", "call", 58.0),
                ("XLE", "call", 60.0),
                ("XLU", "put", 44.0),
            ]

    if not targets:
        log.warning("No targets to score. Using default planned trades.")
        targets = [
            ("XLF", "call", 58.0),
            ("XLE", "call", 60.0),
            ("XLU", "put", 44.0),
        ]

    results = []
    for ticker, direction, strike in targets:
        result = compute_timing_score(ticker, direction, strike)
        results.append(result)

    # Build output
    output = {
        "generated_at": datetime.now().isoformat(),
        "n_scored": len(results),
        "scores": results,
        "summary": {},
    }

    # Summary table
    for r in results:
        output["summary"][r["ticker"]] = {
            "score": r["total_score"],
            "recommendation": r["recommendation"],
            "direction": r["direction"],
            "strike": r["strike"],
            "macro": r["macro"]["score"],
            "technical": r["technical"]["score"],
            "structure": r["market_structure"]["score"],
        }

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Results saved to {OUTPUT_FILE}")

    # Print summary
    print("\n" + "=" * 70)
    print("ENTRY TIMING SCORES")
    print("=" * 70)
    for r in results:
        strike_str = f" ${r['strike']}" if r['strike'] else ""
        print(
            f"\n  {r['ticker']} {r['direction'].upper()}{strike_str}"
            f"  →  SCORE: {r['total_score']}/100  [{r['recommendation']}]"
        )
        print(f"    Macro: {r['macro']['score']:.0f}  |  Technical: {r['technical']['score']:.0f}"
              f"  |  Structure: {r['market_structure']['score']:.0f}")
        print(f"    {r['action']}")

        # Key details
        td = r["technical"].get("details", {})
        md = r["macro"].get("details", {})
        sd = r["market_structure"].get("details", {})
        detail_parts = []
        if "rsi" in td:
            detail_parts.append(f"RSI={td['rsi']}")
        if "momentum_5d_pct" in td:
            detail_parts.append(f"5dMom={td['momentum_5d_pct']:+.1f}%")
        if "spy_regime" in md:
            detail_parts.append(f"SPY={md['spy_regime']}")
        if "rel_str_5d" in sd:
            detail_parts.append(f"RS5d={sd['rel_str_5d']:+.1f}%")
        if detail_parts:
            print(f"    Key: {', '.join(detail_parts)}")

    print("\n" + "=" * 70)
    return output


if __name__ == "__main__":
    main()
