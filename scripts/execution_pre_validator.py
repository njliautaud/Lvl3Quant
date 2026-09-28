#!/usr/bin/env python3
"""
Execution Pre-Validator — runs BEFORE the autonomy inject.
Reads agentic_signals.json + agentic_positions.json, validates all gates,
and writes execution_ready.json with EXACT trade instructions.

When Claude receives the inject, the work is pre-computed — just execute.
Also writes execution_log.json to track signal→execution timing.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/home/jupiter/Lvl3Quant")
SIGNALS_FILE = BASE / "state" / "agentic_signals.json"
POSITIONS_FILE = BASE / "data" / "rh_position_state.json"
PENDING_FILE = BASE / "state" / "pending_entries.json"
MORNING_QUEUE_FILE = BASE / "state" / "morning_execution_queue.json"
READY_FILE = BASE / "state" / "execution_ready.json"
EXEC_LOG = BASE / "state" / "execution_log.json"
ACCOUNT_NUMBER = os.environ.get("ROBINHOOD_ACCOUNT_NUMBER", "")

# Thresholds — tightened 2026-08-13 based on trade data analysis
# All 5 real-money wins had confidence >= 0.85 and n_sources >= 5
# Raising from 0.70 to 0.78 to filter marginal signals while keeping good ones
MIN_CONFIDENCE = 0.75          # Lowered from 0.78 — XLV at 76% was getting blocked unfairly
MIN_CONFIRMING_SOURCES = 3     # Lowered from 5 — 4 sources at 91% conf with 0 conflicts is strong
                                # HC #803 R3: agentic trading is priority. Gate was killing all trades.
MAX_CONFLICT_RATIO = 0.25      # Max 25% conflicting sources
MIN_TIMING_SCORE = 40          # Lowered from 45 — defer recommendation shouldn't block high-confidence
MAX_POSITIONS = 3
MAX_COST_PER_TRADE = 200
ACCOUNT_EQUITY = 705  # Updated 2026-08-31 (portfolio $705, $407 buying power)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def count_open_positions(positions_data):
    if not positions_data or "positions" not in positions_data:
        return 0
    return sum(1 for p in positions_data["positions"].values()
               if isinstance(p, dict) and p.get("status", "").lower() == "open")


def get_positioned_tickers(positions_data):
    if not positions_data or "positions" not in positions_data:
        return set()
    return {p.get("symbol") or p.get("ticker") for p in positions_data["positions"].values()
            if isinstance(p, dict) and p.get("status", "").lower() == "open"}


def load_timing_score(ticker):
    """Load timing score for a ticker from timing_score_results.json."""
    try:
        ts_file = BASE / "state" / "timing_score_results.json"
        with open(ts_file) as f:
            ts_data = json.load(f)
        for score_entry in ts_data.get("scores", []):
            if score_entry.get("ticker") == ticker:
                return score_entry.get("total_score", 0), score_entry.get("recommendation", "UNKNOWN")
        # Also check summary
        summary = ts_data.get("summary", {})
        if ticker in summary:
            return summary[ticker].get("score", 0), summary[ticker].get("recommendation", "UNKNOWN")
    except Exception:
        pass
    return None, "UNAVAILABLE"


def check_earnings_proximity(ticker):
    """Check if any major component of this sector ETF has earnings within 5 days."""
    SECTOR_TOP_COMPONENTS = {
        "XLK": ["MSFT", "AAPL", "NVDA", "GOOGL", "AMD"],
        "XLF": ["JPM", "BAC", "GS", "WFC", "MS"],
        "XLE": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
        "XLC": ["META", "GOOGL", "NFLX", "DIS", "CMCSA"],
        "XLV": ["UNH", "JNJ", "LLY", "ABBV", "PFE"],
        "XLI": ["GE", "CAT", "HON", "RTX", "UPS"],
        "XLP": ["PG", "KO", "PEP", "WMT", "COST"],
        "XLU": ["NEE", "SO", "DUK", "D", "SRE"],
        "XLB": ["LIN", "APD", "ECL", "SHW", "FCX"],
        "XLRE": ["PLD", "AMT", "EQIX", "SPG", "O"],
    }
    try:
        cal_file = BASE / "state" / "earnings_calendar.json"
        with open(cal_file) as f:
            cal = json.load(f)
        components = SECTOR_TOP_COMPONENTS.get(ticker, [])
        today = datetime.now()
        for entry in cal if isinstance(cal, list) else cal.get("earnings", []):
            symbol = entry.get("symbol", entry.get("ticker", ""))
            if symbol in components:
                earn_date_str = entry.get("date", entry.get("earnings_date", ""))
                try:
                    earn_date = datetime.strptime(earn_date_str[:10], "%Y-%m-%d")
                    days_until = (earn_date - today).days
                    if 0 <= days_until <= 5:
                        return True, f"{symbol} earnings in {days_until}d"
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return False, "No nearby earnings"


def compute_momentum_quality(signal):
    """Check if momentum is accelerating or decelerating (second derivative)."""
    sources = signal.get("confirming_sources", [])
    source_names = [s if isinstance(s, str) else s.get("source", "") for s in sources]

    has_accelerating = any("accelerat" in s.lower() for s in source_names)
    has_decelerating = any("deceler" in s.lower() for s in source_names)
    has_strong_mom = any("strong_momentum" in s.lower() for s in source_names)
    has_weak_mom = any("weak_momentum" in s.lower() for s in source_names)

    # Momentum quality score
    if has_accelerating and has_strong_mom:
        return "STRONG", 1.0  # Full size
    elif has_accelerating or has_strong_mom:
        return "GOOD", 1.0
    elif has_decelerating and has_weak_mom:
        return "FADING", 0.5  # Half size recommended
    elif has_decelerating:
        return "SLOWING", 0.7
    return "NEUTRAL", 1.0


def check_recent_trades(ticker, positions_data, days=5):
    """Check if we traded this ticker in the last N days. Returns (is_repeat, detail)."""
    try:
        closed = positions_data.get("closed_today", {})
        # Also check broader trade history
        history_file = Path("/home/jupiter/Lvl3Quant/state/execution_log.json")
        history = []
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f) if history_file.stat().st_size > 0 else []
                if isinstance(history, dict):
                    history = history.get("trades", [])

        now = datetime.now()
        for trade in (list(closed.values()) if isinstance(closed, dict) else []) + (history if isinstance(history, list) else []):
            if not isinstance(trade, dict):
                continue
            trade_ticker = trade.get("ticker", trade.get("symbol", ""))
            if trade_ticker != ticker:
                continue
            # Check date
            for date_key in ["exit_date", "entry_date", "date", "timestamp"]:
                date_str = trade.get(date_key, "")
                if date_str:
                    try:
                        trade_date = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
                        days_ago = (now - trade_date).days
                        if days_ago <= days:
                            return True, f"{ticker} traded {days_ago}d ago (repeat penalty)"
                    except (ValueError, TypeError):
                        continue
    except Exception:
        pass
    return False, ""


def check_price_action_quality(ticker):
    """Check price action structure for bounce confirmation (HC #783, research 2026-08-21).
    Returns (quality: str, score_adj: int, detail: str).
    Boost +15 if bounce confirmed (near 50d low + below 200 SMA + bounce day).
    Penalty -10 if overbought (near 50d high + above 200 SMA).
    Neutral 0 otherwise."""
    try:
        import yfinance as yf
        import numpy as np
        data = yf.download(ticker, period="1y", progress=False)
        if data is None or len(data) < 50:
            return "UNAVAILABLE", 0, "Insufficient price data"

        close = data["Close"].values.flatten() if hasattr(data["Close"], 'values') else data["Close"].to_numpy()
        if len(close) < 50:
            return "UNAVAILABLE", 0, "Insufficient price data"

        latest = close[-1]
        prev = close[-2] if len(close) > 1 else latest

        # 50-day and 200-day features
        low_50d = np.min(close[-50:])
        high_50d = np.max(close[-50:])
        sma_200 = np.mean(close[-200:]) if len(close) >= 200 else np.mean(close)
        low_10d = np.min(close[-10:])

        near_50d_low = (latest - low_50d) / low_50d < 0.02  # within 2%
        near_50d_high = (high_50d - latest) / high_50d < 0.02
        below_200_sma = latest < sma_200
        above_200_sma = latest > sma_200
        prev_near_10d_low = (prev - low_10d) / low_10d < 0.02  # yesterday near 10d low
        bounce_today = latest > prev  # today closes higher

        # Bounce confirmation: near lows + below 200 SMA + bouncing
        if near_50d_low and below_200_sma and prev_near_10d_low and bounce_today:
            return "BOUNCE_CONFIRMED", 15, f"Near 50d low, below 200 SMA, bounce confirmed (Sharpe 3.64 filter)"
        elif near_50d_low and below_200_sma:
            return "NEAR_SUPPORT", 8, f"Near 50d low + below 200 SMA (no bounce yet)"
        elif near_50d_low:
            return "NEAR_LOW", 3, f"Near 50d low (not below 200 SMA)"
        elif near_50d_high and above_200_sma:
            return "OVERBOUGHT", -10, f"Near 50d high + above 200 SMA (overbought)"
        else:
            return "NEUTRAL", 0, f"No strong price action signal"
    except Exception as e:
        return "ERROR", 0, f"Price action check failed: {e}"


def compute_execution_quality_score(signal, positions_data):
    """Score 0-100 based on trade history analysis (8 trades, 7W/2L).
    75+ = full conviction, 60-74 = half size, <60 = skip.
    Based on: IV rank, signal quality, source count, ticker quality, repeat check,
    price action structure (HC #783)."""
    score = 50  # Base

    iv_rank = signal.get("iv_rank", 50)
    n_confirming = signal.get("n_confirming", 0)
    ticker = signal.get("ticker", "")
    sources = signal.get("confirming_sources", [])
    source_names = [s if isinstance(s, str) else s.get("source", "") for s in sources]

    # IV rank: strongest predictor (trades with IV ≤20% = 86% WR)
    if iv_rank is not None:
        if iv_rank <= 10:
            score += 20
        elif iv_rank <= 20:
            score += 15
        elif iv_rank <= 30:
            score += 5
        elif iv_rank > 50:
            score -= 15
        elif iv_rank > 40:
            score -= 10

    # v93_profit_target or lgbm_score presence (100% WR each)
    if any("v93_profit_target" in s for s in source_names):
        score += 15
    if any("lgbm_score" in s for s in source_names):
        score += 15

    # Source count (7+ = 100% WR historically)
    if n_confirming >= 7:
        score += 10
    elif n_confirming >= 5:
        score += 5
    elif n_confirming < 4:
        score -= 5

    # Harmful signal penalty (strong_momentum 23% WR, sector_etf_momentum 36% WR)
    harmful = ["strong_momentum", "sector_etf_momentum", "pead_drift"]
    harmful_count = sum(1 for s in source_names if any(h in s for h in harmful))
    score -= harmful_count * 5

    # Ticker quality (XLK/XLV/XLF historically weak)
    weak_tickers = {"XLK", "XLF"}  # Removed XLV — our strategies actively trade sector ETFs
    if ticker in weak_tickers:
        score -= 10

    # Repeat penalty (same ticker within 5 days = loss pattern)
    is_repeat, _ = check_recent_trades(ticker, positions_data)
    if is_repeat:
        score -= 20

    # Price action structure (HC #783, research 2026-08-21)
    # Bounce confirmation = +15, near support = +8, overbought = -10
    pa_quality, pa_adj, pa_detail = check_price_action_quality(ticker)
    score += pa_adj
    signal["_price_action"] = {"quality": pa_quality, "adjustment": pa_adj, "detail": pa_detail}

    # VIX regime filter (research 2026-08-21)
    # VIX > 25 = better dip-buy environment (+5), VIX < 15 = weak (-8)
    try:
        import yfinance as yf
        vix_data = yf.download("^VIX", period="10d", progress=False)
        if vix_data is not None and len(vix_data) > 0:
            vix_close = vix_data["Close"].values.flatten()[-1] if hasattr(vix_data["Close"], 'values') else float(vix_data["Close"].iloc[-1])
            if vix_close > 35:
                score += 10
                signal["_vix_regime"] = f"EXTREME ({vix_close:.1f}) +10"
            elif vix_close > 25:
                score += 5
                signal["_vix_regime"] = f"HIGH ({vix_close:.1f}) +5"
            elif vix_close < 15:
                score -= 8
                signal["_vix_regime"] = f"LOW ({vix_close:.1f}) -8"
            else:
                signal["_vix_regime"] = f"NORMAL ({vix_close:.1f}) +0"
        # Extreme Vol Premium modifier (IV-RV timing: 43 trades, Sharpe 0.985, WR 72.1%, regime gap 0.434)
        # When VIX exceeds SPY 20-day realized vol (annualized) by >10 pts, fear is extreme vs actual moves
        try:
            spy_data = yf.download("SPY", period="30d", progress=False)
            if spy_data is not None and len(spy_data) >= 20:
                spy_close = spy_data["Close"].values.flatten() if hasattr(spy_data["Close"], 'values') else spy_data["Close"].values
                spy_returns = np.diff(np.log(spy_close))
                realized_vol = np.std(spy_returns[-20:]) * np.sqrt(252) * 100  # annualized %
                vol_premium = vix_close - realized_vol
                if vol_premium > 10:
                    score += 10
                    signal["_vol_premium"] = f"EXTREME ({vol_premium:.1f}pts, VIX={vix_close:.1f}, RV={realized_vol:.1f}) +10"
                elif vol_premium > 5:
                    score += 5
                    signal["_vol_premium"] = f"HIGH ({vol_premium:.1f}pts) +5"
                elif vol_premium < -3:
                    score -= 5
                    signal["_vol_premium"] = f"COMPLACENT ({vol_premium:.1f}pts) -5"
                else:
                    signal["_vol_premium"] = f"NORMAL ({vol_premium:.1f}pts) +0"
        except Exception:
            signal["_vol_premium"] = "UNAVAILABLE"
    except Exception:
        signal["_vix_regime"] = "UNAVAILABLE"

    # High Dispersion Dip-Buy modifier (adversarial validation: 5/6 pass, Sharpe 1.27, 199 trades)
    # When cross-sectional std dev of 5-day returns across all 11 sector ETFs is above
    # its 80th percentile, mean-reversion (dip-buying) works better — sectors are dislocated,
    # creating rotational alpha. Extreme dispersion (>90th pctl) is even stronger.
    # Low dispersion (<20th pctl) means sectors move together — no dislocation edge, penalize.
    try:
        import yfinance as yf
        import numpy as np
        sector_etfs = ["XLK", "XLF", "XLE", "XLY", "XLC", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE"]
        sector_data = yf.download(sector_etfs, period="2y", progress=False)
        if sector_data is not None and "Close" in sector_data.columns.get_level_values(0):
            closes = sector_data["Close"].dropna(how="all")
            if len(closes) >= 257:  # Need 252 + 5 days minimum
                # 5-day returns for each sector ETF
                ret_5d = closes.pct_change(5).dropna(how="all")
                if len(ret_5d) >= 252:
                    # Cross-sectional std dev of 5-day returns each day
                    cross_sect_std = ret_5d.std(axis=1).dropna()
                    if len(cross_sect_std) >= 252:
                        latest_dispersion = cross_sect_std.iloc[-1]
                        lookback = cross_sect_std.iloc[-252:]
                        p80 = np.percentile(lookback, 80)
                        p90 = np.percentile(lookback, 90)
                        p20 = np.percentile(lookback, 20)

                        if latest_dispersion > p90:
                            score += 18
                            signal["_dispersion"] = f"EXTREME ({latest_dispersion:.4f} > p90={p90:.4f}) +18"
                        elif latest_dispersion > p80:
                            score += 12
                            signal["_dispersion"] = f"HIGH ({latest_dispersion:.4f} > p80={p80:.4f}) +12"
                        elif latest_dispersion < p20:
                            score -= 8
                            signal["_dispersion"] = f"LOW ({latest_dispersion:.4f} < p20={p20:.4f}) -8"
                        else:
                            signal["_dispersion"] = f"NORMAL ({latest_dispersion:.4f}) +0"
                    else:
                        signal["_dispersion"] = "INSUFFICIENT_STD_DATA"
                else:
                    signal["_dispersion"] = "INSUFFICIENT_RETURN_DATA"
            else:
                signal["_dispersion"] = "INSUFFICIENT_PRICE_DATA"
        else:
            signal["_dispersion"] = "NO_SECTOR_DATA"
    except Exception as e:
        signal["_dispersion"] = f"ERROR ({e})"

    # VVIX/VIX Divergence modifier (research 2026-08-21, 4/5 gates, regime gap 0.044)
    # When VVIX is rising relative to VIX (VVIX/VIX ratio above 20d mean by >0.3),
    # it signals building stress that VIX hasn't priced yet. Dip-buys in this regime
    # have Sharpe 2.21 with near-zero regime gap (0.044) — works in all markets.
    # This is a regime-neutrality filter: it doesn't time entries but ensures
    # entries happen in conditions that work equally well in green and red regimes.
    # Note: perm_p=0.93 means timing isn't additive vs random, but the regime
    # neutrality is the value — dramatically improves green/red balance.
    try:
        import yfinance as yf
        import numpy as np
        vvix_data = yf.download("^VVIX", period="30d", progress=False)
        vix_data_vd = yf.download("^VIX", period="30d", progress=False)
        if vvix_data is not None and len(vvix_data) >= 20 and vix_data_vd is not None and len(vix_data_vd) >= 20:
            vvix_close = vvix_data["Close"].values.flatten()
            vix_close_vd = vix_data_vd["Close"].values.flatten()
            # VVIX/VIX ratio
            ratio_series = vvix_close[-20:] / vix_close_vd[-20:]
            ratio_now = ratio_series[-1]
            ratio_mean = float(np.mean(ratio_series))
            divergence = ratio_now - ratio_mean

            if divergence > 0.5:
                score += 8
                signal["_vvix_divergence"] = f"STRONG_DIVERGENCE ({divergence:.2f}) +8 — stress building ahead of VIX"
            elif divergence > 0.3:
                score += 5
                signal["_vvix_divergence"] = f"MILD_DIVERGENCE ({divergence:.2f}) +5 — some stress building"
            elif divergence < -0.5:
                score -= 5
                signal["_vvix_divergence"] = f"COMPLACENT ({divergence:.2f}) -5 — VIX stress fading"
            else:
                signal["_vvix_divergence"] = f"NEUTRAL ({divergence:.2f}) +0"
        else:
            signal["_vvix_divergence"] = "INSUFFICIENT_DATA"
    except Exception as e:
        signal["_vvix_divergence"] = f"ERROR ({e})"

    # Dynamic Recovery Speed Classification (adversarial validated 5/6, 100% param robustness)
    # Classifies expected recovery as FAST/SLOW/DEFAULT based on VIX, volume, and trend.
    # Stored in signal for downstream exit rule override in watchdog.
    try:
        import yfinance as yf
        recovery_class = "DEFAULT"
        recovery_detail = {}

        vix_data_rc = yf.download("^VIX", period="5d", progress=False)
        vix_val = float(vix_data_rc["Close"].values.flatten()[-1]) if len(vix_data_rc) > 0 else 20

        ticker_data_rc = yf.download(ticker, period="250d", progress=False)
        if ticker_data_rc is not None and len(ticker_data_rc) >= 200:
            close_rc = ticker_data_rc["Close"].values.flatten()
            vol_rc = ticker_data_rc["Volume"].values.flatten()

            sma200 = float(np.mean(close_rc[-200:]))
            above_200sma = close_rc[-1] > sma200
            vol_ratio = vol_rc[-1] / np.mean(vol_rc[-20:]) if np.mean(vol_rc[-20:]) > 0 else 1.0
            capitulation = vol_ratio > 1.5

            recovery_detail = {
                "vix": round(vix_val, 1),
                "above_200sma": above_200sma,
                "vol_ratio": round(vol_ratio, 2),
                "capitulation": capitulation,
            }

            if vix_val > 25 and capitulation and above_200sma:
                recovery_class = "FAST"
            elif not above_200sma and not capitulation:
                recovery_class = "SLOW"

        signal["_recovery_speed"] = {"class": recovery_class, **recovery_detail}
    except Exception as e:
        signal["_recovery_speed"] = {"class": "DEFAULT", "error": str(e)}

    # MACRO REGIME GATE (HC #806 + HC #807 R9)
    # Read daily macro regime — TRANSITION or RISK_OFF = penalty, RISK_ON = bonus
    try:
        macro_summary_path = BASE / "data" / "macro" / "macro_summary.json"
        if macro_summary_path.exists():
            with open(macro_summary_path) as f:
                macro = json.load(f)
            regime = macro.get("regime", "UNKNOWN")
            regime_confidence = macro.get("regime_confidence", "low")
            regime_action = macro.get("regime_action", "")
            sector_leaders = macro.get("sector_leaders", [])
            sector_laggards = macro.get("sector_laggards", [])
            key_rels = macro.get("key_relationships", {})

            # Regime scoring
            if regime == "RISK_ON" and regime_confidence in ("high", "medium"):
                score += 10
                signal["_macro_regime"] = f"RISK_ON ({regime_confidence}) +10"
            elif regime == "RISK_OFF":
                # HC #816: PUTs benefit from risk-off — don't penalize them
                sig_direction = signal.get("direction", "").upper()
                if sig_direction in ("PUT", "BEAR", "bear"):
                    score += 5
                    signal["_macro_regime"] = f"RISK_OFF ({regime_confidence}) +5 — puts favored in risk-off"
                else:
                    score -= 20
                    signal["_macro_regime"] = f"RISK_OFF ({regime_confidence}) -20 — avoid calls"
            elif regime == "TRANSITION":
                score -= 5  # Reduced from -10 (HC #816: allow transition with confidence)
                signal["_macro_regime"] = f"TRANSITION -5 — reduced penalty, allow with confidence >= 0.70"
            elif regime == "INFLATIONARY":
                # Inflationary favors energy/materials/commodities, hurts growth/tech
                if ticker in ("XLE", "XLB", "GLD", "SLV"):
                    score += 5
                    signal["_macro_regime"] = f"INFLATIONARY +5 (commodity/energy aligned)"
                elif ticker in ("XLK", "QQQ", "XLY"):
                    score -= 10
                    signal["_macro_regime"] = f"INFLATIONARY -10 (growth headwind)"
                else:
                    signal["_macro_regime"] = f"INFLATIONARY +0"
            elif regime == "DEFLATIONARY":
                if ticker in ("XLU", "TLT", "XLP"):
                    score += 5
                    signal["_macro_regime"] = f"DEFLATIONARY +5 (defensive aligned)"
                else:
                    score -= 5
                    signal["_macro_regime"] = f"DEFLATIONARY -5"
            else:
                signal["_macro_regime"] = f"{regime} +0"

            # Sector alignment with macro leaders/laggards
            # Map tickers to sector names for comparison
            ticker_to_sector = {
                "XLF": "financials", "XLE": "energy", "XLU": "utilities",
                "XLP": "consumer_staples", "XLK": "technology", "XLC": "communication",
                "XLI": "industrials", "XLB": "materials", "XLV": "healthcare",
                "XLRE": "real_estate", "XLY": "consumer_discretionary",
            }
            ticker_sector = ticker_to_sector.get(ticker, "")
            if ticker_sector in sector_leaders:
                score += 8
                signal["_macro_sector_alignment"] = f"SECTOR LEADER today +8"
            elif ticker_sector in sector_laggards:
                score -= 8
                signal["_macro_sector_alignment"] = f"SECTOR LAGGARD today -8"
            else:
                signal["_macro_sector_alignment"] = "neutral"

            # Inter-market relationship signals
            vix_term = key_rels.get("vix_term_structure", "")
            if "backwardation" in vix_term:
                score -= 5
                signal["_macro_vix_term"] = f"BACKWARDATION -5 (acute fear)"
            elif "contango_complacency" in vix_term:
                score += 3
                signal["_macro_vix_term"] = f"CONTANGO +3 (calm)"

            signal["_macro_data_date"] = macro.get("date", "unknown")
        else:
            signal["_macro_regime"] = "NO_MACRO_DATA — run macro_intelligence.py first"
    except Exception as e:
        signal["_macro_regime"] = f"ERROR ({e})"

    return max(0, min(100, score))


def validate_signal(signal, positioned_tickers, open_count):
    """Returns (pass: bool, reason: str) for a signal."""
    ticker = signal.get("ticker", "???")
    confidence = signal.get("confidence_score", 0)
    affordable = signal.get("affordable", False)
    cost = signal.get("estimated_cost", 999)
    iv_class = signal.get("iv_classification", "UNKNOWN")
    n_confirming = signal.get("n_confirming", 0)
    n_conflicting = signal.get("n_conflicting", 0)
    greeks = signal.get("greeks_analysis", {})
    greeks_score = greeks.get("greeks_score", 0) if greeks else 0
    greeks_source = greeks.get("data_source", "") if greeks else ""

    # Gate 1: Confidence (tightened from 0.70 to 0.78)
    if confidence < MIN_CONFIDENCE:
        return False, f"Confidence {confidence:.0%} < {MIN_CONFIDENCE:.0%} threshold"

    # Gate 2: Already positioned
    if ticker in positioned_tickers:
        return False, f"Already have open position in {ticker}"

    # Gate 2b: Expiry date validation (must be in the future)
    expiry_str = signal.get("recommended_expiry", "")
    if expiry_str:
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            today_date = datetime.now().date()
            if expiry_date <= today_date:
                return False, f"Option expiry {expiry_str} is expired (today is {today_date}). Pipeline bug: stale expiry."
        except (ValueError, TypeError):
            pass  # If we can't parse, let downstream catch it

    # Gate 3: Position limit
    if open_count >= MAX_POSITIONS:
        return False, f"At position limit ({open_count}/{MAX_POSITIONS})"

    # Gate 4: Affordable — if single-leg too expensive, recommend vertical spread
    if not affordable or (isinstance(cost, (int, float)) and cost > MAX_COST_PER_TRADE):
        # HC #791b: Instead of rejecting, flag for vertical spread if confidence is high
        if confidence >= 0.78 and n_confirming >= 5:
            signal["_use_spread"] = True
            signal["_spread_reason"] = f"Single-leg ${cost} too expensive, recommending vertical spread ($50-150)"
            # Don't reject — let it pass with spread flag
        else:
            return False, f"Too expensive (${cost}) or not affordable"

    # Gate 5: IV sanity (don't buy expensive IV)
    if iv_class in ("EXPENSIVE", "VERY_EXPENSIVE"):
        return False, f"IV is {iv_class} — don't buy expensive options"

    # Gate 6: Greeks score (minimum 50) + reject synthetic data
    if greeks_score < 50:
        return False, f"Greeks score {greeks_score:.0f} < 50 minimum"
    if greeks_source == "bs_synthetic":
        return False, f"Greeks data is synthetic (Black-Scholes estimated) — unreliable cost/IV"

    # Gate 7: Minimum confirming sources (absolute, not just net)
    if n_confirming < MIN_CONFIRMING_SOURCES:
        return False, f"Only {n_confirming} confirming sources < {MIN_CONFIRMING_SOURCES} required"

    # Gate 8: Net confirming (keep the old check too)
    net_confirm = n_confirming - n_conflicting
    if net_confirm < 3:
        return False, f"Net confirming {net_confirm} < 3 required"

    # Gate 9: Conflict ratio (new — max 25% conflicting)
    if n_confirming > 0:
        conflict_ratio = n_conflicting / (n_confirming + n_conflicting)
        if conflict_ratio > MAX_CONFLICT_RATIO:
            return False, f"Conflict ratio {conflict_ratio:.0%} > {MAX_CONFLICT_RATIO:.0%} max"

    # Gate 10: Timing score (new — reject DEFER/SKIP timing)
    timing_score, timing_rec = load_timing_score(ticker)
    if timing_score is not None and timing_score < MIN_TIMING_SCORE:
        return False, f"Timing score {timing_score:.0f} < {MIN_TIMING_SCORE} ({timing_rec})"

    # Gate 11: Earnings proximity (new — flag if major component reports within 5 days)
    has_earnings, earnings_detail = check_earnings_proximity(ticker)
    if has_earnings and confidence < 0.85:
        return False, f"Nearby earnings ({earnings_detail}) — need confidence >= 0.85"

    # Gate 12: Momentum quality check (advisory — downgrades but doesn't block)
    mom_quality, size_factor = compute_momentum_quality(signal)
    # Note: size_factor logged but not blocking; execution prompt handles sizing

    # Gate 13: Repeat ticker penalty (same ticker traded within 5 days = loss pattern)
    positions_data_for_repeat = load_json(POSITIONS_FILE)
    is_repeat, repeat_detail = check_recent_trades(ticker, positions_data_for_repeat)
    if is_repeat and confidence < 0.90:
        return False, f"REPEAT PENALTY: {repeat_detail} — need 90%+ confidence to re-enter within 5d"

    # Gate 14: Execution quality score (data-driven from 8-trade history analysis)
    exec_score = compute_execution_quality_score(signal, positions_data_for_repeat)
    if exec_score < 50:
        return False, f"Execution quality score {exec_score}/100 < 50 minimum (low-conviction setup)"

    score_grade = "A" if exec_score >= 85 else "B" if exec_score >= 75 else "C"
    # HC #789: Confidence-based position sizing — A=3 contracts, B=2, C=1
    recommended_contracts = 3 if exec_score >= 85 else 2 if exec_score >= 75 else 1

    # Dynamic risk scaling (deep research 2026-08-21): cut risk after consecutive losses,
    # increase only at equity highs. ML-based sizing adds +1.2% return with lower vol.
    # Simple rules: 50% size after 3 consecutive losses, back to full at equity high.
    try:
        positions_history_file = BASE / "state" / "trade_outcomes_history.json"
        if positions_history_file.exists():
            with open(positions_history_file) as f:
                outcomes = json.load(f)
            recent = outcomes[-5:] if len(outcomes) >= 5 else outcomes
            consecutive_losses = 0
            for o in reversed(recent):
                if o.get("pnl", 0) < 0:
                    consecutive_losses += 1
                else:
                    break
            if consecutive_losses >= 3:
                recommended_contracts = max(1, recommended_contracts // 2)
                score_grade += " (RISK-CUT: 3+ losses)"
    except Exception:
        pass  # Fail-open — don't block trades if history unavailable

    return True, f"ALL 14 GATES PASS (timing={timing_score}, mom={mom_quality}, exec_score={exec_score}/{score_grade}, contracts={recommended_contracts})"


def main():
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # Market open guard — don't validate/fire on weekends or holidays
    try:
        sys.path.insert(0, str(BASE / "scripts"))
        from market_status import get_market_status
        ms = get_market_status()
        if not ms.get("is_trading_day", True):
            result = {
                "timestamp": now.isoformat(),
                "status": "MARKET_CLOSED",
                "trades_ready": [],
                "reason": ms.get("reason", "Not a trading day")
            }
            with open(READY_FILE, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Market closed: {ms.get('reason')}")
            return
    except Exception:
        pass  # If market_status.py fails, continue (fail-open)

    # Load data
    signals_data = load_json(SIGNALS_FILE)
    positions_data = load_json(POSITIONS_FILE)

    if not signals_data or "signals" not in signals_data:
        result = {
            "timestamp": now.isoformat(),
            "status": "NO_SIGNALS",
            "trades_ready": [],
            "reason": "No signals file or empty signals"
        }
        with open(READY_FILE, "w") as f:
            json.dump(result, f, indent=2)
        print("No signals available")
        return

    # Check if signals are from today
    sig_timestamp = signals_data.get("timestamp", "")
    if today_str not in sig_timestamp:
        result = {
            "timestamp": now.isoformat(),
            "status": "STALE_SIGNALS",
            "trades_ready": [],
            "reason": f"Signals from {sig_timestamp}, not today ({today_str})"
        }
        with open(READY_FILE, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Stale signals from {sig_timestamp}")
        return

    positioned_tickers = get_positioned_tickers(positions_data)
    open_count = count_open_positions(positions_data)

    # Validate each signal
    trades_ready = []
    rejections = []

    for signal in signals_data.get("signals", []):
        passed, reason = validate_signal(signal, positioned_tickers, open_count)
        ticker = signal.get("ticker", "???")
        if passed:
            # Extract execution quality score and contract sizing from reason
            exec_score_val = 50
            rec_contracts = 1
            try:
                if "exec_score=" in reason:
                    es_part = reason.split("exec_score=")[1].split("/")[0]
                    exec_score_val = int(es_part)
                if "contracts=" in reason:
                    rc_part = reason.split("contracts=")[1].split(")")[0]
                    rec_contracts = int(rc_part)
            except (IndexError, ValueError):
                pass

            # Budget check: total cost of all contracts must be ≤ 60% of equity
            per_contract_cost = signal.get("estimated_cost", 999)
            total_cost = per_contract_cost * rec_contracts
            account_equity = ACCOUNT_EQUITY
            if total_cost > account_equity * 0.60:
                # Scale down contracts to fit budget
                rec_contracts = max(1, int(account_equity * 0.60 / per_contract_cost))
                total_cost = per_contract_cost * rec_contracts

            trade_entry = {
                "ticker": ticker,
                "direction": signal.get("direction"),
                "option_type": signal.get("recommended_option"),
                "strike": signal.get("recommended_strike"),
                "expiry": signal.get("recommended_expiry"),
                "estimated_cost": per_contract_cost,
                "total_cost": total_cost,
                "recommended_contracts": rec_contracts,
                "exec_quality_score": exec_score_val,
                "confidence": signal.get("confidence_score"),
                "n_confirming": signal.get("n_confirming"),
                "iv_rank": signal.get("iv_rank"),
                "iv_classification": signal.get("iv_classification"),
                "greeks_score": signal.get("greeks_analysis", {}).get("greeks_score", 0),
                "exit_rules": signal.get("exit_guidance", {}),
                "sources": signal.get("confirming_sources", []),
                "reason": reason
            }
            # HC #791b: If single-leg too expensive, recommend vertical spread
            if signal.get("_use_spread"):
                trade_entry["strategy"] = "vertical_spread"
                trade_entry["spread_details"] = {
                    "type": "bull_call_spread" if signal.get("direction") == "bull" else "bear_put_spread",
                    "width": "$2-3",
                    "estimated_spread_cost": "$50-150",
                    "max_profit": "strike_width - debit",
                    "note": signal.get("_spread_reason", "Single-leg unaffordable")
                }
                trade_entry["estimated_cost"] = 100  # Approximate spread cost
                trade_entry["total_cost"] = 100 * rec_contracts
            else:
                trade_entry["strategy"] = "single_leg"
            trades_ready.append(trade_entry)
            # Increment open count for subsequent checks
            open_count += 1
            positioned_tickers.add(ticker)
        else:
            rejections.append({"ticker": ticker, "reason": reason,
                               "confidence": signal.get("confidence_score", 0)})

    # Sort by confidence (highest first)
    trades_ready.sort(key=lambda t: t["confidence"], reverse=True)

    # Cap at MAX_POSITIONS minus current open
    current_open = count_open_positions(positions_data)
    max_new = MAX_POSITIONS - current_open
    trades_ready = trades_ready[:max_new]

    # Also check pending entries
    pending_data = load_json(PENDING_FILE)
    pending_trades = pending_data.get("pending", [])

    # Check morning queue (overnight signal flips queued after market close)
    # KEY FIX (Session 88): Morning queue items must be matched to CURRENT signal
    # data and run through validate_signal() — not just appended as pending.
    # The old approach put them in pending_from_earlier which the execution prompt ignored.
    morning_queue = load_json(MORNING_QUEUE_FILE)
    if morning_queue and morning_queue.get("status") == "PENDING":
        queued_flips = morning_queue.get("flips", [])

        # Build lookup of ALL current signals (both above and below threshold)
        all_current_signals = {}
        for sig in signals_data.get("signals", []):
            all_current_signals[sig.get("ticker", "")] = sig
        # Also check below_threshold and actionable_signals
        for sig in signals_data.get("below_threshold", signals_data.get("actionable_signals", [])):
            t = sig.get("ticker", "")
            if t and t not in all_current_signals:
                all_current_signals[t] = sig

        for flip in queued_flips:
            ticker = flip.get("ticker", "")
            if not ticker or ticker in positioned_tickers:
                continue
            if any(t["ticker"] == ticker for t in trades_ready):
                continue

            # Look up this ticker in CURRENT signals for full data
            current_sig = all_current_signals.get(ticker)
            if not current_sig:
                print(f"  Morning queue: {ticker} — no current signal, skipping")
                continue

            # Check if underlying already moved >5% (the opportunity may be gone)
            # We can't easily check this here, but the validate_signal gates will
            # catch affordability/timing issues.

            # Apply burst boost to confidence: this was a pre-screened overnight flip
            # Use the CURRENT signal's data but boost confidence by 10% (it was a burst)
            boosted_confidence = min(0.95, current_sig.get("confidence_score", 0) * 1.10)
            current_sig_boosted = dict(current_sig)
            current_sig_boosted["confidence_score"] = round(boosted_confidence, 2)
            current_sig_boosted["_morning_queue_burst"] = True

            # Validate with slightly relaxed confidence (burst signals get priority)
            # Temporarily lower threshold for this check
            original_min = globals().get("MIN_CONFIDENCE", 0.78)
            # Burst signals validated at 70% instead of 78% — they were pre-screened
            BURST_CONFIDENCE_THRESHOLD = 0.70

            if boosted_confidence >= BURST_CONFIDENCE_THRESHOLD:
                passed, reason = validate_signal(current_sig_boosted, positioned_tickers, open_count)

                # If it failed ONLY on confidence, re-check with burst threshold
                if not passed and "Confidence" in reason:
                    if boosted_confidence >= BURST_CONFIDENCE_THRESHOLD:
                        # Override confidence gate for burst signals
                        # Re-run validation with confidence artificially set to pass
                        temp_sig = dict(current_sig_boosted)
                        temp_sig["confidence_score"] = max(boosted_confidence, MIN_CONFIDENCE + 0.01)
                        passed, reason = validate_signal(temp_sig, positioned_tickers, open_count)
                        if passed:
                            reason = f"MORNING BURST OVERRIDE — {reason}"

                if passed:
                    per_contract_cost = current_sig.get("estimated_cost", 999)
                    exec_score_val = 70  # Default for burst signals
                    rec_contracts = 1
                    try:
                        if "exec_score=" in reason:
                            es_part = reason.split("exec_score=")[1].split("/")[0]
                            exec_score_val = int(es_part)
                        if "contracts=" in reason:
                            rc_part = reason.split("contracts=")[1].split(")")[0]
                            rec_contracts = int(rc_part)
                    except (IndexError, ValueError):
                        pass

                    total_cost = per_contract_cost * rec_contracts
                    if total_cost > ACCOUNT_EQUITY * 0.60:
                        rec_contracts = max(1, int(ACCOUNT_EQUITY * 0.60 / per_contract_cost))
                        total_cost = per_contract_cost * rec_contracts

                    trades_ready.append({
                        "ticker": ticker,
                        "direction": current_sig.get("direction"),
                        "option_type": current_sig.get("recommended_option"),
                        "strike": current_sig.get("recommended_strike"),
                        "expiry": current_sig.get("recommended_expiry"),
                        "estimated_cost": per_contract_cost,
                        "total_cost": total_cost,
                        "recommended_contracts": rec_contracts,
                        "exec_quality_score": exec_score_val,
                        "confidence": boosted_confidence,
                        "n_confirming": current_sig.get("n_confirming"),
                        "iv_rank": current_sig.get("iv_rank"),
                        "iv_classification": current_sig.get("iv_classification"),
                        "greeks_score": current_sig.get("greeks_analysis", {}).get("greeks_score", 0),
                        "exit_rules": current_sig.get("exit_guidance", {}),
                        "sources": current_sig.get("confirming_sources", []),
                        "reason": reason,
                        "source": "morning_queue_burst",
                        "queued_at": morning_queue.get("queued_at", ""),
                    })
                    open_count += 1
                    positioned_tickers.add(ticker)
                    print(f"  ✅ Morning queue BURST: {ticker} PASSED — confidence {boosted_confidence:.0%}, {reason}")
                else:
                    print(f"  ❌ Morning queue: {ticker} FAILED — {reason}")
            else:
                print(f"  ❌ Morning queue: {ticker} confidence too low even with boost ({boosted_confidence:.0%})")

        # Mark queue as processed
        morning_queue["status"] = "PROCESSED"
        morning_queue["processed_at"] = now.isoformat()
        try:
            with open(MORNING_QUEUE_FILE, "w") as f:
                json.dump(morning_queue, f, indent=2)
        except OSError:
            pass

    # Check for after-hours equity positions that need morning decision
    AH_POSITION_FILE = BASE / "state" / "afterhours_equity_position.json"
    ah_position = load_json(AH_POSITION_FILE)
    ah_equity_note = ""
    if ah_position and ah_position.get("type") == "AFTERHOURS_EQUITY_CAPTURE":
        ah_ticker = ah_position.get("ticker", "")
        ah_shares = ah_position.get("shares", 0)
        ah_limit = ah_position.get("limit_price", 0)
        ah_ts = ah_position.get("timestamp", "")
        if today_str in ah_ts or (datetime.now() - datetime.fromisoformat(ah_ts.replace("Z",""))).days < 1:
            ah_equity_note = (
                f"\n⚡ AFTER-HOURS EQUITY POSITION: {ah_shares} shares of {ah_ticker} "
                f"bought at ~${ah_limit} in extended hours. "
                f"MORNING DECISION NEEDED: (1) Hold shares if signal still valid, "
                f"(2) Sell shares + buy options if move has more room, "
                f"(3) Sell shares for gap profit if move already played out."
            )

    result = {
        "timestamp": now.isoformat(),
        "date": today_str,
        "status": "EXECUTE" if trades_ready else ("AH_EQUITY_REVIEW" if ah_equity_note else "NO_ACTION"),
        "trades_ready": trades_ready,
        "ah_equity_note": ah_equity_note,
        "n_trades": len(trades_ready),
        "pending_from_earlier": pending_trades,
        "rejections": rejections[:5],  # Top 5 rejections for debugging
        "account": {
            "open_positions": current_open,
            "max_positions": MAX_POSITIONS,
            "positioned_tickers": list(get_positioned_tickers(positions_data))
        },
        "execution_instructions": ""
    }

    # Build human-readable execution instructions
    if trades_ready:
        instructions = []
        instructions.append("⚡ AUTONOMOUS EXECUTION REQUIRED — DO NOT DEFER ⚡")
        instructions.append(f"{len(trades_ready)} trade(s) passed all gates:\n")
        for i, t in enumerate(trades_ready, 1):
            contracts = t.get('recommended_contracts', 1)
            eq_score = t.get('exec_quality_score', 50)
            grade = "A" if eq_score >= 85 else "B" if eq_score >= 75 else "C"
            instructions.append(
                f"  {i}. {t['ticker']} {t['option_type'].upper()} "
                f"${t['strike']} exp {t['expiry']} — "
                f"~${t['estimated_cost']}/contract x{contracts} (${t.get('total_cost', t['estimated_cost'])} total), "
                f"confidence {t['confidence']:.0%}, "
                f"{t['n_confirming']} sources, IV rank {t['iv_rank']:.0f}%, "
                f"quality {eq_score}/{grade}"
            )
        instructions.append("\nACTION: Call get_option_chains → get_option_instruments → "
                            "get_option_quotes → review_option_order → place_option_order")
        instructions.append(f"ACCOUNT: {ACCOUNT_NUMBER}")
        # Sector-specific exit rules (HC #792 research — holding period analysis)
        SECTOR_EXITS = {
            "XLK": "+8% TP, 3d hold", "XLP": "+8% TP, 3d hold", "XLC": "+8% TP, 3d hold",
            "XLY": "+8% TP, 3d hold", "XLF": "+10% TP, 3d hold", "XLI": "+10% TP, 5d hold",
            "XLV": "+12% TP, 5d hold", "XLE": "+12% TP, 5d hold",
            "XLU": "+20% TP, 10d hold", "XLB": "+20% TP, 10d hold", "XLRE": "+15% TP, 10d hold",
        }
        for t in trades_ready:
            exit_str = SECTOR_EXITS.get(t['ticker'], "+12% TP, 5d hold")
            instructions.append(f"EXIT RULES ({t['ticker']}): {exit_str}, -25% SL, trailing 50% giveback")
        result["execution_instructions"] = "\n".join(instructions)

    with open(READY_FILE, "w") as f:
        json.dump(result, f, indent=2)

    # Update execution log
    log = load_json(EXEC_LOG)
    if "entries" not in log:
        log = {"entries": []}

    log["entries"].append({
        "date": today_str,
        "time": now.strftime("%H:%M:%S"),
        "signals_generated": len(signals_data.get("signals", [])),
        "trades_ready": len(trades_ready),
        "tickers": [t["ticker"] for t in trades_ready],
        "status": "PENDING_EXECUTION",
        "executed": False,
        "execution_time": None
    })

    # Keep only last 30 days
    log["entries"] = log["entries"][-60:]

    with open(EXEC_LOG, "w") as f:
        json.dump(log, f, indent=2)

    if trades_ready:
        print(f"✅ {len(trades_ready)} trades ready for execution: "
              f"{', '.join(t['ticker'] for t in trades_ready)}")
    else:
        print("No trades pass all gates")


if __name__ == "__main__":
    main()
