#!/usr/bin/env python3
"""
Timing Score Backtest
=====================
Tests whether the timing score system at timing_score.py actually improves
trade outcomes vs taking every signal.

Approach:
1. Generate synthetic daily "sector rotation" signals for the sector ETFs
   the agentic system trades (XLF, XLE, XLU, XLK, XLC, XLV, XLI, XLP, XLY, XLB, XLRE)
   over the past 8 months using the same signal logic the aggregator uses
   (momentum, RSI, relative strength).
2. For each signal date, reconstruct what the timing score WOULD HAVE BEEN
   using historical yfinance data (VIX, treasuries, SPY, technicals).
3. Measure the trade outcome: the ETF's return over a 5-day hold period
   (matching the strategy's avg_hold_days of ~2-5 days).
4. Compare groups: All signals, timing >= 60, timing >= 50.
5. Run permutation test for statistical significance.

Also includes the REAL trade log analysis for the 15 closed trades.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add parent path for timing_score imports
sys.path.insert(0, str(Path("/home/jupiter/Lvl3Quant/scripts")))

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance required. pip install yfinance")
    sys.exit(1)

# ─── Configuration ───────────────────────────────────────────────────────────

SECTOR_ETFS = ["XLF", "XLE", "XLU", "XLK", "XLC", "XLV", "XLI", "XLP", "XLY", "XLB", "XLRE"]
HOLD_DAYS = 5  # strategy avg hold
LOOKBACK_MONTHS = 8  # how far back to go
COMMISSION_PCT = 0.005  # ~0.5% round-trip for ETF options (bid-ask + commission)
PERMUTATION_SHUFFLES = 2000

# Timing score thresholds to test
THRESHOLDS = [50, 60, 70]

TRADE_LOG_PATH = Path("/home/jupiter/Lvl3Quant/state/agentic_trade_log.json")

# ─── Data Fetching ───────────────────────────────────────────────────────────

def fetch_all_data():
    """Fetch historical data for all tickers needed."""
    print("Fetching historical data from yfinance...")
    tickers_to_fetch = SECTOR_ETFS + ["SPY", "^VIX", "^VIX3M", "^TNX"]

    data = {}
    for ticker in tickers_to_fetch:
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="1y", auto_adjust=True)
            if not df.empty:
                data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
            else:
                print(f"  {ticker}: NO DATA")
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")

    return data


# ─── Technical Indicators (from timing_score.py) ────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_pctb(series, period=20, num_std=2.0):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = upper - lower
    pct_b = (series - lower) / band_width.replace(0, np.nan)
    return pct_b


# ─── Signal Generation ──────────────────────────────────────────────────────

def generate_signals(data):
    """
    Generate daily sector rotation signals using the same logic as the
    agentic signal aggregator: momentum, RSI, relative strength.

    For each day and each sector ETF, determine if there's a call or put signal.
    """
    spy_df = data.get("SPY")
    if spy_df is None:
        print("ERROR: No SPY data")
        return []

    signals = []

    for ticker in SECTOR_ETFS:
        df = data.get(ticker)
        if df is None or len(df) < 60:
            continue

        close = df["Close"]
        spy_close = spy_df["Close"]

        # Compute indicators
        rsi = compute_rsi(close)
        mom_5d = close.pct_change(5) * 100
        mom_21d = close.pct_change(21) * 100

        # Relative strength vs SPY (align indices)
        common_idx = close.index.intersection(spy_close.index)
        ticker_aligned = close.reindex(common_idx)
        spy_aligned = spy_close.reindex(common_idx)
        rs_5d = ticker_aligned.pct_change(5) * 100 - spy_aligned.pct_change(5) * 100
        rs_21d = ticker_aligned.pct_change(21) * 100 - spy_aligned.pct_change(21) * 100

        # Generate signals starting from day 50 (need lookback)
        for i in range(50, len(common_idx)):
            dt = common_idx[i]

            r = rsi.reindex(common_idx).iloc[i] if dt in rsi.index else 50
            m5 = mom_5d.reindex(common_idx).iloc[i] if dt in mom_5d.index else 0
            m21 = mom_21d.reindex(common_idx).iloc[i] if dt in mom_21d.index else 0
            r5 = rs_5d.iloc[i] if i < len(rs_5d) else 0
            r21 = rs_21d.iloc[i] if i < len(rs_21d) else 0

            if pd.isna(r) or pd.isna(m5) or pd.isna(m21):
                continue

            # Signal logic matching the aggregator:
            # Bullish: positive momentum + RSI not overbought + positive relative strength
            # Bearish: negative momentum + RSI not oversold + negative relative strength

            bull_score = 0
            bear_score = 0

            # Momentum signals
            if m21 > 2: bull_score += 1
            if m21 < -2: bear_score += 1
            if m5 > 1: bull_score += 1
            if m5 < -1: bear_score += 1

            # RSI signals
            if 40 < r < 65: bull_score += 1  # not overbought, room to run
            if r > 65: bear_score += 0.5     # overbought -> put signal
            if r < 35: bull_score += 0.5     # oversold -> contrarian call
            if 35 < r < 55: bear_score += 0.5

            # Relative strength
            if r5 > 1: bull_score += 1
            if r5 < -1: bear_score += 1
            if r21 > 2: bull_score += 1
            if r21 < -2: bear_score += 1

            # Need at least 2 confirming factors
            if bull_score >= 2 and bull_score > bear_score:
                signals.append({
                    "date": dt,
                    "ticker": ticker,
                    "direction": "call",
                    "confidence": min(bull_score / 5, 1.0),
                    "rsi": r,
                    "mom_5d": m5,
                    "mom_21d": m21,
                    "rs_5d": r5,
                    "rs_21d": r21,
                })
            elif bear_score >= 2 and bear_score > bull_score:
                signals.append({
                    "date": dt,
                    "ticker": ticker,
                    "direction": "put",
                    "confidence": min(bear_score / 5, 1.0),
                    "rsi": r,
                    "mom_5d": m5,
                    "mom_21d": m21,
                    "rs_5d": r5,
                    "rs_21d": r21,
                })

    print(f"Generated {len(signals)} signals across {len(SECTOR_ETFS)} ETFs")
    return signals


# ─── Timing Score Reconstruction ─────────────────────────────────────────────

def reconstruct_timing_score(signal, data):
    """
    Reconstruct the timing score for a signal on a historical date,
    using the same logic as timing_score.py but with historical data.
    """
    dt = signal["date"]
    ticker = signal["ticker"]
    direction = signal["direction"]
    is_call = direction == "call"

    ticker_df = data.get(ticker)
    spy_df = data.get("SPY")
    vix_df = data.get("^VIX")
    vix3m_df = data.get("^VIX3M")
    tnx_df = data.get("^TNX")

    # ─── MACRO SCORE (35% of total per HC #782 weights) ───
    macro_scores = {}

    # VIX level + trend (35% of macro)
    if vix_df is not None and len(vix_df) > 0:
        vix_close = vix_df["Close"]
        vix_at = vix_close[vix_close.index <= dt]
        if len(vix_at) == 0:
            macro_scores["vix"] = 50
        else:
            vix = float(vix_at.iloc[-1])
            vix_5d = float(vix_at.iloc[-5]) if len(vix_at) >= 5 else None

            if vix < 13:
                vix_env = 90 if is_call else 30
            elif vix < 15:
                vix_env = 75 if is_call else 40
            elif vix < 18:
                vix_env = 55
            elif vix < 22:
                vix_env = 35 if is_call else 70
            else:
                vix_env = 20 if is_call else 85

            if vix_5d is not None:
                vix_change = vix - vix_5d
                if vix_change < -1.5:
                    vix_env += 10 if is_call else -10
                elif vix_change > 1.5:
                    vix_env += -10 if is_call else 10

            macro_scores["vix"] = max(0, min(100, vix_env))
    else:
        macro_scores["vix"] = 50

    # Term structure (20% of macro)
    if vix_df is not None and vix3m_df is not None:
        try:
            # Find closest available dates
            vix_close = vix_df["Close"]
            vix3m_close = vix3m_df["Close"]
            # Use last available before dt
            vix_val = vix_close[vix_close.index <= dt].iloc[-1] if len(vix_close[vix_close.index <= dt]) > 0 else None
            vix3m_val = vix3m_close[vix3m_close.index <= dt].iloc[-1] if len(vix3m_close[vix3m_close.index <= dt]) > 0 else None

            if vix_val is not None and vix3m_val is not None:
                if vix_val < vix3m_val:
                    macro_scores["term"] = 70  # contango
                else:
                    macro_scores["term"] = 30 if is_call else 65  # backwardation
            else:
                macro_scores["term"] = 50
        except:
            macro_scores["term"] = 50
    else:
        macro_scores["term"] = 50

    # Treasury (20% of macro)
    RATE_SENS = {
        "XLF": "positive", "XLE": "neutral", "XLU": "negative",
        "XLK": "negative", "XLC": "neutral", "XLV": "neutral",
        "XLI": "neutral", "XLP": "neutral", "XLY": "negative",
        "XLB": "neutral", "XLRE": "negative",
    }

    if tnx_df is not None:
        try:
            tnx_close = tnx_df["Close"]
            tnx_at = tnx_close[tnx_close.index <= dt]
            if len(tnx_at) >= 5:
                yield_change = float(tnx_at.iloc[-1] - tnx_at.iloc[-5])
                rate_sens = RATE_SENS.get(ticker, "neutral")

                if yield_change > 0.05:
                    yield_trend = "rising"
                elif yield_change < -0.05:
                    yield_trend = "falling"
                else:
                    yield_trend = "flat"

                if rate_sens == "positive":
                    if yield_trend == "rising":
                        macro_scores["treasury"] = 80 if is_call else 30
                    elif yield_trend == "falling":
                        macro_scores["treasury"] = 30 if is_call else 70
                    else:
                        macro_scores["treasury"] = 55
                elif rate_sens == "negative":
                    if yield_trend == "rising":
                        macro_scores["treasury"] = 30 if is_call else 75
                    elif yield_trend == "falling":
                        macro_scores["treasury"] = 75 if is_call else 30
                    else:
                        macro_scores["treasury"] = 55
                else:
                    macro_scores["treasury"] = 55
            else:
                macro_scores["treasury"] = 55
        except:
            macro_scores["treasury"] = 55
    else:
        macro_scores["treasury"] = 55

    # SPY regime (25% of macro)
    if spy_df is not None:
        try:
            spy_close = spy_df["Close"]
            spy_at = spy_close[spy_close.index <= dt]
            if len(spy_at) >= 50:
                price = float(spy_at.iloc[-1])
                sma20 = float(spy_at.tail(20).mean())
                sma50 = float(spy_at.tail(50).mean())
                above_20 = price > sma20
                above_50 = price > sma50
                if above_20 and above_50:
                    regime = "bull"
                elif above_50 and not above_20:
                    regime = "pullback"
                elif not above_50 and above_20:
                    regime = "recovery"
                else:
                    regime = "bear"

                if regime == "bull":
                    macro_scores["spy"] = 80 if is_call else 25
                elif regime == "pullback":
                    macro_scores["spy"] = 50
                elif regime == "recovery":
                    macro_scores["spy"] = 60 if is_call else 45
                else:
                    macro_scores["spy"] = 20 if is_call else 80
            else:
                macro_scores["spy"] = 50
        except:
            macro_scores["spy"] = 50
    else:
        macro_scores["spy"] = 50

    macro_total = (
        macro_scores.get("vix", 50) * 0.35
        + macro_scores.get("term", 50) * 0.20
        + macro_scores.get("treasury", 55) * 0.20
        + macro_scores.get("spy", 50) * 0.25
    )

    # ─── TECHNICAL SCORE (25% of total per HC #782) ───
    tech_scores = {}

    if ticker_df is not None:
        try:
            close = ticker_df["Close"]
            close_at = close[close.index <= dt]

            if len(close_at) >= 26:
                # RSI (30% of tech)
                rsi_series = compute_rsi(close_at)
                rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50

                if is_call:
                    if rsi > 75: tech_scores["rsi"] = 10
                    elif rsi > 70: tech_scores["rsi"] = 20
                    elif rsi > 60: tech_scores["rsi"] = 55
                    elif rsi > 45: tech_scores["rsi"] = 80
                    elif rsi > 35: tech_scores["rsi"] = 65
                    else: tech_scores["rsi"] = 40
                else:
                    if rsi < 25: tech_scores["rsi"] = 10
                    elif rsi < 30: tech_scores["rsi"] = 20
                    elif rsi < 40: tech_scores["rsi"] = 55
                    elif rsi < 55: tech_scores["rsi"] = 70
                    elif rsi < 65: tech_scores["rsi"] = 80
                    else: tech_scores["rsi"] = 85

                # MACD (25% of tech)
                _, _, hist = compute_macd(close_at)
                h_val = float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else 0
                h_prev = float(hist.iloc[-2]) if len(hist) >= 2 and not pd.isna(hist.iloc[-2]) else 0
                accelerating = h_val > h_prev

                if is_call:
                    if h_val > 0 and accelerating: tech_scores["macd"] = 90
                    elif h_val > 0: tech_scores["macd"] = 65
                    elif h_val < 0 and not accelerating: tech_scores["macd"] = 25
                    else: tech_scores["macd"] = 45
                else:
                    if h_val < 0 and not accelerating: tech_scores["macd"] = 85
                    elif h_val < 0: tech_scores["macd"] = 60
                    elif h_val > 0 and accelerating: tech_scores["macd"] = 20
                    else: tech_scores["macd"] = 45

                # Bollinger (25% of tech)
                pctb_series = compute_bollinger_pctb(close_at)
                pctb = float(pctb_series.iloc[-1]) if not pd.isna(pctb_series.iloc[-1]) else 0.5

                if is_call:
                    if pctb < 0.2: tech_scores["bb"] = 85
                    elif pctb < 0.4: tech_scores["bb"] = 75
                    elif pctb < 0.6: tech_scores["bb"] = 65
                    elif pctb < 0.8: tech_scores["bb"] = 45
                    else: tech_scores["bb"] = 25
                else:
                    if pctb > 0.8: tech_scores["bb"] = 85
                    elif pctb > 0.6: tech_scores["bb"] = 75
                    elif pctb > 0.4: tech_scores["bb"] = 60
                    elif pctb > 0.2: tech_scores["bb"] = 40
                    else: tech_scores["bb"] = 20

                # Momentum (20% of tech)
                if len(close_at) >= 6:
                    mom5d = float((close_at.iloc[-1] / close_at.iloc[-5] - 1) * 100)
                else:
                    mom5d = 0

                if is_call:
                    if mom5d > 5: tech_scores["mom"] = 40
                    elif mom5d > 2: tech_scores["mom"] = 80
                    elif mom5d > 0: tech_scores["mom"] = 70
                    elif mom5d > -2: tech_scores["mom"] = 55
                    else: tech_scores["mom"] = 30
                else:
                    if mom5d < -5: tech_scores["mom"] = 40
                    elif mom5d < -2: tech_scores["mom"] = 80
                    elif mom5d < 0: tech_scores["mom"] = 70
                    elif mom5d < 2: tech_scores["mom"] = 55
                    else: tech_scores["mom"] = 30
        except:
            pass

    tech_total = (
        tech_scores.get("rsi", 50) * 0.30
        + tech_scores.get("macd", 50) * 0.25
        + tech_scores.get("bb", 50) * 0.25
        + tech_scores.get("mom", 50) * 0.20
    )

    # ─── MARKET STRUCTURE SCORE (40% of total per HC #782) ───
    struct_scores = {}

    if ticker_df is not None and spy_df is not None:
        try:
            close = ticker_df["Close"]
            spy_close = spy_df["Close"]
            close_at = close[close.index <= dt]
            spy_at = spy_close[spy_close.index <= dt]

            if len(close_at) >= 21 and len(spy_at) >= 21:
                # Relative strength at multiple timeframes
                rs_vals = []
                for days in [5, 10, 21]:
                    if len(close_at) >= days and len(spy_at) >= days:
                        t_ret = (close_at.iloc[-1] / close_at.iloc[-days] - 1) * 100
                        s_ret = (spy_at.iloc[-1] / spy_at.iloc[-days] - 1) * 100
                        rs_vals.append(t_ret - s_ret)

                if rs_vals:
                    avg_rs = np.mean(rs_vals)
                    if is_call:
                        if avg_rs > 3: struct_scores["rs"] = 90
                        elif avg_rs > 1: struct_scores["rs"] = 75
                        elif avg_rs > -1: struct_scores["rs"] = 55
                        elif avg_rs > -3: struct_scores["rs"] = 35
                        else: struct_scores["rs"] = 15
                    else:
                        if avg_rs < -3: struct_scores["rs"] = 90
                        elif avg_rs < -1: struct_scores["rs"] = 75
                        elif avg_rs < 1: struct_scores["rs"] = 55
                        elif avg_rs < 3: struct_scores["rs"] = 35
                        else: struct_scores["rs"] = 15

                # Volume profile
                if "Volume" in ticker_df.columns:
                    vol_at = ticker_df["Volume"][ticker_df.index <= dt]
                    if len(vol_at) >= 20:
                        avg_vol = vol_at.tail(20).mean()
                        last_vol = vol_at.iloc[-1]
                        ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
                        if ratio > 1.5: struct_scores["vol"] = 80
                        elif ratio > 1.0: struct_scores["vol"] = 65
                        elif ratio > 0.7: struct_scores["vol"] = 45
                        else: struct_scores["vol"] = 30

                # Multi-timeframe consistency
                if rs_vals and len(rs_vals) >= 3:
                    if is_call:
                        agree = sum(1 for r in rs_vals if r > 0)
                    else:
                        agree = sum(1 for r in rs_vals if r < 0)
                    if agree == 3: struct_scores["mtf"] = 90
                    elif agree == 2: struct_scores["mtf"] = 60
                    elif agree == 1: struct_scores["mtf"] = 35
                    else: struct_scores["mtf"] = 15

                # Sector flow (we don't have this historically, use neutral)
                struct_scores["flow"] = 50
        except:
            pass

    struct_total = (
        struct_scores.get("rs", 50) * 0.35
        + struct_scores.get("vol", 50) * 0.25
        + struct_scores.get("flow", 50) * 0.25
        + struct_scores.get("mtf", 50) * 0.15
    )

    # ─── TOTAL SCORE (HC #782 weights: structure 40%, macro 35%, tech 25%) ───
    total = macro_total * 0.35 + tech_total * 0.25 + struct_total * 0.40

    return round(total, 1), {
        "macro": round(macro_total, 1),
        "technical": round(tech_total, 1),
        "structure": round(struct_total, 1),
    }


# ─── Outcome Measurement ────────────────────────────────────────────────────

def measure_outcomes(signals, data):
    """
    For each signal, measure the ETF return over the hold period.
    For calls: return = (exit - entry) / entry
    For puts: return = (entry - exit) / entry
    """
    results = []

    for sig in signals:
        ticker = sig["ticker"]
        dt = sig["date"]
        direction = sig["direction"]

        df = data.get(ticker)
        if df is None:
            continue

        close = df["Close"]

        # Find entry date and exit date
        try:
            entry_idx = close.index.get_loc(dt)
        except KeyError:
            continue

        exit_idx = entry_idx + HOLD_DAYS
        if exit_idx >= len(close):
            continue

        entry_price = float(close.iloc[entry_idx])
        exit_price = float(close.iloc[exit_idx])

        # Also compute MFE/MAE over hold period
        hold_prices = close.iloc[entry_idx:exit_idx + 1]

        if direction == "call":
            raw_return = (exit_price - entry_price) / entry_price
            mfe = float((hold_prices.max() - entry_price) / entry_price)
            mae = float((hold_prices.min() - entry_price) / entry_price)
        else:
            raw_return = (entry_price - exit_price) / entry_price
            mfe = float((entry_price - hold_prices.min()) / entry_price)
            mae = float((entry_price - hold_prices.max()) / entry_price)

        # Net return after commission
        net_return = raw_return - COMMISSION_PCT

        sig["raw_return"] = raw_return
        sig["net_return"] = net_return
        sig["mfe"] = mfe
        sig["mae"] = mae
        sig["win"] = 1 if net_return > 0 else 0

        results.append(sig)

    return results


# ─── Performance Metrics ─────────────────────────────────────────────────────

def compute_metrics(returns, label=""):
    """Compute Sharpe, Sortino, WR, PF, avg return."""
    if len(returns) == 0:
        return {"label": label, "n": 0, "sharpe": 0, "sortino": 0, "wr": 0, "pf": 0, "avg_ret": 0}

    arr = np.array(returns)
    n = len(arr)
    avg = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 1e-9

    # Sharpe (annualized assuming ~252/5 = ~50 trades/year per ticker)
    sharpe = (avg / std) * np.sqrt(50) if std > 1e-9 else 0

    # Sortino
    downside = arr[arr < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg / downside_std) * np.sqrt(50) if downside_std > 1e-9 else 0

    # Win rate
    wins = np.sum(arr > 0)
    wr = wins / n * 100

    # Profit factor
    gross_profit = np.sum(arr[arr > 0])
    gross_loss = abs(np.sum(arr[arr < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "label": label,
        "n": n,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "avg_ret": round(avg * 100, 3),  # as percentage
        "std_ret": round(std * 100, 3),
        "total_ret": round(np.sum(arr) * 100, 2),
    }


# ─── Permutation Test ───────────────────────────────────────────────────────

def permutation_test(all_returns, filtered_returns, n_perms=PERMUTATION_SHUFFLES):
    """
    Test whether the filtered group's mean return is significantly different
    from a random subset of the same size drawn from all returns.
    """
    if len(filtered_returns) == 0 or len(all_returns) == 0:
        return {"p_value": 1.0, "significant": False}

    observed_diff = np.mean(filtered_returns) - np.mean(all_returns)
    n_filtered = len(filtered_returns)
    all_arr = np.array(all_returns)

    count_extreme = 0
    rng = np.random.default_rng(42)

    for _ in range(n_perms):
        perm_idx = rng.choice(len(all_arr), size=n_filtered, replace=False)
        perm_mean = np.mean(all_arr[perm_idx])
        if perm_mean - np.mean(all_arr) >= observed_diff:
            count_extreme += 1

    p_value = count_extreme / n_perms

    return {
        "observed_diff_pct": round(observed_diff * 100, 4),
        "p_value": round(p_value, 4),
        "significant_5pct": p_value < 0.05,
        "significant_10pct": p_value < 0.10,
        "n_permutations": n_perms,
    }


# ─── Real Trade Log Analysis ────────────────────────────────────────────────

def analyze_real_trades(data):
    """Analyze our actual closed trades and compute what timing scores they would have had."""
    print("\n" + "=" * 70)
    print("SECTION 1: REAL TRADE LOG ANALYSIS")
    print("=" * 70)

    try:
        with open(TRADE_LOG_PATH) as f:
            trade_log = json.load(f)
    except:
        print("Could not load trade log")
        return

    trades = trade_log.get("trades", [])
    closed = [t for t in trades if t.get("status") in ("WIN", "LOSS")]

    print(f"\nClosed trades: {len(closed)} (W:{sum(1 for t in closed if t['status']=='WIN')}, "
          f"L:{sum(1 for t in closed if t['status']=='LOSS')})")

    scored_trades = []
    for t in closed:
        ticker = t["ticker"]
        direction = t.get("direction", "call")
        entry_date_str = t.get("entry_date")

        if not entry_date_str:
            continue

        # Create a pseudo-signal for timing score calculation
        try:
            entry_dt = pd.Timestamp(entry_date_str)
        except:
            continue

        # Find closest trading date in data
        ticker_data = data.get(ticker)
        if ticker_data is None:
            # Try to fetch it
            try:
                t_obj = yf.Ticker(ticker)
                ticker_data = t_obj.history(period="1y", auto_adjust=True)
                if not ticker_data.empty:
                    data[ticker] = ticker_data
            except:
                pass

        if ticker_data is None or ticker_data.empty:
            continue

        # Find the closest available date in the data index
        available_dates = ticker_data.index
        # Convert entry_dt to tz-aware if needed
        if available_dates.tz is not None:
            entry_dt_tz = entry_dt.tz_localize(available_dates.tz)
        else:
            entry_dt_tz = entry_dt

        # Find nearest date on or before entry_dt
        mask = available_dates <= entry_dt_tz
        if mask.any():
            closest_dt = available_dates[mask][-1]
        elif len(available_dates) > 0:
            closest_dt = available_dates[0]
        else:
            continue

        signal = {"date": closest_dt, "ticker": ticker, "direction": direction}
        score, layers = reconstruct_timing_score(signal, data)

        outcome = t.get("outcome_detail", {})
        pct_move = outcome.get("pct_move", t.get("exit_pnl", 0))
        mfe = outcome.get("max_favorable_pct", 0)

        scored_trades.append({
            "ticker": ticker,
            "direction": direction,
            "entry_date": entry_date_str,
            "status": t["status"],
            "pct_move": pct_move,
            "mfe": mfe,
            "timing_score": score,
            "macro": layers["macro"],
            "technical": layers["technical"],
            "structure": layers["structure"],
        })

    if not scored_trades:
        print("No trades could be scored")
        return

    print(f"\nScored {len(scored_trades)} trades with reconstructed timing scores:\n")
    print(f"{'Ticker':<8} {'Dir':<5} {'Date':<12} {'Result':<6} {'Move%':>7} {'MFE%':>7} {'Score':>6} "
          f"{'Macro':>6} {'Tech':>6} {'Struct':>6}")
    print("-" * 80)

    for st in scored_trades:
        print(f"{st['ticker']:<8} {st['direction']:<5} {st['entry_date']:<12} {st['status']:<6} "
              f"{st['pct_move']:>7.2f} {st['mfe']:>7.2f} {st['timing_score']:>6.1f} "
              f"{st['macro']:>6.1f} {st['technical']:>6.1f} {st['structure']:>6.1f}")

    # Split by score threshold
    for threshold in THRESHOLDS:
        above = [t for t in scored_trades if t["timing_score"] >= threshold]
        below = [t for t in scored_trades if t["timing_score"] < threshold]
        above_wr = sum(1 for t in above if t["status"] == "WIN") / len(above) * 100 if above else 0
        below_wr = sum(1 for t in below if t["status"] == "WIN") / len(below) * 100 if below else 0
        print(f"\n  Score >= {threshold}: {len(above)} trades, WR={above_wr:.0f}%")
        print(f"  Score <  {threshold}: {len(below)} trades, WR={below_wr:.0f}%")


# ─── Main Backtest ───────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 70)
    print("TIMING SCORE BACKTEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Hold period: {HOLD_DAYS} days | Commission: {COMMISSION_PCT*100:.1f}%")
    print(f"Permutation shuffles: {PERMUTATION_SHUFFLES}")
    print("=" * 70)

    # Fetch data
    data = fetch_all_data()

    if "SPY" not in data:
        print("FATAL: Could not fetch SPY data")
        return

    # Part 1: Analyze real trades
    analyze_real_trades(data)

    # Part 2: Full synthetic backtest
    print("\n\n" + "=" * 70)
    print("SECTION 2: SYNTHETIC SIGNAL BACKTEST (8 MONTHS)")
    print("=" * 70)

    # Generate signals
    signals = generate_signals(data)

    if len(signals) == 0:
        print("No signals generated!")
        return

    # Compute timing scores
    print("\nComputing timing scores for all signals...")
    for i, sig in enumerate(signals):
        score, layers = reconstruct_timing_score(sig, data)
        sig["timing_score"] = score
        sig["macro_score"] = layers["macro"]
        sig["tech_score"] = layers["technical"]
        sig["struct_score"] = layers["structure"]
        if (i + 1) % 200 == 0:
            print(f"  Scored {i+1}/{len(signals)}")

    print(f"  Scored all {len(signals)} signals")

    # Measure outcomes
    print("\nMeasuring trade outcomes...")
    results = measure_outcomes(signals, data)
    print(f"  {len(results)} signals with measurable outcomes")

    if len(results) == 0:
        print("No measurable outcomes!")
        return

    # Score distribution
    scores = [r["timing_score"] for r in results]
    print(f"\nTiming Score Distribution:")
    print(f"  Mean: {np.mean(scores):.1f}")
    print(f"  Median: {np.median(scores):.1f}")
    print(f"  Std: {np.std(scores):.1f}")
    print(f"  Min: {np.min(scores):.1f} | Max: {np.max(scores):.1f}")

    # Quintile analysis
    pctiles = np.percentile(scores, [20, 40, 60, 80])
    print(f"  P20={pctiles[0]:.1f} P40={pctiles[1]:.1f} P60={pctiles[2]:.1f} P80={pctiles[3]:.1f}")

    # ─── Group comparison ───
    print("\n" + "=" * 70)
    print("RESULTS: PERFORMANCE BY TIMING SCORE THRESHOLD")
    print("=" * 70)

    all_returns = np.array([r["net_return"] for r in results])

    # All signals (baseline)
    m_all = compute_metrics(all_returns, "ALL SIGNALS (baseline)")
    print(f"\n{'Group':<30} {'N':>5} {'AvgRet%':>8} {'WR%':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6}")
    print("-" * 75)
    print(f"{m_all['label']:<30} {m_all['n']:>5} {m_all['avg_ret']:>8.3f} {m_all['wr']:>6.1f} "
          f"{m_all['sharpe']:>7.2f} {m_all['sortino']:>8.2f} {m_all['pf']:>6.2f}")

    perm_results = {}

    for threshold in THRESHOLDS:
        filtered = [r for r in results if r["timing_score"] >= threshold]
        filtered_returns = np.array([r["net_return"] for r in filtered])

        m = compute_metrics(filtered_returns, f"Score >= {threshold}")
        print(f"{m['label']:<30} {m['n']:>5} {m['avg_ret']:>8.3f} {m['wr']:>6.1f} "
              f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['pf']:>6.2f}")

        # Also show the rejected signals
        rejected = [r for r in results if r["timing_score"] < threshold]
        rej_returns = np.array([r["net_return"] for r in rejected])
        m_rej = compute_metrics(rej_returns, f"Score < {threshold} (rejected)")
        print(f"{m_rej['label']:<30} {m_rej['n']:>5} {m_rej['avg_ret']:>8.3f} {m_rej['wr']:>6.1f} "
              f"{m_rej['sharpe']:>7.2f} {m_rej['sortino']:>8.2f} {m_rej['pf']:>6.2f}")

        # Permutation test
        if len(filtered_returns) > 0:
            perm = permutation_test(all_returns, filtered_returns)
            perm_results[threshold] = perm

    # ─── Call vs Put breakdown ───
    print("\n" + "=" * 70)
    print("RESULTS: CALLS vs PUTS")
    print("=" * 70)

    for direction in ["call", "put"]:
        dir_results = [r for r in results if r["direction"] == direction]
        if not dir_results:
            continue

        dir_returns = np.array([r["net_return"] for r in dir_results])
        m_dir = compute_metrics(dir_returns, f"All {direction.upper()}s")
        print(f"\n{m_dir['label']:<30} {m_dir['n']:>5} {m_dir['avg_ret']:>8.3f} {m_dir['wr']:>6.1f} "
              f"{m_dir['sharpe']:>7.2f} {m_dir['sortino']:>8.2f} {m_dir['pf']:>6.2f}")

        for threshold in [50, 60]:
            filtered = [r for r in dir_results if r["timing_score"] >= threshold]
            if not filtered:
                continue
            f_returns = np.array([r["net_return"] for r in filtered])
            m_f = compute_metrics(f_returns, f"  {direction.upper()} score>={threshold}")
            print(f"{m_f['label']:<30} {m_f['n']:>5} {m_f['avg_ret']:>8.3f} {m_f['wr']:>6.1f} "
                  f"{m_f['sharpe']:>7.2f} {m_f['sortino']:>8.2f} {m_f['pf']:>6.2f}")

    # ─── Quintile analysis ───
    print("\n" + "=" * 70)
    print("RESULTS: QUINTILE ANALYSIS")
    print("=" * 70)

    scores_arr = np.array([r["timing_score"] for r in results])
    returns_arr = np.array([r["net_return"] for r in results])

    quintile_edges = np.percentile(scores_arr, [0, 20, 40, 60, 80, 100])
    print(f"\n{'Quintile':<15} {'Score Range':<18} {'N':>5} {'AvgRet%':>8} {'WR%':>6} {'Sharpe':>7} {'PF':>6}")
    print("-" * 65)

    for q in range(5):
        lo = quintile_edges[q]
        hi = quintile_edges[q + 1]
        if q == 4:
            mask = (scores_arr >= lo) & (scores_arr <= hi)
        else:
            mask = (scores_arr >= lo) & (scores_arr < hi)

        q_returns = returns_arr[mask]
        m_q = compute_metrics(q_returns, f"Q{q+1}")
        print(f"Q{q+1} ({'worst' if q==0 else 'best' if q==4 else '     '}){'':<5} "
              f"[{lo:5.1f}-{hi:5.1f}]{'':>5} {m_q['n']:>5} {m_q['avg_ret']:>8.3f} "
              f"{m_q['wr']:>6.1f} {m_q['sharpe']:>7.2f} {m_q['pf']:>6.2f}")

    # ─── Layer decomposition ───
    print("\n" + "=" * 70)
    print("RESULTS: WHICH LAYER MATTERS MOST?")
    print("=" * 70)

    for layer_name, layer_key in [("Macro", "macro_score"), ("Technical", "tech_score"), ("Structure", "struct_score")]:
        layer_scores = np.array([r[layer_key] for r in results])
        median_layer = np.median(layer_scores)

        above_med = returns_arr[layer_scores >= median_layer]
        below_med = returns_arr[layer_scores < median_layer]

        m_above = compute_metrics(above_med, f"{layer_name} >= median")
        m_below = compute_metrics(below_med, f"{layer_name} < median")

        print(f"\n{layer_name} (median={median_layer:.1f}):")
        print(f"  Above: N={m_above['n']}, AvgRet={m_above['avg_ret']:.3f}%, WR={m_above['wr']:.1f}%, "
              f"Sharpe={m_above['sharpe']:.2f}, PF={m_above['pf']:.2f}")
        print(f"  Below: N={m_below['n']}, AvgRet={m_below['avg_ret']:.3f}%, WR={m_below['wr']:.1f}%, "
              f"Sharpe={m_below['sharpe']:.2f}, PF={m_below['pf']:.2f}")

        diff = m_above['avg_ret'] - m_below['avg_ret']
        print(f"  Diff: {'+' if diff > 0 else ''}{diff:.3f}%")

    # ─── Permutation test results ───
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (Permutation Tests)")
    print("=" * 70)

    for threshold, perm in perm_results.items():
        if "significant_5pct" not in perm:
            print(f"\n  Score >= {threshold}: No signals (skipped)")
            continue
        sig_marker = "***" if perm["significant_5pct"] else ("*" if perm["significant_10pct"] else "n.s.")
        print(f"\n  Score >= {threshold} vs All Signals:")
        print(f"    Observed mean return diff: {perm['observed_diff_pct']:+.4f}%")
        print(f"    p-value: {perm['p_value']:.4f} {sig_marker}")
        print(f"    Significant at 5%: {'YES' if perm['significant_5pct'] else 'NO'}")
        print(f"    Significant at 10%: {'YES' if perm['significant_10pct'] else 'NO'}")

    # ─── Correlation analysis ───
    print("\n" + "=" * 70)
    print("CORRELATION: TIMING SCORE vs RETURNS")
    print("=" * 70)

    corr = np.corrcoef(scores_arr, returns_arr)[0, 1]
    print(f"\n  Pearson correlation: {corr:.4f}")

    # Rank correlation (Spearman)
    from scipy import stats as scipy_stats
    spearman_r, spearman_p = scipy_stats.spearmanr(scores_arr, returns_arr)
    print(f"  Spearman rank correlation: {spearman_r:.4f} (p={spearman_p:.4f})")

    # ─── VERDICT ───
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Determine verdict based on results
    best_threshold = None
    best_sharpe_improvement = 0

    for threshold in THRESHOLDS:
        filtered = [r for r in results if r["timing_score"] >= threshold]
        if len(filtered) < 20:
            continue
        f_returns = np.array([r["net_return"] for r in filtered])
        m_f = compute_metrics(f_returns)
        improvement = m_f["sharpe"] - m_all["sharpe"]
        if improvement > best_sharpe_improvement:
            best_sharpe_improvement = improvement
            best_threshold = threshold

    any_significant = any(p["significant_10pct"] for p in perm_results.values())

    if corr > 0.05 and best_sharpe_improvement > 0.3 and any_significant:
        print(f"\n  TIMING SCORE ADDS VALUE.")
        print(f"  Best threshold: >= {best_threshold}")
        print(f"  Sharpe improvement: +{best_sharpe_improvement:.2f}")
        print(f"  Statistically significant: YES")
        print(f"  Recommendation: Use timing score >= {best_threshold} as filter.")
    elif corr > 0.02 and best_sharpe_improvement > 0:
        print(f"\n  TIMING SCORE SHOWS WEAK POSITIVE EFFECT.")
        print(f"  Best threshold: >= {best_threshold}")
        print(f"  Sharpe improvement: +{best_sharpe_improvement:.2f}")
        print(f"  Statistically significant: {'YES' if any_significant else 'NO'}")
        print(f"  Recommendation: Marginal benefit. Use as tiebreaker, not hard filter.")
    else:
        print(f"\n  TIMING SCORE DOES NOT MEANINGFULLY IMPROVE RESULTS.")
        print(f"  Correlation with returns: {corr:.4f}")
        print(f"  Best Sharpe improvement: {best_sharpe_improvement:+.2f}")
        print(f"  Recommendation: Do not use as hard filter. May have value for")
        print(f"  position sizing (scale down on low scores) rather than binary skip/enter.")

    print(f"\n  Key caveat: This backtest reconstructs timing scores from historical")
    print(f"  data. The 'sector flow' component of the market structure layer is")
    print(f"  set to neutral (50) since we don't have historical agentic signal")
    print(f"  data. In live use, that component would vary and could change results.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_backtest()
