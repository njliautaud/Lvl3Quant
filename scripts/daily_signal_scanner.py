#!/usr/bin/env python3
"""
Daily Signal Scanner — Unified morning scan for quality stock buy signals.

Downloads fresh data via yfinance and checks all validated strategy conditions
against a curated universe of quality large-caps.

Usage:
    python3 daily_signal_scanner.py

Strategies scanned:
    1. Multi-TF Dual Signal L — dip + RSI + reversal + weekly RSI decline (#1 strategy)
    2. Dual Signal D — dip + RSI + reversal (no weekly filter)
    3. Quality MR-A — dip + RSI
    4. RSI Divergence C — bullish RSI divergence on new lows (pending adversarial validation)

Output:
    - Clean terminal summary with market context, signals, watchlist
    - JSON saved to /home/jupiter/Lvl3Quant/state/daily_signals.json
"""

import json
import sys
import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP",
    "HD", "COST", "UNH", "LLY", "V", "MA", "ABBV", "MRK",
    "WMT", "AMZN", "GOOGL", "META",
]

STATE_PATH = Path("/home/jupiter/Lvl3Quant/state/daily_signals.json")

# Strategy thresholds
DIP_PCT = 0.05              # 5% dip from 20-day high
RSI_THRESHOLD = 35
RSI_PERIOD = 14
HIGH_LOOKBACK = 20
CONSEC_RED_MIN = 3
WEEKLY_RSI_DECLINE_WEEKS = 2
VOLUME_AVG_PERIOD = 20

# Watchlist thresholds
WATCHLIST_RSI_MAX = 45
WATCHLIST_DIP_MIN = 0.03


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scalar(val):
    """Extract scalar from potentially multi-index yfinance output."""
    if hasattr(val, "iloc"):
        return float(val.iloc[0]) if len(val) > 0 else float(val)
    return float(val)


def _flatten_cols(df):
    """Flatten multi-level columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def pct_from_high(close: pd.Series, lookback: int = 20) -> float:
    """Current percentage drop from rolling high. Positive = dip."""
    rolling_high = close.rolling(window=lookback, min_periods=1).max()
    dip = (rolling_high - close) / rolling_high
    return float(dip.iloc[-1])


def consecutive_red_days_before_today(df: pd.DataFrame) -> int:
    """Count consecutive red (close < open) days ending just before the last bar."""
    if len(df) < 2:
        return 0
    count = 0
    for i in range(len(df) - 2, -1, -1):
        if df["Close"].iloc[i] < df["Open"].iloc[i]:
            count += 1
        else:
            break
    return count


def is_green_day(df: pd.DataFrame) -> bool:
    """Latest bar is green (close > open)."""
    if len(df) < 1:
        return False
    return bool(df["Close"].iloc[-1] > df["Open"].iloc[-1])


# ---------------------------------------------------------------------------
# Strategy: Quality MR-A
# ---------------------------------------------------------------------------

def check_quality_mr_a(df: pd.DataFrame, rsi: pd.Series) -> dict:
    """Price dropped >5% from 20-day high AND RSI(14) < 35."""
    dip = pct_from_high(df["Close"], HIGH_LOOKBACK)
    current_rsi = float(rsi.iloc[-1])
    fired = dip >= DIP_PCT and current_rsi < RSI_THRESHOLD
    return {"fired": bool(fired), "dip_pct": round(dip * 100, 2), "rsi": round(current_rsi, 1)}


# ---------------------------------------------------------------------------
# Strategy: Dual Signal D
# ---------------------------------------------------------------------------

def check_dual_signal_d(df: pd.DataFrame, rsi: pd.Series) -> dict:
    """Quality MR-A conditions + first green day after 3+ consecutive red days."""
    mr = check_quality_mr_a(df, rsi)
    green = is_green_day(df)
    red_count = consecutive_red_days_before_today(df)
    fired = mr["fired"] and green and red_count >= CONSEC_RED_MIN
    return {
        "fired": bool(fired),
        "dip_pct": mr["dip_pct"],
        "rsi": mr["rsi"],
        "green_today": green,
        "consec_red_before": red_count,
    }


# ---------------------------------------------------------------------------
# Strategy: Multi-TF Dual Signal L
# ---------------------------------------------------------------------------

def check_multi_tf_dual_signal_l(df_daily: pd.DataFrame, df_weekly: pd.DataFrame,
                                  rsi_daily: pd.Series) -> dict:
    """Dual Signal D + weekly RSI declining for 2+ consecutive weeks."""
    dual = check_dual_signal_d(df_daily, rsi_daily)

    weekly_rsi_val = None
    weekly_declining = False

    if df_weekly is not None and len(df_weekly) >= RSI_PERIOD + WEEKLY_RSI_DECLINE_WEEKS + 1:
        w_rsi = compute_rsi(df_weekly["Close"], RSI_PERIOD)
        valid = w_rsi.dropna()
        if len(valid) >= WEEKLY_RSI_DECLINE_WEEKS + 1:
            weekly_rsi_val = float(valid.iloc[-1])
            declining = True
            for i in range(1, WEEKLY_RSI_DECLINE_WEEKS + 1):
                if valid.iloc[-i] >= valid.iloc[-i - 1]:
                    declining = False
                    break
            weekly_declining = declining

    fired = dual["fired"] and weekly_declining
    return {
        "fired": bool(fired),
        "dip_pct": dual["dip_pct"],
        "rsi": dual["rsi"],
        "green_today": dual["green_today"],
        "consec_red_before": dual["consec_red_before"],
        "weekly_rsi": round(weekly_rsi_val, 1) if weekly_rsi_val is not None else None,
        "weekly_rsi_declining": weekly_declining,
    }


# ---------------------------------------------------------------------------
# Strategy: RSI Divergence C
# ---------------------------------------------------------------------------

def check_rsi_divergence_c(df: pd.DataFrame, rsi: pd.Series) -> dict:
    """Bullish RSI divergence: price makes lower 20-day low, RSI makes higher low,
    volume below 20-day average. Pending adversarial validation."""
    result = {
        "fired": False,
        "price_lower_low": False,
        "rsi_higher_low": False,
        "volume_declining": False,
        "volume_ratio": None,
        "note": "pending adversarial validation",
    }

    close = df["Close"]
    if len(close) < 60:
        return result

    # Find local lows (minima within 10-bar windows) in last 60 bars
    arr = close.values
    cutoff = len(arr) - 60
    lows_idx = []
    for i in range(max(cutoff, 10), len(arr)):
        window_start = max(0, i - 10)
        if arr[i] == np.min(arr[window_start:i + 1]):
            lows_idx.append(i)

    if len(lows_idx) < 2:
        return result

    prev_i, curr_i = lows_idx[-2], lows_idx[-1]

    # Current bar should be near the latest low (within 3 bars)
    if len(arr) - 1 - curr_i > 3:
        return result

    price_lower = bool(arr[curr_i] < arr[prev_i])
    rsi_vals = rsi.values
    rsi_higher = bool(rsi_vals[curr_i] > rsi_vals[prev_i]) if not (
        np.isnan(rsi_vals[curr_i]) or np.isnan(rsi_vals[prev_i])) else False

    # Volume check
    vol_declining = False
    vol_ratio = None
    if "Volume" in df.columns:
        vol = df["Volume"]
        avg_vol = vol.rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        if hasattr(avg_vol, "iloc"):
            avg_vol = float(avg_vol.iloc[0])
        avg_vol = float(avg_vol)
        if avg_vol > 0:
            cur_vol = float(vol.iloc[-1]) if not hasattr(vol.iloc[-1], "iloc") else float(vol.iloc[-1].iloc[0])
            vol_ratio = round(cur_vol / avg_vol, 2)
            vol_declining = vol_ratio < 1.0

    fired = price_lower and rsi_higher and vol_declining
    result.update({
        "fired": bool(fired),
        "price_lower_low": price_lower,
        "rsi_higher_low": rsi_higher,
        "volume_declining": vol_declining,
        "volume_ratio": vol_ratio,
    })
    return result


# ---------------------------------------------------------------------------
# Watchlist (approaching signal territory)
# ---------------------------------------------------------------------------

def check_watchlist(df: pd.DataFrame, rsi: pd.Series) -> dict:
    """Stocks not yet signaling but close: RSI 35-45 or dip 3-5%."""
    dip = pct_from_high(df["Close"], HIGH_LOOKBACK)
    current_rsi = float(rsi.iloc[-1])

    near_rsi = RSI_THRESHOLD <= current_rsi <= WATCHLIST_RSI_MAX
    near_dip = WATCHLIST_DIP_MIN <= dip < DIP_PCT

    reasons = []
    if near_rsi:
        reasons.append(f"RSI {current_rsi:.1f} approaching {RSI_THRESHOLD}")
    if near_dip:
        reasons.append(f"Dip {dip*100:.1f}% approaching {DIP_PCT*100:.0f}%")

    return {
        "on_watchlist": bool(reasons),
        "dip_pct": round(dip * 100, 2),
        "rsi": round(current_rsi, 1),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Market Context
# ---------------------------------------------------------------------------

def get_market_context() -> dict:
    """SPY vs 200-SMA, VIX level, regime classification."""
    ctx = {
        "spy_price": None, "spy_sma200": None, "spy_above_200sma": None,
        "vix": None, "regime": "unknown",
    }

    try:
        spy = yf.download("SPY", period="1y", interval="1d", progress=False, auto_adjust=True)
        spy = _flatten_cols(spy)
        if len(spy) >= 200:
            ctx["spy_price"] = round(_scalar(spy["Close"].iloc[-1]), 2)
            ctx["spy_sma200"] = round(float(spy["Close"].rolling(200).mean().iloc[-1]), 2)
            ctx["spy_above_200sma"] = ctx["spy_price"] > ctx["spy_sma200"]
    except Exception as e:
        print(f"  Warning: SPY fetch failed: {e}")

    try:
        vix = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
        vix = _flatten_cols(vix)
        if len(vix) > 0:
            ctx["vix"] = round(_scalar(vix["Close"].iloc[-1]), 2)
    except Exception as e:
        print(f"  Warning: VIX fetch failed: {e}")

    # Regime classification
    above = ctx["spy_above_200sma"]
    v = ctx["vix"]
    if above is not None and v is not None:
        if above and v < 20:
            ctx["regime"] = "bullish-calm"
        elif above and v >= 20:
            ctx["regime"] = "bullish-volatile"
        elif not above and v < 25:
            ctx["regime"] = "bearish-calm"
        else:
            ctx["regime"] = "bearish-volatile"

    return ctx


# ---------------------------------------------------------------------------
# Main Scanner
# ---------------------------------------------------------------------------

STRATEGY_NAMES = {
    "multi_tf_dual_signal_l": "Multi-TF Dual Signal L",
    "dual_signal_d": "Dual Signal D",
    "quality_mr_a": "Quality MR-A",
    "rsi_divergence_c": "RSI Divergence C",
}


def scan_stock(ticker: str) -> dict:
    """Run all strategy checks on a single ticker."""
    result = {"ticker": ticker, "error": None, "signals": {}, "watchlist": {}, "metrics": {}}

    try:
        df_daily = yf.download(ticker, period="6mo", interval="1d",
                               progress=False, auto_adjust=True)
        if df_daily is None or len(df_daily) < HIGH_LOOKBACK + RSI_PERIOD:
            result["error"] = "insufficient daily data"
            return result
        df_daily = _flatten_cols(df_daily)

        df_weekly = yf.download(ticker, period="2y", interval="1wk",
                                progress=False, auto_adjust=True)
        if df_weekly is not None:
            df_weekly = _flatten_cols(df_weekly)

        rsi = compute_rsi(df_daily["Close"], RSI_PERIOD)

        # Metrics
        result["metrics"]["price"] = round(float(df_daily["Close"].iloc[-1]), 2)
        result["metrics"]["rsi"] = round(float(rsi.iloc[-1]), 1)
        result["metrics"]["dip_pct"] = round(pct_from_high(df_daily["Close"], HIGH_LOOKBACK) * 100, 2)

        if "Volume" in df_daily.columns:
            vol = df_daily["Volume"]
            avg_vol = float(vol.rolling(VOLUME_AVG_PERIOD).mean().iloc[-1])
            if avg_vol > 0:
                result["metrics"]["volume_ratio"] = round(float(vol.iloc[-1]) / avg_vol, 2)

        # Strategy checks
        result["signals"]["multi_tf_dual_signal_l"] = check_multi_tf_dual_signal_l(
            df_daily, df_weekly, rsi)
        result["signals"]["dual_signal_d"] = check_dual_signal_d(df_daily, rsi)
        result["signals"]["quality_mr_a"] = check_quality_mr_a(df_daily, rsi)
        result["signals"]["rsi_divergence_c"] = check_rsi_divergence_c(df_daily, rsi)

        # Watchlist
        result["watchlist"] = check_watchlist(df_daily, rsi)

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*65}")
    print(f"  DAILY SIGNAL SCANNER")
    print(f"  {today_str}")
    print(f"{'='*65}")

    # Market context
    print("\n  Fetching market context...")
    market = get_market_context()

    spy_dir = "ABOVE" if market["spy_above_200sma"] else "BELOW"
    print(f"\n  SPY: ${market['spy_price']}  |  200-SMA: ${market['spy_sma200']}  |  {spy_dir} 200-SMA")
    print(f"  VIX: {market['vix']}  |  Regime: {market['regime'].upper()}")
    print(f"{'─'*65}")

    # Scan stocks
    print(f"\n  Scanning {len(UNIVERSE)} stocks...\n")
    results = []

    for ticker in UNIVERSE:
        sys.stdout.write(f"    {ticker:6s} ")
        sys.stdout.flush()
        r = scan_stock(ticker)
        results.append(r)

        if r["error"]:
            print(f"ERROR ({r['error']})")
            continue

        fired = [k for k, v in r["signals"].items() if v.get("fired")]
        if fired:
            names = [STRATEGY_NAMES.get(k, k) for k in fired]
            print(f"*** SIGNAL: {', '.join(names)} ***")
        elif r["watchlist"].get("on_watchlist"):
            print(f"watch ({'; '.join(r['watchlist']['reasons'])})")
        else:
            print("--")

    # Separate into categories
    signal_stocks = [r for r in results if any(v.get("fired") for v in r["signals"].values())]
    watchlist_stocks = [r for r in results
                        if r["watchlist"].get("on_watchlist") and r not in signal_stocks]
    error_stocks = [r for r in results if r["error"]]

    # ── Signals detail ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  SIGNALS FIRED")
    print(f"{'='*65}")

    if not signal_stocks:
        print("\n  No signals today.\n")
    else:
        for r in signal_stocks:
            m = r["metrics"]
            print(f"\n  {r['ticker']}  ${m.get('price','?')}  "
                  f"RSI={m.get('rsi','?')}  Dip={m.get('dip_pct','?')}%  "
                  f"VolRatio={m.get('volume_ratio','?')}")
            for key, sig in r["signals"].items():
                if not sig.get("fired"):
                    continue
                label = STRATEGY_NAMES.get(key, key)
                parts = []
                if "dip_pct" in sig:
                    parts.append(f"dip={sig['dip_pct']}%")
                if "rsi" in sig:
                    parts.append(f"RSI={sig['rsi']}")
                if "consec_red_before" in sig:
                    parts.append(f"red_days={sig['consec_red_before']}")
                if "green_today" in sig:
                    parts.append(f"green={'Y' if sig['green_today'] else 'N'}")
                if sig.get("weekly_rsi") is not None:
                    parts.append(f"wkly_RSI={sig['weekly_rsi']}")
                if sig.get("weekly_rsi_declining"):
                    parts.append("wkly_RSI_declining=Y")
                if sig.get("volume_ratio") is not None:
                    parts.append(f"vol_ratio={sig['volume_ratio']}")
                if sig.get("note"):
                    parts.append(f"[{sig['note']}]")
                print(f"    -> {label}: {', '.join(parts)}")

    # ── Watchlist ──────────────────────────────────────────────────────
    if watchlist_stocks:
        print(f"\n{'─'*65}")
        print("  WATCHLIST (approaching signal territory)")
        print(f"{'─'*65}")
        for r in watchlist_stocks:
            m = r["metrics"]
            reasons = "; ".join(r["watchlist"]["reasons"])
            print(f"    {r['ticker']:6s} ${m.get('price','?'):>8}  "
                  f"RSI={m.get('rsi','?'):>5}  Dip={m.get('dip_pct','?')}%  |  {reasons}")

    # ── Summary ────────────────────────────────────────────────────────
    total_signals = sum(sum(1 for v in r["signals"].values() if v.get("fired")) for r in results)
    signal_tickers = [r["ticker"] for r in signal_stocks]

    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"    Signals:    {total_signals} across {len(signal_stocks)} stock(s)"
          + (f" — {', '.join(signal_tickers)}" if signal_tickers else ""))
    print(f"    Watchlist:  {len(watchlist_stocks)} stock(s)")
    if error_stocks:
        print(f"    Errors:     {len(error_stocks)} ({', '.join(e['ticker'] for e in error_stocks)})")
    print(f"{'='*65}\n")

    # ── Save JSON ──────────────────────────────────────────────────────
    output = {
        "scan_date": datetime.now().strftime("%Y-%m-%d"),
        "scan_time": datetime.now().strftime("%H:%M:%S"),
        "market_context": market,
        "universe_size": len(UNIVERSE),
        "total_signals": total_signals,
        "signal_stocks": [
            {
                "ticker": r["ticker"],
                "price": r["metrics"].get("price"),
                "strategies_fired": [k for k, v in r["signals"].items() if v.get("fired")],
                "signals": r["signals"],
                "metrics": r["metrics"],
            }
            for r in signal_stocks
        ],
        "watchlist_stocks": [
            {
                "ticker": r["ticker"],
                "price": r["metrics"].get("price"),
                "rsi": r["metrics"].get("rsi"),
                "dip_pct": r["metrics"].get("dip_pct"),
                "reasons": r["watchlist"]["reasons"],
            }
            for r in watchlist_stocks
        ],
        "all_results": [
            {
                "ticker": r["ticker"],
                "metrics": r["metrics"],
                "signals": r["signals"],
                "watchlist": r["watchlist"],
                "error": r["error"],
            }
            for r in results
        ],
    }

    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Saved to {STATE_PATH}\n")
    except Exception as e:
        print(f"  Warning: could not save JSON: {e}\n")

    return output


if __name__ == "__main__":
    main()
