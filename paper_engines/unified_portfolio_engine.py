#!/usr/bin/env python3
"""
Unified Portfolio Decision Engine
==================================
Daily cron job (9:35 AM ET) that checks all strategy signals and outputs
a single recommended action. Does NOT execute trades.

Strategy Priority:
  1. ROTATION = baseline (always active unless RSI interrupts)
  2. RSI oversold signals INTERRUPT rotation for 5-10 day trades
  3. Earnings signals only in BULL regime AND when no RSI active

State file:  /home/jupiter/Lvl3Quant/state/unified_portfolio_state.json
Output file: /home/jupiter/Lvl3Quant/state/unified_portfolio_recommendation.json
"""

import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

STATE_FILE = Path("/home/jupiter/Lvl3Quant/state/unified_portfolio_state.json")
RECOMMENDATION_FILE = Path("/home/jupiter/Lvl3Quant/state/unified_portfolio_recommendation.json")

GROWTH_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "AVGO", "CRM", "NFLX", "SHOP", "XYZ", "SNOW", "PLTR", "COIN",
    "MELI", "MDB", "DDOG", "TTD",
]

SAFE_HAVENS = ["GLD", "TLT", "UUP"]

RSI_MAX_HOLD_DAYS = 10  # default, overridden by vol bucket below
RSI_HOLD_BY_VOL = {"low": 15, "medium": 10, "high": 5}
RSI_EXIT_THRESHOLD = 50
ROTATION_REBALANCE_DAYS = 30
ROTATION_LOOKBACK_MONTHS = 3
KILL_SWITCH_VIX = 20
RSI_PERIOD = 5
SMA_200_PERIOD = 200
SMA_50_PERIOD = 50


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch_yfinance():
    """Lazy import to keep startup fast when testing."""
    import yfinance as yf
    return yf


def fetch_price_history(symbol: str, period: str = "1y", interval: str = "1d"):
    """Fetch OHLCV history via yfinance. Returns a DataFrame or None on failure."""
    yf = _fetch_yfinance()
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(period=period, interval=interval)
        if df is None or df.empty:
            print(f"  [WARN] No data returned for {symbol}")
            return None
        return df
    except Exception as e:
        print(f"  [ERR]  yfinance fetch failed for {symbol}: {e}")
        return None


def compute_rsi(series, period=14):
    """Compute RSI from a pandas Series of close prices.
    Uses SIMPLE moving average (not EWM) to match our validated strategy signals.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_sma(series, period):
    """Simple moving average."""
    return series.rolling(window=period).mean()


def compute_annualized_vol(df, window=60):
    """Annualized volatility from daily returns (last *window* days)."""
    if df is None or len(df) < window:
        return None
    rets = df["Close"].pct_change().dropna().tail(window)
    if len(rets) < 20:
        return None
    return float(rets.std() * np.sqrt(252) * 100)  # percent


# ---------------------------------------------------------------------------
# Market state
# ---------------------------------------------------------------------------

def get_market_state() -> dict:
    """Fetch SPY + VIX and derive regime / kill-switch."""
    spy_df = fetch_price_history("SPY", period="2y")
    vix_df = fetch_price_history("^VIX", period="1mo")

    state = {
        "spy_price": None,
        "spy_200sma": None,
        "spy_50sma": None,
        "vix": None,
        "regime": "unknown",
        "kill_switch": False,
    }

    if spy_df is not None and len(spy_df) >= SMA_200_PERIOD:
        spy_close = spy_df["Close"]
        state["spy_price"] = round(float(spy_close.iloc[-1]), 2)
        state["spy_200sma"] = round(float(compute_sma(spy_close, SMA_200_PERIOD).iloc[-1]), 2)
        state["spy_50sma"] = round(float(compute_sma(spy_close, SMA_50_PERIOD).iloc[-1]), 2)

        if state["spy_price"] > state["spy_200sma"]:
            state["regime"] = "bull"
        else:
            state["regime"] = "bear"

    if vix_df is not None and not vix_df.empty:
        state["vix"] = round(float(vix_df["Close"].iloc[-1]), 2)

    # Kill switch: VIX > 20 AND SPY < 50-SMA
    if (state["vix"] is not None and state["spy_price"] is not None
            and state["spy_50sma"] is not None):
        if state["vix"] > KILL_SWITCH_VIX and state["spy_price"] < state["spy_50sma"]:
            state["kill_switch"] = True

    return state


# ---------------------------------------------------------------------------
# RSI scanner
# ---------------------------------------------------------------------------

def _vol_bucket_threshold(vol_pct):
    """Return RSI threshold based on volatility bucket."""
    if vol_pct is None:
        return 20  # default medium
    if vol_pct < 20:
        return 15  # low vol
    elif vol_pct <= 35:
        return 20  # medium vol
    else:
        return 30  # high vol


def _vol_bucket_label(vol_pct):
    if vol_pct is None:
        return "medium"
    if vol_pct < 20:
        return "low"
    elif vol_pct <= 35:
        return "medium"
    return "high"


def scan_rsi_signals() -> list:
    """Scan growth universe for RSI(5) oversold signals.

    Requirements:
      - Stock must be above its 200-SMA
      - RSI(5) must be below the volatility-adjusted threshold
    """
    signals = []
    for symbol in GROWTH_UNIVERSE:
        try:
            df = fetch_price_history(symbol, period="2y")
            if df is None or len(df) < SMA_200_PERIOD:
                continue

            close = df["Close"]
            current_price = float(close.iloc[-1])
            sma200 = float(compute_sma(close, SMA_200_PERIOD).iloc[-1])

            # Must be above 200-SMA
            if current_price <= sma200:
                continue

            rsi5 = float(compute_rsi(close, RSI_PERIOD).iloc[-1])
            vol = compute_annualized_vol(df)
            threshold = _vol_bucket_threshold(vol)

            if rsi5 < threshold:
                signals.append({
                    "symbol": symbol,
                    "rsi5": round(rsi5, 1),
                    "vol_bucket": _vol_bucket_label(vol),
                    "vol_pct": round(vol, 1) if vol else None,
                    "threshold": threshold,
                    "price": round(current_price, 2),
                    "sma200": round(sma200, 2),
                })
        except Exception as e:
            print(f"  [ERR]  RSI scan failed for {symbol}: {e}")

    # Sort by how far below threshold (most oversold first)
    signals.sort(key=lambda s: s["rsi5"] - s["threshold"])
    return signals


# ---------------------------------------------------------------------------
# Rotation logic
# ---------------------------------------------------------------------------

def rank_safe_havens() -> list:
    """Rank safe havens by vol-adjusted 3-month relative strength."""
    rankings = []
    spy_df = fetch_price_history("SPY", period="1y")
    if spy_df is None or len(spy_df) < 63:
        return rankings
    spy_ret_3m = float(spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-63] - 1)

    for symbol in SAFE_HAVENS:
        try:
            df = fetch_price_history(symbol, period="1y")
            if df is None or len(df) < 63:
                continue
            close = df["Close"]
            ret_3m = float(close.iloc[-1] / close.iloc[-63] - 1)
            rel_strength = ret_3m - spy_ret_3m
            vol = compute_annualized_vol(df) or 15.0
            vol_adj = rel_strength / (vol / 100.0) if vol > 0 else rel_strength
            rankings.append({
                "symbol": symbol,
                "ret_3m": round(ret_3m * 100, 2),
                "rel_strength": round(rel_strength * 100, 2),
                "vol_adj_score": round(vol_adj, 4),
            })
        except Exception as e:
            print(f"  [ERR]  Rotation rank failed for {symbol}: {e}")

    rankings.sort(key=lambda r: r["vol_adj_score"], reverse=True)
    return rankings


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load previous state from JSON file. Returns default if missing."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARN] Could not read state file: {e}")
    return _default_state()


def _default_state() -> dict:
    return {
        "last_update": None,
        "current_position": {
            "symbol": None,
            "shares": 0,
            "entry_date": None,
            "entry_price": 0,
            "strategy": None,
        },
        "last_rotation_date": None,
        "recommendation": None,
        "market_state": {},
        "signals_scanned": {},
    }


def _days_since(date_str: str) -> int:
    """Days between a date string and today."""
    if not date_str:
        return 9999
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return (datetime.now() - d).days
    except ValueError:
        return 9999


def run_engine():
    """Main decision loop. Returns (state_dict, recommendation_dict, summary_str)."""
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%dT%H:%M:%S")

    prev_state = load_state()
    position = prev_state.get("current_position") or _default_state()["current_position"]
    last_rotation_date = prev_state.get("last_rotation_date")

    # ------------------------------------------------------------------
    # 1. Market state
    # ------------------------------------------------------------------
    print("=" * 60)
    print("UNIFIED PORTFOLIO ENGINE")
    print(f"Run time: {now_str}")
    print("=" * 60)

    print("\n[1] Fetching market state...")
    market = get_market_state()
    print(f"    SPY: {market['spy_price']}  200-SMA: {market['spy_200sma']}  "
          f"50-SMA: {market['spy_50sma']}  VIX: {market['vix']}")
    print(f"    Regime: {market['regime'].upper()}  Kill switch: {market['kill_switch']}")

    recommendation = None
    rsi_signals = []
    rotation_rankings = []
    summary_lines = []

    # ------------------------------------------------------------------
    # 2. Kill switch override
    # ------------------------------------------------------------------
    if market["kill_switch"]:
        print("\n*** KILL SWITCH ACTIVE: VIX > 20 AND SPY < 50-SMA ***")
        if position.get("symbol"):
            recommendation = {
                "action": "SELL",
                "symbol": position["symbol"],
                "shares": position.get("shares", 0),
                "strategy": "kill_switch",
                "reason": f"Kill switch: VIX={market['vix']}, SPY={market['spy_price']} < 50-SMA={market['spy_50sma']}",
                "exit_conditions": {},
                "priority": "CRITICAL",
            }
            summary_lines.append(f"KILL SWITCH: Sell {position['symbol']} immediately, go to cash")
        else:
            recommendation = {
                "action": "HOLD_CASH",
                "symbol": None,
                "shares": 0,
                "strategy": "kill_switch",
                "reason": f"Kill switch active, stay in cash. VIX={market['vix']}",
                "exit_conditions": {},
                "priority": "HIGH",
            }
            summary_lines.append("KILL SWITCH: Stay in cash")

    # ------------------------------------------------------------------
    # 3. Check current RSI trade exit
    # ------------------------------------------------------------------
    if recommendation is None and position.get("strategy") == "adaptive_rsi_e":
        print(f"\n[3] In RSI trade: {position['symbol']} (entered {position.get('entry_date')})")
        hold_days = _days_since(position.get("entry_date"))
        # Fetch current RSI for the position
        pos_df = fetch_price_history(position["symbol"], period="6mo")
        current_rsi5 = None
        if pos_df is not None and len(pos_df) > RSI_PERIOD:
            current_rsi5 = float(compute_rsi(pos_df["Close"], RSI_PERIOD).iloc[-1])

        # Get vol-bucketed max hold from position exit_conditions, or compute
        vol_bucket = position.get("vol_bucket", "medium")
        max_hold = RSI_HOLD_BY_VOL.get(vol_bucket, RSI_MAX_HOLD_DAYS)

        exit_reason = None
        if current_rsi5 is not None and current_rsi5 > RSI_EXIT_THRESHOLD:
            exit_reason = f"RSI(5)={current_rsi5:.1f} > {RSI_EXIT_THRESHOLD} exit threshold"
        elif hold_days >= max_hold:
            exit_reason = f"Max hold {max_hold} days reached (held {hold_days}d, {vol_bucket} vol)"

        if exit_reason:
            print(f"    EXIT: {exit_reason}")
            recommendation = {
                "action": "SELL",
                "symbol": position["symbol"],
                "shares": position.get("shares", 0),
                "strategy": "adaptive_rsi_e_exit",
                "reason": exit_reason,
                "exit_conditions": {},
                "priority": "HIGH",
                "next_step": "check_rotation",
            }
            summary_lines.append(f"RSI EXIT: Sell {position['symbol']} - {exit_reason}")
        else:
            rsi_str = f"RSI(5)={current_rsi5:.1f}" if current_rsi5 else "RSI unknown"
            print(f"    HOLD: {rsi_str}, held {hold_days}d")
            recommendation = {
                "action": "HOLD",
                "symbol": position["symbol"],
                "shares": position.get("shares", 0),
                "strategy": "adaptive_rsi_e",
                "reason": f"Holding RSI trade: {rsi_str}, day {hold_days}/{RSI_MAX_HOLD_DAYS}",
                "exit_conditions": {"rsi_above": RSI_EXIT_THRESHOLD, "max_hold_days": RSI_MAX_HOLD_DAYS},
                "priority": "LOW",
            }
            summary_lines.append(f"HOLD RSI: {position['symbol']} - {rsi_str}, day {hold_days}")

    # ------------------------------------------------------------------
    # 4. Scan for new RSI signals (if not already in RSI trade)
    # ------------------------------------------------------------------
    if recommendation is None:
        print("\n[4] Scanning for RSI oversold signals...")
        rsi_signals = scan_rsi_signals()
        if rsi_signals:
            best = rsi_signals[0]
            print(f"    SIGNAL: {best['symbol']} RSI(5)={best['rsi5']} "
                  f"(threshold={best['threshold']}, {best['vol_bucket']} vol)")

            action = "BUY"
            sell_first = None
            if position.get("symbol") and position["symbol"] != best["symbol"]:
                # Need to sell current position first
                sell_first = {
                    "action": "SELL",
                    "symbol": position["symbol"],
                    "shares": position.get("shares", 0),
                    "strategy": "rotation_exit_for_rsi",
                    "reason": f"Exiting {position['symbol']} to enter RSI trade on {best['symbol']}",
                }
                summary_lines.append(f"SELL {position['symbol']} to free capital for RSI trade")

            recommendation = {
                "action": action,
                "symbol": best["symbol"],
                "shares": 0,  # sizing left to execution layer
                "strategy": "adaptive_rsi_e",
                "reason": (f"RSI(5)={best['rsi5']}, {best['vol_bucket']}-vol bucket "
                           f"(threshold<{best['threshold']}), above 200-SMA"),
                "exit_conditions": {
                    "rsi_above": RSI_EXIT_THRESHOLD,
                    "max_hold_days": RSI_HOLD_BY_VOL.get(best["vol_bucket"], RSI_MAX_HOLD_DAYS),
                    "vol_bucket": best["vol_bucket"],
                },
                "priority": "HIGH",
            }
            if sell_first:
                recommendation["sell_first"] = sell_first

            summary_lines.append(
                f"RSI BUY: {best['symbol']} RSI(5)={best['rsi5']} "
                f"({best['vol_bucket']} vol, threshold {best['threshold']}, "
                f"hold {RSI_HOLD_BY_VOL.get(best['vol_bucket'], RSI_MAX_HOLD_DAYS)}d)"
            )
        else:
            print("    No RSI signals found.")

    # ------------------------------------------------------------------
    # 5. Rotation check (if no RSI signal)
    # ------------------------------------------------------------------
    if recommendation is None:
        days_since_rotation = _days_since(last_rotation_date)
        print(f"\n[5] Checking rotation... (last rotation: {last_rotation_date}, "
              f"{days_since_rotation}d ago)")

        rotation_rankings = rank_safe_havens()
        top_asset = rotation_rankings[0]["symbol"] if rotation_rankings else None

        if rotation_rankings:
            print(f"    Rankings: {', '.join(r['symbol'] + '=' + str(r['vol_adj_score']) for r in rotation_rankings)}")
            print(f"    Top asset: {top_asset}")

        needs_rebalance = days_since_rotation >= ROTATION_REBALANCE_DAYS
        current_sym = position.get("symbol")
        top_changed = top_asset and top_asset != current_sym

        if needs_rebalance and top_changed and top_asset:
            print(f"    ROTATE: {current_sym or 'cash'} -> {top_asset}")
            recommendation = {
                "action": "BUY",
                "symbol": top_asset,
                "shares": 0,
                "strategy": "rotation",
                "reason": (f"Monthly rebalance ({days_since_rotation}d since last). "
                           f"Top safe haven: {top_asset} "
                           f"(vol-adj score={rotation_rankings[0]['vol_adj_score']})"),
                "exit_conditions": {},
                "priority": "MEDIUM",
            }
            if current_sym:
                recommendation["sell_first"] = {
                    "action": "SELL",
                    "symbol": current_sym,
                    "shares": position.get("shares", 0),
                    "strategy": "rotation_exit",
                    "reason": f"Rotation from {current_sym} to {top_asset}",
                }
                summary_lines.append(f"ROTATE: Sell {current_sym}, buy {top_asset}")
            else:
                summary_lines.append(f"ROTATE: Buy {top_asset} (from cash)")
        elif needs_rebalance and not top_changed and current_sym:
            print(f"    HOLD rotation: {current_sym} still top-ranked")
            recommendation = {
                "action": "HOLD",
                "symbol": current_sym,
                "shares": position.get("shares", 0),
                "strategy": "rotation",
                "reason": f"{current_sym} still top-ranked safe haven. No change needed.",
                "exit_conditions": {},
                "priority": "LOW",
            }
            summary_lines.append(f"HOLD ROTATION: {current_sym} still top-ranked")
        elif not needs_rebalance and current_sym:
            print(f"    HOLD rotation: {days_since_rotation}d < {ROTATION_REBALANCE_DAYS}d threshold")
            recommendation = {
                "action": "HOLD",
                "symbol": current_sym,
                "shares": position.get("shares", 0),
                "strategy": "rotation",
                "reason": (f"Only {days_since_rotation}d since last rotation "
                           f"(need {ROTATION_REBALANCE_DAYS}d). Hold {current_sym}."),
                "exit_conditions": {},
                "priority": "LOW",
            }
            summary_lines.append(f"HOLD ROTATION: {current_sym}, rebalance in {ROTATION_REBALANCE_DAYS - days_since_rotation}d")
        else:
            # No position, time to rotate into top asset
            if top_asset:
                recommendation = {
                    "action": "BUY",
                    "symbol": top_asset,
                    "shares": 0,
                    "strategy": "rotation",
                    "reason": f"No position. Top safe haven: {top_asset}",
                    "exit_conditions": {},
                    "priority": "MEDIUM",
                }
                summary_lines.append(f"ROTATE: Buy {top_asset} (from cash)")
            else:
                recommendation = {
                    "action": "HOLD_CASH",
                    "symbol": None,
                    "shares": 0,
                    "strategy": "rotation",
                    "reason": "No safe haven data available. Stay in cash.",
                    "exit_conditions": {},
                    "priority": "LOW",
                }
                summary_lines.append("HOLD CASH: No rotation data")

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    new_state = {
        "last_update": now_str,
        "current_position": position,
        "last_rotation_date": last_rotation_date,
        "recommendation": recommendation,
        "market_state": market,
        "signals_scanned": {
            "rsi_signals": rsi_signals[:5],  # top 5 only
            "rotation_top": rotation_rankings[0]["symbol"] if rotation_rankings else None,
            "rotation_rankings": rotation_rankings,
            "earnings_pending": [],  # placeholder for future earnings integration
        },
    }

    # Also build a standalone recommendation file
    rec_output = {
        "generated_at": now_str,
        "market_regime": market.get("regime", "unknown"),
        "kill_switch": market.get("kill_switch", False),
        "recommendation": recommendation,
        "current_position": position,
        "summary": " | ".join(summary_lines) if summary_lines else "No action needed",
    }

    # Summary
    summary = "\n".join([
        "",
        "-" * 60,
        "RECOMMENDATION SUMMARY",
        "-" * 60,
        f"  Regime: {market.get('regime', '?').upper()}  |  VIX: {market.get('vix', '?')}  |  Kill switch: {market.get('kill_switch', False)}",
        f"  Current position: {position.get('symbol') or 'CASH'} ({position.get('strategy', 'none')})",
        "",
    ] + [f"  >>> {line}" for line in summary_lines] + [
        "",
        f"  Action: {recommendation.get('action', 'NONE')}",
        f"  Symbol: {recommendation.get('symbol', 'N/A')}",
        f"  Strategy: {recommendation.get('strategy', 'N/A')}",
        f"  Reason: {recommendation.get('reason', 'N/A')}",
        f"  Priority: {recommendation.get('priority', 'N/A')}",
        "-" * 60,
    ])

    return new_state, rec_output, summary


def save_outputs(state: dict, rec: dict):
    """Write state and recommendation files atomically."""
    for path, data in [(STATE_FILE, state), (RECOMMENDATION_FILE, rec)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp.rename(path)
        print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        state, rec, summary = run_engine()
        save_outputs(state, rec)
        print(summary)
        return 0
    except Exception as e:
        print(f"\n[FATAL] Engine failed: {e}")
        traceback.print_exc()
        # Write error state so downstream knows something went wrong
        error_rec = {
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "error": str(e),
            "recommendation": {
                "action": "ERROR",
                "symbol": None,
                "strategy": "engine_failure",
                "reason": f"Engine crashed: {e}",
                "priority": "CRITICAL",
            },
            "summary": f"ENGINE ERROR: {e}",
        }
        try:
            RECOMMENDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(RECOMMENDATION_FILE, "w") as f:
                json.dump(error_rec, f, indent=2)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
