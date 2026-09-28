#!/usr/bin/env python3
"""
Play Scanner v2 — Pre-Market Options Scanner
=============================================
Enhanced scanner with earnings filter, real options pricing, momentum decay
detection, sector correlation, VIX regime filter, and trade tracking.

Setup types:
  A) Momentum continuation: above all SMAs, RSI 55-70, MFI>60, OBV rising, 1-3% pullback
  B) Oversold bounce: 10%+ weekly drop or RSI<35 (VALIDATED p=0.01, 69% WR, 5d hold)
  C) Flow divergence: OBV rising while price flat/down, MFI turning up from <30
  D) Post-earnings bounce: 8%+ drop in 2d after earnings (VALIDATED all gates, 63% WR, 10d hold)

Runs at 8:30 AM ET (pre-market) via PM2 cron: "30 12 * * 1-5"
State: /home/jupiter/Lvl3Quant/state/play_scanner_state.json
History: /home/jupiter/Lvl3Quant/state/play_scanner_history.jsonl
Trade log: /home/jupiter/Lvl3Quant/state/agentic_trade_log.json
"""
import json
import re
import sys
import time
import traceback
import urllib.request
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")


# ── Discord alerting ──────────────────────────────────────────────────────
def _get_discord_webhook() -> str | None:
    """Get Discord webhook URL from webhooks.json config."""
    try:
        wf = Path("/home/jupiter/teleclaude-main/config/webhooks.json")
        if wf.exists():
            config = json.loads(wf.read_text())
            webhooks = config.get("webhooks", {})
            # Use notifications (system-status) webhook for scanner alerts
            url = webhooks.get("notifications") or webhooks.get("default")
            if url:
                return url
    except Exception:
        pass
    # Fallback: try API_KEYS.md (legacy)
    try:
        wf = Path("/home/jupiter/teleclaude-main/API_KEYS.md")
        if wf.exists():
            for line in wf.read_text().split('\n'):
                if 'discord' in line.lower() and 'webhook' in line.lower() and 'http' in line:
                    urls = re.findall(r'https://discord\.com/api/webhooks/\S+', line)
                    if urls:
                        return urls[0].strip('`').strip()
    except Exception:
        pass
    return None


def _send_discord(message: str):
    """Send scanner results to Discord."""
    url = _get_discord_webhook()
    if not url:
        print("  [Discord] No webhook configured, skipping alert")
        return
    try:
        data = json.dumps({"content": message[:2000]}).encode()
        req = urllib.request.Request(url, data=data,
                                      headers={"Content-Type": "application/json"},
                                      method="POST")
        urllib.request.urlopen(req, timeout=10)
        print("  [Discord] Alert sent successfully")
    except Exception as e:
        print(f"  [Discord] Failed: {e}")
sys.stdout.reconfigure(line_buffering=True)

ET = pytz.timezone("US/Eastern")
STATE_DIR = Path("/home/jupiter/Lvl3Quant/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "play_scanner_state.json"
HISTORY_FILE = STATE_DIR / "play_scanner_history.jsonl"
TRADE_LOG_FILE = STATE_DIR / "agentic_trade_log.json"

# Budget constraint
MAX_CONTRACT_COST = 300  # $300 max per contract for $645 account
MAX_BID_ASK_SPREAD_PCT_RTH = 0.20  # 20% of mid price during regular trading hours
MAX_BID_ASK_SPREAD_PCT_AH = 0.40   # 40% tolerance after hours (spreads widen)

# ── Universes with sector mapping ──
SECTOR_MAP = {
    # Sector ETFs
    "XLK": "Technology", "XLF": "Financials", "XLV": "Healthcare",
    "XLE": "Energy", "XLI": "Industrials", "XLC": "Communications",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLU": "Utilities", "XLRE": "Real Estate", "XLB": "Materials",
    # Index ETFs
    "SPY": "Broad Market", "QQQ": "Technology", "IWM": "Small Cap",
    "GLD": "Commodities", "TLT": "Bonds", "SLV": "Commodities",
    # Top S&P 500
    "AAPL": "Technology", "MSFT": "Technology", "AMZN": "Consumer Discretionary",
    "NVDA": "Technology", "GOOGL": "Communications", "META": "Communications",
    "BRK-B": "Financials", "LLY": "Healthcare", "AVGO": "Technology",
    "JPM": "Financials", "TSLA": "Consumer Discretionary", "UNH": "Healthcare",
    "V": "Financials", "XOM": "Energy", "MA": "Financials",
    "COST": "Consumer Staples", "PG": "Consumer Staples", "JNJ": "Healthcare",
    "HD": "Consumer Discretionary", "ABBV": "Healthcare", "MRK": "Healthcare",
    "WMT": "Consumer Staples", "BAC": "Financials", "CRM": "Technology",
    "NFLX": "Communications", "AMD": "Technology", "ORCL": "Technology",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "TMO": "Healthcare",
    # Budget-friendly additions
    "F": "Consumer Discretionary", "INTC": "Technology", "SNAP": "Communications",
    "PLTR": "Technology", "SOFI": "Financials", "RIVN": "Consumer Discretionary",
    "NIO": "Consumer Discretionary", "MARA": "Financials", "COIN": "Financials",
    "T": "Communications", "VZ": "Communications", "PFE": "Healthcare",
    "CSCO": "Technology", "GM": "Consumer Discretionary", "UBER": "Technology",
    "ROKU": "Communications", "DKNG": "Consumer Discretionary",
    "HOOD": "Financials",
    "UPST": "Financials", "OPEN": "Real Estate", "FUTU": "Financials", "MQ": "Technology",
}

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]
INDEX_ETFS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "SLV"]
TOP_30_SP500 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "V", "XOM", "MA", "COST", "PG", "JNJ", "HD", "ABBV",
    "MRK", "WMT", "BAC", "CRM", "NFLX", "AMD", "ORCL", "KO", "PEP", "TMO",
]
# Budget-friendly stocks (sub-$100) — more likely to produce affordable oversold bounce options
# for the ~$676 agentic account. High-beta + liquid options + part of S&P 500/large-cap.
BUDGET_FRIENDLY = [
    "F", "INTC", "SNAP", "PLTR", "SOFI", "RIVN", "NIO", "MARA", "COIN",
    "T", "VZ", "PFE", "CSCO", "GM", "UBER",  "ROKU", "DKNG", "HOOD",
    "UPST", "OPEN", "FUTU", "MQ",  # Added for more sub-$50 option coverage
]
ALL_TICKERS = SECTOR_ETFS + INDEX_ETFS + TOP_30_SP500 + BUDGET_FRIENDLY


# ═══════════════════════════════════════════
# Indicator Functions
# ═══════════════════════════════════════════

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_mfi(high, low, close, volume, period: int = 14) -> pd.Series:
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    delta = typical_price.diff()
    pos_flow = raw_money_flow.where(delta > 0, 0.0)
    neg_flow = raw_money_flow.where(delta <= 0, 0.0)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def compute_bollinger_position(close: pd.Series, period: int = 20) -> pd.Series:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    width = upper - lower
    return (close - lower) / width.replace(0, np.nan)


# ═══════════════════════════════════════════
# Feature 1: Earnings Calendar Filter
# ═══════════════════════════════════════════

def check_earnings_proximity(ticker_str: str, days_ahead: int = 14) -> dict:
    """Check if earnings are within the next N days."""
    result = {"has_upcoming_earnings": False, "earnings_date": None, "days_until": None}
    try:
        tk = yf.Ticker(ticker_str)
        # Try calendar first
        cal = tk.calendar
        if cal is not None and not (isinstance(cal, pd.DataFrame) and cal.empty):
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, list):
                        ed = ed[0]
                    if hasattr(ed, "date"):
                        ed = ed.date() if callable(ed.date) else ed.date
                    elif isinstance(ed, str):
                        ed = datetime.strptime(ed[:10], "%Y-%m-%d").date()
                    days_diff = (ed - datetime.now(ET).date()).days
                    if 0 <= days_diff <= days_ahead:
                        result["has_upcoming_earnings"] = True
                        result["earnings_date"] = str(ed)
                        result["days_until"] = days_diff
                        return result
            elif isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.index:
                    ed_val = cal.loc["Earnings Date"].iloc[0]
                    if hasattr(ed_val, "date"):
                        ed = ed_val.date() if callable(ed_val.date) else ed_val.date
                    else:
                        ed = datetime.strptime(str(ed_val)[:10], "%Y-%m-%d").date()
                    days_diff = (ed - datetime.now(ET).date()).days
                    if 0 <= days_diff <= days_ahead:
                        result["has_upcoming_earnings"] = True
                        result["earnings_date"] = str(ed)
                        result["days_until"] = days_diff
                        return result

        # Fallback: earnings_dates
        try:
            edates = tk.earnings_dates
            if edates is not None and not edates.empty:
                today = datetime.now(ET).date()
                for idx in edates.index:
                    d = idx.date() if hasattr(idx, "date") else idx
                    days_diff = (d - today).days
                    if 0 <= days_diff <= days_ahead:
                        result["has_upcoming_earnings"] = True
                        result["earnings_date"] = str(d)
                        result["days_until"] = days_diff
                        return result
        except Exception:
            pass
    except Exception:
        pass
    return result


def check_recent_earnings_drop(ticker_str: str, close: pd.Series) -> dict | None:
    """
    Check if stock had earnings in the last 5 trading days AND dropped 8%+
    in the 2 days after earnings. This is the post-earnings bounce signal.

    Backtest validated (2026-07-24): 8% drop + 10d hold passes ALL 5 adversarial gates.
    51 trades, 62.7% WR, PF 1.70, Sharpe 1.50, regime gap 0.33.
    """
    try:
        tk = yf.Ticker(ticker_str)
        today = datetime.now(ET).date()

        # Get recent earnings dates
        edates = None
        try:
            cal = tk.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        if isinstance(ed, list):
                            ed = ed[0]
                        if hasattr(ed, "date"):
                            ed = ed.date() if callable(ed.date) else ed.date
                        elif isinstance(ed, str):
                            ed = datetime.strptime(ed[:10], "%Y-%m-%d").date()
                        tdays_ago = len(pd.bdate_range(ed, today, inclusive="neither"))
                        if 2 <= tdays_ago <= 7:
                            edates = ed
        except Exception:
            pass

        if edates is None:
            try:
                edf = tk.earnings_dates
                if edf is not None and not edf.empty:
                    for idx in edf.index:
                        d = idx.date() if hasattr(idx, "date") else idx
                        tdays_ago = len(pd.bdate_range(d, today, inclusive="neither"))
                        if 2 <= tdays_ago <= 7:
                            edates = d
                            break
            except Exception:
                pass

        if edates is None:
            return None

        # Check if there was an 8%+ drop in the 2 trading days after earnings
        # Use trading days (business days) not calendar days to handle weekends/holidays
        trading_days_since = len(pd.bdate_range(edates, today, inclusive="neither"))
        calendar_days_since = (today - edates).days

        if trading_days_since < 2 or trading_days_since > 7:
            return None

        if len(close) < trading_days_since + 3:
            return None

        # Find the close on earnings day and 2 trading days after
        # Use trading_days_since to index into price series correctly
        earnings_idx = len(close) - trading_days_since - 1
        if earnings_idx < 0 or earnings_idx + 2 >= len(close):
            return None

        price_at_earnings = float(close.iloc[earnings_idx])
        price_2d_after = float(close.iloc[earnings_idx + 2])
        drop_2d = (price_2d_after / price_at_earnings) - 1

        if drop_2d <= -0.08:  # 8%+ drop
            return {
                "earnings_date": str(edates),
                "days_since": trading_days_since,
                "drop_2d_pct": round(drop_2d * 100, 1),
                "price_at_earnings": round(price_at_earnings, 2),
                "price_after_drop": round(price_2d_after, 2),
            }

    except Exception:
        pass
    return None


# ═══════════════════════════════════════════
# Feature 2: Real Options Pricing
# ═══════════════════════════════════════════

def is_market_hours() -> bool:
    """Check if US equity options markets are open."""
    now = datetime.now(ET)
    if now.weekday() >= 5:  # Weekend
        return False
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    return market_open <= now <= market_close


def get_option_pricing(ticker_str: str, price: float, direction: str = "call",
                       target_dte: int = 21) -> dict | None:
    """Get real option chain data for nearest expiry around target_dte ATM/slightly-OTM strike.

    Args:
        target_dte: Target days to expiry. 21 for momentum, 14 for oversold bounce (5d hold).
    """
    try:
        tk = yf.Ticker(ticker_str)
        exp_dates = tk.options
        if not exp_dates:
            return None

        today = datetime.now(ET).date()
        target_date = today + timedelta(days=target_dte)

        # Find nearest expiry to target (allow ±50% range)
        min_dte = max(7, int(target_dte * 0.5))
        max_dte = int(target_dte * 2.0)
        best_exp = None
        best_diff = 999
        for exp_str in exp_dates:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            diff = (exp_date - today).days
            if min_dte <= diff <= max_dte:
                if abs(diff - target_dte) < best_diff:
                    best_diff = abs(diff - target_dte)
                    best_exp = exp_str

        if not best_exp:
            # Fall back to nearest available expiry >= 14 days
            for exp_str in exp_dates:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                diff = (exp_date - today).days
                if diff >= 7:
                    best_exp = exp_str
                    break

        if not best_exp:
            return None

        chain = tk.option_chain(best_exp)
        opts = chain.calls if direction == "call" else chain.puts

        if opts.empty:
            return None

        # Find ATM or slightly OTM strike
        if direction == "call":
            # Slightly OTM = strike just above price
            candidates = opts[opts["strike"] >= price * 0.98].head(5)
        else:
            # Slightly OTM put = strike just below price
            candidates = opts[opts["strike"] <= price * 1.02].tail(5)

        if candidates.empty:
            return None

        # Pick the strike closest to ATM
        candidates = candidates.copy()
        candidates["dist"] = abs(candidates["strike"] - price)
        best_row = candidates.loc[candidates["dist"].idxmin()]

        bid = float(best_row.get("bid", 0))
        ask = float(best_row.get("ask", 0))
        last = float(best_row.get("lastPrice", 0))
        volume = int(best_row.get("volume", 0)) if pd.notna(best_row.get("volume")) else 0
        oi = int(best_row.get("openInterest", 0)) if pd.notna(best_row.get("openInterest")) else 0
        iv = float(best_row.get("impliedVolatility", 0)) if pd.notna(best_row.get("impliedVolatility")) else 0

        # After hours, bid/ask may be 0 — use last price as fallback
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid if mid > 0 else 999
        elif last > 0:
            mid = last
            spread_pct = 0.05  # Assume ~5% spread estimate when market closed
        else:
            return None
        cost_per_contract = mid * 100  # Options are per 100 shares

        exp_date = datetime.strptime(best_exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days

        spread_limit = MAX_BID_ASK_SPREAD_PCT_RTH if is_market_hours() else MAX_BID_ASK_SPREAD_PCT_AH

        return {
            "strike": float(best_row["strike"]),
            "expiry": best_exp,
            "dte": dte,
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "mid": round(mid, 2),
            "last": round(last, 2),
            "spread_pct": round(spread_pct, 4),
            "cost_per_contract": round(cost_per_contract, 2),
            "volume": volume,
            "open_interest": oi,
            "implied_vol": round(iv, 4),
            "direction": direction,
            "liquid": spread_pct <= spread_limit,
            "after_hours": not is_market_hours(),
            "within_budget": cost_per_contract <= MAX_CONTRACT_COST,
        }

    except Exception as e:
        print(f"  [WARN] Options data for {ticker_str}: {e}")
        return None


# ═══════════════════════════════════════════
# Feature 3: Momentum Decay Detection
# ═══════════════════════════════════════════

def assess_momentum_health(mom_5d: float, mom_20d: float, rsi: float) -> dict:
    """Assess whether momentum is fresh, steady, or exhausting."""
    # Momentum ratio: 5d vs 20d/4 (normalized to same period basis)
    mom_20d_norm = mom_20d / 4  # Approximate per-5d rate
    decaying = mom_5d < mom_20d_norm if mom_20d > 0 else False

    overbought = rsi > 72

    if mom_5d > mom_20d_norm * 1.5 and not overbought:
        health = "fresh_breakout"
        health_score = 1.0
    elif not decaying and not overbought:
        health = "steady_trend"
        health_score = 0.7
    elif decaying and not overbought:
        health = "decaying_momentum"
        health_score = 0.4
    elif overbought and not decaying:
        health = "overbought_risk"
        health_score = 0.3
    else:
        health = "exhausting_trend"
        health_score = 0.2

    return {
        "health": health,
        "health_score": round(health_score, 2),
        "decaying": decaying,
        "overbought": overbought,
        "mom_ratio": round(mom_5d / mom_20d_norm, 2) if mom_20d_norm != 0 else 0,
    }


# ═══════════════════════════════════════════
# Feature 5: VIX Regime Filter
# ═══════════════════════════════════════════

def get_vix_regime() -> dict:
    """Get current VIX level and determine regime."""
    try:
        vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False, timeout=10)
        if vix_data.empty:
            return {"vix": None, "regime": "unknown", "note": "Could not fetch VIX"}

        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.get_level_values(0)

        vix = float(vix_data["Close"].iloc[-1])

        if vix > 25:
            regime = "high_vol"
            note = "VIX > 25: favor oversold bounces over momentum. Mean reversion works better."
            preferred_setups = ["B_oversold_bounce", "D_post_earnings_bounce", "C_flow_divergence"]
        elif vix < 15:
            regime = "low_vol"
            note = "VIX < 15: low-vol environment. Smaller expected moves, use tighter stops."
            preferred_setups = ["A_momentum_continuation", "D_post_earnings_bounce", "C_flow_divergence"]
        else:
            regime = "normal"
            note = "VIX 15-25: normal regime. All play types valid."
            preferred_setups = ["A_momentum_continuation", "B_oversold_bounce", "D_post_earnings_bounce", "C_flow_divergence"]

        return {"vix": round(vix, 2), "regime": regime, "note": note, "preferred_setups": preferred_setups}
    except Exception:
        return {"vix": None, "regime": "unknown", "note": "VIX fetch failed", "preferred_setups": []}


# ═══════════════════════════════════════════
# Feature 5b: VIX Options Analysis (HC #747)
# ═══════════════════════════════════════════

def analyze_vix_options(vix_level: float | None) -> dict | None:
    """
    Analyze VIX level and suggest VIX option plays for agentic account.
    HC #747: VIX options are in scope — calls for hedging when low, puts/spreads for income when high.
    """
    if vix_level is None:
        return None

    result = {"vix": vix_level, "plays": []}

    try:
        # Get VIX option chain
        vix_ticker = yf.Ticker("^VIX")
        expirations = vix_ticker.options
        if not expirations:
            return None

        # Find nearest expiry 14-30 days out
        today = pd.Timestamp.now().normalize()
        best_exp = None
        for exp_str in expirations:
            exp_date = pd.Timestamp(exp_str)
            dte = (exp_date - today).days
            if 14 <= dte <= 45:
                best_exp = exp_str
                break

        if not best_exp:
            # Fall back to nearest available
            for exp_str in expirations:
                exp_date = pd.Timestamp(exp_str)
                if (exp_date - today).days > 7:
                    best_exp = exp_str
                    break

        if not best_exp:
            return None

        exp_date = pd.Timestamp(best_exp)
        dte = (exp_date - today).days
        chain = vix_ticker.option_chain(best_exp)

        # VIX < 15 → BUY VIX CALLS (cheap crash insurance)
        if vix_level < 15:
            # Look for OTM calls around VIX + 5-10 points (cheap lottery/hedge)
            target_strike = round(vix_level + 5)
            calls = chain.calls
            if not calls.empty:
                mask = (calls["strike"] >= target_strike - 1) & (calls["strike"] <= target_strike + 3)
                candidates = calls[mask].copy()
                if not candidates.empty:
                    # Pick cheapest with decent OI
                    candidates = candidates[candidates["openInterest"] > 100] if "openInterest" in candidates.columns else candidates
                    if not candidates.empty:
                        pick = candidates.iloc[0]
                        cost = float(pick.get("lastPrice", pick.get("ask", 0))) * 100
                        result["plays"].append({
                            "type": "VIX_CALL_HEDGE",
                            "direction": "BUY CALL",
                            "strike": float(pick["strike"]),
                            "expiry": best_exp,
                            "dte": dte,
                            "cost": round(cost, 0),
                            "reason": f"VIX at {vix_level:.1f} — cheap crash insurance. If VIX spikes to 25+, this 5-10x's.",
                            "risk": "Max loss = premium paid",
                            "validated": False,  # No backtest yet, theoretical edge from mean-reversion
                        })

        # VIX > 25 → VIX PUT or BEAR CALL SPREAD (mean-reversion play)
        elif vix_level > 25:
            # VIX mean-reverts from spikes. Buy ATM puts or sell call spreads.
            target_strike = round(vix_level)
            puts = chain.puts
            if not puts.empty:
                mask = (puts["strike"] >= target_strike - 2) & (puts["strike"] <= target_strike + 2)
                candidates = puts[mask].copy()
                if not candidates.empty:
                    candidates = candidates[candidates["openInterest"] > 50] if "openInterest" in candidates.columns else candidates
                    if not candidates.empty:
                        pick = candidates.iloc[0]
                        cost = float(pick.get("lastPrice", pick.get("ask", 0))) * 100
                        if cost <= 300:  # Budget constraint
                            result["plays"].append({
                                "type": "VIX_PUT_MEANREV",
                                "direction": "BUY PUT",
                                "strike": float(pick["strike"]),
                                "expiry": best_exp,
                                "dte": dte,
                                "cost": round(cost, 0),
                                "reason": f"VIX at {vix_level:.1f} — elevated, mean-reversion likely. VIX put profits as fear subsides.",
                                "risk": "Max loss = premium. VIX could stay elevated in crisis.",
                                "validated": False,
                            })

            # Also check bear call spread (defined risk income play)
            calls = chain.calls
            if not calls.empty:
                sell_strike = round(vix_level + 5)
                buy_strike = sell_strike + 5
                sell_mask = abs(calls["strike"] - sell_strike) <= 1
                buy_mask = abs(calls["strike"] - buy_strike) <= 1
                sell_calls = calls[sell_mask]
                buy_calls = calls[buy_mask]
                if not sell_calls.empty and not buy_calls.empty:
                    credit = (float(sell_calls.iloc[0].get("bid", 0)) - float(buy_calls.iloc[0].get("ask", 0))) * 100
                    if credit > 50:
                        result["plays"].append({
                            "type": "VIX_BEAR_CALL_SPREAD",
                            "direction": "SELL CALL SPREAD",
                            "sell_strike": float(sell_calls.iloc[0]["strike"]),
                            "buy_strike": float(buy_calls.iloc[0]["strike"]),
                            "expiry": best_exp,
                            "dte": dte,
                            "credit": round(credit, 0),
                            "max_loss": round((float(buy_calls.iloc[0]["strike"]) - float(sell_calls.iloc[0]["strike"])) * 100 - credit, 0),
                            "reason": f"VIX at {vix_level:.1f} — sell elevated vol premium. VIX rarely stays above {sell_strike} for long.",
                            "risk": "Max loss = width - credit. VIX could spike further in crisis.",
                            "validated": False,
                        })

        # VIX 15-25 (normal) → No strong VIX play, but note contango opportunity
        else:
            # Check VIX term structure for contango play
            result["plays"].append({
                "type": "VIX_NOTE",
                "reason": f"VIX at {vix_level:.1f} — normal range. No strong VIX option play. Watch for move below 14 (buy calls) or above 25 (buy puts).",
                "validated": False,
            })

    except Exception as e:
        result["error"] = str(e)

    return result if result.get("plays") else None


# ═══════════════════════════════════════════
# Feature 6: Trade Tracking
# ═══════════════════════════════════════════

def load_trade_log() -> dict:
    """Load existing trade log or create new one."""
    if TRADE_LOG_FILE.exists():
        try:
            return json.loads(TRADE_LOG_FILE.read_text())
        except Exception:
            pass
    return {"trades": [], "stats": {"total": 0, "wins": 0, "losses": 0, "open": 0}}


def save_trade_log(log: dict):
    TRADE_LOG_FILE.write_text(json.dumps(log, indent=2))


def log_recommendation(log: dict, ticker: str, setup_type: str, strike: float,
                       expiry: str, estimated_cost: float, confidence: float,
                       direction: str):
    """Log a trade recommendation (deduped — same ticker+strike+expiry = skip)."""
    today = datetime.now(ET).strftime("%Y-%m-%d")

    # Dedup: don't log if same ticker+strike+expiry already recommended today
    for t in log["trades"]:
        if (t["ticker"] == ticker and t.get("strike") == strike
                and t.get("expiry") == expiry and t.get("entry_date") == today):
            return  # Already logged today

    entry = {
        "id": len(log["trades"]) + 1,
        "ticker": ticker,
        "setup_type": setup_type,
        "direction": direction,
        "entry_date": today,
        "strike": strike,
        "expiry": expiry,
        "estimated_cost": round(estimated_cost, 2),
        "confidence": round(confidence, 3),
        "status": "recommended",
        "exit_date": None,
        "exit_pnl": None,
    }
    log["trades"].append(entry)
    log["stats"]["total"] += 1
    log["stats"]["open"] += 1


def get_historical_stats(log: dict) -> dict:
    """Compute win rate and stats from past trades."""
    closed = [t for t in log["trades"] if t.get("status") == "closed" and t.get("exit_pnl") is not None]
    if not closed:
        return {"win_rate": None, "avg_pnl": None, "total_closed": 0, "wins": 0, "losses": 0}

    wins = [t for t in closed if t["exit_pnl"] > 0]
    losses = [t for t in closed if t["exit_pnl"] <= 0]
    avg_pnl = sum(t["exit_pnl"] for t in closed) / len(closed)

    return {
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "avg_pnl": round(avg_pnl, 2),
        "total_closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
    }


def get_last_trade_sector(log: dict) -> str | None:
    """Get sector of last recommended/executed trade."""
    for trade in reversed(log["trades"]):
        ticker = trade.get("ticker")
        if ticker and ticker in SECTOR_MAP:
            return SECTOR_MAP[ticker]
    return None


# ═══════════════════════════════════════════
# Feature 4: Sector Concentration Check
# ═══════════════════════════════════════════

def check_sector_concentration(results: list, trade_log: dict) -> dict:
    """Check if top setups are concentrated in one sector."""
    top3 = results[:3]
    if len(top3) < 2:
        return {"concentrated": False, "dominant_sector": None, "warning": None}

    sectors = [SECTOR_MAP.get(r["ticker"], "Unknown") for r in top3]
    from collections import Counter
    counts = Counter(sectors)
    dominant = counts.most_common(1)[0]

    concentrated = dominant[1] >= 2  # 2+ of top 3 in same sector

    last_sector = get_last_trade_sector(trade_log)
    doubling_up = last_sector and last_sector == dominant[0]

    warning = None
    if concentrated:
        warning = f"Top setups concentrated in {dominant[0]} ({dominant[1]}/3). Consider diversifying."
    if doubling_up:
        extra = f" Last trade was also {last_sector} — avoid doubling up."
        warning = (warning + extra) if warning else f"Last trade was {last_sector} — watch concentration."

    return {
        "concentrated": concentrated,
        "dominant_sector": dominant[0],
        "sectors": sectors,
        "doubling_up": doubling_up,
        "warning": warning,
    }


# ═══════════════════════════════════════════
# Feature 7: Trade Plan Builder
# ═══════════════════════════════════════════

def build_trade_plan(result: dict, option_data: dict | None, momentum: dict) -> dict:
    """Build a complete trade plan for a setup."""
    price = result["price"]
    atr_pct = result["atr_pct"]
    best_setup = max(result["setups"], key=lambda s: s["score"])
    setup_type = best_setup["type"]

    # Direction
    if setup_type == "B_oversold_bounce":
        direction = "call"  # Buying calls on oversold bounce
    elif setup_type == "A_momentum_continuation":
        direction = "call" if result["mom_5d"] > 0 else "put"
    else:
        direction = "call"  # Default for flow divergence (bullish signal)

    plan = {
        "ticker": result["ticker"],
        "direction": direction,
        "setup_type": setup_type,
        "setup_reason": best_setup["reason"],
        "score": best_setup["score"],
        "sector": SECTOR_MAP.get(result["ticker"], "Unknown"),
        "momentum_health": momentum["health"],
        "momentum_score": momentum["health_score"],
    }

    if option_data:
        plan["strike"] = option_data["strike"]
        plan["expiry"] = option_data["expiry"]
        plan["dte"] = option_data["dte"]
        plan["estimated_cost"] = option_data["cost_per_contract"]
        plan["bid"] = option_data["bid"]
        plan["ask"] = option_data["ask"]
        plan["mid"] = option_data["mid"]
        plan["spread_pct"] = option_data["spread_pct"]
        plan["implied_vol"] = option_data["implied_vol"]
        plan["volume"] = option_data["volume"]
        plan["open_interest"] = option_data["open_interest"]
        plan["liquid"] = option_data["liquid"]
        plan["within_budget"] = option_data["within_budget"]

        # Stop loss: close if option drops 40% from entry
        plan["stop_loss_option_pct"] = -40
        plan["stop_loss_option_price"] = round(option_data["mid"] * 0.60, 2)

        # Take profit: 80%+ gain
        plan["take_profit_option_pct"] = 80
        plan["take_profit_option_price"] = round(option_data["mid"] * 1.80, 2)

        # Bull call spread alternative for oversold bounces (better for small accounts)
        # Analysis shows: same +22.7% expected return, max loss -17.5% vs -37%, cheaper entry
        if setup_type in ("B_oversold_bounce", "D_post_earnings_bounce"):
            spread_strike = round(price * 1.05, 2)  # OTM leg at +5%
            # Estimate spread cost as ~50% of naked call (conservative)
            spread_cost_est = round(option_data["cost_per_contract"] * 0.50, 2)
            plan["spread_alternative"] = {
                "type": "bull_call_spread",
                "buy_strike": option_data["strike"],
                "sell_strike": spread_strike,
                "estimated_cost": spread_cost_est,
                "advantage": "Lower risk (-17.5% max loss vs -37%), same expected return",
                "recommended_for": "accounts under $1000",
            }

        # Underlying stop/target based on ATR
        atr_dollar = price * atr_pct / 100
        if direction == "call":
            plan["stop_loss_underlying"] = round(price - 1.5 * atr_dollar, 2)
            plan["take_profit_underlying"] = round(price + 2.5 * atr_dollar, 2)
        else:
            plan["stop_loss_underlying"] = round(price + 1.5 * atr_dollar, 2)
            plan["take_profit_underlying"] = round(price - 2.5 * atr_dollar, 2)

        # Max hold: oversold bounce = 5d, post-earnings bounce = 10d (backtest optimal), others = dte - 7
        if setup_type == "B_oversold_bounce":
            max_hold_days = 5  # Backtest-validated: 5-day hold is optimal
        elif setup_type == "D_post_earnings_bounce":
            max_hold_days = 10  # Backtest-validated: 10-day hold passes all gates
        else:
            max_hold_days = max(option_data["dte"] - 7, 3)
        plan["max_hold_days"] = max_hold_days
        plan["exit_by_date"] = (datetime.now(ET).date() + timedelta(days=max_hold_days)).strftime("%Y-%m-%d")

        # Risk/reward ratio
        risk = option_data["mid"] * 0.40  # Max loss = 40%
        reward = option_data["mid"] * 0.80  # Target gain = 80%
        plan["risk_reward_ratio"] = round(reward / risk, 2) if risk > 0 else 0

    else:
        plan["strike"] = None
        plan["expiry"] = None
        plan["estimated_cost"] = None
        plan["note"] = "No suitable option chain data available"

    return plan


# ═══════════════════════════════════════════
# Main Analysis
# ═══════════════════════════════════════════

def analyze_ticker(ticker: str) -> dict | None:
    """Compute all indicators and check for setups."""
    try:
        df = yf.download(ticker, period="120d", interval="1d", progress=False, timeout=10)
        if df.empty or len(df) < 60:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        price = float(close.iloc[-1])
        rsi = float(compute_rsi(close).iloc[-1])
        mfi = float(compute_mfi(high, low, close, volume).iloc[-1])

        obv = compute_obv(close, volume)
        obv_slope = float(np.polyfit(range(10), obv.iloc[-10:].values, 1)[0])
        obv_rising = obv_slope > 0

        mom_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0
        mom_20d = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0
        mom_60d = float((close.iloc[-1] / close.iloc[-61] - 1) * 100) if len(close) >= 61 else 0

        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.rolling(len(close)).mean().iloc[-1])

        above_sma20 = price > sma20
        above_sma50 = price > sma50
        above_sma200 = price > sma200

        vol_ratio = float(volume.iloc[-1] / volume.iloc[-20:].mean()) if volume.iloc[-20:].mean() > 0 else 1.0
        bb_pos = float(compute_bollinger_position(close).iloc[-1])

        recent_high = float(high.iloc[-5:].max())
        pullback_pct = (recent_high - price) / recent_high * 100

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_pct = float(tr.rolling(14).mean().iloc[-1] / price * 100)

        # Resistance level (highest close in 60d)
        resistance = float(close.iloc[-60:].max()) if len(close) >= 60 else float(close.max())
        support = float(close.iloc[-60:].min()) if len(close) >= 60 else float(close.min())

        result = {
            "ticker": ticker,
            "price": round(price, 2),
            "rsi": round(rsi, 1),
            "mfi": round(mfi, 1),
            "obv_rising": obv_rising,
            "mom_5d": round(mom_5d, 2),
            "mom_20d": round(mom_20d, 2),
            "mom_60d": round(mom_60d, 2),
            "above_sma20": above_sma20,
            "above_sma50": above_sma50,
            "above_sma200": above_sma200,
            "vol_ratio": round(vol_ratio, 2),
            "bb_pos": round(bb_pos, 3),
            "pullback_pct": round(pullback_pct, 2),
            "atr_pct": round(atr_pct, 2),
            "resistance": round(resistance, 2),
            "support": round(support, 2),
            "sector": SECTOR_MAP.get(ticker, "Unknown"),
            "setups": [],
        }

        # ── Setup A: Momentum Continuation ──
        if (above_sma20 and above_sma50 and above_sma200
                and 55 <= rsi <= 70
                and mfi > 60
                and obv_rising
                and 1.0 <= pullback_pct <= 3.0):
            result["setups"].append({
                "type": "A_momentum_continuation",
                "score": round((rsi - 55) / 15 * 0.3 + (mfi - 60) / 40 * 0.3 + min(mom_20d / 10, 1) * 0.4, 3),
                "reason": f"Above all SMAs, RSI {rsi:.0f}, MFI {mfi:.0f}, OBV rising, {pullback_pct:.1f}% pullback",
            })

        # ── Setup B: Oversold Bounce (aligned with backtest v1 — p=0.01, 69% WR) ──
        # Backtest winner: "drop_10pct_1wk" — pure 10% weekly drop, no SMA filter
        # Two entry paths: (1) classic RSI<35 or (2) large recent drop (backtest-aligned)
        high_20d = float(high.iloc[-20:].max()) if len(high) >= 20 else float(high.max())
        drop_from_high_pct = (high_20d - price) / high_20d * 100
        # Weekly return (5 trading days) — matches backtest exactly
        weekly_ret_pct = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0
        large_weekly_drop = weekly_ret_pct <= -10.0  # Exact backtest criterion
        moderate_drop = drop_from_high_pct >= 8.0 and above_sma200  # Relaxed, but needs SMA200 safety
        classic_oversold = rsi < 35 and not above_sma20 and vol_ratio > 1.3 and above_sma200

        # Large weekly drops don't need SMA200 filter (backtest validated without it)
        if classic_oversold or moderate_drop or large_weekly_drop:
            # Score: deeper drop = higher score, vol spike bonus, RSI bonus
            drop_score = min(drop_from_high_pct / 15, 1.0) * 0.40  # 15% drop = max
            vol_score = min((vol_ratio - 1) / 2, 1) * 0.25
            rsi_score = max(0, (40 - rsi) / 40) * 0.20  # Lower RSI = more oversold
            sma200_score = (1 if above_sma200 else 0) * 0.15
            total_score = round(drop_score + vol_score + rsi_score + sma200_score, 3)

            reason_parts = []
            if large_weekly_drop:
                reason_parts.append(f"{weekly_ret_pct:+.1f}% weekly drop (backtest-matched)")
            if drop_from_high_pct >= 8:
                reason_parts.append(f"{drop_from_high_pct:.1f}% off 20d high")
            if rsi < 35:
                reason_parts.append(f"RSI {rsi:.0f} oversold")
            if vol_ratio > 1.3:
                reason_parts.append(f"vol {vol_ratio:.1f}x avg")
            if above_sma200:
                reason_parts.append("above 200 SMA")

            result["setups"].append({
                "type": "B_oversold_bounce",
                "score": total_score,
                "reason": ", ".join(reason_parts),
                "drop_from_high_pct": round(drop_from_high_pct, 1),
                "backtest_validated": True,
            })

        # ── Setup D: Post-Earnings Bounce (VALIDATED — all 5 gates pass) ──
        # Buy stocks that dropped 8%+ in 2 days after earnings, hold 10 days
        pe_drop = check_recent_earnings_drop(ticker, close)
        if pe_drop is not None:
            pe_score = min(abs(pe_drop["drop_2d_pct"]) / 20, 1.0) * 0.50  # Deeper = better
            pe_score += min(vol_ratio / 3, 1.0) * 0.25  # Volume confirmation
            pe_score += (1 if pe_drop["days_since"] <= 4 else 0.5) * 0.25  # Fresher = better
            result["setups"].append({
                "type": "D_post_earnings_bounce",
                "score": round(pe_score, 3),
                "reason": f"Dropped {pe_drop['drop_2d_pct']:.1f}% after earnings on {pe_drop['earnings_date']}, {pe_drop['days_since']}d ago",
                "drop_2d_pct": pe_drop["drop_2d_pct"],
                "earnings_date": pe_drop["earnings_date"],
                "backtest_validated": True,
            })

        # ── Setup C: Flow Divergence ──
        price_flat_or_down = mom_5d <= 0.5
        mfi_turning_up = mfi < 40 and mfi > float(compute_mfi(high, low, close, volume).iloc[-3])
        if obv_rising and price_flat_or_down and mfi_turning_up:
            result["setups"].append({
                "type": "C_flow_divergence",
                "score": round(0.3 + min(abs(mom_5d) / 5, 1) * 0.3 + (40 - mfi) / 40 * 0.4, 3),
                "reason": f"OBV rising while price flat/down ({mom_5d:+.1f}%), MFI turning up from {mfi:.0f}",
            })

        return result if result["setups"] else None

    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")
        return None


def run_scan():
    now = datetime.now(ET)
    print(f"=== Play Scanner v2 — {now.strftime('%Y-%m-%d %H:%M ET')} ===")
    print(f"Scanning {len(ALL_TICKERS)} tickers with enhanced filters...\n")

    # ── VIX Regime ──
    print("Checking VIX regime...")
    vix_info = get_vix_regime()
    print(f"  VIX: {vix_info['vix']} — Regime: {vix_info['regime']}")
    print(f"  {vix_info['note']}\n")

    # ── VIX Options Analysis (HC #747) ──
    print("Checking VIX options opportunities...")
    vix_options = analyze_vix_options(vix_info.get("vix"))
    if vix_options and vix_options.get("plays"):
        for play in vix_options["plays"]:
            if play["type"] != "VIX_NOTE":
                print(f"  📊 VIX PLAY: {play['direction']} {play.get('strike', '')} exp {play.get('expiry', '')} — ${play.get('cost', play.get('credit', '?'))}")
                print(f"     {play['reason']}")
            else:
                print(f"  {play['reason']}")
    else:
        print("  No VIX option plays at current level.\n")

    # ── Load trade log ──
    trade_log = load_trade_log()
    hist_stats = get_historical_stats(trade_log)

    # ── Scan tickers ──
    print("Scanning tickers for setups...")
    raw_results = []
    for i, ticker in enumerate(ALL_TICKERS):
        if i % 10 == 0 and i > 0:
            print(f"  Scanned {i}/{len(ALL_TICKERS)}...")
        r = analyze_ticker(ticker)
        if r:
            raw_results.append(r)

    print(f"  Found {len(raw_results)} raw setups\n")

    # ── VIX Regime Filter (Feature 5) ──
    if vix_info["regime"] == "high_vol" and vix_info.get("preferred_setups"):
        preferred = vix_info["preferred_setups"]
        for r in raw_results:
            r["setups"] = [s for s in r["setups"] if s["type"] in preferred]
        raw_results = [r for r in raw_results if r["setups"]]
        print(f"  After VIX regime filter: {len(raw_results)} setups\n")

    # ── Score and rank (with backtest-validated confidence multipliers) ──
    # Backtest v1 (2026-07-24): Oversold bounce is the ONLY setup that passes
    # permutation test (p=0.01, 69% WR, PF 3.2). Momentum continuation and
    # flow divergence fail permutation — not significantly better than random.
    BACKTEST_MULTIPLIER = {
        "B_oversold_bounce": 1.5,       # VALIDATED (p=0.01) — boost score
        "D_post_earnings_bounce": 1.5,  # VALIDATED (all 5 gates) — boost score
        "C_flow_divergence": 0.8,       # marginal — slight discount
        "A_momentum_continuation": 0.6, # FAILS perm — significant discount
    }
    # High-vol VIX boost: oversold bounces are 89% WR in bear markets (vs 69% overall)
    VIX_HIGH_BOUNCE_BOOST = 1.3 if vix_info.get("regime") == "high_vol" else 1.0

    for r in raw_results:
        for s in r["setups"]:
            mult = BACKTEST_MULTIPLIER.get(s["type"], 1.0)
            # Extra boost for oversold bounces when VIX > 25 (89% WR in bear markets)
            if s["type"] == "B_oversold_bounce":
                mult *= VIX_HIGH_BOUNCE_BOOST
            s["score"] = round(s["score"] * mult, 3)
            s["backtest_validated"] = s["type"] in ("B_oversold_bounce", "D_post_earnings_bounce")

        # ── Confluence bonus: multiple validated signals = higher conviction ──
        validated_types = [s["type"] for s in r["setups"] if s.get("backtest_validated")]
        r["confluence_count"] = len(validated_types)
        if len(validated_types) >= 2:
            # Double-confirmed signal (e.g., oversold bounce + post-earnings bounce)
            # Boost all scores by 50% — this is the highest-conviction setup possible
            for s in r["setups"]:
                s["score"] = round(s["score"] * 1.5, 3)
            r["confluence_note"] = f"DOUBLE CONFIRMED: {' + '.join(validated_types)}"
            print(f"  🎯 {r['ticker']}: CONFLUENCE — {r['confluence_note']}")

        r["best_score"] = max(s["score"] for s in r["setups"])
    raw_results.sort(key=lambda x: x["best_score"], reverse=True)

    # ── Process top candidates with enhanced analysis ──
    print("Running enhanced analysis on top candidates...")
    top_candidates = raw_results[:8]  # Analyze more, show top 5
    enhanced_results = []

    for r in top_candidates:
        ticker = r["ticker"]
        price = r["price"]
        best_setup = max(r["setups"], key=lambda s: s["score"])

        print(f"\n  Analyzing {ticker}...")

        # Feature 1: Earnings check
        earnings = check_earnings_proximity(ticker)
        r["earnings"] = earnings
        if earnings["has_upcoming_earnings"]:
            days_until = earnings.get("days_until", 999)
            print(f"    EARNINGS WARNING: {ticker} reports in {days_until} days ({earnings['earnings_date']})")
            r["earnings_risk"] = True

            # Hard block: oversold bounce + earnings within 7 days = IV crush will kill trade
            # Exception: post-earnings bounce specifically trades AFTER earnings (IV is already crushed)
            if best_setup["type"] == "B_oversold_bounce" and days_until <= 7:
                print(f"    BLOCKED: oversold bounce with earnings in {days_until}d — IV crush risk too high")
                continue
            if best_setup["type"] == "D_post_earnings_bounce":
                pass  # Post-earnings bounce is EXPECTED to have nearby earnings — that's the signal
        else:
            r["earnings_risk"] = False

        # Feature 3: Momentum health
        momentum = assess_momentum_health(r["mom_5d"], r["mom_20d"], r["rsi"])
        r["momentum"] = momentum

        # Downgrade score if momentum decaying
        if momentum["decaying"]:
            for s in r["setups"]:
                s["score"] = round(s["score"] * 0.7, 3)
            r["best_score"] = max(s["score"] for s in r["setups"])
            print(f"    Momentum decaying — score downgraded")

        if momentum["overbought"]:
            for s in r["setups"]:
                s["score"] = round(s["score"] * 0.8, 3)
            r["best_score"] = max(s["score"] for s in r["setups"])
            print(f"    RSI > 72 overbought — score downgraded")

        # Feature 2: Real options pricing
        direction = "call"
        if best_setup["type"] == "A_momentum_continuation" and r["mom_5d"] < 0:
            direction = "put"

        # Oversold bounce: shorter DTE (10-14d) — backtest shows 5-day hold is optimal
        # DTE: oversold bounce = 14d (5d hold), post-earnings = 21d (10d hold), others = 21d
        if best_setup["type"] == "B_oversold_bounce":
            target_dte = 14
        elif best_setup["type"] == "D_post_earnings_bounce":
            target_dte = 21  # 10-day hold needs more time value
        else:
            target_dte = 21
        option_data = get_option_pricing(ticker, price, direction, target_dte=target_dte)
        r["option_data"] = option_data

        if option_data:
            if not option_data["liquid"]:
                print(f"    Bid-ask spread {option_data['spread_pct']:.0%} > 20% — ILLIQUID, skipping")
                continue
            if not option_data["within_budget"]:
                print(f"    Cost ${option_data['cost_per_contract']:.0f} > $300 budget — skipping")
                continue
            print(f"    Option: ${option_data['strike']} {direction} exp {option_data['expiry']} @ ${option_data['mid']:.2f} ({option_data['dte']}d)")
        else:
            print(f"    No suitable options data — skipping for actionable recommendation")
            # Still include but flag
            r["no_options_data"] = True

        # Feature 7: Build trade plan
        trade_plan = build_trade_plan(r, option_data, momentum)
        r["trade_plan"] = trade_plan

        enhanced_results.append(r)

    # Take top 5
    enhanced_results.sort(key=lambda x: x["best_score"], reverse=True)
    top5 = enhanced_results[:5]

    # ── Feature 4: Sector concentration check ──
    concentration = check_sector_concentration(top5, trade_log)

    # ── Print results ──
    print(f"\n{'='*70}")
    print(f"PLAY SCANNER v2 RESULTS — {now.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*70}")
    print(f"VIX: {vix_info['vix']} ({vix_info['regime']}) | Setups found: {len(raw_results)}")

    if hist_stats["total_closed"] > 0:
        print(f"Historical: {hist_stats['wins']}/{hist_stats['total_closed']} wins ({hist_stats['win_rate']}% WR), avg P&L ${hist_stats['avg_pnl']}")

    if concentration["warning"]:
        print(f"\n⚠ CONCENTRATION: {concentration['warning']}")

    print(f"\n--- Top {len(top5)} Actionable Setups ---\n")

    for i, r in enumerate(top5, 1):
        tp = r.get("trade_plan", {})
        best_setup = max(r["setups"], key=lambda s: s["score"])

        print(f"#{i} {r['ticker']} ({r['sector']}) — {best_setup['type']}")
        print(f"   Price: ${r['price']} | RSI: {r['rsi']} | Mom 5d: {r['mom_5d']:+.1f}%")
        print(f"   Momentum: {r.get('momentum', {}).get('health', 'n/a')} (score: {r.get('momentum', {}).get('health_score', 'n/a')})")
        print(f"   Reason: {best_setup['reason']}")

        if r.get("earnings_risk"):
            print(f"   ⚠ EARNINGS in {r['earnings']['days_until']}d ({r['earnings']['earnings_date']}) — IV crush risk!")

        if tp.get("strike"):
            print(f"   TRADE PLAN:")
            print(f"     {tp['direction'].upper()} ${tp['strike']} exp {tp['expiry']} ({tp.get('dte', '?')}d)")
            print(f"     Cost: ${tp['estimated_cost']:.0f}/contract | Spread: {tp.get('spread_pct', 0):.1%}")
            print(f"     Stop: close at ${tp['stop_loss_option_price']:.2f} (-40%) | Target: ${tp['take_profit_option_price']:.2f} (+80%)")
            print(f"     Underlying stop: ${tp.get('stop_loss_underlying', '?')} | Target: ${tp.get('take_profit_underlying', '?')}")
            print(f"     Max hold: {tp.get('max_hold_days', '?')}d | Exit by: {tp.get('exit_by_date', '?')}")
            print(f"     Risk/Reward: {tp.get('risk_reward_ratio', '?')}:1")
            print(f"     Volume: {tp.get('volume', 0)} | OI: {tp.get('open_interest', 0)} | IV: {tp.get('implied_vol', 0):.0%}")

            # Log recommendation
            log_recommendation(
                trade_log, r["ticker"], best_setup["type"],
                tp["strike"], tp["expiry"], tp["estimated_cost"],
                best_setup["score"], tp["direction"]
            )
        elif r.get("no_options_data"):
            print(f"   (No actionable option chain data available)")
        print()

    if not top5:
        print("  No actionable setups found today.\n")

    # ── Save state ──
    state = {
        "scan_time": now.isoformat(),
        "scanner_version": "v2",
        "total_scanned": len(ALL_TICKERS),
        "setups_found": len(raw_results),
        "vix": vix_info,
        "vix_options": vix_options,
        "sector_concentration": concentration,
        "historical_stats": hist_stats,
        "top_5": [
            {
                "ticker": r["ticker"],
                "price": r["price"],
                "sector": r.get("sector"),
                "rsi": r["rsi"],
                "mfi": r["mfi"],
                "mom_5d": r["mom_5d"],
                "mom_20d": r["mom_20d"],
                "momentum_health": r.get("momentum", {}).get("health"),
                "earnings_risk": r.get("earnings_risk", False),
                "earnings_date": r.get("earnings", {}).get("earnings_date"),
                "best_setup": max(r["setups"], key=lambda s: s["score"]),
                "trade_plan": r.get("trade_plan"),
            }
            for r in top5
        ],
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))

    # Save trade log
    save_trade_log(trade_log)

    # Append history
    history_entry = {
        "scan_time": now.isoformat(),
        "version": "v2",
        "vix": vix_info.get("vix"),
        "vix_regime": vix_info.get("regime"),
        "setups_found": len(raw_results),
        "top_5_tickers": [r["ticker"] for r in top5],
        "top_5_types": [max(r["setups"], key=lambda s: s["score"])["type"] for r in top5] if top5 else [],
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(history_entry) + "\n")

    print(f"State saved. Trade log updated ({len(trade_log['trades'])} total recommendations).")

    # ── Discord alert with top setups ──
    if top5:
        lines = [f"**Pre-Market Scanner** ({now.strftime('%b %d')}) — VIX {vix_info.get('vix', '?')} ({vix_info.get('regime', '?')})"]
        for i, r in enumerate(top5, 1):
            best = max(r["setups"], key=lambda s: s["score"])
            tp = r.get("trade_plan", {})
            earn_flag = " ⚠️EARNINGS" if r.get("earnings_risk") else ""
            mom = r.get("momentum", {}).get("health", "")
            validated = "✅" if best.get("backtest_validated") or best["type"] in ("B_oversold_bounce", "D_post_earnings_bounce") else "❓"
            setup_name = best['type'].replace('_', ' ')
            confluence = " 🎯DOUBLE" if r.get("confluence_count", 0) >= 2 else ""
            line = f"**#{i} {r['ticker']}** {validated} {setup_name}{earn_flag}{confluence}"
            if tp.get("strike"):
                hold_note = f" | hold {tp.get('max_hold_days', '?')}d" if best["type"] in ("B_oversold_bounce", "D_post_earnings_bounce") else ""
                line += f"\n   {tp['direction']} ${tp['strike']} exp {tp['expiry']} — ${tp['estimated_cost']:.0f}/contract"
                line += f" | R:R {tp.get('risk_reward_ratio', '?')}:1 | {mom}{hold_note}"
                # Add spread alternative for oversold bounces
                spread = tp.get("spread_alternative")
                if spread:
                    line += f"\n   💡 Spread alt: buy ${spread['buy_strike']}/sell ${spread['sell_strike']} — ~${spread['estimated_cost']:.0f}"
            lines.append(line)
        lines.append("_✅ = backtest validated (p<0.05) | ❓ = unvalidated_")
        if hist_stats.get("total_recommended", 0) > 0:
            wr = hist_stats.get("win_rate_pct", 0)
            lines.append(f"_Track record: {hist_stats['total_recommended']} picks, {wr:.0f}% WR_")

        # Add VIX options play if available (HC #747)
        if vix_options and vix_options.get("plays"):
            actionable_vix = [p for p in vix_options["plays"] if p["type"] != "VIX_NOTE"]
            if actionable_vix:
                lines.append("")
                lines.append("**VIX Options:**")
                for play in actionable_vix:
                    if "credit" in play:
                        lines.append(f"  {play['direction']} {play.get('sell_strike', '')}/{play.get('buy_strike', '')} exp {play['expiry']} — ${play['credit']} credit (max loss ${play.get('max_loss', '?')})")
                    else:
                        lines.append(f"  {play['direction']} ${play['strike']} exp {play['expiry']} — ${play['cost']}/contract")
                    lines.append(f"  _{play['reason']}_")

        _send_discord("\n".join(lines))

    print("Done.")


if __name__ == "__main__":
    run_scan()
