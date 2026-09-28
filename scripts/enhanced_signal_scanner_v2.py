#!/usr/bin/env python3
"""
Enhanced Signal Scanner v2 — Unified Daily Strategy Scanner
============================================================
Checks ALL validated and supplementary strategy signals across 10 strategies,
applies a kill switch, computes confluence, and produces actionable trade
recommendations sized for a small Robinhood account.

Usage:
    python3 scripts/enhanced_signal_scanner_v2.py          # full run, saves state
    python3 scripts/enhanced_signal_scanner_v2.py --dry     # print only, no state files

Validated Strategies (proven alpha):
  1. LGBM Sector Rotation        (Sharpe 2.25)
  2. IV Run-Up / Pre-Earnings     (Sharpe 2.27)
  3. PEAD / Post-Earnings Drift   (Sharpe 1.51)
  4. SPY Iron Condor              (Sharpe 3.55)
  5. Contrarian Sector Reversion  (Sharpe 0.975)

  6. Consec. Days Reversal       (Sharpe 1.15, 5/5 gates + 4/6 adversarial)

Supplementary Signals:
  7. Stock Split Pre-Announcement (Sharpe 1.508)
  8. VIX Spike Fade              (Sharpe 0.841)
  9. Fear Signal / Options Flow   (Sharpe 1.4-1.8)
 10. Drawdown Recovery            (Sharpe 1.621)
 11. Sector Momentum Crash Prot.

Dependencies: yfinance, numpy, pandas (all pre-installed).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
BROAD_MARKET = ["SPY", "QQQ", "IWM"]
VIX_TICKER = "^VIX"
GROWTH_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX",
    "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "UBER", "LYFT", "COIN",
    "RBLX", "DDOG", "TTD", "SHOP", "NET", "ROKU",
]

ALL_TICKERS = SECTOR_ETFS + BROAD_MARKET + GROWTH_STOCKS  # VIX handled separately

SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "CRM": "XLK",
    "DDOG": "XLK", "NET": "XLK", "TTD": "XLK", "SHOP": "XLK", "PLTR": "XLK",
    "META": "XLC", "GOOGL": "XLC", "NFLX": "XLC", "SNAP": "XLC", "PINS": "XLC",
    "ROKU": "XLC",
    "AMZN": "XLY", "TSLA": "XLY",
    "COIN": "XLF", "HOOD": "XLF", "SOFI": "XLF",
    "UBER": "XLI", "LYFT": "XLI",
    "RBLX": "XLC",
}

SECTOR_NAMES = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services",
}

# Strategy backtested Sharpe ratios — used for confluence weighting
STRATEGY_SHARPE = {
    "lgbm_sector_rotation": 2.25,
    "iv_runup": 2.27,
    "pead": 1.51,
    "spy_iron_condor": 3.55,
    "contrarian_reversion": 0.975,
    "stock_split": 1.508,
    "vix_spike_fade": 0.841,
    "fear_signal": 1.60,       # midpoint of 1.4-1.8
    "drawdown_recovery": 1.621,
    "momentum_crash_protection": 1.0,  # override signal, nominal weight
    "consecutive_days_reversal": 1.15,  # 5/5 gates + 4/6 adversarial
    "sector_momentum_crash": 1.0,
}

# Account sizing
ACCOUNT_CASH = 667.73
MAX_PER_TRADE_SHARES = 250.0   # $200-300 range, use $250 default
MAX_PER_TRADE_OPTIONS = 125.0  # $100-150 range, use $125 default
OPTIONS_COMMISSION = 0.65      # per contract

STATE_DIR = Path("/home/jupiter/Lvl3Quant/state")
SIGNALS_FILE = STATE_DIR / "enhanced_signals_v2.json"
ALERT_FILE = STATE_DIR / "high_confidence_alert_v2.txt"

# ──────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────

def download_data() -> tuple[pd.DataFrame, pd.Series]:
    """Download 2 years of daily OHLCV for all tickers + VIX. Returns (prices_df, vix_series)."""
    end = datetime.now()
    start = end - timedelta(days=750)  # ~2 years of trading days

    print("Downloading market data (2 years)...")
    # Download all equity tickers at once
    raw = yf.download(ALL_TICKERS, start=start, end=end, progress=False, auto_adjust=True, threads=True)
    if raw.empty:
        print("ERROR: Failed to download market data.")
        sys.exit(1)

    # Download VIX separately (it's an index)
    vix_raw = yf.download(VIX_TICKER, start=start, end=end, progress=False, auto_adjust=True)

    # Extract close prices — handle both multi-level and single-level columns
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].copy()
        close.columns = ALL_TICKERS[:1]  # single ticker edge case

    # VIX close
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix = vix_raw["Close"].squeeze()
    else:
        vix = vix_raw["Close"].squeeze()

    # Flatten any remaining MultiIndex issues
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    if isinstance(vix.name, tuple):
        vix.name = "VIX"

    print(f"  Got {len(close)} trading days, {len(close.columns)} tickers, VIX {len(vix)} days.")
    return close, vix


# ──────────────────────────────────────────────────────────────────────────────
# KILL SWITCH
# ──────────────────────────────────────────────────────────────────────────────

def check_kill_switch(close: pd.DataFrame, vix: pd.Series) -> dict:
    """
    Kill switch logic:
      ACTIVE: VIX > 20 AND SPY < 50-SMA  -> pause momentum, allow contrarian/vix/recovery only
      CAUTION: Either condition alone      -> reduce position size 50%
      OFF: Neither condition               -> full speed ahead
    """
    current_vix = float(vix.dropna().iloc[-1])
    spy_close = close["SPY"].dropna()
    spy_current = float(spy_close.iloc[-1])
    spy_sma50 = float(spy_close.rolling(50).mean().iloc[-1])
    spy_vs_sma = spy_current - spy_sma50
    spy_vs_sma_pct = (spy_vs_sma / spy_sma50) * 100

    vix_high = current_vix > 20
    spy_below_sma = spy_current < spy_sma50

    if vix_high and spy_below_sma:
        status = "ACTIVE"
        effect = "PAUSE all momentum/rotation entries. ALLOW: contrarian, VIX fade, drawdown recovery only."
        size_mult = 0.0  # no new momentum
    elif vix_high or spy_below_sma:
        status = "CAUTION"
        reason = "VIX > 20" if vix_high else "SPY < 50-SMA"
        effect = f"Reduce position size 50% ({reason})."
        size_mult = 0.5
    else:
        status = "OFF"
        effect = "All systems go. Full position sizing."
        size_mult = 1.0

    return {
        "status": status,
        "effect": effect,
        "vix": round(current_vix, 2),
        "spy_price": round(spy_current, 2),
        "spy_sma50": round(spy_sma50, 2),
        "spy_vs_sma_pct": round(spy_vs_sma_pct, 2),
        "vix_elevated": vix_high,
        "spy_below_sma": spy_below_sma,
        "size_multiplier": size_mult,
    }


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE COMPUTATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def compute_returns(series: pd.Series, window: int) -> float:
    """Total return over last `window` trading days."""
    s = series.dropna()
    if len(s) < window + 1:
        return 0.0
    return float((s.iloc[-1] / s.iloc[-window - 1]) - 1)


def compute_sharpe(series: pd.Series, window: int) -> float:
    """Annualized Sharpe over last `window` days (daily returns)."""
    s = series.dropna()
    if len(s) < window + 1:
        return 0.0
    rets = s.pct_change().iloc[-window:]
    if rets.std() == 0:
        return 0.0
    return float((rets.mean() / rets.std()) * np.sqrt(252))


def compute_trend_slope(series: pd.Series, window: int) -> float:
    """Normalized OLS slope of log-price over last `window` days."""
    s = series.dropna()
    if len(s) < window:
        return 0.0
    y = np.log(s.iloc[-window:].values)
    x = np.arange(len(y))
    if len(y) < 2:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return float(slope * window)  # total log-return implied by slope


def compute_momentum_accel(series: pd.Series) -> float:
    """Momentum acceleration: ret_5d - ret_21d (positive = accelerating)."""
    return compute_returns(series, 5) - compute_returns(series, 21)


def compute_max_drawdown_from_high(series: pd.Series, window: int) -> float:
    """Current drawdown from rolling high over `window` days."""
    s = series.dropna()
    if len(s) < window:
        return 0.0
    rolling_high = s.iloc[-window:].max()
    current = s.iloc[-1]
    return float((current / rolling_high) - 1)


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL MODULES
# ──────────────────────────────────────────────────────────────────────────────

def signal_lgbm_sector_rotation(close: pd.DataFrame, kill: dict) -> list[dict]:
    """
    Strategy 1: LGBM Sector Rotation (Sharpe 2.25)
    Rank 11 sector ETFs by composite momentum score. Top 3 = BUY, bottom 3 = AVOID.
    """
    if kill["status"] == "ACTIVE":
        return []  # momentum strategy blocked by kill switch

    scores = {}
    for etf in SECTOR_ETFS:
        if etf not in close.columns:
            continue
        s = close[etf].dropna()
        if len(s) < 80:
            continue
        ret_21 = compute_returns(s, 21)
        ret_63 = compute_returns(s, 63)
        sharpe_63 = compute_sharpe(s, 63)
        trend_63 = compute_trend_slope(s, 63)
        mom_accel = compute_momentum_accel(s)

        composite = (ret_21 * 0.30 + ret_63 * 0.25 + sharpe_63 * 0.02 +
                     trend_63 * 0.15 + mom_accel * 0.10)
        # Note: sharpe is on a different scale, so we dampen its weight in the composite
        # The 0.02 is intentional — sharpe_63 is annualized (~1-3), other features are ~(-0.1 to 0.1)
        scores[etf] = {
            "composite": composite,
            "ret_21d": round(ret_21 * 100, 2),
            "ret_63d": round(ret_63 * 100, 2),
            "sharpe_63d": round(sharpe_63, 2),
            "trend_63d": round(trend_63 * 100, 2),
            "mom_accel": round(mom_accel * 100, 2),
        }

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1]["composite"], reverse=True)
    signals = []
    size_mult = kill["size_multiplier"]

    for i, (etf, info) in enumerate(ranked[:3]):
        confidence = max(30, min(85, 70 + int(info["composite"] * 500)))
        effective_size = MAX_PER_TRADE_SHARES * size_mult
        signals.append({
            "ticker": etf,
            "direction": "BUY",
            "confidence": confidence,
            "strategy": "lgbm_sector_rotation",
            "rationale": (f"Ranked #{i+1}/11 sectors. 21d ret: {info['ret_21d']}%, "
                         f"63d ret: {info['ret_63d']}%, Sharpe63: {info['sharpe_63d']}, "
                         f"Accel: {info['mom_accel']}%"),
            "suggested_action": f"BUY ~${effective_size:.0f} of {etf} ({SECTOR_NAMES.get(etf, etf)})",
            "hold_days": 21,
        })

    for i, (etf, info) in enumerate(ranked[-3:]):
        signals.append({
            "ticker": etf,
            "direction": "AVOID",
            "confidence": 50,
            "strategy": "lgbm_sector_rotation",
            "rationale": (f"Ranked #{len(ranked) - 2 + i}/11 sectors. 21d ret: {info['ret_21d']}%, "
                         f"63d ret: {info['ret_63d']}%"),
            "suggested_action": f"AVOID {etf} — weakest momentum",
            "hold_days": 0,
        })

    return signals


def signal_contrarian_reversion(close: pd.DataFrame, kill: dict) -> list[dict]:
    """
    Strategy 5: Contrarian Sector Reversion (Sharpe 0.975)
    Buy worst-performing sector ETF over 21d when it's starting to bounce (ret_5d > 0).
    Allowed even when kill switch is ACTIVE.
    """
    signals = []
    for etf in SECTOR_ETFS:
        if etf not in close.columns:
            continue
        s = close[etf].dropna()
        if len(s) < 30:
            continue

        ret_21 = compute_returns(s, 21)
        ret_5 = compute_returns(s, 5)

        if ret_21 < -0.05 and ret_5 > 0:
            # Deeper drawdown = higher confidence
            depth = abs(ret_21)
            confidence = max(35, min(80, int(depth * 800)))
            size = MAX_PER_TRADE_SHARES  # contrarian allowed at full size even in kill switch
            signals.append({
                "ticker": etf,
                "direction": "BUY",
                "confidence": confidence,
                "strategy": "contrarian_reversion",
                "rationale": (f"21d return: {ret_21*100:.1f}% (oversold), "
                             f"5d return: {ret_5*100:.1f}% (bouncing). "
                             f"Mean-reversion setup in {SECTOR_NAMES.get(etf, etf)}."),
                "suggested_action": f"BUY ~${size:.0f} of {etf} — contrarian bounce play",
                "hold_days": 15,
            })

    return signals


def signal_vix_spike_fade(close: pd.DataFrame, vix: pd.Series, kill: dict) -> list[dict]:
    """
    Strategy 7: VIX Spike Fade (Sharpe 0.841, perm p=0.032)
    After VIX spikes >25 then drops 10%+ from peak, buy SPY/QQQ.
    Allowed even when kill switch is ACTIVE.
    """
    v = vix.dropna()
    if len(v) < 15:
        return []

    current_vix = float(v.iloc[-1])
    vix_10d_high = float(v.iloc[-10:].max())

    # Check if VIX was >25 in the last 10 days
    vix_spiked = float(v.iloc[-10:].max()) > 25
    # Check if current VIX is >10% below its 10-day high
    vix_fading = current_vix < vix_10d_high * 0.90

    if not (vix_spiked and vix_fading):
        return []

    drop_pct = (1 - current_vix / vix_10d_high) * 100
    confidence = max(40, min(75, int(40 + drop_pct)))

    signals = []
    for ticker in ["SPY", "QQQ"]:
        signals.append({
            "ticker": ticker,
            "direction": "BUY",
            "confidence": confidence,
            "strategy": "vix_spike_fade",
            "rationale": (f"VIX spiked to {vix_10d_high:.1f} in last 10d, now at {current_vix:.1f} "
                         f"({drop_pct:.1f}% off peak). Fear subsiding."),
            "suggested_action": f"BUY ~${MAX_PER_TRADE_SHARES:.0f} of {ticker} — VIX fade trade, hold 5-10d",
            "hold_days": 7,
        })

    return signals


def signal_fear_signal(close: pd.DataFrame, vix: pd.Series, kill: dict) -> list[dict]:
    """
    Strategy 8: Fear Signal / Options Flow (Sharpe 1.4-1.8)
    Buy SPY/QQQ after VIX fear episodes: VIX > 22 AND VIX dropping 2+ consecutive days from peak >25.
    Allowed even when kill switch is ACTIVE.
    """
    v = vix.dropna()
    if len(v) < 15:
        return []

    current_vix = float(v.iloc[-1])
    vix_changes = v.diff()

    # VIX must be > 22
    if current_vix <= 22:
        return []

    # Need 2+ consecutive down days
    recent_changes = vix_changes.iloc[-3:]
    consec_down = all(float(c) < 0 for c in recent_changes.iloc[-2:])
    if not consec_down:
        return []

    # Check that VIX peaked > 25 recently (last 15 days)
    vix_15d_high = float(v.iloc[-15:].max())
    if vix_15d_high <= 25:
        return []

    confidence = max(45, min(80, int(50 + (vix_15d_high - current_vix) * 3)))

    signals = []
    for ticker in ["SPY", "QQQ"]:
        signals.append({
            "ticker": ticker,
            "direction": "BUY",
            "confidence": confidence,
            "strategy": "fear_signal",
            "rationale": (f"VIX peaked at {vix_15d_high:.1f}, now {current_vix:.1f} and dropping "
                         f"for 2+ days. Fear episode subsiding — buy the dip."),
            "suggested_action": f"BUY ~${MAX_PER_TRADE_SHARES:.0f} of {ticker} — fear fade, hold 5-10d",
            "hold_days": 7,
        })

    return signals


def signal_drawdown_recovery(close: pd.DataFrame, kill: dict) -> list[dict]:
    """
    Strategy 9: Drawdown Recovery (Sharpe 1.621, perm p=0.001)
    When sector ETF drops >10% from 63-day high and has 2 consecutive up days, buy.
    Allowed even when kill switch is ACTIVE.
    """
    signals = []
    for etf in SECTOR_ETFS:
        if etf not in close.columns:
            continue
        s = close[etf].dropna()
        if len(s) < 65:
            continue

        dd = compute_max_drawdown_from_high(s, 63)
        if dd > -0.10:
            continue  # not in a >10% drawdown

        # Check 2 consecutive up days
        recent = s.iloc[-3:]
        if len(recent) < 3:
            continue
        day1_up = float(recent.iloc[-2]) > float(recent.iloc[-3])
        day2_up = float(recent.iloc[-1]) > float(recent.iloc[-2])

        if not (day1_up and day2_up):
            continue

        depth = abs(dd)
        confidence = max(40, min(80, int(depth * 500)))

        signals.append({
            "ticker": etf,
            "direction": "BUY",
            "confidence": confidence,
            "strategy": "drawdown_recovery",
            "rationale": (f"Down {dd*100:.1f}% from 63-day high, now showing 2 consecutive up days. "
                         f"Recovery signal in {SECTOR_NAMES.get(etf, etf)}."),
            "suggested_action": f"BUY ~${MAX_PER_TRADE_SHARES:.0f} of {etf} — drawdown recovery",
            "hold_days": 20,
        })

    return signals


def signal_pead_monitor(close: pd.DataFrame, kill: dict) -> list[dict]:
    """
    Strategy 3: PEAD / Post-Earnings Drift (Sharpe 1.51, 40-day hold)
    Detect stocks that gapped up >3% in the last 5 trading days. Flag as PEAD entry.
    """
    if kill["status"] == "ACTIVE":
        return []  # momentum-like, blocked by kill switch

    signals = []
    size_mult = kill["size_multiplier"]

    for stock in GROWTH_STOCKS:
        if stock not in close.columns:
            continue
        s = close[stock].dropna()
        if len(s) < 10:
            continue

        # Check each of the last 5 days for a >3% gap up
        for lookback in range(1, 6):
            if len(s) < lookback + 2:
                continue
            day_ret = float(s.iloc[-lookback] / s.iloc[-lookback - 1]) - 1
            if day_ret > 0.03:
                sector_etf = SECTOR_MAP.get(stock, "SPY")
                effective_size = MAX_PER_TRADE_SHARES * size_mult
                confidence = max(40, min(70, int(40 + day_ret * 300)))

                signals.append({
                    "ticker": stock,
                    "direction": "BUY",
                    "confidence": confidence,
                    "strategy": "pead",
                    "rationale": (f"Gapped up {day_ret*100:.1f}% on day -{lookback} "
                                 f"(potential earnings beat). PEAD drift typically continues 40 trading days. "
                                 f"Sector ETF: {sector_etf}."),
                    "suggested_action": (f"BUY ~${effective_size:.0f} of {stock} "
                                        f"(or {sector_etf} for diversified exposure). Hold 40 trading days."),
                    "hold_days": 40,
                })

                # Also flag the sector ETF
                if sector_etf in SECTOR_ETFS:
                    signals.append({
                        "ticker": sector_etf,
                        "direction": "BUY",
                        "confidence": max(30, confidence - 15),
                        "strategy": "pead",
                        "rationale": (f"Sector ETF play on {stock} earnings gap (+{day_ret*100:.1f}%). "
                                     f"Diversified PEAD exposure via {SECTOR_NAMES.get(sector_etf, sector_etf)}."),
                        "suggested_action": f"BUY ~${effective_size:.0f} of {sector_etf} — sector PEAD play",
                        "hold_days": 40,
                    })
                break  # only flag the most recent gap for each stock

    return signals


def signal_momentum_crash_protection(close: pd.DataFrame, vix: pd.Series, kill: dict) -> list[dict]:
    """
    Strategy 10: Momentum Crash Protection
    If VIX > 25, flag ALL momentum positions as REDUCE/EXIT. Override signal.
    """
    v = vix.dropna()
    if len(v) < 1:
        return []

    current_vix = float(v.iloc[-1])
    if current_vix <= 25:
        return []

    signals = []
    for etf in SECTOR_ETFS:
        signals.append({
            "ticker": etf,
            "direction": "REDUCE",
            "confidence": 90,
            "strategy": "momentum_crash_protection",
            "rationale": f"VIX at {current_vix:.1f} (>25). Crash protection: reduce/exit all momentum positions.",
            "suggested_action": f"REDUCE or EXIT any momentum position in {etf}",
            "hold_days": 0,
        })

    return signals


def signal_spy_iron_condor(close: pd.DataFrame, vix: pd.Series, kill: dict) -> list[dict]:
    """
    Strategy 4: SPY Iron Condor (Sharpe 3.55)
    Sell weekly SPY iron condors when VIX is in normal range (15-25).
    Informational signal — options strategy.
    """
    v = vix.dropna()
    if len(v) < 1:
        return []

    current_vix = float(v.iloc[-1])

    if 15 <= current_vix <= 25:
        confidence = max(50, min(80, int(60 + (current_vix - 15) * 2)))
        return [{
            "ticker": "SPY",
            "direction": "SELL_IC",
            "confidence": confidence,
            "strategy": "spy_iron_condor",
            "rationale": (f"VIX at {current_vix:.1f} — in the sweet spot (15-25) for iron condors. "
                         f"Premium is fair and likely to decay."),
            "suggested_action": (f"SELL SPY weekly iron condor. Budget ~${MAX_PER_TRADE_OPTIONS:.0f} max risk. "
                                f"Wings 3-5% OTM. Commission: ${OPTIONS_COMMISSION*4:.2f} (4 legs)."),
            "hold_days": 5,
        }]
    elif current_vix < 15:
        return [{
            "ticker": "SPY",
            "direction": "HOLD",
            "confidence": 30,
            "strategy": "spy_iron_condor",
            "rationale": f"VIX at {current_vix:.1f} — too low for iron condors. Premium not worth the risk.",
            "suggested_action": "SKIP iron condor this week — VIX too low.",
            "hold_days": 0,
        }]
    else:
        return [{
            "ticker": "SPY",
            "direction": "HOLD",
            "confidence": 30,
            "strategy": "spy_iron_condor",
            "rationale": f"VIX at {current_vix:.1f} — too high for iron condors. Tail risk elevated.",
            "suggested_action": "SKIP iron condor this week — VIX too high, consider put credit spread instead.",
            "hold_days": 0,
        }]


def signal_iv_runup(kill: dict) -> list[dict]:
    """
    Strategy 2: IV Run-Up (Sharpe 2.27)
    Buy calls ~10 days before earnings on high-beta stocks when IV is still cheap.
    We can't fully automate this without an earnings calendar API, so this is a reminder.
    """
    if kill["status"] == "ACTIVE":
        return []

    return [{
        "ticker": "EARNINGS_WATCH",
        "direction": "INFO",
        "confidence": 0,
        "strategy": "iv_runup",
        "rationale": ("IV Run-Up strategy: check earnings calendar for stocks reporting in 7-14 days. "
                     "Buy ATM calls when IV rank < 50. Targets: AAPL, MSFT, GOOGL, AMZN, META, "
                     "NVDA, TSLA, AMD, NFLX, CRM, PLTR."),
        "suggested_action": ("Check earnings calendar (earningswhispers.com). If any watchlist stock "
                            "reports in 7-14 days with IV rank < 50, buy 1 ATM call (~${:.0f} budget).".format(
                                MAX_PER_TRADE_OPTIONS)),
        "hold_days": 10,
    }]


def signal_stock_split_monitor() -> list[dict]:
    """
    Strategy 6: Stock Split Pre-Announcement (Sharpe 1.508, 88% WR)
    Manual check — can't auto-detect from price data.
    """
    return [{
        "ticker": "SPLIT_WATCH",
        "direction": "INFO",
        "confidence": 0,
        "strategy": "stock_split",
        "rationale": ("Stock Split strategy: manually check news for forward split announcements. "
                     "When found, buy and hold 10 days pre-split date. "
                     "Historical: 88% WR, Sharpe 1.508, 17 trades."),
        "suggested_action": "Check financial news for recent stock split announcements in watchlist names.",
        "hold_days": 10,
    }]


def signal_sector_momentum_crash(close: pd.DataFrame, vix: pd.Series, kill: dict) -> list[dict]:
    """
    Strategy 10 (alt): Sector Momentum with Crash Protection
    If VIX > 25, go to cash. Otherwise sector selection alpha applies (covered by lgbm_sector_rotation).
    """
    v = vix.dropna()
    current_vix = float(v.iloc[-1])

    if current_vix > 25:
        return [{
            "ticker": "CASH",
            "direction": "HOLD",
            "confidence": 85,
            "strategy": "sector_momentum_crash",
            "rationale": f"VIX at {current_vix:.1f} (>25). Crash protection: hold cash, no new sector momentum entries.",
            "suggested_action": "HOLD CASH — crash protection active. Wait for VIX < 25.",
            "hold_days": 0,
        }]
    return []


def signal_consecutive_days_reversal(close: pd.DataFrame, kill: dict) -> list[dict]:
    """
    Strategy 11: Consecutive Days Reversal (Sharpe 1.15, 5/5 gates, 4/6 adversarial)
    Buy SPY after 4+ consecutive down days, hold 5 days.
    Bear Sharpe 1.28 > Bull 1.01 — genuinely regime-agnostic.
    ALLOWED even when kill switch ACTIVE (works in bear markets).
    """
    spy = close["SPY"].dropna()
    if len(spy) < 10:
        return []

    # Count consecutive down days (close < previous close)
    daily_returns = spy.pct_change().dropna()
    recent_returns = daily_returns.iloc[-10:]  # last 10 days

    consecutive_down = 0
    for ret in reversed(recent_returns.values):
        if ret < 0:
            consecutive_down += 1
        else:
            break

    if consecutive_down >= 4:
        # Calculate the magnitude of the decline
        total_decline = float((spy.iloc[-1] / spy.iloc[-consecutive_down - 1] - 1) * 100)
        confidence = min(95, 60 + consecutive_down * 8)  # 4 days=92, 5 days=100

        return [{
            "ticker": "SPY",
            "direction": "BUY",
            "confidence": confidence,
            "strategy": "consecutive_days_reversal",
            "rationale": (f"SPY has {consecutive_down} consecutive down days ({total_decline:.1f}% decline). "
                         f"Backtest: Sharpe 1.15, WR 69%, PF 4.16, bear Sharpe 1.28. "
                         f"Hold 5 trading days."),
            "suggested_action": (f"BUY ~$200 of SPY. Sell after 5 trading days. "
                                f"Historically wins 69% with 4:1 profit factor."),
            "hold_days": 5,
        }]

    # If close to triggering (3 down days), flag as watch
    if consecutive_down == 3:
        return [{
            "ticker": "SPY",
            "direction": "WATCH",
            "confidence": 40,
            "strategy": "consecutive_days_reversal",
            "rationale": (f"SPY has 3 consecutive down days. One more down day triggers the "
                         f"Consecutive Days Reversal signal (Sharpe 1.15, WR 69%)."),
            "suggested_action": "WATCH — signal fires if SPY closes red tomorrow.",
            "hold_days": 0,
        }]

    return []


# ──────────────────────────────────────────────────────────────────────────────
# CONFLUENCE SCORING
# ──────────────────────────────────────────────────────────────────────────────

def compute_confluence(signals: list[dict]) -> list[dict]:
    """
    For each ticker appearing in multiple BUY signals, compute a confluence score
    weighted by each strategy's backtested Sharpe ratio.
    """
    # Group BUY signals by ticker
    ticker_signals: dict[str, list[dict]] = {}
    for sig in signals:
        if sig["direction"] not in ("BUY",):
            continue
        ticker = sig["ticker"]
        if ticker not in ticker_signals:
            ticker_signals[ticker] = []
        ticker_signals[ticker].append(sig)

    confluence = []
    for ticker, sigs in ticker_signals.items():
        if len(sigs) < 2:
            continue

        strategies = [s["strategy"] for s in sigs]
        sharpe_sum = sum(STRATEGY_SHARPE.get(s, 1.0) for s in strategies)
        max_possible = sum(sorted(STRATEGY_SHARPE.values(), reverse=True)[:len(strategies)])

        # Normalized confluence score (0-100)
        raw_score = (sharpe_sum / max(max_possible, 1)) * 100
        # Boost for more agreeing strategies
        n_boost = min(20, (len(sigs) - 1) * 10)
        final_score = min(100, int(raw_score + n_boost))

        # Average individual confidence
        avg_confidence = int(np.mean([s["confidence"] for s in sigs]))

        # Combined confidence = max of individual and confluence
        combined = max(avg_confidence, final_score)

        level = "VERY HIGH" if combined > 80 else ("HIGH" if combined > 60 else "MODERATE")

        confluence.append({
            "ticker": ticker,
            "confluence_score": combined,
            "level": level,
            "n_strategies": len(sigs),
            "strategies": strategies,
            "sharpe_weighted_score": round(sharpe_sum, 2),
            "rationale": f"{len(sigs)} strategies agree: {', '.join(strategies)}. Sharpe-weighted score: {sharpe_sum:.2f}.",
            "suggested_action": next(
                (s["suggested_action"] for s in sigs if s["confidence"] == max(s2["confidence"] for s2 in sigs)),
                sigs[0]["suggested_action"]
            ),
        })

    confluence.sort(key=lambda x: x["confluence_score"], reverse=True)
    return confluence


# ──────────────────────────────────────────────────────────────────────────────
# MARKET CONTEXT
# ──────────────────────────────────────────────────────────────────────────────

def compute_market_context(close: pd.DataFrame, vix: pd.Series) -> dict:
    """Summarize broad market regime and trends."""
    spy = close["SPY"].dropna()
    qqq = close["QQQ"].dropna() if "QQQ" in close.columns else spy
    iwm = close["IWM"].dropna() if "IWM" in close.columns else spy
    v = vix.dropna()

    current_vix = float(v.iloc[-1])
    spy_ret_5d = compute_returns(spy, 5)
    spy_ret_21d = compute_returns(spy, 21)
    spy_ret_63d = compute_returns(spy, 63)
    spy_sma50 = float(spy.rolling(50).mean().iloc[-1])
    spy_sma200 = float(spy.rolling(200).mean().iloc[-1])
    spy_current = float(spy.iloc[-1])

    # Regime classification
    if current_vix > 30:
        regime = "CRISIS"
    elif current_vix > 20:
        regime = "ELEVATED_VOL"
    elif spy_current > spy_sma50 > spy_sma200:
        regime = "BULL_TREND"
    elif spy_current < spy_sma50 < spy_sma200:
        regime = "BEAR_TREND"
    elif spy_current > spy_sma200 and spy_current < spy_sma50:
        regime = "PULLBACK"
    else:
        regime = "MIXED"

    # Breadth: how many sector ETFs are above their 50-SMA
    breadth_count = 0
    for etf in SECTOR_ETFS:
        if etf not in close.columns:
            continue
        s = close[etf].dropna()
        if len(s) < 50:
            continue
        if float(s.iloc[-1]) > float(s.rolling(50).mean().iloc[-1]):
            breadth_count += 1
    breadth_pct = breadth_count / len(SECTOR_ETFS) * 100

    observations = []
    if spy_ret_5d > 0.02:
        observations.append(f"SPY strong short-term: +{spy_ret_5d*100:.1f}% in 5d")
    elif spy_ret_5d < -0.02:
        observations.append(f"SPY weak short-term: {spy_ret_5d*100:.1f}% in 5d")
    if current_vix > 25:
        observations.append(f"VIX elevated at {current_vix:.1f} — fear in the market")
    if breadth_pct < 40:
        observations.append(f"Narrow breadth: only {breadth_count}/11 sectors above 50-SMA")
    elif breadth_pct > 70:
        observations.append(f"Broad participation: {breadth_count}/11 sectors above 50-SMA")
    if spy_current > spy_sma200:
        observations.append("SPY above 200-SMA — long-term uptrend intact")
    else:
        observations.append("SPY below 200-SMA — long-term trend broken")

    return {
        "regime": regime,
        "vix": round(current_vix, 2),
        "spy_price": round(spy_current, 2),
        "spy_5d_ret": round(spy_ret_5d * 100, 2),
        "spy_21d_ret": round(spy_ret_21d * 100, 2),
        "spy_63d_ret": round(spy_ret_63d * 100, 2),
        "spy_vs_50sma": round((spy_current / spy_sma50 - 1) * 100, 2),
        "spy_vs_200sma": round((spy_current / spy_sma200 - 1) * 100, 2),
        "breadth_above_50sma": f"{breadth_count}/11",
        "breadth_pct": round(breadth_pct, 1),
        "observations": observations,
    }


# ──────────────────────────────────────────────────────────────────────────────
# POSITION SIZING
# ──────────────────────────────────────────────────────────────────────────────

def size_positions(signals: list[dict], confluence: list[dict], kill: dict) -> list[dict]:
    """
    Produce final trade recommendations with specific position sizes for $667 account.
    Prioritize confluence signals, then highest-confidence individual signals.
    """
    recommendations = []
    remaining_cash = ACCOUNT_CASH

    # First: confluence signals (highest priority)
    for csig in confluence:
        if remaining_cash < 50:
            break
        if csig["level"] in ("HIGH", "VERY HIGH"):
            alloc = min(MAX_PER_TRADE_SHARES, remaining_cash * 0.35)
            alloc = min(alloc, remaining_cash)
            if alloc < 50:
                continue
            recommendations.append({
                "ticker": csig["ticker"],
                "action": "BUY",
                "amount": round(alloc, 2),
                "confidence": csig["confluence_score"],
                "reason": csig["rationale"],
                "priority": "HIGH" if csig["level"] == "VERY HIGH" else "MEDIUM",
            })
            remaining_cash -= alloc

    # Then: top individual BUY signals not already covered by confluence
    confluence_tickers = {c["ticker"] for c in confluence}
    buy_signals = sorted(
        [s for s in signals if s["direction"] == "BUY" and s["ticker"] not in confluence_tickers],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    for sig in buy_signals[:5]:  # limit to top 5
        if remaining_cash < 50:
            break
        alloc = min(MAX_PER_TRADE_SHARES, remaining_cash * 0.25)
        alloc = min(alloc, remaining_cash)
        if alloc < 50:
            continue
        recommendations.append({
            "ticker": sig["ticker"],
            "action": "BUY",
            "amount": round(alloc, 2),
            "confidence": sig["confidence"],
            "reason": sig["rationale"],
            "priority": "MEDIUM" if sig["confidence"] > 60 else "LOW",
        })
        remaining_cash -= alloc

    return recommendations


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(kill: dict, signals: list[dict], confluence: list[dict],
                  context: dict, recommendations: list[dict]):
    """Print a clean human-readable summary to stdout."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    w = 80
    print("=" * w)
    print(f"  ENHANCED SIGNAL SCANNER v2 — {now}")
    print("=" * w)

    # Kill switch
    ks = kill
    print(f"\n{'KILL SWITCH':>14}: {ks['status']}")
    print(f"{'VIX':>14}: {ks['vix']}")
    print(f"{'SPY':>14}: ${ks['spy_price']} (50-SMA: ${ks['spy_sma50']}, {ks['spy_vs_sma_pct']:+.2f}%)")
    print(f"{'Effect':>14}: {ks['effect']}")

    # Market context
    print(f"\n{'--- MARKET CONTEXT ---':^{w}}")
    print(f"  Regime:       {context['regime']}")
    print(f"  SPY returns:  5d {context['spy_5d_ret']:+.2f}%  |  21d {context['spy_21d_ret']:+.2f}%  |  63d {context['spy_63d_ret']:+.2f}%")
    print(f"  SPY vs SMAs:  50-SMA {context['spy_vs_50sma']:+.2f}%  |  200-SMA {context['spy_vs_200sma']:+.2f}%")
    print(f"  Breadth:      {context['breadth_above_50sma']} sectors above 50-SMA ({context['breadth_pct']}%)")
    for obs in context["observations"]:
        print(f"  * {obs}")

    # Active signals
    buy_signals = [s for s in signals if s["direction"] == "BUY"]
    avoid_signals = [s for s in signals if s["direction"] in ("AVOID", "REDUCE")]
    info_signals = [s for s in signals if s["direction"] in ("INFO", "HOLD", "SELL_IC")]

    print(f"\n{'--- BUY SIGNALS ---':^{w}}")
    if buy_signals:
        for s in sorted(buy_signals, key=lambda x: x["confidence"], reverse=True):
            print(f"  [{s['confidence']:3d}] {s['ticker']:6s}  {s['strategy']:25s}  {s['rationale'][:60]}")
    else:
        print("  No BUY signals today.")

    print(f"\n{'--- AVOID / REDUCE ---':^{w}}")
    if avoid_signals:
        for s in avoid_signals:
            print(f"  [{s['confidence']:3d}] {s['ticker']:6s}  {s['strategy']:25s}  {s['suggested_action']}")
    else:
        print("  No AVOID/REDUCE signals.")

    print(f"\n{'--- INFO / OPTIONS ---':^{w}}")
    for s in info_signals:
        print(f"  [{s['ticker']:15s}]  {s['suggested_action'][:70]}")

    # Confluence
    print(f"\n{'--- CONFLUENCE (2+ strategies agree) ---':^{w}}")
    if confluence:
        for c in confluence:
            level_marker = "***" if c["level"] == "VERY HIGH" else "**" if c["level"] == "HIGH" else "*"
            print(f"  {level_marker} {c['ticker']:6s}  Score: {c['confluence_score']}  "
                  f"({c['n_strategies']} strategies: {', '.join(c['strategies'])})")
    else:
        print("  No confluence signals (no ticker with 2+ agreeing strategies).")

    # Recommendations
    print(f"\n{'--- TRADE RECOMMENDATIONS (sized for $667 account) ---':^{w}}")
    if recommendations:
        total_alloc = 0
        for r in recommendations:
            print(f"  [{r['priority']:6s}] {r['action']} ${r['amount']:.0f} of {r['ticker']:6s}  "
                  f"(confidence: {r['confidence']})")
            total_alloc += r["amount"]
        print(f"\n  Total allocated: ${total_alloc:.0f} / ${ACCOUNT_CASH:.0f}  "
              f"(${ACCOUNT_CASH - total_alloc:.0f} cash remaining)")
    else:
        print("  No actionable trades today. Hold cash.")

    print("\n" + "=" * w)
    print(f"  Scanner complete. {len(buy_signals)} BUY | {len(avoid_signals)} AVOID | {len(confluence)} CONFLUENCE")
    print("=" * w)


def save_state(kill: dict, signals: list[dict], confluence: list[dict],
               context: dict, recommendations: list[dict], dry: bool):
    """Save scanner output to JSON and optional alert file."""
    if dry:
        print("\n[DRY RUN] Skipping state save.")
        return

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "scanner_version": "v2.0",
        "account_cash": ACCOUNT_CASH,
        "kill_switch": kill,
        "market_context": context,
        "signals": signals,
        "confluence_signals": confluence,
        "recommendations": recommendations,
        "summary": {
            "total_buy_signals": len([s for s in signals if s["direction"] == "BUY"]),
            "total_avoid_signals": len([s for s in signals if s["direction"] in ("AVOID", "REDUCE")]),
            "confluence_count": len(confluence),
            "very_high_confidence": len([c for c in confluence if c["level"] == "VERY HIGH"]),
        },
    }

    with open(SIGNALS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved signals to {SIGNALS_FILE}")

    # High confidence alert
    very_high = [c for c in confluence if c["level"] == "VERY HIGH"]
    if very_high:
        with open(ALERT_FILE, "w") as f:
            f.write(f"HIGH CONFIDENCE ALERT — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 60 + "\n\n")
            for c in very_high:
                f.write(f"TICKER: {c['ticker']}\n")
                f.write(f"SCORE:  {c['confluence_score']}\n")
                f.write(f"STRATEGIES: {', '.join(c['strategies'])}\n")
                f.write(f"RATIONALE: {c['rationale']}\n")
                f.write(f"ACTION: {c['suggested_action']}\n\n")
        print(f"ALERT: Very high confidence signals! See {ALERT_FILE}")
    elif ALERT_FILE.exists():
        ALERT_FILE.unlink()  # remove stale alerts


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enhanced Signal Scanner v2")
    parser.add_argument("--dry", action="store_true", help="Dry run — print only, don't save state")
    args = parser.parse_args()

    # 1. Download data
    close, vix = download_data()

    # 2. Kill switch check
    kill = check_kill_switch(close, vix)

    # 3. Run all signal modules
    all_signals = []

    print("\nScanning strategies...")
    modules = [
        ("LGBM Sector Rotation", lambda: signal_lgbm_sector_rotation(close, kill)),
        ("Contrarian Reversion", lambda: signal_contrarian_reversion(close, kill)),
        ("VIX Spike Fade", lambda: signal_vix_spike_fade(close, vix, kill)),
        ("Fear Signal", lambda: signal_fear_signal(close, vix, kill)),
        ("Drawdown Recovery", lambda: signal_drawdown_recovery(close, kill)),
        ("PEAD Monitor", lambda: signal_pead_monitor(close, kill)),
        ("SPY Iron Condor", lambda: signal_spy_iron_condor(close, vix, kill)),
        ("IV Run-Up Reminder", lambda: signal_iv_runup(kill)),
        ("Stock Split Monitor", lambda: signal_stock_split_monitor()),
        ("Momentum Crash Prot.", lambda: signal_momentum_crash_protection(close, vix, kill)),
        ("Sector Mom. Crash", lambda: signal_sector_momentum_crash(close, vix, kill)),
        ("Consec. Days Reversal", lambda: signal_consecutive_days_reversal(close, kill)),
    ]

    for name, fn in modules:
        sigs = fn()
        buy_count = len([s for s in sigs if s["direction"] == "BUY"])
        total = len(sigs)
        status = f"{buy_count} BUY" if buy_count else f"{total} signals" if total else "no signals"
        print(f"  {name:30s} -> {status}")
        all_signals.extend(sigs)

    # 4. Confluence scoring
    confluence = compute_confluence(all_signals)

    # 5. Market context
    context = compute_market_context(close, vix)

    # 6. Position sizing / recommendations
    recommendations = size_positions(all_signals, confluence, kill)

    # 7. Output
    print_summary(kill, all_signals, confluence, context, recommendations)
    save_state(kill, all_signals, confluence, context, recommendations, args.dry)


if __name__ == "__main__":
    main()
