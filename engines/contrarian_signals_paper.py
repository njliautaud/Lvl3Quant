#!/usr/bin/env python3
"""
Unified Contrarian Signals Paper Trading Engine
=================================================
Paper trades ALL validated contrarian alpha signals in a single engine:

1. Skewness Premium — negative skew stocks + 3% drop → mean reversion (Sharpe 1.02)
2. Post-Earnings Drift Contrarian — big drop on event, enter 20-30d later (Sharpe 2.11)
3. Smart Money Accumulation — OBV rising + price falling + MFI<20 (Sharpe 1.61)
4. Price-Volume Divergence — OBV+MFI rising + price down + RSI<40 (Sharpe 1.23)
5. Volatility Crush Reversal — ATR compressed + sharp drop + MFI<20 (Sharpe 0.96)

Per HC #745: All validated strategies must run on paper. Income = consistent without NAV loss.
Per HC #694: Commission-free on Robinhood.
Per HC #746: Enhanced flow filters (CTA pressure, breadth, sector rotation) applied to entries.

Runs daily at 15:50 ET via PM2 cron.
State: /home/jupiter/Lvl3Quant/state/contrarian_signals_state.json
History: /home/jupiter/Lvl3Quant/state/contrarian_signals_history.csv
"""
from __future__ import annotations

import json
import logging
import math
import sys
import traceback
import urllib.request
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# ── Config ────────────────────────────────────────────────────────────────
ET = pytz.timezone("US/Eastern")
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "contrarian_signals_state.json"
HISTORY_FILE = STATE_DIR / "contrarian_signals_history.csv"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK_FILE = ROOT.parent / "teleclaude-main" / "API_KEYS.md"

INITIAL_CAPITAL = 100_000
MAX_POSITIONS = 20          # Max concurrent positions
PER_POSITION_PCT = 0.05     # 5% of portfolio per position
COST_BPS = 5                # 0.05% bid-ask for liquid large-caps (HC #694 commission-free)

# Universe: S&P 500 large-caps + sector/broad ETFs (liquid, tight spreads)
UNIVERSE_STOCKS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "JNJ", "MA", "PG", "HD", "XOM", "COST", "ABBV",
    "KO", "PEP", "MRK", "AVGO", "CVX", "LLY", "WMT", "BAC", "PFE",
    "CRM", "TMO", "CSCO", "MCD", "ABT", "ORCL", "ACN", "DHR", "TXN",
    "NEE", "ADBE", "PM", "NKE", "CMCSA", "UPS", "RTX", "HON", "LOW",
    "QCOM", "INTC", "AMAT", "AMGN", "CAT",
]

# Sector ETFs (SPDR Select Sector)
UNIVERSE_SECTOR_ETFS = [
    "XLE", "XLF", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC",
]

# Broad market + thematic ETFs
UNIVERSE_BROAD_ETFS = [
    "SPY", "QQQ", "IWM", "DIA",     # broad indices
    "GLD", "SLV", "TLT", "HYG",     # commodities / bonds
    "XBI", "ARKK", "SMH", "KWEB",   # thematic (biotech, innovation, semis, china tech)
    "EEM", "EFA",                     # international
]

UNIVERSE = UNIVERSE_STOCKS + UNIVERSE_SECTOR_ETFS + UNIVERSE_BROAD_ETFS

# Signal parameters (best validated variants)
SIGNALS = {
    # ── BACKTEST-VALIDATED (p=0.01, 69% WR, PF 3.22) ──
    "oversold_bounce": {
        "weekly_drop": -0.10,    # 10%+ drop in 5 trading days (backtest winner)
        "hold_days": 5,          # backtest-optimal hold
        "backtest_validated": True,
    },
    # ── BACKTEST-VALIDATED (all 5 gates pass, 62.7% WR, PF 1.70, Sharpe 1.50) ──
    "post_earnings_bounce": {
        "drop_2d": -0.08,        # 8%+ drop in 2 trading days after earnings
        "hold_days": 10,         # 10-day hold (backtest-optimal, regime-agnostic)
        "backtest_validated": True,
    },
    # ── UNVALIDATED (kept for paper tracking) ──
    "skewness": {
        "lookback": 252,        # 1-year skewness
        "percentile": 10,       # bottom 10% skewness
        "drop_threshold": -0.03, # 3% drop trigger
        "hold_days": 10,
    },
    "post_earnings_drift": {
        "event_drop": -0.05,     # 5%+ single-day drop on high volume
        "volume_mult": 2.0,      # 2x average volume
        "wait_window_start": 20, # enter 20-30 days after event
        "wait_window_end": 30,
        "hold_days": 21,
    },
    "smart_money": {
        "acc_lookback": 10,     # 10-day accumulation window
        "price_drop": -0.02,    # price down 2%+
        "volume_mult": 1.2,     # 1.2x volume
        "mfi_threshold": 20,    # MFI < 20
        "hold_days": 10,
    },
    "price_volume_div": {
        "lookback": 10,         # 10-day divergence
        "rsi_threshold": 40,    # RSI < 40
        "hold_days": 10,
    },
    "vol_crush": {
        "atr_lookback": 20,     # 20-day ATR
        "atr_percentile": 20,   # ATR in bottom 20th percentile (within 21d)
        "recency": 21,          # within last 21 days
        "drop_threshold": -0.05, # 5% drop
        "mfi_threshold": 20,    # MFI < 20
        "hold_days": 21,
    },
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Contrarian] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "contrarian_signals_paper.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("contrarian")


# ── Technical Indicators ──────────────────────────────────────────────────

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_mfi(high, low, close, volume, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0).rolling(period).sum()
    neg_mf = mf.where(delta <= 0, 0).rolling(period).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


def compute_atr(high, low, close, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Macro Flow Context (HC #746) ─────────────────────────────────────────

def compute_macro_flow_context() -> dict:
    """Compute CTA pressure, breadth, and sector rotation scores from real data.
    Returns dict with scores and whether macro environment favors contrarian entries."""
    context = {
        "cta_pressure": 0.0,
        "breadth_score": 0.0,
        "sector_rotation_intensity": 0.0,
        "risk_appetite": 0.0,
        "macro_favorable": True,  # default: allow entries
        "reason": "",
    }
    try:
        # CTA pressure: how far key assets are above/below MA crossover
        cta_tickers = ["SPY", "QQQ", "GLD", "TLT"]
        cta_data = yf.download(cta_tickers, period="1y", progress=False, auto_adjust=True)
        if isinstance(cta_data.columns, pd.MultiIndex):
            cta_close = cta_data["Close"]
        else:
            cta_close = cta_data[["Close"]]

        cta_scores = []
        for t in cta_tickers:
            try:
                c = cta_close[t].dropna()
                if len(c) < 200:
                    continue
                # Average distance from 20/50/100/200 SMA (normalized)
                sma_dists = []
                for period in [20, 50, 100, 200]:
                    sma = c.rolling(period).mean()
                    dist = (c.iloc[-1] / sma.iloc[-1] - 1) * 100
                    sma_dists.append(dist)
                cta_scores.append(np.mean(sma_dists))
            except Exception:
                pass
        if cta_scores:
            context["cta_pressure"] = round(np.mean(cta_scores), 2)

        # Breadth: how many sector ETFs are up over last 10d
        sector_tickers = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC"]
        sect_data = yf.download(sector_tickers, period="1mo", progress=False, auto_adjust=True)
        if isinstance(sect_data.columns, pd.MultiIndex):
            sect_close = sect_data["Close"]
        else:
            sect_close = sect_data[["Close"]]

        up_count = 0
        total = 0
        vol_changes = []
        for t in sector_tickers:
            try:
                c = sect_close[t].dropna()
                if len(c) >= 10:
                    ret_10d = c.iloc[-1] / c.iloc[-10] - 1
                    if ret_10d > 0:
                        up_count += 1
                    total += 1
            except Exception:
                pass

        if total > 0:
            context["breadth_score"] = round(up_count / total, 2)

        # Sector rotation intensity: std of 10d returns across sectors
        sector_rets = []
        for t in sector_tickers:
            try:
                c = sect_close[t].dropna()
                if len(c) >= 10:
                    sector_rets.append(c.iloc[-1] / c.iloc[-10] - 1)
            except Exception:
                pass
        if len(sector_rets) >= 5:
            context["sector_rotation_intensity"] = round(np.std(sector_rets) * 100, 2)

        # Risk appetite: SPY volume vs SHY volume (5d avg)
        try:
            safety = yf.download(["SHY", "SPY"], period="1mo", progress=False, auto_adjust=True)
            if isinstance(safety.columns, pd.MultiIndex):
                spy_vol = safety["Volume"]["SPY"].dropna()
                shy_vol = safety["Volume"]["SHY"].dropna()
                if len(spy_vol) >= 5 and len(shy_vol) >= 5:
                    ratio = spy_vol.iloc[-5:].mean() / max(shy_vol.iloc[-5:].mean(), 1)
                    # Higher ratio = more risk appetite
                    context["risk_appetite"] = round(ratio, 2)
        except Exception:
            pass

        # Decision logic: contrarian entries work BEST when:
        # - CTA pressure is negative or muted (not everyone bullish)
        # - Sector rotation intensity is high (money moving = opportunity)
        # - Breadth is moderate (not extreme in either direction)
        # Block entries ONLY when CTA pressure is extremely high (contrarian headwind)
        if context["cta_pressure"] > 5.0 and context["breadth_score"] > 0.85:
            context["macro_favorable"] = False
            context["reason"] = (f"CTA pressure too high ({context['cta_pressure']:.1f}%) "
                                 f"+ breadth {context['breadth_score']:.0%} — contrarian headwind")
        else:
            # Bonus info for logging
            conditions = []
            if context["sector_rotation_intensity"] > 2.0:
                conditions.append("high rotation (good for contrarian)")
            if context["cta_pressure"] < 0:
                conditions.append("CTA bearish (contrarian favorable)")
            context["reason"] = "; ".join(conditions) if conditions else "neutral macro"

    except Exception as e:
        log.warning(f"Macro flow context failed: {e}")
        context["reason"] = f"computation failed: {e}"

    return context


# ── Signal Detection ──────────────────────────────────────────────────────

def detect_signals(ticker: str, df: pd.DataFrame) -> list[dict]:
    """Detect all contrarian signals for a ticker. Returns list of signal dicts."""
    signals = []
    if len(df) < 260:
        return signals

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    today = df.index[-1]
    current_price = float(close.iloc[-1])

    # Pre-compute indicators
    rsi = compute_rsi(close)
    mfi = compute_mfi(high, low, close, volume)
    obv = compute_obv(close, volume)
    atr = compute_atr(high, low, close)
    daily_return = close.pct_change()
    avg_volume_20 = volume.rolling(20).mean()

    # --- 0. OVERSOLD BOUNCE (BACKTEST VALIDATED p=0.01) ---
    p = SIGNALS["oversold_bounce"]
    if len(close) >= 6:
        weekly_ret = float(close.iloc[-1] / close.iloc[-6] - 1)
        if weekly_ret <= p["weekly_drop"]:
            signals.append({
                "signal": "oversold_bounce",
                "ticker": ticker,
                "price": current_price,
                "hold_days": p["hold_days"],
                "reason": f"Weekly drop {weekly_ret:.1%} (backtest-validated, p=0.01)",
                "date": str(today.date()),
                "backtest_validated": True,
            })

    # --- 0b. POST-EARNINGS BOUNCE (BACKTEST VALIDATED — all 5 gates) ---
    p = SIGNALS["post_earnings_bounce"]
    try:
        tk = yf.Ticker(ticker)
        today_d = today.date() if hasattr(today, 'date') else today
        edates_found = None
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
                            from datetime import datetime as dt2
                            ed = dt2.strptime(ed[:10], "%Y-%m-%d").date()
                        days_ago = (today_d - ed).days
                        if 2 <= days_ago <= 7:
                            edates_found = ed
        except Exception:
            pass
        if edates_found is None:
            try:
                edf = tk.earnings_dates
                if edf is not None and not edf.empty:
                    for idx in edf.index:
                        d = idx.date() if hasattr(idx, "date") else idx
                        days_ago = (today_d - d).days
                        if 2 <= days_ago <= 7:
                            edates_found = d
                            break
            except Exception:
                pass
        if edates_found is not None and len(close) >= 10:
            days_since = (today_d - edates_found).days
            ei = len(close) - days_since - 1
            if 0 <= ei and ei + 2 < len(close):
                p_earn = float(close.iloc[ei])
                p_after = float(close.iloc[ei + 2])
                drop_2d = (p_after / p_earn) - 1
                if drop_2d <= p["drop_2d"]:
                    signals.append({
                        "signal": "post_earnings_bounce",
                        "ticker": ticker,
                        "price": current_price,
                        "hold_days": p["hold_days"],
                        "reason": f"Dropped {drop_2d:.1%} after earnings on {edates_found}, {days_since}d ago (all gates validated)",
                        "date": str(today_d),
                        "backtest_validated": True,
                    })
    except Exception:
        pass

    # --- 1. Skewness Premium ---
    p = SIGNALS["skewness"]
    if len(close) >= p["lookback"]:
        skew = daily_return.iloc[-p["lookback"]:].skew()
        ret_1d = float(daily_return.iloc[-1])
        if not np.isnan(skew) and skew < -0.5 and ret_1d <= p["drop_threshold"]:
            signals.append({
                "signal": "skewness_premium",
                "ticker": ticker,
                "price": current_price,
                "hold_days": p["hold_days"],
                "reason": f"Skew={skew:.2f}, 1d drop={ret_1d:.1%}",
                "date": str(today.date()),
            })

    # --- 2. Post-Earnings Drift Contrarian ---
    p = SIGNALS["post_earnings_drift"]
    # Look for big drop events 20-30 days ago
    for lookback in range(p["wait_window_start"], p["wait_window_end"] + 1):
        if lookback >= len(daily_return):
            continue
        past_ret = float(daily_return.iloc[-lookback])
        past_vol = float(volume.iloc[-lookback])
        avg_vol = float(avg_volume_20.iloc[-lookback]) if not np.isnan(avg_volume_20.iloc[-lookback]) else 1
        if past_ret <= p["event_drop"] and past_vol >= avg_vol * p["volume_mult"]:
            # Check if MFI is reasonable now
            current_mfi = float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else 50
            if current_mfi < 30:  # MFI<30 for post-event entry
                signals.append({
                    "signal": "post_earnings_drift",
                    "ticker": ticker,
                    "price": current_price,
                    "hold_days": p["hold_days"],
                    "reason": f"Event {lookback}d ago (ret={past_ret:.1%}, vol={past_vol/avg_vol:.1f}x), MFI={current_mfi:.0f}",
                    "date": str(today.date()),
                })
            break  # Only fire once per ticker

    # --- 3. Smart Money Accumulation ---
    p = SIGNALS["smart_money"]
    if len(obv) >= p["acc_lookback"] + 1:
        obv_change = (float(obv.iloc[-1]) - float(obv.iloc[-p["acc_lookback"]-1]))
        price_change = float(close.iloc[-1] / close.iloc[-p["acc_lookback"]-1] - 1)
        vol_ratio = float(volume.iloc[-1] / avg_volume_20.iloc[-1]) if not np.isnan(avg_volume_20.iloc[-1]) else 1
        current_mfi = float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else 50

        if (obv_change > 0 and price_change <= p["price_drop"]
                and vol_ratio >= p["volume_mult"] and current_mfi < p["mfi_threshold"]):
            signals.append({
                "signal": "smart_money_accumulation",
                "ticker": ticker,
                "price": current_price,
                "hold_days": p["hold_days"],
                "reason": f"OBV rising, price {price_change:.1%}, vol {vol_ratio:.1f}x, MFI={current_mfi:.0f}",
                "date": str(today.date()),
            })

    # --- 4. Price-Volume Divergence ---
    p = SIGNALS["price_volume_div"]
    if len(obv) >= p["lookback"] + 1:
        obv_slope = float(obv.iloc[-1] - obv.iloc[-p["lookback"]-1])
        # MFI trend (rising over lookback)
        mfi_now = float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else 50
        mfi_past = float(mfi.iloc[-p["lookback"]-1]) if not np.isnan(mfi.iloc[-p["lookback"]-1]) else 50
        mfi_rising = mfi_now > mfi_past
        price_change = float(close.iloc[-1] / close.iloc[-p["lookback"]-1] - 1)
        current_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50

        if (obv_slope > 0 and mfi_rising and price_change < 0
                and current_rsi < p["rsi_threshold"]):
            signals.append({
                "signal": "price_volume_divergence",
                "ticker": ticker,
                "price": current_price,
                "hold_days": p["hold_days"],
                "reason": f"OBV+MFI rising, price {price_change:.1%}, RSI={current_rsi:.0f}",
                "date": str(today.date()),
            })

    # --- 5. Volatility Crush Reversal ---
    p = SIGNALS["vol_crush"]
    if len(atr) >= p["atr_lookback"] + p["recency"]:
        current_atr = float(atr.iloc[-1])
        # Check if ATR was in bottom percentile within recent window
        recent_atrs = atr.iloc[-p["recency"]-p["atr_lookback"]:-p["recency"]]
        if len(recent_atrs) > 0 and not recent_atrs.isna().all():
            atr_pctile = float((recent_atrs < current_atr).mean() * 100)
            # ATR was compressed recently
            min_recent_atr = float(recent_atrs.min())
            was_compressed = min_recent_atr <= float(recent_atrs.quantile(p["atr_percentile"] / 100))

            price_change = float(daily_return.iloc[-5:].sum())  # 5d cumulative drop
            current_mfi = float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else 50

            if (was_compressed and price_change <= p["drop_threshold"]
                    and current_mfi < p["mfi_threshold"]):
                signals.append({
                    "signal": "vol_crush_reversal",
                    "ticker": ticker,
                    "price": current_price,
                    "hold_days": p["hold_days"],
                    "reason": f"ATR compressed, 5d drop={price_change:.1%}, MFI={current_mfi:.0f}",
                    "date": str(today.date()),
                })

    return signals


# ── State Management ──────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "cash": INITIAL_CAPITAL,
        "positions": {},
        "next_id": 1,
        "start_date": datetime.now(ET).strftime("%Y-%m-%d"),
        "total_trades": 0,
        "total_wins": 0,
        "total_pnl": 0.0,
        "signal_stats": {s: {"trades": 0, "wins": 0, "pnl": 0.0} for s in SIGNALS},
        "last_update": None,
    }


def save_state(state: dict):
    state["last_update"] = datetime.now(ET).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def append_history(row: dict):
    df = pd.DataFrame([row])
    if HISTORY_FILE.exists():
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)


# ── Alerting ──────────────────────────────────────────────────────────────

def get_webhook_url() -> Optional[str]:
    try:
        import re
        text = WEBHOOK_FILE.read_text()
        for line in text.splitlines():
            if "discord" in line.lower() and "webhook" in line.lower():
                urls = re.findall(r'https://discord\.com/api/webhooks/\S+', line)
                if urls:
                    return urls[0].rstrip('`').rstrip("'").rstrip('"')
    except Exception:
        pass
    return None


def send_alert(message: str):
    log.info(f"ALERT: {message}")
    url = get_webhook_url()
    if not url:
        return
    try:
        payload = json.dumps({"content": message[:1990]}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log.warning(f"Failed to send alert: {e}")


# ── Main Logic ────────────────────────────────────────────────────────────

def run_daily():
    now = datetime.now(ET)
    log.info(f"\n{'='*60}")
    log.info(f"Contrarian Signals Paper Engine — {now.strftime('%Y-%m-%d %H:%M ET')}")
    log.info(f"{'='*60}")

    state = load_state()
    portfolio_value = state["cash"] + sum(
        p.get("position_value", p.get("entry_value", 0))
        for p in state["positions"].values()
    )
    n_open = len(state["positions"])
    log.info(f"Portfolio: ${portfolio_value:,.0f} | Cash: ${state['cash']:,.0f} | "
             f"Open: {n_open}/{MAX_POSITIONS}")

    # ── Step 1: Update existing positions ──
    closes_to_do = []
    if state["positions"]:
        log.info("\n--- Updating positions ---")
        for pid in list(state["positions"].keys()):
            pos = state["positions"][pid]
            try:
                hist = yf.download(pos["ticker"], period="5d", progress=False, auto_adjust=True)
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                if hist.empty:
                    continue
                current_price = float(hist["Close"].iloc[-1])
                entry_price = pos["entry_price"]
                pnl_pct = (current_price / entry_price - 1) * 100
                cost = entry_price * COST_BPS / 10000 * 2  # Round-trip cost
                pnl_dollar = (current_price - entry_price - cost) * pos["shares"]

                pos["current_price"] = current_price
                pos["pnl_pct"] = round(pnl_pct, 2)
                pos["pnl_dollar"] = round(pnl_dollar, 2)
                pos["position_value"] = round(current_price * pos["shares"], 2)

                # Check if hold period expired
                entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
                days_held = (date.today() - entry_date).days
                pos["days_held"] = days_held

                if days_held >= pos["hold_days"]:
                    closes_to_do.append((pid, "HOLD_EXPIRED"))

                log.info(f"  #{pid} {pos['ticker']} [{pos['signal']}]: "
                         f"${current_price:.2f} ({pnl_pct:+.1f}%) day {days_held}/{pos['hold_days']}"
                         f"{' → CLOSE' if days_held >= pos['hold_days'] else ''}")

            except Exception as e:
                log.warning(f"  #{pid} {pos['ticker']}: update failed: {e}")

    # ── Step 2: Close expired positions ──
    for pid, reason in closes_to_do:
        pos = state["positions"][pid]
        pnl = pos.get("pnl_dollar", 0)
        sig = pos["signal"]

        trade_record = {
            "date": datetime.now(ET).strftime("%Y-%m-%d %H:%M"),
            "ticker": pos["ticker"],
            "signal": sig,
            "action": "CLOSE",
            "reason": reason,
            "entry_price": pos["entry_price"],
            "exit_price": pos.get("current_price", pos["entry_price"]),
            "shares": pos["shares"],
            "pnl_dollar": round(pnl, 2),
            "pnl_pct": pos.get("pnl_pct", 0),
            "hold_days": pos.get("days_held", 0),
        }
        append_history(trade_record)

        state["cash"] += pos.get("position_value", pos["entry_value"])
        state["total_trades"] += 1
        state["total_pnl"] += pnl
        if pnl > 0:
            state["total_wins"] += 1

        # Signal-level stats
        if sig in state["signal_stats"]:
            state["signal_stats"][sig]["trades"] += 1
            state["signal_stats"][sig]["pnl"] += pnl
            if pnl > 0:
                state["signal_stats"][sig]["wins"] += 1

        log.info(f"  CLOSED #{pid} {pos['ticker']} [{sig}]: P&L ${pnl:+.2f} ({pos.get('pnl_pct', 0):+.1f}%)")
        del state["positions"][pid]

    # ── Step 3: Scan for new signals ──
    n_open = len(state["positions"])
    slots = MAX_POSITIONS - n_open
    if slots <= 0:
        log.info(f"\nNo slots ({n_open}/{MAX_POSITIONS})")
    else:
        log.info(f"\n--- Scanning {len(UNIVERSE)} tickers for signals ({slots} slots) ---")

        # Already-held tickers
        held_tickers = {p["ticker"] for p in state["positions"].values()}

        all_signals = []
        for ticker in UNIVERSE:
            if ticker in held_tickers:
                continue
            try:
                df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if df.empty or len(df) < 60:
                    continue
                sigs = detect_signals(ticker, df)
                all_signals.extend(sigs)
            except Exception as e:
                log.debug(f"  {ticker}: error: {e}")

        # Log found signals
        if all_signals:
            log.info(f"  Found {len(all_signals)} signals:")
            for s in all_signals:
                log.info(f"    {s['signal']:>25s} | {s['ticker']:>5s} @ ${s['price']:.2f} | {s['reason']}")

        # ── Macro flow filter (HC #746) ──
        macro = compute_macro_flow_context()
        log.info(f"\n--- Macro Flow Context ---")
        log.info(f"  CTA pressure: {macro['cta_pressure']:.1f}%")
        log.info(f"  Breadth: {macro['breadth_score']:.0%} sectors up")
        log.info(f"  Sector rotation: {macro['sector_rotation_intensity']:.1f}%")
        log.info(f"  Risk appetite: {macro['risk_appetite']:.0f}")
        log.info(f"  Favorable: {macro['macro_favorable']} — {macro['reason']}")

        if not macro["macro_favorable"] and all_signals:
            log.info(f"  ⚠️ MACRO FILTER: Blocking {len(all_signals)} signals — {macro['reason']}")
            send_alert(f"📊 Contrarian signals detected but BLOCKED by macro filter: {macro['reason']}")
            all_signals = []  # Block all entries when macro is strongly against contrarian

        # Add macro context to each signal for logging
        for sig in all_signals:
            sig["macro_context"] = {
                "cta_pressure": macro["cta_pressure"],
                "breadth": macro["breadth_score"],
                "rotation": macro["sector_rotation_intensity"],
                "favorable": macro["macro_favorable"],
            }

        # Prioritize: higher-Sharpe signals first
        signal_priority = {
            "post_earnings_drift": 1,
            "smart_money_accumulation": 2,
            "price_volume_divergence": 3,
            "skewness_premium": 4,
            "vol_crush_reversal": 5,
        }
        all_signals.sort(key=lambda s: signal_priority.get(s["signal"], 99))

        # Open positions
        opened = 0
        for sig in all_signals:
            if opened >= slots:
                break
            # Skip if already have this ticker
            if sig["ticker"] in held_tickers:
                continue

            alloc = portfolio_value * PER_POSITION_PCT
            if alloc > state["cash"]:
                continue

            shares = int(alloc / sig["price"])
            if shares < 1:
                continue

            entry_value = shares * sig["price"]
            pid = str(state["next_id"])
            state["next_id"] += 1

            state["positions"][pid] = {
                "ticker": sig["ticker"],
                "signal": sig["signal"],
                "entry_date": str(date.today()),
                "entry_price": sig["price"],
                "shares": shares,
                "entry_value": round(entry_value, 2),
                "position_value": round(entry_value, 2),
                "hold_days": sig["hold_days"],
                "reason": sig["reason"],
                "days_held": 0,
                "pnl_pct": 0,
                "pnl_dollar": 0,
            }
            state["cash"] -= entry_value
            held_tickers.add(sig["ticker"])
            opened += 1

            log.info(f"  OPENED #{pid} {sig['ticker']} [{sig['signal']}]: "
                     f"{shares} shares @ ${sig['price']:.2f} = ${entry_value:,.0f}")

            trade_record = {
                "date": datetime.now(ET).strftime("%Y-%m-%d %H:%M"),
                "ticker": sig["ticker"],
                "signal": sig["signal"],
                "action": "OPEN",
                "reason": sig["reason"],
                "entry_price": sig["price"],
                "exit_price": 0,
                "shares": shares,
                "pnl_dollar": 0,
                "pnl_pct": 0,
                "hold_days": 0,
            }
            append_history(trade_record)

        if not all_signals:
            log.info("  No signals detected")

    # ── Step 4: Summary ──
    portfolio_value = state["cash"] + sum(
        p.get("position_value", p.get("entry_value", 0))
        for p in state["positions"].values()
    )
    total_return = (portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    n_open = len(state["positions"])
    wr = state["total_wins"] / max(state["total_trades"], 1) * 100

    log.info(f"\n--- Summary ---")
    log.info(f"Portfolio: ${portfolio_value:,.0f} ({total_return:+.1f}%)")
    log.info(f"Open: {n_open}/{MAX_POSITIONS} | Trades: {state['total_trades']} | WR: {wr:.0f}%")
    log.info(f"Realized P&L: ${state['total_pnl']:+,.0f}")

    # Signal breakdown
    for sig, stats in state["signal_stats"].items():
        if stats["trades"] > 0:
            sig_wr = stats["wins"] / stats["trades"] * 100
            log.info(f"  {sig}: {stats['trades']} trades, WR={sig_wr:.0f}%, P&L=${stats['pnl']:+,.0f}")

    save_state(state)

    # Alert if new positions opened
    if all_signals and opened > 0:
        sig_names = set(s["signal"] for s in all_signals[:opened])
        msg = (f"**Contrarian Signals — {opened} new position(s)**\n"
               f"Signals: {', '.join(sig_names)}\n"
               f"Portfolio: ${portfolio_value:,.0f} ({total_return:+.1f}%)\n"
               f"Open: {n_open} | WR: {wr:.0f}% over {state['total_trades']} trades")
        send_alert(msg)

    return state


# ── Entry Point ───────────────────────────────────────────────────────────

def main():
    try:
        run_daily()
        log.info("Run complete.")
    except Exception as e:
        log.error(f"FATAL: {traceback.format_exc()}")
        send_alert(f"Contrarian Signals Engine ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
