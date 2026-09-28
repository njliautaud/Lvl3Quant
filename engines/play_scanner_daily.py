#!/usr/bin/env python3
"""
Play Scanner Pre-Market Cron
=============================
Scans sector ETFs, index ETFs, and top S&P 500 stocks for actionable setups.

Setup types:
  A) Momentum continuation: above all SMAs, RSI 55-70, MFI>60, OBV rising, 1-3% pullback
  B) Oversold bounce: RSI<35, below 20 SMA, BUT above 200 SMA, volume spike
  C) Flow divergence: OBV rising while price flat/down, MFI turning up from <30

Runs at 8:30 AM ET (pre-market) via PM2 cron: "30 12 * * 1-5"
State: /home/jupiter/Lvl3Quant/state/play_scanner_state.json
History: /home/jupiter/Lvl3Quant/state/play_scanner_history.jsonl
"""
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

ET = pytz.timezone("US/Eastern")
STATE_DIR = Path("/home/jupiter/Lvl3Quant/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "play_scanner_state.json"
HISTORY_FILE = STATE_DIR / "play_scanner_history.jsonl"

# ── Universes ──
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]
INDEX_ETFS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "SLV"]
TOP_30_SP500 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "V", "XOM", "MA", "COST", "PG", "JNJ", "HD", "ABBV",
    "MRK", "WMT", "BAC", "CRM", "NFLX", "AMD", "ORCL", "KO", "PEP", "TMO",
]

ALL_TICKERS = SECTOR_ETFS + INDEX_ETFS + TOP_30_SP500


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


def analyze_ticker(ticker: str) -> dict | None:
    """Compute all indicators and check for setups."""
    try:
        df = yf.download(ticker, period="120d", interval="1d", progress=False, timeout=10)
        if df.empty or len(df) < 60:
            return None

        # Flatten multi-level columns if present
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

        # Momentum
        mom_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0
        mom_20d = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0
        mom_60d = float((close.iloc[-1] / close.iloc[-61] - 1) * 100) if len(close) >= 61 else 0

        # SMAs
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.rolling(len(close)).mean().iloc[-1])

        above_sma20 = price > sma20
        above_sma50 = price > sma50
        above_sma200 = price > sma200

        # Volume ratio (today vs 20d avg)
        vol_ratio = float(volume.iloc[-1] / volume.iloc[-20:].mean()) if volume.iloc[-20:].mean() > 0 else 1.0

        # Bollinger position
        bb_pos = float(compute_bollinger_position(close).iloc[-1])

        # Pullback from recent high (5d)
        recent_high = float(high.iloc[-5:].max())
        pullback_pct = (recent_high - price) / recent_high * 100

        # ATR for context
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_pct = float(tr.rolling(14).mean().iloc[-1] / price * 100)

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

        # ── Setup B: Oversold Bounce ──
        if (rsi < 35
                and not above_sma20
                and above_sma200
                and vol_ratio > 1.3):
            result["setups"].append({
                "type": "B_oversold_bounce",
                "score": round((35 - rsi) / 35 * 0.4 + min((vol_ratio - 1) / 2, 1) * 0.3 + (1 if above_sma200 else 0) * 0.3, 3),
                "reason": f"RSI {rsi:.0f} oversold, below 20 SMA, above 200 SMA, vol ratio {vol_ratio:.1f}x",
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
    print(f"=== Play Scanner — {now.strftime('%Y-%m-%d %H:%M ET')} ===")
    print(f"Scanning {len(ALL_TICKERS)} tickers...")

    results = []
    for i, ticker in enumerate(ALL_TICKERS):
        if i % 10 == 0 and i > 0:
            print(f"  Scanned {i}/{len(ALL_TICKERS)}...")
        r = analyze_ticker(ticker)
        if r:
            results.append(r)

    # Score and rank — use best setup score per ticker
    for r in results:
        r["best_score"] = max(s["score"] for s in r["setups"])
    results.sort(key=lambda x: x["best_score"], reverse=True)

    top5 = results[:5]

    # Save state
    state = {
        "scan_time": now.isoformat(),
        "total_scanned": len(ALL_TICKERS),
        "setups_found": len(results),
        "top_5": top5,
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))

    # Append history
    history_entry = {
        "scan_time": now.isoformat(),
        "setups_found": len(results),
        "top_5_tickers": [r["ticker"] for r in top5],
        "top_5_types": [r["setups"][0]["type"] for r in top5] if top5 else [],
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(history_entry) + "\n")

    # Print summary
    print(f"\n--- Results: {len(results)} setups found ---")
    for r in top5:
        best_setup = max(r["setups"], key=lambda s: s["score"])
        print(f"  {r['ticker']:6s} | ${r['price']:<8.2f} | {best_setup['type']:25s} | score={best_setup['score']:.3f}")
        print(f"         RSI={r['rsi']:.0f} MFI={r['mfi']:.0f} OBV_up={r['obv_rising']} Vol={r['vol_ratio']:.1f}x")
        print(f"         {best_setup['reason']}")

    if not top5:
        print("  No setups found today.")

    print(f"\nState saved to {STATE_FILE}")
    print("Done.")


if __name__ == "__main__":
    run_scan()
