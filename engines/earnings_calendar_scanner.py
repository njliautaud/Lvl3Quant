#!/usr/bin/env python3
"""
Earnings Calendar Iron Condor Scanner (HC #750)
================================================
Scans mega-cap universe for upcoming earnings and generates iron condor trade
ideas based on validated strategy (89% WR, perm p=0.000, profitable on 10/10 tickers).

Thesis: Implied vol consistently overestimates actual earnings moves for mega-caps.
Selling iron condors captures the vol crush after earnings announcements.

Runs daily via PM2 cron. Outputs JSON + human-readable summary.
Sends to Discord webhook if WEBHOOK_URL env var is set.

State: /home/jupiter/Lvl3Quant/state/earnings_calendar/upcoming_plays.json
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

# ── Configuration ───────────────────────────────────────────────────────────
STATE_DIR = Path("/home/jupiter/Lvl3Quant/state/earnings_calendar")
STATE_FILE = STATE_DIR / "upcoming_plays.json"
AUDIT_LOG = STATE_DIR / "scan_audit.jsonl"
STATE_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")

# Validated 10 mega-caps (89% WR backtest)
VALIDATED_TICKERS = [
    "MSFT", "AAPL", "AMZN", "GOOGL", "META", "V", "PG", "JNJ", "UNH", "JPM",
]
# Extended universe (20 additional large-caps)
EXTENDED_TICKERS = [
    "NVDA", "TSLA", "WMT", "HD", "MCD", "COST", "NFLX", "ADBE", "CRM", "AMD",
    "INTC", "PEP", "KO", "LLY", "ABBV", "MRK", "TMO", "ACN", "TXN", "AVGO",
]
UNIVERSE = VALIDATED_TICKERS + EXTENDED_TICKERS

IC_WIDTH = 5.0          # Dollar width of spreads
LOOK_AHEAD_DAYS = 7     # Trading days to scan ahead
VIX_GATE = 25.0         # HC #750: skip if VIX >= 25
IV_RANK_MIN = 50.0      # HC #750: IV rank must be > 50%
QUALITY_PERCENTILE = 70  # Only show top 30% setups

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ECS] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("earnings_calendar_scanner")


# ── Data Fetching ───────────────────────────────────────────────────────────

def get_vix() -> float:
    """Current VIX level for macro gating."""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        log.warning(f"VIX fetch failed: {e}")
    return 20.0  # conservative default


def get_earnings_date(ticker: yf.Ticker) -> Optional[date]:
    """Try multiple methods to get next earnings date."""
    # Method 1: yfinance calendar
    try:
        cal = ticker.calendar
        if cal is not None:
            if isinstance(cal, dict) and "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                if isinstance(ed, list) and len(ed) > 0:
                    return pd.Timestamp(ed[0]).date()
                return pd.Timestamp(ed).date()
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                dates = cal.loc["Earnings Date"]
                if hasattr(dates, "iloc"):
                    return pd.Timestamp(dates.iloc[0]).date()
                return pd.Timestamp(dates).date()
    except Exception:
        pass

    # Method 2: yfinance earnings_dates attribute
    try:
        ed = ticker.earnings_dates
        if ed is not None and not ed.empty:
            future = ed.index[ed.index >= pd.Timestamp.now(tz="America/New_York")]
            if len(future) > 0:
                return future[0].date()
    except Exception:
        pass

    return None


def compute_iv_rank(ticker: yf.Ticker, symbol: str) -> float:
    """Approximate IV rank using 1-year percentile of 20-day realized vol."""
    try:
        hist = ticker.history(period="1y")
        if len(hist) < 40:
            return 50.0  # default if insufficient data
        returns = hist["Close"].pct_change().dropna()
        # Rolling 20-day realized vol (annualized)
        roll_vol = returns.rolling(20).std() * np.sqrt(252)
        roll_vol = roll_vol.dropna()
        if len(roll_vol) < 20:
            return 50.0
        current_vol = float(roll_vol.iloc[-1])
        rank = float((roll_vol < current_vol).sum() / len(roll_vol) * 100)
        return round(rank, 1)
    except Exception as e:
        log.warning(f"{symbol} IV rank calc failed: {e}")
        return 50.0


def compute_historical_earnings_move(ticker: yf.Ticker, symbol: str) -> dict:
    """Estimate historical post-earnings move magnitude (last 4 quarters)."""
    try:
        ed = ticker.earnings_dates
        if ed is None or ed.empty:
            return {"avg_move_pct": 5.0, "max_move_pct": 10.0, "n_quarters": 0}
        # Past earnings dates
        past = ed.index[ed.index < pd.Timestamp.now(tz="America/New_York")]
        if len(past) == 0:
            return {"avg_move_pct": 5.0, "max_move_pct": 10.0, "n_quarters": 0}
        hist = ticker.history(period="2y")
        if hist.empty:
            return {"avg_move_pct": 5.0, "max_move_pct": 10.0, "n_quarters": 0}
        moves = []
        for edate in past[:8]:  # last 8 quarters max
            edate_naive = edate.tz_localize(None) if edate.tzinfo else edate
            # Find nearest trading day
            mask = hist.index.tz_localize(None) if hist.index.tzinfo else hist.index
            idx = mask.searchsorted(edate_naive)
            if idx <= 0 or idx >= len(hist):
                continue
            pre = float(hist["Close"].iloc[idx - 1])
            post = float(hist["Close"].iloc[min(idx, len(hist) - 1)])
            if pre > 0:
                moves.append(abs(post - pre) / pre * 100)
        if not moves:
            return {"avg_move_pct": 5.0, "max_move_pct": 10.0, "n_quarters": 0}
        return {
            "avg_move_pct": round(np.mean(moves), 2),
            "max_move_pct": round(np.max(moves), 2),
            "n_quarters": len(moves),
        }
    except Exception as e:
        log.warning(f"{symbol} earnings history failed: {e}")
        return {"avg_move_pct": 5.0, "max_move_pct": 10.0, "n_quarters": 0}


def compute_atr(hist: pd.DataFrame, period: int = 14) -> float:
    """Average True Range for strike placement."""
    if len(hist) < period + 1:
        return float(hist["Close"].iloc[-1] * 0.02)  # fallback 2%
    h, l, c = hist["High"], hist["Low"], hist["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


# ── Iron Condor Construction ───────────────────────────────────────────────

def build_iron_condor(price: float, atr: float, iv_rank: float, dte: int) -> dict:
    """
    Build iron condor strikes using ATR-based wing placement.
    Short strikes at ±1 ATR, long strikes ±(1 ATR + width).
    """
    short_put = round(price - atr, 2)
    long_put = round(short_put - IC_WIDTH, 2)
    short_call = round(price + atr, 2)
    long_call = round(short_call + IC_WIDTH, 2)

    # Round to nearest $1 for options (most stocks have $1 strike intervals)
    short_put = math.floor(short_put)
    long_put = math.floor(long_put - IC_WIDTH) + IC_WIDTH  # keep width exact
    long_put = short_put - IC_WIDTH
    short_call = math.ceil(short_call)
    long_call = short_call + IC_WIDTH

    # Simplified premium estimate based on vol and DTE
    # Higher IV rank → more premium; shorter DTE → less time value but vol crush benefit
    vol_factor = iv_rank / 100.0  # 0-1 scale
    dte_factor = min(dte / 7.0, 1.5)  # normalize to ~7 DTE
    # Approximate: each spread collects 15-35% of width depending on vol
    spread_premium_pct = 0.15 + 0.20 * vol_factor * dte_factor
    premium_per_spread = round(IC_WIDTH * spread_premium_pct, 2)
    total_premium = round(premium_per_spread * 2, 2)  # both sides
    max_loss = round(IC_WIDTH - total_premium, 2)

    return {
        "long_put": long_put,
        "short_put": short_put,
        "short_call": short_call,
        "long_call": long_call,
        "premium_per_spread": premium_per_spread,
        "total_premium": total_premium,
        "max_loss_per_contract": max_loss * 100,  # in dollars
        "total_premium_dollars": total_premium * 100,
        "width": IC_WIDTH,
        "breakeven_low": round(short_put - total_premium / 2, 2),
        "breakeven_high": round(short_call + total_premium / 2, 2),
    }


def score_setup(iv_rank: float, earnings_hist: dict, is_validated: bool,
                atr_pct: float) -> float:
    """
    Score setup quality 0-100. Factors:
    - IV rank (higher = more premium to sell, weight 30%)
    - Low historical surprise (smaller avg move = safer IC, weight 25%)
    - Validated ticker bonus (proven in backtest, weight 20%)
    - ATR as % of price (lower = less volatile stock = safer, weight 15%)
    - Number of historical quarters available (more data = more confidence, weight 10%)
    """
    # IV rank score: linear 0-100 mapped from 0-100 IV rank
    iv_score = min(iv_rank, 100)

    # Surprise score: lower avg move = better (invert: 10% move → 0, 1% move → 90)
    avg_move = earnings_hist.get("avg_move_pct", 5.0)
    surprise_score = max(0, min(100, 100 - avg_move * 10))

    # Validated ticker bonus
    validated_score = 100 if is_validated else 40

    # ATR% score: lower volatility = better for IC
    atr_score = max(0, min(100, 100 - atr_pct * 30))

    # Data confidence: more quarters = more reliable
    n_q = earnings_hist.get("n_quarters", 0)
    data_score = min(100, n_q * 15)  # 7+ quarters = 100

    score = (iv_score * 0.30 + surprise_score * 0.25 + validated_score * 0.20 +
             atr_score * 0.15 + data_score * 0.10)
    return round(score, 1)


# ── Main Scanner ────────────────────────────────────────────────────────────

def scan() -> dict:
    """Run the full earnings calendar scan. Returns results dict."""
    today = date.today()
    cutoff = today + timedelta(days=LOOK_AHEAD_DAYS + 2)  # buffer for weekends
    log.info(f"Scanning {len(UNIVERSE)} tickers for earnings {today} → {cutoff}")

    # Macro gate: VIX check
    vix = get_vix()
    log.info(f"VIX = {vix:.1f} (gate: < {VIX_GATE})")
    if vix >= VIX_GATE:
        log.warning(f"VIX {vix:.1f} >= {VIX_GATE} — macro gate BLOCKED, no plays generated")
        return {
            "scan_date": str(today),
            "vix": vix,
            "vix_blocked": True,
            "plays": [],
            "summary": f"No plays — VIX at {vix:.1f} exceeds {VIX_GATE} threshold.",
        }

    candidates = []
    skipped = {"no_earnings_date": [], "too_far": [], "low_iv_rank": []}

    for symbol in UNIVERSE:
        try:
            tk = yf.Ticker(symbol)
            edate = get_earnings_date(tk)
            if edate is None:
                skipped["no_earnings_date"].append(symbol)
                continue
            if edate < today or edate > cutoff:
                skipped["too_far"].append(symbol)
                continue

            # Days to earnings
            dte = (edate - today).days

            # Get price data
            hist = tk.history(period="3mo")
            if hist.empty:
                continue
            price = float(hist["Close"].iloc[-1])

            # IV rank gate (HC #750)
            iv_rank = compute_iv_rank(tk, symbol)
            if iv_rank < IV_RANK_MIN:
                skipped["low_iv_rank"].append(f"{symbol} (IVR={iv_rank:.0f})")
                continue

            # Historical earnings moves
            earnings_hist = compute_historical_earnings_move(tk, symbol)

            # ATR for strike placement
            atr = compute_atr(hist)
            atr_pct = atr / price * 100

            # Build the iron condor
            ic = build_iron_condor(price, atr, iv_rank, dte)

            # Score it
            is_validated = symbol in VALIDATED_TICKERS
            quality = score_setup(iv_rank, earnings_hist, is_validated, atr_pct)

            candidates.append({
                "symbol": symbol,
                "earnings_date": str(edate),
                "dte": dte,
                "price": round(price, 2),
                "iv_rank": iv_rank,
                "atr": round(atr, 2),
                "atr_pct": round(atr_pct, 2),
                "historical_moves": earnings_hist,
                "iron_condor": ic,
                "quality_score": quality,
                "is_validated": is_validated,
                "tier": "VALIDATED" if is_validated else "EXTENDED",
            })
            log.info(f"  {symbol}: earnings {edate} (DTE {dte}), price ${price:.0f}, "
                     f"IVR {iv_rank:.0f}%, score {quality:.0f}")

        except Exception as e:
            log.warning(f"  {symbol}: error — {e}")

    # Sort by quality score descending
    candidates.sort(key=lambda x: x["quality_score"], reverse=True)

    # Filter: top percentile only (HC #750)
    if candidates:
        threshold_idx = max(1, len(candidates) * (100 - QUALITY_PERCENTILE) // 100)
        plays = candidates[:threshold_idx]
        filtered_out = candidates[threshold_idx:]
    else:
        plays = []
        filtered_out = []

    # Build summary
    summary_lines = [f"Earnings IC Scanner — {today}  |  VIX: {vix:.1f}"]
    summary_lines.append(f"Scanned {len(UNIVERSE)} tickers, "
                         f"{len(candidates)} candidates, {len(plays)} plays")
    summary_lines.append("")
    if plays:
        for p in plays:
            ic = p["iron_condor"]
            tag = " [VALIDATED]" if p["is_validated"] else ""
            summary_lines.append(
                f"  {p['symbol']}{tag} — earnings {p['earnings_date']} (DTE {p['dte']})")
            summary_lines.append(
                f"    Price: ${p['price']:.2f}  |  IVR: {p['iv_rank']:.0f}%  |  "
                f"Score: {p['quality_score']:.0f}/100")
            summary_lines.append(
                f"    IC: {ic['long_put']}/{ic['short_put']}p — "
                f"{ic['short_call']}/{ic['long_call']}c  |  "
                f"Credit: ${ic['total_premium_dollars']:.0f}  |  "
                f"Max loss: ${ic['max_loss_per_contract']:.0f}")
            summary_lines.append(
                f"    Hist avg move: {p['historical_moves']['avg_move_pct']:.1f}%  |  "
                f"ATR: {p['atr_pct']:.1f}%")
            summary_lines.append("")
    else:
        summary_lines.append("  No qualifying setups this scan.")
    if skipped["no_earnings_date"]:
        summary_lines.append(
            f"  Skipped (no date): {len(skipped['no_earnings_date'])} tickers")

    summary = "\n".join(summary_lines)

    result = {
        "scan_date": str(today),
        "scan_time": datetime.now().isoformat(),
        "vix": vix,
        "vix_blocked": False,
        "universe_size": len(UNIVERSE),
        "candidates_found": len(candidates),
        "plays_shown": len(plays),
        "filtered_below_threshold": len(filtered_out),
        "quality_threshold_pct": QUALITY_PERCENTILE,
        "skipped_summary": {k: len(v) for k, v in skipped.items()},
        "plays": plays,
        "all_candidates": candidates,  # full list for audit
        "summary": summary,
    }
    return result


def save_state(result: dict):
    """Save results to state file (idempotent — overwrites previous scan)."""
    STATE_FILE.write_text(json.dumps(result, indent=2, default=str))
    log.info(f"State saved to {STATE_FILE}")
    # Append to audit log
    audit_entry = {
        "scan_date": result["scan_date"],
        "scan_time": result.get("scan_time"),
        "vix": result["vix"],
        "plays": [p["symbol"] for p in result["plays"]],
        "candidates": result["candidates_found"],
    }
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(audit_entry) + "\n")


def send_discord(summary: str):
    """Send summary to Discord webhook if configured."""
    if not WEBHOOK_URL:
        return
    import urllib.request
    payload = json.dumps({"content": f"```\n{summary[:1900]}\n```"})
    req = urllib.request.Request(
        WEBHOOK_URL,
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        log.info("Discord webhook sent")
    except Exception as e:
        log.warning(f"Discord webhook failed: {e}")


def main():
    log.info("=" * 60)
    log.info("Earnings Calendar IC Scanner starting")
    log.info("=" * 60)

    result = scan()
    save_state(result)

    # Print human-readable summary
    print("\n" + result["summary"])

    # Discord notification
    if WEBHOOK_URL:
        send_discord(result["summary"])

    log.info("Scan complete")
    return result


if __name__ == "__main__":
    main()
