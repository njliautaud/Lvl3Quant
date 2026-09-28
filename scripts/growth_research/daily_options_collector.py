#!/usr/bin/env python3
"""
Daily Options Flow Collector
Collects options volume, OI, and unusual activity for top 100 liquid S&P 500 stocks.
Designed to run at 4:30 PM ET weekdays. Saves daily parquet snapshots.

Usage:
    python daily_options_collector.py          # collect today
    python daily_options_collector.py 2026-07-14  # collect specific date (uses live chain)
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "/home/jupiter/Lvl3Quant/data/options_flow/daily"
ALERT_DIR = "/home/jupiter/Lvl3Quant/data/options_flow/alerts"
MAX_DTE = 60  # only expirations within 60 days
SLEEP_BETWEEN = 0.5  # seconds between tickers (rate limit)
UNUSUAL_VOL_MULT = 3.0  # total vol > 3x 20d avg → unusual
UNUSUAL_STRIKE_OI_MULT = 10.0  # single strike vol > 10x its OI → unusual
OTM_THRESHOLD = 0.02  # 2% OTM threshold

# Top 100 most liquid S&P 500 stocks (by typical daily volume)
UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AVGO", "ORCL", "CRM",
    "AMD", "INTC", "ADBE", "CSCO", "QCOM", "AMAT", "MU", "NFLX", "INTU", "NOW",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "BMY", "AMGN",
    # Consumer
    "WMT", "HD", "COST", "NKE", "SBUX", "MCD", "TGT", "LOW", "TJX", "BKNG",
    # Industrials
    "CAT", "BA", "GE", "HON", "UPS", "RTX", "DE", "LMT", "MMM", "UNP",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
    # Communication / Media
    "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "PLTR", "WBD", "NWSA", "EA",
    # Materials / Other
    "LIN", "APD", "FCX", "NEM", "DOW", "DD", "PPG", "SHW", "ECL", "VMC",
    # High-volume ETFs that trade like stocks (useful flow signals)
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLV", "XLI", "GLD", "SLV",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("options_flow")


def collect_ticker(ticker: str, today: datetime) -> dict | None:
    """Collect options flow data for a single ticker. Returns dict or None on failure."""
    try:
        tk = yf.Ticker(ticker)

        # Get current price
        hist = tk.history(period="1d")
        if hist.empty:
            log.warning(f"{ticker}: no price data, skipping")
            return None
        current_price = float(hist["Close"].iloc[-1])

        # 20-day average stock volume
        hist_20d = tk.history(period="1mo")
        avg_volume_20d = float(hist_20d["Volume"].mean()) if not hist_20d.empty else 0.0

        # Get expirations within MAX_DTE days
        try:
            expirations = tk.options
        except Exception:
            log.warning(f"{ticker}: no options chain available")
            return None

        if not expirations:
            log.warning(f"{ticker}: no expirations found")
            return None

        cutoff = today + timedelta(days=MAX_DTE)
        valid_exps = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            if exp_date <= cutoff:
                valid_exps.append(exp_str)

        if not valid_exps:
            log.warning(f"{ticker}: no expirations within {MAX_DTE} days")
            return None

        # Aggregate across all valid expirations
        total_call_vol = 0
        total_put_vol = 0
        total_call_oi = 0
        total_put_oi = 0
        otm_call_vol = 0
        otm_put_vol = 0
        max_call_strike_vol = 0
        max_put_strike_vol = 0
        max_call_strike_vol_vs_oi = 0.0
        max_put_strike_vol_vs_oi = 0.0
        atm_call_iv = np.nan
        atm_put_iv = np.nan
        atm_iv_dist = float("inf")
        n_expirations = len(valid_exps)

        for exp_str in valid_exps:
            try:
                chain = tk.option_chain(exp_str)
            except Exception as e:
                log.debug(f"{ticker} {exp_str}: chain error: {e}")
                continue

            calls = chain.calls
            puts = chain.puts

            if not calls.empty:
                cv = calls["volume"].fillna(0).astype(float)
                coi = calls["openInterest"].fillna(0).astype(float)
                total_call_vol += int(cv.sum())
                total_call_oi += int(coi.sum())

                # Max single-strike call volume
                max_cv = int(cv.max())
                if max_cv > max_call_strike_vol:
                    max_call_strike_vol = max_cv
                    idx = cv.idxmax()
                    oi_at_max = coi.loc[idx] if idx in coi.index else 1
                    max_call_strike_vol_vs_oi = max_cv / max(oi_at_max, 1)

                # OTM calls (strike > price * 1.02)
                otm_mask = calls["strike"] > current_price * (1 + OTM_THRESHOLD)
                otm_call_vol += int(cv[otm_mask].sum())

                # ATM IV (closest strike to current price)
                if "impliedVolatility" in calls.columns:
                    dist = (calls["strike"] - current_price).abs()
                    min_dist = dist.min()
                    if min_dist < atm_iv_dist:
                        atm_iv_dist = min_dist
                        atm_idx = dist.idxmin()
                        iv_val = calls.loc[atm_idx, "impliedVolatility"]
                        if pd.notna(iv_val) and iv_val > 0:
                            atm_call_iv = float(iv_val)

            if not puts.empty:
                pv = puts["volume"].fillna(0).astype(float)
                poi = puts["openInterest"].fillna(0).astype(float)
                total_put_vol += int(pv.sum())
                total_put_oi += int(poi.sum())

                max_pv = int(pv.max())
                if max_pv > max_put_strike_vol:
                    max_put_strike_vol = max_pv
                    idx = pv.idxmax()
                    oi_at_max = poi.loc[idx] if idx in poi.index else 1
                    max_put_strike_vol_vs_oi = max_pv / max(oi_at_max, 1)

                # OTM puts (strike < price * 0.98)
                otm_mask = puts["strike"] < current_price * (1 - OTM_THRESHOLD)
                otm_put_vol += int(pv[otm_mask].sum())

                # ATM put IV
                if "impliedVolatility" in puts.columns:
                    dist = (puts["strike"] - current_price).abs()
                    min_dist_p = dist.min()
                    if min_dist_p < atm_iv_dist * 1.5:  # allow some slack
                        atm_idx = dist.idxmin()
                        iv_val = puts.loc[atm_idx, "impliedVolatility"]
                        if pd.notna(iv_val) and iv_val > 0:
                            atm_put_iv = float(iv_val)

        # Ratios
        pc_vol_ratio = total_put_vol / max(total_call_vol, 1)
        pc_oi_ratio = total_put_oi / max(total_call_oi, 1)

        # Unusual activity flags
        # We don't have historical options volume avg yet, so use a heuristic:
        # unusual if single-strike vol > 10x its OI
        unusual_call = bool(max_call_strike_vol_vs_oi > UNUSUAL_STRIKE_OI_MULT)
        unusual_put = bool(max_put_strike_vol_vs_oi > UNUSUAL_STRIKE_OI_MULT)

        # ATM IV (average of call and put if both available)
        if pd.notna(atm_call_iv) and pd.notna(atm_put_iv):
            atm_iv = (atm_call_iv + atm_put_iv) / 2
        elif pd.notna(atm_call_iv):
            atm_iv = atm_call_iv
        elif pd.notna(atm_put_iv):
            atm_iv = atm_put_iv
        else:
            atm_iv = np.nan

        return {
            "ticker": ticker,
            "date": today.strftime("%Y-%m-%d"),
            "price": round(current_price, 2),
            "avg_volume_20d": round(avg_volume_20d, 0),
            "n_expirations": n_expirations,
            "total_call_vol": total_call_vol,
            "total_put_vol": total_put_vol,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "pc_vol_ratio": round(pc_vol_ratio, 4),
            "pc_oi_ratio": round(pc_oi_ratio, 4),
            "otm_call_vol": otm_call_vol,
            "otm_put_vol": otm_put_vol,
            "max_call_strike_vol": max_call_strike_vol,
            "max_put_strike_vol": max_put_strike_vol,
            "max_call_strike_vol_vs_oi": round(max_call_strike_vol_vs_oi, 2),
            "max_put_strike_vol_vs_oi": round(max_put_strike_vol_vs_oi, 2),
            "atm_iv": round(atm_iv, 4) if pd.notna(atm_iv) else None,
            "unusual_call_volume": unusual_call,
            "unusual_put_volume": unusual_put,
        }

    except Exception as e:
        log.error(f"{ticker}: unexpected error: {e}")
        return None


def main():
    # Determine date
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
        today = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        today = datetime.now()

    date_label = today.strftime("%Y-%m-%d")

    # Idempotency check
    out_path = os.path.join(DATA_DIR, f"{date_label}.parquet")
    if os.path.exists(out_path):
        log.info(f"Already collected for {date_label}, skipping. ({out_path})")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ALERT_DIR, exist_ok=True)

    log.info(f"Starting options flow collection for {date_label} — {len(UNIVERSE)} tickers")
    t0 = time.time()

    results = []
    errors = []
    for i, ticker in enumerate(UNIVERSE):
        row = collect_ticker(ticker, today)
        if row is not None:
            results.append(row)
        else:
            errors.append(ticker)

        # Progress log every 20 tickers
        if (i + 1) % 20 == 0:
            log.info(f"  Progress: {i+1}/{len(UNIVERSE)} tickers ({len(results)} ok, {len(errors)} failed)")

        if i < len(UNIVERSE) - 1:
            time.sleep(SLEEP_BETWEEN)

    elapsed = time.time() - t0

    if not results:
        log.error("No data collected — all tickers failed")
        return

    df = pd.DataFrame(results)

    # Save main snapshot
    df.to_parquet(out_path, index=False)

    # Save alerts file (unusual activity only)
    alerts = df[df["unusual_call_volume"] | df["unusual_put_volume"]].copy()
    if not alerts.empty:
        alert_path = os.path.join(ALERT_DIR, f"{date_label}_alerts.parquet")
        alerts.to_parquet(alert_path, index=False)
        alert_count = len(alerts)
    else:
        alert_count = 0

    # Summary stats
    avg_pc = df["pc_vol_ratio"].mean()
    median_iv = df["atm_iv"].median()

    summary = (
        f"Options flow {date_label}: {len(results)} tickers collected, "
        f"{len(errors)} failed, {alert_count} unusual alerts, "
        f"avg P/C ratio={avg_pc:.2f}, median ATM IV={median_iv:.1%}, "
        f"elapsed={elapsed:.0f}s"
    )
    log.info(summary)
    # One-line for PM2
    print(summary)


if __name__ == "__main__":
    main()
