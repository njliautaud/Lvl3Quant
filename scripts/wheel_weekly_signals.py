#!/usr/bin/env python3
"""
wheel_weekly_signals.py — Generate this week's wheel trade signals.

Uses the optimized v2 config:
- 197-ticker universe, 60% margin cap, 4% per name
- 65% profit-take, bear gate (SPY < 50d SMA)
- Weekly DTE (~14 days), 0.25 delta puts

Outputs: ranked list of CSPs to sell this week with strikes, premiums, and margin requirements.
Run every Monday morning to get the week's trade candidates.

Usage:
    python3 scripts/wheel_weekly_signals.py [--capital 100000] [--max-positions 15]
"""
import argparse
import math
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output" / "wheel_signals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========================= OPTIMAL CONFIG ====================================
RISK_FREE = 0.04
PUT_DELTA = 0.25
DTE_TARGET = 14
PROFIT_TAKE = 0.65
VIX_MAX = 35.0
MARGIN_REQ_PCT = 0.20
MAX_PORTFOLIO_MARGIN = 0.60
MAX_PER_NAME_PCT = 0.04
COST_PER_CONTRACT = 0.65

# Full 197-ticker universe (from expanded sweep)
UNIVERSE = [
    # High-beta tech/growth
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'TSLA',
    'NFLX', 'CRM', 'ADBE', 'AVGO', 'ASML', 'ARM', 'APP', 'AFRM',
    'BILL', 'BROS', 'CELH', 'COIN', 'CRWD', 'DDOG', 'ENPH', 'FSLR',
    'HOOD', 'HUBS', 'MELI', 'MSTR', 'NET', 'PANW', 'PLTR', 'RIVN',
    'SHOP', 'SNAP', 'SNOW', 'SQ', 'TTD', 'UBER', 'UPST', 'ZS',
    # Semis
    'AMAT', 'ADI', 'INTC', 'MU', 'QCOM', 'TXN', 'KLAC', 'LRCX', 'MRVL', 'ON',
    # Financials
    'JPM', 'BAC', 'GS', 'MS', 'AXP', 'V', 'MA', 'BLK', 'SCHW', 'C',
    # Healthcare/Biotech
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'AMGN', 'GILD', 'BIIB',
    'BMY', 'CVS', 'ABT', 'TMO', 'ISRG', 'DXCM', 'MRNA',
    # Energy
    'XOM', 'CVX', 'COP', 'VLO', 'SLB', 'OXY', 'DVN', 'HAL', 'MPC', 'EOG',
    # Consumer
    'HD', 'WMT', 'COST', 'TGT', 'NKE', 'SBUX', 'MCD', 'DIS', 'WYNN',
    'BABA', 'PG', 'CL', 'KO', 'PEP',
    # Industrials
    'CAT', 'DE', 'HON', 'UNP', 'BA', 'GE', 'LMT', 'RTX', 'UPS', 'CARR',
    # Utilities
    'NEE', 'DUK', 'SO', 'AEP', 'EXC', 'ED', 'D', 'SRE',
    # REITs
    'AMT', 'PLD', 'DLR', 'SPG', 'O', 'IRM', 'EQIX', 'PSA',
    # Communication
    'VZ', 'T', 'TMUS', 'EA', 'ATVI',
    # Materials
    'LIN', 'APD', 'FCX', 'NEM', 'AA',
    # Other
    'IBM', 'CSCO', 'ORCL', 'NOW', 'PYPL',
]

# ========================= PRICING ==========================================
def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
    return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)

def find_strike(S, sigma, T, delta_target, kind="put"):
    """Find strike for target absolute delta using bisection.
    Put delta (abs) = N(-d1) = _Phi(-d1), ranges 0 to 0.5 for OTM.
    Call delta = N(d1) = _Phi(d1), ranges 0.5 to 1 for OTM.
    """
    if kind == "put":
        lo, hi = S * 0.3, S * 1.0
    else:
        lo, hi = S * 1.0, S * 2.0
    for _ in range(60):
        K = (lo + hi) / 2
        d1 = (math.log(S / K) + (RISK_FREE + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T) + 1e-9)
        if kind == "put":
            delta_abs = _Phi(-d1)  # Put |delta| = N(-d1)
        else:
            delta_abs = _Phi(d1)   # Call delta = N(d1)
        if delta_abs > delta_target:
            # Too ITM — for put: strike too high, need lower; for call: strike too low, need higher
            if kind == "put":
                hi = K
            else:
                lo = K
        else:
            # Too OTM — for put: strike too low, need higher; for call: strike too high, need lower
            if kind == "put":
                lo = K
            else:
                hi = K
    return round(K * 2) / 2


def get_bear_regime():
    """Check if we're currently in bear regime (SPY < 50d SMA)."""
    spy = yf.download("SPY", period="100d", progress=False)
    if spy.empty:
        return False, None, None
    close = float(spy["Close"].iloc[-1].item() if hasattr(spy["Close"].iloc[-1], 'item') else spy["Close"].iloc[-1])
    sma50 = float(spy["Close"].rolling(50).mean().iloc[-1].item() if hasattr(spy["Close"].rolling(50).mean().iloc[-1], 'item') else spy["Close"].rolling(50).mean().iloc[-1])
    is_bear = close < sma50
    return is_bear, close, sma50


def get_current_vix():
    """Get current VIX level."""
    vix = yf.download("^VIX", period="5d", progress=False)
    if vix.empty:
        return 20.0
    val = vix["Close"].iloc[-1]
    return float(val.item() if hasattr(val, 'item') else val)


def generate_signals(capital=100_000, max_positions=15):
    """Generate this week's CSP trade signals."""
    print(f"{'='*70}")
    print(f"WHEEL WEEKLY SIGNALS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    print(f"Capital: ${capital:,.0f} | Max positions: {max_positions}")
    print()

    # Check regime
    is_bear, spy_close, spy_sma = get_bear_regime()
    vix = get_current_vix()

    print(f"Market Regime:")
    print(f"  SPY: ${spy_close:.2f} | 50d SMA: ${spy_sma:.2f}")
    print(f"  {'🔴 BEAR (below SMA) — NO NEW CSPs' if is_bear else '🟢 BULL (above SMA) — CSPs OK'}")
    print(f"  VIX: {vix:.1f} {'(⚠️ HIGH — capped)' if vix > VIX_MAX else '(OK)'}")
    print()

    if is_bear:
        print("BEAR REGIME ACTIVE — No new CSP positions recommended.")
        print("Action: Close existing CSPs if any, wait for SPY > 50d SMA.")
        return []

    if vix > VIX_MAX:
        print(f"VIX too high ({vix:.1f} > {VIX_MAX}). Wait for volatility to settle.")
        return []

    # Download current prices for universe
    print(f"Fetching prices for {len(UNIVERSE)} tickers...")
    data = yf.download(UNIVERSE, period="30d", progress=False, group_by='ticker')

    signals = []
    available_margin = capital * MAX_PORTFOLIO_MARGIN
    per_name_limit = capital * MAX_PER_NAME_PCT

    for ticker in UNIVERSE:
        try:
            if ticker in data.columns.get_level_values(0):
                tdf = data[ticker].dropna()
            else:
                continue

            if len(tdf) < 20:
                continue

            px = float(tdf["Close"].iloc[-1])
            if px <= 0:
                continue

            # Calculate 20-day realized vol
            log_rets = np.log(tdf["Close"] / tdf["Close"].shift(1)).dropna()
            sigma = float(log_rets.std() * np.sqrt(252))
            sigma = max(0.05, min(sigma, 2.0))

            # Check margin
            notional = px * 100
            margin_req = notional * MARGIN_REQ_PCT

            if margin_req > per_name_limit:
                continue
            if margin_req > available_margin:
                continue

            # Calculate strike and premium
            T = DTE_TARGET / 365
            K = find_strike(px, sigma, T, PUT_DELTA, kind="put")
            premium = bs_price(px, K, T, sigma)

            if premium < 0.10:
                continue

            # Calculate key metrics
            prem_yield = premium / K * 100  # Premium as % of strike
            annualized_yield = prem_yield * (365 / DTE_TARGET)
            otm_pct = (px - K) / px * 100  # How far OTM

            signals.append({
                "ticker": ticker,
                "price": round(px, 2),
                "strike": K,
                "premium": round(premium, 2),
                "margin_req": round(margin_req, 0),
                "prem_yield_pct": round(prem_yield, 2),
                "annual_yield_pct": round(annualized_yield, 1),
                "otm_pct": round(otm_pct, 1),
                "sigma": round(sigma, 3),
                "dte": DTE_TARGET,
            })
        except Exception:
            continue

    # Rank by premium yield (best bang for margin buck)
    signals.sort(key=lambda x: x["annual_yield_pct"], reverse=True)

    # Select top N
    selected = signals[:max_positions]

    print(f"\nFound {len(signals)} viable candidates. Top {max_positions}:")
    print()
    print(f"{'Ticker':<7} {'Price':>8} {'Strike':>8} {'Prem':>7} {'Yield%':>7} {'Ann%':>7} {'OTM%':>6} {'Margin':>8}")
    print("-" * 70)

    total_margin = 0
    total_premium = 0
    for s in selected:
        print(f"{s['ticker']:<7} ${s['price']:>7.2f} ${s['strike']:>7.2f} ${s['premium']:>5.2f}  {s['prem_yield_pct']:>5.2f}%  {s['annual_yield_pct']:>5.1f}%  {s['otm_pct']:>4.1f}%  ${s['margin_req']:>7,.0f}")
        total_margin += s["margin_req"]
        total_premium += s["premium"] * 100

    print("-" * 70)
    print(f"{'TOTAL':<7} {'':>8} {'':>8} ${total_premium:>6,.0f}  {'':>7} {'':>7} {'':>6}  ${total_margin:>7,.0f}")
    print(f"\nMargin utilization: {total_margin/capital*100:.1f}% of ${capital:,.0f}")
    print(f"Expected 2-week income: ${total_premium:.0f} ({total_premium/capital*100:.2f}% of capital)")
    print(f"Annualized premium rate: {total_premium/capital*100*(365/DTE_TARGET):.1f}%")

    # Save signals
    out_path = OUT_DIR / f"signals_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out_path, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "regime": {"is_bear": is_bear, "spy": spy_close, "sma50": spy_sma, "vix": vix},
            "config": {
                "capital": capital, "margin_cap": MAX_PORTFOLIO_MARGIN,
                "per_name_pct": MAX_PER_NAME_PCT, "put_delta": PUT_DELTA,
                "dte": DTE_TARGET, "profit_take": PROFIT_TAKE,
            },
            "signals": selected,
            "all_candidates": len(signals),
        }, f, indent=2)

    print(f"\nSaved to {out_path}")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--max-positions", type=int, default=15)
    args = parser.parse_args()
    generate_signals(capital=args.capital, max_positions=args.max_positions)
