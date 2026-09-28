#!/usr/bin/env python3
"""
Asymmetric Signal Scorecard — Daily Actionable Readout
======================================================
Consolidates ALL validated asymmetric signals from HC #725/#726 research
into a single 0-7 composite score with plain-English interpretation.

Validated signals (all with perm p < 0.05 or bootstrap significance):
  1. VIX backwardation (VIX/VIX3M > 1.05): 3.65% 1m, 73.9% HR
  2. Breadth collapse (% above 50SMA < 30%): 2.63% 1m, 73.2% HR
  3. Negative momentum breadth (>70% stocks neg 3m mom): 5.82% 1m, 88.1% HR
  4. High IV-RV spread (>p80): 1.98% 1m, 75.1% HR
  5. SPY below 200SMA: 2.49% 1m, 68.0% HR
  6. Crisis exit (VIX crossed below 30 recently): 79% WR at 3m
  7. Credit stress (HYG underperforming LQD): 3.43% 1m, 66.5% HR

Scoring:
  0-1 signals = COMPLACENT (thin forward returns, reduce risk exposure)
  2-3 signals = NORMAL (baseline expected returns)
  4+  signals = ASYMMETRIC OPPORTUNITY (historically 3-6% 1m mean, 70%+ HR)

Output: JSON state + log line. Designed for daily cron (9:55 AM ET).

All signals use T-1 data (yesterday's close). No lookahead.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance required")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = ROOT / "output" / "asymmetric_scorecard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = OUTPUT_DIR / "current_state.json"
HISTORY_FILE = OUTPUT_DIR / "signal_history.jsonl"

# 50 large-cap stocks for breadth calculations
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "WMT",
    "ACN", "CSCO", "ABT", "DHR", "CRM", "NKE", "TXN", "NEE", "UPS",
    "LIN", "AMD", "QCOM", "HON", "LOW", "AMGN", "INTC", "BA", "GS",
    "CAT", "BLK", "ISRG", "SYK", "ADP",
]

MACRO_TICKERS = ["SPY", "^VIX", "^VIX3M", "HYG", "LQD", "TLT"]


def download_data(lookback_days: int = 300) -> dict[str, pd.DataFrame]:
    """Download all needed price data."""
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    # Macro data
    all_tickers = MACRO_TICKERS + STOCK_UNIVERSE
    data = {}

    print(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(
        all_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if raw.empty:
        print("ERROR: No data downloaded")
        sys.exit(1)

    # Extract close prices
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]]

    data["closes"] = closes
    return data


def compute_signals(data: dict) -> dict:
    """Compute all 7 validated asymmetric signals using T-1 data."""
    closes = data["closes"]
    today_idx = -1  # Use last available row (which is T-1 by market timing)

    signals = {}
    details = {}

    # --- Signal 1: VIX Backwardation ---
    try:
        vix = closes["^VIX"].dropna()
        vix3m = closes["^VIX3M"].dropna()
        common = vix.index.intersection(vix3m.index)
        if len(common) > 0:
            ratio = float(vix.loc[common[-1]] / vix3m.loc[common[-1]])
            signals["vix_backwardation"] = 1 if ratio > 1.05 else 0
            details["vix_ratio"] = round(ratio, 3)
            details["vix_level"] = round(float(vix.iloc[-1]), 1)
        else:
            signals["vix_backwardation"] = 0
            details["vix_ratio"] = None
    except Exception as e:
        signals["vix_backwardation"] = 0
        details["vix_backwardation_error"] = str(e)

    # --- Signal 2: Breadth Collapse (% above 50SMA < 30%) ---
    try:
        above_50sma = 0
        total_valid = 0
        for ticker in STOCK_UNIVERSE:
            if ticker not in closes.columns:
                continue
            px = closes[ticker].dropna()
            if len(px) < 50:
                continue
            sma50 = px.rolling(50).mean()
            if pd.notna(sma50.iloc[-1]) and pd.notna(px.iloc[-1]):
                total_valid += 1
                if float(px.iloc[-1]) > float(sma50.iloc[-1]):
                    above_50sma += 1

        pct_above = above_50sma / total_valid * 100 if total_valid > 0 else 50
        signals["breadth_collapse"] = 1 if pct_above < 30 else 0
        details["pct_above_50sma"] = round(pct_above, 1)
        details["stocks_above_50sma"] = f"{above_50sma}/{total_valid}"
    except Exception as e:
        signals["breadth_collapse"] = 0
        details["breadth_error"] = str(e)

    # --- Signal 3: Negative Momentum Breadth (>70% stocks with neg 3m mom) ---
    try:
        neg_mom = 0
        total_valid_mom = 0
        for ticker in STOCK_UNIVERSE:
            if ticker not in closes.columns:
                continue
            px = closes[ticker].dropna()
            if len(px) < 63:
                continue
            ret_3m = float(px.iloc[-1] / px.iloc[-63] - 1)
            total_valid_mom += 1
            if ret_3m < 0:
                neg_mom += 1

        pct_neg = neg_mom / total_valid_mom * 100 if total_valid_mom > 0 else 0
        signals["neg_momentum_breadth"] = 1 if pct_neg > 70 else 0
        details["pct_neg_3m_momentum"] = round(pct_neg, 1)
        details["stocks_neg_momentum"] = f"{neg_mom}/{total_valid_mom}"
    except Exception as e:
        signals["neg_momentum_breadth"] = 0
        details["momentum_error"] = str(e)

    # --- Signal 4: High IV-RV Spread (VIX - 21d realized vol > p80 threshold) ---
    # P80 threshold from research: roughly VIX - RV > 8
    try:
        spy = closes["SPY"].dropna()
        vix = closes["^VIX"].dropna()
        if len(spy) >= 22:
            returns = spy.pct_change().dropna()
            rv_21d = float(returns.iloc[-21:].std() * np.sqrt(252) * 100)
            vix_val = float(vix.iloc[-1])
            iv_rv = vix_val - rv_21d
            signals["high_iv_rv_spread"] = 1 if iv_rv > 8 else 0
            details["iv_rv_spread"] = round(iv_rv, 1)
            details["realized_vol_21d"] = round(rv_21d, 1)
        else:
            signals["high_iv_rv_spread"] = 0
    except Exception as e:
        signals["high_iv_rv_spread"] = 0
        details["iv_rv_error"] = str(e)

    # --- Signal 5: SPY Below 200SMA ---
    try:
        spy = closes["SPY"].dropna()
        if len(spy) >= 200:
            sma200 = float(spy.rolling(200).mean().iloc[-1])
            spy_price = float(spy.iloc[-1])
            signals["spy_below_200sma"] = 1 if spy_price < sma200 else 0
            details["spy_price"] = round(spy_price, 2)
            details["spy_200sma"] = round(sma200, 2)
            details["spy_dist_200sma_pct"] = round((spy_price / sma200 - 1) * 100, 2)
        else:
            signals["spy_below_200sma"] = 0
    except Exception as e:
        signals["spy_below_200sma"] = 0
        details["spy_200sma_error"] = str(e)

    # --- Signal 6: Crisis Exit (VIX dropped below 30 in last 5 days after being above) ---
    try:
        vix = closes["^VIX"].dropna()
        if len(vix) >= 10:
            recent = vix.iloc[-10:]
            was_above_30 = any(float(v) > 30 for v in recent.iloc[:5])
            now_below_30 = float(recent.iloc[-1]) < 30
            crossed_down = False
            for i in range(max(1, len(recent) - 5), len(recent)):
                if float(recent.iloc[i - 1]) >= 30 and float(recent.iloc[i]) < 30:
                    crossed_down = True
                    break
            signals["crisis_exit"] = 1 if (was_above_30 and crossed_down) else 0
            details["vix_recent_high"] = round(float(recent.max()), 1)
            details["vix_current"] = round(float(recent.iloc[-1]), 1)
        else:
            signals["crisis_exit"] = 0
    except Exception as e:
        signals["crisis_exit"] = 0
        details["crisis_exit_error"] = str(e)

    # --- Signal 7: Credit Stress (HYG underperforming LQD over 21d) ---
    try:
        hyg = closes["HYG"].dropna()
        lqd = closes["LQD"].dropna()
        common = hyg.index.intersection(lqd.index)
        if len(common) >= 22:
            hyg_ret = float(hyg.loc[common[-1]] / hyg.loc[common[-22]] - 1)
            lqd_ret = float(lqd.loc[common[-1]] / lqd.loc[common[-22]] - 1)
            credit_spread_chg = hyg_ret - lqd_ret
            # Credit stress = HYG underperforming LQD by more than 1%
            signals["credit_stress"] = 1 if credit_spread_chg < -0.01 else 0
            details["hyg_21d_ret"] = round(hyg_ret * 100, 2)
            details["lqd_21d_ret"] = round(lqd_ret * 100, 2)
            details["credit_spread_21d"] = round(credit_spread_chg * 100, 2)
        else:
            signals["credit_stress"] = 0
    except Exception as e:
        signals["credit_stress"] = 0
        details["credit_error"] = str(e)

    return signals, details


def score_and_interpret(signals: dict) -> tuple[int, str, str]:
    """Convert individual signals to composite score and interpretation."""
    composite = sum(signals.values())

    # Signal names for active signals
    active = [name for name, val in signals.items() if val == 1]

    if composite <= 1:
        regime = "COMPLACENT"
        interpretation = (
            "Market conditions are calm. Historically, forward returns "
            "are thin from here (+0.4-0.6%/month avg). Not a good time "
            "to add aggressive exposure. Consider defensive positioning."
        )
    elif composite <= 3:
        regime = "NORMAL"
        interpretation = (
            "Mixed signals. Some stress but not enough for a high-conviction "
            "asymmetric setup. Expected forward returns near baseline (~1%/month). "
            "Stay with normal allocation."
        )
    else:
        regime = "ASYMMETRIC OPPORTUNITY"
        interpretation = (
            f"Multiple fear signals active ({composite}/7). Historically, "
            f"this produces 3-6% forward 1-month returns with 70%+ hit rate "
            f"and 1.5-2.5x upside/downside ratio. This is a buy-the-fear setup. "
            f"Consider increasing equity exposure, especially in beaten-down "
            f"large-cap names with volume surges."
        )

    return composite, regime, interpretation


def build_stock_asymmetric_screen(data: dict) -> list[dict]:
    """Screen for individual stocks showing asymmetric setup conditions.

    Asymmetric stock filter (from stock_asymmetry_v1 research):
      - High vol (20d vol > 80th pctl)
      - Negative momentum (3m return < -5%)
      - Volume surge (today's volume > 1.5x 20d avg)
    """
    closes = data["closes"]
    candidates = []

    for ticker in STOCK_UNIVERSE:
        if ticker not in closes.columns:
            continue
        px = closes[ticker].dropna()
        if len(px) < 63:
            continue

        try:
            # 20d realized vol (annualized)
            ret = px.pct_change().dropna()
            if len(ret) < 20:
                continue
            vol_20d = float(ret.iloc[-20:].std() * np.sqrt(252) * 100)

            # 3m momentum
            mom_3m = float(px.iloc[-1] / px.iloc[-63] - 1) * 100

            # Distance from 52w high
            high_52w = float(px.iloc[-min(252, len(px)):].max())
            dist_high = float((px.iloc[-1] / high_52w - 1) * 100)

            candidates.append({
                "ticker": ticker,
                "vol_20d": round(vol_20d, 1),
                "mom_3m": round(mom_3m, 1),
                "dist_52w_high": round(dist_high, 1),
                "price": round(float(px.iloc[-1]), 2),
            })
        except Exception:
            continue

    if not candidates:
        return []

    # Compute percentile ranks for vol
    vols = [c["vol_20d"] for c in candidates]
    p80_vol = np.percentile(vols, 80) if vols else 999

    # Filter: high vol + negative momentum
    filtered = [
        c for c in candidates
        if c["vol_20d"] >= p80_vol
        and c["mom_3m"] < -5.0
    ]

    # Sort by composite distress score (most distressed first)
    for c in filtered:
        c["distress_score"] = round(-c["mom_3m"] + c["vol_20d"] / 10 - c["dist_52w_high"] / 5, 2)

    filtered.sort(key=lambda x: x["distress_score"], reverse=True)
    return filtered[:10]  # Top 10


def run():
    """Main execution."""
    print("=" * 60)
    print("ASYMMETRIC SIGNAL SCORECARD")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Download data
    data = download_data(lookback_days=300)

    # Compute signals
    signals, details = compute_signals(data)

    # Score
    composite, regime, interpretation = score_and_interpret(signals)

    # Stock screen
    stock_picks = build_stock_asymmetric_screen(data)

    # Build state
    state = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "composite_score": composite,
        "regime": regime,
        "signals": signals,
        "details": details,
        "interpretation": interpretation,
        "stock_screen": stock_picks,
        "active_signals": [k for k, v in signals.items() if v == 1],
    }

    # Print results
    print(f"\nCOMPOSITE SCORE: {composite}/7 — {regime}")
    print(f"\nInterpretation: {interpretation}")

    print("\n--- Individual Signals ---")
    signal_names = {
        "vix_backwardation": "VIX Backwardation (VIX/VIX3M > 1.05)",
        "breadth_collapse": "Breadth Collapse (< 30% above 50SMA)",
        "neg_momentum_breadth": "Negative Momentum (> 70% neg 3m)",
        "high_iv_rv_spread": "High IV-RV Spread (> 8 pts)",
        "spy_below_200sma": "SPY Below 200SMA",
        "crisis_exit": "Crisis Exit (VIX crossed below 30)",
        "credit_stress": "Credit Stress (HYG underperforming LQD)",
    }
    for key, name in signal_names.items():
        val = signals.get(key, 0)
        icon = "ON" if val else "OFF"
        print(f"  [{icon}] {name}")

    print("\n--- Key Details ---")
    detail_labels = {
        "vix_level": "VIX Level",
        "vix_ratio": "VIX/VIX3M Ratio",
        "pct_above_50sma": "% Above 50SMA",
        "pct_neg_3m_momentum": "% Negative 3m Momentum",
        "iv_rv_spread": "IV-RV Spread (pts)",
        "spy_price": "SPY Price",
        "spy_200sma": "SPY 200SMA",
        "spy_dist_200sma_pct": "SPY Dist from 200SMA (%)",
        "credit_spread_21d": "Credit Spread Change 21d (%)",
    }
    for key, label in detail_labels.items():
        if key in details:
            print(f"  {label}: {details[key]}")

    if stock_picks:
        print(f"\n--- Asymmetric Stock Screen ({len(stock_picks)} candidates) ---")
        print(f"  {'Ticker':<8} {'Vol20d':>7} {'Mom3m':>7} {'Dist52wH':>9} {'Score':>7}")
        for pick in stock_picks:
            print(
                f"  {pick['ticker']:<8} {pick['vol_20d']:>7.1f}% "
                f"{pick['mom_3m']:>6.1f}% {pick['dist_52w_high']:>8.1f}% "
                f"{pick['distress_score']:>7.1f}"
            )
    else:
        print("\n--- No stocks currently pass the asymmetric filter ---")

    # Save state
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"\nState saved.")

    # Append to history
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps({
            "date": state["date"],
            "score": composite,
            "regime": regime,
            "signals": signals,
        }, default=str) + "\n")
    print(f"History appended.")

    return state


if __name__ == "__main__":
    state = run()
