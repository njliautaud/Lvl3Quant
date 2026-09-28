#!/usr/bin/env python3
"""
Daily Macro Intelligence Collector (HC #806)
Pulls comprehensive macro data: rates, commodities, equities, vol, credit, sentiment.
Stores daily snapshots for trend analysis and regime classification.

Usage:
    python3 macro_collector.py              # Collect today's snapshot
    python3 macro_collector.py --regime     # Collect + print regime classification
    python3 macro_collector.py --history 5  # Show last 5 days trend
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

MACRO_DIR = Path("/home/jupiter/Lvl3Quant/data/macro")
MACRO_DIR.mkdir(parents=True, exist_ok=True)

# ── Asset Universe ──────────────────────────────────────────────────────────

TREASURY_TICKERS = {
    "US02Y": "^IRX",       # 13-week T-bill (proxy for short end)
    "US10Y": "^TNX",       # 10-year yield
    "US30Y": "^TYX",       # 30-year yield
    "TLT":   "TLT",        # Long bond ETF
    "IEF":   "IEF",        # 7-10yr treasury ETF
    "SHY":   "SHY",        # 1-3yr treasury ETF
}

COMMODITY_TICKERS = {
    "Gold":   "GLD",
    "Silver": "SLV",
    "Oil":    "USO",
    "Copper": "CPER",      # Copper ETF
    "NatGas": "UNG",
    "DBA":    "DBA",       # Agriculture
}

EQUITY_INDEX_TICKERS = {
    "SPY":  "SPY",
    "QQQ":  "QQQ",
    "IWM":  "IWM",
    "DIA":  "DIA",
    "VTI":  "VTI",
}

SECTOR_TICKERS = {
    "XLK": "XLK",  # Tech
    "XLF": "XLF",  # Financials
    "XLE": "XLE",  # Energy
    "XLU": "XLU",  # Utilities
    "XLP": "XLP",  # Staples
    "XLY": "XLY",  # Discretionary
    "XLC": "XLC",  # Comms
    "XLI": "XLI",  # Industrials
    "XLB": "XLB",  # Materials
    "XLV": "XLV",  # Healthcare
    "XLRE": "XLRE", # Real Estate
}

VOLATILITY_TICKERS = {
    "VIX":    "^VIX",
    "VIX3M":  "^VIX3M",   # 3-month VIX (term structure)
    "VVIX":   "^VVIX",    # Vol of vol
}

CREDIT_TICKERS = {
    "HYG": "HYG",  # High yield
    "LQD": "LQD",  # Investment grade
    "JNK": "JNK",  # Junk bonds
}

CURRENCY_TICKERS = {
    "DXY":     "DX-Y.NYB",  # Dollar index
    "EUR_USD": "EURUSD=X",
    "GBP_USD": "GBPUSD=X",
    "USD_JPY": "USDJPY=X",
}

SPECIAL_TICKERS = {
    "Bitcoin":  "BTC-USD",
    "Ethereum": "ETH-USD",
}

# Newsletter thesis tickers (HC #806 R4)
DEBASEMENT_TICKERS = {
    "TIP":   "TIP",        # TIPS ETF — real yield proxy (rising TIP = falling real yields = gold bullish)
    "GDX":   "GDX",        # Gold miners ETF
    "GDXJ":  "GDXJ",       # Junior gold miners
    "ITA":   "ITA",        # Defense/aerospace ETF (Fortress America)
    "XAR":   "XAR",        # S&P Aerospace & Defense
    "EEM":   "EEM",        # Emerging markets (Leapfrog hedge)
    "REMX":  "REMX",       # Rare earth/strategic metals
}


def fetch_prices(ticker_dict: dict, period: str = "5d") -> dict:
    """Fetch latest prices and changes for a group of tickers."""
    results = {}
    all_tickers = list(ticker_dict.values())

    try:
        data = yf.download(all_tickers, period=period, progress=False, threads=True)
        if data.empty:
            return results

        for name, ticker in ticker_dict.items():
            try:
                if len(all_tickers) == 1:
                    close = data["Close"]
                else:
                    close = data["Close"][ticker] if ticker in data["Close"].columns else None

                if close is None or close.dropna().empty:
                    continue

                close = close.dropna()
                current = float(close.iloc[-1])
                prev = float(close.iloc[-2]) if len(close) > 1 else current
                chg_pct = ((current - prev) / prev) * 100 if prev != 0 else 0

                # 5-day change
                first = float(close.iloc[0]) if len(close) > 0 else current
                chg_5d_pct = ((current - first) / first) * 100 if first != 0 else 0

                results[name] = {
                    "price": round(current, 4),
                    "change_1d_pct": round(chg_pct, 3),
                    "change_5d_pct": round(chg_5d_pct, 3),
                }
            except Exception as e:
                results[name] = {"error": str(e)}
    except Exception as e:
        print(f"  [WARN] Batch download failed: {e}", file=sys.stderr)

    return results


def compute_yield_curve(rates: dict) -> dict:
    """Compute yield curve metrics from rate data."""
    curve = {}

    us10y = rates.get("US10Y", {}).get("price")
    us02y_proxy = rates.get("US02Y", {}).get("price")  # 13-week as proxy
    us30y = rates.get("US30Y", {}).get("price")

    if us10y and us02y_proxy:
        curve["2s10s_spread"] = round(us10y - us02y_proxy, 3)
        curve["2s10s_inverted"] = us10y < us02y_proxy

    if us30y and us02y_proxy:
        curve["2s30s_spread"] = round(us30y - us02y_proxy, 3)

    if us30y and us10y:
        curve["10s30s_spread"] = round(us30y - us10y, 3)

    return curve


def compute_credit_spread(credit: dict) -> dict:
    """Compute credit spread proxies."""
    spread = {}
    hyg = credit.get("HYG", {}).get("price")
    lqd = credit.get("LQD", {}).get("price")

    if hyg and lqd:
        # HYG/LQD ratio — declining = widening spreads = risk-off
        spread["hyg_lqd_ratio"] = round(hyg / lqd, 4)

    return spread


def compute_vol_term_structure(vol: dict) -> dict:
    """Analyze VIX term structure."""
    ts = {}
    vix = vol.get("VIX", {}).get("price")
    vix3m = vol.get("VIX3M", {}).get("price")

    if vix and vix3m:
        ts["vix_vix3m_ratio"] = round(vix / vix3m, 4)
        ts["contango"] = vix < vix3m  # Normal = contango (risk-on)
        ts["backwardation"] = vix > vix3m  # Inverted = fear (risk-off)

    return ts


def classify_regime(snapshot: dict) -> dict:
    """
    Classify macro regime based on collected data.

    Risk-On indicators:
      - VIX < 18, falling
      - Yields stable or falling
      - SPY rising
      - HYG/LQD ratio rising (credit tightening)
      - DXY falling
      - Commodities mixed/rising
      - VIX in contango

    Risk-Off indicators:
      - VIX > 22, rising
      - Yields spiking
      - SPY falling
      - HYG/LQD falling (credit widening)
      - DXY rising (flight to safety)
      - Gold rising, copper falling
      - VIX in backwardation

    Transition: mixed signals
    """
    scores = {"risk_on": 0, "risk_off": 0, "signals": []}

    # VIX level
    vix = snapshot.get("volatility", {}).get("VIX", {}).get("price")
    if vix:
        if vix < 16:
            scores["risk_on"] += 2
            scores["signals"].append(f"VIX low ({vix:.1f})")
        elif vix < 20:
            scores["risk_on"] += 1
            scores["signals"].append(f"VIX moderate ({vix:.1f})")
        elif vix < 25:
            scores["risk_off"] += 1
            scores["signals"].append(f"VIX elevated ({vix:.1f})")
        else:
            scores["risk_off"] += 2
            scores["signals"].append(f"VIX high ({vix:.1f})")

    # VIX change
    vix_chg = snapshot.get("volatility", {}).get("VIX", {}).get("change_1d_pct", 0)
    if vix_chg > 5:
        scores["risk_off"] += 1
        scores["signals"].append(f"VIX spiking +{vix_chg:.1f}%")
    elif vix_chg < -5:
        scores["risk_on"] += 1
        scores["signals"].append(f"VIX dropping {vix_chg:.1f}%")

    # VIX term structure
    ts = snapshot.get("vol_term_structure", {})
    if ts.get("contango"):
        scores["risk_on"] += 1
        scores["signals"].append("VIX contango (normal)")
    elif ts.get("backwardation"):
        scores["risk_off"] += 1
        scores["signals"].append("VIX backwardation (fear)")

    # SPY direction
    spy_chg = snapshot.get("equity_indices", {}).get("SPY", {}).get("change_1d_pct", 0)
    spy_5d = snapshot.get("equity_indices", {}).get("SPY", {}).get("change_5d_pct", 0)
    if spy_chg > 0.5:
        scores["risk_on"] += 1
        scores["signals"].append(f"SPY up {spy_chg:.2f}%")
    elif spy_chg < -0.5:
        scores["risk_off"] += 1
        scores["signals"].append(f"SPY down {spy_chg:.2f}%")
    if spy_5d > 1:
        scores["risk_on"] += 1
    elif spy_5d < -1:
        scores["risk_off"] += 1

    # Dollar
    dxy_chg = snapshot.get("currencies", {}).get("DXY", {}).get("change_1d_pct", 0)
    if dxy_chg > 0.3:
        scores["risk_off"] += 1
        scores["signals"].append(f"Dollar rising +{dxy_chg:.2f}%")
    elif dxy_chg < -0.3:
        scores["risk_on"] += 1
        scores["signals"].append(f"Dollar falling {dxy_chg:.2f}%")

    # Gold — rising gold can be risk-off OR inflation hedge
    gold_chg = snapshot.get("commodities", {}).get("Gold", {}).get("change_1d_pct", 0)
    if gold_chg > 0.5:
        scores["risk_off"] += 0.5
        scores["signals"].append(f"Gold bid +{gold_chg:.2f}%")

    # Copper/Gold ratio — rising = growth, falling = contraction
    copper = snapshot.get("commodities", {}).get("Copper", {}).get("price")
    gold = snapshot.get("commodities", {}).get("Gold", {}).get("price")
    if copper and gold and gold > 0:
        cu_au_ratio = copper / gold
        scores["signals"].append(f"Copper/Gold ratio: {cu_au_ratio:.4f}")

    # Credit
    hyg_chg = snapshot.get("credit", {}).get("HYG", {}).get("change_1d_pct", 0)
    if hyg_chg < -0.3:
        scores["risk_off"] += 1
        scores["signals"].append(f"Credit widening (HYG {hyg_chg:.2f}%)")
    elif hyg_chg > 0.3:
        scores["risk_on"] += 1
        scores["signals"].append(f"Credit tightening (HYG +{hyg_chg:.2f}%)")

    # 10Y yield spike
    y10_chg = snapshot.get("treasury_rates", {}).get("US10Y", {}).get("change_1d_pct", 0)
    if y10_chg > 3:
        scores["risk_off"] += 1
        scores["signals"].append(f"10Y yield spiking +{y10_chg:.1f}%")

    # ── DEBASEMENT TRADE DETECTION ──
    # Thesis: fiscal deficits + monetary easing + Treasury buybacks = gold/hard assets bid
    # Gold rising + dollar falling + yields falling = debasement trade ACTIVE
    # Gold rising + dollar rising = flight to safety (different signal)
    gold_5d = snapshot.get("commodities", {}).get("Gold", {}).get("change_5d_pct", 0)
    dxy_5d = snapshot.get("currencies", {}).get("DXY", {}).get("change_5d_pct", 0)
    silver_chg = snapshot.get("commodities", {}).get("Silver", {}).get("change_1d_pct", 0)

    debasement_active = False
    if gold_chg > 0.5 and dxy_chg < 0:
        debasement_active = True
        scores["signals"].append("DEBASEMENT TRADE ACTIVE: gold up + dollar down")
    if gold_5d > 2 and dxy_5d < -0.5:
        debasement_active = True
        scores["signals"].append(f"DEBASEMENT TREND: gold +{gold_5d:.1f}% / DXY {dxy_5d:.1f}% over 5d")
    if gold_chg > 0.5 and silver_chg > 0.5:
        scores["signals"].append(f"Precious metals bid: gold +{gold_chg:.1f}%, silver +{silver_chg:.1f}%")

    # Yield curve + gold = fiscal stress signal
    yc = snapshot.get("yield_curve", {})
    if yc.get("2s10s_inverted") and gold_chg > 0:
        scores["risk_off"] += 1
        scores["signals"].append("Inverted curve + gold bid = fiscal stress")

    # Sector rotation context for trade selection
    tech_chg = snapshot.get("sectors", {}).get("XLK", {}).get("change_1d_pct", 0)
    if tech_chg < -0.5:
        scores["signals"].append(f"Tech sector weak ({tech_chg:.2f}%) — avoid tech calls")

    # Classify
    on = scores["risk_on"]
    off = scores["risk_off"]
    total = on + off if (on + off) > 0 else 1

    if on / total >= 0.65:
        regime = "RISK_ON"
        confidence = on / total
    elif off / total >= 0.65:
        regime = "RISK_OFF"
        confidence = off / total
    else:
        regime = "TRANSITION"
        confidence = max(on, off) / total

    return {
        "regime": regime,
        "confidence": round(confidence, 3),
        "risk_on_score": on,
        "risk_off_score": off,
        "debasement_trade_active": debasement_active,
        "signals": scores["signals"],
        "guidance": {
            "RISK_ON":     "Favor long calls, momentum plays, cyclicals. Full position sizing.",
            "RISK_OFF":    "Favor puts, hedges, defensives. Reduce sizing or skip.",
            "TRANSITION":  "Mixed signals. Reduce position size by 50% or skip. Wait for clarity.",
        }.get(regime, ""),
        "debasement_guidance": (
            "Debasement trade active — gold/silver/hard assets favored over dollar-denominated. "
            "Consider: GLD/SLV calls, mining ETFs (GDX/GDXJ), commodity plays. "
            "Avoid: long-duration bonds, pure USD plays."
        ) if debasement_active else "Debasement trade not active.",
    }


def collect_daily_snapshot() -> dict:
    """Collect full macro snapshot."""
    print("📊 Collecting macro snapshot...")

    snapshot = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "collected_at": datetime.now().isoformat(),
    }

    # Fetch all groups
    print("  [1/8] Treasury rates & bonds...")
    snapshot["treasury_rates"] = fetch_prices(TREASURY_TICKERS)

    print("  [2/8] Commodities (gold, silver, oil, copper)...")
    snapshot["commodities"] = fetch_prices(COMMODITY_TICKERS)

    print("  [3/8] Equity indices...")
    snapshot["equity_indices"] = fetch_prices(EQUITY_INDEX_TICKERS)

    print("  [4/8] Sector ETFs...")
    snapshot["sectors"] = fetch_prices(SECTOR_TICKERS)

    print("  [5/8] Volatility (VIX, term structure)...")
    snapshot["volatility"] = fetch_prices(VOLATILITY_TICKERS)

    print("  [6/8] Credit (HYG, LQD)...")
    snapshot["credit"] = fetch_prices(CREDIT_TICKERS)

    print("  [7/8] Currencies & crypto...")
    snapshot["currencies"] = fetch_prices(CURRENCY_TICKERS)
    snapshot["crypto"] = fetch_prices(SPECIAL_TICKERS)

    print("  [8/8] Debasement trade assets (Newsletter thesis)...")
    snapshot["debasement_assets"] = fetch_prices(DEBASEMENT_TICKERS)

    # Derived metrics
    print("  Computing derived metrics...")
    snapshot["yield_curve"] = compute_yield_curve(snapshot["treasury_rates"])
    snapshot["credit_spread"] = compute_credit_spread(snapshot["credit"])
    snapshot["vol_term_structure"] = compute_vol_term_structure(snapshot["volatility"])

    # Regime classification
    print("  Classifying macro regime...")
    snapshot["regime"] = classify_regime(snapshot)

    # Sector rotation analysis
    sector_changes = {}
    for name, data in snapshot.get("sectors", {}).items():
        if isinstance(data, dict) and "change_1d_pct" in data:
            sector_changes[name] = data["change_1d_pct"]

    if sector_changes:
        sorted_sectors = sorted(sector_changes.items(), key=lambda x: x[1], reverse=True)
        snapshot["sector_rotation"] = {
            "leaders": [{"sector": s, "change": c} for s, c in sorted_sectors[:3]],
            "laggards": [{"sector": s, "change": c} for s, c in sorted_sectors[-3:]],
            "breadth": sum(1 for c in sector_changes.values() if c > 0),
            "total_sectors": len(sector_changes),
        }

    return snapshot


def save_snapshot(snapshot: dict) -> str:
    """Save snapshot to dated JSON file."""
    date_str = snapshot["date"].replace("-", "")
    filepath = MACRO_DIR / f"daily_{date_str}.json"

    with open(filepath, "w") as f:
        json.dump(snapshot, f, indent=2)

    # Also save latest for easy access
    latest_path = MACRO_DIR / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    return str(filepath)


def show_history(days: int = 5):
    """Show trend over last N days."""
    files = sorted(MACRO_DIR.glob("daily_*.json"), reverse=True)[:days]

    if not files:
        print("No historical macro data found.")
        return

    print(f"\n{'Date':<12} {'SPY':>8} {'VIX':>8} {'10Y':>8} {'Gold':>8} {'DXY':>8} {'Regime':<12}")
    print("-" * 70)

    for f in reversed(files):
        with open(f) as fh:
            d = json.load(fh)

        spy = d.get("equity_indices", {}).get("SPY", {}).get("price", "—")
        vix = d.get("volatility", {}).get("VIX", {}).get("price", "—")
        y10 = d.get("treasury_rates", {}).get("US10Y", {}).get("price", "—")
        gold = d.get("commodities", {}).get("Gold", {}).get("price", "—")
        dxy = d.get("currencies", {}).get("DXY", {}).get("price", "—")
        regime = d.get("regime", {}).get("regime", "—")

        spy_s = f"{spy:>8.2f}" if isinstance(spy, (int, float)) else f"{spy:>8}"
        vix_s = f"{vix:>8.2f}" if isinstance(vix, (int, float)) else f"{vix:>8}"
        y10_s = f"{y10:>8.3f}" if isinstance(y10, (int, float)) else f"{y10:>8}"
        gold_s = f"{gold:>8.2f}" if isinstance(gold, (int, float)) else f"{gold:>8}"
        dxy_s = f"{dxy:>8.2f}" if isinstance(dxy, (int, float)) else f"{dxy:>8}"

        print(f"{d['date']:<12} {spy_s} {vix_s} {y10_s} {gold_s} {dxy_s} {regime:<12}")


def print_regime_summary(snapshot: dict):
    """Print human-readable regime summary."""
    r = snapshot.get("regime", {})
    print(f"\n{'='*60}")
    print(f"  MACRO REGIME: {r.get('regime', '?')} (confidence: {r.get('confidence', 0):.0%})")
    print(f"  Risk-On: {r.get('risk_on_score', 0)} | Risk-Off: {r.get('risk_off_score', 0)}")
    print(f"{'='*60}")
    print(f"  Guidance: {r.get('guidance', '')}")
    print(f"\n  Signals:")
    for s in r.get("signals", []):
        print(f"    • {s}")

    # Sector rotation
    sr = snapshot.get("sector_rotation", {})
    if sr:
        print(f"\n  Sector Breadth: {sr.get('breadth', 0)}/{sr.get('total_sectors', 0)} green")
        leaders = [f"{l['sector']} +{l['change']:.2f}%" for l in sr.get("leaders", [])]
        laggards = [f"{l['sector']} {l['change']:.2f}%" for l in sr.get("laggards", [])]
        print(f"  Leaders:  {', '.join(leaders)}")
        print(f"  Laggards: {', '.join(laggards)}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Daily Macro Intelligence Collector")
    parser.add_argument("--regime", action="store_true", help="Print regime classification")
    parser.add_argument("--history", type=int, default=0, help="Show N-day history")
    args = parser.parse_args()

    if args.history > 0:
        show_history(args.history)
    else:
        snapshot = collect_daily_snapshot()
        filepath = save_snapshot(snapshot)
        print(f"\n✅ Snapshot saved: {filepath}")

        if args.regime or True:  # Always show regime
            print_regime_summary(snapshot)
