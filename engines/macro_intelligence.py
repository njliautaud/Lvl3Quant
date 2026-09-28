#!/usr/bin/env python3
"""
Macro Intelligence Engine (HC #806)
====================================
Collects comprehensive daily macro data, analyzes inter-market relationships,
and classifies the current macro regime for trade decision support.

Covers: Yields (2Y/5Y/10Y/30Y), Dollar (DXY), Commodities (Gold/Silver/Oil/Copper),
Equity indices + all 11 sectors, Volatility (VIX/VIX term structure),
Credit (HYG/LQD), Sentiment (put/call ratio approximation).

Inter-market relationships tracked:
  - Yields <-> equities (rising yields = pressure on growth, support banks)
  - Dollar <-> commodities (strong dollar = typically weak gold/oil)
  - Gold <-> fiscal deficits (expanding fiscal = gold bid, monetary debasement)
  - Credit spreads <-> equity risk (widening = risk-off ahead)
  - Yield curve <-> recession/expansion
  - Copper/gold ratio <-> economic health
  - VIX term structure <-> fear regime

Output: /home/jupiter/Lvl3Quant/data/macro/daily_YYYYMMDD.json
        /home/jupiter/Lvl3Quant/data/macro/regime_current.json
        /home/jupiter/Lvl3Quant/data/macro/relationships.json

Usage:
  python3 macro_intelligence.py              # Run daily collection + analysis
  python3 macro_intelligence.py --regime     # Just print current regime
  python3 macro_intelligence.py --history    # Analyze trend from historical snapshots
"""

import json
import os
import sys
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("macro_intelligence")

BASE = Path("/home/jupiter/Lvl3Quant")
MACRO_DIR = BASE / "data" / "macro"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

# ── Asset Universe ──────────────────────────────────────────────────────────
YIELDS = {
    "^IRX": "us_3m_yield",     # 13-week T-bill
    "^FVX": "us_5y_yield",     # 5-year Treasury
    "^TNX": "us_10y_yield",    # 10-year Treasury
    "^TYX": "us_30y_yield",    # 30-year Treasury
}

# 2Y yield isn't on Yahoo directly; we'll use SHY ETF as proxy + compute from curve
YIELD_ETFS = {
    "SHY": "short_treasury_etf",   # 1-3Y Treasury (2Y proxy)
    "IEF": "mid_treasury_etf",     # 7-10Y Treasury
    "TLT": "long_treasury_etf",    # 20+Y Treasury
    "TIP": "tips_etf",             # TIPS (inflation expectations)
}

DOLLAR = {
    "DX-Y.NYB": "dxy_index",      # US Dollar Index
    "UUP": "dollar_bull_etf",      # Dollar bull ETF (backup)
}

COMMODITIES = {
    "GC=F": "gold_futures",
    "SI=F": "silver_futures",
    "CL=F": "crude_oil_futures",
    "HG=F": "copper_futures",
    "GLD": "gold_etf",
    "SLV": "silver_etf",
    "USO": "oil_etf",
    "DBA": "agriculture_etf",
}

EQUITY_INDICES = {
    "SPY": "sp500",
    "QQQ": "nasdaq100",
    "IWM": "russell2000",
    "DIA": "dow30",
    "MDY": "midcap400",
}

SECTORS = {
    "XLF": "financials",
    "XLE": "energy",
    "XLU": "utilities",
    "XLP": "consumer_staples",
    "XLK": "technology",
    "XLC": "communication",
    "XLI": "industrials",
    "XLB": "materials",
    "XLV": "healthcare",
    "XLRE": "real_estate",
    "XLY": "consumer_discretionary",
}

VOLATILITY = {
    "^VIX": "vix",
    "VIXY": "vix_short_term_etf",
    "VIXM": "vix_mid_term_etf",   # For term structure
}

CREDIT = {
    "HYG": "high_yield_bond_etf",
    "LQD": "inv_grade_bond_etf",
    "JNK": "junk_bond_etf",
}

INDUSTRIES = {
    "XME": "metals_mining",
    "XOP": "oil_gas_exploration",
    "XHB": "homebuilders",
    "XBI": "biotech",
    "SMH": "semiconductors",
    "ITB": "construction",
    "KRE": "regional_banks",
    "XRT": "retail",
    "JETS": "airlines",
    "HACK": "cybersecurity",
    "ARKK": "innovation_growth",
    "XSD": "semiconductor_equipment",  # HC #776 — sub-sector depth
    "CAT": "heavy_equipment_proxy",    # HC #775 — equipment/industrial
    "DE": "ag_equipment_proxy",
    "COPX": "copper_miners",           # HC #775 — mining
    "GDXJ": "junior_gold_miners",
    "GDX": "gold_miners",
    "SIL": "silver_miners",
}

METALS_EXTENDED = {
    "PPLT": "platinum_etf",
    "PALL": "palladium_etf",
    "CPER": "copper_etf",
    "URA": "uranium_etf",
}


def fetch_quotes(symbols: Dict[str, str], period: str = "5d") -> Dict[str, Dict]:
    """Fetch price data for a dict of {ticker: label}. Returns {label: {price, change_1d, change_5d, ...}}."""
    if not YF_AVAILABLE:
        log.error("yfinance not available")
        return {}

    results = {}
    all_tickers = list(symbols.keys())

    try:
        data = yf.download(all_tickers, period=period, progress=False, threads=True)
        if data.empty:
            log.warning("yfinance returned empty data")
            return {}

        for ticker, label in symbols.items():
            try:
                if len(all_tickers) > 1:
                    close = data["Close"][ticker].dropna()
                else:
                    close = data["Close"].dropna()

                if len(close) < 1:
                    continue

                current = float(close.iloc[-1])
                prev = float(close.iloc[-2]) if len(close) >= 2 else current
                first = float(close.iloc[0]) if len(close) >= 1 else current

                results[label] = {
                    "ticker": ticker,
                    "price": round(current, 4),
                    "change_1d_pct": round((current / prev - 1) * 100, 3) if prev else 0,
                    "change_5d_pct": round((current / first - 1) * 100, 3) if first else 0,
                    "prev_close": round(prev, 4),
                }
            except Exception as e:
                log.warning(f"Error processing {ticker}: {e}")
                continue

    except Exception as e:
        log.error(f"yfinance download failed: {e}")

    return results


def compute_yield_curve(yields_data: Dict) -> Dict:
    """Compute yield curve metrics."""
    curve = {}

    y10 = yields_data.get("us_10y_yield", {}).get("price")
    y5 = yields_data.get("us_5y_yield", {}).get("price")
    y30 = yields_data.get("us_30y_yield", {}).get("price")
    y3m = yields_data.get("us_3m_yield", {}).get("price")

    # Yahoo yields are in percentage points (e.g., 4.25 = 4.25%)
    if y10 and y3m:
        curve["spread_10y_3m"] = round(y10 - y3m, 3)
        curve["inverted_10y_3m"] = y10 < y3m
    if y10 and y5:
        curve["spread_10y_5y"] = round(y10 - y5, 3)
    if y30 and y10:
        curve["spread_30y_10y"] = round(y30 - y10, 3)
    if y30 and y3m:
        curve["spread_30y_3m"] = round(y30 - y3m, 3)

    # Curve shape classification
    if curve.get("inverted_10y_3m"):
        curve["shape"] = "inverted"
        curve["signal"] = "recession_warning"
    elif curve.get("spread_10y_3m", 0) < 0.25:
        curve["shape"] = "flat"
        curve["signal"] = "late_cycle"
    elif curve.get("spread_10y_3m", 0) > 1.5:
        curve["shape"] = "steep"
        curve["signal"] = "early_recovery"
    else:
        curve["shape"] = "normal"
        curve["signal"] = "expansion"

    return curve


def compute_credit_spreads(credit_data: Dict) -> Dict:
    """Compute credit spread indicators."""
    spreads = {}
    hyg = credit_data.get("high_yield_bond_etf", {}).get("price")
    lqd = credit_data.get("inv_grade_bond_etf", {}).get("price")

    if hyg and lqd:
        # HYG/LQD ratio — declining = widening spreads = risk-off
        ratio = hyg / lqd
        spreads["hyg_lqd_ratio"] = round(ratio, 4)
        hyg_chg = credit_data.get("high_yield_bond_etf", {}).get("change_1d_pct", 0)
        lqd_chg = credit_data.get("inv_grade_bond_etf", {}).get("change_1d_pct", 0)
        spreads["hyg_vs_lqd_1d"] = round(hyg_chg - lqd_chg, 3)

        if spreads["hyg_vs_lqd_1d"] < -0.3:
            spreads["credit_signal"] = "risk_off_widening"
        elif spreads["hyg_vs_lqd_1d"] > 0.3:
            spreads["credit_signal"] = "risk_on_tightening"
        else:
            spreads["credit_signal"] = "neutral"

    return spreads


def compute_intermarket_relationships(data: Dict) -> Dict:
    """
    Analyze inter-market relationships (HC #806 R5).

    Key relationships:
    - Yields up + equities down = risk-off rotation
    - Dollar up + gold down = typical (inverse), break = something changing
    - Gold up + yields up = inflation fear (not just risk-off)
    - Copper/gold ratio rising = economic optimism
    - VIX term structure backwardation = acute fear
    """
    rels = {}
    yields = data.get("yields", {})
    equities = data.get("equity_indices", {})
    commodities = data.get("commodities", {})
    dollar = data.get("dollar", {})
    vol = data.get("volatility", {})
    credit = data.get("credit_spreads", {})

    # 1. Yields vs Equities
    y10_chg = yields.get("us_10y_yield", {}).get("change_1d_pct", 0)
    spy_chg = equities.get("sp500", {}).get("change_1d_pct", 0)
    if y10_chg > 0.5 and spy_chg < -0.3:
        rels["yields_equities"] = "risk_off_rotation"
    elif y10_chg < -0.5 and spy_chg > 0.3:
        rels["yields_equities"] = "risk_on_rally"
    elif y10_chg > 0.5 and spy_chg > 0.3:
        rels["yields_equities"] = "growth_optimism"
    elif y10_chg < -0.5 and spy_chg < -0.3:
        rels["yields_equities"] = "deflation_scare"
    else:
        rels["yields_equities"] = "neutral"

    # 2. Dollar vs Gold (typically inverse)
    dxy_chg = dollar.get("dxy_index", {}).get("change_1d_pct",
              dollar.get("dollar_bull_etf", {}).get("change_1d_pct", 0))
    gold_chg = commodities.get("gold_futures", {}).get("change_1d_pct",
               commodities.get("gold_etf", {}).get("change_1d_pct", 0))
    if dxy_chg > 0.3 and gold_chg > 0.3:
        rels["dollar_gold"] = "both_up_unusual_fear_or_inflation"
    elif dxy_chg < -0.3 and gold_chg < -0.3:
        rels["dollar_gold"] = "both_down_unusual"
    elif dxy_chg > 0.3 and gold_chg < -0.3:
        rels["dollar_gold"] = "normal_inverse_dollar_strength"
    elif dxy_chg < -0.3 and gold_chg > 0.3:
        rels["dollar_gold"] = "normal_inverse_dollar_weakness"
    else:
        rels["dollar_gold"] = "neutral"

    # 3. Gold vs Fiscal/Monetary (the newsletter's thesis)
    # Gold rising + yields rising = inflation expectations, fiscal debasement concern
    gold_price = commodities.get("gold_futures", {}).get("price", 0)
    if gold_chg > 0.5 and y10_chg > 0.2:
        rels["gold_fiscal"] = "inflation_debasement_concern"
    elif gold_chg > 0.5 and y10_chg < -0.2:
        rels["gold_fiscal"] = "flight_to_safety"
    elif gold_chg < -0.5:
        rels["gold_fiscal"] = "confidence_in_fiat"
    else:
        rels["gold_fiscal"] = "neutral"

    # 4. Copper/Gold ratio (economic health barometer)
    copper = commodities.get("copper_futures", {}).get("price", 0)
    gold = commodities.get("gold_futures", {}).get("price", 0)
    if copper and gold and gold > 0:
        ratio = copper / gold
        rels["copper_gold_ratio"] = round(ratio, 6)
        copper_chg = commodities.get("copper_futures", {}).get("change_1d_pct", 0)
        if copper_chg - gold_chg > 0.5:
            rels["copper_gold_signal"] = "economic_optimism"
        elif gold_chg - copper_chg > 0.5:
            rels["copper_gold_signal"] = "economic_pessimism"
        else:
            rels["copper_gold_signal"] = "neutral"

    # 5. VIX term structure
    # Note: VIXY/VIXM ETF ratio is a proxy, not true VIX/VIX3M.
    # "acute_fear" requires BOTH inverted term structure AND elevated VIX (>=18).
    # VIX < 18 = calm market regardless of ETF ratio quirks.
    vix_price = vol.get("vix", {}).get("price", 0)
    vix_st = vol.get("vix_short_term_etf", {}).get("price", 0)
    vix_mt = vol.get("vix_mid_term_etf", {}).get("price", 0)
    if vix_st and vix_mt and vix_mt > 0:
        term_ratio = vix_st / vix_mt
        rels["vix_term_ratio"] = round(term_ratio, 4)
        if term_ratio > 1.05 and vix_price >= 18:
            rels["vix_term_structure"] = "backwardation_acute_fear"
        elif term_ratio > 1.05 and vix_price < 18:
            # Mild inversion but VIX is low — not real fear
            rels["vix_term_structure"] = "backwardation_mild"
        elif term_ratio < 0.90:
            rels["vix_term_structure"] = "contango_complacency"
        else:
            rels["vix_term_structure"] = "normal"

    # 6. Growth vs Value (QQQ vs DIA or IWM)
    qqq_chg = equities.get("nasdaq100", {}).get("change_1d_pct", 0)
    dia_chg = equities.get("dow30", {}).get("change_1d_pct", 0)
    iwm_chg = equities.get("russell2000", {}).get("change_1d_pct", 0)
    if qqq_chg - dia_chg > 0.5:
        rels["growth_value"] = "growth_leading"
    elif dia_chg - qqq_chg > 0.5:
        rels["growth_value"] = "value_leading"
    else:
        rels["growth_value"] = "neutral"

    # 7. Small cap vs large cap (breadth)
    if iwm_chg - spy_chg > 0.5:
        rels["breadth"] = "broadening_bullish"
    elif spy_chg - iwm_chg > 0.5:
        rels["breadth"] = "narrowing_fragile"
    else:
        rels["breadth"] = "neutral"

    # 8. CTA flow proxy (Newsletter framework — HC #806)
    # CTAs are trend followers: when SPY is above 20d MA and momentum positive,
    # CTAs are forced buyers. When below, forced sellers. Light positioning +
    # forced CTA buying = lean long (per the newsletter's BAML survey + CTA flows thesis).
    spy_price = equities.get("sp500", {}).get("price", 0)
    spy_20d_ma = equities.get("sp500", {}).get("ma_20d", 0)
    if spy_price and spy_20d_ma and spy_20d_ma > 0:
        spy_vs_ma = (spy_price - spy_20d_ma) / spy_20d_ma * 100
        rels["spy_vs_20d_ma_pct"] = round(spy_vs_ma, 2)
        if spy_vs_ma > 2.0:
            rels["cta_flow_proxy"] = "strong_buy_pressure"
        elif spy_vs_ma > 0.5:
            rels["cta_flow_proxy"] = "mild_buy_pressure"
        elif spy_vs_ma < -2.0:
            rels["cta_flow_proxy"] = "strong_sell_pressure"
        elif spy_vs_ma < -0.5:
            rels["cta_flow_proxy"] = "mild_sell_pressure"
        else:
            rels["cta_flow_proxy"] = "neutral_near_ma"

    # 9. Gold miners leverage (GDX/GLD ratio — the newsletter's metals conviction indicator)
    # When miners outperform gold, it signals confidence in the mining sector
    # and risk appetite within the precious metals complex
    gdx = commodities.get("gold_miners", {})
    gld = commodities.get("gold_etf", {})
    gdx_chg = gdx.get("change_1d_pct", 0) if gdx else 0
    gld_chg = gld.get("change_1d_pct", 0) if gld else 0
    if gdx_chg and gld_chg:
        miner_leverage = gdx_chg - gld_chg
        if miner_leverage > 0.5:
            rels["gold_miner_leverage"] = "miners_outperforming_bullish"
        elif miner_leverage < -0.5:
            rels["gold_miner_leverage"] = "miners_lagging_caution"
        else:
            rels["gold_miner_leverage"] = "neutral"

    return rels


def classify_regime(data: Dict) -> Dict:
    """
    Classify the current macro regime based on all collected data.

    Regimes:
    - RISK_ON: falling yields, falling VIX, rising equities, weak dollar, strong commodities
    - RISK_OFF: rising yields, rising VIX, falling equities, strong dollar, gold bid
    - INFLATIONARY: rising yields, rising commodities, rising gold, falling bonds
    - DEFLATIONARY: falling yields, falling commodities, falling equities
    - TRANSITION: mixed signals — reduce sizing
    """
    scores = {"risk_on": 0, "risk_off": 0, "inflationary": 0, "deflationary": 0}

    yields = data.get("yields", {})
    equities = data.get("equity_indices", {})
    commodities = data.get("commodities", {})
    vol = data.get("volatility", {})
    relationships = data.get("relationships", {})
    credit = data.get("credit_spreads", {})

    # VIX level
    vix = vol.get("vix", {}).get("price", 20)
    if vix < 15:
        scores["risk_on"] += 2
    elif vix > 25:
        scores["risk_off"] += 2
    elif vix > 20:
        scores["risk_off"] += 1

    # VIX direction
    vix_chg = vol.get("vix", {}).get("change_1d_pct", 0)
    if vix_chg < -3:
        scores["risk_on"] += 1
    elif vix_chg > 5:
        scores["risk_off"] += 2

    # SPY direction
    spy_chg = equities.get("sp500", {}).get("change_1d_pct", 0)
    if spy_chg > 0.5:
        scores["risk_on"] += 2
    elif spy_chg < -0.5:
        scores["risk_off"] += 2

    # Yield direction
    y10_chg = yields.get("us_10y_yield", {}).get("change_1d_pct", 0)
    if y10_chg > 2:  # yields rising fast
        scores["inflationary"] += 2
        scores["risk_off"] += 1
    elif y10_chg < -2:
        scores["deflationary"] += 1
        scores["risk_on"] += 1

    # Gold direction
    gold_chg = commodities.get("gold_futures", {}).get("change_1d_pct",
               commodities.get("gold_etf", {}).get("change_1d_pct", 0))
    if gold_chg > 1:
        scores["risk_off"] += 1
        scores["inflationary"] += 1
    elif gold_chg < -1:
        scores["risk_on"] += 1

    # Oil direction
    oil_chg = commodities.get("crude_oil_futures", {}).get("change_1d_pct", 0)
    if oil_chg > 2:
        scores["inflationary"] += 1
    elif oil_chg < -2:
        scores["deflationary"] += 1

    # Credit spreads
    credit_sig = credit.get("credit_signal", "neutral")
    if credit_sig == "risk_off_widening":
        scores["risk_off"] += 2
    elif credit_sig == "risk_on_tightening":
        scores["risk_on"] += 1

    # Dollar
    dxy_chg = data.get("dollar", {}).get("dxy_index", {}).get("change_1d_pct", 0)
    if dxy_chg > 0.5:
        scores["risk_off"] += 1
    elif dxy_chg < -0.5:
        scores["risk_on"] += 1

    # Breadth
    breadth = relationships.get("breadth", "neutral")
    if breadth == "broadening_bullish":
        scores["risk_on"] += 1
    elif breadth == "narrowing_fragile":
        scores["risk_off"] += 1

    # Determine regime
    max_score = max(scores.values())
    if max_score <= 2:
        regime = "TRANSITION"
        confidence = "low"
        action = "reduce_sizing_or_skip"
    else:
        regime = max(scores, key=scores.get).upper()
        total = sum(scores.values())
        conf_pct = max_score / total if total > 0 else 0
        if conf_pct > 0.6:
            confidence = "high"
        elif conf_pct > 0.4:
            confidence = "medium"
        else:
            confidence = "low"
            regime = "TRANSITION"

        action_map = {
            "RISK_ON": "favor_calls_growth_sectors",
            "RISK_OFF": "favor_puts_defensives_gold",
            "INFLATIONARY": "favor_commodities_energy_materials_tips",
            "DEFLATIONARY": "favor_bonds_utilities_quality",
            "TRANSITION": "reduce_sizing_or_skip",
        }
        action = action_map.get(regime, "reduce_sizing_or_skip")

    return {
        "regime": regime,
        "confidence": confidence,
        "scores": scores,
        "action": action,
        "vix_level": vix,
        "timestamp": datetime.now().isoformat(),
    }


def compute_sector_heatmap(sectors_data: Dict) -> Dict:
    """Rank sectors by 1-day and 5-day performance."""
    heatmap = {}
    for label, d in sectors_data.items():
        heatmap[label] = {
            "ticker": d.get("ticker", ""),
            "change_1d": d.get("change_1d_pct", 0),
            "change_5d": d.get("change_5d_pct", 0),
        }

    # Sort by 1d performance
    ranked = sorted(heatmap.items(), key=lambda x: x[1]["change_1d"], reverse=True)
    for i, (label, d) in enumerate(ranked):
        d["rank_1d"] = i + 1

    # Leaders and laggards
    if ranked:
        heatmap["_leaders_1d"] = [r[0] for r in ranked[:3]]
        heatmap["_laggards_1d"] = [r[0] for r in ranked[-3:]]

    return heatmap


def run_daily_collection() -> Dict:
    """Run the full daily macro data collection."""
    log.info("Starting daily macro intelligence collection...")
    today = date.today().isoformat()

    data = {
        "date": today,
        "collected_at": datetime.now().isoformat(),
    }

    # Fetch all data in batches
    log.info("Fetching yields...")
    data["yields"] = fetch_quotes(YIELDS, period="5d")

    log.info("Fetching yield ETFs...")
    data["yield_etfs"] = fetch_quotes(YIELD_ETFS, period="5d")

    log.info("Fetching dollar...")
    data["dollar"] = fetch_quotes(DOLLAR, period="5d")

    log.info("Fetching commodities...")
    data["commodities"] = fetch_quotes(COMMODITIES, period="5d")

    log.info("Fetching metals extended...")
    data["metals_extended"] = fetch_quotes(METALS_EXTENDED, period="5d")

    log.info("Fetching equity indices...")
    data["equity_indices"] = fetch_quotes(EQUITY_INDICES, period="5d")

    log.info("Fetching sectors...")
    data["sectors"] = fetch_quotes(SECTORS, period="5d")

    log.info("Fetching industries...")
    data["industries"] = fetch_quotes(INDUSTRIES, period="5d")

    log.info("Fetching volatility...")
    data["volatility"] = fetch_quotes(VOLATILITY, period="5d")

    log.info("Fetching credit...")
    data["credit"] = fetch_quotes(CREDIT, period="5d")

    # Computed analytics
    log.info("Computing yield curve...")
    data["yield_curve"] = compute_yield_curve(data["yields"])

    log.info("Computing credit spreads...")
    data["credit_spreads"] = compute_credit_spreads(data["credit"])

    # Compute SPY 20d MA for CTA flow proxy (Newsletter framework)
    log.info("Computing SPY 20d MA for CTA flow proxy...")
    try:
        spy_hist = yf.download("SPY", period="2mo", progress=False)
        if not spy_hist.empty and len(spy_hist) >= 20:
            spy_20d = float(spy_hist["Close"].iloc[-20:].mean())
            if "sp500" in data.get("equity_indices", {}):
                data["equity_indices"]["sp500"]["ma_20d"] = round(spy_20d, 4)
    except Exception as e:
        log.warning(f"SPY 20d MA computation failed: {e}")

    log.info("Computing inter-market relationships...")
    data["relationships"] = compute_intermarket_relationships(data)

    log.info("Computing sector heatmap...")
    data["sector_heatmap"] = compute_sector_heatmap(data["sectors"])

    log.info("Classifying macro regime...")
    data["regime"] = classify_regime(data)

    return data


def save_daily(data: Dict):
    """Save daily snapshot and regime files."""
    today = date.today().strftime("%Y%m%d")

    # Daily snapshot
    daily_path = MACRO_DIR / f"daily_{today}.json"
    with open(daily_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Saved daily snapshot: {daily_path}")

    # Current regime (always-current pointer)
    regime_path = MACRO_DIR / "regime_current.json"
    with open(regime_path, "w") as f:
        json.dump(data["regime"], f, indent=2, default=str)
    log.info(f"Updated current regime: {regime_path}")

    # Relationships
    rels_path = MACRO_DIR / "relationships.json"
    with open(rels_path, "w") as f:
        json.dump(data["relationships"], f, indent=2, default=str)
    log.info(f"Updated relationships: {rels_path}")

    # Summary for signal aggregator consumption
    summary = {
        "date": data["date"],
        "regime": data["regime"]["regime"],
        "regime_confidence": data["regime"]["confidence"],
        "regime_action": data["regime"]["action"],
        "vix": data["volatility"].get("vix", {}).get("price"),
        "spy_change_1d": data["equity_indices"].get("sp500", {}).get("change_1d_pct"),
        "yield_10y": data["yields"].get("us_10y_yield", {}).get("price"),
        "yield_curve_shape": data["yield_curve"].get("shape"),
        "gold_change_1d": data["commodities"].get("gold_futures", {}).get("change_1d_pct",
                          data["commodities"].get("gold_etf", {}).get("change_1d_pct")),
        "dxy_change_1d": data["dollar"].get("dxy_index", {}).get("change_1d_pct"),
        "credit_signal": data["credit_spreads"].get("credit_signal"),
        "sector_leaders": data["sector_heatmap"].get("_leaders_1d", []),
        "sector_laggards": data["sector_heatmap"].get("_laggards_1d", []),
        "key_relationships": {
            k: v for k, v in data["relationships"].items()
            if not k.startswith("_") and v != "neutral" and not isinstance(v, (int, float))
        },
    }
    summary_path = MACRO_DIR / "macro_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Updated macro summary: {summary_path}")


def print_regime_report(data: Dict):
    """Print human-readable regime report."""
    r = data["regime"]
    print(f"\n{'='*60}")
    print(f"  MACRO REGIME: {r['regime']} (confidence: {r['confidence']})")
    print(f"  Action: {r['action']}")
    print(f"  VIX: {r['vix_level']}")
    print(f"{'='*60}")

    print(f"\n  Scores: {r['scores']}")

    # Yield curve
    yc = data.get("yield_curve", {})
    if yc:
        print(f"\n  Yield Curve: {yc.get('shape', '?')} ({yc.get('signal', '?')})")
        if "spread_10y_3m" in yc:
            print(f"    10Y-3M spread: {yc['spread_10y_3m']:.3f}%")

    # Key relationships
    rels = data.get("relationships", {})
    print(f"\n  Inter-Market Signals:")
    for k, v in rels.items():
        if not isinstance(v, (int, float)) and v != "neutral":
            print(f"    {k}: {v}")

    # Sector heatmap
    heatmap = data.get("sector_heatmap", {})
    leaders = heatmap.get("_leaders_1d", [])
    laggards = heatmap.get("_laggards_1d", [])
    if leaders:
        print(f"\n  Sector Leaders (1d): {', '.join(leaders)}")
    if laggards:
        print(f"  Sector Laggards (1d): {', '.join(laggards)}")

    print()


def analyze_history(days: int = 20) -> Dict:
    """Analyze regime trends from historical daily snapshots."""
    files = sorted(MACRO_DIR.glob("daily_*.json"))[-days:]
    if not files:
        log.warning("No historical snapshots found")
        return {}

    history = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
            history.append({
                "date": d.get("date"),
                "regime": d.get("regime", {}).get("regime"),
                "vix": d.get("volatility", {}).get("vix", {}).get("price"),
                "spy_chg": d.get("equity_indices", {}).get("sp500", {}).get("change_1d_pct"),
                "gold_chg": d.get("commodities", {}).get("gold_futures", {}).get("change_1d_pct"),
                "y10": d.get("yields", {}).get("us_10y_yield", {}).get("price"),
            })

    # Regime frequency
    regimes = [h["regime"] for h in history if h["regime"]]
    regime_counts = {}
    for r in regimes:
        regime_counts[r] = regime_counts.get(r, 0) + 1

    return {
        "days_analyzed": len(history),
        "regime_distribution": regime_counts,
        "dominant_regime": max(regime_counts, key=regime_counts.get) if regime_counts else "unknown",
        "history": history,
    }


if __name__ == "__main__":
    if "--regime" in sys.argv:
        regime_path = MACRO_DIR / "regime_current.json"
        if regime_path.exists():
            with open(regime_path) as f:
                print(json.dumps(json.load(f), indent=2))
        else:
            print("No regime data. Run without flags first.")
    elif "--history" in sys.argv:
        result = analyze_history()
        print(json.dumps(result, indent=2, default=str))
    else:
        data = run_daily_collection()
        save_daily(data)
        print_regime_report(data)
        print("Daily macro intelligence collection complete.")
