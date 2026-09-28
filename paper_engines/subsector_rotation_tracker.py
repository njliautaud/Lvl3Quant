#!/usr/bin/env python3
"""
Sub-Sector Rotation Tracker
============================
Deep sub-industry taxonomy covering the full US equity market.
Tracks momentum, relative strength, money flow, and rotation patterns
at the sub-sector/sub-industry level.

Goes DEEPER than broad sector ETFs: semiconductor equipment vs heavy equipment,
biotech vs medtech vs pharma, etc.

Outputs: /home/jupiter/Lvl3Quant/state/subsector_rotation_state.json

Designed to run at 4:15 PM ET weekdays (after market close, before
the signal aggregator at 4:30 PM).

Usage:
    python3 paper_engines/subsector_rotation_tracker.py
"""

import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = BASE_DIR / "state"
STATE_DIR.mkdir(exist_ok=True)
DATA_DIR = BASE_DIR / "data"

OUTPUT_PATH = STATE_DIR / "subsector_rotation_state.json"
CACHE_PATH = DATA_DIR / "shared_market_cache.pkl"
SUBSECTOR_CACHE_PATH = DATA_DIR / "subsector_rotation_cache.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SubSectorRotation] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "subsector_rotation_tracker.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SUB-SECTOR TAXONOMY — Comprehensive US Market Coverage
# Each sub-industry has 3-5 liquid tickers available on Robinhood
# Plus an ETF proxy where applicable
# ---------------------------------------------------------------------------
SUBSECTOR_UNIVERSE = {
    # === INFORMATION TECHNOLOGY ===
    "semiconductors": {
        "tickers": ["NVDA", "AMD", "INTC", "QCOM", "MU"],
        "etf": "SMH",
        "gics_sector": "Information Technology",
        "description": "Chip designers and manufacturers",
    },
    "semicon_equipment": {
        "tickers": ["AMAT", "LRCX", "KLAC", "ASML", "TER"],
        "etf": "SMH",
        "gics_sector": "Information Technology",
        "description": "Semiconductor manufacturing equipment",
    },
    "enterprise_software": {
        "tickers": ["MSFT", "CRM", "ORCL", "NOW", "ADBE"],
        "etf": "IGV",
        "gics_sector": "Information Technology",
        "description": "Enterprise/cloud SaaS platforms",
    },
    "cybersecurity": {
        "tickers": ["PANW", "CRWD", "FTNT", "ZS", "OKTA"],
        "etf": "CIBR",
        "gics_sector": "Information Technology",
        "description": "Cybersecurity software and services",
    },
    "it_hardware": {
        "tickers": ["AAPL", "DELL", "HPQ", "CSCO", "ANET"],
        "etf": "XLK",
        "gics_sector": "Information Technology",
        "description": "Hardware, networking, consumer electronics",
    },

    # === HEALTHCARE ===
    "biotech": {
        "tickers": ["REGN", "GILD", "VRTX", "MRNA", "BIIB"],
        "etf": "IBB",
        "gics_sector": "Health Care",
        "description": "Biotech and genomics",
    },
    "medtech_devices": {
        "tickers": ["ISRG", "ABT", "MDT", "SYK", "EW"],
        "etf": "IHI",
        "gics_sector": "Health Care",
        "description": "Medical devices, surgical robots, diagnostics",
    },
    "pharma_large": {
        "tickers": ["LLY", "JNJ", "ABBV", "MRK", "PFE"],
        "etf": "XLV",
        "gics_sector": "Health Care",
        "description": "Large-cap pharma",
    },
    "health_services": {
        "tickers": ["UNH", "HCA", "CNC", "ELV", "CI"],
        "etf": "XLV",
        "gics_sector": "Health Care",
        "description": "Health insurers and hospital operators",
    },

    # === FINANCIALS ===
    "megabank": {
        "tickers": ["JPM", "BAC", "WFC", "C", "GS"],
        "etf": "XLF",
        "gics_sector": "Financials",
        "description": "Money-center and investment banks",
    },
    "regional_bank": {
        "tickers": ["USB", "PNC", "TFC", "FITB", "KEY"],
        "etf": "KRE",
        "gics_sector": "Financials",
        "description": "Regional banks",
    },
    "insurance": {
        "tickers": ["BRK-B", "PGR", "AIG", "MET", "ALL"],
        "etf": "KIE",
        "gics_sector": "Financials",
        "description": "Property/casualty and life insurance",
    },
    "fintech_payments": {
        "tickers": ["V", "MA", "PYPL", "AFRM", "FIS"],
        "etf": "IPAY",
        "gics_sector": "Financials",
        "description": "Payments, fintech, card networks",
    },

    # === ENERGY ===
    "oil_integrated": {
        "tickers": ["XOM", "CVX", "COP", "EOG", "OXY"],
        "etf": "XLE",
        "gics_sector": "Energy",
        "description": "Integrated oil and E&P",
    },
    "oilfield_services": {
        "tickers": ["SLB", "HAL", "BKR", "FTI", "NOV"],
        "etf": "OIH",
        "gics_sector": "Energy",
        "description": "Oilfield equipment and services",
    },
    "midstream_pipelines": {
        "tickers": ["WMB", "KMI", "OKE", "ET", "MPLX"],
        "etf": "AMLP",
        "gics_sector": "Energy",
        "description": "Pipelines and midstream MLPs",
    },

    # === INDUSTRIALS ===
    "aerospace_defense": {
        "tickers": ["LMT", "RTX", "NOC", "GD", "LHX"],
        "etf": "ITA",
        "gics_sector": "Industrials",
        "description": "Defense contractors and aerospace",
    },
    "heavy_equipment": {
        "tickers": ["CAT", "DE", "CMI", "PCAR", "TTC"],
        "etf": "XLI",
        "gics_sector": "Industrials",
        "description": "Construction, farm, and industrial equipment",
    },
    "industrial_automation": {
        "tickers": ["EMR", "ROK", "ETN", "IR", "AME"],
        "etf": "XLI",
        "gics_sector": "Industrials",
        "description": "Electrical equipment, automation, power management",
    },
    "transportation": {
        "tickers": ["UNP", "UPS", "FDX", "CSX", "DAL"],
        "etf": "IYT",
        "gics_sector": "Industrials",
        "description": "Rail, trucking, airlines, logistics",
    },

    # === CONSUMER DISCRETIONARY ===
    "ecommerce_retail": {
        "tickers": ["AMZN", "HD", "LOW", "TJX", "COST"],
        "etf": "XLY",
        "gics_sector": "Consumer Discretionary",
        "description": "E-commerce and big-box retail",
    },
    "autos_ev": {
        "tickers": ["TSLA", "GM", "F", "RIVN", "ON"],
        "etf": "CARZ",
        "gics_sector": "Consumer Discretionary",
        "description": "Autos, EVs, and auto semiconductors",
    },
    "restaurants_leisure": {
        "tickers": ["MCD", "SBUX", "CMG", "DRI", "YUM"],
        "etf": "PEJ",
        "gics_sector": "Consumer Discretionary",
        "description": "QSR, fast casual, restaurants",
    },

    # === CONSUMER STAPLES ===
    "consumer_staples_food": {
        "tickers": ["PG", "KO", "PEP", "MDLZ", "GIS"],
        "etf": "XLP",
        "gics_sector": "Consumer Staples",
        "description": "Packaged food, beverages, household products",
    },
    "staples_retail": {
        "tickers": ["WMT", "COST", "TGT", "DG", "KR"],
        "etf": "XLP",
        "gics_sector": "Consumer Staples",
        "description": "Grocery and discount retailers",
    },

    # === MATERIALS ===
    "mining_metals": {
        "tickers": ["FCX", "NEM", "GOLD", "SCCO", "CLF"],
        "etf": "XME",
        "gics_sector": "Materials",
        "description": "Gold, copper, steel miners",
    },
    "chemicals": {
        "tickers": ["LIN", "APD", "ECL", "SHW", "DD"],
        "etf": "XLB",
        "gics_sector": "Materials",
        "description": "Specialty chemicals, industrial gases, coatings",
    },

    # === REAL ESTATE ===
    "reits_data_towers": {
        "tickers": ["PLD", "AMT", "EQIX", "DLR", "SPG"],
        "etf": "XLRE",
        "gics_sector": "Real Estate",
        "description": "Data centers, towers, logistics, malls",
    },

    # === UTILITIES ===
    "utilities_electric": {
        "tickers": ["NEE", "DUK", "SO", "AEP", "SRE"],
        "etf": "XLU",
        "gics_sector": "Utilities",
        "description": "Regulated electric utilities and renewables",
    },

    # === COMMUNICATION SERVICES ===
    "big_tech_comm": {
        "tickers": ["META", "GOOGL", "NFLX", "DIS", "CMCSA"],
        "etf": "XLC",
        "gics_sector": "Communication Services",
        "description": "Social media, streaming, search, media",
    },
    "telecom": {
        "tickers": ["T", "VZ", "TMUS", "AMX", "LUMN"],
        "etf": "XLC",
        "gics_sector": "Communication Services",
        "description": "Legacy and wireless telecom",
    },
}

# Collect all unique tickers
ALL_TICKERS = sorted(set(
    t for sub in SUBSECTOR_UNIVERSE.values()
    for t in sub["tickers"]
))

# Add ETFs (unique)
ALL_ETFS = sorted(set(
    sub["etf"] for sub in SUBSECTOR_UNIVERSE.values()
    if sub.get("etf")
))

# Macro tickers for regime detection
MACRO_TICKERS = ["SPY", "^VIX", "TLT", "HYG", "GLD", "^TNX"]

ALL_DOWNLOAD_TICKERS = sorted(set(ALL_TICKERS + ALL_ETFS + MACRO_TICKERS))


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_from_shared_cache() -> Optional[dict]:
    """Try to load from shared market cache first."""
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            ts = cache.get("timestamp", "unknown")
            log.info(f"Loaded shared cache (timestamp: {ts})")
            return cache
        except Exception as e:
            log.warning(f"Could not load shared cache: {e}")
    return None


def download_data(tickers: list, period: str = "2y") -> Optional[pd.DataFrame]:
    """Download OHLCV data for tickers using yfinance."""
    if not HAS_YF:
        log.error("yfinance not installed")
        return None
    try:
        import time
        # Download in batches to avoid rate limits
        batch_size = 40
        all_data = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            log.info(f"  Downloading batch {i // batch_size + 1}: {len(batch)} tickers")
            data = yf.download(batch, period=period, progress=False, threads=False)
            if data is not None and not data.empty:
                all_data.append(data)
            if i + batch_size < len(tickers):
                time.sleep(1)

        if not all_data:
            log.error("All download batches failed")
            return None

        if len(all_data) > 1:
            merged = pd.concat(all_data, axis=1)
            merged = merged.loc[:, ~merged.columns.duplicated()]
        else:
            merged = all_data[0]

        return merged
    except Exception as e:
        log.error(f"Download failed: {e}")
        return None


def get_ticker_close(data, ticker: str) -> Optional[pd.Series]:
    """Extract close series for a ticker from multi-ticker download."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data["Close"].columns:
                return data["Close"][ticker].dropna()
        elif ticker in data.columns:
            return data[ticker].dropna()
    except Exception:
        pass
    return None


def get_ticker_volume(data, ticker: str) -> Optional[pd.Series]:
    """Extract volume series for a ticker."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data["Volume"].columns:
                return data["Volume"][ticker].dropna()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Signal Computation
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_subsector_metrics(
    data, subsector_name: str, subsector_info: dict, spy_close: Optional[pd.Series]
) -> Optional[dict]:
    """Compute comprehensive rotation metrics for one sub-sector."""
    tickers = subsector_info["tickers"]
    close_series = {}
    volume_series = {}

    for t in tickers:
        c = get_ticker_close(data, t)
        v = get_ticker_volume(data, t)
        if c is not None and len(c) >= 60:
            close_series[t] = c
            if v is not None:
                volume_series[t] = v

    if len(close_series) < 2:
        return None

    # Equal-weight sub-sector return series
    # Align all series to common dates
    close_df = pd.DataFrame(close_series).dropna()
    if len(close_df) < 60:
        return None

    returns_df = close_df.pct_change().dropna()
    subsector_ret = returns_df.mean(axis=1)  # equal-weight daily return

    # Cumulative return series
    cum_ret = (1 + subsector_ret).cumprod()

    # --- Momentum metrics ---
    try:
        ret_5d = float(cum_ret.iloc[-1] / cum_ret.iloc[-6] - 1) if len(cum_ret) >= 6 else 0
        ret_10d = float(cum_ret.iloc[-1] / cum_ret.iloc[-11] - 1) if len(cum_ret) >= 11 else 0
        ret_21d = float(cum_ret.iloc[-1] / cum_ret.iloc[-22] - 1) if len(cum_ret) >= 22 else 0
        ret_63d = float(cum_ret.iloc[-1] / cum_ret.iloc[-64] - 1) if len(cum_ret) >= 64 else 0
        ret_126d = float(cum_ret.iloc[-1] / cum_ret.iloc[-127] - 1) if len(cum_ret) >= 127 else 0
        ret_252d = float(cum_ret.iloc[-1] / cum_ret.iloc[-253] - 1) if len(cum_ret) >= 253 else None
    except (IndexError, KeyError):
        return None

    # --- Relative strength vs SPY ---
    rel_str_21d = 0.0
    rel_str_63d = 0.0
    if spy_close is not None and len(spy_close) >= 64:
        spy_ret_21d = float(spy_close.iloc[-1] / spy_close.iloc[-22] - 1) if len(spy_close) >= 22 else 0
        spy_ret_63d = float(spy_close.iloc[-1] / spy_close.iloc[-64] - 1) if len(spy_close) >= 64 else 0
        rel_str_21d = ret_21d - spy_ret_21d
        rel_str_63d = ret_63d - spy_ret_63d

    # --- Relative strength change (rotation velocity) ---
    # How fast is relative strength CHANGING? (key rotation signal)
    if spy_close is not None and len(spy_close) >= 32 and len(cum_ret) >= 32:
        # RS 10 days ago vs now
        try:
            ret_21d_prev = float(cum_ret.iloc[-11] / cum_ret.iloc[-32] - 1)
            spy_ret_21d_prev = float(spy_close.iloc[-11] / spy_close.iloc[-32] - 1)
            rel_str_21d_prev = ret_21d_prev - spy_ret_21d_prev
            rotation_velocity = rel_str_21d - rel_str_21d_prev
        except (IndexError, KeyError):
            rotation_velocity = 0.0
    else:
        rotation_velocity = 0.0

    # --- Volatility ---
    vol_21d = float(subsector_ret.tail(21).std() * np.sqrt(252)) if len(subsector_ret) >= 21 else 0.3

    # --- RSI on sub-sector cumulative returns ---
    rsi_series = compute_rsi(cum_ret, 14)
    rsi_14 = float(rsi_series.iloc[-1]) if not rsi_series.empty and not np.isnan(rsi_series.iloc[-1]) else 50

    # --- Money flow / volume metrics ---
    vol_ratio = 1.0
    money_flow_score = 0.0
    if volume_series:
        vol_df = pd.DataFrame(volume_series).dropna()
        if len(vol_df) >= 21:
            recent_vol = vol_df.tail(5).mean().mean()
            avg_vol = vol_df.tail(21).mean().mean()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

            # Money flow: volume * direction
            # Positive = more volume on up days, negative = more volume on down days
            if len(returns_df) >= 10 and len(vol_df) >= 10:
                aligned = returns_df.tail(10).mean(axis=1)
                aligned_vol = vol_df.tail(10).mean(axis=1)
                # Align indices
                common_idx = aligned.index.intersection(aligned_vol.index)
                if len(common_idx) >= 5:
                    al_ret = aligned.loc[common_idx]
                    al_vol = aligned_vol.loc[common_idx]
                    up_vol = al_vol[al_ret > 0].sum()
                    down_vol = al_vol[al_ret <= 0].sum()
                    total_vol = up_vol + down_vol
                    money_flow_score = float((up_vol - down_vol) / total_vol) if total_vol > 0 else 0

    # --- SMA positioning ---
    sma_20 = float(cum_ret.rolling(20).mean().iloc[-1]) if len(cum_ret) >= 20 else float(cum_ret.iloc[-1])
    sma_50 = float(cum_ret.rolling(50).mean().iloc[-1]) if len(cum_ret) >= 50 else sma_20
    current = float(cum_ret.iloc[-1])
    above_sma20 = current > sma_20
    above_sma50 = current > sma_50

    # --- Breadth: how many tickers are in uptrends ---
    n_above_sma20 = 0
    for t, c in close_series.items():
        if len(c) >= 20:
            if float(c.iloc[-1]) > float(c.rolling(20).mean().iloc[-1]):
                n_above_sma20 += 1
    breadth_pct = n_above_sma20 / len(close_series) if close_series else 0

    # --- Composite rotation score ---
    # Positive = money flowing IN, negative = money flowing OUT
    rotation_score = 0.0
    rotation_score += np.clip(rotation_velocity * 50, -2, 2)   # Rotation velocity (most important)
    rotation_score += np.clip(rel_str_21d * 10, -1.5, 1.5)    # Current relative strength
    rotation_score += np.clip(money_flow_score * 2, -1, 1)     # Volume-weighted direction
    rotation_score += np.clip((vol_ratio - 1) * 0.5, -0.5, 0.5)  # Volume expansion
    rotation_score += (breadth_pct - 0.5) * 1.0                # Breadth

    # Determine rotation phase
    if rotation_score > 1.5:
        rotation_phase = "INFLOW"  # Strong money flowing in
    elif rotation_score > 0.5:
        rotation_phase = "ACCUMULATING"  # Early rotation in
    elif rotation_score < -1.5:
        rotation_phase = "OUTFLOW"  # Strong money flowing out
    elif rotation_score < -0.5:
        rotation_phase = "DISTRIBUTING"  # Early rotation out
    else:
        rotation_phase = "NEUTRAL"

    # Per-ticker detail
    ticker_details = {}
    for t, c in close_series.items():
        price = float(c.iloc[-1])
        r5 = float(c.pct_change(5).iloc[-1]) if len(c) >= 6 else 0
        r21 = float(c.pct_change(21).iloc[-1]) if len(c) >= 22 else 0
        ticker_details[t] = {
            "price": round(price, 2),
            "ret_5d_pct": round(r5 * 100, 2),
            "ret_21d_pct": round(r21 * 100, 2),
        }

    return {
        "subsector": subsector_name,
        "gics_sector": subsector_info["gics_sector"],
        "description": subsector_info["description"],
        "etf_proxy": subsector_info.get("etf"),
        "n_tickers": len(close_series),
        "rotation_phase": rotation_phase,
        "rotation_score": round(rotation_score, 3),
        "metrics": {
            "ret_5d_pct": round(ret_5d * 100, 2),
            "ret_10d_pct": round(ret_10d * 100, 2),
            "ret_21d_pct": round(ret_21d * 100, 2),
            "ret_63d_pct": round(ret_63d * 100, 2),
            "ret_126d_pct": round(ret_126d * 100, 2),
            "ret_252d_pct": round(ret_252d * 100, 2) if ret_252d is not None else None,
            "rel_str_vs_spy_21d": round(rel_str_21d * 100, 2),
            "rel_str_vs_spy_63d": round(rel_str_63d * 100, 2),
            "rotation_velocity": round(rotation_velocity * 100, 3),
            "vol_21d": round(vol_21d, 3),
            "rsi_14": round(rsi_14, 1),
            "vol_ratio_5d_vs_21d": round(vol_ratio, 2),
            "money_flow_score": round(money_flow_score, 3),
            "above_sma20": above_sma20,
            "above_sma50": above_sma50,
            "breadth_pct_above_sma20": round(breadth_pct, 2),
        },
        "tickers": ticker_details,
    }


# ---------------------------------------------------------------------------
# Regime Detection
# ---------------------------------------------------------------------------

def detect_macro_regime(data) -> dict:
    """Detect current macro regime from VIX, yields, etc."""
    regime = {
        "vix_level": None,
        "vix_regime": "NORMAL",
        "yield_curve": "UNKNOWN",
        "risk_appetite": "NEUTRAL",
    }

    vix = get_ticker_close(data, "^VIX")
    if vix is not None and len(vix) >= 5:
        regime["vix_level"] = round(float(vix.iloc[-1]), 2)
        vix_val = regime["vix_level"]
        if vix_val >= 25:
            regime["vix_regime"] = "ELEVATED"
        elif vix_val >= 18:
            regime["vix_regime"] = "MODERATE"
        else:
            regime["vix_regime"] = "LOW"

    # TLT vs HYG as risk appetite proxy
    tlt = get_ticker_close(data, "TLT")
    hyg = get_ticker_close(data, "HYG")
    if tlt is not None and hyg is not None and len(tlt) >= 22 and len(hyg) >= 22:
        tlt_ret = float(tlt.iloc[-1] / tlt.iloc[-22] - 1)
        hyg_ret = float(hyg.iloc[-1] / hyg.iloc[-22] - 1)
        # HYG outperforming TLT = risk-on
        spread = hyg_ret - tlt_ret
        if spread > 0.01:
            regime["risk_appetite"] = "RISK_ON"
        elif spread < -0.01:
            regime["risk_appetite"] = "RISK_OFF"

    # Gold as safe-haven proxy
    gld = get_ticker_close(data, "GLD")
    if gld is not None and len(gld) >= 22:
        gld_ret = float(gld.iloc[-1] / gld.iloc[-22] - 1)
        regime["gold_21d_ret"] = round(gld_ret * 100, 2)

    return regime


# ---------------------------------------------------------------------------
# Cross-Sector Rotation Detection
# ---------------------------------------------------------------------------

def detect_rotation_patterns(subsector_results: list) -> dict:
    """Detect rotation patterns: where is money flowing to/from?"""
    if not subsector_results:
        return {"pattern": "NO_DATA"}

    # Sort by rotation score
    sorted_subs = sorted(subsector_results, key=lambda x: x["rotation_score"], reverse=True)

    inflow = [s for s in sorted_subs if s["rotation_phase"] in ("INFLOW", "ACCUMULATING")]
    outflow = [s for s in sorted_subs if s["rotation_phase"] in ("OUTFLOW", "DISTRIBUTING")]

    # Classify rotation type
    inflow_sectors = set(s["gics_sector"] for s in inflow)
    outflow_sectors = set(s["gics_sector"] for s in outflow)

    pattern = "MIXED"
    rotation_description = ""

    # Defensive rotation: utilities, staples, healthcare IN; tech, discretionary OUT
    defensive = {"Utilities", "Consumer Staples", "Health Care", "Real Estate"}
    cyclical = {"Information Technology", "Consumer Discretionary", "Financials", "Industrials", "Energy", "Materials"}

    def_in = inflow_sectors & defensive
    cyc_out = outflow_sectors & cyclical
    cyc_in = inflow_sectors & cyclical
    def_out = outflow_sectors & defensive

    if len(def_in) >= 2 and len(cyc_out) >= 2:
        pattern = "DEFENSIVE_ROTATION"
        rotation_description = (
            f"Money rotating INTO defensive ({', '.join(def_in)}) "
            f"and OUT OF cyclical ({', '.join(cyc_out)})"
        )
    elif len(cyc_in) >= 2 and len(def_out) >= 1:
        pattern = "RISK_ON_ROTATION"
        rotation_description = (
            f"Money rotating INTO cyclicals ({', '.join(cyc_in)}) "
            f"and OUT OF defensives ({', '.join(def_out)})"
        )
    elif len(inflow) >= 3 and len(outflow) <= 1:
        pattern = "BROAD_BUYING"
        rotation_description = "Broad-based buying across multiple sub-sectors"
    elif len(outflow) >= 3 and len(inflow) <= 1:
        pattern = "BROAD_SELLING"
        rotation_description = "Broad-based selling across multiple sub-sectors"
    else:
        rotation_description = "Mixed rotation, no clear sector-level pattern"

    # Top/bottom sub-sectors
    top_3 = sorted_subs[:3]
    bottom_3 = sorted_subs[-3:]

    return {
        "pattern": pattern,
        "description": rotation_description,
        "inflow_count": len(inflow),
        "outflow_count": len(outflow),
        "top_3_inflow": [
            {"subsector": s["subsector"], "score": s["rotation_score"],
             "ret_21d": s["metrics"]["ret_21d_pct"], "phase": s["rotation_phase"]}
            for s in top_3
        ],
        "bottom_3_outflow": [
            {"subsector": s["subsector"], "score": s["rotation_score"],
             "ret_21d": s["metrics"]["ret_21d_pct"], "phase": s["rotation_phase"]}
            for s in bottom_3
        ],
        "sector_flow_summary": {
            sector: {
                "n_inflow": len([s for s in inflow if s["gics_sector"] == sector]),
                "n_outflow": len([s for s in outflow if s["gics_sector"] == sector]),
                "avg_rotation_score": round(
                    np.mean([s["rotation_score"] for s in sorted_subs if s["gics_sector"] == sector]), 3
                ),
            }
            for sector in sorted(set(s["gics_sector"] for s in sorted_subs))
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tracker():
    """Main tracker loop."""
    log.info("=" * 60)
    log.info(f"Sub-Sector Rotation Tracker — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"Tracking {len(SUBSECTOR_UNIVERSE)} sub-sectors, {len(ALL_TICKERS)} tickers")

    # Try shared cache first, then download
    data = None
    cache = load_from_shared_cache()
    if cache and "Close" in cache:
        # Check if our tickers are in the cache
        cached_tickers = set(cache.get("tickers", []))
        missing = set(ALL_TICKERS) - cached_tickers
        if len(missing) > len(ALL_TICKERS) * 0.3:
            log.info(f"  Cache missing {len(missing)} tickers, downloading fresh")
            data = download_data(ALL_DOWNLOAD_TICKERS)
        else:
            # Build a pseudo-MultiIndex DataFrame from cache
            log.info(f"  Using shared cache ({len(missing)} tickers missing, downloading those)")
            # Download missing tickers only
            if missing:
                extra = download_data(list(missing), period="2y")
            else:
                extra = None

            # For simplicity, just download everything if cache is incomplete
            if missing:
                data = download_data(ALL_DOWNLOAD_TICKERS)
            else:
                # Reconstruct from cache
                close_df = cache["Close"]
                vol_df = cache.get("Volume", pd.DataFrame())
                # Make it look like yfinance multi-ticker output
                arrays = {}
                if not close_df.empty:
                    arrays["Close"] = close_df
                if not vol_df.empty:
                    arrays["Volume"] = vol_df
                if arrays:
                    data = pd.concat(arrays, axis=1)
    else:
        data = download_data(ALL_DOWNLOAD_TICKERS)

    if data is None:
        log.error("No data available. Exiting.")
        return None

    # Get SPY for relative strength
    spy_close = get_ticker_close(data, "SPY")

    # Compute metrics for each sub-sector
    results = []
    for name, info in SUBSECTOR_UNIVERSE.items():
        metrics = compute_subsector_metrics(data, name, info, spy_close)
        if metrics:
            results.append(metrics)
            phase_symbol = {"INFLOW": "+", "ACCUMULATING": "~+", "OUTFLOW": "-",
                            "DISTRIBUTING": "~-", "NEUTRAL": "="}
            log.info(
                f"  {name:25s} [{phase_symbol.get(metrics['rotation_phase'], '?')}] "
                f"score={metrics['rotation_score']:+.2f}  "
                f"ret21d={metrics['metrics']['ret_21d_pct']:+.1f}%  "
                f"relStr={metrics['metrics']['rel_str_vs_spy_21d']:+.1f}%  "
                f"mflow={metrics['metrics']['money_flow_score']:+.2f}"
            )
        else:
            log.warning(f"  {name}: insufficient data")

    if not results:
        log.error("No sub-sector results computed")
        return None

    # Detect macro regime
    regime = detect_macro_regime(data)
    log.info(f"  Macro: VIX={regime.get('vix_level')}, "
             f"regime={regime['vix_regime']}, risk={regime['risk_appetite']}")

    # Detect rotation patterns
    rotation = detect_rotation_patterns(results)
    log.info(f"  Rotation pattern: {rotation['pattern']}")
    log.info(f"  {rotation['description']}")

    # Sort results by rotation score
    results_sorted = sorted(results, key=lambda x: x["rotation_score"], reverse=True)

    # Build output
    output = {
        "generated_at": datetime.now().isoformat(),
        "n_subsectors_tracked": len(results),
        "n_tickers_total": sum(r["n_tickers"] for r in results),
        "macro_regime": regime,
        "rotation_pattern": rotation,
        "subsectors": {r["subsector"]: r for r in results_sorted},
        "rankings": {
            "by_rotation_score": [
                {"rank": i + 1, "subsector": r["subsector"], "score": r["rotation_score"],
                 "phase": r["rotation_phase"], "gics": r["gics_sector"]}
                for i, r in enumerate(results_sorted)
            ],
            "by_momentum_21d": [
                {"rank": i + 1, "subsector": r["subsector"],
                 "ret_21d_pct": r["metrics"]["ret_21d_pct"]}
                for i, r in enumerate(
                    sorted(results, key=lambda x: x["metrics"]["ret_21d_pct"], reverse=True)
                )
            ],
            "by_relative_strength": [
                {"rank": i + 1, "subsector": r["subsector"],
                 "rel_str_21d": r["metrics"]["rel_str_vs_spy_21d"]}
                for i, r in enumerate(
                    sorted(results, key=lambda x: x["metrics"]["rel_str_vs_spy_21d"], reverse=True)
                )
            ],
        },
    }

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Output written to {OUTPUT_PATH}")

    # Save raw data cache for ML model
    try:
        subsector_cache = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "regime": regime,
        }
        with open(SUBSECTOR_CACHE_PATH, "wb") as f:
            pickle.dump(subsector_cache, f)
        log.info(f"Subsector cache saved for ML model")
    except Exception as e:
        log.warning(f"Could not save subsector cache: {e}")

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"ROTATION SUMMARY: {rotation['pattern']}")
    log.info(f"  {rotation['description']}")
    log.info(f"  Top inflow:  {', '.join(r['subsector'] for r in rotation['top_3_inflow'])}")
    log.info(f"  Top outflow: {', '.join(r['subsector'] for r in rotation['bottom_3_outflow'])}")
    log.info(f"{'='*60}\n")

    return output


if __name__ == "__main__":
    run_tracker()
