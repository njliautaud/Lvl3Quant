#!/usr/bin/env python3
"""
Broad Market Sector Screener — HC #775
========================================

Extends coverage beyond megacaps into small/mid caps, mining & equipment,
healthcare, industrials, and materials. Runs daily, scores tickers on
momentum + mean-reversion + relative strength, and feeds signals to
the agentic signal aggregator.

Sectors (per user directive):
  - Mining & Metals: FCX, NEM, GOLD, AEM, RIO, TECK, SCCO, CLF, X, AA
  - Equipment & Industrials: CAT, DE, PCAR, CMI, EMR, ETN, ROK, IR
  - Healthcare (beyond megacaps): ISRG, VRTX, DXCM, HOLX, ALGN, VEEV, PODD, STE
  - Biotech: IBB (ETF), REGN, MRNA, BIIB, GILD, ILMN
  - Small-Cap ETFs: IWM, SCHA, IJR, VB
  - Precious Metals ETFs: GDX, GDXJ, GLD, SLV
  - Materials & Chemicals: LIN, APD, ECL, SHW, DD, EMN, CE

Usage:
  python3 paper_engines/broad_market_sector_screener.py
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'broad_market_sector_screener.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

STATE_DIR = Path(__file__).resolve().parent.parent / 'state'
STATE_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = STATE_DIR / 'broad_market_signals.json'

# ---------------------------------------------------------------------------
# UNIVERSE — beyond megacaps, focused on user-requested sectors
# ---------------------------------------------------------------------------
SECTOR_UNIVERSE = {
    "mining_metals": {
        "tickers": ["FCX", "NEM", "GOLD", "AEM", "RIO", "TECK", "SCCO", "CLF", "AA", "HBM"],
        "etf": "GDX",
        "description": "Gold/copper/steel miners and metal producers"
    },
    "equipment_industrials": {
        "tickers": ["CAT", "DE", "PCAR", "CMI", "EMR", "ETN", "ROK", "IR"],
        "etf": "XLI",
        "description": "Heavy equipment, farm equipment, industrial machinery"
    },
    "healthcare_medtech": {
        "tickers": ["ISRG", "VRTX", "DXCM", "EW", "ALGN", "VEEV", "PODD", "STE"],
        "etf": "XLV",
        "description": "Medical devices, surgical robots, diagnostics"
    },
    "biotech": {
        "tickers": ["REGN", "MRNA", "BIIB", "GILD", "ILMN", "BMRN"],
        "etf": "IBB",
        "description": "Biotech and genomics"
    },
    "small_cap_etfs": {
        "tickers": ["IWM", "SCHA", "IJR", "VB"],
        "etf": None,
        "description": "Small and micro-cap index ETFs"
    },
    "precious_metals": {
        "tickers": ["GLD", "SLV", "GDX", "GDXJ"],
        "etf": None,
        "description": "Gold, silver, and miner ETFs"
    },
    "materials_chemicals": {
        "tickers": ["LIN", "APD", "ECL", "SHW", "DD", "EMN", "CE"],
        "etf": "XLB",
        "description": "Specialty chemicals, coatings, industrial gases"
    },
    "defense": {
        "tickers": ["LMT", "RTX", "NOC", "GD", "HII", "LHX"],
        "etf": "ITA",
        "description": "Aerospace and defense"
    },
}

# All unique tickers
ALL_TICKERS = sorted(set(
    t for sector in SECTOR_UNIVERSE.values()
    for t in sector["tickers"]
    if t  # skip None
))

# Add sector ETFs
SECTOR_ETFS = [s["etf"] for s in SECTOR_UNIVERSE.values() if s.get("etf")]
ALL_TICKERS_WITH_ETFS = sorted(set(ALL_TICKERS + SECTOR_ETFS))

# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Compute RSI for the latest bar."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def compute_signals(df: pd.DataFrame, ticker: str) -> dict:
    """Compute momentum + mean-reversion signals for a single ticker."""
    if df is None or len(df) < 60:
        return None

    close = df['Close']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)

    try:
        ret_5d = float(close.pct_change(5).iloc[-1])
        ret_10d = float(close.pct_change(10).iloc[-1])
        ret_21d = float(close.pct_change(21).iloc[-1])
        ret_63d = float(close.pct_change(63).iloc[-1]) if len(close) >= 64 else 0.0

        rsi_5 = compute_rsi(close, 5)
        rsi_14 = compute_rsi(close, 14)

        # SMA positioning
        sma_20 = float(close.rolling(20).mean().iloc[-1])
        sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma_20
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma_50
        price = float(close.iloc[-1])

        above_sma20 = price > sma_20
        above_sma50 = price > sma_50
        above_sma200 = price > sma_200

        # Volatility (20d realized)
        log_ret = np.log(close / close.shift(1)).dropna()
        vol_20d = float(log_ret.tail(20).std() * np.sqrt(252)) if len(log_ret) >= 20 else 0.3

        # Volume trend
        vol_avg_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else 0
        vol_today = float(volume.iloc[-1])
        vol_ratio = vol_today / vol_avg_20 if vol_avg_20 > 0 else 1.0

        # Composite score: momentum + mean reversion + trend
        # Momentum component (higher = more bullish momentum)
        mom_score = 0.0
        mom_score += np.clip(ret_5d * 10, -2, 2)    # short-term momentum
        mom_score += np.clip(ret_21d * 5, -2, 2)     # medium-term
        mom_score += np.clip(ret_63d * 3, -2, 2)     # longer-term trend

        # Mean-reversion component (oversold = bullish signal)
        mr_score = 0.0
        if rsi_5 < 20:
            mr_score += 2.0  # deeply oversold
        elif rsi_5 < 30:
            mr_score += 1.0
        elif rsi_5 > 80:
            mr_score -= 2.0  # overbought
        elif rsi_5 > 70:
            mr_score -= 1.0

        # Trend alignment
        trend_score = 0.0
        if above_sma200:
            trend_score += 1.0
        if above_sma50:
            trend_score += 0.5
        if above_sma20:
            trend_score += 0.5

        # Volume confirmation
        vol_score = 0.0
        if vol_ratio > 1.5 and ret_5d > 0:
            vol_score += 0.5  # high volume on up move
        elif vol_ratio > 1.5 and ret_5d < 0:
            vol_score -= 0.5  # high volume on down move

        total_score = mom_score + mr_score + trend_score + vol_score

        # Determine signal direction
        if total_score > 2.0:
            signal = "BULLISH"
            confidence = min(0.95, 0.5 + (total_score - 2.0) * 0.1)
        elif total_score < -2.0:
            signal = "BEARISH"
            confidence = min(0.95, 0.5 + abs(total_score + 2.0) * 0.1)
        else:
            signal = "NEUTRAL"
            confidence = 0.3

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "signal": signal,
            "confidence": round(confidence, 3),
            "total_score": round(total_score, 2),
            "components": {
                "momentum": round(mom_score, 2),
                "mean_reversion": round(mr_score, 2),
                "trend": round(trend_score, 2),
                "volume": round(vol_score, 2),
            },
            "indicators": {
                "rsi_5": round(rsi_5, 1),
                "rsi_14": round(rsi_14, 1),
                "ret_5d_pct": round(ret_5d * 100, 2),
                "ret_21d_pct": round(ret_21d * 100, 2),
                "ret_63d_pct": round(ret_63d * 100, 2),
                "vol_20d": round(vol_20d, 3),
                "vol_ratio": round(vol_ratio, 2),
                "above_sma200": above_sma200,
            },
        }
    except Exception as e:
        log.warning(f"  Error computing signals for {ticker}: {e}")
        return None


def run_screener():
    """Main screener loop."""
    if not HAS_YF:
        log.error("yfinance not installed. Cannot run screener.")
        return

    log.info(f"=== Broad Market Sector Screener — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    log.info(f"Scanning {len(ALL_TICKERS_WITH_ETFS)} tickers across {len(SECTOR_UNIVERSE)} sectors")

    # Download data (250 days for SMA200 + buffer)
    try:
        data = yf.download(ALL_TICKERS_WITH_ETFS, period="1y", progress=False, group_by='ticker')
    except Exception as e:
        log.error(f"Download failed: {e}")
        return

    results_by_sector = {}
    all_signals = []
    top_bullish = []
    top_bearish = []

    for sector_name, sector_info in SECTOR_UNIVERSE.items():
        sector_signals = []
        tickers = sector_info["tickers"]
        if sector_info.get("etf"):
            tickers = tickers + [sector_info["etf"]]

        for ticker in tickers:
            try:
                if len(ALL_TICKERS_WITH_ETFS) > 1:
                    if ticker in data.columns.get_level_values(0):
                        ticker_df = data[ticker].dropna()
                    else:
                        continue
                else:
                    ticker_df = data.dropna()

                sig = compute_signals(ticker_df, ticker)
                if sig:
                    sig["sector"] = sector_name
                    sector_signals.append(sig)
                    all_signals.append(sig)
            except Exception as e:
                log.warning(f"  Skipping {ticker}: {e}")

        results_by_sector[sector_name] = {
            "description": sector_info["description"],
            "signals": sorted(sector_signals, key=lambda x: x["total_score"], reverse=True),
            "count": len(sector_signals),
        }

        # Log sector summary
        bullish = [s for s in sector_signals if s["signal"] == "BULLISH"]
        bearish = [s for s in sector_signals if s["signal"] == "BEARISH"]
        log.info(f"  {sector_name}: {len(bullish)} bullish, {len(bearish)} bearish, "
                 f"{len(sector_signals) - len(bullish) - len(bearish)} neutral")

    # Sort for top signals
    all_sorted = sorted(all_signals, key=lambda x: abs(x["total_score"]), reverse=True)
    top_bullish = [s for s in all_sorted if s["signal"] == "BULLISH"][:10]
    top_bearish = [s for s in all_sorted if s["signal"] == "BEARISH"][:10]

    # Actionable signals (high confidence, either direction)
    actionable = [s for s in all_signals if s["confidence"] >= 0.65]
    actionable_sorted = sorted(actionable, key=lambda x: x["confidence"], reverse=True)

    output = {
        "generated_at": datetime.now().isoformat(),
        "total_tickers_scanned": len(all_signals),
        "sectors_scanned": len(SECTOR_UNIVERSE),
        "summary": {
            "bullish_count": len([s for s in all_signals if s["signal"] == "BULLISH"]),
            "bearish_count": len([s for s in all_signals if s["signal"] == "BEARISH"]),
            "neutral_count": len([s for s in all_signals if s["signal"] == "NEUTRAL"]),
        },
        "top_bullish": top_bullish[:5],
        "top_bearish": top_bearish[:5],
        "actionable_signals": actionable_sorted[:10],
        "sectors": results_by_sector,
    }

    # Write output
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Output written to {OUTPUT_PATH}")

    # Summary log
    log.info(f"\n{'='*60}")
    log.info(f"SUMMARY: {output['summary']['bullish_count']} bullish, "
             f"{output['summary']['bearish_count']} bearish, "
             f"{output['summary']['neutral_count']} neutral")
    if top_bullish:
        bull_str = ', '.join(f"{s['ticker']}({s['total_score']})" for s in top_bullish[:5])
        log.info(f"TOP BULLISH: {bull_str}")
    if top_bearish:
        bear_str = ', '.join(f"{s['ticker']}({s['total_score']})" for s in top_bearish[:5])
        log.info(f"TOP BEARISH: {bear_str}")
    if actionable_sorted:
        act_str = ', '.join(f"{s['ticker']}({s['signal'][0]},{s['confidence']})" for s in actionable_sorted[:5])
        log.info(f"ACTIONABLE ({len(actionable_sorted)}): {act_str}")
    log.info(f"{'='*60}\n")

    return output


if __name__ == "__main__":
    run_screener()
