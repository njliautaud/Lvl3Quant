#!/usr/bin/env python3
"""
Informed Flow Screener — Detects unusual options activity signaling smart money positioning.

Signals detected:
1. Unusual Options Volume (2x+ 20-day avg)
2. IV Skew Shifts (day-over-day ATM IV changes)
3. OI Buildup at Specific Strikes (via enriched data)
4. Volume-to-OI Ratio Spike (new position opening detection)
5. Pre-Earnings Unusual Activity (earnings <= 5 days + unusual flow)
6. Dark Pool / Block Trade Proxy (relative volume 3x+)
7. Cross-Asset Flow Divergence (sector ETF vs constituent flow)
8. Gamma Exposure Estimation (dealer gamma from OI distribution)

Output: /home/jupiter/Lvl3Quant/state/flow_screener_signals.json
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
BASE = Path("/home/jupiter/Lvl3Quant")
FLOW_CSV = BASE / "data" / "options_flow_history.csv"
UNIVERSE_JSON = BASE / "data" / "quality_universe.json"
EARNINGS_JSON = BASE / "state" / "earnings_calendar.json"
ENRICHED_DIR = BASE / "data" / "flow_enriched"
OUTPUT_JSON = BASE / "state" / "flow_screener_signals.json"
LOG_DIR = BASE / "logs" / "flow_screener"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "screener.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("flow_screener")


def load_universe() -> list:
    with open(UNIVERSE_JSON) as f:
        return json.load(f)["tickers"]


def load_flow_data() -> pd.DataFrame:
    """Load options flow history CSV. Expects columns:
    date, timestamp, ticker, price, daily_pct_change, call_volume, put_volume,
    volume_pc_ratio, call_oi, put_oi, oi_pc_ratio, atm_iv, hv_20d,
    iv_hv_spread, relative_volume
    """
    if not FLOW_CSV.exists():
        log.error("Flow CSV not found at %s", FLOW_CSV)
        return pd.DataFrame()
    df = pd.read_csv(FLOW_CSV, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_earnings() -> dict:
    """Load earnings calendar. Returns {ticker: days_until}."""
    if not EARNINGS_JSON.exists():
        return {}
    with open(EARNINGS_JSON) as f:
        data = json.load(f)
    result = {}
    for cat in ["imminent", "upcoming", "scheduled"]:
        for entry in data.get(cat, []):
            result[entry["ticker"]] = entry.get("days_until", 999)
    return result


def load_enriched_data() -> dict:
    """Load strike-level enriched data if available. Returns {ticker: DataFrame}."""
    enriched = {}
    if not ENRICHED_DIR.exists():
        return enriched
    for f in sorted(ENRICHED_DIR.glob("*.parquet")):
        try:
            ticker = f.stem.split("_")[0]
            enriched[ticker] = pd.read_parquet(f)
        except Exception as e:
            log.warning("Failed to load enriched data %s: %s", f.name, e)
    return enriched


# ---------- SECTOR ETF MAPPING ----------
SECTOR_ETFS = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "AMD", "INTU", "TXN", "AMAT", "ACN"],
    "XLV": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ISRG"],
    "XLY": ["AMZN", "HD", "MCD", "LOW", "COST", "NFLX"],
    "XLF": ["JPM", "V", "MA", "BRK-B"],
    "XLP": ["PG", "PEP", "KO", "COST"],
    "XLI": ["LIN"],
    "XLE": [],
    "XLC": ["META", "GOOGL", "NFLX"],
    "XLB": ["LIN"],
    "XLRE": [],
    "XLU": [],
}

# Reverse mapping: stock -> sector ETF
STOCK_TO_SECTOR = {}
for etf, stocks in SECTOR_ETFS.items():
    for s in stocks:
        STOCK_TO_SECTOR[s] = etf


def classify_direction(row: pd.Series) -> str:
    """Classify flow as bullish/bearish/neutral from volume P/C ratio."""
    pc = row.get("volume_pc_ratio", 1.0)
    if pd.isna(pc):
        return "neutral"
    if pc < 0.5:
        return "bullish"  # very low put/call = heavy call buying
    elif pc > 1.5:
        return "bearish"  # heavy put buying
    return "neutral"


# ==========================================
# SIGNAL DETECTORS
# ==========================================

def detect_unusual_volume(df: pd.DataFrame, universe: list) -> list:
    """Signal 1: Options volume 2x+ its 20-day average."""
    signals = []
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]

    for _, row in latest.iterrows():
        ticker = row["ticker"]
        if ticker not in universe:
            continue
        rv = row.get("relative_volume", 1.0)
        if pd.isna(rv):
            continue
        if rv >= 2.0:
            total_vol = row.get("call_volume", 0) + row.get("put_volume", 0)
            direction = classify_direction(row)
            confidence = min(95, int(30 + (rv - 2.0) * 20))
            signals.append({
                "ticker": ticker,
                "signal_type": "unusual_volume",
                "direction": direction,
                "confidence": confidence,
                "evidence": (
                    f"Options volume is {rv:.1f}x the 20-day average "
                    f"({int(total_vol):,} contracts). "
                    f"P/C ratio: {row.get('volume_pc_ratio', 0):.2f}."
                ),
                "metrics": {"relative_volume": round(rv, 2), "total_volume": int(total_vol)},
                "timestamp": datetime.utcnow().isoformat(),
            })
    return signals


def detect_iv_skew_shift(df: pd.DataFrame, universe: list) -> list:
    """Signal 2: ATM IV day-over-day shifts indicating directional positioning."""
    signals = []
    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        return signals

    latest_date = dates[-1]
    prev_date = dates[-2]
    latest = df[df["date"] == latest_date].set_index("ticker")
    prev = df[df["date"] == prev_date].set_index("ticker")

    for ticker in universe:
        if ticker not in latest.index or ticker not in prev.index:
            continue
        curr_iv = latest.loc[ticker, "atm_iv"] if isinstance(latest.loc[ticker], pd.Series) else latest.loc[ticker].iloc[0]["atm_iv"]
        prev_iv = prev.loc[ticker, "atm_iv"] if isinstance(prev.loc[ticker], pd.Series) else prev.loc[ticker].iloc[0]["atm_iv"]

        if pd.isna(curr_iv) or pd.isna(prev_iv) or prev_iv == 0:
            continue

        # Sanity check: IV values < 1% are bad/missing data (no real equity has IV that low)
        if curr_iv < 0.01 or prev_iv < 0.01:
            log.debug("Skipping %s: IV below 1%% threshold (curr=%.4f, prev=%.4f) — likely bad data", ticker, curr_iv, prev_iv)
            continue

        # Sanity check: ATM IV below 5% is stale/missing options data
        if curr_iv < 0.05:
            log.debug("Skipping %s: ATM IV %.2f%% below 5%% floor — stale/missing options data", ticker, curr_iv * 100)
            continue

        iv_change = (curr_iv - prev_iv) / prev_iv
        iv_hv = latest.loc[ticker, "iv_hv_spread"] if isinstance(latest.loc[ticker], pd.Series) else latest.loc[ticker].iloc[0]["iv_hv_spread"]

        # Sanity check: day-over-day IV shifts > 80% are almost certainly data errors
        # (post-earnings IV crush is typically 30-60%, never 80%+)
        is_data_suspect = abs(iv_change) > 0.80

        if abs(iv_change) >= 0.10:  # 10%+ IV shift
            direction = "bearish" if iv_change > 0 else "bullish"  # IV spike = fear/puts
            confidence = min(85, int(30 + abs(iv_change) * 100))

            # Downgrade confidence and flag suspect data
            if is_data_suspect:
                confidence = min(confidence, 25)  # cap confidence for suspect signals
                log.warning("DATA_SUSPECT: %s IV shift %.1f%% exceeds 80%% threshold — likely data error", ticker, iv_change * 100)

            signal_type = "iv_skew_shift_DATA_SUSPECT" if is_data_suspect else "iv_skew_shift"
            suspect_note = " [DATA_SUSPECT: shift exceeds 80% — likely data error, not real IV move]" if is_data_suspect else ""

            signals.append({
                "ticker": ticker,
                "signal_type": signal_type,
                "direction": direction,
                "confidence": confidence,
                "evidence": (
                    f"ATM IV shifted {iv_change:+.1%} day-over-day "
                    f"(from {prev_iv:.1%} to {curr_iv:.1%}). "
                    f"IV-HV spread: {iv_hv:+.1%}."
                    f"{suspect_note}"
                ),
                "metrics": {
                    "iv_change_pct": round(iv_change * 100, 1),
                    "current_iv": round(curr_iv, 4),
                    "iv_hv_spread": round(iv_hv, 4) if not pd.isna(iv_hv) else None,
                    "data_suspect": is_data_suspect,
                },
                "timestamp": datetime.utcnow().isoformat(),
            })
    return signals


def detect_oi_buildup(enriched: dict, universe: list) -> list:
    """Signal 3: Large new OI at specific strikes (from enriched strike-level data)."""
    signals = []
    for ticker in universe:
        if ticker not in enriched:
            continue
        edf = enriched[ticker]
        if edf.empty or "openInterest" not in edf.columns:
            continue

        # Need at least 2 dates to compare
        if "snapshot_date" not in edf.columns:
            continue
        dates = sorted(edf["snapshot_date"].unique())
        if len(dates) < 2:
            continue

        latest = edf[edf["snapshot_date"] == dates[-1]]
        prev = edf[edf["snapshot_date"] == dates[-2]]

        # Compare OI by strike
        for _, row in latest.iterrows():
            strike = row.get("strike")
            opt_type = row.get("option_type", row.get("type", ""))
            curr_oi = row.get("openInterest", 0)
            if pd.isna(curr_oi) or curr_oi < 500:
                continue

            prev_match = prev[
                (prev["strike"] == strike)
                & (prev.get("option_type", prev.get("type", "")) == opt_type)
            ]
            if prev_match.empty:
                prev_oi = 0
            else:
                prev_oi = prev_match.iloc[0].get("openInterest", 0)

            oi_change = curr_oi - prev_oi
            if oi_change > 1000 and (prev_oi == 0 or oi_change / max(prev_oi, 1) > 0.5):
                direction = "bearish" if str(opt_type).lower().startswith("p") else "bullish"
                confidence = min(80, int(30 + (oi_change / 1000) * 5))
                signals.append({
                    "ticker": ticker,
                    "signal_type": "oi_buildup",
                    "direction": direction,
                    "confidence": confidence,
                    "evidence": (
                        f"Large OI buildup at ${strike} {opt_type}: "
                        f"+{oi_change:,} contracts (now {int(curr_oi):,} total). "
                        f"Institutional-sized positioning."
                    ),
                    "metrics": {
                        "strike": strike,
                        "option_type": str(opt_type),
                        "oi_change": int(oi_change),
                        "current_oi": int(curr_oi),
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                })
    return signals


def detect_volume_oi_spike(df: pd.DataFrame, universe: list) -> list:
    """Signal 4: Volume >> OI = new positions being opened."""
    signals = []
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]

    for _, row in latest.iterrows():
        ticker = row["ticker"]
        if ticker not in universe:
            continue

        call_vol = row.get("call_volume", 0)
        put_vol = row.get("put_volume", 0)
        call_oi = row.get("call_oi", 1)
        put_oi = row.get("put_oi", 1)

        if pd.isna(call_vol) or pd.isna(put_vol):
            continue

        # Put volume >> put OI = new bearish bets
        put_vol_oi = put_vol / max(put_oi, 1)
        call_vol_oi = call_vol / max(call_oi, 1)

        if put_vol_oi > 2.0:
            confidence = min(85, int(35 + (put_vol_oi - 2.0) * 15))
            signals.append({
                "ticker": ticker,
                "signal_type": "volume_oi_spike",
                "direction": "bearish",
                "confidence": confidence,
                "evidence": (
                    f"Put volume is {put_vol_oi:.1f}x put OI — "
                    f"new bearish positions being opened. "
                    f"Put vol: {int(put_vol):,}, Put OI: {int(put_oi):,}."
                ),
                "metrics": {"put_vol_oi_ratio": round(put_vol_oi, 2)},
                "timestamp": datetime.utcnow().isoformat(),
            })

        if call_vol_oi > 2.0:
            confidence = min(85, int(35 + (call_vol_oi - 2.0) * 15))
            signals.append({
                "ticker": ticker,
                "signal_type": "volume_oi_spike",
                "direction": "bullish",
                "confidence": confidence,
                "evidence": (
                    f"Call volume is {call_vol_oi:.1f}x call OI — "
                    f"new bullish positions being opened. "
                    f"Call vol: {int(call_vol):,}, Call OI: {int(call_oi):,}."
                ),
                "metrics": {"call_vol_oi_ratio": round(call_vol_oi, 2)},
                "timestamp": datetime.utcnow().isoformat(),
            })
    return signals


def detect_pre_earnings_flow(df: pd.DataFrame, universe: list, earnings: dict) -> list:
    """Signal 5: Unusual flow on tickers with earnings <= 5 days out. Highest-value signal."""
    signals = []
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]

    for _, row in latest.iterrows():
        ticker = row["ticker"]
        if ticker not in universe:
            continue
        days_until = earnings.get(ticker, 999)
        if days_until > 5:
            continue

        # Any unusual activity on an earnings-imminent ticker is high-value
        rv = row.get("relative_volume", 1.0)
        pc_ratio = row.get("volume_pc_ratio", 1.0)
        iv_hv = row.get("iv_hv_spread", 0)

        if pd.isna(rv):
            rv = 1.0

        unusual_factors = []
        base_confidence = 40

        if rv >= 1.5:
            unusual_factors.append(f"volume {rv:.1f}x normal")
            base_confidence += 15
        if not pd.isna(pc_ratio) and (pc_ratio > 1.3 or pc_ratio < 0.4):
            direction_word = "put-heavy" if pc_ratio > 1.3 else "call-heavy"
            unusual_factors.append(f"{direction_word} (P/C={pc_ratio:.2f})")
            base_confidence += 10
        if not pd.isna(iv_hv) and iv_hv > 0.05:
            unusual_factors.append(f"IV premium +{iv_hv:.1%} over HV")
            base_confidence += 10

        if not unusual_factors:
            # Still flag earnings-imminent tickers even without unusual flow
            unusual_factors.append("earnings imminent — monitoring")
            base_confidence = 30

        direction = classify_direction(row)
        confidence = min(95, base_confidence)

        signals.append({
            "ticker": ticker,
            "signal_type": "pre_earnings_flow",
            "direction": direction,
            "confidence": confidence,
            "evidence": (
                f"Earnings in {days_until} day(s). "
                f"Flow characteristics: {'; '.join(unusual_factors)}."
            ),
            "metrics": {
                "days_until_earnings": days_until,
                "relative_volume": round(rv, 2),
                "pc_ratio": round(pc_ratio, 2) if not pd.isna(pc_ratio) else None,
            },
            "timestamp": datetime.utcnow().isoformat(),
        })
    return signals


def detect_dark_pool_proxy(df: pd.DataFrame, universe: list) -> list:
    """Signal 6: Relative equity volume 3x+ = big players moving."""
    signals = []
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]

    for _, row in latest.iterrows():
        ticker = row["ticker"]
        if ticker not in universe:
            continue
        rv = row.get("relative_volume", 1.0)
        if pd.isna(rv) or rv < 3.0:
            continue

        direction = classify_direction(row)
        price_chg = row.get("daily_pct_change", 0)
        if not pd.isna(price_chg):
            if price_chg > 1:
                direction = "bullish"
            elif price_chg < -1:
                direction = "bearish"

        confidence = min(90, int(40 + (rv - 3.0) * 15))
        signals.append({
            "ticker": ticker,
            "signal_type": "dark_pool_proxy",
            "direction": direction,
            "confidence": confidence,
            "evidence": (
                f"Relative volume is {rv:.1f}x normal — "
                f"institutional-sized activity. "
                f"Price moved {price_chg:+.1f}% on the day."
            ),
            "metrics": {"relative_volume": round(rv, 2), "price_change_pct": round(price_chg, 1) if not pd.isna(price_chg) else None},
            "timestamp": datetime.utcnow().isoformat(),
        })
    return signals


def detect_cross_asset_divergence(df: pd.DataFrame) -> list:
    """Signal 7: When sector ETF flow diverges from constituent stock flow."""
    signals = []
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].set_index("ticker")

    for etf, constituents in SECTOR_ETFS.items():
        if etf not in latest.index or not constituents:
            continue

        etf_row = latest.loc[etf]
        if isinstance(etf_row, pd.DataFrame):
            etf_row = etf_row.iloc[0]
        etf_pc = etf_row.get("volume_pc_ratio", 1.0)
        if pd.isna(etf_pc):
            continue
        etf_direction = "bearish" if etf_pc > 1.2 else ("bullish" if etf_pc < 0.6 else "neutral")

        # Average constituent P/C
        constituent_pcs = []
        for stock in constituents:
            if stock in latest.index:
                srow = latest.loc[stock]
                if isinstance(srow, pd.DataFrame):
                    srow = srow.iloc[0]
                spc = srow.get("volume_pc_ratio", None)
                if not pd.isna(spc):
                    constituent_pcs.append(spc)

        if len(constituent_pcs) < 2:
            continue

        avg_stock_pc = np.mean(constituent_pcs)
        stock_direction = "bearish" if avg_stock_pc > 1.2 else ("bullish" if avg_stock_pc < 0.6 else "neutral")

        # Divergence: ETF says one thing, stocks say another
        if etf_direction != stock_direction and etf_direction != "neutral" and stock_direction != "neutral":
            confidence = min(75, int(40 + abs(etf_pc - avg_stock_pc) * 30))
            signals.append({
                "ticker": etf,
                "signal_type": "cross_asset_divergence",
                "direction": "neutral",  # divergence is informational
                "confidence": confidence,
                "evidence": (
                    f"{etf} flow is {etf_direction} (P/C={etf_pc:.2f}) but "
                    f"constituent stocks are {stock_direction} (avg P/C={avg_stock_pc:.2f}). "
                    f"Likely institutional hedging, not directional conviction."
                ),
                "metrics": {
                    "etf_pc_ratio": round(etf_pc, 2),
                    "avg_constituent_pc": round(avg_stock_pc, 2),
                },
                "timestamp": datetime.utcnow().isoformat(),
            })
    return signals


def estimate_gamma_exposure(enriched: dict, universe: list) -> list:
    """Signal 8: Estimate dealer gamma from OI distribution near current price.
    When dealers are short gamma (lots of OI near spot), moves accelerate.
    When long gamma, price is pinned. Predicts volatility, not direction.
    """
    signals = []
    for ticker in universe:
        if ticker not in enriched:
            continue
        edf = enriched[ticker]
        if edf.empty:
            continue

        # Need strike, OI, option_type, and spot price
        required = {"strike", "openInterest", "snapshot_date"}
        if not required.issubset(set(edf.columns)):
            continue

        latest_date = edf["snapshot_date"].max()
        latest = edf[edf["snapshot_date"] == latest_date].copy()

        if "spot_price" not in latest.columns:
            continue

        spot = latest["spot_price"].iloc[0]
        if pd.isna(spot) or spot <= 0:
            continue

        # OI within 5% of spot price
        near_money = latest[
            (latest["strike"] >= spot * 0.95)
            & (latest["strike"] <= spot * 1.05)
        ]

        if near_money.empty:
            continue

        total_near_oi = near_money["openInterest"].sum()
        total_oi = latest["openInterest"].sum()
        if total_oi == 0:
            continue

        oi_concentration = total_near_oi / total_oi

        # High concentration near spot = dealers likely have big gamma positions
        # Call OI near spot = dealers likely long gamma (sold calls, delta hedged)
        # Put OI near spot = dealers likely short gamma
        opt_type_col = "option_type" if "option_type" in near_money.columns else "type"
        if opt_type_col not in near_money.columns:
            continue

        call_oi_near = near_money[near_money[opt_type_col].str.lower().str.startswith("c")]["openInterest"].sum()
        put_oi_near = near_money[near_money[opt_type_col].str.lower().str.startswith("p")]["openInterest"].sum()

        if call_oi_near > put_oi_near * 1.5:
            gamma_state = "long_gamma"
            evidence_text = "Dealers likely LONG gamma (call OI dominant near spot). Price pinning expected — low vol."
        elif put_oi_near > call_oi_near * 1.5:
            gamma_state = "short_gamma"
            evidence_text = "Dealers likely SHORT gamma (put OI dominant near spot). Moves may accelerate — high vol."
        else:
            gamma_state = "balanced"
            evidence_text = "Gamma exposure roughly balanced near spot."

        if oi_concentration > 0.3:  # meaningful concentration
            confidence = min(70, int(25 + oi_concentration * 100))
            signals.append({
                "ticker": ticker,
                "signal_type": "gamma_exposure",
                "direction": "neutral",  # gamma predicts vol, not direction
                "confidence": confidence,
                "evidence": (
                    f"{evidence_text} "
                    f"OI concentration near spot: {oi_concentration:.0%} "
                    f"({int(total_near_oi):,} of {int(total_oi):,} contracts within 5% of ${spot:.0f})."
                ),
                "metrics": {
                    "gamma_state": gamma_state,
                    "oi_concentration": round(oi_concentration, 3),
                    "call_oi_near": int(call_oi_near),
                    "put_oi_near": int(put_oi_near),
                    "spot_price": round(spot, 2),
                },
                "timestamp": datetime.utcnow().isoformat(),
            })
    return signals


def build_alerts(all_signals: list, earnings: dict) -> list:
    """Build IMMINENT alerts: high-confidence signals on earnings-imminent tickers."""
    alerts = []
    for sig in all_signals:
        ticker = sig["ticker"]
        days = earnings.get(ticker, 999)
        is_earnings_imminent = days <= 5
        is_high_confidence = sig["confidence"] >= 60

        if is_earnings_imminent and is_high_confidence:
            alerts.append({
                **sig,
                "alert_reason": f"Earnings in {days} day(s) with high-confidence {sig['signal_type']} signal",
                "urgency": "high" if days <= 2 else "medium",
            })
        elif sig["confidence"] >= 80 and sig["signal_type"] in ("unusual_volume", "dark_pool_proxy"):
            alerts.append({
                **sig,
                "alert_reason": f"Very high confidence {sig['signal_type']} — possible informed positioning",
                "urgency": "medium",
            })
    return sorted(alerts, key=lambda x: (-x["confidence"], x.get("urgency", "low") == "high"))


def run():
    log.info("=== Informed Flow Screener starting ===")

    universe = load_universe()
    log.info("Universe: %d tickers", len(universe))

    df = load_flow_data()
    if df.empty:
        log.error("No flow data available. Exiting.")
        return

    log.info("Flow data: %d rows, dates %s to %s", len(df), df["date"].min().date(), df["date"].max().date())

    earnings = load_earnings()
    log.info("Earnings calendar: %d tickers tracked", len(earnings))

    enriched = load_enriched_data()
    log.info("Enriched strike data available for %d tickers", len(enriched))

    # Run all detectors
    all_signals = []

    detectors = [
        ("unusual_volume", lambda: detect_unusual_volume(df, universe)),
        ("iv_skew_shift", lambda: detect_iv_skew_shift(df, universe)),
        ("oi_buildup", lambda: detect_oi_buildup(enriched, universe)),
        ("volume_oi_spike", lambda: detect_volume_oi_spike(df, universe)),
        ("pre_earnings_flow", lambda: detect_pre_earnings_flow(df, universe, earnings)),
        ("dark_pool_proxy", lambda: detect_dark_pool_proxy(df, universe)),
        ("cross_asset_divergence", lambda: detect_cross_asset_divergence(df)),
        ("gamma_exposure", lambda: estimate_gamma_exposure(enriched, universe)),
    ]

    for name, detector in detectors:
        try:
            sigs = detector()
            log.info("  %s: %d signals", name, len(sigs))
            all_signals.extend(sigs)
        except Exception as e:
            log.error("  %s FAILED: %s", name, e, exc_info=True)

    # Build imminent alerts
    alerts = build_alerts(all_signals, earnings)
    log.info("Total signals: %d, Imminent alerts: %d", len(all_signals), len(alerts))

    # Sort signals by confidence descending
    all_signals.sort(key=lambda x: -x["confidence"])

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "data_through": str(df["date"].max().date()),
        "universe_size": len(universe),
        "total_signals": len(all_signals),
        "total_alerts": len(alerts),
        "signals": all_signals,
        "alerts": alerts,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info("Output written to %s", OUTPUT_JSON)

    # Summary
    if alerts:
        log.info("=== IMMINENT ALERTS ===")
        for a in alerts:
            log.info(
                "  [%s] %s — %s (%s, conf=%d): %s",
                a.get("urgency", "?").upper(),
                a["ticker"],
                a["signal_type"],
                a["direction"],
                a["confidence"],
                a["evidence"],
            )

    log.info("=== Screener complete ===")
    return output


if __name__ == "__main__":
    run()
