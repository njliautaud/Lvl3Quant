"""
Unified Portfolio Brain — ONE system that ingests ALL signals and outputs a daily trade plan.

Inputs:
  - agentic_signals.json (11 strategy signals)
  - flow_screener_signals.json (flow screener — optional)
  - earnings_calendar.json (earnings proximity)
  - spread_recommendations.json (spread structures from spread_executor)

Processing:
  1. Signal Aggregation — count strategy agreement per ticker
  2. Position Sizing (Kelly Criterion) — edge * confidence
  3. Correlation/Sector Check — reduce concentrated risk
  4. Regime Overlay — VIX gating
  5. Earnings Avoidance — skip tickers with imminent earnings
  6. Spread Selection — route to spread structures

Output: daily_trade_plan.json
"""

import json
import os
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_DIR = os.path.join(BASE_DIR, "state")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Brain] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "unified_brain.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Risk parameters
DEFAULT_ACCOUNT_EQUITY = 750.0
MAX_POSITION_PCT = 0.20  # max 20% per trade
MAX_PORTFOLIO_HEAT = 0.60  # max 60% total capital at risk
MIN_CONFIDENCE = 0.40  # skip signals below this
HIGH_CONVICTION_SOURCES = 3  # 3+ confirming sources = high conviction

# Sector mapping for correlation check
TICKER_SECTOR = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "AMZN": "tech",
    "NVDA": "tech", "META": "tech", "AMD": "tech", "CRM": "tech",
    "AVGO": "tech", "INTU": "tech", "TXN": "tech", "AMAT": "tech",
    "NFLX": "tech", "ACN": "tech",
    "BRK-B": "finance", "JPM": "finance", "V": "finance", "MA": "finance",
    "LLY": "health", "UNH": "health", "JNJ": "health", "ABBV": "health",
    "MRK": "health", "TMO": "health", "ISRG": "health",
    "PG": "consumer", "HD": "consumer", "COST": "consumer", "MCD": "consumer",
    "PEP": "consumer", "KO": "consumer", "LOW": "consumer",
    "LIN": "industrial",
    "XLK": "tech", "XLF": "finance", "XLE": "energy", "XLV": "health",
    "XLI": "industrial", "XLP": "consumer", "XLU": "utilities",
    "XLB": "materials", "XLC": "comm", "XLRE": "real_estate", "XLY": "consumer",
    "SPY": "broad", "QQQ": "tech",
}

# VIX regime thresholds
VIX_HIGH = 25
VIX_EXTREME = 35

# Earnings avoidance window
EARNINGS_AVOID_DAYS = 2


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[Dict]:
    """Load JSON file, return None on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning(f"Could not load {os.path.basename(path)}: {e}")
        return None


def load_agentic_signals() -> Dict:
    return _load_json(os.path.join(STATE_DIR, "agentic_signals.json")) or {}


def load_flow_screener() -> Dict:
    return _load_json(os.path.join(STATE_DIR, "flow_screener_signals.json")) or {}


def load_earnings_calendar() -> Dict:
    return _load_json(os.path.join(STATE_DIR, "earnings_calendar.json")) or {}


def load_spread_recommendations() -> Dict:
    return _load_json(os.path.join(STATE_DIR, "spread_recommendations.json")) or {}


def load_quality_universe() -> List[str]:
    data = _load_json(os.path.join(DATA_DIR, "quality_universe.json"))
    if data and "tickers" in data:
        return data["tickers"]
    return []


# ---------------------------------------------------------------------------
# Step 1: Signal Aggregation
# ---------------------------------------------------------------------------

def aggregate_signals(
    agentic: Dict, flow: Dict
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate signals across all sources.
    Returns dict keyed by ticker with direction, confidence, sources, etc.
    """
    ticker_signals = defaultdict(lambda: {
        "directions": [], "confidences": [], "sources": [],
        "reasons": [], "exit_guidance": None,
    })

    # Agentic signals (primary)
    for sig in agentic.get("signals", []):
        ticker = sig.get("ticker", "")
        if not ticker:
            continue
        entry = ticker_signals[ticker]
        entry["directions"].append(sig.get("direction", ""))
        entry["confidences"].append(sig.get("confidence_score", 0))
        entry["sources"].extend(sig.get("confirming_sources", []))
        entry["reasons"].append(sig.get("reason", ""))
        if sig.get("exit_guidance"):
            entry["exit_guidance"] = sig["exit_guidance"]

    # Flow screener signals (supplementary)
    # Break out sub-types as separate sources so multiple flow detectors
    # firing on the same ticker naturally boost conviction (n_sources > 1).
    # Sub-type field is "signal_type" in flow signal JSON.
    FLOW_SUBTYPE_TO_SOURCE = {
        "unusual_volume": "flow_unusual_volume",
        "iv_skew_shift": "flow_iv_skew_shift",
        "iv_skew_shift_DATA_SUSPECT": "flow_iv_skew_shift",  # map suspect variant to same source
        "volume_oi_spike": "flow_volume_oi_spike",
        "pre_earnings_flow": "flow_pre_earnings",
        "dark_pool_proxy": "flow_dark_pool_proxy",
        "cross_asset_divergence": "flow_cross_asset_divergence",
        "gamma_exposure": "flow_gamma_exposure",
        "oi_buildup": "flow_oi_buildup",
    }
    for sig in flow.get("signals", []):
        ticker = sig.get("ticker", "")
        if not ticker:
            continue
        entry = ticker_signals[ticker]
        entry["directions"].append(sig.get("direction", ""))
        # Flow screener uses 0-100 scale; normalize to 0-1
        raw_conf = sig.get("confidence", 0)
        conf = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
        entry["confidences"].append(conf)
        # Use sub-type as source label; fall back to generic if unknown
        sub_type = sig.get("signal_type", "unknown")
        source_label = FLOW_SUBTYPE_TO_SOURCE.get(sub_type, f"flow_{sub_type}")
        entry["sources"].append(source_label)

    # Compute consensus for each ticker
    result = {}
    for ticker, data in ticker_signals.items():
        # Majority direction
        bull_count = sum(1 for d in data["directions"] if d in ("bull", "bullish", "long"))
        bear_count = sum(1 for d in data["directions"] if d in ("bear", "bearish", "short"))
        direction = "bull" if bull_count >= bear_count else "bear"

        # Average confidence
        avg_conf = sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else 0

        # Unique sources
        unique_sources = list(set(data["sources"]))
        n_sources = len(unique_sources)

        # Conviction level
        if n_sources >= HIGH_CONVICTION_SOURCES:
            conviction = "high"
        elif n_sources >= 2:
            conviction = "medium"
        else:
            conviction = "low"

        result[ticker] = {
            "direction": direction,
            "confidence": avg_conf,
            "n_sources": n_sources,
            "sources": unique_sources,
            "conviction": conviction,
            "reasons": data["reasons"],
            "exit_guidance": data["exit_guidance"],
        }

    return result


# ---------------------------------------------------------------------------
# Step 2: Kelly Criterion Position Sizing
# ---------------------------------------------------------------------------

def kelly_size(
    confidence: float,
    historical_sharpe: float = 1.0,
    account_equity: float = DEFAULT_ACCOUNT_EQUITY,
) -> float:
    """
    Kelly-inspired position sizing for a small options account (~$750).

    Practical approach for defined-risk spreads:
    - win_prob = confidence (our signal's estimated win rate)
    - odds = 2.0 (typical spread pays 2:1)
    - Kelly f* = (odds * p - (1-p)) / odds
    - We use quarter-Kelly for safety
    - Minimum trade = $30 (below this not worth it on RH)

    Returns dollar amount to risk.
    """
    if confidence < MIN_CONFIDENCE:
        return 0.0

    win_prob = confidence
    odds = 2.0  # typical spread reward/risk ratio

    # Kelly criterion: f* = (b*p - q) / b where b=odds, p=win_prob, q=1-p
    kelly_f = (odds * win_prob - (1 - win_prob)) / odds
    kelly_f = max(0, kelly_f)

    # Quarter Kelly for safety (spreads have binary outcomes)
    quarter_kelly = kelly_f * 0.25

    # Sharpe adjustment
    sharpe_adj = min(1.5, max(0.5, historical_sharpe))
    position_pct = quarter_kelly * sharpe_adj

    # Cap at max position
    position_pct = min(position_pct, MAX_POSITION_PCT)

    size = round(position_pct * account_equity, 2)

    # Minimum trade size for RH spreads
    if size < 30:
        return 0.0

    return size


# ---------------------------------------------------------------------------
# Step 3: Sector Correlation Check
# ---------------------------------------------------------------------------

def check_sector_concentration(
    trades: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    If 3+ signals fire on the same sector, reduce position sizes by 50%.
    """
    sector_counts = defaultdict(int)
    for t in trades:
        sector = TICKER_SECTOR.get(t["ticker"], "other")
        sector_counts[sector] += 1

    concentrated_sectors = {s for s, c in sector_counts.items() if c >= 3}

    if not concentrated_sectors:
        return trades

    adjusted = []
    for t in trades:
        sector = TICKER_SECTOR.get(t["ticker"], "other")
        if sector in concentrated_sectors:
            t = dict(t)  # copy
            t["size"] = round(t["size"] * 0.50, 2)
            t["size_adjustment"] = f"Reduced 50% — {sector} sector concentrated ({sector_counts[sector]} signals)"
            log.info(f"{t['ticker']}: sector concentration adjustment ({sector})")
        adjusted.append(t)

    return adjusted


# ---------------------------------------------------------------------------
# Step 4: Regime Overlay (VIX gating)
# ---------------------------------------------------------------------------

def apply_regime_overlay(
    trades: List[Dict[str, Any]], vix_level: float
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Apply VIX-based regime overlay (informational only).
    Backtesting showed VIX gating HURTS performance — mean-reversion dip-buying
    thrives in high-vol periods (HC #771 backtest finding). VIX is logged for
    awareness but does NOT reduce positions or reject trades.
    VIX > 35 only: flag as extreme for manual review, but still pass trades through.
    """
    if vix_level > VIX_EXTREME:
        regime = "extreme_fear"
        log.warning(f"VIX={vix_level:.1f} > {VIX_EXTREME}: EXTREME — trades passed through (MR thrives in fear)")
        for t in trades:
            t["regime_note"] = f"VIX={vix_level:.1f} extreme — review manually"
        return trades, regime

    regime = "normal"
    if vix_level > VIX_HIGH:
        regime = "elevated_vol"
        log.info(f"VIX={vix_level:.1f}: elevated vol — MR signals historically stronger here")

    return trades, regime


# ---------------------------------------------------------------------------
# Step 5: Earnings Avoidance
# ---------------------------------------------------------------------------

def filter_earnings(
    trades: List[Dict[str, Any]], earnings_data: Dict
) -> List[Dict[str, Any]]:
    """
    Skip tickers with earnings in <= EARNINGS_AVOID_DAYS,
    unless the signal source IS an earnings strategy.
    """
    imminent_tickers = set()
    for entry in earnings_data.get("imminent", []):
        days = entry.get("days_until", 999)
        if days <= EARNINGS_AVOID_DAYS:
            imminent_tickers.add(entry.get("ticker", ""))

    if not imminent_tickers:
        return trades

    filtered = []
    for t in trades:
        if t["ticker"] in imminent_tickers:
            # Check if this is an earnings-driven signal
            sources = t.get("sources", [])
            is_earnings_signal = any(
                "earning" in s.lower() for s in sources
            )
            if is_earnings_signal:
                t = dict(t)
                t["earnings_note"] = f"Kept — earnings-driven signal (earnings in <={EARNINGS_AVOID_DAYS} days)"
                filtered.append(t)
                log.info(f"{t['ticker']}: kept despite imminent earnings (earnings-driven signal)")
            else:
                log.info(f"{t['ticker']}: SKIPPED — earnings in <={EARNINGS_AVOID_DAYS} days")
        else:
            filtered.append(t)

    return filtered


# ---------------------------------------------------------------------------
# Step 6: Spread Selection
# ---------------------------------------------------------------------------

def attach_spread_structures(
    trades: List[Dict[str, Any]], spread_recs: Dict
) -> List[Dict[str, Any]]:
    """Attach spread recommendation to each trade."""
    # Index spread recs by ticker
    spread_map = {}
    for rec in spread_recs.get("recommendations", []):
        ticker = rec.get("ticker", "")
        if ticker and rec.get("liquidity_ok"):
            spread_map[ticker] = rec

    for t in trades:
        ticker = t["ticker"]
        if ticker in spread_map:
            rec = spread_map[ticker]
            t["structure"] = {
                "type": rec["spread_type"],
                "category": rec["spread_category"],
                "legs": rec["legs"],
                "max_risk": rec["max_risk"],
                "max_reward": rec["max_reward"],
                "reward_risk": rec["reward_risk_ratio"],
                "breakeven": rec["breakeven"],
            }
        else:
            # No spread rec — suggest single leg as fallback with warning
            t["structure"] = {
                "type": "single_leg_option",
                "category": "debit",
                "warning": "No spread recommendation available — consider spreads manually",
            }

    return trades


# ---------------------------------------------------------------------------
# Portfolio Heat Calculation
# ---------------------------------------------------------------------------

def compute_portfolio_heat(
    trades: List[Dict[str, Any]], account_equity: float
) -> float:
    """Compute total portfolio heat (% of capital at risk)."""
    total_risk = sum(t.get("size", 0) for t in trades)
    return round(total_risk / account_equity * 100, 1) if account_equity > 0 else 0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_daily_plan() -> Dict[str, Any]:
    """Full pipeline: ingest signals -> output daily trade plan."""
    log.info("=" * 60)
    log.info("Unified Portfolio Brain — generating daily trade plan")

    # Load all inputs
    agentic = load_agentic_signals()
    flow = load_flow_screener()
    earnings = load_earnings_calendar()
    spread_recs = load_spread_recommendations()
    quality_universe = load_quality_universe()

    account_equity = agentic.get("account_equity", DEFAULT_ACCOUNT_EQUITY)
    vix_level = agentic.get("vix_level", 16.0)
    market_regime = agentic.get("market_regime", "neutral")

    log.info(f"Account: ${account_equity:.0f}  VIX: {vix_level:.1f}  Regime: {market_regime}")

    # Step 1: Aggregate signals
    aggregated = aggregate_signals(agentic, flow)
    log.info(f"Step 1: {len(aggregated)} tickers with signals")

    # Step 1b: Regime gate for flow signals
    # Flat-regime flow signals have Sharpe -2.0 vs bull +2.1 (flow research 2026-08-06).
    # Reduce flow signal confidence by 50% in flat/neutral regime.
    if market_regime in ("neutral", "flat"):
        for ticker, sig_data in aggregated.items():
            flow_sources = [s for s in sig_data["sources"] if s.startswith("flow_")]
            if not flow_sources:
                continue
            non_flow_sources = [s for s in sig_data["sources"] if not s.startswith("flow_")]
            if not non_flow_sources:
                # Pure flow signal (no non-flow confirming sources) — halve confidence
                sig_data["confidence"] *= 0.5
                log.info(f"{ticker}: flow confidence halved (regime={market_regime}, flow signals unreliable in flat markets)")
            else:
                # Mixed signal — reduce the flow contribution proportionally
                n_total = sig_data["n_sources"]
                n_flow = len(flow_sources)
                # Apply a blended reduction: flow portion gets 50% weight
                flow_ratio = n_flow / max(n_total, 1)
                reduction = 1.0 - (flow_ratio * 0.5)  # e.g., 2 of 4 sources are flow → 0.75x
                sig_data["confidence"] *= reduction
                log.info(f"{ticker}: flow confidence reduced {reduction:.0%} (regime={market_regime})")

    # Build trade list
    trades = []
    for ticker, sig_data in aggregated.items():
        if sig_data["confidence"] < MIN_CONFIDENCE:
            log.info(f"{ticker}: skipped (confidence {sig_data['confidence']:.0%} < {MIN_CONFIDENCE:.0%})")
            continue

        # Filter to quality universe if available
        if quality_universe and ticker not in quality_universe:
            # Also check sector ETFs
            if not ticker.startswith("XL") and ticker not in ("SPY", "QQQ"):
                log.info(f"{ticker}: skipped (not in quality universe)")
                continue

        # Step 2: Kelly sizing
        size = kelly_size(
            confidence=sig_data["confidence"],
            historical_sharpe=1.0,  # default if no per-strategy Sharpe
            account_equity=account_equity,
        )

        if size <= 0:
            continue

        trades.append({
            "ticker": ticker,
            "direction": sig_data["direction"],
            "confidence": sig_data["confidence"],
            "conviction": sig_data["conviction"],
            "n_sources": sig_data["n_sources"],
            "sources": sig_data["sources"],
            "size": size,
            "rationale": "; ".join(sig_data["reasons"][:3]),
            "exit_guidance": sig_data["exit_guidance"],
        })

    log.info(f"Step 2: {len(trades)} trades after Kelly sizing")

    # Step 3: Sector concentration check
    trades = check_sector_concentration(trades)
    log.info(f"Step 3: {len(trades)} trades after sector check")

    # Step 4: Regime overlay
    trades, regime = apply_regime_overlay(trades, vix_level)
    log.info(f"Step 4: {len(trades)} trades after regime overlay (regime={regime})")

    # Step 5: Earnings avoidance
    trades = filter_earnings(trades, earnings)
    log.info(f"Step 5: {len(trades)} trades after earnings filter")

    # Step 6: Spread selection
    trades = attach_spread_structures(trades, spread_recs)
    log.info(f"Step 6: {len(trades)} trades with spread structures")

    # Enforce portfolio heat cap
    heat = compute_portfolio_heat(trades, account_equity)
    if heat > MAX_PORTFOLIO_HEAT * 100:
        # Scale down proportionally
        scale = (MAX_PORTFOLIO_HEAT * 100) / heat
        for t in trades:
            t["size"] = round(t["size"] * scale, 2)
        heat = compute_portfolio_heat(trades, account_equity)
        log.info(f"Portfolio heat capped: scaled positions by {scale:.2f}")

    # Sort by conviction (high first), then confidence
    conviction_order = {"high": 0, "medium": 1, "low": 2}
    trades.sort(key=lambda t: (conviction_order.get(t["conviction"], 3), -t["confidence"]))

    # Build output
    plan = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(),
        "account_equity": account_equity,
        "regime": regime,
        "vix_level": vix_level,
        "portfolio_heat": heat,
        "n_trades": len(trades),
        "trades": trades,
        "signal_summary": {
            "total_signals_ingested": len(aggregated),
            "passed_confidence_filter": sum(1 for v in aggregated.values() if v["confidence"] >= MIN_CONFIDENCE),
            "passed_all_filters": len(trades),
            "high_conviction": sum(1 for t in trades if t["conviction"] == "high"),
            "medium_conviction": sum(1 for t in trades if t["conviction"] == "medium"),
        },
        "strategy_agreement": _build_agreement_matrix(aggregated),
    }

    # Save
    path = os.path.join(STATE_DIR, "daily_trade_plan.json")
    with open(path, "w") as f:
        json.dump(plan, f, indent=2)
    log.info(f"Saved daily trade plan to {path}")

    return plan


def _build_agreement_matrix(aggregated: Dict) -> Dict:
    """Build matrix showing which strategies agree on each ticker."""
    matrix = {}
    for ticker, data in aggregated.items():
        if data["n_sources"] >= 2:
            matrix[ticker] = {
                "direction": data["direction"],
                "n_sources": data["n_sources"],
                "sources": data["sources"],
                "confidence": data["confidence"],
            }
    return matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    plan = generate_daily_plan()

    print(f"\n{'='*60}")
    print(f"DAILY TRADE PLAN — {plan['date']}")
    print(f"{'='*60}")
    print(f"Account: ${plan['account_equity']:.0f}  VIX: {plan['vix_level']:.1f}  "
          f"Regime: {plan['regime']}  Heat: {plan['portfolio_heat']:.0f}%")
    print(f"Trades: {plan['n_trades']}  "
          f"(High conv: {plan['signal_summary']['high_conviction']}, "
          f"Med: {plan['signal_summary']['medium_conviction']})")

    if plan["trades"]:
        print(f"\n{'Ticker':<8} {'Dir':<6} {'Conv':<8} {'Conf':>6} {'Size':>7} {'Structure':<22}")
        print("-" * 65)
        for t in plan["trades"]:
            struct = t.get("structure", {}).get("type", "unknown")
            print(f"{t['ticker']:<8} {t['direction']:<6} {t['conviction']:<8} "
                  f"{t['confidence']:5.0%} ${t['size']:>6.0f} {struct:<22}")

    agreement = plan.get("strategy_agreement", {})
    if agreement:
        print(f"\nStrategy Agreement (2+ sources):")
        for ticker, info in sorted(agreement.items(), key=lambda x: -x[1]["n_sources"]):
            print(f"  {ticker}: {info['n_sources']} sources {info['direction']} "
                  f"({', '.join(info['sources'][:4])})")


if __name__ == "__main__":
    main()
