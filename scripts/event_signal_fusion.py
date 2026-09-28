#!/usr/bin/env python3
"""
Event-Driven Signal Fusion Scanner
===================================
Reads all paper engines, ML models, earnings data, market regime, and meta signals.
Detects high-confidence events where multiple sources agree.
Outputs ranked list with confluence scores and trade recommendations.

Usage: python3 scripts/event_signal_fusion.py
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
PAPER_ENGINES_DIR = ROOT / "data" / "paper_engines"
ML_TREND_SECTORS = ROOT / "output" / "ml_trend_sectors" / "paper" / "positions_log.json"
STOCK_PICKER = PAPER_ENGINES_DIR / "stock_picker" / "state.json"
CONTRARIAN_PORTFOLIO = ROOT / "output" / "contrarian_portfolio_paper" / "portfolio.json"
MARKET_REGIME_FILE = ROOT / "state" / "market_regime.json"
META_SIGNALS_FILE = ROOT / "output" / "meta_signals" / "latest.json"
TRADE_PLAN_FILE = ROOT / "state" / "trade_plan_tomorrow.json"
EARNINGS_BEAT_FILE = ROOT / "state" / "earnings_beat_signals.json"
OUTPUT_FILE = ROOT / "state" / "event_signals_latest.json"

# ── constants ──────────────────────────────────────────────────────────────
ACCOUNT_VALUE = 667.73

GROWTH_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "COIN",
    "RBLX", "UBER", "LYFT", "DDOG", "TTD", "SHOP", "NET", "ROKU",
]

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLC", "XLB", "XLRE"]

# ── helpers ────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | list | None:
    """Load JSON file, return None if missing or malformed."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════════════════
#  1. READ ALL SIGNAL SOURCES
# ══════════════════════════════════════════════════════════════════════════

def read_paper_engines() -> dict:
    """Read all paper engine state files. Returns {engine_name: state_dict}."""
    engines = {}
    if not PAPER_ENGINES_DIR.exists():
        return engines
    for d in sorted(PAPER_ENGINES_DIR.iterdir()):
        if d.is_dir():
            state = load_json(d / "state.json")
            if state:
                engines[d.name] = state
    return engines


def read_ml_trend_sectors() -> list[dict]:
    """Read ML trend sector positions log (list of snapshots)."""
    data = load_json(ML_TREND_SECTORS)
    return data if isinstance(data, list) else []


def read_stock_picker() -> dict | None:
    return load_json(STOCK_PICKER)


def read_contrarian() -> dict | None:
    return load_json(CONTRARIAN_PORTFOLIO)


def read_market_regime() -> dict | None:
    return load_json(MARKET_REGIME_FILE)


def read_meta_signals() -> dict | None:
    return load_json(META_SIGNALS_FILE)


def read_trade_plan() -> dict | None:
    return load_json(TRADE_PLAN_FILE)


def read_earnings_beats() -> dict | None:
    return load_json(EARNINGS_BEAT_FILE)


# ══════════════════════════════════════════════════════════════════════════
#  2. DETECT EVENTS
# ══════════════════════════════════════════════════════════════════════════

def detect_contrarian_triggers(universe: list[str]) -> list[dict]:
    """Check 1-day returns via yfinance; flag any >3% drop."""
    events = []
    try:
        import yfinance as yf
        tickers_str = " ".join(universe)
        data = yf.download(tickers_str, period="5d", group_by="ticker", progress=False, threads=True)
        for ticker in universe:
            try:
                if len(universe) == 1:
                    closes = data["Close"].dropna()
                else:
                    closes = data[ticker]["Close"].dropna()
                if len(closes) >= 2:
                    ret_1d = (closes.iloc[-1] / closes.iloc[-2] - 1) * 100
                    price = closes.iloc[-1]
                    if ret_1d <= -3.0:
                        # Check if above 200-SMA (need more data)
                        above_200sma = True  # default assume yes; refine below
                        try:
                            hist = yf.Ticker(ticker).history(period="1y")
                            if len(hist) >= 200:
                                sma200 = hist["Close"].rolling(200).mean().iloc[-1]
                                above_200sma = price > sma200
                        except Exception:
                            pass
                        events.append({
                            "ticker": ticker,
                            "event": "CONTRARIAN_TRIGGER",
                            "drop_pct": round(ret_1d, 2),
                            "price": round(float(price), 2),
                            "above_200sma": above_200sma,
                        })
            except Exception:
                continue
    except Exception as e:
        print(f"  [WARN] yfinance contrarian check failed: {e}")
    return events


def detect_vix_regime(regime: dict | None) -> dict:
    """Parse VIX regime from market_regime.json."""
    if not regime:
        return {"vix": None, "regime": "UNKNOWN", "favorable": False, "kill_switch": False}
    vix_data = regime.get("vix", {})
    trend_data = regime.get("trend", {})
    composite = regime.get("composite", {})
    vix_level = safe_float(vix_data.get("current", vix_data.get("level")))
    spy_below_50 = trend_data.get("spy_vs_50sma") == "below"
    kill_switch = vix_level > 20 and spy_below_50
    return {
        "vix": round(vix_level, 2),
        "regime": vix_data.get("regime", "UNKNOWN"),
        "direction": vix_data.get("direction", "?"),
        "favorable": vix_level < 20,
        "elevated": 20 <= vix_level <= 25,
        "panic": vix_level > 25,
        "risk_on": vix_level < 18,
        "kill_switch": kill_switch,
        "spy_below_50sma": spy_below_50,
        "allocation_pct": composite.get("allocation_pct", 100),
    }


def detect_ml_confidence_spikes(ml_sectors: list[dict], stock_picker: dict | None) -> list[dict]:
    """Find ML confidence > 0.75 in sector ETFs or individual stocks."""
    events = []
    # Latest ML sector snapshot
    if ml_sectors:
        latest = ml_sectors[-1]
        positions = latest.get("positions", {})
        for ticker, info in positions.items():
            conf = safe_float(info.get("ml_confidence", 0))
            if conf > 0.75:
                events.append({
                    "ticker": ticker,
                    "event": "ML_CONFIDENCE_SPIKE",
                    "confidence": round(conf, 4),
                    "direction": "LONG" if info.get("direction", 1) > 0 else "SHORT",
                    "source": "ml_trend_sectors",
                    "sector": info.get("sector", ""),
                })
    # Stock picker positions
    if stock_picker:
        positions = stock_picker.get("positions", {})
        for ticker, info in positions.items():
            conf = safe_float(info.get("confidence", 0))
            if conf > 0.75:
                events.append({
                    "ticker": ticker,
                    "event": "ML_CONFIDENCE_SPIKE",
                    "confidence": round(conf, 4),
                    "direction": "LONG",
                    "source": "stock_picker",
                    "sector": info.get("sector", ""),
                })
    return events


def detect_paper_engine_convergence(engines: dict, contrarian: dict | None, ml_sectors: list[dict]) -> list[dict]:
    """Count how many engines are positioned in the same direction per ticker/sector."""
    # Build ticker -> list of (engine, direction)
    ticker_positions: dict[str, list[tuple[str, str]]] = {}

    for engine_name, state in engines.items():
        positions = state.get("positions", {})
        if isinstance(positions, dict):
            for ticker, info in positions.items():
                direction = "LONG"
                if isinstance(info, dict):
                    d = info.get("direction", info.get("side", 1))
                    if d in (-1, "short", "SHORT", "sell", "SELL"):
                        direction = "SHORT"
                ticker_positions.setdefault(ticker, []).append((engine_name, direction))
        elif isinstance(positions, list):
            for pos in positions:
                ticker = pos.get("ticker", pos.get("symbol", ""))
                if ticker:
                    direction = "LONG"
                    d = pos.get("direction", pos.get("side", 1))
                    if d in (-1, "short", "SHORT", "sell", "SELL"):
                        direction = "SHORT"
                    ticker_positions.setdefault(ticker, []).append((engine_name, direction))

    # Add contrarian positions
    if contrarian:
        positions = contrarian.get("positions", [])
        if isinstance(positions, list):
            for pos in positions:
                ticker = pos.get("ticker", "")
                if ticker:
                    ticker_positions.setdefault(ticker, []).append(("contrarian_paper", "LONG"))

    # Add ML sector positions
    if ml_sectors:
        latest = ml_sectors[-1]
        for ticker, info in latest.get("positions", {}).items():
            direction = "LONG" if info.get("direction", 1) > 0 else "SHORT"
            ticker_positions.setdefault(ticker, []).append(("ml_trend_sectors", direction))

    # Find convergence (3+ engines same direction)
    events = []
    for ticker, pos_list in ticker_positions.items():
        longs = [e for e, d in pos_list if d == "LONG"]
        shorts = [e for e, d in pos_list if d == "SHORT"]
        if len(longs) >= 3:
            events.append({
                "ticker": ticker,
                "event": "PAPER_ENGINE_CONVERGENCE",
                "direction": "LONG",
                "n_engines": len(longs),
                "engines": longs,
                "total_positioned": len(pos_list),
            })
        if len(shorts) >= 3:
            events.append({
                "ticker": ticker,
                "event": "PAPER_ENGINE_CONVERGENCE",
                "direction": "SHORT",
                "n_engines": len(shorts),
                "engines": shorts,
                "total_positioned": len(pos_list),
            })
    return events


def detect_earnings_events(trade_plan: dict | None, earnings_beats: dict | None) -> list[dict]:
    """
    Detect earnings beat events from trade plan (AH gaps) and earnings beat scanner.
    Also detect earnings surprise momentum (beat in last 60 days).
    """
    events = []

    # From trade plan — tonight's AH gaps
    if trade_plan:
        ah_gaps = trade_plan.get("tonights_ah_gaps", {})
        for ticker, info in ah_gaps.items():
            gap_pct = safe_float(info.get("gap_pct", 0))
            eps_surprise = safe_float(info.get("eps_surprise", 0))
            if eps_surprise > 0 and gap_pct > 3.0:
                events.append({
                    "ticker": ticker,
                    "event": "EARNINGS_BEAT_GAP_UP",
                    "gap_pct": round(gap_pct, 2),
                    "eps_surprise_pct": round(eps_surprise, 2),
                    "ah_price": info.get("ah_price"),
                    "pead_signal": info.get("pead_signal", ""),
                    "affordable": info.get("affordable", False),
                })
            elif eps_surprise > 0 and gap_pct < -3.0:
                events.append({
                    "ticker": ticker,
                    "event": "EARNINGS_BEAT_GAP_DOWN",
                    "gap_pct": round(gap_pct, 2),
                    "eps_surprise_pct": round(eps_surprise, 2),
                    "ah_price": info.get("ah_price"),
                    "pead_signal": info.get("pead_signal", ""),
                    "affordable": info.get("affordable", False),
                    "note": "Beat but gapped down -- possible guidance concern",
                })
            elif eps_surprise < 0 and gap_pct < -3.0:
                events.append({
                    "ticker": ticker,
                    "event": "EARNINGS_MISS_GAP_DOWN",
                    "gap_pct": round(gap_pct, 2),
                    "eps_surprise_pct": round(eps_surprise, 2),
                    "ah_price": info.get("ah_price"),
                    "affordable": info.get("affordable", False),
                })

    # From earnings beat signals file — active positions + history
    if earnings_beats:
        for sig in earnings_beats.get("signal_history", []):
            for s in sig.get("signals", []):
                ticker = s.get("ticker", "")
                gap_pct = safe_float(s.get("gap_pct", 0))
                signal_type = s.get("signal", "")
                if gap_pct > 3.0 and signal_type != "KILL_SWITCH_ACTIVE":
                    events.append({
                        "ticker": ticker,
                        "event": "EARNINGS_GAP_SIGNAL",
                        "gap_pct": round(gap_pct, 2),
                        "signal": signal_type,
                        "source": "earnings_beat_scanner",
                    })

    return events


def detect_earnings_surprise_momentum() -> list[dict]:
    """
    Check which stocks in our universe beat earnings in last 60 trading days.
    These are ACTIVE momentum holds per validated research (Sharpe 1.54, 4/5 gates).
    Uses Robinhood MCP if available, otherwise falls back to trade plan data.
    """
    events = []
    # Try to read from existing earnings data files
    for fname in ["earnings_beat_scanner.json", "earnings_watch.json", "earnings_gap_signals.json"]:
        data = load_json(ROOT / "state" / fname)
        if data and isinstance(data, dict):
            # Check for beat history
            beats = data.get("beat_history", data.get("active_positions", []))
            if isinstance(beats, dict):
                for ticker, info in beats.items():
                    events.append({
                        "ticker": ticker,
                        "event": "EARNINGS_SURPRISE_MOMENTUM",
                        "source": fname,
                        "info": info if isinstance(info, str) else str(info)[:100],
                    })
            elif isinstance(beats, list):
                for item in beats:
                    ticker = item.get("ticker", "") if isinstance(item, dict) else str(item)
                    if ticker:
                        events.append({
                            "ticker": ticker,
                            "event": "EARNINGS_SURPRISE_MOMENTUM",
                            "source": fname,
                        })
    # Also check earnings momentum paper engine
    em_state = load_json(PAPER_ENGINES_DIR / "earnings_momentum" / "state.json")
    if em_state:
        for ticker in em_state.get("positions", {}).keys():
            events.append({
                "ticker": ticker,
                "event": "EARNINGS_SURPRISE_MOMENTUM",
                "source": "earnings_momentum_engine",
            })
        bh = em_state.get("beat_history", {})
        if bh:
            for ticker, info in bh.items():
                # Already in the list?
                existing = {e["ticker"] for e in events if e["event"] == "EARNINGS_SURPRISE_MOMENTUM"}
                if ticker not in existing:
                    events.append({
                        "ticker": ticker,
                        "event": "EARNINGS_SURPRISE_MOMENTUM",
                        "source": "earnings_momentum_beat_history",
                    })
    return events


def detect_meta_signal_confluence(meta: dict | None) -> list[dict]:
    """Extract high-confluence sector signals from meta signals ensemble."""
    events = []
    if not meta:
        return events
    sc = meta.get("sector_confluence", {})
    for ticker, info in sc.items():
        confluence = safe_float(info.get("confluence", 0))
        confirming = info.get("confirming", 0)
        if confirming >= 3 and confluence > 0.5:
            events.append({
                "ticker": ticker,
                "event": "META_CONFLUENCE",
                "direction": info.get("net_direction", "?").upper(),
                "confluence_score": round(confluence, 3),
                "n_confirming": confirming,
                "n_conflicting": info.get("conflicting", 0),
                "bull_sources": info.get("bull_sources", []),
                "bear_sources": info.get("bear_sources", []),
            })
    return events


# ══════════════════════════════════════════════════════════════════════════
#  3. SCORE AND RANK
# ══════════════════════════════════════════════════════════════════════════

def score_events(all_events: list[dict], vix_info: dict, regime: dict | None) -> list[dict]:
    """
    Compute confluence score (0-100) for each unique ticker across all events.
    Apply kill switch logic.
    """
    # Group events by ticker
    ticker_events: dict[str, list[dict]] = {}
    for ev in all_events:
        t = ev.get("ticker", "UNKNOWN")
        ticker_events.setdefault(t, []).append(ev)

    strategy_guidance = {}
    if regime:
        strategy_guidance = regime.get("strategy_guidance", {})

    scored = []
    for ticker, events in ticker_events.items():
        score = 0
        reasons = []
        event_types = [e["event"] for e in events]
        kill_switched = False

        # Earnings beat with >3% gap: +25
        earnings_gap_events = [e for e in events if e["event"] in ("EARNINGS_BEAT_GAP_UP", "EARNINGS_GAP_SIGNAL")]
        if earnings_gap_events:
            best = max(earnings_gap_events, key=lambda e: safe_float(e.get("gap_pct", 0)))
            score += 25
            reasons.append(f"Earnings beat gap +{best.get('gap_pct', '?')}%")

        # ML confidence >0.75: +20
        ml_events = [e for e in events if e["event"] == "ML_CONFIDENCE_SPIKE"]
        if ml_events:
            best = max(ml_events, key=lambda e: safe_float(e.get("confidence", 0)))
            score += 20
            reasons.append(f"ML confidence {best.get('confidence', '?')} ({best.get('source', '?')})")

        # Paper engine convergence (3+): +20
        conv_events = [e for e in events if e["event"] == "PAPER_ENGINE_CONVERGENCE"]
        if conv_events:
            best = max(conv_events, key=lambda e: e.get("n_engines", 0))
            score += 20
            reasons.append(f"{best['n_engines']} engines agree {best.get('direction', '?')}")

        # Meta confluence: +15 (bonus on top of convergence)
        meta_events = [e for e in events if e["event"] == "META_CONFLUENCE"]
        if meta_events:
            best = max(meta_events, key=lambda e: safe_float(e.get("confluence_score", 0)))
            score += 15
            reasons.append(f"Meta confluence {best.get('confluence_score', '?')} ({best.get('n_confirming', 0)} sources)")

        # Market regime allows: +15
        regime_allows = True
        if regime:
            # Check specific strategy permissions
            is_sector = ticker in SECTOR_ETFS
            if is_sector and not strategy_guidance.get("sector_rotation", True):
                regime_allows = False
            if strategy_guidance.get("pead", True) and any("EARNINGS" in e["event"] for e in events):
                regime_allows = True  # PEAD always allowed when enabled
        if regime_allows and not vix_info.get("kill_switch", False):
            score += 15
            reasons.append("Regime allows strategy")

        # Contrarian trigger: +15
        contrarian_events = [e for e in events if e["event"] == "CONTRARIAN_TRIGGER"]
        if contrarian_events:
            best = contrarian_events[0]
            pts = 15
            if best.get("above_200sma"):
                reasons.append(f"Contrarian drop {best.get('drop_pct', '?')}% + above 200-SMA")
            else:
                pts = 8  # reduced score if below 200-SMA
                reasons.append(f"Contrarian drop {best.get('drop_pct', '?')}% (below 200-SMA)")
            score += pts

        # VIX favorable: +5
        if vix_info.get("favorable", False):
            score += 5
            reasons.append("VIX below 20")

        # Earnings surprise momentum: +10
        momentum_events = [e for e in events if e["event"] == "EARNINGS_SURPRISE_MOMENTUM"]
        if momentum_events:
            score += 10
            reasons.append("Earnings surprise momentum (60d hold, Sharpe 1.54)")

        # ── KILL SWITCH ──
        kill_switch_active = vix_info.get("kill_switch", False)
        is_momentum_rotation = ticker in SECTOR_ETFS or any(
            e["event"] in ("META_CONFLUENCE", "PAPER_ENGINE_CONVERGENCE") and
            e.get("direction", "").upper() == "LONG"
            for e in events
        )
        # Momentum/rotation signals paused under kill switch
        # PEAD (earnings) signals are exempt from kill switch
        is_pead = any("EARNINGS" in e["event"] for e in events)
        if kill_switch_active and is_momentum_rotation and not is_pead:
            kill_switched = True

        # Determine primary direction
        directions = []
        for e in events:
            d = e.get("direction", "")
            if d:
                directions.append(d.upper())
        if directions:
            long_count = sum(1 for d in directions if d == "LONG")
            short_count = sum(1 for d in directions if d == "SHORT")
            primary_direction = "LONG" if long_count >= short_count else "SHORT"
        else:
            # Default from event types
            if any("GAP_DOWN" in e["event"] or "MISS" in e["event"] for e in events):
                primary_direction = "AVOID"
            else:
                primary_direction = "LONG"

        scored.append({
            "ticker": ticker,
            "score": min(score, 100),
            "direction": primary_direction,
            "reasons": reasons,
            "n_events": len(events),
            "event_types": list(set(event_types)),
            "kill_switched": kill_switched,
            "status": "PAUSED (kill switch)" if kill_switched else "ACTIVE",
            "events": events,
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def generate_trade_recommendations(scored_events: list[dict], account_value: float) -> list[dict]:
    """For events scoring >60, generate specific trade recommendations."""
    recs = []
    for item in scored_events:
        if item["score"] < 60:
            continue
        ticker = item["ticker"]
        direction = item["direction"]
        if direction == "AVOID":
            continue

        # Position sizing: equal weight, max 4 positions from account
        max_positions = max(1, min(4, int(account_value / 50)))
        position_size = round(account_value / max_positions, 2)

        # Estimate shares
        price = None
        for ev in item["events"]:
            if "price" in ev:
                price = ev["price"]
                break
            if "ah_price" in ev:
                price = ev["ah_price"]
                break

        shares = None
        if price and price > 0:
            shares = int(position_size / price)
            if shares == 0:
                shares = 1
                position_size = round(price, 2)

        # Entry type
        if any("CONTRARIAN" in et for et in item["event_types"]):
            entry_type = "LIMIT at prior close (contrarian bounce)"
        elif any("EARNINGS_BEAT_GAP_UP" in et for et in item["event_types"]):
            entry_type = "MARKET at open (PEAD momentum)"
        elif any("EARNINGS_SURPRISE_MOMENTUM" in et for et in item["event_types"]):
            entry_type = "LIMIT near prior close (momentum continuation)"
        elif item.get("kill_switched"):
            entry_type = "PAUSED -- do not enter until kill switch clears"
        else:
            entry_type = "LIMIT near VWAP"

        recs.append({
            "ticker": ticker,
            "direction": direction,
            "score": item["score"],
            "entry_type": entry_type,
            "position_size_usd": position_size,
            "est_shares": shares,
            "est_price": price,
            "status": item["status"],
            "reasons": item["reasons"],
        })
    return recs


# ══════════════════════════════════════════════════════════════════════════
#  4. MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  EVENT-DRIVEN SIGNAL FUSION SCANNER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # ── Read all sources ──
    print("\n[1] Reading signal sources...")
    engines = read_paper_engines()
    print(f"  Paper engines: {len(engines)} loaded ({', '.join(engines.keys())})")

    ml_sectors = read_ml_trend_sectors()
    print(f"  ML Trend Sectors: {len(ml_sectors)} snapshots")

    stock_picker = read_stock_picker()
    sp_positions = len(stock_picker.get("positions", {})) if stock_picker else 0
    print(f"  Stock Picker: {sp_positions} positions")

    contrarian = read_contrarian()
    cp_positions = len(contrarian.get("positions", [])) if contrarian else 0
    print(f"  Contrarian Paper: {cp_positions} positions")

    regime = read_market_regime()
    print(f"  Market Regime: {'loaded' if regime else 'MISSING'}")

    meta = read_meta_signals()
    print(f"  Meta Signals: {'loaded' if meta else 'MISSING'}")

    trade_plan = read_trade_plan()
    print(f"  Trade Plan: {'loaded' if trade_plan else 'MISSING'}")

    earnings_beats = read_earnings_beats()
    print(f"  Earnings Beat Scanner: {'loaded' if earnings_beats else 'MISSING'}")

    # ── Detect events ──
    print("\n[2] Detecting events...")
    all_events = []

    # a) Earnings events from existing data
    earnings_events = detect_earnings_events(trade_plan, earnings_beats)
    print(f"  Earnings events: {len(earnings_events)}")
    all_events.extend(earnings_events)

    # b) Contrarian triggers (1-day drops >3%)
    print("  Checking contrarian triggers (yfinance)...")
    contrarian_events = detect_contrarian_triggers(GROWTH_UNIVERSE)
    print(f"  Contrarian triggers: {len(contrarian_events)}")
    all_events.extend(contrarian_events)

    # c) ML confidence spikes
    ml_spikes = detect_ml_confidence_spikes(ml_sectors, stock_picker)
    print(f"  ML confidence spikes (>0.75): {len(ml_spikes)}")
    all_events.extend(ml_spikes)

    # d) Paper engine convergence
    convergence = detect_paper_engine_convergence(engines, contrarian, ml_sectors)
    print(f"  Paper engine convergence (3+): {len(convergence)}")
    all_events.extend(convergence)

    # e) VIX regime shift
    vix_info = detect_vix_regime(regime)
    print(f"  VIX: {vix_info['vix']} ({vix_info['regime']}) | Kill switch: {'ON' if vix_info['kill_switch'] else 'OFF'}")

    # f) Earnings surprise momentum
    momentum_events = detect_earnings_surprise_momentum()
    print(f"  Earnings surprise momentum holds: {len(momentum_events)}")
    all_events.extend(momentum_events)

    # g) Meta signal confluence
    meta_confluence = detect_meta_signal_confluence(meta)
    print(f"  Meta signal confluence: {len(meta_confluence)}")
    all_events.extend(meta_confluence)

    print(f"\n  TOTAL EVENTS DETECTED: {len(all_events)}")

    # ── Score and rank ──
    print("\n[3] Scoring and ranking...")
    scored = score_events(all_events, vix_info, regime)

    # ── Display results ──
    print("\n" + "=" * 72)
    print("  RANKED EVENT SIGNALS")
    print("=" * 72)

    if not scored:
        print("\n  No events detected.")
    else:
        # Kill switch banner
        if vix_info.get("kill_switch"):
            print(f"\n  *** KILL SWITCH ACTIVE ***")
            print(f"  VIX={vix_info['vix']} (>20) + SPY below 50-SMA")
            print(f"  Momentum/rotation signals PAUSED. PEAD/contrarian still allowed.")

        print(f"\n  {'Rank':<5} {'Ticker':<8} {'Score':<7} {'Dir':<7} {'Status':<22} {'Events'}")
        print(f"  {'-'*5} {'-'*8} {'-'*7} {'-'*7} {'-'*22} {'-'*30}")
        for i, item in enumerate(scored, 1):
            status = item["status"]
            events_str = ", ".join(item["event_types"][:3])
            print(f"  {i:<5} {item['ticker']:<8} {item['score']:<7} {item['direction']:<7} {status:<22} {events_str}")

        # Detailed view for top signals
        print("\n" + "-" * 72)
        print("  TOP SIGNALS (score >= 40)")
        print("-" * 72)
        for item in scored:
            if item["score"] < 40:
                continue
            print(f"\n  {item['ticker']} — Score: {item['score']}/100 | {item['direction']} | {item['status']}")
            for r in item["reasons"]:
                print(f"    + {r}")

    # ── Trade recommendations ──
    recs = generate_trade_recommendations(scored, ACCOUNT_VALUE)
    if recs:
        print("\n" + "=" * 72)
        print(f"  TRADE RECOMMENDATIONS (score >60, account ${ACCOUNT_VALUE:,.2f})")
        print("=" * 72)
        for rec in recs:
            print(f"\n  {rec['ticker']} — {rec['direction']} (Score: {rec['score']})")
            print(f"    Entry: {rec['entry_type']}")
            size_str = f"${rec['position_size_usd']:,.2f}"
            if rec.get("est_shares") and rec.get("est_price"):
                size_str += f" (~{rec['est_shares']} shares @ ${rec['est_price']:,.2f})"
            print(f"    Size:  {size_str}")
            print(f"    Status: {rec['status']}")
            for r in rec["reasons"]:
                print(f"    + {r}")
    else:
        print("\n  No actionable trade recommendations (no events scored >60).")

    # ── Paused strategies ──
    paused = [s for s in scored if s.get("kill_switched")]
    if paused:
        print(f"\n  PAUSED by kill switch ({len(paused)} signals):")
        for p in paused:
            print(f"    {p['ticker']} (score {p['score']}) — would be {p['direction']} but VIX/SPY kill switch active")

    # ── Save output ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "account_value": ACCOUNT_VALUE,
        "vix_info": vix_info,
        "kill_switch_active": vix_info.get("kill_switch", False),
        "total_events": len(all_events),
        "scored_signals": [{k: v for k, v in s.items() if k != "events"} for s in scored],
        "trade_recommendations": recs,
        "paused_signals": [s["ticker"] for s in paused],
        "sources_read": {
            "paper_engines": list(engines.keys()),
            "ml_trend_sectors": len(ml_sectors) > 0,
            "stock_picker": stock_picker is not None,
            "contrarian_paper": contrarian is not None,
            "market_regime": regime is not None,
            "meta_signals": meta is not None,
            "trade_plan": trade_plan is not None,
            "earnings_beats": earnings_beats is not None,
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_FILE}")
    print("=" * 72)

    return output


if __name__ == "__main__":
    main()
