#!/usr/bin/env python3
"""
Daily Prediction Markets Scanner

Orchestrates all prediction market strategies into one daily report.
Runs every morning before market open (or on demand) and posts to Discord.

Usage:
    python daily_scanner.py                  # Full scan + Discord report
    python daily_scanner.py --report-only    # Reload last scan, post report
    python daily_scanner.py --quiet          # Scan without Discord post
    python daily_scanner.py --spx 5900 --hours 4.0  # Custom parameters
    python daily_scanner.py --no-engine      # Skip strategy engine (faster)
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Add parent dirs to path
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(THIS_DIR.parent))
if str(THIS_DIR.parents[1]) not in sys.path:
    sys.path.insert(0, str(THIS_DIR.parents[1]))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [daily_scanner] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("daily_scanner")

# ---------------------------------------------------------------------------
# Optional Discord notifier (Node.js bridge uses different channel system,
# but we can also use a direct webhook if DISCORD_WEBHOOK_URL is set)
# ---------------------------------------------------------------------------
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
PREDICTION_MARKETS_CHANNEL = "prediction-markets"


def _post_discord(message: str, quiet: bool = False) -> bool:
    """Post a message to Discord. Returns True on success."""
    if quiet:
        log.info(f"[DISCORD SUPPRESSED] {message[:80]}...")
        return True

    # Method 1: Direct webhook (preferred if configured)
    if DISCORD_WEBHOOK_URL:
        try:
            import urllib.request
            import urllib.parse
            payload = json.dumps({"content": message}).encode("utf-8")
            req = urllib.request.Request(
                DISCORD_WEBHOOK_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status in (200, 204):
                    return True
        except Exception as e:
            log.warning(f"Discord webhook failed: {e}")

    # Method 2: Write to a sidecar file that the Node.js bridge picks up
    # (teleclaude reads from this file on its next tick)
    sidecar = THIS_DIR.parents[1] / ".discord_pending.json"
    try:
        pending = []
        if sidecar.exists():
            try:
                with open(sidecar) as f:
                    pending = json.load(f)
            except Exception:
                pending = []
        pending.append({
            "channel": PREDICTION_MARKETS_CHANNEL,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        with open(sidecar, "w") as f:
            json.dump(pending, f, indent=2)
        log.info(f"Message queued to Discord sidecar ({len(message)} chars)")
        return True
    except Exception as e:
        log.warning(f"Sidecar write failed: {e}")

    # Method 3: stdout fallback (CI / no Discord)
    print(f"\n{'='*60}\nDISCORD MESSAGE:\n{message}\n{'='*60}\n")
    return False


# ---------------------------------------------------------------------------
# Vol signal reader (reads the signal file written by vol_signal_writer.py)
# ---------------------------------------------------------------------------
VOL_SIGNAL_FILE = DATA_DIR / "vol_signal.json"
MAX_SIGNAL_AGE_HOURS = 4.0  # Treat signal as stale if older than this


def read_vol_signal() -> dict:
    """Read the current vol signal from the signal file.

    Returns a dict with signal data, plus a 'status' key:
      'LIVE'    - fresh signal from the model
      'STALE'   - signal exists but is older than MAX_SIGNAL_AGE_HOURS
      'MISSING' - no signal file found
      'FALLBACK'- signal has source == FALLBACK_NO_DATA
    """
    if not VOL_SIGNAL_FILE.exists():
        return {
            "status": "MISSING",
            "annualized_vol": 0.18,
            "raw_prediction_pct": 18.0,
            "z_score": 0.0,
            "trailing_rvol_pct": 18.0,
            "model_ic": 0.0,
            "confidence": 0.1,
            "horizon": "30min",
            "source": "MISSING",
        }

    try:
        with open(VOL_SIGNAL_FILE) as f:
            signal = json.load(f)
    except Exception as e:
        log.warning(f"Could not read vol signal: {e}")
        return {
            "status": "MISSING",
            "annualized_vol": 0.18,
            "raw_prediction_pct": 18.0,
            "z_score": 0.0,
            "trailing_rvol_pct": 18.0,
            "model_ic": 0.0,
            "confidence": 0.1,
            "horizon": "30min",
            "source": "READ_ERROR",
        }

    # Check staleness
    ts_str = signal.get("timestamp")
    signal_age_hours = None
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_secs = (datetime.now(timezone.utc) - ts).total_seconds()
            signal_age_hours = age_secs / 3600.0
        except Exception:
            pass

    source = signal.get("source", "")
    if source == "FALLBACK_NO_DATA":
        signal["status"] = "FALLBACK"
    elif signal_age_hours is not None and signal_age_hours > MAX_SIGNAL_AGE_HOURS:
        signal["status"] = "STALE"
        signal["age_hours"] = round(signal_age_hours, 1)
    else:
        signal["status"] = "LIVE"
        if signal_age_hours is not None:
            signal["age_hours"] = round(signal_age_hours, 1)

    return signal


# ---------------------------------------------------------------------------
# Individual strategy runners (each wrapped for safe failure)
# ---------------------------------------------------------------------------

def run_fomc_scan() -> dict:
    """Run FOMC divergence scan. Returns result dict or error dict."""
    try:
        from fomc_tracker import FOMCTracker
        tracker = FOMCTracker()
        result = tracker.run_scan()
        return {
            "status": "OK",
            "result": result,
            "days_to_meeting": tracker.days_to_next_fomc(),
            "next_meeting": tracker.next_fomc_date(),
        }
    except Exception as e:
        log.error(f"FOMC scan failed: {e}")
        return {"status": "ERROR", "error": str(e)}


def run_bracket_scan(spx: float, hours: float, vol_signal: dict) -> dict:
    """Run SPX bracket scan using vol model. Returns result dict or error dict."""
    try:
        from scan_brackets import scan as bracket_scan

        # Use vol from signal if fresh, otherwise let VolModel auto-estimate
        vol = None
        vol_source = "auto"
        if vol_signal.get("status") == "LIVE":
            vol = vol_signal.get("annualized_vol")
            vol_source = f"model (IC={vol_signal.get('model_ic', 0):.3f})"

        result = bracket_scan(spx=spx, vol=vol, hours=hours, min_net_edge=0.005)
        result["vol_source"] = vol_source
        result["spx_used"] = spx
        result["hours_used"] = hours
        return {"status": "OK", "result": result}
    except Exception as e:
        log.error(f"Bracket scan failed: {e}")
        return {"status": "ERROR", "error": str(e)}


def run_strategy_engine(spx: float, hours: float, vol_signal: dict) -> dict:
    """Run full strategy engine scan_all(). Returns result dict or error dict."""
    try:
        from strategy_engine import StrategyEngine

        predicted_vol = 0.18
        if vol_signal.get("status") == "LIVE":
            predicted_vol = vol_signal.get("annualized_vol", 0.18)

        engine = StrategyEngine(mode="demo", use_llm=False)
        result = engine.scan_all(
            current_spx=spx,
            predicted_vol=predicted_vol,
            hours_to_close=hours,
            include_llm=False,
        )
        return {"status": "OK", "result": result}
    except Exception as e:
        log.error(f"Strategy engine failed: {e}")
        return {"status": "ERROR", "error": str(e)}


def run_polymarket_scan(min_net_edge: float = 0.02, limit: int = 200) -> dict:
    """
    Run Polymarket financial market scan.

    Steps:
    1. Fetch financial markets via scan_polymarket.get_financial_markets()
    2. Find combinatorial mispricings via PolymarketScanner.find_mispricings()
    3. Scan for edge opportunities via scan_polymarket.scan()

    Returns a structured result dict matching the style of other scan steps.
    """
    try:
        from scan_polymarket import scan as polymarket_scan, get_financial_markets
        from polymarket_client import PolymarketScanner

        scanner = PolymarketScanner()

        # Step A: Get financial markets (structured list for reporting)
        log.info("  Polymarket: fetching financial markets...")
        financial_markets = get_financial_markets(limit=limit)
        log.info(f"  Polymarket: {len(financial_markets)} financial markets found")

        # Step B: Find combinatorial mispricings — requires raw market dicts
        # Re-fetch raw markets for find_mispricings (needs outcomePrices etc.)
        log.info("  Polymarket: checking combinatorial mispricings...")
        raw_markets = scanner.get_active_markets(limit=limit)
        financial_raw = scanner.filter_financial(raw_markets)
        mispricings = scanner.find_mispricings(financial_raw)
        log.info(f"  Polymarket: {len(mispricings)} combinatorial mispricings found")

        # Step C: Scan for edge opportunities
        log.info("  Polymarket: scanning for edge opportunities...")
        opportunities = polymarket_scan(min_net_edge=min_net_edge, limit=limit)
        log.info(f"  Polymarket: {len(opportunities)} opportunities with edge >{min_net_edge}")

        return {
            "status": "OK",
            "result": {
                "n_financial_markets": len(financial_markets),
                "n_mispricings": len(mispricings),
                "n_opportunities": len(opportunities),
                "top_markets": financial_markets[:10],
                "mispricings": mispricings[:10],
                "opportunities": opportunities[:10],
                "min_net_edge_used": min_net_edge,
            },
        }
    except Exception as e:
        log.error(f"Polymarket scan failed: {e}")
        return {"status": "ERROR", "error": str(e)}


# ---------------------------------------------------------------------------
# Data persistence
# ---------------------------------------------------------------------------

def save_scan_results(scan_data: dict) -> Path:
    """Save scan results to dated JSON file. Returns saved path."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_path = DATA_DIR / f"daily_scan_{date_str}.json"
    with open(out_path, "w") as f:
        json.dump(scan_data, f, indent=2, default=str)
    log.info(f"Scan results saved to {out_path}")
    return out_path


def load_latest_scan() -> dict | None:
    """Load the most recent daily scan file."""
    scan_files = sorted(DATA_DIR.glob("daily_scan_*.json"))
    if not scan_files:
        return None
    try:
        with open(scan_files[-1]) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not load latest scan: {e}")
        return None


# ---------------------------------------------------------------------------
# Market efficiency check
# ---------------------------------------------------------------------------

def _load_historical_opportunity_slugs() -> dict:
    """
    Load all previously seen opportunity slugs/questions from past daily scans.

    Returns dict: {identifier: first_seen_date_str}
    """
    seen = {}
    scan_files = sorted(DATA_DIR.glob("daily_scan_*.json"))
    for sf in scan_files:
        date_str = sf.stem.replace("daily_scan_", "")
        try:
            with open(sf) as f:
                data = json.load(f)
        except Exception:
            continue

        # Collect identifiers from all opportunity sources
        for section_key in ("engine", "polymarket"):
            section = data.get(section_key, {})
            if section.get("status") != "OK":
                continue
            result = section.get("result", {})
            for opp_list_key in ("top_opportunities", "opportunities", "mispricings"):
                for opp in result.get(opp_list_key, []):
                    ident = (
                        opp.get("market_slug")
                        or opp.get("slug")
                        or opp.get("question", "")[:60]
                        or opp.get("group", "")[:60]
                    )
                    if ident and ident not in seen:
                        seen[ident] = date_str
    return seen


def annotate_market_efficiency(scan_data: dict) -> dict:
    """
    Annotate opportunities with a 'days_mispriced' field.

    Markets that have appeared as "mispriced" in previous scans are likely
    efficiently priced — the apparent edge is probably not real.

    Adds to each opportunity:
      - days_mispriced: int or None (how many days this opp has appeared)
      - efficiency_warning: str or None (human-readable warning)
    """
    historical = _load_historical_opportunity_slugs()
    today_str = datetime.now().strftime("%Y-%m-%d")

    annotated_count = 0
    warned_count = 0

    for section_key in ("engine", "polymarket"):
        section = scan_data.get(section_key, {})
        if section.get("status") != "OK":
            continue
        result = section.get("result", {})
        for opp_list_key in ("top_opportunities", "opportunities", "mispricings"):
            for opp in result.get(opp_list_key, []):
                ident = (
                    opp.get("market_slug")
                    or opp.get("slug")
                    or opp.get("question", "")[:60]
                    or opp.get("group", "")[:60]
                )
                if not ident:
                    continue

                first_seen = historical.get(ident)
                if first_seen and first_seen != today_str:
                    try:
                        first_dt = datetime.strptime(first_seen, "%Y-%m-%d")
                        days = (datetime.now() - first_dt).days
                    except ValueError:
                        days = None

                    opp["days_mispriced"] = days
                    annotated_count += 1

                    if days is not None and days >= 3:
                        opp["efficiency_warning"] = (
                            f"This 'mispricing' has appeared in scans for {days} days. "
                            f"Markets that stay mispriced this long are probably fairly priced — "
                            f"the edge may not be real."
                        )
                        warned_count += 1
                    elif days is not None and days >= 1:
                        opp["efficiency_warning"] = (
                            f"Seen in scans for {days} day(s). Monitor but be cautious."
                        )
                else:
                    opp["days_mispriced"] = 0

    if annotated_count > 0:
        log.info(
            f"Market efficiency check: {annotated_count} recurring opportunities, "
            f"{warned_count} with staleness warnings (>=3 days)"
        )

    return scan_data


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def _fmt_pct(val: float | None, decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def _fmt_edge(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.4f} ({val * 100:.2f}c)"


def _emoji_signal(signal_str: str) -> str:
    mapping = {
        "STRONG_DIVERGENCE": "🔴",
        "MODERATE_DIVERGENCE": "🟡",
        "MILD_DIVERGENCE": "🟢",
        "NO_SIGNAL": "⬜",
        "TOO_LATE": "⏰",
    }
    return mapping.get(signal_str, "❓")


def _vol_status_line(vol_signal: dict) -> str:
    status = vol_signal.get("status", "UNKNOWN")
    pred_pct = vol_signal.get("raw_prediction_pct")
    trailing_pct = vol_signal.get("trailing_rvol_pct")
    z = vol_signal.get("z_score")
    ic = vol_signal.get("model_ic")
    age = vol_signal.get("age_hours")
    horizon = vol_signal.get("horizon", "30min")

    if status == "MISSING":
        return "**Vol Model:** No signal file found — using fallback (18% annualized)"
    if status == "FALLBACK":
        return "**Vol Model:** FALLBACK mode — MBO data unavailable. Using 18% annualized."

    age_str = f", age={age:.1f}h" if age is not None else ""
    stale_tag = " [STALE]" if status == "STALE" else ""

    if pred_pct is not None and trailing_pct is not None:
        delta = pred_pct - trailing_pct
        delta_str = f"{delta:+.1f}%"
        return (
            f"**Vol Model ({horizon}{age_str}{stale_tag}):** "
            f"Predicted={pred_pct:.1f}% ann | Trailing={trailing_pct:.1f}% | "
            f"Delta={delta_str} | z={z:+.2f} | IC={ic:.3f}"
        )
    return f"**Vol Model:** status={status}{stale_tag}"


def _fomc_section(fomc_result: dict) -> list[str]:
    lines = []
    if fomc_result.get("status") == "ERROR":
        lines.append(f"**FOMC:** Error — {fomc_result.get('error', 'unknown')[:80]}")
        return lines

    result = fomc_result.get("result", {})
    days = fomc_result.get("days_to_meeting", "?")
    meeting = fomc_result.get("next_meeting", "?")
    div = result.get("divergence", {})
    rec = result.get("recommendation", {})

    signal_str = div.get("signal", "NO_SIGNAL")
    max_div = div.get("max_divergence", 0)
    rate_div_bps = div.get("rate_divergence_bps", 0)

    emoji = _emoji_signal(signal_str)
    lines.append(
        f"**FOMC ({meeting} — {days} days):** "
        f"{emoji} {signal_str} | Divergence: {max_div:.1%} | Rate gap: {rate_div_bps:.1f} bps"
    )

    outcomes = div.get("outcomes", {})
    outcome_parts = []
    for outcome, data in outcomes.items():
        if data.get("cme", 0) > 0.01 or data.get("kalshi", 0) > 0.01:
            gap = data.get("divergence", 0)
            outcome_parts.append(f"{outcome}: CME {data['cme']:.0%} vs Kalshi {data['kalshi']:.0%} ({gap:+.0%})")
    if outcome_parts:
        lines.append("  " + " | ".join(outcome_parts))

    # Flag FRED-derived data (no real market data)
    if div.get("both_fred_derived"):
        lines.append(
            "  ⚠ Both sources are FRED-derived — divergence is artificial, not from live market data"
        )

    action = rec.get("action", "NO_TRADE")
    if action != "NO_TRADE":
        trusted = rec.get("trusted_source", "?")
        direction = rec.get("direction", "?")
        size_pct = rec.get("recommended_size_pct", 0)
        lines.append(
            f"  → **{action}**: {direction} | Trust: {trusted} | Size: {size_pct:.1f}% of capital"
        )
    else:
        reason = rec.get("reason", "Markets in agreement")
        lines.append(f"  → {action}: {reason}")

    return lines


def _bracket_section(bracket_result: dict) -> list[str]:
    lines = []
    if bracket_result.get("status") == "ERROR":
        lines.append(f"**SPX Brackets:** Error — {bracket_result.get('error', 'unknown')[:80]}")
        return lines

    result = bracket_result.get("result", {})
    n_opps = result.get("n_opportunities", 0)
    vol_used = result.get("vol_used", 0)
    vol_info = result.get("vol_info", {})
    vol_src = result.get("vol_source", "?")
    spx = result.get("spx_used", "?")

    lines.append(
        f"**SPX Brackets (SPX={spx}, vol={vol_used:.1%}, source={vol_src}):** "
        f"{n_opps} opportunities with edge >0.5%"
    )

    opps = result.get("opportunities", [])
    for opp in opps[:5]:
        ticker = opp.get("ticker", "?")
        net_edge = opp.get("net_edge_after_fees", 0)
        action = opp.get("best_action", opp.get("action", "?"))
        fair = opp.get("fair_value", opp.get("model_price", None))
        market_p = opp.get("market_price", opp.get("yes_ask", None))
        parts = [f"  {ticker}"]
        if fair is not None and market_p is not None:
            parts.append(f"Fair={fair:.3f} vs Market={market_p:.3f}")
        parts.append(f"Edge={net_edge:.4f}")
        parts.append(f"Action={action}")
        lines.append(" | ".join(parts))

    return lines


def _engine_section(engine_result: dict, max_opps: int = 8) -> list[str]:
    lines = []
    if engine_result.get("status") == "ERROR":
        lines.append(f"**Strategy Engine:** Error — {engine_result.get('error', 'unknown')[:80]}")
        return lines

    result = engine_result.get("result", {})
    total = result.get("total_opportunities", 0)
    strategies = result.get("strategies", {})
    summary = result.get("summary", {})
    top_opps = result.get("top_opportunities", [])

    verdict = summary.get("verdict", "N/A")
    actionable = summary.get("actionable_count", 0)
    total_edge = summary.get("total_estimated_edge", 0)
    best_strat = summary.get("best_strategy", "?").replace("_", " ")

    lines.append(
        f"**Strategy Engine:** {verdict} | {total} total opps | "
        f"{actionable} actionable | Total edge: {total_edge:.3f} | "
        f"Best: {best_strat}"
    )

    # Strategy breakdown (non-zero only)
    active_strategies = {
        k: v for k, v in strategies.items()
        if v.get("count", 0) > 0 or "error" in v
    }
    if active_strategies:
        strat_parts = []
        for name, data in active_strategies.items():
            if "error" in data:
                strat_parts.append(f"{name.replace('_', ' ')}: ERR")
            else:
                strat_parts.append(f"{name.replace('_', ' ')}: {data['count']}")
        lines.append("  Strategies: " + " | ".join(strat_parts))

    # Top opportunities
    if top_opps:
        lines.append("  **Top Opportunities:**")
        for i, opp in enumerate(top_opps[:max_opps], 1):
            edge = opp.get("net_edge_after_fees", 0)
            if edge < 0.001:
                continue
            strategy = opp.get("strategy", "?").replace("_", " ")
            platform = opp.get("platform", "?")
            action = opp.get("action", "?")
            desc = (
                opp.get("question")
                or opp.get("ticker")
                or opp.get("group")
                or opp.get("event_type")
                or "?"
            )
            desc = str(desc)[:55]
            weighted_edge = edge * opp.get("weight", 0.1)
            lines.append(
                f"  {i}. [{platform}] {desc}  "
                f"| {strategy} | edge={edge:.4f} | w-edge={weighted_edge:.4f} | {action}"
            )

    return lines


def _polymarket_section(polymarket_result: dict, max_opps: int = 6) -> list[str]:
    lines = []
    if polymarket_result.get("status") == "ERROR":
        lines.append(
            f"**Polymarket:** Error — {polymarket_result.get('error', 'unknown')[:80]}"
        )
        return lines
    if polymarket_result.get("status") == "SKIPPED":
        return []

    result = polymarket_result.get("result", {})
    n_markets = result.get("n_financial_markets", 0)
    n_mispricings = result.get("n_mispricings", 0)
    n_opps = result.get("n_opportunities", 0)
    min_edge = result.get("min_net_edge_used", 0.02)

    lines.append(
        f"**Polymarket:** {n_markets} financial mkts | "
        f"{n_mispricings} combinatorial mispricings | "
        f"{n_opps} edge opps (>{min_edge * 100:.0f}c net)"
    )

    # Combinatorial mispricings
    mispricings = result.get("mispricings", [])
    if mispricings:
        lines.append("  **Combinatorial Mispricings:**")
        for mp in mispricings[:4]:
            group = mp.get("group", "?")
            direction = mp.get("direction", "?")
            prob_sum = mp.get("probability_sum", 0)
            edge = mp.get("edge_per_contract", 0)
            n = mp.get("markets", 0)
            lines.append(
                f"  • {group} | {direction} (sum={prob_sum:.3f}, {n} mkts) "
                f"| edge/contract={edge:.4f}"
            )

    # Edge opportunities
    opps = result.get("opportunities", [])
    if opps:
        lines.append("  **Top Edge Opportunities:**")
        for i, opp in enumerate(opps[:max_opps], 1):
            question = str(opp.get("question", "?"))[:55]
            market_p = opp.get("market_price", 0)
            fv = opp.get("fair_value", 0)
            net_edge = opp.get("net_edge", 0)
            action = opp.get("action", "?")
            end_date = opp.get("end_date", "?")
            vol = opp.get("volume", 0)
            lines.append(
                f"  {i}. {question}  "
                f"| Market={market_p:.3f} FV={fv:.3f} "
                f"| net_edge={net_edge:+.4f} | {action} | exp={end_date} "
                f"| vol=${vol / 1e6:.1f}M"
            )

    return lines


def _action_items(scan_data: dict) -> list[str]:
    """Generate prioritized action items from scan results."""
    items = []

    # FOMC action
    fomc = scan_data.get("fomc", {})
    if fomc.get("status") == "OK":
        rec = fomc.get("result", {}).get("recommendation", {})
        action = rec.get("action", "NO_TRADE")
        days = fomc.get("days_to_meeting", 99)
        if action == "TRADE":
            items.append(f"[HIGH] FOMC trade: {rec.get('direction', '?')} (D-{days})")
        elif action == "MONITOR" and days <= 7:
            items.append(f"[MED] FOMC: Monitor divergence, {days} days to meeting")

    # Vol signal action
    vol = scan_data.get("vol_signal", {})
    z = vol.get("z_score", 0)
    if vol.get("status") == "LIVE" and abs(z) > 1.5:
        direction = "HIGH" if z > 0 else "LOW"
        items.append(
            f"[MED] Vol signal elevated: z={z:+.2f} (vol is {direction}) — "
            f"check SPX brackets for directional mispricing"
        )

    # Bracket opportunities
    brackets = scan_data.get("brackets", {})
    if brackets.get("status") == "OK":
        n = brackets.get("result", {}).get("n_opportunities", 0)
        if n > 0:
            items.append(f"[MED] {n} SPX bracket(s) with edge >0.5% — review before open")

    # Engine top opportunities
    engine = scan_data.get("engine", {})
    if engine.get("status") == "OK":
        top = engine.get("result", {}).get("top_opportunities", [])
        high_edge = [o for o in top if o.get("net_edge_after_fees", 0) * o.get("weight", 0.1) > 0.02]
        if high_edge:
            best = high_edge[0]
            desc = (
                best.get("question") or best.get("ticker") or best.get("group") or "?"
            )
            items.append(
                f"[HIGH] Best weighted edge: {desc[:50]} "
                f"(edge={best.get('net_edge_after_fees', 0):.4f}, "
                f"strat={best.get('strategy', '?').replace('_', ' ')})"
            )

    # Polymarket opportunities
    polymarket = scan_data.get("polymarket", {})
    if polymarket.get("status") == "OK":
        poly_result = polymarket.get("result", {})
        n_opps = poly_result.get("n_opportunities", 0)
        n_mispricings = poly_result.get("n_mispricings", 0)
        top_opps = poly_result.get("opportunities", [])
        if n_mispricings > 0:
            best_mp = poly_result.get("mispricings", [{}])[0]
            items.append(
                f"[MED] Polymarket: {n_mispricings} combinatorial mispricing(s) — "
                f"best: {best_mp.get('group', '?')} "
                f"({best_mp.get('direction', '?')}, edge={best_mp.get('edge_per_contract', 0):.4f})"
            )
        if n_opps > 0 and top_opps:
            best_opp = top_opps[0]
            desc = str(best_opp.get("question", "?"))[:50]
            items.append(
                f"[MED] Polymarket: {n_opps} edge opp(s) — best: {desc} "
                f"(net_edge={best_opp.get('net_edge', 0):+.4f}, "
                f"{best_opp.get('action', '?')})"
            )

    # Efficiency warnings: flag stale mispricings
    stale_count = 0
    for section_key in ("engine", "polymarket"):
        section = scan_data.get(section_key, {})
        if section.get("status") != "OK":
            continue
        result = section.get("result", {})
        for opp_list_key in ("top_opportunities", "opportunities"):
            for opp in result.get(opp_list_key, []):
                days = opp.get("days_mispriced")
                if days is not None and days >= 3:
                    stale_count += 1
    if stale_count > 0:
        items.append(
            f"[WARN] {stale_count} opportunities have been 'mispriced' for 3+ days — "
            f"likely efficiently priced, not real edge"
        )

    if not items:
        items.append("[INFO] No high-priority action items today — markets appear fairly priced")

    return items


def format_discord_report(scan_data: dict) -> str:
    """Format the full daily scan as a Discord message."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M ET")
    now_utc = scan_data.get("scan_timestamp", datetime.now(timezone.utc).isoformat())[:19]

    lines = [
        f"**PREDICTION MARKETS DAILY SCAN — {date_str} {time_str}**",
        f"Scan time: {now_utc} UTC | SPX: {scan_data.get('spx', '?')} | Hours: {scan_data.get('hours', '?')}",
        "",
    ]

    # Vol model status
    vol_signal = scan_data.get("vol_signal", {})
    lines.append(_vol_status_line(vol_signal))
    lines.append("")

    # FOMC section
    fomc_data = scan_data.get("fomc", {})
    lines.extend(_fomc_section(fomc_data))
    lines.append("")

    # Bracket scan section
    bracket_data = scan_data.get("brackets", {})
    lines.extend(_bracket_section(bracket_data))
    lines.append("")

    # Strategy engine section
    engine_data = scan_data.get("engine", {})
    if engine_data:
        lines.extend(_engine_section(engine_data))
        lines.append("")

    # Polymarket section
    polymarket_data = scan_data.get("polymarket", {})
    if polymarket_data:
        poly_lines = _polymarket_section(polymarket_data)
        if poly_lines:
            lines.extend(poly_lines)
            lines.append("")

    # Action items
    lines.append("**ACTION ITEMS:**")
    actions = _action_items(scan_data)
    for item in actions:
        lines.append(f"  {item}")

    # Footer
    scan_duration = scan_data.get("scan_duration_seconds")
    dur_str = f" ({scan_duration:.1f}s)" if scan_duration else ""
    lines.append("")
    lines.append(f"*Scan complete{dur_str}. Results: {scan_data.get('output_file', 'data/daily_scan_*.json')}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def run_scan(
    spx: float = 5900.0,
    hours: float = 4.0,
    run_engine: bool = True,
    quiet: bool = False,
) -> dict:
    """
    Run the full daily scan and return the aggregated results dict.

    Steps:
    1. Read vol signal
    2. FOMC divergence check
    3. SPX bracket scan
    4. Full strategy engine scan (if run_engine=True)
    5. Polymarket financial market scan (combinatorial arb + edge opportunities)
    6. Save results to data/daily_scan_{date}.json
    7. Post Discord report (unless quiet=True)
    """
    t_start = datetime.now(timezone.utc)

    log.info("=" * 60)
    log.info(f"DAILY PREDICTION MARKETS SCAN — {t_start.strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"  SPX={spx}  hours={hours}  engine={run_engine}")
    log.info("=" * 60)

    scan_data = {
        "scan_timestamp": t_start.isoformat(),
        "spx": spx,
        "hours": hours,
        "vol_signal": {},
        "fomc": {},
        "brackets": {},
        "engine": {},
        "polymarket": {},
    }

    # Step 1: Vol signal
    log.info("Step 1/5: Reading vol signal...")
    vol_signal = read_vol_signal()
    scan_data["vol_signal"] = vol_signal
    log.info(
        f"  Vol signal: status={vol_signal['status']} | "
        f"vol={vol_signal.get('raw_prediction_pct', '?'):.1f}% | "
        f"z={vol_signal.get('z_score', 0):+.2f}"
        if isinstance(vol_signal.get('raw_prediction_pct'), (int, float))
        else f"  Vol signal: status={vol_signal['status']}"
    )

    # Step 2: FOMC divergence
    log.info("Step 2/5: Running FOMC scan...")
    fomc_result = run_fomc_scan()
    scan_data["fomc"] = fomc_result
    if fomc_result.get("status") == "OK":
        days = fomc_result.get("days_to_meeting", "?")
        meeting = fomc_result.get("next_meeting", "?")
        div_result = fomc_result.get("result", {}).get("divergence", {})
        signal = div_result.get("signal", "?")
        max_div = div_result.get("max_divergence", 0)
        log.info(f"  FOMC: {meeting} in {days} days | signal={signal} | divergence={max_div:.1%}")
    else:
        log.warning(f"  FOMC scan failed: {fomc_result.get('error', '?')}")

    # Step 3: Bracket scan
    log.info("Step 3/5: Running bracket scan...")
    bracket_result = run_bracket_scan(spx=spx, hours=hours, vol_signal=vol_signal)
    scan_data["brackets"] = bracket_result
    if bracket_result.get("status") == "OK":
        n = bracket_result.get("result", {}).get("n_opportunities", 0)
        vol_used = bracket_result.get("result", {}).get("vol_used", 0)
        log.info(f"  Brackets: {n} opportunities | vol_used={vol_used:.1%}")
    else:
        log.warning(f"  Bracket scan failed: {bracket_result.get('error', '?')}")

    # Step 4: Strategy engine (optional, slower)
    if run_engine:
        log.info("Step 4/5: Running strategy engine (all 8 strategies)...")
        engine_result = run_strategy_engine(spx=spx, hours=hours, vol_signal=vol_signal)
        scan_data["engine"] = engine_result
        if engine_result.get("status") == "OK":
            total = engine_result.get("result", {}).get("total_opportunities", 0)
            log.info(f"  Engine: {total} total opportunities")
        else:
            log.warning(f"  Engine failed: {engine_result.get('error', '?')}")
    else:
        log.info("Step 4/5: Strategy engine skipped (--no-engine)")
        scan_data["engine"] = {"status": "SKIPPED"}

    # Step 5: Polymarket scan
    log.info("Step 5/5: Running Polymarket scan...")
    polymarket_result = run_polymarket_scan()
    scan_data["polymarket"] = polymarket_result
    if polymarket_result.get("status") == "OK":
        poly_res = polymarket_result.get("result", {})
        n_opps = poly_res.get("n_opportunities", 0)
        n_mispricings = poly_res.get("n_mispricings", 0)
        n_markets = poly_res.get("n_financial_markets", 0)
        log.info(
            f"  Polymarket: {n_markets} financial mkts | "
            f"{n_mispricings} mispricings | {n_opps} edge opportunities"
        )
    else:
        log.warning(f"  Polymarket scan failed: {polymarket_result.get('error', '?')}")

    # Compute scan duration
    t_end = datetime.now(timezone.utc)
    scan_data["scan_duration_seconds"] = (t_end - t_start).total_seconds()

    # Market efficiency check: flag stale "mispricings" that have persisted across scans
    log.info("Running market efficiency check...")
    scan_data = annotate_market_efficiency(scan_data)

    # Save results
    out_path = save_scan_results(scan_data)
    scan_data["output_file"] = str(out_path)

    # Format and post Discord report
    report = format_discord_report(scan_data)
    log.info(f"Scan complete in {scan_data['scan_duration_seconds']:.1f}s")
    _post_discord(report, quiet=quiet)

    return scan_data


def run_report_only(quiet: bool = False) -> bool:
    """Load the latest scan results and re-post the Discord report."""
    data = load_latest_scan()
    if data is None:
        msg = "No scan results found. Run `python daily_scanner.py` first."
        log.warning(msg)
        _post_discord(msg, quiet=quiet)
        return False

    report = format_discord_report(data)
    scan_date = data.get("scan_timestamp", "?")[:10]
    log.info(f"Reposting report from scan {scan_date}")
    _post_discord(report, quiet=quiet)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Daily Prediction Markets Scanner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Load last scan results and post report (no new scan)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run scan but suppress Discord posting (print report to stdout)",
    )
    parser.add_argument(
        "--no-engine",
        action="store_true",
        help="Skip strategy engine (faster, only FOMC + brackets)",
    )
    parser.add_argument(
        "--spx",
        type=float,
        default=5900.0,
        help="Current SPX price",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=4.0,
        help="Hours to market close",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.report_only:
        success = run_report_only(quiet=args.quiet)
        sys.exit(0 if success else 1)

    try:
        scan_data = run_scan(
            spx=args.spx,
            hours=args.hours,
            run_engine=not args.no_engine,
            quiet=args.quiet,
        )

        if args.quiet:
            # Print formatted report to stdout when quiet mode
            report = format_discord_report(scan_data)
            # Use errors='replace' for Windows console compatibility with Unicode chars
            sys.stdout.buffer.write((report + "\n").encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()

        # Exit 0 = success, 1 = scan ran but with errors
        has_errors = any(
            scan_data.get(k, {}).get("status") == "ERROR"
            for k in ("fomc", "brackets", "engine", "polymarket")
        )
        sys.exit(1 if has_errors else 0)

    except KeyboardInterrupt:
        log.info("Scan interrupted by user")
        sys.exit(130)
    except Exception as e:
        log.exception(f"Fatal error during scan: {e}")
        _post_discord(
            f"**DAILY SCANNER FATAL ERROR**: {str(e)[:200]}\n```{traceback.format_exc()[-500:]}```",
            quiet=False,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
