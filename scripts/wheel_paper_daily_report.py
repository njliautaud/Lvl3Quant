#!/usr/bin/env python3
"""
Wheel Paper Trading Daily Report
=================================
Reads all wheel_*_state/state.json files, fetches live prices via yfinance,
and produces a risk dashboard with distance-to-breach analysis.

Usage:  python3 scripts/wheel_paper_daily_report.py
"""

import json, os, sys, glob
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STATE_ROOT = Path("/home/jupiter/Lvl3Quant/live_trading_linux")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/wheel_paper_dashboard")
STARTING_CAPITAL = 100_000.0

# Engine display names — order matters for the summary table
ENGINE_META = {
    "wheel_bps_state":               {"name": "BPS Aggressive ($10w)", "type": "bps"},
    "wheel_bps_conservative_state":  {"name": "BPS Conservative",     "type": "bps"},
    "wheel_ic_state":                {"name": "Iron Condor",           "type": "ic"},
    "wheel_v4_state":                {"name": "V4 CSP Base",           "type": "csp"},
    "wheel_v5_state":                {"name": "V5 Income",             "type": "csp"},
    "wheel_diversified_state":       {"name": "Diversified",           "type": "csp"},
    "wheel_paper_balanced_state":    {"name": "Balanced",              "type": "csp"},
    "wheel_paper_state":             {"name": "SPY Base",              "type": "csp"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_state(state_dir: str) -> dict | None:
    fp = STATE_ROOT / state_dir / "state.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f)


def fetch_prices(tickers: list[str]) -> dict[str, float | None]:
    """Batch-fetch current prices via yfinance. Returns {ticker: price}."""
    if not tickers:
        return {}
    prices = {}
    # yfinance download is faster for batch
    try:
        data = yf.download(tickers, period="1d", progress=False, threads=True)
        if data.empty:
            raise ValueError("empty download")
        # data columns may be MultiIndex (ticker, field) or simple if 1 ticker
        for t in tickers:
            try:
                if len(tickers) == 1:
                    prices[t] = float(data["Close"].iloc[-1])
                else:
                    prices[t] = float(data["Close"][t].iloc[-1])
            except Exception:
                prices[t] = None
    except Exception:
        # Fallback: one-by-one
        for t in tickers:
            try:
                tk = yf.Ticker(t)
                h = tk.history(period="1d")
                prices[t] = float(h["Close"].iloc[-1]) if not h.empty else None
            except Exception:
                prices[t] = None
    return prices


def days_until(expiry_str: str) -> int:
    exp = datetime.fromisoformat(expiry_str)
    now = datetime.now()
    return max(0, (exp.date() - now.date()).days)


def risk_status(buffer_pct: float) -> str:
    if buffer_pct > 5.0:
        return "SAFE"
    elif buffer_pct >= 2.0:
        return "WATCH"
    else:
        return "DANGER"


# ---------------------------------------------------------------------------
# Position extractors — normalize each engine type to common format
# ---------------------------------------------------------------------------
def extract_bps_positions(state: dict) -> list[dict]:
    """BPS engine: spreads with short_strike / long_strike (put credit spreads)."""
    positions = []
    for s in state.get("spreads", []):
        positions.append({
            "ticker":       s["ticker"],
            "type":         "BPS",
            "short_strike": s["short_strike"],
            "long_strike":  s["long_strike"],
            "contracts":    s["contracts"],
            "premium":      s["premium_received"],
            "expiry":       s["expiry"],
            "dte":          days_until(s["expiry"]),
            "width":        s.get("spread_width", abs(s["short_strike"] - s["long_strike"])),
            "margin":       s.get("margin_held", 0),
            "breach_side":  "below",   # put spread breaches when price drops below short strike
        })
    return positions


def extract_ic_positions(state: dict) -> list[dict]:
    """Iron condor engine: spreads with put_short/put_long + call_short/call_long."""
    positions = []
    for s in state.get("spreads", []):
        # Two legs per IC — report both breach directions
        base = {
            "ticker":    s["ticker"],
            "contracts": s["contracts"],
            "premium":   s["premium_received"],
            "expiry":    s["expiry"],
            "dte":       days_until(s["expiry"]),
            "width":     s.get("spread_width", 10),
            "margin":    s.get("margin_held", 0),
        }
        # Put side
        positions.append({
            **base,
            "type":         "IC-Put",
            "short_strike": s["put_short"],
            "long_strike":  s["put_long"],
            "breach_side":  "below",
        })
        # Call side
        positions.append({
            **base,
            "type":         "IC-Call",
            "short_strike": s["call_short"],
            "long_strike":  s["call_long"],
            "breach_side":  "above",
        })
    return positions


def extract_csp_positions(state: dict) -> list[dict]:
    """CSP / wheel engines: positions with strike (short puts or covered calls).
    Handles two schemas:
      - v4/v5/diversified: ticker, expiry, entry_premium, margin_held
      - balanced/SPY base: underlying, expiration, entry_premium (no margin_held)
    """
    positions = []
    for p in state.get("positions", []):
        side = p.get("side", "short_put")
        ticker = p.get("ticker") or p.get("underlying", "???")
        expiry = p.get("expiry") or p.get("expiration", "")
        contracts = p.get("contracts", 1)
        premium_per = p.get("entry_premium", 0)
        margin = p.get("margin_held", premium_per * 100 * contracts * 0.2)  # fallback ~20% of notional
        positions.append({
            "ticker":       ticker,
            "type":         "CSP" if "put" in side else "CC",
            "short_strike": p["strike"],
            "long_strike":  0,  # naked put — no long leg
            "contracts":    contracts,
            "premium":      premium_per * 100 * contracts,
            "expiry":       expiry,
            "dte":          days_until(expiry) if expiry else 0,
            "width":        p["strike"],  # max risk = strike (naked)
            "margin":       margin,
            "breach_side":  "below" if "put" in side else "above",
        })
    return positions


EXTRACTORS = {
    "bps": extract_bps_positions,
    "ic":  extract_ic_positions,
    "csp": extract_csp_positions,
}


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------
def generate_report() -> str:
    now = datetime.now()
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"  WHEEL PAPER TRADING — DAILY RISK REPORT")
    lines.append(f"  Generated: {now.strftime('%Y-%m-%d %H:%M ET')}")
    lines.append(f"{'='*80}")
    lines.append("")

    # Collect all tickers across all engines
    all_engines = {}  # state_dir -> (state, positions)
    all_tickers = set()

    for state_dir, meta in ENGINE_META.items():
        state = load_state(state_dir)
        if state is None:
            all_engines[state_dir] = (None, [])
            continue
        extractor = EXTRACTORS[meta["type"]]
        positions = extractor(state)
        all_engines[state_dir] = (state, positions)
        for p in positions:
            all_tickers.add(p["ticker"])

    # Fetch prices
    lines.append(f"Fetching prices for {len(all_tickers)} tickers...")
    prices = fetch_prices(sorted(all_tickers))
    fetched = sum(1 for v in prices.values() if v is not None)
    lines.append(f"Got {fetched}/{len(all_tickers)} prices.\n")

    # -----------------------------------------------------------------------
    # Section 1: Summary table
    # -----------------------------------------------------------------------
    lines.append(f"{'='*80}")
    lines.append("  ENGINE COMPARISON")
    lines.append(f"{'='*80}")
    hdr = f"{'Engine':<24} {'NAV':>10} {'Return':>8} {'Realized':>10} {'Open':>5} {'Risk$':>10} {'Trades':>6} {'Days':>5}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for state_dir, meta in ENGINE_META.items():
        state, positions = all_engines[state_dir]
        if state is None:
            lines.append(f"{meta['name']:<24} {'(no state)':>10}")
            continue

        cash = state.get("cash", STARTING_CAPITAL)
        realized = state.get("realized_pnl", 0)
        trades = state.get("trade_count", 0)
        start = state.get("start_date", "")

        # Compute margin held / max risk
        total_margin = sum(p["margin"] for p in positions)
        n_open = len(positions)

        # NAV estimate: cash + realized is already in cash for most engines
        # For these engines, NAV = cash (cash includes realized, margins are deducted)
        nav = cash
        ret_pct = ((nav - STARTING_CAPITAL) / STARTING_CAPITAL) * 100

        days_active = 0
        if start:
            try:
                start_dt = datetime.fromisoformat(start)
                days_active = (now - start_dt).days
            except Exception:
                pass

        lines.append(
            f"{meta['name']:<24} ${nav:>9,.0f} {ret_pct:>+7.2f}% ${realized:>9,.2f} {n_open:>5} ${total_margin:>9,.0f} {trades:>6} {days_active:>5}"
        )

    lines.append("")

    # -----------------------------------------------------------------------
    # Section 2: Per-engine position details with breach analysis
    # -----------------------------------------------------------------------
    for state_dir, meta in ENGINE_META.items():
        state, positions = all_engines[state_dir]
        if state is None or not positions:
            continue

        lines.append(f"{'='*80}")
        lines.append(f"  {meta['name'].upper()} — POSITION DETAIL")
        lines.append(f"{'='*80}")

        # Sort: DANGER first, then WATCH, then SAFE; within each group sort by DTE
        def sort_key(p):
            price = prices.get(p["ticker"])
            if price is None:
                return (0, p["dte"])
            if p["breach_side"] == "below":
                buf = ((price - p["short_strike"]) / price) * 100
            else:
                buf = ((p["short_strike"] - price) / price) * 100
            status = risk_status(buf)
            rank = {"DANGER": 0, "WATCH": 1, "SAFE": 2}[status]
            return (rank, p["dte"])

        positions_sorted = sorted(positions, key=sort_key)

        if meta["type"] in ("bps", "ic"):
            hdr2 = f"{'Ticker':<7} {'Type':<8} {'Short':>7} {'Long':>7} {'Ct':>3} {'Price':>8} {'Dist':>7} {'Buf%':>6} {'DTE':>4} {'Status':<7} {'Prem$':>8}"
        else:
            hdr2 = f"{'Ticker':<7} {'Type':<5} {'Strike':>8} {'Ct':>3} {'Price':>8} {'Dist':>7} {'Buf%':>6} {'DTE':>4} {'Status':<7} {'Prem$':>8}"

        lines.append(hdr2)
        lines.append("-" * len(hdr2))

        danger_count = 0
        watch_count = 0

        for p in positions_sorted:
            price = prices.get(p["ticker"])
            if price is None:
                price_str = "N/A"
                dist_str = "N/A"
                buf_str = "N/A"
                status = "???"
            else:
                price_str = f"${price:>7.2f}"
                if p["breach_side"] == "below":
                    distance = price - p["short_strike"]
                    buf_pct = (distance / price) * 100
                else:
                    distance = p["short_strike"] - price
                    buf_pct = (distance / price) * 100
                dist_str = f"{distance:>+7.2f}"
                buf_str = f"{buf_pct:>5.1f}%"
                status = risk_status(buf_pct)

                if status == "DANGER":
                    danger_count += 1
                elif status == "WATCH":
                    watch_count += 1

            prem_str = f"${p['premium']:>7.2f}"

            if meta["type"] in ("bps", "ic"):
                lines.append(
                    f"{p['ticker']:<7} {p['type']:<8} {p['short_strike']:>7.1f} {p['long_strike']:>7.1f} {p['contracts']:>3} "
                    f"{price_str} {dist_str} {buf_str} {p['dte']:>4} {status:<7} {prem_str}"
                )
            else:
                lines.append(
                    f"{p['ticker']:<7} {p['type']:<5} {p['short_strike']:>8.1f} {p['contracts']:>3} "
                    f"{price_str} {dist_str} {buf_str} {p['dte']:>4} {status:<7} {prem_str}"
                )

        # Per-engine summary
        total_prem = sum(p["premium"] for p in positions)
        total_margin = sum(p["margin"] for p in positions)
        lines.append("")
        lines.append(f"  Positions: {len(positions)}  |  Premium: ${total_prem:,.2f}  |  Margin held: ${total_margin:,.0f}")
        if danger_count:
            lines.append(f"  *** {danger_count} DANGER position(s) — price within 2% of short strike ***")
        if watch_count:
            lines.append(f"  * {watch_count} WATCH position(s) — price within 2-5% of short strike *")
        lines.append("")

    # -----------------------------------------------------------------------
    # Section 3: Expiry calendar (next 10 days)
    # -----------------------------------------------------------------------
    lines.append(f"{'='*80}")
    lines.append("  EXPIRY CALENDAR (next 10 days)")
    lines.append(f"{'='*80}")

    expiry_map: dict[str, list[tuple[str, str, str]]] = {}
    for state_dir, meta in ENGINE_META.items():
        _, positions = all_engines[state_dir]
        for p in positions:
            if p["dte"] <= 10:
                exp_date = datetime.fromisoformat(p["expiry"]).strftime("%Y-%m-%d")
                expiry_map.setdefault(exp_date, []).append(
                    (meta["name"], p["ticker"], p["type"])
                )

    if expiry_map:
        for date_str in sorted(expiry_map.keys()):
            items = expiry_map[date_str]
            dte = (datetime.strptime(date_str, "%Y-%m-%d").date() - now.date()).days
            label = "TODAY" if dte == 0 else f"TOMORROW" if dte == 1 else f"in {dte}d"
            lines.append(f"\n  {date_str} ({label}) — {len(items)} positions expiring:")
            for engine, ticker, ptype in sorted(items):
                lines.append(f"    {engine:<24} {ticker:<7} ({ptype})")
    else:
        lines.append("  No positions expiring within 10 days.")

    lines.append("")
    lines.append(f"{'='*80}")
    lines.append(f"  END OF REPORT")
    lines.append(f"{'='*80}")

    return "\n".join(lines)


def main():
    report = generate_report()

    # Print to stdout
    print(report)

    # Save to file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"daily_report_{today}.txt"
    with open(out_path, "w") as f:
        f.write(report)

    # Also update latest symlink
    latest = OUTPUT_DIR / "daily_report_latest.txt"
    with open(latest, "w") as f:
        f.write(report)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
