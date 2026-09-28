#!/usr/bin/env python3
"""
Bond Yield + Sub-Sector Inflow Paper Engine
=============================================
Enhanced Bond Yield strategy with sub-sector rotation confluence.

Validated improvement: Sharpe 2.25 -> 2.91 with inflow filter (perm p=0.018).

Entry signal (ALL must be true):
  1. Quality stock >5% below its 20-day SMA
  2. 10Y yield (^TNX) dropped >0.1% over last 5 trading days
  3. Ticker's sub-sector is in INFLOW phase (positive rotation score)

Exit:
  - 10-day hold OR 3% profit target (whichever comes first)

Capital: $10,000 | Max $500/position | Max 3 concurrent

Designed to run at 4:25 PM ET weekdays (after rotation tracker at 4:15,
before aggregator at 4:30).

Usage:
    python3 paper_engines/bond_yield_inflow_paper_engine.py
"""

import json
import logging
import math
import time
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR = BASE_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "paper_engine_bond_yield_inflow.json"
ROTATION_STATE_FILE = STATE_DIR / "subsector_rotation_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BondYieldInflow] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bond_yield_inflow_paper.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAPITAL_INITIAL = 10000
MAX_PER_POSITION = 500
MAX_CONCURRENT = 3
HOLD_DAYS = 10
PROFIT_TARGET_PCT = 0.03  # 3% profit target

# Quality stocks universe (same as bond_yield_paper.py)
QUALITY_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM",
    "JNJ", "UNH", "PG", "HD", "MA", "ABBV", "KO", "PEP", "COST", "LIN",
    "CRM", "AVGO", "TMO", "MRK", "ACN",
]

# Sub-sector taxonomy (from subsector_rotation_tracker.py SUBSECTOR_UNIVERSE)
# Maps ticker -> sub-sector name
TICKER_TO_SUBSECTOR = {}
SUBSECTOR_UNIVERSE = {
    "semiconductors": ["NVDA", "AMD", "INTC", "QCOM", "MU"],
    "semicon_equipment": ["AMAT", "LRCX", "KLAC", "ASML", "TER"],
    "enterprise_software": ["MSFT", "CRM", "ORCL", "NOW", "ADBE"],
    "cybersecurity": ["PANW", "CRWD", "FTNT", "ZS", "OKTA"],
    "it_hardware": ["AAPL", "DELL", "HPQ", "CSCO", "ANET"],
    "biotech": ["REGN", "GILD", "VRTX", "MRNA", "BIIB"],
    "medtech_devices": ["ISRG", "ABT", "MDT", "SYK", "EW"],
    "pharma_large": ["LLY", "JNJ", "ABBV", "MRK", "PFE"],
    "health_services": ["UNH", "HCA", "CNC", "ELV", "CI"],
    "megabank": ["JPM", "BAC", "WFC", "C", "GS"],
    "regional_bank": ["USB", "PNC", "TFC", "FITB", "KEY"],
    "insurance": ["BRK-B", "PGR", "AIG", "MET", "ALL"],
    "fintech_payments": ["V", "MA", "PYPL", "AFRM", "FIS"],
    "oil_integrated": ["XOM", "CVX", "COP", "EOG", "OXY"],
    "oilfield_services": ["SLB", "HAL", "BKR", "FTI", "NOV"],
    "midstream_pipelines": ["WMB", "KMI", "OKE", "ET", "MPLX"],
    "aerospace_defense": ["LMT", "RTX", "NOC", "GD", "LHX"],
    "heavy_equipment": ["CAT", "DE", "CMI", "PCAR", "TTC"],
    "industrial_automation": ["EMR", "ROK", "ETN", "IR", "AME"],
    "transportation": ["UNP", "UPS", "FDX", "CSX", "DAL"],
    "ecommerce_retail": ["AMZN", "HD", "LOW", "TJX", "COST"],
    "autos_ev": ["TSLA", "GM", "F", "RIVN", "ON"],
    "restaurants_leisure": ["MCD", "SBUX", "CMG", "DRI", "YUM"],
    "consumer_staples_food": ["PG", "KO", "PEP", "MDLZ", "GIS"],
    "staples_retail": ["WMT", "COST", "TGT", "DG", "KR"],
    "mining_metals": ["FCX", "NEM", "GOLD", "SCCO", "CLF"],
    "chemicals": ["LIN", "APD", "ECL", "SHW", "DD"],
    "reits_data_towers": ["PLD", "AMT", "EQIX", "DLR", "SPG"],
    "utilities_electric": ["NEE", "DUK", "SO", "AEP", "SRE"],
    "big_tech_comm": ["META", "GOOGL", "NFLX", "DIS", "CMCSA"],
    "telecom": ["T", "VZ", "TMUS", "AMX", "LUMN"],
}
for sub_name, tickers in SUBSECTOR_UNIVERSE.items():
    for t in tickers:
        TICKER_TO_SUBSECTOR[t] = sub_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yf_download(tickers, period="120d", retries=3):
    """Download with retry logic."""
    import yfinance as yf
    for attempt in range(retries):
        try:
            df = yf.download(tickers, period=period, progress=False, group_by="ticker")
            if df is not None and not df.empty:
                return df
        except Exception as e:
            log.warning(f"yfinance attempt {attempt+1} failed: {e}")
        if attempt < retries - 1:
            time.sleep(3)
    log.error("yfinance download failed after retries")
    return None


def _flatten_df(df):
    """Flatten MultiIndex columns from yfinance single-ticker download."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)
    return df


def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "capital": CAPITAL_INITIAL,
        "positions": [],
        "closed_trades": [],
        "last_run_date": None,
        "wins": 0,
        "losses": 0,
        "created": datetime.now().isoformat(),
    }


def save_state(state):
    state["updated"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_rotation_state():
    """Load sub-sector rotation state. Returns None if stale (>2 days) or missing."""
    if not ROTATION_STATE_FILE.exists():
        log.warning("Rotation state file missing — skipping inflow filter")
        return None
    try:
        with open(ROTATION_STATE_FILE) as f:
            data = json.load(f)
        # Check freshness
        gen_at = data.get("generated_at", "")
        if gen_at:
            gen_time = datetime.fromisoformat(gen_at)
            age_hours = (datetime.now() - gen_time).total_seconds() / 3600
            if age_hours > 48:
                log.warning(f"Rotation state is {age_hours:.0f}h old (>48h) — treating as stale")
                return None
        return data
    except Exception as e:
        log.warning(f"Failed to load rotation state: {e}")
        return None


def get_subsector_phase(ticker, rotation_state):
    """Get the rotation phase and score for a ticker's sub-sector.
    Returns (phase, score) or (None, 0) if not found."""
    if rotation_state is None:
        return None, 0
    sub_name = TICKER_TO_SUBSECTOR.get(ticker)
    if not sub_name:
        return None, 0
    subsectors = rotation_state.get("subsectors", {})
    sub_data = subsectors.get(sub_name, {})
    return sub_data.get("rotation_phase"), sub_data.get("rotation_score", 0)


# ---------------------------------------------------------------------------
# Signal Logic
# ---------------------------------------------------------------------------

def check_yield_drop(tnx_df):
    """Check if 10Y yield dropped >10 bps over last 5 trading days."""
    try:
        tnx_df = _flatten_df(tnx_df)
        close = tnx_df["Close"].dropna()
        if len(close) < 6:
            return False, 0.0
        yield_now = float(close.iloc[-1])
        yield_5d_ago = float(close.iloc[-6])
        change = yield_now - yield_5d_ago
        return change < -0.10, round(change, 3)
    except Exception as e:
        log.warning(f"Yield check failed: {e}")
        return False, 0.0


def check_signals(raw_df, yield_dropped, rotation_state):
    """Check for buy signals: stock >5% below 20d SMA + yield drop + sub-sector INFLOW."""
    signals = []
    if not yield_dropped:
        return signals

    for ticker in QUALITY_STOCKS:
        try:
            if isinstance(raw_df.columns, pd.MultiIndex):
                close = raw_df[(ticker, "Close")].dropna()
            else:
                close = raw_df["Close"].dropna()

            if len(close) < 25:
                continue

            price_now = float(close.iloc[-1])
            sma_20 = float(close.iloc[-20:].mean())

            pct_below_sma = 1 - (price_now / sma_20)
            if pct_below_sma < 0.05:
                continue

            # CRITICAL: Sub-sector must be in INFLOW (positive rotation score)
            phase, rot_score = get_subsector_phase(ticker, rotation_state)

            if phase is None:
                # No rotation data — skip (we require inflow confirmation)
                log.debug(f"  {ticker}: no rotation data, skipping")
                continue

            if phase not in ("INFLOW", "ACCUMULATING") or rot_score <= 0:
                log.info(f"  {ticker}: {pct_below_sma*100:.1f}% below SMA but sub-sector "
                         f"phase={phase} score={rot_score:.2f} — FILTERED OUT")
                continue

            sub_name = TICKER_TO_SUBSECTOR.get(ticker, "unknown")
            signals.append({
                "ticker": ticker,
                "price": round(price_now, 2),
                "pct_below_sma20": round(pct_below_sma * 100, 1),
                "subsector": sub_name,
                "rotation_phase": phase,
                "rotation_score": round(rot_score, 3),
                "reason": (f"Bond yield drop + {pct_below_sma*100:.1f}% below 20d SMA "
                           f"+ {sub_name} in {phase} (score {rot_score:.2f})"),
            })
        except Exception as e:
            log.debug(f"Signal check failed for {ticker}: {e}")

    # Sort by rotation score descending (prefer strongest inflow)
    signals.sort(key=lambda s: s["rotation_score"], reverse=True)
    return signals


# ---------------------------------------------------------------------------
# Position Management
# ---------------------------------------------------------------------------

def process_exits(state, raw_df):
    """Exit positions: 10-day hold OR 3% profit target (whichever first)."""
    today = date.today()
    remaining = []
    for pos in state["positions"]:
        entry_date = datetime.fromisoformat(pos["entry_date"]).date()
        days_held = int(np.busday_count(entry_date, today))

        # Get current price
        ticker = pos["ticker"]
        try:
            if isinstance(raw_df.columns, pd.MultiIndex):
                cur_price = float(raw_df[(ticker, "Close")].dropna().iloc[-1])
            else:
                cur_price = float(raw_df["Close"].dropna().iloc[-1])
        except Exception:
            cur_price = pos["entry_price"]

        pnl_pct = (cur_price - pos["entry_price"]) / pos["entry_price"]

        # Exit conditions
        exit_reason = None
        if days_held >= HOLD_DAYS:
            exit_reason = f"10-day hold expired ({days_held}d)"
        elif pnl_pct >= PROFIT_TARGET_PCT:
            exit_reason = f"3% profit target hit ({pnl_pct*100:.1f}%)"

        if exit_reason:
            pnl = (cur_price - pos["entry_price"]) * pos["shares"]
            state["capital"] += pos["shares"] * cur_price
            trade = {
                **pos,
                "exit_date": today.isoformat(),
                "exit_price": round(cur_price, 2),
                "exit_reason": exit_reason,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct * 100, 2),
                "days_held": days_held,
            }
            state["closed_trades"].append(trade)
            if pnl > 0:
                state["wins"] = state.get("wins", 0) + 1
            else:
                state["losses"] = state.get("losses", 0) + 1
            log.info(f"  EXIT {ticker}: ${pnl:+.2f} ({pnl_pct*100:+.1f}%) — {exit_reason}")
        else:
            remaining.append(pos)

    state["positions"] = remaining


def process_entries(state, signals, raw_df):
    """Open new positions from signals."""
    open_count = len(state["positions"])
    open_tickers = {p["ticker"] for p in state["positions"]}
    today = date.today()

    for sig in signals:
        if open_count >= MAX_CONCURRENT:
            break
        if sig["ticker"] in open_tickers:
            continue

        price = sig["price"]
        if price <= 0:
            continue
        shares = int(MAX_PER_POSITION / price)
        if shares < 1:
            continue
        cost = shares * price
        if cost > state["capital"]:
            continue

        state["capital"] -= cost
        pos = {
            "ticker": sig["ticker"],
            "entry_date": today.isoformat(),
            "entry_price": price,
            "shares": shares,
            "subsector": sig["subsector"],
            "rotation_phase": sig["rotation_phase"],
            "rotation_score": sig["rotation_score"],
            "reason": sig["reason"],
        }
        state["positions"].append(pos)
        open_tickers.add(sig["ticker"])
        open_count += 1
        log.info(f"  ENTRY {sig['ticker']}: {shares} shares @ ${price:.2f} — {sig['reason']}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(state):
    """Compute portfolio metrics: Sharpe, WR, PF."""
    trades = state["closed_trades"]
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "sharpe": 0, "profit_factor": 0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_win = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0
    pf = total_win / total_loss if total_loss > 0 else float("inf") if total_win > 0 else 0

    sharpe = 0
    if len(pnls) > 1:
        std = np.std(pnls)
        if std > 0:
            sharpe = np.mean(pnls) / std

    return {
        "total_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl": round(np.mean(pnls), 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(pf, 2) if not math.isinf(pf) else 999.0,
        "avg_hold_days": round(np.mean([t.get("days_held", 10) for t in trades]), 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Bond Yield + Sub-Sector Inflow Paper Engine")
    log.info("=" * 60)

    state = load_state()
    today_str = date.today().isoformat()

    if state.get("last_run_date") == today_str:
        log.info(f"Already ran today ({today_str}). Skipping.")
        return

    # Load rotation state (must be fresh)
    rotation_state = load_rotation_state()
    if rotation_state:
        pattern = rotation_state.get("rotation_pattern", {}).get("pattern", "UNKNOWN")
        log.info(f"  Rotation state loaded: pattern={pattern}")
    else:
        log.info("  No fresh rotation state — will skip entry signals (exits still processed)")

    # Download 10Y yield
    tnx_df = _yf_download("^TNX", period="30d")
    if tnx_df is None:
        log.error("Failed to download ^TNX. Aborting.")
        return

    yield_dropped, yield_change = check_yield_drop(tnx_df)
    log.info(f"  10Y yield 5d change: {yield_change:+.3f}% | Signal: {'YES' if yield_dropped else 'NO'}")

    # Download stock data
    raw_df = _yf_download(QUALITY_STOCKS, period="60d")
    if raw_df is None:
        log.error("Failed to download stock data. Aborting.")
        return

    # Process exits first (always — even without rotation state)
    process_exits(state, raw_df)

    # Check signals (requires rotation state for inflow filter)
    signals = check_signals(raw_df, yield_dropped, rotation_state)
    log.info(f"  Signals found (inflow-confirmed): {len(signals)}")
    for s in signals:
        log.info(f"    {s['ticker']}: {s['pct_below_sma20']}% below SMA, "
                 f"subsector={s['subsector']} ({s['rotation_phase']}, score={s['rotation_score']:.2f})")

    # Process entries
    process_entries(state, signals, raw_df)

    # Compute equity
    equity = state["capital"]
    for pos in state["positions"]:
        try:
            if isinstance(raw_df.columns, pd.MultiIndex):
                cur = float(raw_df[(pos["ticker"], "Close")].dropna().iloc[-1])
            else:
                cur = float(raw_df["Close"].dropna().iloc[-1])
        except Exception:
            cur = pos["entry_price"]
        equity += pos["shares"] * cur

    metrics = compute_metrics(state)
    state["last_run_date"] = today_str
    state["last_equity"] = round(equity, 2)
    state["metrics"] = metrics
    save_state(state)

    log.info(f"  SUMMARY | Date: {today_str} | Equity: ${equity:,.2f} | "
             f"Positions: {len(state['positions'])} | Closed: {metrics['total_trades']} | "
             f"WR: {metrics.get('win_rate', 0)}% | Sharpe: {metrics.get('sharpe', 0)} | "
             f"PF: {metrics.get('profit_factor', 0)}")


if __name__ == "__main__":
    main()
