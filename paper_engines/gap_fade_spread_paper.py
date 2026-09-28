#!/usr/bin/env python3
"""
Post-Gap Fade with Spreads Paper Engine
========================================

Event-driven catalyst strategy for the $681 RH agentic account.

Stocks in the quality universe that gap down 5%+ on high volume tend to
revert. This engine buys bull call spreads 1-2 strikes OTM after a >5%
gap down, using standard monthly (3rd Friday) expirations at 30 DTE.

Combines proven mean-reversion edge with defined-risk option spreads.

Quality Universe:
    AAPL, MSFT, GOOGL, AMZN, META, NVDA, AVGO, JPM, UNH, V, MA, JNJ,
    PG, HD, COST, ABBV, LLY, MRK, PEP, KO, CRM, ADBE, ACN, TMO, ABT,
    DHR, TXN, NEE, LIN, LOW

Key design:
  - Bull call spreads 1-2 strikes OTM after >5% gap down
  - Standard monthly expirations only (3rd Friday)
  - 30 DTE target, $150 max per trade
  - Volume filter: gap day volume must be >2x 20-day average
  - Max 3 concurrent positions
  - Commission: $0.65/leg ($2.60 RT for a spread)

Usage:
    python gap_fade_spread_paper.py              # normal daily run
    python gap_fade_spread_paper.py --dry-run    # simulate without state changes
"""
import calendar
import json
import logging
import math
import os
import sys
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──
BASE = Path("/home/jupiter/Lvl3Quant")
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = BASE / "state"
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / "gap_fade_spread_paper_state.json"
TRADE_LOG = LOG_DIR / "gap_fade_spread_trades.jsonl"
LOG_FILE = LOG_DIR / "gap_fade_spread_paper.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── DRY-RUN ──
DRY_RUN = "--dry-run" in sys.argv

# ==================== STRATEGY CONFIG ====================
QUALITY_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO",
    "JPM", "UNH", "V", "MA", "JNJ", "PG", "HD", "COST",
    "ABBV", "LLY", "MRK", "PEP", "KO", "CRM", "ADBE", "ACN",
    "TMO", "ABT", "DHR", "TXN", "NEE", "LIN", "LOW",
]
INITIAL_CAPITAL = 681.0
MAX_POS_COST = 150.0
COMMISSION_PER_LEG = 0.65
SPREAD_COMMISSION = 4 * COMMISSION_PER_LEG  # $2.60 RT
GAP_THRESHOLD = -5.0          # gap down >5% to trigger
VOLUME_MULTIPLIER = 2.0       # volume must be 2x the 20-day avg
MAX_CONCURRENT = 3
DTE_TARGET_MIN = 21
DTE_TARGET_MAX = 45
SPREAD_WIDTH_STRIKES = 5.0    # $5 wide spread for stocks >$50

# Exit rules (mean reversion trades hold longer than momentum)
TP_PCT = 0.50                 # +50% take profit (spread on a reverting stock)
SL_PCT = -0.60                # -60% stop loss (defined risk anyway)
MAX_HOLD_DAYS = 20            # close after 20 trading days
TRAILING_ACTIVATE_PCT = 0.25  # activate trailing at +25%
TRAILING_GIVEBACK_PCT = 0.50  # give back 50% of peak gain

# Cooldown: don't re-enter same ticker within N days
COOLDOWN_DAYS = 10


# ==================== HELPERS ====================

def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given month/year."""
    cal = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
    return date(year, month, fridays[2])


def find_standard_expiry(ref_date: date = None, dte_min: int = 21, dte_max: int = 45) -> date:
    """Find nearest standard monthly expiry in range."""
    if ref_date is None:
        ref_date = date.today()

    monthlies = []
    for offset in range(0, 5):
        y = ref_date.year + (ref_date.month + offset - 1) // 12
        m = (ref_date.month + offset - 1) % 12 + 1
        tf = _third_friday(y, m)
        dte = (tf - ref_date).days
        if dte >= 7:
            monthlies.append((tf, dte))

    in_window = [(tf, dte) for tf, dte in monthlies if dte_min <= dte <= dte_max]
    if in_window:
        return in_window[0][0]

    viable = [(tf, dte) for tf, dte in monthlies if dte >= 21]
    if viable:
        target = (dte_min + dte_max) // 2
        viable.sort(key=lambda x: abs(x[1] - target))
        return viable[0][0]

    if monthlies:
        return monthlies[0][0]

    start = ref_date + timedelta(days=21)
    days_to_friday = (4 - start.weekday()) % 7
    if days_to_friday == 0 and start.weekday() != 4:
        days_to_friday = 7
    return start + timedelta(days=days_to_friday)


# ==================== STATE MANAGEMENT ====================

def _default_state():
    return {
        "config_version": "gap_fade_spread_v1",
        "equity": INITIAL_CAPITAL,
        "cash": INITIAL_CAPITAL,
        "open_positions": [],
        "closed_trades": [],
        "recent_entries": {},  # {ticker: last_entry_date} for cooldown
        "total_trades": 0,
        "total_pnl": 0.0,
        "wins": 0,
        "losses": 0,
        "gaps_detected": 0,
        "gaps_traded": 0,
        "created": datetime.now().isoformat(),
    }


def load_state():
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH) as f:
                state = json.load(f)
            defaults = _default_state()
            for k, v in defaults.items():
                if k not in state:
                    state[k] = v
            return state
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"State file corrupt, starting fresh: {e}")
    return _default_state()


def save_state(state):
    if DRY_RUN:
        log.info("[DRY-RUN] State NOT saved")
        return
    state["last_updated"] = datetime.now().isoformat()
    tmp = STATE_PATH.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, STATE_PATH)
    except Exception as e:
        log.error(f"Failed to save state: {e}")
        if tmp.exists():
            tmp.unlink()


def log_trade(record):
    if DRY_RUN:
        log.info(f"[DRY-RUN] Trade NOT logged: {record.get('action')} {record.get('ticker', 'N/A')}")
        return
    try:
        with open(TRADE_LOG, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        log.error(f"Failed to log trade: {e}")


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download quality universe stock data via yfinance."""
    import yfinance as yf
    all_tickers = QUALITY_UNIVERSE + ["^VIX"]
    raw = yf.download(all_tickers, start="2024-01-01", progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)

    if mi:
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
        opn = raw["Open"].copy()
    else:
        close = raw[["Close"]].copy()
        volume = raw[["Volume"]].copy()
        opn = raw[["Open"]].copy()

    for df in [close, volume, opn]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

    rename_map = {"^VIX": "VIX"}
    close = close.rename(columns=rename_map).ffill()
    volume = volume.rename(columns=rename_map).ffill()
    opn = opn.rename(columns=rename_map).ffill()

    vix = close["VIX"].dropna() if "VIX" in close.columns else None

    return close, volume, opn, vix


# ==================== GAP DETECTION ====================

def detect_gaps(close: pd.DataFrame, volume: pd.DataFrame,
                opn: pd.DataFrame) -> list:
    """
    Detect stocks that gapped down >5% on high volume today.

    Gap = (today's open - yesterday's close) / yesterday's close

    Returns list of dicts with gap details.
    """
    if len(close) < 22:
        return []

    gaps = []
    today = close.index[-1]
    yesterday_idx = -2

    for ticker in QUALITY_UNIVERSE:
        if ticker not in close.columns or ticker not in volume.columns:
            continue
        if ticker not in opn.columns:
            continue

        try:
            prev_close = float(close[ticker].iloc[yesterday_idx])
            today_open = float(opn[ticker].iloc[-1])
            today_close = float(close[ticker].iloc[-1])
            today_vol = float(volume[ticker].iloc[-1])

            if prev_close <= 0 or np.isnan(prev_close):
                continue
            if np.isnan(today_open) or np.isnan(today_vol):
                continue

            gap_pct = (today_open - prev_close) / prev_close * 100

            # Volume check: 20-day average
            vol_20d = float(volume[ticker].iloc[-22:-2].mean())
            if vol_20d <= 0 or np.isnan(vol_20d):
                continue
            vol_ratio = today_vol / vol_20d

            if gap_pct <= GAP_THRESHOLD and vol_ratio >= VOLUME_MULTIPLIER:
                gaps.append({
                    "ticker": ticker,
                    "gap_pct": round(gap_pct, 2),
                    "prev_close": round(prev_close, 2),
                    "today_open": round(today_open, 2),
                    "today_close": round(today_close, 2),
                    "volume_ratio": round(vol_ratio, 2),
                    "date": str(today.date() if hasattr(today, "date") else today),
                })

        except (IndexError, KeyError, ValueError):
            continue

    return gaps


# ==================== SPREAD PRICING ====================

def estimate_bull_call_spread(S: float, vix_val: float, dte_days: int = 30) -> dict:
    """
    Estimate bull call spread 1-2 strikes OTM.

    Buy call at strike ~= S + 1 strike, sell call at strike + width.
    For stocks $50-500, strikes are $5 apart.
    """
    iv = (vix_val / 100.0) * 1.1
    t = dte_days / 365.0
    sqrt_t = math.sqrt(t)

    # Determine strike spacing
    if S < 50:
        strike_step = 1.0
    elif S < 200:
        strike_step = 5.0
    else:
        strike_step = 10.0

    # 1 strike OTM for the long leg
    long_strike = math.ceil(S / strike_step) * strike_step
    short_strike = long_strike + SPREAD_WIDTH_STRIKES

    # Adjust width based on stock price
    if S >= 200:
        short_strike = long_strike + 10.0
    elif S < 50:
        short_strike = long_strike + 2.0

    width = short_strike - long_strike

    # Price estimation (simplified BS)
    # Slightly OTM call: ~0.35 * S * sqrt(T) * IV * exp(-moneyness_factor)
    moneyness_long = (long_strike - S) / S
    moneyness_short = (short_strike - S) / S

    base_price = 0.4 * S * sqrt_t * iv
    long_price = base_price * max(0.05, math.exp(-3 * max(0, moneyness_long)))
    short_price = base_price * max(0.02, math.exp(-3 * max(0, moneyness_short)))

    net_debit = long_price - short_price
    net_debit = max(net_debit, 0.10)  # floor
    contract_cost = net_debit * 100 + SPREAD_COMMISSION
    max_profit = (width - net_debit) * 100 - SPREAD_COMMISSION

    return {
        "long_strike": long_strike,
        "short_strike": short_strike,
        "width": width,
        "net_debit_ps": round(net_debit, 4),
        "contract_cost": round(contract_cost, 2),
        "max_profit": round(max_profit, 2),
    }


def estimate_spread_value(pos: dict, current_price: float,
                          days_held: int) -> float:
    """Estimate current spread value for mark-to-market."""
    entry_debit = pos["net_debit_ps"]
    S_entry = pos["entry_price"]
    S_now = current_price
    dte_at_entry = pos["dte"]
    dte_remaining = max(dte_at_entry - days_held, 1)
    width = pos["spread_width"]

    # Bull call spread gains when stock goes up
    move_pct = (S_now - S_entry) / S_entry

    # Delta effect: spread delta ~0.25-0.35 for slightly OTM
    delta_pnl = 0.30 * (S_now - S_entry) * (dte_remaining / dte_at_entry)

    # Theta decay (spreads have lower theta than naked)
    theta_factor = 0.4 * (days_held / dte_at_entry)
    theta_decay = entry_debit * theta_factor

    current_value = entry_debit + delta_pnl - theta_decay
    # Clamp
    current_value = max(0.01, min(current_value, width))

    return round(current_value, 4)


# ==================== POSITION MANAGEMENT ====================

def open_position(state: dict, gap: dict, vix_val: float,
                  expiry: date) -> dict:
    """Open a bull call spread on a gapped-down quality stock."""
    ticker = gap["ticker"]
    price = gap["today_close"]  # enter at close after gap
    dte = (expiry - date.fromisoformat(gap["date"])).days

    spread = estimate_bull_call_spread(price, vix_val, dte)

    if spread["contract_cost"] > MAX_POS_COST:
        log.info(f"  SKIP {ticker}: cost ${spread['contract_cost']:.2f} > ${MAX_POS_COST}")
        return state

    if spread["contract_cost"] > state["cash"]:
        log.info(f"  SKIP {ticker}: insufficient cash "
                 f"(${state['cash']:.2f} < ${spread['contract_cost']:.2f})")
        return state

    position = {
        "ticker": ticker,
        "spread_type": "bull_call",
        "entry_date": gap["date"],
        "expiry": str(expiry),
        "entry_price": price,
        "gap_pct": gap["gap_pct"],
        "volume_ratio": gap["volume_ratio"],
        "long_strike": spread["long_strike"],
        "short_strike": spread["short_strike"],
        "spread_width": spread["width"],
        "net_debit_ps": spread["net_debit_ps"],
        "cost": spread["contract_cost"],
        "max_profit": spread["max_profit"],
        "dte": dte,
        "peak_value_ps": spread["net_debit_ps"],
        "trailing_active": False,
        "vix_at_entry": round(vix_val, 2),
    }

    state["open_positions"].append(position)
    state["cash"] -= spread["contract_cost"]
    state["total_trades"] += 1
    state["gaps_traded"] = state.get("gaps_traded", 0) + 1
    state["recent_entries"][ticker] = gap["date"]

    log_trade({
        "action": "OPEN",
        "date": gap["date"],
        "ticker": ticker,
        "gap_pct": gap["gap_pct"],
        "volume_ratio": gap["volume_ratio"],
        "entry_price": price,
        "long_strike": spread["long_strike"],
        "short_strike": spread["short_strike"],
        "net_debit": spread["net_debit_ps"],
        "cost": spread["contract_cost"],
        "vix": round(vix_val, 2),
        "cash_after": round(state["cash"], 2),
    })

    log.info(f"  ENTER {ticker} BULL CALL SPREAD "
             f"strikes={spread['long_strike']}/{spread['short_strike']} "
             f"debit=${spread['net_debit_ps']:.2f}/sh (${spread['contract_cost']:.2f} total) "
             f"gap={gap['gap_pct']:.1f}% vol_ratio={gap['volume_ratio']:.1f}x")

    return state


def check_exits(state: dict, close: pd.DataFrame, today) -> dict:
    """Check all open positions for exit conditions."""
    positions_to_close = []

    for i, pos in enumerate(state["open_positions"]):
        tk = pos["ticker"]
        if tk not in close.columns:
            continue

        entry_date = pd.Timestamp(pos["entry_date"])
        days_held = len(close.index[(close.index > entry_date) & (close.index <= today)])
        if days_held == 0:
            continue

        current_price = float(close[tk].iloc[-1])
        current_ps = estimate_spread_value(pos, current_price, days_held)

        entry_ps = pos["net_debit_ps"]
        if entry_ps <= 0:
            continue
        pct_change = (current_ps - entry_ps) / entry_ps

        # Update peak
        if current_ps > pos["peak_value_ps"]:
            pos["peak_value_ps"] = round(current_ps, 4)

        if pct_change >= TRAILING_ACTIVATE_PCT:
            pos["trailing_active"] = True

        exit_reason = None

        # 1. Take profit
        if pct_change >= TP_PCT:
            exit_reason = "take_profit"

        # 2. Stop loss
        elif pct_change <= SL_PCT:
            exit_reason = "stop_loss"

        # 3. Trailing stop
        elif pos["trailing_active"]:
            peak_ps = pos["peak_value_ps"]
            peak_gain = peak_ps - entry_ps
            current_gain = current_ps - entry_ps
            if peak_gain > 0 and current_gain < peak_gain * (1 - TRAILING_GIVEBACK_PCT):
                exit_reason = "trailing_stop"

        # 4. Max hold
        elif days_held >= MAX_HOLD_DAYS:
            exit_reason = "max_hold"

        # 5. Near expiry
        elif pos.get("expiry"):
            exp_date = date.fromisoformat(pos["expiry"])
            today_date = today.date() if hasattr(today, "date") else today
            if (exp_date - today_date).days <= 2:
                exit_reason = "near_expiry"

        if exit_reason:
            pnl = (current_ps - entry_ps) * 100 - SPREAD_COMMISSION
            positions_to_close.append(
                (i, pnl, exit_reason, days_held, current_ps, current_price)
            )

    # Close in reverse order
    for i, pnl, reason, days_held, exit_ps, exit_price in reversed(positions_to_close):
        pos = state["open_positions"].pop(i)
        state["equity"] += pnl
        state["cash"] += pos["cost"] + pnl
        state["total_pnl"] += pnl

        is_win = pnl > 0
        if is_win:
            state["wins"] += 1
        else:
            state["losses"] += 1

        pct_ret = pnl / pos["cost"] * 100 if pos["cost"] > 0 else 0

        log.info(f"  EXIT {pos['ticker']} BULL CALL SPREAD: "
                 f"PnL ${pnl:.2f} ({pct_ret:+.1f}%) reason={reason} held={days_held}d "
                 f"(gap was {pos['gap_pct']:.1f}%)")

        trade_record = {
            "action": "CLOSE",
            "date": str(today.date() if hasattr(today, "date") else today),
            "ticker": pos["ticker"],
            "spread_type": "bull_call",
            "entry_date": pos["entry_date"],
            "days_held": days_held,
            "gap_pct": pos["gap_pct"],
            "entry_price": pos["entry_price"],
            "exit_price": round(exit_price, 2),
            "net_debit": pos["net_debit_ps"],
            "exit_value": round(exit_ps, 4),
            "cost": pos["cost"],
            "pnl": round(pnl, 2),
            "pct_return": round(pct_ret, 2),
            "exit_reason": reason,
            "equity_after": round(state["equity"], 2),
            "cash_after": round(state["cash"], 2),
        }
        state.setdefault("closed_trades", []).append(trade_record)
        log_trade(trade_record)

    return state


# ==================== MAIN DAILY RUN ====================

def _print_summary(state: dict):
    """Print end-of-run summary."""
    total = state["wins"] + state["losses"]
    wr = (state["wins"] / total * 100) if total > 0 else 0
    log.info(f"\n--- Summary ---")
    log.info(f"Equity: ${state['equity']:.2f} | Cash: ${state['cash']:.2f}")
    log.info(f"Open positions: {len(state['open_positions'])}")
    log.info(f"Total trades: {state['total_trades']} | W/L: {state['wins']}/{state['losses']} ({wr:.0f}%)")
    log.info(f"Total PnL: ${state['total_pnl']:.2f}")
    log.info(f"Gaps detected: {state.get('gaps_detected', 0)} | Traded: {state.get('gaps_traded', 0)}")


def run_daily():
    """Main daily run. Called once per trading day after market close."""
    state = load_state()

    if DRY_RUN:
        log.info("=" * 60)
        log.info("  DRY-RUN MODE -- no state changes will be persisted")
        log.info("=" * 60)

    log.info("=== Post-Gap Fade with Spreads Paper Engine ===")
    log.info(f"Equity: ${state['equity']:.2f} | Cash: ${state['cash']:.2f} | "
             f"Open: {len(state['open_positions'])} | "
             f"Trades: {state['total_trades']} | W/L: {state['wins']}/{state['losses']}")

    # Weekend check
    now = datetime.now()
    if now.weekday() >= 5:
        log.info("Weekend — skipping")
        return

    # Download data
    try:
        close, volume, opn, vix = download_data()
    except Exception as e:
        log.error(f"Data download failed: {e}")
        save_state(state)
        return

    if close.empty:
        log.error("Empty data received")
        save_state(state)
        return

    today = close.index[-1]
    current_vix = float(vix.iloc[-1]) if vix is not None and not vix.empty else 20.0
    log.info(f"Date: {today.date()} | VIX: {current_vix:.1f}")

    # ── Check exits on existing positions ──
    state = check_exits(state, close, today)

    # ── Detect gap-down events ──
    gaps = detect_gaps(close, volume, opn)
    state["gaps_detected"] = state.get("gaps_detected", 0) + len(gaps)

    if gaps:
        log.info(f"\n*** {len(gaps)} GAP-DOWN DETECTED ***")
        for g in gaps:
            log.info(f"  {g['ticker']}: gap {g['gap_pct']:.1f}% | "
                     f"vol {g['volume_ratio']:.1f}x avg | "
                     f"close ${g['today_close']:.2f}")
    else:
        log.info("No qualifying gap-downs detected today.")

    # ── Filter and enter new positions ──
    n_open = len(state["open_positions"])
    today_date_str = str(today.date() if hasattr(today, "date") else today)

    # Sort gaps by magnitude (biggest gap first)
    gaps.sort(key=lambda x: x["gap_pct"])

    for gap in gaps:
        if n_open >= MAX_CONCURRENT:
            log.info(f"Max concurrent positions ({MAX_CONCURRENT}) reached.")
            break

        ticker = gap["ticker"]

        # Check cooldown
        recent = state.get("recent_entries", {})
        if ticker in recent:
            try:
                last_entry = date.fromisoformat(recent[ticker])
                ref_date = today.date() if hasattr(today, "date") else today
                if (ref_date - last_entry).days < COOLDOWN_DAYS:
                    log.info(f"  SKIP {ticker}: cooldown ({COOLDOWN_DAYS}d, "
                             f"last entry {recent[ticker]})")
                    continue
            except (ValueError, TypeError):
                pass

        # Check if already holding
        held = set(p["ticker"] for p in state["open_positions"])
        if ticker in held:
            log.info(f"  SKIP {ticker}: already holding")
            continue

        # Find expiry
        ref = today.date() if hasattr(today, "date") else today
        expiry = find_standard_expiry(ref, DTE_TARGET_MIN, DTE_TARGET_MAX)

        # Open position
        state = open_position(state, gap, current_vix, expiry)
        n_open = len(state["open_positions"])

    # ── Clean up old cooldown entries ──
    recent = state.get("recent_entries", {})
    cleaned = {}
    ref_date = today.date() if hasattr(today, "date") else today
    for tk, dt_str in recent.items():
        try:
            if (ref_date - date.fromisoformat(dt_str)).days < COOLDOWN_DAYS * 3:
                cleaned[tk] = dt_str
        except (ValueError, TypeError):
            pass
    state["recent_entries"] = cleaned

    _print_summary(state)
    save_state(state)


if __name__ == "__main__":
    run_daily()
