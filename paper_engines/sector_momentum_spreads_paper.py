#!/usr/bin/env python3
"""
Sector Rotation Momentum Spreads Paper Engine
==============================================

Event-driven catalyst strategy for the $681 RH agentic account.

When signal_watcher detects a regime change, buy bull call spreads on the
leading sector ETF and bear put spreads on the lagging sector.
Pairs trade with defined risk on both sides.

Uses 10-day vs 20-day momentum relative strength to find leaders/laggards.

Key design:
  - Bull call spreads on leader ETF, bear put spreads on laggard ETF
  - Standard monthly expirations only (3rd Friday)
  - 30 DTE target, $150 max per trade
  - Defined risk: max loss = spread debit paid
  - Regime-driven entries from signal_watcher_state.json
  - Commission: $0.65/leg ($2.60 RT for a spread)

Usage:
    python sector_momentum_spreads_paper.py              # normal daily run
    python sector_momentum_spreads_paper.py --dry-run    # simulate without state changes
"""
import calendar
import json
import logging
import math
import os
import sys
import tempfile
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
STATE_PATH = STATE_DIR / "sector_momentum_spreads_paper_state.json"
TRADE_LOG = LOG_DIR / "sector_momentum_spreads_trades.jsonl"
LOG_FILE = LOG_DIR / "sector_momentum_spreads_paper.log"
SIGNAL_WATCHER_PATH = STATE_DIR / "signal_watcher_state.json"

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
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLC", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLY"]
EXTRA_TICKERS = ["SPY", "^VIX"]
INITIAL_CAPITAL = 681.0
MAX_POS_COST = 150.0          # max $150 per spread trade
COMMISSION_PER_LEG = 0.65
SPREAD_COMMISSION = 4 * COMMISSION_PER_LEG  # $2.60 RT for a spread
DTE_TARGET_MIN = 21
DTE_TARGET_MAX = 45
SPREAD_WIDTH_PCT = 3.0        # spread width as % of underlying price
MAX_CONCURRENT_PAIRS = 2      # max 2 rotation pairs at once
MOM_SHORT = 10                # 10-day momentum window
MOM_LONG = 20                 # 20-day momentum window

# Exit rules
TP_PCT = 0.40                 # +40% take profit (spread appreciation)
SL_PCT = -0.50                # -50% stop loss (defined risk anyway)
MAX_HOLD_DAYS = 15            # close after 15 trading days
TRAILING_ACTIVATE_PCT = 0.20  # activate trailing at +20%
TRAILING_GIVEBACK_PCT = 0.50  # give back 50% of peak gain


# ==================== HELPERS ====================

def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given month/year (standard monthly expiry)."""
    cal = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
    return date(year, month, fridays[2])


def find_standard_expiry(ref_date: date = None, dte_min: int = 21, dte_max: int = 45) -> date:
    """Find the nearest standard monthly option expiry (3rd Friday) in range."""
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
        "config_version": "sector_momentum_spreads_v1",
        "equity": INITIAL_CAPITAL,
        "cash": INITIAL_CAPITAL,
        "open_positions": [],
        "closed_trades": [],
        "last_regime": None,
        "last_regime_change_date": None,
        "total_trades": 0,
        "total_pnl": 0.0,
        "wins": 0,
        "losses": 0,
        "pair_wins": 0,
        "pair_losses": 0,
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


# ==================== SIGNAL WATCHER REGIME ====================

def load_signal_watcher():
    """Load signal watcher state for regime detection."""
    if not SIGNAL_WATCHER_PATH.exists():
        log.warning("Signal watcher state not found")
        return None
    try:
        with open(SIGNAL_WATCHER_PATH) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to read signal watcher: {e}")
        return None


def detect_regime(sw: dict) -> str:
    """Determine current regime from signal watcher.
    Returns one of: 'UPRO', 'SPY', 'HEDGE', 'CASH'."""
    if not sw:
        return "UNKNOWN"
    regime_info = sw.get("regime", {})
    vmr = sw.get("vmr", {})
    return vmr.get("regime", regime_info.get("vol_tier", "UNKNOWN"))


def regime_changed(state: dict, current_regime: str) -> bool:
    """Check if regime has changed since last run."""
    last = state.get("last_regime")
    if last is None:
        return True  # first run
    return last != current_regime


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download sector ETF + VIX data via yfinance."""
    import yfinance as yf
    all_tickers = SECTOR_ETFS + EXTRA_TICKERS
    raw = yf.download(all_tickers, start="2024-01-01", progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    close = close.ffill()
    rename_map = {"^VIX": "VIX"}
    close = close.rename(columns=rename_map)

    vix = close["VIX"].dropna() if "VIX" in close.columns else None
    if vix is None:
        raise ValueError("VIX data not available")

    spy = close["SPY"].dropna() if "SPY" in close.columns else None
    sc = close[[c for c in SECTOR_ETFS if c in close.columns]].dropna(how="all")

    ix = sc.index.intersection(vix.index)
    if spy is not None:
        ix = ix.intersection(spy.index)

    return close.loc[ix], sc.loc[ix], vix.loc[ix]


# ==================== MOMENTUM RANKING ====================

def compute_relative_strength(sc: pd.DataFrame) -> dict:
    """
    Compute relative strength for each sector ETF.
    Uses 10d vs 20d momentum acceleration to find leaders and laggards.

    Returns dict: {ticker: {'mom_10d': float, 'mom_20d': float,
                            'accel': float, 'rank_score': float}}
    """
    if len(sc) < 25:
        log.warning("Not enough data for momentum calculation")
        return {}

    results = {}
    for ticker in sc.columns:
        px = sc[ticker].dropna()
        if len(px) < 25:
            continue

        mom_10d = float(px.iloc[-1] / px.iloc[-11] - 1) * 100 if len(px) > 11 else 0.0
        mom_20d = float(px.iloc[-1] / px.iloc[-21] - 1) * 100 if len(px) > 21 else 0.0

        # Acceleration: how fast momentum is building
        accel = mom_10d - (mom_20d / 2)  # 10d outperforming half of 20d = accelerating

        # Composite rank score: weight short-term momentum higher
        rank_score = 0.6 * mom_10d + 0.4 * mom_20d

        results[ticker] = {
            "mom_10d": round(mom_10d, 3),
            "mom_20d": round(mom_20d, 3),
            "accel": round(accel, 3),
            "rank_score": round(rank_score, 3),
            "price": round(float(px.iloc[-1]), 2),
        }

    return results


def find_leader_laggard(rankings: dict) -> tuple:
    """Find the leading and lagging sector ETFs.
    Returns (leader_ticker, leader_data, laggard_ticker, laggard_data) or None."""
    if len(rankings) < 3:
        return None

    sorted_sectors = sorted(rankings.items(), key=lambda x: x[1]["rank_score"], reverse=True)

    leader_ticker, leader_data = sorted_sectors[0]
    laggard_ticker, laggard_data = sorted_sectors[-1]

    # Minimum spread between leader and laggard (avoid noise)
    spread = leader_data["rank_score"] - laggard_data["rank_score"]
    if spread < 2.0:  # less than 2% spread = not enough differentiation
        log.info(f"Leader-laggard spread too narrow ({spread:.2f}%). No trade.")
        return None

    return leader_ticker, leader_data, laggard_ticker, laggard_data


# ==================== SPREAD PRICING ====================

def estimate_spread_price(S: float, spread_type: str, vix_val: float,
                          dte_days: int = 30) -> dict:
    """
    Estimate bull call spread or bear put spread price.

    Bull call spread: buy ATM call + sell OTM call (debit spread)
    Bear put spread: buy ATM put + sell OTM put (debit spread)

    Returns dict with spread details.
    """
    iv = (vix_val / 100.0) * 1.2  # sector ETFs slightly higher IV than VIX
    t = dte_days / 365.0
    sqrt_t = math.sqrt(t)

    width = S * SPREAD_WIDTH_PCT / 100.0  # spread width in dollars

    if spread_type == "bull_call":
        # Buy ATM call, sell ATM+width call
        long_strike = round(S)
        short_strike = round(S + width)
        # ATM call price ~ 0.4 * S * sqrt(T) * IV
        long_price = 0.4 * S * sqrt_t * iv
        # OTM call price is less
        moneyness = width / S
        short_price = long_price * max(0.1, 1.0 - moneyness * 4)  # rough decay
        net_debit = long_price - short_price
        max_profit = width - net_debit
    else:  # bear_put
        # Buy ATM put, sell ATM-width put
        long_strike = round(S)
        short_strike = round(S - width)
        long_price = 0.35 * S * sqrt_t * iv
        moneyness = width / S
        short_price = long_price * max(0.1, 1.0 - moneyness * 4)
        net_debit = long_price - short_price
        max_profit = width - net_debit

    contract_cost = net_debit * 100 + SPREAD_COMMISSION

    return {
        "spread_type": spread_type,
        "long_strike": long_strike,
        "short_strike": short_strike,
        "net_debit_ps": round(net_debit, 4),
        "contract_cost": round(contract_cost, 2),
        "max_profit_ps": round(max_profit, 4),
        "max_profit": round(max_profit * 100 - SPREAD_COMMISSION, 2),
        "width": round(width, 2),
    }


def estimate_spread_value(pos: dict, current_underlying: float,
                          days_held: int, vix_val: float) -> float:
    """
    Estimate current spread value given underlying move and time decay.

    Bull call spread: gains when underlying moves up
    Bear put spread: gains when underlying moves down
    """
    entry_debit = pos["net_debit_ps"]
    S_entry = pos["entry_price_underlying"]
    S_now = current_underlying
    dte_at_entry = pos["dte"]
    dte_remaining = max(dte_at_entry - days_held, 1)
    width = pos["spread_width"]

    # Delta-based P&L (spread delta ~0.3 for ATM spreads)
    if pos["spread_type"] == "bull_call":
        move_pct = (S_now - S_entry) / S_entry
        # Spread value increases as underlying goes up, capped at width
        delta_pnl = 0.3 * (S_now - S_entry)
    else:  # bear_put
        move_pct = (S_entry - S_now) / S_entry
        delta_pnl = 0.3 * (S_entry - S_now)

    # Theta decay (proportional, spread theta is lower than naked options)
    theta_decay = entry_debit * 0.5 * (days_held / dte_at_entry)

    current_value = entry_debit + delta_pnl - theta_decay

    # Clamp: spread can't be worth less than 0 or more than width
    current_value = max(0.01, min(current_value, width))

    return round(current_value, 4)


# ==================== POSITION MANAGEMENT ====================

def open_pair_trade(state: dict, leader: str, leader_data: dict,
                    laggard: str, laggard_data: dict,
                    vix_val: float, today, expiry: date) -> dict:
    """Open a rotation pair: bull call spread on leader, bear put spread on laggard."""
    dte = (expiry - today.date() if hasattr(today, 'date') else expiry - today).days
    pair_id = f"{leader}_{laggard}_{today.date() if hasattr(today, 'date') else today}"

    positions_opened = []

    for ticker, data, spread_type in [
        (leader, leader_data, "bull_call"),
        (laggard, laggard_data, "bear_put"),
    ]:
        price = data["price"]
        spread = estimate_spread_price(price, spread_type, vix_val, dte)

        if spread["contract_cost"] > MAX_POS_COST:
            log.info(f"  SKIP {ticker} {spread_type}: cost ${spread['contract_cost']:.2f} > ${MAX_POS_COST}")
            continue

        if spread["contract_cost"] > state["cash"]:
            log.info(f"  SKIP {ticker} {spread_type}: insufficient cash "
                     f"(${state['cash']:.2f} < ${spread['contract_cost']:.2f})")
            continue

        position = {
            "ticker": ticker,
            "spread_type": spread_type,
            "pair_id": pair_id,
            "entry_date": str(today.date() if hasattr(today, 'date') else today),
            "expiry": str(expiry),
            "entry_price_underlying": price,
            "long_strike": spread["long_strike"],
            "short_strike": spread["short_strike"],
            "net_debit_ps": spread["net_debit_ps"],
            "spread_width": spread["width"],
            "cost": spread["contract_cost"],
            "dte": dte,
            "peak_value_ps": spread["net_debit_ps"],
            "trailing_active": False,
            "vix_at_entry": round(vix_val, 2),
            "mom_10d": data["mom_10d"],
            "mom_20d": data["mom_20d"],
            "rank_score": data["rank_score"],
        }

        state["open_positions"].append(position)
        state["cash"] -= spread["contract_cost"]
        state["total_trades"] += 1
        positions_opened.append(position)

        log_trade({
            "action": "OPEN",
            "date": str(today.date() if hasattr(today, 'date') else today),
            "ticker": ticker,
            "spread_type": spread_type,
            "pair_id": pair_id,
            "long_strike": spread["long_strike"],
            "short_strike": spread["short_strike"],
            "net_debit": round(spread["net_debit_ps"], 4),
            "cost": round(spread["contract_cost"], 2),
            "vix": round(vix_val, 2),
            "rank_score": data["rank_score"],
            "cash_after": round(state["cash"], 2),
        })

        log.info(f"  ENTER {ticker} {spread_type.upper()} "
                 f"strikes={spread['long_strike']}/{spread['short_strike']} "
                 f"debit=${spread['net_debit_ps']:.2f}/sh (${spread['contract_cost']:.2f} total) "
                 f"rank={data['rank_score']:.2f}")

    return state


def check_exits(state: dict, sc: pd.DataFrame, vix: pd.Series, today) -> dict:
    """Check all open positions for exit conditions."""
    positions_to_close = []

    for i, pos in enumerate(state["open_positions"]):
        tk = pos["ticker"]
        if tk not in sc.columns:
            continue

        entry_date = pd.Timestamp(pos["entry_date"])
        days_held = len(sc.index[(sc.index > entry_date) & (sc.index <= today)])
        if days_held == 0:
            continue

        current_underlying = float(sc[tk].iloc[-1])
        current_vix = float(vix.iloc[-1])
        current_ps = estimate_spread_value(pos, current_underlying, days_held, current_vix)

        entry_ps = pos["net_debit_ps"]
        if entry_ps <= 0:
            continue
        pct_change = (current_ps - entry_ps) / entry_ps

        # Update peak for trailing stop
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

        # 5. Expiry approaching (2 days before)
        elif pos.get("expiry"):
            exp_date = date.fromisoformat(pos["expiry"])
            today_date = today.date() if hasattr(today, "date") else today
            if (exp_date - today_date).days <= 2:
                exit_reason = "near_expiry"

        if exit_reason:
            pnl = (current_ps - entry_ps) * 100 - SPREAD_COMMISSION
            positions_to_close.append(
                (i, pnl, exit_reason, days_held, current_ps, current_underlying)
            )

    # Close in reverse order
    for i, pnl, reason, days_held, exit_ps, exit_underlying in reversed(positions_to_close):
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

        log.info(f"  EXIT {pos['ticker']} {pos['spread_type'].upper()}: "
                 f"PnL ${pnl:.2f} ({pct_ret:+.1f}%) reason={reason} held={days_held}d")

        trade_record = {
            "action": "CLOSE",
            "date": str(today.date() if hasattr(today, "date") else today),
            "ticker": pos["ticker"],
            "spread_type": pos["spread_type"],
            "pair_id": pos.get("pair_id"),
            "entry_date": pos["entry_date"],
            "days_held": days_held,
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
    log.info(f"Last regime: {state.get('last_regime', 'N/A')}")


def run_daily():
    """Main daily run. Called once per trading day after market close."""
    state = load_state()

    if DRY_RUN:
        log.info("=" * 60)
        log.info("  DRY-RUN MODE -- no state changes will be persisted")
        log.info("=" * 60)

    log.info("=== Sector Rotation Momentum Spreads Paper Engine ===")
    log.info(f"Equity: ${state['equity']:.2f} | Cash: ${state['cash']:.2f} | "
             f"Open: {len(state['open_positions'])} | "
             f"Trades: {state['total_trades']} | W/L: {state['wins']}/{state['losses']}")

    # Weekend/holiday check
    now = datetime.now()
    if now.weekday() >= 5:
        log.info("Weekend — skipping")
        return

    # Download data
    try:
        close_df, sc, vix = download_data()
    except Exception as e:
        log.error(f"Data download failed: {e}")
        save_state(state)
        return

    if sc.empty or vix.empty:
        log.error("Empty data received")
        save_state(state)
        return

    today = sc.index[-1]
    current_vix = float(vix.iloc[-1])
    log.info(f"Date: {today.date()} | VIX: {current_vix:.1f}")

    # ── Check exits on existing positions ──
    state = check_exits(state, sc, vix, today)

    # ── Load signal watcher for regime ──
    sw = load_signal_watcher()
    current_regime = detect_regime(sw)
    log.info(f"Current regime: {current_regime} (previous: {state.get('last_regime', 'N/A')})")

    # ── Check for regime change (entry trigger) ──
    is_regime_change = regime_changed(state, current_regime)

    # Count current open pairs
    pair_ids = set(p.get("pair_id") for p in state["open_positions"] if p.get("pair_id"))
    n_open_pairs = len(pair_ids)

    if not is_regime_change:
        log.info("No regime change detected. Exits only today.")
        state["last_regime"] = current_regime
        _print_summary(state)
        save_state(state)
        return

    log.info(f"*** REGIME CHANGE: {state.get('last_regime', 'N/A')} -> {current_regime} ***")
    state["last_regime"] = current_regime
    state["last_regime_change_date"] = str(today.date() if hasattr(today, "date") else today)

    if n_open_pairs >= MAX_CONCURRENT_PAIRS:
        log.info(f"Max concurrent pairs ({MAX_CONCURRENT_PAIRS}) reached. Wait for exits.")
        _print_summary(state)
        save_state(state)
        return

    # ── Compute momentum rankings ──
    rankings = compute_relative_strength(sc)
    if not rankings:
        log.warning("No momentum rankings available. Skipping entry.")
        _print_summary(state)
        save_state(state)
        return

    log.info("Sector momentum rankings:")
    for ticker, data in sorted(rankings.items(), key=lambda x: x[1]["rank_score"], reverse=True):
        log.info(f"  {ticker}: 10d={data['mom_10d']:+.2f}% 20d={data['mom_20d']:+.2f}% "
                 f"accel={data['accel']:+.2f} score={data['rank_score']:+.2f}")

    # ── Find leader/laggard pair ──
    pair = find_leader_laggard(rankings)
    if pair is None:
        log.info("No suitable leader/laggard pair found.")
        _print_summary(state)
        save_state(state)
        return

    leader, leader_data, laggard, laggard_data = pair
    log.info(f"PAIR: Leader={leader} (score={leader_data['rank_score']:.2f}) | "
             f"Laggard={laggard} (score={laggard_data['rank_score']:.2f})")

    # ── Skip if already holding either ticker ──
    held_tickers = set(p["ticker"] for p in state["open_positions"])
    if leader in held_tickers or laggard in held_tickers:
        log.info(f"Already holding {leader} or {laggard}. Skip.")
        _print_summary(state)
        save_state(state)
        return

    # ── Find standard monthly expiry ──
    ref_date = today.date() if hasattr(today, "date") else today
    expiry = find_standard_expiry(ref_date, DTE_TARGET_MIN, DTE_TARGET_MAX)
    log.info(f"Target expiry: {expiry} (DTE: {(expiry - ref_date).days})")

    # ── Open pair trade ──
    state = open_pair_trade(state, leader, leader_data, laggard, laggard_data,
                            current_vix, today, expiry)

    _print_summary(state)
    save_state(state)


if __name__ == "__main__":
    run_daily()
