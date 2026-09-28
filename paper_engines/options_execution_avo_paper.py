#!/usr/bin/env python3
"""
Options Execution Paper Engine (AVO-evolved v25)
==================================================
Sector ETF call options on oversold dips, regime-aware exits.
Black-Scholes theoretical pricing. AVO-evolved over 25 steps.
Sharpe >1.4 all folds, 212% compound return, regime gap 0.148.

Strategy source: AVO run options_execution-20260823-062807, step 25.

Cron: 25 16 * * 1-5  (4:25 PM ET, after market close)
State: /home/jupiter/Lvl3Quant/paper_engines/state/options_execution_avo_state.json
Trades: /home/jupiter/Lvl3Quant/paper_engines/logs/options_execution_avo_trades.csv
"""
import csv
import json
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from scipy.stats import norm

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

warnings.filterwarnings("ignore")

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "paper_engines" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "options_execution_avo_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "options_execution_avo.log"
TRADES_FILE = LOG_DIR / "options_execution_avo_trades.csv"
CALLBACK_SCRIPT = BASE / "scripts" / "run_engine_with_callback.sh"

# ── Strategy Parameters (from AVO v25 strategy.py) ──

# Account
ACCOUNT_SIZE = 650.0
MAX_CONCURRENT = 4
MAX_PER_TRADE_PCT = 0.22        # 22% of account per trade
SLIPPAGE_PCT = 0.001            # 0.1% options spread cost

# Entry — RSI oversold dip-buy
RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 35
RSI_DEEP_OVERSOLD = 30
MIN_VOLUME_RATIO = 0.8

# Ticker universe
PREFERRED_TICKERS = ['XLE', 'XLU', 'XLK', 'XLI', 'XLF', 'XLB', 'XLRE']
BENCHMARK = 'SPY'

# Options parameters
OPTION_DTE = 30
OPTION_DELTA_TARGET = 0.35
RISK_FREE_RATE = 0.05

# Exit — base parameters
HOLD_DAYS_MAX = 5
TP_PCT = 0.43                   # +43% low-vol
HIGHVOL_TP_PCT = 0.25           # +25% high-vol
SL_PCT = -0.17                  # -17%
TRAILING_ACTIVATE_PCT = 0.08    # Activate at +8%
TRAILING_GIVEBACK_PCT = 0.20    # Give back 20% of peak gain
HIGHVOL_GIVEBACK_PCT = 0.35

# Sector-specific hold periods
SECTOR_HOLD_DAYS = {
    'XLK': 4, 'XLF': 4,
    'XLE': 5, 'XLI': 5,
    'XLU': 10, 'XLB': 9, 'XLRE': 11,
}

# Regime risk controls
HIGHVOL_VIX = 25
EXTREME_VIX = 35
HIGHVOL_SL_PCT = -0.12
HIGHVOL_HOLD_REDUCTION = 3
HIGHVOL_TRAILING_ACTIVATE_PCT = 0.10
HIGHVOL_IV_THRESHOLD = 0.35

# Sector relative strength filter
REL_STRENGTH_LOOKBACK = 10
REL_STRENGTH_MIN = -0.04

# Winner hold extension
WINNER_HOLD_EXTENSION = 2

# Idle capital relaxation
IDLE_DAYS_THRESHOLD = 4
RSI_RELAXED_THRESHOLD = 38

# High-vol early exit for failed bounces
HIGHVOL_EARLY_EXIT_DAY = 1
HIGHVOL_EARLY_EXIT_LOSS = -0.05

# Data lookback
DATA_LOOKBACK_DAYS = 120


# ── Core Functions (verbatim from strategy.py) ──

def compute_rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def estimate_iv(prices, window=20):
    """Estimate implied volatility from historical realized vol with a premium."""
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252)
    return rv * 1.15


def generate_signals(prices, spy, vix):
    """
    Generate buy signals for sector ETFs with regime-aware filtering.
    Includes idle capital relaxation (v24) and two-pass signal logic.
    """
    sectors = [c for c in prices.columns if c in set(PREFERRED_TICKERS)]
    signals = pd.DataFrame(False, index=prices.index, columns=sectors)

    spy_ret_10d = spy.pct_change(REL_STRENGTH_LOOKBACK)

    ticker_data = {}
    for ticker in sectors:
        px = prices[ticker].dropna()
        if len(px) < RSI_PERIOD + 5:
            continue

        rsi = compute_rsi(px, RSI_PERIOD)
        vol = px.pct_change().abs()
        avg_vol = vol.rolling(20).mean()

        oversold = rsi < RSI_ENTRY_THRESHOLD
        relaxed_oversold = rsi < RSI_RELAXED_THRESHOLD
        deep_oversold = rsi < RSI_DEEP_OVERSOLD
        vol_ok = vol > avg_vol * MIN_VOLUME_RATIO

        bounce = px > px.shift(1)

        vix_aligned = vix.reindex(px.index, method='ffill')
        vix_prev = vix.shift(1).reindex(px.index, method='ffill')
        vix_prev2 = vix.shift(2).reindex(px.index, method='ffill')

        is_lowvol = vix_aligned < HIGHVOL_VIX
        is_highvol = (vix_aligned >= HIGHVOL_VIX) & (vix_aligned < EXTREME_VIX)
        vix_declining_2d = (vix_aligned < vix_prev) & (vix_prev < vix_prev2)

        sector_ret_10d = px.pct_change(REL_STRENGTH_LOOKBACK)
        spy_ret_aligned = spy_ret_10d.reindex(px.index, method='ffill')
        has_rel_strength = (sector_ret_10d - spy_ret_aligned) > REL_STRENGTH_MIN

        # LOW-VOL: standard oversold dip-buy
        lowvol_signal = oversold & vol_ok & is_lowvol
        relaxed_lowvol_signal = relaxed_oversold & vol_ok & is_lowvol

        # HIGH-VOL: deep oversold + bounce + 2d VIX decline + rel strength
        highvol_signal = (deep_oversold & vol_ok & bounce & vix_declining_2d
                          & is_highvol & has_rel_strength)

        standard_signal = lowvol_signal | highvol_signal
        relaxed_signal = relaxed_lowvol_signal | highvol_signal

        signals.loc[standard_signal.index, ticker] = standard_signal

        ticker_data[ticker] = {
            'standard': standard_signal,
            'relaxed': relaxed_signal,
        }

    # Second pass: idle capital relaxation
    any_signal = signals.any(axis=1)
    signal_count = any_signal.astype(int).rolling(
        IDLE_DAYS_THRESHOLD, min_periods=IDLE_DAYS_THRESHOLD).sum()
    is_idle = signal_count == 0

    for ticker, data in ticker_data.items():
        relaxed_only = data['relaxed'] & ~data['standard']
        idle_relaxed = relaxed_only & is_idle.reindex(relaxed_only.index, fill_value=False)
        combined = signals[ticker] | idle_relaxed.reindex(signals.index, fill_value=False)
        signals[ticker] = combined

    return signals


def compute_option_position(underlying_price, iv_val, vix_now):
    """
    Compute option entry details: strike, premium, greeks.
    Strike chosen for ~0.35 delta (roughly 3% OTM for 30 DTE).
    """
    # Approximate strike for target delta ~0.35 (slightly OTM)
    # For 30 DTE, delta 0.35 is roughly 2-4% OTM depending on vol
    T = OPTION_DTE / 365.0
    sigma = max(iv_val, 0.10)  # Floor IV at 10%

    # Iterate to find strike giving roughly target delta
    # Start with ~3% OTM as initial guess
    strike = underlying_price * 1.03
    for _ in range(10):
        d1 = (np.log(underlying_price / strike) + (RISK_FREE_RATE + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1)
        if abs(delta - OPTION_DELTA_TARGET) < 0.01:
            break
        # Adjust strike: higher strike = lower delta
        if delta > OPTION_DELTA_TARGET:
            strike *= 1.005
        else:
            strike *= 0.995

    premium = bs_call_price(underlying_price, strike, T, RISK_FREE_RATE, sigma)
    premium = max(premium, 0.01)  # Floor at 1 cent

    return {
        'strike': round(strike, 2),
        'premium': round(premium, 4),
        'iv': round(sigma, 4),
        'delta': round(norm.cdf((np.log(underlying_price / strike) +
                   (RISK_FREE_RATE + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))), 4),
        'dte': OPTION_DTE,
    }


def should_exit(position, current_price, current_date):
    """
    Full exit logic from strategy.py v25:
    - Time stop (sector-specific, winner extension)
    - Take profit (regime-conditional)
    - Stop loss (regime-conditional)
    - Failed bounce early exit (low-vol and high-vol variants)
    - Trailing stop (regime-conditional activation + giveback)
    """
    days_held = np.busday_count(
        np.datetime64(position['entry_date'], 'D'),
        np.datetime64(current_date, 'D')
    )

    entry_underlying = position['entry_underlying']
    option_entry = position['option_entry_price']
    peak_option = position.get('peak_option_price', option_entry)
    ticker = position['ticker']

    # Recompute current option price via BS
    T_remaining = max((OPTION_DTE - days_held) / 365.0, 0.001)
    iv = position.get('iv', 0.25)
    strike = position['strike']

    current_option = bs_call_price(current_price, strike, T_remaining,
                                   RISK_FREE_RATE, iv)
    option_return = ((current_option - option_entry) / option_entry
                     if option_entry > 0 else 0)

    # Update peak
    if current_option > peak_option:
        position['peak_option_price'] = current_option
        peak_option = current_option

    peak_return = ((peak_option - option_entry) / option_entry
                   if option_entry > 0 else 0)

    is_highvol_entry = position.get('is_highvol_entry', False)

    # Sector-specific hold period
    sector_hold = SECTOR_HOLD_DAYS.get(ticker, HOLD_DAYS_MAX)
    if is_highvol_entry:
        sector_hold = max(2, sector_hold - HIGHVOL_HOLD_REDUCTION)

    # Time stop with winner extension
    effective_hold = sector_hold
    if option_return > 0:
        effective_hold = sector_hold + WINNER_HOLD_EXTENSION
    if days_held >= effective_hold:
        return True, "time_stop", current_option, option_return

    # Take profit (regime-conditional)
    tp = HIGHVOL_TP_PCT if is_highvol_entry else TP_PCT
    if option_return >= tp:
        return True, "take_profit", current_option, option_return

    # Stop loss (regime-aware)
    sl = HIGHVOL_SL_PCT if is_highvol_entry else SL_PCT
    if option_return <= sl:
        return True, "stop_loss", current_option, option_return

    # Early exit for failed bounces in low-vol
    if not is_highvol_entry:
        early_exit_day = 2 if sector_hold >= 7 else 1
        if days_held >= early_exit_day:
            underlying_below_entry = current_price < entry_underlying
            if underlying_below_entry and option_return <= -0.12:
                return True, "failed_bounce_lowvol", current_option, option_return

    # Early exit for failed bounces in high-vol
    if is_highvol_entry:
        if days_held >= HIGHVOL_EARLY_EXIT_DAY:
            underlying_below_entry = current_price < entry_underlying
            if underlying_below_entry and option_return <= HIGHVOL_EARLY_EXIT_LOSS:
                return True, "failed_bounce_highvol", current_option, option_return

    # Trailing stop (regime-conditional activation + giveback)
    trail_activate = (HIGHVOL_TRAILING_ACTIVATE_PCT if is_highvol_entry
                      else TRAILING_ACTIVATE_PCT)
    if peak_return >= trail_activate:
        giveback = peak_return - option_return
        gb_pct = (HIGHVOL_GIVEBACK_PCT if is_highvol_entry
                  else TRAILING_GIVEBACK_PCT)
        max_giveback = peak_return * gb_pct
        if giveback >= max_giveback:
            return True, "trailing_stop", current_option, option_return

    return False, "", current_option, option_return


# ── Engine Infrastructure ──

def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [OPTIONS-EXEC-AVO] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {
        "positions": [],
        "closed_trades": [],
        "equity_curve": [],
        "capital": ACCOUNT_SIZE,
        "equity": ACCOUNT_SIZE,
        "peak_equity": ACCOUNT_SIZE,
        "max_drawdown": 0.0,
        "total_pnl": 0.0,
        "wins": 0,
        "losses": 0,
        "n_trades": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "created": datetime.now(ET).isoformat(),
        "last_run_date": None,
        "last_updated": None,
    }


def save_state(state: dict):
    state["last_updated"] = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(STATE_FILE)


def log_trade_csv(trade: dict):
    """Append trade to CSV log."""
    file_exists = TRADES_FILE.exists()
    fieldnames = [
        "ticker", "entry_date", "exit_date", "entry_underlying", "exit_underlying",
        "strike", "iv", "option_entry_price", "option_exit_price",
        "contracts", "notional", "pnl", "pnl_pct", "days_held",
        "exit_reason", "regime_at_entry", "vix_at_entry",
    ]
    try:
        with open(TRADES_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade)
    except Exception as e:
        log(f"WARN: CSV log failed: {e}")


def fire_callback(trade_info: dict):
    """Fire signal callback for trade taken."""
    if not CALLBACK_SCRIPT.exists():
        return
    try:
        import os
        env_vars = {
            "ENGINE_NAME": "options_execution_avo",
            "TRADE_TICKER": trade_info.get("ticker", ""),
            "TRADE_DIRECTION": trade_info.get("direction", "long"),
            "TRADE_ACTION": trade_info.get("action", ""),
            "TRADE_PRICE": str(trade_info.get("price", 0)),
            "TRADE_PNL": str(trade_info.get("pnl", 0)),
        }
        full_env = {**os.environ, **env_vars}
        subprocess.Popen(
            [str(CALLBACK_SCRIPT)],
            env=full_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        log(f"WARN: Callback failed: {e}")


def fetch_data():
    """Fetch sector ETFs + SPY + VIX daily data via yfinance."""
    end = datetime.now(ET)
    start = end - timedelta(days=DATA_LOOKBACK_DAYS)

    tickers = PREFERRED_TICKERS + [BENCHMARK, "^VIX"]
    try:
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=30,
        )
    except Exception as e:
        log(f"ERROR: yfinance download failed: {e}")
        return None, None, None

    if raw.empty:
        log("ERROR: yfinance returned empty data")
        return None, None, None

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    vix_series = None
    if "^VIX" in close.columns:
        vix_series = close["^VIX"].dropna()

    spy_series = None
    if BENCHMARK in close.columns:
        spy_series = close[BENCHMARK].dropna()

    sector_prices = close[[c for c in PREFERRED_TICKERS if c in close.columns]].dropna(how="all")

    return sector_prices, spy_series, vix_series


def run():
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    if today.weekday() >= 5:
        log("Weekend -- skipping.")
        return

    log(f"=== Options Execution AVO Paper Engine -- {today_str} ===")
    state = load_state()

    if state.get("last_run_date") == today_str:
        log("Already ran today -- skipping.")
        print_summary(state)
        return

    # Fetch data
    sector_prices, spy_series, vix_series = fetch_data()
    if sector_prices is None or spy_series is None:
        log("ERROR: No price data. Aborting.")
        save_state(state)
        return

    if len(sector_prices) < RSI_PERIOD + 25:
        log(f"ERROR: Insufficient data ({len(sector_prices)} rows). Aborting.")
        save_state(state)
        return

    # Check data freshness
    last_data_date = sector_prices.index[-1]
    if hasattr(last_data_date, 'date'):
        last_data_date = last_data_date.date()
    data_age = (today - last_data_date).days
    if data_age > 3:
        log(f"WARNING: Latest data is {data_age} days old ({last_data_date}).")

    vix_now = float(vix_series.iloc[-1]) if vix_series is not None and len(vix_series) > 0 else None
    vix_str = f"{vix_now:.2f}" if vix_now else "N/A"

    if vix_now and vix_now >= EXTREME_VIX:
        regime = "EXTREME"
    elif vix_now and vix_now >= HIGHVOL_VIX:
        regime = "HIGH-VOL"
    else:
        regime = "LOW-VOL"

    log(f"VIX: {vix_str} | Regime: {regime}")

    # Current prices for each sector
    current_prices = {}
    for etf in PREFERRED_TICKERS:
        if etf in sector_prices.columns:
            val = sector_prices[etf].dropna()
            if len(val) > 0:
                current_prices[etf] = float(val.iloc[-1])

    # Compute IV estimates for each sector
    iv_estimates = {}
    for etf in PREFERRED_TICKERS:
        if etf in sector_prices.columns:
            iv_series = estimate_iv(sector_prices[etf], window=20)
            iv_val = iv_series.dropna()
            if len(iv_val) > 0:
                iv_estimates[etf] = float(iv_val.iloc[-1])
            else:
                iv_estimates[etf] = 0.25  # Default

    # --- 1. Check exits on open positions ---
    still_open = []
    exits_today = []
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker not in current_prices:
            log(f"  WARN: No price for {ticker}, keeping position open")
            still_open.append(pos)
            continue

        price = current_prices[ticker]
        exit_flag, exit_reason, current_option, option_return = should_exit(
            pos, price, today_str)

        if exit_flag:
            # Apply slippage on exit
            exit_option = current_option * (1 - SLIPPAGE_PCT)
            contracts = pos["contracts"]
            notional = pos["notional"]
            # PnL = (exit_premium - entry_premium) * contracts * 100
            pnl = (exit_option - pos["option_entry_price"]) * contracts * 100
            pnl_pct = option_return * 100
            days_held = int(np.busday_count(
                np.datetime64(pos["entry_date"], 'D'),
                np.datetime64(today_str, 'D')))

            state["capital"] += notional + pnl  # Return notional + PnL
            state["total_pnl"] += pnl
            state["n_trades"] += 1
            if pnl >= 0:
                state["wins"] += 1
                state["gross_profit"] += pnl
            else:
                state["losses"] += 1
                state["gross_loss"] += abs(pnl)

            trade_record = {
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "exit_date": today_str,
                "entry_underlying": pos["entry_underlying"],
                "exit_underlying": round(price, 2),
                "strike": pos["strike"],
                "iv": pos["iv"],
                "option_entry_price": pos["option_entry_price"],
                "option_exit_price": round(exit_option, 4),
                "contracts": contracts,
                "notional": round(notional, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "days_held": days_held,
                "exit_reason": exit_reason,
                "regime_at_entry": pos.get("regime_at_entry", "unknown"),
                "vix_at_entry": pos.get("vix_at_entry"),
            }
            state["closed_trades"].append(trade_record)
            state["closed_trades"] = state["closed_trades"][-200:]
            exits_today.append(trade_record)

            log_trade_csv(trade_record)

            log(f"  EXIT {ticker}: ${pnl:+.2f} ({pnl_pct:+.1f}%) after {days_held}d "
                f"[{exit_reason}] opt:{pos['option_entry_price']:.2f}->{exit_option:.2f}")

            fire_callback({
                "ticker": ticker, "action": "exit", "direction": "sell",
                "price": exit_option, "pnl": round(pnl, 2),
            })
        else:
            # Update peak in state
            still_open.append(pos)

    state["positions"] = still_open

    # --- 2. Generate signals and check for new entries ---
    entries_today = []
    n_open = len(state["positions"])

    if n_open < MAX_CONCURRENT and regime != "EXTREME":
        signals = generate_signals(sector_prices, spy_series, vix_series)

        last_idx = signals.index[-1]
        today_signals = signals.loc[last_idx]
        firing = [etf for etf in PREFERRED_TICKERS if today_signals.get(etf, False)]

        if firing:
            log(f"  Signals firing: {', '.join(firing)}")

        held_tickers = {p["ticker"] for p in state["positions"]}

        # HC #807 R2: No re-entry on same ticker within 5 trading days
        # of a recent exit (thesis exhaustion prevention)
        REENTRY_COOLDOWN_DAYS = 5
        recent_exit_tickers = set()
        for ct in state.get("closed_trades", [])[-20:]:
            exit_date = ct.get("exit_date", "")
            if exit_date:
                try:
                    days_since = int(np.busday_count(
                        np.datetime64(exit_date, 'D'),
                        np.datetime64(today_str, 'D')))
                    if days_since <= REENTRY_COOLDOWN_DAYS:
                        recent_exit_tickers.add(ct["ticker"])
                except Exception:
                    pass

        for etf in firing:
            if n_open >= MAX_CONCURRENT:
                log(f"  SKIP {etf}: max concurrent ({MAX_CONCURRENT}) reached")
                break
            if etf in held_tickers:
                log(f"  SKIP {etf}: already holding")
                continue
            if etf in recent_exit_tickers:
                log(f"  SKIP {etf}: recent exit within {REENTRY_COOLDOWN_DAYS}d (HC #807 R2)")
                continue
            if etf not in current_prices:
                continue

            underlying_price = current_prices[etf]
            iv_val = iv_estimates.get(etf, 0.25)

            # Compute option position
            opt = compute_option_position(underlying_price, iv_val, vix_now)

            # Position sizing: 22% of equity per trade
            trade_budget = state["equity"] * MAX_PER_TRADE_PCT
            premium_per_contract = opt['premium'] * 100  # 100 shares per contract

            if premium_per_contract <= 0:
                log(f"  SKIP {etf}: zero premium")
                continue

            contracts = max(1, int(trade_budget / premium_per_contract))
            notional = contracts * opt['premium'] * 100

            # Apply slippage on entry
            entry_premium = opt['premium'] * (1 + SLIPPAGE_PCT)
            notional_with_slippage = contracts * entry_premium * 100

            if notional_with_slippage > state["capital"]:
                # Try fewer contracts
                contracts = max(1, int(state["capital"] / (entry_premium * 100)))
                notional_with_slippage = contracts * entry_premium * 100

            if notional_with_slippage > state["capital"] or state["capital"] < 20:
                log(f"  SKIP {etf}: insufficient capital (${state['capital']:.0f})")
                continue

            state["capital"] -= notional_with_slippage

            is_highvol = iv_val > HIGHVOL_IV_THRESHOLD or (vix_now and vix_now >= HIGHVOL_VIX)

            pos = {
                "ticker": etf,
                "contracts": contracts,
                "notional": round(notional_with_slippage, 2),
                "entry_underlying": round(underlying_price, 2),
                "strike": opt['strike'],
                "iv": opt['iv'],
                "delta": opt['delta'],
                "dte_at_entry": opt['dte'],
                "option_entry_price": round(entry_premium, 4),
                "peak_option_price": round(entry_premium, 4),
                "entry_date": today_str,
                "vix_at_entry": round(vix_now, 2) if vix_now else None,
                "regime_at_entry": regime,
                "is_highvol_entry": is_highvol,
            }
            state["positions"].append(pos)
            held_tickers.add(etf)
            n_open += 1
            entries_today.append(pos)

            log(f"  ENTRY {etf}: {contracts} C @ ${entry_premium:.2f} "
                f"(K={opt['strike']:.0f}, IV={opt['iv']:.0%}, "
                f"d={opt['delta']:.2f}, regime={regime})")

            fire_callback({
                "ticker": etf, "action": "entry", "direction": "long",
                "price": entry_premium, "pnl": 0,
            })
    elif regime == "EXTREME":
        log(f"  VIX {vix_str} >= {EXTREME_VIX} -- no new calls (EXTREME regime)")
    else:
        log(f"  Max concurrent positions ({MAX_CONCURRENT}) -- skipping signal scan")

    # --- 3. Mark to market ---
    portfolio_value = state["capital"]
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker in current_prices:
            price = current_prices[ticker]
            days_held = np.busday_count(
                np.datetime64(pos["entry_date"], 'D'),
                np.datetime64(today_str, 'D'))
            T_remaining = max((OPTION_DTE - days_held) / 365.0, 0.001)
            current_opt = bs_call_price(price, pos["strike"], T_remaining,
                                        RISK_FREE_RATE, pos["iv"])
            mtm_value = current_opt * pos["contracts"] * 100
            portfolio_value += mtm_value
        else:
            portfolio_value += pos["notional"]

    state["equity"] = round(portfolio_value, 2)

    if state["equity"] > state["peak_equity"]:
        state["peak_equity"] = state["equity"]
    current_dd = 0.0
    if state["peak_equity"] > 0:
        current_dd = (state["equity"] - state["peak_equity"]) / state["peak_equity"]
    if current_dd < state["max_drawdown"]:
        state["max_drawdown"] = round(current_dd, 6)

    state["equity_curve"].append({
        "date": today_str,
        "equity": state["equity"],
        "positions": len(state["positions"]),
        "vix": round(vix_now, 2) if vix_now else None,
    })
    state["equity_curve"] = state["equity_curve"][-500:]

    state["last_run_date"] = today_str
    save_state(state)

    # --- 4. Print summary ---
    print_summary(state, entries_today, exits_today, current_prices, today_str)

    log("Done.")


def print_summary(state, entries_today=None, exits_today=None,
                  current_prices=None, today_str=None):
    """Print clean status summary to stdout."""
    entries_today = entries_today or []
    exits_today = exits_today or []

    total_trades = state["n_trades"]
    wr = state["wins"] / total_trades * 100 if total_trades > 0 else 0
    pf = state["gross_profit"] / state["gross_loss"] if state["gross_loss"] > 0 else float('inf')
    avg_win = state["gross_profit"] / state["wins"] if state["wins"] > 0 else 0
    avg_loss = state["gross_loss"] / state["losses"] if state["losses"] > 0 else 0
    return_pct = (state["equity"] / ACCOUNT_SIZE - 1) * 100
    dd_pct = state["max_drawdown"] * 100

    print("\n" + "=" * 60)
    print("  OPTIONS EXECUTION AVO -- Paper Trading Status")
    print("=" * 60)
    print(f"  Equity:     ${state['equity']:,.2f}  ({return_pct:+.2f}%)")
    print(f"  Cash:       ${state['capital']:,.2f}")
    print(f"  Max DD:     {dd_pct:.2f}%")
    print(f"  Total PnL:  ${state['total_pnl']:+,.2f}")
    print("-" * 60)
    print(f"  Trades:     {total_trades}  |  W/L: {state['wins']}/{state['losses']}  |  WR: {wr:.1f}%")
    if total_trades > 0:
        print(f"  PF: {pf:.2f}  |  Avg Win: ${avg_win:.2f}  |  Avg Loss: ${avg_loss:.2f}")
    print("-" * 60)

    if state["positions"]:
        print("  Open Positions:")
        for pos in state["positions"]:
            opt_str = (f"K={pos['strike']:.0f} IV={pos['iv']:.0%}")
            days = ""
            if today_str:
                d = np.busday_count(
                    np.datetime64(pos['entry_date'], 'D'),
                    np.datetime64(today_str, 'D'))
                # Compute current option return
                if current_prices and pos['ticker'] in current_prices:
                    price = current_prices[pos['ticker']]
                    T_rem = max((OPTION_DTE - d) / 365.0, 0.001)
                    cur_opt = bs_call_price(price, pos['strike'], T_rem,
                                            RISK_FREE_RATE, pos['iv'])
                    opt_ret = (cur_opt / pos['option_entry_price'] - 1) * 100
                    days = f"{d}d, {opt_ret:+.1f}%"
                else:
                    days = f"{d}d"
            print(f"    {pos['ticker']:5s}  {pos['contracts']}C @ ${pos['option_entry_price']:.2f}  "
                  f"({opt_str})  [{days}]  {pos.get('regime_at_entry', '?')}")
    else:
        print("  No open positions.")

    if entries_today:
        print(f"\n  Today's Entries: {', '.join(p['ticker'] for p in entries_today)}")
    if exits_today:
        exit_strs = [f"{t['ticker']} ${t['pnl']:+.2f} [{t['exit_reason']}]" for t in exits_today]
        print(f"  Today's Exits:  {', '.join(exit_strs)}")

    print("=" * 60 + "\n")

    log(f"Equity: ${state['equity']:.2f} | Pos: {len(state['positions'])} | "
        f"Trades: {total_trades} | WR: {wr:.0f}% | PF: {pf:.2f} | "
        f"PnL: ${state['total_pnl']:+.2f} | DD: {dd_pct:.2f}%")


if __name__ == "__main__":
    run()
