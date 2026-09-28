#!/usr/bin/env python3
"""
Vol Compression Sector Rotation Paper Engine (AVO-evolved v17)
===============================================================
Validated strategy: Sharpe 1.28 OOS, AVO-evolved over 17 steps.
Tri-regime (low/mid/high VIX) with vol compression + Bollinger squeeze.

Strategy source: AVO run vol_compression-20260823-052034, step 17.

Cron: 5 16 * * 1-5  (4:05 PM ET, after market close)
State: /home/jupiter/Lvl3Quant/paper_engines/state/vol_compression_avo_state.json
"""
import json
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "paper_engines" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "vol_compression_avo_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "vol_compression_avo.log"
CALLBACK_SCRIPT = BASE / "scripts" / "run_engine_with_callback.sh"

# ── Strategy Parameters (from AVO v17) ──
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP',
               'XLU', 'XLRE', 'XLB']
BENCHMARK = 'SPY'

INITIAL_CAPITAL = 10_000.0
MAX_PER_TRADE = 2_500.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 1  # 1 bps

# Vol compression parameters
VOL_SHORT_WINDOW = 20
VOL_LONG_WINDOW = 60
VOL_COMPRESSION_RATIO = 0.75

# Bollinger bandwidth squeeze
BB_WINDOW = 20
BB_STD = 2.0
BB_BW_PERCENTILE_WINDOW = 60
BB_BW_THRESHOLD = 0.20

# Normal regime filters
MOMENTUM_SMA = 5
RSI_PERIOD = 14
RSI_THRESHOLD_LOWVOL = 48
RSI_THRESHOLD_MIDVOL = 38

# High-vol regime
VIX_HIGH_THRESHOLD = 25
VIX_LOW_THRESHOLD = 16
REL_STRENGTH_LOOKBACK = 10
REL_STRENGTH_MIN = 0.025
HIGH_VOL_SMA = 10

# Trade management
MAX_HOLD_DAYS = 12
TAKE_PROFIT_PCT = 0.0475
HARD_STOP_PCT = -0.035
TRAILING_STOP_INITIAL = -0.025
TRAILING_STOP_FINAL = -0.022
DEAD_MONEY_DAYS = 5
DEAD_MONEY_THRESHOLD = 0.005

# Profit floor
PROFIT_FLOOR_ACTIVATION = 0.020
PROFIT_FLOOR_EXIT_EARLY = 0.005
PROFIT_FLOOR_EXIT_LATE = 0.036
PROFIT_FLOOR_RAMP_DAYS = 8

# Portfolio stress
STRESS_DD_THRESHOLD = -0.0075
STRESS_DEAD_MONEY_DAYS = 3
STRESS_DEAD_MONEY_THRESHOLD = 0.01

# RSI momentum filter
RSI_RISING_LOOKBACK = 3

# Sector dispersion filter
DISPERSION_WINDOW = 10
DISPERSION_PERCENTILE_WINDOW = 60
DISPERSION_THRESHOLD = 0.50

# Cross-sector ranking
MAX_SIGNALS_PER_DAY = 4

# Minimum data lookback (days of history to fetch)
DATA_LOOKBACK_DAYS = 120


# ── Helper Functions (from AVO strategy.py) ──

def realized_vol(series, window):
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bandwidth_squeeze(series, window=20, num_std=2.0, pct_window=60, threshold=0.20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    bandwidth = (2 * num_std * std) / sma
    bw_rank = bandwidth.rolling(pct_window).apply(
        lambda x: (x[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5, raw=True)
    return bw_rank < threshold


def vix_interpolated_rsi_threshold(vix_series):
    frac = (vix_series - VIX_LOW_THRESHOLD) / (VIX_HIGH_THRESHOLD - VIX_LOW_THRESHOLD)
    frac = frac.clip(0.0, 1.0)
    return RSI_THRESHOLD_LOWVOL + frac * (RSI_THRESHOLD_MIDVOL - RSI_THRESHOLD_LOWVOL)


def generate_signals(prices, spy, vix=None):
    """Tri-regime signal generation -- verbatim from AVO v17 strategy.py."""
    signals = pd.DataFrame(False, index=prices.index, columns=SECTOR_ETFS)

    if vix is not None and len(vix) > 0:
        vix_aligned = vix.reindex(prices.index).ffill().fillna(20)
        is_highvol = vix_aligned > VIX_HIGH_THRESHOLD
        is_midvol = (vix_aligned >= VIX_LOW_THRESHOLD) & (vix_aligned <= VIX_HIGH_THRESHOLD)
        is_lowvol = vix_aligned < VIX_LOW_THRESHOLD
    else:
        vix_aligned = pd.Series(20.0, index=prices.index)
        is_highvol = pd.Series(False, index=prices.index)
        is_midvol = pd.Series(True, index=prices.index)
        is_lowvol = pd.Series(False, index=prices.index)

    spy_ret = spy.pct_change(REL_STRENGTH_LOOKBACK)

    # Sector dispersion filter
    sector_cols = [c for c in SECTOR_ETFS if c in prices.columns]
    sector_rets = prices[sector_cols].pct_change(DISPERSION_WINDOW)
    cross_sector_disp = sector_rets.std(axis=1)
    disp_rank = cross_sector_disp.rolling(DISPERSION_PERCENTILE_WINDOW).apply(
        lambda x: (x[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5, raw=True)
    low_dispersion = disp_rank < DISPERSION_THRESHOLD

    midvol_rsi_thresh = vix_interpolated_rsi_threshold(vix_aligned)

    vix_change_2d = vix_aligned.pct_change(2)
    vix_not_rising = vix_change_2d <= 0.02
    vix_change_5d = vix_aligned.pct_change(5)
    vix_stable_lowvol = vix_change_5d <= 0.05

    for etf in SECTOR_ETFS:
        if etf not in prices.columns:
            continue
        p = prices[etf]

        vol_short = realized_vol(p, VOL_SHORT_WINDOW)
        vol_long = realized_vol(p, VOL_LONG_WINDOW)
        vol_ratio = vol_short / vol_long
        vol_compressed = vol_ratio < VOL_COMPRESSION_RATIO

        bb_squeeze = bollinger_bandwidth_squeeze(p, BB_WINDOW, BB_STD,
                                                 BB_BW_PERCENTILE_WINDOW,
                                                 BB_BW_THRESHOLD)
        compressed = vol_compressed | bb_squeeze

        price_sma = p.rolling(MOMENTUM_SMA).mean()
        momentum_up = p > price_sma
        rsi = compute_rsi(p, RSI_PERIOD)

        daily_ret = p.pct_change(1)
        positive_day = daily_ret > 0

        rsi_rising = rsi > rsi.shift(RSI_RISING_LOOKBACK)

        sig_lowvol = (compressed & momentum_up & positive_day & rsi_rising
                      & (rsi < RSI_THRESHOLD_LOWVOL) & vix_stable_lowvol & is_lowvol)

        sig_midvol = (compressed & momentum_up & positive_day & rsi_rising
                      & (rsi < midvol_rsi_thresh) & vix_not_rising & low_dispersion & is_midvol)

        etf_ret = p.pct_change(REL_STRENGTH_LOOKBACK)
        rel_strength = etf_ret - spy_ret
        is_strong = rel_strength > REL_STRENGTH_MIN
        price_sma_hv = p.rolling(HIGH_VOL_SMA).mean()
        above_sma = p > price_sma_hv
        vix_below_recent_high = vix_aligned < vix_aligned.rolling(3).max()
        vix_daily_change = vix_aligned.pct_change(1)
        vix_gentle_rise = vix_daily_change <= 0.04
        vix_filter = vix_below_recent_high | vix_gentle_rise
        sig_highvol = is_strong & above_sma & is_highvol & vix_filter

        signals[etf] = sig_lowvol | sig_midvol | sig_highvol

    # Cross-sector ranking
    rel_str_all = pd.DataFrame(index=prices.index, columns=SECTOR_ETFS, dtype=float)
    for etf in SECTOR_ETFS:
        if etf not in prices.columns:
            continue
        etf_ret = prices[etf].pct_change(REL_STRENGTH_LOOKBACK)
        rel_str_all[etf] = etf_ret - spy_ret

    for day in signals.index:
        active = [etf for etf in SECTOR_ETFS if signals.at[day, etf]]
        if len(active) <= MAX_SIGNALS_PER_DAY:
            continue
        strengths = {etf: rel_str_all.at[day, etf] for etf in active
                     if pd.notna(rel_str_all.at[day, etf])}
        if not strengths:
            continue
        ranked = sorted(strengths, key=strengths.get, reverse=True)
        to_remove = ranked[MAX_SIGNALS_PER_DAY:]
        for etf in to_remove:
            signals.at[day, etf] = False

    return signals


def should_exit(pos, current_price, current_date, portfolio_dd):
    """Exit logic -- verbatim from AVO v17 strategy.py."""
    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D'))
    entry_price = pos.get('entry_price_adj', pos.get('entry_price', current_price))
    pnl_pct = (current_price - entry_price) / entry_price

    hwm_key = 'hwm' if 'hwm' in pos else 'high_water'
    if current_price > pos.get(hwm_key, entry_price):
        pos[hwm_key] = current_price
    high_water = pos.get(hwm_key, entry_price)
    dd = (current_price - high_water) / high_water

    if pnl_pct <= HARD_STOP_PCT:
        return True, "hard_stop"
    if days_held >= MAX_HOLD_DAYS:
        return True, "max_hold"
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True, "take_profit"

    age_frac = min(days_held / MAX_HOLD_DAYS, 1.0)
    trailing_threshold = TRAILING_STOP_INITIAL + age_frac * (TRAILING_STOP_FINAL - TRAILING_STOP_INITIAL)
    if dd <= trailing_threshold:
        return True, "trailing_stop"

    hwm_pnl = (high_water - entry_price) / entry_price
    if hwm_pnl >= PROFIT_FLOOR_ACTIVATION:
        floor_age_frac = min(days_held / PROFIT_FLOOR_RAMP_DAYS, 1.0)
        profit_floor = PROFIT_FLOOR_EXIT_EARLY + floor_age_frac * (PROFIT_FLOOR_EXIT_LATE - PROFIT_FLOOR_EXIT_EARLY)
        if pnl_pct < profit_floor:
            return True, "profit_floor"

    if portfolio_dd < STRESS_DD_THRESHOLD:
        if days_held >= STRESS_DEAD_MONEY_DAYS and pnl_pct < STRESS_DEAD_MONEY_THRESHOLD:
            return True, "stress_dead_money"
    else:
        if days_held >= DEAD_MONEY_DAYS and pnl_pct < DEAD_MONEY_THRESHOLD:
            return True, "dead_money"

    return False, ""


# ── Engine Infrastructure ──

def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [VOL-COMPRESSION-AVO] {msg}"
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
        "capital": INITIAL_CAPITAL,
        "equity": INITIAL_CAPITAL,
        "peak_equity": INITIAL_CAPITAL,
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


def fire_callback(trade_info: dict):
    """Fire signal callback for trade taken."""
    if not CALLBACK_SCRIPT.exists():
        log(f"WARN: Callback script not found, skipping")
        return
    try:
        env_vars = {
            "ENGINE_NAME": "vol_compression_avo",
            "TRADE_TICKER": trade_info.get("ticker", ""),
            "TRADE_DIRECTION": trade_info.get("direction", "long"),
            "TRADE_ACTION": trade_info.get("action", ""),
            "TRADE_PRICE": str(trade_info.get("price", 0)),
            "TRADE_PNL": str(trade_info.get("pnl", 0)),
        }
        import os
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

    tickers = SECTOR_ETFS + [BENCHMARK, "^VIX"]
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

    # Extract VIX as a series
    vix_series = None
    if "^VIX" in close.columns:
        vix_series = close["^VIX"].dropna()

    # SPY series
    spy_series = None
    if BENCHMARK in close.columns:
        spy_series = close[BENCHMARK].dropna()

    # Sector price DataFrame (only sector ETFs)
    sector_prices = close[[c for c in SECTOR_ETFS if c in close.columns]].dropna(how="all")

    return sector_prices, spy_series, vix_series


def run():
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    # Skip weekends
    if today.weekday() >= 5:
        log("Weekend -- skipping.")
        return

    log(f"=== Vol Compression AVO Paper Engine -- {today_str} ===")
    state = load_state()

    # Skip if already ran today
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

    if len(sector_prices) < VOL_LONG_WINDOW + 5:
        log(f"ERROR: Insufficient data ({len(sector_prices)} rows, need {VOL_LONG_WINDOW + 5}). Aborting.")
        save_state(state)
        return

    # Check for new data (last row should be today or very recent)
    last_data_date = sector_prices.index[-1]
    if hasattr(last_data_date, 'date'):
        last_data_date = last_data_date.date()
    data_age = (today - last_data_date).days
    if data_age > 3:
        log(f"WARNING: Latest data is {data_age} days old ({last_data_date}). Possible holiday/no new data.")

    vix_now = float(vix_series.iloc[-1]) if vix_series is not None and len(vix_series) > 0 else None
    vix_str = f"{vix_now:.2f}" if vix_now else "N/A"
    regime = "CRISIS" if (vix_now and vix_now > 40) else ("HIGH-VOL" if (vix_now and vix_now > VIX_HIGH_THRESHOLD) else ("MID-VOL" if (vix_now and vix_now >= VIX_LOW_THRESHOLD) else "LOW-VOL"))
    log(f"VIX: {vix_str} | Regime: {regime}")

    # Get current prices for each sector (last row)
    current_prices = {}
    for etf in SECTOR_ETFS:
        if etf in sector_prices.columns:
            val = sector_prices[etf].dropna()
            if len(val) > 0:
                current_prices[etf] = float(val.iloc[-1])

    # --- 1. Check exits on open positions ---
    # Compute portfolio drawdown for stress mode
    portfolio_dd = 0.0
    if state["peak_equity"] > 0:
        portfolio_dd = (state["equity"] - state["peak_equity"]) / state["peak_equity"]

    still_open = []
    exits_today = []
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker not in current_prices:
            log(f"  WARN: No price for {ticker}, keeping position open")
            still_open.append(pos)
            continue

        price = current_prices[ticker]

        # Update high water mark
        if price > pos.get("hwm", pos["entry_price"]):
            pos["hwm"] = price

        exit_flag, exit_reason = should_exit(pos, price, today_str, portfolio_dd)

        if exit_flag:
            # Apply slippage on exit
            exit_price = price * (1 - SLIPPAGE_BPS / 10_000)
            shares = pos["shares"]
            pnl = (exit_price - pos["entry_price"]) * shares
            pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
            days_held = np.busday_count(
                np.datetime64(pos["entry_date"], 'D'),
                np.datetime64(today_str, 'D'))

            state["capital"] += shares * exit_price
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
                "entry_price": pos["entry_price"],
                "exit_price": round(exit_price, 4),
                "shares": shares,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "days_held": int(days_held),
                "exit_reason": exit_reason,
                "regime_at_entry": pos.get("regime_at_entry", "unknown"),
            }
            state["closed_trades"].append(trade_record)
            state["closed_trades"] = state["closed_trades"][-200:]  # Keep last 200
            exits_today.append(trade_record)

            log(f"  EXIT {ticker}: ${pnl:+.2f} ({pnl_pct:+.2f}%) after {days_held}d [{exit_reason}]")

            fire_callback({
                "ticker": ticker, "action": "exit", "direction": "sell",
                "price": exit_price, "pnl": round(pnl, 2),
            })
        else:
            still_open.append(pos)

    state["positions"] = still_open

    # --- 2. Generate signals and check for new entries ---
    entries_today = []
    n_open = len(state["positions"])

    if n_open < MAX_CONCURRENT:
        signals = generate_signals(sector_prices, spy_series, vix_series)

        # Get today's signals (last row)
        last_idx = signals.index[-1]
        today_signals = signals.loc[last_idx]
        firing = [etf for etf in SECTOR_ETFS if today_signals.get(etf, False)]

        if firing:
            log(f"  Signals firing: {', '.join(firing)}")

        held_tickers = {p["ticker"] for p in state["positions"]}

        for etf in firing:
            if n_open >= MAX_CONCURRENT:
                log(f"  SKIP {etf}: max concurrent ({MAX_CONCURRENT}) reached")
                break
            if etf in held_tickers:
                log(f"  SKIP {etf}: already holding")
                continue
            if etf not in current_prices:
                continue

            # Position sizing: max $2,500 per position, limited by available capital
            price = current_prices[etf]
            entry_price = price * (1 + SLIPPAGE_BPS / 10_000)  # Slippage on entry
            position_size = min(MAX_PER_TRADE, state["capital"] * 0.95)  # Keep 5% cash buffer

            if position_size < 50:
                log(f"  SKIP {etf}: insufficient capital (${state['capital']:.0f})")
                continue

            shares = position_size / entry_price
            cost = shares * entry_price
            state["capital"] -= cost

            pos = {
                "ticker": etf,
                "shares": round(shares, 6),
                "entry_price": round(entry_price, 4),
                "entry_date": today_str,
                "hwm": round(entry_price, 4),
                "vix_at_entry": round(vix_now, 2) if vix_now else None,
                "regime_at_entry": regime,
            }
            state["positions"].append(pos)
            held_tickers.add(etf)
            n_open += 1
            entries_today.append(pos)

            log(f"  ENTRY {etf}: {shares:.4f} shares @ ${entry_price:.2f} (regime={regime})")

            fire_callback({
                "ticker": etf, "action": "entry", "direction": "long",
                "price": entry_price, "pnl": 0,
            })
    else:
        log(f"  Max concurrent positions ({MAX_CONCURRENT}) -- skipping signal scan")

    # --- 3. Mark to market ---
    portfolio_value = state["capital"]
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker in current_prices:
            portfolio_value += pos["shares"] * current_prices[ticker]
        else:
            portfolio_value += pos["shares"] * pos["entry_price"]

    state["equity"] = round(portfolio_value, 2)

    # Update peak equity and drawdown
    if state["equity"] > state["peak_equity"]:
        state["peak_equity"] = state["equity"]
    current_dd = 0.0
    if state["peak_equity"] > 0:
        current_dd = (state["equity"] - state["peak_equity"]) / state["peak_equity"]
    if current_dd < state["max_drawdown"]:
        state["max_drawdown"] = round(current_dd, 6)

    # Record equity curve point
    state["equity_curve"].append({
        "date": today_str,
        "equity": state["equity"],
        "positions": len(state["positions"]),
    })
    state["equity_curve"] = state["equity_curve"][-500:]  # Keep last 500 days

    state["last_run_date"] = today_str
    save_state(state)

    # --- 4. Print summary ---
    print_summary(state, entries_today, exits_today)

    log("Done.")


def print_summary(state, entries_today=None, exits_today=None):
    """Print clean status summary to stdout."""
    entries_today = entries_today or []
    exits_today = exits_today or []

    total_trades = state["n_trades"]
    wr = state["wins"] / total_trades * 100 if total_trades > 0 else 0
    pf = state["gross_profit"] / state["gross_loss"] if state["gross_loss"] > 0 else float('inf')
    avg_win = state["gross_profit"] / state["wins"] if state["wins"] > 0 else 0
    avg_loss = state["gross_loss"] / state["losses"] if state["losses"] > 0 else 0
    return_pct = (state["equity"] / INITIAL_CAPITAL - 1) * 100
    dd_pct = state["max_drawdown"] * 100

    print("\n" + "=" * 60)
    print("  VOL COMPRESSION AVO -- Paper Trading Status")
    print("=" * 60)
    print(f"  Equity:     ${state['equity']:,.2f}  ({return_pct:+.2f}%)")
    print(f"  Cash:       ${state['capital']:,.2f}")
    print(f"  Max DD:     {dd_pct:.2f}%")
    print(f"  Total PnL:  ${state['total_pnl']:+,.2f}")
    print("-" * 60)
    print(f"  Trades:     {total_trades}  |  W/L: {state['wins']}/{state['losses']}  |  WR: {wr:.1f}%")
    print(f"  PF: {pf:.2f}  |  Avg Win: ${avg_win:.2f}  |  Avg Loss: ${avg_loss:.2f}")
    print("-" * 60)

    if state["positions"]:
        print("  Open Positions:")
        for pos in state["positions"]:
            print(f"    {pos['ticker']:5s}  {pos['shares']:.2f} sh @ ${pos['entry_price']:.2f}  "
                  f"(entered {pos['entry_date']}, {pos.get('regime_at_entry', '?')})")
    else:
        print("  No open positions.")

    if entries_today:
        print(f"\n  Today's Entries: {', '.join(p['ticker'] for p in entries_today)}")
    if exits_today:
        exit_strs = [f"{t['ticker']} ${t['pnl']:+.2f}" for t in exits_today]
        print(f"  Today's Exits:  {', '.join(exit_strs)}")

    print("=" * 60 + "\n")

    log(f"Equity: ${state['equity']:.2f} | Positions: {len(state['positions'])} | "
        f"Trades: {total_trades} | WR: {wr:.0f}% | PF: {pf:.2f} | PnL: ${state['total_pnl']:+.2f} | DD: {dd_pct:.2f}%")


if __name__ == "__main__":
    run()
