#!/usr/bin/env python3
"""
Vol Regime Mean-Revert Paper Engine (AVO-evolved, score 7.09)
==============================================================
Lockbox validated Sharpe 1.18. Detects SPY realized vol spikes then
compression back (rv_cc_5d/rv_cc_20d ratio), with Parkinson/CC filter
to avoid intraday whipsaw false compressions.

Strategy source: AVO run vol_regime_mean_revert-20260827-051042, step final.

Cron: 56 16 * * 1-5  (4:56 PM ET, after market close)
State: /home/jupiter/Lvl3Quant/paper_engines/state/vol_regime_mean_revert_state.json
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
STATE_FILE = STATE_DIR / "vol_regime_mean_revert_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "vol_regime_mean_revert.log"
CALLBACK_SCRIPT = BASE / "scripts" / "run_engine_with_callback.sh"

# ── Strategy Parameters (from AVO vol_regime_mean_revert) ──
INITIAL_CAPITAL = 10_000.0

# Vol spike/compression detection
SPIKE_THRESHOLD = 1.20
COMPRESSION_THRESHOLD = 0.98
SPIKE_LOOKBACK = 14
TREND_GUARD_THRESHOLD = -0.25
COMPRESSION_CONFIRM_DAYS = 2
RET_D_FLOOR = -0.020
PK_CC_CEILING = 1.10

# Defensive sector ETF signals
EXTRA_ETFS = ['XLU', 'XLP', 'XLV', 'XLRE', 'XLI']
EXTRA_SIGNAL_SPIKE_THRESHOLD = 0.10

# Position sizing
MAX_PER_TRADE = 1500.0
MAX_CONCURRENT = 4
SLIPPAGE_PCT = 0.0001  # 0.01% each way

# Exit parameters — WIDENED from AVO v25 values (HC #812 feedback)
# AVO evaluator uses daily-close alignment which makes -0.7% stops work.
# Real paper trading with daily-close evaluation triggers on normal noise.
# Paper result with old params: 1W/10L (trailing_stop every time).
# Widened 2-3x to survive normal intraday/daily volatility.
TRAILING_STOP_PCT = -0.018       # was -0.007 — widened 2.5x
GAIN_LOCK_THRESHOLD = 0.012      # was 0.005 — lock gains at 1.2%+
GAIN_LOCK_STOP = -0.014          # was -0.006 — tighter once in profit
GAIN_LOCK15_THRESHOLD = 0.020    # was 0.008 — lock at 2%+
GAIN_LOCK15_STOP = -0.010        # was -0.004
GAIN_LOCK2_THRESHOLD = 0.035     # was 0.015 — lock at 3.5%+
GAIN_LOCK2_STOP = -0.005         # was -0.001 — tight lock for big gains
TAKE_PROFIT_PCT = 0.045          # unchanged — 4.5% TP
MAX_HOLD_DAYS = 5                # unchanged
UNDERWATER_CUT_DAYS = 2          # unchanged
PORTFOLIO_DD_EXIT = -0.015       # was -0.007 — widened 2x

# Data lookback for fit() and signal generation
DATA_LOOKBACK_DAYS = 280  # ~252 trading days + buffer

ALL_TICKERS = ['SPY'] + EXTRA_ETFS


# ── Helper Functions ──

def compute_rv_cc(close_series, window):
    """Close-to-close realized vol (annualized)."""
    log_ret = np.log(close_series / close_series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def compute_rv_parkinson(high_series, low_series, window):
    """Parkinson realized vol (annualized)."""
    hl_ratio = np.log(high_series / low_series)
    pk_var = hl_ratio ** 2 / (4 * np.log(2))
    return np.sqrt(pk_var.rolling(window).mean() * 252)


def build_features(spy_df):
    """Build rv_cc_5d, rv_cc_20d, rv_pk_20d, ret_20d, ret_d from SPY OHLC data."""
    df = spy_df.copy()
    df['rv_cc_5d'] = compute_rv_cc(df['Close'], 5)
    df['rv_cc_20d'] = compute_rv_cc(df['Close'], 20)
    if 'High' in df.columns and 'Low' in df.columns:
        df['rv_pk_20d'] = compute_rv_parkinson(df['High'], df['Low'], 20)
    else:
        df['rv_pk_20d'] = np.nan
    df['ret_20d'] = df['Close'].pct_change(20)
    df['ret_d'] = df['Close'].pct_change(1)
    return df


# ── Strategy Logic (verbatim from AVO strategy.py) ──

def fit(train_data):
    """Learn from SPY's vol regime patterns (uses last 252 days)."""
    spy = train_data.sort_index().copy()

    if len(spy) < 30 or 'rv_cc_5d' not in spy.columns or 'rv_cc_20d' not in spy.columns:
        return {'spy_stats': {}}

    spy['rv_ratio'] = spy['rv_cc_5d'] / spy['rv_cc_20d'].clip(lower=1e-8)
    rv_vals = spy['rv_ratio'].values
    close_vals = spy['Close'].values if 'Close' in spy.columns else None

    fwd_returns = []
    n_events = 0
    for i in range(SPIKE_LOOKBACK, len(spy) - 5):
        lb_max = np.nanmax(rv_vals[max(0, i - SPIKE_LOOKBACK):i])
        if lb_max > SPIKE_THRESHOLD and rv_vals[i] < COMPRESSION_THRESHOLD:
            n_events += 1
            if close_vals is not None and i + 5 < len(close_vals):
                fwd = (close_vals[i + 5] - close_vals[i]) / close_vals[i]
                fwd_returns.append(fwd)

    return {
        'spy_stats': {
            'n_events': n_events,
            'avg_fwd': float(np.mean(fwd_returns)) if fwd_returns else 0,
            'hit_rate': float(np.mean([r > 0 for r in fwd_returns])) if fwd_returns else 0,
        }
    }


def generate_signals(spy_features):
    """
    SPY vol compression signals + defensive sector ETFs on strong days.
    Returns list of dicts with 'date', 'etf', 'score' for today's signals.
    """
    spy = spy_features.sort_index().copy()
    if len(spy) < SPIKE_LOOKBACK + 1:
        return []

    if 'rv_cc_5d' not in spy.columns or 'rv_cc_20d' not in spy.columns:
        return []

    spy['rv_ratio'] = spy['rv_cc_5d'] / spy['rv_cc_20d'].clip(lower=1e-8)
    rv_vals = spy['rv_ratio'].values

    # Parkinson/CC ratio filter
    has_pk = 'rv_pk_20d' in spy.columns
    if has_pk:
        spy['pk_cc_ratio'] = spy['rv_pk_20d'] / spy['rv_cc_20d'].clip(lower=1e-8)
        pk_cc_vals = spy['pk_cc_ratio'].values
    else:
        pk_cc_vals = None

    signals = []

    for i in range(SPIKE_LOOKBACK, len(spy)):
        row_idx = spy.index[i]
        curr = rv_vals[i]
        if pd.isna(curr):
            continue

        ret_20d = spy['ret_20d'].iloc[i] if 'ret_20d' in spy.columns else 0
        if not pd.isna(ret_20d) and ret_20d < TREND_GUARD_THRESHOLD:
            continue

        ret_d = spy['ret_d'].iloc[i] if 'ret_d' in spy.columns else 0
        if not pd.isna(ret_d) and ret_d < RET_D_FLOOR:
            continue

        # Skip if Parkinson/CC vol ratio is elevated
        if pk_cc_vals is not None and not pd.isna(pk_cc_vals[i]):
            if pk_cc_vals[i] > PK_CC_CEILING:
                continue

        lb = rv_vals[max(0, i - SPIKE_LOOKBACK):i]
        if len(lb) == 0:
            continue
        lb_max = np.nanmax(lb)

        if lb_max > SPIKE_THRESHOLD and curr < COMPRESSION_THRESHOLD:
            # Compression confirmation: require rv_ratio declining for N days
            if i >= COMPRESSION_CONFIRM_DAYS:
                confirmed = True
                for d in range(1, COMPRESSION_CONFIRM_DAYS + 1):
                    prev_idx = i - d
                    if prev_idx < 0 or pd.isna(rv_vals[prev_idx]):
                        confirmed = False
                        break
                    if d == 1:
                        if rv_vals[i] >= rv_vals[prev_idx]:
                            confirmed = False
                            break
                    else:
                        if rv_vals[i - d + 1] >= rv_vals[prev_idx]:
                            confirmed = False
                            break
                if not confirmed:
                    continue
            else:
                continue

            spike_mag = max(lb_max - SPIKE_THRESHOLD, 0.01)
            comp_speed = max(COMPRESSION_THRESHOLD - curr, 0.01)
            score = spike_mag * comp_speed

            signals.append({
                'date': row_idx,
                'etf': 'SPY',
                'score': score,
            })

            # Defensive sector ETFs on moderate+ spike days
            if spike_mag > EXTRA_SIGNAL_SPIKE_THRESHOLD:
                for etf in EXTRA_ETFS:
                    signals.append({
                        'date': row_idx,
                        'etf': etf,
                        'score': score * 0.85,
                    })

    return signals


def should_exit(pos, current_price, current_date, portfolio_dd):
    """Check if a position should be exited -- verbatim from AVO strategy.py."""
    entry_price = pos.get('entry_price_adj', pos.get('entry_price', current_price))
    high_water = pos.get('high_water_mark', entry_price)
    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D')
    )

    pnl_pct = (current_price - entry_price) / entry_price

    if days_held >= MAX_HOLD_DAYS:
        return True, "max_hold"

    if pnl_pct >= TAKE_PROFIT_PCT:
        return True, "take_profit"

    if high_water > 0:
        drawdown_from_high = (current_price - high_water) / high_water
        # Graduated gain lock: tighter stops as gains increase
        hwm_gain = (high_water - entry_price) / entry_price
        if hwm_gain >= GAIN_LOCK2_THRESHOLD:
            stop = GAIN_LOCK2_STOP
        elif hwm_gain >= GAIN_LOCK15_THRESHOLD:
            stop = GAIN_LOCK15_STOP
        elif hwm_gain >= GAIN_LOCK_THRESHOLD:
            stop = GAIN_LOCK_STOP
        else:
            stop = TRAILING_STOP_PCT
        if drawdown_from_high <= stop:
            return True, "trailing_stop"

    if days_held >= UNDERWATER_CUT_DAYS and pnl_pct < 0:
        return True, "underwater_cut"

    # Portfolio-level risk
    if portfolio_dd < PORTFOLIO_DD_EXIT and pnl_pct < 0:
        return True, "portfolio_dd"

    return False, ""


# ── Engine Infrastructure ──

def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [VOL-REGIME-MR] {msg}"
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
        log("WARN: Callback script not found, skipping")
        return
    try:
        import os
        env_vars = {
            "ENGINE_NAME": "vol_regime_mean_revert",
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
    """Fetch SPY + sector ETFs + VIX daily data via yfinance."""
    end = datetime.now(ET)
    start = end - timedelta(days=DATA_LOOKBACK_DAYS)

    tickers = ALL_TICKERS + ["^VIX"]
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

    # Extract close prices for all tickers
    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    # Extract high/low for SPY (needed for Parkinson vol)
    spy_high = None
    spy_low = None
    if mi:
        high_df = raw["High"] if "High" in raw.columns.get_level_values(0) else None
        low_df = raw["Low"] if "Low" in raw.columns.get_level_values(0) else None
        if high_df is not None and "SPY" in high_df.columns:
            spy_high = high_df["SPY"]
        if low_df is not None and "SPY" in low_df.columns:
            spy_low = low_df["SPY"]

    # VIX series
    vix_series = None
    if "^VIX" in close.columns:
        vix_series = close["^VIX"].dropna()

    # Build SPY features DataFrame
    if "SPY" not in close.columns:
        log("ERROR: SPY data not found")
        return None, None, None

    spy_df = pd.DataFrame({'Close': close['SPY']})
    if spy_high is not None:
        spy_df['High'] = spy_high
    if spy_low is not None:
        spy_df['Low'] = spy_low
    spy_df = spy_df.dropna(subset=['Close'])

    spy_features = build_features(spy_df)

    # Sector ETF close prices (for entries on defensive ETFs)
    etf_prices = {}
    for etf in ALL_TICKERS:
        if etf in close.columns:
            val = close[etf].dropna()
            if len(val) > 0:
                etf_prices[etf] = float(val.iloc[-1])

    return spy_features, etf_prices, vix_series


def run():
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    # Skip weekends
    if today.weekday() >= 5:
        log("Weekend -- skipping.")
        return

    log(f"=== Vol Regime Mean-Revert Paper Engine -- {today_str} ===")
    state = load_state()

    # Skip if already ran today
    if state.get("last_run_date") == today_str:
        log("Already ran today -- skipping.")
        print_summary(state)
        return

    # Fetch data
    spy_features, etf_prices, vix_series = fetch_data()
    if spy_features is None or etf_prices is None:
        log("ERROR: No price data. Aborting.")
        save_state(state)
        return

    if len(spy_features) < 30:
        log(f"ERROR: Insufficient data ({len(spy_features)} rows, need 30+). Aborting.")
        save_state(state)
        return

    # Check data freshness
    last_data_date = spy_features.index[-1]
    if hasattr(last_data_date, 'date'):
        last_data_date = last_data_date.date()
    data_age = (today - last_data_date).days
    if data_age > 3:
        log(f"WARNING: Latest data is {data_age} days old ({last_data_date}). Possible holiday/no new data.")

    vix_now = float(vix_series.iloc[-1]) if vix_series is not None and len(vix_series) > 0 else None
    vix_str = f"{vix_now:.2f}" if vix_now else "N/A"
    log(f"VIX: {vix_str} | SPY: ${etf_prices.get('SPY', 0):.2f}")

    # Run fit() on training data (last 252 trading days)
    params = fit(spy_features)
    stats = params.get('spy_stats', {})
    if stats.get('n_events', 0) > 0:
        log(f"Fit stats: {stats['n_events']} historical events, avg fwd: {stats['avg_fwd']:.4f}, hit rate: {stats['hit_rate']:.2f}")

    # --- 1. Check exits on open positions ---
    portfolio_dd = 0.0
    if state["peak_equity"] > 0:
        portfolio_dd = (state["equity"] - state["peak_equity"]) / state["peak_equity"]

    still_open = []
    exits_today = []
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker not in etf_prices:
            log(f"  WARN: No price for {ticker}, keeping position open")
            still_open.append(pos)
            continue

        price = etf_prices[ticker]

        # Update high water mark
        if price > pos.get("high_water_mark", pos["entry_price"]):
            pos["high_water_mark"] = price

        exit_flag, exit_reason = should_exit(pos, price, today_str, portfolio_dd)

        if exit_flag:
            exit_price = price * (1 - SLIPPAGE_PCT)
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
            }
            state["closed_trades"].append(trade_record)
            state["closed_trades"] = state["closed_trades"][-200:]
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
        all_signals = generate_signals(spy_features)

        # Filter to today's signals only
        today_signals = [s for s in all_signals
                         if (hasattr(s['date'], 'date') and s['date'].date() == today)
                         or str(s['date'])[:10] == today_str]

        # If no signals for today, check last available date (data might be 1 day behind)
        if not today_signals and all_signals:
            last_signal_date = all_signals[-1]['date']
            if hasattr(last_signal_date, 'date'):
                last_signal_date = last_signal_date.date()
            last_signal_str = str(last_signal_date)[:10]
            if data_age <= 1:
                today_signals = [s for s in all_signals
                                 if str(s['date'])[:10] == last_signal_str]

        firing_etfs = list({s['etf'] for s in today_signals})
        # Sort by score descending
        etf_scores = {}
        for s in today_signals:
            if s['etf'] not in etf_scores or s['score'] > etf_scores[s['etf']]:
                etf_scores[s['etf']] = s['score']
        firing_etfs.sort(key=lambda e: etf_scores.get(e, 0), reverse=True)

        if firing_etfs:
            log(f"  Signals firing: {', '.join(firing_etfs)} (scores: {', '.join(f'{e}={etf_scores[e]:.4f}' for e in firing_etfs)})")

        held_tickers = {p["ticker"] for p in state["positions"]}

        # HC #807 R2: No re-entry within 5 trading days
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

        for etf in firing_etfs:
            if n_open >= MAX_CONCURRENT:
                log(f"  SKIP {etf}: max concurrent ({MAX_CONCURRENT}) reached")
                break
            if etf in held_tickers:
                log(f"  SKIP {etf}: already holding")
                continue
            if etf in recent_exit_tickers:
                log(f"  SKIP {etf}: recent exit within {REENTRY_COOLDOWN_DAYS}d (HC #807 R2)")
                continue
            if etf not in etf_prices:
                log(f"  SKIP {etf}: no price data")
                continue

            price = etf_prices[etf]
            entry_price = price * (1 + SLIPPAGE_PCT)
            position_size = min(MAX_PER_TRADE, state["capital"] * 0.95)

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
                "entry_price_adj": round(entry_price, 4),
                "entry_date": today_str,
                "high_water_mark": round(entry_price, 4),
                "signal_score": round(etf_scores.get(etf, 0), 6),
            }
            state["positions"].append(pos)
            held_tickers.add(etf)
            n_open += 1
            entries_today.append(pos)

            log(f"  ENTRY {etf}: {shares:.4f} shares @ ${entry_price:.2f}")

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
        if ticker in etf_prices:
            portfolio_value += pos["shares"] * etf_prices[ticker]
        else:
            portfolio_value += pos["shares"] * pos["entry_price"]

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
    })
    state["equity_curve"] = state["equity_curve"][-500:]

    state["last_run_date"] = today_str
    save_state(state)

    # --- 4. Print summary ---
    print_summary(state, entries_today, exits_today)
    log("Done.")


def print_summary(state, entries_today=None, exits_today=None):
    """Print clean status summary."""
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
    print("  VOL REGIME MEAN-REVERT -- Paper Trading Status")
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
                  f"(entered {pos['entry_date']})")
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
