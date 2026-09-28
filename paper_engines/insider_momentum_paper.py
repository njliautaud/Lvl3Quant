#!/usr/bin/env python3
"""
Insider Momentum Sector Rotation Paper Engine (AVO-evolved, score 6.50)
========================================================================
Defensive sector dip-buy with insider confirmation gate.
Buy XLU/XLP/XLV when they dip >= 1.8% (5d) with positive insider buying z-score.

Strategy source: AVO run insider_momentum-20260823-232706, lockbox validated.

Cron: 57 16 * * 1-5  (4:57 PM ET weekdays, after market close)
State: paper_engines/state/insider_momentum_state.json
"""
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance required. pip install yfinance")
    sys.exit(1)

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "paper_engines" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "insider_momentum_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "insider_momentum_paper.log"

# ── Strategy Parameters (from AVO insider_momentum strategy.py -- verbatim) ──

TRADEABLE_SECTORS = ['XLU', 'XLP', 'XLV']
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP',
               'XLU', 'XLRE', 'XLB']
BENCHMARK = 'SPY'

# Dip-buy parameters
DIP_LOOKBACK = 5
DIP_THRESHOLD = -0.018          # dip >= 1.8%
TREND_LOOKBACK = 30
TREND_MIN = -0.10               # not in major downtrend

# Insider signal parameters (GATE + BOOST)
INSIDER_LOOKBACK = 35
INSIDER_GATE_THRESHOLD = 0.0    # must be non-negative
INSIDER_BOOST_THRESHOLD = 0.5   # conviction boost threshold

# Insider data paths
INSIDER_DIR = BASE / "data" / "feature_store" / "insider_signals"

# Google Trends attention filter
GTRENDS_DIR = BASE / "data" / "feature_store" / "google_trends"
GTRENDS_ATTENTION_GATE = 2.0

# Trade management
MAX_HOLD_DAYS = 3               # max hold for non-winning trades
MAX_HOLD_WINNERS = 4            # let winners run one extra day
UNDERWATER_EXIT_DAYS = 1        # cut losers after 1 day
MAX_CONCURRENT = 1

# Position sizing
INITIAL_CAPITAL = 10_000.0
MAX_PER_TRADE = 2000.0
SLIPPAGE_PCT = 0.0

# Exit parameters
TRAILING_STOP_PCT = -0.02       # wider trail for high-vol moves
TAKE_PROFIT_PCT = 0.035         # wider TP

# Data lookback for indicators
DATA_LOOKBACK_DAYS = 120        # calendar days to fetch

# GICS sector mapping for insider aggregation
SECTOR_TICKERS = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM', 'ORCL', 'AMD', 'ADBE', 'CSCO', 'ACN',
            'INTC', 'IBM', 'INTU', 'NOW', 'QCOM', 'TXN', 'AMAT', 'MU', 'LRCX', 'KLAC'],
    'XLF': ['JPM', 'BRK-B', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI', 'BLK',
            'AXP', 'C', 'SCHW', 'CB', 'MMC', 'PGR', 'ICE', 'AON', 'CME', 'USB'],
    'XLV': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
            'AMGN', 'ISRG', 'SYK', 'GILD', 'MDT', 'VRTX', 'CI', 'ELV', 'BSX', 'REGN'],
    'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'MPC', 'EOG', 'PSX', 'VLO', 'WMB', 'OKE',
            'HES', 'HAL', 'DVN', 'FANG', 'BKR', 'TRGP', 'OXY', 'KMI', 'CTRA', 'APA'],
    'XLI': ['GE', 'CAT', 'UNP', 'HON', 'RTX', 'BA', 'DE', 'LMT', 'UPS', 'ADP',
            'MMM', 'EMR', 'ITW', 'ETN', 'GD', 'TDG', 'NSC', 'WM', 'PH', 'CARR'],
    'XLC': ['META', 'GOOG', 'GOOGL', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'CHTR', 'TMUS'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG',
            'MAR', 'ORLY', 'GM', 'F', 'ROST', 'DHI', 'LEN', 'AZO', 'EBAY', 'YUM'],
    'XLP': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
            'GIS', 'SYY', 'HSY', 'K', 'KHC', 'STZ', 'MKC', 'CHD', 'CAG', 'CLX'],
    'XLU': ['NEE', 'SO', 'DUK', 'D', 'AEP', 'SRE', 'XEL', 'EXC', 'WEC', 'ED',
            'ES', 'AWK', 'DTE', 'ETR', 'PEG', 'FE', 'PPL', 'CMS', 'AES', 'ATO'],
    'XLRE': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'DLR', 'O', 'WELL', 'SPG', 'VICI',
             'AVB', 'EQR', 'ARE', 'MAA', 'UDR', 'VTR', 'HST', 'KIM', 'REG', 'BXP'],
    'XLB': ['LIN', 'SHW', 'APD', 'FCX', 'ECL', 'NEM', 'DOW', 'NUE', 'VMC', 'MLM',
            'PPG', 'DD', 'ALB', 'CF', 'IFF', 'CTVA', 'EMN', 'CE', 'FMC', 'MOS'],
}


# ── Logging ──

def log(msg):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── State Management ──

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "capital": INITIAL_CAPITAL,
        "equity": INITIAL_CAPITAL,
        "positions": [],
        "trades": [],
        "daily_pnl": [],
        "created": datetime.now(ET).isoformat(),
    }


def save_state(state):
    state["updated"] = datetime.now(ET).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Data Fetching ──

def fetch_data():
    """Fetch price data for tradeable sectors + SPY via yfinance."""
    end = datetime.now(ET)
    start = end - timedelta(days=DATA_LOOKBACK_DAYS)

    tickers = TRADEABLE_SECTORS + [BENCHMARK]
    log(f"Fetching data for {len(tickers)} tickers from {start.date()} to {end.date()}")

    data = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                       end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)

    if data.empty:
        log("ERROR: No data returned from yfinance")
        return None, None

    close = data['Close'] if 'Close' in data.columns else data

    spy = close[BENCHMARK] if BENCHMARK in close.columns else None
    etf_cols = [c for c in TRADEABLE_SECTORS if c in close.columns]
    prices = close[etf_cols]

    return prices, spy


def load_insider_data():
    """Load insider signal data from feature store, or return None if unavailable.

    Looks for parquet files in the insider_signals directory with columns:
    ticker, date, net_insider_usd (or gross_buy_usd), optionally n_buyers.
    """
    if not INSIDER_DIR.is_dir():
        log("No insider data directory found, insider gate will pass all (z=0 fallback)")
        return None

    frames = []
    for fpath in INSIDER_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(fpath)
            frames.append(df)
        except Exception:
            continue

    # Also check for a single combined CSV
    csv_path = INSIDER_DIR / "insider_signals.csv"
    if csv_path.exists():
        try:
            frames.append(pd.read_csv(csv_path))
        except Exception:
            pass

    if not frames:
        log("No insider data files found, insider gate will pass all (z=0 fallback)")
        return None

    combined = pd.concat(frames, ignore_index=True)
    if 'ticker' not in combined.columns or 'date' not in combined.columns:
        log("Insider data missing required columns (ticker, date), using z=0 fallback")
        return None

    combined['date'] = pd.to_datetime(combined['date'])
    log(f"Loaded {len(combined)} insider signal rows")
    return combined


def compute_sector_insider_z(insider_data, dates):
    """Aggregate stock insider buys to sector level, compute rolling z-scores."""
    sector_insider_z = {}

    for etf, tickers in SECTOR_TICKERS.items():
        if etf not in TRADEABLE_SECTORS:
            continue

        sector_df = insider_data[insider_data['ticker'].isin(tickers)].copy()
        if sector_df.empty:
            sector_insider_z[etf] = {}
            continue

        # Use net insider USD (buys - sells) for true conviction
        if 'net_insider_usd' in sector_df.columns:
            daily_buys = sector_df.groupby('date')['net_insider_usd'].sum()
        elif 'gross_buy_usd' in sector_df.columns:
            daily_buys = sector_df.groupby('date')['gross_buy_usd'].sum()
        else:
            sector_insider_z[etf] = {}
            continue

        daily_buys = daily_buys.reindex(dates, fill_value=0.0)

        rolling_sum = daily_buys.rolling(INSIDER_LOOKBACK, min_periods=10).sum()
        rolling_mean = rolling_sum.rolling(252, min_periods=60).mean()
        rolling_std = rolling_sum.rolling(252, min_periods=60).std()

        z_score = (rolling_sum - rolling_mean) / rolling_std.clip(lower=1e-6)
        z_score = z_score.fillna(0.0)

        # Penalize z-score when driven by single large transaction
        if 'n_buyers' in sector_df.columns:
            daily_n = sector_df.groupby('date')['n_buyers'].sum()
            daily_n = daily_n.reindex(dates, fill_value=0.0)
            rolling_n = daily_n.rolling(INSIDER_LOOKBACK, min_periods=5).sum()
            breadth_factor = (rolling_n.clip(upper=5) / 5.0).clip(lower=0.3)
            z_score = z_score * breadth_factor

        sector_insider_z[etf] = z_score.to_dict()

    return sector_insider_z


def compute_sector_attention(dates):
    """Aggregate Google Trends attention z-scores to sector level."""
    sector_attention = {}
    gtrends_dir = str(GTRENDS_DIR)
    if not os.path.isdir(gtrends_dir):
        return sector_attention

    for etf in TRADEABLE_SECTORS:
        tickers = SECTOR_TICKERS.get(etf, [])
        all_z = []
        for ticker in tickers:
            fpath = os.path.join(gtrends_dir, f'{ticker}.parquet')
            if not os.path.exists(fpath):
                continue
            try:
                df = pd.read_parquet(fpath)
                df['date'] = pd.to_datetime(df['date'])
                z = df.set_index('date')['search_interest_z'].reindex(dates, method='ffill')
                all_z.append(z)
            except Exception:
                continue
        if all_z:
            sector_z = pd.concat(all_z, axis=1).mean(axis=1)
            sector_attention[etf] = sector_z.to_dict()
        else:
            sector_attention[etf] = {}

    return sector_attention


# ── Strategy Logic (from AVO insider_momentum strategy.py) ──

def check_signals_today(prices, spy, today_idx, insider_data):
    """Check for dip-buy signals with insider gate on today's bar."""
    signals = []
    today_date = prices.index[today_idx]

    # Need enough history for lookbacks
    if today_idx < max(DIP_LOOKBACK, TREND_LOOKBACK) + 1:
        return signals

    # Compute insider z-scores if data available
    has_insider = insider_data is not None and not insider_data.empty
    sector_insider_z = {}
    if has_insider:
        sector_insider_z = compute_sector_insider_z(insider_data, prices.index)

    # Compute attention z-scores
    sector_attention = compute_sector_attention(prices.index)

    candidates = []
    for etf in TRADEABLE_SECTORS:
        if etf not in prices.columns:
            continue

        # Dip check: 5-day return <= -1.8%
        p_now = prices[etf].iloc[today_idx]
        p_dip = prices[etf].iloc[today_idx - DIP_LOOKBACK]
        if pd.isna(p_now) or pd.isna(p_dip) or p_dip <= 0:
            continue
        dip_ret = (p_now - p_dip) / p_dip
        if dip_ret > DIP_THRESHOLD:
            continue

        # Trend guard: 30-day return > -10%
        if today_idx < TREND_LOOKBACK:
            continue
        p_trend = prices[etf].iloc[today_idx - TREND_LOOKBACK]
        if pd.isna(p_trend) or p_trend <= 0:
            continue
        trend_ret = (p_now - p_trend) / p_trend
        if trend_ret < TREND_MIN:
            continue

        # Dip deceleration: skip if today alone drops > 1.2%
        if today_idx >= 1:
            p_prev = prices[etf].iloc[today_idx - 1]
            if not pd.isna(p_prev) and p_prev > 0:
                daily_ret = (p_now - p_prev) / p_prev
                if daily_ret < -0.012:
                    continue

        # Insider GATE: require non-negative z-score
        if has_insider:
            iz = sector_insider_z.get(etf, {}).get(today_date, -1.0)
            if iz < INSIDER_GATE_THRESHOLD:
                continue
        else:
            iz = 0.0  # no insider data = gate passes (neutral)

        # Attention GATE: skip when sector has abnormally high search attention
        if etf in sector_attention:
            att = sector_attention[etf].get(today_date, 0.0)
            if not pd.isna(att) and att > GTRENDS_ATTENTION_GATE:
                continue

        # Score = dip magnitude * trend strength, boosted by insider conviction
        trend_bonus = max(0.0, trend_ret + 0.05) * 10.0
        score = abs(dip_ret) * (1.0 + trend_bonus)
        if iz >= INSIDER_BOOST_THRESHOLD:
            score *= (1.0 + 2.0 * iz)

        candidates.append((etf, score, dip_ret, trend_ret, iz))

    candidates.sort(key=lambda x: x[1], reverse=True)

    for etf, score, dip_ret, trend_ret, iz in candidates[:MAX_CONCURRENT]:
        signals.append({
            "ticker": etf,
            "score": round(score, 4),
            "dip_5d": round(dip_ret, 4),
            "trend_30d": round(trend_ret, 4),
            "insider_z": round(float(iz), 3),
        })

    return signals


def should_exit(pos, current_price, portfolio_dd=None):
    """Exit with winning continuation logic from AVO strategy."""
    entry_date_str = pos["entry_date"]
    entry_date = datetime.fromisoformat(entry_date_str).date() if isinstance(entry_date_str, str) else entry_date_str
    today = datetime.now(ET).date()
    days_held = int(np.busday_count(
        np.datetime64(entry_date, 'D'),
        np.datetime64(today, 'D')))

    entry_price = pos["entry_price"]
    pnl_pct = (current_price - entry_price) / entry_price

    hwm = pos.get("high_water_mark", entry_price)
    if current_price > hwm:
        pos["high_water_mark"] = round(current_price, 4)
        hwm = current_price

    drawdown_from_high = (current_price - hwm) / hwm if hwm > 0 else 0

    reason = None

    # Winning continuation: cut losers after 1 day
    if days_held >= UNDERWATER_EXIT_DAYS and pnl_pct < 0:
        reason = f"underwater_exit ({pnl_pct:+.2%} after {days_held}d)"

    # Asymmetric hold: only clear winners hold longer
    elif days_held >= (MAX_HOLD_WINNERS if pnl_pct > 0.025 else MAX_HOLD_DAYS):
        reason = f"max_hold ({days_held}d, pnl={pnl_pct:+.2%})"

    # Wider TP for extended-hold winners
    else:
        tp = 1.0 if (days_held >= MAX_HOLD_DAYS and pnl_pct > 0.025) else TAKE_PROFIT_PCT
        if pnl_pct >= tp:
            reason = f"take_profit ({pnl_pct:+.2%})"
        elif drawdown_from_high <= TRAILING_STOP_PCT:
            reason = f"trailing_stop (dd={drawdown_from_high:.2%})"

    # Portfolio-level risk: exit breakeven positions when in drawdown
    if reason is None and portfolio_dd is not None and portfolio_dd < -0.02:
        if days_held >= 2 and pnl_pct < 0.005:
            reason = f"portfolio_dd_exit (port_dd={portfolio_dd:.2%}, pnl={pnl_pct:+.2%})"

    return reason


# ── Main Engine ──

def run():
    """Run one daily cycle of the insider momentum paper engine."""
    log("=" * 60)
    log("Insider Momentum Paper Engine -- daily run")
    log("=" * 60)

    state = load_state()
    now = datetime.now(ET)
    today_str = now.strftime("%Y-%m-%d")

    # Skip weekends
    if now.weekday() >= 5:
        log(f"Weekend ({now.strftime('%A')}), skipping.")
        return

    # Check if already ran today
    if state.get("daily_pnl") and state["daily_pnl"][-1].get("date") == today_str:
        log(f"Already ran today ({today_str}), skipping.")
        return

    # Fetch market data
    result = fetch_data()
    if result[0] is None:
        log("ERROR: Failed to fetch data, aborting.")
        return
    prices, spy = result

    if len(prices) < TREND_LOOKBACK + 5:
        log(f"ERROR: Not enough data ({len(prices)} rows, need {TREND_LOOKBACK + 5})")
        return

    today_idx = len(prices) - 1
    today_date = prices.index[today_idx]
    log(f"Latest data date: {today_date.strftime('%Y-%m-%d')}")

    # Load insider data (may be None if unavailable)
    insider_data = load_insider_data()

    # ── Step 1: Check exits on existing positions ──
    exits_today = []
    remaining_positions = []
    realized_pnl = 0.0

    # Compute portfolio drawdown for exit logic
    portfolio_dd = None
    if state.get("daily_pnl") and len(state["daily_pnl"]) >= 5:
        recent_equity = [d["equity"] for d in state["daily_pnl"][-20:]]
        peak = max(recent_equity)
        if peak > 0:
            portfolio_dd = (state["equity"] - peak) / peak

    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker not in prices.columns:
            log(f"WARNING: {ticker} not in price data, keeping position")
            remaining_positions.append(pos)
            continue

        current_price = float(prices[ticker].iloc[today_idx])
        if pd.isna(current_price):
            log(f"WARNING: NaN price for {ticker}, keeping position")
            remaining_positions.append(pos)
            continue

        exit_reason = should_exit(pos, current_price, portfolio_dd)
        if exit_reason:
            exit_price = current_price  # SLIPPAGE_PCT = 0
            shares = pos["shares"]
            trade_pnl = (exit_price - pos["entry_price"]) * shares
            realized_pnl += trade_pnl

            trade_record = {
                "ticker": ticker,
                "side": "sell",
                "entry_date": pos["entry_date"],
                "entry_price": pos["entry_price"],
                "exit_date": today_str,
                "exit_price": round(exit_price, 2),
                "shares": shares,
                "pnl": round(trade_pnl, 2),
                "pnl_pct": round((exit_price - pos["entry_price"]) / pos["entry_price"], 4),
                "reason": exit_reason,
                "insider_z_at_entry": pos.get("insider_z_at_entry", 0.0),
            }
            state["trades"].append(trade_record)
            exits_today.append(trade_record)
            state["capital"] += exit_price * shares

            log(f"EXIT {ticker}: {exit_reason} | PnL: ${trade_pnl:+.2f} "
                f"({trade_record['pnl_pct']:+.2%}) | {shares} shares @ ${exit_price:.2f}")
        else:
            # Update HWM
            if current_price > pos.get("high_water_mark", pos["entry_price"]):
                pos["high_water_mark"] = round(current_price, 4)
            remaining_positions.append(pos)

    state["positions"] = remaining_positions

    # ── Step 2: Check for new signals ──
    entries_today = []

    if len(state["positions"]) < MAX_CONCURRENT:
        signals = check_signals_today(prices, spy, today_idx, insider_data)

        # Filter out tickers we already hold
        held_tickers = {p["ticker"] for p in state["positions"]}
        signals = [s for s in signals if s["ticker"] not in held_tickers]

        slots_available = MAX_CONCURRENT - len(state["positions"])
        signals = signals[:slots_available]

        for sig in signals:
            ticker = sig["ticker"]
            current_price = float(prices[ticker].iloc[today_idx])
            entry_price = current_price  # SLIPPAGE_PCT = 0

            trade_capital = min(MAX_PER_TRADE, state["capital"] * 0.95)
            if trade_capital < 50:
                log(f"Insufficient capital (${state['capital']:.2f}), skipping {ticker}")
                continue

            shares = int(trade_capital / entry_price)
            if shares < 1:
                log(f"Price too high for {ticker} (${entry_price:.2f}), skipping")
                continue

            cost = shares * entry_price
            state["capital"] -= cost

            position = {
                "ticker": ticker,
                "entry_date": today_str,
                "entry_price": round(entry_price, 2),
                "shares": shares,
                "cost": round(cost, 2),
                "high_water_mark": round(entry_price, 2),
                "dip_5d": sig["dip_5d"],
                "trend_30d": sig["trend_30d"],
                "insider_z_at_entry": sig["insider_z"],
                "score": sig["score"],
            }
            state["positions"].append(position)
            entries_today.append(position)

            log(f"ENTRY {ticker}: {shares} shares @ ${entry_price:.2f} "
                f"(${cost:.2f}) | dip_5d={sig['dip_5d']:+.2%} trend_30d={sig['trend_30d']:+.2%} "
                f"insider_z={sig['insider_z']:.3f} score={sig['score']:.4f}")

    # ── Step 3: Mark-to-market ──
    unrealized_pnl = 0.0
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker in prices.columns:
            current_price = float(prices[ticker].iloc[today_idx])
            if not pd.isna(current_price):
                unrealized_pnl += (current_price - pos["entry_price"]) * pos["shares"]

    position_value = sum(
        float(prices[p["ticker"]].iloc[today_idx]) * p["shares"]
        for p in state["positions"]
        if p["ticker"] in prices.columns and not pd.isna(prices[p["ticker"]].iloc[today_idx])
    )
    state["equity"] = round(state["capital"] + position_value, 2)

    # ── Step 4: Daily P&L record ──
    prev_equity = state["daily_pnl"][-1]["equity"] if state["daily_pnl"] else INITIAL_CAPITAL
    daily_change = state["equity"] - prev_equity

    daily_record = {
        "date": today_str,
        "equity": state["equity"],
        "capital": round(state["capital"], 2),
        "daily_pnl": round(daily_change, 2),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "positions_held": len(state["positions"]),
        "entries": len(entries_today),
        "exits": len(exits_today),
    }
    state["daily_pnl"].append(daily_record)

    # Keep last 252 daily records
    if len(state["daily_pnl"]) > 252:
        state["daily_pnl"] = state["daily_pnl"][-252:]

    # ── Step 5: Summary stats ──
    total_trades = len(state["trades"])
    if total_trades > 0:
        wins = sum(1 for t in state["trades"] if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in state["trades"])
        avg_pnl = total_pnl / total_trades
        win_rate = wins / total_trades
    else:
        total_pnl = 0
        avg_pnl = 0
        win_rate = 0

    cum_return = (state["equity"] - INITIAL_CAPITAL) / INITIAL_CAPITAL

    log(f"--- Daily Summary ---")
    log(f"Equity: ${state['equity']:,.2f} ({cum_return:+.2%} total return)")
    log(f"Cash: ${state['capital']:,.2f} | Positions: {len(state['positions'])}")
    log(f"Today: {len(entries_today)} entries, {len(exits_today)} exits, PnL: ${daily_change:+.2f}")
    log(f"All-time: {total_trades} trades, WR: {win_rate:.0%}, Avg PnL: ${avg_pnl:+.2f}")

    if state["positions"]:
        log(f"Open positions:")
        for p in state["positions"]:
            ticker = p["ticker"]
            if ticker in prices.columns:
                curr = float(prices[ticker].iloc[today_idx])
                pos_pnl = (curr - p["entry_price"]) * p["shares"]
                pos_pct = (curr - p["entry_price"]) / p["entry_price"]
                log(f"  {ticker}: {p['shares']} sh @ ${p['entry_price']:.2f} "
                    f"-> ${curr:.2f} ({pos_pct:+.2%}, ${pos_pnl:+.2f})")

    save_state(state)
    log("State saved.")
    log("=" * 60)


if __name__ == "__main__":
    run()
