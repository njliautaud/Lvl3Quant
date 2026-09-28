#!/usr/bin/env python3
"""
Macro Regime Rotation Paper Engine (AVO-evolved v19, score 8.00)
================================================================
Lockbox-validated strategy: Sharpe 4.64 OOS, AVO-evolved over 19 steps.
Defensive sector dip-buy with VIX-adaptive thresholds and macro regime context.

Trades XLU, XLP, XLV, XLRE based on macro regime LEVEL. Uses the macro
dial as a confidence multiplier and dip-quality filter. Fear signals
(VIX term inversion, yield curve inversion, DXY strong) allow shallower dips.

Strategy source: AVO run macro_regime_rotation-20260824-001917, step 19.

Cron: 58 16 * * 1-5  (4:58 PM ET, after market close)
State: /home/jupiter/Lvl3Quant/paper_engines/state/macro_regime_rotation_state.json
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
STATE_FILE = STATE_DIR / "macro_regime_rotation_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "macro_regime_rotation.log"
CALLBACK_SCRIPT = BASE / "scripts" / "run_engine_with_callback.sh"

# ── Strategy Parameters (from AVO v19) ──
TRADEABLE_SECTORS = ['XLU', 'XLP', 'XLV', 'XLRE']  # defensives + real estate

SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP',
               'XLU', 'XLRE', 'XLB']
BENCHMARK = 'SPY'

INITIAL_CAPITAL = 10_000.0

# Dip parameters
DIP_LOOKBACK = 5
DIP_THRESHOLD = -0.015
DIP_THRESHOLD_SPYSTRONG = -0.023  # deeper when SPY rallying hard
DIP_THRESHOLD_HIGHVOL = -0.024    # require deeper dip when VIX > 25
DIP_THRESHOLD_TERM_INV = -0.006   # shallower dip OK during fear signals

# SPY momentum gate
SPY_MOM_LOOKBACK = 20
SPY_MOM_STRONG = 0.05  # SPY 20d return above this = strong rally

# VIX spike signal
VIX_SPIKE_LOOKBACK = 5
VIX_SPIKE_THRESHOLD = 4.0  # VIX up >= 4 points in 5 days = acute fear

# Trend guard
TREND_LOOKBACK = 30
TREND_MIN = -0.10

# Dial parameters
DIAL_SMOOTHING = 5
DIAL_FALLING_THRESHOLD = -0.03

# Dip deceleration
DAILY_DECEL_THRESHOLD = -0.012

# Trade management
MAX_HOLD_DAYS = 3
MAX_PER_TRADE = 2_000.0
MAX_CONCURRENT = 1
SLIPPAGE_BPS = 0.25  # 0.25 bps (SLIPPAGE_PCT=0.000025 from strategy)

# Exit parameters
TRAILING_STOP_PCT = -0.02
TAKE_PROFIT_PCT = 0.035
UNDERWATER_EXIT_DAYS = 1

# Minimum data lookback (days of history to fetch)
DATA_LOOKBACK_DAYS = 90


# ── Helper Functions ──

def compute_macro_dial(vix_series, vix3m_series, tlt_series, shy_series, uup_series):
    """
    Build a simplified macro risk dial from live market data.

    Components (each 0-100, higher = more risk-on):
    - VIX level (inverted: low VIX = risk-on)
    - VIX term structure (contango = risk-on, backwardation = risk-off)
    - Yield curve proxy via TLT/SHY ratio trend
    - DXY proxy via UUP trend (strong dollar = risk-off)

    Returns DataFrame with columns: risk_dial, gate_vix_term_inverted,
    gate_yc_inverted, gate_dxy_strong
    """
    idx = vix_series.index
    result = pd.DataFrame(index=idx)

    # VIX component: scale 10-40 to 100-0
    vix_score = ((40.0 - vix_series.clip(10, 40)) / 30.0 * 100.0).fillna(50)

    # VIX term structure: VIX / VIX3M ratio
    # < 1.0 = contango (normal, risk-on), > 1.0 = backwardation (fear)
    if vix3m_series is not None and len(vix3m_series) > 0:
        vix3m_aligned = vix3m_series.reindex(idx).ffill().fillna(vix_series)
        term_ratio = vix_series / vix3m_aligned
        term_score = ((1.2 - term_ratio.clip(0.7, 1.2)) / 0.5 * 100.0).fillna(50)
        term_inverted = term_ratio > 1.0
    else:
        term_score = pd.Series(50.0, index=idx)
        term_inverted = pd.Series(False, index=idx)

    # Yield curve proxy: TLT/SHY 20d change (falling = flattening/inverting = risk-off)
    if tlt_series is not None and shy_series is not None:
        tlt_a = tlt_series.reindex(idx).ffill()
        shy_a = shy_series.reindex(idx).ffill()
        yc_ratio = tlt_a / shy_a
        yc_change = yc_ratio.pct_change(20).fillna(0)
        # Steepening = positive change = risk-on
        yc_score = ((yc_change.clip(-0.05, 0.05) + 0.05) / 0.10 * 100.0).fillna(50)
        yc_inverted = yc_change < -0.02
    else:
        yc_score = pd.Series(50.0, index=idx)
        yc_inverted = pd.Series(False, index=idx)

    # DXY proxy via UUP: strong dollar = risk-off
    if uup_series is not None:
        uup_a = uup_series.reindex(idx).ffill()
        uup_change = uup_a.pct_change(20).fillna(0)
        # Dollar weakening = positive for risk = risk-on
        dxy_score = ((0.03 - uup_change.clip(-0.03, 0.03)) / 0.06 * 100.0).fillna(50)
        dxy_strong = uup_change > 0.02
    else:
        dxy_score = pd.Series(50.0, index=idx)
        dxy_strong = pd.Series(False, index=idx)

    # Composite dial: equal-weight
    result['risk_dial'] = (vix_score + term_score + yc_score + dxy_score) / 4.0
    result['gate_vix_term_inverted'] = term_inverted
    result['gate_yc_inverted'] = yc_inverted
    result['gate_dxy_strong'] = dxy_strong

    return result


def generate_signals(prices, spy, vix, macro_dial=None):
    """Generate defensive dip-buy signals enhanced by macro regime context.
    Verbatim logic from AVO v19 strategy.py."""
    signals = pd.DataFrame(False, index=prices.index, columns=SECTOR_ETFS)

    # Compute sector returns
    dip_ret = {}
    trend_ret = {}
    daily_ret = {}
    for etf in TRADEABLE_SECTORS:
        if etf in prices.columns:
            dip_ret[etf] = prices[etf].pct_change(DIP_LOOKBACK)
            trend_ret[etf] = prices[etf].pct_change(TREND_LOOKBACK)
            daily_ret[etf] = prices[etf].pct_change(1)

    # SPY momentum for dip quality assessment
    spy_mom = spy.pct_change(SPY_MOM_LOOKBACK)

    # VIX spike detection (absolute change, not pct)
    vix_change = vix.diff(VIX_SPIKE_LOOKBACK)

    # Build macro dial signal and fear signal map
    dial_change = {}
    fear_signal = {}
    if macro_dial is not None and not macro_dial.empty:
        md = macro_dial.copy()
        if 'date' in md.columns:
            md['date'] = pd.to_datetime(md['date'])
            md = md.set_index('date').sort_index()
        smooth_dial = md['risk_dial'].rolling(DIAL_SMOOTHING, min_periods=1).mean()
        dc = smooth_dial.diff(DIAL_SMOOTHING)
        dial_change = dc.to_dict()
        # Fear signal: VIX term inversion OR yield curve inversion OR DXY strong
        vti = md.get('gate_vix_term_inverted', pd.Series(False, index=md.index))
        yci = md.get('gate_yc_inverted', pd.Series(False, index=md.index))
        dxy = md.get('gate_dxy_strong', pd.Series(False, index=md.index))
        for dt in md.index:
            v = vti.get(dt, False) if hasattr(vti, 'get') else (vti.loc[dt] if dt in vti.index else False)
            y = yci.get(dt, False) if hasattr(yci, 'get') else (yci.loc[dt] if dt in yci.index else False)
            d = dxy.get(dt, False) if hasattr(dxy, 'get') else (dxy.loc[dt] if dt in dxy.index else False)
            if ((not pd.isna(v) and v) or (not pd.isna(y) and y)
                    or (not pd.isna(d) and d)):
                fear_signal[dt] = True

    for date in prices.index:
        vix_val = vix.get(date, 20.0) if date in vix.index else 20.0
        is_fear = fear_signal.get(date, False)

        # Get SPY momentum
        sm = spy_mom.get(date, 0.0) if date in spy_mom.index else 0.0
        if pd.isna(sm):
            sm = 0.0
        spy_strong = sm > SPY_MOM_STRONG

        # VIX spike detection
        vc = vix_change.get(date, 0.0) if date in vix_change.index else 0.0
        if pd.isna(vc):
            vc = 0.0
        vix_spike = vc >= VIX_SPIKE_THRESHOLD

        # Five-tier dip threshold:
        # 1. Fear signals (macro_dial): shallower dip OK
        # 2. VIX spike while sub-25: shallower dip OK
        # 3. VIX > 25: deeper dip (high vol = choppy)
        # 4. SPY rallying hard: deeper dip
        # 5. Normal: standard threshold
        if is_fear:
            dip_thresh = DIP_THRESHOLD_TERM_INV
        elif vix_spike and (pd.isna(vix_val) or vix_val <= 25):
            dip_thresh = DIP_THRESHOLD_TERM_INV
        elif not pd.isna(vix_val) and vix_val > 25:
            dip_thresh = DIP_THRESHOLD_HIGHVOL
        elif spy_strong:
            dip_thresh = DIP_THRESHOLD_SPYSTRONG
        else:
            dip_thresh = DIP_THRESHOLD

        candidates = []
        for etf in TRADEABLE_SECTORS:
            # Dip check (adaptive threshold)
            if etf not in dip_ret or date not in dip_ret[etf].index:
                continue
            dv = dip_ret[etf].get(date, np.nan)
            if pd.isna(dv) or dv > dip_thresh:
                continue

            # Trend guard
            if etf not in trend_ret or date not in trend_ret[etf].index:
                continue
            tv = trend_ret[etf].get(date, np.nan)
            if pd.isna(tv) or tv < TREND_MIN:
                continue

            # Dip deceleration filter
            if etf in daily_ret and date in daily_ret[etf].index:
                dr = daily_ret[etf].get(date, 0.0)
                if not pd.isna(dr) and dr < DAILY_DECEL_THRESHOLD:
                    continue

            # Score = dip magnitude with trend bonus
            trend_bonus = max(0.0, tv + 0.05) * 10.0
            score = abs(dv) * (1.0 + trend_bonus)

            # Macro dial bonus: falling dial = risk deteriorating = defensives needed more
            dc_val = dial_change.get(date, 0.0)
            if not pd.isna(dc_val) and dc_val < DIAL_FALLING_THRESHOLD:
                score *= 1.5  # 50% bonus when macro risk is rising

            candidates.append((etf, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        for etf, _ in candidates[:MAX_CONCURRENT]:
            signals.loc[date, etf] = True

    return signals


def should_exit(pos, current_price, current_date, portfolio_dd):
    """Exit with winning continuation: cut underwater positions after 1 day.
    Verbatim logic from AVO v19 strategy.py."""
    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D'))

    entry_price = pos.get('entry_price_adj', pos.get('entry_price', current_price))
    pnl_pct = (current_price - entry_price) / entry_price

    hwm_key = 'hwm' if 'hwm' in pos else 'high_water'
    if current_price > pos.get(hwm_key, entry_price):
        pos[hwm_key] = current_price

    high_water = pos.get(hwm_key, entry_price)
    drawdown_from_high = (current_price - high_water) / high_water

    # Winning continuation: cut losers after 1 day
    if days_held >= UNDERWATER_EXIT_DAYS and pnl_pct < 0:
        return True, "underwater_cut"

    if days_held >= MAX_HOLD_DAYS:
        return True, "max_hold"
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True, "take_profit"
    if drawdown_from_high <= TRAILING_STOP_PCT:
        return True, "trailing_stop"

    return False, ""


# ── Engine Infrastructure ──

def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [MACRO-REGIME-ROTATION] {msg}"
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
            "ENGINE_NAME": "macro_regime_rotation",
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
    """Fetch sector ETFs + SPY + VIX + macro proxy data via yfinance."""
    end = datetime.now(ET)
    start = end - timedelta(days=DATA_LOOKBACK_DAYS)

    # Core tickers + macro proxy tickers
    macro_tickers = ["^VIX", "^VIX3M", "TLT", "SHY", "UUP"]
    all_tickers = SECTOR_ETFS + [BENCHMARK] + macro_tickers

    try:
        raw = yf.download(
            all_tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=30,
        )
    except Exception as e:
        log(f"ERROR: yfinance download failed: {e}")
        return None, None, None, None

    if raw.empty:
        log("ERROR: yfinance returned empty data")
        return None, None, None, None

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    # Extract series
    def get_series(col):
        if col in close.columns:
            return close[col].dropna()
        return None

    vix_series = get_series("^VIX")
    vix3m_series = get_series("^VIX3M")
    spy_series = get_series(BENCHMARK)
    tlt_series = get_series("TLT")
    shy_series = get_series("SHY")
    uup_series = get_series("UUP")

    # Build macro dial from live data
    macro_dial = None
    if vix_series is not None and len(vix_series) > 20:
        macro_dial = compute_macro_dial(vix_series, vix3m_series,
                                        tlt_series, shy_series, uup_series)

    # Sector price DataFrame (only sector ETFs)
    sector_prices = close[[c for c in SECTOR_ETFS if c in close.columns]].dropna(how="all")

    return sector_prices, spy_series, vix_series, macro_dial


def run():
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    # Skip weekends
    if today.weekday() >= 5:
        log("Weekend -- skipping.")
        return

    log(f"=== Macro Regime Rotation Paper Engine -- {today_str} ===")
    state = load_state()

    # Skip if already ran today
    if state.get("last_run_date") == today_str:
        log("Already ran today -- skipping.")
        print_summary(state)
        return

    # Fetch data
    sector_prices, spy_series, vix_series, macro_dial = fetch_data()
    if sector_prices is None or spy_series is None:
        log("ERROR: No price data. Aborting.")
        save_state(state)
        return

    if len(sector_prices) < TREND_LOOKBACK + 5:
        log(f"ERROR: Insufficient data ({len(sector_prices)} rows, need {TREND_LOOKBACK + 5}). Aborting.")
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

    # Determine fear/regime for logging
    fear_active = False
    if macro_dial is not None and len(macro_dial) > 0:
        last_dial = macro_dial.iloc[-1]
        fear_active = (last_dial.get('gate_vix_term_inverted', False)
                       or last_dial.get('gate_yc_inverted', False)
                       or last_dial.get('gate_dxy_strong', False))
        dial_val = last_dial.get('risk_dial', 50)
        log(f"VIX: {vix_str} | Dial: {dial_val:.1f} | Fear: {'YES' if fear_active else 'no'}")
    else:
        log(f"VIX: {vix_str} | Dial: N/A (no macro data)")

    # Get current prices for each tradeable sector (last row)
    current_prices = {}
    for etf in TRADEABLE_SECTORS:
        if etf in sector_prices.columns:
            val = sector_prices[etf].dropna()
            if len(val) > 0:
                current_prices[etf] = float(val.iloc[-1])

    # --- 1. Check exits on open positions ---
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
                "vix_at_entry": pos.get("vix_at_entry"),
                "fear_at_entry": pos.get("fear_at_entry", False),
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
        signals = generate_signals(sector_prices, spy_series, vix_series, macro_dial)

        # Get today's signals (last row)
        last_idx = signals.index[-1]
        today_signals = signals.loc[last_idx]
        firing = [etf for etf in TRADEABLE_SECTORS if today_signals.get(etf, False)]

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

            # Position sizing: max $2,000 per position, limited by available capital
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
                "fear_at_entry": fear_active,
            }
            state["positions"].append(pos)
            held_tickers.add(etf)
            n_open += 1
            entries_today.append(pos)

            log(f"  ENTRY {etf}: {shares:.4f} shares @ ${entry_price:.2f} (VIX={vix_str}, fear={'Y' if fear_active else 'N'})")

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
    print("  MACRO REGIME ROTATION -- Paper Trading Status")
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
