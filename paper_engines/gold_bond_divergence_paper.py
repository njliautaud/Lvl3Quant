#!/usr/bin/env python3
"""
Gold-Bond Divergence Paper Engine (AVO-evolved, lockbox validated)
==================================================================
AVO score: 6.80 | Lockbox Sharpe: 3.83
Trades sector ETFs based on z-score divergence between gold and bonds.

Two divergence pairs:
1. GLD vs TLT (gold vs bonds) -- primary signal
2. GLD vs UUP (gold vs dollar) -- confirmation
3. GLD vs IEF (gold vs intermediate bonds) -- confirmation

VIX-adaptive parameters for lookback, stops, and take-profit.
SPY trend filter for deflation signals.

Strategy source: AVO run gold_bond_divergence-20260827-014527, step 13.

Cron: 59 16 * * 1-5  (4:59 PM ET, after market close)
State: /home/jupiter/Lvl3Quant/paper_engines/state/gold_bond_divergence_state.json
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
STATE_FILE = STATE_DIR / "gold_bond_divergence_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "gold_bond_divergence.log"
CALLBACK_SCRIPT = BASE / "scripts" / "run_engine_with_callback.sh"

# ── Strategy Parameters (from AVO gold_bond_divergence step 13) ──
ZSCORE_WINDOW = 60

DEFLATION_SECTORS = ['XLU', 'XLP', 'XLV', 'XLF']
INFLATION_SECTORS = ['XLE', 'XLB', 'XLI']
ALL_SECTOR_ETFS = INFLATION_SECTORS + DEFLATION_SECTORS

INITIAL_CAPITAL = 10_000.0
MAX_PER_TRADE = 350.0
MAX_CONCURRENT = 3
MAX_HOLD_DAYS = 8
SLIPPAGE_PCT = 0.0001

# VIX-adaptive stops
TRAILING_STOP_LV = -0.010
TRAILING_STOP_MV = -0.016
TRAILING_STOP_HV = -0.020
TAKE_PROFIT_LV = 0.060
TAKE_PROFIT_MV = 0.060
TAKE_PROFIT_HV = 0.100

SECTOR_MOM_LOOKBACK = 5

# VIX-adaptive divergence lookback
LOOKBACK_LV = 5
LOOKBACK_MV = 10
LOOKBACK_HV = 20

# Z-score thresholds
Z_ENTRY = 0.7
Z_CONFIRM1 = 0.7
Z_CONFIRM2 = 0.4

# Macro tickers needed
MACRO_TICKERS = ['GLD', 'TLT', 'UUP', 'IEF']
BENCHMARK = 'SPY'

# Minimum data lookback (days of history to fetch)
DATA_LOOKBACK_DAYS = 180


# ── Strategy Logic (verbatim from AVO strategy.py) ──

def _compute_zscore_series(series_a, series_b, lookback):
    """Compute z-score of divergence between two return series."""
    ret_a = series_a.pct_change(lookback)
    ret_b = series_b.pct_change(lookback)
    div = ret_a - ret_b
    dm = div.rolling(ZSCORE_WINDOW, min_periods=30).mean()
    ds = div.rolling(ZSCORE_WINDOW, min_periods=30).std()
    return (div - dm) / ds.replace(0, np.nan)


def generate_signals(prices, spy, vix_aligned, macro_data):
    """Multi-pair divergence with trend filter.

    Two divergence pairs:
    1. GLD vs TLT (gold vs bonds) -- original signal
    2. GLD vs UUP (gold vs dollar) -- complementary signal

    Both must agree on direction for a signal to fire.
    SPY must be above 50-day SMA (trend filter) for deflation signals.

    Returns float signals (0.0/1.0).
    """
    sector_etfs = [c for c in prices.columns
                   if c not in ['^VIX', 'SPY', 'TLT', 'IEF', 'GLD', 'USO', 'UUP']
                   and not c.startswith('^')]
    signals = pd.DataFrame(0.0, index=prices.index, columns=sector_etfs)

    gld = macro_data['GLD'] if 'GLD' in macro_data.columns else None
    tlt = macro_data['TLT'] if 'TLT' in macro_data.columns else None
    uup = macro_data['UUP'] if 'UUP' in macro_data.columns else None
    ief = macro_data['IEF'] if 'IEF' in macro_data.columns else None
    if gld is None or tlt is None:
        return signals

    # Pre-compute divergence z-scores for GLD/TLT pair
    zscores_gt = {}
    for lb in [LOOKBACK_LV, LOOKBACK_MV, LOOKBACK_HV]:
        zscores_gt[lb] = _compute_zscore_series(gld, tlt, lb)

    # Pre-compute divergence z-scores for GLD/UUP pair (if available)
    zscores_gu = {}
    if uup is not None:
        for lb in [LOOKBACK_LV, LOOKBACK_MV, LOOKBACK_HV]:
            zscores_gu[lb] = _compute_zscore_series(gld, uup, lb)

    # Pre-compute GLD/IEF z-scores (intermediate bonds, less noisy)
    zscores_gi = {}
    if ief is not None:
        for lb in [LOOKBACK_LV, LOOKBACK_MV, LOOKBACK_HV]:
            zscores_gi[lb] = _compute_zscore_series(gld, ief, lb)

    # Rolling correlation between GLD and TLT returns
    gld_daily = gld.pct_change()
    tlt_daily = tlt.pct_change()
    corr_gt = gld_daily.rolling(30, min_periods=20).corr(tlt_daily)
    corr_gt_median = corr_gt.rolling(120, min_periods=60).median()

    spy_sma10 = spy.rolling(10).mean()
    spy_sma50 = spy.rolling(50).mean()

    spy_mom = spy.pct_change(SECTOR_MOM_LOOKBACK)
    sector_mom_abs = {}
    sector_mom_rel = {}
    for etf in sector_etfs:
        if etf in prices.columns:
            abs_mom = prices[etf].pct_change(SECTOR_MOM_LOOKBACK)
            sector_mom_abs[etf] = abs_mom
            sector_mom_rel[etf] = abs_mom - spy_mom

    for date in prices.index:
        current_vix = vix_aligned.get(date, 20.0)
        if pd.isna(current_vix):
            current_vix = 20.0

        # Select lookback and z-threshold based on VIX regime
        if current_vix < 16:
            lb = LOOKBACK_LV
            z_thresh = 0.8
        elif current_vix > 25:
            lb = LOOKBACK_HV
            z_thresh = 0.6
        else:
            lb = LOOKBACK_MV
            z_thresh = Z_ENTRY

        # GLD/TLT z-score with confirmation
        zscore_gt = zscores_gt[lb]
        if date not in zscore_gt.index:
            continue

        z_gt_raw = zscore_gt.get(date, np.nan)
        if pd.isna(z_gt_raw):
            continue

        date_idx = zscore_gt.index.get_loc(date)
        if date_idx < 2:
            continue
        prev_z1_gt_raw = zscore_gt.iloc[date_idx - 1]
        prev_z2_gt_raw = zscore_gt.iloc[date_idx - 2]
        if pd.isna(prev_z1_gt_raw) or pd.isna(prev_z2_gt_raw):
            continue

        z_gt = z_gt_raw
        prev_z1_gt = prev_z1_gt_raw
        prev_z2_gt = prev_z2_gt_raw

        # GLD/UUP z-score (supplementary confirmation)
        z_gu = 0.0
        has_gu = False
        if lb in zscores_gu:
            zscore_gu = zscores_gu[lb]
            if date in zscore_gu.index:
                z_gu = zscore_gu.get(date, 0.0)
                if not pd.isna(z_gu):
                    has_gu = True
                else:
                    z_gu = 0.0

        # GLD/IEF z-score (intermediate bond confirmation)
        z_gi = 0.0
        has_gi = False
        if lb in zscores_gi:
            zscore_gi = zscores_gi[lb]
            if date in zscore_gi.index:
                z_gi_val = zscore_gi.get(date, 0.0)
                if not pd.isna(z_gi_val):
                    z_gi = z_gi_val
                    has_gi = True

        # Correlation breakdown check
        corr_now = corr_gt.get(date, np.nan) if date in corr_gt.index else np.nan
        corr_med = corr_gt_median.get(date, np.nan) if date in corr_gt_median.index else np.nan
        corr_breaking = False
        if not pd.isna(corr_now) and not pd.isna(corr_med):
            corr_breaking = corr_now < corr_med

        # Correlation breakdown = stronger divergence, lower z-threshold
        eff_z = z_thresh - 0.1 if corr_breaking else z_thresh

        # Inflation signal: gold up, bonds down (z > 0)
        if (z_gt > eff_z and prev_z1_gt > eff_z
                and prev_z2_gt > Z_CONFIRM2):
            if has_gu and z_gu < -0.3:
                pass  # Dollar strengthening contradicts inflation thesis
            elif has_gi and z_gi < -0.3:
                pass  # IEF divergence contradicts TLT divergence
            else:
                for sector in INFLATION_SECTORS:
                    if sector in sector_etfs and sector in sector_mom_abs:
                        mom = sector_mom_abs[sector].get(date, np.nan)
                        if not pd.isna(mom) and mom > 0:
                            signals.loc[date, sector] = 1.0

        # Deflation signal: gold down, bonds up (z < 0)
        elif (z_gt < -eff_z and prev_z1_gt < -eff_z
                and prev_z2_gt < -Z_CONFIRM2):
            spy_price = spy.get(date, np.nan)
            spy_ma10 = spy_sma10.get(date, np.nan)
            spy_ma50 = spy_sma50.get(date, np.nan)
            if not pd.isna(spy_price) and not pd.isna(spy_ma50):
                trend_ok = spy_price > spy_ma50
                if not corr_breaking:
                    trend_ok = trend_ok and (not pd.isna(spy_ma10) and spy_price > spy_ma10)
                if trend_ok:
                    for sector in DEFLATION_SECTORS:
                        if sector in sector_etfs and sector in sector_mom_rel:
                            mom = sector_mom_rel[sector].get(date, np.nan)
                            if not pd.isna(mom) and mom > 0:
                                signals.loc[date, sector] = 1.0

    return signals


def should_exit(pos, current_price, current_date, portfolio_dd, vix_now):
    """VIX-adaptive trailing stop + time-decay stop for underwater positions."""
    entry_price = pos.get('entry_price_adj', pos.get('entry_price', current_price))
    hwm_key = 'hwm' if 'hwm' in pos else 'high_water_mark'
    high_water = pos.get(hwm_key, entry_price)

    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D')
    )
    pnl_pct = (current_price - entry_price) / entry_price

    if days_held >= MAX_HOLD_DAYS:
        return True, "max_hold"

    current_vix = vix_now if vix_now is not None else 20.0

    # Portfolio risk exit
    if portfolio_dd < -0.005 and pnl_pct < 0.0:
        return True, "portfolio_risk"

    # VIX-adaptive take profit
    if current_vix < 16:
        tp_pct = TAKE_PROFIT_LV
    elif current_vix > 25:
        tp_pct = TAKE_PROFIT_HV
    else:
        tp_pct = TAKE_PROFIT_MV

    if pnl_pct >= tp_pct:
        return True, "take_profit"

    # VIX-adaptive trailing stop
    if current_vix < 16:
        stop_pct = TRAILING_STOP_LV
    elif current_vix > 25:
        stop_pct = TRAILING_STOP_HV
    else:
        stop_pct = TRAILING_STOP_MV

    # Breakeven stop: once position was profitable, don't let it go red
    hwm_pnl = (high_water - entry_price) / entry_price if entry_price > 0 else 0
    be_thresh = 0.012 if current_vix < 16 else (0.003 if current_vix > 25 else 0.008)
    if hwm_pnl >= be_thresh and pnl_pct < 0.0:
        return True, "breakeven_stop"

    if high_water > 0:
        dd = (current_price - high_water) / high_water
        if dd <= stop_pct:
            return True, "trailing_stop"

    return False, ""


# ── Engine Infrastructure ──

def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [GOLD-BOND-DIVERGENCE] {msg}"
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
            "ENGINE_NAME": "gold_bond_divergence",
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
    """Fetch sector ETFs + macro tickers + SPY + VIX daily data via yfinance."""
    end = datetime.now(ET)
    start = end - timedelta(days=DATA_LOOKBACK_DAYS)

    tickers = ALL_SECTOR_ETFS + MACRO_TICKERS + [BENCHMARK, "^VIX"]
    # Deduplicate
    tickers = list(dict.fromkeys(tickers))

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
        return None, None, None, None

    if raw.empty:
        log("ERROR: yfinance returned empty data")
        return None, None, None, None

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
    sector_prices = close[[c for c in ALL_SECTOR_ETFS if c in close.columns]].dropna(how="all")

    # Macro data DataFrame (GLD, TLT, UUP, IEF)
    macro_cols = [c for c in MACRO_TICKERS if c in close.columns]
    macro_data = close[macro_cols].dropna(how="all") if macro_cols else pd.DataFrame()

    return sector_prices, spy_series, vix_series, macro_data


def run():
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    # Skip weekends
    if today.weekday() >= 5:
        log("Weekend -- skipping.")
        return

    log(f"=== Gold-Bond Divergence Paper Engine -- {today_str} ===")
    state = load_state()

    # Skip if already ran today
    if state.get("last_run_date") == today_str:
        log("Already ran today -- skipping.")
        print_summary(state)
        return

    # Fetch data
    sector_prices, spy_series, vix_series, macro_data = fetch_data()
    if sector_prices is None or spy_series is None:
        log("ERROR: No price data. Aborting.")
        save_state(state)
        return

    if len(sector_prices) < ZSCORE_WINDOW + 10:
        log(f"ERROR: Insufficient data ({len(sector_prices)} rows, need {ZSCORE_WINDOW + 10}). Aborting.")
        save_state(state)
        return

    if macro_data is None or macro_data.empty or 'GLD' not in macro_data.columns or 'TLT' not in macro_data.columns:
        log("ERROR: Missing GLD/TLT macro data. Aborting.")
        save_state(state)
        return

    # Check for new data
    last_data_date = sector_prices.index[-1]
    if hasattr(last_data_date, 'date'):
        last_data_date = last_data_date.date()
    data_age = (today - last_data_date).days
    if data_age > 3:
        log(f"WARNING: Latest data is {data_age} days old ({last_data_date}). Possible holiday/no new data.")

    vix_now = float(vix_series.iloc[-1]) if vix_series is not None and len(vix_series) > 0 else None
    vix_str = f"{vix_now:.2f}" if vix_now else "N/A"
    if vix_now and vix_now > 25:
        regime = "HIGH-VOL"
    elif vix_now and vix_now >= 16:
        regime = "MID-VOL"
    else:
        regime = "LOW-VOL"
    log(f"VIX: {vix_str} | Regime: {regime}")

    # Align VIX to prices index for signal generation
    vix_aligned = vix_series.reindex(sector_prices.index).ffill().fillna(20.0) if vix_series is not None else pd.Series(20.0, index=sector_prices.index)

    # Get current prices for each sector (last row)
    current_prices = {}
    for etf in ALL_SECTOR_ETFS:
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

        exit_flag, exit_reason = should_exit(pos, price, today_str, portfolio_dd, vix_now)

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
                "regime_at_entry": pos.get("regime_at_entry", "unknown"),
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
        # Build combined prices DataFrame for signal generation
        # Include sector ETFs + macro tickers aligned to same index
        all_price_cols = {}
        for etf in ALL_SECTOR_ETFS:
            if etf in sector_prices.columns:
                all_price_cols[etf] = sector_prices[etf]
        combined_prices = pd.DataFrame(all_price_cols)

        # Align macro_data to same index
        macro_aligned = macro_data.reindex(combined_prices.index).ffill()
        spy_aligned = spy_series.reindex(combined_prices.index).ffill()

        signals = generate_signals(combined_prices, spy_aligned, vix_aligned, macro_aligned)

        # Get today's signals (last row)
        last_idx = signals.index[-1]
        today_signals = signals.loc[last_idx]
        firing = [etf for etf in ALL_SECTOR_ETFS if today_signals.get(etf, 0.0) > 0.5]

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

            price = current_prices[etf]
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
    print("  GOLD-BOND DIVERGENCE -- Paper Trading Status")
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
