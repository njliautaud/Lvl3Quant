#!/usr/bin/env python3
"""
Options Overlay Paper Engine — Amplify ALL Validated Strategy Signals via Calls
================================================================================
Reads BUY signals from all lockbox-validated paper engines and expresses them as
ATM call option trades (delta ~0.50, 30-45 DTE) using Black-Scholes pricing.

Backtest result: 34% CAGR (11x amplification over shares-only).

Signal sources (paper_engines/state/):
  - sentiment_contrarian_avo_state.json
  - vol_compression_avo_state.json
  - flow_reversal_2x_paper_state.json
  - momentum_growth_paper_state.json
  - cross_asset_macro_state.json
  - cross_type_confluence_state.json
  - rsi_divergence_paper_state.json
  + agentic_signals.json (unified aggregator)

Exit rules:
  - Take profit: underlying moves +3%
  - Stop loss: option premium drops 50%
  - Theta decay: 21 DTE remaining
  - Trailing stop: option drops 25% from peak value

Cron: 10 16 * * 1-5  (4:10 PM ET, after market close)
State: paper_engines/state/options_overlay_paper_state.json
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance required. pip install yfinance")
    sys.exit(1)

warnings.filterwarnings("ignore")

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "paper_engines" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "options_overlay_paper_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "options_overlay_paper.log"
CALLBACK_SCRIPT = BASE / "scripts" / "run_engine_with_callback.sh"

# ── Strategy Parameters (from validated backtest) ──
INITIAL_CAPITAL = 10_000.0
MAX_ALLOC_PER_TRADE = 0.07       # 7% of equity per option trade
MAX_CONCURRENT = 5               # max open positions
COMMISSION_PER_CONTRACT = 0.65   # per contract, per leg
CONTRACT_MULTIPLIER = 100        # 1 option = 100 shares

# Options pricing
DTE_TARGET = 37                  # ~5 weeks to expiry
RISK_FREE_RATE = 0.045
IV_MULTIPLIER = 1.20             # IV premium over 20d realized vol
TARGET_DELTA = 0.50              # ATM

# Exit rules
UNDERLYING_TP_PCT = 0.03         # underlying +3% -> sell option
OPTION_SL_PCT = -0.50            # premium drops 50% -> stop loss
THETA_EXIT_DTE = 21              # exit at 21 DTE
TRAILING_STOP_PCT = 0.25         # 25% drop from peak option value

# Data
DATA_LOOKBACK_DAYS = 60          # days of history for IV calc

# Signal source state files (relative to paper_engines/state/ or state/)
PE_STATE = BASE / "paper_engines" / "state"
GLOBAL_STATE = BASE / "state"

SIGNAL_SOURCES = {
    "sentiment_contrarian_avo": PE_STATE / "sentiment_contrarian_avo_state.json",
    "vol_compression_avo": PE_STATE / "vol_compression_avo_state.json",
    "flow_reversal_2x": PE_STATE / "flow_reversal_2x_paper_state.json",
    "momentum_growth": PE_STATE / "momentum_growth_paper_state.json",
    "cross_asset_macro": PE_STATE / "cross_asset_macro_state.json",
    "cross_type_confluence": PE_STATE / "cross_type_confluence_state.json",
    "rsi_divergence": PE_STATE / "rsi_divergence_paper_state.json",
    "sector_combined_v10": GLOBAL_STATE / "sector_combined_v10_optimal_paper_state.json",
    "sector_combined_v93": GLOBAL_STATE / "sector_combined_v93_paper_state.json",
    "bond_yield": PE_STATE / "bond_yield_paper_state.json",
    "iv_rv_gap": PE_STATE / "iv_rv_gap_paper_state.json",
    "liquidity_signal": PE_STATE / "liquidity_signal_paper_state.json",
    "vol_term_structure": PE_STATE / "vol_term_structure_state.json",
}

AGENTIC_SIGNALS_FILE = GLOBAL_STATE / "agentic_signals.json"

# Source weights for confluence scoring
SOURCE_WEIGHTS = {
    "sentiment_contrarian_avo": 0.55,  # Lockbox Sharpe 2.73
    "vol_compression_avo": 0.40,       # Lockbox Sharpe 1.28
    "flow_reversal_2x": 0.45,
    "momentum_growth": 0.45,
    "cross_asset_macro": 0.40,
    "cross_type_confluence": 0.40,
    "rsi_divergence": 0.35,
    "sector_combined_v10": 0.45,
    "sector_combined_v93": 0.40,
    "bond_yield": 0.35,
    "iv_rv_gap": 0.35,
    "liquidity_signal": 0.35,
    "vol_term_structure": 0.35,
    "agentic_unified": 0.50,
}


# ── Black-Scholes ──

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_call(S, K, T, r, sigma):
    if T <= 1e-8 or sigma <= 1e-8:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_call_delta(S, K, T, r, sigma):
    if T <= 1e-8 or sigma <= 1e-8:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

def find_strike_for_delta(S, T, r, sigma, target_delta=0.50):
    K_low = S * 0.80
    K_high = S * 1.20
    for _ in range(50):
        K_mid = (K_low + K_high) / 2.0
        d = bs_call_delta(S, K_mid, T, r, sigma)
        if d > target_delta:
            K_low = K_mid
        else:
            K_high = K_mid
    return round((K_low + K_high) / 2.0)

def compute_iv(close_series, window=20):
    """20-day realized vol * IV_MULTIPLIER, floored/capped."""
    log_ret = np.log(close_series / close_series.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252)
    iv = rv * IV_MULTIPLIER
    return iv.clip(lower=0.10, upper=1.50)


# ── Engine Infrastructure ──

def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [OPTIONS-OVERLAY] {msg}"
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
    if not CALLBACK_SCRIPT.exists():
        return
    try:
        import os
        env_vars = {
            "ENGINE_NAME": "options_overlay_paper",
            "TRADE_TICKER": trade_info.get("ticker", ""),
            "TRADE_DIRECTION": trade_info.get("direction", "long_call"),
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


def load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ── Signal Collection ──

def collect_signals() -> list[dict]:
    """
    Read all validated strategy state files. Extract tickers with active
    BUY/long positions. Each becomes a candidate for an options overlay trade.

    Returns list of: {ticker, source, weight, score, entry_date}
    """
    signals = []
    seen_sources = set()

    # 1. Read individual strategy state files
    for source_name, state_path in SIGNAL_SOURCES.items():
        data = load_json(state_path)
        if not data:
            continue

        positions = data.get("positions", [])
        if not positions:
            continue

        seen_sources.add(source_name)
        weight = SOURCE_WEIGHTS.get(source_name, 0.30)

        for pos in positions:
            ticker = pos.get("ticker", "")
            if not ticker:
                continue
            # Only long/buy signals
            direction = pos.get("direction", "long")
            if direction in ("short", "bear"):
                continue

            signals.append({
                "ticker": ticker,
                "source": source_name,
                "weight": weight,
                "score": pos.get("score", weight),
                "entry_date": pos.get("entry_date", ""),
            })

    # 2. Read unified agentic signals
    agentic = load_json(AGENTIC_SIGNALS_FILE)
    if agentic:
        recommendations = agentic.get("recommendations", [])
        if not recommendations:
            # Try alternate structure
            recommendations = agentic.get("signals", [])
        for rec in recommendations:
            ticker = rec.get("ticker", rec.get("symbol", ""))
            if not ticker:
                continue
            direction = rec.get("direction", rec.get("action", "buy"))
            if direction in ("sell", "bear", "short"):
                continue
            confidence = rec.get("confidence", rec.get("score", 0.5))
            if confidence < 0.4:
                continue
            signals.append({
                "ticker": ticker,
                "source": "agentic_unified",
                "weight": SOURCE_WEIGHTS["agentic_unified"],
                "score": float(confidence),
                "entry_date": rec.get("date", ""),
            })

    log(f"Collected {len(signals)} raw signals from {len(seen_sources)} strategy sources")
    return signals


def rank_signals(signals: list[dict]) -> list[dict]:
    """
    Aggregate signals per ticker. Tickers mentioned by multiple strategies
    get higher confluence scores. Return ranked list, strongest first.
    """
    ticker_agg = {}
    for sig in signals:
        tk = sig["ticker"]
        if tk not in ticker_agg:
            ticker_agg[tk] = {"sources": [], "weights": [], "scores": []}
        ticker_agg[tk]["sources"].append(sig["source"])
        ticker_agg[tk]["weights"].append(sig["weight"])
        ticker_agg[tk]["scores"].append(sig["score"])

    ranked = []
    for tk, agg in ticker_agg.items():
        n_sources = len(set(agg["sources"]))
        # Confluence bonus: more sources = higher score
        confluence_bonus = 1.0 + 0.15 * (n_sources - 1)
        avg_weight = np.mean(agg["weights"])
        composite_score = avg_weight * confluence_bonus
        ranked.append({
            "ticker": tk,
            "composite_score": round(composite_score, 4),
            "n_sources": n_sources,
            "sources": list(set(agg["sources"])),
        })

    ranked.sort(key=lambda x: x["composite_score"], reverse=True)
    return ranked


# ── Data Fetch ──

def fetch_prices(tickers: list[str]) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """Fetch daily close prices for tickers + VIX."""
    end = datetime.now(ET)
    start = end - timedelta(days=DATA_LOOKBACK_DAYS)

    all_tickers = list(set(tickers + ["^VIX"]))
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
        return None, None

    if raw.empty:
        log("ERROR: yfinance returned empty data")
        return None, None

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    vix_series = None
    if "^VIX" in close.columns:
        vix_series = close["^VIX"].dropna()

    close = close.drop(columns=["^VIX"], errors="ignore").ffill().dropna(how="all")
    return close, vix_series


# ── Core Engine ──

def run():
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    if today.weekday() >= 5:
        log("Weekend -- skipping.")
        return

    log(f"=== Options Overlay Paper Engine -- {today_str} ===")
    state = load_state()

    if state.get("last_run_date") == today_str:
        log("Already ran today -- skipping.")
        print_summary(state)
        return

    # 1. Collect signals from all validated strategies
    raw_signals = collect_signals()
    ranked = rank_signals(raw_signals)

    if ranked:
        top_strs = [r["ticker"] + "(" + str(r["n_sources"]) + "src)" for r in ranked[:8]]
        log("Top candidates: " + ", ".join(top_strs))

    # Build ticker list: current positions + top candidates
    pos_tickers = [p["ticker"] for p in state["positions"]]
    candidate_tickers = [r["ticker"] for r in ranked[:15]]
    all_tickers = list(set(pos_tickers + candidate_tickers))

    if not all_tickers:
        log("No tickers to process. Done.")
        state["last_run_date"] = today_str
        save_state(state)
        return

    # 2. Fetch price data
    close, vix_series = fetch_prices(all_tickers)
    if close is None:
        log("ERROR: No price data. Aborting.")
        state["last_run_date"] = today_str
        save_state(state)
        return

    vix_now = float(vix_series.iloc[-1]) if vix_series is not None and len(vix_series) > 0 else 20.0
    log(f"VIX: {vix_now:.2f}")

    # Compute IV for available tickers
    iv_dict = {}
    for tk in close.columns:
        iv_series = compute_iv(close[tk])
        if len(iv_series.dropna()) > 0:
            iv_dict[tk] = float(iv_series.dropna().iloc[-1])

    # Current prices (last row)
    current_prices = {}
    for tk in close.columns:
        val = close[tk].dropna()
        if len(val) > 0:
            current_prices[tk] = float(val.iloc[-1])

    # 3. Check exits on open positions
    still_open = []
    exits_today = []

    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker not in current_prices:
            log(f"  WARN: No price for {ticker}, keeping position open")
            still_open.append(pos)
            continue

        price = current_prices[ticker]
        days_held = (today - datetime.fromisoformat(pos["entry_date"]).date()).days
        dte_remain = pos["dte_at_entry"] - days_held
        T_remain = max(dte_remain / 365.0, 1e-8)

        # Current IV
        current_iv = iv_dict.get(ticker, pos["iv"])

        # Reprice the option
        current_option_price = bs_call(
            price, pos["strike"], T_remain,
            RISK_FREE_RATE, current_iv
        )

        # Update peak
        if current_option_price > pos.get("peak_option_price", 0):
            pos["peak_option_price"] = current_option_price

        # Exit checks
        exit_reason = None
        underlying_return = (price - pos["entry_underlying"]) / pos["entry_underlying"]
        option_return = (current_option_price - pos["option_entry_price"]) / pos["option_entry_price"] if pos["option_entry_price"] > 0 else 0

        # 1. Underlying TP: +3%
        if underlying_return >= UNDERLYING_TP_PCT:
            exit_reason = "take_profit"

        # 2. Option SL: -50% of premium
        elif option_return <= OPTION_SL_PCT:
            exit_reason = "stop_loss"

        # 3. Theta decay: 21 DTE
        elif dte_remain <= THETA_EXIT_DTE:
            exit_reason = "theta_exit"

        # 4. Trailing stop: 25% from peak
        elif pos.get("peak_option_price", 0) > 0:
            drawdown_from_peak = (current_option_price - pos["peak_option_price"]) / pos["peak_option_price"]
            if drawdown_from_peak <= -TRAILING_STOP_PCT:
                exit_reason = "trailing_stop"

        if exit_reason:
            contracts = pos["contracts"]
            proceeds = current_option_price * CONTRACT_MULTIPLIER * contracts - COMMISSION_PER_CONTRACT * contracts
            cost_basis = pos["notional"] + COMMISSION_PER_CONTRACT * contracts
            pnl = proceeds - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0

            state["capital"] += proceeds
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
                "exit_underlying": round(price, 4),
                "strike": pos["strike"],
                "iv": pos["iv"],
                "option_entry_price": pos["option_entry_price"],
                "option_exit_price": round(current_option_price, 4),
                "contracts": contracts,
                "notional": pos["notional"],
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "days_held": days_held,
                "exit_reason": exit_reason,
                "sources": pos.get("sources", []),
            }
            state["closed_trades"].append(trade_record)
            state["closed_trades"] = state["closed_trades"][-200:]
            exits_today.append(trade_record)

            log(f"  EXIT {ticker}: ${pnl:+.2f} ({pnl_pct:+.1f}%) after {days_held}d [{exit_reason}]")
            fire_callback({"ticker": ticker, "action": "exit", "direction": "sell", "price": current_option_price, "pnl": round(pnl, 2)})
        else:
            # Update position with current values
            pos["current_option_price"] = round(current_option_price, 4)
            pos["current_underlying"] = round(price, 4)
            pos["current_value"] = round(current_option_price * CONTRACT_MULTIPLIER * pos["contracts"], 2)
            pos["dte_remaining"] = dte_remain
            pos["unrealized_pnl"] = round(
                (current_option_price - pos["option_entry_price"]) * CONTRACT_MULTIPLIER * pos["contracts"], 2
            )
            still_open.append(pos)

    state["positions"] = still_open

    # 4. Open new positions from ranked signals
    entries_today = []
    n_open = len(state["positions"])
    held_tickers = {p["ticker"] for p in state["positions"]}

    # HC #807 R2: No re-entry on same ticker within 5 trading days
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

    if n_open < MAX_CONCURRENT and ranked:
        for candidate in ranked:
            if n_open >= MAX_CONCURRENT:
                break

            ticker = candidate["ticker"]
            if ticker in held_tickers:
                continue
            if ticker in recent_exit_tickers:
                log(f"  SKIP {ticker}: recent exit within {REENTRY_COOLDOWN_DAYS}d (HC #807 R2)")
                continue
            if ticker not in current_prices:
                log(f"  SKIP {ticker}: no price data")
                continue
            if ticker not in iv_dict:
                log(f"  SKIP {ticker}: no IV data")
                continue

            price = current_prices[ticker]
            iv = iv_dict[ticker]

            if price <= 0 or iv <= 0:
                continue

            T = DTE_TARGET / 365.0

            # Find strike for target delta
            strike = find_strike_for_delta(price, T, RISK_FREE_RATE, iv, TARGET_DELTA)

            # Price the call
            option_price = bs_call(price, strike, T, RISK_FREE_RATE, iv)
            if option_price < 0.10:
                log(f"  SKIP {ticker}: option price too low (${option_price:.2f})")
                continue

            delta = bs_call_delta(price, strike, T, RISK_FREE_RATE, iv)

            # Position size: 5-7% of equity
            alloc = state["equity"] * MAX_ALLOC_PER_TRADE
            cost_per_contract = option_price * CONTRACT_MULTIPLIER + COMMISSION_PER_CONTRACT
            num_contracts = max(1, int(alloc / cost_per_contract))

            # Reduce if needed
            total_cost = cost_per_contract * num_contracts
            while total_cost > alloc and num_contracts > 1:
                num_contracts -= 1
                total_cost = cost_per_contract * num_contracts

            # Safety: never risk >20% of equity on one trade
            if total_cost > state["equity"] * 0.20:
                log(f"  SKIP {ticker}: cost ${total_cost:.0f} exceeds 20% of equity")
                continue

            # Check available capital
            if total_cost > state["capital"]:
                log(f"  SKIP {ticker}: insufficient capital (${state['capital']:.0f} < ${total_cost:.0f})")
                continue

            notional = option_price * CONTRACT_MULTIPLIER * num_contracts
            state["capital"] -= total_cost

            pos = {
                "ticker": ticker,
                "contracts": num_contracts,
                "notional": round(notional, 2),
                "entry_underlying": round(price, 4),
                "strike": strike,
                "iv": round(iv, 4),
                "delta": round(delta, 4),
                "dte_at_entry": DTE_TARGET,
                "option_entry_price": round(option_price, 4),
                "peak_option_price": round(option_price, 4),
                "entry_date": today_str,
                "vix_at_entry": round(vix_now, 2),
                "sources": candidate["sources"],
                "n_sources": candidate["n_sources"],
                "composite_score": candidate["composite_score"],
            }
            state["positions"].append(pos)
            held_tickers.add(ticker)
            n_open += 1
            entries_today.append(pos)

            log(f"  ENTRY {ticker}: {num_contracts} contract(s) @ ${option_price:.2f} "
                f"(K={strike}, delta={delta:.2f}, IV={iv:.0%}, {candidate['n_sources']} sources)")
            fire_callback({"ticker": ticker, "action": "entry", "direction": "long_call", "price": option_price, "pnl": 0})
    else:
        if n_open >= MAX_CONCURRENT:
            log(f"  Max concurrent ({MAX_CONCURRENT}) reached -- skipping new entries")

    # 5. Mark to market
    portfolio_value = state["capital"]
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker in current_prices and ticker in iv_dict:
            price = current_prices[ticker]
            days_held = (today - datetime.fromisoformat(pos["entry_date"]).date()).days
            dte_remain = pos["dte_at_entry"] - days_held
            T_remain = max(dte_remain / 365.0, 1e-8)
            current_iv = iv_dict.get(ticker, pos["iv"])
            opt_price = bs_call(price, pos["strike"], T_remain, RISK_FREE_RATE, current_iv)
            portfolio_value += opt_price * CONTRACT_MULTIPLIER * pos["contracts"]
        else:
            portfolio_value += pos.get("notional", 0)

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
        "vix": round(vix_now, 2),
    })
    state["equity_curve"] = state["equity_curve"][-500:]

    state["last_run_date"] = today_str
    save_state(state)

    print_summary(state, entries_today, exits_today)
    log("Done.")


def print_summary(state, entries_today=None, exits_today=None):
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
    print("  OPTIONS OVERLAY -- Paper Trading Status")
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
        print("  Open Options Positions:")
        for pos in state["positions"]:
            unrealized = pos.get("unrealized_pnl", 0)
            dte = pos.get("dte_remaining", pos.get("dte_at_entry", "?"))
            print(f"    {pos['ticker']:5s}  {pos['contracts']}x K={pos['strike']} @ ${pos['option_entry_price']:.2f}  "
                  f"(delta={pos.get('delta', 0):.2f}, DTE={dte}, "
                  f"P&L=${unrealized:+.2f}, {pos.get('n_sources', 1)} src)")
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
