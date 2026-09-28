#!/usr/bin/env python3
"""
IV-Regime Options Backtest — Sector-Relative IV Gating (HC #781)
================================================================

Research question: Do our validated equity dip-buying signals produce
profitable OPTIONS trades when gated by sector-relative IV cheapness?

Hypothesis: Sessions 65-66 options overlays failed because theta decay
destroyed the edge. If we only buy options when IV is cheap FOR THAT
SECTOR (IV rank < 25th percentile), theta cost is small enough for
the directional edge to survive.

Signals tested:
  1. Base Mean Reversion: RSI < 30 + price > 5% below 20-SMA
  2. Bond Yield Signal:   10Y yield drops > 10bps in 5 days + dip
  3. IV-RV Gap Signal:    VIX > realized vol by 5+ pts + dip + RSI<40

Options structure: ~0.38 delta call, ~2% OTM, 14 DTE
Hold: 2-5 days (signal-dependent), exit at TP/SL or expiry
Sizing: $100 max premium per trade

Sector-relative IV regimes (from config/sector_iv_baselines.json):
  - Cheap:    IV rank < sector cheap_rank (typically 20-25)
  - Normal:   cheap_rank <= IV rank < expensive_rank
  - Expensive: IV rank >= sector expensive_rank

Output: /home/jupiter/Lvl3Quant/research_results/iv_regime_options_results.json

Usage:
    python scripts/growth_research/iv_regime_options_backtest.py
"""

import json
import sys
import warnings
import functools
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path("/home/jupiter/Lvl3Quant")
SECTOR_IV_CONFIG = BASE / "config" / "sector_iv_baselines.json"
OUTPUT_FILE = BASE / "research_results" / "iv_regime_options_results.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────
START_DATE = "2023-06-01"      # warmup buffer
BACKTEST_START = "2024-01-01"  # actual backtest start
END_DATE = "2026-07-31"
RISK_FREE = 0.045
COMMISSION_PER_CONTRACT = 0.65  # per leg
CONTRACT_MULT = 100
MAX_PREMIUM = 100.0             # max premium per trade ($)
TARGET_DELTA = 0.38             # ~0.38 delta OTM call
TARGET_OTM_PCT = 0.02           # 2% OTM strike
DEFAULT_DTE = 14                # 14 DTE options
HOLD_DAYS = 5                   # exit after 5 trading days
PROFIT_TARGET_PCT = 0.50        # 50% gain on premium
STOP_LOSS_PCT = -0.60           # -60% loss on premium

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "UNH",
    "V", "MA", "HD", "PG", "JNJ", "LLY", "AVGO", "CRM", "AMD",
    "COST", "NFLX", "ADBE",
]

MACRO_TICKERS = ["SPY", "^VIX", "^TNX", "TLT"]

# ═══════════════════════════════════════════════════════════════════════
# Black-Scholes Pricing
# ═══════════════════════════════════════════════════════════════════════

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price. T in years."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_delta(S, K, T, r, sigma):
    """BS call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1))


def bs_theta_daily(S, K, T, r, sigma):
    """BS call theta (daily, negative = cost per day)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = (
        -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )
    return float(theta / 365.0)


def find_otm_strike(S, otm_pct=0.02):
    """Find strike ~2% OTM for a call (above spot)."""
    raw = S * (1 + otm_pct)
    # Round to nearest 0.5 or 1.0 depending on price level
    if S > 200:
        return round(raw / 5) * 5
    elif S > 50:
        return round(raw)
    else:
        return round(raw * 2) / 2


# ═══════════════════════════════════════════════════════════════════════
# IV Estimation
# ═══════════════════════════════════════════════════════════════════════

def estimate_iv_from_hv(close_series, window=20):
    """
    Estimate implied vol from historical vol (annualized).
    Simple approach: IV ~ HV * 1.15 (IV typically trades at premium to HV).
    Returns annualized IV as decimal (e.g. 0.30 = 30%).
    """
    returns = close_series.pct_change()
    hv = returns.rolling(window).std() * np.sqrt(252)
    iv_est = hv * 1.15  # IV premium over HV
    return iv_est


def compute_iv_rank(iv_series, lookback=252):
    """
    IV Rank: where current IV sits relative to past year's range.
    0 = at the low, 100 = at the high.
    """
    iv_min = iv_series.rolling(lookback, min_periods=60).min()
    iv_max = iv_series.rolling(lookback, min_periods=60).max()
    iv_range = iv_max - iv_min
    iv_rank = ((iv_series - iv_min) / iv_range.replace(0, np.nan)) * 100
    return iv_rank


# ═══════════════════════════════════════════════════════════════════════
# Signal Generators
# ═══════════════════════════════════════════════════════════════════════

def compute_rsi(series, period=14):
    """Standard RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def signal_base_mean_reversion(close_df, ticker):
    """
    Signal 1 — Base Mean Reversion
    BUY when: RSI(14) < 30 AND price > 5% below 20-SMA
    """
    px = close_df[ticker].dropna()
    if len(px) < 60:
        return pd.Series(False, index=close_df.index, dtype=bool)

    rsi = compute_rsi(px)
    sma20 = px.rolling(20).mean()
    pct_below_sma = (px - sma20) / sma20

    signal = (rsi < 30) & (pct_below_sma < -0.05)
    return signal.reindex(close_df.index).fillna(False)


def signal_bond_yield(close_df, ticker):
    """
    Signal 2 — Bond Yield Signal
    BUY when: 10Y yield dropped > 0.10 in 5 days AND stock > 5% below 20-SMA
    """
    px = close_df[ticker].dropna()
    if len(px) < 60:
        return pd.Series(False, index=close_df.index, dtype=bool)

    sma20 = px.rolling(20).mean()
    pct_below_sma = (px - sma20) / sma20

    # Use TNX (10Y yield) if available, else TLT proxy
    if "^TNX" in close_df.columns:
        tnx = close_df["^TNX"].ffill()
        yield_drop = tnx.diff(5)
        bond_cond = yield_drop < -0.10
    elif "TLT" in close_df.columns:
        tlt = close_df["TLT"].ffill()
        tlt_5d = tlt.pct_change(5)
        bond_cond = tlt_5d > 0.01  # TLT up = yields down
    else:
        bond_cond = pd.Series(False, index=close_df.index)

    signal = bond_cond.reindex(px.index).fillna(False) & (pct_below_sma < -0.05)
    return signal.reindex(close_df.index).fillna(False)


def signal_iv_rv_gap(close_df, ticker):
    """
    Signal 3 — IV-RV Gap
    BUY when: VIX > 20-day SPY realized vol (annualized) by 5+ pts
              AND stock > 5% below 20-SMA AND RSI < 40
    """
    px = close_df[ticker].dropna()
    if len(px) < 60:
        return pd.Series(False, index=close_df.index, dtype=bool)

    rsi = compute_rsi(px)
    sma20 = px.rolling(20).mean()
    pct_below_sma = (px - sma20) / sma20

    if "^VIX" not in close_df.columns or "SPY" not in close_df.columns:
        return pd.Series(False, index=close_df.index, dtype=bool)

    vix = close_df["^VIX"].ffill()
    spy = close_df["SPY"].ffill()
    spy_rvol = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100  # annualized %

    iv_rv_gap = vix - spy_rvol
    gap_cond = (iv_rv_gap >= 5.0) & (vix > 20)

    signal = (
        gap_cond.reindex(px.index).fillna(False) &
        (pct_below_sma < -0.05) &
        (rsi < 40)
    )
    return signal.reindex(close_df.index).fillna(False)


SIGNALS = {
    "base_mean_reversion": {
        "func": signal_base_mean_reversion,
        "hold_days": 5,
        "description": "RSI<30 + >5% below 20-SMA",
    },
    "bond_yield": {
        "func": signal_bond_yield,
        "hold_days": 5,
        "description": "10Y yield drop >10bps + dip below SMA",
    },
    "iv_rv_gap": {
        "func": signal_iv_rv_gap,
        "hold_days": 5,
        "description": "VIX > realized vol by 5+ pts + dip + RSI<40",
    },
}

# ═══════════════════════════════════════════════════════════════════════
# Data Download
# ═══════════════════════════════════════════════════════════════════════

def download_data():
    """Download price data with retry logic."""
    import yfinance as yf

    cache_file = BASE / "output" / "growth_research" / "_iv_regime_cache.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        import pickle
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        if "close" in data and len(data["close"]) > 100:
            print(f"  Reused cache: {len(data['close'].columns)} tickers, {len(data['close'])} days")
            return data

    all_tickers = list(set(UNIVERSE + MACRO_TICKERS))
    print(f"\n[DATA] Downloading {len(all_tickers)} tickers from {START_DATE} to {END_DATE}...")

    for attempt in range(3):
        try:
            raw = yf.download(
                all_tickers, start=START_DATE, end=END_DATE,
                auto_adjust=True, progress=False, threads=True,
            )
            if raw is not None and not raw.empty:
                break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    else:
        raise RuntimeError("yfinance download failed after 3 attempts")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        high = raw["High"]
        low = raw["Low"]
    else:
        close = high = low = raw

    close = close.ffill().dropna(how="all")
    data = {
        "close": close,
        "high": high.reindex(close.index).ffill(),
        "low": low.reindex(close.index).ffill(),
    }

    import pickle
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    print(f"  Downloaded: {len(close.columns)} tickers, {len(close)} days")
    return data


# ═══════════════════════════════════════════════════════════════════════
# IV Regime Classification
# ═══════════════════════════════════════════════════════════════════════

def load_sector_config():
    """Load sector IV baselines from config file."""
    with open(SECTOR_IV_CONFIG) as f:
        config = json.load(f)
    return config


def classify_iv_regime(iv_rank_value, ticker, sector_config):
    """
    Classify a single IV rank observation into cheap/normal/expensive
    using sector-relative thresholds.
    """
    ticker_map = sector_config.get("ticker_to_sector", {})
    sector_etf = ticker_map.get(ticker, "XLK")  # default to tech
    sector_info = sector_config.get("sectors", {}).get(sector_etf, {})

    cheap_rank = sector_info.get("cheap_rank", 25)
    expensive_rank = sector_info.get("expensive_rank", 70)

    if np.isnan(iv_rank_value):
        return "unknown"
    elif iv_rank_value < cheap_rank:
        return "cheap"
    elif iv_rank_value < expensive_rank:
        return "normal"
    else:
        return "expensive"


# ═══════════════════════════════════════════════════════════════════════
# Options Trade Simulator
# ═══════════════════════════════════════════════════════════════════════

def simulate_option_trade(
    close_series, entry_idx, entry_date, iv_at_entry, hold_days=5,
):
    """
    Simulate buying a ~0.38 delta call (2% OTM, 14 DTE) and holding.

    Returns dict with trade results including P&L, theta cost, etc.
    """
    S0 = close_series.iloc[entry_idx]
    K = find_otm_strike(S0, TARGET_OTM_PCT)
    T0 = DEFAULT_DTE / 365.0
    sigma = iv_at_entry  # annualized decimal

    if sigma <= 0 or np.isnan(sigma):
        sigma = 0.30  # fallback

    # Entry price
    entry_premium = bs_call_price(S0, K, T0, RISK_FREE, sigma)
    if entry_premium < 0.10:
        return None  # too cheap, unrealistic

    # Number of contracts (max $100 premium)
    n_contracts = max(1, int(MAX_PREMIUM / (entry_premium * CONTRACT_MULT)))
    total_cost = entry_premium * CONTRACT_MULT * n_contracts + COMMISSION_PER_CONTRACT * n_contracts

    # Daily theta at entry
    theta_daily = bs_theta_daily(S0, K, T0, RISK_FREE, sigma) * CONTRACT_MULT * n_contracts

    # Walk forward through holding period
    exit_idx = min(entry_idx + hold_days, len(close_series) - 1)
    best_pnl_pct = -1.0
    worst_pnl_pct = 0.0
    exit_reason = "hold_expiry"
    actual_hold = 0
    cumulative_theta = 0.0

    for day_offset in range(1, hold_days + 1):
        cur_idx = entry_idx + day_offset
        if cur_idx >= len(close_series):
            break

        actual_hold = day_offset
        S_t = close_series.iloc[cur_idx]
        T_t = max(0, (DEFAULT_DTE - day_offset)) / 365.0

        # Price the option at current state
        # IV tends to mean-revert, use a simple model: IV decays 2% per day toward median
        sigma_t = sigma * (1 - 0.005 * day_offset)  # slight IV contraction
        cur_premium = bs_call_price(S_t, K, T_t, RISK_FREE, sigma_t)

        # Daily theta cost tracking
        daily_theta = bs_theta_daily(S_t, K, T_t, RISK_FREE, sigma_t) * CONTRACT_MULT * n_contracts
        cumulative_theta += abs(daily_theta)

        # P&L
        cur_value = cur_premium * CONTRACT_MULT * n_contracts - COMMISSION_PER_CONTRACT * n_contracts
        pnl = cur_value - total_cost
        pnl_pct = pnl / total_cost

        best_pnl_pct = max(best_pnl_pct, pnl_pct)
        worst_pnl_pct = min(worst_pnl_pct, pnl_pct)

        # Check TP/SL
        if pnl_pct >= PROFIT_TARGET_PCT:
            exit_reason = "profit_target"
            break
        if pnl_pct <= STOP_LOSS_PCT:
            exit_reason = "stop_loss"
            break

    # Final P&L
    final_idx = entry_idx + actual_hold
    if final_idx >= len(close_series):
        final_idx = len(close_series) - 1
    S_exit = close_series.iloc[final_idx]
    T_exit = max(0, (DEFAULT_DTE - actual_hold)) / 365.0
    sigma_exit = sigma * (1 - 0.005 * actual_hold)
    exit_premium = bs_call_price(S_exit, K, T_exit, RISK_FREE, sigma_exit)
    exit_value = exit_premium * CONTRACT_MULT * n_contracts - COMMISSION_PER_CONTRACT * n_contracts
    final_pnl = exit_value - total_cost
    final_pnl_pct = final_pnl / total_cost

    # Underlying move
    underlying_return = (S_exit - S0) / S0

    return {
        "entry_date": str(entry_date),
        "exit_date": str(close_series.index[final_idx]),
        "ticker": "",  # filled by caller
        "spot_entry": round(S0, 2),
        "strike": round(K, 2),
        "iv_at_entry": round(sigma * 100, 1),
        "entry_premium": round(entry_premium, 2),
        "exit_premium": round(exit_premium, 2),
        "n_contracts": n_contracts,
        "total_cost": round(total_cost, 2),
        "final_pnl": round(final_pnl, 2),
        "final_pnl_pct": round(final_pnl_pct * 100, 2),
        "underlying_return_pct": round(underlying_return * 100, 2),
        "hold_days": actual_hold,
        "exit_reason": exit_reason,
        "mfe_pct": round(best_pnl_pct * 100, 2),
        "mae_pct": round(worst_pnl_pct * 100, 2),
        "cumulative_theta_cost": round(cumulative_theta, 2),
        "theta_as_pct_of_premium": round(cumulative_theta / total_cost * 100, 2) if total_cost > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main Backtest Loop
# ═══════════════════════════════════════════════════════════════════════

def run_backtest():
    print("=" * 80)
    print("IV-REGIME OPTIONS BACKTEST — Sector-Relative IV Gating (HC #781)")
    print("=" * 80)

    # Load data
    data = download_data()
    close = data["close"]
    sector_config = load_sector_config()

    # Pre-compute IV estimates and IV rank for all tickers
    print("\n[IV] Computing IV estimates and IV rank for all tickers...")
    iv_estimates = {}
    iv_ranks = {}
    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        iv_est = estimate_iv_from_hv(close[ticker])
        iv_rank = compute_iv_rank(iv_est)
        iv_estimates[ticker] = iv_est
        iv_ranks[ticker] = iv_rank

    # Run each signal
    all_results = {}
    backtest_mask = close.index >= BACKTEST_START

    for sig_name, sig_info in SIGNALS.items():
        print(f"\n{'='*60}")
        print(f"SIGNAL: {sig_name} — {sig_info['description']}")
        print(f"{'='*60}")

        signal_trades = []
        signal_func = sig_info["func"]
        hold = sig_info["hold_days"]

        for ticker in UNIVERSE:
            if ticker not in close.columns:
                continue
            if ticker not in iv_estimates:
                continue

            # Generate signals
            signals = signal_func(close, ticker)
            ticker_close = close[ticker].dropna()
            iv_est = iv_estimates[ticker]
            iv_rank = iv_ranks[ticker]

            # Find signal trigger dates within backtest window
            trigger_dates = signals[backtest_mask & signals].index

            # Enforce minimum gap between trades on same ticker
            last_trade_date = None
            for trig_date in trigger_dates:
                if last_trade_date is not None:
                    gap = (trig_date - last_trade_date).days
                    if gap < hold + 2:
                        continue

                # Get idx in ticker_close
                if trig_date not in ticker_close.index:
                    continue
                idx = ticker_close.index.get_loc(trig_date)
                if idx + 1 >= len(ticker_close):
                    continue

                # Get IV at entry
                iv_val = iv_est.get(trig_date, np.nan) if isinstance(iv_est, dict) else iv_est.reindex([trig_date]).iloc[0] if trig_date in iv_est.index else np.nan
                if np.isnan(iv_val) or iv_val <= 0:
                    iv_val = 0.30

                # Get IV rank at entry
                ivr_val = iv_rank.get(trig_date, np.nan) if isinstance(iv_rank, dict) else iv_rank.reindex([trig_date]).iloc[0] if trig_date in iv_rank.index else np.nan

                # Classify IV regime
                regime = classify_iv_regime(ivr_val, ticker, sector_config)

                # Simulate the trade
                trade = simulate_option_trade(
                    ticker_close, idx, trig_date, iv_val, hold_days=hold,
                )
                if trade is None:
                    continue

                trade["ticker"] = ticker
                trade["signal"] = sig_name
                trade["iv_rank"] = round(ivr_val, 1) if not np.isnan(ivr_val) else None
                trade["iv_regime"] = regime

                signal_trades.append(trade)
                last_trade_date = trig_date

        print(f"  Total trades: {len(signal_trades)}")

        # Aggregate by IV regime
        regime_results = {}
        for regime in ["cheap", "normal", "expensive", "unknown"]:
            regime_trades = [t for t in signal_trades if t["iv_regime"] == regime]
            if not regime_trades:
                regime_results[regime] = {
                    "trade_count": 0,
                    "avg_pnl_pct": None,
                    "win_rate": None,
                    "sharpe": None,
                    "avg_theta_cost_pct": None,
                }
                continue

            pnls = [t["final_pnl_pct"] for t in regime_trades]
            wins = sum(1 for p in pnls if p > 0)
            avg_pnl = np.mean(pnls)
            std_pnl = np.std(pnls) if len(pnls) > 1 else 1.0
            sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
            avg_theta = np.mean([t["theta_as_pct_of_premium"] for t in regime_trades])
            avg_underlying = np.mean([t["underlying_return_pct"] for t in regime_trades])
            avg_hold = np.mean([t["hold_days"] for t in regime_trades])

            # Exit reason distribution
            exit_reasons = defaultdict(int)
            for t in regime_trades:
                exit_reasons[t["exit_reason"]] += 1

            regime_results[regime] = {
                "trade_count": len(regime_trades),
                "avg_pnl_pct": round(avg_pnl, 2),
                "median_pnl_pct": round(np.median(pnls), 2),
                "win_rate": round(wins / len(regime_trades) * 100, 1),
                "sharpe": round(sharpe, 3),
                "avg_theta_cost_pct": round(avg_theta, 2),
                "avg_underlying_return_pct": round(avg_underlying, 2),
                "avg_hold_days": round(avg_hold, 1),
                "max_gain_pct": round(max(pnls), 2),
                "max_loss_pct": round(min(pnls), 2),
                "exit_reasons": dict(exit_reasons),
            }

            print(f"\n  [{regime.upper()}] {len(regime_trades)} trades:")
            print(f"    Avg P&L: {avg_pnl:+.2f}% | Win Rate: {wins}/{len(regime_trades)} ({wins/len(regime_trades)*100:.0f}%)")
            print(f"    Sharpe: {sharpe:.3f} | Avg Theta Cost: {avg_theta:.1f}% of premium")
            print(f"    Avg Underlying Move: {avg_underlying:+.2f}%")

        all_results[sig_name] = {
            "description": sig_info["description"],
            "total_trades": len(signal_trades),
            "regimes": regime_results,
            "all_trades": signal_trades,  # keep full detail
        }

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: IV REGIME IMPACT ON OPTIONS P&L")
    print("=" * 80)

    summary_table = []
    for sig_name, result in all_results.items():
        for regime in ["cheap", "normal", "expensive"]:
            r = result["regimes"].get(regime, {})
            if r.get("trade_count", 0) == 0:
                continue
            summary_table.append({
                "signal": sig_name,
                "iv_regime": regime,
                "trades": r["trade_count"],
                "avg_pnl": r["avg_pnl_pct"],
                "win_rate": r["win_rate"],
                "sharpe": r["sharpe"],
                "theta_cost": r["avg_theta_cost_pct"],
            })

    if summary_table:
        df_summary = pd.DataFrame(summary_table)
        print(df_summary.to_string(index=False))

    # ── Hypothesis Test ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("HYPOTHESIS TEST: Cheap IV vs Expensive IV")
    print("=" * 80)

    for sig_name, result in all_results.items():
        cheap = result["regimes"].get("cheap", {})
        expensive = result["regimes"].get("expensive", {})
        normal = result["regimes"].get("normal", {})

        cheap_n = cheap.get("trade_count", 0)
        exp_n = expensive.get("trade_count", 0)

        print(f"\n  {sig_name}:")
        if cheap_n > 0 and exp_n > 0:
            cheap_pnl = cheap["avg_pnl_pct"]
            exp_pnl = expensive["avg_pnl_pct"]
            spread = cheap_pnl - exp_pnl
            print(f"    Cheap IV avg P&L: {cheap_pnl:+.2f}%  ({cheap_n} trades)")
            print(f"    Expensive IV avg P&L: {exp_pnl:+.2f}%  ({exp_n} trades)")
            print(f"    Spread (cheap - expensive): {spread:+.2f}%")
            if spread > 5:
                print(f"    --> CONFIRMED: Cheap IV significantly better. Filter adds edge.")
            elif spread > 0:
                print(f"    --> Marginal: Cheap IV slightly better but not conclusive.")
            else:
                print(f"    --> REJECTED: Cheap IV NOT better. IV regime may not be the issue.")
        elif cheap_n > 0:
            print(f"    Cheap IV: {cheap['avg_pnl_pct']:+.2f}% ({cheap_n} trades) — no expensive IV trades to compare")
        elif exp_n > 0:
            print(f"    Expensive IV: {expensive['avg_pnl_pct']:+.2f}% ({exp_n} trades) — no cheap IV trades to compare")
        else:
            print(f"    Insufficient data for both regimes")

    # ── Save Results ───────────────────────────────────────────────────
    # Strip raw trades for the summary output to keep it manageable
    save_results = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "backtest_period": f"{BACKTEST_START} to {END_DATE}",
            "universe_size": len(UNIVERSE),
            "options_structure": f"~{TARGET_DELTA:.0%} delta call, {TARGET_OTM_PCT:.0%} OTM, {DEFAULT_DTE} DTE",
            "hold_days": HOLD_DAYS,
            "profit_target": f"{PROFIT_TARGET_PCT:.0%}",
            "stop_loss": f"{STOP_LOSS_PCT:.0%}",
            "hypothesis": "Options overlays fail due to theta decay. Filtering for cheap sector-relative IV should improve P&L.",
        },
        "signals": {},
    }

    for sig_name, result in all_results.items():
        save_results["signals"][sig_name] = {
            "description": result["description"],
            "total_trades": result["total_trades"],
            "regimes": result["regimes"],
            # Include sample trades (first 10 per regime) for auditability
            "sample_trades": {
                regime: [t for t in result["all_trades"] if t["iv_regime"] == regime][:10]
                for regime in ["cheap", "normal", "expensive"]
            },
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n[SAVED] Results to {OUTPUT_FILE}")

    return save_results


if __name__ == "__main__":
    results = run_backtest()
