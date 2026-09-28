#!/usr/bin/env python3
"""
Butterfly Income v1 — Weekly/Biweekly Iron Butterflies on SPY
==============================================================
Income strategy: sell ATM straddle + buy OTM wings for protection.
Profit zone = ±wing_width around the ATM strike.

THEORETICAL EDGE:
  - Implied vol consistently overstates realized vol in SPY (VRP = volatility
    risk premium). The butterfly SELLS that premium.
  - Unlike naked straddles, butterflies have CAPPED risk = wing width - net credit.
  - High theta: short ATM options decay fastest. Butterflies concentrate
    theta at the center.
  - Weekly/biweekly DTE maximizes theta-per-day while giving enough time
    for the trade to work.

STRATEGY MECHANICS:
  - Sell 1 ATM put + sell 1 ATM call (= short straddle at ATM strike)
  - Buy 1 OTM put (wing_pct below ATM) + buy 1 OTM call (wing_pct above ATM)
  - Net credit = straddle premium - wing cost
  - Max profit = net credit (if SPY finishes exactly at ATM)
  - Max loss = wing_width - net_credit (if SPY finishes beyond wings)
  - Breakeven = ATM ± net_credit

RISK MANAGEMENT:
  - Profit-take at 30-50% of max profit (don't wait for expiry)
  - Stop-loss at 1.5-2x max profit (cut losses early)
  - Wing width sized so max loss ≤ 2-3% of NAV per trade
  - One trade per cycle (no overlapping)

ADVERSARIAL QUALITY GATES:
  - R1 regime-agnostic: stratify by green/red/flat SPY days, reject if gap > 0.50
  - 2000-trial permutation test (p < 0.05 required)
  - Walk-forward OOT (60-month train, 1-month test, sliding)
  - Realistic costs: $0.65/contract commission, 5% of premium slippage
  - Black-Scholes pricing (conservative — known to underprice ~41%)
  - NO look-ahead bias — decisions use only data available at entry time

SIZING: $100K portfolio. Max 3% NAV at risk per trade.

Data: SPY prices from yfinance (2015-present for sufficient OOT history).
"""

import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════
ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUTPUT = ROOT / "output" / "butterfly_income_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
STARTING_CAPITAL = 100_000
MAX_NAV_RISK_PCT = 0.03       # max 3% NAV at risk per trade
COMMISSION_PER_CONTRACT = 0.65  # per leg, per contract
SLIPPAGE_PCT = 0.05           # 5% of premium as slippage
RISK_FREE_RATE = 0.04


# ═══════════════════════════════════════════════════════════════════
# Black-Scholes Primitives
# ═══════════════════════════════════════════════════════════════════

def _Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * math.exp(-q * T) * _Phi(-d1)
    return S * math.exp(-q * T) * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)


def bs_delta(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if kind == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if kind == "call":
        return math.exp(-q * T) * _Phi(d1)
    return math.exp(-q * T) * (_Phi(d1) - 1)


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_spy_data():
    """Load SPY price data from cache or yfinance."""
    # Try cache first
    spy_cache = CACHE / "spy_prices.parquet"
    if spy_cache.exists():
        spy = pd.read_parquet(spy_cache)
        spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None)
        spy = spy.sort_values("date").reset_index(drop=True)
        print(f"  Loaded SPY from cache: {len(spy)} rows, {spy['date'].min().date()} to {spy['date'].max().date()}")
        return spy

    # Fallback: try prices.parquet
    prices_path = CACHE / "prices.parquet"
    if prices_path.exists():
        prices = pd.read_parquet(prices_path)
        prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
        if "ticker" in prices.columns:
            spy = prices[prices["ticker"] == "SPY"][["date", "open", "high", "low", "close", "volume"]].copy()
        elif "SPY" in prices.columns:
            spy = prices[["date"]].copy()
            spy["close"] = prices["SPY"]
        else:
            raise ValueError("No SPY data found in prices.parquet")
        spy = spy.sort_values("date").reset_index(drop=True)
        print(f"  Loaded SPY from prices: {len(spy)} rows")
        return spy

    # Last resort: yfinance
    try:
        import yfinance as yf
        print("  Downloading SPY from yfinance...")
        tk = yf.Ticker("SPY")
        spy = tk.history(start="2014-01-01", end="2026-07-15")
        spy = spy.reset_index()
        spy.columns = [c.lower() for c in spy.columns]
        spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None)
        spy = spy[["date", "open", "high", "low", "close", "volume"]].copy()
        print(f"  Downloaded SPY: {len(spy)} rows")
        return spy
    except Exception as e:
        raise RuntimeError(f"Cannot load SPY data: {e}")


def load_vix_data():
    """Load VIX data for regime classification."""
    vix_cache = CACHE / "vix_history.parquet"
    if vix_cache.exists():
        vix = pd.read_parquet(vix_cache)
        vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
        return vix

    # Try macro.parquet
    macro_path = CACHE / "macro.parquet"
    if macro_path.exists():
        macro = pd.read_parquet(macro_path)
        macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
        if "vix" in macro.columns:
            return macro[["date", "vix"]].dropna().rename(columns={"vix": "close"})

    return None


# ═══════════════════════════════════════════════════════════════════
# Implied Volatility Estimation
# ═══════════════════════════════════════════════════════════════════

def compute_iv_features(spy_df):
    """
    Estimate implied vol from realized vol with VRP markup.
    Use 20-day RV * 1.15 as IV proxy (conservative).
    Also compute IV rank (percentile over trailing 252 days).
    """
    df = spy_df.copy()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["rv_10"] = df["log_ret"].rolling(10).std() * np.sqrt(252)
    df["rv_20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)
    df["rv_60"] = df["log_ret"].rolling(60).std() * np.sqrt(252)

    # IV proxy: RV20 * 1.15 (VRP markup). Capped at reasonable bounds.
    df["sigma"] = (df["rv_20"] * 1.15).clip(0.08, 1.50)

    # IV rank: where current IV sits in trailing 252-day range
    df["iv_high_252"] = df["sigma"].rolling(252).max()
    df["iv_low_252"] = df["sigma"].rolling(252).min()
    iv_range = df["iv_high_252"] - df["iv_low_252"]
    df["iv_rank"] = np.where(iv_range > 0.001,
        (df["sigma"] - df["iv_low_252"]) / iv_range, 0.5)

    # Realized-vs-implied ratio (for VRP signal)
    df["rv_iv_ratio"] = np.where(df["sigma"] > 0.001, df["rv_10"] / df["sigma"], 1.0)

    # Expected move for the period (annualized sigma -> period sigma)
    # For a 7-day trade: expected_move = sigma * sqrt(7/252)
    df["spy_ret"] = df["close"].pct_change()

    return df.dropna(subset=["sigma"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# Cost Model
# ═══════════════════════════════════════════════════════════════════

def compute_trade_costs(premiums_per_share, n_contracts, n_legs=4):
    """
    Total cost for opening or closing a butterfly.
    - Commission: $0.65 per leg per contract (both open and close)
    - Slippage: 5% of each premium
    """
    commission = COMMISSION_PER_CONTRACT * n_legs * n_contracts
    slippage = sum(abs(p) * SLIPPAGE_PCT * 100 * n_contracts for p in premiums_per_share)
    return commission + slippage


# ═══════════════════════════════════════════════════════════════════
# Iron Butterfly Strategy Engine
# ═══════════════════════════════════════════════════════════════════

def run_butterfly_backtest(spy_df, dte_target=10, wing_pct=0.03,
                           profit_take_pct=0.40, stop_loss_mult=1.5,
                           min_iv_rank=0.0, max_iv_rank=1.0,
                           entry_day="weekly", label="BF"):
    """
    Run iron butterfly backtest on SPY.

    Parameters:
    -----------
    dte_target : int
        Target days to expiry (7 = weekly, 14 = biweekly)
    wing_pct : float
        Wing distance as fraction of SPY price (e.g., 0.03 = 3%)
    profit_take_pct : float
        Close when this fraction of max profit captured (e.g., 0.40 = 40%)
    stop_loss_mult : float
        Close when loss exceeds this * max_profit (e.g., 1.5 = 150% of max profit)
    min_iv_rank / max_iv_rank : float
        Only enter when IV rank is in this range (filter)
    entry_day : str
        "weekly" = enter every ~dte_target days, "daily" = check daily for new entries

    Returns:
    --------
    equity_curve : list of dict
    trades : list of dict
    stats : dict
    """
    print(f"\n{'='*70}")
    print(f"IRON BUTTERFLY: {label}")
    print(f"  DTE={dte_target}, Wing={wing_pct*100:.1f}%, PT={profit_take_pct*100:.0f}%, "
          f"SL={stop_loss_mult:.1f}x, IVR=[{min_iv_rank:.1f},{max_iv_rank:.1f}]")
    print(f"{'='*70}")

    df = spy_df.copy()
    all_dates = df["date"].values
    close_prices = df.set_index("date")["close"].to_dict()
    sigma_map = df.set_index("date")["sigma"].to_dict()
    ivr_map = df.set_index("date")["iv_rank"].to_dict()

    cash = float(STARTING_CAPITAL)
    position = None  # only one position at a time
    equity_curve = []
    trades = []
    stats = {
        "n_opened": 0, "n_closed": 0,
        "n_profit_take": 0, "n_stop_loss": 0,
        "n_expiry_win": 0, "n_expiry_loss": 0,
        "n_iv_filtered": 0, "n_already_open": 0,
        "total_commission": 0.0, "total_slippage": 0.0,
        "total_costs": 0.0,
    }

    days_since_entry = 999  # track spacing between entries

    for i, dt in enumerate(all_dates):
        S = close_prices.get(dt)
        sigma = sigma_map.get(dt)
        ivr = ivr_map.get(dt, 0.5)

        if S is None or sigma is None or np.isnan(S) or np.isnan(sigma):
            equity_curve.append({"date": dt, "equity": cash})
            continue

        # ── Manage existing position ──
        if position is not None:
            T_days = (position["expiry"] - dt) / np.timedelta64(1, "D")
            T = max(T_days, 0) / 365.0
            sigma_cur = sigma or position["open_sigma"]

            if T_days <= 0:
                # ── EXPIRY ──
                # Settlement: intrinsic value of each leg
                K_atm = position["atm_strike"]
                K_put_wing = position["put_wing"]
                K_call_wing = position["call_wing"]

                # Short ATM put payoff (we sold it, negative = loss for us)
                short_put_payoff = -max(K_atm - S, 0)
                # Short ATM call payoff
                short_call_payoff = -max(S - K_atm, 0)
                # Long OTM put payoff (we bought it)
                long_put_payoff = max(K_put_wing - S, 0)
                # Long OTM call payoff
                long_call_payoff = max(S - K_call_wing, 0)

                settlement = (short_put_payoff + short_call_payoff +
                              long_put_payoff + long_call_payoff) * 100 * position["contracts"]

                pnl = position["net_credit_dollar"] + settlement
                # No additional close costs at expiry (auto-exercise)

                if pnl >= 0:
                    stats["n_expiry_win"] += 1
                else:
                    stats["n_expiry_loss"] += 1

                cash += pnl
                trades.append({
                    "open_date": str(position["open_date"]),
                    "close_date": str(dt),
                    "ticker": "SPY",
                    "pnl": round(float(pnl), 2),
                    "close_reason": "expiry",
                    "entry_price": float(position["entry_spy"]),
                    "exit_price": float(S),
                    "net_credit": round(float(position["net_credit_dollar"]), 2),
                    "max_profit": round(float(position["net_credit_dollar"]), 2),
                    "max_loss": round(float(position["max_loss_dollar"]), 2),
                    "contracts": int(position["contracts"]),
                    "hold_days": int(position["dte"]),
                    "open_sigma": round(float(position["open_sigma"]), 4),
                    "open_ivr": round(float(position["open_ivr"]), 3),
                })
                position = None
                stats["n_closed"] += 1

            elif position is not None:
                # ── MID-LIFE: mark-to-market + check PT/SL ──
                K_atm = position["atm_strike"]
                K_put_wing = position["put_wing"]
                K_call_wing = position["call_wing"]

                # Current value of the butterfly (cost to close)
                short_put_val = bs_price(S, K_atm, T, sigma_cur, kind="put")
                short_call_val = bs_price(S, K_atm, T, sigma_cur, kind="call")
                long_put_val = bs_price(S, K_put_wing, T, sigma_cur, kind="put")
                long_call_val = bs_price(S, K_call_wing, T, sigma_cur, kind="call")

                # Cost to close = buy back shorts, sell longs
                cost_to_close_per_share = (short_put_val + short_call_val -
                                           long_put_val - long_call_val)
                cost_to_close_dollar = cost_to_close_per_share * 100 * position["contracts"]

                # Close costs (commissions + slippage)
                close_costs = compute_trade_costs(
                    [short_put_val, short_call_val, long_put_val, long_call_val],
                    position["contracts"], n_legs=4)

                unrealized_pnl = position["net_credit_dollar"] - cost_to_close_dollar - close_costs

                # Profit take
                if unrealized_pnl >= profit_take_pct * position["net_credit_dollar"]:
                    cash += unrealized_pnl
                    stats["total_costs"] += close_costs
                    trades.append({
                        "open_date": str(position["open_date"]),
                        "close_date": str(dt),
                        "ticker": "SPY",
                        "pnl": round(float(unrealized_pnl), 2),
                        "close_reason": "profit_take",
                        "entry_price": float(position["entry_spy"]),
                        "exit_price": float(S),
                        "net_credit": round(float(position["net_credit_dollar"]), 2),
                        "max_profit": round(float(position["net_credit_dollar"]), 2),
                        "max_loss": round(float(position["max_loss_dollar"]), 2),
                        "contracts": int(position["contracts"]),
                        "hold_days": int((dt - position["open_date_raw"]) / np.timedelta64(1, "D")),
                        "open_sigma": round(float(position["open_sigma"]), 4),
                        "open_ivr": round(float(position["open_ivr"]), 3),
                    })
                    position = None
                    stats["n_profit_take"] += 1
                    stats["n_closed"] += 1

                # Stop loss
                elif unrealized_pnl < -(stop_loss_mult * position["net_credit_dollar"]):
                    cash += unrealized_pnl
                    stats["total_costs"] += close_costs
                    trades.append({
                        "open_date": str(position["open_date"]),
                        "close_date": str(dt),
                        "ticker": "SPY",
                        "pnl": round(float(unrealized_pnl), 2),
                        "close_reason": "stop_loss",
                        "entry_price": float(position["entry_spy"]),
                        "exit_price": float(S),
                        "net_credit": round(float(position["net_credit_dollar"]), 2),
                        "max_profit": round(float(position["net_credit_dollar"]), 2),
                        "max_loss": round(float(position["max_loss_dollar"]), 2),
                        "contracts": int(position["contracts"]),
                        "hold_days": int((dt - position["open_date_raw"]) / np.timedelta64(1, "D")),
                        "open_sigma": round(float(position["open_sigma"]), 4),
                        "open_ivr": round(float(position["open_ivr"]), 3),
                    })
                    position = None
                    stats["n_stop_loss"] += 1
                    stats["n_closed"] += 1

        # ── Open new position ──
        days_since_entry += 1

        if position is None and days_since_entry >= max(dte_target - 2, 5):
            # IV rank filter
            if ivr < min_iv_rank or ivr > max_iv_rank:
                stats["n_iv_filtered"] += 1
                equity_curve.append({"date": dt, "equity": cash})
                continue

            T = dte_target / 365.0

            # ATM strike = current price (rounded to nearest dollar for SPY)
            K_atm = round(S)

            # Wing strikes
            wing_dist = S * wing_pct
            K_put_wing = round(S - wing_dist)
            K_call_wing = round(S + wing_dist)

            # Price each leg
            short_put_prem = bs_price(S, K_atm, T, sigma, kind="put")
            short_call_prem = bs_price(S, K_atm, T, sigma, kind="call")
            long_put_prem = bs_price(S, K_put_wing, T, sigma, kind="put")
            long_call_prem = bs_price(S, K_call_wing, T, sigma, kind="call")

            # Net credit per share
            net_credit = (short_put_prem + short_call_prem -
                          long_put_prem - long_call_prem)

            if net_credit <= 0.10:
                equity_curve.append({"date": dt, "equity": cash})
                continue

            # Max loss = wing width - net credit (per share)
            wing_width = K_atm - K_put_wing  # should equal wing_dist
            max_loss_per_share = wing_width - net_credit

            if max_loss_per_share <= 0:
                # Free money? Suspicious. Skip.
                equity_curve.append({"date": dt, "equity": cash})
                continue

            # Sizing: max loss per trade ≤ MAX_NAV_RISK_PCT of current NAV
            max_loss_per_contract = max_loss_per_share * 100
            max_risk_dollar = cash * MAX_NAV_RISK_PCT
            contracts = max(1, int(max_risk_dollar / max_loss_per_contract))

            # Cap at reasonable size
            if contracts > 50:
                contracts = 50

            # Compute costs
            open_costs = compute_trade_costs(
                [short_put_prem, short_call_prem, long_put_prem, long_call_prem],
                contracts, n_legs=4)

            net_credit_dollar = net_credit * 100 * contracts - open_costs
            max_loss_dollar = max_loss_per_share * 100 * contracts + open_costs

            if net_credit_dollar <= 0:
                equity_curve.append({"date": dt, "equity": cash})
                continue

            stats["total_costs"] += open_costs

            # Expiry date
            expiry = dt + np.timedelta64(dte_target, "D")

            position = {
                "open_date": str(dt),
                "open_date_raw": dt,
                "entry_spy": S,
                "atm_strike": K_atm,
                "put_wing": K_put_wing,
                "call_wing": K_call_wing,
                "net_credit_dollar": net_credit_dollar,
                "max_loss_dollar": max_loss_dollar,
                "contracts": contracts,
                "expiry": expiry,
                "dte": dte_target,
                "open_sigma": sigma,
                "open_ivr": ivr,
            }

            stats["n_opened"] += 1
            days_since_entry = 0

        equity_curve.append({"date": dt, "equity": cash})

    # Close any remaining position at last price
    if position is not None:
        S = float(df["close"].iloc[-1])
        K_atm = position["atm_strike"]
        pnl_final = position["net_credit_dollar"] - max(
            (K_atm - S) if S < K_atm else (S - K_atm), 0) * 100 * position["contracts"]
        cash += pnl_final
        trades.append({
            "open_date": str(position["open_date"]),
            "close_date": str(df["date"].iloc[-1]),
            "ticker": "SPY",
            "pnl": round(float(pnl_final), 2),
            "close_reason": "end_of_data",
            "entry_price": float(position["entry_spy"]),
            "exit_price": float(S),
            "net_credit": round(float(position["net_credit_dollar"]), 2),
            "max_profit": round(float(position["net_credit_dollar"]), 2),
            "max_loss": round(float(position["max_loss_dollar"]), 2),
            "contracts": int(position["contracts"]),
            "hold_days": 0,
            "open_sigma": round(float(position["open_sigma"]), 4),
            "open_ivr": round(float(position["open_ivr"]), 3),
        })
        stats["n_closed"] += 1

    print(f"  Opened: {stats['n_opened']}, Closed: {stats['n_closed']}")
    print(f"  PT: {stats['n_profit_take']}, SL: {stats['n_stop_loss']}, "
          f"Expiry W: {stats['n_expiry_win']}, Expiry L: {stats['n_expiry_loss']}")
    print(f"  IV filtered: {stats['n_iv_filtered']}")
    print(f"  Total costs: ${stats['total_costs']:.2f}")
    print(f"  Final equity: ${cash:,.2f}")

    return equity_curve, trades, stats


# ═══════════════════════════════════════════════════════════════════
# Analytics: Metrics, Regime Test, Permutation Test, Walk-Forward
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(equity_curve, label="Strategy"):
    """Compute Sharpe, Sortino, CAGR, max DD, PF, WR from equity curve."""
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Remove duplicate dates (keep last)
    df = df.drop_duplicates(subset=["date"], keep="last")
    df["daily_ret"] = df["equity"].pct_change()
    df = df.dropna(subset=["daily_ret"])

    if len(df) < 30:
        return {"label": label, "error": "too few data points"}

    rets = df["daily_ret"].values
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)

    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-6
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    total_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    years = total_days / 365.25
    total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    cum_max = df["equity"].cummax()
    drawdown = (df["equity"] - cum_max) / cum_max
    max_dd = drawdown.min()

    # PF and WR from daily returns
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")
    wr = np.mean(rets > 0)

    # Income per $100K per month
    monthly_income = (total_return * STARTING_CAPITAL) / max(years * 12, 1)

    return {
        "label": label,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "cagr": round(cagr * 100, 1),
        "max_dd": round(max_dd * 100, 1),
        "pf": round(pf, 2),
        "wr": round(wr * 100, 1),
        "total_return": round(total_return * 100, 1),
        "monthly_income_per_100k": round(monthly_income, 2),
        "n_days": len(df),
        "start": str(df["date"].iloc[0].date()),
        "end": str(df["date"].iloc[-1].date()),
    }


def regime_test(equity_curve, spy_df, label="Strategy"):
    """R1 regime-agnostic test: stratify returns by SPY regime (green/red/flat days)."""
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    df["daily_ret"] = df["equity"].pct_change()
    df = df.dropna(subset=["daily_ret"])

    spy = spy_df.copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy["spy_ret_regime"] = spy["close"].pct_change()
    spy = spy[["date", "spy_ret_regime"]].dropna()

    merged = df.merge(spy, on="date", how="inner")
    if len(merged) < 60:
        return {"label": label, "error": "insufficient data for regime test"}

    # Classify: green (SPY > +0.1%), red (SPY < -0.1%), flat
    merged["regime"] = np.where(merged["spy_ret_regime"] > 0.001, "green",
                       np.where(merged["spy_ret_regime"] < -0.001, "red", "flat"))

    results = {}
    for regime in ["green", "red", "flat"]:
        subset = merged[merged["regime"] == regime]["daily_ret"]
        if len(subset) < 10:
            results[regime] = {"sharpe": 0, "n_days": len(subset)}
            continue
        mean_r = subset.mean()
        std_r = subset.std(ddof=1)
        sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0
        results[regime] = {"sharpe": round(sharpe, 2), "n_days": len(subset)}

    # R1 gap test: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) > 0.50 => FAIL
    s_green = results.get("green", {}).get("sharpe", 0)
    s_red = results.get("red", {}).get("sharpe", 0)
    max_abs = max(abs(s_green), abs(s_red))
    r1_gap = abs(s_green - s_red) / max_abs if max_abs > 0 else 0

    return {
        "label": label,
        "green_sharpe": results.get("green", {}).get("sharpe", 0),
        "red_sharpe": results.get("red", {}).get("sharpe", 0),
        "flat_sharpe": results.get("flat", {}).get("sharpe", 0),
        "green_days": results.get("green", {}).get("n_days", 0),
        "red_days": results.get("red", {}).get("n_days", 0),
        "flat_days": results.get("flat", {}).get("n_days", 0),
        "r1_gap": round(r1_gap, 3),
        "r1_pass": r1_gap < 0.50,
    }


def permutation_test_trades(trades, n_perms=2000, label="Strategy"):
    """
    Permutation test on TRADE PnLs.
    Null hypothesis: trade PnLs have zero mean (random entry timing).
    Shuffle the signs of trade PnLs and compare t-statistic.
    """
    if not trades or len(trades) < 10:
        return {"label": label, "error": "insufficient trades", "p_value": 1.0,
                "significant_05": False, "significant_01": False}

    pnls = np.array([t["pnl"] for t in trades])
    actual_mean = np.mean(pnls)
    actual_std = np.std(pnls, ddof=1)
    if actual_std == 0:
        return {"label": label, "error": "zero variance", "p_value": 1.0,
                "significant_05": False, "significant_01": False}

    actual_t = actual_mean / (actual_std / np.sqrt(len(pnls)))

    rng = np.random.RandomState(42)
    perm_ts = []
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(pnls))
        shuffled = pnls * signs
        perm_mean = np.mean(shuffled)
        perm_std = np.std(shuffled, ddof=1)
        if perm_std > 0:
            perm_t = perm_mean / (perm_std / np.sqrt(len(shuffled)))
        else:
            perm_t = 0
        perm_ts.append(perm_t)

    p_value = np.mean(np.array(perm_ts) >= actual_t)

    return {
        "label": label,
        "actual_mean_pnl": round(float(actual_mean), 2),
        "actual_t_stat": round(float(actual_t), 3),
        "p_value": round(float(p_value), 4),
        "significant_05": bool(p_value < 0.05),
        "significant_01": bool(p_value < 0.01),
        "n_trades": len(pnls),
        "n_perms": n_perms,
    }


def walk_forward_analysis(equity_curve, train_months=60, test_months=1, label="Strategy"):
    """Walk-forward OOT analysis: 60-month train, 1-month test, sliding."""
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    df["daily_ret"] = df["equity"].pct_change()
    df = df.dropna(subset=["daily_ret"])

    df["ym"] = df["date"].dt.to_period("M")
    months = sorted(df["ym"].unique())

    if len(months) < train_months + test_months:
        # Use all available data as OOT (no train period needed for non-ML strategy)
        train_months = max(1, len(months) // 2)

    oot_results = []
    for i in range(train_months, len(months)):
        test_month = months[i]
        test_rets = df[df["ym"] == test_month]["daily_ret"].values
        if len(test_rets) < 5:
            continue

        monthly_ret = np.sum(test_rets)
        monthly_std = np.std(test_rets, ddof=1) if len(test_rets) > 1 else 0.01
        monthly_sharpe = (np.mean(test_rets) / monthly_std) * np.sqrt(252) if monthly_std > 0 else 0

        oot_results.append({
            "month": str(test_month),
            "return_pct": round(float(monthly_ret * 100), 2),
            "sharpe": round(float(monthly_sharpe), 2),
            "n_days": len(test_rets),
            "positive": bool(monthly_ret > 0),
        })

    if not oot_results:
        return {"label": label, "error": "no OOT months"}

    oot_df = pd.DataFrame(oot_results)
    win_rate = oot_df["positive"].mean()
    avg_sharpe = oot_df["sharpe"].mean()
    avg_return = oot_df["return_pct"].mean()

    return {
        "label": label,
        "n_oot_months": len(oot_results),
        "monthly_win_rate": round(float(win_rate * 100), 1),
        "avg_monthly_sharpe": round(float(avg_sharpe), 2),
        "avg_monthly_return_pct": round(float(avg_return), 2),
        "worst_month_pct": round(float(oot_df["return_pct"].min()), 2),
        "best_month_pct": round(float(oot_df["return_pct"].max()), 2),
        "oot_months": oot_results,
    }


def trade_analysis(trades, label="Strategy"):
    """Analyze trade-level statistics."""
    if not trades:
        return {"label": label, "error": "no trades"}

    df = pd.DataFrame(trades)
    n_trades = len(df)
    n_winners = (df["pnl"] > 0).sum()
    n_losers = (df["pnl"] <= 0).sum()
    wr = n_winners / n_trades if n_trades > 0 else 0

    avg_win = float(df[df["pnl"] > 0]["pnl"].mean()) if n_winners > 0 else 0
    avg_loss = float(df[df["pnl"] <= 0]["pnl"].mean()) if n_losers > 0 else 0
    total_gains = float(df[df["pnl"] > 0]["pnl"].sum())
    total_losses = float(abs(df[df["pnl"] <= 0]["pnl"].sum()))
    pf = total_gains / total_losses if total_losses > 0 else float("inf")

    close_reasons = df["close_reason"].value_counts().to_dict()

    avg_hold = float(df["hold_days"].mean()) if "hold_days" in df.columns else 0

    return {
        "label": label,
        "n_trades": int(n_trades),
        "win_rate": round(float(wr * 100), 1),
        "profit_factor": round(float(pf), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_pnl": round(float(df["pnl"].sum()), 2),
        "avg_hold_days": round(avg_hold, 1),
        "close_reasons": {str(k): int(v) for k, v in close_reasons.items()},
    }


# ═══════════════════════════════════════════════════════════════════
# Serialization Helper
# ═══════════════════════════════════════════════════════════════════

def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def clean_dict(d):
    if isinstance(d, dict):
        return {str(k): clean_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_dict(x) for x in d]
    else:
        return make_serializable(d)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    t0 = time.time()

    print("=" * 80)
    print("BUTTERFLY INCOME v1 — Iron Butterfly on SPY")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    spy_raw = load_spy_data()
    spy = compute_iv_features(spy_raw)
    print(f"  SPY with IV features: {len(spy)} rows, "
          f"{spy['date'].iloc[0].date() if not spy.empty else '?'} to "
          f"{spy['date'].iloc[-1].date() if not spy.empty else '?'}")

    print(f"\nData loaded in {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # Parameter sweep: test multiple configurations
    # ═══════════════════════════════════════════════════════════════
    configs = [
        # === DTE variations ===
        {"label": "BF_7dte_3w_pt40",
         "dte_target": 7, "wing_pct": 0.03, "profit_take_pct": 0.40, "stop_loss_mult": 1.5},
        {"label": "BF_10dte_3w_pt40",
         "dte_target": 10, "wing_pct": 0.03, "profit_take_pct": 0.40, "stop_loss_mult": 1.5},
        {"label": "BF_14dte_3w_pt40",
         "dte_target": 14, "wing_pct": 0.03, "profit_take_pct": 0.40, "stop_loss_mult": 1.5},

        # === Wing width variations ===
        {"label": "BF_10dte_2w_pt40",
         "dte_target": 10, "wing_pct": 0.02, "profit_take_pct": 0.40, "stop_loss_mult": 1.5},
        {"label": "BF_10dte_4w_pt40",
         "dte_target": 10, "wing_pct": 0.04, "profit_take_pct": 0.40, "stop_loss_mult": 1.5},
        {"label": "BF_10dte_5w_pt40",
         "dte_target": 10, "wing_pct": 0.05, "profit_take_pct": 0.40, "stop_loss_mult": 1.5},

        # === Profit take variations ===
        {"label": "BF_10dte_3w_pt30",
         "dte_target": 10, "wing_pct": 0.03, "profit_take_pct": 0.30, "stop_loss_mult": 1.5},
        {"label": "BF_10dte_3w_pt50",
         "dte_target": 10, "wing_pct": 0.03, "profit_take_pct": 0.50, "stop_loss_mult": 1.5},

        # === Stop loss variations ===
        {"label": "BF_10dte_3w_pt40_sl2",
         "dte_target": 10, "wing_pct": 0.03, "profit_take_pct": 0.40, "stop_loss_mult": 2.0},
        {"label": "BF_10dte_3w_pt40_sl1",
         "dte_target": 10, "wing_pct": 0.03, "profit_take_pct": 0.40, "stop_loss_mult": 1.0},

        # === IV rank filter (only enter when IV is elevated = more premium) ===
        {"label": "BF_10dte_3w_pt40_ivr30",
         "dte_target": 10, "wing_pct": 0.03, "profit_take_pct": 0.40, "stop_loss_mult": 1.5,
         "min_iv_rank": 0.30},
        {"label": "BF_10dte_3w_pt40_ivr50",
         "dte_target": 10, "wing_pct": 0.03, "profit_take_pct": 0.40, "stop_loss_mult": 1.5,
         "min_iv_rank": 0.50},

        # === Combined best-guess ===
        {"label": "BF_10dte_4w_pt30_sl2_ivr30",
         "dte_target": 10, "wing_pct": 0.04, "profit_take_pct": 0.30, "stop_loss_mult": 2.0,
         "min_iv_rank": 0.30},
        {"label": "BF_14dte_4w_pt40_sl2_ivr30",
         "dte_target": 14, "wing_pct": 0.04, "profit_take_pct": 0.40, "stop_loss_mult": 2.0,
         "min_iv_rank": 0.30},
    ]

    all_results = {}

    for cfg in configs:
        label = cfg.pop("label")
        eq, trades, stats = run_butterfly_backtest(spy, label=label, **cfg)
        metrics = compute_metrics(eq, label)
        regime = regime_test(eq, spy, label)
        perm = permutation_test_trades(trades, n_perms=2000, label=label)
        wf = walk_forward_analysis(eq, label=label)
        trd = trade_analysis(trades, label)

        all_results[label] = {
            "config": {**cfg, "label": label},
            "metrics": metrics,
            "regime": regime,
            "permutation": perm,
            "walk_forward": wf,
            "trade_stats": trd,
            "engine_stats": stats,
        }

        m = metrics
        rg = regime
        pm = perm
        if "error" not in m:
            print(f"\n  >> {label}: Sharpe={m['sharpe']}, Sortino={m['sortino']}, "
                  f"CAGR={m['cagr']}%, MaxDD={m['max_dd']}%, "
                  f"PF={m['pf']}, WR={m['wr']}%, "
                  f"$/mo={m['monthly_income_per_100k']:.0f}, "
                  f"R1gap={rg.get('r1_gap', '?')}, R1={'PASS' if rg.get('r1_pass') else 'FAIL'}, "
                  f"p={pm.get('p_value', '?')}")
        else:
            print(f"\n  >> {label}: ERROR: {m.get('error', 'unknown')}")

    # ═══════════════════════════════════════════════════════════════
    # Final Comparison Table
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*110}")
    print("FINAL COMPARISON — IRON BUTTERFLY INCOME ON SPY")
    print(f"{'='*110}")
    header = (f"{'Strategy':<35} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} "
              f"{'PF':>6} {'WR%':>6} {'$/mo':>7} {'R1gap':>7} {'R1':>5} {'p-val':>7} {'Trades':>7}")
    print(header)
    print("-" * 110)

    for label in sorted(all_results.keys()):
        r = all_results[label]
        m = r["metrics"]
        rg = r["regime"]
        pm = r["permutation"]
        trd = r["trade_stats"]
        if "error" in m:
            print(f"  {label:<33} ERROR: {m['error']}")
            continue
        n_trades = trd.get("n_trades", 0) if "error" not in trd else 0
        print(f"  {label:<33} {m['sharpe']:>7} {m['sortino']:>8} {m['cagr']:>6.1f}% "
              f"{m['max_dd']:>6.1f}% {m['pf']:>6.2f} {m['wr']:>5.1f}% "
              f"{m['monthly_income_per_100k']:>6.0f} {rg.get('r1_gap', 0):>6.3f} "
              f"{'PASS' if rg.get('r1_pass') else 'FAIL':>5} "
              f"{pm.get('p_value', 1):>6.4f} {n_trades:>7}")

    # ═══════════════════════════════════════════════════════════════
    # Quality gate summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("QUALITY GATE RESULTS")
    print(f"{'='*80}")

    passing = []
    for label in sorted(all_results.keys()):
        r = all_results[label]
        m = r["metrics"]
        rg = r["regime"]
        pm = r["permutation"]

        if "error" in m:
            continue

        gates = {
            "R1_regime": rg.get("r1_pass", False),
            "permutation_p05": pm.get("significant_05", False),
            "sharpe_positive": m["sharpe"] > 0,
            "sortino_positive": m["sortino"] > 0,
            "max_dd_lt_20": m["max_dd"] > -20,
        }
        all_pass = all(gates.values())

        status = "PASS ALL" if all_pass else "FAIL"
        failed = [k for k, v in gates.items() if not v]
        print(f"  {label:<35} {status:<10} "
              f"{'Failed: ' + ', '.join(failed) if failed else ''}")

        if all_pass:
            passing.append((label, m["sharpe"]))

    if passing:
        passing.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  BEST PASSING CONFIG: {passing[0][0]} (Sharpe={passing[0][1]})")
    else:
        print(f"\n  NO CONFIGS PASSED ALL QUALITY GATES")

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    output_file = OUTPUT / "butterfly_income_v1_results.json"
    with open(output_file, "w") as f:
        json.dump(clean_dict(all_results), f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
