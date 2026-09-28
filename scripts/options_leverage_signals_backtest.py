#!/usr/bin/env python3
"""
Options Leverage Signals Backtest
=================================
Combines two validated share-based signals with OPTIONS for 2-5x leverage.

Signals:
  1. Adaptive RSI Vol-Regime: RSI(5) dip buy, hold 5d (high-vol) / 10d (low-vol). Share Sharpe 1.51.
  2. Earnings Surprise Momentum: Buy after >3% positive earnings gap, hold 60d. Share Sharpe 1.54.

Variants (6):
  A) RSI + ATM Calls (2-week expiry)
  B) RSI + OTM Calls (5% OTM, 2-week)
  C) Earnings Beat + ATM Calls (30-day expiry)
  D) Earnings Beat + Deep ITM Calls (10% ITM, 45-day)
  E) Combined: shares for RSI, calls for earnings
  F) Earnings Beat + Call Spreads ($5 wide)

Account: $669, walk-forward OOT: Jan 2022 – Jul 2026.
5-Gate: Sharpe>0.5, perm p<0.05, regime gap<0.5, MDD>-50%, >=20 trades.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Configuration ────────────────────────────────────────────────────────
CAPITAL = 669.0
MAX_POS_COST = 200.0  # Max $200 per trade
COMMISSION_PER_CONTRACT = 0.65  # each way
BID_ASK_HAIRCUT = 0.10  # 10% of premium
CONTRACT_MULT = 100
SLIPPAGE_SHARES = 0.0002  # for share-based variant E

START = "2020-01-01"  # lookback for indicators
END = "2026-07-30"
OOT_START = "2022-01-01"
N_PERM = 1000

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "COIN",
    "RBLX", "UBER",
]

ALL_TICKERS = sorted(set(TICKERS + ["SPY"]))

# ── Data Download ────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(ALL_TICKERS, start=START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        if len(ALL_TICKERS) == 1:
            s = raw["Close"].dropna()
        else:
            s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in ALL_TICKERS}
spy_close = closes.get("SPY", pd.Series(dtype=float))

loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 252)
print(f"  Tickers with sufficient data: {loaded}/{len(TICKERS)}")
print(f"  SPY rows: {len(spy_close)}")


# ── Indicator Helpers ────────────────────────────────────────────────────
def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_sma(series, period):
    return series.rolling(period).mean()


def realized_vol(series, window=20):
    """Annualized realized vol from daily log returns."""
    lr = np.log(series / series.shift(1))
    return lr.rolling(window).std() * np.sqrt(252)


# ── Options Pricing Model ────────────────────────────────────────────────
def atm_call_premium(stock_price, dte, iv_adj=0.0):
    """ATM call premium ≈ S × 0.04 × sqrt(DTE/30) × (1 + IV_adj)."""
    return stock_price * 0.04 * np.sqrt(max(dte, 1) / 30.0) * (1 + iv_adj)


def otm_call_premium(stock_price, dte, otm_pct, sigma_20d, iv_adj=0.0):
    """OTM call: ATM × exp(-0.5 × (OTM%/sigma)^2)."""
    atm = atm_call_premium(stock_price, dte, iv_adj)
    sigma = max(sigma_20d, 0.05)  # floor
    return atm * np.exp(-0.5 * (otm_pct / sigma) ** 2)


def deep_itm_premium(stock_price, itm_pct, dte, iv_adj=0.0):
    """Deep ITM: intrinsic + time_value (time_value = ATM × 0.4)."""
    intrinsic = stock_price * itm_pct
    atm = atm_call_premium(stock_price, dte, iv_adj)
    return intrinsic + atm * 0.4


def theta_decay(premium, dte, hold_days):
    """Daily theta: premium × (1/DTE) × 1.5, accumulated over hold_days."""
    total = 0.0
    remaining = dte
    for _ in range(hold_days):
        if remaining <= 0:
            break
        daily = premium * (1.0 / max(remaining, 0.5)) * 1.5
        total += daily
        remaining -= 1
    return min(total, premium)  # can't lose more than premium


def option_entry_cost(premium_per_share, n_contracts):
    """Total cost to enter an option position."""
    raw = premium_per_share * CONTRACT_MULT * n_contracts
    haircut = raw * BID_ASK_HAIRCUT
    commission = COMMISSION_PER_CONTRACT * n_contracts
    return raw + haircut + commission


def option_exit_value(premium_per_share, n_contracts):
    """Net proceeds when exiting (selling back) option."""
    raw = premium_per_share * CONTRACT_MULT * n_contracts
    haircut = raw * BID_ASK_HAIRCUT
    commission = COMMISSION_PER_CONTRACT * n_contracts
    return max(0, raw - haircut - commission)


def spread_entry_cost(net_premium_per_share, n_contracts):
    """Call spread: pay net debit."""
    raw = net_premium_per_share * CONTRACT_MULT * n_contracts
    haircut = raw * BID_ASK_HAIRCUT
    commission = COMMISSION_PER_CONTRACT * n_contracts * 2  # 2 legs
    return raw + haircut + commission


def spread_exit_value(net_premium_per_share, n_contracts):
    """Call spread exit."""
    raw = net_premium_per_share * CONTRACT_MULT * n_contracts
    haircut = raw * BID_ASK_HAIRCUT
    commission = COMMISSION_PER_CONTRACT * n_contracts * 2
    return max(0, raw - haircut - commission)


# ── Regime Classification ────────────────────────────────────────────────
spy_sma200 = calc_sma(spy_close, 200)
regime_bull = spy_close > spy_sma200  # True = Bull


# ── Signal Generation ────────────────────────────────────────────────────
def generate_rsi_signals():
    """
    Adaptive RSI Vol-Regime signal.
    Buy when RSI(5) < threshold (30 high-vol, 20 low-vol).
    Hold 5d (high-vol) or 10d (low-vol).
    """
    signals = []
    for ticker in TICKERS:
        px = closes.get(ticker, pd.Series(dtype=float))
        if len(px) < 252:
            continue
        rsi = calc_rsi(px, 5)
        vol = realized_vol(px, 20)
        vol_median = vol.rolling(252, min_periods=60).median()

        for i in range(252, len(px)):
            date = px.index[i]
            if date < pd.Timestamp(OOT_START):
                continue
            if pd.isna(rsi.iloc[i]) or pd.isna(vol.iloc[i]):
                continue

            high_vol = vol.iloc[i] > vol_median.iloc[i] if not pd.isna(vol_median.iloc[i]) else False
            threshold = 30 if high_vol else 20
            hold_days = 5 if high_vol else 10

            if rsi.iloc[i] < threshold:
                # Check SMA filter
                sma200 = calc_sma(px, 200)
                if pd.isna(sma200.iloc[i]) or px.iloc[i] < sma200.iloc[i]:
                    continue
                # Find exit
                entry_price = px.iloc[i]
                exit_idx = min(i + hold_days, len(px) - 1)
                exit_price = px.iloc[exit_idx]
                exit_date = px.index[exit_idx]
                actual_hold = exit_idx - i

                sigma_20d = vol.iloc[i] / np.sqrt(252) * np.sqrt(20)  # 20d vol as fraction
                sigma_20d_ann = vol.iloc[i] if not pd.isna(vol.iloc[i]) else 0.3

                signals.append({
                    "ticker": ticker,
                    "entry_date": date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "hold_days": actual_hold,
                    "signal_type": "RSI",
                    "vol_regime": "high" if high_vol else "low",
                    "sigma_20d": sigma_20d_ann,
                    "stock_return": (exit_price - entry_price) / entry_price,
                })
    return signals


def generate_earnings_signals():
    """
    Earnings Surprise Momentum.
    Buy after >3% positive earnings gap, hold 60 trading days.
    Uses gap-ups > 3% as earnings proxy, but filters to max 4 per ticker
    per year (roughly quarterly earnings cadence) to avoid non-earnings gaps.
    Also requires: gap is >2x the 20d avg absolute daily return (unusual move).
    """
    signals = []
    for ticker in TICKERS:
        px = closes.get(ticker, pd.Series(dtype=float))
        if len(px) < 252:
            continue
        daily_ret = px.pct_change()
        abs_ret_20d = daily_ret.abs().rolling(20).mean()
        vol = realized_vol(px, 20)

        # Track last signal date per ticker to enforce spacing (min 45 trading days apart)
        last_signal_idx = -999

        for i in range(60, len(px)):
            date = px.index[i]
            if date < pd.Timestamp(OOT_START):
                continue
            if pd.isna(daily_ret.iloc[i]) or pd.isna(abs_ret_20d.iloc[i]):
                continue

            # Gap-up > 3% AND > 2x normal daily move (filters out routine volatility)
            avg_move = abs_ret_20d.iloc[i]
            if daily_ret.iloc[i] > 0.03 and daily_ret.iloc[i] > 2.0 * avg_move:
                # Enforce min spacing of 45 trading days between signals per ticker
                if i - last_signal_idx < 45:
                    continue
                last_signal_idx = i
                entry_price = px.iloc[i]
                # Hold 60 trading days for shares, shorter for options
                exit_idx_60 = min(i + 60, len(px) - 1)
                exit_idx_20 = min(i + 20, len(px) - 1)
                exit_idx_40 = min(i + 40, len(px) - 1)

                sigma_20d_ann = vol.iloc[i] if not pd.isna(vol.iloc[i]) else 0.3

                signals.append({
                    "ticker": ticker,
                    "entry_date": date,
                    "entry_price": entry_price,
                    "exit_price_20d": px.iloc[exit_idx_20],
                    "exit_price_40d": px.iloc[exit_idx_40],
                    "exit_price_60d": px.iloc[exit_idx_60],
                    "exit_date_20d": px.index[exit_idx_20],
                    "exit_date_40d": px.index[exit_idx_40],
                    "exit_date_60d": px.index[exit_idx_60],
                    "signal_type": "EARNINGS",
                    "gap_pct": daily_ret.iloc[i],
                    "sigma_20d": sigma_20d_ann,
                    "stock_return_60d": (px.iloc[exit_idx_60] - entry_price) / entry_price,
                })
    return signals


print("\nGenerating signals ...")
rsi_signals = generate_rsi_signals()
earnings_signals = generate_earnings_signals()
print(f"  RSI signals: {len(rsi_signals)}")
print(f"  Earnings signals: {len(earnings_signals)}")


# ── Backtest Engine ──────────────────────────────────────────────────────
def backtest_variant_a(rsi_sigs):
    """RSI + ATM Calls (2-week expiry, 10 DTE)."""
    trades = []
    for sig in rsi_sigs:
        stock_price = sig["entry_price"]
        dte = 10  # ~2 weeks
        premium = atm_call_premium(stock_price, dte)
        cost_per_contract = premium * CONTRACT_MULT
        cost_with_fees = option_entry_cost(premium, 1)

        # Can we afford it?
        if cost_with_fees > MAX_POS_COST or cost_with_fees > CAPITAL:
            continue

        n_contracts = 1  # always 1 for small account

        entry_cost = option_entry_cost(premium, n_contracts)
        hold = min(sig["hold_days"], dte - 2)  # exit before expiry-2d
        hold = max(hold, 1)

        stock_move = sig["exit_price"] - sig["entry_price"]
        delta = 0.50  # ATM
        option_value_at_exit = max(0, premium + stock_move * delta / stock_price * premium * 2
                                   - theta_decay(premium, dte, hold))
        # More accurate: option P&L = delta × stock_move - theta_decay
        raw_exit_prem = max(0, premium + delta * stock_move - theta_decay(premium, dte, hold))
        exit_proceeds = option_exit_value(raw_exit_prem, n_contracts)
        pnl = exit_proceeds - entry_cost

        trades.append({
            "ticker": sig["ticker"],
            "entry_date": str(sig["entry_date"].date()),
            "exit_date": str(sig["exit_date"].date()),
            "entry_cost": round(entry_cost, 2),
            "exit_proceeds": round(exit_proceeds, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / entry_cost * 100, 2) if entry_cost > 0 else 0,
            "hold_days": hold,
            "regime": "bull" if regime_bull.reindex(
                [sig["entry_date"]], method="ffill").iloc[0] else "bear",
        })
    return trades


def backtest_variant_b(rsi_sigs):
    """RSI + OTM Calls (5% OTM, 2-week)."""
    trades = []
    for sig in rsi_sigs:
        stock_price = sig["entry_price"]
        dte = 10
        otm_pct = 0.05
        sigma = sig["sigma_20d"] if sig["sigma_20d"] > 0 else 0.3
        premium = otm_call_premium(stock_price, dte, otm_pct, sigma)
        cost_with_fees = option_entry_cost(premium, 1)

        if cost_with_fees > MAX_POS_COST or cost_with_fees > CAPITAL:
            continue

        n_contracts = 1
        entry_cost = option_entry_cost(premium, n_contracts)
        hold = min(sig["hold_days"], dte - 2)
        hold = max(hold, 1)

        stock_move = sig["exit_price"] - sig["entry_price"]
        delta = 0.35  # 5% OTM
        strike = stock_price * 1.05
        intrinsic_at_exit = max(0, sig["exit_price"] - strike)
        time_val_at_exit = max(0, premium - theta_decay(premium, dte, hold))
        raw_exit_prem = max(0, intrinsic_at_exit + time_val_at_exit * 0.5)
        exit_proceeds = option_exit_value(raw_exit_prem, n_contracts)
        pnl = exit_proceeds - entry_cost

        trades.append({
            "ticker": sig["ticker"],
            "entry_date": str(sig["entry_date"].date()),
            "exit_date": str(sig["exit_date"].date()),
            "entry_cost": round(entry_cost, 2),
            "exit_proceeds": round(exit_proceeds, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / entry_cost * 100, 2) if entry_cost > 0 else 0,
            "hold_days": hold,
            "regime": "bull" if regime_bull.reindex(
                [sig["entry_date"]], method="ffill").iloc[0] else "bear",
        })
    return trades


def backtest_variant_c(earn_sigs):
    """Earnings Beat + ATM Calls (30-day expiry), hold 20d max."""
    trades = []
    for sig in earn_sigs:
        stock_price = sig["entry_price"]
        dte = 30
        iv_adj = 0.5  # elevated IV post-earnings
        premium = atm_call_premium(stock_price, dte, iv_adj)
        cost_with_fees = option_entry_cost(premium, 1)

        if cost_with_fees > MAX_POS_COST or cost_with_fees > CAPITAL:
            continue

        n_contracts = 1
        entry_cost = option_entry_cost(premium, n_contracts)
        hold = 20  # max hold for monthly

        stock_move = sig["exit_price_20d"] - sig["entry_price"]
        delta = 0.50
        raw_exit_prem = max(0, premium + delta * stock_move
                           - theta_decay(premium, dte, hold))
        exit_proceeds = option_exit_value(raw_exit_prem, n_contracts)
        pnl = exit_proceeds - entry_cost

        trades.append({
            "ticker": sig["ticker"],
            "entry_date": str(sig["entry_date"].date()),
            "exit_date": str(sig["exit_date_20d"].date()),
            "entry_cost": round(entry_cost, 2),
            "exit_proceeds": round(exit_proceeds, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / entry_cost * 100, 2) if entry_cost > 0 else 0,
            "hold_days": hold,
            "regime": "bull" if regime_bull.reindex(
                [sig["entry_date"]], method="ffill").iloc[0] else "bear",
        })
    return trades


def backtest_variant_d(earn_sigs):
    """Earnings Beat + Deep ITM Calls (10% ITM, 45-day), hold 40d."""
    trades = []
    for sig in earn_sigs:
        stock_price = sig["entry_price"]
        dte = 45
        itm_pct = 0.10
        premium = deep_itm_premium(stock_price, itm_pct, dte, iv_adj=0.3)
        cost_with_fees = option_entry_cost(premium, 1)

        if cost_with_fees > MAX_POS_COST or cost_with_fees > CAPITAL:
            continue

        n_contracts = 1
        entry_cost = option_entry_cost(premium, n_contracts)
        hold = 40

        stock_move = sig["exit_price_40d"] - sig["entry_price"]
        delta = 0.80  # Deep ITM
        raw_exit_prem = max(0, premium + delta * stock_move
                           - theta_decay(premium * 0.3, dte, hold))  # theta on time value only
        exit_proceeds = option_exit_value(raw_exit_prem, n_contracts)
        pnl = exit_proceeds - entry_cost

        trades.append({
            "ticker": sig["ticker"],
            "entry_date": str(sig["entry_date"].date()),
            "exit_date": str(sig["exit_date_40d"].date()),
            "entry_cost": round(entry_cost, 2),
            "exit_proceeds": round(exit_proceeds, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / entry_cost * 100, 2) if entry_cost > 0 else 0,
            "hold_days": hold,
            "regime": "bull" if regime_bull.reindex(
                [sig["entry_date"]], method="ffill").iloc[0] else "bear",
        })
    return trades


def backtest_variant_e(rsi_sigs, earn_sigs):
    """Combined: shares for RSI, calls for earnings."""
    trades = []

    # RSI trades as shares
    for sig in rsi_sigs:
        entry_price = sig["entry_price"]
        shares = int(MAX_POS_COST / entry_price)
        if shares < 1:
            continue
        cost = shares * entry_price * (1 + SLIPPAGE_SHARES)
        proceeds = shares * sig["exit_price"] * (1 - SLIPPAGE_SHARES)
        pnl = proceeds - cost

        trades.append({
            "ticker": sig["ticker"],
            "entry_date": str(sig["entry_date"].date()),
            "exit_date": str(sig["exit_date"].date()),
            "entry_cost": round(cost, 2),
            "exit_proceeds": round(proceeds, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / cost * 100, 2) if cost > 0 else 0,
            "hold_days": sig["hold_days"],
            "instrument": "shares",
            "regime": "bull" if regime_bull.reindex(
                [sig["entry_date"]], method="ffill").iloc[0] else "bear",
        })

    # Earnings trades as ATM calls (same as variant C)
    for sig in earn_sigs:
        stock_price = sig["entry_price"]
        dte = 30
        iv_adj = 0.5
        premium = atm_call_premium(stock_price, dte, iv_adj)
        cost_with_fees = option_entry_cost(premium, 1)
        if cost_with_fees > MAX_POS_COST or cost_with_fees > CAPITAL:
            continue

        entry_cost = option_entry_cost(premium, 1)
        hold = 20
        stock_move = sig["exit_price_20d"] - sig["entry_price"]
        delta = 0.50
        raw_exit_prem = max(0, premium + delta * stock_move
                           - theta_decay(premium, dte, hold))
        exit_proceeds = option_exit_value(raw_exit_prem, 1)
        pnl = exit_proceeds - entry_cost

        trades.append({
            "ticker": sig["ticker"],
            "entry_date": str(sig["entry_date"].date()),
            "exit_date": str(sig["exit_date_20d"].date()),
            "entry_cost": round(entry_cost, 2),
            "exit_proceeds": round(exit_proceeds, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / entry_cost * 100, 2) if entry_cost > 0 else 0,
            "hold_days": hold,
            "instrument": "call",
            "regime": "bull" if regime_bull.reindex(
                [sig["entry_date"]], method="ffill").iloc[0] else "bear",
        })
    return trades


def backtest_variant_f(earn_sigs):
    """Earnings Beat + Call Spreads ($5 wide). Max 3 concurrent positions."""
    trades = []
    active_exits = []  # track concurrent positions

    sorted_sigs = sorted(earn_sigs, key=lambda x: x["entry_date"])

    for sig in sorted_sigs:
        entry_date = sig["entry_date"]
        # Clean expired positions
        active_exits = [d for d in active_exits if d > entry_date]
        if len(active_exits) >= 3:
            continue

        stock_price = sig["entry_price"]
        dte = 30
        iv_adj = 0.5

        # Long ATM call (strike = stock_price)
        atm_prem = atm_call_premium(stock_price, dte, iv_adj)
        # Short call $5 OTM (strike = stock_price + 5)
        spread_width = 5.0
        otm_strike = stock_price + spread_width
        otm_strike_pct = spread_width / stock_price
        sigma = sig["sigma_20d"] if sig["sigma_20d"] > 0 else 0.3
        otm_prem = otm_call_premium(stock_price, dte, otm_strike_pct, sigma, iv_adj)
        net_debit = atm_prem - otm_prem  # per share

        # Floor net debit: spread can't cost less than ~40% of width for realistic pricing
        min_debit = spread_width * 0.35
        net_debit = max(net_debit, min_debit)

        cost_with_fees = spread_entry_cost(net_debit, 1)
        if cost_with_fees > MAX_POS_COST or cost_with_fees > CAPITAL:
            continue

        entry_cost = spread_entry_cost(net_debit, 1)
        hold = 20
        max_profit_per_share = spread_width  # $5 max

        # At exit with 10 DTE remaining, spread value =
        # min(max(0, exit_price - atm_strike), spread_width) + remaining_time_value
        exit_price = sig["exit_price_20d"]
        long_intrinsic = max(0, exit_price - stock_price)
        short_intrinsic = max(0, exit_price - otm_strike)
        spread_intrinsic = long_intrinsic - short_intrinsic  # capped at spread_width
        spread_intrinsic = min(spread_intrinsic, spread_width)

        # Remaining time value of the spread (10 DTE left, reduced)
        remaining_dte = dte - hold
        time_val = net_debit * 0.3 * max(0, remaining_dte / dte)  # decayed time value

        raw_exit_val = max(0, spread_intrinsic + time_val)
        # Can't exceed spread width
        raw_exit_val = min(raw_exit_val, spread_width)
        exit_proceeds = spread_exit_value(raw_exit_val, 1)
        pnl = exit_proceeds - entry_cost

        trades.append({
            "ticker": sig["ticker"],
            "entry_date": str(sig["entry_date"].date()),
            "exit_date": str(sig["exit_date_20d"].date()),
            "entry_cost": round(entry_cost, 2),
            "exit_proceeds": round(exit_proceeds, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / entry_cost * 100, 2) if entry_cost > 0 else 0,
            "hold_days": hold,
            "regime": "bull" if regime_bull.reindex(
                [sig["entry_date"]], method="ffill").iloc[0] else "bear",
        })
        active_exits.append(sig["exit_date_20d"])

    return trades


# ── Run All Variants ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RUNNING 6 OPTION VARIANTS")
print("=" * 70)

variants = {
    "A_RSI_ATM_Calls": backtest_variant_a(rsi_signals),
    "B_RSI_OTM_Calls": backtest_variant_b(rsi_signals),
    "C_Earnings_ATM_Calls": backtest_variant_c(earnings_signals),
    "D_Earnings_DeepITM": backtest_variant_d(earnings_signals),
    "E_Combined_Shares_Calls": backtest_variant_e(rsi_signals, earnings_signals),
    "F_Earnings_CallSpreads": backtest_variant_f(earnings_signals),
}


# ── Analytics ────────────────────────────────────────────────────────────
def compute_metrics(trades, variant_name):
    """Compute performance metrics with 5-gate validation."""
    if not trades:
        return {
            "variant": variant_name,
            "n_trades": 0,
            "gates_passed": 0,
            "gate_detail": "NO TRADES",
            "total_pnl": 0,
            "sharpe": 0,
            "sortino": 0,
        }

    pnls = np.array([t["pnl"] for t in trades])
    returns = np.array([t["return_pct"] / 100 for t in trades])
    n = len(trades)
    total_pnl = float(np.sum(pnls))

    # Win rate
    wr = float(np.mean(pnls > 0)) * 100

    # Annualized Sharpe (assume ~1 trade per week average)
    avg_hold = np.mean([t["hold_days"] for t in trades])
    trades_per_year = 252 / max(avg_hold, 1)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) if np.std(returns) > 0 else 1e-6
    sharpe = mean_ret / std_ret * np.sqrt(trades_per_year)

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-6
    sortino = mean_ret / downside_std * np.sqrt(trades_per_year)

    # Profit Factor
    gross_profit = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 1e-6
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max Drawdown (on cumulative P&L)
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(np.min(dd))
    max_dd_pct = max_dd / CAPITAL * 100 if CAPITAL > 0 else 0

    # Avg trade
    avg_pnl = float(np.mean(pnls))
    median_pnl = float(np.median(pnls))

    # Per-regime Sharpe
    bull_rets = [t["return_pct"] / 100 for t in trades if t["regime"] == "bull"]
    bear_rets = [t["return_pct"] / 100 for t in trades if t["regime"] == "bear"]

    def regime_sharpe(rets):
        if len(rets) < 3:
            return 0.0
        r = np.array(rets)
        s = np.std(r)
        if s < 1e-8:
            return 0.0
        return float(np.mean(r) / s * np.sqrt(trades_per_year))

    sharpe_bull = regime_sharpe(bull_rets)
    sharpe_bear = regime_sharpe(bear_rets)
    regime_gap = (abs(sharpe_bull - sharpe_bear) /
                  max(abs(sharpe_bull), abs(sharpe_bear), 1e-6))

    # Permutation test (Sharpe)
    observed_sharpe = sharpe
    count_ge = 0
    for _ in range(N_PERM):
        perm_rets = np.random.permutation(returns)
        perm_sharpe = np.mean(perm_rets) / (np.std(perm_rets) + 1e-8) * np.sqrt(trades_per_year)
        if perm_sharpe >= observed_sharpe:
            count_ge += 1
    perm_p = count_ge / N_PERM

    # 5-Gate Validation
    gates = {
        "sharpe_gt_0.5": sharpe > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime_gap < 0.5,
        "mdd_gt_neg50pct": max_dd_pct > -50,
        "trades_ge_20": n >= 20,
    }
    gates_passed = sum(gates.values())
    gate_str = " | ".join(
        f"{'✓' if v else '✗'} {k}" for k, v in gates.items()
    )

    # Effective leverage
    avg_cost = np.mean([t["entry_cost"] for t in trades])
    avg_stock_exposure = np.mean([
        closes.get(t["ticker"], pd.Series(dtype=float)).reindex(
            [pd.Timestamp(t["entry_date"])], method="ffill"
        ).iloc[0] * CONTRACT_MULT
        if t.get("instrument", "call") == "call" else
        t["entry_cost"]
        for t in trades
    ]) if trades else 0
    leverage = avg_stock_exposure / avg_cost if avg_cost > 0 else 1

    return {
        "variant": variant_name,
        "n_trades": n,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / CAPITAL * 100, 2),
        "avg_pnl": round(avg_pnl, 2),
        "median_pnl": round(median_pnl, 2),
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown_dollar": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "avg_hold_days": round(avg_hold, 1),
        "effective_leverage": round(leverage, 1),
        "sharpe_bull": round(sharpe_bull, 2),
        "sharpe_bear": round(sharpe_bear, 2),
        "regime_gap": round(regime_gap, 2),
        "perm_p_value": round(perm_p, 4),
        "gates_passed": gates_passed,
        "gate_detail": gate_str,
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
    }


# ── Compute All Results ──────────────────────────────────────────────────
results = {}
for name, trades in variants.items():
    print(f"\n{'─' * 50}")
    print(f"Variant {name}: {len(trades)} trades")
    metrics = compute_metrics(trades, name)
    results[name] = metrics

    print(f"  Total P&L: ${metrics['total_pnl']:+,.2f} "
          f"({metrics['total_return_pct']:+.1f}% on ${CAPITAL})")
    print(f"  WR: {metrics['win_rate']:.1f}% | PF: {metrics['profit_factor']:.2f} | "
          f"Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f}")
    print(f"  MaxDD: ${metrics['max_drawdown_dollar']:,.2f} ({metrics['max_drawdown_pct']:.1f}%)")
    print(f"  Avg Hold: {metrics['avg_hold_days']:.1f}d | "
          f"Bull/Bear trades: {metrics['bull_trades']}/{metrics['bear_trades']}")
    print(f"  Sharpe Bull: {metrics['sharpe_bull']:.2f} | Bear: {metrics['sharpe_bear']:.2f} | "
          f"Gap: {metrics['regime_gap']:.2f}")
    print(f"  Perm p-value: {metrics['perm_p_value']:.4f}")
    print(f"  Gates: {metrics['gates_passed']}/5 — {metrics['gate_detail']}")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY: OPTIONS LEVERAGE BACKTEST")
print("=" * 70)
print(f"{'Variant':<30} {'Trades':>6} {'P&L':>10} {'Sharpe':>7} {'WR':>6} "
      f"{'MDD%':>7} {'Gates':>6}")
print("-" * 70)
for name, m in results.items():
    marker = " ★" if m["gates_passed"] >= 4 else ""
    print(f"{name:<30} {m['n_trades']:>6} ${m['total_pnl']:>8,.2f} "
          f"{m['sharpe']:>7.2f} {m['win_rate']:>5.1f}% "
          f"{m['max_drawdown_pct']:>6.1f}% {m['gates_passed']:>3}/5{marker}")

# Find champion
if results:
    champ_name = max(results, key=lambda k: (
        results[k]["gates_passed"],
        results[k]["sharpe"] if results[k]["n_trades"] >= 20 else -999
    ))
    champ = results[champ_name]
    print(f"\n★ CHAMPION: {champ_name}")
    print(f"  Sharpe {champ['sharpe']:.2f} | Sortino {champ['sortino']:.2f} | "
          f"WR {champ['win_rate']:.1f}% | PF {champ['profit_factor']:.2f}")
    print(f"  Total P&L: ${champ['total_pnl']:+,.2f} on ${CAPITAL} "
          f"({champ['total_return_pct']:+.1f}%)")
    print(f"  Gates: {champ['gates_passed']}/5")

# ── Save Results ─────────────────────────────────────────────────────────
output = {
    "metadata": {
        "script": "options_leverage_signals_backtest.py",
        "run_date": datetime.now().isoformat(),
        "account_size": CAPITAL,
        "max_position_cost": MAX_POS_COST,
        "oot_period": f"{OOT_START} to {END}",
        "universe": TICKERS,
        "n_permutations": N_PERM,
        "commission_per_contract": COMMISSION_PER_CONTRACT,
        "bid_ask_haircut": BID_ASK_HAIRCUT,
    },
    "signal_counts": {
        "rsi_signals": len(rsi_signals),
        "earnings_signals": len(earnings_signals),
    },
    "variant_results": results,
    "champion": champ_name if results else None,
    "five_gate_summary": {
        name: {
            "passed": m["gates_passed"],
            "sharpe": m["sharpe"],
            "total_pnl": m["total_pnl"],
        }
        for name, m in results.items()
    },
}

out_path = Path("/home/jupiter/Lvl3Quant/data/options_leverage_signals_results.json")
out_path.write_text(json.dumps(output, indent=2, default=str))
print(f"\nResults saved to {out_path}")
print("Done.")
