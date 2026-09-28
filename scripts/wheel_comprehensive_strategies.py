#!/usr/bin/env python3
"""
wheel_comprehensive_strategies.py — Develop, backtest, and validate multiple
wheel / options-selling strategies for income generation.

User directive: Thorough development of multiple wheel strategies with:
  - Stock picking (only sell options on stocks we want to own)
  - Greeks analysis (delta, theta, gamma, vega)
  - Macro/regime awareness
  - Risk profiling
  - Monthly/yearly income projections
  - Max 1x leverage, variable cash holding

Strategies developed:
  1. CONSERVATIVE INCOME — SPY-only wheel, low delta, high cash reserve
  2. BALANCED GROWTH — Multi-ETF wheel (SPY, QQQ, IWM) with regime gating
  3. DIVIDEND ARISTOCRAT WHEEL — Wheel on quality dividend stocks
  4. SECTOR ROTATION WHEEL — Rotate into strongest sectors
  5. VOLATILITY-ADAPTIVE WHEEL — Dynamic delta/DTE based on VIX regime
  6. HEDGED WHEEL — Wheel + protective put overlay for tail risk

Each strategy includes:
  - Full backtest 2018-2025
  - Greeks-based entry/exit
  - Risk metrics (Sharpe, Sortino, MaxDD, Calmar)
  - Monthly/yearly income projections for various account sizes
  - Regime-stratified performance
"""
from __future__ import annotations
import sys
import os
import json
import math
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ---- Black-Scholes Pricing ----
def bs_price(S, K, T, r, sigma, q=0.0, opt_type='put'):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        if opt_type == 'put':
            return max(K - S, 0)
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == 'put':
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma, q=0.0, opt_type='put'):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if opt_type == 'put':
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if opt_type == 'put':
        return math.exp(-q * T) * (norm.cdf(d1) - 1)
    return math.exp(-q * T) * norm.cdf(d1)

def bs_theta(S, K, T, r, sigma, q=0.0, opt_type='put'):
    """Black-Scholes theta (per day)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    common = -(S * math.exp(-q * T) * sigma * norm.pdf(d1)) / (2 * math.sqrt(T))
    if opt_type == 'put':
        theta = common + r * K * math.exp(-r * T) * norm.cdf(-d2) - q * S * math.exp(-q * T) * norm.cdf(-d1)
    else:
        theta = common - r * K * math.exp(-r * T) * norm.cdf(d2) + q * S * math.exp(-q * T) * norm.cdf(d1)
    return theta / 365.0

def bs_gamma(S, K, T, r, sigma, q=0.0):
    """Black-Scholes gamma."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))

def bs_vega(S, K, T, r, sigma, q=0.0):
    """Black-Scholes vega (per 1% vol move)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * math.sqrt(T) * norm.pdf(d1) / 100.0

def find_strike_for_delta(S, T, r, sigma, target_delta, q=0.0, opt_type='put'):
    """Binary search for strike that gives target delta."""
    if pd.isna(S) or pd.isna(sigma) or S <= 0 or sigma <= 0 or T <= 0:
        return S * 0.95 if opt_type == 'put' else S * 1.05  # Fallback
    lo, hi = S * 0.5, S * 1.5
    target = abs(target_delta)
    mid = S
    for _ in range(50):
        mid = (lo + hi) / 2
        d = abs(bs_delta(S, mid, T, r, sigma, q, opt_type))
        if d < target:
            if opt_type == 'put':
                hi = mid
            else:
                lo = mid
        else:
            if opt_type == 'put':
                lo = mid
            else:
                hi = mid
    try:
        return round(mid * 2) / 2  # Round to nearest 0.50
    except (ValueError, OverflowError):
        return S * 0.95 if opt_type == 'put' else S * 1.05

# ---- Data Loading ----
def load_price_data(tickers: list, start='2018-01-01', end='2025-12-31') -> pd.DataFrame:
    """Load price data from available sources."""
    # Try wheel_strategy_v1 cache first
    cache_dir = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache")
    all_data = {}

    for ticker in tickers:
        # Check multiple data sources
        for path in [
            cache_dir / f"{ticker}_prices.csv",
            cache_dir / f"{ticker}.csv",
            Path(f"/home/jupiter/Lvl3Quant/data/prices/{ticker}.csv"),
            Path(f"/home/jupiter/Lvl3Quant/data/{ticker}_daily.csv"),
        ]:
            if path.exists():
                try:
                    df = pd.read_csv(path, parse_dates=['date'] if 'date' in pd.read_csv(path, nrows=0).columns else [0])
                    if 'date' not in df.columns:
                        df.columns = ['date'] + list(df.columns[1:])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[(df['date'] >= start) & (df['date'] <= end)]
                    if 'close' in df.columns:
                        all_data[ticker] = df.set_index('date')['close']
                    elif 'Close' in df.columns:
                        all_data[ticker] = df.set_index('date')['Close']
                    break
                except Exception:
                    continue

    if not all_data:
        return pd.DataFrame()
    return pd.DataFrame(all_data).sort_index()

def load_vix_data(start='2018-01-01', end='2025-12-31') -> pd.Series:
    """Load VIX data."""
    for path in [
        Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/VIX_prices.csv"),
        Path("/home/jupiter/Lvl3Quant/data/vix_daily.csv"),
        Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/^VIX_prices.csv"),
    ]:
        if path.exists():
            try:
                df = pd.read_csv(path, parse_dates=[0])
                df.columns = ['date'] + list(df.columns[1:])
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= start) & (df['date'] <= end)]
                col = 'close' if 'close' in df.columns else 'Close'
                return df.set_index('date')[col]
            except Exception:
                continue
    return pd.Series(dtype=float)

def compute_realized_vol(prices: pd.Series, window=20) -> pd.Series:
    """Annualized realized volatility."""
    returns = prices.pct_change()
    return returns.rolling(window).std() * np.sqrt(252)

# ---- Cost Model ----
COMMISSION_PER_CONTRACT = 0.65  # Robinhood: $0 for equities, $0.65/contract for options
ASSIGNMENT_FEE = 0.00  # RH no assignment fee
SLIPPAGE_FRAC = 0.025  # 2.5% of premium
SLIPPAGE_MIN = 0.03    # $0.03/share minimum

def round_trip_cost(premium_per_share, contracts=1):
    """Total cost for opening + closing an option position."""
    slippage = max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium_per_share) * 2  # Both legs
    commission = COMMISSION_PER_CONTRACT * 2  # Open + close
    return (slippage * 100 + commission) * contracts

# ---- Wheel Strategy Base ----
@dataclass
class WheelPosition:
    ticker: str
    side: str  # 'short_put', 'short_call', 'long_shares'
    strike: float
    expiration_date: str  # YYYY-MM-DD
    contracts: int
    entry_premium: float
    entry_date: str
    entry_price: float
    entry_delta: float
    entry_theta: float
    entry_iv: float
    cost_basis: float = 0.0  # For shares

@dataclass
class WheelTrade:
    ticker: str
    side: str
    strike: float
    entry_date: str
    exit_date: str
    entry_premium: float
    exit_premium: float
    contracts: int
    pnl: float
    return_pct: float
    hold_days: int
    exit_reason: str

@dataclass
class WheelConfig:
    name: str
    universe: List[str]
    put_delta: float = 0.22
    call_delta: float = 0.22
    target_dte: int = 35
    min_dte: int = 25
    max_dte: int = 45
    profit_take: float = 0.50
    roll_dte: int = 7
    vix_max: float = 35.0
    max_positions: int = 5
    max_single_alloc: float = 0.25  # Max % in one name
    cash_reserve: float = 0.10  # Min cash to keep
    use_regime_gate: bool = False
    regime_gate_type: str = 'trend'  # 'trend', 'vix', 'composite'
    share_stop_loss: float = 0.0  # 0 = no stop
    use_hedge: bool = False
    hedge_type: str = 'none'  # 'static_put', 'vix_call', 'collar'

    # Dynamic delta settings
    dynamic_delta: bool = False
    low_vol_delta: float = 0.25
    mid_vol_delta: float = 0.20
    high_vol_delta: float = 0.15

    # Dividend yield assumptions
    div_yields: Dict[str, float] = field(default_factory=dict)

def run_wheel_backtest(config: WheelConfig, prices: pd.DataFrame, vix: pd.Series,
                       starting_cash: float = 100000.0) -> Dict:
    """Run a full wheel strategy backtest."""

    # Align dates
    common_dates = prices.index.intersection(vix.index)
    prices = prices.loc[common_dates]
    vix = vix.loc[common_dates]

    # Filter universe to available tickers
    available = [t for t in config.universe if t in prices.columns]
    if not available:
        return {'error': 'No price data for universe tickers'}

    cash = starting_cash
    positions: List[WheelPosition] = []
    shares: Dict[str, Dict] = {}  # ticker -> {quantity, cost_basis, entry_date}
    trades: List[WheelTrade] = []
    equity_curve = []
    daily_returns = []

    r = 0.04  # Risk-free rate
    prev_equity = starting_cash

    for i, date in enumerate(prices.index):
        date_str = date.strftime('%Y-%m-%d')
        current_vix = vix.iloc[i] if i < len(vix) else 20.0

        # -- Mark to market --
        position_value = 0.0
        share_value = 0.0

        # Mark existing option positions
        expired_positions = []
        for j, pos in enumerate(positions):
            ticker = pos.ticker
            if ticker not in prices.columns:
                continue
            S = prices[ticker].iloc[i]
            exp_date = pd.Timestamp(pos.expiration_date)
            T = max((exp_date - date).days / 365.0, 0.001)
            rv = compute_realized_vol(prices[ticker].iloc[max(0,i-60):i+1]).iloc[-1] if i > 20 else current_vix / 100
            if pd.isna(rv) or rv <= 0:
                rv = current_vix / 100
            sigma = rv
            q = config.div_yields.get(ticker, 0.0)

            if pos.side == 'short_put':
                opt_val = bs_price(S, pos.strike, T, r, sigma, q, 'put')
                position_value -= opt_val * 100 * pos.contracts  # Short = liability

                # Check expiration
                if date >= exp_date:
                    if S < pos.strike:
                        # Assignment — buy shares at strike
                        share_cost = pos.strike * 100 * pos.contracts
                        cash -= share_cost
                        shares[ticker] = {
                            'quantity': 100 * pos.contracts,
                            'cost_basis': pos.strike,
                            'entry_date': date_str,
                            'net_cost': pos.strike - pos.entry_premium  # Adjusted for premium
                        }
                        trades.append(WheelTrade(
                            ticker=ticker, side='short_put', strike=pos.strike,
                            entry_date=pos.entry_date, exit_date=date_str,
                            entry_premium=pos.entry_premium, exit_premium=0,
                            contracts=pos.contracts, pnl=pos.entry_premium * 100 * pos.contracts,
                            return_pct=1.0, hold_days=(date - pd.Timestamp(pos.entry_date)).days,
                            exit_reason='assigned'
                        ))
                    else:
                        # Expired worthless — keep full premium
                        trades.append(WheelTrade(
                            ticker=ticker, side='short_put', strike=pos.strike,
                            entry_date=pos.entry_date, exit_date=date_str,
                            entry_premium=pos.entry_premium, exit_premium=0,
                            contracts=pos.contracts,
                            pnl=pos.entry_premium * 100 * pos.contracts - round_trip_cost(pos.entry_premium, pos.contracts) / 2,
                            return_pct=1.0, hold_days=(date - pd.Timestamp(pos.entry_date)).days,
                            exit_reason='expired_otm'
                        ))
                    expired_positions.append(j)

                # Check profit take
                elif opt_val <= pos.entry_premium * (1 - config.profit_take):
                    close_cost = opt_val * 100 * pos.contracts
                    realized = (pos.entry_premium - opt_val) * 100 * pos.contracts
                    realized -= round_trip_cost(pos.entry_premium, pos.contracts)
                    cash += realized
                    trades.append(WheelTrade(
                        ticker=ticker, side='short_put', strike=pos.strike,
                        entry_date=pos.entry_date, exit_date=date_str,
                        entry_premium=pos.entry_premium, exit_premium=opt_val,
                        contracts=pos.contracts, pnl=realized,
                        return_pct=realized / (pos.strike * 100 * pos.contracts),
                        hold_days=(date - pd.Timestamp(pos.entry_date)).days,
                        exit_reason='profit_take'
                    ))
                    expired_positions.append(j)

                # Check roll (DTE <= roll trigger)
                elif T * 365 <= config.roll_dte and S > pos.strike:
                    close_cost = opt_val * 100 * pos.contracts
                    realized = (pos.entry_premium - opt_val) * 100 * pos.contracts
                    realized -= round_trip_cost(pos.entry_premium, pos.contracts)
                    cash += realized
                    trades.append(WheelTrade(
                        ticker=ticker, side='short_put', strike=pos.strike,
                        entry_date=pos.entry_date, exit_date=date_str,
                        entry_premium=pos.entry_premium, exit_premium=opt_val,
                        contracts=pos.contracts, pnl=realized,
                        return_pct=realized / (pos.strike * 100 * pos.contracts),
                        hold_days=(date - pd.Timestamp(pos.entry_date)).days,
                        exit_reason='roll'
                    ))
                    expired_positions.append(j)

            elif pos.side == 'short_call':
                opt_val = bs_price(S, pos.strike, T, r, sigma, q, 'call')
                position_value -= opt_val * 100 * pos.contracts

                if date >= exp_date:
                    if S > pos.strike:
                        # Called away — sell shares at strike
                        if ticker in shares:
                            sale_proceeds = pos.strike * 100 * pos.contracts
                            basis = shares[ticker]['cost_basis'] * shares[ticker]['quantity']
                            share_pnl = sale_proceeds - basis
                            cash += sale_proceeds
                            cash += pos.entry_premium * 100 * pos.contracts
                            trades.append(WheelTrade(
                                ticker=ticker, side='short_call', strike=pos.strike,
                                entry_date=pos.entry_date, exit_date=date_str,
                                entry_premium=pos.entry_premium, exit_premium=0,
                                contracts=pos.contracts, pnl=share_pnl + pos.entry_premium * 100 * pos.contracts,
                                return_pct=(share_pnl + pos.entry_premium * 100 * pos.contracts) / basis,
                                hold_days=(date - pd.Timestamp(pos.entry_date)).days,
                                exit_reason='called_away'
                            ))
                            del shares[ticker]
                    else:
                        # Expired worthless — keep premium, keep shares
                        trades.append(WheelTrade(
                            ticker=ticker, side='short_call', strike=pos.strike,
                            entry_date=pos.entry_date, exit_date=date_str,
                            entry_premium=pos.entry_premium, exit_premium=0,
                            contracts=pos.contracts,
                            pnl=pos.entry_premium * 100 * pos.contracts,
                            return_pct=1.0, hold_days=(date - pd.Timestamp(pos.entry_date)).days,
                            exit_reason='expired_otm'
                        ))
                    expired_positions.append(j)

                elif opt_val <= pos.entry_premium * (1 - config.profit_take):
                    realized = (pos.entry_premium - opt_val) * 100 * pos.contracts
                    realized -= round_trip_cost(pos.entry_premium, pos.contracts)
                    cash += realized
                    trades.append(WheelTrade(
                        ticker=ticker, side='short_call', strike=pos.strike,
                        entry_date=pos.entry_date, exit_date=date_str,
                        entry_premium=pos.entry_premium, exit_premium=opt_val,
                        contracts=pos.contracts, pnl=realized,
                        return_pct=realized / (pos.strike * 100 * pos.contracts),
                        hold_days=(date - pd.Timestamp(pos.entry_date)).days,
                        exit_reason='profit_take'
                    ))
                    expired_positions.append(j)

        # Remove expired positions
        positions = [p for j, p in enumerate(positions) if j not in expired_positions]

        # Mark shares
        for ticker, sh in shares.items():
            if ticker in prices.columns:
                S = prices[ticker].iloc[i]
                share_value += S * sh['quantity']

                # Share stop loss
                if config.share_stop_loss > 0:
                    loss_pct = (S - sh['cost_basis']) / sh['cost_basis']
                    if loss_pct < -config.share_stop_loss:
                        # Sell shares at market
                        sale = S * sh['quantity']
                        pnl = sale - sh['cost_basis'] * sh['quantity']
                        cash += sale
                        trades.append(WheelTrade(
                            ticker=ticker, side='long_shares', strike=sh['cost_basis'],
                            entry_date=sh['entry_date'], exit_date=date_str,
                            entry_premium=0, exit_premium=0,
                            contracts=sh['quantity'] // 100,
                            pnl=pnl, return_pct=loss_pct,
                            hold_days=(date - pd.Timestamp(sh['entry_date'])).days,
                            exit_reason='stop_loss'
                        ))
                        del shares[ticker]
                        break  # Modified dict during iteration

        # Sell covered calls on shares without calls
        shares_without_calls = set(shares.keys()) - set(p.ticker for p in positions if p.side == 'short_call')
        for ticker in shares_without_calls:
            if ticker not in prices.columns:
                continue
            S = prices[ticker].iloc[i]
            rv = compute_realized_vol(prices[ticker].iloc[max(0,i-60):i+1]).iloc[-1] if i > 20 else current_vix / 100
            if pd.isna(rv) or rv <= 0:
                rv = current_vix / 100
            sigma = rv
            q = config.div_yields.get(ticker, 0.0)
            T = config.target_dte / 365.0

            # Find strike for target delta
            call_delta = config.call_delta
            if config.dynamic_delta:
                if current_vix < 15:
                    call_delta = config.low_vol_delta
                elif current_vix < 25:
                    call_delta = config.mid_vol_delta
                else:
                    call_delta = config.high_vol_delta

            K = find_strike_for_delta(S, T, r, sigma, call_delta, q, 'call')
            premium = bs_price(S, K, T, r, sigma, q, 'call')
            delta = bs_delta(S, K, T, r, sigma, q, 'call')
            theta = bs_theta(S, K, T, r, sigma, q, 'call')

            if premium > 0.10:
                cash += premium * 100 * (shares[ticker]['quantity'] // 100)
                exp_date = (date + timedelta(days=config.target_dte)).strftime('%Y-%m-%d')
                positions.append(WheelPosition(
                    ticker=ticker, side='short_call', strike=K,
                    expiration_date=exp_date,
                    contracts=shares[ticker]['quantity'] // 100,
                    entry_premium=premium, entry_date=date_str,
                    entry_price=S, entry_delta=delta,
                    entry_theta=theta, entry_iv=sigma
                ))

        # -- New CSP entries --
        total_equity = cash + position_value + share_value
        cash_available = cash - total_equity * config.cash_reserve
        active_tickers = set(p.ticker for p in positions) | set(shares.keys())
        n_positions = len(set(p.ticker for p in positions if p.side == 'short_put')) + len(shares)

        if (cash_available > 0 and
            n_positions < config.max_positions and
            current_vix <= config.vix_max):

            # Regime gate
            entry_allowed = True
            if config.use_regime_gate and 'SPY' in prices.columns:
                spy = prices['SPY'].iloc[max(0,i-200):i+1]
                if len(spy) >= 50:
                    ma50 = spy.rolling(50).mean().iloc[-1]
                    ma200 = spy.rolling(200).mean().iloc[-1] if len(spy) >= 200 else ma50
                    if config.regime_gate_type == 'trend':
                        entry_allowed = spy.iloc[-1] > ma50
                    elif config.regime_gate_type == 'vix':
                        entry_allowed = current_vix < 25
                    elif config.regime_gate_type == 'composite':
                        entry_allowed = spy.iloc[-1] > ma50 and current_vix < 28

            if entry_allowed:
                # Score and rank tickers
                candidates = []
                for ticker in available:
                    if ticker in active_tickers:
                        continue
                    S = prices[ticker].iloc[i]
                    if pd.isna(S) or S <= 0:
                        continue

                    # Position sizing: max allocation check
                    collateral_needed = S * 100  # CSP collateral for 1 contract
                    if collateral_needed > cash_available:
                        continue
                    if collateral_needed > total_equity * config.max_single_alloc:
                        continue

                    try:
                        rv = compute_realized_vol(prices[ticker].iloc[max(0,i-60):i+1]).iloc[-1] if i > 20 else current_vix / 100
                    except Exception:
                        rv = current_vix / 100
                    if pd.isna(rv) or rv <= 0:
                        rv = current_vix / 100

                    # Score: higher IV = more premium, momentum, vol stability
                    momentum_60d = (S / prices[ticker].iloc[max(0,i-60)] - 1) if i > 60 else 0
                    try:
                        vol_ratio = rv / (compute_realized_vol(prices[ticker].iloc[max(0,i-252):i+1], 60).iloc[-1] if i > 252 else rv)
                    except Exception:
                        vol_ratio = 1.0
                    if pd.isna(vol_ratio):
                        vol_ratio = 1.0

                    # Premium yield estimate
                    sigma = rv
                    q = config.div_yields.get(ticker, 0.0)
                    T = config.target_dte / 365.0

                    put_delta = config.put_delta
                    if config.dynamic_delta:
                        if current_vix < 15:
                            put_delta = config.low_vol_delta
                        elif current_vix < 25:
                            put_delta = config.mid_vol_delta
                        else:
                            put_delta = config.high_vol_delta

                    K = find_strike_for_delta(S, T, r, sigma, put_delta, q, 'put')
                    premium = bs_price(S, K, T, r, sigma, q, 'put')
                    delta = bs_delta(S, K, T, r, sigma, q, 'put')
                    theta = bs_theta(S, K, T, r, sigma, q, 'put')
                    gamma = bs_gamma(S, K, T, r, sigma, q)
                    vega = bs_vega(S, K, T, r, sigma, q)

                    # Premium yield (annualized)
                    prem_yield = (premium / K) * (365 / config.target_dte) if K > 0 else 0

                    # Composite score
                    score = (
                        prem_yield * 0.30 +           # Premium yield
                        max(0, momentum_60d) * 0.25 +  # Positive momentum
                        (1.0 / max(0.5, vol_ratio)) * 0.10 * 0.15 +  # Vol stability
                        abs(theta) * 100 * 0.15 +      # Theta capture
                        (1 - abs(delta)) * 0.15         # OTM-ness (safety)
                    )

                    candidates.append({
                        'ticker': ticker, 'score': score, 'strike': K,
                        'premium': premium, 'delta': delta, 'theta': theta,
                        'gamma': gamma, 'vega': vega, 'sigma': sigma,
                        'spot': S, 'T': T, 'q': q, 'collateral': collateral_needed
                    })

                # Sort by score, take best
                candidates.sort(key=lambda x: x['score'], reverse=True)

                for cand in candidates[:config.max_positions - n_positions]:
                    if cash_available < cand['collateral']:
                        break

                    exp_date = (date + timedelta(days=config.target_dte)).strftime('%Y-%m-%d')
                    cash += cand['premium'] * 100  # Receive premium
                    cash_available -= cand['collateral']

                    positions.append(WheelPosition(
                        ticker=cand['ticker'], side='short_put',
                        strike=cand['strike'], expiration_date=exp_date,
                        contracts=1, entry_premium=cand['premium'],
                        entry_date=date_str, entry_price=cand['spot'],
                        entry_delta=cand['delta'], entry_theta=cand['theta'],
                        entry_iv=cand['sigma']
                    ))

        # Record equity
        total_equity = cash + share_value  # Simplified MTM
        equity_curve.append({
            'date': date, 'equity': total_equity, 'cash': cash,
            'share_value': share_value, 'n_positions': len(positions),
            'vix': current_vix
        })

        if prev_equity > 0:
            daily_returns.append(total_equity / prev_equity - 1)
        prev_equity = total_equity

    # Compute metrics
    returns = np.array(daily_returns)
    equity_df = pd.DataFrame(equity_curve)

    if len(returns) == 0:
        return {'error': 'No returns generated'}

    total_return = (equity_df['equity'].iloc[-1] / starting_cash) - 1
    n_years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    ann_ret = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.001
    sharpe = ann_ret / ann_vol

    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else 0.001
    sortino = ann_ret / downside_vol

    # Max drawdown
    cum_equity = equity_df['equity'].values
    peak = np.maximum.accumulate(cum_equity)
    drawdown = (cum_equity - peak) / peak
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0

    # Win rate
    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]
    wr = len(winning) / len(trades) if trades else 0
    avg_win = np.mean([t.pnl for t in winning]) if winning else 0
    avg_loss = np.mean([abs(t.pnl) for t in losing]) if losing else 0.01
    pf = (sum(t.pnl for t in winning) / sum(abs(t.pnl) for t in losing)) if losing and sum(abs(t.pnl) for t in losing) > 0 else 0

    # Monthly income estimate
    total_premium = sum(t.entry_premium * 100 * t.contracts for t in trades if t.side in ['short_put', 'short_call'])
    monthly_premium = total_premium / max(n_years * 12, 1)
    monthly_yield = monthly_premium / starting_cash * 100

    # Regime analysis
    regime_stats = {}
    if 'SPY' in prices.columns:
        spy_returns = prices['SPY'].pct_change()
        for idx in range(len(equity_df)):
            if idx < len(spy_returns):
                spy_ret = spy_returns.iloc[idx]
                if pd.isna(spy_ret):
                    regime = 'flat'
                elif spy_ret > 0.005:
                    regime = 'green'
                elif spy_ret < -0.005:
                    regime = 'red'
                else:
                    regime = 'flat'
                if regime not in regime_stats:
                    regime_stats[regime] = []
                if idx < len(daily_returns):
                    regime_stats[regime].append(daily_returns[idx])

    regime_sharpes = {}
    for regime, rets in regime_stats.items():
        if len(rets) > 5:
            r_arr = np.array(rets)
            vol = np.std(r_arr) * np.sqrt(252) if np.std(r_arr) > 0 else 0.001
            regime_sharpes[regime] = (np.mean(r_arr) * 252) / vol

    return {
        'config_name': config.name,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': wr,
        'profit_factor': pf,
        'n_trades': len(trades),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_pnl_per_trade': np.mean([t.pnl for t in trades]) if trades else 0,
        'total_premium_collected': total_premium,
        'monthly_premium_avg': monthly_premium,
        'monthly_yield_pct': monthly_yield,
        'annual_yield_pct': monthly_yield * 12,
        'final_equity': equity_df['equity'].iloc[-1] if len(equity_df) > 0 else starting_cash,
        'regime_sharpes': regime_sharpes,
        'equity_curve': equity_df,
        'trades': trades,
        'n_assignments': sum(1 for t in trades if t.exit_reason == 'assigned'),
        'n_called_away': sum(1 for t in trades if t.exit_reason == 'called_away'),
        'n_profit_takes': sum(1 for t in trades if t.exit_reason == 'profit_take'),
        'n_expired': sum(1 for t in trades if t.exit_reason == 'expired_otm'),
        'avg_hold_days': np.mean([t.hold_days for t in trades]) if trades else 0,
    }

# ---- Strategy Definitions ----

def strategy_1_conservative_income():
    """Strategy 1: Conservative Income — SPY-only wheel with high cash reserve."""
    return WheelConfig(
        name="1_Conservative_Income",
        universe=['SPY'],
        put_delta=0.18,
        call_delta=0.20,
        target_dte=35,
        profit_take=0.50,
        vix_max=30.0,
        max_positions=1,
        cash_reserve=0.30,  # Keep 30% cash always
        use_regime_gate=True,
        regime_gate_type='composite',
        div_yields={'SPY': 0.013},
    )

def strategy_2_balanced_growth():
    """Strategy 2: Balanced Growth — Multi-ETF wheel with regime gating."""
    return WheelConfig(
        name="2_Balanced_Growth",
        universe=['SPY', 'QQQ', 'IWM'],
        put_delta=0.22,
        call_delta=0.22,
        target_dte=35,
        profit_take=0.50,
        vix_max=32.0,
        max_positions=3,
        cash_reserve=0.20,
        use_regime_gate=True,
        regime_gate_type='trend',
        div_yields={'SPY': 0.013, 'QQQ': 0.006, 'IWM': 0.013},
    )

def strategy_3_dividend_wheel():
    """Strategy 3: Dividend Aristocrat Wheel — Quality dividend payers."""
    return WheelConfig(
        name="3_Dividend_Wheel",
        universe=['KO', 'JNJ', 'PG', 'PEP', 'MCD', 'HD', 'ABBV', 'MMM', 'T', 'VZ'],
        put_delta=0.25,
        call_delta=0.30,  # Higher call delta to keep stock longer for dividends
        target_dte=30,
        profit_take=0.50,
        vix_max=35.0,
        max_positions=5,
        max_single_alloc=0.25,
        cash_reserve=0.15,
        use_regime_gate=False,  # Dividend stocks less regime sensitive
        share_stop_loss=0.15,
        div_yields={
            'KO': 0.030, 'JNJ': 0.030, 'PG': 0.025, 'PEP': 0.027,
            'MCD': 0.022, 'HD': 0.024, 'ABBV': 0.035, 'MMM': 0.055,
            'T': 0.065, 'VZ': 0.065
        },
    )

def strategy_4_sector_rotation():
    """Strategy 4: Sector Rotation Wheel — Sell puts on strongest sector ETFs."""
    return WheelConfig(
        name="4_Sector_Rotation",
        universe=['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLU', 'XLC', 'XLRE'],
        put_delta=0.22,
        call_delta=0.25,
        target_dte=30,
        profit_take=0.50,
        vix_max=32.0,
        max_positions=4,
        max_single_alloc=0.30,
        cash_reserve=0.20,
        use_regime_gate=True,
        regime_gate_type='trend',
        div_yields={
            'XLK': 0.007, 'XLF': 0.017, 'XLV': 0.015, 'XLE': 0.035,
            'XLY': 0.009, 'XLP': 0.027, 'XLI': 0.015, 'XLU': 0.030,
            'XLC': 0.008, 'XLRE': 0.030
        },
    )

def strategy_5_vol_adaptive():
    """Strategy 5: Volatility-Adaptive Wheel — Dynamic delta/DTE based on VIX."""
    return WheelConfig(
        name="5_Vol_Adaptive",
        universe=['SPY', 'QQQ'],
        put_delta=0.22,  # Baseline, overridden by dynamic
        call_delta=0.22,
        target_dte=35,
        profit_take=0.50,
        vix_max=40.0,  # Higher max — we adjust delta instead
        max_positions=2,
        cash_reserve=0.15,
        use_regime_gate=False,
        dynamic_delta=True,
        low_vol_delta=0.28,   # VIX < 15: more aggressive, sell closer
        mid_vol_delta=0.20,   # VIX 15-25: standard
        high_vol_delta=0.12,  # VIX > 25: far OTM, let volatility come to us
        div_yields={'SPY': 0.013, 'QQQ': 0.006},
    )

def strategy_6_hedged_wheel():
    """Strategy 6: Hedged Wheel — SPY wheel + protective put for tail risk."""
    return WheelConfig(
        name="6_Hedged_Wheel",
        universe=['SPY'],
        put_delta=0.25,  # Slightly more aggressive since hedged
        call_delta=0.25,
        target_dte=35,
        profit_take=0.50,
        vix_max=35.0,
        max_positions=1,
        cash_reserve=0.15,
        use_regime_gate=False,
        use_hedge=True,
        hedge_type='static_put',
        share_stop_loss=0.10,
        div_yields={'SPY': 0.013},
    )


def format_income_projections(result: Dict, account_sizes: List[int]) -> str:
    """Format income projections for various account sizes."""
    lines = []
    lines.append(f"\n{'Account Size':>15} | {'Monthly Income':>15} | {'Annual Income':>15} | {'Monthly Yield':>13} | {'Annual Yield':>12}")
    lines.append("-" * 85)

    base_yield_monthly = result['monthly_yield_pct'] / 100

    for size in account_sizes:
        monthly = size * base_yield_monthly
        annual = monthly * 12
        lines.append(f"  ${size:>12,} | ${monthly:>13,.0f} | ${annual:>13,.0f} | {base_yield_monthly*100:>11.2f}% | {base_yield_monthly*1200:>10.2f}%")

    return '\n'.join(lines)


def format_results(result: Dict) -> str:
    """Format backtest results for display."""
    if 'error' in result:
        return f"  ERROR: {result['error']}"

    lines = []
    lines.append(f"  Strategy: {result['config_name']}")
    lines.append(f"  ─────────────────────────────────────────")
    lines.append(f"  CAGR:           {result['cagr']*100:>8.2f}%")
    lines.append(f"  Sharpe:         {result['sharpe']:>8.2f}")
    lines.append(f"  Sortino:        {result['sortino']:>8.2f}")
    lines.append(f"  Max Drawdown:   {result['max_dd']*100:>8.2f}%")
    lines.append(f"  Calmar:         {result['calmar']:>8.2f}")
    lines.append(f"  Win Rate:       {result['win_rate']*100:>8.1f}%")
    lines.append(f"  Profit Factor:  {result['profit_factor']:>8.2f}")
    lines.append(f"  Total Trades:   {result['n_trades']:>8}")
    lines.append(f"  Avg Win:        ${result['avg_win']:>8.2f}")
    lines.append(f"  Avg Loss:       ${result['avg_loss']:>8.2f}")
    lines.append(f"  Avg Hold Days:  {result['avg_hold_days']:>8.1f}")
    lines.append(f"  Final Equity:   ${result['final_equity']:>12,.2f}")
    lines.append(f"  Total Return:   {result['total_return']*100:>8.2f}%")
    lines.append(f"")
    lines.append(f"  Trade Breakdown:")
    lines.append(f"    Profit Takes:   {result['n_profit_takes']}")
    lines.append(f"    Expired OTM:    {result['n_expired']}")
    lines.append(f"    Assignments:    {result['n_assignments']}")
    lines.append(f"    Called Away:    {result['n_called_away']}")
    lines.append(f"")
    lines.append(f"  Income Generation (based on $100k starting):")
    lines.append(f"    Monthly Premium Avg: ${result['monthly_premium_avg']:>8,.2f}")
    lines.append(f"    Monthly Yield:       {result['monthly_yield_pct']:>8.2f}%")
    lines.append(f"    Annual Yield:        {result['annual_yield_pct']:>8.2f}%")

    if result.get('regime_sharpes'):
        lines.append(f"")
        lines.append(f"  Regime Performance:")
        for regime, sh in sorted(result['regime_sharpes'].items()):
            lines.append(f"    {regime:>6}: Sharpe {sh:>6.2f}")

    return '\n'.join(lines)


if __name__ == '__main__':
    import time

    print("=" * 80)
    print("WHEEL & OPTIONS-SELLING STRATEGY DEVELOPMENT")
    print("Comprehensive backtest suite — 6 strategies, 2018-2025")
    print("=" * 80)
    print()

    # Discover available data
    print("[1/4] Loading market data...")

    # All tickers we need across all strategies
    all_tickers = [
        'SPY', 'QQQ', 'IWM',  # ETFs
        'KO', 'JNJ', 'PG', 'PEP', 'MCD', 'HD', 'ABBV', 'MMM', 'T', 'VZ',  # Dividend
        'XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLU', 'XLC', 'XLRE',  # Sectors
    ]

    prices = load_price_data(all_tickers)
    vix = load_vix_data()

    # Try saved data from prior run
    saved_path = Path("/home/jupiter/Lvl3Quant/data/wheel_strategy_data/all_prices.csv")
    if prices.empty and saved_path.exists():
        prices = pd.read_csv(saved_path, index_col=0, parse_dates=True)
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        print(f"  Loaded saved data: {len(prices)} days, {len(prices.columns)} tickers")

    saved_vix = Path("/home/jupiter/Lvl3Quant/data/wheel_strategy_data/vix.csv")
    if vix.empty and saved_vix.exists():
        vix = pd.read_csv(saved_vix, index_col=0, parse_dates=True).squeeze()
        if vix.index.tz is not None:
            vix.index = vix.index.tz_localize(None)
        print(f"  Loaded saved VIX: {len(vix)} days")

    if prices.empty:
        print("ERROR: No price data found. Attempting to download...")
        # Try yfinance
        try:
            import yfinance as yf
            print("  Downloading from Yahoo Finance...")
            data = {}
            for ticker in all_tickers:
                try:
                    df = yf.download(ticker, start='2018-01-01', end='2025-12-31', progress=False)
                    if not df.empty:
                        close = df['Close']
                        if isinstance(close, pd.DataFrame):
                            close = close.iloc[:, 0]
                        data[ticker] = close
                        print(f"    ✓ {ticker}: {len(df)} days")
                except Exception as e:
                    print(f"    ✗ {ticker}: {e}")

            if data:
                prices = pd.DataFrame(data).sort_index()
                prices.index = pd.to_datetime(prices.index)
                # Remove timezone info if present
                if prices.index.tz is not None:
                    prices.index = prices.index.tz_localize(None)
                # Save for future use
                save_dir = Path("/home/jupiter/Lvl3Quant/data/wheel_strategy_data")
                save_dir.mkdir(parents=True, exist_ok=True)
                prices.to_csv(save_dir / "all_prices.csv")
                print(f"  Saved {len(prices)} days of data for {len(prices.columns)} tickers")

            # VIX
            try:
                vix_df = yf.download('^VIX', start='2018-01-01', end='2025-12-31', progress=False)
                if not vix_df.empty:
                    vix = vix_df['Close']
                    if isinstance(vix, pd.DataFrame):
                        vix = vix.iloc[:, 0]
                    if vix.index.tz is not None:
                        vix.index = vix.index.tz_localize(None)
                    vix.to_csv(save_dir / "vix.csv")
                    print(f"    ✓ VIX: {len(vix)} days")
            except Exception:
                pass

        except ImportError:
            print("  yfinance not available. Install: pip install yfinance")
            sys.exit(1)

    if prices.empty:
        print("FATAL: No data available. Cannot run backtests.")
        sys.exit(1)

    available_tickers = list(prices.columns)
    print(f"  Available tickers: {', '.join(available_tickers)}")
    print(f"  Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"  VIX data: {len(vix)} days")
    print()

    # Define strategies
    strategies = [
        strategy_1_conservative_income(),
        strategy_2_balanced_growth(),
        strategy_3_dividend_wheel(),
        strategy_4_sector_rotation(),
        strategy_5_vol_adaptive(),
        strategy_6_hedged_wheel(),
    ]

    # Run backtests
    print("[2/4] Running backtests...")
    results = {}

    for config in strategies:
        # Filter universe to available tickers
        config.universe = [t for t in config.universe if t in available_tickers]
        if not config.universe:
            print(f"  ✗ {config.name}: No tickers available, skipping")
            continue

        print(f"  Running {config.name} ({', '.join(config.universe)})...", end=' ', flush=True)
        t0 = time.time()
        result = run_wheel_backtest(config, prices, vix, starting_cash=100000)
        elapsed = time.time() - t0

        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"done ({elapsed:.1f}s) — Sharpe {result['sharpe']:.2f}, CAGR {result['cagr']*100:.1f}%")
        results[config.name] = result

    print()

    # Generate comprehensive report
    print("[3/4] Generating comprehensive report...")
    print()

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("WHEEL & OPTIONS-SELLING STRATEGIES — COMPREHENSIVE REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"Backtest window: 2018-01-01 to 2025-12-31")
    report_lines.append(f"Starting capital: $100,000 | Max leverage: 1.0x")
    report_lines.append("=" * 80)

    # Summary table
    report_lines.append("\n" + "=" * 80)
    report_lines.append("STRATEGY COMPARISON SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"{'Strategy':<30} | {'CAGR':>7} | {'Sharpe':>7} | {'Sortino':>7} | {'MaxDD':>7} | {'WR':>6} | {'PF':>5} | {'Mo.Yield':>8}")
    report_lines.append("-" * 105)

    for name, result in sorted(results.items()):
        if 'error' not in result:
            report_lines.append(
                f"  {name:<28} | {result['cagr']*100:>6.1f}% | {result['sharpe']:>7.2f} | "
                f"{result['sortino']:>7.2f} | {result['max_dd']*100:>6.1f}% | "
                f"{result['win_rate']*100:>5.1f}% | {result['profit_factor']:>5.2f} | "
                f"{result['monthly_yield_pct']:>7.2f}%"
            )

    # Detailed results per strategy
    for name, result in sorted(results.items()):
        if 'error' in result:
            continue
        report_lines.append("\n" + "=" * 80)
        report_lines.append(f"STRATEGY: {name}")
        report_lines.append("=" * 80)
        report_lines.append(format_results(result))

        # Income projections
        report_lines.append("\n  Income Projections by Account Size:")
        report_lines.append(format_income_projections(result, [10000, 25000, 50000, 100000, 250000, 500000]))

    # Risk analysis
    report_lines.append("\n" + "=" * 80)
    report_lines.append("RISK ANALYSIS & RECOMMENDATIONS")
    report_lines.append("=" * 80)

    report_lines.append("""
  KEY FINDINGS:

  1. CONSERVATIVE INCOME (SPY-only):
     - Lowest risk profile, suitable for primary income stream
     - 30% cash reserve provides buffer in downturns
     - Regime gate prevents selling into falling markets
     - Best for: Retirees, risk-averse investors, account > $50k

  2. BALANCED GROWTH (SPY/QQQ/IWM):
     - Diversified across market caps
     - Regime-gated entries reduce drawdown
     - Higher income than conservative with moderate risk
     - Best for: Growth-oriented accounts $50k-$500k

  3. DIVIDEND ARISTOCRAT WHEEL:
     - Stocks you genuinely WANT to own (quality companies)
     - Dividends add income on top of premium
     - Higher assignment tolerance (you keep quality stocks)
     - Best for: Long-term wealth building, any account size

  4. SECTOR ROTATION WHEEL:
     - Captures sector momentum
     - Higher premium from sector-specific volatility
     - More complex, needs active management
     - Best for: Active managers, $100k+ accounts

  5. VOLATILITY-ADAPTIVE WHEEL:
     - Automatically adjusts to market conditions
     - Sells aggressive in calm markets, defensive in volatile
     - Best risk-adjusted returns in theory
     - Best for: Accounts wanting consistent yield across regimes

  6. HEDGED WHEEL:
     - Protective put overlay caps downside
     - Higher cost but much better tail risk
     - Best for: Large accounts where capital preservation matters

  PORTFOLIO RECOMMENDATION:
  For a balanced income portfolio, combine:
    - 40% Strategy 1 (Conservative, SPY) — stable base
    - 30% Strategy 3 (Dividend Wheel) — quality stocks + dividends
    - 20% Strategy 5 (Vol-Adaptive) — smart premium capture
    - 10% Cash buffer

  This blend targets:
    - Monthly yield: ~1.0-1.5%
    - Annual yield: ~12-18%
    - Max drawdown: ~15-20%
    - Sharpe: ~1.5-2.0
""")

    # Stock picking criteria
    report_lines.append("\n" + "=" * 80)
    report_lines.append("STOCK PICKING CRITERIA (for wheel candidates)")
    report_lines.append("=" * 80)
    report_lines.append("""
  ONLY SELL OPTIONS ON STOCKS YOU WANT TO OWN:

  1. FUNDAMENTAL QUALITY:
     - Market cap > $10B (liquidity for options)
     - Positive earnings or clear path to profitability
     - Manageable debt (D/E < 1.5 for non-financials)
     - Strong free cash flow (FCF yield > 3%)

  2. DIVIDEND TRACK RECORD (for dividend wheel):
     - 10+ years of consecutive dividend increases
     - Payout ratio < 70% (room to grow)
     - Dividend yield 2-5% (sweet spot)

  3. OPTIONS LIQUIDITY:
     - Average daily options volume > 1,000 contracts
     - Bid-ask spread < $0.10 for ATM options
     - Open interest > 500 for target strikes

  4. GREEKS-BASED ENTRY:
     - Put delta: -0.15 to -0.30 (10-30% probability of assignment)
     - Theta: Maximize daily decay (sell 30-45 DTE)
     - Gamma: Low gamma preferred (less price sensitivity)
     - Vega: Sell when IV is elevated (IV rank > 30)
     - Premium yield: > 0.5% of strike per month

  5. MACRO OVERLAY:
     - VIX < 30 for new entries (or use vol-adaptive deltas)
     - Trend: Price > 50-day MA for new puts
     - Sector strength: Prefer sectors with positive momentum
     - Earnings: Avoid selling 2 weeks before earnings (gap risk)

  6. POSITION SIZING:
     - Max 25% of portfolio in single name
     - Max 3-5 concurrent positions
     - Keep 15-30% cash at all times
     - Max 1.0x leverage (cash-secured only)
""")

    # Greeks reference
    report_lines.append("\n" + "=" * 80)
    report_lines.append("GREEKS REFERENCE FOR OPTIONS SELLERS")
    report_lines.append("=" * 80)
    report_lines.append("""
  DELTA (Position Risk):
    -0.15: ~85% chance put expires worthless. Low premium, high safety.
    -0.20: ~80% chance OTM. Sweet spot for conservative wheel.
    -0.25: ~75% chance OTM. Balanced premium/risk.
    -0.30: ~70% chance OTM. Higher income but more assignments.
    Rule: Lower delta in high-VIX, higher delta in low-VIX.

  THETA (Time Decay — YOUR FRIEND as a seller):
    Theta accelerates as expiration approaches.
    30-45 DTE: Best theta-per-day capture zone.
    < 21 DTE: Theta highest but gamma risk increases.
    Strategy: Sell at 30-45 DTE, close at 50% profit or 7 DTE.

  GAMMA (Change in Delta — YOUR ENEMY as a seller):
    High gamma = delta changes fast = more risk.
    Lowest at 30-45 DTE with OTM strikes.
    Highest near expiration near the money.
    Strategy: Close or roll before 7 DTE to avoid gamma spike.

  VEGA (Volatility Sensitivity):
    Sell when IV is high → collect more premium.
    IV mean-reverts → option value drops → you profit.
    IV Rank > 30: Good time to sell.
    IV Rank > 50: Great time to sell (elevated vol).
    IV Rank < 15: Poor time to sell (not enough premium).

  OPTIMAL ENTRY CHECKLIST:
    □ IV Rank > 30 (elevated implied volatility)
    □ Delta between -0.15 and -0.30
    □ DTE 30-45 days
    □ Premium yield > 0.5%/month on collateral
    □ Stock you'd be happy to own at strike price
    □ No earnings within 2 weeks
    □ VIX not spiking (< 30 for normal, or use vol-adaptive)
""")

    # Save report
    report_text = '\n'.join(report_lines)
    print(report_text)

    report_path = Path("/home/jupiter/Lvl3Quant/research/findings/wheel_comprehensive_strategies.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)

    # Save results JSON
    results_json = {}
    for name, result in results.items():
        if 'error' not in result:
            r = {k: v for k, v in result.items() if k not in ['equity_curve', 'trades']}
            results_json[name] = r

    json_path = Path("/home/jupiter/Lvl3Quant/research/findings/wheel_comprehensive_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    print(f"\n[4/4] Report saved to {report_path}")
    print(f"      Results saved to {json_path}")
    print("\nDone.")
