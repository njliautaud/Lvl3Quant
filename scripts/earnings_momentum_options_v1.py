#!/usr/bin/env python3
"""
Earnings Momentum Options v1 — Post-Earnings Drift with Options Leverage
=========================================================================
Proven: buying stocks after earnings beats + holding 40-60d → Sharpe 1.54, perm p=0.001.
Problem: shares-only returns too slow for $645 account. Solution: OPTIONS for leverage.

STRATEGY:
  After a stock gaps up >3% on earnings day (proxy for earnings beat),
  buy ATM or slightly OTM call options with 45-60 DTE (to match 40-day drift).
  Sell before expiry.

6 VARIANTS:
  A) ATM Calls 45-DTE: Buy 1 ATM call after >3% gap, hold 30 trading days, sell
  B) OTM Calls (10% OTM) 45-DTE: Cheaper, more leverage, lower delta
  C) Cheap Stocks Only: Only trade stocks where 1 ATM call < $200
  D) Debit Spreads: Buy ATM call + sell 10% OTM call (reduce cost, cap gains)
  E) Momentum Filter: Only enter if stock has positive 20d momentum before earnings
  F) Regime + Cheap: Only enter when VIX < 25 AND stock < $50

OPTIONS COST MODEL (realistic):
  - Premium via Black-Scholes with realistic IV
  - Commission: $0.65/contract/leg ($1.30 RT single leg, $2.60 RT spread)
  - Bid-ask spread: 10% haircut on entry premium, 10% on exit
  - Each contract = 100 shares, cost = premium × 100
  - Max $200 per trade, starting capital $645

OOT: Jan 2022 – Jul 2026.

VALIDATION: Sharpe >0.5, perm p<0.05, regime gap <0.5, MDD >-50%, ≥20 trades.
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings('ignore')

# --- Path setup ---
for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONSTANTS
# ============================================================

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'NFLX', 'CRM', 'PLTR', 'SOFI', 'HOOD', 'SNAP', 'PINS', 'COIN',
    'RBLX', 'UBER', 'LYFT', 'DDOG', 'TTD', 'SHOP', 'NET', 'ROKU',
]

CHEAP_STOCKS = ['SOFI', 'SNAP', 'HOOD', 'PLTR', 'PINS', 'RBLX', 'LYFT', 'ROKU']

STARTING_CAPITAL = 645.0
COMMISSION_PER_LEG = 0.65
COMMISSION_RT_SINGLE = 1.30
COMMISSION_RT_SPREAD = 2.60  # 2 legs each way
MAX_POSITION_DOLLARS = 200.0
RISK_FREE_RATE = 0.05
START_DATE = '2022-01-01'
END_DATE = '2026-07-30'
N_PERMUTATIONS = 1000
GAP_THRESHOLD = 0.03  # 3% gap up = earnings beat proxy
HOLD_DAYS = 30  # trading days (~42 calendar days)
DTE_TARGET = 45  # 45 DTE options
OTM_PCT = 0.10  # 10% OTM for variant B
BID_ASK_HAIRCUT = 0.10  # 10% slippage each way
ATM_PREMIUM_PCT = 0.05  # ~5% of stock price for ATM 45-DTE
OTM_PREMIUM_PCT = 0.03  # ~3% of stock price for 10% OTM 45-DTE

# ============================================================
# BLACK-SCHOLES PRICING
# ============================================================

def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_call_price(S, K, T, r, sigma):
    if T <= 1e-8:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    """Call delta."""
    if T <= 1e-8:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1)

def bs_gamma(S, K, T, r, sigma):
    """Call gamma."""
    if T <= 1e-8 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, sigma):
    """Call vega (per 1 unit sigma change)."""
    if T <= 1e-8 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load daily OHLCV + earnings dates via yfinance with caching."""
    import yfinance as yf

    cache_path = os.path.join(LVL3_ROOT, 'data', 'earnings_momentum_cache.parquet')
    earnings_cache_path = os.path.join(LVL3_ROOT, 'data', 'earnings_dates_cache.json')

    # --- Price data ---
    prices_df = None
    if os.path.exists(cache_path):
        try:
            prices_df = pd.read_parquet(cache_path)
            if len(prices_df) > 0:
                latest = prices_df.index.get_level_values('date').max()
                if pd.Timestamp(latest) >= pd.Timestamp('2026-07-20'):
                    print(f"Loaded cached price data: {len(prices_df)} rows, latest={latest}")
                else:
                    prices_df = None
        except Exception:
            prices_df = None

    if prices_df is None:
        print(f"Downloading price data for {len(STOCK_UNIVERSE)} stocks + SPY + ^VIX...")
        tickers = STOCK_UNIVERSE + ['SPY', '^VIX']
        all_frames = []

        for ticker in tickers:
            try:
                data = yf.download(ticker, start=START_DATE, end=END_DATE,
                                   progress=False, auto_adjust=True)
                if len(data) < 50:
                    print(f"  WARNING: {ticker} has only {len(data)} rows, skipping")
                    continue
                data.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                                for c in data.columns]
                data['ticker'] = ticker
                data.index.name = 'date'
                all_frames.append(data)
                print(f"  {ticker}: {len(data)} rows")
            except Exception as e:
                print(f"  ERROR downloading {ticker}: {e}")

        if not all_frames:
            raise RuntimeError("No price data downloaded")

        prices_df = pd.concat(all_frames)
        prices_df = prices_df.reset_index().set_index(['ticker', 'date']).sort_index()

        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            prices_df.to_parquet(cache_path)
            print(f"Cached price data")
        except Exception as e:
            print(f"Warning: Could not cache price data: {e}")

    # --- Earnings dates ---
    earnings_dates = {}
    if os.path.exists(earnings_cache_path):
        try:
            with open(earnings_cache_path, 'r') as f:
                earnings_dates = json.load(f)
            if len(earnings_dates) >= len(STOCK_UNIVERSE) * 0.5:
                total_dates = sum(len(v) for v in earnings_dates.values())
                print(f"Loaded cached earnings dates: {len(earnings_dates)} tickers, "
                      f"{total_dates} total dates")
            else:
                earnings_dates = {}
        except Exception:
            earnings_dates = {}

    if not earnings_dates:
        print("Fetching earnings dates...")
        for ticker in STOCK_UNIVERSE:
            try:
                t = yf.Ticker(ticker)
                ed = t.earnings_dates
                if ed is not None and len(ed) > 0:
                    dates_list = sorted([
                        str(d.date()) if hasattr(d, 'date') else str(d)[:10]
                        for d in ed.index
                    ])
                    dates_list = [d for d in dates_list if START_DATE <= d <= END_DATE]
                    if dates_list:
                        earnings_dates[ticker] = dates_list
                        print(f"  {ticker}: {len(dates_list)} earnings dates")
            except Exception as e:
                print(f"  ERROR fetching earnings for {ticker}: {e}")

        if earnings_dates:
            try:
                with open(earnings_cache_path, 'w') as f:
                    json.dump(earnings_dates, f, indent=2)
            except Exception:
                pass

    return prices_df, earnings_dates


def get_ticker_prices(prices_df, ticker):
    try:
        return prices_df.loc[ticker].copy()
    except KeyError:
        return None


def get_vix_for_date(vix_data, date):
    mask = vix_data.index <= date
    if mask.any():
        return vix_data.loc[mask, 'close'].iloc[-1]
    return 20.0


def compute_realized_vol(close_prices, lookback=21):
    if len(close_prices) < lookback + 1:
        return 0.35
    log_rets = np.diff(np.log(close_prices[-(lookback + 1):]))
    return max(np.std(log_rets) * np.sqrt(252), 0.15)

# ============================================================
# OPTION PRICING HELPERS
# ============================================================

def price_option_entry(stock_price, strike, dte_years, vol, is_spread=False):
    """
    Price an option at entry with realistic costs.
    Returns (premium_per_share, total_cost_1_contract, delta).

    Total cost includes: BS price + 10% bid-ask haircut + commission.
    """
    bs_price = bs_call_price(stock_price, strike, dte_years, RISK_FREE_RATE, vol)
    # Bid-ask haircut: you pay ask, which is ~10% above mid
    entry_price = bs_price * (1 + BID_ASK_HAIRCUT)
    contract_cost = entry_price * 100  # 1 contract = 100 shares
    total_cost = contract_cost + COMMISSION_PER_LEG  # entry commission
    delta = bs_delta(stock_price, strike, dte_years, RISK_FREE_RATE, vol)
    return entry_price, total_cost, delta


def price_option_exit(stock_price, strike, dte_years, vol):
    """
    Price an option at exit with realistic costs.
    Returns net proceeds per share after bid-ask haircut and commission.
    """
    bs_price = bs_call_price(stock_price, strike, dte_years, RISK_FREE_RATE, vol)
    # Bid-ask haircut: you sell at bid, which is ~10% below mid
    exit_price = bs_price * (1 - BID_ASK_HAIRCUT)
    exit_price = max(exit_price, 0)
    contract_proceeds = exit_price * 100
    net_proceeds = contract_proceeds - COMMISSION_PER_LEG  # exit commission
    return max(net_proceeds, 0)


def price_spread_entry(stock_price, strike_long, strike_short, dte_years, vol):
    """
    Price a bull call debit spread at entry.
    Buy ATM call, sell OTM call.
    Returns (net_debit_per_share, total_cost, long_delta, short_delta).
    """
    long_bs = bs_call_price(stock_price, strike_long, dte_years, RISK_FREE_RATE, vol)
    short_bs = bs_call_price(stock_price, strike_short, dte_years, RISK_FREE_RATE, vol)

    # Pay ask on long, receive bid on short
    long_entry = long_bs * (1 + BID_ASK_HAIRCUT)
    short_entry = short_bs * (1 - BID_ASK_HAIRCUT)
    net_debit = long_entry - short_entry
    net_debit = max(net_debit, 0.01)  # floor

    contract_cost = net_debit * 100
    total_cost = contract_cost + COMMISSION_RT_SPREAD / 2  # entry legs only

    long_delta = bs_delta(stock_price, strike_long, dte_years, RISK_FREE_RATE, vol)
    short_delta = bs_delta(stock_price, strike_short, dte_years, RISK_FREE_RATE, vol)

    return net_debit, total_cost, long_delta, short_delta


def price_spread_exit(stock_price, strike_long, strike_short, dte_years, vol):
    """
    Price spread exit. Net proceeds = (long_bid - short_ask).
    """
    long_bs = bs_call_price(stock_price, strike_long, dte_years, RISK_FREE_RATE, vol)
    short_bs = bs_call_price(stock_price, strike_short, dte_years, RISK_FREE_RATE, vol)

    long_exit = long_bs * (1 - BID_ASK_HAIRCUT)
    short_exit = short_bs * (1 + BID_ASK_HAIRCUT)
    net_credit = long_exit - short_exit
    net_credit = max(net_credit, 0)

    contract_proceeds = net_credit * 100
    net_proceeds = contract_proceeds - COMMISSION_RT_SPREAD / 2  # exit legs
    return max(net_proceeds, 0)


# ============================================================
# BACKTEST ENGINE
# ============================================================

def find_earnings_gaps(prices_df, earnings_dates, ticker, gap_threshold=GAP_THRESHOLD):
    """
    Find instances where stock gapped up > threshold on/after earnings.
    Returns list of (earnings_date, gap_pct, entry_price, pre_earnings_close).
    """
    ticker_prices = get_ticker_prices(prices_df, ticker)
    if ticker_prices is None or len(ticker_prices) < 60:
        return []

    if ticker not in earnings_dates or not earnings_dates[ticker]:
        return []

    gaps = []
    trading_dates = ticker_prices.index.tolist()

    for ed_str in earnings_dates[ticker]:
        ed = pd.Timestamp(ed_str)

        # Find the first trading day on or after earnings date
        post_dates = [d for d in trading_dates if d >= ed]
        if len(post_dates) < 2:
            continue

        # Find the trading day before earnings
        pre_dates = [d for d in trading_dates if d < ed]
        if len(pre_dates) < 21:  # need enough history for vol
            continue

        earnings_day = post_dates[0]
        pre_day = pre_dates[-1]

        pre_close = ticker_prices.loc[pre_day, 'close']
        post_open = ticker_prices.loc[earnings_day, 'open']

        if pre_close <= 0:
            continue

        gap_pct = (post_open - pre_close) / pre_close

        if gap_pct >= gap_threshold:
            # Get 21-day realized vol from pre-earnings closes
            recent_closes = ticker_prices.loc[pre_dates[-22:], 'close'].values
            realized_vol = compute_realized_vol(recent_closes)

            # Get 20-day momentum (for variant E)
            if len(pre_dates) >= 21:
                mom_start = ticker_prices.loc[pre_dates[-21], 'close']
                mom_end = pre_close
                momentum_20d = (mom_end - mom_start) / mom_start
            else:
                momentum_20d = 0

            gaps.append({
                'ticker': ticker,
                'earnings_date': ed_str,
                'entry_date': earnings_day,
                'gap_pct': gap_pct,
                'entry_price': post_open,  # enter at open on gap day
                'pre_close': pre_close,
                'realized_vol': realized_vol,
                'momentum_20d': momentum_20d,
            })

    return gaps


def run_variant(prices_df, vix_data, all_gaps, variant='A'):
    """
    Run a single variant backtest.

    Variants:
    A) ATM Calls 45-DTE, hold 30 trading days
    B) OTM Calls (10% OTM), hold 30 trading days
    C) Cheap Stocks Only (ATM calls, stock < ~$50 so option < $200)
    D) Debit Spreads (ATM - 10% OTM)
    E) Momentum Filter (positive 20d momentum required)
    F) Regime + Cheap (VIX < 25 AND stock < $50)
    """
    equity = STARTING_CAPITAL
    equity_curve = [equity]
    trades = []
    peak_equity = equity
    max_dd = 0

    # Sort gaps by date
    sorted_gaps = sorted(all_gaps, key=lambda x: x['entry_date'])

    for gap in sorted_gaps:
        ticker = gap['ticker']
        entry_date = gap['entry_date']
        stock_price = gap['entry_price']
        vol = gap['realized_vol']
        momentum = gap['momentum_20d']

        # Get VIX on entry date
        vix = get_vix_for_date(vix_data, entry_date)

        # --- Variant filters ---
        if variant == 'C':
            if ticker not in CHEAP_STOCKS:
                continue
        elif variant == 'E':
            if momentum <= 0:
                continue
        elif variant == 'F':
            if vix >= 25 or stock_price >= 50:
                continue

        # --- IV estimate: use higher of realized vol and VIX-implied ---
        iv = max(vol, vix / 100.0)
        # Bump IV slightly for post-earnings (event vol premium lingers ~1 week)
        iv = iv * 1.1

        dte_years = DTE_TARGET / 365.0

        # --- Position sizing and pricing ---
        is_spread = (variant == 'D')

        if is_spread:
            strike_long = round(stock_price, 0)  # ATM
            strike_short = round(stock_price * (1 + OTM_PCT), 0)  # 10% OTM
            net_debit, entry_cost, long_delta, short_delta = price_spread_entry(
                stock_price, strike_long, strike_short, dte_years, iv)

            if entry_cost > MAX_POSITION_DOLLARS or entry_cost > equity * 0.50:
                continue
            if entry_cost <= 0:
                continue

            # Max gain = (strike_short - strike_long) * 100 - entry_cost
            max_gain = (strike_short - strike_long) * 100 - entry_cost

            n_contracts = 1
            total_entry_cost = entry_cost
        else:
            # Single call
            if variant == 'B':
                strike = round(stock_price * (1 + OTM_PCT), 0)  # 10% OTM
            else:
                strike = round(stock_price, 0)  # ATM

            entry_premium, entry_cost, delta = price_option_entry(
                stock_price, strike, dte_years, iv)

            if entry_cost > MAX_POSITION_DOLLARS or entry_cost > equity * 0.50:
                # Try with fewer contracts or skip
                continue
            if entry_cost <= 0:
                continue

            n_contracts = 1
            total_entry_cost = entry_cost

        # Not enough capital
        if total_entry_cost > equity:
            continue

        # --- Find exit date (30 trading days later) ---
        ticker_prices = get_ticker_prices(prices_df, ticker)
        if ticker_prices is None:
            continue

        trading_dates = ticker_prices.index.tolist()
        entry_idx = None
        for i, d in enumerate(trading_dates):
            if d >= entry_date:
                entry_idx = i
                break
        if entry_idx is None:
            continue

        exit_idx = min(entry_idx + HOLD_DAYS, len(trading_dates) - 1)
        if exit_idx <= entry_idx:
            continue

        exit_date = trading_dates[exit_idx]
        exit_stock_price = ticker_prices.loc[exit_date, 'close']

        # Time remaining at exit
        calendar_days_held = (exit_date - entry_date).days
        dte_remaining = max(DTE_TARGET - calendar_days_held, 1)
        dte_remaining_years = dte_remaining / 365.0

        # IV at exit: use same base vol (conservative; drift period isn't event-driven)
        exit_iv = max(vol, vix / 100.0)  # no event premium at exit

        # --- Exit pricing ---
        if is_spread:
            exit_proceeds = price_spread_exit(
                exit_stock_price, strike_long, strike_short, dte_remaining_years, exit_iv)
            pnl = exit_proceeds - total_entry_cost
        else:
            exit_proceeds = price_option_exit(
                exit_stock_price, strike, dte_remaining_years, exit_iv)
            pnl = exit_proceeds - total_entry_cost

        pnl_pct = pnl / total_entry_cost * 100 if total_entry_cost > 0 else 0
        stock_return = (exit_stock_price - stock_price) / stock_price

        equity += pnl
        equity_curve.append(equity)

        # Track drawdown
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        max_dd = min(max_dd, dd)

        trades.append({
            'ticker': ticker,
            'earnings_date': gap['earnings_date'],
            'entry_date': str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date)[:10],
            'exit_date': str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date)[:10],
            'entry_stock_price': round(stock_price, 2),
            'exit_stock_price': round(exit_stock_price, 2),
            'stock_return_pct': round(stock_return * 100, 2),
            'gap_pct': round(gap['gap_pct'] * 100, 2),
            'entry_cost': round(total_entry_cost, 2),
            'exit_proceeds': round(exit_proceeds, 2),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 1),
            'is_spread': is_spread,
            'iv_entry': round(iv, 3),
            'vix_entry': round(vix, 1),
            'momentum_20d': round(momentum * 100, 2),
        })

    return equity, equity_curve, trades, max_dd


def compute_metrics(trades, equity_curve, final_equity):
    """Compute strategy metrics from trade list."""
    if not trades:
        return {
            'final_equity': STARTING_CAPITAL, 'total_return_pct': 0,
            'total_trades': 0, 'win_rate': 0, 'avg_pnl': 0,
            'profit_factor': 0, 'sharpe': 0, 'sortino': 0,
            'max_drawdown_pct': 0, 'avg_hold_days': 0, 'avg_stock_return': 0,
        }

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    avg_pnl = np.mean(pnls)

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    # Sharpe and Sortino from PnL returns
    pnl_arr = np.array(pnls)
    if len(pnl_arr) > 1 and np.std(pnl_arr) > 0:
        # Annualized: assume ~8 trades/year average
        trades_per_year = max(len(trades) / 4.5, 1)  # 4.5 years of data
        sharpe = (np.mean(pnl_arr) / np.std(pnl_arr)) * np.sqrt(trades_per_year)
        downside = pnl_arr[pnl_arr < 0]
        if len(downside) > 0:
            downside_std = np.std(downside)
            sortino = (np.mean(pnl_arr) / downside_std * np.sqrt(trades_per_year)
                       if downside_std > 0 else sharpe * 2)
        else:
            sortino = sharpe * 3
    else:
        sharpe = 0
        sortino = 0

    # Max drawdown from equity curve
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd = dd.min() * 100

    # Average holding period
    hold_days_list = []
    for t in trades:
        try:
            d1 = pd.Timestamp(t['entry_date'])
            d2 = pd.Timestamp(t['exit_date'])
            hold_days_list.append((d2 - d1).days)
        except Exception:
            pass
    avg_hold = np.mean(hold_days_list) if hold_days_list else 0

    avg_stock_return = np.mean([t['stock_return_pct'] for t in trades])

    return {
        'final_equity': round(final_equity, 2),
        'total_return_pct': round((final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 1),
        'total_trades': len(trades),
        'win_rate': round(win_rate, 1),
        'avg_pnl': round(avg_pnl, 2),
        'avg_pnl_pct': round(np.mean([t['pnl_pct'] for t in trades]), 1),
        'profit_factor': round(profit_factor, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_drawdown_pct': round(max_dd, 1),
        'avg_hold_days': round(avg_hold, 1),
        'avg_stock_return_pct': round(avg_stock_return, 1),
    }


def permutation_test(trades, n_permutations=N_PERMUTATIONS):
    """
    Permutation test: randomly flip trade signs to test if total return is due to luck.
    Returns p-value.
    """
    if len(trades) < 5:
        return 1.0

    pnls = np.array([t['pnl'] for t in trades])
    actual_total = pnls.sum()

    rng = np.random.RandomState(42)
    count_better = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(pnls))
        if (pnls * signs).sum() >= actual_total:
            count_better += 1

    return (count_better + 1) / (n_permutations + 1)


def regime_analysis(trades, vix_data):
    """
    Stratify trades by regime: bull (VIX<20), neutral (20-30), bear (VIX>30).
    Returns regime Sharpes and regime gap metric.
    """
    if len(trades) < 5:
        return {'regime_gap': 1.0, 'regimes': {}}

    regime_trades = {'bull': [], 'neutral': [], 'bear': []}

    for t in trades:
        vix = t.get('vix_entry', 20)
        if vix < 20:
            regime_trades['bull'].append(t['pnl'])
        elif vix < 30:
            regime_trades['neutral'].append(t['pnl'])
        else:
            regime_trades['bear'].append(t['pnl'])

    regime_sharpes = {}
    for regime, pnls in regime_trades.items():
        if len(pnls) >= 3:
            arr = np.array(pnls)
            if np.std(arr) > 0:
                regime_sharpes[regime] = round(np.mean(arr) / np.std(arr) * np.sqrt(len(arr)), 2)
            else:
                regime_sharpes[regime] = 0
        else:
            regime_sharpes[regime] = None

    # Compute regime gap
    valid_sharpes = [v for v in regime_sharpes.values() if v is not None]
    if len(valid_sharpes) >= 2:
        max_s = max(abs(s) for s in valid_sharpes) or 1
        regime_gap = (max(valid_sharpes) - min(valid_sharpes)) / max_s
    else:
        regime_gap = 0

    return {
        'regime_gap': round(regime_gap, 3),
        'regime_sharpes': regime_sharpes,
        'regime_counts': {k: len(v) for k, v in regime_trades.items()},
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("EARNINGS MOMENTUM OPTIONS v1 — Post-Earnings Drift with Leverage")
    print("=" * 70)
    print(f"Universe: {len(STOCK_UNIVERSE)} stocks")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Starting capital: ${STARTING_CAPITAL}")
    print(f"Gap threshold: {GAP_THRESHOLD*100:.0f}%")
    print(f"Hold period: {HOLD_DAYS} trading days")
    print(f"DTE: {DTE_TARGET} days")
    print()

    # Load data
    prices_df, earnings_dates = load_data()

    # Get VIX data
    try:
        vix_data = get_ticker_prices(prices_df, '^VIX')
        if vix_data is None:
            # Try alternate
            vix_data = get_ticker_prices(prices_df, 'VIX')
    except Exception:
        vix_data = None

    if vix_data is None:
        print("WARNING: No VIX data available, using default VIX=20")
        # Create a dummy VIX series
        sample_ticker = STOCK_UNIVERSE[0]
        sample_prices = get_ticker_prices(prices_df, sample_ticker)
        if sample_prices is not None:
            vix_data = sample_prices.copy()
            vix_data['close'] = 20.0
        else:
            raise RuntimeError("Cannot create VIX fallback")

    # Find all earnings gaps
    print("\nScanning for >3% earnings gaps...")
    all_gaps = []
    for ticker in STOCK_UNIVERSE:
        gaps = find_earnings_gaps(prices_df, earnings_dates, ticker)
        all_gaps.extend(gaps)
        if gaps:
            print(f"  {ticker}: {len(gaps)} qualifying gaps "
                  f"(avg gap: {np.mean([g['gap_pct'] for g in gaps])*100:.1f}%)")

    print(f"\nTotal qualifying earnings gaps: {len(all_gaps)}")
    if not all_gaps:
        print("ERROR: No earnings gaps found. Check data.")
        return

    # Run all variants
    variant_configs = {
        'A': 'ATM Calls 45-DTE (all stocks, >3% gap)',
        'B': 'OTM Calls 10% OTM 45-DTE (more leverage)',
        'C': 'Cheap Stocks Only (ATM calls, affordable options)',
        'D': 'Debit Spreads (ATM - 10% OTM, reduced cost)',
        'E': 'Momentum Filter (+20d momentum required)',
        'F': 'Regime + Cheap (VIX<25 AND stock<$50)',
    }

    results = {
        'metadata': {
            'strategy': 'earnings_momentum_options_drift_v1',
            'description': 'Post-earnings drift captured via options leverage',
            'universe_size': len(STOCK_UNIVERSE),
            'period': f'{START_DATE} to {END_DATE}',
            'starting_capital': STARTING_CAPITAL,
            'gap_threshold_pct': GAP_THRESHOLD * 100,
            'hold_days': HOLD_DAYS,
            'dte_target': DTE_TARGET,
            'cost_model': 'BS + 10% bid-ask haircut each way + $0.65/leg commission',
            'max_per_trade': MAX_POSITION_DOLLARS,
            'total_earnings_gaps': len(all_gaps),
            'timestamp': datetime.now().isoformat(),
        },
        'variant_metrics': [],
        'best_variant': None,
        'validation': {},
    }

    print("\n" + "=" * 70)
    best_sharpe = -999
    best_variant = None

    for variant_key, variant_name in variant_configs.items():
        print(f"\n--- Variant {variant_key}: {variant_name} ---")

        final_equity, equity_curve, trades, max_dd = run_variant(
            prices_df, vix_data, all_gaps, variant=variant_key)

        metrics = compute_metrics(trades, equity_curve, final_equity)

        # Permutation test
        perm_p = permutation_test(trades)
        metrics['perm_p_value'] = round(perm_p, 4)

        # Regime analysis
        regime = regime_analysis(trades, vix_data)
        metrics['regime_gap'] = regime['regime_gap']
        metrics['regime_sharpes'] = regime.get('regime_sharpes', {})
        metrics['regime_counts'] = regime.get('regime_counts', {})

        # Validation checks
        passes_sharpe = metrics['sharpe'] > 0.5
        passes_perm = perm_p < 0.05
        passes_regime = regime['regime_gap'] < 0.5
        passes_mdd = metrics['max_drawdown_pct'] > -50
        passes_trades = metrics['total_trades'] >= 20

        metrics['validation'] = {
            'sharpe_gt_0.5': passes_sharpe,
            'perm_p_lt_0.05': passes_perm,
            'regime_gap_lt_0.5': passes_regime,
            'mdd_gt_neg50': passes_mdd,
            'trades_gte_20': passes_trades,
            'all_pass': all([passes_sharpe, passes_perm, passes_regime,
                            passes_mdd, passes_trades]),
        }

        # Add variant info
        metrics['variant'] = variant_key
        metrics['name'] = variant_name

        # Print summary
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Final equity: ${metrics['final_equity']:,.2f} "
              f"(return: {metrics['total_return_pct']:.1f}%)")
        print(f"  Win rate: {metrics['win_rate']:.1f}%")
        print(f"  Profit factor: {metrics['profit_factor']:.2f}")
        print(f"  Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f}")
        print(f"  Max DD: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Avg stock return: {metrics['avg_stock_return_pct']:.1f}% | "
              f"Avg option PnL: {metrics['avg_pnl_pct']:.1f}%")
        print(f"  Perm p-value: {perm_p:.4f}")
        print(f"  Regime gap: {regime['regime_gap']:.3f} "
              f"(sharpes: {regime.get('regime_sharpes', {})})")
        val = metrics['validation']
        status = "PASS" if val['all_pass'] else "FAIL"
        fails = [k for k, v in val.items() if k != 'all_pass' and not v]
        print(f"  Validation: {status}" + (f" (failed: {', '.join(fails)})" if fails else ""))

        # Store trade details for top variant
        metrics['trades'] = trades[:10]  # first 10 for inspection

        results['variant_metrics'].append(metrics)

        if metrics['sharpe'] > best_sharpe and metrics['total_trades'] >= 10:
            best_sharpe = metrics['sharpe']
            best_variant = variant_key

    # Summary
    results['best_variant'] = best_variant

    passing_variants = [v for v in results['variant_metrics']
                        if v.get('validation', {}).get('all_pass', False)]
    results['validation'] = {
        'passing_variants': len(passing_variants),
        'total_variants': len(variant_configs),
        'best_variant': best_variant,
        'best_sharpe': best_sharpe,
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<8} {'Trades':<8} {'Return%':<10} {'WR%':<8} {'PF':<8} "
          f"{'Sharpe':<8} {'Sortino':<8} {'MDD%':<8} {'PermP':<8} {'Pass?':<6}")
    print("-" * 84)
    for v in results['variant_metrics']:
        status = "YES" if v.get('validation', {}).get('all_pass', False) else "NO"
        print(f"{v['variant']:<8} {v['total_trades']:<8} {v['total_return_pct']:<10.1f} "
              f"{v['win_rate']:<8.1f} {v['profit_factor']:<8.2f} "
              f"{v['sharpe']:<8.2f} {v['sortino']:<8.2f} {v['max_drawdown_pct']:<8.1f} "
              f"{v['perm_p_value']:<8.4f} {status:<6}")

    if best_variant:
        bv = next(v for v in results['variant_metrics'] if v['variant'] == best_variant)
        print(f"\nBest variant: {best_variant} — {bv['name']}")
        print(f"  Sharpe {bv['sharpe']:.2f}, Return {bv['total_return_pct']:.1f}%, "
              f"{bv['total_trades']} trades")

    # Save results
    output_path = os.path.join(LVL3_ROOT, 'data', 'earnings_momentum_options_results.json')
    # Remove non-serializable items for JSON
    save_results = json.loads(json.dumps(results, default=str))
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    results = main()
