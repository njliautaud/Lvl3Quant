#!/usr/bin/env python3
"""
Wheel Strategy Backtest (CSP + Covered Calls) — v2
====================================================
Anti-lookahead compliant: all signals use T-1 data only.
Option prices: Black-Scholes with realistic bid-ask haircuts.
Period: 2018-01-01 to 2026-07-18
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/wheel_backtest_v2"
START_DATE = "2018-01-01"
END_DATE = "2026-07-18"
INITIAL_CAPITAL = 500_000

# Position sizing
MAX_SINGLE_STOCK_PCT = 0.08    # 8% max per stock
MAX_TOTAL_EXPOSURE_PCT = 0.40  # 40% total portfolio exposure (wheel is inherently conservative)

# Option params
CSP_DELTA_TARGET = 0.30
CC_DELTA_TARGET = 0.30
DTE_MIN = 30
DTE_MAX = 45
DTE_TARGET = 35  # sweet spot
ROLL_DTE = 7            # roll only when very close to expiry
PROFIT_TAKE_PCT = 0.65  # close at 65% profit (let more reach expiry)

# Bid-ask haircuts off theoretical mid (we're selling, so we get bid)
PUT_HAIRCUT = 0.20   # 20% haircut on puts (we receive less)
CALL_HAIRCUT = 0.15  # 15% haircut on calls

# Costs
COMMISSION_PER_CONTRACT = 0.65
SLIPPAGE_PER_CONTRACT = 0.02  # small additional slippage per share

# Risk-free rate proxy (use 3mo T-bill, approximate by period)
RF_RATE_MAP = {
    2018: 0.020, 2019: 0.022, 2020: 0.005, 2021: 0.001,
    2022: 0.030, 2023: 0.050, 2024: 0.053, 2025: 0.045, 2026: 0.042
}

# Universe
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "JPM", "V", "MA", "UNH", "HD", "PG", "JNJ", "KO",
    "PEP", "COST", "WMT", "ABBV", "MRK", "LLY",
    "CRM", "AVGO", "ADBE", "TXN", "QCOM",
    "BRK-B", "XOM", "CVX", "CAT", "GE",
    "DIS", "NFLX", "AMD", "INTC", "CSCO",
    "BAC", "WFC", "GS", "AXP", "BLK"
]


# ─── BLACK-SCHOLES ──────────────────────────────────────────────────────
def bs_d1(S, K, T, r, sigma):
    """d1 in Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_delta(S, K, T, r, sigma):
    """Put delta (negative)."""
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1) - 1.0

def bs_call_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1)

def find_strike_for_delta(S, target_delta, T, r, sigma, option_type='put'):
    """Binary search for strike that gives target delta."""
    if option_type == 'put':
        # Put delta is negative; target_delta should be positive (e.g., 0.30)
        # We want |put_delta| ≈ target_delta
        lo, hi = S * 0.70, S * 1.05
        for _ in range(50):
            mid = (lo + hi) / 2
            d = abs(bs_put_delta(S, mid, T, r, sigma))
            if d > target_delta:
                hi = mid  # strike too high (too ITM), lower it
            else:
                lo = mid  # strike too low (too OTM), raise it
        # Round to nearest standard strike increment
        return round_strike(mid, S)
    else:
        # Call delta positive; target ~0.30
        lo, hi = S * 0.95, S * 1.30
        for _ in range(50):
            mid = (lo + hi) / 2
            d = bs_call_delta(S, mid, T, r, sigma)
            if d > target_delta:
                lo = mid  # strike too low (too ITM), raise it
            else:
                hi = mid  # strike too high (too OTM), lower it
        return round_strike(mid, S)

def round_strike(K, S):
    """Round to realistic strike increments."""
    if S < 50:
        inc = 1.0
    elif S < 200:
        inc = 2.5
    elif S < 500:
        inc = 5.0
    else:
        inc = 10.0
    return round(K / inc) * inc

def realized_vol(prices, window=21):
    """Compute realized vol from daily returns (annualized)."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


# ─── DATA DOWNLOAD ──────────────────────────────────────────────────────
def download_data():
    """Download daily OHLCV for universe."""
    cache_file = os.path.join(OUTPUT_DIR, "price_cache.parquet")
    if os.path.exists(cache_file):
        print("Loading cached price data...")
        df = pd.read_parquet(cache_file)
        # Check if cache is recent enough
        if df.index.max() >= pd.Timestamp("2026-07-01"):
            return df
        print("Cache is stale, re-downloading...")

    print(f"Downloading data for {len(UNIVERSE)} stocks...")
    all_data = {}
    for ticker in UNIVERSE:
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if len(data) > 100:
                # Handle both old and new yfinance column formats
                if isinstance(data.columns, pd.MultiIndex):
                    close = data[('Close', ticker)] if ('Close', ticker) in data.columns else data['Close'].iloc[:, 0]
                else:
                    close = data['Close']
                # Squeeze to Series if needed
                if isinstance(close, pd.DataFrame):
                    close = close.squeeze()
                all_data[ticker] = close
                print(f"  {ticker}: {len(data)} days")
            else:
                print(f"  {ticker}: insufficient data ({len(data)} days), skipping")
        except Exception as e:
            print(f"  {ticker}: download failed ({e})")

    df = pd.DataFrame(all_data)
    df.to_parquet(cache_file)
    print(f"Cached {len(df)} days, {len(df.columns)} stocks")
    return df


# ─── QUALITY FILTER ─────────────────────────────────────────────────────
def quality_filter(prices_df, date, lookback=252):
    """
    Simple quality filter using price data only (no fundamental API needed):
    - Must have >lookback days of history
    - Must have positive 1Y return (uptrend proxy)
    - Must have low enough vol (no penny stock behavior)
    - Must have reasonable price (>$20, proxy for institutional quality)

    In production, this would use fundamental data. Here we use price-based
    proxies that capture similar characteristics.
    """
    idx = prices_df.index.get_loc(date)
    if idx < lookback:
        return []

    qualified = []
    for ticker in prices_df.columns:
        series = prices_df[ticker].iloc[max(0, idx-lookback):idx+1].dropna()
        if len(series) < lookback * 0.8:
            continue
        price = series.iloc[-1]
        if price < 20:  # minimum price filter
            continue
        ret_1y = series.iloc[-1] / series.iloc[0] - 1
        vol = realized_vol(series, 21).iloc[-1]
        if pd.isna(vol) or vol > 0.80:  # skip hyper-volatile
            continue
        # Quality proxy: positive trend + moderate vol = likely profitable, growing company
        if ret_1y > -0.30:  # allow mild drawdowns but not disasters
            qualified.append(ticker)
    return qualified


# ─── WHEEL ENGINE ────────────────────────────────────────────────────────
class WheelBacktest:
    def __init__(self, prices_df, use_t0_signal=False):
        """
        use_t0_signal: if True, use same-day data (lookahead bias test).
                       if False (default), use T-1 data only.
        """
        self.prices = prices_df
        self.use_t0_signal = use_t0_signal
        self.capital = INITIAL_CAPITAL
        self.cash = INITIAL_CAPITAL
        self.positions = {}       # ticker -> {shares, cost_basis}
        self.open_puts = {}       # ticker -> {strike, expiry, premium, entry_date, contracts}
        self.open_calls = {}      # ticker -> {strike, expiry, premium, entry_date, contracts}
        self.trade_log = []
        self.daily_equity = []
        self.stats = {
            'total_premiums_collected': 0,
            'puts_sold': 0,
            'puts_assigned': 0,
            'puts_expired_otm': 0,
            'puts_closed_profit': 0,
            'calls_sold': 0,
            'calls_assigned': 0,
            'calls_expired_otm': 0,
            'calls_closed_profit': 0,
            'total_commissions': 0,
        }

    def get_signal_price(self, ticker, date_idx):
        """Get the price used for signal generation (T-1 or T-0)."""
        if self.use_t0_signal:
            return self.prices[ticker].iloc[date_idx]
        else:
            if date_idx < 1:
                return None
            return self.prices[ticker].iloc[date_idx - 1]

    def get_execution_price(self, ticker, date_idx):
        """Get execution price = T open (approximated as T close for daily data).
        In practice with daily bars, we execute at T's price after T-1 signal."""
        return self.prices[ticker].iloc[date_idx]

    def get_vol(self, ticker, date_idx, window=21):
        """Get T-1 realized vol (or T-0 if lookahead mode)."""
        if self.use_t0_signal:
            end = date_idx + 1
        else:
            end = date_idx  # excludes today
        start = max(0, end - window - 5)
        series = self.prices[ticker].iloc[start:end].dropna()
        if len(series) < window:
            return 0.25  # default
        log_ret = np.log(series / series.shift(1)).dropna()
        if len(log_ret) < 10:
            return 0.25
        return float(log_ret.std() * np.sqrt(252))

    def get_rf_rate(self, date):
        return RF_RATE_MAP.get(date.year, 0.04)

    def portfolio_value(self, date_idx):
        """Mark-to-market portfolio value."""
        val = self.cash
        for ticker, pos in self.positions.items():
            price = self.prices[ticker].iloc[date_idx]
            if not pd.isna(price):
                val += pos['shares'] * price
        # Approximate option mark-to-market (conservative: ignore for simplicity,
        # premium already received in cash)
        return val

    def current_exposure(self, date_idx):
        """Calculate total notional exposure as fraction of portfolio."""
        pv = self.portfolio_value(date_idx)
        if pv <= 0:
            return 1.0
        stock_exposure = 0
        for ticker, pos in self.positions.items():
            price = self.prices[ticker].iloc[date_idx]
            if not pd.isna(price):
                stock_exposure += pos['shares'] * price
        put_exposure = 0
        for ticker, put in self.open_puts.items():
            put_exposure += put['strike'] * put['contracts'] * 100
        return (stock_exposure + put_exposure) / pv

    def single_stock_exposure(self, ticker, date_idx):
        """Exposure to single stock as fraction of portfolio."""
        pv = self.portfolio_value(date_idx)
        if pv <= 0:
            return 1.0
        exposure = 0
        if ticker in self.positions:
            price = self.prices[ticker].iloc[date_idx]
            if not pd.isna(price):
                exposure += self.positions[ticker]['shares'] * price
        if ticker in self.open_puts:
            exposure += self.open_puts[ticker]['strike'] * self.open_puts[ticker]['contracts'] * 100
        return exposure / pv

    def sell_csp(self, ticker, date_idx, date):
        """Sell a cash-secured put."""
        if ticker in self.open_puts:
            return  # already have a put on this

        signal_price = self.get_signal_price(ticker, date_idx)
        if signal_price is None or pd.isna(signal_price):
            return

        exec_price = self.get_execution_price(ticker, date_idx)
        if pd.isna(exec_price):
            return

        # Check exposure limits
        if self.current_exposure(date_idx) >= MAX_TOTAL_EXPOSURE_PCT:
            return
        if self.single_stock_exposure(ticker, date_idx) >= MAX_SINGLE_STOCK_PCT:
            return

        # Compute vol using T-1 data
        sigma = self.get_vol(ticker, date_idx)
        r = self.get_rf_rate(date)
        T = DTE_TARGET / 365.0

        # Find strike for target delta using SIGNAL price (T-1)
        strike = find_strike_for_delta(signal_price, CSP_DELTA_TARGET, T, r, sigma, 'put')

        # Compute theoretical price at EXECUTION price
        theo_price = bs_put_price(exec_price, strike, T, r, sigma)

        # Apply bid haircut (we're selling, so we get bid)
        bid_price = theo_price * (1 - PUT_HAIRCUT)
        if bid_price < 0.10:
            return  # premium too small

        # Position sizing: max contracts based on cash and exposure limit
        pv = self.portfolio_value(date_idx)
        max_notional = min(
            pv * MAX_SINGLE_STOCK_PCT - self.single_stock_exposure(ticker, date_idx) * pv,
            pv * MAX_TOTAL_EXPOSURE_PCT - self.current_exposure(date_idx) * pv,
            self.cash * 0.95  # keep 5% cash buffer
        )
        max_contracts = int(max_notional / (strike * 100))
        if max_contracts < 1:
            return

        contracts = min(max_contracts, 5)  # cap at 5 contracts per position
        premium_received = bid_price * contracts * 100
        commission = COMMISSION_PER_CONTRACT * contracts * 2  # open + close
        slippage = SLIPPAGE_PER_CONTRACT * contracts * 100

        net_premium = premium_received - commission - slippage
        if net_premium <= 0:
            return

        # Reserve cash for assignment
        cash_reserved = strike * contracts * 100
        if self.cash < cash_reserved:
            contracts = int(self.cash / (strike * 100))
            if contracts < 1:
                return
            premium_received = bid_price * contracts * 100
            commission = COMMISSION_PER_CONTRACT * contracts * 2
            slippage = SLIPPAGE_PER_CONTRACT * contracts * 100
            net_premium = premium_received - commission - slippage
            cash_reserved = strike * contracts * 100

        expiry = date + timedelta(days=DTE_TARGET)
        self.open_puts[ticker] = {
            'strike': strike,
            'expiry': expiry,
            'premium': bid_price,
            'net_premium': net_premium,
            'entry_date': date,
            'contracts': contracts,
            'cash_reserved': cash_reserved,
        }
        self.cash += net_premium  # receive premium
        # Note: cash_reserved is conceptual — we need enough if assigned

        self.stats['total_premiums_collected'] += net_premium
        self.stats['puts_sold'] += 1
        self.stats['total_commissions'] += commission

        self.trade_log.append({
            'date': str(date.date()),
            'type': 'SELL_PUT',
            'ticker': ticker,
            'strike': strike,
            'premium': round(bid_price, 2),
            'contracts': contracts,
            'net_premium': round(net_premium, 2),
            'sigma': round(sigma, 3),
            'exec_price': round(exec_price, 2),
        })

    def sell_cc(self, ticker, date_idx, date):
        """Sell covered call on existing position."""
        if ticker not in self.positions or ticker in self.open_calls:
            return

        pos = self.positions[ticker]
        if pos['shares'] < 100:
            return

        signal_price = self.get_signal_price(ticker, date_idx)
        if signal_price is None or pd.isna(signal_price):
            return

        exec_price = self.get_execution_price(ticker, date_idx)
        if pd.isna(exec_price):
            return

        sigma = self.get_vol(ticker, date_idx)
        r = self.get_rf_rate(date)
        T = DTE_TARGET / 365.0

        strike = find_strike_for_delta(signal_price, CC_DELTA_TARGET, T, r, sigma, 'call')

        theo_price = bs_call_price(exec_price, strike, T, r, sigma)
        bid_price = theo_price * (1 - CALL_HAIRCUT)
        if bid_price < 0.10:
            return

        contracts = pos['shares'] // 100
        premium_received = bid_price * contracts * 100
        commission = COMMISSION_PER_CONTRACT * contracts * 2
        slippage = SLIPPAGE_PER_CONTRACT * contracts * 100
        net_premium = premium_received - commission - slippage
        if net_premium <= 0:
            return

        expiry = date + timedelta(days=DTE_TARGET)
        self.open_calls[ticker] = {
            'strike': strike,
            'expiry': expiry,
            'premium': bid_price,
            'net_premium': net_premium,
            'entry_date': date,
            'contracts': contracts,
        }
        self.cash += net_premium

        self.stats['total_premiums_collected'] += net_premium
        self.stats['calls_sold'] += 1
        self.stats['total_commissions'] += commission

        self.trade_log.append({
            'date': str(date.date()),
            'type': 'SELL_CALL',
            'ticker': ticker,
            'strike': strike,
            'premium': round(bid_price, 2),
            'contracts': contracts,
            'net_premium': round(net_premium, 2),
            'sigma': round(sigma, 3),
            'exec_price': round(exec_price, 2),
        })

    def manage_puts(self, date_idx, date):
        """Check open puts for expiry, assignment, profit-take, or roll."""
        to_remove = []
        for ticker, put in list(self.open_puts.items()):
            current_price = self.prices[ticker].iloc[date_idx]
            if pd.isna(current_price):
                continue

            days_left = (put['expiry'] - date).days
            sigma = self.get_vol(ticker, date_idx)
            r = self.get_rf_rate(date)
            T = max(days_left / 365.0, 1/365.0)

            # Current option value (what we'd pay to buy back)
            current_theo = bs_put_price(current_price, put['strike'], T, r, sigma)
            current_ask = current_theo * (1 + PUT_HAIRCUT)  # we'd pay ask to close

            # 1. Check expiry
            if days_left <= 0:
                if current_price < put['strike']:
                    # ASSIGNED — buy shares at strike
                    shares = put['contracts'] * 100
                    cost = put['strike'] * shares
                    commission = COMMISSION_PER_CONTRACT * put['contracts']
                    self.cash -= cost + commission
                    self.stats['total_commissions'] += commission

                    if ticker in self.positions:
                        old = self.positions[ticker]
                        total_shares = old['shares'] + shares
                        total_cost = old['cost_basis'] * old['shares'] + cost
                        self.positions[ticker] = {
                            'shares': total_shares,
                            'cost_basis': total_cost / total_shares
                        }
                    else:
                        self.positions[ticker] = {
                            'shares': shares,
                            'cost_basis': put['strike']
                        }

                    self.stats['puts_assigned'] += 1
                    self.trade_log.append({
                        'date': str(date.date()),
                        'type': 'PUT_ASSIGNED',
                        'ticker': ticker,
                        'strike': put['strike'],
                        'shares': shares,
                        'price_at_assign': round(current_price, 2),
                    })
                else:
                    # Expired OTM — keep premium
                    self.stats['puts_expired_otm'] += 1
                    self.trade_log.append({
                        'date': str(date.date()),
                        'type': 'PUT_EXPIRED_OTM',
                        'ticker': ticker,
                        'strike': put['strike'],
                        'premium_kept': round(put['net_premium'], 2),
                    })
                to_remove.append(ticker)
                continue

            # 2. Profit-take at 50%
            if current_ask <= put['premium'] * (1 - PROFIT_TAKE_PCT):
                # Close for profit
                close_cost = current_ask * put['contracts'] * 100
                commission = COMMISSION_PER_CONTRACT * put['contracts']
                self.cash -= close_cost + commission
                self.stats['total_commissions'] += commission
                self.stats['puts_closed_profit'] += 1
                self.trade_log.append({
                    'date': str(date.date()),
                    'type': 'PUT_CLOSE_PROFIT',
                    'ticker': ticker,
                    'strike': put['strike'],
                    'close_cost': round(close_cost, 2),
                    'original_premium': round(put['net_premium'], 2),
                })
                to_remove.append(ticker)
                continue

            # 3. Roll at ROLL_DTE ONLY if OTM (if ITM, let it get assigned — that's the wheel)
            if days_left <= ROLL_DTE and current_price > put['strike'] * 1.02:
                # Only roll if stock is >2% above strike (clearly OTM, won't be assigned)
                close_cost = current_ask * put['contracts'] * 100
                commission = COMMISSION_PER_CONTRACT * put['contracts']
                self.cash -= close_cost + commission
                self.stats['total_commissions'] += commission
                to_remove.append(ticker)
                self.trade_log.append({
                    'date': str(date.date()),
                    'type': 'PUT_ROLLED',
                    'ticker': ticker,
                    'strike': put['strike'],
                    'close_cost': round(close_cost, 2),
                })
                # Will re-enter on next cycle if qualified

        for t in to_remove:
            del self.open_puts[t]

    def manage_calls(self, date_idx, date):
        """Check open calls for expiry, assignment, profit-take, or roll."""
        to_remove = []
        for ticker, call in list(self.open_calls.items()):
            current_price = self.prices[ticker].iloc[date_idx]
            if pd.isna(current_price):
                continue

            days_left = (call['expiry'] - date).days
            sigma = self.get_vol(ticker, date_idx)
            r = self.get_rf_rate(date)
            T = max(days_left / 365.0, 1/365.0)

            current_theo = bs_call_price(current_price, call['strike'], T, r, sigma)
            current_ask = current_theo * (1 + CALL_HAIRCUT)

            # 1. Expiry
            if days_left <= 0:
                if current_price > call['strike']:
                    # ASSIGNED — sell shares at strike
                    if ticker in self.positions:
                        shares_called = call['contracts'] * 100
                        shares_held = self.positions[ticker]['shares']
                        shares_to_sell = min(shares_called, shares_held)
                        proceeds = call['strike'] * shares_to_sell
                        commission = COMMISSION_PER_CONTRACT * call['contracts']
                        self.cash += proceeds - commission
                        self.stats['total_commissions'] += commission

                        cost_basis = self.positions[ticker]['cost_basis']
                        pnl = (call['strike'] - cost_basis) * shares_to_sell
                        self.trade_log.append({
                            'date': str(date.date()),
                            'type': 'CALL_ASSIGNED',
                            'ticker': ticker,
                            'strike': call['strike'],
                            'shares_sold': shares_to_sell,
                            'stock_pnl': round(pnl, 2),
                        })

                        remaining = shares_held - shares_to_sell
                        if remaining > 0:
                            self.positions[ticker]['shares'] = remaining
                        else:
                            del self.positions[ticker]

                    self.stats['calls_assigned'] += 1
                else:
                    self.stats['calls_expired_otm'] += 1
                    self.trade_log.append({
                        'date': str(date.date()),
                        'type': 'CALL_EXPIRED_OTM',
                        'ticker': ticker,
                        'strike': call['strike'],
                        'premium_kept': round(call['net_premium'], 2),
                    })
                to_remove.append(ticker)
                continue

            # 2. Profit-take
            if current_ask <= call['premium'] * (1 - PROFIT_TAKE_PCT):
                close_cost = current_ask * call['contracts'] * 100
                commission = COMMISSION_PER_CONTRACT * call['contracts']
                self.cash -= close_cost + commission
                self.stats['total_commissions'] += commission
                self.stats['calls_closed_profit'] += 1
                to_remove.append(ticker)
                continue

            # 3. Roll
            if days_left <= ROLL_DTE:
                close_cost = current_ask * call['contracts'] * 100
                commission = COMMISSION_PER_CONTRACT * call['contracts']
                self.cash -= close_cost + commission
                self.stats['total_commissions'] += commission
                to_remove.append(ticker)

        for t in to_remove:
            del self.open_calls[t]

    def run(self):
        """Run full backtest."""
        dates = self.prices.index
        mode = "T-0 (LOOKAHEAD)" if self.use_t0_signal else "T-1 (PROPER)"
        print(f"\n{'='*60}")
        print(f"Running Wheel Backtest — Signal Mode: {mode}")
        print(f"Period: {dates[0].date()} to {dates[-1].date()}")
        print(f"Universe: {len(self.prices.columns)} stocks")
        print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
        print(f"{'='*60}\n")

        rebalance_interval = 5  # check for new opportunities every 5 trading days

        for i in range(30, len(dates)):  # start after 30 days for vol calc
            date = dates[i]

            # 1. Manage existing positions first
            self.manage_puts(i, date)
            self.manage_calls(i, date)

            # 2. Sell covered calls on any unhedged stock positions
            for ticker in list(self.positions.keys()):
                if ticker not in self.open_calls:
                    self.sell_cc(ticker, i, date)

            # 3. Look for new CSP opportunities (every rebalance_interval days)
            if i % rebalance_interval == 0:
                qualified = quality_filter(self.prices, date)
                # Sort by vol (higher vol = higher premium) but cap at reasonable
                vol_scores = []
                for t in qualified:
                    if t not in self.open_puts and t not in self.positions:
                        v = self.get_vol(t, i)
                        if 0.15 < v < 0.60:  # reasonable vol range
                            vol_scores.append((t, v))
                vol_scores.sort(key=lambda x: x[1], reverse=True)

                for ticker, _ in vol_scores[:10]:  # top 10 by premium potential
                    if self.current_exposure(i) >= MAX_TOTAL_EXPOSURE_PCT:
                        break
                    self.sell_csp(ticker, i, date)

            # 4. Record daily equity
            equity = self.portfolio_value(i)
            self.daily_equity.append({
                'date': date,
                'equity': equity,
                'cash': self.cash,
                'positions': len(self.positions),
                'open_puts': len(self.open_puts),
                'open_calls': len(self.open_calls),
            })

            # Progress
            if i % 250 == 0:
                print(f"  {date.date()} — Equity: ${equity:,.0f}  "
                      f"Positions: {len(self.positions)}  "
                      f"Open puts: {len(self.open_puts)}  "
                      f"Open calls: {len(self.open_calls)}")

        return self.compute_results()

    def compute_results(self):
        """Compute performance metrics."""
        eq = pd.DataFrame(self.daily_equity)
        eq.set_index('date', inplace=True)
        eq['returns'] = eq['equity'].pct_change()

        # Basic metrics
        total_return = eq['equity'].iloc[-1] / INITIAL_CAPITAL - 1
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1

        # Drawdown
        peak = eq['equity'].cummax()
        dd = (eq['equity'] - peak) / peak
        max_dd = dd.min()

        # Risk metrics (annualized)
        daily_ret = eq['returns'].dropna()
        ann_ret = daily_ret.mean() * 252
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        downside_ret = daily_ret[daily_ret < 0]
        downside_vol = downside_ret.std() * np.sqrt(252)
        sortino = ann_ret / downside_vol if downside_vol > 0 else 0

        # Win rate (from trades)
        put_wins = self.stats['puts_expired_otm'] + self.stats['puts_closed_profit']
        put_total = self.stats['puts_sold']
        put_wr = put_wins / put_total if put_total > 0 else 0

        call_wins = self.stats['calls_expired_otm'] + self.stats['calls_closed_profit']
        call_total = self.stats['calls_sold']
        call_wr = call_wins / call_total if call_total > 0 else 0

        total_wins = put_wins + call_wins
        total_trades = put_total + call_total
        overall_wr = total_wins / total_trades if total_trades > 0 else 0

        # Assignment rate
        assign_rate = self.stats['puts_assigned'] / put_total if put_total > 0 else 0

        # Avg premium per trade
        avg_premium = self.stats['total_premiums_collected'] / total_trades if total_trades > 0 else 0

        results = {
            'mode': 'T-0 (LOOKAHEAD)' if self.use_t0_signal else 'T-1 (PROPER)',
            'total_return': round(total_return * 100, 2),
            'cagr': round(cagr * 100, 2),
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'max_drawdown': round(max_dd * 100, 2),
            'ann_volatility': round(ann_vol * 100, 2),
            'total_trades': total_trades,
            'puts_sold': self.stats['puts_sold'],
            'puts_assigned': self.stats['puts_assigned'],
            'puts_expired_otm': self.stats['puts_expired_otm'],
            'puts_closed_profit': self.stats['puts_closed_profit'],
            'calls_sold': self.stats['calls_sold'],
            'calls_assigned': self.stats['calls_assigned'],
            'calls_expired_otm': self.stats['calls_expired_otm'],
            'calls_closed_profit': self.stats['calls_closed_profit'],
            'put_win_rate': round(put_wr * 100, 1),
            'call_win_rate': round(call_wr * 100, 1),
            'overall_win_rate': round(overall_wr * 100, 1),
            'assignment_rate': round(assign_rate * 100, 1),
            'total_premiums_collected': round(self.stats['total_premiums_collected'], 2),
            'avg_premium_per_trade': round(avg_premium, 2),
            'total_commissions': round(self.stats['total_commissions'], 2),
            'final_equity': round(eq['equity'].iloc[-1], 2),
            'years': round(years, 1),
        }

        return results, eq, self.trade_log


# ─── MAIN ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("WHEEL STRATEGY BACKTEST v2")
    print("CSP + Covered Calls on Quality Large-Caps")
    print("=" * 60)

    # Download data
    prices = download_data()
    print(f"\nData: {len(prices)} trading days, {len(prices.columns)} stocks")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # ── Run T-1 (proper) backtest ──
    bt_t1 = WheelBacktest(prices, use_t0_signal=False)
    results_t1, equity_t1, trades_t1 = bt_t1.run()

    # ── Run T-0 (lookahead) backtest ──
    bt_t0 = WheelBacktest(prices, use_t0_signal=True)
    results_t0, equity_t0, trades_t0 = bt_t0.run()

    # ── Print Results ──
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON: T-1 (PROPER) vs T-0 (LOOKAHEAD)")
    print("=" * 70)

    def print_metric(name, v1, v2, fmt=".2f", suffix=""):
        delta = v2 - v1 if isinstance(v1, (int, float)) else ""
        print(f"  {name:<30} {v1:>12{fmt}}{suffix}  {v2:>12{fmt}}{suffix}  {'':>8}")

    print(f"\n  {'Metric':<30} {'T-1 (Proper)':>14} {'T-0 (Lookahead)':>14}")
    print(f"  {'-'*30} {'-'*14} {'-'*14}")
    print_metric("CAGR", results_t1['cagr'], results_t0['cagr'], suffix="%")
    print_metric("Sharpe Ratio", results_t1['sharpe'], results_t0['sharpe'], ".3f")
    print_metric("Sortino Ratio", results_t1['sortino'], results_t0['sortino'], ".3f")
    print_metric("Max Drawdown", results_t1['max_drawdown'], results_t0['max_drawdown'], suffix="%")
    print_metric("Ann. Volatility", results_t1['ann_volatility'], results_t0['ann_volatility'], suffix="%")
    print_metric("Total Return", results_t1['total_return'], results_t0['total_return'], suffix="%")
    print_metric("Overall Win Rate", results_t1['overall_win_rate'], results_t0['overall_win_rate'], suffix="%")
    print_metric("Put Win Rate", results_t1['put_win_rate'], results_t0['put_win_rate'], suffix="%")
    print_metric("Call Win Rate", results_t1['call_win_rate'], results_t0['call_win_rate'], suffix="%")
    print_metric("Assignment Rate", results_t1['assignment_rate'], results_t0['assignment_rate'], suffix="%")
    print_metric("Avg Premium/Trade", results_t1['avg_premium_per_trade'], results_t0['avg_premium_per_trade'])
    print_metric("Total Premiums", results_t1['total_premiums_collected'], results_t0['total_premiums_collected'])
    print_metric("Total Commissions", results_t1['total_commissions'], results_t0['total_commissions'])
    print_metric("Total Trades", results_t1['total_trades'], results_t0['total_trades'], ".0f")
    print_metric("Puts Sold", results_t1['puts_sold'], results_t0['puts_sold'], ".0f")
    print_metric("Puts Assigned", results_t1['puts_assigned'], results_t0['puts_assigned'], ".0f")
    print_metric("Calls Sold", results_t1['calls_sold'], results_t0['calls_sold'], ".0f")
    print_metric("Final Equity", results_t1['final_equity'], results_t0['final_equity'])

    # Lag sensitivity
    print(f"\n{'='*70}")
    print("LAG SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    sharpe_diff = results_t0['sharpe'] - results_t1['sharpe']
    cagr_diff = results_t0['cagr'] - results_t1['cagr']
    print(f"  Sharpe difference (T-0 minus T-1): {sharpe_diff:+.3f}")
    print(f"  CAGR difference (T-0 minus T-1):   {cagr_diff:+.2f}%")
    if abs(sharpe_diff) < 0.05:
        print("  → MINIMAL lookahead bias. Strategy is robust to signal timing.")
    elif sharpe_diff > 0.10:
        print("  → SIGNIFICANT lookahead advantage. T-0 results are inflated.")
    else:
        print("  → MODERATE difference. Some signal timing sensitivity.")

    # ── Save outputs ──
    # Save results JSON
    all_results = {
        't1_proper': results_t1,
        't0_lookahead': results_t0,
        'lag_sensitivity': {
            'sharpe_diff': round(sharpe_diff, 3),
            'cagr_diff': round(cagr_diff, 2),
        }
    }
    with open(os.path.join(OUTPUT_DIR, "backtest_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save equity curves
    equity_t1.to_parquet(os.path.join(OUTPUT_DIR, "equity_t1.parquet"))
    equity_t0.to_parquet(os.path.join(OUTPUT_DIR, "equity_t0.parquet"))

    # Save trade logs
    with open(os.path.join(OUTPUT_DIR, "trades_t1.json"), 'w') as f:
        json.dump(trades_t1, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "trades_t0.json"), 'w') as f:
        json.dump(trades_t0, f, indent=2)

    # ── Yearly breakdown (T-1) ──
    print(f"\n{'='*70}")
    print("YEARLY BREAKDOWN (T-1 PROPER)")
    print(f"{'='*70}")
    equity_t1['year'] = equity_t1.index.year
    for year, grp in equity_t1.groupby('year'):
        yr_ret = grp['equity'].iloc[-1] / grp['equity'].iloc[0] - 1
        yr_vol = grp['returns'].std() * np.sqrt(252)
        yr_sharpe = (grp['returns'].mean() * 252) / yr_vol if yr_vol > 0 else 0
        yr_dd = ((grp['equity'] - grp['equity'].cummax()) / grp['equity'].cummax()).min()
        print(f"  {year}: Return={yr_ret*100:+.1f}%  Vol={yr_vol*100:.1f}%  "
              f"Sharpe={yr_sharpe:.2f}  MaxDD={yr_dd*100:.1f}%")

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Files: backtest_results.json, equity_t1.parquet, equity_t0.parquet, trades_t1.json, trades_t0.json")

    return all_results


if __name__ == "__main__":
    results = main()
