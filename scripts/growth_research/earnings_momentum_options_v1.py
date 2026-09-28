#!/usr/bin/env python3
"""
Earnings Momentum Options Strategy v1
======================================
Single-leg options strategy targeting post-earnings moves.

KEY INSIGHT:
- Earnings gap scanner shows stocks like ENPH (avg 11.5% gap), RBLX (14.8%),
  PLTR (11.0%) produce large post-earnings moves.
- Momentum burst strategy (Sharpe 1.28) proves short-term single-leg options
  work with proper exit management.
- This script tests whether targeting earnings catalysts specifically
  improves single-leg options performance.

6 VARIANTS:
  A. Post-gap buyer: gap > 5%, buy ATM in gap direction at open, hold 1-3 days
  B. Post-gap buyer (10% threshold): same as A but gaps > 10%
  C. Pre-earnings momentum: buy call/put 2d before earnings based on 5d momentum,
     sell at open after earnings
  D. Pre-earnings straddle-like: buy BOTH call+put 2d before (IV benefits from move)
  E. Post-gap + momentum filter: A but only if pre-earnings momentum aligns
  F. Post-gap contrarian: gap DOWN > 10% -> buy call; gap UP > 10% -> buy put

PRICING:
  - Black-Scholes inline (self-contained)
  - Earnings IV = max(VIX/100 * 1.5, realized_vol_21d * 2.0) -- elevated pre-earnings
  - Post-earnings IV crush: IV drops 30-50% after earnings (modeled)
  - Commission: $0.65/leg ($1.30 RT)
  - Account: $645, max $200/position, max 2 concurrent

EXIT: +30% TP, -25% SL, 50% trailing giveback, 3-day time stop

DATA: yfinance for price + earnings dates. 2022-01-01 to 2026-07-25.

MLflow experiment: "earnings_momentum_options"
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

# --- Path setup (works on Jupiter and Neptune) ---
for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'earnings_momentum')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'SQ', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH',
    'DXCM', 'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT',
    'DASH', 'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL',
    'UPST', 'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD',
    'NIO', 'XPEV', 'LI',
]

STARTING_CAPITAL = 645.0
COMMISSION_PER_LEG = 0.65
COMMISSION_RT = 1.30
MAX_POSITIONS = 2
MAX_POSITION_DOLLARS = 200.0
MAX_POSITION_PCT = 0.30
RISK_FREE_RATE = 0.05
START_DATE = '2022-01-01'
END_DATE = '2026-07-25'
N_PERMUTATIONS = 100
DTE = 14  # days to expiration for options

# IV crush parameters
IV_CRUSH_MIN = 0.30  # IV drops at least 30% after earnings
IV_CRUSH_MAX = 0.50  # IV drops up to 50% after earnings
IV_CRUSH_DEFAULT = 0.40  # average crush

# ============================================================
# BLACK-SCHOLES PRICING (inline, self-contained)
# ============================================================

def bs_d1(S, K, T, r, sigma):
    """d1 in Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_call_price(S, K, T, r, sigma):
    """European call price via Black-Scholes."""
    if T <= 1e-8:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """European put price via Black-Scholes."""
    if T <= 1e-8:
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def option_price(S, K, T, r, sigma, option_type='call'):
    """Price a call or put."""
    if option_type == 'call':
        return bs_call_price(S, K, T, r, sigma)
    else:
        return bs_put_price(S, K, T, r, sigma)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """
    Load daily OHLCV for stock universe + SPY + VIX via yfinance with caching.
    Also loads earnings dates for each ticker.
    """
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
                    prices_df = None  # stale cache
        except Exception:
            prices_df = None

    if prices_df is None:
        print(f"Downloading price data for {len(STOCK_UNIVERSE)} stocks + SPY + VIX...")
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
            print(f"Cached price data to {cache_path}")
        except Exception as e:
            print(f"Warning: Could not cache price data: {e}")

    # --- Earnings dates ---
    earnings_dates = {}
    if os.path.exists(earnings_cache_path):
        try:
            with open(earnings_cache_path, 'r') as f:
                earnings_dates = json.load(f)
            # Validate cache is populated enough
            if len(earnings_dates) >= len(STOCK_UNIVERSE) * 0.5:
                total_dates = sum(len(v) for v in earnings_dates.values())
                print(f"Loaded cached earnings dates: {len(earnings_dates)} tickers, "
                      f"{total_dates} total dates")
            else:
                earnings_dates = {}
        except Exception:
            earnings_dates = {}

    if not earnings_dates:
        print("Fetching earnings dates for each ticker...")
        for ticker in STOCK_UNIVERSE:
            try:
                t = yf.Ticker(ticker)
                # Try earnings_dates attribute (has historical earnings dates)
                ed = t.earnings_dates
                if ed is not None and len(ed) > 0:
                    # earnings_dates index has timezone-aware timestamps
                    dates_list = sorted([
                        str(d.date()) if hasattr(d, 'date') else str(d)[:10]
                        for d in ed.index
                    ])
                    # Filter to our date range
                    dates_list = [d for d in dates_list
                                  if START_DATE <= d <= END_DATE]
                    if dates_list:
                        earnings_dates[ticker] = dates_list
                        print(f"  {ticker}: {len(dates_list)} earnings dates")
                    else:
                        print(f"  {ticker}: no earnings dates in range")
                else:
                    # Fallback: try quarterly earnings
                    qe = getattr(t, 'quarterly_earnings', None)
                    if qe is not None and len(qe) > 0:
                        dates_list = sorted([
                            str(d.date()) if hasattr(d, 'date') else str(d)[:10]
                            for d in qe.index
                        ])
                        dates_list = [d for d in dates_list
                                      if START_DATE <= d <= END_DATE]
                        if dates_list:
                            earnings_dates[ticker] = dates_list
                            print(f"  {ticker}: {len(dates_list)} earnings dates (quarterly)")
                    else:
                        print(f"  {ticker}: no earnings data available")
            except Exception as e:
                print(f"  ERROR fetching earnings for {ticker}: {e}")

        # Cache earnings dates
        if earnings_dates:
            try:
                with open(earnings_cache_path, 'w') as f:
                    json.dump(earnings_dates, f, indent=2)
                print(f"Cached earnings dates to {earnings_cache_path}")
            except Exception as e:
                print(f"Warning: Could not cache earnings dates: {e}")

    return prices_df, earnings_dates


def get_ticker_prices(prices_df, ticker):
    """Extract price DataFrame for a single ticker."""
    try:
        return prices_df.loc[ticker].copy()
    except KeyError:
        return None


def get_vix_for_date(vix_data, date):
    """Get VIX level for a given date."""
    mask = vix_data.index <= date
    if mask.any():
        return vix_data.loc[mask, 'close'].iloc[-1]
    return 20.0


def compute_realized_vol(close_prices, lookback=21):
    """Compute annualized realized volatility from close prices."""
    if len(close_prices) < lookback + 1:
        return 0.30  # default
    log_rets = np.diff(np.log(close_prices[-(lookback + 1):]))
    return np.std(log_rets) * np.sqrt(252)


def estimate_earnings_iv(vix_level, realized_vol):
    """
    Estimate IV for earnings plays.
    Earnings IV is elevated: max(VIX/100 * 1.5, realized_vol * 2.0).
    """
    return max(vix_level / 100.0 * 1.5, realized_vol * 2.0)


def estimate_post_earnings_iv(pre_earnings_iv, crush_pct=IV_CRUSH_DEFAULT):
    """IV after earnings drops by crush_pct (30-50%)."""
    return pre_earnings_iv * (1.0 - crush_pct)


def compute_5d_momentum(close_prices):
    """Compute 5-day momentum (return)."""
    if len(close_prices) < 6:
        return 0.0
    return (close_prices[-1] / close_prices[-6]) - 1.0


# ============================================================
# EARNINGS EVENT DETECTION
# ============================================================

def build_earnings_events(prices_df, earnings_dates, vix_data):
    """
    Build a list of earnings events with pre/post data for backtesting.

    For each earnings date:
    - close_before: close on the trading day before earnings
    - open_after: open on the trading day of/after earnings announcement
    - gap_pct: (open_after / close_before) - 1
    - pre_momentum_5d: 5-day momentum before earnings
    - pre_iv: estimated IV before earnings
    - post_iv: estimated IV after earnings (with crush)
    """
    events = []

    for ticker, earn_dates in earnings_dates.items():
        ticker_prices = get_ticker_prices(prices_df, ticker)
        if ticker_prices is None or len(ticker_prices) < 30:
            continue

        trading_days = ticker_prices.index.sort_values()

        for earn_date_str in earn_dates:
            earn_date = pd.Timestamp(earn_date_str)

            # Find the trading day on or just after earnings
            post_mask = trading_days >= earn_date
            if not post_mask.any():
                continue
            post_days = trading_days[post_mask]
            if len(post_days) < 4:  # need at least 3 days post-earnings for hold
                continue
            earnings_day = post_days[0]  # first trading day >= earnings date

            # Find the trading day before earnings
            pre_mask = trading_days < earn_date
            if not pre_mask.any():
                continue
            pre_days = trading_days[pre_mask]
            if len(pre_days) < 10:  # need lookback for momentum
                continue

            close_before = ticker_prices.loc[pre_days[-1], 'close']
            open_after = ticker_prices.loc[earnings_day, 'open']

            if close_before <= 0 or open_after <= 0:
                continue

            gap_pct = (open_after / close_before) - 1.0

            # Pre-earnings momentum (5-day return ending day before earnings)
            pre_close_series = ticker_prices.loc[pre_days[-10:], 'close'].values
            pre_momentum_5d = compute_5d_momentum(pre_close_series)

            # Realized vol for IV estimation
            pre_close_long = ticker_prices.loc[pre_days[-30:], 'close'].values
            realized_vol = compute_realized_vol(pre_close_long)

            # VIX on the day before earnings
            vix_level = get_vix_for_date(vix_data, pre_days[-1])

            # IV estimates
            pre_iv = estimate_earnings_iv(vix_level, realized_vol)
            # Randomize crush within range for realism
            crush_pct = IV_CRUSH_MIN + (IV_CRUSH_MAX - IV_CRUSH_MIN) * (
                hash(f"{ticker}_{earn_date_str}") % 100 / 100.0
            )
            post_iv = estimate_post_earnings_iv(pre_iv, crush_pct)

            # Post-earnings trading days (for holding period simulation)
            post_trading_days = post_days[:5].tolist()  # up to 5 days post
            post_prices = []
            for d in post_trading_days:
                row = ticker_prices.loc[d]
                post_prices.append({
                    'date': d,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                })

            # Pre-earnings trading days (for pre-earnings entry)
            # 2 days before earnings
            if len(pre_days) >= 3:
                pre_entry_day = pre_days[-3]  # 2 trading days before the day before earnings
                pre_entry_close = ticker_prices.loc[pre_entry_day, 'close']
            else:
                pre_entry_day = pre_days[-1]
                pre_entry_close = close_before

            # Days between pre-entry and earnings day (for pre-earnings holds)
            pre_to_post_days = []
            between_mask = (trading_days >= pre_entry_day) & (trading_days <= earnings_day)
            for d in trading_days[between_mask]:
                row = ticker_prices.loc[d]
                pre_to_post_days.append({
                    'date': d,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                })

            events.append({
                'ticker': ticker,
                'earnings_date': earn_date_str,
                'earnings_day': earnings_day,
                'close_before': close_before,
                'open_after': open_after,
                'gap_pct': gap_pct,
                'abs_gap_pct': abs(gap_pct),
                'gap_direction': 'up' if gap_pct > 0 else 'down',
                'pre_momentum_5d': pre_momentum_5d,
                'realized_vol': realized_vol,
                'vix_level': vix_level,
                'pre_iv': pre_iv,
                'post_iv': post_iv,
                'iv_crush_pct': crush_pct,
                'post_prices': post_prices,
                'pre_entry_day': pre_entry_day,
                'pre_entry_close': pre_entry_close,
                'pre_to_post_days': pre_to_post_days,
            })

    # Sort by date
    events.sort(key=lambda e: e['earnings_date'])
    print(f"\nBuilt {len(events)} earnings events across {len(earnings_dates)} tickers")

    # Print gap distribution
    gaps = [abs(e['gap_pct']) for e in events]
    if gaps:
        print(f"  Gap distribution: mean={np.mean(gaps)*100:.1f}%, "
              f"median={np.median(gaps)*100:.1f}%, "
              f"p90={np.percentile(gaps, 90)*100:.1f}%")
        print(f"  Events with |gap| > 5%: {sum(1 for g in gaps if g > 0.05)}")
        print(f"  Events with |gap| > 10%: {sum(1 for g in gaps if g > 0.10)}")
        print(f"  Events with |gap| > 20%: {sum(1 for g in gaps if g > 0.20)}")

    return events


# ============================================================
# POSITION TRACKING
# ============================================================

class Position:
    """Tracks an open option position."""
    def __init__(self, ticker, option_type, strike, entry_price, entry_date,
                 entry_spot, dte, iv, cost, n_contracts=1, variant_tag=''):
        self.ticker = ticker
        self.option_type = option_type  # 'call' or 'put'
        self.strike = strike
        self.entry_price = entry_price  # option premium per share
        self.entry_date = entry_date
        self.entry_spot = entry_spot
        self.dte = dte
        self.iv = iv  # IV at entry
        self.cost = cost  # total cost including commission
        self.n_contracts = n_contracts
        self.days_held = 0
        self.peak_value = entry_price  # for trailing stop
        self.exit_price = None
        self.exit_date = None
        self.exit_reason = None
        self.pnl = None
        self.variant_tag = variant_tag


# ============================================================
# STRATEGY VARIANTS
# ============================================================

VARIANTS = {
    'A': {
        'name': 'Post-Gap Buyer (5%)',
        'description': 'Gap > 5% -> buy ATM in gap direction at open, hold 1-3 days',
        'entry_type': 'post_gap',
        'gap_threshold': 0.05,
        'gap_direction_mode': 'with_gap',  # buy in gap direction
        'require_momentum_alignment': False,
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 3,
        'trailing_giveback': 0.50,
        'dte': DTE,
    },
    'B': {
        'name': 'Post-Gap Buyer (10%)',
        'description': 'Gap > 10% -> buy ATM in gap direction at open, hold 1-3 days',
        'entry_type': 'post_gap',
        'gap_threshold': 0.10,
        'gap_direction_mode': 'with_gap',
        'require_momentum_alignment': False,
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 3,
        'trailing_giveback': 0.50,
        'dte': DTE,
    },
    'C': {
        'name': 'Pre-Earnings Momentum',
        'description': 'Buy call/put 2d before earnings based on 5d momentum, sell at open after',
        'entry_type': 'pre_earnings_momentum',
        'gap_threshold': 0.0,  # no gap filter for pre-earnings
        'gap_direction_mode': 'with_momentum',
        'require_momentum_alignment': False,
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 5,  # will typically exit at earnings open
        'trailing_giveback': 0.50,
        'dte': DTE,
    },
    'D': {
        'name': 'Pre-Earnings Straddle',
        'description': 'Buy BOTH call+put 2d before earnings (profits from any big move)',
        'entry_type': 'pre_earnings_straddle',
        'gap_threshold': 0.0,
        'gap_direction_mode': 'both',
        'require_momentum_alignment': False,
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 5,
        'trailing_giveback': 0.50,
        'dte': DTE,
    },
    'E': {
        'name': 'Post-Gap + Momentum Filter',
        'description': 'Gap > 5% AND pre-earnings momentum aligns with gap direction',
        'entry_type': 'post_gap',
        'gap_threshold': 0.05,
        'gap_direction_mode': 'with_gap',
        'require_momentum_alignment': True,
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 3,
        'trailing_giveback': 0.50,
        'dte': DTE,
    },
    'F': {
        'name': 'Post-Gap Contrarian',
        'description': 'Gap DOWN > 10% -> buy call; Gap UP > 10% -> buy put (mean reversion)',
        'entry_type': 'post_gap',
        'gap_threshold': 0.10,
        'gap_direction_mode': 'against_gap',  # contrarian
        'require_momentum_alignment': False,
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 3,
        'trailing_giveback': 0.50,
        'dte': DTE,
    },
}


# ============================================================
# BACKTESTING ENGINE
# ============================================================

def simulate_post_gap_trade(event, variant_cfg, equity, open_positions_count):
    """
    Simulate a post-gap earnings trade.
    Entry at market open after earnings. Exit over next 1-3 days.

    Returns list of trade dicts (usually 1, but 0 if filtered out).
    """
    gap_pct = event['gap_pct']
    abs_gap = abs(gap_pct)

    # Check gap threshold
    if abs_gap < variant_cfg['gap_threshold']:
        return []

    # Check momentum alignment if required
    if variant_cfg['require_momentum_alignment']:
        momentum = event['pre_momentum_5d']
        if gap_pct > 0 and momentum <= 0:
            return []  # gap up but momentum was down -- no alignment
        if gap_pct < 0 and momentum >= 0:
            return []  # gap down but momentum was up -- no alignment

    # Determine option direction
    direction_mode = variant_cfg['gap_direction_mode']
    if direction_mode == 'with_gap':
        opt_type = 'call' if gap_pct > 0 else 'put'
    elif direction_mode == 'against_gap':
        opt_type = 'put' if gap_pct > 0 else 'call'  # contrarian
    else:
        return []  # shouldn't happen for post-gap

    # Entry at open after earnings
    entry_spot = event['open_after']
    strike = round(entry_spot, 0)  # ATM

    # Post-earnings IV (crushed)
    post_iv = event['post_iv']
    dte = variant_cfg['dte']
    T = dte / 252.0

    entry_premium = option_price(entry_spot, strike, T, RISK_FREE_RATE, post_iv, opt_type)
    if entry_premium < 0.10:
        return []

    contract_cost = entry_premium * 100  # 1 contract = 100 shares
    max_spend = min(MAX_POSITION_DOLLARS, equity * MAX_POSITION_PCT)
    if contract_cost > max_spend or contract_cost + COMMISSION_PER_LEG > equity:
        return []

    total_cost = contract_cost + COMMISSION_PER_LEG

    # Simulate hold period using post_prices
    post_prices = event['post_prices']
    if len(post_prices) < 2:
        return []

    peak_value = entry_premium
    exit_premium = entry_premium
    exit_date = post_prices[0]['date']
    exit_reason = 'time_stop'
    days_held = 0

    # Start from day 1 (day 0 is entry day)
    for day_idx in range(1, min(len(post_prices), variant_cfg['time_stop_days'] + 1)):
        day_data = post_prices[day_idx]
        days_held = day_idx

        # Use close for daily mark-to-market
        current_spot = day_data['close']
        remaining_dte = max(dte - day_idx, 0)
        T_remaining = remaining_dte / 252.0

        # IV continues to normalize after earnings
        current_iv = post_iv * (1.0 + 0.02 * day_idx)  # slight IV recovery over days

        current_value = option_price(current_spot, strike, T_remaining, RISK_FREE_RATE,
                                     current_iv, opt_type)

        if current_value > peak_value:
            peak_value = current_value

        pct_change = (current_value - entry_premium) / entry_premium

        # Take profit
        if pct_change >= variant_cfg['tp_pct']:
            exit_premium = current_value
            exit_date = day_data['date']
            exit_reason = 'take_profit'
            break

        # Stop loss
        if pct_change <= -variant_cfg['sl_pct']:
            exit_premium = current_value
            exit_date = day_data['date']
            exit_reason = 'stop_loss'
            break

        # Trailing stop
        if peak_value > entry_premium:
            unrealized_from_peak = peak_value - entry_premium
            giveback = peak_value - current_value
            if giveback > unrealized_from_peak * variant_cfg['trailing_giveback']:
                exit_premium = current_value
                exit_date = day_data['date']
                exit_reason = 'trailing_stop'
                break

        # Time stop
        if day_idx >= variant_cfg['time_stop_days']:
            exit_premium = current_value
            exit_date = day_data['date']
            exit_reason = 'time_stop'
            break

        exit_premium = current_value
        exit_date = day_data['date']

    # P&L
    exit_value = exit_premium * 100
    pnl = exit_value - total_cost - COMMISSION_PER_LEG  # exit commission
    pnl_pct = (exit_premium - entry_premium) / entry_premium * 100

    return [{
        'ticker': event['ticker'],
        'earnings_date': event['earnings_date'],
        'type': opt_type,
        'strike': strike,
        'entry_date': str(post_prices[0]['date'].date()) if hasattr(post_prices[0]['date'], 'date')
                      else str(post_prices[0]['date'])[:10],
        'exit_date': str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date)[:10],
        'gap_pct': round(event['gap_pct'] * 100, 2),
        'pre_momentum_5d': round(event['pre_momentum_5d'] * 100, 2),
        'entry_premium': round(entry_premium, 4),
        'exit_premium': round(exit_premium, 4),
        'entry_iv': round(post_iv, 4),
        'iv_crush_pct': round(event['iv_crush_pct'] * 100, 1),
        'days_held': days_held,
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 1),
        'exit_reason': exit_reason,
        'cost': round(total_cost, 2),
    }]


def simulate_pre_earnings_momentum_trade(event, variant_cfg, equity, open_positions_count):
    """
    Simulate a pre-earnings momentum trade.
    Entry 2 days before earnings based on 5d momentum direction.
    Exit at market open after earnings.
    """
    momentum = event['pre_momentum_5d']

    # Need meaningful momentum (at least 1% in either direction)
    if abs(momentum) < 0.01:
        return []

    # Direction based on pre-earnings momentum
    opt_type = 'call' if momentum > 0 else 'put'

    entry_spot = event['pre_entry_close']
    strike = round(entry_spot, 0)  # ATM

    # Pre-earnings IV is elevated
    pre_iv = event['pre_iv']
    dte = variant_cfg['dte']
    T = dte / 252.0

    entry_premium = option_price(entry_spot, strike, T, RISK_FREE_RATE, pre_iv, opt_type)
    if entry_premium < 0.10:
        return []

    contract_cost = entry_premium * 100
    max_spend = min(MAX_POSITION_DOLLARS, equity * MAX_POSITION_PCT)
    if contract_cost > max_spend or contract_cost + COMMISSION_PER_LEG > equity:
        return []

    total_cost = contract_cost + COMMISSION_PER_LEG

    # Exit at open after earnings
    exit_spot = event['open_after']

    # Post-earnings IV crush
    post_iv = event['post_iv']

    # Days held = number of trading days from pre-entry to earnings
    pre_to_post = event['pre_to_post_days']
    days_held = len(pre_to_post) - 1 if pre_to_post else 2

    remaining_dte = max(dte - days_held, 0)
    T_exit = remaining_dte / 252.0

    # Option value at exit: price moved but IV crushed
    exit_premium = option_price(exit_spot, strike, T_exit, RISK_FREE_RATE, post_iv, opt_type)

    pnl = (exit_premium * 100) - total_cost - COMMISSION_PER_LEG
    pnl_pct = (exit_premium - entry_premium) / entry_premium * 100

    entry_date_str = (str(event['pre_entry_day'].date())
                      if hasattr(event['pre_entry_day'], 'date')
                      else str(event['pre_entry_day'])[:10])
    exit_date_str = (str(event['earnings_day'].date())
                     if hasattr(event['earnings_day'], 'date')
                     else str(event['earnings_day'])[:10])

    return [{
        'ticker': event['ticker'],
        'earnings_date': event['earnings_date'],
        'type': opt_type,
        'strike': strike,
        'entry_date': entry_date_str,
        'exit_date': exit_date_str,
        'gap_pct': round(event['gap_pct'] * 100, 2),
        'pre_momentum_5d': round(event['pre_momentum_5d'] * 100, 2),
        'entry_premium': round(entry_premium, 4),
        'exit_premium': round(exit_premium, 4),
        'entry_iv': round(pre_iv, 4),
        'iv_crush_pct': round(event['iv_crush_pct'] * 100, 1),
        'days_held': days_held,
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 1),
        'exit_reason': 'earnings_exit',
        'cost': round(total_cost, 2),
    }]


def simulate_pre_earnings_straddle_trade(event, variant_cfg, equity, open_positions_count):
    """
    Simulate a pre-earnings straddle-like trade.
    Buy BOTH call AND put 2 days before earnings.
    Exit at market open after earnings.
    The idea: IV is elevated, any big move profits on one leg more than IV crush hurts.
    """
    entry_spot = event['pre_entry_close']
    strike = round(entry_spot, 0)  # ATM

    pre_iv = event['pre_iv']
    dte = variant_cfg['dte']
    T = dte / 252.0

    call_premium = option_price(entry_spot, strike, T, RISK_FREE_RATE, pre_iv, 'call')
    put_premium = option_price(entry_spot, strike, T, RISK_FREE_RATE, pre_iv, 'put')

    if call_premium < 0.10 or put_premium < 0.10:
        return []

    # Total cost for both legs (2 contracts, 2 entry commissions)
    total_premium = call_premium + put_premium
    contract_cost = total_premium * 100
    # Straddle uses double the position limit but counts as one "trade event"
    max_spend = min(MAX_POSITION_DOLLARS * 2, equity * MAX_POSITION_PCT * 2)
    total_cost = contract_cost + COMMISSION_PER_LEG * 2  # 2 entry legs

    if total_cost > max_spend or total_cost > equity:
        return []

    # Exit at open after earnings
    exit_spot = event['open_after']
    post_iv = event['post_iv']
    pre_to_post = event['pre_to_post_days']
    days_held = len(pre_to_post) - 1 if pre_to_post else 2
    remaining_dte = max(dte - days_held, 0)
    T_exit = remaining_dte / 252.0

    call_exit = option_price(exit_spot, strike, T_exit, RISK_FREE_RATE, post_iv, 'call')
    put_exit = option_price(exit_spot, strike, T_exit, RISK_FREE_RATE, post_iv, 'put')

    exit_value = (call_exit + put_exit) * 100
    # 2 exit legs
    pnl = exit_value - total_cost - COMMISSION_PER_LEG * 2
    pnl_pct = ((call_exit + put_exit) - total_premium) / total_premium * 100

    entry_date_str = (str(event['pre_entry_day'].date())
                      if hasattr(event['pre_entry_day'], 'date')
                      else str(event['pre_entry_day'])[:10])
    exit_date_str = (str(event['earnings_day'].date())
                     if hasattr(event['earnings_day'], 'date')
                     else str(event['earnings_day'])[:10])

    return [{
        'ticker': event['ticker'],
        'earnings_date': event['earnings_date'],
        'type': 'straddle',
        'strike': strike,
        'entry_date': entry_date_str,
        'exit_date': exit_date_str,
        'gap_pct': round(event['gap_pct'] * 100, 2),
        'pre_momentum_5d': round(event['pre_momentum_5d'] * 100, 2),
        'entry_premium': round(total_premium, 4),
        'exit_premium': round(call_exit + put_exit, 4),
        'entry_iv': round(pre_iv, 4),
        'iv_crush_pct': round(event['iv_crush_pct'] * 100, 1),
        'days_held': days_held,
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 1),
        'exit_reason': 'earnings_exit',
        'cost': round(total_cost, 2),
        'call_entry': round(call_premium, 4),
        'put_entry': round(put_premium, 4),
        'call_exit': round(call_exit, 4),
        'put_exit': round(put_exit, 4),
    }]


def run_variant(variant_key, variant_cfg, events):
    """Run a single strategy variant over all earnings events."""
    equity = STARTING_CAPITAL
    equity_curve = [equity]
    equity_dates = ['2022-01-01']
    all_trades = []
    daily_returns = []

    # Track concurrent positions by date
    open_positions = []  # list of (exit_date, cost) for concurrency tracking

    for event in events:
        # Clean up expired positions
        event_date = event['earnings_date']
        open_positions = [(ed, c) for ed, c in open_positions if ed > event_date]

        # Check position limit
        if len(open_positions) >= MAX_POSITIONS:
            continue

        # Dispatch to appropriate simulation
        entry_type = variant_cfg['entry_type']
        if entry_type == 'post_gap':
            trades = simulate_post_gap_trade(event, variant_cfg, equity, len(open_positions))
        elif entry_type == 'pre_earnings_momentum':
            trades = simulate_pre_earnings_momentum_trade(event, variant_cfg, equity,
                                                          len(open_positions))
        elif entry_type == 'pre_earnings_straddle':
            trades = simulate_pre_earnings_straddle_trade(event, variant_cfg, equity,
                                                          len(open_positions))
        else:
            continue

        for trade in trades:
            # Update equity
            equity += trade['pnl']
            all_trades.append(trade)

            # Track daily return
            prev_eq = equity_curve[-1] if equity_curve else STARTING_CAPITAL
            daily_ret = trade['pnl'] / max(prev_eq, 1.0)
            daily_returns.append(daily_ret)
            equity_curve.append(equity)
            equity_dates.append(trade['exit_date'])

            # Track concurrency
            open_positions.append((trade['exit_date'], trade['cost']))

            # Prevent going below zero
            if equity <= 0:
                equity = 0
                break

        if equity <= 0:
            break

    return {
        'equity_curve': equity_curve,
        'equity_dates': equity_dates,
        'daily_returns': daily_returns,
        'trades': all_trades,
        'final_equity': equity,
    }


# ============================================================
# ANALYSIS & VALIDATION
# ============================================================

def compute_metrics(result, variant_key, variant_cfg):
    """Compute Sharpe, Sortino, WR, PF, max DD, avg hold, avg P&L."""
    trades = result['trades']
    daily_rets = np.array(result['daily_returns'])

    metrics = {
        'variant': variant_key,
        'name': variant_cfg['name'],
        'description': variant_cfg['description'],
        'final_equity': round(result['final_equity'], 2),
        'total_return_pct': round((result['final_equity'] / STARTING_CAPITAL - 1) * 100, 2),
        'total_trades': len(trades),
    }

    if len(trades) == 0:
        metrics.update({
            'sharpe': 0.0, 'sortino': 0.0, 'win_rate': 0.0,
            'profit_factor': 0.0, 'max_drawdown_pct': 0.0,
            'avg_hold_days': 0.0, 'avg_pnl': 0.0, 'avg_pnl_pct': 0.0,
        })
        return metrics

    # Trade-level stats
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    metrics['win_rate'] = round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0
    metrics['avg_pnl'] = round(np.mean(pnls), 2)
    metrics['avg_pnl_pct'] = round(np.mean([t['pnl_pct'] for t in trades]), 1)
    metrics['avg_hold_days'] = round(np.mean([t['days_held'] for t in trades]), 1)

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    metrics['profit_factor'] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (
        999.0 if gross_profit > 0 else 0.0)

    # Daily returns stats (using trade-level returns as proxy for daily)
    if len(daily_rets) > 5:
        ann_factor = np.sqrt(len(daily_rets))  # scale by sqrt(n_trades) for trade-level Sharpe
        # Cap at sqrt(252) to avoid inflating with very few trades
        ann_factor = min(ann_factor, np.sqrt(252))
        mean_ret = np.mean(daily_rets)
        std_ret = np.std(daily_rets)
        metrics['sharpe'] = round((mean_ret / std_ret) * ann_factor, 2) if std_ret > 0 else 0.0

        downside_rets = daily_rets[daily_rets < 0]
        downside_std = np.std(downside_rets) if len(downside_rets) > 1 else std_ret
        metrics['sortino'] = round((mean_ret / downside_std) * ann_factor, 2) if downside_std > 0 else 0.0
    else:
        metrics['sharpe'] = 0.0
        metrics['sortino'] = 0.0

    # Max drawdown from equity curve
    eq = np.array(result['equity_curve'])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1.0)
    metrics['max_drawdown_pct'] = round(np.min(dd) * 100, 2)

    # Exit reason breakdown
    reasons = defaultdict(int)
    for t in trades:
        reasons[t['exit_reason']] += 1
    metrics['exit_reasons'] = dict(reasons)

    return metrics


def gap_magnitude_analysis(trades):
    """Analyze win rate by gap magnitude buckets."""
    if not trades:
        return {}

    buckets = {
        '5-10%': {'trades': [], 'label': '5-10% gap'},
        '10-20%': {'trades': [], 'label': '10-20% gap'},
        '20%+': {'trades': [], 'label': '20%+ gap'},
    }

    for t in trades:
        abs_gap = abs(t.get('gap_pct', 0))
        if abs_gap >= 20:
            buckets['20%+']['trades'].append(t)
        elif abs_gap >= 10:
            buckets['10-20%']['trades'].append(t)
        elif abs_gap >= 5:
            buckets['5-10%']['trades'].append(t)

    results = {}
    for bucket_key, bucket in buckets.items():
        bt = bucket['trades']
        if not bt:
            results[bucket_key] = {'count': 0}
            continue

        pnls = [t['pnl'] for t in bt]
        wins = [p for p in pnls if p > 0]
        results[bucket_key] = {
            'count': len(bt),
            'win_rate': round(len(wins) / len(pnls) * 100, 1),
            'avg_pnl': round(np.mean(pnls), 2),
            'total_pnl': round(sum(pnls), 2),
            'avg_gap': round(np.mean([abs(t['gap_pct']) for t in bt]), 1),
        }

    return results


def permutation_test(result, n_perms=N_PERMUTATIONS):
    """Shuffle trade P&Ls to test if the strategy has real edge."""
    trades = result['trades']
    if len(trades) < 5:
        return {'p_value': 1.0, 'actual_mean_pnl': 0.0, 'perm_mean_pnl_mean': 0.0}

    pnls = np.array([t['pnl'] for t in trades])
    actual_mean = np.mean(pnls)

    rng = np.random.RandomState(42)
    perm_means = []
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(pnls))
        perm_means.append(np.mean(pnls * signs))

    perm_means = np.array(perm_means)
    p_value = np.mean(perm_means >= actual_mean)

    return {
        'p_value': round(p_value, 4),
        'actual_mean_pnl': round(actual_mean, 2),
        'perm_mean_pnl_mean': round(np.mean(perm_means), 2),
        'perm_mean_pnl_std': round(np.std(perm_means), 2),
    }


def top_movers_analysis(trades, n_top=10):
    """Find the best and worst individual trades."""
    if not trades:
        return {'best': [], 'worst': []}

    sorted_by_pnl = sorted(trades, key=lambda t: t['pnl'], reverse=True)
    return {
        'best': sorted_by_pnl[:n_top],
        'worst': sorted_by_pnl[-n_top:],
    }


def ticker_performance(trades):
    """Performance breakdown by ticker."""
    if not trades:
        return {}

    by_ticker = defaultdict(list)
    for t in trades:
        by_ticker[t['ticker']].append(t)

    results = {}
    for ticker, ticker_trades in sorted(by_ticker.items()):
        pnls = [t['pnl'] for t in ticker_trades]
        wins = [p for p in pnls if p > 0]
        results[ticker] = {
            'count': len(ticker_trades),
            'win_rate': round(len(wins) / len(pnls) * 100, 1),
            'avg_pnl': round(np.mean(pnls), 2),
            'total_pnl': round(sum(pnls), 2),
            'avg_gap': round(np.mean([abs(t.get('gap_pct', 0)) for t in ticker_trades]), 1),
        }

    return results


# ============================================================
# REPORTING
# ============================================================

def print_report(all_metrics, all_gap_analysis, all_perm_tests, all_ticker_perf):
    """Print formatted report to console."""
    print("\n" + "=" * 90)
    print("EARNINGS MOMENTUM OPTIONS STRATEGY v1 — BACKTEST RESULTS")
    print("=" * 90)
    print(f"Universe: {len(STOCK_UNIVERSE)} stocks | Period: {START_DATE} to {END_DATE}")
    print(f"Account: ${STARTING_CAPITAL} | Max/position: ${MAX_POSITION_DOLLARS} | "
          f"Commission: ${COMMISSION_RT}/RT")
    print(f"Exits: +30% TP, -25% SL, 50% trailing giveback, 3-day time stop")
    print(f"IV model: pre-earnings = max(VIX*1.5, RV*2.0) | "
          f"post-earnings crush = {IV_CRUSH_MIN*100:.0f}-{IV_CRUSH_MAX*100:.0f}%")
    print("=" * 90)

    # Summary table
    print(f"\n{'Var':>3} {'Name':<30} {'Trades':>6} {'WR%':>6} {'Sharpe':>7} "
          f"{'Sortino':>8} {'PF':>6} {'MaxDD%':>7} {'AvgPnL':>8} {'Final$':>8}")
    print("-" * 90)

    for m in all_metrics:
        print(f"{m['variant']:>3} {m['name']:<30} {m['total_trades']:>6} "
              f"{m['win_rate']:>5.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['profit_factor']:>6.2f} {m['max_drawdown_pct']:>6.1f}% "
              f"${m['avg_pnl']:>7.2f} ${m['final_equity']:>7.2f}")

    # Gap magnitude analysis for each variant
    print("\n" + "=" * 90)
    print("GAP MAGNITUDE ANALYSIS")
    print("=" * 90)

    for m in all_metrics:
        vk = m['variant']
        ga = all_gap_analysis.get(vk, {})
        if not ga:
            continue

        print(f"\n  Variant {vk} ({m['name']}):")
        for bucket, stats in ga.items():
            if stats.get('count', 0) == 0:
                print(f"    {bucket:>8}: no trades")
            else:
                print(f"    {bucket:>8}: {stats['count']:>3} trades, "
                      f"WR={stats['win_rate']:.1f}%, "
                      f"avg P&L=${stats['avg_pnl']:.2f}, "
                      f"total=${stats['total_pnl']:.2f}")

    # Permutation tests
    print("\n" + "=" * 90)
    print("PERMUTATION TESTS (100 shuffles, sign-randomization)")
    print("=" * 90)

    for m in all_metrics:
        vk = m['variant']
        pt = all_perm_tests.get(vk, {})
        p_val = pt.get('p_value', 1.0)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        print(f"  {vk} ({m['name']:<30}): p={p_val:.4f} {sig}  "
              f"actual_mean=${pt.get('actual_mean_pnl', 0):.2f}  "
              f"perm_mean=${pt.get('perm_mean_pnl_mean', 0):.2f}")

    # Top tickers by variant (show best variant only)
    best_variant = max(all_metrics, key=lambda m: m.get('sharpe', 0))
    bv_key = best_variant['variant']
    bv_ticker_perf = all_ticker_perf.get(bv_key, {})

    if bv_ticker_perf:
        print(f"\n{'=' * 90}")
        print(f"TICKER BREAKDOWN — Best Variant {bv_key} ({best_variant['name']})")
        print(f"{'=' * 90}")
        # Sort by total P&L
        sorted_tickers = sorted(bv_ticker_perf.items(), key=lambda x: x[1]['total_pnl'],
                                reverse=True)
        print(f"  {'Ticker':<8} {'Trades':>6} {'WR%':>6} {'AvgPnL':>8} {'TotalPnL':>10} "
              f"{'AvgGap%':>8}")
        print(f"  {'-' * 50}")
        for ticker, stats in sorted_tickers[:15]:
            print(f"  {ticker:<8} {stats['count']:>6} {stats['win_rate']:>5.1f}% "
                  f"${stats['avg_pnl']:>7.2f} ${stats['total_pnl']:>9.2f} "
                  f"{stats['avg_gap']:>7.1f}%")

    # Exit reason breakdown for best variant
    if best_variant.get('exit_reasons'):
        print(f"\n  Exit reasons ({bv_key}):", best_variant['exit_reasons'])

    print(f"\n{'=' * 90}")
    print("CONCLUSION")
    print(f"{'=' * 90}")

    profitable = [m for m in all_metrics if m.get('sharpe', 0) > 0.5 and m['total_trades'] >= 10]
    if profitable:
        best = max(profitable, key=lambda m: m['sharpe'])
        print(f"  Best variant: {best['variant']} ({best['name']})")
        print(f"  Sharpe={best['sharpe']:.2f}, Sortino={best['sortino']:.2f}, "
              f"WR={best['win_rate']:.1f}%, PF={best['profit_factor']:.2f}")
        print(f"  {best['total_trades']} trades, avg hold={best['avg_hold_days']:.1f}d, "
              f"MaxDD={best['max_drawdown_pct']:.1f}%")
        pt = all_perm_tests.get(best['variant'], {})
        if pt.get('p_value', 1) < 0.05:
            print(f"  Permutation test: SIGNIFICANT (p={pt['p_value']:.4f})")
        else:
            print(f"  Permutation test: NOT significant (p={pt.get('p_value', 1):.4f})")
    else:
        print("  No variant achieved Sharpe > 0.5 with 10+ trades.")
        print("  Earnings momentum options may not have reliable single-leg edge,")
        print("  or the IV crush / commission drag overwhelms the directional move.")


# ============================================================
# MLFLOW LOGGING
# ============================================================

def log_to_mlflow(all_metrics, all_perm_tests, all_gap_analysis):
    """Log results to MLflow."""
    if not MLFLOW_AVAILABLE:
        print("\nMLflow not available, skipping logging")
        return

    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("earnings_momentum_options")
    except Exception as e:
        print(f"\nMLflow setup failed: {e}")
        return

    for m in all_metrics:
        vk = m['variant']
        try:
            with mlflow.start_run(run_name=f"variant_{vk}_{m['name'].replace(' ', '_')}"):
                # Parameters
                variant_cfg = VARIANTS[vk]
                mlflow.log_param("variant", vk)
                mlflow.log_param("variant_name", m['name'])
                mlflow.log_param("entry_type", variant_cfg['entry_type'])
                mlflow.log_param("gap_threshold", variant_cfg['gap_threshold'])
                mlflow.log_param("gap_direction_mode", variant_cfg['gap_direction_mode'])
                mlflow.log_param("require_momentum_alignment",
                                 variant_cfg['require_momentum_alignment'])
                mlflow.log_param("tp_pct", variant_cfg['tp_pct'])
                mlflow.log_param("sl_pct", variant_cfg['sl_pct'])
                mlflow.log_param("time_stop_days", variant_cfg['time_stop_days'])
                mlflow.log_param("universe_size", len(STOCK_UNIVERSE))
                mlflow.log_param("start_date", START_DATE)
                mlflow.log_param("end_date", END_DATE)
                mlflow.log_param("starting_capital", STARTING_CAPITAL)
                mlflow.log_param("commission_rt", COMMISSION_RT)

                # Metrics
                mlflow.log_metric("sharpe", m.get('sharpe', 0))
                mlflow.log_metric("sortino", m.get('sortino', 0))
                mlflow.log_metric("win_rate", m.get('win_rate', 0))
                mlflow.log_metric("profit_factor", m.get('profit_factor', 0))
                mlflow.log_metric("max_drawdown_pct", m.get('max_drawdown_pct', 0))
                mlflow.log_metric("total_trades", m.get('total_trades', 0))
                mlflow.log_metric("avg_pnl", m.get('avg_pnl', 0))
                mlflow.log_metric("avg_hold_days", m.get('avg_hold_days', 0))
                mlflow.log_metric("final_equity", m.get('final_equity', 0))
                mlflow.log_metric("total_return_pct", m.get('total_return_pct', 0))

                # Permutation test
                pt = all_perm_tests.get(vk, {})
                mlflow.log_metric("perm_p_value", pt.get('p_value', 1.0))

                # Gap analysis as tags
                ga = all_gap_analysis.get(vk, {})
                for bucket, stats in ga.items():
                    if stats.get('count', 0) > 0:
                        clean_bucket = bucket.replace('%', 'pct').replace('+', 'plus')
                        mlflow.log_metric(f"gap_{clean_bucket}_count", stats['count'])
                        mlflow.log_metric(f"gap_{clean_bucket}_wr", stats.get('win_rate', 0))

            print(f"  Logged variant {vk} to MLflow")
        except Exception as e:
            print(f"  Failed to log variant {vk}: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 90)
    print("EARNINGS MOMENTUM OPTIONS STRATEGY v1")
    print("=" * 90)
    print(f"Universe: {len(STOCK_UNIVERSE)} stocks")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Account: ${STARTING_CAPITAL}, max ${MAX_POSITION_DOLLARS}/position")
    print()

    # --- Load data ---
    prices_df, earnings_dates = load_data()

    # Extract SPY and VIX
    spy_prices = get_ticker_prices(prices_df, 'SPY')
    vix_data = get_ticker_prices(prices_df, '^VIX')

    if spy_prices is None or vix_data is None:
        print("ERROR: Could not load SPY or VIX data")
        return

    # --- Build earnings events ---
    events = build_earnings_events(prices_df, earnings_dates, vix_data)
    if not events:
        print("ERROR: No earnings events found. Check earnings date data.")
        return

    # --- Run all variants ---
    all_metrics = []
    all_results = {}
    all_gap_analysis = {}
    all_perm_tests = {}
    all_ticker_perf = {}

    for vk, vcfg in VARIANTS.items():
        print(f"\nRunning variant {vk}: {vcfg['name']}...")
        result = run_variant(vk, vcfg, events)
        all_results[vk] = result

        metrics = compute_metrics(result, vk, vcfg)
        all_metrics.append(metrics)
        print(f"  {metrics['total_trades']} trades, WR={metrics['win_rate']:.1f}%, "
              f"Sharpe={metrics['sharpe']:.2f}, Final=${metrics['final_equity']:.2f}")

        # Gap magnitude analysis
        all_gap_analysis[vk] = gap_magnitude_analysis(result['trades'])

        # Permutation test
        all_perm_tests[vk] = permutation_test(result)

        # Ticker performance
        all_ticker_perf[vk] = ticker_performance(result['trades'])

    # --- Print report ---
    print_report(all_metrics, all_gap_analysis, all_perm_tests, all_ticker_perf)

    # --- Save results ---
    output = {
        'metadata': {
            'strategy': 'earnings_momentum_options_v1',
            'universe_size': len(STOCK_UNIVERSE),
            'period': f"{START_DATE} to {END_DATE}",
            'starting_capital': STARTING_CAPITAL,
            'commission_rt': COMMISSION_RT,
            'iv_model': 'pre=max(VIX*1.5, RV*2.0), post=crush 30-50%',
            'timestamp': datetime.now().isoformat(),
        },
        'variant_metrics': all_metrics,
        'gap_analysis': {k: v for k, v in all_gap_analysis.items()},
        'permutation_tests': all_perm_tests,
        'ticker_performance': {
            vk: dict(sorted(tp.items(), key=lambda x: x[1]['total_pnl'], reverse=True)[:20])
            for vk, tp in all_ticker_perf.items()
        },
    }

    # Save summary JSON
    summary_path = os.path.join(OUTPUT_DIR, 'earnings_momentum_results.json')
    with open(summary_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {summary_path}")

    # Save trade logs per variant
    for vk, result in all_results.items():
        trades_path = os.path.join(OUTPUT_DIR, f'trades_variant_{vk}.json')
        with open(trades_path, 'w') as f:
            json.dump(result['trades'], f, indent=2, default=str)

    # Save equity curves
    eq_curves = {}
    for vk, result in all_results.items():
        eq_curves[vk] = {
            'dates': [str(d) for d in result['equity_dates']],
            'equity': [round(e, 2) for e in result['equity_curve']],
        }
    eq_path = os.path.join(OUTPUT_DIR, 'equity_curves.json')
    with open(eq_path, 'w') as f:
        json.dump(eq_curves, f, indent=2)

    # --- MLflow ---
    log_to_mlflow(all_metrics, all_perm_tests, all_gap_analysis)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
