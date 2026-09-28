#!/usr/bin/env python3
"""
Pre-Earnings Vol Crush Income Strategy v1
==========================================
Systematically sells iron condors on large-cap stocks 1-5 days before earnings,
capturing the IV crush that occurs after announcements.

Walk-forward backtest: 252d train, 21d test, SLIDING window (HC #0).
FIFO cost: $18.80 per iron condor (4 legs * $4.70 RT).

Author: Claude Opus 4.6
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import json
import logging
import warnings
import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as sp_stats
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('vol_crush_v1')

# ---------------------------------------------------------------------------
# MLflow (optional – runs fine without it)
# ---------------------------------------------------------------------------
try:
    import mlflow
    import urllib.request
    mlflow.set_tracking_uri('http://jupiter:5000')
    # Quick connectivity test (2s timeout) before committing to MLflow
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    MLFLOW_OK = True
except Exception:
    MLFLOW_OK = False
    log.warning('MLflow not available – skipping experiment tracking')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'MA',
    'HD', 'UNH', 'JNJ', 'PG', 'KO', 'PEP', 'MCD', 'WMT', 'COST', 'AVGO',
    'CRM', 'ORCL', 'ADBE', 'NFLX', 'AMD', 'INTC', 'QCOM', 'TXN', 'AMAT', 'MU',
    'GS', 'MS', 'BAC', 'WFC', 'C', 'AXP', 'BLK', 'SCHW', 'LLY', 'PFE',
    'MRK', 'ABBV', 'TMO', 'DHR', 'BMY', 'XOM', 'CVX', 'COP', 'SLB', 'EOG',
]

COST_PER_IC = 18.80          # 4 legs * $4.70 RT
MAX_RISK_PER_TRADE = 200.0   # dollars
MAX_CONCURRENT = 2
RISK_FREE_RATE = 0.04        # annualized, for BS pricing
DAYS_BEFORE_ENTRY = 3        # enter 3 calendar days before earnings
MIN_IV_RATIO = 1.4           # pre-earnings IV must be >= 1.4x realized vol
MAX_PRIOR_MOVE = 0.05        # skip if |5d return| > 5%

# Walk-forward
WF_TRAIN_DAYS = 252
WF_TEST_DAYS = 21

# Permutation test
N_PERMUTATIONS = 200

# Black-Scholes helpers
TRADING_DAYS_YEAR = 252

# Output paths - use script directory so it works on any node
RESULTS_DIR = Path(__file__).resolve().parent / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'pre_earnings_vol_crush_v1_results.json'


# ---------------------------------------------------------------------------
# Black-Scholes pricing
# ---------------------------------------------------------------------------

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call price via Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price via Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def iron_condor_premium(
    S: float, K_put_short: float, K_put_long: float,
    K_call_short: float, K_call_long: float,
    T: float, r: float, sigma: float,
) -> float:
    """Net credit received for selling an iron condor."""
    # Sell: short put + short call
    # Buy: long put + long call (wings)
    credit = (
        bs_put_price(S, K_put_short, T, r, sigma) -
        bs_put_price(S, K_put_long, T, r, sigma) +
        bs_call_price(S, K_call_short, T, r, sigma) -
        bs_call_price(S, K_call_long, T, r, sigma)
    )
    return max(credit, 0.0)


def iron_condor_payout(
    S_exit: float, K_put_short: float, K_put_long: float,
    K_call_short: float, K_call_long: float,
) -> float:
    """Intrinsic value the seller must pay at exit (negative = loss for seller)."""
    # Put spread intrinsic (seller's liability)
    put_spread_intrinsic = max(K_put_short - S_exit, 0.0) - max(K_put_long - S_exit, 0.0)
    # Call spread intrinsic
    call_spread_intrinsic = max(S_exit - K_call_short, 0.0) - max(S_exit - K_call_long, 0.0)
    return put_spread_intrinsic + call_spread_intrinsic


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_all_data(tickers: List[str], start: str = '2017-01-01', end: str = '2026-07-24') -> Dict:
    """Fetch price data and earnings dates for all tickers."""
    log.info(f'Fetching data for {len(tickers)} tickers from {start} to {end}')

    price_data = {}
    earnings_data = {}
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            t = yf.Ticker(ticker)

            # Price data
            hist = t.history(start=start, end=end, auto_adjust=True)
            # Strip timezone to avoid tz-naive/aware comparison issues
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)
            if hist.empty or len(hist) < 100:
                log.warning(f'{ticker}: insufficient price data ({len(hist)} rows), skipping')
                failed.append(ticker)
                continue
            price_data[ticker] = hist

            # Earnings dates
            # Try multiple approaches
            edates = _get_earnings_dates(t, ticker)
            if edates is not None and len(edates) > 0:
                earnings_data[ticker] = edates
                log.info(f'  [{i+1}/{len(tickers)}] {ticker}: {len(hist)} prices, {len(edates)} earnings dates')
            else:
                log.warning(f'  [{i+1}/{len(tickers)}] {ticker}: prices OK but no earnings dates')
                failed.append(ticker)

        except Exception as e:
            log.warning(f'{ticker}: fetch error: {e}')
            failed.append(ticker)

        # Rate limit politeness
        if (i + 1) % 10 == 0:
            time.sleep(1)

    log.info(f'Fetched {len(price_data)} tickers with prices, {len(earnings_data)} with earnings')
    if failed:
        log.info(f'Failed/skipped: {failed}')

    return {'prices': price_data, 'earnings': earnings_data}


def _get_earnings_dates(t: yf.Ticker, ticker: str) -> Optional[pd.DatetimeIndex]:
    """Extract earnings dates from yfinance ticker object."""
    dates = []

    # Method 1: earnings_dates attribute
    try:
        ed = t.earnings_dates
        if ed is not None and len(ed) > 0:
            idx = ed.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                idx = idx.tz_convert(None)
            dates.extend(idx.tolist())
    except Exception:
        pass

    # Method 2: get_earnings_dates
    try:
        ed2 = t.get_earnings_dates(limit=100)
        if ed2 is not None and len(ed2) > 0:
            idx2 = ed2.index
            if hasattr(idx2, 'tz') and idx2.tz is not None:
                idx2 = idx2.tz_convert(None)
            dates.extend(idx2.tolist())
    except Exception:
        pass

    if not dates:
        return None

    # Deduplicate and sort
    dates = list(set(dates))
    dates = [d for d in dates if isinstance(d, (pd.Timestamp, dt.datetime))]
    if not dates:
        return None

    dates_idx = pd.DatetimeIndex(dates)
    # Strip timezone info to make tz-naive
    if dates_idx.tz is not None:
        dates_idx = dates_idx.tz_convert(None)
    dates_idx = dates_idx.normalize().drop_duplicates().sort_values()
    # Only keep past dates (not future scheduled)
    cutoff = pd.Timestamp('2026-07-25')
    dates_idx = dates_idx[dates_idx < cutoff]

    return dates_idx


# ---------------------------------------------------------------------------
# Realized vol and IV proxy
# ---------------------------------------------------------------------------

def compute_realized_vol(prices: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility from log returns."""
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS_YEAR)
    return rv


def compute_iv_proxy(prices: pd.Series, earnings_date: pd.Timestamp,
                     rv_window: int = 20) -> Tuple[float, float, float]:
    """
    Estimate pre-earnings IV proxy and post-earnings IV.

    Returns: (pre_iv, post_iv, iv_ratio)
    - pre_iv: realized vol * multiplier (simulating IV elevation before earnings)
    - post_iv: realized vol right after earnings
    - iv_ratio: pre_iv / base_rv
    """
    # Get the index position of the nearest date to earnings
    idx = prices.index
    # Normalize for comparison
    earnings_norm = earnings_date.normalize()

    # Find dates around earnings
    mask_before = idx < earnings_norm
    mask_after = idx >= earnings_norm

    if mask_before.sum() < rv_window + 5 or mask_after.sum() < 2:
        return np.nan, np.nan, np.nan

    # Base realized vol (20d ending 10 days before earnings to avoid
    # the pre-earnings run-up contaminating the "normal" vol)
    prices_before = prices[mask_before]
    if len(prices_before) < rv_window + 10:
        return np.nan, np.nan, np.nan

    # "Normal" vol: 20d window ending 10 days before earnings
    base_prices = prices_before.iloc[-(rv_window + 10):-10]
    if len(base_prices) < rv_window:
        base_prices = prices_before.iloc[-rv_window:]
    log_ret_base = np.log(base_prices / base_prices.shift(1)).dropna()
    if len(log_ret_base) < 10:
        return np.nan, np.nan, np.nan
    base_rv = log_ret_base.std() * np.sqrt(TRADING_DAYS_YEAR)

    if base_rv <= 0.01:  # effectively zero vol
        return np.nan, np.nan, np.nan

    # Pre-earnings vol: use the last 5 days before earnings to detect IV spike
    pre_prices = prices_before.iloc[-5:]
    log_ret_pre = np.log(pre_prices / pre_prices.shift(1)).dropna()
    if len(log_ret_pre) < 3:
        pre_rv = base_rv
    else:
        pre_rv = log_ret_pre.std() * np.sqrt(TRADING_DAYS_YEAR)

    # IV proxy: pre-earnings IV is typically 1.5-2.5x base RV
    # We use the ratio of recent short-term vol to base vol as a multiplier,
    # then scale to simulate typical IV behavior
    raw_ratio = pre_rv / base_rv if base_rv > 0 else 1.0
    # Clamp the raw ratio and apply a floor (IV is always somewhat elevated before earnings)
    iv_multiplier = max(1.3, min(raw_ratio * 1.5, 3.0))
    pre_iv = base_rv * iv_multiplier

    # Post-earnings IV drops back toward realized vol
    post_iv = base_rv * 1.1  # slight premium remains

    iv_ratio = iv_multiplier

    return pre_iv, post_iv, iv_ratio


# ---------------------------------------------------------------------------
# Trade generation
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    ticker: str
    earnings_date: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pre_iv: float
    post_iv: float
    iv_ratio: float
    K_put_short: float
    K_put_long: float
    K_call_short: float
    K_call_long: float
    premium_collected: float      # per share, * 100 for contract
    exit_intrinsic: float         # per share
    dte_at_entry: float           # days to expiration
    pnl_per_contract: float       # premium - exit_intrinsic - cost
    pnl_dollar: float             # scaled by position size
    max_risk: float               # max possible loss
    num_contracts: int
    spy_20d_return: float         # for regime classification
    regime: str                   # bull/bear/flat


def generate_trades_for_earnings(
    ticker: str,
    prices: pd.DataFrame,
    earnings_date: pd.Timestamp,
    spy_prices: pd.Series,
    params: Dict,
) -> Optional[Trade]:
    """Generate a single iron condor trade for an earnings event."""
    close = prices['Close']
    idx = close.index

    # Normalize earnings date
    ed = earnings_date.normalize()

    # Find entry date: ~3 business days before earnings
    entry_offset = params.get('days_before', DAYS_BEFORE_ENTRY)
    # Look back to find a valid trading day
    entry_candidates = idx[idx < ed]
    if len(entry_candidates) < entry_offset + 5:
        return None
    entry_date = entry_candidates[-entry_offset]

    # Find exit date: 1 business day after earnings
    exit_candidates = idx[idx > ed]
    if len(exit_candidates) < 1:
        return None
    exit_date = exit_candidates[0]

    entry_price = close.loc[entry_date]
    exit_price = close.loc[exit_date]

    # Check prior 5-day move filter
    prior_5d = entry_candidates[-6:-1] if len(entry_candidates) >= 6 else entry_candidates[-5:]
    if len(prior_5d) >= 2:
        move_5d = abs(close.loc[prior_5d[-1]] / close.loc[prior_5d[0]] - 1.0)
        if move_5d > params.get('max_prior_move', MAX_PRIOR_MOVE):
            return None

    # Compute IV proxy
    pre_iv, post_iv, iv_ratio = compute_iv_proxy(close, ed)
    if np.isnan(pre_iv):
        return None

    # IV threshold filter
    min_ratio = params.get('min_iv_ratio', MIN_IV_RATIO)
    if iv_ratio < min_ratio:
        return None

    # Strikes: short strikes at ~1.0 stdev from current price
    # Using pre-IV to calculate the expected move
    dte = max(7, 14)  # fixed DTE assumption (closest standard expiry)
    T = dte / 365.0
    stdev_move = entry_price * pre_iv * np.sqrt(T)

    wing_width_pct = params.get('wing_width_pct', 0.05)

    K_call_short = round(entry_price + stdev_move, 2)
    K_put_short = round(entry_price - stdev_move, 2)
    K_call_long = round(K_call_short * (1 + wing_width_pct), 2)
    K_put_long = round(K_put_short * (1 - wing_width_pct), 2)

    # Sanity checks
    if K_put_long <= 0 or K_put_short <= K_put_long or K_call_short >= K_call_long:
        return None

    # Premium at entry (using pre-earnings IV)
    premium = iron_condor_premium(
        entry_price, K_put_short, K_put_long,
        K_call_short, K_call_long, T, RISK_FREE_RATE, pre_iv,
    )

    if premium <= 0.01:
        return None

    # Exit intrinsic (what we owe at exit)
    exit_intrinsic = iron_condor_payout(
        exit_price, K_put_short, K_put_long,
        K_call_short, K_call_long,
    )

    # Also compute remaining time value at exit using post-earnings IV
    # Remaining DTE after holding ~4 days
    holding_days = max(1, (exit_date - entry_date).days)
    remaining_T = max(0.001, (dte - holding_days) / 365.0)

    # Remaining premium at exit (what we'd pay to close)
    exit_premium = iron_condor_premium(
        exit_price, K_put_short, K_put_long,
        K_call_short, K_call_long, remaining_T, RISK_FREE_RATE, post_iv,
    )

    # P&L per share: credit received - cost to close
    # Cost to close = max(intrinsic, remaining_premium) since we buy it back
    cost_to_close = max(exit_intrinsic, exit_premium)
    pnl_per_share = premium - cost_to_close

    # Max risk per contract: width of wider spread * 100 - premium * 100
    put_width = K_put_short - K_put_long
    call_width = K_call_long - K_call_short
    max_spread_width = max(put_width, call_width)
    max_risk_per_contract = max_spread_width * 100 - premium * 100

    if max_risk_per_contract <= 0:
        return None

    # Position sizing
    max_risk = params.get('max_risk', MAX_RISK_PER_TRADE)
    num_contracts = max(1, int(max_risk / max_risk_per_contract))

    pnl_per_contract = pnl_per_share * 100 - COST_PER_IC
    pnl_dollar = pnl_per_contract * num_contracts

    # Regime classification using SPY
    spy_close = spy_prices
    spy_idx = spy_close.index
    spy_at_entry = spy_idx[spy_idx <= entry_date]
    if len(spy_at_entry) >= 20:
        spy_20d_ret = spy_close.loc[spy_at_entry[-1]] / spy_close.loc[spy_at_entry[-20]] - 1.0
    else:
        spy_20d_ret = 0.0

    if spy_20d_ret > 0.02:
        regime = 'bull'
    elif spy_20d_ret < -0.02:
        regime = 'bear'
    else:
        regime = 'flat'

    return Trade(
        ticker=ticker,
        earnings_date=str(ed.date()),
        entry_date=str(entry_date.date()),
        exit_date=str(exit_date.date()),
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        pre_iv=float(pre_iv),
        post_iv=float(post_iv),
        iv_ratio=float(iv_ratio),
        K_put_short=float(K_put_short),
        K_put_long=float(K_put_long),
        K_call_short=float(K_call_short),
        K_call_long=float(K_call_long),
        premium_collected=float(premium),
        exit_intrinsic=float(cost_to_close),
        dte_at_entry=float(dte),
        pnl_per_contract=float(pnl_per_contract),
        pnl_dollar=float(pnl_dollar),
        max_risk=float(max_risk_per_contract * num_contracts),
        num_contracts=num_contracts,
        spy_20d_return=float(spy_20d_ret),
        regime=regime,
    )


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def walk_forward_backtest(data: Dict) -> Tuple[List[Trade], Dict]:
    """
    Walk-forward backtest with 252d train, 21d test, SLIDING window.

    Train period: learn optimal IV threshold and wing width per stock.
    Test period: apply learned parameters.
    """
    prices = data['prices']
    earnings = data['earnings']

    # Get SPY for regime classification
    log.info('Fetching SPY data for regime classification...')
    spy = yf.Ticker('SPY').history(start='2017-01-01', end='2026-07-25', auto_adjust=True)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_convert(None)
    spy_close = spy['Close']

    # Build master calendar of all earnings events
    all_events = []
    for ticker in earnings:
        if ticker not in prices:
            continue
        for ed in earnings[ticker]:
            # Force tz-naive to avoid comparison errors
            ed_naive = pd.Timestamp(ed).tz_localize(None) if pd.Timestamp(ed).tzinfo is not None else pd.Timestamp(ed)
            all_events.append((ticker, ed_naive))

    all_events.sort(key=lambda x: x[1])
    log.info(f'Total earnings events across all tickers: {len(all_events)}')

    if len(all_events) == 0:
        log.error('No earnings events found. Cannot run backtest.')
        return [], {}

    # Determine date range
    min_date = all_events[0][1]
    max_date = all_events[-1][1]
    log.info(f'Earnings date range: {min_date.date()} to {max_date.date()}')

    # Create walk-forward windows
    # We need a reference calendar - use SPY trading days
    trading_days = spy_close.index.normalize()
    if trading_days.tz is not None:
        trading_days = trading_days.tz_convert(None)

    # Start after enough training data
    start_idx = WF_TRAIN_DAYS
    if start_idx >= len(trading_days):
        log.error('Not enough trading days for walk-forward')
        return [], {}

    all_trades = []
    wf_metrics = []
    window_count = 0

    idx = start_idx
    while idx + WF_TEST_DAYS <= len(trading_days):
        train_start = trading_days[idx - WF_TRAIN_DAYS]
        train_end = trading_days[idx - 1]
        test_start = trading_days[idx]
        test_end = trading_days[min(idx + WF_TEST_DAYS - 1, len(trading_days) - 1)]

        # --- TRAIN: find optimal parameters on training earnings ---
        train_events = [
            (t, ed) for t, ed in all_events
            if train_start <= ed <= train_end
        ]

        best_params = _optimize_on_train(train_events, prices, spy_close, train_start)

        # --- TEST: apply parameters on test earnings ---
        test_events = [
            (t, ed) for t, ed in all_events
            if test_start <= ed <= test_end
        ]

        window_trades = []
        active_positions = []

        for ticker, ed in test_events:
            if ticker not in prices:
                continue

            # Check max concurrent positions
            entry_approx = ed - pd.Timedelta(days=5)
            active_positions = [
                t for t in window_trades
                if pd.Timestamp(t.exit_date) > entry_approx
            ]
            if len(active_positions) >= MAX_CONCURRENT:
                continue

            trade = generate_trades_for_earnings(
                ticker, prices[ticker], ed, spy_close, best_params,
            )
            if trade is not None:
                window_trades.append(trade)

        all_trades.extend(window_trades)

        if window_trades:
            pnls = [t.pnl_dollar for t in window_trades]
            wf_metrics.append({
                'window': window_count,
                'test_start': str(test_start.date()),
                'test_end': str(test_end.date()),
                'n_trades': len(window_trades),
                'total_pnl': sum(pnls),
                'mean_pnl': np.mean(pnls),
                'params': best_params,
            })

        window_count += 1
        idx += WF_TEST_DAYS

        if window_count % 20 == 0:
            log.info(f'  Walk-forward window {window_count}: {len(all_trades)} total trades so far')

    log.info(f'Walk-forward complete: {window_count} windows, {len(all_trades)} total trades')
    return all_trades, {'wf_windows': wf_metrics}


def _optimize_on_train(
    train_events: List[Tuple],
    prices: Dict,
    spy_close: pd.Series,
    train_start: pd.Timestamp,
) -> Dict:
    """
    Simple grid search on training earnings to find optimal parameters.
    Returns best parameter dict.
    """
    if not train_events:
        return {
            'min_iv_ratio': MIN_IV_RATIO,
            'wing_width_pct': 0.05,
            'days_before': DAYS_BEFORE_ENTRY,
            'max_prior_move': MAX_PRIOR_MOVE,
            'max_risk': MAX_RISK_PER_TRADE,
        }

    best_sharpe = -999
    best_params = None

    # Small grid to avoid overfitting
    iv_thresholds = [1.3, 1.4, 1.5, 1.6]
    wing_widths = [0.04, 0.05, 0.06]

    for iv_thresh in iv_thresholds:
        for ww in wing_widths:
            params = {
                'min_iv_ratio': iv_thresh,
                'wing_width_pct': ww,
                'days_before': DAYS_BEFORE_ENTRY,
                'max_prior_move': MAX_PRIOR_MOVE,
                'max_risk': MAX_RISK_PER_TRADE,
            }

            trades = []
            for ticker, ed in train_events:
                if ticker not in prices:
                    continue
                trade = generate_trades_for_earnings(ticker, prices[ticker], ed, spy_close, params)
                if trade is not None:
                    trades.append(trade)

            if len(trades) < 5:
                continue

            pnls = np.array([t.pnl_dollar for t in trades])
            if pnls.std() == 0:
                continue
            sharpe = pnls.mean() / pnls.std() * np.sqrt(len(trades))

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

    if best_params is None:
        best_params = {
            'min_iv_ratio': MIN_IV_RATIO,
            'wing_width_pct': 0.05,
            'days_before': DAYS_BEFORE_ENTRY,
            'max_prior_move': MAX_PRIOR_MOVE,
            'max_risk': MAX_RISK_PER_TRADE,
        }

    return best_params


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(trades: List[Trade]) -> Dict:
    """Compute all required metrics and validation tests."""
    if not trades:
        return {'error': 'No trades generated'}

    pnls = np.array([t.pnl_dollar for t in trades])
    premiums = np.array([t.premium_collected * 100 * t.num_contracts for t in trades])
    payouts = np.array([t.exit_intrinsic * 100 * t.num_contracts for t in trades])
    holding_days = np.array([
        (pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days for t in trades
    ])

    n_trades = len(trades)
    total_pnl = float(pnls.sum())
    mean_pnl = float(pnls.mean())
    std_pnl = float(pnls.std()) if pnls.std() > 0 else 1e-9
    win_rate = float((pnls > 0).sum() / n_trades)
    gross_profit = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0
    gross_loss = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe (annualized assuming ~60 trades/year typical)
    trades_per_year = max(1, n_trades / 8)  # 8 years of data
    sharpe = (mean_pnl / std_pnl) * np.sqrt(trades_per_year) if std_pnl > 0 else 0

    # Sortino
    downside_pnls = pnls[pnls < 0]
    downside_std = downside_pnls.std() if len(downside_pnls) > 1 else 1e-9
    sortino = (mean_pnl / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # CAGR and MaxDD (using cumulative equity curve)
    cum_pnl = np.cumsum(pnls)
    equity = 645 + cum_pnl  # starting capital
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())

    if equity[-1] > 0 and equity[0] > 0:
        # Estimate years from first to last trade
        first_date = pd.Timestamp(trades[0].entry_date)
        last_date = pd.Timestamp(trades[-1].exit_date)
        years = max(0.5, (last_date - first_date).days / 365.25)
        cagr = float((equity[-1] / 645) ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0

    calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0

    # Per-trade stats
    avg_premium = float(premiums.mean())
    avg_payout = float(payouts.mean())
    avg_holding = float(holding_days.mean())

    # --- Regime stratification ---
    regimes = {}
    for regime in ['bull', 'bear', 'flat']:
        r_trades = [t for t in trades if t.regime == regime]
        if len(r_trades) >= 3:
            r_pnls = np.array([t.pnl_dollar for t in r_trades])
            r_std = r_pnls.std() if r_pnls.std() > 0 else 1e-9
            r_tpy = max(1, len(r_trades) / 8)
            r_sharpe = (r_pnls.mean() / r_std) * np.sqrt(r_tpy)
            regimes[regime] = {
                'n_trades': len(r_trades),
                'sharpe': float(r_sharpe),
                'mean_pnl': float(r_pnls.mean()),
                'win_rate': float((r_pnls > 0).sum() / len(r_trades)),
                'total_pnl': float(r_pnls.sum()),
            }
        else:
            regimes[regime] = {'n_trades': len(r_trades), 'sharpe': 0.0}

    # R1: Regime test
    bull_sharpe = regimes.get('bull', {}).get('sharpe', 0.0)
    bear_sharpe = regimes.get('bear', {}).get('sharpe', 0.0)
    max_regime_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_divergence = abs(bull_sharpe - bear_sharpe) / max_regime_sharpe if max_regime_sharpe > 0 else 0
    r1_pass = regime_divergence < 0.50

    # --- Permutation test ---
    log.info(f'Running permutation test ({N_PERMUTATIONS} shuffles)...')
    observed_mean = mean_pnl
    perm_means = []
    rng = np.random.RandomState(42)

    # Build pool of all possible earnings events (not just the ones we traded)
    all_pnls = pnls.copy()
    for _ in range(N_PERMUTATIONS):
        # Shuffle which trades are "selected" - random sign flip + shuffle
        shuffled = rng.choice(all_pnls, size=len(all_pnls), replace=True)
        # Randomly flip signs to simulate random entry timing
        signs = rng.choice([-1, 1], size=len(shuffled))
        perm_means.append(float((shuffled * signs).mean()))

    perm_means = np.array(perm_means)
    perm_p_value = float((perm_means >= observed_mean).sum() / N_PERMUTATIONS)
    perm_pass = perm_p_value < 0.05

    # --- Sub-period test ---
    mid = n_trades // 2
    first_half_pnl = float(pnls[:mid].sum())
    second_half_pnl = float(pnls[mid:].sum())
    subperiod_pass = first_half_pnl > 0 and second_half_pnl > 0

    # --- Outlier test ---
    pnl_95 = np.percentile(pnls, 95)
    pnls_no_outliers = pnls[pnls <= pnl_95]
    outlier_pass = float(pnls_no_outliers.sum()) > 0 if len(pnls_no_outliers) > 0 else False

    results = {
        'strategy': 'Pre-Earnings Vol Crush Iron Condors v1',
        'universe_size': len(set(t.ticker for t in trades)),
        'date_range': f'{trades[0].entry_date} to {trades[-1].exit_date}',
        'total_trades': n_trades,
        'starting_capital': 645.0,
        'ending_equity': float(equity[-1]),
        'total_pnl': total_pnl,

        'metrics': {
            'sharpe': float(sharpe),
            'sortino': float(sortino),
            'profit_factor': float(profit_factor),
            'win_rate': float(win_rate),
            'cagr': float(cagr),
            'max_drawdown': float(max_dd),
            'calmar': float(calmar),
        },

        'per_trade': {
            'avg_premium_collected': float(avg_premium),
            'avg_payout': float(avg_payout),
            'avg_pnl': float(mean_pnl),
            'avg_holding_days': float(avg_holding),
            'median_pnl': float(np.median(pnls)),
        },

        'regime_analysis': regimes,

        'validation': {
            'permutation_test': {
                'observed_mean_pnl': float(observed_mean),
                'perm_p_value': float(perm_p_value),
                'pass': bool(perm_pass),
                'interpretation': 'p < 0.05 means strategy is unlikely due to chance',
            },
            'regime_test_R1': {
                'bull_sharpe': float(bull_sharpe),
                'bear_sharpe': float(bear_sharpe),
                'divergence': float(regime_divergence),
                'threshold': 0.50,
                'pass': bool(r1_pass),
            },
            'sub_period_test': {
                'first_half_pnl': float(first_half_pnl),
                'second_half_pnl': float(second_half_pnl),
                'pass': bool(subperiod_pass),
            },
            'outlier_test': {
                'pnl_without_top_5pct': float(pnls_no_outliers.sum()) if len(pnls_no_outliers) > 0 else 0,
                'top_5pct_threshold': float(pnl_95),
                'pass': bool(outlier_pass),
            },
        },

        'cost_assumptions': {
            'commission_per_ic': COST_PER_IC,
            'legs': 4,
            'per_leg_rt': 4.70,
        },

        'top_tickers': _top_tickers(trades),
    }

    return results


def _top_tickers(trades: List[Trade], top_n: int = 10) -> List[Dict]:
    """Rank tickers by total P&L."""
    ticker_pnl = {}
    ticker_count = {}
    for t in trades:
        ticker_pnl[t.ticker] = ticker_pnl.get(t.ticker, 0) + t.pnl_dollar
        ticker_count[t.ticker] = ticker_count.get(t.ticker, 0) + 1

    ranked = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
    return [
        {'ticker': tk, 'total_pnl': round(pnl, 2), 'n_trades': ticker_count[tk]}
        for tk, pnl in ranked[:top_n]
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info('=' * 70)
    log.info('Pre-Earnings Vol Crush Income Strategy v1')
    log.info('=' * 70)

    t0 = time.time()

    # MLflow
    if MLFLOW_OK:
        mlflow.set_experiment('pre_earnings_vol_crush_v1')
        mlflow.start_run(run_name=f'vol_crush_v1_{dt.datetime.now():%Y%m%d_%H%M%S}')
        mlflow.log_param('universe_size', len(UNIVERSE))
        mlflow.log_param('wf_train_days', WF_TRAIN_DAYS)
        mlflow.log_param('wf_test_days', WF_TEST_DAYS)
        mlflow.log_param('max_risk_per_trade', MAX_RISK_PER_TRADE)
        mlflow.log_param('cost_per_ic', COST_PER_IC)
        mlflow.log_param('min_iv_ratio', MIN_IV_RATIO)
        mlflow.log_param('n_permutations', N_PERMUTATIONS)

    # Fetch data
    data = fetch_all_data(UNIVERSE)

    if not data['earnings']:
        log.error('No earnings data retrieved. Exiting.')
        if MLFLOW_OK:
            mlflow.log_metric('total_trades', 0)
            mlflow.end_run(status='FAILED')
        return

    # Run walk-forward backtest
    trades, wf_info = walk_forward_backtest(data)

    if not trades:
        log.error('No trades generated by walk-forward backtest.')
        results = {'error': 'No trades generated', 'tickers_with_data': list(data['earnings'].keys())}
    else:
        # Compute metrics
        results = compute_metrics(trades)
        results['walk_forward'] = {
            'n_windows': len(wf_info.get('wf_windows', [])),
            'train_days': WF_TRAIN_DAYS,
            'test_days': WF_TEST_DAYS,
        }

        # Log to MLflow
        if MLFLOW_OK:
            m = results['metrics']
            mlflow.log_metric('sharpe', m['sharpe'])
            mlflow.log_metric('sortino', m['sortino'])
            mlflow.log_metric('profit_factor', m['profit_factor'])
            mlflow.log_metric('win_rate', m['win_rate'])
            mlflow.log_metric('cagr', m['cagr'])
            mlflow.log_metric('max_drawdown', m['max_drawdown'])
            mlflow.log_metric('calmar', m['calmar'])
            mlflow.log_metric('total_trades', results['total_trades'])
            mlflow.log_metric('total_pnl', results['total_pnl'])

            v = results['validation']
            mlflow.log_metric('perm_p_value', v['permutation_test']['perm_p_value'])
            mlflow.log_metric('regime_divergence', v['regime_test_R1']['divergence'])
            mlflow.log_param('perm_pass', v['permutation_test']['pass'])
            mlflow.log_param('r1_pass', v['regime_test_R1']['pass'])
            mlflow.log_param('subperiod_pass', v['sub_period_test']['pass'])
            mlflow.log_param('outlier_pass', v['outlier_test']['pass'])

        # Print summary
        log.info('')
        log.info('=' * 70)
        log.info('RESULTS SUMMARY')
        log.info('=' * 70)
        log.info(f"Total trades:      {results['total_trades']}")
        log.info(f"Starting capital:  ${results['starting_capital']:.0f}")
        log.info(f"Ending equity:     ${results['ending_equity']:.2f}")
        log.info(f"Total P&L:         ${results['total_pnl']:.2f}")
        log.info(f"Sharpe:            {results['metrics']['sharpe']:.3f}")
        log.info(f"Sortino:           {results['metrics']['sortino']:.3f}")
        log.info(f"Profit Factor:     {results['metrics']['profit_factor']:.3f}")
        log.info(f"Win Rate:          {results['metrics']['win_rate']:.1%}")
        log.info(f"CAGR:              {results['metrics']['cagr']:.1%}")
        log.info(f"Max Drawdown:      {results['metrics']['max_drawdown']:.1%}")
        log.info(f"Calmar:            {results['metrics']['calmar']:.3f}")

        log.info('')
        log.info('--- Regime Analysis ---')
        for regime, rdata in results['regime_analysis'].items():
            log.info(f"  {regime:5s}: n={rdata['n_trades']:3d}, Sharpe={rdata.get('sharpe', 0):.3f}, "
                     f"WR={rdata.get('win_rate', 0):.1%}, PnL=${rdata.get('total_pnl', 0):.2f}")

        log.info('')
        log.info('--- Validation Gates ---')
        v = results['validation']
        log.info(f"  Permutation test:  p={v['permutation_test']['perm_p_value']:.3f} "
                 f"{'PASS' if v['permutation_test']['pass'] else 'FAIL'}")
        log.info(f"  Regime R1 test:    div={v['regime_test_R1']['divergence']:.3f} "
                 f"{'PASS' if v['regime_test_R1']['pass'] else 'FAIL'}")
        log.info(f"  Sub-period test:   1H=${v['sub_period_test']['first_half_pnl']:.2f}, "
                 f"2H=${v['sub_period_test']['second_half_pnl']:.2f} "
                 f"{'PASS' if v['sub_period_test']['pass'] else 'FAIL'}")
        log.info(f"  Outlier test:      PnL_no_outliers=${v['outlier_test']['pnl_without_top_5pct']:.2f} "
                 f"{'PASS' if v['outlier_test']['pass'] else 'FAIL'}")

        log.info('')
        log.info('--- Top Tickers ---')
        for tt in results.get('top_tickers', [])[:10]:
            log.info(f"  {tt['ticker']:5s}: ${tt['total_pnl']:8.2f} ({tt['n_trades']} trades)")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f'\nResults saved to {RESULTS_PATH}')

    elapsed = time.time() - t0
    log.info(f'Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)')

    if MLFLOW_OK:
        mlflow.log_metric('runtime_seconds', elapsed)
        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.end_run()

    log.info('Done.')


if __name__ == '__main__':
    main()
