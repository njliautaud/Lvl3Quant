#!/usr/bin/env python3
"""
Earnings-Timed Jade Lizard Income Strategy v1
==============================================
Jade lizard = short OTM put + short OTM call spread.
Sells jade lizards 5 days before earnings when IV rank is elevated,
closes after earnings vol crush.

Walk-forward: 252d train, 21d test, SLIDING window.
CPU-only. No MLflow.
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import json
import logging
import time
import warnings
from datetime import timedelta
from math import exp, log, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'MA',
    'HD', 'UNH', 'JNJ', 'PG', 'KO', 'PEP', 'MCD', 'WMT', 'COST', 'AVGO',
    'CRM', 'ORCL', 'ADBE', 'NFLX', 'AMD', 'INTC', 'QCOM', 'TXN', 'AMAT', 'MU',
    'GS', 'MS', 'BAC', 'WFC', 'C', 'AXP', 'BLK', 'SCHW', 'LLY', 'PFE',
    'MRK', 'ABBV', 'TMO', 'DHR', 'BMY', 'XOM', 'CVX', 'COP', 'SLB', 'EOG',
]

COMMISSION_PER_LEG = 4.70  # AMP RT
JADE_LIZARD_LEGS = 3
TOTAL_COMMISSION = COMMISSION_PER_LEG * JADE_LIZARD_LEGS  # $14.10
MAX_RISK_PER_TRADE = 300.0
MAX_CONCURRENT = 3
ENTRY_DAYS_BEFORE = 5
DTE_EXPIRY = 21
IV_MULTIPLIER_PRE = 1.5
IV_MULTIPLIER_POST = 1.0
RISK_FREE_RATE = 0.045

# Walk-forward params
WF_TRAIN_DAYS = 252
WF_TEST_DAYS = 21

# Permutation test
N_PERMUTATIONS = 200

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research', 'findings', 'earnings_jade_lizard_v1_results.json')


# ── Black-Scholes ──────────────────────────────────────────────────────────────
def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ── Data Fetching ──────────────────────────────────────────────────────────────
def strip_tz(idx):
    """Strip timezone from DatetimeIndex or return as-is."""
    if hasattr(idx, 'tz') and idx.tz is not None:
        return idx.tz_convert(None)
    return idx


def strip_tz_timestamp(ts):
    """Strip timezone from a single Timestamp."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def fetch_price_data(symbols: List[str], start: str = '2017-01-01', end: str = '2026-07-24') -> Dict[str, pd.DataFrame]:
    """Fetch daily OHLCV data for all symbols."""
    logger.info(f"Fetching price data for {len(symbols)} symbols...")
    price_data = {}
    for i, sym in enumerate(symbols):
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start=start, end=end, auto_adjust=True)
            if hist.empty:
                logger.warning(f"  {sym}: no data")
                continue
            hist.index = strip_tz(hist.index)
            hist.index = hist.index.normalize()
            price_data[sym] = hist
            if (i + 1) % 10 == 0:
                logger.info(f"  Fetched {i+1}/{len(symbols)} symbols")
        except Exception as e:
            logger.warning(f"  {sym}: fetch error: {e}")
    logger.info(f"Got price data for {len(price_data)} symbols")
    return price_data


def fetch_earnings_dates(symbols: List[str]) -> Dict[str, List[pd.Timestamp]]:
    """Fetch earnings dates for all symbols using yfinance."""
    logger.info(f"Fetching earnings dates for {len(symbols)} symbols...")
    earnings = {}
    for i, sym in enumerate(symbols):
        try:
            ticker = yf.Ticker(sym)
            # Try get_earnings_dates first (more reliable)
            try:
                edates = ticker.get_earnings_dates(limit=100)
                if edates is not None and not edates.empty:
                    dates = list(edates.index)
                else:
                    dates = []
            except Exception:
                dates = []

            # Fallback to earnings_dates attribute
            if not dates:
                try:
                    edates = ticker.earnings_dates
                    if edates is not None and not edates.empty:
                        dates = list(edates.index)
                except Exception:
                    pass

            if dates:
                # Strip timezone from each date
                clean_dates = []
                for d in dates:
                    d = strip_tz_timestamp(d)
                    clean_dates.append(d.normalize())
                earnings[sym] = sorted(set(clean_dates))
                if (i + 1) % 10 == 0:
                    logger.info(f"  Fetched earnings for {i+1}/{len(symbols)} symbols")
            else:
                logger.warning(f"  {sym}: no earnings dates found")
        except Exception as e:
            logger.warning(f"  {sym}: earnings fetch error: {e}")
    logger.info(f"Got earnings dates for {len(earnings)} symbols")
    return earnings


def fetch_spy_data(start: str = '2017-01-01', end: str = '2026-07-24') -> pd.DataFrame:
    """Fetch SPY data for regime classification."""
    logger.info("Fetching SPY data for regime classification...")
    ticker = yf.Ticker('SPY')
    spy = ticker.history(start=start, end=end, auto_adjust=True)
    spy.index = strip_tz(spy.index)
    spy.index = spy.index.normalize()
    return spy


# ── Realized Vol & IV Proxy ───────────────────────────────────────────────────
def compute_realized_vol(prices: pd.Series, window: int = 21) -> pd.Series:
    """Annualized realized vol from log returns."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * sqrt(252)


def compute_iv_rank(vol_series: pd.Series, lookback: int = 252) -> pd.Series:
    """IV rank = percentile of current vol vs trailing lookback."""
    def pctrank(x):
        if len(x) < 10:
            return np.nan
        return (x.values[:-1] < x.values[-1]).sum() / (len(x) - 1)
    return vol_series.rolling(lookback + 1).apply(pctrank, raw=False)


# ── Jade Lizard Pricing ───────────────────────────────────────────────────────
def price_jade_lizard(
    S: float, vol: float, T: float, r: float = RISK_FREE_RATE,
    put_stdev: float = 1.0, call_stdev: float = 0.8,
    call_spread_pct: float = 0.05, iv_multiplier: float = IV_MULTIPLIER_PRE
) -> Optional[Dict]:
    """
    Price a jade lizard position.
    Returns dict with strikes, premiums, max risk, or None if condition not met.
    """
    iv = vol * iv_multiplier
    if iv <= 0 or T <= 0 or S <= 0:
        return None

    # Strikes
    put_strike = round(S * (1 - put_stdev * iv * sqrt(T)), 2)
    short_call_strike = round(S * (1 + call_stdev * iv * sqrt(T)), 2)
    long_call_strike = round(short_call_strike * (1 + call_spread_pct), 2)

    if put_strike <= 0 or short_call_strike <= put_strike:
        return None

    # Premiums
    put_premium = bs_put(S, put_strike, T, r, iv)
    short_call_premium = bs_call(S, short_call_strike, T, r, iv)
    long_call_premium = bs_call(S, long_call_strike, T, r, iv)
    call_spread_credit = short_call_premium - long_call_premium

    if call_spread_credit < 0:
        return None

    total_premium = put_premium + call_spread_credit

    # Jade lizard condition: call spread credit >= put premium => no upside risk
    jade_condition = call_spread_credit >= put_premium

    # Max risk on downside = put_strike - premium collected (per share)
    # On upside: if jade condition met, max risk = 0 (call spread loss offset by total credit)
    # If not met, upside risk = (long_call - short_call) - total_premium
    downside_risk = (put_strike - total_premium) * 100  # per contract
    upside_risk = 0.0 if jade_condition else ((long_call_strike - short_call_strike) - total_premium) * 100

    max_risk = max(downside_risk, upside_risk)
    if max_risk <= 0:
        return None

    return {
        'put_strike': put_strike,
        'short_call_strike': short_call_strike,
        'long_call_strike': long_call_strike,
        'put_premium': put_premium,
        'call_spread_credit': call_spread_credit,
        'total_premium': total_premium,
        'total_premium_dollar': total_premium * 100,
        'max_risk': max_risk,
        'jade_condition': jade_condition,
        'iv_used': iv,
    }


def price_jade_lizard_at_exit(
    S_exit: float, vol_exit: float, T_remaining: float,
    jl: Dict, r: float = RISK_FREE_RATE
) -> float:
    """
    Price the jade lizard at exit to compute P&L.
    Returns the cost to close (negative = profit when closing short position).
    """
    iv_exit = vol_exit * IV_MULTIPLIER_POST  # post-earnings vol crush

    put_cost = bs_put(S_exit, jl['put_strike'], T_remaining, r, iv_exit)
    short_call_cost = bs_call(S_exit, jl['short_call_strike'], T_remaining, r, iv_exit)
    long_call_value = bs_call(S_exit, jl['long_call_strike'], T_remaining, r, iv_exit)

    # Cost to close = buy back short put + buy back short call - sell long call
    close_cost = put_cost + short_call_cost - long_call_value
    return close_cost


# ── Trade Simulation ──────────────────────────────────────────────────────────
def simulate_trade(
    sym: str, entry_date: pd.Timestamp, earnings_date: pd.Timestamp,
    price_df: pd.DataFrame, vol_series: pd.Series,
    put_stdev: float = 1.0, call_stdev: float = 0.8,
    call_spread_pct: float = 0.05
) -> Optional[Dict]:
    """Simulate a single jade lizard trade."""
    # Get entry price
    if entry_date not in price_df.index:
        # Find nearest prior date
        mask = price_df.index <= entry_date
        if mask.sum() == 0:
            return None
        entry_date = price_df.index[mask][-1]

    S_entry = price_df.loc[entry_date, 'Close']
    if pd.isna(S_entry) or S_entry <= 0:
        return None

    # Get vol at entry
    if entry_date not in vol_series.index:
        mask = vol_series.index <= entry_date
        if mask.sum() == 0:
            return None
        entry_date_vol = vol_series.index[mask][-1]
    else:
        entry_date_vol = entry_date
    vol_entry = vol_series.loc[entry_date_vol]
    if pd.isna(vol_entry) or vol_entry <= 0:
        return None

    # DTE
    T_entry = DTE_EXPIRY / 252.0

    # Price the jade lizard
    jl = price_jade_lizard(
        S_entry, vol_entry, T_entry,
        put_stdev=put_stdev, call_stdev=call_stdev,
        call_spread_pct=call_spread_pct
    )
    if jl is None:
        return None

    # Skip if jade lizard condition not met
    if not jl['jade_condition']:
        return None

    # Position sizing: max $300 risk
    if jl['max_risk'] <= 0:
        return None
    n_contracts = max(1, int(MAX_RISK_PER_TRADE / jl['max_risk']))

    # Exit: 1 day after earnings or at DTE expiry
    exit_date = earnings_date + timedelta(days=1)
    # Find next trading day on or after exit_date
    mask = price_df.index >= exit_date
    if mask.sum() == 0:
        return None
    exit_date = price_df.index[mask][0]

    # Cap at DTE expiry
    expiry_date = entry_date + timedelta(days=DTE_EXPIRY)
    if exit_date > expiry_date:
        mask2 = price_df.index <= expiry_date
        if mask2.sum() == 0:
            return None
        exit_date = price_df.index[mask2][-1]

    S_exit = price_df.loc[exit_date, 'Close']
    if pd.isna(S_exit) or S_exit <= 0:
        return None

    # Vol at exit
    if exit_date not in vol_series.index:
        mask = vol_series.index <= exit_date
        if mask.sum() == 0:
            return None
        exit_date_vol = vol_series.index[mask][-1]
    else:
        exit_date_vol = exit_date
    vol_exit = vol_series.loc[exit_date_vol]
    if pd.isna(vol_exit) or vol_exit <= 0:
        vol_exit = vol_entry * 0.7  # assume vol crush

    # Time remaining at exit
    days_held = (exit_date - entry_date).days
    T_remaining = max(0, (DTE_EXPIRY - days_held) / 252.0)

    # Price at exit (vol crush)
    close_cost = price_jade_lizard_at_exit(S_exit, vol_exit, T_remaining, jl)

    # P&L per share = premium collected - cost to close
    pnl_per_share = jl['total_premium'] - close_cost
    pnl_per_contract = pnl_per_share * 100
    total_pnl = pnl_per_contract * n_contracts - TOTAL_COMMISSION * n_contracts

    return {
        'symbol': sym,
        'entry_date': entry_date,
        'exit_date': exit_date,
        'earnings_date': earnings_date,
        'entry_price': float(S_entry),
        'exit_price': float(S_exit),
        'put_strike': jl['put_strike'],
        'short_call_strike': jl['short_call_strike'],
        'long_call_strike': jl['long_call_strike'],
        'premium_collected': float(jl['total_premium_dollar'] * n_contracts),
        'close_cost': float(close_cost * 100 * n_contracts),
        'commission': float(TOTAL_COMMISSION * n_contracts),
        'pnl': float(total_pnl),
        'n_contracts': n_contracts,
        'days_held': days_held,
        'vol_entry': float(vol_entry),
        'vol_exit': float(vol_exit),
        'iv_used': float(jl['iv_used']),
        'jade_condition': jl['jade_condition'],
        'max_risk': float(jl['max_risk'] * n_contracts),
        'return_pct': float(total_pnl / (jl['max_risk'] * n_contracts)) if jl['max_risk'] > 0 else 0.0,
    }


# ── Walk-Forward Engine ───────────────────────────────────────────────────────
def run_walk_forward(
    price_data: Dict[str, pd.DataFrame],
    earnings_data: Dict[str, List[pd.Timestamp]],
    spy_data: pd.DataFrame,
) -> Tuple[List[Dict], Dict]:
    """Run walk-forward backtest with parameter optimization."""
    # Build a unified date index
    all_dates = set()
    for sym, df in price_data.items():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)
    if len(all_dates) < WF_TRAIN_DAYS + WF_TEST_DAYS:
        logger.error("Not enough dates for walk-forward")
        return [], {}

    # Precompute vol series for all symbols
    logger.info("Precomputing volatility series...")
    vol_data = {}
    iv_rank_data = {}
    for sym, df in price_data.items():
        vol = compute_realized_vol(df['Close'], window=21)
        vol_data[sym] = vol
        iv_rank_data[sym] = compute_iv_rank(vol, lookback=252)

    # Parameter grid for optimization
    param_grid = [
        {'iv_threshold': 0.40, 'put_stdev': 1.0, 'call_stdev': 0.8, 'spread_pct': 0.05},
        {'iv_threshold': 0.50, 'put_stdev': 1.0, 'call_stdev': 0.8, 'spread_pct': 0.05},
        {'iv_threshold': 0.60, 'put_stdev': 1.0, 'call_stdev': 0.8, 'spread_pct': 0.05},
        {'iv_threshold': 0.50, 'put_stdev': 0.8, 'call_stdev': 0.7, 'spread_pct': 0.05},
        {'iv_threshold': 0.50, 'put_stdev': 1.2, 'call_stdev': 1.0, 'spread_pct': 0.05},
        {'iv_threshold': 0.50, 'put_stdev': 1.0, 'call_stdev': 0.8, 'spread_pct': 0.03},
        {'iv_threshold': 0.50, 'put_stdev': 1.0, 'call_stdev': 0.8, 'spread_pct': 0.07},
        {'iv_threshold': 0.55, 'put_stdev': 1.0, 'call_stdev': 0.8, 'spread_pct': 0.05},
    ]

    # Walk-forward windows
    all_trades = []
    wf_step = 0
    start_idx = 0

    while start_idx + WF_TRAIN_DAYS + WF_TEST_DAYS <= len(all_dates):
        train_start = all_dates[start_idx]
        train_end = all_dates[start_idx + WF_TRAIN_DAYS - 1]
        test_start = all_dates[start_idx + WF_TRAIN_DAYS]
        test_end_idx = min(start_idx + WF_TRAIN_DAYS + WF_TEST_DAYS - 1, len(all_dates) - 1)
        test_end = all_dates[test_end_idx]

        wf_step += 1
        if wf_step % 10 == 1:
            logger.info(f"WF step {wf_step}: train {train_start.date()}-{train_end.date()}, "
                        f"test {test_start.date()}-{test_end.date()}")

        # ── TRAIN: find best params on training period ──
        best_sharpe = -999
        best_params = param_grid[1]  # default

        for params in param_grid:
            train_trades = _generate_trades_for_period(
                train_start, train_end, price_data, earnings_data,
                vol_data, iv_rank_data,
                iv_threshold=params['iv_threshold'],
                put_stdev=params['put_stdev'],
                call_stdev=params['call_stdev'],
                spread_pct=params['spread_pct'],
            )
            if len(train_trades) < 5:
                continue
            pnls = [t['pnl'] for t in train_trades]
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        # ── TEST: apply best params to test period ──
        test_trades = _generate_trades_for_period(
            test_start, test_end, price_data, earnings_data,
            vol_data, iv_rank_data,
            iv_threshold=best_params['iv_threshold'],
            put_stdev=best_params['put_stdev'],
            call_stdev=best_params['call_stdev'],
            spread_pct=best_params['spread_pct'],
        )

        for t in test_trades:
            t['wf_step'] = wf_step
            t['wf_params'] = best_params.copy()
        all_trades.extend(test_trades)

        # Slide window
        start_idx += WF_TEST_DAYS

    logger.info(f"Walk-forward complete: {wf_step} steps, {len(all_trades)} OOT trades")

    # Classify regime for each trade
    spy_ret_20d = spy_data['Close'].pct_change(20)
    for t in all_trades:
        entry = t['entry_date']
        if entry in spy_ret_20d.index:
            ret = spy_ret_20d.loc[entry]
        else:
            mask = spy_ret_20d.index <= entry
            if mask.sum() > 0:
                ret = spy_ret_20d.iloc[mask.values.argmax() if mask.sum() > 0 else -1]
                # get the closest prior
                idx = spy_ret_20d.index[mask][-1]
                ret = spy_ret_20d.loc[idx]
            else:
                ret = 0.0
        if pd.isna(ret):
            ret = 0.0
        if ret > 0.02:
            t['regime'] = 'bull'
        elif ret < -0.02:
            t['regime'] = 'bear'
        else:
            t['regime'] = 'flat'

    return all_trades, {'wf_steps': wf_step}


def _generate_trades_for_period(
    period_start, period_end,
    price_data, earnings_data, vol_data, iv_rank_data,
    iv_threshold=0.50, put_stdev=1.0, call_stdev=0.8, spread_pct=0.05
) -> List[Dict]:
    """Generate trades for a given period with given parameters."""
    trades = []
    active_positions = []  # track concurrent

    for sym in UNIVERSE:
        if sym not in price_data or sym not in earnings_data:
            continue
        if sym not in vol_data or sym not in iv_rank_data:
            continue

        df = price_data[sym]
        vol = vol_data[sym]
        ivr = iv_rank_data[sym]

        for edate in earnings_data[sym]:
            # Entry = 5 days before earnings
            entry_date = edate - timedelta(days=ENTRY_DAYS_BEFORE + 2)  # +2 for weekends
            # Find nearest trading day
            mask = df.index <= entry_date
            if mask.sum() == 0:
                continue

            # Get the trading day ~5 business days before earnings
            earn_idx = df.index.searchsorted(edate)
            entry_idx = earn_idx - ENTRY_DAYS_BEFORE
            if entry_idx < 0 or entry_idx >= len(df.index):
                continue
            entry_date = df.index[entry_idx]

            # Must be in period
            if entry_date < period_start or entry_date > period_end:
                continue

            # IV rank filter
            if entry_date not in ivr.index:
                continue
            rank = ivr.loc[entry_date]
            if pd.isna(rank) or rank < iv_threshold:
                continue

            # Price movement filter: no >8% move in prior 5 days
            if entry_idx < 5:
                continue
            price_5d_ago = df.iloc[entry_idx - 5]['Close']
            price_now = df.iloc[entry_idx]['Close']
            if abs(price_now / price_5d_ago - 1) > 0.08:
                continue

            # Concurrent position check (approximate)
            active_positions = [p for p in active_positions if p['exit_date'] > entry_date]
            if len(active_positions) >= MAX_CONCURRENT:
                continue

            # Simulate trade
            trade = simulate_trade(
                sym, entry_date, edate, df, vol,
                put_stdev=put_stdev, call_stdev=call_stdev,
                call_spread_pct=spread_pct
            )
            if trade is not None:
                trades.append(trade)
                active_positions.append(trade)

    return trades


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(trades: List[Dict]) -> Dict:
    """Compute strategy performance metrics."""
    if not trades:
        return {'error': 'no trades'}

    pnls = np.array([t['pnl'] for t in trades])
    returns = np.array([t['return_pct'] for t in trades])
    n = len(pnls)

    total_pnl = float(np.sum(pnls))
    avg_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls))
    wins = int((pnls > 0).sum())
    losses = int((pnls <= 0).sum())
    wr = wins / n if n > 0 else 0.0

    # Sharpe (annualized assuming ~50 trades/year as rough scale)
    trades_per_year = 50  # rough estimate
    sharpe = (avg_pnl / std_pnl * sqrt(trades_per_year)) if std_pnl > 0 else 0.0

    # Sortino
    downside = pnls[pnls < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 1.0
    sortino = (avg_pnl / downside_std * sqrt(trades_per_year)) if downside_std > 0 else 0.0

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0])) if wins > 0 else 0.0
    gross_loss = float(abs(np.sum(pnls[pnls < 0]))) if losses > 0 else 1.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # CAGR (approximate)
    if len(trades) >= 2:
        first_date = min(t['entry_date'] for t in trades)
        last_date = max(t['exit_date'] for t in trades)
        years = max((last_date - first_date).days / 365.25, 0.5)
    else:
        years = 1.0

    # Equity curve for MaxDD and CAGR
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    max_dd_pct = float(max_dd / peak[np.argmin(drawdown)]) if peak[np.argmin(drawdown)] != 0 else 0.0

    # Approximate CAGR from total return on notional
    initial_capital = MAX_RISK_PER_TRADE * MAX_CONCURRENT * 10  # rough capital base
    cagr = ((initial_capital + total_pnl) / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0

    # Calmar
    calmar = cagr / abs(max_dd_pct) if max_dd_pct != 0 else 0.0

    # Per-trade stats
    avg_premium = float(np.mean([t['premium_collected'] for t in trades]))
    avg_max_loss = float(np.mean([t['max_risk'] for t in trades]))
    avg_holding = float(np.mean([t['days_held'] for t in trades]))

    return {
        'total_trades': n,
        'wins': wins,
        'losses': losses,
        'win_rate': round(wr, 4),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 2),
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'profit_factor': round(pf, 4),
        'cagr': round(cagr, 4),
        'max_drawdown_dollar': round(max_dd, 2),
        'max_drawdown_pct': round(max_dd_pct, 4),
        'calmar': round(calmar, 4),
        'avg_premium_collected': round(avg_premium, 2),
        'avg_max_loss': round(avg_max_loss, 2),
        'avg_holding_days': round(avg_holding, 1),
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
    }


def compute_regime_metrics(trades: List[Dict]) -> Dict:
    """Compute metrics stratified by regime."""
    regimes = {}
    for regime in ['bull', 'bear', 'flat']:
        regime_trades = [t for t in trades if t.get('regime') == regime]
        if regime_trades:
            regimes[regime] = compute_metrics(regime_trades)
        else:
            regimes[regime] = {'total_trades': 0, 'sharpe': 0.0}
    return regimes


# ── Statistical Tests ──────────────────────────────────────────────────────────
def permutation_test(trades: List[Dict], n_perms: int = N_PERMUTATIONS) -> Dict:
    """Permutation test: shuffle which earnings we trade."""
    if not trades:
        return {'p_value': 1.0, 'observed_sharpe': 0.0}

    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    observed_mean = np.mean(pnls)
    observed_std = np.std(pnls)
    observed_sharpe = observed_mean / observed_std if observed_std > 0 else 0.0

    rng = np.random.RandomState(42)
    count_better = 0

    for _ in range(n_perms):
        # Randomly select same number of trades (shuffle selection)
        # This simulates random entry timing
        shuffled_pnls = rng.choice(pnls, size=n, replace=True)
        # Randomly flip signs to simulate random direction
        signs = rng.choice([-1, 1], size=n)
        shuffled_pnls = shuffled_pnls * signs
        shuf_mean = np.mean(shuffled_pnls)
        shuf_std = np.std(shuffled_pnls)
        shuf_sharpe = shuf_mean / shuf_std if shuf_std > 0 else 0.0
        if shuf_sharpe >= observed_sharpe:
            count_better += 1

    p_value = (count_better + 1) / (n_perms + 1)

    return {
        'p_value': round(p_value, 4),
        'observed_sharpe': round(observed_sharpe, 4),
        'n_permutations': n_perms,
        'count_better': count_better,
    }


def regime_test(regime_metrics: Dict) -> Dict:
    """R1 regime test: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|) < 0.50"""
    s_bull = regime_metrics.get('bull', {}).get('sharpe', 0.0)
    s_bear = regime_metrics.get('bear', {}).get('sharpe', 0.0)

    denom = max(abs(s_bull), abs(s_bear))
    if denom == 0:
        ratio = 0.0
    else:
        ratio = abs(s_bull - s_bear) / denom

    passed = ratio < 0.50

    return {
        'sharpe_bull': s_bull,
        'sharpe_bear': s_bear,
        'regime_ratio': round(ratio, 4),
        'passed': passed,
        'threshold': 0.50,
    }


def sub_period_test(trades: List[Dict]) -> Dict:
    """Split trades in half chronologically. Both halves must be profitable."""
    if len(trades) < 4:
        return {'passed': False, 'reason': 'too few trades'}

    sorted_trades = sorted(trades, key=lambda t: t['entry_date'])
    mid = len(sorted_trades) // 2
    first_half = sorted_trades[:mid]
    second_half = sorted_trades[mid:]

    m1 = compute_metrics(first_half)
    m2 = compute_metrics(second_half)

    passed = m1['total_pnl'] > 0 and m2['total_pnl'] > 0

    return {
        'first_half_pnl': m1['total_pnl'],
        'first_half_trades': m1['total_trades'],
        'first_half_sharpe': m1['sharpe'],
        'second_half_pnl': m2['total_pnl'],
        'second_half_trades': m2['total_trades'],
        'second_half_sharpe': m2['sharpe'],
        'passed': passed,
    }


def outlier_test(trades: List[Dict]) -> Dict:
    """Remove top 5% of trades by P&L. Must still be profitable."""
    if len(trades) < 20:
        return {'passed': False, 'reason': 'too few trades for outlier test'}

    sorted_by_pnl = sorted(trades, key=lambda t: t['pnl'], reverse=True)
    cutoff = max(1, int(len(sorted_by_pnl) * 0.05))
    trimmed = sorted_by_pnl[cutoff:]

    m = compute_metrics(trimmed)
    passed = m['total_pnl'] > 0

    return {
        'removed_count': cutoff,
        'remaining_trades': m['total_trades'],
        'trimmed_pnl': m['total_pnl'],
        'trimmed_sharpe': m['sharpe'],
        'trimmed_wr': m['win_rate'],
        'passed': passed,
    }


# ── JSON Serialization Helper ─────────────────────────────────────────────────
def make_serializable(obj):
    """Convert numpy/pandas types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    elif isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    return obj


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Earnings-Timed Jade Lizard Income Strategy v1")
    logger.info("=" * 70)

    # 1. Fetch data
    price_data = fetch_price_data(UNIVERSE)
    earnings_data = fetch_earnings_dates(UNIVERSE)
    spy_data = fetch_spy_data()

    if not price_data:
        logger.error("No price data fetched. Exiting.")
        return
    if not earnings_data:
        logger.error("No earnings data fetched. Exiting.")
        return

    # Log data summary
    total_earnings = sum(len(v) for v in earnings_data.values())
    logger.info(f"Data: {len(price_data)} symbols, {total_earnings} total earnings dates")

    # 2. Run walk-forward backtest
    logger.info("Starting walk-forward backtest...")
    trades, wf_info = run_walk_forward(price_data, earnings_data, spy_data)

    if not trades:
        logger.warning("No trades generated. Check data coverage.")
        results = {'error': 'no trades generated', 'data_symbols': len(price_data),
                    'earnings_symbols': len(earnings_data)}
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(make_serializable(results), f, indent=2, default=str)
        logger.info(f"Results saved to {RESULTS_PATH}")
        return

    logger.info(f"Generated {len(trades)} OOT trades")

    # 3. Compute metrics
    logger.info("Computing metrics...")
    overall = compute_metrics(trades)
    regime = compute_regime_metrics(trades)

    # 4. Statistical tests
    logger.info("Running permutation test (200 shuffles)...")
    perm = permutation_test(trades)
    logger.info(f"Permutation p-value: {perm['p_value']}")

    regime_r1 = regime_test(regime)
    logger.info(f"Regime test: ratio={regime_r1['regime_ratio']}, passed={regime_r1['passed']}")

    subperiod = sub_period_test(trades)
    logger.info(f"Sub-period test: passed={subperiod['passed']}")

    outlier = outlier_test(trades)
    logger.info(f"Outlier test: passed={outlier['passed']}")

    # 5. Build results
    # Trade detail (sample for JSON, keep it manageable)
    trade_details = []
    for t in trades:
        trade_details.append({
            'symbol': t['symbol'],
            'entry_date': t['entry_date'].isoformat(),
            'exit_date': t['exit_date'].isoformat(),
            'earnings_date': t['earnings_date'].isoformat(),
            'pnl': round(t['pnl'], 2),
            'return_pct': round(t['return_pct'], 4),
            'premium_collected': round(t['premium_collected'], 2),
            'days_held': t['days_held'],
            'regime': t.get('regime', 'unknown'),
            'vol_entry': round(t['vol_entry'], 4),
            'n_contracts': t['n_contracts'],
        })

    # Symbol breakdown
    sym_stats = {}
    for sym in set(t['symbol'] for t in trades):
        sym_trades = [t for t in trades if t['symbol'] == sym]
        sym_pnls = [t['pnl'] for t in sym_trades]
        sym_stats[sym] = {
            'trades': len(sym_trades),
            'total_pnl': round(sum(sym_pnls), 2),
            'avg_pnl': round(np.mean(sym_pnls), 2),
            'win_rate': round(sum(1 for p in sym_pnls if p > 0) / len(sym_pnls), 4),
        }

    elapsed = time.time() - t0

    results = {
        'strategy': 'Earnings-Timed Jade Lizard Income Strategy v1',
        'backtest_period': '2018-2026',
        'walk_forward': {
            'train_days': WF_TRAIN_DAYS,
            'test_days': WF_TEST_DAYS,
            'window': 'SLIDING',
            'steps': wf_info.get('wf_steps', 0),
        },
        'overall_metrics': overall,
        'regime_metrics': regime,
        'statistical_tests': {
            'permutation_test': perm,
            'regime_test_r1': regime_r1,
            'sub_period_test': subperiod,
            'outlier_test': outlier,
        },
        'all_tests_passed': all([
            regime_r1.get('passed', False),
            subperiod.get('passed', False),
            outlier.get('passed', False),
            perm.get('p_value', 1.0) < 0.10,
        ]),
        'symbol_breakdown': sym_stats,
        'trade_details': trade_details,
        'cost_assumptions': {
            'commission_per_leg': COMMISSION_PER_LEG,
            'total_commission_3leg': TOTAL_COMMISSION,
            'max_risk_per_trade': MAX_RISK_PER_TRADE,
        },
        'runtime_seconds': round(elapsed, 1),
    }

    # 6. Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(make_serializable(results), f, indent=2, default=str)
    logger.info(f"Results saved to {RESULTS_PATH}")

    # 7. Print summary
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total trades:     {overall['total_trades']}")
    logger.info(f"Win rate:         {overall['win_rate']:.1%}")
    logger.info(f"Total P&L:        ${overall['total_pnl']:,.2f}")
    logger.info(f"Avg P&L/trade:    ${overall['avg_pnl']:,.2f}")
    logger.info(f"Sharpe:           {overall['sharpe']:.3f}")
    logger.info(f"Sortino:          {overall['sortino']:.3f}")
    logger.info(f"Profit Factor:    {overall['profit_factor']:.3f}")
    logger.info(f"CAGR:             {overall['cagr']:.2%}")
    logger.info(f"Max Drawdown:     ${overall['max_drawdown_dollar']:,.2f} ({overall['max_drawdown_pct']:.2%})")
    logger.info(f"Calmar:           {overall['calmar']:.3f}")
    logger.info(f"Avg premium:      ${overall['avg_premium_collected']:,.2f}")
    logger.info(f"Avg holding:      {overall['avg_holding_days']:.1f} days")
    logger.info("-" * 40)
    logger.info(f"Regime (R1):      {'PASS' if regime_r1['passed'] else 'FAIL'} (ratio={regime_r1['regime_ratio']:.3f})")
    logger.info(f"Permutation:      p={perm['p_value']:.3f}")
    logger.info(f"Sub-period:       {'PASS' if subperiod['passed'] else 'FAIL'}")
    logger.info(f"Outlier:          {'PASS' if outlier['passed'] else 'FAIL'}")
    logger.info(f"ALL TESTS:        {'PASS' if results['all_tests_passed'] else 'FAIL'}")
    logger.info(f"Runtime:          {elapsed:.1f}s")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
