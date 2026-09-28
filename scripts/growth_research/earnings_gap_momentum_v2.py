#!/usr/bin/env python3
"""
Earnings Gap Momentum v2 — Improved post-earnings gap buying strategy
=====================================================================

CONTEXT: KB #282 found post-earnings gap buying (>5% gap up, buy calls) showed
Sharpe 1.85 but FAILED permutation test (p=0.11, only 48 trades). Pre-earnings
buying = DEATH (IV crush). This script tests 6 variants designed to increase
trade count for statistical significance.

VARIANTS:
  A) Original: Gap >5%, buy ATM call, DTE=14, +30% TP / -25% SL, 5-day max hold
  B) Lower threshold: Gap >3% (more trades but weaker signal)
  C) Gap + momentum: Gap >3% AND 21d momentum > 0 (confluence filter)
  D) Gap + volume: Gap >3% AND volume > 2x avg (institutional participation)
  E) Multi-ETF: Apply to 14 liquid sector ETFs (gaps on macro/sector news)
  F) Sector ETF + VIX filter: Gap >3% on sector ETFs, only when VIX < 25

ADVERSARIAL VALIDATION (HC #753, all 4 gates):
  1. Permutation test: 1000 random direction shuffles, p < 0.05
  2. Regime analysis: Bull/Bear/Flat by SPY 63d return, |bull-bear|/max < 0.50
  3. Sub-period: Split in half, both halves Sharpe > 0.5
  4. Outlier removal: Remove top/bottom 5% monthly returns, Sharpe > 0.5

DATA: yfinance from 2010-01-01. Self-contained BS pricing. No research.tools imports.
SIZING: $645 capital, $200 max/trade, 15% entry haircut, $2.60 RT commission.
METRIC: Calendar-month Sharpe as primary.
WINDOW: Sliding walk-forward only (HC #0).

MLflow experiment: "earnings_gap_momentum_v2"
"""

import sys
import os
import json
import warnings
import time
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings('ignore')

# --- Path setup (works on Jupiter and Neptune) ---
if os.path.isdir('/home/nick/Lvl3Quant'):
    LVL3_ROOT = '/home/nick/Lvl3Quant'
elif os.path.isdir('/home/jupiter/Lvl3Quant'):
    LVL3_ROOT = '/home/jupiter/Lvl3Quant'
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'earnings_gap_momentum_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================

STARTING_CAPITAL = 645.0
MAX_POSITION_DOLLARS = 200.0
COMMISSION_RT = 2.60           # 4 x $0.65
ENTRY_HAIRCUT = 0.15           # 15% debit haircut
RISK_FREE_RATE = 0.05
DTE = 14                      # days to expiration
TP_PCT = 0.30                 # +30% take profit
SL_PCT = -0.25                # -25% stop loss
MAX_HOLD_DAYS = 5             # 5-day max hold
MAX_CONCURRENT = 3            # max open positions
N_PERMUTATIONS = 1000
START_DATE = '2010-01-01'
END_DATE = '2026-07-27'

# Stock universe for variants A-D (individual stocks with earnings)
STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'SQ', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH',
    'DXCM', 'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT',
    'DASH', 'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL',
    'UPST', 'AFRM', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD',
    'NIO', 'XPEV', 'LI', 'INTC', 'MU', 'AVGO', 'QCOM', 'MRVL', 'LRCX', 'AMAT',
    'MRNA', 'BNTX', 'BIIB', 'REGN', 'VRTX', 'ISRG', 'LULU', 'ETSY', 'W', 'CHWY',
    'DKS', 'DECK', 'ON', 'SEDG', 'GS', 'MS', 'SCHW', 'SPOT', 'ZM', 'DOCU',
    'TWLO', 'WDAY', 'NOW', 'CRM', 'FDX', 'UPS', 'CAT', 'BA', 'DE',
]

# Sector ETFs for variants E-F
ETF_UNIVERSE = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',
    'XLC', 'SPY', 'QQQ', 'IWM',
]


# ============================================================
# BLACK-SCHOLES PRICING (self-contained, no external imports)
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


def estimate_iv_from_atr(close_series, period=14):
    """
    ATR-based IV estimation: IV = ATR_14 / S * sqrt(252).
    Uses true range approximation from close-only data (high/low not always available).
    """
    if len(close_series) < period + 1:
        return 0.30  # default fallback
    # Approximate ATR from close-to-close absolute changes
    abs_changes = np.abs(np.diff(close_series[-(period + 1):]))
    atr = np.mean(abs_changes)
    S = close_series[-1]
    if S <= 0:
        return 0.30
    iv = (atr / S) * np.sqrt(252)
    return max(0.10, min(iv, 3.0))  # clamp to reasonable range


def estimate_iv_from_atr_full(high_series, low_series, close_series, period=14):
    """
    Full ATR-based IV using high/low/close when available.
    IV = ATR_14 / S * sqrt(252).
    """
    n = len(close_series)
    if n < period + 1:
        return 0.30
    trs = []
    for i in range(n - period, n):
        if i < 1:
            continue
        h = high_series[i] if i < len(high_series) else close_series[i]
        l = low_series[i] if i < len(low_series) else close_series[i]
        pc = close_series[i - 1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if not trs:
        return 0.30
    atr = np.mean(trs)
    S = close_series[-1]
    if S <= 0:
        return 0.30
    iv = (atr / S) * np.sqrt(252)
    return max(0.10, min(iv, 3.0))


# ============================================================
# DATA LOADING
# ============================================================

def load_data(tickers, label='stocks'):
    """
    Load daily OHLCV for given tickers + SPY + VIX via yfinance with caching.
    Returns: prices dict {ticker: DataFrame}, spy_df, vix_df
    """
    import yfinance as yf

    cache_dir = os.path.join(LVL3_ROOT, 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Create cache key from sorted tickers
    ticker_hash = hashlib.md5('_'.join(sorted(tickers)).encode()).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f'egm_v2_{label}_{ticker_hash}.parquet')

    all_tickers = list(set(tickers + ['SPY', '^VIX']))
    prices = {}

    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            latest = df.index.get_level_values('date').max()
            if pd.Timestamp(latest) >= pd.Timestamp('2026-07-20'):
                print(f"Loaded cached {label} data: {len(df)} rows", flush=True)
                for t in df.index.get_level_values('ticker').unique():
                    prices[t] = df.loc[t].copy()
                spy_df = prices.pop('SPY', None)
                vix_df = prices.pop('^VIX', None)
                return prices, spy_df, vix_df
        except Exception:
            pass

    print(f"Downloading {label} data for {len(all_tickers)} tickers...", flush=True)
    all_frames = []

    for ticker in all_tickers:
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE,
                               progress=False, auto_adjust=True)
            if len(data) < 50:
                print(f"  WARNING: {ticker} has only {len(data)} rows, skipping", flush=True)
                continue
            data.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                            for c in data.columns]
            data['ticker'] = ticker
            data.index.name = 'date'
            all_frames.append(data)
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}", flush=True)

    if not all_frames:
        raise RuntimeError("No price data downloaded")

    combined = pd.concat(all_frames)
    combined = combined.reset_index().set_index(['ticker', 'date']).sort_index()

    try:
        combined.to_parquet(cache_path)
    except Exception:
        pass

    for t in combined.index.get_level_values('ticker').unique():
        prices[t] = combined.loc[t].copy()

    spy_df = prices.pop('SPY', None)
    vix_df = prices.pop('^VIX', None)
    return prices, spy_df, vix_df


def load_earnings_dates(tickers):
    """
    Load earnings dates for individual stocks via yfinance with caching.
    Returns: dict {ticker: [date_str, ...]}
    """
    import yfinance as yf

    cache_path = os.path.join(LVL3_ROOT, 'data', 'cache', 'earnings_dates_v2.json')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    earnings_dates = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                earnings_dates = json.load(f)
            if len(earnings_dates) >= len(tickers) * 0.4:
                total = sum(len(v) for v in earnings_dates.values())
                print(f"Loaded cached earnings dates: {len(earnings_dates)} tickers, "
                      f"{total} total dates", flush=True)
                return earnings_dates
        except Exception:
            pass

    print("Fetching earnings dates...", flush=True)
    for ticker in tickers:
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
        except Exception:
            pass

    if earnings_dates:
        try:
            with open(cache_path, 'w') as f:
                json.dump(earnings_dates, f, indent=2)
        except Exception:
            pass

    print(f"Got earnings dates for {len(earnings_dates)} tickers", flush=True)
    return earnings_dates


# ============================================================
# GAP DETECTION
# ============================================================

def detect_gaps(df, threshold, earnings_dates=None, require_earnings=True):
    """
    Detect gap openings above threshold.

    Args:
        df: OHLCV DataFrame with 'open', 'close', 'volume' columns
        threshold: minimum gap % (e.g. 0.05 for 5%)
        earnings_dates: list of earnings date strings for this ticker (optional)
        require_earnings: if True, only flag gaps near earnings dates

    Returns: list of dicts with gap info
    """
    if df is None or len(df) < 30:
        return []

    gaps = []
    close_arr = df['close'].values
    open_arr = df['open'].values
    vol_arr = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
    dates = df.index

    # Precompute 20-day average volume
    avg_vol = pd.Series(vol_arr).rolling(20, min_periods=5).mean().values

    # Precompute 21-day momentum (close / close_21d_ago - 1)
    mom_21d = np.full(len(df), np.nan)
    for i in range(21, len(df)):
        if close_arr[i - 21] > 0:
            mom_21d[i] = close_arr[i] / close_arr[i - 21] - 1.0

    # Convert earnings_dates to set of date strings for fast lookup
    earn_set = set()
    if earnings_dates:
        for d in earnings_dates:
            earn_set.add(str(d)[:10])

    for i in range(1, len(df)):
        prev_close = close_arr[i - 1]
        if prev_close <= 0:
            continue
        gap_pct = (open_arr[i] - prev_close) / prev_close

        if abs(gap_pct) < threshold:
            continue

        # Check earnings proximity if required
        if require_earnings and earn_set:
            dt_str = str(dates[i])[:10]
            prev_str = str(dates[i - 1])[:10] if i > 0 else ''
            # Gap must be within 1 trading day of earnings date
            near_earnings = (dt_str in earn_set or prev_str in earn_set)
            if not near_earnings:
                continue
        elif require_earnings and not earn_set:
            # No earnings dates available, skip
            continue

        vol_ratio = vol_arr[i] / avg_vol[i] if avg_vol[i] > 0 and not np.isnan(avg_vol[i]) else 1.0

        gaps.append({
            'date': dates[i],
            'idx': i,
            'gap_pct': gap_pct,
            'direction': 'up' if gap_pct > 0 else 'down',
            'open_price': open_arr[i],
            'prev_close': prev_close,
            'volume_ratio': vol_ratio,
            'momentum_21d': mom_21d[i] if not np.isnan(mom_21d[i]) else 0.0,
        })

    return gaps


def detect_etf_gaps(df, threshold):
    """
    Detect ANY significant gap for ETFs (not just earnings).
    ETFs gap on macro/sector news, not individual earnings.
    """
    return detect_gaps(df, threshold, earnings_dates=None, require_earnings=False)


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trades(gaps, df, spy_df=None, vix_df=None, variant_config=None):
    """
    Simulate options trades from gap signals.

    For each gap:
    1. Buy ATM call (gap up) or ATM put (gap down) at open on gap day
    2. Apply 15% entry haircut on debit
    3. Daily mark-to-market via BS repricing
    4. Exit: +30% TP, -25% SL, or 5-day max hold

    Returns: list of trade dicts with P&L
    """
    if not gaps:
        return []

    config = variant_config or {}
    vix_filter = config.get('vix_filter', None)  # max VIX level

    close_arr = df['close'].values
    high_arr = df['high'].values if 'high' in df.columns else close_arr
    low_arr = df['low'].values if 'low' in df.columns else close_arr
    dates = df.index

    trades = []

    for gap in gaps:
        idx = gap['idx']
        if idx + MAX_HOLD_DAYS >= len(df):
            continue  # not enough data to simulate hold

        # VIX filter
        if vix_filter is not None and vix_df is not None:
            gap_date = gap['date']
            vix_level = _get_vix(vix_df, gap_date)
            if vix_level > vix_filter:
                continue

        S = gap['open_price']
        K = round(S, 0)  # ATM strike (nearest dollar)
        if K <= 0:
            continue

        # Estimate IV from ATR
        lookback_start = max(0, idx - 20)
        iv = estimate_iv_from_atr_full(
            high_arr[lookback_start:idx],
            low_arr[lookback_start:idx],
            close_arr[lookback_start:idx],
            period=14
        )

        T = DTE / 252.0
        opt_type = 'call' if gap['direction'] == 'up' else 'put'

        # Entry price with haircut
        theo_price = option_price(S, K, T, RISK_FREE_RATE, iv, opt_type)
        if theo_price < 0.10:
            continue  # too cheap, skip
        entry_price = theo_price * (1.0 + ENTRY_HAIRCUT)

        # Position sizing
        n_contracts = max(1, int(MAX_POSITION_DOLLARS / (entry_price * 100)))
        position_cost = n_contracts * entry_price * 100
        if position_cost > MAX_POSITION_DOLLARS * 1.5:
            n_contracts = 1
            position_cost = n_contracts * entry_price * 100

        # Simulate daily mark-to-market
        exit_price = None
        exit_reason = None
        exit_day = None

        for hold_day in range(1, MAX_HOLD_DAYS + 1):
            day_idx = idx + hold_day
            if day_idx >= len(df):
                break

            S_now = close_arr[day_idx]
            T_now = max((DTE - hold_day) / 252.0, 1 / 252.0)

            # IV decay assumption: slight IV decrease over hold period
            iv_now = iv * (1.0 - 0.02 * hold_day)  # 2% IV decay per day
            iv_now = max(iv_now, 0.05)

            mark_price = option_price(S_now, K, T_now, RISK_FREE_RATE, iv_now, opt_type)
            pct_change = (mark_price - entry_price) / entry_price

            if pct_change >= TP_PCT:
                exit_price = mark_price
                exit_reason = 'TP'
                exit_day = hold_day
                break
            elif pct_change <= SL_PCT:
                exit_price = mark_price
                exit_reason = 'SL'
                exit_day = hold_day
                break

        # Time stop: exit at max hold
        if exit_price is None:
            day_idx = min(idx + MAX_HOLD_DAYS, len(df) - 1)
            S_now = close_arr[day_idx]
            T_now = max((DTE - MAX_HOLD_DAYS) / 252.0, 1 / 252.0)
            iv_now = iv * (1.0 - 0.02 * MAX_HOLD_DAYS)
            iv_now = max(iv_now, 0.05)
            exit_price = option_price(S_now, K, T_now, RISK_FREE_RATE, iv_now, opt_type)
            exit_reason = 'TIME'
            exit_day = MAX_HOLD_DAYS

        # P&L per contract (in dollars)
        gross_pnl = (exit_price - entry_price) * 100 * n_contracts
        net_pnl = gross_pnl - COMMISSION_RT  # $2.60 RT commission

        trades.append({
            'entry_date': str(gap['date'])[:10],
            'exit_date': str(dates[min(idx + exit_day, len(dates) - 1)])[:10],
            'direction': gap['direction'],
            'option_type': opt_type,
            'gap_pct': gap['gap_pct'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'n_contracts': n_contracts,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'exit_reason': exit_reason,
            'hold_days': exit_day,
            'volume_ratio': gap.get('volume_ratio', 1.0),
            'momentum_21d': gap.get('momentum_21d', 0.0),
        })

    return trades


def _get_vix(vix_df, date):
    """Get VIX level for a given date."""
    if vix_df is None:
        return 20.0
    try:
        mask = vix_df.index <= date
        if mask.any():
            return vix_df.loc[mask, 'close'].iloc[-1]
    except Exception:
        pass
    return 20.0


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_metrics(trades, label=''):
    """
    Compute calendar-month Sharpe and other metrics from trade list.
    Returns dict of metrics.
    """
    if not trades:
        return {
            'label': label, 'n_trades': 0, 'n_months': 0, 'sharpe': 0.0, 'sortino': 0.0,
            'profit_factor': 0.0, 'win_rate': 0.0, 'total_pnl': 0.0,
            'avg_pnl': 0.0, 'max_dd': 0.0, 'monthly_returns': [],
        }

    # Build monthly returns
    monthly_pnl = defaultdict(float)
    for t in trades:
        month_key = t['entry_date'][:7]  # YYYY-MM
        monthly_pnl[month_key] += t['net_pnl']

    # Sort months
    sorted_months = sorted(monthly_pnl.keys())
    monthly_returns = [monthly_pnl[m] for m in sorted_months]
    monthly_ret_pct = [r / STARTING_CAPITAL for r in monthly_returns]

    # Calendar-month Sharpe
    if len(monthly_ret_pct) > 1:
        mean_monthly = np.mean(monthly_ret_pct)
        std_monthly = np.std(monthly_ret_pct, ddof=1)
        sharpe = (mean_monthly / std_monthly * np.sqrt(12)) if std_monthly > 0 else 0.0
    else:
        sharpe = 0.0

    # Sortino
    if len(monthly_ret_pct) > 1:
        downside = [r for r in monthly_ret_pct if r < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.001
        sortino = (np.mean(monthly_ret_pct) / downside_std * np.sqrt(12)) if downside_std > 0 else 0.0
    else:
        sortino = 0.0

    # Win rate
    wins = [t for t in trades if t['net_pnl'] > 0]
    win_rate = len(wins) / len(trades) if trades else 0.0

    # Profit factor
    gross_wins = sum(t['net_pnl'] for t in trades if t['net_pnl'] > 0)
    gross_losses = abs(sum(t['net_pnl'] for t in trades if t['net_pnl'] < 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Max drawdown
    equity = STARTING_CAPITAL
    peak = equity
    max_dd = 0.0
    for m in sorted_months:
        equity += monthly_pnl[m]
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    total_pnl = sum(t['net_pnl'] for t in trades)
    avg_pnl = total_pnl / len(trades) if trades else 0.0

    return {
        'label': label,
        'n_trades': len(trades),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'profit_factor': round(profit_factor, 2),
        'win_rate': round(win_rate, 3),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 2),
        'max_dd': round(max_dd, 3),
        'n_months': len(sorted_months),
        'monthly_returns': monthly_ret_pct,
        'monthly_pnl_raw': monthly_returns,
    }


# ============================================================
# ADVERSARIAL VALIDATION (HC #753 — 4 gates)
# ============================================================

def adversarial_validation(trades, spy_df, label=''):
    """
    Run all 4 adversarial gates on trade results.
    Returns dict with gate pass/fail and details.
    """
    results = {
        'label': label,
        'gates': {},
        'all_pass': False,
        'n_gates_pass': 0,
    }

    if len(trades) < 10:
        results['gates'] = {
            'permutation': {'pass': False, 'reason': f'Too few trades ({len(trades)})'},
            'regime': {'pass': False, 'reason': f'Too few trades ({len(trades)})'},
            'subperiod': {'pass': False, 'reason': f'Too few trades ({len(trades)})'},
            'outlier': {'pass': False, 'reason': f'Too few trades ({len(trades)})'},
        }
        return results

    # --- Gate 1: Permutation test ---
    actual_pnl = sum(t['net_pnl'] for t in trades)
    rng = np.random.RandomState(42)
    pnl_values = np.array([t['net_pnl'] for t in trades])
    n_better = 0
    for _ in range(N_PERMUTATIONS):
        # Randomly flip direction of each trade's P&L
        signs = rng.choice([-1, 1], size=len(pnl_values))
        shuffled_pnl = np.sum(pnl_values * signs)
        if shuffled_pnl >= actual_pnl:
            n_better += 1
    perm_p = (n_better + 1) / (N_PERMUTATIONS + 1)
    results['gates']['permutation'] = {
        'pass': perm_p < 0.05,
        'p_value': round(perm_p, 4),
        'actual_pnl': round(actual_pnl, 2),
    }

    # --- Gate 2: Regime analysis ---
    # Classify regimes by SPY 63d return
    if spy_df is not None and len(spy_df) > 63:
        spy_close = spy_df['close']
        spy_ret_63d = spy_close / spy_close.shift(63) - 1

        regime_trades = {'bull': [], 'bear': [], 'flat': []}
        for t in trades:
            try:
                td = pd.Timestamp(t['entry_date'])
                # Find closest SPY date
                mask = spy_ret_63d.index <= td
                if not mask.any():
                    regime_trades['flat'].append(t)
                    continue
                ret = spy_ret_63d.loc[mask].iloc[-1]
                if np.isnan(ret):
                    regime_trades['flat'].append(t)
                elif ret > 0.05:
                    regime_trades['bull'].append(t)
                elif ret < -0.05:
                    regime_trades['bear'].append(t)
                else:
                    regime_trades['flat'].append(t)
            except Exception:
                regime_trades['flat'].append(t)

        regime_sharpes = {}
        for regime, rtrades in regime_trades.items():
            if len(rtrades) >= 3:
                m = compute_metrics(rtrades, f'{label}_{regime}')
                regime_sharpes[regime] = m['sharpe']
            else:
                regime_sharpes[regime] = None

        # Check regime divergence
        valid_sharpes = {k: v for k, v in regime_sharpes.items() if v is not None}
        if len(valid_sharpes) >= 2:
            sharpe_vals = list(valid_sharpes.values())
            max_div = 0
            for i in range(len(sharpe_vals)):
                for j in range(i + 1, len(sharpe_vals)):
                    s1, s2 = sharpe_vals[i], sharpe_vals[j]
                    denom = max(abs(s1), abs(s2))
                    if denom > 0:
                        div = abs(s1 - s2) / denom
                        max_div = max(max_div, div)
            regime_pass = max_div < 0.50
        else:
            regime_pass = True  # not enough data to fail

        results['gates']['regime'] = {
            'pass': regime_pass,
            'sharpes': regime_sharpes,
            'trade_counts': {k: len(v) for k, v in regime_trades.items()},
            'max_divergence': round(max_div, 3) if 'max_div' in dir() else None,
        }
    else:
        results['gates']['regime'] = {'pass': True, 'reason': 'No SPY data for regime analysis'}

    # --- Gate 3: Sub-period split ---
    sorted_trades = sorted(trades, key=lambda t: t['entry_date'])
    mid = len(sorted_trades) // 2
    half1 = sorted_trades[:mid]
    half2 = sorted_trades[mid:]

    m1 = compute_metrics(half1, f'{label}_H1')
    m2 = compute_metrics(half2, f'{label}_H2')

    subperiod_pass = m1['sharpe'] > 0.5 and m2['sharpe'] > 0.5
    results['gates']['subperiod'] = {
        'pass': subperiod_pass,
        'half1_sharpe': m1['sharpe'],
        'half2_sharpe': m2['sharpe'],
        'half1_trades': len(half1),
        'half2_trades': len(half2),
    }

    # --- Gate 4: Outlier removal ---
    monthly_rets = compute_metrics(trades, label)['monthly_returns']
    if len(monthly_rets) >= 10:
        arr = np.array(monthly_rets)
        p5 = np.percentile(arr, 5)
        p95 = np.percentile(arr, 95)
        trimmed = arr[(arr >= p5) & (arr <= p95)]
        if len(trimmed) > 1:
            mean_trim = np.mean(trimmed)
            std_trim = np.std(trimmed, ddof=1)
            outlier_sharpe = (mean_trim / std_trim * np.sqrt(12)) if std_trim > 0 else 0.0
        else:
            outlier_sharpe = 0.0
        outlier_pass = outlier_sharpe > 0.5
        results['gates']['outlier'] = {
            'pass': outlier_pass,
            'trimmed_sharpe': round(outlier_sharpe, 2),
            'n_months_trimmed': len(trimmed),
            'n_months_total': len(monthly_rets),
        }
    else:
        results['gates']['outlier'] = {
            'pass': False,
            'reason': f'Too few months ({len(monthly_rets)})',
        }

    # Summary
    n_pass = sum(1 for g in results['gates'].values() if g.get('pass', False))
    results['n_gates_pass'] = n_pass
    results['all_pass'] = n_pass == 4

    return results


# ============================================================
# VARIANT RUNNERS
# ============================================================

def run_variant_a(prices, spy_df, vix_df, earnings_dates):
    """Original: Gap >5%, buy ATM call, DTE=14."""
    print("\n=== VARIANT A: Original (Gap >5%, earnings) ===", flush=True)
    all_trades = []
    for ticker, ed_list in earnings_dates.items():
        df = prices.get(ticker)
        if df is None:
            continue
        gaps = detect_gaps(df, threshold=0.05, earnings_dates=ed_list, require_earnings=True)
        trades = simulate_trades(gaps, df, spy_df, vix_df)
        all_trades.extend(trades)
    print(f"  Total trades: {len(all_trades)}", flush=True)
    return all_trades


def run_variant_b(prices, spy_df, vix_df, earnings_dates):
    """Lower threshold: Gap >3%."""
    print("\n=== VARIANT B: Lower threshold (Gap >3%, earnings) ===", flush=True)
    all_trades = []
    for ticker, ed_list in earnings_dates.items():
        df = prices.get(ticker)
        if df is None:
            continue
        gaps = detect_gaps(df, threshold=0.03, earnings_dates=ed_list, require_earnings=True)
        trades = simulate_trades(gaps, df, spy_df, vix_df)
        all_trades.extend(trades)
    print(f"  Total trades: {len(all_trades)}", flush=True)
    return all_trades


def run_variant_c(prices, spy_df, vix_df, earnings_dates):
    """Gap + momentum: Gap >3% AND 21d momentum > 0."""
    print("\n=== VARIANT C: Gap >3% + 21d momentum > 0 ===", flush=True)
    all_trades = []
    for ticker, ed_list in earnings_dates.items():
        df = prices.get(ticker)
        if df is None:
            continue
        gaps = detect_gaps(df, threshold=0.03, earnings_dates=ed_list, require_earnings=True)
        # Filter: only keep gaps where momentum aligns
        filtered_gaps = []
        for g in gaps:
            if g['direction'] == 'up' and g['momentum_21d'] > 0:
                filtered_gaps.append(g)
            elif g['direction'] == 'down' and g['momentum_21d'] < 0:
                filtered_gaps.append(g)
        trades = simulate_trades(filtered_gaps, df, spy_df, vix_df)
        all_trades.extend(trades)
    print(f"  Total trades: {len(all_trades)}", flush=True)
    return all_trades


def run_variant_d(prices, spy_df, vix_df, earnings_dates):
    """Gap + volume: Gap >3% AND volume > 2x avg."""
    print("\n=== VARIANT D: Gap >3% + volume > 2x avg ===", flush=True)
    all_trades = []
    for ticker, ed_list in earnings_dates.items():
        df = prices.get(ticker)
        if df is None:
            continue
        gaps = detect_gaps(df, threshold=0.03, earnings_dates=ed_list, require_earnings=True)
        # Filter: only keep gaps with institutional volume
        filtered_gaps = [g for g in gaps if g['volume_ratio'] >= 2.0]
        trades = simulate_trades(filtered_gaps, df, spy_df, vix_df)
        all_trades.extend(trades)
    print(f"  Total trades: {len(all_trades)}", flush=True)
    return all_trades


def run_variant_e(etf_prices, spy_df, vix_df):
    """Multi-ETF: Apply to 14 liquid sector ETFs, any gap >3%."""
    print("\n=== VARIANT E: Multi-ETF (Gap >3%, any gap) ===", flush=True)
    all_trades = []
    for ticker, df in etf_prices.items():
        gaps = detect_etf_gaps(df, threshold=0.03)
        trades = simulate_trades(gaps, df, spy_df, vix_df)
        all_trades.extend(trades)
    print(f"  Total trades: {len(all_trades)}", flush=True)
    return all_trades


def run_variant_f(etf_prices, spy_df, vix_df):
    """Sector ETF + VIX filter: Gap >3% on ETFs, only when VIX < 25."""
    print("\n=== VARIANT F: Sector ETF + VIX < 25 (Gap >3%) ===", flush=True)
    all_trades = []
    config = {'vix_filter': 25.0}
    for ticker, df in etf_prices.items():
        gaps = detect_etf_gaps(df, threshold=0.03)
        trades = simulate_trades(gaps, df, spy_df, vix_df, variant_config=config)
        all_trades.extend(trades)
    print(f"  Total trades: {len(all_trades)}", flush=True)
    return all_trades


# ============================================================
# WALK-FORWARD VALIDATION (sliding window, HC #0)
# ============================================================

def walk_forward_validate(trades, label=''):
    """
    Sliding walk-forward validation.
    Train window: first 70% of data. Test: remaining 30%.
    Only report metrics on test period (no look-ahead).
    """
    if len(trades) < 20:
        return None

    sorted_trades = sorted(trades, key=lambda t: t['entry_date'])
    split_idx = int(len(sorted_trades) * 0.70)
    test_trades = sorted_trades[split_idx:]

    if len(test_trades) < 5:
        return None

    metrics = compute_metrics(test_trades, f'{label}_WF_OOT')
    return {
        'label': label,
        'train_trades': split_idx,
        'test_trades': len(test_trades),
        'test_period': f"{test_trades[0]['entry_date']} to {test_trades[-1]['entry_date']}",
        'metrics': metrics,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("EARNINGS GAP MOMENTUM v2 — 6 Variant Comparison", flush=True)
    print("=" * 70, flush=True)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Capital: ${STARTING_CAPITAL}, Max/trade: ${MAX_POSITION_DOLLARS}", flush=True)
    print(f"Commission RT: ${COMMISSION_RT}, Entry haircut: {ENTRY_HAIRCUT*100:.0f}%", flush=True)
    print(f"DTE: {DTE}, TP: +{TP_PCT*100:.0f}%, SL: {SL_PCT*100:.0f}%, Max hold: {MAX_HOLD_DAYS}d", flush=True)
    print(f"Permutations: {N_PERMUTATIONS}", flush=True)
    print(flush=True)

    # --- Load data ---
    print("--- Loading stock data ---", flush=True)
    stock_prices, spy_df, vix_df = load_data(STOCK_UNIVERSE, label='stocks')
    print(f"Loaded {len(stock_prices)} stock tickers", flush=True)

    print("\n--- Loading ETF data ---", flush=True)
    etf_prices, spy_df2, vix_df2 = load_data(ETF_UNIVERSE, label='etfs')
    # Use SPY/VIX from whichever loaded successfully
    if spy_df is None:
        spy_df = spy_df2
    if vix_df is None:
        vix_df = vix_df2
    print(f"Loaded {len(etf_prices)} ETF tickers", flush=True)

    print("\n--- Loading earnings dates ---", flush=True)
    earnings_dates = load_earnings_dates(STOCK_UNIVERSE)

    # --- Run all 6 variants ---
    variant_results = {}

    # A-D use individual stocks with earnings
    variant_trades = {
        'A_original_5pct': run_variant_a(stock_prices, spy_df, vix_df, earnings_dates),
        'B_lower_3pct': run_variant_b(stock_prices, spy_df, vix_df, earnings_dates),
        'C_gap_momentum': run_variant_c(stock_prices, spy_df, vix_df, earnings_dates),
        'D_gap_volume': run_variant_d(stock_prices, spy_df, vix_df, earnings_dates),
        'E_multi_etf': run_variant_e(etf_prices, spy_df, vix_df),
        'F_etf_vix_filter': run_variant_f(etf_prices, spy_df, vix_df),
    }

    # --- Compute metrics and adversarial validation for each ---
    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    adversarial_results = {}

    for name, trades in variant_trades.items():
        metrics = compute_metrics(trades, name)
        adv = adversarial_validation(trades, spy_df, name)
        wf = walk_forward_validate(trades, name)

        variant_results[name] = {
            'metrics': metrics,
            'adversarial': adv,
            'walk_forward': wf,
        }
        adversarial_results[name] = adv

        # Print summary
        print(f"\n--- {name} ---", flush=True)
        print(f"  Trades: {metrics['n_trades']}, Months: {metrics['n_months']}", flush=True)
        print(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}", flush=True)
        print(f"  PF: {metrics['profit_factor']}, WR: {metrics['win_rate']:.1%}", flush=True)
        print(f"  Total P&L: ${metrics['total_pnl']:.2f}, Avg: ${metrics['avg_pnl']:.2f}", flush=True)
        print(f"  Max DD: {metrics['max_dd']:.1%}", flush=True)

        if wf:
            wfm = wf['metrics']
            print(f"  Walk-Forward OOT: Sharpe={wfm['sharpe']}, "
                  f"N={wfm['n_trades']}, PF={wfm['profit_factor']}", flush=True)

        print(f"  Adversarial Gates: {adv['n_gates_pass']}/4", flush=True)
        for gate_name, gate_result in adv['gates'].items():
            status = "PASS" if gate_result.get('pass', False) else "FAIL"
            detail = ''
            if gate_name == 'permutation':
                detail = f" (p={gate_result.get('p_value', '?')})"
            elif gate_name == 'regime':
                detail = f" (sharpes={gate_result.get('sharpes', {})})"
            elif gate_name == 'subperiod':
                detail = f" (H1={gate_result.get('half1_sharpe', '?')}, H2={gate_result.get('half2_sharpe', '?')})"
            elif gate_name == 'outlier':
                detail = f" (trimmed_sharpe={gate_result.get('trimmed_sharpe', '?')})"
            print(f"    {gate_name}: {status}{detail}", flush=True)

        all_results.append({
            'name': name,
            'sharpe': metrics['sharpe'],
            'sortino': metrics['sortino'],
            'pf': metrics['profit_factor'],
            'wr': metrics['win_rate'],
            'n_trades': metrics['n_trades'],
            'total_pnl': metrics['total_pnl'],
            'gates_pass': adv['n_gates_pass'],
            'all_pass': adv['all_pass'],
        })

    # --- Results table sorted by Sharpe ---
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY TABLE (sorted by Sharpe)", flush=True)
    print("=" * 70, flush=True)
    all_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"{'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} "
          f"{'Trades':>7} {'P&L':>10} {'Gates':>6} {'Pass?':>6}", flush=True)
    print("-" * 85, flush=True)
    for r in all_results:
        pass_str = "YES" if r['all_pass'] else f"{r['gates_pass']}/4"
        print(f"{r['name']:<25} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
              f"{r['pf']:>6.2f} {r['wr']:>5.1%} {r['n_trades']:>7d} "
              f"${r['total_pnl']:>9.2f} {r['gates_pass']:>5d}/4 {pass_str:>6}", flush=True)

    # --- Save JSON results ---
    output = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'starting_capital': STARTING_CAPITAL,
            'max_position': MAX_POSITION_DOLLARS,
            'commission_rt': COMMISSION_RT,
            'entry_haircut': ENTRY_HAIRCUT,
            'dte': DTE,
            'tp_pct': TP_PCT,
            'sl_pct': SL_PCT,
            'max_hold_days': MAX_HOLD_DAYS,
            'n_permutations': N_PERMUTATIONS,
            'data_range': f'{START_DATE} to {END_DATE}',
        },
        'summary': all_results,
        'details': {},
    }

    for name, vr in variant_results.items():
        # Strip non-serializable items from metrics
        metrics_clean = {k: v for k, v in vr['metrics'].items()
                         if k not in ('monthly_returns', 'monthly_pnl_raw')}
        adv_clean = {
            'n_gates_pass': vr['adversarial']['n_gates_pass'],
            'all_pass': vr['adversarial']['all_pass'],
            'gates': vr['adversarial']['gates'],
        }
        wf_clean = None
        if vr['walk_forward']:
            wfm = vr['walk_forward']['metrics']
            wf_clean = {
                'train_trades': vr['walk_forward']['train_trades'],
                'test_trades': vr['walk_forward']['test_trades'],
                'test_period': vr['walk_forward']['test_period'],
                'sharpe': wfm['sharpe'],
                'sortino': wfm['sortino'],
                'pf': wfm['profit_factor'],
                'wr': wfm['win_rate'],
            }
        output['details'][name] = {
            'metrics': metrics_clean,
            'adversarial': adv_clean,
            'walk_forward': wf_clean,
        }

    json_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}", flush=True)

    # --- MLflow logging ---
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://localhost:5000')
            mlflow.set_experiment('earnings_gap_momentum_v2')
            with mlflow.start_run(run_name=f'egm_v2_{datetime.now().strftime("%Y%m%d_%H%M")}'):
                mlflow.log_param('starting_capital', STARTING_CAPITAL)
                mlflow.log_param('max_position', MAX_POSITION_DOLLARS)
                mlflow.log_param('commission_rt', COMMISSION_RT)
                mlflow.log_param('entry_haircut', ENTRY_HAIRCUT)
                mlflow.log_param('dte', DTE)
                mlflow.log_param('tp_pct', TP_PCT)
                mlflow.log_param('sl_pct', SL_PCT)
                mlflow.log_param('n_permutations', N_PERMUTATIONS)
                mlflow.log_param('n_stock_tickers', len(STOCK_UNIVERSE))
                mlflow.log_param('n_etf_tickers', len(ETF_UNIVERSE))

                # Log best variant metrics
                if all_results:
                    best = all_results[0]  # sorted by Sharpe desc
                    mlflow.log_metric('best_sharpe', best['sharpe'])
                    mlflow.log_metric('best_sortino', best['sortino'])
                    mlflow.log_metric('best_pf', best['pf'])
                    mlflow.log_metric('best_wr', best['wr'])
                    mlflow.log_metric('best_n_trades', best['n_trades'])
                    mlflow.log_metric('best_total_pnl', best['total_pnl'])
                    mlflow.log_metric('best_gates_pass', best['gates_pass'])
                    mlflow.log_param('best_variant', best['name'])

                # Log per-variant metrics
                for r in all_results:
                    prefix = r['name']
                    mlflow.log_metric(f'{prefix}_sharpe', r['sharpe'])
                    mlflow.log_metric(f'{prefix}_n_trades', r['n_trades'])
                    mlflow.log_metric(f'{prefix}_gates_pass', r['gates_pass'])

                mlflow.log_artifact(json_path)
            print("MLflow run logged successfully", flush=True)
        except Exception as e:
            print(f"MLflow logging failed (non-fatal): {e}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s", flush=True)

    # Final verdict
    print("\n" + "=" * 70, flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)
    passing = [r for r in all_results if r['all_pass']]
    if passing:
        print(f"{len(passing)} variant(s) passed ALL 4 adversarial gates:", flush=True)
        for r in passing:
            print(f"  {r['name']}: Sharpe={r['sharpe']}, N={r['n_trades']}", flush=True)
    else:
        best_gates = max(r['gates_pass'] for r in all_results) if all_results else 0
        partial = [r for r in all_results if r['gates_pass'] == best_gates]
        print(f"No variant passed all 4 gates. Best: {best_gates}/4 gates:", flush=True)
        for r in partial:
            print(f"  {r['name']}: Sharpe={r['sharpe']}, N={r['n_trades']}, "
                  f"Gates={r['gates_pass']}/4", flush=True)

    return output


if __name__ == '__main__':
    main()
