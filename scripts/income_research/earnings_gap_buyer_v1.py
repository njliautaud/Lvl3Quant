#!/usr/bin/env python3
"""
Earnings Gap Buyer — PEAD Strategy Backtest v1
================================================
Buy stocks (or options) in the gap direction after ACTUAL earnings
announcements, exploiting Post-Earnings Announcement Drift.

CRITICAL FIX vs pead_drift_v1.py:
  - Uses ACTUAL earnings dates from yfinance (not month heuristics)
  - Triple-validates: earnings date + gap 5%+ + volume > 1.5x avg
  - Avoids the 56% false-positive problem from PEAD v1

Entry signals tested:
  A) "gap_5pct": Earnings gap >= 5%
  B) "gap_7pct": Earnings gap >= 7%
  C) "gap_10pct": Earnings gap >= 10%

Trade setup:
  - Equity sim: buy at open in gap direction, hold N days
  - Options: BS-priced debit spread (calls for gap-up, puts for gap-down)
  - Hold periods: 1, 2, 5 trading days
  - Size: $250/trade, max 2 concurrent
  - Start: $10K

INLINE quality gates (HC #705):
  - Permutation test: 200 shuffles with random DATE entry (not return shuffling)
  - Regime test: SPY green/red/flat, reject if gap > 0.50
  - Sub-period consistency
  - Outlier removal
  - Ticker concentration

Universe: 30 large-cap stocks, 2019-2026
"""

import sys, json, warnings, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "earnings_gap_buyer_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

STARTING_CAPITAL       = 10_000
RISK_PER_TRADE         = 250       # $250 per trade
MAX_CONCURRENT         = 2
EQUITY_SLIPPAGE_PCT    = 0.001     # 0.1% slippage on open
OPTIONS_COMMISSION_LEG = 0.65      # $0.65 per leg (Robinhood)
RISK_FREE_RATE         = 0.04
N_PERMUTATIONS         = 200

TICKERS = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','NFLX','AMD','INTC',
    'BA','DIS','SBUX','HD','LOW','MCD','NKE','COST','WMT',
    'JPM','GS','BAC','MS','JNJ','PG','KO','UNH','ABBV','CRM','NOW',
]

GAP_THRESHOLDS = {
    'gap_5pct':  0.05,
    'gap_7pct':  0.07,
    'gap_10pct': 0.10,
}

HOLD_PERIODS = [1, 2, 5]

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def fetch_prices(tickers, cache_path):
    """Fetch daily OHLCV from yfinance with caching."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Loaded cached prices: {len(df)} rows, {df['ticker'].nunique()} tickers")
        return df

    import yfinance as yf
    print(f"  Downloading prices for {len(tickers)} tickers...")
    all_frames = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start='2018-12-01', end='2026-07-15',
                               progress=False, auto_adjust=True)
            if len(data) > 100:
                data = data.reset_index()
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
                data['ticker'] = ticker
                data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                     'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                data.columns = [c.lower() if isinstance(c, str) else c for c in data.columns]
                all_frames.append(data[['date','open','high','low','close','volume','ticker']])
                print(f"    {ticker}: {len(data)} days")
        except Exception as e:
            print(f"    {ticker}: error - {e}")
        time.sleep(0.2)

    df = pd.concat(all_frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df.to_parquet(cache_path, index=False)
    print(f"  Saved {len(df)} rows to cache")
    return df


def fetch_spy(cache_path):
    """Fetch SPY for regime classification."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    import yfinance as yf
    spy = yf.download('SPY', start='2018-12-01', end='2026-07-15',
                       progress=False, auto_adjust=True).reset_index()
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy.rename(columns={'Date': 'date', 'Close': 'close'}, inplace=True)
    spy.columns = [c.lower() if isinstance(c, str) else c for c in spy.columns]
    spy = spy[['date','close']].copy()
    spy['date'] = pd.to_datetime(spy['date'])
    spy.to_parquet(cache_path, index=False)
    return spy


def fetch_earnings_dates(tickers, cache_path):
    """
    Fetch ACTUAL earnings dates from yfinance.
    Returns dict: ticker -> set of earnings dates (normalized to trading day).

    CRITICAL: This is the fix for the pead_drift_v1 false-positive problem.
    Old approach used month heuristics (EARNINGS_MONTHS) which caught ANY gap
    in those months. This uses yfinance's actual earnings calendar.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Loaded cached earnings dates: {len(df)} events across {df['ticker'].nunique()} tickers")
        result = {}
        for ticker in df['ticker'].unique():
            dates = pd.to_datetime(df[df['ticker'] == ticker]['earnings_date']).dt.normalize()
            result[ticker] = set(dates)
        return result

    import yfinance as yf
    print(f"  Fetching earnings dates for {len(tickers)} tickers...")
    all_earnings = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            # Try multiple approaches to get earnings dates
            edates = None

            # Approach 1: get_earnings_dates (most reliable)
            try:
                ed = t.get_earnings_dates(limit=60)
                if ed is not None and len(ed) > 0:
                    edates = ed.index.tolist()
            except Exception:
                pass

            # Approach 2: earnings_dates attribute
            if edates is None or len(edates) == 0:
                try:
                    ed = t.earnings_dates
                    if ed is not None and len(ed) > 0:
                        edates = ed.index.tolist()
                except Exception:
                    pass

            # Approach 3: quarterly_earnings
            if edates is None or len(edates) == 0:
                try:
                    qe = t.quarterly_earnings
                    if qe is not None and len(qe) > 0:
                        edates = qe.index.tolist()
                except Exception:
                    pass

            if edates and len(edates) > 0:
                for d in edates:
                    try:
                        dt = pd.Timestamp(d).normalize()
                        if pd.Timestamp('2019-01-01') <= dt <= pd.Timestamp('2026-07-15'):
                            all_earnings.append({'ticker': ticker, 'earnings_date': dt})
                    except Exception:
                        continue
                print(f"    {ticker}: {sum(1 for e in all_earnings if e['ticker']==ticker)} earnings dates")
            else:
                print(f"    {ticker}: no earnings dates found (will use fallback)")
        except Exception as e:
            print(f"    {ticker}: error - {e}")
        time.sleep(0.3)

    if all_earnings:
        df = pd.DataFrame(all_earnings)
        df.to_parquet(cache_path, index=False)
        print(f"  Saved {len(df)} earnings events to cache")
        result = {}
        for ticker in df['ticker'].unique():
            dates = pd.to_datetime(df[df['ticker'] == ticker]['earnings_date']).dt.normalize()
            result[ticker] = set(dates)
        return result
    else:
        print("  WARNING: No earnings dates fetched. Will use fallback detection.")
        return {}


def fallback_earnings_detection(df_ticker):
    """
    FALLBACK: Detect likely earnings days when yfinance earnings dates unavailable.
    Uses triple-filter: gap > 5% AND volume > 3x avg AND post-gap realized vol spike.

    This is MUCH more conservative than the old month-based heuristic.
    """
    df = df_ticker.sort_values('date').copy()
    df['prev_close'] = df['close'].shift(1)
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['vol_avg_20d'] = df['volume'].rolling(20).mean()

    # Realized vol: 5-day realized vol after vs trailing 30-day
    df['fwd_rvol_5d'] = df['close'].pct_change().rolling(5).std().shift(-5)
    df['trail_rvol_30d'] = df['close'].pct_change().rolling(30).std()

    likely_earnings = set()
    for i in range(30, len(df) - 5):
        row = df.iloc[i]
        gap = abs(row['gap_pct']) if not pd.isna(row['gap_pct']) else 0
        vol = row['volume']
        vol_avg = row['vol_avg_20d']
        fwd_rvol = row['fwd_rvol_5d']
        trail_rvol = row['trail_rvol_30d']

        if pd.isna(vol_avg) or pd.isna(fwd_rvol) or pd.isna(trail_rvol):
            continue

        # Triple filter: big gap + huge volume + vol spike after
        if (gap >= 0.05
            and vol_avg > 0 and vol >= 3.0 * vol_avg
            and trail_rvol > 0 and fwd_rvol >= trail_rvol):
            likely_earnings.add(pd.Timestamp(row['date']).normalize())

    return likely_earnings


# ═════════════════════════════════════════════════════════════════════════════
# EARNINGS EVENT DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_earnings_gaps(all_data, earnings_dates_dict):
    """
    Detect ACTUAL earnings gaps with cross-validation.

    For each ticker:
      1. Check if date is within 1 trading day of a known earnings date
      2. Verify gap >= threshold
      3. Verify volume > 1.5x 20-day average

    Returns list of earnings gap events.
    """
    events = []
    tickers = all_data['ticker'].unique()
    false_positive_count = 0
    validated_count = 0

    for ticker in tickers:
        df = all_data[all_data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        if len(df) < 30:
            continue

        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        df['vol_avg_20d'] = df['volume'].rolling(20).mean()

        # Get earnings dates for this ticker
        known_dates = earnings_dates_dict.get(ticker, set())

        # If no known dates, use fallback
        if not known_dates:
            known_dates = fallback_earnings_detection(df)
            if known_dates:
                print(f"    {ticker}: using fallback detection ({len(known_dates)} likely earnings)")

        for i in range(1, len(df)):
            row = df.iloc[i]
            date = pd.Timestamp(row['date']).normalize()

            # Check 1: Is this within 1 trading day of an earnings date?
            is_earnings = False
            for ed in known_dates:
                delta = abs((date - ed).days)
                if delta <= 2:  # within 2 calendar days (accounts for weekend)
                    is_earnings = True
                    break

            if not is_earnings:
                continue

            # Check 2: Gap size
            gap = row['gap_pct']
            if pd.isna(gap):
                continue
            abs_gap = abs(gap)

            # Check 3: Volume confirmation (> 1.5x average)
            vol = row['volume']
            vol_avg = row['vol_avg_20d']
            if pd.isna(vol_avg) or vol_avg <= 0:
                continue
            vol_ratio = vol / vol_avg

            if abs_gap < 0.03:  # minimum gap to even consider (below any threshold)
                continue

            if vol_ratio < 1.5:
                false_positive_count += 1
                continue

            validated_count += 1
            direction = 'long' if gap > 0 else 'short'

            events.append({
                'ticker': ticker,
                'date': date,
                'gap_pct': gap,
                'abs_gap_pct': abs_gap,
                'direction': direction,
                'volume': vol,
                'vol_avg_20d': vol_avg,
                'vol_ratio': vol_ratio,
                'prev_close': row['prev_close'],
                'open_price': row['open'],
                'idx': i,
            })

    print(f"  Detected {validated_count} validated earnings gaps "
          f"({false_positive_count} rejected on volume filter)")
    return events


# ═════════════════════════════════════════════════════════════════════════════
# EQUITY BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

def run_equity_backtest(all_data, events, gap_threshold, hold_days, spy_regime):
    """
    Simulate buying/shorting after earnings gap.
    Gap UP -> buy at open, sell N days later
    Gap DOWN -> short at open, cover N days later
    """
    trades = []
    filtered_events = [e for e in events if e['abs_gap_pct'] >= gap_threshold]

    for ev in filtered_events:
        ticker = ev['ticker']
        df = all_data[all_data['ticker'] == ticker].sort_values('date').reset_index(drop=True)

        # Find the index of the event day
        date_matches = df[df['date'] == ev['date']]
        if len(date_matches) == 0:
            continue
        idx = date_matches.index[0]

        # Entry: open of event day (gap has already happened)
        entry_price = df.iloc[idx]['open']
        exit_idx = min(idx + hold_days, len(df) - 1)
        if exit_idx <= idx:
            continue
        exit_price = df.iloc[exit_idx]['close']

        # Apply slippage
        if ev['direction'] == 'long':
            entry_price *= (1 + EQUITY_SLIPPAGE_PCT)
            exit_price  *= (1 - EQUITY_SLIPPAGE_PCT)
            ret = (exit_price - entry_price) / entry_price
        else:  # short
            entry_price *= (1 - EQUITY_SLIPPAGE_PCT)
            exit_price  *= (1 + EQUITY_SLIPPAGE_PCT)
            ret = (entry_price - exit_price) / entry_price

        entry_dt = pd.Timestamp(ev['date']).normalize()
        exit_dt = pd.Timestamp(df.iloc[exit_idx]['date']).normalize()
        regime = spy_regime.get(entry_dt, 'unknown')

        trades.append({
            'ticker': ticker,
            'entry_date': entry_dt,
            'exit_date': exit_dt,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return_pct': ret,
            'direction': ev['direction'],
            'gap_pct': ev['gap_pct'],
            'vol_ratio': ev['vol_ratio'],
            'regime': regime,
            'hold_days': hold_days,
        })

    return trades


# ═════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES OPTIONS PRICING
# ═════════════════════════════════════════════════════════════════════════════

def bs_call_price(S, K, T, sigma, r=0.04):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, sigma, r=0.04):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def estimate_post_earnings_iv(ticker, date, prices_df):
    """
    Post-earnings IV is CRUSHED. Estimate as:
    - Pre-earnings: 30d realized vol * 1.5 (IV premium into earnings)
    - Post-earnings: immediate crush to ~0.7x pre-earnings IV
    """
    hist = prices_df[(prices_df['ticker'] == ticker) & (prices_df['date'] <= date)]
    if len(hist) < 35:
        return 0.40, 0.28  # default pre/post
    rets = hist['close'].pct_change().dropna().tail(30)
    rv = rets.std() * np.sqrt(252)
    iv_pre = max(rv * 1.5, 0.20)   # elevated into earnings
    iv_post = iv_pre * 0.70         # IV crush post-earnings
    return min(iv_pre, 2.0), min(iv_post, 1.5)


def estimate_options_pnl(trades, all_data):
    """
    For each trade, estimate debit spread P&L.
    Gap UP -> call debit spread (buy ATM call, sell OTM call)
    Gap DOWN -> put debit spread (buy ATM put, sell OTM put)
    """
    options_trades = []
    for t in trades:
        ticker = t['ticker']
        entry_date = t['entry_date']
        S = t['entry_price']
        S_exit = t['exit_price']
        hold = t['hold_days']
        direction = t['direction']

        iv_pre, iv_post = estimate_post_earnings_iv(ticker, entry_date, all_data)
        # We enter AFTER earnings -> IV is already crushed
        iv_entry = iv_post
        iv_exit = iv_post * 0.95  # slight further decay

        spread_pct = 0.05  # 5% wide spread
        T_entry = 21 / 252  # ~21 DTE (weekly options, 3 weeks out)
        T_exit = hold / 252

        if direction == 'long':
            # Call debit spread
            K_long = S             # ATM call
            K_short = S * (1 + spread_pct)  # OTM call
            long_entry = bs_call_price(S, K_long, T_entry, iv_entry)
            short_entry = bs_call_price(S, K_short, T_entry, iv_entry)
            debit = long_entry - short_entry

            T_remain = max(T_entry - T_exit, 1/252)
            long_exit = bs_call_price(S_exit, K_long, T_remain, iv_exit)
            short_exit = bs_call_price(S_exit, K_short, T_remain, iv_exit)
            exit_val = long_exit - short_exit
        else:
            # Put debit spread
            K_long = S             # ATM put
            K_short = S * (1 - spread_pct)  # OTM put
            long_entry = bs_put_price(S, K_long, T_entry, iv_entry)
            short_entry = bs_put_price(S, K_short, T_entry, iv_entry)
            debit = long_entry - short_entry

            T_remain = max(T_entry - T_exit, 1/252)
            long_exit = bs_put_price(S_exit, K_long, T_remain, iv_exit)
            short_exit = bs_put_price(S_exit, K_short, T_remain, iv_exit)
            exit_val = long_exit - short_exit

        if debit <= 0:
            continue

        commission = 4 * OPTIONS_COMMISSION_LEG  # 4 legs (open + close)
        n_contracts = max(1, int(RISK_PER_TRADE / (debit * 100)))
        n_contracts = min(n_contracts, 5)

        pnl_per_contract = (exit_val - debit) * 100
        total_pnl = pnl_per_contract * n_contracts - commission

        options_trades.append({
            **t,
            'iv_entry': iv_entry,
            'iv_pre_earnings': iv_pre,
            'debit_paid': debit,
            'exit_value': exit_val,
            'spread_width': S * spread_pct,
            'n_contracts': n_contracts,
            'pnl_per_contract': pnl_per_contract,
            'total_pnl': total_pnl,
            'commission': commission,
            'options_return_pct': total_pnl / (debit * 100 * n_contracts) if debit > 0 else 0,
        })

    return options_trades


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def simulate_portfolio(trades_list, starting_capital=STARTING_CAPITAL):
    if not trades_list:
        return [], []

    trades = sorted(trades_list, key=lambda t: t['entry_date'])
    capital = starting_capital
    open_positions = []
    executed = []
    equity_curve = [{'date': trades[0]['entry_date'], 'equity': capital}]

    for trade in trades:
        still_open = []
        for pos in open_positions:
            if trade['entry_date'] >= pos['exit_date']:
                capital += pos.get('total_pnl', pos['return_pct'] * RISK_PER_TRADE)
            else:
                still_open.append(pos)
        open_positions = still_open

        if len(open_positions) >= MAX_CONCURRENT:
            continue

        cost = trade.get('debit_paid', 0) * 100 * trade.get('n_contracts', 1) if 'debit_paid' in trade else RISK_PER_TRADE
        if capital < cost:
            continue

        open_positions.append(trade)
        executed.append(trade)
        equity_curve.append({'date': trade['entry_date'], 'equity': capital})

    for pos in open_positions:
        capital += pos.get('total_pnl', pos['return_pct'] * RISK_PER_TRADE)

    if executed:
        equity_curve.append({'date': executed[-1]['exit_date'], 'equity': capital})

    return equity_curve, executed


# ═════════════════════════════════════════════════════════════════════════════
# QUALITY GATES (ALL INLINE per HC #705)
# ═════════════════════════════════════════════════════════════════════════════

def classify_spy_regime(spy_df):
    """Classify each day as green/red/flat based on SPY daily return."""
    spy = spy_df.sort_values('date').copy()
    spy['ret'] = spy['close'].pct_change()
    regime = {}
    for _, row in spy.iterrows():
        dt = pd.Timestamp(row['date']).normalize()
        if pd.isna(row['ret']):
            regime[dt] = 'flat'
        elif row['ret'] > 0.003:
            regime[dt] = 'green'
        elif row['ret'] < -0.003:
            regime[dt] = 'red'
        else:
            regime[dt] = 'flat'
    return regime


def regime_agnostic_test(trades):
    """
    R1: Regime-agnostic validation.
    Reject if |Sharpe_green - Sharpe_red| / max(...) > 0.50
    """
    if not trades:
        return {'pass': False, 'reason': 'no trades'}

    df = pd.DataFrame(trades)
    results = {}
    for regime in ['green', 'red', 'flat']:
        subset = df[df['regime'] == regime]
        if len(subset) < 3:
            results[regime] = {'sharpe': 0, 'n': 0, 'wr': 0, 'avg_ret': 0}
            continue
        rets = subset['return_pct']
        avg_hold = subset['hold_days'].mean()
        sharpe = rets.mean() / rets.std() * np.sqrt(252 / max(avg_hold, 1)) if rets.std() > 0 else 0
        results[regime] = {
            'sharpe': round(sharpe, 3),
            'n': len(subset),
            'wr': round((rets > 0).mean(), 3),
            'avg_ret': round(rets.mean() * 100, 3),
        }

    sg = results.get('green', {}).get('sharpe', 0)
    sr = results.get('red', {}).get('sharpe', 0)
    denom = max(abs(sg), abs(sr), 0.001)
    gap = abs(sg - sr) / denom

    passed = gap <= 0.50
    return {
        'pass': passed,
        'gap': round(gap, 3),
        'regimes': results,
        'note': 'PASS' if passed else f'FAIL: regime gap={gap:.3f} > 0.50',
    }


def permutation_test_date_shuffle(trades, all_data, n_perms=N_PERMUTATIONS):
    """
    CRITICAL BUG FIX: Random DATE entry permutation test.

    Instead of shuffling returns (which preserves cross-sectional structure
    and can be biased), we:
      1. Compute observed mean return from actual trade entries
      2. For each permutation: randomly pick N dates from the universe of
         all trading dates, compute forward returns, get mean
      3. p-value = fraction of permutations with mean >= observed

    This tests: "Could we get the same returns by entering on random dates?"
    """
    if len(trades) < 10:
        return {'pass': False, 'p_value': 1.0, 'reason': 'too few trades'}

    df = pd.DataFrame(trades)
    obs_mean = df['return_pct'].mean()
    n_trades = len(df)

    # Build universe of all possible (ticker, date, hold) entries
    # For each trade we know the hold period -- use the most common one
    hold_days = int(df['hold_days'].mode().iloc[0])

    # Build forward return lookup per ticker
    fwd_returns = {}
    tickers = df['ticker'].unique()
    for ticker in tickers:
        tdf = all_data[all_data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        if len(tdf) < hold_days + 10:
            continue
        dates = tdf['date'].values
        opens = tdf['open'].values
        closes = tdf['close'].values
        for i in range(len(tdf) - hold_days):
            entry_p = opens[i] * (1 + EQUITY_SLIPPAGE_PCT)
            exit_p = closes[min(i + hold_days, len(tdf) - 1)] * (1 - EQUITY_SLIPPAGE_PCT)
            # For simplicity, compute long return (direction doesn't matter for null)
            ret = (exit_p - entry_p) / entry_p
            fwd_returns[(ticker, pd.Timestamp(dates[i]).normalize())] = ret

    if len(fwd_returns) < n_trades * 2:
        # Fallback to simple return shuffling if not enough data
        return _permutation_test_simple(trades, n_perms)

    all_keys = list(fwd_returns.keys())
    all_rets_arr = np.array([fwd_returns[k] for k in all_keys])
    n_available = len(all_keys)

    rng = np.random.RandomState(42)
    count_ge = 0
    for _ in range(n_perms):
        # Randomly sample N dates (with replacement, mimicking random entry)
        idxs = rng.randint(0, n_available, size=n_trades)
        perm_mean = all_rets_arr[idxs].mean()
        if perm_mean >= obs_mean:
            count_ge += 1

    p_value = count_ge / n_perms
    return {
        'pass': p_value < 0.05,
        'p_value': round(p_value, 4),
        'observed_mean_pct': round(obs_mean * 100, 4),
        'n_perms': n_perms,
        'method': 'random_date_entry',
        'n_universe_dates': n_available,
    }


def _permutation_test_simple(trades, n_perms):
    """Fallback: simple return shuffling (less rigorous)."""
    df = pd.DataFrame(trades)
    rets = df['return_pct'].values
    obs_mean = rets.mean()
    rng = np.random.RandomState(42)
    count_ge = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(rets)
        if shuffled.mean() >= obs_mean:
            count_ge += 1
    p_value = count_ge / n_perms
    return {
        'pass': p_value < 0.05,
        'p_value': round(p_value, 4),
        'observed_mean_pct': round(obs_mean * 100, 4),
        'n_perms': n_perms,
        'method': 'return_shuffle_fallback',
    }


def adversarial_tests(trades):
    """
    HC #705 adversarial checks:
    1. Sub-period consistency (split in half by time)
    2. Outlier removal (drop top/bottom 5%)
    3. Ticker concentration (no single ticker > 30% of P&L)
    """
    if len(trades) < 15:
        return {'overall_pass': False, 'reason': f'too few trades ({len(trades)})'}

    df = pd.DataFrame(trades)
    results = {}
    warnings_list = []

    # 1. Sub-period consistency
    df_sorted = df.sort_values('entry_date')
    mid = len(df_sorted) // 2
    first_half = df_sorted.iloc[:mid]['return_pct']
    second_half = df_sorted.iloc[mid:]['return_pct']
    wr1 = (first_half > 0).mean()
    wr2 = (second_half > 0).mean()
    avg1 = first_half.mean()
    avg2 = second_half.mean()
    sub_pass = avg1 > 0 and avg2 > 0
    if not sub_pass:
        warnings_list.append(f"SUB-PERIOD FAIL: 1st half avg={avg1*100:.2f}%, 2nd half avg={avg2*100:.2f}%")
    results['sub_period'] = {
        'pass': sub_pass,
        'first_half': {'wr': round(wr1, 3), 'avg_ret': round(avg1*100, 3), 'n': len(first_half)},
        'second_half': {'wr': round(wr2, 3), 'avg_ret': round(avg2*100, 3), 'n': len(second_half)},
    }

    # 2. Outlier removal
    rets = df['return_pct'].values
    p5, p95 = np.percentile(rets, [5, 95])
    trimmed = rets[(rets >= p5) & (rets <= p95)]
    outlier_pass = trimmed.mean() > 0
    if not outlier_pass:
        warnings_list.append(f"OUTLIER FAIL: trimmed mean={trimmed.mean()*100:.2f}% (full={rets.mean()*100:.2f}%)")
    results['outlier_removal'] = {
        'pass': outlier_pass,
        'full_mean_pct': round(rets.mean()*100, 3),
        'trimmed_mean_pct': round(trimmed.mean()*100, 3),
        'n_removed': len(rets) - len(trimmed),
    }

    # 3. Ticker concentration
    ticker_pnl = df.groupby('ticker')['return_pct'].sum()
    total_positive = ticker_pnl[ticker_pnl > 0].sum()
    if total_positive > 0:
        max_conc = ticker_pnl.max() / total_positive
    else:
        max_conc = 1.0
    conc_pass = max_conc < 0.30
    top_tickers = ticker_pnl.nlargest(5)
    if not conc_pass:
        worst = ticker_pnl.idxmax()
        warnings_list.append(f"TICKER CONCENTRATION FAIL: {worst} = {max_conc:.1%} of gains")
    results['ticker_concentration'] = {
        'pass': conc_pass,
        'max_concentration': round(max_conc, 3),
        'top_5': {k: round(v*100, 2) for k, v in top_tickers.items()},
    }

    overall = all(r.get('pass', False) for r in results.values())
    results['overall_pass'] = overall
    results['warnings'] = warnings_list
    return results


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(trades, label=''):
    if not trades:
        return {'label': label, 'n_trades': 0}

    df = pd.DataFrame(trades)
    rets = df['return_pct']
    n = len(rets)
    avg_ret = rets.mean()
    std_ret = rets.std()
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    wr = (rets > 0).mean()

    avg_hold = df['hold_days'].mean()
    ann_factor = np.sqrt(252 / max(avg_hold, 1))

    sharpe = avg_ret / std_ret * ann_factor if std_ret > 0 else 0
    downside = rets[rets < 0].std()
    sortino = avg_ret / downside * ann_factor if downside > 0 and len(rets[rets < 0]) > 2 else 0

    gross_wins = wins.sum() if len(wins) > 0 else 0
    gross_losses = abs(losses.sum()) if len(losses) > 0 else 0.001
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

    cum_rets = (1 + rets).cumprod()
    running_max = cum_rets.cummax()
    max_dd = (cum_rets / running_max - 1).min() * 100

    # Direction breakdown
    dir_breakdown = {}
    for d in ['long', 'short']:
        sub = df[df['direction'] == d] if 'direction' in df.columns else pd.DataFrame()
        if len(sub) > 0:
            dir_breakdown[d] = {
                'n': len(sub),
                'wr': round((sub['return_pct'] > 0).mean(), 3),
                'avg_ret': round(sub['return_pct'].mean() * 100, 3),
            }

    return {
        'label': label,
        'n_trades': n,
        'win_rate': round(wr, 4),
        'avg_return_pct': round(avg_ret * 100, 3),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 3),
        'avg_win_pct': round(avg_win * 100, 3),
        'avg_loss_pct': round(avg_loss * 100, 3),
        'max_drawdown_pct': round(max_dd, 2),
        'best_trade_pct': round(rets.max() * 100, 2),
        'worst_trade_pct': round(rets.min() * 100, 2),
        'direction_breakdown': dir_breakdown,
    }


def per_ticker_breakdown(trades):
    df = pd.DataFrame(trades)
    results = {}
    for ticker, grp in df.groupby('ticker'):
        rets = grp['return_pct']
        results[ticker] = {
            'n': len(grp),
            'wr': round((rets > 0).mean(), 3),
            'avg_ret': round(rets.mean() * 100, 3),
            'total_ret': round(rets.sum() * 100, 2),
            'avg_gap': round(grp['gap_pct'].mean() * 100, 2) if 'gap_pct' in grp.columns else 0,
        }
    return dict(sorted(results.items(), key=lambda x: -x[1]['total_ret']))


def validate_earnings_detection(events, earnings_dates_dict):
    """
    ANTI-FALSE-POSITIVE CHECK: Compare our detected events to known earnings.
    Print warnings if detection looks suspicious.
    """
    print("\n  ── Earnings Detection Validation ──")
    for ticker in sorted(set(e['ticker'] for e in events)):
        ticker_events = [e for e in events if e['ticker'] == ticker]
        known = earnings_dates_dict.get(ticker, set())
        n_events = len(ticker_events)
        n_known = len(known)

        # Sanity: large-cap stocks have ~4 earnings/year, so 2019-2026 = ~28-32
        if n_events > n_known * 1.5 and n_known > 0:
            print(f"  WARNING: {ticker} has {n_events} gap events but only {n_known} "
                  f"known earnings - possible false positives!")
        elif n_events > 40:
            print(f"  WARNING: {ticker} has {n_events} events (expected ~28 for 7 years)")
        else:
            pass  # looks reasonable

    # Summary
    total = len(events)
    unique_tickers = len(set(e['ticker'] for e in events))
    avg_per_ticker = total / max(unique_tickers, 1)
    print(f"  Total events: {total}, Unique tickers: {unique_tickers}, "
          f"Avg/ticker: {avg_per_ticker:.1f}")
    if avg_per_ticker > 35:
        print("  *** CRITICAL WARNING: Average events/ticker > 35 suggests false positives! ***")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("EARNINGS GAP BUYER — PEAD STRATEGY BACKTEST v1")
    print("  Uses ACTUAL earnings dates (fixes pead_drift_v1 false-positive bug)")
    print("=" * 80)

    # ── Load data ──
    print("\n[1/7] Loading price data...")
    # Try to reuse cached prices from other backtests
    for cache_dir in ['oversold_bounce_v1', 'momentum_breakout_v1', 'pead_drift_v1']:
        existing = ROOT / "output" / cache_dir / "prices_cache.parquet"
        if existing.exists():
            prices_cache = existing
            break
    else:
        prices_cache = OUTPUT / "prices_cache.parquet"

    spy_cache = OUTPUT / "spy_cache.parquet"
    for cache_dir in ['oversold_bounce_v1', 'momentum_breakout_v1', 'pead_drift_v1']:
        existing = ROOT / "output" / cache_dir / "spy_cache.parquet"
        if existing.exists():
            spy_cache = existing
            break

    all_data = fetch_prices(TICKERS, prices_cache)
    spy_df   = fetch_spy(spy_cache)

    print(f"  Universe: {all_data['ticker'].nunique()} tickers, "
          f"{all_data['date'].min().date()} to {all_data['date'].max().date()}")

    # ── Fetch ACTUAL earnings dates ──
    print("\n[2/7] Fetching ACTUAL earnings dates (this fixes PEAD v1 bug)...")
    earnings_cache = OUTPUT / "earnings_dates_cache.parquet"
    earnings_dates = fetch_earnings_dates(TICKERS, earnings_cache)

    total_known = sum(len(v) for v in earnings_dates.values())
    tickers_with_dates = len([t for t in TICKERS if t in earnings_dates and len(earnings_dates[t]) > 0])
    print(f"  Known earnings dates: {total_known} across {tickers_with_dates} tickers")
    if tickers_with_dates < 15:
        print("  WARNING: Less than half of tickers have earnings dates. "
              "Fallback detection will be used for missing tickers.")

    # ── Detect earnings gaps ──
    print("\n[3/7] Detecting earnings gaps with cross-validation...")
    events = detect_earnings_gaps(all_data, earnings_dates)

    if not events:
        print("\n  FATAL: No earnings gap events detected. Cannot proceed.")
        return

    # Validate detection quality
    validate_earnings_detection(events, earnings_dates)

    # Show gap distribution
    gaps = [e['abs_gap_pct'] for e in events]
    print(f"  Gap distribution: min={min(gaps)*100:.1f}%, median={np.median(gaps)*100:.1f}%, "
          f"max={max(gaps)*100:.1f}%")
    for thresh in [0.03, 0.05, 0.07, 0.10, 0.15]:
        n = sum(1 for g in gaps if g >= thresh)
        print(f"    Gap >= {thresh*100:.0f}%: {n} events")

    directions = [e['direction'] for e in events]
    n_long = directions.count('long')
    n_short = directions.count('short')
    print(f"  Direction split: {n_long} gap-up (long), {n_short} gap-down (short)")

    # ── Classify SPY regime ──
    print("\n[4/7] Classifying SPY regime...")
    spy_regime = classify_spy_regime(spy_df)
    regime_counts = defaultdict(int)
    for v in spy_regime.values():
        regime_counts[v] += 1
    print(f"  Regime distribution: {dict(regime_counts)}")

    # ── Run equity backtests ──
    print("\n[5/7] Running equity backtests...")
    all_results = {}
    best_combo = None
    best_sharpe = -999

    for gap_name, gap_thresh in GAP_THRESHOLDS.items():
        for hold in HOLD_PERIODS:
            label = f"{gap_name}_hold{hold}"
            print(f"  Testing {label}...", end='')

            trades = run_equity_backtest(all_data, events, gap_thresh, hold, spy_regime)
            metrics = compute_metrics(trades, label)
            n = metrics['n_trades']
            print(f"  n={n}", end='')
            if n > 0:
                print(f", WR={metrics.get('win_rate',0):.1%}, "
                      f"Sharpe={metrics.get('sharpe',0):.2f}, "
                      f"PF={metrics.get('profit_factor',0):.2f}, "
                      f"AvgRet={metrics.get('avg_return_pct',0):.2f}%")
            else:
                print(" (no trades)")

            all_results[label] = {
                'metrics': metrics,
                'trades': trades,
            }

            if metrics.get('sharpe', 0) > best_sharpe and n >= 10:
                best_sharpe = metrics['sharpe']
                best_combo = label

    # ── Quality gates on all combos ──
    print("\n[6/7] Quality gates (HC #705: all adversarial checks inline)...")
    quality_results = {}
    for label, res in all_results.items():
        trades = res['trades']
        if len(trades) < 8:
            quality_results[label] = {'skip': True, 'reason': f'only {len(trades)} trades'}
            print(f"  {label}: SKIP ({len(trades)} trades)")
            continue

        r1 = regime_agnostic_test(trades)
        perm = permutation_test_date_shuffle(trades, all_data)
        adv = adversarial_tests(trades) if len(trades) >= 15 else {'overall_pass': False, 'reason': 'too few'}

        all_pass = r1['pass'] and perm['pass'] and adv.get('overall_pass', False)
        quality_results[label] = {
            'regime_test': r1,
            'permutation_test': perm,
            'adversarial': adv,
            'ALL_PASS': all_pass,
        }

        # Print warning banners for failures
        status = "PASS" if all_pass else "FAIL"
        print(f"  {label}: {status}")
        print(f"    R1 regime: {'PASS' if r1['pass'] else 'FAIL'} (gap={r1.get('gap',0):.3f})")
        print(f"    Permutation: {'PASS' if perm['pass'] else 'FAIL'} (p={perm['p_value']:.4f}, method={perm.get('method','')})")
        if isinstance(adv, dict) and 'overall_pass' in adv:
            print(f"    Adversarial: {'PASS' if adv['overall_pass'] else 'FAIL'}")
            if adv.get('warnings'):
                for w in adv['warnings']:
                    print(f"      *** WARNING: {w} ***")

    # ── Options P&L estimation ──
    print("\n[7/7] Options P&L estimation (BS debit spreads)...")
    options_results = {}

    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]['metrics'].get('sharpe', -999), reverse=True)

    for label, res in ranked[:6]:
        trades = res['trades']
        if len(trades) < 5:
            continue

        opt_trades = estimate_options_pnl(trades, all_data)
        if not opt_trades:
            continue

        opt_df = pd.DataFrame(opt_trades)
        total_pnl = opt_df['total_pnl'].sum()
        avg_pnl = opt_df['total_pnl'].mean()
        wr = (opt_df['total_pnl'] > 0).mean()
        n = len(opt_df)

        options_results[label] = {
            'n_trades': n,
            'total_pnl': round(total_pnl, 2),
            'avg_pnl_per_trade': round(avg_pnl, 2),
            'win_rate': round(wr, 4),
            'avg_debit': round(opt_df['debit_paid'].mean(), 2),
            'avg_contracts': round(opt_df['n_contracts'].mean(), 1),
            'total_commission': round(opt_df['commission'].sum(), 2),
        }

        # Portfolio sim with $440 starting capital
        eq_curve_small, exec_small = simulate_portfolio(opt_trades, 440)
        eq_curve_10k, exec_10k = simulate_portfolio(opt_trades, STARTING_CAPITAL)

        if exec_10k:
            final_10k = STARTING_CAPITAL + sum(t['total_pnl'] for t in exec_10k)
            options_results[label]['portfolio_10k_final'] = round(final_10k, 2)
            options_results[label]['portfolio_10k_return_pct'] = round((final_10k / STARTING_CAPITAL - 1) * 100, 2)
            options_results[label]['portfolio_10k_n_executed'] = len(exec_10k)

        if exec_small:
            final_440 = 440 + sum(t['total_pnl'] for t in exec_small)
            options_results[label]['portfolio_440_final'] = round(final_440, 2)
            options_results[label]['portfolio_440_return_pct'] = round((final_440 / 440 - 1) * 100, 2)
            options_results[label]['portfolio_440_n_executed'] = len(exec_small)

        print(f"  {label}: {n} trades, P&L=${total_pnl:+,.0f}, WR={wr:.1%}, avg=${avg_pnl:+,.0f}/trade")

    # ═════════════════════════════════════════════════════════════════════════
    # COMPILE REPORT
    # ═════════════════════════════════════════════════════════════════════════

    ticker_breakdown = {}
    if best_combo and all_results[best_combo]['trades']:
        ticker_breakdown = per_ticker_breakdown(all_results[best_combo]['trades'])

    report = {
        'backtest': 'Earnings Gap Buyer v1 — PEAD with ACTUAL Earnings Dates',
        'date_run': str(datetime.now()),
        'universe': f'{len(TICKERS)} large-cap stocks',
        'period': f"{all_data['date'].min().date()} to {all_data['date'].max().date()}",
        'starting_capital': STARTING_CAPITAL,
        'risk_per_trade': RISK_PER_TRADE,
        'max_concurrent': MAX_CONCURRENT,
        'best_combo': best_combo,
        'best_sharpe': round(best_sharpe, 3) if best_sharpe > -999 else None,
        'n_earnings_events': len(events),
        'tickers_with_known_earnings': tickers_with_dates,

        'equity_sim_results': {
            label: res['metrics']
            for label, res in sorted(all_results.items(),
                                     key=lambda x: x[1]['metrics'].get('sharpe', -999),
                                     reverse=True)
        },

        'quality_gates': quality_results,
        'options_pnl_estimation': options_results,
        'ticker_breakdown_best': ticker_breakdown,
    }

    report_path = OUTPUT / "backtest_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")

    # Save trades for the best combo
    if best_combo and all_results[best_combo]['trades']:
        trades_df = pd.DataFrame(all_results[best_combo]['trades'])
        trades_path = OUTPUT / f"trades_{best_combo}.csv"
        trades_df.to_csv(trades_path, index=False)

    # ═════════════════════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("SUMMARY — EQUITY SIMULATION (Signal Validation)")
    print("=" * 90)

    print(f"\n{'Config':<25} {'N':>5} {'WR':>7} {'AvgRet':>8} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'MaxDD':>7}")
    print("-" * 85)
    for label, res in sorted(all_results.items(),
                             key=lambda x: x[1]['metrics'].get('sharpe', -999), reverse=True):
        m = res['metrics']
        if m['n_trades'] < 3:
            continue
        print(f"{label:<25} {m['n_trades']:>5} {m.get('win_rate',0):>6.1%} "
              f"{m.get('avg_return_pct',0):>7.2f}% {m.get('sharpe',0):>7.2f} "
              f"{m.get('sortino',0):>7.2f} {m.get('profit_factor',0):>6.2f} "
              f"{m.get('max_drawdown_pct',0):>6.1f}%")

    # Direction breakdown for best combo
    if best_combo:
        m = all_results[best_combo]['metrics']
        db = m.get('direction_breakdown', {})
        if db:
            print(f"\n  Direction breakdown ({best_combo}):")
            for d, info in db.items():
                print(f"    {d.upper()}: n={info['n']}, WR={info['wr']:.1%}, avg_ret={info['avg_ret']:.2f}%")

    print("\n" + "=" * 90)
    print("QUALITY GATES (HC #705)")
    print("=" * 90)

    any_pass = False
    for label, qr in quality_results.items():
        if 'skip' in qr:
            print(f"  {label:<25} SKIP ({qr['reason']})")
            continue
        status = "PASS" if qr['ALL_PASS'] else "FAIL"
        if qr['ALL_PASS']:
            any_pass = True
        r1_note = qr['regime_test'].get('note', '')
        perm_p = qr['permutation_test'].get('p_value', 1)
        perm_method = qr['permutation_test'].get('method', '')
        adv_pass = qr['adversarial'].get('overall_pass', False)
        print(f"  {label:<25} {status}  R1: {r1_note}")
        print(f"  {'':25}       perm_p={perm_p:.4f} ({perm_method}), adv={adv_pass}")

        # Print warning banners
        if not qr['ALL_PASS']:
            failures = []
            if not qr['regime_test']['pass']:
                failures.append('REGIME')
            if not qr['permutation_test']['pass']:
                failures.append('PERMUTATION')
            if not qr['adversarial'].get('overall_pass', False):
                adv = qr['adversarial']
                if isinstance(adv, dict) and adv.get('warnings'):
                    for w in adv['warnings']:
                        failures.append(w)
            if failures:
                print(f"  {'':25}       *** FAILED: {', '.join(failures)} ***")

    if options_results:
        print("\n" + "=" * 90)
        print("OPTIONS P&L ESTIMATION (BS Debit Spreads — Calls for gap-up, Puts for gap-down)")
        print("=" * 90)
        for label, opr in options_results.items():
            print(f"  {label}:")
            print(f"    Trades: {opr['n_trades']}, WR: {opr['win_rate']:.1%}, "
                  f"Total P&L: ${opr['total_pnl']:+,.0f}, "
                  f"Avg/trade: ${opr['avg_pnl_per_trade']:+,.0f}")
            print(f"    Avg debit: ${opr['avg_debit']:.2f}, "
                  f"Avg contracts: {opr['avg_contracts']:.1f}")
            if 'portfolio_10k_final' in opr:
                print(f"    $10K portfolio: ${opr['portfolio_10k_final']:,.0f} "
                      f"({opr['portfolio_10k_return_pct']:+.1f}%), "
                      f"{opr['portfolio_10k_n_executed']} executed")
            if 'portfolio_440_final' in opr:
                print(f"    $440 portfolio: ${opr['portfolio_440_final']:,.0f} "
                      f"({opr['portfolio_440_return_pct']:+.1f}%), "
                      f"{opr['portfolio_440_n_executed']} executed")

    if ticker_breakdown:
        print(f"\n{'─'*80}")
        print(f"TOP/BOTTOM TICKERS (best combo: {best_combo})")
        print(f"{'─'*80}")
        items = list(ticker_breakdown.items())
        for ticker, tb in items[:8]:
            print(f"  {ticker:<6} n={tb['n']:>3}  WR={tb['wr']:.1%}  "
                  f"avg={tb['avg_ret']:+.2f}%  total={tb['total_ret']:+.1f}%  "
                  f"avg_gap={tb['avg_gap']:+.1f}%")
        if len(items) > 12:
            print("  ...")
            for ticker, tb in items[-4:]:
                print(f"  {ticker:<6} n={tb['n']:>3}  WR={tb['wr']:.1%}  "
                      f"avg={tb['avg_ret']:+.2f}%  total={tb['total_ret']:+.1f}%  "
                      f"avg_gap={tb['avg_gap']:+.1f}%")

    # ── Final verdict ──
    print("\n" + "=" * 90)
    if any_pass:
        passing = [l for l, qr in quality_results.items() if qr.get('ALL_PASS', False)]
        print(f"VERDICT: {len(passing)} config(s) PASSED all quality gates:")
        for p in passing:
            m = all_results[p]['metrics']
            print(f"  >> {p}: Sharpe={m['sharpe']:.2f}, WR={m['win_rate']:.1%}, "
                  f"PF={m['profit_factor']:.2f}, AvgRet={m['avg_return_pct']:.2f}%")
        print("\nPost-Earnings Announcement Drift shows a statistically robust edge.")
        print("Recommended: start paper-trading the best config with $250/trade on Robinhood.")
    else:
        print("VERDICT: NO configs passed all quality gates.")
        print("Earnings Gap Buyer does NOT show a robust, regime-agnostic edge")
        print("with this universe and parameter set.")

        # Identify closest to passing
        closest = None
        closest_fails = 999
        for label, qr in quality_results.items():
            if 'skip' in qr:
                continue
            n_fails = sum(1 for k in ['regime_test', 'permutation_test']
                         if not qr.get(k, {}).get('pass', False))
            if not qr.get('adversarial', {}).get('overall_pass', False):
                n_fails += 1
            if n_fails < closest_fails:
                closest_fails = n_fails
                closest = label
        if closest:
            print(f"\nClosest to passing: {closest} ({closest_fails} gate(s) failed)")

    print("=" * 90)


if __name__ == '__main__':
    main()
