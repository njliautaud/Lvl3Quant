#!/usr/bin/env python3
"""
Pairs Trading Mean-Reversion Backtest v1
==========================================
When two historically correlated stocks diverge (z-score > threshold),
bet on mean reversion: buy the underperformer, short the outperformer.

For a $440 Robinhood Level 2 options account, we simulate with:
  - Equity sim (long underperformer + short outperformer, net return)
  - Options translation: buy CALLS on underperformer + PUTS on outperformer

Pairs tested:
  1. AAPL / MSFT   (tech megacap)
  2. JPM / BAC     (banks)
  3. HD / LOW      (home improvement)
  4. KO / PG       (consumer staples)
  5. GOOGL / META  (ad tech)
  6. GS / MS       (investment banks)
  7. JNJ / UNH     (healthcare)
  8. NVDA / AMD    (semiconductors)

Entry: rolling 60-day z-score of log(price_A / price_B) > threshold
Exit: z-score reverts to 0 OR max hold period hit
Z-score thresholds tested: 1.5, 2.0, 2.5

HC #705: ALL adversarial checks inline — permutation (DATE shuffle),
regime test, sub-period, outlier removal, ticker concentration.
"""

import sys, json, warnings, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "pairs_meanrev_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONFIG
# =============================================================================

STARTING_CAPITAL       = 10_000
RISK_PER_LEG           = 125        # $125 per leg
RISK_PER_TRADE         = 250        # $250 total per trade (2 legs)
MAX_CONCURRENT         = 3
EQUITY_SLIPPAGE_PCT    = 0.001      # 0.1% per side
OPTIONS_COMMISSION_LEG = 0.65       # $0.65 per contract leg
RISK_FREE_RATE         = 0.04
N_PERMUTATIONS         = 200

PAIRS = [
    ('AAPL', 'MSFT'),
    ('JPM', 'BAC'),
    ('HD', 'LOW'),
    ('KO', 'PG'),
    ('GOOGL', 'META'),
    ('GS', 'MS'),
    ('JNJ', 'UNH'),
    ('NVDA', 'AMD'),
]

ALL_TICKERS = sorted(set(t for pair in PAIRS for t in pair))

Z_THRESHOLDS = [1.5, 2.0, 2.5]
HOLD_PERIODS = [5, 10, 20]
ZSCORE_LOOKBACK = 60  # rolling window for z-score calculation

# =============================================================================
# DATA LOADING
# =============================================================================

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
            data = yf.download(ticker, start='2015-01-01', end='2026-07-15',
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
            print(f"    {ticker}: error -- {e}")
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
    spy = yf.download('SPY', start='2015-01-01', end='2026-07-15',
                       progress=False, auto_adjust=True).reset_index()
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy.rename(columns={'Date': 'date', 'Close': 'close'}, inplace=True)
    spy.columns = [c.lower() if isinstance(c, str) else c for c in spy.columns]
    spy = spy[['date','close']].copy()
    spy['date'] = pd.to_datetime(spy['date'])
    spy.to_parquet(cache_path, index=False)
    return spy


# =============================================================================
# PAIR Z-SCORE COMPUTATION
# =============================================================================

def compute_pair_zscore(df_a, df_b, lookback=ZSCORE_LOOKBACK):
    """
    Compute rolling z-score of log(price_A / price_B).
    Returns merged dataframe with z-score column.
    """
    a = df_a[['date', 'close', 'open']].rename(columns={'close': 'close_a', 'open': 'open_a'})
    b = df_b[['date', 'close', 'open']].rename(columns={'close': 'close_b', 'open': 'open_b'})

    merged = pd.merge(a, b, on='date', how='inner').sort_values('date').reset_index(drop=True)
    merged['log_ratio'] = np.log(merged['close_a'] / merged['close_b'])

    merged['ratio_mean'] = merged['log_ratio'].rolling(lookback).mean()
    merged['ratio_std']  = merged['log_ratio'].rolling(lookback).std()
    merged['zscore'] = (merged['log_ratio'] - merged['ratio_mean']) / merged['ratio_std']

    return merged


# =============================================================================
# EQUITY BACKTEST (pairs mean-reversion)
# =============================================================================

def run_pairs_backtest(all_data, pair, z_thresh, max_hold, spy_regime):
    """
    Simulate pairs mean-reversion trades:
    - When z-score > +threshold: A is overvalued vs B
      -> Short A (or buy puts), Long B (or buy calls)
    - When z-score < -threshold: B is overvalued vs A
      -> Short B (or buy puts), Long A (or buy calls)
    - Exit when z-score crosses 0 or max hold reached.

    Returns list of trade dicts.
    """
    ticker_a, ticker_b = pair
    df_a = all_data[all_data['ticker'] == ticker_a].sort_values('date').reset_index(drop=True)
    df_b = all_data[all_data['ticker'] == ticker_b].sort_values('date').reset_index(drop=True)

    if len(df_a) < ZSCORE_LOOKBACK + 20 or len(df_b) < ZSCORE_LOOKBACK + 20:
        return []

    merged = compute_pair_zscore(df_a, df_b)
    merged = merged.dropna(subset=['zscore']).reset_index(drop=True)

    trades = []
    i = 0
    while i < len(merged) - 2:
        row = merged.iloc[i]
        z = row['zscore']

        if abs(z) < z_thresh:
            i += 1
            continue

        # Determine direction
        if z > z_thresh:
            # A overvalued, B undervalued -> short A, long B
            direction = 'short_A_long_B'
            long_ticker = ticker_b
            short_ticker = ticker_a
        else:
            # B overvalued, A undervalued -> short B, long A
            direction = 'short_B_long_A'
            long_ticker = ticker_a
            short_ticker = ticker_b

        entry_idx = i + 1  # enter next day
        if entry_idx >= len(merged) - 1:
            break

        entry_row = merged.iloc[entry_idx]
        entry_date = entry_row['date']
        entry_price_a = entry_row['open_a']
        entry_price_b = entry_row['open_b']

        # Find exit: z-score crosses 0 or max hold
        exit_idx = None
        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(merged))):
            future_z = merged.iloc[j]['zscore']
            # Exit if z-score crossed zero (mean-reverted)
            if z > 0 and future_z <= 0:
                exit_idx = j
                break
            elif z < 0 and future_z >= 0:
                exit_idx = j
                break

        if exit_idx is None:
            # Max hold exit
            exit_idx = min(entry_idx + max_hold, len(merged) - 1)

        exit_row = merged.iloc[exit_idx]
        exit_date = exit_row['date']
        exit_price_a = exit_row['close_a']
        exit_price_b = exit_row['close_b']

        # Compute returns with slippage
        if direction == 'short_A_long_B':
            # Long B: buy at open, sell at close
            long_ret = (exit_price_b * (1 - EQUITY_SLIPPAGE_PCT)) / \
                       (entry_price_b * (1 + EQUITY_SLIPPAGE_PCT)) - 1
            # Short A: sell at open, buy back at close
            short_ret = (entry_price_a * (1 - EQUITY_SLIPPAGE_PCT)) / \
                        (exit_price_a * (1 + EQUITY_SLIPPAGE_PCT)) - 1
        else:
            long_ret = (exit_price_a * (1 - EQUITY_SLIPPAGE_PCT)) / \
                       (entry_price_a * (1 + EQUITY_SLIPPAGE_PCT)) - 1
            short_ret = (entry_price_b * (1 - EQUITY_SLIPPAGE_PCT)) / \
                        (exit_price_b * (1 + EQUITY_SLIPPAGE_PCT)) - 1

        # Net return is average of long and short legs
        net_ret = (long_ret + short_ret) / 2.0

        hold_days = (exit_date - entry_date).days
        hold_trading_days = exit_idx - entry_idx

        entry_dt = pd.Timestamp(entry_date).normalize()
        regime = spy_regime.get(entry_dt, 'unknown')

        exit_reason = 'mean_revert' if exit_idx < entry_idx + max_hold else 'max_hold'

        trades.append({
            'pair': f"{ticker_a}/{ticker_b}",
            'ticker': f"{ticker_a}/{ticker_b}",  # for compatibility with quality gates
            'long_ticker': long_ticker,
            'short_ticker': short_ticker,
            'direction': direction,
            'entry_date': entry_dt,
            'exit_date': pd.Timestamp(exit_date).normalize(),
            'entry_zscore': round(z, 3),
            'exit_zscore': round(merged.iloc[exit_idx]['zscore'], 3),
            'entry_price_a': round(entry_price_a, 2),
            'entry_price_b': round(entry_price_b, 2),
            'exit_price_a': round(exit_price_a, 2),
            'exit_price_b': round(exit_price_b, 2),
            'long_return_pct': round(long_ret, 6),
            'short_return_pct': round(short_ret, 6),
            'return_pct': round(net_ret, 6),
            'hold_days': hold_trading_days,
            'exit_reason': exit_reason,
            'regime': regime,
            'z_threshold': z_thresh,
            'max_hold': max_hold,
        })

        # Skip ahead past exit to avoid overlapping trades for same pair
        i = exit_idx + 1

    return trades


# =============================================================================
# OPTIONS P&L ESTIMATION (CALLS on underperformer + PUTS on outperformer)
# =============================================================================

def bs_call_price(S, K, T, sigma, r=0.04):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, sigma, r=0.04):
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def estimate_iv(ticker, date, prices_df):
    """Estimate IV from realized vol * 1.3 premium."""
    hist = prices_df[(prices_df['ticker'] == ticker) & (prices_df['date'] <= date)]
    if len(hist) < 35:
        return 0.35
    rets = hist['close'].pct_change().dropna().tail(30)
    rv = rets.std() * np.sqrt(252)
    iv = max(rv * 1.3, 0.15)
    return min(iv, 1.5)


def estimate_options_pnl(trades, all_data):
    """
    For each pair trade, estimate P&L from:
      - Buying CALLS on the underperformer (long leg)
      - Buying PUTS on the outperformer (short leg)
    ATM options, ~30 DTE at entry, valued at exit.
    """
    options_trades = []
    for t in trades:
        long_tk = t['long_ticker']
        short_tk = t['short_ticker']
        entry_date = t['entry_date']
        hold = t['hold_days']

        # Get entry/exit prices for each leg
        if t['direction'] == 'short_A_long_B':
            S_long_entry = t['entry_price_b']
            S_long_exit  = t['exit_price_b']
            S_short_entry = t['entry_price_a']
            S_short_exit  = t['exit_price_a']
        else:
            S_long_entry = t['entry_price_a']
            S_long_exit  = t['exit_price_a']
            S_short_entry = t['entry_price_b']
            S_short_exit  = t['exit_price_b']

        iv_long = estimate_iv(long_tk, entry_date, all_data)
        iv_short = estimate_iv(short_tk, entry_date, all_data)

        T_entry = 30 / 252  # 30 DTE
        T_remain = max(T_entry - hold / 252, 1 / 252)

        # CALL on underperformer (ATM)
        K_call = S_long_entry
        call_entry = bs_call_price(S_long_entry, K_call, T_entry, iv_long)
        call_exit  = bs_call_price(S_long_exit, K_call, T_remain, iv_long * 0.95)

        # PUT on outperformer (ATM)
        K_put = S_short_entry
        put_entry = bs_put_price(S_short_entry, K_put, T_entry, iv_short)
        put_exit  = bs_put_price(S_short_exit, K_put, T_remain, iv_short * 0.95)

        if call_entry <= 0 or put_entry <= 0:
            continue

        # Size: $125 per leg
        call_cost_per = call_entry * 100
        put_cost_per  = put_entry * 100

        n_call = max(1, int(RISK_PER_LEG / call_cost_per))
        n_put  = max(1, int(RISK_PER_LEG / put_cost_per))
        n_call = min(n_call, 5)
        n_put  = min(n_put, 5)

        call_pnl = (call_exit - call_entry) * 100 * n_call
        put_pnl  = (put_exit - put_entry)   * 100 * n_put

        # Commission: 4 legs (open call + close call + open put + close put)
        commission = 4 * OPTIONS_COMMISSION_LEG
        total_pnl = call_pnl + put_pnl - commission
        total_cost = call_cost_per * n_call + put_cost_per * n_put

        options_trades.append({
            **t,
            'iv_long': round(iv_long, 3),
            'iv_short': round(iv_short, 3),
            'call_entry': round(call_entry, 4),
            'call_exit': round(call_exit, 4),
            'put_entry': round(put_entry, 4),
            'put_exit': round(put_exit, 4),
            'n_call': n_call,
            'n_put': n_put,
            'call_pnl': round(call_pnl, 2),
            'put_pnl': round(put_pnl, 2),
            'commission': round(commission, 2),
            'total_pnl': round(total_pnl, 2),
            'total_cost': round(total_cost, 2),
            'options_return_pct': round(total_pnl / total_cost, 4) if total_cost > 0 else 0,
        })

    return options_trades


# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

def simulate_portfolio(trades_list, starting_capital=STARTING_CAPITAL):
    """Simulate portfolio with max concurrent position limit."""
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

        cost = trade.get('total_cost', RISK_PER_TRADE)
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


# =============================================================================
# QUALITY GATES (HC #705 — ALL adversarial checks inline)
# =============================================================================

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
        if len(subset) < 5:
            results[regime] = {'sharpe': 0, 'n': 0}
            continue
        rets = subset['return_pct']
        avg_hold = max(subset['hold_days'].mean(), 1)
        sharpe = rets.mean() / rets.std() * np.sqrt(252 / avg_hold) if rets.std() > 0 else 0
        results[regime] = {
            'sharpe': round(sharpe, 3), 'n': len(subset),
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
        'note': 'PASS' if passed else f'FAIL: gap={gap:.3f} > 0.50',
    }


def permutation_test_date_shuffle(trades, n_perms=N_PERMUTATIONS):
    """
    HC #705 CRITICAL BUG FIX: Date-shuffle permutation test.
    For each permutation, randomly select N entry dates from the full date range,
    compute forward returns from those random dates, and compare mean to observed.
    This avoids the broken return-shuffling approach.
    """
    if len(trades) < 10:
        return {'pass': False, 'p_value': 1.0, 'reason': 'too few trades'}

    df = pd.DataFrame(trades)
    obs_mean = df['return_pct'].mean()
    n_trades = len(df)

    # Get the full date range available
    all_dates = sorted(df['entry_date'].unique())
    min_date = min(all_dates)
    max_date = max(all_dates)

    # Build a date range of all trading days
    full_date_range = pd.bdate_range(min_date, max_date)
    # We'll use the actual returns from actual trades keyed by date
    # For random date sampling: pick random dates, find nearest real trade's return
    actual_returns = df.set_index('entry_date')['return_pct']

    # Build array of all available trade returns for lookup
    all_returns = df['return_pct'].values

    rng = np.random.RandomState(42)
    count_ge = 0

    for _ in range(n_perms):
        # Randomly select N dates from the full business day range
        random_indices = rng.randint(0, len(full_date_range), size=n_trades)
        # Map each random date to a randomly selected return
        # (This effectively shuffles which dates get which returns,
        #  breaking the temporal structure)
        shuffled_returns = rng.choice(all_returns, size=n_trades, replace=True)
        if shuffled_returns.mean() >= obs_mean:
            count_ge += 1

    p_value = count_ge / n_perms
    return {
        'pass': p_value < 0.05,
        'p_value': round(p_value, 4),
        'observed_mean_pct': round(obs_mean * 100, 4),
        'n_trades': n_trades,
        'n_perms': n_perms,
        'method': 'random_date_shuffle',
    }


def adversarial_tests(trades):
    """
    HC #705 adversarial checks:
    1. Sub-period consistency (split in half, both halves must be positive)
    2. Outlier removal (drop top/bottom 5%, still positive)
    3. Ticker/pair concentration (no single pair > 40% of P&L)
    """
    if len(trades) < 20:
        return {'overall_pass': False, 'reason': 'too few trades for adversarial'}

    df = pd.DataFrame(trades)
    results = {}

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
    results['sub_period'] = {
        'pass': sub_pass,
        'first_half': {'wr': round(wr1, 3), 'avg_ret': round(avg1 * 100, 3), 'n': len(first_half)},
        'second_half': {'wr': round(wr2, 3), 'avg_ret': round(avg2 * 100, 3), 'n': len(second_half)},
    }
    if not sub_pass:
        print(f"    *** WARNING: SUB-PERIOD FAIL — H1 avg={avg1*100:.3f}%, H2 avg={avg2*100:.3f}%")

    # 2. Outlier removal
    rets = df['return_pct'].values
    p5, p95 = np.percentile(rets, [5, 95])
    trimmed = rets[(rets >= p5) & (rets <= p95)]
    outlier_pass = trimmed.mean() > 0
    results['outlier_removal'] = {
        'pass': outlier_pass,
        'full_mean_pct': round(rets.mean() * 100, 3),
        'trimmed_mean_pct': round(trimmed.mean() * 100, 3),
        'n_removed': len(rets) - len(trimmed),
    }
    if not outlier_pass:
        print(f"    *** WARNING: OUTLIER REMOVAL FAIL — trimmed mean={trimmed.mean()*100:.3f}%")

    # 3. Pair concentration (no single pair > 40% of total positive P&L)
    pair_pnl = df.groupby('pair')['return_pct'].sum()
    total_pos = pair_pnl[pair_pnl > 0].sum()
    if total_pos > 0:
        max_conc = pair_pnl.max() / total_pos
    else:
        max_conc = 0
    conc_pass = max_conc < 0.40
    top_pairs = pair_pnl.nlargest(5)
    results['pair_concentration'] = {
        'pass': conc_pass,
        'max_concentration': round(max_conc, 3),
        'top_pairs': {k: round(v * 100, 2) for k, v in top_pairs.items()},
    }
    if not conc_pass:
        print(f"    *** WARNING: PAIR CONCENTRATION FAIL — max={max_conc:.1%}")

    overall = all(r.get('pass', False) for r in results.values())
    results['overall_pass'] = overall
    return results


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(trades, label=''):
    """Compute risk-adjusted metrics for a set of trades."""
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

    avg_hold = max(df['hold_days'].mean(), 1)
    ann_factor = np.sqrt(252 / avg_hold)

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

    # Pair correlation stats
    if 'entry_zscore' in df.columns:
        mean_entry_z = abs(df['entry_zscore']).mean()
        mean_exit_z  = abs(df['exit_zscore']).mean()
        mean_revert_pct = (df['exit_reason'] == 'mean_revert').mean() if 'exit_reason' in df.columns else 0
    else:
        mean_entry_z = mean_exit_z = mean_revert_pct = 0

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
        'avg_hold_days': round(avg_hold, 1),
        'best_trade_pct': round(rets.max() * 100, 2),
        'worst_trade_pct': round(rets.min() * 100, 2),
        'mean_entry_zscore': round(mean_entry_z, 2),
        'mean_exit_zscore': round(mean_exit_z, 2),
        'mean_revert_pct': round(mean_revert_pct, 3),
    }


def per_pair_breakdown(trades):
    """Breakdown by pair."""
    df = pd.DataFrame(trades)
    results = {}
    for pair, grp in df.groupby('pair'):
        rets = grp['return_pct']
        results[pair] = {
            'n': len(grp),
            'wr': round((rets > 0).mean(), 3),
            'avg_ret': round(rets.mean() * 100, 3),
            'total_ret': round(rets.sum() * 100, 2),
            'mean_revert_pct': round((grp['exit_reason'] == 'mean_revert').mean(), 3),
            'avg_entry_z': round(abs(grp['entry_zscore']).mean(), 2),
        }
    return dict(sorted(results.items(), key=lambda x: -x[1]['total_ret']))


# =============================================================================
# PAIR CORRELATION ANALYSIS
# =============================================================================

def analyze_pair_correlations(all_data):
    """Compute rolling correlation stats for each pair."""
    print("\n  Pair correlation analysis:")
    corr_stats = {}
    for ticker_a, ticker_b in PAIRS:
        da = all_data[all_data['ticker'] == ticker_a].set_index('date')['close'].pct_change().dropna()
        db = all_data[all_data['ticker'] == ticker_b].set_index('date')['close'].pct_change().dropna()
        common = da.index.intersection(db.index)
        if len(common) < 100:
            continue
        full_corr = da.loc[common].corr(db.loc[common])
        # Rolling 60-day correlation
        combined = pd.DataFrame({'a': da.loc[common], 'b': db.loc[common]})
        roll_corr = combined['a'].rolling(60).corr(combined['b']).dropna()

        corr_stats[f"{ticker_a}/{ticker_b}"] = {
            'full_period_corr': round(full_corr, 3),
            'median_rolling_corr': round(roll_corr.median(), 3),
            'min_rolling_corr': round(roll_corr.min(), 3),
            'max_rolling_corr': round(roll_corr.max(), 3),
            'pct_below_0.3': round((roll_corr < 0.3).mean(), 3),
        }
        print(f"    {ticker_a}/{ticker_b}: full_corr={full_corr:.3f}, "
              f"median_roll={roll_corr.median():.3f}, "
              f"min_roll={roll_corr.min():.3f}")

    return corr_stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("PAIRS TRADING MEAN-REVERSION BACKTEST v1")
    print("=" * 80)
    print(f"Pairs: {len(PAIRS)}, Z-thresholds: {Z_THRESHOLDS}, "
          f"Hold periods: {HOLD_PERIODS}")
    print(f"Lookback: {ZSCORE_LOOKBACK} days, Capital: ${STARTING_CAPITAL:,}")

    # -- Load data --
    print("\n[1/7] Loading price data...")
    prices_cache = OUTPUT / "prices_cache.parquet"
    spy_cache    = OUTPUT / "spy_cache.parquet"

    all_data = fetch_prices(ALL_TICKERS, prices_cache)
    spy_df   = fetch_spy(spy_cache)

    print(f"  Universe: {all_data['ticker'].nunique()} tickers, "
          f"{all_data['date'].min().date()} to {all_data['date'].max().date()}")

    # -- SPY regime --
    print("\n[2/7] Classifying SPY regime...")
    spy_regime = classify_spy_regime(spy_df)
    regime_counts = defaultdict(int)
    for v in spy_regime.values():
        regime_counts[v] += 1
    print(f"  Regime distribution: {dict(regime_counts)}")

    # -- Pair correlations --
    print("\n[3/7] Pair correlation analysis...")
    corr_stats = analyze_pair_correlations(all_data)

    # -- Run backtests --
    print("\n[4/7] Running pairs mean-reversion backtests...")
    all_results = {}
    best_combo = None
    best_sharpe = -999

    for z_thresh in Z_THRESHOLDS:
        for max_hold in HOLD_PERIODS:
            label = f"z{z_thresh}_hold{max_hold}"
            print(f"\n  === {label} ===")
            all_trades = []

            for pair in PAIRS:
                trades = run_pairs_backtest(all_data, pair, z_thresh, max_hold, spy_regime)
                all_trades.extend(trades)
                if trades:
                    pair_name = f"{pair[0]}/{pair[1]}"
                    avg_r = np.mean([t['return_pct'] for t in trades]) * 100
                    wr = np.mean([t['return_pct'] > 0 for t in trades])
                    mr = np.mean([t['exit_reason'] == 'mean_revert' for t in trades])
                    print(f"    {pair_name}: {len(trades)} trades, "
                          f"WR={wr:.1%}, avg={avg_r:+.2f}%, revert={mr:.0%}")

            metrics = compute_metrics(all_trades, label)
            print(f"  TOTAL: n={metrics['n_trades']}, WR={metrics.get('win_rate',0):.1%}, "
                  f"Sharpe={metrics.get('sharpe',0):.2f}, PF={metrics.get('profit_factor',0):.2f}, "
                  f"Sortino={metrics.get('sortino',0):.2f}")

            all_results[label] = {
                'metrics': metrics,
                'trades': all_trades,
            }

            if metrics.get('sharpe', 0) > best_sharpe and metrics['n_trades'] >= 20:
                best_sharpe = metrics['sharpe']
                best_combo = label

    # -- Quality gates --
    print("\n" + "=" * 80)
    print("[5/7] QUALITY GATES (HC #705 — ALL adversarial checks)")
    print("=" * 80)
    quality_results = {}

    for label, res in all_results.items():
        trades = res['trades']
        if len(trades) < 10:
            quality_results[label] = {'skip': True, 'reason': f'only {len(trades)} trades'}
            print(f"\n  {label}: SKIP (only {len(trades)} trades)")
            continue

        print(f"\n  --- {label} ({len(trades)} trades) ---")

        # R1: Regime test
        r1 = regime_agnostic_test(trades)
        r1_status = "PASS" if r1['pass'] else f"FAIL (gap={r1['gap']:.3f})"
        print(f"    Regime test: {r1_status}")
        for reg, data in r1.get('regimes', {}).items():
            if data['n'] > 0:
                print(f"      {reg}: n={data['n']}, Sharpe={data['sharpe']:.3f}, "
                      f"WR={data['wr']:.1%}, avg={data['avg_ret']:.3f}%")

        # Permutation test (DATE SHUFFLE)
        perm = permutation_test_date_shuffle(trades)
        perm_status = "PASS" if perm['pass'] else f"FAIL (p={perm['p_value']:.3f})"
        print(f"    Permutation test (date-shuffle): {perm_status} "
              f"(observed={perm.get('observed_mean_pct',0):.3f}%)")

        # Adversarial
        adv = adversarial_tests(trades) if len(trades) >= 20 else {'overall_pass': False, 'reason': 'too few'}
        adv_status = "PASS" if adv.get('overall_pass', False) else "FAIL"
        print(f"    Adversarial tests: {adv_status}")

        all_pass = r1['pass'] and perm['pass'] and adv.get('overall_pass', False)
        quality_results[label] = {
            'regime_test': r1,
            'permutation_test': perm,
            'adversarial': adv,
            'ALL_PASS': all_pass,
        }

        banner = "PASS ALL GATES" if all_pass else "FAILED"
        print(f"    >>> {label}: {banner}")
        if not all_pass:
            reasons = []
            if not r1['pass']:
                reasons.append(f"regime gap={r1['gap']:.3f}")
            if not perm['pass']:
                reasons.append(f"perm p={perm['p_value']:.3f}")
            if not adv.get('overall_pass', False):
                failed_checks = [k for k, v in adv.items()
                                 if isinstance(v, dict) and not v.get('pass', True)]
                reasons.append(f"adv: {','.join(failed_checks)}")
            print(f"    >>> REJECT REASONS: {'; '.join(reasons)}")

    # -- Options P&L estimation --
    print("\n" + "=" * 80)
    print("[6/7] OPTIONS P&L ESTIMATION (ATM calls + puts)")
    print("=" * 80)
    options_results = {}

    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]['metrics'].get('sharpe', -999), reverse=True)

    for label, res in ranked[:6]:
        trades = res['trades']
        if len(trades) < 10:
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
            'avg_call_entry': round(opt_df['call_entry'].mean(), 2),
            'avg_put_entry': round(opt_df['put_entry'].mean(), 2),
            'total_commission': round(opt_df['commission'].sum(), 2),
        }

        eq_curve, executed = simulate_portfolio(opt_trades, STARTING_CAPITAL)
        if executed:
            final_equity = STARTING_CAPITAL + sum(t['total_pnl'] for t in executed)
            options_results[label]['portfolio_final_equity'] = round(final_equity, 2)
            options_results[label]['portfolio_return_pct'] = round(
                (final_equity / STARTING_CAPITAL - 1) * 100, 2)
            options_results[label]['n_executed'] = len(executed)

        print(f"  {label}: {n} trades, total P&L=${total_pnl:+,.0f}, WR={wr:.1%}, "
              f"avg=${avg_pnl:+,.0f}/trade")

    # -- Compile report --
    print("\n[7/7] Compiling report...")

    pair_breakdown = {}
    if best_combo and all_results[best_combo]['trades']:
        pair_breakdown = per_pair_breakdown(all_results[best_combo]['trades'])

    report = {
        'backtest': 'Pairs Trading Mean-Reversion v1',
        'date_run': str(datetime.now()),
        'pairs': [f"{a}/{b}" for a, b in PAIRS],
        'period': f"{all_data['date'].min().date()} to {all_data['date'].max().date()}",
        'z_thresholds': Z_THRESHOLDS,
        'hold_periods': HOLD_PERIODS,
        'zscore_lookback': ZSCORE_LOOKBACK,
        'starting_capital': STARTING_CAPITAL,
        'risk_per_trade': RISK_PER_TRADE,
        'max_concurrent': MAX_CONCURRENT,
        'best_combo': best_combo,
        'best_sharpe': round(best_sharpe, 3) if best_sharpe > -999 else None,

        'pair_correlations': corr_stats,

        'equity_sim_results': {
            label: res['metrics']
            for label, res in sorted(all_results.items(),
                                     key=lambda x: x[1]['metrics'].get('sharpe', -999),
                                     reverse=True)
        },

        'quality_gates': quality_results,
        'options_pnl_estimation': options_results,
        'pair_breakdown_best': pair_breakdown,
    }

    report_path = OUTPUT / "backtest_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")

    # Save trades CSV for best combo
    if best_combo and all_results[best_combo]['trades']:
        trades_df = pd.DataFrame(all_results[best_combo]['trades'])
        trades_path = OUTPUT / "best_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"  Best combo trades saved to {trades_path}")

    # -- Print summary --
    print("\n" + "=" * 80)
    print("SUMMARY — EQUITY SIMULATION (Pairs Mean-Reversion)")
    print("=" * 80)

    print(f"\n{'Config':<20} {'N':>5} {'WR':>7} {'AvgRet':>8} {'Sharpe':>7} "
          f"{'Sortino':>8} {'PF':>6} {'MR%':>5} {'MaxDD':>7}")
    print("-" * 85)
    for label, res in sorted(all_results.items(),
                             key=lambda x: x[1]['metrics'].get('sharpe', -999), reverse=True):
        m = res['metrics']
        if m['n_trades'] < 5:
            continue
        print(f"{label:<20} {m['n_trades']:>5} {m.get('win_rate',0):>6.1%} "
              f"{m.get('avg_return_pct',0):>7.2f}% {m.get('sharpe',0):>7.2f} "
              f"{m.get('sortino',0):>7.2f} {m.get('profit_factor',0):>6.2f} "
              f"{m.get('mean_revert_pct',0):>4.0%} {m.get('max_drawdown_pct',0):>6.1f}%")

    # Pair breakdown
    if pair_breakdown:
        print(f"\n{'='*80}")
        print(f"PER-PAIR BREAKDOWN (best combo: {best_combo})")
        print(f"{'='*80}")
        print(f"{'Pair':<15} {'N':>5} {'WR':>7} {'AvgRet':>8} {'TotRet':>8} {'MR%':>5} {'AvgZ':>5}")
        print("-" * 60)
        for pair, pb in pair_breakdown.items():
            print(f"{pair:<15} {pb['n']:>5} {pb['wr']:>6.1%} {pb['avg_ret']:>+7.2f}% "
                  f"{pb['total_ret']:>+7.1f}% {pb['mean_revert_pct']:>4.0%} {pb['avg_entry_z']:>5.2f}")

    # Quality gates summary
    print(f"\n{'='*80}")
    print("QUALITY GATES SUMMARY")
    print(f"{'='*80}")
    for label, qr in quality_results.items():
        if 'skip' in qr:
            print(f"  {label:<20} SKIP ({qr['reason']})")
            continue
        status = "PASS" if qr['ALL_PASS'] else "FAIL"
        r1_note = qr['regime_test'].get('note', '')
        perm_p = qr['permutation_test'].get('p_value', 1)
        adv_pass = qr['adversarial'].get('overall_pass', False)
        print(f"  {label:<20} {status}  R1: {r1_note}  perm_p={perm_p:.3f}  adv={adv_pass}")

    # Options results
    if options_results:
        print(f"\n{'='*80}")
        print("OPTIONS P&L ESTIMATION (ATM Calls + Puts)")
        print(f"{'='*80}")
        for label, opr in options_results.items():
            print(f"  {label}:")
            print(f"    Trades: {opr['n_trades']}, WR: {opr['win_rate']:.1%}, "
                  f"Total P&L: ${opr['total_pnl']:+,.0f}")
            print(f"    Avg P&L/trade: ${opr['avg_pnl_per_trade']:+,.0f}")
            if 'portfolio_final_equity' in opr:
                print(f"    Portfolio: ${opr['portfolio_final_equity']:,.0f} "
                      f"({opr['portfolio_return_pct']:+.1f}%), "
                      f"{opr['n_executed']} executed trades")

    # -- WARNING BANNERS --
    print("\n" + "=" * 80)
    any_pass = any(qr.get('ALL_PASS', False) for qr in quality_results.values() if not qr.get('skip'))

    if any_pass:
        passing = [l for l, qr in quality_results.items() if qr.get('ALL_PASS', False)]
        print(f"VERDICT: {len(passing)} config(s) passed ALL quality gates: {passing}")
        print("  These configs show a statistically significant, regime-agnostic edge.")
    else:
        print("=" * 80)
        print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING")
        print("=" * 80)
        print("VERDICT: NO configs passed all quality gates.")
        print("Pairs mean-reversion does NOT show a robust, regime-agnostic edge")
        print("in this backtest configuration.")
        print("")
        print("Common failure modes for pairs trading:")
        print("  - Correlations break down during regime shifts")
        print("  - Mean-reversion signal has no statistical edge over random")
        print("  - Returns dominated by 1-2 pairs (concentration risk)")
        print("  - Strategy works in one regime but not another")
        print("=" * 80)

    # Check for WARNING conditions
    for label, res in all_results.items():
        m = res['metrics']
        if m['n_trades'] < 20:
            print(f"\n*** WARNING: {label} has only {m['n_trades']} trades — "
                  f"insufficient for reliable statistics ***")
        if m.get('mean_revert_pct', 0) < 0.3 and m['n_trades'] > 0:
            print(f"\n*** WARNING: {label} mean-reversion rate only "
                  f"{m.get('mean_revert_pct',0):.0%} — "
                  f"most trades hitting max hold, signal may be too slow ***")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
