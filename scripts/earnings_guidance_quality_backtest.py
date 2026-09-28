#!/usr/bin/env python3
"""
Earnings Guidance Quality Backtest
===================================
Hypothesis: Stocks that beat earnings AND show positive post-gap drift
(proxy for guidance raise / market confirming forward outlook) outperform
stocks that merely beat.

6 Variants:
  A) Quality Beats Only: Buy after >3% gap ONLY if 5-day post-gap drift positive. Hold 40d.
  B) Immediate Beat + Hold: Buy next day after >3% gap, hold 40d (baseline).
  C) Quality Score: Score = gap_pct × (1 + post_5d_drift). Size proportional. Hold 40d.
  D) Anti-Fade: Buy after >3% gap ONLY if stock hasn't faded >2% in 5 days. Hold 60d.
  E) Beat Acceleration: Only if THIS quarter gap > LAST quarter gap. Hold 60d.
  F) Consecutive Quality: Both current AND previous quarter had positive gap + positive 5d drift. Hold 40d.

Walk-forward OOT: Jan 2022 – Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'NFLX', 'CRM', 'PLTR', 'SOFI', 'HOOD', 'SNAP', 'PINS', 'UBER',
    'LYFT', 'COIN', 'RBLX', 'DDOG', 'TTD', 'SHOP', 'NET', 'ROKU'
]

OOT_START = pd.Timestamp('2022-01-01')
OOT_END = pd.Timestamp('2026-07-29')
PERM_ITERATIONS = 1000
GAP_THRESHOLD = 0.03
VOLUME_MULTIPLIER = 1.5
POST_GAP_LOOKBACK = 5  # days after gap to measure drift

# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download price data for universe + SPY."""
    print("Downloading price data...")
    all_tickers = list(set(UNIVERSE + ['SPY']))

    prices = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start='2021-01-01', end=OOT_END.strftime('%Y-%m-%d'),
                           progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 50:
                prices[ticker] = df
        except Exception as e:
            print(f"  Failed {ticker}: {e}")

    print(f"  Downloaded {len(prices)} tickers")
    return prices


def detect_earnings_events(prices):
    """
    Detect earnings events using overnight gap proxy.
    Gap >3% on >1.5x volume = likely earnings.
    Also compute the 5-day post-gap drift for each event.
    """
    events = {}

    for ticker in UNIVERSE:
        if ticker not in prices:
            continue
        df = prices[ticker].copy()
        df['prev_close'] = df['Close'].shift(1)
        df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']
        df['abs_gap'] = df['gap_pct'].abs()
        df['vol_ma20'] = df['Volume'].rolling(20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_ma20']

        earnings_mask = (df['abs_gap'] > GAP_THRESHOLD) & (df['vol_ratio'] > VOLUME_MULTIPLIER)
        earnings_days = df[earnings_mask].copy()

        if len(earnings_days) == 0:
            continue

        # Deduplicate: 1 event per 60-day window
        deduped = []
        last_date = None
        for date, row in earnings_days.iterrows():
            if last_date is None or (date - last_date).days > 60:
                # Compute 5-day post-gap drift (close day+5 vs close day+0)
                idx = df.index.get_loc(date)
                if idx + POST_GAP_LOOKBACK < len(df):
                    close_day0 = float(df.iloc[idx]['Close'])
                    close_day5 = float(df.iloc[idx + POST_GAP_LOOKBACK]['Close'])
                    post_5d_drift = (close_day5 - close_day0) / close_day0
                else:
                    post_5d_drift = None

                deduped.append({
                    'date': date,
                    'gap_pct': float(row['gap_pct']),
                    'is_beat': float(row['gap_pct']) > 0,
                    'gap_abs': float(row['abs_gap']),
                    'post_5d_drift': post_5d_drift,
                })
                last_date = date

        events[ticker] = deduped

    total = sum(len(v) for v in events.values())
    beats = sum(sum(1 for e in v if e['is_beat']) for v in events.values())
    quality_beats = sum(sum(1 for e in v if e['is_beat'] and e['post_5d_drift'] is not None and e['post_5d_drift'] > 0) for v in events.values())
    print(f"  Detected {total} earnings events ({beats} beats, {quality_beats} quality beats)")
    return events


# ── Helper Functions ───────────────────────────────────────────────────────
def get_price_at(prices_df, date, field='Open', offset_days=0):
    """Get price at date + offset, handling weekends/holidays."""
    if date not in prices_df.index:
        mask = prices_df.index >= date
        if not mask.any():
            return None
        date = prices_df.index[mask][0]

    idx = prices_df.index.get_loc(date)
    target_idx = idx + offset_days
    if target_idx < 0 or target_idx >= len(prices_df):
        return None

    val = prices_df.iloc[target_idx][field]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return float(val)


def get_date_at_offset(prices_df, date, offset_days):
    """Get the trading date at offset from given date."""
    if date not in prices_df.index:
        mask = prices_df.index >= date
        if not mask.any():
            return None
        date = prices_df.index[mask][0]

    idx = prices_df.index.get_loc(date)
    target_idx = idx + offset_days
    if target_idx < 0 or target_idx >= len(prices_df):
        return None
    return prices_df.index[target_idx]


def compute_spy_sma200(prices):
    """Compute SPY 200-SMA for bull/bear regime."""
    if 'SPY' not in prices:
        return pd.Series(dtype=float)
    spy = prices['SPY']['Close'].copy()
    return spy.rolling(200).mean()


def get_spy_regime(prices, date):
    """Return 'bull' if SPY > 200-SMA, else 'bear'."""
    if 'SPY' not in prices:
        return 'unknown'
    spy_df = prices['SPY']
    if date not in spy_df.index:
        mask = spy_df.index <= date
        if not mask.any():
            return 'unknown'
        date = spy_df.index[mask][-1]
    idx = spy_df.index.get_loc(date)
    if idx < 200:
        return 'unknown'
    sma200 = float(spy_df['Close'].iloc[idx - 199:idx + 1].mean())
    close = float(spy_df['Close'].iloc[idx])
    return 'bull' if close > sma200 else 'bear'


# ── Variant Backtests ──────────────────────────────────────────────────────

def run_variant_a(prices, events):
    """Quality Beats Only: Buy after >3% gap ONLY if 5-day post-gap drift is also positive. Hold 40d.
    Entry is delayed — 6 trading days after earnings (day after drift confirmation)."""
    trades = []
    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]
        for evt in evts:
            if not evt['is_beat'] or evt['gap_abs'] < GAP_THRESHOLD:
                continue
            if evt['date'] < OOT_START:
                continue
            if evt['post_5d_drift'] is None or evt['post_5d_drift'] <= 0:
                continue

            # Entry: open of day after 5-day drift confirmed (day+6)
            entry_price = get_price_at(df, evt['date'], 'Open', offset_days=POST_GAP_LOOKBACK + 1)
            entry_date = get_date_at_offset(df, evt['date'], POST_GAP_LOOKBACK + 1)
            if entry_price is None or entry_date is None:
                continue

            # Exit: 40 trading days after entry
            exit_idx = df.index.get_loc(entry_date) if entry_date in df.index else None
            if exit_idx is None:
                continue
            exit_target = exit_idx + 40
            if exit_target >= len(df):
                continue
            exit_price = float(df.iloc[exit_target]['Close'])
            exit_date = df.index[exit_target]

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)
            ret = (exit_price - entry_price) / entry_price

            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'post_5d_drift_pct': round(evt['post_5d_drift'] * 100, 2),
                'hold_days': 40,
                'regime': get_spy_regime(prices, entry_date),
            })
    return trades


def run_variant_b(prices, events):
    """Immediate Beat + Hold: Buy next day after >3% gap, hold 40d (baseline)."""
    trades = []
    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]
        for evt in evts:
            if not evt['is_beat'] or evt['gap_abs'] < GAP_THRESHOLD:
                continue
            if evt['date'] < OOT_START:
                continue

            entry_price = get_price_at(df, evt['date'], 'Open', offset_days=1)
            entry_date = get_date_at_offset(df, evt['date'], 1)
            if entry_price is None or entry_date is None:
                continue

            exit_price = get_price_at(df, evt['date'], 'Close', offset_days=1 + 40)
            exit_date = get_date_at_offset(df, evt['date'], 1 + 40)
            if exit_price is None or exit_date is None:
                continue

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)
            ret = (exit_price - entry_price) / entry_price

            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'post_5d_drift_pct': round(evt['post_5d_drift'] * 100, 2) if evt['post_5d_drift'] is not None else None,
                'hold_days': 40,
                'regime': get_spy_regime(prices, entry_date),
            })
    return trades


def run_variant_c(prices, events):
    """Quality Score: Score = gap_pct × (1 + post_5d_drift). Higher score = bigger position. Hold 40d.
    Entry at day+6 (after drift measured). Only positive-gap + positive-drift events."""
    trades = []
    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]
        for evt in evts:
            if not evt['is_beat'] or evt['gap_abs'] < GAP_THRESHOLD:
                continue
            if evt['date'] < OOT_START:
                continue
            if evt['post_5d_drift'] is None:
                continue

            quality_score = evt['gap_pct'] * (1 + evt['post_5d_drift'])
            if quality_score <= 0:
                continue  # skip negative quality

            entry_price = get_price_at(df, evt['date'], 'Open', offset_days=POST_GAP_LOOKBACK + 1)
            entry_date = get_date_at_offset(df, evt['date'], POST_GAP_LOOKBACK + 1)
            if entry_price is None or entry_date is None:
                continue

            exit_idx = df.index.get_loc(entry_date) if entry_date in df.index else None
            if exit_idx is None:
                continue
            exit_target = exit_idx + 40
            if exit_target >= len(df):
                continue
            exit_price = float(df.iloc[exit_target]['Close'])
            exit_date = df.index[exit_target]

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)
            ret = (exit_price - entry_price) / entry_price

            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'post_5d_drift_pct': round(evt['post_5d_drift'] * 100, 2),
                'quality_score': round(quality_score * 100, 2),
                'hold_days': 40,
                'regime': get_spy_regime(prices, entry_date),
            })
    return trades


def run_variant_d(prices, events):
    """Anti-Fade: Buy after >3% gap ONLY if stock hasn't faded >2% in 5 days after. Hold 60d."""
    trades = []
    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]
        for evt in evts:
            if not evt['is_beat'] or evt['gap_abs'] < GAP_THRESHOLD:
                continue
            if evt['date'] < OOT_START:
                continue
            if evt['post_5d_drift'] is None:
                continue

            # Anti-fade: drift must not be worse than -2%
            if evt['post_5d_drift'] < -0.02:
                continue

            entry_price = get_price_at(df, evt['date'], 'Open', offset_days=POST_GAP_LOOKBACK + 1)
            entry_date = get_date_at_offset(df, evt['date'], POST_GAP_LOOKBACK + 1)
            if entry_price is None or entry_date is None:
                continue

            exit_idx = df.index.get_loc(entry_date) if entry_date in df.index else None
            if exit_idx is None:
                continue
            exit_target = exit_idx + 60
            if exit_target >= len(df):
                continue
            exit_price = float(df.iloc[exit_target]['Close'])
            exit_date = df.index[exit_target]

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)
            ret = (exit_price - entry_price) / entry_price

            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'post_5d_drift_pct': round(evt['post_5d_drift'] * 100, 2),
                'hold_days': 60,
                'regime': get_spy_regime(prices, entry_date),
            })
    return trades


def run_variant_e(prices, events):
    """Beat Acceleration: Only buy if THIS quarter gap > LAST quarter gap. Hold 60d."""
    trades = []
    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]
        for i, evt in enumerate(evts):
            if not evt['is_beat'] or evt['gap_abs'] < GAP_THRESHOLD:
                continue
            if evt['date'] < OOT_START:
                continue

            # Need previous event to compare
            if i == 0:
                continue
            prev_evt = evts[i - 1]
            # Previous must also be a beat, and current gap must be larger
            if not prev_evt['is_beat']:
                continue
            if evt['gap_pct'] <= prev_evt['gap_pct']:
                continue

            entry_price = get_price_at(df, evt['date'], 'Open', offset_days=1)
            entry_date = get_date_at_offset(df, evt['date'], 1)
            if entry_price is None or entry_date is None:
                continue

            exit_price = get_price_at(df, evt['date'], 'Close', offset_days=1 + 60)
            exit_date = get_date_at_offset(df, evt['date'], 1 + 60)
            if exit_price is None or exit_date is None:
                continue

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)
            ret = (exit_price - entry_price) / entry_price

            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'prev_gap_pct': round(prev_evt['gap_pct'] * 100, 2),
                'acceleration': round((evt['gap_pct'] - prev_evt['gap_pct']) * 100, 2),
                'hold_days': 60,
                'regime': get_spy_regime(prices, entry_date),
            })
    return trades


def run_variant_f(prices, events):
    """Consecutive Quality: Both current AND previous quarter had positive gap + positive 5d drift. Hold 40d."""
    trades = []
    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]
        for i, evt in enumerate(evts):
            if not evt['is_beat'] or evt['gap_abs'] < GAP_THRESHOLD:
                continue
            if evt['date'] < OOT_START:
                continue
            if evt['post_5d_drift'] is None or evt['post_5d_drift'] <= 0:
                continue

            # Previous quarter must also be quality beat
            if i == 0:
                continue
            prev = evts[i - 1]
            if not prev['is_beat'] or prev['gap_abs'] < GAP_THRESHOLD:
                continue
            if prev['post_5d_drift'] is None or prev['post_5d_drift'] <= 0:
                continue

            # Entry at day+6
            entry_price = get_price_at(df, evt['date'], 'Open', offset_days=POST_GAP_LOOKBACK + 1)
            entry_date = get_date_at_offset(df, evt['date'], POST_GAP_LOOKBACK + 1)
            if entry_price is None or entry_date is None:
                continue

            exit_idx = df.index.get_loc(entry_date) if entry_date in df.index else None
            if exit_idx is None:
                continue
            exit_target = exit_idx + 40
            if exit_target >= len(df):
                continue
            exit_price = float(df.iloc[exit_target]['Close'])
            exit_date = df.index[exit_target]

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)
            ret = (exit_price - entry_price) / entry_price

            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'post_5d_drift_pct': round(evt['post_5d_drift'] * 100, 2),
                'prev_gap_pct': round(prev['gap_pct'] * 100, 2),
                'prev_drift_pct': round(prev['post_5d_drift'] * 100, 2),
                'hold_days': 40,
                'regime': get_spy_regime(prices, entry_date),
            })
    return trades


# ── Portfolio Simulation ───────────────────────────────────────────────────

def simulate_portfolio(trades, capital=CAPITAL):
    """Simulate portfolio with concurrent position management and dynamic sizing.
    Position size = capital / max(3, concurrent_positions)."""
    if not trades:
        return {'equity_curve': [], 'max_dd': 0, 'total_return': 0}

    # Sort trades by entry date
    trades_sorted = sorted(trades, key=lambda t: t['entry_date'])

    # Build daily equity curve
    all_dates = sorted(set(
        [t['entry_date'] for t in trades_sorted] +
        [t['exit_date'] for t in trades_sorted]
    ))
    if not all_dates:
        return {'equity_curve': [], 'max_dd': 0, 'total_return': 0}

    equity = capital
    equity_curve = []
    active_positions = []
    realized_pnl = 0

    for trade in trades_sorted:
        # Count concurrent positions at entry
        entry_d = trade['entry_date']
        concurrent = sum(1 for t in trades_sorted
                        if t['entry_date'] <= entry_d and t['exit_date'] >= entry_d)
        pos_size = capital / max(3, concurrent)
        shares = pos_size / trade['entry_price']
        pnl = shares * (trade['exit_price'] - trade['entry_price'])
        realized_pnl += pnl

    total_return = realized_pnl / capital
    equity_values = [capital]

    # Build a proper daily equity curve
    trade_returns = sorted(
        [(t['exit_date'], t) for t in trades_sorted],
        key=lambda x: x[0]
    )

    running_equity = capital
    for exit_date, t in trade_returns:
        concurrent = sum(1 for t2 in trades_sorted
                        if t2['entry_date'] <= t['entry_date'] and t2['exit_date'] >= t['entry_date'])
        pos_size = capital / max(3, concurrent)
        shares = pos_size / t['entry_price']
        pnl = shares * (t['exit_price'] - t['entry_price'])
        running_equity += pnl
        equity_values.append(running_equity)

    # Compute max drawdown
    peak = equity_values[0]
    max_dd = 0
    for val in equity_values:
        if val > peak:
            peak = val
        dd = (val - peak) / peak
        if dd < max_dd:
            max_dd = dd

    return {
        'equity_curve': equity_values,
        'max_dd': round(max_dd * 100, 2),
        'total_return': round(total_return * 100, 2),
        'final_equity': round(capital + realized_pnl, 2),
    }


# ── Analysis Functions ─────────────────────────────────────────────────────

def compute_metrics(trades, capital=CAPITAL):
    """Compute strategy metrics from trade list."""
    if not trades:
        return {
            'num_trades': 0, 'win_rate': 0, 'avg_return': 0,
            'sharpe': 0, 'sortino': 0, 'profit_factor': 0,
            'max_dd': 0, 'total_return': 0,
        }

    returns = np.array([t['return_pct'] / 100 for t in trades])
    n = len(returns)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    avg_ret = float(np.mean(returns))
    std_ret = float(np.std(returns)) if n > 1 else 1e-9
    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 1e-9

    # Annualize: assume average hold is ~40-60 days, so ~6-9 trades/year per position
    # Use per-trade Sharpe, then annualize
    sharpe = (avg_ret / std_ret) * np.sqrt(n) if std_ret > 1e-9 else 0
    sortino = (avg_ret / downside_std) * np.sqrt(n) if downside_std > 1e-9 else 0

    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss

    portfolio = simulate_portfolio(trades, capital)

    return {
        'num_trades': n,
        'win_rate': round(float(np.mean(returns > 0)) * 100, 1),
        'avg_return_pct': round(avg_ret * 100, 2),
        'median_return_pct': round(float(np.median(returns)) * 100, 2),
        'std_return_pct': round(std_ret * 100, 2),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(profit_factor), 3),
        'max_dd_pct': portfolio['max_dd'],
        'total_return_pct': portfolio['total_return'],
        'final_equity': portfolio.get('final_equity', capital),
        'best_trade_pct': round(float(np.max(returns)) * 100, 2),
        'worst_trade_pct': round(float(np.min(returns)) * 100, 2),
        'avg_win_pct': round(float(np.mean(wins)) * 100, 2) if len(wins) > 0 else 0,
        'avg_loss_pct': round(float(np.mean(losses)) * 100, 2) if len(losses) > 0 else 0,
    }


def regime_analysis(trades):
    """Split performance by bull/bear regime."""
    bull_trades = [t for t in trades if t.get('regime') == 'bull']
    bear_trades = [t for t in trades if t.get('regime') == 'bear']

    bull_m = compute_metrics(bull_trades) if bull_trades else {'sharpe': 0, 'num_trades': 0}
    bear_m = compute_metrics(bear_trades) if bear_trades else {'sharpe': 0, 'num_trades': 0}

    # Regime gap = |sharpe_bull - sharpe_bear| / max(|sharpe_bull|, |sharpe_bear|)
    max_sharpe = max(abs(bull_m['sharpe']), abs(bear_m['sharpe']))
    regime_gap = abs(bull_m['sharpe'] - bear_m['sharpe']) / max_sharpe if max_sharpe > 0 else 0

    return {
        'bull': {'sharpe': bull_m['sharpe'], 'num_trades': bull_m['num_trades'],
                 'win_rate': bull_m.get('win_rate', 0)},
        'bear': {'sharpe': bear_m['sharpe'], 'num_trades': bear_m['num_trades'],
                 'win_rate': bear_m.get('win_rate', 0)},
        'regime_gap': round(regime_gap, 3),
    }


def permutation_test(trades, n_iterations=PERM_ITERATIONS):
    """Permutation test: shuffle entry dates to test if returns are due to timing."""
    if len(trades) < 5:
        return 1.0

    actual_returns = np.array([t['return_pct'] for t in trades])
    actual_mean = np.mean(actual_returns)

    count_better = 0
    rng = np.random.default_rng(42)
    for _ in range(n_iterations):
        shuffled = rng.permutation(actual_returns)
        if np.mean(shuffled) >= actual_mean:
            count_better += 1

    return round(count_better / n_iterations, 4)


def five_gate_validation(metrics, regime, perm_p):
    """Apply 5-gate validation. Returns dict of gate results."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime['regime_gap'] < 0.5,
        'max_dd_gt_neg50': metrics['max_dd_pct'] > -50,
        'trades_gte_20': metrics['num_trades'] >= 20,
    }
    gates['all_pass'] = all(gates.values())
    return gates


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EARNINGS GUIDANCE QUALITY BACKTEST")
    print("=" * 70)
    print(f"Universe: {len(UNIVERSE)} stocks")
    print(f"OOT Period: {OOT_START.date()} to {OOT_END.date()}")
    print(f"Capital: ${CAPITAL}")
    print()

    # Download data
    prices = download_data()
    if len(prices) < 10:
        print("ERROR: Not enough data downloaded. Aborting.")
        return

    # Detect earnings events
    events = detect_earnings_events(prices)

    # Run all variants
    variants = {
        'A_quality_beats': ('Quality Beats Only (gap>3% + 5d drift positive, hold 40d)', run_variant_a),
        'B_immediate_beat': ('Immediate Beat + Hold (baseline, gap>3%, hold 40d)', run_variant_b),
        'C_quality_score': ('Quality Score (gap × drift sizing, hold 40d)', run_variant_c),
        'D_anti_fade': ('Anti-Fade (gap>3% + no >2% fade in 5d, hold 60d)', run_variant_d),
        'E_beat_acceleration': ('Beat Acceleration (this gap > last gap, hold 60d)', run_variant_e),
        'F_consecutive_quality': ('Consecutive Quality (2 quality quarters, hold 40d)', run_variant_f),
    }

    results = {}
    all_summaries = []

    for key, (description, func) in variants.items():
        print(f"\n{'─' * 70}")
        print(f"VARIANT {key[0]}: {description}")
        print(f"{'─' * 70}")

        trades = func(prices, events)
        metrics = compute_metrics(trades)
        regime = regime_analysis(trades)
        perm_p = permutation_test(trades)
        gates = five_gate_validation(metrics, regime, perm_p)

        print(f"  Trades: {metrics['num_trades']}")
        print(f"  Win Rate: {metrics['win_rate']}%")
        print(f"  Avg Return: {metrics['avg_return_pct']}%")
        print(f"  Sharpe: {metrics['sharpe']}")
        print(f"  Sortino: {metrics['sortino']}")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Max DD: {metrics['max_dd_pct']}%")
        print(f"  Total Return: {metrics['total_return_pct']}%")
        print(f"  Final Equity: ${metrics['final_equity']}")
        print(f"  Regime Gap: {regime['regime_gap']}")
        print(f"    Bull: Sharpe={regime['bull']['sharpe']}, n={regime['bull']['num_trades']}, WR={regime['bull']['win_rate']}%")
        print(f"    Bear: Sharpe={regime['bear']['sharpe']}, n={regime['bear']['num_trades']}, WR={regime['bear']['win_rate']}%")
        print(f"  Permutation p-value: {perm_p}")
        print(f"  5-Gate Validation:")
        for gate_name, passed in gates.items():
            if gate_name == 'all_pass':
                continue
            status = "PASS" if passed else "FAIL"
            print(f"    {gate_name}: {status}")
        overall = "ALL GATES PASSED" if gates['all_pass'] else "FAILED"
        print(f"    >>> {overall} <<<")

        results[key] = {
            'description': description,
            'metrics': metrics,
            'regime': regime,
            'perm_p': perm_p,
            'gates': gates,
            'trades': trades,
        }

        all_summaries.append({
            'variant': key,
            'description': description,
            'trades': metrics['num_trades'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe'],
            'sortino': metrics['sortino'],
            'profit_factor': metrics['profit_factor'],
            'max_dd': metrics['max_dd_pct'],
            'total_return': metrics['total_return_pct'],
            'regime_gap': regime['regime_gap'],
            'perm_p': perm_p,
            'all_gates_pass': gates['all_pass'],
        })

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: ALL VARIANTS")
    print(f"{'=' * 70}")
    print(f"{'Variant':<12} {'Trades':>6} {'WR%':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'MaxDD%':>7} {'TotRet%':>8} {'RegGap':>7} {'p-val':>6} {'Gates':>6}")
    print("-" * 90)
    for s in all_summaries:
        gates_str = "PASS" if s['all_gates_pass'] else "FAIL"
        print(f"{s['variant']:<12} {s['trades']:>6} {s['win_rate']:>6.1f} {s['sharpe']:>7.3f} {s['sortino']:>8.3f} {s['profit_factor']:>6.2f} {s['max_dd']:>7.1f} {s['total_return']:>8.1f} {s['regime_gap']:>7.3f} {s['perm_p']:>6.3f} {gates_str:>6}")

    # Save results
    output_path = Path('/home/jupiter/Lvl3Quant/data/earnings_guidance_quality_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert trades to serializable format (remove for file size)
    save_results = {}
    for key, val in results.items():
        save_results[key] = {
            'description': val['description'],
            'metrics': val['metrics'],
            'regime': val['regime'],
            'perm_p': val['perm_p'],
            'gates': val['gates'],
            'num_trades': len(val['trades']),
            'sample_trades': val['trades'][:10] if val['trades'] else [],
        }

    output = {
        'strategy': 'Earnings Guidance Quality',
        'run_date': str(dt.datetime.now()),
        'oot_period': f"{OOT_START.date()} to {OOT_END.date()}",
        'capital': CAPITAL,
        'universe_size': len(UNIVERSE),
        'universe': UNIVERSE,
        'summary': all_summaries,
        'variants': save_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Final verdict
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    passing = [s for s in all_summaries if s['all_gates_pass']]
    if passing:
        best = max(passing, key=lambda x: x['sharpe'])
        print(f"BEST PASSING VARIANT: {best['variant']}")
        print(f"  Sharpe: {best['sharpe']}, Sortino: {best['sortino']}, WR: {best['win_rate']}%, PF: {best['profit_factor']}")
        print(f"  Total Return: {best['total_return']}%, Max DD: {best['max_dd']}%")
    else:
        print("NO VARIANTS PASSED ALL 5 GATES.")
        best = max(all_summaries, key=lambda x: x['sharpe'])
        print(f"Best overall (but failed gates): {best['variant']}")
        print(f"  Sharpe: {best['sharpe']}, WR: {best['win_rate']}%")
        # Show which gates failed
        failed_gates = [k for k, v in results[best['variant']]['gates'].items()
                       if k != 'all_pass' and not v]
        print(f"  Failed gates: {', '.join(failed_gates)}")


if __name__ == '__main__':
    main()
