#!/usr/bin/env python3
"""
ANALYST REVISION MOMENTUM BACKTEST
===================================
Tests price/volume proxies for analyst revision momentum across 6 variants.
Uses yfinance data, $645 starting capital, Robinhood constraints.

Variants:
  A) New 52-week high + volume breakout, hold 21d
  B) Momentum + vol compression, hold 21d
  C) Relative strength breakout vs sector, hold 21d
  D) Consecutive positive weeks + rising volume, hold 40d
  E) Combined: any 2+ of above signals agree, hold 30d
  F) ADVERSARIAL: random entry baseline
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import sys
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

# Force unbuffered output
import builtins as _builtins
def pprint(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _builtins.print(*args, **kwargs)

# ── Config ──────────────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'CRM',
    'PLTR', 'SOFI', 'HOOD', 'SNAP', 'PINS', 'UBER', 'COIN', 'RBLX', 'DDOG', 'TTD',
    'SHOP', 'NET', 'ROKU', 'ABNB', 'SQ', 'DASH', 'CRWD', 'ZS', 'PANW', 'SNOW'
]

# Sector ETF mapping for relative strength
SECTOR_MAP = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLC', 'AMZN': 'XLY', 'META': 'XLC',
    'NVDA': 'XLK', 'TSLA': 'XLY', 'AMD': 'XLK', 'NFLX': 'XLC', 'CRM': 'XLK',
    'PLTR': 'XLK', 'SOFI': 'XLF', 'HOOD': 'XLF', 'SNAP': 'XLC', 'PINS': 'XLC',
    'UBER': 'XLY', 'COIN': 'XLF', 'RBLX': 'XLC', 'DDOG': 'XLK', 'TTD': 'XLK',
    'SHOP': 'XLK', 'NET': 'XLK', 'ROKU': 'XLC', 'ABNB': 'XLY', 'SQ': 'XLK',
    'DASH': 'XLY', 'CRWD': 'XLK', 'ZS': 'XLK', 'PANW': 'XLK', 'SNOW': 'XLK'
}

START_CAPITAL = 645.0
MAX_POSITION = 300.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = '2022-01-01'
OOT_END = '2026-07-25'
N_PERMUTATIONS = 500  # Balance between statistical power and runtime

# ── Data Download ───────────────────────────────────────────────────────────────
def download_data():
    """Download all required price data using bulk download."""
    all_tickers = sorted(set(UNIVERSE + list(SECTOR_MAP.values()) + ['SPY']))
    pprint(f"Downloading data for {len(all_tickers)} tickers (bulk)...")

    data_start = '2021-01-01'  # Extra year for 52-week high calc

    # Bulk download
    raw = yf.download(all_tickers, start=data_start, end=OOT_END, progress=True, auto_adjust=True, group_by='ticker', threads=True)

    prices = {}
    volumes = {}
    failed = []

    for ticker in all_tickers:
        try:
            if len(all_tickers) > 1:
                df = raw[ticker].dropna(how='all')
            else:
                df = raw.dropna(how='all')
            if len(df) > 100:
                prices[ticker] = df['Close'].squeeze()
                if 'Volume' in df.columns:
                    volumes[ticker] = df['Volume'].squeeze()
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)

    if failed:
        pprint(f"  Failed/insufficient data: {failed}")

    price_df = pd.DataFrame(prices)
    vol_df = pd.DataFrame(volumes)

    pprint(f"  Got data for {len(price_df.columns)} tickers, {len(price_df)} trading days")
    return price_df, vol_df


# ── Signal Generation ───────────────────────────────────────────────────────────
def signal_A_52wk_high_volume(prices, volumes, ticker, date_idx):
    """Signal A: New 52-week high + above-average volume."""
    if date_idx < 252:
        return False
    price_hist = prices[ticker].iloc[date_idx-252:date_idx+1]
    if price_hist.isna().sum() > 50:
        return False
    current_price = price_hist.iloc[-1]
    high_252 = price_hist.iloc[:-1].max()
    if np.isnan(current_price) or np.isnan(high_252):
        return False

    # Must be at or above 52-week high
    if current_price < high_252 * 0.99:
        return False

    # Volume must be above 20-day average by 30%+
    if ticker not in volumes.columns:
        return False
    vol_hist = volumes[ticker].iloc[date_idx-20:date_idx+1]
    if vol_hist.isna().sum() > 5:
        return False
    current_vol = vol_hist.iloc[-1]
    avg_vol = vol_hist.iloc[:-1].mean()
    if np.isnan(current_vol) or np.isnan(avg_vol) or avg_vol == 0:
        return False

    return current_vol > avg_vol * 1.3


def signal_B_momentum_vol_compression(prices, ticker, date_idx):
    """Signal B: 21d momentum > 5% + decreasing volatility (vol compression)."""
    if date_idx < 42:
        return False
    price_hist = prices[ticker].iloc[date_idx-42:date_idx+1]
    if price_hist.isna().sum() > 10:
        return False

    current = price_hist.iloc[-1]
    past_21 = price_hist.iloc[-22]
    if np.isnan(current) or np.isnan(past_21) or past_21 == 0:
        return False

    ret_21d = (current / past_21) - 1
    if ret_21d <= 0.05:
        return False

    # Vol compression: recent 10d vol < prior 21d vol
    returns = price_hist.pct_change().dropna()
    if len(returns) < 30:
        return False
    recent_vol = returns.iloc[-10:].std()
    prior_vol = returns.iloc[-31:-10].std()
    if np.isnan(recent_vol) or np.isnan(prior_vol) or prior_vol == 0:
        return False

    return recent_vol < prior_vol * 0.85  # Vol decreased by 15%+


def signal_C_relative_strength(prices, ticker, date_idx, sector_etf):
    """Signal C: 10d return exceeds sector ETF by >3%."""
    if date_idx < 10:
        return False
    if sector_etf not in prices.columns:
        return False

    stock_hist = prices[ticker].iloc[date_idx-10:date_idx+1]
    sector_hist = prices[sector_etf].iloc[date_idx-10:date_idx+1]

    if stock_hist.isna().sum() > 2 or sector_hist.isna().sum() > 2:
        return False

    stock_ret = (stock_hist.iloc[-1] / stock_hist.iloc[0]) - 1
    sector_ret = (sector_hist.iloc[-1] / sector_hist.iloc[0]) - 1

    if np.isnan(stock_ret) or np.isnan(sector_ret):
        return False

    return (stock_ret - sector_ret) > 0.03


def signal_D_consecutive_weeks_rising_vol(prices, volumes, ticker, date_idx):
    """Signal D: 3+ consecutive weeks positive with rising volume."""
    if date_idx < 21:
        return False
    if ticker not in volumes.columns:
        return False

    # Check last 3 weeks (approximately 15 trading days, 5 per week)
    price_hist = prices[ticker].iloc[date_idx-15:date_idx+1]
    vol_hist = volumes[ticker].iloc[date_idx-15:date_idx+1]

    if price_hist.isna().sum() > 3 or vol_hist.isna().sum() > 3:
        return False

    # 3 weekly returns
    week_rets = []
    week_vols = []
    for w in range(3):
        start = w * 5
        end = (w + 1) * 5
        w_prices = price_hist.iloc[start:end+1].dropna()
        w_vols = vol_hist.iloc[start:end+1].dropna()
        if len(w_prices) < 3 or len(w_vols) < 3:
            return False
        week_rets.append((w_prices.iloc[-1] / w_prices.iloc[0]) - 1)
        week_vols.append(w_vols.mean())

    # All 3 weeks positive
    if not all(r > 0 for r in week_rets):
        return False

    # Rising volume (each week higher than prior)
    return week_vols[1] > week_vols[0] and week_vols[2] > week_vols[1]


# ── Backtest Engine ─────────────────────────────────────────────────────────────
class Position:
    def __init__(self, ticker, entry_price, shares, entry_date, hold_days):
        self.ticker = ticker
        self.entry_price = entry_price
        self.shares = shares
        self.entry_date = entry_date
        self.hold_days = hold_days
        self.days_held = 0


def run_backtest(prices, volumes, variant, seed=42):
    """Run a single backtest variant. Returns equity curve and trade list."""
    rng = np.random.RandomState(seed)

    oot_mask = prices.index >= OOT_START
    oot_indices = np.where(oot_mask)[0]
    if len(oot_indices) == 0:
        return None, []

    start_idx = oot_indices[0]
    end_idx = oot_indices[-1]

    capital = START_CAPITAL
    positions = []
    trades = []
    equity_curve = []

    # SPY 200-SMA for regime
    spy_prices = prices.get('SPY')

    for idx in range(start_idx, end_idx + 1):
        date = prices.index[idx]

        # Close expired positions
        new_positions = []
        for pos in positions:
            pos.days_held += 1
            if pos.days_held >= pos.hold_days:
                # Exit
                if pos.ticker in prices.columns:
                    exit_price_raw = prices[pos.ticker].iloc[idx]
                    if not np.isnan(exit_price_raw):
                        exit_price = exit_price_raw * (1 - SLIPPAGE_PCT)
                        pnl = (exit_price - pos.entry_price) * pos.shares
                        capital += exit_price * pos.shares + pnl  # Actually: capital += exit_price * shares
                        # Correct: we get back shares * exit_price
                        capital = capital - exit_price * pos.shares + exit_price * pos.shares
                        # Simpler: capital already excluded entry cost. Just add back sale proceeds.
                        # Let me redo the accounting properly below
                        trades.append({
                            'ticker': pos.ticker,
                            'entry_date': str(pos.entry_date.date()),
                            'exit_date': str(date.date()),
                            'entry_price': pos.entry_price,
                            'exit_price': exit_price,
                            'shares': pos.shares,
                            'pnl': (exit_price - pos.entry_price) * pos.shares,
                            'ret': (exit_price / pos.entry_price) - 1,
                            'hold_days': pos.days_held
                        })
                    else:
                        new_positions.append(pos)
                        continue
                else:
                    new_positions.append(pos)
                    continue
            else:
                new_positions.append(pos)
        positions = new_positions

        # Calculate current equity
        pos_value = 0
        for pos in positions:
            if pos.ticker in prices.columns:
                p = prices[pos.ticker].iloc[idx]
                if not np.isnan(p):
                    pos_value += p * pos.shares

        # Available capital = total equity - position value invested
        # But we need proper accounting. Let me use a simpler model:
        # Track cash separately.
        # Actually let me restart with proper accounting in the outer loop.

        equity_curve.append({'date': str(date.date()), 'equity': capital + pos_value})

        # Check for new entries (only if room for more positions)
        if len(positions) >= MAX_CONCURRENT:
            continue

        # Generate signals for all stocks
        candidates = []
        for ticker in UNIVERSE:
            if ticker not in prices.columns:
                continue
            # Skip if already holding
            if any(p.ticker == ticker for p in positions):
                continue

            if variant == 'A':
                if signal_A_52wk_high_volume(prices, volumes, ticker, idx):
                    candidates.append((ticker, 21))
            elif variant == 'B':
                if signal_B_momentum_vol_compression(prices, ticker, idx):
                    candidates.append((ticker, 21))
            elif variant == 'C':
                sector_etf = SECTOR_MAP.get(ticker, 'XLK')
                if signal_C_relative_strength(prices, ticker, idx, sector_etf):
                    candidates.append((ticker, 21))
            elif variant == 'D':
                if signal_D_consecutive_weeks_rising_vol(prices, volumes, ticker, idx):
                    candidates.append((ticker, 40))
            elif variant == 'E':
                # Combined: 2+ signals agree
                score = 0
                sector_etf = SECTOR_MAP.get(ticker, 'XLK')
                if signal_A_52wk_high_volume(prices, volumes, ticker, idx): score += 1
                if signal_B_momentum_vol_compression(prices, ticker, idx): score += 1
                if signal_C_relative_strength(prices, ticker, idx, sector_etf): score += 1
                if signal_D_consecutive_weeks_rising_vol(prices, volumes, ticker, idx): score += 1
                if score >= 2:
                    candidates.append((ticker, 30))
            elif variant == 'F':
                # Random: ~2% chance per stock per day
                if rng.random() < 0.02:
                    candidates.append((ticker, 21))

        # Enter positions (pick randomly if multiple candidates)
        if candidates:
            rng.shuffle(candidates)
            for ticker, hold_days in candidates:
                if len(positions) >= MAX_CONCURRENT:
                    break

                entry_price_raw = prices[ticker].iloc[idx]
                if np.isnan(entry_price_raw):
                    continue
                entry_price = entry_price_raw * (1 + SLIPPAGE_PCT)

                # Position sizing: min of max_position and available capital
                available = capital
                invest = min(MAX_POSITION, available)
                if invest < 10:  # Min $10 trade
                    continue

                shares = invest / entry_price
                cost = shares * entry_price
                capital -= cost

                positions.append(Position(ticker, entry_price, shares, date, hold_days))

    # Force-close any remaining positions at end
    for pos in positions:
        if pos.ticker in prices.columns:
            exit_price_raw = prices[pos.ticker].iloc[end_idx]
            if not np.isnan(exit_price_raw):
                exit_price = exit_price_raw * (1 - SLIPPAGE_PCT)
                trades.append({
                    'ticker': pos.ticker,
                    'entry_date': str(pos.entry_date.date()),
                    'exit_date': str(prices.index[end_idx].date()),
                    'entry_price': pos.entry_price,
                    'exit_price': exit_price,
                    'shares': pos.shares,
                    'pnl': (exit_price - pos.entry_price) * pos.shares,
                    'ret': (exit_price / pos.entry_price) - 1,
                    'hold_days': pos.days_held
                })

    return equity_curve, trades


def run_backtest_proper(prices, volumes, variant, seed=42):
    """Proper accounting backtest. Returns equity_curve, trades."""
    rng = np.random.RandomState(seed)

    oot_mask = prices.index >= OOT_START
    oot_indices = np.where(oot_mask)[0]
    if len(oot_indices) == 0:
        return [], []

    start_idx = oot_indices[0]
    end_idx = oot_indices[-1]

    cash = START_CAPITAL
    positions = []  # list of Position
    trades = []
    equity_curve = []

    for idx in range(start_idx, end_idx + 1):
        date = prices.index[idx]

        # 1) Close expired positions
        still_open = []
        for pos in positions:
            pos.days_held += 1
            if pos.days_held >= pos.hold_days:
                if pos.ticker in prices.columns:
                    exit_raw = prices[pos.ticker].iloc[idx]
                    if not np.isnan(exit_raw):
                        exit_price = exit_raw * (1 - SLIPPAGE_PCT)
                        proceeds = exit_price * pos.shares
                        cash += proceeds
                        trades.append({
                            'ticker': pos.ticker,
                            'entry_date': str(pos.entry_date.date()),
                            'exit_date': str(date.date()),
                            'entry_price': pos.entry_price,
                            'exit_price': exit_price,
                            'shares': round(pos.shares, 4),
                            'pnl': round((exit_price - pos.entry_price) * pos.shares, 2),
                            'ret': round((exit_price / pos.entry_price) - 1, 6),
                            'hold_days': pos.days_held
                        })
                        continue
                # Can't close (missing data) - keep holding
                still_open.append(pos)
            else:
                still_open.append(pos)
        positions = still_open

        # 2) Mark-to-market equity
        pos_value = 0
        for pos in positions:
            if pos.ticker in prices.columns:
                p = prices[pos.ticker].iloc[idx]
                if not np.isnan(p):
                    pos_value += p * pos.shares
        equity = cash + pos_value
        equity_curve.append({'date': str(date.date()), 'equity': round(equity, 2)})

        # 3) New entries
        if len(positions) >= MAX_CONCURRENT:
            continue

        candidates = []
        for ticker in UNIVERSE:
            if ticker not in prices.columns:
                continue
            if any(p.ticker == ticker for p in positions):
                continue

            if variant == 'A':
                if signal_A_52wk_high_volume(prices, volumes, ticker, idx):
                    candidates.append((ticker, 21))
            elif variant == 'B':
                if signal_B_momentum_vol_compression(prices, ticker, idx):
                    candidates.append((ticker, 21))
            elif variant == 'C':
                sector_etf = SECTOR_MAP.get(ticker, 'XLK')
                if signal_C_relative_strength(prices, ticker, idx, sector_etf):
                    candidates.append((ticker, 21))
            elif variant == 'D':
                if signal_D_consecutive_weeks_rising_vol(prices, volumes, ticker, idx):
                    candidates.append((ticker, 40))
            elif variant == 'E':
                score = 0
                sector_etf = SECTOR_MAP.get(ticker, 'XLK')
                if signal_A_52wk_high_volume(prices, volumes, ticker, idx): score += 1
                if signal_B_momentum_vol_compression(prices, ticker, idx): score += 1
                if signal_C_relative_strength(prices, ticker, idx, sector_etf): score += 1
                if signal_D_consecutive_weeks_rising_vol(prices, volumes, ticker, idx): score += 1
                if score >= 2:
                    candidates.append((ticker, 30))
            elif variant == 'F':
                if rng.random() < 0.02:
                    candidates.append((ticker, 21))

        if candidates:
            rng.shuffle(candidates)
            for ticker, hold_days in candidates:
                if len(positions) >= MAX_CONCURRENT:
                    break

                entry_raw = prices[ticker].iloc[idx]
                if np.isnan(entry_raw):
                    continue
                entry_price = entry_raw * (1 + SLIPPAGE_PCT)

                invest = min(MAX_POSITION, cash)
                if invest < 10:
                    continue

                shares = invest / entry_price
                cost = shares * entry_price
                cash -= cost
                positions.append(Position(ticker, entry_price, shares, date, hold_days))

    # Force-close remaining
    for pos in positions:
        if pos.ticker in prices.columns:
            exit_raw = prices[pos.ticker].iloc[end_idx]
            if not np.isnan(exit_raw):
                exit_price = exit_raw * (1 - SLIPPAGE_PCT)
                cash += exit_price * pos.shares
                trades.append({
                    'ticker': pos.ticker,
                    'entry_date': str(pos.entry_date.date()),
                    'exit_date': str(prices.index[end_idx].date()),
                    'entry_price': pos.entry_price,
                    'exit_price': exit_price,
                    'shares': round(pos.shares, 4),
                    'pnl': round((exit_price - pos.entry_price) * pos.shares, 2),
                    'ret': round((exit_price / pos.entry_price) - 1, 6),
                    'hold_days': pos.days_held
                })

    return equity_curve, trades


# ── Metrics ─────────────────────────────────────────────────────────────────────
def compute_metrics(equity_curve, trades):
    """Compute Sharpe, Sortino, WR, PF, MDD from equity curve and trades."""
    if not equity_curve or not trades:
        return {
            'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
            'max_drawdown': 0, 'n_trades': 0, 'total_return': 0,
            'final_equity': START_CAPITAL, 'avg_trade_ret': 0
        }

    eq = pd.Series([e['equity'] for e in equity_curve])
    daily_rets = eq.pct_change().dropna()

    # Sharpe (annualized)
    if daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_rets.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0

    # Win rate
    trade_rets = [t['ret'] for t in trades]
    winners = [r for r in trade_rets if r > 0]
    wr = len(winners) / len(trade_rets) if trade_rets else 0

    # Profit factor
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    # Max drawdown
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    mdd = dd.min()

    total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(wr, 4),
        'profit_factor': round(min(pf, 999), 3),
        'max_drawdown': round(mdd, 4),
        'n_trades': len(trades),
        'total_return': round(total_ret, 4),
        'final_equity': round(eq.iloc[-1], 2),
        'avg_trade_ret': round(np.mean(trade_rets), 6) if trade_rets else 0
    }


# ── Permutation Test ────────────────────────────────────────────────────────────
def permutation_test(prices, volumes, variant, real_sharpe, n_perms=N_PERMUTATIONS):
    """
    Shuffle which stock is selected at each entry point.
    Returns p-value: fraction of shuffled runs with Sharpe >= real.
    """
    perm_sharpes = []

    for i in range(n_perms):
        seed = 10000 + i
        # Run with a different random seed that shuffles candidates differently
        eq, tr = run_backtest_proper(prices, volumes, variant, seed=seed)
        if eq and tr:
            m = compute_metrics(eq, tr)
            perm_sharpes.append(m['sharpe'])
        else:
            perm_sharpes.append(0)

        if (i + 1) % 200 == 0:
            pprint(f"    Permutation {i+1}/{n_perms}...")

    # For variant F (random), permutation is meaningless — it's already random
    if variant == 'F':
        return 1.0  # Not significant by design

    # For real variants, we compare against the adversarial (random) distribution
    # But the proper permutation test shuffles the stock selection
    # Since our backtest already shuffles candidates, different seeds = different stock picks
    # p-value = fraction of permuted sharpes >= real sharpe
    count_ge = sum(1 for s in perm_sharpes if s >= real_sharpe)
    p_val = count_ge / len(perm_sharpes) if perm_sharpes else 1.0

    return round(p_val, 4)


def permutation_test_shuffle_labels(prices, volumes, variant, real_sharpe, n_perms=N_PERMUTATIONS):
    """
    More proper permutation test: for each entry signal, randomly assign which stock
    from the universe is actually bought (breaking the signal-stock link).
    """
    oot_mask = prices.index >= OOT_START
    oot_indices = np.where(oot_mask)[0]
    if len(oot_indices) == 0:
        return 1.0
    start_idx = oot_indices[0]
    end_idx = oot_indices[-1]

    # First, collect all entry points from the real strategy
    entry_points = []  # (idx, hold_days) - dates when signals fired
    for idx in range(start_idx, end_idx + 1):
        for ticker in UNIVERSE:
            if ticker not in prices.columns:
                continue
            fired = False
            hold_days = 21
            if variant == 'A':
                fired = signal_A_52wk_high_volume(prices, volumes, ticker, idx)
                hold_days = 21
            elif variant == 'B':
                fired = signal_B_momentum_vol_compression(prices, ticker, idx)
                hold_days = 21
            elif variant == 'C':
                sector_etf = SECTOR_MAP.get(ticker, 'XLK')
                fired = signal_C_relative_strength(prices, ticker, idx, sector_etf)
                hold_days = 21
            elif variant == 'D':
                fired = signal_D_consecutive_weeks_rising_vol(prices, volumes, ticker, idx)
                hold_days = 40
            elif variant == 'E':
                score = 0
                sector_etf = SECTOR_MAP.get(ticker, 'XLK')
                if signal_A_52wk_high_volume(prices, volumes, ticker, idx): score += 1
                if signal_B_momentum_vol_compression(prices, ticker, idx): score += 1
                if signal_C_relative_strength(prices, ticker, idx, sector_etf): score += 1
                if signal_D_consecutive_weeks_rising_vol(prices, volumes, ticker, idx): score += 1
                fired = score >= 2
                hold_days = 30

            if fired:
                entry_points.append((idx, ticker, hold_days))

    if not entry_points:
        return 1.0

    # Now run permutations: keep entry dates but shuffle which stocks are bought
    available_stocks = [t for t in UNIVERSE if t in prices.columns]
    perm_sharpes = []

    for perm_i in range(n_perms):
        rng = np.random.RandomState(20000 + perm_i)

        cash = START_CAPITAL
        positions = []
        trades = []
        equity_curve = []

        # Shuffle the stock assignments
        shuffled_entries = []
        for (idx, orig_ticker, hd) in entry_points:
            new_ticker = rng.choice(available_stocks)
            shuffled_entries.append((idx, new_ticker, hd))

        # Sort by index
        shuffled_entries.sort(key=lambda x: x[0])
        entry_iter = iter(shuffled_entries)
        next_entry = next(entry_iter, None)

        for idx in range(oot_indices[0], oot_indices[-1] + 1):
            date = prices.index[idx]

            # Close expired
            still_open = []
            for pos in positions:
                pos.days_held += 1
                if pos.days_held >= pos.hold_days:
                    if pos.ticker in prices.columns:
                        exit_raw = prices[pos.ticker].iloc[idx]
                        if not np.isnan(exit_raw):
                            exit_price = exit_raw * (1 - SLIPPAGE_PCT)
                            cash += exit_price * pos.shares
                            trades.append({
                                'pnl': (exit_price - pos.entry_price) * pos.shares,
                                'ret': (exit_price / pos.entry_price) - 1
                            })
                            continue
                    still_open.append(pos)
                else:
                    still_open.append(pos)
            positions = still_open

            # MTM
            pos_value = sum(
                prices[p.ticker].iloc[idx] * p.shares
                for p in positions
                if p.ticker in prices.columns and not np.isnan(prices[p.ticker].iloc[idx])
            )
            equity_curve.append({'equity': cash + pos_value})

            # Enter from shuffled entries
            while next_entry and next_entry[0] == idx:
                if len(positions) < MAX_CONCURRENT:
                    _, ticker, hold_days = next_entry
                    if ticker in prices.columns:
                        entry_raw = prices[ticker].iloc[idx]
                        if not np.isnan(entry_raw):
                            entry_price = entry_raw * (1 + SLIPPAGE_PCT)
                            invest = min(MAX_POSITION, cash)
                            if invest >= 10:
                                shares = invest / entry_price
                                cash -= shares * entry_price
                                positions.append(Position(ticker, entry_price, shares, date, hold_days))
                next_entry = next(entry_iter, None)
                if next_entry is None:
                    break
            # Consume remaining entries for this idx
            while next_entry and next_entry[0] == idx:
                next_entry = next(entry_iter, None)

        # Close remaining
        for pos in positions:
            if pos.ticker in prices.columns:
                exit_raw = prices[pos.ticker].iloc[oot_indices[-1]]
                if not np.isnan(exit_raw):
                    exit_price = exit_raw * (1 - SLIPPAGE_PCT)
                    cash += exit_price * pos.shares
                    trades.append({
                        'pnl': (exit_price - pos.entry_price) * pos.shares,
                        'ret': (exit_price / pos.entry_price) - 1
                    })

        if equity_curve:
            eq_s = pd.Series([e['equity'] for e in equity_curve])
            dr = eq_s.pct_change().dropna()
            if dr.std() > 0:
                perm_sharpes.append((dr.mean() / dr.std()) * np.sqrt(252))
            else:
                perm_sharpes.append(0)
        else:
            perm_sharpes.append(0)

        if (perm_i + 1) % 200 == 0:
            pprint(f"    Permutation {perm_i+1}/{n_perms}...")

    count_ge = sum(1 for s in perm_sharpes if s >= real_sharpe)
    p_val = count_ge / len(perm_sharpes)
    return round(p_val, 4)


# ── Regime Analysis ─────────────────────────────────────────────────────────────
def regime_gap(prices, trades):
    """Compute regime gap: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|)."""
    if 'SPY' not in prices.columns or not trades:
        return 1.0  # Fail-safe

    spy = prices['SPY']
    spy_sma200 = spy.rolling(200).mean()

    bull_trades = []
    bear_trades = []

    for t in trades:
        entry_date = pd.Timestamp(t['entry_date'])
        if entry_date in spy.index:
            idx = spy.index.get_loc(entry_date)
        else:
            # Find nearest
            idx = spy.index.searchsorted(entry_date)
            if idx >= len(spy):
                idx = len(spy) - 1

        if idx >= 200:
            if spy.iloc[idx] > spy_sma200.iloc[idx]:
                bull_trades.append(t['ret'])
            else:
                bear_trades.append(t['ret'])
        else:
            bull_trades.append(t['ret'])  # Default to bull if not enough history

    if not bull_trades or not bear_trades:
        return 1.0  # Can't compute gap

    bull_rets = np.array(bull_trades)
    bear_rets = np.array(bear_trades)

    sharpe_bull = bull_rets.mean() / bull_rets.std() * np.sqrt(12) if bull_rets.std() > 0 else 0
    sharpe_bear = bear_rets.mean() / bear_rets.std() * np.sqrt(12) if bear_rets.std() > 0 else 0

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear))
    if max_abs == 0:
        return 0

    gap = abs(sharpe_bull - sharpe_bear) / max_abs
    return round(gap, 4)


# ── Gates ───────────────────────────────────────────────────────────────────────
def check_gates(metrics, perm_p, reg_gap):
    """Check all 5 validation gates."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': reg_gap < 0.5,
        'mdd_gt_neg50pct': metrics['max_drawdown'] > -0.50,
        'trades_gte_20': metrics['n_trades'] >= 20
    }
    gates['all_pass'] = all(gates.values())
    return gates


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    pprint("=" * 80)
    pprint("ANALYST REVISION MOMENTUM BACKTEST")
    pprint("=" * 80)
    pprint()

    prices, volumes = download_data()
    pprint()

    variants = {
        'A': '52-week high + volume breakout (hold 21d)',
        'B': 'Momentum + vol compression (hold 21d)',
        'C': 'Relative strength vs sector (hold 21d)',
        'D': 'Consecutive positive weeks + rising vol (hold 40d)',
        'E': 'Combined: 2+ signals agree (hold 30d)',
        'F': 'ADVERSARIAL: random entry (hold 21d)'
    }

    results = {}

    for var_key, var_desc in variants.items():
        pprint(f"\n{'─' * 70}")
        pprint(f"Variant {var_key}: {var_desc}")
        pprint(f"{'─' * 70}")

        # Run backtest
        eq, trades = run_backtest_proper(prices, volumes, var_key, seed=42)
        metrics = compute_metrics(eq, trades)

        pprint(f"  Trades: {metrics['n_trades']}")
        pprint(f"  Final equity: ${metrics['final_equity']:.2f} (from ${START_CAPITAL})")
        pprint(f"  Total return: {metrics['total_return']*100:.1f}%")
        pprint(f"  Sharpe: {metrics['sharpe']:.3f}")
        pprint(f"  Sortino: {metrics['sortino']:.3f}")
        pprint(f"  Win rate: {metrics['win_rate']*100:.1f}%")
        pprint(f"  Profit factor: {metrics['profit_factor']:.3f}")
        pprint(f"  Max drawdown: {metrics['max_drawdown']*100:.1f}%")

        # Regime gap
        rg = regime_gap(prices, trades)
        pprint(f"  Regime gap: {rg:.4f}")

        # Permutation test (skip for F or if too few trades)
        if var_key == 'F' or metrics['n_trades'] < 5:
            perm_p = 1.0
            pprint(f"  Perm p-value: N/A (baseline or too few trades)")
        else:
            pprint(f"  Running permutation test ({N_PERMUTATIONS} shuffles)...")
            perm_p = permutation_test_shuffle_labels(prices, volumes, var_key, metrics['sharpe'], n_perms=N_PERMUTATIONS)
            pprint(f"  Perm p-value: {perm_p:.4f}")

        # Gates
        gates = check_gates(metrics, perm_p, rg)
        pprint(f"  Gates: {gates}")

        results[var_key] = {
            'description': var_desc,
            'metrics': metrics,
            'perm_p': perm_p,
            'regime_gap': rg,
            'gates': gates,
            'sample_trades': trades[:10] if trades else []
        }

    # ── Summary Table ───────────────────────────────────────────────────────────
    pprint("\n\n" + "=" * 100)
    pprint("SUMMARY TABLE")
    pprint("=" * 100)
    pprint(f"{'Var':<4} {'Description':<45} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>7} {'MDD':>7} {'Trades':>7} {'Perm_p':>7} {'RGap':>6} {'Pass':>5}")
    pprint("-" * 100)

    for var_key in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = results[var_key]
        m = r['metrics']
        pass_str = "YES" if r['gates']['all_pass'] else "NO"
        pprint(f"{var_key:<4} {r['description']:<45} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['win_rate']*100:>5.1f}% {m['profit_factor']:>7.3f} {m['max_drawdown']*100:>6.1f}% {m['n_trades']:>7} {r['perm_p']:>7.4f} {r['regime_gap']:>6.4f} {pass_str:>5}")

    pprint("-" * 100)

    # ── Save Results ────────────────────────────────────────────────────────────
    output_path = '/home/jupiter/Lvl3Quant/data/analyst_revision_momentum_results.json'

    # Make JSON serializable
    output = {
        'backtest': 'Analyst Revision Momentum',
        'run_date': str(datetime.now()),
        'config': {
            'universe_size': len(UNIVERSE),
            'oot_period': f'{OOT_START} to {OOT_END}',
            'starting_capital': START_CAPITAL,
            'max_position': MAX_POSITION,
            'max_concurrent': MAX_CONCURRENT,
            'slippage_pct': SLIPPAGE_PCT,
            'n_permutations': N_PERMUTATIONS
        },
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    pprint(f"\nResults saved to {output_path}")

    # ── Verdict ─────────────────────────────────────────────────────────────────
    pprint("\n" + "=" * 80)
    pprint("VERDICT")
    pprint("=" * 80)

    passing = [k for k, v in results.items() if v['gates']['all_pass'] and k != 'F']
    if passing:
        pprint(f"PASSING VARIANTS: {', '.join(passing)}")
        for k in passing:
            m = results[k]['metrics']
            pprint(f"  {k}: Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, WR={m['win_rate']*100:.1f}%, PF={m['profit_factor']:.3f}")
    else:
        pprint("NO VARIANTS PASSED ALL 5 GATES.")
        # Find best anyway
        best_k = max(
            [k for k in results if k != 'F'],
            key=lambda k: results[k]['metrics']['sharpe']
        )
        m = results[best_k]['metrics']
        pprint(f"Best variant: {best_k} (Sharpe={m['sharpe']:.3f}) — failed gates: ", end="")
        failed = [g for g, v in results[best_k]['gates'].items() if not v and g != 'all_pass']
        pprint(", ".join(failed))

    # Check adversarial
    adv = results['F']['metrics']
    best_real = max(results[k]['metrics']['sharpe'] for k in results if k != 'F')
    if adv['sharpe'] >= best_real:
        pprint("\nWARNING: Adversarial (random) baseline matched or beat best strategy — no real edge detected!")
    else:
        pprint(f"\nAdversarial Sharpe: {adv['sharpe']:.3f} vs best strategy: {best_real:.3f} — strategy shows {best_real - adv['sharpe']:.3f} Sharpe advantage over random.")


if __name__ == '__main__':
    main()
