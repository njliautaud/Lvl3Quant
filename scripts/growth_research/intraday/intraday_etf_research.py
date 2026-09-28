#!/usr/bin/env python3
"""
Intraday ETF Growth Strategy Research — Phase 1-3
===================================================
Tests 4 intraday strategies on liquid ETFs using walk-forward validation.

Data: yfinance 1h bars (2 years) + 5m bars (60 days for VWAP/RSI strategies)
Strategies: ORB, Mean Reversion, Gap Fill, VWAP Reversion
Validation: Sliding window walk-forward, regime stratification, permutation tests

Commission: $0 (Robinhood), slippage: 0.01% per side
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/intraday')
SCRIPT_DIR = Path('/home/jupiter/Lvl3Quant/scripts/growth_research/intraday')
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
TICKERS = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLE', 'XLF']
SLIPPAGE_BPS = 1.0  # 0.01% per side = 1 bps
SLIPPAGE_FRAC = SLIPPAGE_BPS / 10000.0
TRAIN_DAYS = 60  # sliding window
OOT_DAYS = 1     # 1 day out-of-sample
N_PERMUTATIONS = 100

# ── Phase 1: Data Download ────────────────────────────────────────────────

def download_data():
    """Download intraday data from yfinance."""
    print("=" * 70)
    print("PHASE 1: DATA DOWNLOAD")
    print("=" * 70)

    all_data = {}

    for ticker in TICKERS:
        print(f"\nDownloading {ticker}...")
        t = yf.Ticker(ticker)

        # 1h bars — 2 years
        h1 = t.history(interval='1h', period='2y')
        if len(h1) > 0:
            h1.to_parquet(DATA_DIR / f'{ticker}_1h.parquet')
            print(f"  1h: {len(h1)} bars, {h1.index[0].date()} to {h1.index[-1].date()}")
            all_data[f'{ticker}_1h'] = h1

        # 5m bars — 60 days
        m5 = t.history(interval='5m', period='60d')
        if len(m5) > 0:
            m5.to_parquet(DATA_DIR / f'{ticker}_5m.parquet')
            print(f"  5m: {len(m5)} bars, {m5.index[0].date()} to {m5.index[-1].date()}")
            all_data[f'{ticker}_5m'] = m5

        # Daily bars — for regime classification and gap calculation
        daily = t.history(interval='1d', period='5y')
        if len(daily) > 0:
            daily.to_parquet(DATA_DIR / f'{ticker}_daily.parquet')
            print(f"  Daily: {len(daily)} bars, {daily.index[0].date()} to {daily.index[-1].date()}")
            all_data[f'{ticker}_daily'] = daily

    # ES proxy for regime classification (use SPY as proxy)
    print("\nData download complete.")
    return all_data


def load_data():
    """Load previously downloaded data."""
    all_data = {}
    for f in DATA_DIR.glob('*.parquet'):
        key = f.stem
        all_data[key] = pd.read_parquet(f)
    return all_data


# ── Regime Classification ─────────────────────────────────────────────────

def classify_regimes(daily_df):
    """Classify each trading day as green/red/flat using close-to-close returns."""
    daily = daily_df.copy()
    daily['date'] = daily.index.date
    daily['ret'] = daily['Close'].pct_change()

    regime = {}
    for date, ret in zip(daily['date'], daily['ret']):
        if pd.isna(ret):
            regime[date] = 'flat'
        elif ret > 0.001:
            regime[date] = 'green'
        elif ret < -0.001:
            regime[date] = 'red'
        else:
            regime[date] = 'flat'
    return regime


# ── Strategy Implementations ──────────────────────────────────────────────

def prepare_daily_bars_from_hourly(h1_df):
    """Group 1h bars into trading days with OHLCV."""
    df = h1_df.copy()
    df['date'] = df.index.date

    daily_bars = []
    for date, group in df.groupby('date'):
        if len(group) < 4:  # need at least 4 hours of data
            continue
        daily_bars.append({
            'date': date,
            'Open': group['Open'].iloc[0],
            'High': group['High'].max(),
            'Low': group['Low'].min(),
            'Close': group['Close'].iloc[-1],
            'Volume': group['Volume'].sum(),
            'bars': group,
        })
    return daily_bars


class StrategyORB:
    """Opening Range Breakout using 1h bars.

    Uses first hour's range as the opening range.
    Enter on breakout, stop at opposite side, target at R:R ratio.
    """

    def __init__(self, rr_ratio=1.5):
        self.rr_ratio = rr_ratio
        self.name = f"ORB_RR{rr_ratio}"

    def run_day(self, day_bars, params=None):
        """Run strategy on a single day's 1h bars. Returns trade result."""
        rr = params.get('rr_ratio', self.rr_ratio) if params else self.rr_ratio
        bars = day_bars['bars']

        if len(bars) < 3:
            return None

        # First bar = opening range
        or_high = bars['High'].iloc[0]
        or_low = bars['Low'].iloc[0]
        or_range = or_high - or_low

        if or_range <= 0:
            return None

        # Check subsequent bars for breakout
        for i in range(1, len(bars)):
            bar = bars.iloc[i]

            # Long breakout
            if bar['High'] > or_high:
                entry = or_high * (1 + SLIPPAGE_FRAC)
                stop = or_low
                target = entry + rr * or_range

                # Simulate within remaining bars
                pnl = self._simulate_trade(bars.iloc[i:], entry, stop, target, 'long')
                return {'direction': 'long', 'entry': entry, 'pnl_pct': pnl}

            # Short breakout
            if bar['Low'] < or_low:
                entry = or_low * (1 - SLIPPAGE_FRAC)
                stop = or_high
                target = entry - rr * or_range

                pnl = self._simulate_trade(bars.iloc[i:], entry, stop, target, 'short')
                return {'direction': 'short', 'entry': entry, 'pnl_pct': pnl}

        return None  # No breakout

    def _simulate_trade(self, bars, entry, stop, target, direction):
        """Simulate trade through remaining bars, apply slippage on exit."""
        for i in range(len(bars)):
            bar = bars.iloc[i]
            if direction == 'long':
                if bar['Low'] <= stop:
                    exit_price = stop * (1 - SLIPPAGE_FRAC)
                    return (exit_price - entry) / entry
                if bar['High'] >= target:
                    exit_price = target * (1 - SLIPPAGE_FRAC)
                    return (exit_price - entry) / entry
            else:  # short
                if bar['High'] >= stop:
                    exit_price = stop * (1 + SLIPPAGE_FRAC)
                    return (entry - exit_price) / entry
                if bar['Low'] <= target:
                    exit_price = target * (1 + SLIPPAGE_FRAC)
                    return (entry - exit_price) / entry

        # Close at end of day
        exit_price = bars['Close'].iloc[-1]
        if direction == 'long':
            exit_price *= (1 - SLIPPAGE_FRAC)
            return (exit_price - entry) / entry
        else:
            exit_price *= (1 + SLIPPAGE_FRAC)
            return (entry - exit_price) / entry


class StrategyGapFill:
    """Gap Fill Strategy using daily/1h bars.

    Measures overnight gap. If gap > threshold, trade toward fill.
    """

    def __init__(self, gap_threshold=0.003):
        self.gap_threshold = gap_threshold
        self.name = f"GapFill_{gap_threshold}"

    def run_day(self, day_bars, prev_close, params=None):
        """Run on a single day. prev_close is yesterday's close."""
        threshold = params.get('gap_threshold', self.gap_threshold) if params else self.gap_threshold
        bars = day_bars['bars']

        if prev_close is None or len(bars) < 2:
            return None

        today_open = bars['Open'].iloc[0]
        gap_pct = (today_open - prev_close) / prev_close

        if abs(gap_pct) < threshold:
            return None  # Gap too small

        # Trade toward gap fill
        if gap_pct > 0:  # Gap up → short toward prev close
            entry = today_open * (1 - SLIPPAGE_FRAC)
            target = prev_close
            stop = today_open * (1 + abs(gap_pct))  # Stop at double gap
            direction = 'short'
        else:  # Gap down → long toward prev close
            entry = today_open * (1 + SLIPPAGE_FRAC)
            target = prev_close
            stop = today_open * (1 - abs(gap_pct))  # Stop at double gap
            direction = 'long'

        # Simulate through hourly bars (exit by lunch = first 4 bars)
        max_bars = min(4, len(bars))
        for i in range(max_bars):
            bar = bars.iloc[i]
            if direction == 'long':
                if bar['Low'] <= stop:
                    exit_p = stop * (1 - SLIPPAGE_FRAC)
                    return {'direction': direction, 'entry': entry, 'pnl_pct': (exit_p - entry) / entry, 'gap_pct': gap_pct}
                if bar['High'] >= target:
                    exit_p = target * (1 - SLIPPAGE_FRAC)
                    return {'direction': direction, 'entry': entry, 'pnl_pct': (exit_p - entry) / entry, 'gap_pct': gap_pct}
            else:
                if bar['High'] >= stop:
                    exit_p = stop * (1 + SLIPPAGE_FRAC)
                    return {'direction': direction, 'entry': entry, 'pnl_pct': (entry - exit_p) / entry, 'gap_pct': gap_pct}
                if bar['Low'] <= target:
                    exit_p = target * (1 + SLIPPAGE_FRAC)
                    return {'direction': direction, 'entry': entry, 'pnl_pct': (entry - exit_p) / entry, 'gap_pct': gap_pct}

        # Exit at end of window
        exit_p = bars['Close'].iloc[max_bars - 1]
        if direction == 'long':
            exit_p *= (1 - SLIPPAGE_FRAC)
            pnl = (exit_p - entry) / entry
        else:
            exit_p *= (1 + SLIPPAGE_FRAC)
            pnl = (entry - exit_p) / entry

        return {'direction': direction, 'entry': entry, 'pnl_pct': pnl, 'gap_pct': gap_pct}


class StrategyMeanReversion:
    """Hourly RSI mean reversion.

    When RSI on 1h bars hits extremes, fade the move.
    Target reversion to mean, stop at further extreme.
    """

    def __init__(self, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.name = f"MeanRev_RSI{rsi_period}_{rsi_oversold}_{rsi_overbought}"

    def compute_rsi(self, prices, period=14):
        """Compute RSI from a price series."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def run_on_hourly(self, h1_df, params=None):
        """Run on full hourly dataframe, return list of trades."""
        period = params.get('rsi_period', self.rsi_period) if params else self.rsi_period
        oversold = params.get('rsi_oversold', self.rsi_oversold) if params else self.rsi_oversold
        overbought = params.get('rsi_overbought', self.rsi_overbought) if params else self.rsi_overbought

        df = h1_df.copy()
        df['rsi'] = self.compute_rsi(df['Close'], period)
        df['date'] = df.index.date

        trades = []
        in_trade = False
        entry_price = None
        direction = None
        entry_date = None
        bars_held = 0
        max_hold = 6  # max 6 hours

        for i in range(len(df)):
            row = df.iloc[i]

            if in_trade:
                bars_held += 1
                if direction == 'long':
                    # Exit when RSI > 50 or max hold
                    if row['rsi'] > 50 or bars_held >= max_hold:
                        exit_p = row['Close'] * (1 - SLIPPAGE_FRAC)
                        pnl = (exit_p - entry_price) / entry_price
                        trades.append({'date': entry_date, 'direction': direction,
                                      'pnl_pct': pnl, 'bars_held': bars_held})
                        in_trade = False
                else:
                    if row['rsi'] < 50 or bars_held >= max_hold:
                        exit_p = row['Close'] * (1 + SLIPPAGE_FRAC)
                        pnl = (entry_price - exit_p) / entry_price
                        trades.append({'date': entry_date, 'direction': direction,
                                      'pnl_pct': pnl, 'bars_held': bars_held})
                        in_trade = False

            if not in_trade and not pd.isna(row['rsi']):
                if row['rsi'] < oversold:
                    entry_price = row['Close'] * (1 + SLIPPAGE_FRAC)
                    direction = 'long'
                    entry_date = row['date']
                    in_trade = True
                    bars_held = 0
                elif row['rsi'] > overbought:
                    entry_price = row['Close'] * (1 - SLIPPAGE_FRAC)
                    direction = 'short'
                    entry_date = row['date']
                    in_trade = True
                    bars_held = 0

        return trades


class StrategyVWAPReversion:
    """VWAP Reversion using 1h bars.

    Compute running VWAP each day.
    When price deviates > threshold from VWAP, trade reversion.
    """

    def __init__(self, std_threshold=1.5):
        self.std_threshold = std_threshold
        self.name = f"VWAP_Rev_{std_threshold}std"

    def run_day(self, day_bars, params=None):
        """Run on a single day's hourly bars."""
        threshold = params.get('std_threshold', self.std_threshold) if params else self.std_threshold
        bars = day_bars['bars']

        if len(bars) < 4:
            return None

        # Compute VWAP
        cum_vol = bars['Volume'].cumsum()
        cum_vp = (bars['Close'] * bars['Volume']).cumsum()
        vwap = cum_vp / cum_vol.replace(0, np.nan)

        # Compute deviation std from rolling window
        deviation = bars['Close'] - vwap
        dev_std = deviation.rolling(3, min_periods=2).std()

        trades = []
        in_trade = False

        for i in range(3, len(bars)):
            if pd.isna(dev_std.iloc[i]) or dev_std.iloc[i] == 0:
                continue

            z_score = deviation.iloc[i] / dev_std.iloc[i]

            if not in_trade:
                if z_score > threshold:
                    # Price way above VWAP → short
                    entry = bars['Close'].iloc[i] * (1 - SLIPPAGE_FRAC)
                    in_trade = True
                    direction = 'short'
                    entry_bar = i
                elif z_score < -threshold:
                    # Price way below VWAP → long
                    entry = bars['Close'].iloc[i] * (1 + SLIPPAGE_FRAC)
                    in_trade = True
                    direction = 'long'
                    entry_bar = i
            elif in_trade:
                # Exit when z_score reverts to 0 or at day end
                if abs(z_score) < 0.5 or i == len(bars) - 1:
                    exit_p = bars['Close'].iloc[i]
                    if direction == 'long':
                        exit_p *= (1 - SLIPPAGE_FRAC)
                        pnl = (exit_p - entry) / entry
                    else:
                        exit_p *= (1 + SLIPPAGE_FRAC)
                        pnl = (entry - exit_p) / entry
                    trades.append({'direction': direction, 'pnl_pct': pnl})
                    in_trade = False

        if not trades:
            return None

        total_pnl = sum(t['pnl_pct'] for t in trades)
        return {'pnl_pct': total_pnl, 'n_trades': len(trades)}


# ── Walk-Forward Engine ───────────────────────────────────────────────────

def sliding_window_walkforward(daily_bars, strategy_func, param_grid, regimes,
                                train_window=60, oot_days=1):
    """
    Sliding window walk-forward optimization.

    - Train on `train_window` days, test on `oot_days` days
    - Slide by `oot_days` each step
    - On train window: find best params
    - On OOT window: trade with those params

    Returns list of OOT trade results with dates and regimes.
    """
    n_days = len(daily_bars)
    if n_days < train_window + oot_days:
        return []

    oot_results = []

    for start in range(0, n_days - train_window - oot_days + 1, oot_days):
        train_slice = daily_bars[start:start + train_window]
        oot_slice = daily_bars[start + train_window:start + train_window + oot_days]

        # Optimize on train window
        best_params = None
        best_sharpe = -np.inf

        for params in param_grid:
            train_pnls = []
            for day in train_slice:
                result = strategy_func(day, params)
                if result is not None:
                    train_pnls.append(result['pnl_pct'])

            if len(train_pnls) > 5:
                sharpe = np.mean(train_pnls) / (np.std(train_pnls) + 1e-10) * np.sqrt(252)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params

        if best_params is None:
            best_params = param_grid[0]

        # Test on OOT
        for day in oot_slice:
            result = strategy_func(day, best_params)
            date = day['date']
            regime = regimes.get(date, 'flat')

            if result is not None:
                oot_results.append({
                    'date': date,
                    'regime': regime,
                    'pnl_pct': result['pnl_pct'],
                    'direction': result.get('direction', 'mixed'),
                    'params': str(best_params),
                })
            else:
                # No trade day
                oot_results.append({
                    'date': date,
                    'regime': regime,
                    'pnl_pct': 0.0,
                    'direction': 'none',
                    'params': str(best_params),
                })

    return oot_results


# ── Risk Metrics ──────────────────────────────────────────────────────────

def compute_metrics(oot_results):
    """Compute risk-adjusted metrics from OOT results."""
    if not oot_results:
        return {}

    pnls = [r['pnl_pct'] for r in oot_results]
    pnls = np.array(pnls)

    # Filter to actual trade days
    trade_pnls = pnls[pnls != 0]

    n_days = len(pnls)
    n_trades = len(trade_pnls)

    if n_trades == 0:
        return {'n_days': n_days, 'n_trades': 0, 'total_return': 0}

    # Daily metrics (including no-trade days as 0)
    mean_daily = np.mean(pnls)
    std_daily = np.std(pnls)

    # Annualized
    ann_return = mean_daily * 252
    sharpe = mean_daily / (std_daily + 1e-10) * np.sqrt(252)

    # Sortino
    downside = pnls[pnls < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-10
    sortino = mean_daily / (downside_std + 1e-10) * np.sqrt(252)

    # Win rate (on actual trades)
    wr = np.mean(trade_pnls > 0) if len(trade_pnls) > 0 else 0

    # Profit factor
    gross_profit = np.sum(trade_pnls[trade_pnls > 0])
    gross_loss = abs(np.sum(trade_pnls[trade_pnls < 0]))
    pf = gross_profit / (gross_loss + 1e-10)

    # Max drawdown
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_dd = np.min(drawdown)

    # CAGR approximation
    total_return = np.sum(pnls)
    years = n_days / 252
    if years > 0 and total_return > -1:
        cagr = (1 + total_return) ** (1 / years) - 1
    else:
        cagr = -1

    # Calmar
    calmar = cagr / (abs(max_dd) + 1e-10)

    return {
        'n_days': n_days,
        'n_trades': n_trades,
        'trade_freq': n_trades / n_days,
        'total_return_pct': total_return * 100,
        'cagr_pct': cagr * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'win_rate': wr,
        'profit_factor': pf,
        'max_dd_pct': max_dd * 100,
        'calmar': calmar,
        'avg_trade_pct': np.mean(trade_pnls) * 100,
        'median_trade_pct': np.median(trade_pnls) * 100,
    }


def regime_analysis(oot_results):
    """Stratify results by regime. Check R1 gap test."""
    by_regime = defaultdict(list)
    for r in oot_results:
        by_regime[r['regime']].append(r['pnl_pct'])

    regime_metrics = {}
    for regime, pnls in by_regime.items():
        pnls = np.array(pnls)
        trade_pnls = pnls[pnls != 0]
        n = len(pnls)
        mean_d = np.mean(pnls)
        std_d = np.std(pnls) + 1e-10
        sharpe = mean_d / std_d * np.sqrt(252)
        wr = np.mean(trade_pnls > 0) if len(trade_pnls) > 0 else 0
        regime_metrics[regime] = {
            'n_days': n,
            'n_trades': len(trade_pnls),
            'sharpe': sharpe,
            'win_rate': wr,
            'total_return_pct': np.sum(pnls) * 100,
            'avg_daily_pct': mean_d * 100,
        }

    # R1 test
    sharpes = {k: v['sharpe'] for k, v in regime_metrics.items()}
    s_green = sharpes.get('green', 0)
    s_red = sharpes.get('red', 0)
    max_s = max(abs(s_green), abs(s_red))
    r1_gap = abs(s_green - s_red) / (max_s + 1e-10) if max_s > 0 else 0
    r1_pass = r1_gap <= 0.50

    return regime_metrics, r1_gap, r1_pass


def permutation_test(oot_results, n_perms=100):
    """Permutation test: shuffle trade directions, compute p-value."""
    if not oot_results:
        return 1.0

    actual_pnls = np.array([r['pnl_pct'] for r in oot_results])
    actual_sharpe = np.mean(actual_pnls) / (np.std(actual_pnls) + 1e-10) * np.sqrt(252)

    # Shuffle signs of returns
    count_better = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(actual_pnls))
        shuffled = actual_pnls * signs
        perm_sharpe = np.mean(shuffled) / (np.std(shuffled) + 1e-10) * np.sqrt(252)
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    p_value = count_better / n_perms
    return p_value


# ── Main Research Loop ────────────────────────────────────────────────────

def run_research():
    """Main research function."""

    # Phase 1: Download data
    print("\n" + "=" * 70)
    print("INTRADAY ETF GROWTH STRATEGY RESEARCH")
    print("=" * 70)

    # Check if data already downloaded
    existing = list(DATA_DIR.glob('*_1h.parquet'))
    if len(existing) >= len(TICKERS):
        print("Data already downloaded, loading...")
        all_data = load_data()
    else:
        all_data = download_data()

    # Summary
    print("\n── Data Summary ──")
    for key in sorted(all_data.keys()):
        df = all_data[key]
        print(f"  {key}: {len(df)} bars")

    # Phase 2: Run strategies
    print("\n" + "=" * 70)
    print("PHASE 2: STRATEGY BACKTESTS (Walk-Forward)")
    print("=" * 70)

    all_strategy_results = {}

    for ticker in TICKERS:
        h1_key = f'{ticker}_1h'
        daily_key = f'{ticker}_daily'

        if h1_key not in all_data or daily_key not in all_data:
            print(f"\nSkipping {ticker} — missing data")
            continue

        h1_df = all_data[h1_key]
        daily_df = all_data[daily_key]

        # Classify regimes using SPY (or self for SPY)
        regime_source = all_data.get('SPY_daily', daily_df)
        regimes = classify_regimes(regime_source)

        # Prepare daily bars from hourly
        daily_bars = prepare_daily_bars_from_hourly(h1_df)
        print(f"\n{'─' * 50}")
        print(f"TICKER: {ticker} — {len(daily_bars)} trading days from 1h bars")

        # ── Strategy A: Opening Range Breakout ──
        print(f"\n  [A] Opening Range Breakout...")
        orb = StrategyORB()
        orb_param_grid = [{'rr_ratio': rr} for rr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]

        def orb_func(day, params):
            return orb.run_day(day, params)

        orb_oot = sliding_window_walkforward(
            daily_bars, orb_func, orb_param_grid, regimes,
            train_window=TRAIN_DAYS, oot_days=OOT_DAYS
        )
        orb_metrics = compute_metrics(orb_oot)
        all_strategy_results[f'{ticker}_ORB'] = {'metrics': orb_metrics, 'oot': orb_oot}
        print(f"      CAGR: {orb_metrics.get('cagr_pct', 0):.1f}%  Sharpe: {orb_metrics.get('sharpe', 0):.2f}  "
              f"WR: {orb_metrics.get('win_rate', 0):.1%}  PF: {orb_metrics.get('profit_factor', 0):.2f}  "
              f"MaxDD: {orb_metrics.get('max_dd_pct', 0):.1f}%  Trades: {orb_metrics.get('n_trades', 0)}")

        # ── Strategy B: Mean Reversion (RSI) ──
        print(f"\n  [B] Mean Reversion (Hourly RSI)...")
        mr = StrategyMeanReversion()

        # Walk-forward on hourly RSI: optimize RSI params on train window
        mr_param_grid = [
            {'rsi_period': p, 'rsi_oversold': os, 'rsi_overbought': ob}
            for p in [7, 14, 21]
            for os, ob in [(20, 80), (25, 75), (30, 70), (35, 65)]
        ]

        # For mean reversion, we need to run on contiguous hourly blocks
        # Split hourly data into train/test windows by date
        dates = sorted(set(h1_df.index.date))
        mr_oot_results = []

        for start_idx in range(0, len(dates) - TRAIN_DAYS - OOT_DAYS + 1, OOT_DAYS):
            train_dates = dates[start_idx:start_idx + TRAIN_DAYS]
            oot_dates = dates[start_idx + TRAIN_DAYS:start_idx + TRAIN_DAYS + OOT_DAYS]

            train_set = set(train_dates)
            oot_set = set(oot_dates)
            idx_dates = np.array([d.date() for d in h1_df.index])
            train_mask = np.array([d in train_set for d in idx_dates])
            oot_mask = np.array([d in oot_set for d in idx_dates])
            train_h1 = h1_df[train_mask]
            oot_h1 = h1_df[oot_mask]

            if len(train_h1) < 50 or len(oot_h1) < 2:
                continue

            # Optimize on train
            best_params = None
            best_sharpe = -np.inf

            for params in mr_param_grid:
                trades = mr.run_on_hourly(train_h1, params)
                if len(trades) > 3:
                    pnls = [t['pnl_pct'] for t in trades]
                    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params

            if best_params is None:
                best_params = mr_param_grid[0]

            # Test on OOT
            # Need enough context for RSI, so prepend some train data
            context_h1 = pd.concat([train_h1.tail(50), oot_h1])
            trades = mr.run_on_hourly(context_h1, best_params)

            # Filter to OOT dates only
            for t in trades:
                if t['date'] in set(oot_dates):
                    regime = regimes.get(t['date'], 'flat')
                    mr_oot_results.append({
                        'date': t['date'],
                        'regime': regime,
                        'pnl_pct': t['pnl_pct'],
                        'direction': t['direction'],
                        'params': str(best_params),
                    })

            # Add no-trade days
            traded_dates = set(t['date'] for t in trades if t['date'] in set(oot_dates))
            for d in oot_dates:
                if d not in traded_dates:
                    mr_oot_results.append({
                        'date': d, 'regime': regimes.get(d, 'flat'),
                        'pnl_pct': 0.0, 'direction': 'none', 'params': str(best_params)
                    })

        mr_metrics = compute_metrics(mr_oot_results)
        all_strategy_results[f'{ticker}_MeanRev'] = {'metrics': mr_metrics, 'oot': mr_oot_results}
        print(f"      CAGR: {mr_metrics.get('cagr_pct', 0):.1f}%  Sharpe: {mr_metrics.get('sharpe', 0):.2f}  "
              f"WR: {mr_metrics.get('win_rate', 0):.1%}  PF: {mr_metrics.get('profit_factor', 0):.2f}  "
              f"MaxDD: {mr_metrics.get('max_dd_pct', 0):.1f}%  Trades: {mr_metrics.get('n_trades', 0)}")

        # ── Strategy C: Gap Fill ──
        print(f"\n  [C] Gap Fill...")
        gf = StrategyGapFill()
        gf_param_grid = [{'gap_threshold': t} for t in [0.002, 0.003, 0.005, 0.007, 0.01]]

        def gf_func(day, params):
            # Need prev_close
            idx = daily_bars.index(day) if day in daily_bars else -1
            if idx <= 0:
                return None
            prev_close = daily_bars[idx - 1]['Close']
            return gf.run_day(day, prev_close, params)

        # Manual walk-forward since we need prev_close context
        gf_oot_results = []
        for start in range(0, len(daily_bars) - TRAIN_DAYS - OOT_DAYS + 1, OOT_DAYS):
            train_slice = daily_bars[start:start + TRAIN_DAYS]
            oot_slice = daily_bars[start + TRAIN_DAYS:start + TRAIN_DAYS + OOT_DAYS]

            # Optimize
            best_params = None
            best_sharpe = -np.inf

            for params in gf_param_grid:
                pnls = []
                for i, day in enumerate(train_slice):
                    if i == 0:
                        continue
                    prev_close = train_slice[i-1]['Close']
                    result = gf.run_day(day, prev_close, params)
                    if result is not None:
                        pnls.append(result['pnl_pct'])

                if len(pnls) > 3:
                    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params

            if best_params is None:
                best_params = gf_param_grid[0]

            # OOT
            for day in oot_slice:
                idx = daily_bars.index(day)
                prev_close = daily_bars[idx - 1]['Close'] if idx > 0 else None
                result = gf.run_day(day, prev_close, best_params)
                date = day['date']
                regime = regimes.get(date, 'flat')

                if result is not None:
                    gf_oot_results.append({
                        'date': date, 'regime': regime, 'pnl_pct': result['pnl_pct'],
                        'direction': result['direction'], 'params': str(best_params)
                    })
                else:
                    gf_oot_results.append({
                        'date': date, 'regime': regime, 'pnl_pct': 0.0,
                        'direction': 'none', 'params': str(best_params)
                    })

        gf_metrics = compute_metrics(gf_oot_results)
        all_strategy_results[f'{ticker}_GapFill'] = {'metrics': gf_metrics, 'oot': gf_oot_results}
        print(f"      CAGR: {gf_metrics.get('cagr_pct', 0):.1f}%  Sharpe: {gf_metrics.get('sharpe', 0):.2f}  "
              f"WR: {gf_metrics.get('win_rate', 0):.1%}  PF: {gf_metrics.get('profit_factor', 0):.2f}  "
              f"MaxDD: {gf_metrics.get('max_dd_pct', 0):.1f}%  Trades: {gf_metrics.get('n_trades', 0)}")

        # ── Strategy D: VWAP Reversion ──
        print(f"\n  [D] VWAP Reversion...")
        vwap = StrategyVWAPReversion()
        vwap_param_grid = [{'std_threshold': t} for t in [1.0, 1.5, 2.0, 2.5, 3.0]]

        def vwap_func(day, params):
            return vwap.run_day(day, params)

        vwap_oot = sliding_window_walkforward(
            daily_bars, vwap_func, vwap_param_grid, regimes,
            train_window=TRAIN_DAYS, oot_days=OOT_DAYS
        )
        vwap_metrics = compute_metrics(vwap_oot)
        all_strategy_results[f'{ticker}_VWAP'] = {'metrics': vwap_metrics, 'oot': vwap_oot}
        print(f"      CAGR: {vwap_metrics.get('cagr_pct', 0):.1f}%  Sharpe: {vwap_metrics.get('sharpe', 0):.2f}  "
              f"WR: {vwap_metrics.get('win_rate', 0):.1%}  PF: {vwap_metrics.get('profit_factor', 0):.2f}  "
              f"MaxDD: {vwap_metrics.get('max_dd_pct', 0):.1f}%  Trades: {vwap_metrics.get('n_trades', 0)}")

    # Phase 3: Find best strategy and do deep analysis
    print("\n" + "=" * 70)
    print("PHASE 3: ANALYSIS — BEST STRATEGY")
    print("=" * 70)

    # Rank by Sharpe (only strategies with >10 trades)
    ranked = []
    for key, val in all_strategy_results.items():
        m = val['metrics']
        if m.get('n_trades', 0) > 10 and m.get('sharpe', 0) > 0:
            ranked.append((key, m))

    ranked.sort(key=lambda x: x[1].get('sharpe', 0), reverse=True)

    print("\n── Strategy Ranking (by Sharpe, min 10 trades) ──")
    print(f"{'Strategy':<25} {'CAGR%':>8} {'Sharpe':>8} {'Sortino':>8} {'WR':>8} {'PF':>8} {'MaxDD%':>8} {'Trades':>8}")
    print("─" * 85)
    for key, m in ranked[:15]:
        print(f"{key:<25} {m['cagr_pct']:>7.1f}% {m['sharpe']:>8.2f} {m['sortino']:>8.2f} "
              f"{m['win_rate']:>7.1%} {m['profit_factor']:>8.2f} {m['max_dd_pct']:>7.1f}% {m['n_trades']:>8}")

    if not ranked:
        print("\n*** NO STRATEGIES PRODUCED POSITIVE SHARPE WITH >10 TRADES ***")
        print("This is an honest result. Intraday strategies on hourly bars may not")
        print("have enough resolution for consistent edge.")

        # Still show all results
        print("\n── All Results (including negative) ──")
        print(f"{'Strategy':<25} {'CAGR%':>8} {'Sharpe':>8} {'WR':>8} {'Trades':>8}")
        print("─" * 55)
        for key, val in sorted(all_strategy_results.items()):
            m = val['metrics']
            print(f"{key:<25} {m.get('cagr_pct', 0):>7.1f}% {m.get('sharpe', 0):>8.2f} "
                  f"{m.get('win_rate', 0):>7.1%} {m.get('n_trades', 0):>8}")

    # Deep analysis on top 3
    print("\n── Deep Analysis on Top Strategies ──")
    for key, m in ranked[:3]:
        print(f"\n{'=' * 60}")
        print(f"STRATEGY: {key}")
        print(f"{'=' * 60}")

        oot = all_strategy_results[key]['oot']

        # Full metrics
        print(f"\n  Performance:")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

        # Regime analysis
        regime_metrics, r1_gap, r1_pass = regime_analysis(oot)
        print(f"\n  Regime Stratification:")
        for regime, rm in sorted(regime_metrics.items()):
            print(f"    {regime:>6}: Sharpe={rm['sharpe']:.2f}  WR={rm['win_rate']:.1%}  "
                  f"Return={rm['total_return_pct']:.2f}%  Days={rm['n_days']}")
        print(f"    R1 Gap: {r1_gap:.3f} ({'PASS' if r1_pass else 'FAIL — regime-dependent'})")

        # Permutation test
        p_val = permutation_test(oot, N_PERMUTATIONS)
        print(f"\n  Permutation Test (100 trials): p-value = {p_val:.3f} "
              f"({'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'})")

        # Slippage sensitivity
        print(f"\n  Slippage Sensitivity:")
        for slip_bps in [0, 1, 2, 5, 10]:
            # Adjust returns for different slippage levels
            adj_pnls = []
            for r in oot:
                pnl = r['pnl_pct']
                if pnl != 0:  # actual trade
                    # Remove existing slippage, add new
                    raw = pnl + 2 * SLIPPAGE_FRAC  # undo current slippage (entry + exit)
                    new_slip = 2 * slip_bps / 10000
                    pnl = raw - new_slip
                adj_pnls.append(pnl)
            adj_pnls = np.array(adj_pnls)
            adj_ret = np.sum(adj_pnls) * 100
            adj_sharpe = np.mean(adj_pnls) / (np.std(adj_pnls) + 1e-10) * np.sqrt(252)
            print(f"    {slip_bps:>2} bps: Return={adj_ret:.2f}%  Sharpe={adj_sharpe:.2f}")

        # Capacity check
        avg_trade_pct = m.get('avg_trade_pct', 0)
        print(f"\n  Capacity Analysis (Robinhood $441 account):")
        print(f"    Avg trade return: {avg_trade_pct:.4f}%")
        print(f"    Avg $ per trade: ${441 * avg_trade_pct / 100:.2f}")
        print(f"    Est annual trades: {m.get('n_trades', 0) * (252 / max(m.get('n_days', 1), 1)):.0f}")
        est_annual_dollar = 441 * m.get('cagr_pct', 0) / 100
        print(f"    Est annual $ return: ${est_annual_dollar:.2f}")

    # Save results
    results_summary = {}
    for key, val in all_strategy_results.items():
        results_summary[key] = val['metrics']
        # Convert dates to strings for JSON
        oot_serializable = []
        for r in val['oot']:
            r_copy = dict(r)
            r_copy['date'] = str(r_copy['date'])
            oot_serializable.append(r_copy)

        # Save per-strategy OOT results
        pd.DataFrame(oot_serializable).to_csv(DATA_DIR / f'{key}_oot_results.csv', index=False)

    with open(DATA_DIR / 'strategy_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"\n\nResults saved to {DATA_DIR}")
    print("=" * 70)
    print("RESEARCH COMPLETE")
    print("=" * 70)

    return all_strategy_results, ranked


if __name__ == '__main__':
    results, ranked = run_research()
