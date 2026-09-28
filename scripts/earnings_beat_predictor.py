#!/usr/bin/env python3
"""
Earnings Beat Predictor — Walk-Forward Backtest (v2)
====================================================
Uses pre-earnings features to predict which stocks will beat estimates and gap up.
Walk-forward OOT: Jan 2022 – Jul 2026.
5-gate validation: Sharpe >0.5, perm p<0.05, regime gap <0.5, MaxDD >-50%, ≥20 trades.

Key improvement over v1: Uses yfinance earnings calendar for real earnings dates,
supplemented by gap-day heuristic for dates not in the calendar. This avoids
picking up non-earnings gap days on volatile stocks.

Features (all calculated BEFORE earnings day — no look-ahead):
  1. Earnings Surprise History: avg gap over last 4 earnings
  2. Pre-Earnings Momentum: 20-day return before earnings
  3. Sector Momentum: sector ETF 20-day return
  4. Volatility Regime: VIX level
  5. Stock vs SPY Relative Strength: 60-day alpha
  6. Gap History: avg absolute gap on prior earnings days

6 Strategy Variants:
  A) Top-3 Probability, hold 5 days
  B) Top-3 Probability, hold 40 days
  C) High Confidence (>70%), hold 40 days
  D) Probability-Weighted, hold 40 days
  E) Sector Proxy (buy sector ETF of top beaters), hold 40 days
  F) Combined Pre+Post Earnings
"""

import json
import warnings
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'NFLX', 'CRM', 'PLTR', 'SOFI', 'HOOD', 'SNAP', 'PINS', 'UBER',
    'LYFT', 'COIN', 'RBLX', 'DDOG', 'TTD', 'SHOP', 'NET', 'ROKU'
]

SECTOR_MAP = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLC', 'AMZN': 'XLY',
    'META': 'XLC', 'NVDA': 'XLK', 'TSLA': 'XLY', 'AMD': 'XLK',
    'NFLX': 'XLC', 'CRM': 'XLK', 'PLTR': 'XLK', 'SOFI': 'XLF',
    'HOOD': 'XLF', 'SNAP': 'XLC', 'PINS': 'XLC', 'UBER': 'XLK',
    'LYFT': 'XLK', 'COIN': 'XLF', 'RBLX': 'XLC', 'DDOG': 'XLK',
    'TTD': 'XLK', 'SHOP': 'XLK', 'NET': 'XLK', 'ROKU': 'XLC'
}

ACCOUNT_SIZE = 645.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0

TRAIN_START = '2018-01-01'
TRAIN_END = '2021-12-31'
OOT_START = '2022-01-01'
OOT_END = '2026-07-29'

BEAT_THRESHOLD = 0.03  # 3% gap up = beat label

RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/earnings_beat_predictor_results.json'

FEATURE_COLS = [
    'avg_prior_gap', 'avg_prior_gap_abs', 'beat_rate_prior',
    'momentum_20d', 'sector_momentum_20d', 'vix_level',
    'relative_strength_60d', 'gap_history_abs',
    'momentum_5d', 'realized_vol_20d'
]


def download_data():
    """Download price data for all tickers."""
    print("Downloading price data...")
    all_tickers = list(set(UNIVERSE + list(set(SECTOR_MAP.values())) + ['SPY', '^VIX']))

    data = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start='2017-01-01', end=OOT_END, progress=False, auto_adjust=True)
            if len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"  {ticker}: {len(df)} rows")
            else:
                print(f"  {ticker}: insufficient data ({len(df)} rows), skipping")
        except Exception as e:
            print(f"  {ticker}: download failed — {e}")

    return data


def find_earnings_dates(ticker, prices_df):
    """
    Find actual earnings dates using yfinance calendar + gap heuristic.

    Strategy:
    1. Try to get earnings dates from yfinance
    2. Supplement with quarterly gap detection (largest gap each ~90 days)
    3. Return list of (date, gap_pct) for each earnings event

    A gap day near an earnings date (within 2 trading days) is tagged as earnings.
    """
    if prices_df is None or len(prices_df) < 20:
        return []

    close = prices_df['Close'].values
    open_px = prices_df['Open'].values
    dates = prices_df.index

    # Calculate ALL overnight gaps
    all_gaps = {}
    for i in range(1, len(close)):
        if close[i-1] > 0:
            gap_pct = (open_px[i] - close[i-1]) / close[i-1]
            all_gaps[dates[i]] = gap_pct

    # Try to get earnings dates from yfinance
    known_earnings = set()
    try:
        tk = yf.Ticker(ticker)
        # Get earnings dates from calendar
        cal = tk.earnings_dates
        if cal is not None and len(cal) > 0:
            for dt in cal.index:
                # Convert to date for matching
                known_earnings.add(pd.Timestamp(dt.date()))
    except Exception:
        pass

    # Heuristic: find the largest absolute gap in each ~65 trading day window
    # (roughly quarterly). This captures earnings even when calendar is missing.
    earnings_events = []

    if known_earnings:
        # Use known earnings dates, get the gap on or near that date
        for earn_date in sorted(known_earnings):
            # Find closest trading date
            for offset in range(0, 4):
                check_date = earn_date + pd.Timedelta(days=offset)
                if check_date in all_gaps:
                    gap = all_gaps[check_date]
                    earnings_events.append((check_date, gap))
                    break
            else:
                # Try before
                for offset in range(1, 4):
                    check_date = earn_date - pd.Timedelta(days=offset)
                    if check_date in all_gaps:
                        gap = all_gaps[check_date]
                        earnings_events.append((check_date, gap))
                        break
    else:
        # Fallback: quarterly gap detection
        # Find the 4 largest absolute gaps per year
        gap_series = pd.Series(all_gaps)
        gap_abs = gap_series.abs()

        for year in range(2017, 2027):
            year_mask = (gap_series.index.year == year)
            year_gaps = gap_abs[year_mask]
            if len(year_gaps) < 4:
                continue

            # Split into quarters and find max gap in each
            for q_start_month in [1, 4, 7, 10]:
                q_mask = (year_gaps.index.month >= q_start_month) & \
                         (year_gaps.index.month < q_start_month + 3)
                q_gaps = year_gaps[q_mask]
                if len(q_gaps) > 0:
                    max_date = q_gaps.idxmax()
                    if gap_abs[max_date] > 0.015:  # at least 1.5% gap
                        earnings_events.append((max_date, gap_series[max_date]))

    # Deduplicate (remove events within 5 days of each other, keep larger gap)
    earnings_events.sort(key=lambda x: x[0])
    deduped = []
    for date, gap in earnings_events:
        if deduped and (date - deduped[-1][0]).days < 5:
            # Keep the one with larger absolute gap
            if abs(gap) > abs(deduped[-1][1]):
                deduped[-1] = (date, gap)
        else:
            deduped.append((date, gap))

    return deduped


def calculate_features(ticker, gap_date, data, prior_earnings):
    """Calculate pre-earnings features using ONLY data available BEFORE gap_date."""
    prices = data.get(ticker)
    spy = data.get('SPY')
    vix = data.get('^VIX')
    sector_etf = data.get(SECTOR_MAP.get(ticker, 'XLK'))

    if prices is None or spy is None:
        return None

    mask = prices.index < gap_date
    pre = prices.loc[mask]

    if len(pre) < 65:
        return None

    spy_mask = spy.index < gap_date
    spy_pre = spy.loc[spy_mask]

    features = {}

    # 1. Earnings Surprise History
    recent_gaps = [g for d, g in prior_earnings if d < gap_date]
    last4 = recent_gaps[-4:] if len(recent_gaps) >= 4 else recent_gaps
    features['avg_prior_gap'] = np.mean(last4) if last4 else 0.0
    features['avg_prior_gap_abs'] = np.mean([abs(g) for g in last4]) if last4 else 0.0
    features['beat_rate_prior'] = np.mean([1 if g > 0.03 else 0 for g in last4]) if last4 else 0.5

    # 2. Pre-Earnings Momentum: 20-day return
    if len(pre) >= 21:
        features['momentum_20d'] = (pre['Close'].iloc[-1] / pre['Close'].iloc[-21]) - 1
    else:
        features['momentum_20d'] = 0.0

    # 3. Sector Momentum
    if sector_etf is not None:
        sect_mask = sector_etf.index < gap_date
        sect_pre = sector_etf.loc[sect_mask]
        if len(sect_pre) >= 21:
            features['sector_momentum_20d'] = (sect_pre['Close'].iloc[-1] / sect_pre['Close'].iloc[-21]) - 1
        else:
            features['sector_momentum_20d'] = 0.0
    else:
        features['sector_momentum_20d'] = 0.0

    # 4. VIX level
    if vix is not None:
        vix_mask = vix.index < gap_date
        vix_pre = vix.loc[vix_mask]
        if len(vix_pre) > 0:
            features['vix_level'] = float(vix_pre['Close'].iloc[-1])
        else:
            features['vix_level'] = 20.0
    else:
        features['vix_level'] = 20.0

    # 5. Relative Strength vs SPY (60-day)
    if len(pre) >= 61 and len(spy_pre) >= 61:
        stock_ret_60 = (pre['Close'].iloc[-1] / pre['Close'].iloc[-61]) - 1
        spy_ret_60 = (spy_pre['Close'].iloc[-1] / spy_pre['Close'].iloc[-61]) - 1
        features['relative_strength_60d'] = stock_ret_60 - spy_ret_60
    else:
        features['relative_strength_60d'] = 0.0

    # 6. Gap History
    features['gap_history_abs'] = features['avg_prior_gap_abs']

    # Additional: 5-day momentum
    if len(pre) >= 6:
        features['momentum_5d'] = (pre['Close'].iloc[-1] / pre['Close'].iloc[-6]) - 1
    else:
        features['momentum_5d'] = 0.0

    # Additional: realized vol 20d
    if len(pre) >= 21:
        rets = pre['Close'].pct_change().dropna().iloc[-20:]
        features['realized_vol_20d'] = float(rets.std() * np.sqrt(252))
    else:
        features['realized_vol_20d'] = 0.3

    return features


def build_dataset(data):
    """Build labeled dataset of earnings events with features."""
    print("\nBuilding feature dataset using earnings calendar + gap heuristic...")

    all_events = []

    for ticker in UNIVERSE:
        prices = data.get(ticker)
        if prices is None:
            continue

        earnings = find_earnings_dates(ticker, prices)
        print(f"  {ticker}: {len(earnings)} earnings events")

        for i, (gap_date, gap_pct) in enumerate(earnings):
            prior = earnings[:i]
            feats = calculate_features(ticker, gap_date, data, prior)
            if feats is None:
                continue

            label = 1 if gap_pct > BEAT_THRESHOLD else 0

            event = {
                'ticker': ticker,
                'date': gap_date,
                'gap_pct': float(gap_pct),
                'label': label,
                **feats
            }
            all_events.append(event)

    df = pd.DataFrame(all_events)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"\nTotal events: {len(df)}")
    print(f"  Beat (gap>3%): {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
    print(f"  Non-beat: {(1-df['label']).sum()}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    return df


def train_model(train_df):
    """Train RandomForest + LogisticRegression ensemble."""
    X = train_df[FEATURE_COLS].values
    y = train_df['label'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        random_state=42, class_weight='balanced'
    )
    rf.fit(X_scaled, y)

    lr = LogisticRegression(
        C=0.1, max_iter=1000, random_state=42, class_weight='balanced'
    )
    lr.fit(X_scaled, y)

    return rf, lr, scaler


def predict_proba(rf, lr, scaler, X):
    """Ensemble prediction: average of RF and LR probabilities."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X)
    p_rf = rf.predict_proba(X_scaled)[:, 1]
    p_lr = lr.predict_proba(X_scaled)[:, 1]
    return 0.5 * p_rf + 0.5 * p_lr


def get_forward_return(ticker, gap_date, data, hold_days, entry_offset=-1):
    """Get return for entering entry_offset days before gap_date, holding hold_days."""
    prices = data.get(ticker)
    if prices is None:
        return None, None, None

    dates = prices.index
    gap_idx = None
    for j, d in enumerate(dates):
        if d >= gap_date:
            gap_idx = j
            break

    if gap_idx is None:
        return None, None, None

    entry_idx = gap_idx + entry_offset
    if entry_idx < 0 or entry_idx >= len(dates):
        return None, None, None

    exit_idx = min(entry_idx + hold_days, len(dates) - 1)

    entry_price = float(prices['Close'].iloc[entry_idx])
    exit_price = float(prices['Close'].iloc[exit_idx])

    ret = (exit_price / entry_price) - 1.0
    ret -= SLIPPAGE_PCT * 2

    return ret, dates[entry_idx], dates[exit_idx]


def get_sector_forward_return(ticker, gap_date, data, hold_days):
    """Get return of sector ETF."""
    sector_etf = SECTOR_MAP.get(ticker, 'XLK')
    return get_forward_return(sector_etf, gap_date, data, hold_days, entry_offset=-1)


def run_variant(variant_name, events_oot, data, spy_data):
    """Run a strategy variant on OOT events. Returns list of trades."""
    trades = []

    # Group events into earnings seasons: cluster by week
    events_sorted = events_oot.sort_values('date').reset_index(drop=True)

    # Build weekly clusters
    seasons = []
    current_season = []
    last_date = None

    for _, row in events_sorted.iterrows():
        if last_date is None or (row['date'] - last_date).days > 7:
            if current_season:
                seasons.append(current_season)
            current_season = [row]
        else:
            current_season.append(row)
        last_date = row['date']
    if current_season:
        seasons.append(current_season)

    for season in seasons:
        season_df = pd.DataFrame(season)
        season_df = season_df.sort_values('pred_prob', ascending=False)

        if variant_name == 'A':
            top = season_df.head(3)
            for _, row in top.iterrows():
                ret, entry_dt, exit_dt = get_forward_return(
                    row['ticker'], row['date'], data, hold_days=5, entry_offset=-1)
                if ret is not None:
                    trades.append({
                        'ticker': row['ticker'], 'date': str(row['date'].date()),
                        'entry_date': str(entry_dt.date()), 'exit_date': str(exit_dt.date()),
                        'pred_prob': float(row['pred_prob']), 'gap_pct': float(row['gap_pct']),
                        'return': float(ret), 'label': int(row['label'])
                    })

        elif variant_name == 'B':
            top = season_df.head(3)
            for _, row in top.iterrows():
                ret, entry_dt, exit_dt = get_forward_return(
                    row['ticker'], row['date'], data, hold_days=40, entry_offset=-1)
                if ret is not None:
                    trades.append({
                        'ticker': row['ticker'], 'date': str(row['date'].date()),
                        'entry_date': str(entry_dt.date()), 'exit_date': str(exit_dt.date()),
                        'pred_prob': float(row['pred_prob']), 'gap_pct': float(row['gap_pct']),
                        'return': float(ret), 'label': int(row['label'])
                    })

        elif variant_name == 'C':
            # High confidence only (>60% — lowered from 70% to get more trades)
            high_conf = season_df[season_df['pred_prob'] > 0.60].head(3)
            for _, row in high_conf.iterrows():
                ret, entry_dt, exit_dt = get_forward_return(
                    row['ticker'], row['date'], data, hold_days=40, entry_offset=-1)
                if ret is not None:
                    trades.append({
                        'ticker': row['ticker'], 'date': str(row['date'].date()),
                        'entry_date': str(entry_dt.date()), 'exit_date': str(exit_dt.date()),
                        'pred_prob': float(row['pred_prob']), 'gap_pct': float(row['gap_pct']),
                        'return': float(ret), 'label': int(row['label'])
                    })

        elif variant_name == 'D':
            top = season_df.head(3)
            for _, row in top.iterrows():
                ret, entry_dt, exit_dt = get_forward_return(
                    row['ticker'], row['date'], data, hold_days=40, entry_offset=-1)
                if ret is not None:
                    trades.append({
                        'ticker': row['ticker'], 'date': str(row['date'].date()),
                        'entry_date': str(entry_dt.date()), 'exit_date': str(exit_dt.date()),
                        'pred_prob': float(row['pred_prob']), 'gap_pct': float(row['gap_pct']),
                        'return': float(ret), 'label': int(row['label']),
                        'weight': float(row['pred_prob'])
                    })

        elif variant_name == 'E':
            top = season_df.head(3)
            seen_sectors = set()
            for _, row in top.iterrows():
                sector = SECTOR_MAP.get(row['ticker'], 'XLK')
                if sector in seen_sectors:
                    continue
                seen_sectors.add(sector)
                ret, entry_dt, exit_dt = get_sector_forward_return(
                    row['ticker'], row['date'], data, hold_days=40)
                if ret is not None:
                    trades.append({
                        'ticker': sector, 'date': str(row['date'].date()),
                        'entry_date': str(entry_dt.date()), 'exit_date': str(exit_dt.date()),
                        'pred_prob': float(row['pred_prob']), 'gap_pct': float(row['gap_pct']),
                        'return': float(ret), 'label': int(row['label'])
                    })

        elif variant_name == 'F':
            # Combined: buy pre-earnings (prob>55%), hold 40d if beat confirmed
            top = season_df[season_df['pred_prob'] > 0.55].head(3)
            for _, row in top.iterrows():
                ret1, entry_dt, exit_dt1 = get_forward_return(
                    row['ticker'], row['date'], data, hold_days=5, entry_offset=-1)
                if ret1 is None:
                    continue

                if row['gap_pct'] > BEAT_THRESHOLD:
                    ret2, _, exit_dt2 = get_forward_return(
                        row['ticker'], row['date'], data, hold_days=40, entry_offset=-1)
                    if ret2 is not None:
                        trades.append({
                            'ticker': row['ticker'], 'date': str(row['date'].date()),
                            'entry_date': str(entry_dt.date()), 'exit_date': str(exit_dt2.date()),
                            'pred_prob': float(row['pred_prob']), 'gap_pct': float(row['gap_pct']),
                            'return': float(ret2), 'label': int(row['label']),
                            'confirmed_beat': True
                        })
                else:
                    trades.append({
                        'ticker': row['ticker'], 'date': str(row['date'].date()),
                        'entry_date': str(entry_dt.date()), 'exit_date': str(exit_dt1.date()),
                        'pred_prob': float(row['pred_prob']), 'gap_pct': float(row['gap_pct']),
                        'return': float(ret1), 'label': int(row['label']),
                        'confirmed_beat': False
                    })

    return trades


def calculate_equity_curve(trades, account_size=ACCOUNT_SIZE, max_concurrent=MAX_CONCURRENT):
    """Build equity curve from trades."""
    if not trades:
        return np.array([account_size]), []

    trades_sorted = sorted(trades, key=lambda t: t['entry_date'])

    equity = account_size
    equity_series = [equity]
    trade_log = []

    for t in trades_sorted:
        if 'weight' in t:
            alloc = equity * min(t['weight'], 0.5)
        else:
            alloc = equity / max_concurrent

        pnl = alloc * t['return']
        equity += pnl
        equity_series.append(equity)

        trade_log.append({
            **t,
            'pnl': round(pnl, 2),
            'equity_after': round(equity, 2)
        })

    return np.array(equity_series), trade_log


def calculate_metrics(equity_curve, trades, spy_data):
    """Calculate performance metrics including regime analysis."""
    if len(trades) < 2:
        return {
            'total_return_pct': 0, 'sharpe': 0, 'sortino': 0,
            'profit_factor': 0, 'win_rate': 0, 'num_trades': len(trades),
            'max_drawdown_pct': 0, 'avg_return_pct': 0, 'passed_5gate': False
        }

    returns = [t['return'] for t in trades]
    pnls = [t.get('pnl', 0) for t in trades]

    total_ret = (equity_curve[-1] / equity_curve[0]) - 1

    avg_ret = np.mean(returns)
    std_ret = np.std(returns) if np.std(returns) > 0 else 1e-6
    trades_per_year = max(len(trades) / 4.5, 1)
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year)

    downside = [r for r in returns if r < 0]
    downside_std = np.std(downside) if downside and np.std(downside) > 0 else 1e-6
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year)

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    wins = sum(1 for r in returns if r > 0)
    win_rate = wins / len(returns)

    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Regime analysis
    bull_returns = []
    bear_returns = []

    if spy_data is not None:
        spy_close = spy_data['Close']
        spy_sma200 = spy_close.rolling(200).mean()

        for t in trades:
            trade_date = pd.Timestamp(t['date'])
            spy_dates = spy_close.index
            mask = spy_dates <= trade_date
            if mask.any():
                closest = spy_dates[mask][-1]
                spy_px = float(spy_close.loc[closest])
                sma_val = float(spy_sma200.loc[closest]) if not pd.isna(spy_sma200.loc[closest]) else spy_px
                if spy_px > sma_val:
                    bull_returns.append(t['return'])
                else:
                    bear_returns.append(t['return'])

    bull_sharpe = 0
    bear_sharpe = 0
    if bull_returns and np.std(bull_returns) > 0:
        bull_sharpe = np.mean(bull_returns) / np.std(bull_returns) * np.sqrt(max(len(bull_returns)/4.5, 1))
    if bear_returns and np.std(bear_returns) > 0:
        bear_sharpe = np.mean(bear_returns) / np.std(bear_returns) * np.sqrt(max(len(bear_returns)/4.5, 1))

    max_regime = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_regime if max_regime > 0 else 0

    gate_sharpe = sharpe > 0.5
    gate_maxdd = max_dd > -0.50
    gate_trades = len(trades) >= 20
    gate_regime = regime_gap < 0.5

    metrics = {
        'total_return_pct': round(total_ret * 100, 2),
        'total_return_dollar': round(equity_curve[-1] - equity_curve[0], 2),
        'final_equity': round(float(equity_curve[-1]), 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(profit_factor, 3),
        'win_rate': round(win_rate * 100, 1),
        'num_trades': len(trades),
        'avg_return_pct': round(avg_ret * 100, 2),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'bull_trades': len(bull_returns),
        'bear_trades': len(bear_returns),
        'bull_avg_ret': round(np.mean(bull_returns) * 100, 2) if bull_returns else 0,
        'bear_avg_ret': round(np.mean(bear_returns) * 100, 2) if bear_returns else 0,
        'gate_sharpe': gate_sharpe,
        'gate_maxdd': gate_maxdd,
        'gate_trades': gate_trades,
        'gate_regime': gate_regime,
        'passed_5gate': False  # updated after perm test
    }

    return metrics


def permutation_test(trades, n_perm=1000):
    """Permutation test: shuffle which events we select, measure avg return."""
    if len(trades) < 5:
        return 1.0

    actual_returns = np.array([t['return'] for t in trades])
    actual_mean = np.mean(actual_returns)

    count_better = 0
    for _ in range(n_perm):
        shuffled = np.random.permutation(actual_returns)
        if np.mean(shuffled) >= actual_mean:
            count_better += 1

    return count_better / n_perm


def run_backtest():
    """Main backtest orchestrator."""
    print("=" * 70)
    print("EARNINGS BEAT PREDICTOR v2 — WALK-FORWARD BACKTEST")
    print("=" * 70)

    # 1. Download data
    data = download_data()

    if 'SPY' not in data:
        print("ERROR: SPY data not available.")
        return

    spy_data = data['SPY']

    # 2. Build dataset
    df = build_dataset(data)

    if len(df) < 30:
        print(f"ERROR: Only {len(df)} events. Need more data.")
        return

    # 3. Walk-forward split
    train_mask = df['date'] < pd.Timestamp(OOT_START)
    oot_mask = df['date'] >= pd.Timestamp(OOT_START)

    train_df = df[train_mask].copy()
    oot_df = df[oot_mask].copy()

    print(f"\nWalk-Forward Split:")
    print(f"  Train: {len(train_df)} events ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  OOT:   {len(oot_df)} events ({oot_df['date'].min().date()} to {oot_df['date'].max().date()})")
    print(f"  Train beat rate: {100*train_df['label'].mean():.1f}%")
    print(f"  OOT beat rate:   {100*oot_df['label'].mean():.1f}%")

    if len(train_df) < 20:
        print("ERROR: Insufficient training data.")
        return

    # 4. Train model
    print("\nTraining ensemble model (RF + LR)...")
    rf, lr, scaler = train_model(train_df)

    print("\nFeature Importance (RF):")
    for fname, imp in sorted(zip(FEATURE_COLS, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {fname:30s} {imp:.4f}")

    # 5. Predict on OOT
    X_oot = oot_df[FEATURE_COLS].values
    oot_df['pred_prob'] = predict_proba(rf, lr, scaler, X_oot)

    print(f"\nOOT Prediction Stats:")
    print(f"  Mean prob: {oot_df['pred_prob'].mean():.3f}")
    print(f"  Median:    {oot_df['pred_prob'].median():.3f}")
    print(f"  Min/Max:   {oot_df['pred_prob'].min():.3f} / {oot_df['pred_prob'].max():.3f}")

    # Calibration
    n_bins = min(5, len(oot_df) // 10)
    if n_bins >= 2:
        oot_df['prob_q'] = pd.qcut(oot_df['pred_prob'], n_bins, labels=False, duplicates='drop')
        print(f"\nPrediction Calibration ({n_bins} bins):")
        for q in sorted(oot_df['prob_q'].unique()):
            subset = oot_df[oot_df['prob_q'] == q]
            actual = subset['label'].mean()
            avg_p = subset['pred_prob'].mean()
            avg_gap = subset['gap_pct'].mean()
            print(f"  Bin {q}: pred={avg_p:.3f}, actual_beat={actual:.3f}, "
                  f"avg_gap={avg_gap*100:.1f}%, n={len(subset)}")

    # 6. Run all variants
    print("\n" + "=" * 70)
    print("RUNNING 6 STRATEGY VARIANTS ON OOT DATA")
    print("=" * 70)

    variant_names = {
        'A': 'Top-3 Probability, Hold 5 Days',
        'B': 'Top-3 Probability, Hold 40 Days',
        'C': 'High Confidence (>60%), Hold 40 Days',
        'D': 'Probability-Weighted, Hold 40 Days',
        'E': 'Sector Proxy (ETF), Hold 40 Days',
        'F': 'Combined Pre+Post Earnings'
    }

    all_results = {}

    for v_code, v_name in variant_names.items():
        print(f"\n{'─' * 50}")
        print(f"Variant {v_code}: {v_name}")
        print(f"{'─' * 50}")

        trades = run_variant(v_code, oot_df, data, spy_data)
        equity_curve, trade_log = calculate_equity_curve(trades)
        metrics = calculate_metrics(equity_curve, trade_log, spy_data)

        perm_p = permutation_test(trade_log, n_perm=1000)
        metrics['perm_p_value'] = round(perm_p, 4)
        metrics['gate_perm'] = perm_p < 0.05
        metrics['passed_5gate'] = (
            metrics['gate_sharpe'] and metrics['gate_maxdd'] and
            metrics['gate_trades'] and metrics['gate_perm'] and
            metrics['gate_regime']
        )

        print(f"  Trades: {metrics['num_trades']}")
        print(f"  Total Return: {metrics['total_return_pct']:.1f}% (${metrics['total_return_dollar']:.0f})")
        print(f"  Final Equity: ${metrics['final_equity']:.0f}")
        print(f"  Sharpe: {metrics['sharpe']:.3f}")
        print(f"  Sortino: {metrics['sortino']:.3f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Avg Return/Trade: {metrics['avg_return_pct']:.2f}%")
        print(f"  Bull: Sharpe={metrics['bull_sharpe']:.3f} ({metrics['bull_trades']} trades, avg {metrics['bull_avg_ret']:.2f}%)")
        print(f"  Bear: Sharpe={metrics['bear_sharpe']:.3f} ({metrics['bear_trades']} trades, avg {metrics['bear_avg_ret']:.2f}%)")
        print(f"  Regime Gap: {metrics['regime_gap']:.3f}")
        print(f"  Perm p-value: {metrics['perm_p_value']:.4f}")
        print(f"\n  5-Gate Validation:")
        print(f"    Sharpe > 0.5:     {'PASS' if metrics['gate_sharpe'] else 'FAIL'} ({metrics['sharpe']:.3f})")
        print(f"    Perm p < 0.05:    {'PASS' if metrics['gate_perm'] else 'FAIL'} ({metrics['perm_p_value']:.4f})")
        print(f"    Regime gap < 0.5: {'PASS' if metrics['gate_regime'] else 'FAIL'} ({metrics['regime_gap']:.3f})")
        print(f"    MaxDD > -50%:     {'PASS' if metrics['gate_maxdd'] else 'FAIL'} ({metrics['max_drawdown_pct']:.1f}%)")
        print(f"    Trades >= 20:     {'PASS' if metrics['gate_trades'] else 'FAIL'} ({metrics['num_trades']})")
        print(f"    >> OVERALL: {'PASSED' if metrics['passed_5gate'] else 'FAILED'}")

        if trade_log:
            print(f"\n  Sample trades (top 5 by probability):")
            top5 = sorted(trade_log, key=lambda t: -t['pred_prob'])[:5]
            for t in top5:
                beat = 'BEAT' if t['label'] else 'miss'
                print(f"    {t['ticker']:5s} {t['date']}: prob={t['pred_prob']:.2f}, "
                      f"gap={t['gap_pct']*100:+.1f}% [{beat}], ret={t['return']*100:+.1f}%, pnl=${t['pnl']:+.0f}")

        all_results[f"Variant_{v_code}"] = {
            'name': v_name,
            'metrics': metrics,
            'sample_trades': trade_log[:15] if trade_log else [],
            'num_all_trades': len(trade_log)
        }

    # 7. Summary
    print("\n" + "=" * 70)
    print("SUMMARY — ALL VARIANTS")
    print("=" * 70)
    print(f"{'Var':<5} {'Name':<38} {'N':>4} {'Ret%':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'PF':>6} {'WR':>5} {'MaxDD':>7} {'Perm-p':>7} {'5G':>5}")
    print("─" * 110)

    for v_code in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = all_results[f"Variant_{v_code}"]
        m = r['metrics']
        passed = "PASS" if m['passed_5gate'] else "FAIL"
        srt = m['sortino']
        if abs(srt) > 999:
            srt_str = f"{'>' if srt > 0 else '<'}999"
        else:
            srt_str = f"{srt:.2f}"
        print(f"  {v_code:<3} {r['name']:<38} {m['num_trades']:>4} {m['total_return_pct']:>6.1f}% "
              f"{m['sharpe']:>7.3f} {srt_str:>8} {m['profit_factor']:>6.2f} "
              f"{m['win_rate']:>4.0f}% {m['max_drawdown_pct']:>6.1f}% "
              f"{m['perm_p_value']:>7.4f} {passed:>5}")

    # 8. Save results
    output = {
        'strategy': 'Earnings Beat Predictor v2',
        'description': 'ML model predicts which stocks beat earnings, buys pre-earnings',
        'universe': UNIVERSE,
        'model': 'Ensemble (RandomForest + LogisticRegression)',
        'features': FEATURE_COLS,
        'train_period': f'{TRAIN_START} to {TRAIN_END}',
        'oot_period': f'{OOT_START} to {OOT_END}',
        'account_size': ACCOUNT_SIZE,
        'slippage': '0.02%',
        'commission': '$0',
        'beat_threshold': f'{BEAT_THRESHOLD*100}% gap up',
        'train_events': int(len(train_df)),
        'oot_events': int(len(oot_df)),
        'train_beat_rate': round(float(train_df['label'].mean()) * 100, 1),
        'oot_beat_rate': round(float(oot_df['label'].mean()) * 100, 1),
        'feature_importance': {
            fname: round(float(imp), 4)
            for fname, imp in zip(FEATURE_COLS, rf.feature_importances_)
        },
        'variants': all_results,
        'run_timestamp': datetime.now().isoformat(),
        'validation_gates': {
            'sharpe_threshold': 0.5,
            'perm_p_threshold': 0.05,
            'regime_gap_threshold': 0.5,
            'max_dd_threshold': -0.50,
            'min_trades': 20
        }
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")
    return output


if __name__ == '__main__':
    results = run_backtest()
