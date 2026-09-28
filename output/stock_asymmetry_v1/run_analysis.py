#!/usr/bin/env python3
"""
Stock-Level & Market-Wide Asymmetric Signal Analysis
=====================================================
Analyzes 50 large-cap stocks for asymmetric return patterns.
All signals are T-1 (no lookahead). Period: 2015-2026.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import json

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/stock_asymmetry_v1'

# ── UNIVERSE ──────────────────────────────────────────────────────────────────
STOCKS = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','JPM','GS','BAC',
    'V','MA','UNH','JNJ','PG','KO','PEP','MRK','ABBV','LLY',
    'HD','COST','WMT','CRM','AMD','NFLX','ADBE','INTC','CSCO','QCOM',
    'XOM','CVX','PFE','TMO','ABT','AVGO','TXN','MCD','NKE','DIS',
    'CMCSA','T','VZ','NEE','SO','SHW','LMT','RTX','CAT','DE'
]

# Sector mapping
SECTOR_MAP = {
    'AAPL':'XLK','MSFT':'XLK','GOOGL':'XLC','AMZN':'XLY','META':'XLC',
    'NVDA':'XLK','TSLA':'XLY','JPM':'XLF','GS':'XLF','BAC':'XLF',
    'V':'XLK','MA':'XLK','UNH':'XLV','JNJ':'XLV','PG':'XLP',
    'KO':'XLP','PEP':'XLP','MRK':'XLV','ABBV':'XLV','LLY':'XLV',
    'HD':'XLY','COST':'XLP','WMT':'XLP','CRM':'XLK','AMD':'XLK',
    'NFLX':'XLC','ADBE':'XLK','INTC':'XLK','CSCO':'XLK','QCOM':'XLK',
    'XOM':'XLE','CVX':'XLE','PFE':'XLV','TMO':'XLV','ABT':'XLV',
    'AVGO':'XLK','TXN':'XLK','MCD':'XLY','NKE':'XLY','DIS':'XLC',
    'CMCSA':'XLC','T':'XLC','VZ':'XLC','NEE':'XLU','SO':'XLU',
    'SHW':'XLB','LMT':'XLI','RTX':'XLI','CAT':'XLI','DE':'XLI'
}

SECTOR_ETFS = list(set(SECTOR_MAP.values()))
MACRO_TICKERS = ['SPY','TLT','GLD','VIX']  # VIX = ^VIX

START = '2014-06-01'  # extra history for lookback windows
END = '2026-07-20'

# ── DATA DOWNLOAD ─────────────────────────────────────────────────────────────
def download_data():
    """Download all price data."""
    all_tickers = STOCKS + SECTOR_ETFS + ['SPY','TLT','GLD','^VIX']
    all_tickers = list(set(all_tickers))

    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=START, end=END, auto_adjust=True, threads=True)

    # Extract close and volume
    close = data['Close'].copy()
    volume = data['Volume'].copy()

    # Rename ^VIX to VIX
    if '^VIX' in close.columns:
        close.rename(columns={'^VIX': 'VIX'}, inplace=True)
    if '^VIX' in volume.columns:
        volume.rename(columns={'^VIX': 'VIX'}, inplace=True)

    print(f"Data shape: {close.shape}, date range: {close.index[0]} to {close.index[-1]}")
    return close, volume


# ── SIGNAL COMPUTATION ────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    """RSI calculation."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_stock_signals(close, volume):
    """Compute all T-1 signals for each stock."""
    signals = {}

    for ticker in STOCKS:
        if ticker not in close.columns:
            print(f"  Skipping {ticker} - no data")
            continue

        px = close[ticker].dropna()
        vol = volume[ticker].dropna() if ticker in volume.columns else pd.Series(dtype=float)

        if len(px) < 252:
            print(f"  Skipping {ticker} - insufficient data ({len(px)} rows)")
            continue

        df = pd.DataFrame(index=px.index)
        df['close'] = px

        # a. RSI(14)
        df['rsi'] = compute_rsi(px, 14)

        # b. Distance from 52-week high (%)
        df['high_252'] = px.rolling(252, min_periods=200).max()
        df['dist_from_high'] = (px / df['high_252'] - 1) * 100  # negative = below high

        # c. 20d realized vol (annualized)
        df['ret'] = px.pct_change()
        df['vol_20d'] = df['ret'].rolling(20).std() * np.sqrt(252) * 100

        # d. Volume surge: 5d avg / 63d avg
        if len(vol) > 0:
            vol_aligned = vol.reindex(df.index)
            df['vol_5d'] = vol_aligned.rolling(5).mean()
            df['vol_63d'] = vol_aligned.rolling(63).mean()
            df['volume_surge'] = df['vol_5d'] / df['vol_63d'].replace(0, np.nan)
        else:
            df['volume_surge'] = np.nan

        # e. 3-month momentum (63d return)
        df['mom_3m'] = px.pct_change(63) * 100

        # f. Earnings proximity (approximate: flag around Jan/Apr/Jul/Oct quarterly dates)
        df['month'] = df.index.month
        df['day'] = df.index.day
        earnings_months = [1, 4, 7, 10]
        df['near_earnings'] = 0
        for em in earnings_months:
            # Flag last 5 trading days of prior month + first 15 of earnings month
            mask = ((df['month'] == em) & (df['day'] <= 15)) | \
                   ((df['month'] == (em - 1 if em > 1 else 12)) & (df['day'] >= 25))
            df.loc[mask, 'near_earnings'] = 1

        # g. Sector relative strength (21d)
        sector_etf = SECTOR_MAP.get(ticker)
        if sector_etf and sector_etf in close.columns:
            sector_px = close[sector_etf].reindex(df.index)
            df['stock_ret_21d'] = px.pct_change(21) * 100
            df['sector_ret_21d'] = sector_px.pct_change(21) * 100
            df['sector_rel_strength'] = df['stock_ret_21d'] - df['sector_ret_21d']
        else:
            df['sector_rel_strength'] = np.nan

        # h. Mean reversion score: z-score of price vs 63d mean
        df['ma_63'] = px.rolling(63).mean()
        df['std_63'] = px.rolling(63).std()
        df['mean_rev_zscore'] = (px - df['ma_63']) / df['std_63'].replace(0, np.nan)

        # Forward returns (these are FUTURE, used as targets)
        df['fwd_1w'] = px.pct_change(5).shift(-5) * 100
        df['fwd_1m'] = px.pct_change(21).shift(-21) * 100
        df['fwd_3m'] = px.pct_change(63).shift(-63) * 100

        # All signals are T-1: shift signals forward by 1 day
        signal_cols = ['rsi', 'dist_from_high', 'vol_20d', 'volume_surge',
                       'mom_3m', 'near_earnings', 'sector_rel_strength', 'mean_rev_zscore']
        for col in signal_cols:
            df[col] = df[col].shift(1)

        df['ticker'] = ticker

        # Filter to analysis period (2015+)
        df = df.loc['2015-01-01':]

        signals[ticker] = df

    return signals


# ── QUINTILE ANALYSIS ─────────────────────────────────────────────────────────
def quintile_analysis(all_data, signal_col, fwd_col='fwd_1m', n_quintiles=5):
    """Compute forward returns by signal quintile."""
    data = all_data[[signal_col, fwd_col]].dropna()
    if len(data) < 100:
        return None

    try:
        data['quintile'] = pd.qcut(data[signal_col], n_quintiles, labels=False, duplicates='drop')
    except ValueError:
        return None

    results = []
    for q in sorted(data['quintile'].unique()):
        subset = data[data['quintile'] == q][fwd_col]
        results.append({
            'quintile': int(q) + 1,
            'signal': signal_col,
            'horizon': fwd_col,
            'n_obs': len(subset),
            'mean_ret': subset.mean(),
            'median_ret': subset.median(),
            'p10': subset.quantile(0.10),
            'p90': subset.quantile(0.90),
            'std': subset.std(),
            'hit_rate': (subset > 0).mean() * 100,
            'sharpe': subset.mean() / subset.std() * np.sqrt(12) if subset.std() > 0 else 0,
            'skew': subset.skew(),
            'signal_range_lo': data[data['quintile'] == q][signal_col].min(),
            'signal_range_hi': data[data['quintile'] == q][signal_col].max(),
        })

    return pd.DataFrame(results)


def bootstrap_ci(data, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True))
                      for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return np.percentile(means, alpha * 100), np.percentile(means, (1 - alpha) * 100)


# ── CONDITIONAL ANALYSIS ──────────────────────────────────────────────────────
def conditional_analysis(all_data):
    """
    Key question: After a stock drops 20%+ from 52wk high with RSI<35 and volume surge,
    what happens next?
    """
    results = []

    # Define conditional setups
    setups = [
        {
            'name': 'Deep Pullback + Oversold + Volume Surge',
            'filter': lambda d: (d['dist_from_high'] <= -20) & (d['rsi'] < 35) & (d['volume_surge'] > 1.5),
        },
        {
            'name': 'Deep Pullback + Oversold (no vol filter)',
            'filter': lambda d: (d['dist_from_high'] <= -20) & (d['rsi'] < 35),
        },
        {
            'name': 'Moderate Pullback + Oversold',
            'filter': lambda d: (d['dist_from_high'].between(-15, -10)) & (d['rsi'] < 40),
        },
        {
            'name': 'Extreme Oversold (RSI < 25)',
            'filter': lambda d: (d['rsi'] < 25),
        },
        {
            'name': 'Overbought + Extended (RSI>75, >5% above 63d MA)',
            'filter': lambda d: (d['rsi'] > 75) & (d['mean_rev_zscore'] > 2),
        },
        {
            'name': 'High Vol + Negative Momentum + Volume Surge',
            'filter': lambda d: (d['vol_20d'] > d['vol_20d'].quantile(0.8)) & (d['mom_3m'] < -10) & (d['volume_surge'] > 1.5),
        },
        {
            'name': 'Low Vol Compression + No Trend',
            'filter': lambda d: (d['vol_20d'] < d['vol_20d'].quantile(0.2)) & (d['mom_3m'].abs() < 5),
        },
        {
            'name': 'Strong Momentum + Low Vol (Trend Continuation)',
            'filter': lambda d: (d['mom_3m'] > 15) & (d['vol_20d'] < d['vol_20d'].quantile(0.4)),
        },
        {
            'name': 'Sector Laggard + Oversold',
            'filter': lambda d: (d['sector_rel_strength'] < -10) & (d['rsi'] < 40),
        },
        {
            'name': 'Extreme Mean Reversion (Z < -2.5)',
            'filter': lambda d: (d['mean_rev_zscore'] < -2.5),
        },
        {
            'name': 'Near Earnings + High Vol + Pullback',
            'filter': lambda d: (d['near_earnings'] == 1) & (d['vol_20d'] > d['vol_20d'].quantile(0.7)) & (d['dist_from_high'] < -10),
        },
        {
            'name': 'Baseline (all observations)',
            'filter': lambda d: pd.Series(True, index=d.index),
        },
    ]

    for setup in setups:
        try:
            mask = setup['filter'](all_data)
            subset = all_data[mask]
        except Exception as e:
            print(f"  Error in setup '{setup['name']}': {e}")
            continue

        for horizon in ['fwd_1w', 'fwd_1m', 'fwd_3m']:
            rets = subset[horizon].dropna()
            if len(rets) < 20:
                continue

            ci_lo, ci_hi = bootstrap_ci(rets.values, n_boot=2000)

            results.append({
                'setup': setup['name'],
                'horizon': horizon,
                'n_obs': len(rets),
                'n_stocks': subset['ticker'].nunique() if 'ticker' in subset.columns else 0,
                'mean_ret': rets.mean(),
                'median_ret': rets.median(),
                'p10': rets.quantile(0.10),
                'p25': rets.quantile(0.25),
                'p75': rets.quantile(0.75),
                'p90': rets.quantile(0.90),
                'hit_rate': (rets > 0).mean() * 100,
                'sharpe_ann': rets.mean() / rets.std() * np.sqrt(252/5 if '1w' in horizon else 12 if '1m' in horizon else 4),
                'skew': rets.skew(),
                'ci_95_lo': ci_lo,
                'ci_95_hi': ci_hi,
                'upside_downside_ratio': abs(rets.quantile(0.90)) / abs(rets.quantile(0.10)) if abs(rets.quantile(0.10)) > 0.01 else np.nan,
            })

    return pd.DataFrame(results)


# ── CROSS-STOCK BREADTH SIGNALS ──────────────────────────────────────────────
def cross_stock_signals(signals_dict, close):
    """When >30% of stocks show signal X, what happens to the market?"""

    # Build daily panel of signals
    dates = close.loc['2015-01-01':].index
    spy = close['SPY'].reindex(dates)
    spy_fwd_1w = spy.pct_change(5).shift(-5) * 100
    spy_fwd_1m = spy.pct_change(21).shift(-21) * 100
    spy_fwd_3m = spy.pct_change(63).shift(-63) * 100

    # Count stocks in each signal state per day
    panel = pd.DataFrame(index=dates)
    panel['spy_fwd_1w'] = spy_fwd_1w
    panel['spy_fwd_1m'] = spy_fwd_1m
    panel['spy_fwd_3m'] = spy_fwd_3m

    n_stocks = len(signals_dict)

    # Initialize counts
    for sig in ['pct_oversold', 'pct_overbought', 'pct_deep_pullback',
                'pct_vol_surge', 'pct_neg_momentum', 'pct_above_50sma',
                'pct_extreme_zscore_neg', 'pct_extreme_zscore_pos']:
        panel[sig] = 0.0

    for ticker, df in signals_dict.items():
        df_aligned = df.reindex(dates)

        panel['pct_oversold'] += (df_aligned['rsi'] < 30).astype(float).fillna(0) / n_stocks * 100
        panel['pct_overbought'] += (df_aligned['rsi'] > 70).astype(float).fillna(0) / n_stocks * 100
        panel['pct_deep_pullback'] += (df_aligned['dist_from_high'] < -20).astype(float).fillna(0) / n_stocks * 100
        panel['pct_vol_surge'] += (df_aligned['volume_surge'] > 1.5).astype(float).fillna(0) / n_stocks * 100
        panel['pct_neg_momentum'] += (df_aligned['mom_3m'] < 0).astype(float).fillna(0) / n_stocks * 100

        # % above 50d SMA
        ma50 = close[ticker].rolling(50).mean().reindex(dates) if ticker in close.columns else pd.Series(np.nan, index=dates)
        px = close[ticker].reindex(dates) if ticker in close.columns else pd.Series(np.nan, index=dates)
        panel['pct_above_50sma'] += (px > ma50).astype(float).fillna(0) / n_stocks * 100

        panel['pct_extreme_zscore_neg'] += (df_aligned['mean_rev_zscore'] < -2).astype(float).fillna(0) / n_stocks * 100
        panel['pct_extreme_zscore_pos'] += (df_aligned['mean_rev_zscore'] > 2).astype(float).fillna(0) / n_stocks * 100

    # Shift breadth signals by 1 day for T-1
    breadth_cols = [c for c in panel.columns if c.startswith('pct_')]
    panel[breadth_cols] = panel[breadth_cols].shift(1)

    return panel


def analyze_breadth_signals(panel):
    """Analyze market returns conditional on breadth signals."""
    results = []

    setups = [
        ('High Oversold Breadth (>30% RSI<30)', lambda p: p['pct_oversold'] > 30),
        ('High Oversold Breadth (>20% RSI<30)', lambda p: p['pct_oversold'] > 20),
        ('High Overbought Breadth (>40% RSI>70)', lambda p: p['pct_overbought'] > 40),
        ('Many Deep Pullbacks (>20% stocks -20% from high)', lambda p: p['pct_deep_pullback'] > 20),
        ('Many Deep Pullbacks (>30%)', lambda p: p['pct_deep_pullback'] > 30),
        ('Volume Surge Breadth (>30% stocks)', lambda p: p['pct_vol_surge'] > 30),
        ('Negative Momentum Breadth (>70%)', lambda p: p['pct_neg_momentum'] > 70),
        ('Poor Breadth (% above 50SMA < 30%)', lambda p: p['pct_above_50sma'] < 30),
        ('Strong Breadth (% above 50SMA > 80%)', lambda p: p['pct_above_50sma'] > 80),
        ('Breadth Divergence: SPY near high but breadth declining',
         lambda p: (p['pct_above_50sma'] < 50) & (p['pct_above_50sma'].shift(21) > 70)),
        ('Extreme Negative Z-scores (>15% stocks Z<-2)', lambda p: p['pct_extreme_zscore_neg'] > 15),
        ('Baseline (all days)', lambda p: pd.Series(True, index=p.index)),
    ]

    for name, filt in setups:
        try:
            mask = filt(panel)
            subset = panel[mask]
        except Exception:
            continue

        for horizon in ['spy_fwd_1w', 'spy_fwd_1m', 'spy_fwd_3m']:
            rets = subset[horizon].dropna()
            if len(rets) < 10:
                continue

            ci_lo, ci_hi = bootstrap_ci(rets.values, n_boot=2000) if len(rets) >= 20 else (np.nan, np.nan)

            results.append({
                'setup': name,
                'horizon': horizon.replace('spy_', ''),
                'n_days': len(rets),
                'mean_ret': rets.mean(),
                'median_ret': rets.median(),
                'p10': rets.quantile(0.10),
                'p90': rets.quantile(0.90),
                'hit_rate': (rets > 0).mean() * 100,
                'sharpe_ann': rets.mean() / rets.std() * np.sqrt(252/5 if '1w' in horizon else 12 if '1m' in horizon else 4) if rets.std() > 0 else 0,
                'skew': rets.skew(),
                'ci_95_lo': ci_lo,
                'ci_95_hi': ci_hi,
            })

    return pd.DataFrame(results)


# ── MARKET-WIDE ANALYSIS (Task 2) ────────────────────────────────────────────
def market_wide_analysis(close, volume, signals_dict):
    """Sector dispersion, factor crowding, vol compression, cross-asset signals."""
    dates = close.loc['2015-01-01':].index
    spy = close['SPY'].reindex(dates)

    mkt = pd.DataFrame(index=dates)
    mkt['spy'] = spy
    mkt['spy_ret'] = spy.pct_change()
    mkt['spy_fwd_1w'] = spy.pct_change(5).shift(-5) * 100
    mkt['spy_fwd_1m'] = spy.pct_change(21).shift(-21) * 100
    mkt['spy_fwd_3m'] = spy.pct_change(63).shift(-63) * 100

    # a. Sector dispersion
    sector_rets = pd.DataFrame()
    for etf in SECTOR_ETFS:
        if etf in close.columns:
            sector_rets[etf] = close[etf].reindex(dates).pct_change(21) * 100
    mkt['sector_dispersion'] = sector_rets.std(axis=1)

    # b. Factor crowding - momentum stocks RSI
    # Top quintile momentum stocks' average RSI
    mom_rsi_list = []
    for ticker, df in signals_dict.items():
        df_a = df[['mom_3m', 'rsi']].reindex(dates)
        mom_rsi_list.append(df_a.rename(columns={'mom_3m': f'mom_{ticker}', 'rsi': f'rsi_{ticker}'}))

    if mom_rsi_list:
        mom_panel = pd.concat(mom_rsi_list, axis=1)
        mom_cols = [c for c in mom_panel.columns if c.startswith('mom_')]
        rsi_cols = [c for c in mom_panel.columns if c.startswith('rsi_')]

        # For each day, find top quintile momentum stocks, get their avg RSI
        def top_mom_avg_rsi(row):
            moms = row[mom_cols].dropna()
            rsis = row[rsi_cols].dropna()
            if len(moms) < 10:
                return np.nan
            threshold = moms.quantile(0.8)
            top_tickers = [c.replace('mom_', '') for c in moms[moms >= threshold].index]
            top_rsi_cols = [f'rsi_{t}' for t in top_tickers if f'rsi_{t}' in rsis.index]
            if not top_rsi_cols:
                return np.nan
            return rsis[top_rsi_cols].mean()

        # Sample every 5th day for speed
        sample_dates = dates[::5]
        avg_rsi_vals = mom_panel.loc[sample_dates].apply(top_mom_avg_rsi, axis=1)
        mkt['momentum_crowding_rsi'] = avg_rsi_vals.reindex(dates).ffill()

    # c. Volatility compression
    # Individual stock vol (median across stocks) + market vol (SPY 20d vol)
    stock_vols = pd.DataFrame()
    for ticker, df in signals_dict.items():
        stock_vols[ticker] = df['vol_20d'].reindex(dates)
    mkt['median_stock_vol'] = stock_vols.median(axis=1)
    mkt['spy_vol_20d'] = mkt['spy_ret'].rolling(20).std() * np.sqrt(252) * 100

    # Vol percentile (1-year rolling)
    mkt['stock_vol_pctile'] = mkt['median_stock_vol'].rolling(252).rank(pct=True)
    mkt['spy_vol_pctile'] = mkt['spy_vol_20d'].rolling(252).rank(pct=True)

    # d. Breadth divergence - handled in cross_stock_signals

    # e. Cross-asset: TLT + GLD rising while SPY flat
    if 'TLT' in close.columns and 'GLD' in close.columns:
        tlt = close['TLT'].reindex(dates)
        gld = close['GLD'].reindex(dates)
        mkt['tlt_ret_21d'] = tlt.pct_change(21) * 100
        mkt['gld_ret_21d'] = gld.pct_change(21) * 100
        mkt['spy_ret_21d'] = spy.pct_change(21) * 100

    # VIX level
    if 'VIX' in close.columns:
        mkt['vix'] = close['VIX'].reindex(dates)

    # Shift all signals by 1 for T-1
    signal_cols = ['sector_dispersion', 'momentum_crowding_rsi', 'median_stock_vol',
                   'spy_vol_20d', 'stock_vol_pctile', 'spy_vol_pctile',
                   'tlt_ret_21d', 'gld_ret_21d', 'spy_ret_21d', 'vix']
    for col in signal_cols:
        if col in mkt.columns:
            mkt[col] = mkt[col].shift(1)

    return mkt


def analyze_market_setups(mkt):
    """Analyze market-wide conditional setups."""
    results = []

    setups = [
        # Sector dispersion
        ('Extreme Sector Dispersion (top 10%)',
         lambda m: m['sector_dispersion'] > m['sector_dispersion'].quantile(0.9)),
        ('Low Sector Dispersion (bottom 10%)',
         lambda m: m['sector_dispersion'] < m['sector_dispersion'].quantile(0.1)),

        # Factor crowding
        ('Momentum Crowding (top mom stocks RSI > 70)',
         lambda m: m['momentum_crowding_rsi'] > 70),
        ('Momentum Crowding Extreme (RSI > 75)',
         lambda m: m['momentum_crowding_rsi'] > 75),

        # Vol compression
        ('Dual Vol Compression (both stock & SPY vol bottom 20%)',
         lambda m: (m['stock_vol_pctile'] < 0.2) & (m['spy_vol_pctile'] < 0.2)),
        ('Vol Expansion (both top 20%)',
         lambda m: (m['stock_vol_pctile'] > 0.8) & (m['spy_vol_pctile'] > 0.8)),

        # Cross-asset flight to safety
        ('Flight to Safety (TLT>2%, GLD>2%, SPY<1% over 21d)',
         lambda m: (m['tlt_ret_21d'] > 2) & (m['gld_ret_21d'] > 2) & (m['spy_ret_21d'] < 1)),
        ('Risk-On (SPY>3%, TLT<0% over 21d)',
         lambda m: (m['spy_ret_21d'] > 3) & (m['tlt_ret_21d'] < 0)),

        # VIX regimes
        ('VIX > 30 (Fear)',
         lambda m: m['vix'] > 30),
        ('VIX > 25',
         lambda m: m['vix'] > 25),
        ('VIX < 13 (Complacency)',
         lambda m: m['vix'] < 13),

        # Combinations
        ('Vol Compression + Low VIX (<15)',
         lambda m: (m['stock_vol_pctile'] < 0.2) & (m['vix'] < 15)),
        ('High Dispersion + High VIX (>25)',
         lambda m: (m['sector_dispersion'] > m['sector_dispersion'].quantile(0.8)) & (m['vix'] > 25)),

        ('Baseline', lambda m: pd.Series(True, index=m.index)),
    ]

    for name, filt in setups:
        try:
            mask = filt(mkt)
            subset = mkt[mask]
        except Exception:
            continue

        for horizon in ['spy_fwd_1w', 'spy_fwd_1m', 'spy_fwd_3m']:
            rets = subset[horizon].dropna()
            if len(rets) < 10:
                continue

            ci_lo, ci_hi = bootstrap_ci(rets.values, n_boot=2000) if len(rets) >= 20 else (np.nan, np.nan)

            results.append({
                'setup': name,
                'horizon': horizon.replace('spy_', ''),
                'n_days': len(rets),
                'mean_ret': rets.mean(),
                'median_ret': rets.median(),
                'p10': rets.quantile(0.10),
                'p90': rets.quantile(0.90),
                'hit_rate': (rets > 0).mean() * 100,
                'sharpe_ann': rets.mean() / rets.std() * np.sqrt(252/5 if '1w' in horizon else 12 if '1m' in horizon else 4) if rets.std() > 0 else 0,
                'skew': rets.skew(),
                'ci_95_lo': ci_lo,
                'ci_95_hi': ci_hi,
            })

    return pd.DataFrame(results)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("STOCK-LEVEL & MARKET-WIDE ASYMMETRIC SIGNAL ANALYSIS")
    print("=" * 80)

    # 1. Download data
    print("\n[1/6] Downloading price data...")
    close, volume = download_data()

    # 2. Compute stock signals
    print("\n[2/6] Computing stock-level signals...")
    signals_dict = compute_stock_signals(close, volume)
    print(f"  Computed signals for {len(signals_dict)} stocks")

    # 3. Combine into single DataFrame for analysis
    print("\n[3/6] Running quintile + conditional analysis...")
    all_data = pd.concat(signals_dict.values(), ignore_index=False)

    # Quintile analysis for each signal x horizon
    quintile_results = []
    signal_cols = ['rsi', 'dist_from_high', 'vol_20d', 'volume_surge',
                   'mom_3m', 'sector_rel_strength', 'mean_rev_zscore']

    for sig in signal_cols:
        for horizon in ['fwd_1w', 'fwd_1m', 'fwd_3m']:
            qr = quintile_analysis(all_data, sig, horizon)
            if qr is not None:
                quintile_results.append(qr)

    quintile_df = pd.concat(quintile_results, ignore_index=True) if quintile_results else pd.DataFrame()
    print(f"  Quintile analysis: {len(quintile_df)} rows")

    # Conditional analysis
    cond_df = conditional_analysis(all_data)
    print(f"  Conditional analysis: {len(cond_df)} rows")

    # 4. Cross-stock breadth
    print("\n[4/6] Computing breadth signals...")
    breadth_panel = cross_stock_signals(signals_dict, close)
    breadth_results = analyze_breadth_signals(breadth_panel)
    print(f"  Breadth analysis: {len(breadth_results)} rows")

    # 5. Market-wide analysis
    print("\n[5/6] Running market-wide analysis...")
    mkt = market_wide_analysis(close, volume, signals_dict)
    market_results = analyze_market_setups(mkt)
    print(f"  Market-wide analysis: {len(market_results)} rows")

    # 6. Save results
    print("\n[6/6] Saving results...")

    # Stock signal analysis CSV (quintile + conditional combined)
    stock_csv = pd.concat([
        quintile_df.assign(analysis_type='quintile'),
        cond_df.rename(columns={'setup': 'signal'}).assign(analysis_type='conditional')
    ], ignore_index=True)
    stock_csv.to_csv(f'{OUTPUT_DIR}/stock_signal_analysis.csv', index=False)

    # Market wide CSV (breadth + market setups)
    market_csv = pd.concat([
        breadth_results.assign(analysis_type='breadth'),
        market_results.assign(analysis_type='market_regime')
    ], ignore_index=True)
    market_csv.to_csv(f'{OUTPUT_DIR}/market_wide_analysis.csv', index=False)

    # ── SUMMARY REPORT ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("STOCK-LEVEL & MARKET-WIDE ASYMMETRIC SIGNAL ANALYSIS")
    report_lines.append(f"Period: 2015-01 to 2026-07 | Universe: {len(signals_dict)} large-cap stocks")
    report_lines.append(f"Total stock-day observations: {len(all_data):,}")
    report_lines.append("=" * 80)

    # ── TOP STOCK-LEVEL FINDINGS ──
    report_lines.append("\n" + "─" * 80)
    report_lines.append("SECTION 1: STOCK-LEVEL QUINTILE ANALYSIS (KEY FINDINGS)")
    report_lines.append("─" * 80)

    # Show extreme quintiles for key signals
    for sig in signal_cols:
        sig_data = quintile_df[quintile_df['signal'] == sig]
        if sig_data.empty:
            continue

        report_lines.append(f"\n  Signal: {sig}")
        for horizon in ['fwd_1w', 'fwd_1m', 'fwd_3m']:
            h_data = sig_data[sig_data['horizon'] == horizon]
            if h_data.empty:
                continue
            q1 = h_data[h_data['quintile'] == 1].iloc[0] if len(h_data[h_data['quintile'] == 1]) > 0 else None
            q5 = h_data[h_data['quintile'] == h_data['quintile'].max()].iloc[0] if len(h_data) > 0 else None
            if q1 is not None and q5 is not None:
                spread = q1['mean_ret'] - q5['mean_ret']
                report_lines.append(
                    f"    {horizon}: Q1 mean={q1['mean_ret']:+.2f}% (HR={q1['hit_rate']:.0f}%) | "
                    f"Q5 mean={q5['mean_ret']:+.2f}% (HR={q5['hit_rate']:.0f}%) | "
                    f"Spread={spread:+.2f}%"
                )

    # ── TOP 5 STOCK-LEVEL SETUPS ──
    report_lines.append("\n" + "─" * 80)
    report_lines.append("SECTION 2: TOP 5 STOCK-LEVEL ASYMMETRIC SETUPS")
    report_lines.append("─" * 80)

    # Rank by asymmetry: positive mean + high hit rate + positive skew at 1m horizon
    cond_1m = cond_df[cond_df['horizon'] == 'fwd_1m'].copy()
    baseline_1m = cond_1m[cond_1m['setup'] == 'Baseline (all observations)']
    baseline_mean = baseline_1m['mean_ret'].values[0] if len(baseline_1m) > 0 else 0

    cond_1m_filtered = cond_1m[cond_1m['setup'] != 'Baseline (all observations)'].copy()
    cond_1m_filtered['asymmetry_score'] = (
        (cond_1m_filtered['mean_ret'] - baseline_mean) *
        np.clip(cond_1m_filtered['hit_rate'] / 50, 0.5, 2) *
        (1 + np.clip(cond_1m_filtered['upside_downside_ratio'] - 1, -0.5, 2))
    )

    top5_stock = cond_1m_filtered.nlargest(5, 'asymmetry_score')

    for i, (_, row) in enumerate(top5_stock.iterrows(), 1):
        report_lines.append(f"\n  #{i}: {row['setup']}")
        report_lines.append(f"      1-Month Forward: mean={row['mean_ret']:+.2f}%, median={row['median_ret']:+.2f}%")
        report_lines.append(f"      Hit Rate: {row['hit_rate']:.1f}% | N={row['n_obs']:,} observations across {row['n_stocks']} stocks")
        report_lines.append(f"      Distribution: p10={row['p10']:+.2f}%, p25={row['p25']:+.2f}%, p75={row['p75']:+.2f}%, p90={row['p90']:+.2f}%")
        report_lines.append(f"      Upside/Downside: {row['upside_downside_ratio']:.2f}x | Skew: {row['skew']:+.2f}")
        report_lines.append(f"      95% CI on mean: [{row['ci_95_lo']:+.2f}%, {row['ci_95_hi']:+.2f}%]")

        # Also show 1w and 3m for this setup
        for h in ['fwd_1w', 'fwd_3m']:
            h_row = cond_df[(cond_df['setup'] == row['setup']) & (cond_df['horizon'] == h)]
            if len(h_row) > 0:
                hr = h_row.iloc[0]
                label = '1-Week' if '1w' in h else '3-Month'
                report_lines.append(f"      {label}: mean={hr['mean_ret']:+.2f}%, HR={hr['hit_rate']:.1f}%")

    # ── WORST 3 SETUPS (traps to avoid) ──
    report_lines.append("\n" + "─" * 80)
    report_lines.append("SECTION 3: WORST SETUPS (TRAPS TO AVOID)")
    report_lines.append("─" * 80)

    worst3 = cond_1m_filtered.nsmallest(3, 'asymmetry_score')
    for i, (_, row) in enumerate(worst3.iterrows(), 1):
        report_lines.append(f"\n  TRAP #{i}: {row['setup']}")
        report_lines.append(f"      1-Month Forward: mean={row['mean_ret']:+.2f}%, HR={row['hit_rate']:.1f}%")
        report_lines.append(f"      Distribution: p10={row['p10']:+.2f}%, p90={row['p90']:+.2f}%")

    # ── TOP 5 MARKET-WIDE SETUPS ──
    report_lines.append("\n" + "─" * 80)
    report_lines.append("SECTION 4: TOP 5 MARKET-WIDE ASYMMETRIC SETUPS")
    report_lines.append("─" * 80)

    all_market = pd.concat([breadth_results, market_results], ignore_index=True)
    mkt_1m = all_market[all_market['horizon'] == 'fwd_1m'].copy()
    mkt_baseline = mkt_1m[mkt_1m['setup'] == 'Baseline']
    mkt_baseline_mean = mkt_baseline['mean_ret'].values[0] if len(mkt_baseline) > 0 else 0

    mkt_1m_filt = mkt_1m[~mkt_1m['setup'].str.contains('Baseline')].copy()
    mkt_1m_filt['edge'] = mkt_1m_filt['mean_ret'] - mkt_baseline_mean
    mkt_1m_filt['asymmetry'] = mkt_1m_filt['edge'] * (mkt_1m_filt['hit_rate'] / 50)

    top5_mkt = mkt_1m_filt.nlargest(5, 'asymmetry')

    for i, (_, row) in enumerate(top5_mkt.iterrows(), 1):
        report_lines.append(f"\n  #{i}: {row['setup']}")
        report_lines.append(f"      SPY 1-Month Forward: mean={row['mean_ret']:+.2f}%, median={row['median_ret']:+.2f}%")
        report_lines.append(f"      Hit Rate: {row['hit_rate']:.1f}% | N={row['n_days']} days")
        report_lines.append(f"      Distribution: p10={row['p10']:+.2f}%, p90={row['p90']:+.2f}%")
        report_lines.append(f"      Edge vs baseline: {row['edge']:+.2f}%")
        if not np.isnan(row.get('ci_95_lo', np.nan)):
            report_lines.append(f"      95% CI: [{row['ci_95_lo']:+.2f}%, {row['ci_95_hi']:+.2f}%]")

        for h in ['fwd_1w', 'fwd_3m']:
            h_row = all_market[(all_market['setup'] == row['setup']) & (all_market['horizon'] == h)]
            if len(h_row) > 0:
                hr = h_row.iloc[0]
                label = '1-Week' if '1w' in h else '3-Month'
                report_lines.append(f"      {label}: mean={hr['mean_ret']:+.2f}%, HR={hr['hit_rate']:.1f}%")

    # ── WORST MARKET SETUPS ──
    report_lines.append("\n" + "─" * 80)
    report_lines.append("SECTION 5: WORST MARKET SETUPS (REDUCE EXPOSURE)")
    report_lines.append("─" * 80)

    worst3_mkt = mkt_1m_filt.nsmallest(3, 'asymmetry')
    for i, (_, row) in enumerate(worst3_mkt.iterrows(), 1):
        report_lines.append(f"\n  WARNING #{i}: {row['setup']}")
        report_lines.append(f"      SPY 1-Month Forward: mean={row['mean_ret']:+.2f}%, HR={row['hit_rate']:.1f}%")
        report_lines.append(f"      Distribution: p10={row['p10']:+.2f}%, p90={row['p90']:+.2f}%")

    # ── KEY CONDITIONAL QUESTION ──
    report_lines.append("\n" + "─" * 80)
    report_lines.append("SECTION 6: THE KEY QUESTION — DEEP PULLBACK + OVERSOLD + VOLUME SURGE")
    report_lines.append("─" * 80)

    key_setup = cond_df[cond_df['setup'] == 'Deep Pullback + Oversold + Volume Surge']
    if len(key_setup) > 0:
        for _, row in key_setup.iterrows():
            label = row['horizon'].replace('fwd_', '')
            report_lines.append(f"\n  Horizon: {label}")
            report_lines.append(f"    N={row['n_obs']} obs across {row['n_stocks']} stocks")
            report_lines.append(f"    Mean return: {row['mean_ret']:+.2f}% | Median: {row['median_ret']:+.2f}%")
            report_lines.append(f"    Hit rate: {row['hit_rate']:.1f}%")
            report_lines.append(f"    p10={row['p10']:+.2f}% | p25={row['p25']:+.2f}% | p75={row['p75']:+.2f}% | p90={row['p90']:+.2f}%")
            report_lines.append(f"    Skew: {row['skew']:+.2f}")
            if not np.isnan(row.get('ci_95_lo', np.nan)):
                report_lines.append(f"    95% CI: [{row['ci_95_lo']:+.2f}%, {row['ci_95_hi']:+.2f}%]")
    else:
        report_lines.append("  Insufficient observations for this specific combination.")
        # Show the relaxed version
        relaxed = cond_df[cond_df['setup'] == 'Deep Pullback + Oversold (no vol filter)']
        if len(relaxed) > 0:
            report_lines.append("  Relaxed version (no volume filter):")
            for _, row in relaxed.iterrows():
                label = row['horizon'].replace('fwd_', '')
                report_lines.append(f"    {label}: mean={row['mean_ret']:+.2f}%, HR={row['hit_rate']:.1f}%, N={row['n_obs']}")

    # ── CROSS-STOCK BREADTH KEY FINDING ──
    report_lines.append("\n" + "─" * 80)
    report_lines.append("SECTION 7: BREADTH DIVERGENCE ANALYSIS")
    report_lines.append("─" * 80)

    div_setup = breadth_results[breadth_results['setup'].str.contains('Divergence')]
    if len(div_setup) > 0:
        for _, row in div_setup.iterrows():
            report_lines.append(f"  {row['setup']} ({row['horizon']})")
            report_lines.append(f"    N={row['n_days']} days | Mean SPY return: {row['mean_ret']:+.2f}%")
            report_lines.append(f"    Hit rate: {row['hit_rate']:.1f}% | p10={row['p10']:+.2f}%, p90={row['p90']:+.2f}%")
    else:
        report_lines.append("  No breadth divergence events found with sufficient observations.")

    # ── METHODOLOGY ──
    report_lines.append("\n" + "─" * 80)
    report_lines.append("METHODOLOGY NOTES")
    report_lines.append("─" * 80)
    report_lines.append("  - All signals are T-1 (computed on prior day's data, no lookahead)")
    report_lines.append("  - Forward returns are simple percentage returns")
    report_lines.append("  - Bootstrap CIs: 2000 resamples, 95% confidence")
    report_lines.append("  - Earnings proximity is approximate (quarterly month flags)")
    report_lines.append("  - Quintiles computed cross-sectionally across all stock-days")
    report_lines.append("  - Hit rate = P(return > 0 | signal state)")
    report_lines.append("  - Asymmetry score = (mean_excess_ret) * (hit_rate/50) * (1 + upside/downside - 1)")

    report = '\n'.join(report_lines)

    with open(f'{OUTPUT_DIR}/summary_report.txt', 'w') as f:
        f.write(report)

    print(report)
    print(f"\n\nFiles saved to {OUTPUT_DIR}/")
    print("  - stock_signal_analysis.csv")
    print("  - market_wide_analysis.csv")
    print("  - summary_report.txt")


if __name__ == '__main__':
    main()
