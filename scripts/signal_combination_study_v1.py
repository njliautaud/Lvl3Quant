#!/usr/bin/env python3
"""
Signal Combination Study v1 — HC #725 (signal-first, find correlations)

Analyzes cross-signal correlations and conditional return distributions
to find when COMBINATIONS of signals predict asymmetric upside.

Individual validated signals:
  1. IV Rank ≥ 70% (jade lizard, Sharpe 0.95)
  2. Post-earnings drift predictor (earnings asymmetry, Sharpe 1.05 but sparse)
  3. Momentum (3m, 6m) — top feature in stock ranker
  4. Vol compression (vol_20d / vol_63d ratio)
  5. Distance from 52-week high/low
  6. VIX regime (term structure)

Question: Do combinations create ASYMMETRIC payoff profiles (3:1+ upside/downside)?

Output: Correlation matrix, conditional return distributions, combined signal backtest.
All signals computed on T-1 data (HC #724 anti-lookahead).
SLIDING walk-forward (HC #0).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# === CONFIG ===
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/signal_combination_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "signal_combo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Universe — same 50 stocks as jade lizard + earnings
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'GS', 'BAC',
    'V', 'MA', 'UNH', 'JNJ', 'PG', 'KO', 'PEP', 'MRK', 'ABBV', 'LLY',
    'HD', 'COST', 'WMT', 'CRM', 'AMD', 'NFLX', 'ADBE', 'INTC', 'CSCO', 'QCOM',
    'XOM', 'CVX', 'PFE', 'TMO', 'ABT', 'AVGO', 'TXN', 'MCD', 'NKE', 'DIS',
    'CMCSA', 'T', 'VZ', 'NEE', 'SO', 'SHW', 'LMT', 'RTX', 'CAT', 'DE',
]

SECTORS = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'AMZN': 'ConsDisc', 'META': 'Tech',
    'NVDA': 'Tech', 'TSLA': 'ConsDisc', 'JPM': 'Fin', 'GS': 'Fin', 'BAC': 'Fin',
    'V': 'Fin', 'MA': 'Fin', 'UNH': 'Health', 'JNJ': 'Health', 'PG': 'Staples',
    'KO': 'Staples', 'PEP': 'Staples', 'MRK': 'Health', 'ABBV': 'Health', 'LLY': 'Health',
    'HD': 'ConsDisc', 'COST': 'Staples', 'WMT': 'Staples', 'CRM': 'Tech', 'AMD': 'Tech',
    'NFLX': 'Tech', 'ADBE': 'Tech', 'INTC': 'Tech', 'CSCO': 'Tech', 'QCOM': 'Tech',
    'XOM': 'Energy', 'CVX': 'Energy', 'PFE': 'Health', 'TMO': 'Health', 'ABT': 'Health',
    'AVGO': 'Tech', 'TXN': 'Tech', 'MCD': 'ConsDisc', 'NKE': 'ConsDisc', 'DIS': 'ConsDisc',
    'CMCSA': 'Comm', 'T': 'Comm', 'VZ': 'Comm', 'NEE': 'Util', 'SO': 'Util',
    'SHW': 'Materials', 'LMT': 'Indust', 'RTX': 'Indust', 'CAT': 'Indust', 'DE': 'Indust',
}

START_DATE = '2015-01-01'
END_DATE = '2026-07-18'
COST_BPS = 10  # 10 bps round-trip FIFO
FORWARD_HORIZONS = [5, 10, 21, 42, 63]  # 1wk, 2wk, 1mo, 2mo, 3mo
ASYMMETRY_RATIO_MIN = 2.0  # Minimum upside/downside ratio to flag

def download_data():
    """Download price data + VIX for signal computation."""
    logger.info("Downloading price data for %d tickers...", len(UNIVERSE))

    # Download stock prices
    tickers = UNIVERSE + ['^VIX', '^VIX3M', 'SPY']
    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, threads=True)

    close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data['Adj Close']
    volume = data['Volume']
    high = data['High']
    low = data['Low']

    logger.info("Downloaded %d days of data", len(close))
    return close, volume, high, low


def compute_signals(close, volume, high, low):
    """
    Compute all signals on T-1 data (HC #724 anti-lookahead).
    Returns DataFrame with multi-index (date, ticker) and signal columns.
    """
    logger.info("Computing signals (all T-1 lagged)...")

    vix = close['^VIX'] if '^VIX' in close.columns else None
    vix3m = close['^VIX3M'] if '^VIX3M' in close.columns else None
    spy = close['SPY'] if 'SPY' in close.columns else None

    records = []

    # Pre-compute VIX signals (outside loop for speed)
    vix_shifted = vix.shift(1) if vix is not None else None
    vix_pctrank_series = None
    if vix is not None and len(vix) > 252:
        vix_pctrank_series = vix.rolling(252).apply(
            lambda x: (x[-1] >= x).sum() / len(x), raw=True
        ).shift(1)
    vts_series = None
    if vix3m is not None and vix is not None:
        vts_series = (vix3m / vix.replace(0, np.nan)).shift(1)
    spy_sma200_series = spy.rolling(200).mean().shift(1) if spy is not None else None
    spy_shifted = spy.shift(1) if spy is not None else None
    spy_mom_1m_series = spy.pct_change(21).shift(1) if spy is not None else None

    for ticker in UNIVERSE:
        if ticker not in close.columns:
            logger.warning("Skipping %s — no data", ticker)
            continue

        c = close[ticker].dropna()
        v = volume[ticker].dropna() if ticker in volume.columns else None
        h = high[ticker].dropna() if ticker in high.columns else None
        l = low[ticker].dropna() if ticker in low.columns else None

        if len(c) < 252:
            continue

        # Compute signals — all use data through T-1 (shift by 1)
        # Momentum signals
        mom_1m = c.pct_change(21).shift(1)
        mom_3m = c.pct_change(63).shift(1)
        mom_6m = c.pct_change(126).shift(1)
        mom_12m = c.pct_change(252).shift(1)

        # Momentum acceleration (3m vs 6m)
        mom_accel = (mom_3m - mom_6m / 2).shift(0)  # already shifted

        # RSI-14
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - 100 / (1 + rs)).shift(1)

        # Volatility signals
        ret = c.pct_change()
        vol_20d = ret.rolling(20).std().shift(1) * np.sqrt(252)
        vol_63d = ret.rolling(63).std().shift(1) * np.sqrt(252)
        vol_ratio = (vol_20d / vol_63d.replace(0, np.nan))  # Vol compression indicator

        # Distance from 52-week high/low
        high_52w = c.rolling(252).max().shift(1)
        low_52w = c.rolling(252).min().shift(1)
        dist_high = ((c.shift(1) - high_52w) / high_52w)
        dist_low = ((c.shift(1) - low_52w) / low_52w.replace(0, np.nan))

        # IV rank proxy (realized vol percentile over 1 year) — fast vectorized
        # Count how many of the past 252 values are <= current value
        vol_pctrank = vol_20d.rolling(252).apply(
            lambda x: (x[-1] >= x).sum() / len(x), raw=True
        )

        # Volume trend
        vol_ma_20 = v.rolling(20).mean().shift(1) if v is not None else pd.Series(np.nan, index=c.index)
        vol_ma_63 = v.rolling(63).mean().shift(1) if v is not None else pd.Series(np.nan, index=c.index)
        vol_trend = (vol_ma_20 / vol_ma_63.replace(0, np.nan)) if v is not None else pd.Series(np.nan, index=c.index)

        # Mean reversion z-score (price vs 20d MA)
        sma_20 = c.rolling(20).mean().shift(1)
        mr_zscore = ((c.shift(1) - sma_20) / (ret.rolling(20).std().shift(1) * c.shift(1))).replace([np.inf, -np.inf], np.nan)

        # Price vs 200 SMA
        sma_200 = c.rolling(200).mean().shift(1)
        price_vs_sma200 = (c.shift(1) / sma_200.replace(0, np.nan)) - 1

        # Max drawdown in trailing 252 days
        rolling_max = c.rolling(252).max().shift(1)
        max_dd_252 = ((c.shift(1) - rolling_max) / rolling_max)

        # Forward returns at multiple horizons (for analysis — NOT used as features)
        fwd_rets = {}
        for h in FORWARD_HORIZONS:
            fwd_rets[f'fwd_ret_{h}d'] = c.pct_change(h).shift(-h)  # True future return

        # Build per-date records
        common_idx = c.index[252:]  # Need 252 days of history

        for dt in common_idx:
            if pd.isna(mom_3m.get(dt, np.nan)):
                continue

            rec = {
                'date': dt,
                'ticker': ticker,
                'sector': SECTORS.get(ticker, 'Other'),
                # Signals
                'mom_1m': mom_1m.get(dt, np.nan),
                'mom_3m': mom_3m.get(dt, np.nan),
                'mom_6m': mom_6m.get(dt, np.nan),
                'mom_12m': mom_12m.get(dt, np.nan),
                'mom_accel': mom_accel.get(dt, np.nan),
                'rsi': rsi.get(dt, np.nan),
                'vol_20d': vol_20d.get(dt, np.nan),
                'vol_63d': vol_63d.get(dt, np.nan),
                'vol_ratio': vol_ratio.get(dt, np.nan),
                'dist_high': dist_high.get(dt, np.nan),
                'dist_low': dist_low.get(dt, np.nan),
                'iv_rank_proxy': vol_pctrank.get(dt, np.nan),
                'vol_trend': vol_trend.get(dt, np.nan),
                'mr_zscore': mr_zscore.get(dt, np.nan),
                'price_vs_sma200': price_vs_sma200.get(dt, np.nan),
                'max_dd_252': max_dd_252.get(dt, np.nan),
            }

            # Add VIX signals (pre-computed)
            if vix_shifted is not None and dt in vix_shifted.index:
                rec['vix'] = vix_shifted.get(dt, np.nan)
                rec['vix_pctrank'] = vix_pctrank_series.get(dt, np.nan) if vix_pctrank_series is not None else np.nan

            if vts_series is not None and dt in vts_series.index:
                rec['vix_term_structure'] = vts_series.get(dt, np.nan)

            # SPY regime (pre-computed)
            if spy_shifted is not None and dt in spy_shifted.index:
                spy_sma200 = spy_sma200_series.get(dt, np.nan) if spy_sma200_series is not None else np.nan
                rec['spy_above_200sma'] = 1.0 if spy_shifted.get(dt, np.nan) > spy_sma200 else 0.0
                rec['spy_mom_1m'] = spy_mom_1m_series.get(dt, np.nan) if spy_mom_1m_series is not None else np.nan

            # Forward returns
            for h in FORWARD_HORIZONS:
                rec[f'fwd_ret_{h}d'] = fwd_rets[f'fwd_ret_{h}d'].get(dt, np.nan)

            records.append(rec)

    df = pd.DataFrame(records)
    logger.info("Signal matrix: %d rows, %d columns", len(df), len(df.columns))
    return df


def analyze_signal_correlations(df):
    """Step 1: Pairwise correlations between signals and forward returns."""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Signal-Return Correlations")
    logger.info("="*60)

    signal_cols = [c for c in df.columns if c not in ['date', 'ticker', 'sector'] and not c.startswith('fwd_')]
    fwd_cols = [c for c in df.columns if c.startswith('fwd_')]

    results = {}

    for horizon_col in fwd_cols:
        corrs = {}
        for sig in signal_cols:
            valid = df[[sig, horizon_col]].dropna()
            if len(valid) > 100:
                corrs[sig] = valid[sig].corr(valid[horizon_col])

        # Sort by absolute correlation
        sorted_corrs = dict(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True))
        results[horizon_col] = sorted_corrs

        logger.info("\n%s — Top 10 correlations:", horizon_col)
        for i, (sig, corr) in enumerate(sorted_corrs.items()):
            if i >= 10:
                break
            logger.info("  %2d. %-20s  r=%.4f", i+1, sig, corr)

    return results


def analyze_conditional_returns(df):
    """
    Step 2: For each signal, split into quintiles and measure forward return distribution.
    Focus on asymmetry: mean / abs(mean of losers), skewness, P(>5% gain) / P(>5% loss).
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Conditional Return Distributions (Asymmetry Analysis)")
    logger.info("="*60)

    signal_cols = ['mom_3m', 'mom_6m', 'rsi', 'vol_ratio', 'dist_high', 'dist_low',
                   'iv_rank_proxy', 'mr_zscore', 'price_vs_sma200', 'max_dd_252',
                   'vix_pctrank', 'vix_term_structure', 'spy_above_200sma']

    results = {}

    for sig in signal_cols:
        if sig not in df.columns:
            continue

        valid = df[[sig, 'fwd_ret_21d']].dropna()
        if len(valid) < 500:
            continue

        # Create quintiles
        try:
            valid['quintile'] = pd.qcut(valid[sig], 5, labels=['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high'], duplicates='drop')
        except ValueError:
            continue

        sig_results = {}
        for q in valid['quintile'].unique():
            subset = valid[valid['quintile'] == q]['fwd_ret_21d']
            if len(subset) < 20:
                continue

            gains = subset[subset > 0]
            losses = subset[subset < 0]
            big_gains = subset[subset > 0.05]  # >5% gain
            big_losses = subset[subset < -0.05]  # >5% loss

            avg_gain = gains.mean() if len(gains) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.001

            sig_results[str(q)] = {
                'n': len(subset),
                'mean': float(subset.mean()),
                'median': float(subset.median()),
                'std': float(subset.std()),
                'skew': float(subset.skew()),
                'wr': float(len(gains) / len(subset)),
                'avg_gain': float(avg_gain),
                'avg_loss': float(avg_loss),
                'gain_loss_ratio': float(avg_gain / avg_loss) if avg_loss > 0 else 0,
                'p_big_gain': float(len(big_gains) / len(subset)),
                'p_big_loss': float(len(big_losses) / len(subset)),
                'asymmetry_ratio': float(len(big_gains) / max(len(big_losses), 1)),
            }

        results[sig] = sig_results

        # Log key findings
        if 'Q5_high' in sig_results and 'Q1_low' in sig_results:
            q5 = sig_results['Q5_high']
            q1 = sig_results['Q1_low']
            logger.info("\n%s:", sig)
            logger.info("  Q1 (low):  mean=%.3f%%, WR=%.1f%%, asymmetry=%.1f, P(>5%%)=%.1f%%, P(<-5%%)=%.1f%%",
                        q1['mean']*100, q1['wr']*100, q1['asymmetry_ratio'], q1['p_big_gain']*100, q1['p_big_loss']*100)
            logger.info("  Q5 (high): mean=%.3f%%, WR=%.1f%%, asymmetry=%.1f, P(>5%%)=%.1f%%, P(<-5%%)=%.1f%%",
                        q5['mean']*100, q5['wr']*100, q5['asymmetry_ratio'], q5['p_big_gain']*100, q5['p_big_loss']*100)

            # Flag asymmetric setups
            for qname, qdata in sig_results.items():
                if qdata['asymmetry_ratio'] >= ASYMMETRY_RATIO_MIN:
                    logger.info("  *** ASYMMETRIC SETUP: %s %s — big gain/loss ratio = %.1f ***",
                               sig, qname, qdata['asymmetry_ratio'])

    return results


def analyze_combined_signals(df):
    """
    Step 3: Test signal COMBINATIONS for stronger asymmetry.

    Hypotheses from validated research:
    H1: High IV rank + positive momentum → jade lizard + growth overlap
    H2: Oversold (low RSI) + near 52-week low + vol compression → mean reversion bounce
    H3: Strong momentum + vol expansion → trend continuation
    H4: VIX backwardation + oversold stocks → market-wide asymmetric upside
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Combined Signal Analysis")
    logger.info("="*60)

    combinations = {
        'H1_highIV_posMom': {
            'description': 'High IV rank (>70 pctile) + positive 3m momentum',
            'filter': lambda d: (d['iv_rank_proxy'] > 0.70) & (d['mom_3m'] > 0),
        },
        'H2_oversold_volCompress': {
            'description': 'Oversold (RSI<30) + vol compression (ratio<0.8) + near 52w low (<10%)',
            'filter': lambda d: (d['rsi'] < 30) & (d['vol_ratio'] < 0.8) & (d['dist_low'] < 0.10),
        },
        'H3_momentum_volExpand': {
            'description': 'Strong 3m mom (>15%) + vol expanding (ratio>1.2)',
            'filter': lambda d: (d['mom_3m'] > 0.15) & (d['vol_ratio'] > 1.2),
        },
        'H4_vixBackward_oversold': {
            'description': 'VIX term structure < 1.0 (backwardation) + RSI < 40',
            'filter': lambda d: (d.get('vix_term_structure', 1.0) < 1.0) & (d['rsi'] < 40),
        },
        'H5_deepValue_recovery': {
            'description': 'Max DD > 20% from 52w high + momentum turning positive (1m>0, 3m<0)',
            'filter': lambda d: (d['max_dd_252'] < -0.20) & (d['mom_1m'] > 0) & (d['mom_3m'] < 0),
        },
        'H6_qualityMomentum': {
            'description': 'Above 200 SMA + 3m mom > 10% + low vol (vol_20d < 25%)',
            'filter': lambda d: (d['price_vs_sma200'] > 0) & (d['mom_3m'] > 0.10) & (d['vol_20d'] < 0.25),
        },
        'H7_ivRank_nearLow_momentum': {
            'description': 'IV rank > 60% + within 20% of 52w low + 1m mom positive (bounce setup)',
            'filter': lambda d: (d['iv_rank_proxy'] > 0.60) & (d['dist_low'] < 0.20) & (d['mom_1m'] > 0),
        },
        'H8_volCrush_setup': {
            'description': 'IV rank > 80% + vol ratio > 1.5 (vol spike) — premium selling sweet spot',
            'filter': lambda d: (d['iv_rank_proxy'] > 0.80) & (d['vol_ratio'] > 1.5),
        },
    }

    results = {}

    for name, combo in combinations.items():
        try:
            mask = combo['filter'](df)
            subset = df[mask].dropna(subset=['fwd_ret_21d'])
        except (KeyError, TypeError):
            logger.warning("  %s: missing required columns, skipping", name)
            continue

        baseline = df.dropna(subset=['fwd_ret_21d'])

        if len(subset) < 20:
            logger.info("  %s: only %d observations, skipping (need ≥20)", name, len(subset))
            continue

        fwd = subset['fwd_ret_21d']
        base_fwd = baseline['fwd_ret_21d']

        gains = fwd[fwd > 0]
        losses = fwd[fwd < 0]
        big_gains = fwd[fwd > 0.05]
        big_losses = fwd[fwd < -0.05]

        avg_gain = gains.mean() if len(gains) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.001

        # Multi-horizon analysis
        horizon_stats = {}
        for h in FORWARD_HORIZONS:
            col = f'fwd_ret_{h}d'
            if col in subset.columns:
                h_fwd = subset[col].dropna()
                if len(h_fwd) > 10:
                    horizon_stats[f'{h}d'] = {
                        'mean': float(h_fwd.mean()),
                        'median': float(h_fwd.median()),
                        'wr': float((h_fwd > 0).mean()),
                        'sharpe': float(h_fwd.mean() / h_fwd.std() * np.sqrt(252/h)) if h_fwd.std() > 0 else 0,
                    }

        result = {
            'description': combo['description'],
            'n_observations': int(len(subset)),
            'pct_of_total': float(len(subset) / len(baseline) * 100),
            'unique_dates': int(subset['date'].nunique()),
            'unique_tickers': int(subset['ticker'].nunique()),
            'mean_ret_21d': float(fwd.mean()),
            'median_ret_21d': float(fwd.median()),
            'std_ret_21d': float(fwd.std()),
            'skewness': float(fwd.skew()),
            'wr': float(len(gains) / len(fwd)),
            'avg_gain': float(avg_gain),
            'avg_loss': float(avg_loss),
            'gain_loss_ratio': float(avg_gain / avg_loss),
            'p_big_gain': float(len(big_gains) / len(fwd)),
            'p_big_loss': float(len(big_losses) / len(fwd)),
            'asymmetry_ratio': float(len(big_gains) / max(len(big_losses), 1)),
            'vs_baseline_mean': float(fwd.mean() - base_fwd.mean()),
            'vs_baseline_wr': float(len(gains)/len(fwd) - (base_fwd > 0).mean()),
            'horizon_stats': horizon_stats,
        }

        results[name] = result

        logger.info("\n%s: %s", name, combo['description'])
        logger.info("  N=%d (%.1f%% of total), %d unique dates, %d tickers",
                    result['n_observations'], result['pct_of_total'],
                    result['unique_dates'], result['unique_tickers'])
        logger.info("  21d return: mean=%.2f%%, median=%.2f%%, WR=%.1f%%",
                    result['mean_ret_21d']*100, result['median_ret_21d']*100, result['wr']*100)
        logger.info("  Gain/Loss ratio: %.2f, Asymmetry (big gain/big loss): %.1f",
                    result['gain_loss_ratio'], result['asymmetry_ratio'])
        logger.info("  P(>5%% gain)=%.1f%%, P(>5%% loss)=%.1f%%",
                    result['p_big_gain']*100, result['p_big_loss']*100)
        logger.info("  vs baseline: mean %+.2f%%, WR %+.1f%%",
                    result['vs_baseline_mean']*100, result['vs_baseline_wr']*100)

        # Multi-horizon
        for h_name, h_stats in horizon_stats.items():
            logger.info("  %s: mean=%.2f%%, WR=%.1f%%, Sharpe=%.2f",
                        h_name, h_stats['mean']*100, h_stats['wr']*100, h_stats['sharpe'])

        if result['asymmetry_ratio'] >= ASYMMETRY_RATIO_MIN:
            logger.info("  *** ASYMMETRIC SETUP DETECTED — ratio %.1f ***", result['asymmetry_ratio'])

    return results


def walk_forward_backtest(df, signal_name, filter_fn, hold_days=21, top_n=5):
    """
    Step 4: Walk-forward backtest of best combined signals.
    Monthly rebalance, pick top_n stocks from qualifying set by momentum.
    SLIDING window, T-1 signals only.
    """
    logger.info("\nWalk-forward backtest: %s (hold=%dd, top_n=%d)", signal_name, hold_days, top_n)

    dates = sorted(df['date'].unique())

    # Monthly rebalance dates
    rebal_dates = []
    last_month = None
    for dt in dates:
        m = pd.Timestamp(dt).month
        if m != last_month:
            rebal_dates.append(dt)
            last_month = m

    # Need enough history
    rebal_dates = rebal_dates[24:]  # Skip first 2 years for warmup

    portfolio_returns = []
    spy_returns = []

    for i in range(len(rebal_dates) - 1):
        dt = rebal_dates[i]
        next_dt = rebal_dates[i + 1]

        # Get qualifying stocks on this date
        day_data = df[df['date'] == dt].copy()

        try:
            mask = filter_fn(day_data)
            qualifying = day_data[mask].copy()
        except (KeyError, TypeError):
            continue

        if len(qualifying) == 0:
            portfolio_returns.append({'date': dt, 'ret': 0.0, 'n_stocks': 0})
            continue

        # Pick top N by 3m momentum (strongest trend)
        qualifying = qualifying.nlargest(top_n, 'mom_3m')
        tickers = qualifying['ticker'].tolist()

        # Get actual forward returns for these tickers
        period_data = df[(df['date'] >= dt) & (df['date'] < next_dt)]

        ticker_rets = []
        for t in tickers:
            t_data = period_data[period_data['ticker'] == t]
            if len(t_data) > 0 and not pd.isna(t_data.iloc[0].get('fwd_ret_21d', np.nan)):
                ticker_rets.append(t_data.iloc[0]['fwd_ret_21d'])

        if len(ticker_rets) > 0:
            port_ret = np.mean(ticker_rets) - (COST_BPS / 10000)  # Deduct costs
            portfolio_returns.append({'date': dt, 'ret': port_ret, 'n_stocks': len(ticker_rets)})

        # SPY benchmark
        spy_data = period_data[period_data['ticker'] == 'SPY'] if 'SPY' in period_data['ticker'].values else None
        # Use the first qualifying ticker's date to get SPY return
        if 'spy_mom_1m' in day_data.columns:
            spy_returns.append({'date': dt, 'ret': day_data.iloc[0].get('spy_mom_1m', 0)})

    if len(portfolio_returns) == 0:
        logger.info("  No trades generated")
        return None

    ret_series = pd.Series([r['ret'] for r in portfolio_returns])
    n_stocks = [r['n_stocks'] for r in portfolio_returns]

    # Compute metrics
    mean_ret = ret_series.mean()
    std_ret = ret_series.std()
    sharpe = (mean_ret / std_ret * np.sqrt(12)) if std_ret > 0 else 0  # Annualized from monthly

    neg_ret = ret_series[ret_series < 0]
    downside_std = neg_ret.std() if len(neg_ret) > 1 else std_ret
    sortino = (mean_ret / downside_std * np.sqrt(12)) if downside_std > 0 else 0

    cum_ret = (1 + ret_series).cumprod()
    max_dd = ((cum_ret / cum_ret.cummax()) - 1).min()

    winners = (ret_series > 0).sum()
    wr = winners / len(ret_series)

    total_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0

    avg_gain = ret_series[ret_series > 0].mean() if (ret_series > 0).any() else 0
    avg_loss = abs(ret_series[ret_series < 0].mean()) if (ret_series < 0).any() else 0.001
    pf = avg_gain / avg_loss if avg_loss > 0 else 0

    result = {
        'n_periods': len(ret_series),
        'mean_monthly_ret': float(mean_ret),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'pf': float(pf),
        'wr': float(wr),
        'total_return': float(total_ret),
        'max_dd': float(max_dd),
        'avg_n_stocks': float(np.mean(n_stocks)),
        'monthly_returns': [float(r) for r in ret_series.tolist()],
    }

    logger.info("  Periods: %d, Sharpe: %.2f, Sortino: %.2f, PF: %.2f, WR: %.1f%%",
                result['n_periods'], sharpe, sortino, pf, wr*100)
    logger.info("  Total return: %.1f%%, MaxDD: %.1f%%, Avg stocks/period: %.1f",
                total_ret*100, max_dd*100, np.mean(n_stocks))

    return result


def find_regime_conditional_asymmetry(df):
    """
    Step 5: Find regime-conditional asymmetric setups.
    When does the market offer 3:1+ payoff ratios?
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Regime-Conditional Asymmetry")
    logger.info("="*60)

    results = {}

    # Define regimes
    regimes = {
        'vix_low': lambda d: d.get('vix', 20) < 15,
        'vix_medium': lambda d: (d.get('vix', 20) >= 15) & (d.get('vix', 20) < 25),
        'vix_high': lambda d: d.get('vix', 20) >= 25,
        'bull': lambda d: d.get('spy_above_200sma', 1) == 1,
        'bear': lambda d: d.get('spy_above_200sma', 1) == 0,
    }

    for regime_name, regime_fn in regimes.items():
        try:
            mask = regime_fn(df)
            regime_data = df[mask].dropna(subset=['fwd_ret_21d'])
        except (KeyError, TypeError):
            continue

        if len(regime_data) < 100:
            continue

        fwd = regime_data['fwd_ret_21d']
        big_gains = fwd[fwd > 0.05]
        big_losses = fwd[fwd < -0.05]

        result = {
            'n': int(len(regime_data)),
            'mean_ret': float(fwd.mean()),
            'median_ret': float(fwd.median()),
            'wr': float((fwd > 0).mean()),
            'p_big_gain': float(len(big_gains) / len(fwd)),
            'p_big_loss': float(len(big_losses) / len(fwd)),
            'asymmetry': float(len(big_gains) / max(len(big_losses), 1)),
            'skewness': float(fwd.skew()),
        }

        results[regime_name] = result

        logger.info("\n%s (N=%d):", regime_name, result['n'])
        logger.info("  Mean=%.2f%%, WR=%.1f%%, Skew=%.2f",
                    result['mean_ret']*100, result['wr']*100, result['skewness'])
        logger.info("  P(>5%%)=%.1f%%, P(<-5%%)=%.1f%%, Asymmetry=%.2f",
                    result['p_big_gain']*100, result['p_big_loss']*100, result['asymmetry'])

    return results


def main():
    logger.info("="*60)
    logger.info("Signal Combination Study v1 — Starting")
    logger.info("HC #725: Signal-first, find correlations before brute-forcing")
    logger.info("="*60)

    t0 = datetime.now()

    # Step 0: Download data
    close, volume, high, low = download_data()

    # Step 1: Compute all signals
    df = compute_signals(close, volume, high, low)

    # Save signal matrix
    df.to_parquet(OUTPUT_DIR / "signal_matrix.parquet", index=False)
    logger.info("Saved signal matrix: %d rows", len(df))

    # Step 2: Signal-return correlations
    corr_results = analyze_signal_correlations(df)

    # Step 3: Conditional return distributions
    cond_results = analyze_conditional_returns(df)

    # Step 4: Combined signal analysis
    combo_results = analyze_combined_signals(df)

    # Step 5: Regime-conditional asymmetry
    regime_results = find_regime_conditional_asymmetry(df)

    # Step 6: Walk-forward backtest of top combined signals
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Walk-Forward Backtests of Best Combined Signals")
    logger.info("="*60)

    backtest_configs = {
        'H1_highIV_posMom': lambda d: (d['iv_rank_proxy'] > 0.70) & (d['mom_3m'] > 0),
        'H5_deepValue_recovery': lambda d: (d['max_dd_252'] < -0.20) & (d['mom_1m'] > 0) & (d['mom_3m'] < 0),
        'H6_qualityMomentum': lambda d: (d['price_vs_sma200'] > 0) & (d['mom_3m'] > 0.10) & (d['vol_20d'] < 0.25),
        'H7_ivRank_nearLow_momentum': lambda d: (d['iv_rank_proxy'] > 0.60) & (d['dist_low'] < 0.20) & (d['mom_1m'] > 0),
    }

    backtest_results = {}
    for name, filter_fn in backtest_configs.items():
        bt = walk_forward_backtest(df, name, filter_fn)
        if bt is not None:
            backtest_results[name] = bt

    # Also test a "baseline" — random top 5 by momentum
    baseline_bt = walk_forward_backtest(df, "BASELINE_top5_mom", lambda d: pd.Series(True, index=d.index))
    if baseline_bt is not None:
        backtest_results['BASELINE_top5_mom'] = baseline_bt

    # Step 7: Summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)

    # Rank combined signals by asymmetry ratio
    logger.info("\nCombined signals ranked by asymmetry ratio:")
    sorted_combos = sorted(combo_results.items(), key=lambda x: x[1].get('asymmetry_ratio', 0), reverse=True)
    for name, data in sorted_combos:
        logger.info("  %-30s  asym=%.1f  mean=%.2f%%  WR=%.1f%%  N=%d",
                    name, data['asymmetry_ratio'], data['mean_ret_21d']*100,
                    data['wr']*100, data['n_observations'])

    # Rank backtests by Sharpe
    logger.info("\nBacktests ranked by Sharpe:")
    sorted_bt = sorted(backtest_results.items(), key=lambda x: x[1].get('sharpe', 0), reverse=True)
    for name, data in sorted_bt:
        logger.info("  %-30s  Sharpe=%.2f  Sortino=%.2f  PF=%.2f  WR=%.1f%%  TotRet=%.1f%%  MaxDD=%.1f%%",
                    name, data['sharpe'], data['sortino'], data['pf'],
                    data['wr']*100, data['total_return']*100, data['max_dd']*100)

    # Save all results
    all_results = {
        'experiment': 'signal_combination_v1',
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': (datetime.now() - t0).total_seconds(),
        'n_observations': len(df),
        'n_tickers': len(df['ticker'].unique()),
        'signal_correlations': {k: {sig: float(corr) for sig, corr in list(v.items())[:15]}
                               for k, v in corr_results.items()},
        'conditional_returns': cond_results,
        'combined_signals': combo_results,
        'regime_asymmetry': regime_results,
        'backtests': {k: {kk: vv for kk, vv in v.items() if kk != 'monthly_returns'}
                     for k, v in backtest_results.items()},
        'config': {
            'universe_size': len(UNIVERSE),
            'start_date': START_DATE,
            'end_date': END_DATE,
            'cost_bps': COST_BPS,
            'forward_horizons': FORWARD_HORIZONS,
            'asymmetry_ratio_min': ASYMMETRY_RATIO_MIN,
            'window_type': 'SLIDING (HC #0)',
            'signal_lag': 'T-1 (HC #724)',
        }
    }

    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\nSaved results.json")
    logger.info("Runtime: %.1f seconds", (datetime.now() - t0).total_seconds())
    logger.info("DONE")


if __name__ == '__main__':
    main()
