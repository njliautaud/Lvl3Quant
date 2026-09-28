#!/usr/bin/env python3
"""
Post-Earnings Announcement Drift (PEAD) Backtest — v1
=====================================================
Academic anomaly: stocks continue drifting in the earnings surprise
direction for 1-5 days after announcement. We measure "surprise" as
the overnight gap from prior close to post-earnings open.

Strategy:
  - After earnings (pre-market or after-close), measure gap vs prior close
  - If gap > threshold: buy at open (simulates buying calls), hold N days
  - If gap < -threshold: short at open (simulates buying puts), hold N days

Configurations tested:
  A) Gap threshold: 3%, 5%, 7%
  B) Hold period: 1, 2, 3, 5 days
  C) Direction: gap-following only

Quality gates (MANDATORY):
  - HC #428 R1: regime-agnostic (SPY green/red/flat, reject if gap > 0.50)
  - Permutation test: 100+ shuffles, p < 0.05
  - HC #704 adversarial: sub-period consistency, outlier removal, ticker concentration
  - Simulated as equity trades (no options pricing needed)

Data: yfinance daily OHLCV 2019-2026 for 30 large-cap stocks.
"""

import sys, json, warnings, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "pead_drift_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 10_000  # realistic for $440 account scaling analysis
COMMISSION_PER_TRADE = 0.0  # Robinhood equity is commission-free
SLIPPAGE_BPS = 5  # 5 bps slippage on open

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC',
    'BA', 'DIS', 'SBUX', 'HD', 'LOW', 'MCD', 'NKE', 'COST', 'WMT',
    'JPM', 'GS', 'BAC', 'MS', 'JNJ', 'PG', 'KO', 'UNH', 'ABBV', 'CRM', 'NOW',
]

# Earnings months (quarters): Jan/Feb, Apr/May, Jul/Aug, Oct/Nov
EARNINGS_MONTHS = {1, 2, 4, 5, 7, 8, 10, 11}


# ===============================================================================
# Data Download & Caching
# ===============================================================================

def download_prices(tickers, cache_path):
    """Download daily OHLCV from yfinance, cache to parquet."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Loaded cached prices: {len(df)} rows, {df['ticker'].nunique()} tickers")
        return df

    import yfinance as yf
    print(f"  Downloading prices for {len(tickers)} tickers from yfinance...")
    all_dfs = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start='2018-12-01', end='2026-07-15',
                               auto_adjust=True, progress=False)
            if len(data) > 0:
                data = data.reset_index()
                data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]
                data['ticker'] = ticker
                all_dfs.append(data[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']])
                print(f"    {ticker}: {len(data)} rows")
        except Exception as e:
            print(f"    {ticker}: FAILED - {e}")
        time.sleep(0.2)

    df = pd.concat(all_dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df.to_parquet(cache_path)
    print(f"  Saved {len(df)} rows to cache")
    return df


def download_spy(cache_path):
    """Download SPY for regime classification."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    import yfinance as yf
    print("  Downloading SPY for regime classification...")
    data = yf.download('SPY', start='2018-12-01', end='2026-07-15',
                       auto_adjust=True, progress=False)
    data = data.reset_index()
    data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]
    data = data[['date', 'close']].copy()
    data['date'] = pd.to_datetime(data['date'])
    data.to_parquet(cache_path)
    return data


# ===============================================================================
# Earnings Gap Detection
# ===============================================================================

def detect_earnings_gaps(prices_df, min_gap_pct=2.0):
    """
    Detect earnings gaps: large overnight gaps during earnings months.

    Logic: For each ticker, find days where |open/prev_close - 1| > min_gap_pct
    AND the day falls in an earnings month (Jan/Feb, Apr/May, Jul/Aug, Oct/Nov).

    Returns DataFrame of earnings events with gap info.
    """
    events = []

    for ticker in prices_df['ticker'].unique():
        tdf = prices_df[prices_df['ticker'] == ticker].sort_values('date').copy()
        tdf = tdf.reset_index(drop=True)

        if len(tdf) < 10:
            continue

        tdf['prev_close'] = tdf['close'].shift(1)
        tdf['gap_pct'] = (tdf['open'] / tdf['prev_close'] - 1) * 100
        tdf['month'] = tdf['date'].dt.month

        # Filter to earnings months and significant gaps
        mask = (
            tdf['month'].isin(EARNINGS_MONTHS) &
            (tdf['gap_pct'].abs() >= min_gap_pct) &
            tdf['prev_close'].notna()
        )

        gaps = tdf[mask].copy()

        # Deduplicate: if multiple gaps in same ticker within 5 days, keep largest
        if len(gaps) > 0:
            gaps = gaps.sort_values('date')
            keep = []
            last_date = None
            for _, row in gaps.iterrows():
                if last_date is None or (row['date'] - last_date).days > 5:
                    keep.append(row)
                    last_date = row['date']
                else:
                    # Keep larger gap
                    if abs(row['gap_pct']) > abs(keep[-1]['gap_pct']):
                        keep[-1] = row
                        last_date = row['date']

            for row in keep:
                events.append({
                    'ticker': ticker,
                    'date': row['date'],
                    'open': float(row['open']),
                    'prev_close': float(row['prev_close']),
                    'gap_pct': float(row['gap_pct']),
                    'gap_direction': 'up' if row['gap_pct'] > 0 else 'down',
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                })

    edf = pd.DataFrame(events)
    if len(edf) > 0:
        edf['date'] = pd.to_datetime(edf['date'])
        edf = edf[edf['date'] >= '2019-01-01'].copy()

    return edf


# ===============================================================================
# PEAD Backtest Engine
# ===============================================================================

def run_pead_backtest(events_df, prices_df, gap_threshold, hold_days):
    """
    Run PEAD backtest for a specific gap threshold and hold period.

    For each earnings gap event:
    - Entry: open price on gap day (we're buying AFTER the gap is known)
    - Exit: close price N trading days later
    - Direction: follow the gap (buy on positive gap, short on negative gap)
    - Slippage: SLIPPAGE_BPS on entry and exit
    """
    trades = []

    # Filter events by threshold
    qualified = events_df[events_df['gap_pct'].abs() >= gap_threshold].copy()

    for _, event in qualified.iterrows():
        ticker = event['ticker']
        entry_date = event['date']
        gap_pct = event['gap_pct']
        direction = 1 if gap_pct > 0 else -1  # 1 = long, -1 = short

        # Get future prices for this ticker
        tdf = prices_df[
            (prices_df['ticker'] == ticker) &
            (prices_df['date'] >= entry_date)
        ].sort_values('date').head(hold_days + 1)

        if len(tdf) < hold_days + 1:
            continue

        entry_price = float(tdf.iloc[0]['open'])
        exit_price = float(tdf.iloc[hold_days]['close'])

        if entry_price <= 0:
            continue

        # Apply slippage
        slippage_mult = SLIPPAGE_BPS / 10000
        if direction == 1:
            entry_price *= (1 + slippage_mult)  # buy higher
            exit_price *= (1 - slippage_mult)    # sell lower
        else:
            entry_price *= (1 - slippage_mult)  # short lower
            exit_price *= (1 + slippage_mult)    # cover higher

        # Return calculation
        raw_return = direction * (exit_price / entry_price - 1)

        # Track intra-trade MFE/MAE
        intra = tdf.iloc[1:hold_days+1]  # days after entry
        if direction == 1:
            mfe = float((intra['high'].max() / entry_price - 1) * 100) if len(intra) > 0 else 0
            mae = float((intra['low'].min() / entry_price - 1) * 100) if len(intra) > 0 else 0
        else:
            mfe = float((1 - intra['low'].min() / entry_price) * 100) if len(intra) > 0 else 0
            mae = float((1 - intra['high'].max() / entry_price) * 100) if len(intra) > 0 else 0

        # Day-0 close (same day as gap) for intraday drift
        day0_close = float(tdf.iloc[0]['close'])
        day0_drift = direction * (day0_close / entry_price - 1) * 100

        trades.append({
            'ticker': ticker,
            'entry_date': entry_date,
            'exit_date': tdf.iloc[hold_days]['date'],
            'gap_pct': round(gap_pct, 2),
            'gap_direction': 'up' if gap_pct > 0 else 'down',
            'direction': 'long' if direction == 1 else 'short',
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'return_pct': round(raw_return * 100, 4),
            'return_dollar': round(raw_return * STARTING_CAPITAL * 0.10, 2),  # 10% per trade
            'mfe_pct': round(mfe, 2),
            'mae_pct': round(mae, 2),
            'day0_drift_pct': round(day0_drift, 2),
            'hold_days': hold_days,
            'gap_threshold': gap_threshold,
        })

    return pd.DataFrame(trades)


# ===============================================================================
# Portfolio Simulation
# ===============================================================================

def simulate_portfolio(trades_df, starting_capital=STARTING_CAPITAL, risk_per_trade=0.10):
    """
    Simulate portfolio with fixed fractional sizing.
    Risk 10% of equity per trade (appropriate for $440 account).
    """
    if len(trades_df) == 0:
        return trades_df, pd.DataFrame()

    tdf = trades_df.sort_values('entry_date').copy()
    equity = starting_capital
    results = []
    equity_curve = []

    for _, trade in tdf.iterrows():
        position_size = equity * risk_per_trade
        trade_pnl = position_size * (trade['return_pct'] / 100)
        equity += trade_pnl

        t = trade.to_dict()
        t['position_size'] = round(position_size, 2)
        t['trade_pnl'] = round(trade_pnl, 2)
        t['equity_after'] = round(equity, 2)
        results.append(t)

        equity_curve.append({
            'date': trade['entry_date'],
            'equity': round(equity, 2),
        })

    return pd.DataFrame(results), pd.DataFrame(equity_curve)


# ===============================================================================
# Performance Metrics
# ===============================================================================

def compute_metrics(sized_df, equity_df, label="Strategy"):
    """Compute comprehensive performance metrics."""
    if len(sized_df) == 0:
        return {'label': label, 'n_trades': 0}

    returns = sized_df['trade_pnl'].values
    ret_pct = sized_df['return_pct'].values
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    wr = len(wins) / len(returns) if len(returns) > 0 else 0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0
    pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 999.0

    eq = equity_df.copy()
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.sort_values('date')
    years = max((eq['date'].iloc[-1] - eq['date'].iloc[0]).days / 365.25, 0.1)
    end_equity = float(eq['equity'].iloc[-1])
    cagr = (end_equity / STARTING_CAPITAL) ** (1 / years) - 1

    peak = eq['equity'].cummax()
    dd = (eq['equity'] - peak) / peak
    max_dd = float(dd.min())

    trades_per_year = len(returns) / years
    ann_factor = np.sqrt(trades_per_year) if trades_per_year > 0 else 1

    r = returns / STARTING_CAPITAL
    sharpe = float(r.mean() / r.std() * ann_factor) if r.std() > 0 else 0

    downside = r[r < 0]
    down_std = float(downside.std()) if len(downside) > 1 else 1e-9
    sortino = float(r.mean() / down_std * ann_factor) if down_std > 0 else 0

    # Average drift stats
    avg_return = float(ret_pct.mean())
    median_return = float(np.median(ret_pct))
    avg_gap = float(sized_df['gap_pct'].abs().mean())
    avg_mfe = float(sized_df['mfe_pct'].mean())
    avg_mae = float(sized_df['mae_pct'].mean())

    return {
        'label': label,
        'n_trades': int(len(returns)),
        'years': round(years, 1),
        'total_pnl': round(float(returns.sum()), 2),
        'cagr_pct': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd_pct': round(max_dd * 100, 2),
        'win_rate_pct': round(wr * 100, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(float(pf), 3) if pf != float('inf') else 999.0,
        'end_equity': round(end_equity, 2),
        'avg_return_pct': round(avg_return, 3),
        'median_return_pct': round(median_return, 3),
        'avg_gap_pct': round(avg_gap, 2),
        'avg_mfe_pct': round(avg_mfe, 2),
        'avg_mae_pct': round(avg_mae, 2),
        'trades_per_year': round(trades_per_year, 1),
    }


# ===============================================================================
# Regime Analysis (HC #428 R1)
# ===============================================================================

def regime_analysis(sized_df, spy_df):
    """Classify trades by SPY regime (30-day trailing return)."""
    spy = spy_df.copy()
    spy['date'] = pd.to_datetime(spy['date'])
    spy = spy.sort_values('date').set_index('date')
    spy['spy_ret_30'] = spy['close'].pct_change(30)

    def classify(date):
        try:
            d = pd.Timestamp(date)
            candidates = spy.index[spy.index <= d]
            if len(candidates) == 0:
                return 'unknown'
            r = float(spy.loc[candidates[-1], 'spy_ret_30'])
            if pd.isna(r):
                return 'unknown'
            if r > 0.03:
                return 'green'
            elif r < -0.03:
                return 'red'
            else:
                return 'flat'
        except:
            return 'unknown'

    df = sized_df.copy()
    df['regime'] = df['entry_date'].apply(classify)

    regime_sharpes = {}
    regime_results = {}
    for regime in ['green', 'red', 'flat', 'unknown']:
        sub = df[df['regime'] == regime]
        if len(sub) == 0:
            continue
        r = sub['trade_pnl'].values / STARTING_CAPITAL
        sharpe = float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else 0
        wr = float((sub['trade_pnl'] > 0).mean() * 100)
        pnl = float(sub['trade_pnl'].sum())
        avg_ret = float(sub['return_pct'].mean())
        regime_results[regime] = {
            'n': int(len(sub)), 'wr': round(wr, 1),
            'sharpe': round(sharpe, 3), 'total_pnl': round(pnl, 0),
            'avg_return_pct': round(avg_ret, 3),
        }
        if regime in ('green', 'red'):
            regime_sharpes[regime] = sharpe

    gap_test = 'N/A'
    gap_ratio = None
    if 'green' in regime_sharpes and 'red' in regime_sharpes:
        sg, sr = regime_sharpes['green'], regime_sharpes['red']
        denom = max(abs(sg), abs(sr))
        gap_ratio = abs(sg - sr) / denom if denom > 0 else 0
        gap_test = 'PASS' if gap_ratio <= 0.50 else 'REJECT'

    return df, {
        'regimes': regime_results,
        'gap_ratio': round(gap_ratio, 3) if gap_ratio is not None else None,
        'gap_test': gap_test,
    }


# ===============================================================================
# Permutation Test
# ===============================================================================

def permutation_test(sized_df, n_trials=200):
    """Sign-flip permutation test on trade P&Ls."""
    pnls = sized_df['trade_pnl'].values.copy()
    real_mean = float(pnls.mean())

    rng = np.random.default_rng(42)
    perm_means = np.array([
        float((np.abs(pnls) * rng.choice([-1, 1], size=len(pnls))).mean())
        for _ in range(n_trials)
    ])

    p_value = float((perm_means >= real_mean).mean())
    return {
        'real_mean_pnl': round(real_mean, 2),
        'perm_p95': round(float(np.percentile(perm_means, 95)), 2),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
    }


# ===============================================================================
# Adversarial Checks (HC #704)
# ===============================================================================

def adversarial_checks(sized_df):
    """
    HC #704 adversarial validation:
    1. Sub-period consistency (2019-2021, 2022-2024, 2025-2026)
    2. Outlier removal (top 5 trades removed)
    3. Ticker concentration
    4. Long vs short breakdown
    """
    results = {}
    df = sized_df.copy()
    df['year'] = pd.to_datetime(df['entry_date']).dt.year

    # 1. Sub-period consistency
    periods = {
        '2019-2021': df[df['year'].between(2019, 2021)],
        '2022-2024': df[df['year'].between(2022, 2024)],
        '2025-2026': df[df['year'].between(2025, 2026)],
    }
    period_stats = {}
    for pname, pdata in periods.items():
        if len(pdata) < 3:
            period_stats[pname] = {'n': len(pdata), 'wr': 0, 'avg_ret': 0, 'sharpe': 0}
            continue
        wr = float((pdata['trade_pnl'] > 0).mean() * 100)
        avg_ret = float(pdata['return_pct'].mean())
        r = pdata['trade_pnl'].values / STARTING_CAPITAL
        sharpe = float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else 0
        period_stats[pname] = {
            'n': int(len(pdata)), 'wr': round(wr, 1),
            'avg_ret_pct': round(avg_ret, 3), 'sharpe': round(sharpe, 3),
        }
    results['sub_periods'] = period_stats

    # Check consistency: all periods should be profitable
    profitable_periods = sum(1 for v in period_stats.values() if v.get('avg_ret_pct', v.get('avg_ret', 0)) > 0)
    results['sub_period_consistent'] = profitable_periods == len([v for v in period_stats.values() if v['n'] >= 3])

    # 2. Outlier removal (remove top 5 trades by absolute P&L)
    if len(df) > 10:
        df_sorted = df.reindex(df['trade_pnl'].abs().sort_values(ascending=False).index)
        df_no_outliers = df_sorted.iloc[5:]  # remove top 5
        wr_no = float((df_no_outliers['trade_pnl'] > 0).mean() * 100)
        avg_ret_no = float(df_no_outliers['return_pct'].mean())
        r_no = df_no_outliers['trade_pnl'].values / STARTING_CAPITAL
        sharpe_no = float(r_no.mean() / r_no.std() * np.sqrt(len(r_no))) if r_no.std() > 0 else 0
        results['outlier_removal'] = {
            'n_removed': 5,
            'n_remaining': len(df_no_outliers),
            'wr_pct': round(wr_no, 1),
            'avg_ret_pct': round(avg_ret_no, 3),
            'sharpe': round(sharpe_no, 3),
            'still_profitable': avg_ret_no > 0,
        }

    # 3. Ticker concentration
    ticker_pnl = df.groupby('ticker')['trade_pnl'].sum()
    total_pnl = ticker_pnl.sum()
    if total_pnl > 0:
        top_ticker_pct = float(ticker_pnl.max() / total_pnl * 100) if total_pnl > 0 else 0
    else:
        top_ticker_pct = float(ticker_pnl.min() / total_pnl * 100) if total_pnl < 0 else 0
    results['ticker_concentration'] = {
        'top_ticker': str(ticker_pnl.abs().idxmax()) if len(ticker_pnl) > 0 else 'N/A',
        'top_ticker_pnl_pct': round(abs(top_ticker_pct), 1),
        'n_tickers_traded': int(df['ticker'].nunique()),
        'concentrated': abs(top_ticker_pct) > 40,
    }

    # 4. Long vs short breakdown
    for side in ['long', 'short']:
        sub = df[df['direction'] == side]
        if len(sub) > 0:
            wr = float((sub['trade_pnl'] > 0).mean() * 100)
            avg_ret = float(sub['return_pct'].mean())
            results[f'{side}_side'] = {
                'n': int(len(sub)),
                'wr_pct': round(wr, 1),
                'avg_ret_pct': round(avg_ret, 3),
                'total_pnl': round(float(sub['trade_pnl'].sum()), 2),
            }

    return results


# ===============================================================================
# Gap Size Analysis
# ===============================================================================

def gap_size_analysis(sized_df):
    """Analyze drift by gap size buckets."""
    df = sized_df.copy()
    df['abs_gap'] = df['gap_pct'].abs()

    buckets = [
        ('3-5%', 3, 5),
        ('5-7%', 5, 7),
        ('7-10%', 7, 10),
        ('10-15%', 10, 15),
        ('15%+', 15, 100),
    ]

    results = {}
    for bname, lo, hi in buckets:
        sub = df[(df['abs_gap'] >= lo) & (df['abs_gap'] < hi)]
        if len(sub) < 3:
            continue
        wr = float((sub['trade_pnl'] > 0).mean() * 100)
        avg_ret = float(sub['return_pct'].mean())
        avg_drift = float(sub['return_pct'].mean())
        results[bname] = {
            'n': int(len(sub)),
            'wr_pct': round(wr, 1),
            'avg_drift_pct': round(avg_drift, 3),
            'avg_mfe_pct': round(float(sub['mfe_pct'].mean()), 2),
            'avg_mae_pct': round(float(sub['mae_pct'].mean()), 2),
        }

    return results


# ===============================================================================
# Per-Ticker Breakdown
# ===============================================================================

def per_ticker_stats(sized_df):
    """Compute per-ticker performance summary."""
    stats = sized_df.groupby('ticker').agg(
        n_trades=('trade_pnl', 'count'),
        total_pnl=('trade_pnl', 'sum'),
        win_rate=('trade_pnl', lambda x: round((x > 0).mean() * 100, 1)),
        avg_return=('return_pct', 'mean'),
        avg_gap=('gap_pct', lambda x: round(x.abs().mean(), 1)),
        avg_mfe=('mfe_pct', 'mean'),
        n_long=('direction', lambda x: (x == 'long').sum()),
        n_short=('direction', lambda x: (x == 'short').sum()),
    ).round(3).sort_values('total_pnl', ascending=False)
    return stats


# ===============================================================================
# Year-by-Year
# ===============================================================================

def year_by_year(sized_df):
    df = sized_df.copy()
    df['year'] = pd.to_datetime(df['entry_date']).dt.year
    results = {}
    for yr, grp in sorted(df.groupby('year')):
        n = len(grp)
        wr = float((grp['trade_pnl'] > 0).mean() * 100)
        w = grp[grp['trade_pnl'] > 0]['trade_pnl']
        l = grp[grp['trade_pnl'] < 0]['trade_pnl']
        pf = abs(float(w.sum()) / float(l.sum())) if l.sum() != 0 else 999.0
        yr_pnl = float(grp['trade_pnl'].sum())
        r = grp['trade_pnl'].values / STARTING_CAPITAL
        sh = float(r.mean() / r.std() * np.sqrt(n)) if r.std() > 0 else 0
        avg_ret = float(grp['return_pct'].mean())
        results[int(yr)] = {
            'n': int(n), 'wr': round(wr, 1), 'pf': round(pf, 2),
            'sharpe': round(sh, 3), 'total_pnl': round(yr_pnl, 0),
            'avg_ret_pct': round(avg_ret, 3),
        }
    return results


# ===============================================================================
# Main
# ===============================================================================

def main():
    print("=" * 70)
    print("POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD) BACKTEST -- v1")
    print("=" * 70)

    # ── Step 1: Load data ──
    print("\nStep 1: Loading price data...")
    prices_cache = OUTPUT / "prices_cache.parquet"
    spy_cache = OUTPUT / "spy_cache.parquet"

    prices_df = download_prices(TICKERS, prices_cache)
    spy_df = download_spy(spy_cache)

    print(f"  Universe: {prices_df['ticker'].nunique()} tickers, "
          f"{prices_df['date'].min().date()} to {prices_df['date'].max().date()}")

    # ── Step 2: Detect earnings gaps ──
    print("\nStep 2: Detecting earnings gaps (>2% overnight gap in earnings months)...")
    events_df = detect_earnings_gaps(prices_df, min_gap_pct=2.0)
    print(f"  Found {len(events_df)} earnings gap events across {events_df['ticker'].nunique()} tickers")

    # Distribution
    for thresh in [3, 5, 7, 10]:
        n = (events_df['gap_pct'].abs() >= thresh).sum()
        n_up = ((events_df['gap_pct'] >= thresh)).sum()
        n_dn = ((events_df['gap_pct'] <= -thresh)).sum()
        print(f"    |gap| >= {thresh}%: {n} events ({n_up} up, {n_dn} down)")

    events_df.to_csv(OUTPUT / "earnings_events.csv", index=False)

    # ── Step 3: Run configuration grid ──
    print("\nStep 3: Running PEAD backtest grid...")

    gap_thresholds = [3, 5, 7]
    hold_periods = [1, 2, 3, 5]

    all_results = {}
    best_sharpe = -999
    best_config = None

    for gap_thresh in gap_thresholds:
        for hold_days in hold_periods:
            label = f"gap{gap_thresh}pct_hold{hold_days}d"

            trades_df = run_pead_backtest(events_df, prices_df, gap_thresh, hold_days)

            if len(trades_df) < 10:
                print(f"  {label}: only {len(trades_df)} trades (skipped)")
                continue

            sized_df, equity_df = simulate_portfolio(trades_df)
            metrics = compute_metrics(sized_df, equity_df, label=label)
            perm = permutation_test(sized_df, n_trials=200)
            sized_df_regime, regime = regime_analysis(sized_df, spy_df)
            adversarial = adversarial_checks(sized_df)
            gap_analysis = gap_size_analysis(sized_df)
            yby = year_by_year(sized_df)

            result = {
                'config': {'gap_threshold': gap_thresh, 'hold_days': hold_days},
                'metrics': metrics,
                'permutation': perm,
                'regime': regime,
                'adversarial': adversarial,
                'gap_analysis': gap_analysis,
                'year_by_year': yby,
            }
            all_results[label] = result

            sh = metrics.get('sharpe', 0)
            wr = metrics.get('win_rate_pct', 0)
            n = metrics.get('n_trades', 0)
            pv = perm.get('p_value', 1)
            gt = regime.get('gap_test', 'N/A')
            avg_ret = metrics.get('avg_return_pct', 0)

            flag = ""
            if sh > best_sharpe and n >= 20:
                best_sharpe = sh
                best_config = label
                flag = " <-- BEST"

            print(f"  {label}: n={n} Sharpe={sh:.3f} Sortino={metrics.get('sortino',0):.3f} "
                  f"WR={wr:.1f}% PF={metrics.get('profit_factor',0):.2f} "
                  f"AvgRet={avg_ret:.3f}% p={pv:.4f} regime={gt}{flag}")

    # ── Step 4: Save all results ──
    print(f"\nStep 4: Saving results...")

    def make_serializable(obj):
        if isinstance(obj, (pd.Timestamp, np.datetime64)):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        return obj

    with open(OUTPUT / "all_configs_results.json", 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)

    # ── Step 5: Summary table ──
    print("\n" + "=" * 100)
    print("CONFIGURATION COMPARISON -- SORTED BY SHARPE")
    print("=" * 100)

    summary_rows = []
    for label, res in all_results.items():
        m = res['metrics']
        p = res['permutation']
        r = res['regime']
        summary_rows.append({
            'config': label,
            'n': m.get('n_trades', 0),
            'sharpe': m.get('sharpe', 0),
            'sortino': m.get('sortino', 0),
            'cagr_pct': m.get('cagr_pct', 0),
            'max_dd': m.get('max_dd_pct', 0),
            'wr_pct': m.get('win_rate_pct', 0),
            'pf': m.get('profit_factor', 0),
            'avg_ret': m.get('avg_return_pct', 0),
            'med_ret': m.get('median_return_pct', 0),
            'p_val': p.get('p_value', 1),
            'regime': r.get('gap_test', 'N/A'),
            'regime_gap': r.get('gap_ratio', None),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values('sharpe', ascending=False)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT / "config_comparison.csv", index=False)

    # ── Step 6: Deep dive on best config ──
    if best_config and best_config in all_results:
        print(f"\n{'=' * 70}")
        print(f"DEEP DIVE -- BEST CONFIG: {best_config}")
        print(f"{'=' * 70}")

        best = all_results[best_config]
        m = best['metrics']

        print(f"\n  CORE METRICS:")
        print(f"  Trades:           {m['n_trades']} ({m['trades_per_year']}/year)")
        print(f"  CAGR:             {m['cagr_pct']}%")
        print(f"  Sharpe:           {m['sharpe']}")
        print(f"  Sortino:          {m['sortino']}")
        print(f"  Max Drawdown:     {m['max_dd_pct']}%")
        print(f"  Win Rate:         {m['win_rate_pct']}%")
        print(f"  Profit Factor:    {m['profit_factor']}")
        print(f"  Avg Return:       {m['avg_return_pct']}% per trade")
        print(f"  Median Return:    {m['median_return_pct']}% per trade")
        print(f"  Avg MFE:          {m['avg_mfe_pct']}%")
        print(f"  Avg MAE:          {m['avg_mae_pct']}%")
        print(f"  Total P&L:        ${m['total_pnl']:,.0f} on ${STARTING_CAPITAL:,}")
        print(f"  End Equity:       ${m['end_equity']:,.0f}")

        print(f"\n  PERMUTATION TEST:")
        p = best['permutation']
        print(f"  p-value:          {p['p_value']:.4f} ({'SIGNIFICANT' if p['significant'] else 'NOT SIGNIFICANT'})")
        print(f"  Real mean P&L:    ${p['real_mean_pnl']:.2f}")
        print(f"  Perm 95th pctl:   ${p['perm_p95']:.2f}")

        print(f"\n  REGIME ANALYSIS:")
        for reg, stats in best['regime']['regimes'].items():
            print(f"    {reg.upper():8s}: n={stats['n']:4d}  WR={stats['wr']:.1f}%  "
                  f"Sharpe={stats['sharpe']:.3f}  AvgRet={stats['avg_return_pct']:.3f}%  "
                  f"PnL=${stats['total_pnl']:,.0f}")
        if best['regime']['gap_ratio'] is not None:
            print(f"    Gap ratio: {best['regime']['gap_ratio']:.3f} [{best['regime']['gap_test']}]")

        print(f"\n  ADVERSARIAL CHECKS:")
        adv = best['adversarial']

        print(f"    Sub-period consistency: {'PASS' if adv.get('sub_period_consistent', False) else 'FAIL'}")
        for pname, pstats in adv.get('sub_periods', {}).items():
            print(f"      {pname}: n={pstats['n']}  WR={pstats.get('wr',0):.1f}%  "
                  f"AvgRet={pstats.get('avg_ret_pct', pstats.get('avg_ret',0)):.3f}%  "
                  f"Sharpe={pstats['sharpe']:.3f}")

        if 'outlier_removal' in adv:
            o = adv['outlier_removal']
            print(f"    Outlier removal (top 5 removed): "
                  f"WR={o['wr_pct']:.1f}%  AvgRet={o['avg_ret_pct']:.3f}%  "
                  f"Sharpe={o['sharpe']:.3f}  "
                  f"{'PASS' if o['still_profitable'] else 'FAIL'}")

        tc = adv.get('ticker_concentration', {})
        print(f"    Ticker concentration: top={tc.get('top_ticker','N/A')} "
              f"({tc.get('top_ticker_pnl_pct',0):.1f}% of PnL)  "
              f"{'CONCENTRATED' if tc.get('concentrated', False) else 'DIVERSIFIED'}")

        for side in ['long', 'short']:
            s = adv.get(f'{side}_side', {})
            if s:
                print(f"    {side.upper()} side: n={s['n']}  WR={s['wr_pct']:.1f}%  "
                      f"AvgRet={s['avg_ret_pct']:.3f}%  PnL=${s['total_pnl']:,.0f}")

        print(f"\n  GAP SIZE ANALYSIS (avg drift by gap magnitude):")
        for bname, bstats in best.get('gap_analysis', {}).items():
            print(f"    {bname:8s}: n={bstats['n']:3d}  WR={bstats['wr_pct']:.1f}%  "
                  f"AvgDrift={bstats['avg_drift_pct']:.3f}%  "
                  f"MFE={bstats['avg_mfe_pct']:.2f}%  MAE={bstats['avg_mae_pct']:.2f}%")

        print(f"\n  YEAR-BY-YEAR:")
        for yr, stats in sorted(best['year_by_year'].items()):
            print(f"    {yr}: n={stats['n']:3d}  WR={stats['wr']:.1f}%  "
                  f"Sharpe={stats['sharpe']:.3f}  AvgRet={stats['avg_ret_pct']:.3f}%  "
                  f"PnL=${stats['total_pnl']:,.0f}")

        # Per-ticker breakdown
        cfg = best['config']
        trades_df = run_pead_backtest(events_df, prices_df, cfg['gap_threshold'], cfg['hold_days'])
        sized_df, equity_df = simulate_portfolio(trades_df)

        ticker_stats = per_ticker_stats(sized_df)
        print(f"\n  PER-TICKER BREAKDOWN:")
        print(ticker_stats.to_string())
        ticker_stats.to_csv(OUTPUT / "best_config_per_ticker.csv")
        sized_df.to_csv(OUTPUT / "best_config_trades.csv", index=False)
        equity_df.to_csv(OUTPUT / "best_config_equity_curve.csv", index=False)

    # ── Step 7: Quality gate summary ──
    print(f"\n{'=' * 70}")
    print("QUALITY GATE SUMMARY")
    print(f"{'=' * 70}")

    passing = []
    for label, res in all_results.items():
        m = res['metrics']
        p = res['permutation']
        r = res['regime']
        adv = res['adversarial']

        passes_perm = p.get('significant', False)
        passes_regime = r.get('gap_test', 'REJECT') in ('PASS', 'N/A')
        passes_sharpe = m.get('sharpe', 0) > 0
        passes_subperiod = adv.get('sub_period_consistent', False)
        passes_outlier = adv.get('outlier_removal', {}).get('still_profitable', False)
        passes_concentration = not adv.get('ticker_concentration', {}).get('concentrated', True)

        n_pass = sum([passes_perm, passes_regime, passes_sharpe, passes_subperiod,
                      passes_outlier, passes_concentration])

        status = {
            'config': label,
            'sharpe': m.get('sharpe', 0),
            'sortino': m.get('sortino', 0),
            'wr': m.get('win_rate_pct', 0),
            'pf': m.get('profit_factor', 0),
            'avg_ret': m.get('avg_return_pct', 0),
            'p_value': p.get('p_value', 1),
            'regime': r.get('gap_test', 'N/A'),
            'sub_period': 'PASS' if passes_subperiod else 'FAIL',
            'outlier': 'PASS' if passes_outlier else 'FAIL',
            'concentration': 'PASS' if passes_concentration else 'FAIL',
            'gates_passed': f"{n_pass}/6",
        }

        if n_pass >= 5:
            passing.append(status)

        print(f"  {label}: Sharpe={status['sharpe']:.3f} WR={status['wr']:.1f}% "
              f"p={status['p_value']:.4f} regime={status['regime']} "
              f"subperiod={status['sub_period']} outlier={status['outlier']} "
              f"conc={status['concentration']} => {status['gates_passed']}")

    if passing:
        print(f"\n  {len(passing)} configs pass 5+/6 quality gates:")
        for p in sorted(passing, key=lambda x: -x['sharpe']):
            print(f"    {p['config']}: Sharpe={p['sharpe']:.3f} WR={p['wr']:.1f}% "
                  f"AvgRet={p['avg_ret']:.3f}%")
    else:
        print("\n  NO configs pass 5+/6 quality gates.")

    # ── Step 8: $440 Account Projection ──
    print(f"\n{'=' * 70}")
    print("$440 ROBINHOOD ACCOUNT PROJECTION")
    print(f"{'=' * 70}")

    if best_config and best_config in all_results:
        m = all_results[best_config]['metrics']
        avg_ret_per_trade = m['avg_return_pct']
        trades_per_year = m['trades_per_year']
        wr = m['win_rate_pct']

        print(f"\n  Strategy: {best_config}")
        print(f"  Avg return per trade: {avg_ret_per_trade:.3f}%")
        print(f"  Trades per year: {trades_per_year:.0f}")
        print(f"  Win rate: {wr:.1f}%")

        # With options leverage (buying calls/puts at ~$1-5 each)
        print(f"\n  EQUITY SIMULATION ($440 capital, 10% risk per trade):")
        capital = 440
        for yr in range(1, 4):
            n_trades = int(trades_per_year)
            for _ in range(n_trades):
                trade_size = capital * 0.10
                capital += trade_size * (avg_ret_per_trade / 100)
            print(f"    Year {yr}: ${capital:,.0f}")

        print(f"\n  OPTIONS LEVERAGE SCENARIO (buying $1-3 options, 3x-5x stock move):")
        capital = 440
        for mult_label, mult in [('3x leverage', 3), ('5x leverage', 5)]:
            cap = 440
            for yr in range(1, 4):
                n_trades = int(trades_per_year)
                for _ in range(n_trades):
                    trade_size = cap * 0.10
                    option_ret = avg_ret_per_trade * mult / 100
                    # With options: can lose 100% of premium, but gain is leveraged
                    if option_ret > 0:
                        cap += trade_size * option_ret
                    else:
                        cap += trade_size * max(option_ret, -1.0)  # max loss = 100% of premium
                print(f"    {mult_label} Year {yr}: ${cap:,.0f}")
            print()

    print(f"\nAll outputs saved to {OUTPUT}/")
    print("DONE.")


if __name__ == "__main__":
    main()
