#!/usr/bin/env python3
"""
HYBRID MOMENTUM — Stock-within-Sector + Dual Momentum + Dynamic Exits
======================================================================
Combines three proven strategies:

1. SECTOR ROTATION + STOCK SELECTION:
   Top 3 sectors by 6-month momentum → top 5 stocks per sector (15 positions)

2. DUAL MOMENTUM (absolute momentum filter):
   Only invest when SPY > 200-day SMA. Below → 100% cash (SHY proxy).

3. DYNAMIC EXITS (from breakout/MR research):
   - Trailing stop at 2x ATR
   - Momentum breakdown (10-day momentum flips negative → exit)
   - Sector exit: if sector drops out of top 4 → exit all positions in that sector
   - Position stop-loss: -15% from entry

Additional features:
   - Monthly rebalance for universe, DAILY position health checks
   - Walk-forward: 60-month train, 12-month test, sliding (8+ windows)
   - Multi-config comparison
   - Regime analysis + permutation test

Sliding window ONLY. Slippage: 0.3% per trade (Robinhood, no commission).
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from itertools import product as iterproduct
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/hybrid_momentum'
CACHE_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/cache'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

###############################################################################
# SECTOR ETF → GICS MAPPING
###############################################################################

SECTOR_ETFS = {
    'XLK': 'Information Technology',
    'XLV': 'Health Care',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLC': 'Communication Services',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLB': 'Materials',
}

ETF_FOR_SECTOR = {v: k for k, v in SECTOR_ETFS.items()}

# SHY proxy return (annualized ~4% in cash when out of market)
SHY_DAILY_RETURN = 0.04 / 252

###############################################################################
# S&P 500 SECTOR MAPPING
###############################################################################

def get_sp500_with_sectors():
    """Get S&P 500 tickers with GICS sector. Try Wikipedia, fallback to hardcoded."""
    try:
        tables = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            attrs={'id': 'constituents'}
        )
        df = tables[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        sector_col = [c for c in df.columns if 'sector' in c.lower()][0]
        mapping = dict(zip(df['Symbol'], df[sector_col]))
        print(f"  Got {len(mapping)} stocks from Wikipedia")
        return mapping
    except Exception as e:
        print(f"  Wikipedia fetch failed ({e}), using hardcoded mapping")
        return _hardcoded_sector_map()


def _hardcoded_sector_map():
    """Fallback: ~260 stocks with sector assignments."""
    m = {}
    for t in ['AAPL','MSFT','NVDA','AVGO','CSCO','ACN','TXN','QCOM','INTC','ADI',
              'AMAT','MU','LRCX','KLAC','SNPS','CDNS','MCHP','ON','FTNT','PANW',
              'CRWD','ADBE','CRM','NOW','INTU','ADP','ORCL','IBM','FIS','FISV',
              'GPN','IT','MPWR','NXPI','SWKS','HPQ','HPE','KEYS','ZBRA','EPAM']:
        m[t] = 'Information Technology'
    for t in ['UNH','JNJ','LLY','MRK','ABBV','TMO','ABT','DHR','BMY','AMGN',
              'GILD','ISRG','SYK','VRTX','REGN','ZTS','CI','BDX','HUM','ELV',
              'MDT','BSX','EW','A','IQV','DXCM','IDXX','MTD','ALGN','HOLX']:
        m[t] = 'Health Care'
    for t in ['JPM','V','MA','BRK-B','BAC','WFC','GS','MS','BLK','SCHW',
              'CB','MMC','CME','ICE','PNC','USB','AIG','AFL','MET','PRU',
              'MSCI','TRV','ALL','SPGI','MCO','AON','AJG','C','COF','TFC']:
        m[t] = 'Financials'
    for t in ['XOM','CVX','COP','SLB','EOG','PSX','VLO','OXY','MPC','PXD',
              'WMB','KMI','DVN','HAL','FANG','HES','BKR','OKE']:
        m[t] = 'Energy'
    for t in ['UNP','RTX','HON','CAT','BA','DE','GE','LMT','GD','NOC',
              'FDX','WM','ITW','EMR','NSC','CSX','MMM','JCI','ETN','PH',
              'ROK','PCAR','FAST','CTAS','CARR','OTIS','AME','TT','IR','SWK']:
        m[t] = 'Industrials'
    for t in ['GOOGL','GOOG','META','NFLX','DIS','CMCSA','VZ','T','TMUS',
              'CHTR','EA','ATVI','TTWO','WBD','MTCH','OMC','IPG','LYV']:
        m[t] = 'Communication Services'
    for t in ['AMZN','TSLA','HD','MCD','LOW','BKNG','TJX','SBUX','NKE','CMG',
              'ORLY','AZO','ROST','MAR','HLT','DHI','LEN','GM','F','EBAY',
              'YUM','DPZ','POOL','BBY']:
        m[t] = 'Consumer Discretionary'
    for t in ['PG','PEP','KO','COST','WMT','PM','MDLZ','CL','MO','EL',
              'SYY','GIS','KMB','HSY','K','ADM','STZ','KHC','TSN','MKC']:
        m[t] = 'Consumer Staples'
    for t in ['NEE','SO','DUK','D','AEP','EXC','SRE','XEL','WEC','ED',
              'AWK','DTE','ETR','FE','PPL','AES','CMS','CEG']:
        m[t] = 'Utilities'
    for t in ['PLD','AMT','CCI','EQIX','SPG','WELL','PSA','O','DLR','VICI',
              'SBAC','ARE','AVB','EQR','MAA','ESS','UDR','INVH']:
        m[t] = 'Real Estate'
    for t in ['LIN','APD','SHW','FCX','NEM','ECL','DOW','DD','NUE','PPG',
              'VMC','MLM','ALB','IFF','CE','EMN','BALL','PKG','IP','CF','MOS']:
        m[t] = 'Materials'
    return m


###############################################################################
# DATA ACQUISITION (with caching)
###############################################################################

def _cache_path(name):
    return os.path.join(CACHE_DIR, f'{name}.parquet')


def download_all_data(sector_map, start='2012-01-01', end='2026-07-13'):
    """Download sector ETFs + SPY + SHY + individual stock prices. Caches to disk."""
    import yfinance as yf

    etf_cache = _cache_path('hybrid_etf_prices')
    stock_cache = _cache_path('hybrid_stock_prices')

    use_cache = True
    for cp in [etf_cache, stock_cache]:
        if not os.path.exists(cp):
            use_cache = False
            break
        age_hours = (time.time() - os.path.getmtime(cp)) / 3600
        if age_hours > 12:
            use_cache = False
            break

    if use_cache:
        print("  Using cached data (< 12 hours old)")
        etf_df = pd.read_parquet(etf_cache)
        stock_df = pd.read_parquet(stock_cache)
        print(f"  ETFs: {etf_df.shape[1]} tickers, {etf_df.shape[0]} days")
        print(f"  Stocks: {stock_df.shape[1]} tickers, {stock_df.shape[0]} days")
        return etf_df, stock_df

    # Download ETFs + SPY + SHY
    etf_tickers = list(SECTOR_ETFS.keys()) + ['SPY', 'SHY']
    print(f"  Downloading {len(etf_tickers)} sector ETFs + SPY + SHY...")
    etf_str = ' '.join(etf_tickers)
    etf_data = yf.download(etf_str, start=start, end=end, progress=False,
                           group_by='ticker', threads=True)

    etf_prices = {}
    for t in etf_tickers:
        try:
            if isinstance(etf_data.columns, pd.MultiIndex):
                col = etf_data[t]['Close'].dropna()
            else:
                col = etf_data['Close'].dropna()
            if len(col) > 200:
                # Flatten MultiIndex if needed
                if isinstance(col.index, pd.MultiIndex):
                    col.index = col.index.get_level_values(0)
                etf_prices[t] = col
        except:
            pass
    etf_df = pd.DataFrame(etf_prices)
    etf_df.index = pd.to_datetime(etf_df.index)

    # Retry missing ETFs individually
    for t in etf_tickers:
        if t not in etf_df.columns:
            try:
                time.sleep(1)
                d = yf.download(t, start=start, end=end, progress=False)
                if isinstance(d.columns, pd.MultiIndex):
                    col = d['Close'].iloc[:, 0].dropna()
                else:
                    col = d['Close'].dropna()
                if len(col) > 200:
                    etf_df[t] = col
                    print(f"    Recovered {t}")
            except:
                print(f"    FAILED to recover {t}")

    print(f"  Got {etf_df.shape[1]} ETFs, {etf_df.shape[0]} days")

    # Download individual stocks
    stock_tickers = list(sector_map.keys())
    print(f"  Downloading {len(stock_tickers)} stocks in batches...")
    all_stock_prices = {}
    batch_size = 40
    for i in range(0, len(stock_tickers), batch_size):
        batch = stock_tickers[i:i+batch_size]
        ticker_str = ' '.join(batch)
        try:
            data = yf.download(ticker_str, start=start, end=end, progress=False,
                               group_by='ticker', threads=True)
            for t in batch:
                try:
                    if len(batch) > 1 and isinstance(data.columns, pd.MultiIndex):
                        col = data[t]['Close'].dropna()
                    else:
                        col = data['Close'].dropna()
                    if isinstance(col.index, pd.MultiIndex):
                        col.index = col.index.get_level_values(0)
                    if len(col) > 252:
                        all_stock_prices[t] = col
                except:
                    pass
        except Exception as e:
            print(f"    Batch {i//batch_size} error: {e}")
        time.sleep(0.5)

    stock_df = pd.DataFrame(all_stock_prices)
    stock_df.index = pd.to_datetime(stock_df.index)
    print(f"  Got {stock_df.shape[1]} stocks, {stock_df.shape[0]} days")

    # Also download volume data for volume confirmation
    vol_cache = _cache_path('hybrid_stock_volumes')
    print(f"  Downloading volume data...")
    all_stock_vols = {}
    for i in range(0, len(stock_tickers), batch_size):
        batch = stock_tickers[i:i+batch_size]
        ticker_str = ' '.join(batch)
        try:
            data = yf.download(ticker_str, start=start, end=end, progress=False,
                               group_by='ticker', threads=True)
            for t in batch:
                try:
                    if len(batch) > 1 and isinstance(data.columns, pd.MultiIndex):
                        col = data[t]['Volume'].dropna()
                    else:
                        col = data['Volume'].dropna()
                    if isinstance(col.index, pd.MultiIndex):
                        col.index = col.index.get_level_values(0)
                    if len(col) > 252:
                        all_stock_vols[t] = col
                except:
                    pass
        except:
            pass
        time.sleep(0.3)

    vol_df = pd.DataFrame(all_stock_vols)
    vol_df.index = pd.to_datetime(vol_df.index)

    # Also get High/Low for ATR calculation
    hl_cache = _cache_path('hybrid_stock_highlow')
    print(f"  Downloading high/low data for ATR...")
    all_highs = {}
    all_lows = {}
    for i in range(0, len(stock_tickers), batch_size):
        batch = stock_tickers[i:i+batch_size]
        ticker_str = ' '.join(batch)
        try:
            data = yf.download(ticker_str, start=start, end=end, progress=False,
                               group_by='ticker', threads=True)
            for t in batch:
                try:
                    if len(batch) > 1 and isinstance(data.columns, pd.MultiIndex):
                        h = data[t]['High'].dropna()
                        l = data[t]['Low'].dropna()
                    else:
                        h = data['High'].dropna()
                        l = data['Low'].dropna()
                    if isinstance(h.index, pd.MultiIndex):
                        h.index = h.index.get_level_values(0)
                        l.index = l.index.get_level_values(0)
                    if len(h) > 252:
                        all_highs[t] = h
                        all_lows[t] = l
                except:
                    pass
        except:
            pass
        time.sleep(0.3)

    high_df = pd.DataFrame(all_highs)
    high_df.index = pd.to_datetime(high_df.index)
    low_df = pd.DataFrame(all_lows)
    low_df.index = pd.to_datetime(low_df.index)

    # Cache
    try:
        etf_df.to_parquet(etf_cache)
        stock_df.to_parquet(stock_cache)
        vol_df.to_parquet(vol_cache)
        high_df.to_parquet(hl_cache.replace('.parquet', '_high.parquet'))
        low_df.to_parquet(hl_cache.replace('.parquet', '_low.parquet'))
        print("  Cached all data to disk")
    except Exception as e:
        print(f"  Cache write failed: {e}")

    return etf_df, stock_df, vol_df, high_df, low_df


def load_supplementary_data():
    """Load cached volume and high/low data if available."""
    vol_cache = _cache_path('hybrid_stock_volumes')
    hl_cache = _cache_path('hybrid_stock_highlow')

    vol_df = None
    high_df = None
    low_df = None

    if os.path.exists(vol_cache):
        vol_df = pd.read_parquet(vol_cache)
    if os.path.exists(hl_cache.replace('.parquet', '_high.parquet')):
        high_df = pd.read_parquet(hl_cache.replace('.parquet', '_high.parquet'))
    if os.path.exists(hl_cache.replace('.parquet', '_low.parquet')):
        low_df = pd.read_parquet(hl_cache.replace('.parquet', '_low.parquet'))

    return vol_df, high_df, low_df


###############################################################################
# ATR CALCULATION
###############################################################################

def compute_atr(close_df, high_df, low_df, period=14):
    """Compute ATR for each stock. Returns DataFrame of ATR values."""
    atr_dict = {}
    common_cols = set(close_df.columns) & set(high_df.columns) & set(low_df.columns)

    for t in common_cols:
        c = close_df[t].dropna()
        h = high_df[t].reindex(c.index).dropna()
        lo = low_df[t].reindex(c.index).dropna()

        # Align
        common_idx = c.index.intersection(h.index).intersection(lo.index)
        if len(common_idx) < period + 10:
            continue
        c = c.loc[common_idx]
        h = h.loc[common_idx]
        lo = lo.loc[common_idx]

        # True Range
        prev_c = c.shift(1)
        tr = pd.concat([
            h - lo,
            (h - prev_c).abs(),
            (lo - prev_c).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        atr_dict[t] = atr

    return pd.DataFrame(atr_dict)


###############################################################################
# STRATEGY LOGIC
###############################################################################

def rank_sectors(etf_prices, date, lookback_days=126, n_top=3):
    """Rank sector ETFs by 6-month momentum, return top N sector names."""
    sector_etf_cols = [c for c in etf_prices.columns if c in SECTOR_ETFS]
    prices_to_date = etf_prices.loc[:date, sector_etf_cols]
    if len(prices_to_date) < lookback_days:
        return [], {}

    current = prices_to_date.iloc[-1]
    past = prices_to_date.iloc[-lookback_days]
    momentum = (current / past - 1).dropna()

    # Return all ranked sectors (for sector exit checks)
    all_ranked = momentum.sort_values(ascending=False)
    all_sector_ranks = {SECTOR_ETFS[etf]: rank+1
                        for rank, etf in enumerate(all_ranked.index)
                        if etf in SECTOR_ETFS}

    top_etfs = all_ranked.head(n_top).index.tolist()
    top_sectors = [SECTOR_ETFS[e] for e in top_etfs if e in SECTOR_ETFS]
    return top_sectors, all_sector_ranks


def pick_stocks_in_sectors(stock_prices, sector_map, winning_sectors, date,
                           stocks_per_sector=5, mom_lookback=63, skip_recent=21,
                           volume_df=None, require_volume_confirm=False):
    """Pick top momentum stocks within winning sectors."""
    selected = {}

    for sector in winning_sectors:
        sector_stocks = [t for t, s in sector_map.items()
                         if s == sector and t in stock_prices.columns]
        if len(sector_stocks) < 3:
            continue

        prices_to_date = stock_prices.loc[:date, sector_stocks]
        if len(prices_to_date) < mom_lookback + skip_recent:
            continue

        # 3-month momentum, skipping recent month (avoid mean reversion)
        current = prices_to_date.iloc[-skip_recent - 1]
        past = prices_to_date.iloc[-skip_recent - mom_lookback]
        mom = (current / past - 1).dropna()

        # Only take stocks with positive momentum
        mom_pos = mom[mom > 0]
        if len(mom_pos) < 2:
            mom_pos = mom.nlargest(2)

        candidates = mom_pos.sort_values(ascending=False).head(stocks_per_sector * 2)

        # Volume filter: require current volume > average 20-day volume
        if require_volume_confirm and volume_df is not None:
            filtered = []
            for t in candidates.index:
                if t in volume_df.columns:
                    vol_to_date = volume_df.loc[:date, t].dropna()
                    if len(vol_to_date) > 20:
                        avg_vol = vol_to_date.iloc[-20:].mean()
                        current_vol = vol_to_date.iloc[-1]
                        if current_vol >= avg_vol:  # >= 1x average
                            filtered.append(t)
                else:
                    filtered.append(t)  # no volume data = include
            candidates = candidates.loc[[t for t in filtered if t in candidates.index]]

        top = candidates.head(stocks_per_sector).index.tolist()
        selected[sector] = top

    return selected


def check_spy_above_200sma(spy_prices, date):
    """Check if SPY is above its 200-day SMA (dual momentum filter)."""
    spy_to_date = spy_prices.loc[:date]
    if len(spy_to_date) < 200:
        return True  # Not enough data, assume bull
    sma_200 = spy_to_date.iloc[-200:].mean()
    current = spy_to_date.iloc[-1]
    return current > sma_200


###############################################################################
# MAIN BACKTEST ENGINE
###############################################################################

def run_hybrid_backtest(etf_prices, stock_prices, sector_map, config,
                        volume_df=None, atr_df=None, spy_prices=None):
    """
    Walk-forward hybrid momentum backtest with configurable features.

    Config dict keys:
    - n_top_sectors: int (3)
    - stocks_per_sector: int (5)
    - sector_mom_lookback: int (126 = 6 months)
    - stock_mom_lookback: int (63 = 3 months)
    - use_cash_filter: bool (SPY > 200 SMA filter)
    - use_trailing_stop: bool
    - trailing_stop_atr_mult: float (2.0 = 2x ATR)
    - trailing_stop_pct: float (fallback if no ATR data, 0.15)
    - use_momentum_exit: bool (exit if 10d momentum flips negative)
    - use_sector_exit: bool (exit if sector drops out of top N)
    - sector_exit_rank_threshold: int (4 = exit if sector rank > 4)
    - position_stop_loss_pct: float (0.15 = -15% from entry)
    - require_volume_confirm: bool
    - train_months: int (60)
    - test_months: int (12)
    - slippage_pct: float (0.003)
    """
    n_top_sectors = config.get('n_top_sectors', 3)
    stocks_per_sector = config.get('stocks_per_sector', 5)
    sector_mom_lookback = config.get('sector_mom_lookback', 126)
    stock_mom_lookback = config.get('stock_mom_lookback', 63)
    use_cash_filter = config.get('use_cash_filter', True)
    use_trailing_stop = config.get('use_trailing_stop', True)
    trailing_stop_atr_mult = config.get('trailing_stop_atr_mult', 2.0)
    trailing_stop_pct_fallback = config.get('trailing_stop_pct', 0.15)
    use_momentum_exit = config.get('use_momentum_exit', True)
    use_sector_exit = config.get('use_sector_exit', True)
    sector_exit_rank_threshold = config.get('sector_exit_rank_threshold', 4)
    position_stop_loss_pct = config.get('position_stop_loss_pct', 0.15)
    require_volume_confirm = config.get('require_volume_confirm', False)
    train_months = config.get('train_months', 60)
    test_months = config.get('test_months', 12)
    slippage_pct = config.get('slippage_pct', 0.003)

    # Monthly rebalance dates
    monthly = stock_prices.resample('ME').last().index
    monthly = monthly[monthly >= stock_prices.index[max(252, sector_mom_lookback + 30)]]

    if len(monthly) < train_months + test_months + 1:
        print(f"  Not enough data: {len(monthly)} months, need {train_months + test_months + 1}")
        return None, {}

    all_returns = []
    all_dates = []
    trade_log = []
    exit_reasons = {'trailing_stop': 0, 'momentum_exit': 0, 'sector_exit': 0,
                    'stop_loss': 0, 'cash_filter': 0, 'rebalance': 0}
    sector_selection_log = []
    cash_days = 0
    invested_days = 0

    # Track active positions across monthly windows
    # Each position: {ticker, sector, entry_price, trailing_high, entry_date}
    active_positions = {}

    # Walk-forward windows
    n_windows = 0
    window_start = train_months
    while window_start + test_months <= len(monthly):
        window_rebal_dates = monthly[window_start:window_start + test_months]
        n_windows += 1

        for rebal_idx, rebal_date in enumerate(window_rebal_dates):
            # Check SPY cash filter at monthly rebalance
            in_cash = False
            if use_cash_filter and spy_prices is not None:
                if not check_spy_above_200sma(spy_prices, rebal_date):
                    in_cash = True

            # Determine next rebalance date
            rebal_pos = list(monthly).index(rebal_date)
            if rebal_pos + 1 < len(monthly):
                next_rebal = monthly[rebal_pos + 1]
            else:
                next_rebal = stock_prices.index[-1]

            # Get daily prices for this month
            month_prices = stock_prices.loc[rebal_date:next_rebal]
            if len(month_prices) < 2:
                continue

            if in_cash:
                # All cash — earn SHY return
                n_days = len(month_prices) - 1
                cash_days += n_days
                for d in range(1, len(month_prices)):
                    all_returns.append(SHY_DAILY_RETURN)
                    all_dates.append(month_prices.index[d])
                    exit_reasons['cash_filter'] += 1

                # Close all active positions
                for ticker in list(active_positions.keys()):
                    trade_log.append({
                        'ticker': ticker,
                        'exit_reason': 'cash_filter',
                        'entry_date': str(active_positions[ticker]['entry_date']),
                        'exit_date': str(rebal_date.date()),
                    })
                active_positions = {}
                continue

            # Monthly sector ranking
            winning_sectors, all_sector_ranks = rank_sectors(
                etf_prices, rebal_date,
                lookback_days=sector_mom_lookback,
                n_top=n_top_sectors
            )
            if len(winning_sectors) < 1:
                continue

            # Sector exit check: close positions in sectors that dropped out
            if use_sector_exit:
                for ticker in list(active_positions.keys()):
                    pos = active_positions[ticker]
                    sector = pos['sector']
                    rank = all_sector_ranks.get(sector, 99)
                    if rank > sector_exit_rank_threshold:
                        trade_log.append({
                            'ticker': ticker,
                            'exit_reason': 'sector_exit',
                            'sector': sector,
                            'sector_rank': rank,
                            'entry_date': str(pos['entry_date']),
                            'exit_date': str(rebal_date.date()),
                        })
                        exit_reasons['sector_exit'] += 1
                        del active_positions[ticker]

            # Pick new stocks
            selected = pick_stocks_in_sectors(
                stock_prices, sector_map, winning_sectors, rebal_date,
                stocks_per_sector=stocks_per_sector,
                mom_lookback=stock_mom_lookback, skip_recent=21,
                volume_df=volume_df,
                require_volume_confirm=require_volume_confirm,
            )
            if not selected:
                continue

            # Build target portfolio
            target_stocks = {}
            for sector, tickers in selected.items():
                for t in tickers:
                    target_stocks[t] = sector

            sector_selection_log.append({
                'date': str(rebal_date.date()),
                'sectors': list(selected.keys()),
                'n_stocks': len(target_stocks),
            })

            # Close positions not in target universe (rebalance exits)
            for ticker in list(active_positions.keys()):
                if ticker not in target_stocks:
                    exit_reasons['rebalance'] += 1
                    trade_log.append({
                        'ticker': ticker,
                        'exit_reason': 'rebalance',
                        'entry_date': str(active_positions[ticker]['entry_date']),
                        'exit_date': str(rebal_date.date()),
                    })
                    del active_positions[ticker]

            # Open new positions
            n_target = len(target_stocks)
            for ticker, sector in target_stocks.items():
                if ticker not in active_positions:
                    if ticker in month_prices.columns and not pd.isna(month_prices[ticker].iloc[0]):
                        entry_price = month_prices[ticker].iloc[0]
                        # Get ATR for trailing stop
                        atr_val = None
                        if atr_df is not None and ticker in atr_df.columns:
                            atr_to_date = atr_df.loc[:rebal_date, ticker].dropna()
                            if len(atr_to_date) > 0:
                                atr_val = atr_to_date.iloc[-1]

                        active_positions[ticker] = {
                            'sector': sector,
                            'entry_price': entry_price * (1 + slippage_pct),
                            'trailing_high': entry_price,
                            'entry_date': rebal_date.date(),
                            'atr_at_entry': atr_val,
                        }

            # Daily position health checks
            n_active_start = len(active_positions)
            if n_active_start == 0:
                # No positions, earn cash
                for d in range(1, len(month_prices)):
                    all_returns.append(SHY_DAILY_RETURN)
                    all_dates.append(month_prices.index[d])
                    cash_days += 1
                continue

            base_weight = 1.0 / max(n_target, 1)

            for d in range(1, len(month_prices)):
                date_now = month_prices.index[d]
                day_return = 0.0
                stopped_today = []
                n_active = len(active_positions)

                if n_active == 0:
                    all_returns.append(SHY_DAILY_RETURN)
                    all_dates.append(date_now)
                    cash_days += 1
                    continue

                invested_days += 1

                for ticker in list(active_positions.keys()):
                    if ticker not in month_prices.columns:
                        continue

                    pos = active_positions[ticker]
                    p = month_prices[ticker].iloc[d]
                    p_prev = month_prices[ticker].iloc[d - 1]

                    if pd.isna(p) or pd.isna(p_prev) or p_prev == 0:
                        continue

                    # Daily return contribution
                    ret = (p - p_prev) / p_prev
                    day_return += base_weight * ret

                    # Update trailing high
                    if p > pos['trailing_high']:
                        pos['trailing_high'] = p

                    exit_this = False
                    exit_reason = None

                    # EXIT 1: Trailing stop (2x ATR or fallback %)
                    if use_trailing_stop and pos['trailing_high'] > 0:
                        if pos['atr_at_entry'] is not None and pos['atr_at_entry'] > 0:
                            stop_distance = trailing_stop_atr_mult * pos['atr_at_entry']
                            stop_price = pos['trailing_high'] - stop_distance
                            if p <= stop_price:
                                exit_this = True
                                exit_reason = 'trailing_stop'
                        else:
                            dd_from_peak = 1 - p / pos['trailing_high']
                            if dd_from_peak >= trailing_stop_pct_fallback:
                                exit_this = True
                                exit_reason = 'trailing_stop'

                    # EXIT 2: Momentum breakdown (20-day momentum negative)
                    # Uses full stock_prices for lookback beyond current month
                    mom_lookback_exit = config.get('momentum_exit_lookback', 20)
                    if not exit_this and use_momentum_exit:
                        idx_in_full = stock_prices.index.get_indexer([date_now], method='ffill')[0]
                        if idx_in_full >= mom_lookback_exit:
                            p_Nago = stock_prices[ticker].iloc[idx_in_full - mom_lookback_exit]
                            if not pd.isna(p_Nago) and p_Nago > 0:
                                mom_Nd = (p - p_Nago) / p_Nago
                                if mom_Nd < -0.05:  # require >5% decline, not just any negative
                                    exit_this = True
                                    exit_reason = 'momentum_exit'

                    # EXIT 3: Position stop-loss from entry
                    if not exit_this and pos['entry_price'] > 0:
                        loss = 1 - p / pos['entry_price']
                        if loss >= position_stop_loss_pct:
                            exit_this = True
                            exit_reason = 'stop_loss'

                    # EXIT 4: Biweekly sector rank check (if sector dropped out of top N)
                    if not exit_this and use_sector_exit:
                        # Check biweekly (every 10 trading days) to avoid excessive churn
                        if d % 10 == 0:
                            _, daily_ranks = rank_sectors(
                                etf_prices, date_now,
                                lookback_days=sector_mom_lookback,
                                n_top=n_top_sectors
                            )
                            sector = pos['sector']
                            rank = daily_ranks.get(sector, 99)
                            if rank > sector_exit_rank_threshold:
                                exit_this = True
                                exit_reason = 'sector_exit'

                    if exit_this:
                        stopped_today.append((ticker, exit_reason))

                # Process exits
                for ticker, reason in stopped_today:
                    if ticker in active_positions:
                        day_return -= base_weight * slippage_pct  # exit slippage
                        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                        trade_log.append({
                            'ticker': ticker,
                            'exit_reason': reason,
                            'entry_date': str(active_positions[ticker]['entry_date']),
                            'exit_date': str(date_now.date()),
                        })
                        del active_positions[ticker]

                # Cash for empty slots
                active_frac = len(active_positions) / max(n_target, 1)
                cash_frac = 1 - active_frac
                if cash_frac > 0:
                    day_return += cash_frac * SHY_DAILY_RETURN

                all_returns.append(day_return)
                all_dates.append(date_now)

        window_start += test_months

    if not all_returns:
        return None, {}

    results = pd.Series(all_returns, index=pd.DatetimeIndex(all_dates))
    results = results[~results.index.duplicated(keep='last')]
    results = results.sort_index()

    # Compute avg hold period from trade_log
    hold_days = []
    for t in trade_log:
        try:
            entry = pd.Timestamp(t['entry_date'])
            exit_ = pd.Timestamp(t['exit_date'])
            hold_days.append((exit_ - entry).days)
        except:
            pass

    meta = {
        'total_trades': len(trade_log),
        'exit_reasons': exit_reasons,
        'n_walk_forward_windows': n_windows,
        'n_rebalance_periods': len(sector_selection_log),
        'cash_days': cash_days,
        'invested_days': invested_days,
        'cash_pct': f"{cash_days / max(cash_days + invested_days, 1):.1%}",
        'avg_hold_days': round(np.mean(hold_days), 1) if hold_days else 0,
        'median_hold_days': round(np.median(hold_days), 1) if hold_days else 0,
        'trades_per_year': round(len(trade_log) / max(len(results) / 252, 0.1), 1),
        'sector_selections_sample': (sector_selection_log[:3] + ['...'] + sector_selection_log[-3:]
            if len(sector_selection_log) > 6 else sector_selection_log),
    }

    return results, meta


###############################################################################
# METRICS, REGIME, PERMUTATION
###############################################################################

def compute_metrics(returns, name="Strategy"):
    """Compute risk-adjusted metrics."""
    if returns is None or len(returns) < 30:
        return {}

    ann_factor = 252
    total_days = len(returns)
    years = total_days / ann_factor

    cum_ret = (1 + returns).prod() - 1
    cagr = (1 + cum_ret) ** (1 / max(years, 0.1)) - 1

    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = (returns.mean() * ann_factor) / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(ann_factor) if len(returns[returns < 0]) > 0 else 1e-6
    sortino = (returns.mean() * ann_factor) / downside

    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else float('inf')

    win_rate = (returns > 0).mean()

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'name': name,
        'CAGR': round(cagr * 100, 2),
        'CAGR_str': f"{cagr:.1%}",
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'Max_DD': round(max_dd * 100, 2),
        'Max_DD_str': f"{max_dd:.1%}",
        'Calmar': round(calmar, 2),
        'Win_Rate': round(win_rate * 100, 1),
        'Profit_Factor': round(pf, 2),
        'Total_Return': f"{cum_ret:.1%}",
        'N_Days': total_days,
        'Ann_Vol': round(ann_vol * 100, 1),
    }


def regime_analysis(returns, spy_prices):
    """Stratify by bull/bear regime (SPY above/below 200MA)."""
    spy_ma200 = spy_prices.rolling(200).mean()
    regime = (spy_prices > spy_ma200).astype(int)
    regime.index = pd.to_datetime(regime.index)

    common = returns.index.intersection(regime.index)
    if len(common) < 30:
        return None, None, None

    r = returns.loc[common]
    reg = regime.loc[common]

    bull_returns = r[reg == 1]
    bear_returns = r[reg == 0]

    bull_metrics = compute_metrics(bull_returns, "Bull Regime") if len(bull_returns) > 30 else {}
    bear_metrics = compute_metrics(bear_returns, "Bear Regime") if len(bear_returns) > 30 else {}

    if bull_metrics and bear_metrics:
        s_bull = bull_metrics['Sharpe']
        s_bear = bear_metrics['Sharpe']
        denom = max(abs(s_bull), abs(s_bear))
        regime_gap = abs(s_bull - s_bear) / denom if denom > 0 else 0
    else:
        regime_gap = None

    return bull_metrics, bear_metrics, regime_gap


def permutation_test(returns, n_perms=500):
    """Permutation test: shuffle daily returns, compare Sharpe."""
    actual_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    rng = np.random.RandomState(42)
    perm_sharpes = []
    vals = returns.values
    for _ in range(n_perms):
        shuffled = rng.permutation(vals)
        s = shuffled.mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
        perm_sharpes.append(s)

    p_value = (np.sum(np.array(perm_sharpes) >= actual_sharpe) + 1) / (n_perms + 1)
    return actual_sharpe, p_value, perm_sharpes


###############################################################################
# CONFIGURATION MATRIX
###############################################################################

def get_config_matrix():
    """Return a list of named configs to test."""
    configs = {}

    # Config A: Full hybrid (all features on)
    configs['A_full_hybrid'] = {
        'n_top_sectors': 3,
        'stocks_per_sector': 5,
        'sector_mom_lookback': 126,
        'stock_mom_lookback': 63,
        'use_cash_filter': True,
        'use_trailing_stop': True,
        'trailing_stop_atr_mult': 2.0,
        'trailing_stop_pct': 0.15,
        'use_momentum_exit': True,
        'use_sector_exit': True,
        'sector_exit_rank_threshold': 4,
        'position_stop_loss_pct': 0.15,
        'require_volume_confirm': True,
        'train_months': 60,
        'test_months': 12,
        'slippage_pct': 0.003,
    }

    # Config B: No cash filter (test contribution of dual momentum)
    configs['B_no_cash_filter'] = {**configs['A_full_hybrid'],
        'use_cash_filter': False,
    }

    # Config C: No trailing stop (test contribution of dynamic exits)
    configs['C_no_trailing_stop'] = {**configs['A_full_hybrid'],
        'use_trailing_stop': False,
    }

    # Config D: No momentum exit
    configs['D_no_momentum_exit'] = {**configs['A_full_hybrid'],
        'use_momentum_exit': False,
    }

    # Config E: No sector exit
    configs['E_no_sector_exit'] = {**configs['A_full_hybrid'],
        'use_sector_exit': False,
    }

    # Config F: Cash filter only (no dynamic exits at all)
    configs['F_cash_filter_only'] = {**configs['A_full_hybrid'],
        'use_trailing_stop': False,
        'use_momentum_exit': False,
        'use_sector_exit': False,
        'position_stop_loss_pct': 0.99,  # effectively off
    }

    # Config G: Vanilla stock-in-sector (no cash filter, no dynamic exits)
    configs['G_vanilla_stock_sector'] = {**configs['A_full_hybrid'],
        'use_cash_filter': False,
        'use_trailing_stop': False,
        'use_momentum_exit': False,
        'use_sector_exit': False,
        'position_stop_loss_pct': 0.99,
        'require_volume_confirm': False,
    }

    # Config H: Tight stops variant
    configs['H_tight_stops'] = {**configs['A_full_hybrid'],
        'trailing_stop_atr_mult': 1.5,
        'position_stop_loss_pct': 0.10,
    }

    # Config I: Cash filter + trailing stop only (no momentum/sector intra-month exits)
    configs['I_cash_plus_trail'] = {**configs['A_full_hybrid'],
        'use_momentum_exit': False,
        'use_sector_exit': False,
    }

    # Config J: 4 sectors instead of 3 (broader diversification)
    configs['J_four_sectors'] = {**configs['A_full_hybrid'],
        'n_top_sectors': 4,
        'use_momentum_exit': False,  # off based on first run findings
    }

    return configs


###############################################################################
# MAIN
###############################################################################

def main():
    print("=" * 80)
    print("HYBRID MOMENTUM BACKTESTER")
    print("Stock-within-Sector + Dual Momentum Filter + Dynamic Exits")
    print("=" * 80)
    print("Combining: Sector rotation (26.7% CAGR) + Cash filter (-11.8% MaxDD)")
    print("         + Dynamic exits (trailing ATR stop, momentum breakdown, sector exit)")
    print(f"Walk-forward: 60-month train, 12-month test, sliding")
    print("=" * 80)

    # 1. Get sector mapping
    print("\n[1/7] Getting S&P 500 sector mapping...")
    sector_map = get_sp500_with_sectors()
    sectors_found = set(sector_map.values())
    print(f"  {len(sector_map)} stocks mapped to {len(sectors_found)} sectors")

    # 2. Download data
    print("\n[2/7] Downloading price data (2012-2026)...")
    result = download_all_data(sector_map, start='2012-01-01', end='2026-07-13')

    if len(result) == 5:
        etf_prices, stock_prices, volume_df, high_df, low_df = result
    else:
        etf_prices, stock_prices = result[0], result[1]
        volume_df, high_df, low_df = load_supplementary_data()

    # Extract SPY
    spy_close = etf_prices['SPY'] if 'SPY' in etf_prices.columns else None
    if spy_close is None:
        print("ERROR: Could not get SPY data")
        return

    # Compute ATR
    print("\n[3/7] Computing ATR for all stocks...")
    atr_df = None
    if high_df is not None and low_df is not None:
        atr_df = compute_atr(stock_prices, high_df, low_df, period=14)
        print(f"  ATR computed for {atr_df.shape[1]} stocks")
    else:
        print("  WARNING: No high/low data available, will use percentage trailing stop")

    # 4. Run all configs
    print("\n[4/7] Running multi-config walk-forward backtests...")
    configs = get_config_matrix()
    all_results = {}
    all_meta = {}
    all_metrics = {}

    for name, config in configs.items():
        print(f"\n  --- Config {name} ---")
        desc_parts = []
        if config.get('use_cash_filter'): desc_parts.append("cash_filter")
        if config.get('use_trailing_stop'): desc_parts.append("trail_stop")
        if config.get('use_momentum_exit'): desc_parts.append("mom_exit")
        if config.get('use_sector_exit'): desc_parts.append("sector_exit")
        if config.get('require_volume_confirm'): desc_parts.append("vol_confirm")
        print(f"  Features: {', '.join(desc_parts) if desc_parts else 'none (vanilla)'}")

        returns, meta = run_hybrid_backtest(
            etf_prices, stock_prices, sector_map, config,
            volume_df=volume_df, atr_df=atr_df, spy_prices=spy_close
        )

        if returns is not None and len(returns) > 30:
            all_results[name] = returns
            all_meta[name] = meta
            metrics = compute_metrics(returns, name)
            all_metrics[name] = metrics
            print(f"  CAGR: {metrics['CAGR_str']}, Sharpe: {metrics['Sharpe']}, "
                  f"MaxDD: {metrics['Max_DD_str']}, WR: {metrics['Win_Rate']}%")
            print(f"  Trades: {meta['total_trades']}, "
                  f"Avg Hold: {meta['avg_hold_days']}d, "
                  f"Cash: {meta['cash_pct']}")
        else:
            print(f"  FAILED — insufficient data")

    # SPY benchmark
    print("\n  --- Benchmark: SPY Buy & Hold ---")
    spy_ret = spy_close.pct_change().dropna()
    # Align to common period
    if all_results:
        first_config = list(all_results.keys())[0]
        common_start = all_results[first_config].index[0]
        common_end = all_results[first_config].index[-1]
        spy_aligned = spy_ret.loc[common_start:common_end]
        spy_metrics = compute_metrics(spy_aligned, "SPY_BuyHold")
        all_metrics['SPY_BuyHold'] = spy_metrics
        print(f"  CAGR: {spy_metrics['CAGR_str']}, Sharpe: {spy_metrics['Sharpe']}, "
              f"MaxDD: {spy_metrics['Max_DD_str']}")

    # 5. Regime analysis for all configs
    print("\n[5/7] Regime analysis (SPY above/below 200 SMA)...")
    regime_results = {}
    for name, returns in all_results.items():
        bull, bear, gap = regime_analysis(returns, spy_close)
        regime_results[name] = {
            'bull': bull,
            'bear': bear,
            'gap': gap,
            'pass': gap is not None and gap < 0.50,
        }
        status = 'PASS' if gap is not None and gap < 0.50 else 'FAIL'
        gap_str = f"{gap:.3f}" if gap is not None else "N/A"
        bull_sharpe = bull.get('Sharpe', 'N/A') if bull else 'N/A'
        bear_sharpe = bear.get('Sharpe', 'N/A') if bear else 'N/A'
        print(f"  {name}: Bull Sharpe={bull_sharpe}, Bear Sharpe={bear_sharpe}, "
              f"Gap={gap_str} -> {status}")

    # 6. Permutation test on best config
    print("\n[6/7] Permutation test (500 shuffles on top configs)...")
    perm_results = {}
    # Test top 3 by Sharpe
    sorted_configs = sorted(all_metrics.items(),
                            key=lambda x: x[1].get('Sharpe', 0) if x[0] != 'SPY_BuyHold' else -99,
                            reverse=True)
    for name, metrics in sorted_configs[:3]:
        if name == 'SPY_BuyHold':
            continue
        if name not in all_results:
            continue
        actual_s, p_val, _ = permutation_test(all_results[name], n_perms=500)
        perm_results[name] = {'actual_sharpe': round(actual_s, 3), 'p_value': round(p_val, 4)}
        sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
        print(f"  {name}: Sharpe={actual_s:.3f}, p={p_val:.4f} -> {sig}")

    # 7. Summary comparison table
    print("\n" + "=" * 80)
    print("[7/7] SUMMARY COMPARISON")
    print("=" * 80)

    header = f"{'Config':<28} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'WR%':>6} {'PF':>5} {'Trades/yr':>10}"
    print(header)
    print("-" * len(header))

    for name, metrics in sorted(all_metrics.items(), key=lambda x: x[1].get('Sharpe', 0), reverse=True):
        meta = all_meta.get(name, {})
        trades_yr = meta.get('trades_per_year', '-')
        print(f"{name:<28} {metrics.get('CAGR_str', 'N/A'):>7} {metrics.get('Sharpe', 'N/A'):>7} "
              f"{metrics.get('Sortino', 'N/A'):>8} {metrics.get('Max_DD_str', 'N/A'):>8} "
              f"{metrics.get('Calmar', 'N/A'):>7} {metrics.get('Win_Rate', 'N/A'):>6} "
              f"{metrics.get('Profit_Factor', 'N/A'):>5} {str(trades_yr):>10}")

    # Regime summary
    print("\n--- REGIME TEST RESULTS ---")
    for name, res in regime_results.items():
        status = 'PASS' if res['pass'] else 'FAIL'
        gap = f"{res['gap']:.3f}" if res['gap'] is not None else 'N/A'
        print(f"  {name:<28} Gap={gap} -> {status}")

    # Key hypothesis test
    print("\n--- KEY HYPOTHESIS ---")
    if 'A_full_hybrid' in regime_results and 'G_vanilla_stock_sector' in regime_results:
        full = regime_results['A_full_hybrid']
        vanilla = regime_results['G_vanilla_stock_sector']
        full_gap_str = f"{full['gap']:.3f}" if full['gap'] is not None else 'N/A'
        vanilla_gap_str = f"{vanilla['gap']:.3f}" if vanilla['gap'] is not None else 'N/A'
        print(f"  Full hybrid regime gap:    {full_gap_str} "
              f"({'PASS' if full['pass'] else 'FAIL'})")
        print(f"  Vanilla stock-sector gap:  {vanilla_gap_str} "
              f"({'PASS' if vanilla['pass'] else 'FAIL'})")
        if full['pass'] and not vanilla['pass']:
            print("  CONFIRMED: Cash filter fixes regime dependence!")
        elif full['pass'] and vanilla['pass']:
            print("  Both pass regime test — cash filter may not be needed for regime agnosticity")
        elif not full['pass']:
            print("  HYPOTHESIS REJECTED: Even full hybrid fails regime test")
    else:
        print("  Could not compare — missing config results")

    # Feature contribution analysis
    print("\n--- FEATURE CONTRIBUTION ANALYSIS ---")
    if 'A_full_hybrid' in all_metrics:
        full_sharpe = all_metrics['A_full_hybrid']['Sharpe']
        for name in ['B_no_cash_filter', 'C_no_trailing_stop', 'D_no_momentum_exit', 'E_no_sector_exit']:
            if name in all_metrics:
                this_sharpe = all_metrics[name]['Sharpe']
                delta = full_sharpe - this_sharpe
                feature = name.replace('B_no_', '').replace('C_no_', '').replace('D_no_', '').replace('E_no_', '')
                sign = '+' if delta > 0 else ''
                print(f"  Removing {feature:<20} -> Sharpe {this_sharpe:.2f} (delta: {sign}{delta:.2f})")

    # Permutation summary
    print("\n--- PERMUTATION TESTS ---")
    for name, res in perm_results.items():
        sig = "SIGNIFICANT" if res['p_value'] < 0.05 else "NOT SIGNIFICANT"
        print(f"  {name:<28} p={res['p_value']:.4f} -> {sig}")

    # Save all results
    print("\n--- SAVING RESULTS ---")
    output = {
        'strategy': 'Hybrid Momentum (Stock-within-Sector + Dual Momentum + Dynamic Exits)',
        'run_date': str(datetime.now()),
        'configs_tested': len(configs),
        'walk_forward': {'train_months': 60, 'test_months': 12, 'window_type': 'sliding'},
        'metrics': {},
        'regime_analysis': {},
        'permutation_tests': perm_results,
        'meta': {},
    }

    for name in all_metrics:
        output['metrics'][name] = all_metrics[name]
        if name in regime_results:
            output['regime_analysis'][name] = {
                'bull': regime_results[name]['bull'],
                'bear': regime_results[name]['bear'],
                'gap': regime_results[name]['gap'],
                'pass': regime_results[name]['pass'],
            }
        if name in all_meta:
            output['meta'][name] = all_meta[name]

    # Determine best config
    best_name = None
    best_sharpe = -999
    for name, m in all_metrics.items():
        if name == 'SPY_BuyHold':
            continue
        # Must pass regime test
        if name in regime_results and not regime_results[name]['pass']:
            continue
        if m['Sharpe'] > best_sharpe:
            best_sharpe = m['Sharpe']
            best_name = name

    if best_name is None:
        # No config passed regime test — pick best Sharpe anyway
        for name, m in all_metrics.items():
            if name == 'SPY_BuyHold':
                continue
            if m['Sharpe'] > best_sharpe:
                best_sharpe = m['Sharpe']
                best_name = name
        output['best_config'] = {'name': best_name, 'note': 'NO CONFIG PASSED REGIME TEST'}
    else:
        output['best_config'] = {'name': best_name, 'regime_test': 'PASS'}

    if best_name:
        output['best_config']['metrics'] = all_metrics[best_name]
        print(f"\n  BEST CONFIG: {best_name} (Sharpe {best_sharpe:.2f})")

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, 'hybrid_momentum_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results JSON saved")

    # Save daily returns for best config
    if best_name and best_name in all_results:
        csv_path = os.path.join(OUTPUT_DIR, 'hybrid_momentum_returns.csv')
        all_results[best_name].to_csv(csv_path)

    # Save all config returns
    for name, returns in all_results.items():
        csv_path = os.path.join(OUTPUT_DIR, f'returns_{name}.csv')
        returns.to_csv(csv_path)

    print(f"\n  All output saved to {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == '__main__':
    main()
