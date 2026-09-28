#!/usr/bin/env python3
"""
Sector/Factor Rotation Growth Strategy Backtester
Three approaches compared head-to-head:
  1. Sector Momentum — Top 3-4 sectors by 3/6-mo momentum via ETFs
  2. Dual Momentum — Absolute + relative momentum with bear-market cash filter
  3. Stock-within-Sector — Top 3 sectors, top 5 stocks within each

Walk-forward: 60-month train, 12-month test, sliding
Dynamic exits: trailing stops, momentum breakdown, absolute momentum filter
Regime test: SPY above/below 200 SMA
Permutation test for statistical significance
"""

import os, sys, json, warnings, time, traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/sector_momentum'
os.makedirs(OUTPUT_DIR, exist_ok=True)

###############################################################################
# CONSTANTS
###############################################################################

SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Health Care',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discretionary',
    'XLC': 'Communication Services',
    'XLP': 'Consumer Staples',
    'XLRE': 'Real Estate',
    'XLB': 'Materials',
    'XLU': 'Utilities',
}

# Map S&P 500 stocks to GICS sectors (representative sample per sector)
SECTOR_STOCKS = {
    'Technology': ['AAPL','MSFT','NVDA','AVGO','CSCO','ACN','TXN','QCOM','INTC','ADI',
                   'AMAT','LRCX','KLAC','MCHP','CDNS','SNPS','FTNT','PANW','CRM','NOW',
                   'ADBE','ORCL','INTU','PLTR','ANET','MPWR','ON','NXPI','GEN','KEYS'],
    'Financials': ['JPM','V','MA','BAC','WFC','GS','MS','BLK','SCHW','CB',
                   'AXP','CME','ICE','PNC','USB','TFC','MMC','AON','AJG','MSCI',
                   'MET','PRU','AFL','AIG','ALL','TRV','PGR','SPGI','MCO','FI'],
    'Energy': ['XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY','PXD',
               'HES','DVN','FANG','HAL','BKR','WMB','OKE','KMI','TRGP','CTRA'],
    'Health Care': ['UNH','JNJ','LLY','ABBV','MRK','TMO','ABT','DHR','PFE','AMGN',
                    'ISRG','SYK','VRTX','REGN','GILD','CI','ELV','HUM','ZTS','BDX',
                    'BSX','MDT','DXCM','A','IQV','EW','IDXX','MTD','HOLX','ALGN'],
    'Industrials': ['CAT','DE','UNP','HON','RTX','BA','GE','LMT','NOC','GD',
                    'ITW','EMR','FDX','UPS','WM','RSG','CSX','NSC','PCAR','TDG',
                    'ROK','SWK','IR','PH','CTAS','FAST','ODFL','J','VRSK','CPRT'],
    'Consumer Discretionary': ['AMZN','TSLA','HD','MCD','LOW','NKE','SBUX','TJX','BKNG','ORLY',
                                'AZO','ROST','CMG','DHI','LEN','PHM','GM','F','YUM','DPZ',
                                'POOL','GRMN','ULTA','EBAY','ETSY','BBY','KMX','GPC','LKQ','APTV'],
    'Communication Services': ['META','GOOGL','GOOG','NFLX','DIS','CMCSA','T','VZ','TMUS','CHTR',
                                'EA','TTWO','ATVI','MTCH','LYV','PARA','WBD','FOX','IPG','OMC'],
    'Consumer Staples': ['PG','PEP','KO','COST','WMT','PM','MO','CL','MDLZ','KDP',
                          'GIS','SJM','HSY','K','CAG','STZ','BF-B','TAP','KR','SYY',
                          'ADM','TSN','HRL','MKC','CHD','CLX','EL','KMB','MNST','WBA'],
    'Real Estate': ['PLD','AMT','CCI','EQIX','SPG','PSA','WELL','O','DLR','VICI',
                     'ARE','AVB','EQR','ESS','MAA','UDR','CPT','SUI','ELS','REG'],
    'Materials': ['LIN','APD','SHW','FCX','NEM','NUE','DOW','DD','ECL','PPG',
                   'VMC','MLM','CTVA','ALB','CE','EMN','IFF','FMC','CF','MOS'],
    'Utilities': ['NEE','SO','DUK','D','AEP','SRE','EXC','XEL','WEC','ES',
                   'ED','AEE','CMS','DTE','FE','ETR','PEG','PPL','AWK','ATO'],
}

# Flatten for downloading
ALL_STOCKS = []
STOCK_TO_SECTOR = {}
for sector, stocks in SECTOR_STOCKS.items():
    for s in stocks:
        if s not in STOCK_TO_SECTOR:
            ALL_STOCKS.append(s)
            STOCK_TO_SECTOR[s] = sector

###############################################################################
# 1. DATA ACQUISITION
###############################################################################

def download_prices(tickers, start='2010-01-01', end='2026-07-13'):
    """Download adjusted close prices with batching."""
    import yfinance as yf

    all_prices = {}
    batch_size = 50
    total = len(tickers)
    for i in range(0, total, batch_size):
        batch = tickers[i:i+batch_size]
        ticker_str = ' '.join(batch)
        try:
            data = yf.download(ticker_str, start=start, end=end, progress=False,
                               threads=True)
            # yfinance 1.2+ returns MultiIndex (Price, Ticker) by default
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        col = data['Close'][t].dropna()
                        if len(col) > 252:
                            all_prices[t] = col
                    except:
                        pass
            else:
                # Single ticker fallback
                col = data['Close'].dropna()
                if len(col) > 252:
                    all_prices[batch[0]] = col
        except Exception as e:
            print(f"  Batch {i//batch_size} error: {e}")
        if i + batch_size < total:
            time.sleep(0.3)
        print(f"  Downloaded {min(i+batch_size, total)}/{total} tickers")

    prices = pd.DataFrame(all_prices)
    prices.index = pd.to_datetime(prices.index)
    # Flatten MultiIndex columns if present
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    prices = prices.sort_index()
    return prices


def load_all_data():
    """Load or download all price data."""
    cache_etf = os.path.join(OUTPUT_DIR, 'cache_sector_etfs.parquet')
    cache_stocks = os.path.join(OUTPUT_DIR, 'cache_sp500_stocks.parquet')
    cache_spy = os.path.join(OUTPUT_DIR, 'cache_spy.parquet')

    # Sector ETFs
    if os.path.exists(cache_etf):
        print("Loading cached sector ETF prices...")
        etf_prices = pd.read_parquet(cache_etf)
    else:
        print("Downloading sector ETF prices...")
        etf_tickers = list(SECTOR_ETFS.keys())
        etf_prices = download_prices(etf_tickers)
        etf_prices.to_parquet(cache_etf)
    print(f"  Sector ETFs: {etf_prices.shape[1]} tickers, {etf_prices.shape[0]} days")

    # SPY for regime filter
    if os.path.exists(cache_spy):
        spy_prices = pd.read_parquet(cache_spy)
    else:
        print("Downloading SPY...")
        spy_prices = download_prices(['SPY'])
        spy_prices.to_parquet(cache_spy)

    # Individual stocks
    if os.path.exists(cache_stocks):
        print("Loading cached stock prices...")
        stock_prices = pd.read_parquet(cache_stocks)
    else:
        print("Downloading S&P 500 stock prices...")
        stock_prices = download_prices(ALL_STOCKS)
        stock_prices.to_parquet(cache_stocks)
    print(f"  Stocks: {stock_prices.shape[1]} tickers, {stock_prices.shape[0]} days")

    return etf_prices, stock_prices, spy_prices


###############################################################################
# 2. FEATURE ENGINEERING
###############################################################################

def compute_momentum(prices, windows=[63, 126, 252]):
    """Compute momentum (total return) over various lookback windows."""
    mom = {}
    for w in windows:
        label = f'mom_{w}d'
        mom[label] = prices.pct_change(w)
    return mom


def compute_sma(prices, window=200):
    """Compute simple moving average."""
    return prices.rolling(window).mean()


def rank_sectors_by_momentum(etf_prices, date, lookbacks=[63, 126]):
    """Rank sectors by blended momentum score at a given date."""
    loc = etf_prices.index.get_indexer([date], method='ffill')[0]
    if loc < max(lookbacks):
        return []

    scores = {}
    for etf in etf_prices.columns:
        if etf not in SECTOR_ETFS:
            continue
        vals = etf_prices[etf].iloc[:loc+1]
        if len(vals) < max(lookbacks) + 1:
            continue
        # Blended: 50% 3-month + 50% 6-month momentum
        m3 = vals.iloc[-1] / vals.iloc[-min(63, len(vals)-1)] - 1 if len(vals) > 63 else 0
        m6 = vals.iloc[-1] / vals.iloc[-min(126, len(vals)-1)] - 1 if len(vals) > 126 else 0
        scores[etf] = 0.5 * m3 + 0.5 * m6

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def rank_stocks_in_sector(stock_prices, sector, date, lookbacks=[63, 126], top_n=5):
    """Rank individual stocks within a sector by momentum."""
    sector_tickers = [s for s in SECTOR_STOCKS.get(sector, []) if s in stock_prices.columns]
    if not sector_tickers:
        return []

    loc = stock_prices.index.get_indexer([date], method='ffill')[0]
    if loc < max(lookbacks):
        return []

    scores = {}
    for ticker in sector_tickers:
        vals = stock_prices[ticker].iloc[:loc+1].dropna()
        if len(vals) < max(lookbacks) + 1:
            continue
        m3 = vals.iloc[-1] / vals.iloc[-min(63, len(vals)-1)] - 1 if len(vals) > 63 else 0
        m6 = vals.iloc[-1] / vals.iloc[-min(126, len(vals)-1)] - 1 if len(vals) > 126 else 0
        scores[ticker] = 0.5 * m3 + 0.5 * m6

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


###############################################################################
# 3. STRATEGY IMPLEMENTATIONS
###############################################################################

def strategy_sector_momentum(etf_prices, spy_prices, train_start, train_end,
                              test_start, test_end, params):
    """
    Strategy 1: Sector Momentum
    Go long top N sectors by 3/6-mo momentum. Monthly rebalance.
    Dynamic exit: trailing stop + momentum breakdown.
    """
    n_sectors = params.get('n_sectors', 4)
    trailing_stop_pct = params.get('trailing_stop', 0.08)
    rebal_freq = params.get('rebal_freq', 21)  # trading days

    # Use training period to calibrate parameters (optimize n_sectors, trailing stop)
    # For walk-forward we just use the params as-is and apply to test period

    test_dates = etf_prices.loc[test_start:test_end].index
    if len(test_dates) == 0:
        return pd.Series(dtype=float)

    # HC #718 R3: Transaction costs — 5 bps per leg on turnover
    COST_BPS = 5

    portfolio_returns = []
    current_holdings = {}
    high_water = {}
    days_since_rebal = rebal_freq  # force rebalance on first day

    for date in test_dates:
        daily_ret = 0.0
        days_since_rebal += 1

        # Check trailing stops and momentum breakdown for current holdings
        exits = []
        for etf, entry_info in current_holdings.items():
            if etf not in etf_prices.columns:
                continue
            loc = etf_prices.index.get_indexer([date], method='ffill')[0]
            if loc < 1:
                continue
            price = etf_prices[etf].iloc[loc]
            prev_price = etf_prices[etf].iloc[loc-1]

            # Update high water mark
            if etf not in high_water or price > high_water[etf]:
                high_water[etf] = price

            # Trailing stop
            drawdown = (high_water[etf] - price) / high_water[etf]
            if drawdown > trailing_stop_pct:
                exits.append(etf)
                continue

            # Momentum breakdown: if 1-month return goes negative
            if loc >= 21:
                m1 = price / etf_prices[etf].iloc[loc-21] - 1
                if m1 < -0.05:  # 5% decline over 1 month
                    exits.append(etf)

        # HC #718 R3: transaction costs on stop/momentum exits
        if exits and current_holdings:
            exit_frac = len(exits) / len(current_holdings)
            daily_ret -= exit_frac * COST_BPS / 10000  # sell cost

        for etf in exits:
            del current_holdings[etf]
            if etf in high_water:
                del high_water[etf]

        # Monthly rebalance
        if days_since_rebal >= rebal_freq:
            ranked = rank_sectors_by_momentum(etf_prices, date)
            if ranked:
                old_holdings = set(current_holdings.keys())
                top_sectors = [r[0] for r in ranked[:n_sectors]]
                new_holdings = set(etf for etf in top_sectors if etf in etf_prices.columns)
                # HC #718 R3: transaction costs on rebalance turnover
                if old_holdings:
                    changed = len(old_holdings.symmetric_difference(new_holdings))
                    turnover_frac = changed / max(len(old_holdings), len(new_holdings))
                    daily_ret -= turnover_frac * COST_BPS / 10000  # cost per changed leg
                else:
                    daily_ret -= COST_BPS / 10000  # initial buy
                current_holdings = {etf: {'date': date} for etf in new_holdings}
                high_water = {}
                for etf in current_holdings:
                    loc = etf_prices.index.get_indexer([date], method='ffill')[0]
                    if loc >= 0:
                        high_water[etf] = etf_prices[etf].iloc[loc]
                days_since_rebal = 0

        # Calculate daily return (equal weight)
        if current_holdings:
            weight = 1.0 / len(current_holdings)
            for etf in current_holdings:
                loc = etf_prices.index.get_indexer([date], method='ffill')[0]
                if loc >= 1:
                    ret = etf_prices[etf].iloc[loc] / etf_prices[etf].iloc[loc-1] - 1
                    daily_ret += weight * ret

        portfolio_returns.append(daily_ret)

    return pd.Series(portfolio_returns, index=test_dates)


def strategy_dual_momentum(etf_prices, spy_prices, train_start, train_end,
                            test_start, test_end, params):
    """
    Strategy 2: Dual Momentum
    Absolute filter: Only invest when SPY > 200-day SMA.
    Relative: Pick top N sectors by momentum among qualifying.
    Goes to cash in bear markets.
    """
    n_sectors = params.get('n_sectors', 3)
    trailing_stop_pct = params.get('trailing_stop', 0.10)
    rebal_freq = params.get('rebal_freq', 21)
    sma_window = params.get('sma_window', 200)

    # Merge SPY into the date range
    spy_col = spy_prices.columns[0] if isinstance(spy_prices, pd.DataFrame) else 'SPY'

    test_dates = etf_prices.loc[test_start:test_end].index
    if len(test_dates) == 0:
        return pd.Series(dtype=float)

    # HC #718 R3: Transaction costs — 5 bps per leg on turnover
    COST_BPS = 5

    portfolio_returns = []
    current_holdings = {}
    high_water = {}
    days_since_rebal = rebal_freq
    cash_days = 0

    for date in test_dates:
        daily_ret = 0.0
        days_since_rebal += 1

        # Absolute momentum filter: is SPY above 200-day SMA?
        spy_loc = spy_prices.index.get_indexer([date], method='ffill')[0]
        in_uptrend = False
        if spy_loc >= sma_window:
            spy_sma = spy_prices.iloc[spy_loc-sma_window:spy_loc+1, 0].mean()
            spy_price = spy_prices.iloc[spy_loc, 0]
            in_uptrend = spy_price > spy_sma

        if not in_uptrend:
            # HC #718 R3: cost to liquidate when going to cash
            if current_holdings:
                daily_ret -= COST_BPS / 10000  # sell cost
            current_holdings = {}
            high_water = {}
            cash_days += 1
            portfolio_returns.append(daily_ret)
            continue

        # Check trailing stops
        exits = []
        for etf in list(current_holdings.keys()):
            loc = etf_prices.index.get_indexer([date], method='ffill')[0]
            if loc < 1 or etf not in etf_prices.columns:
                continue
            price = etf_prices[etf].iloc[loc]
            if etf not in high_water or price > high_water[etf]:
                high_water[etf] = price
            drawdown = (high_water[etf] - price) / high_water[etf]
            if drawdown > trailing_stop_pct:
                exits.append(etf)

        if exits and current_holdings:
            exit_frac = len(exits) / len(current_holdings)
            daily_ret -= exit_frac * COST_BPS / 10000

        for etf in exits:
            del current_holdings[etf]
            if etf in high_water:
                del high_water[etf]

        # Monthly rebalance
        if days_since_rebal >= rebal_freq:
            ranked = rank_sectors_by_momentum(etf_prices, date)
            if ranked:
                old_holdings = set(current_holdings.keys())
                # Additional sector-level absolute momentum filter
                qualified = [(etf, score) for etf, score in ranked if score > 0]
                top = [r[0] for r in qualified[:n_sectors]]
                if top:
                    new_holdings = set(etf for etf in top if etf in etf_prices.columns)
                    # HC #718 R3: transaction costs on rebalance turnover
                    if old_holdings:
                        changed = len(old_holdings.symmetric_difference(new_holdings))
                        turnover_frac = changed / max(len(old_holdings), len(new_holdings))
                        daily_ret -= turnover_frac * COST_BPS / 10000
                    else:
                        daily_ret -= COST_BPS / 10000  # initial buy
                    current_holdings = {etf: {'date': date} for etf in new_holdings}
                    high_water = {}
                    for etf in current_holdings:
                        loc = etf_prices.index.get_indexer([date], method='ffill')[0]
                        if loc >= 0:
                            high_water[etf] = etf_prices[etf].iloc[loc]
                else:
                    if current_holdings:
                        daily_ret -= COST_BPS / 10000  # sell all to cash
                    current_holdings = {}
                    high_water = {}
                days_since_rebal = 0

        # Calculate daily return
        if current_holdings:
            weight = 1.0 / len(current_holdings)
            for etf in current_holdings:
                loc = etf_prices.index.get_indexer([date], method='ffill')[0]
                if loc >= 1:
                    ret = etf_prices[etf].iloc[loc] / etf_prices[etf].iloc[loc-1] - 1
                    daily_ret += weight * ret

        portfolio_returns.append(daily_ret)

    return pd.Series(portfolio_returns, index=test_dates)


def strategy_stock_within_sector(etf_prices, stock_prices, spy_prices,
                                  train_start, train_end, test_start, test_end, params):
    """
    Strategy 3: Stock-within-Sector
    Pick top 3 sectors by momentum, then top 5 stocks within each sector.
    More concentrated, higher potential returns.
    """
    n_sectors = params.get('n_sectors', 3)
    n_stocks_per_sector = params.get('n_stocks', 5)
    trailing_stop_pct = params.get('trailing_stop', 0.12)
    rebal_freq = params.get('rebal_freq', 21)
    sma_window = params.get('sma_window', 200)

    test_dates = stock_prices.loc[test_start:test_end].index
    if len(test_dates) == 0:
        return pd.Series(dtype=float)

    # HC #718 R3: Transaction costs — 5 bps per leg on turnover
    COST_BPS = 5

    portfolio_returns = []
    current_holdings = {}
    high_water = {}
    days_since_rebal = rebal_freq

    for date in test_dates:
        daily_ret = 0.0
        days_since_rebal += 1

        # Absolute momentum filter
        spy_loc = spy_prices.index.get_indexer([date], method='ffill')[0]
        in_uptrend = True
        if spy_loc >= sma_window:
            spy_sma = spy_prices.iloc[spy_loc-sma_window:spy_loc+1, 0].mean()
            spy_price = spy_prices.iloc[spy_loc, 0]
            in_uptrend = spy_price > spy_sma

        if not in_uptrend:
            # HC #718 R3: cost to liquidate when going to cash
            if current_holdings:
                daily_ret -= COST_BPS / 10000
            current_holdings = {}
            high_water = {}
            portfolio_returns.append(daily_ret)
            continue

        # Check trailing stops
        exits = []
        for ticker in list(current_holdings.keys()):
            if ticker not in stock_prices.columns:
                continue
            loc = stock_prices.index.get_indexer([date], method='ffill')[0]
            if loc < 1:
                continue
            price = stock_prices[ticker].iloc[loc]
            if pd.isna(price):
                continue
            if ticker not in high_water or price > high_water[ticker]:
                high_water[ticker] = price
            drawdown = (high_water[ticker] - price) / high_water[ticker]
            if drawdown > trailing_stop_pct:
                exits.append(ticker)

        if exits and current_holdings:
            exit_frac = len(exits) / len(current_holdings)
            daily_ret -= exit_frac * COST_BPS / 10000

        for t in exits:
            del current_holdings[t]
            if t in high_water:
                del high_water[t]

        # Monthly rebalance
        if days_since_rebal >= rebal_freq:
            ranked_sectors = rank_sectors_by_momentum(etf_prices, date)
            if ranked_sectors:
                old_holdings_set = set(current_holdings.keys())
                top_sector_etfs = [r[0] for r in ranked_sectors[:n_sectors]]
                top_sector_names = [SECTOR_ETFS[e] for e in top_sector_etfs if e in SECTOR_ETFS]

                new_holdings = {}
                for sector_name in top_sector_names:
                    top_stocks = rank_stocks_in_sector(stock_prices, sector_name, date,
                                                       top_n=n_stocks_per_sector)
                    for ticker, score in top_stocks:
                        if score > 0:  # positive momentum only
                            new_holdings[ticker] = {'sector': sector_name, 'date': date}

                if new_holdings:
                    # HC #718 R3: transaction costs on rebalance turnover
                    new_set = set(new_holdings.keys())
                    if old_holdings_set:
                        changed = len(old_holdings_set.symmetric_difference(new_set))
                        turnover_frac = changed / max(len(old_holdings_set), len(new_set))
                        daily_ret -= turnover_frac * COST_BPS / 10000
                    else:
                        daily_ret -= COST_BPS / 10000
                    current_holdings = new_holdings
                    high_water = {}
                    for ticker in current_holdings:
                        loc = stock_prices.index.get_indexer([date], method='ffill')[0]
                        if loc >= 0 and ticker in stock_prices.columns:
                            price = stock_prices[ticker].iloc[loc]
                            if not pd.isna(price):
                                high_water[ticker] = price
                days_since_rebal = 0

        # Calculate daily return
        if current_holdings:
            weight = 1.0 / len(current_holdings)
            for ticker in current_holdings:
                if ticker not in stock_prices.columns:
                    continue
                loc = stock_prices.index.get_indexer([date], method='ffill')[0]
                if loc >= 1:
                    price = stock_prices[ticker].iloc[loc]
                    prev = stock_prices[ticker].iloc[loc-1]
                    if not pd.isna(price) and not pd.isna(prev) and prev > 0:
                        ret = price / prev - 1
                        daily_ret += weight * ret

        portfolio_returns.append(daily_ret)

    return pd.Series(portfolio_returns, index=test_dates)


###############################################################################
# 4. WALK-FORWARD ENGINE
###############################################################################

def generate_wf_windows(start_date, end_date, train_months=60, test_months=12):
    """Generate walk-forward windows: 60-month train, 12-month test, sliding."""
    windows = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    while True:
        train_start = current
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            # Partial last window
            if test_start < end:
                test_end = end
                windows.append((train_start, train_end, test_start, test_end))
            break

        windows.append((train_start, train_end, test_start, test_end))
        current += pd.DateOffset(months=test_months)  # slide by test_months

    return windows


def run_walk_forward(strategy_name, etf_prices, stock_prices, spy_prices, params):
    """Run walk-forward backtest for a given strategy."""
    data_start = max(etf_prices.index[0], spy_prices.index[0])
    data_end = min(etf_prices.index[-1], spy_prices.index[-1])

    windows = generate_wf_windows(data_start, data_end, train_months=60, test_months=12)
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"Walk-forward windows: {len(windows)}")
    print(f"Data range: {data_start.date()} to {data_end.date()}")
    print(f"{'='*60}")

    all_returns = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        print(f"  Window {i+1}/{len(windows)}: train {tr_s.date()}-{tr_e.date()}, "
              f"test {te_s.date()}-{te_e.date()}")

        if strategy_name == 'sector_momentum':
            rets = strategy_sector_momentum(etf_prices, spy_prices,
                                            tr_s, tr_e, te_s, te_e, params)
        elif strategy_name == 'dual_momentum':
            rets = strategy_dual_momentum(etf_prices, spy_prices,
                                          tr_s, tr_e, te_s, te_e, params)
        elif strategy_name == 'stock_within_sector':
            rets = strategy_stock_within_sector(etf_prices, stock_prices, spy_prices,
                                                tr_s, tr_e, te_s, te_e, params)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        if len(rets) > 0:
            all_returns.append(rets)

    if not all_returns:
        return pd.Series(dtype=float)

    combined = pd.concat(all_returns)
    # Remove duplicates (overlapping windows)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()
    return combined


###############################################################################
# 5. PERFORMANCE ANALYTICS
###############################################################################

def compute_metrics(returns, name='Strategy'):
    """Compute comprehensive performance metrics."""
    if len(returns) == 0:
        return {}

    returns = returns.fillna(0)
    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1

    # CAGR
    n_years = len(returns) / 252
    if n_years > 0 and cum.iloc[-1] > 0:
        cagr = (cum.iloc[-1]) ** (1/n_years) - 1
    else:
        cagr = 0

    # Sharpe (annualized, RF=0)
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = returns.mean() / downside.std() * np.sqrt(252)
    else:
        sortino = 0

    # Max Drawdown
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate (daily)
    trading_days = returns[returns != 0]
    if len(trading_days) > 0:
        wr = (trading_days > 0).mean()
    else:
        wr = 0

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Monthly turnover estimate (rebalance monthly = ~100% turnover/month for concentrated)
    monthly_rets = returns.resample('ME').apply(lambda x: (1+x).prod()-1)
    avg_monthly_ret = monthly_rets.mean()

    # Days in market
    invested_days = (returns != 0).sum()
    total_days = len(returns)
    pct_invested = invested_days / total_days if total_days > 0 else 0

    metrics = {
        'name': name,
        'total_return': total_ret,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'win_rate': wr,
        'profit_factor': pf,
        'calmar': calmar,
        'n_years': n_years,
        'pct_invested': pct_invested,
        'avg_daily_return': returns.mean(),
        'daily_vol': returns.std(),
        'total_days': total_days,
        'invested_days': invested_days,
    }
    return metrics


def regime_analysis(returns, spy_prices):
    """Split performance by SPY regime (above/below 200 SMA)."""
    spy = spy_prices.iloc[:, 0] if isinstance(spy_prices, pd.DataFrame) else spy_prices
    sma200 = spy.rolling(200).mean()

    common_idx = returns.index.intersection(spy.index)
    if len(common_idx) == 0:
        return None

    returns_aligned = returns.reindex(common_idx).fillna(0)
    spy_aligned = spy.reindex(common_idx)
    sma_aligned = sma200.reindex(common_idx)

    bull = spy_aligned > sma_aligned
    bear = ~bull

    bull_rets = returns_aligned[bull]
    bear_rets = returns_aligned[bear]

    bull_metrics = compute_metrics(bull_rets, 'Bull (SPY > 200 SMA)')
    bear_metrics = compute_metrics(bear_rets, 'Bear (SPY < 200 SMA)')

    # Regime gap test (HC #428 R1)
    bull_sharpe = bull_metrics.get('sharpe', 0)
    bear_sharpe = bear_metrics.get('sharpe', 0)
    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    if max_sharpe > 0:
        regime_gap = abs(bull_sharpe - bear_sharpe) / max_sharpe
    else:
        regime_gap = 0

    return {
        'bull': bull_metrics,
        'bear': bear_metrics,
        'regime_gap': regime_gap,
        'gap_pass': regime_gap < 0.50,
        'bull_days': bull.sum(),
        'bear_days': bear.sum(),
    }


def permutation_test(returns, benchmark_returns=None, n_perms=1000, seed=42):
    """Permutation test: does the strategy beat a random timing null?

    Null hypothesis: randomly assigning invested/cash days produces similar Sharpe.
    We shuffle the mapping between days and invested/not-invested status via
    block permutation (21-day blocks to preserve autocorrelation).
    """
    rng = np.random.RandomState(seed)
    arr = returns.values.copy()
    n = len(arr)

    # Actual Sharpe on the invested days only
    invested_mask = arr != 0
    invested_rets = arr[invested_mask]
    if len(invested_rets) == 0 or invested_rets.std() == 0:
        actual_sharpe = 0.0
    else:
        # Full series Sharpe (including cash days as 0 return)
        actual_sharpe = arr.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0

    # Block permutation: shuffle blocks of returns to break momentum signal timing
    block_size = 21
    n_blocks = n // block_size
    if n_blocks < 3:
        n_blocks = n
        block_size = 1

    shuffled_sharpes = []
    for _ in range(n_perms):
        # Create block-shuffled version
        block_indices = list(range(n_blocks))
        rng.shuffle(block_indices)
        shuffled = np.zeros(n)
        for j, bi in enumerate(block_indices):
            src_start = bi * block_size
            dst_start = j * block_size
            length = min(block_size, n - src_start, n - dst_start)
            shuffled[dst_start:dst_start+length] = arr[src_start:src_start+length]
        # Fill remainder
        remainder_start = n_blocks * block_size
        if remainder_start < n:
            shuffled[remainder_start:] = arr[remainder_start:]

        s = shuffled.mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
        shuffled_sharpes.append(s)

    shuffled_sharpes = np.array(shuffled_sharpes)
    p_value = (shuffled_sharpes >= actual_sharpe).mean()

    # Excess Sharpe over benchmark
    excess_sharpe = None
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        common = returns.index.intersection(benchmark_returns.index)
        if len(common) > 100:
            excess = returns.reindex(common).fillna(0) - benchmark_returns.reindex(common).fillna(0)
            excess_sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    return {
        'actual_sharpe': actual_sharpe,
        'p_value': p_value,
        'significant_5pct': p_value < 0.05,
        'perm_sharpe_mean': shuffled_sharpes.mean(),
        'perm_sharpe_std': shuffled_sharpes.std(),
        'excess_sharpe_vs_benchmark': excess_sharpe,
    }


def print_metrics(metrics, prefix=''):
    """Pretty-print a metrics dict."""
    if not metrics:
        print(f"{prefix}No metrics available")
        return
    print(f"{prefix}{'='*50}")
    print(f"{prefix}{metrics.get('name', 'Strategy')}")
    print(f"{prefix}{'='*50}")
    print(f"{prefix}  CAGR:           {metrics.get('cagr', 0)*100:>8.2f}%")
    print(f"{prefix}  Total Return:   {metrics.get('total_return', 0)*100:>8.2f}%")
    print(f"{prefix}  Sharpe:         {metrics.get('sharpe', 0):>8.3f}")
    print(f"{prefix}  Sortino:        {metrics.get('sortino', 0):>8.3f}")
    print(f"{prefix}  Max Drawdown:   {metrics.get('max_drawdown', 0)*100:>8.2f}%")
    print(f"{prefix}  Win Rate:       {metrics.get('win_rate', 0)*100:>8.2f}%")
    print(f"{prefix}  Profit Factor:  {metrics.get('profit_factor', 0):>8.3f}")
    print(f"{prefix}  Calmar:         {metrics.get('calmar', 0):>8.3f}")
    print(f"{prefix}  % Invested:     {metrics.get('pct_invested', 0)*100:>8.1f}%")
    print(f"{prefix}  Period:         {metrics.get('n_years', 0):>8.1f} years")
    print(f"{prefix}  Trading Days:   {metrics.get('total_days', 0):>8d}")


###############################################################################
# 6. SPY BUY-AND-HOLD BENCHMARK
###############################################################################

def spy_benchmark(spy_prices, start_date, end_date):
    """Compute SPY buy-and-hold returns for comparison."""
    spy = spy_prices.iloc[:, 0] if isinstance(spy_prices, pd.DataFrame) else spy_prices
    spy = spy.loc[start_date:end_date]
    returns = spy.pct_change().dropna()
    return returns


###############################################################################
# 7. MAIN
###############################################################################

def main():
    t0 = time.time()
    print("=" * 70)
    print("SECTOR/FACTOR ROTATION GROWTH STRATEGY BACKTESTER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    etf_prices, stock_prices, spy_prices = load_all_data()

    # Strategy parameters
    params = {
        'sector_momentum': {
            'n_sectors': 4,
            'trailing_stop': 0.08,
            'rebal_freq': 21,
        },
        'dual_momentum': {
            'n_sectors': 3,
            'trailing_stop': 0.10,
            'rebal_freq': 21,
            'sma_window': 200,
        },
        'stock_within_sector': {
            'n_sectors': 3,
            'n_stocks': 5,
            'trailing_stop': 0.12,
            'rebal_freq': 21,
            'sma_window': 200,
        },
    }

    results = {}
    strat_returns = {}

    # Run all three strategies (collect returns first)
    for strat_name in ['sector_momentum', 'dual_momentum', 'stock_within_sector']:
        try:
            returns = run_walk_forward(strat_name, etf_prices, stock_prices,
                                       spy_prices, params[strat_name])
            if len(returns) == 0:
                print(f"  WARNING: {strat_name} produced no returns")
                continue
            strat_returns[strat_name] = returns
        except Exception as e:
            print(f"  ERROR in {strat_name}: {e}")
            traceback.print_exc()

    # SPY benchmark (compute early so we can pass to permutation tests)
    spy_rets = None
    if strat_returns:
        first_rets = list(strat_returns.values())[0]
        spy_start = first_rets.index[0]
        spy_end = first_rets.index[-1]
        spy_rets = spy_benchmark(spy_prices, spy_start, spy_end)
        spy_metrics = compute_metrics(spy_rets, 'SPY Buy & Hold')
        results['spy_benchmark'] = {
            'returns': spy_rets,
            'metrics': spy_metrics,
        }

    # Now compute metrics, regime analysis, permutation tests
    for strat_name, returns in strat_returns.items():
        metrics = compute_metrics(returns, strat_name)
        regime = regime_analysis(returns, spy_prices)
        perm = permutation_test(returns, benchmark_returns=spy_rets, n_perms=1000)

        results[strat_name] = {
            'returns': returns,
            'metrics': metrics,
            'regime': regime,
            'permutation': perm,
        }

        print_metrics(metrics)

        # Regime analysis
        if regime:
            print(f"\n  REGIME ANALYSIS:")
            print(f"    Bull Sharpe: {regime['bull'].get('sharpe', 0):.3f} "
                  f"({regime['bull_days']} days)")
            print(f"    Bear Sharpe: {regime['bear'].get('sharpe', 0):.3f} "
                  f"({regime['bear_days']} days)")
            print(f"    Regime Gap:  {regime['regime_gap']:.3f} "
                  f"({'PASS' if regime['gap_pass'] else 'FAIL'} < 0.50)")
            # Note: strategies with absolute momentum filter go to cash in bear markets
            # so bear Sharpe = 0 is by design, not a flaw
            if regime['bear'].get('sharpe', 0) == 0 and regime['bear'].get('pct_invested', 0) < 0.05:
                print(f"    Note: Strategy goes to CASH in bear regime (by design)")

        # Permutation test
        print(f"\n  PERMUTATION TEST (1000 circular shifts):")
        print(f"    Actual Sharpe: {perm['actual_sharpe']:.3f}")
        print(f"    p-value:       {perm['p_value']:.4f} "
              f"({'SIGNIFICANT' if perm['significant_5pct'] else 'NOT significant'} at 5%)")
        if perm.get('excess_sharpe_vs_benchmark') is not None:
            print(f"    Excess Sharpe vs SPY: {perm['excess_sharpe_vs_benchmark']:.3f}")

    # Print SPY benchmark
    if 'spy_benchmark' in results:
        print_metrics(results['spy_benchmark']['metrics'])

    # HEAD-TO-HEAD COMPARISON
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 70)

    header = f"{'Strategy':<25} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'WR':>8} {'PF':>8} {'Calmar':>8} {'%Inv':>6}"
    print(header)
    print("-" * len(header))

    for name, res in results.items():
        m = res['metrics']
        line = (f"{m['name']:<25} "
                f"{m['cagr']*100:>7.2f}% "
                f"{m['sharpe']:>8.3f} "
                f"{m['sortino']:>8.3f} "
                f"{m['max_drawdown']*100:>7.2f}% "
                f"{m['win_rate']*100:>7.1f}% "
                f"{m['profit_factor']:>8.3f} "
                f"{m['calmar']:>8.3f} "
                f"{m['pct_invested']*100:>5.1f}%")
        print(line)

    # Regime gap summary
    print(f"\n{'Strategy':<25} {'Bull Sharpe':>12} {'Bear Sharpe':>12} {'Gap':>8} {'Pass':>6}")
    print("-" * 65)
    for name, res in results.items():
        if 'regime' in res and res['regime']:
            r = res['regime']
            print(f"{name:<25} "
                  f"{r['bull'].get('sharpe',0):>12.3f} "
                  f"{r['bear'].get('sharpe',0):>12.3f} "
                  f"{r['regime_gap']:>8.3f} "
                  f"{'YES' if r['gap_pass'] else 'NO':>6}")

    # Statistical significance summary
    print(f"\n{'Strategy':<25} {'Sharpe':>8} {'p-value':>8} {'Significant':>12}")
    print("-" * 55)
    for name, res in results.items():
        if 'permutation' in res:
            p = res['permutation']
            print(f"{name:<25} "
                  f"{p['actual_sharpe']:>8.3f} "
                  f"{p['p_value']:>8.4f} "
                  f"{'YES' if p['significant_5pct'] else 'NO':>12}")

    # Save results
    print(f"\nSaving results to {OUTPUT_DIR}...")

    # Save returns as CSV
    returns_df = pd.DataFrame({name: res['returns'] for name, res in results.items()})
    returns_df.to_csv(os.path.join(OUTPUT_DIR, 'daily_returns.csv'))

    # Save metrics as JSON
    metrics_out = {}
    for name, res in results.items():
        m = res['metrics'].copy()
        metrics_out[name] = m
        if 'regime' in res and res['regime']:
            r = res['regime']
            metrics_out[name]['regime_gap'] = r['regime_gap']
            metrics_out[name]['regime_gap_pass'] = r['gap_pass']
            metrics_out[name]['bull_sharpe'] = r['bull'].get('sharpe', 0)
            metrics_out[name]['bear_sharpe'] = r['bear'].get('sharpe', 0)
        if 'permutation' in res:
            metrics_out[name]['perm_p_value'] = res['permutation']['p_value']
            metrics_out[name]['perm_significant'] = res['permutation']['significant_5pct']

    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2, default=str)

    # Save equity curves
    equity_df = pd.DataFrame()
    for name, res in results.items():
        equity_df[name] = (1 + res['returns']).cumprod()
    equity_df.to_csv(os.path.join(OUTPUT_DIR, 'equity_curves.csv'))

    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(3, 1, figsize=(14, 16))

        # Equity curves
        ax = axes[0]
        for name in equity_df.columns:
            ax.plot(equity_df.index, equity_df[name], label=name, linewidth=1.5)
        ax.set_title('Sector/Factor Rotation Strategies — Equity Curves (Walk-Forward OOT)')
        ax.set_ylabel('Growth of $1')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Drawdowns
        ax = axes[1]
        for name, res in results.items():
            cum = (1 + res['returns']).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            ax.fill_between(dd.index, dd.values, 0, alpha=0.3, label=name)
        ax.set_title('Drawdowns')
        ax.set_ylabel('Drawdown %')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        # Rolling 12-month returns
        ax = axes[2]
        for name, res in results.items():
            rolling = res['returns'].rolling(252).apply(lambda x: (1+x).prod()-1, raw=True)
            ax.plot(rolling.index, rolling * 100, label=name, linewidth=1)
        ax.set_title('Rolling 12-Month Returns')
        ax.set_ylabel('Return %')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'sector_momentum_results.png'), dpi=150)
        plt.close()
        print("  Plot saved.")
    except Exception as e:
        print(f"  Plot error: {e}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
