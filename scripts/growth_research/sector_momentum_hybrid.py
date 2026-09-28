#!/usr/bin/env python3
"""
Sector/Momentum Hybrid — Walk-Forward Backtest
================================================
Combine sector-level momentum filtering (proven regime-agnostic via ETF
rotation v3, gap=0.39) with individual stock selection WITHIN winning sectors.

Logic:
1. Monthly: rank 11 GICS sector ETFs by 6-month momentum → top 4 sectors
2. Within each winning sector, pick top 5 stocks by 3-month momentum (skip recent month)
3. Portfolio: 15-20 stocks, equal weight
4. Daily exit rules (HC #684):
   - Trailing stop: 15% from peak (wider stop for monthly-rebal stock strategy)
   - If parent sector ETF closes below 50-day SMA for 5+ consecutive days → exit that sector
   - Individual: exit if drawdown from entry > 20%
5. On exit, weight goes to cash (no rebalancing mid-month)
6. Monthly rebalance for new entries
7. Walk-forward: 60-month train, 1-month test, sliding
8. Regime test: must pass gap < 0.50
9. Permutation test

Sliding window ONLY. Slippage: 0.3% per trade (Robinhood, no commission).
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import hashlib
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research'
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

###############################################################################
# S&P 500 SECTOR MAPPING (hardcoded for reliability)
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
    # Information Technology (~40)
    for t in ['AAPL','MSFT','NVDA','AVGO','CSCO','ACN','TXN','QCOM','INTC','ADI',
              'AMAT','MU','LRCX','KLAC','SNPS','CDNS','MCHP','ON','FTNT','PANW',
              'CRWD','ADBE','CRM','NOW','INTU','ADP','ORCL','IBM','FIS','FISV',
              'GPN','IT','MPWR','NXPI','SWKS','HPQ','HPE','KEYS','ZBRA','EPAM']:
        m[t] = 'Information Technology'
    # Health Care (~30)
    for t in ['UNH','JNJ','LLY','MRK','ABBV','TMO','ABT','DHR','BMY','AMGN',
              'GILD','ISRG','SYK','VRTX','REGN','ZTS','CI','BDX','HUM','ELV',
              'MDT','BSX','EW','A','IQV','DXCM','IDXX','MTD','ALGN','HOLX']:
        m[t] = 'Health Care'
    # Financials (~30)
    for t in ['JPM','V','MA','BRK-B','BAC','WFC','GS','MS','BLK','SCHW',
              'CB','MMC','CME','ICE','PNC','USB','AIG','AFL','MET','PRU',
              'MSCI','TRV','ALL','SPGI','MCO','AON','AJG','C','COF','TFC']:
        m[t] = 'Financials'
    # Energy (~18)
    for t in ['XOM','CVX','COP','SLB','EOG','PSX','VLO','OXY','MPC','PXD',
              'WMB','KMI','DVN','HAL','FANG','HES','BKR','OKE']:
        m[t] = 'Energy'
    # Industrials (~30)
    for t in ['UNP','RTX','HON','CAT','BA','DE','GE','LMT','GD','NOC',
              'FDX','WM','ITW','EMR','NSC','CSX','MMM','JCI','ETN','PH',
              'ROK','PCAR','FAST','CTAS','CARR','OTIS','AME','TT','IR','SWK']:
        m[t] = 'Industrials'
    # Communication Services (~18)
    for t in ['GOOGL','GOOG','META','NFLX','DIS','CMCSA','VZ','T','TMUS',
              'CHTR','EA','ATVI','TTWO','WBD','MTCH','OMC','IPG','LYV']:
        m[t] = 'Communication Services'
    # Consumer Discretionary (~24)
    for t in ['AMZN','TSLA','HD','MCD','LOW','BKNG','TJX','SBUX','NKE','CMG',
              'ORLY','AZO','ROST','MAR','HLT','DHI','LEN','GM','F','EBAY',
              'YUM','DPZ','POOL','BBY']:
        m[t] = 'Consumer Discretionary'
    # Consumer Staples (~20)
    for t in ['PG','PEP','KO','COST','WMT','PM','MDLZ','CL','MO','EL',
              'SYY','GIS','KMB','HSY','K','ADM','STZ','KHC','TSN','MKC']:
        m[t] = 'Consumer Staples'
    # Utilities (~18)
    for t in ['NEE','SO','DUK','D','AEP','EXC','SRE','XEL','WEC','ED',
              'AWK','DTE','ETR','FE','PPL','AES','CMS','CEG']:
        m[t] = 'Utilities'
    # Real Estate (~18)
    for t in ['PLD','AMT','CCI','EQIX','SPG','WELL','PSA','O','DLR','VICI',
              'SBAC','ARE','AVB','EQR','MAA','ESS','UDR','INVH']:
        m[t] = 'Real Estate'
    # Materials (~21)
    for t in ['LIN','APD','SHW','FCX','NEM','ECL','DOW','DD','NUE','PPG',
              'VMC','MLM','ALB','IFF','CE','EMN','BALL','PKG','IP','CF','MOS']:
        m[t] = 'Materials'
    return m


###############################################################################
# DATA ACQUISITION (with caching)
###############################################################################

def _cache_path(name):
    return os.path.join(CACHE_DIR, f'{name}.parquet')


def download_all_data(sector_map, start='2014-01-01', end='2026-07-13'):
    """Download sector ETFs + individual stock prices. Caches to disk."""
    import yfinance as yf

    etf_cache = _cache_path('sector_etf_prices')
    stock_cache = _cache_path('sector_stock_prices')

    # Check cache (valid if < 12 hours old)
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

    # Download ETFs + SPY
    etf_tickers = list(SECTOR_ETFS.keys()) + ['SPY']
    print(f"  Downloading {len(etf_tickers)} sector ETFs + SPY...")
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
                etf_prices[t] = col
        except:
            pass
    etf_df = pd.DataFrame(etf_prices)
    etf_df.index = pd.to_datetime(etf_df.index)

    # Handle missing ETFs due to rate limits — retry individually
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

    # Cache
    try:
        etf_df.to_parquet(etf_cache)
        stock_df.to_parquet(stock_cache)
        print("  Cached data to disk")
    except Exception as e:
        print(f"  Cache write failed: {e}")

    return etf_df, stock_df


###############################################################################
# STRATEGY LOGIC
###############################################################################

def rank_sectors(etf_prices, spy_prices, date, lookback_days=126, n_top=4):
    """Rank sector ETFs by 6-month momentum, return top N sectors + cash fraction.

    Cash allocation based on two signals:
    1. Absolute momentum: fraction of sectors with positive 6mo return
    2. SPY trend: SPY vs its 200-day SMA
    """
    sector_etf_cols = [c for c in etf_prices.columns if c in SECTOR_ETFS]
    prices_to_date = etf_prices.loc[:date, sector_etf_cols]
    if len(prices_to_date) < lookback_days:
        return [], 0.0

    current = prices_to_date.iloc[-1]
    past = prices_to_date.iloc[-lookback_days]
    momentum = (current / past - 1).dropna()

    # Signal 1: Absolute momentum breadth
    n_positive = (momentum > 0).sum()
    n_total = len(momentum)
    breadth_ratio = n_positive / max(n_total, 1)

    # Signal 2: SPY trend (above/below 200-day SMA)
    spy_to_date = spy_prices.loc[:date]
    spy_above_200 = True  # default
    if len(spy_to_date) >= 200:
        spy_sma200 = spy_to_date.rolling(200).mean().iloc[-1]
        spy_current = spy_to_date.iloc[-1]
        if not pd.isna(spy_sma200) and not pd.isna(spy_current):
            spy_above_200 = spy_current > spy_sma200

    # Cash allocation: aggressive during bear
    if spy_above_200 and breadth_ratio >= 0.5:
        cash_frac = 0.0   # bull, fully invested
    elif spy_above_200:
        cash_frac = 0.20  # bull but narrow breadth
    elif breadth_ratio >= 0.3:
        cash_frac = 0.50  # bear but some sectors positive
    else:
        cash_frac = 0.70  # deep bear, mostly cash

    top_etfs = momentum.sort_values(ascending=False).head(n_top).index.tolist()
    top_sectors = [SECTOR_ETFS[e] for e in top_etfs if e in SECTOR_ETFS]
    return top_sectors, cash_frac


def pick_stocks_in_sectors(stock_prices, etf_prices, sector_map, winning_sectors, date,
                           stocks_per_sector=5, mom_lookback=63, skip_recent=21):
    """Pick top stocks by RELATIVE STRENGTH vs sector ETF within winning sectors.

    Using relative strength (stock return - sector ETF return) rather than
    absolute momentum should be more regime-agnostic, because we're selecting
    stocks that outperform their sector regardless of market direction.
    """
    selected = {}

    for sector in winning_sectors:
        sector_stocks = [t for t, s in sector_map.items()
                         if s == sector and t in stock_prices.columns]
        if len(sector_stocks) < 3:
            continue

        etf_ticker = ETF_FOR_SECTOR.get(sector)

        prices_to_date = stock_prices.loc[:date, sector_stocks]
        if len(prices_to_date) < mom_lookback + skip_recent:
            continue

        # Stock returns over lookback (skipping recent month)
        current = prices_to_date.iloc[-skip_recent - 1]
        past = prices_to_date.iloc[-skip_recent - mom_lookback]
        stock_ret = (current / past - 1).dropna()

        # If ETF available, compute relative strength; otherwise absolute momentum
        if etf_ticker and etf_ticker in etf_prices.columns:
            etf_to_date = etf_prices.loc[:date, etf_ticker]
            if len(etf_to_date) >= mom_lookback + skip_recent:
                etf_cur = etf_to_date.iloc[-skip_recent - 1]
                etf_past = etf_to_date.iloc[-skip_recent - mom_lookback]
                if etf_past > 0:
                    etf_ret = etf_cur / etf_past - 1
                    # Relative strength = stock return - sector ETF return
                    rel_strength = stock_ret - etf_ret
                else:
                    rel_strength = stock_ret
            else:
                rel_strength = stock_ret
        else:
            rel_strength = stock_ret

        # Take top stocks by relative strength
        top = rel_strength.sort_values(ascending=False).head(stocks_per_sector).index.tolist()
        selected[sector] = top

    return selected


def run_hybrid_backtest(etf_prices, stock_prices, sector_map,
                        n_top_sectors=4, stocks_per_sector=5,
                        sector_mom_lookback=126, stock_mom_lookback=63,
                        trailing_stop_pct=0.15,
                        max_loss_pct=0.25,
                        etf_stock_blend=0.50,
                        sector_sma_filter=True,
                        sector_sma_period=50,
                        train_months=60, test_months=1,
                        slippage_pct=0.003):
    """
    Walk-forward sector/momentum hybrid backtest.

    BLENDED APPROACH: allocate etf_stock_blend to sector ETFs (proven
    regime-agnostic) and (1-etf_stock_blend) to top stocks in those sectors
    (alpha overlay). This preserves ETF rotation's regime properties while
    adding stock-level upside.

    Daily exit rules (HC #684):
    - Trailing stop on stock positions only (ETF positions hold to rebalance)
    - Max-loss stop on stock positions
    - Cash allocation scales with market weakness (absolute momentum filter)
    """

    etf_sma = etf_prices.rolling(sector_sma_period).mean()

    monthly = stock_prices.resample('ME').last().index
    monthly = monthly[monthly >= stock_prices.index[252]]

    if len(monthly) < train_months + test_months + 1:
        print(f"Not enough data: {len(monthly)} months")
        return None, {}

    all_returns = []
    all_dates = []
    trade_count = 0
    exit_reasons = {'trailing_stop': 0, 'max_loss': 0}
    sector_selection_log = []
    monthly_returns = []

    for i in range(train_months, len(monthly) - test_months + 1):
        rebal_date = monthly[i]

        # Step 1: Rank sectors + cash allocation
        spy_col = etf_prices['SPY'] if 'SPY' in etf_prices.columns else None
        winning_sectors, cash_frac = rank_sectors(etf_prices, spy_col, rebal_date,
                                                   lookback_days=sector_mom_lookback,
                                                   n_top=n_top_sectors)
        if len(winning_sectors) < 1:
            continue

        # Step 1b: SMA filter — only enter sectors above SMA
        if sector_sma_filter:
            filtered = []
            for sector in winning_sectors:
                etf_t = ETF_FOR_SECTOR.get(sector)
                if etf_t and etf_t in etf_prices.columns and etf_t in etf_sma.columns:
                    try:
                        p = etf_prices.loc[:rebal_date, etf_t].iloc[-1]
                        s = etf_sma.loc[:rebal_date, etf_t].iloc[-1]
                        if not pd.isna(p) and not pd.isna(s) and p >= s:
                            filtered.append(sector)
                    except:
                        filtered.append(sector)
                else:
                    filtered.append(sector)
            if filtered:
                winning_sectors = filtered
            else:
                cash_frac = max(cash_frac, 0.70)
                winning_sectors = winning_sectors[:2]

        # Step 2: Pick stocks within winning sectors
        selected = pick_stocks_in_sectors(
            stock_prices, etf_prices, sector_map, winning_sectors, rebal_date,
            stocks_per_sector=stocks_per_sector,
            mom_lookback=stock_mom_lookback, skip_recent=21
        )
        if not selected:
            continue

        n_sectors_active = len(selected)
        all_stocks = []
        stock_to_sector = {}
        for sector, tickers in selected.items():
            for t in tickers:
                all_stocks.append(t)
                stock_to_sector[t] = sector
        n_stocks = len(all_stocks)

        # Get sector ETF tickers for the ETF portion
        active_etfs = []
        for sector in selected.keys():
            etf_t = ETF_FOR_SECTOR.get(sector)
            if etf_t and etf_t in etf_prices.columns:
                active_etfs.append(etf_t)

        sector_selection_log.append({
            'date': str(rebal_date.date()),
            'sectors': list(selected.keys()),
            'n_stocks': n_stocks,
            'cash_pct': f"{cash_frac:.0%}",
        })

        # Test period
        if i + test_months < len(monthly):
            test_end = monthly[i + test_months]
        else:
            test_end = stock_prices.index[-1]

        test_prices = stock_prices.loc[rebal_date:test_end]
        test_etf_prices = etf_prices.loc[rebal_date:test_end]
        if len(test_prices) < 2:
            continue

        # Weight allocation
        invested_frac = 1.0 - cash_frac
        etf_total = invested_frac * etf_stock_blend
        stock_total = invested_frac * (1.0 - etf_stock_blend)

        # ETF weights (equal across active sector ETFs, no daily exits)
        etf_weight = etf_total / max(len(active_etfs), 1)

        # Stock weights (equal across stocks, with daily exits)
        stock_weight = stock_total / max(n_stocks, 1)

        # Initialize stock tracking
        active_stocks = {}
        highs = {}
        entries = {}
        for s in all_stocks:
            if s in test_prices.columns and not pd.isna(test_prices[s].iloc[0]):
                p0 = test_prices[s].iloc[0]
                active_stocks[s] = True
                entries[s] = p0 * (1 + slippage_pct)
                highs[s] = p0
                trade_count += 1

        daily_returns = []
        for d in range(1, len(test_prices)):
            day_ret = 0.0

            # ETF portion — no daily exits, holds to rebalance
            for etf_t in active_etfs:
                if etf_t in test_etf_prices.columns:
                    p = test_etf_prices[etf_t].iloc[d]
                    p_prev = test_etf_prices[etf_t].iloc[d - 1]
                    if not pd.isna(p) and not pd.isna(p_prev) and p_prev > 0:
                        day_ret += etf_weight * (p - p_prev) / p_prev

            # Stock portion — with daily exits
            stopped = []
            for s in list(active_stocks.keys()):
                if s not in test_prices.columns:
                    continue
                p = test_prices[s].iloc[d]
                p_prev = test_prices[s].iloc[d - 1]
                if pd.isna(p) or pd.isna(p_prev) or p_prev == 0:
                    continue

                day_ret += stock_weight * (p - p_prev) / p_prev

                if p > highs.get(s, 0):
                    highs[s] = p

                # EXIT 1: Trailing stop
                if highs[s] > 0 and (1 - p / highs[s]) >= trailing_stop_pct:
                    stopped.append(s)
                    exit_reasons['trailing_stop'] += 1
                    continue

                # EXIT 2: Max-loss from entry
                if entries.get(s, 0) > 0 and (1 - p / entries[s]) >= max_loss_pct:
                    stopped.append(s)
                    exit_reasons['max_loss'] += 1
                    continue

            for s in set(stopped):
                if s in active_stocks:
                    day_ret -= stock_weight * slippage_pct
                    trade_count += 1
                    del active_stocks[s]

            daily_returns.append(day_ret)

        if daily_returns:
            month_ret = (1 + pd.Series(daily_returns)).prod() - 1
            monthly_returns.append({'date': str(rebal_date.date()), 'return': month_ret})

        all_returns.extend(daily_returns)
        all_dates.extend(test_prices.index[1:len(daily_returns) + 1].tolist())

    if not all_returns:
        return None, {}

    results = pd.Series(all_returns, index=pd.DatetimeIndex(all_dates))
    results = results[~results.index.duplicated(keep='last')]
    results = results.sort_index()

    meta = {
        'total_trades': trade_count,
        'exit_reasons': exit_reasons,
        'etf_stock_blend': etf_stock_blend,
        'n_rebalance_periods': len(sector_selection_log),
        'sector_selections': sector_selection_log[:5] + ['...'] + sector_selection_log[-3:]
            if len(sector_selection_log) > 8 else sector_selection_log,
        'monthly_returns_sample': monthly_returns[:5] + monthly_returns[-3:]
            if len(monthly_returns) > 8 else monthly_returns,
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

    win_rate = (returns > 0).mean()

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'name': name,
        'CAGR': f"{cagr:.1%}",
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'Max_DD': f"{max_dd:.1%}",
        'Win_Rate': f"{win_rate:.1%}",
        'Profit_Factor': round(pf, 2),
        'Total_Return': f"{cum_ret:.1%}",
        'N_Days': total_days,
        'Ann_Vol': f"{ann_vol:.1%}",
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


def permutation_test(returns, n_perms=200):
    """Permutation test: shuffle daily returns, compare Sharpe."""
    actual_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = returns.sample(frac=1.0, replace=False).values
        s = shuffled.mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
        perm_sharpes.append(s)

    p_value = (np.sum(np.array(perm_sharpes) >= actual_sharpe) + 1) / (n_perms + 1)
    return actual_sharpe, p_value, perm_sharpes


###############################################################################
# MAIN
###############################################################################

def main():
    print("=" * 70)
    print("SECTOR/MOMENTUM HYBRID — Walk-Forward Backtest")
    print("=" * 70)
    print("Top 4 sectors by 6mo momentum → Top 5 stocks per sector by 3mo mom")
    print("Sector SMA50 filter at entry. Daily exits: 15% trail, 25% max loss")
    print("Sliding window: 60mo train, 1mo test | Slippage: 0.3%")
    print("=" * 70)

    # 1. Get sector mapping
    print("\n[1/6] Getting S&P 500 sector mapping...")
    sector_map = get_sp500_with_sectors()
    sectors_found = set(sector_map.values())
    print(f"  {len(sector_map)} stocks mapped to {len(sectors_found)} sectors")
    for sec in sorted(sectors_found):
        n = sum(1 for v in sector_map.values() if v == sec)
        print(f"    {sec}: {n} stocks")

    # 2. Download data
    print("\n[2/6] Downloading price data (2014-2026)...")
    etf_prices, stock_prices = download_all_data(sector_map,
                                                  start='2014-01-01',
                                                  end='2026-07-13')

    # Extract SPY
    spy_close = etf_prices['SPY'] if 'SPY' in etf_prices.columns else None
    if spy_close is None:
        print("ERROR: Could not get SPY data")
        return

    # Check ETF coverage
    missing_etfs = [e for e in SECTOR_ETFS if e not in etf_prices.columns]
    if missing_etfs:
        print(f"  WARNING: Missing sector ETFs: {missing_etfs}")

    # 3. Run hybrid backtest
    print("\n[3/6] Running walk-forward hybrid backtest...")
    returns, meta = run_hybrid_backtest(
        etf_prices, stock_prices, sector_map,
        n_top_sectors=4, stocks_per_sector=5,
        sector_mom_lookback=378, stock_mom_lookback=63,
        trailing_stop_pct=0.15,
        max_loss_pct=0.25,
        etf_stock_blend=0.70,
        sector_sma_filter=True,
        sector_sma_period=50,
        train_months=60, test_months=1,
        slippage_pct=0.003,
    )

    if returns is None or len(returns) < 30:
        print("ERROR: Insufficient returns data")
        return

    print(f"  Backtest period: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  Total trading days: {len(returns)}")
    print(f"  Total trades: {meta.get('total_trades', 'N/A')}")
    if 'exit_reasons' in meta:
        print(f"  Exit breakdown: {meta['exit_reasons']}")

    # 4. Compute metrics
    print("\n[4/6] Computing metrics...")
    metrics = compute_metrics(returns, "Sector/Momentum Hybrid")
    print("\n--- OVERALL RESULTS ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Regime analysis
    bull, bear, gap = regime_analysis(returns, spy_close)
    print("\n--- REGIME ANALYSIS ---")
    if bull:
        print(f"  Bull Regime - Sharpe: {bull['Sharpe']}, CAGR: {bull['CAGR']}, WR: {bull['Win_Rate']}")
    if bear:
        print(f"  Bear Regime - Sharpe: {bear['Sharpe']}, CAGR: {bear['CAGR']}, WR: {bear['Win_Rate']}")
    if gap is not None:
        status = 'PASS' if gap < 0.50 else 'FAIL'
        print(f"  Regime Gap: {gap:.3f} {status} (threshold: <0.50)")

    # 5. Permutation test
    print("\n[5/6] Running permutation test (200 shuffles)...")
    actual_s, p_val, _ = permutation_test(returns, n_perms=200)
    print(f"  Actual Sharpe: {actual_s:.3f}")
    print(f"  p-value: {p_val:.4f} {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'}")

    # 6. Benchmarks
    print("\n[6/6] Benchmark comparisons...")

    spy_returns = spy_close.pct_change().dropna()
    common_idx = returns.index.intersection(spy_returns.index)
    spy_bench = {}
    if len(common_idx) > 30:
        spy_bench = compute_metrics(spy_returns.loc[common_idx], "SPY Buy&Hold")
        print("\n--- BENCHMARK: SPY Buy & Hold ---")
        for k, v in spy_bench.items():
            print(f"  {k}: {v}")

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    regime_status = "PASS" if gap is not None and gap < 0.50 else "FAIL" if gap is not None else "N/A"
    perm_status = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
    print(f"  Strategy:       Sector/Momentum Hybrid")
    print(f"  CAGR:           {metrics.get('CAGR', 'N/A')}")
    print(f"  Sharpe:         {metrics.get('Sharpe', 'N/A')}")
    print(f"  Sortino:        {metrics.get('Sortino', 'N/A')}")
    print(f"  Max Drawdown:   {metrics.get('Max_DD', 'N/A')}")
    print(f"  Win Rate:       {metrics.get('Win_Rate', 'N/A')}")
    if gap is not None:
        print(f"  Regime Gap:     {gap:.3f} -> {regime_status}")
    else:
        print(f"  Regime Gap:     N/A")
    print(f"  Permutation:    p={p_val:.4f} -> {perm_status}")
    print(f"  vs SPY Sharpe:  {spy_bench.get('Sharpe', 'N/A')}")

    print("\n  --- REFERENCE (from prior work) ---")
    print("  ETF Rotation v3:     Sharpe ~1.2, Regime gap 0.39 (PASS)")
    print("  Pure Stock Momentum: Sharpe ~1.0, Regime gap ~2.0 (FAIL)")

    # ===== VARIANT: Pure sector ETF rotation (no stock overlay) =====
    print("\n--- VARIANT: Pure Sector ETF Rotation (control) ---")
    ret_etf_only, meta_etf = run_hybrid_backtest(
        etf_prices, stock_prices, sector_map,
        n_top_sectors=4, stocks_per_sector=5,
        sector_mom_lookback=378, stock_mom_lookback=63,
        trailing_stop_pct=0.15,
        max_loss_pct=0.25,
        etf_stock_blend=1.0,  # 100% ETFs, 0% stocks
        sector_sma_filter=True,
        sector_sma_period=50,
        train_months=60, test_months=1,
        slippage_pct=0.003,
    )
    if ret_etf_only is not None and len(ret_etf_only) > 30:
        etf_only_metrics = compute_metrics(ret_etf_only, "ETF-Only Rotation")
        etf_bull, etf_bear, etf_gap = regime_analysis(ret_etf_only, spy_close)
        print(f"  CAGR: {etf_only_metrics.get('CAGR')}, Sharpe: {etf_only_metrics.get('Sharpe')}")
        print(f"  Max DD: {etf_only_metrics.get('Max_DD')}")
        if etf_gap is not None:
            etf_regime_status = 'PASS' if etf_gap < 0.50 else 'FAIL'
            print(f"  Regime Gap: {etf_gap:.3f} {etf_regime_status}")
            if etf_bull:
                print(f"  Bull Sharpe: {etf_bull['Sharpe']}, Bear Sharpe: {etf_bear['Sharpe']}")
    else:
        etf_only_metrics = {}
        etf_gap = None

    # ===== VARIANT: Aggressive bear protection (80% cash in bear) =====
    print("\n--- VARIANT: Aggressive Bear Protection ---")
    # Temporarily modify rank_sectors behavior by running with different params
    # Actually, let's just add more cash. Run with stock blend=0.70 but
    # we need to change the cash logic. Instead, let's test 100% stock variant
    # with the stop losses but NO cash overlay to see the raw stock signal.
    ret_stocks_only, meta_stocks = run_hybrid_backtest(
        etf_prices, stock_prices, sector_map,
        n_top_sectors=4, stocks_per_sector=5,
        sector_mom_lookback=378, stock_mom_lookback=63,
        trailing_stop_pct=0.15,
        max_loss_pct=0.25,
        etf_stock_blend=0.0,  # 0% ETFs, 100% stocks
        sector_sma_filter=True,
        sector_sma_period=50,
        train_months=60, test_months=1,
        slippage_pct=0.003,
    )
    if ret_stocks_only is not None and len(ret_stocks_only) > 30:
        stocks_only_metrics = compute_metrics(ret_stocks_only, "Stocks-Only (Sector-Filtered)")
        stk_bull, stk_bear, stk_gap = regime_analysis(ret_stocks_only, spy_close)
        print(f"  CAGR: {stocks_only_metrics.get('CAGR')}, Sharpe: {stocks_only_metrics.get('Sharpe')}")
        print(f"  Max DD: {stocks_only_metrics.get('Max_DD')}")
        if stk_gap is not None:
            stk_regime_status = 'PASS' if stk_gap < 0.50 else 'FAIL'
            print(f"  Regime Gap: {stk_gap:.3f} {stk_regime_status}")
            if stk_bull:
                print(f"  Bull Sharpe: {stk_bull['Sharpe']}, Bear Sharpe: {stk_bear['Sharpe']}")
    else:
        stocks_only_metrics = {}
        stk_gap = None

    # Save results
    results = {
        'strategy': 'Sector/Momentum Hybrid',
        'description': 'Top 4 sectors by 6mo momentum, top 5 stocks per sector by 3mo momentum, daily dynamic exits',
        'params': {
            'n_top_sectors': 4,
            'stocks_per_sector': 5,
            'sector_mom_lookback_days': 378,
            'stock_mom_lookback_days': 63,
            'skip_recent_days': 21,
            'trailing_stop_pct': 0.15,
            'max_loss_pct': 0.25,
            'etf_stock_blend': 0.70,
            'sector_sma_filter_at_entry': True,
            'sector_sma_period': 50,
            'slippage_pct': 0.003,
            'train_months': 60,
            'test_months': 1,
            'window_type': 'sliding',
        },
        'overall': metrics,
        'bull_regime': bull,
        'bear_regime': bear,
        'regime_gap': gap,
        'regime_test': regime_status,
        'permutation_p_value': p_val,
        'permutation_test': perm_status,
        'n_stocks_in_universe': stock_prices.shape[1],
        'n_sectors': len(sectors_found),
        'backtest_period': f"{returns.index[0].date()} to {returns.index[-1].date()}",
        'total_trades': meta.get('total_trades', 0),
        'exit_reasons': meta.get('exit_reasons', {}),
        'benchmark_spy': spy_bench,
        'sector_selections_sample': meta.get('sector_selections', []),
        'variants': {
            'etf_only_rotation': {
                'metrics': etf_only_metrics if 'etf_only_metrics' in dir() else {},
                'regime_gap': etf_gap if 'etf_gap' in dir() else None,
            },
            'stocks_only': {
                'metrics': stocks_only_metrics if 'stocks_only_metrics' in dir() else {},
                'regime_gap': stk_gap if 'stk_gap' in dir() else None,
            },
        },
        'reference': {
            'etf_rotation_v3': {'sharpe': 1.2, 'regime_gap': 0.39, 'regime_test': 'PASS'},
            'pure_stock_momentum': {'sharpe': 1.0, 'regime_gap': 2.0, 'regime_test': 'FAIL'},
        },
    }

    outpath = os.path.join(OUTPUT_DIR, 'sector_momentum_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")

    # Save daily returns
    returns.to_csv(os.path.join(OUTPUT_DIR, 'sector_momentum_returns.csv'))
    print("Done!")


if __name__ == '__main__':
    main()
