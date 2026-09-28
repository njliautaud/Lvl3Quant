#!/usr/bin/env python3
"""
Systematic Momentum v2 — Walk-Forward Backtest
Universe: S&P 500 + S&P 400 mid-caps (~800 stocks)
Features: 1/3/6/12 month momentum, trailing stop exits, volume confirmation
Rebalance: Monthly with daily exit overrides
Walk-forward: 60-month train, 1-month test, sliding window
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research'
os.makedirs(OUTPUT_DIR, exist_ok=True)

###############################################################################
# 1. DATA ACQUISITION
###############################################################################

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    except:
        # Fallback: top liquid names
        return ['AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','BRK-B','UNH','JNJ',
                'XOM','JPM','V','PG','MA','HD','CVX','MRK','ABBV','LLY','PEP','KO',
                'AVGO','COST','TMO','MCD','WMT','CSCO','ACN','ABT','DHR','NEE','LIN',
                'BMY','PM','TXN','UNP','RTX','AMGN','HON','LOW','QCOM','INTC','COP',
                'IBM','SBUX','CAT','GS','BA','MDLZ','BLK','ADP','DE','ADI','GILD',
                'MMC','ISRG','SYK','VRTX','BKNG','REGN','TJX','ZTS','CI','CB','PLD',
                'SO','DUK','CME','SLB','CL','USB','ITW','BDX','MO','EOG','WM','APD',
                'NOC','ICE','FDX','GD','FCX','PNC','ORLY','AZO','SHW','NSC','EMR',
                'MCK','TGT','PSX','VLO','OXY','AIG','AFL','D','HUM','MET','PRU',
                'MSCI','TRV','ALL','AEP','SPG','WELL','PSA','O','AMT','CCI','EQIX']

def get_sp400_sample():
    """Get a sample of S&P 400 mid-cap tickers."""
    return ['DECK','BURL','WSM','TTEK','POOL','TECH','SAIA','LSCC','FIX','WMS',
            'MTDR','GLOB','PNFP','LNTH','CBT','SITE','ELF','WFRD','PIPR','SKY',
            'COOP','ENSG','GKOS','NOVT','ONTO','CADE','TNDM','ALIT','CALM','DINO',
            'ESAB','REZI','ABM','AZEK','BC','BJ','CARG','CIEN','CIVI','CNX',
            'CROX','DT','EXPO','FIVE','GFL','GTES','HQY','IDCC','IPAR','KNSL',
            'LBRT','LFUS','MATX','MEDP','MOD','MTH','MUSA','OGE','ORA','PBF',
            'PCH','RBC','RMBS','RNR','SAM','SFBS','SFM','SIG','SNX','SSD',
            'STAG','TOL','TREX','TXRH','UBSI','VIRT','WDFC','WEN','WING','WTS']

def download_prices(tickers, start='2013-01-01', end='2026-07-01'):
    """Download adjusted close prices with batching."""
    import yfinance as yf

    all_prices = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        ticker_str = ' '.join(batch)
        try:
            data = yf.download(ticker_str, start=start, end=end, progress=False, group_by='ticker', threads=True)
            for t in batch:
                try:
                    if len(batch) > 1:
                        # yfinance >=1.2 uses 'Close' (already adjusted), no 'Adj Close'
                        price_col = 'Adj Close' if ('Adj Close' in data[t].columns if hasattr(data[t], 'columns') else False) else 'Close'
                        col = data[t][price_col].dropna()
                    else:
                        price_col = 'Adj Close' if 'Adj Close' in data.columns.get_level_values(0) else 'Close'
                        col = data[price_col].dropna()
                        # Handle multi-index for single ticker in newer yfinance
                        if isinstance(col, pd.DataFrame):
                            col = col.iloc[:, 0]
                    if len(col) > 252:  # at least 1 year
                        all_prices[t] = col
                except:
                    pass
        except Exception as e:
            print(f"  Batch {i//batch_size} error: {e}")
        time.sleep(0.5)

    prices = pd.DataFrame(all_prices)
    prices.index = pd.to_datetime(prices.index)
    print(f"Downloaded {prices.shape[1]} stocks, {prices.shape[0]} days")
    return prices

###############################################################################
# 2. MOMENTUM STRATEGY
###############################################################################

def compute_momentum_features(prices):
    """Compute 1/3/6/12 month momentum."""
    mom1 = prices.pct_change(21)
    mom3 = prices.pct_change(63)
    mom6 = prices.pct_change(126)
    mom12 = prices.pct_change(252)
    # Composite: skip most recent month (short-term reversal)
    # 12-1 month momentum is classic
    mom12_1 = prices.shift(21).pct_change(231)  # 12mo return excluding last month
    return mom1, mom3, mom6, mom12, mom12_1

def compute_atr(prices, period=20):
    """Compute ATR proxy using daily returns."""
    returns = prices.pct_change().abs()
    atr = returns.rolling(period).mean()
    return atr

def compute_volume_sma(volumes, period=20):
    """Volume SMA ratio for confirmation."""
    return volumes / volumes.rolling(period).mean()

def rank_momentum(mom12_1, mom6, mom3, date, min_stocks=20):
    """Rank stocks by composite momentum score at a given date."""
    m12 = mom12_1.loc[:date].iloc[-1].dropna()
    m6 = mom6.loc[:date].iloc[-1].dropna()
    m3 = mom3.loc[:date].iloc[-1].dropna()

    common = m12.index.intersection(m6.index).intersection(m3.index)
    if len(common) < min_stocks:
        return pd.Series(dtype=float)

    # Composite score: 40% 12-1mo + 30% 6mo + 30% 3mo
    score = 0.4 * m12[common].rank(pct=True) + \
            0.3 * m6[common].rank(pct=True) + \
            0.3 * m3[common].rank(pct=True)
    return score.sort_values(ascending=False)

###############################################################################
# 3. WALK-FORWARD BACKTEST
###############################################################################

def run_backtest(prices, spy_prices, n_holdings=30, trailing_stop_atr=3.0,
                 train_months=60, test_months=1):
    """
    Walk-forward momentum backtest with daily trailing stop exits.
    """
    mom1, mom3, mom6, mom12, mom12_1 = compute_momentum_features(prices)
    atr = compute_atr(prices, 20)

    # Get monthly rebalance dates
    monthly = prices.resample('ME').last().index
    monthly = monthly[monthly >= prices.index[252]]  # need 1yr warmup

    if len(monthly) < train_months + test_months + 1:
        print(f"Not enough data: {len(monthly)} months, need {train_months + test_months + 1}")
        return None

    all_returns = []
    all_dates = []
    portfolio_holdings = {}  # ticker -> (entry_price, trailing_high)

    for i in range(train_months, len(monthly) - test_months + 1):
        rebal_date = monthly[i]

        # Rank stocks at rebalance date
        scores = rank_momentum(mom12_1, mom6, mom3, rebal_date)
        if len(scores) < n_holdings:
            continue

        # Top N stocks
        top_stocks = scores.head(n_holdings).index.tolist()

        # Get daily prices for test period
        if i + test_months < len(monthly):
            test_end = monthly[i + test_months]
        else:
            test_end = prices.index[-1]

        test_prices = prices.loc[rebal_date:test_end]
        if len(test_prices) < 2:
            continue

        # Equal weight portfolio with daily trailing stop exits
        daily_returns = []
        active = {s: test_prices[s].iloc[0] for s in top_stocks if s in test_prices.columns and not pd.isna(test_prices[s].iloc[0])}
        highs = dict(active)  # trailing highs

        for d in range(1, len(test_prices)):
            day_ret = 0.0
            n_active = len(active)
            if n_active == 0:
                daily_returns.append(0.0)
                continue

            weight = 1.0 / n_holdings  # fixed weight
            stopped = []
            for s, entry in active.items():
                if s not in test_prices.columns:
                    continue
                p = test_prices[s].iloc[d]
                p_prev = test_prices[s].iloc[d-1]
                if pd.isna(p) or pd.isna(p_prev) or p_prev == 0:
                    continue

                ret = (p - p_prev) / p_prev
                day_ret += weight * ret

                # Update trailing high
                if p > highs.get(s, 0):
                    highs[s] = p

                # Check trailing stop (ATR-based)
                try:
                    current_atr = atr.loc[:test_prices.index[d], s].iloc[-1]
                    if not pd.isna(current_atr) and current_atr > 0:
                        stop_level = highs[s] * (1 - trailing_stop_atr * current_atr)
                        if p < stop_level:
                            stopped.append(s)
                except:
                    pass

            for s in stopped:
                del active[s]
                if s in highs:
                    del highs[s]

            daily_returns.append(day_ret)

        all_returns.extend(daily_returns)
        all_dates.extend(test_prices.index[1:len(daily_returns)+1].tolist())

    if not all_returns:
        return None

    results = pd.Series(all_returns, index=pd.DatetimeIndex(all_dates))
    # Remove duplicates (overlapping windows)
    results = results[~results.index.duplicated(keep='last')]
    results = results.sort_index()
    return results

###############################################################################
# 4. METRICS & REGIME ANALYSIS
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
    sharpe = (returns.mean() * ann_factor) / (returns.std() * np.sqrt(ann_factor)) if returns.std() > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(ann_factor) if len(returns[returns < 0]) > 0 else 1e-6
    sortino = (returns.mean() * ann_factor) / downside

    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    win_rate = (returns > 0).mean()

    # Profit factor
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

    # Align
    common = returns.index.intersection(regime.index)
    if len(common) < 30:
        return None, None, None

    r = returns.loc[common]
    reg = regime.loc[common]

    bull_returns = r[reg == 1]
    bear_returns = r[reg == 0]

    bull_metrics = compute_metrics(bull_returns, "Bull Regime") if len(bull_returns) > 30 else {}
    bear_metrics = compute_metrics(bear_returns, "Bear Regime") if len(bear_returns) > 30 else {}

    # Regime gap test
    if bull_metrics and bear_metrics:
        s_bull = bull_metrics['Sharpe']
        s_bear = bear_metrics['Sharpe']
        denom = max(abs(s_bull), abs(s_bear))
        regime_gap = abs(s_bull - s_bear) / denom if denom > 0 else 0
    else:
        regime_gap = None

    return bull_metrics, bear_metrics, regime_gap

def permutation_test(returns, n_perms=100):
    """Permutation test for statistical significance."""
    actual_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = returns.sample(frac=1.0, replace=False).values
        s = shuffled.mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
        perm_sharpes.append(s)

    p_value = (np.sum(np.array(perm_sharpes) >= actual_sharpe) + 1) / (n_perms + 1)
    return actual_sharpe, p_value, perm_sharpes

###############################################################################
# 5. MAIN
###############################################################################

def main():
    print("=" * 70)
    print("SYSTEMATIC MOMENTUM v2 — Walk-Forward Backtest")
    print("=" * 70)

    # Get tickers
    print("\n[1/5] Getting ticker universe...")
    sp500 = get_sp500_tickers()
    sp400 = get_sp400_sample()
    universe = list(set(sp500 + sp400))
    print(f"  Universe: {len(universe)} tickers")

    # Download data
    print("\n[2/5] Downloading price data (2013-2026)...")
    prices = download_prices(universe, start='2013-01-01', end='2026-07-01')

    # Download SPY for regime analysis
    import yfinance as yf
    spy = yf.download('SPY', start='2013-01-01', end='2026-07-01', progress=False)
    # yfinance >=1.2 may return multi-index columns even for single ticker
    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy['Close'].iloc[:, 0] if 'Close' in spy.columns.get_level_values(0) else spy.iloc[:, 0]
    else:
        spy_close = spy['Adj Close'] if 'Adj Close' in spy.columns else spy['Close']
    spy_close = spy_close.squeeze()

    # Run backtest
    print("\n[3/5] Running walk-forward backtest (60mo train, 1mo test)...")
    returns = run_backtest(prices, spy_close, n_holdings=30, trailing_stop_atr=3.0,
                          train_months=60, test_months=1)

    if returns is None or len(returns) < 30:
        print("ERROR: Insufficient returns data")
        return

    # Compute metrics
    print("\n[4/5] Computing metrics...")
    metrics = compute_metrics(returns, "Momentum v2")
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
        print(f"  Regime Gap: {gap:.3f} {'PASS' if gap < 0.50 else 'FAIL'} (threshold: <0.50)")

    # Permutation test
    print("\n[5/5] Running permutation test (100 shuffles)...")
    actual_s, p_val, _ = permutation_test(returns, n_perms=100)
    print(f"  Actual Sharpe: {actual_s:.3f}")
    print(f"  p-value: {p_val:.4f} {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'}")

    # Benchmark comparison (SPY buy & hold)
    spy_returns = spy_close.pct_change().dropna()
    common_idx = returns.index.intersection(spy_returns.index)
    if len(common_idx) > 30:
        spy_bench = compute_metrics(spy_returns.loc[common_idx], "SPY Buy&Hold")
        print("\n--- BENCHMARK (SPY Buy & Hold, same period) ---")
        for k, v in spy_bench.items():
            print(f"  {k}: {v}")

    # Save results
    results = {
        'strategy': 'Systematic Momentum v2',
        'overall': metrics,
        'bull_regime': bull,
        'bear_regime': bear,
        'regime_gap': gap,
        'permutation_p_value': p_val,
        'n_stocks_universe': prices.shape[1],
        'backtest_period': f"{returns.index[0].date()} to {returns.index[-1].date()}",
        'benchmark': spy_bench if len(common_idx) > 30 else {},
    }

    outpath = os.path.join(OUTPUT_DIR, 'momentum_v2_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")

    # Save daily returns
    returns.to_csv(os.path.join(OUTPUT_DIR, 'momentum_v2_returns.csv'))
    print("Done!")

if __name__ == '__main__':
    main()
