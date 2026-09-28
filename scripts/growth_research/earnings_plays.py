#!/usr/bin/env python3
"""
Earnings-Based Growth Strategy Backtester
==========================================
Two strategies:
  1. Pre-Earnings Momentum — buy stocks with strong momentum 5-10 days before
     earnings, sell day after announcement.
  2. Post-Earnings Drift (PEAD) — buy stocks with positive earnings surprise on
     announcement day, hold 5-20 days.

Universe: S&P 500
Data: yfinance (2018-2026)
Walk-forward: 3-year train, 1-year test, sliding
Dynamic exits: stop-loss (2x ATR), trailing stop, thesis-break exit
Regime test: SPY green/red days, regime gap < 0.50
Permutation test for significance
"""

import os, sys, json, warnings, time, pickle, hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/earnings_plays'
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Cost assumptions for equities
COMMISSION_PER_SHARE = 0.005  # typical IBKR
SLIPPAGE_BPS = 5  # 5 bps slippage per side

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
        return _fallback_tickers()

def _fallback_tickers():
    """Large fallback list of liquid S&P 500 names."""
    return [
        'AAPL','MSFT','AMZN','GOOGL','GOOG','META','NVDA','TSLA','BRK-B','UNH',
        'JNJ','XOM','JPM','V','PG','MA','HD','CVX','MRK','ABBV','LLY','PEP','KO',
        'AVGO','COST','TMO','MCD','WMT','CSCO','ACN','ABT','DHR','NEE','LIN',
        'BMY','PM','TXN','UNP','RTX','AMGN','HON','LOW','QCOM','INTC','COP',
        'IBM','SBUX','CAT','GS','BA','MDLZ','BLK','ADP','DE','ADI','GILD',
        'MMC','ISRG','SYK','VRTX','BKNG','REGN','TJX','ZTS','CI','CB','PLD',
        'SO','DUK','CME','SLB','CL','USB','ITW','BDX','MO','EOG','WM','APD',
        'NOC','ICE','FDX','GD','FCX','PNC','ORLY','AZO','SHW','NSC','EMR',
        'MCK','TGT','PSX','VLO','OXY','AIG','AFL','D','HUM','MET','PRU',
        'MSCI','TRV','ALL','AEP','SPG','WELL','PSA','O','AMT','CCI','EQIX',
        'CRM','ORCL','NFLX','AMD','PYPL','NOW','INTU','AMAT','AXP','T','VZ',
        'MDLZ','SCHW','MS','C','WFC','GE','MMM','F','GM','DIS','CMCSA',
        'PFE','NKE','UPS','COP','SNPS','CDNS','KLAC','LRCX','MCHP','FTNT',
        'PANW','DDOG','CRWD','ZS','SNOW','NET','MDB','WDAY','TEAM','VEEV',
        'ABNB','UBER','LYFT','SQ','COIN','HOOD','RIVN','LCID','PLTR','SOFI',
        'DASH','RBLX','U','TTD','PINS','SNAP','ROKU',
        'EL','CHTR','TMUS','BIIB','ILMN','MRNA','DXCM','ALGN',
        'ENPH','SEDG','FSLR','RUN','PLUG','CHPT',
        'CMG','YUM','SBUX','DPZ','QSR',
        'LMT','GD','NOC','RTX','HII','BA','TDG',
        'WM','RSG','WCN','CLH',
        'SHW','PPG','ECL','APD','LIN',
        'CTAS','PAYX','ADP','PAYC',
        'MSCI','SPGI','MCO','ICE','CME','CBOE','NDAQ',
        'AMT','CCI','SBAC','EQIX',
    ]

def download_prices(tickers, start='2017-06-01', end='2026-07-13'):
    """Download adjusted close + volume with batching and caching."""
    import yfinance as yf

    cache_key = hashlib.md5(f"{sorted(tickers)}{start}{end}".encode()).hexdigest()[:12]
    cache_file = os.path.join(CACHE_DIR, f'prices_{cache_key}.pkl')

    if os.path.exists(cache_file):
        age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_hours < 24:
            print(f"Loading cached prices ({age_hours:.1f}h old)")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

    all_close = {}
    all_volume = {}
    all_high = {}
    all_low = {}
    batch_size = 50
    total = len(tickers)

    for i in range(0, total, batch_size):
        batch = tickers[i:i+batch_size]
        ticker_str = ' '.join(batch)
        try:
            data = yf.download(ticker_str, start=start, end=end, progress=False,
                             group_by='ticker', threads=True)
            for t in batch:
                try:
                    if len(batch) > 1:
                        close = data[t]['Close'].dropna()
                        vol = data[t]['Volume'].dropna()
                        high = data[t]['High'].dropna()
                        low = data[t]['Low'].dropna()
                    else:
                        close = data['Close'].dropna()
                        vol = data['Volume'].dropna()
                        high = data['High'].dropna()
                        low = data['Low'].dropna()
                    if len(close) > 252:
                        all_close[t] = close
                        all_volume[t] = vol
                        all_high[t] = high
                        all_low[t] = low
                except:
                    pass
        except Exception as e:
            print(f"  Batch {i//batch_size} error: {e}")
        time.sleep(0.3)
        print(f"  Downloaded {min(i+batch_size, total)}/{total} tickers")

    prices = pd.DataFrame(all_close)
    volumes = pd.DataFrame(all_volume)
    highs = pd.DataFrame(all_high)
    lows = pd.DataFrame(all_low)
    prices.index = pd.to_datetime(prices.index)
    volumes.index = pd.to_datetime(volumes.index)
    highs.index = pd.to_datetime(highs.index)
    lows.index = pd.to_datetime(lows.index)

    result = {'prices': prices, 'volumes': volumes, 'highs': highs, 'lows': lows}

    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"Downloaded {prices.shape[1]} stocks, {prices.shape[0]} trading days")
    return result


def get_earnings_dates(tickers, prices):
    """
    Get earnings dates from yfinance.
    Falls back to estimating quarterly earnings from price volatility spikes.
    """
    import yfinance as yf

    cache_file = os.path.join(CACHE_DIR, 'earnings_dates.pkl')
    if os.path.exists(cache_file):
        age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_hours < 24:
            print("Loading cached earnings dates")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

    earnings = {}
    errors = 0
    total = len(tickers)

    for idx, ticker in enumerate(tickers):
        if idx % 50 == 0:
            print(f"  Fetching earnings dates: {idx}/{total}")
        try:
            tk = yf.Ticker(ticker)
            # Try earnings_dates first (has actual dates)
            try:
                ed = tk.earnings_dates
                if ed is not None and len(ed) > 0:
                    dates = pd.to_datetime(ed.index).normalize()
                    # Filter to our price range
                    dates = dates[(dates >= prices.index.min()) & (dates <= prices.index.max())]
                    if len(dates) > 0:
                        earnings[ticker] = sorted(dates.tolist())
                        continue
            except:
                pass

            # Fallback: estimate from large overnight gaps (earnings typically cause gaps)
            if ticker in prices.columns:
                p = prices[ticker].dropna()
                daily_ret = p.pct_change()
                # Earnings gaps are typically >3% absolute moves
                big_moves = daily_ret[daily_ret.abs() > 0.03]
                if len(big_moves) > 0:
                    # Filter to roughly quarterly (keep only dates >45 days apart)
                    dates_list = sorted(big_moves.index.tolist())
                    filtered = [dates_list[0]]
                    for d in dates_list[1:]:
                        if (d - filtered[-1]).days > 45:
                            filtered.append(d)
                    earnings[ticker] = filtered

        except Exception as e:
            errors += 1
        time.sleep(0.05)

    print(f"Got earnings dates for {len(earnings)} stocks ({errors} errors)")

    with open(cache_file, 'wb') as f:
        pickle.dump(earnings, f)

    return earnings


###############################################################################
# 2. FEATURE COMPUTATION
###############################################################################

def compute_atr(highs, lows, closes, period=20):
    """True Range approximation using high/low/close."""
    tr1 = highs - lows
    tr2 = (highs - closes.shift(1)).abs()
    tr3 = (lows - closes.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1) if isinstance(tr1, pd.Series) else tr1
    # For DataFrames, compute element-wise max
    if isinstance(tr1, pd.DataFrame):
        tr = tr1.copy()
        for col in tr.columns:
            tr[col] = pd.concat([tr1[col], tr2[col], tr3[col]], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    return tr.rolling(period).mean()


def compute_momentum(prices, lookback=10):
    """Simple momentum: return over lookback days."""
    return prices.pct_change(lookback)


def compute_relative_volume(volumes, lookback=20):
    """Volume relative to its 20-day average."""
    return volumes / volumes.rolling(lookback).mean()


###############################################################################
# 3. STRATEGY 1: PRE-EARNINGS MOMENTUM
###############################################################################

def pre_earnings_momentum(prices, volumes, highs, lows, earnings_dates,
                           spy_prices, train_start, train_end, test_start, test_end,
                           params=None):
    """
    Buy stocks with strong momentum 5-10 days before earnings.
    Sell day after earnings announcement.

    Training phase: learn optimal momentum threshold and entry timing.
    Test phase: apply learned parameters.
    """
    if params is None:
        # Default params — will be optimized in training
        params = {
            'entry_days_before': 7,    # buy N days before earnings
            'momentum_lookback': 20,   # momentum calculation window
            'momentum_threshold': 0.05, # minimum momentum to enter (5%)
            'volume_threshold': 1.2,    # minimum relative volume
            'stop_loss_atr_mult': 2.0,  # stop-loss at 2x ATR
            'trailing_stop_pct': 0.03,  # 3% trailing stop
            'max_positions': 20,        # max concurrent positions
        }

    atr = compute_atr(highs, lows, prices)
    mom = compute_momentum(prices, params['momentum_lookback'])
    rvol = compute_relative_volume(volumes)

    trades = []
    test_dates = prices.index[(prices.index >= test_start) & (prices.index <= test_end)]

    for ticker, edates in earnings_dates.items():
        if ticker not in prices.columns:
            continue

        for edate in edates:
            edate = pd.Timestamp(edate)
            if edate < test_start or edate > test_end:
                continue

            # Find entry date: N trading days before earnings
            entry_days = params['entry_days_before']
            valid_dates = prices.index[prices.index < edate]
            if len(valid_dates) < entry_days + 5:
                continue
            entry_date = valid_dates[-entry_days]

            # Check momentum criteria
            if entry_date not in mom.index or ticker not in mom.columns:
                continue
            m = mom.loc[entry_date, ticker]
            if pd.isna(m) or m < params['momentum_threshold']:
                continue

            # Check volume
            if entry_date in rvol.index and ticker in rvol.columns:
                rv = rvol.loc[entry_date, ticker]
                if pd.isna(rv) or rv < params['volume_threshold']:
                    continue

            # Entry price
            entry_price = prices.loc[entry_date, ticker]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # ATR for stop-loss
            if entry_date in atr.index and ticker in atr.columns:
                current_atr = atr.loc[entry_date, ticker]
            else:
                current_atr = entry_price * 0.02  # fallback 2%

            stop_loss = entry_price - params['stop_loss_atr_mult'] * current_atr

            # Simulate holding until day after earnings with dynamic exits
            hold_dates = prices.index[(prices.index >= entry_date) & (prices.index <= edate)]
            # Add one more day after earnings
            post_earn = prices.index[prices.index > edate]
            if len(post_earn) > 0:
                hold_dates = hold_dates.append(pd.DatetimeIndex([post_earn[0]]))

            if len(hold_dates) < 2:
                continue

            exit_price = None
            exit_date = None
            exit_reason = 'earnings_exit'
            peak_price = entry_price

            for hd in hold_dates[1:]:
                if ticker not in prices.columns or hd not in prices.index:
                    continue
                p = prices.loc[hd, ticker]
                if pd.isna(p):
                    continue

                peak_price = max(peak_price, p)

                # Stop-loss check
                if p <= stop_loss:
                    exit_price = p
                    exit_date = hd
                    exit_reason = 'stop_loss'
                    break

                # Trailing stop check
                if p <= peak_price * (1 - params['trailing_stop_pct']):
                    exit_price = p
                    exit_date = hd
                    exit_reason = 'trailing_stop'
                    break

            if exit_price is None:
                # Normal exit: day after earnings
                last_date = hold_dates[-1]
                exit_price = prices.loc[last_date, ticker]
                exit_date = last_date
                if pd.isna(exit_price):
                    continue

            # Calculate return with costs
            gross_ret = (exit_price / entry_price) - 1
            cost = 2 * SLIPPAGE_BPS / 10000  # round-trip slippage
            net_ret = gross_ret - cost

            holding_days = (exit_date - entry_date).days

            trades.append({
                'ticker': ticker,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'earnings_date': edate,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'gross_return': gross_ret,
                'net_return': net_ret,
                'holding_days': holding_days,
                'exit_reason': exit_reason,
                'momentum': m,
            })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


###############################################################################
# 4. STRATEGY 2: POST-EARNINGS DRIFT (PEAD)
###############################################################################

def post_earnings_drift(prices, volumes, highs, lows, earnings_dates,
                         spy_prices, train_start, train_end, test_start, test_end,
                         params=None):
    """
    Buy stocks with positive earnings surprise (large positive gap on earnings day).
    Hold for N days, capturing the well-documented post-earnings drift anomaly.
    """
    if params is None:
        params = {
            'surprise_threshold': 0.03,  # minimum 3% positive gap = "beat"
            'hold_days': 10,              # hold for N trading days
            'stop_loss_atr_mult': 2.0,
            'trailing_stop_pct': 0.05,    # 5% trailing stop (wider for drift)
            'max_positions': 20,
            'thesis_break_pct': -0.03,    # exit if drops 3% below earnings close
        }

    atr = compute_atr(highs, lows, prices)
    trades = []

    for ticker, edates in earnings_dates.items():
        if ticker not in prices.columns:
            continue

        for edate in edates:
            edate = pd.Timestamp(edate)
            if edate < test_start or edate > test_end:
                continue

            # Check if stock had a positive surprise (gap up on earnings day)
            if edate not in prices.index:
                # Find nearest trading day
                valid = prices.index[prices.index >= edate]
                if len(valid) == 0:
                    continue
                edate = valid[0]

            pre_dates = prices.index[prices.index < edate]
            if len(pre_dates) == 0:
                continue
            pre_date = pre_dates[-1]

            pre_price = prices.loc[pre_date, ticker]
            earn_price = prices.loc[edate, ticker]

            if pd.isna(pre_price) or pd.isna(earn_price) or pre_price <= 0:
                continue

            earnings_gap = (earn_price / pre_price) - 1

            # Only take positive surprises above threshold
            if earnings_gap < params['surprise_threshold']:
                continue

            # Entry on earnings day close
            entry_price = earn_price
            entry_date = edate

            # ATR for stop-loss
            if edate in atr.index and ticker in atr.columns:
                current_atr = atr.loc[edate, ticker]
            else:
                current_atr = entry_price * 0.02

            stop_loss = entry_price - params['stop_loss_atr_mult'] * current_atr
            thesis_break = entry_price * (1 + params['thesis_break_pct'])

            # Hold for N trading days with dynamic exits
            future_dates = prices.index[prices.index > entry_date]
            hold_dates = future_dates[:params['hold_days']]

            if len(hold_dates) == 0:
                continue

            exit_price = None
            exit_date = None
            exit_reason = 'time_exit'
            peak_price = entry_price

            for hd in hold_dates:
                if ticker not in prices.columns or hd not in prices.index:
                    continue
                p = prices.loc[hd, ticker]
                if pd.isna(p):
                    continue

                peak_price = max(peak_price, p)

                # Stop-loss
                if p <= stop_loss:
                    exit_price = p
                    exit_date = hd
                    exit_reason = 'stop_loss'
                    break

                # Trailing stop
                if p <= peak_price * (1 - params['trailing_stop_pct']):
                    exit_price = p
                    exit_date = hd
                    exit_reason = 'trailing_stop'
                    break

                # Thesis break: if price drops below earnings close, drift thesis is dead
                if p <= thesis_break:
                    exit_price = p
                    exit_date = hd
                    exit_reason = 'thesis_break'
                    break

            if exit_price is None:
                exit_date = hold_dates[-1]
                exit_price = prices.loc[exit_date, ticker]
                if pd.isna(exit_price):
                    continue

            gross_ret = (exit_price / entry_price) - 1
            cost = 2 * SLIPPAGE_BPS / 10000
            net_ret = gross_ret - cost

            holding_days = (exit_date - entry_date).days

            trades.append({
                'ticker': ticker,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'earnings_date': edate,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'gross_return': gross_ret,
                'net_return': net_ret,
                'holding_days': holding_days,
                'exit_reason': exit_reason,
                'earnings_gap': earnings_gap,
            })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


###############################################################################
# 5. WALK-FORWARD ENGINE
###############################################################################

def optimize_params_pre_earnings(prices, volumes, highs, lows, earnings_dates,
                                  spy_prices, train_start, train_end):
    """Grid search over pre-earnings momentum parameters in training window."""
    best_sharpe = -999
    best_params = None

    param_grid = [
        {'entry_days_before': edb, 'momentum_lookback': ml, 'momentum_threshold': mt,
         'volume_threshold': 1.0, 'stop_loss_atr_mult': 2.0, 'trailing_stop_pct': 0.03,
         'max_positions': 20}
        for edb in [5, 7, 10]
        for ml in [10, 20]
        for mt in [0.03, 0.05, 0.08]
    ]

    for params in param_grid:
        trades = pre_earnings_momentum(
            prices, volumes, highs, lows, earnings_dates, spy_prices,
            train_start, train_end, train_start, train_end, params
        )
        if len(trades) < 10:
            continue
        sharpe = compute_sharpe(trades['net_return'])
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params.copy()

    if best_params is None:
        best_params = {
            'entry_days_before': 7, 'momentum_lookback': 20, 'momentum_threshold': 0.05,
            'volume_threshold': 1.0, 'stop_loss_atr_mult': 2.0, 'trailing_stop_pct': 0.03,
            'max_positions': 20
        }

    return best_params


def optimize_params_pead(prices, volumes, highs, lows, earnings_dates,
                          spy_prices, train_start, train_end):
    """Grid search over PEAD parameters in training window."""
    best_sharpe = -999
    best_params = None

    param_grid = [
        {'surprise_threshold': st, 'hold_days': hd,
         'stop_loss_atr_mult': 2.0, 'trailing_stop_pct': ts,
         'max_positions': 20, 'thesis_break_pct': -0.03}
        for st in [0.02, 0.03, 0.05]
        for hd in [5, 10, 15, 20]
        for ts in [0.04, 0.05, 0.07]
    ]

    for params in param_grid:
        trades = post_earnings_drift(
            prices, volumes, highs, lows, earnings_dates, spy_prices,
            train_start, train_end, train_start, train_end, params
        )
        if len(trades) < 10:
            continue
        sharpe = compute_sharpe(trades['net_return'])
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params.copy()

    if best_params is None:
        best_params = {
            'surprise_threshold': 0.03, 'hold_days': 10,
            'stop_loss_atr_mult': 2.0, 'trailing_stop_pct': 0.05,
            'max_positions': 20, 'thesis_break_pct': -0.03
        }

    return best_params


def compute_sharpe(returns, annual_factor=None):
    """Annualized Sharpe ratio from trade returns."""
    if len(returns) < 2:
        return 0.0
    mean_r = returns.mean()
    std_r = returns.std()
    if std_r == 0:
        return 0.0
    # Approximate: assume ~60 trades/year for earnings strategies
    if annual_factor is None:
        annual_factor = np.sqrt(min(len(returns), 60))
    return (mean_r / std_r) * annual_factor


def walk_forward_backtest(prices, volumes, highs, lows, earnings_dates, spy_prices,
                           strategy_name='pre_earnings',
                           train_years=3, test_years=1):
    """
    Walk-forward validation with sliding window.
    3-year train, 1-year test.
    """
    all_oot_trades = []
    window_results = []

    # Define walk-forward windows
    min_date = prices.index.min()
    max_date = prices.index.max()

    # Start test windows
    first_test_start = min_date + pd.DateOffset(years=train_years)
    test_starts = pd.date_range(first_test_start, max_date - pd.DateOffset(months=6),
                                 freq='12MS')

    print(f"\n{'='*60}")
    print(f"Walk-Forward: {strategy_name}")
    print(f"{'='*60}")
    print(f"Total windows: {len(test_starts)}")

    for i, test_start in enumerate(test_starts):
        train_start = test_start - pd.DateOffset(years=train_years)
        train_end = test_start - pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.DateOffset(days=1)
        test_end = min(test_end, max_date)

        print(f"\n  Window {i+1}: Train {train_start.date()} -> {train_end.date()}, "
              f"Test {test_start.date()} -> {test_end.date()}")

        if strategy_name == 'pre_earnings':
            # Optimize on training data
            best_params = optimize_params_pre_earnings(
                prices, volumes, highs, lows, earnings_dates, spy_prices,
                train_start, train_end
            )
            # Apply to test data
            trades = pre_earnings_momentum(
                prices, volumes, highs, lows, earnings_dates, spy_prices,
                train_start, train_end, test_start, test_end, best_params
            )
        else:
            best_params = optimize_params_pead(
                prices, volumes, highs, lows, earnings_dates, spy_prices,
                train_start, train_end
            )
            trades = post_earnings_drift(
                prices, volumes, highs, lows, earnings_dates, spy_prices,
                train_start, train_end, test_start, test_end, best_params
            )

        if len(trades) == 0:
            print(f"    No trades in test window")
            continue

        sharpe = compute_sharpe(trades['net_return'])
        wr = (trades['net_return'] > 0).mean()
        avg_ret = trades['net_return'].mean()

        print(f"    Trades: {len(trades)}, WR: {wr:.1%}, "
              f"Avg Return: {avg_ret:.2%}, Sharpe: {sharpe:.2f}")
        print(f"    Params: {best_params}")

        trades['window'] = i
        trades['train_start'] = train_start
        trades['test_start'] = test_start
        all_oot_trades.append(trades)

        window_results.append({
            'window': i,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'n_trades': len(trades),
            'sharpe': sharpe,
            'win_rate': wr,
            'avg_return': avg_ret,
            'best_params': best_params,
        })

    if not all_oot_trades:
        return pd.DataFrame(), pd.DataFrame(window_results)

    all_trades = pd.concat(all_oot_trades, ignore_index=True)
    return all_trades, pd.DataFrame(window_results)


###############################################################################
# 6. REGIME ANALYSIS
###############################################################################

def regime_analysis(trades, spy_prices):
    """
    Split trades by SPY regime (green/red days).
    Check regime gap < 0.50 per HC #428.
    """
    if len(trades) == 0:
        return {'regime_gap': None, 'pass': False, 'detail': 'No trades'}

    spy_daily_ret = spy_prices.pct_change()

    # Classify each trade by SPY regime during holding period
    green_trades = []
    red_trades = []

    for _, trade in trades.iterrows():
        entry = pd.Timestamp(trade['entry_date'])
        exit_d = pd.Timestamp(trade['exit_date'])
        try:
            # Use asof to find nearest available date <= target
            spy_entry_val = float(spy_prices.asof(entry))
            spy_exit_val = float(spy_prices.asof(exit_d))
            if np.isnan(spy_entry_val) or np.isnan(spy_exit_val) or spy_entry_val <= 0:
                continue
            spy_ret = (spy_exit_val / spy_entry_val) - 1
        except Exception:
            continue

        if spy_ret >= 0:
            green_trades.append(trade['net_return'])
        else:
            red_trades.append(trade['net_return'])

    green_trades = pd.Series(green_trades) if green_trades else pd.Series(dtype=float)
    red_trades = pd.Series(red_trades) if red_trades else pd.Series(dtype=float)

    sharpe_green = compute_sharpe(green_trades) if len(green_trades) >= 5 else 0
    sharpe_red = compute_sharpe(red_trades) if len(red_trades) >= 5 else 0

    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 0 else 0

    result = {
        'n_green': len(green_trades),
        'n_red': len(red_trades),
        'sharpe_green': sharpe_green,
        'sharpe_red': sharpe_red,
        'wr_green': (green_trades > 0).mean() if len(green_trades) > 0 else 0,
        'wr_red': (red_trades > 0).mean() if len(red_trades) > 0 else 0,
        'avg_ret_green': green_trades.mean() if len(green_trades) > 0 else 0,
        'avg_ret_red': red_trades.mean() if len(red_trades) > 0 else 0,
        'regime_gap': regime_gap,
        'pass': regime_gap < 0.50,
    }

    return result


###############################################################################
# 7. PERMUTATION TEST
###############################################################################

def permutation_test(trades, n_perms=1000, seed=42):
    """
    Permutation test: shuffle trade returns to assess significance.
    Returns p-value: probability of observing Sharpe >= actual under null.
    """
    if len(trades) < 10:
        return {'p_value': 1.0, 'actual_sharpe': 0, 'null_sharpes_mean': 0}

    returns = trades['net_return'].values
    actual_sharpe = compute_sharpe(pd.Series(returns))

    rng = np.random.RandomState(seed)
    null_sharpes = []

    for _ in range(n_perms):
        # Shuffle sign of returns (null: no directional edge)
        shuffled = returns * rng.choice([-1, 1], size=len(returns))
        null_sharpes.append(compute_sharpe(pd.Series(shuffled)))

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= actual_sharpe).mean()

    return {
        'p_value': p_value,
        'actual_sharpe': actual_sharpe,
        'null_sharpes_mean': null_sharpes.mean(),
        'null_sharpes_std': null_sharpes.std(),
        'perm_95th': np.percentile(null_sharpes, 95),
        'perm_99th': np.percentile(null_sharpes, 99),
    }


###############################################################################
# 8. PERFORMANCE METRICS
###############################################################################

def compute_full_metrics(trades, strategy_name):
    """Compute comprehensive performance metrics."""
    if len(trades) == 0:
        return {}

    returns = trades['net_return']
    n_trades = len(trades)

    # Basic stats
    total_return = (1 + returns).prod() - 1
    avg_return = returns.mean()
    median_return = returns.median()
    std_return = returns.std()
    win_rate = (returns > 0).mean()

    # Sharpe & Sortino
    sharpe = compute_sharpe(returns)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.001
    sortino = (avg_return / downside_std) * np.sqrt(min(n_trades, 60)) if downside_std > 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown (sequential equity curve)
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    # CAGR (approximate from total return and time span)
    if 'entry_date' in trades.columns:
        date_range = (trades['exit_date'].max() - trades['entry_date'].min()).days
        years = max(date_range / 365.25, 0.5)
    else:
        years = n_trades / 60  # rough estimate
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1

    # Holding period
    avg_holding = trades['holding_days'].mean() if 'holding_days' in trades.columns else 0

    # Trades per year
    trades_per_year = n_trades / years

    # Exit reason breakdown
    exit_breakdown = {}
    if 'exit_reason' in trades.columns:
        exit_breakdown = trades['exit_reason'].value_counts().to_dict()

    return {
        'strategy': strategy_name,
        'n_trades': n_trades,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'max_drawdown': max_dd,
        'avg_return': avg_return,
        'median_return': median_return,
        'std_return': std_return,
        'avg_holding_days': avg_holding,
        'trades_per_year': trades_per_year,
        'years': years,
        'exit_breakdown': exit_breakdown,
    }


###############################################################################
# 9. REPORTING
###############################################################################

def print_report(metrics, regime, perm_test, strategy_name):
    """Print formatted performance report."""
    print(f"\n{'='*70}")
    print(f"  {strategy_name.upper()} — OUT-OF-SAMPLE RESULTS")
    print(f"{'='*70}")

    if not metrics:
        print("  No trades generated.")
        return

    print(f"\n  Performance Metrics:")
    print(f"  {'─'*50}")
    print(f"  Total Trades:       {metrics['n_trades']}")
    print(f"  Trades/Year:        {metrics['trades_per_year']:.0f}")
    print(f"  Avg Holding (days): {metrics['avg_holding_days']:.1f}")
    print(f"  {'─'*50}")
    print(f"  CAGR:               {metrics['cagr']:.2%}")
    print(f"  Total Return:       {metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio:       {metrics['sharpe']:.2f}")
    print(f"  Sortino Ratio:      {metrics['sortino']:.2f}")
    print(f"  Profit Factor:      {metrics['profit_factor']:.2f}")
    print(f"  Win Rate:           {metrics['win_rate']:.1%}")
    print(f"  Max Drawdown:       {metrics['max_drawdown']:.2%}")
    print(f"  Avg Return/Trade:   {metrics['avg_return']:.2%}")
    print(f"  Median Return:      {metrics['median_return']:.2%}")

    if metrics['exit_breakdown']:
        print(f"\n  Exit Breakdown:")
        for reason, count in sorted(metrics['exit_breakdown'].items()):
            print(f"    {reason}: {count} ({count/metrics['n_trades']:.0%})")

    print(f"\n  Regime Analysis (HC #428):")
    print(f"  {'─'*50}")
    print(f"  Green (SPY up) trades:  {regime.get('n_green', 0)}")
    print(f"  Red (SPY down) trades:  {regime.get('n_red', 0)}")
    print(f"  Sharpe (green days):    {regime.get('sharpe_green', 0):.2f}")
    print(f"  Sharpe (red days):      {regime.get('sharpe_red', 0):.2f}")
    print(f"  WR (green):             {regime.get('wr_green', 0):.1%}")
    print(f"  WR (red):               {regime.get('wr_red', 0):.1%}")
    print(f"  Regime Gap:             {regime.get('regime_gap', 0):.2f}")
    regime_pass = regime.get('pass', False)
    print(f"  Regime Test:            {'PASS' if regime_pass else 'FAIL'} (threshold < 0.50)")

    print(f"\n  Permutation Test:")
    print(f"  {'─'*50}")
    print(f"  Actual Sharpe:          {perm_test.get('actual_sharpe', 0):.2f}")
    print(f"  Null Sharpe (mean):     {perm_test.get('null_sharpes_mean', 0):.2f}")
    print(f"  Null 95th pctl:         {perm_test.get('perm_95th', 0):.2f}")
    print(f"  Null 99th pctl:         {perm_test.get('perm_99th', 0):.2f}")
    print(f"  p-value:                {perm_test.get('p_value', 1):.4f}")
    sig = perm_test.get('p_value', 1) < 0.05
    print(f"  Significant (p<0.05):   {'YES' if sig else 'NO'}")

    print(f"{'='*70}\n")


def save_results(trades, metrics, regime, perm_test, window_results, strategy_name):
    """Save all results to output directory."""
    prefix = os.path.join(OUTPUT_DIR, strategy_name)

    if len(trades) > 0:
        trades.to_csv(f'{prefix}_trades.csv', index=False)

    if len(window_results) > 0:
        window_results.to_csv(f'{prefix}_windows.csv', index=False)

    summary = {
        'metrics': {k: v for k, v in metrics.items() if k != 'exit_breakdown'},
        'exit_breakdown': metrics.get('exit_breakdown', {}),
        'regime': {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in regime.items()},
        'permutation_test': {k: float(v) if isinstance(v, (np.floating, float)) else v
                              for k, v in perm_test.items()},
        'timestamp': datetime.now().isoformat(),
    }

    with open(f'{prefix}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Results saved to {prefix}_*.csv/json")


###############################################################################
# 10. MAIN
###############################################################################

def main():
    print("=" * 70)
    print("  EARNINGS-BASED GROWTH STRATEGY BACKTESTER")
    print("  Walk-Forward | Dynamic Exits | Regime-Tested")
    print("=" * 70)

    # 1. Get universe
    print("\n[1/6] Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    # Ensure SPY is included for regime analysis
    if 'SPY' not in tickers:
        tickers.append('SPY')
    # Deduplicate
    tickers = list(dict.fromkeys(tickers))
    print(f"  Universe: {len(tickers)} tickers")

    # 2. Download prices
    print("\n[2/6] Downloading price data (2017-2026)...")
    data = download_prices(tickers, start='2017-06-01', end='2026-07-13')
    prices = data['prices']
    volumes = data['volumes']
    highs = data['highs']
    lows = data['lows']

    # SPY for regime analysis — try from main download first, then standalone
    import yfinance as yf
    if 'SPY' in prices.columns:
        spy_prices = prices['SPY'].dropna()
    else:
        try:
            spy_data = yf.download('SPY', start='2017-06-01', end='2026-07-13', progress=False)
            spy_close = spy_data['Close']
            if isinstance(spy_close, pd.DataFrame):
                spy_close = spy_close.iloc[:, 0]
            spy_prices = spy_close.copy()
            spy_prices.index = pd.to_datetime(spy_prices.index)
        except:
            print("  WARNING: Could not download SPY, regime analysis will be empty")
            spy_prices = pd.Series(dtype=float)

    # 3. Get earnings dates
    print("\n[3/6] Getting earnings dates...")
    # Only fetch earnings for tickers we have price data for
    valid_tickers = [t for t in tickers if t in prices.columns]
    earnings_dates = get_earnings_dates(valid_tickers, prices)
    n_total_dates = sum(len(v) for v in earnings_dates.values())
    print(f"  Total earnings events: {n_total_dates} across {len(earnings_dates)} stocks")

    # 4. Strategy 1: Pre-Earnings Momentum
    print("\n[4/6] Running Pre-Earnings Momentum walk-forward...")
    pre_trades, pre_windows = walk_forward_backtest(
        prices, volumes, highs, lows, earnings_dates, spy_prices,
        strategy_name='pre_earnings', train_years=3, test_years=1
    )

    pre_metrics = compute_full_metrics(pre_trades, 'Pre-Earnings Momentum')
    pre_regime = regime_analysis(pre_trades, spy_prices) if len(pre_trades) > 0 else {}
    pre_perm = permutation_test(pre_trades) if len(pre_trades) > 0 else {}

    print_report(pre_metrics, pre_regime, pre_perm, 'Pre-Earnings Momentum')
    save_results(pre_trades, pre_metrics, pre_regime, pre_perm, pre_windows, 'pre_earnings')

    # 5. Strategy 2: Post-Earnings Drift
    print("\n[5/6] Running Post-Earnings Drift walk-forward...")
    pead_trades, pead_windows = walk_forward_backtest(
        prices, volumes, highs, lows, earnings_dates, spy_prices,
        strategy_name='post_earnings_drift', train_years=3, test_years=1
    )

    pead_metrics = compute_full_metrics(pead_trades, 'Post-Earnings Drift')
    pead_regime = regime_analysis(pead_trades, spy_prices) if len(pead_trades) > 0 else {}
    pead_perm = permutation_test(pead_trades) if len(pead_trades) > 0 else {}

    print_report(pead_metrics, pead_regime, pead_perm, 'Post-Earnings Drift')
    save_results(pead_trades, pead_metrics, pead_regime, pead_perm, pead_windows, 'post_earnings_drift')

    # 6. Combined summary
    print("\n[6/6] Combined Summary")
    print("=" * 70)
    print(f"{'Metric':<25} {'Pre-Earnings':>15} {'PEAD':>15}")
    print(f"{'─'*25} {'─'*15} {'─'*15}")
    for key in ['n_trades', 'cagr', 'sharpe', 'sortino', 'profit_factor',
                'win_rate', 'max_drawdown', 'trades_per_year', 'avg_holding_days']:
        v1 = pre_metrics.get(key, 0)
        v2 = pead_metrics.get(key, 0)
        if key in ['cagr', 'win_rate', 'max_drawdown']:
            print(f"  {key:<23} {v1:>14.1%} {v2:>14.1%}")
        elif key in ['sharpe', 'sortino', 'profit_factor']:
            print(f"  {key:<23} {v1:>14.2f} {v2:>14.2f}")
        else:
            print(f"  {key:<23} {v1:>14.1f} {v2:>14.1f}")

    print(f"\n  Regime Test Pass:  Pre-Earnings={'PASS' if pre_regime.get('pass') else 'FAIL'}, "
          f"PEAD={'PASS' if pead_regime.get('pass') else 'FAIL'}")
    print(f"  Significance:      Pre-Earnings p={pre_perm.get('p_value', 1):.4f}, "
          f"PEAD p={pead_perm.get('p_value', 1):.4f}")
    print(f"\n  Output: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
