#!/usr/bin/env python3
"""
Multi-Strategy Backtest: Breakout/Trend-Following + Oversold Mean Reversion
Universe: S&P 500
Data: yfinance 2015-2026
Walk-forward: test each year independently
Position sizing: equal weight, max 10 positions, 10% per position
Dynamic exits per HC #684, regime test, permutation test
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/breakout_meanrev'
os.makedirs(OUTPUT_DIR, exist_ok=True)

###############################################################################
# 1. DATA ACQUISITION
###############################################################################

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia with fallback."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    except:
        return ['AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','BRK-B','UNH','JNJ',
                'XOM','JPM','V','PG','MA','HD','CVX','MRK','ABBV','LLY','PEP','KO',
                'AVGO','COST','TMO','MCD','WMT','CSCO','ACN','ABT','DHR','NEE','LIN',
                'BMY','PM','TXN','UNP','RTX','AMGN','HON','LOW','QCOM','INTC','COP',
                'IBM','SBUX','CAT','GS','BA','MDLZ','BLK','ADP','DE','ADI','GILD',
                'MMC','ISRG','SYK','VRTX','BKNG','REGN','TJX','ZTS','CI','CB','PLD',
                'SO','DUK','CME','SLB','CL','USB','ITW','BDX','MO','EOG','WM','APD',
                'NOC','ICE','FDX','GD','FCX','PNC','ORLY','AZO','SHW','NSC','EMR',
                'MCK','TGT','PSX','VLO','OXY','AIG','AFL','D','HUM','MET','PRU',
                'MSCI','TRV','ALL','AEP','SPG','WELL','PSA','O','AMT','CCI','EQIX',
                'NFLX','AMD','CRM','NOW','INTU','PYPL','AMAT','MU','LRCX','KLAC',
                'SNPS','CDNS','ADBE','PANW','CRWD','MRVL','ON','GEV','FICO','URI',
                'CMG','CTAS','ODFL','FAST','CPRT','PWR','VRSK','IR','DOV','WST',
                'ROP','IDXX','DXCM','IQV','MTD','WAT','KEYS','TDY','ZBRA','FTV',
                'BR','TRGP','OKE','WMB','KMI','ET','FANG','DVN','MPC','HAL']


def download_all_data(tickers, start='2014-01-01', end='2026-07-13'):
    """Download OHLCV data for all tickers."""
    import yfinance as yf

    all_close = {}
    all_high = {}
    all_low = {}
    all_volume = {}

    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        ticker_str = ' '.join(batch)
        try:
            data = yf.download(ticker_str, start=start, end=end,
                               progress=False, group_by='ticker', threads=True)
            for t in batch:
                try:
                    if len(batch) > 1:
                        df = data[t]
                    else:
                        df = data
                    close = df['Close'].dropna() if 'Close' in df.columns else df['Adj Close'].dropna()
                    high = df['High'].dropna()
                    low = df['Low'].dropna()
                    vol = df['Volume'].dropna()
                    if len(close) > 504:  # at least 2 years
                        all_close[t] = close
                        all_high[t] = high
                        all_low[t] = low
                        all_volume[t] = vol
                except:
                    pass
        except Exception as e:
            print(f"  Batch {i//batch_size} error: {e}")
        time.sleep(0.3)

    close_df = pd.DataFrame(all_close)
    high_df = pd.DataFrame(all_high)
    low_df = pd.DataFrame(all_low)
    vol_df = pd.DataFrame(all_volume)

    for df in [close_df, high_df, low_df, vol_df]:
        df.index = pd.to_datetime(df.index)

    print(f"  Downloaded {close_df.shape[1]} stocks, {close_df.shape[0]} trading days")
    return close_df, high_df, low_df, vol_df


###############################################################################
# 2. TECHNICAL INDICATORS
###############################################################################

def compute_rsi(prices, period=5):
    """RSI calculation."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(high, low, close, period=14):
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1) if isinstance(tr1, pd.Series) else \
         np.maximum(np.maximum(tr1, tr2), tr3)
    atr = tr.rolling(period).mean()
    return atr


def compute_atr_df(high_df, low_df, close_df, period=14):
    """ATR for all stocks as a DataFrame."""
    atr_dict = {}
    for col in close_df.columns:
        if col in high_df.columns and col in low_df.columns:
            tr1 = high_df[col] - low_df[col]
            tr2 = (high_df[col] - close_df[col].shift(1)).abs()
            tr3 = (low_df[col] - close_df[col].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_dict[col] = tr.rolling(period).mean()
    return pd.DataFrame(atr_dict)


def compute_sma(prices, period):
    """Simple Moving Average."""
    return prices.rolling(period).mean()


def compute_bollinger_bands(prices, period=20, num_std=2):
    """Bollinger Bands — returns (upper, middle, lower)."""
    middle = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


###############################################################################
# 3. STRATEGY 1: BREAKOUT / TREND FOLLOWING
###############################################################################

def generate_breakout_signals(close_df, high_df, vol_df, lookback=20, vol_mult=1.5):
    """
    Breakout entry: stock makes new 20-day high on >1.5x average volume.
    Returns DataFrame of boolean signals.
    """
    rolling_high = close_df.rolling(lookback).max()
    vol_sma = vol_df.rolling(lookback).mean()
    vol_ratio = vol_df / vol_sma

    # Entry: close >= rolling high AND volume > vol_mult * avg
    signals = (close_df >= rolling_high) & (vol_ratio > vol_mult)

    # Additional filter: must be above 50-day SMA (not buying breakouts in downtrends)
    sma50 = compute_sma(close_df, 50)
    signals = signals & (close_df > sma50)

    return signals


def run_breakout_strategy(close_df, high_df, low_df, vol_df, atr_df,
                          year_start, year_end,
                          lookback=20, vol_mult=1.5, atr_stop_mult=2.0,
                          max_positions=10, max_hold_days=20):
    """
    Run breakout strategy for a given test period.
    Exit: below 10-day low OR trailing stop at 2x ATR OR max hold.
    """
    mask = (close_df.index >= year_start) & (close_df.index <= year_end)
    dates = close_df.index[mask]
    if len(dates) < 10:
        return pd.Series(dtype=float), []

    # Precompute signals for warmup + test period
    warmup_start = year_start - pd.Timedelta(days=100)
    warmup_mask = close_df.index >= warmup_start
    close_w = close_df.loc[warmup_mask]
    high_w = high_df.loc[warmup_mask]
    vol_w = vol_df.loc[warmup_mask]

    signals = generate_breakout_signals(close_w, high_w, vol_w, lookback, vol_mult)
    rolling_low_10 = close_w.rolling(10).min()

    # Track positions
    positions = {}  # ticker -> {entry_price, entry_date, trailing_high, days_held}
    daily_returns = []
    trade_log = []

    for i, date in enumerate(dates):
        if date not in close_df.index:
            continue
        idx = close_df.index.get_loc(date)

        port_ret = 0.0
        n_active = len(positions)

        # --- CHECK EXITS ---
        to_exit = []
        for ticker, pos in positions.items():
            if ticker not in close_df.columns:
                to_exit.append((ticker, 'missing'))
                continue

            current_price = close_df[ticker].iloc[idx]
            prev_price = close_df[ticker].iloc[idx - 1] if idx > 0 else current_price

            if pd.isna(current_price) or pd.isna(prev_price) or prev_price == 0:
                continue

            # Daily return for this position
            ret = (current_price - prev_price) / prev_price
            port_ret += ret / max_positions  # equal weight 10%

            pos['days_held'] += 1

            # Update trailing high
            if current_price > pos['trailing_high']:
                pos['trailing_high'] = current_price

            # Exit conditions (dynamic exits)
            exit_reason = None

            # 1. Below 10-day low
            if date in rolling_low_10.index and ticker in rolling_low_10.columns:
                low10 = rolling_low_10.loc[date, ticker]
                if not pd.isna(low10) and current_price < low10:
                    exit_reason = '10d_low'

            # 2. Trailing stop at 2x ATR
            if ticker in atr_df.columns and date in atr_df.index:
                current_atr = atr_df.loc[date, ticker]
                if not pd.isna(current_atr) and current_atr > 0:
                    stop = pos['trailing_high'] - atr_stop_mult * current_atr
                    if current_price < stop:
                        exit_reason = 'atr_stop'

            # 3. Max hold time stop
            if pos['days_held'] >= max_hold_days:
                exit_reason = 'time_stop'

            if exit_reason:
                trade_ret = (current_price - pos['entry_price']) / pos['entry_price']
                trade_log.append({
                    'ticker': ticker,
                    'entry_date': str(pos['entry_date']),
                    'exit_date': str(date),
                    'days_held': pos['days_held'],
                    'return': round(trade_ret, 4),
                    'exit_reason': exit_reason,
                    'strategy': 'breakout',
                })
                to_exit.append((ticker, exit_reason))

        for ticker, reason in to_exit:
            if ticker in positions:
                del positions[ticker]

        # --- CHECK ENTRIES ---
        if len(positions) < max_positions and date in signals.index:
            day_signals = signals.loc[date].dropna()
            candidates = day_signals[day_signals == True].index.tolist()

            # Filter out already held
            candidates = [t for t in candidates if t not in positions]

            # Rank by volume surge (strongest breakout first)
            if candidates and date in vol_df.index:
                vol_sma20 = vol_df.rolling(20).mean()
                if date in vol_sma20.index:
                    vol_ratios = {}
                    for t in candidates:
                        if t in vol_df.columns and t in vol_sma20.columns:
                            v = vol_df.loc[date, t]
                            vs = vol_sma20.loc[date, t]
                            if not pd.isna(v) and not pd.isna(vs) and vs > 0:
                                vol_ratios[t] = v / vs
                    candidates = sorted(vol_ratios.keys(), key=lambda x: vol_ratios[x], reverse=True)

            slots = max_positions - len(positions)
            for t in candidates[:slots]:
                if t in close_df.columns:
                    price = close_df.loc[date, t]
                    if not pd.isna(price) and price > 0:
                        positions[t] = {
                            'entry_price': price,
                            'entry_date': date,
                            'trailing_high': price,
                            'days_held': 0,
                        }

        daily_returns.append(port_ret)

    ret_series = pd.Series(daily_returns, index=dates[:len(daily_returns)])
    return ret_series, trade_log


###############################################################################
# 4. STRATEGY 2: OVERSOLD BOUNCE / MEAN REVERSION
###############################################################################

def generate_meanrev_signals(close_df, rsi_df, sma200):
    """
    Mean reversion entry: RSI(5) < 25 AND price above 200-day SMA.
    Buy quality dips in uptrends only.
    """
    signals = (rsi_df < 25) & (close_df > sma200)
    return signals


def run_meanrev_strategy(close_df, rsi_df, sma200,
                         year_start, year_end,
                         max_positions=10, max_hold_days=10):
    """
    Run mean reversion strategy for a given test period.
    Exit: RSI(5) > 60 OR 10-day time stop.
    """
    mask = (close_df.index >= year_start) & (close_df.index <= year_end)
    dates = close_df.index[mask]
    if len(dates) < 10:
        return pd.Series(dtype=float), []

    signals = generate_meanrev_signals(close_df, rsi_df, sma200)

    positions = {}  # ticker -> {entry_price, entry_date, days_held}
    daily_returns = []
    trade_log = []

    for i, date in enumerate(dates):
        if date not in close_df.index:
            continue
        idx = close_df.index.get_loc(date)

        port_ret = 0.0

        # --- CHECK EXITS ---
        to_exit = []
        for ticker, pos in positions.items():
            if ticker not in close_df.columns:
                to_exit.append((ticker, 'missing'))
                continue

            current_price = close_df[ticker].iloc[idx]
            prev_price = close_df[ticker].iloc[idx - 1] if idx > 0 else current_price

            if pd.isna(current_price) or pd.isna(prev_price) or prev_price == 0:
                continue

            ret = (current_price - prev_price) / prev_price
            port_ret += ret / max_positions

            pos['days_held'] += 1
            exit_reason = None

            # 1. RSI > 60 (mean reverted)
            if ticker in rsi_df.columns and date in rsi_df.index:
                current_rsi = rsi_df.loc[date, ticker]
                if not pd.isna(current_rsi) and current_rsi > 60:
                    exit_reason = 'rsi_target'

            # 2. Time stop
            if pos['days_held'] >= max_hold_days:
                exit_reason = 'time_stop'

            if exit_reason:
                trade_ret = (current_price - pos['entry_price']) / pos['entry_price']
                trade_log.append({
                    'ticker': ticker,
                    'entry_date': str(pos['entry_date']),
                    'exit_date': str(date),
                    'days_held': pos['days_held'],
                    'return': round(trade_ret, 4),
                    'exit_reason': exit_reason,
                    'strategy': 'meanrev',
                })
                to_exit.append((ticker, exit_reason))

        for ticker, reason in to_exit:
            if ticker in positions:
                del positions[ticker]

        # --- CHECK ENTRIES ---
        if len(positions) < max_positions and date in signals.index:
            day_signals = signals.loc[date].dropna()
            candidates = day_signals[day_signals == True].index.tolist()
            candidates = [t for t in candidates if t not in positions]

            # Rank by how oversold (lowest RSI first = most oversold)
            if candidates and date in rsi_df.index:
                rsi_vals = {}
                for t in candidates:
                    if t in rsi_df.columns:
                        r = rsi_df.loc[date, t]
                        if not pd.isna(r):
                            rsi_vals[t] = r
                candidates = sorted(rsi_vals.keys(), key=lambda x: rsi_vals[x])

            slots = max_positions - len(positions)
            for t in candidates[:slots]:
                if t in close_df.columns:
                    price = close_df.loc[date, t]
                    if not pd.isna(price) and price > 0:
                        positions[t] = {
                            'entry_price': price,
                            'entry_date': date,
                            'days_held': 0,
                        }

        daily_returns.append(port_ret)

    ret_series = pd.Series(daily_returns, index=dates[:len(daily_returns)])
    return ret_series, trade_log


###############################################################################
# 5. COMBINED PORTFOLIO
###############################################################################

def combine_strategies(breakout_returns, meanrev_returns, weight_bo=0.5, weight_mr=0.5):
    """Combine two strategy return streams with given weights."""
    common = breakout_returns.index.intersection(meanrev_returns.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    combined = weight_bo * breakout_returns.loc[common] + weight_mr * meanrev_returns.loc[common]
    return combined


###############################################################################
# 6. METRICS, REGIME, PERMUTATION
###############################################################################

def compute_metrics(returns, name="Strategy"):
    """Compute risk-adjusted metrics."""
    if returns is None or len(returns) < 10:
        return {}

    ann_factor = 252
    total_days = len(returns)
    years = total_days / ann_factor

    cum_ret = (1 + returns).prod() - 1
    cagr = (1 + cum_ret) ** (1 / max(years, 0.1)) - 1 if cum_ret > -1 else -1.0

    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = (returns.mean() * ann_factor) / (returns.std() * np.sqrt(ann_factor)) if returns.std() > 0 else 0

    downside_returns = returns[returns < 0]
    downside = downside_returns.std() * np.sqrt(ann_factor) if len(downside_returns) > 0 else 1e-6
    sortino = (returns.mean() * ann_factor) / downside if downside > 0 else 0

    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    win_rate = (returns[returns != 0] > 0).mean() if (returns != 0).sum() > 0 else 0

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


def compute_trade_stats(trade_log, strategy_name="Strategy"):
    """Compute trade-level statistics."""
    if not trade_log:
        return {}
    df = pd.DataFrame(trade_log)
    wins = df[df['return'] > 0]
    losses = df[df['return'] <= 0]
    return {
        'strategy': strategy_name,
        'total_trades': len(df),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': f"{len(wins)/len(df):.1%}" if len(df) > 0 else "N/A",
        'avg_return': f"{df['return'].mean():.2%}",
        'avg_winner': f"{wins['return'].mean():.2%}" if len(wins) > 0 else "N/A",
        'avg_loser': f"{losses['return'].mean():.2%}" if len(losses) > 0 else "N/A",
        'avg_hold_days': round(df['days_held'].mean(), 1),
        'median_hold_days': round(df['days_held'].median(), 1),
        'best_trade': f"{df['return'].max():.2%}",
        'worst_trade': f"{df['return'].min():.2%}",
        'exit_reasons': df['exit_reason'].value_counts().to_dict(),
    }


def regime_analysis(returns, spy_close):
    """Stratify by bull/bear regime (SPY above/below 200-day SMA)."""
    spy_sma200 = spy_close.rolling(200).mean()
    regime = (spy_close > spy_sma200).astype(int)
    regime.index = pd.to_datetime(regime.index)

    common = returns.index.intersection(regime.index)
    if len(common) < 30:
        return None, None, None

    r = returns.loc[common]
    reg = regime.loc[common]

    bull_returns = r[reg == 1]
    bear_returns = r[reg == 0]

    bull_metrics = compute_metrics(bull_returns, "Bull Regime") if len(bull_returns) > 20 else {}
    bear_metrics = compute_metrics(bear_returns, "Bear Regime") if len(bear_returns) > 20 else {}

    regime_gap = None
    if bull_metrics and bear_metrics:
        s_bull = bull_metrics['Sharpe']
        s_bear = bear_metrics['Sharpe']
        denom = max(abs(s_bull), abs(s_bear))
        regime_gap = abs(s_bull - s_bear) / denom if denom > 0 else 0

    return bull_metrics, bear_metrics, regime_gap


def permutation_test(returns, n_perms=1000, block_size=5):
    """
    Block-bootstrap permutation test for Sharpe significance.
    Shuffles blocks of consecutive returns to preserve local autocorrelation
    while breaking the signal's temporal structure.
    Randomly flips the sign of each block to create the null distribution
    (the strategy has no directional edge).
    """
    if returns.std() == 0:
        return 0, 1.0
    actual_sharpe = returns.mean() / returns.std() * np.sqrt(252)

    vals = returns.values.copy()
    n = len(vals)
    n_blocks = n // block_size

    perm_sharpes = np.zeros(n_perms)
    for i in range(n_perms):
        # Create permuted series by randomly sign-flipping blocks
        perm = vals.copy()
        for b in range(n_blocks):
            if np.random.random() < 0.5:
                start = b * block_size
                end = min(start + block_size, n)
                perm[start:end] = -perm[start:end]
        s = perm.std()
        perm_sharpes[i] = perm.mean() / s * np.sqrt(252) if s > 0 else 0

    p_value = (np.sum(perm_sharpes >= actual_sharpe) + 1) / (n_perms + 1)
    return actual_sharpe, p_value


def per_year_metrics(returns, label="Strategy"):
    """Break down metrics by calendar year."""
    yearly = {}
    for year in sorted(returns.index.year.unique()):
        yr_ret = returns[returns.index.year == year]
        if len(yr_ret) > 10:
            yearly[str(year)] = compute_metrics(yr_ret, f"{label} {year}")
    return yearly


###############################################################################
# 7. MAIN
###############################################################################

def main():
    print("=" * 72)
    print("MULTI-STRATEGY BACKTEST: Breakout + Mean Reversion")
    print("=" * 72)

    # ---- DATA ----
    print("\n[1/6] Getting S&P 500 universe...")
    tickers = get_sp500_tickers()
    print(f"  Universe: {len(tickers)} tickers")

    print("\n[2/6] Downloading OHLCV data (2014-2026)...")
    close_df, high_df, low_df, vol_df = download_all_data(
        tickers, start='2014-01-01', end='2026-07-13'
    )

    # SPY for benchmark + regime
    import yfinance as yf
    spy = yf.download('SPY', start='2014-01-01', end='2026-07-13', progress=False)
    spy_close = spy['Close'].squeeze() if 'Close' in spy.columns else spy['Adj Close'].squeeze()

    # ---- PRECOMPUTE INDICATORS ----
    print("\n[3/6] Computing indicators...")
    rsi5 = compute_rsi(close_df, period=5)
    sma200 = compute_sma(close_df, 200)
    atr_df = compute_atr_df(high_df, low_df, close_df, period=14)
    print("  RSI(5), SMA(200), ATR(14) computed")

    # ---- RUN STRATEGIES YEAR BY YEAR ----
    test_years = list(range(2015, 2027))
    print(f"\n[4/6] Running walk-forward backtest ({test_years[0]}-{test_years[-1]})...")

    all_bo_returns = []
    all_mr_returns = []
    all_bo_trades = []
    all_mr_trades = []
    yearly_bo = {}
    yearly_mr = {}

    for year in test_years:
        year_start = pd.Timestamp(f'{year}-01-01')
        year_end = pd.Timestamp(f'{year}-12-31')

        # Breakout
        bo_ret, bo_trades = run_breakout_strategy(
            close_df, high_df, low_df, vol_df, atr_df,
            year_start, year_end,
            lookback=20, vol_mult=1.5, atr_stop_mult=2.0,
            max_positions=10, max_hold_days=20,
        )
        if len(bo_ret) > 0:
            all_bo_returns.append(bo_ret)
            all_bo_trades.extend(bo_trades)
            m = compute_metrics(bo_ret, f"Breakout {year}")
            yearly_bo[str(year)] = m
            bo_info = f"Sharpe={m.get('Sharpe','?')}, CAGR={m.get('CAGR','?')}, trades={len(bo_trades)}"
        else:
            bo_info = "no data"

        # Mean Reversion
        mr_ret, mr_trades = run_meanrev_strategy(
            close_df, rsi5, sma200,
            year_start, year_end,
            max_positions=10, max_hold_days=10,
        )
        if len(mr_ret) > 0:
            all_mr_returns.append(mr_ret)
            all_mr_trades.extend(mr_trades)
            m = compute_metrics(mr_ret, f"MeanRev {year}")
            yearly_mr[str(year)] = m
            mr_info = f"Sharpe={m.get('Sharpe','?')}, CAGR={m.get('CAGR','?')}, trades={len(mr_trades)}"
        else:
            mr_info = "no data"

        print(f"  {year}: Breakout [{bo_info}]  |  MeanRev [{mr_info}]")

    # Concat all returns
    bo_full = pd.concat(all_bo_returns).sort_index() if all_bo_returns else pd.Series(dtype=float)
    mr_full = pd.concat(all_mr_returns).sort_index() if all_mr_returns else pd.Series(dtype=float)
    bo_full = bo_full[~bo_full.index.duplicated(keep='last')]
    mr_full = mr_full[~mr_full.index.duplicated(keep='last')]

    # Combined 50/50
    combined = combine_strategies(bo_full, mr_full, 0.5, 0.5)

    # ---- METRICS ----
    print("\n[5/6] Computing overall metrics...")
    bo_metrics = compute_metrics(bo_full, "Breakout")
    mr_metrics = compute_metrics(mr_full, "Mean Reversion")
    comb_metrics = compute_metrics(combined, "Combined 50/50")

    bo_trade_stats = compute_trade_stats(all_bo_trades, "Breakout")
    mr_trade_stats = compute_trade_stats(all_mr_trades, "Mean Reversion")

    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    for label, metrics in [("BREAKOUT", bo_metrics), ("MEAN REVERSION", mr_metrics), ("COMBINED 50/50", comb_metrics)]:
        print(f"\n--- {label} ---")
        for k, v in metrics.items():
            print(f"  {k:15s}: {v}")

    # Trade stats
    for label, ts in [("BREAKOUT TRADES", bo_trade_stats), ("MEAN REV TRADES", mr_trade_stats)]:
        print(f"\n--- {label} ---")
        for k, v in ts.items():
            print(f"  {k:20s}: {v}")

    # Per-year table
    print("\n--- PER-YEAR SHARPE ---")
    print(f"  {'Year':>6s}  {'Breakout':>10s}  {'MeanRev':>10s}  {'Combined':>10s}")
    yearly_comb = per_year_metrics(combined, "Combined") if len(combined) > 0 else {}
    for year in test_years:
        y = str(year)
        bo_s = yearly_bo.get(y, {}).get('Sharpe', '-')
        mr_s = yearly_mr.get(y, {}).get('Sharpe', '-')
        co_s = yearly_comb.get(y, {}).get('Sharpe', '-')
        print(f"  {y:>6s}  {str(bo_s):>10s}  {str(mr_s):>10s}  {str(co_s):>10s}")

    # ---- REGIME ANALYSIS ----
    print("\n--- REGIME ANALYSIS (Combined) ---")
    bull, bear, gap = regime_analysis(combined, spy_close)
    if bull:
        print(f"  Bull: Sharpe={bull['Sharpe']}, CAGR={bull['CAGR']}, WR={bull['Win_Rate']}")
    if bear:
        print(f"  Bear: Sharpe={bear['Sharpe']}, CAGR={bear['CAGR']}, WR={bear['Win_Rate']}")
    if gap is not None:
        verdict = 'PASS' if gap < 0.50 else 'FAIL'
        print(f"  Regime Gap: {gap:.3f} — {verdict} (threshold <0.50)")

    # Regime for individual strategies
    for label, rets in [("Breakout", bo_full), ("Mean Reversion", mr_full)]:
        b, br, g = regime_analysis(rets, spy_close)
        if b and br and g is not None:
            v = 'PASS' if g < 0.50 else 'FAIL'
            print(f"  {label}: Bull Sharpe={b['Sharpe']}, Bear Sharpe={br['Sharpe']}, Gap={g:.3f} {v}")

    # ---- PERMUTATION TEST ----
    print("\n[6/6] Permutation tests (1000 shuffles)...")
    for label, rets in [("Breakout", bo_full), ("Mean Reversion", mr_full), ("Combined", combined)]:
        if len(rets) > 30:
            actual_s, p_val = permutation_test(rets, n_perms=1000)
            sig = 'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'
            print(f"  {label:15s}: Sharpe={actual_s:.3f}, p={p_val:.4f} — {sig}")

    # ---- BENCHMARK ----
    print("\n--- BENCHMARK: SPY Buy & Hold (same period) ---")
    spy_ret = spy_close.pct_change().dropna()
    common_idx = combined.index.intersection(spy_ret.index)
    if len(common_idx) > 30:
        spy_bench = compute_metrics(spy_ret.loc[common_idx], "SPY B&H")
        for k, v in spy_bench.items():
            print(f"  {k:15s}: {v}")

    # ---- SAVE RESULTS ----
    results = {
        'strategy': 'Breakout + Mean Reversion Multi-Strategy',
        'test_period': f"{bo_full.index[0].date()} to {bo_full.index[-1].date()}" if len(bo_full) > 0 else "N/A",
        'universe': f"S&P 500 ({close_df.shape[1]} stocks downloaded)",
        'breakout': {
            'overall': bo_metrics,
            'trade_stats': bo_trade_stats,
            'per_year': yearly_bo,
        },
        'mean_reversion': {
            'overall': mr_metrics,
            'trade_stats': mr_trade_stats,
            'per_year': yearly_mr,
        },
        'combined_50_50': {
            'overall': comb_metrics,
            'per_year': yearly_comb,
        },
        'regime_analysis': {
            'bull': bull,
            'bear': bear,
            'regime_gap': gap,
            'regime_gap_verdict': 'PASS' if gap is not None and gap < 0.50 else ('FAIL' if gap is not None else 'N/A'),
        },
        'benchmark_spy': spy_bench if len(common_idx) > 30 else {},
    }

    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save daily returns
    if len(bo_full) > 0:
        bo_full.to_csv(os.path.join(OUTPUT_DIR, 'breakout_returns.csv'))
    if len(mr_full) > 0:
        mr_full.to_csv(os.path.join(OUTPUT_DIR, 'meanrev_returns.csv'))
    if len(combined) > 0:
        combined.to_csv(os.path.join(OUTPUT_DIR, 'combined_returns.csv'))

    # Save trade logs
    if all_bo_trades:
        pd.DataFrame(all_bo_trades).to_csv(os.path.join(OUTPUT_DIR, 'breakout_trades.csv'), index=False)
    if all_mr_trades:
        pd.DataFrame(all_mr_trades).to_csv(os.path.join(OUTPUT_DIR, 'meanrev_trades.csv'), index=False)

    print("\nDone!")


if __name__ == '__main__':
    main()
